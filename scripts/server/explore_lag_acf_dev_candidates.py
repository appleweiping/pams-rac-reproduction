#!/usr/bin/env python
"""Two-phase, development-only lag-ACF candidate diagnostic.

The ``predict`` phase is label-free.  It records autocorrelation candidates
from normalized pose, pre-position-encoding projections, and the
post-Transformer embeddings optimized by PAMS TCC.  The ``score`` phase is a
separate development-label diagnostic that ranks a bounded deterministic
selection grid and reports explicit candidate-oracle ceilings.

This script is exploratory only.  It cannot authorize sealed-test access or a
paper-table claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _write_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _load_json_unique(path: Path) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )


def _assert_sha256(path: Path, expected: str, role: str) -> None:
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"{role} expected SHA-256 is malformed")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"{role} SHA-256 mismatch: expected={expected} actual={actual}")


def _representation_diagnostics(
    values: Any,
    valid_mask: Any,
    *,
    detrend: bool,
    minimum: int,
    maximum: int,
) -> list[dict[str, object]]:
    from scipy.signal import find_peaks

    from pams.period import (
        linear_detrend_projected_position,
        vector_autocorrelation_fft,
    )

    source = (
        linear_detrend_projected_position(values, valid_mask=valid_mask)
        if detrend
        else values
    )
    autocorrelation = vector_autocorrelation_fft(source, valid_mask)
    if autocorrelation.ndim == 1:
        autocorrelation = autocorrelation.unsqueeze(0)
    rows: list[dict[str, object]] = []
    for sample_ac, sample_mask in zip(autocorrelation, valid_mask, strict=True):
        valid_frames = int(sample_mask.sum())
        upper = min(maximum, valid_frames // 2)
        acf_values = [
            float(value)
            for value in sample_ac[: maximum + 1].detach().cpu().tolist()
        ]
        peaks, properties = find_peaks(
            sample_ac.detach().cpu().numpy(),
            distance=minimum,
            prominence=0.0,
        )
        candidates = [
            (int(lag), float(sample_ac[int(lag)]), float(prominence))
            for lag, prominence in zip(
                peaks,
                properties["prominences"],
                strict=True,
            )
            if minimum <= int(lag) <= upper and float(sample_ac[int(lag)]) > 0.0
        ]
        rows.append(
            {
                "valid_frames": valid_frames,
                "maximum_searched_lag": upper,
                "acf_lag_zero_to_128": acf_values,
                "positive_peak_lags": [item[0] for item in candidates],
                "positive_peak_heights": [item[1] for item in candidates],
                "positive_peak_prominences": [item[2] for item in candidates],
            }
        )
    return rows


def _predict(arguments: argparse.Namespace) -> None:
    import torch

    from pams.config import load_config
    from pams.data import load_pose_cache_set, load_pose_input_manifest
    from pams.reproducibility import hardware_fingerprint
    from pams.training import (
        CheckpointProvenance,
        collate_pose_sequences,
        load_model_checkpoint,
    )

    _assert_sha256(arguments.experiment_config, arguments.experiment_sha256, "experiment")
    _assert_sha256(arguments.checkpoint, arguments.checkpoint_sha256, "checkpoint")
    _assert_sha256(arguments.dev_inputs, arguments.dev_inputs_sha256, "dev inputs")
    if len(arguments.source_revision) != 40:
        raise ValueError("source revision must be a full Git SHA")
    config = load_config(arguments.experiment_config)
    if config.protocol != "ucfrep_526" or config.seed != 2026:
        raise ValueError("diagnostic requires the frozen UCFRep seed-2026 config")
    manifest = load_pose_input_manifest(arguments.dev_inputs)
    if manifest.protocol != "ucfrep_526" or manifest.split != "dev":
        raise ValueError("diagnostic accepts only the UCFRep development sidecar")
    if len(manifest.records) != 84:
        raise ValueError("diagnostic requires exactly 84 development records")
    manifest.validate_exact_membership()
    sequences, snapshot = load_pose_cache_set(
        manifest.records,
        cache_dir=arguments.pose_cache,
        pose_fingerprint=config.pose_fingerprint,
    )
    if snapshot.fingerprint != arguments.pose_set_sha256:
        raise ValueError("development pose-cache-set fingerprint mismatch")

    raw_checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    provenance = CheckpointProvenance.from_mapping(raw_checkpoint["provenance"])
    if not torch.cuda.is_available():
        raise RuntimeError("target-free candidate extraction requires CUDA")
    device = torch.device("cuda:0")
    model = load_model_checkpoint(
        arguments.checkpoint,
        config,
        device=device,
        expected_stage="encoder",
        expected_provenance=provenance,
    ).eval()

    records: list[dict[str, object]] = []
    with torch.inference_mode():
        for start in range(0, len(sequences), 32):
            batch = collate_pose_sequences(sequences[start : start + 32]).to(device)
            embeddings, projected = model.encoder.forward_with_pre_pe(
                batch.poses,
                batch.valid_mask,
            )
            pose = batch.poses.flatten(start_dim=2)
            representations = {
                "pose-centered": (pose, False),
                "projected-centered": (projected, False),
                "projected-linear-detrended": (projected, True),
                "embedding-centered": (embeddings, False),
                "embedding-linear-detrended": (embeddings, True),
            }
            diagnostics = {
                name: _representation_diagnostics(
                    values,
                    batch.valid_mask,
                    detrend=detrend,
                    minimum=config.period.minimum,
                    maximum=config.period.maximum,
                )
                for name, (values, detrend) in representations.items()
            }
            for index, video_id in enumerate(batch.video_ids):
                sampled_frames = int(batch.lengths[index])
                valid_frames = int(batch.valid_mask[index].sum())
                if sampled_frames != config.data.frames:
                    raise RuntimeError("sampled timeline length drifted")
                if valid_frames not in {0, sampled_frames}:
                    raise RuntimeError("partial valid-mask coverage is outside this diagnostic")
                row_diagnostics = {
                    name: values[index] for name, values in diagnostics.items()
                }
                if any(
                    int(value["valid_frames"]) != valid_frames
                    for value in row_diagnostics.values()
                ):
                    raise RuntimeError("representation valid-frame counts disagree")
                records.append(
                    {
                        "video_id": video_id,
                        "sampled_frames": sampled_frames,
                        "valid_frames": valid_frames,
                        "representations": row_diagnostics,
                    }
                )
    if len(records) != 84 or len({row["video_id"] for row in records}) != 84:
        raise RuntimeError("candidate extraction did not produce 84 unique records")
    payload = {
        "schema_version": 1,
        "artifact_type": "pams_lag_acf_dev_candidate_diagnostics",
        "classification": "development-only inferred candidate diagnostic",
        "eligible_for_paper_table": False,
        "protocol": "ucfrep_526",
        "split": "dev",
        "seed": 2026,
        "source_revision": arguments.source_revision,
        "experiment_config_sha256": arguments.experiment_sha256,
        "encoder_checkpoint_sha256": arguments.checkpoint_sha256,
        "dev_inputs_sha256": arguments.dev_inputs_sha256,
        "dev_pose_cache_set_sha256": snapshot.fingerprint,
        "record_total": len(records),
        "representations": [
            "pose-centered",
            "projected-centered",
            "projected-linear-detrended",
            "embedding-centered",
            "embedding-linear-detrended",
        ],
        "period_range_frames": [config.period.minimum, config.period.maximum],
        "hardware": hardware_fingerprint(),
        "mount_audit": {
            "network": "none",
            "dev_targets_mounted": False,
            "sealed_test_assets_mounted": False,
        },
        "records": records,
    }
    _write_exclusive(arguments.output, payload)


def _metrics(raw_predictions: list[float], targets: list[int]) -> dict[str, float]:
    rounded = [int(math.floor(max(value, 0.0) + 0.5)) for value in raw_predictions]
    errors = [abs(prediction - target) for prediction, target in zip(rounded, targets, strict=True)]
    raw_errors = [
        abs(prediction - target)
        for prediction, target in zip(raw_predictions, targets, strict=True)
    ]
    count = len(targets)
    return {
        "nmae_rounded": sum(
            error / target for error, target in zip(errors, targets, strict=True)
        )
        / count,
        "obo_rounded": sum(error <= 1 for error in errors) / count,
        "exact_rounded": sum(error == 0 for error in errors) / count,
        "mae_raw_prediction": sum(raw_errors) / count,
        "mae_rounded": sum(errors) / count,
        "rmse_raw_prediction": math.sqrt(
            sum(
                (prediction - target) ** 2
                for prediction, target in zip(raw_predictions, targets, strict=True)
            )
            / count
        ),
        "rmse_rounded": math.sqrt(sum(error**2 for error in errors) / count),
    }


def _candidate_tuples(diagnostic: dict[str, object]) -> list[tuple[int, float, float]]:
    return [
        (int(lag), float(height), float(prominence))
        for lag, height, prominence in zip(
            diagnostic["positive_peak_lags"],
            diagnostic["positive_peak_heights"],
            diagnostic["positive_peak_prominences"],
            strict=True,
        )
    ]


def _select_period(
    diagnostic: dict[str, object],
    *,
    family: str,
    parameter: float,
) -> float | None:
    candidates = _candidate_tuples(diagnostic)
    if not candidates:
        return None
    if family in {"near-small", "near-large"}:
        maximum = max(item[1] for item in candidates)
        eligible = [item for item in candidates if item[1] >= parameter * maximum]
        chosen = eligible[0] if family == "near-small" else eligible[-1]
        return float(chosen[0])
    if family in {"height-power", "prominence-power"}:
        index = 1 if family == "height-power" else 2
        chosen = max(
            candidates,
            key=lambda item: (
                max(item[index], 0.0) * (item[0] / 128.0) ** parameter,
                -item[0],
            ),
        )
        return float(chosen[0])
    if family == "harmonic-ladder":
        acf = [float(value) for value in diagnostic["acf_lag_zero_to_128"]]

        def score(item: tuple[int, float, float]) -> tuple[float, int]:
            lag, height, _ = item
            multiples = [
                max(acf[multiple], 0.0)
                for multiple in range(2 * lag, len(acf), lag)
            ][:3]
            consistency = sum(multiples) / len(multiples) if multiples else 0.0
            value = (height + parameter * consistency) * math.sqrt(lag / 128.0)
            return value, -lag

        return float(max(candidates, key=score)[0])
    raise ValueError(f"unknown selection family: {family}")


def _strategy_grid(representations: list[str]) -> list[tuple[str, str, float, float]]:
    strategies: list[tuple[str, str, float, float]] = []
    for representation in representations:
        for threshold in (0.50, 0.70, 0.80, 0.90, 0.95, 0.99, 1.00):
            for family in ("near-small", "near-large"):
                for period_scale in (0.5, 1.0, 2.0):
                    strategies.append((representation, family, threshold, period_scale))
        for power in (-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
            for family in ("height-power", "prominence-power"):
                strategies.append((representation, family, power, 1.0))
        for weight in (0.25, 0.50, 1.0, 2.0):
            strategies.append((representation, "harmonic-ladder", weight, 1.0))
    return strategies


def _strategy_predictions(
    records: list[dict[str, object]],
    strategy: tuple[str, str, float, float],
) -> list[float]:
    representation, family, parameter, period_scale = strategy
    predictions: list[float] = []
    for row in records:
        if int(row["valid_frames"]) == 0:
            predictions.append(0.0)
            continue
        diagnostic = row["representations"][representation]
        period = _select_period(diagnostic, family=family, parameter=parameter)
        if period is None:
            predictions.append(0.0)
            continue
        period = min(max(period * period_scale, 4.0), 128.0)
        predictions.append((int(row["sampled_frames"]) - 1) / period)
    return predictions


def _strategy_name(strategy: tuple[str, str, float, float]) -> str:
    representation, family, parameter, scale = strategy
    return f"{representation}.{family}.p{parameter:g}.scale{scale:g}"


def _score(arguments: argparse.Namespace) -> None:
    _assert_sha256(arguments.diagnostics, arguments.diagnostics_sha256, "diagnostics")
    _assert_sha256(arguments.dev_targets, arguments.dev_targets_sha256, "dev targets")
    diagnostic_payload = _load_json_unique(arguments.diagnostics)
    target_payload = _load_json_unique(arguments.dev_targets)
    if not isinstance(diagnostic_payload, dict) or diagnostic_payload.get(
        "artifact_type"
    ) != "pams_lag_acf_dev_candidate_diagnostics":
        raise ValueError("candidate diagnostic artifact type mismatch")
    if diagnostic_payload.get("record_total") != 84:
        raise ValueError("candidate diagnostic record count mismatch")
    mount = diagnostic_payload.get("mount_audit")
    if not isinstance(mount, dict) or mount.get("dev_targets_mounted") is not False:
        raise ValueError("target-free diagnostic isolation is not proven")
    if mount.get("sealed_test_assets_mounted") is not False:
        raise ValueError("candidate diagnostic crossed the sealed-test boundary")
    if not isinstance(target_payload, dict) or target_payload.get("manifest_type") != "dev_targets":
        raise ValueError("development target artifact type mismatch")
    records = diagnostic_payload["records"]
    target_rows = target_payload["records"]
    if [row["video_id"] for row in records] != [
        row["video_id"] for row in target_rows
    ]:
        raise ValueError("candidate diagnostics and targets are not order aligned")
    targets = [int(row["count"]) for row in target_rows]
    if len(targets) != 84 or any(value <= 0 for value in targets):
        raise ValueError("development targets must contain 84 positive counts")
    representations = [str(value) for value in diagnostic_payload["representations"]]
    results: list[dict[str, object]] = []
    prediction_by_name: dict[str, list[float]] = {}
    for strategy in _strategy_grid(representations):
        name = _strategy_name(strategy)
        predictions = _strategy_predictions(records, strategy)
        prediction_by_name[name] = predictions
        results.append({"strategy": name, "metrics": _metrics(predictions, targets)})
    ranked_nmae = sorted(
        results,
        key=lambda item: (
            item["metrics"]["nmae_rounded"],
            -item["metrics"]["obo_rounded"],
            item["strategy"],
        ),
    )
    ranked_obo = sorted(
        results,
        key=lambda item: (
            -item["metrics"]["obo_rounded"],
            item["metrics"]["nmae_rounded"],
            item["strategy"],
        ),
    )

    oracle_rows: list[dict[str, object]] = []
    for representation in representations + ["union"]:
        predictions: list[float] = []
        candidate_hits = 0
        for row, target in zip(records, targets, strict=True):
            if int(row["valid_frames"]) == 0:
                predictions.append(0.0)
                continue
            names = representations if representation == "union" else [representation]
            candidates = sorted(
                {
                    lag
                    for name in names
                    for lag, _, _ in _candidate_tuples(row["representations"][name])
                }
            )
            if not candidates:
                predictions.append(0.0)
                continue
            oracle_period = (int(row["sampled_frames"]) - 1) / target
            selected = min(candidates, key=lambda lag: (abs(lag - oracle_period), lag))
            candidate_hits += int(abs(selected - oracle_period) / oracle_period <= 0.20)
            predictions.append((int(row["sampled_frames"]) - 1) / selected)
        oracle_rows.append(
            {
                "representation": representation,
                "classification": "ground-truth candidate oracle; diagnostic only",
                "candidate_within_20_percent_total": candidate_hits,
                "metrics": _metrics(predictions, targets),
            }
        )
    selected_names = {
        ranked_nmae[0]["strategy"],
        ranked_obo[0]["strategy"],
    }
    selected_predictions = {
        name: [
            {
                "video_id": row["video_id"],
                "raw_count": prediction,
                "rounded_count": int(math.floor(max(prediction, 0.0) + 0.5)),
            }
            for row, prediction in zip(records, prediction_by_name[name], strict=True)
        ]
        for name in sorted(selected_names)
    }
    payload = {
        "schema_version": 1,
        "artifact_type": "pams_lag_acf_dev_candidate_sweep",
        "classification": "development-label-selected exploratory diagnostic",
        "eligible_for_paper_table": False,
        "protocol": "ucfrep_526",
        "split": "dev",
        "record_total": 84,
        "diagnostics_sha256": arguments.diagnostics_sha256,
        "dev_targets_sha256": arguments.dev_targets_sha256,
        "strategy_total": len(results),
        "best_nmae": ranked_nmae[0],
        "best_obo": ranked_obo[0],
        "top_10_nmae": ranked_nmae[:10],
        "top_10_obo": ranked_obo[:10],
        "candidate_oracles": oracle_rows,
        "selected_predictions": selected_predictions,
        "claim_boundary": {
            "development_labels_selected_strategy": True,
            "may_authorize_sealed_test": False,
            "verifies_pams": False,
        },
        "mount_audit": {
            "network": "none",
            "diagnostics_mounted": True,
            "dev_targets_mounted": True,
            "pose_mounted": False,
            "checkpoint_mounted": False,
            "full_source_tree_mounted": False,
            "sealed_test_assets_mounted": False,
        },
    }
    _write_exclusive(arguments.output, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--experiment-config", type=Path, required=True)
    predict.add_argument("--experiment-sha256", required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--checkpoint-sha256", required=True)
    predict.add_argument("--dev-inputs", type=Path, required=True)
    predict.add_argument("--dev-inputs-sha256", required=True)
    predict.add_argument("--pose-cache", type=Path, required=True)
    predict.add_argument("--pose-set-sha256", required=True)
    predict.add_argument("--source-revision", required=True)
    predict.add_argument("--output", type=Path, required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--diagnostics", type=Path, required=True)
    score.add_argument("--diagnostics-sha256", required=True)
    score.add_argument("--dev-targets", type=Path, required=True)
    score.add_argument("--dev-targets-sha256", required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "predict":
        _predict(arguments)
    elif arguments.command == "score":
        _score(arguments)
    else:
        raise AssertionError("unreachable command")


if __name__ == "__main__":
    main()
