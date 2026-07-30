"""Predev-bound dev84 prediction and isolated scoring for NoAbsPE v12.

``predict`` has no target or sealed-test input.  Before it may open a dev
identity sidecar or pose cache, it validates a six-of-six passed v12 predev
artifact, exact checkpoint/config/source bindings, and replays the frozen
2,048-sample white-noise null.

``score`` first validates immutable prediction bytes and their receipt.  Only
after that validation may its separate process open the dev target manifest.
Neither command exposes a sealed-test input.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

import pams.consensus as consensus_module
import pams.data as data_module
import pams.period as period_module
from pams.config import PAMSConfig, load_config
from pams.consensus import MultiExpertCounter
from pams.data import (
    PoseInputManifest,
    UnlabeledVideoRecord,
    load_dev_target_manifest,
    load_pose_cache_set,
    load_pose_input_commitment,
    load_pose_input_manifest,
    pose_input_identity_sha256,
    validate_pose_input_binding,
)
from pams.diagnostics import _device, _peek_checkpoint, _stable_file_sha256
from pams.metrics import compute_count_metrics
from pams.reproducibility import hardware_fingerprint
from pams.training import collate_pose_sequences, load_model_checkpoint
from pams.types import CountResult, PoseSequence


def _load_sibling_predev_module() -> Any:
    source = Path(__file__).resolve().with_name("run_noabspe_projected_null_predev_gate.py")
    spec = importlib.util.spec_from_file_location(
        "_pams_noabspe_projected_null_predev_gate",
        source,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load the sibling v12 predev gate")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


predev = _load_sibling_predev_module()

METHOD_ID = "pams-noabspe-projected-null-multiexpert-v12"
CLASSIFICATION = "exploratory-derived predev-bound dev candidate"
PREDICTION_BATCH_SIZE = 8
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 2026
_PREDICTION_NAME = "predictions.json"
_PREDICTION_RECEIPT_NAME = "prediction.receipt.json"
_EVALUATION_NAME = "evaluation.json"
_EVALUATION_RECEIPT_NAME = "evaluation.receipt.json"
_EXPECTED_PREDEV_SOURCE_SHA256 = "78a843f35799194f0fba88bf00add9d4c00a764da68fd2f82553f8ce808a3f98"
_EXPECTED_PROTOCOL_FREEZE = {
    "derived_from_post_v11_exploration": True,
    "must_not_rewrite_v11_results": True,
    "frozen_before_seed3407_training": True,
    "candidate_config_fingerprint": (
        "e64029a8c1ae1bf258a71eabe03fd35cd9cd3f18bdf0c7e4ae17657cf9c6a13e"
    ),
    "null_seed": 73_400_711,
    "null_sample_total": 2_048,
    "significance_alpha": 0.01,
    "white_noise_batch_size": 32,
    "diagnostic_seed": 2026,
    "sampled_training_video_total": 64,
    "required_device_type": "cuda",
}
_EXPECTED_PREDEV_METHOD = {
    "period_input": "encoder projected_pose_pre_pe",
    "period_estimator": "projected-pose velocity vector-ACF spectrum",
    "period_prediction_changed_by_calibration": False,
    "raw_statistic": "selected bounded-band vector-ACF spectral power share",
    "null": "fixed seeded IID U[0,1] temporal white pose noise",
    "empirical_p": "(1 + count(null_stat >= observed_stat)) / (N + 1)",
    "confidence": "max(0, 1 - empirical_p / alpha)",
}
_EXPECTED_PREDEV_CRITERIA = {
    "frame_index_r2": ("<=", 0.10),
    "zero_pose_period_confidence": ("<=", 0.10),
    "random_pose_period_confidence": ("<=", 0.10),
    "training_period_top_bin_share": ("<=", 0.25),
    "position_invariance_median_cosine": (">=", 0.95),
    "synthetic_period_median_relative_error": ("<=", 0.10),
}
_FROZEN_COUNTER_PARAMETERS: dict[str, Any] = {
    "sigma_multipliers": [0.05, 0.12, 0.15],
    "distance_multipliers": [0.5, 0.8, 1.2],
    "short_window_multiplier": 0.5,
    "long_window_multiplier": 2.0,
    "height_factor": 0.6,
    "prominence_factor": 0.25,
    "long_window_weight": 0.6,
    "expert_mode": "multi",
}
_PLANNED_STRESS_SCENARIOS = (
    "clean",
    "accelerate_0.5_to_2.0",
    "oscillating_speed_0.5_to_2.0",
    "pause_20_percent",
    "noise_sigma_0.08",
    "joint_occlusion_20_percent_time_10_of_33_joints",
)
_STRESS_NMAE_MAXIMUM = 0.05
_STRESS_OBO_MINIMUM = 0.95


def _sha256_file(path: str | Path) -> str:
    return _stable_file_sha256(Path(path))[0]


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json_object(path: Path, *, document: str) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant in {document}: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{document} root must be an object")
    return payload


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    with path.open("xb") as handle:
        handle.write(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _module_path(module: Any) -> Path:
    source = getattr(module, "__file__", None)
    if not source:
        raise RuntimeError(f"source path unavailable for module {module!r}")
    return Path(source).resolve()


def _source_hashes() -> dict[str, str]:
    paths = {
        "dev_runner": Path(__file__).resolve(),
        "predev_gate": _module_path(predev),
        "pams_consensus": _module_path(consensus_module),
        "pams_data": _module_path(data_module),
        "pams_period": _module_path(period_module),
    }
    return {name: _sha256_file(path) for name, path in paths.items()}


def _require_frozen_predev_source(source_hashes: Mapping[str, str]) -> None:
    if source_hashes.get("predev_gate") != _EXPECTED_PREDEV_SOURCE_SHA256:
        raise ValueError("v12 predev source bytes differ from the frozen candidate")


def _algorithm_parameters() -> dict[str, Any]:
    return {
        "method_id": METHOD_ID,
        "prediction_batch_size": PREDICTION_BATCH_SIZE,
        "period_readout": {
            "input": "projected_pose_pre_pe",
            "estimator": "estimate_period_from_projected_pose",
            "period_argmax_changed_from_predev": False,
            "confidence": _EXPECTED_PREDEV_METHOD["confidence"],
            "null_seed": _EXPECTED_PROTOCOL_FREEZE["null_seed"],
            "null_sample_total": _EXPECTED_PROTOCOL_FREEZE["null_sample_total"],
            "null_batch_size": _EXPECTED_PROTOCOL_FREEZE["white_noise_batch_size"],
            "significance_alpha": _EXPECTED_PROTOCOL_FREEZE["significance_alpha"],
        },
        "counting_stream": "temporal_component(projected_pose_pre_pe)",
        "counter": _FROZEN_COUNTER_PARAMETERS,
        "synthetic_stress": {
            "counts": [2, 40],
            "scenarios": list(_PLANNED_STRESS_SCENARIOS),
            "per_scenario_nmae_maximum": _STRESS_NMAE_MAXIMUM,
            "per_scenario_obo_minimum": _STRESS_OBO_MINIMUM,
            "planned_scenarios_passed_before_dev_runner_freeze": True,
            "extended_full_frame_invalid_gap": {
                "planned_requirement": False,
                "status": "failed",
                "nmae": 0.2578248795524859,
                "obo": 4 / 39,
                "interpretation": (
                    "the frozen counter does not impute repetitions inside a "
                    "contiguous 20-percent interval with every frame invalid"
                ),
            },
        },
        "rounding": "counter integer peak counts; no target-conditioned rounding",
    }


def _load_bound_dev_input(
    sidecar_path: Path,
    commitment_path: Path,
) -> tuple[PoseInputManifest, Any, str, str]:
    sidecar_sha256 = _sha256_file(sidecar_path)
    commitment_sha256 = _sha256_file(commitment_path)
    sidecar = load_pose_input_manifest(sidecar_path, validate_exact=True)
    commitment = load_pose_input_commitment(commitment_path)
    validate_pose_input_binding(
        sidecar,
        commitment,
        sidecar_sha256=sidecar_sha256,
    )
    if sidecar.protocol != "ucfrep_526" or sidecar.split != "dev":
        raise ValueError("prediction requires the canonical ucfrep_526 dev sidecar")
    if len(sidecar.records) != 84:
        raise ValueError("prediction requires exactly 84 dev identities")
    if _sha256_file(sidecar_path) != sidecar_sha256:
        raise RuntimeError("dev sidecar changed while it was loaded")
    if _sha256_file(commitment_path) != commitment_sha256:
        raise RuntimeError("dev commitment changed while it was loaded")
    return sidecar, commitment, sidecar_sha256, commitment_sha256


def _validate_numeric_criterion(
    name: str,
    item: Any,
) -> float:
    if not isinstance(item, Mapping) or item.get("pass") is not True:
        raise ValueError(f"passed-predev criterion did not pass: {name}")
    value = item.get("value")
    threshold = item.get("threshold")
    expected_operator, expected_threshold = _EXPECTED_PREDEV_CRITERIA[name]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or item.get("operator") != expected_operator
        or float(threshold) != expected_threshold
    ):
        raise ValueError(f"passed-predev criterion is malformed: {name}")
    numeric = float(value)
    satisfied = (
        numeric <= expected_threshold
        if expected_operator == "<="
        else numeric >= expected_threshold
    )
    if not satisfied:
        raise ValueError(f"passed-predev criterion value failed: {name}")
    return numeric


def _validate_passed_predev_payload(
    payload: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    checkpoint_bytes: int,
    config_sha256: str,
    config_bytes: int,
    config: PAMSConfig,
) -> str:
    """Require exact v12 semantics, inputs, constants, and six passing gates."""

    if payload.get("artifact_type") != ("pams_noabspe_projected_null_predev_gate_v12"):
        raise ValueError("passed-predev artifact_type is not the v12 gate")
    if payload.get("classification") != ("exploratory-derived target-free candidate protocol"):
        raise ValueError("passed-predev classification mismatch")
    if payload.get("disclosed_by_pams_authors") is not False:
        raise ValueError("passed-predev must remain independently inferred")
    if payload.get("eligible_for_paper_table") is not False:
        raise ValueError("passed-predev cannot be paper-table eligible")
    if payload.get("protocol_freeze") != _EXPECTED_PROTOCOL_FREEZE:
        raise ValueError("passed-predev protocol constants differ from v12")
    if payload.get("method") != _EXPECTED_PREDEV_METHOD:
        raise ValueError("passed-predev period/confidence semantics differ")

    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("passed-predev inputs must be an object")
    expected_inputs = {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_stage": "encoder",
        "config_sha256": config_sha256,
        "config_bytes": config_bytes,
        "config_fingerprint": config.fingerprint,
        "pose_fingerprint": config.pose_fingerprint,
        "diagnostic_seed": 2026,
        "device_type": "cuda",
        "full_checkpoint_pose_cache_set_verified": True,
    }
    mismatched = [
        name for name, expected in expected_inputs.items() if inputs.get(name) != expected
    ]
    if mismatched:
        raise ValueError(f"passed-predev input binding mismatch: {mismatched}")

    firewall = payload.get("label_firewall")
    if not isinstance(firewall, Mapping):
        raise ValueError("passed-predev label firewall is missing")
    for name in (
        "dataset_manifest_argument_supported",
        "action_or_count_label_argument_supported",
        "development_input_mounted",
        "development_pose_mounted",
        "development_labels_mounted",
        "sealed_test_input_mounted",
        "sealed_test_labels_mounted",
    ):
        if firewall.get(name) is not False:
            raise ValueError(f"passed-predev label firewall mismatch: {name}")
    if firewall.get("label_fields_accessed") != []:
        raise ValueError("passed-predev accessed label fields")

    null_distribution = payload.get("null_distribution")
    if not isinstance(null_distribution, Mapping):
        raise ValueError("passed-predev null distribution is missing")
    expected_null = {
        "sample_total": 2_048,
        "seed": 73_400_711,
        "generation_batch_size": 32,
        "probe_random_seed_is_disjoint": True,
        "alpha": 0.01,
        "dtype": "little-endian float32",
    }
    null_mismatched = [
        name for name, expected in expected_null.items() if null_distribution.get(name) != expected
    ]
    if null_mismatched:
        raise ValueError(f"passed-predev null constants mismatch: {null_mismatched}")
    null_sha256 = null_distribution.get("sha256")
    if (
        not isinstance(null_sha256, str)
        or len(null_sha256) != 64
        or any(character not in "0123456789abcdef" for character in null_sha256)
    ):
        raise ValueError("passed-predev null SHA-256 is malformed")

    gate = payload.get("gate")
    if not isinstance(gate, Mapping):
        raise ValueError("passed-predev gate must be an object")
    expected_flags = {
        "thresholds_frozen_before_seed3407_training": True,
        "overall_pass": True,
        "dev84_prediction_authorized": True,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }
    flag_mismatch = [
        name for name, expected in expected_flags.items() if gate.get(name) is not expected
    ]
    if flag_mismatch:
        raise ValueError(f"passed-predev authorization mismatch: {flag_mismatch}")
    criteria = gate.get("criteria")
    if not isinstance(criteria, Mapping) or set(criteria) != set(_EXPECTED_PREDEV_CRITERIA):
        raise ValueError("passed-predev must contain exactly six criteria")
    criterion_values = {
        name: _validate_numeric_criterion(name, criteria[name])
        for name in _EXPECTED_PREDEV_CRITERIA
    }

    synthetic = payload.get("synthetic_period_recovery")
    if not isinstance(synthetic, Mapping):
        raise ValueError("passed-predev synthetic recovery must be an object")
    if (
        synthetic.get("periods") != [4, 8, 16, 32, 64, 128]
        or synthetic.get("prediction_period_unchanged_by_calibration") is not True
        or synthetic.get("truth_source") != "deterministic in-run synthetic generation only"
    ):
        raise ValueError("passed-predev synthetic semantics mismatch")
    rows = synthetic.get("rows")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("passed-predev requires six synthetic probes")
    errors: list[float] = []
    for expected, row in zip((4, 8, 16, 32, 64, 128), rows, strict=True):
        if not isinstance(row, Mapping):
            raise ValueError("passed-predev synthetic row must be an object")
        predicted = row.get("predicted_period_frames")
        reported_error = row.get("relative_error")
        if (
            row.get("expected_period_frames") != expected
            or isinstance(predicted, bool)
            or not isinstance(predicted, (int, float))
            or not math.isfinite(float(predicted))
            or isinstance(reported_error, bool)
            or not isinstance(reported_error, (int, float))
            or not math.isfinite(float(reported_error))
        ):
            raise ValueError("passed-predev synthetic row is malformed")
        computed = abs(float(predicted) - expected) / expected
        if not math.isclose(
            computed,
            float(reported_error),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("passed-predev synthetic relative error mismatch")
        for field in (
            "raw_peak_share",
            "null_exceedance_total",
            "empirical_one_sided_p",
            "calibrated_periodicity_confidence",
        ):
            if field not in row:
                raise ValueError(f"passed-predev synthetic evidence is missing {field}")
        errors.append(computed)
    if not math.isclose(
        float(np.median(np.asarray(errors))),
        criterion_values["synthetic_period_median_relative_error"],
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError("passed-predev synthetic median mismatch")

    if payload.get("read_only_verification") != {
        "checkpoint_sha256_unchanged": True,
        "config_sha256_unchanged": True,
        "model_or_training_state_updated": False,
        "pose_cache_write_operations": 0,
    }:
        raise ValueError("passed-predev read-only verification mismatch")
    return null_sha256


def _load_passed_predev(
    path: Path,
    *,
    checkpoint_sha256: str,
    checkpoint_bytes: int,
    config_sha256: str,
    config_bytes: int,
    config: PAMSConfig,
) -> tuple[dict[str, Any], str, str]:
    artifact_sha256 = _sha256_file(path)
    payload = _load_json_object(path, document="passed predev artifact")
    null_sha256 = _validate_passed_predev_payload(
        payload,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_bytes=checkpoint_bytes,
        config_sha256=config_sha256,
        config_bytes=config_bytes,
        config=config,
    )
    if _sha256_file(path) != artifact_sha256:
        raise RuntimeError("passed-predev artifact changed while validated")
    return payload, artifact_sha256, null_sha256


def _validate_replayed_null_sha(
    passed_predev: Mapping[str, Any],
    null: np.ndarray,
) -> str:
    if null.dtype != np.dtype("<f4") or null.shape != (2_048,):
        raise ValueError("replayed null shape/dtype differs from v12")
    null_sha256 = hashlib.sha256(null.tobytes(order="C")).hexdigest()
    expected = passed_predev["null_distribution"]["sha256"]
    if null_sha256 != expected:
        raise ValueError("replayed null SHA-256 differs from passed predev")
    return null_sha256


def _validated_counter(config: PAMSConfig) -> MultiExpertCounter:
    actual = {
        "sigma_multipliers": list(config.consensus.sigma_multipliers),
        "distance_multipliers": list(config.consensus.distance_multipliers),
        "short_window_multiplier": config.consensus.short_window_multiplier,
        "long_window_multiplier": config.consensus.long_window_multiplier,
        "height_factor": config.consensus.height_factor,
        "prominence_factor": config.consensus.prominence_factor,
        "long_window_weight": config.consensus.long_window_weight,
        "expert_mode": config.consensus.expert_mode,
    }
    if actual != _FROZEN_COUNTER_PARAMETERS:
        raise ValueError("configuration differs from frozen multi-expert defaults")
    return MultiExpertCounter(
        sigma_multipliers=config.consensus.sigma_multipliers,
        distance_multipliers=config.consensus.distance_multipliers,
        short_window_multiplier=config.consensus.short_window_multiplier,
        long_window_multiplier=config.consensus.long_window_multiplier,
        height_factor=config.consensus.height_factor,
        prominence_factor=config.consensus.prominence_factor,
        long_window_weight=config.consensus.long_window_weight,
        expert_mode=config.consensus.expert_mode,
    )


def _synthetic_projected_case(
    expected_count: int,
    scenario: str,
) -> tuple[Tensor, Tensor]:
    """Build one deterministic projected stream stress case."""

    if expected_count not in range(2, 41):
        raise ValueError("synthetic count must be in [2, 40]")
    if scenario not in (*_PLANNED_STRESS_SCENARIOS, "full_frame_invalid_gap"):
        raise ValueError("unknown projected-stream stress scenario")
    frames = 256
    time = torch.arange(frames, dtype=torch.float32)
    weights = torch.ones(frames, dtype=torch.float32)
    if scenario == "accelerate_0.5_to_2.0":
        weights = torch.linspace(0.5, 2.0, frames)
    elif scenario == "oscillating_speed_0.5_to_2.0":
        weights = 1.25 + 0.75 * torch.sin(2.0 * torch.pi * time / frames)
    elif scenario == "pause_20_percent":
        weights[102:154] = 0.0
    progress = (torch.cumsum(weights, dim=0) - weights[0]) / weights.sum()
    phase = 2.0 * torch.pi * expected_count * progress - torch.pi / 2.0
    raw = torch.stack(
        [
            (0.65 + 0.7 * index / 98.0) * torch.sin(phase + 2.0 * torch.pi * index / 99.0)
            for index in range(99)
        ],
        dim=-1,
    )
    valid = torch.ones(frames, dtype=torch.bool)
    if scenario == "noise_sigma_0.08":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(71_000 + expected_count)
        raw += 0.08 * torch.randn(
            raw.shape,
            generator=generator,
            dtype=raw.dtype,
        )
    elif scenario == "joint_occlusion_20_percent_time_10_of_33_joints":
        raw[102:154, :30] = 0.0
    elif scenario == "full_frame_invalid_gap":
        valid[102:154] = False
        raw[~valid] = 0.0
    projection_generator = torch.Generator(device="cpu")
    projection_generator.manual_seed(91_337)
    projection = torch.randn(
        (99, 64),
        generator=projection_generator,
        dtype=torch.float32,
    ) / math.sqrt(99.0)
    return raw @ projection, valid


def _stress_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = np.asarray(
        [int(row["expected_count"]) for row in rows],
        dtype=np.int64,
    )
    predicted = np.asarray(
        [int(row["predicted_count"]) for row in rows],
        dtype=np.int64,
    )
    absolute = np.abs(predicted - expected)
    return {
        "sample_total": len(rows),
        "nmae": float(np.mean(absolute / expected)),
        "obo": float(np.mean(absolute <= 1)),
        "exact": float(np.mean(absolute == 0)),
        "maximum_absolute_error": int(np.max(absolute)),
    }


def run_projected_stream_synthetic_stress() -> dict[str, Any]:
    """Run the pre-registered label-free count/stream stress suite."""

    counter = MultiExpertCounter()
    reports: dict[str, Any] = {}
    for scenario in (*_PLANNED_STRESS_SCENARIOS, "full_frame_invalid_gap"):
        rows: list[dict[str, Any]] = []
        for expected_count in range(2, 41):
            projected, valid = _synthetic_projected_case(
                expected_count,
                scenario,
            )
            period, raw_confidence = period_module.estimate_period_from_projected_pose(
                projected,
                minimum=4,
                maximum=128,
                valid_mask=valid,
            )
            stream = period_module.temporal_component(projected, valid)
            result = counter.count(
                stream,
                period_frames=float(period[0]),
                valid_mask=valid,
                period_confidence=float(raw_confidence[0]),
            )
            rows.append(
                {
                    "expected_count": expected_count,
                    "predicted_count": result.count,
                    "period_frames": float(period[0]),
                    "expert_counts": list(result.expert_counts),
                }
            )
        metrics = _stress_metrics(rows)
        planned = scenario in _PLANNED_STRESS_SCENARIOS
        threshold_pass = (
            metrics["nmae"] <= _STRESS_NMAE_MAXIMUM and metrics["obo"] >= _STRESS_OBO_MINIMUM
        )
        reports[scenario] = {
            "planned_requirement": planned,
            "threshold_pass": threshold_pass,
            "metrics": metrics,
            "rows": rows,
        }
    return {
        "schema_version": 1,
        "artifact_type": "pams_projected_stream_synthetic_stress_v12",
        "classification": "target-free synthetic diagnostic",
        "counts": [2, 40],
        "frames": 256,
        "planned_thresholds": {
            "per_scenario_nmae_maximum": _STRESS_NMAE_MAXIMUM,
            "per_scenario_obo_minimum": _STRESS_OBO_MINIMUM,
        },
        "planned_scenarios": list(_PLANNED_STRESS_SCENARIOS),
        "planned_overall_pass": all(
            reports[name]["threshold_pass"] for name in _PLANNED_STRESS_SCENARIOS
        ),
        "extended_full_frame_invalid_gap_pass": reports["full_frame_invalid_gap"]["threshold_pass"],
        "reports": reports,
        "label_firewall": {
            "dataset_or_video_input_used": False,
            "count_or_action_label_input_used": False,
            "development_or_test_input_used": False,
            "truth_source": "deterministic in-run synthetic generation only",
        },
    }


def _prediction_record(
    sequence: PoseSequence,
    result: CountResult,
    evidence: Mapping[str, Any],
    *,
    video_sha256: str,
    pose_cache_sha256: str,
) -> dict[str, Any]:
    return {
        "video_id": sequence.video_id,
        "video_sha256": video_sha256,
        "pose_cache_sha256": pose_cache_sha256,
        **result.to_dict(include_stream=True),
        "valid_frames": int(np.sum(sequence.valid_mask)),
        "period_evidence": dict(evidence),
    }


def predict_projected_null_batches(
    model: nn.Module,
    sequences: Sequence[PoseSequence],
    config: PAMSConfig,
    *,
    null: np.ndarray,
    null_sha256: str,
    video_sha256: Mapping[str, str],
    pose_cache_sha256: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    """Produce frozen counts with the predev-identical period/confidence."""

    items = tuple(sequences)
    if not items:
        raise ValueError("prediction requires at least one pose sequence")
    if hashlib.sha256(null.tobytes(order="C")).hexdigest() != null_sha256:
        raise ValueError("prediction null bytes do not match bound SHA-256")
    counter = _validated_counter(config)
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    model.eval()
    records: list[dict[str, Any]] = []
    for start in range(0, len(items), PREDICTION_BATCH_SIZE):
        batch_sequences = items[start : start + PREDICTION_BATCH_SIZE]
        batch = collate_pose_sequences(batch_sequences).to(model_device)
        with torch.inference_mode():
            _, projected = model.encoder.forward_with_pre_pe(
                batch.poses,
                batch.valid_mask,
            )
            periods, raw_confidences = period_module.estimate_period_from_projected_pose(
                projected,
                minimum=config.period.minimum,
                maximum=config.period.maximum,
                valid_mask=batch.valid_mask,
            )
            streams = period_module.temporal_component(
                projected,
                batch.valid_mask,
            )
        for index, sequence in enumerate(batch_sequences):
            length = sequence.num_frames
            raw_confidence = float(raw_confidences[index])
            calibration = predev._calibrate_peak_share(
                raw_confidence,
                null,
            )
            calibrated_confidence = float(calibration["calibrated_periodicity_confidence"])
            period_frames = float(periods[index])
            result = counter.count(
                streams[index, :length],
                period_frames=period_frames,
                valid_mask=batch.valid_mask[index, :length],
                period_confidence=calibrated_confidence,
            ).to_count_result()
            evidence = {
                "period_input": "projected_pose_pre_pe",
                "period_estimator": "estimate_period_from_projected_pose",
                "period_argmax_changed_from_predev": False,
                "period_frames": period_frames,
                **calibration,
                "null_sha256": null_sha256,
                "counting_stream": "temporal_component(projected_pose_pre_pe)",
            }
            records.append(
                _prediction_record(
                    sequence,
                    result,
                    evidence,
                    video_sha256=video_sha256[sequence.video_id],
                    pose_cache_sha256=pose_cache_sha256[sequence.video_id],
                )
            )
    return tuple(records)


def run_predict(arguments: argparse.Namespace) -> dict[str, Any]:
    """Freeze dev84 predictions without opening or accepting targets."""

    checkpoint = arguments.checkpoint.resolve()
    config_path = arguments.config.resolve()
    passed_predev_path = arguments.passed_predev.resolve()
    output_dir = arguments.output_dir.resolve()
    predictions_path = output_dir / _PREDICTION_NAME
    receipt_path = output_dir / _PREDICTION_RECEIPT_NAME
    collisions = [str(path) for path in (predictions_path, receipt_path) if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite prediction artifacts: {collisions}")

    source_hashes = _source_hashes()
    _require_frozen_predev_source(source_hashes)
    source_code_sha256 = _sha256_json(source_hashes)
    parameters = _algorithm_parameters()
    parameter_sha256 = _sha256_json(parameters)
    checkpoint_sha256, checkpoint_bytes = _stable_file_sha256(checkpoint)
    config_sha256, config_bytes = _stable_file_sha256(config_path)
    config = load_config(config_path)
    predev._validate_candidate_config(config)
    _validated_counter(config)
    passed_predev, passed_predev_sha256, declared_null_sha256 = _load_passed_predev(
        passed_predev_path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_bytes=checkpoint_bytes,
        config_sha256=config_sha256,
        config_bytes=config_bytes,
        config=config,
    )
    stage, provenance = _peek_checkpoint(checkpoint, config)
    if stage != "encoder":
        raise ValueError("prediction requires a frozen encoder checkpoint")
    device = _device("cuda")
    model = load_model_checkpoint(
        checkpoint,
        config,
        device=device,
        expected_stage="encoder",
        expected_provenance=provenance,
    ).eval()

    # Replay and bind the exact null before any dev identity or pose access.
    null = predev._build_white_noise_null(
        model,
        config,
        device=device,
    )
    replayed_null_sha256 = _validate_replayed_null_sha(
        passed_predev,
        null,
    )
    if replayed_null_sha256 != declared_null_sha256:
        raise RuntimeError("validated predev null binding changed")

    # First permitted dev input access: predev and null replay already passed.
    dev_inputs = arguments.dev_inputs.resolve()
    dev_commitment = arguments.dev_commitment.resolve()
    pose_cache_dir = arguments.pose_cache_dir.resolve()
    sidecar, commitment, sidecar_sha256, commitment_sha256 = _load_bound_dev_input(
        dev_inputs, dev_commitment
    )
    sequences, pose_snapshot = load_pose_cache_set(
        sidecar.records,
        cache_dir=pose_cache_dir,
        pose_fingerprint=config.pose_fingerprint,
    )
    expected_ids = tuple(record.video_id for record in sidecar.records)
    if tuple(sequence.video_id for sequence in sequences) != expected_ids:
        raise RuntimeError("dev pose-cache order differs from committed sidecar")
    video_hashes = {record.video_id: str(record.video_sha256) for record in sidecar.records}
    cache_hashes = {entry.video_id: entry.cache_sha256 for entry in pose_snapshot.entries}
    records = predict_projected_null_batches(
        model,
        sequences,
        config,
        null=null,
        null_sha256=replayed_null_sha256,
        video_sha256=video_hashes,
        pose_cache_sha256=cache_hashes,
    )
    if tuple(str(record["video_id"]) for record in records) != expected_ids:
        raise RuntimeError("prediction order differs from committed dev sidecar")

    if _stable_file_sha256(checkpoint) != (
        checkpoint_sha256,
        checkpoint_bytes,
    ):
        raise RuntimeError("checkpoint changed during prediction")
    if _stable_file_sha256(config_path) != (config_sha256, config_bytes):
        raise RuntimeError("configuration changed during prediction")
    if _sha256_file(passed_predev_path) != passed_predev_sha256:
        raise RuntimeError("passed-predev artifact changed during prediction")
    if _sha256_file(dev_inputs) != sidecar_sha256:
        raise RuntimeError("dev sidecar changed during prediction")
    if _sha256_file(dev_commitment) != commitment_sha256:
        raise RuntimeError("dev commitment changed during prediction")
    if _source_hashes() != source_hashes:
        raise RuntimeError("prediction source changed during prediction")

    payload = {
        "schema_version": 1,
        "artifact_type": "pams_noabspe_projected_null_dev_predictions_v12",
        "method_id": METHOD_ID,
        "classification": CLASSIFICATION,
        "diagnostic_only": True,
        "eligible_for_paper_table": False,
        "protocol": "ucfrep_526",
        "split": "dev",
        "record_total": 84,
        "source_files_sha256": source_hashes,
        "source_code_sha256": source_code_sha256,
        "predev_source_sha256": source_hashes["predev_gate"],
        "algorithm_parameters": parameters,
        "algorithm_parameters_sha256": parameter_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_source_git_sha": provenance.source_git_sha,
        "config_sha256": config_sha256,
        "config_bytes": config_bytes,
        "config_fingerprint": config.fingerprint,
        "pose_fingerprint": config.pose_fingerprint,
        "passed_predev_sha256": passed_predev_sha256,
        "null_sha256": replayed_null_sha256,
        "passed_predev_authorization": {
            "overall_pass": passed_predev["gate"]["overall_pass"],
            "dev84_prediction_authorized": passed_predev["gate"]["dev84_prediction_authorized"],
            "dev84_scoring_authorized": passed_predev["gate"]["dev84_scoring_authorized"],
            "test105_evaluation_authorized": passed_predev["gate"]["test105_evaluation_authorized"],
        },
        "dev_inputs_sha256": sidecar_sha256,
        "dev_commitment_sha256": commitment_sha256,
        "dev_identity_sha256": commitment.identity_sha256,
        "dev_pose_cache_set_sha256": pose_snapshot.fingerprint,
        "runtime": hardware_fingerprint(),
        "mount_audit": {
            "network": "none",
            "checkpoint_mounted": True,
            "config_mounted": True,
            "passed_predev_mounted": True,
            "dev_identity_mounted": True,
            "dev_pose_mounted": True,
            "dev_targets_mounted": False,
            "test_identity_mounted": False,
            "test_pose_mounted": False,
            "test_targets_mounted": False,
        },
        "records": list(records),
    }
    prediction_sha256 = _write_json_exclusive(predictions_path, payload)
    receipt = {
        "schema_version": 1,
        "artifact_type": ("pams_noabspe_projected_null_dev_prediction_receipt_v12"),
        "method_id": METHOD_ID,
        "protocol": "ucfrep_526",
        "split": "dev",
        "prediction_file": _PREDICTION_NAME,
        "prediction_sha256": prediction_sha256,
        "prediction_bytes": predictions_path.stat().st_size,
        "record_total": 84,
        "source_code_sha256": source_code_sha256,
        "predev_source_sha256": source_hashes["predev_gate"],
        "algorithm_parameters_sha256": parameter_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "passed_predev_sha256": passed_predev_sha256,
        "null_sha256": replayed_null_sha256,
        "dev_inputs_sha256": sidecar_sha256,
        "dev_commitment_sha256": commitment_sha256,
        "dev_identity_sha256": commitment.identity_sha256,
        "dev_pose_cache_set_sha256": pose_snapshot.fingerprint,
        "label_firewall": {
            "prediction_loaded_dev_targets": False,
            "prediction_loaded_test_identity": False,
            "prediction_loaded_test_targets": False,
            "gt_count_or_action_oracle_used": False,
        },
    }
    receipt_sha256 = _write_json_exclusive(receipt_path, receipt)
    return {
        "method_id": METHOD_ID,
        "record_total": 84,
        "predictions_path": str(predictions_path),
        "prediction_sha256": prediction_sha256,
        "prediction_receipt_path": str(receipt_path),
        "prediction_receipt_sha256": receipt_sha256,
        "dev_scoring_authorized": True,
        "test105_evaluation_authorized": False,
    }


def _validated_count_result(row: Mapping[str, Any]) -> CountResult:
    experts = row.get("expert_counts")
    if not isinstance(experts, list) or len(experts) != 3:
        raise ValueError("prediction expert_counts must contain three values")
    stream = row.get("period_stream")
    if not isinstance(stream, list) or len(stream) != 256:
        raise ValueError("prediction period_stream must contain 256 values")
    return CountResult(
        count=row.get("count"),  # type: ignore[arg-type]
        period_frames=float(row.get("period_frames", math.nan)),
        expert_counts=tuple(experts),  # type: ignore[arg-type]
        confidence=float(row.get("confidence", math.nan)),
        period_stream=np.asarray(stream, dtype=np.float32),
    )


def _validate_canonical_identity(payload: Mapping[str, Any]) -> None:
    rows = payload["records"]
    records = tuple(
        UnlabeledVideoRecord(
            video_id=str(row["video_id"]),
            video_path=f"frozen-dev/{row['video_id']}.avi",
            video_sha256=str(row["video_sha256"]),
        )
        for row in rows
    )
    manifest = PoseInputManifest(
        protocol="ucfrep_526",
        split="dev",
        records=records,
    )
    manifest.validate_exact_membership()
    if pose_input_identity_sha256(records) != payload["dev_identity_sha256"]:
        raise ValueError("prediction rows do not bind declared dev identity")


def _validate_frozen_predictions(
    predictions_path: Path,
    receipt_path: Path,
) -> tuple[dict[str, Any], str, str]:
    prediction_sha256 = _sha256_file(predictions_path)
    receipt_sha256 = _sha256_file(receipt_path)
    predictions = _load_json_object(predictions_path, document="predictions")
    receipt = _load_json_object(
        receipt_path,
        document="prediction receipt",
    )
    if predictions.get("artifact_type") != ("pams_noabspe_projected_null_dev_predictions_v12"):
        raise ValueError("unexpected prediction artifact_type")
    if receipt.get("artifact_type") != ("pams_noabspe_projected_null_dev_prediction_receipt_v12"):
        raise ValueError("unexpected prediction receipt artifact_type")
    if predictions.get("method_id") != METHOD_ID or receipt.get("method_id") != METHOD_ID:
        raise ValueError("prediction method identity mismatch")
    if predictions.get("protocol") != "ucfrep_526" or predictions.get("split") != "dev":
        raise ValueError("prediction protocol/split mismatch")
    if receipt.get("prediction_sha256") != prediction_sha256:
        raise ValueError("prediction receipt SHA-256 mismatch")
    if receipt.get("prediction_bytes") != predictions_path.stat().st_size:
        raise ValueError("prediction receipt byte count mismatch")
    current_sources = _source_hashes()
    _require_frozen_predev_source(current_sources)
    if predictions.get("source_files_sha256") != current_sources:
        raise ValueError("scoring source differs from prediction source")
    if predictions.get("source_code_sha256") != _sha256_json(current_sources):
        raise ValueError("prediction aggregate source hash mismatch")
    if predictions.get("algorithm_parameters") != _algorithm_parameters():
        raise ValueError("prediction algorithm parameters drifted")
    if predictions.get("algorithm_parameters_sha256") != _sha256_json(_algorithm_parameters()):
        raise ValueError("prediction parameter hash mismatch")
    for field in (
        "source_code_sha256",
        "predev_source_sha256",
        "algorithm_parameters_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "passed_predev_sha256",
        "null_sha256",
        "dev_inputs_sha256",
        "dev_commitment_sha256",
        "dev_identity_sha256",
        "dev_pose_cache_set_sha256",
    ):
        if receipt.get(field) != predictions.get(field):
            raise ValueError(f"prediction receipt metadata mismatch: {field}")
    authorization = predictions.get("passed_predev_authorization")
    if authorization != {
        "overall_pass": True,
        "dev84_prediction_authorized": True,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }:
        raise ValueError("prediction predev authorization is invalid")
    rows = predictions.get("records")
    if not isinstance(rows, list) or len(rows) != 84:
        raise ValueError("prediction artifact must contain 84 records")
    identifiers: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("prediction rows must be objects")
        if "target" in row or "action" in row:
            raise ValueError("prediction rows must remain target-free")
        identifier = str(row.get("video_id", ""))
        if not identifier:
            raise ValueError("prediction video_id must be non-empty")
        identifiers.append(identifier)
        _validated_count_result(row)
        evidence = row.get("period_evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("prediction period_evidence must be an object")
        expected_evidence = {
            "period_input": "projected_pose_pre_pe",
            "period_estimator": "estimate_period_from_projected_pose",
            "period_argmax_changed_from_predev": False,
            "null_sha256": predictions.get("null_sha256"),
            "counting_stream": ("temporal_component(projected_pose_pre_pe)"),
        }
        for field, expected in expected_evidence.items():
            if evidence.get(field) != expected:
                raise ValueError(f"prediction period evidence mismatch: {field}")
        for field in (
            "period_frames",
            "raw_peak_share",
            "null_exceedance_total",
            "empirical_one_sided_p",
            "calibrated_periodicity_confidence",
        ):
            if field not in evidence:
                raise ValueError(f"prediction period evidence is missing {field}")
    if len(set(identifiers)) != 84:
        raise ValueError("prediction video_id values must be unique")
    _validate_canonical_identity(predictions)
    if _sha256_file(predictions_path) != prediction_sha256:
        raise RuntimeError("predictions changed while validated")
    if _sha256_file(receipt_path) != receipt_sha256:
        raise RuntimeError("prediction receipt changed while validated")
    return predictions, prediction_sha256, receipt_sha256


def run_score(arguments: argparse.Namespace) -> dict[str, Any]:
    """Score frozen predictions at the only label-bearing boundary."""

    predictions_path = arguments.predictions.resolve()
    receipt_path = arguments.prediction_receipt.resolve()
    output_dir = arguments.output_dir.resolve()
    evaluation_path = output_dir / _EVALUATION_NAME
    evaluation_receipt_path = output_dir / _EVALUATION_RECEIPT_NAME
    collisions = [str(path) for path in (evaluation_path, evaluation_receipt_path) if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite evaluation artifacts: {collisions}")

    # This complete validator does not stat, hash, or deserialize labels.
    predictions, prediction_sha256, receipt_sha256 = _validate_frozen_predictions(
        predictions_path, receipt_path
    )
    frozen_source_hashes = _source_hashes()

    # First permitted target access: predictions are already frozen.
    targets_path = arguments.dev_targets.resolve()
    dev_targets_sha256 = _sha256_file(targets_path)
    targets = load_dev_target_manifest(targets_path)
    rows = predictions["records"]
    prediction_ids = tuple(str(row["video_id"]) for row in rows)
    target_ids = tuple(record.video_id for record in targets.records)
    if prediction_ids != target_ids:
        raise ValueError("dev target order/identity differs from frozen predictions")
    if _sha256_file(targets_path) != dev_targets_sha256:
        raise RuntimeError("dev targets changed while loaded")

    report = compute_count_metrics(
        [int(row["count"]) for row in rows],
        [record.count for record in targets.records],
        video_ids=prediction_ids,
        actions=[record.action for record in targets.records],
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
        confidence_level=0.95,
    )
    if _sha256_file(predictions_path) != prediction_sha256:
        raise RuntimeError("predictions changed during scoring")
    if _sha256_file(receipt_path) != receipt_sha256:
        raise RuntimeError("prediction receipt changed during scoring")
    if _sha256_file(targets_path) != dev_targets_sha256:
        raise RuntimeError("dev targets changed during scoring")
    if _source_hashes() != frozen_source_hashes:
        raise RuntimeError("scoring source changed during scoring")

    evaluation = {
        "schema_version": 1,
        "artifact_type": "pams_noabspe_projected_null_dev_evaluation_v12",
        "method_id": METHOD_ID,
        "classification": CLASSIFICATION,
        "diagnostic_only": True,
        "eligible_for_paper_table": False,
        "protocol": "ucfrep_526",
        "split": "dev",
        "prediction_sha256": prediction_sha256,
        "prediction_receipt_sha256": receipt_sha256,
        "dev_targets_sha256": dev_targets_sha256,
        "bootstrap_pairing": "paired_prediction_target_rows",
        "metrics": report.to_dict(),
        "acceptance_gate": {
            "required_nmae_at_most": 0.228,
            "required_obo_at_least": 0.666,
            "nmae_pass": report.nmae <= 0.228,
            "obo_pass": report.obo >= 0.666,
            "joint_pass": report.nmae <= 0.228 and report.obo >= 0.666,
            "verified_reproduction": False,
        },
        "mount_audit": {
            "network": "none",
            "prediction_mounted": True,
            "dev_targets_mounted": True,
            "dev_pose_mounted": False,
            "checkpoint_mounted": False,
            "test_identity_mounted": False,
            "test_pose_mounted": False,
            "test_targets_mounted": False,
        },
    }
    evaluation_sha256 = _write_json_exclusive(
        evaluation_path,
        evaluation,
    )
    evaluation_receipt_sha256 = _write_json_exclusive(
        evaluation_receipt_path,
        {
            "schema_version": 1,
            "artifact_type": ("pams_noabspe_projected_null_dev_evaluation_receipt_v12"),
            "method_id": METHOD_ID,
            "evaluation_sha256": evaluation_sha256,
            "evaluation_bytes": evaluation_path.stat().st_size,
            "prediction_sha256": prediction_sha256,
            "prediction_receipt_sha256": receipt_sha256,
            "dev_targets_sha256": dev_targets_sha256,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "test105_evaluation_authorized": False,
        },
    )
    metrics = report.to_dict()
    metrics.pop("per_video")
    return {
        "method_id": METHOD_ID,
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": evaluation_sha256,
        "evaluation_receipt_path": str(evaluation_receipt_path),
        "evaluation_receipt_sha256": evaluation_receipt_sha256,
        "prediction_sha256": prediction_sha256,
        "dev_targets_sha256": dev_targets_sha256,
        "metrics": metrics,
        "test105_evaluation_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--config", type=Path, required=True)
    predict.add_argument("--passed-predev", type=Path, required=True)
    predict.add_argument("--dev-inputs", type=Path, required=True)
    predict.add_argument("--dev-commitment", type=Path, required=True)
    predict.add_argument("--pose-cache-dir", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.set_defaults(handler=run_predict)

    score = subparsers.add_parser("score")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--prediction-receipt", type=Path, required=True)
    score.add_argument("--dev-targets", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.set_defaults(handler=run_score)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    payload = arguments.handler(arguments)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
