from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

SOURCE_GIT_SHA = "6ee25a2e1291c35c563140999e16c5aa65bc8ced"
DEV_TARGETS_SHA256 = "1ab3bc010ccb5d869af7662586594c973e53841838643e1e9b8cc1787b0efab6"
CONFIG_FINGERPRINTS = {
    42: "1" * 64,
    2026: "2" * 64,
    3407: "3" * 64,
}


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_pams_v7_seed_evaluations.py"
    specification = importlib.util.spec_from_file_location(
        "aggregate_pams_v7_seed_evaluations",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_evaluation(
    path: Path,
    *,
    seed: int,
    offset: int,
    strict: bool,
    tamper_last_id: bool = False,
    tamper_source: bool = False,
) -> None:
    targets = np.asarray([(index % 7) + 1 for index in range(84)], dtype=np.int64)
    predictions = targets + np.asarray(
        [((index + offset) % 3) - 1 for index in range(84)],
        dtype=np.int64,
    )
    predictions = np.maximum(predictions, 0)
    absolute = np.abs(predictions - targets)
    rows = []
    compact_predictions = []
    for index in range(84):
        video_id = f"video-{index:03d}"
        if tamper_last_id and index == 83:
            video_id = "different-video"
        rows.append(
            {
                "video_id": video_id,
                "action": f"action-{index % 4}",
                "prediction": float(predictions[index]),
                "rounded_prediction": int(predictions[index]),
                "target": int(targets[index]),
                "absolute_error": int(absolute[index]),
                "normalized_absolute_error": float(absolute[index] / targets[index]),
                "within_one": bool(absolute[index] <= 1),
                "exact": bool(absolute[index] == 0),
            }
        )
        compact_predictions.append(
            {
                "video_id": video_id,
                "count": int(predictions[index]),
                "period_frames": 16.0,
                "expert_counts": [int(predictions[index])] * 3,
                "confidence": 0.5,
            }
        )
    report = {
        "sample_count": 84,
        "nmae": float(np.mean(absolute / targets)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(absolute)))),
        "obo": float(np.mean(absolute <= 1)),
        "exact": float(np.mean(absolute == 0)),
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 2026,
        "confidence_intervals": {},
        "per_video": rows,
    }
    payload = {
        "schema_version": 1,
        "variant": "sshead",
        "split": "dev",
        "config_fingerprint": CONFIG_FINGERPRINTS[seed],
        "checkpoint_sha256": f"{offset + 4:064x}",
        "predictions": compact_predictions,
        "report": report,
    }
    if strict:
        source = "0" * 40 if tamper_source else SOURCE_GIT_SHA
        payload.update(
            {
                "artifact_type": "pams_checkpoint_dev_evaluation",
                "classification": "inferred repair",
                "table2_eligible": False,
                "protocol": "ucfrep_526",
                "method_key": "pams-sshead-inferred",
                "config_file_sha256": f"{offset + 7:064x}",
                "prediction_sha256": f"{offset + 10:064x}",
                "dev_targets_sha256": DEV_TARGETS_SHA256,
                "checkpoint_source_git_sha": source,
                "prediction_source_git_sha": source,
                "scoring_source_git_sha": source,
            }
        )
    else:
        payload.update(
            {
                "method_id": "pams-sshead",
                "stage": "sshead",
            }
        )
    path.write_text(json.dumps(payload), encoding="utf-8")


def _mixed_specifications(
    tmp_path: Path,
    *,
    tamper_last_id: bool = False,
    tamper_source: bool = False,
) -> list[tuple[int, Path]]:
    specifications = []
    for offset, seed in enumerate((42, 2026, 3407)):
        path = tmp_path / f"seed-{seed}.json"
        _write_evaluation(
            path,
            seed=seed,
            offset=offset,
            strict=seed == 3407,
            tamper_last_id=tamper_last_id and seed == 3407,
            tamper_source=tamper_source and seed == 3407,
        )
        specifications.append((seed, path))
    return specifications


def test_mixed_three_seed_aggregate_is_deterministic_paired_and_post_hoc(
    tmp_path: Path,
) -> None:
    module = _load_script()
    specifications = _mixed_specifications(tmp_path)

    first = module.aggregate_evaluations(
        specifications,
        source_git_sha=SOURCE_GIT_SHA,
        dev_targets_sha256=DEV_TARGETS_SHA256,
        config_fingerprints=CONFIG_FINGERPRINTS,
    )
    second = module.aggregate_evaluations(
        specifications,
        source_git_sha=SOURCE_GIT_SHA,
        dev_targets_sha256=DEV_TARGETS_SHA256,
        config_fingerprints=CONFIG_FINGERPRINTS,
    )

    assert first == second
    assert first["classification"] == "post-hoc-three-seed-variance-follow-up"
    assert first["seeds"] == [42, 2026, 3407]
    assert first["input_schema_coverage"] == {
        "legacy_seeds": [42, 2026],
        "strict_dev_score_seeds": [3407],
    }
    assert first["claim_boundary"]["eligible_for_original_paper_table"] is False
    assert first["claim_boundary"]["eligible_for_verified_pams_reproduction_claim"] is False
    assert first["bootstrap"]["samples"] == 10_000
    assert (
        first["metrics_after_half_up_rounding"]["nmae"]["paired_bootstrap_95_percent"]
        == second["metrics_after_half_up_rounding"]["nmae"]["paired_bootstrap_95_percent"]
    )
    assert first["input_evaluations"][0]["source_binding_evidence"].startswith("unverifiable")
    assert first["input_evaluations"][2]["source_binding_evidence"].startswith("verified")


@pytest.mark.parametrize(
    ("tamper_last_id", "tamper_source", "message"),
    [
        (True, False, "membership"),
        (False, True, "source"),
    ],
)
def test_mixed_three_seed_aggregate_rejects_binding_or_membership_drift(
    tmp_path: Path,
    tamper_last_id: bool,
    tamper_source: bool,
    message: str,
) -> None:
    module = _load_script()
    specifications = _mixed_specifications(
        tmp_path,
        tamper_last_id=tamper_last_id,
        tamper_source=tamper_source,
    )

    with pytest.raises(ValueError, match=message):
        module.aggregate_evaluations(
            specifications,
            source_git_sha=SOURCE_GIT_SHA,
            dev_targets_sha256=DEV_TARGETS_SHA256,
            config_fingerprints=CONFIG_FINGERPRINTS,
        )


def test_exclusive_writer_refuses_to_replace_existing_file(tmp_path: Path) -> None:
    module = _load_script()
    output = tmp_path / "aggregate.json"
    module._write_json_exclusive(output, {"first": True})

    with pytest.raises(FileExistsError):
        module._write_json_exclusive(output, {"first": False})
    assert json.loads(output.read_text(encoding="utf-8")) == {"first": True}
