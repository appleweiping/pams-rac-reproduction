from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _load_script() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "aggregate_jtsps_seed_evaluations.py"
    )
    specification = importlib.util.spec_from_file_location(
        "aggregate_jtsps_seed_evaluations",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_evaluation(path: Path, *, offset: int, tamper_last_id: bool = False) -> None:
    targets = np.asarray([(index % 7) + 1 for index in range(84)], dtype=np.int64)
    predictions = targets + np.asarray(
        [((index + offset) % 3) - 1 for index in range(84)],
        dtype=np.int64,
    )
    predictions = np.maximum(predictions, 0)
    absolute = np.abs(predictions - targets)
    rows = []
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
    metrics = {
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
        "method": "jtsps-count-only",
        "status": "inferred-clean-room-not-source-parity",
        "prediction_sha256": f"{offset + 1:064x}",
        "dev_targets_sha256": (
            "1ab3bc010ccb5d869af7662586594c973e53841838643e1e9b8cc1787b0efab6"
        ),
        "metrics": metrics,
        "raw_metrics": {
            "nmae": metrics["nmae"] + 0.01,
            "mae": metrics["mae"] + 0.01,
            "rmse": metrics["rmse"] + 0.01,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_three_seed_aggregate_is_deterministic_and_paired(tmp_path: Path) -> None:
    module = _load_script()
    specifications = []
    for offset, seed in enumerate((42, 2026, 3407)):
        path = tmp_path / f"seed-{seed}.json"
        _write_evaluation(path, offset=offset)
        specifications.append((seed, path))

    first = module.aggregate_evaluations(specifications)
    second = module.aggregate_evaluations(specifications)

    assert first == second
    assert first["seeds"] == [42, 2026, 3407]
    assert first["sample_count_per_seed"] == 84
    assert first["bootstrap"]["samples"] == 10_000
    assert (
        first["metrics_after_half_up_rounding"]["nmae"][
            "paired_bootstrap_95_percent"
        ]
        == second["metrics_after_half_up_rounding"]["nmae"][
            "paired_bootstrap_95_percent"
        ]
    )


def test_three_seed_aggregate_rejects_membership_drift(tmp_path: Path) -> None:
    module = _load_script()
    specifications = []
    for offset, seed in enumerate((42, 2026, 3407)):
        path = tmp_path / f"seed-{seed}.json"
        _write_evaluation(path, offset=offset, tamper_last_id=seed == 3407)
        specifications.append((seed, path))

    with pytest.raises(ValueError, match="membership"):
        module.aggregate_evaluations(specifications)
