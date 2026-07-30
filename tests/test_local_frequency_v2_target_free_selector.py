from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import pams.data as data_module
from pams.synthetic import SyntheticSpec, generate_synthetic_sample
from pams.types import PoseSequence

ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "scripts"
    / "server"
    / "run_local_frequency_v2_target_free_selector.py"
)


def _load_runner() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "local_frequency_v2_target_free_selector_test_module",
        RUNNER,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _short_periodic_sequence(video_id: str = "train-pose") -> PoseSequence:
    sample = generate_synthetic_sample(
        SyntheticSpec(
            video_id=video_id,
            frames=32,
            count=4,
            seed=3407,
        )
    )
    return sample.sequence


def test_cli_surface_has_only_train_sidecar_pose_cache_and_output() -> None:
    runner = _load_runner()
    parsed = runner._parse_arguments(
        [
            "--train-inputs",
            "train.inputs.json",
            "--train-pose-cache-dir",
            "pose-cache",
            "--output",
            "selection.json",
        ]
    )

    assert vars(parsed) == {
        "train_inputs": Path("train.inputs.json"),
        "train_pose_cache_dir": Path("pose-cache"),
        "output": Path("selection.json"),
    }
    assert set(inspect.signature(runner.run_selector).parameters) == {
        "train_inputs",
        "train_pose_cache_dir",
        "output",
    }
    source = RUNNER.read_text(encoding="utf-8")
    assert 'add_argument("--dev' not in source
    assert 'add_argument("--test' not in source
    assert 'add_argument("--target' not in source
    assert 'add_argument("--count' not in source
    assert 'add_argument("--action' not in source


def test_grid_is_exactly_512_and_candidate_metadata_is_complete() -> None:
    runner = _load_runner()
    methods = runner._candidate_methods()

    assert len(methods) == len(set(methods)) == 512
    assert (
        "xyz.trim0.w64.sum.c1.mean" in methods
        and "centered.trim0.1.norm.c1.5.median.scale_max" in methods
    )
    window = runner._candidate_parameters("centered.trim0.05.w96.norm.c1.5.median")
    multiscale = runner._candidate_parameters(
        "xyz.trim0.075.sum.c1.mean.scale_median"
    )
    assert window["window_frames"] == 96
    assert window["minimum_cycles_per_window"] == 1.5
    assert window["normalized_dimensions"] is True
    assert multiscale["window_frames"] == [64, 96, 128, 192, 256]
    assert multiscale["scale_reducer"] == "scale_median"


def test_full_grid_prediction_is_deterministic() -> None:
    runner = _load_runner()
    sequence = _short_periodic_sequence()

    first = runner._predict_all_candidates(sequence)
    second = runner._predict_all_candidates(sequence)

    assert first == second
    assert set(first) == set(runner._candidate_methods())
    assert all(2 <= value <= 40 for value in first.values())


def test_transform_suite_is_deterministic_and_has_frozen_scope() -> None:
    runner = _load_runner()
    sequence = _short_periodic_sequence()
    first = runner._training_transforms(sequence)
    second = runner._training_transforms(sequence)

    assert tuple(first) == runner._TRANSFORM_NAMES
    assert len(first) == 8
    for name in runner._TRANSFORM_NAMES:
        assert np.array_equal(first[name].xyz, second[name].xyz)
        assert np.array_equal(first[name].valid_mask, second[name].valid_mask)
    duplicate = runner._duplicate_time(sequence)
    assert duplicate.num_frames == 2 * sequence.num_frames
    assert np.array_equal(duplicate.xyz[: sequence.num_frames], sequence.xyz)
    assert np.array_equal(duplicate.xyz[sequence.num_frames :], sequence.xyz)


def test_constant_boundary_predictor_receives_explicit_degeneracy_penalty() -> None:
    runner = _load_runner()
    method = "xyz.trim0.w64.sum.c1.mean"
    truth = [2.0, 3.0, 4.0]
    robust = runner._score_candidate(
        method,
        count_sweep_predictions=[2, 3, 4],
        count_sweep_truth=truth,
        stress_predictions=[8, 8, 8],
        stress_truth=[8.0, 8.0, 8.0],
        original_predictions=[3, 5, 7],
        transformed_predictions=[[3] * 8, [5] * 8, [7] * 8],
        duplicate_predictions=[6, 10, 14],
    )
    collapsed = runner._score_candidate(
        method,
        count_sweep_predictions=[2, 2, 2],
        count_sweep_truth=truth,
        stress_predictions=[2, 2, 2],
        stress_truth=[8.0, 8.0, 8.0],
        original_predictions=[2, 2, 2],
        transformed_predictions=[[2] * 8, [2] * 8, [2] * 8],
        duplicate_predictions=[2, 2, 2],
    )

    assert robust.degenerate_candidate_flag == 0
    assert collapsed.degenerate_candidate_flag == 1
    assert collapsed.training_prediction_boundary_share == 1.0
    assert collapsed.training_prediction_mode_share == 1.0
    assert collapsed.prediction_boundary_and_mode_penalty == 2.0
    assert robust.rank_key < collapsed.rank_key


def test_selector_core_never_calls_dataset_label_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(
        data_module,
        "load_dev_target_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dataset label loader must not be called")
        ),
    )
    methods = runner._candidate_methods()

    def predictor(
        sequences: tuple[PoseSequence, ...],
    ) -> dict[str, dict[str, int]]:
        return {
            sequence.video_id: {
                method: 2 + (index % 5)
                for index, method in enumerate(methods)
            }
            for sequence in sequences
        }

    ranked, selected, hashes = runner._evaluate_candidates(
        training_sequences=(_short_periodic_sequence(),),
        predictor=predictor,
    )
    repeated, repeated_selected, repeated_hashes = runner._evaluate_candidates(
        training_sequences=(_short_periodic_sequence(),),
        predictor=predictor,
    )

    assert len(ranked) == 512
    assert ranked[0].candidate_key in methods
    assert [audit.rank_key for audit in ranked] == [
        audit.rank_key for audit in repeated
    ]
    assert selected == repeated_selected
    assert hashes == repeated_hashes
    assert len(selected["synthetic_count_sweep"]) == 39
    assert selected["unlabeled_train"][0]["video_id"] == "train-pose"
    assert set(hashes) == {"train-pose"}


def test_exclusive_binary_output_rejects_overwrite(tmp_path: Path) -> None:
    runner = _load_runner()
    output = tmp_path / "selector.json"
    runner._write_new_json(output, {"finite": 1.0})

    assert json.loads(output.read_text(encoding="utf-8")) == {"finite": 1.0}
    assert output.read_bytes().endswith(b"\n")
    with pytest.raises(FileExistsError):
        runner._write_new_json(output, {"finite": 2.0})


def test_replacement_train337_sidecar_bytes_are_rejected_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    replacement = tmp_path / "train.inputs.json"
    replacement.write_text('{"replacement":true}\n', encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "load_pose_input_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("replacement bytes must be rejected before parsing")
        ),
    )

    with pytest.raises(ValueError, match="exact frozen train337 sidecar bytes"):
        runner.run_selector(
            train_inputs=replacement,
            train_pose_cache_dir=tmp_path / "pose-cache",
            output=tmp_path / "selection.json",
        )


def test_frozen_train337_binding_includes_bytes_manifest_and_identity() -> None:
    runner = _load_runner()

    assert runner._FROZEN_TRAIN337_SIDECAR_SHA256 == (
        "e238d307b4ed37b2f6c19eefdf7fa8812ec3d231cd2a7091be38e1c279f04207"
    )
    assert runner._FROZEN_TRAIN337_MANIFEST_FINGERPRINT == (
        "e3d6c979f5d160e37199726bff773a803cc77a0357079d046462b58da6454662"
    )
    assert runner._FROZEN_TRAIN337_IDENTITY_SHA256 == (
        "88367535e296213377c366ed69abbd1e808c38049d79bb948091688e4c5b7b3f"
    )
