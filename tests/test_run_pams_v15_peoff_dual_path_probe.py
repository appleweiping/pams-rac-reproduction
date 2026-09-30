from __future__ import annotations

import hashlib
import inspect
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.server import run_pams_v15_peoff_dual_path_probe as runner


def _passing_inputs() -> dict[str, object]:
    return {
        "projected_distribution": {"boundary_share": 0.24, "mode_share": 0.24},
        "projected_time_scale": {
            "candidate_comparison_total": 10,
            "eligible_comparison_total": 5,
            "relative_error": {"median": 0.15},
        },
        "peoff_post_distribution": {"boundary_share": 0.24, "mode_share": 0.24},
        "peoff_post_time_scale": {
            "candidate_comparison_total": 10,
            "eligible_comparison_total": 5,
            "relative_error": {"median": 0.15},
        },
        "cross_path": {
            "candidate_comparison_total": 10,
            "eligible_comparison_total": 5,
            "relative_error": {"median": 0.25},
        },
    }


def _artifact_payload() -> dict[str, object]:
    digest = "a" * 64
    return {
        "status": "peoff_gate_pass",
        "gate": {
            "peoff_gate_pass": True,
            "standalone_noabs_training_authorized": False,
        },
        "inputs": {
            "encoder_checkpoint_sha256": digest,
            "encoder_progress_sha256": digest,
            "config_sha256": digest,
            "source_export_receipt_sha256": digest,
            "train337_pose_cache_set_sha256": digest,
            "checkpoint_algorithm_source_git_sha": "b" * 40,
            "gate_code_source_git_sha": "c" * 40,
            "code_files_sha256_commitment": digest,
        },
        "hardware_sha256": digest,
        "runtime_sha256": digest,
    }


def test_cli_surface_is_read_only_train337_encoder_only() -> None:
    parsed = runner._parse_arguments(
        [
            "--encoder-checkpoint",
            "encoder.pt",
            "--encoder-progress",
            "encoder.jsonl",
            "--config",
            "v15.yaml",
            "--pose-cache-dir",
            "train337-pose",
            "--source-receipt",
            "source-export.receipt.json",
            "--output",
            "peoff-probe.json",
        ]
    )

    assert set(vars(parsed)) == {
        "encoder_checkpoint",
        "encoder_progress",
        "config",
        "pose_cache_dir",
        "source_receipt",
        "output",
        "device",
        "batch_size",
    }
    assert set(inspect.signature(runner.run_peoff_probe).parameters) == {
        "encoder_checkpoint_path",
        "encoder_progress_path",
        "config_path",
        "pose_cache_dir",
        "source_receipt_path",
        "device",
        "batch_size",
    }
    parser_source = inspect.getsource(runner._parse_arguments)
    for forbidden in (
        "--manifest",
        "--targets",
        "--action",
        "--count",
        "--dev",
        "--test",
        "--train",
        "--sshead",
        "--alias",
    ):
        assert f'"{forbidden}"' not in parser_source
        assert f"'{forbidden}'" not in parser_source


def test_gate_requires_all_ten_but_never_independently_authorizes_training() -> None:
    decision = runner._gate_decision(**_passing_inputs())

    assert len(decision["criteria"]) == 10
    assert decision["peoff_gate_pass"] is True
    assert decision["all_ten_diagnostic_criteria_pass"] is True
    assert decision["standalone_noabs_training_authorized"] is False
    assert decision["separate_alias_probe_result_available_to_this_runner"] is False
    assert decision["sshead_training_authorized"] is False
    assert decision["dev84_prediction_authorized"] is False
    assert decision["dev84_scoring_authorized"] is False
    assert decision["test105_evaluation_authorized"] is False
    assert sum(name.startswith("peoff_post_pe_") for name in decision["criteria"]) == 4

    failures: tuple[tuple[tuple[str, ...], object], ...] = (
        (("projected_distribution", "boundary_share"), 0.25),
        (("projected_distribution", "mode_share"), 0.25),
        (("projected_time_scale", "eligible_comparison_total"), 4),
        (("projected_time_scale", "relative_error", "median"), 0.151),
        (("peoff_post_distribution", "boundary_share"), 0.25),
        (("peoff_post_distribution", "mode_share"), 0.25),
        (("peoff_post_time_scale", "eligible_comparison_total"), 4),
        (("peoff_post_time_scale", "relative_error", "median"), 0.151),
        (("cross_path", "eligible_comparison_total"), 4),
        (("cross_path", "relative_error", "median"), 0.251),
    )
    for path, value in failures:
        inputs = deepcopy(_passing_inputs())
        destination = inputs
        for field in path[:-1]:
            destination = destination[field]  # type: ignore[assignment,index]
        destination[path[-1]] = value  # type: ignore[index]
        rejected = runner._gate_decision(**inputs)
        assert rejected["peoff_gate_pass"] is False
        assert rejected["standalone_noabs_training_authorized"] is False


def test_exact_terminal_v15_checkpoint_and_progress_hashes_are_required() -> None:
    valid = {
        "encoder_checkpoint": (runner._EXPECTED_ENCODER_CHECKPOINT_SHA256, 123),
        "encoder_progress": (runner._EXPECTED_ENCODER_PROGRESS_SHA256, 456),
    }
    runner._validate_exact_v15_artifact_identities(valid)

    wrong_checkpoint = deepcopy(valid)
    wrong_checkpoint["encoder_checkpoint"] = ("0" * 64, 123)
    with pytest.raises(ValueError, match="exact frozen terminal v15 artifacts"):
        runner._validate_exact_v15_artifact_identities(wrong_checkpoint)

    wrong_progress = deepcopy(valid)
    wrong_progress["encoder_progress"] = ("0" * 64, 456)
    with pytest.raises(ValueError, match="exact frozen terminal v15 artifacts"):
        runner._validate_exact_v15_artifact_identities(wrong_progress)


class _ModeOnlyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.position_encoding_mode = "sinusoidal"
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))


class _ModeOnlyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _ModeOnlyEncoder()


def test_in_memory_mode_is_restored_and_tensor_state_stays_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ModeOnlyModel()
    calls: list[str] = []
    empty_variant = runner._VariantEncoding(
        paths={"post_pe": (), "projected_pre_pe": ()},
        projected_batches=(),
        projected_tensor_sha256=hashlib.sha256(b"").hexdigest(),
        projected_tensor_elements=0,
        projected_tensor_max_abs_drift=None,
    )
    empty_mode = runner._ModeEncoding(original=empty_variant, scaled={})

    def fake_encode_mode(*args: object, **kwargs: object) -> runner._ModeEncoding:
        del args, kwargs
        calls.append(model.encoder.position_encoding_mode)
        return empty_mode

    monkeypatch.setattr(runner, "_encode_mode", fake_encode_mode)
    before = runner._module_state_sha256(model)
    _, _, audit = runner._run_in_memory_modes(
        model,  # type: ignore[arg-type]
        (),
        config=SimpleNamespace(),  # type: ignore[arg-type]
        device=torch.device("cpu"),
        batch_size=1,
    )

    assert calls == ["sinusoidal", "none"]
    assert model.encoder.position_encoding_mode == "sinusoidal"
    assert runner._module_state_sha256(model) == before
    assert audit["mode_restored"] is True
    assert audit["model_tensor_state_unchanged"] is True


def test_mode_is_restored_even_when_peoff_evaluation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ModeOnlyModel()
    empty_variant = runner._VariantEncoding(
        paths={"post_pe": (), "projected_pre_pe": ()},
        projected_batches=(),
        projected_tensor_sha256=hashlib.sha256(b"").hexdigest(),
        projected_tensor_elements=0,
        projected_tensor_max_abs_drift=None,
    )
    empty_mode = runner._ModeEncoding(original=empty_variant, scaled={})
    calls = 0

    def fake_encode_mode(*args: object, **kwargs: object) -> runner._ModeEncoding:
        nonlocal calls
        del args, kwargs
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic PE-off failure")
        return empty_mode

    monkeypatch.setattr(runner, "_encode_mode", fake_encode_mode)
    with pytest.raises(RuntimeError, match="synthetic PE-off failure"):
        runner._run_in_memory_modes(
            model,  # type: ignore[arg-type]
            (),
            config=SimpleNamespace(),  # type: ignore[arg-type]
            device=torch.device("cpu"),
            batch_size=1,
        )
    assert model.encoder.position_encoding_mode == "sinusoidal"


def test_projected_identity_requires_exact_periods_confidences_and_tensors() -> None:
    sample = runner._v14._PeriodSample("v", 12.0, 0.5)
    tensor = torch.tensor([[[1.0, 2.0]]])
    batch = runner._ProjectedBatch(("v",), tensor)
    digest_hasher = hashlib.sha256()
    runner._update_projected_hash(
        digest_hasher,
        video_ids=("v",),
        projected=tensor,
    )
    digest = digest_hasher.hexdigest()
    normal_variant = runner._VariantEncoding(
        paths={"projected_pre_pe": (sample,), "post_pe": (sample,)},
        projected_batches=(batch,),
        projected_tensor_sha256=digest,
        projected_tensor_elements=2,
        projected_tensor_max_abs_drift=None,
    )
    peoff_variant = runner._VariantEncoding(
        paths={"projected_pre_pe": (sample,), "post_pe": (sample,)},
        projected_batches=(),
        projected_tensor_sha256=digest,
        projected_tensor_elements=2,
        projected_tensor_max_abs_drift=0.0,
    )
    normal = runner._ModeEncoding(
        original=normal_variant,
        scaled={factor: normal_variant for factor in runner._TIME_SCALE_FACTORS},
    )
    peoff = runner._ModeEncoding(
        original=peoff_variant,
        scaled={factor: peoff_variant for factor in runner._TIME_SCALE_FACTORS},
    )

    audit = runner._projected_identity_audit(normal, peoff)
    assert audit["all_variants_projected_period_confidence_exact"] is True
    assert audit["overall_projected_tensor_max_abs_drift"] == 0.0

    changed = deepcopy(peoff)
    changed.original.paths["projected_pre_pe"] = (  # type: ignore[index]
        runner._v14._PeriodSample("v", 13.0, 0.5),
    )
    with pytest.raises(RuntimeError, match="projected period path"):
        runner._projected_identity_audit(normal, changed)


def test_artifact_and_receipt_are_exclusive_and_hash_bound(tmp_path: Path) -> None:
    output = tmp_path / "peoff-probe.json"
    payload = _artifact_payload()

    receipt_path, digest = runner._write_artifact_and_receipt(output, payload)

    artifact = output.read_bytes()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert digest == hashlib.sha256(artifact).hexdigest()
    assert receipt["artifact_sha256"] == digest
    assert receipt["artifact_bytes"] == len(artifact)
    assert receipt["peoff_gate_pass"] is True
    assert receipt["standalone_noabs_training_authorized"] is False
    with pytest.raises(FileExistsError, match="overwrite"):
        runner._write_artifact_and_receipt(output, payload)
