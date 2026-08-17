from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from pams import data as data_module
from pams.baselines import transrac_official_runner as runner
from pams.baselines import transrac_official_score as score
from pams.data import (
    DevTargetManifest,
    DevTargetRecord,
    PoseInputCommitment,
    PoseInputManifest,
    UnlabeledVideoRecord,
    pose_input_identity_sha256,
)

REPOSITORY = Path(__file__).parents[1]
FIXTURE_GIT_SHA = "a" * 40


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _runner_code_sha256() -> str:
    return _sha256(
        (REPOSITORY / runner.RUNNER_CODE_RELATIVE_PATH).read_bytes()
    )


def _dev_ids() -> tuple[str, ...]:
    return tuple(
        (REPOSITORY / "data/splits/ucfrep_526_dev_84.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )


def _write_label_free_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, tuple[UnlabeledVideoRecord, ...], dict[str, object]]:
    records = tuple(
        UnlabeledVideoRecord(
            video_id=video_id,
            video_path=f"videos/{video_id}.avi",
            video_sha256=_sha256(video_id.encode()),
        )
        for video_id in _dev_ids()
    )
    manifest = PoseInputManifest(
        protocol="ucfrep_526",
        split="dev",
        records=records,
    )
    sidecar = tmp_path / "dev.inputs.json"
    sidecar.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    commitment_value = PoseInputCommitment(
        protocol=manifest.protocol,
        split=manifest.split,
        record_total=len(records),
        identity_sha256=pose_input_identity_sha256(records),
        sidecar_sha256=_sha256(sidecar.read_bytes()),
        sidecar_fingerprint=manifest.fingerprint,
    )
    commitment = tmp_path / "dev.inputs.commitment.json"
    commitment.write_text(
        json.dumps(commitment_value.to_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    binding: dict[str, object] = {
        "protocol": manifest.protocol,
        "split": manifest.split,
        "sample_count": len(records),
        "sidecar_sha256": _sha256(sidecar.read_bytes()),
        "commitment_sha256": _sha256(commitment.read_bytes()),
        "sidecar_fingerprint": manifest.fingerprint,
        "identity_sha256": pose_input_identity_sha256(records),
    }
    return sidecar, commitment, records, binding


def _prediction_payload(
    counts: tuple[float, ...],
    records: tuple[UnlabeledVideoRecord, ...],
    binding: dict[str, object],
) -> dict[str, object]:
    video_ids = tuple(record.video_id for record in records)
    rounded = tuple(runner.round_count(value) for value in counts)
    return {
        "schema_version": 1,
        "method_id": runner.METHOD_ID,
        "classification": runner.CLASSIFICATION,
        "eligible_for_original_pams_transrac_cell": False,
        "labels_loaded": False,
        "scoring_performed": False,
        "runner_provenance": {
            "source_git_sha": FIXTURE_GIT_SHA,
            "runner_code_sha256": _runner_code_sha256(),
            "compatibility_image_digest": runner.COMPATIBILITY_IMAGE_DIGEST,
        },
        "source": {
            "repository": runner.OFFICIAL_SOURCE_REPOSITORY,
            "commit": runner.OFFICIAL_SOURCE_COMMIT,
            "archive": runner.SOURCE_ARCHIVE_SPEC.public_dict(),
            "tree_sha256": runner.OFFICIAL_SOURCE_TREE_SHA256,
            "tree_files": runner.OFFICIAL_SOURCE_TREE_FILES,
            "tree_bytes": runner.OFFICIAL_SOURCE_TREE_BYTES,
            "license_file_sha256": runner.OFFICIAL_LICENSE_SHA256,
            "license_status": (
                "ambiguous-apache-file-versus-anti-996-readme-badge"
            ),
        },
        "input": binding,
        "selection": {
            "selected_count": 84,
            "selected_video_ids": list(video_ids),
            "selected_video_ids_sha256": runner.sha256_json(list(video_ids)),
        },
        "assets": {
            "backbone": runner.BACKBONE_SPEC.public_dict(),
            "checkpoint": runner.CHECKPOINT_SPEC.public_dict(),
            "third_party_assets_redistributed": False,
        },
        "config": {
            "sha256": runner.FROZEN_TRANSRAC_OFFICIAL_CONFIG.fingerprint,
            "values": runner.FROZEN_TRANSRAC_OFFICIAL_CONFIG.to_dict(),
        },
        "runtime_versions": {
            "python": "3.11.10",
            "torch": "2.5.1+cu124",
            "cuda_runtime": "12.4",
            "mmcv": "1.4.0",
            "timm": "0.4.12",
            "einops": "0.3.2",
            "kornia": "0.5.11",
            "gpu_name": "NVIDIA RTX A6000",
        },
        "restore_audit": {
            "checkpoint_epoch": 174,
            "checkpoint_filename_epoch_token": 171,
            "state_dict_keys": 230,
            "loaded_key_count": 230,
            "missing_keys": [],
            "unexpected_keys": [],
            "strict_key_coverage": True,
        },
        "decode_ok_total": 84,
        "failure_total": 0,
        "failures": [],
        "predictions": [
            {
                "video_id": record.video_id,
                "video_locator": record.video_path,
                "expected_video_sha256": record.video_sha256,
                "observed_video_sha256": record.video_sha256,
                "raw_count": raw,
                "rounded_count": rounded_count,
                "decoded_frame_count": 90,
                "sampled_frame_count": 64,
                "density_min": 0.0,
                "density_max": 0.5,
                "density_mean": raw / 64,
                "peak_allocated_mib": 700.0,
                "peak_reserved_mib": 800.0,
                "decode_status": "ok",
                "failure_reason": None,
            }
            for record, raw, rounded_count in zip(
                records,
                counts,
                rounded,
                strict=True,
            )
        ],
    }


def _write_prediction_pair(
    tmp_path: Path,
    counts: tuple[float, ...],
) -> tuple[
    Path,
    Path,
    dict[str, object],
    dict[str, object],
    Path,
    Path,
]:
    sidecar, commitment, records, binding = _write_label_free_inputs(tmp_path)
    payload = _prediction_payload(counts, records, binding)
    prediction = tmp_path / "prediction.json"
    prediction.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    binding = payload["input"]
    assert isinstance(binding, dict)
    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "transrac_official_prediction_receipt",
        "method_id": runner.METHOD_ID,
        "prediction_file": prediction.name,
        "prediction_sha256": _sha256(prediction.read_bytes()),
        "prediction_bytes": prediction.stat().st_size,
        "prediction_total": 84,
        "decode_ok_total": 84,
        "failure_total": 0,
        "source_commit": runner.OFFICIAL_SOURCE_COMMIT,
        "source_archive_sha256": runner.SOURCE_ARCHIVE_SPEC.sha256,
        "source_tree_sha256": runner.OFFICIAL_SOURCE_TREE_SHA256,
        "backbone_sha256": runner.BACKBONE_SPEC.sha256,
        "checkpoint_sha256": runner.CHECKPOINT_SPEC.sha256,
        "input_sidecar_sha256": binding["sidecar_sha256"],
        "input_commitment_sha256": binding["commitment_sha256"],
        "input_identity_sha256": binding["identity_sha256"],
        "config_sha256": runner.FROZEN_TRANSRAC_OFFICIAL_CONFIG.fingerprint,
        "compatibility_image_digest": runner.COMPATIBILITY_IMAGE_DIGEST,
        "runner_source_git_sha": FIXTURE_GIT_SHA,
        "runner_code_sha256": _runner_code_sha256(),
        "labels_loaded": False,
        "scoring_performed": False,
    }
    receipt_path = tmp_path / "prediction.receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return prediction, receipt_path, payload, receipt, sidecar, commitment


def _write_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    counts: tuple[int, ...],
) -> Path:
    records = tuple(
        DevTargetRecord(
            video_id=video_id,
            action=video_id.removeprefix("v_").split("_g", maxsplit=1)[0],
            count=count,
        )
        for video_id, count in zip(_dev_ids(), counts, strict=True)
    )
    monkeypatch.setattr(
        data_module,
        "_UCFREP_526_DEV_ANNOTATION_SHA256",
        data_module._dev_target_annotation_sha256(records),
    )
    manifest = DevTargetManifest(
        protocol="ucfrep_526",
        records=records,
        source_annotation_sha256=(
            data_module._UCFREP_526_CANONICAL_ANNOTATION_SHA256
        ),
    )
    target_path = tmp_path / "dev.targets.json"
    target_path.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target_path


def test_metrics_and_bootstrap_match_the_declared_definitions() -> None:
    metrics = score.compute_transrac_metrics(
        raw=(2.4, 5.6),
        rounded=(2, 6),
        targets=(4, 5),
    )

    assert metrics["nmae_rounded"] == pytest.approx((2 / 4 + 1 / 5) / 2)
    assert metrics["mae_raw_prediction"] == pytest.approx((1.6 + 0.6) / 2)
    assert metrics["rmse_raw_prediction"] == pytest.approx(
        math.sqrt((1.6**2 + 0.6**2) / 2)
    )
    assert metrics["mae_rounded"] == 1.5
    assert metrics["rmse_rounded"] == pytest.approx(math.sqrt(2.5))
    assert metrics["obo_rounded"] == 0.5
    assert metrics["exact_rounded"] == 0.0
    first = score.paired_bootstrap(
        (2.4, 5.6),
        (2, 6),
        (4, 5),
        samples=100,
        seed=2026,
    )
    assert first == score.paired_bootstrap(
        (2.4, 5.6),
        (2, 6),
        (4, 5),
        samples=100,
        seed=2026,
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["source"].__setitem__("tree_sha256", "0" * 64),
        lambda payload: payload["assets"]["checkpoint"].__setitem__(
            "sha256", "0" * 64
        ),
        lambda payload: payload["input"].__setitem__(
            "sidecar_sha256", "0" * 64
        ),
        lambda payload: payload["runner_provenance"].__setitem__(
            "runner_code_sha256", "0" * 64
        ),
    ],
    ids=("source", "checkpoint", "input", "runner"),
)
def test_prediction_tampering_is_rejected_before_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    prediction, receipt, payload, _, sidecar, commitment = _write_prediction_pair(
        tmp_path,
        (4.0,) * 84,
    )
    mutate(payload)
    prediction.write_text(json.dumps(payload), encoding="utf-8")
    target_loads = 0
    monkeypatch.setattr(score, "_clean_git_revision", lambda _root: FIXTURE_GIT_SHA)

    def forbidden_targets(_path: object) -> None:
        nonlocal target_loads
        target_loads += 1
        raise AssertionError("targets must remain unopened")

    monkeypatch.setattr(score, "load_dev_target_manifest", forbidden_targets)
    with pytest.raises(ValueError):
        score.score_transrac_dev(
            predictions_path=prediction,
            prediction_receipt_path=receipt,
            sidecar_path=sidecar,
            commitment_path=commitment,
            dev_targets_path=tmp_path / "must-not-open.targets.json",
            output_dir=tmp_path / "score",
            repository_root=REPOSITORY,
        )
    assert target_loads == 0


def test_receipt_tampering_is_rejected_before_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        prediction,
        receipt_path,
        _,
        receipt,
        sidecar,
        commitment,
    ) = _write_prediction_pair(
        tmp_path,
        (4.0,) * 84,
    )
    receipt["checkpoint_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    target_loads = 0
    monkeypatch.setattr(score, "_clean_git_revision", lambda _root: FIXTURE_GIT_SHA)

    def forbidden_targets(_path: object) -> None:
        nonlocal target_loads
        target_loads += 1
        raise AssertionError("targets must remain unopened")

    monkeypatch.setattr(score, "load_dev_target_manifest", forbidden_targets)
    with pytest.raises(ValueError, match="receipt does not bind"):
        score.score_transrac_dev(
            predictions_path=prediction,
            prediction_receipt_path=receipt_path,
            sidecar_path=sidecar,
            commitment_path=commitment,
            dev_targets_path=tmp_path / "must-not-open.targets.json",
            output_dir=tmp_path / "score",
            repository_root=REPOSITORY,
        )
    assert target_loads == 0


def test_strict_score_writes_metrics_ci_and_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions = (6.4, *((4.0,) * 83))
    (
        prediction,
        prediction_receipt,
        _,
        _,
        sidecar,
        commitment,
    ) = _write_prediction_pair(
        tmp_path,
        predictions,
    )
    targets = _write_targets(tmp_path, monkeypatch, (4,) * 84)
    monkeypatch.setattr(score, "_clean_git_revision", lambda _root: FIXTURE_GIT_SHA)
    output = tmp_path / "score"

    summary = score.score_transrac_dev(
        predictions_path=prediction,
        prediction_receipt_path=prediction_receipt,
        sidecar_path=sidecar,
        commitment_path=commitment,
        dev_targets_path=targets,
        output_dir=output,
        repository_root=REPOSITORY,
    )

    evaluation = json.loads(
        (output / "evaluation.json").read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (output / "evaluation.receipt.json").read_text(encoding="utf-8")
    )
    assert evaluation["classification"] == score.EVALUATION_CLASSIFICATION
    assert evaluation["eligible_for_original_paper_table"] is False
    assert evaluation["sealed_test_status"] == "untouched"
    assert evaluation["metrics"]["nmae_rounded"] == pytest.approx(0.5 / 84)
    assert evaluation["metrics"]["obo_rounded"] == pytest.approx(83 / 84)
    assert evaluation["metrics"]["mae_raw_prediction"] == pytest.approx(2.4 / 84)
    assert evaluation["bootstrap"]["samples"] == 10_000
    assert set(evaluation["bootstrap"]["confidence_intervals"]) == {
        "nmae_rounded",
        "mae_raw_prediction",
        "rmse_raw_prediction",
        "mae_rounded",
        "rmse_rounded",
        "obo_rounded",
        "exact_rounded",
    }
    assert receipt["prediction_sha256"] == _sha256(prediction.read_bytes())
    assert receipt["prediction_receipt_sha256"] == _sha256(
        prediction_receipt.read_bytes()
    )
    assert receipt["evaluation_sha256"] == _sha256(
        (output / "evaluation.json").read_bytes()
    )
    assert receipt["runner_source_git_sha"] == FIXTURE_GIT_SHA
    assert receipt["runner_code_sha256"] == _runner_code_sha256()
    assert summary["evaluation_receipt_sha256"] == _sha256(
        (output / "evaluation.receipt.json").read_bytes()
    )


def test_prediction_change_while_targets_load_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        prediction,
        prediction_receipt,
        _,
        _,
        sidecar,
        commitment,
    ) = _write_prediction_pair(
        tmp_path,
        (4.0,) * 84,
    )
    targets = _write_targets(tmp_path, monkeypatch, (4,) * 84)
    monkeypatch.setattr(score, "_clean_git_revision", lambda _root: FIXTURE_GIT_SHA)
    original_loader = score.load_dev_target_manifest

    def tampering_loader(path: str | Path) -> DevTargetManifest:
        loaded = original_loader(path)
        prediction.write_bytes(prediction.read_bytes() + b" ")
        return loaded

    monkeypatch.setattr(score, "load_dev_target_manifest", tampering_loader)
    with pytest.raises(
        score.TransRACOfficialScoreError,
        match="prediction artifact changed",
    ):
        score.score_transrac_dev(
            predictions_path=prediction,
            prediction_receipt_path=prediction_receipt,
            sidecar_path=sidecar,
            commitment_path=commitment,
            dev_targets_path=targets,
            output_dir=tmp_path / "score",
            repository_root=REPOSITORY,
        )
    assert not (tmp_path / "score" / "evaluation.json").exists()


def test_label_free_sidecar_change_during_scoring_is_detected_by_full_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        prediction,
        prediction_receipt,
        _,
        _,
        sidecar,
        commitment,
    ) = _write_prediction_pair(tmp_path, (4.0,) * 84)
    targets = _write_targets(tmp_path, monkeypatch, (4,) * 84)
    monkeypatch.setattr(score, "_clean_git_revision", lambda _root: FIXTURE_GIT_SHA)
    original_loader = score.load_dev_target_manifest

    def tampering_loader(path: str | Path) -> DevTargetManifest:
        loaded = original_loader(path)
        metadata = sidecar.stat()
        original = sidecar.read_bytes()
        mutated = original.replace(b'"dev"', b'"dex"', 1)
        assert len(mutated) == len(original) and mutated != original
        sidecar.write_bytes(mutated)
        os.utime(sidecar, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
        return loaded

    monkeypatch.setattr(score, "load_dev_target_manifest", tampering_loader)
    with pytest.raises(
        score.TransRACOfficialScoreError,
        match="sidecar changed during",
    ):
        score.score_transrac_dev(
            predictions_path=prediction,
            prediction_receipt_path=prediction_receipt,
            sidecar_path=sidecar,
            commitment_path=commitment,
            dev_targets_path=targets,
            output_dir=tmp_path / "score",
            repository_root=REPOSITORY,
        )
    assert not (tmp_path / "score" / "evaluation.json").exists()


def test_score_receipt_failure_removes_only_new_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        prediction,
        prediction_receipt,
        _,
        _,
        sidecar,
        commitment,
    ) = _write_prediction_pair(tmp_path, (4.0,) * 84)
    targets = _write_targets(tmp_path, monkeypatch, (4,) * 84)
    monkeypatch.setattr(score, "_clean_git_revision", lambda _root: FIXTURE_GIT_SHA)
    real_writer = score._write_json_exclusive
    calls = 0

    def fail_second_write(path: Path, payload: object) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("fixture receipt failure")
        assert isinstance(payload, dict)
        return real_writer(path, payload)

    monkeypatch.setattr(score, "_write_json_exclusive", fail_second_write)
    output = tmp_path / "score"
    with pytest.raises(OSError, match="fixture receipt failure"):
        score.score_transrac_dev(
            predictions_path=prediction,
            prediction_receipt_path=prediction_receipt,
            sidecar_path=sidecar,
            commitment_path=commitment,
            dev_targets_path=targets,
            output_dir=output,
            repository_root=REPOSITORY,
        )
    assert not (output / "evaluation.json").exists()
    assert not (output / "evaluation.receipt.json").exists()


def test_scorer_cli_requires_verified_label_free_input_pair() -> None:
    option_strings = {
        option
        for action in score._build_parser()._actions
        for option in action.option_strings
    }
    assert "--sidecar" in option_strings
    assert "--commitment" in option_strings
