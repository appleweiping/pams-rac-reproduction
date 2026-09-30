from __future__ import annotations

import hashlib
import json
import math
import os
import types
from collections.abc import Callable
from pathlib import Path

import pytest

import pams.data as data_module
from pams.baselines import ivac_p2l_official_runner as runner
from pams.baselines import ivac_p2l_official_score as score
from pams.data import (
    DevTargetManifest,
    DevTargetRecord,
    PoseInputCommitment,
    PoseInputManifest,
    UnlabeledVideoRecord,
    pose_input_identity_sha256,
)
from pams.metrics import round_count

REPOSITORY = Path(__file__).parents[1]
FIXTURE_GIT_SHA = "a" * 40


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _runner_code_sha256() -> str:
    return _sha256(
        (REPOSITORY / runner.RUNNER_CODE_RELATIVE_PATH).read_bytes()
    )


def _dev_ids() -> tuple[str, ...]:
    return tuple(
        (
            REPOSITORY / "data/splits/ucfrep_526_dev_84.txt"
        ).read_text(encoding="utf-8").splitlines()
    )


def _write_inputs(
    root: Path,
) -> tuple[
    Path,
    Path,
    tuple[UnlabeledVideoRecord, ...],
    dict[str, object],
]:
    records = tuple(
        UnlabeledVideoRecord(
            video_id,
            f"videos/{video_id}.avi",
            _sha256(video_id.encode()),
        )
        for video_id in _dev_ids()
    )
    manifest = PoseInputManifest(
        protocol="ucfrep_526",
        split="dev",
        records=records,
    )
    sidecar = root / "inputs.json"
    sidecar.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    commitment = PoseInputCommitment(
        protocol="ucfrep_526",
        split="dev",
        record_total=84,
        identity_sha256=pose_input_identity_sha256(records),
        sidecar_sha256=_sha256(sidecar.read_bytes()),
        sidecar_fingerprint=manifest.fingerprint,
    )
    commitment_path = root / "inputs.commitment.json"
    commitment_path.write_text(
        json.dumps(commitment.to_dict()),
        encoding="utf-8",
    )
    binding: dict[str, object] = {
        "protocol": "ucfrep_526",
        "split": "dev",
        "sample_count": 84,
        "sidecar_sha256": _sha256(sidecar.read_bytes()),
        "commitment_sha256": _sha256(commitment_path.read_bytes()),
        "sidecar_fingerprint": manifest.fingerprint,
        "identity_sha256": pose_input_identity_sha256(records),
    }
    return sidecar, commitment_path, records, binding


def _prediction_payload(
    records: tuple[UnlabeledVideoRecord, ...],
    binding: dict[str, object],
    counts: tuple[float, ...],
) -> dict[str, object]:
    video_ids = tuple(record.video_id for record in records)
    return {
        "schema_version": 1,
        "method_id": runner.METHOD_ID,
        "classification": runner.CLASSIFICATION,
        "eligible_for_original_pams_ivac_p2l_cell": False,
        "labels_loaded": False,
        "scoring_performed": False,
        "runner_provenance": {
            "source_git_sha": FIXTURE_GIT_SHA,
            "runner_code_sha256": _runner_code_sha256(),
            "compatibility_image_digest": runner.COMPATIBILITY_IMAGE_DIGEST,
        },
        "source": runner.official_source_dict(),
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
            "sha256": runner.FROZEN_IVAC_P2L_OFFICIAL_CONFIG.fingerprint,
            "values": runner.FROZEN_IVAC_P2L_OFFICIAL_CONFIG.to_dict(),
        },
        "protocol_assumptions": list(runner.PROTOCOL_ASSUMPTIONS),
        "runtime_versions": {
            "python": "3.11.10",
            "torch": "2.5.1+cu124",
            "cuda_runtime": "12.4",
            "torchvision": "0.20.1+cu124",
            "mmcv": "1.4.0",
            "opencv": "4.10.0",
            "numpy": "1.26.4",
            "gpu_name": "fixture",
        },
        "restore_audit": {
            "checkpoint_epoch": 67,
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
                "rounded_count": round_count(raw),
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
            for record, raw in zip(records, counts, strict=True)
        ],
    }


def _write_pair(
    root: Path,
    counts: tuple[float, ...] = (4.0,) * 84,
) -> tuple[Path, Path, dict[str, object], dict[str, object], Path, Path]:
    sidecar, commitment, records, binding = _write_inputs(root)
    payload = _prediction_payload(records, binding, counts)
    prediction = root / "prediction.json"
    prediction.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "ivac_p2l_official_prediction_receipt",
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
        "config_sha256": runner.FROZEN_IVAC_P2L_OFFICIAL_CONFIG.fingerprint,
        "compatibility_image_digest": runner.COMPATIBILITY_IMAGE_DIGEST,
        "runner_source_git_sha": FIXTURE_GIT_SHA,
        "runner_code_sha256": _runner_code_sha256(),
        "labels_loaded": False,
        "scoring_performed": False,
    }
    receipt_path = root / "prediction.receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return prediction, receipt_path, payload, receipt, sidecar, commitment


def _rewrite_pair(
    prediction: Path,
    receipt_path: Path,
    payload: dict[str, object],
    receipt: dict[str, object],
) -> None:
    prediction.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt["prediction_sha256"] = _sha256(prediction.read_bytes())
    receipt["prediction_bytes"] = prediction.stat().st_size
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_targets(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    counts: tuple[int, ...] = (4,) * 84,
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
    target = root / "dev.targets.json"
    target.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    return target


def _fast_bootstrap(
    raw: tuple[float, ...],
    rounded: tuple[int, ...],
    targets: tuple[int, ...],
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    assert samples == 10_000 and seed == 2026
    return {
        key: {"low": value, "high": value}
        for key, value in score.compute_ivac_p2l_metrics(
            raw,
            rounded,
            targets,
        ).items()
    }


def test_metrics_and_paired_bootstrap_are_deterministic() -> None:
    metrics = score.compute_ivac_p2l_metrics(
        (2.4, 5.6),
        (2, 6),
        (4, 5),
    )
    assert metrics["nmae_rounded"] == pytest.approx((2 / 4 + 1 / 5) / 2)
    assert metrics["rmse_raw_prediction"] == pytest.approx(
        math.sqrt((1.6**2 + 0.6**2) / 2)
    )
    assert metrics["obo_rounded"] == 0.5
    first = score.paired_bootstrap(
        (2.4, 5.6),
        (2, 6),
        (4, 5),
        samples=50,
        seed=2026,
    )
    assert first == score.paired_bootstrap(
        (2.4, 5.6),
        (2, 6),
        (4, 5),
        samples=50,
        seed=2026,
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["source"].__setitem__("tree_sha256", "0" * 64),
        lambda value: value["config"]["values"].__setitem__(
            "compatibility_image_digest",
            "sha256:" + "0" * 64,
        ),
        lambda value: value["predictions"][0].__setitem__(
            "video_locator",
            "../escape.avi",
        ),
        lambda value: value["predictions"][0].__setitem__(
            "observed_video_sha256",
            "0" * 64,
        ),
        lambda value: value["runner_provenance"].__setitem__(
            "runner_code_sha256",
            "0" * 64,
        ),
        lambda value: value.pop("runner_provenance"),
    ],
    ids=(
        "source",
        "container",
        "locator",
        "video-sha",
        "runner-tamper",
        "runner-schema-missing",
    ),
)
def test_all_bindings_are_rejected_before_first_target_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    prediction, receipt_path, payload, receipt, sidecar, commitment = _write_pair(
        tmp_path
    )
    mutate(payload)
    _rewrite_pair(prediction, receipt_path, payload, receipt)
    target_loads = 0
    monkeypatch.setattr(score, "_clean_git_revision", lambda _root: FIXTURE_GIT_SHA)

    def forbidden(_path: object) -> None:
        nonlocal target_loads
        target_loads += 1
        raise AssertionError("target loader must remain unreachable")

    monkeypatch.setattr(score, "load_dev_target_manifest", forbidden)
    with pytest.raises(ValueError):
        score.score_ivac_p2l_dev(
            predictions_path=prediction,
            prediction_receipt_path=receipt_path,
            sidecar_path=sidecar,
            commitment_path=commitment,
            dev_targets_path=tmp_path / "must-not-open.json",
            output_dir=tmp_path / "score",
            repository_root=REPOSITORY,
        )
    assert target_loads == 0


def test_receipt_tamper_is_rejected_before_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction, receipt_path, _, receipt, sidecar, commitment = _write_pair(
        tmp_path
    )
    receipt["compatibility_image_digest"] = "sha256:" + "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(score, "_clean_git_revision", lambda _root: FIXTURE_GIT_SHA)
    monkeypatch.setattr(
        score,
        "load_dev_target_manifest",
        lambda _path: pytest.fail("target loader must remain unreachable"),
    )
    with pytest.raises(ValueError, match="receipt does not bind"):
        score.score_ivac_p2l_dev(
            predictions_path=prediction,
            prediction_receipt_path=receipt_path,
            sidecar_path=sidecar,
            commitment_path=commitment,
            dev_targets_path=tmp_path / "must-not-open.json",
            output_dir=tmp_path / "score",
            repository_root=REPOSITORY,
        )


def test_dirty_scoring_checkout_is_rejected_before_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction, receipt, _, _, sidecar, commitment = _write_pair(tmp_path)
    replies = iter(
        (
            types.SimpleNamespace(stdout=FIXTURE_GIT_SHA + "\n"),
            types.SimpleNamespace(stdout=" M runner.py\n"),
        )
    )
    monkeypatch.setattr(score.subprocess, "run", lambda *_args, **_kwargs: next(replies))
    monkeypatch.setattr(
        score,
        "load_dev_target_manifest",
        lambda _path: pytest.fail("target loader must remain unreachable"),
    )
    with pytest.raises(score.IVACP2LOfficialScoreError, match="clean full-SHA"):
        score.score_ivac_p2l_dev(
            predictions_path=prediction,
            prediction_receipt_path=receipt,
            sidecar_path=sidecar,
            commitment_path=commitment,
            dev_targets_path=tmp_path / "must-not-open.json",
            output_dir=tmp_path / "score",
            repository_root=REPOSITORY,
        )


def test_score_writes_metrics_and_detects_consumption_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction, receipt, _, _, sidecar, commitment = _write_pair(tmp_path)
    targets = _write_targets(tmp_path, monkeypatch)
    monkeypatch.setattr(score, "_clean_git_revision", lambda _root: FIXTURE_GIT_SHA)
    monkeypatch.setattr(score, "paired_bootstrap", _fast_bootstrap)
    summary = score.score_ivac_p2l_dev(
        predictions_path=prediction,
        prediction_receipt_path=receipt,
        sidecar_path=sidecar,
        commitment_path=commitment,
        dev_targets_path=targets,
        output_dir=tmp_path / "score",
        repository_root=REPOSITORY,
    )
    evaluation = json.loads(
        (tmp_path / "score/evaluation.json").read_text(encoding="utf-8")
    )
    assert summary["metrics"]["nmae_rounded"] == 0.0
    assert evaluation["bootstrap"]["samples"] == 10_000
    assert evaluation["sealed_test_status"] == "untouched"
    assert evaluation["provenance"]["runner_source_git_sha"] == FIXTURE_GIT_SHA
    assert evaluation["provenance"]["runner_code_sha256"] == (
        _runner_code_sha256()
    )
    evaluation_receipt = json.loads(
        (tmp_path / "score/evaluation.receipt.json").read_text(encoding="utf-8")
    )
    assert evaluation_receipt["runner_source_git_sha"] == FIXTURE_GIT_SHA
    assert evaluation_receipt["runner_code_sha256"] == _runner_code_sha256()

    second = tmp_path / "tamper"
    second.mkdir()
    prediction, receipt, _, _, sidecar, commitment = _write_pair(second)
    targets = _write_targets(second, monkeypatch)
    original_loader = score.load_dev_target_manifest

    def tampering_loader(path: str | Path) -> DevTargetManifest:
        loaded = original_loader(path)
        metadata = sidecar.stat()
        raw = sidecar.read_bytes()
        sidecar.write_bytes(raw.replace(b'"dev"', b'"dex"', 1))
        os.utime(sidecar, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
        return loaded

    monkeypatch.setattr(score, "load_dev_target_manifest", tampering_loader)
    with pytest.raises(score.IVACP2LOfficialScoreError, match="sidecar changed"):
        score.score_ivac_p2l_dev(
            predictions_path=prediction,
            prediction_receipt_path=receipt,
            sidecar_path=sidecar,
            commitment_path=commitment,
            dev_targets_path=targets,
            output_dir=second / "score",
            repository_root=REPOSITORY,
        )
    assert not (second / "score/evaluation.json").exists()


def test_runner_change_during_scoring_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction, receipt, _, _, sidecar, commitment = _write_pair(tmp_path)
    targets = _write_targets(tmp_path, monkeypatch)
    monkeypatch.setattr(score, "_clean_git_revision", lambda _root: FIXTURE_GIT_SHA)
    monkeypatch.setattr(score, "paired_bootstrap", _fast_bootstrap)
    real_loader = score.load_dev_target_manifest
    real_digest = score._stable_file_digest
    runner_path = (REPOSITORY / runner.RUNNER_CODE_RELATIVE_PATH).resolve()
    targets_loaded = False

    def mark_target_load(path: str | Path) -> DevTargetManifest:
        nonlocal targets_loaded
        loaded = real_loader(path)
        targets_loaded = True
        return loaded

    def tampered_runner_digest(path: Path) -> object:
        if path == runner_path and targets_loaded:
            return object()
        return real_digest(path)

    monkeypatch.setattr(score, "load_dev_target_manifest", mark_target_load)
    monkeypatch.setattr(score, "_stable_file_digest", tampered_runner_digest)
    with pytest.raises(score.IVACP2LOfficialScoreError, match="runner code changed"):
        score.score_ivac_p2l_dev(
            predictions_path=prediction,
            prediction_receipt_path=receipt,
            sidecar_path=sidecar,
            commitment_path=commitment,
            dev_targets_path=targets,
            output_dir=tmp_path / "score",
            repository_root=REPOSITORY,
        )
    assert not (tmp_path / "score/evaluation.json").exists()


def test_score_pair_write_failure_removes_exact_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction, receipt, _, _, sidecar, commitment = _write_pair(tmp_path)
    targets = _write_targets(tmp_path, monkeypatch)
    monkeypatch.setattr(score, "_clean_git_revision", lambda _root: FIXTURE_GIT_SHA)
    monkeypatch.setattr(score, "paired_bootstrap", _fast_bootstrap)
    real_writer = score._write_json_exclusive
    calls = 0

    def fail_receipt(path: Path, payload: object) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("fixture receipt failure")
        assert isinstance(payload, dict)
        return real_writer(path, payload)

    monkeypatch.setattr(score, "_write_json_exclusive", fail_receipt)
    output = tmp_path / "score"
    with pytest.raises(OSError, match="fixture receipt failure"):
        score.score_ivac_p2l_dev(
            predictions_path=prediction,
            prediction_receipt_path=receipt,
            sidecar_path=sidecar,
            commitment_path=commitment,
            dev_targets_path=targets,
            output_dir=output,
            repository_root=REPOSITORY,
        )
    assert not (output / "evaluation.json").exists()
    assert not (output / "evaluation.receipt.json").exists()


def test_scorer_cli_requires_label_free_input_pair() -> None:
    options = {
        option
        for action in score._build_parser()._actions
        for option in action.option_strings
    }
    assert "--sidecar" in options and "--commitment" in options
    assert "--repository-root" in options
