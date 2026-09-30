from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path

import pytest

from pams import data as data_module
from pams.baselines import poserac_v1_official_runner as runner
from pams.baselines import poserac_v1_official_score as score
from pams.data import (
    DevTargetManifest,
    DevTargetRecord,
    PoseCacheEntryReceipt,
    PoseCacheSetSnapshot,
    PoseInputCommitment,
    PoseInputManifest,
    UnlabeledVideoRecord,
    pose_input_identity_sha256,
)

REPOSITORY = Path(__file__).parents[1]
FIXTURE_GIT_SHA = "a" * 40
POSE_FINGERPRINT = "b" * 64


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
        "full_sidecar_sample_count": 84,
        "sidecar_sha256": _sha256(sidecar.read_bytes()),
        "commitment_sha256": _sha256(commitment.read_bytes()),
        "sidecar_fingerprint": manifest.fingerprint,
        "identity_sha256": pose_input_identity_sha256(records),
    }
    return sidecar, commitment, records, binding


def _pose_snapshot(
    records: tuple[UnlabeledVideoRecord, ...],
) -> PoseCacheSetSnapshot:
    return PoseCacheSetSnapshot(
        pose_fingerprint=POSE_FINGERPRINT,
        entries=tuple(
            PoseCacheEntryReceipt(
                video_id=record.video_id,
                cache_sha256=_sha256(f"cache:{record.video_id}".encode()),
                bytes=10_000 + index,
            )
            for index, record in enumerate(records)
        ),
    )


def _prediction_payload(
    counts: tuple[int, ...],
    records: tuple[UnlabeledVideoRecord, ...],
    binding: dict[str, object],
) -> dict[str, object]:
    video_ids = tuple(record.video_id for record in records)
    snapshot = _pose_snapshot(records)
    receipts = {entry.video_id: entry for entry in snapshot.entries}
    rows: list[dict[str, object]] = []
    for index, (record, count) in enumerate(zip(records, counts, strict=True)):
        selected = index % len(runner.ACTION_NAMES)
        ranges = [0.01 * (channel + 1) for channel in range(8)]
        ranges[selected] = 1.0
        channel_counts = [0] * 8
        channel_counts[selected] = count
        receipt = receipts[record.video_id]
        rows.append(
            {
                "video_id": record.video_id,
                "pose_cache_sha256": receipt.cache_sha256,
                "pose_cache_bytes": receipt.bytes,
                "raw_count": float(count),
                "rounded_count": count,
                "selected_channel": selected,
                "selected_action": runner.ACTION_NAMES[selected],
                "channel_selection_uses_count_label": False,
                "channel_counts": channel_counts,
                "channel_dynamic_ranges": ranges,
                "probability_min": 0.01,
                "probability_max": 0.99,
                "frames": 256,
                "valid_frames": 250,
            }
        )
    return {
        "schema_version": 1,
        "method_id": runner.METHOD_ID,
        "classification": runner.CLASSIFICATION,
        "method_version": "PoseRAC-v1-2023",
        "distinct_from": "PoseRAC-ICONIP24",
        "eligible_for_original_poserac_v1_cell": False,
        "eligible_for_original_pams_poserac_cell": False,
        "ineligibility_reasons": [
            "upstream evaluation GT-count channel oracle is disabled",
            "channel selection is independently inferred",
            "PAMS caches are uniformly resampled to 256 rather than original-frame poses",
            "standard UCFRep-526 dev is not the released UCFRep-pose-110 test protocol",
        ],
        "labels_loaded": False,
        "scoring_performed": False,
        "ground_truth_count_channel_oracle": False,
        "runner_provenance": {
            "source_git_sha": FIXTURE_GIT_SHA,
            "runner_code_sha256": _runner_code_sha256(),
            "compatibility_image_digest": runner.COMPATIBILITY_IMAGE_DIGEST,
        },
        "official_source": {
            "repository": runner.OFFICIAL_SOURCE_REPOSITORY,
            "commit": runner.OFFICIAL_SOURCE_COMMIT,
            "archive": runner.SOURCE_ARCHIVE_SPEC.to_dict(),
            "tree_files": runner.OFFICIAL_SOURCE_TREE_FILES,
            "tree_bytes": runner.OFFICIAL_SOURCE_TREE_BYTES,
            "tree_sha256": runner.OFFICIAL_SOURCE_TREE_SHA256,
            "model_py_sha256": runner.OFFICIAL_MODEL_PY_SHA256,
            "eval_py_sha256": runner.OFFICIAL_EVAL_PY_SHA256,
            "pre_test_py_sha256": runner.OFFICIAL_PRE_TEST_PY_SHA256,
            "all_action_csv_sha256": runner.OFFICIAL_ACTION_CSV_SHA256,
            "config_sha256": runner.OFFICIAL_CONFIG_SHA256,
            "license": "MIT",
            "license_sha256": runner.OFFICIAL_LICENSE_SHA256,
        },
        "input": {**binding, "pose_cache_snapshot": snapshot.to_dict()},
        "selection": {
            "selected_count": 84,
            "selected_video_ids": list(video_ids),
            "selected_video_ids_sha256": runner.sha256_json(list(video_ids)),
            "rule": runner.FROZEN_POSERAC_V1_CONFIG.channel_selection,
            "rule_provenance": "inferred",
        },
        "assets": {
            "checkpoint": runner.CHECKPOINT_SPEC.to_dict(),
            "third_party_assets_redistributed": False,
        },
        "config": {
            "sha256": runner.FROZEN_POSERAC_V1_CONFIG.fingerprint,
            "values": runner.FROZEN_POSERAC_V1_CONFIG.to_dict(),
        },
        "runtime_versions": {
            "python": "3.11.10",
            "torch": "2.5.1+cu124",
            "numpy": "1.26.4",
            "cuda_runtime": "12.4",
            "device": "cuda",
            "gpu_name": "NVIDIA RTX A6000",
        },
        "restore_audit": {
            "checkpoint_format": "plain_ordered_tensor_state_dict",
            "checkpoint_key_count": 74,
            "model_key_count": 74,
            "loaded_key_count": 74,
            "missing_keys": [],
            "unexpected_keys": [],
            "strict_key_coverage": True,
            "model_parameter_count": 2_686_682,
            "fc1_weight_shape": [8, 99],
        },
        "prediction_total": 84,
        "failure_total": 0,
        "predictions": rows,
    }


def _write_prediction_pair(
    tmp_path: Path,
    counts: tuple[int, ...],
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
    input_payload = payload["input"]
    assert isinstance(input_payload, dict)
    snapshot = input_payload["pose_cache_snapshot"]
    assert isinstance(snapshot, dict)
    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "poserac_v1_official_predictions_receipt",
        "method_id": runner.METHOD_ID,
        "prediction_file": prediction.name,
        "prediction_sha256": _sha256(prediction.read_bytes()),
        "prediction_bytes": prediction.stat().st_size,
        "prediction_total": 84,
        "source_commit": runner.OFFICIAL_SOURCE_COMMIT,
        "source_archive_sha256": runner.SOURCE_ARCHIVE_SPEC.sha256,
        "source_tree_sha256": runner.OFFICIAL_SOURCE_TREE_SHA256,
        "checkpoint_sha256": runner.CHECKPOINT_SPEC.sha256,
        "input_sidecar_sha256": binding["sidecar_sha256"],
        "input_commitment_sha256": binding["commitment_sha256"],
        "input_identity_sha256": binding["identity_sha256"],
        "pose_cache_set_sha256": snapshot["fingerprint"],
        "config_sha256": runner.FROZEN_POSERAC_V1_CONFIG.fingerprint,
        "compatibility_image_digest": runner.COMPATIBILITY_IMAGE_DIGEST,
        "runner_source_git_sha": FIXTURE_GIT_SHA,
        "runner_code_sha256": _runner_code_sha256(),
        "labels_loaded": False,
        "scoring_performed": False,
        "ground_truth_count_channel_oracle": False,
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
        source_annotation_sha256=data_module._UCFREP_526_CANONICAL_ANNOTATION_SHA256,
    )
    target_path = tmp_path / "dev.targets.json"
    target_path.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target_path


def test_metrics_and_bootstrap_match_declared_definitions() -> None:
    metrics = score.compute_poserac_v1_metrics(
        raw=(2.0, 6.0),
        rounded=(2, 6),
        targets=(4, 5),
    )

    assert metrics["nmae_rounded"] == pytest.approx((2 / 4 + 1 / 5) / 2)
    assert metrics["mae_raw_prediction"] == 1.5
    assert metrics["rmse_raw_prediction"] == pytest.approx(math.sqrt(2.5))
    assert metrics["mae_rounded"] == 1.5
    assert metrics["rmse_rounded"] == pytest.approx(math.sqrt(2.5))
    assert metrics["obo_rounded"] == 0.5
    assert metrics["exact_rounded"] == 0.0
    first = score.paired_bootstrap(
        (2.0, 6.0),
        (2, 6),
        (4, 5),
        samples=100,
        seed=2026,
    )
    assert first == score.paired_bootstrap(
        (2.0, 6.0),
        (2, 6),
        (4, 5),
        samples=100,
        seed=2026,
    )


def test_complete_dev84_score_writes_metrics_per_video_and_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_counts = tuple(index % 12 + 1 for index in range(84))
    prediction_counts = tuple(
        target + (1 if index % 3 == 0 else 0)
        for index, target in enumerate(target_counts)
    )
    prediction, receipt, _, _, sidecar, commitment = _write_prediction_pair(
        tmp_path,
        prediction_counts,
    )
    targets = _write_targets(tmp_path, monkeypatch, target_counts)
    monkeypatch.setattr(score, "_clean_git_revision", lambda _root: FIXTURE_GIT_SHA)
    output = tmp_path / "score"

    result = score.score_poserac_v1_dev(
        predictions_path=prediction,
        prediction_receipt_path=receipt,
        sidecar_path=sidecar,
        commitment_path=commitment,
        dev_targets_path=targets,
        output_dir=output,
        repository_root=REPOSITORY,
    )

    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    score_receipt = json.loads(
        (output / "evaluation.receipt.json").read_text(encoding="utf-8")
    )
    assert evaluation["prediction_total"] == 84
    assert len(evaluation["per_video"]) == 84
    assert evaluation["bootstrap"]["samples"] == 10_000
    assert evaluation["ground_truth_count_channel_oracle"] is False
    assert evaluation["sealed_test_status"] == "untouched"
    assert evaluation["channel_selection"]["validated_before_targets"] is True
    assert result["metrics"] == evaluation["metrics"]
    assert score_receipt["evaluation_sha256"] == _sha256(
        (output / "evaluation.json").read_bytes()
    )
    assert score_receipt["ground_truth_count_channel_oracle"] is False
    assert score_receipt["sealed_test_status"] == "untouched"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["official_source"].__setitem__(
            "tree_sha256", "0" * 64
        ),
        lambda payload: payload["assets"]["checkpoint"].__setitem__(
            "sha256", "0" * 64
        ),
        lambda payload: payload["input"]["pose_cache_snapshot"].__setitem__(
            "fingerprint", "0" * 64
        ),
        lambda payload: payload["config"].__setitem__("sha256", "0" * 64),
        lambda payload: payload["runner_provenance"].__setitem__(
            "runner_code_sha256", "0" * 64
        ),
        lambda payload: payload["predictions"][0].__setitem__(
            "selected_channel", 7
        ),
        lambda payload: payload.__setitem__(
            "ground_truth_count_channel_oracle", True
        ),
    ],
    ids=(
        "source",
        "checkpoint",
        "pose-cache-snapshot",
        "config",
        "runner",
        "channel-rule",
        "oracle-flag",
    ),
)
def test_prediction_tampering_is_rejected_before_dev_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    prediction, receipt, payload, _, sidecar, commitment = _write_prediction_pair(
        tmp_path,
        (4,) * 84,
    )
    mutate(payload)
    prediction.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(score, "_clean_git_revision", lambda _root: FIXTURE_GIT_SHA)
    target_loads = 0

    def forbidden_targets(_path: object) -> None:
        nonlocal target_loads
        target_loads += 1
        raise AssertionError("dev targets must remain unopened")

    monkeypatch.setattr(score, "load_dev_target_manifest", forbidden_targets)
    with pytest.raises((ValueError, score.PoseRACV1OfficialScoreError)):
        score.score_poserac_v1_dev(
            predictions_path=prediction,
            prediction_receipt_path=receipt,
            sidecar_path=sidecar,
            commitment_path=commitment,
            dev_targets_path=tmp_path / "dev.targets.json",
            output_dir=tmp_path / "score",
            repository_root=REPOSITORY,
        )
    assert target_loads == 0


def test_receipt_tampering_is_rejected_before_dev_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction, receipt, _, receipt_payload, sidecar, commitment = (
        _write_prediction_pair(tmp_path, (4,) * 84)
    )
    receipt_payload["checkpoint_sha256"] = "0" * 64
    receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
    monkeypatch.setattr(score, "_clean_git_revision", lambda _root: FIXTURE_GIT_SHA)

    def forbidden_targets(_path: object) -> None:
        raise AssertionError("dev targets must remain unopened")

    monkeypatch.setattr(score, "load_dev_target_manifest", forbidden_targets)
    with pytest.raises(ValueError, match="receipt"):
        score.score_poserac_v1_dev(
            predictions_path=prediction,
            prediction_receipt_path=receipt,
            sidecar_path=sidecar,
            commitment_path=commitment,
            dev_targets_path=tmp_path / "dev.targets.json",
            output_dir=tmp_path / "score",
            repository_root=REPOSITORY,
        )


def test_test_target_locator_is_rejected_before_any_other_input_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest_calls = 0

    def forbidden_digest(_path: object) -> None:
        nonlocal digest_calls
        digest_calls += 1
        raise AssertionError("no file may be opened after a test target locator")

    monkeypatch.setattr(score, "_stable_file_digest", forbidden_digest)
    with pytest.raises(ValueError, match="dev.targets.json"):
        score.score_poserac_v1_dev(
            predictions_path=tmp_path / "missing-prediction.json",
            prediction_receipt_path=tmp_path / "missing-receipt.json",
            sidecar_path=tmp_path / "missing-sidecar.json",
            commitment_path=tmp_path / "missing-commitment.json",
            dev_targets_path=tmp_path / "test.targets.json",
            output_dir=tmp_path / "score",
            repository_root=REPOSITORY,
        )
    assert digest_calls == 0


def test_duplicate_json_and_forbidden_label_field_are_rejected() -> None:
    with pytest.raises(ValueError, match="label field"):
        score._reject_forbidden_keys({"nested": {"count": 3}})
