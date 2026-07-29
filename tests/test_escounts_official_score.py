from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from pams import data as data_module
from pams.baselines import escounts_official_runner as runner
from pams.baselines import escounts_official_score as score
from pams.data import (
    DevTargetManifest,
    DevTargetRecord,
    PoseInputCommitment,
    PoseInputManifest,
    UnlabeledVideoRecord,
    pose_input_identity_sha256,
)

REPOSITORY = Path(__file__).parents[1]


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def audited_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    repository = tmp_path / "audited-repository"
    baseline_dir = repository / "src/pams/baselines"
    baseline_dir.mkdir(parents=True)
    runner_path = baseline_dir / "escounts_official_runner.py"
    worker_path = baseline_dir / "escounts_official_worker.py"
    scorer_path = baseline_dir / "escounts_official_score.py"
    runner_path.write_bytes(Path(runner.__file__).resolve(strict=True).read_bytes())
    worker_path.write_bytes(runner._WORKER_PATH.read_bytes())
    scorer_path.write_bytes(Path(score.__file__).resolve(strict=True).read_bytes())
    _git(repository, "init")
    _git(repository, "config", "user.email", "tests@example.invalid")
    _git(repository, "config", "user.name", "ESCounts Tests")
    _git(repository, "add", "--", ".")
    _git(repository, "commit", "-m", "freeze runner worker and scorer")
    monkeypatch.setattr(runner, "_RUNNER_PATH", runner_path.resolve(strict=True))
    monkeypatch.setattr(runner, "_WORKER_PATH", worker_path.resolve(strict=True))
    monkeypatch.setattr(score, "_SCORER_PATH", scorer_path.resolve(strict=True))
    return repository.resolve(strict=True)


def _audited_repository_root() -> Path:
    return runner._RUNNER_PATH.parents[3]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _dev_ids() -> tuple[str, ...]:
    return tuple(
        (REPOSITORY / "data/splits/ucfrep_526_dev_84.txt").read_text(encoding="utf-8").splitlines()
    )


def _source_binding() -> dict[str, object]:
    return {
        "repository": runner.OFFICIAL_SOURCE_REPOSITORY,
        "commit": runner.OFFICIAL_SOURCE_COMMIT,
        "tree": runner.OFFICIAL_SOURCE_TREE,
        "license": runner.OFFICIAL_SOURCE_LICENSE,
        "files": [
            {
                "relative_path": item.relative_path,
                "bytes": item.byte_count,
                "sha256": item.sha256,
            }
            for item in runner.OFFICIAL_SOURCE_FILES
        ],
        "pytorchvideo_repository": runner.PYTORCHVIDEO_REPOSITORY,
        "pytorchvideo_commit": runner.PYTORCHVIDEO_COMMIT,
        "pytorchvideo_tree": runner.PYTORCHVIDEO_TREE,
    }


def _asset_binding() -> dict[str, object]:
    files = [
        runner.OFFICIAL_ENCODER.public_dict(),
        runner.OFFICIAL_DECODER.public_dict(),
    ]
    return {
        "aggregate_sha256": runner.sha256_json({"files": files}),
        "files": files,
    }


def _write_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, object]]:
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
    sidecar = tmp_path / "inputs.json"
    sidecar.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    commitment = PoseInputCommitment(
        protocol=manifest.protocol,
        split=manifest.split,
        record_total=len(records),
        identity_sha256=pose_input_identity_sha256(records),
        sidecar_sha256=_sha256(sidecar.read_bytes()),
        sidecar_fingerprint=manifest.fingerprint,
    )
    commitment_path = tmp_path / "inputs.commitment.json"
    commitment_path.write_text(json.dumps(commitment.to_dict()), encoding="utf-8")
    binding: dict[str, object] = {
        "protocol": manifest.protocol,
        "split": manifest.split,
        "sample_count": len(records),
        "sidecar_sha256": _sha256(sidecar.read_bytes()),
        "commitment_sha256": _sha256(commitment_path.read_bytes()),
        "sidecar_fingerprint": manifest.fingerprint,
        "identity_sha256": pose_input_identity_sha256(records),
    }
    monkeypatch.setattr(
        score,
        "CANONICAL_DEV_SIDECAR_SHA256",
        binding["sidecar_sha256"],
    )
    monkeypatch.setattr(
        score,
        "CANONICAL_DEV_COMMITMENT_SHA256",
        binding["commitment_sha256"],
    )
    monkeypatch.setattr(
        score,
        "CANONICAL_DEV_SIDECAR_FINGERPRINT",
        binding["sidecar_fingerprint"],
    )
    monkeypatch.setattr(
        score,
        "CANONICAL_DEV_IDENTITY_SHA256",
        binding["identity_sha256"],
    )
    return sidecar, commitment_path, binding


def _merged_payload(
    counts: tuple[float, ...],
    *,
    video_ids: tuple[str, ...] | None = None,
    input_binding: dict[str, object] | None = None,
) -> dict[str, object]:
    ids = _dev_ids() if video_ids is None else video_ids
    rows = [
        {
            "index": index,
            "video_id": video_id,
            "video_locator": f"videos/{video_id}.avi",
            "expected_video_sha256": _sha256(video_id.encode()),
            "observed_video_sha256": _sha256(video_id.encode()),
            "raw_count": raw_count,
            "rounded_count": runner.round_count(raw_count),
            "finite": True,
            "reported_frame_count": 64,
            "decoded_frames": 64,
            "tail_shortfall": 0,
            "elapsed_seconds": 1.0,
            "resource_tier": runner.PRIMARY_RESOURCE_TIER,
            "memory_limit_bytes": runner.PRIMARY_MEMORY_LIMIT_BYTES,
            "status": "ok",
            "failure_reason": None,
        }
        for index, (video_id, raw_count) in enumerate(zip(ids, counts, strict=True))
    ]
    primary_sha = runner.sha256_json(rows)
    return {
        "schema_version": 1,
        "artifact_type": runner.MERGED_ARTIFACT_TYPE,
        "method_id": runner.METHOD_ID,
        "classification": runner.CLASSIFICATION,
        "eligible_for_original_pams_escounts_cell": False,
        "target_access": False,
        "status": "complete",
        "source": _source_binding(),
        "input": input_binding
        or {
            "protocol": "ucfrep_526",
            "split": "dev",
            "sample_count": 84,
            "sidecar_sha256": "1" * 64,
            "commitment_sha256": "2" * 64,
            "sidecar_fingerprint": "3" * 64,
            "identity_sha256": "4" * 64,
        },
        "selection": {
            "selected_count": 84,
            "selected_video_ids": list(ids),
            "selected_video_ids_sha256": runner.sha256_json(list(ids)),
        },
        "assets": _asset_binding(),
        "config": {
            "sha256": runner.FROZEN_ESCOUNTS_OFFICIAL_CONFIG.fingerprint,
            "values": runner.FROZEN_ESCOUNTS_OFFICIAL_CONFIG.to_dict(),
        },
        "runtime_versions": {"primary": {"runtime": "fixture"}, "retry": {}},
        "resource_tier_counts": {
            runner.PRIMARY_RESOURCE_TIER: 84,
            runner.RETRY_RESOURCE_TIER: 0,
        },
        "resource_limits_bytes": {
            runner.PRIMARY_RESOURCE_TIER: runner.PRIMARY_MEMORY_LIMIT_BYTES,
            runner.RETRY_RESOURCE_TIER: runner.RETRY_MEMORY_LIMIT_BYTES,
        },
        "source_primary_sha256": "5" * 64,
        "source_retry_request_sha256": "6" * 64,
        "source_retry_sha256": "7" * 64,
        "primary_success_rows_sha256": primary_sha,
        "merged_primary_rows_sha256": primary_sha,
        "primary_rows_canonical_value_identical": True,
        "record_total": 84,
        "success_total": 84,
        "failure_total": 0,
        "predictions": rows,
        "failures": [],
    }


def _write_prediction(
    tmp_path: Path,
    counts: tuple[float, ...],
    *,
    video_ids: tuple[str, ...] | None = None,
    input_binding: dict[str, object] | None = None,
) -> Path:
    repository_root = _audited_repository_root()
    repository = runner.verify_runner_repository(repository_root)
    merged_template = _merged_payload(
        counts,
        video_ids=video_ids,
        input_binding=input_binding,
    )
    ids = tuple(merged_template["selection"]["selected_video_ids"])  # type: ignore[index]
    rows = list(merged_template["predictions"])  # type: ignore[arg-type]
    runtime = {
        "python": "fixture",
        "torch": "fixture",
        "torchvision": "fixture",
        "cuda": "fixture",
        "numpy": "fixture",
        "opencv": "fixture",
        "av": "fixture",
        "pytorchvideo_commit": runner.PYTORCHVIDEO_COMMIT,
        "worker_code_sha256": repository.worker_digest.sha256,
        "worker_command_sha256": "6" * 64,
        "module_origins_sha256": "5" * 64,
        "pytorchvideo_origin_sha256": "4" * 64,
        "encoder_just_encode_unused_parameters_json": (
            runner.ENCODER_JUST_ENCODE_UNUSED_PARAMETERS_JSON
        ),
        "encoder_just_encode_unused_parameters_sha256": (
            runner.ENCODER_JUST_ENCODE_UNUSED_PARAMETERS_SHA256
        ),
        "container_image_id": f"sha256:{'9' * 64}",
    }
    final = dict(rows[-1])
    failure = {
        **final,
        "raw_count": None,
        "rounded_count": None,
        "finite": False,
        "reported_frame_count": 0,
        "decoded_frames": 0,
        "tail_shortfall": 0,
        "resource_tier": runner.PRIMARY_RESOURCE_TIER,
        "memory_limit_bytes": runner.PRIMARY_MEMORY_LIMIT_BYTES,
        "status": runner.PredictionStatus.RESOURCE_EXHAUSTED.value,
        "failure_reason": "fixture CUDA out of memory",
    }
    primary_payload = {
        "schema_version": 1,
        "artifact_type": runner.PREDICTION_ARTIFACT_TYPE,
        "method_id": runner.METHOD_ID,
        "classification": runner.CLASSIFICATION,
        "eligible_for_original_pams_escounts_cell": False,
        "target_access": False,
        "status": "partial",
        "source": merged_template["source"],
        "input": merged_template["input"],
        "selection": merged_template["selection"],
        "assets": merged_template["assets"],
        "config": merged_template["config"],
        "runtime_versions": runtime,
        "resource": {
            "tier": runner.PRIMARY_RESOURCE_TIER,
            "memory_limit_bytes": runner.PRIMARY_MEMORY_LIMIT_BYTES,
        },
        "retry_binding": None,
        **repository.provenance_dict(),
        "record_total": len(ids),
        "success_total": len(ids) - 1,
        "failure_total": 1,
        "predictions": rows[:-1],
        "failures": [failure],
    }
    primary = tmp_path / "primary.json"
    primary.write_text(json.dumps(primary_payload), encoding="utf-8")
    request = tmp_path / "retry.request.json"
    request_result = runner.create_retry_request(
        primary_predictions_path=primary,
        output_path=request,
        repository_root=repository_root,
    )
    request_payload = json.loads(request.read_text(encoding="utf-8"))
    retry_row = {
        **final,
        "resource_tier": runner.RETRY_RESOURCE_TIER,
        "memory_limit_bytes": runner.RETRY_MEMORY_LIMIT_BYTES,
    }
    retry_payload = {
        **primary_payload,
        "status": "complete",
        "selection": {
            "selected_count": 1,
            "selected_video_ids": [ids[-1]],
            "selected_video_ids_sha256": runner.sha256_json([ids[-1]]),
        },
        "resource": {
            "tier": runner.RETRY_RESOURCE_TIER,
            "memory_limit_bytes": runner.RETRY_MEMORY_LIMIT_BYTES,
        },
        "retry_binding": {
            "retry_request_sha256": request_result["retry_request_sha256"],
            "source_primary_sha256": request_result["source_primary_sha256"],
            "primary_success_rows_sha256": request_payload["primary_success_rows_sha256"],
            "records_sha256": request_payload["records_sha256"],
        },
        "record_total": 1,
        "success_total": 1,
        "failure_total": 0,
        "predictions": [retry_row],
        "failures": [],
    }
    retry = tmp_path / "retry.json"
    retry.write_text(json.dumps(retry_payload), encoding="utf-8")
    merged = tmp_path / "merged.json"
    receipt = tmp_path / "merge.receipt.json"
    runner.merge_retry_artifacts(
        primary_predictions_path=primary,
        retry_request_path=request,
        retry_predictions_path=retry,
        output_path=merged,
        receipt_path=receipt,
        repository_root=repository_root,
    )
    return merged


def _chain_arguments(prediction: Path) -> dict[str, Path]:
    return {
        "primary_predictions_path": prediction.parent / "primary.json",
        "retry_request_path": prediction.parent / "retry.request.json",
        "retry_predictions_path": prediction.parent / "retry.json",
        "merge_receipt_path": prediction.parent / "merge.receipt.json",
    }


def _write_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    counts: tuple[int, ...],
    *,
    reverse: bool = False,
) -> Path:
    records = tuple(
        DevTargetRecord(
            video_id=video_id,
            action=video_id.removeprefix("v_").split("_g", maxsplit=1)[0],
            count=count,
        )
        for video_id, count in zip(_dev_ids(), counts, strict=True)
    )
    if reverse:
        records = tuple(reversed(records))
    monkeypatch.setattr(
        data_module,
        "_UCFREP_526_DEV_ANNOTATION_SHA256",
        data_module._dev_target_annotation_sha256(records),
    )
    manifest = DevTargetManifest(
        protocol="ucfrep_526",
        records=records,
        source_annotation_sha256=(data_module._UCFREP_526_CANONICAL_ANNOTATION_SHA256),
    )
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    return path


def test_sealed_score_reports_rounded_raw_and_10k_paired_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audited_repository: Path,
) -> None:
    sidecar, commitment, binding = _write_inputs(tmp_path, monkeypatch)
    prediction = _write_prediction(
        tmp_path,
        (4.4,) * 84,
        input_binding=binding,
    )
    targets = _write_targets(tmp_path, monkeypatch, (5,) * 84)
    result = score.score_escounts_dev(
        **_chain_arguments(prediction),
        predictions_path=prediction,
        sidecar_path=sidecar,
        commitment_path=commitment,
        dev_targets_path=targets,
        output_dir=tmp_path / "score",
        repository_root=audited_repository,
    )

    rounded = result["rounded_metrics"]
    raw = result["raw_metrics"]
    assert rounded["nmae"] == pytest.approx(0.2)
    assert rounded["obo"] == 1.0
    assert rounded["mae"] == 1.0
    assert rounded["rmse"] == 1.0
    assert rounded["bootstrap_samples"] == 10_000
    assert raw["mae"] == pytest.approx(0.6)
    assert raw["rmse"] == pytest.approx(0.6)
    assert raw["bootstrap_samples"] == 10_000
    assert raw["bootstrap_pairing"] == "paired_prediction_target_rows"
    receipt = json.loads((tmp_path / "score" / score.RECEIPT_NAME).read_text(encoding="utf-8"))
    evaluation = json.loads(
        (tmp_path / "score" / score.EVALUATION_NAME).read_text(encoding="utf-8")
    )
    repository = runner.verify_runner_repository(audited_repository)
    assert receipt["prediction_sha256"] == _sha256(prediction.read_bytes())
    assert receipt["bootstrap_samples"] == 10_000
    for payload in (evaluation, receipt):
        assert payload["runner_source_git_sha"] == repository.git_sha
        assert payload["runner_code_sha256"] == repository.runner_digest.sha256
        assert payload["worker_code_sha256"] == repository.worker_digest.sha256

    with pytest.raises(FileExistsError, match="overwrite"):
        score.score_escounts_dev(
            **_chain_arguments(prediction),
            predictions_path=prediction,
            sidecar_path=sidecar,
            commitment_path=commitment,
            dev_targets_path=targets,
            output_dir=tmp_path / "score",
            repository_root=audited_repository,
        )


def test_score_receipt_failure_cleans_owned_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audited_repository: Path,
) -> None:
    sidecar, commitment, binding = _write_inputs(tmp_path, monkeypatch)
    prediction = _write_prediction(
        tmp_path,
        (4.4,) * 84,
        input_binding=binding,
    )
    targets = _write_targets(tmp_path, monkeypatch, (5,) * 84)
    original_writer = score._write_json_exclusive
    writes = 0

    def fail_second_write(path: Path, payload: dict[str, object]) -> object:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("fixture receipt failure")
        return original_writer(path, payload)

    monkeypatch.setattr(score, "_write_json_exclusive", fail_second_write)
    output = tmp_path / "score-transaction"
    with pytest.raises(OSError, match="receipt failure"):
        score.score_escounts_dev(
            **_chain_arguments(prediction),
            predictions_path=prediction,
            sidecar_path=sidecar,
            commitment_path=commitment,
            dev_targets_path=targets,
            output_dir=output,
            repository_root=audited_repository,
        )
    assert not (output / score.EVALUATION_NAME).exists()
    assert not (output / score.RECEIPT_NAME).exists()


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.__setitem__("label", 10),
        lambda payload: payload["source"].__setitem__("commit", "0" * 40),
        lambda payload: payload["assets"].__setitem__("aggregate_sha256", "0" * 64),
        lambda payload: payload.__setitem__("runner_source_git_sha", "0" * 40),
        lambda payload: payload.__setitem__("worker_code_sha256", "0" * 64),
    ),
    ids=("label", "source", "asset", "runner-git-sha", "worker-sha"),
)
def test_prediction_tamper_is_rejected_before_target_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audited_repository: Path,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    prediction = _write_prediction(tmp_path, (4.0,) * 84)
    payload = json.loads(prediction.read_text(encoding="utf-8"))
    mutate(payload)
    prediction.write_text(json.dumps(payload), encoding="utf-8")
    loads = 0

    def forbidden_targets(_path: object) -> None:
        nonlocal loads
        loads += 1
        raise AssertionError("labels must follow complete prediction validation")

    monkeypatch.setattr(score, "load_dev_target_manifest", forbidden_targets)
    with pytest.raises((ValueError, runner.ESCountsMergeError, runner.ESCountsSourceError)):
        score.score_escounts_dev(
            **_chain_arguments(prediction),
            predictions_path=prediction,
            sidecar_path=tmp_path / "inputs-never-opened.json",
            commitment_path=tmp_path / "commitment-never-opened.json",
            dev_targets_path=tmp_path / "targets-never-opened.json",
            output_dir=tmp_path / "score",
            repository_root=audited_repository,
        )
    assert loads == 0


def test_noncanonical_prediction_ids_are_rejected_before_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audited_repository: Path,
) -> None:
    ids = list(_dev_ids())
    ids[-1] = "v_StillRings_g01_c01"
    prediction = _write_prediction(
        tmp_path,
        (4.0,) * 84,
        video_ids=tuple(ids),
    )
    loads = 0

    def forbidden_targets(_path: object) -> None:
        nonlocal loads
        loads += 1
        raise AssertionError("labels must not load")

    monkeypatch.setattr(score, "load_dev_target_manifest", forbidden_targets)
    with pytest.raises(ValueError, match="frozen UCFRep dev split"):
        score.score_escounts_dev(
            **_chain_arguments(prediction),
            predictions_path=prediction,
            sidecar_path=tmp_path / "inputs-never-opened.json",
            commitment_path=tmp_path / "commitment-never-opened.json",
            dev_targets_path=tmp_path / "targets-never-opened.json",
            output_dir=tmp_path / "score",
            repository_root=audited_repository,
        )
    assert loads == 0


def test_cross_commit_lineage_is_rejected_before_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audited_repository: Path,
) -> None:
    prediction = _write_prediction(tmp_path, (4.0,) * 84)
    marker = audited_repository / "provenance-marker.txt"
    marker.write_text("advance clean repository commit\n", encoding="utf-8")
    _git(audited_repository, "add", "--", marker.name)
    _git(audited_repository, "commit", "-m", "advance scorer provenance")
    loads = 0

    def forbidden_targets(_path: object) -> None:
        nonlocal loads
        loads += 1
        raise AssertionError("labels must not load for cross-commit lineage")

    monkeypatch.setattr(score, "load_dev_target_manifest", forbidden_targets)
    with pytest.raises(runner.ESCountsSourceError, match="current clean"):
        score.score_escounts_dev(
            **_chain_arguments(prediction),
            predictions_path=prediction,
            sidecar_path=tmp_path / "inputs-never-opened.json",
            commitment_path=tmp_path / "commitment-never-opened.json",
            dev_targets_path=tmp_path / "targets-never-opened.json",
            output_dir=tmp_path / "score",
            repository_root=audited_repository,
        )
    assert loads == 0


@pytest.mark.parametrize("field", ("video_locator", "video_sha256"))
def test_actual_input_pair_binds_each_prediction_row_before_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audited_repository: Path,
    field: str,
) -> None:
    sidecar, commitment, binding = _write_inputs(tmp_path, monkeypatch)
    prediction = _write_prediction(
        tmp_path,
        (4.0,) * 84,
        input_binding=binding,
    )
    payload = json.loads(prediction.read_text(encoding="utf-8"))
    rows = payload["predictions"]
    assert isinstance(rows, list)
    first = rows[0]
    assert isinstance(first, dict)
    if field == "video_locator":
        first["video_locator"] = "videos/different.avi"
    else:
        first["expected_video_sha256"] = "0" * 64
        first["observed_video_sha256"] = "0" * 64
    primary_sha = runner.sha256_json(rows)
    payload["primary_success_rows_sha256"] = primary_sha
    payload["merged_primary_rows_sha256"] = primary_sha
    prediction.write_text(json.dumps(payload), encoding="utf-8")
    loads = 0

    def forbidden_targets(_path: object) -> None:
        nonlocal loads
        loads += 1
        raise AssertionError("labels must not load")

    monkeypatch.setattr(score, "load_dev_target_manifest", forbidden_targets)
    with pytest.raises(
        (ValueError, runner.ESCountsMergeError),
        match="lineage|receipt|integrity",
    ):
        score.score_escounts_dev(
            **_chain_arguments(prediction),
            predictions_path=prediction,
            sidecar_path=sidecar,
            commitment_path=commitment,
            dev_targets_path=tmp_path / "targets-never-opened.json",
            output_dir=tmp_path / "score",
            repository_root=audited_repository,
        )
    assert loads == 0


def test_self_signed_alternate_input_pair_is_rejected_before_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audited_repository: Path,
) -> None:
    sidecar, commitment, binding = _write_inputs(tmp_path, monkeypatch)
    prediction = _write_prediction(
        tmp_path,
        (4.0,) * 84,
        input_binding=binding,
    )
    monkeypatch.setattr(score, "CANONICAL_DEV_SIDECAR_SHA256", "0" * 64)
    loads = 0

    def forbidden_targets(_path: object) -> None:
        nonlocal loads
        loads += 1
        raise AssertionError("labels must not load")

    monkeypatch.setattr(score, "load_dev_target_manifest", forbidden_targets)
    with pytest.raises(ValueError, match="frozen dev input pair"):
        score.score_escounts_dev(
            **_chain_arguments(prediction),
            predictions_path=prediction,
            sidecar_path=sidecar,
            commitment_path=commitment,
            dev_targets_path=tmp_path / "targets-never-opened.json",
            output_dir=tmp_path / "score",
            repository_root=audited_repository,
        )
    assert loads == 0


def test_target_order_must_exactly_match_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audited_repository: Path,
) -> None:
    sidecar, commitment, binding = _write_inputs(tmp_path, monkeypatch)
    prediction = _write_prediction(
        tmp_path,
        (4.0,) * 84,
        input_binding=binding,
    )
    targets = _write_targets(tmp_path, monkeypatch, (5,) * 84, reverse=True)
    with pytest.raises(ValueError, match="IDs/order"):
        score.score_escounts_dev(
            **_chain_arguments(prediction),
            predictions_path=prediction,
            sidecar_path=sidecar,
            commitment_path=commitment,
            dev_targets_path=targets,
            output_dir=tmp_path / "score",
            repository_root=audited_repository,
        )
