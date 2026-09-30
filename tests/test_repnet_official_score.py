from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from pams import data as data_module
from pams.baselines import repnet_official_runner as runner
from pams.baselines import repnet_official_score as score
from pams.data import DevTargetManifest, DevTargetRecord

REPOSITORY = Path(__file__).parents[1]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _dev_ids() -> tuple[str, ...]:
    return tuple(
        (
            REPOSITORY / "data" / "splits" / "ucfrep_526_dev_84.txt"
        ).read_text(encoding="utf-8").splitlines()
    )


def _prediction_payload(
    video_ids: tuple[str, ...],
    rounded_counts: tuple[int, ...],
) -> dict[str, object]:
    checkpoint_files = [
        {
            "name": item.name,
            "bytes": item.byte_count,
            "published_md5": item.md5,
            "sha256": _sha256(item.name.encode("utf-8")),
        }
        for item in runner.OFFICIAL_CKPT70_FILES
    ]
    checkpoint_sha256 = score._sha256_json(
        {
            "checkpoint_prefix": runner.OFFICIAL_CHECKPOINT_PREFIX,
            "files": checkpoint_files,
        }
    )
    return {
        "schema_version": 1,
        "method_id": "repnet-official-current-ckpt70",
        "classification": (
            "official-current-ckpt70 / independent label-free UCFRep evaluation"
        ),
        "eligible_for_original_pams_repnet_cell": False,
        "source": {
            "repository": runner.OFFICIAL_SOURCE_REPOSITORY,
            "commit": runner.OFFICIAL_SOURCE_COMMIT,
            "notebook_path": runner.OFFICIAL_NOTEBOOK_PATH,
            "notebook_sha256": runner.OFFICIAL_NOTEBOOK_SHA256,
        },
        "input": {
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
            "selected_video_ids": list(video_ids),
            "selected_video_ids_sha256": score._sha256_json(list(video_ids)),
        },
        "checkpoint": {
            "prefix": "ckpt-70",
            "aggregate_sha256": checkpoint_sha256,
            "files": checkpoint_files,
        },
        "config": {
            "sha256": runner.FROZEN_REPNET_OFFICIAL_CONFIG.fingerprint,
            "values": runner.FROZEN_REPNET_OFFICIAL_CONFIG.to_dict(),
        },
        "runtime_versions": {
            "numpy": "1.26.4",
            "opencv": "4.10.0",
            "scipy": "1.14.1",
            "tensorflow": "2.17.1",
        },
        "predictions": [
            {
                "video_id": video_id,
                "video_locator": f"videos/{video_id}.avi",
                "expected_video_sha256": _sha256(video_id.encode("utf-8")),
                "observed_video_sha256": _sha256(video_id.encode("utf-8")),
                "raw_count": float(count),
                "rounded_count": count,
                "chosen_stride": 2,
                "confidence": 0.75,
                "decoded_frames": 64,
                "decode_status": "ok",
                "failure_reason": None,
            }
            for video_id, count in zip(video_ids, rounded_counts, strict=True)
        ],
    }


def _write_predictions(
    tmp_path: Path,
    counts: tuple[int, ...],
) -> tuple[Path, dict[str, object]]:
    payload = _prediction_payload(_dev_ids(), counts)
    path = tmp_path / "predictions.json"
    path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path, payload


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
    monkeypatch.setattr(
        data_module,
        "_UCFREP_526_DEV_ANNOTATION_SHA256",
        data_module._dev_target_annotation_sha256(records),
    )
    manifest = DevTargetManifest(
        protocol="ucfrep_526",
        records=tuple(reversed(records)) if reverse else records,
        source_annotation_sha256=(
            data_module._UCFREP_526_CANONICAL_ANNOTATION_SHA256
        ),
    )
    path = tmp_path / "dev.targets.json"
    path.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _clean_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(score, "_clean_git_revision", lambda _root: "a" * 40)


def test_score_computes_frozen_metrics_bootstrap_and_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_counts = (4,) * 84
    prediction_counts = (6, *((4,) * 83))
    predictions_path, _ = _write_predictions(tmp_path, prediction_counts)
    targets_path = _write_targets(tmp_path, monkeypatch, target_counts)
    _clean_git(monkeypatch)
    output = tmp_path / "score"

    summary = score.score_repnet_dev(
        predictions_path=predictions_path,
        dev_targets_path=targets_path,
        output_dir=output,
        repository_root=REPOSITORY,
    )

    evaluation_path = output / "evaluation.json"
    receipt_path = output / "evaluation.receipt.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    report = evaluation["report"]
    assert report["sample_count"] == 84
    assert report["nmae"] == pytest.approx(0.5 / 84)
    assert report["mae"] == pytest.approx(2 / 84)
    assert report["rmse"] == pytest.approx(math.sqrt(4 / 84))
    assert report["obo"] == pytest.approx(83 / 84)
    assert report["exact"] == pytest.approx(83 / 84)
    assert report["bootstrap_samples"] == 10_000
    assert report["bootstrap_seed"] == 2026
    assert set(report["confidence_intervals"]) == {
        "nmae",
        "mae",
        "rmse",
        "obo",
        "exact",
    }
    assert evaluation["bootstrap_pairing"] == "paired_prediction_target_rows"
    assert receipt["prediction_sha256"] == _sha256(predictions_path.read_bytes())
    assert receipt["dev_targets_sha256"] == _sha256(targets_path.read_bytes())
    assert receipt["evaluation_sha256"] == _sha256(evaluation_path.read_bytes())
    assert receipt["scoring_code_sha256"] == _sha256(
        (REPOSITORY / "src/pams/baselines/repnet_official_score.py").read_bytes()
    )
    assert receipt["scoring_source_git_sha"] == "a" * 40
    assert summary["evaluation_receipt_sha256"] == _sha256(receipt_path.read_bytes())

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        score.score_repnet_dev(
            predictions_path=predictions_path,
            dev_targets_path=targets_path,
            output_dir=output,
            repository_root=REPOSITORY,
        )


def test_forbidden_label_key_is_rejected_before_targets_are_loaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions_path, payload = _write_predictions(tmp_path, (4,) * 84)
    predictions = payload["predictions"]
    assert isinstance(predictions, list)
    predictions[0]["target"] = 999
    predictions_path.write_text(json.dumps(payload), encoding="utf-8")
    target_loads = 0

    def forbidden_target_loader(_path: object) -> None:
        nonlocal target_loads
        target_loads += 1
        raise AssertionError("target loader crossed the prediction firewall")

    monkeypatch.setattr(score, "load_dev_target_manifest", forbidden_target_loader)
    with pytest.raises(ValueError, match="forbidden label field 'target'"):
        score.score_repnet_dev(
            predictions_path=predictions_path,
            dev_targets_path=tmp_path / "must-not-open.targets.json",
            output_dir=tmp_path / "score",
            repository_root=REPOSITORY,
        )
    assert target_loads == 0


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload: payload.__setitem__("method_id", "repnet-counterfeit"),
            "method_id",
        ),
        (
            lambda payload: payload["selection"].__setitem__("selected_count", 83),
            "selected_count=84",
        ),
        (
            lambda payload: payload["predictions"][0].__setitem__(
                "decode_status", "decode_failed"
            ),
            "not decoded successfully",
        ),
        (
            lambda payload: payload["checkpoint"].__setitem__(
                "aggregate_sha256", "0" * 64
            ),
            "does not bind checkpoint",
        ),
    ],
)
def test_prediction_tampering_is_rejected_before_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator: object,
    message: str,
) -> None:
    predictions_path, payload = _write_predictions(tmp_path, (4,) * 84)
    assert callable(mutator)
    mutator(payload)
    predictions_path.write_text(json.dumps(payload), encoding="utf-8")
    target_loads = 0

    def forbidden_target_loader(_path: object) -> None:
        nonlocal target_loads
        target_loads += 1
        raise AssertionError("targets must remain unopened")

    monkeypatch.setattr(score, "load_dev_target_manifest", forbidden_target_loader)
    with pytest.raises(ValueError, match=message):
        score.score_repnet_dev(
            predictions_path=predictions_path,
            dev_targets_path=tmp_path / "must-not-open.targets.json",
            output_dir=tmp_path / "score",
            repository_root=REPOSITORY,
        )
    assert target_loads == 0


def test_prediction_change_after_validation_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions_path, _ = _write_predictions(tmp_path, (4,) * 84)
    targets_path = _write_targets(tmp_path, monkeypatch, (4,) * 84)
    _clean_git(monkeypatch)
    original_loader = score.load_dev_target_manifest

    def tampering_loader(path: str | Path) -> DevTargetManifest:
        loaded = original_loader(path)
        predictions_path.write_bytes(predictions_path.read_bytes() + b" ")
        return loaded

    monkeypatch.setattr(score, "load_dev_target_manifest", tampering_loader)
    with pytest.raises(
        score.RepNetOfficialScoreError,
        match="prediction artifact changed",
    ):
        score.score_repnet_dev(
            predictions_path=predictions_path,
            dev_targets_path=targets_path,
            output_dir=tmp_path / "score",
            repository_root=REPOSITORY,
        )
    assert not (tmp_path / "score" / "evaluation.json").exists()


def test_target_order_must_exactly_match_prediction_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions_path, _ = _write_predictions(tmp_path, (4,) * 84)
    targets_path = _write_targets(
        tmp_path,
        monkeypatch,
        (4,) * 84,
        reverse=True,
    )
    _clean_git(monkeypatch)

    with pytest.raises(ValueError, match="IDs/order"):
        score.score_repnet_dev(
            predictions_path=predictions_path,
            dev_targets_path=targets_path,
            output_dir=tmp_path / "score",
            repository_root=REPOSITORY,
        )


def test_cli_forwards_all_strict_score_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_score(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"metrics": {"nmae": 0.25}}

    monkeypatch.setattr(score, "score_repnet_dev", fake_score)
    exit_code = score.main(
        [
            "--predictions",
            str(tmp_path / "predictions.json"),
            "--dev-targets",
            str(tmp_path / "dev.targets.json"),
            "--output-dir",
            str(tmp_path / "score"),
            "--repository-root",
            str(REPOSITORY),
        ]
    )

    assert exit_code == 0
    assert captured == {
        "predictions_path": tmp_path / "predictions.json",
        "dev_targets_path": tmp_path / "dev.targets.json",
        "output_dir": tmp_path / "score",
        "repository_root": REPOSITORY,
    }
    assert json.loads(capsys.readouterr().out)["metrics"]["nmae"] == 0.25
