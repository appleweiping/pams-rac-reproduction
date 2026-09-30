"""Sealed scorer for a complete official ESCounts UCFRep-dev artifact.

The label-free merged prediction ledger is hashed and fully validated before
the target manifest is opened.  This module is intentionally a separate
process boundary from :mod:`pams.baselines.escounts_official_runner`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pams.baselines import escounts_official_runner as runner_module
from pams.baselines.escounts_official_runner import (
    METHOD_ID,
    OFFICIAL_DECODER,
    OFFICIAL_ENCODER,
    OFFICIAL_SOURCE_COMMIT,
    validate_complete_merge_chain,
)
from pams.data import (
    load_dev_target_manifest,
    load_pose_input_commitment,
    load_pose_input_manifest,
    pose_input_identity_sha256,
    validate_pose_input_binding,
)
from pams.metrics import compute_count_metrics

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 2026
CONFIDENCE_LEVEL = 0.95
CANONICAL_DEV_ID_SHA256 = "2199294a1d22da6c67d2fdaaafb4bebaa11e92a5bb3accd4670975815001f2c6"
CANONICAL_DEV_SIDECAR_SHA256 = "f21c49569805e4aea326977f6d7ce2964f503fad129a17e95214594364ca4ae7"
CANONICAL_DEV_COMMITMENT_SHA256 = "a2bb2db86abef64e16a8fe1ac196857ac8e77cc9cd461c568b2193401a8cd169"
CANONICAL_DEV_SIDECAR_FINGERPRINT = (
    "d1349cff6913345899db288c11aa991419cdd2572a124f3dda6aa2aba35fb81d"
)
CANONICAL_DEV_IDENTITY_SHA256 = "bdf944d2aa13e22c07383163794e8edc3198f0c8fd8355b6a6d0255566b3cb5b"
EVALUATION_NAME = "evaluation.json"
RECEIPT_NAME = "evaluation.receipt.json"
_SCORER_PATH = Path(__file__).resolve(strict=True)


class ESCountsOfficialScoreError(RuntimeError):
    """A sealed-boundary or artifact-integrity failure."""


@dataclass(frozen=True, slots=True)
class _Digest:
    sha256: str
    byte_count: int
    device: int
    inode: int
    modified_ns: int


def _stable_digest(path: Path) -> _Digest:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ESCountsOfficialScoreError(
            f"required input must be a regular non-symlink file: {path}"
        )
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
        closed = os.fstat(handle.fileno())
    after = path.lstat()
    identities = {
        (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
        for value in (before, opened, closed, after)
    }
    if len(identities) != 1 or byte_count != before.st_size:
        raise ESCountsOfficialScoreError(f"file changed while hashed: {path}")
    return _Digest(
        sha256=digest.hexdigest(),
        byte_count=byte_count,
        device=before.st_dev,
        inode=before.st_ino,
        modified_ns=before.st_mtime_ns,
    )


def _assert_unchanged(path: Path, expected: _Digest, *, role: str) -> None:
    if _stable_digest(path) != expected:
        raise ESCountsOfficialScoreError(f"{role} changed during sealed scoring")


def _canonical_dev_id_sha256(video_ids: Sequence[str]) -> str:
    encoded = ("\n".join(sorted(video_ids)) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _raw_error_report(
    predictions: Sequence[float],
    targets: Sequence[int],
) -> dict[str, Any]:
    prediction = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if (
        prediction.shape != target.shape
        or prediction.ndim != 1
        or prediction.size < 1
        or not np.isfinite(prediction).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("raw prediction/target vectors are invalid")
    absolute = np.abs(prediction - target)
    squared = np.square(prediction - target)
    mae_samples = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    rmse_samples = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for start in range(0, BOOTSTRAP_SAMPLES, 512):
        stop = min(start + 512, BOOTSTRAP_SAMPLES)
        indices = rng.integers(
            0,
            prediction.size,
            size=(stop - start, prediction.size),
        )
        mae_samples[start:stop] = absolute[indices].mean(axis=1)
        rmse_samples[start:stop] = np.sqrt(squared[indices].mean(axis=1))

    def interval(values: NDArray[np.float64]) -> dict[str, float]:
        low, high = np.quantile(values, [0.025, 0.975])
        return {
            "low": float(low),
            "high": float(high),
            "level": CONFIDENCE_LEVEL,
        }

    return {
        "mae": float(absolute.mean()),
        "rmse": float(math.sqrt(float(squared.mean()))),
        "confidence_intervals": {
            "mae": interval(mae_samples),
            "rmse": interval(rmse_samples),
        },
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_pairing": "paired_prediction_target_rows",
    }


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> _Digest:
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    created = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        runner_module.fsync_directory(path.parent)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        _remove_owned_score_artifact(path, created)
        raise
    return _stable_digest(path)


def _remove_owned_score_artifact(
    path: Path,
    created: os.stat_result | _Digest,
) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return
    if isinstance(created, _Digest):
        expected = (created.device, created.inode)
    else:
        expected = (created.st_dev, created.st_ino)
    actual = (observed.st_dev, observed.st_ino)
    if actual != expected:
        raise ESCountsOfficialScoreError(
            f"refusing to remove replaced score transaction file: {path}"
        )
    path.unlink()
    runner_module.fsync_directory(path.parent)


def score_escounts_dev(
    *,
    primary_predictions_path: str | Path,
    retry_request_path: str | Path,
    retry_predictions_path: str | Path,
    predictions_path: str | Path,
    merge_receipt_path: str | Path,
    sidecar_path: str | Path,
    commitment_path: str | Path,
    dev_targets_path: str | Path,
    output_dir: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Score exactly 84 sealed predictions with 10,000 paired bootstraps."""

    prediction_path = Path(predictions_path).expanduser().resolve(strict=True)
    sidecar_source = Path(sidecar_path).expanduser()
    commitment_source = Path(commitment_path).expanduser()
    targets_path = Path(dev_targets_path).expanduser().resolve(strict=False)
    destination = Path(output_dir).expanduser().resolve(strict=False)
    repository = Path(repository_root).expanduser().resolve(strict=True)
    destination.mkdir(parents=True, exist_ok=True)
    evaluation_path = destination / EVALUATION_NAME
    receipt_path = destination / RECEIPT_NAME
    collisions = [str(path) for path in (evaluation_path, receipt_path) if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite score artifacts: {collisions}")

    # This complete validation, including raw forbidden-label-key scanning,
    # occurs before the target path is opened.
    chain = validate_complete_merge_chain(
        primary_predictions_path=primary_predictions_path,
        retry_request_path=retry_request_path,
        retry_predictions_path=retry_predictions_path,
        merged_predictions_path=prediction_path,
        merge_receipt_path=merge_receipt_path,
        repository_root=repository,
    )
    predictions = chain.merged
    prediction_digest = _stable_digest(prediction_path)
    if prediction_digest.sha256 != predictions.digest.sha256:
        raise ESCountsOfficialScoreError("prediction changed after runner validation")
    if _canonical_dev_id_sha256(predictions.video_ids) != CANONICAL_DEV_ID_SHA256:
        raise ValueError("prediction identities are not the frozen UCFRep dev split")

    sidecar = sidecar_source.resolve(strict=True)
    commitment_file = commitment_source.resolve(strict=True)
    sidecar_digest = _stable_digest(sidecar)
    commitment_digest = _stable_digest(commitment_file)
    for path, document_name in (
        (sidecar, "label-free input sidecar"),
        (commitment_file, "label-free input commitment"),
    ):
        runner_module._reject_forbidden_keys(
            path.read_text(encoding="utf-8"),
            document_name=document_name,
        )
    manifest = load_pose_input_manifest(sidecar, validate_exact=True)
    commitment = load_pose_input_commitment(commitment_file)
    validate_pose_input_binding(
        manifest,
        commitment,
        sidecar_sha256=sidecar_digest.sha256,
    )
    identity_sha256 = pose_input_identity_sha256(manifest.records)
    if (
        sidecar_digest.sha256 != CANONICAL_DEV_SIDECAR_SHA256
        or commitment_digest.sha256 != CANONICAL_DEV_COMMITMENT_SHA256
        or manifest.fingerprint != CANONICAL_DEV_SIDECAR_FINGERPRINT
        or identity_sha256 != CANONICAL_DEV_IDENTITY_SHA256
    ):
        raise ValueError("actual sidecar/commitment is not the frozen dev input pair")
    actual_binding = {
        "protocol": manifest.protocol,
        "split": manifest.split,
        "sample_count": len(manifest.records),
        "sidecar_sha256": sidecar_digest.sha256,
        "commitment_sha256": commitment_digest.sha256,
        "sidecar_fingerprint": manifest.fingerprint,
        "identity_sha256": identity_sha256,
    }
    if predictions.payload["input"] != actual_binding:
        raise ValueError("prediction input binding does not match actual sidecar/commitment")
    if tuple(record.video_id for record in manifest.records) != predictions.video_ids:
        raise ValueError("prediction IDs/order do not match actual input sidecar")
    for index, (record, row) in enumerate(zip(manifest.records, predictions.rows, strict=True)):
        if (
            row["video_locator"] != record.video_path
            or row["expected_video_sha256"] != record.video_sha256
            or row["observed_video_sha256"] != record.video_sha256
        ):
            raise ValueError(
                f"prediction row {index} locator/video SHA differs from actual sidecar"
            )
    _assert_unchanged(sidecar, sidecar_digest, role="label-free input sidecar")
    _assert_unchanged(
        commitment_file,
        commitment_digest,
        role="label-free input commitment",
    )

    verified_repository = runner_module.verify_runner_repository(repository)
    tracked_scorer_digest = verified_repository.verify_tracked_file(
        _SCORER_PATH,
        role="ESCounts sealed scorer",
    )
    scorer_path = _SCORER_PATH
    scorer_digest = _stable_digest(scorer_path)
    if (
        scorer_digest.sha256 != tracked_scorer_digest.sha256
        or scorer_digest.byte_count != tracked_scorer_digest.byte_count
    ):
        raise ESCountsOfficialScoreError("scoring code differs from the audited checkout")
    provenance = {
        "git_sha": verified_repository.git_sha,
        "dirty": False,
    }

    # This is the first label-bearing operation.
    target_digest = _stable_digest(targets_path)
    targets = load_dev_target_manifest(targets_path)
    target_ids = tuple(record.video_id for record in targets.records)
    if target_ids != predictions.video_ids:
        raise ValueError("target IDs/order do not exactly match sealed predictions")
    target_values = [record.count for record in targets.records]
    actions = [record.action for record in targets.records]

    rounded_report = compute_count_metrics(
        predictions.raw_counts,
        target_values,
        video_ids=predictions.video_ids,
        actions=actions,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
        confidence_level=CONFIDENCE_LEVEL,
    ).to_dict()
    raw_report = _raw_error_report(predictions.raw_counts, target_values)

    _assert_unchanged(prediction_path, prediction_digest, role="prediction artifact")
    _assert_unchanged(sidecar, sidecar_digest, role="label-free input sidecar")
    _assert_unchanged(
        commitment_file,
        commitment_digest,
        role="label-free input commitment",
    )
    _assert_unchanged(targets_path, target_digest, role="dev target manifest")
    _assert_unchanged(scorer_path, scorer_digest, role="scoring code")
    revalidated_chain = validate_complete_merge_chain(
        primary_predictions_path=primary_predictions_path,
        retry_request_path=retry_request_path,
        retry_predictions_path=retry_predictions_path,
        merged_predictions_path=prediction_path,
        merge_receipt_path=merge_receipt_path,
        repository_root=repository,
    )
    if (
        revalidated_chain.primary.digest != chain.primary.digest
        or revalidated_chain.request.digest != chain.request.digest
        or revalidated_chain.retry.digest != chain.retry.digest
        or revalidated_chain.merged.digest != chain.merged.digest
        or revalidated_chain.receipt_digest != chain.receipt_digest
    ):
        raise ESCountsOfficialScoreError("prediction lineage changed during scoring")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "escounts_official_dev_evaluation",
        "classification": (
            "official-f18fcf1 RepCount-checkpoint zero-shot / "
            "independent sealed UCFRep dev evaluation"
        ),
        "eligible_for_original_pams_escounts_cell": False,
        "protocol": "ucfrep_526",
        "split": "dev",
        "method_id": METHOD_ID,
        "prediction_sha256": prediction_digest.sha256,
        "primary_prediction_sha256": chain.primary.digest.sha256,
        "retry_request_sha256": chain.request.digest.sha256,
        "retry_prediction_sha256": chain.retry.digest.sha256,
        "merge_receipt_sha256": chain.receipt_digest.sha256,
        "dev_targets_sha256": target_digest.sha256,
        "input_binding": actual_binding,
        "source_binding": predictions.payload["source"],
        "asset_binding": predictions.payload["assets"],
        "config_binding": predictions.payload["config"],
        "runner_source_git_sha": verified_repository.git_sha,
        "runner_code_sha256": verified_repository.runner_digest.sha256,
        "worker_code_sha256": verified_repository.worker_digest.sha256,
        "scoring_code_sha256": scorer_digest.sha256,
        "scoring_repository": provenance,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_pairing": "paired_prediction_target_rows",
        "rounded_metrics": rounded_report,
        "raw_metrics": raw_report,
    }
    evaluation_digest = _write_json_exclusive(evaluation_path, payload)
    receipt = {
        "schema_version": 1,
        "artifact_type": "escounts_official_dev_evaluation_receipt",
        "method_id": METHOD_ID,
        "evaluation_file": EVALUATION_NAME,
        "evaluation_sha256": evaluation_digest.sha256,
        "evaluation_bytes": evaluation_digest.byte_count,
        "prediction_file": prediction_path.name,
        "prediction_sha256": prediction_digest.sha256,
        "prediction_bytes": prediction_digest.byte_count,
        "primary_prediction_sha256": chain.primary.digest.sha256,
        "retry_request_sha256": chain.request.digest.sha256,
        "retry_prediction_sha256": chain.retry.digest.sha256,
        "merge_receipt_sha256": chain.receipt_digest.sha256,
        "runner_source_git_sha": verified_repository.git_sha,
        "runner_code_sha256": verified_repository.runner_digest.sha256,
        "worker_code_sha256": verified_repository.worker_digest.sha256,
        "input_sidecar_file": sidecar.name,
        "input_sidecar_sha256": sidecar_digest.sha256,
        "input_sidecar_bytes": sidecar_digest.byte_count,
        "input_commitment_file": commitment_file.name,
        "input_commitment_sha256": commitment_digest.sha256,
        "input_commitment_bytes": commitment_digest.byte_count,
        "input_identity_sha256": identity_sha256,
        "dev_targets_file": targets_path.name,
        "dev_targets_sha256": target_digest.sha256,
        "dev_targets_bytes": target_digest.byte_count,
        "scoring_code_sha256": scorer_digest.sha256,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "encoder_sha256": OFFICIAL_ENCODER.sha256,
        "decoder_sha256": OFFICIAL_DECODER.sha256,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    receipt_digest: _Digest | None = None
    try:
        final_chain = validate_complete_merge_chain(
            primary_predictions_path=primary_predictions_path,
            retry_request_path=retry_request_path,
            retry_predictions_path=retry_predictions_path,
            merged_predictions_path=prediction_path,
            merge_receipt_path=merge_receipt_path,
            repository_root=repository,
        )
        if (
            final_chain.primary.digest != chain.primary.digest
            or final_chain.request.digest != chain.request.digest
            or final_chain.retry.digest != chain.retry.digest
            or final_chain.merged.digest != chain.merged.digest
            or final_chain.receipt_digest != chain.receipt_digest
        ):
            raise ESCountsOfficialScoreError(
                "prediction lineage changed before score receipt publication"
            )
        _assert_unchanged(prediction_path, prediction_digest, role="prediction artifact")
        _assert_unchanged(sidecar, sidecar_digest, role="label-free input sidecar")
        _assert_unchanged(
            commitment_file,
            commitment_digest,
            role="label-free input commitment",
        )
        _assert_unchanged(targets_path, target_digest, role="dev target manifest")
        _assert_unchanged(scorer_path, scorer_digest, role="scoring code")
        verified_repository.assert_unchanged()
        receipt_digest = _write_json_exclusive(receipt_path, receipt)
        _assert_unchanged(scorer_path, scorer_digest, role="scoring code")
        verified_repository.assert_unchanged()
    except Exception:
        if receipt_digest is not None:
            _remove_owned_score_artifact(receipt_path, receipt_digest)
        _remove_owned_score_artifact(
            evaluation_path,
            evaluation_digest,
        )
        raise
    assert receipt_digest is not None
    compact = dict(rounded_report)
    compact.pop("per_video")
    return {
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": evaluation_digest.sha256,
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_digest.sha256,
        "rounded_metrics": compact,
        "raw_metrics": raw_report,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pams.baselines.escounts_official_score")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--primary-predictions", type=Path, required=True)
    parser.add_argument("--retry-request", type=Path, required=True)
    parser.add_argument("--retry-predictions", type=Path, required=True)
    parser.add_argument("--merge-receipt", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--commitment", type=Path, required=True)
    parser.add_argument("--dev-targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        result = score_escounts_dev(
            primary_predictions_path=arguments.primary_predictions,
            retry_request_path=arguments.retry_request,
            retry_predictions_path=arguments.retry_predictions,
            predictions_path=arguments.predictions,
            merge_receipt_path=arguments.merge_receipt,
            sidecar_path=arguments.sidecar,
            commitment_path=arguments.commitment,
            dev_targets_path=arguments.dev_targets,
            output_dir=arguments.output_dir,
            repository_root=arguments.repository_root,
        )
    except (
        OSError,
        ValueError,
        ESCountsOfficialScoreError,
        runner_module.ESCountsOfficialError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
