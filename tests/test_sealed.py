from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from inspect import signature
from pathlib import Path

import pytest
from pydantic import ValidationError

from pams.sealed import (
    SealedAttemptAlreadyReservedError,
    SealedTestAttemptReceipt,
    create_sealed_test_attempt_receipt,
    load_sealed_test_attempt_receipt,
    reserve_sealed_test_attempt,
    sealed_attempt_id,
)

DATASET_SHA = "a" * 64
CONFIG_SHA = "b" * 64
CHECKPOINT_SHA = "c" * 64
GIT_SHA = "d" * 40
TIMESTAMP = "2026-07-28T07:00:00+00:00"


def _reserve(
    registry: Path,
    *,
    seed: int = 2026,
    input_artifact_sha256: str = CHECKPOINT_SHA,
) -> tuple[SealedTestAttemptReceipt, Path]:
    return reserve_sealed_test_attempt(
        registry,
        protocol="ucfrep_526",
        full_dataset_sha256=DATASET_SHA,
        config_sha256=CONFIG_SHA,
        method_id="pams-sshead",
        experiment_seed=seed,
        input_artifact_role="checkpoint",
        input_artifact_sha256=input_artifact_sha256,
        git_sha=GIT_SHA,
        created_at_utc=TIMESTAMP,
    )


def test_receipt_is_frozen_strict_and_json_round_trips(tmp_path: Path) -> None:
    receipt, path = _reserve(tmp_path)

    assert path.name == f"{receipt.attempt_id}.json"
    assert receipt.status == "reserved"
    assert receipt.attempt_id == sealed_attempt_id(
        protocol="ucfrep_526",
        full_dataset_sha256=DATASET_SHA,
        method_id="pams-sshead",
        experiment_seed=2026,
    )
    assert load_sealed_test_attempt_receipt(path) == receipt
    assert json.loads(path.read_text(encoding="utf-8")) == receipt.model_dump(mode="json")
    with pytest.raises(ValidationError, match="frozen"):
        receipt.method_id = "changed"  # type: ignore[misc]

    payload = receipt.model_dump()
    payload["attempt_id"] = "e" * 64
    with pytest.raises(ValidationError, match="canonical sealed-attempt key"):
        SealedTestAttemptReceipt.model_validate(payload)
    payload = receipt.model_dump()
    payload["full_dataset_sha256"] = "A" * 64
    with pytest.raises(ValidationError, match="lowercase hexadecimal"):
        SealedTestAttemptReceipt.model_validate(payload)
    with pytest.raises(ValidationError, match="must use UTC"):
        create_sealed_test_attempt_receipt(
            protocol="ucfrep_526",
            full_dataset_sha256=DATASET_SHA,
            config_sha256=CONFIG_SHA,
            method_id="pams-sshead",
            experiment_seed=2026,
            input_artifact_role="checkpoint",
            input_artifact_sha256=CHECKPOINT_SHA,
            git_sha=GIT_SHA,
            created_at_utc="2026-07-28T15:00:00+08:00",
        )


def test_loader_rejects_non_strict_json(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON field"):
        load_sealed_test_attempt_receipt(duplicate)

    extra = tmp_path / "extra.json"
    receipt = create_sealed_test_attempt_receipt(
        protocol="ucfrep_526",
        full_dataset_sha256=DATASET_SHA,
        config_sha256=CONFIG_SHA,
        method_id="pams-sshead",
        experiment_seed=2026,
        input_artifact_role="checkpoint",
        input_artifact_sha256=CHECKPOINT_SHA,
        git_sha=GIT_SHA,
        created_at_utc=TIMESTAMP,
    )
    payload = receipt.model_dump(mode="json")
    payload["unexpected"] = True
    extra.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_sealed_test_attempt_receipt(extra)


def test_same_key_retry_and_changed_checkpoint_are_rejected(tmp_path: Path) -> None:
    assert set(signature(sealed_attempt_id).parameters) == {
        "protocol",
        "full_dataset_sha256",
        "method_id",
        "experiment_seed",
    }
    first, first_path = _reserve(tmp_path)

    with pytest.raises(SealedAttemptAlreadyReservedError, match="already reserved"):
        _reserve(tmp_path)
    with pytest.raises(SealedAttemptAlreadyReservedError, match="already reserved"):
        _reserve(tmp_path, input_artifact_sha256="e" * 64)

    assert first_path.is_file()
    assert load_sealed_test_attempt_receipt(first_path) == first
    assert len(tuple(tmp_path.glob("*.json"))) == 1


def test_different_seed_has_a_distinct_allowed_attempt(tmp_path: Path) -> None:
    first, first_path = _reserve(tmp_path, seed=2026)
    second, second_path = _reserve(tmp_path, seed=3407)

    assert first.attempt_id != second.attempt_id
    assert first_path != second_path
    assert len(tuple(tmp_path.glob("*.json"))) == 2


def test_concurrent_same_key_has_exactly_one_winner(tmp_path: Path) -> None:
    def attempt() -> str:
        try:
            receipt, _ = _reserve(tmp_path)
        except SealedAttemptAlreadyReservedError:
            return "rejected"
        return receipt.attempt_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda _index: attempt(), range(16)))

    winners = [outcome for outcome in outcomes if outcome != "rejected"]
    assert len(winners) == 1
    assert outcomes.count("rejected") == 15
    assert len(tuple(tmp_path.glob("*.json"))) == 1
    load_sealed_test_attempt_receipt(next(tmp_path.glob("*.json")))


def test_write_failure_removes_only_its_new_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fsync = __import__("os").fsync
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text('{"owner":"other"}\n', encoding="utf-8")

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr("pams.sealed.os.fsync", fail_fsync)
    with pytest.raises(OSError, match="injected fsync failure"):
        _reserve(tmp_path)
    assert tuple(tmp_path.glob("*.json")) == (unrelated,)
    assert unrelated.read_text(encoding="utf-8") == '{"owner":"other"}\n'

    monkeypatch.setattr("pams.sealed.os.fsync", original_fsync)
    receipt, path = _reserve(tmp_path)
    assert path.is_file()
    assert load_sealed_test_attempt_receipt(path) == receipt
    assert unrelated.read_text(encoding="utf-8") == '{"owner":"other"}\n'
