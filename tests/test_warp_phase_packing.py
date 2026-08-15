from __future__ import annotations

import hashlib
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest

from pams.warp_phase import packing
from pams.warp_phase.data import load_feature_shard
from pams.warp_phase.types import (
    EligibilityDecision,
    EvaluatorVaultRecord,
    FeatureShard,
    SourceVerificationReceipt,
)


def _synthetic_feature(*, key: str = "a" * 64, slot: int = 2) -> FeatureShard:
    motion = np.zeros((320, 17, 3), dtype="<f4")
    motion[:, :, 0] = np.arange(17, dtype=np.float32)[None, :] / np.float32(20.0)
    motion[:, :, 1] = np.float32(0.5)
    motion[:, :, 2] = np.float32(0.9)
    motion[:, 11, :2] = (0.0, 0.0)
    motion[:, 12, :2] = (2.0, 0.0)
    motion[:, 5, :2] = (0.0, 1.0)
    motion[:, 6, :2] = (2.0, 1.0)
    clocks = np.rint(np.linspace(0, 399, 320)).astype("<i8")
    return FeatureShard(
        motion=motion,
        person_mask=True,
        frame_mask=np.ones(320, dtype="|u1"),
        sampled_frame_indices=clocks,
        source_length=400,
        opaque_sample_key=key,
        local_person_slot=slot,
    )


def _canonical_source_path(root: Path, split: str) -> Path:
    return root / packing.CANONICAL_SOURCE_RELATIVE_PATHS[split]  # type: ignore[index]


def _eligible_decision(feature: FeatureShard) -> EligibilityDecision:
    return EligibilityDecision(
        opaque_sample_key=feature.opaque_sample_key,
        local_person_slot=feature.local_person_slot,
        association_ambiguous=False,
        association_one_to_one=True,
        pose_coverage=1.0,
        eligible=True,
        reasons=(),
        retained_distinct_clocks=320,
        valid_adjacent_cells=319,
        feature_frame_coverage=1.0,
        conflict_clocks=(),
        normalization_scale=1.0,
    )


def _source_receipt(tmp_path: Path) -> SourceVerificationReceipt:
    return SourceVerificationReceipt(
        split="train",
        path=tmp_path / "synthetic.pkl",
        relative_path=packing.CANONICAL_SOURCE_RELATIVE_PATHS["train"].as_posix(),
        sha256="b" * 64,
        byte_count=12,
    )


def _vault_record(feature: FeatureShard) -> EvaluatorVaultRecord:
    return EvaluatorVaultRecord(
        opaque_sample_key=feature.opaque_sample_key,
        local_person_slot=feature.local_person_slot,
        source_length=feature.source_length,
        count=2,
        periods=((10, 20), (30, 40)),
    )


def _passing_permission_receipt(layout: packing.PilotLayout) -> dict[str, object]:
    receipt = json.loads(json.dumps(packing.permission_separation_receipt(layout)))
    receipt["status"] = "PASS"
    receipt["blockers"] = []
    receipt["authorizes"] = ["identity_publication"]
    receipt["mechanism"] = "test_distinct_principal_acl"
    receipt["principals"] = {"training": "fixture-training", "evaluator": "fixture-evaluator"}
    receipt["checks"] = {name: True for name in receipt["checks"]}
    return receipt


def test_hash_precedes_trusted_unpickle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _canonical_source_path(tmp_path, "train")
    source.parent.mkdir(parents=True)
    payload = pickle.dumps({"synthetic": [1, 2, 3]}, protocol=4)
    source.write_bytes(payload)
    monkeypatch.setitem(
        packing.CANONICAL_SOURCE_SHA256,
        "train",
        hashlib.sha256(payload).hexdigest(),
    )
    hashed = False
    original_hash = packing._sha256_bytes
    original_loads = packing.pickle.loads

    def observed_hash(value: bytes) -> str:
        nonlocal hashed
        hashed = True
        return original_hash(value)

    def guarded_loads(value: bytes) -> object:
        assert hashed
        assert value == payload
        return original_loads(value)

    monkeypatch.setattr(packing, "_sha256_bytes", observed_hash)
    monkeypatch.setattr(packing.pickle, "loads", guarded_loads)
    decoded, receipt = packing.trusted_load_source_pickle(
        source,
        repository_root=tmp_path,
        split="train",
    )
    assert decoded == {"synthetic": [1, 2, 3]}
    assert receipt.sha256 == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize(
    "relative",
    [
        "data/counting_multirep_skeleton_pose_expanded_v44_len320/test.pkl",
        "data/sealed/train.pkl",
        "data/heldout/train.pkl",
        "results/train.pkl",
        "output/train.pkl",
    ],
)
def test_forbidden_source_paths_fail_before_file_io(tmp_path: Path, relative: str) -> None:
    with pytest.raises(packing.ForbiddenSourcePathError):
        packing.trusted_load_source_pickle(
            tmp_path / relative,
            repository_root=tmp_path,
            split="train",
        )


def test_only_train_and_val_source_splits_are_approved(tmp_path: Path) -> None:
    with pytest.raises(packing.UnapprovedSplitError):
        packing.verify_source_pickle(
            tmp_path / "data/source/test.pkl",
            repository_root=tmp_path,
            split="test",
        )


def test_feature_npz_is_deterministic_and_exactly_whitelisted(tmp_path: Path) -> None:
    first_layout = packing.create_pilot_layout(tmp_path / "first")
    second_layout = packing.create_pilot_layout(tmp_path / "second")
    feature = _synthetic_feature()
    first = packing.write_feature_shard(first_layout, "train", feature)
    second = packing.write_feature_shard(second_layout, "train", feature)
    assert first.path.read_bytes() == second.path.read_bytes()
    assert first.sha256 == second.sha256
    with zipfile.ZipFile(first.path) as archive:
        assert archive.namelist() == [f"{name}.npy" for name in packing.FEATURE_FIELD_NAMES]
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
    loaded = load_feature_shard(first.path)
    assert loaded.opaque_sample_key == feature.opaque_sample_key
    assert loaded.local_person_slot == feature.local_person_slot
    assert loaded.motion.dtype.str == "<f4"
    assert loaded.sampled_frame_indices.dtype.str == "<i8"


def test_feature_writer_rejects_privileged_or_extra_fields(tmp_path: Path) -> None:
    layout = packing.create_pilot_layout(tmp_path / "run")
    feature = _synthetic_feature()
    payload: dict[str, object] = {
        "motion": feature.motion,
        "person_mask": np.asarray([1], dtype="|u1"),
        "frame_mask": feature.frame_mask,
        "sampled_frame_indices": feature.sampled_frame_indices,
        "source_length": np.asarray([feature.source_length], dtype="<i8"),
        "opaque_sample_key": np.asarray(
            [feature.opaque_sample_key.encode("ascii")], dtype="|S64"
        ),
        "local_person_slot": np.asarray([feature.local_person_slot], dtype="<i8"),
        "count_gt": np.asarray([3], dtype="<i8"),
    }
    with pytest.raises(packing.FeatureSchemaError, match="extra"):
        packing.write_feature_shard(layout, "train", payload)


def test_schema_receipt_and_physical_roots_are_separate(tmp_path: Path) -> None:
    layout = packing.create_pilot_layout(tmp_path / "run")
    assert len({layout.features, layout.vault, layout.audit}) == 3
    assert (layout.features / "train").is_dir()
    assert (layout.features / "val").is_dir()
    receipt = packing.write_schema_fixture(layout)
    assert receipt.path.read_bytes() == packing.SCHEMA_FIXTURE.encode("utf-8")
    assert receipt.byte_count == 617
    assert receipt.sha256 == packing.SCHEMA_FIXTURE_SHA256
    assert receipt.path.is_relative_to(layout.audit)
    evaluator_view = packing.load_evaluator_schema_view(receipt.path.read_bytes())
    assert set(evaluator_view) == {"P_eval", "integrity", "interval_semantics"}
    assert "motion" not in evaluator_view


def test_excluded_slot_never_opens_vault_factory(tmp_path: Path) -> None:
    layout = packing.create_pilot_layout(tmp_path / "run")
    feature = _synthetic_feature()
    decision = EligibilityDecision(
        opaque_sample_key=feature.opaque_sample_key,
        local_person_slot=feature.local_person_slot,
        association_ambiguous=True,
        association_one_to_one=True,
        pose_coverage=1.0,
        eligible=False,
        reasons=("association_ambiguous",),
        retained_distinct_clocks=320,
        valid_adjacent_cells=319,
        feature_frame_coverage=1.0,
        conflict_clocks=(),
        normalization_scale=1.0,
    )
    source = _source_receipt(tmp_path)
    called = False

    def forbidden_vault_factory() -> object:
        nonlocal called
        called = True
        raise AssertionError("ineligible identities must not open labels")

    result = packing.pack_identity(
        layout,
        split="train",
        feature=feature,
        eligibility=decision,
        source=source,
        vault_factory=forbidden_vault_factory,  # type: ignore[arg-type]
    )
    assert not called
    assert result.feature is None
    assert result.vault is None
    assert result.audit.path.is_relative_to(layout.audit)


def test_permission_receipt_fails_closed_when_principal_boundary_is_unproven(
    tmp_path: Path,
) -> None:
    layout = packing.create_pilot_layout(tmp_path / "run")
    receipt = packing.permission_separation_receipt(layout)
    if receipt["status"] == "PASS":
        pytest.skip("test host has a pre-provisioned distinct-principal layout")

    with pytest.raises(packing.PhysicalSeparationError, match="BLOCKED") as caught:
        packing.verify_permission_separation(layout, receipt)

    assert caught.value.receipt == receipt
    assert receipt["authorizes"] == []
    assert receipt["blockers"]
    persisted = packing.write_permission_separation_receipt(layout)
    assert json.loads(persisted.path.read_text(encoding="utf-8")) == receipt


def test_unproven_permission_boundary_blocks_before_vault_label_access(
    tmp_path: Path,
) -> None:
    layout = packing.create_pilot_layout(tmp_path / "run")
    if packing.permission_separation_receipt(layout)["status"] == "PASS":
        pytest.skip("test host has a pre-provisioned distinct-principal layout")
    feature = _synthetic_feature()
    called = False

    def forbidden_vault_factory() -> EvaluatorVaultRecord:
        nonlocal called
        called = True
        return _vault_record(feature)

    with pytest.raises(packing.PhysicalSeparationError, match="BLOCKED"):
        packing.pack_identity(
            layout,
            split="train",
            feature=feature,
            eligibility=_eligible_decision(feature),
            source=_source_receipt(tmp_path),
            vault_factory=forbidden_vault_factory,
        )

    assert not called
    assert not tuple((layout.features / "train").glob("*.npz"))
    assert not tuple((layout.vault / "train").glob("*.json"))
    assert (layout.audit / "permissions" / "principal-separation.json").is_file()


def test_vault_failure_leaves_no_partial_identity_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = packing.create_pilot_layout(tmp_path / "run")
    passed = _passing_permission_receipt(layout)
    monkeypatch.setattr(packing, "permission_separation_receipt", lambda _layout: passed)
    feature = _synthetic_feature()

    def broken_vault_factory() -> EvaluatorVaultRecord:
        raise RuntimeError("synthetic vault failure")

    with pytest.raises(RuntimeError, match="synthetic vault failure"):
        packing.pack_identity(
            layout,
            split="train",
            feature=feature,
            eligibility=_eligible_decision(feature),
            source=_source_receipt(tmp_path),
            vault_factory=broken_vault_factory,
        )

    filename = f"{feature.opaque_sample_key}.{feature.local_person_slot}"
    assert not (layout.features / "train" / f"{filename}.npz").exists()
    assert not (layout.vault / "train" / f"{filename}.json").exists()
    assert not (
        layout.audit / "eligibility" / "train" / f"{filename}.eligibility.json"
    ).exists()


def test_complete_identity_is_staged_and_published_with_permission_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = packing.create_pilot_layout(tmp_path / "run")
    passed = _passing_permission_receipt(layout)
    monkeypatch.setattr(packing, "permission_separation_receipt", lambda _layout: passed)
    feature = _synthetic_feature()

    result = packing.pack_identity(
        layout,
        split="train",
        feature=feature,
        eligibility=_eligible_decision(feature),
        source=_source_receipt(tmp_path),
        vault_factory=lambda: _vault_record(feature),
    )

    assert result.feature is not None and result.feature.path.is_file()
    assert result.vault is not None and result.vault.path.is_file()
    assert result.audit.path.is_file()
    assert load_feature_shard(result.feature.path).opaque_sample_key == feature.opaque_sample_key
    audit = json.loads(result.audit.path.read_text(encoding="utf-8"))
    publication = audit["identity_publication"]
    assert publication["status"] == "PASS"
    assert publication["atomicity"]["commit_artifact_published_last"] is True
    assert publication["permission_separation"]["evidence"]["status"] == "PASS"


def test_partial_existing_identity_target_is_refused_before_vault_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = packing.create_pilot_layout(tmp_path / "run")
    passed = _passing_permission_receipt(layout)
    monkeypatch.setattr(packing, "permission_separation_receipt", lambda _layout: passed)
    feature = _synthetic_feature()
    target = (
        layout.features
        / "train"
        / f"{feature.opaque_sample_key}.{feature.local_person_slot}.npz"
    )
    target.write_bytes(b"pre-existing-partial")
    called = False

    def forbidden_vault_factory() -> EvaluatorVaultRecord:
        nonlocal called
        called = True
        return _vault_record(feature)

    with pytest.raises(FileExistsError, match="partial or existing"):
        packing.pack_identity(
            layout,
            split="train",
            feature=feature,
            eligibility=_eligible_decision(feature),
            source=_source_receipt(tmp_path),
            vault_factory=forbidden_vault_factory,
        )

    assert not called
    assert target.read_bytes() == b"pre-existing-partial"


def test_identity_publication_failure_rolls_back_transaction_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = packing.create_pilot_layout(tmp_path / "run")
    passed = _passing_permission_receipt(layout)
    monkeypatch.setattr(packing, "permission_separation_receipt", lambda _layout: passed)
    packing.write_permission_separation_receipt(layout)
    feature = _synthetic_feature()
    real_link = packing.os.link
    calls = 0

    def fail_third_link(
        source: Path,
        target: Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("synthetic publication failure")
        real_link(source, target, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(packing.os, "link", fail_third_link)
    with pytest.raises(OSError, match="synthetic publication failure"):
        packing.pack_identity(
            layout,
            split="train",
            feature=feature,
            eligibility=_eligible_decision(feature),
            source=_source_receipt(tmp_path),
            vault_factory=lambda: _vault_record(feature),
        )

    filename = f"{feature.opaque_sample_key}.{feature.local_person_slot}"
    identity_targets = (
        layout.features / "train" / f"{filename}.npz",
        layout.audit / "feature-shards" / "train" / f"{filename}.npz.receipt.json",
        layout.vault / "train" / f"{filename}.json",
        layout.audit / "vault-records" / "train" / f"{filename}.json.receipt.json",
        layout.audit / "eligibility" / "train" / f"{filename}.eligibility.json",
        layout.audit
        / "eligibility-receipts"
        / "train"
        / f"{filename}.eligibility.json.receipt.json",
    )
    assert all(not target.exists() for target in identity_targets)


def test_training_data_module_has_no_privileged_imports() -> None:
    source = Path("src/pams/warp_phase/data.py").read_text(encoding="utf-8")
    forbidden_imports = ("import pickle", "import evaluator", "import packing", "import vault")
    assert all(fragment not in source for fragment in forbidden_imports)
