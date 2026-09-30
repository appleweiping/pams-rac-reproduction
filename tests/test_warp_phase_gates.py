from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pams.warp_phase import gates, packing, selector
from pams.warp_phase.gates import (
    K4_VECTOR_SHA256,
    PCG64_WITNESS_SHA256,
    SCHEMA_FIXTURE,
    SCHEMA_FIXTURE_SHA256,
    GateError,
    canonical_json_bytes,
    create_output_space_receipt,
    create_training_environment_lock,
    gate0_pack_receipt,
    gate1_fixture_receipt,
    gate2_selector_receipt,
    gate3_cpu_sanity_receipt,
    generate_k4_vectors,
    pcg64_witness,
    sha256_bytes,
    static_gate_receipt,
    validate_gate2_counts,
    verify_output_space_receipt,
    verify_schema_fixture,
    verify_training_environment_lock,
)
from pams.warp_phase.selector import SupportDiagnostics, SupportSelection


def test_frozen_schema_pcg64_and_k4_bytes() -> None:
    assert len(SCHEMA_FIXTURE.encode("utf-8")) == 617
    assert verify_schema_fixture() == SCHEMA_FIXTURE_SHA256
    assert pcg64_witness() == PCG64_WITNESS_SHA256
    vectors = generate_k4_vectors()
    assert vectors.shape == (4, 128)
    assert vectors.dtype == np.dtype("<f8")
    assert sha256_bytes(vectors.tobytes(order="C")) == K4_VECTOR_SHA256


def test_static_receipt_is_non_authorizing_canonical_json() -> None:
    receipt = static_gate_receipt()
    assert receipt.status == "PASS_NON_AUTHORIZING"
    assert receipt.authorizes == ()
    payload = canonical_json_bytes(dataclasses.asdict(receipt))
    assert not payload.endswith(b"\n")


def test_gate2_thresholds_are_inclusive() -> None:
    canonical = np.full(1000, 20, dtype=np.int16)
    weak = canonical.copy()
    weak[:50] = -1
    weak[50] = 22
    validate_gate2_counts(canonical, weak, 20, 20, 12, 12)
    weak[50] = 23
    with pytest.raises(GateError, match="below 95 percent") as failure:
        validate_gate2_counts(canonical, weak, 20, 20, 12, 12)
    assert "agreement=949/1000" in str(failure.value)
    assert "half_overall=20" in str(failure.value)
    assert "double_symmetric=12/250" in str(failure.value)
    with pytest.raises(GateError):
        validate_gate2_counts(canonical, canonical, 21, 20, 12, 12)


def test_selector_unit_schema_is_complete_rng_free_and_separate() -> None:
    assert len(gates._SELECTOR_UNIT_MEMBER_SCHEMA) == 64
    assert len(gates._SELECTOR_UNIT_PACK_FILENAMES) == 65
    assert len(gates._SELECTOR_UNIT_ROOT_FILENAMES) == 66
    assert gates.SELECTOR_UNIT_FIXTURE_RECEIPT not in gates._SELECTOR_UNIT_PACK_FILENAMES
    assert set(gates._FIXTURE_MEMBER_SCHEMA).isdisjoint(
        name
        for name in gates._SELECTOR_UNIT_MEMBER_SCHEMA
        if name != "candidate_period.npy"
    )
    generator_source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "generate_warp_phase_selector_unit_fixtures.py"
    ).read_text(encoding="utf-8")
    assert "np.random" not in generator_source
    assert "os.O_EXCL" in generator_source
    assert '"candidate_only": True' in generator_source


def test_gate2_receipt_preserves_pack_integrity_and_all_threshold_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = np.full(1000, 20, dtype=np.int16)
    weak = np.full(1000, -1, dtype=np.int16)
    weak[:246] = 20
    semantic = np.full(1000, 20.0, dtype=np.float64)
    half_indices = np.concatenate((np.arange(250, 300), np.arange(0, 5)))
    double_indices = np.concatenate((np.arange(300, 366), np.arange(5, 83)))
    semantic[half_indices] = 40.0
    semantic[double_indices] = 10.0
    arrays = {
        "expected_selected_period.npy": canonical,
        "expected_weak_selected_period.npy": weak,
        "semantic_period.npy": semantic,
    }
    monkeypatch.setattr(
        gates,
        "verify_fixture_pack",
        lambda *_args: (arrays, "a" * 64, "b" * 64),
    )

    receipt = gate2_selector_receipt(
        fixture_root=Path("stochastic-pack"),
        pack_receipt_path=Path("stochastic-pack.receipt.json"),
        features_root=None,
        order_receipt_path=None,
    )

    assert receipt.status == "FAIL"
    assert receipt.checks["immutable_1000_fixture_pack"]
    assert not receipt.checks["fixture_thresholds"]
    assert not receipt.checks["fixture_agreement_at_least_950"]
    assert not receipt.checks["fixture_half_overall_at_most_20"]
    assert not receipt.checks["fixture_double_overall_at_most_20"]
    assert not receipt.checks["fixture_half_symmetric_at_most_12"]
    assert not receipt.checks["fixture_double_symmetric_at_most_12"]
    assert receipt.bindings["fixture_agreement_count"] == "246"
    assert receipt.bindings["fixture_half_overall_count"] == "55"
    assert receipt.bindings["fixture_double_overall_count"] == "144"
    assert receipt.bindings["fixture_half_symmetric_count"] == "50"
    assert receipt.bindings["fixture_double_symmetric_count"] == "66"
    assert sum(blocker.startswith("fixture_threshold_failed:") for blocker in receipt.blockers) == 5
    assert not any("selector_unit" in key for key in receipt.checks | receipt.bindings)


def test_ten_case_recomputation_matches_failure_sentinel_and_full_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = np.asarray((4, 5), dtype=np.int64)
    diagnostics = SupportDiagnostics(
        candidates=candidates,
        acf=np.asarray((0.3, np.nan), dtype=np.float64),
        score_by_period=np.asarray((0.4, np.nan), dtype=np.float64),
        score_valid=np.asarray((True, False), dtype=np.bool_),
        local_maximum=np.asarray((False, False), dtype=np.bool_),
        fft_power=np.arange(1, 128, dtype=np.float64),
        harmonic_power=np.asarray(((1.0, 2.0, 3.0), (0.0, 0.0, 0.0))),
        fft_grid=np.arange(256, dtype=np.float64),
        fft_mask=np.zeros((256, 34), dtype=np.bool_),
    )
    failed = SupportSelection(period=None, score=None, diagnostics=diagnostics)
    monkeypatch.setattr(
        selector,
        "normalize_coco17",
        lambda *_args: SimpleNamespace(
            clocks=np.arange(128, dtype=np.int64),
            coordinates=np.zeros((128, 17, 2), dtype=np.float64),
            joint_valid=np.zeros((128, 17), dtype=np.bool_),
        ),
    )
    monkeypatch.setattr(selector, "select_support_period", lambda *_args: failed)

    arrays: dict[str, np.ndarray] = {
        "q.npy": np.arange(128, dtype=np.float64),
        "pose.npy": np.zeros(1000, dtype=np.float64),
        "joint_mask.npy": np.zeros(1000, dtype=np.uint8),
        "weak_pose.npy": np.zeros(1000, dtype=np.float64),
        "weak_joint_mask.npy": np.zeros(1000, dtype=np.uint8),
        "expected_selected_period.npy": np.full(1000, -1, dtype=np.int16),
        "expected_weak_selected_period.npy": np.full(1000, -1, dtype=np.int16),
        "expected_first64_selected_period.npy": np.full(1000, -1, dtype=np.int16),
        "expected_last64_selected_period.npy": np.full(1000, -1, dtype=np.int16),
        "expected_reversal_selected_period.npy": np.full(1000, -1, dtype=np.int16),
        "expected_fft_x.npy": np.tile(diagnostics.fft_grid, (1000, 1)),
        "expected_fft_mask.npy": np.zeros((1000, 256, 34), dtype=np.uint8),
        "expected_fft_power.npy": np.tile(diagnostics.fft_power, (1000, 1)),
        "expected_acf.npy": np.zeros((1000, 125), dtype=np.float64),
        "expected_score.npy": np.zeros((1000, 125), dtype=np.float64),
        "expected_score_valid.npy": np.zeros((1000, 125), dtype=np.uint8),
        "expected_local_max.npy": np.zeros((1000, 125), dtype=np.uint8),
        "expected_harmonic_power.npy": np.zeros((1000, 125, 3), dtype=np.float64),
    }
    arrays["expected_acf.npy"][:, 0] = 0.3
    arrays["expected_score.npy"][:, 0] = 0.4
    arrays["expected_score_valid.npy"][:, 0] = 1
    arrays["expected_harmonic_power.npy"][:, 0] = (1.0, 2.0, 3.0)

    gates._verify_ten_case_recomputation(arrays)
    arrays["expected_harmonic_power.npy"][0, 0, 0] = 99.0
    with pytest.raises(GateError, match="fixture 0 selector diagnostics differ"):
        gates._verify_ten_case_recomputation(arrays)


def test_training_environment_lock_is_canonical_exclusive_and_live(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "training_environment.lock.json"
    digest = create_training_environment_lock(Path.cwd(), lock)

    assert digest == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert not lock.read_bytes().endswith(b"\n")
    assert canonical_json_bytes(json.loads(lock.read_text(encoding="utf-8"))) == lock.read_bytes()
    assert verify_training_environment_lock(Path.cwd(), lock) == digest
    with pytest.raises(GateError, match="overwrite"):
        create_training_environment_lock(Path.cwd(), lock)


def test_output_space_receipt_requires_a_new_still_empty_directory(tmp_path: Path) -> None:
    output = tmp_path / "training-output"
    receipt = tmp_path / "output-space.receipt.json"
    digest = create_output_space_receipt(output, receipt)

    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert verify_output_space_receipt(output, receipt) == digest
    with pytest.raises(GateError, match="already exists"):
        create_output_space_receipt(output, tmp_path / "second.json")

    (output / "partial.bin").write_bytes(b"not empty")
    with pytest.raises(GateError, match="no longer empty"):
        verify_output_space_receipt(output, receipt)


def test_gate0_verifies_sources_but_refuses_to_invent_v44_unpacking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources: dict[str, Path] = {}
    for split in ("train", "val"):
        source = tmp_path / packing.CANONICAL_SOURCE_RELATIVE_PATHS[split]  # type: ignore[index]
        source.parent.mkdir(parents=True, exist_ok=True)
        payload = f"synthetic-{split}".encode()
        source.write_bytes(payload)
        monkeypatch.setitem(
            packing.CANONICAL_SOURCE_SHA256,
            split,
            hashlib.sha256(payload).hexdigest(),
        )
        sources[split] = source

    receipt = gate0_pack_receipt(
        repository_root=tmp_path,
        train_source=sources["train"],
        val_source=sources["val"],
        run_root=tmp_path / "prospective-run",
    )

    assert receipt.status == "BLOCKED"
    assert receipt.checks["approved_train_source"]
    assert receipt.checks["approved_val_source"]
    assert not receipt.checks["pack_adapter_frozen"]
    assert "v44_typed_identity_adapter_not_frozen" in receipt.blockers
    assert not (tmp_path / "prospective-run").exists()
    assert receipt.authorizes == ()


def test_unavailable_gate_inputs_return_machine_readable_blockers(tmp_path: Path) -> None:
    gate1 = gate1_fixture_receipt(
        repository_root=Path.cwd(),
        training_lock_path=None,
        fixture_lock_path=None,
        fixture_root=None,
        pack_receipt_path=None,
    )
    gate2 = gate2_selector_receipt(
        fixture_root=None,
        pack_receipt_path=None,
        features_root=None,
        order_receipt_path=None,
    )

    assert gate1.status == "BLOCKED"
    assert "training_environment_lock_missing" in gate1.blockers
    assert "canonical_fixture_pack_or_receipt_missing" in gate1.blockers
    assert "selector_unit_fixture_pack_or_receipt_missing" in gate1.blockers
    assert not gate1.checks["selector_unit_fixture_pack"]
    assert gate2.status == "BLOCKED"
    assert "selector_pre_draw_api_missing" not in gate2.blockers
    assert "frozen_real_track_predraw_order_not_supplied" in gate2.blockers
    assert not gate2.checks["pre_evaluation_order_receipt"]
    assert not gate2.checks["all_real_tracks_have_selector_answer"]
    assert gate1.authorizes == gate2.authorizes == ()


def test_gate3_cpu_scaffold_checks_execution_without_authorizing_training() -> None:
    receipt = gate3_cpu_sanity_receipt()

    assert receipt.status == "BLOCKED"
    assert receipt.checks["finite_gradients"]
    assert receipt.checks["loss_decrease"]
    assert receipt.checks["identity_isolation"]
    assert receipt.checks["invalid_state_hold"]
    assert receipt.checks["single_source_clock_commit"]
    assert receipt.checks["four_fresh_backwards"]
    assert not receipt.checks["tiny_gpu_jobs"]
    assert "three_tiny_gpu_receipts_missing" in receipt.blockers
    assert receipt.authorizes == ()
