from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from pams.single_expert_readout_v2 import (
    PREDECESSOR_DENIAL_RECEIPT_BYTES,
    PREDECESSOR_DENIAL_RECEIPT_SHA256,
    PREDECESSOR_DENIAL_RESULT_BYTES,
    PREDECESSOR_DENIAL_RESULT_SHA256,
    SpectralCandidateV2,
    derive_domain_separated_map_seed,
    estimate_representation_v2,
    load_segment_local_spectral_config_v2,
    materialize_synthetic_case_v2,
    synthetic_case_plan_v2,
    synthetic_invariance_variants_v2,
)

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY / "configs/readouts/segment_local_spectral_single_v2.yaml"


def _config():
    return load_segment_local_spectral_config_v2(CONFIG_PATH)


def _spec(fragment: str):
    matches = tuple(item for item in synthetic_case_plan_v2() if fragment in item.case_id)
    assert len(matches) == 1
    return matches[0]


def _estimate(fragment: str, *, seed: int = 62027, candidate_index: int = 0):
    config = _config()
    case = materialize_synthetic_case_v2(_spec(fragment), seed=seed)
    return case, estimate_representation_v2(case.video, config.candidates[candidate_index], config)


def test_config_freezes_denial_lineage_four_candidates_and_zero_data_authority() -> None:
    config = _config()
    assert config.predecessor_denial.result_sha256 == PREDECESSOR_DENIAL_RESULT_SHA256
    assert config.predecessor_denial.result_bytes == PREDECESSOR_DENIAL_RESULT_BYTES == 102245
    assert config.predecessor_denial.receipt_sha256 == PREDECESSOR_DENIAL_RECEIPT_SHA256
    assert config.predecessor_denial.receipt_bytes == PREDECESSOR_DENIAL_RECEIPT_BYTES == 1273
    assert config.predecessor_denial.grants_training_authority is False
    assert tuple(item.canonical_id for item in config.candidates) == (
        "cpw3.direct0.12",
        "cpw3.direct0.2",
        "cpw4.direct0.12",
        "cpw4.direct0.2",
    )
    assert config.authority.synthetic_selector_authorized is True
    assert config.authority.train337_authorized is False
    assert config.authority.dev84_authorized is False
    assert config.authority.test105_authorized is False


def test_config_rejects_v1_identity_or_axis_relaxation(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["key"] = "segment_local_spectral_single_v1"
    invalid_key = tmp_path / "invalid-key.yaml"
    invalid_key.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_segment_local_spectral_config_v2(invalid_key)

    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["axes"]["direct_to_dominant_minimum"] = [0.05, 0.20]
    relaxed = tmp_path / "relaxed.yaml"
    relaxed.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_segment_local_spectral_config_v2(relaxed)


def test_count2_uses_l_minus_one_proposal_and_full_segment_scale() -> None:
    _, estimate = _estimate("v2.count.t256.c2.d34")
    assert estimate.status == "eligible"
    assert estimate.rounded_count == 2
    segment = estimate.segments[0]
    assert segment.proposal.state == "periodic"
    assert segment.proposal.frequency == pytest.approx(2.0 / 255.0, abs=2e-4)
    assert segment.adaptive_window_frames == 256
    assert tuple(item.window.start for item in segment.windows) == (0,)
    assert tuple(item.window.stop for item in segment.windows) == (256,)
    assert len(segment.interval_rates) == 255
    assert segment.float_count == pytest.approx(2.0, abs=0.08)


@pytest.mark.parametrize(
    ("fragment", "target"),
    (
        ("v2.harmonic.second_harmonic_amp_1.25.c8.d34", 8),
        ("v2.harmonic.subharmonic_amp_0.5.c8.d34", 8),
    ),
)
def test_fundamental_disambiguation_recovers_strong_2f_and_rejects_weak_half_f(
    fragment: str,
    target: int,
) -> None:
    _, estimate = _estimate(fragment)
    assert estimate.status == "eligible"
    assert estimate.rounded_count == target
    proposal = estimate.segments[0].proposal
    assert proposal.frequency == pytest.approx(target / 255.0, rel=0.04)
    assert any(item.trusted for item in proposal.fundamental_candidates)
    assert any(item.divisor == 2 for item in proposal.fundamental_candidates)


def test_active_pause_writes_real_zero_rates_without_periodic_extrapolation() -> None:
    _, estimate = _estimate("v2.support.active_support_60.c8.d34")
    assert estimate.status == "eligible"
    assert abs(estimate.rounded_count - 8) <= 1
    segment = estimate.segments[0]
    states = tuple(item.state for item in segment.windows)
    assert "periodic" in states
    assert "static_zero_rate" in states
    assert any(value == 0.0 for value in segment.interval_rates)
    static_intervals = {
        interval
        for item in segment.windows
        if item.state == "static_zero_rate"
        for interval in range(item.window.start, item.window.stop - 1)
    }
    assert static_intervals
    assert all(segment.interval_rates[index] == 0.0 for index in static_intervals)


@pytest.mark.parametrize(
    ("fragment", "target"),
    (
        ("v2.tempo.ramp_up.c8.d34", 8),
        ("v2.tempo.up_then_down.c16.d34", 16),
        ("v2.corrupt.noise_sigma_0.02.c8.d34", 8),
        ("v2.corrupt.contiguous_joint_feature_occlusion.c16.d34", 16),
    ),
)
def test_variable_tempo_and_corruption_execute_real_algorithm(
    fragment: str,
    target: int,
) -> None:
    _, estimate = _estimate(fragment)
    assert estimate.status == "eligible"
    assert abs(estimate.rounded_count - target) <= 1
    assert estimate.segments[0].known_interval_union_fraction >= 0.8


def test_null_replicates_change_source_bytes_and_never_form_positive_count() -> None:
    config = _config()
    for family in (
        "constant",
        "drift_only",
        "white_noise",
        "random_walk",
        "ar1_rho_0.9",
        "time_shuffled_periodic",
        "independent_incoherent_frequency_phase",
        "constant_coordinates_oscillating_masks",
    ):
        specs = tuple(
            item
            for item in synthetic_case_plan_v2()
            if item.family == family and item.replicate in (0, 2)
        )
        assert len(specs) == 2
        cases = tuple(materialize_synthetic_case_v2(item, seed=62027) for item in specs)
        assert cases[0].source_sha256 != cases[1].source_sha256
        for case in cases:
            estimate = estimate_representation_v2(case.video, config.candidates[0], config)
            assert estimate.rounded_count in (None, 0)
            if estimate.status == "eligible":
                assert estimate.static_zero_only is True


def test_reset_total_is_undefined_even_when_each_segment_has_periodicity() -> None:
    _, estimate = _estimate("v2.reset.two_segment.r00.d34")
    assert estimate.status == "undefined_multi_segment"
    assert estimate.float_count is None
    assert estimate.rounded_count is None
    assert len(estimate.segments) == 2


def test_real_invariance_transforms_preserve_rounded_count() -> None:
    config = _config()
    case = materialize_synthetic_case_v2(_spec("v2.count.t256.c12.d34"), seed=62027)
    reference = estimate_representation_v2(case.video, config.candidates[0], config)
    assert reference.status == "eligible"
    assert reference.rounded_count == 12
    variants = synthetic_invariance_variants_v2(case, seed=62027)
    assert tuple(variants) == (
        "reverse",
        "sign_flip",
        "feature_permutation",
        "orthogonal_mixing",
        "scale",
    )
    for transformed in variants.values():
        estimate = estimate_representation_v2(transformed, config.candidates[0], config)
        assert estimate.status == "eligible"
        assert estimate.rounded_count == reference.rounded_count


def test_unique_terminal_window_and_diagnostic_order_are_exact() -> None:
    _, estimate = _estimate("v2.count.t192.c12.d34")
    segment = estimate.segments[0]
    starts = tuple(item.window.start for item in segment.windows)
    assert starts == tuple(sorted(set(starts)))
    assert segment.windows[-1].window.stop == 192
    assert segment.diagnostic_stages == (
        "informative_dimensions",
        "global_proposal",
        "window_state",
        "candidate_peaks",
        "signed_vector_acf",
        "interval_coverage",
        "integrated_count",
    )


def test_map1_map2_seed_derivation_is_exact_and_domain_separated() -> None:
    values = {}
    for role in ("map1", "map2"):
        digest = hashlib.sha256()
        digest.update(b"segment_local_spectral_single_v2_map_seed\0")
        digest.update((62027).to_bytes(8, byteorder="little", signed=True))
        for value in ("video-7", "segment-2", role):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
            digest.update(encoded)
        expected = int.from_bytes(digest.digest()[:8], byteorder="little", signed=False)
        actual = derive_domain_separated_map_seed(62027, "video-7", "segment-2", role)
        assert actual == expected
        values[role] = actual
    assert values["map1"] != values["map2"]
    assert derive_domain_separated_map_seed(62027, "video-7", "segment-3", "map1") != values[
        "map1"
    ]


def test_candidate_rejects_values_outside_preregistered_axis() -> None:
    with pytest.raises(ValueError):
        SpectralCandidateV2(2.0, 0.12)
    with pytest.raises(ValueError):
        SpectralCandidateV2(3.0, 0.05)


def test_data_launcher_is_fixed_exit3_and_never_mounts_targets() -> None:
    launcher = REPOSITORY / "scripts/server/run_segment_local_spectral_single_v2.py"
    completed = subprocess.run(
        [sys.executable, str(launcher)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 3
    payload = json.loads(completed.stderr)
    assert payload["synthetic_selector_authorized"] is True
    assert payload["train337_authorized"] is False
    assert payload["dev84_authorized"] is False
    assert payload["test105_authorized"] is False
    assert payload["targets_mounted"] is False
