from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import pams.temporac.certify as certify_module
from pams.temporac.certify import CertificateTrack, certify_target
from pams.temporac.types import _PROVENANCE_AUTHORITY, TargetProvenance


def _provenance(*, teacher: str = "1" * 64) -> TargetProvenance:
    return TargetProvenance(
        source_kind=1,
        source_key_hex="2" * 64,
        source_unit_index=7,
        teacher_sha256=teacher,
        _provenance_authority=_PROVENANCE_AUTHORITY,
    )


def _tau_result(**updates: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "status": "CERTIFIED",
        "reasons": np.empty(0, dtype="<u2"),
        "landmarks": np.asarray([1, 3], dtype="<i4"),
        "traversal_bounds": np.asarray([[1, 3]], dtype="<i4"),
        "seams": np.asarray([1.0], dtype="<f8"),
        "pulse": np.asarray([0, 1, 0, 0], dtype="|u1"),
        "winding": np.asarray([1], dtype="<i4"),
        "target_mask": np.asarray([0, 1, 1, 0], dtype="|u1"),
        "edge_mask": np.asarray([0, 1, 1, 0], dtype="|u1"),
        "decoder_mask": np.asarray([0, 1, 1, 0], dtype="|u1"),
        "provenance": _provenance(),
    }
    fields.update(updates)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("status", "ABSTAIN"),
        ("reasons", np.asarray([14], dtype="<u2")),
        ("landmarks", np.asarray([0, 2], dtype="<i4")),
        ("traversal_bounds", np.asarray([[0, 2]], dtype="<i4")),
        ("seams", np.asarray([1.25], dtype="<f8")),
        ("pulse", np.asarray([1, 0, 0, 0], dtype="|u1")),
        ("winding", np.asarray([0], dtype="<i4")),
        ("target_mask", np.asarray([1, 1, 0, 0], dtype="|u1")),
        ("edge_mask", np.asarray([1, 1, 1, 0], dtype="|u1")),
        ("decoder_mask", np.asarray([0, 1, 0, 0], dtype="|u1")),
        ("provenance", _provenance(teacher="3" * 64)),
    ],
)
def test_tau_comparison_binds_all_ten_ledgers_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    changed: object,
) -> None:
    reference = _tau_result()

    def fake_once(track: object, source_kind: str, tau: float) -> SimpleNamespace:
        del track, source_kind
        return _tau_result(**{field: changed}) if tau == 1e-8 else reference

    monkeypatch.setattr(certify_module, "_certify_once", fake_once)
    result = certify_target(object(), "natural")  # type: ignore[arg-type]
    assert result.status == "ABSTAIN"
    assert result.reasons.tolist() == [14]
    assert result.provenance == reference.provenance
    assert result.pulse.size == 0
    assert result.traversal_bounds.shape == (0, 2)


def test_dense_geometry_arc_uses_exact_66_over_d_masked_f4() -> None:
    geometry = np.zeros((3, 66), dtype="<f8")
    geometry[1, :32] = 1.0
    geometry[1, 32:] = 100.0
    geometry[2] = geometry[1] + 1.0
    continuous = np.zeros((3, 149), dtype="<f8")
    continuous[:, :66] = geometry
    mask = np.ones((3, 149), dtype="|u1")
    mask[0, 32:66] = 0
    phase_angle = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    phase = np.stack((np.cos(phase_angle), np.sin(phase_angle)), axis=1)
    track = CertificateTrack(
        geometry=geometry,
        continuous=continuous,
        continuous_mask=mask,
        phase=phase,
        reconstruction=continuous.copy(),
        run_bounds=np.asarray([[0, 3]], dtype="<i4"),
    )
    dense = certify_module._dense_traversal(track, phase, 0, 2)
    cumulative = dense[-1]
    # Edge zero has D=32 and sum(delta^2)=32, while edge one has D=66
    # and sum(delta^2)=66. F4's 66/D factor makes both lengths sqrt(66).
    assert cumulative.tolist() == pytest.approx([0.0, 0.5, 1.0], abs=1e-15)
    unmasked_first = float(np.linalg.norm(geometry[1] - geometry[0]))
    unmasked_second = float(np.linalg.norm(geometry[2] - geometry[1]))
    assert unmasked_first > 50.0 * unmasked_second


def test_geometry_arc_exact_right_tie_owns_outgoing_edge() -> None:
    cumulative = np.asarray([0.0, 0.25, 1.0], dtype="<f8")
    edge, rho, coordinate = certify_module._arc_sample_coordinate(cumulative, 10, 0.25)
    assert edge == 1
    assert rho == 0.0
    assert coordinate == 11.0


def test_phase_classes_union_adjacent_chain_transitively_and_then_wrap() -> None:
    count, gap = certify_module._phase_classes(np.asarray([0.0, 0.75e-8, 1.5e-8, 0.5], dtype="<f8"))
    assert count == 2
    assert gap == pytest.approx(0.5, abs=2e-8)

    wrapped_count, _ = certify_module._phase_classes(
        np.asarray([1.0 - 0.5e-8, 0.25e-8, 1.0e-8, 0.4], dtype="<f8")
    )
    assert wrapped_count == 2
