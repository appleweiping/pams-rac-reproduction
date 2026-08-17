"""Teacher, integer-landmark, certificate, and topology invariants."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pams.temporac.certify import (
    REASON_SYNTHETIC_PULSE_MISMATCH,
    TOPOLOGY_DEFINITIONS,
    CertificateTrack,
    apply_topology_definition,
    certify_target,
    integer_landmarks,
)
from pams.temporac.teacher import (
    TeacherCheckpointScore,
    TempoRACTeacher,
    phase_increments,
    select_teacher_checkpoint,
    teacher_objective,
)
from pams.temporac.x0 import LANDMARK_RHO


def _ideal_certificate_track() -> CertificateTrack:
    sample_count = 345
    clock = np.arange(sample_count, dtype=np.float64)
    theta = (clock - 8.0) / 32.0
    rho = LANDMARK_RHO / np.linalg.norm(LANDMARK_RHO)
    orthogonal = np.arange(1.0, 67.0)
    orthogonal -= rho * float(np.dot(orthogonal, rho))
    orthogonal /= np.linalg.norm(orthogonal)
    geometry = 0.3 * (
        np.cos(2.0 * np.pi * theta)[:, None] * rho
        + np.sin(2.0 * np.pi * theta)[:, None] * orthogonal
    )
    continuous = np.zeros((sample_count, 149), dtype=np.float64)
    continuous[:, :66] = geometry
    continuous[:, 66:] = 0.3 * np.sin(2.0 * np.pi * theta)[:, None]
    phase = np.stack((np.cos(2.0 * np.pi * theta), np.sin(2.0 * np.pi * theta)), axis=1)
    pulse = np.zeros(sample_count - 1, dtype=np.uint8)
    chi = np.zeros(sample_count - 1, dtype=np.float64)
    for left in range(8, 328, 32):
        pulse[left] = 1
        chi[left : left + 32] = 1.0 / 32.0
    return CertificateTrack(
        geometry=geometry,
        continuous=continuous,
        continuous_mask=np.ones((sample_count, 149), dtype=np.uint8),
        phase=phase,
        reconstruction=continuous.copy(),
        run_bounds=np.asarray([[0, sample_count]], dtype=np.int32),
        analytic_pulse=pulse,
        analytic_chi=chi,
    )


def test_teacher_graph_f11_and_no_bypass_shapes() -> None:
    torch.manual_seed(20260815)
    model = TempoRACTeacher()
    teacher_input = torch.randn(2, 17, 215)
    static_features = torch.randn(2, 298)
    static_code, phase, reconstruction = model(teacher_input, static_features)
    assert static_code.shape == (2, 32)
    assert phase.shape == (2, 17, 2)
    assert reconstruction.shape == (2, 17, 149)
    torch.testing.assert_close(
        torch.linalg.vector_norm(static_code, dim=-1),
        torch.ones(2),
        rtol=0.0,
        atol=2e-6,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(phase, dim=-1),
        torch.ones(2, 17),
        rtol=0.0,
        atol=2e-6,
    )
    assert model.reconstruction.layers[0].in_features == 34
    assert all("gru" not in type(module).__name__.lower() for module in model.modules())

    angle = torch.arange(17, dtype=torch.float32) / 20.0
    observed_phase = torch.stack(
        (torch.cos(2.0 * torch.pi * angle), torch.sin(2.0 * torch.pi * angle)),
        dim=-1,
    )
    torch.testing.assert_close(
        phase_increments(observed_phase),
        torch.full((16,), 0.05),
        rtol=0.0,
        atol=2e-7,
    )
    prediction = torch.zeros(4, 149)
    target = torch.full_like(prediction, 0.1)
    loss = teacher_objective(
        reconstruction=prediction,
        reconstruction_target=target,
        reconstruction_mask=torch.ones_like(prediction),
        reference_phase=observed_phase,
        view_phase=observed_phase,
        observed_phase=observed_phase,
    )
    assert loss.reconstruction.item() == pytest.approx(0.075, abs=1e-7)
    assert loss.correspondence.item() == pytest.approx(0.0, abs=2e-7)
    assert loss.orientation.item() == 0.0
    assert loss.alias.item() == 0.0
    assert loss.total.item() == pytest.approx(0.075, abs=2e-7)


def test_teacher_selection_precedence() -> None:
    rows = (
        TeacherCheckpointScore(20260815, 500, 0.8, 0.01, 2, 0.1, "a"),
        TeacherCheckpointScore(20260815, 1000, 0.8, 0.02, 0, 0.0, "discard"),
        TeacherCheckpointScore(20260816, 500, 0.7, 0.0, 1, 0.3, "b"),
        TeacherCheckpointScore(20260817, 500, 0.6, 0.0, 1, 0.2, "c"),
    )
    selected = select_teacher_checkpoint(rows)
    assert selected.payload == "c"


def test_integer_landmarks_certificate_and_exact_topology_bank() -> None:
    track = _ideal_certificate_track()
    for array in (
        track.geometry,
        track.continuous,
        track.continuous_mask,
        track.phase,
        track.reconstruction,
        track.run_bounds,
        track.analytic_pulse,
        track.analytic_chi,
    ):
        assert array is not None
        with pytest.raises(ValueError):
            array.flags.writeable = True
    expected_landmarks = np.arange(8, 329, 32, dtype=np.int32)
    assert np.array_equal(integer_landmarks(track.geometry, track.run_bounds), expected_landmarks)

    reference = certify_target(track, "X0")
    assert reference.certified
    assert reference.event_count == 10
    assert np.array_equal(reference.landmarks, expected_landmarks)
    assert np.array_equal(reference.winding, np.ones(10, dtype=np.int32))
    assert track.analytic_pulse is not None
    assert np.array_equal(reference.pulse, track.analytic_pulse)
    assert not reference.pulse.flags.writeable

    assert len(TOPOLOGY_DEFINITIONS) == 13
    assert tuple(item.identifier for item in TOPOLOGY_DEFINITIONS) == tuple(
        f"T{index:02d}" for index in range(13)
    )
    for definition in TOPOLOGY_DEFINITIONS:
        result = certify_target(apply_topology_definition(track, definition), "X0")
        if definition.accept:
            assert result.certified, definition.identifier
            assert result.reasons.size == 0
            assert np.array_equal(result.landmarks, reference.landmarks)
            assert np.array_equal(result.traversal_bounds, reference.traversal_bounds)
            assert np.array_equal(result.pulse, reference.pulse)
            assert np.array_equal(result.target_mask, reference.target_mask)
            assert result.event_count == reference.event_count
        else:
            assert not result.certified, definition.identifier
            assert result.reasons.tolist() == [definition.reason]
            assert result.pulse.size == 0


def test_x0_pulse_mismatch_abstains_without_partial_arrays() -> None:
    track = _ideal_certificate_track()
    bad_pulse = np.asarray(track.analytic_pulse).copy()
    bad_pulse[8] = 0
    bad = CertificateTrack(
        geometry=track.geometry,
        continuous=track.continuous,
        continuous_mask=track.continuous_mask,
        phase=track.phase,
        reconstruction=track.reconstruction,
        run_bounds=track.run_bounds,
        analytic_pulse=bad_pulse,
        analytic_chi=track.analytic_chi,
    )
    result = certify_target(bad, "X0")
    assert result.status == "ABSTAIN"
    assert result.reasons.tolist() == [REASON_SYNTHETIC_PULSE_MISMATCH]
    assert result.pulse.size == 0
    assert result.target_mask.size == 0
