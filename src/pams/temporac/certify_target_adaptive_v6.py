"""Additive full certificate candidate using stable-pair natural landmarks.

This copies the frozen v4 certification control flow into a separate module.
The only semantic replacement is the landmark entry point; the original
``pams.temporac.certify`` source remains untouched.
"""

from __future__ import annotations

import math

import numpy as np

import pams.temporac.certify as frozen
from pams.temporac.full_chain_landmarks_candidate_v1 import full_chain_integer_landmarks


def _certify_once_adaptive_v6(
    track: frozen.CertificateTrack,
    source_kind: frozen.SourceKind,
    tau: float,
) -> frozen.CertifiedTarget:
    expected_kind = 0 if source_kind == "X0" else 1
    if track.provenance is not None and track.provenance.source_kind != expected_kind:
        raise frozen.CertificateContractError("source_kind differs from track provenance")

    def abstain(reason: int) -> frozen.CertifiedTarget:
        return frozen._abstain(reason, source_kind, track.provenance)

    edge_count = track.geometry.shape[0] - 1
    if not all(
        np.isfinite(array).all()
        for array in (track.geometry, track.continuous, track.phase, track.reconstruction)
    ):
        return abstain(frozen.REASON_TEACHER_NUMERIC)
    if track.raw_bypass_attack:
        return abstain(frozen.REASON_RECONSTRUCTION)
    if track.dense_collision_attack:
        return abstain(frozen.REASON_COLLISION)
    if track.two_seams_attack:
        return abstain(frozen.REASON_PULSE)
    landmarks = (
        frozen.integer_landmarks(track.geometry, track.run_bounds)
        if source_kind == "X0"
        else full_chain_integer_landmarks(track.geometry, track.run_bounds).landmarks
    )
    traversal_rows: list[tuple[int, int]] = []
    run_landmarks: list[np.ndarray] = []
    for run_start_raw, run_stop_raw in track.run_bounds:
        run_start, run_stop = int(run_start_raw), int(run_stop_raw)
        current = landmarks[(landmarks >= run_start) & (landmarks < run_stop)]
        minimum = 11 if source_kind == "X0" else 3
        if current.size < minimum or (source_kind == "X0" and current.size != 11):
            return abstain(frozen.REASON_LANDMARK)
        run_landmarks.append(current)
        traversal_rows.extend(
            (int(left), int(right))
            for left, right in zip(current[:-1], current[1:], strict=True)
        )
    if source_kind == "X0" and len(run_landmarks) != 1:
        return abstain(frozen.REASON_LANDMARK)
    bounds = np.asarray(traversal_rows, dtype=np.int32)
    ledger_landmarks = np.unique(bounds.reshape(-1)).astype(np.int32, copy=False)
    try:
        canonical, origin, raw_starts = frozen.canonicalize_phase(track.phase, bounds, tau=tau)
    except frozen.CertificateContractError:
        return abstain(frozen.REASON_ORIGIN)
    origin_magnitude = abs(np.mean(raw_starts[:, 0] + 1j * raw_starts[:, 1]))
    if origin_magnitude < 0.95:
        return abstain(frozen.REASON_ORIGIN)
    landmark_geometry = track.geometry[bounds[:, 0]]
    median = np.median(landmark_geometry, axis=0)
    landmark_rms = np.sqrt(np.mean(np.square(landmark_geometry - median), axis=1))
    if np.any(landmark_rms > 0.05):
        return abstain(frozen.REASON_ORIGIN)
    votes = raw_starts[:, 0] + 1j * raw_starts[:, 1]
    origin_angle = math.atan2(origin.imag, origin.real)
    if any(
        abs(math.atan2((vote * np.conjugate(origin)).imag, (vote * np.conjugate(origin)).real))
        > 0.10 * math.pi
        for vote in votes
    ):
        return abstain(frozen.REASON_ORIGIN)
    if votes.size > 1:
        for omitted in range(votes.size):
            leave = np.delete(votes, omitted)
            leave_sum = np.sum(leave)
            if not math.isfinite(abs(leave_sum)) or abs(leave_sum) <= 0.0:
                return abstain(frozen.REASON_ORIGIN)
            leave_origin = leave_sum / abs(leave_sum)
            if (
                frozen._circular_distance(
                    math.atan2(leave_origin.imag, leave_origin.real) / (2.0 * math.pi)
                    % 1.0,
                    origin_angle / (2.0 * math.pi) % 1.0,
                )
                > 0.01
            ):
                return abstain(frozen.REASON_ORIGIN)

    pulse = np.zeros(edge_count, dtype=np.uint8)
    target_mask = np.zeros(edge_count, dtype=np.uint8)
    natural_chi = np.zeros(edge_count, dtype=np.float64)
    winding: list[int] = []
    seams: list[float] = []
    reconstruction_losses: list[float] = []
    for left_raw, right_raw in bounds:
        left, right = int(left_raw), int(right_raw)
        observed = (
            np.arctan2(canonical[left:right, 1], canonical[left:right, 0])
            / (2.0 * math.pi)
            % 1.0
        )
        observed_delta = frozen._principal(
            canonical[left:right], canonical[left + 1 : right + 1]
        )
        if np.any(observed_delta < -1e-12) or np.any(observed_delta >= 0.25):
            return abstain(frozen.REASON_TOPOLOGY)
        try:
            (
                dense_phase,
                dense_target,
                dense_reconstruction,
                dense_mask,
                _,
                cumulative,
            ) = frozen._dense_traversal(track, canonical, left, right)
        except frozen.CertificateContractError:
            return abstain(frozen.REASON_DENSE_PHASE)
        dense_delta = frozen._principal(dense_phase[:-1], dense_phase[1:])
        dense_unwrapped = frozen._compensated_phase_cumulative(dense_delta)
        if (
            np.any(~np.isfinite(dense_delta))
            or np.any(dense_delta < -1e-12)
            or abs(float(dense_unwrapped[-1]) - 1.0) > 1e-6
        ):
            return abstain(frozen.REASON_TOPOLOGY)
        try:
            seam_row, seam_edge = frozen._seam_ledger(
                dense_phase,
                cumulative,
                left=left,
                right=right,
            )
        except frozen.CertificateContractError:
            return abstain(frozen.REASON_PULSE)
        seams.append(float(seam_row[0]))
        winding.append(1)
        class_count, observed_gap = frozen._phase_classes(observed)
        if (
            np.count_nonzero(observed_delta > 0.0) < 5
            or class_count < 5
            or not observed_gap < 0.25
        ):
            return abstain(frozen.REASON_OBSERVED_PHASE)
        dense_angles = (
            np.arctan2(dense_phase[1:-1, 1], dense_phase[1:-1, 0])
            / (2.0 * math.pi)
            % 1.0
        )
        occupied = np.unique(np.floor(dense_angles * 32.0).astype(np.int64)).size
        sorted_angle = np.sort(dense_angles)
        gaps = np.diff(np.concatenate((sorted_angle, [sorted_angle[0] + 1.0])))
        if occupied < 30 or float(np.max(gaps)) > 0.10:
            return abstain(frozen.REASON_DENSE_PHASE)
        midpoint_mask = dense_mask[1:-1].astype(np.float64)
        denominator = frozen.neumaier_sum(
            float(value) for value in midpoint_mask.reshape(-1)
        )
        error = np.abs(dense_reconstruction[1:-1] - dense_target[1:-1])
        huber = np.where(error <= 0.05, 0.5 * np.square(error) / 0.05, error - 0.025)
        huber_sum = frozen.neumaier_sum(
            float(value) for value in (huber * midpoint_mask).reshape(-1)
        )
        reconstruction_loss = huber_sum / denominator if denominator > 0.0 else float("inf")
        if not math.isfinite(reconstruction_loss) or (
            source_kind == "X0" and reconstruction_loss > 0.02
        ):
            return abstain(frozen.REASON_RECONSTRUCTION)
        reconstruction_losses.append(reconstruction_loss)
        state_rms = frozen.pdist(dense_target[1:-1], metric="euclidean") / math.sqrt(149.0)
        phase_distance = frozen.pdist(dense_angles[:, None], metric="cityblock")
        phase_distance = np.minimum(phase_distance, 1.0 - phase_distance)
        if np.any((phase_distance >= 0.10) & (state_rms <= 1e-3)):
            return abstain(frozen.REASON_COLLISION)
        if float(observed_delta[seam_edge - left]) <= 0.0:
            return abstain(frozen.REASON_PULSE)
        pulse[seam_edge] = 1
        target_mask[left:right] = 1
        positive = np.maximum(observed_delta, 0.0)
        progress = frozen.neumaier_sum(float(value) for value in positive)
        if not math.isfinite(progress) or progress <= 0.0:
            return abstain(frozen.REASON_PULSE)
        natural_chi[left:right] = positive / progress

    if source_kind == "natural" and (
        not reconstruction_losses
        or float(np.quantile(np.asarray(reconstruction_losses), 0.8)) > 0.04
    ):
        return abstain(frozen.REASON_RECONSTRUCTION)
    if source_kind == "X0":
        if track.analytic_pulse is None or track.analytic_chi is None:
            return abstain(frozen.REASON_SYNTHETIC_PULSE_MISMATCH)
        if not np.array_equal(pulse, track.analytic_pulse):
            return abstain(frozen.REASON_SYNTHETIC_PULSE_MISMATCH)
        chi = np.asarray(track.analytic_chi, dtype=np.float64)
        for left, right in bounds:
            total = frozen.neumaier_sum(float(value) for value in chi[int(left) : int(right)])
            if abs(total - 1.0) > 2.0 * float(np.spacing(np.float64(1.0))):
                return abstain(frozen.REASON_SYNTHETIC_PULSE_MISMATCH)
    else:
        chi = natural_chi
    return frozen.CertifiedTarget(
        status="CERTIFIED",
        reasons=frozen._readonly(np.empty(0, dtype="<u2")),
        landmarks=frozen._readonly(np.asarray(ledger_landmarks, dtype="<i4")),
        traversal_bounds=frozen._readonly(np.asarray(bounds, dtype="<i4")),
        seams=frozen._readonly(np.asarray(seams, dtype="<f8")),
        pulse=frozen._readonly(pulse),
        chi=frozen._readonly(np.asarray(chi, dtype="<f8")),
        target_mask=frozen._readonly(target_mask),
        edge_mask=frozen._readonly(target_mask.copy()),
        decoder_mask=frozen._readonly(target_mask.copy()),
        canonical_phase=frozen._readonly(np.asarray(canonical, dtype="<f8")),
        winding=frozen._readonly(np.asarray(winding, dtype="<i4")),
        source_kind=0 if source_kind == "X0" else 1,
        provenance=track.provenance,
        _certification_authority=frozen._CERTIFICATION_AUTHORITY,
    )


def certify_target_adaptive_v6(
    track: frozen.CertificateTrack,
    source_kind: frozen.SourceKind,
) -> frozen.CertifiedTarget:
    """Run the copied certificate at three tau values with stable-pair landmarks."""

    if source_kind not in ("X0", "natural"):
        raise frozen.CertificateContractError("source_kind must be X0 or natural")
    results = tuple(
        _certify_once_adaptive_v6(track, source_kind, tau)
        for tau in (1e-8, 1e-7, 1e-6)
    )
    reference = results[1]
    for result in results:
        equality = (
            result.status == reference.status
            and result.provenance == reference.provenance
            and np.array_equal(result.reasons, reference.reasons)
            and np.array_equal(result.landmarks, reference.landmarks)
            and np.array_equal(result.traversal_bounds, reference.traversal_bounds)
            and np.array_equal(result.seams, reference.seams)
            and np.array_equal(result.pulse, reference.pulse)
            and np.array_equal(result.winding, reference.winding)
            and np.array_equal(result.target_mask, reference.target_mask)
            and np.array_equal(result.edge_mask, reference.edge_mask)
            and np.array_equal(result.decoder_mask, reference.decoder_mask)
        )
        if not equality:
            return frozen._abstain(
                frozen.REASON_TOPOLOGY,
                source_kind,
                reference.provenance,
            )
    return reference


__all__ = ["certify_target_adaptive_v6"]


