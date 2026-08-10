"""Deterministic spectral reference used only for smoke tests.

This adapter is intentionally named ``spectral-proxy`` and is never exposed
under the name of a published baseline.  It has no learned parameters and is
not eligible for paper-comparison tables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pams.baselines.base import (
    BaselineSpec,
    BaselineStatus,
    ImplementationKind,
    ProtocolRecord,
)
from pams.types import CountResult, PoseSequence

SPECTRAL_PROXY_SPEC = BaselineSpec(
    key="spectral-proxy",
    paper_name="Deterministic spectral proxy (not a paper baseline)",
    status=BaselineStatus.REFERENCE_ONLY,
    implementation=ImplementationKind.DETERMINISTIC_REFERENCE,
    protocols=(
        ProtocolRecord(
            key="synthetic_and_pipeline_smoke",
            dataset="synthetic-or-prepared-pose",
            train_split="none",
            evaluation_split="diagnostic only",
            modality="33x3 pose",
            source_faithful=False,
            fair_comparison=False,
            notes="Not eligible for a paper baseline table.",
        ),
    ),
    config_path="configs/baselines/spectral_proxy.yaml",
    is_paper_baseline=False,
)


def _to_numpy(value: object) -> np.ndarray:
    """Convert numpy/torch-like values without importing torch."""

    if isinstance(value, np.ndarray):
        return value
    detached = getattr(value, "detach", None)
    if callable(detached):
        value = detached()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        return np.asarray(numpy())
    return np.asarray(value)


def _period_from_autocorrelation(
    signal: np.ndarray,
    minimum_period: int,
    maximum_period: int,
) -> tuple[float, float]:
    """Estimate the dominant bounded period of a signed activity signal.

    The name is retained for compatibility with the first smoke adapter. A
    Hann-windowed spectrum is used because an unnormalised finite-lag
    autocorrelation systematically favours very short lags on slow actions.
    """

    centered = np.asarray(signal, dtype=np.float64) - float(np.mean(signal))
    energy = float(np.dot(centered, centered))
    if centered.size < minimum_period * 2 or energy <= np.finfo(np.float64).eps:
        return float(max(minimum_period, 1)), 0.0

    upper = min(maximum_period, centered.size // 2)
    if upper < minimum_period:
        return float(max(1, upper)), 0.0

    window = np.hanning(centered.size)
    power = np.abs(np.fft.rfft(centered * window)) ** 2
    frequencies = np.fft.rfftfreq(centered.size)
    allowed = (frequencies >= 1.0 / upper) & (frequencies <= 1.0 / minimum_period)
    allowed[0] = False
    band = np.where(allowed, power, 0.0)
    total = float(np.sum(band))
    if total <= np.finfo(np.float64).eps:
        return float(upper), 0.0
    index = int(np.argmax(band))
    period = float(
        np.clip(
            1.0 / frequencies[index],
            float(minimum_period),
            float(upper),
        )
    )
    confidence = float(np.clip(band[index] / total, 0.0, 1.0))
    return period, confidence


def _robust_pose_coordinate(xyz: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Select a high-coverage signed coordinate after mask-aware interpolation.

    Zero-filled joint triplets are the pose-cache convention for a missing
    joint.  Treating those zeros as observations can make the occlusion block
    the highest-variance coordinate and create a spurious low-frequency peak.
    This helper first identifies the coordinates with maximum observation
    coverage, linearly fills their unavailable samples, and only then applies
    the historical highest-variance signed-coordinate rule.
    """

    time = xyz.shape[0]
    flattened = xyz.reshape(time, -1).copy()
    if int(np.count_nonzero(valid)) < 2:
        return np.zeros(time, dtype=np.float64)

    joint_observed = np.any(np.abs(xyz) > np.finfo(np.float32).eps, axis=2)
    joint_observed &= valid[:, None]
    coordinate_observed = np.repeat(joint_observed, xyz.shape[2], axis=1)
    valid_count = int(np.count_nonzero(valid))
    coverage = coordinate_observed.sum(axis=0) / float(valid_count)
    maximum_coverage = float(np.max(coverage, initial=0.0))
    candidates = np.flatnonzero(
        (coverage >= maximum_coverage - 1e-12) & (coverage > 0.0)
    )
    if candidates.size == 0:
        return np.zeros(time, dtype=np.float64)

    frame_grid = np.arange(time)
    for coordinate in candidates:
        known = np.flatnonzero(coordinate_observed[:, coordinate])
        if known.size < 2:
            continue
        flattened[:, coordinate] = np.interp(
            frame_grid,
            known,
            flattened[known, coordinate],
        )

    candidate_values = flattened[:, candidates]
    centered_valid = candidate_values[valid] - candidate_values[valid].mean(
        axis=0,
        keepdims=True,
    )
    coordinate = int(candidates[np.argmax(np.mean(centered_valid**2, axis=0))])
    signal = flattened[:, coordinate].copy()
    signal -= float(np.mean(signal[valid]))
    return signal


@dataclass(slots=True)
class SpectralProxyAdapter:
    """A zero-training pose-energy autocorrelation reference."""

    minimum_period: int = 4
    maximum_period: int = 128

    @property
    def spec(self) -> BaselineSpec:
        return SPECTRAL_PROXY_SPEC

    def predict(self, sample: PoseSequence) -> CountResult:
        xyz = _to_numpy(sample.xyz).astype(np.float64, copy=False)
        valid = _to_numpy(sample.valid_mask).astype(bool, copy=False)
        if xyz.ndim != 3:
            raise ValueError(f"expected xyz shaped [T,K,3], got {xyz.shape}")
        if valid.shape != (xyz.shape[0],):
            raise ValueError(f"expected valid_mask shaped [{xyz.shape[0]}], got {valid.shape}")
        if xyz.shape[0] < 2:
            return CountResult(
                count=0,
                period_frames=float(max(self.minimum_period, 1)),
                expert_counts=(0, 0, 0),
                confidence=0.0,
                period_stream=np.zeros(xyz.shape[0], dtype=np.float32),
            )

        # A magnitude/velocity proxy commonly produces two peaks per action
        # cycle.  Preserve fundamental phase with a signed coordinate while
        # preventing zero-filled missing joints from winning the variance
        # selection.
        signal = _robust_pose_coordinate(xyz, valid)

        period, confidence = _period_from_autocorrelation(
            signal,
            minimum_period=self.minimum_period,
            maximum_period=self.maximum_period,
        )
        effective_frames = int(np.count_nonzero(valid))
        continuous_count = effective_frames / max(period, 1.0)
        quantized_period = max(1, int(round(period)))
        stable_count = math.floor(effective_frames / quantized_period)
        # The continuous peak avoids short-period quantization errors.  For a
        # diffuse peak (for example, a paused action), retain more of the
        # conservative quantized-period estimate.  The spectral confidence is
        # the interpolation weight, so this introduces no tuned threshold.
        blended_count = confidence * continuous_count + (1.0 - confidence) * stable_count
        count = int(math.floor(blended_count + 0.5))
        stream = signal.astype(np.float32)
        return CountResult(
            count=count,
            period_frames=float(period),
            expert_counts=(count, count, count),
            confidence=confidence,
            period_stream=stream,
        )
