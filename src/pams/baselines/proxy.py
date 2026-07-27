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
) -> tuple[int, float]:
    """Estimate the dominant bounded period of a signed activity signal.

    The name is retained for compatibility with the first smoke adapter. A
    Hann-windowed spectrum is used because an unnormalised finite-lag
    autocorrelation systematically favours very short lags on slow actions.
    """

    centered = np.asarray(signal, dtype=np.float64) - float(np.mean(signal))
    energy = float(np.dot(centered, centered))
    if centered.size < minimum_period * 2 or energy <= np.finfo(np.float64).eps:
        return max(minimum_period, 1), 0.0

    upper = min(maximum_period, centered.size // 2)
    if upper < minimum_period:
        return max(1, upper), 0.0

    window = np.hanning(centered.size)
    power = np.abs(np.fft.rfft(centered * window)) ** 2
    frequencies = np.fft.rfftfreq(centered.size)
    allowed = (frequencies >= 1.0 / upper) & (frequencies <= 1.0 / minimum_period)
    allowed[0] = False
    band = np.where(allowed, power, 0.0)
    total = float(np.sum(band))
    if total <= np.finfo(np.float64).eps:
        return upper, 0.0
    index = int(np.argmax(band))
    period = int(round(1.0 / frequencies[index]))
    period = min(max(period, minimum_period), upper)
    confidence = float(np.clip(band[index] / total, 0.0, 1.0))
    return period, confidence


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
        # cycle.  Use the highest-variance signed pose coordinate instead so
        # the diagnostic preserves fundamental phase.
        flattened = xyz.reshape(xyz.shape[0], -1)
        valid_values = flattened[valid]
        if valid_values.shape[0] >= 2:
            centered_valid = valid_values - valid_values.mean(axis=0, keepdims=True)
            coordinate = int(np.argmax(np.mean(centered_valid**2, axis=0)))
            signal = flattened[:, coordinate].copy()
            signal -= float(np.mean(signal[valid]))
            if not valid.all():
                known = np.flatnonzero(valid)
                missing = np.flatnonzero(~valid)
                signal[missing] = np.interp(missing, known, signal[known])
        else:
            signal = np.zeros(xyz.shape[0], dtype=np.float64)

        period, confidence = _period_from_autocorrelation(
            signal,
            minimum_period=self.minimum_period,
            maximum_period=self.maximum_period,
        )
        effective_frames = int(np.count_nonzero(valid))
        count = int(math.floor(effective_frames / max(period, 1)))
        stream = signal.astype(np.float32)
        return CountResult(
            count=count,
            period_frames=float(period),
            expert_counts=(count, count, count),
            confidence=confidence,
            period_stream=stream,
        )
