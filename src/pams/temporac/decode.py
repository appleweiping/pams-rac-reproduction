"""One-shot float32 threshold-0.5 connected-component decoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Float32Array = NDArray[np.float32]
Int32Array = NDArray[np.int32]
UInt16Array = NDArray[np.uint16]


class DecodeError(ValueError):
    """Raised when the fixed decoder interface is not admissible."""


def _readonly(*arrays: NDArray[np.generic]) -> None:
    for array in arrays:
        array.setflags(write=False)


@dataclass(frozen=True, slots=True)
class ComponentTable:
    bounds: Int32Array
    location: Int32Array
    score: Float32Array

    def __post_init__(self) -> None:
        _readonly(self.bounds, self.location, self.score)

    @property
    def count(self) -> int:
        return int(self.bounds.shape[0])


@dataclass(frozen=True, slots=True)
class DecodeResult:
    """Serialized decoder fields plus the exactly-once invocation witness."""

    response: Float32Array
    decoder_mask: NDArray[np.uint8]
    run_bounds: Int32Array
    component_bounds: Int32Array
    component_location: Int32Array
    component_score: Float32Array
    count: int
    abstain: bool
    abstain_reasons: UInt16Array
    decoder_invocations: int

    def __post_init__(self) -> None:
        _readonly(
            self.response,
            self.decoder_mask,
            self.run_bounds,
            self.component_bounds,
            self.component_location,
            self.component_score,
            self.abstain_reasons,
        )
        if self.decoder_invocations != 1:
            raise DecodeError("the component decoder must be invoked exactly once")


def quantize_response(response: ArrayLike) -> Float32Array:
    """Round once, ties-to-even, to C-contiguous little-endian float32."""

    raw = np.asarray(response)
    if raw.ndim != 1 or raw.dtype.kind not in "fiu":
        raise DecodeError("response must be a one-dimensional numeric array")
    binary64 = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(binary64)):
        raise DecodeError("response must be finite before float32 quantization")
    result = np.ascontiguousarray(binary64.astype(np.dtype("<f4")))
    if not np.all(np.isfinite(result)):
        raise DecodeError("response became nonfinite during float32 quantization")
    return result


def _decoder_mask(mask: ArrayLike, edge_count: int) -> NDArray[np.uint8]:
    raw = np.asarray(mask)
    if raw.shape != (edge_count,) or raw.dtype.kind not in "bu":
        raise DecodeError("decoder_mask must be binary shape [E]")
    if not np.all((raw == 0) | (raw == 1)):
        raise DecodeError("decoder_mask must be binary")
    return np.ascontiguousarray(raw, dtype=np.uint8)


def _run_table(run_bounds: ArrayLike, edge_count: int) -> Int32Array:
    raw = np.asarray(run_bounds)
    if raw.ndim != 2 or raw.shape[1:] != (2,) or raw.dtype.kind not in "iu":
        raise DecodeError("run_bounds must be an integer [R,2] array")
    if edge_count == 0:
        if raw.shape != (0, 2):
            raise DecodeError("empty response requires exact run_bounds shape [0,2]")
        return np.empty((0, 2), dtype=np.int32)
    if raw.shape[0] == 0:
        raise DecodeError("nonempty response requires at least one run")
    result = np.asarray(raw, dtype=np.int64)
    previous_stop = -1
    for start, stop in result.tolist():
        if not (0 <= start < stop <= edge_count):
            raise DecodeError("run bound is empty or outside response")
        if start <= previous_stop:
            raise DecodeError("different decoder runs must be ordered and nonadjacent")
        previous_stop = stop
    return np.ascontiguousarray(result, dtype=np.int32)


def connected_components(
    response_f32: ArrayLike,
    decoder_mask: ArrayLike,
    run_bounds: ArrayLike,
) -> ComponentTable:
    """Apply the sole threshold and component pass.

    The caller must provide the already-quantized response.  Keeping
    quantization outside this function ensures the serialized array and the
    decision array are exactly the same bytes.
    """

    response_raw = np.asarray(response_f32)
    if response_raw.ndim != 1 or response_raw.dtype != np.dtype("<f4"):
        raise DecodeError("connected_components requires little-endian float32 response")
    response = np.ascontiguousarray(response_raw, dtype=np.dtype("<f4"))
    if not np.all(np.isfinite(response)):
        raise DecodeError("decoder response must be finite")
    edge_count = response.shape[0]
    mask = _decoder_mask(decoder_mask, edge_count)
    runs = _run_table(run_bounds, edge_count)
    owned = np.zeros(edge_count, dtype=np.bool_)
    for start, stop in runs.tolist():
        owned[start:stop] = True
    if np.any(mask.astype(np.bool_) & ~owned):
        raise DecodeError("decoder_mask cannot select an edge outside the run table")

    active = mask.astype(np.bool_) & (response >= np.float32(0.5))
    component_bounds: list[tuple[int, int]] = []
    component_locations: list[int] = []
    component_scores: list[np.float32] = []
    for run_start, run_stop in runs.tolist():
        edge = run_start
        while edge < run_stop:
            if not active[edge]:
                edge += 1
                continue
            start = edge
            while edge < run_stop and active[edge]:
                edge += 1
            stop = edge
            values = response[start:stop]
            score = np.max(values)
            maxima = np.flatnonzero(values == score) + start
            location = int((int(maxima[0]) + int(maxima[-1])) // 2)
            component_bounds.append((start, stop))
            component_locations.append(location)
            component_scores.append(np.float32(score))

    bounds_array = np.asarray(component_bounds, dtype=np.int32).reshape((-1, 2))
    location_array = np.asarray(component_locations, dtype=np.int32)
    score_array = np.asarray(component_scores, dtype=np.dtype("<f4"))
    return ComponentTable(
        bounds=np.ascontiguousarray(bounds_array, dtype=np.int32),
        location=np.ascontiguousarray(location_array, dtype=np.int32),
        score=np.ascontiguousarray(score_array, dtype=np.dtype("<f4")),
    )


def _reason_vector(reasons: ArrayLike) -> UInt16Array:
    raw = np.asarray(reasons)
    if raw.ndim == 1 and raw.size == 0:
        return np.empty(0, dtype=np.dtype("<u2"))
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise DecodeError("abstain reasons must be a one-dimensional integer array")
    values = np.asarray(raw, dtype=np.uint64)
    if np.any(values == 0) or np.any(values > 24):
        raise DecodeError("abstain reasons must be contract values 1 through 24")
    if values.size > 1 and not np.all(values[1:] > values[:-1]):
        raise DecodeError("abstain reasons must be unique and ascending")
    return np.ascontiguousarray(values, dtype=np.dtype("<u2"))


def decode_identity(
    response: ArrayLike | None,
    decoder_mask: ArrayLike | None,
    run_bounds: ArrayLike | None,
    *,
    abstain_reasons: ArrayLike = (),
) -> DecodeResult:
    """Quantize and invoke the fixed decoder exactly once for one identity.

    Supplying any reason emits the exact empty stub.  The component function
    is nevertheless called once on those empty arrays and its internal zero is
    replaced by the serialized scientific ``count=-1`` abstention sentinel.
    """

    reasons = _reason_vector(abstain_reasons)
    abstain = bool(reasons.size)
    if abstain:
        quantized = np.empty(0, dtype=np.dtype("<f4"))
        mask = np.empty(0, dtype=np.uint8)
        runs = np.empty((0, 2), dtype=np.int32)
    else:
        if response is None or decoder_mask is None or run_bounds is None:
            raise DecodeError("non-abstention decode requires response, mask, and runs")
        quantized = quantize_response(response)
        mask = _decoder_mask(decoder_mask, quantized.shape[0])
        runs = _run_table(run_bounds, quantized.shape[0])

    # This is intentionally the only call site in the identity wrapper.
    components = connected_components(quantized, mask, runs)
    if abstain and components.count != 0:
        raise AssertionError("stub decoding must internally obtain zero components")
    serialized_count = -1 if abstain else components.count
    return DecodeResult(
        response=quantized,
        decoder_mask=mask,
        run_bounds=runs,
        component_bounds=components.bounds,
        component_location=components.location,
        component_score=components.score,
        count=serialized_count,
        abstain=abstain,
        abstain_reasons=reasons,
        decoder_invocations=1,
    )


__all__ = [
    "ComponentTable",
    "DecodeError",
    "DecodeResult",
    "connected_components",
    "decode_identity",
    "quantize_response",
]
