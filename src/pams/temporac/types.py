"""Frozen, shape-checked records private to TempoRAC.

No record in this module inherits from or adapts a historical PAMS type.
Array fields are copied to immutable ``bytes`` backing storage at construction
so a frozen dataclass also has deeply immutable array payloads.  In
particular, callers cannot re-enable NumPy's writeable flag.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, fields
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from pams.temporac.contract import CONTRACT_SHA256, ReasonCode

Float32Array: TypeAlias = NDArray[np.float32]
Float64Array: TypeAlias = NDArray[np.float64]
Int32Array: TypeAlias = NDArray[np.int32]
Int64Array: TypeAlias = NDArray[np.int64]
UInt8Array: TypeAlias = NDArray[np.uint8]
UInt16Array: TypeAlias = NDArray[np.uint16]
GenericArray: TypeAlias = NDArray[np.generic]

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CONTRACT_RAW = bytes.fromhex(CONTRACT_SHA256)
_PROVENANCE_AUTHORITY = object()
_CERTIFICATION_AUTHORITY = object()


class ContractError(ValueError):
    """Fail-closed violation of ``temporac.execution.v4``."""


@dataclass(frozen=True, slots=True)
class ArraySpec:
    """One canonical NPY dtype and shape (``None`` is a variable axis)."""

    dtype: str
    shape: tuple[int | None, ...]

    def __post_init__(self) -> None:
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise ContractError(f"invalid array dtype {self.dtype!r}") from exc
        if dtype.hasobject or dtype.kind in {"O", "U", "V"}:
            raise ContractError("object, Unicode, structured, and void dtypes are forbidden")
        if any(axis is not None and (type(axis) is not int or axis < 0) for axis in self.shape):
            raise ContractError("array shape axes must be nonnegative integers or None")

    def validate(self, array: GenericArray, *, name: str) -> None:
        if not isinstance(array, np.ndarray):
            raise ContractError(f"{name} must be a NumPy array")
        if array.dtype.str != self.dtype:
            raise ContractError(f"{name} dtype is {array.dtype.str}, expected {self.dtype}")
        if len(array.shape) != len(self.shape) or any(
            expected is not None and observed != expected
            for observed, expected in zip(array.shape, self.shape, strict=True)
        ):
            raise ContractError(f"{name} shape is {array.shape}, expected {self.shape}")
        if not array.flags.c_contiguous:
            raise ContractError(f"{name} must be C-contiguous")


def _readonly_exact(
    value: GenericArray,
    *,
    name: str,
    dtype: str,
    shape: tuple[int | None, ...],
) -> GenericArray:
    spec = ArraySpec(dtype=dtype, shape=shape)
    spec.validate(value, name=name)
    contiguous = np.ascontiguousarray(value, dtype=np.dtype(dtype))
    backing = contiguous.tobytes(order="C")
    result = np.frombuffer(backing, dtype=np.dtype(dtype)).reshape(contiguous.shape, order="C")
    spec.validate(result, name=name)
    return cast(GenericArray, result)


def _binary(array: GenericArray, *, name: str) -> None:
    if not np.isin(array, (0, 1)).all():
        raise ContractError(f"{name} must contain only zero and one")


def _sha256(value: str, *, name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")


def _negative_zero(value: float) -> bool:
    return value == 0.0 and math.copysign(1.0, value) < 0.0


def _set_readonly(
    instance: object,
    name: str,
    dtype: str,
    shape: tuple[int | None, ...],
) -> GenericArray:
    array = _readonly_exact(
        cast(GenericArray, getattr(instance, name)), name=name, dtype=dtype, shape=shape
    )
    object.__setattr__(instance, name, array)
    return array


@dataclass(frozen=True, slots=True)
class ArrayMemberRecord:
    """The exact five-key receipt record for one NPY member."""

    bytes: int
    dtype: str
    name: str
    sha256: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.bytes) is not int or self.bytes < 0:
            raise ContractError("member bytes must be a nonnegative integer")
        if not self.name.endswith(".npy") or not self.name.isascii() or "/" in self.name:
            raise ContractError("member name must be a flat ASCII .npy name")
        _sha256(self.sha256, name="member sha256")
        ArraySpec(self.dtype, tuple(self.shape))

    @property
    def byte_count(self) -> int:
        """Readable alias; canonical receipts still use the key ``bytes``."""

        return self.bytes

    def as_dict(self) -> dict[str, object]:
        return {
            "bytes": self.bytes,
            "dtype": self.dtype,
            "name": self.name,
            "sha256": self.sha256,
            "shape": list(self.shape),
        }


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    """The exact seven-array, per-identity feature archive payload."""

    frame_mask: UInt8Array
    local_person_slot: Int64Array
    motion: Float32Array
    opaque_sample_key: UInt8Array
    person_mask: UInt8Array
    sampled_frame_indices: Int64Array
    source_length: Int64Array

    def __post_init__(self) -> None:
        frame_mask = _set_readonly(self, "frame_mask", "|u1", (320,))
        slot = _set_readonly(self, "local_person_slot", "<i8", (1,))
        motion = _set_readonly(self, "motion", "<f4", (320, 17, 3))
        _set_readonly(self, "opaque_sample_key", "|u1", (32,))
        person_mask = _set_readonly(self, "person_mask", "|u1", (1,))
        clocks = _set_readonly(self, "sampled_frame_indices", "<i8", (320,))
        source_length = _set_readonly(self, "source_length", "<i8", (1,))
        _binary(frame_mask, name="frame_mask")
        _binary(person_mask, name="person_mask")
        if int(person_mask[0]) != 1:
            raise ContractError("person_mask must contain the exact value one")
        if not np.isfinite(motion).all():
            raise ContractError("motion must contain only finite float32 values")
        if int(slot[0]) < 0:
            raise ContractError("local_person_slot must be nonnegative")
        if int(source_length[0]) <= 1:
            raise ContractError("source_length must be greater than one")
        if np.any(np.diff(clocks) < 0):
            raise ContractError("sampled_frame_indices must be nondecreasing")

    @property
    def slot(self) -> int:
        return int(self.local_person_slot[0])

    @property
    def opaque_key_bytes(self) -> bytes:
        return self.opaque_sample_key.tobytes(order="C")

    def as_arrays(self) -> Mapping[str, GenericArray]:
        return MappingProxyType(
            {field.name: cast(GenericArray, getattr(self, field.name)) for field in fields(self)}
        )


def _neumaier_sum(values: GenericArray) -> float:
    total = 0.0
    correction = 0.0
    for raw in values:
        value = float(raw)
        updated = total + value
        if abs(total) >= abs(value):
            correction += (total - updated) + value
        else:
            correction += (value - updated) + total
        total = updated
    return total + correction


@dataclass(frozen=True, slots=True)
class TargetProvenance:
    """Immutable identity and selected-teacher binding fixed before certification."""

    source_kind: Literal[0, 1]
    source_key_hex: str
    source_unit_index: int
    teacher_sha256: str
    _provenance_authority: InitVar[object | None] = None

    def __post_init__(self, _provenance_authority: object | None) -> None:
        if _provenance_authority is not _PROVENANCE_AUTHORITY:
            raise ContractError("target provenance must be derived by a canonical source adapter")
        if type(self.source_kind) is not int or self.source_kind not in {0, 1}:
            raise ContractError("provenance source_kind must be integer zero or one")
        _sha256(self.source_key_hex, name="provenance source_key_hex")
        _sha256(self.teacher_sha256, name="provenance teacher_sha256")
        if type(self.source_unit_index) is not int or self.source_unit_index < 0:
            raise ContractError("provenance source_unit_index must be nonnegative")
        if self.source_kind == 0 and self.source_unit_index >= 7 * 96:
            raise ContractError("X0 provenance source_unit_index is outside the frozen inventory")


@dataclass(frozen=True, slots=True)
class CertifiedTarget:
    """The sole immutable result of ``certify_target``.

    A certified value owns the complete audit ledger and is the only value
    accepted by the target-artifact builder.  An abstention retains only its
    ordered reason vector and source kind; every target/audit array is the
    exact empty shape, so it cannot be serialized as a target artifact.
    """

    status: Literal["CERTIFIED", "ABSTAIN"]
    reasons: UInt16Array
    landmarks: Int32Array
    traversal_bounds: Int32Array
    seams: Float64Array
    pulse: UInt8Array
    chi: Float64Array
    target_mask: UInt8Array
    edge_mask: UInt8Array
    decoder_mask: UInt8Array
    canonical_phase: Float64Array
    winding: Int32Array
    source_kind: Literal[0, 1]
    provenance: TargetProvenance | None = None
    _certification_authority: InitVar[object | None] = None

    def __post_init__(self, _certification_authority: object | None) -> None:
        if _certification_authority is not _CERTIFICATION_AUTHORITY:
            raise ContractError("CertifiedTarget must be produced by certify_target")
        reasons = _set_readonly(self, "reasons", "<u2", (None,))
        landmarks = _set_readonly(self, "landmarks", "<i4", (None,))
        traversals = _set_readonly(self, "traversal_bounds", "<i4", (None, 2))
        seams = _set_readonly(self, "seams", "<f8", (traversals.shape[0],))
        pulse = _set_readonly(self, "pulse", "|u1", (None,))
        edge_count = pulse.shape[0]
        chi = _set_readonly(self, "chi", "<f8", (edge_count,))
        target = _set_readonly(self, "target_mask", "|u1", (edge_count,))
        edge = _set_readonly(self, "edge_mask", "|u1", (edge_count,))
        decoder = _set_readonly(self, "decoder_mask", "|u1", (edge_count,))
        canonical = _set_readonly(
            self,
            "canonical_phase",
            "<f8",
            (edge_count + 1 if edge_count else 0, 2),
        )
        winding = _set_readonly(self, "winding", "<i4", (traversals.shape[0],))
        for name, mask in (
            ("pulse", pulse),
            ("target_mask", target),
            ("edge_mask", edge),
            ("decoder_mask", decoder),
        ):
            _binary(mask, name=name)
        reason_values = tuple(int(value) for value in reasons)
        if reason_values != tuple(sorted(set(reason_values))) or any(
            value not in ReasonCode._value2member_map_ for value in reason_values
        ):
            raise ContractError("certification reasons must be known, unique, and ascending")
        if self.status not in {"CERTIFIED", "ABSTAIN"}:
            raise ContractError("certificate status must be CERTIFIED or ABSTAIN")
        if (self.status == "CERTIFIED") != (not reason_values):
            raise ContractError("CERTIFIED has no reasons; ABSTAIN has at least one")
        if type(self.source_kind) is not int or self.source_kind not in {0, 1}:
            raise ContractError("source_kind must be integer zero or one")
        if self.provenance is not None:
            if not isinstance(self.provenance, TargetProvenance):
                raise ContractError("certificate provenance must be an immutable TargetProvenance")
            if self.provenance.source_kind != self.source_kind:
                raise ContractError("certificate source kind differs from its frozen provenance")
        if self.status == "ABSTAIN":
            if any(
                array.size
                for array in (
                    landmarks,
                    traversals,
                    seams,
                    pulse,
                    chi,
                    target,
                    edge,
                    decoder,
                    canonical,
                    winding,
                )
            ):
                raise ContractError("ABSTAIN must not retain target or audit arrays")
            return
        if edge_count == 0 or traversals.shape[0] == 0:
            raise ContractError("CERTIFIED requires at least one edge and traversal")
        if not np.isfinite(chi).all() or np.any(chi < 0.0):
            raise ContractError("certified chi must be finite and nonnegative")
        if not np.isfinite(canonical).all() or not np.isfinite(seams).all():
            raise ContractError("certified phase and seam ledgers must be finite")
        _validate_bounds(traversals, limit=edge_count, name="traversal_bounds")
        expected_landmarks = np.unique(traversals.reshape(-1)).astype(np.int32, copy=False)
        if not np.array_equal(landmarks, expected_landmarks):
            raise ContractError("landmarks must be the ordered distinct traversal endpoints")
        expected_support = np.zeros(edge_count, dtype=np.uint8)
        for left, right in traversals:
            expected_support[int(left) : int(right)] = 1
        if not (
            np.array_equal(target, expected_support)
            and np.array_equal(edge, expected_support)
            and np.array_equal(decoder, expected_support)
        ):
            raise ContractError("all three certified masks must equal the traversal-edge union")
        starts = traversals[:, 0].astype(np.float64)
        if not np.array_equal(seams, starts):
            raise ContractError("the sole seam of every traversal must be its exact start")
        expected_pulse = np.zeros(edge_count, dtype=np.uint8)
        expected_pulse[traversals[:, 0]] = 1
        if not np.array_equal(pulse, expected_pulse):
            raise ContractError("certified pulse must own exactly one traversal-start seam")
        if np.any(chi[target == 0] != 0.0) or np.any(chi[pulse == 1] <= 0.0):
            raise ContractError("chi must be zero off-target and positive on every pulse")
        tolerance = 2.0 * float(np.spacing(np.float64(1.0)))
        for left, right in traversals:
            if abs(_neumaier_sum(chi[int(left) : int(right)]) - 1.0) > tolerance:
                raise ContractError("each traversal chi sum must equal one within two ulp")
        if not np.array_equal(winding, np.ones(traversals.shape[0], dtype=np.int32)):
            raise ContractError("every certified traversal must have winding integer one")

    @property
    def certified(self) -> bool:
        return self.status == "CERTIFIED"

    @property
    def event_count(self) -> int:
        return int(np.sum(self.pulse, dtype=np.int64))

    def sealed_copy(self) -> CertifiedTarget:
        """Reconstruct and fully revalidate this certificate before consumption."""

        return CertifiedTarget(
            status=self.status,
            reasons=self.reasons,
            landmarks=self.landmarks,
            traversal_bounds=self.traversal_bounds,
            seams=self.seams,
            pulse=self.pulse,
            chi=self.chi,
            target_mask=self.target_mask,
            edge_mask=self.edge_mask,
            decoder_mask=self.decoder_mask,
            canonical_phase=self.canonical_phase,
            winding=self.winding,
            source_kind=self.source_kind,
            provenance=self.provenance,
            _certification_authority=_CERTIFICATION_AUTHORITY,
        )

    def _artifact_arrays_unchecked(self) -> Mapping[str, GenericArray]:
        if not self.certified:
            raise ContractError("ABSTAIN has no target artifact")
        if self.provenance is None:
            raise ContractError("unbound certificate has no publishable target artifact")
        teacher_raw = bytes.fromhex(self.provenance.teacher_sha256)
        return MappingProxyType(
            {
                "chi": self.chi,
                "contract_sha256": np.frombuffer(_CONTRACT_RAW, dtype="|u1"),
                "edge_mask": self.edge_mask,
                "pulse": self.pulse,
                "source_kind": np.frombuffer(bytes((self.provenance.source_kind,)), dtype="|u1"),
                "target_mask": self.target_mask,
                "teacher_sha256": np.frombuffer(teacher_raw, dtype="|u1"),
            }
        )

    def artifact_arrays(self) -> Mapping[str, GenericArray]:
        """Return the exact seven target members, only for ``CERTIFIED``."""

        return self.sealed_copy()._artifact_arrays_unchecked()


def _validate_bounds(array: GenericArray, *, limit: int, name: str) -> None:
    if not array.size:
        return
    starts = array[:, 0]
    ends = array[:, 1]
    if (
        np.any(np.less(starts, 0))
        or np.any(np.greater(ends, limit))
        or np.any(np.greater_equal(starts, ends))
        or np.any(np.less(starts[1:], ends[:-1]))
    ):
        raise ContractError(f"{name} must be ordered, disjoint, positive half-open intervals")


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """The exact fourteen-array prediction/stub NPZ payload."""

    abstain: UInt8Array
    abstain_reasons: UInt16Array
    component_bounds: Int32Array
    component_location: Int32Array
    component_score: Float32Array
    condition: UInt8Array
    contract_sha256: UInt8Array
    count: Int64Array
    decoder_mask: UInt8Array
    edge_mask: UInt8Array
    local_person_slot: Int64Array
    opaque_sample_key: UInt8Array
    response: Float32Array
    run_bounds: Int32Array

    def __post_init__(self) -> None:
        abstain = _set_readonly(self, "abstain", "|u1", (1,))
        reasons = _set_readonly(self, "abstain_reasons", "<u2", (None,))
        components = _set_readonly(self, "component_bounds", "<i4", (None, 2))
        locations = _set_readonly(self, "component_location", "<i4", (components.shape[0],))
        scores = _set_readonly(self, "component_score", "<f4", (components.shape[0],))
        condition = _set_readonly(self, "condition", "|u1", (1,))
        contract_hash = _set_readonly(self, "contract_sha256", "|u1", (32,))
        count = _set_readonly(self, "count", "<i8", (1,))
        response = _set_readonly(self, "response", "<f4", (None,))
        edge_count = response.shape[0]
        decoder = _set_readonly(self, "decoder_mask", "|u1", (edge_count,))
        edge = _set_readonly(self, "edge_mask", "|u1", (edge_count,))
        slot = _set_readonly(self, "local_person_slot", "<i8", (1,))
        _set_readonly(self, "opaque_sample_key", "|u1", (32,))
        runs = _set_readonly(self, "run_bounds", "<i4", (None, 2))
        _binary(abstain, name="abstain")
        _binary(condition, name="condition")
        _binary(decoder, name="decoder_mask")
        _binary(edge, name="edge_mask")
        if int(condition[0]) not in {0, 1}:
            raise ContractError("condition must be natural-clean zero or natural-drift one")
        if contract_hash.tobytes() != _CONTRACT_RAW:
            raise ContractError("prediction contract_sha256 does not bind v4")
        if int(slot[0]) < 0:
            raise ContractError("local_person_slot must be nonnegative")
        if (
            not np.isfinite(response).all()
            or np.any(np.less(response, 0.0))
            or np.any(np.greater(response, 1.0))
        ):
            raise ContractError("response must contain finite float32 probabilities")
        if (
            not np.isfinite(scores).all()
            or np.any(np.less(scores, 0.0))
            or np.any(np.greater(scores, 1.0))
        ):
            raise ContractError("component scores must contain finite probabilities")
        if np.any(np.greater(decoder, edge)) or np.any(
            np.not_equal(response[np.equal(edge, 0)], np.float32(0.0))
        ):
            raise ContractError("prediction mask support or canonical outside-mask zero is invalid")
        reason_values = tuple(int(value) for value in reasons)
        if reason_values != tuple(sorted(set(reason_values))) or any(
            value not in ReasonCode._value2member_map_ for value in reason_values
        ):
            raise ContractError("abstain reasons must be known, unique, nonzero, and ascending")
        is_abstain = bool(int(abstain[0]))
        if is_abstain:
            if not reason_values:
                raise ContractError("an abstention must contain at least one reason")
            if int(count[0]) != -1:
                raise ContractError("an abstention count must be exactly -1")
            if edge_count or runs.shape[0] or components.shape[0]:
                raise ContractError("an abstention must use the exact empty stub shapes")
            return
        if reason_values:
            raise ContractError("a non-abstention must have an empty reason vector")
        if edge_count == 0 or runs.shape[0] == 0:
            raise ContractError("a non-abstention must contain edges and at least one run")
        if int(count[0]) != components.shape[0]:
            raise ContractError("non-abstention count must equal the number of components")
        _validate_bounds(runs, limit=edge_count, name="run_bounds")
        if runs.shape[0] > 1 and np.any(np.less_equal(runs[1:, 0], runs[:-1, 1])):
            raise ContractError("distinct runs must be separated by at least one edge")
        _validate_bounds(components, limit=edge_count, name="component_bounds")
        self._validate_components(response, decoder, runs, components, locations, scores)

    @staticmethod
    def _validate_components(
        response: GenericArray,
        decoder: GenericArray,
        runs: GenericArray,
        components: GenericArray,
        locations: GenericArray,
        scores: GenericArray,
    ) -> None:
        active = (decoder == 1) & (response >= np.float32(0.5))
        covered = np.zeros(response.shape, dtype=np.bool_)
        for run_start, run_end in runs:
            covered[int(run_start) : int(run_end)] = True
        if np.any(active & ~covered):
            raise ContractError("an active edge lies outside every identity run")
        expected: list[tuple[int, int]] = []
        for run_start, run_end in runs:
            start: int | None = None
            for edge_index in range(int(run_start), int(run_end)):
                if bool(active[edge_index]) and start is None:
                    start = edge_index
                if start is not None and (
                    not bool(active[edge_index]) or edge_index == int(run_end) - 1
                ):
                    end = edge_index if not bool(active[edge_index]) else edge_index + 1
                    expected.append((start, end))
                    start = None
        if expected != [tuple(map(int, row)) for row in components]:
            raise ContractError("component bounds are not the maximal active runs")
        for index, (start, end) in enumerate(expected):
            segment = response[start:end]
            maximum = np.max(segment)
            maxima = np.flatnonzero(segment == maximum) + start
            expected_location = (int(maxima[0]) + int(maxima[-1])) // 2
            if scores[index] != maximum or int(locations[index]) != expected_location:
                raise ContractError("component score or location is not the exact decoder value")

    def as_arrays(self) -> Mapping[str, GenericArray]:
        return MappingProxyType(
            {field.name: cast(GenericArray, getattr(self, field.name)) for field in fields(self)}
        )


@dataclass(frozen=True, slots=True)
class ResourceRecord:
    """Exact nested resource object used by a v4 run receipt."""

    completed_steps: int
    gpu_seconds: float
    max_cuda_bytes: int
    wall_seconds: float

    def __post_init__(self) -> None:
        if type(self.completed_steps) is not int or self.completed_steps < 0:
            raise ContractError("completed_steps must be a nonnegative integer")
        if type(self.max_cuda_bytes) is not int or self.max_cuda_bytes < 0:
            raise ContractError("max_cuda_bytes must be a nonnegative integer")
        for name in ("gpu_seconds", "wall_seconds"):
            value = getattr(self, name)
            if (
                type(value) is not float
                or not math.isfinite(value)
                or value < 0.0
                or _negative_zero(value)
            ):
                raise ContractError(f"{name} must be a finite nonnegative float64 value")

    def as_dict(self) -> dict[str, int | float]:
        return {
            "completed_steps": self.completed_steps,
            "gpu_seconds": self.gpu_seconds,
            "max_cuda_bytes": self.max_cuda_bytes,
            "wall_seconds": self.wall_seconds,
        }


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """One completely written checkpoint candidate and its resource snapshot."""

    completed_step: int
    artifact_sha256: str
    tune_objective: float
    resource: ResourceRecord

    def __post_init__(self) -> None:
        if type(self.completed_step) is not int or self.completed_step <= 0:
            raise ContractError("completed checkpoint step must be positive")
        if self.completed_step % 500 != 0:
            raise ContractError("checkpoint candidates exist only every 500 completed steps")
        _sha256(self.artifact_sha256, name="checkpoint artifact_sha256")
        if (
            type(self.tune_objective) is not float
            or not math.isfinite(self.tune_objective)
            or _negative_zero(self.tune_objective)
        ):
            raise ContractError("checkpoint tune objective must be finite float64")
        if self.resource.completed_steps != self.completed_step:
            raise ContractError("checkpoint resource step does not match the checkpoint")


def _ascii(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value or not value.isascii() or "\x00" in value:
        raise ContractError(f"{name} must be nonempty NUL-free ASCII")


@dataclass(frozen=True, slots=True)
class IdentityReceiptOwner:
    """Typed owner fields shared by feature and natural-input receipts."""

    split: str
    opaque_key_hex: str
    slot: int

    def __post_init__(self) -> None:
        if self.split not in {"train", "val"}:
            raise ContractError("identity receipt owner split must be train or val")
        _sha256(self.opaque_key_hex, name="identity receipt owner opaque key")
        if type(self.slot) is not int or not 0 <= self.slot < 2**64:
            raise ContractError("identity receipt owner slot must be uint64")


@dataclass(frozen=True, slots=True)
class CheckpointReceiptOwner:
    """Typed owner fields shared by teacher checkpoints and tune evaluations."""

    seed: int
    step: int

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ContractError("checkpoint receipt owner seed must be uint64")
        if type(self.step) is not int or not 0 <= self.step < 2**32:
            raise ContractError("checkpoint receipt owner step must be uint32")


@dataclass(frozen=True, slots=True)
class RunReceiptOwner:
    """Typed decoded job bytes and seed for a run receipt owner."""

    job_name: str
    seed: int

    def __post_init__(self) -> None:
        _ascii(self.job_name, name="run receipt owner job name")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ContractError("run receipt owner seed must be uint64")


@dataclass(frozen=True, slots=True)
class PredictionReceiptOwner:
    """Exact seven typed owner fields for one prediction receipt."""

    split: str
    opaque_key_hex: str
    slot: int
    arm: str
    seed: int
    condition: str

    def __post_init__(self) -> None:
        if self.split not in {"train", "val"}:
            raise ContractError("prediction receipt owner split must be train or val")
        _sha256(self.opaque_key_hex, name="prediction receipt owner opaque key")
        if type(self.slot) is not int or not 0 <= self.slot < 2**64:
            raise ContractError("prediction receipt owner slot must be uint64")
        if self.arm not in {"local", "global", "uniform", "capacity-control"}:
            raise ContractError("prediction receipt owner arm is outside the frozen domain")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ContractError("prediction receipt owner seed must be uint64")
        if self.condition not in {"natural-clean", "natural-drift"}:
            raise ContractError("prediction receipt owner condition is outside the frozen domain")


@dataclass(frozen=True, slots=True)
class TargetReceiptOwner:
    """Typed source-kind/key/unit owner fields for a certified target receipt."""

    source_kind: int
    source_key_hex: str
    source_unit_index: int

    def __post_init__(self) -> None:
        if type(self.source_kind) is not int or self.source_kind not in {0, 1}:
            raise ContractError("target receipt owner source_kind must be zero or one")
        _sha256(self.source_key_hex, name="target receipt owner source key")
        if type(self.source_unit_index) is not int or not 0 <= self.source_unit_index < 2**64:
            raise ContractError("target receipt owner source_unit_index must be uint64")
        if self.source_kind == 0 and self.source_unit_index >= 7 * 96:
            raise ContractError("X0 target receipt owner unit index is outside the inventory")


@dataclass(frozen=True, slots=True)
class SeedReceiptOwner:
    """Typed seed field for X0-inference and NATP receipt owners."""

    seed: int

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ContractError("seed receipt owner seed must be uint64")


@dataclass(frozen=True, slots=True)
class NamedReceiptOwner:
    """Typed exact roster token for stage/G1/K1/K3/K4 receipt owners."""

    owner: str

    def __post_init__(self) -> None:
        _ascii(self.owner, name="named receipt owner")


@dataclass(frozen=True, slots=True)
class SchemaReceiptOwner:
    """Typed exact schema token for tune-input and teacher-selection owners."""

    schema: str

    def __post_init__(self) -> None:
        _ascii(self.schema, name="schema receipt owner")


ReceiptOwner: TypeAlias = (
    IdentityReceiptOwner
    | CheckpointReceiptOwner
    | RunReceiptOwner
    | PredictionReceiptOwner
    | TargetReceiptOwner
    | SeedReceiptOwner
    | NamedReceiptOwner
    | SchemaReceiptOwner
)


@dataclass(frozen=True, slots=True)
class EffectiveContractRow:
    """One immutable row of the accepted five-document effective contract."""

    bytes: int
    role: str
    sha256: str

    def __post_init__(self) -> None:
        if type(self.bytes) is not int or self.bytes <= 0:
            raise ContractError("effective-contract bytes must be a positive integer")
        _ascii(self.role, name="effective-contract role")
        _sha256(self.sha256, name="effective-contract sha256")

    def as_dict(self) -> dict[str, object]:
        return {"bytes": self.bytes, "role": self.role, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class PreG5AInventoryRow:
    """One exact owner/receipt row in the 54-owner pre-G5a inventory."""

    node_class: str
    owner: str
    receipt_schema: str
    receipt_sha256: str
    upstream_owner_tokens: tuple[str, ...]
    upstream_receipt_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        _ascii(self.node_class, name="inventory node_class")
        _ascii(self.owner, name="inventory owner")
        _ascii(self.receipt_schema, name="inventory receipt_schema")
        _sha256(self.receipt_sha256, name="inventory receipt_sha256")
        if (
            type(self.upstream_owner_tokens) is not tuple
            or type(self.upstream_receipt_sha256) is not tuple
        ):
            raise ContractError("inventory upstream fields must be immutable tuples")
        if len(self.upstream_owner_tokens) != len(self.upstream_receipt_sha256):
            raise ContractError("inventory upstream owner and digest arrays differ in length")
        for owner in self.upstream_owner_tokens:
            _ascii(owner, name="inventory upstream owner")
        for digest in self.upstream_receipt_sha256:
            _sha256(digest, name="inventory upstream receipt sha256")

    def as_dict(self) -> dict[str, object]:
        return {
            "node_class": self.node_class,
            "owner": self.owner,
            "receipt_schema": self.receipt_schema,
            "receipt_sha256": self.receipt_sha256,
            "upstream_owner_tokens": list(self.upstream_owner_tokens),
            "upstream_receipt_sha256": list(self.upstream_receipt_sha256),
        }


@dataclass(frozen=True, slots=True)
class ReceiptDagNode:
    """One closed typed node in ``temporac.receipt-dag.v4``."""

    node_class: str
    owner_key: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        _ascii(self.node_class, name="receipt DAG node class")
        _sha256(self.owner_key, name="receipt DAG owner_key")
        _sha256(self.receipt_sha256, name="receipt DAG receipt_sha256")

    def as_dict(self) -> dict[str, object]:
        return {
            "class": self.node_class,
            "owner_key": self.owner_key,
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class ReceiptDagEdge:
    """One ordinal, directional dependency edge in the receipt DAG."""

    from_receipt_sha256: str
    ordinal: int
    role: str
    to_receipt_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.from_receipt_sha256, name="receipt DAG from digest")
        _sha256(self.to_receipt_sha256, name="receipt DAG to digest")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ContractError("receipt DAG edge ordinal must be a nonnegative integer")
        _ascii(self.role, name="receipt DAG edge role")

    def as_dict(self) -> dict[str, object]:
        return {
            "from_receipt_sha256": self.from_receipt_sha256,
            "ordinal": self.ordinal,
            "role": self.role,
            "to_receipt_sha256": self.to_receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class PreG5AReceiptInventory:
    """Validated immutable 54-row inventory and its 106 direct edges."""

    contract_sha256: str
    rows: tuple[PreG5AInventoryRow, ...]
    direct_edges: tuple[ReceiptDagEdge, ...]
    topological_receipt_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        _sha256(self.contract_sha256, name="inventory contract_sha256")
        if any(type(value) is not tuple for value in (self.rows, self.direct_edges)):
            raise ContractError("inventory rows and edges must be immutable tuples")
        if type(self.topological_receipt_sha256) is not tuple:
            raise ContractError("inventory topological order must be an immutable tuple")
        if len(self.rows) != 54 or len(self.direct_edges) != 106:
            raise ContractError("inventory must contain exactly 54 rows and 106 direct edges")
        if len(self.topological_receipt_sha256) != 54:
            raise ContractError("inventory topological order must contain every receipt")


@dataclass(frozen=True, slots=True)
class ReceiptDag:
    """A validated closed receipt DAG with a dependency-first topological order."""

    contract_sha256: str
    nodes: tuple[ReceiptDagNode, ...]
    edges: tuple[ReceiptDagEdge, ...]
    topological_receipt_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        _sha256(self.contract_sha256, name="receipt DAG contract_sha256")
        if any(type(value) is not tuple for value in (self.nodes, self.edges)):
            raise ContractError("receipt DAG nodes and edges must be immutable tuples")
        if type(self.topological_receipt_sha256) is not tuple:
            raise ContractError("receipt DAG topological order must be an immutable tuple")
        if len(self.nodes) != len(self.topological_receipt_sha256):
            raise ContractError("receipt DAG topological order must contain every node")


@dataclass(frozen=True, slots=True)
class NaturalPredictionCompletionReceipt:
    """Typed exact-14-key NATP receipt; it confers no metric or gate authority."""

    clean_seed_index_sha256: str
    clean_seed_root_sha256: str
    code_index_sha256: str
    contract_sha256: str
    drift_seed_index_sha256: str
    drift_seed_root_sha256: str
    environment_sha256: str
    k6_receipt_sha256: str
    owner: str
    pilot_scope_manifest_sha256: str
    seed: int
    status: str
    upstream_receipt_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "clean_seed_index_sha256",
            "clean_seed_root_sha256",
            "code_index_sha256",
            "contract_sha256",
            "drift_seed_index_sha256",
            "drift_seed_root_sha256",
            "environment_sha256",
            "k6_receipt_sha256",
            "pilot_scope_manifest_sha256",
        ):
            _sha256(cast(str, getattr(self, name)), name=name)
        _ascii(self.owner, name="NATP owner")
        if type(self.seed) is not int:
            raise ContractError("NATP seed must be an integer")
        if self.status != "PASS":
            raise ContractError("NATP status must be exact ASCII PASS")
        if (
            type(self.upstream_receipt_sha256) is not tuple
            or len(self.upstream_receipt_sha256) != 1
        ):
            raise ContractError("NATP upstream must be an immutable length-one tuple")
        _sha256(self.upstream_receipt_sha256[0], name="NATP upstream receipt sha256")
        if self.upstream_receipt_sha256[0] != self.k6_receipt_sha256:
            raise ContractError("NATP upstream digest must equal k6_receipt_sha256")

    def as_dict(self, *, schema: str) -> dict[str, object]:
        return {
            "clean_seed_index_sha256": self.clean_seed_index_sha256,
            "clean_seed_root_sha256": self.clean_seed_root_sha256,
            "code_index_sha256": self.code_index_sha256,
            "contract_sha256": self.contract_sha256,
            "drift_seed_index_sha256": self.drift_seed_index_sha256,
            "drift_seed_root_sha256": self.drift_seed_root_sha256,
            "environment_sha256": self.environment_sha256,
            "k6_receipt_sha256": self.k6_receipt_sha256,
            "owner": self.owner,
            "pilot_scope_manifest_sha256": self.pilot_scope_manifest_sha256,
            "schema": schema,
            "seed": self.seed,
            "status": self.status,
            "upstream_receipt_sha256": list(self.upstream_receipt_sha256),
        }


@dataclass(frozen=True, slots=True)
class K1OutcomeRow:
    """One immutable exact-13-key K1 outcome row."""

    certificate_reasons: tuple[int, ...]
    certificate_status: str
    component_key_hex: str
    eligible: bool
    feature_artifact_sha256_or_null: str | None
    feature_receipt_sha256_or_null: str | None
    natural_input_receipt_sha256_or_null: str | None
    opaque_key_hex: str
    population_row_sha256: str
    slot: int
    split: str
    target_artifact_sha256_or_null: str | None
    target_receipt_sha256_or_null: str | None

    def __post_init__(self) -> None:
        if type(self.certificate_reasons) is not tuple:
            raise ContractError("K1 certificate reasons must be an immutable tuple")
        reasons = self.certificate_reasons
        if any(type(value) is not int or not 1 <= value <= 24 for value in reasons):
            raise ContractError("K1 certificate reasons must be uint16 contract codes 1..24")
        if tuple(sorted(set(reasons))) != reasons:
            raise ContractError("K1 certificate reasons must be unique and ascending")
        if self.certificate_status not in {"NOT_EVALUATED", "ABSTAIN", "CERTIFIED"}:
            raise ContractError("K1 certificate_status is invalid")
        _sha256(self.component_key_hex, name="K1 component_key_hex")
        _sha256(self.opaque_key_hex, name="K1 opaque_key_hex")
        _sha256(self.population_row_sha256, name="K1 population_row_sha256")
        if type(self.eligible) is not bool:
            raise ContractError("K1 eligible must be a JSON boolean")
        if type(self.slot) is not int or self.slot < 0:
            raise ContractError("K1 slot must be a nonnegative integer")
        if self.split not in {"train", "val"}:
            raise ContractError("K1 split must be exact ASCII train or val")
        nullable = (
            self.feature_artifact_sha256_or_null,
            self.feature_receipt_sha256_or_null,
            self.natural_input_receipt_sha256_or_null,
            self.target_artifact_sha256_or_null,
            self.target_receipt_sha256_or_null,
        )
        for index, value in enumerate(nullable):
            if value is not None:
                _sha256(value, name=f"K1 nullable digest {index}")
        if (self.feature_artifact_sha256_or_null is None) != (
            self.feature_receipt_sha256_or_null is None
        ):
            raise ContractError("K1 feature artifact/receipt nullability must agree")
        target_present = self.target_artifact_sha256_or_null is not None
        if target_present != (self.target_receipt_sha256_or_null is not None):
            raise ContractError("K1 target artifact/receipt nullability must agree")
        if not self.eligible:
            if self.certificate_status != "NOT_EVALUATED" or not reasons:
                raise ContractError("ineligible K1 rows must be NOT_EVALUATED with reasons")
            if self.natural_input_receipt_sha256_or_null is not None or target_present:
                raise ContractError("ineligible K1 rows cannot bind natural input or target")
        elif self.certificate_status == "CERTIFIED":
            if reasons or self.natural_input_receipt_sha256_or_null is None or not target_present:
                raise ContractError(
                    "CERTIFIED K1 rows require natural input and target, no reasons"
                )
        elif self.certificate_status == "ABSTAIN":
            if not reasons or target_present:
                raise ContractError("ABSTAIN K1 rows require reasons and no target")
        else:
            raise ContractError("eligible K1 rows cannot be NOT_EVALUATED")

    def as_dict(self) -> dict[str, object]:
        return {
            "certificate_reasons": list(self.certificate_reasons),
            "certificate_status": self.certificate_status,
            "component_key_hex": self.component_key_hex,
            "eligible": self.eligible,
            "feature_artifact_sha256_or_null": self.feature_artifact_sha256_or_null,
            "feature_receipt_sha256_or_null": self.feature_receipt_sha256_or_null,
            "natural_input_receipt_sha256_or_null": self.natural_input_receipt_sha256_or_null,
            "opaque_key_hex": self.opaque_key_hex,
            "population_row_sha256": self.population_row_sha256,
            "slot": self.slot,
            "split": self.split,
            "target_artifact_sha256_or_null": self.target_artifact_sha256_or_null,
            "target_receipt_sha256_or_null": self.target_receipt_sha256_or_null,
        }


@dataclass(frozen=True, slots=True)
class K1OutcomeIndex:
    """Validated immutable 402-row K1 outcome index."""

    contract_sha256: str
    population_manifest_sha256: str
    rows: tuple[K1OutcomeRow, ...]
    row_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        _sha256(self.contract_sha256, name="K1 index contract_sha256")
        _sha256(self.population_manifest_sha256, name="K1 population manifest sha256")
        if type(self.rows) is not tuple or type(self.row_sha256) is not tuple:
            raise ContractError("K1 rows and row hashes must be immutable tuples")
        if len(self.rows) != 402 or len(self.row_sha256) != 402:
            raise ContractError("K1 outcome index must contain exactly 402 rows and hashes")
        for digest in self.row_sha256:
            _sha256(digest, name="K1 standalone row sha256")


@dataclass(frozen=True, slots=True)
class OperatorCandidateMember:
    """One immutable frozen operator-candidate member identity."""

    name: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _ascii(self.name, name="operator candidate member name")
        if "/" in self.name or "\\" in self.name:
            raise ContractError("operator candidate member name must be flat")
        if type(self.bytes) is not int or self.bytes <= 0:
            raise ContractError("operator candidate member bytes must be positive")
        _sha256(self.sha256, name="operator candidate member sha256")


@dataclass(frozen=True, slots=True)
class FrozenOperatorCandidate:
    """Read-only candidate/review binding eligible only as a future P2 input."""

    members: tuple[OperatorCandidateMember, ...]
    candidate_receipt_sha256: str
    generation_snapshot_sha256: str
    review_markdown_sha256: str
    review_json_sha256: str
    validated_row_count: int
    candidate_only: bool
    eligible_for_separate_p2: bool
    authority: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        if type(self.members) is not tuple or len(self.members) != 5:
            raise ContractError("operator candidate handle must contain exactly five members")
        for name in (
            "candidate_receipt_sha256",
            "generation_snapshot_sha256",
            "review_markdown_sha256",
            "review_json_sha256",
        ):
            _sha256(cast(str, getattr(self, name)), name=name)
        if self.validated_row_count != 10_368:
            raise ContractError("operator candidate row count must be exactly 10,368")
        if self.candidate_only is not True or self.eligible_for_separate_p2 is not True:
            raise ContractError("operator candidate handle has invalid candidate-only disposition")
        if type(self.authority) is not tuple or any(
            type(name) is not str or type(value) is not bool or value
            for name, value in self.authority
        ):
            raise ContractError("operator candidate handle must carry only false authority bits")
