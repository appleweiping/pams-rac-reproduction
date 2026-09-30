"""Feature-only WARP-PHASE reader, duplicate collapse, and padded collation.

This training-side module deliberately has no pickle, vault, evaluator, or
packer import.  It consumes only the seven-field non-executable NPZ contract.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, cast

import numpy as np
from numpy.typing import NDArray

from pams.warp_phase.types import (
    FEATURE_FIELD_NAMES,
    ArrayMemberReceipt,
    CollapsedTrack,
    EligibilityAssessment,
    EligibilityDecision,
    EligibilityInput,
    FeatureBatch,
    FeatureShard,
    PilotSplit,
    WarpPhaseContractError,
)

_EXPECTED_MEMBER_NAMES = tuple(f"{field}.npy" for field in FEATURE_FIELD_NAMES)
_EXPECTED_DTYPE = {
    "frame_mask": "|u1",
    "local_person_slot": "<i8",
    "motion": "<f4",
    "opaque_sample_key": "|S64",
    "person_mask": "|u1",
    "sampled_frame_indices": "<i8",
    "source_length": "<i8",
}
_EXPECTED_SHAPE = {
    "frame_mask": (320,),
    "local_person_slot": (1,),
    "motion": (320, 17, 3),
    "opaque_sample_key": (1,),
    "person_mask": (1,),
    "sampled_frame_indices": (320,),
    "source_length": (1,),
}
_OPAQUE_KEY = re.compile(r"[0-9a-f]{64}\Z")
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_SCHEMA_FIXTURE_BYTE_COUNT = 617
_SCHEMA_FIXTURE_SHA256 = "31f81549f9c5dbedc6ddb278e868793246826a234de90c06c169c71387428275"
_EXECUTABLE_SUFFIXES = {
    ".bat",
    ".cmd",
    ".com",
    ".dll",
    ".exe",
    ".js",
    ".pkl",
    ".pickle",
    ".ps1",
    ".py",
    ".sh",
    ".so",
}


class FeatureDataError(WarpPhaseContractError):
    """Base exception for a feature-only data contract failure."""


class FeatureArchiveError(FeatureDataError):
    """Raised before NumPy loading when ZIP/NPY structure is unsafe or noncanonical."""


class FeatureArrayError(FeatureDataError):
    """Raised when loaded arrays violate the exact seven-field schema."""


class DuplicateClockError(FeatureDataError):
    """Raised when duplicate-clock collapse cannot produce a coherent track."""


class NormalizationScaleError(FeatureDataError):
    """Raised when no positive finite COCO17 shoulder-to-hip scale exists."""


class EligibilityInputError(FeatureDataError):
    """Raised when an eligibility input is malformed rather than merely ineligible."""


@dataclass(frozen=True, slots=True)
class _InspectedArchive:
    payload: bytes
    sha256: str
    members: tuple[ArrayMemberReceipt, ...]


def _normalize_split(split: str) -> PilotSplit:
    normalized = split.strip().lower()
    if normalized not in {"train", "val"}:
        raise FeatureDataError("feature split must be exactly train or val")
    return cast(PilotSplit, normalized)


def _read_archive_bytes(path: Path) -> bytes:
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"feature shard does not exist: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise FeatureArchiveError("feature shard must be a regular non-symlink file")
    if before.st_size > _MAX_ARCHIVE_BYTES:
        raise FeatureArchiveError("feature archive exceeds the fixed size limit")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read(_MAX_ARCHIVE_BYTES + 1)
            closed = os.fstat(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after = path.lstat()
    if len(payload) > _MAX_ARCHIVE_BYTES:
        raise FeatureArchiveError("feature archive exceeds the fixed size limit")
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field) != getattr(opened, field)
        or getattr(opened, field) != getattr(closed, field)
        or getattr(closed, field) != getattr(after, field)
        for field in stable_fields
    ):
        raise FeatureArchiveError("feature archive changed while it was being read")
    if len(payload) != after.st_size:
        raise FeatureArchiveError("feature archive byte count changed while it was read")
    return payload


def _read_npy_header(handle: BinaryIO) -> tuple[tuple[int, ...], bool, np.dtype[np.generic]]:
    version = np.lib.format.read_magic(handle)
    if version != (2, 0):
        raise FeatureArchiveError("feature members must use NPY version 2.0")
    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
    return tuple(int(value) for value in shape), bool(fortran_order), np.dtype(dtype)


def _inspect_archive_bytes(payload: bytes) -> _InspectedArchive:
    members: list[ArrayMemberReceipt] = []
    total_uncompressed = 0
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            if archive.comment != b"":
                raise FeatureArchiveError("feature archive comment must be empty")
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise FeatureArchiveError("duplicate ZIP member names are forbidden")
            if names != sorted(_EXPECTED_MEMBER_NAMES):
                raise FeatureArchiveError(
                    "feature ZIP members must be the exact sorted seven-field whitelist"
                )
            for info in infos:
                name = info.filename
                pure_name = PurePosixPath(name)
                if (
                    not name
                    or "\\" in name
                    or pure_name.is_absolute()
                    or len(pure_name.parts) != 1
                    or any(part in {"", ".", ".."} for part in pure_name.parts)
                    or pure_name.suffix.casefold() != ".npy"
                ):
                    raise FeatureArchiveError(f"unsafe ZIP member name: {name!r}")
                if any(name.casefold().endswith(suffix) for suffix in _EXECUTABLE_SUFFIXES):
                    raise FeatureArchiveError(f"executable or pickle ZIP member is forbidden: {name}")
                unix_mode = info.external_attr >> 16
                if stat.S_IFMT(unix_mode) != stat.S_IFREG or unix_mode & 0o111:
                    raise FeatureArchiveError("ZIP members must be non-executable regular files")
                if (
                    info.date_time != _ZIP_TIMESTAMP
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.extra != b""
                    or info.comment != b""
                    or info.flag_bits & 0x1
                ):
                    raise FeatureArchiveError("ZIP member metadata is not the frozen form")
                total_uncompressed += info.file_size
                if total_uncompressed > _MAX_UNCOMPRESSED_BYTES:
                    raise FeatureArchiveError("feature members exceed the uncompressed size limit")
                member_bytes = archive.read(info)
                if len(member_bytes) != info.file_size:
                    raise FeatureArchiveError("ZIP member size changed during inspection")
                with io.BytesIO(member_bytes) as member_handle:
                    shape, fortran_order, dtype = _read_npy_header(member_handle)
                field = name.removesuffix(".npy")
                if fortran_order:
                    raise FeatureArchiveError("Fortran-order feature arrays are forbidden")
                if dtype.hasobject or dtype.kind in {"O", "U", "V"}:
                    raise FeatureArchiveError("object, Unicode, and void feature dtypes are forbidden")
                if dtype.str != _EXPECTED_DTYPE[field] or shape != _EXPECTED_SHAPE[field]:
                    raise FeatureArchiveError(
                        f"feature member {field!r} has a noncanonical dtype or shape"
                    )
                members.append(
                    ArrayMemberReceipt(
                        name=name,
                        dtype=dtype.str,
                        shape=shape,
                        sha256=hashlib.sha256(member_bytes).hexdigest(),
                        byte_count=len(member_bytes),
                    )
                )
    except zipfile.BadZipFile as exc:
        raise FeatureArchiveError("feature shard is not a valid ZIP archive") from exc
    return _InspectedArchive(
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        members=tuple(members),
    )


def inspect_feature_archive(path: Path) -> tuple[ArrayMemberReceipt, ...]:
    """Inspect ZIP names, metadata, and NPY headers without loading array payloads."""

    inspected = _inspect_archive_bytes(_read_archive_bytes(path))
    return inspected.members


def load_feature_schema_view(payload: bytes) -> dict[str, str]:
    """Verify the normative fixture and expose only its feature-side view."""

    if (
        len(payload) != _SCHEMA_FIXTURE_BYTE_COUNT
        or hashlib.sha256(payload).hexdigest() != _SCHEMA_FIXTURE_SHA256
    ):
        raise FeatureArrayError("schema fixture byte/hash receipt does not match")
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict) or set(decoded) != {
        "annotation",
        "evaluator_vault",
        "feature_shard",
    }:
        raise FeatureArrayError("schema fixture root does not match the frozen contract")
    feature_view = decoded["feature_shard"]
    if not isinstance(feature_view, dict) or set(feature_view) != set(FEATURE_FIELD_NAMES):
        raise FeatureArrayError("schema fixture feature view is not the seven-field whitelist")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in feature_view.items()):
        raise FeatureArrayError("schema fixture feature entries must be strings")
    return cast(dict[str, str], dict(feature_view))


def _require_binary_mask(array: NDArray[np.generic], *, name: str) -> None:
    if array.dtype != np.dtype("|u1") or not np.isin(array, (0, 1)).all():
        raise FeatureArrayError(f"{name} must be an exact uint8 binary mask")


def _decode_opaque_key(array: NDArray[np.generic]) -> str:
    value = array[0]
    if not isinstance(value, np.bytes_):
        raise FeatureArrayError("opaque_sample_key must use fixed bytes, not Unicode")
    try:
        key = bytes(value).decode("ascii")
    except UnicodeDecodeError as exc:
        raise FeatureArrayError("opaque_sample_key is not ASCII") from exc
    if _OPAQUE_KEY.fullmatch(key) is None:
        raise FeatureArrayError("opaque_sample_key must be 64 lowercase hexadecimal bytes")
    return key


def _feature_from_arrays(
    arrays: dict[str, NDArray[np.generic]],
    *,
    shard_sha256: str,
    path: Path,
) -> FeatureShard:
    if tuple(sorted(arrays)) != FEATURE_FIELD_NAMES:
        raise FeatureArrayError("loaded NPZ does not contain the exact seven feature fields")
    motion = arrays["motion"]
    frame_mask = arrays["frame_mask"]
    person_mask = arrays["person_mask"]
    clocks = arrays["sampled_frame_indices"]
    source_length_array = arrays["source_length"]
    slot_array = arrays["local_person_slot"]
    if motion.dtype.str != "<f4" or not motion.flags.c_contiguous:
        raise FeatureArrayError("motion must be C-order little-endian float32")
    if not np.isfinite(motion).all():
        raise FeatureArrayError("motion contains non-finite values")
    _require_binary_mask(frame_mask, name="frame_mask")
    _require_binary_mask(person_mask, name="person_mask")
    if int(person_mask[0]) != 1:
        raise FeatureArrayError("feature identities must be person-valid")
    if clocks.dtype.str != "<i8" or np.any(np.diff(clocks) < 0):
        raise FeatureArrayError("sampled_frame_indices must be nondecreasing <i8")
    if source_length_array.dtype.str != "<i8" or slot_array.dtype.str != "<i8":
        raise FeatureArrayError("source_length and local_person_slot must be <i8")
    source_length = int(source_length_array[0])
    slot = int(slot_array[0])
    int64_clocks = cast(NDArray[np.int64], clocks)
    if source_length <= 1 or slot < 0:
        raise FeatureArrayError("source_length and local_person_slot are outside their domain")
    if (
        int(clocks[0]) != 0
        or int(clocks[-1]) != source_length - 1
        or np.any(int64_clocks < 0)
        or np.any(int64_clocks >= source_length)
    ):
        raise FeatureArrayError("sampled_frame_indices do not bind the source clock domain")
    opaque_key = _decode_opaque_key(arrays["opaque_sample_key"])
    if path.name != f"{opaque_key}.{slot}.npz":
        raise FeatureArrayError("feature filename does not match its opaque key and local slot")
    typed_motion = np.array(motion, dtype="<f4", order="C", copy=True)
    typed_frame = np.array(frame_mask, dtype="|u1", order="C", copy=True)
    typed_clocks = np.array(clocks, dtype="<i8", order="C", copy=True)
    typed_motion.flags.writeable = False
    typed_frame.flags.writeable = False
    typed_clocks.flags.writeable = False
    return FeatureShard(
        motion=typed_motion,
        person_mask=True,
        frame_mask=typed_frame,
        sampled_frame_indices=typed_clocks,
        source_length=source_length,
        opaque_sample_key=opaque_key,
        local_person_slot=slot,
        shard_sha256=shard_sha256,
    )


def load_feature_shard(path: Path) -> FeatureShard:
    """Load one inspected shard using NumPy's non-pickle path only."""

    shard_path = Path(path)
    inspected = _inspect_archive_bytes(_read_archive_bytes(shard_path))
    arrays: dict[str, NDArray[np.generic]] = {}
    # Inspection above rejects duplicate/path-traversing/executable members and
    # object/Unicode dtypes before this NumPy call is reached.
    with np.load(io.BytesIO(inspected.payload), allow_pickle=False) as archive:
        if tuple(sorted(archive.files)) != FEATURE_FIELD_NAMES:
            raise FeatureArrayError("NumPy member names differ from inspected ZIP names")
        for field in FEATURE_FIELD_NAMES:
            arrays[field] = np.array(archive[field], copy=True, order="C")
    return _feature_from_arrays(
        arrays,
        shard_sha256=inspected.sha256,
        path=shard_path,
    )


def load_feature_split(features_root: Path, split: str) -> tuple[FeatureShard, ...]:
    """Load a deterministic split from a root whose basename is exactly ``features``."""

    approved_split = _normalize_split(split)
    root = Path(features_root)
    if root.name != "features" or root.name.casefold() in {"vault", "audit"}:
        raise FeatureDataError("training accepts only a root named features")
    try:
        root_metadata = root.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"feature root does not exist: {root}") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise FeatureDataError("feature root must be a regular non-symlink directory")
    split_root = root / approved_split
    split_metadata = split_root.lstat()
    if stat.S_ISLNK(split_metadata.st_mode) or not stat.S_ISDIR(split_metadata.st_mode):
        raise FeatureDataError("feature split must be a regular non-symlink directory")
    entries = sorted(split_root.iterdir(), key=lambda item: item.name.encode("utf-8"))
    if not entries:
        raise FeatureDataError(f"feature split is empty: {approved_split}")
    if any(entry.suffix.casefold() != ".npz" for entry in entries):
        raise FeatureDataError("feature split directories may contain only NPZ shards")
    return tuple(load_feature_shard(entry) for entry in entries)


def _source_joint_mask(feature: FeatureShard) -> NDArray[np.bool_]:
    raw = np.asarray(feature.motion)
    if raw.ndim != 3 or raw.shape[1:] != (17, 3):
        raise DuplicateClockError("motion must have shape [T,17,3]")
    if feature.frame_mask.shape != (raw.shape[0],):
        raise DuplicateClockError("frame_mask length does not match motion")
    if feature.sampled_frame_indices.shape != (raw.shape[0],):
        raise DuplicateClockError("clock length does not match motion")
    if np.any(np.diff(feature.sampled_frame_indices) < 0):
        raise DuplicateClockError("source clocks must be nondecreasing before collapse")
    return (
        feature.frame_mask.astype(np.bool_)[:, None]
        & np.isfinite(raw).all(axis=2)
        & (raw[:, :, 2] >= np.float32(0.20))
    )


def _roots_and_scale(
    raw: NDArray[np.float64],
    source_joint_mask: NDArray[np.bool_],
    representative_indices: Sequence[int],
) -> tuple[NDArray[np.float64], NDArray[np.bool_], float]:
    roots = np.zeros((raw.shape[0], 2), dtype=np.float64)
    root_valid = np.zeros(raw.shape[0], dtype=np.bool_)
    for index in range(raw.shape[0]):
        hips = [joint for joint in (11, 12) if source_joint_mask[index, joint]]
        if hips:
            roots[index] = np.mean(raw[index, hips, :2], axis=0, dtype=np.float64)
            root_valid[index] = True
    scales: list[float] = []
    for index in representative_indices:
        if not root_valid[index]:
            continue
        shoulders = [joint for joint in (5, 6) if source_joint_mask[index, joint]]
        if not shoulders:
            continue
        shoulder = np.mean(raw[index, shoulders, :2], axis=0, dtype=np.float64)
        distance = float(np.linalg.norm(shoulder - roots[index]))
        if math.isfinite(distance) and distance > 0.0:
            scales.append(distance)
    if not scales:
        raise NormalizationScaleError(
            "track has no positive finite shoulder-to-hip scale after duplicate collapse"
        )
    scale = max(float(np.median(np.asarray(scales, dtype=np.float64))), 1e-3)
    if not math.isfinite(scale) or scale <= 0.0:
        raise NormalizationScaleError("normalization scale is not positive and finite")
    return roots, root_valid, scale


def _duplicate_group_agrees(
    indices: NDArray[np.int64],
    *,
    raw: NDArray[np.float64],
    original_frame_mask: NDArray[np.bool_],
    source_joint_mask: NDArray[np.bool_],
    roots: NDArray[np.float64],
    root_valid: NDArray[np.bool_],
    scale: float,
) -> bool:
    reference = int(indices[0])
    reference_mask = source_joint_mask[reference]
    for raw_index in indices[1:]:
        index = int(raw_index)
        if original_frame_mask[index] != original_frame_mask[reference]:
            return False
        if not np.array_equal(source_joint_mask[index], reference_mask):
            return False
        if root_valid[index] != root_valid[reference]:
            return False
        valid = reference_mask & root_valid[reference]
        if np.any(valid):
            reference_xy = (raw[reference, valid, :2] - roots[reference]) / scale
            candidate_xy = (raw[index, valid, :2] - roots[index]) / scale
            if float(np.max(np.abs(reference_xy - candidate_xy))) > 1e-5:
                return False
        if float(np.max(np.abs(raw[index, :, 2] - raw[reference, :, 2]))) > 1e-6:
            return False
    return True


def collapse_duplicate_clocks(feature: FeatureShard) -> CollapsedTrack:
    """Collapse exact duplicate clocks and invalidate every conflicting clock.

    A conflict is never averaged.  Scale-dependent agreement is evaluated
    conservatively: once a clock conflicts it is not reintroduced if removing
    it changes the final collapsed-track median scale.
    """

    if not feature.person_mask:
        raise DuplicateClockError("person-invalid tracks cannot be collapsed")
    raw32 = np.asarray(feature.motion)
    if raw32.dtype != np.dtype("<f4") or not np.isfinite(raw32).all():
        raise DuplicateClockError("motion must be finite little-endian float32")
    raw = np.asarray(raw32, dtype=np.float64)
    original_frame_mask = np.asarray(feature.frame_mask, dtype=np.bool_)
    source_joint_mask = _source_joint_mask(feature)
    clocks = np.asarray(feature.sampled_frame_indices, dtype=np.int64)
    unique_clocks, first_indices, inverse = np.unique(
        clocks,
        return_index=True,
        return_inverse=True,
    )
    groups = [
        np.flatnonzero(inverse == group_index).astype(np.int64, copy=False)
        for group_index in range(unique_clocks.size)
    ]
    active = list(range(len(groups)))
    conflict_clocks: set[int] = set()
    while True:
        representatives = [int(first_indices[group_index]) for group_index in active]
        roots, root_valid, scale = _roots_and_scale(raw, source_joint_mask, representatives)
        newly_conflicted = [
            group_index
            for group_index in active
            if not _duplicate_group_agrees(
                groups[group_index],
                raw=raw,
                original_frame_mask=original_frame_mask,
                source_joint_mask=source_joint_mask,
                roots=roots,
                root_valid=root_valid,
                scale=scale,
            )
        ]
        if not newly_conflicted:
            break
        conflict_clocks.update(int(unique_clocks[index]) for index in newly_conflicted)
        rejected = set(newly_conflicted)
        active = [index for index in active if index not in rejected]
        if not active:
            raise DuplicateClockError("every distinct clock conflicted")
    representatives = [int(first_indices[group_index]) for group_index in active]
    roots, root_valid, scale = _roots_and_scale(raw, source_joint_mask, representatives)
    retained_clocks = np.asarray(unique_clocks[active], dtype="<i8")
    retained_raw = raw[representatives]
    retained_joint_valid = source_joint_mask[representatives] & root_valid[representatives, None]
    normalized = np.zeros((len(representatives), 17, 3), dtype="<f4")
    centered = (
        retained_raw[:, :, :2] - roots[np.asarray(representatives), None, :]
    ) / scale
    normalized[:, :, :2][retained_joint_valid] = centered[retained_joint_valid].astype(
        np.float32
    )
    normalized[:, :, 2][retained_joint_valid] = retained_raw[:, :, 2][
        retained_joint_valid
    ].astype(np.float32)
    joint_mask = np.asarray(retained_joint_valid, dtype="|u1")
    frame_mask = np.asarray(
        np.sum(retained_joint_valid, axis=1, dtype=np.int64) >= 8,
        dtype="|u1",
    )
    cell_mask = np.zeros(max(retained_clocks.size - 1, 0), dtype="|u1")
    frozen_conflicts = tuple(sorted(conflict_clocks))
    for index in range(cell_mask.size):
        left = int(retained_clocks[index])
        right = int(retained_clocks[index + 1])
        crosses_conflict = any(left < conflict < right for conflict in frozen_conflicts)
        cell_mask[index] = int(
            bool(frame_mask[index])
            and bool(frame_mask[index + 1])
            and right > left
            and not crosses_conflict
        )
    for array in (normalized, joint_mask, frame_mask, cell_mask, retained_clocks):
        array.flags.writeable = False
    return CollapsedTrack(
        motion=normalized,
        joint_mask=joint_mask,
        frame_mask=frame_mask,
        cell_mask=cell_mask,
        sampled_frame_indices=retained_clocks,
        source_length=feature.source_length,
        opaque_sample_key=feature.opaque_sample_key,
        local_person_slot=feature.local_person_slot,
        conflict_clocks=frozen_conflicts,
        normalization_scale=scale,
    )


def assess_count_blind_eligibility(value: EligibilityInput) -> EligibilityAssessment:
    """Apply only association, coverage, clock, mask, and support predicates."""

    if type(value.association_ambiguous) is not bool:
        raise EligibilityInputError("association_ambiguous must be a bool")
    if type(value.association_one_to_one) is not bool:
        raise EligibilityInputError("association_one_to_one must be a bool")
    if not math.isfinite(value.pose_coverage) or not 0.0 <= value.pose_coverage <= 1.0:
        raise EligibilityInputError("pose_coverage must be finite and in [0,1]")
    reasons: list[str] = []
    if value.association_ambiguous:
        reasons.append("association_ambiguous")
    if not value.association_one_to_one:
        reasons.append("association_not_one_to_one")
    if not value.feature.person_mask:
        reasons.append("person_invalid")
    if value.pose_coverage < 0.80:
        reasons.append("pose_coverage_below_minimum")
    track: CollapsedTrack | None = None
    if value.feature.person_mask:
        try:
            track = collapse_duplicate_clocks(value.feature)
        except NormalizationScaleError:
            reasons.append("missing_normalization_scale")
        except DuplicateClockError:
            reasons.append("duplicate_clock_collapse_failed")
    if track is None:
        retained = len(set(int(clock) for clock in value.feature.sampled_frame_indices))
        valid_cells = 0
        coverage = 0.0
        conflicts: tuple[int, ...] = ()
        scale: float | None = None
    else:
        retained = int(track.sampled_frame_indices.size)
        valid_cells = int(np.sum(track.cell_mask, dtype=np.int64))
        coverage = (
            float(np.sum(track.frame_mask, dtype=np.int64)) / retained if retained else 0.0
        )
        conflicts = track.conflict_clocks
        scale = track.normalization_scale
    if retained < 64:
        reasons.append("insufficient_distinct_clocks")
    if valid_cells < 63:
        reasons.append("insufficient_valid_adjacent_cells")
    if coverage < 0.60:
        reasons.append("insufficient_feature_frame_coverage")
    decision = EligibilityDecision(
        opaque_sample_key=value.feature.opaque_sample_key,
        local_person_slot=value.feature.local_person_slot,
        association_ambiguous=value.association_ambiguous,
        association_one_to_one=value.association_one_to_one,
        pose_coverage=value.pose_coverage,
        eligible=not reasons,
        reasons=tuple(reasons),
        retained_distinct_clocks=retained,
        valid_adjacent_cells=valid_cells,
        feature_frame_coverage=coverage,
        conflict_clocks=conflicts,
        normalization_scale=scale,
    )
    return EligibilityAssessment(decision=decision, track=track)


def _validate_collated_track(track: CollapsedTrack) -> None:
    length = int(track.sampled_frame_indices.size)
    if length < 1 or track.motion.shape != (length, 17, 3):
        raise FeatureDataError("collapsed track motion/clock shapes do not agree")
    if track.joint_mask.shape != (length, 17) or track.frame_mask.shape != (length,):
        raise FeatureDataError("collapsed track masks do not agree")
    if track.cell_mask.shape != (max(length - 1, 0),):
        raise FeatureDataError("collapsed cell mask shape does not agree")
    if np.any(np.diff(track.sampled_frame_indices) <= 0):
        raise FeatureDataError("collapsed clocks must be strictly increasing")


def collate_tracks(tracks: Sequence[CollapsedTrack]) -> FeatureBatch:
    """Right-pad complete identities without tensorizing routing identifiers."""

    if not tracks:
        raise FeatureDataError("cannot collate an empty identity batch")
    for track in tracks:
        _validate_collated_track(track)
    batch_size = len(tracks)
    maximum_length = max(int(track.sampled_frame_indices.size) for track in tracks)
    motion = np.zeros((batch_size, maximum_length, 17, 3), dtype="<f4")
    person_mask = np.ones(batch_size, dtype="|u1")
    frame_mask = np.zeros((batch_size, maximum_length), dtype="|u1")
    joint_mask = np.zeros((batch_size, maximum_length, 17), dtype="|u1")
    cell_mask = np.zeros((batch_size, max(maximum_length - 1, 0)), dtype="|u1")
    padding_mask = np.zeros((batch_size, maximum_length), dtype="|u1")
    clocks = np.zeros((batch_size, maximum_length), dtype="<i8")
    source_lengths = np.zeros(batch_size, dtype="<i8")
    for row, track in enumerate(tracks):
        length = int(track.sampled_frame_indices.size)
        motion[row, :length] = track.motion
        frame_mask[row, :length] = track.frame_mask
        joint_mask[row, :length] = track.joint_mask
        if length > 1:
            cell_mask[row, : length - 1] = track.cell_mask
        padding_mask[row, :length] = 1
        clocks[row, :length] = track.sampled_frame_indices
        source_lengths[row] = track.source_length
    return FeatureBatch(
        motion=motion,
        person_mask=person_mask,
        frame_mask=frame_mask,
        joint_mask=joint_mask,
        cell_mask=cell_mask,
        padding_mask=padding_mask,
        sampled_frame_indices=clocks,
        source_length=source_lengths,
        opaque_sample_keys=tuple(track.opaque_sample_key for track in tracks),
        local_person_slots=tuple(track.local_person_slot for track in tracks),
    )
