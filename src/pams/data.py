"""UCFRep manifests, pose preprocessing, split guards, and pose caches."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pams.types import PoseSequence

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_SPLITS = frozenset({"train", "dev", "test"})
_SPLIT_ALIASES: Mapping[str, str] = {
    "training": "train",
    "validation": "test",
    "val": "test",
    "testing": "test",
}
_EXPECTED_PROTOCOL_SPLITS: Mapping[str, Mapping[str, int]] = {
    "ucfrep_526": {"train": 421, "test": 105},
    "ucfrep_pose_110": {"train": 89, "test": 21},
}
_CSV_FIELDS = ("video_id", "video_path", "split", "action", "count", "video_sha256")


def _canonical_split(value: str) -> str:
    split = str(value).strip().lower()
    split = _SPLIT_ALIASES.get(split, split)
    if split not in _CANONICAL_SPLITS:
        raise ValueError(f"unknown split {value!r}; expected train, dev, test, or val alias")
    return split


def _validate_sha256(value: str | None, name: str) -> str | None:
    if value is None or value == "":
        return None
    digest = str(value).strip().lower()
    if not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return digest


@dataclass(frozen=True, slots=True)
class UCFRepRecord:
    """One row of the canonical UCFRep CSV/JSON manifest."""

    video_id: str
    video_path: str
    split: str
    action: str
    count: int
    video_sha256: str | None = None

    def __post_init__(self) -> None:
        video_id = str(self.video_id).strip()
        path = str(self.video_path).strip()
        action = str(self.action).strip()
        if not video_id:
            raise ValueError("video_id must be non-empty")
        if not path:
            raise ValueError("video_path must be non-empty")
        if not action:
            raise ValueError("action must be non-empty")
        if isinstance(self.count, (bool, np.bool_)):
            raise TypeError("count must be a positive integer, not bool")
        count = int(self.count)
        if count != self.count or count <= 0:
            raise ValueError("count must be a positive integer")

        object.__setattr__(self, "video_id", video_id)
        object.__setattr__(self, "video_path", path)
        object.__setattr__(self, "split", _canonical_split(self.split))
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "count", count)
        object.__setattr__(
            self,
            "video_sha256",
            _validate_sha256(self.video_sha256, "video_sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "split": self.split,
            "action": self.action,
            "count": self.count,
            "video_sha256": self.video_sha256,
        }


@dataclass(frozen=True, slots=True)
class UnlabeledVideoRecord:
    """Training-safe record with count, action, and test identity removed."""

    video_id: str
    video_path: str
    video_sha256: str | None


@dataclass(frozen=True, slots=True)
class UCFRepManifest:
    """Immutable manifest with uniqueness and split-disjointness guarantees."""

    protocol: str
    records: tuple[UCFRepRecord, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        protocol = str(self.protocol).strip()
        if protocol not in _EXPECTED_PROTOCOL_SPLITS:
            raise ValueError(
                f"unsupported protocol {protocol!r}; "
                f"expected one of {sorted(_EXPECTED_PROTOCOL_SPLITS)}"
            )
        if self.schema_version != 1:
            raise ValueError("only manifest schema_version=1 is supported")
        records = tuple(self.records)
        if not records:
            raise ValueError("manifest must contain at least one record")
        if not all(isinstance(record, UCFRepRecord) for record in records):
            raise TypeError("records must all be UCFRepRecord instances")

        identifiers = [record.video_id for record in records]
        if len(set(identifiers)) != len(identifiers):
            duplicates = sorted(
                identifier for identifier in set(identifiers) if identifiers.count(identifier) > 1
            )
            raise ValueError(f"duplicate video_id values: {duplicates[:5]}")
        normalized_paths = [
            os.path.normcase(os.path.normpath(record.video_path)) for record in records
        ]
        if len(set(normalized_paths)) != len(normalized_paths):
            raise ValueError("video_path values must be unique across all splits")

        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "records", records)

    @property
    def fingerprint(self) -> str:
        # Local paths differ across workstations and servers. Dataset identity
        # is based on protocol fields and optional content hashes, never the
        # mount point used to reach a video.
        stable_records = [
            {
                "video_id": record.video_id,
                "split": record.split,
                "action": record.action,
                "count": record.count,
                "video_sha256": record.video_sha256,
            }
            for record in self.records
        ]
        canonical = json.dumps(
            {
                "schema_version": self.schema_version,
                "protocol": self.protocol,
                "records": stable_records,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @property
    def split_counts(self) -> dict[str, int]:
        return {
            split: sum(record.split == split for record in self.records)
            for split in sorted(_CANONICAL_SPLITS)
            if any(record.split == split for record in self.records)
        }

    def records_for(self, split: str) -> tuple[UCFRepRecord, ...]:
        canonical = _canonical_split(split)
        return tuple(record for record in self.records if record.split == canonical)

    def validate_exact_official_splits(self) -> None:
        expected = dict(_EXPECTED_PROTOCOL_SPLITS[self.protocol])
        actual = self.split_counts
        if actual != expected:
            raise ValueError(
                f"{self.protocol} requires exact split counts {expected}, received {actual}"
            )
        assert_split_disjoint(self.records)

    def training_records(self, *, include_dev: bool = False) -> tuple[UnlabeledVideoRecord, ...]:
        """Return a label-free projection and never expose sealed test rows."""

        allowed = {"train", "dev"} if include_dev else {"train"}
        return tuple(
            UnlabeledVideoRecord(
                video_id=record.video_id,
                video_path=record.video_path,
                video_sha256=record.video_sha256,
            )
            for record in self.records
            if record.split in allowed
        )

    def test_targets(self) -> dict[str, int]:
        """Return sealed evaluator targets; training code should not call this."""

        return {record.video_id: record.count for record in self.records if record.split == "test"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol": self.protocol,
            "records": [record.to_dict() for record in self.records],
        }


def assert_split_disjoint(records: Iterable[UCFRepRecord]) -> None:
    """Assert that no identifier or normalized path occurs in two splits."""

    id_to_split: dict[str, str] = {}
    path_to_split: dict[str, str] = {}
    for record in records:
        previous = id_to_split.setdefault(record.video_id, record.split)
        if previous != record.split:
            raise ValueError(
                f"video_id {record.video_id!r} occurs in both {previous} and {record.split}"
            )
        path = os.path.normcase(os.path.normpath(record.video_path))
        previous_path = path_to_split.setdefault(path, record.split)
        if previous_path != record.split:
            raise ValueError(
                f"video_path {record.video_path!r} occurs in both "
                f"{previous_path} and {record.split}"
            )


def validate_exact_protocol_splits(manifest: UCFRepManifest) -> None:
    """Public functional form of the exact official-split check."""

    manifest.validate_exact_official_splits()


def _record_from_mapping(payload: Mapping[str, Any]) -> UCFRepRecord:
    expected = set(_CSV_FIELDS)
    supplied = set(payload)
    missing = expected - supplied - {"video_sha256"}
    unknown = supplied - expected
    if missing:
        raise ValueError(f"manifest record is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"manifest record has unknown fields: {sorted(unknown)}")
    return UCFRepRecord(
        video_id=str(payload["video_id"]),
        video_path=str(payload["video_path"]),
        split=str(payload["split"]),
        action=str(payload["action"]),
        count=int(payload["count"]),
        video_sha256=(
            None if payload.get("video_sha256") in (None, "") else str(payload["video_sha256"])
        ),
    )


def _infer_csv_protocol(records: Sequence[UCFRepRecord]) -> str:
    counts = {
        split: sum(record.split == split for record in records)
        for split in _CANONICAL_SPLITS
        if any(record.split == split for record in records)
    }
    matches = [
        protocol
        for protocol, expected in _EXPECTED_PROTOCOL_SPLITS.items()
        if dict(expected) == counts
    ]
    if len(matches) != 1:
        raise ValueError(
            "CSV protocol cannot be inferred from split counts; pass protocol explicitly"
        )
    return matches[0]


def load_ucfrep_manifest(
    path: str | Path,
    *,
    protocol: str | None = None,
    validate_exact: bool = True,
) -> UCFRepManifest:
    """Load the canonical strict JSON or CSV manifest."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("JSON manifest root must be an object")
        allowed = {"schema_version", "protocol", "records"}
        unknown = set(payload) - allowed
        missing = allowed - set(payload)
        if unknown or missing:
            raise ValueError(
                f"JSON manifest fields mismatch; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        if protocol is not None and payload["protocol"] != protocol:
            raise ValueError("requested protocol does not match JSON manifest protocol")
        raw_records = payload["records"]
        if not isinstance(raw_records, list):
            raise ValueError("JSON records must be a list")
        records = tuple(_record_from_mapping(item) for item in raw_records)
        manifest = UCFRepManifest(
            protocol=str(payload["protocol"]),
            records=records,
            schema_version=int(payload["schema_version"]),
        )
    elif suffix == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("CSV manifest is missing a header")
            required = set(_CSV_FIELDS) - {"video_sha256"}
            supplied = set(reader.fieldnames)
            if not required.issubset(supplied) or supplied - set(_CSV_FIELDS):
                raise ValueError(
                    f"CSV fields must be {_CSV_FIELDS} (video_sha256 optional), "
                    f"received {reader.fieldnames}"
                )
            records = tuple(_record_from_mapping(row) for row in reader)
        selected_protocol = protocol or _infer_csv_protocol(records)
        manifest = UCFRepManifest(protocol=selected_protocol, records=records)
    else:
        raise ValueError("manifest path must end in .json or .csv")

    if validate_exact:
        manifest.validate_exact_official_splits()
    return manifest


def save_ucfrep_manifest(manifest: UCFRepManifest, path: str | Path) -> Path:
    """Write a deterministic JSON or CSV manifest."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower()
    if suffix == ".json":
        target.write_text(
            json.dumps(
                manifest.to_dict(),
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    elif suffix == ".csv":
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            for record in manifest.records:
                writer.writerow(record.to_dict())
    else:
        raise ValueError("manifest path must end in .json or .csv")
    return target


def _count_bin(count: int) -> str:
    if count <= 5:
        return "01-05"
    if count <= 10:
        return "06-10"
    if count <= 20:
        return "11-20"
    if count <= 40:
        return "21-40"
    return "41+"


def deterministic_stratified_split(
    records: Sequence[UCFRepRecord],
    *,
    dev_size: int,
    seed: int = 2026,
) -> tuple[tuple[UCFRepRecord, ...], tuple[UCFRepRecord, ...]]:
    """Split train rows by action/count-bin with exact deterministic allocation."""

    source = tuple(records)
    if not source:
        raise ValueError("records must be non-empty")
    if any(record.split != "train" for record in source):
        raise ValueError("only official train records may be stratified")
    if len({record.video_id for record in source}) != len(source):
        raise ValueError("records must have unique video_id values")
    if not 0 < dev_size < len(source):
        raise ValueError("dev_size must be between zero and len(records)")

    groups: dict[tuple[str, str], list[UCFRepRecord]] = {}
    for record in source:
        groups.setdefault((record.action, _count_bin(record.count)), []).append(record)
    ordered_keys = sorted(groups)
    rng = np.random.default_rng(seed)
    shuffled: dict[tuple[str, str], list[UCFRepRecord]] = {}
    for key in ordered_keys:
        group = sorted(groups[key], key=lambda item: item.video_id)
        order = rng.permutation(len(group))
        shuffled[key] = [group[int(index)] for index in order]

    total = len(source)
    quotas = {key: len(shuffled[key]) * dev_size / total for key in ordered_keys}
    capacities = {key: max(0, len(shuffled[key]) - 1) for key in ordered_keys}
    allocations = {key: min(int(np.floor(quotas[key])), capacities[key]) for key in ordered_keys}
    remaining = dev_size - sum(allocations.values())

    ranked = sorted(
        ordered_keys,
        key=lambda key: (
            -(quotas[key] - np.floor(quotas[key])),
            -len(shuffled[key]),
            key,
        ),
    )
    while remaining:
        progressed = False
        for key in ranked:
            if allocations[key] < capacities[key]:
                allocations[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            # Only singleton strata remain. Exact size takes precedence, with
            # deterministic ordering, and this relaxation is visible in IDs.
            for key in ranked:
                if allocations[key] < len(shuffled[key]):
                    allocations[key] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
        if not progressed:
            raise RuntimeError("failed to allocate requested dev split")

    dev_ids: set[str] = set()
    for key in ordered_keys:
        dev_ids.update(record.video_id for record in shuffled[key][: allocations[key]])
    train = tuple(
        replace(record, split="train")
        for record in sorted(source, key=lambda item: item.video_id)
        if record.video_id not in dev_ids
    )
    dev = tuple(
        replace(record, split="dev")
        for record in sorted(source, key=lambda item: item.video_id)
        if record.video_id in dev_ids
    )
    if len(train) + len(dev) != total or len(dev) != dev_size:
        raise RuntimeError("stratified split size invariant failed")
    assert_split_disjoint((*train, *dev))
    return train, dev


def split_ucfrep_train_dev(
    records: Sequence[UCFRepRecord], *, seed: int = 2026
) -> tuple[tuple[UCFRepRecord, ...], tuple[UCFRepRecord, ...]]:
    """Create the frozen 337/84 split from the 421 standard train rows."""

    if len(records) != 421:
        raise ValueError(f"standard UCFRep split requires 421 rows, got {len(records)}")
    train, dev = deterministic_stratified_split(records, dev_size=84, seed=seed)
    if (len(train), len(dev)) != (337, 84):
        raise RuntimeError("standard split did not produce 337/84")
    return train, dev


def split_ucfrep_pose_train_dev(
    records: Sequence[UCFRepRecord], *, seed: int = 2026
) -> tuple[tuple[UCFRepRecord, ...], tuple[UCFRepRecord, ...]]:
    """Create the frozen 71/18 split from the 89 pose-protocol train rows."""

    if len(records) != 89:
        raise ValueError(f"UCFRep-pose split requires 89 rows, got {len(records)}")
    train, dev = deterministic_stratified_split(records, dev_size=18, seed=seed)
    if (len(train), len(dev)) != (71, 18):
        raise RuntimeError("pose split did not produce 71/18")
    return train, dev


def per_frame_minmax(
    xyz: ArrayLike,
    valid_mask: ArrayLike | None = None,
    *,
    epsilon: float = 1e-8,
) -> NDArray[np.float32]:
    """Normalize every valid frame jointly over its 99 coordinate values."""

    coordinates = np.asarray(xyz, dtype=np.float32)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (33, 3):
        raise ValueError("xyz must have shape [frames, 33, 3]")
    if coordinates.shape[0] < 1:
        raise ValueError("xyz must contain at least one frame")
    mask = (
        np.ones(coordinates.shape[0], dtype=np.bool_)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=np.bool_)
    )
    if mask.shape != (coordinates.shape[0],):
        raise ValueError("valid_mask must have shape [frames]")
    if not np.isfinite(coordinates[mask]).all():
        raise ValueError("valid coordinates must be finite")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    result = np.zeros_like(coordinates, dtype=np.float32)
    if np.any(mask):
        valid = coordinates[mask]
        minimum = valid.min(axis=(1, 2), keepdims=True)
        maximum = valid.max(axis=(1, 2), keepdims=True)
        span = maximum - minimum
        normalized = np.divide(
            valid - minimum,
            span,
            out=np.zeros_like(valid),
            where=span > epsilon,
        )
        result[mask] = normalized
    return result


def uniform_resample(
    xyz: ArrayLike,
    valid_mask: ArrayLike,
    *,
    target_frames: int = 256,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Linearly sample endpoints uniformly without interpolating across gaps."""

    coordinates = np.asarray(xyz, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=np.bool_)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (33, 3):
        raise ValueError("xyz must have shape [frames, 33, 3]")
    if coordinates.shape[0] < 1 or mask.shape != (coordinates.shape[0],):
        raise ValueError("valid_mask must match a non-empty xyz sequence")
    if not np.isfinite(coordinates[mask]).all():
        raise ValueError("valid coordinates must be finite")
    if target_frames < 1:
        raise ValueError("target_frames must be positive")

    source_frames = coordinates.shape[0]
    if source_frames == 1:
        output_mask = np.repeat(mask, target_frames)
        output = np.repeat(coordinates, target_frames, axis=0)
        output[~output_mask] = 0.0
        return output.astype(np.float32), output_mask

    positions = np.linspace(0.0, source_frames - 1, target_frames)
    left = np.floor(positions).astype(np.int64)
    right = np.ceil(positions).astype(np.int64)
    weight = (positions - left).astype(np.float32)
    exact = left == right
    output_mask = np.where(exact, mask[left], mask[left] & mask[right])
    output = (
        coordinates[left] * (1.0 - weight[:, None, None])
        + coordinates[right] * weight[:, None, None]
    )
    output[~output_mask] = 0.0
    return output.astype(np.float32), output_mask.astype(np.bool_)


def longest_valid_span(valid_mask: ArrayLike) -> slice:
    """Return the longest contiguous valid frame span (earliest on ties)."""

    mask = np.asarray(valid_mask, dtype=np.bool_)
    if mask.ndim != 1 or mask.size < 1:
        raise ValueError("valid_mask must be non-empty and one-dimensional")
    best_start = 0
    best_length = 0
    current_start = 0
    current_length = 0
    for index, valid in enumerate(mask):
        if valid:
            if current_length == 0:
                current_start = index
            current_length += 1
            if current_length > best_length:
                best_start = current_start
                best_length = current_length
        else:
            current_length = 0
    if best_length == 0:
        raise ValueError("valid_mask contains no valid frames")
    return slice(best_start, best_start + best_length)


def preprocess_pose_sequence(sequence: PoseSequence, *, target_frames: int = 256) -> PoseSequence:
    """Apply the frozen per-frame normalization and uniform resampling."""

    normalized = per_frame_minmax(sequence.xyz, sequence.valid_mask)
    xyz, mask = uniform_resample(normalized, sequence.valid_mask, target_frames=target_frames)
    if sequence.num_frames == 1 or target_frames == 1:
        fps = sequence.fps
    else:
        fps = sequence.fps * (target_frames - 1) / (sequence.num_frames - 1)
    return PoseSequence(
        video_id=sequence.video_id,
        fps=fps,
        xyz=xyz,
        valid_mask=mask,
    )


@dataclass(frozen=True, slots=True)
class PoseCacheMetadata:
    schema_version: int
    video_id: str
    video_sha256: str
    config_sha256: str
    pose_model: str
    fps: float
    frames: int
    keypoints: int = 33
    coordinates: int = 3

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only pose cache schema_version=1 is supported")
        if not self.video_id.strip() or not self.pose_model.strip():
            raise ValueError("video_id and pose_model must be non-empty")
        _validate_sha256(self.video_sha256, "video_sha256")
        _validate_sha256(self.config_sha256, "config_sha256")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("cache fps must be positive")
        if self.frames < 1 or self.keypoints != 33 or self.coordinates != 3:
            raise ValueError("cache shape metadata must describe [frames, 33, 3]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "video_id": self.video_id,
            "video_sha256": self.video_sha256,
            "config_sha256": self.config_sha256,
            "pose_model": self.pose_model,
            "fps": self.fps,
            "frames": self.frames,
            "keypoints": self.keypoints,
            "coordinates": self.coordinates,
        }


def pose_cache_path(cache_dir: str | Path, video_id: str) -> Path:
    """Map arbitrary dataset identifiers to collision-resistant safe paths."""

    identifier = str(video_id).strip()
    if not identifier:
        raise ValueError("video_id must be non-empty")
    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{digest}.npz"


def write_pose_cache(
    path: str | Path,
    sequence: PoseSequence,
    *,
    video_sha256: str,
    config_sha256: str,
    pose_model: str = "mediapipe-0.10.14",
    overwrite: bool = False,
) -> PoseCacheMetadata:
    """Atomically write a cache whose provenance is validated on every load."""

    metadata = PoseCacheMetadata(
        schema_version=1,
        video_id=sequence.video_id,
        video_sha256=_validate_sha256(video_sha256, "video_sha256") or "",
        config_sha256=_validate_sha256(config_sha256, "config_sha256") or "",
        pose_model=str(pose_model),
        fps=sequence.fps,
        frames=sequence.num_frames,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"pose cache already exists: {target}")
    payload = json.dumps(metadata.to_dict(), sort_keys=True, separators=(",", ":"))
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(
            temporary,
            xyz=sequence.xyz,
            valid_mask=sequence.valid_mask,
            metadata=np.asarray(payload),
        )
        if target.exists() and not overwrite:
            raise FileExistsError(f"pose cache already exists: {target}")
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return metadata


def load_pose_cache(
    path: str | Path,
    *,
    expected_video_sha256: str | None = None,
    expected_config_sha256: str | None = None,
) -> tuple[PoseSequence, PoseCacheMetadata]:
    """Load a cache and reject stale video/config provenance."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        required = {"xyz", "valid_mask", "metadata"}
        if set(archive.files) != required:
            raise ValueError(
                f"pose cache must contain exactly {sorted(required)}, "
                f"received {sorted(archive.files)}"
            )
        raw_metadata = archive["metadata"]
        if raw_metadata.ndim != 0:
            raise ValueError("cache metadata must be a scalar JSON string")
        payload = json.loads(str(raw_metadata.item()))
        if not isinstance(payload, dict):
            raise ValueError("cache metadata JSON must be an object")
        metadata = PoseCacheMetadata(**payload)
        sequence = PoseSequence(
            video_id=metadata.video_id,
            fps=metadata.fps,
            xyz=archive["xyz"],
            valid_mask=archive["valid_mask"],
        )
    if metadata.frames != sequence.num_frames:
        raise ValueError("cache frame count does not match its metadata")
    expected_video = _validate_sha256(expected_video_sha256, "expected_video_sha256")
    expected_config = _validate_sha256(expected_config_sha256, "expected_config_sha256")
    if expected_video is not None and metadata.video_sha256 != expected_video:
        raise ValueError("pose cache video SHA-256 mismatch")
    if expected_config is not None and metadata.config_sha256 != expected_config:
        raise ValueError("pose cache config SHA-256 mismatch")
    return sequence, metadata


class TrainingPoseDataset(Sequence[PoseSequence]):
    """Label-free cached-pose dataset that rejects all sealed-test records."""

    def __init__(
        self,
        records: Sequence[UCFRepRecord],
        *,
        cache_dir: str | Path,
        config_sha256: str,
        include_dev: bool = False,
    ) -> None:
        source = tuple(records)
        allowed = {"train", "dev"} if include_dev else {"train"}
        forbidden = sorted({record.split for record in source if record.split not in allowed})
        if forbidden:
            raise ValueError(
                f"training dataset cannot contain sealed or disallowed splits: {forbidden}"
            )
        if not source:
            raise ValueError("training dataset must contain at least one record")
        self._items = tuple(
            UnlabeledVideoRecord(
                video_id=record.video_id,
                video_path=record.video_path,
                video_sha256=record.video_sha256,
            )
            for record in source
        )
        self._cache_dir = Path(cache_dir)
        validated = _validate_sha256(config_sha256, "config_sha256")
        if validated is None:
            raise ValueError("config_sha256 is required")
        self._config_sha256 = validated

    def __len__(self) -> int:
        return len(self._items)

    @overload
    def __getitem__(self, index: int) -> PoseSequence: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[PoseSequence, ...]: ...

    def __getitem__(self, index: int | slice) -> PoseSequence | tuple[PoseSequence, ...]:
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        item = self._items[index]
        sequence, _ = load_pose_cache(
            pose_cache_path(self._cache_dir, item.video_id),
            expected_video_sha256=item.video_sha256,
            expected_config_sha256=self._config_sha256,
        )
        if sequence.video_id != item.video_id:
            raise ValueError("cache video_id does not match training record")
        return sequence

    def __iter__(self) -> Iterator[PoseSequence]:
        for index in range(len(self)):
            yield self[index]
