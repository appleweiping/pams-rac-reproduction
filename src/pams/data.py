"""UCFRep manifests, pose preprocessing, split guards, and pose caches."""

from __future__ import annotations

import csv
import hashlib
import io
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
_UCF101_VIDEO_ID_PATTERN = re.compile(r"^v_(?P<action>.+)_g(?P<group>\d{2})_c\d+$")
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
_UCFREP_526_CANONICAL_ID_SHA256: Mapping[str, str] = {
    "train": "811d263b61dce27671551326b45799b06ac4b07dba311bdc39d2ed6663729d25",
    "dev": "2199294a1d22da6c67d2fdaaafb4bebaa11e92a5bb3accd4670975815001f2c6",
    "train_pool": "776489ba3c556a105f96bd18788c527a9157dfaa451a0eb766c38faca27e416f",
    "test": "0ca00989fa5d378a5e92fbb9e9663ddde3805d6f1806816a6ed8cd7b3d819b8b",
}
_UCFREP_526_CANONICAL_ANNOTATION_SHA256 = (
    "d371f9f4609730d6484efc337413b444ed73752ad5e994366d02fb79a9960452"
)


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
        if isinstance(self.count, bool | np.bool_):
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

    def __post_init__(self) -> None:
        video_id = str(self.video_id).strip()
        video_path = str(self.video_path).strip()
        if not video_id:
            raise ValueError("video_id must be non-empty")
        if not video_path:
            raise ValueError("video_path must be non-empty")
        object.__setattr__(self, "video_id", video_id)
        object.__setattr__(self, "video_path", video_path)
        object.__setattr__(
            self,
            "video_sha256",
            _validate_sha256(self.video_sha256, "video_sha256"),
        )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "video_sha256": self.video_sha256,
        }


@dataclass(frozen=True, slots=True)
class PoseInputManifest:
    """Count-free, action-field-free video inputs for one pose-extraction split."""

    protocol: str
    split: str
    records: tuple[UnlabeledVideoRecord, ...]
    schema_version: int = 2
    manifest_type: str = "pose_inputs"

    def __post_init__(self) -> None:
        protocol = str(self.protocol).strip()
        if protocol not in _EXPECTED_PROTOCOL_SPLITS:
            raise ValueError(
                f"unsupported protocol {protocol!r}; "
                f"expected one of {sorted(_EXPECTED_PROTOCOL_SPLITS)}"
            )
        if self.schema_version != 2:
            raise ValueError("only pose-input schema_version=2 is supported")
        if self.manifest_type != "pose_inputs":
            raise ValueError("pose-input manifest_type must be 'pose_inputs'")
        split = _canonical_split(self.split)
        records = tuple(self.records)
        if not records:
            raise ValueError("pose-input manifest must contain at least one record")
        if not all(isinstance(record, UnlabeledVideoRecord) for record in records):
            raise TypeError("pose-input records must all be UnlabeledVideoRecord instances")
        identifiers = [record.video_id for record in records]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("pose-input video_id values must be unique")
        normalized_paths = [
            os.path.normcase(os.path.normpath(record.video_path)) for record in records
        ]
        if len(set(normalized_paths)) != len(normalized_paths):
            raise ValueError("pose-input video_path values must be unique")
        for record in records:
            locator = record.video_path
            if "\\" in locator:
                raise ValueError("pose-input video locators must use portable '/' separators")
            locator_path = Path(locator)
            if locator_path.is_absolute() or locator_path.anchor:
                raise ValueError("pose-input video locators must be relative")
            if any(part in {"", ".", ".."} for part in locator.split("/")):
                raise ValueError(
                    "pose-input video locators must not contain empty, '.', or '..' segments"
                )
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "records", records)

    @property
    def fingerprint(self) -> str:
        canonical = {
            "schema_version": self.schema_version,
            "manifest_type": self.manifest_type,
            "protocol": self.protocol,
            "split": self.split,
            "records": [
                {
                    "video_id": record.video_id,
                    "video_sha256": record.video_sha256,
                    "video_locator": record.video_path,
                }
                for record in self.records
            ],
        }
        encoded = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def validate_exact_membership(self) -> None:
        if self.protocol != "ucfrep_526":
            raise ValueError(
                "exact pose-input membership is unavailable until the protocol's "
                "official split identities are frozen"
            )
        expected_key_by_split_and_count = {
            ("train", 421): "train_pool",
            ("train", 337): "train",
            ("dev", 84): "dev",
            ("test", 105): "test",
        }
        expected_key = expected_key_by_split_and_count.get((self.split, len(self.records)))
        if expected_key is None:
            raise ValueError(
                "ucfrep_526 pose inputs require exactly train=421/337, dev=84, or test=105"
            )
        observed_ids_sha256 = _canonical_unlabeled_id_list_sha256(self.records)
        expected_ids_sha256 = _UCFREP_526_CANONICAL_ID_SHA256[expected_key]
        if observed_ids_sha256 != expected_ids_sha256:
            raise ValueError(
                f"ucfrep_526 {self.split} pose-input IDs do not match the "
                "preregistered list"
            )
        if any(record.video_sha256 is None for record in self.records):
            raise ValueError("exact pose inputs require every source-video SHA-256")
        for record in self.records:
            match = _UCF101_VIDEO_ID_PATTERN.fullmatch(record.video_id)
            if match is None:
                raise ValueError(f"invalid canonical UCF101 video_id: {record.video_id!r}")
            group = int(match.group("group"))
            expected_family = "test" if 21 <= group <= 25 else "train"
            actual_family = "test" if self.split == "test" else "train"
            if not 1 <= group <= 25 or expected_family != actual_family:
                raise ValueError(
                    f"canonical source group and pose-input split disagree for "
                    f"{record.video_id!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_type": self.manifest_type,
            "protocol": self.protocol,
            "split": self.split,
            "records": [record.to_dict() for record in self.records],
        }


def pose_input_identity_sha256(records: Iterable[UnlabeledVideoRecord]) -> str:
    """Commit to sorted label-free source identities without paths or labels."""

    normalized = tuple(sorted(tuple(records), key=lambda record: record.video_id))
    if not normalized:
        raise ValueError("pose-input identity commitment requires at least one record")
    if any(record.video_sha256 is None for record in normalized):
        raise ValueError("pose-input identity commitment requires every video SHA-256")
    identifiers = [record.video_id for record in normalized]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("pose-input identity commitment video_id values must be unique")
    encoded = json.dumps(
        [
            {"video_id": record.video_id, "video_sha256": record.video_sha256}
            for record in normalized
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PoseInputCommitment:
    """Independent receipt binding one sidecar file to label-free identities."""

    protocol: str
    split: str
    record_total: int
    identity_sha256: str
    sidecar_sha256: str
    sidecar_fingerprint: str
    schema_version: int = 1
    commitment_type: str = "pose_input_commitment"

    def __post_init__(self) -> None:
        protocol = str(self.protocol).strip()
        if protocol not in _EXPECTED_PROTOCOL_SPLITS:
            raise ValueError(f"unsupported commitment protocol {protocol!r}")
        split = _canonical_split(self.split)
        if self.schema_version != 1:
            raise ValueError("only pose-input commitment schema_version=1 is supported")
        if self.commitment_type != "pose_input_commitment":
            raise ValueError("invalid pose-input commitment_type")
        if (
            isinstance(self.record_total, bool)
            or not isinstance(self.record_total, int)
            or self.record_total < 1
        ):
            raise ValueError("pose-input commitment record_total must be positive")
        for field in ("identity_sha256", "sidecar_sha256", "sidecar_fingerprint"):
            digest = _validate_sha256(getattr(self, field), field)
            assert digest is not None
            object.__setattr__(self, field, digest)
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "split", split)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "commitment_type": self.commitment_type,
            "protocol": self.protocol,
            "split": self.split,
            "record_total": self.record_total,
            "identity_sha256": self.identity_sha256,
            "sidecar_sha256": self.sidecar_sha256,
            "sidecar_fingerprint": self.sidecar_fingerprint,
        }


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
    def sealed_dataset_fingerprint(self) -> str:
        """Return the split/order/path-invariant official annotation identity."""

        if self.protocol != "ucfrep_526":
            raise ValueError(
                "sealed dataset identity is unavailable until the protocol's "
                "official ID/count digest is frozen"
            )
        return _ucfrep_526_annotation_fingerprint(self.records)

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
        validate_canonical_split_membership(self)

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

    def training_fingerprint(self, *, include_dev: bool = False) -> str:
        """Hash only the exact label-free videos exposed to training.

        Sealed-test rows, counts, actions, and machine-specific paths are
        deliberately absent. Changing any of them cannot alter a checkpoint
        or training-run identity.
        """

        records = sorted(
            self.training_records(include_dev=include_dev),
            key=lambda record: record.video_id,
        )
        canonical = json.dumps(
            {
                "schema_version": 1,
                "protocol": self.protocol,
                "records": [
                    {
                        "video_id": record.video_id,
                        "video_sha256": record.video_sha256,
                    }
                    for record in records
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

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
    """Assert that no identifier, path, or known content occurs in two splits."""

    id_to_split: dict[str, str] = {}
    path_to_split: dict[str, str] = {}
    digest_to_split: dict[str, str] = {}
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
        if record.video_sha256 is not None:
            previous_digest = digest_to_split.setdefault(record.video_sha256, record.split)
            if previous_digest != record.split:
                raise ValueError(
                    f"video_sha256 {record.video_sha256!r} occurs in both "
                    f"{previous_digest} and {record.split}"
                )


def _canonical_unlabeled_id_list_sha256(
    records: Iterable[UCFRepRecord | UnlabeledVideoRecord],
) -> str:
    identifiers = sorted(record.video_id for record in records)
    return hashlib.sha256(("\n".join(identifiers) + "\n").encode("utf-8")).hexdigest()


def _ucfrep_526_annotation_fingerprint(records: Iterable[UCFRepRecord]) -> str:
    """Bind every official ID to its action, count, and train/test family."""

    rows: list[dict[str, str | int]] = []
    for record in records:
        match = _UCF101_VIDEO_ID_PATTERN.fullmatch(record.video_id)
        if match is None:
            raise ValueError(f"invalid canonical UCF101 video_id: {record.video_id!r}")
        group = int(match.group("group"))
        if not 1 <= group <= 25:
            raise ValueError(f"canonical UCFRep group is outside 01-25: {record.video_id!r}")
        rows.append(
            {
                "video_id": record.video_id,
                "action": record.action,
                "count": record.count,
                "official_family": "test" if group >= 21 else "train",
            }
        )
    encoded = json.dumps(
        sorted(rows, key=lambda row: str(row["video_id"])),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_canonical_split_membership(manifest: UCFRepManifest) -> None:
    """Bind standard UCFRep splits to the preregistered official ID lists."""

    if manifest.protocol != "ucfrep_526":
        # UCFRep-pose remains blocked until its independently audited 89/21
        # source lists are checked into the repository.
        return

    records_by_split = {
        split: manifest.records_for(split)
        for split in ("train", "dev", "test")
        if manifest.records_for(split)
    }
    expected_keys = {"train", "test"} if "dev" not in records_by_split else {"train", "dev", "test"}
    if set(records_by_split) != expected_keys:
        raise ValueError(
            "ucfrep_526 requires either canonical 421/105 or frozen 337/84/105 membership"
        )

    expected_hashes = (
        {
            "train": _UCFREP_526_CANONICAL_ID_SHA256["train_pool"],
            "test": _UCFREP_526_CANONICAL_ID_SHA256["test"],
        }
        if "dev" not in records_by_split
        else {
            "train": _UCFREP_526_CANONICAL_ID_SHA256["train"],
            "dev": _UCFREP_526_CANONICAL_ID_SHA256["dev"],
            "test": _UCFREP_526_CANONICAL_ID_SHA256["test"],
        }
    )
    for split, expected_hash in expected_hashes.items():
        actual_hash = _canonical_unlabeled_id_list_sha256(records_by_split[split])
        if actual_hash != expected_hash:
            raise ValueError(
                f"ucfrep_526 {split} video IDs do not match the preregistered list: "
                f"expected {expected_hash}, received {actual_hash}"
            )

    for record in manifest.records:
        match = _UCF101_VIDEO_ID_PATTERN.fullmatch(record.video_id)
        if match is None:
            raise ValueError(f"invalid canonical UCF101 video_id: {record.video_id!r}")
        if record.action != match.group("action"):
            raise ValueError(
                f"record action does not match its canonical video_id: {record.video_id!r}"
            )
        group = int(match.group("group"))
        expected_split_family = "test" if 21 <= group <= 25 else "train"
        if not (1 <= group <= 25):
            raise ValueError(f"canonical UCFRep group is outside 01-25: {record.video_id!r}")
        actual_split_family = "test" if record.split == "test" else "train"
        if actual_split_family != expected_split_family:
            raise ValueError(
                f"canonical source group and declared split disagree for {record.video_id!r}"
            )
    annotation_fingerprint = manifest.sealed_dataset_fingerprint
    if annotation_fingerprint != _UCFREP_526_CANONICAL_ANNOTATION_SHA256:
        raise ValueError(
            "ucfrep_526 annotations do not match the frozen official "
            "(video_id, action, count, train/test family) digest: "
            f"expected {_UCFREP_526_CANONICAL_ANNOTATION_SHA256}, "
            f"received {annotation_fingerprint}"
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


def _unlabeled_record_from_mapping(payload: Mapping[str, Any]) -> UnlabeledVideoRecord:
    expected = {"video_id", "video_path", "video_sha256"}
    supplied = set(payload)
    if supplied != expected:
        raise ValueError(
            "pose-input record fields mismatch; "
            f"missing={sorted(expected - supplied)}, unknown={sorted(supplied - expected)}"
        )
    return UnlabeledVideoRecord(
        video_id=str(payload["video_id"]),
        video_path=str(payload["video_path"]),
        video_sha256=(
            None if payload["video_sha256"] in (None, "") else str(payload["video_sha256"])
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


def load_pose_input_manifest(
    path: str | Path,
    *,
    validate_exact: bool = True,
) -> PoseInputManifest:
    """Load a strict JSON manifest containing no count or action fields."""

    def reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field in pose-input manifest: {key!r}")
            result[key] = value
        return result

    def reject_non_finite(value: str) -> None:
        raise ValueError(f"non-finite JSON constant in pose-input manifest: {value}")

    source = Path(path)
    if source.suffix.lower() != ".json":
        raise ValueError("pose-input manifest path must end in .json")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_fields,
            parse_constant=reject_non_finite,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid pose-input manifest JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("pose-input manifest root must be an object")
    expected = {
        "schema_version",
        "manifest_type",
        "protocol",
        "split",
        "records",
    }
    supplied = set(payload)
    if supplied != expected:
        raise ValueError(
            "pose-input manifest fields mismatch; "
            f"missing={sorted(expected - supplied)}, unknown={sorted(supplied - expected)}"
        )
    raw_records = payload["records"]
    if not isinstance(raw_records, list):
        raise ValueError("pose-input records must be a list")
    if any(not isinstance(item, Mapping) for item in raw_records):
        raise ValueError("pose-input records must be JSON objects")
    manifest = PoseInputManifest(
        protocol=str(payload["protocol"]),
        split=str(payload["split"]),
        records=tuple(_unlabeled_record_from_mapping(item) for item in raw_records),
        schema_version=int(payload["schema_version"]),
        manifest_type=str(payload["manifest_type"]),
    )
    if validate_exact:
        manifest.validate_exact_membership()
    return manifest


def load_pose_input_commitment(path: str | Path) -> PoseInputCommitment:
    """Load a strict, path-free commitment produced beside a pose-input sidecar."""

    def reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field in pose-input commitment: {key!r}")
            result[key] = value
        return result

    source = Path(path)
    if source.suffix.lower() != ".json":
        raise ValueError("pose-input commitment path must end in .json")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_fields,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in pose-input commitment: {value}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid pose-input commitment JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("pose-input commitment root must be an object")
    expected = {
        "schema_version",
        "commitment_type",
        "protocol",
        "split",
        "record_total",
        "identity_sha256",
        "sidecar_sha256",
        "sidecar_fingerprint",
    }
    supplied = set(payload)
    if supplied != expected:
        raise ValueError(
            "pose-input commitment fields mismatch; "
            f"missing={sorted(expected - supplied)}, unknown={sorted(supplied - expected)}"
        )
    return PoseInputCommitment(
        protocol=str(payload["protocol"]),
        split=str(payload["split"]),
        record_total=int(payload["record_total"]),
        identity_sha256=str(payload["identity_sha256"]),
        sidecar_sha256=str(payload["sidecar_sha256"]),
        sidecar_fingerprint=str(payload["sidecar_fingerprint"]),
        schema_version=int(payload["schema_version"]),
        commitment_type=str(payload["commitment_type"]),
    )


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

    effective_mask = np.array(sequence.valid_mask, dtype=np.bool_, copy=True)
    valid_indices = np.flatnonzero(effective_mask)
    if valid_indices.size:
        valid_coordinates = sequence.xyz[valid_indices]
        spans = valid_coordinates.max(axis=(1, 2)) - valid_coordinates.min(axis=(1, 2))
        effective_mask[valid_indices[spans <= 1e-8]] = False
    normalized = per_frame_minmax(sequence.xyz, effective_mask)
    xyz, mask = uniform_resample(normalized, effective_mask, target_frames=target_frames)
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
    pose_fingerprint: str
    pose_model: str
    fps: float
    frames: int
    keypoints: int = 33
    coordinates: int = 3

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("only pose cache schema_version=2 is supported")
        if not self.video_id.strip() or not self.pose_model.strip():
            raise ValueError("video_id and pose_model must be non-empty")
        _validate_sha256(self.video_sha256, "video_sha256")
        _validate_sha256(self.pose_fingerprint, "pose_fingerprint")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("cache fps must be positive")
        if self.frames < 1 or self.keypoints != 33 or self.coordinates != 3:
            raise ValueError("cache shape metadata must describe [frames, 33, 3]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "video_id": self.video_id,
            "video_sha256": self.video_sha256,
            "pose_fingerprint": self.pose_fingerprint,
            "pose_model": self.pose_model,
            "fps": self.fps,
            "frames": self.frames,
            "keypoints": self.keypoints,
            "coordinates": self.coordinates,
        }


@dataclass(frozen=True, slots=True)
class PoseCacheEntryReceipt:
    """Content identity for one pose-cache file actually read."""

    video_id: str
    cache_sha256: str
    bytes: int

    def __post_init__(self) -> None:
        identifier = str(self.video_id).strip()
        digest = _validate_sha256(self.cache_sha256, "cache_sha256")
        if not identifier:
            raise ValueError("pose-cache receipt video_id must be non-empty")
        if digest is None:
            raise ValueError("pose-cache receipt SHA-256 is required")
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or self.bytes < 1:
            raise ValueError("pose-cache receipt bytes must be a positive integer")
        object.__setattr__(self, "video_id", identifier)
        object.__setattr__(self, "cache_sha256", digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "cache_sha256": self.cache_sha256,
            "bytes": self.bytes,
        }


@dataclass(frozen=True, slots=True)
class PoseCacheSetSnapshot:
    """Canonical digest of every pose-cache byte stream used by one command."""

    pose_fingerprint: str
    entries: tuple[PoseCacheEntryReceipt, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only pose-cache snapshot schema_version=1 is supported")
        pose_fingerprint = _validate_sha256(self.pose_fingerprint, "pose_fingerprint")
        if pose_fingerprint is None:
            raise ValueError("pose-cache snapshot pose_fingerprint is required")
        raw_entries = tuple(self.entries)
        if not all(isinstance(entry, PoseCacheEntryReceipt) for entry in raw_entries):
            raise TypeError("pose-cache snapshot entries must be PoseCacheEntryReceipt values")
        entries = tuple(sorted(raw_entries, key=lambda entry: entry.video_id))
        if not entries:
            raise ValueError("pose-cache snapshot requires at least one entry")
        identifiers = [entry.video_id for entry in entries]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("pose-cache snapshot video_id values must be unique")
        object.__setattr__(self, "pose_fingerprint", pose_fingerprint)
        object.__setattr__(self, "entries", entries)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            {
                "schema_version": self.schema_version,
                "pose_fingerprint": self.pose_fingerprint,
                "entries": [entry.to_dict() for entry in self.entries],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pose_fingerprint": self.pose_fingerprint,
            "fingerprint": self.fingerprint,
            "entry_count": len(self.entries),
            "entries": [entry.to_dict() for entry in self.entries],
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
    pose_fingerprint: str,
    pose_model: str = "mediapipe-pose-0.10.14",
    overwrite: bool = False,
) -> PoseCacheMetadata:
    """Atomically write a cache whose provenance is validated on every load."""

    metadata = PoseCacheMetadata(
        schema_version=2,
        video_id=sequence.video_id,
        video_sha256=_validate_sha256(video_sha256, "video_sha256") or "",
        pose_fingerprint=_validate_sha256(pose_fingerprint, "pose_fingerprint") or "",
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


def _read_file_bytes_stable(path: Path) -> bytes:
    """Read one regular file while detecting replacement or mutation."""

    before = path.stat()
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        payload = handle.read()
        closed = os.fstat(handle.fileno())
    after = path.stat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field) != getattr(opened, field)
        or getattr(opened, field) != getattr(closed, field)
        or getattr(closed, field) != getattr(after, field)
        for field in stable_fields
    ):
        raise RuntimeError(f"pose cache changed while it was being read: {path}")
    if len(payload) != after.st_size:
        raise RuntimeError(f"pose cache byte count changed while reading: {path}")
    return payload


def load_pose_cache_with_receipt(
    path: str | Path,
    *,
    expected_video_sha256: str | None = None,
    expected_pose_fingerprint: str | None = None,
) -> tuple[PoseSequence, PoseCacheMetadata, PoseCacheEntryReceipt]:
    """Load exact cache bytes and return their cryptographic receipt."""

    source = Path(path)
    source_bytes = _read_file_bytes_stable(source)
    with np.load(io.BytesIO(source_bytes), allow_pickle=False) as archive:
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
        if "config_sha256" in payload or payload.get("schema_version") == 1:
            raise ValueError(
                "legacy pose cache uses the full experiment config_sha256; "
                "regenerate it with schema_version=2 pose_fingerprint metadata"
            )
        try:
            metadata = PoseCacheMetadata(**payload)
        except TypeError as exc:
            raise ValueError(f"invalid pose cache metadata fields: {exc}") from exc
        sequence = PoseSequence(
            video_id=metadata.video_id,
            fps=metadata.fps,
            xyz=archive["xyz"],
            valid_mask=archive["valid_mask"],
        )
    if metadata.frames != sequence.num_frames:
        raise ValueError("cache frame count does not match its metadata")
    expected_video = _validate_sha256(expected_video_sha256, "expected_video_sha256")
    expected_pose = _validate_sha256(
        expected_pose_fingerprint,
        "expected_pose_fingerprint",
    )
    if expected_video is not None and metadata.video_sha256 != expected_video:
        raise ValueError("pose cache video SHA-256 mismatch")
    if expected_pose is not None and metadata.pose_fingerprint != expected_pose:
        raise ValueError("pose cache fingerprint mismatch")
    return (
        sequence,
        metadata,
        PoseCacheEntryReceipt(
            video_id=sequence.video_id,
            cache_sha256=hashlib.sha256(source_bytes).hexdigest(),
            bytes=len(source_bytes),
        ),
    )


def load_pose_cache(
    path: str | Path,
    *,
    expected_video_sha256: str | None = None,
    expected_pose_fingerprint: str | None = None,
) -> tuple[PoseSequence, PoseCacheMetadata]:
    """Load a cache and reject stale video/pose-extractor provenance."""

    sequence, metadata, _ = load_pose_cache_with_receipt(
        path,
        expected_video_sha256=expected_video_sha256,
        expected_pose_fingerprint=expected_pose_fingerprint,
    )
    return sequence, metadata


def load_pose_cache_set(
    records: Sequence[UnlabeledVideoRecord],
    *,
    cache_dir: str | Path,
    pose_fingerprint: str,
    materialize_sequences: bool = True,
) -> tuple[tuple[PoseSequence, ...], PoseCacheSetSnapshot]:
    """Load and hash an ordered set of label-free pose caches exactly once."""

    source = tuple(records)
    if not source:
        raise ValueError("pose-cache set requires at least one record")
    identifiers = [record.video_id for record in source]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("pose-cache set video_id values must be unique")
    validated_pose = _validate_sha256(pose_fingerprint, "pose_fingerprint")
    if validated_pose is None:
        raise ValueError("pose_fingerprint is required")
    directory = Path(cache_dir)
    sequences: list[PoseSequence] = []
    receipts: list[PoseCacheEntryReceipt] = []
    for record in source:
        if record.video_sha256 is None:
            raise ValueError(f"video SHA-256 is required for {record.video_id!r}")
        sequence, _, receipt = load_pose_cache_with_receipt(
            pose_cache_path(directory, record.video_id),
            expected_video_sha256=record.video_sha256,
            expected_pose_fingerprint=validated_pose,
        )
        if sequence.video_id != record.video_id:
            raise ValueError("cache video_id does not match pose-cache record")
        if materialize_sequences:
            sequences.append(sequence)
        receipts.append(receipt)
    return (
        tuple(sequences),
        PoseCacheSetSnapshot(
            pose_fingerprint=validated_pose,
            entries=tuple(receipts),
        ),
    )


class TrainingPoseDataset(Sequence[PoseSequence]):
    """Label-free cached-pose dataset that rejects all sealed-test records."""

    def __init__(
        self,
        records: Sequence[UCFRepRecord],
        *,
        cache_dir: str | Path,
        pose_fingerprint: str,
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
        missing_hashes = sorted(record.video_id for record in source if record.video_sha256 is None)
        if missing_hashes:
            raise ValueError(
                "training requires content-hashed videos; missing video_sha256 for "
                f"{missing_hashes[:5]}"
            )
        self._items = tuple(
            UnlabeledVideoRecord(
                video_id=record.video_id,
                video_path=record.video_path,
                video_sha256=record.video_sha256,
            )
            for record in source
        )
        self._cache_dir = Path(cache_dir)
        validated = _validate_sha256(pose_fingerprint, "pose_fingerprint")
        if validated is None:
            raise ValueError("pose_fingerprint is required")
        self._pose_fingerprint = validated

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
            expected_pose_fingerprint=self._pose_fingerprint,
        )
        if sequence.video_id != item.video_id:
            raise ValueError("cache video_id does not match training record")
        return sequence

    def __iter__(self) -> Iterator[PoseSequence]:
        for index in range(len(self)):
            yield self[index]

    def materialize_snapshot(
        self,
    ) -> tuple[tuple[PoseSequence, ...], PoseCacheSetSnapshot]:
        """Load each training cache once and bind the exact consumed bytes."""

        return load_pose_cache_set(
            self._items,
            cache_dir=self._cache_dir,
            pose_fingerprint=self._pose_fingerprint,
            materialize_sequences=True,
        )

    def cache_snapshot(self) -> PoseCacheSetSnapshot:
        """Hash and validate the set without retaining pose tensors."""

        _, snapshot = load_pose_cache_set(
            self._items,
            cache_dir=self._cache_dir,
            pose_fingerprint=self._pose_fingerprint,
            materialize_sequences=False,
        )
        return snapshot
