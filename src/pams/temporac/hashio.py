"""Canonical hashes, JSON, files, NPY, and NPZ for TempoRAC.

The ZIP writer is deliberately small and uncompressed.  It writes fixed DOS
metadata directly because ``zipfile`` substitutes POSIX attributes when an
entry's external attributes are zero, while v4 requires literal zero.
"""

from __future__ import annotations

import binascii
import hashlib
import io
import json
import math
import os
import stat
import struct
import zipfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from pams.temporac.contract import (
    CONTRACT_SHA256,
    EFFECTIVE_CONTRACT_BYTES,
    EFFECTIVE_CONTRACT_KEYS,
    EFFECTIVE_CONTRACT_ROW_KEYS,
    EFFECTIVE_CONTRACT_ROWS,
    EFFECTIVE_CONTRACT_SCHEMA,
    FEATURE_ARCHIVE_MAX_BYTES,
    FEATURE_EXPANDED_MAX_BYTES,
    FEATURE_MEMBER_SCHEMA,
    JOB_NAMES,
    PRE_G5A_OWNER_ROSTER,
    PREDICTION_MEMBER_SCHEMA,
    SEEDS,
    TARGET_MEMBER_SCHEMA,
    ZIP_TIMESTAMP,
)
from pams.temporac.types import (
    ArrayMemberRecord,
    ArraySpec,
    CertifiedTarget,
    CheckpointReceiptOwner,
    ContractError,
    EffectiveContractRow,
    FeatureRecord,
    GenericArray,
    IdentityReceiptOwner,
    NamedReceiptOwner,
    PredictionReceiptOwner,
    PredictionRecord,
    ReceiptOwner,
    RunReceiptOwner,
    SchemaReceiptOwner,
    SeedReceiptOwner,
    TargetReceiptOwner,
)

_LOCAL_FILE_HEADER = struct.Struct("<IHHHHHIIIHH")
_CENTRAL_FILE_HEADER = struct.Struct("<IHHHHHHIIIHHHHHII")
_END_OF_CENTRAL_DIRECTORY = struct.Struct("<IHHHHIIH")
_LOCAL_SIGNATURE = 0x04034B50
_CENTRAL_SIGNATURE = 0x02014B50
_EOCD_SIGNATURE = 0x06054B50
_ZIP_VERSION = 20
_DOS_TIME = 0
_DOS_DATE = (1 << 5) | 1  # 1980-01-01
_READ_CHUNK = 1024 * 1024


class HashIOError(ContractError):
    """A hash, canonical byte, file, or archive contract violation."""


class ArchiveError(HashIOError):
    """A deterministic NPY/NPZ contract violation."""


def sha256_bytes(payload: bytes | bytearray | memoryview) -> str:
    """Return lowercase SHA-256 for an exact byte string."""

    return hashlib.sha256(bytes(payload)).hexdigest()


def raw_sha256(value: str, *, name: str = "sha256") -> bytes:
    """Decode one exact lowercase 64-character digest."""

    if len(value) != 64 or value != value.lower():
        raise HashIOError(f"{name} must be 64 lowercase hexadecimal characters")
    try:
        result = bytes.fromhex(value)
    except ValueError as exc:
        raise HashIOError(f"{name} is not hexadecimal") from exc
    if result.hex() != value:
        raise HashIOError(f"{name} is not canonical lowercase hexadecimal")
    return result


def uint16_be(value: int) -> bytes:
    if type(value) is not int or not 0 <= value < 2**16:
        raise HashIOError("uint16 value is outside its domain")
    return struct.pack(">H", value)


def uint32_be(value: int) -> bytes:
    if type(value) is not int or not 0 <= value < 2**32:
        raise HashIOError("uint32 value is outside its domain")
    return struct.pack(">I", value)


def uint64_be(value: int) -> bytes:
    if type(value) is not int or not 0 <= value < 2**64:
        raise HashIOError("uint64 value is outside its domain")
    return struct.pack(">Q", value)


def uint8(value: int) -> bytes:
    """Encode one exact unsigned byte for a typed hash preimage."""

    if type(value) is not int or not 0 <= value < 2**8:
        raise HashIOError("uint8 value is outside its domain")
    return bytes((value,))


def _field_bytes(value: bytes | bytearray | memoryview | str) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise HashIOError("H fields must be byte strings or strings")


def H(
    tag: str,
    *fields: bytes | bytearray | memoryview | str,
) -> bytes:
    """Compute the v4 length-prefixed field hash and return its raw 32 bytes."""

    if not isinstance(tag, str) or not tag.isascii() or "\x00" in tag:
        raise HashIOError("H tag must be nonempty NUL-free ASCII")
    digest = hashlib.sha256()
    digest.update(tag.encode("ascii"))
    digest.update(b"\x00")
    for field in fields:
        payload = _field_bytes(field)
        digest.update(uint64_be(len(payload)))
        digest.update(payload)
    return digest.digest()


def H_hex(tag: str, *fields: bytes | bytearray | memoryview | str) -> str:
    """Hexadecimal form of :func:`H`."""

    return H(tag, *fields).hex()


field_hash = H
field_hash_hex = H_hex


_IDENTITY_OWNER_CLASSES = frozenset({"feature", "natural-certificate-input"})
_CHECKPOINT_OWNER_CLASSES = frozenset({"teacher-checkpoint", "teacher-tune-evaluation"})
_SEED_OWNER_CLASSES = frozenset({"natural-prediction-completion", "x0-inference"})
_NAMED_OWNER_CLASSES = frozenset(
    {"g1-teacher-selection", "k1-certificate-outcome", "k3", "k4", "pre-g5a-stage"}
)
_SCHEMA_OWNER_CLASSES = MappingProxyType(
    {
        "teacher-selection": "temporac.teacher-selection-receipt.v4",
        "teacher-tune-input": "temporac.teacher-tune-input-receipt.v4",
    }
)
_PRE_G5A_CLASS_OWNER_PAIRS = frozenset(
    (node_class, owner) for node_class, owner, _schema, _upstream in PRE_G5A_OWNER_ROSTER
)


def receipt_owner_key(node_class: str, owner: ReceiptOwner) -> str:
    """Derive one class-specific v4 receipt owner key from typed fields.

    The caller never supplies an owner digest or a free-form joined identity.
    Every accepted node class selects exactly one frozen typed preimage layout.
    """

    if not isinstance(node_class, str) or not node_class.isascii():
        raise HashIOError("receipt owner node class must be ASCII")
    fields: tuple[bytes | str, ...]
    if node_class in _IDENTITY_OWNER_CLASSES:
        if not isinstance(owner, IdentityReceiptOwner):
            raise HashIOError("receipt owner fields do not match the identity node class")
        fields = (
            node_class,
            owner.split,
            raw_sha256(owner.opaque_key_hex, name="receipt owner opaque key"),
            uint64_be(owner.slot),
        )
    elif node_class in _CHECKPOINT_OWNER_CLASSES:
        if not isinstance(owner, CheckpointReceiptOwner):
            raise HashIOError("receipt owner fields do not match the checkpoint node class")
        if owner.seed not in SEEDS:
            raise HashIOError("checkpoint receipt owner seed is outside the frozen inventory")
        fields = (node_class, uint64_be(owner.seed), uint32_be(owner.step))
    elif node_class == "run":
        if not isinstance(owner, RunReceiptOwner):
            raise HashIOError("receipt owner fields do not match the run node class")
        if owner.job_name not in JOB_NAMES or owner.seed not in SEEDS:
            raise HashIOError("run receipt owner is outside the frozen job/seed inventory")
        if not owner.job_name.endswith(f"/seed={owner.seed}"):
            raise HashIOError("run receipt owner job and seed do not agree")
        fields = (node_class, owner.job_name.encode("ascii"), uint64_be(owner.seed))
    elif node_class == "prediction":
        if not isinstance(owner, PredictionReceiptOwner):
            raise HashIOError("receipt owner fields do not match the prediction node class")
        if owner.seed not in SEEDS:
            raise HashIOError("prediction receipt owner seed is outside the frozen inventory")
        fields = (
            node_class,
            owner.split,
            raw_sha256(owner.opaque_key_hex, name="receipt owner opaque key"),
            uint64_be(owner.slot),
            owner.arm,
            uint64_be(owner.seed),
            owner.condition,
        )
    elif node_class == "target":
        if not isinstance(owner, TargetReceiptOwner):
            raise HashIOError("receipt owner fields do not match the target node class")
        fields = (
            node_class,
            uint8(owner.source_kind),
            raw_sha256(owner.source_key_hex, name="receipt owner source key"),
            uint64_be(owner.source_unit_index),
        )
    elif node_class in _SEED_OWNER_CLASSES:
        if not isinstance(owner, SeedReceiptOwner):
            raise HashIOError("receipt owner fields do not match the seed node class")
        if owner.seed not in SEEDS:
            raise HashIOError("receipt owner seed is outside the frozen inventory")
        fields = (node_class, uint64_be(owner.seed))
    elif node_class in _NAMED_OWNER_CLASSES:
        if not isinstance(owner, NamedReceiptOwner):
            raise HashIOError("receipt owner fields do not match the named node class")
        if (node_class, owner.owner) not in _PRE_G5A_CLASS_OWNER_PAIRS:
            raise HashIOError("named receipt owner is outside the exact pre-G5a roster")
        fields = (node_class, owner.owner)
    elif node_class in _SCHEMA_OWNER_CLASSES:
        if not isinstance(owner, SchemaReceiptOwner):
            raise HashIOError("receipt owner fields do not match the schema node class")
        if owner.schema != _SCHEMA_OWNER_CLASSES[node_class]:
            raise HashIOError("schema receipt owner is not the exact class schema")
        fields = (node_class, owner.schema)
    else:
        raise HashIOError("receipt owner node class is outside the frozen token set")
    return H_hex("temporac.receipt-owner.v4", *fields)


def opaque_sample_key(split: str, source_pickle_sha256: str | bytes, ordinal: int) -> bytes:
    """Derive the v4 opaque source-object key (not the length-prefixed H form)."""

    if split not in {"train", "val"}:
        raise HashIOError("split must be exact ASCII train or val")
    source_hash = (
        raw_sha256(source_pickle_sha256, name="source pickle sha256")
        if isinstance(source_pickle_sha256, str)
        else bytes(source_pickle_sha256)
    )
    if len(source_hash) != 32:
        raise HashIOError("source pickle hash must contain exactly 32 bytes")
    return hashlib.sha256(
        b"temporac.opaque-key.v4\x00"
        + split.encode("ascii")
        + b"\x00"
        + source_hash
        + uint32_be(ordinal)
    ).digest()


def source_binding_sha256(
    split: str,
    source_pickle_sha256: str | bytes,
    object_ordinal: int,
    slot: int,
) -> str:
    source_hash = (
        raw_sha256(source_pickle_sha256, name="source pickle sha256")
        if isinstance(source_pickle_sha256, str)
        else bytes(source_pickle_sha256)
    )
    if split not in {"train", "val"} or len(source_hash) != 32:
        raise HashIOError("source binding has an invalid split or source hash")
    return H_hex(
        "temporac.source-binding.v4",
        split,
        source_hash,
        uint32_be(object_ordinal),
        uint32_be(slot),
    )


def component_key(split: str, canonical_source_id_digests: Sequence[str | bytes]) -> bytes:
    """Derive a component key from sorted distinct raw canonical-ID digests."""

    if split not in {"train", "val"}:
        raise HashIOError("component split must be exact ASCII train or val")
    digests: set[bytes] = set()
    for value in canonical_source_id_digests:
        raw = (
            raw_sha256(value, name="canonical source ID digest")
            if isinstance(value, str)
            else bytes(value)
        )
        if len(raw) != 32:
            raise HashIOError("canonical source ID digests must contain exactly 32 bytes")
        digests.add(raw)
    if not digests:
        raise HashIOError("a component must contain at least one canonical source ID")
    return hashlib.sha256(
        b"temporac.component.v4\x00" + split.encode("ascii") + b"\x00" + b"".join(sorted(digests))
    ).digest()


def x0_source_key(source_id: int) -> bytes:
    """Derive the key for one exact X0 source ID ``U0000..U0039``."""

    if type(source_id) is not int or not 0 <= source_id < 40:
        raise HashIOError("X0 source ID must be an integer in 0..39")
    return hashlib.sha256(b"temporac.x0-unit.v4\x00" + uint32_be(source_id)).digest()


def prediction_pair_root(clean_root_sha256: str, drift_root_sha256: str) -> str:
    return H_hex(
        "temporac.prediction-pair-root.v4",
        raw_sha256(clean_root_sha256, name="clean prediction root"),
        raw_sha256(drift_root_sha256, name="drift prediction root"),
    )


def _reject_negative_zero_tree(payload: Any, *, _active: set[int] | None = None) -> None:
    if isinstance(payload, float):
        if payload == 0.0 and math.copysign(1.0, payload) < 0.0:
            raise HashIOError("negative-zero JSON floats are forbidden")
        return
    if not isinstance(payload, (Mapping, list, tuple)):
        return
    active = set() if _active is None else _active
    identity = id(payload)
    if identity in active:
        raise HashIOError("cyclic JSON containers are forbidden")
    active.add(identity)
    try:
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                _reject_negative_zero_tree(key, _active=active)
                _reject_negative_zero_tree(value, _active=active)
        else:
            for value in payload:
                _reject_negative_zero_tree(value, _active=active)
    finally:
        active.remove(identity)


def canonical_json_bytes(payload: Any) -> bytes:
    """Encode canonical v4 JSON with exactly one terminal LF."""

    _reject_negative_zero_tree(payload)
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HashIOError("payload is not canonical-JSON serializable") from exc
    return encoded + b"\n"


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HashIOError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise HashIOError(f"nonfinite JSON number is forbidden: {value}")


def parse_strict_json_bytes(
    payload: bytes | bytearray | memoryview,
    *,
    canonical: bool,
    max_bytes: int | None = None,
) -> Any:
    """Parse duplicate-free UTF-8 JSON, optionally requiring exact v4 bytes."""

    encoded = bytes(payload)
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
        raise HashIOError("JSON max_bytes must be a nonnegative integer or None")
    if max_bytes is not None and len(encoded) > max_bytes:
        raise HashIOError(f"JSON payload exceeds {max_bytes} bytes")
    if encoded.startswith(b"\xef\xbb\xbf"):
        raise HashIOError("JSON payload must not contain a UTF-8 BOM")
    if canonical and (not encoded.endswith(b"\n") or encoded.endswith(b"\n\n")):
        raise HashIOError("canonical JSON must have exactly one terminal LF")
    try:
        decoded = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except HashIOError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HashIOError("payload is not duplicate-free UTF-8 JSON") from exc
    _reject_negative_zero_tree(decoded)
    if canonical and canonical_json_bytes(decoded) != encoded:
        raise HashIOError("JSON bytes are not in the canonical v4 form")
    return decoded


def effective_contract_rows() -> tuple[EffectiveContractRow, ...]:
    """Return the frozen five rows in accepted role order."""

    return tuple(
        EffectiveContractRow(bytes=byte_count, role=role, sha256=digest)
        for byte_count, role, digest in EFFECTIVE_CONTRACT_ROWS
    )


def effective_contract_index_bytes() -> bytes:
    """Build the sole accepted effective-contract canonical index bytes."""

    encoded = canonical_json_bytes(
        {
            "rows": [row.as_dict() for row in effective_contract_rows()],
            "schema": EFFECTIVE_CONTRACT_SCHEMA,
        }
    )
    if len(encoded) != EFFECTIVE_CONTRACT_BYTES or sha256_bytes(encoded) != CONTRACT_SHA256:
        raise HashIOError("effective-contract constants do not reproduce the accepted index")
    return encoded


def parse_effective_contract_index_bytes(
    payload: bytes | bytearray | memoryview,
) -> tuple[EffectiveContractRow, ...]:
    """Validate exact accepted effective-contract bytes and return typed rows."""

    encoded = bytes(payload)
    decoded = parse_strict_json_bytes(encoded, canonical=True, max_bytes=EFFECTIVE_CONTRACT_BYTES)
    if not isinstance(decoded, dict) or set(decoded) != EFFECTIVE_CONTRACT_KEYS:
        raise HashIOError("effective-contract index has unknown or missing keys")
    rows = decoded.get("rows")
    if decoded.get("schema") != EFFECTIVE_CONTRACT_SCHEMA or not isinstance(rows, list):
        raise HashIOError("effective-contract schema or rows are invalid")
    typed: list[EffectiveContractRow] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != EFFECTIVE_CONTRACT_ROW_KEYS:
            raise HashIOError("effective-contract row has unknown or missing keys")
        try:
            typed.append(
                EffectiveContractRow(
                    bytes=cast(int, row["bytes"]),
                    role=cast(str, row["role"]),
                    sha256=cast(str, row["sha256"]),
                )
            )
        except ContractError as exc:
            raise HashIOError("effective-contract row has an invalid type or value") from exc
    result = tuple(typed)
    if result != effective_contract_rows() or encoded != effective_contract_index_bytes():
        raise HashIOError("effective-contract index differs from the accepted five-row index")
    return result


def standalone_json_sha256(payload: Mapping[str, object]) -> str:
    """Hash an exact standalone canonical object, including its terminal LF."""

    return sha256_bytes(canonical_json_bytes(payload))


def read_regular_file(path: Path, *, max_bytes: int | None = None) -> bytes:
    """Read a stable regular file while rejecting links and concurrent edits."""

    candidate = Path(path)
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
        raise HashIOError("max_bytes must be a nonnegative integer or None")
    try:
        before = candidate.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required file does not exist: {candidate}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise HashIOError(f"required path is not a regular non-symlink file: {candidate}")
    if max_bytes is not None and before.st_size > max_bytes:
        raise HashIOError(f"required file exceeds {max_bytes} bytes: {candidate}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(candidate, flags)
    try:
        opened = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read() if max_bytes is None else handle.read(max_bytes + 1)
            closed = os.fstat(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after = candidate.lstat()
    if stat.S_ISLNK(after.st_mode) or not stat.S_ISREG(after.st_mode):
        raise HashIOError("required file changed type while it was read")
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field) != getattr(opened, field)
        or getattr(opened, field) != getattr(closed, field)
        or getattr(closed, field) != getattr(after, field)
        for field in stable_fields
    ):
        raise HashIOError(f"required file changed while it was read: {candidate}")
    if len(payload) != after.st_size:
        raise HashIOError(f"required file size changed while it was read: {candidate}")
    if max_bytes is not None and len(payload) > max_bytes:
        raise HashIOError(f"required file exceeds {max_bytes} bytes: {candidate}")
    return payload


def sha256_file(path: Path, *, max_bytes: int | None = None) -> str:
    """Hash one stable regular non-symlink file."""

    if max_bytes is not None:
        return sha256_bytes(read_regular_file(path, max_bytes=max_bytes))
    # Preserve the stable-file semantics while avoiding an unnecessary second
    # implementation of the race checks.
    return sha256_bytes(read_regular_file(path))


def _regular_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise HashIOError(f"parent directory does not exist: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise HashIOError(f"parent must be a non-symlink directory: {path}")


def write_bytes_exclusive(path: Path, payload: bytes, *, mode: int = 0o444) -> str:
    """Durably create one regular artifact with ``O_CREAT|O_EXCL``."""

    candidate = Path(path)
    _regular_directory(candidate.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags, mode)
    except FileExistsError as exc:
        raise HashIOError(f"refusing to overwrite frozen artifact: {candidate}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise HashIOError("exclusive artifact target is not a regular file")
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        with suppress(FileNotFoundError):
            candidate.unlink()
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after = candidate.lstat()
    if stat.S_ISLNK(after.st_mode) or not stat.S_ISREG(after.st_mode):
        with suppress(FileNotFoundError):
            candidate.unlink()
        raise HashIOError("published artifact is not a regular non-symlink file")
    with suppress(OSError):
        candidate.chmod(mode)
    return sha256_bytes(payload)


def write_canonical_json_exclusive(path: Path, payload: Any, *, mode: int = 0o444) -> str:
    return write_bytes_exclusive(path, canonical_json_bytes(payload), mode=mode)


def load_canonical_json(path: Path) -> Any:
    """Load JSON only when its exact bytes already satisfy the v4 encoding."""

    return parse_strict_json_bytes(read_regular_file(path), canonical=True)


def npy_v2_bytes(array: GenericArray) -> bytes:
    """Serialize one safe C-order array using the NPY 2.0 format."""

    if not isinstance(array, np.ndarray):
        raise ArchiveError("NPY payload must be a NumPy array")
    if array.dtype.hasobject or array.dtype.kind in {"O", "U", "V"}:
        raise ArchiveError("object, Unicode, structured, and void arrays are forbidden")
    if array.dtype.itemsize > 1 and array.dtype.byteorder == ">":
        raise ArchiveError("multi-byte artifact arrays must be little-endian")
    contiguous = np.ascontiguousarray(array)
    handle = io.BytesIO()
    np.lib.format.write_array(handle, contiguous, version=(2, 0), allow_pickle=False)
    return handle.getvalue()


def _flat_ascii_name(name: str) -> str:
    if not isinstance(name, str) or not name or not name.isascii():
        raise ArchiveError("NPZ field names must be nonempty ASCII")
    if name.endswith(".npy"):
        name = name[:-4]
    member_name = f"{name}.npy"
    pure = PurePosixPath(member_name)
    if (
        "\\" in member_name
        or pure.is_absolute()
        or len(pure.parts) != 1
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ArchiveError("NPZ members must be flat .npy names")
    return member_name


def _normalize_schema(
    schema: Mapping[str, ArraySpec | tuple[str, tuple[int | None, ...]]] | None,
) -> dict[str, ArraySpec] | None:
    if schema is None:
        return None
    normalized: dict[str, ArraySpec] = {}
    for raw_name, raw_spec in schema.items():
        member_name = _flat_ascii_name(raw_name)
        name = member_name.removesuffix(".npy")
        if name in normalized:
            raise ArchiveError("schema has duplicate normalized member names")
        normalized[name] = raw_spec if isinstance(raw_spec, ArraySpec) else ArraySpec(*raw_spec)
    return normalized


def deterministic_npz_bytes(
    arrays: Mapping[str, GenericArray],
    *,
    schema: Mapping[str, ArraySpec | tuple[str, tuple[int | None, ...]]] | None = None,
    max_archive_bytes: int = FEATURE_ARCHIVE_MAX_BYTES,
    max_expanded_bytes: int = FEATURE_EXPANDED_MAX_BYTES,
) -> tuple[bytes, tuple[ArrayMemberRecord, ...]]:
    """Write fixed-metadata ZIP_STORED NPY-v2 members in bytewise ASCII order."""

    normalized_schema = _normalize_schema(schema)
    normalized_arrays: dict[str, GenericArray] = {}
    for raw_name, array in arrays.items():
        name = _flat_ascii_name(raw_name).removesuffix(".npy")
        if name in normalized_arrays:
            raise ArchiveError("duplicate normalized NPZ member name")
        if not isinstance(array, np.ndarray):
            raise ArchiveError(f"{name} must be a NumPy array")
        normalized_arrays[name] = array
    if normalized_schema is not None and set(normalized_arrays) != set(normalized_schema):
        raise ArchiveError("NPZ fields do not match the exact schema")
    ordered_names = sorted(normalized_arrays, key=lambda value: value.encode("ascii"))
    output = io.BytesIO()
    central: list[tuple[bytes, int, int, int, int]] = []
    receipts: list[ArrayMemberRecord] = []
    expanded = 0
    for name in ordered_names:
        array = normalized_arrays[name]
        if normalized_schema is not None:
            normalized_schema[name].validate(array, name=name)
        member = npy_v2_bytes(array)
        expanded += len(member)
        if expanded > max_expanded_bytes:
            raise ArchiveError("NPZ expanded payload exceeds the fixed byte ceiling")
        encoded_name = f"{name}.npy".encode("ascii")
        crc = binascii.crc32(member) & 0xFFFFFFFF
        offset = output.tell()
        output.write(
            _LOCAL_FILE_HEADER.pack(
                _LOCAL_SIGNATURE,
                _ZIP_VERSION,
                0,
                0,
                _DOS_TIME,
                _DOS_DATE,
                crc,
                len(member),
                len(member),
                len(encoded_name),
                0,
            )
        )
        output.write(encoded_name)
        output.write(member)
        central.append((encoded_name, crc, len(member), len(member), offset))
        receipts.append(
            ArrayMemberRecord(
                bytes=len(member),
                dtype=array.dtype.str,
                name=encoded_name.decode("ascii"),
                sha256=sha256_bytes(member),
                shape=tuple(int(axis) for axis in array.shape),
            )
        )
    central_offset = output.tell()
    for encoded_name, crc, compressed_size, file_size, offset in central:
        output.write(
            _CENTRAL_FILE_HEADER.pack(
                _CENTRAL_SIGNATURE,
                _ZIP_VERSION,  # create system is DOS (high byte zero)
                _ZIP_VERSION,
                0,
                0,
                _DOS_TIME,
                _DOS_DATE,
                crc,
                compressed_size,
                file_size,
                len(encoded_name),
                0,
                0,
                0,
                0,
                0,  # literal zero external attributes
                offset,
            )
        )
        output.write(encoded_name)
    central_size = output.tell() - central_offset
    output.write(
        _END_OF_CENTRAL_DIRECTORY.pack(
            _EOCD_SIGNATURE,
            0,
            0,
            len(central),
            len(central),
            central_size,
            central_offset,
            0,
        )
    )
    payload = output.getvalue()
    if len(payload) > max_archive_bytes:
        raise ArchiveError("NPZ archive exceeds the fixed byte ceiling")
    return payload, tuple(receipts)


def _load_npy_member(member: bytes, *, name: str) -> GenericArray:
    handle = io.BytesIO(member)
    try:
        version = np.lib.format.read_magic(handle)
        if version != (2, 0):
            raise ArchiveError(f"{name} must use NPY version 2.0")
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
        if fortran_order:
            raise ArchiveError(f"{name} must use C order")
        parsed_dtype = np.dtype(dtype)
        if parsed_dtype.hasobject or parsed_dtype.kind in {"O", "U", "V"}:
            raise ArchiveError(f"{name} has a forbidden dtype")
        expected_bytes = math.prod(int(axis) for axis in shape) * parsed_dtype.itemsize
        if len(member) - handle.tell() != expected_bytes:
            raise ArchiveError(f"{name} has a noncanonical NPY payload length")
        backing = handle.read()
        array = np.frombuffer(backing, dtype=parsed_dtype).reshape(shape, order="C")
    except (EOFError, ValueError, TypeError) as exc:
        if isinstance(exc, ArchiveError):
            raise
        raise ArchiveError(f"{name} is not a valid non-pickle NPY 2.0 member") from exc
    return cast(GenericArray, array)


def read_deterministic_npz_bytes(
    payload: bytes,
    *,
    schema: Mapping[str, ArraySpec | tuple[str, tuple[int | None, ...]]],
    max_archive_bytes: int = FEATURE_ARCHIVE_MAX_BYTES,
    max_expanded_bytes: int = FEATURE_EXPANDED_MAX_BYTES,
) -> tuple[Mapping[str, GenericArray], tuple[ArrayMemberRecord, ...]]:
    """Fail closed on any noncanonical archive before returning arrays."""

    if len(payload) > max_archive_bytes:
        raise ArchiveError("NPZ archive exceeds the fixed byte ceiling")
    normalized_schema = cast(dict[str, ArraySpec], _normalize_schema(schema))
    expected_names = [
        f"{name}.npy" for name in sorted(normalized_schema, key=lambda value: value.encode("ascii"))
    ]
    arrays: dict[str, GenericArray] = {}
    receipts: list[ArrayMemberRecord] = []
    expanded = 0
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            if archive.comment != b"":
                raise ArchiveError("NPZ archive comment must be empty")
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or names != expected_names:
                raise ArchiveError("NPZ members must be the exact bytewise-sorted schema")
            for info in infos:
                if (
                    info.date_time != ZIP_TIMESTAMP
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.flag_bits != 0
                    or info.extra != b""
                    or info.comment != b""
                    or info.create_system != 0
                    or info.external_attr != 0
                    or info.internal_attr != 0
                    or info.volume != 0
                    or info.is_dir()
                ):
                    raise ArchiveError("NPZ member metadata is not the exact fixed form")
                name = _flat_ascii_name(info.filename).removesuffix(".npy")
                expanded += info.file_size
                if expanded > max_expanded_bytes or info.compress_size != info.file_size:
                    raise ArchiveError("NPZ expanded payload violates the fixed size/storage form")
                member = archive.read(info)
                if len(member) != info.file_size:
                    raise ArchiveError("NPZ member size changed during inspection")
                array = _load_npy_member(member, name=name)
                normalized_schema[name].validate(array, name=name)
                arrays[name] = array
                receipts.append(
                    ArrayMemberRecord(
                        bytes=len(member),
                        dtype=array.dtype.str,
                        name=info.filename,
                        sha256=sha256_bytes(member),
                        shape=tuple(int(axis) for axis in array.shape),
                    )
                )
    except zipfile.BadZipFile as exc:
        raise ArchiveError("artifact is not a valid ZIP archive") from exc
    canonical, _ = deterministic_npz_bytes(
        arrays,
        schema=schema,
        max_archive_bytes=max_archive_bytes,
        max_expanded_bytes=max_expanded_bytes,
    )
    if canonical != payload:
        raise ArchiveError("NPZ bytes are not the unique deterministic encoding")
    return MappingProxyType(arrays), tuple(receipts)


def read_deterministic_npz(
    path: Path,
    *,
    schema: Mapping[str, ArraySpec | tuple[str, tuple[int | None, ...]]],
    max_archive_bytes: int = FEATURE_ARCHIVE_MAX_BYTES,
    max_expanded_bytes: int = FEATURE_EXPANDED_MAX_BYTES,
) -> tuple[Mapping[str, GenericArray], tuple[ArrayMemberRecord, ...]]:
    payload = read_regular_file(path, max_bytes=max_archive_bytes)
    return read_deterministic_npz_bytes(
        payload,
        schema=schema,
        max_archive_bytes=max_archive_bytes,
        max_expanded_bytes=max_expanded_bytes,
    )


def write_deterministic_npz_exclusive(
    path: Path,
    arrays: Mapping[str, GenericArray],
    *,
    schema: Mapping[str, ArraySpec | tuple[str, tuple[int | None, ...]]],
    mode: int = 0o444,
) -> tuple[str, tuple[ArrayMemberRecord, ...]]:
    payload, members = deterministic_npz_bytes(arrays, schema=schema)
    return write_bytes_exclusive(path, payload, mode=mode), members


def feature_npz_bytes(
    record: FeatureRecord,
) -> tuple[bytes, tuple[ArrayMemberRecord, ...]]:
    return deterministic_npz_bytes(record.as_arrays(), schema=FEATURE_MEMBER_SCHEMA)


def target_npz_bytes(
    target: CertifiedTarget,
) -> tuple[bytes, tuple[ArrayMemberRecord, ...]]:
    """Serialize the exact seven-member artifact for one certified target."""

    return deterministic_npz_bytes(
        target.artifact_arrays(),
        schema=TARGET_MEMBER_SCHEMA,
    )


def prediction_npz_bytes(
    record: PredictionRecord,
) -> tuple[bytes, tuple[ArrayMemberRecord, ...]]:
    return deterministic_npz_bytes(record.as_arrays(), schema=PREDICTION_MEMBER_SCHEMA)


def load_feature_record(path: Path) -> FeatureRecord:
    arrays, _ = read_deterministic_npz(path, schema=FEATURE_MEMBER_SCHEMA)
    return FeatureRecord(**cast(Any, dict(arrays)))


def _validate_target_arrays(
    arrays: Mapping[str, GenericArray],
) -> Mapping[str, GenericArray]:
    chi = arrays["chi"]
    edge_mask = arrays["edge_mask"]
    pulse = arrays["pulse"]
    source_kind = arrays["source_kind"]
    target_mask = arrays["target_mask"]
    edge_count = chi.shape[0]
    if edge_count == 0:
        raise ArchiveError("certified target artifact must contain at least one edge")
    if any(array.shape != (edge_count,) for array in (edge_mask, pulse, target_mask)):
        raise ArchiveError("target variable member axes do not share one edge count")
    if arrays["contract_sha256"].tobytes(order="C") != bytes.fromhex(CONTRACT_SHA256):
        raise ArchiveError("target artifact does not bind temporac.execution.v4")
    if int(source_kind[0]) not in {0, 1}:
        raise ArchiveError("target source_kind must be integer zero or one")
    if not (
        np.isin(edge_mask, (0, 1)).all()
        and np.isin(pulse, (0, 1)).all()
        and np.isin(target_mask, (0, 1)).all()
    ):
        raise ArchiveError("target masks and pulse must be binary")
    if (
        not np.isfinite(chi).all()
        or np.any(np.less(chi, 0.0))
        or np.any(np.greater(pulse, target_mask))
        or np.any(np.greater(target_mask, edge_mask))
        or not np.array_equal(target_mask, edge_mask)
        or np.any(chi[target_mask == 0] != 0.0)
        or np.any(np.less_equal(chi[pulse == 1], 0.0))
    ):
        raise ArchiveError("target chi or mask support is invalid")
    return arrays


def read_target_npz_bytes(
    payload: bytes,
) -> tuple[Mapping[str, GenericArray], tuple[ArrayMemberRecord, ...]]:
    """Read an exact seven-member nonempty target NPZ without inventing audit fields."""

    arrays, members = read_deterministic_npz_bytes(payload, schema=TARGET_MEMBER_SCHEMA)
    return _validate_target_arrays(arrays), members


def load_target_arrays(path: Path) -> Mapping[str, GenericArray]:
    """Load and semantically validate one exact certified-target artifact."""

    payload = read_regular_file(path, max_bytes=FEATURE_ARCHIVE_MAX_BYTES)
    arrays, _ = read_target_npz_bytes(payload)
    return arrays


def load_prediction_record(path: Path) -> PredictionRecord:
    arrays, _ = read_deterministic_npz(path, schema=PREDICTION_MEMBER_SCHEMA)
    return PredictionRecord(**cast(Any, dict(arrays)))


def hash_ordered_bytes(payloads: Sequence[bytes]) -> str:
    """Hash an explicitly ordered byte sequence with the v4 H primitive."""

    return H_hex("temporac.ordered-bytes.v4", *payloads)
