"""Official UCFRep annotation acquisition and canonical manifest construction.

The original repository distributes annotations but not the UCF101 video
files.  This module downloads only that annotation archive, verifies its
content hash, resolves the three supported UCF101 layout families (including
one audited official directory-name alias), and derives the exact 421/105
manifest used by this project.
"""

from __future__ import annotations

import io
import hashlib
import os
import re
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pams.data import (
    UCFRepManifest,
    UCFRepRecord,
    split_ucfrep_train_dev,
)
from pams.reproducibility import sha256_file

OFFICIAL_ANNOTATION_URL = (
    "https://raw.githubusercontent.com/"
    "Xiaodomgdomg/Deep-Temporal-Repetition-Counting/master/"
    "data/ori_data/ucf526/ucf526_annotations.zip"
)
OFFICIAL_ANNOTATION_SHA256 = "08b2b8a88c2728e9aa6de6c62dc6b02f1941dfa2c3adb28dab5852c12695ae3c"
_VIDEO_PATTERN = re.compile(r"^v_(?P<action>.+)_g(?P<group>\d{2})_c\d+$")
_UCF101_DIRECTORY_ALIASES: dict[str, tuple[str, ...]] = {
    # UCFRep identifiers use ``HandStandPushups`` while the official UCF101
    # archive stores all 24 matching files under ``HandstandPushups``.
    "HandStandPushups": ("HandstandPushups",),
}


def download_official_annotations(
    destination: str | Path,
    *,
    timeout_seconds: float = 60.0,
) -> Path:
    """Download the small official annotation ZIP once and verify SHA-256."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        digest = sha256_file(target)
        if digest != OFFICIAL_ANNOTATION_SHA256:
            raise ValueError(f"existing annotation archive has unexpected SHA-256: {digest}")
        return target

    request = urllib.request.Request(
        OFFICIAL_ANNOTATION_URL,
        headers={"User-Agent": "pams-rac-reproduction/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read()
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    if digest != OFFICIAL_ANNOTATION_SHA256:
        raise ValueError(f"downloaded annotation archive has unexpected SHA-256: {digest}")

    with tempfile.NamedTemporaryFile(
        "wb", dir=target.parent, delete=False, suffix=".zip"
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _matlab_positive_integer(label: object, field: str) -> int:
    if not hasattr(label, field):
        raise ValueError(f"UCFRep label is missing {field}")
    values = np.asarray(getattr(label, field)).reshape(-1)
    if values.size != 1:
        raise ValueError(f"UCFRep label {field} must contain exactly one value")
    value = values[0]
    if isinstance(value, bool | np.bool_) or not np.isfinite(value):
        raise ValueError(f"UCFRep label {field} must be a finite positive integer")
    integer = int(value)
    if integer != value or integer < 1:
        raise ValueError(f"UCFRep label {field} must be a positive integer")
    return integer


def _annotation_metadata(payload: bytes) -> tuple[int, int, int, str]:
    """Return count and the audited 0-based half-open official input clip.

    The MATLAB annotation stores ``start_frame`` and ``end_frame`` as
    1-based inclusive indices. They are converted once at this privileged
    boundary to ``[start - 1, end)``. Cycle boundaries are used only to
    recover the evaluator count and are never copied into pose-input
    sidecars.
    """

    annotation = loadmat(
        io.BytesIO(payload),
        squeeze_me=True,
        struct_as_record=False,
    )
    if "label" not in annotation:
        raise ValueError("UCFRep .mat annotation is missing 'label'")
    label = annotation["label"]
    if not hasattr(label, "temporal_bound"):
        raise ValueError("UCFRep label is missing temporal_bound")
    boundaries = np.asarray(label.temporal_bound).reshape(-1)
    count = int(boundaries.size - 1)
    if count <= 0:
        raise ValueError("UCFRep annotation must contain at least one cycle")
    start_inclusive = _matlab_positive_integer(label, "start_frame")
    end_inclusive = _matlab_positive_integer(label, "end_frame")
    if end_inclusive < start_inclusive:
        raise ValueError("UCFRep label end_frame must not precede start_frame")
    return (
        count,
        start_inclusive - 1,
        end_inclusive,
        hashlib.sha256(payload).hexdigest(),
    )


def _annotation_count(payload: bytes) -> int:
    """Read the original MATLAB label and return ``len(boundaries) - 1``."""

    return _annotation_metadata(payload)[0]


def _video_path_candidates(
    *,
    video_root: Path,
    action: str,
    filename: str,
) -> tuple[Path, ...]:
    """Return candidates from the three audited UCF101 layout families."""

    directory_names = (action, *_UCF101_DIRECTORY_ALIASES.get(action, ()))
    nested = tuple(video_root / "UCF-101" / name / filename for name in directory_names)
    action_relative = tuple(video_root / name / filename for name in directory_names)
    return (
        *nested,
        *action_relative,
        video_root / "flat" / filename,
    )


def resolve_ucfrep_video_path(
    video_root: str | Path,
    *,
    video_id: str,
    action: str,
    video_suffix: str = ".avi",
    allow_missing: bool = False,
) -> Path:
    """Resolve one UCFRep video without guessing between ambiguous layouts.

    Supported layouts relative to ``video_root`` are:

    - ``UCF-101/<Action>/<video>.avi``;
    - ``<Action>/<video>.avi`` (including when ``video_root`` is UCF-101);
    - ``flat/<video>.avi``.

    The two action-directory forms also recognize the single audited official
    UCF101 alias ``HandStandPushups`` -> ``HandstandPushups``. No
    case-insensitive or recursive fallback is attempted.

    With ``allow_missing=True``, a missing video receives the deterministic
    ``<Action>/<video>.avi`` placeholder used by annotation-only manifests.
    Multiple existing candidates are always an error, including in
    annotation-only mode.
    """

    # Persist an absolute identity in the manifest.  Pose extraction resolves
    # relative record paths against the manifest directory, so allowing a
    # caller-CWD-relative video root here would silently change its meaning
    # after the manifest is moved.
    root = Path(video_root).expanduser().resolve(strict=False)
    identifier = str(video_id).strip()
    action_name = str(action).strip()
    if not identifier or not action_name:
        raise ValueError("video_id and action must be non-empty")
    if not video_suffix.startswith(".") or Path(video_suffix).name != video_suffix:
        raise ValueError("video_suffix must be a filename suffix beginning with a dot")
    if any(separator in action_name for separator in ("/", "\\")):
        raise ValueError("action must be a directory name, not a path")

    filename = f"{identifier}{video_suffix}"
    candidates = _video_path_candidates(
        video_root=root,
        action=action_name,
        filename=filename,
    )
    unique_matches: list[Path] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if any(os.path.samefile(candidate, existing) for existing in unique_matches):
            continue
        unique_matches.append(candidate)
    matches = tuple(unique_matches)
    if len(matches) > 1:
        rendered = ", ".join(path.as_posix() for path in matches)
        raise ValueError(
            f"ambiguous UCFRep video path for {identifier!r}; "
            f"multiple supported layouts contain the file: {rendered}"
        )
    if matches:
        return matches[0]
    if allow_missing:
        # The action-relative form remains useful if VIDEO_ROOT is later set
        # directly to the extracted UCF-101 directory.
        return root / action_name / filename
    searched = ", ".join(path.as_posix() for path in candidates)
    raise FileNotFoundError(
        f"missing UCFRep video {identifier!r}; searched supported layouts: {searched}"
    )


def _record_from_annotation(
    *,
    archive_name: str,
    payload: bytes,
    video_root: Path,
    video_suffix: str,
    hash_existing_videos: bool,
    allow_missing_videos: bool,
) -> UCFRepRecord:
    prefix, filename = archive_name.rsplit("/", 1)
    source_split = prefix.rsplit("/", 1)[-1]
    split = "train" if source_split == "train" else "test"
    video_id = Path(filename).stem
    match = _VIDEO_PATTERN.fullmatch(video_id)
    if match is None:
        raise ValueError(f"unexpected UCFRep video identifier: {video_id}")
    group = int(match.group("group"))
    if split == "train" and not 1 <= group <= 20:
        raise ValueError(f"train record has non-training source group: {video_id}")
    if split == "test" and not 21 <= group <= 25:
        raise ValueError(f"test record has non-test source group: {video_id}")

    action = match.group("action")
    video_path = resolve_ucfrep_video_path(
        video_root,
        video_id=video_id,
        action=action,
        video_suffix=video_suffix,
        allow_missing=allow_missing_videos,
    )
    video_digest = (
        sha256_file(video_path) if hash_existing_videos and video_path.is_file() else None
    )
    count, clip_start, clip_end, annotation_digest = _annotation_metadata(payload)
    return UCFRepRecord(
        video_id=video_id,
        video_path=video_path.as_posix(),
        split=split,
        action=action,
        count=count,
        video_sha256=video_digest,
        annotation_sha256=annotation_digest,
        clip_start_frame=clip_start,
        clip_end_frame=clip_end,
    )


def build_official_manifest(
    annotation_archive: str | Path,
    *,
    video_root: str | Path,
    video_suffix: str = ".avi",
    hash_existing_videos: bool = True,
    allow_missing_videos: bool = False,
) -> UCFRepManifest:
    """Build and validate the official 421/105 UCFRep manifest.

    The default is experiment-strict: all 526 videos must resolve uniquely and
    are content-hashed. ``hash_existing_videos=False`` skips hashing but still
    requires every file. Annotation-only consumers must explicitly pass
    ``allow_missing_videos=True``; any files that do exist are still resolved
    and optionally hashed.
    """

    archive_path = Path(annotation_archive)
    digest = sha256_file(archive_path)
    if digest != OFFICIAL_ANNOTATION_SHA256:
        raise ValueError(f"annotation archive SHA-256 mismatch: {digest}")
    if not video_suffix.startswith(".") or Path(video_suffix).name != video_suffix:
        raise ValueError("video_suffix must be a filename suffix beginning with a dot")

    records: list[UCFRepRecord] = []
    root = Path(video_root).expanduser().resolve(strict=False)
    if not allow_missing_videos and not root.is_dir():
        raise FileNotFoundError(f"video_root is not an existing directory: {root}")
    with zipfile.ZipFile(archive_path) as archive:
        annotation_names = sorted(
            name
            for name in archive.namelist()
            if (name.startswith("annotations/train/") or name.startswith("annotations/val/"))
            and name.endswith(".mat")
        )
        for name in annotation_names:
            records.append(
                _record_from_annotation(
                    archive_name=name,
                    payload=archive.read(name),
                    video_root=root,
                    video_suffix=video_suffix,
                    hash_existing_videos=hash_existing_videos,
                    allow_missing_videos=allow_missing_videos,
                )
            )

    manifest = UCFRepManifest(protocol="ucfrep_526", records=tuple(records))
    manifest.validate_exact_official_splits()
    return manifest


def materialize_standard_split_ids(
    manifest: UCFRepManifest,
    output_dir: str | Path,
    *,
    seed: int = 2026,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write label-free official and frozen-development video ID lists."""

    if manifest.protocol != "ucfrep_526":
        raise ValueError("standard split IDs require protocol='ucfrep_526'")
    manifest.validate_exact_official_splits()
    official_train = manifest.records_for("train")
    sealed_test = manifest.records_for("test")
    development_train, development = split_ucfrep_train_dev(
        official_train,
        seed=seed,
    )
    lists = {
        "ucfrep_526_train_421.txt": tuple(sorted(record.video_id for record in official_train)),
        "ucfrep_526_test_105.txt": tuple(sorted(record.video_id for record in sealed_test)),
        "ucfrep_526_train_337.txt": tuple(sorted(record.video_id for record in development_train)),
        "ucfrep_526_dev_84.txt": tuple(sorted(record.video_id for record in development)),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for filename, identifiers in lists.items():
        path = destination / filename
        content = "\n".join(identifiers) + "\n"
        encoded = content.encode("utf-8")
        if path.exists():
            if path.read_bytes() == encoded:
                paths[filename] = path
                continue
            if not overwrite:
                raise FileExistsError(f"split ID list exists with different content: {path}")
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        paths[filename] = path
    return paths
