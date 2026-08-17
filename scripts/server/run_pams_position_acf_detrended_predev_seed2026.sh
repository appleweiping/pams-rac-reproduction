#!/usr/bin/env bash
set -Eeuo pipefail

# Target-free pre-dev gate for the independently inferred mask-aware linear-
# detrended projected-position ACF. Exactly one network-isolated GPU container
# sees synthetic signals and the frozen train337 identity/pose view. No dev or
# test identity, pose, target, media, action, or count asset is mounted.

readonly ROOT_INPUT="${PAMS_ROOT:-/media/lenovo/data2/pams-rac}"
readonly SOURCE_CHECKOUT_INPUT="${PAMS_SOURCE_CHECKOUT:?PAMS_SOURCE_CHECKOUT is required}"
readonly SOURCE_REVISION="${PAMS_SOURCE_REVISION:?PAMS_SOURCE_REVISION is required}"
readonly ATTEMPT_ID="${PAMS_ATTEMPT_ID:?PAMS_ATTEMPT_ID is required}"
readonly GPU_DEVICE="${PAMS_GPU_DEVICE:-0}"
readonly IMAGE_ID="${PAMS_IMAGE_ID:-sha256:a6d10131c321c323c00bb85408cb8333503ecbc5d780d5059a73152d54df6005}"
readonly ENVIRONMENT_SHA256="${PAMS_ENVIRONMENT_SHA256:-1ccfd265d829e5160b8af39156193f63622ac73eaefec00b0f9dab40106c3508}"
readonly V8_RUN_ROOT_INPUT="${PAMS_V8_RUN_ROOT:-${ROOT_INPUT}/runs/pams-v8-seed2026/pams-v8-s2026-07192de-20260730t0251z}"

readonly EXPECTED_IMAGE_ID="sha256:a6d10131c321c323c00bb85408cb8333503ecbc5d780d5059a73152d54df6005"
readonly EXPECTED_ENVIRONMENT_SHA256="1ccfd265d829e5160b8af39156193f63622ac73eaefec00b0f9dab40106c3508"
readonly EXPERIMENT_CONFIG_RELATIVE="configs/experiments/pams_longest_contiguous_track_v8.yaml"
readonly READOUT_CONFIG_RELATIVE="configs/readouts/projected_position_acf_linear_detrend_predev_v1.yaml"
readonly EXPERIMENT_CONFIG_SHA256="eb4072a195608757cd2542855b33fb000c987f96c94b83a1e448ab78f97e9374"
readonly READOUT_CONFIG_SHA256="6226f2b4c3a8c61240e7a7b7f9226f21fdbd8cf8464780fada2235bd3700094f"
readonly READOUT_CONFIG_SEMANTIC_SHA256="7ebbfa0409e9b6908ccaac78099056b39882e5d4966f330f03fe0c419e3910b0"
readonly ENCODER_SHA256="6817fe468d76c3fa2182dd831695d6fe3d6da208a2b7f8dfa90cffa6872e8053"
readonly TRAIN_INPUT_SHA256="e238d307b4ed37b2f6c19eefdf7fa8812ec3d231cd2a7091be38e1c279f04207"
readonly TRAIN_POSE_SET_SHA256="f32d718ae922120778f535a6a4f467edba79cafeba90373b3a6a55bf11be6ee2"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

assert_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  test -f "$path" || fail "missing frozen input: $path"
  actual="$(sha256_of "$path")"
  [[ "$actual" == "$expected" ]] \
    || fail "SHA-256 mismatch for $path: expected=$expected actual=$actual"
}

[[ "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]] \
  || fail "PAMS_SOURCE_REVISION must be a full lowercase Git SHA"
[[ "$IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || fail "PAMS_IMAGE_ID must be an immutable image ID"
[[ "$ENVIRONMENT_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || fail "PAMS_ENVIRONMENT_SHA256 must be a lowercase SHA-256"
[[ "$IMAGE_ID" == "$EXPECTED_IMAGE_ID" ]] \
  || fail "predev gate must use the exact frozen-v8 image"
[[ "$ENVIRONMENT_SHA256" == "$EXPECTED_ENVIRONMENT_SHA256" ]] \
  || fail "predev gate environment differs from frozen v8"
[[ "$ATTEMPT_ID" =~ ^[a-z0-9][a-z0-9._-]{0,47}$ ]] \
  || fail "PAMS_ATTEMPT_ID must be a lowercase safe slug"
[[ "$ATTEMPT_ID" != *".."* ]] || fail "PAMS_ATTEMPT_ID must not contain '..'"
[[ "$GPU_DEVICE" =~ ^[0-9]+$ ]] \
  || fail "PAMS_GPU_DEVICE must be a non-negative integer"
for mount_path in \
  "$ROOT_INPUT" \
  "$SOURCE_CHECKOUT_INPUT" \
  "$V8_RUN_ROOT_INPUT"; do
  [[ "$mount_path" == /* ]] || fail "host input roots must be absolute"
  [[ "$mount_path" != *","* ]] || fail "Docker bind paths must not contain commas"
done

readonly ROOT="$(realpath -e -- "$ROOT_INPUT")"
readonly SOURCE_CHECKOUT="$(realpath -e -- "$SOURCE_CHECKOUT_INPUT")"
readonly V8_RUN_ROOT="$(realpath -e -- "$V8_RUN_ROOT_INPUT")"
readonly FIREWALL="${ROOT}/runs/firewall/ucfrep-337-84-105-375d8da"
readonly RUN_PARENT="${ROOT}/runs/pams-position-acf-detrended-predev-seed2026"
readonly RUN_ROOT="${RUN_PARENT}/${ATTEMPT_ID}"
readonly SOURCE_VIEW="${RUN_ROOT}/source"
readonly SOURCE_RECEIPT="${RUN_ROOT}/source-export.receipt.json"
readonly INPUT_ROOT="${RUN_ROOT}/inputs"
readonly TRAIN_POSE_VIEW="${INPUT_ROOT}/pose-train337"
readonly OUTPUT_ROOT="${RUN_ROOT}/output"
readonly AUDIT_ROOT="${RUN_ROOT}/audit"
readonly LOG_ROOT="${RUN_ROOT}/logs"
readonly GATE_SCRIPT="${AUDIT_ROOT}/detrended-position-predev.py"
readonly ARTIFACT="${OUTPUT_ROOT}/predev.json"
readonly RECEIPT="${OUTPUT_ROOT}/predev.receipt.json"
readonly HASH_MANIFEST="${AUDIT_ROOT}/artifact-sha256.txt"
readonly LOCK_ROOT="${ROOT}/.pams-gpu-locks"
readonly LOCK_PATH="${LOCK_ROOT}/gpu${GPU_DEVICE}.lock"
readonly CONTAINER_NAME="pacfdetrend-${ATTEMPT_ID}"

readonly ENCODER_CHECKPOINT="${V8_RUN_ROOT}/stages/encoder/run/encoder.pt"
readonly POSE_POOL="${V8_RUN_ROOT}/inputs/pose-train337-dev84"
readonly TRAIN_INPUT="${FIREWALL}/train.inputs.json"

RUN_RESERVED=0
ACTIVE_CONTAINER=""
CURRENT_STAGE="preflight"
CONTAINER_EXIT=255
SOURCE_RECEIPT_SHA256=""

write_status() {
  local state="$1"
  local stage="$2"
  local exit_code="$3"
  local temporary="${RUN_ROOT}/status.json.tmp"
  printf \
    '{"schema_version":1,"status":"%s","stage":"%s","exit_code":%s,"updated_utc":"%s"}\n' \
    "$state" \
    "$stage" \
    "$exit_code" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$temporary"
  mv -- "$temporary" "${RUN_ROOT}/status.json"
}

write_hash_manifest() {
  test -d "$AUDIT_ROOT" || return 0
  (
    cd "$RUN_ROOT"
    find . \
      -type f \
      ! -path './audit/artifact-sha256.txt' \
      ! -name '*.tmp' \
      -print0 \
      | sort -z \
      | xargs -0 -r sha256sum
  ) > "$HASH_MANIFEST"
  chmod 0444 "$HASH_MANIFEST"
}

on_exit() {
  local exit_code="$?"
  trap - EXIT HUP INT TERM
  set +e
  if [[ -n "$ACTIVE_CONTAINER" ]]; then
    docker inspect "$ACTIVE_CONTAINER" \
      > "${AUDIT_ROOT}/${CONTAINER_NAME}.error.inspect.json" 2>/dev/null
    if [[ "$(
      docker inspect "$ACTIVE_CONTAINER" \
        --format '{{.State.Running}}' 2>/dev/null
    )" == "true" ]]; then
      docker stop --time 10 "$ACTIVE_CONTAINER" >/dev/null 2>&1
    fi
    docker rm "$ACTIVE_CONTAINER" >/dev/null 2>&1
  fi
  if [[ "$RUN_RESERVED" -eq 1 ]]; then
    if [[ ! -s "$ARTIFACT" ]]; then
      FALLBACK_ARTIFACT="$ARTIFACT" \
      FALLBACK_STAGE="$CURRENT_STAGE" \
      FALLBACK_EXIT="$exit_code" \
      python3 - <<'HOSTPY'
import json
import os
from pathlib import Path

path = Path(os.environ["FALLBACK_ARTIFACT"])
path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "schema_version": 1,
    "artifact_type": "pams_position_acf_detrended_predev",
    "status": "host-error",
    "stage": os.environ["FALLBACK_STAGE"],
    "exit_code": int(os.environ["FALLBACK_EXIT"]),
    "eligible_for_paper_table": False,
    "dev84_scoring_authorized": False,
    "test105_evaluation_authorized": False,
    "error": "container did not retain a predev artifact",
}
path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
HOSTPY
    fi
    write_status "failed" "$CURRENT_STAGE" "$exit_code" >/dev/null 2>&1
    write_hash_manifest >/dev/null 2>&1
    chmod -R a-w "$RUN_ROOT" >/dev/null 2>&1
  fi
  exit "$exit_code"
}

trap on_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

[[ "$V8_RUN_ROOT" == "${ROOT}/runs/pams-v8-seed2026/"* ]] \
  || fail "PAMS_V8_RUN_ROOT must stay under the frozen v8 run parent"
[[ ! -L "$V8_RUN_ROOT_INPUT" ]] || fail "v8 run root must not be a symlink"
test -d "$FIREWALL" || fail "missing frozen UCFRep firewall"
test -d "$POSE_POOL" || fail "missing exact-v8 pose cache pool"
[[ "$(find "$POSE_POOL" -maxdepth 1 -type f -name '*.npz' | wc -l)" -eq 421 ]] \
  || fail "exact-v8 pose pool must contain 421 NPZ files"
[[ -z "$(find "$POSE_POOL" -mindepth 1 -type l -print -quit)" ]] \
  || fail "exact-v8 pose pool contains a symlink"

assert_sha256 "$ENCODER_CHECKPOINT" "$ENCODER_SHA256"
assert_sha256 "$TRAIN_INPUT" "$TRAIN_INPUT_SHA256"
[[ "$(git -C "$SOURCE_CHECKOUT" rev-parse HEAD)" == "$SOURCE_REVISION" ]] \
  || fail "source checkout revision mismatch"
[[ -z "$(
  git -C "$SOURCE_CHECKOUT" status --porcelain=v1 --untracked-files=all
)" ]] || fail "source checkout is dirty"
readonly EXPERIMENT_CONFIG_SOURCE="${SOURCE_CHECKOUT}/${EXPERIMENT_CONFIG_RELATIVE}"
readonly READOUT_CONFIG_SOURCE="${SOURCE_CHECKOUT}/${READOUT_CONFIG_RELATIVE}"
assert_sha256 "$EXPERIMENT_CONFIG_SOURCE" "$EXPERIMENT_CONFIG_SHA256"
assert_sha256 "$READOUT_CONFIG_SOURCE" "$READOUT_CONFIG_SHA256"
source "${SOURCE_CHECKOUT}/scripts/server/environment_fingerprint.sh"
[[ "$(
  pams_environment_fingerprint "${SOURCE_CHECKOUT}/docker/server"
)" == "$ENVIRONMENT_SHA256" ]] || fail "source environment fingerprint mismatch"
[[ "$(docker image inspect "$IMAGE_ID" --format '{{.Id}}')" == "$IMAGE_ID" ]] \
  || fail "container image mismatch"
[[ "$(
  docker image inspect "$IMAGE_ID" \
    --format \
      '{{index .Config.Labels "org.opencontainers.image.pams.environment-sha256"}}'
)" == "$ENVIRONMENT_SHA256" ]] || fail "container environment label mismatch"
if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  fail "container name already exists: $CONTAINER_NAME"
fi

mkdir -p -- "$RUN_PARENT"
[[ ! -L "$RUN_PARENT" ]] || fail "run parent must not be a symlink"
readonly RUN_PARENT_RESOLVED="$(realpath -e -- "$RUN_PARENT")"
[[ "$RUN_PARENT_RESOLVED" == "${ROOT}/runs/pams-position-acf-detrended-predev-seed2026" ]] \
  || fail "run parent resolves outside PAMS_ROOT"
mkdir -- "$RUN_ROOT" || fail "immutable run root exists: $RUN_ROOT"
RUN_RESERVED=1
mkdir -- \
  "$SOURCE_VIEW" \
  "$INPUT_ROOT" \
  "$OUTPUT_ROOT" \
  "$AUDIT_ROOT" \
  "$LOG_ROOT"
write_status "preparing" "source-export" "null"

git -C "$SOURCE_CHECKOUT" archive \
  --format=tar \
  "$SOURCE_REVISION" \
  src \
  pyproject.toml \
  "$EXPERIMENT_CONFIG_RELATIVE" \
  "$READOUT_CONFIG_RELATIVE" \
  | tar -xf - -C "$SOURCE_VIEW"
test ! -e "${SOURCE_VIEW}/.git" || fail "source export unexpectedly contains .git"
test ! -e "${SOURCE_VIEW}/data" || fail "source export unexpectedly contains data"
test ! -e "${SOURCE_VIEW}/tests" || fail "source export unexpectedly contains tests"
[[ -z "$(find "$SOURCE_VIEW" -type l -print -quit)" ]] \
  || fail "source export contains a symlink"
assert_sha256 \
  "${SOURCE_VIEW}/${EXPERIMENT_CONFIG_RELATIVE}" \
  "$EXPERIMENT_CONFIG_SHA256"
assert_sha256 \
  "${SOURCE_VIEW}/${READOUT_CONFIG_RELATIVE}" \
  "$READOUT_CONFIG_SHA256"

SOURCE_EXPORT_ROOT="$SOURCE_VIEW" \
SOURCE_EXPORT_REVISION="$SOURCE_REVISION" \
SOURCE_EXPORT_RECEIPT="$SOURCE_RECEIPT" \
python3 - <<'HOSTPY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["SOURCE_EXPORT_ROOT"]).resolve(strict=True)
output = Path(os.environ["SOURCE_EXPORT_RECEIPT"])
records = []
for path in sorted(
    root.rglob("*"),
    key=lambda item: item.relative_to(root).as_posix(),
):
    if path.is_symlink():
        raise RuntimeError("source export contains a symlink")
    if path.is_dir():
        continue
    if not path.is_file():
        raise RuntimeError("source export contains a non-file entry")
    data = path.read_bytes()
    records.append(
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    )
payload = {
    "schema_version": 1,
    "source_revision": os.environ["SOURCE_EXPORT_REVISION"],
    "root": ".",
    "files": records,
}
output.write_text(
    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
HOSTPY
SOURCE_RECEIPT_SHA256="$(sha256_of "$SOURCE_RECEIPT")"
readonly SOURCE_RECEIPT_SHA256
chmod -R a-w "$SOURCE_VIEW"
chmod 0444 "$SOURCE_RECEIPT"

CURRENT_STAGE="train337-pose-view"
mkdir -- "$TRAIN_POSE_VIEW"
: > "${AUDIT_ROOT}/pose-train337.sha256.tsv"
pose_count=0
while IFS= read -r video_id; do
  [[ -n "$video_id" ]] || fail "train337 sidecar contains an empty video_id"
  digest="$(printf '%s' "$video_id" | sha256sum | awk '{print $1}')"
  source_pose="${POSE_POOL}/${digest}.npz"
  target_pose="${TRAIN_POSE_VIEW}/${digest}.npz"
  test -f "$source_pose" || fail "missing train337 pose cache for $video_id"
  test ! -e "$target_pose" || fail "duplicate train337 pose cache for $video_id"
  source_pose_sha256="$(sha256_of "$source_pose")"
  cp --reflink=auto --preserve=mode,timestamps -- "$source_pose" "$target_pose"
  [[ ! "$source_pose" -ef "$target_pose" ]] \
    || fail "train337 pose view must not hard-link source files"
  [[ "$(sha256_of "$target_pose")" == "$source_pose_sha256" ]] \
    || fail "train337 pose copy hash mismatch for $video_id"
  printf '%s\t%s\t%s\n' \
    "$video_id" \
    "$digest" \
    "$source_pose_sha256" \
    >> "${AUDIT_ROOT}/pose-train337.sha256.tsv"
  ((pose_count += 1))
done < <(jq -r '.records[].video_id' "$TRAIN_INPUT")
[[ "$pose_count" -eq 337 ]] || fail "train337 pose view must contain 337 records"
[[ "$(find "$TRAIN_POSE_VIEW" -maxdepth 1 -type f -name '*.npz' | wc -l)" -eq 337 ]] \
  || fail "train337 pose view must contain exactly 337 NPZ files"
[[ "$(find "$TRAIN_POSE_VIEW" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 337 ]] \
  || fail "train337 pose view contains unexpected files"
[[ -z "$(find "$TRAIN_POSE_VIEW" -mindepth 1 -type l -print -quit)" ]] \
  || fail "train337 pose view contains a symlink"
chmod 0444 "$TRAIN_POSE_VIEW"/*.npz "${AUDIT_ROOT}/pose-train337.sha256.tsv"
chmod 0555 "$TRAIN_POSE_VIEW"

cat > "$GATE_SCRIPT" <<'CONTAINERPY'
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from pams.config import load_config
from pams.data import load_pose_cache_set, load_pose_input_manifest
from pams.metrics import round_count
from pams.period import (
    estimate_period_from_detrended_projected_position,
    projected_position_detrended_vector_acf_diagnostics,
)
from pams.reproducibility import (
    clean_git_revision,
    hardware_fingerprint,
    sha256_file,
    sha256_json,
)
from pams.training import (
    CheckpointProvenance,
    collate_pose_sequences,
    load_model_checkpoint,
)
from pams.types import PoseSequence


def write_exclusive(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_policy(path):
    raw_sha256 = sha256_file(path)
    if raw_sha256 != os.environ["PAMS_READOUT_CONFIG_SHA256"]:
        raise ValueError("readout config byte identity mismatch")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("readout config root must be a mapping")
    expected_fields = {
        "schema_version",
        "key",
        "classification",
        "eligible_for_paper_table",
        "dev84_scoring_authorized",
        "test105_evaluation_authorized",
        "frozen_inputs",
        "readout",
        "predev_gates",
        "stop_policy",
        "mount_policy",
    }
    if set(payload) != expected_fields:
        raise ValueError("readout config top-level fields differ from frozen schema")
    if payload["schema_version"] != 1:
        raise ValueError("readout schema version mismatch")
    if payload["key"] != "pams-projected-position-acf-linear-detrend-predev-v1":
        raise ValueError("readout key mismatch")
    if payload["eligible_for_paper_table"] is not False:
        raise ValueError("predev candidate may not be paper-table eligible")
    if payload["dev84_scoring_authorized"] is not False:
        raise ValueError("predev candidate may not authorize dev84 scoring")
    if payload["test105_evaluation_authorized"] is not False:
        raise ValueError("predev candidate may not authorize test105")
    if sha256_json(payload) != os.environ["PAMS_READOUT_CONFIG_SEMANTIC_SHA256"]:
        raise ValueError("readout semantic identity mismatch")
    readout = payload["readout"]
    expected_readout = {
        "temporal_transform": "mask_aware_per_feature_least_squares_linear_detrend",
        "temporal_coordinate": "original_frame_index",
        "invalid_frame_output": "exact_zero",
        "minimum_period_frames": 4,
        "maximum_period_frames": 128,
        "prediction_batch_size": 32,
    }
    for name, expected in expected_readout.items():
        if readout.get(name) != expected:
            raise ValueError("readout field drift: " + name)
    return payload


def complete_histogram(rows):
    histogram = {str(index): 0 for index in range(2, 65)}
    histogram["none"] = 0
    for row in rows:
        selected = row["selected_bin"]
        key = "none" if selected is None else str(int(selected))
        histogram[key] += 1
    return histogram


def infer(sequences, model, config, device):
    items = tuple(sequences)
    if not items:
        raise ValueError("predev inference requires at least one sequence")
    if len({item.video_id for item in items}) != len(items):
        raise ValueError("predev sequence identifiers must be unique")
    rows = []
    batch_sizes = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(items), 32):
            batch = collate_pose_sequences(items[start : start + 32]).to(device)
            batch_sizes.append(len(batch.video_ids))
            _, projected = model.encoder.forward_with_pre_pe(
                batch.poses,
                batch.valid_mask,
            )
            periods, confidences = (
                estimate_period_from_detrended_projected_position(
                    projected,
                    minimum=config.period.minimum,
                    maximum=config.period.maximum,
                    valid_mask=batch.valid_mask,
                )
            )
            diagnostics = (
                projected_position_detrended_vector_acf_diagnostics(
                    projected,
                    minimum=config.period.minimum,
                    maximum=config.period.maximum,
                    valid_mask=batch.valid_mask,
                )
            )
            if len(diagnostics) != len(batch.video_ids):
                raise RuntimeError("detrended diagnostic batch mismatch")
            valid_counts = batch.valid_mask.sum(dim=1)
            for index, video_id in enumerate(batch.video_ids):
                diagnostic = diagnostics[index]
                period = float(periods[index])
                confidence = float(confidences[index])
                valid_frames = int(valid_counts[index])
                if int(diagnostic.valid_length) != valid_frames:
                    raise RuntimeError("detrended diagnostic valid length mismatch")
                if not math.isclose(
                    period,
                    float(diagnostic.selected_period),
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ):
                    raise RuntimeError("detrended estimator/diagnostic period mismatch")
                if not math.isclose(
                    confidence,
                    float(diagnostic.confidence),
                    rel_tol=1e-6,
                    abs_tol=1e-7,
                ):
                    raise RuntimeError(
                        "detrended estimator/diagnostic confidence mismatch"
                    )
                if tuple(diagnostic.allowed_bins) != tuple(range(2, 65)):
                    raise RuntimeError("detrended diagnostic allowed bins drifted")
                if len(diagnostic.power_shares) != 63:
                    raise RuntimeError("detrended diagnostic spectrum length drifted")
                selected_bin = (
                    None
                    if diagnostic.selected_bin is None
                    else int(diagnostic.selected_bin)
                )
                raw_count = (
                    float(valid_frames - 1) / period
                    if confidence > 0.0 and valid_frames >= 2
                    else 0.0
                )
                rows.append(
                    {
                        "video_id": video_id,
                        "valid_frames": valid_frames,
                        "period_frames": period,
                        "confidence": confidence,
                        "raw_count": raw_count,
                        "rounded_count": round_count(raw_count),
                        "selected_bin": selected_bin,
                    }
                )
    return rows, batch_sizes


def synthetic_gate(policy, model, config, device):
    gate = policy["predev_gates"]["synthetic"]
    harmonic_orders = tuple(int(value) for value in gate["harmonic_orders"])
    if harmonic_orders != tuple(range(2, 8)):
        raise ValueError("synthetic harmonic orders must be exactly k=2..7")
    frames = int(gate["frames"])
    period = float(gate["fundamental_period_frames"])
    drift = float(gate["linear_drift_per_frame"])
    if frames != 256 or period != 64.0 or drift != 0.02:
        raise ValueError("synthetic detrended gate constants drifted")
    time = np.arange(frames, dtype=np.float64)
    fundamental_phase = 2.0 * np.pi * time / period
    sequences = []
    for harmonic in harmonic_orders:
        waveform = (
            float(gate["fundamental_amplitude"]) * np.sin(fundamental_phase)
            + float(gate["harmonic_amplitude"])
            * np.sin(float(harmonic) * fundamental_phase + 0.37)
            + drift * time
        )
        xyz = np.zeros((frames, 33, 3), dtype=np.float32)
        xyz[:, 0, 0] = waveform.astype(np.float32)
        sequences.append(
            PoseSequence(
                video_id="harmonic-order-{}".format(harmonic),
                fps=30.0,
                xyz=xyz,
                valid_mask=np.ones(frames, dtype=np.bool_),
            )
        )
    rows, batch_sizes = infer(tuple(sequences), model, config, device)
    for row, harmonic in zip(rows, harmonic_orders):
        row["harmonic_order"] = harmonic
        row["linear_drift_per_frame"] = drift
        row["expected_fundamental_period_frames"] = period
    finite = all(
        math.isfinite(float(row[name]))
        for row in rows
        for name in ("period_frames", "confidence", "raw_count")
    )
    positive = all(float(row["confidence"]) > 0.0 for row in rows)
    fraction = sum(
        abs(float(row["period_frames"]) - period) <= 1e-6 for row in rows
    ) / max(1, len(rows))
    conditions = {
        "record_total_exact": len(rows) == int(gate["required_records"]),
        "fixed_batch_32": batch_sizes == [len(rows)],
        "all_outputs_finite": finite if gate["require_finite_outputs"] else True,
        "all_confidences_positive": (
            positive if gate["require_positive_confidence"] else True
        ),
        "fundamental_selection_fraction_gte_frozen_minimum": (
            fraction >= float(gate["minimum_fundamental_selection_fraction"])
        ),
    }
    return {
        "status": "passed" if all(conditions.values()) else "failed",
        "conditions": conditions,
        "summary": {
            "record_total": len(rows),
            "batch_sizes": batch_sizes,
            "harmonic_orders": list(harmonic_orders),
            "fundamental_period_frames": period,
            "linear_drift_per_frame": drift,
            "fundamental_selection_fraction": fraction,
            "selected_bin_histogram": complete_histogram(rows),
        },
        "rows": rows,
    }


def train337_gate(policy, model, config, device, manifest_path, pose_cache_dir):
    gate = policy["predev_gates"]["train337_target_free_distribution"]
    manifest = load_pose_input_manifest(manifest_path)
    if manifest.protocol != "ucfrep_526":
        raise ValueError("train337 protocol mismatch")
    if manifest.split != "train" or len(manifest.records) != 337:
        raise ValueError("train337 identity sidecar mismatch")
    manifest.validate_exact_membership()
    sequences, snapshot = load_pose_cache_set(
        manifest.records,
        cache_dir=pose_cache_dir,
        pose_fingerprint=config.pose_fingerprint,
    )
    expected_snapshot = policy["frozen_inputs"]["train337_pose_cache_set_sha256"]
    if snapshot.fingerprint != expected_snapshot:
        raise ValueError("train337 pose-cache-set fingerprint mismatch")
    rows, batch_sizes = infer(sequences, model, config, device)
    finite = all(
        math.isfinite(float(row[name]))
        for row in rows
        for name in ("period_frames", "confidence", "raw_count")
    )
    positive = [
        row
        for row in rows
        if row["confidence"] > 0.0
        and row["valid_frames"] >= 2
        and row["selected_bin"] is not None
    ]
    denominator = max(1, len(positive))
    zero_fraction = (len(rows) - len(positive)) / max(1, len(rows))
    minimum_fraction = sum(
        row["period_frames"] <= config.period.minimum + 1e-6
        for row in positive
    ) / denominator
    maximum_fraction = sum(
        row["period_frames"] >= config.period.maximum - 1e-6
        for row in positive
    ) / denominator
    bin_counts = Counter(int(row["selected_bin"]) for row in positive)
    mode_fraction = max(bin_counts.values(), default=0) / denominator
    zero_guard = all(
        row["rounded_count"] == 0
        for row in rows
        if row["confidence"] <= 0.0 or row["valid_frames"] < 2
    )
    conditions = {
        "record_total_exact": len(rows) == int(gate["required_records"]),
        "fixed_batch_32_with_tail": batch_sizes == ([32] * 10 + [17]),
        "all_outputs_finite": finite if gate["require_finite_outputs"] else True,
        "zero_confidence_fraction_lte_frozen_maximum": (
            zero_fraction <= float(gate["maximum_zero_confidence_fraction"])
        ),
        "minimum_period_fraction_lte_frozen_maximum": (
            minimum_fraction <= float(gate["maximum_minimum_period_fraction"])
        ),
        "maximum_period_fraction_lte_frozen_maximum": (
            maximum_fraction <= float(gate["maximum_maximum_period_fraction"])
        ),
        "selected_bin_mode_fraction_lte_frozen_maximum": (
            mode_fraction <= float(gate["maximum_selected_bin_mode_fraction"])
        ),
        "unique_selected_bins_gte_frozen_minimum": (
            len(bin_counts) >= int(gate["minimum_unique_selected_bins"])
        ),
        "zero_confidence_zero_count": (
            zero_guard if gate["require_zero_confidence_zero_count"] else True
        ),
    }
    histogram = complete_histogram(rows)
    return {
        "status": "passed" if all(conditions.values()) else "failed",
        "conditions": conditions,
        "summary": {
            "record_total": len(rows),
            "batch_sizes": batch_sizes,
            "positive_confidence_records": len(positive),
            "zero_confidence_records": len(rows) - len(positive),
            "zero_confidence_fraction": zero_fraction,
            "minimum_period_fraction": minimum_fraction,
            "maximum_period_fraction": maximum_fraction,
            "selected_bin_mode_fraction": mode_fraction,
            "unique_selected_bins": len(bin_counts),
            "selected_bin_histogram": histogram,
            "pose_cache_set_sha256": snapshot.fingerprint,
            "train_inputs_fingerprint": manifest.fingerprint,
        },
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--readout-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-inputs", type=Path, required=True)
    parser.add_argument("--pose-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_revision = clean_git_revision("/workspace")
    policy = load_policy(args.readout_config)
    config = load_config(args.experiment_config)
    frozen = policy["frozen_inputs"]
    if sha256_file(args.experiment_config) != frozen["experiment_config_sha256"]:
        raise ValueError("experiment config bytes mismatch")
    if config.fingerprint != frozen["experiment_config_fingerprint"]:
        raise ValueError("experiment config semantic mismatch")
    if config.pose_fingerprint != frozen["pose_fingerprint"]:
        raise ValueError("pose fingerprint mismatch")
    if config.protocol != "ucfrep_526" or config.seed != 2026:
        raise ValueError("experiment protocol or seed mismatch")
    if sha256_file(args.checkpoint) != frozen["encoder_checkpoint_sha256"]:
        raise ValueError("encoder checkpoint bytes mismatch")
    if sha256_file(args.train_inputs) != frozen["train337_inputs_sha256"]:
        raise ValueError("train337 sidecar bytes mismatch")
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    provenance = CheckpointProvenance.from_mapping(raw["provenance"])
    if provenance.source_git_sha != frozen["encoder_source_git_sha"]:
        raise ValueError("encoder source revision mismatch")
    if provenance.container_image_id != frozen["encoder_container_image_id"]:
        raise ValueError("encoder image identity mismatch")
    if (
        provenance.container_environment_sha256
        != frozen["encoder_container_environment_sha256"]
    ):
        raise ValueError("encoder environment identity mismatch")
    if provenance.pose_fingerprint != config.pose_fingerprint:
        raise ValueError("encoder pose fingerprint mismatch")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("formal predev gate requires one CUDA GPU")
    model = load_model_checkpoint(
        args.checkpoint,
        config,
        device=device,
        expected_stage="encoder",
        expected_provenance=provenance,
    ).eval()
    synthetic = synthetic_gate(policy, model, config, device)
    if synthetic["status"] == "passed":
        train337 = train337_gate(
            policy,
            model,
            config,
            device,
            args.train_inputs,
            args.pose_cache,
        )
    else:
        train337 = {
            "status": "skipped",
            "reason": policy["stop_policy"]["synthetic_failure"],
            "conditions": {},
            "summary": {
                "record_total": 0,
                "selected_bin_histogram": complete_histogram(()),
            },
            "rows": [],
        }
    passed = synthetic["status"] == "passed" and train337["status"] == "passed"
    payload = {
        "schema_version": 1,
        "artifact_type": "pams_position_acf_detrended_predev",
        "status": "passed" if passed else "failed",
        "classification": policy["classification"],
        "method_key": policy["key"],
        "eligible_for_paper_table": False,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
        "protocol": "ucfrep_526",
        "seed": 2026,
        "algorithm": (
            "encoder projection before PE -> valid-only per-feature affine "
            "detrend -> masked vector ACF -> Hann allowed-band spectrum"
        ),
        "source_revision": source_revision,
        "readout_config_sha256": sha256_file(args.readout_config),
        "readout_config_semantic_sha256": sha256_json(policy),
        "experiment_config_sha256": sha256_file(args.experiment_config),
        "encoder_checkpoint_sha256": sha256_file(args.checkpoint),
        "train337_inputs_sha256": sha256_file(args.train_inputs),
        "checkpoint_provenance": provenance.to_dict(),
        "runtime_provenance": {
            "container_image_id": os.environ["PAMS_CONTAINER_IMAGE_ID"],
            "container_environment_sha256": os.environ[
                "PAMS_CONTAINER_ENVIRONMENT_SHA256"
            ],
            "container_source_revision": os.environ[
                "PAMS_CONTAINER_SOURCE_REVISION"
            ],
        },
        "hardware": hardware_fingerprint(),
        "mount_audit": {
            "network": "none",
            "train337_identity_mounted": True,
            "train337_pose_mounted": True,
            "dev84_identity_mounted": False,
            "dev84_pose_mounted": False,
            "dev84_targets_mounted": False,
            "test105_identity_mounted": False,
            "test105_media_mounted": False,
            "test105_pose_mounted": False,
            "test105_labels_mounted": False,
        },
        "gates": {
            "synthetic_k2_to_k7_with_linear_drift": synthetic,
            "train337_target_free_distribution": train337,
        },
        "stop_decision": (
            policy["stop_policy"]["pass"]
            if passed
            else (
                policy["stop_policy"]["synthetic_failure"]
                if synthetic["status"] != "passed"
                else policy["stop_policy"]["train337_failure"]
            )
        ),
    }
    write_exclusive(args.output, payload)
    return 0 if passed else 3


if __name__ == "__main__":
    output_argument = None
    try:
        for index, value in enumerate(os.sys.argv[:-1]):
            if value == "--output":
                output_argument = Path(os.sys.argv[index + 1])
                break
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        if output_argument is not None and not output_argument.exists():
            error_payload = {
                "schema_version": 1,
                "artifact_type": "pams_position_acf_detrended_predev",
                "status": "error",
                "eligible_for_paper_table": False,
                "dev84_scoring_authorized": False,
                "test105_evaluation_authorized": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "runtime_provenance": {
                    "container_image_id": os.environ.get(
                        "PAMS_CONTAINER_IMAGE_ID"
                    ),
                    "container_environment_sha256": os.environ.get(
                        "PAMS_CONTAINER_ENVIRONMENT_SHA256"
                    ),
                    "container_source_revision": os.environ.get(
                        "PAMS_CONTAINER_SOURCE_REVISION"
                    ),
                },
            }
            try:
                error_payload["hardware"] = hardware_fingerprint()
            except Exception as hardware_error:
                error_payload["hardware"] = {
                    "status": "unavailable",
                    "error": str(hardware_error),
                }
            write_exclusive(output_argument, error_payload)
        raise
CONTAINERPY
chmod 0444 "$GATE_SCRIPT"

cat > "${RUN_ROOT}/attempt.reservation.json" <<EOF
{
  "schema_version": 1,
  "attempt_id": "${ATTEMPT_ID}",
  "reserved_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "scope": "synthetic-k2-to-k7-plus-train337-target-free-only",
  "seed": 2026,
  "source_revision": "${SOURCE_REVISION}",
  "source_export_receipt_sha256": "${SOURCE_RECEIPT_SHA256}",
  "container_image_id": "${IMAGE_ID}",
  "container_environment_sha256": "${ENVIRONMENT_SHA256}",
  "readout_config_sha256": "${READOUT_CONFIG_SHA256}",
  "readout_config_semantic_sha256": "${READOUT_CONFIG_SEMANTIC_SHA256}",
  "experiment_config_sha256": "${EXPERIMENT_CONFIG_SHA256}",
  "encoder_checkpoint_sha256": "${ENCODER_SHA256}",
  "train337_inputs_sha256": "${TRAIN_INPUT_SHA256}",
  "train337_pose_cache_set_sha256": "${TRAIN_POSE_SET_SHA256}",
  "container_count": 1,
  "network": "none",
  "dev84_scoring_authorized": false,
  "test105_evaluation_authorized": false
}
EOF
chmod 0444 "${RUN_ROOT}/attempt.reservation.json"

mkdir -p -- "$LOCK_ROOT"
[[ ! -L "$LOCK_ROOT" ]] || fail "GPU lock root must not be a symlink"
readonly LOCK_ROOT_RESOLVED="$(realpath -e -- "$LOCK_ROOT")"
[[ "$LOCK_ROOT_RESOLVED" == "${ROOT}/.pams-gpu-locks" ]] \
  || fail "GPU lock root resolves outside PAMS_ROOT"
[[ ! -L "$LOCK_PATH" ]] || fail "GPU lock file must not be a symlink"
exec 9>"$LOCK_PATH"
chmod 0600 "$LOCK_PATH"
if ! flock -n 9; then
  CURRENT_STAGE="gpu-lock"
  exit 75
fi

CURRENT_STAGE="container-create"
docker create \
  --name "$CONTAINER_NAME" \
  --init \
  --network none \
  --read-only \
  --shm-size 8g \
  --pids-limit 4096 \
  --cpus 24 \
  --memory 96g \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --user 1000:1000 \
  --workdir /workspace \
  --gpus "device=${GPU_DEVICE}" \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  --env PYTHONPATH=/workspace/src \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env HOME=/pams/home \
  --env PAMS_AUDIT_MODE=formal \
  --env "PAMS_CONTAINER_IMAGE_ID=${IMAGE_ID}" \
  --env "PAMS_CONTAINER_ENVIRONMENT_SHA256=${ENVIRONMENT_SHA256}" \
  --env "PAMS_CONTAINER_SOURCE_REVISION=${SOURCE_REVISION}" \
  --env PAMS_SOURCE_EXPORT_RECEIPT=/pams/source-export-receipt.json \
  --env "PAMS_SOURCE_EXPORT_RECEIPT_SHA256=${SOURCE_RECEIPT_SHA256}" \
  --env "PAMS_READOUT_CONFIG_SHA256=${READOUT_CONFIG_SHA256}" \
  --env "PAMS_READOUT_CONFIG_SEMANTIC_SHA256=${READOUT_CONFIG_SEMANTIC_SHA256}" \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,uid=1000,gid=1000,size=2g \
  --tmpfs /pams/home:rw,noexec,nosuid,nodev,uid=1000,gid=1000,size=1g \
  --mount "type=bind,src=${SOURCE_VIEW},dst=/workspace,readonly" \
  --mount \
    "type=bind,src=${SOURCE_RECEIPT},dst=/pams/source-export-receipt.json,readonly" \
  --mount \
    "type=bind,src=${SOURCE_VIEW}/${EXPERIMENT_CONFIG_RELATIVE},dst=/pams/input/experiment-config.yaml,readonly" \
  --mount \
    "type=bind,src=${SOURCE_VIEW}/${READOUT_CONFIG_RELATIVE},dst=/pams/input/readout-config.yaml,readonly" \
  --mount \
    "type=bind,src=${ENCODER_CHECKPOINT},dst=/pams/encoder/encoder.pt,readonly" \
  --mount \
    "type=bind,src=${TRAIN_INPUT},dst=/pams/protocol/train.inputs.json,readonly" \
  --mount \
    "type=bind,src=${TRAIN_POSE_VIEW},dst=/pams/pose-cache,readonly" \
  --mount "type=bind,src=${GATE_SCRIPT},dst=/pams/predev.py,readonly" \
  --mount "type=bind,src=${OUTPUT_ROOT},dst=/pams/output" \
  "$IMAGE_ID" \
  python /pams/predev.py \
    --experiment-config /pams/input/experiment-config.yaml \
    --readout-config /pams/input/readout-config.yaml \
    --checkpoint /pams/encoder/encoder.pt \
    --train-inputs /pams/protocol/train.inputs.json \
    --pose-cache /pams/pose-cache \
    --output /pams/output/predev.json \
  > "${AUDIT_ROOT}/${CONTAINER_NAME}.create-id.txt"
ACTIVE_CONTAINER="$CONTAINER_NAME"
docker inspect "$CONTAINER_NAME" \
  > "${AUDIT_ROOT}/${CONTAINER_NAME}.pre-run.inspect.json"
VERIFY_INSPECT="${AUDIT_ROOT}/${CONTAINER_NAME}.pre-run.inspect.json" \
VERIFY_SOURCE_VIEW="$SOURCE_VIEW" \
VERIFY_SOURCE_RECEIPT="$SOURCE_RECEIPT" \
VERIFY_EXPERIMENT_CONFIG="${SOURCE_VIEW}/${EXPERIMENT_CONFIG_RELATIVE}" \
VERIFY_READOUT_CONFIG="${SOURCE_VIEW}/${READOUT_CONFIG_RELATIVE}" \
VERIFY_CHECKPOINT="$ENCODER_CHECKPOINT" \
VERIFY_TRAIN_INPUT="$TRAIN_INPUT" \
VERIFY_TRAIN_POSE="$TRAIN_POSE_VIEW" \
VERIFY_GATE_SCRIPT="$GATE_SCRIPT" \
VERIFY_OUTPUT="$OUTPUT_ROOT" \
VERIFY_IMAGE="$IMAGE_ID" \
VERIFY_GPU="$GPU_DEVICE" \
python3 - <<'HOSTPY'
import json
import os
from pathlib import Path

item = json.loads(
    Path(os.environ["VERIFY_INSPECT"]).read_text(encoding="utf-8")
)[0]
config = item["Config"]
host = item["HostConfig"]
mounts = {
    mount["Destination"]: (mount["Source"], bool(mount["RW"]))
    for mount in item["Mounts"]
}
expected = {
    "/workspace": (os.environ["VERIFY_SOURCE_VIEW"], False),
    "/pams/source-export-receipt.json": (
        os.environ["VERIFY_SOURCE_RECEIPT"],
        False,
    ),
    "/pams/input/experiment-config.yaml": (
        os.environ["VERIFY_EXPERIMENT_CONFIG"],
        False,
    ),
    "/pams/input/readout-config.yaml": (
        os.environ["VERIFY_READOUT_CONFIG"],
        False,
    ),
    "/pams/encoder/encoder.pt": (
        os.environ["VERIFY_CHECKPOINT"],
        False,
    ),
    "/pams/protocol/train.inputs.json": (
        os.environ["VERIFY_TRAIN_INPUT"],
        False,
    ),
    "/pams/pose-cache": (os.environ["VERIFY_TRAIN_POSE"], False),
    "/pams/predev.py": (os.environ["VERIFY_GATE_SCRIPT"], False),
    "/pams/output": (os.environ["VERIFY_OUTPUT"], True),
}
assert config["Image"] == os.environ["VERIFY_IMAGE"]
assert config["User"] == "1000:1000"
assert host["NetworkMode"] == "none"
assert host["ReadonlyRootfs"] is True
assert "ALL" in (host.get("CapDrop") or [])
assert "no-new-privileges:true" in (host.get("SecurityOpt") or [])
assert mounts == expected, (mounts, expected)
requests = host.get("DeviceRequests") or []
assert len(requests) == 1
assert requests[0].get("DeviceIDs") == [os.environ["VERIFY_GPU"]]
all_text = "\0".join(
    list(mounts)
    + [source for source, _ in mounts.values()]
    + list(config.get("Cmd") or [])
)
for forbidden in (
    "dev.inputs",
    "dev.targets",
    "test-identity",
    "test.targets",
    "pose-dev",
    "pose-test",
):
    assert forbidden not in all_text
HOSTPY

CURRENT_STAGE="container-run"
write_status "running" "predev-gates" "null"
set +e
docker start --attach "$CONTAINER_NAME" \
  > "${LOG_ROOT}/${CONTAINER_NAME}.stdout.log" \
  2> "${LOG_ROOT}/${CONTAINER_NAME}.stderr.log"
attach_exit="$?"
set -e
still_running="$(
  docker inspect "$CONTAINER_NAME" --format '{{.State.Running}}'
)"
if [[ "$still_running" == "true" ]]; then
  docker stop --time 10 "$CONTAINER_NAME" >/dev/null
fi
CONTAINER_EXIT="$(
  docker inspect "$CONTAINER_NAME" --format '{{.State.ExitCode}}'
)"
docker inspect "$CONTAINER_NAME" \
  > "${AUDIT_ROOT}/${CONTAINER_NAME}.post-run.inspect.json"
printf '%s\n' "$CONTAINER_EXIT" \
  > "${AUDIT_ROOT}/${CONTAINER_NAME}.exit-code.txt"
docker rm "$CONTAINER_NAME" >/dev/null
ACTIVE_CONTAINER=""
if [[ "$still_running" == "true" || "$attach_exit" -ne "$CONTAINER_EXIT" ]]; then
  CONTAINER_EXIT=125
fi
flock -u 9

CURRENT_STAGE="artifact-freeze"
if [[ ! -s "$ARTIFACT" ]]; then
  FALLBACK_ARTIFACT="$ARTIFACT" \
  FALLBACK_EXIT="$CONTAINER_EXIT" \
  python3 - <<'HOSTPY'
import json
import os
from pathlib import Path

path = Path(os.environ["FALLBACK_ARTIFACT"])
payload = {
    "schema_version": 1,
    "artifact_type": "pams_position_acf_detrended_predev",
    "status": "container-error",
    "container_exit_code": int(os.environ["FALLBACK_EXIT"]),
    "eligible_for_paper_table": False,
    "dev84_scoring_authorized": False,
    "test105_evaluation_authorized": False,
    "error": "container exited without retaining predev.json",
}
path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
HOSTPY
fi

ARTIFACT_PATH="$ARTIFACT" \
RECEIPT_PATH="$RECEIPT" \
SOURCE_RECEIPT_PATH="$SOURCE_RECEIPT" \
READOUT_PATH="${SOURCE_VIEW}/${READOUT_CONFIG_RELATIVE}" \
CONFIG_PATH="${SOURCE_VIEW}/${EXPERIMENT_CONFIG_RELATIVE}" \
CHECKPOINT_PATH="$ENCODER_CHECKPOINT" \
TRAIN_INPUT_PATH="$TRAIN_INPUT" \
POSE_VIEW_RECEIPT="${AUDIT_ROOT}/pose-train337.sha256.tsv" \
PRE_INSPECT="${AUDIT_ROOT}/${CONTAINER_NAME}.pre-run.inspect.json" \
POST_INSPECT="${AUDIT_ROOT}/${CONTAINER_NAME}.post-run.inspect.json" \
CONTAINER_EXIT_VALUE="$CONTAINER_EXIT" \
SOURCE_REVISION_VALUE="$SOURCE_REVISION" \
python3 - <<'HOSTPY'
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


artifact = Path(os.environ["ARTIFACT_PATH"])
artifact_bytes = artifact.read_bytes()
artifact_payload = json.loads(artifact_bytes)
receipt = {
    "schema_version": 1,
    "artifact_type": "pams_position_acf_detrended_predev_receipt",
    "status": artifact_payload.get("status", "unknown"),
    "predev_file": artifact.name,
    "predev_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
    "predev_bytes": len(artifact_bytes),
    "container_exit_code": int(os.environ["CONTAINER_EXIT_VALUE"]),
    "source_revision": os.environ["SOURCE_REVISION_VALUE"],
    "source_export_receipt_sha256": digest(
        os.environ["SOURCE_RECEIPT_PATH"]
    ),
    "readout_config_sha256": digest(os.environ["READOUT_PATH"]),
    "experiment_config_sha256": digest(os.environ["CONFIG_PATH"]),
    "encoder_checkpoint_sha256": digest(os.environ["CHECKPOINT_PATH"]),
    "train337_inputs_sha256": digest(os.environ["TRAIN_INPUT_PATH"]),
    "train337_pose_view_receipt_sha256": digest(
        os.environ["POSE_VIEW_RECEIPT"]
    ),
    "container_pre_run_inspect_sha256": digest(os.environ["PRE_INSPECT"]),
    "container_post_run_inspect_sha256": digest(os.environ["POST_INSPECT"]),
    "network": "none",
    "container_count": 1,
    "dev84_scoring_authorized": False,
    "test105_evaluation_authorized": False,
}
Path(os.environ["RECEIPT_PATH"]).write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
HOSTPY

chmod 0444 "$ARTIFACT" "$RECEIPT"
if [[ "$CONTAINER_EXIT" -eq 0 ]]; then
  write_status "completed" "predev-gates" "0"
  write_hash_manifest
  chmod -R a-w "$RUN_ROOT"
  trap - EXIT HUP INT TERM
  printf '%s\n' "$RUN_ROOT"
  exit 0
fi

CURRENT_STAGE="predev-gates-failed"
exit "$CONTAINER_EXIT"
