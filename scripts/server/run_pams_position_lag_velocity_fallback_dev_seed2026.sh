#!/usr/bin/env bash
set -Eeuo pipefail

# Strict dev84 prediction and independently isolated scoring for the inferred
# v2 projected-position lag-ACF with velocity fallback.  This runner is inert
# unless an immutable formal predev artifact has passed and is bound by hash.
# No test105 asset is named or mounted anywhere in the execution graph.

readonly ROOT_INPUT="${PAMS_ROOT:-/media/lenovo/data2/pams-rac}"
readonly SOURCE_CHECKOUT_INPUT="${PAMS_SOURCE_CHECKOUT:?PAMS_SOURCE_CHECKOUT is required}"
readonly SOURCE_REVISION="${PAMS_SOURCE_REVISION:?PAMS_SOURCE_REVISION is required}"
readonly ATTEMPT_ID="${PAMS_ATTEMPT_ID:?PAMS_ATTEMPT_ID is required}"
readonly READOUT_CONFIG_SHA256="${PAMS_READOUT_CONFIG_SHA256:?PAMS_READOUT_CONFIG_SHA256 is required}"
readonly READOUT_CONFIG_SEMANTIC_SHA256="${PAMS_READOUT_CONFIG_SEMANTIC_SHA256:?PAMS_READOUT_CONFIG_SEMANTIC_SHA256 is required}"
readonly PREDEV_ARTIFACT_INPUT="${PAMS_PREDEV_ARTIFACT:?PAMS_PREDEV_ARTIFACT is required}"
readonly PREDEV_ARTIFACT_SHA256="${PAMS_PREDEV_ARTIFACT_SHA256:?PAMS_PREDEV_ARTIFACT_SHA256 is required}"
readonly GPU_DEVICE="${PAMS_GPU_DEVICE:-0}"
readonly IMAGE_ID="${PAMS_IMAGE_ID:-sha256:a6d10131c321c323c00bb85408cb8333503ecbc5d780d5059a73152d54df6005}"
readonly ENVIRONMENT_SHA256="${PAMS_ENVIRONMENT_SHA256:-1ccfd265d829e5160b8af39156193f63622ac73eaefec00b0f9dab40106c3508}"
readonly V8_RUN_ROOT_INPUT="${PAMS_V8_RUN_ROOT:-${ROOT_INPUT}/runs/pams-v8-seed2026/pams-v8-s2026-07192de-20260730t0251z}"

readonly EXPECTED_IMAGE_ID="sha256:a6d10131c321c323c00bb85408cb8333503ecbc5d780d5059a73152d54df6005"
readonly EXPECTED_ENVIRONMENT_SHA256="1ccfd265d829e5160b8af39156193f63622ac73eaefec00b0f9dab40106c3508"
readonly EXPECTED_SOURCE_REVISION="470d49bd97eef5adad89142d4ffc017125505525"
readonly EXPECTED_READOUT_CONFIG_SHA256="b3c38341e442765cb709d8ca9c5ecfe541121ea61d1ae35c0ff2cb48b6b09a98"
readonly EXPECTED_READOUT_CONFIG_SEMANTIC_SHA256="e17b4105f732be1bc4a40b153c0db28729fc0379a5f921ab0a26d3b2301b4233"
readonly EXPECTED_PREDEV_ARTIFACT_SHA256="f563cffa4235b34e091bf9b93af9123a93fa1d57ee29dc39f54f4feb60e0d928"
readonly EXPERIMENT_CONFIG_RELATIVE="configs/experiments/pams_longest_contiguous_track_v8.yaml"
readonly READOUT_CONFIG_RELATIVE="configs/readouts/projected_position_lag_velocity_fallback_predev_v2.yaml"
readonly EXPERIMENT_CONFIG_SHA256="eb4072a195608757cd2542855b33fb000c987f96c94b83a1e448ab78f97e9374"
readonly ENCODER_SHA256="6817fe468d76c3fa2182dd831695d6fe3d6da208a2b7f8dfa90cffa6872e8053"
readonly DEV_INPUT_SHA256="f21c49569805e4aea326977f6d7ce2964f503fad129a17e95214594364ca4ae7"
readonly DEV_POSE_SET_SHA256="681b0390ff252df9cf5b1e7c41bfcb6ecb8a0c5b4ef341087cacddc1bf5a3de1"
readonly DEV_TARGET_SHA256="1ab3bc010ccb5d869af7662586594c973e53841838643e1e9b8cc1787b0efab6"

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
  test -f "$path" || fail "missing immutable input: $path"
  actual="$(sha256_of "$path")"
  [[ "$actual" == "$expected" ]] \
    || fail "SHA-256 mismatch for $path: expected=$expected actual=$actual"
}

for value_name in \
  SOURCE_REVISION \
  READOUT_CONFIG_SHA256 \
  READOUT_CONFIG_SEMANTIC_SHA256 \
  PREDEV_ARTIFACT_SHA256 \
  ENVIRONMENT_SHA256; do
  value="${!value_name}"
  if [[ "$value_name" == "SOURCE_REVISION" ]]; then
    [[ "$value" =~ ^[0-9a-f]{40}$ ]] \
      || fail "PAMS_SOURCE_REVISION must be a full lowercase Git SHA"
  else
    [[ "$value" =~ ^[0-9a-f]{64}$ ]] \
      || fail "PAMS_${value_name} must be a lowercase SHA-256"
  fi
done
[[ "$IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || fail "PAMS_IMAGE_ID must be an immutable image ID"
[[ "$IMAGE_ID" == "$EXPECTED_IMAGE_ID" ]] \
  || fail "dev84 runner must use the frozen-v8 image"
[[ "$ENVIRONMENT_SHA256" == "$EXPECTED_ENVIRONMENT_SHA256" ]] \
  || fail "dev84 runner environment differs from frozen v8"
[[ "$SOURCE_REVISION" == "$EXPECTED_SOURCE_REVISION" ]] \
  || fail "dev84 runner source differs from the passed formal predev"
[[ "$READOUT_CONFIG_SHA256" == "$EXPECTED_READOUT_CONFIG_SHA256" ]] \
  || fail "dev84 runner readout bytes differ from the passed formal predev"
[[ "$READOUT_CONFIG_SEMANTIC_SHA256" == \
  "$EXPECTED_READOUT_CONFIG_SEMANTIC_SHA256" ]] \
  || fail "dev84 runner readout semantics differ from the passed formal predev"
[[ "$PREDEV_ARTIFACT_SHA256" == "$EXPECTED_PREDEV_ARTIFACT_SHA256" ]] \
  || fail "dev84 runner requires the passed formal predev artifact"
[[ "$ATTEMPT_ID" =~ ^[a-z0-9][a-z0-9._-]{0,47}$ ]] \
  || fail "PAMS_ATTEMPT_ID must be a lowercase safe slug"
[[ "$ATTEMPT_ID" != *".."* ]] || fail "PAMS_ATTEMPT_ID must not contain '..'"
[[ "$GPU_DEVICE" =~ ^[0-9]+$ ]] \
  || fail "PAMS_GPU_DEVICE must be a non-negative integer"
for mount_path in \
  "$ROOT_INPUT" \
  "$SOURCE_CHECKOUT_INPUT" \
  "$PREDEV_ARTIFACT_INPUT" \
  "$V8_RUN_ROOT_INPUT"; do
  [[ "$mount_path" == /* ]] || fail "host input paths must be absolute"
  [[ "$mount_path" != *","* ]] || fail "Docker bind paths must not contain commas"
done

readonly ROOT="$(realpath -e -- "$ROOT_INPUT")"
readonly SOURCE_CHECKOUT="$(realpath -e -- "$SOURCE_CHECKOUT_INPUT")"
readonly PREDEV_ARTIFACT="$(realpath -e -- "$PREDEV_ARTIFACT_INPUT")"
readonly V8_RUN_ROOT="$(realpath -e -- "$V8_RUN_ROOT_INPUT")"
readonly FIREWALL="${ROOT}/runs/firewall/ucfrep-337-84-105-375d8da"
readonly RUN_PARENT="${ROOT}/runs/pams-position-lag-velocity-fallback-dev-seed2026"
readonly RUN_ROOT="${RUN_PARENT}/${ATTEMPT_ID}"
readonly SOURCE_VIEW="${RUN_ROOT}/source"
readonly SOURCE_RECEIPT="${RUN_ROOT}/source-export.receipt.json"
readonly INPUT_ROOT="${RUN_ROOT}/inputs"
readonly DEV_POSE_VIEW="${INPUT_ROOT}/pose-dev84"
readonly PREDICT_STAGE="${RUN_ROOT}/stages/dev-predict"
readonly SCORE_STAGE="${RUN_ROOT}/stages/dev-score"
readonly AUDIT_ROOT="${RUN_ROOT}/audit"
readonly LOG_ROOT="${RUN_ROOT}/logs"
readonly PREDICT_SCRIPT="${AUDIT_ROOT}/predict.py"
readonly SCORE_SCRIPT="${AUDIT_ROOT}/score.py"
readonly HASH_MANIFEST="${AUDIT_ROOT}/artifact-sha256.txt"
readonly LOCK_ROOT="${ROOT}/.pams-gpu-locks"
readonly LOCK_PATH="${LOCK_ROOT}/gpu${GPU_DEVICE}.lock"
readonly PREDICT_NAME="plagvfpred-${ATTEMPT_ID}"
readonly SCORE_NAME="plagvfscore-${ATTEMPT_ID}"

readonly ENCODER_CHECKPOINT="${V8_RUN_ROOT}/stages/encoder/run/encoder.pt"
readonly POSE_POOL="${V8_RUN_ROOT}/inputs/pose-train337-dev84"
readonly DEV_INPUT="${FIREWALL}/dev.inputs.json"
readonly DEV_TARGET="${FIREWALL}/dev.targets.json"

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
      > "${AUDIT_ROOT}/${ACTIVE_CONTAINER}.error.inspect.json" 2>/dev/null
    if [[ "$(
      docker inspect "$ACTIVE_CONTAINER" \
        --format '{{.State.Running}}' 2>/dev/null
    )" == "true" ]]; then
      docker stop --time 10 "$ACTIVE_CONTAINER" >/dev/null 2>&1
    fi
    docker rm "$ACTIVE_CONTAINER" >/dev/null 2>&1
  fi
  if [[ "$RUN_RESERVED" -eq 1 ]]; then
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
test -d "$FIREWALL" || fail "missing frozen UCFRep firewall"
assert_sha256 "$ENCODER_CHECKPOINT" "$ENCODER_SHA256"
assert_sha256 "$PREDEV_ARTIFACT" "$PREDEV_ARTIFACT_SHA256"
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

# This validation is label-free.  It refuses any failed, stale, or differently
# bound predev artifact before reserving a dev84 run directory.
PAMS_BOUND_PREDEV="$PREDEV_ARTIFACT" \
PAMS_BOUND_SOURCE_REVISION="$SOURCE_REVISION" \
PAMS_BOUND_READOUT_SHA256="$READOUT_CONFIG_SHA256" \
PAMS_BOUND_READOUT_SEMANTIC_SHA256="$READOUT_CONFIG_SEMANTIC_SHA256" \
PAMS_BOUND_EXPERIMENT_SHA256="$EXPERIMENT_CONFIG_SHA256" \
PAMS_BOUND_ENCODER_SHA256="$ENCODER_SHA256" \
python3 - <<'HOSTPY'
import json
import os
from pathlib import Path

artifact = json.loads(Path(os.environ["PAMS_BOUND_PREDEV"]).read_text(encoding="utf-8"))
required = {
    "artifact_type": "pams_position_lag_velocity_fallback_predev",
    "status": "passed",
    "method_key": "pams-projected-position-lag-velocity-fallback-predev-v2",
    "protocol": "ucfrep_526",
    "seed": 2026,
    "source_revision": os.environ["PAMS_BOUND_SOURCE_REVISION"],
    "readout_config_sha256": os.environ["PAMS_BOUND_READOUT_SHA256"],
    "readout_config_semantic_sha256": os.environ[
        "PAMS_BOUND_READOUT_SEMANTIC_SHA256"
    ],
    "experiment_config_sha256": os.environ["PAMS_BOUND_EXPERIMENT_SHA256"],
    "encoder_checkpoint_sha256": os.environ["PAMS_BOUND_ENCODER_SHA256"],
    "stop_decision": "eligible_to_build_separate_frozen_dev84_protocol",
    "dev84_scoring_authorized": False,
    "test105_evaluation_authorized": False,
}
for key, expected in required.items():
    if artifact.get(key) != expected:
        raise RuntimeError("formal predev binding mismatch: " + key)
gates = artifact.get("gates")
if not isinstance(gates, dict) or set(gates) != {
    "synthetic_k2_to_k7_with_linear_drift",
    "synthetic_count_2_to_40_with_linear_drift",
    "train337_target_free_distribution",
}:
    raise RuntimeError("formal predev gate set mismatch")
if any(gate.get("status") != "passed" for gate in gates.values()):
    raise RuntimeError("formal predev contains a non-passing gate")
mount = artifact.get("mount_audit")
if not isinstance(mount, dict):
    raise RuntimeError("formal predev mount audit missing")
for key in (
    "dev84_identity_mounted",
    "dev84_pose_mounted",
    "dev84_targets_mounted",
    "test105_identity_mounted",
    "test105_media_mounted",
    "test105_pose_mounted",
    "test105_labels_mounted",
):
    if mount.get(key) is not False:
        raise RuntimeError("formal predev crossed a prohibited boundary: " + key)
HOSTPY

# No dev84 identity or pose path is read until the passed predev artifact and
# all of its immutable source/config/checkpoint bindings have been validated.
assert_sha256 "$DEV_INPUT" "$DEV_INPUT_SHA256"
test -d "$POSE_POOL" || fail "missing exact-v8 pose pool"
[[ "$(find "$POSE_POOL" -maxdepth 1 -type f -name '*.npz' | wc -l)" -eq 421 ]] \
  || fail "exact-v8 pose pool must contain 421 NPZ files"
[[ -z "$(find "$POSE_POOL" -mindepth 1 -type l -print -quit)" ]] \
  || fail "exact-v8 pose pool contains a symlink"

mkdir -p -- "$RUN_PARENT"
mkdir -- "$RUN_ROOT" || fail "immutable run root exists: $RUN_ROOT"
RUN_RESERVED=1
mkdir -p -- \
  "$SOURCE_VIEW" \
  "$DEV_POSE_VIEW" \
  "$PREDICT_STAGE" \
  "$SCORE_STAGE" \
  "$AUDIT_ROOT" \
  "$LOG_ROOT"
readonly RUNNER_PATH="$(realpath -e -- "${BASH_SOURCE[0]}")"
readonly RUNNER_SHA256="$(sha256_of "$RUNNER_PATH")"
cp -- "$RUNNER_PATH" "${AUDIT_ROOT}/orchestrator-runner.sh"
assert_sha256 "${AUDIT_ROOT}/orchestrator-runner.sh" "$RUNNER_SHA256"
chmod 0444 "${AUDIT_ROOT}/orchestrator-runner.sh"
write_status "running" "$CURRENT_STAGE" 0

CURRENT_STAGE="source-export"
git -C "$SOURCE_CHECKOUT" archive --format=tar "$SOURCE_REVISION" -- \
  src \
  pyproject.toml \
  "$EXPERIMENT_CONFIG_RELATIVE" \
  "$READOUT_CONFIG_RELATIVE" \
  | tar -xf - -C "$SOURCE_VIEW"
SOURCE_EXPORT_ROOT="$SOURCE_VIEW" \
SOURCE_EXPORT_REVISION="$SOURCE_REVISION" \
SOURCE_EXPORT_OUTPUT="$SOURCE_RECEIPT" \
python3 - <<'HOSTPY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["SOURCE_EXPORT_ROOT"])
rows = []
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    rows.append(
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
payload = {
    "schema_version": 1,
    "artifact_type": "clean_source_export_receipt",
    "source_revision": os.environ["SOURCE_EXPORT_REVISION"],
    "file_total": len(rows),
    "files": rows,
}
path = Path(os.environ["SOURCE_EXPORT_OUTPUT"])
with path.open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
HOSTPY
SOURCE_RECEIPT_SHA256="$(sha256_of "$SOURCE_RECEIPT")"
chmod -R a-w "$SOURCE_VIEW"
chmod 0444 "$SOURCE_RECEIPT"

CURRENT_STAGE="predev-receipt"
PREDEV_VALIDATION_OUTPUT="${AUDIT_ROOT}/predev-validation.json" \
PREDEV_VALIDATION_PATH="$PREDEV_ARTIFACT" \
PREDEV_VALIDATION_SHA256="$PREDEV_ARTIFACT_SHA256" \
PREDEV_VALIDATION_SOURCE="$SOURCE_REVISION" \
PREDEV_VALIDATION_CONFIG="$READOUT_CONFIG_SHA256" \
python3 - <<'HOSTPY'
import json
import os
from pathlib import Path

payload = {
    "schema_version": 1,
    "artifact_type": "formal_predev_validation_receipt",
    "status": "passed",
    "predev_artifact_path": str(Path(os.environ["PREDEV_VALIDATION_PATH"])),
    "predev_artifact_sha256": os.environ["PREDEV_VALIDATION_SHA256"],
    "source_revision": os.environ["PREDEV_VALIDATION_SOURCE"],
    "readout_config_sha256": os.environ["PREDEV_VALIDATION_CONFIG"],
    "validated_before_dev_inputs_or_pose_access": True,
    "predev_had_dev84_identity_pose_or_targets": False,
}
path = Path(os.environ["PREDEV_VALIDATION_OUTPUT"])
with path.open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
HOSTPY
chmod 0444 "${AUDIT_ROOT}/predev-validation.json"

CURRENT_STAGE="dev84-pose-view"
while IFS= read -r video_id; do
  [[ "$video_id" =~ ^[A-Za-z0-9_]+$ ]] || fail "unsafe dev84 video_id"
  source_pose="${POSE_POOL}/${video_id}.npz"
  test -f "$source_pose" || fail "missing dev84 pose cache: $video_id"
  cp --reflink=auto --preserve=mode,timestamps -- "$source_pose" "$DEV_POSE_VIEW/"
done < <(jq -r '.records[].video_id' "$DEV_INPUT")
[[ "$(find "$DEV_POSE_VIEW" -maxdepth 1 -type f -name '*.npz' | wc -l)" -eq 84 ]] \
  || fail "dev84 pose view must contain exactly 84 NPZ files"
[[ "$(find "$DEV_POSE_VIEW" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 84 ]] \
  || fail "dev84 pose view contains unexpected files"
[[ -z "$(find "$DEV_POSE_VIEW" -mindepth 1 -type l -print -quit)" ]] \
  || fail "dev84 pose view contains a symlink"
(
  cd "$DEV_POSE_VIEW"
  find . -maxdepth 1 -type f -name '*.npz' -print0 \
    | sort -z \
    | xargs -0 sha256sum
) > "${AUDIT_ROOT}/pose-dev84.sha256.tsv"
chmod 0444 "$DEV_POSE_VIEW"/*.npz "${AUDIT_ROOT}/pose-dev84.sha256.tsv"
chmod 0555 "$DEV_POSE_VIEW"

cat > "$PREDICT_SCRIPT" <<'PREDICTPY'
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import traceback
from pathlib import Path

import torch
import yaml

from pams.config import load_config
from pams.data import load_pose_cache_set, load_pose_input_manifest
from pams.metrics import round_count
from pams.period import (
    estimate_period_from_projected_position_lag_velocity_fallback,
    projected_position_lag_velocity_fallback_diagnostics,
)
from pams.reproducibility import hardware_fingerprint, sha256_file, sha256_json
from pams.training import (
    CheckpointProvenance,
    collate_pose_sequences,
    load_model_checkpoint,
)


def write_exclusive(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_policy(path):
    if sha256_file(path) != os.environ["PAMS_READOUT_CONFIG_SHA256"]:
        raise ValueError("readout config byte identity mismatch")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if sha256_json(payload) != os.environ["PAMS_READOUT_CONFIG_SEMANTIC_SHA256"]:
        raise ValueError("readout config semantic identity mismatch")
    if payload.get("key") != (
        "pams-projected-position-lag-velocity-fallback-predev-v2"
    ):
        raise ValueError("readout method key mismatch")
    readout = payload.get("readout", {})
    if readout.get("primary_near_best_height_fraction") != 0.90:
        raise ValueError("near-best height fraction drifted")
    if readout.get("prediction_batch_size") != 32:
        raise ValueError("prediction batch size drifted")
    return payload


def infer(sequences, model, config, device):
    items = tuple(sequences)
    if len(items) != 84 or len({item.video_id for item in items}) != 84:
        raise ValueError("prediction requires exactly 84 unique dev sequences")
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
                estimate_period_from_projected_position_lag_velocity_fallback(
                    projected,
                    minimum=config.period.minimum,
                    maximum=config.period.maximum,
                    valid_mask=batch.valid_mask,
                )
            )
            diagnostics = (
                projected_position_lag_velocity_fallback_diagnostics(
                    projected,
                    minimum=config.period.minimum,
                    maximum=config.period.maximum,
                    valid_mask=batch.valid_mask,
                )
            )
            valid_counts = batch.valid_mask.sum(dim=1)
            for index, video_id in enumerate(batch.video_ids):
                diagnostic = diagnostics[index]
                period = float(periods[index])
                confidence = float(confidences[index])
                valid_frames = int(valid_counts[index])
                if not math.isclose(
                    period,
                    float(diagnostic.selected_period),
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ):
                    raise RuntimeError("estimator/diagnostic period mismatch")
                if not math.isclose(
                    confidence,
                    float(diagnostic.confidence),
                    rel_tol=1e-6,
                    abs_tol=1e-7,
                ):
                    raise RuntimeError("estimator/diagnostic confidence mismatch")
                raw_count = (
                    float(valid_frames - 1) / period
                    if confidence > 0.0 and valid_frames >= 2
                    else 0.0
                )
                row = {
                    "video_id": video_id,
                    "period_frames": period,
                    "confidence": confidence,
                    "selection_source": diagnostic.selection_source,
                    "raw_count": raw_count,
                    "rounded_count": round_count(raw_count),
                }
                if not all(
                    math.isfinite(float(row[name]))
                    for name in ("period_frames", "confidence", "raw_count")
                ):
                    raise RuntimeError("non-finite prediction output")
                if row["selection_source"] not in {
                    "detrended-position-lag-acf",
                    "projected-velocity-spectrum-fallback",
                    "no-evidence",
                }:
                    raise RuntimeError("selection source drifted")
                if confidence <= 0.0 and (
                    raw_count != 0.0 or row["rounded_count"] != 0
                ):
                    raise RuntimeError("zero evidence must produce zero count")
                rows.append(row)
    if batch_sizes != [32, 32, 20]:
        raise RuntimeError("prediction batch partition drifted")
    return rows, batch_sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--readout-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dev-inputs", type=Path, required=True)
    parser.add_argument("--pose-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = load_policy(args.readout_config)
    config = load_config(args.experiment_config)
    if sha256_file(args.experiment_config) != os.environ[
        "PAMS_EXPERIMENT_CONFIG_SHA256"
    ]:
        raise ValueError("experiment config byte identity mismatch")
    if sha256_file(args.checkpoint) != os.environ["PAMS_ENCODER_SHA256"]:
        raise ValueError("encoder checkpoint identity mismatch")
    if sha256_file(args.dev_inputs) != os.environ["PAMS_DEV_INPUT_SHA256"]:
        raise ValueError("dev input identity mismatch")
    if config.protocol != "ucfrep_526" or config.seed != 2026:
        raise ValueError("experiment protocol or seed mismatch")
    manifest = load_pose_input_manifest(args.dev_inputs)
    if manifest.protocol != "ucfrep_526":
        raise ValueError("dev protocol mismatch")
    if manifest.split != "dev" or len(manifest.records) != 84:
        raise ValueError("dev identity sidecar mismatch")
    manifest.validate_exact_membership()
    sequences, snapshot = load_pose_cache_set(
        manifest.records,
        cache_dir=args.pose_cache,
        pose_fingerprint=config.pose_fingerprint,
    )
    if snapshot.fingerprint != os.environ["PAMS_DEV_POSE_SET_SHA256"]:
        raise ValueError("dev pose-cache-set fingerprint mismatch")
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    provenance = CheckpointProvenance.from_mapping(raw["provenance"])
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("formal prediction requires one CUDA GPU")
    model = load_model_checkpoint(
        args.checkpoint,
        config,
        device=device,
        expected_stage="encoder",
        expected_provenance=provenance,
    ).eval()
    rows, batch_sizes = infer(sequences, model, config, device)
    payload = {
        "schema_version": 1,
        "artifact_type": "pams_position_lag_velocity_fallback_dev_predictions",
        "classification": policy["classification"],
        "eligible_for_paper_table": False,
        "protocol": "ucfrep_526",
        "split": "dev",
        "seed": 2026,
        "record_total": 84,
        "batch_sizes": batch_sizes,
        "source_revision": os.environ["PAMS_SOURCE_REVISION"],
        "readout_config_sha256": os.environ["PAMS_READOUT_CONFIG_SHA256"],
        "readout_config_semantic_sha256": os.environ[
            "PAMS_READOUT_CONFIG_SEMANTIC_SHA256"
        ],
        "formal_predev_artifact_sha256": os.environ[
            "PAMS_PREDEV_ARTIFACT_SHA256"
        ],
        "encoder_checkpoint_sha256": sha256_file(args.checkpoint),
        "dev_inputs_sha256": sha256_file(args.dev_inputs),
        "dev_pose_cache_set_sha256": snapshot.fingerprint,
        "hardware": hardware_fingerprint(),
        "mount_audit": {
            "network": "none",
            "source_mounted": True,
            "readout_config_mounted": True,
            "checkpoint_mounted": True,
            "dev84_identity_mounted": True,
            "dev84_pose_mounted": True,
            "dev84_targets_mounted": False,
            "test105_identity_mounted": False,
            "test105_media_mounted": False,
            "test105_pose_mounted": False,
            "test105_labels_mounted": False,
        },
        "records": rows,
    }
    output = args.output / "predictions.json"
    write_exclusive(output, payload)
    prediction_bytes = output.read_bytes()
    receipt = {
        "schema_version": 1,
        "artifact_type": (
            "pams_position_lag_velocity_fallback_dev_prediction_receipt"
        ),
        "prediction_file": output.name,
        "prediction_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "prediction_bytes": len(prediction_bytes),
        "prediction_total": 84,
        "source_revision": os.environ["PAMS_SOURCE_REVISION"],
        "readout_config_sha256": os.environ["PAMS_READOUT_CONFIG_SHA256"],
        "formal_predev_artifact_sha256": os.environ[
            "PAMS_PREDEV_ARTIFACT_SHA256"
        ],
        "dev_inputs_sha256": sha256_file(args.dev_inputs),
        "dev_pose_cache_set_sha256": snapshot.fingerprint,
        "dev84_targets_mounted": False,
        "test105_evaluation_authorized": False,
    }
    write_exclusive(args.output / "prediction.receipt.json", receipt)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
PREDICTPY
chmod 0444 "$PREDICT_SCRIPT"

cat > "$SCORE_SCRIPT" <<'SCOREPY'
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json_unique(path):
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field: " + key)
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("non-finite JSON constant: " + value)
        ),
    )


def write_exclusive(path, payload):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def validate_predictions(prediction_path, receipt_path):
    prediction_bytes = prediction_path.read_bytes()
    receipt_bytes = receipt_path.read_bytes()
    prediction = load_json_unique(prediction_path)
    receipt = load_json_unique(receipt_path)
    if prediction.get("artifact_type") != (
        "pams_position_lag_velocity_fallback_dev_predictions"
    ):
        raise ValueError("prediction artifact type mismatch")
    expected_top = {
        "schema_version",
        "artifact_type",
        "classification",
        "eligible_for_paper_table",
        "protocol",
        "split",
        "seed",
        "record_total",
        "batch_sizes",
        "source_revision",
        "readout_config_sha256",
        "readout_config_semantic_sha256",
        "formal_predev_artifact_sha256",
        "encoder_checkpoint_sha256",
        "dev_inputs_sha256",
        "dev_pose_cache_set_sha256",
        "hardware",
        "mount_audit",
        "records",
    }
    if set(prediction) != expected_top:
        raise ValueError("prediction schema mismatch")
    bindings = {
        "schema_version": 1,
        "protocol": "ucfrep_526",
        "split": "dev",
        "seed": 2026,
        "record_total": 84,
        "batch_sizes": [32, 32, 20],
        "source_revision": os.environ["PAMS_SOURCE_REVISION"],
        "readout_config_sha256": os.environ["PAMS_READOUT_CONFIG_SHA256"],
        "readout_config_semantic_sha256": os.environ[
            "PAMS_READOUT_CONFIG_SEMANTIC_SHA256"
        ],
        "formal_predev_artifact_sha256": os.environ[
            "PAMS_PREDEV_ARTIFACT_SHA256"
        ],
        "encoder_checkpoint_sha256": os.environ["PAMS_ENCODER_SHA256"],
        "dev_inputs_sha256": os.environ["PAMS_DEV_INPUT_SHA256"],
        "dev_pose_cache_set_sha256": os.environ["PAMS_DEV_POSE_SET_SHA256"],
    }
    for key, expected in bindings.items():
        if prediction.get(key) != expected:
            raise ValueError("prediction binding mismatch: " + key)
    mount = prediction.get("mount_audit")
    if not isinstance(mount, dict) or mount.get("dev84_targets_mounted") is not False:
        raise ValueError("prediction target isolation was not proven")
    for key in (
        "test105_identity_mounted",
        "test105_media_mounted",
        "test105_pose_mounted",
        "test105_labels_mounted",
    ):
        if mount.get(key) is not False:
            raise ValueError("prediction crossed a prohibited boundary: " + key)
    rows = prediction.get("records")
    if not isinstance(rows, list) or len(rows) != 84:
        raise ValueError("prediction row count mismatch")
    expected_row = {
        "video_id",
        "period_frames",
        "confidence",
        "selection_source",
        "raw_count",
        "rounded_count",
    }
    ids = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_row:
            raise ValueError("prediction row schema mismatch")
        ids.append(row["video_id"])
        if not isinstance(row["video_id"], str) or not row["video_id"]:
            raise ValueError("prediction video identifier is invalid")
        if row["selection_source"] not in {
            "detrended-position-lag-acf",
            "projected-velocity-spectrum-fallback",
            "no-evidence",
        }:
            raise ValueError("prediction selection source is invalid")
        numeric = [
            float(row["period_frames"]),
            float(row["confidence"]),
            float(row["raw_count"]),
        ]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("prediction contains a non-finite value")
        if numeric[0] < 4.0 or numeric[0] > 128.0:
            raise ValueError("prediction period is outside the frozen range")
        if numeric[1] < 0.0 or numeric[2] < 0.0:
            raise ValueError("prediction confidence/count is negative")
        if int(row["rounded_count"]) != row["rounded_count"]:
            raise ValueError("rounded count is not integral")
        if int(math.floor(numeric[2] + 0.5)) != int(row["rounded_count"]):
            raise ValueError("rounded count is not half-up(raw_count)")
    if len(set(ids)) != 84:
        raise ValueError("prediction video identifiers are not unique")
    if receipt.get("artifact_type") != (
        "pams_position_lag_velocity_fallback_dev_prediction_receipt"
    ):
        raise ValueError("prediction receipt type mismatch")
    if receipt.get("prediction_sha256") != hashlib.sha256(
        prediction_bytes
    ).hexdigest():
        raise ValueError("prediction receipt hash mismatch")
    if receipt.get("prediction_bytes") != len(prediction_bytes):
        raise ValueError("prediction receipt byte count mismatch")
    if receipt.get("prediction_total") != 84:
        raise ValueError("prediction receipt row count mismatch")
    for key in (
        "source_revision",
        "readout_config_sha256",
        "formal_predev_artifact_sha256",
        "dev_inputs_sha256",
        "dev_pose_cache_set_sha256",
    ):
        if receipt.get(key) != prediction.get(key):
            raise ValueError("prediction receipt binding mismatch: " + key)
    return prediction, rows, prediction_bytes, receipt_bytes


def metric_values(raw, rounded, targets):
    raw_array = np.asarray(raw, dtype=np.float64)
    rounded_array = np.asarray(rounded, dtype=np.int64)
    target_array = np.asarray(targets, dtype=np.int64)
    rounded_error = np.abs(rounded_array - target_array)
    raw_error = np.abs(raw_array - target_array)
    return {
        "nmae_rounded": float(np.mean(rounded_error / target_array)),
        "mae_raw_prediction": float(np.mean(raw_error)),
        "rmse_raw_prediction": float(
            np.sqrt(np.mean(np.square(raw_array - target_array)))
        ),
        "mae_rounded": float(np.mean(rounded_error)),
        "rmse_rounded": float(np.sqrt(np.mean(np.square(rounded_error)))),
        "obo_rounded": float(np.mean(rounded_error <= 1)),
        "exact_rounded": float(np.mean(rounded_error == 0)),
    }


def paired_bootstrap(raw, rounded, targets):
    rng = np.random.default_rng(2026)
    count = len(targets)
    samples = {key: np.empty(10_000) for key in metric_values(raw, rounded, targets)}
    for start in range(0, 10_000, 500):
        stop = min(start + 500, 10_000)
        indices = rng.integers(0, count, size=(stop - start, count))
        for offset, sample in enumerate(indices, start=start):
            values = metric_values(
                np.asarray(raw)[sample],
                np.asarray(rounded)[sample],
                np.asarray(targets)[sample],
            )
            for key, value in values.items():
                samples[key][offset] = value
    return {
        key: {
            "low": float(np.quantile(values, 0.025)),
            "high": float(np.quantile(values, 0.975)),
            "level": 0.95,
        }
        for key, values in samples.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-receipt", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--metric-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Every operation above this line is target-free prediction validation.
    prediction, rows, prediction_bytes, receipt_bytes = validate_predictions(
        args.predictions,
        args.prediction_receipt,
    )
    prediction_sha = hashlib.sha256(prediction_bytes).hexdigest()
    prediction_receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    metric_code_sha = sha256_file(args.metric_code)
    spec = importlib.util.spec_from_file_location("frozen_pams_metrics", args.metric_code)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load frozen metric code")
    metric_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = metric_module
    spec.loader.exec_module(metric_module)
    round_counts = metric_module.round_counts

    # First and only target-bearing read happens after prediction validation.
    target_sha_before = sha256_file(args.targets)
    if target_sha_before != os.environ["PAMS_DEV_TARGET_SHA256"]:
        raise ValueError("dev target identity mismatch")
    targets_payload = load_json_unique(args.targets)
    expected_target_top = {
        "schema_version",
        "manifest_type",
        "protocol",
        "split",
        "source_annotation_sha256",
        "records",
    }
    if not isinstance(targets_payload, dict) or set(targets_payload) != expected_target_top:
        raise ValueError("dev target schema mismatch")
    if (
        targets_payload["schema_version"] != 1
        or targets_payload["manifest_type"] != "dev_targets"
        or targets_payload["protocol"] != "ucfrep_526"
        or targets_payload["split"] != "dev"
    ):
        raise ValueError("dev target protocol binding mismatch")
    target_rows = targets_payload.get("records")
    if not isinstance(target_rows, list) or len(target_rows) != 84:
        raise ValueError("dev target row count mismatch")
    if any(
        not isinstance(row, dict)
        or set(row) != {"video_id", "action", "count"}
        for row in target_rows
    ):
        raise ValueError("dev target row schema mismatch")
    if [row["video_id"] for row in target_rows] != [
        row["video_id"] for row in rows
    ]:
        raise ValueError("prediction and target identity/order mismatch")
    target_counts = [int(row["count"]) for row in target_rows]
    if any(
        isinstance(row["count"], bool)
        or target != row["count"]
        or target <= 0
        for row, target in zip(target_rows, target_counts)
    ):
        raise ValueError("dev target counts must be positive integers")
    raw = [float(row["raw_count"]) for row in rows]
    rounded = [int(value) for value in round_counts(raw)]
    if rounded != [int(row["rounded_count"]) for row in rows]:
        raise ValueError("frozen metric code disagrees with prediction rounding")
    metrics = metric_values(raw, rounded, target_counts)
    intervals = paired_bootstrap(raw, rounded, target_counts)
    if sha256_file(args.targets) != target_sha_before:
        raise RuntimeError("dev targets changed during scoring")
    if sha256_file(args.predictions) != prediction_sha:
        raise RuntimeError("predictions changed during scoring")
    if sha256_file(args.prediction_receipt) != prediction_receipt_sha:
        raise RuntimeError("prediction receipt changed during scoring")
    if sha256_file(args.metric_code) != metric_code_sha:
        raise RuntimeError("metric code changed during scoring")
    evaluation = {
        "schema_version": 1,
        "artifact_type": "pams_position_lag_velocity_fallback_dev_evaluation",
        "classification": prediction["classification"],
        "eligible_for_paper_table": False,
        "protocol": "ucfrep_526",
        "split": "dev",
        "seed": 2026,
        "record_total": 84,
        "metrics": metrics,
        "bootstrap": {
            "samples": 10_000,
            "seed": 2026,
            "confidence_level": 0.95,
            "interval_method": "percentile",
            "pairing": "paired_prediction_target_rows",
            "confidence_intervals": intervals,
        },
        "provenance": {
            "source_revision": os.environ["PAMS_SOURCE_REVISION"],
            "readout_config_sha256": os.environ["PAMS_READOUT_CONFIG_SHA256"],
            "formal_predev_artifact_sha256": os.environ[
                "PAMS_PREDEV_ARTIFACT_SHA256"
            ],
            "prediction_sha256": prediction_sha,
            "prediction_receipt_sha256": prediction_receipt_sha,
            "dev_targets_sha256": target_sha_before,
            "metric_code_sha256": metric_code_sha,
        },
        "mount_audit": {
            "network": "none",
            "prediction_mounted": True,
            "dev84_targets_mounted": True,
            "pose_mounted": False,
            "checkpoint_mounted": False,
            "full_source_tree_mounted": False,
            "metric_code_only": True,
            "test105_identity_mounted": False,
            "test105_media_mounted": False,
            "test105_pose_mounted": False,
            "test105_labels_mounted": False,
        },
        "per_video": [
            {
                "video_id": row["video_id"],
                "prediction": raw[index],
                "rounded_prediction": rounded[index],
                "target": target_counts[index],
                "absolute_error_raw": abs(raw[index] - target_counts[index]),
                "absolute_error_rounded": abs(rounded[index] - target_counts[index]),
                "normalized_absolute_error_rounded": (
                    abs(rounded[index] - target_counts[index])
                    / target_counts[index]
                ),
                "within_one_rounded": abs(rounded[index] - target_counts[index]) <= 1,
                "exact_rounded": rounded[index] == target_counts[index],
            }
            for index, row in enumerate(rows)
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    evaluation_path = args.output / "evaluation.json"
    write_exclusive(evaluation_path, evaluation)
    evaluation_bytes = evaluation_path.read_bytes()
    receipt = {
        "schema_version": 1,
        "artifact_type": (
            "pams_position_lag_velocity_fallback_dev_evaluation_receipt"
        ),
        "evaluation_file": evaluation_path.name,
        "evaluation_sha256": hashlib.sha256(evaluation_bytes).hexdigest(),
        "evaluation_bytes": len(evaluation_bytes),
        "prediction_sha256": prediction_sha,
        "prediction_receipt_sha256": prediction_receipt_sha,
        "dev_targets_sha256": target_sha_before,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 2026,
        "test105_evaluation_authorized": False,
    }
    write_exclusive(args.output / "evaluation.receipt.json", receipt)


if __name__ == "__main__":
    main()
SCOREPY
chmod 0444 "$SCORE_SCRIPT"

cat > "${RUN_ROOT}/attempt.reservation.json" <<EOF
{
  "schema_version": 1,
  "attempt_id": "${ATTEMPT_ID}",
  "scope": "formal-predev-bound-dev84-predict-and-isolated-score",
  "seed": 2026,
  "source_revision": "${SOURCE_REVISION}",
  "algorithm_source_revision": "${SOURCE_REVISION}",
  "orchestrator_runner_sha256": "${RUNNER_SHA256}",
  "source_export_receipt_sha256": "${SOURCE_RECEIPT_SHA256}",
  "readout_config_sha256": "${READOUT_CONFIG_SHA256}",
  "readout_config_semantic_sha256": "${READOUT_CONFIG_SEMANTIC_SHA256}",
  "formal_predev_artifact_sha256": "${PREDEV_ARTIFACT_SHA256}",
  "container_image_id": "${IMAGE_ID}",
  "container_environment_sha256": "${ENVIRONMENT_SHA256}",
  "prediction_network": "none",
  "prediction_targets_mounted": false,
  "score_network": "none",
  "score_pose_mounted": false,
  "score_checkpoint_mounted": false,
  "test105_evaluation_authorized": false
}
EOF
chmod 0444 "${RUN_ROOT}/attempt.reservation.json"

CURRENT_STAGE="hardware-audit"
nvidia-smi -q > "${AUDIT_ROOT}/nvidia-smi-q.txt"
lscpu > "${AUDIT_ROOT}/lscpu.txt"
docker version > "${AUDIT_ROOT}/docker-version.txt"
docker image inspect "$IMAGE_ID" > "${AUDIT_ROOT}/image.inspect.json"

for container_name in "$PREDICT_NAME" "$SCORE_NAME"; do
  if docker inspect "$container_name" >/dev/null 2>&1; then
    fail "container name already exists: $container_name"
  fi
done
mkdir -p -- "$LOCK_ROOT"
[[ ! -L "$LOCK_ROOT" ]] || fail "GPU lock root must not be a symlink"
[[ "$(realpath -e -- "$LOCK_ROOT")" == "${ROOT}/.pams-gpu-locks" ]] \
  || fail "GPU lock root resolves outside PAMS_ROOT"
[[ ! -L "$LOCK_PATH" ]] || fail "GPU lock file must not be a symlink"
exec 9>"$LOCK_PATH"
chmod 0600 "$LOCK_PATH"
if ! flock -n 9; then
  CURRENT_STAGE="gpu-lock"
  exit 75
fi

run_container() {
  local name="$1"
  local stage="$2"
  ACTIVE_CONTAINER="$name"
  docker inspect "$name" > "${AUDIT_ROOT}/${name}.pre-run.inspect.json"
  set +e
  docker start --attach "$name" \
    > "${LOG_ROOT}/${name}.stdout.log" \
    2> "${LOG_ROOT}/${name}.stderr.log"
  CONTAINER_EXIT="$?"
  set -e
  docker inspect "$name" > "${AUDIT_ROOT}/${name}.post-run.inspect.json"
  actual_exit="$(docker inspect "$name" --format '{{.State.ExitCode}}')"
  [[ "$actual_exit" -eq "$CONTAINER_EXIT" ]] \
    || fail "$stage attach/container exit mismatch"
  docker rm "$name" > "${AUDIT_ROOT}/${name}.removed-id.txt"
  ACTIVE_CONTAINER=""
  [[ "$CONTAINER_EXIT" -eq 0 ]] || fail "$stage container failed"
}

common_args=(
  --init
  --network none
  --read-only
  --shm-size 8g
  --pids-limit 4096
  --cpus 24
  --memory 96g
  --cap-drop ALL
  --security-opt no-new-privileges:true
  --user 1000:1000
  --env PYTHONDONTWRITEBYTECODE=1
  --env HOME=/pams/home
  --env "PAMS_SOURCE_REVISION=${SOURCE_REVISION}"
  --env "PAMS_EXPERIMENT_CONFIG_SHA256=${EXPERIMENT_CONFIG_SHA256}"
  --env "PAMS_READOUT_CONFIG_SHA256=${READOUT_CONFIG_SHA256}"
  --env "PAMS_READOUT_CONFIG_SEMANTIC_SHA256=${READOUT_CONFIG_SEMANTIC_SHA256}"
  --env "PAMS_PREDEV_ARTIFACT_SHA256=${PREDEV_ARTIFACT_SHA256}"
  --env "PAMS_ENCODER_SHA256=${ENCODER_SHA256}"
  --env "PAMS_DEV_INPUT_SHA256=${DEV_INPUT_SHA256}"
  --env "PAMS_DEV_POSE_SET_SHA256=${DEV_POSE_SET_SHA256}"
  --tmpfs /tmp:rw,noexec,nosuid,nodev,uid=1000,gid=1000,size=2g
  --tmpfs /pams/home:rw,noexec,nosuid,nodev,uid=1000,gid=1000,size=1g
)

CURRENT_STAGE="dev-predict-create"
docker create \
  --name "$PREDICT_NAME" \
  "${common_args[@]}" \
  --gpus "device=${GPU_DEVICE}" \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env PYTHONPATH=/workspace/src \
  --workdir /workspace \
  --mount "type=bind,src=${SOURCE_VIEW},dst=/workspace,readonly" \
  --mount \
    "type=bind,src=${SOURCE_VIEW}/${READOUT_CONFIG_RELATIVE},dst=/pams/input/readout-config.yaml,readonly" \
  --mount \
    "type=bind,src=${SOURCE_VIEW}/${EXPERIMENT_CONFIG_RELATIVE},dst=/pams/input/experiment-config.yaml,readonly" \
  --mount \
    "type=bind,src=${ENCODER_CHECKPOINT},dst=/pams/encoder/encoder.pt,readonly" \
  --mount \
    "type=bind,src=${DEV_INPUT},dst=/pams/protocol/dev.inputs.json,readonly" \
  --mount \
    "type=bind,src=${DEV_POSE_VIEW},dst=/pams/pose-cache,readonly" \
  --mount "type=bind,src=${PREDICT_SCRIPT},dst=/pams/predict.py,readonly" \
  --mount "type=bind,src=${PREDICT_STAGE},dst=/pams/output" \
  "$IMAGE_ID" \
  python /pams/predict.py \
    --experiment-config /pams/input/experiment-config.yaml \
    --readout-config /pams/input/readout-config.yaml \
    --checkpoint /pams/encoder/encoder.pt \
    --dev-inputs /pams/protocol/dev.inputs.json \
    --pose-cache /pams/pose-cache \
    --output /pams/output \
  > "${AUDIT_ROOT}/${PREDICT_NAME}.create-id.txt"
run_container "$PREDICT_NAME" "dev-predict"
test -s "${PREDICT_STAGE}/predictions.json" \
  || fail "prediction artifact missing"
test -s "${PREDICT_STAGE}/prediction.receipt.json" \
  || fail "prediction receipt missing"
sha256sum \
  "${PREDICT_STAGE}/predictions.json" \
  "${PREDICT_STAGE}/prediction.receipt.json" \
  > "${AUDIT_ROOT}/frozen-prediction-sha256.txt"
chmod -R a-w "$PREDICT_STAGE"
flock -u 9

# First target-bearing boundary.  Prediction bytes are already frozen and the
# score container cannot see source, pose, checkpoint, or accelerator devices.
CURRENT_STAGE="dev-score-target-boundary"
assert_sha256 "$DEV_TARGET" "$DEV_TARGET_SHA256"

CURRENT_STAGE="dev-score-create"
docker create \
  --name "$SCORE_NAME" \
  "${common_args[@]}" \
  --env CUDA_VISIBLE_DEVICES= \
  --env "PAMS_DEV_TARGET_SHA256=${DEV_TARGET_SHA256}" \
  --workdir /pams \
  --mount \
    "type=bind,src=${SOURCE_VIEW}/src/pams/metrics.py,dst=/pams/metric-code/metrics.py,readonly" \
  --mount \
    "type=bind,src=${PREDICT_STAGE}/predictions.json,dst=/pams/frozen/predictions.json,readonly" \
  --mount \
    "type=bind,src=${PREDICT_STAGE}/prediction.receipt.json,dst=/pams/frozen/prediction.receipt.json,readonly" \
  --mount \
    "type=bind,src=${DEV_TARGET},dst=/pams/protocol/dev.targets.json,readonly" \
  --mount "type=bind,src=${SCORE_SCRIPT},dst=/pams/score.py,readonly" \
  --mount "type=bind,src=${SCORE_STAGE},dst=/pams/output" \
  "$IMAGE_ID" \
  python /pams/score.py \
    --predictions /pams/frozen/predictions.json \
    --prediction-receipt /pams/frozen/prediction.receipt.json \
    --targets /pams/protocol/dev.targets.json \
    --metric-code /pams/metric-code/metrics.py \
    --output /pams/output \
  > "${AUDIT_ROOT}/${SCORE_NAME}.create-id.txt"
run_container "$SCORE_NAME" "dev-score"
test -s "${SCORE_STAGE}/evaluation.json" \
  || fail "evaluation artifact missing"
test -s "${SCORE_STAGE}/evaluation.receipt.json" \
  || fail "evaluation receipt missing"
sha256sum \
  "${SCORE_STAGE}/evaluation.json" \
  "${SCORE_STAGE}/evaluation.receipt.json" \
  > "${AUDIT_ROOT}/dev-score-sha256.txt"
chmod -R a-w "$SCORE_STAGE"

CURRENT_STAGE="complete"
write_status "completed" "$CURRENT_STAGE" 0
write_hash_manifest
chmod -R a-w "$RUN_ROOT"
RUN_RESERVED=0
trap - EXIT HUP INT TERM
printf 'completed %s\n' "$RUN_ROOT"
