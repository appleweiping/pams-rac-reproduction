#!/usr/bin/env bash
# Frozen train337-only pose-input recovery and paired gate.
#
# This launcher performs no training and mounts no count/action label files. It
# seeks the clip range carried by the count-free official-segment sidecar in the
# selected full source videos. The preseeded heavy model is mounted read-only at
# MediaPipe's package resource path; runtime network access is disabled.
set -Eeuo pipefail
IFS=$'\n\t'
umask 027

fail() {
  printf 'pose-recovery-v4a: %s\n' "$*" >&2
  exit 2
}

sha256_file() {
  sha256sum -- "$1" | awk '{print $1}'
}

require_sha256() {
  local path="$1"
  local expected="$2"
  local role="$3"
  [[ -f "$path" ]] || fail "missing ${role}: ${path}"
  local received
  received="$(sha256_file "$path")"
  [[ "$received" == "$expected" ]] \
    || fail "${role} SHA-256 mismatch: expected ${expected}, received ${received}"
}

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
: "${PAMS_V4A_EXPECTED_SOURCE_REVISION:?set the exact 40-hex source revision}"
: "${PAMS_V4A_ATTEMPT_ID:?set an immutable attempt ID such as 20260807T120000Z}"
[[ "$PAMS_V4A_EXPECTED_SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]] \
  || fail "PAMS_V4A_EXPECTED_SOURCE_REVISION must be lowercase 40-hex"
[[ "$PAMS_V4A_ATTEMPT_ID" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] \
  || fail "PAMS_V4A_ATTEMPT_ID must use YYYYMMDDTHHMMSSZ"

readonly SOURCE_REVISION="$(git -C "$REPOSITORY_ROOT" rev-parse HEAD^{commit})"
[[ "$SOURCE_REVISION" == "$PAMS_V4A_EXPECTED_SOURCE_REVISION" ]] \
  || fail "source revision differs from the preregistered revision"
[[ -z "$(git -C "$REPOSITORY_ROOT" status --porcelain=v1 --untracked-files=all)" ]] \
  || fail "source worktree must be clean, including untracked files"

readonly IMAGE='pams-rac:5e18274a5353'
readonly IMAGE_ID='sha256:0a4d42c2d9911f147a17860e4e15095746b21c4e618e4fc1b20b62dd443c5898'
readonly CONFIG_SHA256='c6fe33584f768e774d975059b7a899a4471aacc4cdda9f4821cc13c57bbaabc5'
readonly CONFIG_FINGERPRINT='c4b2dfae47581750281a43cf8eae92e4ceab91d113af601a4155ad3e2dd2f99a'
readonly POSE_FINGERPRINT='817013890533cd19e6c791969c35c3699d56d0e6656768ce64199bc30ceebe9c'
readonly SIDECAR_SHA256='f95df0050df21f05bbc9b42d0154470714dde279e04cb8b00d0212bf49057d16'
readonly COMMITMENT_SHA256='85d41d2f59872e0900e9058481efbdf6bce91407d4c26bcde0a6547776454e53'
readonly REFERENCE_LEDGER_SHA256='2b356adac8efa6d9c8a184230a7fc09d9a8cc3645b0bbab7c8aa00faf76c2a35'
readonly HEAVY_ASSET_SHA256='59e42d71bcd44cbdbabc419f0ff76686595fd265419566bd4009ef703ea8e1fe'
readonly HEAVY_CONTAINER_PATH='/opt/conda/lib/python3.11/site-packages/mediapipe/modules/pose_landmark/pose_landmark_heavy.tflite'

readonly OFFICIAL_ROOT='/media/lenovo/data2/pams-rac/runs/official-segment-v1/15cc1ec3c1d2-20260805T063653Z'
readonly TRAIN_VIDEO_VIEW="${OFFICIAL_ROOT}/video-views/train337"
readonly TRAIN_SIDECAR="${OFFICIAL_ROOT}/protocol/train.inputs.json"
readonly TRAIN_COMMITMENT="${OFFICIAL_ROOT}/protocol/train.inputs.commitment.json"
readonly REFERENCE_CACHE="${OFFICIAL_ROOT}/stages/input-views/pose-train337"
readonly REFERENCE_LEDGER="${OFFICIAL_ROOT}/ledgers/train337.json"
readonly HEAVY_ASSET='/media/lenovo/data2/pams-rac/assets/mediapipe/pose_landmark_heavy.tflite'
readonly RUN_PARENT='/media/lenovo/data2/pams-rac/runs/pose-recovery-v4a'
readonly RUN_ROOT="${RUN_PARENT}/${SOURCE_REVISION:0:12}-${PAMS_V4A_ATTEMPT_ID}"
readonly SOURCE_EXPORT="${RUN_ROOT}/source"
readonly V4_CACHE="${RUN_ROOT}/pose-cache"
readonly V4_LEDGER_DIR="${RUN_ROOT}/ledgers"
readonly V4_LEDGER="${V4_LEDGER_DIR}/train337.json"
readonly AUDIT_DIR="${RUN_ROOT}/audit"
readonly LOG_DIR="${RUN_ROOT}/logs"
readonly CONFIG_PATH="${SOURCE_EXPORT}/configs/experiments/pams_pose_recovery_v4a.yaml"
readonly GATE_PATH="${SOURCE_EXPORT}/configs/gates/pams_pose_recovery_v4a_train337.yaml"

[[ "$(docker image inspect "$IMAGE" --format '{{.Id}}')" == "$IMAGE_ID" ]] \
  || fail "container image ID mismatch"
[[ -d "$TRAIN_VIDEO_VIEW" ]] || fail "missing frozen train337 video view"
[[ -d "$REFERENCE_CACHE" ]] || fail "missing frozen train337 reference cache view"
require_sha256 "$TRAIN_SIDECAR" "$SIDECAR_SHA256" 'train337 sidecar'
require_sha256 "$TRAIN_COMMITMENT" "$COMMITMENT_SHA256" 'train337 commitment'
require_sha256 "$REFERENCE_LEDGER" "$REFERENCE_LEDGER_SHA256" 'reference ledger'
require_sha256 "$HEAVY_ASSET" "$HEAVY_ASSET_SHA256" 'MediaPipe heavy asset'
[[ ! -e "$RUN_ROOT" ]] || fail "refusing to reuse run root: ${RUN_ROOT}"

mkdir -p -- "$RUN_PARENT"
mkdir -- "$RUN_ROOT" "$SOURCE_EXPORT" "$V4_CACHE" "$V4_LEDGER_DIR" "$AUDIT_DIR" "$LOG_DIR"
git -C "$REPOSITORY_ROOT" archive "$SOURCE_REVISION" | tar -x -C "$SOURCE_EXPORT"
chmod -R a-w -- "$SOURCE_EXPORT"
require_sha256 "$CONFIG_PATH" "$CONFIG_SHA256" 'v4a config'
[[ -f "$GATE_PATH" ]] || fail "frozen v4a gate is absent from source export"

readonly CONFIG_IDENTITIES="$({
  docker run --rm \
    --network none \
    --read-only \
    --tmpfs /tmp:rw,nosuid,nodev,size=64m \
    --env PYTHONPATH=/workspace/src \
    --mount "type=bind,src=${SOURCE_EXPORT},dst=/workspace,readonly" \
    --entrypoint python \
    "$IMAGE" \
    -c 'from pams.config import load_config; c=load_config("/workspace/configs/experiments/pams_pose_recovery_v4a.yaml"); print(c.fingerprint, c.pose_fingerprint)'
} 2>"${LOG_DIR}/config-identity.stderr.log")"
[[ "$CONFIG_IDENTITIES" == "${CONFIG_FINGERPRINT} ${POSE_FINGERPRINT}" ]] \
  || fail "validated config identities differ from frozen values"

readonly EXTRACT_NAME="pams-v4a-extract-${SOURCE_REVISION:0:12}-${PAMS_V4A_ATTEMPT_ID,,}"
readonly AUDIT_NAME="pams-v4a-audit-${SOURCE_REVISION:0:12}-${PAMS_V4A_ATTEMPT_ID,,}"

cleanup_containers() {
  docker rm -f "$EXTRACT_NAME" "$AUDIT_NAME" >/dev/null 2>&1 || true
}
trap cleanup_containers EXIT

run_created_container() {
  local name="$1"
  local log_path="$2"
  local inspect_path="$3"
  docker inspect "$name" >"${inspect_path%.json}.pre.json"
  local status
  set +e
  docker start -a "$name" 2>&1 | tee "$log_path"
  status="${PIPESTATUS[0]}"
  set -e
  docker inspect "$name" >"$inspect_path"
  docker rm "$name" >/dev/null
  return "$status"
}

docker create \
  --name "$EXTRACT_NAME" \
  --network none \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --pids-limit 4096 \
  --memory 48g \
  --cpus 12 \
  --user 1000:1000 \
  --tmpfs /tmp:rw,nosuid,nodev,size=2g \
  --tmpfs /pams/home:rw,nosuid,nodev,uid=1000,gid=1000,size=256m \
  --tmpfs /pams/cache:rw,nosuid,nodev,uid=1000,gid=1000,size=2g \
  --env HOME=/pams/home \
  --env XDG_CACHE_HOME=/pams/cache/xdg \
  --env PYTHONPATH=/workspace/src \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env "PAMS_CONTAINER_SOURCE_REVISION=${SOURCE_REVISION}" \
  --env "PAMS_CONTAINER_IMAGE_ID=${IMAGE_ID}" \
  --mount "type=bind,src=${SOURCE_EXPORT},dst=/workspace,readonly" \
  --mount "type=bind,src=${TRAIN_VIDEO_VIEW},dst=/pams/videos,readonly" \
  --mount "type=bind,src=${TRAIN_SIDECAR},dst=/pams/protocol/inputs.json,readonly" \
  --mount "type=bind,src=${TRAIN_COMMITMENT},dst=/pams/protocol/inputs.commitment.json,readonly" \
  --mount "type=bind,src=${HEAVY_ASSET},dst=${HEAVY_CONTAINER_PATH},readonly" \
  --mount "type=bind,src=${V4_CACHE},dst=/pams/pose-cache" \
  --mount "type=bind,src=${V4_LEDGER_DIR},dst=/pams/ledgers" \
  --workdir /workspace \
  "$IMAGE" \
  python -m pams pose extract \
  /pams/protocol/inputs.json \
  /pams/pose-cache \
  --config /workspace/configs/experiments/pams_pose_recovery_v4a.yaml \
  --label-free-manifest \
  --video-root /pams/videos \
  --input-commitment /pams/protocol/inputs.commitment.json \
  --heavy-model-asset "$HEAVY_CONTAINER_PATH" \
  --failure-ledger /pams/ledgers/train337.json \
  >"${AUDIT_DIR}/extract.create-id.txt"

run_created_container \
  "$EXTRACT_NAME" \
  "${LOG_DIR}/extract.log" \
  "${AUDIT_DIR}/extract.post-run.inspect.json" \
  || fail "v4a train337 extraction failed"
[[ -f "$V4_LEDGER" ]] || fail "v4a extraction did not create its train337 ledger"

docker create \
  --name "$AUDIT_NAME" \
  --network none \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --pids-limit 4096 \
  --memory 16g \
  --cpus 4 \
  --user 1000:1000 \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --env PYTHONPATH=/workspace/src \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env "PAMS_CONTAINER_SOURCE_REVISION=${SOURCE_REVISION}" \
  --env "PAMS_CONTAINER_IMAGE_ID=${IMAGE_ID}" \
  --mount "type=bind,src=${SOURCE_EXPORT},dst=/workspace,readonly" \
  --mount "type=bind,src=${TRAIN_SIDECAR},dst=/pams/protocol/inputs.json,readonly" \
  --mount "type=bind,src=${TRAIN_COMMITMENT},dst=/pams/protocol/inputs.commitment.json,readonly" \
  --mount "type=bind,src=${REFERENCE_CACHE},dst=/pams/reference-cache,readonly" \
  --mount "type=bind,src=${REFERENCE_LEDGER},dst=/pams/reference-ledger.json,readonly" \
  --mount "type=bind,src=${V4_CACHE},dst=/pams/v4-cache,readonly" \
  --mount "type=bind,src=${V4_LEDGER},dst=/pams/v4-ledger.json,readonly" \
  --mount "type=bind,src=${AUDIT_DIR},dst=/pams/audit" \
  --workdir /workspace \
  "$IMAGE" \
  python scripts/server/audit_pose_recovery_v4a.py \
  --gate /workspace/configs/gates/pams_pose_recovery_v4a_train337.yaml \
  --train-input /pams/protocol/inputs.json \
  --train-commitment /pams/protocol/inputs.commitment.json \
  --reference-cache-dir /pams/reference-cache \
  --reference-ledger /pams/reference-ledger.json \
  --v4-cache-dir /pams/v4-cache \
  --v4-ledger /pams/v4-ledger.json \
  --output /pams/audit/paired-gate.json \
  >"${AUDIT_DIR}/audit.create-id.txt"

set +e
run_created_container \
  "$AUDIT_NAME" \
  "${LOG_DIR}/audit.log" \
  "${AUDIT_DIR}/audit.post-run.inspect.json"
readonly AUDIT_STATUS="$?"
set -e

python3 - "$RUN_ROOT" "$SOURCE_REVISION" "$IMAGE_ID" "$CONFIG_SHA256" \
  "$CONFIG_FINGERPRINT" "$POSE_FINGERPRINT" "$HEAVY_ASSET_SHA256" \
  "$AUDIT_STATUS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
gate = root / "audit/paired-gate.json"
payload = {
    "schema_version": 1,
    "artifact_type": "pams_pose_recovery_v4a_train337_run_receipt",
    "source_revision": sys.argv[2],
    "container_image_id": sys.argv[3],
    "config_file_sha256": sys.argv[4],
    "config_fingerprint": sys.argv[5],
    "pose_fingerprint": sys.argv[6],
    "heavy_model_asset_sha256": sys.argv[7],
    "temporal_resampling": "none_native_timeline",
    "scope": "count-free-train337-pose-input-recovery-only",
    "paired_gate_exit_status": int(sys.argv[8]),
    "paired_gate_sha256": (
        hashlib.sha256(gate.read_bytes()).hexdigest() if gate.is_file() else None
    ),
}
target = root / "audit/run.receipt.json"
with target.open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
PY

[[ "$AUDIT_STATUS" -eq 0 ]] || fail "paired train337 pose-recovery gate failed"
printf 'pose-recovery-v4a passed: %s\n' "$RUN_ROOT"
