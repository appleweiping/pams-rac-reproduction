#!/usr/bin/env bash
# Frozen train337-only, label-free v4b long-tail pilot and projected gate.
set -Eeuo pipefail
IFS=$'\n\t'
umask 027

fail() {
  printf 'pose-recovery-v4b-pilot: %s\n' "$*" >&2
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
: "${PAMS_V4B_EXPECTED_SOURCE_REVISION:?set the exact 40-hex source revision}"
: "${PAMS_V4B_ATTEMPT_ID:?set an immutable attempt ID such as 20260807T140000Z}"
[[ "$PAMS_V4B_EXPECTED_SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]] \
  || fail "PAMS_V4B_EXPECTED_SOURCE_REVISION must be lowercase 40-hex"
[[ "$PAMS_V4B_ATTEMPT_ID" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] \
  || fail "PAMS_V4B_ATTEMPT_ID must use YYYYMMDDTHHMMSSZ"

readonly SOURCE_REVISION="$(git -C "$REPOSITORY_ROOT" rev-parse HEAD^{commit})"
[[ "$SOURCE_REVISION" == "$PAMS_V4B_EXPECTED_SOURCE_REVISION" ]] \
  || fail "source revision differs from the preregistered revision"
[[ -z "$(git -C "$REPOSITORY_ROOT" status --porcelain=v1 --untracked-files=all)" ]] \
  || fail "source worktree must be clean, including untracked files"

readonly IMAGE='pams-rac:5e18274a5353'
readonly IMAGE_ID='sha256:0a4d42c2d9911f147a17860e4e15095746b21c4e618e4fc1b20b62dd443c5898'
readonly CONFIG_SHA256='82ea2828f5a3194a9002d791fa84b756a679a98159ed2695631224d4066dda7c'
readonly CONFIG_FINGERPRINT='e02ee5ee1d03c82d60431e5e39860aed463d9faf332169df400c34bb8c1a72c1'
readonly POSE_FINGERPRINT='9108a9b2bdcf646dd21919b4e113639e38d1d33123b6dbd2004ac96a9146c073'
readonly SIDECAR_SHA256='f95df0050df21f05bbc9b42d0154470714dde279e04cb8b00d0212bf49057d16'
readonly COMMITMENT_SHA256='85d41d2f59872e0900e9058481efbdf6bce91407d4c26bcde0a6547776454e53'
readonly V4A_LEDGER_SHA256='4cbba0d0f678cfdbd2c99752bbb55a8ceeb3aaa0b0b994bb6195b55cd0bc6018'
readonly V4A_PAIRED_GATE_SHA256='faf086277f4d0a0687a266b64edc6ae45a2ca0ab4b39cc8fc4da644350c571e5'
readonly V4A_LONG_TAIL_SHA256='d2eeb9d0f5ef75e6fc92f87357d530061c91326c8b97dfd5345a600deab1f398'
readonly HEAVY_ASSET_SHA256='59e42d71bcd44cbdbabc419f0ff76686595fd265419566bd4009ef703ea8e1fe'
readonly HEAVY_CONTAINER_PATH='/opt/conda/lib/python3.11/site-packages/mediapipe/modules/pose_landmark/pose_landmark_heavy.tflite'

readonly OFFICIAL_ROOT='/media/lenovo/data2/pams-rac/runs/official-segment-v1/15cc1ec3c1d2-20260805T063653Z'
readonly V4A_ROOT='/media/lenovo/data2/pams-rac/runs/pose-recovery-v4a/83c877007392-20260807T113023Z'
readonly TRAIN_VIDEO_VIEW="${OFFICIAL_ROOT}/video-views/train337"
readonly TRAIN_SIDECAR="${OFFICIAL_ROOT}/protocol/train.inputs.json"
readonly TRAIN_COMMITMENT="${OFFICIAL_ROOT}/protocol/train.inputs.commitment.json"
readonly V4A_LEDGER="${V4A_ROOT}/ledgers/train337.json"
readonly V4A_PAIRED_GATE="${V4A_ROOT}/audit/paired-gate.json"
readonly V4A_LONG_TAIL="${V4A_ROOT}/audit/long-tail-e644c76/v4a-long-tail-mechanism.json"
readonly HEAVY_ASSET='/media/lenovo/data2/pams-rac/assets/mediapipe/pose_landmark_heavy.tflite'
readonly RUN_PARENT='/media/lenovo/data2/pams-rac/runs/pose-recovery-v4b-pilot'
readonly RUN_ROOT="${RUN_PARENT}/${SOURCE_REVISION:0:12}-${PAMS_V4B_ATTEMPT_ID}"
readonly SOURCE_EXPORT="${RUN_ROOT}/source"
readonly CACHE_DIR="${RUN_ROOT}/pose-cache"
readonly LEDGER_DIR="${RUN_ROOT}/ledgers"
readonly V4B_LEDGER="${LEDGER_DIR}/pilot39.json"
readonly AUDIT_DIR="${RUN_ROOT}/audit"
readonly SELECTION="${AUDIT_DIR}/selection.json"
readonly PILOT_GATE="${AUDIT_DIR}/projected-gate.json"
readonly LOG_DIR="${RUN_ROOT}/logs"
readonly CONFIG_PATH="${SOURCE_EXPORT}/configs/experiments/pams_pose_recovery_v4b.yaml"
readonly GATE_PATH="${SOURCE_EXPORT}/configs/gates/pams_pose_recovery_v4b_pilot.yaml"

[[ "$(docker image inspect "$IMAGE" --format '{{.Id}}')" == "$IMAGE_ID" ]] \
  || fail "container image ID mismatch"
[[ -d "$TRAIN_VIDEO_VIEW" ]] || fail "missing frozen train337 video view"
require_sha256 "$TRAIN_SIDECAR" "$SIDECAR_SHA256" 'train337 sidecar'
require_sha256 "$TRAIN_COMMITMENT" "$COMMITMENT_SHA256" 'train337 commitment'
require_sha256 "$V4A_LEDGER" "$V4A_LEDGER_SHA256" 'v4a ledger'
require_sha256 "$V4A_PAIRED_GATE" "$V4A_PAIRED_GATE_SHA256" 'v4a paired gate'
require_sha256 "$V4A_LONG_TAIL" "$V4A_LONG_TAIL_SHA256" 'v4a long-tail audit'
require_sha256 "$HEAVY_ASSET" "$HEAVY_ASSET_SHA256" 'MediaPipe heavy asset'
[[ ! -e "$RUN_ROOT" ]] || fail "refusing to reuse run root: ${RUN_ROOT}"

mkdir -p -- "$RUN_PARENT"
mkdir -- "$RUN_ROOT" "$SOURCE_EXPORT" "$CACHE_DIR" "$LEDGER_DIR" "$AUDIT_DIR" "$LOG_DIR"
git -C "$REPOSITORY_ROOT" archive "$SOURCE_REVISION" | tar -x -C "$SOURCE_EXPORT"
chmod -R a-w -- "$SOURCE_EXPORT"
require_sha256 "$CONFIG_PATH" "$CONFIG_SHA256" 'v4b config'
[[ -f "$GATE_PATH" ]] || fail "frozen v4b pilot gate is absent from source export"

readonly CONFIG_IDENTITIES="$({
  docker run --rm \
    --network none \
    --read-only \
    --tmpfs /tmp:rw,nosuid,nodev,size=64m \
    --env PYTHONPATH=/workspace/src \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --mount "type=bind,src=${SOURCE_EXPORT},dst=/workspace,readonly" \
    --entrypoint python \
    "$IMAGE" \
    -c 'from pams.config import load_config; c=load_config("/workspace/configs/experiments/pams_pose_recovery_v4b.yaml"); print(c.fingerprint, c.pose_fingerprint)'
} 2>"${LOG_DIR}/config-identity.stderr.log")"
[[ "$CONFIG_IDENTITIES" == "${CONFIG_FINGERPRINT} ${POSE_FINGERPRINT}" ]] \
  || fail "validated config identities differ from frozen values"

readonly EXTRACT_NAME="pams-v4b-pilot-extract-${SOURCE_REVISION:0:12}-${PAMS_V4B_ATTEMPT_ID,,}"
readonly AUDIT_NAME="pams-v4b-pilot-audit-${SOURCE_REVISION:0:12}-${PAMS_V4B_ATTEMPT_ID,,}"

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
  local restore_errexit=0
  [[ $- == *e* ]] && restore_errexit=1
  set +e
  docker start -a "$name" 2>&1 | tee "$log_path"
  status="${PIPESTATUS[0]}"
  if [[ "$restore_errexit" -eq 1 ]]; then
    set -e
  fi
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
  --tmpfs /pams/cache:rw,nosuid,nodev,uid=1000,gid=1000,size=2g \
  --env XDG_CACHE_HOME=/pams/cache/xdg \
  --env PYTHONPATH=/workspace/src \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env "PAMS_CONTAINER_SOURCE_REVISION=${SOURCE_REVISION}" \
  --env "PAMS_CONTAINER_IMAGE_ID=${IMAGE_ID}" \
  --mount "type=bind,src=${SOURCE_EXPORT},dst=/workspace,readonly" \
  --mount "type=bind,src=${TRAIN_VIDEO_VIEW},dst=/pams/videos,readonly" \
  --mount "type=bind,src=${TRAIN_SIDECAR},dst=/pams/protocol/inputs.json,readonly" \
  --mount "type=bind,src=${TRAIN_COMMITMENT},dst=/pams/protocol/inputs.commitment.json,readonly" \
  --mount "type=bind,src=${V4A_LEDGER},dst=/pams/v4a-ledger.json,readonly" \
  --mount "type=bind,src=${V4A_LONG_TAIL},dst=/pams/v4a-long-tail.json,readonly" \
  --mount "type=bind,src=${HEAVY_ASSET},dst=${HEAVY_CONTAINER_PATH},readonly" \
  --mount "type=bind,src=${CACHE_DIR},dst=/pams/pose-cache" \
  --mount "type=bind,src=${LEDGER_DIR},dst=/pams/ledgers" \
  --mount "type=bind,src=${AUDIT_DIR},dst=/pams/audit" \
  --workdir /workspace \
  "$IMAGE" \
  python scripts/server/run_pose_recovery_v4b_pilot.py \
  --config /workspace/configs/experiments/pams_pose_recovery_v4b.yaml \
  --train-input /pams/protocol/inputs.json \
  --train-commitment /pams/protocol/inputs.commitment.json \
  --video-root /pams/videos \
  --heavy-model-asset "$HEAVY_CONTAINER_PATH" \
  --v4a-ledger /pams/v4a-ledger.json \
  --v4a-long-tail-audit /pams/v4a-long-tail.json \
  --cache-dir /pams/pose-cache \
  --ledger /pams/ledgers/pilot39.json \
  --selection /pams/audit/selection.json \
  >"${AUDIT_DIR}/extract.create-id.txt"

run_created_container \
  "$EXTRACT_NAME" \
  "${LOG_DIR}/extract.log" \
  "${AUDIT_DIR}/extract.post-run.inspect.json" \
  || fail "v4b long-tail pilot extraction failed"
[[ -f "$V4B_LEDGER" ]] || fail "pilot extraction did not create its ledger"

docker create \
  --name "$AUDIT_NAME" \
  --network none \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --pids-limit 1024 \
  --memory 8g \
  --cpus 4 \
  --user 1000:1000 \
  --tmpfs /tmp:rw,nosuid,nodev,size=512m \
  --env PYTHONPATH=/workspace/src \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env "PAMS_CONTAINER_SOURCE_REVISION=${SOURCE_REVISION}" \
  --env "PAMS_CONTAINER_IMAGE_ID=${IMAGE_ID}" \
  --mount "type=bind,src=${SOURCE_EXPORT},dst=/workspace,readonly" \
  --mount "type=bind,src=${SELECTION},dst=/pams/selection.json,readonly" \
  --mount "type=bind,src=${V4A_LEDGER},dst=/pams/v4a-ledger.json,readonly" \
  --mount "type=bind,src=${V4A_PAIRED_GATE},dst=/pams/v4a-paired-gate.json,readonly" \
  --mount "type=bind,src=${V4B_LEDGER},dst=/pams/v4b-ledger.json,readonly" \
  --mount "type=bind,src=${AUDIT_DIR},dst=/pams/audit" \
  --workdir /workspace \
  "$IMAGE" \
  python scripts/server/audit_pose_recovery_v4b_pilot.py \
  --gate /workspace/configs/gates/pams_pose_recovery_v4b_pilot.yaml \
  --selection /pams/selection.json \
  --v4a-ledger /pams/v4a-ledger.json \
  --v4a-paired-gate /pams/v4a-paired-gate.json \
  --v4b-ledger /pams/v4b-ledger.json \
  --output /pams/audit/projected-gate.json \
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
selection = root / "audit/selection.json"
ledger = root / "ledgers/pilot39.json"
gate = root / "audit/projected-gate.json"
payload = {
    "schema_version": 1,
    "artifact_type": "pams_pose_recovery_v4b_train337_long_tail_pilot_receipt",
    "source_revision": sys.argv[2],
    "container_image_id": sys.argv[3],
    "config_file_sha256": sys.argv[4],
    "config_fingerprint": sys.argv[5],
    "pose_fingerprint": sys.argv[6],
    "heavy_model_asset_sha256": sys.argv[7],
    "scope": "label-free-train337-long-tail-pilot-only",
    "pilot_gate_exit_status": int(sys.argv[8]),
    "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
    "ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
    "projected_gate_sha256": (
        hashlib.sha256(gate.read_bytes()).hexdigest() if gate.is_file() else None
    ),
}
target = root / "audit/run.receipt.json"
with target.open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
PY

[[ "$AUDIT_STATUS" -eq 0 ]] || fail "projected fixed train337 gate failed"
printf 'pose-recovery-v4b pilot passed: %s\n' "$RUN_ROOT"
