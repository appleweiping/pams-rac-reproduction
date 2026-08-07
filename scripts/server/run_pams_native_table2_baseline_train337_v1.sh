#!/usr/bin/env bash
# Formal train337-only runner for an independently inferred Table-2 proxy.
#
# This launcher trains one encoder to epoch 11, runs a read-only train337
# mechanism gate, and resumes the same immutable lineage to epoch 150 only
# after explicit authorization. Dev84/test105 identity sidecars are mounted
# only because the label-free training CLI requires their frozen protocol
# identities. No dev/test pose cache, target, prediction, scoring, SSHead, or
# separately trained period head is authorized here.
set -Eeuo pipefail
IFS=$'\n\t'
umask 027

fail() {
  printf 'native-table2-proxy-v1: %s\n' "$*" >&2
  exit 2
}

[[ -z "${PYTHONOPTIMIZE:-}" ]] \
  || fail 'PYTHONOPTIMIZE must be unset or empty for fail-closed validation'
unset PYTHONOPTIMIZE

sha256_file() {
  sha256sum -- "$1" | awk '{print $1}'
}

require_sha256() {
  local path="$1"
  local expected="$2"
  local role="$3"
  [[ -f "$path" ]] || fail "missing ${role}: ${path}"
  local actual
  actual="$(sha256_file "$path")"
  [[ "$actual" == "$expected" ]] \
    || fail "${role} SHA-256 mismatch: expected=${expected} actual=${actual}"
}

readonly ROOT_INPUT="${PAMS_ROOT:-/media/lenovo/data2/pams-rac}"
readonly SOURCE_REVISION="${PAMS_SOURCE_REVISION:?PAMS_SOURCE_REVISION is required}"
readonly SOURCE_CHECKOUT_INPUT="${PAMS_SOURCE_CHECKOUT:?PAMS_SOURCE_CHECKOUT is required}"
readonly IMAGE_ID="${PAMS_IMAGE_ID:?PAMS_IMAGE_ID is required}"
readonly ENVIRONMENT_SHA256="${PAMS_ENVIRONMENT_SHA256:?PAMS_ENVIRONMENT_SHA256 is required}"
readonly CONFIG_SHA256="${PAMS_CONFIG_SHA256:?PAMS_CONFIG_SHA256 is required}"
readonly CONFIG_FINGERPRINT="${PAMS_CONFIG_FINGERPRINT:?PAMS_CONFIG_FINGERPRINT is required}"
readonly POSE_RECOVERY_RUN_ROOT_INPUT="${PAMS_POSE_RECOVERY_RUN_ROOT:?PAMS_POSE_RECOVERY_RUN_ROOT is required}"
readonly POSE_RECOVERY_VERSION="${PAMS_POSE_RECOVERY_VERSION:?PAMS_POSE_RECOVERY_VERSION is required}"
readonly POSE_RECOVERY_CONFIG_INPUT="${PAMS_POSE_RECOVERY_CONFIG_PATH:?PAMS_POSE_RECOVERY_CONFIG_PATH is required}"
readonly POSE_RECOVERY_AUTHORIZATION_SHA256="${PAMS_POSE_RECOVERY_AUTHORIZATION_SHA256:?PAMS_POSE_RECOVERY_AUTHORIZATION_SHA256 is required}"
readonly CANDIDATE_ID="${PAMS_CANDIDATE_ID:?PAMS_CANDIDATE_ID is required}"
readonly PRIOR_A_REJECTION_ARTIFACT_INPUT="${PAMS_PRIOR_A_REJECTION_ARTIFACT:-}"
readonly PRIOR_A_REJECTION_RECEIPT_INPUT="${PAMS_PRIOR_A_REJECTION_RECEIPT:-}"
readonly PRIOR_B_REJECTION_ARTIFACT_INPUT="${PAMS_PRIOR_B_REJECTION_ARTIFACT:-}"
readonly PRIOR_B_REJECTION_RECEIPT_INPUT="${PAMS_PRIOR_B_REJECTION_RECEIPT:-}"
readonly ATTEMPT_ID="${PAMS_ATTEMPT_ID:?PAMS_ATTEMPT_ID is required}"
readonly GPU_DEVICE="${PAMS_GPU_DEVICE:-0}"

readonly CONFIG_A_RELATIVE='configs/experiments/pams_native_table2_baseline_proxy_v1.yaml'
readonly CONFIG_B_RELATIVE='configs/experiments/pams_native_table2_baseline_proxy_b_w16_s2.yaml'
readonly CONFIG_C_RELATIVE='configs/experiments/pams_native_table2_baseline_proxy_c_w24_s4.yaml'
case "$CANDIDATE_ID" in
  A)
    CONFIG_RELATIVE="$CONFIG_A_RELATIVE"
    CANDIDATE_WINDOW=16
    CANDIDATE_STRIDE=4
    CANDIDATE_PRIOR_IDS_JSON='[]'
    [[ -z "$PRIOR_A_REJECTION_ARTIFACT_INPUT" \
      && -z "$PRIOR_A_REJECTION_RECEIPT_INPUT" \
      && -z "$PRIOR_B_REJECTION_ARTIFACT_INPUT" \
      && -z "$PRIOR_B_REJECTION_RECEIPT_INPUT" ]] \
      || fail 'candidate A forbids predecessor rejection inputs'
    ;;
  B)
    CONFIG_RELATIVE="$CONFIG_B_RELATIVE"
    CANDIDATE_WINDOW=16
    CANDIDATE_STRIDE=2
    CANDIDATE_PRIOR_IDS_JSON='["A"]'
    [[ -n "$PRIOR_A_REJECTION_ARTIFACT_INPUT" \
      && -n "$PRIOR_A_REJECTION_RECEIPT_INPUT" \
      && -z "$PRIOR_B_REJECTION_ARTIFACT_INPUT" \
      && -z "$PRIOR_B_REJECTION_RECEIPT_INPUT" ]] \
      || fail 'candidate B requires exactly the candidate-A rejection pair'
    ;;
  C)
    CONFIG_RELATIVE="$CONFIG_C_RELATIVE"
    CANDIDATE_WINDOW=24
    CANDIDATE_STRIDE=4
    CANDIDATE_PRIOR_IDS_JSON='["A","B"]'
    [[ -n "$PRIOR_A_REJECTION_ARTIFACT_INPUT" \
      && -n "$PRIOR_A_REJECTION_RECEIPT_INPUT" \
      && -n "$PRIOR_B_REJECTION_ARTIFACT_INPUT" \
      && -n "$PRIOR_B_REJECTION_RECEIPT_INPUT" ]] \
      || fail 'candidate C requires the ordered A and B rejection pairs'
    ;;
  *)
    fail 'PAMS_CANDIDATE_ID must be exactly A, B, or C'
    ;;
esac
readonly CONFIG_RELATIVE CANDIDATE_WINDOW CANDIDATE_STRIDE CANDIDATE_PRIOR_IDS_JSON
readonly VALIDATOR_RELATIVE='scripts/server/validate_pams_native_baseline_inputs.py'
readonly RUNNER_RELATIVE='scripts/server/run_pams_native_table2_baseline_train337_v1.sh'
readonly EPOCH11_GATE_RELATIVE='scripts/server/run_pams_native_epoch11_train_gate.py'
readonly EPOCH11_GATE_SPEC_RELATIVE='configs/gates/pams_native_epoch11_train_gate_v1.yaml'
readonly TERMINAL_GATE_RELATIVE='scripts/server/run_pams_native_terminal_readout_gate.py'
readonly TERMINAL_GATE_SPEC_RELATIVE='configs/gates/pams_native_terminal_readout_gate_v1.yaml'
readonly LAUNCH_AUTHORIZATION_RELATIVE='scripts/server/prepare_pams_native_candidate_launch_authorization.py'
readonly TRAIN_INPUT_SHA256='f95df0050df21f05bbc9b42d0154470714dde279e04cb8b00d0212bf49057d16'
readonly TRAIN_COMMIT_SHA256='85d41d2f59872e0900e9058481efbdf6bce91407d4c26bcde0a6547776454e53'
readonly DEV_INPUT_SHA256='74b6628679c3d4c9b48f82ef8cf7e3a678e0a8298a5cb245512af9912e4337ba'
readonly DEV_COMMIT_SHA256='15083106499f6917c8c7c91f6e3985e4938eedbe370e1f40c281dd00884bcef0'
readonly TEST_ID_INPUT_SHA256='5de62008db8adc54e6d1ce1df55e25fa0c4a43815ccc1580a010bc1070abafff'
readonly TEST_ID_COMMIT_SHA256='634eb578de8b9d96aa33c128ff2cff8e4e4759304297d331f924ae2b354be6c0'
readonly CLASSIFICATION='independently_inferred_proxy_not_author_table2_baseline'

for digest_name in \
  ENVIRONMENT_SHA256 \
  CONFIG_SHA256 \
  CONFIG_FINGERPRINT \
  POSE_RECOVERY_AUTHORIZATION_SHA256; do
  digest="${!digest_name}"
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] \
    || fail "${digest_name} must be a lowercase SHA-256"
done
for revision_name in SOURCE_REVISION; do
  revision="${!revision_name}"
  [[ "$revision" =~ ^[0-9a-f]{40}$ ]] \
    || fail "${revision_name} must be a full lowercase Git SHA"
done
[[ "$IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || fail 'PAMS_IMAGE_ID must be an immutable image ID'
[[ "$ATTEMPT_ID" =~ ^[a-z0-9][a-z0-9._-]{0,39}$ ]] \
  || fail 'PAMS_ATTEMPT_ID must be a lowercase safe slug of at most 40 characters'
[[ "$ATTEMPT_ID" != *'..'* ]] || fail "PAMS_ATTEMPT_ID must not contain '..'"
[[ "$GPU_DEVICE" =~ ^[0-9]+$ ]] \
  || fail 'PAMS_GPU_DEVICE must be a non-negative integer'
[[ "$POSE_RECOVERY_VERSION" =~ ^[a-z0-9][a-z0-9._-]{0,63}$ ]] \
  || fail 'PAMS_POSE_RECOVERY_VERSION must be a lowercase safe slug'
case "$POSE_RECOVERY_VERSION" in
  v4a|official-segment-heavy-missing-retry-full-timeline-v4a)
    fail "pose-recovery version is formally rejected: ${POSE_RECOVERY_VERSION}"
    ;;
esac
for mount_path in \
  "$ROOT_INPUT" \
  "$SOURCE_CHECKOUT_INPUT" \
  "$POSE_RECOVERY_RUN_ROOT_INPUT" \
  "$POSE_RECOVERY_CONFIG_INPUT"; do
  [[ "$mount_path" == /* ]] || fail "host path must be absolute: ${mount_path}"
  [[ "$mount_path" != *','* ]] || fail "Docker bind path contains a comma"
done

readonly ROOT="$(realpath -e -- "$ROOT_INPUT")"
readonly SOURCE_CHECKOUT="$(realpath -e -- "$SOURCE_CHECKOUT_INPUT")"
readonly POSE_RECOVERY_RUN_ROOT="$(realpath -e -- "$POSE_RECOVERY_RUN_ROOT_INPUT")"
readonly POSE_RECOVERY_CONFIG="$(realpath -e -- "$POSE_RECOVERY_CONFIG_INPUT")"
for prior_input in \
  "$PRIOR_A_REJECTION_ARTIFACT_INPUT" \
  "$PRIOR_A_REJECTION_RECEIPT_INPUT" \
  "$PRIOR_B_REJECTION_ARTIFACT_INPUT" \
  "$PRIOR_B_REJECTION_RECEIPT_INPUT"; do
  if [[ -n "$prior_input" ]]; then
    [[ "$prior_input" == /* && "$prior_input" != *','* ]] \
      || fail "prior rejection path must be absolute and comma-free: ${prior_input}"
    [[ -f "$prior_input" && ! -L "$prior_input" ]] \
      || fail "prior rejection input must be a regular non-symlink file: ${prior_input}"
  fi
done
readonly PRIOR_A_REJECTION_ARTIFACT="$({
  [[ -z "$PRIOR_A_REJECTION_ARTIFACT_INPUT" ]] \
    || realpath -e -- "$PRIOR_A_REJECTION_ARTIFACT_INPUT"
})"
readonly PRIOR_A_REJECTION_RECEIPT="$({
  [[ -z "$PRIOR_A_REJECTION_RECEIPT_INPUT" ]] \
    || realpath -e -- "$PRIOR_A_REJECTION_RECEIPT_INPUT"
})"
readonly PRIOR_B_REJECTION_ARTIFACT="$({
  [[ -z "$PRIOR_B_REJECTION_ARTIFACT_INPUT" ]] \
    || realpath -e -- "$PRIOR_B_REJECTION_ARTIFACT_INPUT"
})"
readonly PRIOR_B_REJECTION_RECEIPT="$({
  [[ -z "$PRIOR_B_REJECTION_RECEIPT_INPUT" ]] \
    || realpath -e -- "$PRIOR_B_REJECTION_RECEIPT_INPUT"
})"
readonly OFFICIAL_ROOT="${ROOT}/runs/official-segment-v1/15cc1ec3c1d2-20260805T063653Z"
readonly RUN_PARENT="${ROOT}/runs/pams-native-table2-baseline-v1"
readonly RUN_ROOT="${RUN_PARENT}/${ATTEMPT_ID}"
readonly SOURCE_VIEW="${RUN_ROOT}/source"
readonly SOURCE_RECEIPT="${RUN_ROOT}/source-export.receipt.json"
readonly INPUT_ROOT="${RUN_ROOT}/inputs"
readonly CONFIG_HOST="${INPUT_ROOT}/pams_native_table2_baseline_proxy_v1.yaml"
readonly EPOCH11_ENCODER_STAGE="${RUN_ROOT}/stages/encoder-epoch11"
readonly LAUNCH_AUTHORIZATION_STAGE="${RUN_ROOT}/stages/launch-authorization"
readonly EPOCH11_GATE_STAGE="${RUN_ROOT}/stages/epoch11-gate"
readonly FINAL_ENCODER_STAGE="${RUN_ROOT}/stages/encoder-final"
readonly EPOCH11_ENCODER_RUN="${EPOCH11_ENCODER_STAGE}/run"
readonly EPOCH11_ENCODER_CHECKPOINT="${EPOCH11_ENCODER_RUN}/encoder.pt"
readonly EPOCH11_ENCODER_PROGRESS="${EPOCH11_ENCODER_RUN}/logs/encoder.jsonl"
readonly EPOCH11_POSE_SNAPSHOT="${EPOCH11_ENCODER_RUN}/inputs/training-pose-cache-snapshot.json"
readonly EPOCH11_GATE_ARTIFACT="${EPOCH11_GATE_STAGE}/gate.json"
readonly EPOCH11_GATE_RECEIPT="${EPOCH11_GATE_STAGE}/gate.json.receipt.json"
readonly LAUNCH_POSE_SNAPSHOT="${INPUT_ROOT}/launch-training-pose-cache-snapshot.json"
readonly LAUNCH_AUTHORIZATION_ARTIFACT="${LAUNCH_AUTHORIZATION_STAGE}/authorization.json"
readonly LAUNCH_AUTHORIZATION_RECEIPT="${LAUNCH_AUTHORIZATION_ARTIFACT}.receipt.json"
readonly FINAL_ENCODER_RUN="${FINAL_ENCODER_STAGE}/run"
readonly FINAL_ENCODER_CHECKPOINT="${FINAL_ENCODER_RUN}/encoder.pt"
readonly FINAL_ENCODER_PROGRESS="${FINAL_ENCODER_RUN}/logs/encoder.jsonl"
readonly FINAL_ENCODER_POSE_SNAPSHOT="${FINAL_ENCODER_RUN}/inputs/training-pose-cache-snapshot.json"
readonly AUDIT_ROOT="${RUN_ROOT}/audit"
readonly LOG_ROOT="${RUN_ROOT}/logs"
readonly LOCK_ROOT="${ROOT}/.pams-gpu-locks"
readonly LOCK_PATH="${LOCK_ROOT}/gpu${GPU_DEVICE}.lock"

readonly TRAIN_INPUT="${OFFICIAL_ROOT}/protocol/train.inputs.json"
readonly TRAIN_COMMIT="${OFFICIAL_ROOT}/protocol/train.inputs.commitment.json"
readonly DEV_INPUT="${OFFICIAL_ROOT}/protocol/dev.inputs.json"
readonly DEV_COMMIT="${OFFICIAL_ROOT}/protocol/dev.inputs.commitment.json"
readonly TEST_ID_INPUT="${OFFICIAL_ROOT}/protocol/test-identity.inputs.json"
readonly TEST_ID_COMMIT="${OFFICIAL_ROOT}/protocol/test-identity.inputs.commitment.json"
readonly POSE_RECOVERY_CACHE="${POSE_RECOVERY_RUN_ROOT}/pose-cache"
readonly POSE_RECOVERY_LEDGER="${POSE_RECOVERY_RUN_ROOT}/ledgers/train337.json"
readonly POSE_RECOVERY_AUTHORIZATION="${POSE_RECOVERY_RUN_ROOT}/audit/native-baseline-authorization.json"
readonly POSE_RECOVERY_PAIRED_GATE="${POSE_RECOVERY_RUN_ROOT}/audit/paired-gate.json"
readonly POSE_RECOVERY_RUN_RECEIPT="${POSE_RECOVERY_RUN_ROOT}/audit/run.receipt.json"

readonly PREFLIGHT_NAME="pams-native-proxy-preflight-${ATTEMPT_ID}"
readonly LAUNCH_AUTHORIZATION_NAME="pams-native-proxy-launch-${ATTEMPT_ID}"
readonly EPOCH11_ENCODER_NAME="pams-native-proxy-epoch11-${ATTEMPT_ID}"
readonly EPOCH11_GATE_NAME="pams-native-proxy-gate-${ATTEMPT_ID}"
readonly FINAL_ENCODER_NAME="pams-native-proxy-final-${ATTEMPT_ID}"

RUN_RESERVED=0
CURRENT_STAGE='host-preflight'
ACTIVE_CONTAINER=''

write_status() {
  local state="$1"
  local stage="$2"
  local exit_code="$3"
  local temporary="${RUN_ROOT}/status.json.tmp"
  printf \
    '{"schema_version":1,"status":"%s","stage":"%s","exit_code":%s,"updated_utc":"%s"}\n' \
    "$state" "$stage" "$exit_code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$temporary"
  mv -- "$temporary" "${RUN_ROOT}/status.json"
}

on_exit() {
  local exit_code="$?"
  trap - EXIT HUP INT TERM
  set +e
  if [[ -n "$ACTIVE_CONTAINER" ]]; then
    docker inspect "$ACTIVE_CONTAINER" \
      > "${AUDIT_ROOT}/${ACTIVE_CONTAINER}.error.inspect.json" 2>/dev/null
    if [[ "$(docker inspect "$ACTIVE_CONTAINER" --format '{{.State.Running}}' 2>/dev/null)" == 'true' ]]; then
      docker stop --time 10 "$ACTIVE_CONTAINER" >/dev/null 2>&1
    fi
  fi
  if [[ "$exit_code" -ne 0 && "$RUN_RESERVED" -eq 1 ]]; then
    write_status 'failed' "$CURRENT_STAGE" "$exit_code" >/dev/null 2>&1
  fi
  exit "$exit_code"
}

trap on_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

single_completion_receipt() {
  local output_root="$1"
  local -a matches=()
  mapfile -t matches < <(
    find "${output_root}/run/manifests" -maxdepth 1 -type f \
      -name '*.completed.json' -print | sort
  )
  [[ "${#matches[@]}" -eq 1 ]] \
    || fail "expected one encoder completion receipt, got ${#matches[@]}"
  printf '%s\n' "${matches[0]}"
}

run_created_container() {
  local container_name="$1"
  local stage="$2"
  local log_path="${LOG_ROOT}/${container_name}.log"
  CURRENT_STAGE="$stage"
  write_status 'running' "$stage" 'null'
  ACTIVE_CONTAINER="$container_name"
  set +e
  docker start --attach "$container_name" > "$log_path" 2>&1
  local attach_exit="$?"
  set -e
  local running
  local container_exit
  running="$(docker inspect "$container_name" --format '{{.State.Running}}')"
  if [[ "$running" == 'true' ]]; then
    docker stop --time 10 "$container_name" >/dev/null
  fi
  container_exit="$(docker inspect "$container_name" --format '{{.State.ExitCode}}')"
  docker inspect "$container_name" \
    > "${AUDIT_ROOT}/${container_name}.post-run.inspect.json"
  ACTIVE_CONTAINER=''
  printf '%s\n' "$container_exit" \
    > "${AUDIT_ROOT}/${container_name}.exit-code.txt"
  if [[ "$attach_exit" -ne 0 || "$running" == 'true' || "$container_exit" -ne 0 ]]; then
    fail "container ${container_name} failed: attach=${attach_exit} running=${running} exit=${container_exit}"
  fi
}

GATE_CONTAINER_EXIT=''
run_gate_container() {
  local container_name="$1"
  local log_path="${LOG_ROOT}/${container_name}.log"
  CURRENT_STAGE='epoch11-gate'
  write_status 'running' "$CURRENT_STAGE" 'null'
  ACTIVE_CONTAINER="$container_name"
  set +e
  docker start --attach "$container_name" > "$log_path" 2>&1
  local attach_exit="$?"
  set -e
  local running
  local container_exit
  running="$(docker inspect "$container_name" --format '{{.State.Running}}')"
  if [[ "$running" == 'true' ]]; then
    docker stop --time 10 "$container_name" >/dev/null
  fi
  container_exit="$(docker inspect "$container_name" --format '{{.State.ExitCode}}')"
  docker inspect "$container_name" \
    > "${AUDIT_ROOT}/${container_name}.post-run.inspect.json"
  ACTIVE_CONTAINER=''
  printf '%s\n' "$container_exit" \
    > "${AUDIT_ROOT}/${container_name}.exit-code.txt"
  [[ "$running" == 'false' && "$attach_exit" -eq "$container_exit" ]] \
    || fail "gate container state mismatch: attach=${attach_exit} running=${running} exit=${container_exit}"
  [[ "$container_exit" -eq 0 || "$container_exit" -eq 3 ]] \
    || fail "gate container contract failure: exit=${container_exit}"
  GATE_CONTAINER_EXIT="$container_exit"
}

verify_container() {
  local container_name="$1"
  local stage="$2"
  local inspect_path="${AUDIT_ROOT}/${container_name}.inspect.json"
  docker inspect "$container_name" > "$inspect_path"
  VERIFY_STAGE="$stage" \
  VERIFY_INSPECT="$inspect_path" \
  VERIFY_SOURCE_VIEW="$SOURCE_VIEW" \
  VERIFY_SOURCE_RECEIPT="$SOURCE_RECEIPT" \
  VERIFY_CONFIG_HOST="$CONFIG_HOST" \
  VERIFY_OFFICIAL_ROOT="$OFFICIAL_ROOT" \
  VERIFY_POSE_RECOVERY_ROOT="$POSE_RECOVERY_RUN_ROOT" \
  VERIFY_POSE_RECOVERY_CONFIG="$POSE_RECOVERY_CONFIG" \
  VERIFY_EPOCH11_ENCODER_STAGE="$EPOCH11_ENCODER_STAGE" \
  VERIFY_LAUNCH_AUTHORIZATION_STAGE="$LAUNCH_AUTHORIZATION_STAGE" \
  VERIFY_LAUNCH_POSE_SNAPSHOT="$LAUNCH_POSE_SNAPSHOT" \
  VERIFY_LAUNCH_AUTHORIZATION="$LAUNCH_AUTHORIZATION_ARTIFACT" \
  VERIFY_LAUNCH_RECEIPT="$LAUNCH_AUTHORIZATION_RECEIPT" \
  VERIFY_EPOCH11_GATE_STAGE="$EPOCH11_GATE_STAGE" \
  VERIFY_FINAL_ENCODER_STAGE="$FINAL_ENCODER_STAGE" \
  VERIFY_EPOCH11_CHECKPOINT="$EPOCH11_ENCODER_CHECKPOINT" \
  VERIFY_EPOCH11_PROGRESS="$EPOCH11_ENCODER_PROGRESS" \
  VERIFY_EPOCH11_POSE_SNAPSHOT="$EPOCH11_POSE_SNAPSHOT" \
  VERIFY_EPOCH11_COMPLETION_RECEIPT="${EPOCH11_ENCODER_RECEIPT:-}" \
  VERIFY_AUDIT_ROOT="$AUDIT_ROOT" \
  VERIFY_IMAGE_ID="$IMAGE_ID" \
  VERIFY_SOURCE_REVISION="$SOURCE_REVISION" \
  VERIFY_ENVIRONMENT_SHA256="$ENVIRONMENT_SHA256" \
  VERIFY_SOURCE_RECEIPT_SHA256="$SOURCE_RECEIPT_SHA256" \
  VERIFY_GPU_DEVICE="$GPU_DEVICE" \
  VERIFY_CANDIDATE_ID="$CANDIDATE_ID" \
  VERIFY_PRIOR_A_ARTIFACT="$PRIOR_A_REJECTION_ARTIFACT" \
  VERIFY_PRIOR_A_RECEIPT="$PRIOR_A_REJECTION_RECEIPT" \
  VERIFY_PRIOR_B_ARTIFACT="$PRIOR_B_REJECTION_ARTIFACT" \
  VERIFY_PRIOR_B_RECEIPT="$PRIOR_B_REJECTION_RECEIPT" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

stage = os.environ["VERIFY_STAGE"]
item = json.loads(Path(os.environ["VERIFY_INSPECT"]).read_text(encoding="utf-8"))[0]
config = item["Config"]
host = item["HostConfig"]
mounts = {
    row["Destination"]: (row["Source"], bool(row["RW"]))
    for row in item["Mounts"]
}
require(config["Image"] == os.environ["VERIFY_IMAGE_ID"], "image mismatch")
require(config["User"] == "1000:1000", "container user mismatch")
require(config["WorkingDir"] == "/workspace", "working directory mismatch")
require(host["NetworkMode"] == "none", "network must be disabled")
require(host["ReadonlyRootfs"] is True, "root filesystem must be read-only")
require("ALL" in (host.get("CapDrop") or []), "all capabilities must be dropped")
require(
    "no-new-privileges:true" in (host.get("SecurityOpt") or []),
    "no-new-privileges is missing",
)
require(int(host["PidsLimit"]) == 4096, "PID limit mismatch")
require(int(host["Memory"]) == 96 * 1024**3, "memory limit mismatch")
require(
    set(host.get("Tmpfs") or {}) == {
        "/tmp", "/pams/tmp", "/pams/cache", "/pams/home"
    },
    "tmpfs mount set mismatch",
)

source = {
    "/workspace": (os.environ["VERIFY_SOURCE_VIEW"], False),
    "/pams/source-export-receipt.json": (
        os.environ["VERIFY_SOURCE_RECEIPT"], False
    ),
}
protocol_root = os.environ["VERIFY_OFFICIAL_ROOT"] + "/protocol"
protocol = {
    "/pams/input/config.yaml": (os.environ["VERIFY_CONFIG_HOST"], False),
    "/pams/protocol/train.inputs.json": (
        protocol_root + "/train.inputs.json", False
    ),
    "/pams/protocol/train.inputs.commitment.json": (
        protocol_root + "/train.inputs.commitment.json", False
    ),
    "/pams/protocol/dev.inputs.json": (
        protocol_root + "/dev.inputs.json", False
    ),
    "/pams/protocol/dev.inputs.commitment.json": (
        protocol_root + "/dev.inputs.commitment.json", False
    ),
    "/pams/protocol/test-identity.inputs.json": (
        protocol_root + "/test-identity.inputs.json", False
    ),
    "/pams/protocol/test-identity.inputs.commitment.json": (
        protocol_root + "/test-identity.inputs.commitment.json", False
    ),
}
pose_recovery_root = os.environ["VERIFY_POSE_RECOVERY_ROOT"]
launch_inputs = {
    "/pams/launch/authorization.json": (
        os.environ["VERIFY_LAUNCH_AUTHORIZATION"], False
    ),
    "/pams/launch/authorization.json.receipt.json": (
        os.environ["VERIFY_LAUNCH_RECEIPT"], False
    ),
}
candidate_id = os.environ["VERIFY_CANDIDATE_ID"]
prior_mounts = {}
if candidate_id in {"B", "C"}:
    prior_mounts.update({
        "/pams/prior/A/rejection.json": (
            os.environ["VERIFY_PRIOR_A_ARTIFACT"], False
        ),
        "/pams/prior/A/rejection.json.receipt.json": (
            os.environ["VERIFY_PRIOR_A_RECEIPT"], False
        ),
    })
if candidate_id == "C":
    prior_mounts.update({
        "/pams/prior/B/rejection.json": (
            os.environ["VERIFY_PRIOR_B_ARTIFACT"], False
        ),
        "/pams/prior/B/rejection.json.receipt.json": (
            os.environ["VERIFY_PRIOR_B_RECEIPT"], False
        ),
    })
if stage == "preflight":
    expected = {
        **source,
        **protocol,
        "/pams/pose-recovery/config.yaml": (
            os.environ["VERIFY_POSE_RECOVERY_CONFIG"], False
        ),
        "/pams/pose-recovery/native-baseline-authorization.json": (
            pose_recovery_root + "/audit/native-baseline-authorization.json", False
        ),
        "/pams/pose-recovery/paired-gate.json": (
            pose_recovery_root + "/audit/paired-gate.json", False
        ),
        "/pams/pose-recovery/run.receipt.json": (
            pose_recovery_root + "/audit/run.receipt.json", False
        ),
        "/pams/pose-recovery/train337.ledger.json": (
            pose_recovery_root + "/ledgers/train337.json", False
        ),
        "/pams/pose-cache": (pose_recovery_root + "/pose-cache", False),
        "/pams/output": (os.environ["VERIFY_AUDIT_ROOT"], True),
    }
elif stage == "launch-authorization":
    expected = {
        **source,
        **prior_mounts,
        "/pams/input/launch-training-pose-cache-snapshot.json": (
            os.environ["VERIFY_LAUNCH_POSE_SNAPSHOT"], False
        ),
        "/pams/output": (
            os.environ["VERIFY_LAUNCH_AUTHORIZATION_STAGE"], True
        ),
    }
elif stage == "encoder-epoch11":
    expected = {
        **source,
        **protocol,
        **launch_inputs,
        "/pams/pose-cache": (pose_recovery_root + "/pose-cache", False),
        "/pams/output": (os.environ["VERIFY_EPOCH11_ENCODER_STAGE"], True),
    }
elif stage == "epoch11-gate":
    expected = {
        **source,
        "/pams/pose-cache": (pose_recovery_root + "/pose-cache", False),
        "/pams/epoch11/encoder.pt": (
            os.environ["VERIFY_EPOCH11_CHECKPOINT"], False
        ),
        "/pams/epoch11/encoder.jsonl": (
            os.environ["VERIFY_EPOCH11_PROGRESS"], False
        ),
        "/pams/epoch11/training-pose-cache-snapshot.json": (
            os.environ["VERIFY_EPOCH11_POSE_SNAPSHOT"], False
        ),
        "/pams/epoch11/completion.receipt.json": (
            os.environ["VERIFY_EPOCH11_COMPLETION_RECEIPT"], False
        ),
        "/pams/output": (os.environ["VERIFY_EPOCH11_GATE_STAGE"], True),
    }
elif stage == "encoder-final":
    expected = {
        **source,
        **protocol,
        **launch_inputs,
        "/pams/pose-cache": (pose_recovery_root + "/pose-cache", False),
        "/pams/resume/encoder.pt": (
            os.environ["VERIFY_EPOCH11_CHECKPOINT"], False
        ),
        "/pams/resume/encoder.jsonl": (
            os.environ["VERIFY_EPOCH11_PROGRESS"], False
        ),
        "/pams/output": (os.environ["VERIFY_FINAL_ENCODER_STAGE"], True),
    }
else:
    raise RuntimeError(f"unexpected container verification stage: {stage}")
require(mounts == expected, f"mount set mismatch for {stage}: {mounts!r}")

argv = config.get("Cmd") or []
command = "\0".join(argv)
all_text = "\0".join([command, *mounts, *(row[0] for row in mounts.values())])
for forbidden in (
    ".targets", "/dev-pose", "/test-pose", "--include-dev",
    "train sshead", "evaluate", "dev-predict", "dev-score",
):
    require(forbidden not in all_text, f"forbidden container token: {forbidden}")
def option_values(option):
    return [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == option]

if stage == "launch-authorization":
    require(
        "prepare_pams_native_candidate_launch_authorization.py" in command,
        "launch authorization command mismatch",
    )
    require("train\0encoder" not in command, "launch authorization must not train")
    expected_config = {
        "A": "/workspace/configs/experiments/pams_native_table2_baseline_proxy_v1.yaml",
        "B": "/workspace/configs/experiments/pams_native_table2_baseline_proxy_b_w16_s2.yaml",
        "C": "/workspace/configs/experiments/pams_native_table2_baseline_proxy_c_w24_s4.yaml",
    }[candidate_id]
    expected_options = {
        "--source-receipt": ["/pams/source-export-receipt.json"],
        "--config": [expected_config],
        "--gate-specification": [
            "/workspace/configs/gates/pams_native_terminal_readout_gate_v1.yaml"
        ],
        "--pose-snapshot": [
            "/pams/input/launch-training-pose-cache-snapshot.json"
        ],
        "--candidate-id": [candidate_id],
        "--output": ["/pams/output/authorization.json"],
    }
    for option, expected_values in expected_options.items():
        require(option_values(option) == expected_values, f"launch option mismatch: {option}")
    expected_prior_artifacts = []
    expected_prior_receipts = []
    if candidate_id in {"B", "C"}:
        expected_prior_artifacts.append("/pams/prior/A/rejection.json")
        expected_prior_receipts.append("/pams/prior/A/rejection.json.receipt.json")
    if candidate_id == "C":
        expected_prior_artifacts.append("/pams/prior/B/rejection.json")
        expected_prior_receipts.append("/pams/prior/B/rejection.json.receipt.json")
    require(
        option_values("--prior-rejection-artifact") == expected_prior_artifacts,
        "launch predecessor artifact order mismatch",
    )
    require(
        option_values("--prior-rejection-receipt") == expected_prior_receipts,
        "launch predecessor receipt order mismatch",
    )
elif stage == "encoder-epoch11":
    require("python\0-m\0pams\0train\0encoder" in command, "epoch11 command mismatch")
    require(command.count("train\0encoder") == 1, "epoch11 encoder invocation count")
    require("--epochs\0" + "11" in command, "epoch11 stop epoch missing")
    require("--resume" not in command, "epoch11 must be a fresh run")
    require(
        option_values("--candidate-launch-authorization")
        == ["/pams/launch/authorization.json"],
        "epoch11 launch authorization binding mismatch",
    )
    require(
        option_values("--candidate-launch-receipt")
        == ["/pams/launch/authorization.json.receipt.json"],
        "epoch11 launch receipt binding mismatch",
    )
elif stage == "encoder-final":
    require("python\0-m\0pams\0train\0encoder" in command, "final command mismatch")
    require(command.count("train\0encoder") == 1, "final encoder invocation count")
    require("--epochs\0" + "150" in command, "final stop epoch missing")
    require("--resume" in command, "final run must resume")
    require(
        "--resume-checkpoint\0/pams/resume/encoder.pt" in command,
        "resume checkpoint binding missing",
    )
    require(
        "--resume-progress\0/pams/resume/encoder.jsonl" in command,
        "resume progress binding missing",
    )
    require(
        option_values("--candidate-launch-authorization")
        == ["/pams/launch/authorization.json"],
        "final launch authorization binding mismatch",
    )
    require(
        option_values("--candidate-launch-receipt")
        == ["/pams/launch/authorization.json.receipt.json"],
        "final launch receipt binding mismatch",
    )
elif stage == "epoch11-gate":
    require("run_pams_native_epoch11_train_gate.py" in command, "gate command mismatch")
    require("train\0encoder" not in command, "gate must not train")
    require(
        "--encoder-completion-receipt\0/pams/epoch11/completion.receipt.json"
        in command,
        "gate completion receipt binding missing",
    )
else:
    require("validate_pams_native_baseline_inputs.py" in command, "preflight command mismatch")

environment = set(config.get("Env") or [])
required_environment = {
    "PYTHONPATH=/workspace/src",
    "PYTHONOPTIMIZE=",
    "PAMS_AUDIT_MODE=formal",
    f'PAMS_CONTAINER_IMAGE_ID={os.environ["VERIFY_IMAGE_ID"]}',
    f'PAMS_CONTAINER_SOURCE_REVISION={os.environ["VERIFY_SOURCE_REVISION"]}',
    "PAMS_CONTAINER_ENVIRONMENT_SHA256=" + os.environ["VERIFY_ENVIRONMENT_SHA256"],
    "PAMS_SOURCE_EXPORT_RECEIPT=/pams/source-export-receipt.json",
    "PAMS_SOURCE_EXPORT_RECEIPT_SHA256="
    + os.environ["VERIFY_SOURCE_RECEIPT_SHA256"],
}
require(required_environment <= environment, "required container environment is incomplete")

devices = host.get("DeviceRequests") or []
if stage in {"encoder-epoch11", "epoch11-gate", "encoder-final"}:
    require(len(devices) == 1, "exactly one GPU request is required")
    require("gpu" in devices[0]["Capabilities"][0], "GPU capability is missing")
    require(
        devices[0].get("DeviceIDs") == [os.environ["VERIFY_GPU_DEVICE"]],
        "GPU device binding mismatch",
    )
    require("CUDA_VISIBLE_DEVICES=0" in environment, "CUDA remap is missing")
    require(
        "CUBLAS_WORKSPACE_CONFIG=:4096:8" in environment,
        "deterministic cuBLAS environment is missing",
    )
else:
    require(not devices, f"{stage} must not request a GPU")
    require(
        "CUDA_VISIBLE_DEVICES=" in environment,
        f"{stage} CUDA disablement missing",
    )

source_view = Path(os.environ["VERIFY_SOURCE_VIEW"])
require(
    {entry.name for entry in source_view.iterdir()}
    == {"configs", "pyproject.toml", "scripts", "src"},
    "source export top-level set mismatch",
)
for forbidden in (".git", "data", "results", "tests"):
    require(not (source_view / forbidden).exists(), f"forbidden source path: {forbidden}")

report = {
    "schema_version": 1,
    "stage": stage,
    "verified": True,
    "container_name": item["Name"].lstrip("/"),
    "image_id": config["Image"],
    "network_mode": host["NetworkMode"],
    "read_only_root": host["ReadonlyRootfs"],
    "gpu_requested": bool(devices),
    "mounts": [
        {"destination": key, "source": value[0], "rw": value[1]}
        for key, value in sorted(mounts.items())
    ],
}
Path(os.environ["VERIFY_INSPECT"]).with_suffix(".verification.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
}

validate_completion_receipt() {
  local receipt="$1"
  local checkpoint="$2"
  local progress="$3"
  local pose_cache_set_sha256="$4"
  local expected_epochs="$5"
  local resume_checkpoint="${6:-}"
  local resume_progress="${7:-}"
  [[ "$expected_epochs" == 11 || "$expected_epochs" == 150 ]] \
    || fail 'completion receipt validator expected epoch must be 11 or 150'
  if [[ "$expected_epochs" == 11 ]]; then
    [[ -z "$resume_checkpoint" && -z "$resume_progress" ]] \
      || fail 'epoch11 completion must not name resume inputs'
  else
    [[ -n "$resume_checkpoint" && -n "$resume_progress" ]] \
      || fail 'epoch150 completion requires both immutable resume inputs'
  fi
  VERIFY_RECEIPT="$receipt" \
  VERIFY_CHECKPOINT="$checkpoint" \
  VERIFY_PROGRESS="$progress" \
  VERIFY_SOURCE_REVISION="$SOURCE_REVISION" \
  VERIFY_IMAGE_ID="$IMAGE_ID" \
  VERIFY_ENVIRONMENT_SHA256="$ENVIRONMENT_SHA256" \
  VERIFY_CONFIG_SHA256="$CONFIG_SHA256" \
  VERIFY_CONFIG_FINGERPRINT="$CONFIG_FINGERPRINT" \
  VERIFY_POSE_FINGERPRINT="$POSE_FINGERPRINT" \
  VERIFY_POSE_CACHE_SET_SHA256="$pose_cache_set_sha256" \
  VERIFY_TRAIN_INPUT_SHA256="$TRAIN_INPUT_SHA256" \
  VERIFY_TRAIN_COMMIT_SHA256="$TRAIN_COMMIT_SHA256" \
  VERIFY_DEV_INPUT_SHA256="$DEV_INPUT_SHA256" \
  VERIFY_DEV_COMMIT_SHA256="$DEV_COMMIT_SHA256" \
  VERIFY_TEST_INPUT_SHA256="$TEST_ID_INPUT_SHA256" \
  VERIFY_TEST_COMMIT_SHA256="$TEST_ID_COMMIT_SHA256" \
  VERIFY_EXPECTED_EPOCHS="$expected_epochs" \
  VERIFY_RESUME_CHECKPOINT="$resume_checkpoint" \
  VERIFY_RESUME_PROGRESS="$resume_progress" \
  VERIFY_LAUNCH_AUTHORIZATION="$LAUNCH_AUTHORIZATION_ARTIFACT" \
  VERIFY_LAUNCH_RECEIPT="$LAUNCH_AUTHORIZATION_RECEIPT" \
  python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def load_strict(path: Path):
    def reject_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON field: {key!r}")
            result[key] = value
        return result
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            RuntimeError(f"non-finite JSON constant: {value}")
        ),
    )

receipt_path = Path(os.environ["VERIFY_RECEIPT"])
payload = load_strict(receipt_path)
require(
    set(payload)
    == {
        "schema_version", "receipt_type", "run_id", "status", "finished_at",
        "start_manifest_sha256", "started", "artifacts", "metrics",
    },
    "completion receipt top-level schema mismatch",
)
require(payload["schema_version"] == 3, "completion receipt schema mismatch")
require(payload["receipt_type"] == "completed", "completion receipt type mismatch")
require(payload["status"] == "completed", "completion receipt status mismatch")
started = payload["started"]
require(
    set(started)
    == {
        "schema_version", "receipt_type", "run_id", "created_at_utc", "command",
        "git_sha", "config_sha256", "dataset_sha256", "seed", "protocol",
        "status", "hardware", "notes",
    },
    "started receipt schema mismatch",
)
require(started["schema_version"] == 2, "started receipt schema version mismatch")
require(started["receipt_type"] == "started", "started receipt type mismatch")
require(started["status"] == "started", "started receipt status mismatch")
require(started["git_sha"] == os.environ["VERIFY_SOURCE_REVISION"], "source mismatch")
require(
    started["config_sha256"] == os.environ["VERIFY_CONFIG_FINGERPRINT"],
    "config fingerprint mismatch",
)
require(started["seed"] == 2026, "seed mismatch")
require(started["protocol"] == "ucfrep_526", "protocol mismatch")
require(
    started["command"].count("--candidate-launch-authorization") == 1,
    "started command launch authorization flag mismatch",
)
require(
    started["command"].count("--candidate-launch-receipt") == 1,
    "started command launch receipt flag mismatch",
)
container = started["hardware"]["container"]
require(
    container
    == {
        "image_id": os.environ["VERIFY_IMAGE_ID"],
        "environment_sha256": os.environ["VERIFY_ENVIRONMENT_SHA256"],
        "source_revision": os.environ["VERIFY_SOURCE_REVISION"],
    },
    "completion receipt container identity mismatch",
)
expected_epochs = int(os.environ["VERIFY_EXPECTED_EPOCHS"])
require(
    set(payload["metrics"]) == {"completed_epochs", "final_epoch"},
    "completion receipt metrics schema mismatch",
)
require(
    payload["metrics"]["completed_epochs"] == expected_epochs,
    "completed epoch mismatch",
)

started_path = receipt_path.with_name(f'{payload["run_id"]}.started.json')
require(started_path.is_file(), "sibling started receipt is missing")
require(
    digest(started_path) == payload["start_manifest_sha256"],
    "started receipt SHA mismatch",
)
require(load_strict(started_path) == started, "embedded and sibling started receipts differ")

roles = {item["role"]: item for item in payload["artifacts"]}
require(len(roles) == len(payload["artifacts"]), "duplicate artifact role")
required = {
    "output_encoder_checkpoint",
    "progress_log",
    "input_config",
    "input_dataset_manifest",
    "input_pose_cache_snapshot",
    "input_train_pose_inputs",
    "input_train_pose_input_commitment",
    "input_dev_pose_inputs",
    "input_dev_pose_input_commitment",
    "input_test_identity_pose_inputs",
    "input_test_identity_pose_input_commitment",
    "input_candidate_launch_authorization",
    "input_candidate_launch_authorization_receipt",
}
resume_checkpoint = os.environ["VERIFY_RESUME_CHECKPOINT"]
resume_progress = os.environ["VERIFY_RESUME_PROGRESS"]
if expected_epochs == 11:
    require(set(roles) == required, "epoch11 completion artifact role set mismatch")
else:
    require(bool(resume_checkpoint and resume_progress), "resume input paths are missing")
    require(
        set(roles) == required | {"input_resume_checkpoint", "input_resume_progress"},
        "epoch150 completion artifact role set mismatch",
    )
package_root = receipt_path.parent.parent.resolve(strict=True)
resolved = {}
for item in payload["artifacts"]:
    locator = PurePosixPath(item["locator"])
    require(
        set(item) == {"role", "locator", "sha256", "bytes"},
        "artifact receipt schema mismatch",
    )
    require(
        isinstance(item["locator"], str) and "\\" not in item["locator"],
        "artifact locator is not portable",
    )
    require(bool(locator.parts) and not locator.is_absolute(), "artifact locator is unsafe")
    path = receipt_path.parent.joinpath(*locator.parts).resolve(strict=True)
    try:
        path.relative_to(package_root)
    except ValueError as exc:
        raise RuntimeError("artifact locator escapes the run package") from exc
    require(path.is_file(), "artifact locator is not a file")
    require(path.stat().st_size == item["bytes"], "artifact byte count mismatch")
    require(digest(path) == item["sha256"], "artifact SHA mismatch")
    resolved[item["role"]] = path

require(
    digest(Path(os.environ["VERIFY_CHECKPOINT"]))
    == roles["output_encoder_checkpoint"]["sha256"],
    "live checkpoint SHA differs from completion receipt",
)
require(
    digest(Path(os.environ["VERIFY_PROGRESS"])) == roles["progress_log"]["sha256"],
    "live progress SHA differs from completion receipt",
)
expected_hashes = {
    "input_config": os.environ["VERIFY_CONFIG_SHA256"],
    "input_dataset_manifest": os.environ["VERIFY_TRAIN_INPUT_SHA256"],
    "input_train_pose_inputs": os.environ["VERIFY_TRAIN_INPUT_SHA256"],
    "input_train_pose_input_commitment": os.environ["VERIFY_TRAIN_COMMIT_SHA256"],
    "input_dev_pose_inputs": os.environ["VERIFY_DEV_INPUT_SHA256"],
    "input_dev_pose_input_commitment": os.environ["VERIFY_DEV_COMMIT_SHA256"],
    "input_test_identity_pose_inputs": os.environ["VERIFY_TEST_INPUT_SHA256"],
    "input_test_identity_pose_input_commitment": os.environ["VERIFY_TEST_COMMIT_SHA256"],
    "input_candidate_launch_authorization": digest(
        Path(os.environ["VERIFY_LAUNCH_AUTHORIZATION"])
    ),
    "input_candidate_launch_authorization_receipt": digest(
        Path(os.environ["VERIFY_LAUNCH_RECEIPT"])
    ),
}
for role, expected in expected_hashes.items():
    require(roles[role]["sha256"] == expected, f"input binding mismatch: {role}")
if expected_epochs == 150:
    require(
        roles["input_resume_checkpoint"]["sha256"] == digest(Path(resume_checkpoint)),
        "resume checkpoint SHA mismatch",
    )
    require(
        roles["input_resume_progress"]["sha256"] == digest(Path(resume_progress)),
        "resume progress SHA mismatch",
    )

snapshot = load_strict(resolved["input_pose_cache_snapshot"])
require(
    set(snapshot)
    == {"schema_version", "pose_fingerprint", "fingerprint", "entry_count", "entries"},
    "pose snapshot schema mismatch",
)
require(snapshot["schema_version"] == 1, "pose snapshot schema version mismatch")
require(
    snapshot["pose_fingerprint"] == os.environ["VERIFY_POSE_FINGERPRINT"],
    "pose fingerprint mismatch",
)
require(
    snapshot["fingerprint"] == os.environ["VERIFY_POSE_CACHE_SET_SHA256"],
    "pose-cache set fingerprint mismatch",
)
require(snapshot["entry_count"] == 337, "pose snapshot entry count mismatch")
require(len(snapshot["entries"]) == 337, "pose snapshot entries length mismatch")
PY
}

validate_epoch11_gate_artifacts() {
  local gate_exit="$1"
  local checkpoint_sha256="$2"
  local progress_sha256="$3"
  local completion_receipt_sha256="$4"
  local snapshot_sha256="$5"
  local pose_cache_set_sha256="$6"
  local gate_specification_sha256="$7"
  VERIFY_GATE_ARTIFACT="$EPOCH11_GATE_ARTIFACT" \
  VERIFY_GATE_RECEIPT="$EPOCH11_GATE_RECEIPT" \
  VERIFY_GATE_EXIT="$gate_exit" \
  VERIFY_CHECKPOINT_SHA256="$checkpoint_sha256" \
  VERIFY_PROGRESS_SHA256="$progress_sha256" \
  VERIFY_COMPLETION_RECEIPT_SHA256="$completion_receipt_sha256" \
  VERIFY_SNAPSHOT_SHA256="$snapshot_sha256" \
  VERIFY_POSE_CACHE_SET_SHA256="$pose_cache_set_sha256" \
  VERIFY_GATE_SPECIFICATION_SHA256="$gate_specification_sha256" \
  VERIFY_CONFIG_SHA256="$CONFIG_SHA256" \
  VERIFY_CONFIG_FINGERPRINT="$CONFIG_FINGERPRINT" \
  VERIFY_POSE_FINGERPRINT="$POSE_FINGERPRINT" \
  VERIFY_SOURCE_REVISION="$SOURCE_REVISION" \
  VERIFY_IMAGE_ID="$IMAGE_ID" \
  VERIFY_ENVIRONMENT_SHA256="$ENVIRONMENT_SHA256" \
  VERIFY_CONFIG_RELATIVE="$CONFIG_RELATIVE" \
  VERIFY_GATE_RELATIVE="$EPOCH11_GATE_RELATIVE" \
  VERIFY_GATE_SPEC_RELATIVE="$EPOCH11_GATE_SPEC_RELATIVE" \
  python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

artifact_path = Path(os.environ["VERIFY_GATE_ARTIFACT"])
receipt_path = Path(os.environ["VERIFY_GATE_RECEIPT"])

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

artifact_bytes = artifact_path.read_bytes()
artifact = json.loads(artifact_bytes)
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
exit_code = int(os.environ["VERIFY_GATE_EXIT"])
authorized = exit_code == 0
require(exit_code in {0, 3}, "gate exit status is outside the contract")
require(artifact["schema_version"] == 1, "gate artifact schema mismatch")
require(
    artifact["artifact_type"] == "pams_native_epoch11_train_mechanism_gate_v1",
    "gate artifact type mismatch",
)
require(
    artifact["status"]
    == (
        "encoder_continuation_authorized"
        if authorized
        else "encoder_continuation_rejected"
    ),
    "gate artifact status disagrees with exit code",
)
require(artifact["protocol"] == "ucfrep_526", "gate protocol mismatch")
require(artifact["seed"] == 2026, "gate seed mismatch")
inputs = artifact["inputs"]
expected_input_hashes = {
    "encoder_checkpoint_sha256": os.environ["VERIFY_CHECKPOINT_SHA256"],
    "encoder_progress_sha256": os.environ["VERIFY_PROGRESS_SHA256"],
    "encoder_completion_receipt_sha256": os.environ[
        "VERIFY_COMPLETION_RECEIPT_SHA256"
    ],
    "pose_snapshot_sha256": os.environ["VERIFY_SNAPSHOT_SHA256"],
    "pose_cache_set_sha256": os.environ["VERIFY_POSE_CACHE_SET_SHA256"],
    "gate_specification_sha256": os.environ["VERIFY_GATE_SPECIFICATION_SHA256"],
    "experiment_config_sha256": os.environ["VERIFY_CONFIG_SHA256"],
    "config_fingerprint": os.environ["VERIFY_CONFIG_FINGERPRINT"],
    "pose_fingerprint": os.environ["VERIFY_POSE_FINGERPRINT"],
    "source_git_sha": os.environ["VERIFY_SOURCE_REVISION"],
    "container_image_id": os.environ["VERIFY_IMAGE_ID"],
    "container_environment_sha256": os.environ["VERIFY_ENVIRONMENT_SHA256"],
}
for field, expected in expected_input_hashes.items():
    require(inputs[field] == expected, f"gate input binding mismatch: {field}")
require(inputs["training_video_total"] == 337, "gate training set size mismatch")
require(
    inputs["source_receipt_covered_paths"]
    == {
        "experiment_config": os.environ["VERIFY_CONFIG_RELATIVE"],
        "gate_runner": os.environ["VERIFY_GATE_RELATIVE"],
        "gate_specification": os.environ["VERIFY_GATE_SPEC_RELATIVE"],
    },
    "gate source receipt coverage mismatch",
)
require(
    inputs["encoder_completion_receipt"]
    == {
        "schema_version": 3,
        "run_id": inputs["encoder_completion_receipt"]["run_id"],
        "sha256": os.environ["VERIFY_COMPLETION_RECEIPT_SHA256"],
        "bytes": inputs["encoder_completion_receipt_bytes"],
        "completed_epochs": 11,
        "artifact_roles": [
            "input_candidate_launch_authorization",
            "input_candidate_launch_authorization_receipt",
            "input_config",
            "input_dataset_manifest",
            "input_dev_pose_input_commitment",
            "input_dev_pose_inputs",
            "input_pose_cache_snapshot",
            "input_test_identity_pose_input_commitment",
            "input_test_identity_pose_inputs",
            "input_train_pose_input_commitment",
            "input_train_pose_inputs",
            "output_encoder_checkpoint",
            "progress_log",
        ],
    },
    "gate completion receipt summary mismatch",
)
gate = artifact["gate"]
require(
    gate["thresholds_frozen_before_native_candidate_training"] is True,
    "gate thresholds were not frozen",
)
require(gate["all_core_criteria_pass"] is authorized, "gate criterion aggregate mismatch")
require(
    gate["encoder_continuation_authorized"] is authorized,
    "gate continuation decision mismatch",
)
require(
    gate["prediction_or_scoring_authorized"] is False,
    "gate must never authorize prediction or scoring",
)
criterion_results = [bool(item["pass"]) for item in gate["criteria"].values()]
require(bool(criterion_results), "gate criteria are empty")
require(all(criterion_results) is authorized, "gate criteria disagree with decision")
require(artifact["schedule"]["completed_epochs"] == 11, "gate epoch mismatch")
read_only = artifact["read_only_verification"]
require(read_only["all_file_inputs_unchanged"] is True, "gate input changed")
require(read_only["pose_cache_set_unchanged"] is True, "gate pose cache changed")
require(
    read_only["model_or_optimizer_state_updated"] is False,
    "gate updated model or optimizer state",
)
require(read_only["training_steps_executed"] == 0, "gate executed training steps")
require(read_only["pose_cache_write_operations"] == 0, "gate wrote pose caches")

artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
require(
    receipt
    == {
        "schema_version": 1,
        "artifact_type": "pams_native_epoch11_train_mechanism_gate_receipt_v1",
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": len(artifact_bytes),
        "artifact_status": artifact["status"],
        "encoder_continuation_authorized": authorized,
        "encoder_checkpoint_sha256": os.environ["VERIFY_CHECKPOINT_SHA256"],
        "encoder_progress_sha256": os.environ["VERIFY_PROGRESS_SHA256"],
        "encoder_completion_receipt_sha256": os.environ[
            "VERIFY_COMPLETION_RECEIPT_SHA256"
        ],
        "pose_cache_set_sha256": os.environ["VERIFY_POSE_CACHE_SET_SHA256"],
        "gate_specification_sha256": os.environ["VERIFY_GATE_SPECIFICATION_SHA256"],
        "source_git_sha": os.environ["VERIFY_SOURCE_REVISION"],
    },
    "gate receipt does not exactly bind the artifact",
)
PY
}

validate_resume_lineage() {
  local epoch11_checkpoint_sha256="$1"
  local epoch11_progress_sha256="$2"
  local epoch11_snapshot_sha256="$3"
  local gate_artifact_sha256="$4"
  local gate_receipt_sha256="$5"
  VERIFY_EPOCH11_CHECKPOINT="$EPOCH11_ENCODER_CHECKPOINT" \
  VERIFY_EPOCH11_PROGRESS="$EPOCH11_ENCODER_PROGRESS" \
  VERIFY_EPOCH11_SNAPSHOT="$EPOCH11_POSE_SNAPSHOT" \
  VERIFY_FINAL_PROGRESS="$FINAL_ENCODER_PROGRESS" \
  VERIFY_GATE_ARTIFACT="$EPOCH11_GATE_ARTIFACT" \
  VERIFY_GATE_RECEIPT="$EPOCH11_GATE_RECEIPT" \
  VERIFY_EPOCH11_CHECKPOINT_SHA256="$epoch11_checkpoint_sha256" \
  VERIFY_EPOCH11_PROGRESS_SHA256="$epoch11_progress_sha256" \
  VERIFY_EPOCH11_SNAPSHOT_SHA256="$epoch11_snapshot_sha256" \
  VERIFY_GATE_ARTIFACT_SHA256="$gate_artifact_sha256" \
  VERIFY_GATE_RECEIPT_SHA256="$gate_receipt_sha256" \
  python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

epoch11_checkpoint = Path(os.environ["VERIFY_EPOCH11_CHECKPOINT"])
epoch11_progress = Path(os.environ["VERIFY_EPOCH11_PROGRESS"])
epoch11_snapshot = Path(os.environ["VERIFY_EPOCH11_SNAPSHOT"])
gate_artifact = Path(os.environ["VERIFY_GATE_ARTIFACT"])
gate_receipt = Path(os.environ["VERIFY_GATE_RECEIPT"])
require(
    digest(epoch11_checkpoint) == os.environ["VERIFY_EPOCH11_CHECKPOINT_SHA256"],
    "epoch11 checkpoint changed before resume completion",
)
require(
    digest(epoch11_progress) == os.environ["VERIFY_EPOCH11_PROGRESS_SHA256"],
    "epoch11 progress changed before resume completion",
)
require(
    digest(epoch11_snapshot) == os.environ["VERIFY_EPOCH11_SNAPSHOT_SHA256"],
    "epoch11 pose snapshot changed before resume completion",
)
require(
    digest(gate_artifact) == os.environ["VERIFY_GATE_ARTIFACT_SHA256"],
    "gate artifact changed before resume completion",
)
require(
    digest(gate_receipt) == os.environ["VERIFY_GATE_RECEIPT_SHA256"],
    "gate receipt changed before resume completion",
)

epoch11_bytes = epoch11_progress.read_bytes()
final_bytes = Path(os.environ["VERIFY_FINAL_PROGRESS"]).read_bytes()
require(final_bytes.startswith(epoch11_bytes), "final progress is not an epoch11 prefix resume")
epoch11_rows = tuple(
    json.loads(line) for line in epoch11_bytes.decode("utf-8").splitlines()
)
final_rows = tuple(json.loads(line) for line in final_bytes.decode("utf-8").splitlines())
require(len(epoch11_rows) == 11, "epoch11 progress row count mismatch")
require(len(final_rows) == 150, "final progress row count mismatch")
require(final_rows[:11] == epoch11_rows, "final progress prefix changed")
require(
    [row["epoch"] for row in final_rows] == list(range(1, 151)),
    "final progress epochs are not consecutive 1..150",
)
PY
}

[[ "$POSE_RECOVERY_RUN_ROOT" == "${ROOT}/runs/pose-recovery-"*/* ]] \
  || fail 'PAMS_POSE_RECOVERY_RUN_ROOT must be one exact pose-recovery run under PAMS_ROOT'
[[ ! -L "$POSE_RECOVERY_RUN_ROOT_INPUT" ]] \
  || fail 'pose-recovery run root must not be a symlink'
[[ "$POSE_RECOVERY_CONFIG" == "${POSE_RECOVERY_RUN_ROOT}/source/configs/experiments/"*.yaml ]] \
  || fail 'pose-recovery config must be an exported experiment YAML inside its run root'
[[ ! -L "$POSE_RECOVERY_CONFIG_INPUT" ]] \
  || fail 'pose-recovery config must not be a symlink'
[[ -d "$OFFICIAL_ROOT" ]] || fail 'official-segment-v1 root is missing'
[[ -d "$POSE_RECOVERY_CACHE" ]] || fail 'authorized train337 pose cache is missing'
for upstream_path in \
  "$POSE_RECOVERY_CACHE" \
  "$POSE_RECOVERY_LEDGER" \
  "$POSE_RECOVERY_AUTHORIZATION" \
  "$POSE_RECOVERY_PAIRED_GATE" \
  "$POSE_RECOVERY_RUN_RECEIPT"; do
  [[ ! -L "$upstream_path" ]] \
    || fail "pose-recovery artifact must not be a symlink: ${upstream_path}"
done
[[ -z "$(find "$POSE_RECOVERY_CACHE" -mindepth 1 -type l -print -quit)" ]] \
  || fail 'authorized train337 pose cache contains a symlink'
[[ "$(find "$POSE_RECOVERY_CACHE" -mindepth 1 -maxdepth 1 -type f -name '*.npz' | wc -l)" -eq 337 ]] \
  || fail 'authorized train337 pose cache must contain exactly 337 NPZ files'
[[ "$(find "$POSE_RECOVERY_CACHE" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 337 ]] \
  || fail 'authorized train337 pose cache contains unexpected files'

require_sha256 "$TRAIN_INPUT" "$TRAIN_INPUT_SHA256" 'official train337 identity sidecar'
require_sha256 "$TRAIN_COMMIT" "$TRAIN_COMMIT_SHA256" 'official train337 identity commitment'
require_sha256 "$DEV_INPUT" "$DEV_INPUT_SHA256" 'official dev84 identity sidecar'
require_sha256 "$DEV_COMMIT" "$DEV_COMMIT_SHA256" 'official dev84 identity commitment'
require_sha256 "$TEST_ID_INPUT" "$TEST_ID_INPUT_SHA256" 'official test105 identity sidecar'
require_sha256 "$TEST_ID_COMMIT" "$TEST_ID_COMMIT_SHA256" 'official test105 identity commitment'
require_sha256 \
  "$POSE_RECOVERY_AUTHORIZATION" \
  "$POSE_RECOVERY_AUTHORIZATION_SHA256" \
  'pose-recovery baseline authorization'
readonly POSE_RECOVERY_SOURCE_REVISION="$(
  jq -er '.bindings.source_revision' "$POSE_RECOVERY_AUTHORIZATION"
)"
readonly POSE_RECOVERY_IMAGE_ID="$(
  jq -er '.bindings.container_image_id' "$POSE_RECOVERY_AUTHORIZATION"
)"
readonly POSE_RECOVERY_CONFIG_SHA256="$(
  jq -er '.bindings.config_file_sha256' "$POSE_RECOVERY_AUTHORIZATION"
)"
readonly POSE_RECOVERY_CONFIG_FINGERPRINT="$(
  jq -er '.bindings.config_fingerprint' "$POSE_RECOVERY_AUTHORIZATION"
)"
readonly POSE_FINGERPRINT="$(
  jq -er '.bindings.pose_fingerprint' "$POSE_RECOVERY_AUTHORIZATION"
)"
readonly POSE_RECOVERY_PAIRED_GATE_SHA256="$(
  jq -er '.bindings.paired_gate_sha256' "$POSE_RECOVERY_AUTHORIZATION"
)"
readonly POSE_RECOVERY_RUN_RECEIPT_SHA256="$(
  jq -er '.bindings.run_receipt_sha256' "$POSE_RECOVERY_AUTHORIZATION"
)"
readonly POSE_RECOVERY_LEDGER_SHA256="$(
  jq -er '.bindings.ledger_sha256' "$POSE_RECOVERY_AUTHORIZATION"
)"
readonly AUTHORIZED_POSE_CACHE_SET_SHA256="$(
  jq -er '.bindings.pose_cache_set_sha256' "$POSE_RECOVERY_AUTHORIZATION"
)"
for digest_name in \
  POSE_RECOVERY_CONFIG_SHA256 \
  POSE_RECOVERY_CONFIG_FINGERPRINT \
  POSE_FINGERPRINT \
  POSE_RECOVERY_PAIRED_GATE_SHA256 \
  POSE_RECOVERY_RUN_RECEIPT_SHA256 \
  POSE_RECOVERY_LEDGER_SHA256 \
  AUTHORIZED_POSE_CACHE_SET_SHA256; do
  digest="${!digest_name}"
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] \
    || fail "authorization binding ${digest_name} must be a lowercase SHA-256"
done
[[ "$POSE_RECOVERY_SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]] \
  || fail 'authorized pose-recovery source revision is invalid'
[[ "$POSE_RECOVERY_IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || fail 'authorized pose-recovery image ID is invalid'
[[ "$(docker image inspect "$POSE_RECOVERY_IMAGE_ID" --format '{{.Id}}')" == "$POSE_RECOVERY_IMAGE_ID" ]] \
  || fail 'authorized pose-recovery container image is unavailable'
[[ "$(docker image inspect "$POSE_RECOVERY_IMAGE_ID" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')" == "$POSE_RECOVERY_SOURCE_REVISION" ]] \
  || fail 'authorized pose-recovery image/source label mismatch'

require_sha256 \
  "$POSE_RECOVERY_CONFIG" \
  "$POSE_RECOVERY_CONFIG_SHA256" \
  'authorized pose-recovery config'
require_sha256 \
  "$POSE_RECOVERY_PAIRED_GATE" \
  "$POSE_RECOVERY_PAIRED_GATE_SHA256" \
  'authorized pose-recovery paired gate'
require_sha256 \
  "$POSE_RECOVERY_RUN_RECEIPT" \
  "$POSE_RECOVERY_RUN_RECEIPT_SHA256" \
  'authorized pose-recovery run receipt'
require_sha256 \
  "$POSE_RECOVERY_LEDGER" \
  "$POSE_RECOVERY_LEDGER_SHA256" \
  'authorized pose-recovery train337 ledger'

jq -e \
  --arg version "$POSE_RECOVERY_VERSION" \
  --arg revision "$POSE_RECOVERY_SOURCE_REVISION" \
  --arg image "$POSE_RECOVERY_IMAGE_ID" \
  --arg config_sha "$POSE_RECOVERY_CONFIG_SHA256" \
  --arg config_fp "$POSE_RECOVERY_CONFIG_FINGERPRINT" \
  --arg pose "$POSE_FINGERPRINT" \
  --arg gate_sha "$POSE_RECOVERY_PAIRED_GATE_SHA256" \
  --arg receipt_sha "$POSE_RECOVERY_RUN_RECEIPT_SHA256" \
  --arg ledger_sha "$POSE_RECOVERY_LEDGER_SHA256" \
  --arg cache_set_sha "$AUTHORIZED_POSE_CACHE_SET_SHA256" \
  '
    .schema_version == 1
    and .artifact_type == "pams_native_pose_recovery_train337_authorization"
    and .version == $version
    and .authorized == true
    and .authorized_consumer == "pams_native_table2_baseline_proxy_train337_encoder"
    and .classification == "trusted_native_pose_recovery_train337"
    and .protocol == "ucfrep_526"
    and .split == "train"
    and .record_total == 337
    and (.authorization_sha256 | not)
    and .bindings.source_revision == $revision
    and .bindings.container_image_id == $image
    and .bindings.config_file_sha256 == $config_sha
    and .bindings.config_fingerprint == $config_fp
    and .bindings.pose_fingerprint == $pose
    and .bindings.paired_gate_sha256 == $gate_sha
    and .bindings.run_receipt_sha256 == $receipt_sha
    and .bindings.ledger_sha256 == $ledger_sha
    and .bindings.pose_cache_set_sha256 == $cache_set_sha
    and .scope.pose_timeline == "native"
    and .scope.pose_cache == "train337_only"
    and .scope.source_videos_mounted == false
    and .scope.dev84_mounted == false
    and .scope.test105_mounted == false
    and .scope.targets_mounted == false
    and .scope.network_mode == "none"
    and .scope.source_export_read_only == true
  ' "$POSE_RECOVERY_AUTHORIZATION" >/dev/null \
  || fail 'pose-recovery authorization is absent, rejected, or malformed'
jq -e \
  '
    .schema_version == 1
    and .protocol == "ucfrep_526"
    and .split == "train"
    and .passed == true
    and .metrics.record_total == 337
    and .mount_audit.train_identity_mounted == true
    and .mount_audit.train_pose_caches_mounted == true
    and .mount_audit.source_videos_mounted == false
    and .mount_audit.dev84_mounted == false
    and .mount_audit.test105_mounted == false
    and .mount_audit.targets_mounted == false
  ' "$POSE_RECOVERY_PAIRED_GATE" >/dev/null \
  || fail 'pose-recovery paired gate is absent, failed, or violates the data firewall'
jq -e \
  --arg revision "$POSE_RECOVERY_SOURCE_REVISION" \
  --arg image "$POSE_RECOVERY_IMAGE_ID" \
  --arg config_sha "$POSE_RECOVERY_CONFIG_SHA256" \
  --arg config_fp "$POSE_RECOVERY_CONFIG_FINGERPRINT" \
  --arg pose "$POSE_FINGERPRINT" \
  --arg gate_sha "$POSE_RECOVERY_PAIRED_GATE_SHA256" \
  '
    .schema_version == 1
    and .source_revision == $revision
    and .container_image_id == $image
    and .config_file_sha256 == $config_sha
    and .config_fingerprint == $config_fp
    and .pose_fingerprint == $pose
    and .temporal_resampling == "none_native_timeline"
    and .scope == "count-free-train337-pose-input-recovery-only"
    and .paired_gate_exit_status == 0
    and .paired_gate_sha256 == $gate_sha
  ' "$POSE_RECOVERY_RUN_RECEIPT" >/dev/null \
  || fail 'pose-recovery run receipt violates the authorization binding'
jq -e \
  --arg pose "$POSE_FINGERPRINT" \
  '
    .protocol == "ucfrep_526"
    and .split == "train"
    and .selected == 337
    and .completed == 337
    and .failed == 0
    and .pose_fingerprint == $pose
  ' "$POSE_RECOVERY_LEDGER" >/dev/null \
  || fail 'pose-recovery ledger is not a complete train337 artifact'

[[ "$(git -C "$SOURCE_CHECKOUT" rev-parse HEAD^{commit})" == "$SOURCE_REVISION" ]] \
  || fail 'source checkout revision mismatch'
[[ -z "$(git -C "$SOURCE_CHECKOUT" status --porcelain=v1 --untracked-files=all)" ]] \
  || fail 'source checkout must be clean, including untracked files'
readonly CONFIG_SOURCE="${SOURCE_CHECKOUT}/${CONFIG_RELATIVE}"
require_sha256 "$CONFIG_SOURCE" "$CONFIG_SHA256" 'native proxy config'
[[ -f "${SOURCE_CHECKOUT}/${VALIDATOR_RELATIVE}" ]] || fail 'native proxy validator is missing'
[[ -f "${SOURCE_CHECKOUT}/${RUNNER_RELATIVE}" ]] || fail 'native proxy runner is missing'
[[ -f "${SOURCE_CHECKOUT}/${EPOCH11_GATE_RELATIVE}" ]] \
  || fail 'native epoch11 gate runner is missing'
[[ -f "${SOURCE_CHECKOUT}/${EPOCH11_GATE_SPEC_RELATIVE}" ]] \
  || fail 'native epoch11 gate specification is missing'
[[ -f "${SOURCE_CHECKOUT}/${TERMINAL_GATE_RELATIVE}" ]] \
  || fail 'native terminal gate runner is missing'
[[ -f "${SOURCE_CHECKOUT}/${TERMINAL_GATE_SPEC_RELATIVE}" ]] \
  || fail 'native terminal gate specification is missing'
[[ -f "${SOURCE_CHECKOUT}/${LAUNCH_AUTHORIZATION_RELATIVE}" ]] \
  || fail 'native launch authorization runner is missing'
for candidate_config in \
  "$CONFIG_A_RELATIVE" \
  "$CONFIG_B_RELATIVE" \
  "$CONFIG_C_RELATIVE"; do
  [[ -f "${SOURCE_CHECKOUT}/${candidate_config}" ]] \
    || fail "frozen candidate config is missing: ${candidate_config}"
  [[ ! -L "${SOURCE_CHECKOUT}/${candidate_config}" ]] \
    || fail "frozen candidate config must not be a symlink: ${candidate_config}"
done
source "${SOURCE_CHECKOUT}/scripts/server/environment_fingerprint.sh"
[[ "$(pams_environment_fingerprint "${SOURCE_CHECKOUT}/docker/server")" == "$ENVIRONMENT_SHA256" ]] \
  || fail 'source environment fingerprint mismatch'
[[ "$(docker image inspect "$IMAGE_ID" --format '{{.Id}}')" == "$IMAGE_ID" ]] \
  || fail 'container image ID mismatch'
[[ "$(docker image inspect "$IMAGE_ID" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')" == "$SOURCE_REVISION" ]] \
  || fail 'container source label mismatch'
[[ "$(docker image inspect "$IMAGE_ID" --format '{{index .Config.Labels "org.opencontainers.image.pams.environment-sha256"}}')" == "$ENVIRONMENT_SHA256" ]] \
  || fail 'container environment label mismatch'

mkdir -p -- "$RUN_PARENT"
[[ ! -L "$RUN_PARENT" ]] || fail 'run parent must not be a symlink'
[[ "$(realpath -e -- "$RUN_PARENT")" == "${ROOT}/runs/pams-native-table2-baseline-v1" ]] \
  || fail 'run parent resolves outside PAMS_ROOT'
mkdir -- "$RUN_ROOT" || fail "immutable run root already exists: ${RUN_ROOT}"
RUN_RESERVED=1
mkdir -- "$SOURCE_VIEW" "$INPUT_ROOT" "${RUN_ROOT}/stages" \
  "$LAUNCH_AUTHORIZATION_STAGE" "$EPOCH11_ENCODER_STAGE" "$AUDIT_ROOT" "$LOG_ROOT"
write_status 'preparing' 'source-export' 'null'

git -C "$SOURCE_CHECKOUT" archive --format=tar "$SOURCE_REVISION" \
  src pyproject.toml \
  "$CONFIG_A_RELATIVE" "$CONFIG_B_RELATIVE" "$CONFIG_C_RELATIVE" \
  "$EPOCH11_GATE_SPEC_RELATIVE" "$TERMINAL_GATE_SPEC_RELATIVE" \
  "$RUNNER_RELATIVE" "$VALIDATOR_RELATIVE" "$EPOCH11_GATE_RELATIVE" \
  "$TERMINAL_GATE_RELATIVE" "$LAUNCH_AUTHORIZATION_RELATIVE" \
  | tar -xf - -C "$SOURCE_VIEW"
[[ ! -e "${SOURCE_VIEW}/.git" && ! -e "${SOURCE_VIEW}/data" \
  && ! -e "${SOURCE_VIEW}/results" && ! -e "${SOURCE_VIEW}/tests" ]] \
  || fail 'source export contains a forbidden top-level path'
[[ "$(find "$SOURCE_VIEW" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort | tr '\n' ' ')" == 'configs pyproject.toml scripts src ' ]] \
  || fail 'source export has an unexpected top-level entry'
[[ -z "$(find "$SOURCE_VIEW" -type l -print -quit)" ]] \
  || fail 'source export contains a symlink'
require_sha256 "${SOURCE_VIEW}/${CONFIG_RELATIVE}" "$CONFIG_SHA256" 'exported native proxy config'
readonly EPOCH11_GATE_SPEC_SHA256="$(
  sha256_file "${SOURCE_VIEW}/${EPOCH11_GATE_SPEC_RELATIVE}"
)"
readonly TERMINAL_GATE_SPEC_SHA256="$(
  sha256_file "${SOURCE_VIEW}/${TERMINAL_GATE_SPEC_RELATIVE}"
)"

SOURCE_EXPORT_ROOT="$SOURCE_VIEW" \
SOURCE_EXPORT_REVISION="$SOURCE_REVISION" \
SOURCE_EXPORT_RECEIPT="$SOURCE_RECEIPT" \
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["SOURCE_EXPORT_ROOT"]).resolve(strict=True)
records = []
for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
    if path.is_symlink():
        raise RuntimeError(f"source export contains symlink: {path}")
    if path.is_dir():
        continue
    if not path.is_file():
        raise RuntimeError(f"source export contains non-regular entry: {path}")
    content = path.read_bytes()
    records.append({
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    })
if not records:
    raise RuntimeError("source export is empty")
payload = {
    "schema_version": 1,
    "source_revision": os.environ["SOURCE_EXPORT_REVISION"],
    "root": ".",
    "files": records,
}
with Path(os.environ["SOURCE_EXPORT_RECEIPT"]).open(
    "x", encoding="utf-8", newline="\n"
) as handle:
    json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
PY
readonly SOURCE_RECEIPT_SHA256="$(sha256_file "$SOURCE_RECEIPT")"
cp -- "${SOURCE_VIEW}/${CONFIG_RELATIVE}" "$CONFIG_HOST"
require_sha256 "$CONFIG_HOST" "$CONFIG_SHA256" 'staged native proxy config'
chmod -R a-w -- "$SOURCE_VIEW"
chmod 0444 "$SOURCE_RECEIPT" "$CONFIG_HOST"

RESERVATION_PATH="${RUN_ROOT}/attempt.reservation.json" \
RESERVATION_ATTEMPT="$ATTEMPT_ID" \
RESERVATION_SOURCE="$SOURCE_REVISION" \
RESERVATION_SOURCE_RECEIPT="$SOURCE_RECEIPT_SHA256" \
RESERVATION_IMAGE="$IMAGE_ID" \
RESERVATION_ENVIRONMENT="$ENVIRONMENT_SHA256" \
RESERVATION_CONFIG_SHA="$CONFIG_SHA256" \
RESERVATION_CONFIG_FP="$CONFIG_FINGERPRINT" \
RESERVATION_CANDIDATE_ID="$CANDIDATE_ID" \
RESERVATION_CANDIDATE_WINDOW="$CANDIDATE_WINDOW" \
RESERVATION_CANDIDATE_STRIDE="$CANDIDATE_STRIDE" \
RESERVATION_GATE_SPEC_SHA="$EPOCH11_GATE_SPEC_SHA256" \
RESERVATION_TERMINAL_GATE_SPEC_SHA="$TERMINAL_GATE_SPEC_SHA256" \
RESERVATION_CANDIDATE_PRIOR_IDS_JSON="$CANDIDATE_PRIOR_IDS_JSON" \
RESERVATION_POSE_FP="$POSE_FINGERPRINT" \
RESERVATION_POSE_VERSION="$POSE_RECOVERY_VERSION" \
RESERVATION_POSE_ROOT="$POSE_RECOVERY_RUN_ROOT" \
RESERVATION_POSE_SOURCE="$POSE_RECOVERY_SOURCE_REVISION" \
RESERVATION_POSE_IMAGE="$POSE_RECOVERY_IMAGE_ID" \
RESERVATION_POSE_CONFIG_SHA="$POSE_RECOVERY_CONFIG_SHA256" \
RESERVATION_POSE_CONFIG_FP="$POSE_RECOVERY_CONFIG_FINGERPRINT" \
RESERVATION_POSE_AUTH="$POSE_RECOVERY_AUTHORIZATION_SHA256" \
RESERVATION_POSE_GATE="$POSE_RECOVERY_PAIRED_GATE_SHA256" \
RESERVATION_POSE_RECEIPT="$POSE_RECOVERY_RUN_RECEIPT_SHA256" \
RESERVATION_POSE_LEDGER="$POSE_RECOVERY_LEDGER_SHA256" \
RESERVATION_POSE_CACHE_SET="$AUTHORIZED_POSE_CACHE_SET_SHA256" \
python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "schema_version": 1,
    "attempt_id": os.environ["RESERVATION_ATTEMPT"],
    "reserved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "classification": "independently_inferred_proxy_not_author_table2_baseline",
    "paper_table2_value_claim_eligible": False,
    "protocol": "ucfrep_526",
    "seed": 2026,
    "source_revision": os.environ["RESERVATION_SOURCE"],
    "source_export_receipt_sha256": os.environ["RESERVATION_SOURCE_RECEIPT"],
    "container_image_id": os.environ["RESERVATION_IMAGE"],
    "container_environment_sha256": os.environ["RESERVATION_ENVIRONMENT"],
    "config_file_sha256": os.environ["RESERVATION_CONFIG_SHA"],
    "config_fingerprint": os.environ["RESERVATION_CONFIG_FP"],
    "candidate_id": os.environ["RESERVATION_CANDIDATE_ID"],
    "fixed_period_frames": int(os.environ["RESERVATION_CANDIDATE_WINDOW"]),
    "anchor_stride": int(os.environ["RESERVATION_CANDIDATE_STRIDE"]),
    "candidate_launch_policy": {
        "terminal_gate_specification_sha256": os.environ[
            "RESERVATION_TERMINAL_GATE_SPEC_SHA"
        ],
        "required_prior_scientific_rejection_candidate_ids": json.loads(
            os.environ["RESERVATION_CANDIDATE_PRIOR_IDS_JSON"]
        ),
        "authorization_must_exist_before_encoder_container_creation": True,
        "same_authorization_pair_required_for_epoch11_and_final": True,
    },
    "pose_fingerprint": os.environ["RESERVATION_POSE_FP"],
    "upstream_pose_recovery": {
        "version": os.environ["RESERVATION_POSE_VERSION"],
        "run_root": os.environ["RESERVATION_POSE_ROOT"],
        "source_revision": os.environ["RESERVATION_POSE_SOURCE"],
        "container_image_id": os.environ["RESERVATION_POSE_IMAGE"],
        "config_file_sha256": os.environ["RESERVATION_POSE_CONFIG_SHA"],
        "config_fingerprint": os.environ["RESERVATION_POSE_CONFIG_FP"],
        "authorization_sha256": os.environ["RESERVATION_POSE_AUTH"],
        "paired_gate_sha256": os.environ["RESERVATION_POSE_GATE"],
        "run_receipt_sha256": os.environ["RESERVATION_POSE_RECEIPT"],
        "ledger_sha256": os.environ["RESERVATION_POSE_LEDGER"],
        "pose_cache_set_sha256": os.environ["RESERVATION_POSE_CACHE_SET"],
    },
    "encoder_epochs": 150,
    "epoch11_gate": {
        "gate_epoch": 11,
        "gate_specification_sha256": os.environ["RESERVATION_GATE_SPEC_SHA"],
        "continuation_requires_exact_authorization": True,
        "resume_inputs_must_be_immutable_epoch11_checkpoint_and_progress": True,
    },
    "physical_batch_size": 32,
    "encoder_training_scope": "train337_only",
    "period_head_training_authorized": False,
    "sshead_training_authorized": False,
    "dev_authorized": False,
    "dev_pose_mounted": False,
    "dev_targets_mounted": False,
    "dev_prediction_authorized": False,
    "dev_scoring_authorized": False,
    "test_authorized": False,
    "test_pose_mounted": False,
    "test_targets_mounted": False,
    "test_prediction_authorized": False,
    "test_scoring_authorized": False,
    "identity_only_dev_test_sidecars_required_by_cli_provenance": True,
}
with Path(os.environ["RESERVATION_PATH"]).open(
    "x", encoding="utf-8", newline="\n"
) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
PY
chmod 0444 "${RUN_ROOT}/attempt.reservation.json"

{
  printf 'classification %s\n' "$CLASSIFICATION"
  printf 'source_revision %s\n' "$SOURCE_REVISION"
  printf 'source_export_receipt_sha256 %s\n' "$SOURCE_RECEIPT_SHA256"
  printf 'image_id %s\n' "$IMAGE_ID"
  printf 'environment_sha256 %s\n' "$ENVIRONMENT_SHA256"
  printf 'config_file_sha256 %s\n' "$CONFIG_SHA256"
  printf 'config_fingerprint %s\n' "$CONFIG_FINGERPRINT"
  printf 'candidate_id %s\n' "$CANDIDATE_ID"
  printf 'fixed_period_frames %s\n' "$CANDIDATE_WINDOW"
  printf 'anchor_stride %s\n' "$CANDIDATE_STRIDE"
  printf 'epoch11_gate_specification_sha256 %s\n' "$EPOCH11_GATE_SPEC_SHA256"
  printf 'terminal_gate_specification_sha256 %s\n' "$TERMINAL_GATE_SPEC_SHA256"
  printf 'candidate_prior_ids_json %s\n' "$CANDIDATE_PRIOR_IDS_JSON"
  printf 'candidate_launch_authorization_required true\n'
  printf 'pose_fingerprint %s\n' "$POSE_FINGERPRINT"
  printf 'pose_recovery_version %s\n' "$POSE_RECOVERY_VERSION"
  printf 'pose_recovery_source_revision %s\n' "$POSE_RECOVERY_SOURCE_REVISION"
  printf 'pose_recovery_container_image_id %s\n' "$POSE_RECOVERY_IMAGE_ID"
  printf 'pose_recovery_config_file_sha256 %s\n' "$POSE_RECOVERY_CONFIG_SHA256"
  printf 'pose_recovery_config_fingerprint %s\n' "$POSE_RECOVERY_CONFIG_FINGERPRINT"
  printf 'pose_recovery_authorization_sha256 %s\n' "$POSE_RECOVERY_AUTHORIZATION_SHA256"
  printf 'pose_recovery_paired_gate_sha256 %s\n' "$POSE_RECOVERY_PAIRED_GATE_SHA256"
  printf 'pose_recovery_run_receipt_sha256 %s\n' "$POSE_RECOVERY_RUN_RECEIPT_SHA256"
  printf 'pose_recovery_ledger_sha256 %s\n' "$POSE_RECOVERY_LEDGER_SHA256"
  printf 'authorized_pose_cache_set_sha256 %s\n' "$AUTHORIZED_POSE_CACHE_SET_SHA256"
  printf 'train_identity_sidecar_sha256 %s\n' "$TRAIN_INPUT_SHA256"
  printf 'train_identity_commitment_sha256 %s\n' "$TRAIN_COMMIT_SHA256"
  printf 'dev_identity_sidecar_sha256 %s\n' "$DEV_INPUT_SHA256"
  printf 'dev_identity_commitment_sha256 %s\n' "$DEV_COMMIT_SHA256"
  printf 'test_identity_sidecar_sha256 %s\n' "$TEST_ID_INPUT_SHA256"
  printf 'test_identity_commitment_sha256 %s\n' "$TEST_ID_COMMIT_SHA256"
} > "${AUDIT_ROOT}/frozen-bindings.txt"
chmod 0444 "${AUDIT_ROOT}/frozen-bindings.txt"

CURRENT_STAGE='hardware-audit'
nvidia-smi -q > "${AUDIT_ROOT}/nvidia-smi-q.txt"
lscpu > "${AUDIT_ROOT}/lscpu.txt"
docker version > "${AUDIT_ROOT}/docker-version.txt"
git --version > "${AUDIT_ROOT}/git-version.txt"
docker image inspect "$IMAGE_ID" > "${AUDIT_ROOT}/image.inspect.json"

for container_name in \
  "$PREFLIGHT_NAME" \
  "$LAUNCH_AUTHORIZATION_NAME" \
  "$EPOCH11_ENCODER_NAME" \
  "$EPOCH11_GATE_NAME" \
  "$FINAL_ENCODER_NAME"; do
  if docker inspect "$container_name" >/dev/null 2>&1; then
    fail "container name already exists: ${container_name}"
  fi
done

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
  --workdir /workspace
  --env HOME=/pams/home
  --env XDG_CACHE_HOME=/pams/cache/xdg
  --env PYTHONPATH=/workspace/src
  --env PYTHONOPTIMIZE=
  --env PYTHONDONTWRITEBYTECODE=1
  --env PAMS_AUDIT_MODE=formal
  --env "PAMS_CONTAINER_IMAGE_ID=${IMAGE_ID}"
  --env "PAMS_CONTAINER_ENVIRONMENT_SHA256=${ENVIRONMENT_SHA256}"
  --env "PAMS_CONTAINER_SOURCE_REVISION=${SOURCE_REVISION}"
  --env PAMS_SOURCE_EXPORT_RECEIPT=/pams/source-export-receipt.json
  --env "PAMS_SOURCE_EXPORT_RECEIPT_SHA256=${SOURCE_RECEIPT_SHA256}"
  --tmpfs /tmp:rw,nosuid,nodev,uid=1000,gid=1000,size=1g
  --tmpfs /pams/tmp:rw,nosuid,nodev,uid=1000,gid=1000,size=8g
  --tmpfs /pams/cache:rw,nosuid,nodev,uid=1000,gid=1000,size=8g
  --tmpfs /pams/home:rw,nosuid,nodev,uid=1000,gid=1000,size=256m
)
source_args=(
  --mount "type=bind,src=${SOURCE_VIEW},dst=/workspace,readonly"
  --mount "type=bind,src=${SOURCE_RECEIPT},dst=/pams/source-export-receipt.json,readonly"
)
protocol_args=(
  --mount "type=bind,src=${CONFIG_HOST},dst=/pams/input/config.yaml,readonly"
  --mount "type=bind,src=${TRAIN_INPUT},dst=/pams/protocol/train.inputs.json,readonly"
  --mount "type=bind,src=${TRAIN_COMMIT},dst=/pams/protocol/train.inputs.commitment.json,readonly"
  --mount "type=bind,src=${DEV_INPUT},dst=/pams/protocol/dev.inputs.json,readonly"
  --mount "type=bind,src=${DEV_COMMIT},dst=/pams/protocol/dev.inputs.commitment.json,readonly"
  --mount "type=bind,src=${TEST_ID_INPUT},dst=/pams/protocol/test-identity.inputs.json,readonly"
  --mount "type=bind,src=${TEST_ID_COMMIT},dst=/pams/protocol/test-identity.inputs.commitment.json,readonly"
)
launch_prior_mount_args=()
launch_prior_cli_args=()
if [[ "$CANDIDATE_ID" == 'B' || "$CANDIDATE_ID" == 'C' ]]; then
  launch_prior_mount_args+=(
    --mount "type=bind,src=${PRIOR_A_REJECTION_ARTIFACT},dst=/pams/prior/A/rejection.json,readonly"
    --mount "type=bind,src=${PRIOR_A_REJECTION_RECEIPT},dst=/pams/prior/A/rejection.json.receipt.json,readonly"
  )
  launch_prior_cli_args+=(
    --prior-rejection-artifact /pams/prior/A/rejection.json
    --prior-rejection-receipt /pams/prior/A/rejection.json.receipt.json
  )
fi
if [[ "$CANDIDATE_ID" == 'C' ]]; then
  launch_prior_mount_args+=(
    --mount "type=bind,src=${PRIOR_B_REJECTION_ARTIFACT},dst=/pams/prior/B/rejection.json,readonly"
    --mount "type=bind,src=${PRIOR_B_REJECTION_RECEIPT},dst=/pams/prior/B/rejection.json.receipt.json,readonly"
  )
  launch_prior_cli_args+=(
    --prior-rejection-artifact /pams/prior/B/rejection.json
    --prior-rejection-receipt /pams/prior/B/rejection.json.receipt.json
  )
fi

CURRENT_STAGE='input-preflight-create'
docker create \
  --name "$PREFLIGHT_NAME" \
  "${common_args[@]}" \
  --env CUDA_VISIBLE_DEVICES= \
  "${source_args[@]}" \
  "${protocol_args[@]}" \
  --mount "type=bind,src=${POSE_RECOVERY_CONFIG},dst=/pams/pose-recovery/config.yaml,readonly" \
  --mount "type=bind,src=${POSE_RECOVERY_AUTHORIZATION},dst=/pams/pose-recovery/native-baseline-authorization.json,readonly" \
  --mount "type=bind,src=${POSE_RECOVERY_PAIRED_GATE},dst=/pams/pose-recovery/paired-gate.json,readonly" \
  --mount "type=bind,src=${POSE_RECOVERY_RUN_RECEIPT},dst=/pams/pose-recovery/run.receipt.json,readonly" \
  --mount "type=bind,src=${POSE_RECOVERY_LEDGER},dst=/pams/pose-recovery/train337.ledger.json,readonly" \
  --mount "type=bind,src=${POSE_RECOVERY_CACHE},dst=/pams/pose-cache,readonly" \
  --mount "type=bind,src=${AUDIT_ROOT},dst=/pams/output" \
  "$IMAGE_ID" \
  python scripts/server/validate_pams_native_baseline_inputs.py \
    --candidate-config /pams/input/config.yaml \
    --pose-recovery-version "$POSE_RECOVERY_VERSION" \
    --pose-recovery-config /pams/pose-recovery/config.yaml \
    --train-sidecar /pams/protocol/train.inputs.json \
    --train-commitment /pams/protocol/train.inputs.commitment.json \
    --dev-identity-sidecar /pams/protocol/dev.inputs.json \
    --dev-identity-commitment /pams/protocol/dev.inputs.commitment.json \
    --test-identity-sidecar /pams/protocol/test-identity.inputs.json \
    --test-identity-commitment /pams/protocol/test-identity.inputs.commitment.json \
    --pose-recovery-authorization /pams/pose-recovery/native-baseline-authorization.json \
    --expected-pose-recovery-authorization-sha256 "$POSE_RECOVERY_AUTHORIZATION_SHA256" \
    --pose-recovery-paired-gate /pams/pose-recovery/paired-gate.json \
    --expected-pose-recovery-paired-gate-sha256 "$POSE_RECOVERY_PAIRED_GATE_SHA256" \
    --pose-recovery-run-receipt /pams/pose-recovery/run.receipt.json \
    --expected-pose-recovery-run-receipt-sha256 "$POSE_RECOVERY_RUN_RECEIPT_SHA256" \
    --pose-recovery-ledger /pams/pose-recovery/train337.ledger.json \
    --train-pose-cache /pams/pose-cache \
    --output /pams/output/input-preflight.json \
  > "${AUDIT_ROOT}/${PREFLIGHT_NAME}.create-id.txt"
verify_container "$PREFLIGHT_NAME" 'preflight'
run_created_container "$PREFLIGHT_NAME" 'input-preflight'
[[ -s "${AUDIT_ROOT}/input-preflight.json" ]] \
  || fail 'input preflight artifact was not created'
jq -e \
  --arg classification "$CLASSIFICATION" \
  --arg candidate_id "$CANDIDATE_ID" \
  --arg config_sha "$CONFIG_SHA256" \
  --arg config_fp "$CONFIG_FINGERPRINT" \
  --arg pose_fp "$POSE_FINGERPRINT" \
  --arg version "$POSE_RECOVERY_VERSION" \
  --arg authorization_sha "$POSE_RECOVERY_AUTHORIZATION_SHA256" \
  --arg gate_sha "$POSE_RECOVERY_PAIRED_GATE_SHA256" \
  --arg receipt_sha "$POSE_RECOVERY_RUN_RECEIPT_SHA256" \
  --arg ledger_sha "$POSE_RECOVERY_LEDGER_SHA256" \
  --arg cache_set_sha "$AUTHORIZED_POSE_CACHE_SET_SHA256" \
  --argjson candidate_window "$CANDIDATE_WINDOW" \
  --argjson candidate_stride "$CANDIDATE_STRIDE" \
  '
    .schema_version == 1
    and .passed == true
    and .classification == $classification
    and .paper_table2_value_claim_eligible == false
    and .candidate.candidate_id == $candidate_id
    and .candidate.config_file_sha256 == $config_sha
    and .candidate.config_fingerprint == $config_fp
    and .candidate.pose_fingerprint == $pose_fp
    and .candidate.encoder_epochs == 150
    and .candidate.physical_batch_size == 32
    and .candidate.fixed_period_frames == $candidate_window
    and .candidate.anchor_stride == $candidate_stride
    and .pose_recovery.version == $version
    and .pose_recovery.authorization_sha256 == $authorization_sha
    and .pose_recovery.paired_gate_sha256 == $gate_sha
    and .pose_recovery.run_receipt_sha256 == $receipt_sha
    and .pose_recovery.ledger_sha256 == $ledger_sha
    and .pose_recovery.pose_cache_set_sha256 == $cache_set_sha
    and .pose_recovery.pose_cache_entry_total == 337
    and .mount_policy.train337_pose_cache == true
    and .mount_policy.dev84_pose_cache == false
    and .mount_policy.test105_pose_cache == false
    and .mount_policy.targets == false
    and .authorization.encoder_training == "train337_only"
    and .authorization.undisclosed_period_head_training == false
    and .authorization.sshead_training == false
    and .authorization.dev_prediction_authorized == false
    and .authorization.dev_scoring_authorized == false
    and .authorization.test_prediction_authorized == false
    and .authorization.test_scoring_authorized == false
  ' "${AUDIT_ROOT}/input-preflight.json" >/dev/null \
  || fail 'input preflight artifact violates the training authorization'
readonly INPUT_PREFLIGHT_SHA256="$(sha256_file "${AUDIT_ROOT}/input-preflight.json")"
readonly POSE_CACHE_SET_SHA256="$(jq -er '.pose_recovery.pose_cache_set_sha256' "${AUDIT_ROOT}/input-preflight.json")"
[[ "$POSE_CACHE_SET_SHA256" == "$AUTHORIZED_POSE_CACHE_SET_SHA256" ]] \
  || fail 'validated pose-cache set differs from the authorization binding'
chmod 0444 "${AUDIT_ROOT}/input-preflight.json"

CURRENT_STAGE='launch-pose-snapshot'
POSE_LEDGER_SOURCE="$POSE_RECOVERY_LEDGER" \
LAUNCH_SNAPSHOT_OUTPUT="$LAUNCH_POSE_SNAPSHOT" \
LAUNCH_SNAPSHOT_POSE_FINGERPRINT="$POSE_FINGERPRINT" \
LAUNCH_SNAPSHOT_CACHE_SET_SHA256="$POSE_CACHE_SET_SHA256" \
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path


def reject_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON field: {key!r}")
        result[key] = value
    return result


ledger = json.loads(
    Path(os.environ["POSE_LEDGER_SOURCE"]).read_text(encoding="utf-8"),
    object_pairs_hook=reject_pairs,
    parse_constant=lambda value: (_ for _ in ()).throw(
        RuntimeError(f"non-finite JSON constant: {value}")
    ),
)
snapshot = ledger.get("successful_cache_snapshot")
if not isinstance(snapshot, dict) or set(snapshot) != {
    "schema_version", "pose_fingerprint", "fingerprint", "entry_count", "entries"
}:
    raise RuntimeError("pose ledger lacks the canonical successful cache snapshot")
if (
    snapshot["schema_version"] != 1
    or snapshot["pose_fingerprint"]
    != os.environ["LAUNCH_SNAPSHOT_POSE_FINGERPRINT"]
    or snapshot["fingerprint"]
    != os.environ["LAUNCH_SNAPSHOT_CACHE_SET_SHA256"]
    or snapshot["entry_count"] != 337
    or not isinstance(snapshot["entries"], list)
    or len(snapshot["entries"]) != 337
):
    raise RuntimeError("pose ledger snapshot does not match validated train337 inputs")
entries = []
for index, value in enumerate(snapshot["entries"]):
    if not isinstance(value, dict) or set(value) != {
        "video_id", "cache_sha256", "bytes"
    }:
        raise RuntimeError(f"pose snapshot entry schema mismatch at row {index}")
    video_id = value["video_id"]
    cache_sha256 = value["cache_sha256"]
    byte_total = value["bytes"]
    if not isinstance(video_id, str) or not video_id.strip():
        raise RuntimeError(f"pose snapshot video ID is invalid at row {index}")
    if (
        not isinstance(cache_sha256, str)
        or len(cache_sha256) != 64
        or any(character not in "0123456789abcdef" for character in cache_sha256)
    ):
        raise RuntimeError(f"pose snapshot SHA-256 is invalid at row {index}")
    if isinstance(byte_total, bool) or not isinstance(byte_total, int) or byte_total < 1:
        raise RuntimeError(f"pose snapshot byte total is invalid at row {index}")
    entries.append({
        "video_id": video_id.strip(),
        "cache_sha256": cache_sha256,
        "bytes": byte_total,
    })
entries.sort(key=lambda value: value["video_id"])
if len({value["video_id"] for value in entries}) != 337:
    raise RuntimeError("pose snapshot video IDs are not unique")
fingerprint_payload = {
    "schema_version": 1,
    "pose_fingerprint": snapshot["pose_fingerprint"],
    "entries": entries,
}
fingerprint = hashlib.sha256(
    json.dumps(
        fingerprint_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
if fingerprint != snapshot["fingerprint"]:
    raise RuntimeError("pose ledger snapshot fingerprint is not canonical")
payload = {
    "schema_version": 1,
    "pose_fingerprint": snapshot["pose_fingerprint"],
    "fingerprint": fingerprint,
    "entry_count": len(entries),
    "entries": entries,
}
with Path(os.environ["LAUNCH_SNAPSHOT_OUTPUT"]).open(
    "x", encoding="utf-8", newline="\n"
) as handle:
    json.dump(
        payload,
        handle,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    handle.write("\n")
PY
chmod 0444 "$LAUNCH_POSE_SNAPSHOT"
readonly LAUNCH_POSE_SNAPSHOT_SHA256="$(sha256_file "$LAUNCH_POSE_SNAPSHOT")"

CURRENT_STAGE='launch-authorization-create'
docker create \
  --name "$LAUNCH_AUTHORIZATION_NAME" \
  "${common_args[@]}" \
  --env CUDA_VISIBLE_DEVICES= \
  "${source_args[@]}" \
  "${launch_prior_mount_args[@]}" \
  --mount "type=bind,src=${LAUNCH_POSE_SNAPSHOT},dst=/pams/input/launch-training-pose-cache-snapshot.json,readonly" \
  --mount "type=bind,src=${LAUNCH_AUTHORIZATION_STAGE},dst=/pams/output" \
  "$IMAGE_ID" \
  python "$LAUNCH_AUTHORIZATION_RELATIVE" \
    --source-receipt /pams/source-export-receipt.json \
    --config "/workspace/${CONFIG_RELATIVE}" \
    --gate-specification "/workspace/${TERMINAL_GATE_SPEC_RELATIVE}" \
    --pose-snapshot /pams/input/launch-training-pose-cache-snapshot.json \
    --candidate-id "$CANDIDATE_ID" \
    "${launch_prior_cli_args[@]}" \
    --output /pams/output/authorization.json \
  > "${AUDIT_ROOT}/${LAUNCH_AUTHORIZATION_NAME}.create-id.txt"
verify_container "$LAUNCH_AUTHORIZATION_NAME" 'launch-authorization'
run_created_container "$LAUNCH_AUTHORIZATION_NAME" 'launch-authorization'
[[ -f "$LAUNCH_AUTHORIZATION_ARTIFACT" && ! -L "$LAUNCH_AUTHORIZATION_ARTIFACT" ]] \
  || fail 'candidate launch authorization artifact is missing or unsafe'
[[ -f "$LAUNCH_AUTHORIZATION_RECEIPT" && ! -L "$LAUNCH_AUTHORIZATION_RECEIPT" ]] \
  || fail 'candidate launch authorization receipt is missing or unsafe'
readonly LAUNCH_AUTHORIZATION_SHA256="$(sha256_file "$LAUNCH_AUTHORIZATION_ARTIFACT")"
readonly LAUNCH_AUTHORIZATION_BYTES="$(stat -c '%s' -- "$LAUNCH_AUTHORIZATION_ARTIFACT")"
readonly LAUNCH_AUTHORIZATION_RECEIPT_SHA256="$(sha256_file "$LAUNCH_AUTHORIZATION_RECEIPT")"
jq -e \
  --arg candidate_id "$CANDIDATE_ID" \
  --arg config_sha "$CONFIG_SHA256" \
  --arg config_fp "$CONFIG_FINGERPRINT" \
  --arg gate_sha "$TERMINAL_GATE_SPEC_SHA256" \
  --arg snapshot_sha "$LAUNCH_POSE_SNAPSHOT_SHA256" \
  --arg cache_set_sha "$POSE_CACHE_SET_SHA256" \
  --arg source_receipt_sha "$SOURCE_RECEIPT_SHA256" \
  --arg source_revision "$SOURCE_REVISION" \
  --argjson prior_ids "$CANDIDATE_PRIOR_IDS_JSON" \
  '
    .schema_version == 1
    and .artifact_type == "pams_native_candidate_train337_launch_authorization_v1"
    and .status == "candidate_training_authorized"
    and .candidate.id == $candidate_id
    and (.candidate.prior_scientific_rejections | map(.candidate_id)) == $prior_ids
    and .inputs.experiment_config_sha256 == $config_sha
    and .inputs.config_fingerprint == $config_fp
    and .inputs.gate_specification_sha256 == $gate_sha
    and .inputs.pose_snapshot_sha256 == $snapshot_sha
    and .inputs.pose_cache_set_sha256 == $cache_set_sha
    and .inputs.source_export_receipt_sha256 == $source_receipt_sha
    and .inputs.source_git_sha == $source_revision
    and .inputs.training_video_total == 337
    and .authorization.gate_frozen_before_candidate_a == true
    and .authorization.candidate_training_authorized == true
    and .authorization.encoder_training_scope == "train337_only"
    and .authorization.terminal_checkpoint_or_prediction_authorized == false
    and .authorization.dev84_identity_media_pose_or_scoring_authorized == false
    and .authorization.test105_evaluation_authorized == false
    and .authorization.aggregate_only_prior_receipts == true
  ' "$LAUNCH_AUTHORIZATION_ARTIFACT" >/dev/null \
  || fail 'candidate launch authorization violates the frozen policy'
jq -e \
  --arg artifact_sha "$LAUNCH_AUTHORIZATION_SHA256" \
  --argjson artifact_bytes "$LAUNCH_AUTHORIZATION_BYTES" \
  --arg candidate_id "$CANDIDATE_ID" \
  --arg config_sha "$CONFIG_SHA256" \
  --arg gate_sha "$TERMINAL_GATE_SPEC_SHA256" \
  --arg snapshot_sha "$LAUNCH_POSE_SNAPSHOT_SHA256" \
  --arg cache_set_sha "$POSE_CACHE_SET_SHA256" \
  --arg source_receipt_sha "$SOURCE_RECEIPT_SHA256" \
  --arg source_revision "$SOURCE_REVISION" \
  '
    .schema_version == 1
    and .artifact_type == "pams_native_candidate_train337_launch_authorization_receipt_v1"
    and .artifact_locator == "authorization.json"
    and .artifact_sha256 == $artifact_sha
    and .artifact_bytes == $artifact_bytes
    and .artifact_status == "candidate_training_authorized"
    and .candidate_id == $candidate_id
    and .experiment_config_sha256 == $config_sha
    and .gate_specification_sha256 == $gate_sha
    and .pose_snapshot_sha256 == $snapshot_sha
    and .pose_cache_set_sha256 == $cache_set_sha
    and .source_export_receipt_sha256 == $source_receipt_sha
    and .source_git_sha == $source_revision
    and .candidate_training_authorized == true
    and .dev84_pose_or_scoring_authorized == false
    and .test105_evaluation_authorized == false
  ' "$LAUNCH_AUTHORIZATION_RECEIPT" >/dev/null \
  || fail 'candidate launch authorization receipt is malformed or unbound'
sha256sum -- \
  "$LAUNCH_POSE_SNAPSHOT" \
  "$LAUNCH_AUTHORIZATION_ARTIFACT" \
  "$LAUNCH_AUTHORIZATION_RECEIPT" \
  > "${AUDIT_ROOT}/launch-authorization-sha256.txt"
chmod -R a-w -- "$LAUNCH_AUTHORIZATION_STAGE"

mkdir -p -- "$LOCK_ROOT"
[[ ! -L "$LOCK_ROOT" ]] || fail 'GPU lock root must not be a symlink'
[[ "$(realpath -e -- "$LOCK_ROOT")" == "${ROOT}/.pams-gpu-locks" ]] \
  || fail 'GPU lock root resolves outside PAMS_ROOT'
[[ ! -L "$LOCK_PATH" ]] || fail 'GPU lock file must not be a symlink'
exec 9>"$LOCK_PATH"
chmod 0600 "$LOCK_PATH"
if ! flock -n 9; then
  CURRENT_STAGE='gpu-lock'
  write_status 'failed' "$CURRENT_STAGE" '75'
  exit 75
fi

encoder_cli_args=(
  --config /pams/input/config.yaml
  --device cuda:0
  --microbatch-size 32
  --label-free-inputs
  --input-commitment /pams/protocol/train.inputs.commitment.json
  --dev-inputs /pams/protocol/dev.inputs.json
  --dev-input-commitment /pams/protocol/dev.inputs.commitment.json
  --test-identity-inputs /pams/protocol/test-identity.inputs.json
  --test-identity-commitment /pams/protocol/test-identity.inputs.commitment.json
  --candidate-launch-authorization /pams/launch/authorization.json
  --candidate-launch-receipt /pams/launch/authorization.json.receipt.json
)

require_sha256 \
  "$LAUNCH_AUTHORIZATION_ARTIFACT" \
  "$LAUNCH_AUTHORIZATION_SHA256" \
  'candidate launch authorization before epoch11 training'
require_sha256 \
  "$LAUNCH_AUTHORIZATION_RECEIPT" \
  "$LAUNCH_AUTHORIZATION_RECEIPT_SHA256" \
  'candidate launch authorization receipt before epoch11 training'
CURRENT_STAGE='encoder-epoch11-create'
docker create \
  --name "$EPOCH11_ENCODER_NAME" \
  --gpus "device=${GPU_DEVICE}" \
  "${common_args[@]}" \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "${source_args[@]}" \
  "${protocol_args[@]}" \
  --mount "type=bind,src=${LAUNCH_AUTHORIZATION_ARTIFACT},dst=/pams/launch/authorization.json,readonly" \
  --mount "type=bind,src=${LAUNCH_AUTHORIZATION_RECEIPT},dst=/pams/launch/authorization.json.receipt.json,readonly" \
  --mount "type=bind,src=${POSE_RECOVERY_CACHE},dst=/pams/pose-cache,readonly" \
  --mount "type=bind,src=${EPOCH11_ENCODER_STAGE},dst=/pams/output" \
  "$IMAGE_ID" \
  python -m pams train encoder \
    /pams/protocol/train.inputs.json \
    /pams/pose-cache \
    /pams/output/run \
    --epochs 11 \
    "${encoder_cli_args[@]}" \
  > "${AUDIT_ROOT}/${EPOCH11_ENCODER_NAME}.create-id.txt"
verify_container "$EPOCH11_ENCODER_NAME" 'encoder-epoch11'
run_created_container "$EPOCH11_ENCODER_NAME" 'encoder-epoch11'

readonly EPOCH11_ENCODER_RECEIPT="$(single_completion_receipt "$EPOCH11_ENCODER_STAGE")"
[[ -s "$EPOCH11_ENCODER_CHECKPOINT" ]] || fail 'epoch11 encoder checkpoint is missing'
[[ -s "$EPOCH11_ENCODER_PROGRESS" ]] || fail 'epoch11 encoder progress is missing'
[[ -s "$EPOCH11_POSE_SNAPSHOT" ]] || fail 'epoch11 pose snapshot is missing'
require_sha256 \
  "$EPOCH11_POSE_SNAPSHOT" \
  "$LAUNCH_POSE_SNAPSHOT_SHA256" \
  'epoch11 pose snapshot bound by launch authorization'
validate_completion_receipt \
  "$EPOCH11_ENCODER_RECEIPT" \
  "$EPOCH11_ENCODER_CHECKPOINT" \
  "$EPOCH11_ENCODER_PROGRESS" \
  "$POSE_CACHE_SET_SHA256" \
  11
readonly EPOCH11_CHECKPOINT_SHA256="$(sha256_file "$EPOCH11_ENCODER_CHECKPOINT")"
readonly EPOCH11_PROGRESS_SHA256="$(sha256_file "$EPOCH11_ENCODER_PROGRESS")"
readonly EPOCH11_SNAPSHOT_SHA256="$(sha256_file "$EPOCH11_POSE_SNAPSHOT")"
readonly EPOCH11_COMPLETION_RECEIPT_SHA256="$(
  sha256_file "$EPOCH11_ENCODER_RECEIPT"
)"
sha256sum -- \
  "$EPOCH11_ENCODER_CHECKPOINT" \
  "$EPOCH11_ENCODER_PROGRESS" \
  "$EPOCH11_POSE_SNAPSHOT" \
  "$EPOCH11_ENCODER_RECEIPT" \
  > "${AUDIT_ROOT}/encoder-epoch11-sha256.txt"
chmod -R a-w -- "$EPOCH11_ENCODER_STAGE"

CURRENT_STAGE='epoch11-gate-create'
mkdir -- "$EPOCH11_GATE_STAGE"
docker create \
  --name "$EPOCH11_GATE_NAME" \
  --gpus "device=${GPU_DEVICE}" \
  "${common_args[@]}" \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "${source_args[@]}" \
  --mount "type=bind,src=${POSE_RECOVERY_CACHE},dst=/pams/pose-cache,readonly" \
  --mount "type=bind,src=${EPOCH11_ENCODER_CHECKPOINT},dst=/pams/epoch11/encoder.pt,readonly" \
  --mount "type=bind,src=${EPOCH11_ENCODER_PROGRESS},dst=/pams/epoch11/encoder.jsonl,readonly" \
  --mount "type=bind,src=${EPOCH11_ENCODER_RECEIPT},dst=/pams/epoch11/completion.receipt.json,readonly" \
  --mount "type=bind,src=${EPOCH11_POSE_SNAPSHOT},dst=/pams/epoch11/training-pose-cache-snapshot.json,readonly" \
  --mount "type=bind,src=${EPOCH11_GATE_STAGE},dst=/pams/output" \
  "$IMAGE_ID" \
  python "$EPOCH11_GATE_RELATIVE" \
    --encoder-checkpoint /pams/epoch11/encoder.pt \
    --encoder-progress /pams/epoch11/encoder.jsonl \
    --encoder-completion-receipt /pams/epoch11/completion.receipt.json \
    --expected-encoder-completion-receipt-sha256 "$EPOCH11_COMPLETION_RECEIPT_SHA256" \
    --config "/workspace/${CONFIG_RELATIVE}" \
    --gate-specification "/workspace/${EPOCH11_GATE_SPEC_RELATIVE}" \
    --pose-cache-dir /pams/pose-cache \
    --pose-snapshot /pams/epoch11/training-pose-cache-snapshot.json \
    --output /pams/output/gate.json \
    --device cuda:0 \
    --batch-size 16 \
  > "${AUDIT_ROOT}/${EPOCH11_GATE_NAME}.create-id.txt"
verify_container "$EPOCH11_GATE_NAME" 'epoch11-gate'
run_gate_container "$EPOCH11_GATE_NAME"
[[ -s "$EPOCH11_GATE_ARTIFACT" ]] || fail 'epoch11 gate artifact is missing'
[[ -s "$EPOCH11_GATE_RECEIPT" ]] || fail 'epoch11 gate receipt is missing'
validate_epoch11_gate_artifacts \
  "$GATE_CONTAINER_EXIT" \
  "$EPOCH11_CHECKPOINT_SHA256" \
  "$EPOCH11_PROGRESS_SHA256" \
  "$EPOCH11_COMPLETION_RECEIPT_SHA256" \
  "$EPOCH11_SNAPSHOT_SHA256" \
  "$POSE_CACHE_SET_SHA256" \
  "$EPOCH11_GATE_SPEC_SHA256"
readonly EPOCH11_GATE_ARTIFACT_SHA256="$(sha256_file "$EPOCH11_GATE_ARTIFACT")"
readonly EPOCH11_GATE_RECEIPT_SHA256="$(sha256_file "$EPOCH11_GATE_RECEIPT")"
sha256sum -- "$EPOCH11_GATE_ARTIFACT" "$EPOCH11_GATE_RECEIPT" \
  > "${AUDIT_ROOT}/epoch11-gate-sha256.txt"
chmod -R a-w -- "$EPOCH11_GATE_STAGE"

if [[ "$GATE_CONTAINER_EXIT" -eq 3 ]]; then
  [[ ! -e "$FINAL_ENCODER_STAGE" ]] \
    || fail 'resume stage exists despite a rejected epoch11 gate'
  CURRENT_STAGE='epoch11-gate-rejected'
  REJECTION_PATH="${AUDIT_ROOT}/continuation-rejected.json" \
  REJECTION_ATTEMPT_ID="$ATTEMPT_ID" \
  REJECTION_CANDIDATE_ID="$CANDIDATE_ID" \
  REJECTION_WINDOW="$CANDIDATE_WINDOW" \
  REJECTION_STRIDE="$CANDIDATE_STRIDE" \
  REJECTION_EPOCH11_CHECKPOINT_SHA256="$EPOCH11_CHECKPOINT_SHA256" \
  REJECTION_EPOCH11_PROGRESS_SHA256="$EPOCH11_PROGRESS_SHA256" \
  REJECTION_GATE_ARTIFACT_SHA256="$EPOCH11_GATE_ARTIFACT_SHA256" \
  REJECTION_GATE_RECEIPT_SHA256="$EPOCH11_GATE_RECEIPT_SHA256" \
  python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "schema_version": 1,
    "artifact_type": "pams_native_table2_proxy_continuation_rejection_receipt",
    "status": "encoder_continuation_rejected",
    "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "attempt_id": os.environ["REJECTION_ATTEMPT_ID"],
    "candidate_id": os.environ["REJECTION_CANDIDATE_ID"],
    "fixed_period_frames": int(os.environ["REJECTION_WINDOW"]),
    "anchor_stride": int(os.environ["REJECTION_STRIDE"]),
    "epoch11_encoder_checkpoint_sha256": os.environ[
        "REJECTION_EPOCH11_CHECKPOINT_SHA256"
    ],
    "epoch11_encoder_progress_sha256": os.environ[
        "REJECTION_EPOCH11_PROGRESS_SHA256"
    ],
    "epoch11_gate_artifact_sha256": os.environ[
        "REJECTION_GATE_ARTIFACT_SHA256"
    ],
    "epoch11_gate_receipt_sha256": os.environ["REJECTION_GATE_RECEIPT_SHA256"],
    "resume_stage_created": False,
    "encoder_continuation_authorized": False,
    "dev_or_test_access_authorized": False,
}
with Path(os.environ["REJECTION_PATH"]).open(
    "x", encoding="utf-8", newline="\n"
) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
PY
  chmod 0444 "${AUDIT_ROOT}/continuation-rejected.json"
  write_status 'failed' 'epoch11-gate-rejected' '3'
  flock -u 9
  find "$RUN_ROOT" -type f \
    -not -path "${AUDIT_ROOT}/artifact-sha256.txt" \
    -print0 | sort -z | xargs -0 sha256sum \
    > "${AUDIT_ROOT}/artifact-sha256.txt"
  chmod -R a-w -- "$RUN_ROOT"
  RUN_RESERVED=0
  exit 3
fi

[[ "$GATE_CONTAINER_EXIT" -eq 0 ]] \
  || fail 'unreachable epoch11 gate exit status'
jq -e \
  '.gate.encoder_continuation_authorized == true
   and .status == "encoder_continuation_authorized"' \
  "$EPOCH11_GATE_ARTIFACT" >/dev/null \
  || fail 'epoch11 gate did not authorize encoder continuation'
[[ ! -e "$FINAL_ENCODER_STAGE" ]] \
  || fail 'resume stage was created before gate authorization'
require_sha256 \
  "$LAUNCH_AUTHORIZATION_ARTIFACT" \
  "$LAUNCH_AUTHORIZATION_SHA256" \
  'candidate launch authorization before final training'
require_sha256 \
  "$LAUNCH_AUTHORIZATION_RECEIPT" \
  "$LAUNCH_AUTHORIZATION_RECEIPT_SHA256" \
  'candidate launch authorization receipt before final training'
mkdir -- "$FINAL_ENCODER_STAGE"

CURRENT_STAGE='encoder-final-create'
docker create \
  --name "$FINAL_ENCODER_NAME" \
  --gpus "device=${GPU_DEVICE}" \
  "${common_args[@]}" \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "${source_args[@]}" \
  "${protocol_args[@]}" \
  --mount "type=bind,src=${LAUNCH_AUTHORIZATION_ARTIFACT},dst=/pams/launch/authorization.json,readonly" \
  --mount "type=bind,src=${LAUNCH_AUTHORIZATION_RECEIPT},dst=/pams/launch/authorization.json.receipt.json,readonly" \
  --mount "type=bind,src=${POSE_RECOVERY_CACHE},dst=/pams/pose-cache,readonly" \
  --mount "type=bind,src=${EPOCH11_ENCODER_CHECKPOINT},dst=/pams/resume/encoder.pt,readonly" \
  --mount "type=bind,src=${EPOCH11_ENCODER_PROGRESS},dst=/pams/resume/encoder.jsonl,readonly" \
  --mount "type=bind,src=${FINAL_ENCODER_STAGE},dst=/pams/output" \
  "$IMAGE_ID" \
  python -m pams train encoder \
    /pams/protocol/train.inputs.json \
    /pams/pose-cache \
    /pams/output/run \
    --epochs 150 \
    --resume \
    --resume-checkpoint /pams/resume/encoder.pt \
    --resume-progress /pams/resume/encoder.jsonl \
    "${encoder_cli_args[@]}" \
  > "${AUDIT_ROOT}/${FINAL_ENCODER_NAME}.create-id.txt"
verify_container "$FINAL_ENCODER_NAME" 'encoder-final'
run_created_container "$FINAL_ENCODER_NAME" 'encoder-final'
flock -u 9

readonly FINAL_ENCODER_RECEIPT="$(single_completion_receipt "$FINAL_ENCODER_STAGE")"
[[ -s "$FINAL_ENCODER_CHECKPOINT" ]] || fail 'final encoder checkpoint is missing'
[[ -s "$FINAL_ENCODER_PROGRESS" ]] || fail 'final encoder progress is missing'
[[ -s "$FINAL_ENCODER_POSE_SNAPSHOT" ]] || fail 'final encoder pose snapshot is missing'
require_sha256 \
  "$FINAL_ENCODER_POSE_SNAPSHOT" \
  "$LAUNCH_POSE_SNAPSHOT_SHA256" \
  'final pose snapshot bound by launch authorization'
validate_completion_receipt \
  "$FINAL_ENCODER_RECEIPT" \
  "$FINAL_ENCODER_CHECKPOINT" \
  "$FINAL_ENCODER_PROGRESS" \
  "$POSE_CACHE_SET_SHA256" \
  150 \
  "$EPOCH11_ENCODER_CHECKPOINT" \
  "$EPOCH11_ENCODER_PROGRESS"
readonly FINAL_CHECKPOINT_SHA256="$(sha256_file "$FINAL_ENCODER_CHECKPOINT")"
readonly FINAL_PROGRESS_SHA256="$(sha256_file "$FINAL_ENCODER_PROGRESS")"
readonly FINAL_POSE_SNAPSHOT_SHA256="$(sha256_file "$FINAL_ENCODER_POSE_SNAPSHOT")"
readonly FINAL_COMPLETION_RECEIPT_SHA256="$(
  sha256_file "$FINAL_ENCODER_RECEIPT"
)"
require_sha256 \
  "$LAUNCH_AUTHORIZATION_ARTIFACT" \
  "$LAUNCH_AUTHORIZATION_SHA256" \
  'candidate launch authorization after final training'
require_sha256 \
  "$LAUNCH_AUTHORIZATION_RECEIPT" \
  "$LAUNCH_AUTHORIZATION_RECEIPT_SHA256" \
  'candidate launch authorization receipt after final training'
validate_resume_lineage \
  "$EPOCH11_CHECKPOINT_SHA256" \
  "$EPOCH11_PROGRESS_SHA256" \
  "$EPOCH11_SNAPSHOT_SHA256" \
  "$EPOCH11_GATE_ARTIFACT_SHA256" \
  "$EPOCH11_GATE_RECEIPT_SHA256"
sha256sum -- \
  "$FINAL_ENCODER_CHECKPOINT" \
  "$FINAL_ENCODER_PROGRESS" \
  "$FINAL_ENCODER_POSE_SNAPSHOT" \
  "$FINAL_ENCODER_RECEIPT" \
  > "${AUDIT_ROOT}/encoder-final-sha256.txt"
chmod -R a-w -- "$FINAL_ENCODER_STAGE"

CURRENT_STAGE='final-receipt'
RUN_RECEIPT_PATH="${AUDIT_ROOT}/run.receipt.json" \
RUN_ATTEMPT_ID="$ATTEMPT_ID" \
RUN_CANDIDATE_ID="$CANDIDATE_ID" \
RUN_CANDIDATE_WINDOW="$CANDIDATE_WINDOW" \
RUN_CANDIDATE_STRIDE="$CANDIDATE_STRIDE" \
RUN_SOURCE_REVISION="$SOURCE_REVISION" \
RUN_SOURCE_RECEIPT_SHA256="$SOURCE_RECEIPT_SHA256" \
RUN_IMAGE_ID="$IMAGE_ID" \
RUN_ENVIRONMENT_SHA256="$ENVIRONMENT_SHA256" \
RUN_CONFIG_SHA256="$CONFIG_SHA256" \
RUN_CONFIG_FINGERPRINT="$CONFIG_FINGERPRINT" \
RUN_POSE_FINGERPRINT="$POSE_FINGERPRINT" \
RUN_POSE_CACHE_SET_SHA256="$POSE_CACHE_SET_SHA256" \
RUN_POSE_VERSION="$POSE_RECOVERY_VERSION" \
RUN_POSE_SOURCE_REVISION="$POSE_RECOVERY_SOURCE_REVISION" \
RUN_POSE_IMAGE_ID="$POSE_RECOVERY_IMAGE_ID" \
RUN_POSE_CONFIG_SHA256="$POSE_RECOVERY_CONFIG_SHA256" \
RUN_POSE_CONFIG_FINGERPRINT="$POSE_RECOVERY_CONFIG_FINGERPRINT" \
RUN_POSE_AUTHORIZATION_SHA256="$POSE_RECOVERY_AUTHORIZATION_SHA256" \
RUN_POSE_GATE_SHA256="$POSE_RECOVERY_PAIRED_GATE_SHA256" \
RUN_POSE_RECEIPT_SHA256="$POSE_RECOVERY_RUN_RECEIPT_SHA256" \
RUN_POSE_LEDGER_SHA256="$POSE_RECOVERY_LEDGER_SHA256" \
RUN_PREFLIGHT_SHA256="$INPUT_PREFLIGHT_SHA256" \
RUN_TERMINAL_GATE_SPECIFICATION_SHA256="$TERMINAL_GATE_SPEC_SHA256" \
RUN_LAUNCH_POSE_SNAPSHOT_SHA256="$LAUNCH_POSE_SNAPSHOT_SHA256" \
RUN_LAUNCH_AUTHORIZATION_SHA256="$LAUNCH_AUTHORIZATION_SHA256" \
RUN_LAUNCH_AUTHORIZATION_RECEIPT_SHA256="$LAUNCH_AUTHORIZATION_RECEIPT_SHA256" \
RUN_GATE_SPECIFICATION_SHA256="$EPOCH11_GATE_SPEC_SHA256" \
RUN_EPOCH11_CHECKPOINT_SHA256="$EPOCH11_CHECKPOINT_SHA256" \
RUN_EPOCH11_PROGRESS_SHA256="$EPOCH11_PROGRESS_SHA256" \
RUN_EPOCH11_SNAPSHOT_SHA256="$EPOCH11_SNAPSHOT_SHA256" \
RUN_EPOCH11_COMPLETION_RECEIPT_SHA256="$EPOCH11_COMPLETION_RECEIPT_SHA256" \
RUN_GATE_ARTIFACT_SHA256="$EPOCH11_GATE_ARTIFACT_SHA256" \
RUN_GATE_RECEIPT_SHA256="$EPOCH11_GATE_RECEIPT_SHA256" \
RUN_FINAL_CHECKPOINT_SHA256="$FINAL_CHECKPOINT_SHA256" \
RUN_FINAL_PROGRESS_SHA256="$FINAL_PROGRESS_SHA256" \
RUN_FINAL_POSE_SNAPSHOT_SHA256="$FINAL_POSE_SNAPSHOT_SHA256" \
RUN_FINAL_COMPLETION_RECEIPT_SHA256="$FINAL_COMPLETION_RECEIPT_SHA256" \
python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "schema_version": 1,
    "artifact_type": "pams_native_table2_baseline_proxy_train337_run_receipt",
    "status": "completed",
    "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "attempt_id": os.environ["RUN_ATTEMPT_ID"],
    "classification": "independently_inferred_proxy_not_author_table2_baseline",
    "paper_table2_value_claim_eligible": False,
    "protocol": "ucfrep_526",
    "seed": 2026,
    "candidate_id": os.environ["RUN_CANDIDATE_ID"],
    "fixed_period_frames": int(os.environ["RUN_CANDIDATE_WINDOW"]),
    "anchor_stride": int(os.environ["RUN_CANDIDATE_STRIDE"]),
    "source_revision": os.environ["RUN_SOURCE_REVISION"],
    "source_export_receipt_sha256": os.environ["RUN_SOURCE_RECEIPT_SHA256"],
    "container_image_id": os.environ["RUN_IMAGE_ID"],
    "container_environment_sha256": os.environ["RUN_ENVIRONMENT_SHA256"],
    "config_file_sha256": os.environ["RUN_CONFIG_SHA256"],
    "config_fingerprint": os.environ["RUN_CONFIG_FINGERPRINT"],
    "pose_fingerprint": os.environ["RUN_POSE_FINGERPRINT"],
    "pose_cache_set_sha256": os.environ["RUN_POSE_CACHE_SET_SHA256"],
    "upstream_pose_recovery": {
        "version": os.environ["RUN_POSE_VERSION"],
        "source_revision": os.environ["RUN_POSE_SOURCE_REVISION"],
        "container_image_id": os.environ["RUN_POSE_IMAGE_ID"],
        "config_file_sha256": os.environ["RUN_POSE_CONFIG_SHA256"],
        "config_fingerprint": os.environ["RUN_POSE_CONFIG_FINGERPRINT"],
        "authorization_sha256": os.environ["RUN_POSE_AUTHORIZATION_SHA256"],
        "paired_gate_sha256": os.environ["RUN_POSE_GATE_SHA256"],
        "run_receipt_sha256": os.environ["RUN_POSE_RECEIPT_SHA256"],
        "ledger_sha256": os.environ["RUN_POSE_LEDGER_SHA256"],
    },
    "input_preflight_sha256": os.environ["RUN_PREFLIGHT_SHA256"],
    "candidate_launch_authorization": {
        "terminal_gate_specification_sha256": os.environ[
            "RUN_TERMINAL_GATE_SPECIFICATION_SHA256"
        ],
        "pose_snapshot_sha256": os.environ[
            "RUN_LAUNCH_POSE_SNAPSHOT_SHA256"
        ],
        "authorization_sha256": os.environ[
            "RUN_LAUNCH_AUTHORIZATION_SHA256"
        ],
        "authorization_receipt_sha256": os.environ[
            "RUN_LAUNCH_AUTHORIZATION_RECEIPT_SHA256"
        ],
        "generated_before_encoder_container_creation": True,
        "consumed_by_epoch11_and_final_encoder_commands": True,
    },
    "epoch11_gate": {
        "gate_epoch": 11,
        "gate_specification_sha256": os.environ[
            "RUN_GATE_SPECIFICATION_SHA256"
        ],
        "encoder_checkpoint_sha256": os.environ[
            "RUN_EPOCH11_CHECKPOINT_SHA256"
        ],
        "encoder_progress_sha256": os.environ["RUN_EPOCH11_PROGRESS_SHA256"],
        "pose_snapshot_sha256": os.environ["RUN_EPOCH11_SNAPSHOT_SHA256"],
        "encoder_completion_receipt_sha256": os.environ[
            "RUN_EPOCH11_COMPLETION_RECEIPT_SHA256"
        ],
        "gate_artifact_sha256": os.environ["RUN_GATE_ARTIFACT_SHA256"],
        "gate_receipt_sha256": os.environ["RUN_GATE_RECEIPT_SHA256"],
        "encoder_continuation_authorized": True,
        "gate_exit_code": 0,
        "read_only": True,
    },
    "final_encoder": {
        "checkpoint_sha256": os.environ["RUN_FINAL_CHECKPOINT_SHA256"],
        "progress_sha256": os.environ["RUN_FINAL_PROGRESS_SHA256"],
        "pose_snapshot_sha256": os.environ["RUN_FINAL_POSE_SNAPSHOT_SHA256"],
        "completion_receipt_sha256": os.environ[
            "RUN_FINAL_COMPLETION_RECEIPT_SHA256"
        ],
        "completed_epochs": 150,
        "immutable_resume_checkpoint_sha256": os.environ[
            "RUN_EPOCH11_CHECKPOINT_SHA256"
        ],
        "immutable_resume_progress_sha256": os.environ[
            "RUN_EPOCH11_PROGRESS_SHA256"
        ],
        "progress_has_exact_epoch11_prefix": True,
        "resume_stage_created_after_gate_authorization": True,
    },
    "physical_batch_size": 32,
    "encoder_training_scope": "train337_only",
    "period_head_training_authorized": False,
    "sshead_training_authorized": False,
    "dev_authorized": False,
    "dev_pose_mounted": False,
    "dev_targets_mounted": False,
    "dev_prediction_authorized": False,
    "dev_scoring_authorized": False,
    "test_authorized": False,
    "test_pose_mounted": False,
    "test_targets_mounted": False,
    "test_prediction_authorized": False,
    "test_scoring_authorized": False,
    "identity_only_dev_test_sidecars_required_by_cli_provenance": True,
}
with Path(os.environ["RUN_RECEIPT_PATH"]).open(
    "x", encoding="utf-8", newline="\n"
) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
PY
chmod 0444 "${AUDIT_ROOT}/run.receipt.json"
write_status 'completed' 'encoder-final' '0'
find "$RUN_ROOT" -type f -not -path "${AUDIT_ROOT}/artifact-sha256.txt" \
  -print0 | sort -z | xargs -0 sha256sum \
  > "${AUDIT_ROOT}/artifact-sha256.txt"
chmod -R a-w -- "$RUN_ROOT"
RUN_RESERVED=0
printf '%s\n' "$RUN_ROOT"
