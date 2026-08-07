#!/usr/bin/env bash
# Formal train337-only runner for an independently inferred Table-2 proxy.
#
# This launcher trains one encoder and stops. Dev84/test105 identity sidecars
# are mounted only because the label-free CLI requires their frozen protocol
# identities. No dev/test pose cache, target, prediction, scoring, SSHead, or
# separately trained period head is authorized here.
set -Eeuo pipefail
IFS=$'\n\t'
umask 027

fail() {
  printf 'native-table2-proxy-v1: %s\n' "$*" >&2
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
readonly ATTEMPT_ID="${PAMS_ATTEMPT_ID:?PAMS_ATTEMPT_ID is required}"
readonly GPU_DEVICE="${PAMS_GPU_DEVICE:-0}"

readonly CONFIG_RELATIVE='configs/experiments/pams_native_table2_baseline_proxy_v1.yaml'
readonly VALIDATOR_RELATIVE='scripts/server/validate_pams_native_baseline_inputs.py'
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
readonly OFFICIAL_ROOT="${ROOT}/runs/official-segment-v1/15cc1ec3c1d2-20260805T063653Z"
readonly RUN_PARENT="${ROOT}/runs/pams-native-table2-baseline-v1"
readonly RUN_ROOT="${RUN_PARENT}/${ATTEMPT_ID}"
readonly SOURCE_VIEW="${RUN_ROOT}/source"
readonly SOURCE_RECEIPT="${RUN_ROOT}/source-export.receipt.json"
readonly INPUT_ROOT="${RUN_ROOT}/inputs"
readonly CONFIG_HOST="${INPUT_ROOT}/pams_native_table2_baseline_proxy_v1.yaml"
readonly ENCODER_STAGE="${RUN_ROOT}/stages/encoder"
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
readonly ENCODER_NAME="pams-native-proxy-encoder-${ATTEMPT_ID}"

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
  VERIFY_ENCODER_STAGE="$ENCODER_STAGE" \
  VERIFY_AUDIT_ROOT="$AUDIT_ROOT" \
  VERIFY_IMAGE_ID="$IMAGE_ID" \
  VERIFY_SOURCE_REVISION="$SOURCE_REVISION" \
  VERIFY_ENVIRONMENT_SHA256="$ENVIRONMENT_SHA256" \
  VERIFY_SOURCE_RECEIPT_SHA256="$SOURCE_RECEIPT_SHA256" \
  VERIFY_GPU_DEVICE="$GPU_DEVICE" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

stage = os.environ["VERIFY_STAGE"]
item = json.loads(Path(os.environ["VERIFY_INSPECT"]).read_text(encoding="utf-8"))[0]
config = item["Config"]
host = item["HostConfig"]
mounts = {
    row["Destination"]: (row["Source"], bool(row["RW"]))
    for row in item["Mounts"]
}
assert config["Image"] == os.environ["VERIFY_IMAGE_ID"]
assert config["User"] == "1000:1000"
assert config["WorkingDir"] == "/workspace"
assert host["NetworkMode"] == "none"
assert host["ReadonlyRootfs"] is True
assert "ALL" in (host.get("CapDrop") or [])
assert "no-new-privileges:true" in (host.get("SecurityOpt") or [])
assert int(host["PidsLimit"]) == 4096
assert int(host["Memory"]) == 96 * 1024**3
assert set(host.get("Tmpfs") or {}) == {
    "/tmp", "/pams/tmp", "/pams/cache", "/pams/home"
}

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
elif stage == "encoder":
    expected = {
        **source,
        **protocol,
        "/pams/pose-cache": (pose_recovery_root + "/pose-cache", False),
        "/pams/output": (os.environ["VERIFY_ENCODER_STAGE"], True),
    }
else:
    raise AssertionError(stage)
assert mounts == expected, (stage, mounts, expected)

command = "\0".join(config.get("Cmd") or [])
all_text = "\0".join([command, *mounts, *(row[0] for row in mounts.values())])
for forbidden in (
    ".targets", "/dev-pose", "/test-pose", "--include-dev",
    "train sshead", "evaluate", "dev-predict", "dev-score",
):
    assert forbidden not in all_text, forbidden
if stage == "encoder":
    assert "python\0-m\0pams\0train\0encoder" in command
    assert command.count("train\0encoder") == 1
else:
    assert "validate_pams_native_baseline_inputs.py" in command

environment = set(config.get("Env") or [])
assert "PYTHONPATH=/workspace/src" in environment
assert "PAMS_AUDIT_MODE=formal" in environment
assert f'PAMS_CONTAINER_IMAGE_ID={os.environ["VERIFY_IMAGE_ID"]}' in environment
assert f'PAMS_CONTAINER_SOURCE_REVISION={os.environ["VERIFY_SOURCE_REVISION"]}' in environment
assert (
    "PAMS_CONTAINER_ENVIRONMENT_SHA256="
    + os.environ["VERIFY_ENVIRONMENT_SHA256"]
) in environment
assert "PAMS_SOURCE_EXPORT_RECEIPT=/pams/source-export-receipt.json" in environment
assert (
    "PAMS_SOURCE_EXPORT_RECEIPT_SHA256="
    + os.environ["VERIFY_SOURCE_RECEIPT_SHA256"]
) in environment

devices = host.get("DeviceRequests") or []
if stage == "encoder":
    assert len(devices) == 1
    assert "gpu" in devices[0]["Capabilities"][0]
    assert devices[0].get("DeviceIDs") == [os.environ["VERIFY_GPU_DEVICE"]]
    assert "CUDA_VISIBLE_DEVICES=0" in environment
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in environment
else:
    assert not devices
    assert "CUDA_VISIBLE_DEVICES=" in environment

source_view = Path(os.environ["VERIFY_SOURCE_VIEW"])
assert {entry.name for entry in source_view.iterdir()} == {
    "configs", "pyproject.toml", "scripts", "src"
}
for forbidden in (".git", "data", "results", "tests"):
    assert not (source_view / forbidden).exists()

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
  python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

receipt_path = Path(os.environ["VERIFY_RECEIPT"])
payload = json.loads(receipt_path.read_text(encoding="utf-8"))
assert payload["schema_version"] == 3
assert payload["receipt_type"] == "completed"
assert payload["status"] == "completed"
started = payload["started"]
assert started["schema_version"] == 2
assert started["receipt_type"] == "started"
assert started["status"] == "started"
assert started["git_sha"] == os.environ["VERIFY_SOURCE_REVISION"]
assert started["config_sha256"] == os.environ["VERIFY_CONFIG_FINGERPRINT"]
assert started["seed"] == 2026
assert started["protocol"] == "ucfrep_526"
container = started["hardware"]["container"]
assert container["image_id"] == os.environ["VERIFY_IMAGE_ID"]
assert container["environment_sha256"] == os.environ["VERIFY_ENVIRONMENT_SHA256"]
assert container["source_revision"] == os.environ["VERIFY_SOURCE_REVISION"]
assert payload["metrics"]["completed_epochs"] == 150

started_path = receipt_path.with_name(f'{payload["run_id"]}.started.json')
assert started_path.is_file()
assert digest(started_path) == payload["start_manifest_sha256"]
assert json.loads(started_path.read_text(encoding="utf-8")) == started

roles = {item["role"]: item for item in payload["artifacts"]}
assert len(roles) == len(payload["artifacts"])
required = {
    "output_encoder_checkpoint",
    "progress_log",
    "input_config",
    "input_pose_cache_snapshot",
    "input_train_pose_inputs",
    "input_train_pose_input_commitment",
    "input_dev_pose_inputs",
    "input_dev_pose_input_commitment",
    "input_test_identity_pose_inputs",
    "input_test_identity_pose_input_commitment",
}
assert required <= set(roles)
package_root = receipt_path.parent.parent.resolve(strict=True)
resolved = {}
for item in payload["artifacts"]:
    locator = PurePosixPath(item["locator"])
    assert isinstance(item["locator"], str) and "\\" not in item["locator"]
    assert locator.parts and not locator.is_absolute()
    path = receipt_path.parent.joinpath(*locator.parts).resolve(strict=True)
    path.relative_to(package_root)
    assert path.is_file()
    assert path.stat().st_size == item["bytes"]
    assert digest(path) == item["sha256"]
    resolved[item["role"]] = path

assert digest(Path(os.environ["VERIFY_CHECKPOINT"])) == roles["output_encoder_checkpoint"]["sha256"]
assert digest(Path(os.environ["VERIFY_PROGRESS"])) == roles["progress_log"]["sha256"]
expected_hashes = {
    "input_config": os.environ["VERIFY_CONFIG_SHA256"],
    "input_train_pose_inputs": os.environ["VERIFY_TRAIN_INPUT_SHA256"],
    "input_train_pose_input_commitment": os.environ["VERIFY_TRAIN_COMMIT_SHA256"],
    "input_dev_pose_inputs": os.environ["VERIFY_DEV_INPUT_SHA256"],
    "input_dev_pose_input_commitment": os.environ["VERIFY_DEV_COMMIT_SHA256"],
    "input_test_identity_pose_inputs": os.environ["VERIFY_TEST_INPUT_SHA256"],
    "input_test_identity_pose_input_commitment": os.environ["VERIFY_TEST_COMMIT_SHA256"],
}
for role, expected in expected_hashes.items():
    assert roles[role]["sha256"] == expected

snapshot = json.loads(resolved["input_pose_cache_snapshot"].read_text(encoding="utf-8"))
assert snapshot["schema_version"] == 1
assert snapshot["pose_fingerprint"] == os.environ["VERIFY_POSE_FINGERPRINT"]
assert snapshot["fingerprint"] == os.environ["VERIFY_POSE_CACHE_SET_SHA256"]
assert snapshot["entry_count"] == 337
assert len(snapshot["entries"]) == 337
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
  "$ENCODER_STAGE" "$AUDIT_ROOT" "$LOG_ROOT"
write_status 'preparing' 'source-export' 'null'

git -C "$SOURCE_CHECKOUT" archive --format=tar "$SOURCE_REVISION" \
  src pyproject.toml "$CONFIG_RELATIVE" \
  "$VALIDATOR_RELATIVE" \
  | tar -xf - -C "$SOURCE_VIEW"
[[ ! -e "${SOURCE_VIEW}/.git" && ! -e "${SOURCE_VIEW}/data" \
  && ! -e "${SOURCE_VIEW}/results" && ! -e "${SOURCE_VIEW}/tests" ]] \
  || fail 'source export contains a forbidden top-level path'
[[ "$(find "$SOURCE_VIEW" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort | tr '\n' ' ')" == 'configs pyproject.toml scripts src ' ]] \
  || fail 'source export has an unexpected top-level entry'
[[ -z "$(find "$SOURCE_VIEW" -type l -print -quit)" ]] \
  || fail 'source export contains a symlink'
require_sha256 "${SOURCE_VIEW}/${CONFIG_RELATIVE}" "$CONFIG_SHA256" 'exported native proxy config'

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

for container_name in "$PREFLIGHT_NAME" "$ENCODER_NAME"; do
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
  --arg config_sha "$CONFIG_SHA256" \
  --arg config_fp "$CONFIG_FINGERPRINT" \
  --arg pose_fp "$POSE_FINGERPRINT" \
  --arg version "$POSE_RECOVERY_VERSION" \
  --arg authorization_sha "$POSE_RECOVERY_AUTHORIZATION_SHA256" \
  --arg gate_sha "$POSE_RECOVERY_PAIRED_GATE_SHA256" \
  --arg receipt_sha "$POSE_RECOVERY_RUN_RECEIPT_SHA256" \
  --arg ledger_sha "$POSE_RECOVERY_LEDGER_SHA256" \
  --arg cache_set_sha "$AUTHORIZED_POSE_CACHE_SET_SHA256" \
  '
    .schema_version == 1
    and .passed == true
    and .classification == $classification
    and .paper_table2_value_claim_eligible == false
    and .candidate.config_file_sha256 == $config_sha
    and .candidate.config_fingerprint == $config_fp
    and .candidate.pose_fingerprint == $pose_fp
    and .candidate.encoder_epochs == 150
    and .candidate.physical_batch_size == 32
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

CURRENT_STAGE='encoder-create'
docker create \
  --name "$ENCODER_NAME" \
  --gpus "device=${GPU_DEVICE}" \
  "${common_args[@]}" \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "${source_args[@]}" \
  "${protocol_args[@]}" \
  --mount "type=bind,src=${POSE_RECOVERY_CACHE},dst=/pams/pose-cache,readonly" \
  --mount "type=bind,src=${ENCODER_STAGE},dst=/pams/output" \
  "$IMAGE_ID" \
  python -m pams train encoder \
    /pams/protocol/train.inputs.json \
    /pams/pose-cache \
    /pams/output/run \
    --config /pams/input/config.yaml \
    --device cuda:0 \
    --epochs 150 \
    --microbatch-size 32 \
    --label-free-inputs \
    --input-commitment /pams/protocol/train.inputs.commitment.json \
    --dev-inputs /pams/protocol/dev.inputs.json \
    --dev-input-commitment /pams/protocol/dev.inputs.commitment.json \
    --test-identity-inputs /pams/protocol/test-identity.inputs.json \
    --test-identity-commitment /pams/protocol/test-identity.inputs.commitment.json \
  > "${AUDIT_ROOT}/${ENCODER_NAME}.create-id.txt"
verify_container "$ENCODER_NAME" 'encoder'
run_created_container "$ENCODER_NAME" 'encoder'
flock -u 9

readonly ENCODER_RECEIPT="$(single_completion_receipt "$ENCODER_STAGE")"
readonly ENCODER_CHECKPOINT="${ENCODER_STAGE}/run/encoder.pt"
readonly ENCODER_PROGRESS="${ENCODER_STAGE}/run/logs/encoder.jsonl"
[[ -s "$ENCODER_CHECKPOINT" ]] || fail 'encoder checkpoint is missing'
[[ -s "$ENCODER_PROGRESS" ]] || fail 'encoder progress log is missing'
validate_completion_receipt \
  "$ENCODER_RECEIPT" \
  "$ENCODER_CHECKPOINT" \
  "$ENCODER_PROGRESS" \
  "$POSE_CACHE_SET_SHA256"
sha256sum -- "$ENCODER_CHECKPOINT" "$ENCODER_PROGRESS" "$ENCODER_RECEIPT" \
  > "${AUDIT_ROOT}/encoder-terminal-sha256.txt"
chmod -R a-w -- "${ENCODER_STAGE}/run"

CURRENT_STAGE='final-receipt'
RUN_RECEIPT_PATH="${AUDIT_ROOT}/run.receipt.json" \
RUN_ATTEMPT_ID="$ATTEMPT_ID" \
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
RUN_ENCODER_CHECKPOINT="$ENCODER_CHECKPOINT" \
RUN_ENCODER_PROGRESS="$ENCODER_PROGRESS" \
RUN_ENCODER_RECEIPT="$ENCODER_RECEIPT" \
python3 - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

def digest(name: str) -> str:
    return hashlib.sha256(Path(os.environ[name]).read_bytes()).hexdigest()

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
    "encoder_checkpoint_sha256": digest("RUN_ENCODER_CHECKPOINT"),
    "encoder_progress_sha256": digest("RUN_ENCODER_PROGRESS"),
    "encoder_completion_receipt_sha256": digest("RUN_ENCODER_RECEIPT"),
    "encoder_epochs": 150,
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
write_status 'completed' 'encoder-only' '0'
find "$RUN_ROOT" -type f -not -path "${AUDIT_ROOT}/artifact-sha256.txt" \
  -print0 | sort -z | xargs -0 sha256sum \
  > "${AUDIT_ROOT}/artifact-sha256.txt"
chmod -R a-w -- "$RUN_ROOT"
printf '%s\n' "$RUN_ROOT"
