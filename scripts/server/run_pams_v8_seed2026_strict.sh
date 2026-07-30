#!/usr/bin/env bash
set -Eeuo pipefail

# Formal seed-2026 v8 run.  The target-bearing dev score is a distinct
# container boundary.  This launcher never evaluates the sealed test105 split.

readonly ROOT_INPUT="${PAMS_ROOT:-/media/lenovo/data2/pams-rac}"
readonly SOURCE_REVISION="${PAMS_SOURCE_REVISION:?PAMS_SOURCE_REVISION is required}"
readonly SOURCE_CHECKOUT_INPUT="${PAMS_SOURCE_CHECKOUT:?PAMS_SOURCE_CHECKOUT is required}"
readonly IMAGE_ID="${PAMS_IMAGE_ID:?PAMS_IMAGE_ID is required}"
readonly ENVIRONMENT_SHA256="${PAMS_ENVIRONMENT_SHA256:?PAMS_ENVIRONMENT_SHA256 is required}"
readonly CONFIG_SHA256="${PAMS_CONFIG_SHA256:?PAMS_CONFIG_SHA256 is required}"
readonly CONFIG_FINGERPRINT="${PAMS_CONFIG_FINGERPRINT:?PAMS_CONFIG_FINGERPRINT is required}"
readonly POSE_FINGERPRINT="${PAMS_POSE_FINGERPRINT:?PAMS_POSE_FINGERPRINT is required}"
readonly POSE_RUN_ROOT_INPUT="${PAMS_POSE_RUN_ROOT:?PAMS_POSE_RUN_ROOT is required}"
readonly POSE_ARTIFACT_MANIFEST_SHA256="${PAMS_POSE_ARTIFACT_MANIFEST_SHA256:?PAMS_POSE_ARTIFACT_MANIFEST_SHA256 is required}"
readonly ATTEMPT_ID="${PAMS_ATTEMPT_ID:?PAMS_ATTEMPT_ID is required}"
readonly GPU_DEVICE="${PAMS_GPU_DEVICE:-0}"
readonly CONFIG_RELATIVE="configs/experiments/pams_longest_contiguous_track_v8.yaml"

readonly REFERENCE_V7_NMAE="0.7357284358361944"
readonly REFERENCE_V7_OBO="0.19047619047619047"
readonly EXTENSION_MAX_NMAE="0.60"
readonly EXTENSION_MIN_OBO="0.30"

readonly TRAIN_INPUT_SHA256="e238d307b4ed37b2f6c19eefdf7fa8812ec3d231cd2a7091be38e1c279f04207"
readonly TRAIN_COMMIT_SHA256="ce39b1c9038bb1506354de010e46a23053b307b852222332f13df3f79d799044"
readonly DEV_INPUT_SHA256="f21c49569805e4aea326977f6d7ce2964f503fad129a17e95214594364ca4ae7"
readonly DEV_COMMIT_SHA256="a2bb2db86abef64e16a8fe1ac196857ac8e77cc9cd461c568b2193401a8cd169"
readonly TEST_ID_INPUT_SHA256="9eb2a057246b77fc7a1121ba4c9d27bf55177ad2fcd5dbb87237396a4a6015ca"
readonly TEST_ID_COMMIT_SHA256="232d3db7ce09018716594a1f1b332eec47b37410a8d3068e662b026ee2dc9c66"
readonly DEV_TARGET_SHA256="1ab3bc010ccb5d869af7662586594c973e53841838643e1e9b8cc1787b0efab6"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[[ "$ROOT_INPUT" == /* ]] || fail "PAMS_ROOT must be absolute"
[[ "$SOURCE_CHECKOUT_INPUT" == /* ]] \
  || fail "PAMS_SOURCE_CHECKOUT must be absolute"
[[ "$POSE_RUN_ROOT_INPUT" == /* ]] \
  || fail "PAMS_POSE_RUN_ROOT must be absolute"
for mount_path in "$ROOT_INPUT" "$SOURCE_CHECKOUT_INPUT" "$POSE_RUN_ROOT_INPUT"; do
  [[ "$mount_path" != *","* ]] || fail "Docker bind paths must not contain commas"
done
[[ "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]] \
  || fail "PAMS_SOURCE_REVISION must be a full lowercase Git SHA"
[[ "$IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || fail "PAMS_IMAGE_ID must be an immutable image ID"
for digest_name in \
  ENVIRONMENT_SHA256 \
  CONFIG_SHA256 \
  CONFIG_FINGERPRINT \
  POSE_FINGERPRINT \
  POSE_ARTIFACT_MANIFEST_SHA256; do
  digest="${!digest_name}"
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] \
    || fail "PAMS_${digest_name} must be a lowercase SHA-256"
done
[[ "$ATTEMPT_ID" =~ ^[a-z0-9][a-z0-9._-]{0,39}$ ]] \
  || fail "PAMS_ATTEMPT_ID must be a lowercase safe slug of at most 40 characters"
[[ "$ATTEMPT_ID" != *".."* ]] || fail "PAMS_ATTEMPT_ID must not contain '..'"
[[ "$GPU_DEVICE" =~ ^[0-9]+$ ]] \
  || fail "PAMS_GPU_DEVICE must be a non-negative integer"

readonly ROOT="$(realpath -e -- "$ROOT_INPUT")"
readonly SOURCE_CHECKOUT="$(realpath -e -- "$SOURCE_CHECKOUT_INPUT")"
readonly POSE_RUN_ROOT="$(realpath -e -- "$POSE_RUN_ROOT_INPUT")"
readonly FIREWALL="${ROOT}/runs/firewall/ucfrep-337-84-105-375d8da"
readonly RUN_PARENT="${ROOT}/runs/pams-v8-seed2026"
readonly RUN_ROOT="${RUN_PARENT}/${ATTEMPT_ID}"
readonly SOURCE_VIEW="${RUN_ROOT}/source"
readonly SOURCE_RECEIPT="${RUN_ROOT}/source-export.receipt.json"
readonly INPUT_ROOT="${RUN_ROOT}/inputs"
readonly CONFIG_HOST="${INPUT_ROOT}/pams_longest_contiguous_track_v8.yaml"
readonly TRAIN_POSE_VIEW="${INPUT_ROOT}/pose-train337"
readonly TRAIN_DEV_POSE_VIEW="${INPUT_ROOT}/pose-train337-dev84"
readonly ENCODER_STAGE="${RUN_ROOT}/stages/encoder"
readonly SSHEAD_STAGE="${RUN_ROOT}/stages/sshead"
readonly PREDICT_STAGE="${RUN_ROOT}/stages/dev-predict"
readonly SCORE_STAGE="${RUN_ROOT}/stages/dev-score"
readonly AUDIT_ROOT="${RUN_ROOT}/audit"
readonly LOG_ROOT="${RUN_ROOT}/logs"
readonly LOCK_ROOT="${ROOT}/.pams-gpu-locks"
readonly LOCK_PATH="${LOCK_ROOT}/gpu${GPU_DEVICE}.lock"

readonly TRAIN_INPUT="${FIREWALL}/train.inputs.json"
readonly TRAIN_COMMIT="${FIREWALL}/train.inputs.commitment.json"
readonly DEV_INPUT="${FIREWALL}/dev.inputs.json"
readonly DEV_COMMIT="${FIREWALL}/dev.inputs.commitment.json"
readonly TEST_ID_INPUT="${FIREWALL}/test-identity.inputs.json"
readonly TEST_ID_COMMIT="${FIREWALL}/test-identity.inputs.commitment.json"
readonly DEV_TARGET="${FIREWALL}/dev.targets.json"

readonly POSE_CACHE_SOURCE="${POSE_RUN_ROOT}/pose-cache"
readonly POSE_TRAIN_LEDGER="${POSE_RUN_ROOT}/ledgers/train337.json"
readonly POSE_DEV_LEDGER="${POSE_RUN_ROOT}/ledgers/dev84.json"
readonly POSE_STATUS="${POSE_RUN_ROOT}/status.json"
readonly POSE_RESERVATION="${POSE_RUN_ROOT}/attempt.reservation.json"
readonly POSE_ARTIFACT_MANIFEST="${POSE_RUN_ROOT}/audit/artifact-sha256.txt"

readonly CONFIG_NAME="p8cfg-${ATTEMPT_ID}"
readonly ENCODER_NAME="p8enc-${ATTEMPT_ID}"
readonly SSHEAD_NAME="p8head-${ATTEMPT_ID}"
readonly PREDICT_NAME="p8pred-${ATTEMPT_ID}"
readonly SCORE_NAME="p8score-${ATTEMPT_ID}"

RUN_RESERVED=0
CURRENT_STAGE="preflight"
ACTIVE_CONTAINER=""

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
  fi
  if [[ "$exit_code" -ne 0 && "$RUN_RESERVED" -eq 1 ]]; then
    write_status "failed" "$CURRENT_STAGE" "$exit_code" >/dev/null 2>&1
  fi
  exit "$exit_code"
}

trap on_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

validate_pose_ledger() {
  local split="$1"
  local ledger="$2"
  local expected_count="$3"
  local expected_input_sha256="$4"
  local expected_commitment_sha256="$5"
  local commitment="$6"
  test -f "$ledger" || fail "missing ${split} pose ledger"
  jq -e \
    --argjson expected_count "$expected_count" \
    --arg input_sha "$expected_input_sha256" \
    --arg commitment_sha "$expected_commitment_sha256" \
    --arg pose_fingerprint "$POSE_FINGERPRINT" \
    --arg identity "$(jq -r '.identity_sha256' "$commitment")" \
    '
      .selected == $expected_count
      and .completed == $expected_count
      and .failed == 0
      and .input_file_sha256 == $input_sha
      and .sidecar_sha256 == $input_sha
      and .commitment_file_sha256 == $commitment_sha
      and .identity_sha256 == $identity
      and .pose_fingerprint == $pose_fingerprint
      and .successful_cache_snapshot.entry_count == $expected_count
      and .successful_cache_snapshot.pose_fingerprint == $pose_fingerprint
      and (.caches | length) == $expected_count
      and ([.caches[].selected_source_frames == null] | any | not)
    ' \
    "$ledger" >/dev/null \
    || fail "${split} pose ledger is incomplete or has the wrong fingerprint"
}

copy_pose_view() {
  local view_name="$1"
  local destination="$2"
  local expected_count="$3"
  shift 3
  local receipt="${AUDIT_ROOT}/${view_name}.sha256.tsv"
  local count=0
  local manifest
  local video_id
  local digest
  local source
  local target
  local source_sha256
  local target_sha256

  mkdir -- "$destination"
  : > "$receipt"
  for manifest in "$@"; do
    while IFS= read -r video_id; do
      [[ -n "$video_id" ]] || fail "${view_name} contains an empty video_id"
      [[ "$video_id" != *$'\n'* && "$video_id" != *$'\r'* && "$video_id" != *$'\t'* ]] \
        || fail "${view_name} contains a control character in video_id"
      digest="$(printf '%s' "$video_id" | sha256sum | awk '{print $1}')"
      source="${POSE_CACHE_SOURCE}/${digest}.npz"
      target="${destination}/${digest}.npz"
      test -f "$source" || fail "missing v8 pose cache for ${video_id}: ${source}"
      test ! -e "$target" || fail "duplicate pose cache in ${view_name}: ${video_id}"
      source_sha256="$(sha256_of "$source")"
      cp --reflink=auto --preserve=mode,timestamps -- "$source" "$target"
      [[ ! "$source" -ef "$target" ]] \
        || fail "${view_name} cache is not an independent file: ${video_id}"
      target_sha256="$(sha256_of "$target")"
      [[ "$target_sha256" == "$source_sha256" ]] \
        || fail "${view_name} copied cache hash mismatch: ${video_id}"
      printf '%s\t%s\t%s\n' "$video_id" "$digest" "$target_sha256" >> "$receipt"
      ((count += 1))
    done < <(jq -r '.records[].video_id' "$manifest")
  done
  [[ "$count" -eq "$expected_count" ]] \
    || fail "${view_name} expected ${expected_count} caches, copied ${count}"
  [[ "$(find "$destination" -maxdepth 1 -type f -name '*.npz' | wc -l)" -eq "$expected_count" ]] \
    || fail "${view_name} does not contain exactly ${expected_count} NPZ files"
  [[ "$(find "$destination" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq "$expected_count" ]] \
    || fail "${view_name} contains non-NPZ files"
  [[ -z "$(find "$destination" -mindepth 1 -type l -print -quit)" ]] \
    || fail "${view_name} contains a symlink"
  chmod 0444 "$destination"/*.npz "$receipt"
  chmod 0555 "$destination"
}

single_completion_receipt() {
  local output_root="$1"
  local -a matches=()
  mapfile -t matches < <(
    find "${output_root}/run/manifests" \
      -maxdepth 1 \
      -type f \
      -name '*.completed.json' \
      -print \
      | sort
  )
  [[ "${#matches[@]}" -eq 1 ]] \
    || fail "expected one completion receipt in ${output_root}, got ${#matches[@]}"
  printf '%s\n' "${matches[0]}"
}

validate_completion_receipt() {
  local receipt="$1"
  local output_role="$2"
  local checkpoint="$3"
  local progress="$4"
  local expected_epochs="$5"
  VERIFY_RECEIPT="$receipt" \
  VERIFY_OUTPUT_ROLE="$output_role" \
  VERIFY_CHECKPOINT="$checkpoint" \
  VERIFY_PROGRESS="$progress" \
  VERIFY_EPOCHS="$expected_epochs" \
  VERIFY_SOURCE_REVISION="$SOURCE_REVISION" \
  VERIFY_IMAGE_ID="$IMAGE_ID" \
  VERIFY_ENVIRONMENT_SHA256="$ENVIRONMENT_SHA256" \
  VERIFY_CONFIG_SHA256="$CONFIG_SHA256" \
  VERIFY_CONFIG_FINGERPRINT="$CONFIG_FINGERPRINT" \
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
assert payload["metrics"]["completed_epochs"] == int(os.environ["VERIFY_EPOCHS"])

started_path = receipt_path.with_name(f'{payload["run_id"]}.started.json')
assert started_path.is_file()
assert digest(started_path) == payload["start_manifest_sha256"]
assert json.loads(started_path.read_text(encoding="utf-8")) == started

roles = {item["role"]: item for item in payload["artifacts"]}
assert len(roles) == len(payload["artifacts"])
required = {
    os.environ["VERIFY_OUTPUT_ROLE"],
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
assert receipt_path.parent.name == "manifests"
package_root = receipt_path.parent.parent.resolve(strict=True)
for item in payload["artifacts"]:
    assert isinstance(item["locator"], str)
    assert "\\" not in item["locator"]
    locator = PurePosixPath(item["locator"])
    assert not locator.is_absolute()
    assert locator.parts
    artifact_path = receipt_path.parent.joinpath(*locator.parts).resolve(strict=True)
    artifact_path.relative_to(package_root)
    assert artifact_path.is_file()
    assert artifact_path.stat().st_size == item["bytes"]
    assert digest(artifact_path) == item["sha256"]

checkpoint = Path(os.environ["VERIFY_CHECKPOINT"])
progress = Path(os.environ["VERIFY_PROGRESS"])
assert digest(checkpoint) == roles[os.environ["VERIFY_OUTPUT_ROLE"]]["sha256"]
assert digest(progress) == roles["progress_log"]["sha256"]
assert roles["input_config"]["sha256"] == os.environ["VERIFY_CONFIG_SHA256"]
PY
}

validate_prediction_bundle() {
  local predictions="$1"
  local receipt="$2"
  VERIFY_PREDICTIONS="$predictions" \
  VERIFY_PREDICTION_RECEIPT="$receipt" \
  VERIFY_SOURCE_REVISION="$SOURCE_REVISION" \
  VERIFY_IMAGE_ID="$IMAGE_ID" \
  VERIFY_ENVIRONMENT_SHA256="$ENVIRONMENT_SHA256" \
  VERIFY_CONFIG_SHA256="$CONFIG_SHA256" \
  VERIFY_CONFIG_FINGERPRINT="$CONFIG_FINGERPRINT" \
  VERIFY_POSE_FINGERPRINT="$POSE_FINGERPRINT" \
  python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path


predictions_path = Path(os.environ["VERIFY_PREDICTIONS"])
receipt_path = Path(os.environ["VERIFY_PREDICTION_RECEIPT"])
predictions_bytes = predictions_path.read_bytes()
predictions = json.loads(predictions_bytes)
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
prediction_sha256 = hashlib.sha256(predictions_bytes).hexdigest()

assert predictions["schema_version"] == 2
assert predictions["artifact_type"] == "pams_checkpoint_dev_predictions"
assert predictions["protocol"] == "ucfrep_526"
assert predictions["split"] == "dev"
assert predictions["variant"] == "sshead"
assert predictions["record_total"] == 84
assert len(predictions["records"]) == 84
assert len({row["video_id"] for row in predictions["records"]}) == 84
assert predictions["config_file_sha256"] == os.environ["VERIFY_CONFIG_SHA256"]
assert predictions["training_config_fingerprint"] == os.environ["VERIFY_CONFIG_FINGERPRINT"]
assert predictions["pose_fingerprint"] == os.environ["VERIFY_POSE_FINGERPRINT"]
assert predictions["prediction_source_git_sha"] == os.environ["VERIFY_SOURCE_REVISION"]
assert predictions["prediction_container_image_id"] == os.environ["VERIFY_IMAGE_ID"]
assert (
    predictions["prediction_container_environment_sha256"]
    == os.environ["VERIFY_ENVIRONMENT_SHA256"]
)
assert receipt["schema_version"] == 2
assert receipt["artifact_type"] == "pams_checkpoint_dev_prediction_receipt"
assert receipt["protocol"] == "ucfrep_526"
assert receipt["split"] == "dev"
assert receipt["variant"] == "sshead"
assert receipt["prediction_file"] == predictions_path.name
assert receipt["prediction_sha256"] == prediction_sha256
assert receipt["prediction_bytes"] == len(predictions_bytes)
for field in (
    "training_config_fingerprint",
    "config_fingerprint",
    "checkpoint_sha256",
    "checkpoint_progress_sha256",
    "checkpoint_completion_receipt_sha256",
    "upstream_encoder_checkpoint_sha256",
    "upstream_encoder_progress_sha256",
    "upstream_encoder_completion_receipt_sha256",
    "prediction_source_git_sha",
    "prediction_container_image_id",
    "prediction_container_environment_sha256",
    "prediction_code_sha256",
):
    assert receipt[field] == predictions[field]
PY
}

validate_score_bundle() {
  local evaluation="$1"
  local receipt="$2"
  VERIFY_EVALUATION="$evaluation" \
  VERIFY_EVALUATION_RECEIPT="$receipt" \
  VERIFY_SOURCE_REVISION="$SOURCE_REVISION" \
  VERIFY_CONFIG_FINGERPRINT="$CONFIG_FINGERPRINT" \
  VERIFY_DEV_TARGET_SHA256="$DEV_TARGET_SHA256" \
  python3 - <<'PY'
import hashlib
import json
import math
import os
from pathlib import Path


evaluation_path = Path(os.environ["VERIFY_EVALUATION"])
receipt_path = Path(os.environ["VERIFY_EVALUATION_RECEIPT"])
evaluation_bytes = evaluation_path.read_bytes()
evaluation = json.loads(evaluation_bytes)
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
assert evaluation["schema_version"] == 1
assert evaluation["artifact_type"] == "pams_checkpoint_dev_evaluation"
assert evaluation["protocol"] == "ucfrep_526"
assert evaluation["split"] == "dev"
assert evaluation["variant"] == "sshead"
assert evaluation["training_config_fingerprint"] == os.environ["VERIFY_CONFIG_FINGERPRINT"]
assert evaluation["prediction_source_git_sha"] == os.environ["VERIFY_SOURCE_REVISION"]
assert evaluation["scoring_source_git_sha"] == os.environ["VERIFY_SOURCE_REVISION"]
assert evaluation["dev_targets_sha256"] == os.environ["VERIFY_DEV_TARGET_SHA256"]
report = evaluation["report"]
assert report["sample_count"] == 84
assert report["bootstrap_samples"] == 10_000
assert report["bootstrap_seed"] == 2026
assert len(report["per_video"]) == 84
assert math.isfinite(report["nmae"]) and report["nmae"] >= 0.0
assert math.isfinite(report["obo"]) and 0.0 <= report["obo"] <= 1.0
assert receipt["schema_version"] == 1
assert receipt["artifact_type"] == "pams_checkpoint_dev_evaluation_receipt"
assert receipt["evaluation_file"] == evaluation_path.name
assert receipt["evaluation_sha256"] == hashlib.sha256(evaluation_bytes).hexdigest()
assert receipt["evaluation_bytes"] == len(evaluation_bytes)
assert receipt["dev_targets_sha256"] == os.environ["VERIFY_DEV_TARGET_SHA256"]
assert receipt["prediction_source_git_sha"] == os.environ["VERIFY_SOURCE_REVISION"]
assert receipt["scoring_source_git_sha"] == os.environ["VERIFY_SOURCE_REVISION"]
PY
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
  VERIFY_SOURCE_RECEIPT_SHA256="$SOURCE_RECEIPT_SHA256" \
  VERIFY_CONFIG_HOST="$CONFIG_HOST" \
  VERIFY_FIREWALL="$FIREWALL" \
  VERIFY_TRAIN_POSE="$TRAIN_POSE_VIEW" \
  VERIFY_TRAIN_DEV_POSE="$TRAIN_DEV_POSE_VIEW" \
  VERIFY_ENCODER_STAGE="$ENCODER_STAGE" \
  VERIFY_SSHEAD_STAGE="$SSHEAD_STAGE" \
  VERIFY_PREDICT_STAGE="$PREDICT_STAGE" \
  VERIFY_SCORE_STAGE="$SCORE_STAGE" \
  VERIFY_IMAGE_ID="$IMAGE_ID" \
  VERIFY_SOURCE_REVISION="$SOURCE_REVISION" \
  VERIFY_ENVIRONMENT_SHA256="$ENVIRONMENT_SHA256" \
  VERIFY_GPU_DEVICE="$GPU_DEVICE" \
  python3 - <<'PY'
import json
import os
from pathlib import Path


stage = os.environ["VERIFY_STAGE"]
item = json.loads(
    Path(os.environ["VERIFY_INSPECT"]).read_text(encoding="utf-8")
)[0]
config = item["Config"]
host = item["HostConfig"]
mounts = {
    mount["Destination"]: (mount["Source"], bool(mount["RW"]))
    for mount in item["Mounts"]
}

assert config["Image"] == os.environ["VERIFY_IMAGE_ID"]
assert config["User"] == "1000:1000"
assert host["NetworkMode"] == "none"
assert host["ReadonlyRootfs"] is True
assert "ALL" in (host.get("CapDrop") or [])
assert "no-new-privileges:true" in (host.get("SecurityOpt") or [])
assert int(host["PidsLimit"]) == 4096
assert int(host["Memory"]) == 96 * 1024**3
assert set(host.get("Tmpfs") or {}) == {
    "/tmp",
    "/pams/tmp",
    "/pams/cache",
    "/pams/home",
}

firewall = os.environ["VERIFY_FIREWALL"]
source_mounts = {
    "/workspace": (os.environ["VERIFY_SOURCE_VIEW"], False),
    "/pams/source-export-receipt.json": (
        os.environ["VERIFY_SOURCE_RECEIPT"],
        False,
    ),
}
protocol_mounts = {
    "/pams/input/config.yaml": (os.environ["VERIFY_CONFIG_HOST"], False),
    "/pams/protocol/train.inputs.json": (
        f"{firewall}/train.inputs.json",
        False,
    ),
    "/pams/protocol/train.inputs.commitment.json": (
        f"{firewall}/train.inputs.commitment.json",
        False,
    ),
    "/pams/protocol/dev.inputs.json": (
        f"{firewall}/dev.inputs.json",
        False,
    ),
    "/pams/protocol/dev.inputs.commitment.json": (
        f"{firewall}/dev.inputs.commitment.json",
        False,
    ),
    "/pams/protocol/test-identity.inputs.json": (
        f"{firewall}/test-identity.inputs.json",
        False,
    ),
    "/pams/protocol/test-identity.inputs.commitment.json": (
        f"{firewall}/test-identity.inputs.commitment.json",
        False,
    ),
}
if stage == "config-preflight":
    expected = {
        **source_mounts,
        "/pams/input/config.yaml": (os.environ["VERIFY_CONFIG_HOST"], False),
    }
elif stage == "encoder":
    expected = {
        **source_mounts,
        **protocol_mounts,
        "/pams/pose-cache": (os.environ["VERIFY_TRAIN_POSE"], False),
        "/pams/output": (os.environ["VERIFY_ENCODER_STAGE"], True),
    }
elif stage == "sshead":
    expected = {
        **source_mounts,
        **protocol_mounts,
        "/pams/pose-cache": (os.environ["VERIFY_TRAIN_POSE"], False),
        "/pams/upstream/encoder": (
            f'{os.environ["VERIFY_ENCODER_STAGE"]}/run',
            False,
        ),
        "/pams/output": (os.environ["VERIFY_SSHEAD_STAGE"], True),
    }
elif stage == "dev-predict":
    expected = {
        **source_mounts,
        **protocol_mounts,
        "/pams/pose-cache": (os.environ["VERIFY_TRAIN_DEV_POSE"], False),
        "/pams/upstream/encoder": (
            f'{os.environ["VERIFY_ENCODER_STAGE"]}/run',
            False,
        ),
        "/pams/upstream/sshead": (
            f'{os.environ["VERIFY_SSHEAD_STAGE"]}/run',
            False,
        ),
        "/pams/output": (os.environ["VERIFY_PREDICT_STAGE"], True),
    }
elif stage == "dev-score":
    expected = {
        **source_mounts,
        "/pams/frozen/predictions.json": (
            f'{os.environ["VERIFY_PREDICT_STAGE"]}/run/predictions.json',
            False,
        ),
        "/pams/frozen/prediction.receipt.json": (
            f'{os.environ["VERIFY_PREDICT_STAGE"]}/run/prediction.receipt.json',
            False,
        ),
        "/pams/protocol/dev.targets.json": (
            f"{firewall}/dev.targets.json",
            False,
        ),
        "/pams/output": (os.environ["VERIFY_SCORE_STAGE"], True),
    }
else:
    raise AssertionError(stage)
assert mounts == expected, (stage, mounts, expected)

command_text = "\0".join(config.get("Cmd") or [])
all_text = "\0".join(
    [command_text]
    + list(mounts)
    + [source for source, _ in mounts.values()]
)
if stage != "dev-score":
    assert "dev.targets" not in all_text
else:
    assert "/pams/protocol/dev.targets.json" in command_text
if stage in {"encoder", "sshead"}:
    assert mounts["/pams/pose-cache"][0] == os.environ["VERIFY_TRAIN_POSE"]
if stage == "dev-predict":
    assert mounts["/pams/pose-cache"][0] == os.environ["VERIFY_TRAIN_DEV_POSE"]

environment = set(config.get("Env") or [])
assert "PYTHONPATH=/workspace/src" in environment
assert "PAMS_AUDIT_MODE=formal" in environment
assert (
    f'PAMS_CONTAINER_IMAGE_ID={os.environ["VERIFY_IMAGE_ID"]}'
    in environment
)
assert (
    f'PAMS_CONTAINER_SOURCE_REVISION={os.environ["VERIFY_SOURCE_REVISION"]}'
    in environment
)
assert (
    "PAMS_CONTAINER_ENVIRONMENT_SHA256="
    + os.environ["VERIFY_ENVIRONMENT_SHA256"]
    in environment
)
assert "PAMS_SOURCE_EXPORT_RECEIPT=/pams/source-export-receipt.json" in environment
assert (
    "PAMS_SOURCE_EXPORT_RECEIPT_SHA256="
    + os.environ["VERIFY_SOURCE_RECEIPT_SHA256"]
    in environment
)

device_requests = host.get("DeviceRequests") or []
if stage in {"encoder", "sshead", "dev-predict"}:
    assert len(device_requests) == 1
    request = device_requests[0]
    assert "gpu" in request["Capabilities"][0]
    assert request.get("DeviceIDs") == [os.environ["VERIFY_GPU_DEVICE"]]
    assert "CUDA_VISIBLE_DEVICES=0" in environment
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in environment
else:
    assert not device_requests

source_view = Path(os.environ["VERIFY_SOURCE_VIEW"])
assert not (source_view / ".git").exists()
assert not (source_view / "results").exists()
assert not (source_view / "data").exists()
assert not (source_view / "tests").exists()
assert {path.name for path in source_view.iterdir()} == {
    "configs",
    "pyproject.toml",
    "src",
}

report = {
    "schema_version": 1,
    "container_name": item["Name"].lstrip("/"),
    "stage": stage,
    "verified": True,
    "image_id": config["Image"],
    "network_mode": host["NetworkMode"],
    "read_only_root": host["ReadonlyRootfs"],
    "mounts": [
        {"destination": key, "source": value[0], "rw": value[1]}
        for key, value in sorted(mounts.items())
    ],
}
Path(os.environ["VERIFY_INSPECT"]).with_suffix(".verification.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

run_created_container() {
  local container_name="$1"
  local stage="$2"
  local log_path="${LOG_ROOT}/${container_name}.log"
  CURRENT_STAGE="$stage"
  write_status "running" "$stage" "null"
  ACTIVE_CONTAINER="$container_name"
  set +e
  docker start --attach "$container_name" > "$log_path" 2>&1
  local attach_exit="$?"
  set -e
  local still_running
  still_running="$(
    docker inspect "$container_name" --format '{{.State.Running}}'
  )"
  if [[ "$still_running" == "true" ]]; then
    docker stop --time 10 "$container_name" >/dev/null
  fi
  local container_exit
  container_exit="$(
    docker inspect "$container_name" --format '{{.State.ExitCode}}'
  )"
  ACTIVE_CONTAINER=""
  printf '%s\n' "$container_exit" \
    > "${AUDIT_ROOT}/${container_name}.exit-code.txt"
  docker inspect "$container_name" \
    > "${AUDIT_ROOT}/${container_name}.post-run.inspect.json"
  if [[ "$attach_exit" -ne 0 || "$still_running" == "true" || "$container_exit" -ne 0 ]]; then
    fail \
      "container ${container_name} failed: attach=${attach_exit}, running_after_attach=${still_running}, container=${container_exit}"
  fi
}

[[ "$POSE_RUN_ROOT" == "${ROOT}/runs/pose-protocol/"* ]] \
  || fail "PAMS_POSE_RUN_ROOT must be a completed pose-protocol run under PAMS_ROOT"
[[ ! -L "$POSE_RUN_ROOT_INPUT" ]] || fail "pose run root must not be a symlink"
test -d "$FIREWALL" || fail "missing frozen UCFRep firewall"
test -d "$POSE_CACHE_SOURCE" || fail "missing v8 pose cache pool"

assert_sha256 "$TRAIN_INPUT" "$TRAIN_INPUT_SHA256"
assert_sha256 "$TRAIN_COMMIT" "$TRAIN_COMMIT_SHA256"
assert_sha256 "$DEV_INPUT" "$DEV_INPUT_SHA256"
assert_sha256 "$DEV_COMMIT" "$DEV_COMMIT_SHA256"
assert_sha256 "$TEST_ID_INPUT" "$TEST_ID_INPUT_SHA256"
assert_sha256 "$TEST_ID_COMMIT" "$TEST_ID_COMMIT_SHA256"
# DEV_TARGET is intentionally not stat'ed, read, or hashed before dev-score.

[[ "$(git -C "$SOURCE_CHECKOUT" rev-parse HEAD)" == "$SOURCE_REVISION" ]] \
  || fail "source checkout revision mismatch"
[[ -z "$(
  git -C "$SOURCE_CHECKOUT" status --porcelain=v1 --untracked-files=all
)" ]] || fail "source checkout is dirty"
readonly CONFIG_SOURCE="${SOURCE_CHECKOUT}/${CONFIG_RELATIVE}"
assert_sha256 "$CONFIG_SOURCE" "$CONFIG_SHA256"
source "${SOURCE_CHECKOUT}/scripts/server/environment_fingerprint.sh"
[[ "$(
  pams_environment_fingerprint "${SOURCE_CHECKOUT}/docker/server"
)" == "$ENVIRONMENT_SHA256" ]] || fail "source environment fingerprint mismatch"
[[ "$(docker image inspect "$IMAGE_ID" --format '{{.Id}}')" == "$IMAGE_ID" ]] \
  || fail "container image mismatch"
[[ "$(
  docker image inspect "$IMAGE_ID" \
    --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
)" == "$SOURCE_REVISION" ]] || fail "container source label mismatch"
[[ "$(
  docker image inspect "$IMAGE_ID" \
    --format \
      '{{index .Config.Labels "org.opencontainers.image.pams.environment-sha256"}}'
)" == "$ENVIRONMENT_SHA256" ]] || fail "container environment label mismatch"

assert_sha256 "$POSE_ARTIFACT_MANIFEST" "$POSE_ARTIFACT_MANIFEST_SHA256"
(cd "$POSE_RUN_ROOT" && sha256sum --check --strict "$POSE_ARTIFACT_MANIFEST" >/dev/null) \
  || fail "completed v8 pose run artifact manifest failed validation"
jq -e \
  '.schema_version == 1 and .status == "completed" and .exit_code == 0' \
  "$POSE_STATUS" >/dev/null \
  || fail "v8 pose input run is not completed"
jq -e \
  --arg config_sha256 "$CONFIG_SHA256" \
  --arg pose_fingerprint "$POSE_FINGERPRINT" \
  '
    .schema_version == 1
    and .classification == "v8-longest-contiguous-track-protocol-correction"
    and .config_sha256 == $config_sha256
    and .pose_fingerprint == $pose_fingerprint
    and .test_inputs_mounted == false
    and .targets_mounted == false
    and .split_video_views == true
  ' \
  "$POSE_RESERVATION" >/dev/null \
  || fail "pose run reservation does not match the v8 protocol correction"
validate_pose_ledger \
  "train337" \
  "$POSE_TRAIN_LEDGER" \
  337 \
  "$TRAIN_INPUT_SHA256" \
  "$TRAIN_COMMIT_SHA256" \
  "$TRAIN_COMMIT"
validate_pose_ledger \
  "dev84" \
  "$POSE_DEV_LEDGER" \
  84 \
  "$DEV_INPUT_SHA256" \
  "$DEV_COMMIT_SHA256" \
  "$DEV_COMMIT"
[[ "$(find "$POSE_CACHE_SOURCE" -maxdepth 1 -type f -name '*.npz' | wc -l)" -eq 421 ]] \
  || fail "completed v8 pose run does not contain exactly 421 caches"
[[ "$(find "$POSE_CACHE_SOURCE" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 421 ]] \
  || fail "completed v8 pose pool contains unexpected files"
[[ -z "$(find "$POSE_CACHE_SOURCE" -mindepth 1 -type l -print -quit)" ]] \
  || fail "completed v8 pose pool contains a symlink"

mkdir -p -- "$RUN_PARENT"
[[ ! -L "$RUN_PARENT" ]] || fail "v8 run parent must not be a symlink"
readonly RUN_PARENT_RESOLVED="$(realpath -e -- "$RUN_PARENT")"
[[ "$RUN_PARENT_RESOLVED" == "${ROOT}/runs/pams-v8-seed2026" ]] \
  || fail "v8 run parent resolves outside the project"
mkdir -- "$RUN_ROOT" || fail "immutable run root exists: $RUN_ROOT"
RUN_RESERVED=1
mkdir -- \
  "$SOURCE_VIEW" \
  "$INPUT_ROOT" \
  "${RUN_ROOT}/stages" \
  "$ENCODER_STAGE" \
  "$SSHEAD_STAGE" \
  "$PREDICT_STAGE" \
  "$SCORE_STAGE" \
  "$AUDIT_ROOT" \
  "$LOG_ROOT"
write_status "preparing" "source-export" "null"

git -C "$SOURCE_CHECKOUT" archive \
  --format=tar \
  "$SOURCE_REVISION" \
  src \
  pyproject.toml \
  "$CONFIG_RELATIVE" \
  | tar -xf - -C "$SOURCE_VIEW"
test ! -e "${SOURCE_VIEW}/.git" || fail "source export unexpectedly contains .git"
test ! -e "${SOURCE_VIEW}/results" || fail "source export unexpectedly contains results"
test ! -e "${SOURCE_VIEW}/data" || fail "source export unexpectedly contains data"
test ! -e "${SOURCE_VIEW}/tests" || fail "source export unexpectedly contains tests"
[[ "$(find "$SOURCE_VIEW" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort | tr '\n' ' ')" == "configs pyproject.toml src " ]] \
  || fail "source export has an unexpected top-level entry"
[[ -z "$(find "$SOURCE_VIEW" -type l -print -quit)" ]] \
  || fail "source export contains a symlink"
assert_sha256 "${SOURCE_VIEW}/${CONFIG_RELATIVE}" "$CONFIG_SHA256"

SOURCE_EXPORT_ROOT="$SOURCE_VIEW" \
SOURCE_EXPORT_REVISION="$SOURCE_REVISION" \
SOURCE_EXPORT_RECEIPT="$SOURCE_RECEIPT" \
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path


root = Path(os.environ["SOURCE_EXPORT_ROOT"]).resolve(strict=True)
output = Path(os.environ["SOURCE_EXPORT_RECEIPT"])
records = []
for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
    if path.is_symlink():
        raise RuntimeError(f"source export contains symlink: {path}")
    if path.is_dir():
        continue
    if not path.is_file():
        raise RuntimeError(f"source export contains non-regular entry: {path}")
    payload = path.read_bytes()
    records.append(
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    )
if not records:
    raise RuntimeError("source export is empty")
receipt = {
    "schema_version": 1,
    "source_revision": os.environ["SOURCE_EXPORT_REVISION"],
    "root": ".",
    "files": records,
}
output.write_text(
    json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY
readonly SOURCE_RECEIPT_SHA256="$(sha256_of "$SOURCE_RECEIPT")"
cp -- "${SOURCE_VIEW}/${CONFIG_RELATIVE}" "$CONFIG_HOST"
assert_sha256 "$CONFIG_HOST" "$CONFIG_SHA256"
chmod -R a-w "$SOURCE_VIEW"
chmod 0444 "$SOURCE_RECEIPT" "$CONFIG_HOST"

CURRENT_STAGE="pose-views"
copy_pose_view "pose-train337" "$TRAIN_POSE_VIEW" 337 "$TRAIN_INPUT"
copy_pose_view \
  "pose-train337-dev84" \
  "$TRAIN_DEV_POSE_VIEW" \
  421 \
  "$TRAIN_INPUT" \
  "$DEV_INPUT"

cat > "${RUN_ROOT}/attempt.reservation.json" <<EOF
{
  "schema_version": 1,
  "attempt_id": "${ATTEMPT_ID}",
  "reserved_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "dataset": "UCFRep",
  "split": "dev84",
  "seed": 2026,
  "classification": "v8-longest-contiguous-track-seed2026-strict-gate",
  "source_revision": "${SOURCE_REVISION}",
  "source_export_receipt_sha256": "${SOURCE_RECEIPT_SHA256}",
  "container_image_id": "${IMAGE_ID}",
  "container_environment_sha256": "${ENVIRONMENT_SHA256}",
  "config_sha256": "${CONFIG_SHA256}",
  "config_fingerprint": "${CONFIG_FINGERPRINT}",
  "pose_fingerprint": "${POSE_FINGERPRINT}",
  "pose_run_root": "${POSE_RUN_ROOT}",
  "pose_artifact_manifest_sha256": "${POSE_ARTIFACT_MANIFEST_SHA256}",
  "encoder_epochs": 150,
  "sshead_epochs": 30,
  "encoder_pose_scope": "train337-only",
  "sshead_pose_scope": "train337-only",
  "dev_predict_pose_scope": "train337-plus-dev84",
  "encoder_dev_targets_mounted": false,
  "sshead_dev_targets_mounted": false,
  "dev_predict_targets_mounted": false,
  "dev_score_code_mounts": [
    "read-only-source-export",
    "read-only-source-export-receipt"
  ],
  "dev_score_data_mounts": [
    "frozen-predictions",
    "frozen-prediction-receipt",
    "dev84-targets"
  ],
  "test_pose_mounted": false,
  "test_targets_mounted": false,
  "test_evaluation_authorized": false,
  "identity_only_test_sidecars_required_by_cli": true,
  "extension_gate": {
    "reference": "pams-v7-seed2026-dev84",
    "reference_nmae": ${REFERENCE_V7_NMAE},
    "reference_obo": ${REFERENCE_V7_OBO},
    "required_nmae_lte": ${EXTENSION_MAX_NMAE},
    "required_obo_gte": ${EXTENSION_MIN_OBO},
    "require_strict_nmae_improvement": true,
    "require_strict_obo_improvement": true,
    "seeds_authorized_if_passed": [42, 3407],
    "seeds_42_3407_authorized_initially": false
  },
  "dev_targets_expected_sha256": "${DEV_TARGET_SHA256}"
}
EOF
chmod 0444 "${RUN_ROOT}/attempt.reservation.json"

{
  printf 'source_revision %s\n' "$SOURCE_REVISION"
  printf 'source_export_receipt_sha256 %s\n' "$SOURCE_RECEIPT_SHA256"
  printf 'image_id %s\n' "$IMAGE_ID"
  printf 'environment_sha256 %s\n' "$ENVIRONMENT_SHA256"
  printf 'config_sha256 %s\n' "$(sha256_of "$CONFIG_HOST")"
  printf 'config_fingerprint %s\n' "$CONFIG_FINGERPRINT"
  printf 'pose_fingerprint %s\n' "$POSE_FINGERPRINT"
  printf 'pose_run_artifact_manifest_sha256 %s\n' "$POSE_ARTIFACT_MANIFEST_SHA256"
  printf 'train_inputs_sha256 %s\n' "$(sha256_of "$TRAIN_INPUT")"
  printf 'train_commitment_sha256 %s\n' "$(sha256_of "$TRAIN_COMMIT")"
  printf 'dev_inputs_sha256 %s\n' "$(sha256_of "$DEV_INPUT")"
  printf 'dev_commitment_sha256 %s\n' "$(sha256_of "$DEV_COMMIT")"
  printf 'test_identity_inputs_sha256 %s\n' "$(sha256_of "$TEST_ID_INPUT")"
  printf 'test_identity_commitment_sha256 %s\n' "$(sha256_of "$TEST_ID_COMMIT")"
  printf 'dev_targets_expected_sha256 %s\n' "$DEV_TARGET_SHA256"
} > "${AUDIT_ROOT}/frozen-bindings.txt"
chmod 0444 "${AUDIT_ROOT}/frozen-bindings.txt"

CURRENT_STAGE="hardware-audit"
nvidia-smi -q > "${AUDIT_ROOT}/nvidia-smi-q.txt"
lscpu > "${AUDIT_ROOT}/lscpu.txt"
docker version > "${AUDIT_ROOT}/docker-version.txt"
git --version > "${AUDIT_ROOT}/git-version.txt"
docker image inspect "$IMAGE_ID" > "${AUDIT_ROOT}/image.inspect.json"

for container_name in \
  "$CONFIG_NAME" \
  "$ENCODER_NAME" \
  "$SSHEAD_NAME" \
  "$PREDICT_NAME" \
  "$SCORE_NAME"; do
  if docker inspect "$container_name" >/dev/null 2>&1; then
    fail "container name already exists: $container_name"
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
  --mount \
    "type=bind,src=${SOURCE_RECEIPT},dst=/pams/source-export-receipt.json,readonly"
)

label_free_input_args=(
  --mount "type=bind,src=${CONFIG_HOST},dst=/pams/input/config.yaml,readonly"
  --mount "type=bind,src=${TRAIN_INPUT},dst=/pams/protocol/train.inputs.json,readonly"
  --mount \
    "type=bind,src=${TRAIN_COMMIT},dst=/pams/protocol/train.inputs.commitment.json,readonly"
  --mount "type=bind,src=${DEV_INPUT},dst=/pams/protocol/dev.inputs.json,readonly"
  --mount \
    "type=bind,src=${DEV_COMMIT},dst=/pams/protocol/dev.inputs.commitment.json,readonly"
  --mount \
    "type=bind,src=${TEST_ID_INPUT},dst=/pams/protocol/test-identity.inputs.json,readonly"
  --mount \
    "type=bind,src=${TEST_ID_COMMIT},dst=/pams/protocol/test-identity.inputs.commitment.json,readonly"
)

CURRENT_STAGE="config-preflight-create"
docker create \
  --name "$CONFIG_NAME" \
  "${common_args[@]}" \
  --env CUDA_VISIBLE_DEVICES= \
  "${source_args[@]}" \
  --mount "type=bind,src=${CONFIG_HOST},dst=/pams/input/config.yaml,readonly" \
  --env "PAMS_EXPECTED_CONFIG_FINGERPRINT=${CONFIG_FINGERPRINT}" \
  --env "PAMS_EXPECTED_POSE_FINGERPRINT=${POSE_FINGERPRINT}" \
  "$IMAGE_ID" \
  python -c \
    'import json, os; from pathlib import Path; from pams.config import load_config; from pams.reproducibility import clean_git_revision; c = load_config("/pams/input/config.yaml"); assert c.seed == 2026; assert c.training.epochs == 150; assert c.sshead.epochs == 30; assert c.pose.preprocessing_revision == "longest-contiguous-track-minmax-zero-span-invalid-v3"; assert c.fingerprint == os.environ["PAMS_EXPECTED_CONFIG_FINGERPRINT"]; assert c.pose_fingerprint == os.environ["PAMS_EXPECTED_POSE_FINGERPRINT"]; assert clean_git_revision(Path.cwd()) == os.environ["PAMS_CONTAINER_SOURCE_REVISION"]; print(json.dumps({"schema_version": 1, "seed": c.seed, "config_fingerprint": c.fingerprint, "pose_fingerprint": c.pose_fingerprint, "encoder_epochs": c.training.epochs, "sshead_epochs": c.sshead.epochs}, sort_keys=True))' \
  > "${AUDIT_ROOT}/${CONFIG_NAME}.create-id.txt"
verify_container "$CONFIG_NAME" "config-preflight"
run_created_container "$CONFIG_NAME" "config-preflight"
cp -- "${LOG_ROOT}/${CONFIG_NAME}.log" "${AUDIT_ROOT}/config-identity.json"
chmod 0444 "${AUDIT_ROOT}/config-identity.json"

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
  write_status "failed" "$CURRENT_STAGE" "75"
  exit 75
fi

CURRENT_STAGE="encoder-create"
docker create \
  --name "$ENCODER_NAME" \
  --gpus "device=${GPU_DEVICE}" \
  "${common_args[@]}" \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "${source_args[@]}" \
  "${label_free_input_args[@]}" \
  --mount "type=bind,src=${TRAIN_POSE_VIEW},dst=/pams/pose-cache,readonly" \
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
verify_container "$ENCODER_NAME" "encoder"
run_created_container "$ENCODER_NAME" "encoder"
readonly ENCODER_RECEIPT_HOST="$(single_completion_receipt "$ENCODER_STAGE")"
readonly ENCODER_RECEIPT_NAME="$(basename "$ENCODER_RECEIPT_HOST")"
test -s "${ENCODER_STAGE}/run/encoder.pt"
test -s "${ENCODER_STAGE}/run/logs/encoder.jsonl"
validate_completion_receipt \
  "$ENCODER_RECEIPT_HOST" \
  "output_encoder_checkpoint" \
  "${ENCODER_STAGE}/run/encoder.pt" \
  "${ENCODER_STAGE}/run/logs/encoder.jsonl" \
  150
sha256sum \
  "${ENCODER_STAGE}/run/encoder.pt" \
  "${ENCODER_STAGE}/run/logs/encoder.jsonl" \
  "$ENCODER_RECEIPT_HOST" \
  > "${AUDIT_ROOT}/encoder-terminal-sha256.txt"
chmod -R a-w "${ENCODER_STAGE}/run"

CURRENT_STAGE="sshead-create"
docker create \
  --name "$SSHEAD_NAME" \
  --gpus "device=${GPU_DEVICE}" \
  "${common_args[@]}" \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "${source_args[@]}" \
  "${label_free_input_args[@]}" \
  --mount "type=bind,src=${TRAIN_POSE_VIEW},dst=/pams/pose-cache,readonly" \
  --mount \
    "type=bind,src=${ENCODER_STAGE}/run,dst=/pams/upstream/encoder,readonly" \
  --mount "type=bind,src=${SSHEAD_STAGE},dst=/pams/output" \
  "$IMAGE_ID" \
  python -m pams train sshead \
    /pams/upstream/encoder/encoder.pt \
    /pams/protocol/train.inputs.json \
    /pams/pose-cache \
    /pams/output/run \
    --encoder-progress /pams/upstream/encoder/logs/encoder.jsonl \
    --config /pams/input/config.yaml \
    --device cuda:0 \
    --epochs 30 \
    --microbatch-size 8 \
    --label-free-inputs \
    --input-commitment /pams/protocol/train.inputs.commitment.json \
    --dev-inputs /pams/protocol/dev.inputs.json \
    --dev-input-commitment /pams/protocol/dev.inputs.commitment.json \
    --test-identity-inputs /pams/protocol/test-identity.inputs.json \
    --test-identity-commitment /pams/protocol/test-identity.inputs.commitment.json \
  > "${AUDIT_ROOT}/${SSHEAD_NAME}.create-id.txt"
verify_container "$SSHEAD_NAME" "sshead"
run_created_container "$SSHEAD_NAME" "sshead"
readonly SSHEAD_RECEIPT_HOST="$(single_completion_receipt "$SSHEAD_STAGE")"
readonly SSHEAD_RECEIPT_NAME="$(basename "$SSHEAD_RECEIPT_HOST")"
test -s "${SSHEAD_STAGE}/run/sshead.pt"
test -s "${SSHEAD_STAGE}/run/logs/sshead.jsonl"
validate_completion_receipt \
  "$SSHEAD_RECEIPT_HOST" \
  "output_sshead_checkpoint" \
  "${SSHEAD_STAGE}/run/sshead.pt" \
  "${SSHEAD_STAGE}/run/logs/sshead.jsonl" \
  30
sha256sum \
  "${SSHEAD_STAGE}/run/sshead.pt" \
  "${SSHEAD_STAGE}/run/logs/sshead.jsonl" \
  "$SSHEAD_RECEIPT_HOST" \
  > "${AUDIT_ROOT}/sshead-terminal-sha256.txt"
chmod -R a-w "${SSHEAD_STAGE}/run"

CURRENT_STAGE="dev-predict-create"
docker create \
  --name "$PREDICT_NAME" \
  --gpus "device=${GPU_DEVICE}" \
  "${common_args[@]}" \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "${source_args[@]}" \
  "${label_free_input_args[@]}" \
  --mount \
    "type=bind,src=${TRAIN_DEV_POSE_VIEW},dst=/pams/pose-cache,readonly" \
  --mount \
    "type=bind,src=${ENCODER_STAGE}/run,dst=/pams/upstream/encoder,readonly" \
  --mount \
    "type=bind,src=${SSHEAD_STAGE}/run,dst=/pams/upstream/sshead,readonly" \
  --mount "type=bind,src=${PREDICT_STAGE},dst=/pams/output" \
  "$IMAGE_ID" \
  python -m pams evaluate dev-predict \
    /pams/upstream/sshead/sshead.pt \
    /pams/protocol/train.inputs.json \
    /pams/pose-cache \
    /pams/output/run \
    --checkpoint-progress /pams/upstream/sshead/logs/sshead.jsonl \
    --checkpoint-completion-receipt \
      "/pams/upstream/sshead/manifests/${SSHEAD_RECEIPT_NAME}" \
    --input-commitment /pams/protocol/train.inputs.commitment.json \
    --dev-inputs /pams/protocol/dev.inputs.json \
    --dev-input-commitment /pams/protocol/dev.inputs.commitment.json \
    --test-identity-inputs /pams/protocol/test-identity.inputs.json \
    --test-identity-commitment /pams/protocol/test-identity.inputs.commitment.json \
    --variant sshead \
    --config /pams/input/config.yaml \
    --upstream-encoder-checkpoint /pams/upstream/encoder/encoder.pt \
    --upstream-encoder-progress /pams/upstream/encoder/logs/encoder.jsonl \
    --upstream-encoder-completion-receipt \
      "/pams/upstream/encoder/manifests/${ENCODER_RECEIPT_NAME}" \
    --device cuda:0 \
  > "${AUDIT_ROOT}/${PREDICT_NAME}.create-id.txt"
verify_container "$PREDICT_NAME" "dev-predict"
run_created_container "$PREDICT_NAME" "dev-predict"
test -s "${PREDICT_STAGE}/run/predictions.json"
test -s "${PREDICT_STAGE}/run/prediction.receipt.json"
validate_prediction_bundle \
  "${PREDICT_STAGE}/run/predictions.json" \
  "${PREDICT_STAGE}/run/prediction.receipt.json"
sha256sum \
  "${PREDICT_STAGE}/run/predictions.json" \
  "${PREDICT_STAGE}/run/prediction.receipt.json" \
  "${PREDICT_STAGE}/run/inputs/dev-pose-cache-snapshot.json" \
  > "${AUDIT_ROOT}/frozen-prediction-sha256.txt"
chmod -R a-w "${PREDICT_STAGE}/run"
flock -u 9

# This is the first target-bearing boundary.  It is reached only after
# predictions and their receipt have been frozen read-only.
CURRENT_STAGE="dev-score-target-boundary"
assert_sha256 "$DEV_TARGET" "$DEV_TARGET_SHA256"

CURRENT_STAGE="dev-score-create"
# The two source mounts below are immutable code/provenance, not experiment
# data.  The only read-only experiment-data inputs are the frozen prediction,
# its receipt, and dev84 targets.
docker create \
  --name "$SCORE_NAME" \
  "${common_args[@]}" \
  --env CUDA_VISIBLE_DEVICES= \
  "${source_args[@]}" \
  --mount \
    "type=bind,src=${PREDICT_STAGE}/run/predictions.json,dst=/pams/frozen/predictions.json,readonly" \
  --mount \
    "type=bind,src=${PREDICT_STAGE}/run/prediction.receipt.json,dst=/pams/frozen/prediction.receipt.json,readonly" \
  --mount \
    "type=bind,src=${DEV_TARGET},dst=/pams/protocol/dev.targets.json,readonly" \
  --mount "type=bind,src=${SCORE_STAGE},dst=/pams/output" \
  "$IMAGE_ID" \
  python -m pams evaluate dev-score \
    /pams/frozen/predictions.json \
    /pams/frozen/prediction.receipt.json \
    /pams/protocol/dev.targets.json \
    /pams/output/run \
  > "${AUDIT_ROOT}/${SCORE_NAME}.create-id.txt"
verify_container "$SCORE_NAME" "dev-score"
run_created_container "$SCORE_NAME" "dev-score"
test -s "${SCORE_STAGE}/run/evaluation.json"
test -s "${SCORE_STAGE}/run/evaluation.receipt.json"
validate_score_bundle \
  "${SCORE_STAGE}/run/evaluation.json" \
  "${SCORE_STAGE}/run/evaluation.receipt.json"
sha256sum \
  "${SCORE_STAGE}/run/evaluation.json" \
  "${SCORE_STAGE}/run/evaluation.receipt.json" \
  > "${AUDIT_ROOT}/dev-score-sha256.txt"
chmod -R a-w "${SCORE_STAGE}/run"

CURRENT_STAGE="extension-gate"
GATE_EVALUATION="${SCORE_STAGE}/run/evaluation.json" \
GATE_OUTPUT="${RUN_ROOT}/extension-decision.json" \
GATE_REFERENCE_NMAE="$REFERENCE_V7_NMAE" \
GATE_REFERENCE_OBO="$REFERENCE_V7_OBO" \
GATE_MAX_NMAE="$EXTENSION_MAX_NMAE" \
GATE_MIN_OBO="$EXTENSION_MIN_OBO" \
python3 - <<'PY'
import json
import math
import os
from pathlib import Path


evaluation = json.loads(
    Path(os.environ["GATE_EVALUATION"]).read_text(encoding="utf-8")
)
report = evaluation["report"]
nmae = float(report["nmae"])
obo = float(report["obo"])
reference_nmae = float(os.environ["GATE_REFERENCE_NMAE"])
reference_obo = float(os.environ["GATE_REFERENCE_OBO"])
maximum_nmae = float(os.environ["GATE_MAX_NMAE"])
minimum_obo = float(os.environ["GATE_MIN_OBO"])
if not math.isfinite(nmae) or not math.isfinite(obo):
    raise RuntimeError("extension gate received non-finite metrics")
conditions = {
    "nmae_lte_0_60": nmae <= maximum_nmae,
    "obo_gte_0_30": obo >= minimum_obo,
    "nmae_strictly_better_than_v7_seed2026": nmae < reference_nmae,
    "obo_strictly_better_than_v7_seed2026": obo > reference_obo,
}
payload = {
    "schema_version": 1,
    "decision": (
        "authorize-seeds-42-and-3407"
        if all(conditions.values())
        else "do-not-extend"
    ),
    "authorized": all(conditions.values()),
    "authorized_seeds": [42, 3407] if all(conditions.values()) else [],
    "seed2026": {"nmae": nmae, "obo": obo},
    "reference_v7_seed2026": {
        "nmae": reference_nmae,
        "obo": reference_obo,
    },
    "absolute_thresholds": {
        "nmae_lte": maximum_nmae,
        "obo_gte": minimum_obo,
    },
    "conditions": conditions,
    "test105_evaluation_authorized": False,
}
Path(os.environ["GATE_OUTPUT"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
chmod 0444 "${RUN_ROOT}/extension-decision.json"

CURRENT_STAGE="final-audit"
write_status "completed" "extension-gate" "0"
find "$RUN_ROOT" \
  -type f \
  -not -path "${AUDIT_ROOT}/artifact-sha256.txt" \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "${AUDIT_ROOT}/artifact-sha256.txt"
chmod -R a-w "$RUN_ROOT"
printf '%s\n' "$RUN_ROOT"
