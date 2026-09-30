#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="/media/lenovo/data2/pams-rac"
readonly SOURCE_CHECKOUT="${ROOT}/checkouts/pams-v7-6ee25a2-clone"
readonly SOURCE_REVISION="6ee25a2e1291c35c563140999e16c5aa65bc8ced"
readonly IMAGE_ID="sha256:620173fe7a9084d3b1492c3e775ea5f2bc6ac5a6d851d8560727a4969b3aab66"
readonly ENVIRONMENT_SHA256="1ccfd265d829e5160b8af39156193f63622ac73eaefec00b0f9dab40106c3508"
readonly CONFIG_SHA256="a6926c095a039379d1d1b22931568c6cd05f84a4a0913bccdfc6fc0f2aa9f053"
readonly STAGED_CONFIG="${ROOT}/tmp/pams_sshead_temporal_conv_v7_seed3407.posthoc.yaml"
readonly FIREWALL="${ROOT}/runs/firewall/ucfrep-337-84-105-375d8da"
readonly POSE_SOURCE="${ROOT}/pose-cache"
readonly RUN_ROOT="${ROOT}/runs/followups/pams-sshead-temporal-conv-v7-seed3407-posthoc-6ee25a2-20260729T231000Z-v1"
readonly SOURCE_VIEW="${RUN_ROOT}/source"
readonly INPUT_ROOT="${RUN_ROOT}/inputs"
readonly CONFIG_HOST="${INPUT_ROOT}/pams_sshead_temporal_conv_v7_seed3407.yaml"
readonly TRAIN_POSE_VIEW="${INPUT_ROOT}/pose-train337"
readonly TRAIN_DEV_POSE_VIEW="${INPUT_ROOT}/pose-train337-dev84"
readonly ENCODER_STAGE="${RUN_ROOT}/stages/encoder"
readonly SSHEAD_STAGE="${RUN_ROOT}/stages/sshead"
readonly PREDICT_STAGE="${RUN_ROOT}/stages/dev-predict"
readonly SCORE_STAGE="${RUN_ROOT}/stages/dev-score"
readonly AUDIT_ROOT="${RUN_ROOT}/audit"
readonly LOG_ROOT="${RUN_ROOT}/logs"
readonly LOCK_PATH="${ROOT}/.pams-gpu-locks/gpu0.lock"

readonly TRAIN_INPUT="${FIREWALL}/train.inputs.json"
readonly TRAIN_COMMIT="${FIREWALL}/train.inputs.commitment.json"
readonly DEV_INPUT="${FIREWALL}/dev.inputs.json"
readonly DEV_COMMIT="${FIREWALL}/dev.inputs.commitment.json"
readonly TEST_ID_INPUT="${FIREWALL}/test-identity.inputs.json"
readonly TEST_ID_COMMIT="${FIREWALL}/test-identity.inputs.commitment.json"
readonly DEV_TARGET="${FIREWALL}/dev.targets.json"

readonly TRAIN_INPUT_SHA256="e238d307b4ed37b2f6c19eefdf7fa8812ec3d231cd2a7091be38e1c279f04207"
readonly TRAIN_COMMIT_SHA256="ce39b1c9038bb1506354de010e46a23053b307b852222332f13df3f79d799044"
readonly DEV_INPUT_SHA256="f21c49569805e4aea326977f6d7ce2964f503fad129a17e95214594364ca4ae7"
readonly DEV_COMMIT_SHA256="a2bb2db86abef64e16a8fe1ac196857ac8e77cc9cd461c568b2193401a8cd169"
readonly TEST_ID_INPUT_SHA256="9eb2a057246b77fc7a1121ba4c9d27bf55177ad2fcd5dbb87237396a4a6015ca"
readonly TEST_ID_COMMIT_SHA256="232d3db7ce09018716594a1f1b332eec47b37410a8d3068e662b026ee2dc9c66"
readonly DEV_TARGET_SHA256="1ab3bc010ccb5d869af7662586594c973e53841838643e1e9b8cc1787b0efab6"

readonly ENC_NAME="pams-v7-s3407-enc-20260729t231000z-v1"
readonly HEAD_NAME="pams-v7-s3407-head-20260729t231000z-v1"
readonly PREDICT_NAME="pams-v7-s3407-predict-20260729t231000z-v1"
readonly SCORE_NAME="pams-v7-s3407-score-20260729t231000z-v1"

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
  [[ "$actual" == "$expected" ]] || fail "SHA-256 mismatch for $path: $actual"
}

write_status() {
  local status="$1"
  local stage="$2"
  local exit_code="$3"
  local temporary="${RUN_ROOT}/status.json.tmp"
  printf '{"schema_version":1,"status":"%s","stage":"%s","exit_code":%s,"updated_utc":"%s"}\n' \
    "$status" "$stage" "$exit_code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$temporary"
  mv -- "$temporary" "${RUN_ROOT}/status.json"
}

copy_pose_view() {
  local destination="$1"
  shift
  mkdir -- "$destination"
  local manifest
  local video_id
  local digest
  local source
  for manifest in "$@"; do
    while IFS= read -r video_id; do
      digest="$(printf '%s' "$video_id" | sha256sum | awk '{print $1}')"
      source="${POSE_SOURCE}/${digest}.npz"
      test -f "$source" || fail "missing pose cache for ${video_id}: ${source}"
      cp --reflink=auto --preserve=mode,timestamps -- "$source" "${destination}/${digest}.npz"
    done < <(jq -r '.records[].video_id' "$manifest")
  done
}

single_completion_receipt() {
  local output_root="$1"
  local -a matches=()
  mapfile -t matches < <(find "${output_root}/run/manifests" -maxdepth 1 -type f -name '*.completed.json' -print | sort)
  [[ "${#matches[@]}" -eq 1 ]] || fail "expected one completion receipt in ${output_root}, got ${#matches[@]}"
  printf '%s\n' "${matches[0]}"
}

container_common_args=(
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
  --env CUDA_VISIBLE_DEVICES=0
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8
  --env "PAMS_CONTAINER_IMAGE_ID=${IMAGE_ID}"
  --env "PAMS_CONTAINER_ENVIRONMENT_SHA256=${ENVIRONMENT_SHA256}"
  --env "PAMS_CONTAINER_SOURCE_REVISION=${SOURCE_REVISION}"
  --tmpfs /tmp:rw,nosuid,nodev,uid=1000,gid=1000,size=1g
  --tmpfs /pams/tmp:rw,nosuid,nodev,uid=1000,gid=1000,size=8g
  --tmpfs /pams/cache:rw,nosuid,nodev,uid=1000,gid=1000,size=8g
  --tmpfs /pams/home:rw,nosuid,nodev,uid=1000,gid=1000,size=256m
  --mount "type=bind,src=${SOURCE_VIEW},dst=/workspace,readonly"
)

label_free_input_args=(
  --mount "type=bind,src=${CONFIG_HOST},dst=/pams/input/config.yaml,readonly"
  --mount "type=bind,src=${TRAIN_INPUT},dst=/pams/protocol/train.inputs.json,readonly"
  --mount "type=bind,src=${TRAIN_COMMIT},dst=/pams/protocol/train.inputs.commitment.json,readonly"
  --mount "type=bind,src=${DEV_INPUT},dst=/pams/protocol/dev.inputs.json,readonly"
  --mount "type=bind,src=${DEV_COMMIT},dst=/pams/protocol/dev.inputs.commitment.json,readonly"
  --mount "type=bind,src=${TEST_ID_INPUT},dst=/pams/protocol/test-identity.inputs.json,readonly"
  --mount "type=bind,src=${TEST_ID_COMMIT},dst=/pams/protocol/test-identity.inputs.commitment.json,readonly"
)

verify_container() {
  local container_name="$1"
  local stage="$2"
  local inspect_path="${AUDIT_ROOT}/${container_name}.inspect.json"
  docker inspect "$container_name" > "$inspect_path"
  VERIFY_STAGE="$stage" \
  VERIFY_INSPECT="$inspect_path" \
  VERIFY_RUN_ROOT="$RUN_ROOT" \
  VERIFY_SOURCE_VIEW="$SOURCE_VIEW" \
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
  python3 - <<'PY'
import json
import os
from pathlib import Path

stage = os.environ["VERIFY_STAGE"]
inspect_path = Path(os.environ["VERIFY_INSPECT"])
item = json.loads(inspect_path.read_text(encoding="utf-8"))[0]
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
assert int(host["Memory"]) == 96 * 1024 ** 3
assert mounts["/workspace"] == (os.environ["VERIFY_SOURCE_VIEW"], False)

firewall = os.environ["VERIFY_FIREWALL"]
protocol_mounts = {}
if stage != "dev-score":
    assert mounts["/pams/input/config.yaml"] == (
        os.environ["VERIFY_CONFIG_HOST"],
        False,
    )
    protocol_mounts = {
        "/pams/protocol/train.inputs.json": f"{firewall}/train.inputs.json",
        "/pams/protocol/train.inputs.commitment.json": (
            f"{firewall}/train.inputs.commitment.json"
        ),
        "/pams/protocol/dev.inputs.json": f"{firewall}/dev.inputs.json",
        "/pams/protocol/dev.inputs.commitment.json": (
            f"{firewall}/dev.inputs.commitment.json"
        ),
        "/pams/protocol/test-identity.inputs.json": (
            f"{firewall}/test-identity.inputs.json"
        ),
        "/pams/protocol/test-identity.inputs.commitment.json": (
            f"{firewall}/test-identity.inputs.commitment.json"
        ),
    }
    for destination, source in protocol_mounts.items():
        assert mounts[destination] == (source, False)

expected_stage_mounts = {
    "encoder": {
        "/pams/pose-cache": (os.environ["VERIFY_TRAIN_POSE"], False),
        "/pams/output": (os.environ["VERIFY_ENCODER_STAGE"], True),
    },
    "sshead": {
        "/pams/pose-cache": (os.environ["VERIFY_TRAIN_POSE"], False),
        "/pams/upstream/encoder": (
            f'{os.environ["VERIFY_ENCODER_STAGE"]}/run',
            False,
        ),
        "/pams/output": (os.environ["VERIFY_SSHEAD_STAGE"], True),
    },
    "dev-predict": {
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
    },
    "dev-score": {
        "/pams/frozen/predictions.json": (
            f'{os.environ["VERIFY_PREDICT_STAGE"]}/run/predictions.json',
            False,
        ),
        "/pams/frozen/prediction.receipt.json": (
            f'{os.environ["VERIFY_PREDICT_STAGE"]}/run/prediction.receipt.json',
            False,
        ),
        "/pams/protocol/dev.targets.json": (
            f'{firewall}/dev.targets.json',
            False,
        ),
        "/pams/output": (os.environ["VERIFY_SCORE_STAGE"], True),
    },
}
for destination, expected in expected_stage_mounts[stage].items():
    assert mounts[destination] == expected

allowed = {
    "/workspace",
    *protocol_mounts.keys(),
    *expected_stage_mounts[stage].keys(),
}
if stage != "dev-score":
    allowed.add("/pams/input/config.yaml")
assert set(mounts) == allowed, (stage, sorted(set(mounts) - allowed))

command_text = "\0".join(config.get("Cmd") or [])
all_text = "\0".join(
    [command_text]
    + [source for source, _ in mounts.values()]
    + list(mounts)
)
if stage != "dev-score":
    assert "dev.targets" not in all_text
else:
    assert "/pams/protocol/dev.targets.json" in command_text
assert "/pams/pose-cache" not in mounts or "test" not in mounts["/pams/pose-cache"][0]

environment = set(config.get("Env") or [])
assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in environment
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

device_requests = host.get("DeviceRequests") or []
if stage == "dev-score":
    assert not device_requests
else:
    assert len(device_requests) == 1
    assert "gpu" in device_requests[0]["Capabilities"][0]

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
report_path = inspect_path.with_suffix(".verification.json")
report_path.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

run_created_container() {
  local container_name="$1"
  local stage="$2"
  local log_path="${LOG_ROOT}/${container_name}.log"
  write_status "running" "$stage" "null"
  set +e
  docker start --attach "$container_name" > "$log_path" 2>&1
  local attach_exit="$?"
  set -e
  local container_exit
  container_exit="$(docker inspect "$container_name" --format '{{.State.ExitCode}}')"
  printf '%s\n' "$container_exit" > "${AUDIT_ROOT}/${container_name}.exit-code.txt"
  docker inspect "$container_name" > "${AUDIT_ROOT}/${container_name}.post-run.inspect.json"
  if [[ "$attach_exit" -ne 0 || "$container_exit" -ne 0 ]]; then
    write_status "failed" "$stage" "$container_exit"
    fail "container ${container_name} failed: attach=${attach_exit}, container=${container_exit}"
  fi
}

test ! -e "$RUN_ROOT" || fail "immutable run root already exists: $RUN_ROOT"
for container_name in "$ENC_NAME" "$HEAD_NAME" "$PREDICT_NAME" "$SCORE_NAME"; do
  if docker inspect "$container_name" >/dev/null 2>&1; then
    fail "container name already exists: $container_name"
  fi
done

assert_sha256 "$STAGED_CONFIG" "$CONFIG_SHA256"
assert_sha256 "$TRAIN_INPUT" "$TRAIN_INPUT_SHA256"
assert_sha256 "$TRAIN_COMMIT" "$TRAIN_COMMIT_SHA256"
assert_sha256 "$DEV_INPUT" "$DEV_INPUT_SHA256"
assert_sha256 "$DEV_COMMIT" "$DEV_COMMIT_SHA256"
assert_sha256 "$TEST_ID_INPUT" "$TEST_ID_INPUT_SHA256"
assert_sha256 "$TEST_ID_COMMIT" "$TEST_ID_COMMIT_SHA256"
assert_sha256 "$DEV_TARGET" "$DEV_TARGET_SHA256"
[[ "$(git -C "$SOURCE_CHECKOUT" rev-parse HEAD)" == "$SOURCE_REVISION" ]] \
  || fail "source checkout revision mismatch"
[[ -z "$(git -C "$SOURCE_CHECKOUT" status --porcelain)" ]] \
  || fail "source checkout is dirty"
[[ "$(docker image inspect "$IMAGE_ID" --format '{{.Id}}')" == "$IMAGE_ID" ]] \
  || fail "container image mismatch"

mkdir -p "$RUN_ROOT" "$INPUT_ROOT" "$AUDIT_ROOT" "$LOG_ROOT" \
  "$ENCODER_STAGE" "$SSHEAD_STAGE" "$PREDICT_STAGE" "$SCORE_STAGE"
write_status "preparing" "inputs" "null"

cp -- "$STAGED_CONFIG" "$CONFIG_HOST"
chmod 0444 "$CONFIG_HOST"
assert_sha256 "$CONFIG_HOST" "$CONFIG_SHA256"

git clone --quiet --no-hardlinks --no-checkout "$SOURCE_CHECKOUT" "$SOURCE_VIEW"
git -C "$SOURCE_VIEW" sparse-checkout init --no-cone
git -C "$SOURCE_VIEW" sparse-checkout set '/src/' '/configs/' '/pyproject.toml'
git -C "$SOURCE_VIEW" checkout --quiet --detach "$SOURCE_REVISION"
git -C "$SOURCE_VIEW" remote remove origin
[[ "$(git -C "$SOURCE_VIEW" rev-parse HEAD)" == "$SOURCE_REVISION" ]] \
  || fail "sparse source revision mismatch"
[[ -z "$(git -C "$SOURCE_VIEW" status --porcelain)" ]] \
  || fail "sparse source is dirty"
test ! -e "${SOURCE_VIEW}/results" || fail "sparse source unexpectedly exposes results"

copy_pose_view "$TRAIN_POSE_VIEW" "$TRAIN_INPUT"
copy_pose_view "$TRAIN_DEV_POSE_VIEW" "$TRAIN_INPUT" "$DEV_INPUT"
[[ "$(find "$TRAIN_POSE_VIEW" -maxdepth 1 -type f -name '*.npz' | wc -l)" -eq 337 ]] \
  || fail "train pose view is not exactly 337 files"
[[ "$(find "$TRAIN_DEV_POSE_VIEW" -maxdepth 1 -type f -name '*.npz' | wc -l)" -eq 421 ]] \
  || fail "train+dev pose view is not exactly 421 files"
chmod -R a-w "$TRAIN_POSE_VIEW" "$TRAIN_DEV_POSE_VIEW"

cat > "${RUN_ROOT}/attempt.reservation.json" <<EOF
{
  "schema_version": 1,
  "attempt_id": "pams-sshead-temporal-conv-v7-seed3407-posthoc-20260729T231000Z-v1",
  "reserved_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "dataset": "UCFRep",
  "split": "dev84",
  "seed": 3407,
  "classification": "post-hoc-three-seed-variance-follow-up",
  "pre_registered_success_claim": false,
  "paper_table_2_claim": false,
  "reason": "Seeds 42 and 2026 already failed the frozen gate; this extra seed only closes the variance audit.",
  "source_revision": "${SOURCE_REVISION}",
  "container_image_id": "${IMAGE_ID}",
  "container_environment_sha256": "${ENVIRONMENT_SHA256}",
  "config_sha256": "${CONFIG_SHA256}",
  "dev_targets_sha256": "${DEV_TARGET_SHA256}",
  "test_evaluation_authorized": false
}
EOF
chmod 0444 "${RUN_ROOT}/attempt.reservation.json"

{
  printf 'source_revision %s\n' "$SOURCE_REVISION"
  printf 'image_id %s\n' "$IMAGE_ID"
  printf 'environment_sha256 %s\n' "$ENVIRONMENT_SHA256"
  printf 'config_sha256 %s\n' "$(sha256_of "$CONFIG_HOST")"
  printf 'train_inputs_sha256 %s\n' "$(sha256_of "$TRAIN_INPUT")"
  printf 'train_commitment_sha256 %s\n' "$(sha256_of "$TRAIN_COMMIT")"
  printf 'dev_inputs_sha256 %s\n' "$(sha256_of "$DEV_INPUT")"
  printf 'dev_commitment_sha256 %s\n' "$(sha256_of "$DEV_COMMIT")"
  printf 'test_identity_inputs_sha256 %s\n' "$(sha256_of "$TEST_ID_INPUT")"
  printf 'test_identity_commitment_sha256 %s\n' "$(sha256_of "$TEST_ID_COMMIT")"
  printf 'dev_targets_sha256 %s\n' "$(sha256_of "$DEV_TARGET")"
} > "${AUDIT_ROOT}/frozen-bindings.txt"
nvidia-smi -q > "${AUDIT_ROOT}/nvidia-smi-q.txt"
lscpu > "${AUDIT_ROOT}/lscpu.txt"
docker version > "${AUDIT_ROOT}/docker-version.txt"
git --version > "${AUDIT_ROOT}/git-version.txt"

mkdir -p "$(dirname "$LOCK_PATH")"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  write_status "failed" "gpu-lock" "75"
  exit 75
fi

docker create \
  --name "$ENC_NAME" \
  --gpus device=0 \
  "${container_common_args[@]}" \
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
  > "${AUDIT_ROOT}/${ENC_NAME}.create-id.txt"
verify_container "$ENC_NAME" "encoder"
run_created_container "$ENC_NAME" "encoder"
readonly ENCODER_RECEIPT_HOST="$(single_completion_receipt "$ENCODER_STAGE")"
readonly ENCODER_RECEIPT_NAME="$(basename "$ENCODER_RECEIPT_HOST")"
test -s "${ENCODER_STAGE}/run/encoder.pt"
test -s "${ENCODER_STAGE}/run/logs/encoder.jsonl"
chmod -R a-w "${ENCODER_STAGE}/run"

docker create \
  --name "$HEAD_NAME" \
  --gpus device=0 \
  "${container_common_args[@]}" \
  "${label_free_input_args[@]}" \
  --mount "type=bind,src=${TRAIN_POSE_VIEW},dst=/pams/pose-cache,readonly" \
  --mount "type=bind,src=${ENCODER_STAGE}/run,dst=/pams/upstream/encoder,readonly" \
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
  > "${AUDIT_ROOT}/${HEAD_NAME}.create-id.txt"
verify_container "$HEAD_NAME" "sshead"
run_created_container "$HEAD_NAME" "sshead"
readonly SSHEAD_RECEIPT_HOST="$(single_completion_receipt "$SSHEAD_STAGE")"
readonly SSHEAD_RECEIPT_NAME="$(basename "$SSHEAD_RECEIPT_HOST")"
test -s "${SSHEAD_STAGE}/run/sshead.pt"
test -s "${SSHEAD_STAGE}/run/logs/sshead.jsonl"
chmod -R a-w "${SSHEAD_STAGE}/run"

docker create \
  --name "$PREDICT_NAME" \
  --gpus device=0 \
  "${container_common_args[@]}" \
  "${label_free_input_args[@]}" \
  --mount "type=bind,src=${TRAIN_DEV_POSE_VIEW},dst=/pams/pose-cache,readonly" \
  --mount "type=bind,src=${ENCODER_STAGE}/run,dst=/pams/upstream/encoder,readonly" \
  --mount "type=bind,src=${SSHEAD_STAGE}/run,dst=/pams/upstream/sshead,readonly" \
  --mount "type=bind,src=${PREDICT_STAGE},dst=/pams/output" \
  "$IMAGE_ID" \
  python -m pams evaluate dev-predict \
    /pams/upstream/sshead/sshead.pt \
    /pams/protocol/train.inputs.json \
    /pams/pose-cache \
    /pams/output/run \
    --checkpoint-progress /pams/upstream/sshead/logs/sshead.jsonl \
    --checkpoint-completion-receipt "/pams/upstream/sshead/manifests/${SSHEAD_RECEIPT_NAME}" \
    --input-commitment /pams/protocol/train.inputs.commitment.json \
    --dev-inputs /pams/protocol/dev.inputs.json \
    --dev-input-commitment /pams/protocol/dev.inputs.commitment.json \
    --test-identity-inputs /pams/protocol/test-identity.inputs.json \
    --test-identity-commitment /pams/protocol/test-identity.inputs.commitment.json \
    --variant sshead \
    --config /pams/input/config.yaml \
    --upstream-encoder-checkpoint /pams/upstream/encoder/encoder.pt \
    --upstream-encoder-progress /pams/upstream/encoder/logs/encoder.jsonl \
    --upstream-encoder-completion-receipt "/pams/upstream/encoder/manifests/${ENCODER_RECEIPT_NAME}" \
    --device cuda:0 \
  > "${AUDIT_ROOT}/${PREDICT_NAME}.create-id.txt"
verify_container "$PREDICT_NAME" "dev-predict"
run_created_container "$PREDICT_NAME" "dev-predict"
test -s "${PREDICT_STAGE}/run/predictions.json"
test -s "${PREDICT_STAGE}/run/prediction.receipt.json"
chmod -R a-w "${PREDICT_STAGE}/run"
sha256sum \
  "${PREDICT_STAGE}/run/predictions.json" \
  "${PREDICT_STAGE}/run/prediction.receipt.json" \
  > "${AUDIT_ROOT}/frozen-prediction-sha256.txt"

flock -u 9

docker create \
  --name "$SCORE_NAME" \
  "${container_common_args[@]}" \
  --mount "type=bind,src=${PREDICT_STAGE}/run/predictions.json,dst=/pams/frozen/predictions.json,readonly" \
  --mount "type=bind,src=${PREDICT_STAGE}/run/prediction.receipt.json,dst=/pams/frozen/prediction.receipt.json,readonly" \
  --mount "type=bind,src=${DEV_TARGET},dst=/pams/protocol/dev.targets.json,readonly" \
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
chmod -R a-w "${SCORE_STAGE}/run"

write_status "completed" "dev-score" "0"
find "$RUN_ROOT" -type f \
  -not -path "${SOURCE_VIEW}/.git/*" \
  -not -path "${AUDIT_ROOT}/artifact-sha256.txt" \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "${AUDIT_ROOT}/artifact-sha256.txt"
printf '%s\n' "$RUN_ROOT"
