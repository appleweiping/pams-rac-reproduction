#!/usr/bin/env bash
set -Eeuo pipefail

# Extract train337 and dev84 pose caches for the v8 longest-contiguous-track
# protocol correction. Each container can read only an exact source export,
# one frozen label-free sidecar, and that sidecar's video snapshot.

readonly ROOT_INPUT="${PAMS_ROOT:-/media/lenovo/data2/pams-rac}"
readonly SOURCE_REVISION="${PAMS_SOURCE_REVISION:?PAMS_SOURCE_REVISION is required}"
readonly SOURCE_CHECKOUT_INPUT="${PAMS_SOURCE_CHECKOUT:?PAMS_SOURCE_CHECKOUT is required}"
readonly IMAGE_ID="${PAMS_IMAGE_ID:?PAMS_IMAGE_ID is required}"
readonly ENVIRONMENT_SHA256="${PAMS_ENVIRONMENT_SHA256:?PAMS_ENVIRONMENT_SHA256 is required}"
readonly CONFIG_SHA256="${PAMS_CONFIG_SHA256:?PAMS_CONFIG_SHA256 is required}"
readonly POSE_FINGERPRINT="${PAMS_POSE_FINGERPRINT:?PAMS_POSE_FINGERPRINT is required}"
readonly ATTEMPT_ID="${PAMS_ATTEMPT_ID:?PAMS_ATTEMPT_ID is required}"
readonly CONFIG_RELATIVE="configs/experiments/pams_longest_contiguous_track_v8.yaml"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[[ "$ROOT_INPUT" == /* ]] || fail "PAMS_ROOT must be absolute"
[[ "$ROOT_INPUT" != *","* ]] || fail "PAMS_ROOT must not contain commas"
[[ "$SOURCE_CHECKOUT_INPUT" == /* ]] \
  || fail "PAMS_SOURCE_CHECKOUT must be absolute"
[[ "$ATTEMPT_ID" =~ ^[a-z0-9][a-z0-9._-]{0,47}$ ]] \
  || fail "PAMS_ATTEMPT_ID must be a lowercase safe slug of at most 48 characters"
[[ "$ATTEMPT_ID" != *".."* ]] || fail "PAMS_ATTEMPT_ID must not contain '..'"
[[ "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]] \
  || fail "PAMS_SOURCE_REVISION is not a full Git SHA"
[[ "$IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || fail "PAMS_IMAGE_ID is not immutable"
for digest_name in ENVIRONMENT_SHA256 CONFIG_SHA256 POSE_FINGERPRINT; do
  digest="${!digest_name}"
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] \
    || fail "PAMS_${digest_name} is not a lowercase SHA-256"
done

readonly ROOT="$(realpath -e -- "$ROOT_INPUT")"
readonly SOURCE_CHECKOUT="$(realpath -e -- "$SOURCE_CHECKOUT_INPUT")"
readonly FIREWALL="${ROOT}/runs/firewall/ucfrep-337-84-105-375d8da"
readonly VIDEO_ROOT="$(realpath -e -- "${ROOT}/data/extracted/UCF-101")"
readonly PROTOCOL_ROOT="${ROOT}/runs/pose-protocol"
readonly RUN_ROOT="${PROTOCOL_ROOT}/${ATTEMPT_ID}"
readonly SOURCE_VIEW="${RUN_ROOT}/source"
readonly VIDEO_VIEW_ROOT="${RUN_ROOT}/video-views"
readonly TRAIN_VIDEO_VIEW="${VIDEO_VIEW_ROOT}/train337"
readonly DEV_VIDEO_VIEW="${VIDEO_VIEW_ROOT}/dev84"
readonly CACHE_ROOT="${RUN_ROOT}/pose-cache"
readonly LEDGER_ROOT="${RUN_ROOT}/ledgers"
readonly AUDIT_ROOT="${RUN_ROOT}/audit"
readonly LOG_ROOT="${RUN_ROOT}/logs"
readonly CONFIG_PATH="${SOURCE_CHECKOUT}/${CONFIG_RELATIVE}"
readonly EXPORTED_CONFIG="${SOURCE_VIEW}/${CONFIG_RELATIVE}"
readonly VIDEO_VIEW_RECEIPTS="${AUDIT_ROOT}/video-view-sha256.tsv"

readonly TRAIN_INPUT="${FIREWALL}/train.inputs.json"
readonly TRAIN_COMMIT="${FIREWALL}/train.inputs.commitment.json"
readonly DEV_INPUT="${FIREWALL}/dev.inputs.json"
readonly DEV_COMMIT="${FIREWALL}/dev.inputs.commitment.json"

readonly TRAIN_INPUT_SHA256="e238d307b4ed37b2f6c19eefdf7fa8812ec3d231cd2a7091be38e1c279f04207"
readonly TRAIN_COMMIT_SHA256="ce39b1c9038bb1506354de010e46a23053b307b852222332f13df3f79d799044"
readonly DEV_INPUT_SHA256="f21c49569805e4aea326977f6d7ce2964f503fad129a17e95214594364ca4ae7"
readonly DEV_COMMIT_SHA256="a2bb2db86abef64e16a8fe1ac196857ac8e77cc9cd461c568b2193401a8cd169"

RUN_RESERVED=0
CURRENT_STAGE="preflight"
ACTIVE_CONTAINER=""

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

assert_sha256() {
  local path="$1"
  local expected="$2"
  test -f "$path" || fail "missing frozen input: $path"
  local actual
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
  if [[ -n "$ACTIVE_CONTAINER" ]] \
    && [[ "$(docker inspect "$ACTIVE_CONTAINER" --format '{{.State.Running}}' 2>/dev/null)" == "true" ]]; then
    docker stop --time 10 "$ACTIVE_CONTAINER" >/dev/null 2>&1
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

materialize_video_view() {
  local split="$1"
  local sidecar_split="$2"
  local input="$3"
  local destination="$4"
  local expected_count="$5"
  local count=0
  local relative
  local wrapped_relative
  local expected_digest
  local source
  local resolved_source
  local target
  local actual_digest

  jq -e \
    --arg split "$sidecar_split" \
    --argjson expected_count "$expected_count" \
    '
      .schema_version == 2
      and .manifest_type == "pose_inputs"
      and .protocol == "ucfrep_526"
      and .split == $split
      and (.records | length) == $expected_count
      and all(
        .records[];
        (keys | sort) == ["video_id", "video_path", "video_sha256"]
      )
    ' \
    "$input" >/dev/null \
    || fail "${split} input is not the exact label-free sidecar schema"
  mkdir -- "$destination"
  while IFS=$'\t' read -r relative expected_digest; do
    [[ "$relative" =~ ^[^/]+(/[^/]+)*$ ]] \
      || fail "${split} contains an unsafe video locator: ${relative}"
    [[ "$relative" != *"\\"* ]] \
      || fail "${split} contains a backslash in a video locator"
    wrapped_relative="/${relative}/"
    [[ "$wrapped_relative" != *"/./"* && "$wrapped_relative" != *"/../"* ]] \
      || fail "${split} contains a traversal video locator: ${relative}"
    [[ "$relative" != *$'\n'* && "$relative" != *$'\r'* && "$relative" != *$'\t'* ]] \
      || fail "${split} contains a control character in a video locator"
    [[ "$expected_digest" =~ ^[0-9a-f]{64}$ ]] \
      || fail "${split} contains an invalid video SHA-256"
    source="${VIDEO_ROOT}/${relative}"
    resolved_source="$(realpath -e -- "$source")"
    [[ "$resolved_source" == "${VIDEO_ROOT}/"* ]] \
      || fail "${split} video locator escapes the frozen video root"
    target="${destination}/${relative}"
    mkdir -p -- "$(dirname -- "$target")"
    cp --reflink=auto --preserve=mode,timestamps -- "$resolved_source" "$target"
    actual_digest="$(sha256_of "$target")"
    [[ "$actual_digest" == "$expected_digest" ]] \
      || fail "${split} copied video SHA-256 mismatch: ${relative}"
    printf '%s\t%s\t%s\n' "$split" "$relative" "$actual_digest" \
      >> "$VIDEO_VIEW_RECEIPTS"
    ((count += 1))
  done < <(jq -r '.records[] | [.video_path, .video_sha256] | @tsv' "$input")

  [[ "$count" -eq "$expected_count" ]] \
    || fail "${split} video view count mismatch: ${count}"
  [[ "$(find "$destination" -type f | wc -l)" -eq "$expected_count" ]] \
    || fail "${split} video view has unexpected files"
  chmod -R a-w "$destination"
}

verify_container() {
  local container_name="$1"
  local split="$2"
  local expected_input="$3"
  local expected_commitment="$4"
  local expected_video_view="$5"
  local inspect_path="${AUDIT_ROOT}/${container_name}.inspect.json"
  docker inspect "$container_name" > "$inspect_path"
  VERIFY_INSPECT="$inspect_path" \
  VERIFY_SPLIT="$split" \
  VERIFY_SOURCE_VIEW="$SOURCE_VIEW" \
  VERIFY_IMAGE_ID="$IMAGE_ID" \
  VERIFY_SOURCE_REVISION="$SOURCE_REVISION" \
  VERIFY_ENVIRONMENT_SHA256="$ENVIRONMENT_SHA256" \
  VERIFY_VIDEO_VIEW="$expected_video_view" \
  VERIFY_CACHE_ROOT="$CACHE_ROOT" \
  VERIFY_LEDGER_ROOT="$LEDGER_ROOT" \
  VERIFY_INPUT="$expected_input" \
  VERIFY_COMMITMENT="$expected_commitment" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

item = json.loads(Path(os.environ["VERIFY_INSPECT"]).read_text(encoding="utf-8"))[0]
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
assert not (host.get("DeviceRequests") or [])
assert set(host.get("Tmpfs") or {}) == {
    "/tmp",
    "/pams/tmp",
    "/pams/cache",
    "/pams/home",
}
expected = {
    "/workspace": (os.environ["VERIFY_SOURCE_VIEW"], False),
    "/pams/videos": (os.environ["VERIFY_VIDEO_VIEW"], False),
    "/pams/pose-cache": (os.environ["VERIFY_CACHE_ROOT"], True),
    "/pams/ledgers": (os.environ["VERIFY_LEDGER_ROOT"], True),
    "/pams/protocol/inputs.json": (os.environ["VERIFY_INPUT"], False),
    "/pams/protocol/inputs.commitment.json": (
        os.environ["VERIFY_COMMITMENT"],
        False,
    ),
}
assert mounts == expected, (mounts, expected)
command = "\0".join(config.get("Cmd") or [])
assert "targets" not in command
assert "test" not in command
assert "--label-free-manifest" in command
assert "--input-commitment" in command
assert (
    "/workspace/configs/experiments/pams_longest_contiguous_track_v8.yaml"
    in command
)
environment = set(config.get("Env") or [])
assert (
    f'PAMS_CONTAINER_SOURCE_REVISION={os.environ["VERIFY_SOURCE_REVISION"]}'
    in environment
)
assert (
    "PAMS_CONTAINER_ENVIRONMENT_SHA256="
    + os.environ["VERIFY_ENVIRONMENT_SHA256"]
    in environment
)
source_view = Path(os.environ["VERIFY_SOURCE_VIEW"])
assert not (source_view / ".git").exists()
assert not (source_view / "results").exists()
assert not (source_view / "data").exists()
assert not (source_view / "tests").exists()
assert {path.name for path in source_view.iterdir()} == {"src", "configs"}
PY
}

validate_ledger() {
  local split="$1"
  local ledger="$2"
  local expected_count="$3"
  local expected_input_sha256="$4"
  local expected_commitment_sha256="$5"
  local commitment="$6"
  [[ "$(jq -r '.selected' "$ledger")" -eq "$expected_count" ]] \
    || fail "${split} ledger selected count mismatch"
  [[ "$(jq -r '.completed' "$ledger")" -eq "$expected_count" ]] \
    || fail "${split} ledger completed count mismatch"
  [[ "$(jq -r '.failed' "$ledger")" -eq 0 ]] \
    || fail "${split} ledger contains extraction failures"
  [[ "$(jq -r '.input_file_sha256' "$ledger")" == "$expected_input_sha256" ]] \
    || fail "${split} ledger input SHA-256 mismatch"
  [[ "$(jq -r '.sidecar_sha256' "$ledger")" == "$expected_input_sha256" ]] \
    || fail "${split} ledger sidecar SHA-256 mismatch"
  [[ "$(jq -r '.commitment_file_sha256' "$ledger")" == "$expected_commitment_sha256" ]] \
    || fail "${split} ledger commitment SHA-256 mismatch"
  [[ "$(jq -r '.identity_sha256' "$ledger")" == "$(jq -r '.identity_sha256' "$commitment")" ]] \
    || fail "${split} ledger identity SHA-256 mismatch"
  [[ "$(jq -r '.pose_fingerprint' "$ledger")" == "$POSE_FINGERPRINT" ]] \
    || fail "${split} ledger pose fingerprint mismatch"
  [[ "$(jq -r '.successful_cache_snapshot.entry_count' "$ledger")" -eq "$expected_count" ]] \
    || fail "${split} ledger cache snapshot count mismatch"
  [[ "$(jq -r '.successful_cache_snapshot.pose_fingerprint' "$ledger")" == "$POSE_FINGERPRINT" ]] \
    || fail "${split} ledger cache snapshot fingerprint mismatch"
  [[ "$(jq -r '.caches | length' "$ledger")" -eq "$expected_count" ]] \
    || fail "${split} ledger cache receipt count mismatch"
  [[ "$(jq '[.caches[].selected_source_frames == null] | any' "$ledger")" == "false" ]] \
    || fail "${split} ledger omits selected source-run lengths"
}

run_split() {
  local split="$1"
  local input="$2"
  local commitment="$3"
  local video_view="$4"
  local expected_count="$5"
  local expected_input_sha256="$6"
  local expected_commitment_sha256="$7"
  local container_name="pams-v8-${split}-${ATTEMPT_ID}"
  local ledger_path="${LEDGER_ROOT}/${split}.json"
  local log_path="${LOG_ROOT}/${split}.log"

  CURRENT_STAGE="pose-${split}-create"
  docker create \
    --name "$container_name" \
    --init \
    --network none \
    --read-only \
    --shm-size 4g \
    --pids-limit 4096 \
    --cpus 24 \
    --memory 96g \
    --cap-drop ALL \
    --security-opt no-new-privileges:true \
    --user 1000:1000 \
    --env PYTHONPATH=/workspace/src \
    --env "PAMS_CONTAINER_IMAGE_ID=${IMAGE_ID}" \
    --env "PAMS_CONTAINER_ENVIRONMENT_SHA256=${ENVIRONMENT_SHA256}" \
    --env "PAMS_CONTAINER_SOURCE_REVISION=${SOURCE_REVISION}" \
    --tmpfs /tmp:rw,nosuid,nodev,uid=1000,gid=1000,size=2g \
    --tmpfs /pams/tmp:rw,nosuid,nodev,uid=1000,gid=1000,size=8g \
    --tmpfs /pams/cache:rw,nosuid,nodev,uid=1000,gid=1000,size=8g \
    --tmpfs /pams/home:rw,nosuid,nodev,uid=1000,gid=1000,size=256m \
    --mount "type=bind,src=${SOURCE_VIEW},dst=/workspace,readonly" \
    --mount "type=bind,src=${video_view},dst=/pams/videos,readonly" \
    --mount "type=bind,src=${CACHE_ROOT},dst=/pams/pose-cache" \
    --mount "type=bind,src=${LEDGER_ROOT},dst=/pams/ledgers" \
    --mount "type=bind,src=${input},dst=/pams/protocol/inputs.json,readonly" \
    --mount \
      "type=bind,src=${commitment},dst=/pams/protocol/inputs.commitment.json,readonly" \
    "$IMAGE_ID" \
    python -m pams pose extract \
      /pams/protocol/inputs.json \
      /pams/pose-cache \
      --config "/workspace/${CONFIG_RELATIVE}" \
      --label-free-manifest \
      --video-root /pams/videos \
      --input-commitment /pams/protocol/inputs.commitment.json \
      --failure-ledger "/pams/ledgers/${split}.json" \
    > "${AUDIT_ROOT}/${container_name}.create-id.txt"

  verify_container \
    "$container_name" \
    "$split" \
    "$input" \
    "$commitment" \
    "$video_view"
  CURRENT_STAGE="pose-${split}"
  write_status "running" "$CURRENT_STAGE" "null"
  ACTIVE_CONTAINER="$container_name"
  set +e
  docker start --attach "$container_name" > "$log_path" 2>&1
  local attach_exit="$?"
  set -e
  local was_still_running
  was_still_running="$(
    docker inspect "$container_name" --format '{{.State.Running}}'
  )"
  if [[ "$was_still_running" == "true" ]]; then
    docker stop --time 10 "$container_name" >/dev/null
  fi
  local container_exit
  container_exit="$(docker inspect "$container_name" --format '{{.State.ExitCode}}')"
  ACTIVE_CONTAINER=""
  printf '%s\n' "$container_exit" \
    > "${AUDIT_ROOT}/${container_name}.exit-code.txt"
  docker inspect "$container_name" \
    > "${AUDIT_ROOT}/${container_name}.post-run.inspect.json"
  if [[ "$attach_exit" -ne 0 || "$was_still_running" == "true" || "$container_exit" -ne 0 ]]; then
    fail \
      "pose ${split} failed: attach=${attach_exit}, running_after_attach=${was_still_running}, container=${container_exit}"
  fi
  validate_ledger \
    "$split" \
    "$ledger_path" \
    "$expected_count" \
    "$expected_input_sha256" \
    "$expected_commitment_sha256" \
    "$commitment"
}

assert_sha256 "$CONFIG_PATH" "$CONFIG_SHA256"
assert_sha256 "$TRAIN_INPUT" "$TRAIN_INPUT_SHA256"
assert_sha256 "$TRAIN_COMMIT" "$TRAIN_COMMIT_SHA256"
assert_sha256 "$DEV_INPUT" "$DEV_INPUT_SHA256"
assert_sha256 "$DEV_COMMIT" "$DEV_COMMIT_SHA256"
[[ "$(git -C "$SOURCE_CHECKOUT" rev-parse HEAD)" == "$SOURCE_REVISION" ]] \
  || fail "source checkout revision mismatch"
[[ -z "$(git -C "$SOURCE_CHECKOUT" status --porcelain=v1 --untracked-files=all)" ]] \
  || fail "source checkout is dirty"
source "${SOURCE_CHECKOUT}/scripts/server/environment_fingerprint.sh"
[[ "$(pams_environment_fingerprint "${SOURCE_CHECKOUT}/docker/server")" == "$ENVIRONMENT_SHA256" ]] \
  || fail "source environment fingerprint mismatch"
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

mkdir -p -- "$PROTOCOL_ROOT"
[[ ! -L "$PROTOCOL_ROOT" ]] || fail "pose-protocol root must not be a symlink"
readonly PROTOCOL_ROOT_RESOLVED="$(realpath -e -- "$PROTOCOL_ROOT")"
[[ "$PROTOCOL_ROOT_RESOLVED" == "${ROOT}/runs/pose-protocol" ]] \
  || fail "pose-protocol root resolves outside the project"
mkdir -- "$RUN_ROOT" || fail "immutable run root exists: $RUN_ROOT"
RUN_RESERVED=1
mkdir -- \
  "$SOURCE_VIEW" \
  "$VIDEO_VIEW_ROOT" \
  "$CACHE_ROOT" \
  "$LEDGER_ROOT" \
  "$AUDIT_ROOT" \
  "$LOG_ROOT"
write_status "preparing" "source-export" "null"

git -C "$SOURCE_CHECKOUT" archive \
  --format=tar \
  "$SOURCE_REVISION" \
  src \
  "$CONFIG_RELATIVE" \
  | tar -xf - -C "$SOURCE_VIEW"
test ! -e "${SOURCE_VIEW}/.git" || fail "source export unexpectedly contains .git"
test ! -e "${SOURCE_VIEW}/results" || fail "source export unexpectedly contains results"
test ! -e "${SOURCE_VIEW}/data" || fail "source export unexpectedly contains data"
test ! -e "${SOURCE_VIEW}/tests" || fail "source export unexpectedly contains tests"
assert_sha256 "$EXPORTED_CONFIG" "$CONFIG_SHA256"
chmod -R a-w "$SOURCE_VIEW"

CURRENT_STAGE="video-views"
: > "$VIDEO_VIEW_RECEIPTS"
materialize_video_view "train337" "train" "$TRAIN_INPUT" "$TRAIN_VIDEO_VIEW" 337
materialize_video_view "dev84" "dev" "$DEV_INPUT" "$DEV_VIDEO_VIEW" 84
[[ "$(wc -l < "$VIDEO_VIEW_RECEIPTS")" -eq 421 ]] \
  || fail "video-view receipt is not exactly 421 rows"
chmod 0444 "$VIDEO_VIEW_RECEIPTS"

cat > "${RUN_ROOT}/attempt.reservation.json" <<EOF
{
  "schema_version": 1,
  "attempt_id": "${ATTEMPT_ID}",
  "reserved_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "dataset": "UCFRep",
  "scope": "train337-plus-dev84-pose-only",
  "classification": "v8-longest-contiguous-track-protocol-correction",
  "source_revision": "${SOURCE_REVISION}",
  "container_image_id": "${IMAGE_ID}",
  "container_environment_sha256": "${ENVIRONMENT_SHA256}",
  "config_sha256": "${CONFIG_SHA256}",
  "pose_fingerprint": "${POSE_FINGERPRINT}",
  "train_inputs_sha256": "${TRAIN_INPUT_SHA256}",
  "train_commitment_sha256": "${TRAIN_COMMIT_SHA256}",
  "dev_inputs_sha256": "${DEV_INPUT_SHA256}",
  "dev_commitment_sha256": "${DEV_COMMIT_SHA256}",
  "test_inputs_mounted": false,
  "targets_mounted": false,
  "split_video_views": true
}
EOF
chmod 0444 "${RUN_ROOT}/attempt.reservation.json"

CURRENT_STAGE="hardware-audit"
nvidia-smi -q > "${AUDIT_ROOT}/nvidia-smi-q.txt"
lscpu > "${AUDIT_ROOT}/lscpu.txt"
docker version > "${AUDIT_ROOT}/docker-version.txt"
git --version > "${AUDIT_ROOT}/git-version.txt"

run_split \
  "train337" \
  "$TRAIN_INPUT" \
  "$TRAIN_COMMIT" \
  "$TRAIN_VIDEO_VIEW" \
  337 \
  "$TRAIN_INPUT_SHA256" \
  "$TRAIN_COMMIT_SHA256"
run_split \
  "dev84" \
  "$DEV_INPUT" \
  "$DEV_COMMIT" \
  "$DEV_VIDEO_VIEW" \
  84 \
  "$DEV_INPUT_SHA256" \
  "$DEV_COMMIT_SHA256"

CURRENT_STAGE="final-audit"
[[ "$(find "$CACHE_ROOT" -maxdepth 1 -type f -name '*.npz' | wc -l)" -eq 421 ]] \
  || fail "v8 pose-cache pool is not exactly 421 files"
chmod -R a-w "$CACHE_ROOT" "$LEDGER_ROOT"
write_status "completed" "pose-dev84" "0"
find "$RUN_ROOT" \
  -path "$VIDEO_VIEW_ROOT" -prune \
  -o -type f \
  -not -path "${AUDIT_ROOT}/artifact-sha256.txt" \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "${AUDIT_ROOT}/artifact-sha256.txt"
chmod -R a-w "$RUN_ROOT"
printf '%s\n' "$RUN_ROOT"
