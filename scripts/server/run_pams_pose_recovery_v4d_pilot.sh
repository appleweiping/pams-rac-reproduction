#!/usr/bin/env bash
# Frozen train337-only v4d zero11/same39 pilot. Never mounts labels or targets.
set -Eeuo pipefail
IFS=$'\n\t'
umask 027

fail() { printf 'pose-recovery-v4d-pilot: %s\n' "$*" >&2; exit 2; }
sha256_file() { sha256sum -- "$1" | awk '{print $1}'; }
require_sha256() {
  local path="$1" expected="$2" role="$3"
  [[ -f "$path" ]] || fail "missing ${role}: ${path}"
  [[ "$(sha256_file "$path")" == "$expected" ]] || fail "${role} SHA mismatch"
}

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
: "${PAMS_V4D_EXPECTED_SOURCE_REVISION:?set exact 40-hex source revision}"
: "${PAMS_V4D_ATTEMPT_ID:?set immutable YYYYMMDDTHHMMSSZ attempt ID}"
: "${PAMS_V4D_COHORT:?set zero11 or same39}"
[[ "$PAMS_V4D_EXPECTED_SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]] || fail 'invalid source revision'
[[ "$PAMS_V4D_ATTEMPT_ID" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || fail 'invalid attempt ID'
[[ "$PAMS_V4D_COHORT" == zero11 || "$PAMS_V4D_COHORT" == same39 ]] || fail 'invalid cohort'
readonly SOURCE_REVISION="$(git -C "$REPOSITORY_ROOT" rev-parse HEAD^{commit})"
[[ "$SOURCE_REVISION" == "$PAMS_V4D_EXPECTED_SOURCE_REVISION" ]] || fail 'source revision mismatch'
[[ -z "$(git -C "$REPOSITORY_ROOT" status --porcelain=v1 --untracked-files=all)" ]] || fail 'dirty source worktree'

readonly IMAGE='pams-rac:5e18274a5353'
readonly IMAGE_ID='sha256:0a4d42c2d9911f147a17860e4e15095746b21c4e618e4fc1b20b62dd443c5898'
readonly CONFIG_SHA256='9933126d16735d42108203f512168e0eb438434f954fbcc969c5db622f5b6073'
readonly CONFIG_FINGERPRINT='587ad8a427e6d387a23b142e57cfac4a8bd49ffed93176b477000aeb2c2a5125'
readonly POSE_FINGERPRINT='4cc1f905cfb3484dccc1efc480e3fa59e4a106ffbe20528a85201a3fbd85b5d8'
readonly SIDECAR_SHA256='f95df0050df21f05bbc9b42d0154470714dde279e04cb8b00d0212bf49057d16'
readonly COMMITMENT_SHA256='85d41d2f59872e0900e9058481efbdf6bce91407d4c26bcde0a6547776454e53'
readonly V4A_LEDGER_SHA256='4cbba0d0f678cfdbd2c99752bbb55a8ceeb3aaa0b0b994bb6195b55cd0bc6018'
readonly V4A_PAIRED_SHA256='faf086277f4d0a0687a266b64edc6ae45a2ca0ab4b39cc8fc4da644350c571e5'
readonly SELECTION_SHA256='dada14719c5a48e9dd6ed3f68e53063330f845cef22a07eecd5ae2b5b21e89d4'
readonly MODEL_SHA256='fc266e953d2b302cdcbb9ae66f71f6b0d4649928bf02dc573961e361e4918926'
readonly MODEL_FILENAME='keypointrcnn_resnet50_fpn_coco-fc266e95.pth'

readonly OFFICIAL_ROOT='/media/lenovo/data2/pams-rac/runs/official-segment-v1/15cc1ec3c1d2-20260805T063653Z'
readonly V4A_ROOT='/media/lenovo/data2/pams-rac/runs/pose-recovery-v4a/83c877007392-20260807T113023Z'
readonly V4C_ROOT='/media/lenovo/data2/pams-rac/runs/pose-recovery-v4c-pilot/a75e08f79514-20260807T133622Z'
readonly VIDEOS="${OFFICIAL_ROOT}/video-views/train337"
readonly SIDECAR="${OFFICIAL_ROOT}/protocol/train.inputs.json"
readonly COMMITMENT="${OFFICIAL_ROOT}/protocol/train.inputs.commitment.json"
readonly V4A_CACHE="${V4A_ROOT}/pose-cache"
readonly V4A_LEDGER="${V4A_ROOT}/ledgers/train337.json"
readonly V4A_PAIRED="${V4A_ROOT}/audit/paired-gate.json"
readonly FROZEN_SELECTION="${V4C_ROOT}/audit/selection.json"
readonly MODEL_ASSET="/media/lenovo/data2/pams-rac/assets/torchvision/${MODEL_FILENAME}"
readonly RUN_PARENT="/media/lenovo/data2/pams-rac/runs/pose-recovery-v4d-${PAMS_V4D_COHORT}-pilot"
readonly RUN_ROOT="${RUN_PARENT}/${SOURCE_REVISION:0:12}-${PAMS_V4D_ATTEMPT_ID}"
readonly SOURCE_EXPORT="${RUN_ROOT}/source"
readonly CACHE_DIR="${RUN_ROOT}/pose-cache"
readonly LEDGER_DIR="${RUN_ROOT}/ledgers"
readonly AUDIT_DIR="${RUN_ROOT}/audit"
readonly LOG_DIR="${RUN_ROOT}/logs"
readonly LEDGER="${LEDGER_DIR}/${PAMS_V4D_COHORT}.json"
readonly SELECTION="${AUDIT_DIR}/selection.json"
readonly AUDIT="${AUDIT_DIR}/pilot-gate.json"

[[ "$(docker image inspect "$IMAGE" --format '{{.Id}}')" == "$IMAGE_ID" ]] || fail 'image ID mismatch'
[[ -d "$VIDEOS" && -d "$V4A_CACHE" ]] || fail 'missing train337 inputs'
require_sha256 "$SIDECAR" "$SIDECAR_SHA256" sidecar
require_sha256 "$COMMITMENT" "$COMMITMENT_SHA256" commitment
require_sha256 "$V4A_LEDGER" "$V4A_LEDGER_SHA256" v4a-ledger
require_sha256 "$V4A_PAIRED" "$V4A_PAIRED_SHA256" v4a-paired-gate
require_sha256 "$FROZEN_SELECTION" "$SELECTION_SHA256" frozen-selection
require_sha256 "$MODEL_ASSET" "$MODEL_SHA256" model-asset
[[ ! -e "$RUN_ROOT" ]] || fail "run root already exists: ${RUN_ROOT}"
mkdir -p -- "$RUN_PARENT"
mkdir -- "$RUN_ROOT" "$SOURCE_EXPORT" "$CACHE_DIR" "$LEDGER_DIR" "$AUDIT_DIR" "$LOG_DIR"
git -C "$REPOSITORY_ROOT" archive "$SOURCE_REVISION" | tar -x -C "$SOURCE_EXPORT"
chmod -R a-w -- "$SOURCE_EXPORT"
require_sha256 "$SOURCE_EXPORT/configs/experiments/pams_pose_recovery_v4d.yaml" "$CONFIG_SHA256" config

readonly IDENTITIES="$(docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges:true --user 1000:1000 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,uid=1000,gid=1000,mode=1777,size=64m \
  --env PYTHONPATH=/workspace/src --env PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=${SOURCE_EXPORT},dst=/workspace,readonly" --entrypoint python "$IMAGE" \
  -c 'from pams.config import load_config; c=load_config("/workspace/configs/experiments/pams_pose_recovery_v4d.yaml"); print(c.fingerprint,c.pose_fingerprint)')"
[[ "$IDENTITIES" == "$CONFIG_FINGERPRINT $POSE_FINGERPRINT" ]] || fail 'config identity mismatch'

readonly EXTRACT_NAME="pams-v4d-${PAMS_V4D_COHORT}-${SOURCE_REVISION:0:12}-${PAMS_V4D_ATTEMPT_ID,,}"
readonly AUDIT_NAME="${EXTRACT_NAME}-audit"
cleanup() { docker rm -f "$EXTRACT_NAME" "$AUDIT_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker create --name "$EXTRACT_NAME" --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges:true --pids-limit 4096 --memory 48g --cpus 12 \
  --user 1000:1000 --gpus device=0 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,uid=1000,gid=1000,mode=1777,size=2g \
  --tmpfs /pams/cache:rw,exec,nosuid,nodev,uid=1000,gid=1000,mode=1777,size=4g \
  --env PYTHONPATH=/workspace/src --env PYTHONDONTWRITEBYTECODE=1 \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env TORCH_HOME=/pams/torch-home \
  --env XDG_CACHE_HOME=/pams/cache --env TORCHINDUCTOR_CACHE_DIR=/pams/cache \
  --env TRITON_CACHE_DIR=/pams/cache \
  --mount "type=bind,src=${SOURCE_EXPORT},dst=/workspace,readonly" \
  --mount "type=bind,src=${VIDEOS},dst=/pams/videos,readonly" \
  --mount "type=bind,src=${SIDECAR},dst=/pams/inputs.json,readonly" \
  --mount "type=bind,src=${COMMITMENT},dst=/pams/commitment.json,readonly" \
  --mount "type=bind,src=${V4A_CACHE},dst=/pams/v4a-cache,readonly" \
  --mount "type=bind,src=${V4A_LEDGER},dst=/pams/v4a-ledger.json,readonly" \
  --mount "type=bind,src=${V4A_PAIRED},dst=/pams/v4a-paired.json,readonly" \
  --mount "type=bind,src=${FROZEN_SELECTION},dst=/pams/frozen-selection.json,readonly" \
  --mount "type=bind,src=${MODEL_ASSET},dst=/pams/torch-home/hub/checkpoints/${MODEL_FILENAME},readonly" \
  --mount "type=bind,src=${CACHE_DIR},dst=/pams/output-cache" \
  --mount "type=bind,src=${LEDGER_DIR},dst=/pams/ledgers" \
  --mount "type=bind,src=${AUDIT_DIR},dst=/pams/audit" --workdir /workspace "$IMAGE" \
  python scripts/server/run_pose_recovery_v4d_pilot.py --cohort "$PAMS_V4D_COHORT" \
  --config /workspace/configs/experiments/pams_pose_recovery_v4d.yaml \
  --train-input /pams/inputs.json --train-commitment /pams/commitment.json \
  --video-root /pams/videos --model-asset "/pams/torch-home/hub/checkpoints/${MODEL_FILENAME}" \
  --v4a-cache-dir /pams/v4a-cache --v4a-ledger /pams/v4a-ledger.json \
  --v4a-paired-gate /pams/v4a-paired.json --frozen-selection /pams/frozen-selection.json \
  --cache-dir /pams/output-cache --ledger "/pams/ledgers/${PAMS_V4D_COHORT}.json" \
  --selection /pams/audit/selection.json >"${AUDIT_DIR}/extract.create-id.txt"
docker inspect "$EXTRACT_NAME" >"${AUDIT_DIR}/extract.inspect.pre.json"
python3 - "$AUDIT_DIR/extract.inspect.pre.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))[0]; h=x['HostConfig']; c=x['Config']
def require(condition,message):
    if not condition: raise SystemExit(message)
require(h['ReadonlyRootfs'] and h['NetworkMode']=='none' and 'ALL' in h['CapDrop'],'sandbox mismatch')
require(c['User']=='1000:1000' and any(v=='CUBLAS_WORKSPACE_CONFIG=:4096:8' for v in c['Env']),'user/env mismatch')
t=h['Tmpfs']['/pams/cache']; require(all(v in t.split(',') for v in ('rw','exec','nosuid','nodev','uid=1000','gid=1000','mode=1777')),'cache tmpfs mismatch')
require(h['DeviceRequests'][0]['DeviceIDs']==['0'],'GPU device lock mismatch')
mounts={m['Destination']:m['RW'] for m in x['Mounts']}
for p in ('/workspace','/pams/videos','/pams/inputs.json','/pams/commitment.json','/pams/v4a-cache','/pams/v4a-ledger.json','/pams/v4a-paired.json','/pams/frozen-selection.json','/pams/torch-home/hub/checkpoints/keypointrcnn_resnet50_fpn_coco-fc266e95.pth'):
    require(mounts.get(p) is False,f'non-read-only input mount: {p}')
PY
docker start -a "$EXTRACT_NAME" 2>&1 | tee "$LOG_DIR/extract.log"
docker inspect "$EXTRACT_NAME" >"${AUDIT_DIR}/extract.inspect.post.json"
docker rm "$EXTRACT_NAME" >/dev/null

docker create --name "$AUDIT_NAME" --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges:true --user 1000:1000 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,uid=1000,gid=1000,mode=1777,size=1g \
  --env PYTHONPATH=/workspace/src --env PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=${SOURCE_EXPORT},dst=/workspace,readonly" \
  --mount "type=bind,src=${SIDECAR},dst=/pams/inputs.json,readonly" \
  --mount "type=bind,src=${COMMITMENT},dst=/pams/commitment.json,readonly" \
  --mount "type=bind,src=${V4A_CACHE},dst=/pams/v4a-cache,readonly" \
  --mount "type=bind,src=${V4A_LEDGER},dst=/pams/v4a-ledger.json,readonly" \
  --mount "type=bind,src=${V4A_PAIRED},dst=/pams/v4a-paired.json,readonly" \
  --mount "type=bind,src=${CACHE_DIR},dst=/pams/candidate-cache,readonly" \
  --mount "type=bind,src=${LEDGER},dst=/pams/candidate-ledger.json,readonly" \
  --mount "type=bind,src=${SELECTION},dst=/pams/selection.json,readonly" \
  --mount "type=bind,src=${AUDIT_DIR},dst=/pams/audit" --workdir /workspace "$IMAGE" \
  python scripts/server/audit_pose_recovery_v4d_pilot.py --cohort "$PAMS_V4D_COHORT" \
  --gate /workspace/configs/gates/pams_pose_recovery_v4d_pilot.yaml \
  --train-input /pams/inputs.json --train-commitment /pams/commitment.json \
  --selection /pams/selection.json --v4a-cache-dir /pams/v4a-cache \
  --v4a-ledger /pams/v4a-ledger.json --v4a-paired-gate /pams/v4a-paired.json \
  --candidate-cache-dir /pams/candidate-cache --candidate-ledger /pams/candidate-ledger.json \
  --output /pams/audit/pilot-gate.json >"${AUDIT_DIR}/audit.create-id.txt"
docker inspect "$AUDIT_NAME" >"${AUDIT_DIR}/audit.inspect.pre.json"
docker start -a "$AUDIT_NAME" 2>&1 | tee "$LOG_DIR/audit.log"
docker inspect "$AUDIT_NAME" >"${AUDIT_DIR}/audit.inspect.post.json"
docker rm "$AUDIT_NAME" >/dev/null
printf 'pose-recovery-v4d %s pilot passed: %s\n' "$PAMS_V4D_COHORT" "$RUN_ROOT"
