#!/usr/bin/env bash
# Exact full337 v4d extraction and label-free training-quality gate.
set -Eeuo pipefail
IFS=$'\n\t'
umask 027

fail() { printf 'pose-recovery-v4d-full337: %s\n' "$*" >&2; exit 2; }
sha256_file() { sha256sum -- "$1" | awk '{print $1}'; }
require_sha256() {
  local path="$1" expected="$2" role="$3"
  [[ -f "$path" ]] || fail "missing ${role}: ${path}"
  [[ "$(sha256_file "$path")" == "$expected" ]] || fail "${role} SHA mismatch"
}

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
: "${PAMS_V4D_EXPECTED_SOURCE_REVISION:?set exact 40-hex source revision}"
: "${PAMS_V4D_FULL337_ATTEMPT_ID:?set immutable YYYYMMDDTHHMMSSZ attempt ID}"
: "${PAMS_V4D_EXTRACTION_AUTH_ROOT:?set sealed v2 extraction-authorization root}"
: "${PAMS_V4D_EXPECTED_EXTRACTION_AUTH_SHA256:?set exact v2 authorization receipt SHA}"
[[ "$PAMS_V4D_EXPECTED_SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]] || fail 'invalid source revision'
[[ "$PAMS_V4D_FULL337_ATTEMPT_ID" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || fail 'invalid attempt ID'
[[ "$PAMS_V4D_EXPECTED_EXTRACTION_AUTH_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail 'invalid authorization SHA'
readonly SOURCE_REVISION="$(git -C "$REPOSITORY_ROOT" rev-parse HEAD^{commit})"
[[ "$SOURCE_REVISION" == "$PAMS_V4D_EXPECTED_SOURCE_REVISION" ]] || fail 'source revision mismatch'
[[ -z "$(git -C "$REPOSITORY_ROOT" status --porcelain=v1 --untracked-files=all)" ]] || fail 'dirty source worktree'

readonly IMAGE='pams-rac:5e18274a5353'
readonly IMAGE_ID='sha256:0a4d42c2d9911f147a17860e4e15095746b21c4e618e4fc1b20b62dd443c5898'
readonly GATE_SHA256='c2b322605e404daa51ccb0a1dba2df3c56d5711ab958ee4c4ec44b9265539146'
readonly CONFIG_SHA256='9933126d16735d42108203f512168e0eb438434f954fbcc969c5db622f5b6073'
readonly CONFIG_FINGERPRINT='587ad8a427e6d387a23b142e57cfac4a8bd49ffed93176b477000aeb2c2a5125'
readonly POSE_FINGERPRINT='4cc1f905cfb3484dccc1efc480e3fa59e4a106ffbe20528a85201a3fbd85b5d8'
readonly SIDECAR_SHA256='f95df0050df21f05bbc9b42d0154470714dde279e04cb8b00d0212bf49057d16'
readonly COMMITMENT_SHA256='85d41d2f59872e0900e9058481efbdf6bce91407d4c26bcde0a6547776454e53'
readonly V4A_LEDGER_SHA256='4cbba0d0f678cfdbd2c99752bbb55a8ceeb3aaa0b0b994bb6195b55cd0bc6018'
readonly V4A_PAIRED_SHA256='faf086277f4d0a0687a266b64edc6ae45a2ca0ab4b39cc8fc4da644350c571e5'
readonly SAME39_FAILURE_SHA256='91c702663e6be37bec59cbf558ada047435c0e8f4c4c8efd951f88d505039a00'
readonly SAME39_AUDIT_SHA256='d25a9fb6da457677cc2c33e79725cedfc36ed9b5555e762878569d3f43e754f5'
readonly SAME39_LEDGER_SHA256='2c89a66c3203e73db0a6597a7e6cb0e0912c81866795499d444ea3d66c2db7fc'
readonly SAME39_SELECTION_SHA256='c3a7d1113c4be28e0a9c06e1d59eeea31383a266e3400d7559fd0fab605123d3'
readonly MODEL_SHA256='fc266e953d2b302cdcbb9ae66f71f6b0d4649928bf02dc573961e361e4918926'
readonly MODEL_FILENAME='keypointrcnn_resnet50_fpn_coco-fc266e95.pth'

readonly OFFICIAL_ROOT='/media/lenovo/data2/pams-rac/runs/official-segment-v1/15cc1ec3c1d2-20260805T063653Z'
readonly V4A_ROOT='/media/lenovo/data2/pams-rac/runs/pose-recovery-v4a/83c877007392-20260807T113023Z'
readonly SAME39_ROOT='/media/lenovo/data2/pams-rac/runs/pose-recovery-v4d-same39-pilot/081c3c138395-20260807T150913Z'
readonly VIDEOS="${OFFICIAL_ROOT}/video-views/train337"
readonly SIDECAR="${OFFICIAL_ROOT}/protocol/train.inputs.json"
readonly COMMITMENT="${OFFICIAL_ROOT}/protocol/train.inputs.commitment.json"
readonly V4A_CACHE="${V4A_ROOT}/pose-cache"
readonly V4A_LEDGER="${V4A_ROOT}/ledgers/train337.json"
readonly V4A_PAIRED="${V4A_ROOT}/audit/paired-gate.json"
readonly SAME39_FAILURE="${SAME39_ROOT}/audit/failure.receipt.json"
readonly SAME39_AUDIT="${SAME39_ROOT}/gate-output/pilot-gate.json"
readonly SAME39_LEDGER="${SAME39_ROOT}/ledgers/same39.json"
readonly SAME39_SELECTION="${SAME39_ROOT}/audit/selection.json"
readonly SAME39_CACHE="${SAME39_ROOT}/pose-cache"
readonly EXTRACTION_AUTH="${PAMS_V4D_EXTRACTION_AUTH_ROOT}/authorization/extraction.authorization.json"
readonly MODEL_ASSET="/media/lenovo/data2/pams-rac/assets/torchvision/${MODEL_FILENAME}"
readonly RUN_PARENT='/media/lenovo/data2/pams-rac/runs/pose-recovery-v4d-full337'
readonly RUN_ROOT="${RUN_PARENT}/${SOURCE_REVISION:0:12}-${PAMS_V4D_FULL337_ATTEMPT_ID}"
readonly SOURCE_EXPORT="${RUN_ROOT}/source"
readonly CACHE_DIR="${RUN_ROOT}/pose-cache"
readonly LEDGER_DIR="${RUN_ROOT}/ledgers"
readonly AUDIT_DIR="${RUN_ROOT}/audit"
readonly GATE_OUTPUT_DIR="${RUN_ROOT}/gate-output"
readonly LOG_DIR="${RUN_ROOT}/logs"
readonly LEDGER="${LEDGER_DIR}/train337.json"
readonly GATE_AUDIT="${GATE_OUTPUT_DIR}/full337-gate.json"
readonly TRAIN_AUTH="${GATE_OUTPUT_DIR}/training.authorization.json"
readonly TRAIN_DENIAL="${GATE_OUTPUT_DIR}/training.denial.json"

[[ "$(docker image inspect "$IMAGE" --format '{{.Id}}')" == "$IMAGE_ID" ]] || fail 'image ID mismatch'
[[ -d "$VIDEOS" && -d "$V4A_CACHE" && -d "$SAME39_CACHE" ]] || fail 'missing input directory'
require_sha256 "$SIDECAR" "$SIDECAR_SHA256" sidecar
require_sha256 "$COMMITMENT" "$COMMITMENT_SHA256" commitment
require_sha256 "$V4A_LEDGER" "$V4A_LEDGER_SHA256" v4a-ledger
require_sha256 "$V4A_PAIRED" "$V4A_PAIRED_SHA256" v4a-paired-gate
require_sha256 "$SAME39_FAILURE" "$SAME39_FAILURE_SHA256" same39-failure
require_sha256 "$SAME39_AUDIT" "$SAME39_AUDIT_SHA256" same39-audit
require_sha256 "$SAME39_LEDGER" "$SAME39_LEDGER_SHA256" same39-ledger
require_sha256 "$SAME39_SELECTION" "$SAME39_SELECTION_SHA256" same39-selection
require_sha256 "$EXTRACTION_AUTH" "$PAMS_V4D_EXPECTED_EXTRACTION_AUTH_SHA256" extraction-authorization
require_sha256 "$MODEL_ASSET" "$MODEL_SHA256" model-asset
[[ -z "$(find "$SAME39_ROOT" -xdev -perm /022 -print -quit)" ]] || fail 'same39 root is not sealed'
[[ -z "$(find "$PAMS_V4D_EXTRACTION_AUTH_ROOT" -xdev -perm /022 -print -quit)" ]] || fail 'authorization root is not sealed'
python3 - "$EXTRACTION_AUTH" "$SOURCE_REVISION" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['source_revision']==sys.argv[2]
assert x['authorization_scope']=='full337_pose_extraction_only'
assert x['full337_pose_extraction_authorized'] is True
assert x['baseline_training_authorized'] is False
assert x['training_runner_must_reject'] is True
PY
[[ ! -e "$RUN_ROOT" ]] || fail "run root already exists: ${RUN_ROOT}"
mkdir -p -- "$RUN_PARENT"
mkdir -- "$RUN_ROOT" "$SOURCE_EXPORT" "$CACHE_DIR" "$LEDGER_DIR" "$AUDIT_DIR" "$GATE_OUTPUT_DIR" "$LOG_DIR"
git -C "$REPOSITORY_ROOT" archive "$SOURCE_REVISION" | tar -x -C "$SOURCE_EXPORT"
chmod -R a-w -- "$SOURCE_EXPORT"
require_sha256 "$SOURCE_EXPORT/configs/gates/pams_pose_recovery_v4d_full337.yaml" "$GATE_SHA256" full337-gate
require_sha256 "$SOURCE_EXPORT/configs/experiments/pams_pose_recovery_v4d.yaml" "$CONFIG_SHA256" v4d-config

readonly EXTRACT_NAME="pams-v4d-full337-${SOURCE_REVISION:0:12}-${PAMS_V4D_FULL337_ATTEMPT_ID,,}"
readonly GATE_NAME="${EXTRACT_NAME}-gate"
readonly GPU_LOCK='/media/lenovo/data2/pams-rac/.pams-gpu-locks/gpu1.lock'
mkdir -p -- "$(dirname -- "$GPU_LOCK")"
exec 9>"$GPU_LOCK"
flock -n 9 || fail 'physical GPU1 lock is already held'
readonly GPU_PREFLIGHT="$(nvidia-smi -i 1 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)"
IFS=',' read -r GPU_UUID GPU_USED_MIB GPU_UTIL_PERCENT <<<"$GPU_PREFLIGHT"
GPU_UUID="${GPU_UUID//[[:space:]]/}"
GPU_USED_MIB="${GPU_USED_MIB//[[:space:]]/}"
GPU_UTIL_PERCENT="${GPU_UTIL_PERCENT//[[:space:]]/}"
[[ "$GPU_USED_MIB" =~ ^[0-9]+$ && "$GPU_USED_MIB" -le 256 ]] || fail 'GPU1 memory preflight failed'
[[ "$GPU_UTIL_PERCENT" == 0 ]] || fail 'GPU1 utilization preflight failed'

FAILURE_PHASE='container-create'
cleanup() { docker rm -f "$EXTRACT_NAME" "$GATE_NAME" >/dev/null 2>&1 || true; }
finalize() {
  local status="$?"
  trap - EXIT
  cleanup
  if [[ "$status" -ne 0 && -d "$RUN_ROOT" && ! -e "$AUDIT_DIR/failure.receipt.json" && ! -e "$TRAIN_DENIAL" ]]; then
    python3 - "$RUN_ROOT" "$SOURCE_REVISION" "$IMAGE_ID" "$FAILURE_PHASE" "$status" \
      "$GPU_UUID" "$GPU_USED_MIB" "$GPU_UTIL_PERCENT" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); artifacts={}
for relative in ('audit/extract.inspect.pre.json','audit/extract.inspect.post.json','audit/gate.inspect.pre.json','audit/gate.inspect.post.json','ledgers/train337.json','gate-output/full337-gate.json'):
 path=root/relative
 if path.is_file(): artifacts[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
oom=False
inspect=root/'audit/extract.inspect.post.json'
if inspect.is_file(): oom=bool(json.loads(inspect.read_text())[0]['State'].get('OOMKilled'))
payload={'schema_version':1,'artifact_type':'pams_pose_recovery_v4d_full337_failure_receipt',
 'source_revision':sys.argv[2],'container_image_id':sys.argv[3],'failed_phase':sys.argv[4],
 'exit_status':int(sys.argv[5]),'oom_killed':oom,'baseline_training_authorized':False,
 'physical_gpu_index':1,'gpu_uuid':sys.argv[6],'gpu_memory_used_mib_preflight':int(sys.argv[7]),
 'gpu_utilization_percent_preflight':int(sys.argv[8]),'artifacts':artifacts}
with (root/'audit/failure.receipt.json').open('x',encoding='utf-8',newline='\n') as f:
 json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
PY
  fi
  [[ ! -d "$RUN_ROOT" ]] || chmod -R a-w -- "$RUN_ROOT"
  exit "$status"
}
trap finalize EXIT

docker create --name "$EXTRACT_NAME" --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges:true --pids-limit 4096 --memory 48g --cpus 12 \
  --user 1000:1000 --gpus device=1 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,uid=1000,gid=1000,mode=1777,size=2g \
  --tmpfs /pams/cache:rw,exec,nosuid,nodev,uid=1000,gid=1000,mode=1777,size=4g \
  --env PYTHONPATH=/workspace/src:/workspace/scripts/server --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONOPTIMIZE= --env "PAMS_CONTAINER_SOURCE_REVISION=${SOURCE_REVISION}" \
  --env "PAMS_CONTAINER_IMAGE_ID=${IMAGE_ID}" --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  --env TORCH_HOME=/pams/torch-home --env XDG_CACHE_HOME=/pams/cache \
  --env TORCHINDUCTOR_CACHE_DIR=/pams/cache --env TRITON_CACHE_DIR=/pams/cache \
  --mount "type=bind,src=${SOURCE_EXPORT},dst=/workspace,readonly" \
  --mount "type=bind,src=${VIDEOS},dst=/pams/videos,readonly" \
  --mount "type=bind,src=${SIDECAR},dst=/pams/inputs.json,readonly" \
  --mount "type=bind,src=${COMMITMENT},dst=/pams/commitment.json,readonly" \
  --mount "type=bind,src=${V4A_CACHE},dst=/pams/v4a-cache,readonly" \
  --mount "type=bind,src=${V4A_LEDGER},dst=/pams/v4a-ledger.json,readonly" \
  --mount "type=bind,src=${V4A_PAIRED},dst=/pams/v4a-paired.json,readonly" \
  --mount "type=bind,src=${SAME39_FAILURE},dst=/pams/same39/failure.json,readonly" \
  --mount "type=bind,src=${SAME39_AUDIT},dst=/pams/same39/audit.json,readonly" \
  --mount "type=bind,src=${SAME39_LEDGER},dst=/pams/same39/ledger.json,readonly" \
  --mount "type=bind,src=${SAME39_SELECTION},dst=/pams/same39/selection.json,readonly" \
  --mount "type=bind,src=${SAME39_CACHE},dst=/pams/same39/cache,readonly" \
  --mount "type=bind,src=${EXTRACTION_AUTH},dst=/pams/extraction.authorization.json,readonly" \
  --mount "type=bind,src=${MODEL_ASSET},dst=/pams/torch-home/hub/checkpoints/${MODEL_FILENAME},readonly" \
  --mount "type=bind,src=${CACHE_DIR},dst=/pams/output-cache" \
  --mount "type=bind,src=${LEDGER_DIR},dst=/pams/ledgers" --workdir /workspace "$IMAGE" \
  python scripts/server/run_pose_recovery_v4d_full337.py \
  --source-revision "$SOURCE_REVISION" --container-image-id "$IMAGE_ID" \
  --gate-sha256 "$GATE_SHA256" --extraction-authorization-sha256 "$PAMS_V4D_EXPECTED_EXTRACTION_AUTH_SHA256" \
  --same39-failure-receipt-sha256 "$SAME39_FAILURE_SHA256" \
  --gate /workspace/configs/gates/pams_pose_recovery_v4d_full337.yaml \
  --config /workspace/configs/experiments/pams_pose_recovery_v4d.yaml \
  --train-input /pams/inputs.json --train-commitment /pams/commitment.json \
  --video-root /pams/videos --model-asset "/pams/torch-home/hub/checkpoints/${MODEL_FILENAME}" \
  --v4a-cache-dir /pams/v4a-cache --v4a-ledger /pams/v4a-ledger.json \
  --v4a-paired-gate /pams/v4a-paired.json \
  --extraction-authorization /pams/extraction.authorization.json \
  --same39-failure-receipt /pams/same39/failure.json --same39-audit /pams/same39/audit.json \
  --same39-ledger /pams/same39/ledger.json --same39-selection /pams/same39/selection.json \
  --same39-cache-dir /pams/same39/cache --output-cache-dir /pams/output-cache \
  --output-ledger /pams/ledgers/train337.json >"${AUDIT_DIR}/extract.create-id.txt"
docker inspect "$EXTRACT_NAME" >"${AUDIT_DIR}/extract.inspect.pre.json"
python3 - "$AUDIT_DIR/extract.inspect.pre.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))[0]; h=x['HostConfig']; mounts={m['Destination']:m['RW'] for m in x['Mounts']}
def req(v,m):
 if not v: raise SystemExit(m)
req(h['ReadonlyRootfs'] and h['NetworkMode']=='none' and 'ALL' in h['CapDrop'],'sandbox mismatch')
req(h['DeviceRequests'][0]['DeviceIDs']==['1'],'GPU1 mapping mismatch')
for p in ('/workspace','/pams/videos','/pams/inputs.json','/pams/commitment.json','/pams/v4a-cache','/pams/same39/cache','/pams/extraction.authorization.json'):
 req(mounts.get(p) is False,f'non-read-only input: {p}')
for forbidden in ('/pams/dev','/pams/test','/pams/targets'):
 req(forbidden not in mounts,f'forbidden mount: {forbidden}')
PY
FAILURE_PHASE='extract'
set +e
docker start -a "$EXTRACT_NAME" 2>&1 | tee "$LOG_DIR/extract.log"
EXTRACT_STATUS="${PIPESTATUS[0]}"
set -e
docker inspect "$EXTRACT_NAME" >"${AUDIT_DIR}/extract.inspect.post.json"
docker rm "$EXTRACT_NAME" >/dev/null
[[ "$EXTRACT_STATUS" -eq 0 ]] || fail "extract container failed with status ${EXTRACT_STATUS}"

docker create --name "$GATE_NAME" --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges:true --user 1000:1000 --memory 16g --cpus 8 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,uid=1000,gid=1000,mode=1777,size=2g \
  --env PYTHONPATH=/workspace/src:/workspace/scripts/server --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONOPTIMIZE= --env "PAMS_CONTAINER_SOURCE_REVISION=${SOURCE_REVISION}" \
  --env "PAMS_CONTAINER_IMAGE_ID=${IMAGE_ID}" \
  --mount "type=bind,src=${SOURCE_EXPORT},dst=/workspace,readonly" \
  --mount "type=bind,src=${SIDECAR},dst=/pams/inputs.json,readonly" \
  --mount "type=bind,src=${COMMITMENT},dst=/pams/commitment.json,readonly" \
  --mount "type=bind,src=${V4A_CACHE},dst=/pams/v4a-cache,readonly" \
  --mount "type=bind,src=${V4A_LEDGER},dst=/pams/v4a-ledger.json,readonly" \
  --mount "type=bind,src=${V4A_PAIRED},dst=/pams/v4a-paired.json,readonly" \
  --mount "type=bind,src=${SAME39_FAILURE},dst=/pams/same39/failure.json,readonly" \
  --mount "type=bind,src=${SAME39_AUDIT},dst=/pams/same39/audit.json,readonly" \
  --mount "type=bind,src=${SAME39_LEDGER},dst=/pams/same39/ledger.json,readonly" \
  --mount "type=bind,src=${SAME39_SELECTION},dst=/pams/same39/selection.json,readonly" \
  --mount "type=bind,src=${SAME39_CACHE},dst=/pams/same39/cache,readonly" \
  --mount "type=bind,src=${EXTRACTION_AUTH},dst=/pams/extraction.authorization.json,readonly" \
  --mount "type=bind,src=${CACHE_DIR},dst=/pams/candidate-cache,readonly" \
  --mount "type=bind,src=${LEDGER},dst=/pams/candidate-ledger.json,readonly" \
  --mount "type=bind,src=${GATE_OUTPUT_DIR},dst=/pams/output" --workdir /workspace "$IMAGE" \
  python scripts/server/audit_pose_recovery_v4d_full337.py \
  --source-revision "$SOURCE_REVISION" --container-image-id "$IMAGE_ID" \
  --gate-sha256 "$GATE_SHA256" --extraction-authorization-sha256 "$PAMS_V4D_EXPECTED_EXTRACTION_AUTH_SHA256" \
  --same39-failure-receipt-sha256 "$SAME39_FAILURE_SHA256" \
  --gate /workspace/configs/gates/pams_pose_recovery_v4d_full337.yaml \
  --train-input /pams/inputs.json --train-commitment /pams/commitment.json \
  --v4a-cache-dir /pams/v4a-cache --v4a-ledger /pams/v4a-ledger.json \
  --v4a-paired-gate /pams/v4a-paired.json \
  --extraction-authorization /pams/extraction.authorization.json \
  --same39-failure-receipt /pams/same39/failure.json --same39-audit /pams/same39/audit.json \
  --same39-ledger /pams/same39/ledger.json --same39-selection /pams/same39/selection.json \
  --same39-cache-dir /pams/same39/cache --candidate-cache-dir /pams/candidate-cache \
  --candidate-ledger /pams/candidate-ledger.json --audit-output /pams/output/full337-gate.json \
  --authorization-output /pams/output/training.authorization.json \
  --denial-output /pams/output/training.denial.json >"${AUDIT_DIR}/gate.create-id.txt"
docker inspect "$GATE_NAME" >"${AUDIT_DIR}/gate.inspect.pre.json"
python3 - "$AUDIT_DIR/gate.inspect.pre.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))[0]; h=x['HostConfig']; mounts={m['Destination']:m['RW'] for m in x['Mounts']}
def req(v,m):
 if not v: raise SystemExit(m)
req(h['ReadonlyRootfs'] and h['NetworkMode']=='none' and 'ALL' in h['CapDrop'],'gate sandbox mismatch')
req(not h.get('DeviceRequests'),'label-free gate must not have GPU')
req(mounts.get('/pams/output') is True,'gate output not writable')
for destination,writable in mounts.items():
 if destination!='/pams/output': req(writable is False,f'writable gate input: {destination}')
for forbidden in ('/pams/videos','/pams/model','/pams/dev','/pams/test','/pams/targets'):
 req(forbidden not in mounts,f'forbidden gate mount: {forbidden}')
PY
FAILURE_PHASE='full337-label-free-gate'
set +e
docker start -a "$GATE_NAME" 2>&1 | tee "$LOG_DIR/gate.log"
GATE_STATUS="${PIPESTATUS[0]}"
set -e
docker inspect "$GATE_NAME" >"${AUDIT_DIR}/gate.inspect.post.json"
docker rm "$GATE_NAME" >/dev/null
[[ "$GATE_STATUS" -eq 0 || "$GATE_STATUS" -eq 1 ]] || fail "gate container failed with status ${GATE_STATUS}"
[[ -f "$GATE_AUDIT" ]] || fail 'full337 gate audit missing'
if [[ "$GATE_STATUS" -eq 0 ]]; then
  [[ -f "$TRAIN_AUTH" && ! -e "$TRAIN_DENIAL" ]] || fail 'PASS authorization artifact mismatch'
else
  [[ -f "$TRAIN_DENIAL" && ! -e "$TRAIN_AUTH" ]] || fail 'FAIL denial artifact mismatch'
fi
python3 - "$RUN_ROOT" "$SOURCE_REVISION" "$IMAGE_ID" "$GATE_STATUS" \
  "$GPU_UUID" "$GPU_USED_MIB" "$GPU_UTIL_PERCENT" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); status=int(sys.argv[4]); passed=status==0
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
decision=root/'gate-output'/('training.authorization.json' if passed else 'training.denial.json')
x=json.loads(decision.read_text())
assert x['baseline_training_authorized'] is passed
payload={'schema_version':1,'artifact_type':'pams_pose_recovery_v4d_full337_run_receipt',
 'source_revision':sys.argv[2],'container_image_id':sys.argv[3],'gate_passed':passed,
 'baseline_training_authorized':passed,'training_decision_artifact':decision.name,
 'physical_gpu_index':1,'gpu_uuid':sys.argv[5],'gpu_memory_used_mib_preflight':int(sys.argv[6]),
 'gpu_utilization_percent_preflight':int(sys.argv[7]),'ledger_sha256':digest(root/'ledgers/train337.json'),
 'gate_audit_sha256':digest(root/'gate-output/full337-gate.json'),
 'training_decision_sha256':digest(decision),'extraction_only_authorization_was_not_training_authority':True}
with (root/'audit/run.receipt.json').open('x',encoding='utf-8',newline='\n') as f:
 json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
PY
chmod -R a-w -- "$RUN_ROOT"
trap - EXIT
if [[ "$GATE_STATUS" -eq 0 ]]; then
  printf 'v4d full337 gate passed; baseline training authorized: %s\n' "$RUN_ROOT"
  exit 0
fi
printf 'v4d full337 gate denied baseline training; sealed denial: %s\n' "$RUN_ROOT" >&2
exit 1
