#!/usr/bin/env bash
# Label-free v2 compute authorization. It can never authorize baseline training.
set -Eeuo pipefail
IFS=$'\n\t'
umask 027

fail() { printf 'pose-recovery-v4d-full337-auth-v2: %s\n' "$*" >&2; exit 2; }
sha256_file() { sha256sum -- "$1" | awk '{print $1}'; }
require_sha256() {
  local path="$1" expected="$2" role="$3"
  [[ -f "$path" ]] || fail "missing ${role}: ${path}"
  [[ "$(sha256_file "$path")" == "$expected" ]] || fail "${role} SHA mismatch"
}

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
: "${PAMS_V4D_EXPECTED_SOURCE_REVISION:?set exact 40-hex source revision}"
: "${PAMS_V4D_AUTH_ATTEMPT_ID:?set immutable YYYYMMDDTHHMMSSZ attempt ID}"
[[ "$PAMS_V4D_EXPECTED_SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]] || fail 'invalid source revision'
[[ "$PAMS_V4D_AUTH_ATTEMPT_ID" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || fail 'invalid attempt ID'
readonly SOURCE_REVISION="$(git -C "$REPOSITORY_ROOT" rev-parse HEAD^{commit})"
[[ "$SOURCE_REVISION" == "$PAMS_V4D_EXPECTED_SOURCE_REVISION" ]] || fail 'source revision mismatch'
[[ -z "$(git -C "$REPOSITORY_ROOT" status --porcelain=v1 --untracked-files=all)" ]] || fail 'dirty source worktree'

readonly IMAGE='pams-rac:5e18274a5353'
readonly IMAGE_ID='sha256:0a4d42c2d9911f147a17860e4e15095746b21c4e618e4fc1b20b62dd443c5898'
readonly GATE_SHA256='304af8669703a73bf3d953e19848633986b682d478ee3a5f8be78114c26f11b8'
readonly SIDECAR_SHA256='f95df0050df21f05bbc9b42d0154470714dde279e04cb8b00d0212bf49057d16'
readonly COMMITMENT_SHA256='85d41d2f59872e0900e9058481efbdf6bce91407d4c26bcde0a6547776454e53'
readonly V4A_LEDGER_SHA256='4cbba0d0f678cfdbd2c99752bbb55a8ceeb3aaa0b0b994bb6195b55cd0bc6018'
readonly V4A_PAIRED_SHA256='faf086277f4d0a0687a266b64edc6ae45a2ca0ab4b39cc8fc4da644350c571e5'
readonly SAME39_FAILURE_SHA256='91c702663e6be37bec59cbf558ada047435c0e8f4c4c8efd951f88d505039a00'
readonly SAME39_AUDIT_SHA256='d25a9fb6da457677cc2c33e79725cedfc36ed9b5555e762878569d3f43e754f5'
readonly SAME39_LEDGER_SHA256='2c89a66c3203e73db0a6597a7e6cb0e0912c81866795499d444ea3d66c2db7fc'
readonly SAME39_SELECTION_SHA256='c3a7d1113c4be28e0a9c06e1d59eeea31383a266e3400d7559fd0fab605123d3'

readonly OFFICIAL_ROOT='/media/lenovo/data2/pams-rac/runs/official-segment-v1/15cc1ec3c1d2-20260805T063653Z'
readonly V4A_ROOT='/media/lenovo/data2/pams-rac/runs/pose-recovery-v4a/83c877007392-20260807T113023Z'
readonly SAME39_ROOT='/media/lenovo/data2/pams-rac/runs/pose-recovery-v4d-same39-pilot/081c3c138395-20260807T150913Z'
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
readonly RUN_PARENT='/media/lenovo/data2/pams-rac/runs/pose-recovery-v4d-full337-extraction-authorization-v2'
readonly RUN_ROOT="${RUN_PARENT}/${SOURCE_REVISION:0:12}-${PAMS_V4D_AUTH_ATTEMPT_ID}"
readonly SOURCE_EXPORT="${RUN_ROOT}/source"
readonly OUTPUT_DIR="${RUN_ROOT}/authorization"
readonly AUDIT_DIR="${RUN_ROOT}/audit"
readonly LOG_DIR="${RUN_ROOT}/logs"
readonly AUTHORIZATION="${OUTPUT_DIR}/extraction.authorization.json"

[[ "$(docker image inspect "$IMAGE" --format '{{.Id}}')" == "$IMAGE_ID" ]] || fail 'image ID mismatch'
require_sha256 "$SIDECAR" "$SIDECAR_SHA256" sidecar
require_sha256 "$COMMITMENT" "$COMMITMENT_SHA256" commitment
require_sha256 "$V4A_LEDGER" "$V4A_LEDGER_SHA256" v4a-ledger
require_sha256 "$V4A_PAIRED" "$V4A_PAIRED_SHA256" v4a-paired-gate
require_sha256 "$SAME39_FAILURE" "$SAME39_FAILURE_SHA256" same39-failure-receipt
require_sha256 "$SAME39_AUDIT" "$SAME39_AUDIT_SHA256" same39-audit
require_sha256 "$SAME39_LEDGER" "$SAME39_LEDGER_SHA256" same39-ledger
require_sha256 "$SAME39_SELECTION" "$SAME39_SELECTION_SHA256" same39-selection
[[ -d "$V4A_CACHE" && -d "$SAME39_CACHE" ]] || fail 'missing pose-cache input'
[[ -z "$(find "$SAME39_ROOT" -xdev -perm /022 -print -quit)" ]] || fail 'same39 root is not read-only sealed'
readonly NAME="pams-v4d-full337-auth-v2-${SOURCE_REVISION:0:12}-${PAMS_V4D_AUTH_ATTEMPT_ID,,}"
FAILURE_PHASE='run-root-initialize'
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
finalize() {
  local status="$?"
  trap - EXIT
  cleanup
  if [[ "$status" -ne 0 && -d "$RUN_ROOT" && ! -e "$AUDIT_DIR/failure.receipt.json" ]]; then
    python3 - "$RUN_ROOT" "$SOURCE_REVISION" "$IMAGE_ID" "$FAILURE_PHASE" "$status" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); artifacts={}
for relative in ('audit/container.inspect.pre.json','audit/container.inspect.post.json','authorization/extraction.authorization.json'):
 path=root/relative
 if path.is_file(): artifacts[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={'schema_version':1,'artifact_type':'pams_pose_recovery_v4d_full337_extraction_authorization_v2_failure',
 'source_revision':sys.argv[2],'container_image_id':sys.argv[3],'failed_phase':sys.argv[4],
 'exit_status':int(sys.argv[5]),'baseline_training_authorized':False,'artifacts':artifacts,
 'observed_container_image_ids':sorted({
   json.loads((root/relative).read_text())[0]['Image']
   for relative in ('audit/container.inspect.pre.json','audit/container.inspect.post.json')
   if (root/relative).is_file()})}
target=root/'audit/failure.receipt.json'; target.parent.mkdir(parents=True,exist_ok=True)
with target.open('x',encoding='utf-8',newline='\n') as f:
 json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
PY
  fi
  [[ ! -d "$RUN_ROOT" ]] || chmod -R a-w -- "$RUN_ROOT"
  exit "$status"
}
[[ ! -e "$RUN_ROOT" ]] || fail "run root already exists: ${RUN_ROOT}"
mkdir -p -- "$RUN_PARENT"
mkdir -- "$RUN_ROOT"
trap finalize EXIT
mkdir -- "$SOURCE_EXPORT" "$OUTPUT_DIR" "$AUDIT_DIR" "$LOG_DIR"
git -C "$REPOSITORY_ROOT" archive "$SOURCE_REVISION" | tar -x -C "$SOURCE_EXPORT"
chmod -R a-w -- "$SOURCE_EXPORT"
require_sha256 "$SOURCE_EXPORT/configs/gates/pams_pose_recovery_v4d_full337.yaml" "$GATE_SHA256" full337-gate
FAILURE_PHASE='container-create'

docker create --name "$NAME" --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges:true --user 1000:1000 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,uid=1000,gid=1000,mode=1777,size=1g \
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
  --mount "type=bind,src=${OUTPUT_DIR},dst=/pams/output" --workdir /workspace "$IMAGE_ID" \
  python scripts/server/authorize_pose_recovery_v4d_full337_extraction_v2.py \
  --source-revision "$SOURCE_REVISION" --container-image-id "$IMAGE_ID" \
  --gate-sha256 "$GATE_SHA256" --same39-failure-receipt-sha256 "$SAME39_FAILURE_SHA256" \
  --gate /workspace/configs/gates/pams_pose_recovery_v4d_full337.yaml \
  --train-input /pams/inputs.json --train-commitment /pams/commitment.json \
  --v4a-cache-dir /pams/v4a-cache --v4a-ledger /pams/v4a-ledger.json \
  --v4a-paired-gate /pams/v4a-paired.json --same39-failure-receipt /pams/same39/failure.json \
  --same39-audit /pams/same39/audit.json --same39-ledger /pams/same39/ledger.json \
  --same39-selection /pams/same39/selection.json --same39-cache-dir /pams/same39/cache \
  --output /pams/output/extraction.authorization.json >"${AUDIT_DIR}/container.create-id.txt"
docker inspect "$NAME" >"${AUDIT_DIR}/container.inspect.pre.json"
python3 - "$AUDIT_DIR/container.inspect.pre.json" "$IMAGE_ID" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))[0]; h=x['HostConfig']; c=x['Config']
def req(v,m):
 if not v: raise SystemExit(m)
req(h['ReadonlyRootfs'] and h['NetworkMode']=='none' and 'ALL' in h['CapDrop'],'sandbox mismatch')
req(x['Image']==sys.argv[2],'actual container image-ID mismatch')
req(c['User']=='1000:1000' and not h.get('DeviceRequests'),'authorizer must not have GPU')
mounts={m['Destination']:m['RW'] for m in x['Mounts']}
req(mounts.get('/pams/output') is True,'output is not writable')
for destination,writable in mounts.items():
 if destination!='/pams/output': req(writable is False,f'writable input mount: {destination}')
for forbidden in ('/pams/dev','/pams/test','/pams/targets','/pams/videos'):
 req(forbidden not in mounts,f'forbidden mount: {forbidden}')
PY
FAILURE_PHASE='authorization'
set +e
docker start -a "$NAME" 2>&1 | tee "$LOG_DIR/authorization.log"
STATUS="${PIPESTATUS[0]}"
set -e
docker inspect "$NAME" >"${AUDIT_DIR}/container.inspect.post.json"
python3 - "$AUDIT_DIR/container.inspect.post.json" "$IMAGE_ID" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))[0]
if x['Image']!=sys.argv[2]: raise SystemExit('post-run image-ID mismatch')
PY
docker rm "$NAME" >/dev/null
[[ "$STATUS" -eq 0 ]] || fail "authorization container failed with status ${STATUS}"
[[ -f "$AUTHORIZATION" ]] || fail 'authorization receipt missing'
python3 - "$AUTHORIZATION" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['authorization_scope']=='full337_pose_extraction_only'
assert x['full337_pose_extraction_authorized'] is True
assert x['baseline_training_authorized'] is False
assert x['training_runner_must_reject'] is True
PY
python3 - "$RUN_ROOT" "$SOURCE_REVISION" "$IMAGE_ID" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); expected=sys.argv[3]
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
pre=root/'audit/container.inspect.pre.json'; post=root/'audit/container.inspect.post.json'
pre_payload=json.loads(pre.read_text())[0]; post_payload=json.loads(post.read_text())[0]
assert pre_payload['Image']==post_payload['Image']==expected
authorization=root/'authorization/extraction.authorization.json'
payload={'schema_version':1,'artifact_type':'pams_pose_recovery_v4d_full337_extraction_authorization_v2_run_receipt',
 'source_revision':sys.argv[2],'container_image_id':expected,
 'observed_container_image_id_pre':pre_payload['Image'],
 'observed_container_image_id_post':post_payload['Image'],
 'container_inspect_pre_sha256':digest(pre),'container_inspect_post_sha256':digest(post),
 'extraction_authorization_sha256':digest(authorization),
 'authorization_scope':'full337_pose_extraction_only','baseline_training_authorized':False,
 'training_runner_must_reject':True}
with (root/'audit/run.receipt.json').open('x',encoding='utf-8',newline='\n') as f:
 json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
PY
chmod -R a-w -- "$RUN_ROOT"
trap - EXIT
printf 'v4d full337 extraction-only authorization issued: %s sha256=%s\n' "$RUN_ROOT" "$(sha256_file "$AUTHORIZATION")"
