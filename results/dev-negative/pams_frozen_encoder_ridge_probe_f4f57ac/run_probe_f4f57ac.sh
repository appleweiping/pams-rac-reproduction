#!/usr/bin/env bash
set -Eeuo pipefail

umask 0077

readonly ROOT="/media/lenovo/data2/pams-rac"
readonly PROBE_GIT="f4f57ac434ae02efbde59ed50a2de5da67515bf3"
readonly ENCODER_GIT="6c52288d6a227bc3040fd672c31039408e2d6696"
readonly IMAGE_TAG="pams-rac:f4f57ac"
readonly EXPECTED_IMAGE_ID="sha256:5b2c307be606261ca1a354796475040f799b8e97e6ee4fe7598addbbd7749c4a"
readonly EXPECTED_ENVIRONMENT_SHA256="1ccfd265d829e5160b8af39156193f63622ac73eaefec00b0f9dab40106c3508"
readonly BUILD_CHECKOUT="${ROOT}/formal/probe-f4f57ac-build"
readonly FIREWALL="${ROOT}/runs/firewall/ucfrep-337-84-105-375d8da"
readonly ENCODER_RUN="${ROOT}/runs/ablations/matched-full-multiscale-train337-dev84-b0b85f1-seed2026-150ep-20260729T033000Z-v4"
readonly CONFIG="${ENCODER_RUN}/inputs/receipt-artifacts/97969edcad48af4814df558af910befe46a545320d2658cb2d3cf35c03c7bab9/5c5f4cb734aa9fa496cc7258cb8b1ac07c332f274ae2593db0a75df1fa67a849.artifact"
readonly CHECKPOINT="${ENCODER_RUN}/encoder.pt"
readonly PROGRESS="${ENCODER_RUN}/logs/encoder.jsonl"
readonly ENCODER_RECEIPT="${ENCODER_RUN}/manifests/20260729T033011Z-2d66a0c86c8c.completed.json"
readonly RUN_ROOT="${ROOT}/runs/probes/pams-frozen-encoder-ridge-count-probe-v1-f4f57ac-seed2026-20260729T215500Z-v1"
readonly CODE_VIEW="${RUN_ROOT}/source-view"
readonly TRAIN_POSE_VIEW="${RUN_ROOT}/pose-view-train337"
readonly DEV_POSE_VIEW="${RUN_ROOT}/pose-view-dev84"
readonly TRAIN_OUTPUT="${RUN_ROOT}/train-output"
readonly PREDICT_OUTPUT="${RUN_ROOT}/predict-output"
readonly SCORE_OUTPUT="${RUN_ROOT}/score-output"
readonly LOG_DIR="${RUN_ROOT}/logs"
readonly INSPECT_DIR="${RUN_ROOT}/container-inspect"
readonly DEV_LEDGER_DIR="${ROOT}/runs/dev-attempt-ledger"
readonly DEV_LEDGER="${DEV_LEDGER_DIR}/pams-frozen-encoder-ridge-count-probe-v1.json"
readonly TRAIN_CONTAINER="pams-probe-f4f57ac-train"
readonly PREDICT_CONTAINER="pams-probe-f4f57ac-predict"
readonly SCORE_CONTAINER="pams-probe-f4f57ac-score"

verify_sha256() {
  local expected="$1"
  local path="$2"
  local observed
  observed="$(sha256sum -- "$path" | awk '{print $1}')"
  if [[ "$observed" != "$expected" ]]; then
    echo "SHA-256 mismatch for ${path}: ${observed}" >&2
    return 1
  fi
}

make_pose_view() {
  local manifest="$1"
  local expected_split="$2"
  local expected_count="$3"
  local destination="$4"
  test ! -e "$destination" || {
    echo "refusing reused pose view: ${destination}" >&2
    return 2
  }
  python3 - "$manifest" "${ROOT}/pose-cache" "$destination" "$expected_split" "$expected_count" <<'PY'
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
shared = Path(sys.argv[2]).resolve(strict=True)
destination = Path(sys.argv[3])
expected_split = sys.argv[4]
expected_count = int(sys.argv[5])
payload = json.loads(manifest.read_text(encoding="utf-8"))
assert payload["protocol"] == "ucfrep_526"
assert payload["split"] == expected_split
assert len(payload["records"]) == expected_count
identifiers = [str(record["video_id"]) for record in payload["records"]]
assert len(set(identifiers)) == expected_count
os.mkdir(destination, 0o700)
for identifier in identifiers:
    name = hashlib.sha256(identifier.encode("utf-8")).hexdigest() + ".npz"
    source = (shared / name).resolve(strict=True)
    assert source.parent == shared
    assert source.is_file() and not source.is_symlink()
    output = destination / name
    with source.open("rb") as input_handle, output.open("xb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, 1024 * 1024)
    os.chmod(output, 0o444)
assert len(list(destination.glob("*.npz"))) == expected_count
assert not list(destination.glob("*/*"))
os.chmod(destination, 0o555)
PY
}

verify_container() {
  local container="$1"
  shift
  python3 - "$container" "$@" <<'PY'
import json
import subprocess
import sys

container = sys.argv[1]
expected = {}
for specification in sys.argv[2:]:
    destination, mode = specification.rsplit(":", 1)
    expected[destination] = mode == "rw"
payload = json.loads(
    subprocess.check_output(["docker", "inspect", container], text=True)
)[0]
observed = {
    mount["Destination"]: bool(mount["RW"])
    for mount in payload["Mounts"]
}
assert observed == expected, (observed, expected)
host_config = payload["HostConfig"]
assert host_config["NetworkMode"] == "none"
assert host_config["ReadonlyRootfs"] is True
assert "ALL" in host_config["CapDrop"]
assert "no-new-privileges:true" in host_config["SecurityOpt"]
assert payload["Config"]["Image"].startswith("sha256:")
PY
}

common_create_arguments=(
  --read-only
  --network none
  --shm-size 8g
  --cap-drop ALL
  --security-opt no-new-privileges:true
  --user "$(id -u):$(id -g)"
  --tmpfs /pams/tmp:rw,nosuid,nodev,size=8g,mode=1777
  --env HOME=/pams/tmp
  --env TMPDIR=/pams/tmp
  --env PYTHONPATH=/workspace/src
  --env PYTHONDONTWRITEBYTECODE=1
  --env HF_HUB_OFFLINE=1
  --env TRANSFORMERS_OFFLINE=1
  --env TORCH_HOME=/pams/tmp/torch
  --env TRITON_CACHE_DIR=/pams/tmp/triton
  --env XDG_CACHE_HOME=/pams/tmp/xdg
  --env MPLCONFIGDIR=/pams/tmp/matplotlib
  --env "PAMS_CONTAINER_SOURCE_REVISION=${PROBE_GIT}"
)

test ! -e "$RUN_ROOT"
test "$(git -C "$BUILD_CHECKOUT" rev-parse HEAD)" = "$PROBE_GIT"
test -z "$(git -C "$BUILD_CHECKOUT" status --porcelain=v1 --untracked-files=all)"
for container in "$TRAIN_CONTAINER" "$PREDICT_CONTAINER" "$SCORE_CONTAINER"; do
  if docker inspect "$container" >/dev/null 2>&1; then
    echo "refusing existing container: ${container}" >&2
    exit 2
  fi
done

image_id="$(docker image inspect "$IMAGE_TAG" --format '{{.Id}}')"
image_revision="$(
  docker image inspect "$IMAGE_TAG" \
    --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
)"
image_environment="$(
  docker image inspect "$IMAGE_TAG" \
    --format '{{index .Config.Labels "org.opencontainers.image.pams.environment-sha256"}}'
)"
test "$image_id" = "$EXPECTED_IMAGE_ID"
test "$image_revision" = "$PROBE_GIT"
test "$image_environment" = "$EXPECTED_ENVIRONMENT_SHA256"

verify_sha256 "058c45443b81f6b9861d2896e33544727e8921b7052d61ef331e9ba819be70c8" "$CONFIG"
verify_sha256 "ecaf3c2bd54ffc12c396f9da0044b2799917fb8a6f3811547ba9bcab6fe9d45e" "$CHECKPOINT"
verify_sha256 "4884a3d2740eadad375248e872cb5a1d554f2a23dabb418c4380b8e6eb6536a0" "$PROGRESS"
verify_sha256 "9dc667fa940576c8ea0da6e683d1db63a64e6614b9b42f60d6f2a8e19188bcbc" "$ENCODER_RECEIPT"
verify_sha256 "e238d307b4ed37b2f6c19eefdf7fa8812ec3d231cd2a7091be38e1c279f04207" "${FIREWALL}/train.inputs.json"
verify_sha256 "73999befa45bb55861acfbf8a6b34f04f4a9b34c1e46fa15bc51a88ed6bcfb1e" "${FIREWALL}/train.count-targets.jtsps.json"
verify_sha256 "f21c49569805e4aea326977f6d7ce2964f503fad129a17e95214594364ca4ae7" "${FIREWALL}/dev.inputs.json"
verify_sha256 "1ab3bc010ccb5d869af7662586594c973e53841838643e1e9b8cc1787b0efab6" "${FIREWALL}/dev.targets.json"

mkdir -m 0700 -p "$(dirname "$RUN_ROOT")"
mkdir -m 0700 "$RUN_ROOT"
mkdir -m 0700 "$CODE_VIEW" "$TRAIN_OUTPUT" "$PREDICT_OUTPUT" "$SCORE_OUTPUT" "$LOG_DIR" "$INSPECT_DIR"
sha256sum -- "$0" >"${RUN_ROOT}/launcher.sha256"
git -C "$BUILD_CHECKOUT" archive "$PROBE_GIT" \
  src scripts/pams_encoder_supervised_probe.py \
  | tar -x -C "$CODE_VIEW"
find "$CODE_VIEW" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum >"${RUN_ROOT}/source-view.sha256"

make_pose_view "${FIREWALL}/train.inputs.json" train 337 "$TRAIN_POSE_VIEW"
make_pose_view "${FIREWALL}/dev.inputs.json" dev 84 "$DEV_POSE_VIEW"
find "$TRAIN_POSE_VIEW" -maxdepth 1 -type f -name '*.npz' -print0 \
  | sort -z \
  | xargs -0 sha256sum >"${RUN_ROOT}/train-pose-view.sha256"
find "$DEV_POSE_VIEW" -maxdepth 1 -type f -name '*.npz' -print0 \
  | sort -z \
  | xargs -0 sha256sum >"${RUN_ROOT}/dev-pose-view.sha256"

mkdir -m 0700 -p "$DEV_LEDGER_DIR"
set -o noclobber
printf '%s\n' \
  "method=pams-frozen-encoder-ridge-count-probe-v1" \
  "probe_git=${PROBE_GIT}" \
  "run_root=${RUN_ROOT}" \
  "status=reserved-before-dev-prediction" \
  >"$DEV_LEDGER"
set +o noclobber

exec 9>"${ROOT}/.pams-gpu-locks/gpu0.lock"
flock -n 9

docker create \
  --name "$TRAIN_CONTAINER" \
  --gpus "device=0" \
  "${common_create_arguments[@]}" \
  --mount "type=bind,src=${CODE_VIEW},dst=/workspace,readonly" \
  --mount "type=bind,src=${CONFIG},dst=/pams/data/config.yaml,readonly" \
  --mount "type=bind,src=${CHECKPOINT},dst=/pams/data/encoder.pt,readonly" \
  --mount "type=bind,src=${PROGRESS},dst=/pams/data/encoder.jsonl,readonly" \
  --mount "type=bind,src=${ENCODER_RECEIPT},dst=/pams/data/encoder.completed.json,readonly" \
  --mount "type=bind,src=${FIREWALL}/train.inputs.json,dst=/pams/data/train.inputs.json,readonly" \
  --mount "type=bind,src=${FIREWALL}/train.count-targets.jtsps.json,dst=/pams/data/train.targets.json,readonly" \
  --mount "type=bind,src=${TRAIN_POSE_VIEW},dst=/pams/pose-cache,readonly" \
  --mount "type=bind,src=${TRAIN_OUTPUT},dst=/pams/runs" \
  "$image_id" \
  python /workspace/scripts/pams_encoder_supervised_probe.py train \
    --config /pams/data/config.yaml \
    --encoder-checkpoint /pams/data/encoder.pt \
    --encoder-progress /pams/data/encoder.jsonl \
    --encoder-completion-receipt /pams/data/encoder.completed.json \
    --expected-encoder-source-git-sha "$ENCODER_GIT" \
    --train-inputs /pams/data/train.inputs.json \
    --train-targets /pams/data/train.targets.json \
    --pose-cache /pams/pose-cache \
    --output-dir /pams/runs/train \
    --feature-batch-size 8 \
    --device cuda
docker inspect "$TRAIN_CONTAINER" >"${INSPECT_DIR}/train.before.json"
verify_container "$TRAIN_CONTAINER" \
  /workspace:ro \
  /pams/data/config.yaml:ro \
  /pams/data/encoder.pt:ro \
  /pams/data/encoder.jsonl:ro \
  /pams/data/encoder.completed.json:ro \
  /pams/data/train.inputs.json:ro \
  /pams/data/train.targets.json:ro \
  /pams/pose-cache:ro \
  /pams/runs:rw
docker start -a "$TRAIN_CONTAINER" 2>&1 | tee "${LOG_DIR}/train.log"
docker inspect "$TRAIN_CONTAINER" >"${INSPECT_DIR}/train.after.json"
test -f "${TRAIN_OUTPUT}/train/probe-model.json"
test -f "${TRAIN_OUTPUT}/train/train-run.json"
sha256sum \
  "${TRAIN_OUTPUT}/train/probe-model.json" \
  "${TRAIN_OUTPUT}/train/train-run.json" \
  >"${RUN_ROOT}/train-artifacts.sha256"

docker create \
  --name "$PREDICT_CONTAINER" \
  --gpus "device=0" \
  "${common_create_arguments[@]}" \
  --mount "type=bind,src=${CODE_VIEW},dst=/workspace,readonly" \
  --mount "type=bind,src=${CONFIG},dst=/pams/data/config.yaml,readonly" \
  --mount "type=bind,src=${CHECKPOINT},dst=/pams/data/encoder.pt,readonly" \
  --mount "type=bind,src=${PROGRESS},dst=/pams/data/encoder.jsonl,readonly" \
  --mount "type=bind,src=${ENCODER_RECEIPT},dst=/pams/data/encoder.completed.json,readonly" \
  --mount "type=bind,src=${TRAIN_OUTPUT}/train/probe-model.json,dst=/pams/data/probe-model.json,readonly" \
  --mount "type=bind,src=${FIREWALL}/dev.inputs.json,dst=/pams/data/dev.inputs.json,readonly" \
  --mount "type=bind,src=${DEV_POSE_VIEW},dst=/pams/pose-cache,readonly" \
  --mount "type=bind,src=${PREDICT_OUTPUT},dst=/pams/runs" \
  "$image_id" \
  python /workspace/scripts/pams_encoder_supervised_probe.py predict \
    --config /pams/data/config.yaml \
    --encoder-checkpoint /pams/data/encoder.pt \
    --encoder-progress /pams/data/encoder.jsonl \
    --encoder-completion-receipt /pams/data/encoder.completed.json \
    --expected-encoder-source-git-sha "$ENCODER_GIT" \
    --model /pams/data/probe-model.json \
    --dev-inputs /pams/data/dev.inputs.json \
    --pose-cache /pams/pose-cache \
    --output-dir /pams/runs/predict \
    --feature-batch-size 8 \
    --device cuda
docker inspect "$PREDICT_CONTAINER" >"${INSPECT_DIR}/predict.before.json"
verify_container "$PREDICT_CONTAINER" \
  /workspace:ro \
  /pams/data/config.yaml:ro \
  /pams/data/encoder.pt:ro \
  /pams/data/encoder.jsonl:ro \
  /pams/data/encoder.completed.json:ro \
  /pams/data/probe-model.json:ro \
  /pams/data/dev.inputs.json:ro \
  /pams/pose-cache:ro \
  /pams/runs:rw
docker start -a "$PREDICT_CONTAINER" 2>&1 | tee "${LOG_DIR}/predict.log"
docker inspect "$PREDICT_CONTAINER" >"${INSPECT_DIR}/predict.after.json"
test -f "${PREDICT_OUTPUT}/predict/predictions.json"
test -f "${PREDICT_OUTPUT}/predict/predict-run.json"
sha256sum \
  "${PREDICT_OUTPUT}/predict/predictions.json" \
  "${PREDICT_OUTPUT}/predict/predict-run.json" \
  >"${RUN_ROOT}/prediction-artifacts.frozen.sha256"
chmod 0444 \
  "${PREDICT_OUTPUT}/predict/predictions.json" \
  "${PREDICT_OUTPUT}/predict/predict-run.json"

flock -u 9

docker create \
  --name "$SCORE_CONTAINER" \
  "${common_create_arguments[@]}" \
  --mount "type=bind,src=${CODE_VIEW},dst=/workspace,readonly" \
  --mount "type=bind,src=${PREDICT_OUTPUT}/predict/predictions.json,dst=/pams/data/predictions.json,readonly" \
  --mount "type=bind,src=${PREDICT_OUTPUT}/predict/predict-run.json,dst=/pams/data/predict-run.json,readonly" \
  --mount "type=bind,src=${FIREWALL}/dev.targets.json,dst=/pams/data/dev.targets.json,readonly" \
  --mount "type=bind,src=${SCORE_OUTPUT},dst=/pams/runs" \
  "$image_id" \
  python /workspace/scripts/pams_encoder_supervised_probe.py score \
    --predictions /pams/data/predictions.json \
    --prediction-run /pams/data/predict-run.json \
    --dev-targets /pams/data/dev.targets.json \
    --output /pams/runs/evaluation.json
docker inspect "$SCORE_CONTAINER" >"${INSPECT_DIR}/score.before.json"
verify_container "$SCORE_CONTAINER" \
  /workspace:ro \
  /pams/data/predictions.json:ro \
  /pams/data/predict-run.json:ro \
  /pams/data/dev.targets.json:ro \
  /pams/runs:rw
docker start -a "$SCORE_CONTAINER" 2>&1 | tee "${LOG_DIR}/score.log"
docker inspect "$SCORE_CONTAINER" >"${INSPECT_DIR}/score.after.json"
test -f "${SCORE_OUTPUT}/evaluation.json"
sha256sum "${SCORE_OUTPUT}/evaluation.json" >"${RUN_ROOT}/evaluation.sha256"

printf '%s\n' \
  "method=pams-frozen-encoder-ridge-count-probe-v1" \
  "probe_git=${PROBE_GIT}" \
  "run_root=${RUN_ROOT}" \
  "status=scored-once" \
  >"${DEV_LEDGER}.completed"

python3 - "${SCORE_OUTPUT}/evaluation.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps({
    "metrics": payload["metrics"],
    "raw_metrics": payload["raw_metrics"],
    "claim": payload["claim"],
    "sealed_test105_access": payload["sealed_test105_access"],
}, sort_keys=True))
PY
