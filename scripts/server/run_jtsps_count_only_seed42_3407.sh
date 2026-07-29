#!/usr/bin/env bash
set -Eeuo pipefail

umask 0077

readonly ROOT="/media/lenovo/data2/pams-rac"
readonly SOURCE_CHECKOUT="${ROOT}/checkouts/jtsps-9df2646"
readonly SOURCE_GIT="9df2646df90cb65e60b5da06616cbc47427ba3e9"
readonly RUNNER_SHA256="1fc9311d037b9dc333f81941589955345c0ad480a1aea2fe9b3791fa4d10af31"
readonly SCAFFOLD_SHA256="0e9a43faf794f32332685256a0f6d550cc0aba136ba8027b1ff55c4e82754a1a"
readonly IMAGE_ID="sha256:022103f69a42ef3e88ee43ddcfbd706dabfdeec6247c257d1d01fda2c0ea2f91"
readonly FIREWALL="${ROOT}/runs/firewall/ucfrep-337-84-105-375d8da"
readonly POSE_FINGERPRINT="8ecb6c1384e1d6a088762322ed90bb6aa631683318e0ab7ed21c19541e5f6656"
readonly RUN_ROOT="${ROOT}/runs/baselines/jtsps-count-only-inferred-three-seed-expansion-9df2646-20260729T222000Z-v2"
readonly CODE_VIEW="${RUN_ROOT}/source-view"
readonly POSE_VIEW="${RUN_ROOT}/pose-view-train337-dev84"
readonly LOG_DIR="${RUN_ROOT}/logs"
readonly INSPECT_DIR="${RUN_ROOT}/container-inspect"
readonly DEV_LEDGER_DIR="${ROOT}/runs/dev-attempt-ledger"

verify_sha256() {
  local expected="$1"
  local path="$2"
  local observed
  observed="$(sha256sum -- "$path" | awk '{print $1}')"
  if [[ "$observed" != "$expected" ]]; then
    echo "SHA-256 mismatch for ${path}: expected ${expected}, observed ${observed}" >&2
    return 1
  fi
}

make_combined_pose_view() {
  local train_manifest="$1"
  local dev_manifest="$2"
  local shared_cache="$3"
  local destination="$4"
  local membership_output="$5"
  test ! -e "$destination" || {
    echo "refusing reused pose view: ${destination}" >&2
    return 2
  }
  python3 - \
    "$train_manifest" \
    "$dev_manifest" \
    "$shared_cache" \
    "$destination" \
    "$membership_output" <<'PY'
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

train_manifest = Path(sys.argv[1])
dev_manifest = Path(sys.argv[2])
shared = Path(sys.argv[3]).resolve(strict=True)
destination = Path(sys.argv[4])
membership_output = Path(sys.argv[5])

def load(path, expected_split, expected_count):
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["protocol"] == "ucfrep_526"
    assert payload["split"] == expected_split
    records = payload["records"]
    assert len(records) == expected_count
    identifiers = [str(record["video_id"]) for record in records]
    assert len(set(identifiers)) == expected_count
    return identifiers

train_ids = load(train_manifest, "train", 337)
dev_ids = load(dev_manifest, "dev", 84)
assert not set(train_ids).intersection(dev_ids)
identifiers = train_ids + dev_ids
assert len(identifiers) == 421

os.mkdir(destination, 0o700)
inventory = []
for identifier in identifiers:
    name = hashlib.sha256(identifier.encode("utf-8")).hexdigest() + ".npz"
    source = (shared / name).resolve(strict=True)
    assert source.parent == shared
    assert source.is_file() and not source.is_symlink()
    output = destination / name
    with source.open("rb") as input_handle, output.open("xb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, 1024 * 1024)
    os.chmod(output, 0o444)
    inventory.append(
        {
            "cache_file": name,
            "split": "train" if identifier in set(train_ids) else "dev",
            "video_id": identifier,
        }
    )

assert len(list(destination.glob("*.npz"))) == 421
assert not list(destination.glob("*/*"))
membership_output.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "protocol": "ucfrep_526",
            "train_count": 337,
            "dev_count": 84,
            "test_count": 0,
            "records": inventory,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
os.chmod(membership_output, 0o444)
os.chmod(destination, 0o555)
PY
}

verify_container() {
  local container="$1"
  local expected_gpu="$2"
  shift 2
  python3 - "$container" "$expected_gpu" "$IMAGE_ID" "$@" <<'PY'
import json
import os
import subprocess
import sys

container = sys.argv[1]
expected_gpu = sys.argv[2]
expected_image = sys.argv[3]
expected_mounts = {}
for specification in sys.argv[4:]:
    destination, mode, source = specification.split(":", 2)
    expected_mounts[destination] = {
        "rw": mode == "rw",
        "source": os.path.realpath(source),
    }

payload = json.loads(
    subprocess.check_output(["docker", "inspect", container], text=True)
)[0]
observed_mounts = {
    mount["Destination"]: {
        "rw": bool(mount["RW"]),
        "source": os.path.realpath(mount["Source"]),
    }
    for mount in payload["Mounts"]
}
assert observed_mounts == expected_mounts, (observed_mounts, expected_mounts)
host_config = payload["HostConfig"]
assert host_config["NetworkMode"] == "none"
assert host_config["ReadonlyRootfs"] is True
assert "ALL" in host_config["CapDrop"]
assert "no-new-privileges:true" in host_config["SecurityOpt"]
assert payload["Image"] == expected_image
assert payload["Config"]["User"] == f"{os.getuid()}:{os.getgid()}"
assert "/pams/tmp" in (host_config.get("Tmpfs") or {})

environment = dict(
    item.split("=", 1)
    for item in payload["Config"]["Env"]
    if "=" in item
)
assert environment["PYTHONPATH"] == "/workspace/src"
assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
assert environment["PAMS_CONTAINER_SOURCE_REVISION"] == (
    "9df2646df90cb65e60b5da06616cbc47427ba3e9"
)

requests = host_config.get("DeviceRequests") or []
if expected_gpu == "none":
    assert not requests, requests
else:
    assert len(requests) == 1, requests
    assert requests[0].get("DeviceIDs") == [expected_gpu], requests
    assert "gpu" in (requests[0].get("Capabilities") or [[]])[0], requests
PY
}

common_create_arguments=(
  --read-only
  --network none
  --shm-size 2g
  --cap-drop ALL
  --security-opt no-new-privileges:true
  --user "$(id -u):$(id -g)"
  --tmpfs /pams/tmp:rw,nosuid,nodev,size=4g,mode=1777
  --env HOME=/pams/tmp
  --env TMPDIR=/pams/tmp
  --env PYTHONPATH=/workspace/src
  --env PYTHONDONTWRITEBYTECODE=1
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8
  --env HF_HUB_OFFLINE=1
  --env TRANSFORMERS_OFFLINE=1
  --env TORCH_HOME=/pams/tmp/torch
  --env XDG_CACHE_HOME=/pams/tmp/xdg
  --env MPLCONFIGDIR=/pams/tmp/matplotlib
  --env "PAMS_CONTAINER_SOURCE_REVISION=${SOURCE_GIT}"
)

test ! -e "$RUN_ROOT"
test -d "${ROOT}/.pams-gpu-locks"
test "$(git -C "$SOURCE_CHECKOUT" rev-parse HEAD)" = "$SOURCE_GIT"
test -z "$(git -C "$SOURCE_CHECKOUT" status --porcelain=v1 --untracked-files=all)"
verify_sha256 "$RUNNER_SHA256" \
  "${SOURCE_CHECKOUT}/src/pams/baselines/jtsps_count_only_experiment.py"
verify_sha256 "$SCAFFOLD_SHA256" \
  "${SOURCE_CHECKOUT}/src/pams/baselines/pose_cleanroom.py"
verify_sha256 \
  "e238d307b4ed37b2f6c19eefdf7fa8812ec3d231cd2a7091be38e1c279f04207" \
  "${FIREWALL}/train.inputs.json"
verify_sha256 \
  "73999befa45bb55861acfbf8a6b34f04f4a9b34c1e46fa15bc51a88ed6bcfb1e" \
  "${FIREWALL}/train.count-targets.jtsps.json"
verify_sha256 \
  "f21c49569805e4aea326977f6d7ce2964f503fad129a17e95214594364ca4ae7" \
  "${FIREWALL}/dev.inputs.json"
verify_sha256 \
  "1ab3bc010ccb5d869af7662586594c973e53841838643e1e9b8cc1787b0efab6" \
  "${FIREWALL}/dev.targets.json"
test "$(docker image inspect "$IMAGE_ID" --format '{{.Id}}')" = "$IMAGE_ID"

for seed in 42 3407; do
  for stage in train score; do
    container="jtsps-9df2646-seed${seed}-${stage}-20260729v2"
    if docker inspect "$container" >/dev/null 2>&1; then
      echo "refusing existing container: ${container}" >&2
      exit 2
    fi
  done
  ledger="${DEV_LEDGER_DIR}/jtsps-count-only-inferred-seed${seed}-9df2646-v2.json"
  test ! -e "$ledger" || {
    echo "refusing existing dev-attempt ledger: ${ledger}" >&2
    exit 2
  }
done

mkdir -m 0700 -p "$(dirname "$RUN_ROOT")"
mkdir -m 0700 "$RUN_ROOT"
mkdir -m 0700 "$CODE_VIEW" "$LOG_DIR" "$INSPECT_DIR"
for seed in 42 3407; do
  mkdir -m 0700 \
    "${RUN_ROOT}/seed-${seed}" \
    "${RUN_ROOT}/seed-${seed}/output" \
    "${RUN_ROOT}/seed-${seed}/score"
done

sha256sum -- "$0" >"${RUN_ROOT}/launcher.sha256"
git -C "$SOURCE_CHECKOUT" archive "$SOURCE_GIT" src | tar -x -C "$CODE_VIEW"
find "$CODE_VIEW" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum >"${RUN_ROOT}/source-view.sha256"
find "$CODE_VIEW" -type f -exec chmod 0444 {} +
find "$CODE_VIEW" -type d -exec chmod 0555 {} +
chmod 0555 "$CODE_VIEW"

make_combined_pose_view \
  "${FIREWALL}/train.inputs.json" \
  "${FIREWALL}/dev.inputs.json" \
  "${ROOT}/pose-cache" \
  "$POSE_VIEW" \
  "${RUN_ROOT}/pose-view-membership.json"
find "$POSE_VIEW" -maxdepth 1 -type f -name '*.npz' -print0 \
  | sort -z \
  | xargs -0 sha256sum >"${RUN_ROOT}/pose-view.sha256"

{
  printf 'utc_started='
  date -u +%Y-%m-%dT%H:%M:%SZ
  printf 'hostname='
  hostname
  printf 'kernel='
  uname -srmo
  printf 'docker_server='
  docker version --format '{{.Server.Version}}'
  printf 'image_id=%s\n' "$IMAGE_ID"
  printf 'source_git=%s\n' "$SOURCE_GIT"
  printf 'pose_fingerprint=%s\n' "$POSE_FINGERPRINT"
  printf 'cublas_workspace_config=:4096:8\n'
  nvidia-smi \
    --query-gpu=index,uuid,name,driver_version,memory.total \
    --format=csv,noheader
} >"${RUN_ROOT}/hardware-and-runtime.txt"

mkdir -m 0700 -p "$DEV_LEDGER_DIR"
for seed in 42 3407; do
  ledger="${DEV_LEDGER_DIR}/jtsps-count-only-inferred-seed${seed}-9df2646-v2.json"
  set -o noclobber
  printf '%s\n' \
    "method=jtsps-count-only" \
    "status=inferred-clean-room-not-source-parity" \
    "source_git=${SOURCE_GIT}" \
    "seed=${seed}" \
    "run_root=${RUN_ROOT}/seed-${seed}" \
    "dev_split=84" \
    "test105_access=false" \
    "state=reserved-before-dev-prediction" \
    >"$ledger"
  set +o noclobber
done

run_seed() {
  local seed="$1"
  local gpu="$2"
  local seed_root="${RUN_ROOT}/seed-${seed}"
  local output_root="${seed_root}/output"
  local container="jtsps-9df2646-seed${seed}-train-20260729v2"
  local lock_file="${ROOT}/.pams-gpu-locks/gpu${gpu}.lock"
  local lock_fd

  exec {lock_fd}>"$lock_file"
  flock -n "$lock_fd"

  docker create \
    --name "$container" \
    --gpus "device=${gpu}" \
    "${common_create_arguments[@]}" \
    --mount "type=bind,src=${CODE_VIEW},dst=/workspace,readonly" \
    --mount "type=bind,src=${FIREWALL}/train.inputs.json,dst=/pams/data/train.inputs.json,readonly" \
    --mount "type=bind,src=${FIREWALL}/train.count-targets.jtsps.json,dst=/pams/data/train.targets.json,readonly" \
    --mount "type=bind,src=${FIREWALL}/dev.inputs.json,dst=/pams/data/dev.inputs.json,readonly" \
    --mount "type=bind,src=${POSE_VIEW},dst=/pams/pose-cache,readonly" \
    --mount "type=bind,src=${output_root},dst=/pams/runs" \
    "$IMAGE_ID" \
    python -m pams.baselines.jtsps_count_only_experiment run \
      --train-inputs /pams/data/train.inputs.json \
      --train-targets /pams/data/train.targets.json \
      --dev-inputs /pams/data/dev.inputs.json \
      --pose-cache /pams/pose-cache \
      --pose-fingerprint "$POSE_FINGERPRINT" \
      --output-dir /pams/runs/artifacts \
      --seed "$seed" \
      --epochs 30 \
      --batch-size 8 \
      --lr 1e-4 \
      --device cuda:0
  docker inspect "$container" >"${INSPECT_DIR}/seed-${seed}-train.before.json"
  verify_container "$container" "$gpu" \
    "/workspace:ro:${CODE_VIEW}" \
    "/pams/data/train.inputs.json:ro:${FIREWALL}/train.inputs.json" \
    "/pams/data/train.targets.json:ro:${FIREWALL}/train.count-targets.jtsps.json" \
    "/pams/data/dev.inputs.json:ro:${FIREWALL}/dev.inputs.json" \
    "/pams/pose-cache:ro:${POSE_VIEW}" \
    "/pams/runs:rw:${output_root}"
  docker start -a "$container" \
    > >(tee "${LOG_DIR}/seed-${seed}-train.stdout.log") \
    2> >(tee "${LOG_DIR}/seed-${seed}-train.stderr.log" >&2)
  docker inspect "$container" >"${INSPECT_DIR}/seed-${seed}-train.after.json"
  test "$(docker inspect "$container" --format '{{.State.ExitCode}}')" = "0"
  test -f "${output_root}/artifacts/checkpoint.pt"
  test -f "${output_root}/artifacts/predictions.json"
  test -f "${output_root}/artifacts/run.json"
  sha256sum \
    "${output_root}/artifacts/checkpoint.pt" \
    "${output_root}/artifacts/predictions.json" \
    "${output_root}/artifacts/run.json" \
    >"${seed_root}/train-and-prediction-artifacts.frozen.sha256"
  chmod 0444 \
    "${output_root}/artifacts/checkpoint.pt" \
    "${output_root}/artifacts/predictions.json" \
    "${output_root}/artifacts/run.json"

  flock -u "$lock_fd"
}

run_seed 42 0 &
pid42=$!
run_seed 3407 1 &
pid3407=$!
train_status42=0
train_status3407=0
wait "$pid42" || train_status42=$?
wait "$pid3407" || train_status3407=$?
if ((train_status42 != 0 || train_status3407 != 0)); then
  echo "parallel training failed: seed42=${train_status42}, seed3407=${train_status3407}" >&2
  exit 1
fi

score_seed() {
  local seed="$1"
  local seed_root="${RUN_ROOT}/seed-${seed}"
  local prediction="${seed_root}/output/artifacts/predictions.json"
  local score_root="${seed_root}/score"
  local container="jtsps-9df2646-seed${seed}-score-20260729v2"

  verify_sha256 \
    "$(awk '$2 ~ /predictions.json$/ {print $1}' "${seed_root}/train-and-prediction-artifacts.frozen.sha256")" \
    "$prediction"
  docker create \
    --name "$container" \
    "${common_create_arguments[@]}" \
    --mount "type=bind,src=${CODE_VIEW},dst=/workspace,readonly" \
    --mount "type=bind,src=${prediction},dst=/pams/data/predictions.json,readonly" \
    --mount "type=bind,src=${FIREWALL}/dev.targets.json,dst=/pams/data/dev.targets.json,readonly" \
    --mount "type=bind,src=${score_root},dst=/pams/runs" \
    "$IMAGE_ID" \
    python -m pams.baselines.jtsps_count_only_experiment score \
      --predictions /pams/data/predictions.json \
      --dev-targets /pams/data/dev.targets.json \
      --output /pams/runs/evaluation.json
  docker inspect "$container" >"${INSPECT_DIR}/seed-${seed}-score.before.json"
  verify_container "$container" none \
    "/workspace:ro:${CODE_VIEW}" \
    "/pams/data/predictions.json:ro:${prediction}" \
    "/pams/data/dev.targets.json:ro:${FIREWALL}/dev.targets.json" \
    "/pams/runs:rw:${score_root}"
  docker start -a "$container" \
    > >(tee "${LOG_DIR}/seed-${seed}-score.stdout.log") \
    2> >(tee "${LOG_DIR}/seed-${seed}-score.stderr.log" >&2)
  docker inspect "$container" >"${INSPECT_DIR}/seed-${seed}-score.after.json"
  test "$(docker inspect "$container" --format '{{.State.ExitCode}}')" = "0"
  test -f "${score_root}/evaluation.json"
  sha256sum "${score_root}/evaluation.json" >"${seed_root}/evaluation.sha256"
  chmod 0444 "${score_root}/evaluation.json"
  printf '%s\n' \
    "method=jtsps-count-only" \
    "status=inferred-clean-room-not-source-parity" \
    "source_git=${SOURCE_GIT}" \
    "seed=${seed}" \
    "run_root=${seed_root}" \
    "dev_split=84" \
    "test105_access=false" \
    "state=scored-once" \
    >"${DEV_LEDGER_DIR}/jtsps-count-only-inferred-seed${seed}-9df2646-v2.completed"
}

score_seed 42 &
score_pid42=$!
score_seed 3407 &
score_pid3407=$!
score_status42=0
score_status3407=0
wait "$score_pid42" || score_status42=$?
wait "$score_pid3407" || score_status3407=$?
if ((score_status42 != 0 || score_status3407 != 0)); then
  echo "parallel scoring failed: seed42=${score_status42}, seed3407=${score_status3407}" >&2
  exit 1
fi

python3 - "$RUN_ROOT" <<'PY'
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
rows = []
for seed in (42, 3407):
    path = run_root / f"seed-{seed}" / "score" / "evaluation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    assert metrics["sample_count"] == 84
    values = {
        "seed": seed,
        "evaluation_path": str(path.relative_to(run_root)),
        "evaluation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "nmae": float(metrics["nmae"]),
        "obo": float(metrics["obo"]),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "raw_nmae": float(payload["raw_metrics"]["nmae"]),
        "raw_mae": float(payload["raw_metrics"]["mae"]),
        "raw_rmse": float(payload["raw_metrics"]["rmse"]),
    }
    assert all(math.isfinite(value) for key, value in values.items() if key not in {"seed", "evaluation_path", "evaluation_sha256"})
    rows.append(values)

summary = {
    "schema_version": 1,
    "method": "jtsps-count-only",
    "status": "inferred-clean-room-not-source-parity",
    "claim_scope": "new-seeds-only; combine with frozen seed2026 separately",
    "protocol": "ucfrep_526_train337_dev84",
    "source_git": "9df2646df90cb65e60b5da06616cbc47427ba3e9",
    "seeds": rows,
    "new_seed_mean": {
        key: statistics.fmean(row[key] for row in rows)
        for key in ("nmae", "obo", "mae", "rmse", "raw_nmae", "raw_mae", "raw_rmse")
    },
    "new_seed_sample_sd": {
        key: statistics.stdev(row[key] for row in rows)
        for key in ("nmae", "obo", "mae", "rmse", "raw_nmae", "raw_mae", "raw_rmse")
    },
    "dev_targets_reachable_during_training_or_prediction": False,
    "test105_access": False,
    "cublas_workspace_config": ":4096:8",
}
output = run_root / "new-seeds-summary.json"
output.write_text(
    json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, sort_keys=True, allow_nan=False))
PY
sha256sum "${RUN_ROOT}/new-seeds-summary.json" >"${RUN_ROOT}/new-seeds-summary.sha256"
chmod 0444 "${RUN_ROOT}/new-seeds-summary.json"

{
  printf 'utc_completed='
  date -u +%Y-%m-%dT%H:%M:%SZ
  printf 'test105_access=false\n'
} >"${RUN_ROOT}/completion.txt"
