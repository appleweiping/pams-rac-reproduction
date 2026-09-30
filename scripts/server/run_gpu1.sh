#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/environment_fingerprint.sh"
readonly DEFAULT_IMAGE="pams-rac:pt2.5.1-cu124-mp0.10.14"
readonly FROZEN_APT_SNAPSHOT="20260727T000000Z"

readonly EXPECTED_ROOT="${PAMS_EXPECTED_ROOT:-}"
[[ -n "${EXPECTED_ROOT}" ]] || {
  echo "PAMS_EXPECTED_ROOT is required and must name the approved absolute data root" >&2
  exit 2
}
case "${EXPECTED_ROOT}" in
  /*) ;;
  *)
    echo "PAMS_EXPECTED_ROOT must be an absolute path" >&2
    exit 2
    ;;
esac

PAMS_ROOT="${PAMS_ROOT:-${EXPECTED_ROOT}}"
PROJECT_DIR="${PAMS_PROJECT_DIR:-${PAMS_ROOT}/repo}"
IMAGE_TAG="${PAMS_IMAGE:-${DEFAULT_IMAGE}}"
GPU_DEVICE="${PAMS_GPU_DEVICE:-1}"
NETWORK_MODE="${PAMS_NETWORK_MODE:-bridge}"
WAIT_FOR_IDLE_GPU="${PAMS_WAIT_FOR_IDLE_GPU:-0}"
GPU_WAIT_TIMEOUT_SECONDS="${PAMS_GPU_WAIT_TIMEOUT_SECONDS:-43200}"
GPU_WAIT_INTERVAL_SECONDS="${PAMS_GPU_WAIT_INTERVAL_SECONDS:-30}"
GPU_STABLE_SAMPLES="${PAMS_GPU_STABLE_SAMPLES:-3}"
GPU_MIN_TOTAL_MIB="${PAMS_GPU_MIN_TOTAL_MIB:-49000}"
GPU_MAX_USED_MIB="${PAMS_GPU_MAX_USED_MIB:-4096}"
GPU_MIN_FREE_MIB="${PAMS_GPU_MIN_FREE_MIB:-44000}"
GPU_MAX_UTILIZATION="${PAMS_GPU_MAX_UTILIZATION:-5}"
GPU_HEAVY_PROCESS_MIB="${PAMS_GPU_HEAVY_PROCESS_MIB:-2048}"
RUN_MODE="formal"

usage() {
  cat <<'EOF'
Usage: run_gpu1.sh [--sealed] [--wait-for-idle-gpu]
                   [--network bridge|none] [--] [COMMAND...]

Default mode is formal: source and data are read-only, persistent output
directories are writable, and Docker's bridge network is enabled.

  --sealed       Force --network none and Hugging Face offline mode.
  --wait-for-idle-gpu
                 Hold the project GPU lock and require three stable idle
                 samples before launch. Resource timeout exits with status 75.
  --network MODE Select bridge or none explicitly.

Environment overrides:
  PAMS_EXPECTED_ROOT (required), PAMS_ROOT, PAMS_PROJECT_DIR,
  PAMS_IMAGE, PAMS_GPU_DEVICE,
  PAMS_NETWORK_MODE, PAMS_CONTAINER_NAME,
  PAMS_WAIT_FOR_IDLE_GPU, PAMS_GPU_WAIT_TIMEOUT_SECONDS,
  PAMS_GPU_WAIT_INTERVAL_SECONDS, PAMS_GPU_STABLE_SAMPLES,
  PAMS_GPU_MIN_TOTAL_MIB, PAMS_GPU_MAX_USED_MIB,
  PAMS_GPU_MIN_FREE_MIB, PAMS_GPU_MAX_UTILIZATION,
  PAMS_GPU_HEAVY_PROCESS_MIB
EOF
}

while (($#)); do
  case "$1" in
    --sealed)
      RUN_MODE="sealed"
      NETWORK_MODE="none"
      shift
      ;;
    --wait-for-idle-gpu)
      WAIT_FOR_IDLE_GPU=1
      shift
      ;;
    --network)
      [[ $# -ge 2 ]] || {
        echo "--network requires bridge or none" >&2
        exit 2
      }
      NETWORK_MODE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ "${RUN_MODE}" == "sealed" ]]; then
  NETWORK_MODE="none"
fi

case "${NETWORK_MODE}" in
  bridge|none) ;;
  *)
    echo "network mode must be bridge or none" >&2
    exit 2
    ;;
esac

[[ "${GPU_DEVICE}" =~ ^[0-9]+$ ]] || {
  echo "PAMS_GPU_DEVICE must be one numeric GPU index" >&2
  exit 2
}
case "${WAIT_FOR_IDLE_GPU}" in
  0|1) ;;
  *)
    echo "PAMS_WAIT_FOR_IDLE_GPU must be 0 or 1" >&2
    exit 2
    ;;
esac
for numeric_setting in \
  GPU_WAIT_TIMEOUT_SECONDS \
  GPU_WAIT_INTERVAL_SECONDS \
  GPU_STABLE_SAMPLES \
  GPU_MIN_TOTAL_MIB \
  GPU_MAX_USED_MIB \
  GPU_MIN_FREE_MIB \
  GPU_MAX_UTILIZATION \
  GPU_HEAVY_PROCESS_MIB
do
  numeric_value="${!numeric_setting}"
  if [[ ! "${numeric_value}" =~ ^[0-9]+$ ]] || ((numeric_value < 1)); then
    echo "${numeric_setting} must be a positive integer" >&2
    exit 2
  fi
done

resolved_expected_root="$(realpath -m -- "${EXPECTED_ROOT}")"
resolved_root="$(realpath -m -- "${PAMS_ROOT}")"
resolved_project="$(realpath -m -- "${PROJECT_DIR}")"
if [[ "${resolved_root}" != "${resolved_expected_root}" ]]; then
  echo "refusing to mount unexpected PAMS_ROOT: ${resolved_root}" >&2
  exit 2
fi
case "${resolved_project}" in
  "${resolved_root}"/*) ;;
  *)
    echo "project directory must remain inside ${resolved_root}" >&2
    exit 2
    ;;
esac

[[ -f "${resolved_project}/pyproject.toml" ]] || {
  echo "project checkout not found at ${resolved_project}" >&2
  exit 1
}

command -v docker >/dev/null 2>&1 || {
  echo "docker is required" >&2
  exit 1
}
docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1 || {
  echo "container image is unavailable: ${IMAGE_TAG}" >&2
  exit 1
}
project_revision="$(git -C "${resolved_project}" rev-parse HEAD 2>/dev/null || true)"
[[ "${project_revision}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "formal execution requires a committed Git checkout" >&2
  exit 2
}
if [[ -n "$(git -C "${resolved_project}" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "formal execution requires a clean Git worktree" >&2
  exit 2
fi
expected_environment_sha256="$(
  pams_environment_fingerprint "${resolved_project}/docker/server"
)"
image_id="$(docker image inspect "${IMAGE_TAG}" --format '{{.Id}}')"
image_revision="$(
  docker image inspect "${IMAGE_TAG}" \
    --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
)"
image_environment_sha256="$(
  docker image inspect "${IMAGE_TAG}" \
    --format '{{index .Config.Labels "org.opencontainers.image.pams.environment-sha256"}}'
)"
image_apt_snapshot="$(
  docker image inspect "${IMAGE_TAG}" \
    --format '{{index .Config.Labels "org.opencontainers.image.pams.apt-snapshot"}}'
)"
if [[ "${image_revision}" != "${project_revision}" ]]; then
  echo "container image revision does not match the checked-out Git revision" >&2
  exit 2
fi
if [[ "${image_environment_sha256}" != "${expected_environment_sha256}" ]]; then
  echo "container image environment fingerprint does not match the checkout" >&2
  exit 2
fi
if [[ "${image_apt_snapshot}" != "${FROZEN_APT_SNAPSHOT}" ]]; then
  echo "container image apt snapshot does not match the frozen snapshot" >&2
  exit 2
fi
[[ "${image_id}" =~ ^sha256:[0-9a-f]{64}$ ]] || {
  echo "container image did not resolve to an immutable image ID" >&2
  exit 2
}

umask 0002
for directory in \
  artifacts \
  cache/huggingface \
  cache/matplotlib \
  cache/ruff \
  cache/torch \
  cache/triton \
  cache/xdg \
  checkpoints \
  data \
  home \
  logs \
  pose-cache \
  runs \
  tmp
do
  persistent_path="${resolved_root}/${directory}"
  mkdir -p -- "${persistent_path}"
  resolved_persistent_path="$(realpath -e -- "${persistent_path}")"
  case "${resolved_persistent_path}" in
    "${resolved_root}"/*) ;;
    *)
      echo "persistent directory escapes ${resolved_root}: ${persistent_path}" >&2
      exit 2
      ;;
  esac
done

command -v flock >/dev/null 2>&1 || {
  echo "flock is required for project GPU serialization" >&2
  exit 1
}
current_uid="$(id -u)"
gpu_lock_directory="${resolved_root}/.pams-gpu-locks"
if [[ -L "${gpu_lock_directory}" ]]; then
  echo "refusing symbolic-link GPU lock directory: ${gpu_lock_directory}" >&2
  exit 2
fi
if [[ ! -e "${gpu_lock_directory}" ]]; then
  if ! (umask 0077; mkdir -- "${gpu_lock_directory}") 2>/dev/null \
    && [[ ! -d "${gpu_lock_directory}" ]]
  then
    echo "failed to create the host-only GPU lock directory" >&2
    exit 1
  fi
fi
if [[ -L "${gpu_lock_directory}" || ! -d "${gpu_lock_directory}" ]]; then
  echo "GPU lock directory must be a real directory" >&2
  exit 2
fi
lock_directory_owner="$(stat -c '%u' -- "${gpu_lock_directory}")"
lock_directory_mode="$(stat -c '%a' -- "${gpu_lock_directory}")"
if [[ "${lock_directory_owner}" != "${current_uid}" ]]; then
  echo "GPU lock directory is not owned by the invoking user" >&2
  exit 2
fi
if [[ "${lock_directory_mode}" != "700" ]]; then
  echo "GPU lock directory permissions must be 0700" >&2
  exit 2
fi

gpu_lock_path="${gpu_lock_directory}/gpu${GPU_DEVICE}.lock"
if [[ -L "${gpu_lock_path}" ]]; then
  echo "refusing symbolic-link GPU lock file: ${gpu_lock_path}" >&2
  exit 2
fi
if [[ ! -e "${gpu_lock_path}" ]]; then
  (umask 0077; set -o noclobber; : >"${gpu_lock_path}") 2>/dev/null || {
    if [[ ! -e "${gpu_lock_path}" ]]; then
      echo "failed to create the GPU lock file" >&2
      exit 1
    fi
  }
fi
if [[ -L "${gpu_lock_path}" || ! -f "${gpu_lock_path}" ]]; then
  echo "GPU lock file must be a real regular file" >&2
  exit 2
fi
lock_file_owner="$(stat -c '%u' -- "${gpu_lock_path}")"
lock_file_mode="$(stat -c '%a' -- "${gpu_lock_path}")"
if [[ "${lock_file_owner}" != "${current_uid}" ]]; then
  echo "GPU lock file is not owned by the invoking user" >&2
  exit 2
fi
if [[ "${lock_file_mode}" != "600" ]]; then
  echo "GPU lock file permissions must be 0600" >&2
  exit 2
fi

exec 9<>"${gpu_lock_path}"
if [[ -L "${gpu_lock_path}" ]]; then
  echo "GPU lock file became a symbolic link while opening it" >&2
  exit 2
fi
lock_path_identity="$(stat -Lc '%d:%i' -- "${gpu_lock_path}")"
lock_fd_identity="$(stat -Lc '%d:%i' -- "/proc/$$/fd/9")"
if [[ "${lock_path_identity}" != "${lock_fd_identity}" ]]; then
  echo "GPU lock file changed while it was being opened" >&2
  exit 2
fi
if ! flock -n 9; then
  echo "another PAMS process holds the project lock for GPU ${GPU_DEVICE}" >&2
  exit 74
fi

if [[ "${WAIT_FOR_IDLE_GPU}" == "1" ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "nvidia-smi is required for the idle-GPU gate" >&2
    exit 1
  }
  wait_started="$(date +%s)"
  stable_samples=0
  while true; do
    gpu_row="$(
      nvidia-smi \
        --id="${GPU_DEVICE}" \
        --query-gpu=memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits
    )"
    gpu_row="${gpu_row// /}"
    IFS=, read -r gpu_total gpu_used gpu_free gpu_utilization <<<"${gpu_row}"
    for observed in "${gpu_total}" "${gpu_used}" "${gpu_free}" "${gpu_utilization}"; do
      [[ "${observed}" =~ ^[0-9]+$ ]] || {
        echo "nvidia-smi returned a non-numeric idle-gate sample" >&2
        exit 1
      }
    done
    heavy_processes="$(
      nvidia-smi \
        --id="${GPU_DEVICE}" \
        --query-compute-apps=used_memory \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -v threshold="${GPU_HEAVY_PROCESS_MIB}" \
            '$1 + 0 > threshold {count++} END {print count + 0}'
    )"
    printf \
      'GPU %s idle gate: total=%s used=%s free=%s utilization=%s heavy=%s stable=%s/%s\n' \
      "${GPU_DEVICE}" \
      "${gpu_total}" \
      "${gpu_used}" \
      "${gpu_free}" \
      "${gpu_utilization}" \
      "${heavy_processes}" \
      "${stable_samples}" \
      "${GPU_STABLE_SAMPLES}"
    if ((
      gpu_total >= GPU_MIN_TOTAL_MIB
      && gpu_used <= GPU_MAX_USED_MIB
      && gpu_free >= GPU_MIN_FREE_MIB
      && gpu_utilization <= GPU_MAX_UTILIZATION
      && heavy_processes == 0
    )); then
      stable_samples=$((stable_samples + 1))
    else
      stable_samples=0
    fi
    if ((stable_samples >= GPU_STABLE_SAMPLES)); then
      break
    fi
    if (( $(date +%s) - wait_started >= GPU_WAIT_TIMEOUT_SECONDS )); then
      echo "GPU ${GPU_DEVICE} did not become stably idle before the resource timeout" >&2
      exit 75
    fi
    sleep "${GPU_WAIT_INTERVAL_SECONDS}"
  done
fi

command=("$@")
if ((${#command[@]} == 0)); then
  command=(bash)
fi

container_name="${PAMS_CONTAINER_NAME:-pams-rac-gpu1}"
docker_arguments=(
  run
  --rm
  --name "${container_name}"
  --hostname pams-rac-gpu1
  --init
  --gpus "device=${GPU_DEVICE}"
  --network "${NETWORK_MODE}"
  --shm-size 32g
  --cpus 24
  --memory 96g
  --pids-limit 8192
  --ulimit memlock=-1
  --ulimit stack=67108864
  --cap-drop ALL
  --security-opt no-new-privileges:true
  --user "$(id -u):$(id -g)"
  --workdir /workspace
  --env "HOME=/pams/home"
  --env "PYTHONPATH=/workspace/src"
  --env "PYTHONUNBUFFERED=1"
  --env "PYTHONHASHSEED=2026"
  --env "CUBLAS_WORKSPACE_CONFIG=:4096:8"
  --env "HF_HOME=/pams/cache/huggingface"
  --env "MPLCONFIGDIR=/pams/cache/matplotlib"
  --env "RUFF_CACHE_DIR=/pams/cache/ruff"
  --env "PYTEST_ADDOPTS=-p no:cacheprovider"
  --env "TORCH_HOME=/pams/cache/torch"
  --env "TRITON_CACHE_DIR=/pams/cache/triton"
  --env "XDG_CACHE_HOME=/pams/cache/xdg"
  --env "TMPDIR=/pams/tmp"
  --env "PAMS_DATA_ROOT=/pams/data"
  --env "PAMS_POSE_CACHE=/pams/pose-cache"
  --env "PAMS_CHECKPOINT_ROOT=/pams/checkpoints"
  --env "PAMS_RUN_ROOT=/pams/runs"
  --env "PAMS_ARTIFACT_ROOT=/pams/artifacts"
  --env "PAMS_AUDIT_MODE=${RUN_MODE}"
  --env "PAMS_CONTAINER_IMAGE_ID=${image_id}"
  --env "PAMS_CONTAINER_ENVIRONMENT_SHA256=${image_environment_sha256}"
  --env "PAMS_CONTAINER_SOURCE_REVISION=${image_revision}"
  --mount "type=bind,src=${resolved_project},dst=/workspace,readonly"
  --mount "type=bind,src=${resolved_root}/data,dst=/pams/data,readonly"
  --mount "type=bind,src=${resolved_root}/pose-cache,dst=/pams/pose-cache"
  --mount "type=bind,src=${resolved_root}/cache,dst=/pams/cache"
  --mount "type=bind,src=${resolved_root}/checkpoints,dst=/pams/checkpoints"
  --mount "type=bind,src=${resolved_root}/runs,dst=/pams/runs"
  --mount "type=bind,src=${resolved_root}/artifacts,dst=/pams/artifacts"
  --mount "type=bind,src=${resolved_root}/logs,dst=/pams/logs"
  --mount "type=bind,src=${resolved_root}/home,dst=/pams/home"
  --mount "type=bind,src=${resolved_root}/tmp,dst=/pams/tmp"
)

if [[ "${NETWORK_MODE}" == "none" ]]; then
  docker_arguments+=(
    --env "HF_HUB_OFFLINE=1"
    --env "TRANSFORMERS_OFFLINE=1"
  )
fi

if [[ -t 0 && -t 1 ]]; then
  docker_arguments+=(-it)
fi

exec docker "${docker_arguments[@]}" "${image_id}" "${command[@]}"
