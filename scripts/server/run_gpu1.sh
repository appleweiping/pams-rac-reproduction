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
RUN_MODE="formal"

usage() {
  cat <<'EOF'
Usage: run_gpu1.sh [--sealed] [--network bridge|none] [--] [COMMAND...]

Default mode is formal: source and data are read-only, persistent output
directories are writable, and Docker's bridge network is enabled.

  --sealed       Force --network none and Hugging Face offline mode.
  --network MODE Select bridge or none explicitly.

Environment overrides:
  PAMS_EXPECTED_ROOT (required), PAMS_ROOT, PAMS_PROJECT_DIR,
  PAMS_IMAGE, PAMS_GPU_DEVICE,
  PAMS_NETWORK_MODE, PAMS_CONTAINER_NAME
EOF
}

while (($#)); do
  case "$1" in
    --sealed)
      RUN_MODE="sealed"
      NETWORK_MODE="none"
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
