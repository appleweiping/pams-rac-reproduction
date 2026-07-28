#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly DOCKER_CONTEXT="${REPOSITORY_ROOT}/docker/server"
source "${SCRIPT_DIR}/environment_fingerprint.sh"
readonly DEFAULT_IMAGE="pams-rac:pt2.5.1-cu124-mp0.10.14"
readonly DEFAULT_BASE_IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel@sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014"
readonly FROZEN_BASE_DIGEST="sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014"
readonly DEFAULT_APT_MIRROR="http://archive.ubuntu.com/ubuntu"
readonly FROZEN_APT_SNAPSHOT="20260727T000000Z"

IMAGE_TAG="${PAMS_IMAGE:-${DEFAULT_IMAGE}}"
BASE_IMAGE="${PAMS_BASE_IMAGE:-${DEFAULT_BASE_IMAGE}}"
APT_MIRROR="${PAMS_APT_MIRROR:-${DEFAULT_APT_MIRROR}}"
APT_SNAPSHOT="${PAMS_APT_SNAPSHOT:-${FROZEN_APT_SNAPSHOT}}"
PULL_BASE="${PAMS_PULL_BASE:-1}"
CLEAR_PROXY="${PAMS_CLEAR_PROXY:-0}"

case "${PULL_BASE}" in
  0|1) ;;
  *)
    echo "PAMS_PULL_BASE must be 0 or 1" >&2
    exit 2
    ;;
esac

case "${CLEAR_PROXY}" in
  0|1) ;;
  *)
    echo "PAMS_CLEAR_PROXY must be 0 or 1" >&2
    exit 2
    ;;
esac

if [[ ! "${BASE_IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "PAMS_BASE_IMAGE must be an immutable repository reference ending in @sha256:<64 lowercase hex characters>" >&2
  exit 2
fi
if [[ "${BASE_IMAGE##*@}" != "${FROZEN_BASE_DIGEST}" ]]; then
  echo "PAMS_BASE_IMAGE may change only the repository prefix; its digest must remain ${FROZEN_BASE_DIGEST}" >&2
  exit 2
fi

if [[ ! "${APT_MIRROR}" =~ ^https?://[^/[:space:]]+(/[^[:space:]?#]*)?$ ]]; then
  echo "PAMS_APT_MIRROR must be an http(s) URL without query parameters or fragments" >&2
  exit 2
fi
if [[ "${APT_MIRROR}" == *"@"* ]]; then
  echo "PAMS_APT_MIRROR must not contain URL userinfo or credentials" >&2
  exit 2
fi
APT_MIRROR="${APT_MIRROR%/}"
if [[ "${APT_SNAPSHOT}" != "${FROZEN_APT_SNAPSHOT}" ]]; then
  echo "PAMS_APT_SNAPSHOT must remain ${FROZEN_APT_SNAPSHOT}" >&2
  exit 2
fi

command -v docker >/dev/null 2>&1 || {
  echo "docker is required" >&2
  exit 1
}

pull_arguments=()
if [[ "${PULL_BASE}" == "1" ]]; then
  pull_arguments+=(--pull)
else
  pull_arguments+=(--pull=false)
  docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1 || {
    echo "PAMS_PULL_BASE=0 requires the base image to exist locally: ${BASE_IMAGE}" >&2
    exit 1
  }
fi

proxy_arguments=()
if [[ "${CLEAR_PROXY}" == "1" ]]; then
  proxy_arguments+=(
    --build-arg "HTTP_PROXY="
    --build-arg "HTTPS_PROXY="
    --build-arg "http_proxy="
    --build-arg "https_proxy="
    --build-arg "ALL_PROXY="
    --build-arg "all_proxy="
  )
fi

revision="unknown"
if command -v git >/dev/null 2>&1; then
  revision="$(git -C "${REPOSITORY_ROOT}" rev-parse HEAD 2>/dev/null || printf 'unknown')"
fi
[[ "${revision}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "a committed Git checkout is required to build the formal image" >&2
  exit 2
}
if [[ -n "$(git -C "${REPOSITORY_ROOT}" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "the formal image must be built from a clean Git worktree" >&2
  exit 2
fi
environment_sha256="$(pams_environment_fingerprint "${DOCKER_CONTEXT}")"

echo "Building ${IMAGE_TAG}"
echo "Base image: ${BASE_IMAGE}"
echo "APT mirror: configured (value redacted)"
echo "APT snapshot: ${APT_SNAPSHOT}"
echo "Clear build proxies: ${CLEAR_PROXY}"
echo "Source revision: ${revision}"
echo "Environment SHA-256: ${environment_sha256}"
echo "Build context: ${DOCKER_CONTEXT}"

DOCKER_BUILDKIT=1 docker build \
  "${pull_arguments[@]}" \
  "${proxy_arguments[@]}" \
  --progress=plain \
  --file "${DOCKER_CONTEXT}/Dockerfile" \
  --tag "${IMAGE_TAG}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "APT_MIRROR=${APT_MIRROR}" \
  --build-arg "APT_SNAPSHOT=${APT_SNAPSHOT}" \
  --build-arg "PAMS_UID=$(id -u)" \
  --build-arg "PAMS_GID=$(id -g)" \
  --label "org.opencontainers.image.revision=${revision}" \
  --label "org.opencontainers.image.pams.environment-sha256=${environment_sha256}" \
  --label "org.opencontainers.image.pams.apt-snapshot=${APT_SNAPSHOT}" \
  "${DOCKER_CONTEXT}"

docker image inspect "${IMAGE_TAG}" \
  --format 'image={{.Id}} created={{.Created}} size={{.Size}} environment={{index .Config.Labels "org.opencontainers.image.pams.environment-sha256"}}'
