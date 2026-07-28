# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel@sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014
FROM ${BASE_IMAGE}

ARG BASE_IMAGE

LABEL org.opencontainers.image.title="PAMS RAC reproducibility environment" \
      org.opencontainers.image.description="Pinned PyTorch, MediaPipe, and CountLLM-Lite dependencies; source and data are mounted at runtime" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"

ARG DEBIAN_FRONTEND=noninteractive
ARG APT_MIRROR=http://archive.ubuntu.com/ubuntu
ARG APT_SNAPSHOT=20260727T000000Z
ARG PAMS_UID=1000
ARG PAMS_GID=1000

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# The CUDA runtime is pinned in BASE_IMAGE. Remove every inherited apt
# source, including unreachable NVIDIA/CUDA lists, before using one explicitly
# selected Ubuntu mirror for all Jammy suites.
RUN set -eu; \
    case "${APT_MIRROR}" in \
        http://*|https://*) ;; \
        *) echo "APT_MIRROR must be an http(s) URL" >&2; exit 2 ;; \
    esac; \
    case "${APT_SNAPSHOT}" in \
        20[0-9][0-9][01][0-9][0-3][0-9]T[0-2][0-9][0-5][0-9][0-5][0-9]Z) ;; \
        *) echo "APT_SNAPSHOT must be YYYYMMDDTHHMMSSZ" >&2; exit 2 ;; \
    esac; \
    apt_mirror="${APT_MIRROR%/}"; \
    mkdir -p /etc/apt/sources.list.d; \
    rm -f /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; \
    printf '%s\n' \
        "deb [snapshot=${APT_SNAPSHOT}] ${apt_mirror} jammy main restricted universe multiverse" \
        "deb [snapshot=${APT_SNAPSHOT}] ${apt_mirror} jammy-updates main restricted universe multiverse" \
        "deb [snapshot=${APT_SNAPSHOT}] ${apt_mirror} jammy-security main restricted universe multiverse" \
        "deb [snapshot=${APT_SNAPSHOT}] ${apt_mirror} jammy-backports main restricted universe multiverse" \
        > /etc/apt/sources.list; \
    apt-get update --snapshot "${APT_SNAPSHOT}"; \
    apt-get install --yes --no-install-recommends --snapshot "${APT_SNAPSHOT}" \
        ca-certificates=20260601~22.04.1 \
        ffmpeg=7:4.4.2-0ubuntu0.22.04.1 \
        git=1:2.34.1-1ubuntu1.17 \
        libgl1=1.4.0-1 \
        libglib2.0-0=2.72.4-0ubuntu2.9 \
        libportaudio2=19.6.0-1.1 \
        libsm6=2:1.2.3-1build2 \
        libxext6=2:1.3.4-1build1 \
        libxrender1=1:0.9.10-1build4; \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt constraints.txt /opt/pams-environment/
COPY --chmod=0755 pams /usr/local/bin/pams

# MediaPipe requires the opencv-contrib-python distribution.  Installing any
# second OpenCV wheel would make ownership of the shared cv2 package ambiguous.
RUN python -m pip install --no-cache-dir --upgrade \
        pip==24.3.1 \
        setuptools==75.3.0 \
        wheel==0.44.0 \
    && python -m pip uninstall --yes \
        ninja \
        opencv-python \
        opencv-python-headless \
        opencv-contrib-python-headless

RUN python -m pip install --no-cache-dir \
        --constraint /opt/pams-environment/constraints.txt \
        --requirement /opt/pams-environment/requirements.txt \
    && python -m pip check

# Fail the image build if the ABI-sensitive part of the environment differs
# from the preregistered stack.  This check does not initialize CUDA or
# download models.
RUN python -c "import sys; import cv2; import jax; import jaxlib; import mediapipe as mp; import numpy as np; import torch; from google.protobuf import __version__ as protobuf_version; assert sys.version_info[:2] == (3, 11); assert torch.__version__.split('+')[0] == '2.5.1'; assert np.__version__ == '1.26.4'; assert mp.__version__ == '0.10.14'; assert cv2.__version__ == '4.10.0'; assert protobuf_version == '4.25.3'; assert jax.__version__ == '0.4.28'; assert jaxlib.__version__ == '0.4.28'"

RUN set -eux; \
    if ! getent group "${PAMS_GID}" >/dev/null; then \
        groupadd --gid "${PAMS_GID}" pams; \
    fi; \
    runtime_group="$(getent group "${PAMS_GID}" | cut -d: -f1)"; \
    if ! getent passwd "${PAMS_UID}" >/dev/null; then \
        useradd --uid "${PAMS_UID}" --gid "${runtime_group}" --create-home --shell /bin/bash pams; \
    fi; \
    mkdir -p \
        /workspace \
        /pams/artifacts \
        /pams/cache/huggingface \
        /pams/cache/matplotlib \
        /pams/cache/ruff \
        /pams/cache/torch \
        /pams/cache/triton \
        /pams/cache/xdg \
        /pams/checkpoints \
        /pams/data \
        /pams/home \
        /pams/logs \
        /pams/pose-cache \
        /pams/runs \
        /pams/tmp; \
    chown -R "${PAMS_UID}:${PAMS_GID}" /workspace /pams

ENV HOME=/pams/home \
    HF_HOME=/pams/cache/huggingface \
    MPLCONFIGDIR=/pams/cache/matplotlib \
    PYTHONPATH=/workspace/src \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/pams/cache/torch \
    TRITON_CACHE_DIR=/pams/cache/triton \
    XDG_CACHE_HOME=/pams/cache/xdg \
    TMPDIR=/pams/tmp

USER ${PAMS_UID}:${PAMS_GID}
WORKDIR /workspace

CMD ["/bin/bash"]
