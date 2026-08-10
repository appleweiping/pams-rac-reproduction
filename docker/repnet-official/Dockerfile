# syntax=docker/dockerfile:1.7

# Linux/amd64 manifest of the official Python 3.11.10 slim Bookworm image.
# TensorFlow's CUDA libraries are supplied by the hash-locked `and-cuda`
# wheels, so the host contributes only the NVIDIA driver through the
# container runtime.
ARG BASE_IMAGE=public.ecr.aws/docker/library/python:3.11.10-slim-bookworm@sha256:efc99f05ec45381aac55e2803c9a0245ea5b8c74965264498338e24e4bf66cc7
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG REPNET_UID=1000
ARG REPNET_GID=1000

LABEL org.opencontainers.image.title="Official RepNet ckpt-70 isolated sanity environment" \
      org.opencontainers.image.description="Hash-locked TensorFlow 2.17/Keras 3 environment for the official RepNet custom checkpoint mapping" \
      org.opencontainers.image.source="https://github.com/appleweiping/pams-rac-reproduction" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      org.opencontainers.image.repnet.source="https://github.com/google-research/google-research/tree/ec7c3d346277b737bc2decffcd1b533d4b7ec105/repnet" \
      org.opencontainers.image.repnet.source-revision="ec7c3d346277b737bc2decffcd1b533d4b7ec105"

SHELL ["/bin/sh", "-euxc"]

COPY requirements.lock /opt/repnet-official/requirements.lock

# The full Linux dependency graph is generated for Python 3.11, restricted to
# binary distributions, and hash locked. setuptools is installed separately
# because TensorFlow declares it without a version constraint.
RUN python -m pip install --no-cache-dir --upgrade \
        pip==24.3.1 \
        setuptools==75.3.0 \
    && python -m pip install --no-cache-dir \
        --only-binary=:all: \
        --require-hashes \
        --requirement /opt/repnet-official/requirements.lock \
    && python -m pip check

COPY sitecustomize.py verify_environment.py /opt/repnet-official/

# Import-time verification is CPU-safe. It validates the Keras variable API
# used by the official June-2025 custom mapping and the exact CUDA/cuDNN build
# metadata without downloading code, checkpoints, or data.
ENV PYTHONPATH=/opt/repnet-official
RUN python /opt/repnet-official/verify_environment.py

RUN groupadd --gid "${REPNET_GID}" repnet \
    && useradd \
        --uid "${REPNET_UID}" \
        --gid "${REPNET_GID}" \
        --create-home \
        --home-dir /repnet/home \
        --shell /bin/sh \
        repnet \
    && mkdir -p /repnet/cache /repnet/home /repnet/tmp /workspace \
    && chown -R "${REPNET_UID}:${REPNET_GID}" /repnet /workspace

# sitecustomize is imported before user code. Its default 8192 MiB logical
# device cap prevents TensorFlow from preallocating or growing across the
# shared A6000. Operators may deliberately choose a different positive cap.
ENV HOME=/repnet/home \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONPATH=/opt/repnet-official:/workspace/src \
    PYTHONUNBUFFERED=1 \
    REPNET_GPU_MEMORY_LIMIT_MB=8192 \
    TF_CPP_MIN_LOG_LEVEL=1 \
    TMPDIR=/repnet/tmp \
    XDG_CACHE_HOME=/repnet/cache

USER ${REPNET_UID}:${REPNET_GID}
WORKDIR /workspace

CMD ["python", "/opt/repnet-official/verify_environment.py"]
