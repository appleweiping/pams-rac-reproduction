# Environment-only image for the frozen EveryShotCounts official demo path.
# It intentionally contains neither the upstream ESCounts checkout nor weights.
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime@sha256:cf9197f9321ac3f49276633b4e78c79aa55f22578de3b650b3158ce6e3481f61

ARG DEBIAN_FRONTEND=noninteractive
ARG MICROMAMBA_VERSION=2.8.1
ARG MICROMAMBA_SHA256=a934c3709c997feae403a27fd1e321c106d26ffa4f294800ffb11cbc9a3e8515
ARG PYTORCHVIDEO_COMMIT=fae0d89a194a2c1ca99e59eab6eedd40bde38726

RUN apt-get update \
    && apt-get install -y --no-install-recommends bzip2 ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://micro.mamba.pm/api/micromamba/linux-64/${MICROMAMBA_VERSION}" \
        -o /tmp/micromamba.tar.bz2 \
    && echo "${MICROMAMBA_SHA256}  /tmp/micromamba.tar.bz2" | sha256sum -c - \
    && tar -xjf /tmp/micromamba.tar.bz2 -C /usr/local/bin \
        --strip-components=1 bin/micromamba \
    && rm /tmp/micromamba.tar.bz2

RUN micromamba create -y -p /opt/escounts \
        -c pytorch -c conda-forge \
        python=3.8.13 \
        pytorch=1.10.0 \
        torchvision=0.11.0 \
        cudatoolkit=11.3 \
        numpy=1.21.2 \
        av=10.0.0 \
        ffmpeg=5.1.2 \
        mkl=2021.4 \
        pip=24.0 \
    && micromamba clean --all --yes

ENV PATH=/opt/escounts/bin:${PATH}
ENV PYTHONNOUSERSITE=1
ENV PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir \
        einops==0.3.2 \
        fvcore==0.1.5.post20221221 \
        iopath==0.1.9 \
        opencv-python==4.8.1.78 \
        pandas==1.3.5 \
        parameterized==0.9.0 \
        psutil==5.8.0 \
        scikit-image==0.19.3 \
        simplejson==3.19.3 \
        timm==0.6.13 \
        tqdm==4.59.0 \
        yacs==0.1.8 \
    && python -m pip install --no-cache-dir \
        detectron2==0.6 \
        -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html

RUN git clone --filter=blob:none https://github.com/facebookresearch/pytorchvideo \
        /opt/pytorchvideo \
    && git -C /opt/pytorchvideo checkout --detach "${PYTORCHVIDEO_COMMIT}" \
    && test "$(git -C /opt/pytorchvideo rev-parse HEAD)" = "${PYTORCHVIDEO_COMMIT}" \
    && python -m pip install --no-cache-dir --no-deps /opt/pytorchvideo

RUN python - <<'PY'
import av
import cv2
import numpy
import torch
import torchvision

assert torch.__version__.startswith("1.10.0")
assert torchvision.__version__.startswith("0.11.0")
assert torch.version.cuda == "11.3"
assert numpy.__version__ == "1.21.2"
assert av.__version__ == "10.0.0"
assert cv2.__version__ == "4.8.1"
PY

WORKDIR /workspace
