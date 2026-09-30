ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel@sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.source="https://github.com/appleweiping/pams-rac-reproduction" \
      org.opencontainers.image.ivac-p2l.source-revision="0b1149e6958268ca5131d60ff66498c6e07f3b79" \
      org.opencontainers.image.ivac-p2l.protocol="modern-compat-dev-sanity"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src:/opt/ivac-source

COPY requirements.in /tmp/ivac-p2l-requirements.in
RUN python -m pip install --no-cache-dir -r /tmp/ivac-p2l-requirements.in \
    && python -m pip check \
    && rm -f /tmp/ivac-p2l-requirements.in

ARG IVAC_UID=10001
ARG IVAC_GID=10001
RUN groupadd --gid ${IVAC_GID} ivac \
    && useradd --uid ${IVAC_UID} --gid ${IVAC_GID} \
       --create-home --shell /bin/bash ivac \
    && mkdir -p /workspace /opt/ivac-source /opt/ivac-assets \
    && chown -R ${IVAC_UID}:${IVAC_GID} \
       /workspace /opt/ivac-source /opt/ivac-assets

COPY --chown=${IVAC_UID}:${IVAC_GID} verify_environment.py /opt/verify_environment.py
USER ${IVAC_UID}:${IVAC_GID}
WORKDIR /workspace
CMD ["python", "/opt/verify_environment.py"]
