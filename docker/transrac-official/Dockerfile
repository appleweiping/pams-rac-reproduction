ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel@sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.source="https://github.com/appleweiping/pams-rac-reproduction" \
      org.opencontainers.image.transrac.source-revision="68bdd4daa60ed7c3174a7f6bf86f6537b6fa0979" \
      org.opencontainers.image.transrac.protocol="modern-compat-dev-sanity"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src:/opt/transrac-source

COPY requirements.in /tmp/transrac-requirements.in
RUN python -m pip install --no-cache-dir -r /tmp/transrac-requirements.in \
    && python -m pip check \
    && rm -f /tmp/transrac-requirements.in

ARG TRANSRAC_UID=10001
ARG TRANSRAC_GID=10001
RUN groupadd --gid ${TRANSRAC_GID} transrac \
    && useradd --uid ${TRANSRAC_UID} --gid ${TRANSRAC_GID} \
       --create-home --shell /bin/bash transrac \
    && mkdir -p /workspace /opt/transrac-source /opt/transrac-assets \
    && chown -R ${TRANSRAC_UID}:${TRANSRAC_GID} \
       /workspace /opt/transrac-source /opt/transrac-assets

COPY --chown=${TRANSRAC_UID}:${TRANSRAC_GID} verify_environment.py /opt/verify_environment.py
USER ${TRANSRAC_UID}:${TRANSRAC_GID}
WORKDIR /workspace
CMD ["python", "/opt/verify_environment.py"]
