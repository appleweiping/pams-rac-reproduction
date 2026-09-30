from __future__ import annotations

import re
from pathlib import Path

REPOSITORY = Path(__file__).parents[1]
CONTEXT = REPOSITORY / "docker" / "repnet-official"
BASE_DIGEST = "sha256:efc99f05ec45381aac55e2803c9a0245ea5b8c74965264498338e24e4bf66cc7"
SOURCE_REVISION = "ec7c3d346277b737bc2decffcd1b533d4b7ec105"


def _requirements() -> str:
    return (CONTEXT / "requirements.lock").read_text(encoding="utf-8")


def test_repnet_image_pins_base_source_and_compatibility_stack() -> None:
    dockerfile = (CONTEXT / "Dockerfile").read_text(encoding="utf-8")
    assert (
        "ARG BASE_IMAGE=public.ecr.aws/docker/library/"
        f"python:3.11.10-slim-bookworm@{BASE_DIGEST}"
    ) in dockerfile
    assert f'org.opencontainers.image.repnet.source-revision="{SOURCE_REVISION}"' in dockerfile
    assert "USER ${REPNET_UID}:${REPNET_GID}" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "--only-binary=:all:" in dockerfile
    assert "python -m pip check" in dockerfile

    lock = _requirements()
    for requirement in (
        "tensorflow==2.17.1",
        "keras==3.5.0",
        "numpy==1.26.4",
        "opencv-python-headless==4.10.0.84",
        "scipy==1.14.1",
        "nvidia-cuda-runtime-cu12==12.3.101",
        "nvidia-cudnn-cu12==8.9.7.29",
    ):
        assert requirement in lock
    lines = lock.splitlines()
    package_lines = [
        index
        for index, line in enumerate(lines)
        if line and not line.startswith((" ", "#"))
    ]
    assert package_lines
    for index in package_lines:
        assert lines[index].endswith("\\")
        assert lines[index + 1].lstrip().startswith("--hash=sha256:")


def test_repnet_image_enforces_a_preimport_gpu_memory_limit() -> None:
    dockerfile = (CONTEXT / "Dockerfile").read_text(encoding="utf-8")
    policy = (CONTEXT / "sitecustomize.py").read_text(encoding="utf-8")
    verifier = (CONTEXT / "verify_environment.py").read_text(encoding="utf-8")

    assert "REPNET_GPU_MEMORY_LIMIT_MB=8192" in dockerfile
    runtime_pythonpaths = re.findall(r"^\s*PYTHONPATH=([^ \\\\\n]+)", dockerfile, re.MULTILINE)
    assert runtime_pythonpaths[-1] == "/opt/repnet-official:/workspace/src"
    assert "tf.config.set_logical_device_configuration" in policy
    assert "tf.config.LogicalDeviceConfiguration" in policy
    assert "tf.config.experimental.set_memory_growth" not in policy
    assert "os._exit(78)" in policy
    assert "REPNET_TF_GPU_POLICY_APPLIED" in policy
    assert "get_logical_device_configuration" in verifier


def test_repnet_runtime_has_only_the_dependencies_used_by_selected_notebook_code() -> None:
    dockerfile = (CONTEXT / "Dockerfile").read_text(encoding="utf-8")
    direct_requirements = {
        line.strip()
        for line in (CONTEXT / "requirements.in").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert direct_requirements == {
        "tensorflow[and-cuda]==2.17.1",
        "keras==3.5.0",
        "numpy==1.26.4",
        "opencv-python-headless==4.10.0.84",
        "scipy==1.14.1",
    }
    assert not re.search(r"\b(?:apt-get|apt|apk)\b[^\n]*\bgit\b", dockerfile)
    assert "git clone" not in dockerfile
    assert "apt-get install" not in dockerfile
    for unnecessary_notebook_runtime in (
        "ipython",
        "jupyter",
        "matplotlib",
        "nbformat",
        "pillow",
    ):
        assert unnecessary_notebook_runtime not in direct_requirements


def test_repnet_environment_scope_and_mapping_limitations_are_explicit() -> None:
    note = (CONTEXT / "ENVIRONMENT.md").read_text(encoding="utf-8")
    assert "contains no RepNet source" in note
    assert "PYTHONPATH=/opt/repnet-official:/workspace/src" in note
    assert "Git executable" in note
    assert "no operating-system packages" in note
    assert "does **not** prove" in note
    assert "every mapping entry" in note
    assert "CUDA 12.3" in note
    assert "550.54.14" in note
