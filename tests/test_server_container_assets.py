from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]
SERVER_DOCKER = REPOSITORY_ROOT / "docker" / "server"
SERVER_SCRIPTS = REPOSITORY_ROOT / "scripts" / "server"
ENVIRONMENT_FINGERPRINT = SERVER_SCRIPTS / "environment_fingerprint.sh"
BASE_DIGEST = "sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014"
APT_SNAPSHOT = "20260727T000000Z"


def _non_comment_requirements(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _environment_fingerprint(context: Path) -> str:
    payload = bytearray()
    for relative in ("Dockerfile", "constraints.txt", "pams", "requirements.txt"):
        payload.extend(f"{relative}\n".encode())
        payload.extend(hashlib.sha256((context / relative).read_bytes()).hexdigest().encode())
        payload.extend(b"\n")
    return hashlib.sha256(payload).hexdigest()


def _fake_launcher_environment(
    tmp_path: Path,
) -> tuple[dict[str, str], Path, Path, Path]:
    root = tmp_path / "pams-root"
    project = root / "repo"
    docker_context = project / "docker" / "server"
    docker_context.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    for relative in ("Dockerfile", "constraints.txt", "pams", "requirements.txt"):
        shutil.copy2(SERVER_DOCKER / relative, docker_context / relative)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    revision = "a" * 40
    image_id = "sha256:" + "b" * 64
    fingerprint = _environment_fingerprint(docker_context)
    started = tmp_path / "docker-started"
    arguments = tmp_path / "docker-arguments"
    _write_executable(
        fake_bin / "git",
        f"""#!/usr/bin/env bash
set -Eeuo pipefail
case "$*" in
  *"rev-parse HEAD"*) printf '%s\\n' "{revision}" ;;
  *"status --porcelain=v1 --untracked-files=all"*) ;;
  *) exit 2 ;;
esac
""",
    )
    _write_executable(
        fake_bin / "docker",
        """#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "${1:-}" == "image" && "${2:-}" == "inspect" ]]; then
  format=""
  while (($#)); do
    if [[ "$1" == "--format" ]]; then
      format="$2"
      break
    fi
    shift
  done
  case "${format}" in
    "") ;;
    "{{.Id}}") printf '%s\\n' "${PAMS_TEST_IMAGE_ID}" ;;
    *"org.opencontainers.image.revision"*) printf '%s\\n' "${PAMS_TEST_REVISION}" ;;
    *"org.opencontainers.image.pams.environment-sha256"*)
      printf '%s\\n' "${PAMS_TEST_ENVIRONMENT_SHA256}"
      ;;
    *"org.opencontainers.image.pams.apt-snapshot"*)
      printf '%s\\n' "${PAMS_TEST_APT_SNAPSHOT}"
      ;;
    *) exit 2 ;;
  esac
  exit 0
fi
if [[ "${1:-}" == "run" ]]; then
  printf '%s\\n' "$@" >"${PAMS_TEST_DOCKER_ARGUMENTS}"
  : >"${PAMS_TEST_DOCKER_STARTED}"
  sleep "${PAMS_TEST_DOCKER_HOLD_SECONDS:-0}"
  exit 0
fi
exit 2
""",
    )

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "PAMS_EXPECTED_ROOT": str(root),
            "PAMS_PROJECT_DIR": str(project),
            "PAMS_IMAGE": "fixture:image",
            "PAMS_GPU_DEVICE": "0",
            "PAMS_TEST_APT_SNAPSHOT": APT_SNAPSHOT,
            "PAMS_TEST_DOCKER_ARGUMENTS": str(arguments),
            "PAMS_TEST_DOCKER_STARTED": str(started),
            "PAMS_TEST_ENVIRONMENT_SHA256": fingerprint,
            "PAMS_TEST_IMAGE_ID": image_id,
            "PAMS_TEST_REVISION": revision,
        }
    )
    return environment, root, started, arguments


def _bash_with_flock() -> str | None:
    if os.name == "nt":
        return None
    bash = shutil.which("bash")
    if bash is None or shutil.which("flock") is None:
        return None
    return bash


def test_server_dockerfile_pins_base_and_runs_as_non_root() -> None:
    dockerfile = (SERVER_DOCKER / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel@" + BASE_DIGEST in dockerfile
    assert "FROM ${BASE_IMAGE}" in dockerfile
    assert "USER ${PAMS_UID}:${PAMS_GID}" in dockerfile
    assert "python -m pip check" in dockerfile
    assert "python -m pip uninstall --yes \\\n        ninja" in dockerfile
    assert "COPY --chmod=0755 pams /usr/local/bin/pams" in dockerfile
    assert (
        'org.opencontainers.image.source="https://github.com/'
        'appleweiping/pams-rac-reproduction"'
    ) in dockerfile


def test_server_dockerfile_uses_one_configurable_jammy_mirror() -> None:
    dockerfile = (SERVER_DOCKER / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG APT_MIRROR=http://archive.ubuntu.com/ubuntu" in dockerfile
    assert f"ARG APT_SNAPSHOT={APT_SNAPSHOT}" in dockerfile
    assert "rm -f /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources" in dockerfile
    for suite in ("jammy", "jammy-updates", "jammy-security", "jammy-backports"):
        assert (
            f'"deb [snapshot=${{APT_SNAPSHOT}}] ${{apt_mirror}} {suite} '
            'main restricted universe multiverse"'
        ) in dockerfile
    for package in (
        "ca-certificates=20260601~22.04.1",
        "ffmpeg=7:4.4.2-0ubuntu0.22.04.1",
        "git=1:2.34.1-1ubuntu1.17",
        "libgl1=1.4.0-1",
        "libglib2.0-0=2.72.4-0ubuntu2.9",
        "libportaudio2=19.6.0-1.1",
        "libsm6=2:1.2.3-1build2",
        "libxext6=2:1.3.4-1build1",
        "libxrender1=1:0.9.10-1build4",
    ):
        assert package in dockerfile
    assert 'apt-get update --snapshot "${APT_SNAPSHOT}"' in dockerfile


def test_dynamic_pams_wrapper_uses_the_mounted_source() -> None:
    wrapper = (SERVER_DOCKER / "pams").read_text(encoding="utf-8")
    assert 'exec python -m pams "$@"' in wrapper
    assert "pip install" not in wrapper


def test_pose_stack_has_one_exact_opencv_distribution() -> None:
    requirements = _non_comment_requirements(SERVER_DOCKER / "requirements.txt")
    constraints = _non_comment_requirements(SERVER_DOCKER / "constraints.txt")
    combined = requirements | constraints

    assert "numpy==1.26.4" in combined
    assert "mediapipe==0.10.14" in combined
    assert "opencv-contrib-python==4.10.0.84" in combined
    assert "protobuf==4.25.3" in combined
    assert "ninja==1.11.1.4" in combined
    assert not any(
        dependency.startswith(
            (
                "opencv-python==",
                "opencv-python-headless==",
                "opencv-contrib-python-headless==",
            )
        )
        for dependency in combined
    )
    assert all("==" in dependency for dependency in combined)


def test_gpu1_launcher_is_scoped_and_supports_sealed_mode() -> None:
    launcher = (SERVER_SCRIPTS / "run_gpu1.sh").read_text(encoding="utf-8")
    assert 'EXPECTED_ROOT="${PAMS_EXPECTED_ROOT:-}"' in launcher
    assert "PAMS_EXPECTED_ROOT is required" in launcher
    assert "PAMS_EXPECTED_ROOT must be an absolute path" in launcher
    assert "resolved_expected_root=" in launcher
    assert 'GPU_DEVICE="${PAMS_GPU_DEVICE:-1}"' in launcher
    assert '--network "${NETWORK_MODE}"' in launcher
    assert 'NETWORK_MODE="none"' in launcher
    assert 'if [[ "${RUN_MODE}" == "sealed" ]]' in launcher
    assert "HF_HUB_OFFLINE=1" in launcher
    assert "TRANSFORMERS_OFFLINE=1" in launcher
    assert "persistent directory escapes" in launcher
    assert "dst=/workspace,readonly" in launcher
    assert "dst=/pams/data,readonly" in launcher
    assert "RUFF_CACHE_DIR=/pams/cache/ruff" in launcher
    assert "PYTEST_ADDOPTS=-p no:cacheprovider" in launcher
    assert "org.opencontainers.image.pams.environment-sha256" in launcher
    assert "container image revision does not match" in launcher
    assert "formal execution requires a clean Git worktree" in launcher
    assert "PAMS_CONTAINER_IMAGE_ID" in launcher
    assert "PAMS_CONTAINER_ENVIRONMENT_SHA256" in launcher
    assert "PAMS_CONTAINER_SOURCE_REVISION" in launcher
    assert f'FROZEN_APT_SNAPSHOT="{APT_SNAPSHOT}"' in launcher
    assert "container image apt snapshot does not match" in launcher
    assert "--wait-for-idle-gpu" in launcher
    assert 'gpu_lock_directory="${resolved_root}/.pams-gpu-locks"' in launcher
    assert 'gpu_lock_path="${gpu_lock_directory}/gpu${GPU_DEVICE}.lock"' in launcher
    assert 'lock_directory_mode="$(stat -c \'%a\'' in launcher
    assert '[[ -L "${gpu_lock_directory}"' in launcher
    assert '[[ -L "${gpu_lock_path}"' in launcher
    assert "flock -n 9" in launcher
    assert "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu" in launcher
    assert "GPU_STABLE_SAMPLES" in launcher
    assert "exit 74" in launcher
    assert "exit 75" in launcher
    assert 'exec docker "${docker_arguments[@]}" "${image_id}"' in launcher
    assert "/var/run/docker.sock" not in launcher
    assert "/media/" not in launcher


@pytest.mark.skipif(_bash_with_flock() is None, reason="requires POSIX bash and flock")
def test_gpu1_launcher_serializes_concurrent_processes_with_host_only_lock(
    tmp_path: Path,
) -> None:
    bash = _bash_with_flock()
    assert bash is not None
    environment, root, started, arguments = _fake_launcher_environment(tmp_path)
    environment["PAMS_TEST_DOCKER_HOLD_SECONDS"] = "2"
    command = [bash, str(SERVER_SCRIPTS / "run_gpu1.sh"), "--", "true"]

    first = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not started.exists() and first.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert started.exists(), first.communicate(timeout=1)

        second = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert second.returncode == 74, second.stderr
        assert "another PAMS process holds the project lock for GPU 0" in second.stderr
        assert first.wait(timeout=10) == 0
    finally:
        if first.poll() is None:
            first.terminate()
            first.wait(timeout=5)

    lock_directory = root / ".pams-gpu-locks"
    lock_file = lock_directory / "gpu0.lock"
    assert stat.S_IMODE(lock_directory.stat().st_mode) == 0o700
    assert lock_directory.stat().st_uid == os.getuid()
    assert lock_file.is_file()
    assert stat.S_IMODE(lock_file.stat().st_mode) == 0o600
    assert not lock_directory.is_symlink()
    assert not lock_file.is_symlink()
    docker_arguments = arguments.read_text(encoding="utf-8")
    assert str(lock_directory) not in docker_arguments
    assert "dst=/pams/tmp" in docker_arguments


@pytest.mark.skipif(_bash_with_flock() is None, reason="requires POSIX bash and flock")
@pytest.mark.parametrize("symlink_target", ["directory", "file"])
def test_gpu1_launcher_rejects_lock_symlinks(
    tmp_path: Path,
    symlink_target: str,
) -> None:
    bash = _bash_with_flock()
    assert bash is not None
    environment, root, started, _ = _fake_launcher_environment(tmp_path)
    lock_directory = root / ".pams-gpu-locks"
    symlink_destination = tmp_path / "symlink-destination"
    if symlink_target == "directory":
        symlink_destination.mkdir()
        lock_directory.symlink_to(symlink_destination, target_is_directory=True)
    else:
        lock_directory.mkdir(mode=0o700)
        symlink_destination.write_text("", encoding="utf-8")
        (lock_directory / "gpu0.lock").symlink_to(symlink_destination)

    completed = subprocess.run(
        [bash, str(SERVER_SCRIPTS / "run_gpu1.sh"), "--", "true"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 2
    assert "symbolic-link GPU lock" in completed.stderr
    assert not started.exists()


def test_builder_supports_pinned_local_or_mirrored_base_image() -> None:
    builder = (SERVER_SCRIPTS / "build_image.sh").read_text(encoding="utf-8")
    assert "PAMS_BASE_IMAGE" in builder
    assert "PAMS_BASE_IMAGE must be an immutable repository reference" in builder
    assert 'FROZEN_BASE_DIGEST="sha256:14611869895df612' in builder
    assert "may change only the repository prefix" in builder
    assert '--build-arg "BASE_IMAGE=${BASE_IMAGE}"' in builder
    assert "PAMS_PULL_BASE=0 requires the base image to exist locally" in builder
    assert "docker image inspect" in builder
    assert "--pull=false" in builder
    assert 'DEFAULT_APT_MIRROR="http://archive.ubuntu.com/ubuntu"' in builder
    assert 'APT_MIRROR="${PAMS_APT_MIRROR:-${DEFAULT_APT_MIRROR}}"' in builder
    assert "PAMS_APT_MIRROR must be an http(s) URL" in builder
    assert "must not contain URL userinfo or credentials" in builder
    assert "APT mirror: configured (value redacted)" in builder
    assert 'echo "APT mirror: ${APT_MIRROR}"' not in builder
    assert '--build-arg "APT_MIRROR=${APT_MIRROR}"' in builder
    assert f'FROZEN_APT_SNAPSHOT="{APT_SNAPSHOT}"' in builder
    assert "PAMS_APT_SNAPSHOT must remain" in builder
    assert '--build-arg "APT_SNAPSHOT=${APT_SNAPSHOT}"' in builder
    assert "org.opencontainers.image.pams.apt-snapshot" in builder
    assert 'CLEAR_PROXY="${PAMS_CLEAR_PROXY:-0}"' in builder
    assert "PAMS_CLEAR_PROXY must be 0 or 1" in builder
    assert 'if [[ "${CLEAR_PROXY}" == "1" ]]' in builder
    assert '"${proxy_arguments[@]}"' in builder
    assert 'IMAGE_UID="${PAMS_IMAGE_UID:-1000}"' in builder
    assert 'IMAGE_GID="${PAMS_IMAGE_GID:-1000}"' in builder
    assert '--build-arg "PAMS_UID=${IMAGE_UID}"' in builder
    assert '--build-arg "PAMS_GID=${IMAGE_GID}"' in builder
    assert "the formal image must be built from a clean Git worktree" in builder
    assert "org.opencontainers.image.pams.environment-sha256" in builder
    for proxy_variable in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        assert f'--build-arg "{proxy_variable}="' in builder


def test_environment_fingerprint_has_a_fixed_dependency_input_order() -> None:
    helper = ENVIRONMENT_FINGERPRINT.read_text(encoding="utf-8")
    assert "pams_environment_fingerprint()" in helper
    assert "Dockerfile constraints.txt pams requirements.txt" in helper
    assert "sha256sum" in helper
