from __future__ import annotations

from pathlib import Path

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


def test_server_dockerfile_pins_base_and_runs_as_non_root() -> None:
    dockerfile = (SERVER_DOCKER / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel@" + BASE_DIGEST in dockerfile
    assert "FROM ${BASE_IMAGE}" in dockerfile
    assert "USER ${PAMS_UID}:${PAMS_GID}" in dockerfile
    assert "python -m pip check" in dockerfile
    assert "python -m pip uninstall --yes \\\n        ninja" in dockerfile
    assert "COPY --chmod=0755 pams /usr/local/bin/pams" in dockerfile


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
    assert 'exec docker "${docker_arguments[@]}" "${image_id}"' in launcher
    assert "/var/run/docker.sock" not in launcher
    assert "/media/" not in launcher


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
