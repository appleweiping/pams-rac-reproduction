from __future__ import annotations

import json
from pathlib import Path

REPOSITORY = Path(__file__).parents[1]
CONTEXT = REPOSITORY / "docker/transrac-official"
BASE_DIGEST = (
    "sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014"
)
SOURCE_REVISION = "68bdd4daa60ed7c3174a7f6bf86f6537b6fa0979"


def test_transrac_image_pins_the_audited_modern_compatibility_stack() -> None:
    dockerfile = (CONTEXT / "Dockerfile").read_text(encoding="utf-8")
    requirements = (CONTEXT / "requirements.in").read_text(encoding="utf-8")
    inventory = (CONTEXT / "server-pip-inventory.txt").read_text(encoding="utf-8")

    assert (
        "ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel@"
        f"{BASE_DIGEST}"
    ) in dockerfile
    assert (
        f'org.opencontainers.image.transrac.source-revision="{SOURCE_REVISION}"'
        in dockerfile
    )
    assert "USER ${TRANSRAC_UID}:${TRANSRAC_GID}" in dockerfile
    assert "python -m pip check" in dockerfile
    for requirement in (
        "einops==0.3.2",
        "kornia==0.5.11",
        "mmcv==1.4.0",
        "numpy==1.26.4",
        "opencv-python-headless==4.10.0.84",
        "timm==0.4.12",
    ):
        assert requirement in requirements
    assert "not a hash-locked" in requirements
    assert "torch==2.5.1+cu124" in inventory
    assert "mmcv==1.4.0" in inventory
    assert "not an installable or bit-reproducible lock" in inventory


def test_image_context_contains_no_third_party_source_or_weights() -> None:
    names = {
        path.name.lower()
        for path in CONTEXT.rglob("*")
        if path.is_file()
    }
    assert not any(
        name.endswith((".pt", ".pth", ".tar.gz", ".whl", ".avi", ".mp4"))
        for name in names
    )
    dockerfile = (CONTEXT / "Dockerfile").read_text(encoding="utf-8")
    assert "git clone" not in dockerfile
    assert "wget " not in dockerfile
    assert "curl " not in dockerfile


def test_environment_note_discloses_protocol_and_license_limits() -> None:
    note = (CONTEXT / "ENVIRONMENT.md").read_text(encoding="utf-8")
    assert "does not establish original-protocol parity" in note
    assert "contains no TransRAC source" in note
    assert "Apache-2.0" in note
    assert "Anti-996" in note
    assert "third-party bytes" in note
    assert "ea96abe0e7d17cf4720343e5867256aacd5209aca738087ec92f62b9fe31ff51" in note
    assert "an installable lock" in note
    assert "must not be" in note


def test_published_result_is_compact_path_free_negative_evidence() -> None:
    result_path = (
        REPOSITORY
        / "results/dev-negative/transrac_official_modern_compat_dev84.json"
    )
    raw = result_path.read_text(encoding="utf-8")
    result = json.loads(raw)

    assert result["status"] == "partial_reproduction"
    assert result["split"] == "dev"
    assert result["sample_count"] == 84
    assert result["sealed_test_status"] == "untouched"
    assert result["eligible_for_original_paper_table"] is False
    assert result["metrics"]["nmae_rounded"] == 0.6906921109815201
    assert result["metrics"]["obo_rounded"] == 0.2857142857142857
    assert "predictions" not in result
    assert "per_video" not in result
    assert "8.133.245.52" not in raw
    assert ":33123" not in raw
