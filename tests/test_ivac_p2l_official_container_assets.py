from __future__ import annotations

import json
import re
from pathlib import Path

REPOSITORY = Path(__file__).parents[1]
CONTEXT = REPOSITORY / "docker/ivac-p2l-official"
RESULT = (
    REPOSITORY
    / "results/dev-negative/ivac_p2l_official_dev84_retrospective.json"
)
BASE_DIGEST = (
    "sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014"
)
SOURCE_COMMIT = "0b1149e6958268ca5131d60ff66498c6e07f3b79"


def test_image_is_pinned_nonroot_and_contains_no_third_party_assets() -> None:
    dockerfile = (CONTEXT / "Dockerfile").read_text(encoding="utf-8")
    requirements = (CONTEXT / "requirements.in").read_text(encoding="utf-8")
    assert (
        "ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel@"
        f"{BASE_DIGEST}"
    ) in dockerfile
    assert SOURCE_COMMIT in dockerfile
    assert "USER ${IVAC_UID}:${IVAC_GID}" in dockerfile
    assert "PYTHONPATH=/workspace/src:/opt/ivac-source" in dockerfile
    for requirement in (
        "einops==0.3.2",
        "kornia==0.5.11",
        "mmcv==1.4.0",
        "numpy==1.26.4",
        "opencv-python-headless==4.10.0.84",
        "timm==0.4.12",
    ):
        assert requirement in requirements
    assert not any(
        path.name.lower().endswith(
            (".pt", ".pth", ".tar.gz", ".whl", ".avi", ".mp4", ".npz")
        )
        for path in CONTEXT.rglob("*")
        if path.is_file()
    )
    lowered = dockerfile.lower()
    assert "git clone" not in lowered
    assert "wget " not in lowered
    assert "curl " not in lowered


def test_environment_and_current_result_state_claim_boundaries() -> None:
    note = (CONTEXT / "ENVIRONMENT.md").read_text(encoding="utf-8")
    assert "not evidence of original-protocol parity" in note
    assert "contains no IVAC-P2L source" in note
    assert "checkpoint or the external Swin backbone" in note
    assert "protocol-ambiguous" in note
    assert "two separate processes" in note
    assert (
        "ea96abe0e7d17cf4720343e5867256aacd5209aca738087ec92f62b9fe31ff51"
        in note
    )

    raw = RESULT.read_text(encoding="utf-8")
    result = json.loads(raw)
    assert result["artifact_type"] == "sanitized_retrospective_baseline_result"
    assert result["status"] == "current_runner_rerun_complete"
    compatibility = result["current_code_compatibility"]
    assert compatibility["legacy_artifact_generated_by_current_runner"] is False
    assert compatibility["tested_with_current_loader"] is True
    assert compatibility["raw_predictions_identical_to_legacy"] is True
    assert compatibility["compatible_with_legacy_aggregate_metrics"] is False
    assert compatibility["requires_rerun"] is False
    assert compatibility["rounding_delta"]["legacy_rounded_count"] == 8
    assert compatibility["rounding_delta"]["current_rounded_count"] == 9
    assert result["current_metrics"]["nmae_rounded"] == 0.666136794354159
    assert result["current_artifact_bindings"]["runner_source_git_sha"] == (
        "fb8028a1681cdc71313455c1c0b2f4a7946acb6b"
    )
    assert result["claim_boundary"]["eligible_for_original_paper_table"] is False
    assert result["sealed_test_status"] == "untouched"
    assert "predictions" not in result and "per_video" not in result
    assert "server_endpoint" not in result and "server_path" not in result
    assert re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", raw) is None
    assert re.search(r'(?m)(?:^|[": ])/(?:media|home|Users)/', raw) is None
