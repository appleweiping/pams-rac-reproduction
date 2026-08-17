from __future__ import annotations

import json
from pathlib import Path

REPOSITORY = Path(__file__).parents[1]
CONTEXT = REPOSITORY / "docker/escounts-official"
LEGACY_RESULT = REPOSITORY / "results/dev-negative/escounts_official_dev84_f18fcf1.json"


def test_frozen_container_facts_match_server_observation() -> None:
    dockerfile = (CONTEXT / "Dockerfile").read_text(encoding="utf-8")
    environment = (CONTEXT / "ENVIRONMENT.md").read_text(encoding="utf-8")
    result = json.loads(LEGACY_RESULT.read_text(encoding="utf-8"))
    assert "python=3.8.13" in dockerfile
    assert "pytorch=1.10.0" in dockerfile
    assert "torchvision=0.11.0" in dockerfile
    assert 'startswith("0.11.0")' in dockerfile
    assert "torchvision=0.11.1" not in dockerfile
    assert "torchvision 0.11.0" in environment
    assert result["environment"]["torchvision"] == "0.11.0"


def test_environment_discloses_process_and_rebuild_boundaries() -> None:
    environment = (CONTEXT / "ENVIRONMENT.md").read_text(encoding="utf-8")
    result = json.loads(LEGACY_RESULT.read_text(encoding="utf-8"))
    assert "must **not** be imported in this Python 3.8 image" in environment
    assert "long-lived JSONL worker" in environment
    assert "--repository-root REPOSITORY" in environment
    assert "same clean full Git SHA" in environment
    assert "cross-commit lineage is rejected" in environment
    assert "not a promise" in environment
    assert "ties-to-even" in environment
    assert result["metrics"]["rounding"] == "python_ties_to_even_legacy"
    assert "half-integer" in result["metrics"]["rounding_compatibility"]
