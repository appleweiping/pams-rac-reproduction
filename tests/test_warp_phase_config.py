from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pams.config import PAMSConfig
from pams.warp_phase.config import load_warp_phase_config

CONFIG_PATH = Path("configs/experiments/warp_phase_pilot_v1.yaml")


def test_frozen_pilot_configuration_loads_with_stable_fingerprint() -> None:
    first = load_warp_phase_config(CONFIG_PATH)
    second = load_warp_phase_config(CONFIG_PATH)
    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64
    assert first.training_authorized is False
    assert first.source.allowed_splits == ("train", "val")
    assert first.model.trainable_parameters == 225_026
    assert first.training.seeds == (20270815, 20270816, 20270817)
    assert first.gates.schema_fixture_bytes == 617


def test_configuration_rejects_unknown_or_changed_frozen_values(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["training"]["updates"] = 19_999
    payload["unexpected"] = True
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_warp_phase_config(path)


def test_new_schema_does_not_change_historical_config_identity() -> None:
    assert PAMSConfig().nonseed_fingerprint == (
        "2c995b374bc8cf97df3568d745dabd94f111aa54224c86ece51468cbe419b47b"
    )


def test_package_initializer_has_no_eager_submodule_imports() -> None:
    source = Path("src/pams/warp_phase/__init__.py").read_text(encoding="utf-8")
    assert "from pams.warp_phase" not in source
    assert "import yaml" not in source
    assert "import pydantic" not in source
