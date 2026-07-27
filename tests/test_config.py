from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pams.config import PAMSConfig, load_config


def test_default_config_matches_disclosed_dimensions() -> None:
    config = PAMSConfig()
    assert config.model.input_dim == 33 * 3
    assert config.model.model_dim == 512
    assert len(config.fingerprint) == 64


def test_config_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValidationError, match="input_dim"):
        PAMSConfig(model={"input_dim": 98})


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"unknown": True}), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


def test_repository_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "pams.yaml"
    config = load_config(path)
    assert config.loss.scales == (0.5, 1.0, 1.5)
