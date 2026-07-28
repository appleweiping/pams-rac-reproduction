from pathlib import Path

import pytest

from pams import reproducibility


def test_durable_mkdir_fsyncs_each_new_parent_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synced: list[Path] = []
    monkeypatch.setattr(
        reproducibility,
        "fsync_directory",
        lambda path: synced.append(Path(path)),
    )

    target = tmp_path / "one" / "two" / "three"
    assert reproducibility.durable_mkdir(target) == target

    assert target.is_dir()
    assert synced == [tmp_path, tmp_path / "one", tmp_path / "one" / "two"]
    synced.clear()
    reproducibility.durable_mkdir(target)
    assert synced == []


def test_hardware_fingerprint_declares_driver_inventory_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reproducibility.torch.cuda, "is_available", lambda: False)

    fingerprint = reproducibility.hardware_fingerprint()

    assert fingerprint["nvidia_driver"] == {
        "status": "no_cuda_device",
        "versions": [],
    }
