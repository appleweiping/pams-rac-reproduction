from pathlib import Path

import pytest

from pams import reproducibility


def test_seed_everything_enforces_strict_deterministic_cuda_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deterministic_calls: list[tuple[bool, bool]] = []
    sdpa_calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(reproducibility.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(reproducibility.torch.cuda, "manual_seed_all", lambda _seed: None)
    monkeypatch.setattr(
        reproducibility.torch,
        "use_deterministic_algorithms",
        lambda enabled, *, warn_only: deterministic_calls.append((enabled, warn_only)),
    )
    monkeypatch.setattr(
        reproducibility.torch.backends.cuda,
        "enable_flash_sdp",
        lambda enabled: sdpa_calls.append(("flash", enabled)),
    )
    monkeypatch.setattr(
        reproducibility.torch.backends.cuda,
        "enable_mem_efficient_sdp",
        lambda enabled: sdpa_calls.append(("memory_efficient", enabled)),
    )
    monkeypatch.setattr(
        reproducibility.torch.backends.cuda,
        "enable_math_sdp",
        lambda enabled: sdpa_calls.append(("math", enabled)),
    )
    monkeypatch.setattr(
        reproducibility.torch.backends.cudnn,
        "is_available",
        lambda: False,
    )

    reproducibility.seed_everything(2026)

    assert deterministic_calls == [(True, False)]
    assert sdpa_calls == [
        ("flash", False),
        ("memory_efficient", False),
        ("math", True),
    ]


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
