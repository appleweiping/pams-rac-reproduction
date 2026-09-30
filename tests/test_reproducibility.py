import hashlib
import json
from pathlib import Path

import pytest

from pams import reproducibility


def _write_source_export_receipt(
    root: Path,
    receipt: Path,
    *,
    revision: str,
) -> str:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        payload = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_revision": revision,
                "root": ".",
                "files": rows,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return hashlib.sha256(receipt.read_bytes()).hexdigest()


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


def test_clean_git_revision_accepts_exact_git_object_free_source_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    (root / "src").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "configs" / "run.yaml").write_text("seed: 2026\n", encoding="utf-8")
    receipt = tmp_path / "source-export.receipt.json"
    revision = "a" * 40
    receipt_sha256 = _write_source_export_receipt(
        root,
        receipt,
        revision=revision,
    )
    monkeypatch.setenv("PAMS_CONTAINER_SOURCE_REVISION", revision)
    monkeypatch.setenv("PAMS_SOURCE_EXPORT_RECEIPT", str(receipt))
    monkeypatch.setenv(
        "PAMS_SOURCE_EXPORT_RECEIPT_SHA256",
        receipt_sha256,
    )

    assert reproducibility.git_revision(root) == revision
    assert reproducibility.clean_git_revision(root) == revision


def test_clean_git_revision_rejects_tampered_or_extra_export_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    source = root / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    receipt = tmp_path / "source-export.receipt.json"
    revision = "b" * 40
    receipt_sha256 = _write_source_export_receipt(
        root,
        receipt,
        revision=revision,
    )
    monkeypatch.setenv("PAMS_CONTAINER_SOURCE_REVISION", revision)
    monkeypatch.setenv("PAMS_SOURCE_EXPORT_RECEIPT", str(receipt))
    monkeypatch.setenv(
        "PAMS_SOURCE_EXPORT_RECEIPT_SHA256",
        receipt_sha256,
    )

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="file mismatch"):
        reproducibility.clean_git_revision(root)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    (root / "unreceipted.txt").write_text("hidden\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="file set mismatch"):
        reproducibility.clean_git_revision(root)


def test_clean_git_revision_rejects_incomplete_or_mismatched_export_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    receipt = tmp_path / "source-export.receipt.json"
    receipt_sha256 = _write_source_export_receipt(
        root,
        receipt,
        revision="c" * 40,
    )
    monkeypatch.setenv("PAMS_SOURCE_EXPORT_RECEIPT", str(receipt))
    with pytest.raises(RuntimeError, match="environment is incomplete"):
        reproducibility.clean_git_revision(root)

    monkeypatch.setenv("PAMS_SOURCE_EXPORT_RECEIPT_SHA256", receipt_sha256)
    monkeypatch.setenv("PAMS_CONTAINER_SOURCE_REVISION", "d" * 40)
    with pytest.raises(RuntimeError, match="does not match"):
        reproducibility.clean_git_revision(root)


def test_hardware_fingerprint_declares_driver_inventory_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reproducibility.torch.cuda, "is_available", lambda: False)

    fingerprint = reproducibility.hardware_fingerprint()

    assert fingerprint["nvidia_driver"] == {
        "status": "no_cuda_device",
        "versions": [],
    }
