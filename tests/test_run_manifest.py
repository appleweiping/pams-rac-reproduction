from pathlib import Path

import pytest

from pams.run_manifest import create_run_manifest, write_manifest_exclusive


def test_manifest_is_write_once(tmp_path: Path) -> None:
    manifest = create_run_manifest(
        command=["pams", "train"],
        config_sha256="a" * 64,
        dataset_sha256="b" * 64,
        seed=2026,
        protocol="synthetic",
        cwd=tmp_path,
    )
    path = tmp_path / "manifest.json"
    write_manifest_exclusive(manifest, path)
    assert path.exists()
    with pytest.raises(FileExistsError):
        write_manifest_exclusive(manifest, path)
