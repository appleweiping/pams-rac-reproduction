import io
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from pams.data import UCFRepManifest, UCFRepRecord
from pams.ucfrep import _annotation_count, materialize_standard_split_ids


class _Label:
    def __init__(self, boundaries: np.ndarray) -> None:
        self.temporal_bound = boundaries


def test_annotation_count_matches_official_boundary_definition(tmp_path: Path) -> None:
    path = tmp_path / "annotation.mat"
    label = {
        "duration": np.asarray([[100]]),
        "start_frame": np.asarray([[1]]),
        "end_frame": np.asarray([[100]]),
        "temporal_bound_num": np.asarray([[5]]),
        "temporal_bound": np.asarray([[1], [25], [50], [75], [100]]),
        "offset_next_estimate": np.zeros((1, 100)),
        "offset_pre_estimate": np.zeros((1, 100)),
    }
    savemat(path, {"label": label})
    assert _annotation_count(path.read_bytes()) == 4


def test_annotation_count_rejects_missing_boundaries() -> None:
    buffer = io.BytesIO()
    savemat(buffer, {"other": np.asarray([1])})
    with pytest.raises(ValueError, match="label"):
        _annotation_count(buffer.getvalue())


def test_materialize_standard_split_ids_is_label_free(tmp_path: Path) -> None:
    records = tuple(
        UCFRepRecord(
            video_id=f"v_Action_g{index % 20 + 1:02d}_c{index:02d}",
            video_path=f"videos/train-{index}.avi",
            split="train",
            action=f"action-{index % 7}",
            count=index % 45 + 1,
        )
        for index in range(421)
    ) + tuple(
        UCFRepRecord(
            video_id=f"v_Action_g{index % 5 + 21:02d}_c{index:02d}",
            video_path=f"videos/test-{index}.avi",
            split="test",
            action=f"action-{index % 7}",
            count=index % 45 + 1,
        )
        for index in range(105)
    )
    manifest = UCFRepManifest("ucfrep_526", records)
    outputs = materialize_standard_split_ids(manifest, tmp_path)
    assert len((tmp_path / "ucfrep_526_train_421.txt").read_text().splitlines()) == 421
    assert len((tmp_path / "ucfrep_526_dev_84.txt").read_text().splitlines()) == 84
    assert set(outputs) == {
        "ucfrep_526_train_421.txt",
        "ucfrep_526_test_105.txt",
        "ucfrep_526_train_337.txt",
        "ucfrep_526_dev_84.txt",
    }
    assert "action-" not in (tmp_path / "ucfrep_526_train_337.txt").read_text()
    for output in outputs.values():
        assert b"\r\n" not in output.read_bytes()
