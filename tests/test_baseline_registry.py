from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from pams.baselines import (
    BaselineAdapter,
    BaselineStatus,
    BaselineUnavailableError,
    ProtocolRecord,
    create_baseline,
    get_baseline_spec,
    list_baselines,
)
from pams.baselines.countllm_lite import (
    countllm_lite_spec,
    inspect_countllm_lite_resources,
)
from pams.types import CountResult, PoseSequence

EXPECTED_PAPER_BASELINES = {
    "repnet",
    "transrac",
    "escounts",
    "ivac-p2l",
    "poserac-v1",
    "poserac-iconip24",
    "gmfl",
    "jtsps-count-only",
    "spkdb",
    "bigc",
    "countllm-lite",
}


def test_all_requested_paper_baselines_have_explicit_records() -> None:
    specs = list_baselines(include_references=False)
    assert {spec.key for spec in specs} == EXPECTED_PAPER_BASELINES
    assert all(spec.protocols for spec in specs)
    assert all(spec.status is not BaselineStatus.READY for spec in specs)


def test_every_registration_has_matching_versioned_yaml() -> None:
    repository = Path(__file__).parents[1]
    for spec in list_baselines():
        assert spec.config_path is not None
        config_path = repository / spec.config_path
        assert config_path.is_file(), spec.key
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        assert payload["schema_version"] == 1
        assert payload["key"] == spec.key


def test_unimplemented_method_never_falls_back_to_proxy() -> None:
    spec = get_baseline_spec("RepNet")
    assert spec.status is BaselineStatus.BLOCKED_UNIMPLEMENTED
    assert "Clean-room" in (spec.blocked_reason or "")

    with pytest.raises(BaselineUnavailableError, match="repnet"):
        create_baseline("repnet")


def test_protocol_record_forbids_fair_ground_truth_count_oracle() -> None:
    with pytest.raises(ValueError, match="ground-truth count"):
        ProtocolRecord(
            key="invalid_oracle",
            dataset="fixture",
            train_split="train",
            evaluation_split="test",
            modality="pose",
            source_faithful=True,
            fair_comparison=True,
            ground_truth_count_at_inference=True,
        )


def test_spectral_proxy_is_explicit_reference_and_deterministic() -> None:
    frames = 64
    phase = np.arange(frames, dtype=np.float32) * (2.0 * np.pi / 8.0)
    xyz = np.zeros((frames, 33, 3), dtype=np.float32)
    xyz[:, :, 0] = np.sin(phase)[:, None]
    xyz[:, :, 1] = np.cos(phase)[:, None]
    sample = PoseSequence(
        video_id="synthetic-periodic",
        fps=30.0,
        xyz=xyz,
        valid_mask=np.ones(frames, dtype=bool),
    )

    adapter = create_baseline("spectral-proxy")
    assert isinstance(adapter, BaselineAdapter)
    assert not adapter.spec.is_paper_baseline
    assert adapter.spec.status is BaselineStatus.REFERENCE_ONLY

    first = adapter.predict(sample)
    second = adapter.predict(sample)
    assert isinstance(first, CountResult)
    assert first.to_dict() == second.to_dict()
    assert first.period_frames == pytest.approx(8.0)
    assert first.count == 8
    assert 0.0 <= first.confidence <= 1.0


class _FakeCuda:
    def __init__(self, memories_gib: tuple[float, ...]) -> None:
        self._memories_gib = memories_gib

    def is_available(self) -> bool:
        return bool(self._memories_gib)

    def device_count(self) -> int:
        return len(self._memories_gib)

    def get_device_properties(self, index: int) -> SimpleNamespace:
        return SimpleNamespace(
            name=f"fake-gpu-{index}",
            total_memory=int(self._memories_gib[index] * 1024**3),
        )


def test_countllm_lite_resource_gate_blocks_sub_48_gib_gpu() -> None:
    report = inspect_countllm_lite_resources(
        SimpleNamespace(cuda=_FakeCuda((24.0,))),
    )
    assert not report.eligible_for_training
    assert report.maximum_gpu_memory_gib == pytest.approx(24.0)
    spec = countllm_lite_spec(report)
    assert spec.status is BaselineStatus.SMOKE_ONLY
    assert not spec.runnable


def test_countllm_lite_stays_unimplemented_after_hardware_gate_passes() -> None:
    report = inspect_countllm_lite_resources(
        SimpleNamespace(cuda=_FakeCuda((80.0,))),
    )
    assert report.eligible_for_training
    spec = countllm_lite_spec(report)
    assert spec.status is BaselineStatus.BLOCKED_UNIMPLEMENTED
    assert not spec.runnable
    assert "not been implemented" in (spec.blocked_reason or "")
