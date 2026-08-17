import numpy as np
import pytest

from pams.synthetic import (
    SyntheticSpec,
    generate_count_sweep,
    generate_synthetic_sample,
    synthetic_stress_suite,
)


def test_synthetic_generation_is_deterministic_and_exact_count() -> None:
    spec = SyntheticSpec(
        frames=128,
        count=12,
        noise_std=0.01,
        speed_profile="linear",
        speed_range=(0.5, 2.0),
        seed=9,
    )
    first = generate_synthetic_sample(spec)
    second = generate_synthetic_sample(spec)
    np.testing.assert_array_equal(first.sequence.xyz, second.sequence.xyz)
    np.testing.assert_array_equal(first.phase_radians, second.phase_radians)
    assert first.phase_radians[-1] / (2 * np.pi) == pytest.approx(12)
    increments = np.diff(first.phase_radians)
    assert increments.max() > 3 * increments.min()


def test_pause_creates_flat_phase_without_changing_total_count() -> None:
    sample = generate_synthetic_sample(
        SyntheticSpec(frames=201, count=7, pause_ranges=((0.4, 0.6),))
    )
    middle = np.diff(sample.phase_radians)[85:115]
    assert np.all(middle == 0)
    assert sample.phase_radians[-1] == pytest.approx(14 * np.pi)


def test_descending_linear_speed_is_supported_and_exact_count() -> None:
    sample = generate_synthetic_sample(
        SyntheticSpec(
            frames=256,
            count=8,
            speed_profile="linear",
            speed_range=(2.0, 0.5),
        )
    )
    increments = np.diff(sample.phase_radians)
    assert increments[0] > increments[-1]
    assert sample.phase_radians[-1] == pytest.approx(16 * np.pi)


def test_stress_suite_contains_required_corruptions() -> None:
    suite = synthetic_stress_suite(frames=100)
    assert set(suite) == {
        "clean",
        "variable_speed",
        "pause",
        "noise",
        "occlusion",
        "rotation_negative",
        "rotation_positive",
        "scale_low",
        "scale_high",
        "translation_negative",
        "translation_positive",
        "multiharmonic",
    }
    occlusion = suite["occlusion"]
    invisible = ~occlusion.joint_visibility
    affected_frames = np.any(invisible, axis=1)
    assert affected_frames.sum() == 20
    assert np.max(invisible.sum(axis=1)) == round(33 * 0.3)
    assert not np.array_equal(suite["clean"].sequence.xyz, suite["noise"].sequence.xyz)
    assert not np.array_equal(
        suite["clean"].sequence.xyz,
        suite["rotation_negative"].sequence.xyz,
    )
    assert not np.array_equal(
        suite["rotation_negative"].sequence.xyz,
        suite["rotation_positive"].sequence.xyz,
    )


def test_missing_frames_are_zero_and_masked() -> None:
    sample = generate_synthetic_sample(SyntheticSpec(frames=100, missing_frame_fraction=0.1))
    assert (~sample.sequence.valid_mask).sum() == 10
    assert np.all(sample.sequence.xyz[~sample.sequence.valid_mask] == 0)
    assert not sample.joint_visibility[~sample.sequence.valid_mask].any()


def test_count_sweep_covers_inclusive_2_to_40() -> None:
    samples = generate_count_sweep(frames=64)
    assert len(samples) == 39
    assert samples[0].target_count == 2
    assert samples[-1].target_count == 40


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": 1},
        {"count": 41},
        {"pause_ranges": ((0.8, 0.2),)},
        {"speed_range": (0.0, 0.5)},
        {"occlusion_time_fraction": 1.1},
        {"harmonics": (0.0,)},
    ],
)
def test_synthetic_spec_rejects_invalid_settings(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SyntheticSpec(**kwargs)  # type: ignore[arg-type]
