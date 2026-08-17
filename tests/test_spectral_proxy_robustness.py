from __future__ import annotations

from pams.baselines.proxy import SpectralProxyAdapter
from pams.synthetic import SyntheticSpec, generate_synthetic_sample


def test_partial_joint_occlusion_does_not_create_a_low_frequency_count() -> None:
    """Zero-filled missing joints must not dominate the selected pose signal."""

    sample = generate_synthetic_sample(
        SyntheticSpec(
            video_id="missing-joints-regression",
            frames=256,
            count=8.0,
            occlusion_time_fraction=0.2,
            occluded_joint_fraction=0.3,
            seed=2526,
        )
    )

    result = SpectralProxyAdapter().predict(sample.sequence)

    assert result.count == int(sample.target_count)
    assert result.period_frames == 32.0


def test_short_period_count_avoids_integer_period_quantization_error() -> None:
    """The continuous spectral peak should retain a 39-cycle synthetic truth."""

    sample = generate_synthetic_sample(
        SyntheticSpec(
            video_id="short-period-regression",
            frames=256,
            count=39.0,
            seed=2065,
        )
    )

    result = SpectralProxyAdapter().predict(sample.sequence)

    assert abs(result.count - int(sample.target_count)) <= 1
