"""Deterministic synthetic periodic skeletons for protocol-free testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from pams.types import PoseSequence

SpeedProfile = Literal["constant", "linear", "sinusoidal"]


def _readonly(value: NDArray[Any], dtype: np.dtype[Any]) -> NDArray[Any]:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(contiguous.tobytes(), dtype=dtype).reshape(contiguous.shape)


@dataclass(frozen=True, slots=True)
class SyntheticSpec:
    """Configuration for one auditable synthetic sequence."""

    video_id: str = "synthetic"
    frames: int = 256
    fps: float = 30.0
    count: float = 8.0
    speed_profile: SpeedProfile = "constant"
    speed_range: tuple[float, float] = (1.0, 1.0)
    pause_ranges: tuple[tuple[float, float], ...] = ()
    noise_std: float = 0.0
    occlusion_time_fraction: float = 0.0
    occluded_joint_fraction: float = 0.3
    missing_frame_fraction: float = 0.0
    rotation_degrees: float = 0.0
    scale: float = 1.0
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    harmonics: tuple[float, ...] = (1.0,)
    seed: int = 2026

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("video_id must be non-empty")
        if self.frames < 2:
            raise ValueError("frames must be at least two")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        if not np.isfinite(self.count) or not 2 <= self.count <= 40:
            raise ValueError("count must be finite and in the stress-test range [2, 40]")
        if self.speed_profile not in ("constant", "linear", "sinusoidal"):
            raise ValueError("unknown speed_profile")
        low, high = self.speed_range
        if not np.isfinite([low, high]).all() or low <= 0 or high <= 0:
            raise ValueError("speed_range must contain two positive endpoints")
        for start, stop in self.pause_ranges:
            if not 0 <= start < stop <= 1:
                raise ValueError("pause ranges must satisfy 0 <= start < stop <= 1")
        for name, value in (
            ("noise_std", self.noise_std),
            ("occlusion_time_fraction", self.occlusion_time_fraction),
            ("occluded_joint_fraction", self.occluded_joint_fraction),
            ("missing_frame_fraction", self.missing_frame_fraction),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if not 0 <= self.occlusion_time_fraction <= 1:
            raise ValueError("occlusion_time_fraction must be in [0, 1]")
        if not 0 <= self.occluded_joint_fraction <= 1:
            raise ValueError("occluded_joint_fraction must be in [0, 1]")
        if not 0 <= self.missing_frame_fraction < 1:
            raise ValueError("missing_frame_fraction must be in [0, 1)")
        if not np.isfinite(self.rotation_degrees):
            raise ValueError("rotation_degrees must be finite")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")
        if len(self.translation) != 3 or not np.isfinite(self.translation).all():
            raise ValueError("translation must contain three finite values")
        if not self.harmonics or not np.isfinite(self.harmonics).all():
            raise ValueError("harmonics must be a non-empty finite tuple")
        if self.harmonics[0] == 0 or all(value == 0 for value in self.harmonics):
            raise ValueError("harmonics must contain a non-zero fundamental")


@dataclass(frozen=True, slots=True, eq=False)
class SyntheticSample:
    """Sequence plus hidden generation truth for synthetic-only assertions."""

    sequence: PoseSequence
    target_count: float
    phase_radians: NDArray[np.float64]
    joint_visibility: NDArray[np.bool_]

    def __post_init__(self) -> None:
        phase = np.asarray(self.phase_radians, dtype=np.float64)
        visibility = np.asarray(self.joint_visibility, dtype=np.bool_)
        if phase.shape != (self.sequence.num_frames,):
            raise ValueError("phase_radians must have shape [frames]")
        if visibility.shape != (self.sequence.num_frames, 33):
            raise ValueError("joint_visibility must have shape [frames, 33]")
        if not np.isfinite(phase).all() or np.any(np.diff(phase) < -1e-12):
            raise ValueError("phase_radians must be finite and non-decreasing")
        object.__setattr__(
            self,
            "phase_radians",
            _readonly(phase, np.dtype(np.float64)),
        )
        object.__setattr__(
            self,
            "joint_visibility",
            _readonly(visibility, np.dtype(np.bool_)),
        )


def _base_skeleton() -> NDArray[np.float64]:
    """Create a deterministic MediaPipe-sized, approximately human pose."""

    pose = np.zeros((33, 3), dtype=np.float64)
    # Face (0-10), shoulders/elbows/wrists (11-22), hips/legs/feet (23-32).
    pose[:11, 0] = np.linspace(-0.10, 0.10, 11)
    pose[:11, 1] = 0.90 + 0.03 * np.cos(np.linspace(0, 2 * np.pi, 11))
    pose[11] = (-0.22, 0.72, 0.0)
    pose[12] = (0.22, 0.72, 0.0)
    pose[13] = (-0.36, 0.52, 0.0)
    pose[14] = (0.36, 0.52, 0.0)
    pose[15] = (-0.45, 0.32, 0.0)
    pose[16] = (0.45, 0.32, 0.0)
    pose[17:23] = np.asarray(
        [
            (-0.48, 0.30, 0.0),
            (0.48, 0.30, 0.0),
            (-0.46, 0.28, 0.0),
            (0.46, 0.28, 0.0),
            (-0.43, 0.29, 0.0),
            (0.43, 0.29, 0.0),
        ]
    )
    pose[23] = (-0.16, 0.32, 0.0)
    pose[24] = (0.16, 0.32, 0.0)
    pose[25] = (-0.18, 0.02, 0.0)
    pose[26] = (0.18, 0.02, 0.0)
    pose[27] = (-0.19, -0.28, 0.0)
    pose[28] = (0.19, -0.28, 0.0)
    pose[29] = (-0.22, -0.31, 0.0)
    pose[30] = (0.22, -0.31, 0.0)
    pose[31] = (-0.13, -0.34, 0.08)
    pose[32] = (0.13, -0.34, 0.08)
    return pose


def _phase_from_spec(spec: SyntheticSpec) -> NDArray[np.float64]:
    time = np.linspace(0.0, 1.0, spec.frames)
    low, high = spec.speed_range
    if spec.speed_profile == "constant":
        speed = np.full(spec.frames, (low + high) / 2.0)
    elif spec.speed_profile == "linear":
        speed = np.linspace(low, high, spec.frames)
    else:
        midpoint = (low + high) / 2.0
        amplitude = (high - low) / 2.0
        speed = midpoint + amplitude * np.sin(2.0 * np.pi * time - np.pi / 2.0)
    for start, stop in spec.pause_ranges:
        speed[(time >= start) & (time <= stop)] = 0.0

    increments = 0.5 * (speed[:-1] + speed[1:])
    cumulative = np.concatenate((np.asarray([0.0]), np.cumsum(increments)))
    if cumulative[-1] <= 0:
        raise ValueError("pause ranges leave no moving interval")
    return cumulative / cumulative[-1] * (2.0 * np.pi * spec.count)


def generate_synthetic_sample(spec: SyntheticSpec | None = None) -> SyntheticSample:
    """Generate a pose sequence with exact cycle count and controlled corruptions."""

    if spec is None:
        spec = SyntheticSpec()
    rng = np.random.default_rng(spec.seed)
    phase = _phase_from_spec(spec)
    base = _base_skeleton()
    joint_phase = np.linspace(0.0, 2.0 * np.pi, 33, endpoint=False)
    joint_amplitude = np.linspace(0.025, 0.14, 33)
    wave = np.zeros((spec.frames, 33), dtype=np.float64)
    for harmonic, coefficient in enumerate(spec.harmonics, start=1):
        wave += coefficient / harmonic * np.sin(harmonic * phase[:, None] + joint_phase[None, :])

    xyz = np.repeat(base[None, :, :], spec.frames, axis=0)
    xyz[:, :, 0] += joint_amplitude[None, :] * wave
    xyz[:, :, 1] += 0.75 * joint_amplitude[None, :] * np.cos(phase[:, None] + joint_phase[None, :])
    xyz[:, :, 2] += 0.40 * joint_amplitude[None, :] * np.sin(phase[:, None] - joint_phase[None, :])
    if spec.noise_std:
        xyz += rng.normal(0.0, spec.noise_std, size=xyz.shape)

    angle = np.deg2rad(spec.rotation_degrees)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    xyz = (xyz @ rotation.T) * spec.scale
    xyz += np.asarray(spec.translation, dtype=np.float64)[None, None, :]

    visibility = np.ones((spec.frames, 33), dtype=np.bool_)
    occluded_frames = int(round(spec.frames * spec.occlusion_time_fraction))
    occluded_joints = int(round(33 * spec.occluded_joint_fraction))
    if occluded_frames and occluded_joints:
        start = int(rng.integers(0, spec.frames - occluded_frames + 1))
        joints = np.sort(rng.choice(33, size=occluded_joints, replace=False))
        frame_index = np.arange(start, start + occluded_frames)
        xyz[np.ix_(frame_index, joints)] = 0.0
        visibility[np.ix_(frame_index, joints)] = False

    valid_mask = np.ones(spec.frames, dtype=np.bool_)
    missing_frames = int(round(spec.frames * spec.missing_frame_fraction))
    if missing_frames:
        missing = np.sort(rng.choice(spec.frames, size=missing_frames, replace=False))
        valid_mask[missing] = False
        visibility[missing] = False
        xyz[missing] = 0.0

    sequence = PoseSequence(
        video_id=spec.video_id,
        fps=spec.fps,
        xyz=xyz.astype(np.float32),
        valid_mask=valid_mask,
    )
    return SyntheticSample(
        sequence=sequence,
        target_count=spec.count,
        phase_radians=phase,
        joint_visibility=visibility,
    )


def generate_periodic_skeleton(
    spec: SyntheticSpec | None = None,
) -> PoseSequence:
    """Convenience wrapper returning only the public pose sequence."""

    return generate_synthetic_sample(spec).sequence


def synthetic_stress_suite(
    *, count: float = 8.0, frames: int = 256, seed: int = 2026
) -> dict[str, SyntheticSample]:
    """Return the frozen robustness cases used by CPU integration tests."""

    common: dict[str, Any] = {"count": count, "frames": frames, "seed": seed}
    specs = {
        "clean": SyntheticSpec(video_id="clean", **common),
        "variable_speed": SyntheticSpec(
            video_id="variable_speed",
            speed_profile="linear",
            speed_range=(0.5, 2.0),
            **common,
        ),
        "pause": SyntheticSpec(video_id="pause", pause_ranges=((0.40, 0.55),), **common),
        "noise": SyntheticSpec(video_id="noise", noise_std=0.02, **common),
        "occlusion": SyntheticSpec(
            video_id="occlusion",
            occlusion_time_fraction=0.20,
            occluded_joint_fraction=0.30,
            **common,
        ),
        "rotation_negative": SyntheticSpec(
            video_id="rotation_negative",
            rotation_degrees=-15.0,
            **common,
        ),
        "rotation_positive": SyntheticSpec(
            video_id="rotation_positive",
            rotation_degrees=15.0,
            **common,
        ),
        "scale_low": SyntheticSpec(video_id="scale_low", scale=0.85, **common),
        "scale_high": SyntheticSpec(video_id="scale_high", scale=1.15, **common),
        "translation_negative": SyntheticSpec(
            video_id="translation_negative",
            translation=(-0.1, -0.1, -0.1),
            **common,
        ),
        "translation_positive": SyntheticSpec(
            video_id="translation_positive",
            translation=(0.1, 0.1, 0.1),
            **common,
        ),
        "multiharmonic": SyntheticSpec(
            video_id="multiharmonic", harmonics=(1.0, 0.5, 0.25), **common
        ),
    }
    return {name: generate_synthetic_sample(spec) for name, spec in specs.items()}


def generate_count_sweep(
    *,
    minimum: int = 2,
    maximum: int = 40,
    frames: int = 256,
    seed: int = 2026,
) -> tuple[SyntheticSample, ...]:
    """Generate the required inclusive count 2--40 synthetic sweep."""

    if minimum < 2 or maximum > 40 or maximum < minimum:
        raise ValueError("count sweep bounds must satisfy 2 <= minimum <= maximum <= 40")
    return tuple(
        generate_synthetic_sample(
            SyntheticSpec(
                video_id=f"count-{count:02d}",
                frames=frames,
                count=float(count),
                seed=seed + count,
            )
        )
        for count in range(minimum, maximum + 1)
    )
