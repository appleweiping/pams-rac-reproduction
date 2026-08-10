"""Deterministic, label-free perturbations for cached UCFRep poses."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from pams.types import PoseSequence

StressKind = Literal[
    "clean",
    "temporal_speed",
    "temporal_pause",
    "pose_occlusion",
    "rotation",
    "isotropic_scale",
    "translation",
]


def _exact_keys(value: object, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    result = {str(key): item for key, item in value.items()}
    supplied = set(result)
    if supplied != expected:
        raise ValueError(
            f"{name} fields mismatch; "
            f"missing={sorted(expected - supplied)}, unknown={sorted(supplied - expected)}"
        )
    return result


def _finite(value: object, name: str) -> float:
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _fraction(value: object, name: str, *, include_one: bool = False) -> float:
    result = _finite(value, name)
    upper_ok = result <= 1.0 if include_one else result < 1.0
    if result <= 0.0 or not upper_ok:
        upper = "1" if include_one else "1 (exclusive)"
        raise ValueError(f"{name} must be in (0, {upper}]")
    return result


def _non_empty_numbers(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True, slots=True)
class StressCondition:
    """One expanded condition from the frozen stress configuration."""

    condition_id: str
    kind: StressKind
    parameters: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.condition_id.strip():
            raise ValueError("condition_id must be non-empty")
        json.dumps(self.parameters, sort_keys=True, allow_nan=False)


@dataclass(frozen=True, slots=True)
class StressProtocol:
    """Validated UCFRep portion of ``configs/stress.yaml``."""

    key: str
    seed: int
    config_sha256: str
    conditions: tuple[StressCondition, ...]

    def __post_init__(self) -> None:
        if self.key != "pams_stress_v1":
            raise ValueError("stress key must be pams_stress_v1")
        if self.seed < 0:
            raise ValueError("stress seed must be non-negative")
        if len(self.config_sha256) != 64:
            raise ValueError("config_sha256 must be a SHA-256")
        identifiers = [condition.condition_id for condition in self.conditions]
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("stress condition IDs must be non-empty and unique")
        if identifiers[0] != "clean":
            raise ValueError("clean must be the first stress condition")


@dataclass(frozen=True, slots=True)
class PerturbationAudit:
    """Small per-video description of exactly what was transformed."""

    condition_id: str
    algorithm: str
    parameters: dict[str, Any]
    transformed_pose_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "algorithm": self.algorithm,
            "parameters": self.parameters,
            "transformed_pose_sha256": self.transformed_pose_sha256,
        }


def _condition_suffix(value: float) -> str:
    sign = "neg" if value < 0 else "pos"
    magnitude = format(abs(value), ".12g").replace(".", "p")
    return f"{sign}{magnitude}"


def load_stress_protocol(path: str | Path) -> StressProtocol:
    """Load the stress policy and expand every UCFRep condition."""

    source = Path(path)
    raw_bytes = source.read_bytes()
    try:
        parsed = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid stress YAML: {exc}") from exc
    root = _exact_keys(
        parsed,
        {
            "schema_version",
            "key",
            "protocol",
            "seed",
            "metric_policy",
            "synthetic",
            "ucfrep_perturbations",
        },
        "stress config",
    )
    if root["schema_version"] != 1:
        raise ValueError("only stress schema_version=1 is supported")
    if root["key"] != "pams_stress_v1" or root["protocol"] != "synthetic_then_ucfrep":
        raise ValueError("stress config identity mismatch")
    if isinstance(root["seed"], bool) or int(root["seed"]) != root["seed"]:
        raise ValueError("stress seed must be an integer")
    metric_policy = _exact_keys(
        root["metric_policy"],
        {
            "report_clean_and_each_perturbation_separately",
            "tune_on_stress_results",
        },
        "stress metric_policy",
    )
    if metric_policy != {
        "report_clean_and_each_perturbation_separately": True,
        "tune_on_stress_results": False,
    }:
        raise ValueError("stress metric policy must report separately and forbid tuning")
    rows = root["ucfrep_perturbations"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("ucfrep_perturbations must be a non-empty list")

    conditions: list[StressCondition] = []
    observed_kinds: list[str] = []
    for index, raw_row in enumerate(rows):
        row = _exact_keys(raw_row, {"key", "parameters"}, f"perturbation[{index}]")
        kind = str(row["key"])
        observed_kinds.append(kind)
        parameters = row["parameters"]
        if kind == "clean":
            if parameters != {}:
                raise ValueError("clean parameters must be empty")
            conditions.append(StressCondition("clean", "clean", {}))
        elif kind == "temporal_speed":
            speed_values = _non_empty_numbers(
                _exact_keys(
                    parameters,
                    {"speed_multipliers"},
                    "temporal_speed parameters",
                )["speed_multipliers"],
                "speed_multipliers",
            )
            if any(value <= 0.0 or value > 2.0 for value in speed_values):
                raise ValueError("speed multipliers must be in (0, 2]")
            conditions.extend(
                StressCondition(
                    f"temporal_speed_{_condition_suffix(value)}",
                    "temporal_speed",
                    {"speed_multiplier": value},
                )
                for value in speed_values
            )
        elif kind == "temporal_pause":
            pause_values = _exact_keys(
                parameters,
                {"fraction_of_sequence", "location"},
                "temporal_pause parameters",
            )
            fraction = _fraction(
                pause_values["fraction_of_sequence"],
                "pause fraction",
            )
            if pause_values["location"] != "middle":
                raise ValueError("only the frozen middle pause is supported")
            conditions.append(
                StressCondition(
                    "temporal_pause_middle",
                    "temporal_pause",
                    {"fraction_of_sequence": fraction, "location": "middle"},
                )
            )
        elif kind == "pose_occlusion":
            occlusion_values = _exact_keys(
                parameters,
                {"time_fraction", "joint_fraction", "fill_value", "mark_invalid"},
                "pose_occlusion parameters",
            )
            if occlusion_values["mark_invalid"] is not True:
                raise ValueError("pose occlusion must explicitly mark coordinates invalid")
            conditions.append(
                StressCondition(
                    "pose_occlusion",
                    "pose_occlusion",
                    {
                        "time_fraction": _fraction(
                            occlusion_values["time_fraction"],
                            "occlusion time_fraction",
                            include_one=True,
                        ),
                        "joint_fraction": _fraction(
                            occlusion_values["joint_fraction"],
                            "occlusion joint_fraction",
                            include_one=True,
                        ),
                        "fill_value": _finite(
                            occlusion_values["fill_value"],
                            "occlusion fill_value",
                        ),
                        "mark_invalid": True,
                    },
                )
            )
        elif kind == "rotation":
            rotation_values = _non_empty_numbers(
                _exact_keys(parameters, {"degrees"}, "rotation parameters")["degrees"],
                "rotation degrees",
            )
            conditions.extend(
                StressCondition(
                    f"rotation_{_condition_suffix(value)}deg",
                    "rotation",
                    {"degrees": value},
                )
                for value in rotation_values
            )
        elif kind == "isotropic_scale":
            scale_values = _non_empty_numbers(
                _exact_keys(
                    parameters,
                    {"multipliers"},
                    "isotropic_scale parameters",
                )["multipliers"],
                "scale multipliers",
            )
            if any(value <= 0.0 for value in scale_values):
                raise ValueError("scale multipliers must be positive")
            conditions.extend(
                StressCondition(
                    f"isotropic_scale_{_condition_suffix(value)}",
                    "isotropic_scale",
                    {"multiplier": value},
                )
                for value in scale_values
            )
        elif kind == "translation":
            translation_values = _non_empty_numbers(
                _exact_keys(
                    parameters,
                    {"coordinate_range_fraction"},
                    "translation parameters",
                )["coordinate_range_fraction"],
                "translation coordinate_range_fraction",
            )
            conditions.extend(
                StressCondition(
                    f"translation_{_condition_suffix(value)}",
                    "translation",
                    {"coordinate_range_fraction": value},
                )
                for value in translation_values
            )
        else:
            raise ValueError(f"unsupported UCFRep perturbation key: {kind!r}")

    expected_kinds = [
        "clean",
        "temporal_speed",
        "temporal_pause",
        "pose_occlusion",
        "rotation",
        "isotropic_scale",
        "translation",
    ]
    if observed_kinds != expected_kinds:
        raise ValueError(
            "UCFRep perturbations must use the frozen order "
            f"{expected_kinds}, received {observed_kinds}"
        )
    return StressProtocol(
        key=str(root["key"]),
        seed=int(root["seed"]),
        config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        conditions=tuple(conditions),
    )


def pose_sequence_sha256(sequence: PoseSequence) -> str:
    """Hash the exact model input, including identifier, FPS and frame mask."""

    digest = hashlib.sha256()
    digest.update(b"pams-pose-stress-v1\0")
    digest.update(sequence.video_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.asarray([sequence.fps], dtype="<f8").tobytes())
    digest.update(np.asarray(sequence.xyz, dtype="<f4", order="C").tobytes(order="C"))
    digest.update(np.asarray(sequence.valid_mask, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def _stable_rng(seed: int, video_id: str, condition_id: str) -> np.random.Generator:
    encoded = f"{seed}\0{video_id}\0{condition_id}".encode()
    derived = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")
    return np.random.default_rng(derived)


def _interpolate_time(
    sequence: PoseSequence,
    source_positions: np.ndarray,
) -> PoseSequence:
    frame_count = sequence.num_frames
    clipped = np.clip(np.asarray(source_positions, dtype=np.float64), 0.0, frame_count - 1)
    left = np.floor(clipped).astype(np.int64)
    right = np.minimum(left + 1, frame_count - 1)
    weight = (clipped - left).astype(np.float32)
    xyz = (
        sequence.xyz[left] * (1.0 - weight[:, None, None])
        + sequence.xyz[right] * weight[:, None, None]
    )
    exact = left == right
    mask = (sequence.valid_mask[left] & sequence.valid_mask[right]) | (
        exact & sequence.valid_mask[left]
    )
    return PoseSequence(sequence.video_id, sequence.fps, xyz, mask)


def _temporal_speed(
    sequence: PoseSequence,
    multiplier: float,
) -> tuple[PoseSequence, dict[str, Any]]:
    frame_count = sequence.num_frames
    output_phase = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)
    complement = 2.0 - multiplier
    source_phase = np.where(
        output_phase <= 0.5,
        multiplier * output_phase,
        0.5 * multiplier + complement * (output_phase - 0.5),
    )
    transformed = _interpolate_time(sequence, source_phase * (frame_count - 1))
    return transformed, {
        "speed_multiplier_first_half": multiplier,
        "compensating_multiplier_second_half": complement,
        "output_frames": frame_count,
        "endpoint_preserving": True,
    }


def _temporal_pause(
    sequence: PoseSequence,
    fraction: float,
) -> tuple[PoseSequence, dict[str, Any]]:
    frame_count = sequence.num_frames
    output_phase = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)
    pause_start = (1.0 - fraction) / 2.0
    pause_stop = pause_start + fraction
    source_phase = np.empty_like(output_phase)
    before = output_phase < pause_start
    after = output_phase > pause_stop
    source_phase[before] = output_phase[before] / (1.0 - fraction)
    source_phase[~before & ~after] = 0.5
    source_phase[after] = 0.5 + (output_phase[after] - pause_stop) / (1.0 - fraction)
    transformed = _interpolate_time(sequence, source_phase * (frame_count - 1))
    frozen_frames = int(np.count_nonzero(~before & ~after))
    return transformed, {
        "fraction_of_sequence": fraction,
        "location": "middle",
        "frozen_output_frames": frozen_frames,
        "outside_speed_multiplier": 1.0 / (1.0 - fraction),
        "output_frames": frame_count,
        "endpoint_preserving": True,
    }


def _frame_centroids(xyz: np.ndarray) -> np.ndarray:
    return np.mean(xyz, axis=1, keepdims=True, dtype=np.float64).astype(np.float32)


def apply_stress_condition(
    sequence: PoseSequence,
    condition: StressCondition,
    *,
    seed: int,
) -> tuple[PoseSequence, PerturbationAudit]:
    """Apply one condition without reading an action or count target."""

    if sequence.num_frames != 256:
        raise ValueError(
            f"UCFRep stress requires 256-frame pose caches, got {sequence.num_frames}"
        )
    xyz = np.array(sequence.xyz, dtype=np.float32, copy=True)
    mask = np.array(sequence.valid_mask, dtype=np.bool_, copy=True)
    algorithm: str
    realized: dict[str, Any]

    if condition.kind == "clean":
        transformed = PoseSequence(sequence.video_id, sequence.fps, xyz, mask)
        algorithm = "identity"
        realized = {}
    elif condition.kind == "temporal_speed":
        multiplier = float(condition.parameters["speed_multiplier"])
        transformed, realized = _temporal_speed(sequence, multiplier)
        algorithm = "two_segment_endpoint_preserving_linear_resample"
    elif condition.kind == "temporal_pause":
        fraction = float(condition.parameters["fraction_of_sequence"])
        transformed, realized = _temporal_pause(sequence, fraction)
        algorithm = "middle_freeze_endpoint_preserving_linear_resample"
    elif condition.kind == "pose_occlusion":
        time_count = max(
            1,
            min(
                sequence.num_frames,
                int(math.floor(sequence.num_frames * condition.parameters["time_fraction"] + 0.5)),
            ),
        )
        joint_count = max(
            1,
            min(
                xyz.shape[1],
                int(math.floor(xyz.shape[1] * condition.parameters["joint_fraction"] + 0.5)),
            ),
        )
        rng = _stable_rng(seed, sequence.video_id, condition.condition_id)
        start = int(rng.integers(0, sequence.num_frames - time_count + 1))
        joints = np.sort(rng.choice(xyz.shape[1], size=joint_count, replace=False))
        xyz[start : start + time_count, joints, :] = float(
            condition.parameters["fill_value"]
        )
        transformed = PoseSequence(sequence.video_id, sequence.fps, xyz, mask)
        algorithm = "contiguous_time_random_joint_zero_fill"
        realized = {
            "time_start": start,
            "time_count": time_count,
            "joint_indices": [int(value) for value in joints],
            "joint_count": joint_count,
            "fill_value": float(condition.parameters["fill_value"]),
            "mark_invalid": True,
            "validity_representation": (
                "selected joint coordinates set to fill_value; frame-level valid_mask "
                "unchanged because PoseSequence has no joint-level mask"
            ),
        }
    elif condition.kind == "rotation":
        angle = math.radians(float(condition.parameters["degrees"]))
        cosine = math.cos(angle)
        sine = math.sin(angle)
        center = _frame_centroids(xyz)
        centered = xyz - center
        rotated = centered.copy()
        rotated[..., 0] = cosine * centered[..., 0] - sine * centered[..., 1]
        rotated[..., 1] = sine * centered[..., 0] + cosine * centered[..., 1]
        xyz = rotated + center
        transformed = PoseSequence(sequence.video_id, sequence.fps, xyz, mask)
        algorithm = "per_frame_centroid_z_axis_rotation"
        realized = {"degrees": float(condition.parameters["degrees"])}
    elif condition.kind == "isotropic_scale":
        multiplier = float(condition.parameters["multiplier"])
        center = _frame_centroids(xyz)
        xyz = center + multiplier * (xyz - center)
        transformed = PoseSequence(sequence.video_id, sequence.fps, xyz, mask)
        algorithm = "per_frame_centroid_isotropic_scale"
        realized = {"multiplier": multiplier}
    elif condition.kind == "translation":
        fraction = float(condition.parameters["coordinate_range_fraction"])
        valid_coordinates = xyz[mask]
        coordinate_span = (
            np.zeros(3, dtype=np.float32)
            if valid_coordinates.size == 0
            else np.ptp(valid_coordinates, axis=(0, 1)).astype(np.float32)
        )
        delta = fraction * coordinate_span
        xyz[mask] += delta[None, None, :]
        transformed = PoseSequence(sequence.video_id, sequence.fps, xyz, mask)
        algorithm = "global_valid_coordinate_span_translation_all_axes"
        realized = {
            "coordinate_range_fraction": fraction,
            "coordinate_span": [float(value) for value in coordinate_span],
            "translation_delta": [float(value) for value in delta],
        }
    else:
        raise AssertionError(f"unhandled stress condition: {condition.kind}")

    audit = PerturbationAudit(
        condition_id=condition.condition_id,
        algorithm=algorithm,
        parameters=realized,
        transformed_pose_sha256=pose_sequence_sha256(transformed),
    )
    return transformed, audit
