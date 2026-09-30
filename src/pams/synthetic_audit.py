"""Auditable synthetic acceptance suite bound to ``configs/stress.yaml``."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from pams.baselines.proxy import SpectralProxyAdapter
from pams.config import PAMSConfig, load_config
from pams.metrics import compute_count_metrics
from pams.pams_dev import _encoded_json, _write_json_bundle
from pams.reproducibility import (
    clean_git_revision,
    hardware_fingerprint,
    sha256_file,
    sha256_json,
)
from pams.safety import (
    run_counter_sign_phase_gate,
    run_period_counter_sign_phase_gate,
)
from pams.synthetic import SyntheticSample, SyntheticSpec, generate_synthetic_sample

_ARTIFACT_NAME: Literal["synthetic-acceptance.json"] = "synthetic-acceptance.json"
_RECEIPT_NAME: Literal["synthetic-acceptance.receipt.json"] = (
    "synthetic-acceptance.receipt.json"
)
_REFERENCE_COUNT = 8
_POSE_THRESHOLDS = {
    "max_absolute_error": 1.0,
    "obo": 1.0,
    "nmae": 0.08,
}


def _mapping(value: object, expected: set[str], name: str) -> dict[str, Any]:
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


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return value


def _finite(value: object, name: str) -> float:
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class SyntheticAuditPolicy:
    """Strict synthetic section of the frozen stress policy."""

    seed: int
    frames: int
    count_values: tuple[int, ...]
    constant_speed_multipliers: tuple[float, ...]
    linear_speed_ramps: tuple[tuple[float, float], ...]
    pause_fractions: tuple[float, ...]
    noise_standard_deviations: tuple[float, ...]
    missing_joint_fractions: tuple[tuple[float, float], ...]
    second_harmonic_amplitudes: tuple[float, ...]
    stress_config_sha256: str

    @property
    def expected_case_count(self) -> int:
        return sum(
            (
                len(self.count_values),
                len(self.constant_speed_multipliers),
                len(self.linear_speed_ramps),
                len(self.pause_fractions),
                len(self.noise_standard_deviations),
                len(self.missing_joint_fractions),
                len(self.second_harmonic_amplitudes),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "frames": self.frames,
            "reference_count": _REFERENCE_COUNT,
            "count_values": list(self.count_values),
            "constant_speed_multipliers": list(self.constant_speed_multipliers),
            "linear_speed_ramps": [
                {"start": start, "stop": stop}
                for start, stop in self.linear_speed_ramps
            ],
            "pause_fractions": list(self.pause_fractions),
            "noise_standard_deviations": list(self.noise_standard_deviations),
            "missing_joint_fractions": [
                {"time_fraction": time, "joint_fraction": joints}
                for time, joints in self.missing_joint_fractions
            ],
            "second_harmonic_amplitudes": list(
                self.second_harmonic_amplitudes
            ),
            "stress_config_sha256": self.stress_config_sha256,
            "expected_case_count": self.expected_case_count,
        }


@dataclass(frozen=True, slots=True)
class SyntheticAuditCase:
    """One preregistered pose-level synthetic case."""

    case_id: str
    category: str
    configured_parameters: dict[str, Any]
    spec: SyntheticSpec


def load_synthetic_audit_policy(path: str | Path) -> SyntheticAuditPolicy:
    """Load and strictly validate the synthetic acceptance matrix."""

    source = Path(path)
    raw_bytes = source.read_bytes()
    try:
        raw = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid stress YAML: {exc}") from exc
    root = _mapping(
        raw,
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
    if (
        root["schema_version"] != 1
        or root["key"] != "pams_stress_v1"
        or root["protocol"] != "synthetic_then_ucfrep"
    ):
        raise ValueError("stress config identity mismatch")
    if isinstance(root["seed"], bool) or int(root["seed"]) != root["seed"]:
        raise ValueError("stress seed must be an integer")
    metric_policy = _mapping(
        root["metric_policy"],
        {
            "report_clean_and_each_perturbation_separately",
            "tune_on_stress_results",
        },
        "metric_policy",
    )
    if metric_policy != {
        "report_clean_and_each_perturbation_separately": True,
        "tune_on_stress_results": False,
    }:
        raise ValueError("synthetic audit requires separate reporting and no tuning")
    synthetic = _mapping(
        root["synthetic"],
        {
            "count_values",
            "frames",
            "speed",
            "pauses",
            "gaussian_noise_standard_deviations",
            "missing_joints",
            "harmonics",
        },
        "synthetic policy",
    )
    raw_counts = _list(synthetic["count_values"], "count_values")
    if any(
        isinstance(value, bool) or int(value) != value
        for value in raw_counts
    ):
        raise ValueError("count_values must contain integers")
    counts = tuple(int(value) for value in raw_counts)
    if counts != tuple(range(2, 41)):
        raise ValueError("synthetic count_values must be the frozen inclusive 2--40 sweep")
    if synthetic["frames"] != 256:
        raise ValueError("synthetic acceptance requires exactly 256 frames")

    speed = _mapping(
        synthetic["speed"],
        {"constant_multipliers", "linear_ramps"},
        "synthetic speed",
    )
    constants = tuple(
        _finite(value, f"constant_multipliers[{index}]")
        for index, value in enumerate(
            _list(speed["constant_multipliers"], "constant_multipliers")
        )
    )
    if constants != (0.5, 1.0, 2.0):
        raise ValueError("constant speed multipliers must remain [0.5, 1.0, 2.0]")
    ramps = []
    for index, value in enumerate(_list(speed["linear_ramps"], "linear_ramps")):
        ramp = _mapping(value, {"start", "stop"}, f"linear_ramps[{index}]")
        start = _finite(ramp["start"], f"linear_ramps[{index}].start")
        stop = _finite(ramp["stop"], f"linear_ramps[{index}].stop")
        if start <= 0.0 or stop <= 0.0:
            raise ValueError("linear speed ramp endpoints must be positive")
        ramps.append((start, stop))
    if tuple(ramps) != ((0.5, 2.0), (2.0, 0.5)):
        raise ValueError("linear speed ramps differ from the frozen bidirectional pair")

    pause_fractions = []
    for index, value in enumerate(_list(synthetic["pauses"], "pauses")):
        pause = _mapping(
            value,
            {"fraction_of_sequence", "location"},
            f"pauses[{index}]",
        )
        fraction = _finite(
            pause["fraction_of_sequence"],
            f"pauses[{index}].fraction_of_sequence",
        )
        if not 0.0 < fraction < 1.0 or pause["location"] != "middle":
            raise ValueError("pauses must be positive middle-sequence fractions")
        pause_fractions.append(fraction)
    if tuple(pause_fractions) != (0.1, 0.2):
        raise ValueError("pause fractions differ from the frozen 10%/20% pair")

    noise_values = tuple(
        _finite(value, f"noise[{index}]")
        for index, value in enumerate(
            _list(
                synthetic["gaussian_noise_standard_deviations"],
                "gaussian_noise_standard_deviations",
            )
        )
    )
    if noise_values != (0.0, 0.01, 0.03, 0.05):
        raise ValueError("noise values differ from the frozen four-level sweep")

    missing_values = []
    for index, value in enumerate(
        _list(synthetic["missing_joints"], "missing_joints")
    ):
        missing = _mapping(
            value,
            {"time_fraction", "joint_fraction"},
            f"missing_joints[{index}]",
        )
        time_fraction = _finite(
            missing["time_fraction"],
            f"missing_joints[{index}].time_fraction",
        )
        joint_fraction = _finite(
            missing["joint_fraction"],
            f"missing_joints[{index}].joint_fraction",
        )
        if not 0.0 < time_fraction <= 1.0 or not 0.0 < joint_fraction <= 1.0:
            raise ValueError("missing-joint fractions must be in (0, 1]")
        missing_values.append((time_fraction, joint_fraction))
    if tuple(missing_values) != ((0.2, 0.3),):
        raise ValueError("missing-joint policy differs from frozen 20% x 30%")

    harmonics = _mapping(
        synthetic["harmonics"],
        {"second_harmonic_amplitudes"},
        "harmonics",
    )
    harmonic_values = tuple(
        _finite(value, f"second_harmonic_amplitudes[{index}]")
        for index, value in enumerate(
            _list(
                harmonics["second_harmonic_amplitudes"],
                "second_harmonic_amplitudes",
            )
        )
    )
    if harmonic_values != (0.0, 0.25, 0.5, 1.0):
        raise ValueError("second-harmonic amplitudes differ from the frozen sweep")
    return SyntheticAuditPolicy(
        seed=int(root["seed"]),
        frames=256,
        count_values=counts,
        constant_speed_multipliers=constants,
        linear_speed_ramps=tuple(ramps),
        pause_fractions=tuple(pause_fractions),
        noise_standard_deviations=noise_values,
        missing_joint_fractions=tuple(missing_values),
        second_harmonic_amplitudes=harmonic_values,
        stress_config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def _value_id(value: float) -> str:
    sign = "neg" if value < 0 else "pos"
    return f"{sign}{format(abs(value), '.12g').replace('.', 'p')}"


def build_synthetic_audit_cases(
    policy: SyntheticAuditPolicy,
) -> tuple[SyntheticAuditCase, ...]:
    """Expand the frozen 55-case pose-level suite in canonical order."""

    cases: list[SyntheticAuditCase] = []
    for count in policy.count_values:
        case_id = f"count-{count:02d}"
        cases.append(
            SyntheticAuditCase(
                case_id=case_id,
                category="count_sweep",
                configured_parameters={"count": count},
                spec=SyntheticSpec(
                    video_id=case_id,
                    frames=policy.frames,
                    count=float(count),
                    seed=policy.seed + count,
                ),
            )
        )
    for index, multiplier in enumerate(policy.constant_speed_multipliers):
        case_id = f"speed-constant-{_value_id(multiplier)}"
        cases.append(
            SyntheticAuditCase(
                case_id=case_id,
                category="constant_speed",
                configured_parameters={"multiplier": multiplier},
                spec=SyntheticSpec(
                    video_id=case_id,
                    frames=policy.frames,
                    count=float(_REFERENCE_COUNT),
                    speed_profile="constant",
                    speed_range=(multiplier, multiplier),
                    seed=policy.seed + 100 + index,
                ),
            )
        )
    for index, (start, stop) in enumerate(policy.linear_speed_ramps):
        case_id = f"speed-linear-{_value_id(start)}-to-{_value_id(stop)}"
        cases.append(
            SyntheticAuditCase(
                case_id=case_id,
                category="linear_speed",
                configured_parameters={"start": start, "stop": stop},
                spec=SyntheticSpec(
                    video_id=case_id,
                    frames=policy.frames,
                    count=float(_REFERENCE_COUNT),
                    speed_profile="linear",
                    speed_range=(start, stop),
                    seed=policy.seed + 200 + index,
                ),
            )
        )
    for index, fraction in enumerate(policy.pause_fractions):
        start = (1.0 - fraction) / 2.0
        stop = start + fraction
        case_id = f"pause-middle-{_value_id(fraction)}"
        cases.append(
            SyntheticAuditCase(
                case_id=case_id,
                category="pause",
                configured_parameters={
                    "fraction_of_sequence": fraction,
                    "location": "middle",
                },
                spec=SyntheticSpec(
                    video_id=case_id,
                    frames=policy.frames,
                    count=float(_REFERENCE_COUNT),
                    pause_ranges=((start, stop),),
                    seed=policy.seed + 300 + index,
                ),
            )
        )
    for index, noise_std in enumerate(policy.noise_standard_deviations):
        case_id = f"noise-{_value_id(noise_std)}"
        cases.append(
            SyntheticAuditCase(
                case_id=case_id,
                category="gaussian_noise",
                configured_parameters={"standard_deviation": noise_std},
                spec=SyntheticSpec(
                    video_id=case_id,
                    frames=policy.frames,
                    count=float(_REFERENCE_COUNT),
                    noise_std=noise_std,
                    seed=policy.seed + 400 + index,
                ),
            )
        )
    for index, (time_fraction, joint_fraction) in enumerate(
        policy.missing_joint_fractions
    ):
        case_id = (
            f"missing-joints-{_value_id(time_fraction)}x"
            f"{_value_id(joint_fraction)}"
        )
        cases.append(
            SyntheticAuditCase(
                case_id=case_id,
                category="missing_joints",
                configured_parameters={
                    "time_fraction": time_fraction,
                    "joint_fraction": joint_fraction,
                },
                spec=SyntheticSpec(
                    video_id=case_id,
                    frames=policy.frames,
                    count=float(_REFERENCE_COUNT),
                    occlusion_time_fraction=time_fraction,
                    occluded_joint_fraction=joint_fraction,
                    seed=policy.seed + 500 + index,
                ),
            )
        )
    for index, amplitude in enumerate(policy.second_harmonic_amplitudes):
        case_id = f"second-harmonic-{_value_id(amplitude)}"
        # SyntheticSpec divides coefficient k by harmonic index k. Passing
        # 2*amplitude makes the realized second-harmonic amplitude exact.
        generator_coefficient = 2.0 * amplitude
        cases.append(
            SyntheticAuditCase(
                case_id=case_id,
                category="second_harmonic",
                configured_parameters={
                    "second_harmonic_amplitude": amplitude,
                    "generator_coefficient": generator_coefficient,
                },
                spec=SyntheticSpec(
                    video_id=case_id,
                    frames=policy.frames,
                    count=float(_REFERENCE_COUNT),
                    harmonics=(1.0, generator_coefficient),
                    seed=policy.seed + 600 + index,
                ),
            )
        )
    result = tuple(cases)
    identifiers = [case.case_id for case in result]
    if len(result) != policy.expected_case_count:
        raise RuntimeError("expanded synthetic case count differs from policy")
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("expanded synthetic case IDs are not unique")
    return result


def _hash_array(value: np.ndarray[Any, Any]) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sample_input_hash(sample: SyntheticSample) -> tuple[str, dict[str, str]]:
    components = {
        "xyz": _hash_array(sample.sequence.xyz),
        "valid_mask": _hash_array(sample.sequence.valid_mask),
        "phase_radians": _hash_array(sample.phase_radians),
        "joint_visibility": _hash_array(sample.joint_visibility),
    }
    return sha256_json(components), components


def _source_hashes() -> dict[str, str]:
    directory = Path(__file__).parent
    return {
        "config": sha256_file(directory / "config.py"),
        "metrics": sha256_file(directory / "metrics.py"),
        "proxy": sha256_file(directory / "baselines" / "proxy.py"),
        "safety": sha256_file(directory / "safety.py"),
        "synthetic": sha256_file(directory / "synthetic.py"),
        "synthetic_audit": sha256_file(Path(__file__)),
    }


def _category_reports(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = tuple(dict.fromkeys(str(row["category"]) for row in rows))
    reports = {}
    for category in categories:
        selected = [row for row in rows if row["category"] == category]
        report = compute_count_metrics(
            [row["prediction"] for row in selected],
            [row["target_count"] for row in selected],
            video_ids=[row["case_id"] for row in selected],
            bootstrap_samples=0,
        )
        payload = report.to_dict()
        payload.pop("per_video")
        reports[category] = payload
    return reports


def _pose_suite(
    policy: SyntheticAuditPolicy,
    config: PAMSConfig,
) -> dict[str, Any]:
    adapter = SpectralProxyAdapter(
        minimum_period=config.period.minimum,
        maximum_period=config.period.maximum,
    )
    rows: list[dict[str, Any]] = []
    generation_cycle_errors: list[float] = []
    for case in build_synthetic_audit_cases(policy):
        sample = generate_synthetic_sample(case.spec)
        result = adapter.predict(sample.sequence)
        target = int(sample.target_count)
        generated_cycles = float(sample.phase_radians[-1] / (2.0 * math.pi))
        generation_error = abs(generated_cycles - sample.target_count)
        generation_cycle_errors.append(generation_error)
        input_sha256, component_hashes = _sample_input_hash(sample)
        absolute_error = abs(result.count - target)
        rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "configured_parameters": case.configured_parameters,
                "generator_spec": asdict(case.spec),
                "target_count": target,
                "generated_cycle_count": generated_cycles,
                "generation_cycle_error": generation_error,
                "input_sha256": input_sha256,
                "input_component_sha256": component_hashes,
                "valid_frame_count": int(np.count_nonzero(sample.sequence.valid_mask)),
                "visible_joint_fraction": float(np.mean(sample.joint_visibility)),
                "prediction": result.count,
                "period_frames": result.period_frames,
                "expert_counts": list(result.expert_counts),
                "confidence": result.confidence,
                "absolute_error": absolute_error,
                "normalized_absolute_error": absolute_error / target,
                "within_one": absolute_error <= 1,
            }
        )
    report = compute_count_metrics(
        [row["prediction"] for row in rows],
        [row["target_count"] for row in rows],
        video_ids=[row["case_id"] for row in rows],
        bootstrap_samples=10_000,
        bootstrap_seed=policy.seed,
        confidence_level=0.95,
    )
    maximum_absolute_error = max(int(row["absolute_error"]) for row in rows)
    maximum_generation_error = max(generation_cycle_errors)
    metrics = report.to_dict()
    checks = {
        "case_matrix_complete": len(rows) == policy.expected_case_count,
        "generation_cycle_count_exact": maximum_generation_error <= 1e-12,
        "max_absolute_error": (
            maximum_absolute_error <= _POSE_THRESHOLDS["max_absolute_error"]
        ),
        "obo": metrics["obo"] >= _POSE_THRESHOLDS["obo"],
        "nmae": metrics["nmae"] <= _POSE_THRESHOLDS["nmae"],
    }
    return {
        "scope": "pose_generator_and_spectral_proxy_diagnostic",
        "method": "spectral-proxy",
        "diagnostic_only": True,
        "eligible_for_paper_table": False,
        "thresholds": dict(_POSE_THRESHOLDS),
        "metrics": {
            **{key: value for key, value in metrics.items() if key != "per_video"},
            "max_absolute_error": maximum_absolute_error,
            "max_generation_cycle_error": maximum_generation_error,
        },
        "category_metrics": _category_reports(rows),
        "checks": checks,
        "passed": all(checks.values()),
        "cases": rows,
    }


def run_synthetic_acceptance(
    *,
    output_dir: str | Path,
    config_path: str | Path,
    stress_config_path: str | Path,
    repository_root: str | Path,
    command: list[str],
) -> dict[str, Any]:
    """Run and durably publish both 576-case gates and the 55-case pose suite."""

    destination = Path(output_dir)
    artifact_path = destination / _ARTIFACT_NAME
    receipt_path = destination / _RECEIPT_NAME
    collisions = [str(path) for path in (artifact_path, receipt_path) if path.exists()]
    if collisions:
        raise FileExistsError(
            f"refusing to overwrite synthetic acceptance artifacts: {collisions}"
        )
    config_file = Path(config_path)
    stress_file = Path(stress_config_path)
    repository = Path(repository_root)
    source_git_sha = clean_git_revision(repository)
    config_file_sha256 = sha256_file(config_file)
    config = load_config(config_file)
    policy = load_synthetic_audit_policy(stress_file)
    source_hashes = _source_hashes()
    source_code_sha256 = sha256_json(source_hashes)
    hardware = hardware_fingerprint()
    container = hardware.get("container")
    if isinstance(container, dict) and container.get("source_revision") != source_git_sha:
        raise ValueError("container source revision differs from checked-out Git revision")

    counter = run_counter_sign_phase_gate()
    period_counter = run_period_counter_sign_phase_gate()
    pose_suite = _pose_suite(policy, config)
    checks = {
        "counter_576_passed": bool(counter["passed"]),
        "period_counter_576_passed": bool(period_counter["passed"]),
        "pose_suite_passed": bool(pose_suite["passed"]),
    }
    for path, expected, name in (
        (config_file, config_file_sha256, "PAMS config"),
        (stress_file, policy.stress_config_sha256, "stress config"),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"{name} changed during synthetic acceptance")
    if _source_hashes() != source_hashes:
        raise RuntimeError("synthetic acceptance source changed during execution")
    if clean_git_revision(repository) != source_git_sha:
        raise RuntimeError("Git revision changed during synthetic acceptance")
    artifact = {
        "schema_version": 1,
        "artifact_type": "pams_synthetic_acceptance",
        "classification": "synthetic component acceptance diagnostic",
        "diagnostic_only": True,
        "eligible_for_paper_table": False,
        "sealed_test_accessed": False,
        "source_git_sha": source_git_sha,
        "source_code_files_sha256": source_hashes,
        "source_code_sha256": source_code_sha256,
        "config_file_sha256": config_file_sha256,
        "config_fingerprint": config.fingerprint,
        "stress_config_sha256": policy.stress_config_sha256,
        "policy": policy.to_dict(),
        "hardware": hardware,
        "checks": checks,
        "passed": all(checks.values()),
        "counter_576": counter,
        "period_counter_576": period_counter,
        "pose_suite": pose_suite,
    }
    encoded = _encoded_json(artifact)
    artifact_sha256 = hashlib.sha256(encoded).hexdigest()
    receipt = {
        "schema_version": 1,
        "artifact_type": "pams_synthetic_acceptance_receipt",
        "artifact_file": _ARTIFACT_NAME,
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": len(encoded),
        "passed": artifact["passed"],
        "source_git_sha": source_git_sha,
        "source_code_sha256": source_code_sha256,
        "config_file_sha256": config_file_sha256,
        "config_fingerprint": config.fingerprint,
        "stress_config_sha256": policy.stress_config_sha256,
        "command": command,
        "hardware": hardware,
    }
    written = _write_json_bundle(
        (
            (artifact_path, artifact),
            (receipt_path, receipt),
        )
    )
    return {
        "classification": artifact["classification"],
        "diagnostic_only": True,
        "eligible_for_paper_table": False,
        "sealed_test_accessed": False,
        "passed": artifact["passed"],
        "checks": checks,
        "pose_case_count": policy.expected_case_count,
        "pose_metrics": pose_suite["metrics"],
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": artifact_sha256,
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": written[receipt_path],
    }
