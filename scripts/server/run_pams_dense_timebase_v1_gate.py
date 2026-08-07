"""Run the prospectively frozen posthoc dense-readout synthetic gate.

This program accepts only the frozen readout policy and a new output path.  It
does not expose dataset, pose-cache, checkpoint, label, development, or test
arguments.  Synthetic repetition truth is generated inside the process and is
the only truth used by the gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from torch import Tensor

from pams.consensus import MultiExpertCounter
from pams.period import estimate_period_batch_direct_fft
from pams.reproducibility import clean_git_revision, sha256_file, sha256_json
from pams.safety import run_counter_sign_phase_gate

_ARTIFACT_TYPE = "pams_dense_timebase_v1_synthetic_gate"
_RECEIPT_TYPE = "pams_dense_timebase_v1_synthetic_gate_receipt"
_EXPECTED_CANDIDATE = (
    "pams-reference-relative-dense-resampled-masked-dft-multi-v1"
)
_REQUIRED_THRESHOLDS = {
    "legacy_all_valid_exact_parity",
    "median_period_relative_error_maximum",
    "p95_period_relative_error_maximum",
    "harmonic_alias_fraction_maximum",
    "chain_nmae_maximum",
    "chain_obo_minimum",
    "clean_majority_fraction_minimum",
    "harmonic_majority_fraction_minimum",
    "resample_period_relative_error_maximum",
    "resample_count_equal_fraction_minimum",
    "invalid_payload_exact_invariance",
    "all_selected_modes_multi",
}
_FORBIDDEN_POLICY_TERMS = frozenset(
    {
        "dev_predictions",
        "dev_targets",
        "test_predictions",
        "test_targets",
        "ground_truth_counts",
    }
)


def _encoded_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _stable_file_identity(path: Path) -> tuple[str, int]:
    before = path.stat()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"input must be a regular non-symlink file: {path}")
    payload = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"input changed while it was read: {path}")
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _load_policy(path: str | Path) -> tuple[dict[str, Any], str, str]:
    source = Path(path)
    digest, _ = _stable_file_identity(source)
    raw = source.read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw)
    if not isinstance(parsed, dict):
        raise ValueError("dense-timebase policy root must be a mapping")
    policy = dict(parsed)
    if policy.get("schema_version") != 1:
        raise ValueError("dense-timebase policy requires schema_version=1")
    if policy.get("candidate_id") != _EXPECTED_CANDIDATE:
        raise ValueError("dense-timebase policy candidate_id is not frozen v1")
    if policy.get("table2_eligible") is not False:
        raise ValueError("inferred dense-timebase closure must remain Table-2 ineligible")

    readout = policy.get("readout")
    if not isinstance(readout, Mapping):
        raise ValueError("dense-timebase policy is missing readout")
    expected_readout = {
        "direct_fft_timebase": "dense_resampled",
        "timeline": "full_resampled_clip",
        "detrend_fit": "valid_samples_at_dense_indices",
        "invalid_sample_contribution": "zero_after_valid_only_affine_detrend",
        "interpolation_across_invalid_frames": False,
        "frequency_definition": "k_over_timeline_frames",
        "period_definition": "timeline_frames_over_k",
        "consensus_expert_mode": "multi",
        "majority_first": True,
        "checkpoint_retraining": False,
    }
    for key, expected in expected_readout.items():
        if readout.get(key) != expected:
            raise ValueError(f"dense-timebase policy readout field {key!r} drifted")
    timeline_frames = readout.get("timeline_frames")
    if isinstance(timeline_frames, bool) or not isinstance(timeline_frames, int):
        raise ValueError("policy timeline_frames must be an integer")
    if timeline_frames != 256:
        raise ValueError("synthetic v1 gate is frozen to 256 timeline frames")

    boundary = policy.get("selection_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("dense-timebase policy is missing selection_boundary")
    if boundary.get("candidate_count") != 1 or boundary.get("parameter_sweep") is not False:
        raise ValueError("dense-timebase gate permits exactly one unswept candidate")
    if boundary.get("permitted_label_source") != (
        "deterministic_synthetic_generation_truth_only"
    ):
        raise ValueError("synthetic truth firewall drifted")

    synthetic = policy.get("synthetic_gate")
    if not isinstance(synthetic, Mapping):
        raise ValueError("dense-timebase policy is missing synthetic_gate")
    thresholds = synthetic.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != _REQUIRED_THRESHOLDS:
        raise ValueError("synthetic threshold names do not match frozen v1")
    if tuple(synthetic.get("periods", ())) != (4, 8, 16, 32, 64, 128):
        raise ValueError("synthetic period grid drifted")
    if tuple(synthetic.get("count_range_inclusive", ())) != (2, 40):
        raise ValueError("synthetic count range drifted")
    if tuple(synthetic.get("phase_fractions", ())) != (0.0, 0.125, 0.25, 0.375):
        raise ValueError("synthetic phase grid drifted")
    if tuple(synthetic.get("harmonic_orders", ())) != (2, 3, 4, 5, 6, 7):
        raise ValueError("synthetic harmonic-order grid drifted")
    if tuple(synthetic.get("harmonic_amplitudes", ())) != (0.25, 0.5, 0.75):
        raise ValueError("synthetic harmonic-amplitude grid drifted")
    if tuple(synthetic.get("perturbations", ())) != (
        "linear_drift",
        "internal_gaps",
        "leading_and_trailing_padding",
        "invalid_payload_pollution",
        "two_x_temporal_resample",
        "sign_flip",
    ):
        raise ValueError("synthetic perturbation suite drifted")

    authorization = policy.get("authorization")
    if not isinstance(authorization, Mapping) or authorization.get(
        "test105_attempt_budget"
    ) != 0:
        raise ValueError("synthetic policy must keep test105 attempt budget at zero")
    semantic_sha256 = sha256_json(policy)
    if any(term in policy for term in _FORBIDDEN_POLICY_TERMS):
        raise ValueError("policy contains a forbidden external truth field")
    if _stable_file_identity(source)[0] != digest:
        raise RuntimeError("dense-timebase policy changed while it was validated")
    return policy, digest, semantic_sha256


def _periodic_stream(
    *,
    timeline_frames: int,
    period_frames: float,
    phase_fraction: float,
    drift: float = 0.0,
    sign: float = 1.0,
) -> Tensor:
    time = torch.arange(timeline_frames, dtype=torch.float64)
    angle = 2.0 * math.pi * (time / period_frames + phase_fraction)
    # A broad circular pulse has one unambiguous peak per cycle while retaining
    # a dominant fundamental for the direct FFT decoder.
    circular = torch.atan2(torch.sin(angle), torch.cos(angle))
    stream = torch.exp(-0.5 * torch.square(circular / 0.70))
    if drift:
        stream = stream + drift * time / max(1, timeline_frames - 1)
    return sign * stream


def _harmonic_stream(
    *,
    timeline_frames: int,
    period_frames: float,
    phase_fraction: float,
    harmonic_order: int,
    harmonic_amplitude: float,
) -> Tensor:
    time = torch.arange(timeline_frames, dtype=torch.float64)
    angle = 2.0 * math.pi * (time / period_frames + phase_fraction)
    return torch.sin(angle) + harmonic_amplitude * torch.sin(
        harmonic_order * angle + 0.2718281828459045
    )


def _has_majority(counts: Sequence[int]) -> bool:
    return max(counts.count(value) for value in set(counts)) >= 2


def _decode(
    stream: Tensor,
    mask: Tensor,
    *,
    minimum: int,
    maximum: int,
    timebase: Literal["compact_valid", "dense_resampled"],
    counter: MultiExpertCounter,
) -> dict[str, Any]:
    if stream.ndim != 1 or mask.shape != stream.shape or mask.dtype != torch.bool:
        raise ValueError("stream and boolean mask must have matching one-dimensional shape")
    timeline_lengths = (
        torch.tensor([stream.numel()], dtype=torch.long)
        if timebase == "dense_resampled"
        else None
    )
    periods, confidences = estimate_period_batch_direct_fft(
        stream.unsqueeze(0),
        minimum=minimum,
        maximum=maximum,
        valid_mask=mask.unsqueeze(0),
        timebase=timebase,
        timeline_lengths=timeline_lengths,
    )
    period = float(periods[0])
    confidence = float(confidences[0])
    consensus = counter.count(
        stream,
        period,
        valid_mask=mask,
        period_confidence=confidence,
        reference_frames=(stream.numel() if timebase == "dense_resampled" else None),
    )
    return {
        "period_frames": period,
        "period_confidence": confidence,
        "fft_bin": int(round(stream.numel() / period)),
        "count": consensus.count,
        "expert_counts": list(consensus.expert_counts),
        "reference_count": consensus.reference_count,
        "selection_mode": consensus.selection_mode,
        "selected_expert": consensus.selected_expert,
        "has_majority": _has_majority(list(consensus.expert_counts)),
    }


def _exact_decode_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    fields = (
        "period_frames",
        "period_confidence",
        "count",
        "expert_counts",
        "reference_count",
        "selection_mode",
        "selected_expert",
    )
    return all(left[field] == right[field] for field in fields)


def _mask_variants(timeline_frames: int) -> dict[str, Tensor]:
    index = torch.arange(timeline_frames)
    long_gap = torch.ones(timeline_frames, dtype=torch.bool)
    long_gap[80:176] = False
    fragmented = ((index * 17 + 3) % 23) >= 5
    leading_trailing = torch.ones(timeline_frames, dtype=torch.bool)
    leading_trailing[:24] = False
    leading_trailing[-24:] = False
    return {
        "long_gap": long_gap,
        "fragmented": fragmented,
        "leading_and_trailing_padding": leading_trailing,
    }


def _pollute_invalid(stream: Tensor, mask: Tensor) -> Tensor:
    index = torch.arange(stream.numel(), dtype=stream.dtype)
    payload = 1_000_000.0 * torch.cos(index * 0.6180339887498948) + index
    return torch.where(mask, stream, payload)


def _repeat_two_x(stream: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Double the dense clock without inventing values across mask gaps."""

    return stream.repeat_interleave(2), mask.repeat_interleave(2)


def _fraction(values: Sequence[bool]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty boolean sequence")
    return float(sum(values) / len(values))


def _relative_error(observed: float, expected: float) -> float:
    return abs(observed - expected) / expected


def _evaluate_checks(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "legacy_all_valid_exact_parity": bool(
            metrics["legacy_all_valid_exact_parity"]
        )
        is bool(thresholds["legacy_all_valid_exact_parity"]),
        "median_period_relative_error": float(
            metrics["median_period_relative_error"]
        )
        <= float(thresholds["median_period_relative_error_maximum"]),
        "p95_period_relative_error": float(metrics["p95_period_relative_error"])
        <= float(thresholds["p95_period_relative_error_maximum"]),
        "harmonic_alias_fraction": float(metrics["harmonic_alias_fraction"])
        <= float(thresholds["harmonic_alias_fraction_maximum"]),
        "chain_nmae": float(metrics["chain_nmae"])
        <= float(thresholds["chain_nmae_maximum"]),
        "chain_obo": float(metrics["chain_obo"])
        >= float(thresholds["chain_obo_minimum"]),
        "clean_majority_fraction": float(metrics["clean_majority_fraction"])
        >= float(thresholds["clean_majority_fraction_minimum"]),
        "harmonic_majority_fraction": float(metrics["harmonic_majority_fraction"])
        >= float(thresholds["harmonic_majority_fraction_minimum"]),
        "resample_period_relative_error": float(
            metrics["resample_period_relative_error_maximum"]
        )
        <= float(thresholds["resample_period_relative_error_maximum"]),
        "resample_count_equal_fraction": float(
            metrics["resample_count_equal_fraction"]
        )
        >= float(thresholds["resample_count_equal_fraction_minimum"]),
        "invalid_payload_exact_invariance": bool(
            metrics["invalid_payload_exact_invariance"]
        )
        is bool(thresholds["invalid_payload_exact_invariance"]),
        "all_selected_modes_multi": bool(metrics["all_selected_modes_multi"])
        is bool(thresholds["all_selected_modes_multi"]),
        "existing_576_counter_gate": bool(metrics["existing_576_counter_gate_passed"]),
    }


def run_synthetic_gate(
    policy_path: str | Path,
    *,
    source_git_sha: str,
    runner_sha256: str,
) -> dict[str, Any]:
    """Execute the single frozen posthoc candidate without external data."""

    policy, policy_sha256, policy_semantic_sha256 = _load_policy(policy_path)
    synthetic = policy["synthetic_gate"]
    readout = policy["readout"]
    thresholds = dict(synthetic["thresholds"])
    timeline_frames = int(readout["timeline_frames"])
    periods = tuple(int(value) for value in synthetic["periods"])
    phases = tuple(float(value) for value in synthetic["phase_fractions"])
    minimum, maximum = min(periods), max(periods)
    counter = MultiExpertCounter(expert_mode="multi")

    counter_gate = run_counter_sign_phase_gate(counter)
    counter_gate_summary = {
        "passed": bool(counter_gate["passed"]),
        "scope": counter_gate["scope"],
        "period_source": counter_gate["period_source"],
        "case_count": counter_gate["metrics"]["case_count"],
        "metrics": counter_gate["metrics"],
        "checks": counter_gate["checks"],
        "thresholds": counter_gate["thresholds"],
        "configuration": counter_gate["configuration"],
        "details_sha256": sha256_json(counter_gate["details"]),
    }

    period_errors: list[float] = []
    parity: list[bool] = []
    all_modes: list[bool] = []
    clean_period_rows: list[dict[str, Any]] = []
    for period in periods:
        for phase in phases:
            stream = _periodic_stream(
                timeline_frames=timeline_frames,
                period_frames=period,
                phase_fraction=phase,
                drift=0.20,
                sign=-1.0 if phase in {0.125, 0.375} else 1.0,
            )
            mask = torch.ones(timeline_frames, dtype=torch.bool)
            dense = _decode(
                stream,
                mask,
                minimum=minimum,
                maximum=maximum,
                timebase="dense_resampled",
                counter=counter,
            )
            compact = _decode(
                stream,
                mask,
                minimum=minimum,
                maximum=maximum,
                timebase="compact_valid",
                counter=counter,
            )
            error = _relative_error(float(dense["period_frames"]), float(period))
            period_errors.append(error)
            parity.append(_exact_decode_equal(dense, compact))
            all_modes.extend(
                [dense["selection_mode"] == "multi", compact["selection_mode"] == "multi"]
            )
            clean_period_rows.append(
                {
                    "period_frames": period,
                    "phase_fraction": phase,
                    "period_relative_error": error,
                    "all_valid_compact_dense_exact_parity": parity[-1],
                    "dense": dense,
                }
            )

    count_min, count_max = (
        int(synthetic["count_range_inclusive"][0]),
        int(synthetic["count_range_inclusive"][1]),
    )
    clean_count_rows: list[dict[str, Any]] = []
    chain_errors: list[float] = []
    chain_within_one: list[bool] = []
    clean_majorities: list[bool] = []
    for target_count in range(count_min, count_max + 1):
        target_period = timeline_frames / target_count
        for phase in phases:
            stream = _periodic_stream(
                timeline_frames=timeline_frames,
                period_frames=target_period,
                phase_fraction=phase,
            )
            mask = torch.ones(timeline_frames, dtype=torch.bool)
            dense = _decode(
                stream,
                mask,
                minimum=minimum,
                maximum=maximum,
                timebase="dense_resampled",
                counter=counter,
            )
            compact = _decode(
                stream,
                mask,
                minimum=minimum,
                maximum=maximum,
                timebase="compact_valid",
                counter=counter,
            )
            absolute_error = abs(int(dense["count"]) - target_count)
            chain_errors.append(absolute_error / target_count)
            chain_within_one.append(absolute_error <= 1)
            clean_majorities.append(bool(dense["has_majority"]))
            parity.append(_exact_decode_equal(dense, compact))
            all_modes.extend(
                [dense["selection_mode"] == "multi", compact["selection_mode"] == "multi"]
            )
            clean_count_rows.append(
                {
                    "target_count": target_count,
                    "target_period_frames": target_period,
                    "phase_fraction": phase,
                    "absolute_count_error": absolute_error,
                    "dense": dense,
                }
            )

    harmonic_rows: list[dict[str, Any]] = []
    harmonic_aliases: list[bool] = []
    harmonic_majorities: list[bool] = []
    harmonic_period = 32.0
    expected_harmonic_bin = int(round(timeline_frames / harmonic_period))
    for order in synthetic["harmonic_orders"]:
        for amplitude in synthetic["harmonic_amplitudes"]:
            for phase in phases:
                stream = _harmonic_stream(
                    timeline_frames=timeline_frames,
                    period_frames=harmonic_period,
                    phase_fraction=phase,
                    harmonic_order=int(order),
                    harmonic_amplitude=float(amplitude),
                )
                mask = torch.ones(timeline_frames, dtype=torch.bool)
                dense = _decode(
                    stream,
                    mask,
                    minimum=minimum,
                    maximum=maximum,
                    timebase="dense_resampled",
                    counter=counter,
                )
                compact = _decode(
                    stream,
                    mask,
                    minimum=minimum,
                    maximum=maximum,
                    timebase="compact_valid",
                    counter=counter,
                )
                aliased = int(dense["fft_bin"]) != expected_harmonic_bin
                error = _relative_error(
                    float(dense["period_frames"]), harmonic_period
                )
                harmonic_aliases.append(aliased)
                harmonic_majorities.append(bool(dense["has_majority"]))
                period_errors.append(error)
                parity.append(_exact_decode_equal(dense, compact))
                all_modes.extend(
                    [dense["selection_mode"] == "multi", compact["selection_mode"] == "multi"]
                )
                harmonic_rows.append(
                    {
                        "harmonic_order": int(order),
                        "harmonic_amplitude": float(amplitude),
                        "phase_fraction": phase,
                        "fundamental_period_frames": harmonic_period,
                        "fundamental_alias": aliased,
                        "period_relative_error": error,
                        "dense": dense,
                    }
                )

    mask_rows: list[dict[str, Any]] = []
    payload_invariances: list[bool] = []
    for period in (16, 32):
        for phase in phases:
            base = _periodic_stream(
                timeline_frames=timeline_frames,
                period_frames=float(period),
                phase_fraction=phase,
                drift=0.15,
            )
            for mask_name, mask in _mask_variants(timeline_frames).items():
                clean = torch.where(mask, base, torch.zeros_like(base))
                polluted = _pollute_invalid(clean, mask)
                clean_dense = _decode(
                    clean,
                    mask,
                    minimum=minimum,
                    maximum=maximum,
                    timebase="dense_resampled",
                    counter=counter,
                )
                polluted_dense = _decode(
                    polluted,
                    mask,
                    minimum=minimum,
                    maximum=maximum,
                    timebase="dense_resampled",
                    counter=counter,
                )
                invariant = _exact_decode_equal(clean_dense, polluted_dense)
                error = _relative_error(
                    float(clean_dense["period_frames"]), float(period)
                )
                period_errors.append(error)
                payload_invariances.append(invariant)
                all_modes.extend(
                    [
                        clean_dense["selection_mode"] == "multi",
                        polluted_dense["selection_mode"] == "multi",
                    ]
                )
                mask_rows.append(
                    {
                        "mask": mask_name,
                        "period_frames": period,
                        "phase_fraction": phase,
                        "valid_frames": int(mask.sum()),
                        "period_relative_error": error,
                        "invalid_payload_exact_invariance": invariant,
                        "dense": clean_dense,
                    }
                )

    resample_rows: list[dict[str, Any]] = []
    resample_errors: list[float] = []
    resample_counts_equal: list[bool] = []
    for period in (4, 8, 16, 32, 64):
        for phase in phases:
            stream = _periodic_stream(
                timeline_frames=timeline_frames,
                period_frames=float(period),
                phase_fraction=phase,
            )
            mask = torch.ones(timeline_frames, dtype=torch.bool)
            repeated_stream, repeated_mask = _repeat_two_x(stream, mask)
            original = _decode(
                stream,
                mask,
                minimum=minimum,
                maximum=maximum,
                timebase="dense_resampled",
                counter=counter,
            )
            repeated = _decode(
                repeated_stream,
                repeated_mask,
                minimum=minimum,
                maximum=maximum,
                timebase="dense_resampled",
                counter=counter,
            )
            error = _relative_error(
                float(repeated["period_frames"]),
                2.0 * float(original["period_frames"]),
            )
            counts_equal = int(repeated["count"]) == int(original["count"])
            resample_errors.append(error)
            resample_counts_equal.append(counts_equal)
            all_modes.extend(
                [
                    original["selection_mode"] == "multi",
                    repeated["selection_mode"] == "multi",
                ]
            )
            resample_rows.append(
                {
                    "source_period_frames": period,
                    "phase_fraction": phase,
                    "period_scaling_relative_error": error,
                    "count_equal": counts_equal,
                    "original": original,
                    "two_x_repeated": repeated,
                }
            )

    period_error_array = np.asarray(period_errors, dtype=np.float64)
    metrics: dict[str, Any] = {
        "existing_576_counter_gate_passed": bool(counter_gate["passed"]),
        "existing_576_counter_gate_case_count": int(
            counter_gate["metrics"]["case_count"]
        ),
        "legacy_all_valid_exact_parity": all(parity),
        "all_valid_parity_case_count": len(parity),
        "period_case_count": len(period_errors),
        "median_period_relative_error": float(np.median(period_error_array)),
        "p95_period_relative_error": float(
            np.quantile(period_error_array, 0.95, method="linear")
        ),
        "harmonic_alias_fraction": _fraction(harmonic_aliases),
        "harmonic_case_count": len(harmonic_aliases),
        "chain_nmae": float(np.mean(np.asarray(chain_errors, dtype=np.float64))),
        "chain_obo": _fraction(chain_within_one),
        "chain_case_count": len(chain_errors),
        "clean_majority_fraction": _fraction(clean_majorities),
        "harmonic_majority_fraction": _fraction(harmonic_majorities),
        "mask_case_count": len(mask_rows),
        "resample_period_relative_error_maximum": max(resample_errors),
        "resample_count_equal_fraction": _fraction(resample_counts_equal),
        "resample_case_count": len(resample_rows),
        "invalid_payload_exact_invariance": all(payload_invariances),
        "invalid_payload_case_count": len(payload_invariances),
        "all_selected_modes_multi": all(all_modes),
    }
    checks = _evaluate_checks(metrics, thresholds)
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "artifact_type": _ARTIFACT_TYPE,
        "candidate_id": policy["candidate_id"],
        "classification": policy["classification"],
        "status": "passed" if passed else "failed",
        "passed": passed,
        "table2_eligible": False,
        "labels_accessed": False,
        "dataset_inputs_accessed": False,
        "checkpoint_inputs_accessed": False,
        "dev84_inputs_accessed": False,
        "test105_inputs_accessed": False,
        "source": {
            "runner_source_git_sha": source_git_sha,
            "runner_sha256": runner_sha256,
            "policy_sha256": policy_sha256,
            "policy_semantic_sha256": policy_semantic_sha256,
        },
        "configuration": {
            "timeline_frames": timeline_frames,
            "direct_fft_timebase": "dense_resampled",
            "comparison_timebase": "compact_valid",
            "consensus_expert_mode": counter.expert_mode,
            "synthetic_periods": list(periods),
            "synthetic_count_range_inclusive": [count_min, count_max],
            "synthetic_phase_fractions": list(phases),
            "parameter_sweep": False,
            "candidate_count": 1,
        },
        "thresholds": thresholds,
        "metrics": metrics,
        "checks": checks,
        "existing_counter_gate": counter_gate_summary,
        "audit_rows": {
            "clean_period": clean_period_rows,
            "clean_count": clean_count_rows,
            "harmonic": harmonic_rows,
            "masked": mask_rows,
            "two_x_repeated_resample": resample_rows,
        },
        "authorization": {
            "train337_gate_authorized": passed,
            "dev84_prediction_authorized": False,
            "dev84_scoring_authorized": False,
            "test105_evaluation_authorized": False,
        },
    }


def _receipt_path(output: Path) -> Path:
    return Path(str(output) + ".receipt.json")


def _write_new_regular_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with suppress(OSError):
            path.unlink()
        raise


def _write_artifact_and_receipt(
    output: str | Path,
    payload: Mapping[str, Any],
) -> tuple[Path, str]:
    destination = Path(output)
    receipt_path = _receipt_path(destination)
    if destination.exists() or receipt_path.exists():
        raise FileExistsError("synthetic report and receipt destinations must both be new")
    artifact = _encoded_json(payload)
    artifact_sha256 = hashlib.sha256(artifact).hexdigest()
    receipt = {
        "schema_version": 1,
        "artifact_type": _RECEIPT_TYPE,
        "artifact_locator": destination.name,
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": len(artifact),
        "artifact_status": payload["status"],
        "candidate_id": payload["candidate_id"],
        "policy_sha256": payload["source"]["policy_sha256"],
        "policy_semantic_sha256": payload["source"]["policy_semantic_sha256"],
        "runner_source_git_sha": payload["source"]["runner_source_git_sha"],
        "runner_sha256": payload["source"]["runner_sha256"],
        "labels_accessed": False,
        "dataset_inputs_accessed": False,
        "checkpoint_inputs_accessed": False,
        "train337_gate_authorized": payload["authorization"][
            "train337_gate_authorized"
        ],
        "dev84_prediction_authorized": False,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }
    _write_new_regular_file(destination, artifact)
    try:
        _write_new_regular_file(receipt_path, _encoded_json(receipt))
    except BaseException:
        if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == (
            artifact_sha256
        ):
            destination.unlink()
        raise
    return receipt_path, artifact_sha256


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    synthetic = subcommands.add_parser(
        "synthetic",
        help="run the deterministic synthetic-only dense-timebase gate",
    )
    synthetic.add_argument("--policy", type=Path, required=True)
    synthetic.add_argument("--output", type=Path, required=True)
    synthetic.add_argument("--repository-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    if arguments.command != "synthetic":
        raise AssertionError("unreachable synthetic-only subcommand")
    # Source cleanliness is checked before any output directory is created.
    source_git_sha = clean_git_revision(arguments.repository_root)
    runner_path = Path(__file__).resolve()
    runner_sha256 = sha256_file(runner_path)
    payload = run_synthetic_gate(
        arguments.policy,
        source_git_sha=source_git_sha,
        runner_sha256=runner_sha256,
    )
    # Recheck both immutable source inputs after the potentially long suite.
    if clean_git_revision(arguments.repository_root) != source_git_sha:
        raise RuntimeError("Git source revision changed during synthetic gate")
    if sha256_file(runner_path) != runner_sha256:
        raise RuntimeError("synthetic gate runner changed during execution")
    receipt_path, artifact_sha256 = _write_artifact_and_receipt(
        arguments.output,
        payload,
    )
    print(
        json.dumps(
            {
                "artifact_sha256": artifact_sha256,
                "output": str(arguments.output),
                "receipt": str(receipt_path),
                "status": payload["status"],
                "train337_gate_authorized": payload["authorization"][
                    "train337_gate_authorized"
                ],
                "dev84_prediction_authorized": False,
                "dev84_scoring_authorized": False,
                "test105_evaluation_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0 if payload["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
