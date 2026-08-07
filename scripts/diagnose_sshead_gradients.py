"""Read-only, label-free SSHead loss and gradient diagnosis.

This script deliberately accepts only the canonical 337-row training pose
sidecar.  It never loads counts, actions, development targets, or test data.
It compares the random head embedded in an encoder checkpoint with the
trained head embedded in its bound SSHead checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from pams.config import PAMSConfig, load_config
from pams.data import load_pose_cache_set, load_pose_input_manifest
from pams.losses import SSHeadLoss
from pams.period import estimate_period_batch
from pams.training import (
    _estimate_post_warmup_periods,
    build_pams_model,
    collate_pose_sequences,
    validate_sshead_encoder_binding,
)

_COMPONENTS = ("cycle", "spectral", "variance", "smoothness", "total")
_SCALES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_raw_model(
    checkpoint: Path,
    config: PAMSConfig,
    *,
    expected_stage: str,
    device: torch.device,
) -> nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("stage") != expected_stage:
        raise ValueError(f"expected a {expected_stage!r} checkpoint")
    if payload.get("config_fingerprint") != config.fingerprint:
        raise ValueError("checkpoint/config fingerprint mismatch")
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("checkpoint model_state must be a mapping")
    model = build_pams_model(config)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def _zeros(parameters: tuple[Tensor, ...]) -> list[Tensor]:
    return [
        torch.zeros_like(parameter, dtype=torch.float64, device="cpu")
        for parameter in parameters
    ]


def _add_gradients(
    destination: list[Tensor],
    gradients: tuple[Tensor | None, ...],
    factor: float,
) -> None:
    for output, gradient in zip(destination, gradients, strict=True):
        if gradient is not None:
            output.add_(
                gradient.detach().to(device="cpu", dtype=torch.float64),
                alpha=factor,
            )


def _gradient_stats(gradients: list[Tensor]) -> dict[str, float]:
    squared = sum(float(gradient.square().sum()) for gradient in gradients)
    count = sum(gradient.numel() for gradient in gradients)
    return {
        "l2": math.sqrt(squared),
        "rms": math.sqrt(squared / max(count, 1)),
    }


def _cosine(first: list[Tensor], second: list[Tensor]) -> float | None:
    dot = sum(
        float(left.mul(right).sum())
        for left, right in zip(first, second, strict=True)
    )
    first_norm = math.sqrt(sum(float(value.square().sum()) for value in first))
    second_norm = math.sqrt(sum(float(value.square().sum()) for value in second))
    if first_norm == 0.0 or second_norm == 0.0:
        return None
    return dot / (first_norm * second_norm)


def _parameter_change(
    initial_head: nn.Module,
    final_head: nn.Module,
) -> dict[str, Any]:
    initial = dict(initial_head.named_parameters())
    final = dict(final_head.named_parameters())
    if initial.keys() != final.keys():
        raise ValueError("initial and final head parameter keys differ")
    rows: dict[str, Any] = {}
    total_initial = 0.0
    total_final = 0.0
    total_delta = 0.0
    for name in initial:
        before = initial[name].detach().float().cpu()
        after = final[name].detach().float().cpu()
        delta = after - before
        initial_squared = float(before.square().sum())
        final_squared = float(after.square().sum())
        delta_squared = float(delta.square().sum())
        total_initial += initial_squared
        total_final += final_squared
        total_delta += delta_squared
        rows[name] = {
            "initial_l2": math.sqrt(initial_squared),
            "final_l2": math.sqrt(final_squared),
            "delta_l2": math.sqrt(delta_squared),
        }
    return {
        "per_parameter": rows,
        "all_parameters": {
            "initial_l2": math.sqrt(total_initial),
            "final_l2": math.sqrt(total_final),
            "delta_l2": math.sqrt(total_delta),
            "delta_over_initial": (
                math.sqrt(total_delta / total_initial) if total_initial else None
            ),
        },
    }


def _select_head_inputs(
    input_source: str,
    embeddings: Tensor,
    projected_pose: Tensor | None,
) -> Tensor:
    if input_source == "encoder_embedding":
        return embeddings
    if input_source == "projected_pose_pre_pe":
        if projected_pose is None:
            raise RuntimeError("projected-pose SSHead input was not materialized")
        return projected_pose
    raise AssertionError(
        f"unreachable validated SSHead input source: {input_source!r}"
    )


def _constant_stream_proof(device: torch.device) -> dict[str, Any]:
    stream = torch.zeros((1, 256), device=device, requires_grad=True)
    details = SSHeadLoss().compute(
        stream,
        torch.tensor([16.0], device=device),
    )
    result: dict[str, Any] = {}
    for name in _COMPONENTS:
        value = getattr(details, name)
        gradient = torch.autograd.grad(value, stream, retain_graph=True)[0]
        result[name] = {
            "value": float(value.detach()),
            "stream_grad_l2": float(gradient.norm()),
            "nonzero_gradient_elements": int(torch.count_nonzero(gradient)),
        }
    return result


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    config = load_config(args.config)
    manifest = load_pose_input_manifest(args.train_inputs, validate_exact=True)
    if manifest.protocol != "ucfrep_526" or manifest.split != "train":
        raise ValueError("diagnosis accepts only the UCFRep-526 train sidecar")
    if len(manifest.records) != 337:
        raise ValueError("diagnosis requires exactly the canonical train337 rows")

    sequences, snapshot = load_pose_cache_set(
        manifest.records,
        cache_dir=args.pose_cache_dir,
        pose_fingerprint=config.pose_fingerprint,
    )
    validate_sshead_encoder_binding(args.sshead_checkpoint, args.encoder_checkpoint)
    initial_model = _load_raw_model(
        args.encoder_checkpoint,
        config,
        expected_stage="encoder",
        device=torch.device("cpu"),
    )
    final_model = _load_raw_model(
        args.sshead_checkpoint,
        config,
        expected_stage="sshead",
        device=device,
    )
    initial_model.period_head.to(device).train()
    final_model.period_head.train()
    final_model.encoder.eval()
    for parameter in final_model.encoder.parameters():
        parameter.requires_grad_(False)

    heads = {
        "initial": initial_model.period_head,
        "final": final_model.period_head,
    }
    parameters = {
        name: tuple(head.parameters())
        for name, head in heads.items()
    }
    weights = {
        "cycle": config.sshead.cycle_weight,
        "spectral": config.sshead.spectral_weight,
        "variance": config.sshead.variance_weight,
        "smoothness": config.sshead.smoothness_weight,
        "total": 1.0,
    }
    accumulated = {
        head_name: {
            component: _zeros(parameters[head_name])
            for component in _COMPONENTS
        }
        for head_name in heads
    }
    values = {
        head_name: {component: 0.0 for component in _COMPONENTS}
        for head_name in heads
    }
    scale_derivatives = {
        head_name: {component: 0.0 for component in _COMPONENTS}
        for head_name in heads
    }
    stream_stds: dict[str, list[float]] = {name: [] for name in heads}
    estimated_periods: dict[str, list[float]] = {name: [] for name in heads}
    fully_valid_streams: dict[str, list[Tensor]] = {name: [] for name in heads}
    scale_sweep = {
        head_name: {str(scale): 0.0 for scale in _SCALES}
        for head_name in heads
    }
    target_periods: list[float] = []
    confidences: list[float] = []
    objective = SSHeadLoss(
        cycle_weight=config.sshead.cycle_weight,
        spectral_weight=config.sshead.spectral_weight,
        variance_weight=config.sshead.variance_weight,
        smoothness_weight=config.sshead.smoothness_weight,
    )

    sample_count = len(sequences)
    for start in range(0, sample_count, args.batch_size):
        batch = collate_pose_sequences(
            sequences[start : start + args.batch_size]
        ).to(device)
        batch_fraction = batch.batch_size / sample_count
        with torch.no_grad():
            needs_projected_pose = (
                config.sshead.input_source == "projected_pose_pre_pe"
                or (
                    config.period.training_mode == "adaptive"
                    and config.period.post_warmup_source
                    == "projected_pose_velocity_vector_acf"
                )
            )
            if needs_projected_pose:
                embeddings, projected_pose = final_model.encoder.forward_with_pre_pe(
                    batch.poses,
                    batch.valid_mask,
                )
            else:
                embeddings = final_model.encoder(batch.poses, batch.valid_mask)
                projected_pose = None
            periods, period_confidence, _ = _estimate_post_warmup_periods(
                config=config,
                embeddings=embeddings,
                projected_pose=projected_pose,
                valid_mask=batch.valid_mask,
                timeline_lengths=batch.lengths,
            )
        target_periods.extend(periods.cpu().tolist())
        confidences.extend(period_confidence.cpu().tolist())

        head_inputs = _select_head_inputs(
            config.sshead.input_source,
            embeddings,
            projected_pose,
        )
        for head_name, head in heads.items():
            stream = head(head_inputs.detach()).masked_fill(~batch.valid_mask, 0.0)
            details = objective.compute(
                stream,
                periods,
                batch.valid_mask,
                period_confidence=period_confidence,
            )
            components = {
                "cycle": details.cycle,
                "spectral": details.spectral,
                "variance": details.variance,
                "smoothness": details.smoothness,
                "total": details.total,
            }
            for component, value in components.items():
                weighted = (
                    value
                    if component == "total"
                    else value * weights[component]
                )
                values[head_name][component] += (
                    float(value.detach()) * batch_fraction
                )
                gradients = torch.autograd.grad(
                    weighted,
                    parameters[head_name],
                    retain_graph=True,
                    allow_unused=True,
                )
                _add_gradients(
                    accumulated[head_name][component],
                    gradients,
                    batch_fraction,
                )
                stream_gradient = torch.autograd.grad(
                    weighted,
                    stream,
                    retain_graph=True,
                )[0]
                scale_derivatives[head_name][component] += (
                    float((stream_gradient * stream).sum().detach())
                    * batch_fraction
                )

            with torch.no_grad():
                for sample_stream, sample_valid in zip(
                    stream,
                    batch.valid_mask,
                    strict=True,
                ):
                    selected = sample_stream[sample_valid]
                    stream_stds[head_name].append(
                        float(selected.std(unbiased=False))
                        if selected.numel() >= 2
                        else 0.0
                    )
                    if bool(sample_valid.all()):
                        centered = sample_stream - sample_stream.mean()
                        norm = centered.norm()
                        if float(norm) > 0.0:
                            fully_valid_streams[head_name].append(
                                (centered / norm).cpu()
                            )
                head_periods, _ = estimate_period_batch(
                    stream.detach(),
                    minimum=config.period.minimum,
                    maximum=config.period.maximum,
                    valid_mask=batch.valid_mask,
                )
                estimated_periods[head_name].extend(head_periods.cpu().tolist())
                for scale in _SCALES:
                    scaled = objective.compute(
                        stream.detach() * scale,
                        periods,
                        batch.valid_mask,
                        period_confidence=period_confidence,
                    )
                    scale_sweep[head_name][str(scale)] += (
                        float(scaled.total) * batch_fraction
                    )

    head_results: dict[str, Any] = {}
    for head_name in heads:
        deviations = np.asarray(stream_stds[head_name], dtype=np.float64)
        head_period_array = np.asarray(
            estimated_periods[head_name],
            dtype=np.float64,
        )
        total_gradient = accumulated[head_name]["total"]
        component_results: dict[str, Any] = {}
        for component in _COMPONENTS:
            component_results[component] = {
                "unweighted_value": values[head_name][component],
                "effective_weight": weights[component],
                "parameter_gradient": _gradient_stats(
                    accumulated[head_name][component]
                ),
                "cosine_with_total_parameter_gradient": _cosine(
                    accumulated[head_name][component],
                    total_gradient,
                ),
                "log_scale_directional_derivative": (
                    scale_derivatives[head_name][component]
                ),
            }

        correlation: dict[str, Any] | None = None
        full_streams = fully_valid_streams[head_name]
        if len(full_streams) >= 2:
            matrix = torch.stack(full_streams)
            upper = (matrix @ matrix.T)[
                torch.triu(
                    torch.ones(
                        (len(matrix), len(matrix)),
                        dtype=torch.bool,
                    ),
                    diagonal=1,
                )
            ]
            correlation = {
                "video_count": len(matrix),
                "median": float(upper.median()),
                "mean": float(upper.mean()),
            }
        head_results[head_name] = {
            "loss_and_gradient": component_results,
            "stream_standard_deviation": {
                "mean": float(deviations.mean()),
                "median": float(np.median(deviations)),
                "p10": float(np.percentile(deviations, 10.0)),
                "zero_or_fewer_than_two_valid_frames": int(
                    np.count_nonzero(deviations == 0.0)
                ),
            },
            "head_stream_period": {
                "median": float(np.median(head_period_array)),
                "period_8_count": int(
                    np.count_nonzero(np.isclose(head_period_array, 8.0))
                ),
                "period_16_count": int(
                    np.count_nonzero(np.isclose(head_period_array, 16.0))
                ),
            },
            "fully_valid_centered_stream_correlation": correlation,
            "scale_sweep_total_loss": scale_sweep[head_name],
        }

    confidence_array = np.asarray(confidences, dtype=np.float64)
    target_period_array = np.asarray(target_periods, dtype=np.float64)
    result = {
        "schema_version": 1,
        "classification": "label-free train337-only read-only SSHead gradient diagnosis",
        "dev_targets_loaded": False,
        "test105_accessed": False,
        "config_fingerprint": config.fingerprint,
        "encoder_checkpoint_sha256": _sha256_file(args.encoder_checkpoint),
        "sshead_checkpoint_sha256": _sha256_file(args.sshead_checkpoint),
        "train_sidecar_sha256": _sha256_file(args.train_inputs),
        "train_rows": sample_count,
        "pose_cache_set_sha256": snapshot.fingerprint,
        "training_mode": config.period.training_mode,
        "post_warmup_source": config.period.post_warmup_source,
        "sshead_input_source": config.sshead.input_source,
        "period_confidence_mean": float(confidence_array.mean()),
        "period_valid_fraction": float(
            np.mean(confidence_array != 0.0)
        ),
        "training_target_period": {
            "p10": float(np.percentile(target_period_array, 10.0)),
            "median": float(np.median(target_period_array)),
            "p90": float(np.percentile(target_period_array, 90.0)),
        },
        "constant_stream_stationary_point": _constant_stream_proof(device),
        "head_parameter_change": _parameter_change(
            initial_model.period_head,
            final_model.period_head,
        ),
        "heads": head_results,
    }
    del initial_model, final_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--sshead-checkpoint", type=Path, required=True)
    parser.add_argument("--train-inputs", type=Path, required=True)
    parser.add_argument("--pose-cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    arguments = parser.parse_args()
    if arguments.batch_size < 1:
        parser.error("--batch-size must be positive")
    return arguments


if __name__ == "__main__":
    print(
        json.dumps(
            diagnose(_parse_args()),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
