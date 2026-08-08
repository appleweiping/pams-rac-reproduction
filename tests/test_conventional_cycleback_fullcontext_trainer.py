from __future__ import annotations

import copy
import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import pams.conventional_cycleback.fullcontext_authority as authority_module
from pams.conventional_cycleback.config import (
    ConventionalCycleBackConfig,
    load_conventional_cycleback_config,
)
from pams.conventional_cycleback.fullcontext_authority import (
    authority_scaffold_status,
    validate_fullcontext_predecessor_authority,
)
from pams.conventional_cycleback.fullcontext_trainer import (
    Epoch11GateThresholds,
    FullContextLineage,
    FullContextProgress,
    FullContextRepresentationContract,
    FullContextTrainerContract,
    FullContextTrainingUnit,
    advance_fullcontext_progress,
    atomic_save_fullcontext_checkpoint,
    build_fullcontext_adamw,
    build_fullcontext_checkpoint_payload,
    build_length_bucket_plan,
    epoch11_gate_decision,
    initial_epoch11_progress,
    initial_epoch150_progress,
    initialize_epoch150_from_exact_epoch11_checkpoint,
    load_fullcontext_checkpoint,
    model_state_sha256,
    optimize_real_pair_contexts,
    validate_fullcontext_checkpoint_payload,
    validate_mechanism_seed_checkpoint_payload,
)
from pams.conventional_cycleback.loss import ConventionalCycleBackLoss
from pams.conventional_cycleback.runtime import (
    AugmentedSequenceViews,
    context_encoding_plan,
    encode_window_pairs,
    pair_segment_contexts,
    zero_pair_segment_contexts,
)
from pams.conventional_cycleback.windows import enumerate_native_window_pairs
from pams.model import PAMSEncoder

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/conventional_cycleback/w16_hop4_v1.yaml"


def _candidate_config(*, tiny: bool = False) -> ConventionalCycleBackConfig:
    config = load_conventional_cycleback_config(CONFIG)
    if not tiny:
        return config
    encoder = config.encoder.model_copy(
        update={
            "model_dim": 8,
            "embedding_dim": 8,
            "layers": 1,
            "heads": 2,
            "feedforward_dim": 16,
            "dropout": 0.25,
            "maximum_native_length": 128,
        }
    )
    return config.model_copy(update={"encoder": encoder})


def _contract(*, tiny: bool = False) -> FullContextTrainerContract:
    return FullContextTrainerContract.from_candidate_config(
        _candidate_config(tiny=tiny)
    )


def _lineage() -> FullContextLineage:
    return FullContextLineage(
        source_git_sha="1" * 40,
        source_tree_sha256="2" * 64,
        container_image_id=f"sha256:{'3' * 64}",
        config_sha256="4" * 64,
        config_bytes=101,
        representation_integration_outcome_sha256="5" * 64,
        representation_integration_outcome_bytes=102,
        representation_contract_sha256="c" * 64,
        representation_authorization_sha256="6" * 64,
        representation_authorization_bytes=103,
        mechanism_outcome_sha256="7" * 64,
        mechanism_outcome_bytes=104,
        mechanism_run_receipt_sha256="8" * 64,
        mechanism_run_receipt_bytes=105,
        mechanism_seed_checkpoint_sha256="9" * 64,
        mechanism_seed_checkpoint_bytes=106,
        mechanism_learned_model_state_sha256="a" * 64,
    )


def _unit(
    video_id: str,
    *,
    length: int,
    starts: tuple[int, ...] = (0,),
) -> FullContextTrainingUnit:
    return FullContextTrainingUnit(
        video_id=video_id,
        native_length=length,
        authorized_ranges=((0, length),),
        authorized_pair_starts=starts,
        window_frames=16,
        hop_frames=4,
    )


def _plan(*, epoch: int = 1, tiny: bool = False):
    contract = _contract(tiny=tiny)
    return build_length_bucket_plan(
        (
            _unit("opaque-video-a", length=40, starts=(0, 4, 8)),
            _unit("opaque-video-b", length=80, starts=(0, 4, 8, 12, 16)),
        ),
        contract,
        _lineage(),
        epoch=epoch,
    )


def _pair_inputs(
    lengths: tuple[int, ...] = (48,),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = max(lengths)
    batch = len(lengths)
    first = torch.zeros((batch, maximum, 33, 3), dtype=torch.float32)
    second = torch.zeros_like(first)
    valid = torch.zeros((batch, maximum), dtype=torch.bool)
    joint_valid = torch.zeros((batch, maximum, 17), dtype=torch.bool)
    for row, length in enumerate(lengths):
        frame = torch.arange(length, dtype=torch.float32).view(length, 1, 1)
        first[row, :length, :17, :2] = frame + 1.0 + row
        second[row, :length, :17, :2] = frame + 2.0 + row
        valid[row, :length] = True
        joint_valid[row, :length] = True
    return (
        first,
        second,
        valid,
        joint_valid,
        torch.tensor(lengths, dtype=torch.long),
    )


def _pairs_and_contexts(
    lengths: tuple[int, ...] = (48,),
):
    first, second, valid, joint_valid, native_lengths = _pair_inputs(lengths)
    video_ids = tuple(f"video-{index}" for index in range(len(lengths)))
    segments = tuple(((0, length),) for length in lengths)
    starts = tuple(tuple(range(0, length - 31, 4)) for length in lengths)
    pairs = enumerate_native_window_pairs(
        first,
        second,
        valid,
        joint_valid,
        native_lengths,
        video_ids,
        segments,
        starts,
        starts,
        window_length=16,
        hop_frames=4,
        minimum_valid_frames_per_window=16,
        minimum_window_stable_action_joints=6,
        minimum_window_joint_support_fraction=0.75,
    )
    views = AugmentedSequenceViews(
        video_ids=video_ids,
        poses_a=first,
        poses_b=second,
        valid_mask=valid,
        joint_valid_mask=joint_valid,
        lengths=native_lengths,
        view_seeds=(11, 13),
    )
    return pairs, pair_segment_contexts(pairs, views)


class _DropoutSpyEncoder(torch.nn.Module):
    embedding_dim = 3

    def __init__(self) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.calls: list[tuple[int, int]] = []

    def forward(
        self,
        poses: torch.Tensor,
        valid: torch.Tensor,
        *,
        position_indices: torch.Tensor,
    ) -> torch.Tensor:
        del valid
        self.calls.append((poses.shape[0], poses.shape[1]))
        source = position_indices.to(dtype=poses.dtype).unsqueeze(-1) + 1.0
        return self.dropout(source.expand(-1, -1, self.embedding_dim))


def test_context_plan_is_pair_order_invariant_and_exact_length_bucketed() -> None:
    pairs, contexts = _pairs_and_contexts((48, 56))
    reverse = torch.arange(pairs.pair_count - 1, -1, -1)
    reversed_pairs = pairs.select(reverse)
    first, second, valid, joint_valid, native_lengths = _pair_inputs((48, 56))
    reversed_contexts = pair_segment_contexts(
        reversed_pairs,
        AugmentedSequenceViews(
            video_ids=("video-0", "video-1"),
            poses_a=first,
            poses_b=second,
            valid_mask=valid,
            joint_valid_mask=joint_valid,
            lengths=native_lengths,
            view_seeds=(11, 13),
        ),
    )

    original_plan = context_encoding_plan(contexts, batch_size=8)
    reordered_plan = context_encoding_plan(reversed_contexts, batch_size=8)

    assert original_plan["fingerprint"] == reordered_plan["fingerprint"]
    assert original_plan["mixed_context_lengths_within_batch"] is False
    for side in ("a", "b"):
        for batch in original_plan["side_batches"][side]:
            assert len(
                {original_plan["context_lengths"][key] for key in batch}
            ) == 1


def test_dropout_is_shared_for_every_pair_reusing_one_full_range_view() -> None:
    torch.manual_seed(2026)
    pairs, contexts = _pairs_and_contexts((48,))
    encoder = _DropoutSpyEncoder().train()

    embeddings_a, embeddings_b = encode_window_pairs(
        encoder,
        pairs,
        contexts,
        batch_size=8,
    )

    assert encoder.calls == [(1, 48), (1, 48)]
    assert torch.equal(embeddings_a[0, 4:16], embeddings_a[1, :12])
    assert torch.equal(embeddings_b[0, 4:16], embeddings_b[1, :12])
    assert contexts.transform_contract["same_view_same_range_encoded_once"] is True


def test_duplicate_context_key_rejects_pose_pe_or_joint_identity_drift() -> None:
    _, contexts = _pairs_and_contexts((48,))
    poses = list(contexts.poses_a)
    positions = list(contexts.position_indices_a)
    joints = list(contexts.joint_valid_a)
    poses[1] = poses[1].clone()
    poses[1][0, 0, 0] += 1.0
    positions[1] = positions[1].clone()
    positions[1][0] += 1
    joints[1] = joints[1].clone()
    joints[1][0, 0] = ~joints[1][0, 0]
    variants = (
        replace(contexts, poses_a=tuple(poses)),
        replace(contexts, position_indices_a=tuple(positions)),
        replace(contexts, joint_valid_a=tuple(joints)),
    )

    for altered in variants:
        with pytest.raises(ValueError, match="different pose/PE/joint bytes"):
            context_encoding_plan(altered, batch_size=8)


def test_length_bucket_plan_is_input_order_independent_and_keeps_long_singletons() -> None:
    contract = _contract()
    lineage = _lineage()
    units = (
        _unit("opaque-long", length=3000, starts=(0,)),
        _unit("opaque-short-a", length=40, starts=(0, 4)),
        _unit("opaque-short-b", length=44, starts=(0, 4, 8)),
    )

    first = build_length_bucket_plan(units, contract, lineage, epoch=7)
    second = build_length_bucket_plan(tuple(reversed(units)), contract, lineage, epoch=7)

    assert first.fingerprint == second.fingerprint
    observed = [video_id for batch in first.batches for video_id in batch.video_ids]
    assert set(observed) == {unit.video_id for unit in units}
    assert len(observed) == len(set(observed))
    long_batch = next(batch for batch in first.batches if "opaque-long" in batch.video_ids)
    assert long_batch.video_ids == ("opaque-long",)
    assert long_batch.context_frames == 3000
    with pytest.raises(ValueError, match="positional capacity"):
        build_length_bucket_plan(
            (_unit("opaque-too-long", length=5000, starts=(0,)),),
            contract,
            lineage,
            epoch=7,
        )


def test_representation_contract_types_coco17_and_mediapipe33_without_aliasing() -> None:
    unified2d = FullContextRepresentationContract(
        representation_family="v4e_unified2d_coco17_padded33",
        coordinate_contract="body-centered-uniform-rms-scale-xy-z0-v1",
        pose_joint_count=33,
        support_channel_count=17,
        support_channel_to_pose_joint_indices=tuple(range(17)),
        augmentation_adapter="unified2d_masked_xy_inplane_v1",
        diagnostic_adapter="unified2d_coco17_joint_nulls_v1",
        stable_range_policy="v4e_exact_all_valid_track_stability_ranges_v1",
    )
    mediapipe = FullContextRepresentationContract(
        representation_family="v4a_mediapipe33_single_source",
        coordinate_contract="v4a_sealed_coordinate_contract",
        pose_joint_count=33,
        support_channel_count=33,
        support_channel_to_pose_joint_indices=tuple(range(33)),
        augmentation_adapter="future_v4a_mediapipe33_adapter_not_activated",
        diagnostic_adapter="future_v4a_mediapipe33_joint_nulls_not_activated",
        stable_range_policy="v4a_exact_contiguous_valid_ranges_v1",
    )

    assert unified2d.fingerprint != mediapipe.fingerprint
    assert unified2d.support_channel_to_pose_joint_indices == tuple(range(17))
    assert mediapipe.support_channel_count == 33


def test_progress_binds_mid_epoch_cursor_and_exact_epoch11_prefix() -> None:
    plan = _plan(epoch=1)
    progress = initial_epoch11_progress(plan)
    advanced = advance_fullcontext_progress(progress)

    assert advanced.completed_epochs == 0
    assert advanced.sampler_cursor == 1
    assert advanced.global_optimizer_step == 257

    epoch12 = _plan(epoch=12)
    continuation = initial_epoch150_progress(
        epoch12,
        epoch11_checkpoint_sha256="b" * 64,
        epoch11_checkpoint_bytes=1234,
        epoch11_continuation_optimizer_steps=99,
    )
    assert continuation.completed_epochs == 11
    assert continuation.epoch11_prefix_checkpoint_sha256 == "b" * 64
    with pytest.raises(ValueError, match="epoch-twelve"):
        initial_epoch150_progress(
            plan,
            epoch11_checkpoint_sha256="b" * 64,
            epoch11_checkpoint_bytes=1234,
            epoch11_continuation_optimizer_steps=99,
        )


def _random_optimizer_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float, torch.Tensor]:
    python_value = random.random()
    numpy_value = float(np.random.random())
    inputs = torch.rand((4, 4))
    optimizer.zero_grad(set_to_none=True)
    output = model(inputs).square().sum() * (python_value + numpy_value + 1.0)
    output.backward()
    optimizer.step()
    return python_value, numpy_value, inputs


def _prime_adamw_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
) -> None:
    for parameter in model.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(float(step)),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }


def _assert_nested_equal(first: Any, second: Any) -> None:
    if isinstance(first, torch.Tensor):
        assert isinstance(second, torch.Tensor)
        assert torch.equal(first, second)
    elif isinstance(first, Mapping):
        assert isinstance(second, Mapping)
        assert set(first) == set(second)
        for key in first:
            _assert_nested_equal(first[key], second[key])
    elif isinstance(first, Sequence) and not isinstance(first, str | bytes):
        assert isinstance(second, Sequence)
        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            _assert_nested_equal(left, right)
    else:
        assert first == second


def test_checkpoint_resume_restores_bitwise_model_optimizer_rng_and_view_state(
    tmp_path: Path,
) -> None:
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    contract = _contract()
    lineage = _lineage()
    progress = initial_epoch11_progress(_plan(epoch=1))
    model = torch.nn.Linear(4, 3)
    optimizer = build_fullcontext_adamw(model, contract)
    _prime_adamw_state(model, optimizer, step=256)
    _random_optimizer_update(model, optimizer)
    progress = advance_fullcontext_progress(progress)
    payload = build_fullcontext_checkpoint_payload(
        model,
        optimizer,
        contract,
        lineage,
        progress,
    )
    checkpoint = tmp_path / "fullcontext.pt"
    identity = atomic_save_fullcontext_checkpoint(payload, checkpoint)

    expected_draws = _random_optimizer_update(model, optimizer)
    expected_model = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())

    resumed_model = torch.nn.Linear(4, 3)
    resumed_optimizer = build_fullcontext_adamw(resumed_model, contract)
    restored = load_fullcontext_checkpoint(
        checkpoint,
        expected_sha256=identity[0],
        expected_bytes=identity[1],
        contract=contract,
        lineage=lineage,
        expected_phase="epoch11",
        model=resumed_model,
        optimizer=resumed_optimizer,
    )
    resumed_draws = _random_optimizer_update(resumed_model, resumed_optimizer)

    assert restored.to_dict() == progress.to_dict()
    assert resumed_draws[0] == expected_draws[0]
    assert resumed_draws[1] == expected_draws[1]
    assert torch.equal(resumed_draws[2], expected_draws[2])
    _assert_nested_equal(resumed_model.state_dict(), expected_model)
    _assert_nested_equal(resumed_optimizer.state_dict(), expected_optimizer)


def test_checkpoint_rejects_lineage_and_cursor_tampering() -> None:
    contract = _contract()
    lineage = _lineage()
    model = torch.nn.Linear(4, 3)
    optimizer = build_fullcontext_adamw(model, contract)
    _prime_adamw_state(model, optimizer, step=256)
    payload = build_fullcontext_checkpoint_payload(
        model,
        optimizer,
        contract,
        lineage,
        initial_epoch11_progress(_plan(epoch=1)),
    )

    wrong_lineage = copy.deepcopy(payload)
    wrong_lineage["lineage"]["mechanism_outcome_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="contract or lineage"):
        validate_fullcontext_checkpoint_payload(
            wrong_lineage,
            contract,
            lineage,
            expected_phase="epoch11",
        )

    wrong_cursor = copy.deepcopy(payload)
    wrong_cursor["progress"]["sampler_cursor"] = 10_000
    with pytest.raises(ValueError, match="sampler cursor"):
        validate_fullcontext_checkpoint_payload(
            wrong_cursor,
            contract,
            lineage,
            expected_phase="epoch11",
        )


def test_epoch150_initialization_restores_exact_completed_epoch11_prefix(
    tmp_path: Path,
) -> None:
    contract = _contract()
    lineage = _lineage()
    model = torch.nn.Linear(4, 3)
    optimizer = build_fullcontext_adamw(model, contract)
    _prime_adamw_state(model, optimizer, step=278)
    completed = FullContextProgress(
        phase="epoch11",
        target_epoch=11,
        completed_epochs=11,
        continuation_optimizer_steps=22,
        active_plan=None,
        sampler_cursor=0,
    )
    checkpoint = tmp_path / "epoch11.pt"
    identity = atomic_save_fullcontext_checkpoint(
        build_fullcontext_checkpoint_payload(
            model,
            optimizer,
            contract,
            lineage,
            completed,
        ),
        checkpoint,
    )
    restored_model = torch.nn.Linear(4, 3)
    restored_optimizer = build_fullcontext_adamw(restored_model, contract)

    epoch150 = initialize_epoch150_from_exact_epoch11_checkpoint(
        checkpoint,
        expected_sha256=identity[0],
        expected_bytes=identity[1],
        contract=contract,
        lineage=lineage,
        epoch12_plan=_plan(epoch=12),
        model=restored_model,
        optimizer=restored_optimizer,
    )

    assert model_state_sha256(restored_model) == model_state_sha256(model)
    assert epoch150.completed_epochs == 11
    assert epoch150.active_plan is not None
    assert epoch150.active_plan.epoch == 12
    assert epoch150.epoch11_prefix_checkpoint_sha256 == identity[0]
    assert epoch150.epoch11_prefix_checkpoint_bytes == identity[1]


def test_json_only_mechanism_digest_cannot_substitute_for_learned_l_checkpoint() -> None:
    with pytest.raises(ValueError, match="contract or lineage"):
        validate_mechanism_seed_checkpoint_payload(
            {
                "status": "passed",
                "final_model_state_sha256": "a" * 64,
            },
            _contract(),
            _lineage(),
        )


def test_optimizer_guard_rejects_null_before_zero_grad_or_backward() -> None:
    config = _candidate_config(tiny=True)
    contract = FullContextTrainerContract.from_candidate_config(config)
    pairs, contexts = _pairs_and_contexts((48,))
    encoder = PAMSEncoder(
        input_dim=99,
        model_dim=8,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        dropout=0.25,
        max_length=128,
    )
    optimizer = build_fullcontext_adamw(encoder, contract)
    before = model_state_sha256(encoder)

    with pytest.raises(ValueError, match="diagnostic pair-segment contexts"):
        optimize_real_pair_contexts(
            encoder,
            optimizer,
            ConventionalCycleBackLoss(),
            pairs,
            zero_pair_segment_contexts(contexts),
            contract,
        )

    assert model_state_sha256(encoder) == before
    assert optimizer.state == {}


def test_real_optimizer_step_continues_exact_step256_state_on_full_contexts() -> None:
    config = _candidate_config(tiny=True)
    contract = FullContextTrainerContract.from_candidate_config(config)
    pairs, contexts = _pairs_and_contexts((48,))
    encoder = PAMSEncoder(
        input_dim=99,
        model_dim=8,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        dropout=0.25,
        max_length=128,
    )
    optimizer = build_fullcontext_adamw(encoder, contract)
    _prime_adamw_state(encoder, optimizer, step=256)
    before = model_state_sha256(encoder)

    loss, valid_anchors, possible_anchors = optimize_real_pair_contexts(
        encoder,
        optimizer,
        ConventionalCycleBackLoss(),
        pairs,
        contexts,
        contract,
        expected_prior_optimizer_step=256,
    )

    assert torch.isfinite(loss)
    assert valid_anchors > 0
    assert possible_anchors > 0
    assert model_state_sha256(encoder) != before
    assert {
        int(float(state["step"])) for state in optimizer.state.values()
    } == {257}


def _controls(*, pe_ratio: float = 1.0) -> dict[str, Any]:
    condition_names = (
        "zero_pose",
        "within_video_pose_shuffle",
        "mask_flicker",
        "torso_only",
        "alternating_limb_dropout",
        "permuted_pe",
        "pe_off",
    )
    conditions: dict[str, Any] = {
        "real": {
            "symmetric_position_mse": 0.10,
            "valid_anchor_fraction": 1.0,
            "embedding_temporal_rms": {"values": [0.2, 0.3]},
        }
    }
    for name in condition_names:
        ratio = pe_ratio if name in {"permuted_pe", "pe_off"} else 2.0
        conditions[name] = {"symmetric_position_mse": 0.10 * ratio}
    return {
        "artifact_type": "pams_cycleback_epoch11_label_free_controls_v1",
        "label_free": True,
        "optimizer_updates_during_evaluation": 0,
        "conditions": conditions,
        "ratios": {
            name: (
                pe_ratio if name in {"permuted_pe", "pe_off"} else 2.0
            )
            for name in condition_names
        },
        "authority_boundaries": {
            "epoch11_train337_continuation_authorized": False,
            "epoch150_train337_continuation_authorized": False,
            "development_evaluation_authorized": False,
            "sealed_evaluation_authorized": False,
            "readout_authorized": False,
        },
    }


def test_epoch11_gate_uses_two_sided_pe_bounds_and_never_self_authorizes() -> None:
    thresholds = Epoch11GateThresholds(
        minimum_real_null_position_error_gap=0.05,
        minimum_valid_anchor_fraction=0.5,
        minimum_temporal_rms_median=0.01,
        minimum_null_to_real_ratio=1.2,
        minimum_pe_ratio=0.8,
        maximum_pe_ratio=1.25,
    )

    passed = epoch11_gate_decision(_controls(), thresholds)
    too_low = epoch11_gate_decision(_controls(pe_ratio=0.1), thresholds)
    too_high = epoch11_gate_decision(_controls(pe_ratio=4.0), thresholds)

    assert passed["overall_pass"] is True
    assert passed["scientifically_eligible_for_epoch150_train337"] is True
    assert passed["epoch150_train337_continuation_authorized"] is False
    assert passed["development_evaluation_authorized"] is False
    assert passed["sealed_evaluation_authorized"] is False
    assert too_low["overall_pass"] is False
    assert too_high["overall_pass"] is False


def test_empty_activation_fails_before_any_path_read_or_checkpoint_deserialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = {"read": 0, "load": 0}

    def forbidden_read(*args: object, **kwargs: object) -> bytes:
        del args, kwargs
        calls["read"] += 1
        raise AssertionError("path read crossed empty activation")

    def forbidden_load(*args: object, **kwargs: object) -> object:
        del args, kwargs
        calls["load"] += 1
        raise AssertionError("checkpoint load crossed empty activation")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    monkeypatch.setattr(torch, "load", forbidden_load)

    with pytest.raises(RuntimeError, match="disabled pending"):
        validate_fullcontext_predecessor_authority(
            "epoch11",
            integration_outcome_root=tmp_path / "forged-integration",
            mechanism_outcome_root=tmp_path / "forged-mechanism",
            launch_authorization_root=tmp_path / "forged-launch",
        )

    assert calls == {"read": 0, "load": 0}
    assert authority_scaffold_status() == {
        "mechanism_seed_checkpoint_available": False,
        "epoch11_train337_continuation_authorized": False,
        "epoch150_train337_continuation_authorized": False,
        "development_evaluation_authorized": False,
        "sealed_evaluation_authorized": False,
        "readout_authorized": False,
    }
    assert all(
        not value
        for name, value in vars(authority_module).items()
        if name.startswith("_APPROVED_")
    )
