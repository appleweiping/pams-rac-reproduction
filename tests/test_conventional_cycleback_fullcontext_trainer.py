from __future__ import annotations

import copy
import hashlib
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import pams.conventional_cycleback.fullcontext_authority as authority_module
import pams.conventional_cycleback.fullcontext_trainer as trainer_module
import pams.conventional_cycleback.probe as probe_module
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
    MechanismSeedLoadReceipt,
    MechanismSeedPredecessorLineage,
    MechanismSeedSamplerState,
    atomic_save_fullcontext_checkpoint,
    atomic_save_new_mechanism_seed_checkpoint,
    build_epoch150_transition_bindings,
    build_fullcontext_adamw,
    build_fullcontext_checkpoint_payload,
    build_fullcontext_objective,
    build_length_bucket_plan,
    build_mechanism_seed_checkpoint_payload,
    capture_fullcontext_backend_state,
    capture_fullcontext_rng_state,
    configure_fullcontext_determinism,
    epoch11_gate_decision,
    evaluate_unified2d_epoch11_label_free_controls,
    initialize_epoch11_from_exact_mechanism_seed_checkpoint,
    initialize_epoch150_from_exact_epoch11_checkpoint,
    load_fullcontext_checkpoint,
    model_state_sha256,
    optimizer_state_sha256,
    preserve_fullcontext_diagnostic_state,
    run_fullcontext_optimizer_step,
    validate_fullcontext_checkpoint_payload,
    validate_fullcontext_objective,
    validate_fullcontext_optimizer,
    validate_mechanism_seed_checkpoint_payload,
)
from pams.conventional_cycleback.loss import ConventionalCycleBackLoss
from pams.conventional_cycleback.runtime import (
    AugmentedSequenceViews,
    PairEligibility,
    PairSegmentContexts,
    Unified2DSequence,
    collate_to_device,
    context_encoding_plan,
    encode_window_pairs,
    pair_segment_contexts,
    pairs_from_batch,
    validate_exact_authorized_pair_rows,
    zero_pair_segment_contexts,
)
from pams.conventional_cycleback.windows import (
    NativeWindowPairBatch,
    enumerate_native_window_pairs,
)
from pams.model import PAMSEncoder
from pams.types import PoseSequence

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


def _representation() -> FullContextRepresentationContract:
    return FullContextRepresentationContract(
        representation_family="v4e_unified2d_coco17_padded33",
        coordinate_contract="body-centered-uniform-rms-scale-xy-z0-v1",
        pose_joint_count=33,
        support_channel_count=17,
        support_channel_to_pose_joint_indices=tuple(range(17)),
        augmentation_adapter="unified2d_masked_xy_inplane_v1",
        diagnostic_adapter="unified2d_coco17_joint_nulls_v1",
        stable_range_policy="v4e_exact_all_valid_track_stability_ranges_v1",
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
        representation_contract_sha256=_representation().fingerprint,
        representation_authorization_sha256="6" * 64,
        representation_authorization_bytes=103,
        mechanism_outcome_sha256="7" * 64,
        mechanism_outcome_bytes=104,
        mechanism_run_receipt_sha256="8" * 64,
        mechanism_run_receipt_bytes=105,
        mechanism_seed_checkpoint_sha256="9" * 64,
        mechanism_seed_checkpoint_bytes=106,
        mechanism_learned_model_state_sha256="a" * 64,
        mechanism_optimizer_state_sha256="d" * 64,
        mechanism_rng_state_sha256="b" * 64,
        mechanism_backend_state_sha256="c" * 64,
        mechanism_sampler_state_sha256="e" * 64,
        mechanism_view_state_sha256="f" * 64,
        mechanism_consumed_video_batch_chain_sha256="c" * 64,
        mechanism_consumed_pair_row_chain_sha256="d" * 64,
    )


def _seed_receipt(
    *,
    tiny: bool = False,
) -> MechanismSeedLoadReceipt:
    contract = _contract(tiny=tiny)
    lineage = _lineage()
    return MechanismSeedLoadReceipt(
        checkpoint_sha256=lineage.mechanism_seed_checkpoint_sha256,
        checkpoint_bytes=lineage.mechanism_seed_checkpoint_bytes,
        learned_model_state_sha256=lineage.mechanism_learned_model_state_sha256,
        optimizer_state_sha256=lineage.mechanism_optimizer_state_sha256,
        optimizer_step=256,
        mechanism_sampler_state_sha256=lineage.mechanism_sampler_state_sha256,
        mechanism_view_state_sha256=lineage.mechanism_view_state_sha256,
        mechanism_consumed_video_batch_chain_sha256=(
            lineage.mechanism_consumed_video_batch_chain_sha256
        ),
        mechanism_consumed_pair_row_chain_sha256=(
            lineage.mechanism_consumed_pair_row_chain_sha256
        ),
        rng_state_sha256=lineage.mechanism_rng_state_sha256,
        backend_state_sha256=lineage.mechanism_backend_state_sha256,
        trainer_contract_fingerprint=contract.fingerprint,
        representation_contract_sha256=_representation().fingerprint,
        representation_lineage_fingerprint=lineage.fingerprint,
    )


def _mechanism_sampler_state(
    contract: FullContextTrainerContract,
) -> MechanismSeedSamplerState:
    return MechanismSeedSamplerState(
        eligible_video_total=249,
        ordered_eligible_video_ids_sha256="b" * 64,
        video_batch_size=contract.mechanism_video_batch_size,
        maximum_pairs_per_step=contract.mechanism_maximum_pairs_per_step,
        completed_optimizer_steps=contract.mechanism_optimizer_steps,
        consumed_video_batch_chain_sha256="c" * 64,
        consumed_pair_row_chain_sha256="d" * 64,
        next_cyclic_start_index=(
            contract.mechanism_optimizer_steps
            * contract.mechanism_video_batch_size
        )
        % 249,
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


def _plan(
    *,
    epoch: int = 1,
    tiny: bool = False,
    lineage: FullContextLineage | None = None,
):
    contract = _contract(tiny=tiny)
    return build_length_bucket_plan(
        (
            _unit("opaque-video-a", length=40, starts=(0, 4, 8)),
            _unit("opaque-video-b", length=80, starts=(0, 4, 8, 12, 16)),
        ),
        contract,
        _lineage() if lineage is None else lineage,
        epoch=epoch,
    )


def _initial_progress(*, tiny: bool = False) -> FullContextProgress:
    return trainer_module._initial_epoch11_progress_from_loaded_seed(
        _plan(epoch=1, tiny=tiny),
        mechanism_seed_load_receipt=_seed_receipt(tiny=tiny),
    )


def _initial_progress_for_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    tiny: bool = False,
    lineage: FullContextLineage | None = None,
) -> FullContextProgress:
    bound_lineage = (
        _lineage_for_state(model, optimizer)
        if lineage is None
        else lineage
    )
    receipt = replace(
        _seed_receipt(tiny=tiny),
        checkpoint_sha256=bound_lineage.mechanism_seed_checkpoint_sha256,
        checkpoint_bytes=bound_lineage.mechanism_seed_checkpoint_bytes,
        learned_model_state_sha256=model_state_sha256(model),
        optimizer_state_sha256=optimizer_state_sha256(optimizer),
        rng_state_sha256=trainer_module.fullcontext_rng_state_sha256(
            capture_fullcontext_rng_state()
        ),
        representation_lineage_fingerprint=bound_lineage.fingerprint,
    )
    return trainer_module._initial_epoch11_progress_from_loaded_seed(
        _plan(epoch=1, tiny=tiny, lineage=bound_lineage),
        mechanism_seed_load_receipt=receipt,
    )


def _lineage_for_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> FullContextLineage:
    return replace(
        _lineage(),
        mechanism_learned_model_state_sha256=model_state_sha256(model),
        mechanism_optimizer_state_sha256=optimizer_state_sha256(optimizer),
        mechanism_rng_state_sha256=trainer_module.fullcontext_rng_state_sha256(
            capture_fullcontext_rng_state()
        ),
    )


def _completed_epoch11_progress(*, tiny: bool = False) -> FullContextProgress:
    contract = _contract(tiny=tiny)
    lineage = _lineage()
    progress = _initial_progress(tiny=tiny)
    while progress.active_plan is not None:
        step = progress.next_augmentation_step
        progress = trainer_module._advance_fullcontext_progress(
            progress,
            contract,
            lineage,
            next_model_state_sha256=f"{step:064x}"[-64:],
            next_optimizer_state_sha256=f"{step + 10_000:064x}"[-64:],
            next_rng_state_sha256=f"{step + 20_000:064x}"[-64:],
        )
    return progress


def _enable_test_determinism() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    configure_fullcontext_determinism()


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
    contract = _contract()
    lineage = _lineage()
    progress = _initial_progress()
    advanced = trainer_module._advance_fullcontext_progress(
        progress,
        contract,
        lineage,
        next_model_state_sha256="e" * 64,
        next_optimizer_state_sha256="f" * 64,
        next_rng_state_sha256="0" * 64,
    )

    assert advanced.completed_epochs == 0
    assert advanced.sampler_cursor == 1
    assert advanced.global_optimizer_step == 257
    assert advanced.units == progress.units
    assert advanced.dataset_fingerprint == progress.dataset_fingerprint
    assert advanced.plan_dataset_prefix_sha256 != progress.plan_dataset_prefix_sha256

    completed = _completed_epoch11_progress()
    expected_steps = sum(
        len(build_length_bucket_plan(completed.units, contract, lineage, epoch=epoch).batches)
        for epoch in range(1, 12)
    )
    assert completed.completed_epochs == 11
    assert completed.continuation_optimizer_steps == expected_steps
    assert len(completed.completed_plan_prefix) == 11


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
    elif isinstance(first, np.ndarray):
        assert isinstance(second, np.ndarray)
        assert np.array_equal(first, second)
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
    _enable_test_determinism()
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    contract = _contract()
    model = torch.nn.Linear(4, 3)
    optimizer = build_fullcontext_adamw(model, contract)
    _prime_adamw_state(model, optimizer, step=256)
    lineage = _lineage_for_state(model, optimizer)
    progress = _initial_progress_for_state(model, optimizer, lineage=lineage)
    _random_optimizer_update(model, optimizer)
    progress = trainer_module._advance_fullcontext_progress(
        progress,
        contract,
        lineage,
        next_model_state_sha256=model_state_sha256(model),
        next_optimizer_state_sha256=optimizer_state_sha256(optimizer),
        next_rng_state_sha256=trainer_module.fullcontext_rng_state_sha256(
            capture_fullcontext_rng_state()
        ),
    )
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
    _enable_test_determinism()
    contract = _contract()
    model = torch.nn.Linear(4, 3)
    optimizer = build_fullcontext_adamw(model, contract)
    _prime_adamw_state(model, optimizer, step=256)
    lineage = _lineage_for_state(model, optimizer)
    payload = build_fullcontext_checkpoint_payload(
        model,
        optimizer,
        contract,
        lineage,
        _initial_progress_for_state(model, optimizer, lineage=lineage),
    )
    parsed_only = validate_fullcontext_checkpoint_payload(
        payload,
        contract,
        lineage,
        expected_phase="epoch11",
    )
    with pytest.raises(ValueError, match="semantic evidence"):
        build_fullcontext_checkpoint_payload(
            model,
            optimizer,
            contract,
            lineage,
            parsed_only,
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
    with pytest.raises(ValueError, match="optimizer steps|sampler cursor"):
        validate_fullcontext_checkpoint_payload(
            wrong_cursor,
            contract,
            lineage,
            expected_phase="epoch11",
        )

    wrong_dataset = copy.deepcopy(payload)
    wrong_dataset["progress"]["dataset_fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="dataset fingerprint"):
        validate_fullcontext_checkpoint_payload(
            wrong_dataset,
            contract,
            lineage,
            expected_phase="epoch11",
        )

    wrong_optimizer = copy.deepcopy(payload)
    first_state = next(iter(wrong_optimizer["optimizer_state"]["state"].values()))
    first_state["exp_avg"].add_(1.0)
    with pytest.raises(ValueError, match="optimizer bytes"):
        validate_fullcontext_checkpoint_payload(
            wrong_optimizer,
            contract,
            lineage,
            expected_phase="epoch11",
        )

    decoupled_progress = copy.deepcopy(payload)
    decoupled_progress["progress"]["current_model_state_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="progress derived fields|model-state digest"):
        validate_fullcontext_checkpoint_payload(
            decoupled_progress,
            contract,
            lineage,
            expected_phase="epoch11",
        )


def test_epoch150_initialization_restores_exact_completed_epoch11_prefix(
    tmp_path: Path,
) -> None:
    _enable_test_determinism()
    contract = _contract()
    representation = _representation()
    lineage = _lineage()
    model = torch.nn.Linear(4, 3)
    optimizer = build_fullcontext_adamw(model, contract)
    completed = _completed_epoch11_progress()
    _prime_adamw_state(model, optimizer, step=completed.global_optimizer_step)
    completed = replace(
        completed,
        current_model_state_sha256=model_state_sha256(model),
        current_optimizer_state_sha256=optimizer_state_sha256(optimizer),
        current_rng_state_sha256=trainer_module.fullcontext_rng_state_sha256(
            capture_fullcontext_rng_state()
        ),
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
    threshold_identity = ("d" * 64, 201)
    gate_bindings = _control_bindings(
        checkpoint_identity=identity,
        model_state_sha256=model_state_sha256(model),
        progress=completed,
        contract=contract,
        lineage=lineage,
    )
    gate = epoch11_gate_decision(
        _controls(bindings=gate_bindings),
        _thresholds(),
        threshold_receipt_sha256=threshold_identity[0],
        threshold_receipt_bytes=threshold_identity[1],
    )
    gate_identity = ("e" * 64, 202)
    threshold_receipt = {
        "schema_version": 1,
        "artifact_type": (
            "pams_cycleback_epoch11_threshold_preregistration_receipt_v1"
        ),
        "status": "frozen",
        "candidate_id": contract.candidate_id,
        "thresholds_sha256": gate["thresholds_sha256"],
        "thresholds_frozen_before_execution": True,
        "label_free": True,
        "epoch150_train337_continuation_authorized": False,
        "development_evaluation_authorized": False,
        "sealed_evaluation_authorized": False,
    }
    launch_identity = ("f" * 64, 203)
    launch_bindings = {
        "epoch11_checkpoint_sha256": identity[0],
        "epoch11_checkpoint_bytes": identity[1],
        "epoch11_model_state_sha256": model_state_sha256(model),
        "epoch11_optimizer_state_sha256": (
            completed.current_optimizer_state_sha256
        ),
        "epoch11_progress_sha256": completed.fingerprint,
        "epoch11_gate_outcome_sha256": gate_identity[0],
        "epoch11_gate_outcome_bytes": gate_identity[1],
        "threshold_receipt_sha256": threshold_identity[0],
        "threshold_receipt_bytes": threshold_identity[1],
        "trainer_contract_fingerprint": contract.fingerprint,
        "representation_contract_sha256": representation.fingerprint,
        "representation_lineage_fingerprint": lineage.fingerprint,
    }
    launch_authorization = {
        "schema_version": 1,
        "artifact_type": "pams_cycleback_epoch150_launch_authorization_v1",
        "status": "authorized",
        "candidate_id": contract.candidate_id,
        "bindings": launch_bindings,
        "label_free": True,
        "epoch150_train337_continuation_authorized": True,
        "development_evaluation_authorized": False,
        "sealed_evaluation_authorized": False,
        "readout_authorized": False,
    }
    transition = build_epoch150_transition_bindings(
        gate_outcome=gate,
        gate_outcome_identity=gate_identity,
        threshold_receipt=threshold_receipt,
        threshold_receipt_identity=threshold_identity,
        launch_authorization=launch_authorization,
        launch_authorization_identity=launch_identity,
        epoch11_checkpoint_identity=identity,
        epoch11_model_state_sha256=model_state_sha256(model),
        epoch11_progress=completed,
        contract=contract,
        representation=representation,
        lineage=lineage,
    )
    tampered_launch = copy.deepcopy(launch_authorization)
    tampered_launch["bindings"]["epoch11_gate_outcome_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="launch authorization lineage"):
        build_epoch150_transition_bindings(
            gate_outcome=gate,
            gate_outcome_identity=gate_identity,
            threshold_receipt=threshold_receipt,
            threshold_receipt_identity=threshold_identity,
            launch_authorization=tampered_launch,
            launch_authorization_identity=launch_identity,
            epoch11_checkpoint_identity=identity,
            epoch11_model_state_sha256=model_state_sha256(model),
            epoch11_progress=completed,
            contract=contract,
            representation=representation,
            lineage=lineage,
        )

    epoch150 = initialize_epoch150_from_exact_epoch11_checkpoint(
        checkpoint,
        expected_sha256=identity[0],
        expected_bytes=identity[1],
        contract=contract,
        lineage=lineage,
        epoch12_plan=_plan(epoch=12),
        transition=transition,
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


def test_mechanism_seed_captures_exact_step256_boundary_and_publishes_once(
    tmp_path: Path,
) -> None:
    _enable_test_determinism()
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    contract = _contract()
    model = torch.nn.Linear(4, 3)
    optimizer = build_fullcontext_adamw(model, contract)
    _prime_adamw_state(model, optimizer, step=256)
    lineage = replace(
        _lineage(),
        mechanism_learned_model_state_sha256=model_state_sha256(model),
        mechanism_optimizer_state_sha256=optimizer_state_sha256(optimizer),
        mechanism_rng_state_sha256=trainer_module.fullcontext_rng_state_sha256(
            capture_fullcontext_rng_state()
        ),
        mechanism_backend_state_sha256=(
            trainer_module.fullcontext_backend_state_sha256(
                capture_fullcontext_backend_state()
            )
        ),
        mechanism_sampler_state_sha256=_mechanism_sampler_state(contract).fingerprint,
        mechanism_view_state_sha256="0" * 64,
        mechanism_consumed_video_batch_chain_sha256=(
            _mechanism_sampler_state(contract).consumed_video_batch_chain_sha256
        ),
        mechanism_consumed_pair_row_chain_sha256=(
            _mechanism_sampler_state(contract).consumed_pair_row_chain_sha256
        ),
    )
    predecessor = lineage.mechanism_seed_predecessor()
    assert isinstance(predecessor, MechanismSeedPredecessorLineage)
    payload = build_mechanism_seed_checkpoint_payload(
        model,
        optimizer,
        contract,
        predecessor,
        _mechanism_sampler_state(contract),
    )
    lineage = replace(
        lineage,
        mechanism_view_state_sha256=payload["next_view_seed_state_sha256"],
    )
    validate_mechanism_seed_checkpoint_payload(
        payload,
        contract,
        lineage,
    )
    with pytest.raises(ValueError, match="state closure"):
        validate_mechanism_seed_checkpoint_payload(
            payload,
            contract,
            replace(lineage, mechanism_rng_state_sha256="0" * 64),
        )
    boundary_optimizer = optimizer_state_sha256(optimizer)
    with preserve_fullcontext_diagnostic_state(model):
        model.eval()
        _ = random.random()
        _ = np.random.random()
        _ = torch.rand(3)
    assert model.training
    assert model_state_sha256(model) == payload["model_state_sha256"]
    assert optimizer_state_sha256(optimizer) == boundary_optimizer

    checkpoint = tmp_path / "learned-encoder-L.pt"
    identity = atomic_save_new_mechanism_seed_checkpoint(payload, checkpoint)
    assert identity[0]
    assert identity[1] > 0
    final_lineage = replace(
        lineage,
        mechanism_seed_checkpoint_sha256=identity[0],
        mechanism_seed_checkpoint_bytes=identity[1],
    )
    epoch1_model = torch.nn.Linear(4, 3)
    epoch1_optimizer = build_fullcontext_adamw(epoch1_model, contract)
    epoch1 = initialize_epoch11_from_exact_mechanism_seed_checkpoint(
        checkpoint,
        expected_sha256=identity[0],
        expected_bytes=identity[1],
        contract=contract,
        lineage=final_lineage,
        epoch1_plan=_plan(epoch=1, lineage=final_lineage),
        model=epoch1_model,
        optimizer=epoch1_optimizer,
    )
    assert epoch1.completed_epochs == 0
    receipt = epoch1.mechanism_seed_load_receipt
    assert receipt.loaded_as_epoch1_start is True
    assert receipt.mechanism_sampler_state_sha256 == (
        payload["sampler_state_sha256"]
    )
    assert model_state_sha256(epoch1_model) == payload["model_state_sha256"]
    assert optimizer_state_sha256(epoch1_optimizer) == (
        payload["optimizer_state_sha256"]
    )
    with pytest.raises(ValueError, match="must be absent"):
        atomic_save_new_mechanism_seed_checkpoint(payload, checkpoint)


def test_mechanism_seed_rejects_wrong_step_state_lineage_and_optimizer() -> None:
    _enable_test_determinism()
    contract = _contract()
    model = torch.nn.Linear(4, 3)
    optimizer = build_fullcontext_adamw(model, contract)
    _prime_adamw_state(model, optimizer, step=255)
    predecessor = replace(
        _lineage(),
        mechanism_learned_model_state_sha256=model_state_sha256(model),
    ).mechanism_seed_predecessor()
    with pytest.raises(ValueError, match="optimizer step"):
        build_mechanism_seed_checkpoint_payload(
            model,
            optimizer,
            contract,
            predecessor,
            _mechanism_sampler_state(contract),
        )

    _prime_adamw_state(model, optimizer, step=256)
    payload = build_mechanism_seed_checkpoint_payload(
        model,
        optimizer,
        contract,
        predecessor,
        _mechanism_sampler_state(contract),
    )
    wrong_lineage = copy.deepcopy(payload)
    wrong_lineage["lineage"]["source_tree_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="contract or lineage"):
        validate_mechanism_seed_checkpoint_payload(
            wrong_lineage,
            contract,
            predecessor,
        )
    wrong_optimizer = copy.deepcopy(payload)
    first_state = next(iter(wrong_optimizer["optimizer_state"]["state"].values()))
    first_state["exp_avg"].add_(1.0)
    with pytest.raises(ValueError, match="optimizer-state digest"):
        validate_mechanism_seed_checkpoint_payload(
            wrong_optimizer,
            contract,
            predecessor,
        )
    wrong_sampler = copy.deepcopy(payload)
    wrong_sampler["sampler_state"]["next_cyclic_start_index"] += 1
    with pytest.raises(ValueError, match="cyclic index|sampler state"):
        validate_mechanism_seed_checkpoint_payload(
            wrong_sampler,
            contract,
            predecessor,
        )
    wrong_view = copy.deepcopy(payload)
    wrong_view["next_view_seed_state"]["next_augmentation_step"] = 256
    with pytest.raises(ValueError, match="next-view state"):
        validate_mechanism_seed_checkpoint_payload(
            wrong_view,
            contract,
            predecessor,
        )


def test_mechanism_bundle_is_transactional_and_rejection_has_no_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejected_output = tmp_path / "rejected/mechanism-bundle/mechanism-probe.json"
    rejected_output.parent.parent.mkdir()
    rejected_seed = rejected_output.with_name("learned-encoder-L.pt")
    probe_module._publish_mechanism_bundle(
        rejected_output,
        rejected_seed,
        {"status": "rejected"},
        checkpoint_bytes=None,
        checkpoint_identity=None,
    )
    assert rejected_output.exists()
    assert not rejected_seed.exists()

    passed_output = tmp_path / "passed/mechanism-bundle/mechanism-probe.json"
    passed_output.parent.parent.mkdir()
    passed_seed = passed_output.with_name("learned-encoder-L.pt")
    passed_bytes = b"prepared-pass-seed"
    passed_identity = (
        hashlib.sha256(passed_bytes).hexdigest(),
        len(passed_bytes),
    )
    probe_module._publish_mechanism_bundle(
        passed_output,
        passed_seed,
        {"status": "passed"},
        checkpoint_bytes=passed_bytes,
        checkpoint_identity=passed_identity,
    )
    assert passed_seed.read_bytes() == passed_bytes
    with pytest.raises(ValueError, match="destination appeared"):
        probe_module._publish_mechanism_bundle(
            passed_output,
            passed_seed,
            {"status": "passed"},
            checkpoint_bytes=passed_bytes,
            checkpoint_identity=passed_identity,
        )

    failed_output = tmp_path / "failed/mechanism-bundle/mechanism-probe.json"
    failed_output.parent.parent.mkdir()
    failed_seed = failed_output.with_name("learned-encoder-L.pt")

    def fail_json(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected JSON publication failure")

    monkeypatch.setattr(probe_module, "_write_new_json", fail_json)
    encoded = b"prepared-seed"
    identity = (hashlib.sha256(encoded).hexdigest(), len(encoded))
    with pytest.raises(OSError, match="injected JSON"):
        probe_module._publish_mechanism_bundle(
            failed_output,
            failed_seed,
            {"status": "passed"},
            checkpoint_bytes=encoded,
            checkpoint_identity=identity,
        )
    assert not failed_output.parent.exists()
    assert not failed_seed.exists()
    assert not tuple(
        failed_output.parent.parent.glob(".mechanism-bundle.*.incomplete")
    )
    assert not any(
        failed_output.parent.parent.rglob("learned-encoder-L.pt")
    )


def test_low_level_optimizer_is_private_and_null_role_remains_non_optimizer() -> None:
    assert not hasattr(trainer_module, "optimize_real_pair_contexts")
    assert not hasattr(trainer_module, "advance_fullcontext_progress")
    assert not hasattr(trainer_module, "initial_epoch11_progress")
    assert not hasattr(trainer_module, "initial_epoch150_progress")
    assert not hasattr(trainer_module, "load_mechanism_seed_checkpoint")
    assert hasattr(
        trainer_module,
        "initialize_epoch11_from_exact_mechanism_seed_checkpoint",
    )

    class EqualityForgery:
        def __eq__(self, _other: object) -> bool:
            return True

    with pytest.raises(ValueError, match="validated checkpoint load"):
        replace(
            _initial_progress(),
            _runtime_attestation=EqualityForgery(),
        )
    _, contexts = _pairs_and_contexts((48,))
    controlled = zero_pair_segment_contexts(contexts)
    assert controlled.objective_role == "diagnostic_zero"
    assert controlled.transform_contract["authorized_for_real_optimizer"] is False


def test_objective_must_match_every_frozen_candidate_field() -> None:
    config = _candidate_config(tiny=True)
    contract = FullContextTrainerContract.from_candidate_config(config)
    exact = build_fullcontext_objective(config, contract)
    validate_fullcontext_objective(exact, config, contract)
    with pytest.raises(ValueError, match="objective parameters"):
        validate_fullcontext_objective(
            ConventionalCycleBackLoss(temperature=0.2),
            config,
            contract,
        )


def test_adamw_foreach_fused_and_capture_flags_are_exact() -> None:
    contract = _contract()
    model = torch.nn.Linear(4, 3)
    optimizer = build_fullcontext_adamw(model, contract)
    validate_fullcontext_optimizer(optimizer, contract)
    for key, value in (
        ("foreach", True),
        ("fused", True),
        ("capturable", True),
        ("differentiable", True),
    ):
        altered = build_fullcontext_adamw(torch.nn.Linear(4, 3), contract)
        altered.param_groups[0][key] = value
        with pytest.raises(ValueError, match="parameter group"):
            validate_fullcontext_optimizer(altered, contract)
    extra = build_fullcontext_adamw(torch.nn.Linear(4, 3), contract)
    extra.param_groups[0]["unfrozen_option"] = False
    with pytest.raises(ValueError, match="parameter group"):
        validate_fullcontext_optimizer(extra, contract)


def test_exact_pair_replay_rejects_row_order_source_and_range_tampering() -> None:
    pairs, _ = _pairs_and_contexts((48,))
    expected = validate_exact_authorized_pair_rows(
        pairs,
        ordered_video_ids=("video-0",),
        native_lengths={"video-0": 48},
        authorized_ranges_by_video={"video-0": ((0, 48),)},
        base_valid_starts_by_video={"video-0": (0, 4, 8, 12, 16)},
        eligible_starts_by_video={"video-0": (0, 4, 8, 12, 16)},
        window_frames=16,
        hop_frames=4,
    )
    assert len(expected) == 64
    reversed_pairs = pairs.select(torch.arange(pairs.pair_count - 1, -1, -1))
    with pytest.raises(RuntimeError, match="order/source indices/range"):
        validate_exact_authorized_pair_rows(
            reversed_pairs,
            ordered_video_ids=("video-0",),
            native_lengths={"video-0": 48},
            authorized_ranges_by_video={"video-0": ((0, 48),)},
            base_valid_starts_by_video={"video-0": (0, 4, 8, 12, 16)},
            eligible_starts_by_video={"video-0": (0, 4, 8, 12, 16)},
            window_frames=16,
            hop_frames=4,
        )


def _formal_step_inputs() -> tuple[
    dict[str, Unified2DSequence],
    dict[str, tuple[tuple[int, int], ...]],
    PairEligibility,
]:
    selected = {
        "opaque-video-a": (40, (0, 4, 8)),
        "opaque-video-b": (80, (0, 4, 8, 12, 16)),
    }
    all_ids = (*selected, *(f"opaque-ineligible-{index:03d}" for index in range(335)))
    native_lengths = {
        video_id: selected.get(video_id, (1, ()))[0]
        for video_id in all_ids
    }
    representation_eligible = {
        video_id: video_id in selected for video_id in all_ids
    }
    variants = ("W16_H2", "W16_H4", "W16_H4_PE0", "W24_H4")
    starts_by_variant = {
        variant: {
            video_id: (
                selected[video_id][1]
                if video_id in selected and variant in {"W16_H4", "W16_H4_PE0"}
                else ()
            )
            for video_id in all_ids
        }
        for variant in variants
    }
    pair_eligibility = PairEligibility(
        start_grid_policy="native_zero_based_start_mod_hop_equals_zero",
        frozen_thresholds={},
        minimum_window_stable_action_joints=6,
        minimum_window_joint_support_fraction=0.75,
        native_lengths=native_lengths,
        representation_eligible=representation_eligible,
        base_valid_starts_by_variant=starts_by_variant,
        starts_by_variant=starts_by_variant,
        identity=("a" * 64, 1),
    )
    sequences: dict[str, Unified2DSequence] = {}
    ranges: dict[str, tuple[tuple[int, int], ...]] = {}
    for video_id, (length, _) in selected.items():
        xyz = np.zeros((length, 33, 3), dtype=np.float32)
        timeline = np.arange(length, dtype=np.float32).reshape(length, 1)
        xyz[:, :17, 0] = timeline
        xyz[:, :17, 1] = timeline * 0.5
        valid = np.ones((length,), dtype=np.bool_)
        joint = np.ones((length, 17), dtype=np.bool_)
        sequences[video_id] = Unified2DSequence(
            PoseSequence(video_id=video_id, fps=30.0, xyz=xyz, valid_mask=valid),
            joint,
        )
        ranges[video_id] = ((0, length),)
    return sequences, ranges, pair_eligibility


def _prepare_formal_step(
    encoder: torch.nn.Module,
    progress: FullContextProgress,
    config: ConventionalCycleBackConfig,
    sequences: Mapping[str, Unified2DSequence],
    ranges: Mapping[str, Sequence[tuple[int, int]]],
    eligibility: PairEligibility,
) -> tuple[NativeWindowPairBatch, PairSegmentContexts]:
    if progress.active_plan is None:
        raise AssertionError("test progress unexpectedly completed")
    selected_ids = progress.active_plan.batches[progress.sampler_cursor].video_ids
    selected = tuple(sequences[video_id] for video_id in selected_ids)
    batch = collate_to_device(selected, next(encoder.parameters()).device)
    selected_ranges = {
        video_id: tuple(ranges[video_id]) for video_id in selected_ids
    }
    pairs, views = pairs_from_batch(
        batch,
        config,
        step=progress.next_augmentation_step,
        segment_ranges_by_video=selected_ranges,
        pair_eligibility=eligibility,
    )
    return pairs, pair_segment_contexts(pairs, views)


def test_single_step_api_consumes_progress_and_rejects_skip_or_replay() -> None:
    _enable_test_determinism()
    config = _candidate_config(tiny=True)
    contract = _contract(tiny=True)
    sequences, ranges, eligibility = _formal_step_inputs()
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
    lineage = _lineage_for_state(encoder, optimizer)
    progress = _initial_progress_for_state(
        encoder,
        optimizer,
        tiny=True,
        lineage=lineage,
    )
    pairs, contexts = _prepare_formal_step(
        encoder,
        progress,
        config,
        sequences,
        ranges,
        eligibility,
    )
    preserved_rng = capture_fullcontext_rng_state()
    _ = torch.rand(1)
    with pytest.raises(ValueError, match="model/optimizer/RNG"):
        run_fullcontext_optimizer_step(
            encoder,
            optimizer,
            pairs,
            contexts,
            ranges,
            eligibility,
            config,
            contract,
            _representation(),
            lineage,
            progress,
        )
    trainer_module.restore_fullcontext_rng_state(preserved_rng)
    tampered_positions = replace(
        contexts,
        position_indices_a=(
            contexts.position_indices_a[0] + 1,
            *contexts.position_indices_a[1:],
        ),
    )
    before_rejection = model_state_sha256(encoder)
    with pytest.raises(ValueError, match="identity/absolute positions"):
        run_fullcontext_optimizer_step(
            encoder,
            optimizer,
            pairs,
            tampered_positions,
            ranges,
            eligibility,
            config,
            contract,
            _representation(),
            lineage,
            progress,
        )
    assert model_state_sha256(encoder) == before_rejection
    wrong_objective_config = config.model_copy(
        update={
            "objective": config.objective.model_copy(
                update={"temperature": 0.2}
            )
        }
    )
    with pytest.raises(ValueError, match="config/objective"):
        run_fullcontext_optimizer_step(
            encoder,
            optimizer,
            pairs,
            contexts,
            ranges,
            eligibility,
            wrong_objective_config,
            contract,
            _representation(),
            lineage,
            progress,
        )

    result = run_fullcontext_optimizer_step(
        encoder,
        optimizer,
        pairs,
        contexts,
        ranges,
        eligibility,
        config,
        contract,
        _representation(),
        lineage,
        progress,
    )

    assert result.progress.global_optimizer_step == 257
    assert result.evidence.progress_before_sha256 == progress.fingerprint
    assert result.evidence.progress_after_sha256 == result.progress.fingerprint
    assert len(result.evidence.exact_pair_rows_sha256) == 64
    with pytest.raises(ValueError, match="model/optimizer/RNG bytes"):
        run_fullcontext_optimizer_step(
            encoder,
            optimizer,
            pairs,
            contexts,
            ranges,
            eligibility,
            config,
            contract,
            _representation(),
            lineage,
            progress,
        )

    alternate_encoder = copy.deepcopy(encoder)
    with torch.no_grad():
        next(alternate_encoder.parameters()).add_(1.0)
    alternate_optimizer = build_fullcontext_adamw(alternate_encoder, contract)
    _prime_adamw_state(alternate_encoder, alternate_optimizer, step=257)
    with pytest.raises(ValueError, match="model/optimizer/RNG bytes"):
        run_fullcontext_optimizer_step(
            alternate_encoder,
            alternate_optimizer,
            pairs,
            contexts,
            ranges,
            eligibility,
            config,
            contract,
            _representation(),
            lineage,
            result.progress,
        )

    fresh_encoder = copy.deepcopy(encoder)
    fresh_optimizer = build_fullcontext_adamw(fresh_encoder, contract)
    _prime_adamw_state(fresh_encoder, fresh_optimizer, step=256)
    skipped = trainer_module._advance_fullcontext_progress(
        progress,
        contract,
        lineage,
        next_model_state_sha256="e" * 64,
        next_optimizer_state_sha256="f" * 64,
        next_rng_state_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="model/optimizer/RNG bytes"):
        run_fullcontext_optimizer_step(
            fresh_encoder,
            fresh_optimizer,
            pairs,
            contexts,
            ranges,
            eligibility,
            config,
            contract,
            _representation(),
            lineage,
            skipped,
        )


def test_epoch11_controls_restore_rng_backends_and_bind_exact_checkpoint() -> None:
    _enable_test_determinism()
    config = _candidate_config(tiny=True)
    contract = _contract(tiny=True)
    lineage = _lineage()
    progress = _completed_epoch11_progress(tiny=True)
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
    model_sha256 = model_state_sha256(encoder)
    progress = replace(progress, current_model_state_sha256=model_sha256)
    pair_identity, pair_payload = trainer_module.pair_context_fingerprints(
        pairs, contexts
    )
    before_rng = capture_fullcontext_rng_state()
    before_backend = capture_fullcontext_backend_state()

    controls = evaluate_unified2d_epoch11_label_free_controls(
        encoder,
        pairs,
        contexts,
        config,
        contract,
        _representation(),
        lineage,
        progress,
        epoch11_checkpoint_sha256="d" * 64,
        epoch11_checkpoint_bytes=456,
        expected_epoch11_model_state_sha256=model_sha256,
        expected_epoch11_optimizer_state_sha256=(
            progress.current_optimizer_state_sha256
        ),
        expected_pair_identity_sha256=pair_identity,
        expected_pair_payload_sha256=pair_payload,
    )

    _assert_nested_equal(capture_fullcontext_rng_state(), before_rng)
    assert capture_fullcontext_backend_state() == before_backend
    assert encoder.training is True
    assert controls["bindings"]["epoch11_progress_sha256"] == progress.fingerprint
    assert controls["bindings"]["epoch11_checkpoint_sha256"] == "d" * 64
    assert controls["bindings"]["rng_backend_and_model_restored_by_finally"] is True


def _control_bindings(
    *,
    checkpoint_identity: tuple[str, int] = ("1" * 64, 100),
    model_state_sha256: str = "2" * 64,
    progress: FullContextProgress | None = None,
    contract: FullContextTrainerContract | None = None,
    lineage: FullContextLineage | None = None,
) -> dict[str, Any]:
    resolved_contract = _contract() if contract is None else contract
    resolved_lineage = _lineage() if lineage is None else lineage
    progress_sha256 = "3" * 64 if progress is None else progress.fingerprint
    prefix_sha256 = (
        "4" * 64 if progress is None else progress.plan_dataset_prefix_sha256
    )
    return {
        "epoch11_checkpoint_sha256": checkpoint_identity[0],
        "epoch11_checkpoint_bytes": checkpoint_identity[1],
        "epoch11_model_state_sha256": model_state_sha256,
        "epoch11_optimizer_state_sha256": (
            "a" * 64
            if progress is None
            else progress.current_optimizer_state_sha256
        ),
        "epoch11_progress_sha256": progress_sha256,
        "plan_dataset_prefix_sha256": prefix_sha256,
        "trainer_contract_fingerprint": resolved_contract.fingerprint,
        "representation_contract_sha256": _representation().fingerprint,
        "representation_lineage_fingerprint": resolved_lineage.fingerprint,
        "pair_identity_sha256": "5" * 64,
        "pair_payload_sha256": "6" * 64,
        "pre_gate_rng_state_sha256": "7" * 64,
        "pre_gate_backend_state_sha256": "8" * 64,
        "pre_gate_model_state_sha256": model_state_sha256,
        "rng_backend_and_model_restored_by_finally": True,
    }


def _controls(
    *,
    pe_ratio: float = 1.0,
    bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
    resolved_bindings = dict(_control_bindings() if bindings is None else bindings)
    return {
        "artifact_type": "pams_cycleback_epoch11_label_free_controls_v1",
        "label_free": True,
        "optimizer_updates_during_evaluation": 0,
        "pair_identity_sha256": resolved_bindings["pair_identity_sha256"],
        "pair_payload_sha256": resolved_bindings["pair_payload_sha256"],
        "bindings": resolved_bindings,
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


def _thresholds() -> Epoch11GateThresholds:
    return Epoch11GateThresholds(
        minimum_real_null_position_error_gap=0.05,
        minimum_valid_anchor_fraction=0.5,
        minimum_temporal_rms_median=0.01,
        minimum_null_to_real_ratio=1.2,
        minimum_pe_ratio=0.8,
        maximum_pe_ratio=1.25,
    )


def test_epoch11_gate_uses_two_sided_pe_bounds_and_never_self_authorizes() -> None:
    thresholds = _thresholds()
    kwargs = {
        "threshold_receipt_sha256": "9" * 64,
        "threshold_receipt_bytes": 77,
    }
    passed = epoch11_gate_decision(_controls(), thresholds, **kwargs)
    too_low = epoch11_gate_decision(
        _controls(pe_ratio=0.1), thresholds, **kwargs
    )
    too_high = epoch11_gate_decision(
        _controls(pe_ratio=4.0), thresholds, **kwargs
    )

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
