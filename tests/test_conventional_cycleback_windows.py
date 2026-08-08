from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pams.conventional_cycleback.config import load_conventional_cycleback_config
from pams.conventional_cycleback.runtime import (
    AugmentedSequenceViews,
    augment_unified_2d_pose_batch,
    cap_window_pairs,
    encode_window_pairs,
    independently_permuted_pair_poses,
    independently_permuted_pair_position_indices,
    independently_permuted_pair_segment_contexts,
    independently_permuted_pair_segment_positions,
    joint_support_null_poses,
    joint_support_null_segment_contexts,
    pair_segment_contexts,
    require_real_optimizer_contexts,
    zero_pair_segment_contexts,
)
from pams.conventional_cycleback.windows import (
    NativeWindowPairBatch,
    enumerate_native_window_pairs,
    source_index_invariant_violations,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/conventional_cycleback/w16_hop4_v1.yaml"


def _inputs(
    frames: int,
    *,
    batch: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    active = torch.arange(
        batch * frames * 17 * 2,
        dtype=torch.float32,
    ).reshape(batch, frames, 17, 2)
    values = torch.zeros((batch, frames, 33, 3), dtype=torch.float32)
    values[:, :, :17, :2] = active
    second = values.clone()
    second[:, :, :17, :2] += 1.0
    valid = torch.ones((batch, frames), dtype=torch.bool)
    joint_valid = torch.ones((batch, frames, 17), dtype=torch.bool)
    lengths = torch.full((batch,), frames, dtype=torch.long)
    return values, second, valid, joint_valid, lengths


def _enumerate(
    *,
    frames: int = 24,
    window: int = 8,
    hop: int = 2,
    segments: tuple[tuple[int, int], ...] = ((0, 24),),
    base_starts: tuple[int, ...] = (0, 2, 4, 6, 8),
    final_starts: tuple[int, ...] = (0, 4, 8),
) -> NativeWindowPairBatch:
    first, second, valid, joint_valid, lengths = _inputs(frames)
    return enumerate_native_window_pairs(
        first,
        second,
        valid,
        joint_valid,
        lengths,
        ("video-a",),
        (segments,),
        (base_starts,),
        (final_starts,),
        window_length=window,
        hop_frames=hop,
        minimum_valid_frames_per_window=window,
        minimum_window_stable_action_joints=6,
        minimum_window_joint_support_fraction=0.75,
    )


def test_exact_authorized_disjoint_pairs_and_three_layer_counts() -> None:
    pairs = _enumerate()

    assert pairs.raw_grid_pair_counts == (5,)
    assert pairs.available_pair_counts == (5,)
    assert pairs.eligible_pair_counts == (3,)
    assert pairs.starts_a.tolist() == [0, 4, 8]
    assert pairs.starts_b.tolist() == [8, 12, 16]
    assert pairs.source_indices_a[0].tolist() == list(range(8))
    assert pairs.source_indices_b[0].tolist() == list(range(8, 16))
    violations = source_index_invariant_violations(pairs, expected_hop_frames=2)
    assert violations["source_index_intersection_total"] == 0
    assert violations["total"] == 0


def test_reset_and_quarantine_do_not_inflate_base_valid_denominator() -> None:
    pairs = _enumerate(
        frames=40,
        window=8,
        hop=4,
        segments=((0, 16), (24, 40)),
        base_starts=(0, 24),
        final_starts=(0, 24),
    )

    assert pairs.raw_grid_pair_counts == (7,)
    assert pairs.available_pair_counts == (2,)
    assert pairs.eligible_pair_counts == (2,)
    assert pairs.pair_count / pairs.available_pair_total == 1.0


def test_authority_start_crossing_reset_is_rejected_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="crosses a representation segment"):
        _enumerate(
            segments=((0, 12), (12, 24)),
            base_starts=(0,),
            final_starts=(0,),
        )


def test_consumer_cannot_expand_beyond_authorized_base_valid_starts() -> None:
    with pytest.raises(ValueError, match="subset"):
        _enumerate(base_starts=(0,), final_starts=(0, 2))


def test_joint_support_tuple_is_replayed_over_exact_action_joints() -> None:
    first, second, valid, joint_valid, lengths = _inputs(16)
    joint_valid.zero_()
    joint_valid[:, :, [7, 8, 9, 10, 13]] = True

    with pytest.raises(RuntimeError, match="joint-support replay"):
        enumerate_native_window_pairs(
            first,
            second,
            valid,
            joint_valid,
            lengths,
            ("video-a",),
            (((0, 16),),),
            ((0,),),
            ((0,),),
            window_length=8,
            hop_frames=2,
            minimum_valid_frames_per_window=8,
            minimum_window_stable_action_joints=6,
            minimum_window_joint_support_fraction=1.0,
        )


def test_padded_tail_cannot_be_marked_valid() -> None:
    first, second, valid, joint_valid, _ = _inputs(24)

    with pytest.raises(ValueError, match="padded tail"):
        enumerate_native_window_pairs(
            first,
            second,
            valid,
            joint_valid,
            torch.tensor([20]),
            ("video-a",),
            (((0, 20),),),
            ((0,),),
            ((0,),),
            window_length=8,
            hop_frames=4,
            minimum_valid_frames_per_window=8,
            minimum_window_stable_action_joints=6,
            minimum_window_joint_support_fraction=0.75,
        )


def test_independent_temporal_and_pe_nulls_are_derangements() -> None:
    pairs = _enumerate()
    permuted_a, permuted_b, temporal = independently_permuted_pair_poses(
        pairs,
        seed=2026,
    )
    positions_a, positions_b, pe = independently_permuted_pair_position_indices(
        pairs,
        seed=2026,
    )

    assert temporal["side_seeds_are_distinct"] is True
    assert temporal["source_content_intersection_total"] == 0
    assert temporal["moved_frame_fraction"] == 1.0
    assert temporal["derangement_per_valid_window_side"] is True
    assert pe["side_seeds_are_distinct"] is True
    assert pe["moved_position_fraction"] == 1.0
    assert not torch.equal(permuted_a, pairs.poses_a)
    assert not torch.equal(permuted_b, pairs.poses_b)
    assert not torch.equal(positions_a, pairs.source_indices_a)
    assert not torch.equal(positions_b, pairs.source_indices_b)


class _PositionEchoEncoder(torch.nn.Module):
    embedding_dim = 1

    def __init__(self) -> None:
        super().__init__()
        self.observed_context_lengths: list[int] = []

    def forward(
        self,
        poses: torch.Tensor,
        valid: torch.Tensor,
        *,
        position_indices: torch.Tensor,
    ) -> torch.Tensor:
        self.observed_context_lengths.extend(int(value.sum()) for value in valid)
        return position_indices.to(dtype=poses.dtype).unsqueeze(-1)


def test_encoder_consumes_full_stable_range_before_embedding_windows_are_sliced() -> None:
    pairs = _enumerate(final_starts=(0, 4))
    first, second, valid, joint_valid, lengths = _inputs(24)
    views = AugmentedSequenceViews(
        video_ids=("video-a",),
        poses_a=first,
        poses_b=second,
        valid_mask=valid,
        joint_valid_mask=joint_valid,
        lengths=lengths,
        view_seeds=(11, 13),
    )
    contexts = pair_segment_contexts(pairs, views)
    encoder = _PositionEchoEncoder()

    embeddings_a, embeddings_b = encode_window_pairs(
        encoder,
        pairs,
        contexts,
        batch_size=1,
    )

    assert encoder.observed_context_lengths == [24, 24]
    assert embeddings_a[0, :, 0].tolist() == list(range(8))
    assert embeddings_b[0, :, 0].tolist() == list(range(8, 16))
    assert embeddings_a[1, :, 0].tolist() == list(range(4, 12))
    assert embeddings_b[1, :, 0].tolist() == list(range(12, 20))
    assert contexts.transform_contract["window_only_encoder_path_used"] is False
    assert contexts.transform_contract["unique_view_context_total"] == 2
    assert contexts.transform_contract["reused_pair_side_context_references"] == 2


def test_segment_context_nulls_keep_selected_a_b_content_disjoint() -> None:
    pairs = _enumerate(final_starts=(0,))
    first, second, valid, joint_valid, lengths = _inputs(24)
    views = AugmentedSequenceViews(
        video_ids=("video-a",),
        poses_a=first,
        poses_b=second,
        valid_mask=valid,
        joint_valid_mask=joint_valid,
        lengths=lengths,
        view_seeds=(11, 13),
    )
    contexts = pair_segment_contexts(pairs, views)

    temporal = independently_permuted_pair_segment_contexts(
        contexts,
        pairs,
        seed=2026,
    )
    pe = independently_permuted_pair_segment_positions(
        contexts,
        pairs,
        seed=2026,
    )
    joint_nulls = joint_support_null_segment_contexts(contexts)

    assert temporal.transform_contract["selected_window_moved_fraction"] == 1.0
    assert temporal.transform_contract[
        "selected_source_content_intersection_total"
    ] == 0
    assert pe.transform_contract["selected_moved_position_fraction"] == 1.0
    assert pe.transform_contract["selected_position_intersection_total"] == 0
    assert set(joint_nulls) == {
        "mask_flicker",
        "torso_only",
        "alternating_limb_dropout",
    }


def test_optimizer_guard_rejects_every_diagnostic_context_role() -> None:
    pairs = _enumerate(final_starts=(0,))
    first, second, valid, joint_valid, lengths = _inputs(24)
    views = AugmentedSequenceViews(
        video_ids=("video-a",),
        poses_a=first,
        poses_b=second,
        valid_mask=valid,
        joint_valid_mask=joint_valid,
        lengths=lengths,
        view_seeds=(11, 13),
    )
    real = pair_segment_contexts(pairs, views)
    diagnostics = [
        zero_pair_segment_contexts(real),
        independently_permuted_pair_segment_contexts(real, pairs, seed=2026),
        independently_permuted_pair_segment_positions(real, pairs, seed=2026),
        *joint_support_null_segment_contexts(real).values(),
    ]

    require_real_optimizer_contexts(real)
    assert real.objective_role == "real_optimizer"
    for controlled in diagnostics:
        assert controlled.objective_role.startswith("diagnostic_")
        with pytest.raises(
            ValueError,
            match="diagnostic pair-segment contexts cannot enter the real optimizer",
        ):
            require_real_optimizer_contexts(controlled)


def test_unified_2d_augmentation_preserves_mask_z_and_padding_exact_zero() -> None:
    config = load_conventional_cycleback_config(CONFIG)
    poses = torch.zeros((1, 16, 33, 3), dtype=torch.float32)
    valid = torch.ones((1, 16), dtype=torch.bool)
    joint_valid = torch.zeros((1, 16, 17), dtype=torch.bool)
    joint_valid[:, :, [7, 8, 9, 10, 13, 14]] = True
    poses[:, :, :17, :2] = joint_valid.unsqueeze(-1).to(dtype=poses.dtype)
    generator = torch.Generator().manual_seed(2026)

    augmented = augment_unified_2d_pose_batch(
        poses,
        valid,
        joint_valid,
        config.views.augmentation,
        generator=generator,
    )

    assert torch.count_nonzero(augmented[:, :, 17:, :]) == 0
    assert torch.count_nonzero(augmented[:, :, :17, 2]) == 0
    assert torch.count_nonzero(augmented[:, :, :17, :][~joint_valid]) == 0


def test_joint_support_nulls_are_deterministic_and_preserve_geometry() -> None:
    pairs = _enumerate()
    nulls = joint_support_null_poses(pairs)

    assert set(nulls) == {"mask_flicker", "torso_only", "alternating_limb_dropout"}
    for first, second, contract in nulls.values():
        assert contract["label_free"] is True
        assert contract["dropped_fraction"] > 0.0
        assert contract["frame_valid_mask_unchanged"] is True
        assert first.shape == pairs.poses_a.shape
        assert second.shape == pairs.poses_b.shape


def test_bounded_pair_cap_retains_one_pair_per_video_before_hash_fill() -> None:
    first, second, valid, joint_valid, lengths = _inputs(24, batch=3)
    pairs = enumerate_native_window_pairs(
        first,
        second,
        valid,
        joint_valid,
        lengths,
        ("video-a", "video-b", "video-c"),
        (((0, 24),), ((0, 24),), ((0, 24),)),
        ((0, 2, 4, 6, 8),) * 3,
        ((0, 2, 4, 6, 8),) * 3,
        window_length=8,
        hop_frames=2,
        minimum_valid_frames_per_window=8,
        minimum_window_stable_action_joints=6,
        minimum_window_joint_support_fraction=0.75,
    )

    capped = cap_window_pairs(
        pairs,
        maximum_pairs=3,
        seed=2026,
        purpose="unit-test",
    )

    assert capped.pair_count == 3
    assert set(capped.video_ids) == {"video-a", "video-b", "video-c"}
