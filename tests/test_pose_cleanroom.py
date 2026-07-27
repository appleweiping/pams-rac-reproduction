from __future__ import annotations

import inspect
from collections.abc import Callable

import pytest
import torch

from pams.baselines.pose_cleanroom import (
    BIGC_BODY_PARTS,
    POSE_CLEANROOM_DISCLOSURES,
    SPKDB_SALIENT_JOINTS,
    ActionTrigger,
    BIGCGraphModel,
    GMFLFeatureBuilder,
    GMFLLocalGlobalFusion,
    InterPartGraphBlock,
    IntraPartGraphBlock,
    JointWiseTemporalSelfSimilarity,
    JTSPSCountOnly,
    PoseRACV1,
    SPKDBDualBranch,
)


def _pose_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(2026)
    pose = torch.randn(2, 8, 33, 3, generator=generator, requires_grad=True)
    valid = torch.tensor(
        [
            [True, True, True, False, True, True, True, True],
            [True, False, True, True, True, True, False, True],
        ]
    )
    return pose, valid


def _assert_nonzero_finite_gradient(pose: torch.Tensor) -> None:
    assert pose.grad is not None
    assert torch.isfinite(pose.grad).all()
    assert torch.count_nonzero(pose.grad).item() > 0


def test_action_trigger_counts_alternating_salient_poses_without_oracle() -> None:
    # Channel zero completes low->high twice. Channel one is intentionally flat,
    # so oracle-free dynamic-range selection must choose channel zero.
    probabilities = torch.tensor(
        [
            [
                [0.1, 0.5],
                [0.1, 0.5],
                [0.9, 0.5],
                [0.9, 0.5],
                [0.1, 0.5],
                [0.1, 0.5],
                [0.9, 0.5],
                [0.9, 0.5],
                [0.1, 0.5],
            ]
        ]
    )
    result = ActionTrigger(momentum=0.0)(probabilities)
    assert result.counts.tolist() == [2]
    assert result.selected_channels.tolist() == [0]
    assert result.event_states.shape == (1, 9)
    assert result.smoothed_scores.shape == (1, 9)


def test_action_trigger_masked_excursion_cannot_create_a_count() -> None:
    scores = torch.tensor([0.1, 0.9, 0.1, 0.9, 0.1])
    valid = torch.tensor([True, False, True, False, True])
    result = ActionTrigger(momentum=0.0)(scores, valid)
    assert result.counts.item() == 0
    assert result.smoothed_scores[~valid].eq(0).all()


def test_poserac_v1_shapes_masks_and_gradients() -> None:
    pose, valid = _pose_fixture()
    model = PoseRACV1(
        num_action_channels=5,
        num_heads=3,
        num_layers=1,
        feedforward_dim=128,
        dropout=0.0,
    )
    output = model(pose, valid)
    assert output.logits.shape == (2, 8, 5)
    assert output.embeddings.shape == (2, 8, 99)
    assert output.logits[~valid].eq(0).all()
    assert output.embeddings[~valid].eq(0).all()
    output.logits[valid].sum().backward()
    _assert_nonzero_finite_gradient(pose)


def test_gmfl_modalities_are_well_formed_and_translation_invariant() -> None:
    pose, valid = _pose_fixture()
    builder = GMFLFeatureBuilder()
    features = builder(pose, valid)
    translated = builder(pose + 7.0, valid)
    assert features.coordinates.shape == (2, 8, 33, 3)
    assert features.distances.shape == (2, 8, 33, 33)
    assert features.angles.shape == (2, 8, 33, 33)
    assert torch.allclose(
        features.distances,
        features.distances.transpose(-1, -2),
        atol=1e-6,
    )
    assert torch.allclose(
        features.distances.diagonal(dim1=-2, dim2=-1),
        torch.zeros(2, 8, 33),
        atol=1e-6,
    )
    assert torch.allclose(features.coordinates, translated.coordinates, atol=2e-6)
    assert features.distances[~valid].eq(0).all()
    assert features.angles[~valid].eq(0).all()


def test_gmfl_local_global_fusion_shapes_masks_and_gradients() -> None:
    pose, valid = _pose_fixture()
    model = GMFLLocalGlobalFusion(
        num_outputs=5,
        model_dim=16,
        k_neighbors=3,
    )
    output = model(pose, valid)
    assert output.logits.shape == (2, 8, 5)
    assert output.fused_features.shape == (2, 8, 16)
    assert output.local_features.shape == output.global_features.shape == (2, 8, 16)
    assert output.logits[~valid].eq(0).all()
    output.logits[valid].square().mean().backward()
    _assert_nonzero_finite_gradient(pose)


def test_spkdb_uses_fixed_23_joint_branch_and_backpropagates() -> None:
    pose, valid = _pose_fixture()
    model = SPKDBDualBranch(
        num_outputs=5,
        model_dim=12,
        num_heads=3,
        fusion_dim=16,
        dropout=0.0,
    )
    output = model(pose, valid)
    assert tuple(range(23)) == SPKDB_SALIENT_JOINTS
    assert model.salient_indices.tolist() == list(range(23))
    assert output.logits.shape == (2, 8, 5)
    assert output.global_features.shape == output.salient_features.shape == (2, 8, 12)
    assert output.fused_features.shape == (2, 8, 16)
    assert output.logits[~valid].eq(0).all()
    output.fused_features[valid].sum().backward()
    _assert_nonzero_finite_gradient(pose)


def test_bigc_adjacencies_are_explicit_symmetric_and_model_is_differentiable() -> None:
    intra = IntraPartGraphBlock(8)
    inter = InterPartGraphBlock(8)
    assert intra.adjacency.shape == (33, 33)
    assert inter.adjacency.shape == (len(BIGC_BODY_PARTS), len(BIGC_BODY_PARTS))
    assert torch.allclose(intra.adjacency, intra.adjacency.T)
    assert torch.allclose(inter.adjacency, inter.adjacency.T)
    assert torch.count_nonzero(intra.adjacency - torch.diag(intra.adjacency.diag())) > 0
    assert torch.count_nonzero(inter.adjacency - torch.diag(inter.adjacency.diag())) > 0

    pose, valid = _pose_fixture()
    model = BIGCGraphModel(
        num_outputs=5,
        model_dim=8,
        intra_layers=1,
        inter_layers=1,
    )
    output = model(pose, valid)
    assert output.logits.shape == (2, 8, 5)
    assert output.joint_features.shape == (2, 8, 33, 8)
    assert output.part_features.shape == (2, 8, len(BIGC_BODY_PARTS), 8)
    assert output.logits[~valid].eq(0).all()
    output.logits[valid].sum().backward()
    _assert_nonzero_finite_gradient(pose)


def test_jtsps_similarity_masks_and_impulse_density_heads() -> None:
    pose, valid = _pose_fixture()
    similarity_module = JointWiseTemporalSelfSimilarity(embedding_dim=8)
    similarity = similarity_module(pose, valid)
    assert similarity.shape == (2, 33, 8, 8)
    assert torch.allclose(similarity, similarity.transpose(-1, -2), atol=1e-6)
    pair_valid = valid[:, None, :, None] & valid[:, None, None, :]
    assert similarity.masked_select(~pair_valid.expand_as(similarity)).eq(0).all()

    model = JTSPSCountOnly(joint_embedding_dim=8, hidden_dim=12)
    output = model(pose, valid)
    assert output.impulse_logits.shape == (2, 8)
    assert output.density.shape == (2, 8)
    assert output.temporal_features.shape == (2, 8, 12)
    assert output.impulse_logits[~valid].eq(0).all()
    assert output.density[~valid].eq(0).all()
    assert output.density[valid].ge(0).all()
    (output.impulse_logits[valid].sum() + output.density[valid].sum()).backward()
    _assert_nonzero_finite_gradient(pose)


@pytest.mark.parametrize(
    "inference_callable",
    [
        ActionTrigger.forward,
        PoseRACV1.forward,
        GMFLLocalGlobalFusion.forward,
        SPKDBDualBranch.forward,
        BIGCGraphModel.forward,
        JTSPSCountOnly.forward,
    ],
)
def test_inference_apis_cannot_accept_ground_truth_oracles(
    inference_callable: Callable[..., object],
) -> None:
    parameters = set(inspect.signature(inference_callable).parameters)
    assert "gt_count" not in parameters
    assert "gt_action" not in parameters
    assert "action_identity" not in parameters


def test_every_pose_scaffold_declares_inferences_and_parity_blockers() -> None:
    methods = {disclosure.method for disclosure in POSE_CLEANROOM_DISCLOSURES}
    assert methods == {
        "poserac-v1",
        "gmfl",
        "spkdb",
        "bigc",
        "jtsps-count-only",
    }
    assert all(item.inferred_parameters for item in POSE_CLEANROOM_DISCLOSURES)
    assert all(item.parity_blockers for item in POSE_CLEANROOM_DISCLOSURES)
