"""Clean-room architecture scaffolds for pose-based counting baselines.

These modules are independently authored from paper descriptions and public
configuration facts.  They are deliberately *not* registered as runnable
baselines: architecture code alone does not establish preprocessing,
annotation, checkpoint, or metric parity.  In particular, no inference API in
this module accepts a ground-truth count or ground-truth action identity.

The defaults marked ``inferred`` below are frozen engineering closures, not
claims about unpublished author implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pams.types import CountResult, PoseSequence

NUM_MEDIAPIPE_JOINTS = 33
POSE_FEATURE_DIM = NUM_MEDIAPIPE_JOINTS * 3

# MediaPipe indices 0--22 cover the face and upper-body/hand landmarks.  SPKDB
# discloses a fixed 23-joint salient branch but not a machine-readable index
# list; this choice is therefore an explicit clean-room inference.
SPKDB_SALIENT_JOINTS: tuple[int, ...] = tuple(range(23))

# Explicit undirected MediaPipe skeleton used by the BIGC scaffold.  Face-mesh
# detail is intentionally reduced to a chain because the paper does not
# disclose its exact graph.
MEDIAPIPE_INTRA_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    (29, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
    (30, 32),
)

BIGC_BODY_PARTS: tuple[tuple[int, ...], ...] = (
    tuple(range(11)),  # head
    (11, 13, 15, 17, 19, 21),  # left arm and hand
    (12, 14, 16, 18, 20, 22),  # right arm and hand
    (11, 12, 23, 24),  # torso
    (23, 25, 27, 29, 31),  # left leg
    (24, 26, 28, 30, 32),  # right leg
)
BIGC_INTER_PART_EDGES: tuple[tuple[int, int], ...] = (
    (0, 3),
    (1, 3),
    (2, 3),
    (3, 4),
    (3, 5),
    (1, 2),
    (4, 5),
)


@dataclass(frozen=True, slots=True)
class ScaffoldDisclosure:
    """Audit note explaining why a module is not a runnable paper baseline."""

    method: str
    inferred_parameters: tuple[str, ...]
    parity_blockers: tuple[str, ...]


POSE_CLEANROOM_DISCLOSURES: tuple[ScaffoldDisclosure, ...] = (
    ScaffoldDisclosure(
        method="poserac-v1",
        inferred_parameters=(
            "oracle-free channel selection by valid temporal dynamic range",
            "mask semantics for failed pose frames",
        ),
        parity_blockers=(
            "training-only pose-saliency annotations are not packaged",
            "released evaluator selects a channel using test ground-truth count",
        ),
    ),
    ScaffoldDisclosure(
        method="poserac-iconip24",
        inferred_parameters=(
            "33-joint MediaPipe input in place of the disclosed 3D OpenPose estimator",
            "oracle-free action-channel selection by valid temporal dynamic range",
            "Transformer hidden width, attention heads, and feed-forward width",
        ),
        parity_blockers=(
            "no public ICONIP'24 implementation or checkpoint was located",
            "UCFRep salient-pose definitions and generated training images are not disclosed",
            "the disclosed UniFormerV2-L action recognizer cannot consume PoseSequence inputs",
        ),
    ),
    ScaffoldDisclosure(
        method="gmfl",
        inferred_parameters=(
            "hip-centred coordinate/distance/angle definitions",
            "k-nearest local aggregation and multiplicative global fusion",
        ),
        parity_blockers=("no public author implementation or checkpoint located",),
    ),
    ScaffoldDisclosure(
        method="spkdb",
        inferred_parameters=(
            "salient indices are fixed to MediaPipe landmarks 0--22",
            "CBR temporal fusion details",
        ),
        parity_blockers=("no public author implementation or checkpoint located",),
    ),
    ScaffoldDisclosure(
        method="bigc",
        inferred_parameters=(
            "six body-part definitions",
            "joint-level and part-level adjacency matrices",
        ),
        parity_blockers=("no public author implementation or checkpoint located",),
    ),
    ScaffoldDisclosure(
        method="jtsps-count-only",
        inferred_parameters=("self-similarity convolution and impulse/density heads",),
        parity_blockers=(
            "public material does not uniquely determine the source protocol",
            "cycle-density supervision and decoding policy remain unspecified",
        ),
    ),
)


def _validate_pose_inputs(
    pose: Tensor,
    valid_mask: Tensor | None,
    *,
    expected_joints: int = NUM_MEDIAPIPE_JOINTS,
) -> tuple[Tensor, Tensor]:
    """Validate a batch-first pose tensor and canonicalize its frame mask."""

    if pose.ndim != 4 or pose.shape[2:] != (expected_joints, 3):
        raise ValueError(
            f"pose must have shape [batch, time, {expected_joints}, 3], got {tuple(pose.shape)}"
        )
    if not pose.is_floating_point():
        pose = pose.float()
    batch, time = pose.shape[:2]
    if valid_mask is None:
        valid = torch.ones((batch, time), dtype=torch.bool, device=pose.device)
    else:
        if valid_mask.shape != (batch, time):
            raise ValueError(
                f"valid_mask must have shape {(batch, time)}, got {tuple(valid_mask.shape)}"
            )
        valid = valid_mask.to(device=pose.device, dtype=torch.bool)
    return pose, valid


def _masked_frames(values: Tensor, valid: Tensor) -> Tensor:
    """Set invalid frame rows to exact zero, preserving trailing dimensions."""

    mask = valid
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return values.masked_fill(~mask, 0.0)


class TriggerOutput(NamedTuple):
    """Oracle-free hysteresis result for a batch of score sequences."""

    counts: Tensor
    selected_channels: Tensor
    event_states: Tensor
    smoothed_scores: Tensor


class ActionTrigger(nn.Module):
    """Count alternating high/low salient poses without an action oracle.

    Inputs may contain one score channel (``[B,T]``) or several candidate
    channels (``[B,T,C]``).  For multiple channels, the channel with the
    largest valid-frame dynamic range is selected independently per video.
    This deterministic policy does not read action identity or count labels.

    A repetition is completed when two opposing salient-pose completion events
    alternate in the same direction as the first observed pair.  The logic is
    the stateless batched equivalent of two complementary hysteresis triggers.
    """

    def __init__(
        self,
        enter_threshold: float = 0.78,
        exit_threshold: float = 0.4,
        momentum: float = 0.4,
    ) -> None:
        super().__init__()
        if not 0.5 < enter_threshold <= 1.0:
            raise ValueError("enter_threshold must be in (0.5, 1]")
        if not 0.0 <= exit_threshold < 0.5:
            raise ValueError("exit_threshold must be in [0, 0.5)")
        if exit_threshold >= enter_threshold:
            raise ValueError("exit_threshold must be below enter_threshold")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.enter_threshold = float(enter_threshold)
        self.exit_threshold = float(exit_threshold)
        self.momentum = float(momentum)

    @staticmethod
    def _shape_scores(
        probabilities: Tensor,
        valid_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, bool]:
        squeeze_batch = probabilities.ndim == 1
        if probabilities.ndim == 1:
            probabilities = probabilities[None, :, None]
        elif probabilities.ndim == 2:
            probabilities = probabilities[:, :, None]
        elif probabilities.ndim != 3:
            raise ValueError("probabilities must have shape [T], [B,T], or [B,T,C]")
        if not probabilities.is_floating_point():
            probabilities = probabilities.float()
        batch, time, _ = probabilities.shape
        if valid_mask is None:
            valid = torch.ones((batch, time), dtype=torch.bool, device=probabilities.device)
        else:
            if squeeze_batch and valid_mask.shape == (time,):
                valid_mask = valid_mask.unsqueeze(0)
            if valid_mask.shape != (batch, time):
                raise ValueError(
                    f"valid_mask must have shape {(batch, time)}, got {tuple(valid_mask.shape)}"
                )
            valid = valid_mask.to(device=probabilities.device, dtype=torch.bool)
        if not torch.isfinite(probabilities[valid]).all():
            raise ValueError("valid probabilities must be finite")
        return probabilities.clamp(0.0, 1.0), valid, squeeze_batch

    def forward(
        self,
        probabilities: Tensor,
        valid_mask: Tensor | None = None,
    ) -> TriggerOutput:
        scores, valid, squeeze_batch = self._shape_scores(probabilities, valid_mask)
        batch, time, channels = scores.shape

        # Source configuration initializes the exponential moving average at
        # 0.5. Invalid frames hold their previous value and never trigger.
        previous = torch.full((batch, channels), 0.5, dtype=scores.dtype, device=scores.device)
        smoothed_all: list[Tensor] = []
        for frame_index in range(time):
            candidate = scores[:, frame_index] * (1.0 - self.momentum) + previous * self.momentum
            previous = torch.where(
                valid[:, frame_index, None],
                candidate,
                previous,
            )
            smoothed_all.append(previous)
        smoothed = torch.stack(smoothed_all, dim=1)

        positive_inf = torch.tensor(torch.inf, dtype=smoothed.dtype, device=smoothed.device)
        negative_inf = -positive_inf
        maximum = smoothed.masked_fill(~valid[:, :, None], negative_inf).amax(dim=1)
        minimum = smoothed.masked_fill(~valid[:, :, None], positive_inf).amin(dim=1)
        dynamic_range = maximum - minimum
        no_valid = ~valid.any(dim=1)
        dynamic_range = torch.where(
            no_valid[:, None],
            torch.zeros_like(dynamic_range),
            dynamic_range,
        )
        selected = dynamic_range.argmax(dim=1)
        gather_index = selected[:, None, None].expand(batch, time, 1)
        chosen = smoothed.gather(dim=2, index=gather_index).squeeze(2)

        counts = torch.zeros(batch, dtype=torch.long, device=scores.device)
        event_states = torch.full((batch, time), -1, dtype=torch.int8, device=scores.device)
        high_armed = torch.zeros(batch, dtype=torch.bool, device=scores.device)
        low_armed = torch.zeros(batch, dtype=torch.bool, device=scores.device)
        first_event = torch.full((batch,), -1, dtype=torch.int8, device=scores.device)
        last_event = torch.full((batch,), -1, dtype=torch.int8, device=scores.device)

        low_enter = 1.0 - self.enter_threshold
        low_exit = 1.0 - self.exit_threshold
        for frame_index in range(time):
            current = chosen[:, frame_index]
            frame_valid = valid[:, frame_index]
            high_armed = high_armed | (frame_valid & (current > self.enter_threshold))
            low_armed = low_armed | (frame_valid & (current < low_enter))
            high_event = frame_valid & high_armed & (current < self.exit_threshold)
            low_event = frame_valid & low_armed & (current > low_exit)
            high_armed = high_armed & ~high_event
            low_armed = low_armed & ~low_event

            event = torch.where(
                high_event,
                torch.ones_like(first_event),
                torch.where(low_event, torch.zeros_like(first_event), -1),
            )
            is_event = event >= 0
            first_event = torch.where(is_event & (first_event < 0), event, first_event)
            completes_cycle = is_event & (event != first_event) & (last_event == first_event)
            counts = counts + completes_cycle.to(dtype=torch.long)
            last_event = torch.where(is_event, event, last_event)
            event_states[:, frame_index] = last_event

        chosen = chosen.masked_fill(~valid, 0.0)
        if squeeze_batch:
            return TriggerOutput(
                counts=counts.squeeze(0),
                selected_channels=selected.squeeze(0),
                event_states=event_states.squeeze(0),
                smoothed_scores=chosen.squeeze(0),
            )
        return TriggerOutput(counts, selected, event_states, chosen)


class PoseRACOutput(NamedTuple):
    """Per-frame action-state logits and their 99D Transformer embeddings."""

    logits: Tensor
    embeddings: Tensor


class PoseRACV1(nn.Module):
    """Clean-room scaffold of the released 99D PoseRAC-v1 model.

    The public model reshapes each flattened 33x3 pose into a sequence of one
    99D token.  Consequently attention is frame-independent; this class keeps
    that behavior for source diagnostics rather than silently converting it
    into a temporal Transformer.
    """

    def __init__(
        self,
        num_action_channels: int,
        *,
        feature_dim: int = POSE_FEATURE_DIM,
        num_heads: int = 9,
        num_layers: int = 6,
        feedforward_dim: int = 396,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if feature_dim != POSE_FEATURE_DIM:
            raise ValueError("PoseRAC-v1 requires flattened 33x3 = 99D input")
        if num_action_channels < 1:
            raise ValueError("num_action_channels must be positive")
        if feature_dim % num_heads:
            raise ValueError("feature_dim must be divisible by num_heads")
        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(feature_dim, num_action_channels)
        self.num_action_channels = num_action_channels

    def forward(
        self,
        pose: Tensor,
        valid_mask: Tensor | None = None,
    ) -> PoseRACOutput:
        pose, valid = _validate_pose_inputs(pose, valid_mask)
        batch, time = pose.shape[:2]
        tokens = pose.reshape(batch * time, 1, POSE_FEATURE_DIM)
        embeddings = self.encoder(tokens).reshape(batch, time, POSE_FEATURE_DIM)
        logits = self.classifier(embeddings)
        return PoseRACOutput(
            logits=_masked_frames(logits, valid),
            embeddings=_masked_frames(embeddings, valid),
        )


class PoseRACICONIP24Output(NamedTuple):
    """Per-frame action scores and spatial pose/query features."""

    logits: Tensor
    pose_features: Tensor
    query_features: Tensor


class PoseRACICONIP24(nn.Module):
    """Clean-room architecture scaffold of the distinct ICONIP'24 PoseRAC network.

    Unlike :class:`PoseRACV1`, this model treats joints as spatial tokens for
    every frame.  A pose-wise Transformer encoder produces joint features and
    a Transformer decoder cross-attends one query per action to those features.
    The published architecture specifies six encoder and two decoder layers but
    omits the hidden widths; the defaults below are therefore explicit inferred
    closures.  No temporal context, action identity, or count label enters this
    module.
    """

    def __init__(
        self,
        num_action_channels: int,
        *,
        num_joints: int = NUM_MEDIAPIPE_JOINTS,
        coordinate_dim: int = 3,
        model_dim: int = 96,
        num_heads: int = 8,
        encoder_layers: int = 6,
        decoder_layers: int = 2,
        feedforward_dim: int = 384,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_action_channels < 1:
            raise ValueError("num_action_channels must be positive")
        if num_joints < 1 or coordinate_dim < 1:
            raise ValueError("num_joints and coordinate_dim must be positive")
        if model_dim < 1 or model_dim % num_heads:
            raise ValueError("model_dim must be positive and divisible by num_heads")
        if encoder_layers < 1 or decoder_layers < 1:
            raise ValueError("encoder_layers and decoder_layers must be positive")

        self.num_action_channels = int(num_action_channels)
        self.num_joints = int(num_joints)
        self.coordinate_dim = int(coordinate_dim)
        self.model_dim = int(model_dim)
        self.joint_embedding = nn.Sequential(
            nn.Linear(coordinate_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.pose_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
            enable_nested_tensor=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.pose_text_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers,
        )
        self.action_queries = nn.Parameter(torch.empty(num_action_channels, model_dim))
        nn.init.normal_(self.action_queries, mean=0.0, std=model_dim**-0.5)
        self.score_head = nn.Linear(model_dim, 1)

    def forward(
        self,
        pose: Tensor,
        valid_mask: Tensor | None = None,
        *,
        action_queries: Tensor | None = None,
    ) -> PoseRACICONIP24Output:
        pose, valid = _validate_pose_inputs(
            pose,
            valid_mask,
            expected_joints=self.num_joints,
        )
        if pose.shape[-1] != self.coordinate_dim:
            raise ValueError(
                f"pose coordinate dimension must be {self.coordinate_dim}, "
                f"got {pose.shape[-1]}"
            )
        queries = self.action_queries if action_queries is None else action_queries
        expected_query_shape = (self.num_action_channels, self.model_dim)
        if queries.shape != expected_query_shape:
            raise ValueError(
                f"action_queries must have shape {expected_query_shape}, got {tuple(queries.shape)}"
            )

        batch, time, joints, _ = pose.shape
        frame_count = batch * time
        joint_tokens = self.joint_embedding(pose).reshape(frame_count, joints, self.model_dim)
        pose_features = self.pose_encoder(joint_tokens)
        query_tokens = queries.to(device=pose.device, dtype=pose_features.dtype)
        query_tokens = query_tokens.unsqueeze(0).expand(frame_count, -1, -1)
        query_features = self.pose_text_decoder(query_tokens, pose_features)
        logits = self.score_head(query_features).squeeze(-1)

        logits = logits.reshape(batch, time, self.num_action_channels)
        pose_features = pose_features.reshape(batch, time, joints, self.model_dim)
        query_features = query_features.reshape(
            batch,
            time,
            self.num_action_channels,
            self.model_dim,
        )
        return PoseRACICONIP24Output(
            logits=_masked_frames(logits, valid),
            pose_features=_masked_frames(pose_features, valid),
            query_features=_masked_frames(query_features, valid),
        )


@dataclass(slots=True)
class PoseRACICONIP24Adapter:
    """Smoke-only, oracle-free adapter around a trained clean-room model.

    The paper selects one action channel with an RGB UniFormerV2-L recognizer.
    ``PoseSequence`` intentionally contains no RGB frames, so this preparation
    adapter uses the frozen dynamic-range selector in :class:`ActionTrigger`.
    That substitution is inferred and makes results ineligible for the paper
    table until a separately audited action recognizer and trained checkpoint
    are supplied.
    """

    model: PoseRACICONIP24
    trigger: ActionTrigger
    device: str = "cpu"

    @property
    def spec(self):  # type: ignore[no-untyped-def]
        # Imported lazily to avoid coupling architecture scaffolds to registry
        # construction. The registry deliberately remains blocked.
        from pams.baselines.registry import get_baseline_spec

        return get_baseline_spec("poserac-iconip24")

    def predict(self, sample: PoseSequence) -> CountResult:
        device = torch.device(self.device)
        pose = torch.from_numpy(np.array(sample.xyz, copy=True)).unsqueeze(0).to(device)
        valid = torch.from_numpy(np.array(sample.valid_mask, copy=True)).unsqueeze(0).to(device)
        self.model.to(device)
        self.model.eval()
        self.trigger.to(device)
        self.trigger.eval()
        with torch.inference_mode():
            output = self.model(pose, valid)
            probabilities = torch.sigmoid(output.logits)
            triggered = self.trigger(probabilities, valid)

        count = int(triggered.counts.item())
        stream = triggered.smoothed_scores.squeeze(0).detach().cpu().numpy().astype(np.float32)
        valid_numpy = np.asarray(sample.valid_mask, dtype=bool)
        valid_stream = stream[valid_numpy]
        confidence = (
            float(np.clip(np.ptp(valid_stream), 0.0, 1.0)) if valid_stream.size else 0.0
        )
        valid_frames = max(int(np.count_nonzero(valid_numpy)), 1)
        period_frames = float(valid_frames / count) if count else float(valid_frames)
        return CountResult(
            count=count,
            period_frames=period_frames,
            expert_counts=(count, count, count),
            confidence=confidence,
            period_stream=stream,
        )


class GMFLFeatures(NamedTuple):
    """Hip-centred coordinate, pairwise-distance, and angle features."""

    coordinates: Tensor
    distances: Tensor
    angles: Tensor


class GMFLFeatureBuilder(nn.Module):
    """Construct the three pose modalities described by GMFL.

    Pairwise angle is the cosine between hip-centred joint rays.  The exact
    angle convention is not disclosed by the source and remains an inferred
    choice pending parity evidence.
    """

    def forward(
        self,
        pose: Tensor,
        valid_mask: Tensor | None = None,
    ) -> GMFLFeatures:
        pose, valid = _validate_pose_inputs(pose, valid_mask)
        hip_center = (pose[:, :, 23] + pose[:, :, 24]) * 0.5
        coordinates = pose - hip_center.unsqueeze(2)
        distances = torch.cdist(coordinates, coordinates, p=2)
        # Some CUDA/PyTorch kernels accumulate the two traversal directions
        # differently by a few ulps.  A pairwise Euclidean distance is
        # symmetric by definition, so canonicalize it before kNN selection.
        distances = (distances + distances.transpose(-1, -2)) * 0.5
        diagonal = torch.eye(
            NUM_MEDIAPIPE_JOINTS,
            dtype=torch.bool,
            device=distances.device,
        ).view(1, 1, NUM_MEDIAPIPE_JOINTS, NUM_MEDIAPIPE_JOINTS)
        distances = distances.masked_fill(diagonal, 0.0)
        directions = F.normalize(coordinates, dim=-1, eps=1e-8)
        angles = torch.einsum("btid,btjd->btij", directions, directions)
        return GMFLFeatures(
            coordinates=_masked_frames(coordinates, valid),
            distances=_masked_frames(distances, valid),
            angles=_masked_frames(angles, valid),
        )


class GMFLOutput(NamedTuple):
    """GMFL scaffold outputs for loss construction and audit inspection."""

    logits: Tensor
    fused_features: Tensor
    local_features: Tensor
    global_features: Tensor


class GMFLLocalGlobalFusion(nn.Module):
    """Inferred kNN local-interaction and multiplicative global fusion."""

    def __init__(
        self,
        num_outputs: int,
        *,
        model_dim: int = 64,
        k_neighbors: int = 4,
        num_joints: int = NUM_MEDIAPIPE_JOINTS,
    ) -> None:
        super().__init__()
        if num_outputs < 1 or model_dim < 1:
            raise ValueError("num_outputs and model_dim must be positive")
        if not 1 <= k_neighbors < num_joints:
            raise ValueError("k_neighbors must be in [1, num_joints)")
        self.num_joints = num_joints
        self.k_neighbors = k_neighbors
        self.features = GMFLFeatureBuilder()
        self.coordinate_projection = nn.Linear(3, model_dim)
        self.relation_projection = nn.Linear(2 * num_joints, model_dim)
        self.local_projection = nn.Linear(model_dim, model_dim)
        self.global_projection = nn.Linear(model_dim, model_dim)
        self.fusion = nn.Sequential(
            nn.Linear(3 * model_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.classifier = nn.Linear(model_dim, num_outputs)

    def forward(
        self,
        pose: Tensor,
        valid_mask: Tensor | None = None,
    ) -> GMFLOutput:
        pose, valid = _validate_pose_inputs(pose, valid_mask, expected_joints=self.num_joints)
        modalities = self.features(pose, valid)
        relation = torch.cat((modalities.distances, modalities.angles), dim=-1)
        nodes = F.gelu(
            self.coordinate_projection(modalities.coordinates) + self.relation_projection(relation)
        )

        eye = torch.eye(self.num_joints, dtype=torch.bool, device=pose.device).view(
            1, 1, self.num_joints, self.num_joints
        )
        ranking_distance = modalities.distances.masked_fill(eye, torch.inf)
        neighbor_indices = ranking_distance.topk(self.k_neighbors, dim=-1, largest=False).indices
        weights = torch.zeros_like(ranking_distance)
        weights.scatter_(-1, neighbor_indices, 1.0 / self.k_neighbors)
        local_nodes = torch.einsum("btij,btjd->btid", weights, nodes)
        local_features = local_nodes.mean(dim=2)
        global_features = nodes.mean(dim=2)
        bilinear = torch.tanh(
            self.local_projection(local_features) * self.global_projection(global_features)
        )
        fused = self.fusion(torch.cat((local_features, global_features, bilinear), dim=-1))
        logits = self.classifier(fused)
        return GMFLOutput(
            logits=_masked_frames(logits, valid),
            fused_features=_masked_frames(fused, valid),
            local_features=_masked_frames(local_features, valid),
            global_features=_masked_frames(global_features, valid),
        )


class _JointAttentionBranch(nn.Module):
    """Per-frame joint-attention branch shared by the SPKDB scaffold."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.input_projection = nn.Linear(3, model_dim)
        self.attention = nn.MultiheadAttention(
            model_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.normalization = nn.LayerNorm(model_dim)

    def forward(self, pose: Tensor) -> Tensor:
        batch, time, joints, _ = pose.shape
        tokens = self.input_projection(pose).reshape(batch * time, joints, -1)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.normalization(tokens + attended)
        return tokens.mean(dim=1).reshape(batch, time, -1)


class SPKDBOutput(NamedTuple):
    """Global/salient branch features and fused action-state logits."""

    logits: Tensor
    fused_features: Tensor
    global_features: Tensor
    salient_features: Tensor


class SPKDBDualBranch(nn.Module):
    """Global 33-joint and fixed salient 23-joint clean-room scaffold.

    The CBR layer is an inferred Conv1d--BatchNorm--ReLU temporal fusion.  The
    module remains blocked from registry execution until annotation and
    checkpoint parity are established.
    """

    def __init__(
        self,
        num_outputs: int,
        *,
        model_dim: int = 64,
        num_heads: int = 4,
        fusion_dim: int = 128,
        temporal_kernel: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_outputs < 1 or fusion_dim < 1:
            raise ValueError("num_outputs and fusion_dim must be positive")
        if temporal_kernel < 1 or temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be a positive odd integer")
        self.global_branch = _JointAttentionBranch(model_dim, num_heads, dropout)
        self.salient_branch = _JointAttentionBranch(model_dim, num_heads, dropout)
        self.salient_indices: Tensor
        self.register_buffer(
            "salient_indices",
            torch.tensor(SPKDB_SALIENT_JOINTS, dtype=torch.long),
            persistent=True,
        )
        self.cbr = nn.Sequential(
            nn.Conv1d(
                2 * model_dim,
                fusion_dim,
                temporal_kernel,
                padding=temporal_kernel // 2,
            ),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(fusion_dim, num_outputs)

    def forward(
        self,
        pose: Tensor,
        valid_mask: Tensor | None = None,
    ) -> SPKDBOutput:
        pose, valid = _validate_pose_inputs(pose, valid_mask)
        global_features = self.global_branch(pose)
        salient_pose = pose.index_select(2, self.salient_indices)
        salient_features = self.salient_branch(salient_pose)
        combined = torch.cat((global_features, salient_features), dim=-1)
        combined = _masked_frames(combined, valid)
        fused = self.cbr(combined.transpose(1, 2)).transpose(1, 2)
        logits = self.classifier(fused)
        return SPKDBOutput(
            logits=_masked_frames(logits, valid),
            fused_features=_masked_frames(fused, valid),
            global_features=_masked_frames(global_features, valid),
            salient_features=_masked_frames(salient_features, valid),
        )


def normalized_adjacency(
    num_nodes: int,
    edges: tuple[tuple[int, int], ...],
) -> Tensor:
    """Build an explicit symmetric ``D^-1/2 (A+I) D^-1/2`` adjacency."""

    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    adjacency = torch.eye(num_nodes, dtype=torch.float32)
    for source, target in edges:
        if not (0 <= source < num_nodes and 0 <= target < num_nodes):
            raise ValueError(f"edge {(source, target)} is outside {num_nodes} nodes")
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    inverse_sqrt = degree.rsqrt()
    return inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]


class _GraphBlock(nn.Module):
    """Residual graph message-passing block with a frozen adjacency."""

    def __init__(self, dimension: int, adjacency: Tensor) -> None:
        super().__init__()
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.adjacency: Tensor
        self.register_buffer("adjacency", adjacency, persistent=True)
        self.self_projection = nn.Linear(dimension, dimension)
        self.neighbor_projection = nn.Linear(dimension, dimension)
        self.normalization = nn.LayerNorm(dimension)

    def forward(self, nodes: Tensor) -> Tensor:
        if nodes.ndim != 4 or nodes.shape[2] != self.adjacency.shape[0]:
            raise ValueError(
                f"nodes must have shape [batch,time,{self.adjacency.shape[0]},dimension]"
            )
        neighbors = torch.einsum(
            "ij,btjd->btid",
            self.adjacency.to(dtype=nodes.dtype),
            nodes,
        )
        update = F.gelu(self.self_projection(nodes) + self.neighbor_projection(neighbors))
        return self.normalization(nodes + update)


class IntraPartGraphBlock(_GraphBlock):
    """Joint-level BIGC block with the explicit MediaPipe adjacency."""

    def __init__(self, dimension: int) -> None:
        super().__init__(
            dimension,
            normalized_adjacency(NUM_MEDIAPIPE_JOINTS, MEDIAPIPE_INTRA_EDGES),
        )


class InterPartGraphBlock(_GraphBlock):
    """Six-node body-part BIGC block with an explicit part adjacency."""

    def __init__(self, dimension: int) -> None:
        super().__init__(
            dimension,
            normalized_adjacency(len(BIGC_BODY_PARTS), BIGC_INTER_PART_EDGES),
        )


class BIGCOutput(NamedTuple):
    """BIGC joint/part graph features and per-frame logits."""

    logits: Tensor
    fused_features: Tensor
    joint_features: Tensor
    part_features: Tensor


class BIGCGraphModel(nn.Module):
    """Inferred body inter/intra-part graph clean-room scaffold."""

    def __init__(
        self,
        num_outputs: int,
        *,
        model_dim: int = 64,
        intra_layers: int = 2,
        inter_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_outputs < 1 or model_dim < 1:
            raise ValueError("num_outputs and model_dim must be positive")
        if intra_layers < 1 or inter_layers < 1:
            raise ValueError("graph layer counts must be positive")
        self.input_projection = nn.Linear(3, model_dim)
        self.intra_blocks = nn.ModuleList(
            IntraPartGraphBlock(model_dim) for _ in range(intra_layers)
        )
        self.inter_blocks = nn.ModuleList(
            InterPartGraphBlock(model_dim) for _ in range(inter_layers)
        )
        membership = torch.zeros((len(BIGC_BODY_PARTS), NUM_MEDIAPIPE_JOINTS), dtype=torch.float32)
        for part_index, joints in enumerate(BIGC_BODY_PARTS):
            membership[part_index, list(joints)] = 1.0 / len(joints)
        self.part_membership: Tensor
        self.register_buffer("part_membership", membership, persistent=True)
        self.fusion = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.classifier = nn.Linear(model_dim, num_outputs)

    def forward(
        self,
        pose: Tensor,
        valid_mask: Tensor | None = None,
    ) -> BIGCOutput:
        pose, valid = _validate_pose_inputs(pose, valid_mask)
        joints = self.input_projection(pose)
        for block in self.intra_blocks:
            joints = block(joints)
        parts = torch.einsum(
            "pj,btjd->btpd",
            self.part_membership.to(dtype=joints.dtype),
            joints,
        )
        for block in self.inter_blocks:
            parts = block(parts)
        fused = self.fusion(torch.cat((joints.mean(dim=2), parts.mean(dim=2)), dim=-1))
        logits = self.classifier(fused)
        return BIGCOutput(
            logits=_masked_frames(logits, valid),
            fused_features=_masked_frames(fused, valid),
            joint_features=_masked_frames(joints, valid),
            part_features=_masked_frames(parts, valid),
        )


class JointWiseTemporalSelfSimilarity(nn.Module):
    """Build one cosine temporal self-similarity matrix per joint."""

    def __init__(
        self,
        embedding_dim: int = 32,
        *,
        num_joints: int = NUM_MEDIAPIPE_JOINTS,
    ) -> None:
        super().__init__()
        if embedding_dim < 1 or num_joints < 1:
            raise ValueError("embedding_dim and num_joints must be positive")
        self.num_joints = num_joints
        self.projection = nn.Linear(3, embedding_dim)

    def forward(
        self,
        pose: Tensor,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        pose, valid = _validate_pose_inputs(pose, valid_mask, expected_joints=self.num_joints)
        embeddings = F.normalize(self.projection(pose), dim=-1, eps=1e-8)
        similarity = torch.einsum("btjd,bsjd->bjts", embeddings, embeddings)
        pair_valid = valid[:, None, :, None] & valid[:, None, None, :]
        return similarity.masked_fill(~pair_valid, 0.0)


class JTSPSOutput(NamedTuple):
    """Joint-wise TSM with impulse logits and non-negative density."""

    impulse_logits: Tensor
    density: Tensor
    similarity: Tensor
    temporal_features: Tensor


class JTSPSCountOnly(nn.Module):
    """Count-only JTSPS architecture scaffold.

    The joint-wise TSM is reduced by an inferred 2D convolution, followed by
    separate impulse and softplus density heads.  Decoding and supervision are
    intentionally outside this primitive because their source protocol remains
    unverifiable.
    """

    def __init__(
        self,
        *,
        joint_embedding_dim: int = 32,
        hidden_dim: int = 64,
        num_joints: int = NUM_MEDIAPIPE_JOINTS,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.self_similarity = JointWiseTemporalSelfSimilarity(
            joint_embedding_dim,
            num_joints=num_joints,
        )
        self.similarity_encoder = nn.Sequential(
            nn.Conv2d(num_joints, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.temporal_refinement = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.impulse_head = nn.Linear(hidden_dim, 1)
        self.density_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        pose: Tensor,
        valid_mask: Tensor | None = None,
    ) -> JTSPSOutput:
        pose, valid = _validate_pose_inputs(
            pose,
            valid_mask,
            expected_joints=self.self_similarity.num_joints,
        )
        similarity = self.self_similarity(pose, valid)
        encoded = self.similarity_encoder(similarity)
        target_weights = valid[:, None, None, :].to(dtype=encoded.dtype)
        denominator = target_weights.sum(dim=-1).clamp_min(1.0)
        temporal = (encoded * target_weights).sum(dim=-1) / denominator
        temporal = self.temporal_refinement(temporal).transpose(1, 2)
        impulse = self.impulse_head(temporal).squeeze(-1)
        density = F.softplus(self.density_head(temporal).squeeze(-1))
        return JTSPSOutput(
            impulse_logits=_masked_frames(impulse, valid),
            density=_masked_frames(density, valid),
            similarity=similarity,
            temporal_features=_masked_frames(temporal, valid),
        )
