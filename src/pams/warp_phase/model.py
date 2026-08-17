"""The frozen WARP-PHASE phase network and masked identity recurrence.

The network deliberately accepts no source clock, track length, routing key, or
warp metadata.  Its 68 input channels are exactly 51 pose scalars followed by
17 joint-valid indicators.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

INPUT_CHANNELS = 68
HIDDEN_CHANNELS = 128
PHASE_CHANNELS = 2
EXPECTED_TRAINABLE_PARAMETERS = 225_026


@dataclass(frozen=True)
class WarpPhaseOutput:
    """Outputs from one complete, chronological identity traversal.

    ``hidden`` records the state after the hold/update decision at each clock.
    Invalid and padded clocks therefore repeat the preceding state.  ``phase``
    is emitted only at valid, non-padding frames and is exact zero elsewhere.
    ``commit_count`` counts non-padding source-clock decisions per identity.
    """

    phase: Tensor
    hidden: Tensor
    final_state: Tensor
    commit_count: Tensor


def count_trainable_parameters(module: nn.Module) -> int:
    """Return the number of scalar parameters whose gradients are enabled."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class WarpPhaseModel(nn.Module):
    """Conv68-128-128, masked GRU128, and normalized two-vector phase head.

    The two convolutions are the exact operators listed by the frozen
    architecture.  No unlisted trainable layer, auxiliary head, positional
    code, clock feature, or track-length feature is present.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            INPUT_CHANNELS,
            HIDDEN_CHANNELS,
            kernel_size=5,
            padding=2,
        )
        self.conv2 = nn.Conv1d(
            HIDDEN_CHANNELS,
            HIDDEN_CHANNELS,
            kernel_size=5,
            padding=2,
        )
        self.gru = nn.GRU(
            input_size=HIDDEN_CHANNELS,
            hidden_size=HIDDEN_CHANNELS,
            num_layers=1,
            batch_first=True,
        )
        self.phase_head = nn.Linear(HIDDEN_CHANNELS, PHASE_CHANNELS)

        actual = count_trainable_parameters(self)
        if actual != EXPECTED_TRAINABLE_PARAMETERS:
            raise RuntimeError(
                "frozen WARP-PHASE parameter count mismatch: "
                f"expected {EXPECTED_TRAINABLE_PARAMETERS}, got {actual}"
            )

    @staticmethod
    def _validate_inputs(
        pose: Tensor,
        joint_mask: Tensor,
        frame_mask: Tensor,
        clock_mask: Tensor,
    ) -> tuple[int, int]:
        if not isinstance(pose, Tensor):
            raise TypeError("pose must be a torch.Tensor")
        if pose.ndim != 4 or pose.shape[-2:] != (17, 3):
            raise ValueError("pose must have shape [batch, time, 17, 3]")
        if pose.dtype != torch.float32:
            raise TypeError("pose must have dtype torch.float32")
        if not bool(torch.isfinite(pose).all()):
            raise ValueError("pose must contain only finite values")

        batch, time = pose.shape[:2]
        if batch < 1 or time < 1:
            raise ValueError("pose batch and time dimensions must be positive")
        for name, mask, shape in (
            ("joint_mask", joint_mask, (batch, time, 17)),
            ("frame_mask", frame_mask, (batch, time)),
            ("clock_mask", clock_mask, (batch, time)),
        ):
            if not isinstance(mask, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if mask.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(mask.shape)}")
            if mask.dtype != torch.bool:
                raise TypeError(f"{name} must have dtype torch.bool")
            if mask.device != pose.device:
                raise ValueError(f"{name} must be on the same device as pose")

        if bool((frame_mask & ~clock_mask).any()):
            raise ValueError("a padded clock cannot be frame-valid")
        if bool((joint_mask & ~clock_mask.unsqueeze(-1)).any()):
            raise ValueError("a padded clock cannot contain a valid joint")
        if bool((frame_mask & (joint_mask.sum(dim=-1) < 8)).any()):
            raise ValueError("a frame-valid clock must contain at least eight valid joints")
        hip_valid = joint_mask[..., 11] | joint_mask[..., 12]
        if bool((frame_mask & ~hip_valid).any()):
            raise ValueError("a frame-valid clock must contain at least one valid hip")
        return batch, time

    @staticmethod
    def _input_channels(
        pose: Tensor,
        joint_mask: Tensor,
        frame_mask: Tensor,
        clock_mask: Tensor,
    ) -> Tensor:
        usable_frame = frame_mask & clock_mask
        usable_joint = joint_mask & usable_frame.unsqueeze(-1)
        masked_pose = pose.masked_fill(~usable_joint.unsqueeze(-1), 0.0)
        pose_channels = masked_pose.flatten(start_dim=2)
        mask_channels = usable_joint.to(dtype=pose.dtype)
        channels = torch.cat((pose_channels, mask_channels), dim=-1)
        if channels.shape[-1] != INPUT_CHANNELS:
            raise RuntimeError("internal WARP-PHASE channel construction failed")
        return channels.transpose(1, 2)

    def forward(
        self,
        pose: Tensor,
        joint_mask: Tensor,
        frame_mask: Tensor,
        clock_mask: Tensor,
    ) -> WarpPhaseOutput:
        """Traverse each complete identity exactly once from a zero state.

        Every non-padding source clock performs one state decision.  A valid
        frame commits the GRU candidate, while an invalid frame holds the
        preceding state.  Padding also holds state and is excluded from the
        per-sequence commit receipt.
        """

        batch, time = self._validate_inputs(pose, joint_mask, frame_mask, clock_mask)
        channels = self._input_channels(pose, joint_mask, frame_mask, clock_mask)
        features = self.conv2(self.conv1(channels)).transpose(1, 2)

        state = torch.zeros(
            (batch, HIDDEN_CHANNELS),
            dtype=features.dtype,
            device=features.device,
        )
        states: list[Tensor] = []
        for index in range(time):
            _output, candidate_state = self.gru(
                features[:, index : index + 1],
                state.unsqueeze(0),
            )
            candidate = candidate_state.squeeze(0)
            update = frame_mask[:, index] & clock_mask[:, index]
            state = torch.where(update.unsqueeze(-1), candidate, state)
            states.append(state)

        hidden = torch.stack(states, dim=1)
        raw_phase = self.phase_head(hidden)
        phase = raw_phase / (
            torch.linalg.vector_norm(raw_phase, dim=-1, keepdim=True) + 1e-8
        )
        phase_valid = frame_mask & clock_mask
        phase = phase.masked_fill(~phase_valid.unsqueeze(-1), 0.0)
        return WarpPhaseOutput(
            phase=phase,
            hidden=hidden,
            final_state=state,
            commit_count=clock_mask.sum(dim=1, dtype=torch.int64),
        )
