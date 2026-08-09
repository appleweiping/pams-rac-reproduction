"""Native-timeline window pairing for the cycle-back proxy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class NativeWindowPairBatch:
    """Eligible consecutive window pairs from independently augmented views.

    ``source_indices_*`` are absolute zero-based indices into the native pose
    timeline.  They are never compacted around invalid frames or normalized to
    the padded batch length.
    """

    video_ids: tuple[str, ...]
    poses_a: Tensor
    poses_b: Tensor
    valid_a: Tensor
    valid_b: Tensor
    joint_valid_a: Tensor
    joint_valid_b: Tensor
    source_indices_a: Tensor
    source_indices_b: Tensor
    native_lengths: Tensor
    starts_a: Tensor
    starts_b: Tensor
    segment_starts: Tensor
    segment_ends: Tensor
    source_video_indices: Tensor
    raw_grid_pair_counts: tuple[int, ...]
    available_pair_counts: tuple[int, ...]
    eligible_pair_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.poses_a.ndim != 4 or self.poses_a.shape[2:] != (33, 3):
            raise ValueError("window poses must have shape [pairs, window, 33, 3]")
        if self.poses_b.shape != self.poses_a.shape:
            raise ValueError("paired pose windows must have identical shapes")
        if not self.poses_a.is_floating_point() or not self.poses_b.is_floating_point():
            raise TypeError("paired pose windows must use a floating-point dtype")
        if self.poses_b.dtype != self.poses_a.dtype:
            raise TypeError("paired pose windows must use the same dtype")
        pair_count, window = self.poses_a.shape[:2]
        if len(self.video_ids) != pair_count:
            raise ValueError("video_ids must match pair count")
        if any(not str(identifier).strip() for identifier in self.video_ids):
            raise ValueError("window-pair video IDs must be non-empty")
        for name, value in (
            ("valid_a", self.valid_a),
            ("valid_b", self.valid_b),
        ):
            if value.shape != (pair_count, window) or value.dtype is not torch.bool:
                raise ValueError(f"{name} must be boolean [pairs, window]")
        for name, value in (
            ("joint_valid_a", self.joint_valid_a),
            ("joint_valid_b", self.joint_valid_b),
        ):
            if (
                value.ndim != 3
                or value.shape[:2] != (pair_count, window)
                or value.shape[2] < 1
                or value.dtype is not torch.bool
            ):
                raise ValueError(
                    f"{name} must be boolean [pairs, window, support_channels]"
                )
        if self.joint_valid_a.shape != self.joint_valid_b.shape:
            raise ValueError("paired support-channel masks must have identical shape")
        if bool(
            (self.joint_valid_a & ~self.valid_a.unsqueeze(-1)).any()
            or bool((self.joint_valid_b & ~self.valid_b.unsqueeze(-1)).any())
        ):
            raise ValueError("invalid frames cannot carry valid joints")
        for name, value in (
            ("source_indices_a", self.source_indices_a),
            ("source_indices_b", self.source_indices_b),
        ):
            if value.shape != (pair_count, window) or value.dtype != torch.long:
                raise ValueError(f"{name} must be int64 [pairs, window]")
        for name, value in (
            ("native_lengths", self.native_lengths),
            ("starts_a", self.starts_a),
            ("starts_b", self.starts_b),
            ("segment_starts", self.segment_starts),
            ("segment_ends", self.segment_ends),
            ("source_video_indices", self.source_video_indices),
        ):
            if value.shape != (pair_count,) or value.dtype != torch.long:
                raise ValueError(f"{name} must be int64 [pairs]")
        tensors = (
            self.poses_b,
            self.valid_a,
            self.valid_b,
            self.joint_valid_a,
            self.joint_valid_b,
            self.source_indices_a,
            self.source_indices_b,
            self.native_lengths,
            self.starts_a,
            self.starts_b,
            self.segment_starts,
            self.segment_ends,
            self.source_video_indices,
        )
        if any(value.device != self.poses_a.device for value in tensors):
            raise ValueError("all paired-window tensors must be on the same device")
        if pair_count and int(self.native_lengths.min()) < 2:
            raise ValueError("paired-window native lengths must be at least two")
        if pair_count and (int(self.starts_a.min()) < 0 or int(self.starts_b.min()) < 0):
            raise ValueError("paired-window starts must be non-negative")
        if pair_count and bool(
            (
                (self.segment_starts < 0)
                | (self.segment_starts > self.starts_a)
                | (self.segment_ends < self.starts_b + window)
                | (self.segment_ends > self.native_lengths)
            ).any()
        ):
            raise ValueError("paired windows must lie inside one authorized segment")
        source_video_total = len(self.raw_grid_pair_counts)
        if pair_count and (
            int(self.source_video_indices.min()) < 0
            or int(self.source_video_indices.max()) >= source_video_total
        ):
            raise ValueError("source video indices lie outside pair tallies")
        for poses, valid in (
            (self.poses_a, self.valid_a),
            (self.poses_b, self.valid_b),
        ):
            if bool(valid.any()) and not bool(torch.isfinite(poses[valid]).all()):
                raise ValueError("valid paired-window pose rows must be finite")
        if not (
            len(self.raw_grid_pair_counts)
            == len(self.available_pair_counts)
            == len(self.eligible_pair_counts)
        ):
            raise ValueError("raw-grid, base-valid, and eligible pair tallies must align")
        if any(value < 0 for value in self.raw_grid_pair_counts):
            raise ValueError("raw-grid pair counts must be non-negative")
        if any(value < 0 for value in self.available_pair_counts):
            raise ValueError("available pair counts must be non-negative")
        if any(value < 0 for value in self.eligible_pair_counts):
            raise ValueError("eligible pair counts must be non-negative")
        if sum(self.eligible_pair_counts) != pair_count:
            raise ValueError("eligible pair tallies must sum to pair count")
        if any(
            not 0 <= eligible <= available <= raw_grid
            for eligible, available, raw_grid in zip(
                self.eligible_pair_counts,
                self.available_pair_counts,
                self.raw_grid_pair_counts,
                strict=True,
            )
        ):
            raise ValueError(
                "pair counts must satisfy eligible <= base-valid <= raw-grid"
            )
        observed_counts = tuple(
            int((self.source_video_indices == source_index).sum())
            for source_index in range(len(self.eligible_pair_counts))
        )
        if observed_counts != self.eligible_pair_counts:
            raise ValueError("source video indices differ from eligible pair tallies")

    @property
    def pair_count(self) -> int:
        return int(self.poses_a.shape[0])

    @property
    def window_length(self) -> int:
        return int(self.poses_a.shape[1])

    @property
    def available_pair_total(self) -> int:
        return sum(self.available_pair_counts)

    @property
    def raw_grid_pair_total(self) -> int:
        return sum(self.raw_grid_pair_counts)

    @property
    def eligible_video_total(self) -> int:
        return sum(value > 0 for value in self.eligible_pair_counts)

    def to(self, device: str | torch.device) -> NativeWindowPairBatch:
        return NativeWindowPairBatch(
            video_ids=self.video_ids,
            poses_a=self.poses_a.to(device=device),
            poses_b=self.poses_b.to(device=device),
            valid_a=self.valid_a.to(device=device),
            valid_b=self.valid_b.to(device=device),
            joint_valid_a=self.joint_valid_a.to(device=device),
            joint_valid_b=self.joint_valid_b.to(device=device),
            source_indices_a=self.source_indices_a.to(device=device),
            source_indices_b=self.source_indices_b.to(device=device),
            native_lengths=self.native_lengths.to(device=device),
            starts_a=self.starts_a.to(device=device),
            starts_b=self.starts_b.to(device=device),
            segment_starts=self.segment_starts.to(device=device),
            segment_ends=self.segment_ends.to(device=device),
            source_video_indices=self.source_video_indices.to(device=device),
            raw_grid_pair_counts=self.raw_grid_pair_counts,
            available_pair_counts=self.available_pair_counts,
            eligible_pair_counts=self.eligible_pair_counts,
        )

    def select(self, indices: Tensor) -> NativeWindowPairBatch:
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("window selection indices must be one-dimensional int64")
        if indices.numel() and (
            int(indices.min()) < 0 or int(indices.max()) >= self.pair_count
        ):
            raise ValueError("window selection index lies outside the pair batch")
        if indices.numel() != torch.unique(indices).numel():
            raise ValueError("window selection indices must be unique")
        cpu_indices = indices.detach().to(device="cpu")
        device_indices = indices.to(device=self.poses_a.device)
        selected_ids = tuple(self.video_ids[int(index)] for index in cpu_indices)
        selected_source = self.source_video_indices[device_indices]
        selected_counts = tuple(
            int((selected_source == source_index).sum())
            for source_index in range(len(self.available_pair_counts))
        )
        return NativeWindowPairBatch(
            video_ids=selected_ids,
            poses_a=self.poses_a[device_indices],
            poses_b=self.poses_b[device_indices],
            valid_a=self.valid_a[device_indices],
            valid_b=self.valid_b[device_indices],
            joint_valid_a=self.joint_valid_a[device_indices],
            joint_valid_b=self.joint_valid_b[device_indices],
            source_indices_a=self.source_indices_a[device_indices],
            source_indices_b=self.source_indices_b[device_indices],
            native_lengths=self.native_lengths[device_indices],
            starts_a=self.starts_a[device_indices],
            starts_b=self.starts_b[device_indices],
            segment_starts=self.segment_starts[device_indices],
            segment_ends=self.segment_ends[device_indices],
            source_video_indices=selected_source,
            raw_grid_pair_counts=self.raw_grid_pair_counts,
            available_pair_counts=self.available_pair_counts,
            eligible_pair_counts=selected_counts,
        )


def _validated_lengths(
    lengths: Tensor,
    *,
    batch: int,
    padded_time: int,
) -> Tensor:
    if lengths.shape != (batch,) or lengths.dtype != torch.long:
        raise ValueError("native lengths must be int64 [batch]")
    if lengths.numel() and (
        int(lengths.min()) < 1 or int(lengths.max()) > padded_time
    ):
        raise ValueError("native lengths must lie inside the padded pose timeline")
    return lengths


def enumerate_native_window_pairs(
    view_a: Tensor,
    view_b: Tensor,
    valid_mask: Tensor,
    joint_valid_mask: Tensor,
    lengths: Tensor,
    video_ids: Sequence[str],
    segment_ranges: Sequence[Sequence[tuple[int, int]]],
    authorized_base_valid_starts: Sequence[Sequence[int]],
    authorized_pair_starts: Sequence[Sequence[int]],
    *,
    window_length: int,
    hop_frames: int,
    minimum_valid_frames_per_window: int,
    minimum_window_stable_action_joints: int,
    minimum_window_joint_support_fraction: float,
) -> NativeWindowPairBatch:
    """Enumerate adjacent disjoint windows without timeline resampling.

    Starts follow ``0, hop, 2*hop, ...`` as frozen by representation authority.
    Each pair is ``[s, s+W)`` from
    view A and ``[s+W, s+2W)`` from view B.  Hop controls only the start grid;
    the two source-index sets must be disjoint.  Both windows must be wholly
    valid and their complete ``2W`` span must lie in one explicit authorized
    segment, so no pair can bridge a detector reset or arbitrary pose gap.
    ``authorized_base_valid_starts`` freezes the denominator after segment and
    frame-valid checks but before joint support; ``authorized_pair_starts`` is
    the final subset after the frozen support tuple.  Augmented coordinate
    values never affect selection.
    """

    if view_a.ndim != 4 or view_a.shape[2:] != (33, 3):
        raise ValueError("views must have shape [batch, time, 33, 3]")
    if view_b.shape != view_a.shape:
        raise ValueError("independent views must have identical shapes")
    if not view_a.is_floating_point() or not view_b.is_floating_point():
        raise TypeError("independent views must use a floating-point dtype")
    if view_b.dtype != view_a.dtype or view_b.device != view_a.device:
        raise ValueError("independent views must share dtype and device")
    batch, padded_time = view_a.shape[:2]
    if valid_mask.shape != (batch, padded_time) or valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be boolean [batch, time]")
    if joint_valid_mask.shape != (batch, padded_time, 17) or (
        joint_valid_mask.dtype is not torch.bool
    ):
        raise ValueError("joint_valid_mask must be boolean [batch, time, 17]")
    if (
        valid_mask.device != view_a.device
        or joint_valid_mask.device != view_a.device
        or lengths.device != view_a.device
    ):
        raise ValueError("view, validity, and length tensors must share a device")
    native_lengths = _validated_lengths(
        lengths,
        batch=batch,
        padded_time=padded_time,
    )
    normalized_ids = tuple(str(identifier).strip() for identifier in video_ids)
    if len(normalized_ids) != batch or len(set(normalized_ids)) != batch:
        raise ValueError("video IDs must be unique and match the pose batch")
    if len(segment_ranges) != batch:
        raise ValueError("segment ranges must match the pose batch")
    if len(authorized_pair_starts) != batch:
        raise ValueError("authorized pair starts must match the pose batch")
    if len(authorized_base_valid_starts) != batch:
        raise ValueError("authorized base-valid starts must match the pose batch")
    if any(not identifier for identifier in normalized_ids):
        raise ValueError("video IDs must be non-empty")
    if bool((joint_valid_mask & ~valid_mask.unsqueeze(-1)).any()):
        raise ValueError("invalid frames cannot carry valid joints")
    time = torch.arange(padded_time, device=view_a.device).unsqueeze(0)
    padded_tail = time >= native_lengths.unsqueeze(1)
    if bool((valid_mask & padded_tail).any()):
        raise ValueError("valid_mask marks a padded tail frame as valid")
    if bool(valid_mask.any()):
        for name, value in (("view_a", view_a), ("view_b", view_b)):
            if not bool(torch.isfinite(value[valid_mask]).all()):
                raise ValueError(f"{name} has non-finite coordinates at valid frames")
    for name, value in (
        ("window_length", window_length),
        ("hop_frames", hop_frames),
        ("minimum_valid_frames_per_window", minimum_valid_frames_per_window),
        (
            "minimum_window_stable_action_joints",
            minimum_window_stable_action_joints,
        ),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if hop_frames >= window_length:
        raise ValueError("hop_frames must be smaller than window_length")
    if minimum_valid_frames_per_window != window_length:
        raise ValueError("every frame in each cycle-back window must be valid")
    if minimum_window_stable_action_joints > 8:
        raise ValueError("joint-support threshold exceeds the eight action joints")
    if not isinstance(minimum_window_joint_support_fraction, float) or not (
        0.0 < minimum_window_joint_support_fraction <= 1.0
    ):
        raise ValueError("joint-support fraction must lie in (0,1]")

    poses_a: list[Tensor] = []
    poses_b: list[Tensor] = []
    masks_a: list[Tensor] = []
    masks_b: list[Tensor] = []
    joint_masks_a: list[Tensor] = []
    joint_masks_b: list[Tensor] = []
    indices_a: list[Tensor] = []
    indices_b: list[Tensor] = []
    pair_lengths: list[int] = []
    starts_a: list[int] = []
    starts_b: list[int] = []
    pair_segment_starts: list[int] = []
    pair_segment_ends: list[int] = []
    source_video_indices: list[int] = []
    pair_video_ids: list[str] = []
    raw_grid_counts: list[int] = []
    base_valid_counts: list[int] = []
    eligible_counts: list[int] = []

    for sample_index, identifier in enumerate(normalized_ids):
        length = int(native_lengths[sample_index])
        normalized_segments: list[tuple[int, int]] = []
        previous_end = 0
        for raw_segment in segment_ranges[sample_index]:
            if (
                not isinstance(raw_segment, tuple)
                or len(raw_segment) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in raw_segment
                )
            ):
                raise ValueError("segment ranges must contain integer (start, end) tuples")
            segment_start, segment_end = raw_segment
            if not (0 <= segment_start < segment_end <= length):
                raise ValueError("segment range lies outside the native timeline")
            if normalized_segments and segment_start < previous_end:
                raise ValueError("segment ranges must be sorted and non-overlapping")
            normalized_segments.append((segment_start, segment_end))
            previous_end = segment_end
        maximum_start = length - 2 * window_length
        available_grid = (
            () if maximum_start < 0 else range(0, maximum_start + 1, hop_frames)
        )
        raw_grid_count = len(available_grid)
        eligible = 0
        base_starts = tuple(authorized_base_valid_starts[sample_index])
        starts = tuple(authorized_pair_starts[sample_index])
        if tuple(sorted(set(base_starts))) != base_starts:
            raise ValueError("authorized base-valid starts must be sorted and unique")
        if tuple(sorted(set(starts))) != starts:
            raise ValueError("authorized pair starts must be sorted and unique")
        if not set(starts).issubset(base_starts):
            raise ValueError("authorized final starts are not a subset of base-valid starts")
        base_geometry: dict[int, tuple[int, int]] = {}
        for start_a in base_starts:
            if (
                isinstance(start_a, bool)
                or not isinstance(start_a, int)
                or start_a < 0
                or start_a > maximum_start
                or start_a % hop_frames
            ):
                raise ValueError("authorized pair start violates candidate geometry")
            start_b = start_a + window_length
            end_a = start_a + window_length
            end_b = start_b + window_length
            containing_segment = next(
                (
                    (segment_start, segment_end)
                    for segment_start, segment_end in normalized_segments
                    if segment_start <= start_a and end_b <= segment_end
                ),
                None,
            )
            if containing_segment is None:
                raise RuntimeError(
                    "authorized base-valid pair crosses a representation segment/reset"
                )
            mask_a = valid_mask[sample_index, start_a:end_a]
            mask_b = valid_mask[sample_index, start_b:end_b]
            if (
                int(mask_a.sum()) < minimum_valid_frames_per_window
                or int(mask_b.sum()) < minimum_valid_frames_per_window
            ):
                raise RuntimeError(
                    "authorized base-valid pair fails frozen frame-valid replay"
                )
            base_geometry[start_a] = containing_segment
        for start_a in starts:
            start_b = start_a + window_length
            end_a = start_a + window_length
            end_b = start_b + window_length
            containing_segment = base_geometry[start_a]
            mask_a = valid_mask[sample_index, start_a:end_a]
            mask_b = valid_mask[sample_index, start_b:end_b]
            joint_mask_a = joint_valid_mask[sample_index, start_a:end_a]
            joint_mask_b = joint_valid_mask[sample_index, start_b:end_b]
            pair_joint_mask = torch.cat((joint_mask_a, joint_mask_b))
            action_joint_indices = torch.tensor(
                [7, 8, 9, 10, 13, 14, 15, 16],
                dtype=torch.long,
                device=pair_joint_mask.device,
            )
            action_support = pair_joint_mask[:, action_joint_indices].float().mean(dim=0)
            observed_stable_joints = int(
                (action_support >= minimum_window_joint_support_fraction).sum()
            )
            if (
                observed_stable_joints < minimum_window_stable_action_joints
            ):
                raise RuntimeError(
                    "authorized pair fails frozen validity/joint-support replay"
                )
            eligible += 1
            absolute_a = torch.arange(
                start_a,
                end_a,
                dtype=torch.long,
                device=view_a.device,
            )
            absolute_b = torch.arange(
                start_b,
                end_b,
                dtype=torch.long,
                device=view_a.device,
            )
            poses_a.append(
                view_a[sample_index, start_a:end_a].masked_fill(
                    ~mask_a.unsqueeze(-1).unsqueeze(-1),
                    0.0,
                )
            )
            poses_b.append(
                view_b[sample_index, start_b:end_b].masked_fill(
                    ~mask_b.unsqueeze(-1).unsqueeze(-1),
                    0.0,
                )
            )
            masks_a.append(mask_a)
            masks_b.append(mask_b)
            joint_masks_a.append(joint_mask_a)
            joint_masks_b.append(joint_mask_b)
            indices_a.append(absolute_a)
            indices_b.append(absolute_b)
            pair_lengths.append(length)
            starts_a.append(start_a)
            starts_b.append(start_b)
            pair_segment_starts.append(containing_segment[0])
            pair_segment_ends.append(containing_segment[1])
            source_video_indices.append(sample_index)
            pair_video_ids.append(identifier)
        raw_grid_counts.append(raw_grid_count)
        base_valid_counts.append(len(base_starts))
        eligible_counts.append(eligible)

    if poses_a:
        stacked_a = torch.stack(poses_a)
        stacked_b = torch.stack(poses_b)
        stacked_mask_a = torch.stack(masks_a)
        stacked_mask_b = torch.stack(masks_b)
        stacked_joint_mask_a = torch.stack(joint_masks_a)
        stacked_joint_mask_b = torch.stack(joint_masks_b)
        stacked_indices_a = torch.stack(indices_a)
        stacked_indices_b = torch.stack(indices_b)
    else:
        shape = (0, window_length, 33, 3)
        stacked_a = view_a.new_zeros(shape)
        stacked_b = view_b.new_zeros(shape)
        stacked_mask_a = torch.zeros(
            (0, window_length),
            dtype=torch.bool,
            device=view_a.device,
        )
        stacked_mask_b = stacked_mask_a.clone()
        stacked_joint_mask_a = torch.zeros(
            (0, window_length, 17),
            dtype=torch.bool,
            device=view_a.device,
        )
        stacked_joint_mask_b = stacked_joint_mask_a.clone()
        stacked_indices_a = torch.zeros(
            (0, window_length),
            dtype=torch.long,
            device=view_a.device,
        )
        stacked_indices_b = stacked_indices_a.clone()

    return NativeWindowPairBatch(
        video_ids=tuple(pair_video_ids),
        poses_a=stacked_a,
        poses_b=stacked_b,
        valid_a=stacked_mask_a,
        valid_b=stacked_mask_b,
        joint_valid_a=stacked_joint_mask_a,
        joint_valid_b=stacked_joint_mask_b,
        source_indices_a=stacked_indices_a,
        source_indices_b=stacked_indices_b,
        native_lengths=torch.tensor(
            pair_lengths,
            dtype=torch.long,
            device=view_a.device,
        ),
        starts_a=torch.tensor(starts_a, dtype=torch.long, device=view_a.device),
        starts_b=torch.tensor(starts_b, dtype=torch.long, device=view_a.device),
        segment_starts=torch.tensor(
            pair_segment_starts,
            dtype=torch.long,
            device=view_a.device,
        ),
        segment_ends=torch.tensor(
            pair_segment_ends,
            dtype=torch.long,
            device=view_a.device,
        ),
        source_video_indices=torch.tensor(
            source_video_indices,
            dtype=torch.long,
            device=view_a.device,
        ),
        raw_grid_pair_counts=tuple(raw_grid_counts),
        available_pair_counts=tuple(base_valid_counts),
        eligible_pair_counts=tuple(eligible_counts),
    )


def source_index_invariant_violations(
    pairs: NativeWindowPairBatch,
    *,
    expected_hop_frames: int,
) -> dict[str, int]:
    """Count every native-source indexing invariant violation."""

    if expected_hop_frames < 1 or expected_hop_frames >= pairs.window_length:
        raise ValueError("expected hop must lie in [1, window_length)")
    if pairs.pair_count == 0:
        return {
            "start_offset": 0,
            "start_grid": 0,
            "a_start_identity": 0,
            "b_start_identity": 0,
            "a_contiguity": 0,
            "b_contiguity": 0,
            "source_index_intersection_total": 0,
            "segment_bounds": 0,
            "native_bounds": 0,
            "total": 0,
        }
    ones_a = pairs.source_indices_a[:, 1:] - pairs.source_indices_a[:, :-1]
    ones_b = pairs.source_indices_b[:, 1:] - pairs.source_indices_b[:, :-1]
    bounds_a = (
        (pairs.source_indices_a < 0)
        | (pairs.source_indices_a >= pairs.native_lengths.unsqueeze(1))
    )
    bounds_b = (
        (pairs.source_indices_b < 0)
        | (pairs.source_indices_b >= pairs.native_lengths.unsqueeze(1))
    )
    counts = {
        "start_offset": int(
            (pairs.starts_b - pairs.starts_a != pairs.window_length).sum()
        ),
        "start_grid": int((pairs.starts_a.remainder(expected_hop_frames) != 0).sum()),
        "a_start_identity": int(
            (pairs.source_indices_a[:, 0] != pairs.starts_a).sum()
        ),
        "b_start_identity": int(
            (pairs.source_indices_b[:, 0] != pairs.starts_b).sum()
        ),
        "a_contiguity": int((ones_a != 1).sum()),
        "b_contiguity": int((ones_b != 1).sum()),
        "source_index_intersection_total": sum(
            len(set(row_a.tolist()).intersection(row_b.tolist()))
            for row_a, row_b in zip(
                pairs.source_indices_a.detach().cpu(),
                pairs.source_indices_b.detach().cpu(),
                strict=True,
            )
        ),
        "segment_bounds": int(
            (
                (pairs.segment_starts > pairs.starts_a)
                | (pairs.segment_ends < pairs.starts_b + pairs.window_length)
            ).sum()
        ),
        "native_bounds": int(bounds_a.sum() + bounds_b.sum()),
    }
    counts["total"] = sum(counts.values())
    return counts
