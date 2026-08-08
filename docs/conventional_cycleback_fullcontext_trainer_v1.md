# Full-context cycle-back continuation v1

This is an independently inferred conventional cycle-back proxy, not the
authors' undisclosed baseline implementation and not a paper-table claim.

## Current execution status

The trainer is deliberately nonlaunchable. All production activation anchors
in `fullcontext_authority.py` are empty. The existing 256-step mechanism probe
publishes only JSON containing a semantic model-state digest; it does not
publish the learned encoder `L`, AdamW state, RNG streams, or a canonical
mechanism outcome. A digest cannot be replayed into the exact learned state.
No existing mechanism run may be retroactively upgraded or reconstructed from
its seed.

Before epoch-one continuation, a future prerequisite producer must atomically
write `pams_conventional_cycleback_mechanism_seed_checkpoint_v1`, then bind its
exact bytes in the mechanism result, terminal receipt, sealed outcome, and an
independent epoch11 launch authorization. The seed checkpoint lineage is
acyclic: the checkpoint embeds all upstream predecessor identities but not its
own SHA; the later result and receipt bind the checkpoint SHA and byte count.

## Frozen science contract

- candidates: `W16_H4`, `W16_H2`, and `W24_H4`;
- adjacent disjoint native windows `A=[s,s+W)`, `B=[s+W,s+2W)`;
- window-local target positions retain absolute native-frame spacing;
- every attention context is exactly one authorized, all-valid stable range;
- each unique `(video_id, stable range, augmentation view)` is encoded once;
- contexts are encoded A then B in exact-length buckets and stable key order;
- every pair gathers from that shared tensor by absolute source indices;
- diagnostic zero, independent temporal shuffle, joint-mask, and PE controls
  are typed non-optimizer contexts;
- AdamW uses `lr=1e-4`, `weight_decay=1e-4`, and the exact step-256 optimizer
  state without reset;
- scheduler is explicitly `none` because the paper does not disclose one;
- epoch order is a deterministic label-free length-bucket plan with no dropped
  or repeated eligible training unit;
- checkpoints include model, optimizer, epoch/step, full sampler plan/cursor,
  Python/NumPy/Torch CPU/all CUDA RNG states, next view seeds, config/source/
  image, representation integration, and mechanism lineage.

The first stage ends at exactly 11 completed train337 epochs. Its fixed,
label-free mechanism controls may establish scientific eligibility for a
separate 150-epoch launch authorization, but never self-authorize it. The
150-epoch stage must restore the exact completed epoch11 checkpoint and begin
at epoch 12. Neither stage authorizes development, sealed test, or readout.

## Typed representation boundary

The trainer contract does not equate support channels with COCO17 joints. A
sealed `FullContextRepresentationContract` binds the 33x3 encoder input,
coordinate system, support-channel-to-pose-joint mapping, augmentation adapter,
diagnostic-null adapter, stable-range policy, and exact-start authority.

The existing executable adapter is the v4e unified-2D COCO17-to-padded33 path.
A MediaPipe33/v4a path may use the same sampler, optimizer, loss, and checkpoint
core only after an independent adapter freezes its 33-joint support mapping,
augmentation/null semantics, and exact contiguous-valid ranges. The v4a
coverage observation does not relax or authorize the v4e representation gate.

## Remaining prerequisite work

1. Add a no-retroactive-claim mechanism checkpoint producer and canonical
   sealed mechanism outcome/receipt.
2. Freeze an independent synthetic threshold receipt and epoch11 launch
   authorization after both producer and consumer source trees are exact.
3. Add the secure epoch11/epoch150 launcher (exact argv/env/mount/GPU contract,
   immutable source export, O_EXCL reservation, exact-ID cleanup, sealed
   terminal receipts). Production activation constants must remain empty until
   server tests and independent audit pass.
4. Add the representation-specific v4a adapter before any MediaPipe33 run.
5. Add a later readout gate; completing epoch150 does not authorize dev84 or
   test105 access.
