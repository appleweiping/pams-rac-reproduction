# PoseRAC-ICONIP24 source and implementation audit

## Identity boundary

`PoseRAC-ICONIP24` denotes *PoseRAC: Enhancing Repetitive Action Counting
with Salient Poses*, presented at ICONIP 2024 and published in LNCS 15290.
Its primary source is
[DOI 10.1007/978-981-96-6588-4_18](https://doi.org/10.1007/978-981-96-6588-4_18).

It is not the earlier `PoseRAC-v1` method associated with arXiv:2303.08450 and
the `MiracleDance/PoseRAC` repository. Assets or results from that repository
must therefore not be attributed to the ICONIP'24 method.

## Disclosed ICONIP'24 architecture

The paper describes a per-frame, pose-wise model:

- 3D OpenPose produces `K` keypoints with `D` coordinates per keypoint;
- an MLP embeds each keypoint;
- a six-layer Transformer encoder performs spatial self-attention over the
  keypoints in one frame;
- one action query per class cross-attends to the pose features through a
  two-layer Transformer decoder;
- a one-channel linear projection and sigmoid produce one score per action;
- UniFormerV2-L, pretrained on Kinetics-700, selects the video action class;
- a training-free two-threshold Pose-Action Trigger counts ordered appearances
  of two salient poses.

The paper reports BCE plus triplet-margin training. It generates 1,000
ControlNet/Stable-Diffusion images for each manually defined salient pose and,
for the few-shot setting, adds 200 real salient-pose frames per action.

## Independently inferred scaffold

`pams.baselines.pose_cleanroom.PoseRACICONIP24` implements only the spatial
encoder/query-decoder shape for unit and gradient testing. The following
choices are independent inferences rather than author disclosures:

- MediaPipe `33x3` poses replace 3D OpenPose;
- hidden width, attention heads, feed-forward width, and dropout;
- the keypoint embedding MLP shape;
- random learned action queries for the closed-set smoke path;
- an oracle-free dynamic-range channel selector when only a `PoseSequence`
  (and no RGB video) is available.

The module returns logits and the smoke adapter applies the sigmoid, which is
mathematically equivalent to the paper's final sigmoid but convenient for a
future BCE-with-logits training implementation.

## Parity blockers and claim boundary

No public ICONIP'24 implementation or trained checkpoint has been located.
The UCFRep salient-pose definitions, generated training images, exact hidden
dimensions, trigger thresholds, and per-action training annotations are not
fully disclosed. The required RGB UniFormerV2-L action recognizer also cannot
be reconstructed from `PoseSequence` alone.

Consequently:

- the registry status remains `blocked_unimplemented`;
- the smoke adapter is not registered as a runnable benchmark baseline;
- no score from an untrained scaffold is eligible for a source-faithful,
  clean-room, UCFRep-526, or paper-table result;
- any later completion must preserve these inferred labels and must not reuse
  PoseRAC-v1 results under the ICONIP'24 name.
