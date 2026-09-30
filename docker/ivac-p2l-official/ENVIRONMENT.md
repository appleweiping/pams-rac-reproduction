# IVAC-P2L official modern-compatibility environment

This image description records the environment used by the retrospective
official-checkpoint UCFRep dev run. It is a modern compatibility environment,
not an upstream environment lock and not evidence of original-protocol parity.
The IVAC-P2L repository does not publish a complete, immutable environment
lock from which the paper runtime can be reconstructed uniquely.

The audited run used compatibility image ID
`sha256:ea96abe0e7d17cf4720343e5867256aacd5209aca738087ec92f62b9fe31ff51`,
Python 3.11, PyTorch 2.5.1+cu124, CUDA 12.4, MMCV 1.4.0, timm 0.4.12,
einops 0.3.2, and kornia 0.5.11. The build starts from the immutable base:

```text
pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel@sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014
```

[`requirements.in`](requirements.in) pins direct Python dependencies only.
It is not a hash-locked, bit-reproducible dependency lock. Rebuilding this
directory produces a new image identity; results from that image require a
fresh run through the current IVAC-P2L runner and scorer. In particular, the
legacy audited artifact was not emitted by the current public runner and must
not be represented as a current-runner result.

The image contains no IVAC-P2L source tree, checkpoint, Swin backbone, UCFRep
video, label, or prediction. It does not download any of those artifacts.
Acquire them separately, verify them against the public runner's frozen
byte-count and SHA-256 commitments, and mount them read-only. Mount the pinned
source tree at `/opt/ivac-source`, external model assets under
`/opt/ivac-assets`, and this repository at `/workspace`.

The pinned IVAC-P2L source revision is
`0b1149e6958268ca5131d60ff66498c6e07f3b79`. Its archived source includes an
MIT license. That license does not establish redistribution permission for
the published IVAC-P2L checkpoint or the external Swin backbone; their
licenses remain unspecified. Neither model asset is redistributed here.

The upstream evaluation path first serializes sampled video frames to NPZ and
then reads NPZ tensors, while other descriptions can be read as direct raw
video inference. The precise raw-video/NPZ boundary is therefore
protocol-ambiguous. The current runner freezes one audited, label-free
interpretation and records it as an inferred modern-compatibility protocol;
it is not claimed to be uniquely disclosed by the paper.

Prediction and scoring must run as two separate processes. The prediction
process receives only the committed label-free sidecar and videos. The scorer
may open dev targets only after it has validated the prediction, receipt,
source and asset bindings, the clean project Git revision and exact
runner/scorer bytes, every row's video locator and digest, and the unchanged
sidecar plus commitment. Test targets remain sealed.
