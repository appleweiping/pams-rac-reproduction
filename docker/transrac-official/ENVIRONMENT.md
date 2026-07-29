# TransRAC official modern-compatibility environment

This image description reconstructs the environment used for the
development-only official-checkpoint sanity run. It is not the environment
claimed by the TransRAC paper and does not establish original-protocol parity.
The audited run used compatibility image ID
`sha256:ea96abe0e7d17cf4720343e5867256aacd5209aca738087ec92f62b9fe31ff51`,
Python 3.11.10, PyTorch 2.5.1+cu124, CUDA 12.4,
MMCV 1.4.0, timm 0.4.12, einops 0.3.2, and kornia 0.5.11 on an RTX A6000.

The upstream requirements cannot be installed literally: they simultaneously
list `mmcv==1.4.8` and `mmcv-full==1.3.16`, pin PyTorch 1.7.0+cu110, and
elsewhere mention CUDA 11.4. This directory therefore records the actual
modern compatibility stack and labels every resulting metric accordingly.

[`requirements.in`](requirements.in) contains only direct version pins. It
does not pin transitive wheel identities or package hashes and must not be
described as a bit-reproducible lock. The path-free
[`server-pip-inventory.txt`](server-pip-inventory.txt) records every Python
distribution observed in the successful image; it is an audit inventory, not
an installable lock. Exact reruns use the frozen image ID above, while a
rebuild receives a new identity and requires a new sanity run.

The image contains no TransRAC source, checkpoint, Swin backbone, UCFRep
video, label, or prediction. Mount the independently acquired, hash-verified
source tree read-only at `/opt/transrac-source`, assets read-only under
`/opt/transrac-assets`, and this repository at `/workspace`. The strict
runner takes only the label-free sidecar, its commitment, and videos. A
separate scorer receives the dev targets only after validating prediction
and receipt bytes. Both commands also require the same clean repository root;
its full Git SHA and exact runner/scorer bytes are bound into their receipts.

License status is unresolved for redistribution. The pinned source archive
contains an Apache-2.0 `LICENSE` file, while its README advertises and links
an Anti-996 license. The archive also embeds or references MMCV/MMACTION
files, wheels, a Swin backbone, a TransRAC checkpoint, and an NVIDIA base
image whose licenses must be audited independently. For that reason none of
those third-party bytes are copied into this repository or image context.
