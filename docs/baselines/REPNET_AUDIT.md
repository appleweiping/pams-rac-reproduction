# RepNet source and protocol audit

Audit date: 2026-07-28. This record distinguishes a runnable current
official checkpoint from the RepNet number copied into the PAMS comparison
table. They are not assumed to be the same experiment.

## Official current track

- Source: `google-research/google-research` commit
  `ec7c3d346277b737bc2decffcd1b533d4b7ec105`, directory `repnet`.
- License: Apache-2.0 at the repository root.
- Published material: README and Colab notebook; the README does not provide
  a UCFRep evaluator.
- Current checkpoint prefix: `ckpt-70`.

The three current GCS objects must be acquired into an isolated baseline
environment and verified before use:

| Object | Bytes | Published GCS MD5/ETag | Generation |
|---|---:|---|---:|
| `checkpoint` | 33 | `95c36d059d6d0d6d9be5916c0f26d73e` | `1747092885745382` |
| `ckpt-70.index` | 10,209 | `5292903f0dfd1c916db65d04ea634040` | `1747092885745276` |
| `ckpt-70.data-00000-of-00001` | 309,204,174 | `7d97aa53df6817737a84a4b1c4a10204` | `1747092903683100` |

The object URLs use the prefix
`https://storage.googleapis.com/semantic_repetitions/repnet_ckpt/`.
The source does not publish SHA-256 values. A formal run must first verify the
published MD5 values and object generation, then record locally computed
SHA-256 values. This audit did not download the checkpoint.

The source notebook freezes the independent adapter inputs as:

- OpenCV decode, BGR to RGB, and initial `224x224` resize;
- float32 `(pixel-127.5)/127.5`, followed by bilinear `112x112` resize;
- 64-frame windows, with strides `[1,2,3,4]`;
- inference batch 20;
- global periodicity threshold `0.2` and frame threshold `0.5`;
- `constant_speed=False`, `median_filter=True`, `fully_periodic=False`;
- model-confidence stride selection, never ground-truth-count selection;
- raw count `sum(per_frame_counts)`, with the shared evaluator applying its
  declared rounding policy.

The only honest name for the planned source sanity row is
`RepNet-official-current-ckpt70 / independent UCFRep evaluation`.

## Legacy PyTorch port

`materight/RepNet-pytorch` commit
`c455c53a7e48724f73a2e65e336aee065c965418` has no repository license.
Its Hugging Face model snapshot
`3d66d66c9313da842f91a1e010ed7e3a0f4c311c` contains
`pytorch_weights.pth` (102,830,765 bytes, SHA-256
`649ab92e22ff6d5a6ed0e8938cfdef5c57647930eebb7a58ab34fd63748899c6`)
without a declared model license.

That port represents the older `ckpt-88`, not current `ckpt-70`. It has 32
period classes instead of 64, defaults to strides `[1,2,3,4,8]`, drops a
short final window instead of padding it, omits current threshold/median
filter behavior, and does not pin the external `timm` source commit. It may
only be run privately as `RepNet-legacy-port diagnostic`; its code or weights
must not be copied into this Apache-2.0 repository.

## PAMS-table identifiability

The PAMS paper reports RepNet on UCFRep as `0.987 / 0.018` but does not state
the checkpoint generation, implementation, strides, tail policy, thresholds,
median filtering, adapter, rounding, dependency versions, or checkpoint
digest. Current `ckpt-70` and legacy `ckpt-88` differ in architecture and
post-processing, so that table cell is `protocol-unverifiable`.

The planned fair result is a new independent measurement on the fixed 105
UCFRep videos, not a recovery claim for that cell. Before opening test labels:

1. verify and hash `ckpt-70`;
2. run 3--5 label-free training-video sanity cases;
3. freeze the source-notebook parameters above;
4. run the 105-video adapter once;
5. publish every raw/rounded prediction, confidence, chosen stride, decode
   status, checkpoint hash, configuration hash, split hash, and Git SHA.
