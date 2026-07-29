# Frozen official ESCounts runtime

This sidecar image recreates the environment that completed the 84-video
UCFRep dev run. It does not download or embed the EveryShotCounts repository,
videos, encoder checkpoint, or decoder checkpoint. Mount those read-only at
runtime and let the audited runner verify every byte before importing Torch.

## Frozen identities

- Base image:
  `pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime@sha256:cf9197f9321ac3f49276633b4e78c79aa55f22578de3b650b3158ce6e3481f61`
- Successful built image ID:
  `sha256:760d74488cf96dd120e8b796a63f8081b78bef029000727b4176f7db2fb5f5f0`
- Python 3.8.13, PyTorch 1.10.0, torchvision 0.11.0, CUDA 11.3,
  NumPy 1.21.2, PyAV 10.0.0, MKL 2021.4.
- PyTorchVideo:
  `https://github.com/facebookresearch/pytorchvideo` at
  `fae0d89a194a2c1ca99e59eab6eedd40bde38726`, clean tree
  `bd12d614191c96f3625f111284dec2b0a56a178a`.
- EveryShotCounts:
  `https://github.com/sinhasaptarshi/EveryShotCounts` at
  `f18fcf1933abb3d4f199fd12ee3cafc53581576a`; clean tree
  `3bedc8d117c4c86954fb58d891326e7c9c6a1a3d`.

Clone the source outside the image:

```bash
git clone https://github.com/sinhasaptarshi/EveryShotCounts
git -C EveryShotCounts checkout --detach f18fcf1933abb3d4f199fd12ee3cafc53581576a
test "$(git -C EveryShotCounts rev-parse HEAD)" = \
  f18fcf1933abb3d4f199fd12ee3cafc53581576a
test -z "$(git -C EveryShotCounts status --porcelain=v1 --untracked-files=all)"
```

The runner additionally verifies the byte count and SHA-256 of `demo.py`,
`video_mae_cross_full_attention.py`, `configs/pretrain_config.yaml`,
`slowfast/utils/parser.py`, and `LICENSE`.

## External assets (never bundled)

| Role | Published source | Bytes | SHA-256 |
|---|---|---:|---|
| VideoMAE encoder | `https://dl.fbaipublicfiles.com/pyslowfast/masked_models/VIT_B_16x4_MAE_PT.pyth` | 1,207,498,009 | `b6d1d0b539dbdc992c3a3544a9bc5bbb0591179b615dc728f951285875d824e8` |
| RepCount decoder | `https://drive.google.com/uc?id=1cwUtgUM0XotOx5fM4v4ZU29hlKUxze48` | 239,106,669 | `297fc53000417ac6ffa4e08d6876e033df89800c97632fa09d2f3945ba225559` |

For the public Drive object, a credential-free direct endpoint was also
observed:

```text
https://drive.usercontent.google.com/download?id=1cwUtgUM0XotOx5fM4v4ZU29hlKUxze48&export=download&confirm=t
```

The upstream repository is MIT licensed. That source license does not by
itself establish redistribution rights for checkpoint bytes or their training
datasets. Users must review the checkpoint hosts and dataset terms; this
project records hashes and URLs only.

The image also contains PyTorch, torchvision, PyTorchVideo, Detectron2,
OpenCV, FFmpeg, PyAV, NumPy and their transitive dependencies under their own
licenses. Apache-2.0 covers this project's original adapter only and does not
relicense those packages. Before redistributing an image, generate an SBOM
from the built image and retain every dependency's license/notice; this
repository publishes the recipe but does not publish the historical image.

## Build and execution boundary

```bash
docker build -t escounts-official:f18fcf1 docker/escounts-official
```

Run prediction and scoring as separate processes. The prediction process may
receive only the count/action-free sidecar plus its commitment. The scorer
opens the dev target manifest only after the merged 84-row prediction file
passes complete source, asset, configuration, membership, row, and
forbidden-key validation.

The finite resource policy is one 8 GiB primary allocator tier and one 12 GiB
retry containing exactly the primary CUDA-OOM rows. Decode, missing-file,
hash, or inference failures are not retryable and cannot be hidden by merge.

The environment is intentionally split across processes. The repository's
Python 3.10+ audit runner must **not** be imported in this Python 3.8 image.
Instead, mount `src/pams/baselines/escounts_official_worker.py` read-only and
start it as the long-lived JSONL worker command supplied to
`escounts_official_runner predict`. The host validates every request/response
ID and SHA-256, while the worker loads the official models exactly once.

The released encoder checkpoint leaves exactly four tensors unmatched:
`example_spatial_pos_embed` (`[196,512]`, float32), `shot_token`
(`[1568,512]`, float32), `map.weight` (`[1,512]`, float32), and `map.bias`
(`[1]`, float32). The frozen official `forward` returns the encoder latent
tokens at lines 487--488 when `just_encode=True`, before `decoder_embed` and
the decoder-only path that can read these tensors. The compatibility shim is
therefore an exact allowlist, not a relaxed checkpoint load: any fifth tensor
or any name, shape, or dtype drift fails initialization. The worker preserves
the official random initialization of these unused tensors and emits the
canonical name/shape/dtype/unused-path list plus its stable structural
SHA-256. The host independently compares both values to its frozen constants
and records them in every prediction artifact. The digest intentionally
covers structure rather than random tensor bytes, so independently started
primary and retry workers retain official initialization semantics while
remaining lineage-compatible.

The successful historical image ID is evidence for that run, not a promise
that a later Docker build is byte-identical. Conda, pip, apt repositories, and
their transitive package builds can change even when the top-level versions
above remain fixed. A rebuilt image must receive a new inspected image ID and
must not reuse the historical ID.

The legacy 84-video ledger used Python's ties-to-even `round()` operation.
None of its 84 raw predictions landed on a half-integer, so recomputing with
the reproduction project's half-up rule changes no rounded prediction or
reported rounded metric. New audited runs use the project's frozen half-up
rule in the Python 3.10+ host.

The runner exposes the complete finite pipeline as subcommands:

```text
python -m pams.baselines.escounts_official_runner predict \
  --repository-root REPOSITORY ...
python -m pams.baselines.escounts_official_runner create-retry-request \
  --repository-root REPOSITORY ...
python -m pams.baselines.escounts_official_runner predict \
  --repository-root REPOSITORY \
  --resource-tier allocator_12g_retry \
  --primary-predictions PRIMARY.json --retry-request RETRY_REQUEST.json ...
python -m pams.baselines.escounts_official_runner merge \
  --repository-root REPOSITORY \
  --primary-predictions PRIMARY.json --retry-request RETRY_REQUEST.json \
  --retry-predictions RETRY.json --output MERGED.json --receipt MERGE_RECEIPT.json
```

`--worker-command-json` is a JSON string array. For Docker it should end in
`python /read-only/repository/src/pams/baselines/escounts_official_worker.py`;
the host appends the frozen source, PyTorchVideo checkout, video-root,
checkpoint, device, allocator and image-ID arguments. Bind every referenced
absolute path at the identical absolute path inside the container with `:ro`,
keep stdin attached (`-i`), and set
`ESCOUNTS_IMAGE_ID=sha256:<docker-image-inspect-id>` inside the container.
The host's `--expected-container-image-id` must contain the same full ID.

Scoring requires the actual five-file lineage, not self-reported hashes:

```text
python -m pams.baselines.escounts_official_score \
  --repository-root REPOSITORY \
  --primary-predictions PRIMARY.json --retry-request RETRY_REQUEST.json \
  --retry-predictions RETRY.json --predictions MERGED.json \
  --merge-receipt MERGE_RECEIPT.json \
  --sidecar DEV_INPUTS.json --commitment DEV_INPUTS.commitment.json \
  --dev-targets DEV_TARGETS.json --output-dir SCORE_DIR
```

Every prediction, retry request, merged artifact, merge receipt, evaluation,
and evaluation receipt binds the same clean full Git SHA plus the tracked
runner and worker byte hashes. A dirty checkout, an untracked adapter, or a
cross-commit lineage is rejected before scoring opens the target manifest.
