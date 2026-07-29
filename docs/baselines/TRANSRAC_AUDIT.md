# TransRAC official-checkpoint audit

## Admissible claim

The completed run is classified only as **“TransRAC official checkpoint /
modern compatibility / local 84-dev sanity.”** It processed all 84 frozen
UCFRep-526 development videos without decode or non-finite failures and was
scored by a separate process. It is not a result on the sealed 105-video
test split, not a reconstruction of the paper's native evaluator, and not
eligible for the original PAMS comparison cell.

The exact compact evidence is
[`results/dev-negative/transrac_official_modern_compat_dev84.json`](../../results/dev-negative/transrac_official_modern_compat_dev84.json).
Raw predictions, UCFRep videos, official source, and model weights are not
redistributed.

The recorded server prediction predates the strict public runner/scorer
schema in this repository. Its hashes and metrics are retained as
retrospective evidence, but the current code has not yet emitted or validated
that artifact. A clean-commit rerun is required before claiming
current-code reconstruction.

## Frozen official identity

- Repository: `SvipRepetitionCounting/TransRAC`
- Commit: `68bdd4daa60ed7c3174a7f6bf86f6537b6fa0979`
- Source archive SHA-256:
  `b3ec4563f28ff251a87c789de6a0270a709d8656fdc7ec3280648619de10e4ed`
- Extracted 498-file tree SHA-256:
  `2ca6ee835e7f0beb1c505fd7a2d061981c86d023d83b982a5f79da41837d1547`
- Swin backbone SHA-256:
  `9950e3be6b0b3763f80575dfbbe7db7cbeb18345e32363b73e8d44952ef76acc`
- RepCount-A checkpoint SHA-256:
  `32311c4f08bfb1988bfaede8a4691eba5cd5db6928478ee1f26f527e10341a80`

The checkpoint filename contains epoch token `171`, while the loaded payload
records epoch `174`. All 230 state-dict keys loaded with no missing or
unexpected keys. The runner freezes the official `VideoRead` preprocessing,
64 sampled `224×224` frames, scales `{1,4,8}`, and `TransferModel`; shared
half-up integer rounding and the repository metrics are applied outside that
official model path.

## Label firewall

Prediction and scoring are different processes:

For the pending current-code rerun:

1. the runner accepts a label-free sidecar, its byte commitment, the video
   root, and read-only official assets; its CLI has no count, action, label,
   or target argument;
2. it verifies the complete 84-ID commitment, every video hash, source
   archive/tree, backbone, checkpoint, frozen config, clean full Git revision,
   runner bytes, restore audit, output, and prediction receipt;
3. the scorer independently reopens the count-free sidecar and commitment,
   validates their exact 84-video membership and full byte identities, and
   binds every prediction row to its ID, portable locator, and video SHA-256
   before opening the dev-target file;
4. all prediction, receipt, sidecar, commitment, target, runner, and scorer
   files are fully rehashed before an exclusive evaluation/receipt write.

The sealed 105-video test split received no predictions or scoring.

## Why this is modern compatibility, not original parity

The released requirements simultaneously pin PyTorch 1.7.0+cu110,
`mmcv==1.4.8`, and `mmcv-full==1.3.16`, while installation notes also refer
to CUDA 11.4. The audited successful environment instead used Python
3.11.10, PyTorch 2.5.1+cu124, CUDA 12.4, MMCV 1.4.0, timm 0.4.12,
einops 0.3.2, and kornia 0.5.11. Its reconstruction is under
[`docker/transrac-official`](../../docker/transrac-official/ENVIRONMENT.md).
The successful compatibility image ID is
`sha256:ea96abe0e7d17cf4720343e5867256aacd5209aca738087ec92f62b9fe31ff51`.
The direct pip pins and path-free server package inventory document that
image, but are explicitly not a hash-locked, bit-reproducible rebuild recipe.

The released UCFRep evaluation path directly reads labels, contains a
machine-specific dataset path, leaves `ckpt=None`, shuffles evaluation
samples, and reports a count-normalized error with denominator `count+0.1`.
Our adapter calls only the official video loader and model, while the
separate scorer uses the preregistered rounded NMAE definition. These
protocol differences prevent a paper-number parity claim.

## License and redistribution

The source archive contains an Apache-2.0 `LICENSE` file whose SHA-256 is
`ae5e6184c04d6c36e58a8f1c875f3dd3ab4a25a5884ccf41d40bad2dd634a111`,
but the upstream README advertises and links an Anti-996 license. Embedded
or referenced MMACTION/MMCV material, wheels, backbone/checkpoint files, and
the NVIDIA container have separate licensing obligations. Until those are
resolved, this repository publishes only independently authored
runner/scorer/container descriptions and path-free aggregate evidence.
