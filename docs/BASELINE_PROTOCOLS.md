# Baseline protocols and clean-room policy

The registry is an audit surface, not a promise that every named paper has already been
implemented. Querying a `BaselineSpec` is always safe. Creating an adapter succeeds only
for an honestly runnable implementation; blocked methods raise
`BaselineUnavailableError` with a machine-readable status and reason.

## Registered methods

| Registry key | Target evidence | Current status | Comparison protocol |
|---|---|---|---|
| `repnet` | official current `ckpt-70` sanity, then independent adapter | `blocked_unimplemented`; PAMS table cell `protocol-unverifiable` | UCFRep-526 fair |
| `transrac` | official checkpoint modern-compat dev sanity complete, then clean-room | `blocked_unimplemented`; parity/test blocked | UCFRep-526 fair |
| `escounts` | official sanity, then clean-room | `blocked_unimplemented` | UCFRep-526 fair |
| `ivac-p2l` | official checkpoint modern-compat dev sanity complete, then clean-room | `blocked_unimplemented`; parity/test blocked | UCFRep-526 fair |
| `poserac-v1` | released oracle diagnostic plus fair rewrite | `blocked_unimplemented` | UCFRep-pose-110 |
| `poserac-iconip24` | separately verified rewrite | `blocked_unimplemented` | UCFRep-526 fair |
| `gmfl` | clean-room | `blocked_unimplemented` | UCFRep-pose-110 |
| `jtsps-count-only` | protocol audit before implementation | `blocked_protocol` | UCFRep-526 target |
| `spkdb` | clean-room | `blocked_unimplemented` | UCFRep-pose-110 |
| `bigc` | clean-room | `blocked_unimplemented` | UCFRep-pose-110 |
| `countllm-lite` | reduced-resource recipe | dynamic `blocked`/`smoke_only` | non-comparable UCFRep-526 |
| `spectral-proxy` | deterministic pipeline smoke | `reference_only` | no leaderboard |

The status is deliberately conservative. An architecture moves to `ready` only after
its model, preprocessing, checkpoint provenance and prediction artifact have passed
method-specific parity tests.
No blocked baseline method ID is accepted by the sealed metrics command. The
production sealed-report allowlist is empty in this revision; this prevents an
arbitrary JSON file from being published under RepNet, TransRAC, or another
paper method name.

## Table separation

### Reported/source-faithful table

This table answers “what did the source report or what does its own released evaluator
produce?” Every row carries its native dataset, split, metric code, checkpoint and oracle
flags. It may contain the PoseRAC-v1 GT-count diagnostic, but that row must display an
`ORACLE: TEST COUNT USED` warning and cannot be ranked against fair rows.

### Fair UCFRep-526 table

This table answers “how do independently runnable methods compare on the same 421/105
video IDs?” Inclusion requires all of the following:

- prediction generated for every admissible held-out video under one published failure
  policy;
- no held-out count, action identity, cycle boundary or result-dependent threshold is
  read by the model or adapter;
- primary NMAE and OBO computed by the shared repository evaluator;
- per-video predictions, split hash, configuration hash, checkpoint hash and Git SHA;
- modality and use of external pretraining shown in the row.

Reported UCFRep-pose-110 numbers cannot fill missing UCFRep-526 cells.

### Fair UCFRep-pose-110 table

This table uses exactly 89 training and 21 held-out videos from the five-action pose
subset. The same no-label-at-inference rule applies. Training-only salient-pose or cycle
annotations must be versioned and may cover only the 89 training videos.

## Clean-room implementation policy

Primary papers and public artifacts may be studied to write an independent behavioral
specification. Third-party source is not copied into `src/pams`. Official code runs, when
licenses allow it, live in an isolated environment and serve only as provenance-labeled
sanity evidence. Each independent implementation must document:

1. architecture and preprocessing statements supported by a primary source;
2. every inferred choice and its frozen value;
3. checkpoint origin, license, digest and conversion;
4. native versus fair-protocol differences;
5. a tiny deterministic parity fixture before any full evaluation.

No spectral, constant-count or nearest-result proxy may be registered under a paper's
name. The only deterministic placeholder is explicitly called `spectral-proxy`, has
`is_paper_baseline=False`, and is ineligible for comparison tables.

## Source notes

- RepNet: [official source](https://github.com/google-research/google-research/tree/ec7c3d346277b737bc2decffcd1b533d4b7ec105/repnet),
  [project page](https://sites.google.com/view/repnet), and the separately
  maintained [unlicensed legacy PyTorch conversion](https://github.com/materight/RepNet-pytorch).
  See the [asset/protocol audit](baselines/REPNET_AUDIT.md).
- TransRAC: [official repository](https://github.com/SvipRepetitionCounting/TransRAC).
  The official checkpoint has a strict development-only modern-compatibility
  sanity result; original evaluator parity and sealed test evaluation remain
  blocked. See the [asset/protocol audit](baselines/TRANSRAC_AUDIT.md).
- Every Shot Counts: [official repository](https://github.com/sinhasaptarshi/EveryShotCounts).
- IVAC-P2L: [official repository](https://github.com/hwang-cs-ime/IVAC-P2L).
  The official RepCount-A checkpoint has a strict current-code
  development-only modern-compatibility sanity result. The one-row legacy
  ties-to-even versus frozen half-up rounding difference is reported
  explicitly; original evaluator parity and sealed test evaluation remain
  blocked.
- PoseRAC: [paper](https://arxiv.org/abs/2303.08450) and
  [official repository](https://github.com/MiracleDance/PoseRAC).
- GMFL: [primary paper](https://arxiv.org/abs/2409.00330).
- JTSPS: [primary DOI](https://doi.org/10.1109/TCSVT.2024.3402728).
- SPKDB: [primary article](https://www.sciencedirect.com/science/article/pii/S1077314225001572).
- BIGC: [primary DOI](https://doi.org/10.1016/j.engappai.2025.110996).
- CountLLM: [primary paper](https://arxiv.org/abs/2503.17690).

## CountLLM-Lite resource gate

`inspect_countllm_lite_resources()` reads CUDA device properties without loading a
model. Training eligibility requires at least one GPU with **48 GiB total memory**.

- No CUDA: `blocked_resource`.
- CUDA below 48 GiB: `smoke_only`; only import/configuration smoke tests are permitted.
- At least 48 GiB: hardware gate passes, but status remains `blocked_unimplemented`
  until the independent Vicuna/video-encoder adapter is built and validated.

The recipe uses Vicuna-7B, 32 frames, LoRA rank 16, 4-bit NF4, batch 1 with accumulation
32, and UCFRep stages 2/3 for 50 epochs each. It omits WebVid-10M stage 1 and therefore
must never occupy the original CountLLM same-condition result column.
