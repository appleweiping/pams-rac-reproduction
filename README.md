# PAMS RAC Reproduction

[![CI](https://github.com/appleweiping/pams-rac-reproduction/actions/workflows/ci.yml/badge.svg)](https://github.com/appleweiping/pams-rac-reproduction/actions/workflows/ci.yml)

Independent, auditable reproduction of **Count What Repeats:
Period-Adaptive Multi-Scale Consistency for Self-Supervised Repetitive
Action Counting** (CVPR Findings 2026).

> **Current status: partial reproduction / no verified benchmark claim.**
> Historical PAMS-Literal and inferred PAMS-SSHead runs completed all three
> seeds on the fixed 84-video development split and missed the frozen gate.
> The v8 longest-contiguous-track protocol correction then improved matched
> seed-2026 NMAE/OBO from `0.735728 / 0.190476` to
> `0.643123 / 0.297619`, but still failed its preregistered
> `0.60 / 0.30` expansion gate; seeds 42/3407 were therefore not authorized.
> The official current RepNet checkpoint and official TransRAC and IVAC-P2L
> checkpoints have separate
> development-only sanity results; both use explicitly labeled modern
> compatibility environments and are not original-protocol parity. The sealed 105-video
> UCFRep test split remains untouched, so no test metric or successful
> reproduction claim exists.

## Why this repository exists

The paper does not publish an implementation and leaves several details
non-identifiable. In particular, its Period Head is described as being used
only at inference and no loss or gradient path is specified for training it.
The comparison table also mixes standard UCFRep (526 videos) with results
reported on UCFRep-pose (110 videos). This project preserves those facts
instead of silently filling gaps and presenting the result as author code.

Two explicitly separated PAMS variants are provided:

- `PAMS-Literal`: the Period Head remains randomly initialized, matching the
  literal training description. It is a diagnostic, not the expected primary
  model.
- `PAMS-SSHead`: freezes the PAMS encoder and trains the head with an inferred
  self-supervised cycle/spectral objective. It is always labelled as a
  reproduction completion, not an author-disclosed component.

See the [implementation specification](docs/METHOD_SPEC.md),
[paper audit](docs/PAPER_AUDIT.md), and
[assumption register](docs/ASSUMPTIONS.md) before interpreting results.

## Reproduction protocol

- Standard UCFRep: 421 train / 105 sealed test.
- Development split: deterministic stratified 337 / 84 using seed `2026`.
- Fixed experiment seeds: `42`, `2026`, `3407`.
- Primary metric: normalized MAE,
  `mean(abs(round(prediction) - ground_truth) / ground_truth)`.
- Additional metrics: raw MAE, RMSE, OBO, per-video predictions and paired
  bootstrap confidence intervals.
- A run is `verified` only when the three-seed mean reaches NMAE ≤ 0.228 and
  OBO ≥ 0.666, at least two seeds pass independently, and the preregistered
  ablation directions hold.

The Table 2 YAML is currently a preregistration ledger only. Its non-full
variants are not executable and cannot enter sealed scoring until their
behavior and config provenance are implemented. UCFRep-pose sealed scoring
and all paper-baseline sealed IDs are also blocked in this revision.

## Quick start

Python 3.10–3.12 is supported.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
pytest
pams --help
```

Validate the frozen configuration and run the synthetic smoke evaluation:

```bash
pams config validate configs/pams.yaml
pams synthetic evaluate --config configs/pams.yaml
pams synthetic counter-gate
pams synthetic period-counter-gate
pams synthetic sshead-collapse
```

`synthetic evaluate` defaults to the preregistered count sweep `2..40` and
the frozen speed, pause, missing-joint, harmonic, rotation, scale, and
translation stresses. It is a deterministic spectral-proxy diagnostic, not
an encoder/head or benchmark result.
`counter-gate` is a frozen 576-case sign/phase test of the multi-expert
counter only: it is deliberately supplied the exact synthetic period and is
not an encoder/head/period-estimator or end-to-end PAMS result.
`period-counter-gate` runs the same frozen matrix through FFT/autocorrelation
period estimation and then the counter. It still bypasses the learned encoder
and Period Head, so it is a component-chain test rather than UCFRep evidence.
`sshead-collapse` is deliberately diagnostic and currently exits nonzero: it
documents the inferred loss's exact-constant dead point and near-collapse
gradient spike. Formal SSHead training therefore also has per-step finite,
collapse, and gradient guards before every optimizer mutation.

UCFRep videos are deliberately not redistributed. Follow [DATA.md](DATA.md)
to construct a checksummed local manifest and pose cache. Preparation accepts
`UCF-101/<Action>/<video>.avi`, `<Action>/<video>.avi`, and
`flat/<video>.avi` relative to the supplied root. Its default strict mode
requires and hashes all 526 videos; duplicate candidates across layouts fail.
The resolver contains one explicit audited directory alias:
UCFRep's `HandStandPushups` identifiers map to the official UCF101
`HandstandPushups` directory. It performs no recursive or case-insensitive
fallback.

```bash
pams data prepare-ucfrep /path/containing/UCF-101 \
  --output data/manifests/ucfrep_526.json
pams data split data/manifests/ucfrep_526.json \
  --output data/manifests/ucfrep_337_84_105.json
```

`--no-hash-videos` keeps the 526-file existence check but skips video hashing.
`--annotation-only` is the explicit non-experiment mode that permits missing
videos.

Pose caches use a dedicated `pose_fingerprint` derived only from frozen data
preprocessing and MediaPipe settings. Training seeds and optimizer changes do
not invalidate the cache. Resume an interrupted extraction with
`pams pose extract ... --skip-existing`; an existing entry is skipped only
after its source-video hash and complete cache provenance match.

## Repository layout

```text
configs/              frozen PAMS, protocol, ablation and baseline configs
docs/                 audit, assumptions and baseline protocol ledger
src/pams/             models, losses, inference, data, CLI and adapters
tests/                CPU unit, property and integration tests
```

Every actual training/evaluation run writes an immutable JSON manifest with
the Git commit, full experiment-configuration and dataset hashes, seed,
command and hardware. The smaller pose fingerprint is used only to identify
pose caches; an additional ordered cache-set digest binds the exact `.npz`
bytes consumed by each checkpoint and evaluation.
The current publication state is recorded in [results/README.md](results/README.md);
empty cells are intentional and are never filled with proxy measurements.
Designated GPU execution follows the credential-free
[server runbook](docs/SERVER_RUNBOOK.md).

## Baseline policy

Official checkpoints are used only in a separately labelled source-sanity
track. Fair-table implementations are clean-room rewrites and never use test
count or test action labels to choose an output. A paper baseline remains
`not-implemented`, `protocol-unverifiable`, or `resource-blocked` until the
corresponding implementation and assets genuinely exist.

## License

Original code in this repository is licensed under Apache-2.0. Datasets,
papers, third-party code and model weights retain their own licences and are
not relicensed or redistributed here.
