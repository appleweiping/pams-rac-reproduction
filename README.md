# PAMS RAC Reproduction

[![CI](https://github.com/appleweiping/pams-rac-reproduction/actions/workflows/ci.yml/badge.svg)](https://github.com/appleweiping/pams-rac-reproduction/actions/workflows/ci.yml)

Independent, auditable reproduction of **Count What Repeats:
Period-Adaptive Multi-Scale Consistency for Self-Supervised Repetitive
Action Counting** (CVPR Findings 2026).

> **Current status: implementation / no verified benchmark claim.**
> The source code and protocol are being validated locally. UCFRep videos and
> the designated GPU server are not connected yet, so this repository does
> not publish invented or locally substituted benchmark numbers.

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
pams synthetic evaluate --config configs/pams.yaml --counts 2,4,8,16
```

UCFRep videos are deliberately not redistributed. Follow [DATA.md](DATA.md)
to construct a checksummed local manifest and pose cache.

```bash
pams data prepare-ucfrep /path/to/UCF101 \
  --output data/manifests/ucfrep_526.json
pams data split data/manifests/ucfrep_526.json \
  --output data/manifests/ucfrep_337_84_105.json
```

## Repository layout

```text
configs/              frozen PAMS, protocol, ablation and baseline configs
docs/                 audit, assumptions and baseline protocol ledger
src/pams/             models, losses, inference, data, CLI and adapters
tests/                CPU unit, property and integration tests
```

Every actual training/evaluation run writes an immutable JSON manifest with
the Git commit, configuration and dataset hashes, seed, command and hardware.
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
