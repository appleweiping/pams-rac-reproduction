# TempoRAC code and evidence status (2026-09-30)

This repository now includes the TempoRAC candidate runtime, preprocessing,
certificate, diagnostic, and test sources that were previously present only in
the project's server checkout. The `candidate_v*` modules are preserved as
versioned research implementations; their presence is not an approval of a
final protocol or a claim of benchmark performance.

## What is reproducible here

- `src/pams/temporac/` contains the original contract modules and the
  additive preprocessing, runtime, certificate, and landmark candidates.
- `scripts/experiments/` and `scripts/diagnostics/` contain the corresponding
  training probes and deterministic diagnostic entry points.
- `tests/temporac/` covers the contracts and candidate behavior.
- `data/temporac_p00_v4/` contains only generated X0/operator fixture
  candidates and receipts needed by the deterministic regression tests. These
  are synthetic fixtures, not UCFRep videos, pose features, or model weights.

From a local Windows Python 3.11 environment, `ruff check src tests scripts`
passed; `pytest tests/temporac -q` reported **157 passed, 1 skipped**, and the
full local `pytest -q` reported **1179 passed, 7 skipped**. The skips reflect
Windows symlink/POSIX or unavailable CUDA conditions. GitHub Actions runs a
portable public-checkout suite across Python 3.10–3.12. It explicitly
deselects five historical WARP-PHASE assertions that require an exact frozen
CPython/NumPy runtime or an ignored 125 MB local fixture pack; those tests
remain in the repository and are not counted as passing in Actions. These
checks establish code/fixture consistency, not natural-data efficacy.

## Experiment boundary

The available natural pose cache covers 110 training and 51 validation videos
(268 and 134 identities). It is a **partial-cache, GT-bbox-assisted,
supplied-track pilot**. It cannot support the planned main-paper comparisons.
No sealed test split was used in this synchronization.

The latest server-side v15 certificate-bound diagnostic reported 213/268
training identities and 109/134 validation identities accepted. Because that
candidate was developed after inspecting validation failures, this numerical
coverage is exploratory, not an independent gate pass. The 27 formal
TempoRAC training jobs remain unrun; the manuscript's proposed-method result
cells and paired confidence intervals remain empty. A 56/56 X0 certificate
diagnostic and 157 passing local unit tests do not replace those results.

The server also has limited free space on its system and active-data volumes,
and both GPUs have allocations from other jobs. This repository does not
contain raw training data, licensed weights, private checkpoints, pose caches,
machine credentials, or host-specific paths.

## Reproduction checks and next scientific steps

```bash
python -m pip install -e ".[dev]"
ruff check src tests scripts
pytest tests/temporac -q
```

For a confirmatory paper, first establish a complete, rights-cleared
train/validation pose cache and an independent, untouched evaluation protocol.
Then bind one candidate runtime, run its natural-data entrance audit and
three-seed training matrix, produce paired predictions and intervals, and only
then populate the manuscript tables. Until those steps are complete, the
paper package is a pre-results draft, not a submitted or validated top-conference
paper.
