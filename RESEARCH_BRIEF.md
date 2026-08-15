# Research Brief: Multi-Person Repetition Counting

## Objective

Develop and evaluate an ICASSP 2027 submission on pose-driven multi-person
repetition action counting (MRAC). The intended scientific question is whether
identity-indexed, window-local tempo modeling can handle non-stationary pace
within each person's trajectory without using per-person count or period
annotations for the counting objective.

## Target venue and deliverable

- Venue: ICASSP 2027 regular paper, IEEE conference format
- Paper budget: four technical pages plus a fifth page restricted to references,
  funding, and ethics material
- Working title family: an acronym followed by a colon and a descriptive title
- Final paper must include complete author metadata; only Weiping Yan,
  University of Minnesota Twin Cities, yan00944@umn.edu is currently confirmed
- Until all coauthors, the official 2027 template, and evidence gates are
  complete, submission status must remain blocked or provisional

## Scientific constraints

The preferred hypothesis is a shared pose encoder and shared tempo-specialized
experts with identity-specific state. Each valid track is split into overlapping
local windows; local periodicity and confidence drive soft routing; window
responses are reconstructed per track and decoded once per identity. This is a
research hypothesis, not an implementation claim. ARIS must retain, revise, or
reject it based on code and experiments.

The primary output is a vector of per-person counts. Scene-total count is a
derived diagnostic only. External detector, pose-estimator, and tracker
supervision must be disclosed. Do not use the terms `first`, `annotation-free`,
`end-to-end`, `constant-time`, or `state of the art` without exact qualifying
evidence.

## Required novelty boundary

- MultiCounter already defines multi-person repetition counting
- MultiCounter+ already addresses spatial-temporal consistency and long/short
  period awareness
- PAMS already covers pose-driven self-supervised period-adaptive counting and
  speed-variation analysis
- PoseRAC, DeTRC, D2-STX, RepNet, and TransRAC are required context or baselines

The defensible novelty must come from the intersection of MRAC, pose-driven
counting without per-person count/period labels, identity-conditioned state, and
window-local non-stationary tempo modeling. Novelty review must reject any claim
that merely renames established tracking or repetition-counting components.

## Server evidence discovered on 2026-08-15

Version numbers on the experiment server are not a single ordered release line.
The numerically largest candidate, v63, is a code-only PoseRAC channel classifier
with no matching run, result, model, log, configuration, environment, or data
manifest. The largest result-bearing candidate in that line is v62, a single-seed
dev result with insufficient provenance.

The most complete code/data/result bundle is v46. Its actual method uses
motion/spectral features, an ExtraTrees count expert, integer calibration, and
optional within-video interaction. It does not implement local tempo routing,
fast/medium/slow shared experts, normalized overlap-add, or per-track response
decoding. Its reported metrics are reproducible from the copied result files, but
the 71-video evaluation is outcome-selected and mixes prior validation/test
sources, while the 180-video evaluation records train-test and validation-test
canonical-source overlaps. Existing v46/v62/v63 numbers are therefore audit
inputs only and are ineligible for paper claims.

Hash-verified quarantine copies live outside the repository at
`D:/Project/rac-server-quarantine-20260815`. Raw datasets, credentials, and model
weights must not be committed.

## Experiment requirements

1. Establish a leakage-free MultiRep release, split, and checksum before any
   headline evaluation
2. Bind every run to a Git commit, configuration, seed, software environment,
   hardware record, raw prediction artifact, and log
3. Evaluate at least three seeds and report mean, sample standard deviation, and
   paired video-cluster bootstrap 95% confidence intervals
4. Report Period-mAP, AP50, AP75, AvgMAE, and AvgOBO under protocol-matched
   evaluation
5. Separate oracle/supplied-track and predicted-track settings; retain HOTA,
   IDF1, and ID switches when predicted tracks are used
6. Compare native/contextual MultiCounter and MultiCounter+ results separately
   from common-track baselines such as Track-PAMS and track-global tempo
7. Include local-versus-global tempo, uniform/hard/soft routing, expert removal,
   boundary reconstruction, and only actually implemented auxiliary-loss
   ablations
8. Include a predeclared piecewise time-warp diagnostic that plots local period,
   routing probabilities or entropy, response, and clean-to-corrupt degradation
9. Do not access any sealed single-person test set without a separately frozen
   protocol and explicit authorization

## ARIS execution policy

- Upstream: Auto-Research-In-Sleep commit
  `e12e07c7b85ee1a4dc07e5463089aa16836af2bf`
- Use all official Codex-base skills; no Claude or Gemini overlay
- Every semantic reviewer must be a fresh `gpt-5.6-sol` agent at the reasoning
  tier required by the skill
- Record same-family review honestly as provisional, regardless of numerical
  reviewer score
- Run idea discovery, research refinement, experiment planning, code review,
  server experiment execution/monitoring, experiment audit, result-to-claim,
  auto-review, paper writing, compilation, improvement rounds, proof audit,
  claim audit, citation audit, kill argument, deterministic forensics, and the
  final ARIS verifier
- Do not remove draft warnings or publish to GitHub `main` until the corresponding
  evidence gates have actually passed

