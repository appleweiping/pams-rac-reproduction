# Paper Improvement Log

## Baseline (Round 0)

- Artifact: `paper/main_round0_original.pdf`
- Build: 9 pages, letter paper, all fonts embedded, zero final LaTeX warnings,
  zero undefined references/citations, zero duplicate labels, zero overfull or
  underfull boxes.
- Status: anonymous pre-results draft; empirical fields intentionally pending.
- Review coverage: fresh same-family reviewer; Claude cross-family overlay was
  installed but not healthy, so assurance remains provisional.

## Round 1

Reviewer trace: `.aris/traces/paper-improvement/round1/reviewer.md`

Changes applied or queued:

- Marked the training objective as an incomplete schematic and enumerated the
  exact expert/decoder/specialization/calibration evidence required at sync.
- Added an evaluator-ready scored-event contract and blocked official metrics
  until adapter and predicted-track matching tests exist.
- Defined valid-frame mean, units, non-DC frequency support, low-confidence
  fallback, direct-gradient boundary, route-collapse diagnostics, and
  within-track-only continuity semantics.
- Softened overlap-add claims to removal of arithmetic double-counting rather
  than a guarantee against phase-induced duplicate peaks.
- Added independent/sliding-window PAMS, deterministic routing, and stronger
  boundary-fusion controls; removed arbitrary person-indexed weights.
- Pre-registered Period-mAP versus track-conditioned PAMS as the primary paired
  contrast, separated seed/video uncertainty, and scoped multiplicity.
- Expanded common-interface baseline rows and added a protocol-disclosure
  table for modality, labels, track source, and run count.
- Split robustness by pipeline level and added active-identity efficiency
  reporting for counting-only and full-pipeline costs.
- Reworked evidence schemas, story/notation checks, and acceptance-contract
  gates in response to the independent contract review.
- Simplified both editable vector figures, increased task-figure label sizes,
  collapsed the framework to one eight-stage inference spine, and retained an
  explicit provisional training strip.
- Reflowed the soft-router equation and efficiency table to eliminate the two
  new overfull boxes introduced during revision.

Frozen round-1 artifact: `paper/main_round1.pdf`, SHA-256
`63794996aff675a57f522fc54e8c80725b45c688fe122a70bca8eb9de3417ae1`.
Deterministic build QA passes on nine letter-size pages with zero final LaTeX
warnings, undefined citations/references, duplicate labels, overfull boxes, or
unembedded fonts. All nine rendered pages passed visual inspection; the
framework-density and sparse-appendix observations remain non-blocking draft
advisories.

Blocked rather than invented:

- exact expert and decoder learning signals;
- routing/continuity loss definitions and weights;
- official evaluator adapter and matching policy;
- multi-person implementation, artifacts, and all empirical interpretations.

## Round 2

Reviewer trace: `.aris/traces/paper-improvement/round2/reviewer.md`

The fresh reviewer correctly kept the paper at strong-reject/data-pending
status because the synchronized training graph and all MultiRep evidence are
absent. Safe specification changes made without inventing those facts:

- Distinguished ground-truth people from predicted tracker outputs: the raw
  predicted vector is track-wise, and person-wise metrics exist only after the
  frozen evaluator's matching step.
- Replaced broad identity-conditioning language with track-indexed computation
  and stated explicitly that no identity embedding or cross-person interaction
  is currently specified.
- Defined a uniform lag grid, valid-pair support mask, zero-filled unsupported
  lags, and a provisional eligibility/zero-response path for pauses and
  unsupported windows; harmonic, mask-artifact, and activity rules remain
  synchronization blockers.
- Pre-registered dataset tempo/pause/gap/window-eligibility characterization,
  a two-level paired seed/video bootstrap that recomputes Period-mAP, exact
  adapter disclosures, and a stricter label firewall.
- Added controlled/native table panels, matched metric terminology,
  removal/factorial/capacity controls, route-collapse diagnostics, stratum
  sample sizes, a closest-work scope matrix, and explicit synthetic-data limits.
- Labeled Figure 1's left panel as a naïve pooled failure mode, suppressed PDF
  creation metadata, and documented numerical-stabilizer sensitivity checks.

Still blocked rather than invented:

- every expert/decoder/event-score learning signal and loss weight;
- static-window activity/null-expert and harmonic policy from frozen code;
- official event adapter, matching semantics, and MultiRep release statistics;
- any result, improvement, generalization, or real-world MRAC conclusion.

Frozen round-2 artifact: `paper/main_round2.pdf`, SHA-256
`1b6501e8c79d51adaae0660e21d7905123f28442a216eb6de76847d8755bd91d`.
It remains the immutable round-2 snapshot. The delivery `paper/main.pdf` was
subsequently rebuilt after citation-context corrections, so the two files are
intentionally no longer byte-identical. Round-2 deterministic QA passed on
eleven letter-size pages: the main text and planned-result table shells end on
page 8, references and the extended experimental contract follow, and the
conclusion is on page 7.

The final acceptance-contract review was the third and last negotiated round.
Its remaining infrastructure findings were implemented after review, but ARIS
forbids a fourth acceptance round; the contract therefore remains contested
pending a human tie-break, independently of the already-blocking missing data,
official CVPR 2027 kit, and cross-family reviewer health.

## Post-round citation correction and delivery freeze

The citation audit identified one scientifically material wording error: PAMS
tests uniform whole-video speed resampling, not local time warps. The
Introduction now states the supported global-resampling result. TransRAC
pagination and title capitalization were normalized to the IEEE/Crossref
version of record, and ByteTrack title capitalization was normalized to the
Springer version of record. No empirical value or performance claim was added.

An initial fresh proof audit then caught an over-exact algebraic sentence around
Eq. (11): the positive numerical stabilizer means overlap-add does not have
literal coverage-count invariance. The final prose now claims only removal of
the direct unnormalized coverage-count scaling and explicitly discloses the
remaining support-dependent stabilizer bias. The failed pre-fix review is
preserved in `.aris/traces/proof-checker/final-pre-fix/`.

The current delivery PDF is `paper/main.pdf`, SHA-256
`de951023634d5627f3c38cd7512da9fd7df20086781d24caf0a548b93b3d9522`.
Its source-freeze SHA-256 is
`1376b6a388d049b12548a589e9753237bd9e62802091d822cd46cb450aa26184`.
Deterministic build QA passes with 11 letter-size pages, zero final LaTeX
warnings, zero undefined citations/references, zero duplicate labels, zero
overfull/underfull boxes, and all fonts embedded. Every page of this exact PDF
was rendered and visually inspected. This remains a pre-results delivery, not a
submission-ready artifact.
