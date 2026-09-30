---
contract_version: 1
created_at: 2026-08-14T20:51:58+08:00
venue: ICASSP 2027 regular
assurance: submission
status: accepted
negotiation_rounds: 3
review_independence: same-family
reviewer_agent: /root/icassp_contract_review
submission_ceiling: provisional/data-pending
evidence_freeze: 26e5b7176f1f7c678391163c3278b5c390d54115
amended_at: 2026-08-14T22:45:00+08:00
amendment_scope: MultiRep metric nomenclature only
amendment_review_status: accepted
---

# ICASSP 2027 Paper Acceptance Contract

This contract grades the requested **pre-results paper package**. It may pass
while submission readiness remains blocked, but it must make every blocker
visible and machine-detectable. A later results synchronization reopens the
claim-bearing assertions and requires a fresh review.

Post-acceptance amendment: a fresh citation audit established that the official
MultiRep names are `Period-AP50` and `Period-AP75`. AC06 uses those names instead
of the earlier shorthand `AP50` and `AP75`; the required evidence, gate, and
acceptance threshold are unchanged. This correction must receive a fresh
same-family amendment review and cannot raise the provisional submission ceiling.

## Assertions

### AC01 — Title and novelty boundary

The draft title is exactly *Identity-Indexed Local Tempo Routing for
Multi-Person Repetition Counting*. No manuscript prose claims the first
multi-person, person-wise, asynchronous, variable-speed, pose-based,
self-supervised, or tempo-adaptive counter.

**Check:** PDF text plus forbidden-claim scan
**Evidence:** `NARRATIVE_REPORT.md`, `CLAIMS_EVIDENCE_MATRIX.md:C01,C15`

### AC02 — Task and output contract

The paper defines supplied identity-indexed pose tracks
(X_i\in\mathbb{R}^{T_i\times J\times C}), validity masks (m_i), and the
primary person-wise vector (\hat{\mathbf c}). A scene total is described only
as a derived diagnostic.

**Check:** method source/PDF and method manifest
**Evidence:** `method_manifest.yaml`; synchronized implementation required for
submission

### AC03 — Implementation-status firewall

Every method component is bound to a collaborator source anchor and test, or is
visibly marked `SYNC-REQUIRED` in draft mode. Submission mode fails if any
component remains unbound. Concept slides and the current single-person PAMS
checkout cannot establish implementation.

**Check:** method-manifest validator and submission build
**Evidence:** collaborator commit, source anchors, unit/integration tests

### AC04 — Supervision qualifier

The only permitted low-label statement is: the intended counting objective uses
no person-wise count or period annotations. Before submission enables this
sentence, the supervision firewall must pass for data loaders, pseudo-labels and
router targets, auxiliary losses, tuning, checkpoint selection, early stopping,
calibration, inference, pose estimation, detection, tracking, and evaluation.
The paper separately discloses every external supervision and pretraining source.

**Check:** exact-context scan and all-scope supervision-firewall manifest
**Evidence:** `method_manifest.yaml`, collaborator training/config artifacts

### AC05 — Abstract result binding

The 140–160-word abstract contains one controlled result sentence. Draft mode
renders a conspicuous pending sentence and banner; submission mode fails unless
every number is generated from eligible frozen evidence and bound in
`evidence_values.tex`. Every populated result must belong to a new synchronized
MultiRep artifact family with `table_eligible=true`, exact protocol bindings,
required seeds/uncertainty, and passing experiment-audit and result-to-claim.
Every artifact in the C11 exclusion register is forbidden. The abstract contains
no citation or undefined acronym.

**Check:** word/count scanner, PDF text, generated-value provenance

### AC06 — Main comparison integrity

Table 1 visually separates native/contextual MultiCounter and MultiCounter+
results from common-track Track-PAMS, track-global tempo, and the proposed
method. It never bolds or ranks values across incompatible protocols, and each
cell maps to the same eligible synchronized MultiRep artifact family or remains
visibly pending. It reports Period-mAP, Period-AP50, Period-AP75, AvgMAE, and AvgOBO; where
predicted tracks apply, HOTA, IDF1, and IDSW are reported as frontend
diagnostics. Every reproduced learned method uses at least three seeds, mean,
sample standard deviation, and paired video-cluster bootstrap 95% confidence
intervals for the principal comparison. A genuinely unavailable contextual
value is marked `N/A`, never silently dropped. Every C11 exclusion-register
artifact is forbidden.

**Check:** table source, caption, manifest keys, protocol labels

### AC07 — Mechanism evidence

Table 2 contains only implemented variants: local/global tempo,
uniform/hard/soft routing, no-expert routing, boundary reconstruction
alternatives, and removals of auxiliary losses that actually exist. Every row is
bound to one commit/config and paired seed set within an eligible synchronized
MultiRep artifact family; absent modules are deleted, not simulated. The table
cannot populate before experiment-audit and result-to-claim pass.

**Check:** ablation manifest and generated table

### AC08 — Figure provenance and meaning

Figure 1 is an editable vector asset showing supplied tracks/masks, shared
parameters versus per-track state, local period evidence, soft experts,
normalized overlap-add, and one decoder per identity. Figure 2 is explicitly a
data-pending shell until generated from a frozen diagnostic; no illustrative
curve is presented as measurement. A populated Figure 2 must bind the
preregistered selection rule, paired comparison, source data, deterministic
generator, exact protocol, and eligible synchronized MultiRep artifact family.

**Check:** draw.io/SVG/PDF sources, figure manifest, captions, visual review

### AC09 — Citation existence, metadata, and context

Every cited item resolves through DBLP, CrossRef, CVF, IEEE, Springer, or an
official publication page; BibTeX matches the final published version; and every
citation context is independently judged to support the sentence. Google Scholar
is discovery-only. No uncited entry remains.

**Check:** fresh `CITATION_AUDIT.{md,json}` and BibTeX/cite-key diff

### AC10 — ICASSP 4+1 page rule

Pages 1–4 contain all technical content. If page 5 exists, it contains only
references, funding acknowledgments, and Compliance with Ethical Standards.
There is no appendix, technical footnote, equation, figure, table, caption, or
new claim on page 5.

**Check:** page-aware text/object scanner and rendered visual review

### AC11 — Author and privacy gate

Tracked files contain no personal author identity or email. Draft PDF shows an
author-list-pending marker. Submission mode requires the private author include,
`RACAuthorMetadataComplete=true`, exact agreement of all names, affiliations,
emails, and order with the submission system, and recorded assent from every
author to the roster, order, and final submitted manuscript.

**Check:** tracked-tree privacy scan, author include validator, PDF text

### AC12 — Venue, ethics, funding, and AI-policy gate

The temporary 2026 template is labeled a scaffold. Submission mode requires a
verified ICASSP 2027 kit, exact EDICS selection, complete funding/COI and ethics
statements, documented author review against the official 2027 LLM policy,
originality/no simultaneous-submission attestations, and confirmation that every
author remains within the ICASSP/SPS nine-paper limit. None is guessed.

**Check:** template provenance and declaration manifest

### AC13 — Claim-language gate

A scanner rejects `first`, `annotation-free`, `label-free`, `end-to-end`,
`constant-time`, and unmatched `SOTA/state of the art` in title, abstract,
Introduction, Method, Experiments, contributions, captions, results, and
conclusion, except clearly attributed prior-work statements or explicit
negation. The paper-claim audit maps every novelty, method, empirical,
comparative, and causal assertion to a ledger row whose status licenses its
wording; an unmapped or blocked affirmative claim fails submission.

**Check:** source/PDF forbidden-claim scan with allowlisted contexts

### AC14 — Dual-mode build behavior

Draft build succeeds with `DRAFT — RESULTS PENDING` and visible pending cells.
Submission build fails on `PENDING`, `SYNC-REQUIRED`, `[VERIFY]`, empty
tables, uncontrolled numbers, missing authors, missing 2027 kit, or stale
evidence hashes.

**Check:** positive draft build plus negative submission self-tests

### AC15 — Fresh ARIS assurance chain

The ICASSP source receives two fresh zero-context reviews and fresh proof,
paper-claim, citation, and kill-argument audits. All audit inputs bind to the
current ICASSP source/PDF. Both review rounds and every mandatory audit must be
nonblocking; actionable findings are resolved and rechecked against the final
source/PDF. A proof audit may return `NOT_APPLICABLE` only with a reason. The
pinned ARIS verifier is executed under `assurance=submission` and must exit 0
with no blocking audit verdict for submission. A required cross-family review
is healthy only when it returns a substantive nonblocking assessment, all
actionable findings are resolved and rechecked, and the request/response trace
is complete; a merely successful response is insufficient. Otherwise the
outcome remains provisional or blocked.

**Check:** round0/1/2 PDFs, traces, four audit JSON files, verifier report

### AC16 — Derivation and delivery integrity

The package records the CVPR freeze parent, ARIS pin, input hashes, template
provenance, generated outputs, and current blockers in append-only manifests.
The original `D:\Project\rac` HEAD and 96-entry dirty state are unchanged; no
experiment branch is merged, no sealed test is run, and no remote is pushed.

**Check:** `DERIVATION_PROVENANCE.json`, `MANIFEST.md`, before/after Git audit

## Current grading ceiling

- Pre-results paper package: eligible for acceptance when AC01–AC16 are
  implemented with explicit blockers and the draft PDF passes deterministic QA.
- Conference submission: **blocked** until collaborator code/results, complete
  authorship, ICASSP 2027 kit/policies, fresh audits, and healthy cross-family
  review are all present.
