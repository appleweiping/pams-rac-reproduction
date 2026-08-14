# ICASSP 2027 pre-results paper package

This directory contains the evidence-gated regular-paper draft
**TempoRAC: Identity-Indexed Local Tempo Routing for Multi-Person Repetition Counting**.
It targets four pages of technical content plus an optional fifth page that
contains references only. The official ICASSP 2027 author kit is not yet
available, so the tracked ICASSP 2026 `spconf` files are a temporary scaffold;
`TEMPLATE_PROVENANCE.md` records that boundary.

## Current state

The deliverable is `provisional/data-pending`. Draft mode prints
`DRAFT -- RESULTS PENDING`, renders every controlled result as `PENDING`, and
marks unsynchronized method interfaces as `SYNC-REQUIRED`. The manuscript
contains no proposed-method number or comparative performance conclusion.
Current single-person development artifacts are ineligible for its tables.
Figures 1 and 2 are full-canvas PDF exports of the two committed one-slide
PowerPoint sources; the former custom framework graphic is retired from the
active manuscript. Their captions keep all unimplemented mechanisms
proposal-level. All four display-equation groups contain no punctuation, and
the 16 cited references are real, metadata-checked, and used in the text.

The paper is single-anonymous rather than author-blind. The tracked draft shows
the explicitly approved Weiping Yan, University of Minnesota Twin Cities, and
`yan00944@umn.edu` fields from `author_public.tex`. Unconfirmed collaborator
details remain in the ignored `private/author_metadata.tex`; the submission
validator still rejects an incomplete roster, order, or attestation set.

## Build modes

- Draft: `powershell -ExecutionPolicy Bypass -File scripts/build_draft.ps1`
- Submission checks: `powershell -ExecutionPolicy Bypass -File scripts/check_submission.ps1`

Do not compile `submission_wrapper.tex` directly. Submission checks must bind
the official 2027 kit, a complete author roster, synchronized source contracts,
eligible MultiRep results, generated evidence values, and every required audit.
Until then, submission mode is intentionally fail-closed.

The frozen pre-results PDF is five pages with technical content on pages 1--4
and references only on page 5. The pinned ARIS verifier exits 0 at draft
assurance (`provisional`) and exits 1 at submission assurance (`blocked`). The
proof audit is `NOT_APPLICABLE`, the citation audit passes provisionally, and
the claim/kill-argument audits remain blocking until synchronized method and
result evidence arrives.

## Page-five rule

The live ICASSP 2027 publishing guidance restricts the optional fifth page to
references. Funding, conflict-of-interest, and ethical-compliance declarations
therefore appear before the reference section, within pages 1--4. References
may begin in the remaining page-4 space; page 5 contains only their continuation.
There is no appendix or supplemental technical material in this regular-paper
source.
