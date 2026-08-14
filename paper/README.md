# CVPR 2027 pre-results paper package

This directory contains the anonymous, evidence-gated manuscript for
**Count Each Person at Their Changing Pace: Multi-Person Tempo-Adaptive
Repetition Counting**.

The current deliverable is a pre-results draft. It deliberately compiles with
the visible banner `DRAFT — RESULTS PENDING`, contains no new-method result
number, and makes no performance or state-of-the-art claim. The collaborator's
frozen implementation and MultiRep artifacts are required before the paper can
enter submission mode.

## Build modes

- Draft: `powershell -ExecutionPolicy Bypass -File scripts/build_draft.ps1`
- Submission gate: `powershell -ExecutionPolicy Bypass -File scripts/check_submission.ps1`

The draft command compiles `main.pdf`. The submission gate is intentionally
expected to fail while controlled evidence fields, `SYNC-REQUIRED` interfaces,
or draft markers remain. It also rejects unsupported headline terms such as
`first`, `annotation-free`, `end-to-end`, `constant-time`, and unaudited
state-of-the-art language.

## Anonymity

No author name, affiliation, email address, Git identity, or private metadata
belongs in this directory. Author metadata is maintained outside the Git
worktree and may only be injected after the venue's blind-review phase.

## Template status

CVPR 2027 has not published an official author kit as of 2026-08-12. This draft
vendors the official CVPR 2026 LaTeX kit at tag `CVPR2026-v1(latex)` as a
provisional layout dependency. See `TEMPLATE_PROVENANCE.md`; replacing it with
the official CVPR 2027 kit is an acceptance-contract gate.
