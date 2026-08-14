---
contract_version: 2
status: contested-round3-human-tiebreak-required
negotiation_rounds: 3
reviewer_route: same-family/fresh-zero-context
same_family_verdict_ceiling: provisional
assurance: submission
submission_readiness_ceiling: provisional/data-pending
item_count: 16
evidence_freeze: 73cbc4bb9b581f24fbc90f2fe7421bc11e7c7454
venue: CVPR 2027
template_status: CVPR2027-template-pending
---

# Paper Acceptance Contract

This contract defines machine-checkable completion criteria for the anonymous
pre-results manuscript and its later result synchronization. Three same-family
rounds returned required revisions; the contract is now contested and requires
a human tie-break rather than a fourth same-family review. Missing collaborator
evidence, the official CVPR 2027 kit, and a
healthy cross-family reviewer remain hard blockers rather than waivable prose
qualifications.

## Machine-Checkable Assertions

| ID | Assertion | Deterministic or artifact-based check | Current state | Blocking closure |
|---|---|---|---|---|
| AC01 | The evidence and delivery freezes are immutable and mutually identified. | `paper/scripts/validate_delivery_freeze.py` enforces the exact v2 freeze fields, independently computes the source-freeze digest, verifies every `MANIFEST.sha256` entry and exact tracked-deliverable coverage, requires the manifest to exclude itself, and checks an external receipt against `HEAD`, an actually clean worktree, the manifest digest, and the source-freeze digest; its submission-mode report is written outside the repository so the clean-tree assertion remains true. | BLOCKED | After the final local commit, verify the clean tree and manifest, then create the exact-schema external receipt without changing the committed tree. Any collaborator or source sync creates a new source freeze and reruns all audits. |
| AC02 | The manuscript tells exactly one What–Why–So-What story with six fixed mechanism qualifiers. | `paper/evidence/story_contract.json` v2 fixes canonical What/Why/So-What text, source anchors, method order, core equations, and the sole claim maps. `paper/scripts/validate_story.py` checks source and extracted-PDF title, exactly one draft abstract placeholder with no result number, notation, equation punctuation/order, exactly three contributions, and a complete one-to-one map for every experiment subsection. | PROVISIONAL | Rerun against text extracted from every rebuilt PDF; synchronize the contract if collaborator code changes the story. |
| AC03 | The exact title remains “Count Each Person at Their Changing Pace: Multi-Person Tempo-Adaptive Repetition Counting” without implying unsupported primacy or validation. | Story checker and extracted-PDF check require an exact normalized title. The full-scope claim scanner in AC09 must report zero non-allowlisted primacy, annotation-free, end-to-end, constant-time, or SOTA claims. | PROVISIONAL | Rewrite only through a reviewed plan/contract revision. |
| AC04 | Draft Abstract has exactly one controlled result placeholder and no fabricated value; submission requires one audited replacement. | `sections/0_abstract.tex` contains exactly one `\ControlledAbstractResult` and no `\R`, `\C`, explicit metric value, percent improvement, or mean±SD. Submission validation requires exactly one `\DeclareAbstractResult`, whose `paper_bindings.abstract_result.result_keys` resolve to audited manifest rows. | PROVISIONAL | Populate only after experiment-audit, result-to-claim, and paper-claim audit PASS on the same freeze. |
| AC05 | Every proposed component has a checkable transition from BLOCKED to CONFIRMED. | `paper/evidence/method_contract.yaml` v2 defines “component,” required fields, anchor/test schemas, allowed verdicts, and a closure rule. Every ordered module and training term has `status`, `sync_required`, `evidence_class`, sources, implementation anchors, test artifacts, hashes, verdicts, and a closure rule. A validator rejects CONFIRMED unless at least one anchor and all tests are digest-bound PASS artifacts at the collaborator freeze. | BLOCKED | Fill collaborator paths/symbols/commit/hashes and pass all named tests; a name or prose description alone never closes a component. |
| AC06 | Method notation and operation order are explicit and checkable. | `paper/evidence/notation_inventory.json` enumerates every required symbol and its definition regex, including (d) and (epsilon). `validate_story.py` requires every anchor. A LaTeX QA scan requires the seven core objects, zero punctuation immediately after display equations, and no final sum of independently rounded window counts. | PROVISIONAL | Reconcile the inventory and equations with frozen code, then rerun notation and compile QA. |
| AC07 | The supervision claim is limited to no person-wise count **or period** labels in the intended counting objective. | `method_contract.yaml:task.supervision_claim` fixes the allowed text and audits loaders, pseudo-label generation, losses, router inputs/targets, tuning, checkpoint selection, early stopping, calibration, and inference for both count and period leakage. Stronger “annotation-free,” “label-free,” or fully self-supervised-pipeline wording fails AC09. | BLOCKED | Produce a digest-bound collaborator label-firewall audit with source anchors and PASS verdict. |
| AC08 | Exactly three contribution bullets and every experiment subsection map to authoritative ledger IDs. | `story_contract.json` is the sole map: contribution 1→C01/C02, 2→C03/C04/C05/C06, 3→C10/C12/C15. Every contribution and experiment subsection contains an invisible `\ClaimMap{...}` marker; `validate_story.py` requires exact ordered equality. Blocked contributions remain proposed or future-tense. | PROVISIONAL | Update the ledger, story contract, and LaTeX markers together through reviewed revision. |
| AC09 | Unsupported headline and comparative language is absent from every submission-facing artifact. | `paper/scripts/claim_scan.py` scans all submission LaTeX, appendix/supplement, generated text/tables, `.drawio`/`.svg` text, and an extracted-PDF file re-derived from and SHA-bound to the supplied PDF in submission mode. Only named exact phrase-level denials are allowed; own-method positive comparisons require both a manifest-resolved `\\C{...}` claim and `\\Comparison{...}` comparison in local context. | PROVISIONAL | Keep the prohibition unless a protocol/novelty audit supports narrower wording; submission mode cannot run source-only and extracted text must be regenerated after every build. |
| AC10 | Two editable, vector, explicitly conceptual figures exist and are visually usable. | `paper/scripts/validate_figures.py` requires and hashes the `.drawio`/`.svg`/`.pdf` triplets, validates Draw.io and SVG XML with zero errors/warnings, rejects rasterized figure PDFs, checks captions and `visual_reference_only` PPT classification, and requires a digest-bound standalone plus current-PDF page-render visual review. | PROVISIONAL | Standalone figures passed visual inspection; regenerate and inspect current PDF page renders, then rerun after collaborator method synchronization and CVPR 2027 template migration. |
| AC11 | Every result/protocol/claim binding and every generated table or plot is immutable and traceable without a digest cycle. | Result schema v2 represents one method/protocol/config per family and digest-binds every GT, prediction, log, metric source, audit, table, figure, and source-data artifact. Paths must be safe forward-slash paths relative to the bundle root. `evidence_precheck.py` resolves every artifact, verifies exact SHA-256/media, and requires exact package coverage. The package manifest excludes only itself; `external_delivery_receipt_id` is a logical identifier and the actual receipt is supplied by an out-of-bundle CLI path, so no package path can point back to the receipt. Every protocol binding names `target_scope` and an RFC 6901 `target_path`; the validator resolves the exact target value and verifies that value plus the declared digest against the bound source bytes. | BLOCKED | Ingest synchronized artifacts; generate the acyclic package manifest and truly external receipt; pass JSON Schema, `validate_result_manifest.py`, `evidence_precheck.py`, exact `generate_evidence_tex.py` verification, and all empirical audits. The absent synchronized manifest remains a deliberate submission blocker, and current dev84/proxy values remain permanently excluded. |
| AC12 | Main comparisons separate common-input rows from native MRAC context and fully disclose protocol. | Every family carries digest-bound provenance and explicit `eligible`, `context_only`, or `excluded` state; eligibility must agree with `table_eligible`, real-GT provenance, and the exact required PASS audits. A repeated method ID must retain identical semantic identity fields across protocols. Eligible oracle and common-predicted blocks must contain exactly one each of RepNet, TransRAC, PoseRAC, PAMS, and Ours; eligible native context must contain exactly MultiCounter and MultiCounter+. All eligible common-predicted rows must share identical frontend/config/source/track hashes and identical HOTA/IDF1/IDSW values; oracle rows must share one oracle-track hash. Generated outputs and protocol bindings may reference only eligible families. | BLOCKED | Supply protocol-complete, provenance-complete families for every displayed row and pass the exact-set and cross-family hash validator. |
| AC13 | Repeated-seed and uncertainty claims are arithmetically reproducible from completed runs. | Every family has a unique and complete protocol-specific metric set. `multi_seed_reproduction` requires at least three distinct seeds; each aggregate has a global ID and its exact run set, count, mean, and sample SD are recomputed. `official_single_checkpoint` requires exactly one run and null SD. Exactly one eligible primary comparison must be common-predicted Ours-versus-PAMS Period-mAP. Its one-to-one run pairs cover both families exactly once at identical seeds. A digest-bound JSON artifact supplies unique video clusters, the full seed/run pairing, and per-side values; the validator recomputes cluster differences, the overall estimate, and the two-sided 95% type-7 percentile-bootstrap bounds from the declared seed/resample count, rejecting non-finite, reversed, or out-of-range values. | BLOCKED | Run and freeze all paired seeds and per-video records, then pass deterministic aggregate and bootstrap recomputation; official one-checkpoint rows remain contextual and make no variance claim. |
| AC14 | Every citation exists, has canonical final-version metadata, and entails each local context. | Zero undefined keys at compile; every cited key receives its own fresh-review trace. `aggregate_citation_audit.py` digest-binds every entry and reviewer trace, computes the current paper/source freeze, and accepts only an exact one-to-one `reviewed_contexts` set of `{file,line,context_sha256,verdict: SUPPORTS}` records. A citation occurrence is never auto-labelled SUPPORTS. Submission requires no unresolved FIX/REPLACE/REMOVE, context mismatch, missing trace, or `[VERIFY]`; content replacement/removal requires user approval. | BLOCKED | Explicitly confirm the generated exact-context binding template against the prior fresh reviews, copy the confirmed rows into each entry, and rerun after every citation-context change. |
| AC15 | Current files and anonymous PDF contain no private author identity; history scope is explicit. | `paper/scripts/anonymity_scan.py` takes an out-of-repository secret list; scans submission text and release-selected logs; inspects PDF/PNG binary metadata and raw secret encodings; re-extracts and SHA-binds PDF text; and requires exactly one anonymous author declaration in both source and visible PDF text. Reports expose only redacted counts and digests. Reachable Git history remains a separate pre-push gate. | PROVISIONAL | Rerun against the final PDF and selected release binaries; run the separate reachable-blob and commit-metadata scan before any future blind push. |
| AC16 | Final readiness is artifact-driven and tied to one freeze. | `paper/scripts/validate_readiness.py` invokes every deterministic gate, mandatory result `evidence_precheck.py`, and byte-exact `generate_evidence_tex.py` verification. It validates semantic audits, external receipts/manifests, same-source-freeze/timestamp bindings, exact build-source coverage and source freeze, official CVPR 2027 vendored-file coverage and digests, PDF/page/font/anonymity checks, two reviews, contract state, the pinned upstream verifier, and digest-bound Claude health trace plus probe. Submission-mode runtime reports remain outside the repository. Missing inputs and any provisional/BLOCKED/FAIL/ERROR state fail closed; only PASS and proof-only `NOT_APPLICABLE: no_theorems` are accepted. | BLOCKED | Obtain collaborator evidence, official kit, current page review, final receipts/manifests, and a healthy digest-bound Claude review; regenerate every audit with the same source freeze, obtain the required human contract tie-break, close AC01–AC15, and rerun the aggregator. |

## Global Failure Conditions

Submission mode fails if any submission-facing source, generated artifact, or
extracted PDF contains `TBD`, `SYNC-REQUIRED`, `[VERIFY]`, a pending/draft
banner, an unresolved evidence macro, an empty result cell, an unbound number,
or a non-allowlisted forbidden claim. It also fails on a dirty/unmanifested
delivery freeze, missing digest, stale audit, anonymity leak, or non-acceptable
verdict. A draft build may render pending markers only while the PDF visibly
states `DRAFT — RESULTS PENDING` and the state remains
`provisional/data-pending`.

No human waiver exists at contract creation. A later waiver must name exactly
one assertion, give a technical reason, identify an approving human only in a
private record, and record the date and affected freeze before re-grading.

## Round-1 Revision Record

All eleven required reviewer items were addressed: immutable freeze mechanics;
machine-readable story/notation inventories; component transition schema;
count-and-period firewall; one authoritative claim map; full-scope claim scan;
figure-state reconciliation; per-method/protocol result families; digest-bound
artifact and external manifest-sidecar semantics; per-family seed/bootstrap
linkage; and executable anonymity/final-readiness gates. Acceptance remains
pending until a fresh round-2 reviewer checks this revision.

## Round-2 Infrastructure Hardening Record

AC01, AC02, AC09, AC10, AC15, and AC16 now name executable fail-closed
validators rather than prose-only intentions. Their draft checks may report
`PROVISIONAL` or `BLOCKED`; those states are deliberately not submission
passes. This record documents infrastructure remediation only and does not
pre-empt the independent round-2 acceptance decision.

## Round-2 Implementation Record (AC11--AC13)

The result contract now removes the manifest/sidecar cycle: the package
manifest binds the result manifest and all referenced artifacts, while an
external receipt binds the exact result-manifest and package-manifest bytes.
The deterministic precheck verifies safe path resolution, bytes, media, and
complete package coverage. Cross-family validation now enforces stable method
identity, globally unique record IDs, per-family provenance/eligibility, exact
protocol family sets, shared frontend/track evidence, and exact paper targets.
Arithmetic validation now requires complete metric sets, recomputes every mean
and sample SD, constrains official checkpoints to one run, and reconstructs the
primary video-cluster bootstrap from explicit same-seed run pairs. No result
manifest was created: absent collaborator evidence remains a hard submission
blocker rather than fixture evidence.

## Round-3 Final Same-Family Review Record

The final permitted same-family acceptance review returned `REVISE`. Its
remaining infrastructure findings were implemented as fail-closed checks:
mandatory evidence precheck and exact evidence-TeX reproduction; exact build
source coverage/source-freeze and official-template digest binding; Claude
trace/probe binding; broader primacy detection; exact citation-context and
per-entry trace binding; concrete protocol JSON-pointer targets; collaborator
manifest/commit binding for confirmed method evidence; and a truly external,
acyclic result receipt selected by CLI. Regression tests cover each bypass.

No fourth same-family acceptance review will be launched. Under the declared
`same_family_verdict_ceiling: provisional`, the contract remains
`contested-round3-human-tiebreak-required`. This record documents remediation,
not acceptance; empirical evidence, the official CVPR 2027 kit, cross-family
health, and a human tie-break remain independent blockers.
