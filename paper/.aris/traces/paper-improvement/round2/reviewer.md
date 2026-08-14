# ARIS Paper-Improvement Round 2 Review

## Scope and snapshot

Fresh, zero-context, read-only review of the exact allowlist. No manuscript files were edited, and no prior reports, logs, history, implementation artifacts, or external evidence were consulted. The initial and final hashes matched.

Visual sources inspected: :codex-file-citation{path="D:\Project\rac-paper-writing-icassp27\paper\main.pdf" purpose="source"} and :codex-file-citation{path="D:\Project\rac-paper-writing-icassp27\paper\figures\icassp_framework.pdf" purpose="source"}.

| File | SHA-256 |
|---|---|
| `paper/main.tex` | `A4FE29C47F200946C7D6F3522BD273F5895AABDA2D7A959FDBF16397FE1CBE70` |
| `paper/preamble.tex` | `763B456CB2E22585CA23160C2B12D641E764C70DA94641247F6268AF1A7306A2` |
| `paper/math_commands.tex` | `E6DABB7CE789E0D78026E6BDA287FE17FAEDB6D20C0DF9B4F603CB3B6C123BEE` |
| `paper/sections/0_abstract.tex` | `03CA63E9CF29F323ACE7157244F49964CCB9A36A1987F4A7403BB8387FCF07A6` |
| `paper/sections/1_introduction.tex` | `9BAFE068B320BA13CF810F9D55FA9FE12FF85D0532DD06F440689E46F3350354` |
| `paper/sections/3_method.tex` | `CD2EF91C2DD69C89E2B514FDB25584073625CFB2A178EAC817DEDC86D7BBFF5B` |
| `paper/sections/4_experiments.tex` | `91F2FD272225A192BCE0DA9A535846A2088651C9CF93F74A9EC9F0882905A5CD` |
| `paper/sections/5_conclusion.tex` | `E5C656A98343F10B367C175273D572F3A42D5436539429F28BA87764452E7B3C` |
| `paper/references.bib` | `322A5D16B0078E8B3436608D4C6158B67C5C8922B0DCAE94537EA456C9F6F646` |
| `paper/figures/icassp_framework.pdf` | `EBF76174BBFA52E9FADA50F3BEBC2B56F07AAD91818ADBB9EBEF84E5249BC8A7` |
| `paper/main.pdf` | `0E521AD303B21FBB98D7C3EA52CB2AEC2C5376EA07400986BB1721B0CA3C9A17` |
| `paper/VENUE_COMPLIANCE.md` | `D0DB63BF5AA4C8FADCC65FAFA311682800B1100D05FE96077A8A097552620C70` |
| `paper/TEMPLATE_PROVENANCE.md` | `B2E1506FA1D82448306EDE74D40AB813E8AA3D725150A8A9E7251981912FD354` |

Automated consistency checks found 16 cited keys, 16 bibliography entries, no missing/unused bibliography keys, no unresolved cross-references, and no visible `??` tokens.

## Summary

This is an unusually honest pre-results draft. It clearly distinguishes proposal, hypothesis, planned controls, and unavailable evidence; it does not fabricate implementation details or results. Its ICASSP signal-processing relevance is credible through masked nonstationary signals, local ACF/Fourier evidence, temporal routing, and normalized overlap-add.

The draft is not yet Round-2 acceptable because the scientific object remains underdetermined at its core: routing/expert specialization, learning objective, decoder, and track-to-person semantics are not operational; the main experiment omits the promised oracle-track isolation; and the ablation does not directly test the central reconstruct-before-decode mechanism. These are independent of the intentionally pending results.

The same-family ceiling remains **provisional**.

## Strengths

- Strong honesty boundary: absent implementation and MultiRep evidence are disclosed repeatedly and conspicuously (`0_abstract.tex:14`; `1_introduction.tex:44-60`; `4_experiments.tex:4-8`; `5_conclusion.tex:8-14`).
- Clear primary task definition: person-wise vector output is distinguished from a diagnostic scene sum (`1_introduction.tex:7-11`; `3_method.tex:28-50`).
- Good supervision qualification: absence of person-wise count/period labels is limited to the counting objective, while pose/tracking supervision is disclosed (`0_abstract.tex:5-7`; `1_introduction.tex:41-46`; `3_method.tex:146-149`).
- Strong experimental hygiene plan: protocol separation, frozen manifests, checksums, seeds, paired video-cluster bootstrap intervals, multiplicity policy, and predeclared exclusion rules (`4_experiments.tex:10-35,108-121`).
- Useful missingness controls that distinguish tempo evidence from visibility leakage (`3_method.tex:79-88`; `4_experiments.tex:148-160`).
- Claims are appropriately qualified: no runtime, expert-specialization, result, or implementation claim is enabled prematurely.
- The compiled PDF has no clipping, overlap, broken glyphs, unresolved citations, or page-count violation.

## CRITICAL findings

None under the explicitly stated pre-results constraint. The lack of results is not treated as misconduct or a drafting defect. It does, however, cap the verdict at provisional and prevents empirical acceptance.

## MAJOR findings

1. **The proposed method is not yet a uniquely executable algorithm.**  
   Anchors: `paper/sections/3_method.tex:52-88,90-113,115-149`.  
   `\Phi`, the evidence projection, expert scale construction, router inputs, fallback, decoder, event calibration, and actual training objective are all open interfaces. The displayed PAMS-TCC plus route/track loss is explicitly only a synchronization contract. Consequently, materially different systems could satisfy the prose while exhibiting different behavior, and there is no defined mechanism by which fast/medium/slow experts acquire distinct specializations without period labels. The displayed ACF also uses one window mean while the prose later requires pair-dependent centering or another estimator (`3_method.tex:60-88`).  
   **Fix:** after source freeze, replace every core `SYNC-REQUIRED` interface with tensor shapes, time/frequency units, window/stride rules, exact `\Phi`, expert receptive fields or scale constraints, routing temperature and fallback, the actual losses/targets/gradients, decoder/event rule, and calibration. Add compact end-to-end pseudocode. Make the equation match the chosen irregular-sample centering.

2. **The promised person-wise output is unresolved for predicted tracklets.**  
   Anchors: `paper/sections/0_abstract.tex:1-7`; `paper/sections/1_introduction.tex:7-9`; `paper/sections/3_method.tex:4-13,28-50,129-130`; `paper/sections/4_experiments.tex:21-24`.  
   The paper promises a count per person, but correctly admits that a predicted identity index is only a current tracklet. No inference-time rule turns fragmented/re-entering/switched tracklets into persistent person counts. Ground-truth assignment for evaluation does not solve the deployed output semantics. Figure 1’s “each person’s count” language is therefore stronger than the predicted-track method currently supports.  
   **Fix:** either scope the main claim explicitly to supplied persistent identity tracks, or define and evaluate tracklet stitching, re-entry, merge/split, retirement, switch handling, and count reconciliation. Align the abstract, title, equation, figure caption, and metric unit with that choice.

3. **Novelty is plausible but not yet demonstrated at mechanism level.**  
   Anchors: `paper/sections/1_introduction.tex:13-36,62-73`.  
   The manuscript concedes that windowed spectral estimation, irregular-sample frequency analysis, overlap-add, and expert routing are established. Its novelty therefore rests on the exact combination and invariant, but the contrast with PAMS, MultiCounter+, TransRAC, and the track-global control remains prose-level. A reviewer can reasonably see this as a trackwise assembly of known components.  
   **Fix:** add a compact mechanism comparison covering identity unit, local versus track-global tempo, mask handling, routing granularity, response-before-decode reconstruction, supervision, and output semantics. State precisely which row-level combination is new and which experiment falsifies each claimed distinction.

4. **The planned ablation does not directly test the central reconstruct-before-decode claim.**  
   Anchors: `paper/sections/0_abstract.tex:10-14`; `paper/sections/3_method.tex:115-144`; `paper/sections/4_experiments.tex:85-106,108-121`.  
   “Soft + average” versus “soft + OLA” is ambiguous because normalized OLA is itself a weighted average. Both may still reconstruct before decoding, so the table does not isolate the stated structural source of boundary double-counting. Also, the full row’s deltas must be exactly zero by definition, not pending.  
   **Fix:** add a parameter-matched “decode each window, then aggregate counts/events” control; define “average” as a precise boxcar/unweighted fusion if retained; include an event-at-each-window-offset boundary test with one-time decode rate or count error; set the full-row deltas to `0`.

5. **The main table omits the oracle-track isolation promised by the protocol.**  
   Anchors: `paper/sections/3_method.tex:4-7`; `paper/sections/4_experiments.tex:10-24,37-68,70-83`; `paper/sections/5_conclusion.tex:11-12`.  
   Three protocols are declared, including oracle tracks, but Table 1 contains native/contextual and common predicted-track rows only. This leaves the core local-tempo hypothesis confounded by tracking quality in the primary comparison, despite the conclusion requiring matched oracle and common-track evaluation.  
   **Fix:** add a separate oracle-track block or table for Track-PAMS, track-global tempo, and local routing with identical pose inputs, training budget, seeds, and evaluator. Keep predicted-track results as a second robustness protocol.

6. **Evaluation units, metric semantics, and the tempo diagnostic remain too underspecified for a frozen protocol.**  
   Anchors: `paper/sections/4_experiments.tex:21-35,137-160`.  
   Period-mAP, AP50, AP75, AvgMAE, and AvgOBO are named but not defined in the manuscript package: the per-person/per-tracklet unit, aggregation order, empty-track handling, assignment cost, and fragmentation policy are unknown. The piecewise warp promises preserved source events but does not specify warped timestamps/annotations, boundary handling, duration changes, or a natural-data tempo-change stratum.  
   **Fix:** define or cite the evaluator formulas and aggregation unit; freeze matching and unmatched-track rules; state how warp transforms frames, masks, event timestamps, and evaluation support; predefine tempo-change severity bins and include a held-out natural nonstationarity analysis alongside the synthetic stress test.

## MINOR findings

7. **“Causal order” is ambiguous and likely misleading.**  
   Anchor: `paper/sections/1_introduction.tex:52-57`, contrasted with `paper/sections/3_method.tex:63-75`.  
   The method uses overlapping windows and future-shifted samples, so “causal” may be read as an online/zero-lookahead claim.  
   **Fix:** use “processing order” unless latency and causal window support are explicitly defined.

8. **Framework labels are marginally small and inconsistently typeset.**  
   Anchors: `paper/sections/3_method.tex:15-24`; `paper/figures/icassp_framework.pdf`.  
   Effective figure text reaches roughly 5.6–6.6 pt in the compiled page; pale magenta equations and literal underscore-style labels are difficult at print scale.  
   **Fix:** raise the effective minimum to about 7.5–8 pt, darken low-contrast text, and use consistent mathematical typesetting for subscripts/superscripts. Split the training-contract strip if necessary.

9. **Figure/table float order disrupts narrative order.**  
   Anchors: `paper/sections/4_experiments.tex:85-106,123-146`; compiled PDF page 3.  
   Figure 2 appears at the top of page 3 before Section 3.3 and well before Section 3.4 introduces it, while Table 2 is deferred to page 4.  
   **Fix:** move the diagnostic figure source after its subsection lead or adjust float placement so the first textual introduction precedes the figure.

10. **The title is accurate but jargon-heavy and visually long.**  
    Anchor: `paper/main.tex:13`; compiled PDF page 1.  
    **Fix:** consider “Local Tempo Routing for Identity-Aware Multi-Person Repetition Counting” or another shorter title that foregrounds the task before the implementation detail.

11. **The 4+1 layout is compliant but underuses available reference-page capacity.**  
    Anchors: `paper/main.tex:50-60`; compiled PDF pages 4–5.  
    References start midway down page 4, while page 5 uses only part of its left column and leaves the right column blank.  
    **Fix:** once extra definitions are added, push more or all references to page 5 and reclaim the page-4 right-column space for the executable method and metric definitions.

## Page-by-page visual verdict

| Page | Verdict |
|---|---|
| 1 | **Pass with density concern.** Clean two-column layout, no clipping, clear draft/result-pending status. Long title and red pending prose dominate the opening, but the task and hypothesis are understandable. |
| 2 | **Pass.** Equations are aligned and legible; no overflow. Dense red synchronization markers emphasize that the core method remains provisional. |
| 3 | **Pass with readability/order concerns.** Framework is vector-clean but its smallest labels are marginal at print size. Figure 2 floats before its subsection, and the ablation table is deferred. |
| 4 | **Pass.** Both tables are readable; conclusion, funding/COI, and ethics text remain on page 4. References begin in the right column with a visually clear transition. |
| 5 | **Strict rule pass.** Contains only continuation of references [9]–[16]; no technical prose, acknowledgments, declarations, header, or other non-reference content. Large unused space remains. |

## Venue/layout compliance

- Exact five-page, US-letter PDF.
- Page 5 is references-only.
- Funding, conflict, and ethics placeholders remain on page 4.
- Current build uses the stated ICASSP 2026 scaffold; `VENUE_COMPLIANCE.md:12-14` and `TEMPLATE_PROVENANCE.md:21-24,43-45` correctly keep 2027-kit, EDICS, and policy status provisional.
- Author roster, originality/cap attestations, official 2027 kit, and final declarations remain submission blockers. No claim of submission readiness is made.

## Scores

| Dimension | Score | Rationale |
|---|---:|---|
| Technical soundness | 3/5 | Coherent signal path and sensible safeguards, but core training/decoder/identity semantics are not operational. |
| Significance | 3/5 | Important failure mode and potentially useful invariant; incremental-combination risk remains. |
| Clarity | 4/5 | Exceptionally candid and logically organized; some jargon, density, and float-order issues. |
| ICASSP relevance | 4/5 | Strong masked-signal, spectral, temporal-routing, and reconstruction framing. |
| Reproducibility readiness | 2/5 | Excellent evidence-gating plan, but no frozen implementation, exact algorithm, evaluator semantics, or eligible results. |

## Overall verdict

**PROVISIONAL MAJOR REVISION / WEAK REJECT.**

The manuscript is honest, visually compliant, and scientifically promising, but resolving the six major findings is necessary before it can function as an assessable ICASSP paper. Pending results should remain pending; they must not be invented.

**ROUND2_ACCEPTED: NO**
