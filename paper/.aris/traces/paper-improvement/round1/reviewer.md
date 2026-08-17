# ARIS Paper-Improvement Review — Round 1

Scope was restricted to the stated manuscript allowlist. No Git history, previous reviews, reports, contracts, implementation artifacts, or external evidence were consulted. No manuscript files were modified.

Visual evidence reviewed: :codex-file-citation{path="D:\Project\rac-paper-writing-icassp27\paper\main.pdf" purpose="source"} and :codex-file-citation{path="D:\Project\rac-paper-writing-icassp27\paper\figures\icassp_framework.pdf" purpose="source"}.

## Snapshot hashes

SHA-256:

| File | SHA-256 |
|---|---|
| `paper/main.tex` | `a4fe29c47f200946c7d6f3522bd273f5895aabda2d7a959fdbf16397fe1cbe70` |
| `paper/preamble.tex` | `763b456cb2e22585ca23160c2b12d641e764c70da94641247f6268af1a7306a2` |
| `paper/math_commands.tex` | `e6dabb7ce789e0d78026e6bda287fe17faedb6d20c0df9b4f603cb3b6c123bee` |
| `paper/sections/0_abstract.tex` | `9920f4ec88aa92908537b5160cf0f6fe6d459c00961248e58c09e9a7f0adee8c` |
| `paper/sections/1_introduction.tex` | `d8744390b04f15755211b21ed295970f7900943879787392d8efa86a178b1c10` |
| `paper/sections/3_method.tex` | `39d9314cee3265a206ee1b9da7967fc748ecd913ce37ec604ca44a46a2f1e7ca` |
| `paper/sections/4_experiments.tex` | `e631af4ea6df22549fea30d1aef76e39e83bb09f1ec34033dba5c43c470662ac` |
| `paper/sections/5_conclusion.tex` | `e5c656a98343f10b367c175273d572f3a42d5436539429f28ba87764452e7b3c` |
| `paper/references.bib` | `6be235be72cbcf358d0b80d19a646bd7c3869d344bb0657072822a36df62d885` |
| `paper/figures/icassp_framework.pdf` | `12d6d131843b7e5e766e5e1b2a3e4121ee62200cf11f6888ad6cd5f19cb470c7` |
| `paper/main.pdf` | `26eeb9e5b913fa8160276447e9c31b38c9fe147ef7f64666aed41144427ecd26` |
| `paper/VENUE_COMPLIANCE.md` | `d0db63bf5aa4c8fadcc65fafa311682800b1100d05fe96077a8a097552620c70` |
| `paper/TEMPLATE_PROVENANCE.md` | `b2e1506fa1d82448306ede74d40ab813e8aa3d725150a8a9e7251981912fd354` |

Rendered snapshot: five US-letter pages, PDF 1.7, all fonts embedded. The framework asset is a one-page vector PDF.

## Summary

The manuscript has a coherent, ICASSP-relevant signal-processing story: masked identity-indexed pose signals, window-local periodic evidence, soft temporal-scale routing, normalized overlap-add, and one decoder per track. It is exceptionally honest about missing implementation and results, never fabricates numbers, visually distinguishes pending content, and sets strong evidence/protocol gates. The principal weakness is therefore not the intentionally absent results: it is that the central trainable graph remains an interface contract rather than an implementable method. The signal projection, period/confidence estimator, expert specialization, losses, decoder, lifecycle semantics, and several preregistration decisions are unresolved. Novelty is also insufficiently positioned against relevant local time-frequency, missing-data spectral, adaptive-routing, and overlap-add literature. The current five-page render is clean, and page 5 is references-only, but the paper is not technically closed enough for Round 1 acceptance.

## Strengths

- The paper cleanly separates scene totals from the primary per-person count vector and distinguishes shared parameters from identity-local state.
- Equations expose the intended signal path instead of relying solely on architectural prose.
- Oracle-track, common-frontend, and native-pipeline protocols are explicitly separated; incompatible rows are not presented as a matched ranking.
- The evidence-control design is unusually strong: pending cells, abstract/result gates, structured result binding, checksums, paired video-cluster intervals, and source-bound ablations.
- The manuscript repeatedly and unambiguously states that there is no local multi-person implementation or eligible MultiRep result. No empirical advantage is implied.
- The draft correctly qualifies “without person-wise count or period labels” so it does not conceal external supervision in pose and tracking frontends.
- The figure caption and diagnostic shell explicitly deny being implementation or result evidence.
- The rendered paper has no visible clipping, overlap, broken glyphs, or unembedded fonts.

## Findings

### CRITICAL 1 — The core contribution is not yet an implementable or trainable algorithm

Anchors: `paper/sections/1_introduction.tex:52-54`; `paper/sections/3_method.tex:10-11`, `60-63`, `79-97`, `101-128`, `149-157`, `166-171`.

The paper leaves the feature-to-scalar projection, \(\Phi\), confidence definition, router aggregation, expert architectures, route/track losses, gradient paths, decoder, and fallbacks as synchronization contracts. In particular, it does not explain how the experts acquire fast/medium/slow specialization or how response peaks become counts without person-wise count/period labels. A soft mixture can collapse to one expert or permit experts to remain interchangeable; the current equations provide no identifiability, balancing, or specialization mechanism.

Concrete fix:

- Freeze the actual computational graph before strengthening any method claim.
- Specify tensor shapes, \(z\!\to s\), exact \(\Phi\), period units, router pooling, expert receptive fields and outputs, decoder/event rule, thresholds, training and inference algorithms.
- State every supervision and pseudo-target path, which losses train which modules, and where gradients are stopped.
- Define collapse controls or a falsifiable specialization test, including the mapping between expert scale and tempo ranges.
- Delete any router, expert, loss, or decoder component absent from the synchronized source.

### MAJOR 2 — The masked spectral estimator can confuse missingness with motion periodicity

Anchor: `paper/sections/3_method.tex:58-97`.

Multiplication by \(q_{i\ell}(t)\) windows the signal by the observation mask, so periodic occlusion or track gaps can generate Fourier peaks. Max-normalizing \(F\) makes its strongest frequency approximately one even for weak/noisy windows, which removes absolute evidence needed for confidence. The autocorrelation also uses one global masked mean while the valid pair set changes with lag. Boundary indexing, minimum pair support, sampling rate, frequency-to-period conversion, and the statistical meaning of \(\gamma\) are undefined.

Concrete fix:

- Define out-of-range mask behavior and a minimum valid-pair threshold for every lag.
- Use lag-pair-aware centering or justify the common mean.
- Use an irregular-sample spectral estimator or explicitly correct/test mask-spectrum leakage.
- Define \(\Omega\), sampling units, \(p(\omega)\), interpolation, harmonic selection, and confidence calibration.
- Add synthetic controls with periodic and nonperiodic missingness so routing cannot succeed by reading the mask pattern.

### MAJOR 3 — “Stable identity indices” are incompatible with the predicted-track protocol as currently specified

Anchors: `paper/sections/3_method.tex:4-15`, `47-56`; `paper/sections/4_experiments.tex:14-20`, `28-31`.

The method assumes stable identity indices, while common predicted tracks are explicitly evaluated with ID switches. It does not define what happens to buffers and counts after fragmentation, merges, switches, re-entry, or track retirement, nor how predicted count vectors are matched to ground-truth people. Reporting HOTA/IDF1 separately does not resolve the counting semantics.

Concrete fix:

- Define tracklet creation, buffer reset/continuation, termination, re-entry, fragmentation, merge, and ID-switch behavior.
- Define how per-track counts are associated with ground-truth identities and whether fragmented counts may be reconciled.
- State these rules separately for oracle, common-frontend, and native protocols.
- If the method cannot tolerate predicted identity discontinuity, narrow the claim and title to counting from stable supplied tracks.

### MAJOR 4 — The experiment section is a protocol skeleton, not yet a completed preregistration

Anchors: `paper/sections/4_experiments.tex:10-31`, `66-79`, `100-111`, `127-144`.

Several outcome-affecting decisions remain open: “at least” three seeds, conditional baseline inclusion, metric definitions/evaluator revision, the primary ablation metric and direction, warp schedule, and whether the diagnostic shows an example or aggregate. Calling the section preregistered is premature while these choices can still change after implementation or data inspection.

Concrete fix:

- Before executing runs, freeze the exact dataset release/checksum, split, evaluator, metric definitions, seed values, training budget, model-selection rule, and preprocessing.
- Register one primary outcome and the planned contrasts, bootstrap unit/replicates, multiplicity policy, and failure/exclusion rules.
- Freeze the warp generator, schedules/ranges, interpolation, clean-corrupt pairing, and visualization-selection rule.
- Until frozen, call this a “planned protocol” or “preregistration template.”

### MAJOR 5 — The baseline and ablation design does not yet isolate the claimed mechanism

Anchors: `paper/sections/4_experiments.tex:46-61`, `66-79`, `81-111`.

The matched block currently contains Track-PAMS, one global-tempo control, and the full method. That does not separate benefits from local periodic evidence, multiple receptive fields, learned routing, extra parameters, or reconstruction. The prose says removal of expert routing is tested, but Table 2 has no explicit single-expert/no-routing row; uniform mixing still uses all experts. \(\Delta\)MAE and \(\Delta\)OBO also lack a declared reference row and sign convention.

Concrete fix:

- Add matched local-ACF-only, local-FFT-only, fixed multi-scale/uniform, single-expert/no-router, hard-router, oracle-period upper-bound, and independent-window-decode controls as applicable.
- Match parameter count, temporal context, training budget, and frontend.
- Define every delta relative to a named reference and its direction.
- Make the table rows exactly match the mechanism claims in the prose.

### MAJOR 6 — Literature framing is too CV-counter-centric to establish the signal-processing novelty

Anchors: `paper/sections/1_introduction.tex:13-30`; `paper/references.bib:1-136`.

The manuscript cites repetition-counting and tracking work, but the proposed ingredients also sit directly in local time-frequency estimation, spectral analysis with missing samples, multirate/multi-scale processing, mixture routing, tapering, and overlap-add reconstruction. Without that framing, the assertion that prior work does not resolve the narrower signal problem is insufficiently supported, and the ICASSP novelty may look like a combination of established components.

Concrete fix:

- Add the closest signal-processing and adaptive-routing precedents.
- Explain precisely which prior assumptions fail for identity-indexed, nonstationary, masked trajectories.
- State the novel technical unit narrowly—ideally as a property or hypothesis—rather than novelty by component combination.
- Add corresponding matched classical/local spectral baselines where feasible.

### MINOR 7 — “Prevents duplicated events” is stronger than the reconstruction equation guarantees

Anchors: `paper/sections/0_abstract.tex:10-14`; `paper/sections/3_method.tex:130-164`.

Decode-after-overlap-add removes the structural duplication caused by independently decoding every window, but it cannot guarantee that the reconstructed response contains only one peak. Misaligned expert responses, zero taper coverage, or seam artifacts can still create duplicate detections.

Concrete fix:

- Replace “prevents duplicated events” with “avoids independently counting every overlapping window” or equivalent.
- State the sufficient coverage condition \(\sum_\ell a_{i\ell}(t)q_{i\ell}(t)>0\) for valid decoded samples and require globally aligned response timestamps.
- Use the planned boundary-offset test to support any stronger invariant.

### MINOR 8 — The framework figure is sharp but its mathematical notation is inconsistent and too small when embedded

Anchor: `paper/sections/3_method.tex:17-26`; asset `paper/figures/icassp_framework.pdf`.

The standalone asset is clean, but labels such as raw underscore-indexed variables and abbreviated parameters do not match the typeset equations. Several annotations become marginal at the scale used on page 3.

Concrete fix:

- Export figure labels using the paper’s mathematical notation, including \(i,\ell,k\), \(\mathcal E_\theta\), \(g_{i\ell k}\), and \(\hat{\mathbf c}\).
- Increase the minimum label size and simplify the training-contract strip.
- Retain the conspicuous proposal/synchronization qualifier.

### MINOR 9 — The central hypothesis should be stated more sharply in the title/story

Anchors: `paper/main.tex:13`; `paper/sections/1_introduction.tex:56-67`.

“Identity-indexed” accurately describes state separation, but much of that is required bookkeeping once stable tracks are supplied. The falsifiable research hypothesis is local scale routing plus reconstruct-before-decode under within-track tempo drift.

Concrete fix:

- State a result-free primary hypothesis, such as whether local routing improves robustness to within-track tempo changes relative to a matched global-tempo model while preserving stationary-track behavior.
- Consider emphasizing “local tempo routing” and “overlap-add reconstruction” over identity indexing in the title or subtitle.
- Keep identity handling as an explicit task/input assumption.

### MINOR 10 — Draft markers aid honesty but materially interrupt reading

Anchors: `paper/preamble.tex:26-33`, `74-100`; `paper/main.tex:15-29`.

The banner and red synchronization tokens are appropriate for this internal artifact, but pages 1–2 are visually dominated by status text, and the large author placeholder consumes substantial first-page space.

Concrete fix:

- Keep this draft mode for internal circulation.
- In the audited submission build, require the validators to eliminate every banner, placeholder, pending cell, and sync token rather than merely recoloring them.
- Recheck the four-page body and page-5 reference boundary after final authorship metadata is inserted.

## Page-by-page visual verdict

- **Page 1 — Pass with readability reservations.** Clean two-column composition, legible title and abstract, no clipping. The title/author placeholder block is tall, and the long red pending-evidence sentence plus synchronization marker interrupt reading. The core problem and proposal are nevertheless immediately understandable.
- **Page 2 — Pass with density reservations.** Equations (1)–(4) are legible and aligned. No overflow or broken math. The page is text-dense, and red synchronization clauses compete with the mathematical narrative.
- **Page 3 — Pass with figure-label reservation.** Figure 1 is sharp, well aligned, and visually communicates the signal path. Small raw-variable labels are harder to read than the surrounding typeset text. Section transitions and columns are otherwise clean.
- **Page 4 — Pass.** Both tables fit without collision, pending entries are unmistakable, and the diagnostic shell cannot be mistaken for data. The mixture of full-width table, technical body, declarations, and the start of references is visually busy but readable. All non-reference technical/declaration material remains on pages 1–4.
- **Page 5 — Pass; references-only compliant.** It contains only continuation entries [10]–[12]. No body text, declaration, caption, footnote, or result content appears. The page has substantial unused space, but this is not a 4+1 compliance violation.

## Scores

| Dimension | Score | Rationale |
|---|---:|---|
| Soundness | **2/5** | Coherent signal path, but training, specialization, decoder semantics, identity lifecycle, and missing-data spectral behavior are unresolved. |
| Significance | **3/5** | Important task and plausible mechanism; novelty and effect attribution remain conditional on stronger related work, controls, implementation, and results. |
| Clarity | **4/5** | Exceptionally clear about scope and absent evidence; causal story is easy to follow despite dense prose and status markers. |
| Relevance | **4/5** | Strong multidimensional-signal, autocorrelation/Fourier, masking, routing, and overlap-add content; signal-processing literature positioning needs expansion. |
| Reproducibility readiness | **3/5** | Excellent provenance/evidence architecture, but the algorithm and preregistration fields are not yet frozen enough to reproduce. |

## Overall verdict

**PROVISIONAL MAJOR REVISION / NOT SUBMISSION-READY.**

The absence of results is appropriately disclosed and is not itself misconduct or a drafting failure. However, the same-family review ceiling remains **provisional** until the implementation-bound method, complete preregistration, and audited MultiRep evidence exist. The current manuscript is a strong and honest research specification, not yet a complete ICASSP paper.

**ROUND1_ACCEPTED: NO**
