---
narrative_version: 2
created_at: 2026-08-14T20:38:01+08:00
venue: ICASSP 2027
track: regular conference paper
assurance: submission
paper_state: provisional/data-pending
result_route: pivot
cvpr_freeze_commit: 26e5b7176f1f7c678391163c3278b5c390d54115
repository_base_commit: 73cbc4bb9b581f24fbc90f2fe7421bc11e7c7454
aris_commit: e12e07c7b85ee1a4dc07e5463089aa16836af2bf
---

# ICASSP 2027 Narrative Report

**Working title:** *Identity-Indexed Local Tempo Routing for Multi-Person Repetition Counting*  
**Submission format:** four technical pages plus a fifth page containing only references, funding acknowledgments, and the Compliance with Ethical Standards statement  
**Review mode:** single-anonymous; complete author order and affiliations are required for submission but are not stored in this tracked report  
**Current outcome:** a complete pre-results writing contract, not a result-bearing submission

## 1. Evidence Policy and Derivation

This ICASSP narrative is derived from the frozen CVPR pre-results package at
commit `26e5b7176f1f7c678391163c3278b5c390d54115`; the earlier repository base is
`73cbc4bb9b581f24fbc90f2fe7421bc11e7c7454`. The derivation and source hashes
are recorded in `DERIVATION_PROVENANCE.json`. The original dirty workspace at
`D:/Project/rac` is read-only for this writing task. No experiment branch is
merged, no sealed UCFRep test105 prediction, evaluation, or metric exists, and nothing is
pushed.

Evidence is resolved in this strict order:

1. collaborator final code, clean frozen configuration, run artifacts, and audited results;
2. tracked code and audit documents at the repository evidence freeze;
3. committed PowerPoint files as `visual_reference_only` material;
4. informal prose, meeting notes, and summaries as narrative hints only.

The collaborator's multi-person implementation and MultiRep result bundle are
not present. Consequently, every proposed multi-person mechanism remains
`SYNC-REQUIRED`, and all empirical claims remain `data-pending`. A concept
figure can explain a proposed signal path; it cannot establish that the path
was implemented, trained, or validated.

ARIS is pinned to upstream `main@e12e07c7b85ee1a4dc07e5463089aa16836af2bf`
with `assurance=submission`. A fresh network clone did not yield a usable clean
checkout in this run, so the workflow uses the locally verified pin and records
the downgrade. This does not relax any evidence gate. Cross-family review is
also not credited unless a live Claude response and trace are obtained.

## 2. Verified Local State

The tracked repository implements an independent, single-person PAMS
reproduction, not the proposed multi-person repetition-counting system. Its
current input is one masked pose sequence, preprocessing selects a dominant
person rather than maintaining cross-person identities, and the public result
contains one scalar count. This boundary is established by `README.md`,
`docs/METHOD_SPEC.md`, `docs/PAPER_AUDIT.md`, and `docs/ASSUMPTIONS.md`.

The repository establishes only reusable starting ingredients:

- a pose Transformer encoder in `src/pams/model.py`;
- bounded single-sequence FFT/autocorrelation period estimators in
  `src/pams/period.py`;
- a PAMS-TCC starting objective and single-sequence inference consensus in
  `docs/METHOD_SPEC.md` and the tracked implementation;
- a separation between label-free prediction and the count-bearing evaluator;
- deterministic single-instance metrics and paired bootstrap utilities.

None of these artifacts proves a person axis, persistent identities,
identity-conditioned state, window-local routing, learned tempo experts,
normalized overlap-add over per-person responses, a trajectory-level decoder,
or MultiRep performance.

The available result tree is material negative evidence. It contains
single-person UCFRep dev84 failures, proxies, implementation diagnostics, and
official-checkpoint sanity runs under independent protocols. PAMS-Literal has
an untrained random Period Head under the disclosed training specification;
PAMS-SSHead is an independently inferred repair. These artifacts are retained
for audit history but are all `table_eligible=false` for the ICASSP paper. The
sealed test105 split remains unscored; an older development CLI deserialized its full manifest,
but no predictions, evaluation, or metrics exist, and no MultiRep result exists locally.

## 3. Problem and Signal-Processing View

The intended application is person-wise repetition monitoring in scenes where
several people exercise, rehabilitate, work, or train at different and changing
paces. A scene-level scalar can conceal who performed which repetitions and
cannot expose person-specific pauses or tempo changes. The primary target is
therefore a count vector indexed by persistent identities.

The ICASSP framing treats each identity-indexed pose trajectory as an irregular,
masked, multidimensional non-stationary signal. Two factors must be separated:
inter-person asynchrony and intra-person tempo drift. A single clip-level
period or one mixed temporal response cannot express both. The narrow technical
question is whether local periodic evidence can select an appropriate temporal
scale for every identity window while shared parameters retain statistical and
computational efficiency.

This framing deliberately does not claim the first multi-person, person-wise,
asynchronous, variable-speed, pose-based, or low-label repetition counter.
MultiCounter already defines multi-person repetitive action counting;
MultiCounter+ addresses identity continuity and long/short repetition regimes;
PAMS already studies pose-driven self-supervised period adaptation. These
works set the novelty boundary. The proposed paper must establish the narrower
intersection of identity indexing, window-local non-stationary tempo routing,
response reconstruction, and the precisely audited supervision setting.

## 4. One-Sentence Technical Story

Given supplied identity-indexed pose tracks, we propose to maintain separate
masked temporal state per identity while sharing an encoder and tempo experts,
estimate periodic evidence in overlapping local windows, softly route each
window by its changing pace, reconstruct one continuous response per identity,
and decode each trajectory exactly once into a person-wise count.

Every achieved-action verb in this sentence must remain future/proposal tense
until the collaborator artifact binds the mechanism to executable code.

## 5. Provisional Task Contract

For person \(i\in\{1,\ldots,N\}\), the input is a pose trajectory

\[
X_i\in\mathbb{R}^{T_i\times J\times C},\qquad
m_i\in\{0,1\}^{T_i}
\]

where \(m_i\) identifies valid observations. The primary output is

\[
\hat{\mathbf c}=[\hat c_1,\ldots,\hat c_N]^{\mathsf T}
\]

The scene sum \(\sum_i\hat c_i\) is a derived diagnostic and must never replace
the person-wise task definition. The experiment protocol must state whether
tracks are oracle, produced by one frozen common frontend, or generated by a
method's native pipeline. The paper evaluates the counter and reports tracking
quality separately whenever predicted tracks are used.

The intended counting objective uses no person-wise repetition-count or period
labels. This is a blocked supervision claim until the synchronized loaders,
losses, pseudo-labels, routing inputs, tuning, selection, early stopping,
calibration, and inference path all pass a source-level firewall audit. The
paper must not shorten this qualifier to `annotation-free`, `label-free`, or
`fully self-supervised pipeline`, because pose and tracking frontends may carry
external supervision.

## 6. Provisional Four-Equation-Group Method Contract

The ICASSP paper uses four compact equation groups. They define the writing
default; they are not implementation evidence.

### Group 1: identity-indexed masked representation

A shared encoder maps overlapping windows from every \(X_i\) to local features,
while masks, cached state, and outputs remain indexed by identity. Shared means
shared learned parameters, not separate models per person. Exact tensor shapes,
window length, stride, padding, short-track handling, and identity lifecycle are
`SYNC-REQUIRED`.

### Group 2: local masked ACF/FFT period evidence

Within window \(\ell\) of track \(i\), the draft computes a mask-normalized
non-circular autocorrelation over valid encoded samples, transforms it to a
bounded spectral score, and obtains a local period estimate
\(\hat p_{i\ell}\) and confidence \(\gamma_{i\ell}\). The exact signal used by
the estimator, mean removal, lag bounds, sampling units, spectral interpolation,
and low-confidence behavior must be replaced by the collaborator implementation.
No clip-level period is described as window-local evidence.

### Group 3: soft routing over shared tempo experts

The provisional router produces simplex weights
\(\mathbf g_{i\ell}\) over shared fast, medium, and slow experts from local
features, period evidence, and confidence. A fixed fallback distribution may
appear only as a visible `SYNC-REQUIRED` placeholder until code determines
whether it exists. Each expert emits a local continuous counting response; it
does not finalize or round a window count. Router inputs, temperature,
parameterization, gradients, expert definitions, and specialization tests are
all synchronization obligations.

### Group 4: response fusion, normalized overlap-add, and one decoder

Expert responses are mixed with the routing weights before overlapping windows
are reconstructed. The writing default uses a mask- and taper-normalized
overlap-add to form one response \(R_i\) per identity and then applies one
trajectory-level event decoder \(\mathcal D\):

\[
\hat c_i=\left|\mathcal D(R_i,m_i)\right|
\]

This order prevents the manuscript from summing independently rounded window
counts and double-counting overlap events. The taper, denominator, gap policy,
peak parameters, confidence, and event semantics remain `SYNC-REQUIRED`.

The training section may name only the following unresolved interface:

\[
\mathcal L=\mathcal L_{\mathrm{PAMS\text{-}TCC}}
+\lambda_{\mathrm r}\mathcal L_{\mathrm{route}}
+\lambda_{\mathrm t}\mathcal L_{\mathrm{track}}
\]

Only the local PAMS-TCC starting component is documented. The existence,
definitions, supervision sources, weights, sample construction, and gradient
paths of routing consistency and track continuity are `SYNC-REQUIRED`; they
must not be narrated as implemented facts.

## 7. Claim Ceiling Before Synchronization

Allowed now:

- a clearly marked task formulation and signal-processing motivation;
- provisional equations and an explicitly proposed framework;
- verified descriptions of the tracked single-person reproduction;
- primary-source-verified literature positioning;
- a preregistered experiment and artifact contract;
- explicit `DRAFT — RESULTS PENDING` and `SYNC-REQUIRED` markers.

Blocked now:

- any statement that this worktree implements or validates the multi-person method;
- any proposed-method number, abstract result, comparative gain, robustness,
  generalization, or efficiency conclusion;
- `first`, `annotation-free`, `end-to-end`, `constant-time`, or unmatched-protocol
  `state of the art`/`SOTA` wording;
- a claim that the current UCFRep results transfer to MultiRep or to the proposed method;
- a claim that a conceptual framework figure depicts measured behavior;
- `submission-ready: yes`.

## 8. Required Empirical Closure

The collaborator bundle must identify a unique clean commit, resolved
configuration and checksum, environment, hardware, seeds, raw logs, and the
complete training graph. It must bind the exact MultiRep release, official
split, video/annotation/split checksums, evaluator revision, preprocessing
fingerprint, per-person ground truth, per-person predictions, and track mode.

The principal counting metrics are Period-mAP, AP50, AP75, AvgMAE, and AvgOBO.
Predicted-track protocols additionally retain HOTA, IDF1, and IDSW as frontend
diagnostics. Reproduced learned methods require at least three preregistered
seeds, means, sample standard deviations, and paired video-cluster bootstrap
95% confidence intervals. Official single checkpoints may be contextual rows
but cannot support variance claims.

The compact ICASSP evidence set is:

1. one main table separating native/contextual MultiCounter and MultiCounter+
   from common-track Track-PAMS, a track-global-tempo control, and the proposed method;
2. one ablation table for local/global tempo, uniform/hard/soft routing, no
   expert routing, boundary reconstruction alternatives, and only those
   auxiliary losses that exist in the frozen code;
3. one piecewise time-warp diagnostic showing local period, routing
   probabilities or entropy, reconstructed response, and clean-to-corrupt change;
4. one compact framework figure that exposes the perception boundary, shared
   parameters, per-identity state, local routing, reconstruction, and decoding.

All tables and plots must be generated from frozen JSON, CSV, or Parquet files.
The sequence is `experiment-audit -> result-to-claim -> paper-claim-audit`.
Only after the first two gates pass may empirical prose or the abstract result
sentence be enabled.

## 9. ICASSP Writing Contract

The technical narrative occupies four pages: a 140–160-word abstract; an
approximately 500-word Introduction that synthesizes related work; an
approximately 900-word method centered on the four equation groups; an
approximately 700-word experiment and analysis section; and a 100–120-word
conclusion. The body contains one compact framework figure, one data-bound
tempo diagnostic, and two compact tables. Page five contains only references,
funding acknowledgments, and the required ethics statement. No appendix,
technical footnote, equation, figure, table, or new method discussion may be
placed on page five.

The official ICASSP 2027 author kit is not yet bound to this worktree. Any
earlier IEEE/ICASSP template is a visibly provisional scaffold. Submission
mode must fail if the official 2027 kit, complete author order and affiliations,
controlled empirical bindings, or mandatory ARIS audit artifacts are absent.

## 10. Current Assurance Outcome

The workflow operates under submission-level fail-closed rules, but the paper
state is `provisional/data-pending`. The current result-to-claim route is
`pivot`: write the complete method and evaluation contract while withholding
positive empirical conclusions. Same-family reviews may improve the draft but
cannot lift the status beyond provisional. A healthy, traced cross-family
review, synchronized collaborator evidence, official 2027 template, author
completion, and all deterministic gates are required before any final
submission-ready declaration.
