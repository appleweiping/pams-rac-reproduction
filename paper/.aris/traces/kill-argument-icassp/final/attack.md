# ARIS Kill Argument - Independent Attack

## Provisional attack verdict

**KILL / STRONG REJECT (very high confidence).** The current artifact is an honest pre-results design memorandum, not a completed scientific paper. Its candor removes any accusation of hidden or fabricated evidence; it does not supply the implementation, identifiability, supervision contract, executable method, or results needed to evaluate the paper's central hypothesis. No rebuttal consisting only of prose can cure the P1 findings. They require a new frozen system and a completed experiment package.

This is an attack-stage verdict only. It is not final adjudication and must remain provisional until an independent defense is heard.

## Severity scale

- **P1 - fatal now:** independently prevents scientific acceptance of the current artifact.
- **P2 - fundamental:** remains a major blocker even if numbers are inserted, unless the method or task is materially completed or redesigned.
- **P3 - validity:** makes the intended mechanism claim non-identifiable, unfalsifiable, or unfairly tested.
- **P4 - material:** weakens mathematical or statistical credibility but is not independently fatal once higher-priority defects are fixed.
- **P5 - readiness:** blocks submission readiness or evidentiary completeness without by itself refuting the idea.
- **P6 - precision:** localized terminology, exposition, or coverage defect.

## What the draft discloses honestly - and why that is not a defense

The PDF visibly labels itself `DRAFT - RESULTS PENDING`; every result cell is `PENDING`; Figure 2 is a `DATA-PENDING DIAGNOSTIC SHELL`; the introduction says the local worktree has no multi-person implementation; the method labels source-sensitive interfaces `SYNC-REQUIRED`; and the conclusion repeats that no eligible MultiRep results exist. The manifests agree. This is exemplary separation of proposal from evidence and should not be attacked as concealment.

The scientific consequence is nevertheless fatal. ICASSP evaluates a research contribution, not a plan for one. Honest absence of an implemented method and empirical evidence is still absence. The manuscript offers neither a theorem establishing the claimed robustness nor measurements establishing it. Its principal contribution is therefore not presently decidable.

Visual inspection of all five pages found no clipping, overlap, unreadable table, or broken figure. The body occupies pages 1-4 and page 5 contains references only. The kill case is scientific, not typographic.

## P1 findings - independently fatal

### P1.1 The proposed method does not exist in the reviewed artifact

The strongest evidence is the method manifest, not merely cautionary prose. The only tracked checkout is classified as a `single_person_dominant_pose_counter`, selects one dominant pose trajectory, accepts `PoseSequence[T,33,3]`, and returns one scalar `CountResult`. It explicitly has no multi-person identity state, no learned soft tempo router, no normalized overlap-add, and only fixed smoothing/peak configurations with hard voting. Its declared claim ceiling is that it **cannot evidence the proposed multi-person method**.

For the proposed submission method, the implementation commit is null. All eight central components - shared encoder, per-identity state, local masked period estimator, soft router, shared experts, response fusion, normalized overlap-add, and single decode - are `SYNC-REQUIRED`, with null source anchors and empty tests. The task contract, fallback, and training objective are also unbound. Consequently, Figure 1 and Eqs. (1)-(4) describe a possible architecture family, not an algorithm that can be run or reproduced.

This is not a missing implementation appendix. It is an explicit mismatch between the only available executable system and every claimed technical component.

### P1.2 There is zero eligible evidence for feasibility, accuracy, robustness, or mechanism

The results manifest is `data-pending`, unfrozen, and table-ineligible. Its artifact ID, method commit, MultiRep release, split, checksum, common frontend, evaluator, evaluation type, seeds, hardware, configurations, logs, predictions, source artifacts, and metrics are empty or null. Every eligibility gate is false, including method binding, dataset binding, protocol binding, three seeds, uncertainty, raw-artifact binding, and audits.

The PDF reflects this exactly: all 40 main-comparison cells are pending; all non-reference ablation deltas are pending; HOTA/IDF1/ID-switch values are pending; Figure 2 contains no data; and the abstract withholds its result sentence. There is no pilot showing that the network trains, no boundary test showing that overlap-add reduces duplicate events, no routing trace showing expert specialization, no count result, no natural-tempo result, and no failure analysis.

The hypothesis that local routing is more robust than global tempo while preserving stationary behavior is therefore supported by neither proof nor observation. A full empirical signal-processing paper cannot be accepted on a future-work protocol alone.

### P1.3 The mathematical specification is non-executable at every decision-bearing interface

The apparent precision of four equations does not determine a model:

- The scalar evidence signal $s_{i\ell}(t)$ is only "provisionally projected" from a learned multidimensional feature. The projection, normalization, temporal alignment, and validity propagation are unspecified.
- The period/confidence map \(\Phi\), lag and frequency domains, units, interpolation, minimum support, confidence semantics, and low-support fallback are unspecified. \(\Phi\) can contain essentially the whole algorithm.
- Router inputs, temperature, architectures, expert receptive fields, gradient paths, specialization constraints, and fallback are unspecified.
- A "continuous response" has no defined target, scale, phase, or calibration. Mixing such responses is meaningless unless experts share a time and amplitude semantics.
- Tapers, gap handling, timestamp mapping, thresholding, nonmaximum suppression, event confidence, and decoder semantics are unspecified. One invocation of an unspecified decoder does not define one count behavior.
- The only displayed objective is explicitly a synchronization contract. The route and track losses may not exist; their formulas, weights, samples, supervision, and gradients are unknown and will be deleted if absent.

These are not tunable hyperparameters around a fixed method. They decide what is learned, what "tempo" means, what a response peak means, and how a count is produced. Many mutually incompatible systems satisfy the prose. Reproduction and even internal falsification are impossible from the current artifact.

## P2 findings - fundamental scientific blockers

### P2.1 "Person-wise" semantics collapse under the supplied-track contract

The paper's primary output is advertised as a count per person, but its actual primitive is a supplied index. With oracle persistent tracks, identity and temporal reconciliation have already been solved by privileged ground truth. With predicted tracks, the manuscript correctly admits that an index denotes a current tracklet, not a guaranteed person. A fragmented or re-entering person yields several indices and several partial counts; an ID switch merges evidence from different people; merge, retirement, and reconciliation can duplicate or erase counts. HOTA, IDF1, and switch counts diagnose tracking quality but do not turn tracklet counts into person counts.

The lifecycle and assignment contract that could close this semantic gap is entirely `SYNC-REQUIRED`. Until it exists, the output vector changes meaning across oracle and predicted protocols: ground-truth-person counts in one block, un-reconciled tracklet counts in another. The title and abstract's "multi-person" claim is therefore not supported by a stable estimand.

More fundamentally, supplied persistent tracks assume away the multi-person identity problem, after which the method processes each identity independently and forbids any cross-person window. Shared parameters are ordinary batching/weight sharing. The claimed multi-person formulation can reduce to running a single-person local counter $N$ times. The paper must show a scientific contribution beyond bookkeeping over oracle-separated tracks; it currently does not.

### P2.2 The novelty claim is a composition claim without a demonstrated new principle

The paper itself concedes that windowed spectral estimation, irregular-sampling frequency analysis, overlap-add, and learned expert routing are established. Its closest cited literature already covers multi-person outputs, identity continuity, pose-driven counting, label-light objectives, multiscale temporal correlation, and period-adaptive multiscale learning. What remains is the ordering "local tempo cue -> soft expert mixture -> overlap-add -> one decode" applied independently to supplied tracks.

No new estimator is derived; \(\Phi\) is unspecified. No new routing objective is defined. No reconstruction theorem applies to the context-dependent nonlinear expert responses. No new identity model is present. No empirical comparison shows that existing period-adaptive approaches fail specifically on within-track tempo changes. The asserted niche - several pace changes within one supplied trajectory - is a problem slice, not by itself an algorithmic novelty contribution.

Because the proposed composition is unimplemented and unmeasured, the paper cannot demonstrate that the combination yields a non-obvious interaction rather than standard sliding-window adaptation plus mixture-of-experts plus averaging. The novelty case is therefore assertion-level.

### P2.3 The claimed supervision boundary is unauditable and the proposed training signal is unknown

The abstract says the method targets counting without person-wise count or period labels, and the prose narrows this to an intended counting objective. The manifest, however, marks every relevant firewall scope `SYNC-REQUIRED`: loaders, pseudo-labels, router targets, auxiliary losses, hyperparameter tuning, checkpoint selection, early stopping, calibration, inference, pose, detection, tracking, and evaluation. It lists no external supervision sources and defines no gradient paths.

This matters scientifically, not just administratively. The router is supposed to learn fast/medium/slow specialization; the response experts must learn an event-like amplitude and phase; and the decoder must calibrate peaks to counts. PAMS-TCC is named as a starting point, but the paper does not show how that representation objective identifies these new components without period, event, or count information. Count-label-free training can be invalidated by supervised pseudo-target construction, model selection on labeled counts, threshold calibration, or period-band tuning even when the nominal loss contains no count label.

Acknowledging external pose/tracker supervision is necessary but insufficient. Until the entire learning and selection path is frozen, the low-label property is not a scientific property of any existing system.

## P3 findings - validity and falsifiability failures

### P3.1 The core robustness hypothesis has no frozen estimand or success criterion

"More robust to within-track tempo changes" and "preserving behavior on stationary tracks" are not operationalized. The paper does not predeclare a primary metric, tempo-change severity distribution, effect threshold, equivalence/noninferiority margin for stationary tracks, or minimum acceptable natural-data result. The piecewise warp generator, segment boundaries, interpolation, duration, support, bins, exclusions, and selection rule are all future decisions. Those choices can materially determine whether the method appears robust.

The promised natural nonstationarity pairing is also undefined. Without a frozen identification rule, difficult natural clips can be included or excluded after model behavior is known. The hypothesis is linguistically falsifiable but experimentally unconstrained.

### P3.2 The planned ablation does not isolate the claimed mechanism

The reference is local/ACF+FFT/soft/NOLA-to-decode, but the only explicit `global` row simultaneously changes local tempo to global **and soft routing to uniform routing**. Its delta therefore cannot be attributed to local versus global estimation. The main-table `Track-global tempo` baseline is also undefined, so it does not repair the mechanism contrast.

The router receives the full representation \(z\) in addition to \(\hat p\) and \(\gamma\), and every expert receives the same \(z\). It can ignore the explicit period evidence, while gains can arise from added capacity or from latent tempo encoded directly in \(z\). Conversely, \(\Phi\) can perform most of the useful inference. ACF-only/FFT-only and hard/uniform rows do not establish causal use of the claimed cue. Missing controls include cue shuffling or blocking, a soft global router matched in all other respects, capacity-matched non-routing mixtures, and source-bound tests of expert specialization/collapse.

Accordingly, even favorable future numbers under the displayed table would not identify "identity-local tempo routing" as the cause.

### P3.3 Protocol fairness is promised but not instantiated

Separating native/contextual, oracle-track, and common-predicted-track blocks is correct bookkeeping, but none of the matched rows presently exists. The strongest native methods consume video and own their detection/tracking/context pipeline; the proposed system consumes externally supervised pose tracks, and its oracle block receives privileged persistent identities. Adapting RepNet, TransRAC, PoseRAC, PAMS, or video-based multi-person counters to a common pose-track frontend can remove modalities, alter temporal context, or require a new decoder. "Parameter matched" and "same training budget" do not make such adapted systems scientifically equivalent.

The proposed Track-PAMS and Track-global-tempo baselines have no commit, training definition, or parity proof. The release, split, evaluator, evaluation unit, predicted-to-ground-truth assignment cost, unmatched rule, aggregation, empty support, and zero-count behavior are null. Thus the table does not yet define the population or quantity being compared. Native rows cannot support ranking, while the supposedly fair blocks have no executable baselines or metrics.

### P3.4 The reconstruction language overstates what normalized averaging guarantees

Classical overlap-add reconstruction relies on compatible windowed representations and window conditions. Here, local responses are outputs of context-dependent nonlinear experts with window-dependent router weights. Two overlapping windows can assign different phase, amplitude, or even event multiplicity to the same time. Dividing their weighted sum by coverage merely averages inconsistent predictions; it does not reconstruct a unique latent response or guarantee boundary safety.

The paper cautiously says the operation does not guarantee one peak, yet it still presents reconstruct-before-decode as removing a structural source of boundary double-counting. A mixture can create two peaks near a boundary, erase a true peak by destructive mismatch, or shift it across the decoder threshold. Since the taper and decoder are unspecified and the promised boundary test is absent, even this limited structural claim is presently unsupported.

## P4 findings - material methodological weaknesses

### P4.1 Scalar local tempo is not identifiable for the stated signal class

A multi-joint repetitive action can contain harmonics, alternating submotions, pauses, asymmetric cycles, and joint-specific visibility. Projecting it to one learned scalar creates half/double-period ambiguity, while a local window must trade temporal localization against enough cycles to resolve slow tempo. The draft freezes neither minimum cycles nor the relationship among window length, lag range, expert bands, and target periods.

The single frame-level mask \(m_i(t)\) also does not represent joint-specific confidence or partial pose validity. Periodic occlusion can create spectral peaks; global masked centering can bias lag-dependent autocorrelation; and the paper itself concedes that pair-dependent centering or another irregular-sample estimator may be required. Planned leakage controls are sensible but nonexistent, so the central cue has no demonstrated identifiability under the motivating missing-data regime.

### P4.2 The statistical plan is incomplete even as a preregistration

At least three seeds, sample standard deviations, and video-cluster bootstrap intervals are a useful minimum plan, but the primary endpoint, contrast direction, bootstrap replicates, multiplicity policy, failures, and exclusions remain future decisions by the paper's own admission. There is no sample-size or power rationale, no defined natural-nonstationarity cohort, and no equivalence analysis for the stationary-preservation clause. Three seeds alone cannot rescue a design whose estimand and comparison unit are unresolved.

## P5 findings - submission and evidence readiness

The author roster, affiliations, funding, conflicts, and ethical-compliance statement are visibly pending. The draft uses the 2026 style only as a temporary scaffold because the official 2027 kit is unavailable. These are not the scientific reason for rejection and are appropriately visible in a pre-results artifact, but the current PDF is not submission-eligible. The manifest independently marks submission eligibility false.

## P6 findings - precision and terminology

- `person`, `persistent identity track`, and `current tracklet` are used around one output notation even though they denote different evaluation units.
- `state` in Figure 1 and prose is not defined mathematically; Eq. (1) shows a window encoder but no persistent temporal-state update.
- `fast`, `medium`, and `slow` are interpretive labels with no units or band boundaries and are explicitly not observed specializations.
- "Reconstruction invariant" is invoked without a stated invariant or assumptions under which it holds.

These defects are secondary to the absent method and evidence, but they reveal that central terms currently name intentions rather than verifiable objects.

## Strongest kill chain

1. The advertised contribution is a person-wise multi-person counter with local tempo routing, normalized reconstruction, and no person-wise count/period labels.
2. The only tracked executable artifact is a single-person dominant-pose scalar counter lacking every advertised component.
3. The proposed components, objective, supervision path, lifecycle semantics, decoder, source anchors, and tests are all null or `SYNC-REQUIRED`.
4. Under predicted tracks, the output is a tracklet count rather than a stable person count; under oracle tracks, identity has been supplied by privileged ground truth and the model decomposes into independent single-person runs.
5. The novelty rests on arranging admitted prior ingredients, with neither a new formal result nor evidence of non-obvious benefit.
6. The displayed validation plan does not yet define a frozen estimand and confounds the central local-versus-global mechanism contrast.
7. The result package contains no release, split, evaluator, seed, prediction, metric, uncertainty, or audit; every empirical cell is pending.

Therefore there is presently no executable scientific object and no evidence with which to accept, falsify, or reproduce the claimed contribution. The correct attack-stage outcome is **KILL / STRONG REJECT**.
