# Research Contract: TempoRAC Supplied-Track Pilot

> **Focused active-idea contract derived from the official ARIS research-contract template.**
> **Prospective only: method refinement is READY at 9.125/10 under same-family review; P00 implementation is in progress; zero natural-data jobs, eligible results, or paper claims exist.**

## Selected Idea

- **Description:** TempoRAC studies person-wise repetitive-action counting from externally supplied pose tracks when different people move asynchronously and each person's tempo changes within a track. It keeps preprocessing and recurrent state private to each identity, shares one encoder and three matched slow/medium/fast response experts across identities, routes each local window with a detached irregular-clock NUDFT cue, reconstructs one response per track by positive normalized overlap-add, and invokes one fixed threshold-0.5 connected-component decoder exactly once per identity.
- **Source:** `refine-logs/temporac/FINAL_PROPOSAL.md`, derived from the original TempoRAC anchor in `idea-stage/RESEARCH_BRIEF.md` and adjudicated by `idea-stage/TEMPORAC_RESEARCH_REVIEW.md`.
- **Selection rationale:** It preserves the collaborators' original identity-indexed, window-local tempo-routing idea while turning its previously under-specified primitive-cycle, routing, NOLA, decoding, label-firewall, and evaluation interfaces into one executable and falsifiable contract. Five bounded refinement rounds ended with `READY`, weighted score `9.125/10`, and no remaining method blocker. This is same-family provisional review, not novelty clearance or a performance result.

## Core Claims

1. **Conditional primary claim:** test whether identity-local, window-local relative-tempo routing improves person-wise counting under deterministic within-track tempo drift over the strongest matched comparator while retaining clean-track accuracy. The claim is permitted only if every prerequisite gate passes and the frozen K7 point, paired-bootstrap, clean-retention, and primitive-unit conditions all pass.
2. **Conditional supporting mechanism claim:** test whether the shared slow/medium/fast experts specialize according to the frozen synthetic tempo strata and whether local soft routing, rather than capacity, a global or uniform gate, metadata shortcuts, identity leakage, window overlap, or decoder changes, explains any effect.
3. **Scope claim:** the current natural-data route is only a `GT-bbox-assisted AlphaPose supplied-track pilot` using a `partial-cache`. It may produce controlled pilot or appendix evidence only. It cannot support a main-paper result, abstract number, headline claim, predicted-track claim, official MultiRep claim, or deployment claim without a complete train/val pose cache and separate review and clearance.

TempoRAC does **not** claim first multi-person counting, first person-wise counting, first asynchronous or variable-speed counting, first pose-based counting, first self-supervised counting, end-to-end inference, annotation-free learning, constant-time scaling, or SOTA performance. Its precise supervision statement is that the counting objective uses no human per-person count, period, density, or boundary labels; external detector, tracker, and pose-estimator supervision must be disclosed.

## Method Summary

The input is a set of externally supplied, person-indexed COCO-17 pose tracks with source-frame clocks and separate joint/frame validity masks. A trusted packer deduplicates clocks, applies frozen root-scale geometry, separates feature-only archives from a permissioned evaluator vault, and enforces identity-private processing. The shared encoder produces per-edge representations. A detached, clock-aware NUDFT cue estimates a full local relative-tempo distribution on fixed 128-sample windows with stride 32; no hard period estimate or evaluator label enters the gradient path.

Three parameter-matched recurrent response experts receive the same encoded track representation. A soft router mixes their sigmoid event responses per window. Strictly positive normalized overlap-add reconstructs one full-track response without summing window counts. Quantization occurs once, after reconstruction, and one frozen threshold-0.5 connected-component pass produces each person's count. The scene total is only a derived diagnostic; the primary output is the person-wise count vector.

The primitive-event target is produced only when a separately trained synthetic teacher passes the frozen degree-one, topology, resampler, coverage, orientation, landmark, and one-event certificate. Natural pseudo-targets use the same certified graph. The matched training arms keep encoder, expert shapes, parameter count, optimizer, seeds, checkpoint selection, and decoder fixed; only the routed mechanism or its explicitly named control changes. The exact capability order is the proposal's F23 chain, and any gate failure terminates the route without retuning, filtering, fallback decoding, or replacement experiments.

## Experiment Design

- **Datasets:** synthetic X0 source orbits and frozen topology/operator/metric fixtures for validity; after separate acquisition authority, checksum-frozen v44 train/val feature archives only. The current cache contains 110 train videos / 51 val videos and 268 train tracks / 134 val tracks. Test, sealed, heldout, historical-result, and official-complete paths are forbidden.
- **Protocol labels:** every surrounding natural-data artifact must state `GT-bbox-assisted AlphaPose supplied-track pilot`, `partial-cache`, `GT-bbox-assisted`, and `supplied-track`. Closed receipt schemas remain byte-exact and do not receive ad-hoc keys.
- **Baselines:** exactly three families: same-response routing controls (`global NUDFT`, `uniform`, with `blocked` byte-identical to uniform and `shuffled` diagnostic only); matched `capacity-control`; and six shortcut adversaries (`no-pose-timestamp`, `nuisance-only`, `pose-shuffle`, `static-code-only`, `warp-metadata`, `track-length`). The strongest efficacy comparator is reselected within each bootstrap draw from global NUDFT, uniform, and capacity-control.
- **Metrics:** K7 uses person-level normalized absolute error aggregated video-first over three ordered seeds, paired component-cluster bootstrap confidence intervals, clean retention, and descriptive OBO. Validity blocks report certificate coverage, bitwise pulse/count equality, phase error, decoder boundary error, routing-specialization matrices, shortcut margins, and one pulse per evaluator interval with zero pulses outside the interval union. A desensitized AvgMAE/AvgOBO and bootstrap scorer fixture must pass before evaluator capability can exist.
- **Key hyperparameters:** seeds `20260815`, `20260816`, `20260817`; 128-sample windows, stride 32; three matched experts; sigmoid responses; positive NOLA; one threshold-0.5 connected-component decoder; fixed job steps and checkpoint selection from the canonical proposal.
- **Compute budget:** exactly 27 training jobs (3 teacher + 24 response), 93 allocated A6000-hours, maximum concurrency 2, 24 GiB CUDA cap per job, and 64 GiB total scoped artifact cap. The 7-hour outer margin cannot be reassigned automatically.

## Baselines

| Method family | Role | Dataset domain | Status | Source |
|---|---|---|---|---|
| Global NUDFT / uniform routing | Same-response efficacy controls | X0 plus future authorized supplied-track train/val | Not run | Frozen TempoRAC proposal and experiment plan |
| Matched capacity-control | Separates routing from parameter capacity | X0 plus future authorized supplied-track train/val | Not run | Frozen TempoRAC proposal and experiment plan |
| Six shortcut adversaries | Falsifies clock, mask, nuisance, metadata, and length shortcuts | Synthetic/desensitized plus future authorized feature-only train | Not run | Frozen TempoRAC proposal and experiment plan |

External published systems such as MultiCounter, MultiCounter+, PoseRAC, PAMS, RepNet, and TransRAC remain related-work or future protocol-matched baselines. They are not silently mixed into this controlled causal comparison, and no cross-protocol number is treated as a fair ranking.

## Current Results

> No efficacy, gate, training, prediction, or natural-data result exists yet. P00 implementation uses only contract bytes and synthetic/desensitized tests.

| Method | Dataset | Metric | Score | Notes |
|---|---|---|---|---|
| TempoRAC | Synthetic/desensitized P00 checks | Contract conformance | Not run to receipt completion | Implementation in progress; no scientific claim |
| TempoRAC | v44 train/val supplied-track pilot | K7 natural decision | Not run | Blocked behind P00--P3, S0, acquisition, firewall, teacher, response, and one-use evaluator gates |

## Key Decisions

- **Canonical graph:** shared encoder and shared experts with identity-private state; detached cue-only local routing; probability fusion before positive NOLA; one final decoder per identity.
- **Primitive-unit defense:** a response target is usable only after the frozen teacher and degree-one certificate establish one event per primitive traversal. Failure ends the route; integral, NMS, period-fit, alternate threshold, or post-failure repair is forbidden.
- **Evidence boundary:** the learner never deserializes count, period, density, boundary, bbox, object mapping, raw identity, or evaluator payloads. The evaluator vault becomes reachable only through one non-resumable G5b-to-K7 capability after all exact code and fixture hashes agree.
- **Current-data limitation:** the partial GT-bbox-assisted cache is permanently pilot/appendix-only within this protocol, even if K7 passes. Main-paper promotion needs a complete-cache protocol plus a new independent review.
- **Historical branches:** WARP-PHASE failed its selector route and remains historical; the pose-missingness direction is backup-only. Neither may replace TempoRAC or contribute results to it.
- **Assurance ceiling:** all semantic agents are `gpt-5.6-sol` as requested. Under official ARIS this is same-family review, so the maximum honest submission assurance remains `provisional` even if all deterministic checks pass.

## Status

- [x] Canonical idea selected and drift boundary frozen
- [x] Five-round method refinement completed (`READY`, 9.125/10, same-family provisional)
- [x] Experiment plan and tracker independently reviewed after two rounds (`PASS`)
- [ ] P00 package implementation and receipt complete (in progress)
- [ ] P01 static/import/firewall checks complete
- [ ] P2 synthetic/desensitized fixtures and metric receipt complete
- [ ] Fresh TempoRAC environment and resource preflight complete
- [ ] S0 commitment complete
- [ ] Separately authorized feature acquisition and label firewall complete
- [ ] Teacher and response training complete
- [ ] Natural pilot decision complete
- [ ] Complete-cache main-paper evidence cleared
- [ ] ICASSP paper and full ARIS paper audits complete
