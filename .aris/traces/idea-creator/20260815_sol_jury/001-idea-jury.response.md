{
  "reviewer_model": "gpt-5.6-sol",
  "review_independence": "same-family",
  "acceptance_status": "provisional",
  "ranked_candidates": [
    {
      "rank": 1,
      "dedup_key": "warp-equivariant-private-phase-flow",
      "scientific_novelty": 7.4,
      "protocol_defensibility": 7.6,
      "implementation_feasibility": 7.4,
      "icassp_fit": 8.7,
      "overall_10": 7.9,
      "primary_strength": "Directly tests label-free, identity-private local phase learning with a medium-sized implementation and explicit shortcut and collapse controls.",
      "kill_risk": "Synthetic-warp equivariance may encode interpolation or timestamp artifacts rather than semantic repetition; harmonic phase is not identifiable without strong controls."
    },
    {
      "rank": 2,
      "dedup_key": "masked-geometric-tssm-ridge-completion",
      "scientific_novelty": 5.9,
      "protocol_defensibility": 7.9,
      "implementation_feasibility": 8.6,
      "icassp_fit": 7.9,
      "overall_10": 7.5,
      "primary_strength": "The cheapest credible way to establish a label-free local-response backbone and obtain an early empirical kill decision.",
      "kill_risk": "Temporal self-similarity is heavily occupied by RepNet, JTSPS-Net, TransRAC, and related work; the network may merely complete smooth matrices."
    },
    {
      "rank": 3,
      "dedup_key": "identity-private-state-isolation",
      "scientific_novelty": 5.4,
      "protocol_defensibility": 8.7,
      "implementation_feasibility": 7.3,
      "icassp_fit": 8.3,
      "overall_10": 7.3,
      "primary_strength": "Provides the cleanest causal diagnostic for whether per-identity state prevents cross-person tempo interference.",
      "kill_risk": "Private recurrent state is not independently novel and may only memorize track length, visibility, or confidence; it requires an implemented base response learner."
    },
    {
      "rank": 4,
      "dedup_key": "artifact-controlled-synthetic-tempo-routing",
      "scientific_novelty": 6.2,
      "protocol_defensibility": 7.3,
      "implementation_feasibility": 7.4,
      "icassp_fit": 8.2,
      "overall_10": 7.2,
      "primary_strength": "Offers a direct, falsifiable specialization test for soft tempo routing with modest architectural complexity.",
      "kill_risk": "It sits close to PAMS, generic mixture-of-experts, MultiCounter+, and TWCRAC, while resampling artifacts can make the routing result scientifically vacuous."
    },
    {
      "rank": 5,
      "dedup_key": "masked-oscillator-residual-bottleneck",
      "scientific_novelty": 6.3,
      "protocol_defensibility": 7.4,
      "implementation_feasibility": 7.1,
      "icassp_fit": 8.0,
      "overall_10": 7.1,
      "primary_strength": "Its residual-only and shuffled-motion controls can determine whether a periodic branch adds real predictive information.",
      "kill_risk": "Human actions are not clean oscillators, and residual capacity is difficult to tune without either bypassing or crippling the periodic branch."
    },
    {
      "rank": 6,
      "dedup_key": "masked_chirplet_ola",
      "scientific_novelty": 7.1,
      "protocol_defensibility": 7.5,
      "implementation_feasibility": 5.4,
      "icassp_fit": 8.7,
      "overall_10": 7.0,
      "primary_strength": "Explicit within-window chirp rate is a strong signal-processing hypothesis for acceleration and deceleration.",
      "kill_risk": "The 4-6 week complex-valued implementation is disproportionate before data and evaluators are frozen, and chirp slope may follow camera or limb harmonics."
    },
    {
      "rank": 7,
      "dedup_key": "dual-track-protocol-count-gap",
      "scientific_novelty": 4.0,
      "protocol_defensibility": 9.2,
      "implementation_feasibility": 6.2,
      "icassp_fit": 7.5,
      "overall_10": 6.9,
      "primary_strength": "It is mandatory evidence infrastructure for separating counting error from tracker and identity-association error.",
      "kill_risk": "It is not a sufficient method contribution and currently lacks an authorized predicted-track release, frozen tracker, and complete association artifacts."
    },
    {
      "rank": 8,
      "dedup_key": "overlap_phase_sync_graph",
      "scientific_novelty": 5.7,
      "protocol_defensibility": 8.0,
      "implementation_feasibility": 5.9,
      "icassp_fit": 8.0,
      "overall_10": 6.7,
      "primary_strength": "Isolates seam errors with clear grid-shift and phase-residual diagnostics and relatively light computation once complex responses exist.",
      "kill_risk": "No current backbone emits identifiable complex phasors, and phase synchronization plus overlap-add is established machinery rather than standalone novelty."
    },
    {
      "rank": 9,
      "dedup_key": "fragmentation-equivariant-stateful-response",
      "scientific_novelty": 6.0,
      "protocol_defensibility": 8.0,
      "implementation_feasibility": 5.4,
      "icassp_fit": 7.7,
      "overall_10": 6.6,
      "primary_strength": "Connects label-free response learning to a genuinely multi-person failure mode with measurable clean-to-fragmented drift.",
      "kill_risk": "It depends on a missing base model and predicted-track artifacts, while synthetic fragmentation may poorly approximate tracker failures."
    },
    {
      "rank": 10,
      "dedup_key": "irregular_lomb_phase",
      "scientific_novelty": 6.5,
      "protocol_defensibility": 7.1,
      "implementation_feasibility": 5.2,
      "icassp_fit": 7.8,
      "overall_10": 6.4,
      "primary_strength": "Treats timestamps, missingness, and uncertainty as part of the signal model rather than hiding them behind interpolation.",
      "kill_risk": "Exact timestamps and raw confidence masks may be unavailable, and batched covariance-aware solves can be unstable and harmonic-sensitive."
    },
    {
      "rank": 11,
      "dedup_key": "mass-conserving-person-count-ledger",
      "scientific_novelty": 4.6,
      "protocol_defensibility": 8.3,
      "implementation_feasibility": 5.5,
      "icassp_fit": 7.1,
      "overall_10": 6.3,
      "primary_strength": "Creates a useful accounting invariant for one vector entry per identity and exposes unresolved mass instead of hiding reassignment errors.",
      "kill_risk": "It is principally bookkeeping, depends on missing response and lineage representations, and can conserve mass while attributing it to the wrong person."
    },
    {
      "rank": 12,
      "dedup_key": "masked_multitaper_uncertainty",
      "scientific_novelty": 5.5,
      "protocol_defensibility": 7.3,
      "implementation_feasibility": 5.0,
      "icassp_fit": 7.8,
      "overall_10": 6.1,
      "primary_strength": "Provides a classical, interpretable test of spectral leakage and estimator dispersion in short masked windows.",
      "kill_risk": "Masked taper re-orthogonalization and repeated expert evaluation are expensive, while taper dispersion may not predict model or count error."
    },
    {
      "rank": 13,
      "dedup_key": "identity_oscillator_filter",
      "scientific_novelty": 7.0,
      "protocol_defensibility": 6.8,
      "implementation_feasibility": 3.9,
      "icassp_fit": 8.1,
      "overall_10": 6.0,
      "primary_strength": "An explicit identity-private phase posterior offers a principled representation of tempo, pauses, uncertainty, and missing observations.",
      "kill_risk": "Differentiable switching filters are a long sequential implementation with severe half-frequency, double-frequency, and calibration failure modes."
    },
    {
      "rank": 14,
      "dedup_key": "variational_monotone_phase_clock",
      "scientific_novelty": 7.2,
      "protocol_defensibility": 6.5,
      "implementation_feasibility": 3.3,
      "icassp_fit": 8.2,
      "overall_10": 5.9,
      "primary_strength": "The monotone phase coordinate is scientifically elegant and directly exposes instantaneous tempo and boundary uncertainty.",
      "kill_risk": "Spline inference, harmonic reconstruction, posterior approximation, and decoding form an overlarge first build with multiple unidentifiable failure points."
    },
    {
      "rank": 15,
      "dedup_key": "pose-phase-reentry-cache",
      "scientific_novelty": 4.7,
      "protocol_defensibility": 6.8,
      "implementation_feasibility": 4.6,
      "icassp_fit": 6.6,
      "overall_10": 5.6,
      "primary_strength": "Targets fragmentation-induced duplicate outputs and phase resets at the per-person vector level.",
      "kill_risk": "It is close to generic dormant-track re-identification, lacks crowded re-entry data, and false merges can be worse than conservative fragmentation."
    },
    {
      "rank": 16,
      "dedup_key": "ragged-shared-expert-scaling",
      "scientific_novelty": 3.4,
      "protocol_defensibility": 8.4,
      "implementation_feasibility": 4.1,
      "icassp_fit": 5.8,
      "overall_10": 5.3,
      "primary_strength": "Could provide honest person-count scaling and numerical-equivalence evidence after a method exists.",
      "kill_risk": "There is no canonical implementation to optimize, and batching or cache reuse is a systems result rather than the required scientific contribution."
    },
    {
      "rank": 17,
      "dedup_key": "ambiguous-switch-state-hypotheses",
      "scientific_novelty": 4.2,
      "protocol_defensibility": 6.6,
      "implementation_feasibility": 3.5,
      "icassp_fit": 6.2,
      "overall_10": 5.0,
      "primary_strength": "Correctly recognizes that an ID switch can preserve the scene total while corrupting the per-person count vector.",
      "kill_risk": "Multiple-hypothesis tracking is established, branching is combinatorial, association logits and crossing-heavy data are absent, and this does not solve the core counting representation."
    }
  ],
  "selected_pilots": [
    {
      "role": "primary",
      "dedup_key": "warp-equivariant-private-phase-flow",
      "minimal_build": "From scratch on supplied tracks: a shared pose encoder, one identity-private GRU, 64-frame overlapping windows, unit-circle phase and periodicity outputs, known monotone piecewise-warp equivariance, capacity-limited pose-velocity reconstruction, normalized overlap-add, and one decode per identity. Use three independent seeds for the decision-bearing pilot.",
      "required_data": [
        "Checksum-frozen, canonical-source-disjoint MultiRep train and development partitions",
        "Authorized per-person pose tracks with timestamps and visibility masks",
        "Count and period annotations accessible only inside the evaluator",
        "Frozen Period-mAP, AP50, AP75, AvgMAE, AvgOBO, and paired video-cluster bootstrap implementations"
      ],
      "preregistered_success": [
        "Median correspondence phase error no greater than 0.15 cycles on unseen warp schedules",
        "Phase-feature effective rank at least 8 and median temporal phase variance above 0.05",
        "At least 5% relative AvgMAE improvement or 2.0 absolute Period-mAP improvement over track-global autocorrelation",
        "Piecewise-warp AvgMAE degradation no greater than 10% relative to clean development performance",
        "The primary paired source-cluster bootstrap interval supports the selected downstream improvement"
      ],
      "preregistered_failure": [
        "Phase or response standard deviation below 0.02 on more than 20% of active tracks",
        "A no-pose timestamp-only model retains more than 80% of downstream performance",
        "Neither downstream improvement threshold is met",
        "Clean-to-warp degradation exceeds 20% or results are not consistent across independent seeds"
      ]
    },
    {
      "role": "secondary",
      "dedup_key": "masked-geometric-tssm-ridge-completion",
      "minimal_build": "Independently implement 32- and 64-frame masked off-diagonal temporal self-similarity completion using three fixed geometric pose views, a small shared completion encoder, deterministic ridge-to-response decoding, identity-private response buffers, normalized overlap-add, and one final per-track decode.",
      "required_data": [
        "The same frozen source-disjoint train and development manifests used by the primary pilot",
        "Pose confidence and missing-joint masks",
        "Evaluator-only person counts and period intervals",
        "Raw geometric autocorrelation, interpolation, row-mean, and unmasked-autoencoder controls"
      ],
      "preregistered_success": [
        "Masked off-diagonal correlation exceeds the no-temporal-context control by at least 0.10",
        "Dominant-frequency disagreement across geometric views is no greater than 15% on periodic development tracks",
        "Response standard deviation exceeds 0.03 on at least 90% of active tracks and completed-matrix effective rank is at least 6",
        "At least 5% relative AvgMAE or 2.0 absolute Period-mAP improvement over fixed geometric autocorrelation"
      ],
      "preregistered_failure": [
        "Constant, row-mean, or interpolation completion comes within 2% of the learned completion loss",
        "Frame-order shuffling retains more than 70% of Period-mAP",
        "Collapse gates fail or neither downstream improvement threshold is reached",
        "Any apparent gain requires test-set parameter selection"
      ]
    },
    {
      "role": "control",
      "dedup_key": "identity-private-state-isolation",
      "minimal_build": "Using the frozen primary backbone, compare identical private-state, stateless-window, and shared-scene-state variants. Warp exactly one supplied identity at a time while preserving every other trajectory, then measure changes in untouched identities and reset or swap state as negative controls.",
      "required_data": [
        "Stable supplied-track person identifiers on the frozen development split",
        "The implemented label-free primary response path",
        "Deterministic per-person piecewise time-warps",
        "Per-person predictions, routing histories, state-reset logs, and paired source-cluster bootstrap support"
      ],
      "preregistered_success": [
        "At least 8% relative AvgMAE reduction versus shared scene-state on the piecewise-warp diagnostic",
        "At least 3 percentage-point AvgOBO improvement with the paired bootstrap interval excluding zero",
        "Median absolute count change no greater than 0.25 for untouched identities when another identity is warped"
      ],
      "preregistered_failure": [
        "Private state is more than 3% worse than stateless windows on clean supplied tracks",
        "Untouched identities change by more than 0.5 count on median",
        "Routing histories remain indistinguishable across imposed tempo profiles",
        "State-reset or track-length-only controls reproduce the private-state result"
      ]
    }
  ],
  "synthesis_decision": {
    "may_compose": true,
    "allowed_composition": "Use warp-equivariant private phase flow as the sole proposed learning mechanism; treat identity-private state isolation as an architectural invariant and causal ablation. Normalized overlap-add, half-open boundary handling, and one-decode-per-identity may be used as established reconstruction operators. The masked-geometric completion pilot should initially remain an independent fallback; it may later replace or initialize the encoder only if it wins the preregistered comparison.",
    "claim_boundary": "The only potentially defensible method claim is the empirically validated intersection of multi-person per-identity output, externally supplied pose tracks, no per-person count or period labels in the counting objective, window-local phase or tempo adaptation, and identity-private state. Identity state, pose input, self-supervision, mixture-of-experts, tracking, phase estimation, and overlap-add are not individually novel.",
    "forbidden_inflation": [
      "Do not stack chirplets, Lomb-Scargle, multitaper estimation, variational clocks, graph synchronization, ledgers, and tracking caches merely to accumulate contributions.",
      "Do not present the supplied-versus-predicted protocol, overlap-add, private buffers, or ragged batching as the central algorithmic novelty.",
      "Do not convert prospective thresholds, synthetic diagnostics, v46 results, or v63 code into performance evidence.",
      "Do not describe the system as annotation-free, end-to-end, first, constant-time, or state of the art."
    ]
  },
  "strongest_objections": [
    "No candidate has an end-to-end implementation or an eligible run. Every score and threshold here is prospective.",
    "The novelty margin is narrow: MultiCounter and MultiCounter+ already cover MRAC and identity consistency; PAMS covers pose-driven self-supervised period adaptation; RepNet covers time-varying period; TWCRAC and HTRM-Net occupy local non-stationary modeling.",
    "The most attractive candidates can learn augmentation fingerprints, absolute time, smoothness, or limb harmonics. No-pose, timestamp-only, shuffled-time, unseen-resampler, reversal, and effective-rank controls are mandatory.",
    "The leakage-free MultiRep release, canonical-source split, timestamps, confidence masks, evaluator, and predicted-track artifacts are unavailable or unfrozen, so even a correct implementation cannot yet generate admissible evidence.",
    "Identity-private state can be mere bookkeeping or track-length memory. It needs state-reset, state-swap, shared-scene, stateless, and cross-person tempo-interference controls.",
    "A supplied-track result isolates counting but does not establish deployable MRAC. Predicted-track results must later exclude GT boxes and GT identities and report HOTA, IDF1, ID switches, unmatched persons, and count degradation separately.",
    "A four-page ICASSP paper cannot defend several spectral estimators, a tracking subsystem, uncertainty calibration, and a protocol contribution simultaneously. One mechanism and a small number of decisive ablations are the credible ceiling.",
    "The current same-family jury cannot provide independent acceptance evidence; its status must remain provisional."
  ],
  "top_venue_assessment": {
    "icassp_2027": "Potentially suitable if the primary pilot survives shortcut controls and shows protocol-matched, multi-seed improvement specifically under within-track piecewise tempo changes. The signal-processing framing is strongest when phase equivariance, local tempo behavior, and identity interference are directly measured.",
    "current_readiness": "Not submission-ready. There is no canonical implementation, eligible result, frozen split, frozen evaluator, predicted-track evaluation, uncertainty analysis, complete author metadata, or confirmed 2027 template.",
    "broader_top_venue_ceiling": "Insufficient for a top computer-vision or machine-learning venue at present. The component novelty is too crowded and the data/evidence scale too limited unless the work reveals a robust new phenomenon beyond a narrowly engineered MRAC improvement.",
    "paper_scope": "One core phase-learning mechanism, one identity-isolation ablation, one local-versus-global tempo comparison, and a compact supplied-versus-predicted protocol separation are the maximum defensible four-page scope."
  },
  "immediate_next_actions": [
    "Freeze an authorized MultiRep release, canonical-source mapping, source-disjoint train/development/test split, and checksums before accessing outcome labels for model decisions.",
    "Freeze and test the official or faithfully reproduced Period-mAP, AP50, AP75, AvgMAE, AvgOBO, unmatched-person, and paired video-cluster bootstrap evaluators.",
    "Create a clean canonical implementation commit with run binding for configuration, seed, environment, hardware, logs, checkpoints, raw predictions, and completion receipts.",
    "Implement track-global autocorrelation and stationary local-window baselines before the primary candidate so every claimed gain has a protocol-matched comparator.",
    "Run the three selected pilots on supplied tracks without opening any sealed test set; execute collapse, harmonic, no-pose, timestamp-only, shuffled-time, grid-shift, and cross-person interference diagnostics.",
    "Kill the primary mechanism if it misses its preregistered downstream and representation gates; do not rescue it by adding multiple untested candidate modules.",
    "If the primary fails, promote masked-geometric completion only if its independent preregistered pilot passes; otherwise stop method expansion and revisit the research question.",
    "After a supplied-track mechanism passes, implement the dual-track protocol with one frozen detector, pose estimator, and tracker, ensuring predicted inference consumes no GT boxes or GT identities.",
    "Resolve the full technical overlap with TWCRAC before drafting novelty language.",
    "Keep v46, v62, and v63 quarantined as historical audit material only; none may seed a paper metric or method-validation claim."
  ],
  "verdict": {
    "decision": "Advance three bounded development pilots, but do not freeze a manuscript method or claim performance.",
    "preferred_path": "Primary: warp-equivariant private phase flow. Secondary fallback: masked geometric TSSM completion. Control: identity-private state isolation.",
    "evidence_status": "Zero eligible method results. v46, v62, and v63 are not candidate evidence, and no current server tree implements the proposed system end to end.",
    "claim_cap": "At most, future evidence may support a qualified claim that identity-private, window-local phase learning improves pose-driven per-person counting under frozen within-track non-stationary tempo diagnostics without using per-person count or period labels in the counting objective."
  }
}
