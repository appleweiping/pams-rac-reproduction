# Fragment-Boundary Count-Mass Transport — strict novelty check

Run: `20260815_fragment_transport_sol`  
Candidate: `fragment-boundary-count-mass-transport`  
Cutoff: 2026-08-15  
Reviewer/model: `gpt-5.6-sol` / OpenAI  
Fresh context: `false`  
Non-fresh reason: `agent_thread_lifetime_limit`  
Independence/status: same-family / provisional

## Verdict

**ABANDON — 3.9/10.** Abandon this as a primary ICASSP method-novelty candidate. At most retain it as a bounded diagnostic if the project owner explicitly reopens it.

No checked primary method contained the exact five-part conjunction of (1) synthetic cut/gap, (2) endpoint-state response transport, (3) intact response-mass conservation, (4) wrong-ID negatives, and (5) residual abstention. That negative search does not rescue the idea. Its central bridges already exist: learned fragment/gap correction is established in tracking, while additive count/density conservation and temporal count-flow decomposition are established in counting. Wrong-ID negatives, consistency supervision, and abstention are generic. The current proposal is therefore an obvious task-specific composition, not a defensible new principle.

This review is fail-closed. The full IEEE method for MultiCounter+ and the full TWCRAC method were not independently retrievable, so the narrow residual receives no novelty clearance.

## Candidate boundary

The candidate freezes a label-restricted supplied-track base response model, deterministically cuts complete training tracks in feature space, and learns a small correction from endpoint pose and velocity such that

`intact frozen-response mass ≈ sum(fragment frozen-response mass) + boundary correction`.

The supplied identity is only a routing key. Mismatched endpoint identities are negatives. At inference, a large transport residual triggers abstention or an interval. The learner may not access count, period, density, or boundary labels.

WARP-PHASE is terminally killed by immutable Gate2 vector `[246, 55, 144, 50, 66]`. It supplies no efficacy or novelty evidence and may not be revived, tuned, or reinterpreted by this idea.

## Four audited claims

### C1 — compositional count mass

Queries included:

- `repetition counting fragmented tracks count mass conservation endpoint correction 2024 2025 2026`
- `event counting temporal fragments additive density boundary correction video 2024 2025 2026`
- `video repetition count crop cut consistency density integral additivity fragments`
- `site:arxiv.org/abs after:2026-02-15 repetition counting fragment gap count conservation endpoint`

Finding: not novel as a principle. RepNet explicitly obtains a per-frame count contribution `p_i/l_i` and sums it. Density-based RAC similarly integrates density. More damaging, ICCV 2025 *Video Individual Counting for Moving Drones* learns shared density between adjacent frames, subtracts it from global density to obtain inflow/outflow, and sums inflow over time. The exact frozen-RAC fragment residual was not found, but count additivity/conservation is occupied.

### C2 — endpoint-conditioned fragment repair and identity isolation

Queries included:

- `Automated repair of fragmented tracks with 1D CNNs Tekin Image and Vision Computing 101 103982`
- `tracklet association gap interpolation endpoint motion re-identification 2024 2025 2026`
- `occlusion fragmented tracks re-entry wrong-ID negative endpoint velocity`
- `site:arxiv.org/abs after:2026-02-15 tracklet gap filling re-entry endpoint state uncertainty`

Finding: mechanistically crowded. Tekin et al. encode left/right track fragments plus the gap and learn correction deltas over linear interpolation. OC-SORT uses observation-centric re-update to form a virtual trajectory between the last observation and reappearance. TPTrack links fragments and interpolates gaps; Deep OC-SORT adds adaptive appearance association. These works repair geometry or association, not repetition-response mass, but endpoint/gap transport itself is not novel.

### C3 — synthetic cuts, intact consistency, and wrong-ID negatives

Queries included:

- `self-supervised temporal cut consistency synthetic gaps intact fragment conservation`
- `temporal crop consistency action counting sum fragment count`
- `cut paste temporal consistency missing interval video repetition self supervision`
- `site:arxiv.org/abs after:2026-02-15 temporal crop cut consistency repetition counting`

Finding: the exact construction was not found, but the ingredients are standard. PAMS learns skeleton periodic structure through intra-video multi-scale temporal cycle consistency plus temporal/cross-video discrimination. Mismatched identities are ordinary negatives in association/re-ID. The intact frozen output is merely a pseudo-target and can preserve the base counter's bias.

### C4 — residual abstention

Queries included:

- `selective prediction abstention tracklet association gap uncertainty 2024 2025 2026`
- `uncertainty abstain fragmented trajectory repair endpoint states`
- `repetition counting abstention uncertain period interval`
- `site:arxiv.org/abs after:2026-02-15 selective prediction temporal partial observability abstention`

Finding: no checked RAC paper used this exact residual, but selective prediction and risk/coverage analysis are mature. Recent 2026 work explicitly uses temporal hypothesis margins to abstain under partial observability. A residual threshold is a safety policy, not a standalone novelty.

## Closest primary priors and precise delta

1. **Tekin et al., 2020, *Automated repair of fragmented tracks with 1D CNNs*.** Closest mechanism: endpoint fragments and a gap drive learned correction over interpolation. Delta: response-space count correction rather than missing box coordinates.
2. **Fan et al., ICCV 2025, *Video Individual Counting for Moving Drones*.** Closest conservation prior: learned shared/global density decomposition, inflow/outflow, and temporal summation. Delta: same-identity synthetic RAC fragments rather than adjacent-frame pedestrian flow.
3. **RepNet and density-based RAC.** Per-frame count measures are summed. Delta: diagnose and correct non-additivity introduced by fragment boundaries in a frozen response.
4. **OC-SORT, TPTrack, Deep OC-SORT, and tracklet/re-ID work.** Gap bridging, virtual trajectory re-update, interpolation, re-entry, and identity association are established. Delta: do not reconstruct geometry or replace the tracker.
5. **PAMS and TWCRAC.** Pose-driven temporal/cycle consistency and non-stationary local periodic structure are occupied. Delta: conservation of a frozen response across fragments rather than representation or phase learning.
6. **MultiCounter / MultiCounter+.** MultiCounter formally allows `L_t^k = ∅` for an invisible instance and jointly detects, tracks, localizes period boundaries, and counts. The official MultiCounter+ record confirms mixed spatiotemporal interaction, spatial-temporal consistency, long-short-period awareness, and synthetic pretraining. Its full method was unavailable; the task-supplied Siamese-query/dynamic-update description is conservatively treated as occupied, not independently reverified.

The precise residual is narrow: learn, from intact-versus-synthetically cut pose features of one supplied identity, an endpoint-conditioned boundary term for a frozen per-track RAC response; reject mismatched identities and abstain when the response residual is unidentifiable. This is not a new tracker, re-ID method, density/count law, temporal-consistency principle, or abstention method.

## Why the combination is obvious

The proposal composes five familiar moves:

- Tekin/OC-SORT-style gap correction;
- RepNet/density-style additive count measure;
- PAMS-style self-consistency;
- standard mismatched-identity negatives;
- generic selective abstention.

Not finding the exact conjunction is insufficient. To become more than a composition, the work would need a nontrivial conservation or identifiability result and evidence that the learned term cannot be reduced to reconstruct-then-count, gap/cut metadata, scalar residual calibration, a parameter-matched endpoint MLP, or ordinary crop consistency.

## Fatal risks

- **Pseudo-target tautology:** learning `intact − fragments` reproduces the frozen counter's bias.
- **Shortcut leakage:** gap length, cut position, padding, or fragment length can predict the residual.
- **Non-identifiability:** endpoints do not determine cycles hidden in a long or phase-ambiguous gap.
- **Trivial wrong-ID gate:** a supplied identity key can make mismatched pairs mechanically rejectable.
- **Synthetic-to-real gap:** artificial feature cuts do not establish robustness to detector failure or ID switches.
- **Measure degeneracy:** near-additive responses yield edge calibration; non-additive responses make conservation enforce a defective measure.
- **Reviewer obviousness objection:** tracking repair + density additivity + consistency + reject option is an evident combination.
- **Incomplete 2026 full-text clearance:** MultiCounter+ and TWCRAC remain fail-closed.

## Allowed claims

- A label-restricted diagnostic for compositional failure of a frozen per-track repetition response under deterministic synthetic fragmentation of supplied identity tracks.
- The learner does not receive count, period, density, or boundary labels. Do not shorten this to “annotation-free.”
- Only after a predeclared pass: the correction reduced intact-versus-fragment discrepancy beyond matched baselines on source-disjoint unseen cut families.
- Only after separate evidence: residual rejection improved a predeclared risk/coverage metric.
- Any contribution is a supplied-track failure diagnostic/correction, not a tracker or base counter.

## Forbidden claims

- first fragmentation-robust RAC, first MRAC, or first per-identity RAC;
- first count additivity, density integration, mass conservation, temporal flow, or crop consistency;
- first gap repair, occlusion recovery, re-entry handling, re-ID, wrong-ID negative, or abstention;
- novel endpoint transport without direct Tekin/OC-SORT/TPTrack and reconstruct-then-count comparisons;
- annotation-free, supervision-free, label-free system, or end-to-end MRAC;
- robustness to real predicted tracks, detector failures, or ID switches;
- SOTA, improved count accuracy, or any result claim before evaluator-only validation;
- novelty from phrase absence or from the failed WARP-PHASE result.

## Minimum gate if explicitly reopened as a diagnostic

This review does not authorize implementation or training. If the owner explicitly reopens the diagnostic, freeze feature-only source-disjoint train/validation tracks and predeclare unseen cut/gap families. Compare raw fragment sums, gap/cut-only regression, linear endpoints, a parameter-matched endpoint MLP, pose-interpolate-then-recount, generic crop consistency, and residual-threshold abstention.

Kill if any of the following occurs:

- unseen-cut reconstruction error exceeds 5% of intact frozen-response mass;
- wrong-ID pairs receive more than 10% transport weight;
- effective-rank, constant, or gap-only collapse;
- cutting one person changes any byte-identical untouched-person output;
- the proposed operator fails to beat the best matched comparator on every predeclared unseen-cut family;
- evaluator-only hidden labels show no true-count improvement over raw fragments and interpolate-then-count, or conservation worsens count error;
- abstention fails to improve the predeclared risk/coverage curve at matched coverage.

Required shortcut controls are endpoint shuffling at fixed gap/cut metadata, identity removal/permutation, left/right swap and valid time reversal, held-out gap ranges/actions/source videos, effective-rank audit, and a direct measurement of how much gain gap/cut metadata alone recovers.

Even a pass supports only a diagnostic unless the correction is shown not to reduce to tracking reconstruction, scalar calibration, or ordinary crop consistency.

## Source URLs

- RepNet: https://arxiv.org/abs/2006.15418
- MultiCounter: https://doi.org/10.3233/FAIA240494
- MultiCounter+: https://doi.org/10.1109/TCSVT.2026.3670243
- PAMS: https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.html
- TWCRAC: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6630216
- Tekin et al.: https://doi.org/10.1016/j.imavis.2020.103982
- OC-SORT: https://arxiv.org/abs/2203.14360
- TPTrack: https://doi.org/10.1016/j.compeleceng.2024.109078
- Deep OC-SORT: https://arxiv.org/abs/2302.11813
- Video Individual Counting: https://openaccess.thecvf.com/content/ICCV2025/html/Fan_Video_Individual_Counting_for_Moving_Drones_ICCV_2025_paper.html
- WACV 2024 video people flux: https://openaccess.thecvf.com/content/WACV2024/html/Wan_Density-Based_Flow_Mask_Integration_via_Deformable_Convolution_for_Video_People_WACV_2024_paper.html
- 2026 abstention: https://proceedings.mlr.press/v336/yu26a.html
- CHASE: https://arxiv.org/abs/2605.01346
- VCBench: https://arxiv.org/abs/2603.12703
- TOC-Bench: https://arxiv.org/abs/2605.09904

No implementation, training, MANIFEST edit, next-stage launch, or server/test/sealed/heldout/results access was performed.
