# Source context — `20260815_fragment_transport_sol`

Technical conclusions below use primary papers, official proceedings, official publisher records, or author arXiv manuscripts. Search snippets from secondary indexes were used only to resolve primary URLs, not as substantive evidence.

## Repetitive-action counting

### RepNet — Dwibedi et al., CVPR 2020

- Primary: https://arxiv.org/abs/2006.15418
- Official explanation: https://research.google/blog/repnet-counting-repetitions-in-videos/
- Context read: abstract and official method explanation. RepNet predicts per-frame period length `l_i` and periodicity `p_i`; the count contribution is `p_i/l_i`, summed across frames for the final count.
- Use in decision: direct evidence that a count-density/additivity prior is old. It is not fragment-boundary transport.

### MultiCounter — Tang et al., ECAI 2024

- Primary DOI: https://doi.org/10.3233/FAIA240494
- Author manuscript: https://arxiv.org/abs/2409.04035
- Context read: abstract, task definition, related work, and method sections. The output is per person and per frame; `L_t^k = ∅` when person `k` is not visible because of occlusion or temporal absence. Instance queries, mixed spatial-temporal interaction, and instance/period heads jointly detect, track, localize period boundaries, and count.
- Use in decision: MRAC, missing visibility semantics, per-identity outputs, and joint tracking/counting are occupied. It does not disclose fragment-response mass correction.

### MultiCounter+ — Tang et al., IEEE TCSVT 2026

- Official publisher: https://ieeexplore.ieee.org/document/11421441/
- DOI: https://doi.org/10.1109/TCSVT.2026.3670243
- Context read: official abstract and introduction. The record confirms volume 36(7), pages 9462–9476, publication 4 March 2026, mixed spatiotemporal interaction, task-specific heads with spatial-temporal consistency and long-short period awareness, MultiRep, and large-scale synthetic pretraining.
- Access limitation: full method text was blocked. The task-supplied Siamese-query and dynamic-update detail was not independently reverified. It is treated conservatively as an occupied boundary and not as an independently established fact.
- Use in decision: no first identity-consistency, robust MRAC, synthetic-pretraining, or variable-period claim.

### PAMS — Gao et al., CVPR Findings 2026

- Official paper: https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.pdf
- Proceedings page: https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.html
- Context read: abstract, method overview, and related work. PAMS-TCC aligns corresponding moments within a video over multiple scales, adds temporal and cross-video discrimination, and uses an inference committee varying window, smoothing, and peak thresholds.
- Use in decision: pose/skeleton self-supervision, cycle consistency, cross-video negatives, and multi-scale tempo robustness are occupied. PAMS does not disclose fragment-boundary mass transport.

### TWCRAC — 2026 primary SSRN record

- Primary record: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6630216
- Context read: accessible abstract/record covering time-window-cycle framing, intra-video temporal cycle consistency, pose, local periodic statistics, dynamic thresholds, and non-stationary motion.
- Access limitation: full method and related work remained inaccessible.
- Use in decision: fail-closed against broad local-window/cycle-consistency claims. Exact fragment overlap was not cleared.

## Fragmented-track repair, occlusion, and re-identification

### Tekin et al. — Automated repair of fragmented tracks with 1D CNNs

- Publisher DOI: https://doi.org/10.1016/j.imavis.2020.103982
- Venue: Image and Vision Computing 101 (2020), article 103982.
- Context read: publisher abstract and method excerpts. A fragment is represented through the left track, untracked gap, and right track. A 1D CNN/GRU encoding learns correction deltas over linear interpolation using box coordinates/detections; evaluation measures geometric gap repair and downstream tracking.
- Use in decision: strongest mechanism collision. Learned endpoint/gap correction is not new. The remaining task delta is response mass rather than boxes.

### OC-SORT — Cao et al., CVPR 2023

- Official proceedings: https://openaccess.thecvf.com/content/CVPR2023/html/Cao_Observation-Centric_SORT_Rethinking_SORT_for_Robust_Multi-Object_Tracking_CVPR_2023_paper.html
- Author manuscript: https://arxiv.org/abs/2203.14360
- Context read: abstract and method description of observation-centric re-update. ORU forms a virtual trajectory between the last observation and the reappearing observation, re-running the state update to mitigate accumulated Kalman error over occlusion.
- Use in decision: observation-endpoint state repair over a gap is established tracking machinery, not count transport.

### TPTrack — 2024

- Publisher DOI: https://doi.org/10.1016/j.compeleceng.2024.109078
- Context read: publisher abstract. The method globally links fragmented trajectories and uses Gaussian-process interpolation for missing detections.
- Use in decision: fragment linking and gap interpolation are occupied.

### Deep OC-SORT — Maggiolino et al., 2023

- Author manuscript: https://arxiv.org/abs/2302.11813
- Context read: abstract and method boundary. Appearance embeddings and adaptive association add re-identification to observation-centric tracking.
- Use in decision: identity re-entry and appearance association are occupied. A wrong-ID negative is not independently novel.

## Count-flow conservation neighbors

### Video Individual Counting for Moving Drones — Fan et al., ICCV 2025

- Official proceedings: https://openaccess.thecvf.com/content/ICCV2025/html/Fan_Video_Individual_Counting_for_Moving_Drones_ICCV_2025_paper.html
- Author manuscript: https://arxiv.org/abs/2503.10701
- Context read: abstract, related work, and method. SDNet uses cross-frame attention to estimate a shared density map for consecutive frames. Shared density is subtracted from each global density map to produce outflow/inflow; inflow is summed through the clip to count unique pedestrians.
- Use in decision: strongest conservation collision. A learned temporal density decomposition and cumulative count-flow law are already explicit. The task differs from RAC fragment boundaries, but “mass conservation” cannot be claimed as new.

### Density-Based Flow Mask Integration — Wan et al., WACV 2024

- Official proceedings: https://openaccess.thecvf.com/content/WACV2024/html/Wan_Density-Based_Flow_Mask_Integration_via_Deformable_Convolution_for_Video_People_WACV_2024_paper.html
- Context read: abstract, introduction, and related-work boundary. The work estimates people entering/exiting a video region in density space because MOT identity failures corrupt unique-person flow counts.
- Use in decision: further prior for learned density flow under tracking/occlusion failure.

## Abstention and recent-six-month coverage

### Distribution-Free Sequential Prediction with Abstentions — Yu and Blanchard, COLT 2026

- Official PMLR: https://proceedings.mlr.press/v336/yu26a.html
- Context read: abstract and method statement for an explicit prediction-error/abstention tradeoff.
- Use in decision: abstention is mature generic machinery.

### CHASE — May 2026

- Primary: https://arxiv.org/abs/2605.01346
- Context read: abstract and method summary. Structured temporal hypothesis margins trigger abstention under partial observability, evaluated at controlled coverage.
- Use in decision: residual/margin rejection under temporal ambiguity is an occupied design pattern.

### VCBench and TOC-Bench — March/May 2026

- VCBench: https://arxiv.org/abs/2603.12703
- TOC-Bench: https://arxiv.org/abs/2605.09904
- Context read: abstracts. These benchmarks diagnose streaming object/event count state and identity consistency across disappearance, occlusion, and reappearance.
- Use in decision: not isomorphic methods, but they confirm active 2026 scrutiny of identity-sensitive counting and temporal state maintenance.

## Search conclusion

No checked primary method contained all five candidate elements in one method. This is a bounded negative result, not evidence of firstness. The closest two bridges are independently strong:

1. Tekin/OC-SORT/TPTrack: endpoints and gaps drive learned repair or state re-update.
2. RepNet/Video Individual Counting: local count measures are additive, decomposed, and accumulated.

Adding PAMS-style consistency, mismatched-identity negatives, and a generic reject option is an obvious composition. The output therefore fails closed as `ABANDON_AS_PRIMARY_METHOD_NOVELTY_RETAIN_ONLY_AS_BOUNDED_DIAGNOSTIC`.
