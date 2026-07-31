# Frozen assumptions register

The PAMS paper leaves implementation-critical choices unspecified. This register makes
the independent closures auditable. “Disclosed” values come from the
[main paper](https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.html)
or [supplement](https://openaccess.thecvf.com/content/CVPR2026F/supplemental/Gao_Count_What_Repeats_CVPRF_2026_supplemental.pdf);
all other values are frozen **inferences**, not claims about author code.

| Area | Frozen choice | Evidence class | Consequence / audit |
|---|---|---|---|
| Primary data | UCFRep-526, 421 train / 105 held out | recovered | Split IDs and hash must accompany results |
| Development split | 337/84 stratified within the 421, seed 2026 | inferred | Test 105 remain sealed until configuration freeze |
| Pose | MediaPipe Pose 0.10.14, complexity 1, smooth landmarks, detection/tracking confidence .5, 33 joints × xyz | inferred from pose-baseline ecosystem | Frozen extractor block and source-video hash enter cache identity |
| Pose cache identity | SHA-256 of only canonical `data` + `pose` blocks | independent audit closure | Seeds/training settings share caches; extractor changes cannot |
| Person policy (current v3) | legacy MediaPipe is single-person; choose a dominant candidate if supplied, then select the earliest longest contiguous run of detections before normalization/resampling | reproduction-plan closure | This is contiguous detection-run selection, not cross-person identity tracking; all-invalid videos remain explicit caches |
| Person policy (historical v2) | crop only before the first and after the last detection and retain every internal miss as an exact-zero masked frame | superseded inferred closure | Retained only so published v2 artifacts remain byte-identifiable; it is not used for the v8 protocol-correction experiment |
| One-frame v3 run | retain the original video duration but emit an all-zero/all-invalid 256-frame cache | independent safety closure | A single observation has no temporal duration and is never expanded into a fabricated valid trajectory; selected raw-run length is recorded |
| Missing pose | all-zero frame plus `valid_mask=False`; inference filters each contiguous valid run independently | inferred/recovered from PoseRAC plus independent closure | Models receive the mask; no smoothing or threshold statistic crosses an invalid gap; evaluator reports invalid fraction |
| Normalization | per-frame min-max over valid coordinates | inferred/recovered from PoseRAC | Zero-range frames become invalid, never NaN |
| Temporal shape | uniform resampling to 256 frames | inferred | Original FPS and frame count remain in metadata |
| Encoder | 99→512; 4 layers; 16 heads; FFN 2048; dropout .1; post-LN; L2 output | disclosed except post-LN | `norm_first=False` is independently fixed |
| Period range | 4–128 resampled frames; preserve fractional FFT-bin period until integer indexing is required | inferred | Avoids quantizing values such as 256/40=6.4 before consensus |
| Warm-up period | pose-energy for first 10 epochs | partially disclosed/inferred | Thereafter use stop-gradient embedding velocity |
| V14 period proxy | full 512-D embedding-velocity vector autocorrelation and bounded FFT selector | inferred interpretation of the paper's embedding autocorrelation | Orthogonal-basis invariant; distinct fingerprint; seed-2026 train/synthetic gate precedes any dev84 run |
| Fixed-period Table 2 proxy | `training_mode=fixed_period_inferred`, 16 resampled frames, single scale | inferred; the paper says “conventional TCC”/fixed window but does not disclose its chosen baseline window or full geometry; Supplementary Fig. 12a only displays candidate sizes | Requires a new encoder (and a new SSHead if that inferred readout is evaluated); never claim the paper's Table 2 baseline value |
| PAMS scales | 0.5, 1.0, 1.5; inferred symmetric radius `max(1, round(s*T/2))` around `t±T` | scales disclosed, window geometry inferred | Shortest-period windows remain distinct; report single-scale ablation separately |
| Positives | adjacent frames and TSM maximum near `t±T_hat` | partially disclosed/inferred | Exact window clipping is unit-tested |
| Negatives | temporal frames, every valid frame from other physical-batch videos, and distinct equal-count cross-cluster prototypes from a frozen full-training bank | cross-video frames disclosed; prototype-bank closure inferred | Current-batch IDs are excluded from bank top-k; formal runs fail on any explicit shortfall |
| KMeans | 8 video-level clusters and full video prototype bank, refreshed every 5 epochs | refresh disclosed; `k=8` and frozen bank inferred | Ordered bank/features/clusters are checkpointed for bitwise resume |
| Contrastive loss | equal scale weights; mean of per-positive InfoNCE terms from paper Eq. (1); temperature .1 | disclosed except candidate-set closure | No hidden class/count labels; not a log-sum-exp positive surrogate |
| Optimizer | AdamW lr/weight decay `1e-4`; batch 32; 150 epochs | optimizer disclosed; batch/epochs inferred | Full config and scheduler state are checkpointed |
| V14 skeleton augmentation | centered three-axis rotation `+/-15` degrees, isotropic scale `[.85,1.15]`, Gaussian jitter std `.01`; encoder optimization view only | augmentation types disclosed; ranges inferred from supplement robustness settings except jitter | Clean view remains the source for period evidence, clustering, SSHead and inference; no label inputs |
| Period Head literal | random, never trained | literal consequence | Diagnostic only; not expected to meet target |
| SSHead | freeze encoder; 30 epochs; cycle + spectral + .1 variance + .01 smoothness | inferred repair | Always labeled `PAMS-SSHead`, never author-faithful |
| Expert smoothing | sigma multipliers .05/.12/.15 of period | inferred from supplement ranges | Fixed before held-out evaluation |
| Peak distance | .5/.8/1.2 of period | inferred from supplement ranges | Rounded/clamped deterministically |
| Threshold windows | short .5×period, long 2×period | inferred | Boundary padding is deterministic |
| Thresholds | height .6, prominence .25, long weight .6 | inferred from supplement centers | No result-dependent test tuning |
| Consensus fallback | majority vote; otherwise closest to FFT reference; ties favor Medium; confidence is vote confidence × period confidence | disclosed plus inferred tie/confidence rule | Expert counts and the composite confidence are saved per video |
| Metric | NMAE with rounded prediction, OBO within 1 | recovered/corrected | Also publish raw MAE and RMSE |
| Seeds | 42, 2026, 3407 | preregistered | Publish every seed, mean, standard deviation and paired bootstrap CI |
| Verification | mean NMAE ≤ .228 and OBO ≥ .666; 2/3 seeds pass; ablation order holds | preregistered | Otherwise release is explicitly partial |

## Baseline assumptions

- “Reported,” “official-asset rerun,” and “clean-room fair rerun” are different evidence
  types and occupy different columns.
- UCFRep-pose-110 (89/21) is a separate dataset protocol from UCFRep-526 (421/105).
- PoseRAC-v1's test-count-selected output is an oracle diagnostic. The fair adapter cannot
  read ground-truth count or action identity.
- `poserac-v1` and `poserac-iconip24` are separate registry identifiers until their
  source/protocol relationship is independently verified.
- GMFL, SPKDB and BIGC remain blocked until their missing implementations pass a
  source-specific parity fixture; publication numbers are never emitted as run results.
- JTSPS-count-only remains `blocked_protocol` while its public material cannot uniquely
  identify the source-faithful training/evaluation procedure.
- CountLLM-Lite is a reduced, resource-gated recipe. Omitting WebVid-10M prevents a
  same-condition claim regardless of eventual UCFRep score.
- The deterministic spectral proxy is a smoke/reference adapter only. It is not PAMS,
  RepNet, or any other paper baseline.

Changes to a frozen assumption require a new configuration fingerprint and a new result
namespace. They may not overwrite a previously reported run.
