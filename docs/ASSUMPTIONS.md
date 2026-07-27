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
| Pose | MediaPipe/BlazePose 33 joints × xyz | inferred from pose-baseline ecosystem | Pose model/version and file hash enter every cache manifest |
| Person policy | longest valid primary-person track | inferred | Decode/track failures use one published rule |
| Missing pose | all-zero frame plus `valid_mask=False` | inferred/recovered from PoseRAC | Models receive the mask; evaluator reports invalid fraction |
| Normalization | per-frame min-max over valid coordinates | inferred/recovered from PoseRAC | Zero-range frames become invalid, never NaN |
| Temporal shape | uniform resampling to 256 frames | inferred | Original FPS and frame count remain in metadata |
| Encoder | 99→512; 4 layers; 16 heads; FFN 2048; dropout .1; post-LN; L2 output | disclosed except post-LN | `norm_first=False` is independently fixed |
| Period range | 4–128 resampled frames | inferred | Synthetic tests cover boundary and harmonic errors |
| Warm-up period | pose-energy for first 10 epochs | partially disclosed/inferred | Thereafter use stop-gradient embedding velocity |
| PAMS scales | 0.5, 1.0, 1.5 | disclosed | Report single-scale ablation separately |
| Positives | adjacent frames and TSM maximum near `t±T_hat` | partially disclosed/inferred | Exact window clipping is unit-tested |
| Negatives | temporal, cross-video, equal-count cross-cluster | partially disclosed | Sampling seed and indices are reproducible |
| KMeans | 8 video-level clusters, refresh every 5 epochs | refresh disclosed; `k=8` inferred | Empty/small clusters use deterministic fallback |
| Contrastive loss | equal scale weights, multi-positive InfoNCE, temperature .1 | partially disclosed | No hidden class/count labels |
| Optimizer | AdamW lr/weight decay `1e-4`; batch 32; 150 epochs | optimizer disclosed; batch/epochs inferred | Full config and scheduler state are checkpointed |
| Period Head literal | random, never trained | literal consequence | Diagnostic only; not expected to meet target |
| SSHead | freeze encoder; 30 epochs; cycle + spectral + .1 variance + .01 smoothness | inferred repair | Always labeled `PAMS-SSHead`, never author-faithful |
| Expert smoothing | sigma multipliers .05/.12/.15 of period | inferred from supplement ranges | Fixed before held-out evaluation |
| Peak distance | .5/.8/1.2 of period | inferred from supplement ranges | Rounded/clamped deterministically |
| Threshold windows | short .5×period, long 2×period | inferred | Boundary padding is deterministic |
| Thresholds | height .6, prominence .25, long weight .6 | inferred from supplement centers | No result-dependent test tuning |
| Consensus fallback | majority vote; otherwise closest to FFT reference; ties favor Medium | disclosed plus inferred tie rule | Expert counts are saved per video |
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
