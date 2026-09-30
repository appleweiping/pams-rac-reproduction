# PAMS paper and protocol audit

This repository is an independent reproduction, not an author release. The canonical
public sources are the
[CVF paper page](https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.html)
and its [supplement](https://openaccess.thecvf.com/content/CVPR2026F/supplemental/Gao_Count_What_Repeats_CVPRF_2026_supplemental.pdf).
Every repository claim uses one of these labels:

- **disclosed**: stated unambiguously in a primary paper or released artifact;
- **recovered**: observed in public author code or annotations;
- **inferred**: an independently chosen closure for missing information;
- **diagnostic**: useful for auditing but ineligible for the fair leaderboard.

## What the paper discloses

The main paper, Sections 3.1–3.3 and Algorithm 1, discloses the following core design:

- a four-layer, 512-dimensional Transformer pose encoder with 16 attention heads,
  2,048-dimensional feed-forward layers, 0.1 dropout, sinusoidal positions and
  L2-normalized frame embeddings;
- a two-layer `512 -> 128 -> 1` period head;
- period-adaptive temporal windows at scales 0.5, 1.0 and 1.5;
- local and cycle-correspondence positives plus temporal, cross-video and
  cross-cluster negatives in a multi-positive contrastive objective;
- KMeans refresh every five epochs, InfoNCE temperature 0.1, AdamW learning rate and
  weight decay both `1e-4`, and training on one A100;
- FFT-guided selection among fast, medium and slow peak-counting experts.

Table 1 reports PAMS on UCFRep as NMAE/“MAE” `0.208` and OBO `0.686`. Table 2 reports the
UCFRep ablation path from `0.340/0.423` for the stated baseline to `0.208/0.686` for the
full method. These are targets, not results produced by this repository.

The supplement provides sensitivity ranges for TCC window/stride/temperature and expert
peak parameters, and reports an end-to-end throughput of 24.81 FPS. It does not identify
a unique final setting for every expert parameter.

## Fatal Period Head gap

The text applies PAMS TCC directly to the encoder embeddings. No objective, pseudo-target,
gradient path or pretraining procedure is specified for the Period Head, yet inference is
described as using a trained encoder and Period Head. A literal implementation therefore
leaves the head random.

We preserve that literal behavior as `PAMS-Literal`, a diagnostic. The useful
`PAMS-SSHead` result is an explicitly **inferred** repair: freeze the PAMS encoder and
train the head with cycle consistency, fundamental-frequency spectral concentration,
anti-collapse variance, and second-order smoothness. It must never be described as the
authors' undisclosed method.

## Metric correction

The paper calls its primary error “MAE,” but the scale of the reported values and the
field's released evaluators establish that it is normalized MAE:

```text
NMAE = mean(abs(round(prediction) - ground_truth) / ground_truth)
OBO  = mean(abs(round(prediction) - ground_truth) <= 1)
```

This interpretation is corroborated by the released
[PoseRAC evaluator](https://github.com/MiracleDance/PoseRAC/blob/469590b611bde3595eaf163b517263da634e2096/eval.py#L123-L132),
[Every Shot Counts implementation](https://github.com/sinhasaptarshi/EveryShotCounts),
and the formula in the [CountLLM paper](https://arxiv.org/abs/2503.17690). We call the
metric **NMAE (paper label: MAE)** and additionally publish raw MAE and RMSE. A report
that silently labels NMAE as raw MAE is invalid.

## UCFRep protocol mismatch

The original [UCFRep release](https://github.com/Xiaodomgdomg/Deep-Temporal-Repetition-Counting)
contains 526 videos. The standard split used here is 421 train and 105 held-out videos.
By contrast, the [PoseRAC-v1 paper](https://arxiv.org/abs/2303.08450) and released
[PoseRAC repository](https://github.com/MiracleDance/PoseRAC) use a five-action
UCFRep-pose subset of 110 videos with an 89/21 split. GMFL, SPKDB and BIGC also report
the small pose protocol.

PAMS Table 1 places those published numbers beside a PAMS result on the standard
UCFRep evaluation without flagging the split difference. They are not an apples-to-apples
leaderboard. This repository publishes:

1. a **reported/source-faithful table** that preserves each source's split and warnings;
2. a **fair UCFRep-526 table** in which every included method uses the same 421/105 IDs;
3. a separate **fair UCFRep-pose-110 table** using the same 89/21 IDs.

No number is copied between those tables as though the protocols were equivalent.

## PoseRAC ground-truth-count leakage

The released PoseRAC-v1 evaluation reads `gt_count`, iterates over action-conditioned
mappings and retains the candidate with the smallest error to that test video's true
count ([permalink to the audited revision](https://github.com/MiracleDance/PoseRAC/blob/469590b611bde3595eaf163b517263da634e2096/eval.py#L56-L132)).
The ground-truth count therefore selects the prediction at inference. This is oracle
leakage, not a fair action-agnostic counter.

The source-faithful value may be reproduced only as a clearly marked `oracle/diagnostic`
row. It is structurally forbidden from `fair_comparison=True` by
`ProtocolRecord`. The fair adapter must select its output without count labels. An
oracle action-class diagnostic is allowed in a separate column; a GT-count oracle is
never allowed in a legitimate leaderboard.

## Other unresolved evidence

- The paper does not disclose the pose estimator, number of joints, person tracking,
  frame rate/resampling policy, batch size, epoch count, KMeans cluster count, period
  bounds, exact positive/negative sampling, or all expert defaults.
- The figure says “Pose Estimation or ResNet,” while the PAMS row is discussed as
  skeleton-based. This is insufficient to claim an RGB PAMS implementation.
- The supplement's expert sensitivity plots provide ranges but do not mark a unique
  optimum. Several heatmaps appear numerically identical despite different expert
  ranges, so they cannot recover the missing defaults.
- JTSPS public material does not currently determine a unique source-faithful training
  and evaluation protocol. Its registry status is `blocked_protocol`, not a guessed
  runnable implementation.
- CountLLM's original three-stage training includes WebVid-10M and eight A100 GPUs.
  `CountLLM-Lite` omits stage 1 and is deliberately non-comparable to the reported
  CountLLM row.

All independently fixed choices are recorded in [ASSUMPTIONS.md](ASSUMPTIONS.md).
