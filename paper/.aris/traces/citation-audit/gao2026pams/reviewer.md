# Fresh Citation Audit: `gao2026pams`

Verdict: **FIX**, then **PASS**. The CVPR Findings 2026 metadata is correct.
Pose/self-supervised/period-adaptive, local-warp, and inference-consensus
contexts are supported. The method sentence originally called the cyclic
positives “period-offset positives,” which was more specific than the paper;
it now says “temporally adjacent and cyclically corresponding positives across
temporal scales.”

Primary record:
https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.html

---

## Current-context rebind

- Agent: `/root/citation_context_rebind/rebind_gao2026pams`
- Timestamp (UTC): `2026-08-11T19:59:01Z`
- Route: fresh zero-context, `gpt-5.6-sol`, xhigh
- Inputs disclosed to reviewer: current BibTeX entry, exact current citing paragraphs/sites, and primary official URLs only; no prior verdict or review history

EXISTENCE: SUPPORTS

- The exact [official CVF landing page](https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.html) returns HTTP 200.
- The exact [official CVF PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.pdf) returns HTTP 200 and is a 1,498,139-byte PDF.
- The PDF states that it is the CVF open-access version, identical except for the watermark to the accepted version.

FINAL METADATA: SUPPORTS

Official CVF proceedings metadata:

```text
Type:       InProceedings
Authors:    Shizhao Gao; Jun Li; Qiming Li
Title:      Count What Repeats: Period-Adaptive Multi-Scale Consistency for Self-Supervised Repetitive Action Counting
Booktitle:  Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings
Month:      June
Year:       2026
Pages:      8143-8152
```

The current BibTeX matches all official fields. Its lowercase local key and `8143--8152` BibTeX page-range syntax are harmless normalizations. The URL is the exact official landing page.

SITE appendix/a_experiment_contract.tex:62 [136458327432caa019935e2b3e818f61fbc52a11a6f3b6c69bca54404123d698]: SUPPORTS — The paper takes an unlabeled skeleton/pose sequence as input, learns representations fully self-supervised without count labels, estimates the period to form multiple scale windows, and processes one input sequence. Its conclusion explicitly leaves extension to multi-person repetitive-action counting as future work, supporting the absence of an identity mechanism.

SITE sections/1_introduction.tex:36 [a815993fe282cba936d400aaeda9696b34b1ab4fffe16b02ca98a241b85edb5a]: WRONG — The self-supervised pose-representation and lack-of-simultaneous-identity claims are supported. However, the paper does not evaluate “local time warps.” Section 4.4 uniformly resamples entire RepCount test videos at fixed global speed factors from 0.1× to 2.0×. That does not entail local or time-varying warping.

SITE sections/2_related_work.tex:34 [0426ac52a89964d27d7c715838980245eb397a7bbfc7381b0eab379fe10d4836]: SUPPORTS — The paper directly states learning from unlabeled skeleton sequences with PAMS-TCC. Inference uses deterministic Fast/Medium/Slow peak-detection experts with scale-specific smoothing and thresholds, followed by voting, rather than a learned mixture. Multi-person counting is expressly future work, so preservation of multiple identities is outside the formulation.

SITE sections/3_method.tex:240 [6aa1db2c22104e4fb9d105e84a08a63de4ba5ddc6f2fc2c162542a685838b4d2]: SUPPORTS — The method explicitly constructs positives from temporally adjacent frames and a cyclically corresponding frame within period-adaptive multi-scale windows. It constructs negatives from other temporal locations and videos and samples cross-cluster hard negatives.

SITE sections/4_experiments.tex:95 [f85257bccb350d1360f630afedb60dbbe250eb7127eb135e476415cf86dae3c8]: SUPPORTS — The citation identifies a real PAMS method with the stated pose input and self-supervised supervision regime. Adapting it to the authors’ common track interface and the disclosure protocol are prospective experiment-design statements, not claims that the cited paper itself must establish.

OVERALL: BLOCKED

---

## Fresh exact-current-context re-review after correction

Reviewer task: `/root/final_cite_pams` (fresh, zero-context, same-family,
`gpt-5.6-sol`, `xhigh`). The prior WRONG ruling above remains preserved as the
historical reason for the prose correction.

EXISTENCE: VERIFIED -- official CVF HTML record and paper PDF exist.

FINAL METADATA: Gao, Shizhao; Li, Jun; Li, Qiming. "Count What Repeats:
Period-Adaptive Multi-Scale Consistency for Self-Supervised Repetitive Action
Counting." Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR) Findings, June 2026, pages 8143--8152. Supplied
BibTeX metadata and URL match the official CVF record.

SITE `appendix/a_experiment_contract.tex:62`
(`136458327432caa019935e2b3e818f61fbc52a11a6f3b6c69bca54404123d698`):
SUPPORTS -- PAMS consumes skeleton/pose sequences, uses fully self-supervised
representation learning and period-adaptive multi-scale windows, and has no
multi-person identity mechanism; multi-person counting is explicitly future
work.

SITE `sections/1_introduction.tex:36`
(`a1cdac1a21a419673c557eec3b5f2248716a6a7cff8c15518ffb6e6631487066`):
SUPPORTS -- The paper learns from unlabeled skeleton sequences via PAMS-TCC
without manual repetition labels, evaluates global video resampling from 0.1x
to 2.0x, and identifies extension to multi-person counting as future work.

SITE `sections/2_related_work.tex:34`
(`0426ac52a89964d27d7c715838980245eb397a7bbfc7381b0eab379fe10d4836`):
SUPPORTS -- The source describes unlabeled skeleton learning, adaptive
multi-scale TCC, and inference-time Fast/Medium/Slow peak-detection experts
combined by voting; these are parameterized signal-processing experts, not a
learned mixture, and the method is not multi-person.

SITE `sections/3_method.tex:240`
(`6aa1db2c22104e4fb9d105e84a08a63de4ba5ddc6f2fc2c162542a685838b4d2`):
SUPPORTS -- The method explicitly uses temporally adjacent and cyclically
corresponding positives, with negatives from other temporal locations, other
videos, and different clusters.

SITE `sections/4_experiments.tex:95`
(`f85257bccb350d1360f630afedb60dbbe250eb7127eb135e476415cf86dae3c8`):
SUPPORTS -- PAMS is correctly identified as a repetitive-action-counting
baseline; adaptation to the citing work's common track interface is its own
experimental operation and need not be claimed by the source.

OVERALL: KEEP
