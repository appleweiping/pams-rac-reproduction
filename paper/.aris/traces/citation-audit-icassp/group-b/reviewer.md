# Citation audit B

Overall verdict: **PASS**

All eight requested keys exist. Their current BibTeX metadata agrees with the
authoritative publication record, and every current in-text use is supported.
No bibliography or manuscript change is required by this audit.

## Scope and reproducibility

- Audit time: 2026-08-14 (Asia/Shanghai).
- Evidence was collected independently from the current bibliography and TeX
  source plus the URLs listed below. No prior audit, trace, history, or log was
  read.
- Search/discovery output was not treated as evidence. Google Scholar was not
  used.
- Whole-file hashes are raw-byte SHA-256. BibTeX-entry, citation-line, and
  paragraph hashes are SHA-256 of UTF-8 text after CRLF/CR is normalized to LF;
  entry/line/paragraph text has no terminal newline in the hashed payload.

## Current input hashes

| Input | Raw-byte SHA-256 |
|---|---|
| `paper/references.bib` | `322a5d16b0078e8b3436608d4c6158b67c5c8922b0dcae94537ea456c9f6f646` |
| `paper/main.tex` | `a4fe29c47f200946c7d6f3522bd273f5895aabda2d7a959fdbf16397fe1cbe70` |
| `paper/sections/0_abstract.tex` | `fd69c7936952bffc8e70d7d0ccd35140ab7dbaac82d6ff953e60f4f6461eebef` |
| `paper/sections/1_introduction.tex` | `6b9a712638ab71e48c712f79a3eec5b2931fb0aa98ea405fb686ead0c8c0e539` |
| `paper/sections/3_method.tex` | `7e92c00eb4a7956b059536fd47240cefbea960aee61d4f3fdef1a7e24930c50d` |
| `paper/sections/4_experiments.tex` | `6013d1a197e8557b97ea08f6dc70978b4d900e14a0c9487036f343781a572b62` |
| `paper/sections/5_conclusion.tex` | `e5c656a98343f10b367c175273d572f3a42d5436539429f28ba87764452e7b3c` |

## Exact current citation locations

| Key(s) | Exact citation line(s), line SHA-256 | Paragraph range, paragraph SHA-256 | Supported claim |
|---|---|---|---|
| `ma2026detrc`, `wang2026d2stx` | `paper/sections/1_introduction.tex:30`, `67793971113c32106a69b9c1e27156761319226164c82248a18d0eb424c2baea` | lines 15–38, `3579c90d9e1ebebb80a1ab7a48b93ee964b03b76ea1484fb27381cf63d3f66b4` | Recent counters use dynamic queries and decoupled spatiotemporal attention. |
| `welch1967fft`, `lomb1976unequal`, `griffin1984stft`, `shazeer2017moe` | `paper/sections/1_introduction.tex:34`, `b96b61a12dc0b3e57927f5bff641898701a8493ddd6c37bbebfa5cd3c28ce5df` | lines 15–38, `3579c90d9e1ebebb80a1ab7a48b93ee964b03b76ea1484fb27381cf63d3f66b4` | Windowed spectral estimation, irregular-sample frequency analysis, overlap-add reconstruction, and learned expert routing are established ideas. |
| `lomb1976unequal` | `paper/sections/3_method.tex:88`, `30eabe01f1f7598b4af5ba4061839e5d355fd254d9a2d25b99951dd3b3da2237` | lines 79–88, `1ae7739fc763285defff9012fd62f78d3460dd0637e0a1755dae208786989826` | An irregular-sample frequency estimator is an alternative when handling masked/irregular observations. |
| `luiten2021hota` | `paper/sections/4_experiments.tex:17`, `503de37c46d88fbe7156f9ad634d8ef915ca3b306273c500be1840e114162f5b` | lines 11–24, `a4d0d77c00c413c2209f01c256f7213a0b8f9fca658f704372b6d52fc8d6323f` | HOTA is retained as a predicted-track/frontend diagnostic. |
| `ristani2016idmetrics` | `paper/sections/4_experiments.tex:18`, `f768b0b6f195123b4e063011214e362f7e1509a4282379e69e4199642498f6dd` | lines 11–24, `a4d0d77c00c413c2209f01c256f7213a0b8f9fca658f704372b6d52fc8d6323f` | IDF1 and identity switches are predicted-track/frontend diagnostics. |
| `ma2026detrc` | `paper/sections/4_experiments.tex:93`, `ebc813f68d8d7dbedc2af3098b35860ed53f9f4b22b66b5b3a3aa1371b741d10` | lines 87–101, `0d1367169af7e39ecc70b4d7cdb697fbee5be208caa32e18e8551fea2815b30f` | The cited baseline is dynamic-query action counting. |
| `wang2026d2stx` | `paper/sections/4_experiments.tex:94`, `eb802b4dbcf8d4b6ccabd00bd3ff5e37e9df13f631c3d0e0a398e740ca9a53ef` | lines 87–101, `0d1367169af7e39ecc70b4d7cdb697fbee5be208caa32e18e8551fea2815b30f` | The cited baseline is dual-branch spatiotemporal action counting. |

No requested key occurs in `paper/main.tex`, the abstract, or the conclusion.

## Per-key findings

### `ma2026detrc` — PASS

- Current entry: `paper/references.bib:70–79`; entry SHA-256
  `da4ad389f8f344114d236f9031e102a83f6fd2145782b97de837b4879f272f03`.
- Existence: **PASS**. Springer records a version of record published 25 July
  2026.
- Metadata: **PASS**. Xiaoxuan Ma, Zishi Li, Qiuyan Shang, Wentao Zhu, Hai Ci,
  Yu Qiao, and Yizhou Wang; “Efficient Action Counting with Dynamic Queries”;
  *International Journal of Computer Vision* 134(8), article 364 (2026); DOI
  `10.1007/s11263-026-02939-4`. Crossref also confirms issue 8 and article 364.
- Context: **PASS** at introduction line 30 and experiments line 93. The title
  and abstract explicitly describe dynamic action queries for repetitive action
  counting. The manuscript's statement that this architectural direction is
  complementary is its own scoped comparison, not a claim attributed to the
  source.
- Sources:
  - https://link.springer.com/article/10.1007/s11263-026-02939-4
  - https://api.crossref.org/works/10.1007%2Fs11263-026-02939-4

### `wang2026d2stx` — PASS

- Current entry: `paper/references.bib:102–110`; entry SHA-256
  `71364a3fb14903a04223827200eb8f909ce00b4aad979082e91c5dd33d4a2bca`.
- Existence: **PASS**. The paper is in the official CVF CVPR 2026 Findings
  repository.
- Metadata: **PASS**. Xiaoai Wang, Hang Wang, Yan Liu, Huan Hu, and Bruce X.B.
  Yu; “D²-STX: Decoupling Spatial-Temporal Cross-Attention for Dual-branch
  Repetitive Action Counting”; *Proceedings of the IEEE/CVF Conference on
  Computer Vision and Pattern Recognition (CVPR) Findings*, pages 8205–8214
  (2026). No DOI is supplied by the current entry or needed to identify this
  official proceedings item.
- Context: **PASS** at introduction line 30 and experiments line 94. The
  official title and abstract explicitly identify a dual-branch, decoupled
  spatial-temporal cross-attention RAC framework.
- Source:
  - https://openaccess.thecvf.com/content/CVPR2026F/html/Wang_D2-STX_Decoupling_Spatial-Temporal_Cross-Attention_for_Dual-branch_Repetitive_Action_Counting_CVPRF_2026_paper.html

### `welch1967fft` — PASS

- Current entry: `paper/references.bib:138–148`; entry SHA-256
  `a0976b6fec6f4ac95b2c6491eb6b09ab18c995d76ef5ed81635b42daa0cec528`.
- Existence: **PASS**. IEEE Xplore document 1161901 and Crossref resolve the
  DOI.
- Metadata: **PASS**. Peter D. Welch; “The Use of Fast Fourier Transform for
  the Estimation of Power Spectra: A Method Based on Time Averaging Over Short,
  Modified Periodograms”; *IEEE Transactions on Audio and Electroacoustics*
  15(2), 70–73 (June 1967); DOI `10.1109/TAU.1967.1161901`.
- Context: **PASS** at introduction line 34. Sectioning a record into short
  segments, modifying/windowing their periodograms, and time-averaging the
  estimates is precisely windowed spectral estimation.
- Sources:
  - https://ieeexplore.ieee.org/document/1161901
  - https://api.crossref.org/works/10.1109%2FTAU.1967.1161901

### `lomb1976unequal` — PASS

- Current entry: `paper/references.bib:150–159`; entry SHA-256
  `9f7454e98b20c7375a262ec12ba84c00644f654a55ffa05b4a453aa6b11fac75`.
- Existence: **PASS**. Springer hosts the article under the cited DOI.
- Metadata: **PASS**. N. R. Lomb; “Least-Squares Frequency Analysis of
  Unequally Spaced Data”; *Astrophysics and Space Science* 39(2), 447–462
  (1976); DOI `10.1007/BF00648343`.
- Context: **PASS** at introduction line 34 and method line 88. The paper
  directly studies least-squares frequency analysis of unequally spaced data.
  In the method paragraph, the citation supports “another irregular-sample
  estimator”; it is not presented as evidence for the separately offered
  pair-dependent-centering choice.
- Sources:
  - https://link.springer.com/article/10.1007/BF00648343
  - https://api.crossref.org/works/10.1007%2FBF00648343

### `griffin1984stft` — PASS

- Current entry: `paper/references.bib:161–171`; entry SHA-256
  `df69974800faaa77f3569c2118d384846720478767b5a25d26022f55bac5a25c`.
- Existence: **PASS**. IEEE Xplore and Crossref resolve the journal article.
- Metadata: **PASS**. Daniel W. Griffin and Jae S. Lim; “Signal Estimation from
  Modified Short-Time Fourier Transform”; *IEEE Transactions on Acoustics,
  Speech, and Signal Processing* 32(2), 236–243 (April 1984); DOI
  `10.1109/TASSP.1984.1164317`.
- Context: **PASS** at introduction line 34. The paper derives STFT signal
  reconstruction/estimation, explicitly relates its solution to standard and
  weighted overlap-add procedures, and discusses overlap-add implementation.
  It is not the earliest overlap-add paper, but the manuscript only calls the
  idea established and does not claim Griffin and Lim invented it.
- Sources:
  - https://ieeexplore.ieee.org/document/1164317
  - https://doi.org/10.1109/TASSP.1984.1164317
  - https://api.crossref.org/works/10.1109%2FTASSP.1984.1164317
  - https://dub.ucsd.edu/CATbox/Reader/GriffinLimMSTFT.pdf (scan used to inspect the paper's overlap-add discussion; IEEE/Crossref are the metadata authorities)

### `shazeer2017moe` — PASS

- Current entry: `paper/references.bib:173–179`; entry SHA-256
  `4f273838b336c3e3600311a56ddac9a7e6547f9b83af8af2272b3f519962a887`.
- Existence: **PASS**. The official OpenReview identifier is valid, and the
  authors' Google Research publication page records it as ICLR 2017.
- Metadata: **PASS**. Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy
  Davis, Quoc V. Le, Geoffrey E. Hinton, and Jeff Dean; “Outrageously Large
  Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer”; ICLR 2017;
  OpenReview `B1ckMDqlg`. A DOI is not expected for this item.
- Context: **PASS** at introduction line 34. The abstract states that a
  trainable gating network selects a sparse combination of feed-forward
  experts, directly supporting “learned expert routing.”
- Sources:
  - https://openreview.net/forum?id=B1ckMDqlg
  - https://research.google/pubs/outrageously-large-neural-networks-the-sparsely-gated-mixture-of-experts-layer/

### `luiten2021hota` — PASS

- Current entry: `paper/references.bib:112–121`; entry SHA-256
  `682ff790cd11649ec005de1b50e045d34b135ea92b76aeb3a8de3ac5f49b9e31`.
- Existence: **PASS**. Springer hosts the open-access version of record.
- Metadata: **PASS**. Jonathon Luiten, Aljoša Ošep, Patrick Dendorfer, Philip
  Torr, Andreas Geiger, Laura Leal-Taixé, and Bastian Leibe; “HOTA: A Higher
  Order Metric for Evaluating Multi-object Tracking”; *International Journal of
  Computer Vision* 129(2), 548–578 (2021); DOI
  `10.1007/s11263-020-01375-2`. The 2020 DOI/publication-online date and 2021
  issue year are consistent; Springer itself cites the journal item as 2021.
- Context: **PASS** at experiments line 17. The source introduces HOTA as an
  MOT evaluation metric balancing detection, association, and localization, so
  using it as a predicted-track/frontend diagnostic is supported.
- Sources:
  - https://link.springer.com/article/10.1007/s11263-020-01375-2
  - https://api.crossref.org/works/10.1007%2Fs11263-020-01375-2

### `ristani2016idmetrics` — PASS

- Current entry: `paper/references.bib:123–136`; entry SHA-256
  `5953f0c8a841cf28f6931797760929fd4d153791d0b8907545561e2a881bc60e`.
- Existence: **PASS**. Springer records the ECCV 2016 Workshops chapter.
- Metadata: **PASS**. Ergys Ristani, Francesco Solera, Roger Zou, Rita
  Cucchiara, and Carlo Tomasi; “Performance Measures and a Data Set for
  Multi-Target, Multi-Camera Tracking”; in Gang Hua and Hervé Jégou (eds.),
  *Computer Vision – ECCV 2016 Workshops*, LNCS 9914, 17–35, Springer, Cham
  (2016); DOI `10.1007/978-3-319-48881-3_2`; online ISBN
  `978-3-319-48881-3`.
- Context: **PASS** at experiments line 18. The paper proposes ID precision and
  recall and reports/analyses IDF1; it also directly discusses ID-switch counts
  as an existing event-based diagnostic and contrasts them with identity-based
  measures. The manuscript does not claim that Ristani et al. invented identity
  switches, so this combined diagnostic citation is supported.
- Sources:
  - https://link.springer.com/chapter/10.1007/978-3-319-48881-3_2
  - https://arxiv.org/abs/1609.01775
  - https://api.crossref.org/works/10.1007%2F978-3-319-48881-3_2

## Site accounting

Evidence authorities used: `link.springer.com`, `api.crossref.org`,
`openaccess.thecvf.com`, `ieeexplore.ieee.org`, `openreview.net`,
`research.google`, and `arxiv.org`. A scan of the Griffin–Lim article at
`dub.ucsd.edu` was used only to inspect the original paper text; metadata was
checked against IEEE and Crossref. Discovery-only results from `cir.nii.ac.jp`
were not used for any verdict. An attempted Welch PDF URL at `utdallas.edu`
returned HTML rather than a paper and was not used. No Google Scholar evidence
was used.

## Final disposition

**PASS.** Existence 8/8 PASS; metadata 8/8 PASS; current citation contexts 8/8
PASS. No blocking uncertainty and no requested corrective edit.
