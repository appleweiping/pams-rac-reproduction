# ICASSP Citation Final Audit — Group B

## Overall verdict

**PASS — 8 KEEP, 0 FIX, 0 REPLACE, 0 REMOVE.** All eight cited works exist. The current BibTeX metadata matches the applicable final publisher record (or the official proceedings record where no DOI is deposited), and all 11 active citation occurrences are context-appropriate.

This is a fresh same-family review and therefore carries **provisional** acceptance status.

## Evidence policy and site-level result

Sources were re-opened on 2026-08-14. Publisher/official proceedings records were treated as canonical; Crossref and DBLP were used as independent metadata checks where they cover the work. A missing DBLP/Crossref hit was not treated as non-existence when a primary official record exists.

| Key | Primary/official evidence | Springer | IEEE | CVF | DBLP | Crossref | Result |
|---|---|---|---|---|---|---|---|
| `ma2026detrc` | IJCV version of record | exact | n/a | n/a | exact final IJCV record | exact DOI deposit | clean |
| `wang2026d2stx` | CVPR 2026 Findings open-access record | n/a | proceedings version noted by CVF, not needed for verification | exact | no exact hit at audit time | no exact deposit found | clean |
| `welch1967fft` | IEEE Xplore DOI/document and article scan | n/a | exact | n/a | no exact hit; outside useful DBLP coverage | exact DOI deposit | clean |
| `lomb1976unequal` | Springer article page | exact | n/a | n/a | n/a | exact DOI deposit | clean |
| `griffin1984stft` | IEEE Xplore DOI/document and article scan | n/a | exact | n/a | DBLP exposes the distinct 1983 ICASSP short paper, not a contradiction | exact 1984 journal DOI deposit | clean |
| `shazeer2017moe` | OpenReview ICLR record, arXiv full record | n/a | n/a | n/a | exact ICLR 2017 record | no Crossref conference DOI expected | clean |
| `luiten2021hota` | IJCV version of record | exact | n/a | n/a | exact final IJCV record | exact DOI deposit | clean |
| `ristani2016idmetrics` | Springer ECCV Workshops chapter | exact | n/a | n/a | exact 2016 CoRR record corroborates identity | exact chapter DOI deposit | clean |

IEEE Xplore returned its JavaScript verification shell in the text browser, but both IEEE DOI URLs resolved to the expected IEEE document IDs; the publisher-deposited Crossref metadata and the article headers/full text independently matched every field. OpenReview likewise presented a browser challenge, so its stable forum identifier was cross-checked against DBLP and arXiv.

## Per-key verdicts

### `ma2026detrc` — KEEP

- **Existence:** YES. Springer lists the IJCV version of record, published 25 July 2026, volume 134, issue 8, article 364, DOI `10.1007/s11263-026-02939-4`.
- **Metadata:** correct. Title, final author order (Xiaoxuan Ma; Zishi Li; Qiuyan Shang; Wentao Zhu; Hai Ci; Yu Qiao; Yizhou Wang), journal, year, volume, issue, article number, and DOI all match Springer and Crossref. DBLP's final IJCV record also agrees. The older arXiv history once exposed a different first-author order in an index record; the Bib correctly follows the publisher's final version.
- **Context:**
  - `sections/1_introduction.tex:30` — **SUPPORTS**. The paper is a repetitive-action counter built around dynamic action queries.
  - `sections/4_experiments.tex:94` — **SUPPORTS**. Describing it as dynamic-query counting and conditionally considering it as a parity-controlled baseline is faithful.
- **Sources:** [Springer version of record](https://link.springer.com/article/10.1007/s11263-026-02939-4), [Crossref record](https://api.crossref.org/works/10.1007%2Fs11263-026-02939-4), [DBLP final record](https://dblp.org/rec/journals/ijcv/MaLSZCQW26), [arXiv record](https://arxiv.org/abs/2403.01543).

### `wang2026d2stx` — KEEP

- **Existence:** YES. The Computer Vision Foundation hosts the CVPR 2026 Findings paper and its official proceedings metadata.
- **Metadata:** correct. The rendered title `D²-STX: Decoupling Spatial-Temporal Cross-Attention for Dual-branch Repetitive Action Counting`, five authors, CVPR Findings venue, June 2026, and pages 8205–8214 exactly match the CVF record. No exact DBLP or Crossref record was found on the audit date; this is a coverage/ingestion lag, not contrary evidence.
- **Context:**
  - `sections/1_introduction.tex:30` — **SUPPORTS**. The official abstract explicitly describes a dual-branch decoupled spatial-temporal cross-attention RAC framework.
  - `sections/4_experiments.tex:95` — **SUPPORTS**. “Dual-branch spatiotemporal counting” is an accurate compact description, and the manuscript makes comparison conditional on parity.
- **Source:** [CVF official open-access record](https://openaccess.thecvf.com/content/CVPR2026F/html/Wang_D2-STX_Decoupling_Spatial-Temporal_Cross-Attention_for_Dual-branch_Repetitive_Action_Counting_CVPRF_2026_paper.html).

### `welch1967fft` — KEEP

- **Existence:** YES. DOI `10.1109/TAU.1967.1161901` resolves to IEEE document 1161901.
- **Metadata:** correct. Peter D. Welch; exact title; *IEEE Transactions on Audio and Electroacoustics* 15(2), 70–73, June 1967; DOI all match Crossref's IEEE deposit and the article header.
- **Context:**
  - `sections/1_introduction.tex:34` — **SUPPORTS**. The paper sections a record, applies a window, forms modified periodograms, and averages them; it is a canonical source for windowed spectral estimation.
- **Sources:** [IEEE record](https://ieeexplore.ieee.org/document/1161901), [Crossref record](https://api.crossref.org/works/10.1109%2FTAU.1967.1161901).

### `lomb1976unequal` — KEEP

- **Existence:** YES. Springer hosts the article at DOI `10.1007/BF00648343`.
- **Metadata:** correct. N. R. (Nicholas R.) Lomb; exact title; *Astrophysics and Space Science* 39(2), 447–462 (1976); DOI all agree between Springer and Crossref.
- **Context:**
  - `sections/1_introduction.tex:34` — **SUPPORTS**. It directly establishes least-squares frequency analysis for unequally spaced data.
  - `sections/3_method.tex:88` — **SUPPORTS**. It is an appropriate example of “another irregular-sample estimator.” The citation does not claim that Lomb specifically mandates the manuscript's alternative pair-dependent-centering branch.
- **Sources:** [Springer article](https://link.springer.com/article/10.1007/BF00648343), [Crossref record](https://api.crossref.org/works/10.1007%2FBF00648343).

### `griffin1984stft` — KEEP

- **Existence:** YES. DOI `10.1109/TASSP.1984.1164317` resolves to IEEE document 1164317.
- **Metadata:** correct. Daniel W. Griffin and Jae S. Lim; exact title; *IEEE Transactions on Acoustics, Speech, and Signal Processing* 32(2), 236–243, April 1984; DOI all match Crossref's IEEE deposit and the paper header.
- **Context:**
  - `sections/1_introduction.tex:34` — **SUPPORTS** at the concept level. The paper reconstructs a signal from a modified STFT using overlapping, windowed short-time representations. It supports the broad statement that overlap-based STFT reconstruction is established, while it is not being used as evidence for the manuscript's exact masked normalized-overlap-add formula.
- **Disambiguation:** DBLP also lists a distinct 1983 ICASSP paper with the same short title and pages 804–807. The Bib entry correctly identifies the expanded 1984 IEEE journal article and must not be changed to the conference metadata.
- **Sources:** [IEEE record](https://ieeexplore.ieee.org/document/1164317), [Crossref record](https://api.crossref.org/works/10.1109%2FTASSP.1984.1164317), [article scan](https://dub.ucsd.edu/CATbox/Reader/GriffinLimMSTFT.pdf).

### `shazeer2017moe` — KEEP

- **Existence:** YES. OpenReview forum `B1ckMDqlg`, DBLP, and arXiv all identify the ICLR 2017 work.
- **Metadata:** correct. Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc V. Le, Geoffrey E. Hinton, and Jeff Dean; exact title; ICLR 2017. The lack of a conventional Crossref proceedings DOI is expected.
- **Context:**
  - `sections/1_introduction.tex:34` — **SUPPORTS**. The paper introduces a trainable gating network that selects a sparse combination of experts, directly supporting “learned expert routing” as an established idea.
- **Sources:** [OpenReview forum](https://openreview.net/forum?id=B1ckMDqlg), [DBLP ICLR record](https://dblp.org/rec/conf/iclr/ShazeerMMDLHD17), [arXiv record](https://arxiv.org/abs/1701.06538).

### `luiten2021hota` — KEEP

- **Existence:** YES. Springer hosts the IJCV version of record at DOI `10.1007/s11263-020-01375-2`.
- **Metadata:** correct. All seven authors, title, IJCV 129(2), 548–578, issue year 2021, and DOI agree across Springer, DBLP, and Crossref. Online publication in October 2020 does not make the Bib's 2021 issue year wrong.
- **Context:**
  - `sections/4_experiments.tex:17` — **SUPPORTS**. HOTA is explicitly proposed as a multi-object-tracking evaluation metric balancing detection, association, and localization; using it as a tracking-frontend diagnostic is appropriate.
- **Sources:** [Springer article](https://link.springer.com/article/10.1007/s11263-020-01375-2), [Crossref record](https://api.crossref.org/works/10.1007%2Fs11263-020-01375-2), [DBLP final record](https://dblp.org/rec/journals/ijcv/LuitenODTGLL21).

### `ristani2016idmetrics` — KEEP

- **Existence:** YES. Springer publishes the ECCV 2016 Workshops chapter at DOI `10.1007/978-3-319-48881-3_2`.
- **Metadata:** correct. The five authors, title, editors Gang Hua and Hervé Jégou, LNCS volume 9914, pages 17–35, Springer Cham, 2016, DOI, and online ISBN all match Springer and Crossref. Springer uses “Roger Zou”; DBLP/arXiv expands the middle initial, which is not a conflict with the publisher-form Bib.
- **Context:**
  - `sections/4_experiments.tex:18` — **SUPPORTS**. The work defines ID precision, ID recall, and IDF1 and analyzes identity mismatches/fragmentations and switch behavior, so it supports the diagnostic list. It is the primary source for IDF1; if a future sentence specifically claimed that it introduced the conventional ID-switch count, that narrower historical claim would instead need the original CLEAR MOT source.
- **Sources:** [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-319-48881-3_2), [Crossref record](https://api.crossref.org/works/10.1007%2F978-3-319-48881-3_2), [DBLP record](https://dblp.org/rec/journals/corr/RistaniSZCT16), [arXiv record](https://arxiv.org/abs/1609.01775).

## Final ledger

| Key | Existence | Metadata | Context occurrence verdicts | Entry verdict |
|---|---|---|---|---|
| `ma2026detrc` | YES | correct | SUPPORTS, SUPPORTS | KEEP |
| `wang2026d2stx` | YES | correct | SUPPORTS, SUPPORTS | KEEP |
| `welch1967fft` | YES | correct | SUPPORTS | KEEP |
| `lomb1976unequal` | YES | correct | SUPPORTS, SUPPORTS | KEEP |
| `griffin1984stft` | YES | correct | SUPPORTS | KEEP |
| `shazeer2017moe` | YES | correct | SUPPORTS | KEEP |
| `luiten2021hota` | YES | correct | SUPPORTS | KEEP |
| `ristani2016idmetrics` | YES | correct | SUPPORTS | KEEP |

No manuscript or BibTeX mutation is recommended.

