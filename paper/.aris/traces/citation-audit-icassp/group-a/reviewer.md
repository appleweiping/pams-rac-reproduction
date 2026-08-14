# Citation audit reviewer report — group A

## Overall verdict: FAIL

All eight requested BibTeX keys exist, and their core bibliographic metadata is supported by authoritative records. The manuscript nevertheless fails citation-context audit for two independent reasons:

1. `levy2015live` is placed under “hand-designed periodic motion analysis,” but Levy and Wolf explicitly evaluate cycle length with a convolutional neural network. Runia et al. is a hand-designed flow/wavelet method; Levy et al. is not.
2. The MultiRep metric names in the prose and table are written as generic `AP50` and `AP75`. The release-native paper names are `Period-AP50` and `Period-AP75` (alongside `Period-mAP`, `AvgMAE`, and `AvgOBO`). Because the text explicitly claims release-native definitions, the abbreviation is not source-exact unless it is defined as an alias.

No authoritative-evidence blocker remained. No bibliography or manuscript file was changed.

## Frozen input files

| Current input | Exact file SHA-256 |
|---|---|
| `paper/main.tex` | `a4fe29c47f200946c7d6f3522bd273f5895aabda2d7a959fdbf16397fe1cbe70` |
| `paper/sections/0_abstract.tex` | `fd69c7936952bffc8e70d7d0ccd35140ab7dbaac82d6ff953e60f4f6461eebef` |
| `paper/sections/1_introduction.tex` | `6b9a712638ab71e48c712f79a3eec5b2931fb0aa98ea405fb686ead0c8c0e539` |
| `paper/sections/3_method.tex` | `7e92c00eb4a7956b059536fd47240cefbea960aee61d4f3fdef1a7e24930c50d` |
| `paper/sections/4_experiments.tex` | `6013d1a197e8557b97ea08f6dc70978b4d900e14a0c9487036f343781a572b62` |
| `paper/sections/5_conclusion.tex` | `e5c656a98343f10b367c175273d572f3a42d5436539429f28ba87764452e7b3c` |
| `paper/references.bib` | `322a5d16b0078e8b3436608d4c6158b67c5c8922b0dcae94537ea456c9f6f646` |

## Exact citation-context fingerprints

| ID | Current file and lines | Exact paragraph SHA-256 | Requested keys present | Context synopsis |
|---|---|---|---|---|
| C1 | `paper/sections/1_introduction.tex:15-38` | `3579c90d9e1ebebb80a1ab7a48b93ee964b03b76ea1484fb27381cf63d3f66b4` | all 8 | Prior-work taxonomy, method summaries, and comparative gap claim. |
| C2 | `paper/sections/3_method.tex:115-144` | `8bbcef81e15da09fbafbe13bcbd58c1118f6683c636023b1d7bf47d487674cd7` | `gao2026pams` | Calls PAMS-TCC the documented starting objective; all added losses are explicitly a synchronization contract. |
| C3 | `paper/sections/4_experiments.tex:26-41` | `be45451b23b2aa122dc474dce7e1e0fc1a7c9251240e38a669745786497aee7b` | `tang2024multicounter`, `tang2026multicounterplus` | Claims release-native MultiRep metric definitions, but names `AP50`/`AP75`. |
| C4 | `paper/sections/4_experiments.tex:43-84` | `8f89f315218849ce10d27bd8ee4f68450ac132270a324ef25352211e0fc99d6a` | `gao2026pams`, `tang2024multicounter`, `tang2026multicounterplus` | Planned comparison table; generic `AP50`/`AP75` headers; manuscript-defined Track-PAMS adaptations. |
| C5 | `paper/sections/4_experiments.tex:86-101` | `fff9ed007881150db8654f27cd7a33a78fb3481cdde4946bcb1ae182d670cf49` | `dwibedi2020repnet`, `hu2022transrac`, `yao2025poserac` | Conditional baseline inclusion only if protocol parity can be established. |

No requested key occurs in `paper/main.tex`, `0_abstract.tex`, or `5_conclusion.tex`.

### Exact cite-command occurrence inventory

| Key | Exact current cite-command sites | Context fingerprint(s) |
|---|---|---|
| `levy2015live` | `paper/sections/1_introduction.tex:16` | C1 |
| `runia2018realworld` | `paper/sections/1_introduction.tex:16` | C1 |
| `dwibedi2020repnet` | `paper/sections/1_introduction.tex:18`; `paper/sections/4_experiments.tex:92` | C1, C5 |
| `hu2022transrac` | `paper/sections/1_introduction.tex:18`; `paper/sections/4_experiments.tex:92` | C1, C5 |
| `yao2025poserac` | `paper/sections/1_introduction.tex:19`; `paper/sections/4_experiments.tex:93` | C1, C5 |
| `gao2026pams` | `paper/sections/1_introduction.tex:21`; `paper/sections/3_method.tex:138`; `paper/sections/4_experiments.tex:63,73` | C1, C2, C4 |
| `tang2024multicounter` | `paper/sections/1_introduction.tex:22`; `paper/sections/4_experiments.tex:27`; `paper/sections/4_experiments.tex:56` | C1, C3, C4 |
| `tang2026multicounterplus` | `paper/sections/1_introduction.tex:24`; `paper/sections/4_experiments.tex:28`; `paper/sections/4_experiments.tex:59` | C1, C3, C4 |

## BibTeX-entry fingerprints

| Key | `paper/references.bib` lines | Exact entry SHA-256 |
|---|---:|---|
| `levy2015live` | 1-10 | `58976da59f5e0890b53319cd636f116ff1b5ccf01f49c3d72ec2dc97daa03da5` |
| `runia2018realworld` | 12-22 | `5c11a79e21b8885fa9febd6d2a06de99c1f1909960878e13913e425d55cd0f87` |
| `dwibedi2020repnet` | 24-33 | `84acc71e0713766c36b0b46e38adf48b9a95ef3d8c97d7694368070b5a2bdcab` |
| `hu2022transrac` | 35-44 | `1cb9cc0607b843b989a82c1e318a95aeddda87c8bc6008204003e9d226c41142` |
| `tang2024multicounter` | 46-56 | `6f6238de4f6e0aa79edb5d128b49d4d9752f2fadda7a33e494c447bf0e402950` |
| `yao2025poserac` | 58-68 | `191d675ad71f0345133c2f93269ea2b268f457f0a828bfa24b7fbc38c1b21b82` |
| `tang2026multicounterplus` | 81-90 | `7901027b6b434637c5a263712f2459d3320a945f10c88706b290ed8382918017` |
| `gao2026pams` | 92-100 | `0a1ddfc8761b834a311f21dcc6e5ef75abaf68b5096e38e3de2267752622be98` |

## Per-key audit

### `levy2015live` — FAIL

- Existence: **PASS**. Entry exists at `paper/references.bib:1-10`.
- Metadata: **PASS**. Ofir Levy; Lior Wolf; “Live Repetition Counting”; ICCV 2015; 3020-3028; DOI `10.1109/ICCV.2015.346`. CVF and the Crossref DOI deposit agree.
- Context C1: **FAIL**. The source says its cycle-length classifier is a CNN and calls that CNN the learned part of the system. It therefore does not support grouping Levy 2015 as “hand-designed periodic motion analysis.” Its online shifted-window/count-integration characterization is supported, but that is not the phrase currently attached to the citation.
- Evidence sites/URLs:
  - CVF HTML: https://openaccess.thecvf.com/content_iccv_2015/html/Levy_Live_Repetition_Counting_ICCV_2015_paper.html
  - CVF accepted paper: https://openaccess.thecvf.com/content_iccv_2015/papers/Levy_Live_Repetition_Counting_ICCV_2015_paper.pdf
  - Crossref DOI record: https://api.crossref.org/works/10.1109%2FICCV.2015.346

### `runia2018realworld` — PASS

- Existence: **PASS**. Entry exists at `paper/references.bib:12-22`.
- Metadata: **PASS**. Tom F. H. Runia; Cees G. M. Snoek; Arnold W. M. Smeulders; CVPR 2018; 9009-9017; DOI `10.1109/CVPR.2018.00939`.
- Context C1: **PASS**. The paper derives hand-designed gradient/divergence/curl motion signals, applies foreground motion segmentation, and uses wavelets for non-stationary repetition estimation. This supports the hand-designed periodic-motion characterization.
- Evidence sites/URLs:
  - CVF HTML: https://openaccess.thecvf.com/content_cvpr_2018/html/Runia_Real-World_Repetition_Estimation_CVPR_2018_paper.html
  - CVF accepted paper: https://openaccess.thecvf.com/content_cvpr_2018/papers/Runia_Real-World_Repetition_Estimation_CVPR_2018_paper.pdf
  - Crossref DOI record: https://api.crossref.org/works/10.1109%2FCVPR.2018.00939
  - DBLP: https://dblp.org/rec/conf/cvpr/RuniaSS18

### `dwibedi2020repnet` — PASS

- Existence: **PASS**. Entry exists at `paper/references.bib:24-33`.
- Metadata: **PASS**. Debidatta Dwibedi; Yusuf Aytar; Jonathan Tompson; Pierre Sermanet; Andrew Zisserman; CVPR 2020; DOI `10.1109/CVPR42600.2020.01040`. The bibliography's 10384-10393 pagination matches the Crossref DOI deposit and DBLP. CVF's OA landing page/PDF displays the alternate official pagination 10387-10396; this is an official-source pagination discrepancy, not a fabricated entry.
- Context C1: **PASS**. RepNet is explicitly a learned class-agnostic period predictor whose intermediate bottleneck is a temporal self-similarity matrix.
- Context C5: **PASS**. It merely identifies RepNet as a candidate baseline and makes inclusion conditional on protocol parity; no unsupported parity/result claim is made.
- Evidence sites/URLs:
  - CVF HTML: https://openaccess.thecvf.com/content_CVPR_2020/html/Dwibedi_Counting_Out_Time_Class_Agnostic_Video_Repetition_Counting_in_the_CVPR_2020_paper.html
  - CVF accepted paper: https://openaccess.thecvf.com/content_CVPR_2020/papers/Dwibedi_Counting_Out_Time_Class_Agnostic_Video_Repetition_Counting_in_the_CVPR_2020_paper.pdf
  - Crossref DOI record: https://api.crossref.org/works/10.1109%2FCVPR42600.2020.01040
  - DBLP: https://dblp.org/rec/conf/cvpr/DwibediATSZ20

### `hu2022transrac` — PASS

- Existence: **PASS**. Entry exists at `paper/references.bib:35-44`.
- Metadata: **PASS**. Huazhang Hu; Sixun Dong; Yiqun Zhao; Dongze Lian; Zhengxin Li; Shenghua Gao; CVPR 2022; DOI `10.1109/CVPR52688.2022.01843`. The bibliography's 18991-19000 pagination matches the Crossref DOI deposit. CVF's OA landing page/PDF displays the alternate official pagination 19013-19022; this is the same class of official dual-pagination discrepancy.
- Context C1: **PASS**. The source proposes a learned transformer that encodes multi-scale temporal correlation and regresses a density map.
- Context C5: **PASS**. It conditionally identifies TransRAC as a possible baseline without claiming achieved parity or a result.
- Evidence sites/URLs:
  - CVF HTML: https://openaccess.thecvf.com/content/CVPR2022/html/Hu_TransRAC_Encoding_Multi-Scale_Temporal_Correlation_With_Transformers_for_Repetitive_Action_CVPR_2022_paper.html
  - CVF accepted paper: https://openaccess.thecvf.com/content/CVPR2022/papers/Hu_TransRAC_Encoding_Multi-Scale_Temporal_Correlation_With_Transformers_for_Repetitive_Action_CVPR_2022_paper.pdf
  - Crossref DOI record: https://api.crossref.org/works/10.1109%2FCVPR52688.2022.01843

### `yao2025poserac` — PASS

- Existence: **PASS**. Entry exists at `paper/references.bib:58-68`.
- Metadata: **PASS**. The published Springer chapter is by Ziyu Yao and Yuexian Zou, titled “PoseRAC: Enhancing Repetitive Action Counting with Salient Poses,” pages 255-270, LNCS 15290, Springer Nature Singapore, publication year 2025, DOI `10.1007/978-981-96-6588-4_18`. `ICONIP 2024` in the booktitle is the conference name/year; 2025 is the proceedings publication year.
- Context C1: **PASS**. The title, Springer record, and author-official implementation all substantiate the salient-pose emphasis.
- Context C5: **PASS**. It only identifies PoseRAC as a conditional baseline candidate.
- Evidence sites/URLs:
  - Springer proceedings: https://link.springer.com/book/10.1007/978-981-96-6588-4
  - Chapter DOI: https://doi.org/10.1007/978-981-96-6588-4_18
  - Crossref DOI record: https://api.crossref.org/works/10.1007%2F978-981-96-6588-4_18
  - DBLP: https://dblp.org/rec/conf/iconip/YaoZ24
  - Author-official code: https://github.com/MiracleDance/PoseRAC

### `gao2026pams` — PASS

- Existence: **PASS**. Entry exists at `paper/references.bib:92-100`.
- Metadata: **PASS**. Shizhao Gao; Jun Li; Qiming Li; “Count What Repeats: Period-Adaptive Multi-Scale Consistency for Self-Supervised Repetitive Action Counting”; CVPR 2026 Findings; 8143-8152; June 2026. The current CVF record supplies no DOI, and an exact-title Crossref query produced no exact deposit on the audit date; omission of a DOI is therefore not a defect.
- Context C1: **PASS**. The paper explicitly learns from unlabeled skeleton/pose sequences and says the dual objective uses no ground-truth count labels; the method and objective are named PAMS and PAMS TCC.
- Context C2: **PASS**. The source explicitly documents PAMS TCC as its training objective. The manuscript carefully labels its additional route/track terms as its own synchronization contract.
- Context C4: **PASS with scope note**. “Track-PAMS” under oracle/predicted tracks is a planned manuscript adaptation, not a claim that the source paper evaluated those tracking protocols. The table has no results and does not attribute the protocol to Gao et al.
- Evidence sites/URLs:
  - CVF HTML: https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.html
  - CVF accepted paper: https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.pdf
  - CVF supplement: https://openaccess.thecvf.com/content/CVPR2026F/supplemental/Gao_Count_What_Repeats_CVPRF_2026_supplemental.pdf
  - Crossref exact-title search: https://api.crossref.org/works?query.title=Count%20What%20Repeats%3A%20Period-Adaptive%20Multi-Scale%20Consistency%20for%20Self-Supervised%20Repetitive%20Action%20Counting

### `tang2024multicounter` — FAIL

- Existence: **PASS**. Entry exists at `paper/references.bib:46-56`.
- Metadata: **PASS**. Yin Tang; Wei Luo; Jinrui Zhang; Wei Huang; Ruihai Jing; Deyu Zhang; ECAI 2024; FAIA volume 392; 242-249; IOS Press; DOI `10.3233/FAIA240494`.
- Context C1: **PASS**. The source explicitly formulates multi-instance repetitive action counting in untrimmed videos and simultaneously detects, tracks, and counts multiple human instances.
- Context C3: **FAIL (nomenclature)**. The official paper defines `Period-mAP`, `Period-AP50`, `Period-AP75`, `AvgMAE`, and `AvgOBO`; the manuscript says the definitions are release-native but writes `AP50` and `AP75` without defining them as aliases.
- Context C4: **FAIL for the same nomenclature issue**. The planned MultiCounter row is otherwise an appropriate native/contextual row and contains no fabricated result.
- Evidence sites/URLs:
  - DOI landing page: https://doi.org/10.3233/FAIA240494
  - Official open-access paper: https://journals.sagepub.com/doi/pdf/10.3233/FAIA240494
  - Crossref DOI record: https://api.crossref.org/works/10.3233%2FFAIA240494
  - ECAI accepted papers: https://www.ecai2024.eu/programme/accepted-papers
  - DBLP: https://dblp.org/rec/conf/ecai/TangLZHJZ24
  - IOS Press DOI endpoint (robots-blocked during retrieval; not needed for verdict): https://ebooks.iospress.nl/doi/10.3233/FAIA240494

### `tang2026multicounterplus` — FAIL

- Existence: **PASS**. Entry exists at `paper/references.bib:81-90`.
- Metadata: **PASS**. Yin Tang; Deyu Zhang; Wei Luo; Jinrui Zhang; Wei Huang; Ruihai Jing; Yaoxue Zhang; IEEE TCSVT 36(7), July 2026, 9462-9476; DOI `10.1109/TCSVT.2026.3670243`. IEEE Xplore, Crossref, and DBLP agree.
- Context C1: **PASS**. IEEE states that MultiCounter+ tracks multiple instances over time and uses task-specific heads with spatial-temporal consistency and long-short period awareness. This supports the manuscript's compact “identity continuity and long-versus-short repetition regimes” summary.
- Context C3: **FAIL (shared nomenclature)**. The paragraph calls the generic `AP50`/`AP75` names release-native MultiRep definitions. The source family uses Period-AP thresholds; the manuscript must either retain `Period-AP50`/`Period-AP75` or explicitly define its abbreviation.
- Context C4: **FAIL for the same shared table-header issue**. The MultiCounter+ row itself is a valid contextual placeholder and makes no performance claim.
- Evidence sites/URLs:
  - IEEE Xplore: https://ieeexplore.ieee.org/abstract/document/11421441/
  - DOI: https://doi.org/10.1109/TCSVT.2026.3670243
  - Crossref DOI record: https://api.crossref.org/works/10.1109%2FTCSVT.2026.3670243
  - DBLP: https://dblp.org/rec/journals/tcsv/TangZLZHJZ26
  - Author-official publication announcement: https://zhangjinrui1992.github.io/

## Source-access ledger and exclusions

- Authoritative verdict evidence was drawn only from `openaccess.thecvf.com`, `api.crossref.org`/`doi.org`, `dblp.org`, `link.springer.com`, the publisher-hosted IOS/SAGE DOI pages, `ecai2024.eu`, `ieeexplore.ieee.org`, and author-official GitHub/web pages.
- Direct opens of several CVF HTML pages initially returned HTTP 403, but the same official pages were available through indexed official records and their official PDFs were readable.
- IEEE's PDF text-mining/staging URL for MultiCounter+ returned HTTP 418. IEEE's official abstract/metadata page, Crossref deposit, and DBLP record were sufficient for the checked claims.
- The IOS Press DOI endpoint was robots-blocked. Its DOI-resolved official open-access paper, Crossref deposit, ECAI record, and DBLP record were sufficient.
- Third-party result aggregators, reviews, institutional mirrors, ResearchGate, and Google Scholar were not used as verdict evidence.

## Required corrections (reported only; not applied)

1. Do not characterize `levy2015live` as hand-designed. Split Levy's learned CNN-based online counter from Runia's hand-engineered wavelet/flow analysis, or rewrite the taxonomy.
2. Replace or explicitly map `AP50`/`AP75` to the source-native `Period-AP50`/`Period-AP75` in `paper/sections/4_experiments.tex:26-28` and the table header at lines 53-54.
