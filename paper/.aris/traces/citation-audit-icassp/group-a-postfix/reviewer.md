# ICASSP citation post-fix audit — group A

## 总体结论

**Verdict: PASS WITH NOTES — provisional / same-family.**

八个 citation key 均指向真实且可定位的论文；当前 Bib 的核心元数据与出版社沉积、正式索引或官方开放论文页一致。活动 TeX 中的主要语义归因均有原文依据。未发现需要阻断提交的不存在论文、错配 DOI、错配作者/题名、或把相反结论归给来源的情况。

保留三项非阻断说明：

1. RepNet 和 TransRAC 的 CVF 接收版页面/PDF页码分别显示 `10387–10396`、`19013–19022`，而 IEEE 出版社向 Crossref 沉积且 DBLP 收录的正式出版页码分别为 `10384–10393`、`18991–19000`。当前 Bib 采用后者，判为正确；不能仅按 CVF 页码把它“修回去”。
2. `gao2026pams` 的 source-native 范围是基于骨架/pose sequence 的自监督 PAMS；正文中的 `Track-PAMS`、oracle/predicted-track 协议和 MultiRep 指标套用是稿件拟议适配，不是 Gao 等人的原生实验。当前稿件用 `pending` 单元格、parity 条件及 “documented starting objective” 明确限制了归因，因此不构成误引。
3. MultiCounter+ 原文术语是 “spatial-temporal consistency” 与 “long-short period awareness”。导言把前者概括为 “identity continuity” 属于合理但稍具解释性的转述，而不是论文原句。

## 输入与上下文锁定

本次只读取了当前活动链：`paper/main.tex`、其活动 section 中与八键有关的上下文，以及当前 `paper/references.bib`；未读取旧 audit/history/log。

| 文件 | SHA-256 | 状态 |
|---|---|---|
| `paper/sections/1_introduction.tex` | `6823651162f59e5f5263949f2d308f594b667cb3cba0044433c39031ae04db9d` | 与任务锁一致 |
| `paper/sections/4_experiments.tex` | `6052c816e94f7d7f50b560d397495a37863b05d240ed5e418e4b6ec4d3aa131b` | 与任务锁一致 |
| `paper/references.bib` | `322a5d16b0078e8b3436608d4c6158b67c5c8922b0dcae94537ea456c9f6f646` | 当前审计输入 |
| `paper/main.pdf` | `873c83da0dbf4b24a6e1199f3dcd980e17d0b032580e0591f88a7c545ec67a45` | 与任务锁一致 |

活动引用上下文位置：导言第 15–24 行；方法第 138–140 行；实验第 26–32、54–64、74、92–96 行。

## 重点措辞复核

### Early-work wording

**PASS.** “Early work addressed live and real-world repetition counting” 是有限定、非排他的历史概括，并未声称两篇是绝对最早工作。

- Levy–Wolf 2015 的题名即 *Live Repetition Counting*，摘要明确说明方法在线处理 live video stream，并处理 real-world video；[CVF official page](https://openaccess.thecvf.com/content_iccv_2015/html/Levy_Live_Repetition_Counting_ICCV_2015_paper.html)。
- Runia–Snoek–Smeulders 2018 的题名即 *Real-World Repetition Estimation by Div, Grad and Curl*，摘要明确以非静态、非平稳真实视频为目标；[CVF official page](https://openaccess.thecvf.com/content_cvpr_2018/html/Runia_Real-World_Repetition_Estimation_CVPR_2018_paper.html)。
- 年序 2015/2018 早于紧接着列出的 RepNet 2020 与 TransRAC 2022，因此 “Early work ...; later ...” 的局部时间关系成立。

### MultiRep metric nomenclature

**PASS.** MultiCounter 正式论文 Table 1 的逐字表头为：

`Period-mAP`, `Period-AP50`, `Period-AP75`, `AvgMAE`, `AvgOBO`。

论文指标段落把统称写作 `Period-AP`，并报告 temporal sIoU `50%`、`75%`、以及 `50%–95%`（步长 5%）；对应表格中的 `Period-AP50`、`Period-AP75`、`Period-mAP`。因此活动实验表与叙述中的三个 Period 名称完全匹配正式表头。正文段落另写 `Avg-MAE`/`Avg-OBO`，但正式表头与摘要使用无连字符的 `AvgMAE`/`AvgOBO`，故稿件当前拼法同样成立。[IOS/SAGE official-version PDF](https://journals.sagepub.com/doi/pdf/10.3233/FAIA240494)

MultiCounter+ 是同一 MultiRep/MRAC 方法族的扩展；IEEE 官方摘要确认它在 MultiRep 上评估并延续多实例检测、跟踪、计数框架。由于本次无法从 IEEE 公共摘要页逐字抽取其完整指标表，MultiCounter+ 对这组指标的复核以正式 MultiCounter 定义和同族延续为依据；这正是本审计保持 `provisional` 的一项原因。[IEEE Xplore](https://ieeexplore.ieee.org/document/11421441)

## 逐 key 审计

### `levy2015live` — PASS

- **存在性/元数据：** CVF、DBLP、IEEE/Crossref 一致确认 Ofir Levy、Lior Wolf，ICCV 2015，pp. 3020–3028，DOI `10.1109/ICCV.2015.346`。当前 Bib 匹配。
- **上下文：** “live” 由题名和在线处理摘要直接支持；列作 early work 合理。
- **来源：** [CVF](https://openaccess.thecvf.com/content_iccv_2015/html/Levy_Live_Repetition_Counting_ICCV_2015_paper.html), [DBLP](https://dblp.org/rec/conf/iccv/LevyW15), [Crossref API](https://api.crossref.org/works/10.1109%2FICCV.2015.346)。

### `runia2018realworld` — PASS

- **存在性/元数据：** CVF、DBLP、IEEE/Crossref 一致确认 Tom F. H. Runia、Cees G. M. Snoek、Arnold W. M. Smeulders，CVPR 2018，pp. 9009–9017，DOI `10.1109/CVPR.2018.00939`。当前 Bib 匹配。
- **上下文：** 题名和摘要直接支撑 “real-world repetition counting/estimation”；列作 early work 合理。
- **来源：** [CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Runia_Real-World_Repetition_Estimation_CVPR_2018_paper.html), [DBLP](https://dblp.org/rec/conf/cvpr/RuniaSS18), [Crossref API](https://api.crossref.org/works/10.1109%2FCVPR.2018.00939)。

### `dwibedi2020repnet` — PASS

- **存在性/元数据：** 作者、题名、CVPR 2020、DOI `10.1109/CVPR42600.2020.01040` 均一致。IEEE/Crossref 与 DBLP 给出正式页码 10384–10393，当前 Bib 匹配。
- **上下文：** 原文明确把 temporal self-similarity matrix 作为中间表示瓶颈并由其预测周期；“later temporal similarity ... models include RepNet” 准确。实验段只在输入/输出/跟踪/评测 parity 可证明时纳入，未过度声称可直接比较。
- **页码说明：** CVF accepted-version PDF 印为 10387–10396；这是源间版本差异，不推翻出版社沉积页码。
- **来源：** [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Dwibedi_Counting_Out_Time_Class_Agnostic_Video_Repetition_Counting_in_the_CVPR_2020_paper.html), [DBLP](https://dblp.org/rec/conf/cvpr/DwibediATSZ20), [Crossref API](https://api.crossref.org/works/10.1109%2FCVPR42600.2020.01040)。

### `hu2022transrac` — PASS

- **存在性/元数据：** 作者、题名、CVPR 2022、DOI `10.1109/CVPR52688.2022.01843` 均一致。IEEE/Crossref 与 DBLP 给出正式页码 18991–19000，当前 Bib 匹配。
- **上下文：** 原文明确提出 multi-scale temporal correlation with transformers；导言的 “temporal ... correlation models include ... TransRAC” 直接受支持。实验中的 parity 限制同样避免跨协议硬比较。
- **页码说明：** CVF accepted-version 页面/PDF 为 19013–19022；当前 Bib 采用正式出版索引页码是可辩护的正确选择。
- **来源：** [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Hu_TransRAC_Encoding_Multi-Scale_Temporal_Correlation_With_Transformers_for_Repetitive_Action_CVPR_2022_paper.html), [DBLP](https://dblp.org/rec/conf/cvpr/HuDZLLG22), [Crossref API](https://api.crossref.org/works/10.1109%2FCVPR52688.2022.01843)。

### `yao2025poserac` — PASS

- **存在性/元数据：** Springer/Crossref 确认 Ziyu Yao、Yuexian Zou，题名 *PoseRAC: Enhancing Repetitive Action Counting with Salient Poses*，LNCS 15290，pp. 255–270，DOI `10.1007/978-981-96-6588-4_18`，出版社年份 2025。DBLP 按会议年记作 ICONIP (5) 2024；这是会议年与 Springer 出版年差异，当前 Bib 用 2025 并在 booktitle 保留 ICONIP 2024，准确且信息充分。
- **上下文：** 原文摘要反复使用 salient pose 概念；“PoseRAC emphasizes salient poses” 是直接概括。
- **消歧：** 此 DOI 对应两作者的 ICONIP 论文，不应与三作者的 2023 arXiv *Pose Saliency Transformer* 记录混为一条。当前 Bib 没有混淆。
- **来源：** [Springer DOI](https://doi.org/10.1007/978-981-96-6588-4_18), [Peking University author-hosted paper](https://web.pkusz.edu.cn/adsp/files/2025/09/%E5%A7%9A%E5%AD%90%E8%A3%95-ICONIP2024-PoseRAC-Enhancing-Repetitive-Action-Counting-with-Salient-Poses.pdf), [DBLP](https://dblp.org/rec/conf/iconip/YaoZ24), [Crossref API](https://api.crossref.org/works/10.1007%2F978-981-96-6588-4_18)。

### `gao2026pams` — PASS WITH SCOPE NOTE

- **存在性/元数据：** CVF 官方页确认 Shizhao Gao、Jun Li、Qiming Li，CVPR Findings 2026，pp. 8143–8152；当前题名、作者、venue、年份、页码与 URL 匹配。未在 Crossref 精确题名检索中定位到 DOI 记录，当前 Bib 不虚构 DOI，处理正确。
- **上下文：** 官方摘要明确称其从 unlabeled skeleton sequences 学习 Period-Adaptive Multi-Scale Temporal Cycle Consistency，并且 fully self-supervised、without manual labels；因此“从 pose sequences 学习 period-adaptive multi-scale consistency，counting objective 不用 count labels”得到支持。方法段的 `PAMS-TCC` 是对原文 `PAMS TCC` 的排版变体。
- **范围：** 原论文不提供当前稿件的 `Track-PAMS` oracle/predicted-track 或 MultiRep 评测协议。稿件目前把它限定为 adapted starting objective、pending cells 和 parity gate，归因边界尚清楚。
- **来源：** [CVF official page](https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.html), [CVF paper PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.pdf)。

### `tang2024multicounter` — PASS

- **存在性/元数据：** IOS Press/SAGE 正式论文、DBLP 与 Crossref 一致确认作者、题名、ECAI 2024、FAIA volume 392、pp. 242–249、DOI `10.3233/FAIA240494`。当前 Bib 匹配；`booktitle = {ECAI 2024}` 是可接受的正式会议简称。
- **上下文：** 原文正式提出 Multi-instance Repetitive Action Counting，并在 untrimmed videos 中同时检测、跟踪、计数多个人类实例；“establishes multi-person counting in untrimmed video” 有直接依据。
- **指标：** 正式 Table 1 精确使用当前稿件的五个指标表头；指标定义段说明阈值与聚合范围。
- **来源：** [official-version PDF](https://journals.sagepub.com/doi/pdf/10.3233/FAIA240494), [DBLP](https://dblp.org/rec/conf/ecai/TangLZHJZ24), [Crossref API](https://api.crossref.org/works/10.3233%2FFAIA240494), [IOS volume 392](https://ebooks.iospress.nl/volumearticle/69553)。

### `tang2026multicounterplus` — PASS WITH WORDING NOTE

- **存在性/元数据：** IEEE Xplore、DBLP 与 Crossref 一致确认七位作者、题名、TCSVT volume 36 issue 7、pp. 9462–9476、2026、DOI `10.1109/TCSVT.2026.3670243`。当前 Bib 完全匹配。
- **上下文：** IEEE 摘要明确 MultiCounter+ 同时检测、跟踪、计数多实例，使用 spatial-temporal consistency 与 long-short period awareness。导言的 “long-versus-short repetition regimes” 直接对应；“identity continuity” 是对跟踪与时空一致性的解释性概括，语义可接受但不是原文术语。
- **指标：** IEEE 摘要确认在 MultiRep 上实验；完整三阈值表头的逐字核验主要由其前身 MultiCounter 正式论文完成。因此本条维持 provisional，而非声称已独立抽取 IEEE 全文表格。
- **来源：** [IEEE Xplore](https://ieeexplore.ieee.org/document/11421441), [DBLP](https://dblp.org/rec/journals/tcsvt/TangZLZHJZ26), [Crossref API](https://api.crossref.org/works/10.1109%2FTCSVT.2026.3670243)。

## 最终判定

- 书目存在性：`8/8 PASS`。
- 核心元数据：`8/8 PASS`；RepNet/TransRAC 仅存在已解释的 accepted-version 与正式出版页码差异。
- 当前引用语境：`6 PASS + 2 PASS WITH NOTE`（PAMS 适配范围；MultiCounter+ 的 identity-continuity 转述）。
- early-work wording：`PASS`。
- `Period-mAP / Period-AP50 / Period-AP75` nomenclature：`PASS`。
- 总体：`PASS WITH NOTES`，但由于审计者为 same-family 且 MultiCounter+ 完整指标表未能从公共 IEEE 摘要页逐字抽取，状态保持 `provisional`。
