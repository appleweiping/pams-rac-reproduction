# ARIS Idea Discovery Report

**Direction:** Pose-driven multi-person repetitive action counting under
non-stationary per-person tempo

**Venue:** ICASSP 2027

**Run:** `rac-temporac-idea-20260815-sol`

**Reviewer policy:** GPT-5.6-Sol only; same-family judgments remain provisional

**Current phase:** literature landscape complete; candidate jury, novelty
adjudication, research review, and refinement are still running

## Literature Landscape

### 检索范围与可复核性

本轮按 `research-lit` 的 composed 模式执行，检索了 MRAC、pose/skeleton
RAC、无 count-label 学习、局部与非平稳节奏、时序响应解码、tracking
一致性、Mixture-of-Experts 和 overlap-add。检索采用至少五组不同查询，
技术结论仅使用作者原始论文、CVF/IEEE/Springer/IOS/SSRN 出版页面、DOI
记录或官方代码。Google Scholar/普通搜索结果只用于发现候选，不作为元数据
或 claim 的最终依据。

`papers/` 与 `literature/` 均不存在，因此本地论文库没有贡献；当前仓库中的
PDF 是稿件构建产物，不是可当作文献库使用的来源。八个 arXiv 候选由 ARIS
`verify_papers.py` 机械核验，结果为 `PASS`，hallucination rate 和 pending rate
均为 0。核验输出见 `idea-stage/paper_existence_verification.json`。

### 直接相关工作

| 工作 | 已核实的方法边界 | 对本项目的约束 |
|---|---|---|
| MultiCounter, ECAI 2024, [DOI](https://doi.org/10.3233/FAIA240494) | 以 instance queries、Mixed Spatial-Temporal Interaction 和 instance/period heads 联合检测、跟踪、定位周期边界并逐人计数；提出 MultiRep 与 Period-AP | 已正式定义 MRAC，覆盖多人异步与变周期；不得声称首次多人、首次逐人输出或首次异步/变速 RAC |
| MultiCounter+, TCSVT 2026, [DOI](https://doi.org/10.1109/TCSVT.2026.3670243) | 监督式统一 MRAC，加入 spatial-temporal consistency、long-short period awareness 和大规模合成预训练 | 已覆盖身份一致性与长短周期；本项目只能在无逐人 count/period 标注、pose-driven、局部窗口路由与逐轨重建的交集上区分 |
| PAMS, CVPR Findings 2026, [CVF](https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.html) | 在单目标 skeleton 序列上用 PAMS-TCC 与跨视频判别进行自监督学习；推理阶段的“experts”是不同 peak/window/smoothing 参数的 consensus | 已占据 pose、自监督、period-adaptive、multi-scale 与全视频变速测试；本项目的专家必须明确是学习型局部响应函数，且必须证明同轨迹内非平稳节奏 |
| PoseRAC arXiv 2023, [arXiv](https://arxiv.org/abs/2303.08450) | Pose Saliency Transformer，使用 salient-pose annotations 与监督训练 | 已占据 pose-level/salient-pose RAC；它与后续 ICONIP 同名工作不能混写 |
| PoseRAC ICONIP, LNCS 2025, [DOI](https://doi.org/10.1007/978-981-96-6588-4_18) | 使用 foundation generative model 与预训练 text encoder 支撑 zero-shot/open-set 叙事 | zero-shot 不等于无外部监督；该工作仍是单目标，不覆盖 per-identity local tempo routing |
| D²-STX, CVPR Findings 2026, [CVF](https://openaccess.thecvf.com/content/CVPR2026F/html/Wang_D2-STX_Decoupling_Spatial-Temporal_Cross-Attention_for_Dual-branch_Repetitive_Action_Counting_CVPRF_2026_paper.html) | RGB/pose 双分支与细粒度时空 cross-attention 融合 | 不得声称首次利用 pose structure、双分支或时空解耦融合 |
| DeTRC, IJCV 2026, [DOI](https://doi.org/10.1007/s11263-026-02939-4) | 以动态 action queries 定位候选周期并用 inter-query contrastive learning 区分重复与背景；线性复杂度 | dynamic queries 不等于 identity-conditioned tempo routing；不得误写 constant-time |
| RepNet, CVPR 2020, [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Dwibedi_Counting_Out_Time_Class_Agnostic_Video_Repetition_Counting_in_the_CVPR_2020_paper.html) | temporal self-similarity bottleneck，逐帧预测 period 与 periodicity，以合成重复片段及其目标训练 | 已能给出时变 per-frame period；不是 PAMS 意义上的自监督，也不是 MRAC |
| TransRAC, CVPR 2022, [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Hu_TransRAC_Encoding_Multi-Scale_Temporal_Correlation_With_Transformers_for_Repetitive_Action_CVPR_2022_paper.html) | 多尺度 temporal-correlation Transformer 与 density-map regression，使用细粒度 cycle annotations | 已占据监督式多尺度频率与 density response；是 common-track 必需基线 |
| TWCRAC, SSRN 2026, [DOI](https://doi.org/10.2139/ssrn.6630216) | 可访问摘要明确提出 Time-Window-Cycle、intra-video TCC 与基于局部 periodicity statistics 的动态阈值，目标含 non-stationary motion | 是当前最高重叠并行风险；由于全文 PDF 被拒绝访问，只能确认摘要范围。不得把“local windows + cycle consistency + dynamic tempo”单独作为 novelty |
| HTRM-Net, TMM 2025, [DOI](https://doi.org/10.1109/TMM.2025.3535385) | bi-modal TSSM、random matrix dropping、local temporal context 与 adaptive multi-scale fusion | 已覆盖 non-uniform period、interruption 和 local context；本项目必须用多人身份、学习型局部专家与逐轨重建区分 |
| Rethinking Temporal Self-Similarity, ICIP 2024, [DOI](https://doi.org/10.1109/ICIP51287.2024.10647309) | 保留 full-resolution embeddings，预测 action-start response，并用 reference-TSM consistency loss | 已占据 full-resolution response 与 response decoding；不能把“输出响应再计数”本身作为贡献 |
| JTSPS-Net, TCSVT 2024, [DOI](https://doi.org/10.1109/TCSVT.2024.3402728) | skeleton 输入、joint-wise temporal self-similarity periodic selection、coarse-to-fine multi-scale processing | 已占据 joint-wise periodic selection 与多尺度 skeleton modeling |
| SSTRAC, IEEE Access 2025, [DOI](https://doi.org/10.1109/ACCESS.2025.3624029) | skeleton repair、dual-stream spatiotemporal Transformer、multi-scale attention 与 density map | 已覆盖骨架缺失修复、遮挡与多尺度频率；仍为监督式单目标 |
| Runia et al., CVPR 2018, [DOI](https://doi.org/10.1109/CVPR.2018.00939) | 从 optical-flow differential maps 通过 temporal wavelet filters 估计并分割 non-stationary repetition | “非平稳重复”与局部时频分析并不新 |
| Crochiere WOLA, 1980, [DOI](https://doi.org/10.1109/TASSP.1980.1163353)；Griffin--Lim, 1984, [DOI](https://doi.org/10.1109/TASSP.1984.1164317) | weighted/window-normalized overlap-add 与 short-time reconstruction 的经典信号处理基础 | overlap-add/NOLA 只能是已知重建算子，novelty 必须来自逐身份 RAC 响应如何路由和重建 |
| Shazeer et al., ICLR 2017, [OpenReview](https://openreview.net/forum?id=B1ckMDqlg) | trainable gating、shared experts 与负载均衡 | soft MoE 是通用机制，不是论文贡献本身 |

### 主题综合

1. **MRAC 已是成熟的监督式任务定义。** MultiCounter 与 MultiCounter+
   已联合覆盖逐实例检测、跟踪、周期定位、计数、异步开始/停止、变速与暂停。
2. **pose/skeleton RAC 已高度拥挤。** salient poses、joint-wise periodicity、
   geometric features、occlusion repair、body-part graphs、自监督 skeleton
   embeddings 与 RGB-pose fusion 均已有先例。
3. **non-stationary tempo 不是新问题。** wavelet、per-frame period、local
   context、long/short-period awareness、multi-scale correlation 与动态阈值均已
   出现；本项目必须检验更窄的“同一身份轨迹内部 piecewise tempo drift”。
4. **专家概念必须拆开。** PAMS 的 inference consensus、通用 MoE 的学习型 gate、
   以及本项目拟议的 tempo-specialized response functions 是三种不同机制。
5. **overlap-add 是经典重建。** 可防守的问题是：局部响应能否在逐身份约束下
   重建为单一轨迹响应，并比 window-count summation 更少重复/漏计。

### 可检验研究空缺

以下表述均是“本轮检索未发现直接覆盖”，不是绝对 `first` claim：

- 尚未发现同时满足 MRAC 输出、pose trajectories、无逐人 count/period 标注、
  identity-private state、window-local tempo routing、逐轨 overlap-add 后一次解码的
  已验证 RAC 方法。
- PAMS 的速度实验是整段视频使用单一 global resampling factor；piecewise
  within-video acceleration/deceleration 是更窄且更强的诊断。
- 尚未发现 count-label-free MultiRep 模型同时显式评估 track fragmentation、
  re-entry 与 identity switch 对逐人计数的影响。
- supplied/oracle tracks 与 predicted tracks 的分离，以及 HOTA/IDF1/IDSW 和
  count degradation 的联合分析，仍可构成评价协议贡献。

### 禁用 claim 与直接风险

- 禁止 `first multi-person`, `first person-wise`, `first asynchronous`,
  `first variable-speed`, `first pose-level`, `first self-supervised`,
  `annotation-free`, `end-to-end`, `constant-time` 以及无同协议证据的 `SOTA`
- “local window + TCC + dynamic tempo” 会直接撞上 TWCRAC
- “local context + interruption robustness” 会撞上 HTRM-Net
- “multi-expert consensus” 会撞上 PAMS；必须写清本项目是 learned response
  experts、per-window soft routing，并用实验验证 specialization
- “normalized overlap-add” 必须引用经典信号处理，不能作为独立 novelty
- `PoseRAC` 的 2023 arXiv 与 2025 LNCS 版本必须分开引用和描述
- 最准确的 supervision 限定词是 “the counting objective uses no per-person
  count or period labels”，并单独披露 detector/pose/tracker 的外部监督

### 检索失败与限制

- IOS Press 的 MultiCounter landing page 被 robots 限制；已通过 DOI、ECAI
  proceedings PDF 与 arXiv 原文交叉核验
- PAMS 与 D²-STX 的 CVF 正式页面未提供 DOI；使用稳定 CVF proceedings URL，
  不编造 DOI/arXiv ID
- TWCRAC 的 SSRN PDF 返回 HTTP 403；只使用可核验摘要，不推断全文细节
- 本轮未把任何搜索引擎摘要当成 BibTeX 元数据来源

## Ranked Ideas

该节将在 idea-creator fan-out 完成机械去重并经 fresh GPT-5.6-Sol jury 后写入。

## Novelty Verification

该节将在候选方法冻结后由独立 novelty-check 填写。

## External Critical Review

该节将在 novelty-check 后由 fresh GPT-5.6-Sol ultra research-review 填写。

## Refined Proposal

该节将在 research-refine-pipeline 结束后链接到
`refine-logs/FINAL_PROPOSAL.md`、`EXPERIMENT_PLAN.md` 与
`EXPERIMENT_TRACKER.md`。

## ARIS Idea-Creator 决策

机械合并得到 17 个候选且无精确重复。fresh `gpt-5.6-sol` jury 的 same-family provisional 排名、分数、主要 kill risk、三个有界 pilot、synthesis claim boundary、禁用 claims 与零合格结果状态，详见 `idea-stage/IDEA_CANDIDATES.md` 及其时间戳快照 `idea-stage/IDEA_CANDIDATES_20260815_122541.md`。

裁决为：推进三个 development pilots，但不冻结论文方法或声称性能。主方案是 `warp-equivariant-private-phase-flow`，独立后备是 `masked-geometric-tssm-ridge-completion`，因果对照是 `identity-private-state-isolation`。允许的 claim 仅限未来合格证据支持的 MRAC、supplied pose、counting objective 无逐人 count/period labels、window-local phase/tempo adaptation 与 identity-private state 的交集；各组成机制均不得单独声称新颖。

当前仍为 **zero eligible method results**：v46、v62、v63 仅供隔离审计，且不存在端到端 canonical 实现或合格运行。由于 jury 与流水线同属模型家族，本决策只能保持 **same-family provisional**，不能作为独立接受证据。

## Canonical novelty 边界同步（2026-08-15 12:54:12，Asia/Shanghai）

本节仅同步 `idea-stage/NOVELTY_CHECK.json` 的既有裁决，不构成新的文献检索或新颖性评审。该证据文件的 SHA-256 为 `79749893547c834d876801c287bfa2f048b4797c838f7f6d46155318258ed3e5`。

- **PAMS 边界修正：** PAMS supplement 已包含随机 5--7 秒局部片段的内部速度扰动/局部变速实验。因此，within-video 或 localized tempo perturbation 不能再写成空白、首次或 PAMS 未覆盖；本项目只能把预注册的逐轨多段 tempo drift 作为更强压力测试扩展。
- **新增机制近邻：** `DeepPhase` 是无监督局部运动相位流形/周期自动编码近邻；`Motion Feature Learning` 已使用 temporal self-similarity matrix reconstruction loss 做 repetitive action counting。两者分别约束 phase-flow 与 masked-TSSM 的可辩护边界，必须进入机制匹配比较或引用矩阵。
- **唯一主候选裁决：** canonical overall verdict 仅为 `PROVISIONAL_ADVANCE_ONE_KILL_ORIENTED_PILOT_NO_NOVELTY_CLEARANCE`。只允许将 `warp-equivariant-private-phase-flow` 推进为一个有界、以证伪为目标的 development pilot；不得冻结论文方法或 novelty claim。
- **候选角色降级：** `masked-geometric-tssm-ridge-completion` 仅保留为 mechanism-matched control / fallback pretext，不得作为 primary novelty；`identity-private-state-isolation` 仅保留为 causal diagnostic / intervention，不得作为独立方法贡献。
- **未闭合 caveat：** TWCRAC 当前只能可靠核验 SSRN 摘要和 primary record，全文方法重叠尚未闭合。取得并检查完整全文、重新运行 novelty boundary 之前，不得冻结 manuscript claim。
- **禁用措辞：** 禁止任何 `first` 变体，以及 `annotation-free`、`end-to-end` 和无同协议合格结果支持的 `SOTA`。即使 counting objective 不使用逐人 count/period labels，也必须披露 detector、pose estimator、tracker、synthetic-warp correspondence 与任何 video-level supervision。

以上修正覆盖本报告较早的冲突性表述；未被本节明确修正的历史内容原样保留，并继续受 same-family provisional、zero eligible method results 与 no novelty clearance 的上限约束。
