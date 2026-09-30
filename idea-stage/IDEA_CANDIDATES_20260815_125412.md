# ARIS Idea-Creator 候选决策

**生成时间：** 2026-08-15 12:25:41（Asia/Shanghai）  
**评审模型：** `gpt-5.6-sol`；**独立性/状态：** same-family provisional  
**来源：** `CANDIDATE_UNION.json`（17 个，精确去重后仍为 17 个）与 `IDEA_CREATOR_JURY.json`

## 17 个候选排名

| 排名 | 候选（dedup key） | 总分/10 | 主要 kill risk |
|---:|---|---:|---|
| 1 | Warp-Equivariant Private Phase Flow (`warp-equivariant-private-phase-flow`) | 7.9 | 合成 warp 等变性可能只学习插值/时间戳伪影；谐波相位不可辨识。 |
| 2 | Masked Geometric TSSM Ridge Completion (`masked-geometric-tssm-ridge-completion`) | 7.5 | TSSM 领域拥挤，网络可能只补全平滑矩阵。 |
| 3 | Identity-Private Tempo State Isolation (`identity-private-state-isolation`) | 7.3 | 私有状态本身不新，可能仅记忆轨长、可见性或置信度。 |
| 4 | Artifact-Controlled Synthetic Tempo Routing (`artifact-controlled-synthetic-tempo-routing`) | 7.2 | 与 PAMS/MoE/MultiCounter+/TWCRAC 重叠，重采样伪影可能让结论失效。 |
| 5 | Masked Oscillator Residual Bottleneck (`masked-oscillator-residual-bottleneck`) | 7.1 | 人体动作非干净振子；残差容量可能绕过或压垮周期分支。 |
| 6 | Mask-Normalized Chirplet Routing and Synthesis (`masked_chirplet_ola`) | 7.0 | 复杂值实现成本过高，chirp slope 可能跟随相机或肢体谐波。 |
| 7 | Supplied-versus-Predicted Track Protocol Pair (`dual-track-protocol-count-gap`) | 6.9 | 是证据基础设施而非充分方法贡献，且缺预测轨迹和冻结 tracker。 |
| 8 | Overlap-Graph Phase Synchronization (`overlap_phase_sync_graph`) | 6.7 | 无复相量 backbone；同步与 overlap-add 是既有机制。 |
| 9 | Fragmentation-Equivariant Stateful Response (`fragmentation-equivariant-stateful-response`) | 6.6 | 依赖缺失基础模型/预测轨迹，合成 fragmentation 未必代表真实失败。 |
| 10 | Irregular-Time Lomb Phase Counting (`irregular_lomb_phase`) | 6.4 | 时间戳/mask 可能不可用，协方差求解数值不稳且谐波敏感。 |
| 11 | Mass-Conserving Per-Person Count Ledger (`mass-conserving-person-count-ledger`) | 6.3 | 主要是记账；可能守恒总量却归错人物。 |
| 12 | Mask-Adaptive Multitaper Tempo Posterior (`masked_multitaper_uncertainty`) | 6.1 | 重正交和重复推理昂贵，taper dispersion 未必预测计数误差。 |
| 13 | Identity-Private Switching Oscillator Filter (`identity_oscillator_filter`) | 6.0 | 实现链长，存在严重半频、倍频与校准失败。 |
| 14 | Variational Monotone Phase Clock (`variational_monotone_phase_clock`) | 5.9 | spline、重建、后验与解码首轮过大且多处不可辨识。 |
| 15 | Pose-and-Phase Re-entry State Cache (`pose-phase-reentry-cache`) | 5.6 | 接近通用 re-ID；缺重入数据，误合并可能更糟。 |
| 16 | Ragged Shared-Compute Runtime (`ragged-shared-expert-scaling`) | 5.3 | 无 canonical 实现可优化，批处理也非核心科学贡献。 |
| 17 | ID-Switch-Resilient State Hypothesis Bank (`ambiguous-switch-state-hypotheses`) | 5.0 | MHT 已有先例且组合爆炸；缺关联 logits/交叉数据，未解决核心表征。 |

## 选定的 3 个 pilot

1. **Primary：`warp-equivariant-private-phase-flow`。** supplied tracks 上实现共享 pose encoder、identity-private GRU、64 帧重叠窗口、单位圆 phase/periodicity、piecewise-warp 等变约束、限容 pose-velocity reconstruction、normalized overlap-add 与每身份一次解码；三独立 seed。门槛：未见 warp 的中位相位误差 ≤0.15 cycles、effective rank ≥8、AvgMAE 相对改善 ≥5% 或 Period-mAP 绝对改善 ≥2.0、warp degradation ≤10%，并由配对 source-cluster bootstrap 支持。
2. **Secondary fallback：`masked-geometric-tssm-ridge-completion`。** 独立实现 32/64 帧 masked off-diagonal TSSM completion、三种固定几何 view、小型 completion encoder、ridge-to-response、私有 response buffer、overlap-add 与每轨一次解码。仅在预注册独立比较胜出后才可替换/初始化主 encoder。
3. **Control：`identity-private-state-isolation`。** 冻结主 backbone，比较 private-state、stateless-window、shared-scene-state；一次只 warp 一人，测未触碰身份，并执行 state reset/swap 与 track-length-only 负控制。它是架构不变量/因果消融，不是独立 novelty。

## Synthesis claim boundary

仅以 primary 作为拟议学习机制；control 作为不变量/消融；normalized overlap-add、half-open boundary 和 one-decode-per-identity 仅作既有算子；secondary 初期保持独立后备。

唯一可能可辩护的 claim 是经合格实证验证的交集：**多人物逐身份输出 + 外部 supplied pose tracks + counting objective 不用逐人 count/period labels + window-local phase/tempo adaptation + identity-private state**。各组成机制不得单独声称新颖。未来证据至多支持：该交集在冻结的轨迹内非平稳节奏诊断下改善 pose-driven per-person counting。

## 禁用 claims

- 禁用 `first`、`annotation-free`、`end-to-end`、`constant-time`、`SOTA/state of the art`，以及无同协议证据的性能宣称。
- 不得声称首次 multi-person/person-wise/asynchronous/variable-speed/pose-level/self-supervised RAC。
- protocol、overlap-add、private buffers、ragged batching 不得作为中心 novelty；prospective thresholds、synthetic diagnostics、v46 results、v63 code 不得转成性能证据。
- 不得堆叠 chirplet、Lomb-Scargle、multitaper、variational clock、graph synchronization、ledger、tracking cache 来制造贡献。
- “local window + TCC + dynamic tempo”与 TWCRAC、“local context + interruption”与 HTRM-Net、“multi-expert consensus”与 PAMS 高风险重叠；overlap-add 是经典算子。

## 零合格结果状态

当前为 **zero eligible method results**。没有端到端实现或合格运行；v46/v62/v63 仅为隔离审计材料。数据 release/split、评估器、predicted-track evaluation 与 canonical implementation 均未冻结。因此只推进三个有界 development pilots，**不冻结论文方法、不声称性能、不解除 submission blocker**。

## Same-family provisional

jury 与流水线同属 `gpt-5.6-sol` 家族，不能提供独立接受证据。排名、分数、pilot 与 synthesis 决策均为 **same-family provisional**，仅用于实验编排，不是科学结论或投稿就绪证明。

## Canonical novelty 边界同步（2026-08-15 12:54:12，Asia/Shanghai）

本节仅同步 `idea-stage/NOVELTY_CHECK.json` 的既有裁决，不构成新的候选评审。该证据文件的 SHA-256 为 `79749893547c834d876801c287bfa2f048b4797c838f7f6d46155318258ed3e5`。

- canonical overall verdict 只有 `PROVISIONAL_ADVANCE_ONE_KILL_ORIENTED_PILOT_NO_NOVELTY_CLEARANCE`。排名不等于 novelty clearance；唯一可推进的主候选是 `warp-equivariant-private-phase-flow`，且仅限一个 kill-oriented、bounded development pilot。
- `masked-geometric-tssm-ridge-completion` 不再是独立 secondary method 候选，只能作为 mechanism-matched control / fallback pretext；其 TSSM reconstruction 边界还受 `Motion Feature Learning` 这一近邻直接约束。
- `identity-private-state-isolation` 仅是 mandatory causal diagnostic / tempo-swap intervention，不是 standalone architecture novelty。
- phase-flow 的比较矩阵必须纳入 `DeepPhase` 这一无监督局部运动相位近邻；若 DeepPhase-style、PAMS/Track-PAMS、SimPer、CycleCL 或 matched time-equivariant baseline 在不确定性内匹配，则核心主张应被证伪。
- PAMS supplement 已包含随机 5--7 秒局部片段的内部速度扰动/局部变速。预注册的更复杂逐轨 tempo drift 只能表述为更强压力测试扩展，不能表述为首次 localized/within-video speed perturbation。
- TWCRAC 全文方法重叠仍未闭合；在取得完整全文并重新核验之前，所有与 local windows、cycle consistency、dynamic tempo 相关的 manuscript claim 均保持 unresolved。
- 禁止任何 `first` 变体，以及 `annotation-free`、`end-to-end` 和无同协议合格结果支持的 `SOTA`；外部 detector、pose、tracker 与 augmentation-derived supervision 必须逐项披露。

本节是后续使用的 canonical 候选处置；较早的“三个 pilot”或 secondary fallback 推进措辞如与本节冲突，以本节及 `NOVELTY_CHECK.json` 为准。历史排名与 kill-risk 内容原样保留，用于审计而非授权。
