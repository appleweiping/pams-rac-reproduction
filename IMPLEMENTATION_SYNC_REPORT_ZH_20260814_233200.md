# ICASSP 2027 实现差异与待同步报告

**论文题目：** *Identity-Indexed Local Tempo Routing for Multi-Person Repetition Counting*
**当前状态：** `provisional/data-pending`
**CVPR 写作冻结提交：** `26e5b7176f1f7c678391163c3278b5c390d54115`
**代码基线：** `73cbc4bb9b581f24fbc90f2fe7421bc11e7c7454`
**用途：** 合作者代码与实验产物到达后的逐项核对清单
**最高原则：** 合作者最终冻结实现优先于本文的暂定公式和概念图

## 一、当前可以确认的事实

本地仓库是一个单人 PAMS 独立复现，不是论文拟写的多人模型。当前
`PoseSequence` 表示一条姿态序列，预处理选择主导人物而不维护跨人物身份，
公开输出是一个标量计数。仓库文档也明确不声称具备多人身份跟踪能力。

当前可以复用的只是技术起点：姿态编码器、单序列掩码 ACF/FFT 周期估计、
PAMS-TCC 起始目标、单序列峰值共识、计数标签与预测器的边界，以及现有指标/
bootstrap 工具。这些内容均不能证明多人输入、身份状态、局部节奏路由、可学习
专家、overlap-add 重建、逐轨迹解码或 MultiRep 性能。

本地结果全部是 UCFRep dev84 负结果、代理、推断性修复或独立协议下的
checkpoint sanity。PAMS-Literal 的 Period Head 在已披露训练描述下没有训练路径，
PAMS-SSHead 是独立推断修复。它们保留用于失败分析，但对 ICASSP 新方法均为
`table_eligible=false`。UCFRep test105 仍未评分，本写作任务也不访问该封存测试集。

## 二、ICASSP 版本的暂定技术主线

ICASSP 正文把每个人的身份索引姿态轨迹视为带掩码、多维、非平稳信号，区分：

- 不同人物之间的异步节奏；
- 同一个人物轨迹内部随时间变化的节奏。

主输出是逐人计数向量，场景总和仅作为派生诊断。方法叙事固定为四组公式：

1. 身份索引轨迹 \(X_i\)、有效掩码 \(m_i\) 与逐人计数向量；
2. 局部窗口内的掩码 ACF/FFT 周期证据与置信度；
3. 快/中/慢共享专家上的 soft tempo routing；
4. 响应融合、normalized overlap-add，以及每条身份轨迹一次最终解码。

默认写法是所有人物共享编码器与专家参数，每个人只保留独立的输入、掩码、
缓存/状态和输出。专家产生连续局部响应，不先得到窗口整数计数；窗口响应重建为
一条身份响应后才统一提取事件，避免重叠区域重复计数。

以上均是写作阶段的 `SYNC-REQUIRED` 合同，不是对合作者实现的反向约束。若
最终代码不同，必须修改论文公式、框架图、贡献点和实验设计，而不能为了保留
现有叙事去改合作者冻结代码。

## 三、必须从合作者冻结提交中核对的接口

| 项目 | 当前本地状态 | 写作默认值 | 必须回收的实现证据 |
|---|---|---|---|
| 多人输入 | 仅单人主导轨迹 | 身份索引轨迹集合与有效掩码 | 实际张量形状、batch/padding、人数上限、短轨迹、缺失帧、进入/离开语义 |
| 检测/姿态/跟踪 | 单人 MediaPipe 路径 | 计数器接收 supplied tracks | frontend 名称、版本、权重监督来源、阈值、ID 生命周期、oracle/common/native 模式 |
| 参数共享 | 单人模型 | 编码器和专家跨人共享，状态按身份隔离 | 模块路径、参数注册、缓存结构、参数共享与跨身份隔离测试 |
| 窗口化 | 没有多人窗口接口 | 重叠局部窗口 | 窗长、步幅、padding、短轨迹、断轨、掩码传播方式 |
| 局部周期 | 只有单序列 ACF/FFT 起点 | 每身份、每窗口 period/confidence | 输入信号、去均值、滞后/频率范围、采样单位、插值、低置信 fallback |
| 路由器 | 不存在 | fast/medium/slow soft simplex | 输入、参数化、temperature、梯度路径、监督/自监督信号及路由输出 |
| 专家 | 当前只有推理规则共识 | 共享连续响应专家 | 专家是网络、滤波器还是解码规则；是否训练；是否真正分化 |
| 响应融合 | 不存在 | 路由后先融合连续响应 | 融合顺序、张量形状、非负性/归一化及 uniform/hard 等价测试 |
| overlap-add | 不存在 | 掩码和 taper 归一化重建 | taper、分母、边界、无效间隙、重复窗口不变性 |
| 最终解码 | 单序列峰值规则 | 每条身份轨迹只解码一次 | 事件/峰值定义、阈值、距离、置信度、去重、整数化和逐身份调用测试 |
| 训练目标 | 只有 PAMS-TCC 起点 | 暂留 route/track 两个接口名 | 两项是否真实存在；精确公式、样本、标签来源、权重、梯度、配置与消融 |
| 监督边界 | 只验证了本地单人预测/评估边界 | counting objective 不用逐人 count/period 标签 | loader、伪标签、loss、router、调参、选模、早停、校准和推理的完整 firewall |

如果 routing consistency、track continuity 或固定 fallback 在最终代码中不存在，
必须从论文删除，不能以概念设计补写成已实现模块。

## 四、实验产物同步合同

每个允许进入论文的结果 family 必须同时提供：

- 唯一干净 commit、完整 resolved config 及 SHA-256；
- 环境、硬件、seed、训练/预测/评估日志；
- MultiRep 的准确 release、官方 split、视频/标注/split checksum；
- evaluator 版本、预处理 fingerprint、逐视频逐人物 GT 与预测；
- `oracle_tracks`、`common_predicted_tracks` 或 `native_pipeline` 明确标志；
- Period-mAP、Period-AP50、Period-AP75、AvgMAE、AvgOBO；预测轨迹另含 HOTA、IDF1、IDSW；
- 对复现的学习方法至少三个预注册 seed，报告均值、样本标准差和按视频聚类的
  paired bootstrap 95% CI；
- 从同一冻结 JSON/CSV/Parquet 自动生成表格和图的中间文件与哈希。

主表必须分成三个不可混排排名的协议块：MultiCounter/MultiCounter+ 的
native/contextual 块、用于隔离计数器本身的 supplied oracle-track 块，以及共同
冻结跟踪输入下的 common predicted-track 块。后两块都包含 Track-PAMS、
track-global tempo 和完整模型。只有证明输入、输出与协议一致后才能加入其他
基线，也不能跨块加粗所谓“最好”结果。

核心消融固定为 local/global tempo、uniform/hard/soft routing、无专家路由、
boxcar 融合、逐窗口解码后聚合与 NOLA 后统一解码，以及最终代码中真实存在的
辅助损失移除。非平稳诊断采用预注册
的逐人 piecewise time-warp，并绑定局部周期、路由概率/熵、重建响应和
clean-to-corrupt 变化。不存在的模块必须删掉对应消融行。

解锁顺序是：`experiment-audit -> result-to-claim -> paper-claim-audit`。前两项
通过之前，摘要数字、主表数值、性能贡献、因果解释、优于基线或 SOTA 表述全部禁止。

## 五、投稿格式、作者与引用待同步项

ICASSP 2027 regular 稿件采用四页技术内容；按 2026-08-14 的官方网页，第五页
只能放参考文献，funding/COI 与 Compliance with Ethical Standards 必须留在前四页。正文计划为 140–160 词摘要、
约 500 词引言（合并 related work）、约 900 词方法、约 700 词实验、100–120 词
结论，配一张框架图、一张节奏诊断图和两张紧凑表。第五页禁止任何方法、实验、
限制、公式、图表、图注或附录内容。

ICASSP 为单盲审稿，最终 PDF 必须包含完整作者顺序、单位和必要联系方式。当前
tracked tree 不保存个人作者信息，只有私有构建时注入；合作者姓名、顺序、单位尚未
齐全，因此 submission 构建必须失败。官方 ICASSP 2027 author kit 尚未绑定；临时
IEEE/ICASSP 模板只能作为醒目标记的脚手架，不能声称最终格式合规。

Google Scholar 仅用于发现论文和 cited-by 扩展；BibTeX 与引用上下文必须由
DBLP、CrossRef、CVF、IEEE、Springer 或作者官方页面核实。CVPR 版本的 citation
audit 只能作为历史输入，压缩重写后必须重新审计。

## 六、禁止表述与当前门禁

当前禁止声称：

- 本地仓库已经实现或验证了多人方法；
- 首个多人、逐人、异步、变速、姿态、自监督方法；
- `annotation-free`、`label-free`、`end-to-end`、`constant-time`；
- 未经相同协议验证的 SOTA；
- 任何新方法性能、鲁棒性、泛化性或效率结论；
- 概念图展示的是已执行行为；
- `submission-ready: yes`。

当前唯一准确的监督限定是：“预期 counting objective 不使用逐人 repetition-count
或 period 标签”，同时必须披露 pose/tracker 可能使用外部监督。该限定只有在完整
supervision firewall 通过后才能改成已验证事实。

ARIS 固定在 `e12e07c7b85ee1a4dc07e5463089aa16836af2bf`，本次全新网络 clone/
checkout 未形成可用干净工作树，已记录为降级，不进行破坏性 reset/clean。Claude
跨模型审稿只有在实时健康探针返回有效响应并保存 trace 时才计入；否则只能保留
同模型 provisional 审计。模板、作者、数据、实现、跨模型审稿或任何 mandatory
audit 缺失时，状态始终是 `provisional/data-pending`。

## 七、合作者交付后最小核对顺序

1. 冻结唯一代码 commit、clean status、完整 artifact bundle 和 package hash；
2. 对照上表填写每个模块的文件、符号、配置键和测试；
3. 完成监督 firewall，删除代码中不存在的论文机制；
4. 冻结 MultiRep 与三种 track 协议，核验所有 checksum；
5. 生成主结果、核心消融与 piecewise-warp 诊断的原始产物；
6. 依次运行三层实验/claim 审计并更新摘要、贡献、表图和结论；
7. 补齐作者、funding、伦理声明和官方 2027 模板；
8. 重新完成两轮 fresh review、proof/claim/citation/kill-argument 和最终 verifier。

任何一步与现有叙事不一致时，优先重写论文并在本报告追加差异，不修改冻结证据去
迎合稿件。

## 八、本轮预结果论文包交付状态

本轮已完成 ICASSP 2027 英文预结果稿、模块化 LaTeX、16 条已核验参考文献、
可编辑 Draw.io/SVG/PDF 框架图、两张结果表壳、局部节奏诊断壳、方法与结果证据
manifest、16 项 claim--evidence 账本，以及 ARIS 的两轮 fresh review、proof、claim、
citation 和 kill-argument 审计。最终草稿 `paper/main.pdf` 为 5 页，SHA-256 为
`873c83da0dbf4b24a6e1199f3dcd980e17d0b032580e0591f88a7c545ec67a45`；技术内容在
第 1--4 页，第 5 页仅含参考文献。编译检查为零未定义引用、零重复标签、零
overfull box，所有字体均嵌入，五页视觉检查全部通过。

官方 ARIS verifier 的 draft 模式返回 0，并把状态限定为 `provisional`；submission
模式按预期返回 1。Proof audit 为 `NOT_APPLICABLE`，citation audit 为
`PASS/same-family-provisional`，paper-claim audit 为 `BLOCKED/data-pending`，
kill-argument 为 `FAIL`：独立攻击与答辩均认为当前包适合作为诚实、可审计的预结果
写作基线，但若现在投稿，会因方法未同步、关键接口不可执行和没有合格 MultiRep
证据而被强拒。Claude overlay 两次健康探针均未返回有效 JSON，因此没有跨模型接受
证据，不能提升 submission readiness。

本次工作位于隔离分支 `codex/paper-writing-icassp27` 与独立 worktree，未修改
`D:\Project\rac` 原脏工作区，未合并实验分支，未运行或评分 UCFRep sealed test105，
也未推送远端。作者信息只存在于 Git 忽略的私有文件；tracked 论文包和全部 PDF 中
均未出现私人姓名、单位或邮箱。当前仍然缺少四类解锁输入：合作者冻结实现与完整
结果包、完整作者顺序与声明、官方 ICASSP 2027 author kit/最终 AI 政策核对，以及
健康的跨模型审稿证据。在这些输入全部闭合并重跑全流程前，唯一允许的状态仍是
`provisional/data-pending`。
