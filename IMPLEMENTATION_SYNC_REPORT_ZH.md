# 多人变速重复动作计数：实现差异与待同步报告

**状态：** `provisional/data-pending`  
**本地证据基线：** `73cbc4bb9b581f24fbc90f2fe7421bc11e7c7454`  
**用途：** 合作者代码与实验产物到达时的逐项对照清单  
**原则：** 冻结合作者实现优先于本文的暂定数学叙事

## 一、当前已确认的本地实现边界

本地跟踪代码是单人 PAMS 部分复现，不是本文拟写的多人模型。现有
`PoseSequence` 固定为一条 `[T, 33, 3]` 姿态序列；预处理按帧选择主导人物，
没有跨人物身份跟踪；`CountResult` 输出一个标量；训练、推理、指标和 dev84
结果也均按单视频单计数工作。仓库文档明确不声称具备跨人物身份连续性。

当前可以复用的只是单人技术底座：姿态编码器、掩码归一化 ACF/FFT 周期证据、
PAMS-TCC 表征目标和推理期峰值共识。原 PAMS Period Head 的训练目标在来源论文
中未披露；Literal 头在本复现中是随机头，SSHead 是本仓库的推断性修复。因此
二者都不能被写成本文新模型已经验证的模块。

## 二、首版论文采用的暂定方法契约

为避免写作阶段在两套概念图之间摇摆，首版正文统一采用以下接口：

1. 上游检测、姿态估计与跟踪输出身份索引轨迹
   \(X_i\in\mathbb{R}^{T_i\times J\times C}\) 及有效掩码 \(m_i\)。
2. 编码器与快/中/慢专家在所有人物之间共享权重；每个身份仅拥有独立输入、
   掩码、缓存、路由状态和输出，不为每个人复制一套模型参数。
3. 每条轨迹划分为重叠局部窗口；周期证据和 soft tempo routing 随窗口变化，
   而不是给整条人物轨迹分配一个固定速度。
4. 专家先输出连续局部响应；窗口响应经掩码归一化 overlap-add 合成为一条
   身份响应，再对整条轨迹执行一次峰值提取。
5. 主输出是逐人计数向量 \(\hat{\mathbf c}\)；所有人物计数之和仅为派生诊断。
6. 总损失目前只保留 PAMS-TCC、routing consistency、track continuity 三项的
   接口名。后两项没有在本地得到实现定义或权重，正文明确标为
   `SYNC-REQUIRED`。

这套契约解决了概念材料中的两处冲突：将“person-specific model”解释为共享参数
下的逐身份状态；将“boundary stitching”具体化为连续响应域的归一化拼接，再做
一次离散计数，而不是分别计数窗口后相加。它是写作默认值，不是对合作者实现的
反向约束。

## 三、必须从合作者冻结提交中同步的接口

| 项目 | 本地现状 | 论文暂定值 | 合作者产物必须回答的问题 |
|---|---|---|---|
| 多人输入 | 只保留主导人物 | 身份轨迹集合与掩码 | 实际张量形状、padding、batch、人数上限和缺失帧语义是什么？ |
| 检测/姿态/跟踪 | 单人 MediaPipe 路径 | 上游独立 frontend | 具体模型、版本、预训练来源、阈值和 ID 生命周期是什么？ |
| 权重共享 | 单人模型 | 所有人共享编码器/专家 | 是否存在每人独立参数、人物交互或共享缓存？ |
| 窗口化 | 无多人局部窗口 | 长度 \(L\)、步幅 \(S\) | 边界 padding、短轨迹和断裂轨迹如何处理？ |
| 周期证据 | 单序列 ACF/FFT | 窗口级 period/confidence | 周期范围、采样率、低置信度 fallback 和梯度停止是否一致？ |
| 路由器 | 不存在 | soft fast/medium/slow | 路由输入、参数化、温度、专家数、监督/自监督信号是什么？ |
| 专家 | 推理期峰值参数共识 | 可学习局部响应专家 | 专家是网络、滤波器还是解码规则？是否真正学习并分化？ |
| 边界 | 无多人窗口拼接 | normalized overlap-add | taper、归一化、最终峰值阈值和去重规则是什么？ |
| 连续性 | 不存在 | track-continuity objective | 正负样本、断轨重连、ID switch 及损失权重如何定义？ |
| 训练标签 | 单人 PAMS 自监督复现 | 不使用逐人 count/period 标签 | 任何 loader、loss、伪标签、选择器、调参或早停是否访问计数真值？ |
| 输出 | 标量 | 逐人计数与周期事件 | 事件边界、置信度、track ID 和场景总数的接口是什么？ |

若任一答案与首版公式不符，应修改论文和图，而不是为了保留叙事而改动冻结代码。
每一处差异需记录：实现文件与符号、配置键、测试、修改前后的论文位置以及是否影响
贡献或实验协议。

## 四、实验产物同步合同

每个可进入论文的 run 必须同时提供：

- 模型 commit、干净/脏状态、完整配置及其 SHA-256；
- seed、硬件、软件环境、训练和评估日志；
- MultiRep 精确 release、下载时间、官方 split 和视频/标注/split checksum；
- oracle tracks 或 predicted tracks 标志及 frontend 版本；
- 逐视频、逐人物真值与预测，必要时含逐帧周期事件；
- Period-mAP、AP50、AP75、AvgMAE、AvgOBO；预测轨迹还需 HOTA、IDF1、IDSW；
- 至少三个预注册 seed 的均值、样本标准差，以及按视频聚类的配对 bootstrap
  95% 置信区间；
- 由同一冻结记录自动生成的表格/绘图中间文件。

MultiRep 在 MultiCounter 与 MultiCounter+ 资料中存在版本描述差异，禁止仅写
“MultiRep”而不记录 release 和 checksum。共同 predicted-track frontend 下，
HOTA、IDF1 和 IDSW 是输入轨迹诊断，原则上不应随下游计数器变化，也不能被写成
计数模块带来的提升。

## 五、当前结果的排除边界

首轮 ARIS experiment audit 已判定 `FAIL / NO_MULTI_PERSON_REAL_GT_EVIDENCE`。
这表示当前产物不足以支撑新人多论文，而不是把负结果隐藏起来：单人 UCFRep
dev84 负结果、proxy、synthetic、smoke、source-sanity、随机 Period Head 和推断
SSHead 均保留作诊断，但 `table_eligible=false`。没有任何 UCFRep test105 指标，
本任务也不会访问该封存测试集。

合作者多人产物到达后，必须按
`experiment-audit → result-to-claim → paper-claim-audit` 的顺序解锁。任何摘要数字、
性能贡献、优于基线或 SOTA 表述都不得提前填写。

## 六、需要合作者一次性确认的最小问题

1. 冻结代码的唯一 commit/branch 和完整 artifact bundle 在哪里？
2. 实际是共享权重、逐身份状态，还是每人独立参数？
3. 窗口边界究竟在连续响应域拼接，还是先分段计数后做事件级 stitching？
4. routing consistency 与 track continuity 的精确公式、监督来源和权重是什么？
5. MultiRep 的具体 release、split、checksum 以及逐人/周期事件标注格式是什么？
6. frontend 是 oracle、外接 detector/pose/tracker，还是与计数器联合训练？
7. 哪些 run 已完成三种子，哪些数字允许公开，哪些只是内部诊断？

这七项未全部闭合前，论文可以继续完善写作、引用、图和实验合同，但不能升级为
实证完成稿或 submission-ready 状态。
