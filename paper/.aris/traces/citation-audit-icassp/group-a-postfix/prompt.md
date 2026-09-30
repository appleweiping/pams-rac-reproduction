# ICASSP citation post-fix audit prompt — group A

执行一次 fresh、zero-context 的引用复核。不得读取或依赖任何旧的 audit、history、log 或既有结论；只检查当前 `paper/references.bib` 与由 `paper/main.tex` 活动输入链包含的 TeX。

审计键限定为：

- `levy2015live`
- `runia2018realworld`
- `dwibedi2020repnet`
- `hu2022transrac`
- `yao2025poserac`
- `gao2026pams`
- `tang2024multicounter`
- `tang2026multicounterplus`

要求：

1. 重新浏览 primary official/CVF/Springer/IEEE/DBLP/Crossref 来源，逐键检查论文存在性、作者/题名/venue/年份/卷期页/DOI 等元数据，以及当前引用上下文是否由原文支持。
2. 独立核验导言中 “Early work addressed live and real-world repetition counting” 的措辞，不能因任务提示而预设其正确。
3. 独立核验 MultiRep 指标名称，重点检查 `Period-mAP`、`Period-AP50`、`Period-AP75`，并同时核对 `AvgMAE`、`AvgOBO`。
4. 区分 source-native 方法/协议与稿件自行构造的适配基线；不能把后者误归因于被引论文。
5. 不修改正文或 Bib；只生成本目录下的审计产物。
6. 记录当前上下文与文件 SHA-256。预期锁定值为：
   - `paper/sections/1_introduction.tex`: `6823651162f59e5f5263949f2d308f594b667cb3cba0044433c39031ae04db9d`
   - `paper/sections/4_experiments.tex`: `6052c816e94f7d7f50b560d397495a37863b05d240ed5e418e4b6ec4d3aa131b`
   - `paper/main.pdf`: `873c83da0dbf4b24a6e1199f3dcd980e17d0b032580e0591f88a7c545ec67a45`

审计身份必须标记为 `same-family`，结论状态必须标记为 `provisional`。
