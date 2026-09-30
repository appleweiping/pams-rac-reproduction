# TempoRAC Experiment Tracker（temporac.execution.v4）

> **规划状态：零 gate 通过、零 job 启动、零 metric、零 launch authority。**
>
> 第一项只能是未来另行授权的本地 P00 implementation。P06 fresh preflight 即使 PASS，也不授予服务器 launch。所有 data-bearing、GPU、capability 和 evaluator 项当前均为 `BLOCKED`。

**Tracker timestamp：** 2026-08-16 07:41:50 CST
**Normative proposal：** `refine-logs/FINAL_PROPOSAL.md` / SHA-256 `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391`
**Plan：** `refine-logs/EXPERIMENT_PLAN_20260816_074150.md`
**Canonical protocol：** `GT-bbox-assisted AlphaPose supplied-track pilot`。
**Mandatory artifact scope labels：** `partial-cache` / `GT-bbox-assisted` / `supplied-track`；必须出现在 permitted surrounding manifests、artifact paths、tracker notes 与 reports，不得添加为 closed receipt schema 的未知 keys。
**Current controlled-pilot population：** 110 train videos / 51 val videos；268 train tracks / 134 val tracks。该 partial cache 只允许 pilot/appendix evidence；`PAPER-PROMOTION-GATE` 固定 BLOCKED，直到未来 complete train/val pose cache + separate review/clearance；本 tracker 不扩数据。
**Frozen inventory：** 1 primary claim；4 blocks；3 baseline families；27 training jobs；93 A6000h；64 GiB scoped disk cap。
**Allowed natural data：** v44 train/val only。U0000--U0039 是纯合成 X0，不是外部数据分区。
**Canonicalization state：** Amendment 001/002 已整合；certificate/operator implementation contract 已固定，但 canonical candidate 尚未 fresh re-review，故 P00/P01/P02、S0 与所有 gates 仍无 PASS。
**P05M state：** Amendment 003 Round 2 仍为 `REVISE`；`P05M-COUNT-METRIC` 与 P2-METRIC 均为 `BLOCKED`，没有 fresh ACCEPT receipt。
**Authority vector：** `P3/P06 preflight=0`、`S0=0`、`server=0`、`data=0`、`training=0`；`launch authorization=0`。

## Status 与 stop code

| Code | 含义 |
|---|---|
| `NEXT_NO_AUTHORITY` | 依赖上仅是第一项，但本计划没有授予执行权。 |
| `BLOCKED` | 上游 receipt/独立授权未满足；不得执行。 |
| `TODO_AFTER_PASS` | 只有前序 PASS 且另行授权后才可排队。 |
| `PASS` | 所有预注册字段、hash、资源和科学阈值通过；当前没有任何 PASS。 |
| `STOP_GLOBAL` | 失败终止整条 TempoRAC 路线；不得在冻结 campaign 内修复、替换或 rerun。 |

Stop codes：

- `STOP-P`：implementation/static/fixture/preflight 的 byte、schema、row-count、determinism、import、disk 或 environment mismatch。
- `STOP-T`：teacher 未完成 20,000 steps、未写 exactly 40 checkpoints、nonfinite、hash/RNG/schedule mismatch、>6 A6000h 或 >24 GiB CUDA。
- `STOP-R`：response 未完成 10,000 steps、未写 exactly 20 checkpoints、nonfinite、hash/RNG/schedule mismatch、>2.5 A6000h 或 >24 GiB CUDA。
- `STOP-I`：inference 缺 population/stub/artifact、artifact/hash 非法、每 seed 超时、第二次 decode 或读取 vault。
- `STOP-K`：任一 G/K gate point/CI/coverage/margin/equality/cardinality/capability 条件失败。
- `STOP-D`：任一 artifact-class/global disk cap、free-space、inode、isolation 或 no-delete evidence guard 失败。
- `STOP-METRIC`：P2 AvgMAE/AvgOBO count-metric scorer fixture 的 evaluator bytes、input/expected/attack hashes、NAE/OBO aggregation、bootstrap multiplicity、arm reselection/clean reuse 或 fail-closed attack semantics 缺失/不等。
- `STOP-SCOPE`：任何 surrounding manifest/path/report 缺 `partial-cache` / `GT-bbox-assisted` / `supplied-track`，protocol name 不等于 `GT-bbox-assisted AlphaPose supplied-track pilot`，或 current partial cache 被请求 promotion 为 main-paper evidence。

## A. Pre-data、static、fixture 与 fresh preflight

| Run ID | Action | Data domain | Depends on | Required input | Required output | Stop | Status |
|---|---|---|---|---|---|---|---|
| `P00-IMPLEMENT` | 在全新 `pams.temporac` 下实现 package-private packer/records/teacher/certificate/response/NOLA/decoder/gates/evaluator capability；不改 WARP/public records | contract + synthetic/desensitized only | none | P0 proposal hash | S0-candidate code/layout inventory、file hashes、interface diff | `STOP-P` | `NEXT_NO_AUTHORITY` |
| `P01-STATIC` | format/lint/type/config/schema/import graph/forbidden path+key/job+budget scan；验证对 `pams.warp_phase` 和旧 PAMS `data/model/loss/training/period/consensus` 零 import/reachability | P00 bytes only | P00 | implementation receipt | static-conformance + import/firewall receipt | `STOP-P` | `BLOCKED` |
| `P02-X0-FIXTURE` | 40-row X0 manifest、arc/quadrature/phase/pulse fixtures | synthetic only | P01 | code/config/environment candidate hashes | X0 manifest/root + exact/ulp receipt | `STOP-P` | `BLOCKED` |
| `P03-TOPO-FIXTURE` | 13-row T00--T12 accept/reject topology bank | synthetic only | P02 | X0 root | topology manifest/expected-field hashes | `STOP-P` | `BLOCKED` |
| `P04-OP-FIXTURE` | 10,368-row operator bank，全部 layout/gap/width/amplitude/offset | synthetic only | P03 | decoder/NOLA code hashes | operator manifest/root + association truth receipt | `STOP-P` | `BLOCKED` |
| `P05-GRAPH-FIXTURE` | fused-only gradient、positive NOLA、pre-gap、two-identity、blocked=uniform、archive canonical bytes | synthetic/desensitized only | P04 | model/fixture roots | two-process equality、gradient、identity-isolation receipt | `STOP-P` | `BLOCKED` |
| `P05M-COUNT-METRIC` | 零 GPU finite desensitized AvgMAE/AvgOBO scorer fixture：positive integer GT、stub=0、person NAE/OBO、people-within-video→equal-video→ordered-seed、component multiplicity、per-draw comparator reselection + paired clean arm reuse；malformed/duplicate/empty/nonfinite/type attacks global fail | synthetic counts + opaque synthetic video/component tokens only；无 real label/source identity | P01 | exact evaluator implementation candidate bytes + fixture specification | `temporac.count-metric-fixture-receipt.v1` surrounding receipt，绑定 evaluator code/input/expected-output/attack-global-fail hashes | `STOP-METRIC` | `BLOCKED` |
| `P06-FRESH-PREFLIGHT` | TempoRAC-specific sanitized read-only environment/resource preflight；不复用现有 WARP record | no data payload | P01,P05,P05M | exact runtime spec、method+metric fixture hashes、64 GiB disk budget | fresh environment lock candidate、2×A6000/health/disk receipt；`launch_authorization=0` | `STOP-P`,`STOP-D`,`STOP-METRIC` | `BLOCKED` |

## B. Gate 与非训练工作清单

| Run ID | Stage/Gate | Purpose | Data domain | Depends on | Input receipt/root | Output receipt/root | GPU cap | Stop | Status |
|---|---|---|---|---|---|---|---:|---|---|
| `S0-COMMIT` | S0 | 冻结 v4 contract、environment、source allowlist、40/13/10,368 manifests、exact evaluator bytes + P05M input/expected/attack/receipt hashes、pilot-scope surrounding manifest、code/config/interface/job/RNG bytes | no natural data opened | P00--P06 + P05M + separate authority | all P receipts | hash-indexed S0 commitment；closed schemas 不加 scope keys | 0 | `STOP-P`,`STOP-METRIC`,`STOP-SCOPE` | `BLOCKED` |
| `G0-ACQUIRE` | S1:G0 | separately authorized trusted packer；七 member feature archives；402-row count-blind population | v44 train/val | S0 + packer grant | S0/source allowlist | feature receipts、population manifest、eligibility reasons | 0 | `STOP-K` | `BLOCKED` |
| `K0-FIREWALL` | K0 | total join commitments、component mapping、forbidden scan、root separation | v44 train/val metadata only | G0 | G0 roots | K0 firewall/join commitment receipt | 0 | `STOP-K` | `BLOCKED` |
| `G1-TEACHER-AGG` | S2:G1 | 三 teacher jobs 聚合、X0 tune-only selection、U0032--U0039 certificate/topology/pulse checks | X0 only | 3 TCH rows | 3 run receipts + X0 manifests | selected teacher/checkpoint root、G1 receipt | included below | `STOP-K` | `BLOCKED` |
| `K1-COVERAGE` | K1 | frozen G0 denominators 上 train/val identity+component certificate coverage | v44 train/val features only | G1 | G0 population + selected teacher | K1 coverage receipt | 0 | `STOP-K` | `BLOCKED` |
| `G2-CONFORMANCE` | G2 | exact channels/RF/cache/reset/NOLA/gradient/pre-gap/two-identity checks | synthetic + feature-only fixtures | K1 | P05 roots + teacher hash | G2 graph/isolation receipt | 0 | `STOP-K` | `BLOCKED` |
| `K2-TARGETS` | K2 | every certified X0/natural target immutable receipt；failure has no partial target | X0 + v44 train/val features | G2 | teacher + feature roots | target artifact/receipt roots | 0 | `STOP-K` | `BLOCKED` |
| `G3-RESPONSE-AGG` | S3:G3 | exactly 24 response jobs、schedule/checkpoint/resource/hash ledger | X0 train/tune + certified natural train | all 24 response rows | K2 + 24 run receipts | checkpoint roots、G3 93h-ledger receipt | included below | `STOP-K` | `BLOCKED` |
| `X0I-20260815` | S3 inference | fixed synthetic evaluation for seed 20260815 | U0032--U0039 | G3 | canonical/capacity/shortcut checkpoints | seed prediction/statistic shard receipts | 1 A6000h | `STOP-I`,`STOP-D` | `BLOCKED` |
| `X0I-20260816` | S3 inference | fixed synthetic evaluation for seed 20260816 | U0032--U0039 | G3 | same | same | 1 A6000h | `STOP-I`,`STOP-D` | `BLOCKED` |
| `X0I-20260817` | S3 inference | fixed synthetic evaluation for seed 20260817 | U0032--U0039 | G3 | same | same | 1 A6000h | `STOP-I`,`STOP-D` | `BLOCKED` |
| `K3-ROUTE` | K3 | strongest comparator advantage、shuffle removal、six shortcut recovery | synthetic evaluation only | all X0I | prediction/stat roots | K3 canonical statistic receipt | 0 | `STOP-K` | `BLOCKED` |
| `K4-DIAG` | K4 | per-seed+aggregate 3×3 specialization、support fractions | synthetic evaluation only | K3 | same paired shards | K4 matrix/bootstrap receipt | 0 | `STOP-K` | `BLOCKED` |
| `G4-OPERATOR` | S4:G4 | 10,368 finite operator rows + trained physical strata + offsets/batch equality/margins | synthetic evaluation only | K4 | operator root + checkpoints | G4 association/margin receipt | within X0I 3h | `STOP-K` | `BLOCKED` |
| `K5-BOUNDARY` | K5 | `B_err=0`、positive/negative margins `>=0.10` per required stratum | synthetic evaluation only | G4 | G4 rows | K5 receipt | 0 | `STOP-K` | `BLOCKED` |
| `K6-RESAMPLER` | K6 | all source orbits certify、phase error、pulse pullback、tau/count equality | synthetic evaluation only | K5 | G1/G4 artifacts | K6 receipt | 0 | `STOP-K` | `BLOCKED` |
| `NATP-20260815` | freeze natural artifacts | `GT-bbox-assisted AlphaPose supplied-track pilot`；`partial-cache` / `GT-bbox-assisted` / `supplied-track`；110/51 videos、268/134 tracks；local/global/uniform/capacity × clean/drift × 402 identities，含 stubs；appendix only | v44 train/val feature-only | K6 | frozen checkpoints/features/targets + pilot-scope manifest | seed prediction NPZ/receipt roots + surrounding scope index | 4 A6000h | `STOP-I`,`STOP-D`,`STOP-SCOPE` | `BLOCKED` |
| `NATP-20260816` | freeze natural artifacts | `GT-bbox-assisted AlphaPose supplied-track pilot`；`partial-cache` / `GT-bbox-assisted` / `supplied-track`；110/51 videos、268/134 tracks；seed 20260816；appendix only | v44 train/val feature-only | K6 | frozen checkpoints/features/targets + pilot-scope manifest | seed prediction NPZ/receipt roots + surrounding scope index | 4 A6000h | `STOP-I`,`STOP-D`,`STOP-SCOPE` | `BLOCKED` |
| `NATP-20260817` | freeze natural artifacts | `GT-bbox-assisted AlphaPose supplied-track pilot`；`partial-cache` / `GT-bbox-assisted` / `supplied-track`；110/51 videos、268/134 tracks；seed 20260817；appendix only | v44 train/val feature-only | K6 | frozen checkpoints/features/targets + pilot-scope manifest | seed prediction NPZ/receipt roots + surrounding scope index | 4 A6000h | `STOP-I`,`STOP-D`,`STOP-SCOPE` | `BLOCKED` |
| `G5A-COMMIT` | G5a | freeze complete roots、9,648 cardinality、mapped spans、teacher target counts；不计算 human statistic；verify pilot scope | frozen artifacts only；vault unavailable | all NATP + S0/P05M | contract/code/env/feature/population/checkpoint/prediction/receipt roots + metric fixture receipt + scope manifest | exact `temporac.g5a-receipt.v4` + append-only request ledger；closed `evaluator_code_sha256` 必须等于 P05M/S0 code hash | 0 | `STOP-K`,`STOP-METRIC`,`STOP-SCOPE` | `BLOCKED` |
| `CAP-ONEUSE` | capability grant | evaluator owner 另行审批一次 invocation；本计划不申请、不授予 | no payload returned | G5A + separate authority | exact G5a digest/request sequence | exact `temporac.capability-grant.v4` | 0 | `STOP-K` | `BLOCKED` |
| `G5B-JOIN` | G5b | 单一 non-resumable evaluator process 校验 268+134 join/interval/type/commitment | evaluator vault inside capability | CAP-ONEUSE | G5a digest + grant | exact `temporac.g5b-receipt.v4` | 0 | `STOP-K` | `BLOCKED` |
| `K7-FINAL` | K7 | 每 interval 一 pulse、union 外零 pulse、count equality；然后唯一 natural point/CI/clean decision；输出仍是 canonical `GT-bbox-assisted AlphaPose supplied-track pilot` controlled pilot/appendix，强制 `partial-cache` / `GT-bbox-assisted` / `supplied-track` | evaluator process only | G5B PASS, same process；P05M/S0/G5a evaluator bindings exact | G5a+G5b receipts + frozen predictions/vault + count-metric fixture receipt/expected hashes + scope manifest | exact `temporac.k7-receipt.v4` binding metric payload + surrounding pilot report/index；missing/mismatch 拒绝启动 | 0 | `STOP-K`,`STOP-METRIC`,`STOP-SCOPE` | `BLOCKED` |
| `PAPER-PROMOTION-GATE` | post-K7 reporting guard | 阻止 current canonical `GT-bbox-assisted AlphaPose supplied-track pilot` 的 `partial-cache` / `GT-bbox-assisted` / `supplied-track` evidence 进入 main table/abstract/headline/main-paper result；K7 PASS 不能解锁 | no new data or job | K7 final receipt + **future** complete train/val pose cache protocol + separate review/clearance | current pilot scope manifest | 本计划只可输出 `BLOCKED_PARTIAL_CACHE_APPENDIX_ONLY`；无 promotion receipt | 0 | `STOP-SCOPE` | `BLOCKED` |

当前 execution blockers：certificate/operator implementation contract 已固定但未 fresh re-review；Amendment 003 Round 2=`REVISE`，P05M/P2-METRIC=`BLOCKED`；独立 plan/execution authority、fresh TempoRAC preflight、trusted-packer grant、exact runtime/environment lock、one-use evaluator capability均不存在。P3/P06、S0、server、data、training authority 均为 0。当前 paper-promotion blocker：110/51 videos、268/134 tracks 的 partial cache 不是 main-paper eligible；只有未来 complete train/val pose cache + separate review/clearance 可在新协议中解除。任何 blocker 均不得由旧 WARP artifact、K7 PASS 或本 tracker 自行解除。

## C. Frozen training inventory（exactly 27）

### Teacher jobs（3 / 18 A6000h）

| Run ID | Exact ASCII job name | Stage | Training rows / selection | Steps | GPU cap | Input | Output | Stop | Status |
|---|---|---|---|---:|---:|---|---|---|---|
| `TCH-20260815` | `temporac.execution.v4/teacher/seed=20260815` | G1 | X0 U0000--U0023；U0024--U0031 tune-only | 20,000 | 6.0h | S0,K0,X0 roots | run receipt + 40 checkpoints | `STOP-T`,`STOP-D` | `BLOCKED` |
| `TCH-20260816` | `temporac.execution.v4/teacher/seed=20260816` | G1 | same | 20,000 | 6.0h | same | same | `STOP-T`,`STOP-D` | `BLOCKED` |
| `TCH-20260817` | `temporac.execution.v4/teacher/seed=20260817` | G1 | same | 20,000 | 6.0h | same | same | `STOP-T`,`STOP-D` | `BLOCKED` |

### Canonical 与 capacity-control response jobs（6 / 15 A6000h）

每 job 的 training family 严格为 balanced one X0 train source orbit + one certified natural train component/identity；checkpoint selection 只用 X0 U0024--U0031 tune objective。

| Run ID | Exact ASCII job name | Steps | GPU cap | Input | Output | Stop | Status |
|---|---|---:|---:|---|---|---|---|
| `CAN-20260815` | `temporac.execution.v4/response/canonical/seed=20260815` | 10,000 | 2.5h | K2 target roots | run receipt + 20 checkpoints | `STOP-R`,`STOP-D` | `BLOCKED` |
| `CAN-20260816` | `temporac.execution.v4/response/canonical/seed=20260816` | 10,000 | 2.5h | same | same | `STOP-R`,`STOP-D` | `BLOCKED` |
| `CAN-20260817` | `temporac.execution.v4/response/canonical/seed=20260817` | 10,000 | 2.5h | same | same | `STOP-R`,`STOP-D` | `BLOCKED` |
| `CAP-20260815` | `temporac.execution.v4/response/capacity-control/seed=20260815` | 10,000 | 2.5h | K2 target roots | run receipt + 20 checkpoints | `STOP-R`,`STOP-D` | `BLOCKED` |
| `CAP-20260816` | `temporac.execution.v4/response/capacity-control/seed=20260816` | 10,000 | 2.5h | same | same | `STOP-R`,`STOP-D` | `BLOCKED` |
| `CAP-20260817` | `temporac.execution.v4/response/capacity-control/seed=20260817` | 10,000 | 2.5h | same | same | `STOP-R`,`STOP-D` | `BLOCKED` |

### Shortcut response jobs（18 / 45 A6000h）

| Run ID | Exact ASCII job name | Natural training row | Steps | GPU cap | Output | Stop | Status |
|---|---|---|---:|---:|---|---|---|
| `SNP-20260815` | `temporac.execution.v4/response/shortcut/no-pose-timestamp/seed=20260815` | yes | 10,000 | 2.5h | run receipt + 20 checkpoints | `STOP-R`,`STOP-D` | `BLOCKED` |
| `SNP-20260816` | `temporac.execution.v4/response/shortcut/no-pose-timestamp/seed=20260816` | yes | 10,000 | 2.5h | same | `STOP-R`,`STOP-D` | `BLOCKED` |
| `SNP-20260817` | `temporac.execution.v4/response/shortcut/no-pose-timestamp/seed=20260817` | yes | 10,000 | 2.5h | same | `STOP-R`,`STOP-D` | `BLOCKED` |
| `SNU-20260815` | `temporac.execution.v4/response/shortcut/nuisance-only/seed=20260815` | yes | 10,000 | 2.5h | same | `STOP-R`,`STOP-D` | `BLOCKED` |
| `SNU-20260816` | `temporac.execution.v4/response/shortcut/nuisance-only/seed=20260816` | yes | 10,000 | 2.5h | same | `STOP-R`,`STOP-D` | `BLOCKED` |
| `SNU-20260817` | `temporac.execution.v4/response/shortcut/nuisance-only/seed=20260817` | yes | 10,000 | 2.5h | same | `STOP-R`,`STOP-D` | `BLOCKED` |
| `SPS-20260815` | `temporac.execution.v4/response/shortcut/pose-shuffle/seed=20260815` | yes | 10,000 | 2.5h | same | `STOP-R`,`STOP-D` | `BLOCKED` |
| `SPS-20260816` | `temporac.execution.v4/response/shortcut/pose-shuffle/seed=20260816` | yes | 10,000 | 2.5h | same | `STOP-R`,`STOP-D` | `BLOCKED` |
| `SPS-20260817` | `temporac.execution.v4/response/shortcut/pose-shuffle/seed=20260817` | yes | 10,000 | 2.5h | same | `STOP-R`,`STOP-D` | `BLOCKED` |
| `SSC-20260815` | `temporac.execution.v4/response/shortcut/static-code-only/seed=20260815` | yes | 10,000 | 2.5h | same | `STOP-R`,`STOP-D` | `BLOCKED` |
| `SSC-20260816` | `temporac.execution.v4/response/shortcut/static-code-only/seed=20260816` | yes | 10,000 | 2.5h | same | `STOP-R`,`STOP-D` | `BLOCKED` |
| `SSC-20260817` | `temporac.execution.v4/response/shortcut/static-code-only/seed=20260817` | yes | 10,000 | 2.5h | same | `STOP-R`,`STOP-D` | `BLOCKED` |
| `SWM-20260815` | `temporac.execution.v4/response/shortcut/warp-metadata/seed=20260815` | **no; X0-only** | 10,000 | 2.5h | same | `STOP-R`,`STOP-D` | `BLOCKED` |
| `SWM-20260816` | `temporac.execution.v4/response/shortcut/warp-metadata/seed=20260816` | **no; X0-only** | 10,000 | 2.5h | same | `STOP-R`,`STOP-D` | `BLOCKED` |
| `SWM-20260817` | `temporac.execution.v4/response/shortcut/warp-metadata/seed=20260817` | **no; X0-only** | 10,000 | 2.5h | same | `STOP-R`,`STOP-D` | `BLOCKED` |
| `STL-20260815` | `temporac.execution.v4/response/shortcut/track-length/seed=20260815` | yes | 10,000 | 2.5h | same | `STOP-R`,`STOP-D` | `BLOCKED` |
| `STL-20260816` | `temporac.execution.v4/response/shortcut/track-length/seed=20260816` | yes | 10,000 | 2.5h | same | `STOP-R`,`STOP-D` | `BLOCKED` |
| `STL-20260817` | `temporac.execution.v4/response/shortcut/track-length/seed=20260817` | yes | 10,000 | 2.5h | same | `STOP-R`,`STOP-D` | `BLOCKED` |

## D. Budget ledger

| Ledger item | Count | Per-item cap | Planned total | Consumed | Remaining planned | Status |
|---|---:|---:|---:|---:|---:|---|
| Teacher training | 3 | 6.0 A6000h | 18 | 0 | 18 | `BLOCKED` |
| Response training | 24 | 2.5 A6000h | 60 | 0 | 60 | `BLOCKED` |
| Fixed synthetic evaluation inference | 3 | 1 A6000h | 3 | 0 | 3 | `BLOCKED` |
| Natural clean/drift prediction | 3 | 4 A6000h | 12 | 0 | 12 | `BLOCKED` |
| **Allocated** | **27 training jobs + inference** |  | **93** | **0** | **93** |  |
| Outer unallocated margin | n/a | no automatic reassignment | 7 | 0 | 7 | `LOCKED` |
| Campaign hard ceiling |  |  | 100 | 0 | 100 |  |

Disk class caps：S0/fixtures 4 GiB；feature/vault/audit 8；targets 8；teacher checkpoints/logs 6；response checkpoints/logs 20；predictions/receipts 12；G5/evaluator receipts 2；staging/temp 4；总计 64 GiB。当前 consumed 未测量，不能把旧 preflight 的任何数字写入本 ledger。P06 必须 fresh exact-byte 测量后才可建立 consumed 字段。

## E. Tracker completion checks

- [x] 训练表逐行恰好 27 个 exact ASCII job names：3 teacher + 3 canonical + 3 capacity + 18 shortcut。
- [x] 三个且仅三个 seed；inference rows 不计为 training jobs。
- [x] 93 A6000h 分解为 18 + 60 + 3 + 12；没有从 7h margin 预借。
- [x] P00→P01→P02/P03/P04/P05→P06 位于 S0/G0/K0 之前。
- [x] S0→G0→K0→G1→K1→G2→K2→G3→K3→K4→G4→K5→K6→freeze→G5a→grant→G5b→K7 顺序不可绕过。
- [x] 只有 v44 train/val 可进入未来自然数据路径；denylist 路径只允许被静态字符串扫描。
- [x] Human labels 不进入 teacher/response training 或 checkpoint selection；自然 val 不调参。
- [x] 所有状态仍为 `NEXT_NO_AUTHORITY` 或 `BLOCKED`；没有虚构 PASS、metric、artifact 或结果。
- [x] 本 tracker 不授权 implementation、server connection、launch、capability、evaluation 或 rerun。
- [x] Header、NATP、K7 与 final-decision guard 均强制 canonical protocol `GT-bbox-assisted AlphaPose supplied-track pilot` 及 `partial-cache` / `GT-bbox-assisted` / `supplied-track`；current main-paper promotion 固定 `BLOCKED_PARTIAL_CACHE_APPENDIX_ONLY`。
- [x] `P05M-COUNT-METRIC` 是 P2 的零 training-job、零 GPU fixture，覆盖 positive integer/stub/NAE/OBO/video-first/ordered-seed/bootstrap/comparator/clean reuse 与 malformed/duplicate/empty/nonfinite/type global fail；receipt/code/expected hashes绑定 S0→G5a→K7。
