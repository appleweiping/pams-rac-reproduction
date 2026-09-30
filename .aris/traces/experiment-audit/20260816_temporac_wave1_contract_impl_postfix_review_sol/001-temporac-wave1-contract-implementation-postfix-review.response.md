# TempoRAC Wave1 contract implementation postfix 独立复审

- 日期：2026-08-16
- 审查对象：下表所列 exact current bytes；不采信实现者总结
- reviewer：fresh zero-context `gpt-5.6-sol` / ultra，OpenAI same-family
- 独立性：same-family，`provisional`；不构成跨家族独立认证
- 实质裁决：**PASS**
- blocker：**0**；Round1 的 `W1-B1`、`W1-B2`、`W1-B3` 均已闭合
- 权限上限：candidate-only / authority 0

## 结论

对固定新 bytes 的源码逐行复审、权威原件交叉核对、两组独立攻击探针、operator/X0 回归探针以及测试与静态检查均支持 **PASS**。本裁决只接受当前 Wave1 contract/preimage postfix 实现对 `W1-B1/B2/B3` 的闭合；它不声称未来 full receipt DAG 已实现，不产生 P2、P2-METRIC、S0、gate、server、data、GPU、training、evaluation、results、launch、Git 或 paper-claim 权限。

冻结 operator-v2 consumer 仍只是 `candidate_only=true`、`eligible_for_separate_p2=true` 的未来独立 P2 输入。A003/R4 与 A005/R6 的 ACCEPT 仍是规范接受，不是执行或 gate 授权。

## Round1 blocker 闭合

| blocker | 裁决 | 独立复审证据 |
|---|---|---|
| `W1-B1` canonical negative zero | **CLOSED** | `hashio.py:343-424` 在序列化前与严格解析后递归遍历 mapping 的键和值及嵌套 list/tuple，拒绝所有 binary64 `-0.0`；顶层、object、nested object、array、nested array 均被独立探针拒绝。正零保持可用；词法整数 `-0`、嵌套 `-0` 与 `-0e0` 因非 canonical 或 negative-zero 被拒绝。duplicate、NaN/Inf、BOM、非 canonical whitespace 继续 fail-closed。`ResourceRecord`、`CheckpointRecord` 和 run receipt 的 resource 语义层均拒绝 `-0.0`。 |
| `W1-B2` typed receipt owner / raw-byte closure | **CLOSED** | 独立实现了 `SHA256(tag_ASCII || NUL || Σ(uint64_be(len(field)) || field))`，消歧攻击成立；16/16 class KAT 与冻结值一致。A005 原始 JSON 的 54-row owner roster 与代码常量逐字段一致，独立计算的 54 个 owner key 与 `receipt_owner_key` 全部一致且 54 个唯一。run 使用 exact decoded job bytes + uint64 seed；X0I/NATP 使用 uint64 seed；stage/G1/K1/K3/K4 使用 exact roster owner；其余 class 使用各自 typed fields。API 不接受 caller digest、freeform run job、错 class owner 或错 schema。`validate_pre_g5a_receipt_evidence` 从 digest 对应的 immutable canonical raw receipt bytes 重新验证 schema/closed payload 并派生 typed owner；class、owner、schema、payload、job、seed、upstream 攻击全部拒绝。 |
| `W1-B3` exact evidence-backed projection | **CLOSED** | inventory 与 receipt-byte evidence 都是必需输入；baseline 恰为 54 nodes、106 direct edges、54 份 immutable raw receipt bytes。观察到 role 计数为 `stage.upstream=68`、`run.upstream=30`、`g1.teacher-run=3`、`k1.upstream=2`、`natp.k6=3`，合计 106。missing/extra/substitute node，wrong class/owner，missing/extra/substitute edge，wrong role/ordinal，dangling 与 cycle 全部拒绝；evidence missing/extra/mutable/digest mismatch 也全部拒绝。`inventory=None` 或缺 evidence 失败。实现及 docstring 明确限定为 evidence-backed 54/106 pre-G5a projection；加入未来 full-DAG node 失败，不把本实现冒充未来完整 DAG。 |

### W1-B1：数值与 canonical transport

独立探针共记录 34 项检查，其中 29 项为必须拒绝的攻击：

- serializer 前：顶层、object、nested object、array、nested array 中的 float `-0.0` 全拒绝；mapping key 也在递归路径中。
- strict parse 后：同样五类结构全拒绝；`0.0` 保持 canonical。
- integer `-0` / nested `-0` / `-0e0`、duplicate key、NaN、Infinity、BOM 与空白变体保持拒绝。
- `ResourceRecord.gpu_seconds/wall_seconds`、`CheckpointRecord.tune_objective`、run receipt `resource.gpu_seconds/wall_seconds` 的 `-0.0` 全拒绝。

### W1-B2：H、typed owner 与原始 receipt bytes

`src/pams/temporac/hashio.py:138-259` 的 H framing 与 owner dispatch 由不调用生产 `H` 的独立实现重算。固定 16 类覆盖：feature、natural-certificate-input、teacher-checkpoint、teacher-tune-evaluation、run、prediction、target、x0-inference、natural-prediction-completion、pre-g5a-stage、G1、K1、K3、K4、teacher-tune-input、teacher-selection。

`src/pams/temporac/types.py:616-740` 提供封闭的 typed owner records；`src/pams/temporac/receipts.py:853-1006` 只在 raw receipt bytes 的 digest、strict canonical parse、schema、exact keys、owner/seed/job/upstream 全闭合后派生 owner key。独立攻击脚本重建了完整 54-receipt digest 链后再更换 run job、run seed、NATP owner/schema/extra payload/upstream；这样 inventory 自身仍保持 exact/coherent，而 evidence semantic validation 对六项攻击全部给出 `closed typed payload` 拒绝，排除了仅靠旧 digest 或 caller owner 字符串“通过”的路径。

### W1-B3：54/106 exact equality 与未来边界

`src/pams/temporac/receipts.py:750-1127` 先冻结 54-row inventory、其 upstream owner/digest arrays 与 106 条 edge，再从 exact digest→immutable bytes evidence 得到 typed node roster。最终 DAG 对 expected tuple 做 exact equality，不是 membership-only 检查。

第二组独立探针 baseline 为 54/106/54，另执行 25 个必须失败的标签：`inventory-none`、`evidence-none`、四类 evidence 篡改、五类 node 篡改、七类 edge 篡改、六类 raw receipt 闭环篡改及 `future-full-dag-node`；25/25 均拒绝。cycle 攻击明确返回 dependency cycle，dangling 明确返回 dangling endpoint。

当前实现只消费 pre-G5a evidence projection。未来 authoritative artifact indexes 不存在时，不得省略 inventory/evidence 或切换 generic/full mode；本次 PASS 不覆盖未来 full DAG。

## 回归与权限边界

### Effective contract

独立从五个权威文件的 bytes/hash 重建 canonical index，得到 exact `665` bytes 和：

`c203f4aab00879acf041b90829f0124d0b1de231294f191b03f3166f728e3d5c`

五行顺序恰为 canonical proposal、A005 fixed MD、A005 fixed JSON、A005 R6 ACCEPT MD、A005 R6 ACCEPT JSON。proposal root 保持 `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391`；plan、tracker、mutable aliases 与旧 review 均未进入 effective index。

### Frozen operator-v2 / A006

`load_frozen_operator_candidate` 对 exact 五成员目录、10,368 rows、成员 bytes/hash、candidate receipt 与 detached V2 review 做封闭校验：

| artifact | bytes | SHA-256 |
|---|---:|---|
| `candidate_receipt.json` | 6,180 | `d53fcc1138754eee90a255952606cd054aa04c72cb3f97d66bf4464d5de672cb` |
| `temporac.operator-manifest.v4.json` | 15,039,512 | `1f39378d6f7a970b1f183558ed145bd97107f2aeda41902f4736e6778930e1bb` |
| `operator_manifest_schema.json` | 3,607 | `3b29937c641274e73d10afd19bea943f28288b3d2851e8a5bf2313d7ea3e0fc3` |
| `environment_lock.json` | 3,976 | `47c2227b74b627a03bf9b5e4930dd02484caaae5a695ae5c0129f04781115cd8` |
| `replay_witness.json` | 2,896 | `348254cfe64cea35d6e29861755489721ebbbd4031f3ec1c044d7fa24fe38010` |

detached V2 review 为 MD `73c660079377683b2066e4e04e8b6400f3568e1b7473069303908f773d0caad1`、JSON `25792a855e15769ec778276b5066052d15796eea18cad5d44a16f741cb8bd35f`。A006 的 28-file generation snapshot 只作为 frozen foreign key `ded975e31ced2cf1c7d69fb217610125f65d419f641a8e0832d115ec7a4870c0` 被验证；consumer 不调用 generator，也不把当前 live source 与历史 snapshot 做 equality。

返回 handle 的 authority true bits 为空。candidate/review 只保留 future separately authorized P2 eligibility，不提升 P2、S0 或 gate。

### X0 与依赖回归

X0 historical candidate 继续冻结旧 `contract_module_sha256=5ab8fb62dc558ba78e95fc07e52c1f1a1afbff5b1c5003834079eea6143d745f`，proposal 与 proposal-copy 均为 `3b265b...9391`；`authoritative=false`、`authorizes=[]`、P2/S0 均 `NOT_CLAIMED`。这保留历史断言，同时解除对当前 live contract module 的自锁。

对 `src/pams/temporac/*.py` 做 AST import-only 扫描，未发现 legacy、WARP、GRU 或 router 实现依赖。规范要求的 shortcut token `warp-metadata` 仍存在于冻结 job inventory；它不是 WARP 实现 import，不能混为一谈。未观察到 authority 提升或越界写入。

## 独立运行结果

环境：Python 3.12.13、pytest 8.4.2、Ruff 0.16.3、Mypy 1.20.2。

| 命令/检查 | 结果 |
|---|---|
| 四文件 targeted pytest：`test_contract_hashio.py test_prediction_receipts.py test_operator_manifest.py test_x0_manifest.py` | **38 passed, 1 skipped in 64.07s** |
| 完整 `pytest tests/temporac -q` | **134 passed, 1 skipped in 191.42s** |
| `ruff check src/pams/temporac tests/temporac` | **All checks passed** |
| scoped Mypy：4 个实现文件 + 4 个指定测试文件 | **Success: no issues found in 8 source files** |
| full Mypy：`src/pams/temporac tests/temporac` | 43 files 中复现 1 个既有越界 baseline error |

pytest 唯一 skip 是 Windows 环境不允许创建 symlink，来自 `test_contract_hashio.py` 的显式 platform/permission skip，不削弱上述攻击覆盖。

full Mypy 唯一错误为 `tests/temporac/test_x0.py:146`：NumPy `floating[_64Bit]` 实参传给注解为 `float` 的 `_separation_rms`。它与 Round1 同一命令记录的路径、行号和错误完全相同；该文件未在本 postfix 变更范围内，且 scoped 8-file Mypy 零错误，因此记为 out-of-scope baseline，不构成本轮 blocker。

## Exact reviewed bytes 与终态重哈希

### 实现和指定测试

| path | bytes | SHA-256 |
|---|---:|---|
| `src/pams/temporac/contract.py` | 22,199 | `64a579d2e31f2b46baa1442794327f19096e15ad45a1443d4b958b35e649eb99` |
| `src/pams/temporac/hashio.py` | 37,490 | `dd8c3ac3115fbd4e602e76b6106b7d0afafe1c6847fc51a6d0143b94ee73d37e` |
| `src/pams/temporac/types.py` | 46,602 | `6d575ff34ccdcadaada925e5d3b97fb59c76580cbe48ce119cec3d5ed50cf84b` |
| `src/pams/temporac/receipts.py` | 80,371 | `ab4b905d603aa7210e2953d4508d73cddefd60731d06d8315904f2a1a9267ab2` |
| `tests/temporac/test_contract_hashio.py` | 13,845 | `64c724cf1f7c622bce633eeb5c3d25c0b546afb2adedaae648bcad50f0726c29` |
| `tests/temporac/test_prediction_receipts.py` | 25,791 | `962baefc126a25203f2aec6de7257b57c1f7064cdac9283dd340046239746249` |
| `tests/temporac/test_operator_manifest.py` | 25,738 | `22072a0c6fd86889d1881b80bb41d48644c8b686c4199c934da073434b30f56a` |
| `tests/temporac/test_x0_manifest.py` | 7,139 | `d64412237c2479129b0b2bae12bd5bcf7490a0e84ab765efd798acb63dc3b9aa` |

### 权威与回归输入

| input | bytes | SHA-256 |
|---|---:|---|
| A005 fixed MD | 312,728 | `2eaa3cfb746804fdc5890fec4c252c877a031a4abedef7b9ef6eeac0a4839395` |
| A005 fixed JSON | 261,196 | `638efc8708f74c8514247f4f4c5e2311595e90edc6744c9b371886f0789af164` |
| A005 R6 ACCEPT MD | 15,666 | `4a3199c3ac18e3fd2a5bb7ecf6d66e10bc4a5cebb82363803c47867234b70e67` |
| A005 R6 ACCEPT JSON | 10,036 | `fcf8a9d4949f6b456f18bfb4591c512b6628c6ec7eb5a055873a995992f46ad6` |
| A003 fixed MD | 40,558 | `a6b644aaaa29cf6c0423cdd3b287b183a88a9f23cbee3a04afc6db932a4d9005` |
| A003 fixed JSON | 114,825 | `14d59552242965a717f1162208770cb8ded9ebce9f7e060f904ba327617a052a` |
| A003 R4 ACCEPT MD | 23,690 | `4fe4e173c98e5f84257af604a18001d6139d27cbcb7c508508828df408440467` |
| A003 R4 ACCEPT JSON | 9,294 | `d5cfb6c15ea12598dc8765087a91b8bc434c0e8a5dc6506a34e0b07d4e4dd108` |
| canonical proposal（两个 alias byte-identical） | 79,366 | `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391` |
| plan | 34,198 | `4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e` |
| tracker | 22,360 | `714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b` |
| research contract | 12,060 | `3e037a2f74c49dd62e596dc7b875d94216e45b7f693b80a7afb6a60f8244367b` |
| Wave1 Round1 REVISE MD | 15,156 | `0b160e9f88264769ea91e586708173c06468e56f22d7533121c233d830b05ef9` |
| Wave1 Round1 REVISE JSON | 9,917 | `04343f3f93ad0cb91a70621a26032ec16414182a5b1262aa9b8e19535cbd0698` |
| operator V2 review MD | 15,244 | `73c660079377683b2066e4e04e8b6400f3568e1b7473069303908f773d0caad1` |
| operator V2 review JSON | 14,108 | `25792a855e15769ec778276b5066052d15796eea18cad5d44a16f741cb8bd35f` |

审查入口与所有测试/攻击完成后的终态重哈希逐项一致，`final_bound_snapshot_match=true`，未检测到 concurrent bound-input drift。HEAD 为 `238879a475e3bb1fde24da76a8b3258b7fe68dd3`；工作树在本审查开始前已 dirty，本审查未修改生产代码、tests、candidate、docs、plan、tracker、MANIFEST 或 Git，只创建获准的 postfix review/trace 并向既有 event ledger 追加一条审计事件。

## 最终 disposition

**PASS**：`W1-B1`、`W1-B2`、`W1-B3` 对上述 exact bytes 全部 CLOSED，未发现新 blocker。

此 PASS 为 OpenAI same-family provisional review，authority ceiling 全零：`P0=0, P1=0, P2=0, P2-METRIC=0, P3=0, S0=0, gate=0, server=0, data=0, GPU=0, training=0, evaluation=0, results=0, launch=0, Git=0, paper_claim=0`。未来 full DAG、P2、S0、训练、数据、server 或 gate 仍须各自独立授权与审查。
