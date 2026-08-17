# TempoRAC Wave 0/1 contract/preimage/operator-consumer implementation review

日期：2026-08-16  
审查者：`/root/temporac_wave1_contract_impl_review`（fresh zero-context `gpt-5.6-sol`, ultra）  
结论：**REVISE**  
独立性：OpenAI same-family；因此结论仅为 **provisional**  
权限边界：candidate-only / authority 0

## 结论摘要

有效合同的五行 canonical index、冻结 operator-v2 consumer、A006 的 28-path 历史快照覆盖语义，以及两处 X0 历史冻结断言均实现正确。但是，严格 canonical JSON 与 receipt DAG 的三个 fail-open 会让规范明确要求拒绝的输入通过：canonical `-0.0`、未按 typed owner payload 绑定的任意 `owner_key`，以及额外 DAG node / 不同 role 的额外 edge。因此本轮不能接受 Wave 0/1 contract/preimage 实现，结论为 **REVISE**。

operator-v2 的既有 candidate-only 资格不因本结论升级或撤销；它仍然只是后续独立 P2 的可选输入。本审查不授权 P2、P2-METRIC、S0、gate、server、data、GPU、training、evaluation、results、launch、Git 或论文主张。

## 阻断项

### W1-B1 — canonical JSON 接受负零

`src/pams/temporac/hashio.py:227-240` 使用标准 `json.dumps(..., allow_nan=False)` 生成 canonical bytes；`src/pams/temporac/hashio.py:256-284` 在 canonical 模式中仅把解析结果重新编码并比较原始 bytes。Python 会把 binary64 负零重新编码为 `-0.0`，因此 `b'{"x":-0.0}\n'` 可原样往返并被接受。`src/pams/temporac/receipts.py:212-215` 只检查 finite；`src/pams/temporac/types.py:557-565` 的非负检查使用 `value < 0.0`，同样不会拒绝 `-0.0`。

独立攻击探针结果：

- `b'{"x":-0}\n'`：拒绝（canonical mismatch）。
- `b'{"x":-0.0}\n'`：**接受**，结果为 `{'x': -0.0}`。
- `b'{"x":-0e0}\n'`：拒绝（canonical mismatch）。

A005 fixed MD 的 machine mirror 明确规定数学零必须编码为 `+0.0` 且 `-0.0` 必须拒绝；score 规则也明确列出 `-0.0` fail。当前测试覆盖 duplicate、non-finite、BOM 和空白变体，但没有覆盖负零。

修订要求：在 canonical serialization 前和 strict parse 后递归拒绝任何 `value == 0.0 and signbit(value) < 0` 的 float，并在 receipt 的 finite/nonnegative float 语义层同步拒绝；补充顶层、嵌套对象、数组和 receipt 字段的 `-0.0` 攻击测试。

### W1-B2 — receipt DAG 的 `owner_key` 未绑定 typed owner payload

A005 fixed JSON 的 `receipt_dag_contract.owner_key_encoding` 规定：`owner_key` 必须是 `H_hex("temporac.receipt-owner.v4", class ASCII, exact typed owner fields)`，并按 node class 使用对应 typed fields；它绝不是 caller/free-form 或 slash-joined string。

`src/pams/temporac/receipts.py:773-804` 只验证 `class` 属于 closed token set、`owner_key` 是 lowercase SHA-256 字符串、排序与 `(class, owner_key)` 唯一性。`src/pams/temporac/receipts.py:836-856` 只用 inventory receipt digest 与预期 edge 做部分交叉检查，没有从 typed owner fields 派生 owner key，也没有把 inventory row 的 owner/class/schema/payload 与 DAG node 绑定。

独立攻击探针把一个 P00 node 改成 `class="feature"`、`owner_key="f" * 64`；其 receipt digest 与图结构保持不变，`validate_receipt_dag(..., inventory=...)` **接受**。正向单元测试 `tests/temporac/test_prediction_receipts.py:283-287` 还使用 `sha256(row.owner.encode())` 构造 owner key，这不是规范的 domain-separated typed preimage。

修订要求：实现并使用各 class 的 typed owner-key constructor；对每个 inventory receipt 验证 exact class、typed owner fields、receipt schema/payload 与 owner key 的外键一致性，并增加替换 class、owner key、schema、typed field 和 payload-to-owner mismatch 的负向测试。

### W1-B3 — receipt DAG 未执行 exact node/edge closure

A005 fixed MD 第 8 节明确要求 54-owner roster 生成恰好 106 条 direct backward edges，并规定 missing/extra node、role、edge、ordinal、schema、payload equality、multiplicity 或 cycle 均失败。machine mirror 也要求任何额外 node/edge 失败。

当前 `src/pams/temporac/receipts.py:836-839` 使用 `inventory_digests.issubset(receipt_order)`，因此允许额外节点。`src/pams/temporac/receipts.py:840-856` 只逐 source 检查该 inventory row 的一个预期 role，并且只统计目标也在 inventory 中的 edge；不同 allowed role 的额外 edge 不进入比较，因而可通过。

独立攻击探针结果：

- 在正确 54-node/106-edge 图中加入一个合法 token、合法 digest、断开的额外 node：**接受**为 55 nodes / 106 edges，topological sort 消耗 55 个节点。
- 加入一条从最后 inventory receipt 指向第一 receipt 的额外、无环 `prediction.feature` edge：**接受**为 54 nodes / 107 edges。

当前实现能够拒绝缺失预期 edge、完全重复 edge、dangling endpoint、cycle 和不连续 ordinal；问题是它没有拒绝完整闭包以外的合法形状输入。现有 DAG 测试只覆盖基准、一个 missing edge 和 cycle，没有覆盖 extra node、不同-role extra edge 或 owner substitution。

修订要求：由 exact artifact/index rows 与 54-row roster 机械派生完整 node/edge multiset，要求节点数 54、direct edge 数 106、每个 node/class/owner/schema/payload exact，且 observed node/edge multiset 与期望完全相等；补齐 extra/duplicate/substitution/wrong-role/wrong-ordinal/dangling/cycle 攻击矩阵。

## 已确认符合的实现

### Effective contract 与 canonical preimage

- `src/pams/temporac/contract.py:10-36` 保持 `PROPOSAL_SHA256=3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391`。
- canonical index 恰好五行，顺序为 proposal、A005 fixed MD/JSON、A005 Round-6 ACCEPT MD/JSON；重建结果恰为 665 bytes，root 为 `c203f4aab00879acf041b90829f0124d0b1de231294f191b03f3166f728e3d5c`。
- `src/pams/temporac/hashio.py:288-340` 从固定 role/byte-count/digest 重建并进行 exact-byte equality；mutable aliases、旧 review、plan、tracker 均未进入 index。
- duplicate keys、NaN/Infinity、BOM、非 UTF-8、extra/missing/wrong type、wrong key order、null/hex 和非 canonical bytes 的现有拒绝路径均成立；唯一已证实的 canonical number 缺口是 W1-B1。
- `write_bytes_exclusive` 的 Windows 临时探针得到 mode `0444`、`os.access(..., W_OK)==False`，且以 `O_WRONLY` 重开得到 `PermissionError`；未发现通过 mutable alias 重开 effective-contract 输出的路径。

### Inventory、NATP 与 K1 的正向约束

- pre-G5a inventory 强制恰好 54 个唯一 owner，class counts 为 17 pre-g5a-stage、27 run、1 G1、1 K1、3 X0I、1 K3、1 K4、3 NATP。
- roster 机械产生恰好 106 个 direct backward edges，并检查 full topological consumption；K1 的 G0/G1 upstream ordinals 固定为 0/1。
- NATP receipt 使用 exact 14 keys，并要求唯一 `natp.k6` upstream；K1 是独立 exact 13-key canonical JSON+LF preimage。缺失/额外 key、null、wrong type、wrong hex 与基础 ordinal/acyclicity 检查均存在。
- 上述正向约束不能抵消 W1-B2/W1-B3 的 owner binding 与 exact closure 缺口。

### Frozen operator-v2 consumer

`src/pams/temporac/receipts.py:1241-1386` 的 consumer 符合冻结输入语义：

- 只接受目录名 `operator_candidate_v2_20260816`、非 symlink regular directory、且恰好五个成员；逐成员验证固定 byte count 和 SHA-256，并在读取前后检查目录 identity/name set 稳定。
- 固定 candidate receipt 为 6,180 bytes / `d53fcc1138754eee90a255952606cd054aa04c72cb3f97d66bf4464d5de672cb`；generation snapshot foreign key 为 `ded975e31ced2cf1c7d69fb217610125f65d419f641a8e0832d115ec7a4870c0`。
- manifest 为 15,039,512 bytes / `1f39378d6f7a970b1f183558ed145bd97107f2aeda41902f4736e6778930e1bb`，并验证 10,368 rows、exact row keys/order/domain 与 replay witness/root crosswalk。
- detached external review raw bytes 固定为 MD 15,244 bytes / `73c660079377683b2066e4e04e8b6400f3568e1b7473069303908f773d0caad1`、JSON 14,108 bytes / `25792a855e15769ec778276b5066052d15796eea18cad5d44a16f741cb8bd35f`。JSON 在 raw byte/hash 已可信后仅作 duplicate-free UTF-8 parse，这是正确的 foreign-key 信任边界。
- 不重生 candidate，也不把 historical 28-path generation snapshot 与当前 live source 做 equality；这与 A006 的 override 语义一致，且不会把 candidate/review 提升为 P2 或 S0。
- 返回的 `FrozenOperatorCandidate` 是 frozen dataclass，成员为 tuple/frozen records，authority tuple 中所有 bit 必须为 false；handle 为深不可变 candidate-only 表示。

五个冻结成员在审查时均为 readonly：

| member | bytes | SHA-256 |
|---|---:|---|
| `candidate_receipt.json` | 6,180 | `d53fcc1138754eee90a255952606cd054aa04c72cb3f97d66bf4464d5de672cb` |
| `environment_lock.json` | 3,976 | `47c2227b74b627a03bf9b5e4930dd02484caaae5a695ae5c0129f04781115cd8` |
| `operator_manifest_schema.json` | 3,607 | `3b29937c641274e73d10afd19bea943f28288b3d2851e8a5bf2313d7ea3e0fc3` |
| `replay_witness.json` | 2,896 | `348254cfe64cea35d6e29861755489721ebbbd4031f3ec1c044d7fa24fe38010` |
| `temporac.operator-manifest.v4.json` | 15,039,512 | `1f39378d6f7a970b1f183558ed145bd97107f2aeda41902f4736e6778930e1bb` |

### X0 历史冻结与兼容性

- `tests/temporac/test_x0_manifest.py:143-145` 仍冻结历史 candidate 的 `contract_module_sha256=5ab8fb62...745f`；`tests/temporac/test_x0_manifest.py:151` 仍冻结 proposal `3b265bb...9391`。两处变更只解除对当前 live module hash 的 self-lock，没有改写历史 candidate 或 proposal。
- 完整 `tests/temporac` 通过，既有 public surface 保持兼容；`src/pams/temporac/__init__.py` 没有新增 eager import。
- 对审查目标与真实内部依赖的 import/token 扫描未发现 legacy、WARP、GRU 或 router import，也未发现 authority 提升或越界写入。

## 验证结果

| 检查 | 结果 |
|---|---|
| targeted pytest：四个指定测试文件 | `30 passed, 1 skipped in 297.52s` |
| 完整 `tests/temporac` | `126 passed, 1 skipped in 288.53s` |
| Ruff：`src/pams/temporac tests/temporac` | `All checks passed!` |
| scoped Mypy：4 个实现文件 + 4 个指定测试文件 | `Success: no issues found in 8 source files` |
| full Mypy：`src/pams/temporac tests/temporac` | 43 files 中 1 个越界 baseline error |

唯一 pytest skip 位于 `tests/temporac/test_contract_hashio.py:200`：Windows 当前权限不允许创建 symlink，测试先实际尝试后才 skip；这是平台能力限制，不是被测逻辑失败。

full Mypy 唯一错误为 `tests/temporac/test_x0.py:146`：`_separation_rms` 的 NumPy `floating[_64Bit]` 实参与 `float` 注解不兼容。该文件不在 Wave1 变更/指定测试范围，且同一命令的 scoped 8-file 运行完全通过；因此记录为 out-of-scope baseline，不作为本轮新 blocker。一次使用 Windows 设备名 `NUL` 作为 Mypy cache 目录导致 Mypy 1.20.2 internal error，已由合法的 `--no-incremental` 两次运行取代，不计入结果。

## 固定输入哈希

关键权威输入均独立重哈希：

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
| operator V2 external review MD | 15,244 | `73c660079377683b2066e4e04e8b6400f3568e1b7473069303908f773d0caad1` |
| operator V2 external review JSON | 14,108 | `25792a855e15769ec778276b5066052d15796eea18cad5d44a16f741cb8bd35f` |
| canonical proposal | 79,366 | `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391` |
| canonical plan | 34,198 | `4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e` |
| canonical tracker | 22,360 | `714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b` |
| research contract | 12,060 | `3e037a2f74c49dd62e596dc7b875d94216e45b7f693b80a7afb6a60f8244367b` |

A002/A004/A006 accepted operator lineage 也逐文件重哈希并与其固定引用一致：A002 `899d6f...ccb14` / `d52455...54fc` 与 R2 ACCEPT `a86da7...301f` / `3730d9...3e64`；A004 `ece002...3c82` / `61ce25...83f4` 与 R2 ACCEPT `6c45d2...68a7` / `f62135...d50f`；A006 `9b1e3a...0517` / `e44345...9fa61` 与 ACCEPT `ef84a0...388f` / `b59d38...57353`。

被审实现与指定测试的终态哈希：

| path | bytes | SHA-256 |
|---|---:|---|
| `src/pams/temporac/contract.py` | 22,199 | `64a579d2e31f2b46baa1442794327f19096e15ad45a1443d4b958b35e649eb99` |
| `src/pams/temporac/hashio.py` | 31,161 | `3be75602d1dce681cb884a767e857d01194336ef26999b3f71106dac44971f6e` |
| `src/pams/temporac/types.py` | 41,752 | `556816d4be9826d638430ca5a5c7464da793d2da0519eb568ea9228fd6f780b2` |
| `src/pams/temporac/receipts.py` | 68,594 | `fe37c6091108599e663e586b2e8d591bd62b9707ff6b497d3b6a0a7147ff5a36` |
| `tests/temporac/test_contract_hashio.py` | 7,715 | `805cbc85c7c89b408d066324d07844fb673a75f9010b78acc99abf301e061e64` |
| `tests/temporac/test_prediction_receipts.py` | 14,063 | `b2f8fb0f87b5cf168745a7732f3d02fff0b5aed1a37a35860e7cff28f71b5f1f` |
| `tests/temporac/test_operator_manifest.py` | 25,738 | `22072a0c6fd86889d1881b80bb41d48644c8b686c4199c934da073434b30f56a` |
| `tests/temporac/test_x0_manifest.py` | 7,139 | `d64412237c2479129b0b2bae12bd5bcf7490a0e84ab765efd798acb63dc3b9aa` |

这些 bound inputs 在初始固定和最终复核之间 byte/hash 完全一致，未检测到 bound-input drift。审查期间出现一个未绑定的 timestamped A005 JSON copy（`..._20260816_215501.json`）；它不在 effective-contract index、审查输入集合或本轮输出集合中，且没有改变任何 bound input，因此不构成输入漂移。工作树在本审查进入前已是 dirty；本审查未修改生产代码、测试、candidate、docs、plan、tracker、MANIFEST 或 Git。

## 最终处置与权限上限

**REVISE**。在 W1-B1、W1-B2、W1-B3 修复并由 fresh reviewer 对固定新 bytes 复审前，不接受 Wave1 contract/preimage implementation。冻结 operator-v2 consumer 子切片可维持其既有 candidate-only / eligible-for-separate-P2 表示，但不得据此声明 P2、P2-METRIC、S0 或任何 gate 通过。

本审查的 authority ceiling 为全零：`P0=0, P1=0, P2=0, P2-METRIC=0, P3=0, S0=0, gate=0, server=0, data=0, GPU=0, training=0, evaluation=0, results=0, launch=0, Git=0, paper_claim=0`。
