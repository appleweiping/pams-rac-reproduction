# TempoRAC Amendment 002 Operator Implementation Review

> **VERDICT: `IMPLEMENTATION_SEMANTICS_PASS`**
>
> **ARTIFACT AUTHORITY: `BLOCKED_NON_AUTHORITATIVE` (`authority=0`)**
>
> **SAME-FAMILY PROVISIONAL — NO CANONICAL/P2/S0/GATE/DATA/SERVER/TRAINING/RESULT/CLAIM/PAPER/GIT AUTHORITY**

**Date:** 2026-08-16  
**Executor model/family:** `gpt-5.6-sol` / `OpenAI`  
**Reviewer model/family:** `gpt-5.6-sol` / `OpenAI`  
**Review independence:** `same-family`  
**Acceptance status:** `provisional`  
**Fresh zero-context reviewer:** `true`  
**Reviewer task:** `/root/temporac_operator_impl_review`

## 1. 结论

当前 operator 实现的确定性语义通过审查，implementation-semantic blocker 为 `0`。独立重放确认：10-byte `K`、唯一 `uint64_be(10)` 前缀、raw `uint32_be(e)`、`D(e)[0]`、精确 binary32 inactive cap、lower-middle plateau、三类 masks、127-edge/stride-32 windows、binary64 Neumaier NOLA、一次 decoder、truth/component association，以及全部 `10,368` 行均与当前规范一致。

但候选 artifact 仍然没有任何权威性。当前规范冻结了 operator 语义和四个 normative roots，却没有唯一冻结 manifest row schema、每个 array-hash 的规范 preimage、或完整 manifest aggregate preimage。`candidate_schema.json` 正是在代码侧自行声明这些选择；它明确标记 `authoritative=false`、`normative_schema_claimed=false`。因此本审查不能把候选升级为 canonical manifest、P2、S0 或任何 gate receipt。artifact promotion blocker 为 `1`，`authority=0`。

完整 TempoRAC Mypy 还报告一条位于未纳入本次语义输入的 `tests/temporac/test_x0.py:146` 的既有 `arg-type` 错误。该错误不改变 operator implementation-semantics PASS，但阻止宣称“full TempoRAC static suite 全绿”。审查员没有打开或解释该文件内容，只记录 Mypy 的原始诊断。

## 2. 独立性与输入边界

本审查从指定的 raw 当前文件开始，没有读取任何旧 implementation review，也没有采用执行者总结。完整读取了 `experiment-audit`、`experiment-bridge` 及其直接要求的 shared protocols。规范判断仅使用指定 proposal、Amendment 002 MD/JSON、Round 1/2 normative reviews、operator 实现/生成器/两份测试和当前 candidate 目录。

容器复现期间，生成器按其自身 binding 逻辑对 `contract.py`、`cue.py`、`decode.py`、`nola.py` 做了 opaque hash/import/runtime 使用；本审查没有把这些未列入 raw 输入的源码内容作为语义论据。审查员独立 harness 不导入 `pams.temporac`，而是从规范重新实现 operator 算法。

## 3. 规范与实现逐项审查

| 检查 | 结果 | 独立证据 |
|---|---|---|
| 10-byte `K` | PASS | 10,368 个 key 均为 10 bytes、全唯一、按冻结 Cartesian 顺序严格递增；first=`00040001000000000000`，last=`008100040002001f0003`。实现见 `fixtures.py:498`。 |
| `P(e)` / `D(e)[0]` | PASS | 精确 preimage 为 tag + NUL + `uint64_be(10)` + `K` + raw `uint32_be(e)`；五个 amendment witnesses 全部匹配。实现见 `fixtures.py:517-535`。 |
| exact inactive `<f4` cap | PASS | 1,024-byte lookup root 匹配；仅 `b=255` 相对 uncapped 表发生变化，bytes=`cccccc3e`，widened margin=`0.10000002384185791`；实际 inactive `b=255` 共 8,945 edges / 5,665 rows。实现见 `fixtures.py:543-571`。 |
| plateau | PASS | lower-middle=`s+floor((w-1)/2)`；amplitude 1.00 的 width 2/4 共 2,304 rows、6,912 plateaus 均为 exact flat binary32 one，无 unique-maximum 假设。实现见 `fixtures.py:584-629`。 |
| layouts / reset | PASS | interior、left/right-boundary、run-reset 的 run bounds、translation、reset edge 与 truth plateaus 对全部 rows 重建一致。 |
| masks / response | PASS | 每行 target/edge/decoder masks 在 run 内为 1、reset/outside 为 0；response 在 edge mask 外为 canonical float32 zero；thresholded response 与 truth mask 完全相同。 |
| windows / NOLA | PASS | 独立 binary64 Neumaier 重建全部 rows；最低正 denominator=`0.0011528184020849772`；float32 前最大重建偏差=`1.1102230246251565e-16`；quantized bytes 与 generated response 逐字节相等。实现见 `fixtures.py:655-701`。 |
| decoder | PASS | 每行 exactly one invocation，均得到 3 个 components；bounds、lower-middle locations、exact scores 全部匹配。实现验证见 `fixtures.py:703-787`。 |
| association | PASS | 全部 10,368 行的 truth/component bipartite degrees 均为一对一；无 miss/extra/split/merge。 |
| margins | PASS | 全库存 worst positive margin=`0.1499999761581421`，worst negative margin=`0.10000002384185791`。 |
| all row hashes | PASS | 独立 harness 逐行重算并核对 candidate 的 14 个 array/hash fields、edge count、reset edge、negative-component count 和 key。 |
| fail closed / `O_EXCL` | PASS | generator 在既有 output directory 上 exit 2 且不新增成员；`Path.open("xb")` 在既有文件上抛 `FileExistsError` 且原 bytes/hash 不变。实现见 generator `:385-394`。 |

## 4. 四个 normative roots

| Root | Bytes | SHA-256 | 结果 |
|---|---:|---|---|
| semantic-row canonical JSON | 746,714 | `0fe8b2619a3f33cc4a83485053888c04c567b015eea65f81bf6cd9ec715ee4f2` | PASS |
| inactive lookup | 1,024 | `afa6eff0d744acfdf9cf78adc43aa8daf01633f3345d7567d642a7782885040d` | PASS |
| `concat(uint64_be(10)||K_r)` | 186,624 | `18dfdafe237b77565de36bab6d4642efb314ec1cbbd76e852938ed5c4d424a55` | PASS |
| `concat(D_r(0))` raw digests | 331,776 | `ba797f6ae59cbbb40f19a61199afaaa1ac5b2c001a3c9fc34f1dba8f79cf78e4` | PASS |

这些是规范 witnesses，不是 S0 receipts 或科学结果。

## 5. Pinned Docker 再生成与逐字节比较

使用 Docker Engine linux/amd64、`--platform linux/amd64`、image tag `python:3.12.4`，并固定到本机解析的 image digest：

`python@sha256:e8be0ea148390d08bc077840cf87ac6a538d80b0ea1e8752b3e3982987cd0a53`

容器内实际 runtime 为 CPython `3.12.4`、NumPy `2.1.0`、Linux `x86_64`、little-endian。工作树以 read-only mount 输入，复制到隔离 `/tmp/repo`，在全新宿主临时目录 `D:\Temp\temporac_operator_impl_review_c46e56f9afe04a3ebcb83f00b09fd92d\regenerated` 生成。generator exit `0`。候选目录没有被写入。

| Candidate member | Bytes | Workspace SHA-256 | Regenerated SHA-256 | Byte equal |
|---|---:|---|---|---|
| `candidate_receipt.json` | 1,729 | `2e457e01166c6151e4edc688e8fd6f981d1278231fcd8c2bc000179f7a825102` | same | true |
| `candidate_schema.json` | 1,301 | `9f4b2ced821e667f950c1f56bbf9adbdc14b2c387c1932953048caf3f747a8cd` | same | true |
| `environment_lock.json` | 458 | `709db0943752965524b507dd6340554a8d7d218be125317af0fc8b7748a5a947` | same | true |
| `replay_witness.json` | 3,908 | `36f60f41ac111a7b05e5f4846464cc774e09bb9faa44feed10be5a4f35418653` | same | true |
| `temporac.operator-manifest.candidate-v1.json` | 15,039,512 | `1f39378d6f7a970b1f183558ed145bd97107f2aeda41902f4736e6778930e1bb` | same | true |

成员集合也完全相同，无额外或缺失文件。

## 6. Candidate schema / receipt 的非权威边界

生成器在 `:173-181` 把 schema 标为 candidate-only，并说明 Section 9 没有冻结 manifest row keys、per-array serialization 或 aggregate preimage；在 `:412-426` 写入 `authoritative=false`、`authorizes=[]`、`manifest_schema_authoritative=false`、`p2_status=NOT_CLAIMED`、`s0_status=NOT_CLAIMED`。当前 candidate 五个成员都是 canonical LF-terminated JSON，receipt bindings 和 member hashes 全部匹配。

这些良好的 receipt 边界只证明“这个候选由当前代码在记录的 runtime 中可复现”，不证明“它是规范唯一允许的 manifest”。Fresh review 不能凭审查行为制造缺失的规范 schema。要解除 artifact promotion blocker，必须先在独立授权下把 exact manifest schema、array-hash preimages 和 aggregate preimage 写入 canonical contract，重绑所有 consumers，再重新生成和审查。此前：

- candidate 不能改名或解释为 canonical；
- P2/S0/G4/K5/K6 均不能由它通过；
- 不得启动 server、读取 natural/evaluator data、训练、产出 result/claim 或修改 paper；
- `authority=0`。

## 7. 验证命令结果

| 验证 | 结果 |
|---|---|
| Targeted pytest: `test_gates_fixtures.py` + `test_operator_manifest.py` | PASS — `13 passed in 45.07s` |
| Full TempoRAC pytest: `tests/temporac` | PASS with platform skip — `109 passed, 1 skipped in 143.20s` |
| Platform skip | `tests/temporac/test_contract_hashio.py:153: symlink creation is not permitted`（Windows host） |
| Ruff: `src/pams/temporac`, generator, `tests/temporac` | PASS — `All checks passed!` |
| Mypy scoped four reviewed files | PASS — `Success: no issues found in 4 source files` |
| Mypy production package + generator | PASS — `Success: no issues found in 25 source files` |
| Mypy full package + generator + `tests/temporac` | FAIL — 1 diagnostic in `tests/temporac/test_x0.py:146`, incompatible `floating[_64Bit]` to expected `float`; 44 files checked |

Mypy 同时输出 `pyproject.toml: note: unused section(s): module = ['yaml']`；这是配置提示，不是 operator semantic finding。

## 8. 阻塞项与审查状态

### Implementation semantics

- Blocking issues: `0`
- Verdict: `PASS`

### Artifact authority / promotion

- `AUTH-1`（blocking）：canonical amendment/proposal 未唯一冻结 manifest schema、array-hash preimages 和 aggregate preimage；当前 candidate 必须保持 non-authoritative，不能成为 canonical/P2/S0。
- Authority: `0`

### Repository-wide static health

- `STATIC-1`（non-operator, non-semantic）：full TempoRAC Mypy 仍有一条 `test_x0.py:146` 诊断；在修复前不得声称 full static suite 全绿。

## 9. 最终输入哈希

初始与持久化前复核哈希相同；审查过程中所有 raw 输入均未变化。

| Input | SHA-256 |
|---|---|
| `refine-logs/temporac/FINAL_PROPOSAL.md` | `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391` |
| `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY.md` | `899d6f071c89f898bf3825adfb30a326904f9bc7fe44d44351da8a5d0d9ccb14` |
| `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY.json` | `d524556f292ec17a155cdcd962cf8a88bea26f32b810ffb7bd8e45744f1c54fc` |
| Round 1 review MD | `5e60841696333234213fa36675f11311d294e76adfd1b68d8ab74198a409f253` |
| Round 1 review JSON | `41468e0c7a163040cefb035936a35025ad3910da96488e1582c02b83a15c5f65` |
| Round 2 review MD | `a86da733f3ae048b867ec8b2366a9f0262ca2b390ef08687ca7d710654e8301f` |
| Round 2 review JSON | `3730d96eff1d3815499602eaf183774af032c514a588d4e92fe8a1cccf053e64` |
| `src/pams/temporac/fixtures.py` | `979b8cee4fed117f5471cf399794f615c60eae52dbba2c946109ff836206b464` |
| generator | `ffc5b83025375b2c68376f093becb63905504e15be45237e965991c6f762f8c7` |
| `tests/temporac/test_gates_fixtures.py` | `941856b6111cff8697f105bc30dc679069f73206bab3f9c2b930236675e13173` |
| `tests/temporac/test_operator_manifest.py` | `a055247946580679d54204850bfe588a102a37d342da31e06eddba85c45f9dfe` |
| candidate receipt | `2e457e01166c6151e4edc688e8fd6f981d1278231fcd8c2bc000179f7a825102` |
| candidate schema | `9f4b2ced821e667f950c1f56bbf9adbdc14b2c387c1932953048caf3f747a8cd` |
| candidate environment lock | `709db0943752965524b507dd6340554a8d7d218be125317af0fc8b7748a5a947` |
| candidate replay witness | `36f60f41ac111a7b05e5f4846464cc774e09bb9faa44feed10be5a4f35418653` |
| candidate manifest | `1f39378d6f7a970b1f183558ed145bd97107f2aeda41902f4736e6778930e1bb` |

## 10. 最终处置

**Implementation semantics: PASS.**  
**Artifact authority: BLOCKED / non-authoritative.**  
**Acceptance: same-family provisional.**  
**Authority: 0.**

本审查不修改 code、tests、candidate、MANIFEST、Git、server 或 data；也不授予任何后续执行权限。
