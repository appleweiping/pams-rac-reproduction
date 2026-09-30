# TempoRAC Contract Amendment 005 Certificate Provenance — Round 3 Normative Review

Date: 2026-08-16  
Reviewer: `/root/temporac_certificate_amendment005_review`  
Route: Codex `gpt-5.6-sol`, OpenAI same-family, ultra review tier  
Assurance: provisional; not cross-family independent  
Verdict: **REVISE**  
Blocking groups: **2**  
Authority: **none**

## Outcome

The revised Amendment 005 closes most of the Round 2 defects, including the score/replay arithmetic, active floating-point state, the numeric 54-owner inventory, the K1 row preimage, the 9,648-row prediction product, capability ordering, and direct `MappingProxy` composition. It is not yet acceptable as an executable provenance contract.

Two blocking groups remain:

1. the executable-origin/runtime closure names inputs that do not exist in any closed committed schema and begins too late to observe the exact invocation it claims to cover; and
2. the typed pre-G5a inventory/DAG is internally inconsistent for NATP and does not assign a legal direct-edge role to K1's two declared roster upstreams.

These are authority-bearing gaps. A conforming implementation cannot uniquely construct or verify the claimed runtime and receipt graph from the current bytes. The verdict is therefore `REVISE`, not conditional acceptance.

## Reviewed bytes and integrity

I reread the project-local `research-refine`, `research-review`, and `experiment-audit` skills and the linked reviewer-routing, assurance, independence, tracing, output, scope, and integration protocols. I then reread the complete Round 3 amendment pair, the Round 2 review and trace, canonical proposal, plan, tracker, cited postfix audit pair, all 24 bound TempoRAC source files, all 19 bound TempoRAC test files, and the additional package bootstrap files relevant to `python -m` import order. Author summaries and prior intermediate bytes were not treated as evidence.

Start and terminal rehashes agreed:

| Input | Bytes | SHA-256 |
|---|---:|---|
| timestamped Round 3 Markdown `_20260816_141214.md` | 166,625 | `8d02c70186c38959c74b774e211427f86435664d00106159c0da11ce5843dae9` |
| timestamped Round 3 JSON `_20260816_141214.json` | 138,394 | `187aa3450a9e40f1c75a04b135c5e97e856e18f0165e677a062317e3587be7f2` |
| fixed Markdown alias | 166,625 | `8d02c70186c38959c74b774e211427f86435664d00106159c0da11ce5843dae9` |
| fixed JSON alias | 138,394 | `187aa3450a9e40f1c75a04b135c5e97e856e18f0165e677a062317e3587be7f2` |
| archived Round 2 Markdown `_20260816_124239.md` | 99,017 | `bd2c0772eb2f8ca70d23c6217b094d58ad69e1167486dbc811b7a4cec9d6e97c` |
| archived Round 2 JSON `_20260816_124239.json` | 74,712 | `53f3409112550b79b6c11860cf1822e207a8bd600dd9708574ae977548282cd0` |
| Round 2 review Markdown | 21,614 | `9d401fcde35786bfe2925cf31a803bd515668a156f08ae5bc9ef5c27810a4101` |
| Round 2 review JSON | 18,117 | `ed877ba3d8fd15c76c9908dd694bfbb804ed0c7b05f99a97853dc55c51ea56bd` |
| canonical proposal | 79,366 | `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391` |
| experiment plan | 34,198 | `4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e` |
| experiment tracker | 22,360 | `714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b` |

The fixed aliases are byte-identical to the `_20260816_141214` pair. The Markdown Appendix A is byte-identical to the standalone JSON. Both are strict UTF-8 without BOM or CR, have exactly one terminal LF, contain no duplicate JSON keys, and the JSON object keys are recursively UTF-8 lexical. No non-finite number or negative zero is present. All 50 amendment-bound paths rehashed to their exact binding values: 24/24 source, 19/19 tests, and 7/7 planning/review inputs. No bound-input drift was observed.

## Blocking finding A005-R3-B1 — executable-origin closure is not constructible or complete

The precommit/trace split removes the Round 2 self-hash cycle in principle, but the new closed schemas cannot validate all executable bytes of the exact invocation.

1. **The required installed-member ledger is undefined and unbound.** The amendment requires a complete installed wheel index (`JSON:660-669`) but the closed environment has only `wheel_index_sha256` (`JSON:673-693`). A wheel-index row contains only archive-level `bytes`, `distribution`, `filename`, `sha256`, and `version` (`JSON:695-709`). Wheel-origin normalization nevertheless requires exact member bytes to match both the wheel index and an “installed-member ledger” (`JSON:829`). There is no schema, row shape, ordering rule, completeness rule, top-level digest, or contract preimage for that ledger. A wheel archive digest alone does not prove which installed `.py`, extension, or generated install member supplied an executed byte string.

2. **Pure-Python standard-library source has no admissible origin class.** Every import source and compile source is captured (`JSON:807-811`), while membership permits repository origins or environment wheel/native/process origins (`JSON:823-829`). CPython standard-library `.py` files are neither repository allowlist rows nor installed wheel members under the specified identity. The environment binds an interpreter ELF/configuration, not a complete standard-library source/member index. Exact standard-library bytes therefore cannot pass the stated membership rule.

3. **The exact command imports code before the stated bootstrap can start tracing.** The frozen command is `python -I -m pams.temporac.protocol_entry` (`JSON:834-842`), while tracing begins “in the allowlisted bootstrap” (`JSON:787`). Python must import `pams`, `pams.temporac`, and `pams.temporac.protocol_entry` before module code can initialize that bootstrap. Current `src/pams/__init__.py:3` eagerly imports `pams.types`; `src/pams/types.py:10-15` then imports `collections.abc`, `dataclasses`, typing, NumPy, and NumPy typing. Those project, standard-library, and wheel/native origins occur before a Python bootstrap inside `protocol_entry` can observe them. No pre-import launcher, interpreter audit-hook installation, or other committed monitor entry is specified by the exact command.

4. **The committed repository allowlist is not tied to the installed project members actually used under `-I`.** Repository origins must match source-allowlist rows, but wheel origins only match environment indexes (`JSON:823-829`). The command forbids editable fallback (`JSON:835`) and isolated mode does not make the source tree authoritative. Nothing requires an installed `pams` member to be byte-equal to its corresponding repository allowlist row. Consequently `code_index_sha256` may commit one set of repository bytes while the exact command executes different installed project bytes.

5. **The event policy forbids generated Python source that the frozen runtime stack necessarily uses.** Every `compile`/`exec` is covered, and memory-only/generated source is forbidden (`JSON:807-811,825`). CPython 3.12's `dataclasses` implementation constructs functions from generated source and calls `exec`; the current code imports and uses `dataclass`. The policy supplies neither a deterministic generated-source ledger nor an allowed, byte-bound runtime-generation origin. Thus a complete monitor must fail an otherwise normal execution, while a monitor that ignores those events violates the exact event semantics.

Impact: no implementation can both follow the frozen invocation/event semantics and produce a complete, membership-valid trace. More importantly, the contract cannot prove that installed project/stdlib bytes executed were the bytes committed by `code_index_sha256`. This keeps Round 2 B3 open and prevents B2's numeric replay environment from becoming an end-to-end authority-bearing replay closure.

Minimal repair:

- define a closed installed-wheel-member ledger and a closed CPython-standard-library source/member index, with exact row keys, normalization, ordering, completeness, byte counts, hashes, and environment top-level digest bindings;
- bind installed project members byte-for-byte to the corresponding source-allowlist rows, or freeze an invocation that executes the committed repository bytes directly;
- start a committed monitor before any package/module import, including the monitor/launcher itself in the relevant code and environment indexes; and
- reconcile actual generated `compile`/`exec` behavior with the policy by committing deterministic generated-source preimages and origins, or by narrowly specifying a complete byte-bound trusted-runtime-generation mechanism. Recheck acyclicity after choosing that mechanism.

## Blocking finding A005-R3-B2 — the typed inventory/DAG is internally inconsistent

The roster size and most edge rules are now numeric, but two exact-schema contradictions prevent a unique conforming DAG.

1. **NATP cannot satisfy its inventory row rule.** `temporac.natural-prediction-completion-receipt.v4` has exactly 13 keys and contains `k6_receipt_sha256`, not `upstream_receipt_sha256` (`JSON:2065-2082`). The inventory rule nevertheless says that for “X0I/K3/K4/NATP receipts” the payload's upstream array must equal the inventory-row array (`JSON:2148-2149`). Because extra keys are forbidden, no NATP receipt can have that required payload array and also satisfy its exact 13-key schema. The later edge rule correctly special-cases `natp.k6` (`JSON:2238,2243,2258-2259`), but it does not repair the payload-schema contradiction.

2. **K1's two roster upstreams have no uniquely legal DAG role.** The immutable roster declares K1 upstream owners `G0-ACQUIRE` and `G1-TEACHER-AGG` (`JSON:1649-1656`). The explicit K1 edge rule creates only `k1.feature`, `k1.natural-input`, and `k1.target` edges (`JSON:2242,2253-2255`). `stage.upstream` is expressly assigned to pre-G5a-stage, X0I, K3, K4, and NATP receipts (`JSON:2238`), not K1; no `k1.upstream` token exists (`JSON:2246-2267`). The catch-all sentence about “remaining stage upstreams” (`JSON:2244`) cannot select a role without contradicting or extending the enumerated rule. A verifier therefore cannot decide whether the two direct K1 roster dependencies are required, forbidden, or encoded under an unspecified role.

Impact: the exact 54-row inventory cannot be mapped to one satisfiable, uniquely typed node/edge set. NATP/K6/prediction lineage and K1/G0/G1 provenance can be omitted or rejected depending on verifier interpretation. This keeps Round 2 B4 and B6 open despite the corrected counts and roots.

Minimal repair:

- replace the NATP inventory equality clause with an exact schema-aware rule such as `payload.k6_receipt_sha256 == row.upstream_receipt_sha256[0]`, while requiring array length one and owner token `K6-FREEZE`; alternatively add the array to NATP and update its exact key count/schema everywhere; and
- add a `k1.upstream` role with fixed ordinals 0=`G0-ACQUIRE`, 1=`G1-TEACHER-AGG`, or explicitly include K1 in the `stage.upstream` generation rule. State exactly how both digests are validated against the K1 outcome/index evidence and retain the direct-edge requirement.

## Closure matrix

| Prior item | Round 3 result | Adversarial conclusion |
|---|---|---|
| A005-R2-B1 | CLOSED | Normalized SmoothL1 is now an exact binary64 piecewise DAG with the equality branch, Neumaier/reduction order, one-run `<i4[[0,N_v]]` certificate bounds, separate ten traversal bounds, and exact three-run seed-ordered arrays. |
| A005-R2-B2 | SPECIFICATION CLOSED; END-TO-END BLOCKED | FE_TONEAREST, MXCSR `0x00001f80`, FTZ/DAZ off, Torch denormal state/probes, MKLDNN/NNPACK controls, and fresh-process replay are exact. They cannot become provenance authority until R3-B1 is repaired. |
| A005-R2-B3 | OPEN — R3-B1 | Precommit avoids the former trace self-cycle, but installed-member, stdlib, bootstrap, repo/install equality, and generated-source coverage remain incomplete or impossible. |
| A005-R2-B4 | OPEN — R3-B2 | The 54-owner/count/schema roster includes X0I×3, K3, K4, 27 runs, 17 generic stages, and NATP×3, but NATP payload equality is unsatisfiable and K1 direct upstream roles are undefined. |
| A005-R2-B5 | CLOSED | `run.upstream` exists with payload-array ordinals; G1 and its evidence index use exact length-three teacher-run arrays in seed order. |
| A005-R2-B6 | OPEN — R3-B2 | Pilot scope, six 1,608-row seed indexes, two 4,824-row condition indexes, 9,648 predictions, three NATP rows, K6 lineage, and G5a roots are numerically closed, but the NATP receipt/inventory contradiction prevents executable lineage verification. |
| A005-R2-B7 | CLOSED | `k1_outcome_row_sha256` now hashes the exact standalone canonical 13-key row plus LF; Cdev uses those committed row identities. |
| A005-R2-B8 | CLOSED | All 50 current bound files match the Round 3 values, including the two formerly drifted tests. |
| F4 | CLOSED | The builder snapshots a direct mapping before validation and composition; direct `MappingProxyType` input is accepted while state-changing mappings fail. |

## Other adversarial checks that passed

- **Full teacher selection chain:** the contract retains and replays all 120 teacher checkpoints, all 120 tune-output NPZs and evaluations, and the exact shared 56-view input. The selected scalar is rederived through geometry/landmarks, F5, normalized SmoothL1, binary64 Neumaier reduction, penalty, and deterministic argmin; no free score scalar or winner-only authority path remains.
- **Checkpoint exactness:** the 16-member checkpoint ledger matches the current `TempoRACTeacher.state_dict` names, shapes, and dtypes. Strict load, bytewise member validation, retention, and global replay cover winners and losers.
- **NPZ schemas and caps:** independently reconstructed raw/archive sizes are 1,008,348/1,012,518 bytes for a checkpoint, 22,592,544/22,594,258 for the shared tune input, and 15,510,720/15,511,204 for a tune output. They remain below the respective 2/32/16 MiB caps. All declared 9/10/11/12/13/15-key receipt schemas match their listed closed keys.
- **Effective contract:** the proposal, exact `_141214` amendment pair, later accepted review pair, plan, and tracker form an acyclic non-self-hashing constituent index. Downstream artifacts use the resulting single digest. No current value is fabricated.
- **K1/Cdev/targets:** K1 freezes a complete immutable 402-row outcome before G5a. Cdev is exactly the ordered val/CERTIFIED projection, not all 402 and not a late subset, with retained floors of 108 identities and 8 components. K7 bytewise rederives target identities and bytes.
- **Prediction identity:** the population manifest is canonical; each prediction component is derived from its manifest row, caller relabeling is forbidden, and the exact product is 1,608 rows per seed and condition, 4,824 per condition, 9,648 total.
- **Capability order and firewall:** non-vault checks precede atomic consumption; only the consumed process may perform G5b vault validation/join; K7 follows G5b. Pre-consume vault open/stat/hash/enumeration/handle access and retry are forbidden. No data, vault, evaluator, server, or training action occurred in this review.
- **No method/job/claim expansion:** the amendment adds provenance requirements only. It creates no new experiment job, budget, model component, metric, paper claim, or result.

The in-memory arithmetic/schema probes above were review diagnostics, not an execution of the proposed future test contract. No test suite, experiment, evaluator, data access, server action, training, or launch was run.

## Required disposition and authority ceiling

Revise the exact fixed/timestamped Amendment 005 pair to close A005-R3-B1 and A005-R3-B2, then obtain another normative review of those new bytes. Do not treat this review as acceptance of an implementation amendment.

This review is non-authoritative and authorizes nothing. `P0=0`, `P1=0`, `P2=0`, `P2-METRIC=0`, `P3=0`, `S0=0`, `gate=0`, `capability=0`, `evaluator=0`, `launch=0`, `server=0`, `data=0`, `GPU=0`, `training=0`, `test=0`, `results=0`, `Git=0`, and `paper/claim=0`. No amendment, proposal, plan, tracker, MANIFEST, code, test, server, data, experiment, training artifact, evaluator, or Git state was modified by this review.
