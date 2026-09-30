# TempoRAC Contract Amendment 004 Review, Round 1: Operator Manifest Serialization

> **VERDICT: `REVISE`**
>
> **SAME-FAMILY PROVISIONAL — NON-AUTHORITATIVE**
>
> **P2=0, S0=0, LAUNCH=0, SERVER=0, DATA=0, TRAINING=0, GATE=0, GIT=0, PAPER/CLAIM=0**

**Date:** 2026-08-16  
**Reviewer model:** `gpt-5.6-sol`  
**Reviewer family:** `openai`  
**Executor family:** `openai`  
**Review independence:** `same-family`  
**Acceptance status:** `provisional`  
**Continuation agent:** `/root/temporac_operator_manifest_amendment_review`

## 1. Decision

The core 10,368-row manifest serialization is substantially correct, and the
historical candidate is independently reproducible. The final Amendment 004
bytes nevertheless cannot be accepted because the amendment does not close the
array shapes/axes, the machine-readable schema and replay-witness envelopes, the
candidate-receipt bindings and values, or a retrievable immutable runtime. It
also records four experiment blocks as a scientific non-change while the bound
proposal permits exactly three paper-visible evidence blocks and explicitly
forbids a fourth.

These are specification blockers, not candidate-output mismatches. A currently
reproducible candidate does not prove that all future conforming implementations
must emit the same artifacts. Amendment 004 remains
`PROPOSED_PENDING_FRESH_REVIEW`; regeneration and implementation review are not
authorized by this Round 1 disposition.

## 2. Final bound Amendment 004 bytes

The amendment was reread in full from disk after its final `git=false` patch.
Only these final bytes were adjudicated:

| Input | Bytes | SHA-256 |
|---|---:|---|
| `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_004_OPERATOR_MANIFEST_SCHEMA.md` | 11,540 | `4d2d0c88c5bec31c725b856a6ba20bd21dc68410d493fd8fb0d2130d5822f58c` |
| `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_004_OPERATOR_MANIFEST_SCHEMA.json` | 9,500 | `bc3391d0d282c896e305bd20bba96640da7f633f31e6ecc47ee866d7a6464637` |

The companion JSON parses successfully and contains the final closed authority
set with `git=false`.

## 3. Independent complete candidate replay

The 15,039,512-byte historical manifest was read completely with duplicate-key
detection, reparsed, reserialized under the stated manifest rules, and replayed
without importing the production TempoRAC implementation. The independent
replay used CPython 3.12.13 and NumPy 1.26.4 on Windows; that is diagnostic
cross-runtime evidence only, not satisfaction of the proposed v2 runtime lock.

| Check | Independent result |
|---|---|
| top level | one JSON array; no BOM observed; exactly one terminal LF |
| duplicate object keys | none |
| canonical byte replay | byte-identical |
| manifest bytes / SHA-256 | `15,039,512` / `1f39378d6f7a970b1f183558ed145bd97107f2aeda41902f4736e6778930e1bb` |
| rows | `10,368` |
| row schema | exactly 23 keys on every row |
| row type/domain errors | `0` |
| unique 10-byte keys | `10,368` |
| Cartesian/key order | exact and strictly increasing |
| independently recomputed row mismatches | `0 / 10,368` |
| semantic bytes / root | `746,714` / `0fe8b2619a3f33cc4a83485053888c04c567b015eea65f81bf6cd9ec715ee4f2` |
| inactive lookup bytes / root | `1,024` / `afa6eff0d744acfdf9cf78adc43aa8daf01633f3345d7567d642a7782885040d` |
| key preimage bytes / root | `186,624` / `18dfdafe237b77565de36bab6d4642efb314ec1cbbd76e852938ed5c4d424a55` |
| edge-zero digest bytes / root | `331,776` / `ba797f6ae59cbbb40f19a61199afaaa1ac5b2c001a3c9fc34f1dba8f79cf78e4` |
| actual inactive `b=255` | `8,945` edges in `5,665` rows; first key `00040001000000000001`, edge `42`, bytes `cccccc3e` |
| flat amplitude-one cases | `2,304` rows and `6,912` plateaus |

The nine historical aggregate roots also replay exactly:

| Aggregate | Bytes | SHA-256 |
|---|---:|---|
| association | 93,312 | `856710a129ad0e85fa46e2adb287b964163e0cd49d12f302ccfd8b63948712e6` |
| expected components | 248,832 | `6c6e6240e577c16a4a58150102f4dfe3af2d28a914e8ceefce6e25a673411481` |
| decoder masks | 2,345,760 | `6cf03b8331b9c7f9021ae778dac8931e8c8cb3c5375cc0e2018de02cc77eb37a` |
| edge masks | 2,345,760 | `6cf03b8331b9c7f9021ae778dac8931e8c8cb3c5375cc0e2018de02cc77eb37a` |
| NOLA responses | 9,383,040 | `72a958eb3224e4585b7ef7cec9f1729c5c6c68ca8412a6f196cc28eb4fb9c5c2` |
| generated responses | 9,383,040 | `72a958eb3224e4585b7ef7cec9f1729c5c6c68ca8412a6f196cc28eb4fb9c5c2` |
| target masks | 2,345,760 | `6cf03b8331b9c7f9021ae778dac8931e8c8cb3c5375cc0e2018de02cc77eb37a` |
| truth masks | 2,345,760 | `45266d339f54a2a59f977a770d3f1bb71c9319b798b08d8b1b04de7aece8de21` |
| truth plateaus | 248,832 | `6c6e6240e577c16a4a58150102f4dfe3af2d28a914e8ceefce6e25a673411481` |

This confirms that Amendment 004 faithfully describes the observed main
manifest bytes. It does not resolve the following normative ambiguities.

## 4. Exact blockers

### R1-B1 — the asserted non-change count conflicts with the bound proposal

Amendment 004 MD Section 2 and JSON `non_changes.experiment_blocks` freeze
`4`. The bound proposal Section 15 instead states “Exactly three paper-visible
blocks remain” and says audit evidence does not become a fourth block. The
terms are not equated or distinguished anywhere in Amendment 004. A
serialization-only amendment cannot introduce a second unscoped experiment
count and call it unchanged.

**Required correction:** replace `four experiment blocks` / `experiment_blocks:4`
with the proposal's exact `three paper-visible evidence blocks`, or explicitly
define a separate non-paper-visible four-block inventory and bind the governing
normative source without weakening the three-block paper/claim ceiling.

### R1-B2 — raw array hashes do not freeze shape or axis semantics

Section 5 freezes 14 dtypes and C-order bytes but delegates shape to
“proposal-derived shape.” The proposal does not provide a closed serialization
shape table for all 14 arrays. Raw bytes do not bind shape: for example `[3]`
and `[3,1]` location arrays hash identically, and the association table does not
state whether axes are `[truth,component]` or `[component,truth]`. The current
all-identity associations hide this attack because transposition preserves the
bytes.

**Required correction:** enumerate exact shape formulas and axis ownership for
every array, including `association[t,c]`, vector versus column-vector rules,
`[N,2]` interval tables, `[E]` masks/responses, reset-dependent run count, and
the exact window-count formula. Require dtype, shape, finiteness, and C
contiguity validation before hashing. Raw unframed bytes may remain the hash
preimage once those independent shape checks are normative.

### R1-B3 — the schema and replay-witness members are not closed artifacts

`operator_manifest_schema.json` is only required to “restate Sections 3–5”;
its exact keys, value types, arrays, and wording are not defined. The replay
witness “may contain” compact roots and must include “at least” nine aggregates,
but has no schema name, closed keys, exact aggregate field names, value types,
byte-count fields, or closed five-row-witness schema. Multiple byte-distinct
artifacts therefore satisfy the amendment, and their differing hashes flow into
the candidate receipt.

**Required correction:** define closed canonical JSON schemas for both members,
including exact keys and types, exact aggregate-name-to-array mapping, exact
ordering of the five row witnesses, and duplicate-key/BOM rejection. Extra
roots or prose must be forbidden in the canonical member or moved to a detached
non-normative diagnostic file. The candidate manifest itself should retain its
top-level array and no self hash.

### R1-B4 — the candidate receipt is not uniquely serializable and misses executed dependencies

The receipt closes its top level and authority object, but not `bindings`,
`members`, or `blockers`; it does not freeze the exact values/types of
`p2_status` and `s0_status`; and “must include” permits blocker additions and
ordering changes. The phrase “all directly executed TempoRAC source/test
bytes” is not an executable dependency rule. The reviewed import graph already
shows omitted dependencies: `fixtures.py` imports `hashio.py` and `types.py`,
the gate test imports `gates.py`, package import executes
`src/pams/temporac/__init__.py`, and a genuinely executed `cue.py` would also
execute `preprocess.py` and `quadrature.py`. None of those paths appears in
Amendment 004 JSON `input_bindings`.

The no-self-hash design is correct and avoids receipt/manifest circularity, but
an external receipt hash cannot compensate for an underbound dependency set.

**Required correction:** freeze exact closed binding keys or an exact closed
ordered dependency-manifest schema, enumerate every executed project file and
both test entry points, bind bytes and path roles, close `members`, freeze the
blocker array and its order, and set exact string values for
`candidate_status`, `p2_status`, and `s0_status`. Keep the receipt free of its
own digest and require the fresh implementation review to bind the receipt
digest externally.

### R1-B5 — the Docker lock is immutable-looking but not independently reconstructible

The v2 lock contains a bare content digest but no registry/repository reference
or platform-specific child manifest/config digest. A fresh host cannot uniquely
pull a bare digest. The stated fields also do not say whether NumPy 2.1.0 is
already inside that image; if it is installed after container start, neither a
wheel SHA-256 nor an installation lock is frozen. Recording the observed NumPy
`__init__.py` and `RECORD` hashes detects some drift after the fact but does not
define how an independent reviewer constructs the same environment. Mutable
mounts, user-site imports, `PYTHONPATH`, and network installation are not
closed. The historical tag-only lock is correctly nonconforming.

**Required correction:** bind a pullable OCI reference such as an exact
registry/repository plus digest, freeze the linux/amd64 child manifest and
config digest, and either use a purpose-built image that already contains the
exact Python/NumPy environment or bind the exact wheel/archive SHA-256 and
offline installation procedure. Require read-only/no-network replay with
user-site disabled and a closed mount/environment policy. The reviewer must
verify the invoked executable, imported NumPy, and distribution record against
the candidate lock.

## 5. Adversarial checklist disposition

| Attack surface | Disposition |
|---|---|
| top-level manifest JSON | substantially closed; historical bytes replay exactly |
| JSON lexical ambiguity | amplitude spellings and whitespace are closed; canonical members still need explicit duplicate-key and BOM rejection |
| 10,368 row order and 23 keys/types | pass on historical candidate |
| endianness / C order / raw framing | dtype and byte framing are closed |
| shapes / axes | **blocked by R1-B2** |
| aggregate preimages | raw concatenation rule is clear; witness envelope/names remain **blocked by R1-B3** |
| runtime tag drift | tag-only history is rejected, but v2 retrieval/install is **blocked by R1-B5** |
| receipt self-certification | no-self-hash/external-review pattern is sound; closed binding authority is **blocked by R1-B4** |
| manifest-hash circularity | no circularity found |
| executed dependency binding | **blocked by R1-B4** |
| exclusive creation | new-directory plus exclusive-member creation is fail-closed; revised schema should additionally require non-receipt members first and receipt last so a partial tree cannot be mistaken for complete |

## 6. Complete input snapshot

| Input | Bytes | SHA-256 |
|---|---:|---|
| `refine-logs/temporac/FINAL_PROPOSAL.md` | 79,366 | `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391` |
| `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY.md` | 10,622 | `899d6f071c89f898bf3825adfb30a326904f9bc7fe44d44351da8a5d0d9ccb14` |
| `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY.json` | 8,963 | `d524556f292ec17a155cdcd962cf8a88bea26f32b810ffb7bd8e45744f1c54fc` |
| `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY_REVIEW_ROUND1.md` | 7,160 | `5e60841696333234213fa36675f11311d294e76adfd1b68d8ab74198a409f253` |
| `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY_REVIEW_ROUND1.json` | 5,871 | `41468e0c7a163040cefb035936a35025ad3910da96488e1582c02b83a15c5f65` |
| `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY_REVIEW_ROUND2.md` | 11,229 | `a86da733f3ae048b867ec8b2366a9f0262ca2b390ef08687ca7d710654e8301f` |
| `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY_REVIEW_ROUND2.json` | 9,363 | `3730d96eff1d3815499602eaf183774af032c514a588d4e92fe8a1cccf053e64` |
| `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_004_OPERATOR_MANIFEST_SCHEMA.md` | 11,540 | `4d2d0c88c5bec31c725b856a6ba20bd21dc68410d493fd8fb0d2130d5822f58c` |
| `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_004_OPERATOR_MANIFEST_SCHEMA.json` | 9,500 | `bc3391d0d282c896e305bd20bba96640da7f633f31e6ecc47ee866d7a6464637` |
| `src/pams/temporac/contract.py` | 9,079 | `5ab8fb62dc558ba78e95fc07e52c1f1a1afbff5b1c5003834079eea6143d745f` |
| `src/pams/temporac/fixtures.py` | 36,365 | `979b8cee4fed117f5471cf399794f615c60eae52dbba2c946109ff836206b464` |
| `src/pams/temporac/cue.py` | 19,289 | `702784bef27db94c1dc027d036a64d7461ae618613baec5e68061254f98569c7` |
| `src/pams/temporac/nola.py` | 14,395 | `ad6a13203f09c13a37a854333d4458ca8ce5a03bb186c0946f014a4e8cb9a637` |
| `src/pams/temporac/decode.py` | 8,843 | `766cd6ca4b9e133041bd602f50fde840ec7caadd376bd6851799583d44f31f5e` |
| `scripts/experiments/generate_temporac_operator_manifest_v4.py` | 17,717 | `ffc5b83025375b2c68376f093becb63905504e15be45237e965991c6f762f8c7` |
| `tests/temporac/test_operator_manifest.py` | 12,749 | `a055247946580679d54204850bfe588a102a37d342da31e06eddba85c45f9dfe` |
| `tests/temporac/test_gates_fixtures.py` | 5,973 | `941856b6111cff8697f105bc30dc679069f73206bab3f9c2b930236675e13173` |
| `data/temporac_p00_v4/operator_candidate_20260816/candidate_schema.json` | 1,301 | `9f4b2ced821e667f950c1f56bbf9adbdc14b2c387c1932953048caf3f747a8cd` |
| `data/temporac_p00_v4/operator_candidate_20260816/environment_lock.json` | 458 | `709db0943752965524b507dd6340554a8d7d218be125317af0fc8b7748a5a947` |
| `data/temporac_p00_v4/operator_candidate_20260816/replay_witness.json` | 3,908 | `36f60f41ac111a7b05e5f4846464cc774e09bb9faa44feed10be5a4f35418653` |
| `data/temporac_p00_v4/operator_candidate_20260816/temporac.operator-manifest.candidate-v1.json` | 15,039,512 | `1f39378d6f7a970b1f183558ed145bd97107f2aeda41902f4736e6778930e1bb` |
| `data/temporac_p00_v4/operator_candidate_20260816/candidate_receipt.json` | 1,729 | `2e457e01166c6151e4edc688e8fd6f981d1278231fcd8c2bc000179f7a825102` |

No executor summary or operator implementation review was used as review input.

## 7. Authority ceiling and continuation

**REVISE — same-family provisional, non-authoritative.**  
**Blocking issue count:** `5`.  
**Continuation agent:** `/root/temporac_operator_manifest_amendment_review`.

This review authorizes no amendment edit, regeneration, implementation change,
test change, candidate promotion, P2, P3, S0, gate passage, launch, server/GPU
use, natural/evaluator/sealed/held-out data access, training, result, Git action,
paper edit, claim, release, or submission. After the five blockers are repaired,
the revised Amendment 004 MD/JSON require a same-agent Round 2 normative review.
Only a later `ACCEPT` may authorize regeneration into a new absent directory,
and that regenerated candidate would still require a fresh independent
pinned-runtime implementation review before it could be used merely as input to
a separately specified P2 fixture gate.
