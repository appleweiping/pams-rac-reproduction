# TempoRAC Contract Amendment 002 Review, Round 1: Operator Row-Key and Response Bytes

> **VERDICT: `REVISE`**
>
> **SAME-FAMILY PROVISIONAL — NON-AUTHORITATIVE**
>
> **NO IMPLEMENTATION, GIT, S0, GATE, DATA, SERVER, TRAINING, EVALUATOR, RESULT, CLAIM, OR PAPER AUTHORITY**

**Date:** 2026-08-16  
**Reviewer model:** `gpt-5.6-sol`  
**Reviewer family:** `openai`  
**Executor family:** `openai`  
**Review independence:** `same-family`  
**Acceptance status:** `provisional`

## 1. Decision

The Section 9 contradiction is real. The proposal freezes a semantic Cartesian row but does not freeze the row-key bytes, the length-prefix width or endianness, or the digest-byte selection. The general `H(tag,*fields)` helper is not uniquely implied and would length-prefix the edge field as well. Canonical JSON, fixed binary fields, and several digest-byte choices therefore produce different response bytes. Production is correct to remain `BLOCKED_UNDEFINED_ROW_KEY_PREIMAGE`.

The proposed five-field 10-byte key, one `uint64_be(10)` prefix, raw `uint32_be(e)`, and `D(e)[0]` form a coherent non-expansive serialization correction. Round 1 nevertheless cannot accept the amendment because its float32, plateau, witness, inventory, and propagation text contains five exact blockers.

## 2. Historical bound input

This file records the original Round 1 disposition after the amendment was revised in place. The two amendment hashes below identify the exact superseded bytes adjudicated in Round 1; those historical bytes are no longer asserted to exist at the current paths.

| Input at Round 1 | SHA-256 |
|---|---|
| `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY.md` | `3b7d3719ebf51c92f8a25bb94b11c7f86d9d112476016c34f5a1f4602177aa08` |
| `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY.json` | `d3096593d42b65aa1b2c58f7a30693d3a2880f6da22ffc69eb6dfacd70748655` |
| `refine-logs/FINAL_PROPOSAL.md` | `562325dedab0637421f62dec5ad149b3c884f67b47c4b525c8fb54a57bb1233c` |
| `refine-logs/temporac/FINAL_PROPOSAL.md` | `562325dedab0637421f62dec5ad149b3c884f67b47c4b525c8fb54a57bb1233c` |
| `refine-logs/EXPERIMENT_PLAN.md` | `3fd1986871b32674d74b395fbfc9720712d9fba0c8583c24c1d5bddb87086eb6` |
| `refine-logs/EXPERIMENT_TRACKER.md` | `c7e2f32c3c81b66884abb13760a671ecc7cdd314fd6c994a4ed4c46cb754a63b` |
| `src/pams/temporac/contract.py` | `b71255a35cce1701d2741064064b53ba819b742873a348f1852f147149e36467` |
| `src/pams/temporac/hashio.py` | `3fcad105f5ddb1e5c2d2152b825727fec5ab73bd1c5fc51953143bf377ad44d6` |
| `src/pams/temporac/fixtures.py` | `a8d10df7c8e3cbcb5684198192d3a553921da0f6aa92f03e54448ee182671406` |
| `tests/temporac/test_gates_fixtures.py` | `4c33a46d6dfe4119c3a82e1d8df5602d12bc1fd469664805be1a67be75ad6443` |

## 3. Exact blockers

### 3.1 Uncapped `b=255` violates the frozen negative margin

Direct binary32 rounding of the proposed uncapped endpoint produces

```text
RN-even-f32((255 + 3*255)/2550)
= binary32 bits 0x3ecccccd
= little-endian bytes cdcccc3e
= 0.4000000059604645
```

Its exact widened margin to the decoder threshold is only `0.09999999403953552`, which is strictly below the required `0.10`. A full scan finds `b=255` at 8,945 eligible inactive edges in 5,665 rows and in all four layouts. The first actual inactive occurrence is row `(4,1,aidx=0,o=0,lidx=1)`, edge `42`, digest `ff78168a29882190e18b1fd5957cf1eb549c721c227344c8fbfc20ec8d9f4fb3`.

The smallest local correction is to cap only that endpoint at binary32 bits `0x3ecccccc`, little-endian bytes `cccccc3e`. The resulting 1,024-byte increasing-`b` lookup has SHA-256 `afa6eff0d744acfdf9cf78adc43aa8daf01633f3345d7567d642a7782885040d`; only `b=255` differs from the uncapped table, and the exact widened margin becomes `0.10000002384185791`.

### 3.2 Amplitude 1.00 does not have a unique boosted maximum

For amplitude `1.00`, ordinary and boosted values both serialize as binary32 `0x3f800000`, little-endian `0000803f`. Every width-2 and width-4 plateau is therefore flat, not uniquely maximized at its lower-middle edge. This affects 2,304 rows and 6,912 plateaus. The amendment must state that flatness and must not require a unique maximum for those cases.

### 3.3 The `e=32` witness confuses a hypothetical inactive value with the generated response

For row `(4,1,aidx=0,o=0,lidx=0)`, edge `32` is the lower-middle edge of the first width-1 plateau. Its digest byte is `222`, so the hypothetical inactive function returns `1fecb83e`, but its actual generated response is boosted 0.65, binary32 bits `0x3f266666`, little-endian bytes `6666263f`. The witness table must label these as distinct quantities.

### 3.4 Binary inventory preimages are ambiguous

The inventory notation does not state whether a digest contribution is raw bytes or lowercase ASCII hexadecimal, and it leaves room for an outer wrapper or delimiter. It must define direct concatenation explicitly, including the one `uint64_be(10)||K_r` contribution per key and one raw 32-byte `D_r(0)` contribution per row, plus exact byte counts.

### 3.5 Propagation names only one proposal copy

Both `refine-logs/FINAL_PROPOSAL.md` and `refine-logs/temporac/FINAL_PROPOSAL.md` are currently byte-identical and normatively consumed. A valid amendment must require both copies to be patched byte-identically, assigned one shared new digest, and propagated to every proposal/contract-hash consumer. It must not imply that acceptance itself performs or authorizes those edits.

## 4. Minimality qualification

The fixed 10-byte encoding is not the literal shortest possible injective encoding of this finite population. It is acceptable only as the smallest coherent correction in normative scope: five already ordered fields receive one uniform platform-independent integer representation, while the population, Cartesian order, geometry, and scientific method remain unchanged. The amendment must not claim mathematical byte-length minimality.

## 5. Required canonical revision

Before another review, the amendment must:

1. cap the inactive endpoint at exact binary32 `0x3ecccccc`, freeze the operation order, and add the 1,024-byte lookup witness;
2. identify the first actual inactive `b=255` witness and distinguish function-only bytes from generated-response bytes at `e=32`;
3. state the exact lower-middle rule and the amplitude-1.00 width-2/4 flat-plateau exception;
4. define both binary inventory concatenations as raw unwrapped bytes with exact lengths;
5. require byte-identical changes to both proposal copies and complete downstream rebinding under separate authority; and
6. preserve F23, all gates, jobs, resources, data protocol, evidence ceiling, claims, and the zero-authority boundary.

## 6. Authority ceiling

This `REVISE` disposition grants no authority. It does not authorize amendment edits, implementation, tests, manifests, Git, P2/P3, S0, any gate, server or natural/evaluator/sealed/held-out data access, training, result use, claims, or paper promotion. The existing blocker remains in force.

## 7. Round 1 disposition

**REVISE — same-family provisional, zero authority.**  
**Blocking issue count:** `5`.
