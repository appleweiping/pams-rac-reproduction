# TempoRAC Contract Amendment 002 Review, Round 2: Operator Row-Key and Response Bytes

> **VERDICT: `ACCEPT`**
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

`TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY` is accepted as the smallest coherent non-expansive correction in normative scope. The original Section 9 text fixes the 10,368 semantic rows but leaves the row-key bytes, length-prefix encoding, and digest-byte selection undefined; its generated-response bytes are therefore not executable without an extra choice. The amendment removes exactly that ambiguity with a fixed 10-byte categorical key, one `uint64_be(10)` prefix, a raw `uint32_be(e)`, and `D(e)[0]`, and it freezes the binary32 operation and serialization needed to satisfy the existing strict margins.

All five Round 1 blockers are repaired. There are no remaining scientific or contract blockers in the final bound amendment. This accepts only the amendment bytes identified below; it does not patch either proposal, remove the production blocker, pass a gate, or grant any authority.

## 2. Final bound input snapshot

Every required input was reread and rehashed from disk immediately before persistence. The two proposal copies are byte-identical. The amendment JSON parses without duplicate keys, and all nine of its MD/canonical-document/implementation bindings match the exact current bytes.

| Input | SHA-256 |
|---|---|
| `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY.md` | `899d6f071c89f898bf3825adfb30a326904f9bc7fe44d44351da8a5d0d9ccb14` |
| `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY.json` | `d524556f292ec17a155cdcd962cf8a88bea26f32b810ffb7bd8e45744f1c54fc` |
| `refine-logs/FINAL_PROPOSAL.md` | `562325dedab0637421f62dec5ad149b3c884f67b47c4b525c8fb54a57bb1233c` |
| `refine-logs/temporac/FINAL_PROPOSAL.md` | `562325dedab0637421f62dec5ad149b3c884f67b47c4b525c8fb54a57bb1233c` |
| `refine-logs/EXPERIMENT_PLAN.md` | `3fd1986871b32674d74b395fbfc9720712d9fba0c8583c24c1d5bddb87086eb6` |
| `refine-logs/EXPERIMENT_TRACKER.md` | `c7e2f32c3c81b66884abb13760a671ecc7cdd314fd6c994a4ed4c46cb754a63b` |
| `src/pams/temporac/contract.py` | `b71255a35cce1701d2741064064b53ba819b742873a348f1852f147149e36467` |
| `src/pams/temporac/hashio.py` | `2d81e415b45e3d1b612f998b6a8386a967dfdfcfaac31bc914b2c95cbaeeab3a` |
| `src/pams/temporac/fixtures.py` | `a8d10df7c8e3cbcb5684198192d3a553921da0f6aa92f03e54448ee182671406` |
| `tests/temporac/test_gates_fixtures.py` | `4c33a46d6dfe4119c3a82e1d8df5602d12bc1fd469664805be1a67be75ad6443` |

Superseded intermediate amendment hashes are deliberately not review bindings. The final review binds only `899d6f...ccb14` and `d52455...c54fc` above.

## 3. Original Section 9 is undefined

The two current proposal copies specify

```text
gap × width × amplitude × offset × layout
= 9 × 3 × 3 × 32 × 4
= 10,368 rows
```

and refer to a “length-prefixed row key” and a “corresponding byte” of a SHA-256 digest. They do not define the row-key field bytes, the prefix width or endianness, or the digest-byte index. Applying `H(tag,*fields)` is not entailed and would also prefix `uint32_be(e)`, unlike the written preimage. Canonical JSON and several fixed-binary encodings are equally compatible with the prose and produce different digests. Thus `semantic_population_defined=true` and `row_key_bytes_defined=false`, `length_prefix_defined=false`, and `digest_byte_index_defined=false` are the correct findings. The existing `BLOCKED_UNDEFINED_ROW_KEY_PREIMAGE` behavior is mandatory until canonical propagation is separately completed.

## 4. Independent full-inventory replay

The reviewer regenerated the Cartesian rows directly from the frozen values and did not call the production row-key blocker. For each row it formed five big-endian unsigned 16-bit fields in the frozen categorical order.

| Check | Independent result |
|---|---|
| semantic rows | `10,368`; all unique |
| key length | exactly `10` bytes for every row |
| unique keys | `10,368` |
| order | all adjacent keys strictly increasing bytewise in frozen Cartesian order |
| first key | `00040001000000000000` |
| last key | `008100040002001f0003` |
| `I_key` length | `186,624` raw bytes |
| `SHA256(I_key)` | `18dfdafe237b77565de36bab6d4642efb314ec1cbbd76e852938ed5c4d424a55` |
| `I_edge0_digest` length | `331,776` raw bytes |
| `SHA256(I_edge0_digest)` | `ba797f6ae59cbbb40f19a61199afaaa1ac5b2c001a3c9fc34f1dba8f79cf78e4` |
| canonical semantic JSON length | `746,714` bytes |
| canonical semantic JSON SHA-256 | `0fe8b2619a3f33cc4a83485053888c04c567b015eea65f81bf6cd9ec715ee4f2` |

`I_key` was replayed as the raw concatenation of `uint64_be(10)||K_r`. `I_edge0_digest` was replayed as the raw concatenation of 10,368 32-byte digests, with no ASCII hex, delimiter, count, JSON, `H()`, or outer wrapper. This resolves the Round 1 inventory ambiguity.

## 5. Independent digest and row witnesses

The exact preimage used in every case was

```text
ASCII("temporac.operator.v4") || 0x00
|| uint64_be(10) || K || uint32_be(e)
```

All amendment witnesses reproduce exactly:

| `(g,w,aidx,o,lidx)` | `e` | `K` | digest | `b` | inactive-function `<f4` LE | actual generated `<f4` LE |
|---|---:|---|---|---:|---|---|
| `(4,1,0,0,0)` | 0 | `00040001000000000000` | `54ac35705b42e8296cf324b893e5450c123b8b48c45e912cba928a6c432f60c4` | 84 | `65984b3e` | inactive `65984b3e` |
| `(4,1,0,0,0)` | 32 | `00040001000000000000` | `dee0ad937ac053341db0c0b1dc6a179c6684e9a6108b76336070f4cb910e1170` | 222 | `1fecb83e` | boosted plateau `6666263f` |
| `(5,2,1,7,2)` | 17 | `00050002000100070002` | `9406bcc666d0a8487362b65700a5daebd6d3f4496c54c5b2da1f849ba52d3ee1` | 148 | `26598c3e` | inactive `26598c3e` |
| `(129,4,2,31,3)` | 319 | `008100040002001f0003` | `fb06baef760aaeb7ab2a4aeaad5d13dc457afc989fb75db931e1535c4213c215` | 251 | `fe63ca3e` | inactive `fe63ca3e` |
| `(4,1,0,0,1)` | 42 | `00040001000000000001` | `ff78168a29882190e18b1fd5957cf1eb549c721c227344c8fbfc20ec8d9f4fb3` | 255 | `cccccc3e` | inactive `cccccc3e` |

For the first row, the complete first preimage is `74656d706f7261632e6f70657261746f722e763400000000000000000a0004000100000000000000000000`. A full frozen-order scan confirms that the last table row is the first actual inactive `b=255` occurrence. Across the inventory, `b=255` occurs at 8,945 eligible inactive edges in 5,665 rows.

The `e=32` inactive-function value is intentionally hypothetical: the actual edge lies in plateau `[32,33)` and is its lower-middle edge. The corrected actual value is therefore boosted 0.65, little-endian `6666263f`.

## 6. Exact binary32 contract

The reviewer formed the exact integer numerator `255+3*b`, performed one binary64 division by `2550`, performed one round-to-nearest-even binary32 conversion, and then applied the exact binary32 cap `0x3ecccccc`.

| Check | Independent result |
|---|---|
| lookup order and length | `b=0..255`, `1,024` bytes |
| lookup SHA-256 | `afa6eff0d744acfdf9cf78adc43aa8daf01633f3345d7567d642a7782885040d` |
| values differing from uncapped table | only `b=255` |
| uncapped `b=255` bits / LE | `0x3ecccccd` / `cdcccc3e` |
| committed `b=255` bits / LE | `0x3ecccccc` / `cccccc3e` |
| committed maximum | `0.3999999761581421` |
| exact widened negative margin | `0.10000002384185791` |

All 256 lookup entries remain distinct. The cap therefore repairs the strict margin without changing any `b<255` response.

The exact plateau constants are:

| `aidx` | ordinary bits / `<f4` LE | boosted bits / `<f4` LE |
|---:|---|---|
| 0 | `0x3f19999a` / `9a99193f` | `0x3f266666` / `6666263f` |
| 1 | `0x3f400000` / `0000403f` | `0x3f4ccccd` / `cdcc4c3f` |
| 2 | `0x3f800000` / `0000803f` | `0x3f800000` / `0000803f` |

The lower-middle rule `s+floor((w-1)/2)` is executable for widths 1, 2, and 4. For amplitude 1.00, all width-2 and width-4 plateaus are flat: 2,304 rows and 6,912 plateaus. The final amendment explicitly forbids a unique-maximum requirement there.

## 7. Coherence, minimality, and preservation

The correction is injective, platform-independent, independent of float formatting, and preserves the frozen semantic and bytewise row order. Uniform five-field `uint16_be` encoding is not the information-theoretically shortest possible code for 10,368 values; the accepted minimality is normative-scope minimality. It introduces only the missing deterministic encoding and exact response-byte rules, without adding a row, field, layout, source, seed, mechanism, model, metric, threshold relaxation, job, resource, result, or claim.

The final MD and JSON consistently preserve F23 and every gate; the X0/NOLA/decoder/topology contracts; layouts, targets, masks, components, and geometry; data and protocol labels; models, training graph, and metrics; jobs and resource ceilings; the single-claim/evidence ceiling; and all authority boundaries. The plan retains 27 training jobs, 93 allocated A6000-hours, maximum concurrency two, and the 64 GiB disk cap. No scientific result or gate receipt is created by this review.

## 8. Mandatory canonical patch conditions

Acceptance is conditional and non-authorizing. Under separately granted edit authority, canonicalization must:

1. patch Section 9 in both `refine-logs/FINAL_PROPOSAL.md` and `refine-logs/temporac/FINAL_PROPOSAL.md` to byte-identical bytes using exactly the accepted amendment;
2. assign the two proposal copies one new shared SHA-256 and update the plan, tracker, `contract.py`, and every dependent test, manifest, receipt, and proposal/contract-hash consumer;
3. implement exactly five `uint16_be` key fields, one `uint64_be(10)`, raw `uint32_be(e)`, and `D(e)[0]`, with range checks and no `H()` wrapper or additional prefix;
4. implement the frozen binary32 operation order, cap, lower-middle rule, actual-versus-hypothetical distinction, and amplitude-1.00 flat-plateau behavior without reassociation;
5. independently verify all five row witnesses, all three inventory witnesses, the inactive lookup, all 10,368 keys, every generated response hash, the complete manifest, and its aggregate hash in the locked runtime; and
6. keep `BLOCKED_UNDEFINED_ROW_KEY_PREIMAGE` and every downstream stop in force until the canonical patch is fully rebound and freshly reviewed. No S0 or gate status may be inherited from this acceptance.

## 9. Authority ceiling

This `ACCEPT` verdict binds only the reviewed amendment bytes. It grants no authority to perform the canonical patch, edit code or tests, generate a manifest, use Git, run P2/P3, commit S0, pass G4 or any later gate, launch a server, access natural/evaluator/sealed/held-out data, train, use results, advance claims, or promote the paper.

## 10. Final disposition

**ACCEPT — same-family provisional, zero authority.**  
**Blocking issue count:** `0`.  
**Canonical patch conditions:** mandatory and non-authorizing.
