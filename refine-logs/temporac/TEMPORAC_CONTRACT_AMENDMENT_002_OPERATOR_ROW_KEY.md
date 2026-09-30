# TempoRAC Contract Amendment 002: Operator Row-Key and Response Bytes

> **STATUS: `PROPOSED_PENDING_FRESH_REVIEW`**
>
> **SAME-FAMILY PROVISIONAL — NON-AUTHORITATIVE**
>
> **NO GATE PASSAGE, TRAINING, DATA ACCESS, RESULT, CLAIM, PAPER PROMOTION, SERVER LAUNCH, OR GIT AUTHORIZATION**

**Date:** 2026-08-16

**Would amend, only if accepted:** Section 9, lines 788--792, in both `refine-logs/FINAL_PROPOSAL.md` and `refine-logs/temporac/FINAL_PROPOSAL.md`; the two amended proposal files must remain byte-identical

**Review disposition addressed by this revision:** `REVISE`. This revision is not an acceptance, PASS receipt, or authority grant; a new fresh review is still required.

**Scope:** exact bytes for the already frozen 10,368 operator row keys, SHA-256 byte selection, inactive float32 values, and plateau float32 values; the Cartesian population and every downstream scientific contract remain unchanged

## 1. Missing byte contract

Section 9 freezes the operator population as

```text
gap x width x amplitude x offset x layout
= 9 x 3 x 3 x 32 x 4
= 10,368 rows
```

in that Cartesian order. It then defines inactive edge values using the “corresponding byte” from

```text
SHA256("temporac.operator.v4\0" || length-prefixed row key || uint32_be(e))
```

but does not define the row-key field bytes, the length-prefix width and endianness, or which digest byte is corresponding. Applying the general `H(tag,*fields)` helper is not uniquely implied and would also length-prefix `uint32_be(e)`, changing the written preimage. Canonical JSON and several fixed-binary encodings are all plausible and yield different responses. Therefore the current production implementation correctly reports `BLOCKED_UNDEFINED_ROW_KEY_PREIMAGE`; S0 and G4 cannot accept any operator manifest.

## 2. Sole normative correction

If and only if this amendment passes fresh review, define categorical indices in the already frozen order:

```text
aidx(0.60)=0, aidx(0.75)=1, aidx(1.00)=2

lidx(interior)=0
lidx(left-boundary)=1
lidx(right-boundary)=2
lidx(run-reset)=3
```

For row `(g,w,a,o,l)`, define the exact 10-byte key

```text
K = uint16_be(g)
 || uint16_be(w)
 || uint16_be(aidx(a))
 || uint16_be(o)
 || uint16_be(lidx(l))
```

and for zero-based edge index `e`, define

```text
P(e) = ASCII("temporac.operator.v4")
    || 0x00
    || uint64_be(10)
    || K
    || uint32_be(e)

D(e) = SHA256(P(e))
b(e) = unsigned integer value of D(e)[0]
```

There is exactly one `uint64_be(10)` length prefix around `K`. There is no per-field length, no length around `uint32_be(e)`, no `H()` wrapper, no digest cycling, and no `e mod 32` selection. The key is independent of Python float formatting and platform endianness. All integers are range-checked before encoding.

Let `RN-even-f32(x)` denote direct round-to-nearest, ties-to-even conversion of
the exact real `x` to IEEE-754 binary32, and let
`float32_bits(0x3ecccccc)` denote the exact finite binary32 value with that bit
pattern. The inactive edge value is

```text
inactive(e) = min(
    RN-even-f32((255 + 3*b(e)) / 2550),
    float32_bits(0x3ecccccc)
)
```

The uncapped expression is algebraically the proposal's
`0.10+0.30*b/255`. Production must form the exact integer numerator, divide
once in binary64, cast once to NumPy little-endian `<f4`, apply the exact
binary32 cap, and retain that scalar. Fused multiply-add or an alternate
sequence through decimal `0.10` and `0.30` is forbidden for committed bytes.
Relative to the uncapped correctly rounded table, only `b=255` changes: its
uncapped bits are `0x3ecccccd`, its committed bits are `0x3ecccccc`, and its
committed little-endian bytes are `cccccc3e`.

Define the 1,024-byte lookup witness in increasing unsigned byte order as

```text
L_inactive = concat_(b=0)^255 little_endian_uint32(bits(inactive_from_byte(b)))

SHA256(L_inactive)
= afa6eff0d744acfdf9cf78adc43aa8daf01633f3345d7567d642a7782885040d
```

The largest inactive value is exactly the binary32 value
`0x3ecccccc`; after exact widening to binary64, its negative margin against
the decoder threshold is `0.5 - inactive_max = 0.10000002384185791`.

For every plateau `[s,s+w)`, its lower-middle edge is
`s+floor((w-1)/2)`. That edge uses the listed boosted value and every other
plateau edge uses the listed ordinary value. All are exact `<f4` constants:

| `aidx` | amplitude | ordinary binary32 bits | boosted value | boosted binary32 bits |
|---:|---:|---:|---:|---:|
| 0 | 0.60 | `3f19999a` | 0.65 | `3f266666` |
| 1 | 0.75 | `3f400000` | 0.80 | `3f4ccccd` |
| 2 | 1.00 | `3f800000` | 1.00 | `3f800000` |

For amplitude `1.00`, ordinary and boosted values are identical. In
particular, width-2 and width-4 plateaus are flat at binary32 `1.0`; neither
this amendment nor a generated-response check may claim or require a unique
maximum for those plateaus.

Responses are one-dimensional little-endian `<f4` arrays in increasing edge order. All truth plateaus, masks, expected components, and generated response arrays retain the existing Section 9 layout and are committed in the existing Cartesian row order.

## 3. Frozen witnesses

A fresh reviewer must independently recompute at least these witnesses from the preceding bytes:

| `(g,w,aidx,o,lidx)` | `e` | `K` hex | `SHA256(P(e))` | `b(e)` | inactive-function `<f4` bytes | actual generated response |
|---|---:|---|---|---:|---|---|
| `(4,1,0,0,0)` | 0 | `00040001000000000000` | `54ac35705b42e8296cf324b893e5450c123b8b48c45e912cba928a6c432f60c4` | 84 | `65984b3e` | inactive, `65984b3e` |
| `(4,1,0,0,0)` | 32 | `00040001000000000000` | `dee0ad937ac053341db0c0b1dc6a179c6684e9a6108b76336070f4cb910e1170` | 222 | `1fecb83e` | boosted plateau 0.65, `6666263f` |
| `(5,2,1,7,2)` | 17 | `00050002000100070002` | `9406bcc666d0a8487362b65700a5daebd6d3f4496c54c5b2da1f849ba52d3ee1` | 148 | `26598c3e` | inactive, `26598c3e` |
| `(129,4,2,31,3)` | 319 | `008100040002001f0003` | `fb06baef760aaeb7ab2a4aeaad5d13dc457afc989fb75db931e1535c4213c215` | 251 | `fe63ca3e` | inactive, `fe63ca3e` |
| `(4,1,0,0,1)` | 42 | `00040001000000000001` | `ff78168a29882190e18b1fd5957cf1eb549c721c227344c8fbfc20ec8d9f4fb3` | 255 | `cccccc3e` | inactive, `cccccc3e` |

The `e=32` row evaluates the inactive function only hypothetically: edge 32
is the first width-1 interior plateau edge in that row, so its actual generated
response is boosted 0.65 with little-endian bytes `6666263f`. The final row is
the first actual inactive `b=255` occurrence when rows are scanned in frozen
Cartesian order and eligible run edges are scanned in increasing edge order.

The first complete preimage is

```text
74656d706f7261632e6f70657261746f722e7634
00
000000000000000a
00040001000000000000
00000000
```

For full-inventory witnesses, let `K_r` be row `r=0,...,10367` in frozen
Cartesian order and let `D_r(e)=SHA256(P_r(e))`. Concatenation below is direct
byte concatenation with no outer count, delimiter, ASCII hex, JSON, or `H()`
wrapper:

```text
I_key = concat_(r=0)^10367 (uint64_be(10) || K_r)
bytes(I_key) = 186624
SHA256(I_key)
= 18dfdafe237b77565de36bab6d4642efb314ec1cbbd76e852938ed5c4d424a55

I_edge0_digest = concat_(r=0)^10367 D_r(0)
bytes(I_edge0_digest) = 331776
SHA256(I_edge0_digest)
= ba797f6ae59cbbb40f19a61199afaaa1ac5b2c001a3c9fc34f1dba8f79cf78e4

SHA256(current canonical semantic-row JSON bytes)
= 0fe8b2619a3f33cc4a83485053888c04c567b015eea65f81bf6cd9ec715ee4f2
```

Each `D_r(0)` contribution to `I_edge0_digest` is the raw 32-byte digest, not
its 64-byte lowercase hexadecimal rendering. The semantic JSON inventory
contains exactly 746,714 bytes. The binary key inventory must contain exactly
10,368 unique keys of exactly 10 bytes each. These witnesses are specification
checks, not S0 receipts or scientific results.

## 4. Non-expansion and failure behavior

This amendment adds no operator row, layout, gap, width, amplitude, offset, source, seed, model parameter, job, GPU hour, metric, baseline, result, or claim. It does not alter plateau locations, run/reset geometry, target components, NOLA, decoder semantics, topology, F23 order, data protocol, or evidence ceiling. It is a deterministic serialization correction only.

Until a fresh review accepts this amendment and both canonical proposal copies
are patched byte-identically and rehashed:

- `operator_row_key_preimage` must continue to raise `BLOCKED_UNDEFINED_ROW_KEY_PREIMAGE`
- no generated-response array or 10,368-row operator manifest may be treated as canonical
- P2 may enumerate and hash semantic tuples but cannot pass the operator fixture
- P3, S0, G4, K5, K6, training, natural-data acquisition, evaluator capability, result claims, paper promotion, server launch, and Git release remain unauthorized

After acceptance, the two `FINAL_PROPOSAL.md` copies must first be amended to
byte-identical bytes and assigned one new shared SHA-256. The canonical
`EXPERIMENT_PLAN.md`, `EXPERIMENT_TRACKER.md`, `src/pams/temporac/contract.py`,
and every other proposal/contract-hash consumer, test, manifest, and receipt
binding must then be updated to that same digest before the blocker may be
removed. This proposed amendment performs none of those downstream changes.

Implementation must independently verify all five row witnesses, the
1,024-byte inactive lookup witness, both binary-inventory witnesses, the
function-only versus actual `e=32` bytes, all 10,368 unique keys, every
generated response hash, the complete canonical manifest, and its aggregate
hash before the blocker may be removed. The amendment cannot be introduced
after S0 or used to repair a failed later gate.

## 5. Bound snapshot

| Input | SHA-256 |
|---|---|
| `refine-logs/FINAL_PROPOSAL.md` | `562325dedab0637421f62dec5ad149b3c884f67b47c4b525c8fb54a57bb1233c` |
| `refine-logs/temporac/FINAL_PROPOSAL.md` | `562325dedab0637421f62dec5ad149b3c884f67b47c4b525c8fb54a57bb1233c` |
| `refine-logs/EXPERIMENT_PLAN.md` | `3fd1986871b32674d74b395fbfc9720712d9fba0c8583c24c1d5bddb87086eb6` |
| `refine-logs/EXPERIMENT_TRACKER.md` | `c7e2f32c3c81b66884abb13760a671ecc7cdd314fd6c994a4ed4c46cb754a63b` |
| `src/pams/temporac/contract.py` | `b71255a35cce1701d2741064064b53ba819b742873a348f1852f147149e36467` |
| `src/pams/temporac/hashio.py` | `2d81e415b45e3d1b612f998b6a8386a967dfdfcfaac31bc914b2c95cbaeeab3a` |
| `src/pams/temporac/fixtures.py` | `a8d10df7c8e3cbcb5684198192d3a553921da0f6aa92f03e54448ee182671406` |
| `tests/temporac/test_gates_fixtures.py` | `4c33a46d6dfe4119c3a82e1d8df5602d12bc1fd469664805be1a67be75ad6443` |

The amendment remains `PROPOSED_PENDING_FRESH_REVIEW`, same-family provisional, non-authoritative, and non-executing.
