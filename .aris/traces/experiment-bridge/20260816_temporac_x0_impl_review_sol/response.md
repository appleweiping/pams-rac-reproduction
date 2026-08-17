# TempoRAC X0 implementation and candidate review

> **VERDICT: PASS**
>
> **SAME-FAMILY PROVISIONAL — CANDIDATE REVIEW ONLY**
>
> **P2 = 0; S0 = 0; LAUNCH AUTHORIZATION = 0**
>
> **NON-AUTHORITATIVE; AUTHORIZES NOTHING**

**Date:** 2026-08-16  
**Reviewer model:** gpt-5.6-sol  
**Reviewer family:** openai  
**Executor family:** openai  
**Review independence:** same-family  
**Acceptance status:** provisional

## Decision

The canonical TempoRAC X0 implementation and the five-file
data/temporac_p00_v4/x0_candidate_20260816 artifact pass this fresh,
zero-context review. The implementation matches Section 6.1 of proposal
SHA-256 3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391
and the provisionally accepted Amendment 001. A fresh locked linux/amd64
replay regenerated all 2,560 attempts and all five candidate files
byte-for-byte.

This PASS applies only to the reviewed candidate bytes. It grants no P2, S0,
gate, training, natural-data, evaluator, server, result, claim, paper, Git,
or launch authority. The receipt remains authoritative=false, authorizes=[],
p2_status=NOT_CLAIMED, s0_status=NOT_CLAIMED, with
launch_authorization_zero still a blocker.

**Blocking findings:** 0.  
**Candidate-scope non-blocking findings:** 0.

## Bound inputs

| Input | SHA-256 |
|---|---|
| refine-logs/temporac/FINAL_PROPOSAL.md | 3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391 |
| refine-logs/FINAL_PROPOSAL.md | 3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391 |
| Amendment 001 Markdown | ca44173bd2bc2be3fa5eaf9dc362d88d981603d28328bf365efc68567f1ae5bb |
| Amendment 001 JSON | b1fb03f5cd5021ff6397f74b9b6a8cd074d0ed5fd2e72746fc40df44266a41cc |
| Amendment 001 ACCEPT review Markdown | ad14875c91faa184f156f86e9d559a9bb8927db8a2309dc4fbd963074e1a3a58 |
| Amendment 001 ACCEPT review JSON | 8f38d6be24832a03ec8042062c80d5e6a2afbabcb8a78485ec21137b46581844 |
| src/pams/temporac/contract.py | 5ab8fb62dc558ba78e95fc07e52c1f1a1afbff5b1c5003834079eea6143d745f |
| src/pams/temporac/x0.py | 5f4585425a83bf63779265db9004d210c5c2875a579ebc9a6c51297eb647c7a2 |
| tests/temporac/test_x0.py | 90206f53c7fffeabe9bcb3d6374fd60d3c51c69a7d1137df0cba797b583d1ef8 |
| tests/temporac/test_x0_manifest.py | 074bb8df9dce5d1ea9e987499dac0d8824e1803a5f4bdc32495bd957e8c4c6a3 |
| generator | 5a58f4ad3a4d178122c6121347c657bf9f3ad5916d52e23b5647ea2fd2bab0a2 |

The proposal copies are byte-identical, contract.CONTRACT_SHA256 names the
same proposal digest, and all receipt source bindings rehash exactly.

## Numerical implementation audit

### Exact full-bracket outer multiplication

src/pams/temporac/x0.py assigns the unscaled displacement fundamental,
performs the h=2, h=3, and h=4 paired additions in increasing order with
in-place accumulation, then executes exactly one multiplication by the
binary64 X0_SCALE=np.float64(25.0/6.0). The tangent completes its fundamental
and the same increasing-harmonic accumulation before exactly one identical
outer multiplication. The scale is not distributed, reassociated, or applied
to the basis, MGS inputs, or base pose. The pinned np.array_equal operation
order test passed.

### Exact 4096 by 34 separation reduction

Production makes pose and shifted C-contiguous binary64 arrays, forms
difference=pose-shifted, calls np.square, calls
np.mean(...,dtype=np.float64) over all 4096*34 C-order elements, converts once
to Python float, and calls math.sqrt. The stored 4096x17x2 shape has exactly
the contracted 4096x34 C-order stream, including the 18 fixed coordinates.

### Literal reversal FFT

Production retains the required C-order flatten, reversed-grid indexing,
rfft of both arrays on axis 0, conjugate product, coordinate sum, irfft with
n=4096, energy, nonnegative clamp, square root, and minimum sequence. No
closed-form witness replaces the production FFT. The exhaustive generator
invoked this predicate for all 2,560 attempts.

### No F9-and-later drift

The current source from blocks_for_source through generate_view was inspected
and the F9/F10 test rerun. As an auxiliary drift witness, normalized
instruction streams from the pre-canonicalization local CPython cache were
compared against a fresh compile. blocks_for_source, _direct_map,
_sample_knots, window_starts, _analytic_responsibility, and generate_view are
identical; their 37,469-byte normalized inventory has SHA-256
21a1c6a635c63db3b25828077eb22b61b88231fec960083414299067ceed8157.
X0Block and X0View fields are unchanged. Meaningful changes are confined to
F8 scale, Section 6.1 reduction order, and manifest-hash validation.

## Candidate bytes and closed schema

| Member | Bytes | SHA-256 | Fresh replay exact |
|---|---:|---|---|
| attempt_replay.json | 691,831 | 000ae695570c51b51604e53f1189aaa2e6250152965242ff77f22d947a906f9b | yes |
| candidate_receipt.json | 1,341 | d56b14e4bd0371bd10ec6366a3ccfc724c934dcb8f8bac08cc242fe19102c4f4 | yes |
| environment_lock.json | 659 | 2ff61ad14a1e004aab2cac31ff28a7c9ffa8b054ab10054d23b1d7230b9f5d87 | yes |
| replay_witness.json | 1,016 | a55a6db52a2cd43882a1859d58b6ebe0b05e7b915f166b698ce1ee53808df547 | yes |
| temporac.x0-manifest.v4.json | 18,228 | 5800a71af220700ed285c29273d87ac73877296242e1d7ca2e6cb9d2c4780b25 | yes |

Exactly these five canonical UTF-8, single-LF files exist. The manifest has
40 ordered U0000...U0039 rows, split 24/8/8, eight exact row keys, and retains
attempt 0 throughout. attempt_replay has 2,560 ordered rows, six exact row
keys, attempts 0...63 for every source, 2,560 guard passes, and zero MGS
rejections. The receipt has its exact nine-key closed schema; every member
hash matches. Its authority is explicitly false and empty.

## Independent reproduction and non-overwrite

The inputs were copied unchanged to:

D:/Temp/temporac_x0_impl_review_20260816_fc70cbe39889438b8970476f34cf8c55

The generator ran there in Docker platform linux/amd64, python:3.12.4,
CPython 3.12.4, NumPy 2.1.0, SciPy 1.14.1, little endian. It emitted receipt
digest d56b14e4bd0371bd10ec6366a3ccfc724c934dcb8f8bac08cc242fe19102c4f4.
Every regenerated file matched the reviewed byte stream exactly.

A second invocation against the same target emitted the expected refusal,
exited with container code 2, and left all five hashes and UTC modification
times unchanged.

## Gates

| Gate | Result |
|---|---|
| Pinned Docker test_x0.py + test_x0_manifest.py | PASS — 6 passed in 51.41 s |
| Full pytest | PASS — 1,131 passed, 7 skipped, 1 warning in 396.06 s |
| ruff check src tests scripts --no-cache | PASS |
| mypy src/pams/temporac | PASS — 24 source files |
| mypy contract.py, x0.py, generator | PASS — 3 source files |

An extra repository-wide mypy src run found 29 pre-existing errors in three
unrelated baseline/diagnostic modules. Including the X0 tests separately
found one annotation-only np.float64-to-float mismatch at test_x0.py:146.
Neither affects production, generator, artifact bytes, or behavior; no file
was changed to hide these ambient findings.

## Integrity and authority

This deterministic synthetic construction is simulation_only. It has no
natural inputs, labels, evaluator capability, model-derived ground truth,
score normalization, training, or performance claim.

**Final disposition: PASS — same-family provisional, candidate-only, zero
authority. P2 remains 0; S0 remains 0; launch authorization remains 0.**
