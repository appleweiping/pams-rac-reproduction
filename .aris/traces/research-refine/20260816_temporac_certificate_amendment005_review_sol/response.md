# TempoRAC Contract Amendment 005 — Fresh Normative Review

**Verdict:** `REVISE`

**Status:** `REVISE_SAME_FAMILY_PROVISIONAL_ZERO_AUTHORITY`

**Review date:** `2026-08-16`

**Reviewer:** `gpt-5.6-sol` / OpenAI family / same-family provisional

Amendment 005 materially improves the intended certificate design. In
particular, it freezes the correct 16-member teacher state inventory, removes
caller-supplied natural-teacher outputs, preserves the unchanged ten-key target
receipt and seven-member target NPZ, distinguishes the 402-identity prediction
population from the K1-certified development subset, and specifies direct
`MappingProxyType` composition. Those improvements are not yet a closed,
authority-bearing contract. Seven blocking provenance and ordering defects
remain, so this review cannot accept the amendment or authorize implementation.

## 1. Exact reviewed snapshot

The requested amendment inputs were stable throughout review:

| Input | Bytes | SHA-256 |
|---|---:|---|
| `TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE.md` | 25,001 | `fa5ef054de0e243a030905b7a9d0e302aa442cb61b4d39012eaae6dca05a6869` |
| `TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE.json` | 24,383 | `5ea02ae5ccb7ea0f1771222f254ded74318fdb4bc96a5655d5de58a66d7878a2` |
| canonical `FINAL_PROPOSAL.md` | 79,366 | `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391` |
| certificate postfix Round-2 review MD | 12,283 | `93a9cae771b34c185b9057286794b894d615aec5424011df145aaf8900f008d8` |
| certificate postfix Round-2 review JSON | 14,527 | `144d2f22f4f75166b8cbefe54157d08690408979dd7cb34ef8c97e1738432872` |
| `EXPERIMENT_PLAN.md` | 34,198 | `4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e` |
| `EXPERIMENT_TRACKER.md` | 22,360 | `714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b` |

The Markdown and JSON contain the same 16-entry bound-input map, and every
declared digest matches the current file bytes. Both amendment files decode as
strict UTF-8, have no BOM or CR bytes, and have exactly one terminal LF. The
JSON parses with no duplicate object key and no non-finite JSON constant.

## 2. Blocking findings

### A005-B1 — the 120-row selection scores have no authority-bearing source (`R2-F1`)

Section 6.1 commits seven values per row, but the only provenance field is a
checkpoint-receipt digest. Neither the checkpoint receipt nor the successful
run receipt commits the tune-evaluation bytes from which
`tune_objective`, `negative_edge_fraction`, `tune_abstentions`, and
`mean_masked_reconstruction` were obtained. Section 7 supplies the loader with
the score-index bytes, checkpoint/run receipts, and one checkpoint artifact;
it supplies no tune-evaluation artifact, accumulator ledger, or independent
score receipt.

Consequently, an attacker can keep 120 valid `(seed,step)` checkpoint-receipt
digests, replace the finite score scalars so an arbitrary checkpoint wins,
canonicalize the new index, and issue a matching selection receipt. Every
specified Section 6 comparison succeeds because it checks only the submitted
scores. The defect is amplified by the absence of the mathematically required
nonnegative domains for `tune_objective` and
`mean_masked_reconstruction` and by the absence of exact tune denominators and
reduction order.

Required revision: define one closed, deterministic score-evaluation preimage
for every checkpoint, including the exact U0024--U0031 view inventory, code,
configuration, environment, checkpoint artifact, denominators, accumulator
order, and all four output values. Commit that evidence before selection in an
authority-bearing G1/run chain and require independent rederivation. A digest
of a caller-authored score index is not score evidence.

### A005-B2 — the loader cannot verify 119 of the 120 checkpoint receipts (`R2-F1`)

Section 5 requires each 16-row member ledger to be recomputed from its exact
checkpoint artifact. Section 7 nevertheless gives `SelectedTeacher` only the
globally selected artifact, while claiming to perform every artifact, member,
inventory, and run check for all 120 receipts. For the other 119 checkpoints,
the verifier can compare a receipt's declared artifact digest with the run
receipt, but it cannot rederive the artifact byte count, member names, NPY
bytes, member hashes, dtypes, or shapes. The stated construction is therefore
not implementable as written.

Required revision: provide all 120 exact checkpoint artifacts to the verifier,
or replace the claim with a separately closed and previously authority-bearing
artifact/member inventory that is actually available and sufficient for every
stated check. The selected artifact must still be strict-loaded and reserialized
byte-for-byte. The CPU replay lock must also spell out the exact intra-op,
interop, BLAS/oneDNN, CPU ISA/dispatch, deterministic-algorithm, and thread
settings needed for bytewise GELU/linear output reproduction; the phrase
`single-threaded` alone does not define those settings.

### A005-B3 — new receipts are not bound to the effective amended contract (`R2-F1`, `R2-F2`)

Sections 5 and 9 require `contract_sha256` to equal only the old canonical
proposal digest `3b265b...9391`. Section 6 lists `contract_sha256` in the
selection receipt without defining its value, and Section 10 does the same for
the G5a root index. No runtime receipt commits either exact Amendment 005 hash.
The current `contract.py:10` likewise exposes only the old proposal digest.
Thus identical receipt bytes can be interpreted under changed, rejected, or
superseded Amendment 005 semantics.

The claimed exact source snapshot is also incomplete. Bound
`preprocess.py` and `certify.py` directly import unbound `quadrature.py`; bound
`certify.py` directly imports unbound `x0.py`; and bound `evaluator.py` directly
imports unbound `metrics.py`. Prediction construction and training/run
selection also depend on unbound `prediction.py`, `runtime.py`, and
`training.py`. Their current exact hashes are:

| Omitted dependency | SHA-256 |
|---|---|
| `src/pams/temporac/quadrature.py` | `44f4b91dc4d5631442982732177a6f57726c0b8f42594380b5a75f2209efc2f3` |
| `src/pams/temporac/x0.py` | `5f4585425a83bf63779265db9004d210c5c2875a579ebc9a6c51297eb647c7a2` |
| `src/pams/temporac/metrics.py` | `1dadf08e89330bed2ab773afda05bffcb3fe2252b0d3e23c0bcc18c2c3b049dd` |
| `src/pams/temporac/prediction.py` | `7c57725240e83b7c06227a3b27e5f3cadb4590170488e5e126180caba8c13808` |
| `src/pams/temporac/runtime.py` | `9421661046ee686508607529ed972b087e724c2c3a5c487e220c2f8e4038a0c8` |
| `src/pams/temporac/training.py` | `0af8ecbadfcda43d2b9f60853c99210b316786ff4310ed5d0a32d9e366f627a4` |

Required revision: define a noncircular effective-contract digest over the
exact accepted proposal/amendment bytes, or integrate the amendment into a new
canonical proposal and use that new digest. Assign the exact value of every
`contract_sha256` field and propagate it through unchanged receipt key sets,
including feature, run, target, prediction, checkpoint, selection, certified
index, and root-index receipts. Expand the reviewed implementation snapshot to
the complete executed/imported closure and bind the later implementation
review to the exact accepted amendment bytes.

### A005-B4 — the five-key G5a root index does not define a closed receipt DAG (`R2-F2`)

Section 10 says `receipt_sha256` is complete by comparison with frozen
population, prediction, checkpoint, target, plan, and surrounding-scope
indexes. Neither the proposal, plan, tracker, current code, nor Amendment 005
defines closed schemas, byte preimages, order, counts, or class/identity
associations for those prediction/checkpoint/target/scope indexes. The base
proposal also names `feature_root_sha256`, `checkpoint_root_sha256`, and the two
prediction roots without defining their aggregate preimages. A sorted multiset
of submitted digests proves only what was submitted; it cannot prove that the
submitted multiset is the unique complete inventory.

The DAG boundary is additionally ambiguous: “every receipt artifact in the
already frozen G5a receipt inventory” does not expressly exclude the containing
G5a receipt, the capability-request record, or later grant/G5b/K7 receipts.
Including the G5a receipt creates a self-hash cycle; excluding it is sensible
but must be normative. The Markdown also leaves the root-index
`contract_sha256` value and byte-count range undefined, while the JSON alone
calls the byte count positive.

Required revision: freeze every contributing index/root preimage, exact row
schema, order, class count, identity key, artifact/receipt association, and
multiplicity derivation. Specify the complete acyclic edge list and explicitly
exclude the containing and future receipts. Require every digest dereferenced
by the certified-development index to occur in the correct closed inventory,
not merely somewhere in an untyped digest bag. This can use existing G5a keys,
but their root preimages must become unique and mechanically derivable.

### A005-B5 — `Cdev` and same-identity feature provenance are not frozen at K1 (`R2-F2`, `R2-F3`)

The amendment correctly says `Cdev` is every and only K1-certified development
identity, not all 402 identities. It does not provide a K1 coverage-index schema
or a K1 receipt digest that commits that sequence before G5a. The new certified
development index first becomes committed at G5a, and K7 replays only its
listed rows. No step recertifies the excluded development identities. A caller
can therefore choose any later subset meeting 108 identities and eight
components, omit other actually certified identities, and still satisfy every
listed K7 row check.

There is a related swap gap. Section 9 validates each named feature receipt and
identity but never explicitly requires its digest to equal
`feature_receipt_sha256_or_null` in the frozen population-manifest row. A
different well-formed feature for the same `(opaque_key_hex,slot)`, together
with its rederived natural input and target, can therefore form a
self-consistent row unless an underdefined surrounding root happens to reject
it.

Required revision: commit a closed K1 coverage index that enumerates every
frozen G0-eligible identity, certification result, and component mapping, then
require the G5a index's development-certified projection to equal it exactly.
Alternatively, K7 must recertify the complete exact 134-development feature
population and prove both inclusion and exclusion. Each feature receipt and
component key must equal the same frozen population-manifest row, not just a
same-identity receipt supplied with the index.

### A005-B6 — prediction component membership remains caller-controlled (`R2-F3`)

The 402-identity and 9,648-envelope equalities close the demonstrated subset
bypass, but they do not bind source-component membership for metric bootstrap.
The current `PredictionEnvelope.component_token` is caller-supplied ASCII, and
`metric_rows_from_predictions` forwards it directly into the component-cluster
bootstrap. Amendment 005 requires manifest component equality only for Cdev
rows, not for all 134 development identities whose predictions define the K7
metric population.

An attacker can preserve every identity, arm, seed, condition, artifact, and
count while relabelling development videos among components, changing the
bootstrap distribution and possibly the K7 decision. Required revision: derive
every prediction's component key from the exact population manifest inside the
evaluator, require equality across all 24 envelopes for that identity, and
reject any caller component label. This does not expand Cdev: metrics still use
all 134 development identities, including zero-estimate stubs.

### A005-B7 — the pre-consumption vault requirement contradicts F23 (`R2-F3`)

Section 12 makes the validated vault identity projection equal to P402
mandatory “before capability consumption.” The canonical proposal and plan
instead require the grant to be consumed when the one non-resumable evaluator
process starts, after which G5b alone opens and validates the vault. A real
vault projection cannot be validated before the capability that makes the
vault reachable. Moving the join earlier would violate the stated unchanged
F23 order and the data firewall; moving consumption later would make a failed
G5b retryable.

Required revision: before consumption, validate only non-vault commitments:
the exact population manifest, 9,648 predictions, receipt roots, and Cdev
evidence. Atomically consume the grant at evaluator-process start. Then G5b
must validate the actual vault bijection against P402, derive cardinality 402,
and burn the grant on any failure; K7 follows only after G5b PASS in the same
process.

## 3. Closure assessment

| Intended closure | Result | Reason |
|---|---|---|
| `R2-F1` selected teacher and natural inference | `REVISE` | The state inventory, strict selected-artifact load, landmark-before-teacher order, Neumaier F5, and one CPU forward are well directed, but scores are unauthoritative, 119 artifacts are unavailable, and the effective contract/runtime closure is not bound. |
| `R2-F2` target provenance and K7 byte replay | `REVISE` | Bytewise target rebuilding is the right mechanism and keeps the ten-key receipt unchanged, but the receipt DAG, canonical feature link, and K1 sequence are not closed. |
| `R2-F3` populations and evidence | `REVISE` | P402, 9,648, and Cdev semantics are scientifically correct, but component clustering is caller-controlled and the vault pre-consumption rule is impossible. |
| `R2-F4` direct Mapping composition | `SPECIFICATION CLOSURE PRESENT` | Internally copying the six validated builder fields to a plain dict permits direct `MappingProxyType` composition. Implementation remains pending and Amendment 003's separate P05M blockers remain unchanged. |

For F4 hardening, a consumer accepting a generic `Mapping` should first take a
plain-dict snapshot, validate that snapshot, and canonicalize that same
snapshot. The amendment's validate-then-copy order is safe for the current
builder's unexposed `MappingProxyType`, but snapshot-then-validate avoids a
time-of-check/time-of-use gap for a stateful custom mapping.

## 4. Positive checks and non-expansion

- The listed 16 checkpoint members exactly match the current
  `TempoRACTeacher().state_dict()` names, shapes, and float32 dtype. A
  non-authorizing in-memory feasibility probe serialized exactly 16 sorted
  members in a 1,012,518-byte deterministic stored NPZ.
- Geometry-only per-run landmarks precede F5 and teacher inference; the
  written same-run consecutive-landmark construction avoids a cross-run
  traversal. The explicit Neumaier update and written mean/second-moment
  operation order are materially stronger than the current ordinary
  `np.sum`/`np.mean` implementation.
- The unchanged target surface is correctly preserved: seven exact NPZ members
  and the existing ten receipt keys. Bytewise K7 reconstruction is the proper
  way to close same-target digest relabelling without adding target keys.
- `P402` is correctly 268 train plus 134 development identities. Exactly
  `402*4*3*2=9,648` prediction envelopes are required. `Cdev` is correctly a
  subset of the exact 134 development identities with floors 108 identities
  and eight components; it is not all 402 and need not be all 134. K7 metrics
  remain over all 134 development identities with stubs.
- No algorithm, scientific threshold, training job, GPU budget, dataset scope,
  target/G5a receipt key set, claim, or paper-visible evidence block is added.
- Current source/tests do not implement the new checkpoint/selection/root-index
  schemas or `SelectedTeacher`; that is consistent with proposed status but
  means no implementation acceptance follows from this review. No gate-bearing
  test suite was executed and no code or test file was modified.

## 5. Required fresh evidence after revision

A later implementation review must at minimum reject:

1. score changes with unchanged checkpoint/run receipts and score swaps between
   valid seed-step rows;
2. a missing nonwinner artifact, forged nonwinner member ledger, incomplete
   120-row inventory, altered checkpoint bytes, and non-strict load;
3. base-proposal-only or unspecified contract digests and any drift in a direct
   or transitive executed dependency;
4. missing, duplicated, wrong-class, wrong-identity, or self-referential entries
   in the G5a receipt DAG;
5. omission of an actually K1-certified development identity and a
   same-identity alternate feature/natural-input/target chain;
6. prediction component relabelling with otherwise exact P402/9,648 bytes;
7. any prediction/vault/population duplicate, extra, missing, or subset while
   retaining the correct rule that Cdev alone has the 108/8 floors;
8. vault access before consumption, retry after G5b failure, or K7 before G5b;
9. direct canonical builder `MappingProxyType` consumption failure and a
   mutable-mapping snapshot/validation mismatch.

The positive path must independently reproduce the score evidence, selected
teacher, natural input receipt, target NPZ, target receipt, complete root
inventory, P402/9,648 inventory, exact Cdev sequence, G5b cardinality, and K7
inputs byte for byte under one pinned effective contract.

## 6. Authority boundary and final disposition

This review is non-authoritative and authorizes nothing. `P2=0`, `P3=0`,
`S0=0`, `gate=0`, `capability=0`, `launch=0`, `server=0`, `data=0`, `gpu=0`,
`training=0`, `heldout=0`, `results=0`, `paper_claim=0`, `git=0`, and `test=0`.
The existing plan/tracker blockers, the independent Amendment 003 `REVISE`,
the partial-cache appendix-only restriction, and all launch/data/training
barriers remain in force.

Final disposition: `REVISE`. A new exact Amendment 005 MD/JSON pair closing all
seven blockers requires a fresh zero-context normative review before any later
implementation review.
