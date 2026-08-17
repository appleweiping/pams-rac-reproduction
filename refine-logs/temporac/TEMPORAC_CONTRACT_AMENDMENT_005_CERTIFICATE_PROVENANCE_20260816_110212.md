# TempoRAC Contract Amendment 005 — Certificate Provenance and Closed K7 Replay

**Status:** `PROPOSED_PENDING_FRESH_REVIEW`

**Review class:** `same-family provisional`

**Date:** `2026-08-16`

**Purpose:** close only Round-2 findings `R2-F1` through `R2-F4`. This
amendment freezes an authority-bearing selected-teacher chain, the noncircular
natural-teacher inference path, bytewise certified-development replay at K7,
the exact population/evidence equalities, and direct P05M builder/consumer
composition. It does not authorize execution and does not change the TempoRAC
algorithm, data, claims, jobs, thresholds, population, target schema, or gate
order.

## 1. Authority and non-changes

This document and its JSON mirror are proposed contract text only. Each of the
following authority values is exactly zero/false:

| Authority | Value |
|---|---:|
| `P2` | `0` |
| `P3` | `0` |
| `S0` | `0` |
| `capability` | `0` |
| `data` | `0` |
| `gate` | `0` |
| `git` | `0` |
| `gpu` | `0` |
| `heldout` | `0` |
| `launch` | `0` |
| `paper_claim` | `0` |
| `results` | `0` |
| `sealed` | `0` |
| `server` | `0` |
| `test` | `0` |
| `training` | `0` |

The following remain unchanged:

- canonical proposal hash and all F1--F23 scientific mathematics except for
  the provenance interfaces explicitly completed here;
- the 268 train plus 134 development identity population, four frozen natural
  arms, three seeds, two conditions, and exactly 9,648 prediction envelopes;
- the K1 development floors `dev_id>=108` and `dev_component>=8`;
- `temporac.target-receipt.v4` with exactly ten keys and its target NPZ with
  exactly seven members;
- all jobs, schedules, resource ceilings, data paths, dataset/cache scope,
  thresholds, estimands, bootstrap, claims, appendix-only restriction, and
  F23 gate/capability precedence;
- all earlier authority values and blockers. A fresh independent review is
  required before this amendment can support implementation acceptance.

## 2. Bound inputs

This amendment is bound to the following exact current bytes:

| Path | SHA-256 |
|---|---|
| `refine-logs/temporac/FINAL_PROPOSAL.md` | `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391` |
| `refine-logs/temporac/TEMPORAC_CERTIFICATE_IMPLEMENTATION_REVIEW_POSTFIX_ROUND2_20260816.md` | `93a9cae771b34c185b9057286794b894d615aec5424011df145aaf8900f008d8` |
| `refine-logs/temporac/TEMPORAC_CERTIFICATE_IMPLEMENTATION_REVIEW_POSTFIX_ROUND2_20260816.json` | `144d2f22f4f75166b8cbefe54157d08690408979dd7cb34ef8c97e1738432872` |
| `refine-logs/EXPERIMENT_PLAN.md` | `4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e` |
| `refine-logs/EXPERIMENT_TRACKER.md` | `714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b` |
| `src/pams/temporac/contract.py` | `5ab8fb62dc558ba78e95fc07e52c1f1a1afbff5b1c5003834079eea6143d745f` |
| `src/pams/temporac/teacher.py` | `30575fa64875aa6b77de157cf6af348b3b7900851a1c1742993842eea7928b30` |
| `src/pams/temporac/preprocess.py` | `042b5cae1917359cb5596837c81c0e36658ff99a6a8fd6f2602b478ca6f98bc3` |
| `src/pams/temporac/types.py` | `4249eac2738f5b2edf53f38cbb681dffae609c151d493dd03b625fe10ad4c3ca` |
| `src/pams/temporac/certify.py` | `468ce8418bff0f43582d341f3e1b8064cf1c429301d1f56be84b8e13ad59a3f1` |
| `src/pams/temporac/receipts.py` | `564039b5d37a4a2759544a9ae65f267b57d7f6c1e46bca812ae5ed4f4a2c17e4` |
| `src/pams/temporac/hashio.py` | `aa460d5de51f107b2fc6b2ad171747e0da38a87cf30f82a869a335c268f1d613` |
| `src/pams/temporac/evaluator.py` | `f981c6c7061c3388f06ef39466ef98192e65c62c501821be2b211f903cfaf9de` |
| `src/pams/temporac/feature_io.py` | `f54f17ca204c242d304314f4280e7540c28588e86ea588a405fff02176b95428` |
| `src/pams/temporac/trusted_packer.py` | `40d6e3d5be1b26b7ba0f0f7aadf8da7496c95c4e829face5c7e9faaee2cdf94e` |
| `src/pams/temporac/gates.py` | `b6c9d7432443d7ad36738a81700746709fd571d475514aa6f1bbcd39be744b4a` |

Changing any bound input requires a new snapshot and fresh review; this
amendment never silently follows later source edits.

## 3. Canonical byte rules

Every JSON artifact or receipt introduced here uses the proposal's canonical
JSON encoding: UTF-8 without BOM, `ensure_ascii=false`, `allow_nan=false`,
lexicographically sorted object keys, separators `,` and `:`, no insignificant
whitespace, LF only, and exactly one terminal LF. Duplicate, unknown, or missing
keys fail. Hashes are exactly 64 lowercase hexadecimal characters. JSON numbers
that carry scores are finite binary64 values; NaN and either infinity fail.

Raw array hashes are SHA-256 over exact C-order array bytes with no NPY header,
shape prefix, filename, or terminal LF. Artifact hashes are SHA-256 over the
complete exact artifact bytes. A receipt digest is SHA-256 over its complete
canonical JSON bytes. No schema defined here contains its own digest.

## 4. Canonical selected-teacher checkpoint artifact

The selected-teacher checkpoint artifact is a deterministic NPZ produced by
the exact fixed-metadata algorithm bound by `src/pams/temporac/hashio.py`:

- ZIP method is `ZIP_STORED`; compression, encryption, data descriptors,
  archive comments, per-member extras, alternate timestamps, and pickle are
  forbidden;
- members are flat `<state_dict-key>.npy` ASCII names sorted by raw ASCII
  bytes;
- every member is NPY version 2.0, `allow_pickle=false`, little-endian
  `<f4`, C-contiguous, and has its exact shape below;
- the archive has exactly these 16 members and no other member, tensor,
  optimizer state, RNG state, metadata object, or code object.

| Sorted NPZ member name | Exact array shape | Exact dtype |
|---|---:|---|
| `phase.layers.0.bias.npy` | `[256]` | `<f4` |
| `phase.layers.0.weight.npy` | `[256,247]` | `<f4` |
| `phase.layers.2.bias.npy` | `[128]` | `<f4` |
| `phase.layers.2.weight.npy` | `[128,256]` | `<f4` |
| `phase.layers.4.bias.npy` | `[2]` | `<f4` |
| `phase.layers.4.weight.npy` | `[2,128]` | `<f4` |
| `reconstruction.layers.0.bias.npy` | `[256]` | `<f4` |
| `reconstruction.layers.0.weight.npy` | `[256,34]` | `<f4` |
| `reconstruction.layers.2.bias.npy` | `[256]` | `<f4` |
| `reconstruction.layers.2.weight.npy` | `[256,256]` | `<f4` |
| `reconstruction.layers.4.bias.npy` | `[149]` | `<f4` |
| `reconstruction.layers.4.weight.npy` | `[149,256]` | `<f4` |
| `static.layers.0.bias.npy` | `[128]` | `<f4` |
| `static.layers.0.weight.npy` | `[128,298]` | `<f4` |
| `static.layers.2.bias.npy` | `[32]` | `<f4` |
| `static.layers.2.weight.npy` | `[32,128]` | `<f4` |

The table is the complete current `TempoRACTeacher().state_dict()` inventory,
not a subset or a suggested export. `teacher_sha256` means only SHA-256 of this
complete exact checkpoint artifact. It is never a caller label.

## 5. Closed checkpoint receipt

The checkpoint receipt schema is
`temporac.teacher-checkpoint-receipt.v4`. It has exactly these 12 keys:

1. `artifact_bytes`
2. `artifact_sha256`
3. `code_sha256`
4. `config_sha256`
5. `contract_sha256`
6. `environment_sha256`
7. `job_name_hex`
8. `members`
9. `run_receipt_sha256`
10. `schema`
11. `seed`
12. `step`

`artifact_bytes` is the positive exact NPZ byte count. `artifact_sha256` is
the hash of those bytes. `contract_sha256` equals the canonical proposal hash.
`code_sha256`, `config_sha256`, and `environment_sha256` equal the corresponding
closed values in the exact successful teacher run receipt. `run_receipt_sha256`
is the hash of that exact run receipt. `job_name_hex` decodes to exact ASCII
`temporac.execution.v4/teacher/seed=<seed>`. `seed` is one of
`20260815,20260816,20260817`; `step` is exactly `500,1000,...,20000`.

`members` is a 16-row array in the NPZ member order above. Each row has exactly
the existing five member-ledger keys `bytes`, `dtype`, `name`, `sha256`, and
`shape`. The ledger is recomputed from the 16 NPY member bytes. The exact
successful run receipt must list `artifact_sha256` at zero-based position
`step/500-1` of its 40-element `ordered_checkpoint_sha256`; otherwise the
checkpoint receipt fails.

At replay, `code_sha256` and `environment_sha256` must also equal the existing
G5a code/environment commitments. `config_sha256` remains the exact selected
teacher-run configuration commitment; no evaluator-supplied replacement is
accepted.

## 6. Closed 120-row score index and selection receipt

### 6.1 Score index

`temporac.teacher-selection-score-index.v4` is a top-level canonical JSON array
of exactly 120 rows ordered by numeric `(seed,step)`: the three frozen seeds in
ascending order, each with all 40 steps `500,1000,...,20000`. Each row has
exactly these seven keys:

1. `checkpoint_receipt_sha256`
2. `mean_masked_reconstruction`
3. `negative_edge_fraction`
4. `seed`
5. `step`
6. `tune_abstentions`
7. `tune_objective`

The six score fields are exactly the six non-`payload` fields of the current
`TeacherCheckpointScore`. The three score-valued fields are finite binary64;
`negative_edge_fraction` is in `[0,1]`; `tune_abstentions` is a nonnegative
integer. Each `checkpoint_receipt_sha256` names the exact closed checkpoint
receipt for the same seed and step. Duplicate/missing seed-step rows, incomplete
40-step inventories, receipt mismatch, or an extra row fail globally.

### 6.2 Selection receipt

`temporac.teacher-selection-receipt.v4` has exactly these eight keys:

1. `code_sha256`
2. `contract_sha256`
3. `per_seed_winner_checkpoint_receipt_sha256`
4. `schema`
5. `score_index_bytes`
6. `score_index_sha256`
7. `selected_checkpoint_artifact_sha256`
8. `selected_checkpoint_receipt_sha256`

`score_index_bytes` is the positive byte count of the exact 120-row canonical
JSON index, and `score_index_sha256` hashes those bytes. `code_sha256` binds the
selection/verifier code root and must equal the G5a code commitment.
`per_seed_winner_checkpoint_receipt_sha256` is an exact three-element array in
seed order `20260815,20260816,20260817`. The selected receipt and selected
artifact fields bind the unique global winner.

The verifier reloads all 120 score rows and their checkpoint receipts. Within
each seed it discards rows whose `negative_edge_fraction>0.01`, then minimizes
`(tune_objective,step)`. It fails if any seed has no remaining row. Across the
three winners it minimizes
`(tune_abstentions,mean_masked_reconstruction,tune_objective,seed)`. The three
computed receipt digests and the global receipt/artifact digest must equal the
selection receipt byte for byte. Caller-provided winners are not trusted.

## 7. Typed selected teacher and deterministic loader

A `SelectedTeacher` value may be constructed only by the closed verifier from:

- exact 120-row score-index bytes;
- exact selection-receipt bytes;
- all 120 exact checkpoint-receipt bytes and the three exact run-receipt bytes;
- the exact globally selected checkpoint artifact bytes.

Construction performs every hash, schema, inventory, run binding, and
selection check in Sections 4--6. It then instantiates a fresh current
`TempoRACTeacher` on CPU, rejects any state key/shape/dtype not in the 16-member
table, converts no alternate dtype, uses strict state loading, and serializes
the loaded state again through the same deterministic NPZ writer. Reserialized
bytes must equal the input checkpoint artifact bytes exactly. The module is put
in `eval()` mode and exposes its digest only as the hash recomputed from those
exact bytes.

No public natural-certificate API may accept `selected_teacher_sha256`, raw
`phase`, or raw `reconstruction`. Possession of a digest, an untyped module, a
state dictionary, or previously computed arrays cannot construct a
`SelectedTeacher` or `NaturalTeacherOutput`.

## 8. Noncircular canonical natural inference

For one natural identity, the sole allowed order is:

1. Load and validate the typed feature record against exact feature artifact
   and feature-receipt bytes.
2. Run canonical `preprocess_identity` once. Teacher input is exactly the
   resulting C-order `<f4[T,215]`. Continuous teacher values are its first 149
   columns converted once to C-order `<f8[T,149]`; geometry, source clocks,
   masks, and binary64 edge lengths are the same canonical preprocessing
   results. Preprocessing edge-run `[a,b)` is converted to sample-run
   `[a,b+1)` exactly as in the bound certificate adapter.
3. For each sample-run in run order, call the geometry-only two-stage
   `integer_landmarks` rule. It consumes no teacher output. If any run has
   fewer than three retained landmarks, stop immediately with the sole ordered
   reason vector `[11]` (`LANDMARK`); do not run the teacher and do not create a
   natural-teacher output/input receipt, target artifact, or target receipt.
4. Within each run, consecutive landmarks in landmark order form the half-open
   traversal bounds `[L_j,L_(j+1))`. Concatenate bounds in run order. A bound
   never crosses a run and no nonconsecutive landmark pair is added.
5. Compute F5 from continuous values, binary64 edge lengths, and those exact
   traversal bounds. Within a traversal, base-edge subedges are visited from
   left to right. Every denominator and coordinate numerator uses binary64
   Neumaier compensated summation in that order. For each term `x`, update
   `t=s+x`, then add `(s-t)+x` to compensation when `abs(s)>=abs(x)`, otherwise
   add `(x-t)+s`, and return `s+compensation`. The F5 mean term evaluates
   `lambda*((A+B)/2)` and the second-moment term evaluates
   `lambda*(((A*A+A*B)+B*B)/3)` in the written binary64 operation order.
   Traversal means are then Neumaier-summed in traversal order and divided by
   the integer traversal count. Finally compute
   `s=sqrt(max(bar_q-bar_m*bar_m,+0.0))` coordinatewise and concatenate
   `[bar_m,s]` as exact C-order `<f8[298]`. Ordinary `np.sum`, reordered or
   pairwise reductions, edge weighting across traversals, or a teacher-derived
   traversal is nonconforming.
6. Cast the complete static vector exactly once, in C order, to `<f4[298]`.
   Run exactly one forward call of the verified selected teacher with that
   static input and the exact `<f4[T,215]` teacher input. Inference is CPU-only
   float32, single-threaded, `TempoRACTeacher.eval()` plus
   `torch.inference_mode()`, with autocast disabled and CUDA/cuDNN matmul TF32
   flags false. GPU execution, AMP, bfloat16, float64 model execution, training
   mode, stochastic layers, or caller-supplied outputs fail.
7. Phase and reconstruction are computed as float32 model outputs. After shape
   and finiteness validation, each complete CPU C-order output is cast exactly
   once to `<f8[T,2]` and `<f8[T,149]`, respectively. Hashes are over those
   exact C-order float64 bytes.

The natural input receipt remains the closed
`temporac.natural-certificate-input.v4` object with exactly its existing 15
keys: `continuous_mask_sha256`, `continuous_sha256`,
`feature_artifact_sha256`, `feature_receipt_sha256`, `geometry_sha256`,
`opaque_key_hex`, `phase_sha256`, `reconstruction_sha256`,
`run_bounds_sha256`, `sampled_source_clock_sha256`, `schema`, `slot`,
`source_binding_sha256`, `teacher_input_sha256`, and `teacher_sha256`. It is
built only after the canonical forward call. Its `teacher_sha256` is the exact
selected checkpoint artifact digest derived by `SelectedTeacher`; its output
hashes are the exact post-cast float64 hashes. The static vector and traversal
bounds are independently rederived from the feature rather than trusted from
the receipt.

The only conforming natural builder interface accepts the typed feature and
its exact receipt plus a verified `SelectedTeacher`. It has no digest, phase,
or reconstruction parameters. An independent verifier must be able to rerun
this complete procedure and reproduce the natural input receipt and both
output arrays byte for byte.

## 9. Certified-development surrounding index

`temporac.certified-development-index.v4` is a canonical JSON object with
exactly three top-level keys: `contract_sha256`, `rows`, and `schema`.
`contract_sha256` equals the proposal digest and `schema` is the literal schema
name. `rows` is strictly ordered by `(opaque_key_hex,slot)`, contains no
duplicate identity, and contains every and only the frozen K1/G5a certified
development identities.

Each row has exactly these 11 keys:

1. `component_key_hex`
2. `feature_artifact_sha256`
3. `feature_receipt_sha256`
4. `natural_input_receipt_sha256`
5. `opaque_key_hex`
6. `selection_receipt_sha256`
7. `selected_checkpoint_artifact_sha256`
8. `selected_checkpoint_receipt_sha256`
9. `slot`
10. `target_artifact_sha256`
11. `target_receipt_sha256`

`opaque_key_hex` and `component_key_hex` are lowercase 32-byte hex values;
`slot` is the exact nonnegative development local-person slot. Every digest
names exact bytes supplied to K7. All rows bind the one exact selection receipt
and its selected checkpoint artifact/receipt. Feature and target receipts must
validate against their named artifacts and identity. The natural input receipt
must validate against that exact feature and selection. Target source kind is
natural, target key/slot equal the row, and all teacher digests equal the row's
selected checkpoint artifact digest. `component_key_hex` must equal the exact
component key already assigned to that identity by the frozen population
manifest; it is not a new or caller-selectable grouping.

## 10. Existing G5a receipt root commits the new index

No key is added to `temporac.g5a-receipt.v4`. Its existing
`receipt_root_sha256` is the SHA-256 of one canonical
`temporac.g5a-receipt-root-index.v4` JSON object with exactly these five keys:

1. `certified_development_index_bytes`
2. `certified_development_index_sha256`
3. `contract_sha256`
4. `receipt_sha256`
5. `schema`

The first two fields bind the complete exact
`temporac.certified-development-index.v4` bytes. `receipt_sha256` is an array
containing one digest for every receipt artifact in the already frozen G5a
receipt inventory, including all feature, target, natural-input, teacher
checkpoint, teacher selection, run, prediction, and required fixture receipts.
It is sorted by decoded raw 32-byte digest; multiplicity is retained when two
inventory entries have identical bytes. Completeness and multiplicity are
checked against the frozen population, prediction, checkpoint, target, plan,
and surrounding scope indexes. The certified-development index itself is not
mislabelled as a receipt and is committed by its two distinguished fields.

Thus the existing closed G5a field commits both the exact index and every
receipt it dereferences without changing the G5a or target receipt key set.
Missing root-preimage bytes, a changed index, a missing referenced receipt, or
an inventory multiplicity mismatch fails before capability consumption.

## 11. K7 bytewise rederivation

Before vault unit checks or any metric calculation, K7 performs the following
for every row of the committed certified-development index:

1. verify the root-index bytes against G5a `receipt_root_sha256` and verify the
   certified-development index bytes/hash;
2. load and validate exact feature artifact/receipt bytes;
3. reload the 120-row score index, all checkpoint receipts/run receipts, exact
   selection receipt, and selected checkpoint artifact/receipt; reconstruct
   the typed `SelectedTeacher`, require its code/environment bindings to equal
   G5a, and require all index rows to name that winner;
4. rerun Section 8 natural inference and require the rebuilt natural input
   receipt bytes to equal the indexed natural-input receipt bytes;
5. bind the canonical certificate track, rerun `certify_target`, and require
   `CERTIFIED` rather than abstention;
6. rebuild the existing seven-member target NPZ with exactly `chi`,
   `contract_sha256`, `edge_mask`, `pulse`, `source_kind`, `target_mask`, and
   `teacher_sha256`, then rebuild the unchanged ten-key
   `temporac.target-receipt.v4`;
7. require both rebuilt target NPZ bytes and rebuilt target receipt bytes to be
   bytewise identical to the indexed target artifact and receipt.

No cached phase, reconstruction, target arrays, digest label, or typed object
supplied by the caller substitutes for replay. A same-identity target produced
from a different feature, natural input, teacher, or selection fails even when
its ten-key target receipt is otherwise well-formed.

## 12. Exact population and evidence equalities

Define `P402` as the ordered identity projection of the frozen population
manifest: exactly 268 train plus 134 development `(opaque_key_hex,slot)` pairs.
The following are mandatory before capability consumption:

- the validated vault identity projection equals `P402` exactly;
- the prediction identity projection equals `P402` exactly, not merely the
  identities that happen to appear in submitted envelopes;
- prediction envelopes are exactly the full Cartesian product of `P402` with
  the four frozen arms, three frozen seeds, and two frozen conditions: one and
  only one envelope per combination and exactly `402*4*3*2=9,648` total;
- no extra, missing, duplicate, relabelled, or subset identity/envelope is
  accepted;
- G5b `join_cardinality` is computed from the verified equality of population
  and vault identities and must equal 402; a caller-provided cardinality is not
  an input.

Define `Cdev` as the ordered identity projection frozen at K1 and committed by
the certified-development index at G5a. Then:

- `Cdev` is a subset of the exact 134 development identities, not of train and
  not an arbitrary later subset;
- `|Cdev|>=108`, and the projection of its indexed
  `component_key_hex` values contains at least eight unique development
  components;
- the certified-development index identities, target-count rows, actual typed
  `CertifiedDevelopmentTarget` evidence, and K1/G5a committed
  development-certified identity sequence all equal `Cdev` exactly;
- each target-count value equals the sum of pulse bits in the bytewise-verified
  target for that same identity, and the existing target-count root is
  recomputed over exactly those rows;
- no post-K1 filtering or post-G5a subset is allowed.

This amendment does not require all 134 development identities, much less all
402 population identities, to certify. Stubs remain in the 9,648 prediction
population exactly as the proposal requires.

## 13. Direct P05M builder/consumer composition

`count_metric_fixture_receipt(...)` may continue to return an immutable
`Mapping`, including `MappingProxyType`. Every consumer that hashes or serializes
that receipt must accept the builder's return value unchanged. The consumer
first calls the closed mapping validator, then internally materializes a plain
dictionary by reading the six exact validated keys, and only then calls
`canonical_json_bytes` on that internal dictionary. The caller is never
required to call `dict()` or otherwise adapt the builder result.

The required direct-composition invariant is:

`consumer(count_metric_fixture_receipt(...))`

with no intermediate conversion. The internally materialized canonical bytes
must equal the bytes obtained from the same six key/value pairs in a normal
dictionary. Unknown/missing keys or invalid values still fail before hashing.

## 14. Round-2 closure and required fresh tests

| Finding | Normative closure | Required negative/positive evidence |
|---|---|---|
| `R2-F1` | exact checkpoint bytes, closed receipts, complete selection replay, typed selected teacher, and actual canonical CPU inference | reject digest relabelling, raw phase/reconstruction injection, incomplete 120-row index, altered checkpoint bytes/member metadata, non-strict load, and noncanonical inference; positive replay reproduces output receipt bytes |
| `R2-F2` | committed certified-development index plus K7 reconstruction and target NPZ/receipt byte equality; target schema unchanged | same identity/teacher but different feature or natural-input target swap must fail; exact replay must pass |
| `R2-F3` | exact `P402`, 9,648 Cartesian envelopes, exact `Cdev` evidence/count/index equality, and computed G5b cardinality | prediction subset, certified-evidence subset, target-count subset, population/vault mismatch, duplicate, or extra identity must fail before capability consumption |
| `R2-F4` | consumer validates any `Mapping` and internally canonicalizes a plain dictionary | the direct builder return value must serialize and consume without caller adaptation; malformed mapping still fails |

Fresh review must verify both specification mirrors, then a later implementation
review must bind code/tests to these exact bytes. Passing tests cannot itself
set P2, P3, S0, capability, launch, data, training, result, or claim authority.

## 15. Final status

This amendment is `PROPOSED_PENDING_FRESH_REVIEW`, same-family provisional, and
non-authorizing. It closes no gate by its existence. It creates no checkpoint,
selection, feature, target, prediction, vault, evaluator, test, sealed,
heldout, result, server, or training artifact. It changes no job, threshold,
claim, dataset, cache scope, or target/G5a receipt key set.
