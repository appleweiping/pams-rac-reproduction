# TempoRAC Contract Amendment 004 — Operator Manifest Serialization

**Status:** `PROPOSED_PENDING_FRESH_REVIEW`

**Method anchor:** TempoRAC remains identity-indexed, window-local soft tempo
routing over shared response experts with positive NOLA and one final
threshold-0.5 connected-component decode per supplied identity. This amendment
does not change that method, any scientific threshold, any fixture population,
or any paper claim.

**Scope:** close the serialization gap left after accepted Amendment 002. The
accepted row-key, response, NOLA, decoder, and association mathematics stay
byte-for-byte unchanged. This amendment defines only the 10,368-row operator
manifest, its per-array hash preimages, its runtime lock, and the non-authorizing
candidate receipt required for a later independent implementation review.

## 1. Why a separate amendment is required

The canonical proposal fixes the Cartesian population, exact row bytes, raw
witness roots, and states that truth arrays and expected outputs are hashed in a
canonical 10,368-row manifest. It does not fix the manifest row keys, JSON
container, per-array byte preimages, aggregate preimage, or runtime-lock schema.
Multiple honest encodings therefore remain possible. The current generated
artifact is correctly labelled `candidate-only`; neither its successful replay
nor this amendment is P2, S0, launch, data, server, training, or claim authority.

## 2. Frozen scientific and numerical non-changes

The following remain exactly as in the proposal plus accepted Amendment 002:

- row order is the Cartesian product
  `gaps -> widths -> amplitudes -> offsets -> layouts`, with
  `gaps=(4,5,8,16,32,64,127,128,129)`, `widths=(1,2,4)`,
  `amplitudes=(0.60,0.75,1.00)`, `offsets=0...31`, and
  `layouts=(interior,left-boundary,right-boundary,run-reset)`
- there are exactly `9*3*3*32*4=10,368` unique 10-byte keys
- `P(e)`, `D[0]`, exact inactive float32 cap, plateau values, run/reset masks,
  positive NOLA, one decoder call, and half-open association are unchanged
- normative roots remain:
  - semantic rows: `0fe8b2619a3f33cc4a83485053888c04c567b015eea65f81bf6cd9ec715ee4f2`
  - inactive lookup: `afa6eff0d744acfdf9cf78adc43aa8daf01633f3345d7567d642a7782885040d`
  - key inventory: `18dfdafe237b77565de36bab6d4642efb314ec1cbbd76e852938ed5c4d424a55`
  - edge-zero digest inventory: `ba797f6ae59cbbb40f19a61199afaaa1ac5b2c001a3c9fc34f1dba8f79cf78e4`
- the proposal still allocates one primary claim, exactly three paper-visible
  evidence blocks, twenty-seven training jobs, three seeds, and 93 A6000-hours;
  deterministic audit fixtures remain non-paper-visible and never become a
  fourth block

Any implementation that changes one of these values is a new method candidate,
not an implementation of this amendment.

## 3. Canonical JSON bytes

Every JSON object defined here uses UTF-8, `ensure_ascii=false`,
`allow_nan=false`, lexicographically sorted object keys, separators `,` and `:`
with no insignificant whitespace, LF line endings, and exactly one terminal LF.
Integers use their shortest base-10 form. The only non-integer JSON numbers in
the manifest are amplitude values and must appear literally as `0.6`, `0.75`,
or `1.0`. Hash strings are exactly 64 lowercase hexadecimal characters.

The manifest is a top-level JSON array, not an object and not JSONL. It contains
exactly 10,368 row objects in the frozen Cartesian order. Its complete canonical
LF-terminated byte sequence is the manifest hash preimage; there is no filename,
length prefix, wrapper object, self hash, compression, or alternate aggregate.

## 4. Closed manifest row schema

Every row has exactly these 23 keys and no others:

1. `amplitude`
2. `association_sha256`
3. `component_location_sha256`
4. `component_score_sha256`
5. `decoded_components_sha256`
6. `decoder_mask_sha256`
7. `edge_count`
8. `edge_mask_sha256`
9. `expected_components_sha256`
10. `gap`
11. `generated_response_sha256`
12. `key_hex`
13. `layout`
14. `nola_quantized_response_sha256`
15. `offset`
16. `reset_edge`
17. `run_bounds_sha256`
18. `target_mask_sha256`
19. `truth_negative_components`
20. `truth_plateau_mask_sha256`
21. `truth_plateaus_sha256`
22. `width`
23. `windows_sha256`

`gap`, `width`, `offset`, `edge_count`, and `truth_negative_components` are
nonnegative JSON integers with the proposal-defined values. `layout` is exactly
one frozen layout string. `key_hex` is exactly 20 lowercase hex characters and
decodes to the proposal-defined 10-byte key for the row. `reset_edge` is `null`
outside `run-reset` and otherwise is the proposal-defined nonnegative reset-edge
integer. Every `*_sha256` field is SHA-256 over one raw C-order array preimage
defined below. The generator must recompute all fields; copying candidate rows
without recomputation fails.

## 5. Per-array hash preimages and exact shapes

There is no `.npy` header, JSON framing, shape prefix, row-key prefix, or terminal
LF in a per-array hash. Each hash preimage is exactly the raw C-order bytes of the
fully materialized row array after conversion to the following explicit dtype:

| Manifest field | Array | Exact dtype |
|---|---|---|
| `association_sha256` | truth/component bipartite adjacency | `|u1` |
| `component_location_sha256` | decoded component locations | `<i4` |
| `component_score_sha256` | decoded component scores | `<f4` |
| `decoded_components_sha256` | decoded half-open components | `<i4` |
| `decoder_mask_sha256` | decoder edge mask | `|u1` |
| `edge_mask_sha256` | valid edge mask | `|u1` |
| `expected_components_sha256` | reference half-open components | `<i4` |
| `generated_response_sha256` | generated edge response | `<f4` |
| `nola_quantized_response_sha256` | binary64 NOLA output rounded once to `<f4` | `<f4` |
| `run_bounds_sha256` | ordered half-open run bounds | `<i4` |
| `target_mask_sha256` | target edge mask | `|u1` |
| `truth_plateau_mask_sha256` | truth plateau edge mask | `|u1` |
| `truth_plateaus_sha256` | truth half-open plateau intervals | `<i4` |
| `windows_sha256` | ordered half-open window bounds | `<i4` |

For one row let `E` be `edge_count`, `R=1` except `R=2` for `run-reset`,
`K=3` be the truth count, and `C=3` be the decoded-component count.  If a run
has edge length `L_r`, its window count is
`W_r=1+ceil(max(L_r-127,0)/32)` and `W=sum_r W_r`.  These symbols are derived
from the row before any array is hashed.  The exact shapes and axes are:

| Manifest field | Exact shape | Axis semantics |
|---|---|---|
| `association_sha256` | `[K,C]=[3,3]` | row axis is truth index; column axis is decoded-component index |
| `component_location_sha256` | `[C]=[3]` | one lower-middle edge location per decoded component |
| `component_score_sha256` | `[C]=[3]` | one maximum response per decoded component |
| `decoded_components_sha256` | `[C,2]=[3,2]` | rows are decoded half-open `[start,stop)` components |
| `decoder_mask_sha256` | `[E]` | edge axis |
| `edge_mask_sha256` | `[E]` | edge axis |
| `expected_components_sha256` | `[K,2]=[3,2]` | rows are reference half-open `[start,stop)` components |
| `generated_response_sha256` | `[E]` | edge axis |
| `nola_quantized_response_sha256` | `[E]` | edge axis |
| `run_bounds_sha256` | `[R,2]` | rows are ordered half-open edge runs |
| `target_mask_sha256` | `[E]` | edge axis |
| `truth_plateau_mask_sha256` | `[E]` | edge axis |
| `truth_plateaus_sha256` | `[K,2]=[3,2]` | rows are truth half-open `[start,stop)` plateaus |
| `windows_sha256` | `[W,2]` | rows are ordered half-open NOLA windows |

Vectors are never encoded as column vectors.  `association[t,c]=1` iff truth
plateau `t` and decoded component `c` intersect as half-open edge sets.  Every
interval table has exactly two columns in `[start,stop)` order.  All arrays must
match the exact dtype, exact shape, stated axis ownership, finiteness rule, and
C contiguity before hashing. Integer narrowing, native-endian aliases, Fortran
order, shape-only hashes, transposed association arrays, and hashes over Python
object representations are forbidden. `generated_response` and
`nola_quantized_response` must be byte-identical for every row under this
synthetic bank, but both fields remain present and independently recomputed.

## 6. Closed schema and replay-witness envelopes

`operator_manifest_schema.json` has schema
`temporac.operator-manifest-schema.v2` and exactly these top-level keys:
`array_specs`, `canonical_json`, `cartesian_order`, `manifest_filename`,
`manifest_row_count`, `manifest_row_keys`, `normative_roots`,
`per_array_preimage`, `schema`, and `shape_symbols`.  `array_specs` is an array
in the Section 5 table order; each item has exactly `axis_semantics`, `dtype`,
`field`, and `shape`, with all values exact strings from Section 5.
`canonical_json` has exactly `allow_nan`, `bom_allowed`,
`duplicate_keys_allowed`, `encoding`, `ensure_ascii`, `manifest_top_level`,
`object_key_order`, `separators`, and `terminal_lf_count`.
`bom_allowed=false` and `duplicate_keys_allowed=false`. `shape_symbols` has
exactly `C`, `E`, `K`, `R`, `W`, and
`window_count_formula`.  `normative_roots` has exactly the four root names in
Section 2.  No authority, prose, self hash, extra key, duplicate key, or BOM is
allowed.

There is no free wording in this member. `array_specs` is constructed in the
14-row Section 5 order by copying the table's manifest field, dtype, exact-shape
string, and axis-semantics string verbatim. `canonical_json` is the exact object
defined in Section 3. `cartesian_order` is exactly the five-object array
`[{field:gap,values:[4,5,8,16,32,64,127,128,129]},
{field:width,values:[1,2,4]},
{field:amplitude,values:[0.6,0.75,1.0]},
{field:offset,values:[0,...,31]},
{field:layout,values:[interior,left-boundary,right-boundary,run-reset]}]`, where
the displayed object keys and string values are JSON strings and the ellipsis is
the literal integer sequence 0 through 31, not an encoded token.
`manifest_filename` is exact `temporac.operator-manifest.v4.json`,
`manifest_row_count` is integer 10,368, and `manifest_row_keys` is the exact
23-string Section 4 list. `normative_roots` is the exact four-key mapping from
the companion Amendment JSON, including the `_sha256` suffixes and fixed
digests. `per_array_preimage` is exact string
`raw C-order bytes only after exact dtype, shape, finiteness, and C-contiguity validation; no header, framing, shape, row key, or LF`.
`shape_symbols` is the exact companion-JSON object, including its literal
formula strings. These construction rules, followed by Section 3 canonical
serialization, define one byte sequence; a paraphrase is nonconforming.

`replay_witness.json` has schema `temporac.operator-replay-witness.v2` and
exactly these top-level keys: `actual_inactive_b255`, `aggregate_roots`,
`candidate_manifest_bytes`, `candidate_manifest_sha256`, `candidate_status`,
`normative_roots`, `row_witnesses`, `schema`, and `validated_row_count`.
`candidate_status` is exact
`FROZEN_CANDIDATE_PENDING_FRESH_IMPLEMENTATION_REVIEW` and the row count is
10,368. `actual_inactive_b255` has exactly `edge_count`, `first`, and
`row_count`; `first` has exactly `edge`, `key_hex`, and
`response_f4_le_hex`. `normative_roots` has exactly `edge0_digest_inventory`,
`inactive_lookup`, `key_inventory`, and `semantic_inventory`; each value has
exactly integer `bytes` and lowercase-hex `sha256`.

The fixed values are not left to prose: `candidate_manifest_bytes=15039512`,
`candidate_manifest_sha256=1f39378d6f7a970b1f183558ed145bd97107f2aeda41902f4736e6778930e1bb`,
and `actual_inactive_b255` is exactly
`{edge_count:8945,first:{edge:42,key_hex:00040001000000000001,response_f4_le_hex:cccccc3e},row_count:5665}` with JSON
strings around the two hex values. The four normative-root records, nine
aggregate-root records, and five row-witness objects are the exact objects under
`replay_witness.expected_*` in the companion Amendment JSON. Integer fields are
nonnegative JSON integers; every `sha256` or `digest_hex` is 64 lowercase hex;
every `key_hex` is 20 lowercase hex; and each float32 little-endian hex string is
exactly eight lowercase hex characters. No value may be `null`.

`aggregate_roots` has exactly these nine keys in canonical object order:
`association`, `decoded_components`, `decoder_masks`, `edge_masks`,
`nola_responses`, `responses`, `target_masks`, `truth_masks`, and
`truth_plateaus`. Each value has exactly integer `bytes` and lowercase-hex
`sha256`. Each aggregate preimage is the direct concatenation of the named raw
per-row array bytes in frozen Cartesian order, with no separator, length, key,
or digest framing. `decoded_components` maps to the
`decoded_components_sha256` row array, not `expected_components`.

`row_witnesses` is an array of exactly five objects in the accepted Amendment
002 witness order. Each object has exactly `actual_f4_le_hex`, `digest_hex`,
`edge`, `inactive_function_f4_le_hex`, and `key_hex`. Extra roots, prose, or
diagnostics are forbidden in the canonical witness; they may appear only in a
detached non-normative file that is not a candidate member. Aggregate roots are
diagnostic commitments and never replace the 10,368 closed manifest rows.

## 7. Locked offline runtime schema

The runtime lock is canonical JSON schema `temporac.operator-runtime-lock.v3`
with exactly these keys: `architecture`, `base_image`, `byteorder`,
`container_policy`, `numpy`, `platform_system`, `python`, and `schema`.
`environment_lock.json` is exactly the Section 3 canonical serialization of the
companion Amendment JSON's `runtime_lock` object; no regenerated prose or host
metadata may be added. Nested objects are closed as follows.

`base_image` has exactly:

- `pull_reference`:
  `docker.io/library/python@sha256:e8be0ea148390d08bc077840cf87ac6a538d80b0ea1e8752b3e3982987cd0a53`
- `index_digest`:
  `sha256:e8be0ea148390d08bc077840cf87ac6a538d80b0ea1e8752b3e3982987cd0a53`
- `linux_amd64_manifest_digest`:
  `sha256:0c7cf9e198201da2e838fcb61c07717dfc96adb3d1200599f54dea0af1d153bf`
- `linux_amd64_manifest_bytes`: `2516`
- `config_digest`:
  `sha256:c7ed3a18aaaa5a8b439cf84138ebcfe1e82e64d091781f3384ca38827824aefc`
- `config_bytes`: `7096`

`numpy` has exactly `filename`, `files_url`, `init_sha256`,
`record_sha256`, `version`, `wheel_bytes`, and `wheel_sha256`. Values are:

- filename
  `numpy-2.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
- files URL
  `https://files.pythonhosted.org/packages/48/3e/bf807eb050abc23adc556f34fcf931ca2d67ad8dfc9c17fcd9332c01347f/numpy-2.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
- wheel bytes `16040181`
- wheel SHA-256
  `24003ba8ff22ea29a8c306e61d316ac74111cebf942afbf692df65509a05f111`
- imported `__init__.py` SHA-256
  `39c42db027548f958e096e8babe3fa0e3e773d24aa39eb6363fc0e3abbec34b1`
- installed `RECORD` SHA-256
  `65cfe9b7bbac44a46116b2b5f8282f32ec57e1aa9faeb64d523604c712b446d8`

`python` has exactly `executable_sha256`, `implementation`, and `version`,
with values `9a6988f011f466ae829f7945c54000a564e14a1a594cfb7079c02eda9b87620f`,
`CPython`, and `3.12.4`. Top-level `architecture`, `byteorder`, and
`platform_system` are exactly `x86_64`, `little`, and `Linux`.

`container_policy` has exactly `bootstrap_script_sha256`, `bootstrap_steps`,
`docker_argv_template`, `environment`, `mounts`, `network`, `output_rule`,
`root_filesystem`, `source_snapshot_policy`, `user_site`, and `wheel_install`.
Network is
`none`; root filesystem is `read-only`; user site is `disabled`; the wheel is
downloaded before replay, verified by bytes and SHA-256, mounted read-only, and
installed with `python -m pip install --no-index --no-deps` into a fresh venv on
an executable tmpfs. The repository snapshot is mounted read-only, the single
new absent output directory is the only writable bind, and the receipt is
written last. The environment object has exactly `HOME=/nonexistent`,
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONHASHSEED=0`,
`PYTHONNOUSERSITE=1`, `PYTHONPATH=/work/src`, and
`TEMPORAC_OPERATOR_DOCKER_IMAGE` equal to the pull reference. No other host
environment variable, user-site directory, mutable package index, editable
install, network fetch, or writable source mount is permitted.

`docker_argv_template` is the exact ordered argv below; braces name launcher
tokens whose resolved absolute host paths are recorded in the independent
process witness and are never hashed as portable candidate content:

`[docker,run,--rm,--platform,linux/amd64,--network,none,--read-only,--tmpfs,/tmp:rw,exec,nosuid,nodev,size=512m,--mount,type=bind,src={REPO_SNAPSHOT},dst=/work,readonly,--mount,type=bind,src={WHEEL_FILE},dst=/wheel/numpy.whl,readonly,--mount,type=bind,src={OUTPUT_PARENT},dst=/out,--workdir,/work,-e,HOME=/nonexistent,-e,PYTHONDONTWRITEBYTECODE=1,-e,PYTHONHASHSEED=0,-e,PYTHONNOUSERSITE=1,-e,PYTHONPATH=/work/src,-e,TEMPORAC_OPERATOR_DOCKER_IMAGE=docker.io/library/python@sha256:e8be0ea148390d08bc077840cf87ac6a538d80b0ea1e8752b3e3982987cd0a53,docker.io/library/python@sha256:e8be0ea148390d08bc077840cf87ac6a538d80b0ea1e8752b3e3982987cd0a53,/bin/sh,-ceu,{BOOTSTRAP_SCRIPT}]`.

`REPO_SNAPSHOT` is the reviewed allowlist-only 24-file source snapshot defined
in Section 8, `WHEEL_FILE` is the verified exact wheel, and `OUTPUT_PARENT` is
an otherwise empty writable directory whose child
`operator_candidate_v2_20260816` does not exist.
`bootstrap_steps` is the exact ordered array:

1. verify `/wheel/numpy.whl` byte count equals `16040181`
2. verify its SHA-256 equals
   `24003ba8ff22ea29a8c306e61d316ac74111cebf942afbf692df65509a05f111`
3. run `/usr/local/bin/python -m venv /tmp/venv`
4. run `/tmp/venv/bin/python -m pip install --no-index --no-deps /wheel/numpy.whl`
5. run `/tmp/venv/bin/python scripts/experiments/generate_temporac_operator_manifest_v4.py --output /out/operator_candidate_v2_20260816`

`BOOTSTRAP_SCRIPT` is the five exact step strings joined by one LF with exactly
one terminal LF; its 409-byte SHA-256 is
`66fdf1cd03616759948e326e1714a997848c45f40d005240a4ebe170240bf323`.
`/bin/sh -ceu` supplies the error/unset-variable policy. No bootstrap command may be
added, reordered, or silently sourced from a host shell profile.

Generation and independent replay must first resolve the pull reference to the
stated linux/amd64 child manifest/config, then verify the invoked executable,
wheel, imported NumPy file, and distribution record against the lock. A tag-only
or bare-digest-only environment file is historical evidence and cannot satisfy
this amendment.

## 8. Candidate artifact and receipt

The generator writes into a new, absent directory and uses exclusive creation
for every member. A pre-existing member or retry fails before regeneration and
must leave all existing hashes and mtimes unchanged. It writes manifest, schema,
runtime lock, and replay witness first and writes the receipt last; a missing
receipt is always an incomplete non-candidate tree. The candidate contains:

- `temporac.operator-manifest.v4.json`
- `operator_manifest_schema.json`
- `environment_lock.json`
- `replay_witness.json`
- `candidate_receipt.json`

Before the source-dependent receipt is written, the four non-receipt members
have these normative canonical byte commitments:

- `temporac.operator-manifest.v4.json`: 15,039,512 bytes,
  `1f39378d6f7a970b1f183558ed145bd97107f2aeda41902f4736e6778930e1bb`
- `operator_manifest_schema.json`: 3,607 bytes,
  `3b29937c641274e73d10afd19bea943f28288b3d2851e8a5bf2313d7ea3e0fc3`
- `environment_lock.json`: 3,732 bytes,
  `8008df73cc9a807ac5e5262aebc170ae6e01c82b07658575f6207ec7cb0a14fd`
- `replay_witness.json`: 2,896 bytes,
  `348254cfe64cea35d6e29861755489721ebbbd4031f3ec1c044d7fa24fe38010`

Any byte mismatch is a regeneration failure, not an alternate conforming
serialization.

`operator_manifest_schema.json` is the exact closed machine-readable object
defined in Section 6 and has schema `temporac.operator-manifest-schema.v2`. It contains no
`authoritative` boolean and no self hash. Authority comes only from a later fresh
review receipt that binds its external SHA-256.

`candidate_receipt.json` has schema
`temporac.operator-manifest-candidate-receipt.v2`, status
`FROZEN_CANDIDATE_PENDING_FRESH_IMPLEMENTATION_REVIEW`, and exactly these keys:

- `authority`
- `bindings`
- `blockers`
- `candidate_status`
- `members`
- `p2_status`
- `s0_status`
- `schema`
- `validated_row_count`

`authority` is a closed object whose values for `P2`, `S0`, `launch`, `server`,
`data`, `training`, `gate`, `git`, and `paper_claim` are all `false`.

`bindings` is an object with exactly `amendment_002_json_sha256`,
`amendment_002_md_sha256`, `amendment_002_review_json_sha256`,
`amendment_002_review_md_sha256`, `amendment_004_json_sha256`,
`amendment_004_md_sha256`, `amendment_004_review_json_sha256`,
`amendment_004_review_md_sha256`, `generation_project_files`,
`proposal_sha256`, `review_project_files`, `runtime_lock_sha256`, and
`source_snapshot_sha256`.
The two Amendment 004 review digests must identify the accepted same-thread
Round-2 normative review, not the superseded Round-1 `REVISE` artifacts. Each
project-file array is ordered exactly as listed below. Each record has exactly
integer `bytes`, ASCII repository-relative `path`, an ordered `roles` array,
and lowercase-hex `sha256`. Allowed role strings are `generation-bootstrap`,
`generation-entrypoint`, `generation-module`, `review-entrypoint`,
`review-module`, and `review-read-input`; no inferred or extra
dependency is permitted. In the generation list path 1 has exactly
`[generation-entrypoint]`, paths 2--4 exactly `[generation-bootstrap]`, and
paths 5--9 exactly `[generation-module]`.

The eight amendment/review digests are computed from these exact paths, in
binding-key order:

1. `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY.json`
2. `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY.md`
3. `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY_REVIEW_ROUND2.json`
4. `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY_REVIEW_ROUND2.md`
5. `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_004_OPERATOR_MANIFEST_SCHEMA.json`
6. `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_004_OPERATOR_MANIFEST_SCHEMA.md`
7. `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_004_OPERATOR_MANIFEST_SCHEMA_REVIEW_ROUND2.json`
8. `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_004_OPERATOR_MANIFEST_SCHEMA_REVIEW_ROUND2.md`

The exact generation list is:

1. `scripts/experiments/generate_temporac_operator_manifest_v4.py`
2. `src/pams/__init__.py`
3. `src/pams/types.py`
4. `src/pams/temporac/__init__.py`
5. `src/pams/temporac/contract.py`
6. `src/pams/temporac/decode.py`
7. `src/pams/temporac/fixtures.py`
8. `src/pams/temporac/hashio.py`
9. `src/pams/temporac/types.py`

The exact detached implementation-review source list is the generation list
followed, without duplicate records, by:

10. `src/pams/temporac/gates.py`
11. `src/pams/temporac/nola.py`
12. `tests/__init__.py`
13. `tests/temporac/__init__.py`
14. `tests/temporac/test_gates_fixtures.py`
15. `tests/temporac/test_operator_manifest.py`

In the review list path 1 has exactly `[review-read-input]`, paths 2--13
exactly `[review-module]`, and paths 14--15 exactly `[review-entrypoint]`.
Generation executes exactly paths 2--9 plus entrypoint 1 in the locked
NumPy-only runtime. The two pytest entrypoints are not candidate-generation
dependencies and do not execute in that runtime; their hashes are source inputs
to a later detached implementation review. That review executes paths 2--13
and the two entrypoints while reading path 1 only as a bound file, under the
separate closed review protocol below. External stdlib/NumPy/PyTorch/pytest
dependencies are recorded by that review protocol and are never silently
promoted into the candidate generation closure. `cue.py`, `preprocess.py`, and
`quadrature.py` are not operator-generator or operator-review dependencies and
must not be listed unless a revised implementation actually imports them, in
which case this amendment requires a new normative review rather than an
inferred closure extension.

For generation, `proposal_sha256` is the digest of exactly
`refine-logs/temporac/FINAL_PROPOSAL.md`. The complete repository read allowlist
is the 15 review-project records, that proposal, and the eight exact
amendment/review paths above. The generation execute/import allowlist remains
only the nine generation-project records; review-only `gates.py`, `nola.py`,
both package initializers, and both tests are opened solely to compute their
receipt records and must not be imported or executed during candidate
generation. The generator may not read any other repository file. In particular
it must remove the historical diagnostic reads of the root proposal duplicate
and `cue.py`; `preprocess.py` and `quadrature.py` are likewise forbidden. The
implementation review must capture both a clean-process repo-local module-origin
trace and a repo-relative file-open trace, compare READ against the complete
read allowlist, and compare IMPORT/EXECUTE against the narrower generation list.
Runtime files outside the repository remain governed by the closed
OCI/Python/NumPy lock.

`source_snapshot_sha256` is computed over every allowed repository file in
global ASCII-path order as
`SHA256(ASCII("temporac.operator-generation-snapshot.v1") || NUL || concat(uint64_be(path_byte_count) || path_ascii || uint64_be(file_byte_count) || raw_file_bytes))`.
It covers the 15 review-project paths, canonical proposal, and eight contract
reads exactly once. Snapshot construction rejects a missing or extra path,
symlink, hard-link alias, normalized-path collision, `.pyc`, `__pycache__`,
`.pth`, `sitecustomize.py`, `usercustomize.py`, and every unlisted file before
Docker starts.

### 8.1 Detached implementation-review runtime and two-phase evidence

The broader pytest review is deliberately detached from candidate generation.
It creates no candidate member and cannot change a candidate hash. The fresh
reviewer emits one canonical JSON artifact with schema
`temporac.operator-implementation-review-evidence.v2`; the implementation-review
MD and JSON both bind its external SHA-256 and the exact candidate-receipt
SHA-256. It has exactly these top-level keys:
`candidate_receipt_sha256`, `generation_argv`, `generation_environment`,
`generation_module_origins`, `generation_process`,
`generation_project_open_trace`, `generation_runtime_lock_sha256`,
`generation_snapshot_sha256`, `pytest`, `review_argv`, `review_environment`,
`review_module_origins`, `review_process`, `review_project_open_trace`,
`review_runtime`, `review_runtime_sha256`, `review_snapshot`, `schema`, and
`status`. It contains no own digest. `status=PASS` requires both phases below.

The generation phase uses exact argv
`[/tmp/venv/bin/python,scripts/experiments/generate_temporac_operator_manifest_v4.py,--output,/out/operator_candidate_v2_20260816]`,
the exact Section 7 environment, runtime-lock digest
`8008df73cc9a807ac5e5262aebc170ae6e01c82b07658575f6207ec7cb0a14fd`,
and the receipt's `source_snapshot_sha256`. Its READ trace covers all 24 source
snapshot paths exactly once after duplicate collapse; IMPORT/EXECUTE covers
only the nine generation-project paths.

The review phase uses exact argv
`[/opt/temporac-review/bin/python,-B,-P,-m,pytest,-c,/dev/null,--rootdir=/work,-p,no:cacheprovider,-q,tests/temporac/test_gates_fixtures.py,tests/temporac/test_operator_manifest.py]`.
`review_environment` is exactly
`{CUDA_VISIBLE_DEVICES:"",HOME:"/nonexistent",LANG:"C.UTF-8",LC_ALL:"C.UTF-8",MKL_NUM_THREADS:"1",OMP_NUM_THREADS:"1",OPENBLAS_NUM_THREADS:"1",PYTEST_DISABLE_PLUGIN_AUTOLOAD:"1",PYTHONDONTWRITEBYTECODE:"1",PYTHONHASHSEED:"0",PYTHONNOUSERSITE:"1",PYTHONPATH:"/work/src",TZ:"UTC"}`;
`PYTEST_ADDOPTS` and every unlisted host variable are absent. Both phases have
no network, read-only roots/source snapshots, and writable tmpfs only.

`review_runtime` has exactly `architecture`, `base_image`, `byteorder`,
`distribution_records`, `platform_system`, and `python`. `base_image` uses the
closed pull-reference/index/linux-amd64-child/config descriptor shape from
Section 7. `distribution_records` is a nonempty array sorted by ASCII package
name and covers every installed distribution reachable by the entrypoints,
including NumPy, PyTorch, pytest, and transitive dependencies. Each record has
exactly `bytes`, `filename`, `installed_tree_sha256`, `name`, `sha256`,
`source_url`, and `version`; source bytes are verified before offline install.
`installed_tree_sha256` equals
`SHA256(ASCII("temporac.operator-review-distribution-tree.v1") || NUL || concat(uint64_be(relative_path_byte_count) || relative_path_utf8 || uint64_be(file_byte_count) || raw_file_bytes))`
over every regular installed file in global UTF-8 path-byte order; missing,
extra, duplicate, symlinked, or cache files fail. `python` has exactly
`executable_sha256`, `implementation`, and `version`, and the executable token
must resolve to `/opt/temporac-review/bin/python`. `review_runtime_sha256` is
SHA-256 of the Section 3 canonical bytes of this exact object. Version-only or
tag-only records fail.

`review_snapshot` has exactly `candidate_members`, `contract_reads`,
`project_files`, `schema`, and `tree_sha256`. Candidate members are the five
candidate files, contract reads are the proposal plus eight paths above, and
project files are the 15 review-project records. Every record has exactly
integer `bytes`, repository-relative `path`, and lowercase-hex `sha256`, sorted
by ASCII path. `tree_sha256` equals
`SHA256(ASCII("temporac.operator-review-snapshot.v1") || NUL || concat(uint64_be(path_byte_count) || path_ascii || uint64_be(file_byte_count) || raw_file_bytes))`
in global ASCII-path order. Its builder rejects missing/extra files, symlinks,
hard-link aliases, normalized-path collisions, `.pyc`, `__pycache__`, `.pth`,
`sitecustomize.py`, `usercustomize.py`, `conftest.py`, pytest configuration, and
unlisted files. The generation snapshot applies its own 24-file rule.

Each `*_module_origins` value is an array sorted by UTF-8 module-name bytes;
records have exactly `{module_name,origin_kind,origin_token,sha256}`.
For its phase, the array must equal the complete set of every distinct module
actually imported from interpreter startup through process exit, with no
missing, extra, or duplicate module name; an empty or subset inventory fails.
`origin_kind` is exactly one of `repo-source`, `stdlib-source`,
`stdlib-extension`, `distribution-source`, `distribution-extension`, `builtin`,
or `frozen`. Repo tokens are `/work/<ASCII repository path>`; stdlib tokens are
`{PYTHON_STDLIB}/<UTF-8 relative path>`; distribution tokens are
`{DIST:<ASCII normalized name>}/<UTF-8 relative path>`, where the distribution
name is PEP 503 normalized by ASCII lowercasing and replacing every maximal
run matching `[-_.]+` with one `-`; builtin/frozen tokens are their literal
kind. File origins use direct raw-file SHA-256;
builtin/frozen use SHA-256 of empty bytes. Namespace packages and cache origins
fail.

Each `*_project_open_trace` is the unique set of exact
`{access,path,sha256}` records, sorted first by ASCII path and then by access
order `READ`, `IMPORT`, `EXECUTE`. Repeated identical events collapse;
conflicting hashes fail. Generation READ equals the 24-file generation snapshot
and generation IMPORT/EXECUTE equals the nine-file generation set. Review READ
equals all review-snapshot files, review IMPORT/EXECUTE equals review paths
2--15, and review path 1 is READ-only.

Both process objects have exactly `argv_sha256`, `environment_sha256`,
`exit_code`, `fresh_process_identity_sha256`, `stderr_sha256`, and
`stdout_sha256`. `argv_sha256` is SHA-256 of
`ASCII("temporac.operator-process-argv.v1") || NUL || concat(uint64_be(token_byte_count) || token_utf8)`.
`environment_sha256` uses tag `temporac.operator-process-environment.v1` and
the same length framing over ASCII-key-sorted key then value UTF-8 bytes.
`stdout_sha256` and `stderr_sha256` are direct hashes of captured raw bytes.
`fresh_process_identity_sha256` is SHA-256 of
`ASCII("temporac.operator-process-identity.v1") || NUL || uint64_be(boot_id_utf8_bytes) || boot_id_utf8 || uint64_be(pid) || uint64_be(process_start_ticks) || launcher_nonce_raw32`;
the two nonces and identities must differ. Exit codes are zero.

`pytest` has exactly `collected_nodeids_sha256`, `failed`, `passed`, `skipped`,
and `xfailed`. The nodeid digest is the direct SHA-256 of each collected nodeid
encoded UTF-8 in pytest collection order, followed by LF, with one terminal LF
and no prefix. `failed`, `skipped`, and `xfailed` are zero and `passed` is
positive. The external implementation review fails if pinned generation replay
differs bytewise, either phase receipt is absent/nonconforming, or any
source/runtime/snapshot/origin/open-trace binding drifts.

`members` is an object with exactly `environment_lock.json`,
`operator_manifest_schema.json`, `replay_witness.json`, and
`temporac.operator-manifest.v4.json`, each mapped directly to its lowercase raw
file SHA-256. `blockers` is the exact ordered array
`[fresh_implementation_review_required,P2_not_claimed,S0_not_claimed,launch_authorization_zero]`.
`candidate_status` is exact
`FROZEN_CANDIDATE_PENDING_FRESH_IMPLEMENTATION_REVIEW`; `p2_status` and
`s0_status` are exact `NOT_CLAIMED`; `validated_row_count` is exactly 10,368.

The receipt contains no own digest. The fresh reviewer computes its digest
externally. Candidate `PASS` means only that deterministic construction and
replay matched this amendment; it cannot satisfy P2 or any later stage.

## 9. Current candidate evidence and required regeneration

The current candidate demonstrates one concrete serialization and is useful as
the audit basis:

- candidate schema SHA-256:
  `9f4b2ced821e667f950c1f56bbf9adbdc14b2c387c1932953048caf3f747a8cd`
- 15,039,512-byte manifest SHA-256:
  `1f39378d6f7a970b1f183558ed145bd97107f2aeda41902f4736e6778930e1bb`
- replay witness SHA-256:
  `36f60f41ac111a7b05e5f4846464cc774e09bb9faa44feed10be5a4f35418653`
- tag-only environment lock SHA-256:
  `709db0943752965524b507dd6340554a8d7d218be125317af0fc8b7748a5a947`
- candidate receipt SHA-256:
  `2e457e01166c6151e4edc688e8fd6f981d1278231fcd8c2bc000179f7a825102`

These values are historical candidate evidence, not normative output hashes.
After this amendment is fresh-reviewed, the generator must regenerate a new
candidate with the v3 runtime lock and v2 receipt. The manifest bytes should remain
identical if this amendment faithfully freezes the observed schema; any manifest
change must be explained and independently adjudicated rather than silently
accepted.

## 10. Review and authorization boundary

The same `gpt-5.6-sol` reviewer that issued the bound Round-1 `REVISE` record
must continue in-thread, reread this complete amendment, its JSON companion, the
accepted Amendment 002 record, current source/tests, and current candidate bytes,
and issue exactly one Round-2 normative verdict. The review remains
`same-family/provisional`. Only an `ACCEPT` whose MD and JSON digests are bound
by the regenerated receipt may authorize regeneration of the v2 candidate,
followed by a separate fresh implementation review with an independent
pinned-runtime replay.

Even a positive normative and implementation review authorizes only use of the
frozen candidate as an input to the separately specified P2 fixture gate. It does
not make P2 PASS, does not permit P3/S0, and grants no server, data, GPU, training,
paper-claim, Git, or release authority.
