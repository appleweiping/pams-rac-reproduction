# TempoRAC Contract Amendment 006 — Operator Runtime Wheel Path

> **STATUS: `PROPOSED_PENDING_FRESH_REVIEW`**
>
> **SAME-FAMILY PROVISIONAL — NON-AUTHORITATIVE**
>
> **P2=0, P3=0, S0=0, LAUNCH=0, SERVER=0, DATA=0, TRAINING=0, GATE=0, GIT=0, PAPER=0, CLAIM=0**

**Date:** 2026-08-16

**Scope:** close only Round-1 blocker `A006-R1-B1`: retain the runtime-path
correction, bind the path-derived installed `RECORD` value produced by that
correction, and retain the minimum additive provenance needed to prevent a
later candidate from binding only the known-broken Amendment 004 runtime. This
amendment creates no candidate, implementation, test, gate receipt, experiment,
result, or authority.

## 1. Historical contract and precedence

Accepted Amendment 004 and its Round-2 review remain immutable historical
records. Amendment 002 and its accepted Round-2 review remain unchanged. The
canonical `refine-logs/temporac/FINAL_PROPOSAL.md` remains unchanged. If and
only if this Amendment 006 receives a fresh positive normative review, it
additively overrides Amendment 004 on exactly two surfaces for every subsequent
operator candidate:

1. the corrected runtime object, whose semantic delta from accepted Amendment
   004 is exactly six leaves: the wheel bind target, bootstrap steps 1, 2, and
   4, the derived bootstrap-script digest, and `numpy.record_sha256`; and
2. the candidate-receipt provenance closure, expanded to bind Amendment 006
   and its future accepted fresh-review MD/JSON.

Relative to the rejected Amendment 006 Round-1 runtime-object bytes, the sole
runtime-object leaf change in this revision is `numpy.record_sha256`. The
wheel path, corrected bootstrap bytes, receipt counts, and dependency graph do
not change in this Round-2 author revision.

All other Amendment 004 clauses remain in force. A candidate or implementation
review that binds Amendment 004 but omits the four Amendment 006/fresh-review
digests is nonconforming. Historical Amendment 004 and Round-2 files must not
be overwritten, reinterpreted as successful runtime execution, or silently
substituted by this proposal.

## 2. Exact failing runtime witness

The accepted Amendment 004 `environment_lock.json` is 3,732 canonical bytes
with SHA-256
`8008df73cc9a807ac5e5262aebc170ae6e01c82b07658575f6207ec7cb0a14fd`.
Its five LF-joined bootstrap strings plus exactly one terminal LF are 409 bytes
with SHA-256
`66fdf1cd03616759948e326e1714a997848c45f40d005240a4ebe170240bf323`.

Those exact 409 bytes were executed unchanged with:

- platform `linux/amd64`;
- image pull reference and index digest
  `docker.io/library/python@sha256:e8be0ea148390d08bc077840cf87ac6a538d80b0ea1e8752b3e3982987cd0a53`;
- 2,516-byte linux/amd64 child manifest SHA-256
  `0c7cf9e198201da2e838fcb61c07717dfc96adb3d1200599f54dea0af1d153bf`;
- 7,096-byte image config SHA-256
  `c7ed3a18aaaa5a8b439cf84138ebcfe1e82e64d091781f3384ca38827824aefc`;
- `--network none`, `--read-only`, and executable 512 MiB `/tmp` tmpfs;
- the exact 16,040,181-byte NumPy wheel with SHA-256
  `24003ba8ff22ea29a8c306e61d316ac74111cebf942afbf692df65509a05f111`.

The wheel byte-count and SHA-256 steps passed, and the fresh venv step
completed. The exact fourth command

```text
/tmp/venv/bin/python -m pip install --no-index --no-deps /wheel/numpy.whl
```

exited `1`. Captured stdout was empty: 0 bytes, SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
Captured stderr was exactly the following 48 UTF-8/ASCII bytes, including its
single terminal LF:

```text
ERROR: numpy.whl is not a valid wheel filename.
```

Its direct raw-byte SHA-256 is
`9218fcf778490e9a23c6865cbbdf10c000cd690375b16b4e4e5848d0b45371a4`.
Because `/bin/sh -ceu` stopped at that command, the generator step did not run.
The designated candidate child was absent before execution and remained absent
after execution. Wheel integrity passed; the failure is only the shortened
bind-target basename, which pip rejects as a wheel filename.

## 3. Bounded runtime and installed-`RECORD` correction

The old path is exact `/wheel/numpy.whl`. The corrected path is exact:

```text
/wheel/numpy-2.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

Exactly four path substitutions are permitted in the Amendment 004 runtime
object:

1. replace the `dst=/wheel/numpy.whl` suffix in the wheel bind token by the
   full corrected basename;
2. replace the wheel path in bootstrap step 1;
3. replace the wheel path in bootstrap step 2; and
4. replace the wheel path in bootstrap step 4.

The five corrected bootstrap strings, in exact order, are:

```text
test $(stat -c %s /wheel/numpy-2.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl) = 16040181
test $(sha256sum /wheel/numpy-2.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl | cut -d ' ' -f 1) = 24003ba8ff22ea29a8c306e61d316ac74111cebf942afbf692df65509a05f111
/usr/local/bin/python -m venv /tmp/venv
/tmp/venv/bin/python -m pip install --no-index --no-deps /wheel/numpy-2.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
/tmp/venv/bin/python scripts/experiments/generate_temporac_operator_manifest_v4.py --output /out/operator_candidate_v2_20260816
```

Joined by one LF between strings and exactly one terminal LF, these are exactly
592 bytes with SHA-256
`c830092276818bcacf4a08eb8818481fe47b7d8f608229cac1fc672ad2947602`.
No shell command, argument, environment variable, output path, image, wheel
byte, install flag, order, or policy may otherwise change.

Those four path substitutions plus the derived
`container_policy.bootstrap_script_sha256` are the prior five semantic leaf
deltas. The sixth and only newly repaired leaf is
`numpy.record_sha256`. Its authoritative preimage is exactly the bytes returned
after the pinned corrected-path install by:

```python
importlib.metadata.distribution("numpy").read_text("RECORD").encode("utf-8")
```

That preimage is exactly 100,089 bytes with SHA-256
`76ae0d49a0d236c7d045e8394d649e1252a6ef2219ef2dac33f72ac01ea76cca`
and reproduced identically in two fresh pinned containers. The verifier
algorithm is unchanged; the corrected path changes pip's installed metadata.
The rejected Round-1 value
`65cfe9b7bbac44a46116b2b5f8282f32ec57e1aa9faeb64d523604c712b446d8`
is replaced by that authoritative digest in this sole runtime-object leaf.

The corrected runtime object is the companion JSON's
`corrected_runtime_lock`. Serialize that object under Amendment 004 Section 3:
UTF-8, `ensure_ascii=false`, `allow_nan=false`, lexicographically sorted object
keys, separators `,` and `:`, no BOM, no duplicate key, no insignificant
whitespace, and exactly one terminal LF. The result is exactly 3,976 bytes with
SHA-256
`47c2227b74b627a03bf9b5e4930dd02484caaae5a695ae5c0129f04781115cd8`.
The embedded runtime schema remains
`temporac.operator-runtime-lock.v3`; changing it would not be this repair.

The literal Amendment 004 runtime object's 24-path source-snapshot count is
retained inside those 3,976 bytes. For a post-Amendment-006 candidate receipt
v3, Section 5 below additively overrides only that provenance count from 24 to
28; the source-snapshot framing algorithm and tag remain unchanged. This narrow
precedence rule resolves the otherwise contradictory historical count without
changing the runtime schema or any source-snapshot framing leaf.

## 4. Non-authorizing repair probe

A separate diagnostic probe used the same fixed image digest, `linux/amd64`,
`--network none`, `--read-only`, executable 512 MiB `/tmp` tmpfs, and the same
already verified wheel host bytes. It mounted the wheel at the complete
corrected basename above. The corrected first four bootstrap steps completed,
and the venv Python reported NumPy version `2.1.0`; the probe exited `0`.

The finalized Round-1 review repeated the corrected install in two fresh
containers and measured the exact authoritative `read_text(...).encode(...)`
preimage frozen in Section 3. For diagnosis only, the installed raw `RECORD`
file uses CRLF and is 101,316 bytes with SHA-256
`b30c47e53bf16be292c392247bad8948f3fb2407ecd54fde388f74be007426e0`.
It is explicitly non-authoritative because the current generator uses
`read_text`, which normalizes line endings before UTF-8 encoding. The installed
`direct_url.json` is 286 bytes with SHA-256
`3c36ea74fe65236c1dcd3bc80373860762a80217f52a6c41a7a390805ce27e98`;
it is also diagnostic/non-authoritative and explains why the installed
`RECORD` is path-derived.

This author revision closes `A006-R1-B1` at the specification-artifact level
only: it freezes the exact verifier preimage and correct path-derived digest,
repairs every dependent commitment, and removes the false five-leaf-only
semantic-delta claim. Closure remains pending the required fresh Round-2
review and conveys no implementation or candidate authority.

This probe establishes only that the proposed pathname correction is
executable. It did not run the fifth generator step, did not create a candidate,
did not verify candidate members, and is not normative acceptance, an
implementation review, P2, S0, a gate receipt, launch authority, or scientific
evidence. The original failure witness and this repair probe are distinct and
must never be collapsed into one successful accepted-Amendment-004 execution.

## 5. Candidate receipt v3 additive provenance closure

No candidate may be generated before this amendment receives a fresh positive
normative review. If accepted and separately implemented, the next candidate
receipt uses schema `temporac.operator-manifest-candidate-receipt.v3`. Its
top-level keys remain exactly, in canonical lexicographic order:

1. `authority`
2. `bindings`
3. `blockers`
4. `candidate_status`
5. `members`
6. `p2_status`
7. `s0_status`
8. `schema`
9. `validated_row_count`

All Amendment 004 v2 receipt values and sub-schemas remain unchanged except:

- `schema` becomes `temporac.operator-manifest-candidate-receipt.v3`;
- `bindings` has exactly the following 17 keys, with the four Amendment 006
  keys added and no other key:

```text
amendment_002_json_sha256
amendment_002_md_sha256
amendment_002_review_json_sha256
amendment_002_review_md_sha256
amendment_004_json_sha256
amendment_004_md_sha256
amendment_004_review_json_sha256
amendment_004_review_md_sha256
amendment_006_json_sha256
amendment_006_md_sha256
amendment_006_review_json_sha256
amendment_006_review_md_sha256
generation_project_files
proposal_sha256
review_project_files
runtime_lock_sha256
source_snapshot_sha256
```

- `runtime_lock_sha256` and `members["environment_lock.json"]` both equal
  `47c2227b74b627a03bf9b5e4930dd02484caaae5a695ae5c0129f04781115cd8`;
- the other three non-receipt member commitments remain exactly Amendment 004:
  schema 3,607 bytes / `3b29937c...e0fc3`, replay witness 2,896 bytes /
  `348254cf...010`, and manifest 15,039,512 bytes / `1f39378d...e1bb`;
- the receipt still contains no digest of itself.

The twelve contract-read paths, in binding-key order, are exactly:

1. `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY.json`
2. `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY.md`
3. `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY_REVIEW_ROUND2.json`
4. `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY_REVIEW_ROUND2.md`
5. `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_004_OPERATOR_MANIFEST_SCHEMA.json`
6. `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_004_OPERATOR_MANIFEST_SCHEMA.md`
7. `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_004_OPERATOR_MANIFEST_SCHEMA_REVIEW_ROUND2.json`
8. `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_004_OPERATOR_MANIFEST_SCHEMA_REVIEW_ROUND2.md`
9. `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_006_OPERATOR_RUNTIME_WHEEL_PATH.json`
10. `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_006_OPERATOR_RUNTIME_WHEEL_PATH.md`
11. `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_006_OPERATOR_RUNTIME_WHEEL_PATH_REVIEW_ACCEPTED.json`
12. `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_006_OPERATOR_RUNTIME_WHEEL_PATH_REVIEW_ACCEPTED.md`

The two `REVIEW_ACCEPTED` paths may be created only by a fresh review whose
verdict is `ACCEPT`, which externally binds the exact Amendment 006 MD/JSON
digests. A `REVISE` or `REJECT` round must not create or replace them. Candidate
generation fails if either accepted-review file is absent, non-ACCEPT,
self-contradictory, or does not bind the exact Amendment 006 bytes.

The generation snapshot contains exactly 28 unique paths: the unchanged 15
Amendment 004 `review_project_paths`, the canonical proposal, and the twelve
contract-read paths above. Global ASCII-path order is normative. The hash is

```text
SHA256(
  ASCII("temporac.operator-generation-snapshot.v1") || NUL ||
  concat_global_ASCII_path_order(
    uint64_be(path_byte_count) || path_ascii ||
    uint64_be(file_byte_count) || raw_file_bytes
  )
)
```

There is no outer count, filename normalization, JSON, delimiter, digest
rendering, or terminal LF beyond bytes already present in a file. The existing
missing/extra path, symlink, hard-link alias, normalized-path collision,
`.pyc`, `__pycache__`, `.pth`, `sitecustomize.py`, and `usercustomize.py`
rejections remain. Generation READ equals these 28 paths after duplicate
collapse; generation IMPORT/EXECUTE remains the same nine project paths.

The detached review snapshot correspondingly has exactly five candidate
members, thirteen contract reads (canonical proposal plus the twelve above),
and fifteen project files: 33 unique records total. Its existing
`temporac.operator-review-snapshot.v1` framing, ordering, exclusions, module
origin rules, open-trace rules, process rules, and two-phase evidence schema
remain unchanged. A future implementation-review evidence file binds the
external candidate-receipt v3 SHA-256; the candidate receipt itself explicitly
binds Amendment 006 and its accepted review, so the chain is transitive and
closed.

## 6. Exact acyclic dependency graph

The only conforming order is:

1. immutable proposal + accepted Amendment 002/Round-2 + historical Amendment
   004/Round-2 + current implementation inputs determine this Amendment 006;
2. Amendment 006 MD/JSON are persisted without self hashes;
3. a fresh normative review reads and externally hashes both files; only an
   `ACCEPT` may create the two fixed `REVIEW_ACCEPTED` files, which contain no
   self hash;
4. a separately authorized implementation updates the generator/tests and
   builds the exact 28-file snapshot, including all four Amendment 006/review
   files;
5. the generator writes the four committed non-receipt members, including the
   corrected 3,976-byte runtime lock, then writes receipt v3 last without a
   self hash;
6. detached implementation evidence externally binds the receipt v3 digest;
7. later implementation-review MD/JSON externally bind the candidate receipt
   and evidence digests.

No earlier node depends on a later node. Missing accepted-review files, a v2
receipt, the old runtime digest, a 24-path post-006 snapshot, or a candidate
binding only Amendment 004 fails closed.

## 7. Frozen non-changes and authority ceiling

This amendment changes no wheel bytes, wheel hash, image/index/child/config
digest, Python or NumPy version, imported `__init__.py`, installed-`RECORD`
preimage algorithm, network/root/tmpfs policy, environment, install flags,
generator argv/output, algorithm, manifest schema, manifest row, row order,
aggregate, fixture population, NOLA, decoder, threshold, job, seed, GPU hour,
dataset, claim, or paper-visible evidence block. It changes the installed
`RECORD` commitment only because the corrected wheel pathname is recorded by
pip, exactly as frozen in Section 3.

The status remains `PROPOSED_PENDING_FRESH_REVIEW`. It authorizes no edit to
the generator or tests, no candidate generation, no implementation review, no
P2/P3/S0, no launch/server/data/training/gate/Git action, and no paper or claim
action. A positive same-family review remains provisional and may authorize
only a separately executed implementation update and later candidate/review
sequence explicitly stated by that review.

## 8. Bound current input bytes

The companion JSON lists the exact current byte counts and SHA-256 values for
the governing skill, shared output/tracing protocols, canonical proposal,
accepted Amendment 002 files, full Amendment 004 and both review rounds, the
current generator, the two current operator test entrypoints, and the finalized
Amendment 006 Round-1 review: Markdown 16,333 bytes / SHA-256
`45c557b2827cf21885d116cc9baab4f02548054821ed1ba209aa4e2ec1c75cb6`
and JSON 35,832 bytes / SHA-256
`9cf30558d1bf18ef198d4ff5e6def253af3f8bd001faabb736b3aaf47a3c83a6`.
These are historical inputs to this proposed correction, not expected hashes
of a future implementation. Any future candidate must bind the actual reviewed
implementation bytes in its 15 project-file records.
