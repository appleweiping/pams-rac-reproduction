# TempoRAC Contract Amendment 006 Review, Round 1: Operator Runtime Wheel Path

> **VERDICT: `REVISE`**
>
> **SAME-FAMILY PROVISIONAL - NON-AUTHORITATIVE**
>
> **P2=0, P3=0, S0=0, LAUNCH=0, SERVER=0, DATA=0, TRAINING=0, GATE=0, GIT=0, PAPER=0, CLAIM=0**

**Date:** 2026-08-16  
**Reviewer model:** `gpt-5.6-sol`  
**Reasoning effort:** `ultra`  
**Reviewer family:** `openai`  
**Executor family:** `openai`  
**Review independence:** `same-family`  
**Acceptance status:** `provisional`  
**Fresh context:** `true`  
**Reviewer agent:** `/root/temporac_operator_runtime_amendment006_review/fresh_ultra_review`

## 1. Decision

Amendment 006 correctly identifies and repairs pip's invalid short wheel
basename. The exact old first four bootstrap steps fail exactly as claimed,
and the corrected first four steps install and import NumPy 2.1.0 in the
locked environment. The amendment nevertheless cannot be accepted because
the corrected install does not reproduce the retained installed-`RECORD`
commitment in `corrected_runtime_lock.numpy.record_sha256`.

The fixed local wheel was independently verified as a regular, non-symlink
16,040,181-byte file with SHA-256
`24003ba8ff22ea29a8c306e61d316ac74111cebf942afbf692df65509a05f111`.
After the exact corrected first four commands, the same preimage used by the
current generator,
`importlib.metadata.distribution("numpy").read_text("RECORD").encode("utf-8")`,
is 100,089 bytes with SHA-256
`76ae0d49a0d236c7d045e8394d649e1252a6ef2219ef2dac33f72ac01ea76cca`.
That result reproduced in two fresh containers. The proposed lock instead
retains
`65cfe9b7bbac44a46116b2b5f8282f32ec57e1aa9faeb64d523604c712b446d8`.

This is load-bearing. If the prohibited fifth generator step were run, the
current `_runtime_lock_bytes` check would reject the installed environment
before creating the candidate directory. Thus the proposed 3,976-byte
runtime object with digest `71fdf4...` is a correct serialization of the
object as written, but it is not the lock produced by its own corrected
bootstrap and current verification algorithm.

**Blocking issue count:** `1`.

## 2. Exact runtime probes

### 2.1 Pinned inputs

All probes used `linux/amd64`, `--network none`, `--read-only`, executable
`/tmp:rw,exec,nosuid,nodev,size=512m`, the exact six-variable environment,
the repository mounted read-only at `/work`, a fresh writable output parent,
and the pinned pull reference. No probe contained or executed the generator
step. All probe-created temporary directories were outside the worktree and
were deleted.

| Runtime input | Bytes | SHA-256 |
|---|---:|---|
| OCI index | 9,719 | `e8be0ea148390d08bc077840cf87ac6a538d80b0ea1e8752b3e3982987cd0a53` |
| sole unqualified linux/amd64 child manifest | 2,516 | `0c7cf9e198201da2e838fcb61c07717dfc96adb3d1200599f54dea0af1d153bf` |
| image config bound by that child | 7,096 | `c7ed3a18aaaa5a8b439cf84138ebcfe1e82e64d091781f3384ca38827824aefc` |
| resolved `/usr/local/bin/python3.12` | 18,208 | `9a6988f011f466ae829f7945c54000a564e14a1a594cfb7079c02eda9b87620f` |
| exact NumPy wheel | 16,040,181 | `24003ba8ff22ea29a8c306e61d316ac74111cebf942afbf692df65509a05f111` |

The cached OCI index has exactly one unqualified linux/amd64 descriptor, and
that descriptor names the child and config above. The container reports
Linux/x86_64, CPython 3.12.4. The corrected basename parses as distribution
`numpy`, version `2.1.0`, with tags
`cp312-cp312-manylinux_2_17_x86_64` and
`cp312-cp312-manylinux2014_x86_64`; `/wheel/numpy.whl` is rejected by pip's
wheel-filename parser before wheel content is opened.

### 2.2 Old path

The exact old first four bootstrap strings were executed with the exact
wheel mounted at `/wheel/numpy.whl`. Steps 1-3 passed. Step 4 exited `1`.

| Observation | Exact result |
|---|---|
| stdout | 0 bytes, `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| stderr | 48 bytes, `9218fcf778490e9a23c6865cbbdf10c000cd690375b16b4e4e5848d0b45371a4` |
| stderr UTF-8 | `ERROR: numpy.whl is not a valid wheel filename.\n` |
| candidate before/after | absent / absent |
| generator command present or executed | no / no |

Because the full accepted 409-byte script uses `/bin/sh -ceu`, appending its
unchanged fifth line cannot alter this result: shell execution stops on the
step-4 status. The old full script therefore necessarily has the claimed
exit, streams, non-execution, and absence behavior.

### 2.3 Corrected path and the blocking installed metadata

The exact corrected first four commands completed with exit `0` and empty
stderr. Pip reported successful installation of NumPy 2.1.0. A diagnostic
command after those four steps imported version `2.1.0` and reproduced the
22,007-byte `numpy/__init__.py` SHA-256
`39c42db027548f958e096e8babe3fa0e3e773d24aa39eb6363fc0e3abbec34b1`.
The candidate child was absent before and after. This is direct diagnostic
execution, not historical-witness reliance and not candidate or gate evidence.

The installed metadata is path-sensitive:

| Installed artifact/preimage | Bytes | SHA-256 |
|---|---:|---|
| generator's normalized `RECORD` text preimage | 100,089 | `76ae0d49a0d236c7d045e8394d649e1252a6ef2219ef2dac33f72ac01ea76cca` |
| raw installed `RECORD` bytes | 101,316 | `b30c47e53bf16be292c392247bad8948f3fb2407ecd54fde388f74be007426e0` |
| installed `direct_url.json` | 286 | `3c36ea74fe65236c1dcd3bc80373860762a80217f52a6c41a7a390805ce27e98` |

`direct_url.json` records the exact corrected `file:///wheel/numpy-2.1.0-...whl`
URL, and its `RECORD` row commits that file. The wheel-path correction is
therefore not semantically isolated from the installed `RECORD` commitment.
The raw and normalized preimages differ because the raw file has CRLF while
`read_text` normalizes line endings; Amendment 006 must state which preimage
is normative and make the runtime lock and verifier agree.

For diagnosis only, replacing the proposed record digest with the current
generator-compatible `76ae...` value yields 3,976 canonical bytes with
SHA-256 `47c2227b74b627a03bf9b5e4930dd02484caaae5a695ae5c0129f04781115cd8`.
Using direct raw-file SHA-256 instead yields 3,976 bytes with SHA-256
`951fe94022c18ccbc5e377f155a3f577066d176e22aa2e0b482bd1421eac4d0f`.
These are diagnostic alternatives, not accepted replacement commitments.

## 3. Canonical preimages and literal delta

Independent duplicate-rejecting parsing and Amendment 004 canonical
serialization reproduced every stated preimage:

| Projection | Bytes | SHA-256 | Result |
|---|---:|---|---|
| old bootstrap, LF join plus one LF | 409 | `66fdf1cd03616759948e326e1714a997848c45f40d005240a4ebe170240bf323` | match |
| corrected bootstrap, LF join plus one LF | 592 | `c830092276818bcacf4a08eb8818481fe47b7d8f608229cac1fc672ad2947602` | match |
| Amendment 004 `runtime_lock` canonical JSON | 3,732 | `8008df73cc9a807ac5e5262aebc170ae6e01c82b07658575f6207ec7cb0a14fd` | match |
| Amendment 006 `corrected_runtime_lock` canonical JSON | 3,976 | `71fdf4bfe78b59aada9f2e207002eae3fcd10345ed1f47ee9969557d6649f5da` | match as written |

The structural comparison has exactly five differing leaves and a 244-byte
canonical-length increase:

1. `container_policy.bootstrap_steps[0]` - stat path;
2. `container_policy.bootstrap_steps[1]` - SHA-256 path;
3. `container_policy.bootstrap_steps[3]` - pip-install path;
4. `container_policy.docker_argv_template[13]` - bind destination; and
5. `container_policy.bootstrap_script_sha256` - derived digest.

There are exactly four old-path occurrences in the old runtime, exactly four
corrected-path occurrences in the corrected runtime, and no old-path
occurrence in the corrected runtime. Every other literal field/value is
unchanged. Mount target, all three bootstrap references, NumPy filename,
platform, image, environment, workdir, output argv, network, root, and tmpfs
are mutually consistent. The blocker is precisely that literal comparison
misses the path-derived installed metadata produced by pip.

## 4. Candidate receipt v3 and dependency graph

The additive receipt design is otherwise closed and acyclic:

| Check | Result |
|---|---|
| candidate-receipt top-level keys | exactly 9, canonical ASCII order |
| binding keys | exactly 17, canonical ASCII order |
| contract-read paths | exactly 12 and aligned with the first 12 binding keys |
| generation snapshot | exactly 28 unique paths, framed in global ASCII path order |
| generation import/execute paths | exactly 9 |
| detached review snapshot | 5 candidate members + 13 contract reads + 15 project files = 33 unique records |
| receipt self hash | absent |
| accepted-review self hash | forbidden/absent by design |
| future accepted-review injectability | feasible after an `ACCEPT` review exists |

The fixed accepted-review paths are positions 11 and 12 of the contract-read
list. They are feasible without self-lock: a future positive review can bind
the Amendment 006 MD/JSON externally while containing no digest of either
review output; a later generator can then hash both review outputs into the
28-file snapshot and receipt. The receipt is written last without its own
digest, detached evidence hashes it externally, and later implementation
review files hash the receipt/evidence externally. No earlier node depends on
a later node.

The legacy literal `24` retained inside the runtime object is explicitly
overridden to `28` for a post-Amendment-006 receipt by Sections 3 and 5. The
source-snapshot tag and length framing remain unchanged. This precedence is
clear enough, and the global source-snapshot ordering is distinct from the
binding-key ordering of the twelve contract reads. No ordering or DAG blocker
was found.

Because this verdict is `REVISE`, the fixed
`TEMPORAC_CONTRACT_AMENDMENT_006_OPERATOR_RUNTIME_WHEEL_PATH_REVIEW_ACCEPTED`
paths are not created or overwritten. Only the explicit Round-1 review paths
are used.

## 5. Frozen science and current implementation scope

Amendment 006 changes no manifest algorithm/schema, 10,368-row population,
Cartesian row order, 23-key row schema, per-array preimage, aggregate,
fixture geometry, NOLA rule, one-decode rule, threshold, job, seed, GPU-hour
ledger, dataset boundary, primary claim, or three-block paper-visible evidence
ceiling. The unchanged values remain one claim, 27 jobs, seeds
20260815/20260816/20260817, 93 allocated A6000-hours, and three paper-visible
blocks.

The current generator and two tests are correctly identifiable as
pre-amendment implementation inputs: they still bind the 3,732-byte old
runtime, receipt v2, thirteen binding keys, and a 24-path generation snapshot.
That is not evidence that Amendment 006 has been implemented. The present
review evaluates the specification only and makes no implementation change.

## 6. Input binding and JSON validation

All 19 ordered Amendment 006 input bindings matched at review start and were
rehashed unchanged immediately before finalization:

| Bound input | Bytes | SHA-256 |
|---|---:|---|
| `.agents/skills/research-refine/SKILL.md` | 30,984 | `bb489173329fec7a6551c74c319e2103d83f722ea11fc11462a7c4ccacd00b09` |
| `.agents/skills/shared-references/output-versioning.md` | 4,828 | `de8e7ac23069de6c5d0c482d4cef117dd5a543e00b57e2dc6c62560a457290c5` |
| `.agents/skills/shared-references/output-manifest.md` | 1,753 | `e767083e0ef97a7adf7a4e8a574090067a59c0655d5aca422a3747cd32b6afde` |
| `.agents/skills/shared-references/output-language.md` | 2,215 | `0f1447579bd7a2195fd4a074be70ae6345b5c368f1a70d327e89748bf5a378f9` |
| `.agents/skills/shared-references/review-tracing.md` | 4,435 | `4f4936984391da87464b8e8bf6f7c984f3f186e8fdc42890d0ac999b5e2024dc` |
| `refine-logs/temporac/FINAL_PROPOSAL.md` | 79,366 | `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391` |
| Amendment 002 MD | 10,622 | `899d6f071c89f898bf3825adfb30a326904f9bc7fe44d44351da8a5d0d9ccb14` |
| Amendment 002 JSON | 8,963 | `d524556f292ec17a155cdcd962cf8a88bea26f32b810ffb7bd8e45744f1c54fc` |
| Amendment 002 accepted review MD | 11,229 | `a86da733f3ae048b867ec8b2366a9f0262ca2b390ef08687ca7d710654e8301f` |
| Amendment 002 accepted review JSON | 9,363 | `3730d96eff1d3815499602eaf183774af032c514a588d4e92fe8a1cccf053e64` |
| Amendment 004 MD | 35,619 | `ece0025f4844dc04f838b5ec7f629a7fb3b10bd92e6f0bb3d5fe941cbcc83c82` |
| Amendment 004 JSON | 37,403 | `61ce255e3f618041f3b54e19ab0c5d95f3817bafe68ac4b1e6db36d26e3083f4` |
| Amendment 004 Round-1 MD | 15,851 | `573b49150b4f7ca6f15ec656d0375131c3d8a14214da075ad4d138e4edd71dfd` |
| Amendment 004 Round-1 JSON | 10,555 | `918ac01d43e193b9204edff7cb5f5e7d148fa12b180157dea2192a2af76eb482` |
| Amendment 004 accepted Round-2 MD | 15,404 | `6c45d2bc333c8b89915c3e3b881b893bf26203085bab010a28ad55158f9468a7` |
| Amendment 004 accepted Round-2 JSON | 14,557 | `f62135f6ac502b4ca9ccfda8534201c12a60a2853ae4a298bf088c7a24bbd50f` |
| `scripts/experiments/generate_temporac_operator_manifest_v4.py` | 30,081 | `286108c9b222f786ff1db25ddb6bc81a53acf318337a77c252f77092d5dd1e4e` |
| `tests/temporac/test_gates_fixtures.py` | 6,695 | `69690aba7eb366471a28bb7406fd1411964d271ecd250834151243bb8a7c299f` |
| `tests/temporac/test_operator_manifest.py` | 21,494 | `627e58cbb65be96de7d17cd56e8c12c179c2d1c3e11612f0aea5cf79eecdb5de` |

Supplemental consumed inputs are externally bound here:

| Supplemental input | Bytes | SHA-256 |
|---|---:|---|
| `.agents/skills/shared-references/reviewer-independence.md` | 1,809 | `d66262e74d5f69b1614007bf744b9f2220aeb35d1f36ec66196dfe32b0438530` |
| `.agents/skills/shared-references/reviewer-routing.md` | 5,490 | `e1fc0841b342b64130aa1c8fc99b337b1ef9e994697f17a518ebf1bbca664c66` |
| `.agents/skills/shared-references/assurance-contract.md` | 5,797 | `1df4ab0f26a81b7b904fb7ba066e485a7c12cfe0333ec49017f097e922a59aed` |
| `.agents/skills/shared-references/output-composition.md` | 5,293 | `79a72c7ea02102c460ae3ad55853c21a4054b9df93f377c29dd34555e6d3c3a7` |
| Amendment 006 MD | 14,146 | `33240e87560ffd249190a1a85be513a33699755247e37b30795cfa530085cecf` |
| Amendment 006 JSON | 18,955 | `8bc3aa1c16a124ca221faa12f5eb471f8bf05b59032320091abbcf85b5a67d01` |
| X0 historical candidate receipt | 1,341 | `d56b14e4bd0371bd10ec6366a3ccfc724c934dcb8f8bac08cc242fe19102c4f4` |
| operator historical candidate receipt | 1,729 | `2e457e01166c6151e4edc688e8fd6f981d1278231fcd8c2bc000179f7a825102` |
| OCI index blob | 9,719 | `e8be0ea148390d08bc077840cf87ac6a538d80b0ea1e8752b3e3982987cd0a53` |
| linux/amd64 child blob | 2,516 | `0c7cf9e198201da2e838fcb61c07717dfc96adb3d1200599f54dea0af1d153bf` |
| OCI config blob | 7,096 | `c7ed3a18aaaa5a8b439cf84138ebcfe1e82e64d091781f3384ca38827824aefc` |
| exact local wheel | 16,040,181 | `24003ba8ff22ea29a8c306e61d316ac74111cebf942afbf692df65509a05f111` |
| resolved container Python executable | 18,208 | `9a6988f011f466ae829f7945c54000a564e14a1a594cfb7079c02eda9b87620f` |

All eight consumed JSON inputs parsed as UTF-8 JSON with duplicate-key
rejection, no BOM, and one terminal LF. The review JSON and every trace JSON
were also duplicate-key parsed after persistence. Every produced text/JSON
artifact is UTF-8 without BOM and has exactly one terminal LF. The canonical
review Markdown is byte-identical to the full trace response.

## 7. Required revision and authority ceiling

Amendment 006 must be revised before any implementation work. The revision
must:

1. freeze the installed `RECORD` hash preimage explicitly;
2. bind the value actually produced by the corrected full basename under the
   pinned pip/runtime, or change the verifier and contract together under
   fresh review;
3. recompute the corrected runtime commitment and every dependent receipt
   expectation; and
4. withdraw the claim that the only semantic runtime delta is four path
   literals plus the bootstrap hash, because pip's installed metadata is also
   path-derived.

This `REVISE` review is same-family provisional and has zero authority. It
does not authorize edits to the generator, either test, the amendment,
proposal, plan, tracker, manifest, data, candidate, server, training, Git,
paper, claim, gate, P2, P3, S0, or launch state. The only procedural next
sequence is a separately authorized revision of Amendment 006 MD/JSON,
followed by another fresh normative review. No candidate generation or
implementation review is reachable from this disposition.
