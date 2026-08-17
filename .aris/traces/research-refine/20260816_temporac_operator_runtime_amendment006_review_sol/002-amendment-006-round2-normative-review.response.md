# TempoRAC Contract Amendment 006 Review, Round 2: Operator Runtime Wheel Path

> **VERDICT: `ACCEPT`**
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

The revised Amendment 006 is accepted. Round-1 blocker `A006-R1-B1` is
closed: the corrected wheel basename installs NumPy 2.1.0 in the pinned
runtime, and the revised runtime now commits the exact preimage evaluated by
the unchanged generator algorithm:

`importlib.metadata.distribution("numpy").read_text("RECORD").encode("utf-8")`

That preimage is 100,089 bytes with SHA-256
`76ae0d49a0d236c7d045e8394d649e1252a6ef2219ef2dac33f72ac01ea76cca`.
It reproduced byte-for-byte in two fresh installs. The corrected runtime
serializes to 3,976 bytes with SHA-256
`47c2227b74b627a03bf9b5e4930dd02484caaae5a695ae5c0129f04781115cd8`,
and every dependent runtime/member/receipt commitment now uses that digest.

The current fixed Amendment aliases are byte-identical to the revised
`_20260816_123444` pair and are externally bound by this review:

| Amendment input | Bytes | SHA-256 |
|---|---:|---|
| fixed/revised MD | 16,965 | `9b1e3a9c36c360e8d9011145d120f7cff8e6922f4620c60445d8194809070517` |
| fixed/revised JSON | 21,582 | `e443453b93c96dfda4c6daf68332c0f667429cbf2a036e16e0e6c43c89d9fa61` |

**Blocking issue count:** `0`.

## 2. Pinned runtime and independent probes

### 2.1 Inputs and execution boundary

The local wheel was freshly resolved as a regular, non-symlink file and
rehashed before the probes: 16,040,181 bytes,
`24003ba8ff22ea29a8c306e61d316ac74111cebf942afbf692df65509a05f111`.
Local Docker inspection freshly returned the 9,719-byte OCI index descriptor
`e8be0ea148390d08bc077840cf87ac6a538d80b0ea1e8752b3e3982987cd0a53`;
a temporary `--pull never --platform linux/amd64` container selection freshly
returned the sole unqualified linux/amd64 child descriptor, 2,516 bytes,
`0c7cf9e198201da2e838fcb61c07717dfc96adb3d1200599f54dea0af1d153bf`.
The unchanged child-bound config commitment is 7,096 bytes,
`c7ed3a18aaaa5a8b439cf84138ebcfe1e82e64d091781f3384ca38827824aefc`.
The latter is reviewer-owned byte evidence retained in the hash-bound Round-1
trace rather than a new Round-2 OCI export; the actual image selection,
platform, installs, and executable bytes were rechecked directly.

Every runtime probe used the digest-pinned image, `linux/amd64`, network
`none`, read-only root, executable
`/tmp:rw,exec,nosuid,nodev,size=512m`, the exact six environment variables,
read-only work and wheel mounts, and a fresh writable output parent. No
network access or download occurred. The fifth generator command was never
included or executed. All temporary material was outside the worktree and
was deleted.

### 2.2 Old short basename

A fresh replay of the exact old first four steps mounted the verified wheel
at `/wheel/numpy.whl`. Steps 1-3 passed; pip step 4 failed before opening the
wheel payload.

| Observation | Exact result |
|---|---|
| exit | `1` |
| stdout | 0 bytes, `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| stderr | 48 bytes, `9218fcf778490e9a23c6865cbbdf10c000cd690375b16b4e4e5848d0b45371a4` |
| stderr UTF-8 | `ERROR: numpy.whl is not a valid wheel filename.\n` |
| designated child before/after | absent / absent |

The accepted full old script is the same first four commands followed by the
generator and runs under `/bin/sh -ceu`. Therefore the step-4 status makes the
generator unreachable, the script necessarily exits `1`, its streams are the
ones above, and the designated child remains absent.

### 2.3 Corrected full basename

The exact corrected first four commands plus a read-only diagnostic probe
were run twice in separate fresh containers. Both executions exited `0`, pip
reported `Successfully installed numpy-2.1.0`, import reported version
`2.1.0`, and the designated child was absent before and after.

| Installed artifact/preimage | Bytes | SHA-256 | Authority |
|---|---:|---|---|
| `read_text("RECORD").encode("utf-8")` | 100,089 | `76ae0d49a0d236c7d045e8394d649e1252a6ef2219ef2dac33f72ac01ea76cca` | authoritative runtime preimage |
| raw installed `RECORD` | 101,316 | `b30c47e53bf16be292c392247bad8948f3fb2407ecd54fde388f74be007426e0` | diagnostic only |
| installed `direct_url.json` | 286 | `3c36ea74fe65236c1dcd3bc80373860762a80217f52a6c41a7a390805ce27e98` | diagnostic only |
| installed `numpy/__init__.py` | 22,007 | `39c42db027548f958e096e8babe3fa0e3e773d24aa39eb6363fc0e3abbec34b1` | version/import diagnostic |
| resolved venv Python executable | 18,208 | `9a6988f011f466ae829f7945c54000a564e14a1a594cfb7079c02eda9b87620f` | runtime diagnostic |

The raw `RECORD` contained 1,227 CRLF endings. `read_text` performs universal
newline normalization, explaining the 1,227-byte difference. The generator
still reads the distribution metadata with `read_text("RECORD")` and hashes
`record.encode("utf-8")`; Amendment 006 correctly makes only that normalized
100,089-byte preimage normative. Raw `RECORD` and `direct_url.json` remain
explicitly non-authoritative diagnostics.

This is direct Round-2 execution evidence, not reliance on the amendment's
repair witness or on the author's account. It also independently demonstrates
that the corrected path's installed metadata is stable across fresh installs.

## 3. Canonical runtime, bootstrap, and deltas

Duplicate-key-rejecting parsing followed by the Amendment 004 canonical JSON
rule reproduced all commitments:

| Projection | Bytes | SHA-256 |
|---|---:|---|
| accepted Amendment 004 runtime | 3,732 | `8008df73cc9a807ac5e5262aebc170ae6e01c82b07658575f6207ec7cb0a14fd` |
| rejected Amendment 006 Round-1 runtime | 3,976 | `71fdf4bfe78b59aada9f2e207002eae3fcd10345ed1f47ee9969557d6649f5da` |
| revised Amendment 006 runtime | 3,976 | `47c2227b74b627a03bf9b5e4930dd02484caaae5a695ae5c0129f04781115cd8` |
| old bootstrap, exact LF join plus one LF | 409 | `66fdf1cd03616759948e326e1714a997848c45f40d005240a4ebe170240bf323` |
| corrected bootstrap, exact LF join plus one LF | 592 | `c830092276818bcacf4a08eb8818481fe47b7d8f608229cac1fc672ad2947602` |

The recursive leaf diff from accepted Amendment 004 to the revised runtime is
exactly six leaves, in canonical pointer order:

1. `/container_policy/bootstrap_script_sha256`;
2. `/container_policy/bootstrap_steps/0`;
3. `/container_policy/bootstrap_steps/1`;
4. `/container_policy/bootstrap_steps/3`;
5. `/container_policy/docker_argv_template/13`; and
6. `/numpy/record_sha256`.

The recursive leaf diff from rejected Amendment 006 Round 1 to the revised
runtime is exactly one leaf: `/numpy/record_sha256`, changing `65cfe9...` to
`76ae0d...`. There are exactly four `/wheel/numpy.whl` occurrences in the
accepted old runtime, exactly four corrected full-path occurrences in the
revised runtime, and zero old-path occurrences in the revised runtime.

The corrected basename equals `numpy.filename`; the argv has exactly one
read-only wheel bind to that basename; bootstrap steps 1, 2, and 4 each use it
once; and no other bootstrap step uses a wheel path. The image reference is
identical in `base_image.pull_reference`, the environment, and argv. Platform,
network, workdir, output mount/argv, root mode, tmpfs, environment, Python,
wheel bytes, and image fields are otherwise unchanged. The revised statement
of semantic isolation is therefore accurate: four literal wheel-path leaves,
their derived bootstrap hash, and the path-derived installed RECORD digest.

The revised runtime digest occurs at exactly three dependent JSON locations:
`corrected_runtime_commitment.sha256`, receipt
`bindings.runtime_lock_sha256`, and receipt member
`members["environment_lock.json"]`. No stale `71fdf4...` dependency remains.

## 4. Receipt v3, snapshots, and dependency graph

The future receipt contract is closed and mechanically consistent:

| Check | Result |
|---|---|
| receipt top-level keys | exactly 9 |
| receipt binding keys | exactly 17 |
| contract-read paths | exactly 12, in the first 12 binding-key order |
| generation snapshot | exactly 28 unique files |
| generation import/execute paths | exactly 9 |
| project files in review snapshot | exactly 15 |
| detached review snapshot | 5 candidate members + 13 contract reads + 15 project files = 33 unique records |
| receipt self hash | absent |
| review-output self hashes | absent by rule and by these outputs |

The generation snapshot is the unchanged 15-file review-project set, the
canonical proposal, and the 12 ordered contract-read paths, deduplicated and
globally sorted by ASCII path bytes. Its framing remains
`ASCII(temporac.operator-generation-snapshot.v1)||NUL`, followed by, for every
path, `uint64_be(path_byte_count)||path_ascii||uint64_be(file_byte_count)||raw_file_bytes`.
The runtime object's historical literal `24` is retained for byte-level
lineage, while the explicit receipt-v3-only `28` override unambiguously
supersedes it for candidate generation. The two ordering rules are not
conflated: binding order selects the 12 reads; global ASCII order frames the
snapshot.

The fixed accepted-review paths are contract-read positions 11 and 12. They
are feasible without self-lock. This review binds the exact Amendment 006
MD/JSON bytes but contains no digest of either accepted review output. A later
bounded implementation may inject both review hashes; a later candidate may
then hash the 28 fixed source files, write four non-receipt members, and write
the nine-key receipt last without a receipt self hash. A subsequent detached
implementation review hashes the five candidate members and the 28 external
inputs. No node depends on its own bytes or on a future node.

The present generator already validates an absent absolute designated child,
creates the directory with non-overwrite semantics, writes each member with
`O_EXCL`/`O_NOFOLLOW`, and writes the receipt last. It is still pre-amendment
code, so those mechanics do not claim that receipt v3 or the revised runtime
has already been implemented.

## 5. Frozen science and specification/implementation boundary

Amendment 006 changes no manifest algorithm or schema, 10,368-row population,
Cartesian row order (`gaps`, `widths`, `amplitudes`, offsets 0 through 31,
then layouts), 23-field row schema, per-array encoding, aggregate roots,
fixture geometry, decoder, one-decode rule, NOLA reconstruction, thresholds,
dataset boundary, claims, jobs, seeds, or GPU ledger. The population remains
9 gaps x 3 widths x 3 amplitudes x 32 offsets x 4 layouts = 10,368 rows, from
`(4,1,0.60,0,interior)` through `(129,4,1.00,31,run-reset)` in the accepted
key order. The ledger remains 27 jobs, seeds 20260815/20260816/20260817,
93 allocated A6000-hours, one primary claim, and three paper-visible evidence
blocks.

The current generator and two tests remain explicitly pre-amendment: they
still bind the 3,732-byte Amendment 004 runtime, receipt v2, 13 bindings, and
the 24-path source snapshot. That is expected at normative-review time and is
not implementation evidence. No code, test, candidate, data, gate, server,
training, proposal, plan, tracker, manifest, Git, paper, or claim was changed
in this review.

## 6. Input bindings and supplemental evidence

All 21 ordered input bindings in the revised amendment matched current bytes
at review start and again immediately before finalization:

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
| Amendment 006 Round-1 review MD | 16,333 | `45c557b2827cf21885d116cc9baab4f02548054821ed1ba209aa4e2ec1c75cb6` |
| Amendment 006 Round-1 review JSON | 35,832 | `9cf30558d1bf18ef198d4ff5e6def253af3f8bd001faabb736b3aaf47a3c83a6` |
| `scripts/experiments/generate_temporac_operator_manifest_v4.py` | 30,081 | `286108c9b222f786ff1db25ddb6bc81a53acf318337a77c252f77092d5dd1e4e` |
| `tests/temporac/test_gates_fixtures.py` | 6,695 | `69690aba7eb366471a28bb7406fd1411964d271ecd250834151243bb8a7c299f` |
| `tests/temporac/test_operator_manifest.py` | 21,494 | `627e58cbb65be96de7d17cd56e8c12c179c2d1c3e11612f0aea5cf79eecdb5de` |

Supplemental consumed inputs and runtime members were independently hashed:

| Supplemental input/member | Bytes | SHA-256 |
|---|---:|---|
| `AGENTS.md` | 664 | `35eaf270a8f50c943577a7d00c4d5a78a22a02874489709db6ada5c850aa2251` |
| `reviewer-independence.md` | 1,809 | `d66262e74d5f69b1614007bf744b9f2220aeb35d1f36ec66196dfe32b0438530` |
| `reviewer-routing.md` | 5,490 | `e1fc0841b342b64130aa1c8fc99b337b1ef9e994697f17a518ebf1bbca664c66` |
| `assurance-contract.md` | 5,797 | `1df4ab0f26a81b7b904fb7ba066e485a7c12cfe0333ec49017f097e922a59aed` |
| `output-composition.md` | 5,293 | `79a72c7ea02102c460ae3ad55853c21a4054b9df93f377c29dd34555e6d3c3a7` |
| archived rejected A006 MD `_20260816_113951` | 14,146 | `33240e87560ffd249190a1a85be513a33699755247e37b30795cfa530085cecf` |
| archived rejected A006 JSON `_20260816_113951` | 18,955 | `8bc3aa1c16a124ca221faa12f5eb471f8bf05b59032320091abbcf85b5a67d01` |
| revised A006 MD `_20260816_123444` | 16,965 | `9b1e3a9c36c360e8d9011145d120f7cff8e6922f4620c60445d8194809070517` |
| revised A006 JSON `_20260816_123444` | 21,582 | `e443453b93c96dfda4c6daf68332c0f667429cbf2a036e16e0e6c43c89d9fa61` |
| X0 historical candidate receipt | 1,341 | `d56b14e4bd0371bd10ec6366a3ccfc724c934dcb8f8bac08cc242fe19102c4f4` |
| operator historical candidate receipt | 1,729 | `2e457e01166c6151e4edc688e8fd6f981d1278231fcd8c2bc000179f7a825102` |
| Round-1 trace request | 11,674 | `95948086e6507e8b393ae6d02594aa8a001c2e50d143ab69d63234eadd5e41cb` |
| Round-1 trace response | 16,333 | `45c557b2827cf21885d116cc9baab4f02548054821ed1ba209aa4e2ec1c75cb6` |
| Round-1 call metadata | 891 | `e33ecb97380a2c68af5a92af6fbad10017037797ddc77cdc335ed4841d59f8bf` |
| Round-1 run metadata before continuation | 687 | `ab397e7a0e570088ff28123fbd83cbd4756896f8067143afa6feaab4906d48d4` |
| Round-1 deterministic verification | 6,678 | `eb2349a8b9aea57db897058a76412edcca7bb901b93bf7936836217d1c74b7a4` |
| OCI index | 9,719 | `e8be0ea148390d08bc077840cf87ac6a538d80b0ea1e8752b3e3982987cd0a53` |
| linux/amd64 child manifest | 2,516 | `0c7cf9e198201da2e838fcb61c07717dfc96adb3d1200599f54dea0af1d153bf` |
| OCI config | 7,096 | `c7ed3a18aaaa5a8b439cf84138ebcfe1e82e64d091781f3384ca38827824aefc` |
| exact local NumPy wheel | 16,040,181 | `24003ba8ff22ea29a8c306e61d316ac74111cebf942afbf692df65509a05f111` |
| resolved container Python executable | 18,208 | `9a6988f011f466ae829f7945c54000a564e14a1a594cfb7079c02eda9b87620f` |

The intentional Round-2 update to `run.meta.json` is an output transition;
the 687-byte hash above binds the fully consumed pre-continuation version.
All other bound and supplemental filesystem inputs were unchanged at the end.

## 7. Validation, authority ceiling, and next actions

All consumed JSON was parsed with duplicate-key rejection. The revised fixed
and timestamped JSON files are byte-identical, valid UTF-8 without BOM, and
have exactly one terminal LF; the same is true of the corresponding Markdown.
The accepted review JSON and every new trace JSON were parsed again after
persistence. Every produced text/JSON file is UTF-8 without BOM with exactly
one terminal LF. The canonical review Markdown is byte-identical to the full
trace response. No accepted review file contains its own hash.

This `ACCEPT` is same-family provisional and non-authoritative. It does not
authorize P2, P3, S0, launch, server, data, training, gate, Git, paper, claim,
results, release, or evaluator capability. It permits only this ordered,
bounded next sequence:

1. A separate implementer may update exactly
   `scripts/experiments/generate_temporac_operator_manifest_v4.py`,
   `tests/temporac/test_gates_fixtures.py`, and
   `tests/temporac/test_operator_manifest.py` to implement the accepted
   amendment, with no other file or authority mutation.
2. Only after that bounded update, the implementer may generate one candidate
   into a newly absent designated directory, with no overwrite.
3. Only after candidate generation, a fresh reviewer must perform the pinned-
   runtime implementation review. No campaign or gate authority follows from
   either this normative acceptance or candidate creation.
