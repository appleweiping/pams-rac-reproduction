# TempoRAC Contract Amendment 004 Review, Round 2: Operator Manifest Serialization

> **VERDICT: `ACCEPT`**
>
> **SAME-FAMILY PROVISIONAL — NON-AUTHORITATIVE**
>
> **ONLY NEXT STEPS: REGENERATE THE V2 CANDIDATE IN A NEW ABSENT DIRECTORY, THEN PERFORM A FRESH INDEPENDENT IMPLEMENTATION REVIEW**
>
> **P2=0, P3=0, S0=0, LAUNCH=0, SERVER=0, DATA=0, TRAINING=0, GATE=0, GIT=0, PAPER/CLAIM=0**

**Date:** 2026-08-16  
**Reviewer model:** `gpt-5.6-sol`  
**Reviewer family:** `openai`  
**Executor family:** `openai`  
**Review independence:** `same-family`  
**Acceptance status:** `provisional`  
**Continuation agent:** `/root/temporac_operator_manifest_amendment_review`

## 1. Decision

The complete revised `TEMPORAC_CONTRACT_AMENDMENT_004_OPERATOR_MANIFEST_SCHEMA`
is accepted as a serialization-only amendment. It freezes one unique set of
candidate member bytes and a fail-closed construction/review protocol without
changing the operator population, fixture mathematics, decoder, NOLA,
scientific thresholds, jobs, seeds, resource ledger, claims, or paper-visible
evidence inventory.

All five Round-1 blockers are closed. The amendment now fixes the top-level
JSON form, all 10,368 row positions, all 23 row keys and scalar domains, every
array dtype/shape/axis and raw C-order preimage, all aggregate names and roots,
four derived member commitments, an immutable pullable linux/amd64 runtime, a
receipt-last exclusive-create protocol, an acyclic no-self-hash receipt chain,
and exact generation/review project closures backed by detached module-origin
and file-open evidence.

This is normative acceptance of the two amendment files identified below. It
is not an implementation review and does not accept the historical v1
candidate. It permits only the next procedural step: regenerate the v2
candidate under the accepted lock into the specified new absent directory,
then submit that candidate to a fresh independent pinned-runtime implementation
review. No P2 or later authority follows.

## 2. Final bound amendment and input snapshot

The amendment MD and JSON were reread completely after the revision. The JSON
was parsed with duplicate-key rejection. Its 23 ordered `input_bindings` were
independently rehashed from disk; all 23 byte counts and SHA-256 values match.
The Round-1 artifacts remain byte-identical.

| Input | Bytes | SHA-256 |
|---|---:|---|
| `TEMPORAC_CONTRACT_AMENDMENT_004_OPERATOR_MANIFEST_SCHEMA.md` | 35,619 | `ece0025f4844dc04f838b5ec7f629a7fb3b10bd92e6f0bb3d5fe941cbcc83c82` |
| `TEMPORAC_CONTRACT_AMENDMENT_004_OPERATOR_MANIFEST_SCHEMA.json` | 37,403 | `61ce255e3f618041f3b54e19ab0c5d95f3817bafe68ac4b1e6db36d26e3083f4` |
| `FINAL_PROPOSAL.md` | 79,366 | `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391` |
| Amendment 002 MD | 10,622 | `899d6f071c89f898bf3825adfb30a326904f9bc7fe44d44351da8a5d0d9ccb14` |
| Amendment 002 JSON | 8,963 | `d524556f292ec17a155cdcd962cf8a88bea26f32b810ffb7bd8e45744f1c54fc` |
| Amendment 002 accepted Round-2 MD | 11,229 | `a86da733f3ae048b867ec8b2366a9f0262ca2b390ef08687ca7d710654e8301f` |
| Amendment 002 accepted Round-2 JSON | 9,363 | `3730d96eff1d3815499602eaf183774af032c514a588d4e92fe8a1cccf053e64` |
| Amendment 004 Round-1 MD | 15,851 | `573b49150b4f7ca6f15ec656d0375131c3d8a14214da075ad4d138e4edd71dfd` |
| Amendment 004 Round-1 JSON | 10,555 | `918ac01d43e193b9204edff7cb5f5e7d148fa12b180157dea2192a2af76eb482` |

The 15 bound project paths also rehash exactly: generator
`ffc5b830...f8c7`; `src/pams/__init__.py` `8b6ffbf3...67ce`;
`src/pams/types.py` `78c2ad53...3e31`; TempoRAC package initializer
`e3f71c2d...c2a3`; `contract.py` `5ab8fb62...745f`; `decode.py`
`766cd6ca...1f5e`; `fixtures.py` `979b8cee...b464`; `hashio.py`
`aa460d5d...d613`; TempoRAC `types.py` `4249eac2...3ca`; `gates.py`
`b6c9d743...b4a`; `nola.py` `ad6a1320...a637`; the two test package
initializers `27cbdeb3...9282` and `dd07192e...a24e`; and the two test
entrypoints `941856b6...173` and `a0552479...9dfe`.

## 3. Round-1 blocker closure

### R1-B1 — closed: no paper-visible block drift

The amendment now states exactly three paper-visible evidence blocks, one
primary claim, 27 training jobs, seeds 20260815/16/17, and 93 allocated A6000
hours. It explicitly states that the operator audit fixture does not become a
fourth paper-visible block. These values match proposal Sections 12 and 15.

### R1-B2 — closed: shapes and axes are exact

All fourteen arrays now have exact dtypes, shapes, and axis ownership. In
particular, association is truth-major `[K,C]=[3,3]`; locations and scores are
vectors `[3]`; interval tables are `[3,2]`; masks and responses are `[E]`;
run bounds are `[R,2]`, with `R=2` only for `run-reset`; and windows are
`[W,2]`, where each run contributes
`1+ceil(max(L_r-127,0)/32)`. Vectors cannot alias column vectors, interval
columns are `[start,stop)`, and dtype, shape, finiteness, C contiguity, and
axis semantics are checked before the unframed raw-byte hash.

### R1-B3 — closed: schema and replay witness are byte-unique

`operator_manifest_schema.json` and `replay_witness.json` now have exact closed
objects, keys, arrays, values, ordering, duplicate-key/BOM rejection, and
canonical JSON bytes. The reviewer independently serialized the prescribed
objects and reproduced:

| Derived member | Bytes | SHA-256 |
|---|---:|---|
| `operator_manifest_schema.json` | 3,607 | `3b29937c641274e73d10afd19bea943f28288b3d2851e8a5bf2313d7ea3e0fc3` |
| `environment_lock.json` | 3,732 | `8008df73cc9a807ac5e5262aebc170ae6e01c82b07658575f6207ec7cb0a14fd` |
| `replay_witness.json` | 2,896 | `348254cfe64cea35d6e29861755489721ebbbd4031f3ec1c044d7fa24fe38010` |
| `temporac.operator-manifest.v4.json` | 15,039,512 | `1f39378d6f7a970b1f183558ed145bd97107f2aeda41902f4736e6778930e1bb` |

The five bootstrap step strings joined by LF plus one terminal LF independently
produce 409 bytes and SHA-256
`66fdf1cd03616759948e326e1714a997848c45f40d005240a4ebe170240bf323`.

### R1-B4 — closed: receipt and executed dependency binding are exact

The candidate receipt now has an exact nine-key top level, exact status and
blocker strings, an all-false nine-key authority object, exact four-member map,
and exact thirteen-key binding object. It binds the accepted Amendment 002 and
review, the final Amendment 004, this accepted Round-2 review, the proposal,
the runtime, the source snapshot, and ordered generation/review project-file
records. It contains no own digest.

The exact project closures independently recount as:

| Closure | Count | Rule |
|---|---:|---|
| generation import/execute | 9 | generator; three package/bootstrap files; contract, decode, fixtures, hashio, TempoRAC types |
| detached review import/execute | 14 | review paths 2–15; generator path is read-only |
| detached review project records | 15 | the nine generation paths plus gates, NOLA, two test initializers, and two tests |
| generation repository reads | 24 | 15 review paths + canonical proposal + eight amendment/review paths |

Static import analysis confirms the current generator closure: importing the
package executes both package initializers and `pams.types`; fixtures imports
contract, hashio, and TempoRAC types and dynamically imports decode during the
full validation. The tests add gates, NOLA, both test initializers, and the two
test entrypoints. No current generation/review import reaches `cue.py`,
`preprocess.py`, or `quadrature.py`. A future extra project import cannot be
inferred into the receipt: the exact module-origin and project-open traces must
fail and force another normative amendment review.

### R1-B5 — closed: the generation runtime is pullable and immutable

The runtime is no longer tag-only or bare-digest-only. Independent registry
retrieval verified index digest
`sha256:e8be0ea148390d08bc077840cf87ac6a538d80b0ea1e8752b3e3982987cd0a53`.
Its sole unqualified linux/amd64 descriptor is the 2,516-byte child manifest
`sha256:0c7cf9e198201da2e838fcb61c07717dfc96adb3d1200599f54dea0af1d153bf`;
that manifest binds the 7,096-byte config
`sha256:c7ed3a18aaaa5a8b439cf84138ebcfe1e82e64d091781f3384ca38827824aefc`.
Extracting `/usr/local/bin/python3.12` from the bound OCI layer reproduced the
expected 18,208-byte executable SHA-256
`9a6988f011f466ae829f7945c54000a564e14a1a594cfb7079c02eda9b87620f`.

Independent retrieval of the exact NumPy wheel reproduced 16,040,181 bytes and
SHA-256 `24003ba8ff22ea29a8c306e61d316ac74111cebf942afbf692df65509a05f111`;
its `numpy/__init__.py` reproduces
`39c42db027548f958e096e8babe3fa0e3e773d24aa39eb6363fc0e3abbec34b1`.
The amendment correctly distinguishes the post-install `RECORD` commitment
from the wheel's source `RECORD`. The exact base image, pip supplied by that
image, wheel path, offline install command, environment, mounts, network-none
policy, and fresh venv make the installed value uniquely checkable during the
required locked regeneration and fresh implementation review.

## 4. Unique manifest serialization

The manifest is exactly one UTF-8 top-level array, never JSONL or a wrapper.
It has no BOM, duplicate keys, whitespace alternatives, NaN, self hash, or
compression; object keys are sorted, separators are comma/colon, and exactly
one terminal LF is present. Integers use shortest decimal spelling and the
only non-integer manifest tokens are literal `0.6`, `0.75`, and `1.0`.

Every row has exactly 23 closed keys and proposal-derived scalar values. The
rows follow gap × width × amplitude × offset × layout order and their 20-hex
character keys independently bind that order. Per-array SHA-256 preimages are
raw C-order bytes only after the exact little-endian or byte dtype and the
independent shape/axis checks; there is no header, frame, shape, row key, or LF
inside those hashes.

The replay witness closes the nine aggregate names and their raw row-order
concatenations. Consequently identical raw bytes under a wrong shape,
transposed association, native-endian alias, reordered row, alternate JSON
number spelling, or extra witness field cannot conform.

## 5. Independent historical-candidate replay

The historical candidate remains evidence only. Its 15,039,512-byte manifest
was reread in full, parsed with duplicate-key rejection, canonically
reserialized byte-for-byte, and independently regenerated without importing
the TempoRAC implementation. All 10,368 rows, 23-key schemas, scalar values,
row keys, shapes, raw array hashes, NOLA reconstruction, decoder components,
locations, scores, and associations matched; row mismatches were `0 / 10,368`.

| Aggregate | Bytes | SHA-256 |
|---|---:|---|
| association | 93,312 | `856710a129ad0e85fa46e2adb287b964163e0cd49d12f302ccfd8b63948712e6` |
| decoded components | 248,832 | `6c6e6240e577c16a4a58150102f4dfe3af2d28a914e8ceefce6e25a673411481` |
| decoder masks | 2,345,760 | `6cf03b8331b9c7f9021ae778dac8931e8c8cb3c5375cc0e2018de02cc77eb37a` |
| edge masks | 2,345,760 | `6cf03b8331b9c7f9021ae778dac8931e8c8cb3c5375cc0e2018de02cc77eb37a` |
| NOLA responses | 9,383,040 | `72a958eb3224e4585b7ef7cec9f1729c5c6c68ca8412a6f196cc28eb4fb9c5c2` |
| generated responses | 9,383,040 | `72a958eb3224e4585b7ef7cec9f1729c5c6c68ca8412a6f196cc28eb4fb9c5c2` |
| target masks | 2,345,760 | `6cf03b8331b9c7f9021ae778dac8931e8c8cb3c5375cc0e2018de02cc77eb37a` |
| truth masks | 2,345,760 | `45266d339f54a2a59f977a770d3f1bb71c9319b798b08d8b1b04de7aece8de21` |
| truth plateaus | 248,832 | `6c6e6240e577c16a4a58150102f4dfe3af2d28a914e8ceefce6e25a673411481` |

The replay also reconfirmed 8,945 actual inactive `b=255` edges in 5,665 rows,
first at key `00040001000000000001`, edge 42, bytes `cccccc3e`; and 2,304
flat amplitude-one rows containing 6,912 flat plateaus. The four normative
inventory roots and all five row witnesses remain unchanged.

A diagnostic execution of the two current test entrypoints passed `13 / 13`
with bytecode and pytest cache disabled. This host result is not the detached
locked implementation review and grants no candidate acceptance.

## 6. Two-phase evidence, module origins, and receipt DAG

The detached evidence schema closes both processes independently. Generation
is tied to the exact v3 lock, argv, six-value controlled environment, 24-file
snapshot, nine-file import/execute closure, raw stdout/stderr, and fresh process
identity. Review is tied to exact pytest argv/environment, a closed snapshot of
five candidate members, nine contract reads, and 15 project files, plus a
content-addressed base image, Python executable, offline distribution source
artifacts, and complete installed-tree roots.

For each phase, module-origin evidence covers every distinct imported module
from interpreter startup through exit. Records are sorted by UTF-8 module name,
use seven closed origin kinds, hash source/extension bytes directly, and reject
namespace and cache origins. Distribution tokens use an exact PEP 503
normalization: ASCII lowercase followed by replacement of every maximal
`[-_.]+` run by one hyphen. The paired project-open trace separately proves
all reads and the narrower import/execute set, so a complete runtime module
inventory cannot conceal an unbound repository dependency.

The content graph is acyclic:

1. proposal, accepted Amendment 002, Round 1, final Amendment 004, this Round-2
   review, and 15 project files determine the 24-file generation snapshot;
2. the generator creates and verifies the four committed non-receipt members;
3. the receipt is written last, binds those members, runtime, snapshot, and
   this Round-2 review, and deliberately contains no receipt digest;
4. detached implementation evidence binds the receipt digest externally and
   records both fresh processes; and
5. a later implementation-review MD/JSON binds both the receipt and evidence
   digest externally.

No manifest field hashes the manifest, no receipt hashes itself, and no earlier
node depends on a later node. Missing receipt means incomplete tree. Every
candidate member is exclusive-created in a new absent directory, and any
pre-existing path or retry fails without overwrite.

## 7. Preservation and authority ceiling

No method, threshold, population, model, job, seed, resource, dataset,
evaluator capability, result, or claim changed. The amendment only makes the
already observed operator serialization and its audit chain byte-unique.

This `ACCEPT` permits only:

1. subsequent regeneration of the v2 candidate in the exact new absent
   directory under Amendment 004; and
2. a subsequent fresh independent pinned-runtime implementation review of that
   regenerated candidate and its two-phase evidence.

It does not authorize implementation in this review turn, accept the historical
candidate, make the regenerated candidate authoritative, pass P2, enter P3 or
S0, pass any F23 gate, launch a server or GPU job, access natural/evaluator/
sealed/held-out data, train, use a result, edit or promote the paper, make a
claim, use Git, release, or submit. Even a later positive implementation review
may only make the candidate eligible as an input to a separately specified P2
fixture gate; it cannot make P2 PASS.

## 8. Final disposition

**ACCEPT — same-family provisional, zero scientific/gate/release authority.**  
**Blocking issue count:** `0`.  
**Required next sequence:** regenerate v2 candidate → fresh independent
pinned-runtime implementation review → remain P2/S0/launch zero.
