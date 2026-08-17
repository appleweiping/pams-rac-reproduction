# TempoRAC Contract Amendment 005 Certificate Provenance - Round 5 Normative Review

Date: 2026-08-16  
Reviewer: `/root/temporac_certificate_amendment005_review`  
Route: Codex `gpt-5.6-sol`, OpenAI same-family, ultra review tier  
Assurance: provisional; not cross-family independent  
Verdict: **REVISE**  
Blocking groups: **4**  
Authority: **none**

## Outcome

The Round-5 candidate substantially repairs the three Round-4 findings. It
now freezes all 10 Linux-applicable `PyPreConfig` fields and all 64
Linux-applicable `PyConfig` fields for CPython 3.12.13, names the constructors,
setters, status branches, runtime readbacks and `PyConfig_Clear` order, and
defines 20-key launch-FD rows, 15-key readback rows and three readback stages.
It also creates an acyclic bootstrap-only generated-source capsule, requires
two equal discovery outputs before the final environment, gives installed
members one 16-key association preimage, and gives eleven trace-origin kinds
closed object schemas.

Those repairs do not yet form an executable byte-unique contract. Four
authority-bearing defects remain:

1. the sealed-memfd construction closes the only reference before trying to
   reopen the anonymous file through that descriptor;
2. the two discovery launches' configuration and FD readbacks are required in
   prose but have no canonical evidence object or committed digest;
3. the generated-source rules both require the actually nonempty bootstrap
   sequence and declare that the trusted set is empty and every instance must
   be rejected; and
4. the `trusted-generated` trace origin requires three fields that cannot be
   derived from the discovery/trusted row because their preimages and mappings
   are absent.

The first defect makes the stipulated OS setup impossible. The other three
permit omission or multiple inequivalent encodings at the exact provenance
boundary that was meant to close Round-4 B2/B3. Exact supervisor, environment,
output and trace hashes can record a chosen implementation, but they cannot
make an undefined or contradictory construction normative. The disposition is
therefore `REVISE`, not conditional acceptance.

## Reviewed bytes and terminal integrity

The exact inputs matched at review start and terminal rehash:

| Input | Bytes | SHA-256 |
|---|---:|---|
| timestamped Round-5 Markdown `_20260816_173527.md` | 269,255 | `dbf797e80156d627b14290dfe558222878fdec8ccefbf28a051093130e1349e4` |
| timestamped Round-5 JSON `_20260816_173527.json` | 226,592 | `94fc8e5db1031686ea0d5e5646c94606728c18cff19915a6deb231a3bf80cfcb` |
| fixed Markdown alias | 269,255 | `dbf797e80156d627b14290dfe558222878fdec8ccefbf28a051093130e1349e4` |
| fixed JSON alias | 226,592 | `94fc8e5db1031686ea0d5e5646c94606728c18cff19915a6deb231a3bf80cfcb` |
| archived rejected Round-4 Markdown `_20260816_173526.md` | 203,317 | `0552aaca74b34929ead1124cce846d2a583931aa89ad29fcf0f97486b7be0198` |
| archived rejected Round-4 JSON `_20260816_173526.json` | 165,291 | `04d24ad19b19470bcd675974766a279dc7691d3fdaecf665e3fd24e4a4aaeb82` |
| Round-4 review Markdown | 18,713 | `988eb7c2b61eb445a07e1c17b7bcf44be256f9948d6600a6ed6edc62ef71cfc1` |
| Round-4 review JSON | 12,977 | `f8552023abb52c98baec6a9143d15e93e3268d883bf3967c446c3fcbcd9d06d9` |
| canonical proposal | 79,366 | `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391` |
| experiment plan | 34,198 | `4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e` |
| experiment tracker | 22,360 | `714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b` |
| certificate postfix Round-2 Markdown | 12,283 | `93a9cae771b34c185b9057286794b894d615aec5424011df145aaf8900f008d8` |
| certificate postfix Round-2 JSON | 14,527 | `144d2f22f4f75166b8cbefe54157d08690408979dd7cb34ef8c97e1738432872` |

The fixed aliases are byte-identical to the `_20260816_173527` pair. The
`_20260816_173526` archive remains byte-identical to the rejected Round-4
candidate. The current Markdown Appendix A is byte-identical to the standalone
JSON.

Both current files decode as strict UTF-8 without BOM or CR and have exactly
one terminal LF. The JSON has no duplicate key, non-finite value or negative
zero, and every object is recursively UTF-8 lexical-key ordered. All 57
amendment-bound paths match at start and terminal rehash: 3/3 runtime-bootstrap
files, 24/24 TempoRAC source files, 19/19 tests and 11/11 planning/review files.
No bound-input drift was observed.

I fully reread the project-local `research-refine`, `research-review` and
`experiment-audit` skills and their reviewer-routing, assurance, independence,
trace, output, scope, integration and experiment-integrity protocols. I also
reread the complete current amendment pair, Round-4 review pair, canonical
proposal, plan, tracker, cited postfix pair, the current 57 bindings, and the
CPython 3.12.13 initialization documentation, exact header and release-source
bindings. Author summaries and intermediate candidate bytes were not treated
as evidence.

## Blocking finding A005-R5-B1 - the sealed-memfd construction order is impossible

JSON line 1270 defines each sealed input as an anonymous memfd that is created
writable, populated, fully sealed, **closed**, and then reopened “through its
descriptor” as a distinct `O_RDONLY` open-file description. Once the last
reference to a memfd is closed, the anonymous file is released. Moreover,
`/proc/self/fd/N` exists only while descriptor `N` is open. The relevant Linux
semantics are explicit in
[`memfd_create(2)`](https://man7.org/linux/man-pages/man2/memfd_create.2.html)
and
[`proc_pid_fd(5)`](https://man7.org/linux/man-pages/man5/proc_pid_fd.5.html).
There is no descriptor through which to perform the required reopen after that
close.

This is not an implementation preference. A conforming controller cannot
construct FD0/3/4/5 under the written order, while retaining the required
distinct read-only open-file description and forbidding writable inputs at
exec. Reusing the original writer, duplicating it, or reopening a disk path
violates the same closed row semantics.

Minimal repair: while the sealed writable descriptor is still open, open
`/proc/self/fd/<writer-fd>` with exactly `O_RDONLY|O_CLOEXEC` to obtain a
distinct open-file description; validate its access/status flags, offset,
size, seals and exact EOF; then close the original writer and move the
read-only descriptor to the prescribed child FD. State this order explicitly
and retain the existing no-extra/no-shared-description readbacks.

## Blocking finding A005-R5-B2 - twin discovery launch/readback evidence is not committed

The bootstrap DAG is now directionally acyclic: capsule and zero-process
precommit precede two discoveries; their equal output precedes trusted rows and
the final environment; a distinct full P06 follows. The missing object is the
evidence that the two producers actually ran under the closed launch,
configuration and descriptor state.

JSON line 1892 requires both launch/config/FD readbacks to be “bound” before
trusted rows are derived. But the exact discovery-output object at lines
1854-1876 contains only `capsule_sha256`, `rows` and `schema`. The trusted index
at lines 1925-1935 contains one discovery-output byte count/hash/root, not two
launch instances or their readbacks. The final environment at lines 1751-1764
binds the capsule, precommit, output root and trusted index but no twin-evidence
root. The pre-G5a completion index covers P00 through NATP and the 27 job
entrypoints; it does not define two bootstrap-discovery owners or a P06
payload-index schema for their records. Finally, discovery has one output
channel, FD3, whose role is only `discovery-output`; no record framing assigns
canonical bytes to the output, three FD readbacks, configuration readback,
launch instance, exit and join evidence.

Consequently, byte-identical row outputs can be retained while one launch is
omitted, reused, run with a different configuration, or accompanied by no
committed readback. Prose saying “bind” is not an authority-bearing preimage.

Minimal repair: define one canonical discovery-evidence index with exactly two
ordered rows. Each row must commit a fresh process ordinal, launch-instance
bytes/hash, the three exact FD-readback bytes/hashes, complete initialization
readback bytes/hash, exact discovery-output bytes/hash/root, joined exit and
capsule/precommit equality. Define unambiguous pipe framing or separate fixed
FD roles, hash the complete index, require both row outputs byte-equal, and
bind that evidence-index digest in the trusted index and final environment.
This is evidence within the existing P06 owner, not a new job.

## Blocking finding A005-R5-B3 - the generated-source acceptance rules contradict the exact runtime

The discovery-output rule at JSON lines 1854-1876 requires the complete finite
sequence of non-file-backed source compile/exec events caused by importing
exact `dataclasses`, and explicitly permits the freshly observed sequence to
be nonempty. The trusted-index completeness/order at lines 1767-1936 and trace
events `PY_TRUSTED_GENERATED_COMPILE`/`EXEC` at lines 2048-2069 require those
rows to be consumed by every full runtime. In direct contradiction, trace
origin equality at lines 2224-2229 says the trusted set is empty in this
profile and every `trusted-generated` instance is rejected.

The exact CPython 3.12.13 runtime makes the contradiction concrete rather than
hypothetical. The bound
[`Lib/dataclasses.py`](https://raw.githubusercontent.com/python/cpython/v3.12.13/Lib/dataclasses.py)
imports `inspect`. The same-release
[`Lib/inspect.py`](https://raw.githubusercontent.com/python/cpython/v3.12.13/Lib/inspect.py)
creates several `namedtuple` classes during module import. The same-release
[`Lib/collections/__init__.py`](https://raw.githubusercontent.com/python/cpython/v3.12.13/Lib/collections/__init__.py)
implements `namedtuple` by constructing a source string and calling `eval`.
Under the amendment's pre-initialization native audit hook and non-file-backed
source filter, the bootstrap sequence therefore contains generated compile and
execution events. A full runtime must emit them, but the origin verifier is
required to reject every one. No conforming trace exists.

Minimal repair: delete the false empty-set rule. Require the trusted set to be
exactly the complete two-run-equal discovery rows in ordinal order; admit an
instance if and only if it byte-equals that row and occurs in the required
trusted compile-to-exec chain before `DISPATCH_BEGIN`; continue to reject every
extra, missing, reordered, project/third-party or post-boundary event.

## Blocking finding A005-R5-B4 - trusted-generated origins have no reconstructible row preimage

Even after the empty-set contradiction is removed, the event-specific origin
cannot be reconstructed. Discovery and trusted rows at JSON lines 1858-1869
and 1911-1923 contain `generator_origin_class` and `generator_sha256` plus the
generated source hex/hash. The 13-key `trusted-generated` origin at lines
2338-2351 instead requires `trusted_row_sha256`, `generator_origin_kind` and
`generator_origin_sha256`. `trusted_row_sha256` appears nowhere else in the
amendment, so it has no standalone canonical row preimage. No rule maps
`generator_origin_class` to `generator_origin_kind` or `generator_sha256` to
`generator_origin_sha256`.

Two implementations can therefore invent different trusted-row digests or
kind/name mappings while matching every discovery/trusted index byte. The
generic origin canonicalizer hashes either invented object successfully; it
cannot prove membership in the precommitted row it claims to represent. This
is precisely the typed-preimage substitution path that Round-4 B3 required the
revision to close.

Minimal repair: define `trusted_row_sha256` as SHA-256 of one exact standalone
canonical trusted-row object including LF, with all eleven row keys and
`generated_source_hex`; state exact bytewise equality to the discovery row;
define the only permitted class-to-kind and generator-digest field mappings;
and require the origin's profile/ordinal/source length/hash, compile filename
and mode, generator qualname/digest, output root and row digest to reconstruct
one unique trusted row. Parse/re-encode equality and collision failure must be
mandatory.

## Round-4 closure and regression attack results

The parts of the repair not implicated by the four blockers are mechanically
coherent.

- The exact `PyPreConfig` constructor/default/final maps each have the same ten
  Linux header fields. The exact `PyConfig` constructor, discovery-final and
  full-runtime-final maps each have the same 64 header fields. The assignment
  partition is 37 scalar + 21 string + 5 list + constructor-owned
  `_config_init`, exactly 64. The twelve-step call sequence names
  `PySys_AddAuditHook`, `PyPreConfig_InitIsolatedConfig`, `Py_PreInitialize`,
  `PyConfig_InitIsolatedConfig`, ordered setters, no `PyConfig_Read`,
  `Py_InitializeFromConfig`, complete readback, one clear and fail-closed
  `PyStatus` handling. These field sets agree with the official
  [CPython 3.12.13 initialization documentation](https://docs.python.org/3.12/c-api/init_config.html)
  and exact
  [`initconfig.h`](https://raw.githubusercontent.com/python/cpython/v3.12.13/Include/cpython/initconfig.h).
- The initialization readback contains all 10 preconfiguration and 64
  configuration fields plus the closed global/sys/stream projection. The FD
  contract has exact 20-key launch rows, exact 15-key readback rows, seven
  full-runtime and four discovery descriptors, three ordered stages, closed
  extras, flags, offsets, sizes, seals, EOF, device and pipe state. Apart from
  A005-R5-B1's creation order and A005-R5-B2's missing twin evidence container,
  the declared states are unique.
- Installed-member association is now one standalone 16-key canonical
  JSON-plus-LF preimage. Raw ZIP-name ASCII/UTF-8 discrimination,
  wheel/install/repository equality, null tags, zero installer-generated rows,
  and the project installed-to-repository bijection are closed. Ten origin
  kinds other than `trusted-generated` have exact field schemas, value/tag
  equality and event-kind membership; no remaining installed-member blocker
  was found.
- The receipt inventory remains exactly 54 unique owners with class counts
  `17+27+1+1+3+1+1+3=54`. Independent counting reproduced exactly 106 direct
  upstream occurrences. Every upstream owner is earlier in the roster, so a
  topological walk consumes all 54 nodes without a missing, forward or cyclic
  edge. The exact 57 entrypoint descriptors are the 54-owner roster followed
  only by G5A-COMMIT, G5B-JOIN and K7-FINAL.
- NATP remains exact fourteen-key with one ordered K6-equal upstream. K1 keeps
  ordinal zero G0 and ordinal one G1. X0I/K3/K4, run upstreams, NATP scope/K6
  lineage, the twelve-class G5a root index and its self-cycle exclusions did
  not drift.
- The effective-contract index remains acyclic and non-self-hashing as a
  prospective construction: proposal, exact timestamped amendment pair and a
  future accepted Round-5 review pair, with no resulting contract digest in a
  constituent. Because this review is `REVISE`, that prospective index is not
  created and no fixed alias or rejected archive enters it.

The closure matrix is:

| Item | Round-5 assessment |
|---|---|
| A005-R4-B1 | **REVISE via A005-R5-B1** - all 10+64 CPython state and the FD schemas are closed, but the required memfd construction cannot execute in its written order |
| A005-R4-B2 | **REVISE via A005-R5-B2/B3** - the capsule DAG is acyclic, but its twin launch/readback evidence is not committed and its generated-event acceptance rule contradicts the exact runtime |
| A005-R4-B3 | **REVISE via A005-R5-B4** - installed associations are closed; the trusted-generated event origin alone lacks a typed row preimage/mapping |
| A005-R3-B2 | **CLOSED** - NATP exact14, ordered K6 equality, two K1 roles, 54 owners and 106 backward direct edges remain coherent |
| A005-R2-B1 | **CLOSED** - normalized SmoothL1, one-run bounds, ten traversal ledgers, exact three-run order and full 120-score replay are unchanged |
| A005-R2-B2 | **SPECIFICATION CLOSED; END-TO-END BLOCKED** - FE/MXCSR/FTZ/DAZ/denormal/backend controls remain exact, but a conforming provenance runtime cannot yet be constructed |
| A005-R2-B3 | **OPEN only through A005-R5-B2/B3/B4** - precommit-before-trace direction remains noncircular, but discovery evidence and generated-origin membership are not executable |
| A005-R2-B4/B5/B6/B7/B8 | **CLOSED** - inventory, direct run/G1/K1 lineage, NATP scope, K1 row preimage and all 57 current bindings passed |
| F4 | **CLOSED** - snapshot-before-validation direct `MappingProxyType` composition is unchanged |

A recursive comparison against the archived Round-4 JSON found changes only
in round/version/lineage/bindings, the intended CPython/FD/discovery/association
repair, required-test text and disposition. The score, checkpoint, tune input
and output, natural inference, K1/Cdev, target, prediction, capability/vault
order, F4 Mapping composition, 27 jobs, 93-hour allocation, data scope,
thresholds and claims did not drift.

The intended 16-member checkpoint; complete 120-checkpoint/output/evaluation
grid over one exact 56-view tune input; binary64 normalized-SmoothL1 Neumaier
reduction and deterministic selection; losing-artifact retention; 402-row
immutable K1 outcome; exact val/CERTIFIED Cdev projection with 108/8 floors;
seven-member target and ten-key receipt; 9,648 population-derived prediction
envelopes; K7 bytewise target rederivation; preconsume non-vault checks,
atomic consume, G5b vault validation and then K7; data firewall; and direct
MappingProxy snapshot-before-validation all remain structurally present. No
free score, winner-only provenance, caller relabel, late Cdev filter,
preconsume vault access, receipt-root swap or Mapping TOCTOU path was
reintroduced.

## Implementation, scope and authority ceiling

The current source and tests are a non-authorizing snapshot. They do not yet
implement the proposed native supervisor, CPython/FD/discovery ledgers,
association/origin schemas or runtime controls. This review ran only
read-only mechanical/schema/source checks, not a gate-bearing test suite or an
experiment. It accessed no server, natural data, sealed/heldout material,
evaluator vault, credential, GPU, checkpoint, prediction, target, training
entrypoint or result. It changed no amendment, proposal, plan, tracker,
MANIFEST, source, test, paper, Git, server, data or experiment artifact.

The required revisions are construction, evidence and serialization repairs
within the existing P06/runtime boundary. They must not add an experiment job,
model, algorithm, loss, parameter, seed, identity, dataset row, GPU hour,
metric, threshold, target/G5a receipt key, result or claim.

This same-family review is provisional and non-authoritative. It authorizes
nothing: `P0=0`, `P1=0`, `P2=0`, `P2-METRIC=0`, `P3=0`, `S0=0`, `gate=0`,
`capability=0`, `evaluator=0`, `launch=0`, `server=0`, `data=0`, `GPU=0`,
`training=0`, `test=0`, `result=0`, `paper/claim=0`, and `Git=0`.

Final disposition: `REVISE`. The exact `_20260816_173527` pair with hashes
`dbf797e80156d627b14290dfe558222878fdec8ccefbf28a051093130e1349e4`
and `94fc8e5db1031686ea0d5e5646c94606728c18cff19915a6deb231a3bf80cfcb`
cannot form an accepted effective contract. A newly versioned repair pair must
receive another normative review before any separately reviewed implementation
amendment; every execution and launch barrier remains zero.
