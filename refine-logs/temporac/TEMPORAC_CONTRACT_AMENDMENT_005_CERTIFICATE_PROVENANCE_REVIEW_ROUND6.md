# TempoRAC Contract Amendment 005 Certificate Provenance - Round 6 Normative Review

Date: 2026-08-16  
Reviewer: `/root/temporac_certificate_amendment005_review`  
Route: Codex `gpt-5.6-sol`, OpenAI same-family, ultra review tier  
Assurance: provisional; not cross-family independent  
Verdict: **ACCEPT**  
Blocking groups: **0**  
Authority: **none**

## Outcome

The exact `_20260816_184303` Amendment 005 pair closes every Round-5 blocker.
The sealed-memfd construction is now executable and fully observed; generated
source discovery now has two independently launched, five-frame, joined
evidence streams; the trusted set is the exact common nonempty ordered result
of those discoveries and is consumed only by adjacent dedicated compile/exec
pairs; and every trusted-generated origin reconstructs one exact standalone
eleven-key row-plus-LF preimage.

I found no new omission, substitution, ordering, cycle, authority or science
blocker. The disposition is `ACCEPT` for this specification pair. This is not
implementation acceptance and creates no execution authority. It permits only
a later, separately reviewed implementation amendment to target the accepted
specification.

## Reviewed bytes and terminal integrity

The exact inputs matched both the opening and terminal rehash:

| Input | Bytes | SHA-256 |
|---|---:|---|
| timestamped Round-6 Markdown `_20260816_184303.md` | 312,728 | `2eaa3cfb746804fdc5890fec4c252c877a031a4abedef7b9ef6eeac0a4839395` |
| timestamped Round-6 JSON `_20260816_184303.json` | 261,196 | `638efc8708f74c8514247f4f4c5e2311595e90edc6744c9b371886f0789af164` |
| fixed Markdown alias | 312,728 | `2eaa3cfb746804fdc5890fec4c252c877a031a4abedef7b9ef6eeac0a4839395` |
| fixed JSON alias | 261,196 | `638efc8708f74c8514247f4f4c5e2311595e90edc6744c9b371886f0789af164` |
| archived rejected Round-5 Markdown `_20260816_184302.md` | 269,255 | `dbf797e80156d627b14290dfe558222878fdec8ccefbf28a051093130e1349e4` |
| archived rejected Round-5 JSON `_20260816_184302.json` | 226,592 | `94fc8e5db1031686ea0d5e5646c94606728c18cff19915a6deb231a3bf80cfcb` |
| Round-5 review Markdown | 19,671 | `ca3807657ee4197996c49c270cabd73bb43e439422173ca780849f433113a6f7` |
| Round-5 review JSON | 17,110 | `8f2490fcfe8434e1d3f2adc3a2a5db7afb4166302f421fa06e45512b764cf2ff` |
| canonical proposal | 79,366 | `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391` |
| experiment plan | 34,198 | `4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e` |
| experiment tracker | 22,360 | `714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b` |

The fixed aliases are byte-identical to the timestamped pair, and the
`_20260816_184302` archive retains the rejected Round-5 bytes. Markdown
Appendix A is byte-identical to the standalone current JSON. Both files are
strict UTF-8 without BOM or CR and have exactly one terminal LF. Duplicate-key
rejecting parsing succeeds; all object keys are recursively UTF-8 bytewise
ordered; no non-finite number or negative zero occurs.

All 57 amendment-declared current bindings matched at the opening and terminal
snapshots: 3 runtime-bootstrap files, 24 source files, 19 tests and 11
planning/review files. No bound-input drift was observed.

I fully reread the project-local `research-refine`, `research-review` and
`experiment-audit` skills and all directly required reviewer-routing,
assurance, independence, tracing, output, scope, integration and experiment
integrity protocols. I then reread the complete current amendment MD/JSON,
the Round-5 review pair and trace, canonical proposal, plan, tracker, current
57 bindings and cited prerequisite material. Author summaries and intermediate
candidate bytes were not used as authority.

## A005-R5-B1 - closed: sealed memfd construction and handoff

The construction at JSON lines 1278-1392 keeps the writable writer open while
it populates, verifies and applies exactly the four seals. It then opens the
exact `/proc/self/fd/<writer_fd>` path once with `O_RDONLY|O_CLOEXEC`, promotes
the returned reader, and proves that writer and reader name the same regular
anonymous inode but have distinct open-file descriptions. The sole ordered
offset test makes a duplicated/shared OFD fail.

Only after those checks does it close the writer, require `F_GETFD` to fail
with `EBADF`, and repeat the reader's type/device/inode/size, status flags,
descriptor flags, seals, offset-zero, payload-hash and EOF checks. `dup3` then
installs the read-only reader at the prescribed child FD with descriptor flags
zero. The writer and all transient descriptors must be absent from the child.

The exact eight construction-stage rows, thirteen-key stage schema, fourteen
failure tokens, first-failure/errno rule, launch-instance binding and child
readbacks make reordering, dup-of-writer, writable input, shared offset,
wrong inode/seal/EOF, post-close proc reopen or surviving writer detectable.
This closes the impossible close-before-reopen order from Round 5.

## A005-R5-B2 - closed: two authority-bearing discovery runs

JSON lines 1977-2140 define one exact two-row ordered discovery-evidence index.
Run 0 and run 1 are sequential fresh exec processes with distinct
`(child_pid, child_starttime_ticks)` identities and independent newly created
FD3 pipes. The controller fully drains, validates, observes EOF, joins and
commits run 0 before any run-1 process or pipe is created. Process reuse,
overlap, shared framing or output-only evidence fails.

Each child emits exactly the fixed ASCII magic, NUL, big-endian count five and
five ordered frames. Every frame contains its type, big-endian payload length,
raw SHA-256 and complete canonical JSON-plus-LF payload. The five payloads are
the launch instance, exec-entry FD readback, locked-pre-CPython FD readback,
post-initialization FD readback and generated-source output. Per-payload and
whole-stream caps are fixed before allocation; truncation, trailing data,
alternate endian, duplicate/reordered type, wrong digest/schema or missing EOF
fails.

Each evidence row has exactly 23 keys and commits the complete five payloads,
their hashes, the stream byte count/hash, process identity and joined exit.
Each joined attestation has exactly 22 keys and repeats the launch, complete
10+64 configuration readback, three FD-stage readbacks, output/root, capsule,
precommit, stream and process/join state. Equality rules normalize only the
explicit run-dependent fields; all other authority state must be byte equal.

The complete evidence-index bytes receive both a direct SHA-256 and the tagged
discovery-evidence root. The exact trusted index and final CPU environment bind
those digests together with the common output root. No evidence digest depends
on the later final environment, code index or runtime trace, so the chain
`capsule -> precommit -> two discoveries -> evidence/trusted index -> final
environment -> full execution trace` is acyclic.

## A005-R5-B3 - closed: complete generated-source consumption

The trusted rows are now exactly the common, byte-for-byte equal, finite
nonempty row arrays emitted by the two discovery processes. Empty, predicted,
caller-supplied, truncated, coalesced, duplicated, project/site-packages or
late-learned sets fail.

Every full runtime imports exact `dataclasses` in the bootstrap region and
consumes ordinals `0..R-1` once in order. For each row the next anonymous
source action is `PY_TRUSTED_GENERATED_COMPILE`, immediately followed by one
`PY_TRUSTED_GENERATED_EXEC` carrying the byte-identical origin. Only that pair
consumes the row. Generic `PY_CODE_EVAL`/`PY_CODE_EXEC`, code-object-only,
marshal/pickle, missing/extra/reordered/second consumption, project or
third-party generation, and every generated event after `DISPATCH_BEGIN`
fail.

The pairing is mechanically identifiable: the compile event reconstructs the
next unconsumed row from its exact source bytes, filename, mode, generator,
profile and ordinal; the immediately adjacent exec must reuse the exact origin
object/digest before controller state advances. There is no caller relabel or
unordered matching choice.

## A005-R5-B4 - closed: unique trusted-row and origin preimages

JSON lines 2155-2173 freeze an exact standalone row containing precisely the
eleven trusted-row keys, including `generated_source_hex`. It is serialized as
canonical compact recursively sorted-key UTF-8 JSON followed by exactly one LF;
`trusted_row_sha256` hashes all of those bytes. It has no embedded digest,
schema wrapper, parent/root bytes, ordinal prefix or tagged framing.

Trusted, discovery-run-0 and discovery-run-1 rows must parse to the same value
tree and recanonicalize to the same bytes, and the row digest is unique in the
ordered nonempty index. The trusted-generated origin at JSON lines 2479 and
2630-2644 resolves the committed discovery root and this row digest, then
recomputes the row bytes, decoded source length/hash and every field. The only
class mapping is `cpython-stdlib-source -> cpython-stdlib`; the generator digest
is copied directly, not renamed or rehashed, and must resolve one exact
CPython runtime stdlib-source member.

The full one-to-one projection covers filename, mode, generated-source hash,
generator qualname, ordinal, profile, source length, mapped generator kind and
generator digest. Wrong root/row, alternate JSON, omitted source hex, field
swap, ambiguous row, kind remap, digest substitution or collision fails. The
origin is therefore reconstructible and authority-bearing without circularity.

## Regression and mechanical attack results

The complete regression pass found no reopened blocker:

- The exact PyPreConfig maps contain all 10 Linux fields. The exact PyConfig
  maps contain all 64 fields. The assignment partition remains 37 scalar + 21
  owned string + 5 owned list + constructor-owned `_config_init`. Constructor,
  setter, status, no-`PyConfig_Read`, pre-initialize audit-hook, full readback,
  clear and failure paths remain closed.
- The FD contracts retain seven full-runtime descriptors, four discovery
  descriptors, exact OS object classes, access/status/descriptor flags,
  offsets, sizes, seals, EOF, pipe state, inheritability, three child readback
  stages and rejection of every extra FD. The B1 repair changes construction
  order only; it does not relax the child table.
- Installed-distribution, CPython-runtime, native/ELF, project-install to
  repository, bootstrap capsule and trusted-source ledgers retain typed
  canonical preimages, unique normalization/discriminants and online
  post-run-trace membership. The post-run trace is evidence only and cannot
  mutate a precommitted authority index or form a cycle.
- The receipt roster remains exactly 54 unique owners with class counts
  `17+27+1+1+3+1+1+3=54`. Independent generation gives exactly 106 direct
  edges: 58 generic pre-G5a stage edges, 30 run edges, 3 G1 teacher-run edges,
  2 ordered K1 edges, 3 X0I edges, 3 K3 edges, 4 K4 edges and 3 NATP-to-K6
  edges. Every edge points to an earlier owner and a topological walk consumes
  all 54 nodes.
- NATP remains an exact fourteen-key receipt with its sole ordered upstream
  equal to K6. K1 retains `k1.upstream` ordinal 0 = G0 and ordinal 1 = G1.
  X0I/K3/K4, run.upstream, scope/K6/prediction lineage, the twelve-class G5a
  root index, multiplicities and exclusions remain complete and noncircular.
- Checkpoint/tune receipt schemas remain exact 9/13/8/11 keys. The full 16-member
  checkpoint, exact 56-view tune input, all 120 checkpoint artifacts, raw output
  NPZs, evaluations and score rows remain retained and cross-bound. The
  normalized SmoothL1 piecewise formula, one-run `[[0,N_v]]` bounds, ten
  traversal ledgers, three-run order, binary64 operation order, Neumaier
  reduction and deterministic selection remain byte-unique with no free score.
- FE rounding, MXCSR/FTZ/DAZ, denormal probes, CPU backend controls, strict
  checkpoint load and fresh CPU forward remain fixed. The 402-row K1 outcome
  index and row preimages precede G5a; Cdev remains exactly the 134-row val and
  CERTIFIED projection with K1 floors 108/8, not all 402 rows.
- All 9,648 prediction envelopes bind their component token through the
  canonical population manifest; the seven-member target and unchanged ten-key
  receipt, G5a root, G5b vault join and K7 bytewise target rederivation retain
  their exact schemas. Capability order remains preconsume non-vault checks,
  atomic consume, G5b vault validation/join, then K7, with no preconsume vault
  access.
- F4 remains snapshot-before-validation followed by direct immutable
  `MappingProxyType` composition. Data firewall, UTF-8/JSON rules, duplicate-key
  rejection, algorithm, jobs, 93-hour resource ceiling, data scope, thresholds,
  target/G5a keys, paper-visible blocks and claims did not drift.

The closure matrix is:

| Item | Round-6 assessment |
|---|---|
| A005-R5-B1 | **CLOSED** - executable writer-open seal/reopen, distinct-OFD proof, writer EBADF, post-close reader validation and exact child install |
| A005-R5-B2 | **CLOSED** - two fresh sequential processes, independent five-frame streams, exact 23/22-key evidence/attestation, complete SHA/tagged-root/environment binding |
| A005-R5-B3 | **CLOSED** - exact common nonempty ordered rows, one adjacent dedicated compile/exec consumption per ordinal, no generic or post-dispatch path |
| A005-R5-B4 | **CLOSED** - standalone eleven-key-plus-LF row digest, unique row resolution and complete fixed origin projection |
| A005-R4-B1/B2/B3 | **CLOSED by the Round-5 and Round-6 repairs** |
| A005-R3-B1/B2 and A005-R2-B1..B8 | **CLOSED; no regression** |
| F4 | **CLOSED; no regression** |
| Science/jobs/data/claims | **UNCHANGED** |

The prospective effective-contract index remains acyclic and non-self-hashing:
it will contain the canonical proposal, this exact timestamped amendment pair
and the eventual accepted Round-6 review pair. None contains the resulting
effective-contract digest. The digest may be constructed only after this
review pair is finalized and byte-validated, and every later artifact must use
that one digest. Fixed aliases and rejected archives are not constituents.

## Implementation, scope and authority ceiling

The current source and tests do not implement this future native supervisor,
discovery/evidence runtime or complete provenance contract. This review accepts
the normative specification only. It ran read-only schema, byte, source and
mechanical checks; it did not execute a gate-bearing suite or experiment.

No server, natural/test/sealed/heldout data, vault, credential, GPU, checkpoint,
prediction, target, training entrypoint or result was accessed. No amendment,
proposal, plan, tracker, MANIFEST, source, test, paper, Git, server, data or
experiment artifact was modified. The repair adds no job, model, algorithm,
loss, parameter, seed, identity, dataset row, compute hour, metric, threshold,
result or claim.

This same-family review is provisional and non-authoritative. `ACCEPT`
authorizes only a later separately reviewed implementation amendment. It does
not itself authorize implementation or execution: `P0=0`, `P1=0`, `P2=0`,
`P2-METRIC=0`, `P3=0`, `S0=0`, `gate=0`, `capability=0`, `evaluator=0`,
`launch=0`, `server=0`, `data=0`, `GPU=0`, `training=0`, `test=0`, `result=0`,
`paper/claim=0`, and `Git=0`.

Final disposition: `ACCEPT` for the exact 312,728-byte Markdown
`2eaa3cfb746804fdc5890fec4c252c877a031a4abedef7b9ef6eeac0a4839395`
and 261,196-byte JSON
`638efc8708f74c8514247f4f4c5e2311595e90edc6744c9b371886f0789af164`,
subject to same-family provisional assurance and the zero-authority ceiling
above.
