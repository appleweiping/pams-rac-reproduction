# TempoRAC Contract Amendment 005 — Round-2 Normative Review

**Verdict:** `REVISE`

**Status:** `REVISE_SAME_FAMILY_PROVISIONAL_ZERO_AUTHORITY`

**Review date:** `2026-08-16`

**Reviewer:** `gpt-5.6-sol` / OpenAI / same-family provisional

The revised Amendment 005 closes the substantive population, prediction,
capability-order, all-checkpoint-retention, target-replay, and direct-Mapping
defects identified in Round 1. It does not yet define one executable byte-unique
certificate contract. Eight blocking specification defects remain: two tune
score inputs are not exact, the CPU lock omits active floating-point state, the
executed-code trace is circular under its literal scope, the pre-G5a receipt
inventory is incomplete, receipt dependency containers are underdefined, the
NATP scope/freeze lineage is outside the G5a commitment, a Cdev foreign-key
hash has no preimage, and two declared test bindings drifted during review.
Therefore the exact revised pair is not accepted.

## 1. Exact reviewed snapshot and byte validation

The fixed aliases and timestamped final pair were byte-identical at the start
of review and on the final rehash. The effective-contract design correctly uses
the timestamped pair, not the mutable aliases.

| Input | Bytes | SHA-256 |
|---|---:|---|
| final timestamped Amendment MD | 99,017 | `bd2c0772eb2f8ca70d23c6217b094d58ad69e1167486dbc811b7a4cec9d6e97c` |
| final timestamped Amendment JSON | 74,712 | `53f3409112550b79b6c11860cf1822e207a8bd600dd9708574ae977548282cd0` |
| fixed-alias Amendment MD | 99,017 | `bd2c0772eb2f8ca70d23c6217b094d58ad69e1167486dbc811b7a4cec9d6e97c` |
| fixed-alias Amendment JSON | 74,712 | `53f3409112550b79b6c11860cf1822e207a8bd600dd9708574ae977548282cd0` |
| archived rejected `20260816_110212` MD | 25,001 | `fa5ef054de0e243a030905b7a9d0e302aa442cb61b4d39012eaae6dca05a6869` |
| archived rejected `20260816_110212` JSON | 24,383 | `5ea02ae5ccb7ea0f1771222f254ded74318fdb4bc96a5655d5de58a66d7878a2` |
| canonical `FINAL_PROPOSAL.md` | 79,366 | `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391` |
| `EXPERIMENT_PLAN.md` | 34,198 | `4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e` |
| `EXPERIMENT_TRACKER.md` | 22,360 | `714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b` |
| certificate postfix Round-2 MD | 12,283 | `93a9cae771b34c185b9057286794b894d615aec5424011df145aaf8900f008d8` |
| certificate postfix Round-2 JSON | 14,527 | `144d2f22f4f75166b8cbefe54157d08690408979dd7cb34ef8c97e1738432872` |
| Amendment 005 Round-1 review MD | 17,993 | `35e7f3814759911da256605597a787e86ee5e199ecfded7cc1770a112ccea13d` |
| Amendment 005 Round-1 review JSON | 13,286 | `5ccd5c66c93f0791c74f5159d9daeaf2b7314c691dd2f25705b4888da565fa2c` |

The first complete rehash matched all 48 bindings embedded in the final
amendment—24 current source files, 19 current tests, and five planning/review
files. The mandatory final rehash found two concurrent test-byte changes:

| Drifted bound path | Amendment-bound SHA-256 | Final current SHA-256 | Final bytes |
|---|---|---|---:|
| `tests/temporac/test_gates_fixtures.py` | `69690aba7eb366471a28bb7406fd1411964d271ecd250834151243bb8a7c299f` | `d6d91a7dd606c42f672bb015d77c5c10406c2c6de69ed3a8bff76a0ce4f713d3` | 6,691 |
| `tests/temporac/test_operator_manifest.py` | `627e58cbb65be96de7d17cd56e8c12c179c2d1c3e11612f0aea5cf79eecdb5de` | `7f2c90c320d7378a74fcc64f6c808eb6e426cdcfbd73cab0fb1242f690db9db0` | 24,090 |

The current versions were fully reread. All other 46 declared bindings still
match: 24/24 source, 17/19 tests, and 5/5 planning/review. Both final JSON
copies parse under strict UTF-8 with no BOM, duplicate object key, non-finite
constant, or CR byte, and have one terminal LF. Their keys are recursively
sorted. The JSON appendix embedded in each final Markdown file is
byte-identical to the standalone final JSON. The archived rejected pair
remained unchanged and was not confused with the final pair.

## 2. Disposition of the seven Round-1 blockers

| Round-1 blocker | Round-2 result | Assessment |
|---|---|---|
| `A005-B1` score authority | `PARTIAL / REVISE` | One shared 56-view input, all 120 outputs, 120 evaluation receipts, the score/evidence indexes, and global K7 replay remove the free-scalar attack. The Huber function and typed X0 `run_bounds` remain non-unique. |
| `A005-B2` nonwinner artifacts | `SPECIFICATION CLOSED` | All 120 exact artifacts and member ledgers are retained through K7, strict-loaded, reserialized, and replayed; selection is not winner-only. |
| `A005-B3` effective contract/runtime closure | `PARTIAL / REVISE` | The five-row effective-contract index itself is acyclic and correctly excludes fixed aliases, but the CPU lock and executed-code trace do not yet determine executable bytes uniquely. |
| `A005-B4` G5a receipt DAG | `REVISE` | The root preimage and self-exclusions are sound, but the claimed closed node/edge inventory is neither satisfiable nor uniquely derivable. |
| `A005-B5` K1/Cdev provenance | `PARTIAL / REVISE` | K1 now freezes all 402 outcomes and Cdev is the exact val/CERTIFIED projection with 108/8 floors, but `k1_outcome_row_sha256` has no canonical preimage. |
| `A005-B6` component relabelling | `SPECIFICATION CLOSED` | Every one of 9,648 rows derives its component from the canonical population manifest and caller relabelling is forbidden. |
| `A005-B7` vault order | `SPECIFICATION CLOSED` | Non-vault prechecks precede atomic consumption; only the consumed process opens and validates the vault in G5b; K7 follows G5b PASS. |

The F4 closure is also present: the consumer snapshots a generic `Mapping` to a
plain dict before semantic validation, performs a guarded stability probe, and
never rereads the caller mapping.

## 3. Blocking findings

### A005-R2-B1 — tune score bytes are not mathematically unique

Amendment MD lines 88–98 and JSON lines 1507–1526 define reduction order but
name only “F11 Huber delta 0.05.” They never select the loss function. Current
`src/pams/temporac/teacher.py:115-133` uses the normalized SmoothL1 form
`0.5*e^2/delta` for `e<=delta`, otherwise `e-0.5*delta`. The commonly named
Huber form is `0.5*e^2`, otherwise `delta*(e-0.5*delta)`. At delta 0.05 these
differ by a factor of 20. Both satisfy the prose, but can change all 120
`mean_masked_reconstruction` and `tune_objective` values and the selected
checkpoint.

The same typed reconstruction has a second unresolved input. JSON lines
1372–1377 and 1550 store ten traversal bounds but never state the
`CertificateTrack.run_bounds` supplied to full `certify_target`. The canonical
adapter currently supplies one C-order `<i4` run `[[0,N_v]]`
(`certify.py:707-793`, specifically line 788). Supplying the ten traversal rows
as runs changes landmark discovery and can turn the same output into ABSTAIN.

Minimal revision: freeze the exact piecewise loss, comparison boundary, casts,
and binary64 operation order. Also state that each tune view has exactly
`run_bounds=<i4[[0,N_v]]>` derived from `sample_offsets`; the stored ten
`traversal_bounds[v]` must equal bounds rederived from that single run and are
never passed as `run_bounds`.

### A005-R2-B2 — the CPU replay lock omits active floating-point control

The closed environment schema at JSON lines 293–507, especially determinism
lines 361–372, frozen execution lines 449–467, and top-level keys 471–490,
does not record or set the process rounding mode, x86 FTZ/DAZ state, or Torch
flush-denormal state. It also does not freeze the relevant CPU backend switches
such as oneDNN/MKLDNN enabled/deterministic behavior immediately before replay.
Those are mutable process state, not implied by wheel, ELF, CPU, ISA, thread, or
environment-variable hashes.

A read-only local witness under the installed CPU Torch build produced
`4.999999675228202e-39` for float32 `1e-38*0.5` with flush-denormal disabled and
`0.0` with it enabled while every currently listed lock field remained the
same. Thus the claimed bytewise float32 forward and binary64 score replay is
not a function of the environment artifact.

Minimal revision: add one closed floating-point-control record, set and read
back `FE_TONEAREST` and the platform FTZ/DAZ/flush-denormal state, freeze the
applicable CPU backend/fpmath switches, and verify them immediately before
numeric replay in a fresh subprocess. Unsupported or unreadable control state
must fail closed.

### A005-R2-B3 — the executed-code trace is circular under its literal scope

MD lines 46–48 and JSON lines 564–647 require `trace_sha256` inside the code
index and describe a fully instrumented dynamic `open/mmap` trace across P3,
training, tune, prediction, G5a, G5b, and K7. The environment and downstream
artifacts/receipts in turn contain `code_index_sha256` (for example JSON lines
469–490). On the literal rule, an ordinary G5a/K7 open of an environment,
index, receipt, prediction, or target is a trace event whose digest depends on
bytes already containing the code-index digest:

`code index -> trace digest -> opened downstream bytes -> code index`.

If only executable-origin opens were intended, the contract still lacks a
closed operation enum/filter, exact entry-point inventory and arguments,
subprocess propagation, origin/path normalization, capture boundary, and the
point at which the trace is final. A verifier cannot decide whether a data
open is required, forbidden, or ignored.

Minimal revision: scope the trace only to bytes that become executable or
interpreted (imports/loader source, `exec`/`eval`, and `ctypes`/`dlopen`),
explicitly exclude ordinary artifact/data/receipt I/O, freeze exact operation
tokens, entry points, normalization, subprocess coverage, and failure rules,
and complete the discovery trace/code-index precommit before any downstream
object contains its digest. Runtime monitoring may enforce membership but must
not retroactively change the committed trace.

### A005-R2-B4 — the claimed pre-G5a receipt inventory is incomplete

JSON lines 912–924 declare the exact class multiplicities, but omit all three
X0I inference shard/completion receipts and the K3 and K4 receipts. The tracker
lines 46–51 requires `X0I x3 -> K3 -> K4 -> G4`; the plan lines 143 and
153–155 and proposal lines 1400–1404 freeze the same F23 precedence. Yet JSON
line 974 requires each pre-G5a gate to point to its exact plan upstream, so G4
must depend on K4 while the closed inventory has no K4 node. Under the stated
missing/extra-node failure rule, no conforming DAG can be built.

The fallback `pre-g5a gate/fixture` multiplicity is not mechanical. It says
“each ... receipt required by the effective plan” rather than giving a numeric
per-owner inventory; JSON lines 956–963 choose “exact schema or gate-name”
without selecting one token or binding that owner into a closed receipt
payload. The plan uses `P0/P1/P2/P2-METRIC/P3`; the tracker uses
`P00/P01/.../P06/P05M`. No alias table, exact receipt schema/key set, or
per-owner count lets a verifier reject omission, duplication, or cross-owner
digest substitution.

Minimal revision: add one canonical pre-G5a receipt-inventory table with, for
every owner, one exact owner token, receipt schema and closed keys, numeric
count or closed derivation, payload-to-owner equality, and ordered upstreams.
It must include the three X0I receipts and exact K3 and K4 nodes/edges before
G4, without adding jobs or changing F23.

### A005-R2-B5 — direct receipt dependencies and the three-run G1 binding are undertyped

The proposal lines 1232–1239 makes every run receipt carry an ordered
`upstream_receipt_sha256` array. DAG completeness at JSON lines 903–905 says
edges derive from receipt-payload foreign keys, but required edges and the
closed role-token list at lines 965–994 contain no `run.upstream` role. Either
the direct dependencies are silently omitted, contradicting completeness, or
the necessary edge has a forbidden role. The plan does not provide an exact
27-row array/order substitute.

Separately, the ten-key G1 receipt and checkpoint-evidence index use singular
`teacher_run_receipt_sha256` at JSON lines 1073–1088 and 1138–1158 while their
normative text and required G1 edges require all three teacher runs. No value
type, exact length-three rule, seed order, or aggregate hash preimage is
defined. Multiple byte encodings satisfy the prose.

Minimal revision: add `run.upstream`, generate the exact ordered run edges from
each validated 27-job payload, and freeze each job's upstream receipt source
and order. Define the existing G1/evidence field as either an exact three-item
digest array in seed order `20260815,20260816,20260817` or as one explicitly
tagged aggregate preimage; use the same construction everywhere.

### A005-R2-B6 — NATP scope and post-K6 freeze lineage are outside G5a

The exact G5a root-index class list at JSON lines 181–207 contains nine classes
and no pilot-scope/NATP completion index. The tracker lines 73–76 says each of
three NATP rows emits prediction roots plus a surrounding scope index and G5a
requires the scope manifest; the plan line 168 likewise requires it. Those
future bytes are therefore neither a G5a root-index row nor a receipt-DAG node.
The prediction edges point to feature and response run only, so nothing binds
the natural freeze to the required K6 predecessor in F23.

Minimal revision: define an exact canonical scope/freeze index (or exact three
NATP completion receipts) binding the canonical pilot-scope manifest, each
seed's clean/drift prediction roots, and the exact K6 receipt. Add its typed
class/count/owner/edges and bytes/hash to the G5a root index. This is provenance
only; it must not add a job, dataset row, metric, claim, or authority.

### A005-R2-B7 — `k1_outcome_row_sha256` has no byte preimage

The certified-development row schema includes `k1_outcome_row_sha256` at JSON
lines 103–130, but no amendment rule defines the bytes it hashes. In contrast,
JSON line 752 expressly defines `population_row_sha256` as the standalone
canonical eight-key row bytes including LF. Hashing an embedded row, a
standalone 13-key row plus LF, or a tagged row preimage yields different valid
digests. Thus the intended exact Cdev-to-K1 row association is not uniquely
verifiable even though the Cdev set/projection semantics are otherwise right.

Minimal revision: define one exact K1 row hash preimage—preferably the complete
standalone canonical 13-key K1 row bytes including one LF—and require each Cdev
foreign key to equal that digest for the exact projected row. Alternatively,
replace the hash association with an exact ordinal plus byte equality, but do
not leave both encodings possible.

### A005-R2-B8 — the declared review snapshot drifted before persistence

The amendment's non-authorizing binding map names exact current source/test
bytes and the requested review requires those bytes to be stable through final
rehash. The two test files listed in Section 1 changed after their initial read
and before persistence. They now exercise the separately evolving operator
candidate, including Amendment 006 material, and no longer equal the hashes
inside this exact Amendment 005 JSON. The reviewer did not modify either file.

This does not create execution authority and does not weaken the seven semantic
findings above. It does mean the statement that the final amendment's complete
current test snapshot identifies the bytes reviewed is false at persistence;
an exact-byte review cannot silently mix the old bound tests with the new
working-tree tests.

Minimal revision: after the workspace is quiescent, issue a newly versioned
Amendment 005 pair whose non-authorizing source/test bindings match the intended
review snapshot, then keep every bound byte stable through the fresh review.
If the changed operator tests are intentionally outside Amendment 005, freeze
and state that exact historical-snapshot policy instead of calling them the
current snapshot.

## 4. Positive attacks and exact arithmetic

- The 16 checkpoint names, shapes, and `<f4` dtypes exactly match the current
  `TempoRACTeacher.state_dict()`. Raw payload is 1,008,348 bytes. The current
  deterministic writer produced 1,012,518 bytes, below the 2,097,152-byte cap.
- The shared tune input has the exact seven members, 56 views, 25,680 samples,
  and 25,624 edges. Its raw payload is 22,592,544 bytes; the current writer
  produced 22,594,258 bytes, below the 33,554,432-byte cap.
- Each tune output has two members and raw payload 15,510,720 bytes; the current
  writer produced 15,511,204 bytes, below the 16,777,216-byte cap.
- The exact 9-key tune-input receipt, 13-key tune-evaluation receipt, 8-key
  score row, and 11-key selection receipt sets were checked. Apart from the
  explicitly reported Huber/run-bound and three-run-container gaps, their
  transitive checkpoint/input/output/score bindings are coherent.
- Natural inference first derives per-run integer landmarks from geometry
  alone; a run with fewer than three landmarks terminates before F5 or teacher
  access. The exact F5 vector is then accumulated in the prescribed binary64
  Neumaier order, cast once to C-order float32, and passed with the exact
  215-channel sample matrix to one CPU forward through the typed
  `SelectedTeacher`. Caller phase/reconstruction/static/bounds injection is
  excluded.
- All 120 checkpoint artifacts, receipts, raw output NPZs, and evaluation
  receipts are required. Every score is rederived over the same 56 views;
  binary64 bits, per-seed precedence, and global precedence are replayed. No
  free scalar and no winner-only authority path remains in the written design.
- Score-index checkpoint/evaluation references are separately committed by
  the score and evidence indexes and checked by the global replay. Treating
  those as root-bound cross-index associations rather than duplicate DAG edges
  does not itself permit a substitution; direct run payload foreign keys still
  require the `run.upstream` repair in B5.
- The five-row effective-contract index—proposal, exact timestamped MD/JSON,
  eventual accepted review MD/JSON—has no contract field and none of its
  constituents contains the resulting digest. Its construction is acyclic.
  Because this review is `REVISE`, no accepted review row and no effective
  contract digest exists for this version.
- The G5a receipt-root formula itself is noncircular: it hashes the exact root
  index, includes receipt-DAG bytes/hash, and excludes the containing G5a,
  request/grant, G5b, K7, metric, and future receipts. The blocker is the
  incomplete/underdetermined inventory, not a self-cycle in that formula.
- K1 is exactly 402 immutable rows before G5a. Cdev is exactly the ordered
  val/CERTIFIED projection of the 134 development identities, subject to the
  canonical 80-percent rules and retained floors of at least 108 identities
  and eight components. It is not all 402 and is not a caller-chosen subset.
- Population/prediction/vault semantics are exact: P402 is 268 train plus 134
  development; `402*4*3*2=9,648`; each component comes from the population
  manifest; G5b validates an exact vault bijection and derives 402. Metrics
  remain over all 134 development identities including stubs.
- Capability order matches F23: non-vault checks, atomic durable consumption,
  same-process vault validation/join, then K7. No pre-consume vault operation
  is permitted and failure cannot restore the grant.
- K7 keeps the existing seven-member target NPZ and ten-key target receipt,
  performs selection replay once, then bytewise rederives natural input, full
  certificate, target artifact, and receipt for every Cdev row. Digest-only
  target acceptance is forbidden.
- F4 snapshots to plain dict before validation, stability-probes the mapping,
  validates and canonicalizes the same snapshot, and supports direct
  `MappingProxyType` composition.

These archive-size probes are synthetic and non-authorizing. No gate-bearing
test suite was executed. The current code/tests do not implement the proposed
schemas and are not accepted by this review.

## 5. Scope, firewall, and authority

The revision does not add or change an algorithm, trainable component, data
scope, identity, arm, seed, condition, job, checkpoint cadence, GPU hour,
resource ceiling, metric, threshold, claim, target receipt key, G5a receipt
key, or paper-visible block. The reported fixes are provenance/serialization
closure only. This review accessed no server, natural data, evaluator vault,
credential, GPU, or training entry point, and modified no code, test, proposal,
plan, tracker, manifest, Git state, server, data, or paper artifact.

The reviewer itself modified no code or test; the two binding changes were
concurrent shared-worktree drift detected by the required final rehash.

This same-family review is provisional and non-authoritative. It authorizes
nothing: `P0=0`, `P1=0`, `P2=0`, `P2-METRIC=0`, `P3=0`, `S0=0`, `gate=0`,
`capability=0`, `evaluator=0`, `launch=0`, `server=0`, `data=0`, `GPU=0`,
`training=0`, `result=0`, `paper_claim=0`, and `Git=0`.

Final disposition: `REVISE`. The exact timestamped final pair with hashes
`bd2c0772eb2f8ca70d23c6217b094d58ad69e1167486dbc811b7a4cec9d6e97c`
and `53f3409112550b79b6c11860cf1822e207a8bd600dd9708574ae977548282cd0`
cannot form an accepted effective contract. A newly versioned exact repair pair
must receive a fresh normative review before any separately reviewed
implementation amendment; all execution and launch barriers remain zero.
