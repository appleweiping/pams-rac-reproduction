# TempoRAC Contract Amendment 005 Certificate Provenance - Round 8 Normative Review

Date: 2026-08-17  
Reviewer: `/root/temporac_certificate_amendment005_review`  
Route: Codex `gpt-5.6-sol`, OpenAI same-family, ultra review tier  
Assurance: provisional; not cross-family independent  
Verdict: **REVISE**  
Blocking groups: **4**  
Authority: **none**

## Outcome

The exact `_20260817_062337` Amendment 005 pair is materially stronger than
the rejected Round-7 pair, but it is not yet a uniquely executable,
crash-closed and substitution-closed provenance contract. Four blocking
groups remain:

1. the controller/native durable publication profile invokes an undefined
   common error cleanup, promises an unobservable cross-process permanent
   poison state, and mutates the root before outcomes that promise zero
   mutation;
2. the stage-input CAS publisher closes and then ambiguously changes a temp
   file's mode, cannot recover its own post-mode-change crash prefix or
   accepted `EEXIST` loser, and does not freeze the credentials/capabilities
   needed to mutate root-owned mode-0555 shards;
3. the ptrace policy validates a complete transition recipe only after a
   group has run, rather than checking the next ordered recipe row before
   each resume, while its supposedly complete returning-syscall evidence has
   no argument schema or FD/path state for reviewed passthrough syscalls and
   indirect executable-byte writes; and
4. the audit catalog uses `emitter` as a runtime selector even though the
   stock audit callback and the seven-key raw row expose no emitter or
   uniquely derived call-site state.

These are provenance implementability defects, not requests for more science,
jobs, data or claims. The verdict is `REVISE`. No prospective Round-8
effective-contract digest may be assigned. The accepted Round-6 effective
contract remains active.

## Reviewed bytes and terminal integrity

The fixed aliases and timestamped pair matched at the opening snapshot and at
the terminal rehash:

| Input | Bytes | SHA-256 |
|---|---:|---|
| Round-8 Markdown `_20260817_062337.md` | 1,638,523 | `6e0ce987d88c1d3f7abbaec47d28630efdc9a3012c7192da6b692feb9f55d981` |
| Round-8 JSON `_20260817_062337.json` | 1,613,302 | `b135fc10c01aa6611af131d44b5ca762d802df514184ba6c76dfe5b6c7a80ef0` |
| fixed Markdown alias | 1,638,523 | `6e0ce987d88c1d3f7abbaec47d28630efdc9a3012c7192da6b692feb9f55d981` |
| fixed JSON alias | 1,613,302 | `b135fc10c01aa6611af131d44b5ca762d802df514184ba6c76dfe5b6c7a80ef0` |
| archived rejected Round-7 Markdown `_20260817_053246.md` | 629,249 | `217ad51f4ea02f511777a63f0158f8f2864f4a786e201fa06cadbb5872ece751` |
| archived rejected Round-7 JSON `_20260817_053246.json` | 582,615 | `31c6ad83f61214b1f49b83321116bbe53ec8eb2fb5c463afd552e0ace418de17` |
| formal Round-7 review Markdown | 21,368 | `e18cdebf5b1860b3977622a83f94b77b40633190ccf14fb6cff25ff498ad6ea7` |
| formal Round-7 review JSON | 16,645 | `dfaed588a8eac96edf76a669565fb0c616232d57155e04b19da059c6d65eb078` |
| canonical proposal | 79,366 | `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391` |
| experiment plan | 34,198 | `4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e` |
| experiment tracker | 22,360 | `714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b` |

The current Markdown Appendix is byte-identical to the standalone JSON. Both
current files are strict UTF-8 without BOM, NUL or CR and have exactly one
terminal LF. Duplicate-key-rejecting parsing succeeds. JSON keys are
recursively UTF-8 bytewise ordered; no non-finite number or negative zero is
present.

All 76 declared binding records matched: 4 runtime/bootstrap files, 24 source
files, 19 tests and 29 planning/review or accepted-postfix records. All 49
declared lineage path/hash records matched at the opening snapshot. At the
terminal snapshot every immutable record still matched; the only expected
lineage mutation is this same run's authorized turn-008 trace-aggregator
append. No concurrent bound input, source, test, proposal, plan, tracker,
amendment, manifest or Git drift was observed.

I fully reread the project-local `research-review`, `research-refine`,
`experiment-audit` and `integrity-forensics` skills and all directly required
reviewer-routing, assurance, independence, experiment-integrity,
review-tracing, integration, output-composition, output-versioning,
output-manifest and output-language protocols. I then reread the complete
current amendment MD/JSON, the formal Round-7 review and trace, canonical
proposal, plan, tracker, all current bound source/bootstrap/test files and all
cited prerequisite artifacts. Author prechecks were treated only as
non-authoritative inputs.

## Blocking findings

### A005-R8-B1 - Controller durability is not a complete failure/recovery state machine

The primary effective rules are
`/round7_native_wave2_closure/controller/durability/write_algorithm`,
`root_poison`, `root_precondition`, `startup_scan`, `prefix_lattice` and
`eexist` (standalone JSON lines 7602-7774). The `EEXIST` branch has a concrete
winner-check, loser-temp unlink, directory-sync and absence proof. Every other
error, however, is delegated only to the words “follows exact cleanup.” No
referenced cleanup object defines the state, actions or postconditions.

This leaves non-equivalent implementations for errors after temp creation,
write, `fdatasync`, file `fsync`, close, non-`EEXIST` rename, directory
`fsync`, final reopen or readback. In particular, the contract does not say:

- which of `NO_TEMP`, `TEMP_OPEN`, `TEMP_CLOSED` or `FINAL_RENAMED` applies;
- the close/unlink/directory-sync/`ENOENT`-proof order and first-error rule;
- that a final already installed by successful rename must not be removed;
- how a reopen/readback failure is classified; or
- when every transient success-path FD is closed and proved `EBADF` before
  the exact pre-fork descriptor set is checked.

The temp grammar also does not require the two creation-time decimal fields to
equal the current `getpid()` and current process start-time. The native-build
content-store recovery profile at JSON lines 7035-7044 and 7119 inherits the
same undefined algorithm.

There are two additional contradictions inside the same state machine.
`root_poison` at line 7757 makes any cleanup failure permanently bar all later
requests until P00 redeployment, but `durable_paths` contains no durable
generation/poison marker. A fresh controller can see only a legal stale temp
and delete it or see an already clean root, so it cannot know that an earlier
process suffered a transient cleanup failure. Separately,
`root_precondition` at line 7758 removes stale temps before classifying the
current request, whereas `COMMITTED_NONREPLAYABLE` and `INVALID_PREFIX` at
lines 7661-7662 promise zero mutation. A valid stale temp for another request
makes both statements impossible simultaneously. A one-pass scan also does
not freeze whether an invalid entry aborts before later valid-temp deletions.

Minimal repair:

1. Add a closed, referenced cleanup state machine for
   `NO_TEMP`, `TEMP_OPEN`, `TEMP_CLOSED` and `FINAL_RENAMED`, freezing every
   close, unlink, sync, proof and first-error transition. Never delete the
   final after rename; close and `EBADF`-prove every transient FD.
2. Use a validate-only first pass over the complete deterministic root order,
   then a cleanup pass. State precisely whether committed/invalid outcomes
   mean no current-request publication or literally no root mutation.
3. Make a transient cleanup failure fail the current request and require a
   fresh idempotent rescan, reserving permanent poison for a persistently
   observable invalid root entry; otherwise add a P00-bound durable
   generation/poison journal.
4. Bind temp PID/start-time components to the actual creator and add crash and
   syscall-error injection at every publication boundary, including a fresh
   controller after cleanup failure and final-FD leak checks.

### A005-R8-B2 - Stage-input CAS publication has an unrecoverable prefix and no publisher authority model

The primary effective rule is
`/round7_native_wave2_closure/dispatch_abi/stage_input_authority_context_spec/cas_root_profile_spec/publication_contract/rule`
at JSON line 9494, repeated at 15681, 17638 and 18459. It specifies temp mode
0600, writes and syncs, **closes**, then says `chmod0444` before
`RENAME_NOREPLACE`. It does not identify `fchmod` versus a path operation, the
directory FD, object identity, `umask`, flags, EINTR/error behavior or a
metadata `fsync` ordering. A crash after the mode change and before rename
leaves a mode-0444 temp, while recovery recognizes only mode-0600 temps and
therefore poisons a valid protocol prefix.

When rename returns `EEXIST`, acceptance of the winning content-addressed
object does not require removing, syncing and proving absence of the current
publisher's now-mode-0444 loser temp. The root and shards are root:root mode
0555, but the contract does not close the controller/P00 `euid`, `egid`,
`fsuid`, `fsgid`, supplementary groups and capability lifecycle. An ordinary
publisher cannot create, rename or unlink there; granting mutation capability
without a mandatory pre-exec child drop would contradict handler-read-only
authority.

Minimal repair: freeze an exact same-shard-dirfd syscall sequence using
`openat(..., O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW, 0600)`, keep the
verified FD open through exact write/sync, apply `fchmod(fd, 0444)`, sync and
`fstat`, then `renameat2`, directory-sync, close and read-only reopen. Recovery
must validate both pre-mode-change and post-mode-change temp phases. Every
`EEXIST` and failure path must validate the winner, unlink only the exact own
temp, sync the shard, prove `ENOENT`, and revalidate the winner. Bind the
publisher credentials/capabilities and their non-inheritance/drop before any
child execution, or use a separately fixed writer-owned directory design.

The new 57-owner stage-input table/index design itself is otherwise closed:
57 unique recipes are in exact owner order, their canonical-LF hashes
rederive, their row counts total 1,034, all 35 validator literals match their
bindings, all 63 used content triples resolve, and the 20,156-byte content
registry hashes to
`a39425efbfcb00e19086c365204a1408f4ea4af90454bf9dcfe4366a014e803c`.
G5A/G5B/K7 use exact 8/12/10 rows and the same-instance K7 projection is
coherent. The blocker is the authority-bearing CAS publication/recovery
substrate, not the table's row algebra.

### A005-R8-B3 - Ptrace evidence can authorize execution before ordered recipe validation, and its syscall ABI is incomplete

The executable-map monitor is a substantial improvement over Round 7: the
parent owns ptrace, attaches before user execution, models tasks, stop-world
epochs, clone/terminal transitions, mapping states, providers and process
joins. Two independent gaps nevertheless keep the native transition claim
open.

First, `policy/group_recipe_rule` at JSON lines 8830 and 25331 and
`tasks_and_transitions/transition_rule` at lines 9165 and 25666 require only
that the eventual complete address-free group projection equal the
precommitted recipe. No rule maintains an expected-row cursor and compares
the next exact ordered effect while every task is stopped **before** the
tracee resumes. `monotonic_policy` at line 8751 may therefore accept an
individually reviewed provider/effect in the wrong group, order or
multiplicity, resume it into constructors or executable code, and reject only
at group EXIT. `PTRACE_SYSCALL` does provide an entry stop at which arguments
can be inspected before the syscall executes, so the missing pre-resume
comparison is implementable and necessary for the stated authority boundary
([Linux ptrace manual](https://man7.org/linux/man-pages/man2/ptrace.2.html)).

Second, `transition_rule` says every returning native syscall has a nonnull
exact `syscall_args`, while `/syscall_args` at lines 8921-8990 and
25422-25507 defines typed objects only for clone, exec, exit and direct mapping
calls. The constant-ledger table permits `REVIEWED_PASSTHROUGH`, but there is
no generic or per-syscall passthrough argument schema. The security prose also
claims to reject executable-byte writes through `/proc/<pid>/mem` or
`process_vm_writev`, yet the required syscall inventory and schemas omit the
open/openat/openat2, dup/fcntl/close, write/pwrite/writev and `io_uring`
families and define no tracee FD-to-path state. A four-key static syscall row
cannot prove that a write is harmless without knowing how its FD was opened.

Minimal repair:

1. While the world remains stopped, deterministically flatten each returning
   transition's effects and compare each byte-for-byte with the next
   unconsumed recipe row. Reject underflow, overflow, wrong order,
   duplication, null group or post-dispatch effect before any resume; advance
   the cursor only after validation and require exact exhaustion at EXIT.
2. Define an exact argument projection for every allowed returning syscall,
   including reviewed passthrough, or a closed generic six-register ABI plus
   the syscall-specific dereference rules needed for authority.
3. Add a context-sensitive FD/path lifecycle for open/dup/fcntl/close and all
   direct, vectored and asynchronous write surfaces, or reject those syscall
   families entirely. Test wrong-order/wrong-group transitions before native
   constructors and `/proc/*/mem` bypasses.

### A005-R8-B4 - Audit `emitter` is not runtime-observable authority

`/round7_native_wave2_closure/audit_and_trace/catalog/rules/2` at JSON line
5807 and its Round-8 mirror at 25708 select a definition using event name,
arity, exact Python type vector, **emitter**, current stage and outcome. The
raw row at lines 6506-6516 and the extension at 26277-26290 have seven keys:
definition digest, args, args hash, event, process ordinal, schema and
sequence. They carry neither emitter nor a call-site/return-PC state from
which emitter is uniquely derived.

This is not supplied by stock CPython. `Py_AuditHookFunction` receives exactly
`event`, `args` and the registration `userData`; it is not passed the source
emitter or call site ([CPython 3.12 C API](https://docs.python.org/3.12/c-api/sys.html#c.PySys_AddAuditHook)).
The precommitted source manifest proves which sites exist, but cannot tell
which site produced one runtime callback. Because definition and catalog
uniqueness explicitly include emitter, two different emitters may legally
share all observable selector values, so a producer and verifier cannot
select the same authoritative row or reject a cross-emitter substitution.

Minimal repair: remove emitter from runtime selection and require uniqueness
over the actually observable tuple `(event, arity, exact type vector, current
stage, outcome)`, retaining emitter only as verified static metadata; or add
closed precommitted call-site instrumentation whose exact runtime
`emitter_state` is included in the raw row and hash. Add cross-emitter
collision/substitution tests. The finite codec/dedicated projection registry,
four dedicated projection digests, 33-stage child vocabulary and
P00/P06/attestation chronology otherwise recompute and remain acyclic.

## Closure matrix

| Round-7 blocker | Round-8 status | Evidence |
|---|---|---|
| R7-B1 physical FD6 `FIONREAD` | **CLOSED** | One physical-entry sample occurs before FD6 output and must be zero; all three later rows are null and forbid `FIONREAD`; twin equality retains no scheduling value. |
| R7-B2 durable publication | **OPEN** (`A005-R8-B1`) | `EEXIST` improved, but general cleanup, cross-process poison, scan ordering and zero-mutation semantics are not executable. |
| R7-B3 stage-input authority | **OPEN** (`A005-R8-B2`) | The v7/v5 table/index and 57 recipes close the prior undefined-schema defect; their CAS publisher/recovery and privilege lifecycle remain open. |
| R7-B4 generated exec authority | **CLOSED** | Discovery rows have 13 keys including `exec_args_sha256` and `exec_code_authority_sha256`; trusted/origin rows have 14; adjacency and recursive code projection bind the consumed code object. |
| R7-B5 transient executable mappings | **OPEN** (`A005-R8-B3`) | Ptrace observes kernel transitions, but validation is not an online ordered pre-resume prefix and syscall/FD authority is incomplete. |
| R7-B6 audit projection registry | **OPEN** (`A005-R8-B4`) | The finite registry is present, but its emitter selector is not in the runtime evidence. |

## Regression and non-expansion checks

- BASE remains 19 unique roles. FINAL remains 24 unique roles and begins with
  byte-identical BASE19.
- The event-token universe remains exactly 16 unique values and origin union
  exactly 13 unique kinds.
- `PyPreConfig` and `PyConfig` remain exact 10- and 64-field ledgers.
- Dispatch order is 57 unique owners: the first 54 byte-equal the pre-G5a
  roster and the suffix is exactly `G5A-COMMIT`, `G5B-JOIN`, `K7-FINAL`.
- The pre-G5a receipt projection remains 54 unique nodes and 106 direct
  backward edges. It is acyclic, includes ordered K1 G0/G1, X0I/K3/K4 and the
  NATP/K6 lineage, and is not confused with the future 57-owner runtime ABI.
- The native controller failure vocabulary is now 46 unique values (33 child
  plus 13 controller), rather than Round 7's 45. The added
  `KERNEL_TRANSITION_MONITOR` value is a provenance-only enum expansion; the
  current scope guard, canonical proposal and science interface do not freeze
  45, so this is observed but not a blocker.
- Historical science remains typed to the bound CPython 3.12.4 environment;
  the provenance substrate remains separately typed to stock CPython 3.12.13
  and does not relabel historical science.
- Checkpoint/tune/score and CPU replay, K1/Cdev/population,
  prediction/target/K7, capability order, data firewall, G5a roots and F4
  MappingProxy snapshot semantics are byte-unchanged from the accepted
  science contract.
- Algorithm, thresholds, 27-job plan, 93-hour plan/100-hour ceiling, data
  scope, metrics, paper-visible blocks and claims are unchanged. No data,
  server, training, evaluator, GPU, experiment or gate was accessed or run.
- The active 665-byte `temporac.effective-contract-index.v4` rederives to
  `c203f4aab00879acf041b90829f0124d0b1de231294f191b03f3166f728e3d5c`
  and remains acyclic. A five-row prospective Round-8 recipe would be acyclic,
  but is deliberately unconstructed and unassigned under this `REVISE`
  verdict.

The current repository implements only the already reviewed pre-G5a
projection, not this prospective Round-8 native runtime. I made no code or
test changes and ran no gate-bearing tests or future runtime/build process.
Static specification checks do not confer implementation acceptance.

## Authority ceiling

This same-family provisional `REVISE` review authorizes no implementation,
rebind, effective-contract successor, test result, experiment or launch. All
authority remains zero, including `P0`, `P1`, `P2`, `P2-METRIC`, `P3`, `S0`,
G5a, G5b, K7, gate, capability, test, result, server, data, sealed data,
held-out data, GPU, training, evaluator, launch, Git, paper and claim
authority.

The next amendment should make only the minimal provenance repairs above,
preserve every closed science/job/data/claim surface, and receive a new
same-family review plus the required independent acceptance before any later
implementation or rebind can be authorized.
