# TempoRAC handoff — 2026-08-17

This repository snapshot is a fail-closed research handoff. It is not a claim that the experiment gates, server runs, paper results, or publication release are complete.

## Repository and branch

- Repository: `appleweiping/pams-rac-reproduction`
- Working branch: `codex/aris-fullrun-icassp27`
- Intended base: `main`
- All ARIS semantic participants used for this work were OpenAI `gpt-5.6-sol`; no Claude reviewer was used.
- The repository is public. Local ignored datasets, checkpoints, `.env` files, private author metadata, and other ignored secrets are intentionally not part of this handoff.

## Scientific idea

TempoRAC counts repetitions from supplied identity pose tracks under irregular clocks and time reparameterization. It uses a shared encoder, slow/medium/fast sigmoid response experts, a detached irregular-clock NUDFT cue, per-identity/window soft routing, positive NOLA synthesis, and exactly one threshold-0.5 connected-component decode per identity. Count, period, density, and boundary labels are not learner inputs. WARP-PHASE and earlier pivots are preserved as research history, not the current mainline.

## What is complete

- Canonical TempoRAC proposal and experiment plan are present under `refine-logs/` and `refine-logs/temporac/`.
- X0 amendment and pinned candidate replay were independently accepted at candidate-only authority.
- Operator amendment/candidate v2 was independently accepted at candidate-only authority.
- P05M Amendment 003 reached a same-family provisional acceptance at the normative layer.
- Wave 0/1 Python contract work is implemented in `src/pams/temporac/`.
- Wave 1 postfix independent review passed:
  - review MD: `7cf1f1a013932a7f9317f094d28dc54af9ab0da7adb6caab4a5964941231fa3b`
  - review JSON: `bdec8e1f13023e47f2be85010b7cdae09a1e219c106956a24980295bc9ce533e`
  - full TempoRAC tests at that freeze: 134 passed, 1 Windows symlink-permission skip
  - Ruff passed; production/scoped Mypy passed
- The effective contract currently active in code is still the 665-byte Wave 1 index with SHA-256 `c203f4aab00879acf041b90829f0124d0b1de231294f191b03f3166f728e3d5c`.

## Current stop point

No P0/P1/P2/P3/S0/G/K gate, server experiment, paper-result promotion, launch, or GitHub-main authority has been granted.

Amendment 005 Round 8 fixed aliases remain the formally rejected Round 8 candidate:

- fixed MD: `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE.md`
  - 1,638,523 bytes
  - SHA-256 `6e0ce987d88c1d3f7abbaec47d28630efdc9a3012c7192da6b692feb9f55d981`
- fixed JSON: `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE.json`
  - 1,613,302 bytes
  - SHA-256 `b135fc10c01aa6611af131d44b5ca762d802df514184ba6c76dfe5b6c7a80ef0`
- formal Round 8 review: `REVISE`
  - MD SHA-256 `27e7f6080b5af427257b5cbf0bae70cd2121e43e9801df69628720031c9c5165`
  - JSON SHA-256 `71453a8db8eaaa9c1253e8bb9467ae2a8d8d915c585eda08e9fc0839b5e1038c`

Round 9 exists only as an unfinished timestamp candidate and must not be copied to the fixed aliases yet:

- `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260817_134853.md`
  - 2,048,872 bytes
  - SHA-256 `7da6647f7c37f981dd516d7b152123a156b206732a77a1ee82c7a3e492ab77bb`
- `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260817_134853.json`
  - 2,021,793 bytes
  - SHA-256 `4cff6d4ac91c0114093ee6aeed0247eb2fcda2bc6f4192c266ccfe67977ab898`

The Round 9 author/reviewer work was interrupted deliberately for this handoff. Do not infer acceptance from file size, presence, or local strict checks.

## Exact remaining Round 9 issues

Before freezing a replacement timestamp pair, resolve and independently recheck at least these known items:

1. Post-attestation restoration prose:
   - if only FD33 or the `33 -> 2` restoration fails while FD1/FD32 remain usable, the JSON requires `0x7f`, EOF, and exit 123;
   - the current MD overgeneralizes this as no frame / exit 124.
2. Persistent root poison prose:
   - JSON allows P00 to repair in place or replace/rebind the observable root;
   - the current MD incorrectly requires a fresh root identity in every case.
3. Ptrace resume state machine:
   - replace the absolute one-size-fits-all `PTRACE_SYSCALL` resume rule with three mutually exclusive states: `ATTACH_BOOTSTRAP`, `AUTHORITY_TRANSITION`, and `FAILURE_TERMINATION`;
   - special attach/bootstrap EXIT and cleanup/SIGKILL paths must not be forced through the authority transition gate.
4. Audit outcome selection:
   - runtime selection cannot depend on `action` or `token` before the selected row produces them;
   - select uniquely from observable callback inputs, minimally `(event identity K, stage)`, then treat action/token as row outputs and propagate this change through catalog, source manifest, uniqueness rules, mirrors, and prose.

One Round 9 runtime validation was interrupted during the final pass. A new reviewer must reread the complete final bytes rather than resume from an assumed clean checkpoint.

## Required continuation order

1. Rehash the branch and confirm the fixed A005 aliases still equal the rejected Round 8 hashes above.
2. Continue Round 9 in a new timestamp pair only. Repair the four known issues without changing scientific method, jobs, data, claims, 57-owner order, 1,034 stage-input root rows, 54/106 pre-G5A projection, BASE19/FINAL24, 16-token/13-origin, or 10+64 CPython field invariants.
3. Run two independent read-only terminal prechecks on the exact timestamp MD/JSON bytes.
4. Only after both are clean, copy timestamp bytes identically to the fixed A005 aliases.
5. Run the same formal A005 reviewer continuation from a fresh full reread. Same-family acceptance remains provisional and grants no experiment authority.
6. If Round 9 is formally accepted, rebuild the five-row effective-contract index from:
   - canonical proposal,
   - accepted A005 MD,
   - accepted A005 JSON,
   - accepted Round 9 review MD,
   - accepted Round 9 review JSON.
7. Rebind `src/pams/temporac/contract.py`, update only the mechanically affected tests, run targeted/full TempoRAC tests, Ruff, production/scoped Mypy, and obtain a fresh independent implementation review.
8. Implement the native supervisor/provenance runtime in bounded waves. Do not skip the fresh pinned preflight or any P00/P06 gates.
9. Only after all pre-server gates pass may the successor launch server jobs. Preserve train/validation/held-out firewalls and all zero-authority defaults.
10. After verified experiments, update the paper, run independent paper/quality reviews, and only then merge to `main`.

## Suggested commands for the successor

```bash
git clone https://github.com/appleweiping/pams-rac-reproduction.git
cd pams-rac-reproduction
git fetch origin
git switch codex/aris-fullrun-icassp27
git status -sb
git log --oneline --decorate -5
```

Then read, in order:

1. this handoff file;
2. `refine-logs/FINAL_PROPOSAL.md`;
3. `refine-logs/EXPERIMENT_PLAN.md` and `refine-logs/EXPERIMENT_TRACKER.md`;
4. `idea-stage/docs/research_contract.md`;
5. A005 fixed MD/JSON, Round 8 review MD/JSON, and the unfinished Round 9 timestamp pair;
6. Wave 1 postfix review and the full `src/pams/temporac/` plus `tests/temporac/` tree.

Do not run server/data/training commands merely because the branch is available. The branch is a reproducible handoff snapshot, not a launch authorization.
