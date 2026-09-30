# WARP-PHASE Claims From Results

**Verdict ID:** `WARP-PHASE-R2C-20260815-SOL-001`  
**Route:** `pivot`  
**Acceptance status:** `provisional`  
**Review independence:** `same-family`  
**Reviewer:** `gpt-5.6-sol`, reasoning `ultra`  
**Date:** 2026-08-15

## Decision

The frozen WARP-PHASE candidate must pivot. Gate 2 is a terminal K5 selector-stability failure: agreement is `246/1000` against `>=950`; overall half/double selections are `55/144` against `<=20/<=20`; and symmetric half/double selections are `50/66` against `<=12/<=12`. The receipt authorizes nothing. Under the preregistered stop rule, this candidate may not proceed to real-track evaluation, tiny-GPU work, Stage A training, efficacy evaluation, or server launch.

This is an engineering failure, not an efficacy result. It establishes neither benefit nor harm on the supplied-track tempo-drift endpoint. No threshold tuning, fixture replacement, harmonic correction, retry, or server training is an admissible supplement for this failed candidate. A future effort requires a scientifically distinct, newly preregistered candidate and a new evidence chain; it cannot overwrite or reinterpret the frozen WARP-PHASE receipts.

The experiment-integrity audit is `FAIL`, so every confidence field below is deliberately `low`. Its earlier finding that status records suppressed the Gate-2 failure is no longer a current-state description: the reread tracker and pipeline summary now explicitly record `FAIL` and the pivot. That reconciliation does not reverse the audit's remaining failures in ground-truth provenance and evaluator/result-path integration, nor the Round-2 code-review `FAIL`.

## Claim summary

| Claim | Supported | Disposition |
|---|---|---|
| WARP-C1 | `no` | No efficacy result exists; K5 stopped the route before training. |
| WARP-C2 | `partial` | A blocked, out-of-order CPU sanity check reports identity isolation, but no reached V3 or eligible real-data diagnostic exists. |
| WARP-C3 | `no` | The candidate has zero eligible efficacy results. |
| WARP-C4 | `no` | Training and server launch are not authorized. |

## WARP-C1

**Claim:** The signed local warp-integral objective improves the frozen supplied-track tempo-drift endpoint under the preregistered rule.

- **claim_supported:** `no`
- **what_results_support:** No result supports endpoint improvement. The deterministic unit-fixture audit supports only the integrity of its exact Gate-1 input bytes, and the Gate-3 CPU smoke supports only limited implementation behavior.
- **what_results_dont_support:** Gate 2 fails all five frozen selector thresholds and authorizes nothing. Gate 0 and Gate 1 are blocked, Gate 3 is blocked, no training ran, no predictions or evaluator result exist, and no `R`, absolute `D`, or nine-component confidence interval was produced. The integrity and Round-2 code reviews both fail.
- **missing_evidence:** The planned three-seed matched treatment/control Stage-A result, receipt equality, `R>=0.05`, and a paired all-nine-component absolute-effect 95% CI lower endpoint above zero are absent. Because K5 already failed, these are not collectible as supplementation for this candidate.
- **suggested_claim_revision:** “WARP-PHASE produced no eligible efficacy result. The frozen candidate failed the preregistered K5 selector-stability gate before training, so no supplied-track tempo-drift improvement claim is licensed.”
- **next_experiments_needed:** None within this stopped candidate. Pivot to a scientifically distinct, newly preregistered candidate with its own clean gates and independent evidence chain; do not tune thresholds, replace fixtures, correct harmonics, retry this gate, or train locally/remotely.
- **confidence:** `low`

## WARP-C2

**Claim:** Identity-private state isolates untouched identities under one-person intervention.

- **claim_supported:** `partial`
- **what_results_support:** The Gate-3 CPU receipt records `identity_isolation=true`, and the code review reports that the identity-private traversal primitive passes its isolated tests.
- **what_results_dont_support:** Gate 3 is `BLOCKED`, was executed without prerequisite Gate-0/1/2 PASS receipts, lacks prediction schema and tiny-GPU receipts, and is engineering smoke rather than a reached V3 result. No frozen backbone, private/shared-scene/shuffled-key comparison, byte-identical untouched-track receipt, real-data response, or component interval exists. C2 cannot rescue C1.
- **missing_evidence:** A reached preregistered V3 diagnostic after the primary route survives, using eligible real supplied tracks, frozen predictions/backbone, exact untouched-tensor and clock receipts, all three diagnostic states, and component intervals.
- **suggested_claim_revision:** “A non-authorizing CPU sanity diagnostic passed an implementation-level identity-isolation check; no eligible real-data V3 result supports a scientific identity-isolation claim.”
- **next_experiments_needed:** None within WARP-PHASE because K5 stopped the candidate. Only if a scientifically distinct future candidate independently survives its prerequisite and primary gates should its preregistered isolation diagnostic be run; it remains supporting evidence only.
- **confidence:** `low`
- **positive-verdict ceiling:** `provisional`; the partial finding is same-family and implementation-only.

## WARP-C3

**Claim:** This candidate has eligible efficacy results.

- **claim_supported:** `no`
- **what_results_support:** No evidence supports the positive claim. The authoritative current tracker, pipeline summary, config, integrity audit, and code review all state zero eligible efficacy results.
- **what_results_dont_support:** No training, predictions, real-data evaluator invocation, AvgMAE/AvgOBO/Period-AP experiment output, bootstrap result, harmonic gate, or K4 result exists. Synthetic Gate-2 counts and Gate-3 CPU checks are engineering evidence only.
- **missing_evidence:** All result-bearing evidence is absent, but the K5 stop makes it inadmissible to fill this gap for the stopped candidate.
- **suggested_claim_revision:** “This candidate has zero eligible efficacy results; it stopped at a terminal synthetic selector-stability failure before training or evaluation.”
- **next_experiments_needed:** None within this candidate. Preserve the zero-result record and pivot under a new scientific protocol rather than manufacturing or backfilling an efficacy result.
- **confidence:** `low`

## WARP-C4

**Claim:** This candidate is authorized for training or server launch.

- **claim_supported:** `no`
- **what_results_support:** No evidence supports authorization.
- **what_results_dont_support:** The config sets `training_authorized=false`; every gate receipt has an empty `authorizes` list; Gate 2 is `FAIL`; Gate 0, Gate 1, and Gate 3 are `BLOCKED`; the integrity and code reviews explicitly deny training, evaluation, result production, and launch; K8 remains blocked.
- **missing_evidence:** No authorization receipt exists. The missing prerequisites cannot be supplemented after the frozen K5 stop for this candidate.
- **suggested_claim_revision:** “WARP-PHASE is not authorized for local or server training, evaluation, or launch and is closed for scientific pivot.”
- **next_experiments_needed:** None. Do not send this candidate to a server or attempt local training. Any future candidate needs new explicit protocol and launch authority after its own gates.
- **confidence:** `low`

## Routing rationale

- `confirm` is rejected because there is no positive efficacy result to confirm.
- `supplement` is rejected because the absence of efficacy evidence follows a terminal frozen K5 failure, not a narrow missing analysis that may be added to this candidate.
- `pivot` is required by the frozen stop rule and by the corrected current tracker and pipeline summary.

## Exact evidence bindings

| Input | SHA-256 |
|---|---|
| `.aris/claims.json` | `7f5cdfdf409cde494a65787a836fab9b0321a7dec6e5a09afb0fc89e11bd848e` |
| `.aris/evidence_precheck.json` | `73820bb7117d82f46816a4a5d610f364b52e45a8e3c0833003860041ad5a8c99` |
| `idea-stage/docs/research_contract.md` | `4b93a9dd2d32a11a83123923d831597bbfb98b84669dec9ab9c19440c5d76bb0` |
| `refine-logs/FINAL_PROPOSAL.md` | `e8ff129b3a67ac07be0e5ff1af09555bb26f000e7cde9315b13493733cda6418` |
| `refine-logs/EXPERIMENT_PLAN.md` | `020626027a73939f3fd4dfa479e6592699d19f652da592682b90d9235724ce5c` |
| `refine-logs/EXPERIMENT_TRACKER.md` | `2b23dc7399e2624dca0adc54102a366d6d668516f0463bafa90972a06b72dabe` |
| `refine-logs/PIPELINE_SUMMARY.md` | `111972fc1b6a3e6cffdd1600f5772071510346207e9a65409ff319ed3e9e3caa` |
| `configs/experiments/warp_phase_pilot_v1.yaml` | `e59fb34599eaece41545c237b18871b5a206046e53fb1246795f9dcb3d681063` |
| `data/warp_phase_pilot_v1/audit/gate0/gate0-local-inputs.receipt.json` | `2c5d3a6574d946c22e96604a8d88c44b933a423f55e33660f171b29dc4cbf7fd` |
| `data/warp_phase_pilot_v1/audit/gate1/gate1-v2-unit-v1.receipt.json` | `160d21b2bcdc0be9850646a1e187e18eff2a4ed78700490cc58251ff641d469a` |
| `data/warp_phase_pilot_v1/audit/gate1/selector_unit_fixture_pack_audit.md` | `e26a315c6142b58c93298b7280a37db7a158bcd36c74248d49df2e5cc02d0e6e` |
| `data/warp_phase_pilot_v1/audit/gate1/selector_unit_fixture_pack_audit.json` | `0d1c837d272dbecabf954f59edd12482e335f1fc4a18b6cdf324f6dfb6406cf8` |
| `data/warp_phase_pilot_v1/audit/gate2/gate2-stochastic-v2.receipt.json` | `609b2c57853d4dea661957eac810e7936611cac9d037ca9bca2eafe7ebceec03` |
| `data/warp_phase_pilot_v1/audit/gate3/gate3-cpu-sanity.receipt.json` | `157c506f6573d4ddb1cd2e3c080534cda00fd869eb07597cd0d0da37c942e54e` |
| `EXPERIMENT_AUDIT_WARP_PHASE.md` | `770ea5d9c86fce3a88a1f859434edeec893eec2d1b326cf95ec33a6112ac8e7c` |
| `EXPERIMENT_AUDIT_WARP_PHASE.json` | `457d09da6b4c7c7a9e7734e21bc3564f2d2a66dc610cc549fc70203b7fdfd7ed` |
| `refine-logs/EXPERIMENT_CODE_REVIEW_ROUND2.md` | `4d45ec10bfd4b65dc187b7ce137203c1031c7a628e9375e3a20a99626bac47d5` |
| `refine-logs/EXPERIMENT_CODE_REVIEW_ROUND2.json` | `0a528564f639ad32d79934fc5a2e6bff546faf3cd4df048f982a18b591cfbb64` |

Integrity-audit binding: audit ID `20260815_warp_phase_sol`, overall verdict `FAIL`. Round-2 code-review binding: verdict `FAIL`, canonical audited file-map SHA-256 `ad2b85e0d538c538bed408076ca71f56e98f0b4546d3da1921925a0a88c67ddd`, audited-hash manifest SHA-256 `acdea079c506f3b4bfee83dd9526f81c2b2f624446a8c26ad3792f20ccd654d5`.

Status reconciliation: the integrity audit bound older tracker/pipeline hashes `2b30a873f3d0777eaf71bfd8e686efb1dedbf06a65da60a57bfe25c353ffef58` and `217f61a5246af206262817200e5ce17c63760669fafa8cabcbc6ecdf803a10f7`; the exact current reviewed hashes above are corrected and explicitly report Gate 2 `FAIL`.
