# Experiment Tracker: WARP-PHASE Pilot v1

> **Current state:** zero eligible results; Gate 2 `FAIL`; final verdict `REVISE`; same-family provisional; K8 `BLOCKED`; training `NOT AUTHORIZED`.
> **Hard ceiling:** 59 training jobs / 552 GPU-hours. CPU gates and evaluator-only audits are not training jobs.

**Seeds:** `20270815`, `20270816`, `20270817` only.  
**Dependency rule:** Gates 0, 1, and 2 are the first deterministic executions. No training, tuning, tiny-GPU sanity, evaluator label join, or efficacy output may start until all three pass.

## Gate and Audit Queue

| Run ID | Type | Purpose | Dependency | Route | Cost cap | Stop rule | Status |
|---|---|---|---|---|---:|---|---|
| G0-PACK-CPU | CPU gate | Hash-before-unpickle, feature/vault/audit split, whitelist, eligibility, all-nine components, import separation | Contract frozen | `cpu-local` | 0 GPUh | K6/K7 on any failure | BLOCKED — local approved sources absent |
| G1-FIXTURE-CPU | CPU gate | Dual environment/witness, immutable fixture bytes, numerical/K4/evaluator fixtures | G0 PASS | `cpu-local` | 0 GPUh | K4/K5/K6/K7 on any failure; no regeneration | BLOCKED |
| G2-SELECTOR-CPU | CPU gate | Exactly 1,000 fixtures plus every eligible training identity and global weak-view receipt | G1 PASS | `cpu-local` | 0 GPUh | Global K5 on any failure | **FAIL — stochastic fixture thresholds 246/55/144/50/66** |
| G3-SMOKE-CPU | CPU smoke | CLI/import/prediction schema before tiny GPU | G0–G2 PASS | `cpu-local` | 0 GPUh | Stop before GPU on failure | BLOCKED — CPU diagnostics only; no GPU launch |
| B-K4-SHUFFLE-AUDIT | evaluator-only | Matched-time cyclic donor corruption and >=80% donor-change coverage | Stage A K1 PASS; frozen primary checkpoints | `cpu-eval-isolated` | 0 training jobs | K4 on coverage/byte/finite mismatch | BLOCKED |
| B-K4-UNSEEN-AUDIT | evaluator-only | Deterministic 320-point midpoint resampler | Stage A K1 PASS; frozen primary checkpoints | `cpu-eval-isolated` | 0 training jobs | K4 on byte/schedule/finite mismatch | BLOCKED |
| V3-IDENTITY-AUDIT | evaluator-only | Private/shared-scene/shuffled-key one-person intervention | Reached frozen backbone; preceding K-rules survive | `cpu-eval-isolated` | 0 training jobs | Diagnostic only; cannot rescue V1 | BLOCKED |
| K8-TWCRAC-PRIMARY | external audit | Complete primary-source method/loss/training/evaluation adjudication | Complete source available | `external-read-only` | 0 GPUh | Remain BLOCKED until explicit pass | BLOCKED |

## Exact Training-Job Expansion Ledger

Every pattern below expands left-to-right in the displayed seed order. Each expanded identifier is unique. `T1` and `T2` are training-only tuning attempts using fixed seeds `20270815` and `20270816`; they cannot open development labels or select from efficacy metrics.

| Group | Exact expanded Run IDs | Jobs | Per-job cap | Group GPUh cap | Dependency / stop rule | Route | Status |
|---|---|---:|---:|---:|---|---|---|
| G3-SAN | `G3-SAN-20270815`, `G3-SAN-20270816`, `G3-SAN-20270817` | 3 sanity | 2h | 6 | G0–G2 PASS; stop K5/K6 | `gpu-tiny` | BLOCKED |
| A-CONTROL | `A-CONTROL-20270815`, `A-CONTROL-20270816`, `A-CONTROL-20270817` | 3 full | 12h | 36 | G3 PASS; pair by seed | `gpu-full:a` | BLOCKED |
| A-TREATMENT | `A-TREATMENT-20270815`, `A-TREATMENT-20270816`, `A-TREATMENT-20270817` | 3 full | 12h | 36 | G3 PASS; pair by seed; K1 after all six | `gpu-full:a` | BLOCKED |
| B-CLOCK | `B-CLOCK-20270815`, `B-CLOCK-20270816`, `B-CLOCK-20270817` | 3 full | 12h | 36 | K1 PASS and `G>0`; K4 | `gpu-full:b` | BLOCKED |
| B-MASK | `B-MASK-20270815`, `B-MASK-20270816`, `B-MASK-20270817` | 3 full | 12h | 36 | K1 PASS and `G>0`; K4 | `gpu-full:b` | BLOCKED |
| C-DIRECT | `C-DIRECT-20270815`, `C-DIRECT-20270816`, `C-DIRECT-20270817` | 3 full | 12h | 36 | K4 PASS; K3 | `gpu-full:c` | BLOCKED |
| C-GLOBAL | `C-GLOBAL-20270815`, `C-GLOBAL-20270816`, `C-GLOBAL-20270817` | 3 full | 12h | 36 | K4 PASS; K3 | `gpu-full:c` | BLOCKED |
| C-UNSIGNED | `C-UNSIGNED-20270815`, `C-UNSIGNED-20270816`, `C-UNSIGNED-20270817` | 3 full | 12h | 36 | K4 PASS; K3 | `gpu-full:c` | BLOCKED |
| D-PAMS-TUNE | `D-PAMS-T1-20270815`, `D-PAMS-T2-20270816` | 2 tune | 3h | 6 | K1/K3/K4 survive; training-only criterion | `queue:d-tune` | BLOCKED |
| D-PAMS-FULL | `D-PAMS-20270815`, `D-PAMS-20270816`, `D-PAMS-20270817` | 3 full | 12h | 36 | Its two tune receipts frozen | `queue:d-full` | BLOCKED |
| D-TRACKPAMS-TUNE | `D-TRACKPAMS-T1-20270815`, `D-TRACKPAMS-T2-20270816` | 2 tune | 3h | 6 | K1/K3/K4 survive; training-only criterion | `queue:d-tune` | BLOCKED |
| D-TRACKPAMS-FULL | `D-TRACKPAMS-20270815`, `D-TRACKPAMS-20270816`, `D-TRACKPAMS-20270817` | 3 full | 12h | 36 | Its two tune receipts frozen | `queue:d-full` | BLOCKED |
| D-SIMPER-TUNE | `D-SIMPER-T1-20270815`, `D-SIMPER-T2-20270816` | 2 tune | 3h | 6 | K1/K3/K4 survive; training-only criterion | `queue:d-tune` | BLOCKED |
| D-SIMPER-FULL | `D-SIMPER-20270815`, `D-SIMPER-20270816`, `D-SIMPER-20270817` | 3 full | 12h | 36 | Its two tune receipts frozen | `queue:d-full` | BLOCKED |
| D-CYCLECL-TUNE | `D-CYCLECL-T1-20270815`, `D-CYCLECL-T2-20270816` | 2 tune | 3h | 6 | K1/K3/K4 survive; training-only criterion | `queue:d-tune` | BLOCKED |
| D-CYCLECL-FULL | `D-CYCLECL-20270815`, `D-CYCLECL-20270816`, `D-CYCLECL-20270817` | 3 full | 12h | 36 | Its two tune receipts frozen | `queue:d-full` | BLOCKED |
| D-DEEPPHASE-TUNE | `D-DEEPPHASE-T1-20270815`, `D-DEEPPHASE-T2-20270816` | 2 tune | 3h | 6 | K1/K3/K4 survive; training-only criterion | `queue:d-tune` | BLOCKED |
| D-DEEPPHASE-FULL | `D-DEEPPHASE-20270815`, `D-DEEPPHASE-20270816`, `D-DEEPPHASE-20270817` | 3 full | 12h | 36 | Its two tune receipts frozen | `queue:d-full` | BLOCKED |
| D-TIMEEQ-TUNE | `D-TIMEEQ-T1-20270815`, `D-TIMEEQ-T2-20270816` | 2 tune | 3h | 6 | K1/K3/K4 survive; training-only criterion | `queue:d-tune` | BLOCKED |
| D-TIMEEQ-FULL | `D-TIMEEQ-20270815`, `D-TIMEEQ-20270816`, `D-TIMEEQ-20270817` | 3 full | 12h | 36 | Its two tune receipts frozen; Stage D adjudicates K2 | `queue:d-full` | BLOCKED |
| E-TSSM-TUNE | `E-TSSM-T1-20270815`, `E-TSSM-T2-20270816` | 2 tune | 3h | 6 | K2 survives; optional control-only | `gpu-tune:e` | BLOCKED |
| E-TSSM-FULL | `E-TSSM-20270815`, `E-TSSM-20270816`, `E-TSSM-20270817` | 3 full | 12h | 36 | Tune receipts frozen; K9 forbids fallback | `gpu-full:e` | BLOCKED |

## Budget Proof and Stage Ceilings

| Checkpoint | Newly reached jobs | Cumulative training jobs | Cumulative GPUh cap | Required decision |
|---|---:|---:|---:|---|
| Gate 3 | 3 sanity | 3 | 6 | G3 sanity PASS |
| Stage A | 6 full | 9 | 78 | K1 PASS |
| Stage B | 6 full | 15 | 150 | K4 PASS including two zero-job audits |
| Stage C | 9 full | 24 | 258 | K3 survives |
| Stage D | 12 tune + 18 full | 54 | 510 | K2 survives |
| Stage E maximum | 2 tune + 3 full | **59** | **552** | Optional control complete; no fallback |

Arithmetic: `42 full*12 = 504 GPUh`; `14 tune*3 = 42 GPUh`; Gate-3 sanity total `6 GPUh`; total `504+42+6 = 552 GPUh`.

## Queue Routing and Dependency Rules

- A release with at least ten runnable jobs must use the scheduler queue with immutable run IDs, dependency IDs, config hash, environment hash, and output receipt path. Manual/background launch is forbidden.
- Stage D has 12 tuning jobs routed to `queue:d-tune`; only after their training-only receipts freeze are its 18 full jobs routed to `queue:d-full`.
- A/B/C/E groups remain scheduler-addressable even when fewer than ten; no route permits bypassing a K-rule.
- Failed, canceled, non-finite, receipt-mismatched, or over-cap jobs count against the ceiling and are not silently replaced. A replacement requires protocol revision; seeds never change.

## Per-Run Receipt Requirements

Every training job must bind run ID, group, Git commit, configuration/fingerprint, seed, feature-manifest hash, environment/hardware, initialization, warp schedule hash, optimizer/update cap, start/end timestamps, logs, checkpoint/prediction/trace hashes, status, and dependency decision. Full predictions for a reached arm freeze before vault join. Missing or unequal paired receipts invoke K4/K6.

## Current Launch Order

1. `G0-PACK-CPU` produced `BLOCKED`: the approved train/validation source bytes are not present locally and the typed full-pack path is not frozen.
2. `G1-FIXTURE-CPU` produced `BLOCKED`: environment, stochastic pack, ten-case recomputation, and deterministic selector-unit fixtures passed, but the complete numerical witness set is absent.
3. `G2-SELECTOR-CPU` produced a terminal stochastic-fixture `FAIL`: agreement/overall-half/overall-double/symmetric-half/symmetric-double are `246/55/144/50/66`, against `950/20/20/12/12` limits.
4. `G3-SMOKE-CPU` produced `BLOCKED`: its CPU execution checks pass, while prediction schema, tiny-GPU receipts, and prerequisite gate passes are absent.

No retry, threshold tuning, real-track evaluation, GPU job, or server launch is authorized for this candidate. The next ARIS action is integrity audit followed by result-to-claim and a scientific pivot. K8 remains externally blocked and no efficacy result field is populated.

## Frozen Local Gate Receipts

- Gate 0: `data/warp_phase_pilot_v1/audit/gate0/gate0-local-inputs.receipt.json`, SHA-256 `2c5d3a6574d946c22e96604a8d88c44b933a423f55e33660f171b29dc4cbf7fd`
- Gate 1: `data/warp_phase_pilot_v1/audit/gate1/gate1-v2-unit-v1.receipt.json`, SHA-256 `160d21b2bcdc0be9850646a1e187e18eff2a4ed78700490cc58251ff641d469a`
- Gate 2: `data/warp_phase_pilot_v1/audit/gate2/gate2-stochastic-v2.receipt.json`, SHA-256 `609b2c57853d4dea661957eac810e7936611cac9d037ca9bca2eafe7ebceec03`
- Gate 3 CPU: `data/warp_phase_pilot_v1/audit/gate3/gate3-cpu-sanity.receipt.json`, SHA-256 `157c506f6573d4ddb1cd2e3c080534cda00fd869eb07597cd0d0da37c942e54e`
