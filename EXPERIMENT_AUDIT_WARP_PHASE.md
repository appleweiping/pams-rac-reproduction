# WARP-PHASE Experiment Integrity Audit

**Overall verdict:** `FAIL`  
**Acceptance status:** `provisional`  
**Reviewer:** `gpt-5.6-sol`, reasoning `ultra`  
**Review independence:** `same-family`  
**Date:** 2026-08-15

This integrity failure does not halt ARIS, but it prevents every positive selector, phase-mass, training, efficacy, performance, novelty, or submission claim. There is no eligible efficacy result.

## Decision

The receipt-bound synthetic Gate-2 result is a real protocol failure, not a pending gate: agreement is `246/1000` against a minimum of `950`, overall half/double counts are `55/144` against maxima of `20/20`, and symmetric half/double counts are `50/66` against maxima of `12/12`. I independently recomputed all five counts from the 23-member canonical synthetic pack; every member hash and aggregate hash matched.

The failure is not an efficacy result and uses no real or protected data. It is an engineering selector-stability failure that triggers the frozen K5-style stop boundary. The critical integrity problem is that the experiment authority still describes Gate 2 as `BLOCKED` or not run, despite the exact `FAIL` receipt. Ground-truth packaging is also absent, and the WARP evaluator is not invoked anywhere outside its own module.

## Receipt reconciliation

| Gate | Actual receipt | Authorization | Classification |
|---|---|---|---|
| Gate 0 | `BLOCKED` | none | Local train/val files absent; typed adapter, feature/vault split, eligibility/component manifest, and import-isolation receipt missing |
| Gate 1 | `BLOCKED` | none | Fixture inputs pass, but the complete numerical/evaluator witness set is missing |
| Gate 2 | **`FAIL`** | none | Synthetic selector-stability failure on all five thresholds; real-track inputs also absent |
| Gate 3 CPU | `BLOCKED` | none | Reproducible engineering smoke only; prediction schema, tiny-GPU receipts, and upstream pass receipts missing |

Evidence: `data/warp_phase_pilot_v1/audit/gate0/gate0-local-inputs.receipt.json:1`, `data/warp_phase_pilot_v1/audit/gate1/gate1-v2-unit-v1.receipt.json:1`, `data/warp_phase_pilot_v1/audit/gate2/gate2-stochastic-v2.receipt.json:1`, and `data/warp_phase_pilot_v1/audit/gate3/gate3-cpu-sanity.receipt.json:1`.

## A-F audit

| Dimension | Verdict | Finding |
|---|---|---|
| A. Ground-truth provenance | `FAIL` | Historical train/val hashes are documented, but the local sources and required physical label-isolation artifacts do not exist; Gate 0 is blocked. |
| B. Score normalization | `PASS` implementation-only | Video-first normalized AvgMAE, unclipped predictions, half-even secondary AvgOBO, paired component bootstrap, and `[0,100]` supplied-track Period AP are implemented consistently. They have not produced an experiment score. |
| C. Result file/number existence | `WARN` | Gate-2 numbers and Gate-3 CPU evidence exist and reproduce. No narrative, claims-from-results file, prediction result, or eligible efficacy score exists. |
| D. Dead code/metric invocation | `FAIL` | No code outside `evaluator.py` statically imports or references `pams.warp_phase.evaluator`; the CLI has no evaluation/result command. |
| E. Scope versus language | `FAIL` | Scientific documents correctly say prospective/partial-cache/supplied-track/zero-results, but current-status records suppress the Gate-2 failure and retain stale implementation descriptions. |
| F. Evaluation classification | `WARN` | Gate 2 is synthetic engineering evidence, Gate 1 is fixture evidence, and Gate 3 is CPU smoke. None is real-data efficacy evidence, but Gate 2 still stops the frozen route. |

## Material findings

### F-01 — Critical: the authoritative tracker suppresses Gate-2 `FAIL`

The implementation sets `terminal_failure=True` for any violated synthetic threshold and serializes `FAIL` (`src/pams/warp_phase/gates.py:1837`, `src/pams/warp_phase/gates.py:1892`, `src/pams/warp_phase/gates.py:1910`). The receipt contains all five failed checks and an empty authorization list (`data/warp_phase_pilot_v1/audit/gate2/gate2-stochastic-v2.receipt.json:1`).

In contrast, `refine-logs/EXPERIMENT_TRACKER.md:15` records Gate 2 as `BLOCKED`, `refine-logs/EXPERIMENT_TRACKER.md:79` presents it as a future launch, and `refine-logs/PIPELINE_SUMMARY.md:18` says Gates 0–2 have not run. This is a failure-receipt misrepresentation. It does not currently enable launch—the records remain blocking—but it erases the difference between “not yet executable” and “scientifically failed under the frozen selector gate.”

### F-02 — High: no local GT provenance or physical isolation chain

The config declares only train/val sources (`configs/experiments/warp_phase_pilot_v1.yaml:8`), and the historical schema audit establishes that these pickles co-locate pose inputs and privileged count/density/period information (`PILOT_DATA_SCHEMA_AUDIT.md:112`, `PILOT_DATA_SCHEMA_AUDIT.md:122`). It requires a physical feature/vault split before use.

The Gate-0 receipt records `FileNotFoundError` for both local sources and the absence of the adapter, pack/vault audit, nine-component manifest, and training-import isolation receipt (`data/warp_phase_pilot_v1/audit/gate0/gate0-local-inputs.receipt.json:1`). Direct existence checks also found no local train pickle, val pickle, `features`, `vault`, or pilot `results` directory. Therefore no real-data learner or evaluator invocation is provenance-eligible.

### F-03 — High: the evaluator/result path is uninvoked

The count scorer exists at `src/pams/warp_phase/evaluator.py:111`, bootstrap at `:153`, supplied-track Period AP at `:338`, harmonic gate at `:467`, and K4 at `:501`. A clean-room scan of `src/pams` and `scripts/experiments` found zero static imports and zero literal references to `pams.warp_phase.evaluator` outside that file.

The CLI exposes gates plus a training interface explicitly documented as blocked (`src/pams/warp_phase/cli.py:200`). Gate 1 records `complete_numerical_witness_set=false` and a missing evaluator witness schema; Gate 3 records `prediction_schema=false` (`gate1-v2-unit-v1.receipt.json:1`, `gate3-cpu-sanity.receipt.json:1`). Thus there is no experimental AvgMAE, AvgOBO, Period AP, bootstrap, harmonic, or K4 result.

### F-04 — Medium: earlier fixture adjudications bind an older gate source

`fixture_candidate_v2_audit.json:22` and `fixture_candidate_v2_repair_adjudication.json:34` bind/call current a `gates.py` hash beginning `52d573a0`. The actual reviewed file is `3a91e77692ac54e1e85fd56d8f3bff2b3a3c517095554d30b92e9055e1426dfa`, which the later selector-unit receipt correctly binds (`selector-unit-fixtures-v1.receipt.json:1`).

This does not alter the canonical synthetic pack: its receipt binds all 23 members and generator source, and the aggregate independently reproduces. It does mean the earlier repair adjudication is point-in-time pack evidence, not current gate-code review evidence.

### F-05 — Medium: non-finite harmonic behavior conflicts with the addendum

The period addendum specifies `nonfinite_class: off-grid` (`PILOT_PERIOD_SCHEMA_ADDENDUM.json:143`). The evaluator raises `EvaluationError` for a non-finite/non-positive selector or evaluator period (`src/pams/warp_phase/evaluator.py:453`, `src/pams/warp_phase/evaluator.py:455`). The clean-room smoke confirmed the exception. The contract must resolve global failure versus off-grid classification before evaluator freeze. No current result is affected because the evaluator is uninvoked.

## Normalization and metric assessment

- `evaluate_counts` computes `abs(pred-truth)/truth` per person, means within each video, then means videos (`evaluator.py:111-136`). Predictions are continuous and unclipped; `np.rint` is used only for secondary AvgOBO (`:125-133`).
- `paired_component_bootstrap` requires the exact three seeds, 10,000 `PCG64(20270815)` draws, all nine components, paired multiplicities across arms/seeds, linear 2.5%/97.5% quantiles, and the `R>=0.05 && lower>0` rule (`:153-248`).
- Supplied-track Period AP explicitly disclaims detection/tracking parity (`:338-362`) and reports AP/mAP in percentage points (`:408-418`). This is a supplied-track proxy diagnostic, not published end-to-end MultiCounter+ evidence.
- The in-memory declared count fixture reproduced control/treatment normalized video-first AvgMAE `0.45/0.1625`. Exact synthetic interval predictions produced Period-mAP/AP50/AP75 `100/100/100` on `[0,100]`. These are audit smokes, not experiment results.

## Leakage, selection, and phantom-result checks

- **Test/sealed/held-out access:** not observed. This audit opened no server, real, test, sealed, or held-out data. The reviewed config permits only train/val (`warp_phase_pilot_v1.yaml:12`).
- **Real label leakage:** not observed because no training or real evaluation occurred. The unresolved label-mixed-source and physical-separation requirement remains a fatal precondition, not evidence of successful isolation.
- **Synthetic labels:** `semantic_period.npy` is constructor truth for Gate-2 engineering fixtures (`generate_warp_phase_fixture_pack.py:520-521`, `gates.py:1729-1764`). It is not real GT or learner supervision.
- **Proxy versus real:** the 1,000-row selector pack is synthetic; Gate 2 never opened real tracks and records that absence (`gates.py:1793-1799`, `gates.py:1893-1906`). It cannot support efficacy language.
- **Outcome selection:** not detected. The v2 repair preserved the unfavorable Gate-2 vector and did not waive thresholds. There are no training outcomes to cherry-pick.
- **Phantom training/results:** not detected. `evidence_status` is `zero_eligible_results` and `training_authorized=false` (`warp_phase_pilot_v1.yaml:4-5`); provider readiness is `NOT_READY` with zero launch authorizations (`.aris/compute/provider-env.json:7-10`). Optional `refine-logs/NARRATIVE_REPORT.md` and `refine-logs/CLAIMS_FROM_RESULTS.md` are absent.
- **Failure receipt misrepresentation:** detected in the tracker/pipeline status, as described in F-01.

## Independent receipt/number verification

- Canonical fixture receipt: `191a1689197f7e9a3eb965ee0dc0e52fa58e9d8485c097c9e601162fad24cd19`.
- Canonical synthetic pack: `c6dbe56df7b4f7d427ce1f51d5d818ad0e46dcfbf6e1550e0d77116435760196`; 23/23 member hashes matched.
- Canonical selector failures: 240/1000.
- Gate-2 observation vector: exactly `[246,55,144,50,66]`; every frozen threshold fails.
- Gate-3 CPU receipt regenerated from current code with the same `cpu_diagnostic_sha256=052c156cea23437a251856739d24122b1899fae054ab4c294ff63440d9c691ff` and the same `BLOCKED` fields.

## Exact reviewed-input hashes

| Input | SHA-256 |
|---|---|
| `.aris/compute/output-budget.json` | `86e8c76e1cc0f59e6c8931a7060021ecf4dc6feb03ac9a0ab6ab3c425aa3e226` |
| `.aris/compute/provider-env.json` | `ca7e25e8e84d01fd5a71b4d0f33b7338742edc566b5ad5dca8e54c18b7545e1d` |
| `PILOT_DATA_SCHEMA_AUDIT.json` | `ba1c376127d0304d7ee76f902c56dd753c351b71fc11655c45451b98fa0481b8` |
| `PILOT_DATA_SCHEMA_AUDIT.md` | `7058981e8eb37622c09f5808a4b1329bc7cd133f474101723f15e5e4273eab25` |
| `PILOT_PERIOD_SCHEMA_ADDENDUM.json` | `2c95af393fa73f3c581a327f9ee8a5e8a5c69c03a9fe6349d3b6531c48e914` |
| `PILOT_PERIOD_SCHEMA_ADDENDUM.md` | `49edcfe25c302c2b55b8e65bc18e1cb7febe25cc8b76698d631fa24150d4b0b5` |
| `configs/experiments/warp_phase_pilot_v1.yaml` | `e59fb34599eaece41545c237b18871b5a206046e53fb1246795f9dcb3d681063` |
| `data/warp_phase_pilot_v1/audit/gate0/gate0-local-inputs.receipt.json` | `2c5d3a6574d946c22e96604a8d88c44b933a423f55e33660f171b29dc4cbf7fd` |
| `data/warp_phase_pilot_v1/audit/gate1/candidate-20260815-sol-v2/canonical_fixture_pack.receipt.json` | `191a1689197f7e9a3eb965ee0dc0e52fa58e9d8485c097c9e601162fad24cd19` |
| `data/warp_phase_pilot_v1/audit/gate1/fixture_candidate_v2_audit.json` | `82860755e577ffd41e8a4cd7c6ca7c008f444dbd695b6200981655c0609de8a0` |
| `data/warp_phase_pilot_v1/audit/gate1/fixture_candidate_v2_audit.md` | `8450fe176da370e46a549ea20d9770f171fe345a1be57c84fb6c0a881c3420af` |
| `data/warp_phase_pilot_v1/audit/gate1/fixture_candidate_v2_repair_adjudication.json` | `70a583b84abe13b7e861c95017137bd21155fbecd8a6723eb2461e3c7187e44d` |
| `data/warp_phase_pilot_v1/audit/gate1/fixture_candidate_v2_repair_adjudication.md` | `e3f68a33ae0db52274d41941858b695625853e5babd0fda5932a9107b5401a95` |
| `data/warp_phase_pilot_v1/audit/gate1/gate1-v2-unit-v1.receipt.json` | `160d21b2bcdc0be9850646a1e187e18eff2a4ed78700490cc58251ff641d469a` |
| `data/warp_phase_pilot_v1/audit/gate1/selector-unit-fixtures-v1/selector-unit-fixtures-v1.receipt.json` | `913e8946ee27392ab374a26887a5a9d8169e69afd951b98319019f3eb1cf2f5f` |
| `data/warp_phase_pilot_v1/audit/gate1/selector_unit_fixture_pack_audit.json` | `0d1c837d272dbecabf954f59edd12482e335f1fc4a18b6cdf324f6dfb6406cf8` |
| `data/warp_phase_pilot_v1/audit/gate1/selector_unit_fixture_pack_audit.md` | `e26a315c6142b58c93298b7280a37db7a158bcd36c74248d49df2e5cc02d0e6e` |
| `data/warp_phase_pilot_v1/audit/gate2/gate2-stochastic-v2.receipt.json` | `609b2c57853d4dea661957eac810e7936611cac9d037ca9bca2eafe7ebceec03` |
| `data/warp_phase_pilot_v1/audit/gate3/gate3-cpu-sanity.receipt.json` | `157c506f6573d4ddb1cd2e3c080534cda00fd869eb07597cd0d0da37c942e54e` |
| `idea-stage/docs/research_contract.md` | `4b93a9dd2d32a11a83123923d831597bbfb98b84669dec9ab9c19440c5d76bb0` |
| `pyproject.toml` | `14b3b35bbfa57fc982e24ef2f999df288893c0f0d12a53a9f82120babefead38` |
| `refine-logs/EXPERIMENT_TRACKER.md` | `2b30a873f3d0777eaf71bfd8e686efb1dedbf06a65da60a57bfe25c353ffef58` |
| `refine-logs/FINAL_PROPOSAL.md` | `e8ff129b3a67ac07be0e5ff1af09555bb26f000e7cde9315b13493733cda6418` |
| `refine-logs/PIPELINE_SUMMARY.md` | `217f61a5246af206262817200e5ce17c63760669fafa8cabcbc6ecdf803a10f7` |
| `refine-logs/SERVER_PREFLIGHT.md` | `f3b254fe2ce7bf7f61fbc6bf3f416ef7e86f8230d1f484259b668ed23c66444b` |
| `scripts/experiments/generate_warp_phase_fixture_pack.py` | `7af1dac0ce246979d608b3db8e9fcaef9fe18dbfe2a5ecc803616e30919bb4ce` |
| `scripts/experiments/generate_warp_phase_selector_unit_fixtures.py` | `2ae32f4133903691a757557425980eb553b3de4d749b4eca9a8f76719b41847f` |
| `src/pams/warp_phase/cli.py` | `c6b8e344493b100cfe0e396b64f2e390e39b35e843ebdaccfe97dfa2514c2a72` |
| `src/pams/warp_phase/evaluator.py` | `82f6e46e58f15e21e020e031e8bdffd24e884816ca8be0740d440c447e20f38c` |
| `src/pams/warp_phase/gates.py` | `3a91e77692ac54e1e85fd56d8f3bff2b3a3c517095554d30b92e9055e1426dfa` |
| `src/pams/warp_phase/selector.py` | `73b31b7d6c58a67efa8542df754971a58926d783258c9776269ec06765dafa0e` |

Auxiliary CLI-registration evidence: `src/pams/cli.py` SHA-256 `e8fe3e2b48b78e4e122329df333137e74948991ce9600ec8cbaad26b99d30425`.

## Trace binding

- `.aris/traces/experiment-audit/20260815_warp_phase_sol/independent_checks.py`: `1e5c51037bb1e3f1435259f0853ce4a1b9f1a4ddf96f7ef7b825555ab659692e`
- `.aris/traces/experiment-audit/20260815_warp_phase_sol/trace_summary.json`: `edbd52d5202653d4668d6a4ad8db8942baa2f8d228b5cc8cd56528654826d010`

Final disposition: `FAIL`, `provisional`, same-family. ARIS may continue, but the Gate-2 failure and missing provenance/evaluator chain must remain claim-blocking; no current artifact licenses a positive result.
