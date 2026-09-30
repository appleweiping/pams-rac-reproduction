# TempoRAC Contract Amendment 003 — P05M Count-Metric Fixture

**Amendment ID:** `TEMPORAC_CONTRACT_AMENDMENT_003_P05M_FIXTURE`  
**Status:** `PROPOSED_PENDING_FRESH_REVIEW`  
**Review class:** same-family provisional  
**Authority:** no gate authority, no launch authority, no capability authority, no data/server/experiment/Git/paper-promotion authority

## 1. Normative posture and exact non-changes

This document is a proposed byte-level amendment candidate. It does not modify
`temporac.execution.v4`, does not make P05M PASS, and cannot be cited as S0,
G5a, capability, G5b, or K7 authority until a fresh independent review accepts
it and an authorized process separately implements and verifies the accepted
bytes. The machine-readable companion is
`TEMPORAC_CONTRACT_AMENDMENT_003_P05M_FIXTURE.json`.

The following remain unchanged:

- F20 person `NAE`/`OBO`, positive integer ground truth, and stub estimate zero;
- F21 `M_route` and `D_clean` signs;
- people-within-video, equal-video, then ordered-three-seed aggregation;
- comparator set and tie order `(global,uniform,capacity-control)`;
- the same drift-selected comparator in paired clean;
- one shared 10,000-draw PCG64 stream, component multiplicity, per-draw
  comparator reselection, and sorted indices 249/9749;
- three seeds, 27 training jobs, 93 allocated A6000 hours, one claim, four
  experiment blocks, all scientific thresholds, and F23 order.

This amendment adds no natural row, source identity, evaluator label, training
job, GPU work, metric threshold, claim, or result.

## 2. Historical audit basis and fresh candidate authority

The JSON companion separates two objects that must never be conflated.
`historical_audit_basis` binds the exact bytes inspected while drafting this
proposal. Its proposal digest remains
`562325dedab0637421f62dec5ad149b3c884f67b47c4b525c8fb54a57bb1233c`.
The canonical historical-basis object digest is
`8fead557cfc236fe35468e39f69ceed162b5496fb8195a73587b31376cd2f1fb`.
P2 verifies that historical object and its canonical digest only as provenance;
it never requires a fresh implementation to equal those old implementation
hashes.

A later implementation must materialize a new canonical
`temporac.p05m-candidate-conformance-index.v1`. Its exact closed keys, member
object keys, and member order are frozen in the JSON companion. It contains no
self hash. A separate canonical
`temporac.p05m-candidate-review-receipt.v1` must bind the candidate-index hash,
this amendment's externally computed JSON hash, and a fresh reviewer witness,
with `status=ACCEPTED_FOR_P05M_EXECUTION`. Only that freshly reviewed candidate
index is authoritative for execution. Candidate hashes may and normally will
differ from historical provenance; copying or comparing old hashes cannot make
a candidate conformant. Neither index nor review receipt has gate or launch
authority.

## 3. Ordered evaluator executable closure and runtime boundary

The exact closure member order is:

1. `src/pams/temporac/metrics.py`
2. `src/pams/temporac/evaluator.py`
3. `src/pams/temporac/contract.py`
4. `src/pams/temporac/types.py`
5. `src/pams/temporac/hashio.py`
6. `src/pams/temporac/receipts.py`
7. `src/pams/temporac/trusted_packer.py`

The preimage is the exact ASCII bytes
`temporac.p05m.evaluator-executable-closure.v1` followed by one NUL byte, then
for each member in that order:

`uint64_be(path_ascii_byte_length) || path_ascii ||
 uint64_be(raw_file_byte_length) || raw_file_bytes`.

Paths are the literal forward-slash ASCII strings above. File length is the
length of the raw byte string that immediately follows it; raw bytes are not
decoded or newline-normalized. `evaluator_code_sha256` is SHA-256 over this
whole preimage. The historical seven-member closure digest is
`85a1e5f0c81fd399ed84dee5e772a1d8f1ab0c5ae81942fb87e5a7cee085e208`
and is provenance only. In particular, the stable audited hashes are
`hashio.py=2d81e415b45e3d1b612f998b6a8386a967dfdfcfaac31bc914b2c95cbaeeab3a`
and
`receipts.py=0ac800d96f536a41a67775bc8e8e778fe14c312c02f642f7cde78d210244016d`.

The accepted candidate index must list every member's literal path, raw-byte
length, and SHA-256 in that same order and bind the recomputed whole-closure
digest. It separately binds the canonical evaluator dependency contract and a
runtime lock. The only permitted third-party imports are `numpy` and
`numpy.typing` from NumPy 2.1.0. The only permitted standard-library imports are
`binascii`, `collections`, `collections.abc`, `contextlib`, `dataclasses`,
`enum`, `hashlib`, `io`, `json`, `math`, `os`, `pathlib`, `re`, `stat`,
`struct`, `types`, `typing`, and `zipfile`, from the exact CPython 3.12.4
standard-library tree bound by the runtime lock. The frozen platform is Linux
x86_64 with CPython 3.12.4 and NumPy 2.1.0; every unlisted import is a failure.
The canonical dependency-contract digest is
`646ac92f9b20eafb1ca96d8d06692984d4fca21b48a85de5f19ac2fa8c53576d`.
K7 recomputes the complete closure from raw candidate bytes, not from the seven
member digests alone.

## 4. Canonical artifacts and fixed desensitized population

The JSON subobjects `fixture_input`, `normal_expected`, and `attack_expected`
are independently encoded as UTF-8 JSON with sorted keys, separators `,` and
`:`, `ensure_ascii=false`, `allow_nan=false`, and exactly one terminal LF.
Their canonical byte commitments are:

| Artifact | SHA-256 |
|---|---|
| fixture input | `1264779925401d8efeecddefbe7245eca892543e923ec7c19ade08286fa062c9` |
| materialized 288-row array | `759b344fbc7db6ea78b041c84e23ff0fc59733985775be18f6d5c235e8408dbf` |
| component order array | `8583d714619e7b1bddc755bac95b91a801a8c97951e382fbfd7f875a70f066f2` |
| normal expected output | `50f8376d0ee1e245a8521c7831d6d23635ac8cc74694ceeef1f028c8c48e6622` |
| attack expected-global-fail output | `2e62a7b830f453e0505cdeabb766f8a46c0ab2523321d3b1f933bf7fd2cc5f30` |

The component order is exactly:

`cmp-07, cmp-00, cmp-05, cmp-02, cmp-06, cmp-01, cmp-04, cmp-03`.

It is intentionally not bytewise token order. Every bootstrap index is
interpreted against this array. The ten opaque videos and twelve people are:

| Component | Video | Slots / positive GT | Bias eighths |
|---|---|---|---|
| `cmp-07` | `vid-00` | `0/8`, `1/16` | `0`, `2` |
| `cmp-07` | `vid-01` | `0/8` | `0` |
| `cmp-00` | `vid-02` | `0/16` | `0` |
| `cmp-05` | `vid-03` | `0/8`, `1/16` | `0`, `1` |
| `cmp-02` | `vid-04` | `0/8` | `0` |
| `cmp-06` | `vid-05` | `0/16` | `0` |
| `cmp-06` | `vid-06` | `0/8` | `0` |
| `cmp-01` | `vid-07` | `0/16` | `0` |
| `cmp-04` | `vid-08` | `0/8` | `0` |
| `cmp-03` | `vid-09` | `0/16` | `0` |

The JSON companion freezes every arm/condition/seed/component error-eighth
table and the exact materialization order:
component order, identity appearance within component, arms, seeds, then
conditions. For each row,

`e=clamp(base_error_eighths+bias_eighths,0,8)` and
`estimate=count_gt-(count_gt*e//8)`.

The single explicit stub witness is local/20260817/natural-drift/`vid-09`/0,
whose estimate is overridden to integer zero. Each materialized row has exactly
the eight keys frozen in the JSON. The fixture runner must reproduce all 288
rows and their root before executing either implementation.

## 5. Normal expected output and binary64 bytes

All scientific floats in `normal_expected` are represented as the 16 lowercase
hexadecimal characters of IEEE-754 binary64 little-endian bytes. They are not
JSON floating numbers. The complete expected object is frozen in the JSON
companion.

The route and clean draw arrays are exactly C-order `<f8[10000]` raw bytes in
draw order. Their SHA-256 values are respectively
`8a325a51517a7a7669f4f8d4bdfa4d103204383b5d769e19cad409aba5f9e132`
and
`91d1fa29d36ad2e4e41733141d841a756c986b3670617f651315026a7157bdae`.
Their root is
`H("temporac.p05m.draw-pair-root.v1",raw32(route_hash),raw32(clean_hash))`
and equals
`49b83785c436d2abd5cbcda617d142922f27ee21a4e17a39f730135add15cf3e`.
The expected comparator draw counts are global 3567, uniform 4133, and
capacity-control 2300, proving that all three arms are reselected by the fixed
draw stream. Normal `status=PASS` means scorer conformance only; it is not a K7
scientific PASS.

## 6. Independent reference code and runtime boundary

The reference runtime is exactly Linux x86_64, CPython 3.12.4, and NumPy
2.1.0. It may import only the standard-library modules named in the JSON and
NumPy. It may not import `pams`, production evaluator code, filesystem data,
network, server, subprocess, or GPU interfaces. Production and reference may
not import or call one another. The canonical reference-runtime contract digest
is `c8bb58ec5d347573180d39ff369f32604737cccf8784da4cca47067c137d2d23`.

Reference source bytes are exactly the ASCII bytes beginning with
`from __future__ import annotations` after the opening fence below and ending
with the LF immediately after `return rows, expected, attack_expected`. The
fence and HTML markers are excluded. Line endings are LF and there is exactly
one terminal LF. SHA-256 must equal
`b23bf344a10f84a583c38e2326efd5ca70d4c4462fbf76715cd32b9afd20922f`
before compilation with filename `<temporac-p05m-reference-v1>`.

<!-- TEMPORAC_P05M_REFERENCE_SOURCE_V1_BEGIN -->
```python
from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from collections import Counter

import numpy as np

ARMS = ("local", "global", "uniform", "capacity-control")
COMPARATORS = ("global", "uniform", "capacity-control")
CONDITIONS = ("natural-clean", "natural-drift")
SEEDS = (20260815, 20260816, 20260817)
ROW_KEYS = frozenset(("arm", "component_token", "condition", "count_gt", "estimate", "seed", "slot", "video_token"))

class GlobalFail(Exception):
    pass

def canonical_json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"

def sha256(value):
    return hashlib.sha256(value).hexdigest()

def f64le_hex(value):
    return struct.pack("<d", float(value)).hex()

def H(tag, *fields):
    preimage = tag.encode("ascii") + b"\x00"
    for field in fields:
        preimage += len(field).to_bytes(8, "big") + field
    return sha256(preimage)

def materialize(normal):
    rows = []
    component_index = {token: index for index, token in enumerate(normal["component_order"])}
    stub_keys = {(item["arm"], item["seed"], item["condition"], item["video_token"], item["slot"]) for item in normal["stub_overrides"]}
    for identity in normal["identities"]:
        ci = component_index[identity["component_token"]]
        for arm in normal["arms"]:
            for seed in normal["seeds"]:
                for condition in normal["conditions"]:
                    base = normal["base_error_eighths"][condition][arm][str(seed)][ci]
                    eighths = max(0, min(8, base + identity["bias_eighths"]))
                    estimate = identity["count_gt"] - identity["count_gt"] * eighths // 8
                    if (arm, seed, condition, identity["video_token"], identity["slot"]) in stub_keys:
                        estimate = 0
                    rows.append({"arm": arm, "component_token": identity["component_token"], "condition": condition, "count_gt": identity["count_gt"], "estimate": estimate, "seed": seed, "slot": identity["slot"], "video_token": identity["video_token"]})
    return rows

def decode_transport(transport):
    if transport["encoding"] == "canonical-json-scalar":
        return transport["value"]
    if transport["encoding"] == "ieee754-binary64-little-endian":
        return struct.unpack("<d", bytes.fromhex(transport["hex"]))[0]
    raise RuntimeError("unknown frozen attack transport")

def mutate(base_rows, base_order, attack):
    rows = copy.deepcopy(base_rows)
    order = list(base_order)
    mutation = attack["mutation"]
    op = mutation["op"]
    if op == "replace_rows_with_empty_array": rows = []
    elif op == "delete_row": del rows[mutation["row_index"]]
    elif op == "append_copy_of_row": rows.append(copy.deepcopy(rows[mutation["row_index"]]))
    elif op == "add_row_field": rows[mutation["row_index"]][mutation["field"]] = decode_transport(mutation["transport"])
    elif op == "delete_row_field": del rows[mutation["row_index"]][mutation["field"]]
    elif op == "replace_row_field": rows[mutation["row_index"]][mutation["field"]] = decode_transport(mutation["transport"])
    elif op == "replace_component_order_with_empty_array": order = []
    elif op == "append_component_order_token": order.append(mutation["component_token"])
    elif op == "drop_component_and_rows":
        token = mutation["component_token"]
        order = [item for item in order if item != token]
        rows = [row for row in rows if row["component_token"] != token]
    else: raise RuntimeError("unknown frozen attack mutation")
    return rows, order

def fail(code):
    raise GlobalFail(code)

def validate(rows, component_order):
    if type(rows) is not list or not rows: fail("P05M_EMPTY_POPULATION")
    if type(component_order) is not list or not component_order: fail("P05M_COMPONENT_ORDER")
    if any(type(token) is not str or not token or not token.isascii() or "\x00" in token for token in component_order): fail("P05M_COMPONENT_ORDER")
    if len(component_order) != len(set(component_order)): fail("P05M_COMPONENT_ORDER")
    index = {}
    identity_meta = {}
    video_component = {}
    for row in rows:
        if type(row) is not dict or set(row) != ROW_KEYS: fail("P05M_ROW_KEYS")
        for field in ("video_token", "component_token"):
            value = row[field]
            if type(value) is not str or not value or not value.isascii() or "\x00" in value: fail("P05M_TOKEN")
        if row["arm"] not in ARMS or row["condition"] not in CONDITIONS: fail("P05M_ROW_ENUM")
        if type(row["seed"]) is not int or row["seed"] not in SEEDS: fail("P05M_SEED")
        for field in ("count_gt", "estimate"):
            value = row[field]
            if isinstance(value, float) and not math.isfinite(value): fail("P05M_NONFINITE")
            if type(value) is not int: fail("P05M_INTEGER_TYPE")
        if row["count_gt"] < 1: fail("P05M_POSITIVE_GT")
        if row["estimate"] < 0: fail("P05M_NONNEGATIVE_ESTIMATE")
        if type(row["slot"]) is not int or row["slot"] < 0: fail("P05M_INTEGER_TYPE")
        key = (row["arm"], row["seed"], row["condition"], row["video_token"], row["slot"])
        if key in index: fail("P05M_DUPLICATE_OBSERVATION")
        index[key] = row
        identity = (row["video_token"], row["slot"])
        metadata = (row["component_token"], row["count_gt"])
        if identity in identity_meta and identity_meta[identity] != metadata: fail("P05M_IDENTITY_METADATA")
        identity_meta[identity] = metadata
        if row["video_token"] in video_component and video_component[row["video_token"]] != row["component_token"]: fail("P05M_VIDEO_COMPONENT")
        video_component[row["video_token"]] = row["component_token"]
    if set(component_order) != set(video_component.values()): fail("P05M_COMPONENT_ORDER")
    if len(component_order) < 8: fail("P05M_MIN_COMPONENTS")
    identities = tuple(sorted(identity_meta, key=lambda item: (item[0].encode("ascii"), item[1])))
    if len(index) != len(identities) * len(ARMS) * len(SEEDS) * len(CONDITIONS): fail("P05M_INCOMPLETE_CROSS_PRODUCT")
    for video, slot in identities:
        for arm in ARMS:
            for seed in SEEDS:
                for condition in CONDITIONS:
                    if (arm, seed, condition, video, slot) not in index: fail("P05M_INCOMPLETE_CROSS_PRODUCT")
    videos = tuple(sorted(video_component, key=lambda token: token.encode("ascii")))
    return index, identities, videos, video_component

def score(rows, component_order):
    index, identities, videos, video_component = validate(rows, component_order)
    def video_metric(arm, seed, condition, video, metric):
        values = []
        for candidate, slot in identities:
            if candidate == video:
                row = index[(arm, seed, condition, video, slot)]
                difference = abs(row["estimate"] - row["count_gt"])
                values.append(difference / row["count_gt"] if metric == "nae" else float(difference <= 1))
        return math.fsum(values) / len(values)
    values = {(arm, condition, seed, video, metric): video_metric(arm, seed, condition, video, metric) for arm in ARMS for condition in CONDITIONS for seed in SEEDS for video in videos for metric in ("nae", "obo")}
    def aggregate(arm, condition, metric, multiplicity=None):
        seed_values = []
        for seed in SEEDS:
            replicated = []
            for video in videos:
                count = 1 if multiplicity is None else multiplicity[video]
                replicated.extend([values[(arm, condition, seed, video, metric)]] * count)
            if not replicated: fail("P05M_EMPTY_REPLICATED_VIDEOS")
            seed_values.append(math.fsum(replicated) / len(replicated))
        return math.fsum(seed_values) / len(SEEDS)
    mae = {(arm, condition): aggregate(arm, condition, "nae") for arm in ARMS for condition in CONDITIONS}
    obo = {(arm, condition): aggregate(arm, condition, "obo") for arm in ARMS for condition in CONDITIONS}
    strongest = min(COMPARATORS, key=lambda arm: (mae[(arm, "natural-drift")], COMPARATORS.index(arm)))
    seed = int.from_bytes(hashlib.sha256(b"temporac.k7.route-and-clean.v4").digest()[:8], "little")
    generator = np.random.Generator(np.random.PCG64(seed))
    route_draws = []
    clean_draws = []
    selected = Counter()
    for _ in range(10000):
        sampled = generator.integers(0, len(component_order), size=len(component_order), dtype=np.int64)
        component_counts = np.bincount(sampled, minlength=len(component_order))
        multiplicity = {video: int(component_counts[component_order.index(video_component[video])]) for video in videos}
        drift = {arm: aggregate(arm, "natural-drift", "nae", multiplicity) for arm in ARMS}
        draw_arm = min(COMPARATORS, key=lambda arm: (drift[arm], COMPARATORS.index(arm)))
        selected[draw_arm] += 1
        route_draws.append(drift[draw_arm] - drift["local"])
        clean_draws.append(aggregate("local", "natural-clean", "nae", multiplicity) - aggregate(draw_arm, "natural-clean", "nae", multiplicity))
    route_sorted = sorted(route_draws)
    clean_sorted = sorted(clean_draws)
    route_raw = struct.pack("<10000d", *route_draws)
    clean_raw = struct.pack("<10000d", *clean_draws)
    route_hash = sha256(route_raw)
    clean_hash = sha256(clean_raw)
    expected = {"bootstrap": {"clean_draws_raw_f64le_sha256": clean_hash, "clean_upper_f64le": f64le_hex(clean_sorted[9749]), "comparator_draw_counts": {arm: selected[arm] for arm in COMPARATORS}, "draw_count": 10000, "draw_pair_root_sha256": H("temporac.p05m.draw-pair-root.v1", bytes.fromhex(route_hash), bytes.fromhex(clean_hash)), "route_draws_raw_f64le_sha256": route_hash, "route_lower_f64le": f64le_hex(route_sorted[249]), "route_upper_f64le": f64le_hex(route_sorted[9749]), "stream_seed_u64": seed}, "component_order_sha256": sha256(canonical_json_bytes(component_order)), "materialized_rows_sha256": sha256(canonical_json_bytes(rows)), "point": {"avg_mae_f64le": {f"{arm}/{condition}": f64le_hex(mae[(arm, condition)]) for arm in ARMS for condition in CONDITIONS}, "avg_obo_f64le": {f"{arm}/{condition}": f64le_hex(obo[(arm, condition)]) for arm in ARMS for condition in CONDITIONS}, "clean_delta_f64le": f64le_hex(mae[("local", "natural-clean")] - mae[(strongest, "natural-clean")]), "route_margin_f64le": f64le_hex(mae[(strongest, "natural-drift")] - mae[("local", "natural-drift")]), "strongest_comparator": strongest}, "row_count": len(rows), "schema": "temporac.count-metric-fixture-expected-output.v1", "status": "PASS"}
    return expected

def execute(fixture_input):
    rows = materialize(fixture_input["normal"])
    expected = score(rows, list(fixture_input["normal"]["component_order"]))
    results = []
    for attack in fixture_input["attacks"]:
        attacked_rows, attacked_order = mutate(rows, fixture_input["normal"]["component_order"], attack)
        try:
            score(attacked_rows, attacked_order)
        except GlobalFail as exc:
            code = str(exc)
        else:
            raise RuntimeError("attack returned a partial or complete scorer output")
        if code != attack["expected_error_code"]:
            raise RuntimeError("attack failed with a nonfrozen error code")
        results.append({"attack_id": attack["attack_id"], "error_code": code, "status": "GLOBAL_FAIL"})
    attack_expected = {"results": results, "schema": "temporac.count-metric-fixture-attack-expected.v1"}
    return rows, expected, attack_expected
```
<!-- TEMPORAC_P05M_REFERENCE_SOURCE_V1_END -->

Two fresh locked-runtime processes must independently execute production and
reference from the same verified fixture bytes. Production output bytes,
reference output bytes, and the frozen expected bytes must all be identical.

## 7. Exact attacks and atomic failure

The JSON companion freezes seventeen ordered attacks, exact row indices,
fields, mutation operations, scalar transports, and expected error codes.
Nonfinite values are transported only as exact little-endian binary64 hex and
are materialized after canonical JSON parsing; NaN/Inf never appears as a JSON
number.

| ID | Mutation | Required error |
|---|---|---|
| A00 | replace rows by empty array | `P05M_EMPTY_POPULATION` |
| A01 | delete row 287 | `P05M_INCOMPLETE_CROSS_PRODUCT` |
| A02 | append copy of row 0 | `P05M_DUPLICATE_OBSERVATION` |
| A03/A04 | add unknown key / delete estimate | `P05M_ROW_KEYS` |
| A05/A06 | finite float GT / boolean estimate | `P05M_INTEGER_TYPE` |
| A07/A08 | quiet NaN / positive infinity estimate | `P05M_NONFINITE` |
| A09/A10 | zero GT / negative estimate | `P05M_POSITIVE_GT` / `P05M_NONNEGATIVE_ESTIMATE` |
| A11/A12 | empty / duplicate component order | `P05M_COMPONENT_ORDER` |
| A13 | one-row component conflict | `P05M_IDENTITY_METADATA` |
| A14 | floating seed | `P05M_SEED` |
| A15 | NUL video token | `P05M_TOKEN` |
| A16 | remove the eighth component and all its rows | `P05M_MIN_COMPONENTS` |

Each attack invokes the complete scorer. It must raise the exact global error
before returning any point, bootstrap, draw, row, or partial payload. The
attack oracle has no top-level PASS status: each result has exact
`status=GLOBAL_FAIL`.

## 8. Current production is explicitly nonconforming

The historical implementation cannot be entered into a conforming candidate
index unchanged. `metrics.py` derives components by bytewise token sort and its
paired bootstrap accepts no explicit component order, while this fixture's
frozen order is intentionally nonlexical. `evaluator.py` transitions the
one-use evaluator to consumed before validating even the old six-key summary.
`count_metric_fixture_receipt` hashes caller-supplied bytes and declares PASS
without executing production, reference, or attacks. `gates.py` can accept an
arbitrary syntactically valid digest as P2-METRIC PASS. `fixtures.py` uses the
forbidden old name `attack_global_fail_sha256` rather than the sole field name
`attack_expected_global_fail_sha256`. A newly implemented candidate and fresh
review are therefore required; historical hashes prove no conformance.

## 9. Closed candidate, execution, and receipt evidence

The candidate index has exactly the closed keys frozen in the JSON companion:
the candidate identity and status; all fixture, row, component-order, normal,
and attack commitments; the seven ordered closure members and their closure
root; dependency and runtime-lock hashes; production/reference command-spec
hashes; reference source/runtime hashes; runner-code hash; and schema. Each
closure member object has exactly `bytes`, `path`, and `sha256`. The index has
no `candidate_conformance_index_sha256` field: its canonical digest is computed
after serialization and is bound by the separate fresh-review receipt and the
runner-evidence receipt.

The two canonical command specs use
`temporac.p05m-command-spec.v1` and have exactly `argv`,
`candidate_conformance_index_sha256`, `command_role`, `environment_allowlist`,
`runtime_lock_sha256`, `schema`, and `working_directory_token`. Their roles are
exactly `production` and `reference`; each is launched once in a clean process.
Their process witnesses use `temporac.p05m-process-witness.v1`, have exactly the
closed keys frozen in the JSON, require exit code zero, bind the actual command,
executable, environment, stdout, stderr, runtime lock, and distinct OS process
identity, and use the frozen isolated-workdir token. The runtime lock binds
OS/architecture, the CPython executable and standard-library tree, NumPy 2.1.0
package bytes, locale, byte order, the environment allowlist, and disabled
network/GPU state.

Each process produces a canonical `temporac.p05m-observation-root.v1` with
exactly `attack_observed_sha256`, `clean_draws_raw_f64le_sha256`,
`materialized_rows_sha256`, `normal_observed_sha256`,
`route_draws_raw_f64le_sha256`, and `schema`. Thus the observed production and
reference normal bytes, all ordered attack-result bytes, 288-row root, and both
raw-f64 draw hashes are independently bound.

`temporac.count-metric-fixture-receipt.v1` remains a six-key inner hash summary
with exactly:

- `attack_expected_global_fail_sha256`
- `evaluator_code_sha256`
- `expected_output_sha256`
- `fixture_input_sha256`
- `schema`
- `status`

Its `status=PASS` means only that those six summary fields are syntactically
complete. It never suffices for P2, S0, G5a, K7, launch, capability, science, or
paper promotion. `attack_expected_global_fail_sha256` is the only permitted
attack hash field; `attack_global_fail_sha256` is forbidden. Individual attack
`status=GLOBAL_FAIL` means the complete scorer terminated with the exact frozen
error and emitted no point, bootstrap, draw, row, or partial payload.

The only P2-conformance PASS object is the separate closed
`temporac.count-metric-fixture-runner-evidence.v1`. It has exactly these keys
and no others:

- `attack_expected_global_fail_sha256`
- `candidate_conformance_index_sha256`
- `candidate_review_receipt_sha256`
- `evaluator_closure_sha256`
- `expected_output_sha256`
- `fixture_input_sha256`
- `inner_summary_receipt_sha256`
- `production_attack_observed_sha256`
- `production_normal_observed_sha256`
- `production_observation_root_sha256`
- `production_process_witness_sha256`
- `reference_attack_observed_sha256`
- `reference_normal_observed_sha256`
- `reference_observation_root_sha256`
- `reference_process_witness_sha256`
- `reference_runtime_contract_sha256`
- `reference_source_sha256`
- `runner_code_sha256`
- `runtime_lock_sha256`
- `schema`
- `status`

The runner may construct this receipt with `status=PASS` only after validating
the accepted candidate index and exact command specs, launching exactly two
distinct fresh processes, and having each process materialize the 288 rows,
execute normal scoring with retained raw draw hashes, and execute all seventeen
attacks. Both normal observations must equal the frozen expected bytes; both
attack observations must equal the frozen GLOBAL_FAIL oracle; both observation
roots must contain the frozen row and draw hashes. The runner receipt contains
no self hash. Its digest is computed afterward and bound by P2/S0. The inner
six-key receipt's exact digest is merely one input to it, never independent
evidence.

No amendment, candidate index, or runner receipt contains its own digest, and
no PASS-bearing object can establish authority by checking only hashes it
generated. The amendment JSON hash is reported externally and later bound by
the fresh-review receipt. These surrounding bindings eliminate circular
self-certification.

## 10. P2 to S0 to G5a to K7 verification

- **P2:** verify `historical_audit_basis_sha256` only as provenance. Require a
  newly materialized candidate index and separate accepted fresh-review
  receipt; treat their candidate hashes as authoritative even when different
  from history. Launch the two fresh processes and run normal plus all seventeen
  attacks in both. Only after every byte/root equality may the separate
  runner-evidence receipt carry P2-conformance PASS.
- **S0:** commit the exact candidate index, fresh-review receipt,
  runner-evidence receipt, optional inner-summary receipt, command/process
  witnesses, runtime lock, complete closure, input/expected/oracle bytes, and
  observed roots in its surrounding index. No proposal-closed receipt gains a
  field.
- **G5a:** load the S0-bound fresh candidate and review; recompute the complete
  seven-member raw-byte closure; require equality with candidate index, runner
  evidence, S0, and runtime lock; write only that closure digest into the
  existing closed `evaluator_code_sha256` field. G5a still computes no human
  statistic.
- **K7 prestart:** before constructing `OneUseEvaluator` or making any
  consumed-state transition, verify exact candidate-index, fresh-review,
  runner-evidence, inner-summary, command/process-witness, runtime-lock,
  reference-source/runtime, normal/attack observation, and raw-draw-root bytes.
  Recompute all seven closure members and the whole closure and require equality
  with candidate, runner, S0, G5a, and grant `evaluator_code_sha256`. Only after
  this succeeds may the one-use capability be consumed. Missing or mismatched
  bytes refuse start with `STOP-METRIC`.

No failure can be filtered, repaired, retried, or converted into a local skip.
This proposed amendment remains inert until fresh review.
