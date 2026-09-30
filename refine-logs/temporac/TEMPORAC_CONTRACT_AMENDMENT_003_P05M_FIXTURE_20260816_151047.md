# TempoRAC Contract Amendment 003 — P05M Count-Metric Fixture

**Amendment ID:** `TEMPORAC_CONTRACT_AMENDMENT_003_P05M_FIXTURE`  
**Schema:** `temporac.contract-amendment.p05m-fixture.v2`  
**Status:** `PROPOSED_PENDING_FRESH_REVIEW`  
**Review class:** same-family provisional  
**Authority:** no gate, launch, capability, data, server, experiment, test, training, Git, paper-promotion, or claim authority

## 1. Normative posture and exact non-changes

This is a proposed byte-level correction of Amendment 003 after the Round 2
`REVISE` disposition. It does not make P05M or P2 PASS, does not materialize a
candidate, does not authorize implementation or execution, and cannot be cited
as S0, G5a, capability, G5b, or K7 authority. A fresh independent reviewer must
accept both amendment outputs and, later, the complete candidate executable and
orchestration closure before any P05M process may run. The machine-readable
companion is `TEMPORAC_CONTRACT_AMENDMENT_003_P05M_FIXTURE.json`; whenever prose
and a closed JSON key list differ, the JSON companion is the byte-level
authority, subject to fresh review.

The following remain byte-semantically and scientifically unchanged:

- F20 person `NAE`/`OBO`, positive integer ground truth, and stub estimate
  zero;
- F21 `M_route` and `D_clean` signs;
- people-within-video, equal-video, then ordered-three-seed aggregation;
- comparator set and tie order `(global,uniform,capacity-control)`;
- the same drift-selected comparator in paired clean;
- one shared 10,000-draw PCG64 stream, component multiplicity, per-draw
  comparator reselection, and sorted indices 249/9749;
- the fixed 288 synthetic rows, eight-component nonlexical order, 17 attack
  definitions, raw-f64 draw bytes and roots;
- three seeds, 27 training jobs, 93 allocated A6000 hours, one claim, four
  experiment blocks, all scientific thresholds, and F23 order.

This amendment adds no natural row, evaluator label, source identity, model,
training job, GPU work, metric threshold, result, claim, or paper-visible block.
It does not reuse WARP.

## 2. Historical audit basis is provenance, never candidate authority

The JSON `historical_audit_basis` is a closed 55-member record of every
protocol, rejected amendment, Round 1/Round 2 review, current canonical
proposal/plan/tracker/research contract, latest certificate-postfix/operator
review, current implementation source, and current TempoRAC test file read for
this revision. Each member binds literal path token, role, raw byte length, and
SHA-256. Its canonical digest is
`24a80d9007098a7a523c9634c6d5fe9f04cfb473c69b2d019fac036aca1c879c`.

The rejected Round 2 fixed pair was first preserved byte-for-byte as:

| Archive | Bytes | SHA-256 |
|---|---:|---|
| `TEMPORAC_CONTRACT_AMENDMENT_003_P05M_FIXTURE_20260816_145644.md` | 29,722 | `d615def9dc468b7e483488b87cb0524302f1c4c325df8bf092e741e21f6262d1` |
| `TEMPORAC_CONTRACT_AMENDMENT_003_P05M_FIXTURE_20260816_145644.json` | 29,094 | `c73b85731ece838684f78c41b3dfcdb498af5cf9822f7ee84d2add4d09ffe1fc` |

Those timestamped files are the non-self-referential capture of the old fixed
paths. Round 2 review bytes are
`9f4dc98b3f2237dbd17381007105977639b13d60b478e13e162631666504296b`
(MD) and
`b8552d10bf8b9d9e6b3f9202127b1ea56f9b3ea9d43a9f9994e06166ce741fe2`
(JSON). The current canonical proposal, plan, tracker, and research-contract
hashes in the historical object are respectively
`3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391`,
`4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e`,
`714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b`,
and
`3e037a2f74c49dd62e596dc7b875d94216e45b7f693b80a7afb6a60f8244367b`.

P2 verifies this historical object only as amendment provenance. It must never
compare a fresh implementation, runner, runtime, dependency graph, or command
hash with a historical member as a condition of candidate acceptance. A future
canonical `temporac.p05m-candidate-conformance-index.v2` contains no self hash
and becomes authoritative only when a separate
`temporac.p05m-candidate-review-receipt.v2` binds its exact canonical digest,
the external MD/JSON amendment digests, the candidate snapshot, and a closed
fresh independent reviewer witness with
`status=ACCEPTED_FOR_P05M_EXECUTION`. Candidate hashes may and normally will
differ from every historical implementation hash. Neither the index nor its
review receipt has gate or launch authority.

The reviewer witness is not a bare digest: its closed object binds amendment
author, candidate implementer, and reviewer identity tokens; requires those
three opaque non-secret ASCII tokens to be pairwise unequal; binds an
independent pre-review assignment receipt, exact review command/runtime, both
amendment output hashes, and candidate-index hash; and records the fixed
independence declaration. The reviewer process has read-only candidate access.
The review receipt embeds this witness and its canonical hash.

## 3. Complete production/P2/K7 executable closure

Normal CPython package import is mandatory; no isolated-loader shortcut is
allowed. Importing a `pams.temporac` member first executes the exact candidate
`pams/__init__.py`, which imports `pams.types`, and then executes
`pams/temporac/__init__.py`. The production/P2/K7 local member order is exactly:

1. `src/pams/__init__.py`
2. `src/pams/types.py`
3. `src/pams/temporac/__init__.py`
4. `src/pams/temporac/contract.py`
5. `src/pams/temporac/types.py`
6. `src/pams/temporac/hashio.py`
7. `src/pams/temporac/receipts.py`
8. `src/pams/temporac/trusted_packer.py`
9. `src/pams/temporac/quadrature.py`
10. `src/pams/temporac/preprocess.py`
11. `src/pams/temporac/x0.py`
12. `src/pams/temporac/certify.py`
13. `src/pams/temporac/metrics.py`
14. `src/pams/temporac/evaluator.py`
15. `src/pams/temporac/decode.py`
16. `src/pams/temporac/fixtures.py`
17. `src/pams/temporac/gates.py`

The exact preimage is ASCII
`temporac.p05m.production-executable-closure.v2`, one NUL, then for every
member in that order:

`uint64_be(path_ascii_byte_length) || path_ascii ||
 uint64_be(raw_file_byte_length) || raw_file_bytes`.

Paths are the literal forward-slash ASCII strings above. Lengths are unsigned
64-bit big-endian. Raw bytes are neither decoded nor newline-normalized.
`evaluator_code_sha256` is SHA-256 of this whole 17-member preimage, never a
metrics/evaluator subset and never a digest-of-digests. The currently audited
bytes yield
`dfcf258b13aa571d3636101f09e66c9d7889a0f09dd88515f4c332ae2d662943`,
which is historical provenance only.

| Member | Current bytes | Current SHA-256 |
|---|---:|---|
| `src/pams/__init__.py` | 160 | `8b6ffbf3ac8ed7e43f1f40ef5793baa087f73b303eff342d891380bc9a5d67ce` |
| `src/pams/types.py` | 6,210 | `78c2ad53e98856425492a1711faa62fc2bb7764767dc48a22f7c4d5e043a3e31` |
| `src/pams/temporac/__init__.py` | 280 | `e3f71c2d85974ac3f635f76ee61f8e1f793b9bc251b25ff4336563d4da52c2a3` |
| `contract.py` | 9,079 | `5ab8fb62dc558ba78e95fc07e52c1f1a1afbff5b1c5003834079eea6143d745f` |
| `types.py` | 25,746 | `4249eac2738f5b2edf53f38cbb681dffae609c151d493dd03b625fe10ad4c3ca` |
| `hashio.py` | 27,198 | `aa460d5de51f107b2fc6b2ad171747e0da38a87cf30f82a869a335c268f1d613` |
| `receipts.py` | 27,223 | `564039b5d37a4a2759544a9ae65f267b57d7f6c1e46bca812ae5ed4f4a2c17e4` |
| `trusted_packer.py` | 17,137 | `40d6e3d5be1b26b7ba0f0f7aadf8da7496c95c4e829face5c7e9faaee2cdf94e` |
| `quadrature.py` | 11,569 | `44f4b91dc4d5631442982732177a6f57726c0b8f42594380b5a75f2209efc2f3` |
| `preprocess.py` | 24,916 | `042b5cae1917359cb5596837c81c0e36658ff99a6a8fd6f2602b478ca6f98bc3` |
| `x0.py` | 24,963 | `5f4585425a83bf63779265db9004d210c5c2875a579ebc9a6c51297eb647c7a2` |
| `certify.py` | 50,212 | `468ce8418bff0f43582d341f3e1b8064cf1c429301d1f56be84b8e13ad59a3f1` |
| `metrics.py` | 17,608 | `1dadf08e89330bed2ab773afda05bffcb3fe2252b0d3e23c0bcc18c2c3b049dd` |
| `evaluator.py` | 21,977 | `f981c6c7061c3388f06ef39466ef98192e65c62c501821be2b211f903cfaf9de` |
| `decode.py` | 8,843 | `766cd6ca4b9e133041bd602f50fde840ec7caadd376bd6851799583d44f31f5e` |
| `fixtures.py` | 36,365 | `979b8cee4fed117f5471cf399794f615c60eae52dbba2c946109ff836206b464` |
| `gates.py` | 5,929 | `b6c9d7432443d7ad36738a81700746709fd571d475514aa6f1bbcd39be744b4a` |

The JSON freezes the current 34 local import edges, the module-to-path map, and
the implicit package initialization order. Its canonical dependency-DAG digest
is `e3a6170a5d54a3bd9373939351caabeb70cbaeef5695c5aa26b3de94bb261073`.
A corrected candidate may change edges, but fresh review must recompute and
bind its complete exact edge array; every local target must remain inside the
17-member set. Dynamic import, namespace packages, local zip import, import-path
mutation before audit, and unlisted origins are `STOP-METRIC`.

The exact dependency contract digest is
`12ae175a3934a450ddc389291d1e83152c034068de641434be0017d39ee6cc7c`.
It distinguishes direct-source allowlists from transitive runtime imports.
Production direct dependencies include the listed standard-library modules,
NumPy 2.1.0, and SciPy 1.14.1
(`scipy.interpolate`, `scipy.spatial`, `scipy.spatial.distance`,
`scipy.special`) reached through evaluator → certify → x0/preprocess. Every
transitive helper must resolve to an exact locked CPython/stdlib, NumPy, SciPy,
or recursive native-image member and appear in the actual per-process
module-origin manifest. “NumPy only” is not a valid production closure.

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


## 6. Independent reference source and runtime boundary

The reference scorer source remains the exact ASCII/LF block below, with one
terminal LF and SHA-256
`b23bf344a10f84a583c38e2326efd5ca70d4c4462fbf76715cd32b9afd20922f`.
The exact runner member must contain a byte-identical literal; fresh candidate
review extracts the Markdown block and runner literal and compares both bytes
and hash before acceptance. The reference worker compiles it with filename
`<temporac-p05m-reference-v1>` and calls `execute(fixture_input)`.

The reference runtime contract digest is
`3343fdcf99eba48c3c21ad2b65008855cd398cdfa04ed66d7a9ee634e9a32a3d`.
It fixes Linux x86_64 little-endian CPython 3.12.4 and NumPy 2.1.0. It permits
only `__future__`, `collections`, `copy`, `hashlib`, `json`, `math`,
`struct`, and `numpy` as direct imports. Every actual transitive module
origin must lie in the exact locked CPython or NumPy tree and be embedded in the
reference process witness. Every `pams` and SciPy origin is forbidden even
though the production runtime lock contains those trees. The production and
reference scorers execute in distinct fresh OS processes with distinct
challenges. Neither scorer may import, compile, call, or inspect the other
scorer branch; common runner code supplies transport and evidence only.

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

## 8. Exact outer runner and runtime closure

The outer runner path is exactly
`tests/temporac/test_p05m_fixture.py`. It is one raw-byte member, not a
directory glob. Its closure preimage is ASCII
`temporac.p05m.outer-runner-closure.v1`, one NUL, followed by the same
path-length/path/raw-length/raw-byte encoding used above. The candidate index
binds both the raw file SHA-256 as `runner_code_sha256` and the whole preimage
as `runner_closure_sha256`. No repository-local helper outside the 17-member
production closure may be imported or executed. The candidate snapshot
preimage is ASCII `temporac.p05m.candidate-snapshot.v1`, one NUL, then those
17 members followed by this runner member under the same encoding. The accepted
candidate index binds the exact 18 records and snapshot root.

The candidate index also closes three role-keyed containers
(`controller`, `production`, `reference`) for command templates and their
canonical hashes. Its embedded candidate dependency DAG has exactly
`edges`, `module_origins`, `package_initialization_order`, and `schema`;
the origin map is the fixed 17 module/path pairs and package order is exactly
`pams,pams.temporac`. Member records are exactly `bytes,path,sha256`.
`candidate_id` is a pre-review lowercase ASCII token matching the fixed JSON
grammar. These rules leave no candidate-index container open-ended.

The runner has exactly three modes in the same reviewed byte string:
`controller`, `worker/production`, and `worker/reference`. It has no
top-level production import. Controller mode may load only the runner plus
locked CPython origins; production mode may additionally load the 17-member
`pams` closure plus locked NumPy/SciPy; reference mode may additionally load
only the compiled reference and locked NumPy. The JSON freezes the direct runner
allowlist and forbids every other origin.

All processes use exactly Linux x86_64 little-endian CPython 3.12.4, NumPy
2.1.0, and SciPy 1.14.1, with absolute tokenized executable
`$RUNTIME/bin/python3.12`, flags `-I -S -B`, empty read-only working root,
and exact ordered `sys.path`:

1. `$CANDIDATE/src`
2. `$RUNTIME/stdlib/python312.zip`
3. `$RUNTIME/stdlib/python3.12`
4. `$RUNTIME/stdlib/python3.12/lib-dynload`
5. `$RUNTIME/site-packages`

With `-I -S` active, controller first verifies the interpreter bootstrap
paths, then replaces `sys.path` with that exact array before any
candidate-local, NumPy, or SciPy import. Worker role imports occur only after
that transition. An earlier local/third-party import, extra/missing/reordered
path, or origin mismatch is `STOP-METRIC`.

The runtime-lock schema, whose contract digest is
`d693ec66170d3e66ac1d92a049b9b57dd80ce788c38907369bbb335f426d28b1`,
contains the exact interpreter executable, full stdlib tree, exact NumPy and
SciPy installed trees, their wheel and RECORD bytes, and the recursive ELF
interpreter/`DT_NEEDED`/executable-mapping native closure. Tree members use
forward-slash relative NFC UTF-8 paths sorted by raw UTF-8 bytes; each tree
preimage begins `temporac.p05m.runtime-tree.v1\0` and appends
path-length/path, `uint32_be` low-12-bit POSIX mode, raw-length, and raw bytes.
The runtime lock fixes executable, stdlib/site-packages roots, wheel path, and
RECORD relative path. Locked roots admit only mode-0555 directories and
individually listed regular files; symlinks, hard links, devices, sockets, and
FIFOs are forbidden. Missing wheel, RECORD, installed member, extension, shared
object, mode, or hash is `STOP-METRIC`.

The environment is exactly the closed map in JSON:
`LANG=C.UTF-8`, `LC_ALL=C.UTF-8`, `TZ=UTC`,
`PYTHONHASHSEED=0`, `PYTHONDONTWRITEBYTECODE=1`, all four listed numerical
thread counts equal to one, and `CUDA_VISIBLE_DEVICES` empty. A fresh network
namespace exposes no non-loopback interface; socket creation/connect/bind are
denied; no GPU device is mounted. Every process witness embeds the complete
post-scoring `sys.modules` origin manifest. Source, bytecode, and extension
entries bind their exact locked raw files. Built-in and frozen entries bind the
exact CPython executable image. A native-image manifest binds all executable
mappings. Both manifest hashes cover their full canonical arrays plus one LF.
The reference manifest also carries a required compiled-source pseudo-entry
binding `<temporac-p05m-reference-v1>` to the exact reference length/hash even
if its execution dictionary is not inserted into `sys.modules`. These observed
manifests close transitive helper and module-origin ambiguity.

The separately closed sandbox-profile contract has canonical SHA-256
`a1c65c29d8482cc6f266cde3869c1f46bd57052b0d24f58972e55df0ebf08e07`.
It binds exact compiled policy bytes/hash; read-only `$CANDIDATE` and
`$RUNTIME` mounts; one new empty private `$WORK/empty` mount; no other
visible mount; zero GPU device nodes; denied socket create/connect/bind; exactly
two controller children with roles production/reference; and zero worker
children. Every process witness proves the same policy was active for its whole
lifetime. Candidate bytes embed the policy in unique padded RFC 4648 base64,
bind its decoded byte length and SHA-256, and reject any whitespace or alternate
encoding.

The command templates are exact. Their argv arrays are:

- controller: `$RUNTIME/bin/python3.12 -I -S -B
  $CANDIDATE/tests/temporac/test_p05m_fixture.py --p05m-mode controller
  --challenge-hex <CHALLENGE_HEX_64>`;
- production: the same through the runner path, then `--p05m-mode worker
  --role production --challenge-hex <CHALLENGE_HEX_64>`;
- reference: the same with `--role reference`.

The only substitution is one 32-byte challenge rendered as 64 lowercase hex
characters. The P2 verifier creates the controller challenge; the controller
creates independent production and reference challenges with `os.urandom`;
all three differ. Stdin is exactly one canonical closed request plus LF, stdout
is exactly one canonical role-specific evidence object plus LF, successful
stderr is empty, and timeout is 600 seconds. Actual argv, environment,
executable, stdin/stdout/stderr, sandbox, pid/parent pid, module origins, native
images, runtime, role, challenge, and working-directory token are bound in each
process witness.

Any runner/worker transport or verification failure exits 2 with empty stdout
and exactly one canonical `temporac.p05m-runner-error.v1` object plus LF on
stderr. Its closed error vocabulary is
`P05M_REQUEST_INVALID`, `P05M_RUNTIME_MISMATCH`,
`P05M_PROCESS_MISMATCH`, `P05M_OBSERVATION_MISMATCH`,
`P05M_ATTACK_MISMATCH`, `P05M_TIMEOUT`, or `P05M_INTERNAL_ERROR`;
status is `GLOBAL_FAIL`. No role observation, runner evidence, gate evidence,
or partial output exists on that path.

The sandbox binding is not a bare hash. Each process witness embeds the closed
`temporac.p05m-sandbox-witness.v1` object and its canonical hash. It binds
candidate snapshot, runtime and compiled policy; zero GPU devices; zero
non-loopback interfaces; denied socket syscalls; the exact initially empty work
directory; the role; and child count (controller two, each worker zero), with
`status=ACTIVE_FOR_ENTIRE_PROCESS`. The empty-directory witness is SHA-256 of
`ASCII("temporac.p05m.empty-directory.v1") || NUL`, namely
`f5915c913731b3bd1e3487717bfa460cb3acf6798f0069eb9a5f9c9d05b5b29a`.

The fresh process identity is exactly SHA-256 of:

`ASCII("temporac.p05m.process-identity.v1") || NUL ||
 uint64_be(pid) || uint64_be(parent_pid) || raw32(role_challenge) ||
 raw32(actual_argv_sha256) || raw32(executable_sha256)`.

The spawning parent verifies each reported pid against its OS process handle
and parent pid against its own pid. Production, reference, and controller pids
must all differ.

## 9. Directionally closed observations and sole gate evidence

There are exactly 17 attack definitions. Each definition is actually executed
once by production and once by reference, so each role witness proves one
normal scorer call and 17 attack scorer calls. This does not create 34 attack
definitions. Each attack call is atomic: the scorer produces its exact
`GLOBAL_FAIL` error code and no point, bootstrap, draw, row, or partial
payload. A worker failure emits empty stdout and nonzero exit; it can never
leave a partial evidence object.

A `temporac.p05m-observation-root.v2` has exactly:
`attack_observed_sha256`,
`candidate_conformance_index_sha256`,
`clean_draws_raw_f64le_sha256`, `command_role`,
`component_order_sha256`, `fixture_input_sha256`,
`implementation_sha256`, `materialized_rows_sha256`,
`normal_observed_sha256`, `role_challenge_sha256`,
`route_draws_raw_f64le_sha256`, `runtime_lock_sha256`, and `schema`.
Production uses the complete 17-member closure as `implementation_sha256`;
reference uses the reference-source hash.

A `temporac.p05m-role-observation.v1` has exactly:
`candidate_conformance_index_sha256`, `command_role`,
`fresh_process_identity_sha256`, embedded `observation_root` and its hash,
`role_challenge_hex`, `runtime_lock_sha256`, `schema`, and
`status=OBSERVED_COMPLETE`. Its complete canonical bytes are the child's
entire stdout. The parent process witness binds that stdout hash, the same
observation-root hash, role, challenge, command template and actual argv,
executable, runtime, pid/parent pid, stdin, environment, module origins,
native images, and sandbox. Thus command/process identity → stdout role
observation → observation root → actual normal/attack/row/draw bytes is one
directed chain. The two roles cannot be exchanged because role, implementation
hash, independent challenge, argv, pid, and origin policy are all bound on that
chain.

The controller launches exactly one distinct fresh production child and one
distinct fresh reference child. Only after verifying both directed chains and
every fixed equality may it emit one
`temporac.p05m-runner-execution-evidence.v1` with
`status=OBSERVED_COMPLETE`. That closed object embeds both role observations,
both child process witnesses and their hashes; it binds the fresh candidate
review, mandatory inner summary, runner/runtime/source hashes, all fixed
artifact commitments, all three challenges, and exact invocation counts. The
controller's entire stdout is that canonical object plus LF. It is observation
evidence and has no PASS semantics.

After controller exit, the P2 verifier constructs the controller process
witness. For this role, `runner_execution_evidence_sha256` binds the exact
stdout object and `observation_root_sha256` is 64 zeroes; for child roles the
reverse sentinel rule applies. The verifier then builds exactly one closed
`temporac.p05m-gate-evidence.v1` containing:

- external amendment JSON hash;
- the complete candidate index and its hash;
- the complete fresh candidate-review receipt and its hash;
- the mandatory complete six-key inner receipt and its hash;
- the complete runner-execution evidence and its hash;
- the complete outer controller process witness and its hash;
- the P2-verifier 17-member closure hash;
- the plan/tracker semantic-rebinding hash;
- `schema` and `status=PASS`.

The gate-evidence object contains no self hash. P2 can write its PASS only after
recursively revalidating every embedded byte/hash, both child executions, the
controller execution, one normal plus all 17 attacks per implementation, 288
rows, both 10,000-draw raw-f64 streams, all roots, and the accepted fresh review.
This complete, fresh-review-bound outer gate evidence is the sole P05M gate
input for P2, S0, G5a, and K7.

The legacy `temporac.count-metric-fixture-receipt.v1` remains exactly six keys:

- `attack_expected_global_fail_sha256`
- `evaluator_code_sha256`
- `expected_output_sha256`
- `fixture_input_sha256`
- `schema`
- `status`

Its exact bytes and digest are mandatory subordinate evidence at P2, embedded
inside gate evidence, committed by S0, and revalidated at G5a/K7. It is never
optional, but it is also never sufficient or accepted directly. Its
`status=PASS` means only that the hash summary is syntactically complete and
addresses the frozen artifacts. It grants no execution, review, P2, S0, G5a,
K7, capability, launch, science, or paper conclusion. The unique field name is
`attack_expected_global_fail_sha256`; `attack_global_fail_sha256` is
forbidden. An attack's `status=GLOBAL_FAIL` retains only the atomic failure
meaning above.

No amendment, candidate index, role observation, runner evidence, or gate
evidence contains its own digest. Role stdout is bound by a parent witness;
runner stdout is bound by the outer witness; gate evidence is hashed externally
by S0. No PASS-bearing object may certify only hashes it generated or pair
unlinked sibling hashes.

## 10. Plan/tracker rebinding and unique stage/consume order

The current plan and tracker bytes still use the legacy receipt phrase at their
P05M interfaces. This amendment does not edit them. Instead, the closed
`temporac.p05m-plan-tracker-semantic-rebinding.v1`, canonical SHA-256
`0c77607394c413cff2d6d8909ddd1c4538462b87e037e4c70a324c398a4fc250`,
binds plan
`4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e`
and tracker
`714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b`.
At the named plan interfaces P2-METRIC, M-S0, P2 count-metric fixture, G5a,
M-G5B/K7, G5b/K7, and metric-drift guard, and tracker interfaces
P05M-COUNT-METRIC, S0-COMMIT, G5A-COMMIT, K7-FINAL, and the P05M checklist,
every legacy “count-metric fixture receipt” phrase denotes the complete
`temporac.p05m-gate-evidence.v1` object. The old six-key schema denotes only
its mandatory nested `inner_summary_receipt`. This semantic overlay becomes
effective only after fresh acceptance of this amendment and the complete
candidate; before then all stages remain blocked.

The only conforming order is:

1. **P2:** verify history only as provenance; require the fresh-reviewed
   candidate; require the inner summary; launch one fresh controller, which
   launches one fresh production and one fresh reference child; execute normal
   plus all 17 attacks in both; verify the full directed chain; construct the
   sole gate-evidence PASS.
2. **S0:** accept only exact canonical gate-evidence bytes. Recursively verify
   every mandatory nested object/hash, then commit the external gate-evidence
   digest and the immutable content-addressed candidate snapshot. There is no
   optional inner or outer P05M artifact.
3. **G5a:** load only S0-bound gate evidence/snapshot; reject a standalone
   six-key receipt; recompute all 17 raw members and the whole production
   closure; require candidate/P2/S0 equality; write only that digest into the
   existing closed `evaluator_code_sha256` field. Do not add P05M keys or
   compute human statistics.
4. **Capability grant:** issue only after G5a under the unchanged external
   authority. It is still unconsumed.
5. **K7 prestart:** before constructing or starting the one-use evaluator and
   before any consumed-state transition, recursively verify the exact S0 gate
   evidence and fresh review; recompute runner, all 17 production members,
   dependency DAG, runtime trees, origin/native policies, command templates,
   candidate snapshot, all three witnesses, both observations, and every
   normal/attack/row/draw root from the same immutable snapshot to be launched.
   Require gate/S0/G5a/grant equality on the whole production closure. Any
   missing or mismatched byte is `STOP-METRIC` and consumes nothing.
6. **Atomic start:** after successful prestart only, start the sole
   non-resumable evaluator from the already verified read-only
   content-addressed snapshot. Successful OS process creation is the one-use
   capability-consumption point. Then preserve the unchanged G5b → K7 order.

This is a P2/K7 verification subprotocol, not a new scientific gate. F23 remains
exactly
`S0 -> S1:G0,K0 -> S2:G1,K1,G2,K2 -> S3:G3,K3,K4 ->
 S4:G4,K5,K6 -> freeze natural artifacts -> G5a -> capability grant ->
 G5b -> K7`.
No failure may be filtered, repaired, retried, resumed, or converted into a
local skip.

## 11. Current implementation mismatch and zero authority

The currently audited source is deliberately not a candidate PASS:

- `metrics.py` byte-sorts component tokens and
  `paired_component_bootstrap` has no explicit frozen component order;
- `evaluator.py` now validates the old six-key receipt and some target
  commitments before `_consumed=True`, but it has no mandatory outer
  gate-evidence, runner/process/observation, complete-closure, runtime-origin,
  or immutable-snapshot K7 prestart verification;
- `count_metric_fixture_receipt` hashes caller-supplied bytes and emits PASS
  without executing either scorer or any attack;
- `gates.py` accepts arbitrary syntactically valid digest strings for P2;
- `fixtures.py` still exposes the forbidden
  `attack_global_fail_sha256` name;
- the exact outer-runner path does not yet exist.

The prior “stable” Round 2 review hashes
`hashio=2d81e415b45e3d1b612f998b6a8386a967dfdfcfaac31bc914b2c95cbaeeab3a`
and
`receipts=0ac800d96f536a41a67775bc8e8e778fe14c312c02f642f7cde78d210244016d`
are retained only as superseded provenance. Current audited hashes are
`aa460d5de51f107b2fc6b2ad171747e0da38a87cf30f82a869a335c268f1d613`
and
`564039b5d37a4a2759544a9ae65f267b57d7f6c1e46bca812ae5ed4f4a2c17e4`.
The JSON records equivalent source drift for contract, evaluator, types, and
fixtures, as well as the latest certificate-postfix/operator review inputs.
Neither old nor current hashes self-lock the future corrected candidate.

Therefore the only present conclusions are
`PROPOSED_PENDING_FRESH_REVIEW`, same-family provisional, and authority zero.
No review, implementation, fixture, P2, experiment, server, data, training,
capability, launch, Git, or paper action is authorized by these bytes.
