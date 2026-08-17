# TempoRAC Contract Amendment 003 — P05M Count-Metric Fixture

**Amendment ID:** `TEMPORAC_CONTRACT_AMENDMENT_003_P05M_FIXTURE`  
**Schema:** `temporac.contract-amendment.p05m-fixture.v3`  
**Status:** `PROPOSED_PENDING_FRESH_REVIEW`  
**Review class:** same-family provisional  
**Authority:** no gate, launch, capability, data, server, experiment, fixture, test, training, Git, paper-promotion, or claim authority

## 1. Normative posture and exact non-changes

This Round-4 draft closes only the two blocking groups in the fresh Round-3
`REVISE` review: an actual parent-observable process-attestation transport,
and a uniquely constructible closure for the P2 PASS writer. It does not make
P05M or P2 PASS, does not materialize a candidate, and authorizes no
implementation or execution. A fresh independent reviewer must accept both
amendment mirrors and, later, the complete candidate index before any P05M
process can run. The JSON companion is the byte-level authority when prose and
a closed JSON schema differ.

The scientific contract is unchanged:

- F20 person NAE/OBO, positive integer ground truth, and stub estimate zero;
- F21 route margin and clean-retention delta, with the same signs;
- people-within-video, equal-video, then ordered-three-seed aggregation;
- comparator order `(global,uniform,capacity-control)`, per-draw reselection,
  and reuse of the drift-selected comparator on paired clean;
- one 10,000-draw PCG64 stream, component multiplicity, and sorted indices
  249/9749;
- the exact 288 synthetic rows, eight-component nonlexical order, 17 attacks,
  normal canonical JSON, and raw-f64 draw/root bytes;
- seeds `20260815,20260816,20260817`, 27 training jobs, 93 allocated A6000
  hours, one claim, four experiment blocks, every threshold, and F23 order.

This amendment adds no natural row, label, source identity, model, job, GPU
work, threshold, result, claim, or paper block. It does not reuse WARP.

## 2. Historical provenance and fresh-candidate separation

`historical_audit_basis.v3` contains 65 ordered records. The 60
filesystem-backed records bind every current protocol mirror, canonical
proposal/plan/tracker/research contract, three review rounds, rejected
amendment archives, certificate-postfix/operator reviews, the current 17-source
production closure, and all current TempoRAC tests read for this revision. The
five original `skill://` rows remain immutable historical provenance. The
canonical historical-basis digest is
`9f2c09addedd942e5781a0823c2b359ce3872bdcb12df62c84d2325316f2fb03`.

Rejected candidates are preserved without self-reference:

| Archive | Bytes | SHA-256 |
|---|---:|---|
| Round-2 MD `..._20260816_145644.md` | 29,722 | `d615def9dc468b7e483488b87cb0524302f1c4c325df8bf092e741e21f6262d1` |
| Round-2 JSON `..._20260816_145644.json` | 29,094 | `c73b85731ece838684f78c41b3dfcdb498af5cf9822f7ee84d2add4d09ffe1fc` |
| Round-3 MD `..._20260816_162320.md` | 45,024 | `7ed7dd65adba298a176cfef52ea6bc5464a664ec7d664ef49995ec154895f71b` |
| Round-3 JSON `..._20260816_162320.json` | 81,019 | `09b7a835bcc3e3d3c5193b23830c1afc97811665953aaefffc867d8abcf619c7` |

The fresh Round-3 review inputs are MD 23,288 bytes,
`7055c41896a728d7612e352b390d8e8a02c040a1bfe086a8a51dc162a3e734b7`,
and JSON 22,469 bytes,
`35d8640283a75efef7c0b4fe6a53f7cba803ceb3d49dc58dd7f680e0102700be`.
Current project-local protocol hashes are:

| Input | Bytes | SHA-256 |
|---|---:|---|
| `docs/BASELINE_PROTOCOLS.md` | 7,623 | `2084578b70797bf2988768848b96b23f20013d92ff32d13334166350d19d910d` |
| `.agents/skills/research-refine/SKILL.md` | 30,984 | `bb489173329fec7a6551c74c319e2103d83f722ea11fc11462a7c4ccacd00b09` |
| `output-versioning.md` | 4,828 | `de8e7ac23069de6c5d0c482d4cef117dd5a543e00b57e2dc6c62560a457290c5` |
| `output-manifest.md` | 1,753 | `e767083e0ef97a7adf7a4e8a574090067a59c0655d5aca422a3747cd32b6afde` |
| `output-language.md` | 2,215 | `0f1447579bd7a2195fd4a074be70ae6345b5c368f1a70d327e89748bf5a378f9` |
| `output-composition.md` | 5,293 | `79a72c7ea02102c460ae3ad55853c21a4054b9df93f377c29dd34555e6d3c3a7` |

The canonical proposal, plan, tracker, and research contract remain,
respectively,
`3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391`,
`4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e`,
`714c6b690d2c2318d2e18223efa617e55fe746a7f72279e820bf5e799a9b525b`,
and
`3e037a2f74c49dd62e596dc7b875d94216e45b7f693b80a7afb6a60f8244367b`.

Historical bytes are provenance only. P2 may verify them for audit continuity
but must never compare a candidate member against a historical member as an
acceptance condition. A future
`temporac.p05m-candidate-conformance-index.v3` has no self hash and becomes
authoritative only through a separate
`temporac.p05m-candidate-review-receipt.v3` whose closed fresh-independent
witness binds the exact amendment MD/JSON hashes, candidate index/snapshot,
attestation contract, gate-bundle contract, and P2-verifier closure. Candidate
hashes may differ. Neither object has gate or launch authority.

## 3. Exact production and P2-verifier executable closures

Normal CPython package import is mandatory. The 17-member production closure is
ordered exactly as follows:

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

Its preimage is ASCII
`temporac.p05m.production-executable-closure.v2`, one NUL, then, for every
ordered member,
`uint64_be(path_length) || path_ascii || uint64_be(raw_length) || raw_bytes`.
The currently audited preimage is 316,199 bytes and hashes to
`dfcf258b13aa571d3636101f09e66c9d7889a0f09dd88515f4c332ae2d662943`.
It is provenance, not a candidate lock. `evaluator_code_sha256` always means
the freshly reviewed candidate's recomputed 17-member root.

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

The audited 34-edge acyclic DAG remains
`e3a6170a5d54a3bd9373939351caabeb70cbaeef5695c5aa26b3de94bb261073`.
The updated dependency contract, including exact runner imports and four role
origin policies, is
`1f1747012f047d8db69944391f3a63e11ada17edc163697608d3aed6f2758f6c`.
All transitive CPython, NumPy 2.1.0, and SciPy 1.14.1 origins and native images
must be inside the accepted runtime trees and the actual attestation.

The P2-verifier closure is not a bare digest. It is exactly the above 17
members followed by member 18
`tests/temporac/test_p05m_fixture.py`. Its prefix is exactly
`temporac.p05m.candidate-snapshot.v1\0`; its per-member path/raw-length
encoding is identical to the production rule. In the candidate index:

`p2_verifier_closure_members == candidate_snapshot_members`

and

`p2_verifier_closure_sha256 == candidate_snapshot_sha256 ==
SHA256(the exact 18-member preimage)`.

Both equalities are required by fresh candidate review, the P2 capture step,
S0, and K7. No independent hexadecimal value is accepted. The runner member
contains the exact PASS writer
`main -> _run_p2_verifier -> _emit_gate_evidence_pass` and the non-PASS
capture routine `main -> _capture_p2_gate_bundle`. The fixed closure-contract
digest is
`d297372d44e60e7c2443c6295f99ef6cfed3afe7e21b4d993ad81e9cdd7396d9`.
Because the runner does not yet exist, the authoritative 18-member root is
intentionally absent until implementation plus fresh review; this is not an
open field and cannot be filled by an arbitrary 64-hex string.

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

## 8. Exact runner, attestation transport, runtime, and work isolation

The sole runner path is
`tests/temporac/test_p05m_fixture.py`. Its one-member closure prefix is
`temporac.p05m.outer-runner-closure.v2\0`, followed by the same exact
path/raw-byte encoding. The candidate index binds its raw hash, runner-closure
hash, all 18 candidate/P2 members, and four command templates:

| Role | Exact mode | stdin | stdout | success stderr | work root |
|---|---|---|---|---|---|
| P2 verifier | `--p05m-mode p2-verifier` | `temporac.p05m-p2-verifier-request.v1` | `temporac.p05m-gate-evidence.v2` | empty | `$WORK_P2_VERIFY` |
| controller | `--p05m-mode controller` | `temporac.p05m-runner-request.v2` | `temporac.p05m-runner-execution-evidence.v2` | empty | `$WORK_CONTROLLER` |
| production | `--p05m-mode worker --role production` | `temporac.p05m-worker-request.v2` | `temporac.p05m-role-observation.v2` | empty | `$WORK_PRODUCTION` |
| reference | `--p05m-mode worker --role reference` | `temporac.p05m-worker-request.v2` | `temporac.p05m-role-observation.v2` | empty | `$WORK_REFERENCE` |

Every command begins with the locked
`$RUNTIME/bin/python3.12 -I -S -B`, uses the exact environment, replaces only
the final 64-hex challenge token, and has a 600-second timeout. P2 verifier
launches exactly one controller; controller launches production and, only
after fully closing and validating it, reference. Workers launch no child.
Exactly one normal and all 17 attacks run in each implementation.

### 8.1 Dedicated FD-3 attestation channel

Each of the four roles receives one additional output and no other side
channel: an anonymous pipe whose child write descriptor is exactly FD 3. The
immediate parent creates it with `pipe2(O_CLOEXEC)`, retains only the read
end, maps only the write end with `POSIX_SPAWN_DUP2`, and closes every
unrelated nonstandard child FD. The P2-stage capture root uses parent read FD 3
for the P2 verifier; P2 verifier uses read FD 4 for controller; controller uses
read FD 4 for production and, after closure, a new read FD 4 for reference.
Every launch has a distinct kernel pipe, role challenge, and attestation nonce.
Before pipe creation the P2-stage capture root has exactly FDs
`[0,1,2]`, forcing Linux to allocate read/write FDs 3/4; an attested parent
has exactly `[0,1,2,3]`, forcing allocation 4/5. Only the write end is mapped
to child FD 3. Any other prelaunch set, allocation, or inherited descriptor
fails; production closes both parent pipe ends before reference reuses FD 4.

The channel ID preimage is exact: tag
`temporac.p05m.attestation-channel.v1\0`, length-prefixed role, candidate
index hash, role challenge, attestation nonce, independently observed kernel
pipe-token hash, parent PID, child PID, Linux start-time ticks, child write FD,
and parent read FD, with the integer widths frozen in JSON. The parent's
retained pipe link and child FD-3 pipe link must hash to the same
`pipe:[inode]` token. Role, challenge, nonce, process identity, command
template, candidate snapshot, runtime, output hash, and every actual witness
are closed fields in
`temporac.p05m-process-attestation.v1`.

The child must:

1. finish scoring/verification and construct the exact stdout object bytes;
2. take two byte-identical complete module, native, and sandbox snapshots;
3. write exactly one canonical attestation plus one LF to FD 3;
4. never call `fsync` on the pipe; close FD 3 exactly once;
5. only after close, write the already hashed output plus one LF to stdout,
   flush, and exit 0 without a late import/FD/mount change.

For a pipe, `fsync` is forbidden and EOF after writer close is the commit
boundary. The parent drains only its retained FD through EOF before accepting
stdout, with a hard 16,777,216-byte limit. Zero bytes, extra bytes, BOM, CR,
noncanonical JSON, duplicate keys, wrong LF count, role/nonce/challenge/channel
swap, output mismatch, or another transport is
`P05M_ATTESTATION_MISMATCH`. O_EXCL sidecars, writable work directories,
shared memory, sockets, stderr/stdout attestation, or any inherited sibling FD
are forbidden.

### 8.2 Actual post-scoring completeness

The attestation carries, not merely hashes:

- the complete actual non-null `sys.modules` origin array after scoring or P2
  validation;
- the complete actual executable `/proc/self/maps` set plus recursively
  resolved ELF interpreter/DT_NEEDED closure;
- the actual sandbox witness, including open FDs exactly `[0,1,2,3]`, proc
  source hashes, GPU/network/policy state, spawned-child count, and work-mount
  identity.

Two consecutive arrays must be byte-identical and no import is permitted
afterward. The parent recomputes all hashes from received attestation bytes and
may never substitute an expected allowlist. The process witness embeds the
exact attestation and binds actual argv/environment/executable, PID/parent PID,
Linux start ticks, pidfd exit, stdin/stdout/stderr, output schema/root, runtime,
candidate/P2 closure, pipe token, and parent-observed final work mount. The
runtime-lock contract digest is
`bff68eb3db94c4bf5e56e5ba8e2d6f1e961305f5752e6ce748d295f70eb2ba0a`;
the transport-contract digest is
`f01223582fe6ac29112ead3cf172f6ea04664a84a202b30668a92774a238ba47`.

### 8.3 Four distinct empty read-only work mounts

`$WORK_P2_VERIFY`, `$WORK_CONTROLLER`, `$WORK_PRODUCTION`, and
`$WORK_REFERENCE` are distinct, initially empty, read-only mounts. They have
pairwise-distinct Linux mount IDs, st_dev/st_ino pairs, and independent 32-byte
mount nonces; no role namespace resolves another role token. The in-process
witness binds the role/challenge/nonce, mount identity, and both empty-tree
roots. The parent retains a mount handle and, after exit, requires the same
identity and final empty root. Any extra file, directory, link, device, socket,
FIFO, changed root, shared identity, writable bit, or cross-role visibility is
`P05M_WORK_MOUNT_MISMATCH`. There is no attestation file. The sandbox-profile
digest is
`1c6be7154c6b8a5727fc1bc22e83b6796855614990a1c778819d9e6c17be5c2a`.

## 9. Directional evidence, noncircular P2 PASS, and the sole gate input

Production/reference stdout remains a closed nine-key role observation. Its
observation root binds role, challenge hash, candidate/runtime,
implementation root, fixture/rows/component/normal/attack roots, and both
raw-f64 draw roots. Before stdout, the same process attests the exact
role-observation hash on FD 3. The parent process witness therefore
directionally binds actual post-scoring state to the exact stdout/root; a
copied expected manifest or swapped sibling observation fails.

The controller embeds both observations and both child witnesses in
`temporac.p05m-runner-execution-evidence.v2`. It has no PASS semantics.
Before emitting that object, controller attests its exact hash and its own
post-validation state. The P2 verifier derives the controller witness only
from that retained pipe, stdout, pidfd, and mount handle.

The P2 verifier first recomputes the exact 18-member equality, validates the
fresh candidate review and every nested scientific/process equality, then
constructs
`temporac.p05m-gate-evidence.v2` with `status=PASS`. Its closed keys bind
the amendment, candidate/review, mandatory six-key inner summary,
controller/runner evidence, P2 member array/root, transport and plan-overlay
hashes. It then attests those exact gate bytes on FD 3 before stdout and exit.

A P2 process cannot embed its own later exit/stdout witness without a hash
cycle. Therefore gate evidence alone is explicitly inert. The P2-stage capture
root constructs, only after attestation EOF, exact stdout, empty stderr, pidfd
exit 0, and final empty-mount verification, the closed non-PASS object
`temporac.p05m-gate-capture-bundle.v1`:

`gate_evidence, gate_evidence_sha256,
p2_verifier_process_witness, p2_verifier_process_witness_sha256,
schema, status`.

Its status is exactly `CAPTURED_COMPLETE`, never PASS. Its contract digest is
`46bb12eba46483b4846db41bcd67f437cc4965e74a94890f132b4aa14a8d9bcf`.
Only the complete canonical bundle and its external hash can trigger the P2
transition or be consumed by S0/G5a/K7. The nested PASS was written by the
reviewed P2-verifier closure; the surrounding capture object contributes no
authority and prevents circular self-certification.

The legacy six-key
`temporac.count-metric-fixture-receipt.v1` remains mandatory inside the
nested gate evidence with exactly
`attack_expected_global_fail_sha256,evaluator_code_sha256,
expected_output_sha256,fixture_input_sha256,schema,status`. The unique attack
field name is `attack_expected_global_fail_sha256`. The inner receipt is
never optional and is never sufficient; gate evidence alone is also never
sufficient.

Every failure is atomic. Before successful attestation commit, FD 3 closes
with zero bytes. A failure after commit causes the parent to discard every
byte and emit no witness/bundle. Successful stderr is empty. A closed
`temporac.p05m-runner-error.v2` may appear only on stderr with exit 2; stdout
is empty and no partial score, draw, observation, runner evidence, gate
evidence, or bundle is accepted.

## 10. Plan/tracker overlay and unique P2/S0/G5a/K7 order

The unchanged plan and tracker bytes are semantically rebound by
`temporac.p05m-plan-tracker-semantic-rebinding.v2`, canonical digest
`48770609fe6ecbbe8ee456b9c046ab5fb5e8f87e06d1fbe00fc1771b3f7c7e90`.
At every named P05M interface, a phrase naming the old surrounding receipt
means the complete gate-capture bundle. This round does not edit plan/tracker.

- **P2:** historical bytes are provenance only. Fresh review must bind the
  candidate. The exact P2 verifier executes controller→production→reference,
  the parent captures its actual FD-3/process/runtime/module/native/sandbox
  state, and only the complete bundle can transition P2.
- **S0:** recursively validate and commit the exact bundle bytes/hash,
  candidate/review, 18-member P2 closure, 17-member production closure, all
  four process witnesses/attestations, both observations, and immutable
  snapshot. No artifact is optional.
- **G5a:** load only the S0 bundle/snapshot. Recompute both closures. Write only
  the 17-member production root into the existing closed
  `evaluator_code_sha256`; add no P05M key and compute no human statistic.
- **K7 prestart:** before constructing or starting the evaluator and before any
  one-use state transition, recursively verify the bundle, fresh review,
  runner, 17/18 member closures, dependency DAG, runtime trees, four commands,
  transport, mount identities, process/observation/root equalities, and
  agreement among bundle/S0/G5a/grant. Missing/mismatch is `STOP-METRIC` and
  consumes nothing.

Only after all K7 prestart checks succeed may the sole non-resumable evaluator
process be created; successful OS creation is the capability-consumption
point. Existing G5b→K7 order is unchanged. F20/F21, 10,000 draws, 288 rows,
17 attacks, 27 jobs, 93 hours, and claims remain unchanged.

## 11. Current mismatch, hashes, review requirement, and zero authority

The current source is still a historical nonconforming basis. In particular,
production component order is not the frozen nonlexical explicit order;
evaluator/gates accept legacy digest assertions rather than the mandatory
bundle; the fixture receipt hashes caller-provided bytes and declares PASS
without executing both scorers/attacks; the old attack field name remains in
fixtures; and no runner, FD-3 attestation, four read-only mount witnesses,
18-member P2 closure, P2-verifier process witness, or gate-capture bundle
exists. This amendment changes none of those files.

Frozen contract hashes after this Round-4 draft are:

| Object | SHA-256 |
|---|---|
| historical audit basis | `9f2c09addedd942e5781a0823c2b359ce3872bdcb12df62c84d2325316f2fb03` |
| plan/tracker semantic rebinding | `48770609fe6ecbbe8ee456b9c046ab5fb5e8f87e06d1fbe00fc1771b3f7c7e90` |
| historical production dependency DAG | `e3a6170a5d54a3bd9373939351caabeb70cbaeef5695c5aa26b3de94bb261073` |
| evaluator dependency contract | `1f1747012f047d8db69944391f3a63e11ada17edc163697608d3aed6f2758f6c` |
| runtime-lock contract | `bff68eb3db94c4bf5e56e5ba8e2d6f1e961305f5752e6ce748d295f70eb2ba0a` |
| sandbox-profile contract | `1c6be7154c6b8a5727fc1bc22e83b6796855614990a1c778819d9e6c17be5c2a` |
| attestation-transport contract | `f01223582fe6ac29112ead3cf172f6ea04664a84a202b30668a92774a238ba47` |
| P2-verifier closure contract | `d297372d44e60e7c2443c6295f99ef6cfed3afe7e21b4d993ad81e9cdd7396d9` |
| gate-capture bundle contract | `46bb12eba46483b4846db41bcd67f437cc4965e74a94890f132b4aa14a8d9bcf` |
| historical 17-member production closure | `dfcf258b13aa571d3636101f09e66c9d7889a0f09dd88515f4c332ae2d662943` |
| reference runtime | `3343fdcf99eba48c3c21ad2b65008855cd398cdfa04ed66d7a9ee634e9a32a3d` |
| reference source | `b23bf344a10f84a583c38e2326efd5ca70d4c4462fbf76715cd32b9afd20922f` |

The only allowed next action is fresh independent review of the new timestamped
and fixed amendment mirrors. This draft is same-family provisional and
`PROPOSED_PENDING_FRESH_REVIEW`. It grants zero implementation, fixture,
test, P2, S0, G5a, K7, capability, launch, server, data, training, experiment,
Git, paper, or claim authority.
