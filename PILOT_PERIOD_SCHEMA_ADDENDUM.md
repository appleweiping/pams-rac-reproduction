# Pilot Period Schema and K4 Addendum

**Audit date:** 2026-08-15
**Verdict:** `CONDITIONAL_PILOT_ONLY`
**Purpose:** Bind the frozen evaluator-only harmonic gate to the actual MultiRep train/validation period schema and audit the exact K4 feature boundary

## Scope and no-test attestation

This was a read-only schema audit. It inspected only:

- raw MultiRep training annotations under `<MotionBERT-root>/data/train/<sample>/<sample>.json`
- raw MultiRep validation annotations under `<MotionBERT-root>/data/val/<sample>/<sample>.json`
- the v44 train/validation converter outputs and their schema
- the v44 converter, v44 runner, official MultiCounter+ preprocessing code, and official MultiCounter+ evaluator source

No sealed or official test ground truth, test predictions, held-out report, or results directory was opened, hashed, or deserialized. No server file was created or modified, and no training or evaluation job was launched. Connection parameters and credentials are intentionally absent from this report.

## Conclusion

The real raw `period` field is a non-empty list of per-cycle integer source-frame boundary intervals. It is neither a per-frame signal nor a scalar. The observed boundary behavior and the official evaluator geometry require the half-open interpretation `[s,e)` with duration `e-s`. A unique evaluator scalar can therefore be frozen as the float64 NumPy median of the per-cycle spans. No density, count, periodicity, box, resampling lookup, or fallback branch is needed or permitted.

The persisted feature-only whitelist is auditable. The timestamp/no-pose K4 control can be specified from clocks and masks with pose channels zeroed. The nuisance-only K4 control remains open because the current v44 artifacts do not persist legitimate per-example crop, resampler, or augmentation nuisance fields. That control must not be represented as closed until its runtime field list and same-capacity adapter are preregistered.

## Raw annotation evidence

The catalog hash is SHA-256 over canonical JSON containing sorted `[relative_path, file_sha256]` pairs. The content-multiset hash is SHA-256 over canonical JSON containing the sorted file hashes without paths.

| Split | Annotation files | Tracks | Cycle intervals | Catalog SHA-256 | Content-multiset SHA-256 |
|---|---:|---:|---:|---|---|
| train | 794 | 1,878 | 26,010 | `c80f3ce837c28d9e5f2929e41b69c596f8ec1abdc32db7be72b93d8971a65ee6` | `c4c2a119cbb8cd392ff6550115136247d81a37a252771c72b400b70d9ae63d12` |
| val | 183 | 486 | 9,052 | `be42cf759241039c7fd427633135b16c44c18189bec0284608635ee45191f2f2` | `f48d8257e1003ba9044ce9fda0396cf139ed24aeed2f5d79dcc6c058283452b5` |

Every audited `object*` record has exactly:

| Field | Observed type and shape | Privilege |
|---|---|---|
| `bbox` | list of length `L`, each row four finite numeric values | evaluator/audit only |
| `count` | integer scalar | evaluator only |
| `period` | list of two-integer lists, one interval per annotated repetition | evaluator-only period vault |
| `periodicity` | list; empty for every audited train/validation track | evaluator/audit only; unusable for the gate |

Observed integrity:

- `count == len(period)` for all 1,878 training and 486 validation tracks
- all 35,062 interval entries have exactly two integer endpoints
- all intervals have positive duration
- all intervals satisfy `0 <= s < e <= L`
- no missing, empty, malformed, fractional, zero-duration, negative-duration, unsorted, or overlapping period entry was found
- 27 training intervals and 5 validation intervals have `e == source_length`
- training source lengths range from 169 to 2,640 source-frame units
- validation source lengths range from 250 to 1,829 source-frame units

The `e == source_length` observations rule out an inclusive last-frame-index contract. The canonical interval is a source-frame boundary interval `[s,e)`, and its duration is `e-s` without a plus-one correction.

## Source-code evidence

### v44 converter

- repository commit: `e9e8399957daf8c95fbfef21b2db91c7b4a8b279`
- file: `tools/convert_multirep_alphapose_to_counting_v44.py`
- SHA-256: `5a9b148f398e43ac11eac7b54468a0cb22ba2ea63622a962feaa9fdb3cd05e60`
- lines 131--156: converts raw periods to density and includes a count-based fallback when no valid period is available
- lines 159--163: constructs `sampled_frame_indices` with rounded `linspace(0,L-1,T)`
- lines 251--260 and 355--360: reads raw annotations, count, and period
- lines 341--344: samples motion and frame masks at the stored indices
- lines 416--447: emits the current label-mixed v44 sample fields

The density conversion is not an admissible source for `P_eval,p` because it changes clocks, smooths the intervals, rescales mass, and contains a count-based fallback.

### Official MultiCounter+ preprocessing and evaluator

- repository commit: `d15c767d82dac2f7a5df9fde1d7f6b5a4f491062`
- `data_pre/video2frames.py` SHA-256: `ebf4dc08d18070e65d891b6faa5929de7e8f78c000cb4cd98d222ebf19d581aa`
- lines 21--30: defines `period_length = end_frame - start_frame` and fills `range(start_frame,end_frame)`
- `mmdet/datasets/multirep_eval_api.py` SHA-256: `c10a3a56704d8bdf1ba55a2681570dc77371089ecb73c71c56102d26eca1386c`
- lines 534--564: constructs Period AP inputs
- lines 566--648: performs confidence-ordered temporal matching and AP
- lines 650--677: computes temporal IoU from continuous `end-start` spans

The derived binary loop in `video2frames.py` lines 105--110 uses an inclusive end comparison, which conflicts with the span assignment and evaluator geometry. The harmonic gate must consume the raw `period` intervals, never that derived binary signal.

### Local v44 runner

- file: `run_mpcounter_v44.py`
- SHA-256: `ec6b8ee58dd163b86be49f3e7a3461b449c65000184ad498b367e085e71261c6`
- lines 929--946: normalizes positive raw period spans
- lines 949--956: computes endpoint-based interval IoU
- lines 1002--1076: constructs proposals from predicted density, pose masks, sampled indices, and predicted count
- lines 1079--1156: computes the local density-proposal Period metrics

This local scorer is useful for diagnostics but is not the source of `P_eval,p` and is not a drop-in official-faithful replacement for the official evaluator.

## Frozen evaluator definition

For an evaluator-vault object `p` with source length `L`, let

\[
\mathcal C_p=\{[s_{p,k},e_{p,k})\}_{k=1}^{K_p}
\]

The evaluator must assert that `K_p > 0`, every entry contains exactly two Python integers, and `0 <= s_{p,k} < e_{p,k} <= L`. Any violation fails the gate globally. It must not remove a track or consult another label to repair the entry.

Define the per-cycle spans

\[
D_{p,k}=\operatorname{float64}(e_{p,k}-s_{p,k})
\]

and freeze the only track aggregation as

\[
P_{\mathrm{eval},p}=\operatorname{median}_{k=1}^{K_p}D_{p,k}
\]

The implementation must use `numpy.median(np.asarray(spans,dtype=np.float64))`. For an even number of entries, this is the arithmetic mean of the two central sorted spans. There is no harmonic mean, weighted median, trimming, action-conditioned rule, or development-selected alternative.

The `count == len(period)` relation may be verified as a vault-integrity receipt, but `count` is not an input to the computation. `density_gt`, `periodicity`, `bbox`, pose coverage, and all model predictions are likewise excluded.

After all predictions and selector traces are frozen, the selector trace supplies

\[
\widehat P_p=\operatorname{median}(\text{valid pre-vault window selections for }p)
\]

in the same source-frame clock. For `h` in `{0.5,1,2}`, retain the existing no-correction diagnostic

\[
d_{p,h}=\left|\frac{\widehat P_p}{hP_{\mathrm{eval},p}}-1\right|
\]

The unique class within relative tolerance 0.10 is assigned; otherwise the track is `off-grid`. A non-finite value is `off-grid`. This evaluator may reject a claim but may not change a period, count, prediction, track set, checkpoint, or configuration.

### Proposal-ready definition

> Each raw period entry is an integer source-frame boundary interval [s,e) with 0 <= s < e <= L. For person p, the evaluator asserts a non-empty well-formed list C_p and defines P_eval,p as the median of e-s over C_p in float64 source-frame units, using NumPy's default median convention. Any malformed or non-positive entry fails the gate globally; no track is filtered and no count, density, periodicity, box, or fallback value is consulted. The selector is required to emit periods directly in the sampled_frame_indices source clock, so no label resampling or inverse index lookup is permitted.

## Clock contract

The converter stores a model-sample-to-source-frame-center map

\[
q_j=\operatorname{clip}\left(\operatorname{round}\left(\frac{j(L-1)}{T-1}\right),0,L-1\right)
\]

For the current `T=320` artifacts:

| Split | Videos | Tracks | `q` schema | Nondecreasing | Correct endpoints | Videos with duplicate `q` |
|---|---:|---:|---|---:|---:|---:|
| train | 110 | 268 | `int64[320]` | 110 | 110 | 45 |
| val | 51 | 134 | `int64[320]` | 51 | 51 | 0 |

Raw labels use boundary coordinates up to `L`, while sampled frame centers end at `L-1`. Therefore a GT boundary must not be inverted through the sampled-index array. The selector must operate after the proposal's duplicate-clock collapse and valid-cell interpolation and must emit `widehat P` directly in source-frame units. If an implementation emits only uniform 320-grid periods, the contract fails closed; the evaluator must not choose between `T/L`, `(T-1)/(L-1)`, nearest-index, or any data-dependent conversion.

## K4 feature-only audit

The current train/validation pickle hashes are:

- train: `c96fc1dfa233ec4bf1af19e7480ee8c721f4cddee9f121185f4e494c7244e1eb`
- val: `4064ef24dc2970e9b07de8adddf1ecd4932aecb90453a67b4f90d7bd35d10251`

The trusted packer may emit only:

| Field | Type/shape | Model use |
|---|---|---|
| `motion` | `float32[P,320,17,3]` | pose `x,y,confidence` |
| `person_mask` | `bool[P]` | supplied-slot validity |
| `frame_mask` | `bool[P,320]` | pose-frame validity |
| `sampled_frame_indices` | `int64[320]` | source clock |
| `source_length` | Python integer | clock domain |
| `opaque_sample_key` | opaque digest token | dataloader/evaluator join only; never embedded |
| `local_person_slot` | local integer | identity-state routing only; never embedded |

A joint-valid mask may be deterministically derived from the frozen confidence rule before pose zeroing. It is a mask, not a new privileged label.

### Timestamp/no-pose control

The timestamp/no-pose K4 model must:

1. derive and freeze the same person, frame, and joint masks used by the treatment
2. replace every pose `x,y,confidence` channel with zero
3. retain only the masks, `sampled_frame_indices`, `source_length`, and the exact online warp metadata already visible to the treatment
4. use the same training schedule, seed count, state/reset policy, and matched trainable capacity wherever representable

Opaque keys and local slots may route records and state but may not be learned features.

### Nuisance-only control status

**Status: `OPEN_NUISANCE_SCHEMA`**

The current v44 artifacts contain no legitimate per-example crop, resampler, or augmentation nuisance record. Paths, identities, mappings, GT association data, and source hashes are not nuisance features. Before the six budgeted K4 jobs can be claimed executable, the online augmentation layer must freeze:

- the exact nuisance field names, types, shapes, ranges, and normalization
- which fields are visible to both treatment and control
- a deterministic fixed adapter into the same-capacity model
- a receipt proving that no source identity, label, evaluator output, or development metric is encoded

Until then, label isolation is closed but the nuisance-only K4 runtime schema and parameter-matching interface remain open.

## Forbidden fields

The model process must reject before payload deserialization:

- `count_gt`, `density_gt`, raw `count`, `period`, `periodicity`, cycle or boundary fields
- raw annotation `bbox` and every raw or derived GT box field
- `source_json`, `source_json_sha256`, `source_alphapose_json`, `source_alphapose_json_sha256`
- `video_name`, source IDs, canonical IDs, source split, filenames, paths, hashes, and provenance strings
- `object_ids`, `person_object_ids`, `object_mapping`, association structures, or detector identity records
- `gt_object_pose_coverage` and `gt_bbox_assisted_pose_association`
- evaluator periods, count metrics, component labels, development errors, or harmonic classifications

The source/component manifest, checksums, converter version, and association receipts are audit-only. They must not enter a model tensor, selector, objective, checkpoint, or tuning decision.

## Desensitized train/validation-derived fixture

```json
{
  "annotation": {
    "length": "L: int > 1",
    "width": "int",
    "height": "int",
    "object<redacted>": {
      "bbox": "finite numeric list[L][4]",
      "count": "K: int",
      "period": [
        ["s_0: int", "e_0: int"],
        ["s_1: int", "e_1: int"]
      ],
      "periodicity": []
    }
  },
  "feature_shard": {
    "motion": "float32[P,320,17,3]",
    "person_mask": "bool[P]",
    "frame_mask": "bool[P,320]",
    "sampled_frame_indices": "int64[320]",
    "source_length": "int",
    "opaque_sample_key": "non-semantic digest",
    "local_person_slot": "int"
  },
  "evaluator_vault": {
    "interval_semantics": "[s,e) in source-frame boundary units",
    "integrity": "non-empty; integer endpoints; 0 <= s < e <= L",
    "P_eval": "float64 median of all e-s spans for the joined track"
  }
}
```

This fixture reports only validated types and relationships. It contains no video name, person identity, annotation value, prediction, or result.

## Remaining gate

The evaluator period interface is now schema-bound and can be implemented without a branch. The pilot still cannot advance to training until:

1. the feature-only packer and separately permissioned evaluator vault are implemented and checksum-frozen
2. the source-disjoint eligible train/development manifest passes all count-blind assertions
3. the K4 nuisance runtime schema and same-capacity adapter are frozen
4. the official evaluator wrapper and normalized count fixture pass before any decision-bearing run

No result, submission, novelty, or performance claim is authorized by this addendum.
