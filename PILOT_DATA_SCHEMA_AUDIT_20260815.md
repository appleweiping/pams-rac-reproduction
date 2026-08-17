# MultiRep Pilot Data Schema Audit

**Date:** 2026-08-15  
**Mode:** read-only server audit  
**Verdict:** `CONDITIONAL_PILOT_ONLY`

## Scope and attestation

This audit inspected only training/development schemas, provenance metadata, pose-cache structure, and evaluator source code. It did not write any server file, launch training, or open any sealed/official test ground truth, test prediction, or held-out report. Connection parameters and credentials are intentionally absent from this report.

## Conclusion

A checksum-frozen, canonical-source-disjoint train/dev pilot can be built from the v44 expanded-pose artifacts below. The only accurate protocol name is **GT-bbox-assisted AlphaPose supplied-track pilot**. It cannot support predicted-track, end-to-end, HOTA, IDF1, IDSW, or direct same-protocol ranking against published MultiCounter/MultiCounter+ end-to-end results.

Canonical pilot inputs:

- `data/counting_multirep_skeleton_pose_expanded_v44_len320/train.pkl`
- `data/counting_multirep_skeleton_pose_expanded_v44_len320/val.pkl`

Older `pose72` and `highcount` pickles have weaker provenance and should not be canonical inputs. All `real_disjoint_v4x` and `official_complete_v46` artifacts are excluded.

## Frozen source evidence

The MotionBERT tree is at Git HEAD `e9e8399957daf8c95fbfef21b2db91c7b4a8b279`, but the relevant converter and data are not tracked by that commit. The pilot must therefore bind the exact file hashes below.

| Artifact | SHA-256 |
|---|---|
| `tools/convert_multirep_alphapose_to_counting_v44.py` | `5a9b148f398e43ac11eac7b54468a0cb22ba2ea63622a962feaa9fdb3cd05e60` |
| `data/counting_multirep_skeleton_pose_expanded_v44_len320/conversion_manifest_v44.json` | `f9c8dd91d70a8b91f15b47959d05f9ea234064bd3b99c81cfeeb3d74a1d61481` |
| `data/counting_multirep_skeleton_pose_expanded_v44_len320/train.pkl` | `c96fc1dfa233ec4bf1af19e7480ee8c721f4cddee9f121185f4e494c7244e1eb` |
| `data/counting_multirep_skeleton_pose_expanded_v44_len320/val.pkl` | `4064ef24dc2970e9b07de8adddf1ecd4932aecb90453a67b4f90d7bd35d10251` |
| `run_mpcounter_v44.py` | `ec6b8ee58dd163b86be49f3e7a3461b449c65000184ad498b367e085e71261c6` |

The conversion manifest binds the listed converter and train/val pickle hashes, and all three bindings match the files currently present on the server.

## Dataset statistics and source isolation

### Full raw train/dev directories

| Split | Video directories | Canonical sources | Source-connected components | Largest component |
|---|---:|---:|---:|---:|
| train | 794 | 174 | 23 | 186 videos |
| val | 183 | 85 | 15 | 37 videos |

The full raw train/val canonical-source intersection is zero.

### Currently converted pose subset

| Split | Videos | Valid person tracks | Canonical sources | Components | Largest component |
|---|---:|---:|---:|---:|---:|
| train | 110 | 268 | 111 | 18 | 20 videos |
| val | 51 | 134 | 47 | 9 | 14 videos |

The converted train/val canonical-source intersection is also zero. `canonical_source_ids` are derived only after the annotation `video_name`, annotation JSON stem, sample directory, and converter sample name agree.

Pose coverage is incomplete:

- train converts `110 / 794` videos and skips 684 missing AlphaPose inputs
- val converts `51 / 183` videos and skips 132 missing AlphaPose inputs

This creates a plausible pose-success selection bias. The subset is sufficient for a controlled pilot but is not eligible for the paper's main results.

Source reuse within each split is substantial. Resampling, cross-validation, and bootstrap must use source-connected components rather than individual videos.

## Sample schema

Each rich v44 sample contains:

- `motion`: variable-person array `[N, 320, 17, 3]`, `float32`
- `person_mask`: `[N]`, `bool`
- `frame_mask`: `[N, 320]`, `bool`
- `sampled_frame_indices`: `[320]`, `int64`
- `source_length`
- `video_name`, `canonical_video_id`, `canonical_source_ids`, and `source_split`
- `count_gt`: `[N]`, `float32`
- `density_gt`: `[N, 320]`, `float32`
- annotation/AlphaPose hashes and person/object mapping provenance

The pose channels are normalized image `x`, normalized image `y`, and the AlphaPose keypoint confidence. The third channel is not 3D depth. There is no per-frame timestamp or FPS field; temporal reconstruction relies on `sampled_frame_indices` and `source_length`.

There is no autonomous per-frame track-ID sequence. Person slots are formed by greedy per-frame IoU association between GT boxes and AlphaPose detections with `min_iou=0.1`; the manifest explicitly classifies the data as `skeleton-track-conditioned; not end-to-end detection`. Thirteen train slots and eight val slots are marked ambiguous, while the converted subset has no completely missing person slot.

Forty-five of 110 training videos are shorter than 320 frames. Uniform resampling repeats source frames in these videos, with as many as 151 adjacent duplicate sample steps. Local-period estimation must use the stored source-frame indices rather than assume a strictly increasing uniform clock.

All 402 converted person tracks have positive integer counts. `density_gt.sum()` equals `count_gt` within a maximum absolute floating-point difference of `7.62939453125e-06`; density therefore leaks the complete count target.

## Physical label isolation contract

### Model-process fields

- `motion`
- `person_mask`
- `frame_mask`
- `sampled_frame_indices`
- `source_length`
- opaque sample keys derived by hashing canonical video identity
- local person-slot indices

### Audit/split-only fields

- `canonical_source_ids`
- `canonical_video_id`
- `video_name`
- `source_split`
- source annotation and AlphaPose SHA-256 values
- converter schema and version

These may drive component grouping and provenance checks but must never enter model features.

### Evaluator-only fields

- `count_gt`
- `density_gt`
- `source_json`
- raw annotation `count`, `period`, and `bbox`
- `object_ids` and `person_object_ids`
- `object_mapping`
- `gt_object_pose_coverage`
- `gt_bbox_assisted_pose_association`
- `source_alphapose_json`

The current pickle colocates inputs and privileged targets. A Dataset class that merely ignores target keys is not a sufficient basis for the claim that training uses no per-person count or period labels. Before training, a packer must create a feature-only artifact and a separately permissioned evaluator vault, and the training entry point must reject privileged keys before deserialization.

## Metric implementation verdict

The official MultiCounter+ source is at commit `d15c767d82dac2f7a5df9fde1d7f6b5a4f491062`. Its evaluator `mmdet/datasets/multirep_eval_api.py` has SHA-256 `c10a3a56704d8bdf1ba55a2681570dc77371089ecb73c71c56102d26eca1386c`.

| Metric | Available implementation | Verdict |
|---|---|---|
| Period-mAP/AP50/AP75 | Official MultiCounter+ evaluator | Official-faithful only when supplied with instance detections/tracks and the official instance-matching protocol |
| Supplied-track Period AP | Official temporal-AP kernel can be reused | Must be named supplied/oracle-track diagnostic; not directly comparable with published end-to-end values |
| AvgMAE/AvgOBO | Small scorer should be implemented directly from the published equations | Can be official-faithful |
| MotionBERT count/period scorer | `run_mpcounter_v44.py` | Not fully official-faithful |
| PAMS metrics | `src/pams/metrics.py` | Single-person UCFRep only; not a MultiRep evaluator |

The published AvgMAE and AvgOBO first average people within each video and then average videos. MotionBERT's current scorer pools all people, which gives videos with three people more weight than videos with two. Its Period AP also assumes a known GT object mapping and bypasses the official instance-localization cost. MotionBERT returns Period AP on `[0,1]`, whereas the published MultiCounter tables display percentage units such as `14.56`; result generation must freeze one explicit scale.

The current pose pickle has neither predicted temporal bboxes nor autonomous ID trajectories, so HOTA, IDF1, and IDSW are unavailable.

## PAMS-RAC data check

No MultiRep train-only pose cache or MultiRep evaluator exists in the audited PAMS-RAC tree. The available cache is a single-person UCFRep resource:

- PAMS Git HEAD `1fd8cc7b90bd23118d8dcf72792e9dfeb94293bb`
- 337 train plus 84 dev cache entries, about 35 MiB
- each pose cache contains `xyz [256,33,3]`, `valid_mask [256]`, and schema-v2 metadata
- train/dev ID and source-video hash intersections are zero
- 16 train and 6 dev sequences have zero valid pose frames

The label-free PAMS input manifests and their hashes are:

| Artifact | SHA-256 |
|---|---|
| `assets/pams_v14_seed2026/protocol/train.inputs.json` | `e238d307b4ed37b2f6c19eefdf7fa8812ec3d231cd2a7091be38e1c279f04207` |
| `assets/pams_v14_seed2026/protocol/dev.inputs.json` | `f21c49569805e4aea326977f6d7ce2964f503fad129a17e95214594364ca4ae7` |
| training pose-cache snapshot | `b537dede4fa1b9ecd6559b9c795440a5b55351c444e58648349ebbd9ebee574c` |
| dev pose-cache snapshot | `14c290d86097b572cd544853f9d9c8bde379a3e53d7b637b22b66c6fbe1bb417` |

This resource may support separate single-person self-supervised pretraining or unit tests. It cannot substitute for the MultiRep pilot.

## Minimal extraction and expected size

The existing converter can be restricted to train/val:

```bash
python tools/convert_multirep_alphapose_to_counting_v44.py \
  --data_root data \
  --out_dir data/multirep_pose_pilot_source_v1 \
  --splits train val \
  --target_len 320 \
  --max_persons 8 \
  --coord_norm image
```

Adding `--strict` currently fails because many AlphaPose inputs are absent. Once train/val pose extraction is complete, a paper-eligible rebuild must use `--strict` and still exclude test.

The two current source pickles total exactly `27,487,004` bytes (`26.214 MiB`). Their feature/mask/index array payload is `25.543 MiB`; count and density arrays total `0.492 MiB`.

The converter still produces label-mixed pickles. The next implementation gate is a dedicated packer that produces:

1. feature-only train/dev shards
2. an evaluator-only label vault
3. a canonical-source connected-component manifest
4. per-file and per-sample SHA-256 receipts
5. a forbidden-key scan receipt

## Forbidden paths and artifacts

The pilot must not read or consume:

- `data/test/**`
- `data/counting_multirep_skeleton_pose_expanded_v44_len320/test.pkl`
- `data/counting_multirep_skeleton_real_disjoint_v44_len320/**`
- `data/counting_multirep_skeleton_real_disjoint_v46_len320/**`
- especially `data/counting_multirep_skeleton_real_disjoint_v46_len320/val.pkl`, whose historical construction includes original-test sources
- `data/counting_multirep_skeleton_official_complete_v46_len320/**`
- legacy prediction/report artifacts under `results/**` and `output/**`
- `tools/convert_complete_multirep_test_v46.py`
- `tools/select_complete_multirep_test_v46.py`
- `tools/convert_selected_multirep_to_test_v45.py`
- `tools/make_complete_test_queue_v46.py`
- `tools/precommit_official_complete_v46.py`

## Next gate

The pilot may proceed only after the feature/label physical split, forbidden-key tests, source-component manifest, and checksum ledger pass. Every experiment must remain marked `partial-cache`, `GT-bbox-assisted`, and `supplied-track`. Main-paper evidence remains blocked until the complete train/val pose cache, official MultiRep release/split identity, independent seeds, frozen predictions, official metric scorer, and untouched final evaluator are available.
