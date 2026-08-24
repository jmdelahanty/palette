# Palette Detection-Family Zarr Schema Inventory

Status: generated review inventory; not an accepted storage contract

This report keeps declarations, current runtime evidence, and dated physical observations separate. A disagreement is a review item, not something the generator resolves by guessing.

## Result

- `4` declared variants with `35` array bindings;
- `8` current runtime variants with `103` array bindings;
- `13` unresolved declaration/runtime conflicts;
- `6` detection-lineage leaf names propagated outside detection-owned groups;
- `3` dated Sleepyfish observations with `45` physical array bindings.

The current `DETECT_SPEC` is therefore orientation evidence, not an authoritative schema. In particular, raw bounding boxes are `float64` in the current writer and the observed archive, while the declaration says `float32`.
The completed Sleepyfish refined-detection snapshot contains `36` sharded arrays, including publication lineage that is absent from the `25` subgroup bindings in the current StageSpec.
`detection_artifact_runs` is classified as immutable, selector-ineligible quarantined evidence with run-local identity. It remains in the inventory for compatibility and diagnostics, but it is deferred from the future-facing storage implementation and benchmark waves.

## Accepted Future-Facing Decisions

| Decision | Canonical target | Current runtime | Current disposition | Revisit |
| --- | --- | --- | --- | --- |
| canonical_detection_continuous_geometry_dtype.v1 | float32 | float64 | explicit_legacy_transition | canonical_storage_specs_complete |

Canonical detection bounding boxes and centers use exact `float32` in the first storage contract. Current `float64` writers and archives remain explicit transition evidence until migrated; they do not change the accepted target. `float16` and quantized `uint16` representations are deferred and require a new version plus numerical and behavioral validation.

## Current Runtime Variants

| Variant | Path | Arrays | Lifecycle | Compared declaration |
| --- | --- | --- | --- | --- |
| current.detect_yolo_canonical | detect_runs/<run>/ | 10 | build_then_immutable | declared.detect |
| current.detection_artifact_unbound | detection_artifact_runs/<run>/ | 7 | build_then_immutable_nonselector | declared.detect |
| current.detect_quality_nested | detect_runs/<run>/quality_reports/<qrun>/ | 2 | build_then_immutable | declared.detect_quality_nested |
| current.detect_quality_collection_snapshot | detect_quality_runs/<run>/ | 3 | immutable_sharded_snapshot | None |
| current.refined_detect_dense_authoring_root | refined_detect_runs/<run>/ | 19 | editable_random_update_and_projection_sync | None |
| current.refined_detect_instances_projection | refined_detect_runs/<run>/instances/ | 15 | derived_projection_or_immutable_snapshot | declared.refined_detect.instances |
| current.refined_detect_source_projection | refined_detect_runs/<run>/source_detections/ | 11 | derived_projection_or_immutable_snapshot | declared.refined_detect.source_detections |
| current.refined_detect_clipped_collection_snapshot | refined_detect_runs/<run>/{instances,source_detections}/ | 36 | immutable_sharded_snapshot | None |

## Unresolved Conflicts

| Severity | Variant | Array | Field | Declared | Current |
| --- | --- | --- | --- | --- | --- |
| high | current.detect_quality_collection_snapshot | — | variant | None | detect_quality_runs/<run>/ |
| high | current.detect_yolo_canonical | bbox_img_xyxy | array_presence | None | True |
| high | current.detect_yolo_canonical | bbox_norm_coords | dtype | float32 | float64 |
| high | current.detect_yolo_canonical | centers_img_xy | array_presence | None | True |
| medium | current.detect_yolo_canonical | instance_key | required | False | True |
| medium | current.detect_yolo_canonical | n_detections | required | False | True |
| high | current.detect_yolo_canonical | source_acquisition_frame_index | array_presence | None | True |
| high | current.detection_artifact_unbound | artifact_row_id | array_presence | None | True |
| high | current.detection_artifact_unbound | bbox_norm_coords | dtype | float32 | float64 |
| medium | current.detection_artifact_unbound | n_detections | required | False | True |
| high | current.refined_detect_clipped_collection_snapshot | — | variant | None | refined_detect_runs/<run>/{instances,source_detections}/ |
| high | current.refined_detect_dense_authoring_root | — | variant | None | refined_detect_runs/<run>/ |
| high | current.refined_detect_instances_projection | instance_key_origin_codes | array_presence | None | True |

## `current.detect_yolo_canonical`

| Array | dtype | Shape | Required | Expected access |
| --- | --- | --- | --- | --- |
| frame_indices | int32 | ["n_detections"] | True | windowed_row_axis |
| bbox_norm_coords | float64 | ["n_detections",4] | True | per_detection_row |
| scores | float32 | ["n_detections"] | True | windowed_row_axis |
| class_ids | int32 | ["n_detections"] | True | windowed_row_axis |
| instance_key | uint64 | ["n_detections"] | True | windowed_row_axis |
| source_acquisition_frame_index | int64 | ["n_detections"] | True | windowed_row_axis |
| bbox_img_xyxy | float64 | ["n_detections",4] | True | per_detection_row |
| centers_img_xy | float64 | ["n_detections",2] | True | per_detection_row |
| frame_counts | int32 | ["n_frames"] | True | windowed_frame_axis |
| n_detections | int32 | ["n_frames"] | True | windowed_frame_axis |

## `current.detection_artifact_unbound`

| Array | dtype | Shape | Required | Expected access |
| --- | --- | --- | --- | --- |
| frame_indices | int32 | ["n_detections"] | True | windowed_row_axis |
| bbox_norm_coords | float64 | ["n_detections",4] | True | per_detection_row |
| scores | float32 | ["n_detections"] | True | windowed_row_axis |
| class_ids | int32 | ["n_detections"] | True | windowed_row_axis |
| artifact_row_id | uint64 | ["n_detections"] | True | windowed_row_axis |
| frame_counts | int32 | ["n_frames"] | True | windowed_frame_axis |
| n_detections | int32 | ["n_frames"] | True | windowed_frame_axis |

## `current.detect_quality_nested`

| Array | dtype | Shape | Required | Expected access |
| --- | --- | --- | --- | --- |
| detection_quality_labels | int8 | ["n_detections"] | True | windowed_row_axis |
| quality_flags | int8 | ["n_frames"] | True | windowed_frame_axis |

## `current.detect_quality_collection_snapshot`

| Array | dtype | Shape | Required | Expected access |
| --- | --- | --- | --- | --- |
| quality_flags | int8 | ["n_frames"] | True | windowed_frame_axis |
| detection_quality_labels | int8 | ["n_detections"] | True | windowed_row_axis |
| instance_key | uint64 | ["n_detections"] | True | windowed_row_axis |

## `current.refined_detect_dense_authoring_root`

| Array | dtype | Shape | Required | Expected access |
| --- | --- | --- | --- | --- |
| refined_row_ids | int64 | ["n_curated"] | True | windowed_row_axis |
| frame_indices | int32 | ["n_curated"] | True | windowed_row_axis |
| entity_ids | int32 | ["n_curated"] | True | windowed_row_axis |
| bbox_img_xyxy | float64 | ["n_curated",4] | True | per_detection_row |
| bbox_norm_coords | float64 | ["n_curated",4] | True | per_detection_row |
| status_codes | int8 | ["n_curated"] | True | windowed_row_axis |
| source_kind_codes | int8 | ["n_curated"] | True | windowed_row_axis |
| manual_edit_flags | bool | ["n_curated"] | True | windowed_row_axis |
| source_detect_row_index | int32 | ["n_curated"] | True | windowed_row_axis |
| review_state_codes | int8 | ["n_curated"] | True | windowed_row_axis |
| keypoints_state_codes | int8 | ["n_curated"] | True | windowed_row_axis |
| subject_mask_state_codes | int8 | ["n_curated"] | True | windowed_row_axis |
| eye_mask_state_codes | int8 | ["n_curated"] | True | windowed_row_axis |
| swim_bladder_state_codes | int8 | ["n_curated"] | True | windowed_row_axis |
| confidence_scores | float32 | ["n_curated"] | True | windowed_row_axis |
| class_ids | int32 | ["n_curated"] | True | windowed_row_axis |
| detection_source | int8 | ["n_curated"] | False | windowed_row_axis |
| reason_bytes | uint8 | ["n_curated","reason_width"] | True | per_detection_row |
| review_notes | utf8 | ["n_curated"] | False | windowed_row_axis |

## `current.refined_detect_instances_projection`

| Array | dtype | Shape | Required | Expected access |
| --- | --- | --- | --- | --- |
| bbox_img_xyxy | float64 | ["n_instances",4] | True | per_detection_row |
| bbox_norm_coords | float64 | ["n_instances",4] | True | per_detection_row |
| class_ids | int32 | ["n_instances"] | False | windowed_row_axis |
| confidence_scores | float32 | ["n_instances"] | False | windowed_row_axis |
| frame_counts | int32 | ["n_frames"] | True | windowed_frame_axis |
| frame_indices | int32 | ["n_instances"] | True | windowed_row_axis |
| frame_offsets | int64 | ["n_frame_offsets"] | True | windowed_frame_axis |
| instance_key | uint64 | ["n_instances"] | False | windowed_row_axis |
| instance_key_origin_codes | int8 | ["n_instances"] | False | windowed_row_axis |
| manual_edit_flags | bool | ["n_instances"] | True | windowed_row_axis |
| reason_bytes | uint8 | ["n_instances","width"] | True | per_detection_row |
| refined_row_ids | int64 | ["n_instances"] | True | windowed_row_axis |
| review_notes | string | ["n_instances"] | False | windowed_row_axis |
| source_detect_row_index | int32 | ["n_instances"] | True | windowed_row_axis |
| source_kind_codes | int8 | ["n_instances"] | True | windowed_row_axis |

## `current.refined_detect_source_projection`

| Array | dtype | Shape | Required | Expected access |
| --- | --- | --- | --- | --- |
| bbox_img_xyxy | float64 | ["n_source_detections",4] | True | per_detection_row |
| bbox_norm_coords | float64 | ["n_source_detections",4] | True | per_detection_row |
| class_ids | int32 | ["n_source_detections"] | False | windowed_row_axis |
| confidence_scores | float32 | ["n_source_detections"] | False | windowed_row_axis |
| decision_codes | int8 | ["n_source_detections"] | True | windowed_row_axis |
| frame_indices | int32 | ["n_source_detections"] | True | windowed_row_axis |
| instance_key | uint64 | ["n_source_detections"] | False | windowed_row_axis |
| reason_bytes | uint8 | ["n_source_detections","width"] | True | per_detection_row |
| resolved_refined_row_id | int64 | ["n_source_detections"] | True | windowed_row_axis |
| review_notes | string | ["n_source_detections"] | False | windowed_row_axis |
| source_detect_row_index | int32 | ["n_source_detections"] | True | windowed_row_axis |

## `current.refined_detect_clipped_collection_snapshot`

| Array | dtype | Shape | Required | Expected access |
| --- | --- | --- | --- | --- |
| instances/bbox_img_xyxy | float64 | ["n_instances",4] | True | per_detection_row |
| instances/bbox_norm_coords | float64 | ["n_instances",4] | True | per_detection_row |
| instances/class_ids | int32 | ["n_instances"] | True | windowed_row_axis |
| instances/confidence_scores | float32 | ["n_instances"] | True | windowed_row_axis |
| instances/frame_counts | int32 | ["n_frames"] | True | windowed_frame_axis |
| instances/frame_indices | int64 | ["n_instances"] | True | windowed_row_axis |
| instances/frame_offsets | int64 | ["n_frame_offsets"] | True | windowed_frame_axis |
| instances/instance_key | uint64 | ["n_instances"] | True | windowed_row_axis |
| instances/instance_key_origin_codes | int8 | ["n_instances"] | True | windowed_row_axis |
| instances/manual_edit_flags | bool | ["n_instances"] | True | windowed_row_axis |
| instances/reason_bytes | uint8 | ["n_instances",64] | True | per_detection_row |
| instances/refined_row_ids | int64 | ["n_instances"] | True | windowed_row_axis |
| instances/source_clip_detect_row_index | int64 | ["n_instances"] | True | windowed_row_axis |
| instances/source_clip_indices | int64 | ["n_instances"] | True | windowed_row_axis |
| instances/source_clip_local_frame_indices | int64 | ["n_instances"] | True | windowed_row_axis |
| instances/source_detect_row_index | int32 | ["n_instances"] | True | windowed_row_axis |
| instances/source_frame_indices | int64 | ["n_instances"] | True | windowed_row_axis |
| instances/source_kind_codes | int8 | ["n_instances"] | True | windowed_row_axis |
| instances/source_recording_frame_ids | int64 | ["n_instances"] | True | windowed_row_axis |
| instances/source_refined_row_ids | int64 | ["n_instances"] | True | windowed_row_axis |
| source_detections/bbox_img_xyxy | float64 | ["n_source_detections",4] | True | per_detection_row |
| source_detections/bbox_norm_coords | float64 | ["n_source_detections",4] | True | per_detection_row |
| source_detections/class_ids | int32 | ["n_source_detections"] | True | windowed_row_axis |
| source_detections/confidence_scores | float32 | ["n_source_detections"] | True | windowed_row_axis |
| source_detections/decision_codes | int8 | ["n_source_detections"] | True | windowed_row_axis |
| source_detections/frame_indices | int64 | ["n_source_detections"] | True | windowed_row_axis |
| source_detections/instance_key | uint64 | ["n_source_detections"] | True | windowed_row_axis |
| source_detections/reason_bytes | uint8 | ["n_source_detections",64] | True | per_detection_row |
| source_detections/resolved_refined_row_id | int64 | ["n_source_detections"] | True | windowed_row_axis |
| source_detections/source_clip_detect_row_index | int64 | ["n_source_detections"] | True | windowed_row_axis |
| source_detections/source_clip_indices | int64 | ["n_source_detections"] | True | windowed_row_axis |
| source_detections/source_clip_local_frame_indices | int64 | ["n_source_detections"] | True | windowed_row_axis |
| source_detections/source_detect_row_index | int32 | ["n_source_detections"] | True | windowed_row_axis |
| source_detections/source_frame_indices | int64 | ["n_source_detections"] | True | windowed_row_axis |
| source_detections/source_recording_frame_ids | int64 | ["n_source_detections"] | True | windowed_row_axis |
| source_detections/source_resolved_refined_row_id | int64 | ["n_source_detections"] | True | windowed_row_axis |

## Dated Physical Observations

| Observation | Run | Completion | Arrays | Sharded arrays |
| --- | --- | --- | --- | --- |
| sleepyfish_cam2010095_latest_raw_detect_20260723 | detect_runs/detect_2026-05-14_15-39-11 | completion metadata absent | 6 | 0 |
| sleepyfish_cam2010095_quality_snapshot_20260723 | detect_quality_runs/detect_quality_sleepyfish_source_collection_v2_20260715_01 | complete | 3 | 3 |
| sleepyfish_cam2010095_refined_snapshot_20260723 | refined_detect_runs/refined_detect_sleepyfish_allclips_sharded_20260715_01 | complete | 36 | 36 |

### `sleepyfish_cam2010095_latest_raw_detect_20260723`

| Array | dtype | Shape | Inner chunk | Outer shard |
| --- | --- | --- | --- | --- |
| bbox_norm_coords | float64 | [1187087,4] | [1024,4] | — |
| class_ids | int32 | [1187087] | [1024] | — |
| frame_counts | int32 | [1188000] | [1024] | — |
| frame_indices | int32 | [1187087] | [1024] | — |
| n_detections | int32 | [1188000] | [1024] | — |
| scores | float32 | [1187087] | [1024] | — |

### `sleepyfish_cam2010095_quality_snapshot_20260723`

| Array | dtype | Shape | Inner chunk | Outer shard |
| --- | --- | --- | --- | --- |
| detection_quality_labels | int8 | [1186376] | [16384] | [131072] |
| instance_key | uint64 | [1186376] | [16384] | [131072] |
| quality_flags | int8 | [1188000] | [16384] | [131072] |

### `sleepyfish_cam2010095_refined_snapshot_20260723`

| Array | dtype | Shape | Inner chunk | Outer shard |
| --- | --- | --- | --- | --- |
| instances/bbox_img_xyxy | float64 | [1169010,4] | [1024,4] | [131072,4] |
| instances/bbox_norm_coords | float64 | [1169010,4] | [1024,4] | [131072,4] |
| instances/class_ids | int32 | [1169010] | [1024] | [131072] |
| instances/confidence_scores | float32 | [1169010] | [1024] | [131072] |
| instances/frame_counts | int32 | [1188000] | [16384] | [131072] |
| instances/frame_indices | int64 | [1169010] | [16384] | [131072] |
| instances/frame_offsets | int64 | [1188001] | [16384] | [131072] |
| instances/instance_key | uint64 | [1169010] | [16384] | [131072] |
| instances/instance_key_origin_codes | int8 | [1169010] | [16384] | [131072] |
| instances/manual_edit_flags | bool | [1169010] | [1024] | [131072] |
| instances/reason_bytes | uint8 | [1169010,64] | [1024,64] | [131072,64] |
| instances/refined_row_ids | int64 | [1169010] | [16384] | [131072] |
| instances/source_clip_detect_row_index | int64 | [1169010] | [16384] | [131072] |
| instances/source_clip_indices | int64 | [1169010] | [16384] | [131072] |
| instances/source_clip_local_frame_indices | int64 | [1169010] | [16384] | [131072] |
| instances/source_detect_row_index | int32 | [1169010] | [16384] | [131072] |
| instances/source_frame_indices | int64 | [1169010] | [16384] | [131072] |
| instances/source_kind_codes | int8 | [1169010] | [1024] | [131072] |
| instances/source_recording_frame_ids | int64 | [1169010] | [16384] | [131072] |
| instances/source_refined_row_ids | int64 | [1169010] | [16384] | [131072] |
| source_detections/bbox_img_xyxy | float64 | [1186376,4] | [1024,4] | [131072,4] |
| source_detections/bbox_norm_coords | float64 | [1186376,4] | [1024,4] | [131072,4] |
| source_detections/class_ids | int32 | [1186376] | [1024] | [131072] |
| source_detections/confidence_scores | float32 | [1186376] | [1024] | [131072] |
| source_detections/decision_codes | int8 | [1186376] | [1024] | [131072] |
| source_detections/frame_indices | int64 | [1186376] | [16384] | [131072] |
| source_detections/instance_key | uint64 | [1186376] | [16384] | [131072] |
| source_detections/reason_bytes | uint8 | [1186376,64] | [1024,64] | [131072,64] |
| source_detections/resolved_refined_row_id | int64 | [1186376] | [16384] | [131072] |
| source_detections/source_clip_detect_row_index | int64 | [1186376] | [16384] | [131072] |
| source_detections/source_clip_indices | int64 | [1186376] | [16384] | [131072] |
| source_detections/source_clip_local_frame_indices | int64 | [1186376] | [16384] | [131072] |
| source_detections/source_detect_row_index | int32 | [1186376] | [16384] | [131072] |
| source_detections/source_frame_indices | int64 | [1186376] | [16384] | [131072] |
| source_detections/source_recording_frame_ids | int64 | [1186376] | [16384] | [131072] |
| source_detections/source_resolved_refined_row_id | int64 | [1186376] | [16384] | [131072] |

## Downstream Detection Lineage

| Leaf | Occurrences | Observed dtypes | Declared stages |
| --- | --- | --- | --- |
| artifact_row_id | 1 | ["uint64"] | [] |
| detection_indices | 17 | ["int32","int64"] | ["crop","eye_masks","keypoints","refined_eye_masks","refined_keypoints","refined_subject_masks","subject_masks"] |
| detection_source | 29 | ["int8"] | ["crop","eye_masks","keypoints","refined_keypoints","refined_subject_masks","subject_masks"] |
| instance_key | 19 | ["uint64"] | ["crop","eye_masks","keypoints","refined_eye_masks","refined_keypoints","refined_subject_masks","subject_masks","tracking"] |
| source_detect_row_index | 14 | ["int32"] | ["crop","eye_masks","keypoints","refined_eye_masks","refined_keypoints","refined_subject_masks","subject_masks","tracking"] |
| source_refined_row_ids | 19 | ["int64"] | ["crop","eye_masks","keypoints","refined_eye_masks","refined_keypoints","refined_subject_masks","subject_masks","subject_shape","tail_kinematics","tail_posture_view","tracking"] |

## Contract Checklist

Execution order and exit gates are maintained in `docs/canonical_detection_storage_implementation_checklist.md`.

- [x] Use exact `float32` for first-generation canonical detection bounding boxes and centers; treat current `float64` as an explicit transition representation.
- [x] Defer `float16` and quantized integer detection geometry until canonical storage specs are complete; require a new schema version and behavioral benchmarks before adoption.
- [x] Classify `detection_artifact_runs` separately from canonical `detect_runs` as immutable, selector-ineligible quarantined evidence with run-local identity.
- [ ] Add a StageSpec for immutable `detect_quality_runs` snapshots; do not conflate it with nested historical reports.
- [ ] Decide whether dense refined root arrays remain the editable authority or become a compatibility projection of `instances`.
- [ ] Add every clipped/publication lineage column to a versioned snapshot schema.
- [ ] Lock exact dtype, axis names, null/fill semantics, and requiredness before assigning storage plans.
- [ ] Benchmark canonical detections, quality snapshots, editable refined detections, and published refined snapshots in the first implementation wave.
- [ ] Revisit unbound-artifact storage and benchmarks only if a supported future consumer or canonical binding path is approved.

The machine-readable JSON retains all affiliated schema and writer evidence, including dynamic writer sites and physical chunk/shard shapes.
