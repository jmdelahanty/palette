# Stage Array Validation Rollout Audit

Date: 2026-05-21
Status: current
Scope: Palette Zarr run-group completion validation, `StageSpec` required-array contracts, and production writer compatibility.

## Summary

The completion hard gate is now the right default: a run cannot be marked `ok` without a `root`, a resolved run group, and a completed Zarr run marker. Array validation should remain shadow-mode by default until each stage is confirmed writer-compatible.

Recommended first hard-enforcement allowlist after one real-run smoke each:

- `detect`
- `detect_quality`
- `refined_detect`
- `crop`
- `refined_eye_masks`
- `subject_masks`
- `tracking`

Keep these in shadow mode for now:

- `background`: spec requires `frame_indices`, but the current writer stores sampled source frame identity in attrs and does not emit registry stage completion.
- `keypoints`: YOLO keypoint runs do not write the traditional triangle diagnostic arrays required by the current spec.
- `refined_keypoints`: the primary writer appears close, but some registry sync call sites still call `emit_stage_completion(... root=None ...)` and now cannot mark `ok`.
- `eye_masks`: the stage contains multiple method-specific surfaces; traditional, YOLO, and U-Net writers do not all satisfy the same required-array set.
- `refined_subject_masks`: canonical finalizer/assembler outputs look compatible, but mutation and legacy paths should be smoked before hard enforcement.
- `arena_assignment`: current writer does not emit the required `confidence` array.
- `raw_video`: no required arrays and no current stage-completion writer target.

## Current Gate Behavior

`src/fisheye/registry/stage_complete.py` currently does the important safety work before registry status is written:

- `status == "ok"` plus `run_name` requires a non-null `root`.
- The run group is resolved from the stage parent mapping.
- The run group must have a complete Zarr run marker.
- Array validation writes telemetry into `details_json` but is shadow-mode unless the stage is in `_ENFORCE_STAGE_ARRAY_VALIDATION_FOR`.
- Unknown stage specs produce `stage_array_validation_status="no_spec"` and a warning, which avoids silent no-op validation.

This means the most dangerous failure mode, a killed writer leaving a half-written run group that is then marked complete in the registry, is blocked independently of per-stage array enforcement.

## Cross-Cutting Blockers

Rootless `ok` status call sites now need remediation before they can write registry completion status:

- `src/fisheye/refinement/refine_keypoints.py:371`
- `src/fisheye/inference/predict_pose.py:240`
- `src/fisheye/utils/run_keypoints_batch.py:655`
- `src/fisheye/utils/run_eye_masks_batch.py:845`

These should pass or reopen the mutable Zarr root before calling `emit_stage_completion` when `status="ok"` and `run_name` is present.

## Stage Verdicts

| Stage | Verdict | Evidence | Action |
| --- | --- | --- | --- |
| `raw_video` | Not an array-enforcement target | `StageSpec` has no required arrays; no current `emit_stage_completion` writer was found. | Leave out of enforcement allowlist. |
| `background` | Blocked | Spec requires `frame_indices`, but `src/fisheye/preprocessing/background.py:194` and `src/fisheye/preprocessing/background.py:212` write background images while `src/fisheye/preprocessing/background.py:235` stores sampled source frames in attrs as `source_frame_indices`; `src/fisheye/preprocessing/background.py:256` marks the run complete without registry stage completion. | Either demote/add the expected provenance array and add a registry writer, or leave shadow-only. |
| `detect` | Candidate | YOLO writes `bbox_norm_coords`, counts, and aliases at `src/fisheye/detection/detect_yolo.py:1533` through `src/fisheye/detection/detect_yolo.py:1537`; traditional writes `frame_counts`/`n_detections` for detected rows at `src/fisheye/detection/detect_traditional.py:394` and `src/fisheye/detection/detect_traditional.py:400`, and writes required empty arrays for no-detection runs at `src/fisheye/detection/detect_traditional.py:409` through `src/fisheye/detection/detect_traditional.py:415`. | Add to hard allowlist after one real YOLO and one traditional smoke. |
| `detect_quality` | Candidate | `src/fisheye/refinement/detect_quality.py:634` and `src/fisheye/refinement/detect_quality.py:635` write `quality_flags` and `detection_quality_labels`; `src/fisheye/refinement/detect_quality.py:678` marks complete; registry emit uses root at `src/fisheye/refinement/detect_quality.py:58`. | Add to hard allowlist after a nested quality-report smoke. Document quality run names as globally unique within a root or pass parent detect run identity. |
| `refined_detect` | Candidate | Current curated sparse writer writes `instances` arrays at `src/fisheye/shared/refined_detect_curation.py:1506` through `src/fisheye/shared/refined_detect_curation.py:1511`, `source_detections` arrays at `src/fisheye/shared/refined_detect_curation.py:1610` through `src/fisheye/shared/refined_detect_curation.py:1622`, and sets sparse semantics at `src/fisheye/shared/refined_detect_curation.py:2303`. Completion is marked at `src/fisheye/refinement/refine_detect.py:1485`. | Add to hard allowlist after a clipped and non-clipped refined-detect smoke. |
| `crop` | Candidate | Main writer creates/copies `frame_indices`, `bbox_norm_coords`, `frame_counts`, and detection identity around `src/fisheye/tracking/crop.py:2627` through `src/fisheye/tracking/crop.py:2650`; run completion is marked at `src/fisheye/tracking/crop.py:3604`, and registry emit uses root at `src/fisheye/tracking/crop.py:382`. | Add to hard allowlist after smoke. Confirm promotion-created crop runs do not emit incomplete registry status. |
| `keypoints` | Blocked | Traditional writer creates triangle diagnostics at `src/fisheye/detection/detect_keypoints_traditional.py:866`, `src/fisheye/detection/detect_keypoints_traditional.py:875`, and `src/fisheye/detection/detect_keypoints_traditional.py:885`; YOLO writer creates core keypoint arrays but has no `triangle_angles`, `triangle_angles_raw`, or `triangle_area` output. `src/fisheye/inference/predict_pose.py:240` also emits with `root=None`. | Split method-specific required fields or make traditional-only diagnostics optional; fix rootless pose status sync. |
| `refined_keypoints` | Blocked by call sites | Writer creates required geometry and review arrays, including `triangle_area` around `src/fisheye/refinement/refine_keypoints.py:1281`, `triangle_angles` around `src/fisheye/refinement/refine_keypoints.py:1297`, and `failure_indices` around `src/fisheye/refinement/refine_keypoints.py:1601`; completion is marked at `src/fisheye/refinement/refine_keypoints.py:1748`. However registry emit paths still use `root=None` at `src/fisheye/refinement/refine_keypoints.py:371` and `src/fisheye/utils/run_keypoints_batch.py:655`. | Fix rootless emit call sites, then smoke both direct refine and batch auto-review paths before enforcement. |
| `eye_masks` | Blocked | Traditional writer matches the contour/reason-style spec at `src/fisheye/segmentation/eye_segmentation.py:714`, `src/fisheye/segmentation/eye_segmentation.py:755` through `src/fisheye/segmentation/eye_segmentation.py:779`; YOLO writer uses `contour_ptr`, `contours_eye0`, and `contours_eye1` at `src/fisheye/segmentation/eye_segmentation_yolo.py:1164` through `src/fisheye/segmentation/eye_segmentation_yolo.py:1182`; U-Net writer focuses on `mask_probs_roi` and `detection_source` at `src/fisheye/segmentation/infer_unet_eye_masks.py:411` and `src/fisheye/segmentation/infer_unet_eye_masks.py:814`. | Split stage specs by method family or demote contour/ellipse/reason arrays to optional with method-specific validators. |
| `refined_eye_masks` | Candidate | Refiner writes contour arrays at `src/fisheye/refinement/refine_eye_masks.py:1742` through `src/fisheye/refinement/refine_eye_masks.py:1760`, source lineage at `src/fisheye/refinement/refine_eye_masks.py:2093`, root arrays starting around `src/fisheye/refinement/refine_eye_masks.py:2178`, reason columns at `src/fisheye/refinement/refine_eye_masks.py:2542`, and marks complete at `src/fisheye/refinement/refine_eye_masks.py:2782`. | Add to hard allowlist after one current refined-eye run smoke. |
| `subject_masks` | Candidate | Traditional subject writer creates `detection_source`, `masks_roi`, `mask_probs_roi`, and `available_channels` at `src/fisheye/segmentation/subject_segmentation.py:542` through `src/fisheye/segmentation/subject_segmentation.py:546`, plus metrics at `src/fisheye/segmentation/subject_segmentation.py:579` through `src/fisheye/segmentation/subject_segmentation.py:594`. U-Net subject writer creates `mask_probs_roi`, `available_channels`, metrics, and `detection_source` at `src/fisheye/segmentation/infer_unet_subject_masks.py:541`, `src/fisheye/segmentation/infer_unet_subject_masks.py:550`, `src/fisheye/segmentation/infer_unet_subject_masks.py:721`, and `src/fisheye/segmentation/infer_unet_subject_masks.py:969`. | Add to hard allowlist after traditional and U-Net smokes. |
| `refined_subject_masks` | Candidate with legacy caveat | Finalizer creates root arrays at `src/fisheye/refinement/finalize_subject_masks.py:1051` through `src/fisheye/refinement/finalize_subject_masks.py:1070` and metrics at `src/fisheye/refinement/finalize_subject_masks.py:1080` through `src/fisheye/refinement/finalize_subject_masks.py:1109`; assembler requires core arrays at load time around `src/fisheye/refinement/assemble_refined_subject_masks.py:241` through `src/fisheye/refinement/assemble_refined_subject_masks.py:256`. | Smoke finalizer and assembler paths first. Keep legacy mutation paths shadow-only until confirmed. |
| `arena_assignment` | Blocked | Spec requires `arena_ids` and `confidence`, but writer creates `arena_ids` at `src/fisheye/tracking/arena_assignment.py:698` and documents output as `arena_ids, n_detections_per_arena` at `src/fisheye/tracking/arena_assignment.py:914`; no `confidence` writer was found. | Either write a confidence array or demote/remove it from the required spec. |
| `tracking` | Candidate | Writer creates all required arrays at `src/fisheye/tracking/single_subject_per_arena.py:212` through `src/fisheye/tracking/single_subject_per_arena.py:229` and marks complete at `src/fisheye/tracking/single_subject_per_arena.py:318`; status is emitted as `tracks`, which aliases to `tracking`. | Add to hard allowlist after smoke. |

## Recommended Rollout

1. Keep `_ENFORCE_STAGE_ARRAY_VALIDATION_FOR = frozenset()` until real-run smokes are recorded for each candidate.
2. Fix rootless `ok` completion calls before expecting keypoint or batch paths to record registry completion.
3. Enable hard validation for `detect_quality` first; it has the smallest required surface and the easiest failure modes.
4. Enable `detect`, `refined_detect`, and `crop` next as a single detection-family slice.
5. Enable `subject_masks`, `refined_eye_masks`, and `tracking` only after one current writer smoke per stage.
6. Leave `keypoints`, `eye_masks`, `arena_assignment`, and `background` shadow-only until the spec/writer mismatch is resolved.

## Follow-Up Checklist

- Add a per-stage smoke fixture or fake-group test for every candidate stage before adding it to the enforcement allowlist.
- Add direct tests for rootless `ok` status rejection on all remaining completion wrappers.
- Add one test that validates `detect_quality` nested run resolution and documents the uniqueness assumption for quality run names.
- Decide whether `keypoints` should have one broad spec with optional method diagnostics or separate `traditional_keypoints` and `yolo_keypoints` method-family validators.
- Decide whether `eye_masks` should have method-family validators instead of one required contour-heavy spec.
- Update `arena_assignment` by either writing `confidence` or removing it from required arrays.
- If `background` should become registry-visible, add a real source-frame-index array and a stage completion call.
