# Provenance Default Epoch Census

<!-- contract-meta
status: blocked
created: 2026-07-05
owner: jeremy
related: docs/provenance_finalization_enforcement_design.md,
         docs/provenance_enforcement_roadmap.md,
         agents_todo/brief_review_remediation_wave_2026-07-05.md
-->

## Summary

Slice C's default epoch bump is **not ready to apply** on current `sun`.

The epoch-2 provenance gate itself is already implemented. The high-volume detect,
crop, keypoint, and subject-mask bsub paths opt their runs parents into epoch 2 and pass
valid `run_provenance`. However, a repository-wide static census still finds many live
writers that create run parents through the default epoch and then call
`mark_run_complete(...)` without `run_provenance`.

If `require_runs_parent(...)` were changed today to stamp epoch 2 by default, those
writers would fail at completion. That is the intended fail-closed behavior for truly
unprovenanced writers, but the Slice C brief requires the census to be green before the
global default bump. It is not green.

## Census Method

Static AST census over `src/`:

```bash
scripts/py - <<'PY'
from __future__ import annotations
import ast
from pathlib import Path
root = Path("src")
mark = []
req = []
for path in root.rglob("*.py"):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name == "mark_run_complete":
            kws = {kw.arg for kw in node.keywords if kw.arg}
            mark.append((str(path), node.lineno, "run_provenance" in kws, "allow_missing_run_provenance" in kws))
        if name == "require_runs_parent":
            kws = {kw.arg: kw.value for kw in node.keywords if kw.arg}
            epoch = "none"
            if "completion_epoch" in kws:
                epoch = ast.unparse(kws["completion_epoch"])
            req.append((str(path), node.lineno, epoch))
print(len(mark), len(req))
PY
```

## Counts

| Surface | Count |
|---|---:|
| `mark_run_complete(...)` calls in `src/` | 58 |
| Completion calls with explicit `run_provenance` | 7 |
| Completion calls with explicit bypass | 0 |
| Completion calls without provenance/bypass | 51 |
| `require_runs_parent(...)` calls in `src/` | 58 |
| Parent creators explicitly opting into epoch 2 | 7 |
| Parent creators using an indirect `completion_epoch` variable | 1 |
| Parent creators still relying on the default epoch | 50 |

## Already Epoch-2 Instrumented

These are the high-volume paths already using `COMPLETION_EPOCH_REQUIRE_PROVENANCE`
and passing `run_provenance` to completion:

- `src/fisheye/detection/detect_yolo.py`
- `src/fisheye/detection/detect_keypoints_yolo.py`
- `src/fisheye/tracking/crop.py`
- `src/fisheye/utils/run_sam_subject_masks.py`
- `src/fisheye/utils/run_subject_mask_batch_pipeline.py`

The direct registry-model runners synthesize run provenance before invoking those
writers:

- `src/fisheye/utils/run_detect_with_registry_model.py`
- `src/fisheye/utils/run_keypoints_with_registry_model.py`

## Blockers To Default Bump

These completion writers currently call `mark_run_complete(...)` without provenance or
an explicit bypass. They must either synthesize valid run provenance, or be explicitly
kept at epoch 1 with a documented reason, before the global default epoch can safely
become 2.

| File | Line | Current classification |
|---|---:|---|
| `src/fisheye/tracking/arena_assignment.py` | 814 | no provenance |
| `src/fisheye/tracking/single_subject_per_arena.py` | 318 | no provenance |
| `src/fisheye/analysis/tail_kinematics_runs.py` | 701 | no provenance |
| `src/fisheye/analysis/track_kinematics.py` | 586 | no provenance |
| `src/fisheye/analysis/detection_occupancy_runs.py` | 811 | no provenance |
| `src/fisheye/analysis/compute_speed.py` | 1055 | no provenance |
| `src/fisheye/analysis/swim_bout_statistics.py` | 931 | no provenance |
| `src/fisheye/analysis/megabouts_classifier.py` | 531 | no provenance |
| `src/fisheye/analysis/bout_kinematics.py` | 3310 | no provenance |
| `src/fisheye/analysis/stimulus_response.py` | 2027, 2205 | no provenance |
| `src/fisheye/analysis/chaser_distance_runs.py` | 1091 | no provenance |
| `src/fisheye/analysis/subject_shape_runs.py` | 2536 | no provenance |
| `src/fisheye/analysis/import_stimulus_to_zarr.py` | 1989 | no provenance |
| `src/fisheye/analysis/tail_posture_view_runs.py` | 590 | no provenance |
| `src/fisheye/analysis/stimulus_epoch_runs.py` | 415 | no provenance |
| `src/fisheye/analysis/eye_angle_analysis.py` | 3534 | no provenance |
| `src/fisheye/analysis/detect_bouts_multi_level.py` | 2737 | no provenance |
| `src/fisheye/refinement/refine_keypoints.py` | 1804 | no provenance |
| `src/fisheye/refinement/finalize_subject_masks.py` | 4168 | no provenance |
| `src/fisheye/refinement/detect_quality.py` | 744 | no provenance |
| `src/fisheye/refinement/refine_detect.py` | 1520 | no provenance |
| `src/fisheye/tune/detect_training_promotion_backend.py` | 615, 1007 | no provenance |
| `src/fisheye/tune/refined_subject_mask_review.py` | 1909 | no provenance |
| `src/fisheye/segmentation/infer_unet_subject_masks.py` | 1258 | no provenance |
| `src/fisheye/segmentation/subject_segmentation.py` | 694 | no provenance |
| `src/fisheye/detection/detect_keypoints_traditional.py` | 1145 | no provenance |
| `src/fisheye/detection/detect_traditional.py` | 559 | no provenance |
| `src/fisheye/utils/export_acquisition_crop_pose_training_zarr.py` | 866, 867 | no provenance |
| `src/fisheye/utils/build_hybrid_acquisition_offline_crop_run.py` | 944 | no provenance |
| `src/fisheye/utils/backfill_completion_epoch.py` | 354 | no parent, legacy re-mark |
| `src/fisheye/utils/build_analysis_acquisition_crop_run.py` | 513 | no provenance |
| `src/fisheye/utils/refresh_training_review_status.py` | 281 | no provenance |
| `src/fisheye/utils/backfill_subject_mask_runs.py` | 662 | no provenance |
| `src/fisheye/utils/append_acquisition_crop_video_training.py` | 349 | no provenance |
| `src/fisheye/utils/export_subject_mask_training_zarr.py` | 1107, 1108 | no provenance |
| `src/fisheye/utils/detection_profile.py` | 967 | no provenance |
| `src/fisheye/utils/export_detect_training_zarr.py` | 297, 1976 | no provenance |
| `src/fisheye/utils/predict_training_detections.py` | 509 | no provenance |
| `src/fisheye/utils/export_keypoint_training_zarr.py` | 1574, 1575 | no provenance |
| `src/fisheye/utils/keypoint_profile.py` | 900 | no provenance |
| `src/fisheye/utils/import_acquisition_detections_to_detect_run.py` | 361 | no provenance |
| `src/fisheye/utils/merge_subject_mask_runs.py` | 632 | no provenance |
| `src/fisheye/preprocessing/background.py` | 261 | no provenance |
| `src/fisheye/shared/subject_mask_profile.py` | 692 | no provenance |
| `src/fisheye/diagnostics/compare_realtime_offline_detections.py` | 1255 | no provenance |

## Parent Creators Still Relying On Default Epoch

The corresponding parent-creation surface still relies on the default epoch at 50 call
sites. Changing the default to 2 without instrumenting the writers above would make
many of these newly-created parents enforce provenance immediately.

Representative groups:

- analysis run parents: kinematics, stimulus, occupancy, shape, bout, chaser-distance
- refinement/review parents: refined detect, refined keypoints, refined subject masks,
  detect quality
- legacy/traditional production parents: traditional detect/keypoints, background,
  subject segmentation
- training/export parents: detect/keypoint/subject-mask training exports, acquisition
  crop pose exports
- profile/diagnostic parents: detection/keypoint/subject-mask profiles, realtime/offline
  comparison

## Recommendation

Do **not** bump the default epoch in this slice.

Recommended next implementation slice:

1. Add a small shared helper for writer-side run provenance construction, so
   lower-volume writers can pass explicit provenance without reimplementing the same
   boilerplate.
2. Instrument the default-epoch writers in batches, starting with live production and
   training/export surfaces:
   - background, traditional detect/keypoints, arena assignment/tracking
   - refined detect/refined keypoints/refined subject masks/detect quality
   - training/export zarr writers
   - analysis writers
3. Re-run this census. When `mark_run_complete(...)` without provenance is limited to
   intentional legacy maintenance paths with recorded bypasses, then change
   `require_runs_parent(...)` to stamp epoch 2 by default.

Alternative, not recommended for the final enforcement goal: bump the default now and
explicitly opt every uninstrumented writer back down to epoch 1. That would make future
new writers default to epoch 2, but it would also leave most current writers outside the
provenance-required contract and obscure the remaining work.
