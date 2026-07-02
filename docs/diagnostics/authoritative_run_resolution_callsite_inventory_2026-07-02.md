# Authoritative Run Resolution Call-Site Inventory

Status: implementation inventory for `agent/authoritative-run-pointer`  
Date: 2026-07-02

## Rule

Authoritative-first resolution applies to consumption defaults: downstream stage inputs,
analysis readers, visualization readers, and training/export source selection. Inventory,
maintenance, reconciliation, and review/curation surfaces stay on latest/all-run semantics
until those tools explicitly learn the approval workflow.

If no authoritative pointer exists on a run parent, the authoritative resolver falls back
to the same latest-complete behavior used before this slice.

## Rerouted Direct Callers

These direct `resolve_latest_complete_run_name(...)` call sites now use
`resolve_authoritative_run_name(...)` because they choose a default run for downstream
consumption.

- `src/fisheye/analysis/chaser_distance_runs.py`
- `src/fisheye/analysis/chaser_egocentric_bearing.py`
- `src/fisheye/analysis/chaser_escape_freeze.py`
- `src/fisheye/analysis/cra_near_field.py`
- `src/fisheye/analysis/cra_primary_endpoint.py`
- `src/fisheye/analysis/detection_occupancy_runs.py`
- `src/fisheye/analysis/goodcopbadcop_epoch_behavior_summary.py`
- `src/fisheye/analysis/stimulus_epoch_runs.py`
- `src/fisheye/core/pipeline.py`
- `src/fisheye/detection/detect_keypoints_traditional.py`
- `src/fisheye/detection/detect_traditional.py`
- `src/fisheye/diagnostics/compare_realtime_offline_detections.py`
- `src/fisheye/diagnostics/prepare_detect_training.py`
- `src/fisheye/inference/predict_detections.py`
- `src/fisheye/inference/predict_pose.py`
- `src/fisheye/shared/crop_geometry.py`
- `src/fisheye/shared/eye_geometry_source.py`
- `src/fisheye/shared/refined_detect_resolution.py`
- `src/fisheye/shared/refined_subject_masks_io.py` for default run loading
- `src/fisheye/training/train_detection.py`
- `src/fisheye/training/train_pose.py`
- `src/fisheye/training/zarr_yolo_dataset_loader.py`
- `src/fisheye/utils/export_acquisition_crop_pose_training_zarr.py`
- `src/fisheye/visualization/plot_detection_epoch_heatmaps.py`

## Rerouted Indirect Surface

`src/fisheye/shared/zarr_helpers.py::resolve_zarr_run(...)` now resolves
authoritative-first when callers request implicit latest selection. Explicit `run_name`
selection and deterministic sorted fallback behavior are unchanged.

The following callers inherit the new default through `resolve_zarr_run(...)`:

- `src/fisheye/analysis/plot_stimulus_response_omr.py`
- `src/fisheye/analysis/stimulus_response.py`
- `src/fisheye/diagnostics/check_chaser_alignment.py`
- `src/fisheye/diagnostics/check_chaser_periodicity.py`
- `src/fisheye/diagnostics/check_smoothed_distance_nan.py`
- `src/fisheye/diagnostics/diagnose_camera_chaser_mapping.py`
- `src/fisheye/diagnostics/inspect_chaser_states.py`
- `src/fisheye/diagnostics/inspect_frame_alignment.py`
- `src/fisheye/diagnostics/inspect_frame_relationship.py`
- `src/fisheye/diagnostics/inspect_stimulus_events.py`
- `src/fisheye/diagnostics/inspect_stimulus_mapping.py`
- `src/fisheye/diagnostics/plot_chaser_alignment.py`
- `src/fisheye/diagnostics/validate_centroids.py`
- `src/fisheye/utils/backfill_subject_mask_runs.py`
- `src/fisheye/utils/build_virtual_collection_manifest.py`
- `src/fisheye/utils/export_cross_recording_analytics.py`
- `src/fisheye/utils/extend_keypoint_skeleton.py`
- `src/fisheye/utils/run_sam_subject_masks.py`
- `src/fisheye/visualization/detection_coverage_dashboard.py`
- `src/fisheye/visualization/subject_mask_inspector.py`
- `src/fisheye/visualization/visualize_sam_subject_prompts.py`

## Left On Latest/Inventory Semantics

These direct callers intentionally remain latest-complete based in this slice.

- `src/fisheye/registry/db.py`: registry extraction/indexing should reflect latest/all
  state, not the scientific authority pointer.
- `src/fisheye/registry/extractors/masks.py`: registry profile extraction.
- `src/fisheye/registry/maintenance.py`: completion repair/reconcile helpers.
- `src/fisheye/shared/refined_subject_masks_io.py` option discovery: labels the actual
  latest run in selectable run lists; default loading is authoritative-first.
- `src/fisheye/shared/refined_detect_curation.py`: curation mutation/materialization
  helpers target review surfaces; review-UI approval integration is a follow-up.
- `src/fisheye/tune/*`: review and promotion tools still target latest/manual review
  surfaces until they explicitly call `palette approve`.
- `src/fisheye/utils/run_keypoints_batch.py`: stage-presence checks and just-written
  run postprocessing keep latest-stage semantics rather than approval semantics.
- `src/fisheye/shared/zarr_run_completion.py`: primitive latest resolver and summaries
  keep latest-complete as a separately inspectable concept.

## Not Swept In This Slice

Some older UI modules read `latest` / `latest_complete` attrs manually without the shared
resolver. They were not swept here unless already touched for consumption defaults. The
follow-up review-UI integration should either route those through the authoritative
resolver or deliberately keep latest semantics per UI action.
