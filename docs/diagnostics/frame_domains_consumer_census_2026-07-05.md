# FrameDomains Consumer Census - 2026-07-05

Status: Slice C migration census, checkpoint 1 accepted and merged on `sun` as
`e657742`; continuation cleared in `HANDOFF_2026-07-05.md` at `e37e7ea`.

Scope: consumers that translate between stored-zarr, source/acquisition, run-frame, and
crop-video frame domains, or infer frame-domain axis length with local arithmetic.
Deprecated eye-mask paths are intentionally excluded.

## Migration Rules

- Use `Recording.frame_domains()` when the consumer already holds a `Recording`.
- Use direct `FrameDomains(root=...)` for diagnostics/utilities that already hold an
  open zarr root and do not otherwise use `Recording`.
- Preserve existing semantics exactly. If a local translation appears wrong, migrate it
  faithfully or stop and report the latent bug.
- Do not replace vectorized NumPy indexing or count logic in hot paths with the current
  per-value `FrameDomains.convert()` implementation. Add/vectorize the resolver first, or
  defer the consumer as blocked on vectorized resolver support.

## Census

| Proposed order | Site | Domain pair / local arithmetic | Risk | Migration note |
| ---: | --- | --- | --- | --- |
| 1 done | `src/fisheye/diagnostics/check_training_crop_pynvvc_pixel_parity.py` | `stored_zarr_frame -> source_video_frame` via `raw_video/original_frame_indices[local]` | Low | Merged in checkpoint 1. Direct `FrameDomains(root=...)` pattern. |
| 2 in progress | `src/fisheye/visualization/detection_coverage_dashboard.py` | Stored/run frame-count inference; direct `raw_video/original_frame_indices.shape[0]`; `frame_indices.max()+1` for observed minimum | Low | Read-only visualization. Count migration can use `FrameDomains.count()` and keep vectorized `np.bincount`. |
| 3 | `src/fisheye/refinement/detect_quality.py` | Run/stored frame-count inference; sampled import metadata; `frame_indices.max()+1`; `np.bincount(..., minlength=...)` | Medium | Mostly read-only analysis but writes quality reports. Migrate count resolution carefully with old-vs-new report equivalence. |
| 4 | `src/fisheye/utils/training_image_profile.py` | Detection source/acquisition frames to stored raw rows via direct range check plus `original_frame_indices` inverse lookup | Medium | Diagnostic/profile utility. Existing path already uses Python lookup; preserve `-1` unmapped behavior if migrated before vectorized resolver. |
| 5 | `src/fisheye/diagnostics/check_training_crop_pynvvc_pixel_parity.py` sibling paths and `src/fisheye/utils/regenerate_training_crops_pynvvc.py` | Stored crop frame rows to source-video frames via direct `original_frame_indices` indexing | Medium | Some consumers write regenerated crop artifacts. Use direct root resolver only for non-hot mapping, otherwise wait for vectorized support. |
| 6 | `src/fisheye/detection/detect_keypoints_yolo.py` | Crop/run frame arrays to output `frame_counts` / `n_rois`; `frame_indices.max()+1`; `np.bincount` | High | Writer and large-array path. Blocked on vectorized resolver support or explicit count-only migration. |
| 7 | `src/fisheye/shared/crop_image_source.py` | Crop-row to `crop_video_frame` via `source_crop_video_frame_indices` | High / gated | Crop-video consumer migration requires real dropped-frame evidence per design approval record. Do not migrate before that gate. |
| 8 | `src/fisheye/refinement/refine_detect.py`, `src/fisheye/shared/refined_detect_curation.py`, `src/fisheye/tracking/arena_assignment.py`, `src/fisheye/tracking/crop.py` | Run-frame count inference and detection/crop frame count arrays; multiple `np.bincount` and `max()+1` patterns | High / hot path | Writers/curation/tracking paths. Defer broad migration until vectorized/batch resolver support exists. |
| 9 | Producer/exporter stampers: `src/fisheye/utils/build_analysis_acquisition_crop_run.py`, `src/fisheye/utils/build_hybrid_acquisition_offline_crop_run.py`, `src/fisheye/utils/export_acquisition_crop_pose_training_zarr.py`, `src/fisheye/utils/append_acquisition_crop_video_training.py` | Producer-side frame-domain arrays and count stamping | Highest | Writer phase. Add explicit semantics/stamps only after read-side patterns are stable. |

## Notes

- The design doc's named high-risk consumers remain `detect_quality.py`,
  `detect_keypoints_yolo.py`, and `shared/crop_image_source.py`.
- `shared/crop_image_source.py` is deliberately not next despite being design-listed:
  the approved design requires real acquisition crop-video dropped-frame evidence before
  crop-video consumer migration.
- The first continuation migration should stay count-only or diagnostic-only until a
  vectorized/batch resolver API is approved for large frame arrays.
