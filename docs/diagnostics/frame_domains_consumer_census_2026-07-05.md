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
- Hot-path guard SATISFIED as of 2026-07-06: `FrameDomains.convert()` is internally
  vectorized (lazy per-edge lookup caches; commit `9ee35e9`, legacy loop kept as test
  oracle). Large-array consumers may now migrate through `convert()` directly.

## Census

| Proposed order | Site | Domain pair / local arithmetic | Risk | Migration note |
| ---: | --- | --- | --- | --- |
| 1 done | `src/fisheye/diagnostics/check_training_crop_pynvvc_pixel_parity.py` | `stored_zarr_frame -> source_video_frame` via `raw_video/original_frame_indices[local]` | Low | Merged in checkpoint 1. Direct `FrameDomains(root=...)` pattern. |
| 2 done | `src/fisheye/visualization/detection_coverage_dashboard.py` | Stored/run frame-count inference; direct `raw_video/original_frame_indices.shape[0]`; `frame_indices.max()+1` for observed minimum | Low | Merged in checkpoint 2 (`2cbb2b8`). `FrameDomains.count(STORED_ZARR)`; vectorized `np.bincount` untouched. |
| 3 done | `src/fisheye/refinement/detect_quality.py` | Run/stored frame-count inference; sampled import metadata; `frame_indices.max()+1`; `np.bincount(..., minlength=...)` | Medium | Merged in checkpoint 3 (`9034ead`). `FrameDomains.count(RUN_FRAME)` with legacy fallback on resolver error; dict-level report equivalence pinned. |
| 4 done | `src/fisheye/utils/training_image_profile.py` | Detection source/acquisition frames to stored raw rows via direct range check plus `original_frame_indices` inverse lookup | Medium | Merged in checkpoint 4. Resolver inverse conversion with legacy last-wins dict fallback on `FrameDomainUnmappedError`; `-1` sentinel preserved on every branch. |
| 5 partial | `src/fisheye/diagnostics/check_training_crop_pynvvc_pixel_parity.py` sibling paths and `src/fisheye/utils/regenerate_training_crops_pynvvc.py` | Stored crop frame rows to source-video frames via direct `original_frame_indices` indexing | Medium | Parity checker migrated in checkpoint 1. `regenerate_training_crops_pynvvc.py` maps full ROI arrays vectorized — was deferred to the vectorized-resolver gate; UNBLOCKED as of `9ee35e9`. |
| 6 | `src/fisheye/detection/detect_keypoints_yolo.py` | Crop/run frame arrays to output `frame_counts` / `n_rois`; `frame_indices.max()+1`; `np.bincount` | High | Writer and large-array path. UNBLOCKED (resolver vectorized, `9ee35e9`); as a WRITER it requires a checkpoint after migration. |
| 7 | `src/fisheye/shared/crop_image_source.py` | Crop-row to `crop_video_frame` via `source_crop_video_frame_indices` | High / gated | Crop-video consumer migration requires real dropped-frame evidence per design approval record. Do not migrate before that gate. |
| 8 | `src/fisheye/refinement/refine_detect.py`, `src/fisheye/shared/refined_detect_curation.py`, `src/fisheye/tracking/arena_assignment.py`, `src/fisheye/tracking/crop.py` | Run-frame count inference and detection/crop frame count arrays; multiple `np.bincount` and `max()+1` patterns | High / hot path | Writers/curation/tracking paths. UNBLOCKED (resolver vectorized, `9ee35e9`); migrate after item 6's writer checkpoint pattern is approved. |
| 9 | Producer/exporter stampers: `src/fisheye/utils/build_analysis_acquisition_crop_run.py`, `src/fisheye/utils/build_hybrid_acquisition_offline_crop_run.py`, `src/fisheye/utils/export_acquisition_crop_pose_training_zarr.py`, `src/fisheye/utils/append_acquisition_crop_video_training.py` | Producer-side frame-domain arrays and count stamping | Highest | Writer phase. Add explicit semantics/stamps only after read-side patterns are stable. |

## Notes

- The design doc's named high-risk consumers remain `detect_quality.py`,
  `detect_keypoints_yolo.py`, and `shared/crop_image_source.py`.
- `shared/crop_image_source.py` is deliberately not next despite being design-listed:
  the approved design requires real acquisition crop-video dropped-frame evidence before
  crop-video consumer migration.
- The vectorized resolver landed 2026-07-06 (`9ee35e9`): items 5-remainder, 6, and 8
  are unblocked. Item 7's dropped-frame evidence gate and item 9's writer-phase
  ordering are unchanged.
