# FrameDomains Consumer Census - 2026-07-05

Status: COMPLETE for all migratable consumers (2026-07-06). Items 1-6 and 8 are
merged on `sun`; item 7 is gated on dropped-frame evidence and item 9 is the
deferred writer-stamping phase — both are maintainer decisions, not open agent work.

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
| 5 done | `src/fisheye/diagnostics/check_training_crop_pynvvc_pixel_parity.py` sibling paths and `src/fisheye/utils/regenerate_training_crops_pynvvc.py` | Stored crop frame rows to source-video frames via direct `original_frame_indices` indexing | Medium | Parity checker in checkpoint 1; `regenerate_training_crops_pynvvc.py` in hotpath stage 1 (checkpoint-1 pattern replay + legacy fallback on resolver error; full-array equivalence test). |
| 6 done | `src/fisheye/detection/detect_keypoints_yolo.py` | Crop/run frame arrays to output `frame_counts` / `n_rois`; `frame_indices.max()+1`; `np.bincount` | High | Hotpath stage 2 (writer). Domain determination: `crop_source.frame_indices` is the crop run's own RUN_FRAME universe; `count(RUN_FRAME)` slot inserted before the `max()+1` fallback. Commander-verified no-op in every reachable config (count resolves iff `crop_group/frame_counts` exists iff lineage copy bypasses the bincount branch); byte-identity writer test landed. |
| 7 | `src/fisheye/shared/crop_image_source.py` | Crop-row to `crop_video_frame` via `source_crop_video_frame_indices` | High / gated | Crop-video consumer migration requires real dropped-frame evidence per design approval record. Do not migrate before that gate. |
| 8 done | `src/fisheye/refinement/refine_detect.py`, `src/fisheye/shared/refined_detect_curation.py`, `src/fisheye/tracking/arena_assignment.py`, `src/fisheye/tracking/crop.py` | Run-frame count inference and detection/crop frame count arrays; multiple `np.bincount` and `max()+1` patterns | High / hot path | Hotpath stage 3: count-resolution slots migrated with legacy fallbacks in refine_detect/curation/arena_assignment (per-file legacy-oracle tests); `tracking/crop.py` verified no-migration-site (all frame universes metadata-sourced, `shape[0]` uses are row validations). Accepted checkpoint-2-class delta in `_infer_num_frames`: resolver stored count can differ from `images_ds.shape[0]` only on internally inconsistent archives. |
| 9 | Producer/exporter stampers: `src/fisheye/utils/build_analysis_acquisition_crop_run.py`, `src/fisheye/utils/build_hybrid_acquisition_offline_crop_run.py`, `src/fisheye/utils/export_acquisition_crop_pose_training_zarr.py`, `src/fisheye/utils/append_acquisition_crop_video_training.py` | Producer-side frame-domain arrays and count stamping | Highest | Writer phase. Add explicit semantics/stamps only after read-side patterns are stable. |

## Notes

- The design doc's named high-risk consumers remain `detect_quality.py`,
  `detect_keypoints_yolo.py`, and `shared/crop_image_source.py`.
- `shared/crop_image_source.py` is deliberately not next despite being design-listed:
  the approved design requires real acquisition crop-video dropped-frame evidence before
  crop-video consumer migration.
- The vectorized resolver landed 2026-07-06 (`9ee35e9`); all unblocked items then
  migrated the same day. Remaining ad-hoc frame arithmetic is confined to items 7/9
  above plus the shared `metadata.py::get_total_frames` helper (a possible future
  consolidation target, deliberately outside this census's file list).
