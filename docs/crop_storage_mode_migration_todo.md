# Crop Storage Mode Migration TODO

Purpose: migrate `Palette` from materialized-only crop handling to mixed-mode
crop storage support, while keeping training artifacts materialized and allowing
analysis crop planning to default to geometry-only where readers are ready.

Design reference:
- `docs/crop_live_view_vs_materialized_stream_design.md`
- `docs/geometry_live_gpu_design_note.md`

Date anchored: 2026-03-06.

## Decision Snapshot (Current)

- [x] Direct crop writer defaults remain materialized unless the caller passes
  an explicit writer/storage-mode choice.
- [x] `crop_batch` defaults analysis archives to `geometry_only` when neither
  CLI nor config specifies `crop_storage_mode`.
- [x] Training archives default to materialized crop runs and reject
  `geometry_only`.
- [x] Geometry-only runs must not become `crop_runs.latest` during the initial
  migration.
- [x] `crop_runs.latest` remains backward-compatible and materialized-compatible;
  mixed-mode support should add `latest_materialized` and `latest_any` rather
  than redefining `latest`.
- [x] Analysis/production archives should not require imported/downsampled
  image payloads as part of the baseline contract.
- [x] Keypoint and eye-mask training artifacts remain materialized by default.
- [x] Training artifacts are intentionally duplicated, self-contained, stable
  dataset artifacts.
- [x] Training zarrs must reject `crop_storage_mode=geometry_only` writes so
  canonical crop ROIs remain persisted in the archive.
- [x] No separate top-level crop-image datastream is introduced in the initial
  migration.
- [x] The live-crop provenance contract should include `frame_counts` and
  `detection_indices` so ROI-derived data remains fully traceable.
- [x] Bulk ROI inference on large full-frame sources should support a temporary
  ROI cache instead of relying on pure live decode.
- [x] Traditional detection/keypoint pipelines remain explicitly
  imported/materialized-only.
- [x] Manual bbox move/resize edits should be supported as idempotent in-place
  patches to the active manual refined-detect state when row identity is
  preserved.
- [x] Add/delete/split/merge detection edits should be treated as a new manual
  refined-detect revision internally, even if the UI still presents one latest
  manual state.

## Primary Touchpoints

- [x] `src/fisheye/shared/crop_image_source.py`
- [x] `src/fisheye/utils/backfill_crop_storage_metadata.py`
- [x] `src/fisheye/tracking/crop.py`
- [x] `src/fisheye/detection/detect_keypoints_yolo.py`
- [x] `src/fisheye/segmentation/eye_segmentation_yolo.py`
- [x] `src/fisheye/utils/keypoint_retry.py`
- [x] `src/fisheye/detection/detect_traditional.py`
- [x] `src/fisheye/detection/detect_keypoints_traditional.py`
- [ ] `src/fisheye/segmentation/eye_segmentation.py`
- [x] `src/fisheye/segmentation/infer_unet_eye_masks.py`
- [ ] `src/fisheye/training/zarr_yolo_dataset_loader.py`
- [ ] `src/fisheye/training/zarr_eye_mask_dataset.py`
- [ ] `src/fisheye/utils/export_keypoint_training_zarr.py`
- [ ] `src/fisheye/utils/export_eye_mask_training_zarr.py`
- [x] `src/fisheye/shared/zarr/stage_arrays.py`
- [x] `src/fisheye/diagnostics/check_crop_runs.py`
- [ ] Palette/Crimson crop viewers and review tools that currently dereference
  `crop_group["roi_images"]` directly

## Phase 0: Shared Contract + Provenance Groundwork

- [x] Backfill `crop_storage_mode`, `latest_materialized`, and `latest_any`
  across existing analysis and training archives.
- [x] Add `crop_storage_mode` attr support to crop writers.
- [x] Define the minimum live-crop provenance contract:
  `frame_indices`, `frame_counts`, `roi_coordinates_full`, `roi_size`,
  `bbox_norm_coords`, `detection_indices`, source detect lineage, and
  video-source resolution (`raw_video/images_full` or `source_video_path`).
- [ ] Define how live-crop decode details are recorded for reproducibility
  (backend, interpolation, grayscale conversion, dtype/precision).
- [x] Add a shared ROI resolver abstraction that:
  - reads `roi_images` when present,
  - otherwise reconstructs ROI pixels from crop geometry + source frames/video.
- [x] Add test fixtures/helpers for geometry-only crop runs.
- [x] Keep all crop writers materialized by default during this phase.

## Phase 1: Core Sequential ROI Inference

- [x] Migrate YOLO keypoint inference to the shared ROI resolver.
- [x] Migrate YOLO eye-mask inference to the shared ROI resolver.
- [x] Migrate keypoint retry to the shared ROI resolver.
- [x] Preserve source crop storage mode in downstream provenance.
- [x] Support explicit `--crop-run` / selected crop-run reads for geometry-only
  runs without changing default/latest behavior.
- [ ] Add parity tests comparing materialized vs live-crop inference inputs on
  the same crop run metadata.

## Phase 2: Runtime ROI Cache

- [x] Define `roi_cache_policy = never | auto | always` for ROI-model
  inference.
- [x] Implement a shared temporary ROI cache keyed by archive/run/provenance.
- [x] Keep temporary caches outside canonical archives by default.
- [x] Ensure temporary caches do not alter `crop_runs.latest`,
  `latest_materialized`, `latest_any`, or review status.
- [x] Treat temporary ROI cache layout as a separate scratch/runtime policy
  rather than inheriting canonical crop-run layout semantics.
- [x] Add a small ROI cache inspection/pruning utility so scratch caches are
  easy to locate, size, and delete.
- [x] Wire cache support into YOLO keypoint inference, YOLO eye-mask
  inference, and keypoint retry.
- [x] Attempt a smoke-archive `geometry_live` benchmark.
  Result on `2026-03-07`: abandoned after more than `10 minutes` with little
  visible progress, which is sufficient evidence that large-frame sequential ROI
  inference should strongly prefer the cache-backed path.
- [x] Benchmark materialized vs geometry-cache-build vs geometry-cache-reuse on
  the smoke archive.
  Result on `2026-03-07`: after inheriting the crop run ROI layout into the
  temporary cache, warm-cache geometry-only runs were near parity with
  materialized runs for both keypoints and U-Net eye masks.
- [x] Prototype `geometry_live_gpu` for external-video crop sources by reusing
  the GPU decode/crop kernels from the temporary ROI cache writer.
- [x] Benchmark `geometry_live_cpu` vs `geometry_live_gpu` vs
  `geometry_cache_build` vs `geometry_cache_reuse` on a non-smoke archive.
  Latest signal on `2026-04-04`: a representative analysis archive without
  `raw_video/images_full` fell back to `source_video_path`, and the pure
  CPU-backed `geometry_live` benchmark was abandoned after roughly `20 minutes`
  with low GPU utilization.
  Follow-up result on `2026-04-04`: `geometry_live_gpu` keypoints completed,
  but remained strongly ROI-read-bound on `4512x4512` source video
  (`~513.0s` wall, `423.44s` in `roi_read`, about `45 poses/s`), while the
  corresponding materialized keypoint baseline was about `77.2s`. The U-Net
  eye-mask pass hit Decord GPU `CUDA out of memory`. Smaller live settings
  (`roi_live_gpu_chunk_frames=8`, `eye_batch_size=64`) allowed the full
  benchmark to complete, but the path still remained about `5x` slower than
  the materialized/cached baselines for both keypoints and eye masks.
  Follow-up result on `2026-04-04`: with the scratch-cache GPU chunk default
  lowered to `32`, the recommended cache-backed path completed on the same
  archive. `geometry_cache_build` wall time was about `342.4s` for keypoints
  and `179.5s` for eye masks because it included first-time cache population,
  while warm `geometry_cache_reuse` was near parity with the materialized
  baselines (`83.4s` keypoints vs `77.2s` materialized, `170.3s` eye masks vs
  `172.9s` materialized).
  Conclusion: temporary local ROI cache remains the preferred analysis path for
  large full-frame sources; `geometry_live_gpu` is useful as a fallback/debugging
  path, not the default high-throughput workflow.
- [x] Add a flat binary workflow-cache backend and submit wrappers for cluster
  smoke testing:
  - `scripts/py -m fisheye.utils.build_flat_roi_cache`
  - `scripts/submit_flat_roi_cache_bsub.sh`
  - `scripts/submit_crop_flat_roi_cache_bsub.sh`
  These caches are workflow artifacts, not canonical crop runs. The 2026-05-16
  smoke path builds them on node-local scratch and publishes payload-first,
  manifest-last to `/misc/public/palette_cache/<workflow_id>/roi_cache`.

## Phase 3: Secondary ROI Consumers

- [x] Keep traditional detection explicitly imported/materialized-only.
- [x] Keep traditional keypoint inference explicitly materialized-only.
- [x] Decide whether traditional eye-mask inference and U-Net helpers should
  stay materialized-only or gain mixed-mode/cache support.
  Decision: U-Net eye-mask inference supports mixed-mode/cache; traditional
  pipelines remain materialized-only.
- [x] Migrate unified subject-mask inference stages to the shared ROI
  resolver/cache path.
  Result on `2026-04-05`: `run_sam_subject_masks.py` completed successfully on
  a representative `geometry_only` analysis archive with a warm shared
  temporary ROI cache. The timing profile showed the bottleneck was not ROI
  loading (`roi_read` about `29.0s`, `1.0%` of wall time) but the SAM runtime
  itself (`model_predict` about `2416.0s`, `85.2%`) plus output write-back
  (`295.7s`, `10.4%`). This is acceptable because the current SAM path is
  primarily a teacher/pseudo-label generator, not the intended fast production
  segmenter.
- [ ] Migrate Palette viewers, tuners, diagnostics, and failure-review tools to
  the shared ROI resolver or an explicit cache path.
- [ ] Migrate Crimson crop preview / review surfaces to the same resolver/cache
  model.
- [ ] Define the manual bbox patch contract for Crimson/Palette:
  - move/resize existing bbox = row patch
  - add/delete/split/merge = revision bump
  - preserve stable detection identity where possible
- [ ] Benchmark representative review/tuning latency on geometry-only runs,
  with and without temporary ROI cache support.

## Phase 4: Training / Export / Validation

- [ ] Update keypoint training prep/export to accept geometry-only source
  archives and materialize ROIs into the merged training artifact.
- [ ] Update eye-mask training prep/export with the same policy.
- [ ] Document and enforce that analysis archives may be mixed/lean, while
  training artifacts remain self-contained materialized datasets.
- [ ] Document and enforce that temporary ROI caches are runtime accelerators,
  not durable training artifacts.
- [x] Document the cluster shared-cache policy in
  `docs/geometry_only_crop_workflow_cache_design.md`.
- [ ] Keep merged keypoint/eye-mask training zarrs materialized by default.
- [ ] Update training loaders so analysis archives may be mixed-mode, while
  training artifacts continue to assume materialized ROI tensors.
- [ ] Update validators/diagnostics so `roi_images` is conditional for analysis
  archives when `crop_storage_mode=geometry_only`, but still required for
  materialized training artifacts.

## Phase 4b: Manual Edit Propagation

- [ ] Patch manual refined-detect edits into crop runs without requiring a new
  user-visible run when row identity is preserved.
- [ ] Advance refined-detect revision metadata and crop signatures whenever a
  manual bbox edit changes effective crop geometry.
- [ ] Ensure temporary ROI cache identity includes the crop signature/revision
  so stale caches are not reused after manual edits.
- [ ] Add targeted keypoint patch/update support for geometry-only and
  materialized crop runs.
- [ ] Add targeted eye-mask patch/update or stale-marking support for affected
  rows after keypoint/crop edits.
- [ ] Define when curated eye-mask rows are auto-recomputed versus marked stale
  for explicit operator resolution.

## Phase 5: Writer Opt-In Mode

- [ ] Add an explicit crop writer mode:
  `crop_storage_mode = materialized | geometry_only`.
- [ ] Ensure geometry-only runs do not become `crop_runs.latest`.
- [ ] Add explicit pointer attrs:
  `latest_materialized` and `latest_any`.
- [ ] Add `latest_geometry_only` only if a real consumer needs it.
- [ ] Add an explicit materialization command to generate `roi_images` from a
  geometry-only run.
- [ ] Ensure materialization preserves row identity and provenance back to the
  geometry-only source run.

## Phase 6: Benchmarks + Rollout Readiness

- [ ] Benchmark live-crop vs temporary-cache vs materialized performance for:
  - sequential inference,
  - random-access review/tuning,
  - training export from geometry-only source archives.
- [x] Sequential inference benchmarked on the smoke archive for:
  - materialized baseline,
  - geometry-only with first-time cache build,
  - geometry-only with warm-cache reuse.
- [ ] Add at least one non-smoke archive benchmark before changing defaults.
- [ ] Measure storage savings on representative datasets.
- [ ] Run manual validation on a mixed repository containing both materialized
  and geometry-only crop runs, plus cache-backed ROI inference.
- [ ] Confirm that no migrated consumer still directly dereferences
  `crop_group["roi_images"]` outside the shared resolver path or an explicit
  materialized-only path.

## Phase 7: Default Re-Evaluation

- [ ] Decide whether any analysis/archive workflows should default to
  `geometry_only`.
- [ ] Keep keypoint/eye-mask training artifacts materialized regardless of
  analysis default.
- [ ] Revisit whether a separate crop-image cache/datastream is warranted only
  after reader migration is complete and a concrete coordination problem exists.

## Acceptance Criteria

- [ ] Mixed repositories are supported: some recordings may be materialized and
  others geometry-only.
- [ ] Analysis/production archives do not require imported/downsampled image
  payloads as part of their baseline contract.
- [ ] Core ROI inference works through the shared resolver and optional
  temporary ROI cache in both modes.
- [ ] Review/tuning tools can open geometry-only runs without breaking,
  directly or through an explicit cache strategy.
- [ ] Keypoint and eye-mask training exports can consume geometry-only source
  archives but still emit materialized training artifacts.
- [ ] Geometry-only runs cannot silently break `crop_runs.latest`.
- [ ] Traditional materialized-only pipelines fail clearly instead of silently
  selecting geometry-only inputs.
- [ ] Temporary ROI caches do not change canonical archive lineage or latest
  pointers.
- [ ] Materialized-mode behavior remains unchanged for existing users.
- [ ] In-place manual bbox edits preserve operator UX while still advancing
  internal revision/signature lineage.

## Non-Goals (For Initial Migration)

- [ ] Making geometry-only the immediate default for all crop writers.
- [ ] Removing materialized ROI support from production pipelines.
- [ ] Changing keypoint/eye-mask training artifacts to live-crop-at-train-time.
- [ ] Introducing a separate top-level crop-image datastream before the shared
  ROI resolver exists and the current reader set is migrated.

## Open Questions

- [ ] What should trigger `roi_cache_policy=auto`:
  source resolution, measured decode throughput, repeated downstream reuse, or
  explicit workflow intent?
- [ ] For large-frame archives, should `roi_cache_policy=auto` skip live
  probing entirely and go straight to cache-backed execution?
- [ ] Does `roi_cache_policy=auto` need stage-aware behavior
  (for example, sequential keypoints -> eye masks) or is a single archive-level
  heuristic enough?
- [ ] Which review/tuning workflows are fast enough to tolerate geometry-only
  mode without explicit temporary cache materialization?
- [ ] Where should temporary ROI caches live by default:
  per-recording scratch space, shared global cache, or user-configured cache
  root?
- [ ] Should detect-training exports remain the only geometry-first training
  path, or should that become a general model for ROI training exports too?
