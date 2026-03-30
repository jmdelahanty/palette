# Refine Eye Masks Distributed Refactor TODO

Purpose: reduce Dask distributed transfer overhead in
`src/fisheye/refinement/refine_eye_masks.py` without changing the refined
eye-mask contract used by Palette and Crimson.

Date anchored: 2026-03-09.

## Current Problem

- `refine_eye_masks` now supports `threads`, `processes`, and `distributed`
  scheduling.
- In distributed mode, each `_process_and_write_chunk(...)` task currently:
  - refines one ROI chunk,
  - writes the main refined arrays to zarr,
  - returns full per-ROI `ROIOutput` payloads to the driver.
- The driver then reconstructs:
  - metrics arrays,
  - `metrics/reason`,
  - contour stores,
  - summary counts and means,
  - `reason_tag_counts`.
- Dask dashboard traces show `transfer-process_and_write_chunk` dominating the
  task timeline, which indicates the current worker-return payload is too large.

## Constraints We Must Keep

- `refined_eye_masks_runs/<run>` remains the canonical refined artifact.
- Refined outputs must keep:
  - `masks_roi`
  - `ellipse_params`
  - `ellipse_success`
  - `eye_separation` when present
  - `metrics/*`
  - `metrics/reason`
  - contour arrays:
    - `contour_left_ptr`
    - `contour_left_len`
    - `contour_right_ptr`
    - `contour_right_len`
    - `contours_left`
    - `contours_right`
- Contours are still required by the Palette eye-mask patch viewer/editor.
- Current direct run attrs remain the canonical lineage for active tooling.
- Historical nested `provenance` payloads should not be rewritten just to match
  repaired lineage attrs.

## Existing Storage Pattern To Reuse

- Detect/keypoint refinement already uses a fixed-shape primary reason encoding:
  - `reason_bytes` as null-terminated UTF-8 `uint8[n,width]`
  - optional `reason` text mirror
- Shared helper:
  - `src/fisheye/shared/detect_reason_codec.py`
- Runtime string guidance:
  - `docs/zarr_structure.md`

Eye-mask refinement currently does not follow that pattern; it writes only
`metrics/reason` as variable-length UTF-8 after driver-side aggregation.

## Recommended Refactor

### Phase 1: Separate Local vs Global Reason Logic

- Define worker-local reason tags:
  - `union_source`
  - `split_by_keypoint`
  - `empty_union`
  - `empty_mask_left`
  - `empty_mask_right`
  - `ellipse_fail_left`
  - `ellipse_fail_right`
  - `ellipse_fail_pair`
  - `small_area_left`
  - `small_area_right`
  - `small_area_pair`
- Keep driver-global reason tags:
  - `filtered_left`
  - `filtered_right`
  - `filtered_pair`
- Make this split explicit in code and docs before moving writes around.

### Phase 2: Worker-Written Fixed-Shape Outputs

- Precreate all fixed-shape refined output arrays before launching chunk tasks.
- Have workers write their own slices for:
  - `masks_roi`
  - `ellipse_params`
  - `ellipse_success`
  - `eye_separation`
  - optional `mask_probs_roi_refined`
  - fixed-shape metrics arrays under `metrics/`
- Move local per-ROI metric computation fully into the worker.

### Phase 3: Replace Returned ROIOutput Payloads

- Stop returning full per-ROI `ROIOutput` objects to the driver.
- Replace them with compact `ChunkSummary` payloads that contain only:
  - counts,
  - scalar sums/means accumulators,
  - local reason-tag counts,
  - chunk index/span metadata if needed.
- Driver should use `ChunkSummary` only for:
  - `summary_statistics`
  - `metrics_summary`
  - `reason_tag_counts`
  - provenance summary counters.

### Phase 4: Add Fixed-Shape Reason Flags

- Introduce a per-ROI fixed-shape reason representation for eye-mask refinement.
- Recommended form:
  - `metrics/reason_flags` as `uint32` or `uint64`
  - one bit per tag
- Workers write local reason flags directly by slice.
- Driver ORs in the global filter flags after the dataset-wide area statistics
  are known.
- Keep `metrics/reason` as a derived human-readable mirror, not the hot-path
  primary representation.

### Phase 5: Add `reason_bytes`

- Reuse the existing reason codec approach for refined eye masks:
  - primary: `reason_bytes`
  - optional mirror: `reason`
- Materialize `reason_bytes` and `reason` after all local/global flags are
  finalized.
- Update read-order docs to include eye-mask refinement in the same storage
  pattern used by refined detect/keypoint groups.

### Phase 6: Contour Strategy

Contours are the one variable-length output that does not fit neatly into the
fixed-shape worker-write plan. Two viable approaches:

1. Recommended first step: post-pass contour build
   - First pass writes/refines masks and all fixed-shape outputs.
   - Second pass derives contour arrays from final `masks_roi`.
   - This keeps the distributed worker payload small and avoids packing
     variable-length contour buffers inside the chunk task.

2. Later optimization: worker-written chunk contour stores
   - Workers write chunk-local contour buffers into temporary storage.
   - Driver merges chunk-local contour stores into the final packed arrays.

Start with the post-pass.

## Proposed Execution Order

1. Introduce local/global reason-tag separation.
2. Precreate fixed-shape metrics arrays and move worker writes there.
3. Replace returned `ROIOutput[]` with `ChunkSummary`.
4. Add `reason_flags`.
5. Derive `reason_bytes` and `reason`.
6. Move contours to a post-pass.
7. Rebenchmark distributed refinement on the smoke archive.

## Acceptance Criteria

- Distributed refinement no longer returns large per-ROI objects to the driver.
- Dask dashboard shows substantially lower `transfer-process_and_write_chunk`
  overhead.
- Refined run outputs remain contract-compatible for:
  - Palette patch viewer/editor
  - Crimson refined-mask reader
  - provenance/status tooling
- `metrics/reason` remains readable for diagnostics.
- Eye-mask refinement adopts the same primary reason-storage strategy already
  used elsewhere in the repo (`reason_bytes` + optional `reason` mirror).

## Non-Goals

- Changing the refined-eye-mask run contract to remove contours.
- Rewriting historical nested provenance payloads on old runs.
- Moving manual editing from refined runs back onto raw `eye_masks_runs`.
- Redesigning the current review status / approval model.
