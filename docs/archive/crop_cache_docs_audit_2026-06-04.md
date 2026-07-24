<!-- ARCHIVED 2026-07-04: dated point-in-time diagnostic snapshot, retained for history only. -->

# Crop/Cache Docs Audit - 2026-06-04

## Scope

This pass reviewed docs related to crop storage mode, geometry-only crop runs,
flat ROI caches, and downstream keypoint/segmentation consumption.

Cleanup update: the low-risk docs-only cleanup slice recommended below was
applied on 2026-06-04. The old crop persistence note now redirects to an
archived copy, the crop-storage TODO and reader inventory were refreshed, the
live-cropping divergence analysis received a stale-section note, and the
operator guide now includes `--roi-cache-manifest` handoff examples.

Primary files checked:

- `docs/geometry_only_crop_workflow_cache_design.md`
- `docs/archive/crop_live_view_vs_materialized_stream_design.md`
- `docs/crop_storage_mode_migration_todo.md`
- `docs/archive/crop_reader_geometry_only_inventory_2026-05-16.md`
- `docs/crop_persistence_tradeoff.md`
- `docs/geometry_live_gpu_design_note.md`
- `docs/inference_pipeline_divergence_analysis.md`
- `docs/operator_guide/pipeline_workflow.md`
- `docs/cluster_batching_guide.md`
- `docs/cluster_pipeline_migration_checklist.md`

## Current Source Of Truth

`docs/geometry_only_crop_workflow_cache_design.md` should remain the active
architecture document for geometry-only analysis crops plus workflow-local ROI
caches.

The current implementation policy matches the active docs:

- Analysis crop planning can produce `crop_storage_mode=geometry_only`.
- Training/export artifacts remain materialized and self-contained.
- `crop_runs.latest` remains materialized-compatible.
- `crop_runs.latest_any` points at the newest compatible crop run, including
  geometry-only.
- Flat ROI caches are workflow-local accelerators, not canonical zarr content.
- Downstream pose/segmentation stages should consume caches through
  `CropImageSource` / `--roi-cache-manifest`, not parse `.bin` payloads
  directly.

No severe contradiction was found in the active source-of-truth docs.

## Stale Or Partly Stale Docs

### `docs/crop_persistence_tradeoff.md`

Status: archive candidate.

Reason:

- It frames the future option as `crop_mode = persist | on_demand`.
- The implemented terminology is now `crop_storage_mode = materialized |
  geometry_only`, with optional flat ROI caches.
- Its useful content is already covered by
  `crop_live_view_vs_materialized_stream_design.md` and
  `geometry_only_crop_workflow_cache_design.md`.

Recommended action:

- Move to `docs/archive/crop_persistence_tradeoff.md`, or leave a two-line
  redirect to `geometry_only_crop_workflow_cache_design.md` and archive the
  original.

### `docs/inference_pipeline_divergence_analysis.md`

Status: stale section, keep file.

Reason:

- Section `10f` says several core inference components hard-require
  `crop_runs/<run>/roi_images`, including YOLO keypoints/eye masks.
- Current inventory says YOLO pose, YOLO eye masks, U-Net eye masks, U-Net
  subject masks, subject segmentation, SAM subject masks, and related visualizers
  are mixed-mode safe through `CropImageSource`.

Recommended action:

- Add an update note around section `10f` saying this was the pre-migration
  analysis; current mixed-mode reader inventory lives in
  `crop_reader_geometry_only_inventory_2026-05-16.md`.

### `docs/crop_storage_mode_migration_todo.md`

Status: active but needs refresh.

Issues:

- Phase 5 still has unchecked items for writer opt-in mode,
  `latest_materialized`, and `latest_any`, even though those are implemented in
  `tracking/crop.py` and `crop_batch.py`.
- The benchmark paragraph repeats the line `while warm geometry_cache_reuse was
  near parity with the materialized`.
- It does not mention the new local wrapper
  `fisheye.utils.crop_flat_roi_cache_batch`.

Recommended action:

- Mark the implemented Phase 5 items as complete.
- Remove the duplicate repeated line.
- Add the local wrapper to the Phase 2 flat-cache utilities list.
- Keep the open questions about `roi_cache_policy=auto`, review/tuning latency,
  and training/export migration.

### `docs/archive/crop_reader_geometry_only_inventory_2026-05-16.md`

Status: keep, update last-verified.

Reason:

- It is the clearest code-facing inventory of mixed-mode vs materialized-only
  readers.
- It needs a small refresh for the new local cache wrapper:
  `src/fisheye/utils/crop_flat_roi_cache_batch.py`.

Recommended action:

- Update `last_verified` to `2026-06-04`.
- Add `crop_flat_roi_cache_batch.py` to the registry/schema/batch planner row or
  a new workflow-cache row.
- Keep this doc separate from the architecture doc; it answers "which readers
  are safe?", not "what is the design?".

## Duplicated But Useful Historical Docs

### `docs/archive/crop_live_view_vs_materialized_stream_design.md`

Status: keep as historical design note, not active source of truth.

Reason:

- It already has a 2026-05-16 update note redirecting readers to
  `geometry_only_crop_workflow_cache_design.md`.
- It contains benchmark narrative and migration reasoning that is useful
  historical context.
- It overlaps heavily with `geometry_only_crop_workflow_cache_design.md`.

Recommended action:

- Do not archive yet unless doc volume becomes the priority.
- Strengthen the top note to say: "For current operator commands, use
  `operator_guide/pipeline_workflow.md`; for current cache architecture, use
  `geometry_only_crop_workflow_cache_design.md`."

### `docs/geometry_live_gpu_design_note.md`

Status: keep as benchmark/design background.

Reason:

- It documents why GPU live reads are fallback/debugging, not the preferred
  high-throughput path.
- This benchmark result is referenced by the active geometry-only cache design.

Recommended action:

- Add a top note saying the operational recommendation has moved to flat ROI
  caches and `crop_flat_roi_cache_batch`.

## Operator Guide Gaps

### `docs/operator_guide/pipeline_workflow.md`

Status: mostly current after the local-wrapper update.

Remaining gap:

- It shows how to build the flat ROI cache but does not show how to pass the
  resulting `--roi-cache-manifest` into keypoint or segmentation commands.

Recommended action:

- Add a short follow-up snippet after the crop section showing:
  - keypoints with `--roi-cache-manifest`
  - U-Net subject masks with `--roi-cache-manifest`
  - note that batch wrappers may use `--roi-cache-policy always` when no
    explicit manifest is passed.

### `docs/cluster_batching_guide.md`

Status: mostly current for cluster workflows.

Remaining gap:

- It covers `crop_batch` and LSF crop submitters, but the local serial
  crop+cache wrapper is not mentioned.

Recommended action:

- Add a workstation note pointing to
  `fisheye.utils.crop_flat_roi_cache_batch`.
- Keep LSF details in the cluster guide; avoid duplicating the full local
  command there.

## Archive Candidates

Recommended archive now:

- `docs/crop_persistence_tradeoff.md`

Possible archive later, after stronger redirects:

- `docs/archive/crop_live_view_vs_materialized_stream_design.md`
  - Only if the benchmark/history content is folded into
    `geometry_only_crop_workflow_cache_design.md`.
- `docs/geometry_live_gpu_design_note.md`
  - Only if its benchmark summary is folded into the active cache design.

Do not archive:

- `docs/geometry_only_crop_workflow_cache_design.md`
- `docs/archive/crop_reader_geometry_only_inventory_2026-05-16.md`
- `docs/crop_storage_mode_migration_todo.md`

## Suggested Next Cleanup Slice

Low-risk in-place edit set:

1. Archive or redirect `crop_persistence_tradeoff.md`.
2. Refresh `crop_storage_mode_migration_todo.md` checkboxes and remove the
   duplicate repeated sentence.
3. Refresh `crop_reader_geometry_only_inventory_2026-05-16.md` for
   `crop_flat_roi_cache_batch.py`.
4. Add a stale-section update note to
   `inference_pipeline_divergence_analysis.md`.
5. Add one operator guide snippet showing `--roi-cache-manifest` handoff to
   keypoints/segmentation.

These are docs-only changes and should not require code validation beyond
`git diff --check`.
