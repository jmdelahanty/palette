# Refined Subject-Mask Geometry Cache And Propagation Design
<!-- design-meta
status: draft
last_updated: 2026-04-29
-->

Purpose: define how Palette should store contour/topology-derived mask
geometry, how those derived values relate to canonical refined masks, and what
should happen when a refined mask is edited after downstream analysis products
already exist.

This document connects the boundaries defined in:

- [refined_subject_masks_runs_contract.md](refined_subject_masks_runs_contract.md)
- [subject_shape_runs_contract.md](subject_shape_runs_contract.md)
- [subject_body_mask_qc_design.md](subject_body_mask_qc_design.md)
- [repo_wide_staleness_policy.md](repo_wide_staleness_policy.md)

## Decision Summary

- `refined_subject_masks_runs/<run>/masks_roi` is the canonical refined mask
  pixel surface for subject-mask analysis.
- Component contours are mask-local derived caches, not independent biological
  truth.
- Component-local scalar metrics and simple QC should be regenerated
  row-locally when the corresponding mask row/component is edited.
- Downstream analysis products should not silently mutate when a refined mask
  changes. They should be marked stale and recomputed through an explicit
  analysis action.
- Full topology graphs should not be persisted by default yet. Persist stable
  scalar topology/QC metrics first.
- Visualizers should be explicit about whether they draw contours from current
  masks, persisted contour caches, or both.

The governing rule is:

```text
mask edit -> regenerate same-row mask-local caches -> mark downstream derived rows stale -> explicit targeted recompute creates or updates analysis outputs
```

## Terminology

### Canonical Refined Mask

The canonical refined mask is the binary mask stored in:

```text
refined_subject_masks_runs/<run>/masks_roi[row, channel, y, x]
```

This is the editable/refined working artifact. It is the source for downstream
mask-local geometry and biological analysis. It is distinct from raw model
probabilities in `subject_mask_runs`, which remain immutable provenance
evidence.

### Mask-Local Geometry Cache

A mask-local geometry cache is recomputable from one component mask without
choosing a biological coordinate frame.

Examples:

- contour points
- area
- centroid
- bbox
- mask-present flags
- connected-component count
- hole count or hole area
- solidity
- simple external-contour perimeter
- skeleton endpoint/branchpoint counts used as mask QC

These values belong with `refined_subject_masks_runs` because they describe the
mask itself.

### Downstream Biological Analysis

Downstream biological analysis requires an interpretation beyond one mask.

Examples:

- body centerline
- B-spline body model
- tail base/tail tip assignment
- anatomical body frame
- eye angles relative to body frame
- swim-bladder position relative to body centerline
- track kinematics
- swim-bout and bout-kinematics outputs

These values belong in `analysis/<stage>_runs`, such as
`analysis/subject_shape_runs` or `analysis/eye_angle_runs`.

### Topology

Topology can mean several different things. Palette should avoid storing
ambiguous "topology" structures without a specific contract.

Preferred first-class topology outputs are scalar QC metrics:

- `component_count`
- `hole_count`
- `hole_area_px`
- `skeleton_endpoint_count`
- `skeleton_branchpoint_count`
- `thin_spur_score`
- `requires_review`
- `severe_qc_failure`
- `reason_bytes`

Full skeleton graphs, contour nesting trees, and shape graph structures should
stay out of the default refined-mask contract until there is a specific
analysis consumer and schema.

## Current Implementation Status

As of 2026-04-29:

- `eye_left` and `eye_right` have persisted component contours using packed
  variable-length arrays under:
  `components/eye_left/contours/` and `components/eye_right/contours/`.
- `subject_body` and `swim_bladder` contours are not assumed on existing
  archives. They can be added with the component-contour backfill command; the
  feeding canary refined run has been backfilled for validation.
- Subject-shape overlays currently compute body, swim-bladder, and eye display
  contours from `masks_roi` at render time by default, and expose
  `mask`/`persisted`/`auto`/`compare` contour-source modes for audit.
- The subject-shape run persists interpreted outputs such as centerlines,
  body-frame-related arrays, tail anchors, and swim-bladder caudal anchor
  points. Those are not replacements for refined-mask component contours.
- `subject_body` mask-level QC is designed but not fully implemented as the
  default writer for every refined subject-mask run.
- The docs already reserve `components/<component>/contours/` as the correct
  location for component contour caches.
- A generic component contour helper and dry-run/apply backfill command exist:
  `scripts/py -m fisheye.utils.backfill_refined_subject_component_contours`.

This means the contact-sheet overlay command:

```bash
montage /tmp/palette_subject_shape_overlays/subject_shape_overlay_subject_shape_centerline_canary_20260428_row_*.png \
  -tile 3x2 -geometry +8+8 \
  /tmp/palette_subject_shape_overlays/contact_sheet.png
```

used a mixed visualization:

- persisted subject-shape geometry for centerlines and anchor points
- current `masks_roi` values for component masks
- display-time contours recomputed from masks for outlines

It did not draw body/swim contours from persisted contour arrays because those
arrays do not exist yet. It also did not draw persisted eye contours because
the script uses one generic mask-boundary rendering path for every component.

## Component Contour Storage

Contours are variable-length. One frame can have zero points, another can have
hundreds. Zarr does not natively store a clean ragged list per row, so Palette
uses a packed table layout:

```text
components/<component>/contours/
  attrs:
    schema_id                       "component_contours_v1"
    coordinate_space                "roi_pixels"
    source_component                "subject_body" | "swim_bladder" | "eye_left" | "eye_right"
    source_mask_run                 <refined subject-mask run id>
    source_mask_label_schema_id
    method                          e.g. "largest_external_contour"
    method_version                  e.g. 1
    boundary_policy                 e.g. "external_only"
    point_order                     e.g. "xy"
    generated_at_utc
  ptr                               (N,) int64
  len                               (N,) int32
  points_xy                         (M, 2) float32
```

For row `i`:

```python
start = ptr[i]
n = len[i]
if start >= 0 and n > 0:
    contour_xy = points_xy[start : start + n]
else:
    contour_xy = empty contour
```

This is the same concept already used for eye contours.

### Missing Contour Rows

Rows without a usable contour should use:

```text
ptr[row] = -1
len[row] = 0
```

The reason for the missing contour should be captured separately in component
QC or validity arrays, not encoded only by `ptr`/`len`.

### Row-Local Regeneration

When one row/component mask is edited, Palette should not need to rewrite every
contour in the run.

Preferred update behavior:

1. Compute the new contour for the edited row/component.
2. Append its points to `points_xy`.
3. Update only `ptr[row]` and `len[row]`.
4. Record an update event or row revision so downstream readers can detect that
   this row changed.

The old contour points become orphaned cache data. That is acceptable for
row-local editing. A later compaction command can rebuild packed contour arrays
if storage overhead becomes meaningful.

### Row-Local Update Tracking

Each component group may carry row-local revision arrays:

```text
components/<component>/
  attrs:
    row_update_schema_id            "refined_subject_component_row_updates_v1"
    last_row_update_at_utc
    last_row_update_reason
  row_revision                      (N,) int64
  row_updated_at_utc_bytes          (N, 40) uint8
  row_update_reason_bytes           (N, 128) uint8
```

`row_revision[row]` increments whenever Palette explicitly refreshes
mask-local caches for that row/component after a mask edit or source sync. A
downstream analysis run that consumes refined masks should record the source
run plus the relevant row revisions if it wants row-local drift detection.

This row revision is not a biological identity and not a replacement for
stable row IDs. It is only a cache/source-change generation for one semantic
component row.

If a contour cache is created by row-local updates before a full contour
backfill exists, the contour group should be marked as partial:

```text
components/<component>/contours.attrs["cache_coverage"] = "partial_row_updates"
```

Full rebuild/backfill writers should mark:

```text
components/<component>/contours.attrs["cache_coverage"] = "full_indexed_rows"
```

Readers must still treat `ptr[row] = -1` and `len[row] = 0` as "no persisted
contour for this row" and fall back to `masks_roi` when needed.

### Contour Method Policy

Different algorithms produce different contour points. The contour method must
be recorded because these are not interchangeable:

- OpenCV external contour
- scikit-image marching squares
- largest external contour only
- all external contours
- holes included
- subpixel contour versus pixel-edge contour

The first body/swim implementation should use a conservative, documented
method such as:

```text
method = "largest_external_contour"
boundary_policy = "external_only"
coordinate_space = "roi_pixels"
```

This is enough for visualization, simple mask QC, and future Crimson overlays.
Hole/nested-boundary support should be a separate schema extension if needed.

### Contour Size And Simplification

Full-resolution contours can look surprisingly large in point count. For the
feeding canary, the first body/swim contour backfill produced roughly:

- `subject_body`: 8.75 million `(x, y)` points, about 70 MB as float32
- `swim_bladder`: 1.11 million `(x, y)` points, about 9 MB as float32

This is small compared with dense `masks_roi` storage for the same recording,
but it is still not free across many recordings.

Current policy:

- Keep the first persisted contours full-resolution.
- Treat contours as caches derived from canonical masks.
- Do not simplify by default until a consumer need or storage pressure is
  demonstrated.

Future option:

- Add a simplification parameter such as `simplify_tolerance_px`.
- Record the simplification method in attrs, for example
  `simplification_method = "douglas_peucker"` and
  `simplify_tolerance_px = 0.5`.
- Keep full masks canonical so simplified contours never become the only
  representation of the refined shape.
- Prefer simplified contours for visualization and realtime consumers if full
  contour transfer becomes a bottleneck.

## Visualization Policy

A visualization should be explicit about contour source.

Recommended modes:

- `mask`: compute display contours from current `masks_roi`.
- `persisted`: draw stored `components/<component>/contours`.
- `auto`: draw persisted contours when present and source-compatible, otherwise
  compute from `masks_roi`.
- `compare`: draw both persisted and recomputed contours to audit stale or
  divergent geometry.

Default should remain `mask` for general review overlays because it always
matches the current mask pixels. `persisted` and `compare` are audit modes.

This prevents a common failure mode: showing a stale cached contour that no
longer matches an edited mask.

## Edit And Propagation Policy

### What Regenerates Immediately

When a refined mask row/component is edited, the edit transaction should
immediately regenerate same-row, same-component mask-local caches.

Examples:

- `mask_present`
- `area_px`
- centroid
- bbox
- component contour
- simple component metrics
- component-local QC flags and reasons

This keeps the refined-mask run internally consistent.

### What Should Not Regenerate Silently

The following should not update automatically as an invisible side effect of a
mask edit:

- `analysis/subject_shape_runs`
- `analysis/eye_angle_runs`
- `analysis/track_kinematics_runs`
- `analysis/swim_bout_runs`
- `analysis/bout_kinematics_runs`
- persisted plots or reports derived from those analysis runs

These outputs should be marked stale or source-drifted, then recomputed by an
explicit command.

### Why Not Automatically Propagate Everything?

Automatic full propagation is convenient but scientifically risky:

- old plots can change without an explicit analysis action
- downstream values may use smoothing, temporal context, or track identity
- downstream products may have their own manual review or chosen parameters
- one corrected mask row may affect multi-row outputs such as smoothed
  kinematics or bout segmentation

The safer model is explicit targeted recompute.

### Scratch/Dish Artifact Example

If an operator edits one `subject_body` mask row to remove a dish scratch:

1. Update only that row/channel in `masks_roi`.
2. Regenerate that row's `subject_body` mask-local caches.
3. Regenerate that row's `subject_body` contour.
4. Regenerate that row's `subject_body` QC metrics and reason tags.
5. Mark dependent subject-shape rows stale, or record source-drift metadata
   that lets validation detect staleness.
6. Run an explicit targeted subject-shape recompute for that row or affected
   row window.
7. Run explicit downstream recomputes only where the affected subject-shape
   outputs are consumed.

The mask edit should not silently rewrite eye-angle, kinematic, bout, or plot
artifacts.

## Canonical Input Boundary

For downstream analysis, the canonical input is the approved or selected
`refined_subject_masks_runs/<run>` plus its recorded mask labels and source
metadata.

For refinement, the canonical input may be:

- raw probabilities from `subject_mask_runs`
- existing refined masks
- manual edits
- compatibility imported eye/swim/body masks

The output of refinement is still the refined mask run. Once an analysis stage
consumes it, that analysis stage must record enough source identity to detect
later row-local changes.

Recommended source identity fields for downstream analysis:

- `source_refined_subject_masks_run`
- `source_refined_subject_masks_stage`
- `source_mask_labels`
- `source_mask_label_schema_id`
- `source_mask_revision` or equivalent run-level update generation
- optional per-row source revision/checksum in future row-local stale systems
- exact source component names used

If the refined mask run supports row-local edits, downstream analysis needs a
source-revision mechanism. Without that, consumers can know which run they used
but cannot reliably know whether row 500 changed after analysis was written.

## Analysis Recompute Policy

There are two acceptable recompute modes.

### New Run Recompute

Create a new downstream analysis run from the corrected source.

Use when:

- comparing methods or parameters
- producing a stable science artifact
- changing many rows
- changing any temporal smoothing or track-dependent output

This is safest and most reproducible.

### Explicit Targeted In-Place Recompute

Update affected rows inside an existing downstream analysis run, but only
through an explicit command that records the update.

Use when:

- row mapping is stable
- the output is deterministic and row-local
- the affected row set is known
- the stage records update provenance

This is acceptable for cache-like analysis surfaces and operator workflows, but
it should not be an invisible side effect of saving a mask edit.

## Registry And Status Implications

Registry/status surfaces should distinguish:

- complete and current
- complete but stale
- missing
- failed
- present but needs review

For refined mask edits, the refined run remains present. Downstream analysis
runs that consumed the prior row state become stale or source-drifted.

Mask-local cache regeneration should not create a new registry stage. It is
part of maintaining the refined mask run's internal consistency.

## Crimson And Realtime Viewing Implications

Crimson and realtime viewers should treat `masks_roi` as the canonical overlay
source.

Recommended read behavior:

- Use `mask_labels` to resolve semantic channels.
- Draw fills directly from `masks_roi`.
- Use persisted contours when available for efficient contour overlays.
- Fall back to client-side contours when component contours are missing.
- Never assume body/swim contours exist just because eye contours exist.

This keeps Crimson unblocked before body/swim contour backfill exists and
prevents persisted contour caches from becoming a hard dependency for basic
mask viewing.

## Implementation Plan

### Phase 1. Audit Surface

- Implemented: add contour-source modes to subject-shape overlay tooling:
  `mask`, `persisted`, `auto`, and `compare`.
- Implemented: default to `mask`.
- Use `compare` to verify persisted eye contours agree with recomputed mask
  contours.

### Phase 2. Generic Component Contour Utility

- Implemented: generalize the existing eye contour writer into a component-neutral
  helper.
- Implemented: preserve the packed `ptr`, `len`, `points_xy` layout.
- Implemented: require method, coordinate-space, component, and source attrs.
- Implemented: add tests for missing component channels and conservative
  write/skip behavior.
- Open: add row-local append-update tests when the manual edit-save path uses
  the helper.

### Phase 3. Body And Swim-Bladder Contour Backfill

- Implemented: add a conservative backfill command for `subject_body` and
  `swim_bladder`.
- Implemented: use declared `mask_labels`, not hardcoded channel indexes.
- Implemented: do not invent missing components.
- Implemented: do not edit mask pixels.
- Implemented: write contours only when the component mask is present and method attrs can
  be recorded.

Example dry-run:

```bash
scripts/py -m fisheye.utils.backfill_refined_subject_component_contours \
  /path/to/recording_analysis.zarr
```

Example apply:

```bash
scripts/py -m fisheye.utils.backfill_refined_subject_component_contours \
  /path/to/recording_analysis.zarr \
  --apply
```

### Phase 4. Refinement/Finalization Integration

- Implemented: finalizers and metric-refresh commands can opt into full
  body/swim contour cache refresh with `--write-component-contours`.
- Implemented: component contour cache refresh is explicit and defaults off.
- Implemented: manual-edit save paths regenerate component-local contour rows
  using append-only row updates and increment per-component `row_revision`.
- Existing: metric refresh already updates scalar mask-local metrics and QC
  reason tags.
- Implemented: component groups can carry row-local revision/timestamp/reason
  arrays so downstream analysis can detect row-local source changes.

Example finalization with contours:

```bash
scripts/py -m fisheye.refinement.finalize_subject_masks \
  /path/to/recording_analysis.zarr \
  --write-component-contours
```

Example metrics/QC refresh with contours:

```bash
scripts/py -m fisheye.utils.backfill_refined_subject_mask_metrics \
  /path/to/recording_analysis.zarr \
  --components subject_body swim_bladder \
  --write-component-contours
```

### Phase 5. Downstream Staleness And Targeted Recompute

- Mark subject-shape rows stale when source refined masks change.
- Add targeted recompute commands for row-local subject-shape outputs.
- Require explicit downstream recompute for eye angles, kinematics, bouts, and
  plots when their source rows changed.

### Phase 6. Scalar Topology QC

- Implement `components/subject_body/qc/` scalar topology metrics.
- Add equivalent lightweight QC for `swim_bladder` where useful.
- Keep full skeleton graphs out of default refined-mask runs until needed by a
  specific analysis contract.

## Validation Checklist

- A recomputed display contour from `masks_roi` matches the visible mask.
- A persisted contour drawn in `compare` mode agrees with recomputed mask
  contours for unchanged rows.
- Editing one mask row regenerates only same-row mask-local caches.
- Editing one mask row does not silently rewrite subject-shape, eye-angle,
  kinematic, bout, or plot outputs.
- Downstream validation can report stale/source-drift state after a refined
  mask edit.
- Crimson can draw all four components from `masks_roi` even when body/swim
  contours are absent.

## Open Questions

- What exact source-revision mechanism should refined subject-mask runs expose
  for row-local edits?
- Should contour append-only storage be compacted automatically, manually, or
  never unless requested?
- Should body/swim contours store only largest external contour in v1, or also
  support multiple external components?
- Which body-mask topology QC thresholds should be global versus
  recording-specific?
- Which downstream analysis runs should support targeted in-place recompute
  versus always creating a new run?
