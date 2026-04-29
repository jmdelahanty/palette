# Subject Body Mask QC Design
<!-- contract-meta
version: 2
status: implemented-initial
last_verified: 2026-04-29
-->

Purpose: define mask-level quality checks for `subject_body` refined masks
before downstream centerline, tail-anchor, spline, or body-frame analyses trust
those masks.

For the broader contour/cache and row-local propagation policy, see
[refined_subject_mask_geometry_cache_and_propagation_design.md](refined_subject_mask_geometry_cache_and_propagation_design.md).

## Problem

Some recordings contain arena artifacts that can be included in the predicted
or refined `subject_body` mask. A concrete canary example is a dish scratch or
plastic cut that touches the fish body mask and creates a cross-like attached
extension. The body mask remains one connected component, so simple component
count checks pass, but downstream skeleton or centerline extraction can follow
the scratch instead of the animal.

The primary failure is the mask, not the skeleton. Skeleton, centerline, and
spline stages should fail closed when they see impossible topology, but the
first review signal should mark the `subject_body` mask itself as suspicious.

## Ownership Boundary

`refined_subject_masks_runs/<run>` should own mask-local QC:

- Does this component mask look like a plausible binary mask?
- Is the component connected, compact, and anatomically plausible enough for
  review or downstream analysis?
- Which rows should an operator inspect next?

`analysis/subject_shape_runs/<run>` should own interpreted geometry QC:

- Did body-frame construction succeed?
- Did centerline/spline fitting produce a plausible ordered body axis?
- Did tail-base projection and tail-tip estimation fail safely?

Geometry stages may consume refined-mask QC flags, but they should not be the
first place a bad body mask is discovered.

## Layered QC Model

Palette should treat body-mask quality as a layered system:

1. Raw model output remains immutable probability evidence in
   `subject_mask_runs`.
2. Refined/finalized `subject_body` masks store binary mask-local QC metrics
   and reason tags.
3. Subject-shape runs consume those masks and fail closed for geometry when
   mask-level QC is severe.
4. Temporal review tooling may later add track-aware suspicious-frame ranking,
   but temporal context should not mutate mask pixels automatically.

## Initial Mask-Level Metrics

The first implementation should add conservative metrics for
`subject_body`. These are review aids, not automatic deletion rules.

Recommended per-row metrics:

- `component_count`: number of connected foreground components.
- `largest_component_area_px`: area of the largest body component.
- `total_area_px`: total foreground area.
- `largest_component_fraction`: largest component area divided by total area.
- `bbox_width_px`, `bbox_height_px`, `bbox_aspect_ratio`.
- `convex_area_px` and `solidity`, if cheap and stable.
- `filled_area_px` and `hole_area_px`, if hole filling is used.
- `skeleton_endpoint_count`: endpoint count from a mask-topology skeleton.
- `skeleton_branchpoint_count`: count of skeleton pixels with more than two
  skeleton neighbors.
- `thin_spur_score`: optional score for long, narrow attached protrusions.
- `body_anchor_alignment_score`: optional agreement between body mask extent
  and available eye/swim-bladder anchors.

Skeleton-derived counts are allowed here as mask-topology metrics. They should
not reuse or imply that the downstream subject-shape centerline was accepted.

## Initial Reason Tags

Recommended `subject_body` reason tags:

- `ok`
- `missing_subject_body_mask`
- `fragmented_subject_body_mask`
- `small_body_mask`
- `large_body_mask`
- `low_largest_component_fraction`
- `low_solidity`
- `large_hole_area`
- `branched_body_mask`
- `excess_body_skeleton_endpoints`
- `thin_attached_artifact`
- `body_anchor_axis_mismatch`
- `requires_review`

For the dish-scratch/cross failure, the expected tags are usually
`branched_body_mask`, `excess_body_skeleton_endpoints`, and potentially
`thin_attached_artifact`.

A row may carry multiple reason tags. The current compact encoding is a
null-terminated UTF-8 string in `reason_bytes` with `|` as the delimiter, for
example:

```text
branched_body_mask|excess_body_skeleton_endpoints|thin_attached_artifact
```

Consumers should split on `|` when they need individual tags. A single selected
tag may still be used for display, but it should not erase the full tag set.

## Storage

Mask-level QC should live with the refined mask component, not inside
`analysis/subject_shape_runs`.

Recommended additive layout:

```text
refined_subject_masks_runs/<run>/
  components/
    subject_body/
      qc/
        schema_id                         attr "refined_subject_body_mask_qc"
        schema_version                    attr integer
        method                            attr
        method_version                    attr
        component_count                   (N,)
        largest_component_area_px         (N,)
        total_area_px                     (N,)
        largest_component_fraction        (N,)
        bbox_width_px                     (N,)
        bbox_height_px                    (N,)
        bbox_aspect_ratio                 (N,)
        solidity                          (N,) optional
        hole_area_px                      (N,) optional
        skeleton_endpoint_count           (N,) optional
        skeleton_branchpoint_count        (N,) optional
        thin_spur_score                   (N,) optional
        severe_qc_failure                 (N,)
        requires_review                   (N,)
        reason_bytes                      (N, width)
```

Existing component-local arrays such as `mask_present`, `area_px`,
`edit_applied`, and review status remain valid. This QC group is an additive
review and downstream-gating surface.

Initial implementation:

- helper/module: `fisheye.refinement.subject_body_mask_qc`
- method: `subject_body_mask_qc_v1`
- CLI wrapper: `scripts/backfill_subject_body_mask_qc`
- installed console entry point:
  `palette-backfill-subject-body-mask-qc`
- default skeleton severe thresholds are intentionally conservative but not
  hypersensitive: more than 4 endpoints, more than 6 branchpoint pixels, or a
  thin-spur score above 8. Milder skeleton irregularity should remain a review
  tuning problem, not an automatic geometry-blocking failure.

Example:

```bash
scripts/backfill_subject_body_mask_qc /path/to/recording_analysis.zarr \
  --refined-run <refined_subject_masks_run> \
  --chunk-size 256 \
  --json
```

## Review And Approval Semantics

QC flags should not automatically approve or reject a refined mask. They should
drive review state and navigation.

Policy:

- A severe QC failure should prevent automatic approval promotion.
- A warning-level QC issue should mark the row/component as
  `requires_review`.
- Manual operator edits may clear or override a QC flag only through an
  explicit review action that records reviewer, timestamp, and reason.
- Downstream analysis should treat severe mask-level failures as fail-closed
  inputs unless the run records an explicit reviewed override.

## Relationship To Subject Shape

Subject-shape extraction should still run its own safety checks because a mask
can pass simple QC and still fail a centerline or spline method.

Recommended behavior:

- If `subject_body/qc/severe_qc_failure[row]` is true, centerline and spline
  writers should mark geometry invalid with a reason such as
  `source_body_mask_qc_failed`.
- If `subject_body/qc/requires_review[row]` is true, geometry writers may still
  compute candidate geometry, but should propagate a warning flag and avoid
  treating the result as fully trusted.
- Geometry-specific branchpoint and endpoint metrics may be stored in
  `analysis/subject_shape_runs`, but those are method diagnostics, not the
  canonical mask-level QC record.

## Implementation Checklist

- [x] Add a deterministic subject-body mask QC helper that computes the initial
      metrics from `refined_subject_masks_runs/<run>/masks_roi`.
- [x] Persist the additive `components/subject_body/qc/` group.
- [x] Add reason-byte encoding consistent with other refined-mask review
      arrays.
- [x] Add unit tests for connected good fish masks, fragmented masks, and
      attached cross/spur masks.
- [x] Add a CLI/backfill command that can run QC on existing refined-subject
      runs without rewriting mask pixels.
- [x] Update subject-shape writers to consume severe source-mask QC failures
      once the QC group exists.
- [ ] Surface `requires_review` and reason tags in mask review and overlay
      tooling.

## Non-Goals

- Do not mutate or clean masks in this QC pass.
- Do not delete attached artifacts automatically.
- Do not use downstream centerline success as proof that the mask is good.
- Do not require temporal smoothing or tracking for the first implementation.

## Open Questions

- What thresholds should be recording-specific versus global defaults?
- Should `thin_spur_score` use skeleton branch length, local mask width, or a
  distance-transform based measure?
- Should warning and severe thresholds differ for training zarrs and full
  analysis zarrs?
- Should operator approval be able to explicitly whitelist artifact-prone arena
  frames, or should those always be fixed in the mask?
