# Eye-Mask Severance Phase 1 Verification Delta

> Archived execution note, 2026-07-01: this predecessor verification delta has
> been superseded by the completed severance commits `6ab7843`, `cfc0d02`,
> `73ab164`, `4a85e5d`, `b983b38`, and `edb2534`. Retained for historical
> context only.

**Date:** 2026-05-30
**Status:** implemented and locally validated
**Scope:** Palette Phase 1 re-point only. No legacy files, registry migrations, tables, or recording zarr data were deleted.

Phase 1 re-points Palette's live analysis/review consumers away from legacy `refined_eye_masks_runs` eye geometry. Eye geometry now resolves through canonical subject-mask channels:

1. `analysis/subject_shape_runs/<run>`
2. `refined_subject_masks_runs/<run>`

Legacy `refined_eye_masks_runs` compatibility producers and materializers remain in the tree because Phase 0 found Crimson/off-repo compatibility consumers. Their deletion is still Phase 3 and remains blocked until those consumers are repointed.

## Prerequisite Coverage

Phase 0 is recorded in `docs/diagnostics/eye_mask_severance_phase0_coverage_audit_2026-05-28.md`.

The only active-registry legacy-only gap was a smoke archive:

`/nvme1/recordings/smoke/2026-01-28T19-22-28Z_arena_1_DefaultScreen_geometry_smoke_analysis.zarr`

On 2026-05-30 that row was marked `datasets.status = 'inactive_smoke'` in `/nvme1/palette_registry.sqlite` and backed up first at:

`/nvme1/palette_registry_before_smoke_exclusion_20260530.sqlite`

Post-cleanup active coverage has zero dangerous legacy-only production archives.

## Verification Delta

The Phase 1 handoff was treated as a hypothesis and checked against current code before editing.

Verified:

- `src/fisheye/analysis/eye_angle_analysis.py` computes eye angles from `ellipse_params` and `ellipse_success`; `masks_roi` is not used by the angle computation path.
- `src/fisheye/shared/refined_subject_eye_geometry.py`, `src/fisheye/analysis/subject_shape_runs.py`, and `src/fisheye/refinement/refine_eye_masks.py` all derive ellipse geometry through the same `cv2.fitEllipse`-style measurement path.
- The handoff missed one stale-marker consumer: `src/fisheye/tune/keypoint_review_backend.py` also imported/called the legacy downstream eye-mask stale marker and required re-pointing.
- `src/fisheye/utils/materialize_refined_eye_masks_compat.py` intentionally remains. The Phase 1 handoff listed it in the edit set, but Phase 0 says compatibility deletion is a NO-GO while Crimson still consumes the old view.

## Code Changes

Live resolver path:

- Removed `EYE_GEOMETRY_STAGE_REFINED_EYE` from `src/fisheye/shared/eye_geometry_source.py`.
- Removed `refined_eye_run` from `resolve_eye_geometry_source(...)`.
- Removed `_build_refined_eye_source(...)` and legacy fallback lookup.
- Preserved `source_refined_eye_run` as historical lineage metadata only.

Eye-angle consumers:

- Removed `--refined-eye-run` from `src/fisheye/analysis/eye_angle_analysis.py`.
- Removed `--refined-eye-run` from `src/fisheye/visualization/visualize_eye_angle_overlays.py`.
- Existing eye-angle runs whose source stage is `refined_eye_masks_runs` now require recomputation from subject-shape or refined-subject geometry instead of silently resolving the legacy source.

Keypoint edit invalidation:

- Added `mark_downstream_subject_mask_runs_stale(...)` in `src/fisheye/shared/subject_mask_stale.py`.
- Re-pointed keypoint manual edit paths to mark downstream `refined_subject_masks_runs` stale rather than legacy eye-mask runs.
- Kept `src/fisheye/shared/keypoint_stale.py` as a legacy helper with tests, but no live Palette consumer imports remain.
- Kept `src/fisheye/utils/resolve_eye_mask_stale.py` as a deprecated operator-compatible shim that now resolves refined-subject stale markers.

Legacy operator surfaces:

- Removed `eye_mask_tuning` from default bulk tuning keys.
- Removed the `eye-mask-review` dispatcher shim.
- Narrowed `audit_refined_mask_metrics` to refined-subject masks.
- Kept explicit legacy export/materialization compatibility where it is still needed for old data or off-repo readers.

## Validation

Static compile:

```bash
PYTHONPYCACHEPREFIX=/tmp/palette-pycache scripts/py -m py_compile \
  src/fisheye/shared/eye_geometry_source.py \
  src/fisheye/analysis/eye_angle_analysis.py \
  src/fisheye/visualization/visualize_eye_angle_overlays.py \
  src/fisheye/shared/subject_mask_stale.py \
  src/fisheye/tune/keypoint_failure_review.py \
  src/fisheye/tune/keypoint_review_backend.py \
  src/fisheye/utils/patch_keypoints_from_crops.py \
  src/fisheye/utils/resolve_eye_mask_stale.py \
  src/fisheye/utils/apply_tuning_by_camera.py \
  src/fisheye/utils/audit_refined_mask_metrics.py \
  src/fisheye/utils/export_eye_mask_training_zarr.py \
  src/fisheye/tune/dispatcher.py \
  tests/unit/fisheye/test_eye_geometry_source.py \
  tests/unit/fisheye/test_eye_angle_lineage_attrs.py \
  tests/unit/fisheye/test_keypoint_stale_marking.py \
  tests/unit/fisheye/test_keypoint_review_backend.py \
  tests/unit/fisheye/test_audit_refined_mask_metrics.py \
  tests/unit/fisheye/test_resolve_eye_mask_stale.py
```

Focused tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/palette-pycache scripts/py -m pytest -p no:cacheprovider \
  tests/unit/fisheye/test_eye_geometry_source.py \
  tests/unit/fisheye/test_eye_angle_lineage_attrs.py \
  tests/unit/fisheye/test_keypoint_stale_marking.py \
  tests/unit/fisheye/test_keypoint_review_backend.py \
  tests/unit/fisheye/test_audit_refined_mask_metrics.py \
  tests/unit/fisheye/test_resolve_eye_mask_stale.py \
  tests/unit/fisheye/test_apply_tuning_by_camera.py \
  tests/unit/fisheye/test_subject_mask_tuner.py \
  -q
```

Observed result:

`98 passed`

Contract grep checks:

- No code or unit-test import of `EYE_GEOMETRY_STAGE_REFINED_EYE` remains.
- `resolve_eye_geometry_source(...)` no longer accepts a `refined_eye_run` argument.
- No live Palette consumer imports `mark_downstream_eye_mask_runs_stale` or `resolve_downstream_eye_mask_runs_stale`; remaining references are the legacy helper definition and its tests.

## Phase 2 / Phase 3 Handoff

Ready for a Phase 2 delete-now review:

- Live eye-angle analysis is severed from `refined_eye_masks_runs`.
- Live keypoint manual edit invalidation is severed from legacy eye-mask stale markers.
- `eye-mask-review` is no longer reachable through the tune dispatcher.

Still blocked:

- Do not delete `materialize_refined_eye_masks_compat.py`, `refined_eye_masks_compat.py`, old eye-mask registry tables, or compatibility materializers until Crimson/off-repo consumers are confirmed repointed.
- Do not remove legacy `eye_masks_runs` / `refined_eye_masks_runs` training/export compatibility until the training-data migration plan explicitly says those views are obsolete.
