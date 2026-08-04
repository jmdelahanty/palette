# Track Coordinate Precision Contract Correction — 2026-08-04

Status: implemented and validated pre-promotion correction. It does not
promote a profile or modify an authority.

## Decision

Maintained track-kinematics `positions_px` and optional `positions_mm` are
exact `float32[N,2]` coordinate arrays. The generic `kinematics_samples`
Parquet projection preserves those values as Arrow `float32` columns.

This does not reduce the precision of maintained data. It removes a false
`float64` declaration introduced by the 2026-08-03 exact-schema checkpoint.
The false declaration did not describe the canonical crop source, the writer,
or any inspected full-duration Sleepyfish track authority.

Derived scientific calculations may and generally should upcast coordinates
to `float64` locally. Aggregate outputs such as means, standard deviations,
covariances, extrema, and displacement remain `float64`. The persisted source
coordinate authority and a lossless row projection of it do not widen merely
because a consumer performs higher-precision arithmetic.

## Evidence

- `crop_geometry_storage_contract_v1.md` freezes `centers_img_xy` as
  `float32[N,2]`, and `crop_schema.py` enforces that exact dtype.
- `track_kinematics.py` deliberately preserves the source coordinate dtype for
  `positions_px` and creates `positions_mm` in the same persisted dtype. It
  separately upcasts calculation kernels where needed.
- `analysis_stage_arrays.py` already declares track `positions_mm` as
  `float32`.
- A direct metadata census of all 12 offline Sleepyfish track runs found 12
  `positions_px` declarations and 12 `positions_mm` declarations, all
  `float32`; none was `float64`.
- The real exact kinematics-query binder rejected the selected maintained run
  only because the newly added binder incorrectly required `<f8`.
- After correction, that same unchanged selected full-duration run binds as
  one track with `positions_mm=<f4`; the canonical source-binding digest is
  `f702a79bdc6a1f2c1cb36fd32eb187b33dccc6601bc190720298055cedeb9328`.

## Compatibility Boundary

The historical writer test that proves dtype-preserving publication from an
explicit legacy `float64` source remains valid. Such a run is not a maintained
canonical float32 authority and must stay behind an explicit compatibility
path. It cannot satisfy the corrected exact source or flat-candidate contract
by probing or coercion.

Previously materialized selector-ineligible flat candidates whose manifests
claim float64 position authority are obsolete under this correction. They are
not rewritten or silently accepted; a candidate used in a later gate must be
rematerialized and receive new declaration, logical-hash, and manifest
digests. The v2 lineage representation remains v2 because its versioned change
is the structured-to-primitive lineage projection; it has never been promoted.

## Required Proof

- [x] Exact schema, candidate, Arrow, export, and workflow tests pass: the
      expanded focused gate passed 387/387 tests.
- [x] A `float32` maintained track source passes the exact query binder.
- [x] An otherwise matching `float64` source fails the maintained exact binder.
- [x] The selected full-duration Sleepyfish track source binds without casting
      or metadata changes.
- [x] Aggregate activity/spatial output fields remain `float64`.
- [x] No production selector, registry, profile, or source archive changes.
