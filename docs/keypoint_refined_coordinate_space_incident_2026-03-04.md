# Keypoint Refined Coordinate-Space Incident (2026-03-04)

## Summary

For recording `2026-01-28T19-22-28Z_arena_1_DefaultScreen`, a specific refined keypoint run showed coordinate-space inconsistency while the linked source keypoint run was clean.

- Clean source run: `keypoints_2026-02-27_22-46-42`
- Corrupted refined run: `refined_keypoints_2026-03-02_13-40-59`
- Regenerated clean refined run: `refined_keypoints_2026-03-03_23-45-34`

The issue appears to be introduced in refined-run generation or post-refine mutation, not in base keypoint inference.

## Evidence

### 1) Source run audit is clean

Command family (Crimson tooling):
- `audit_keypoint_coordinate_spaces.py` on `keypoints_2026-02-27_22-46-42`
- `analyze_bad_keypoint_row_overlap.py` on `keypoints_2026-02-27_22-46-42`

Observed:
- `Bad rows: 0 / 23287`
- Coordinate-space checks consistent (`img ~= roi + offset`)

### 2) Refined run audit fails

Command family (Crimson tooling):
- `analyze_bad_keypoint_row_overlap.py` on `refined_keypoints_2026-03-02_13-40-59`

Observed:
- `Bad rows: 573 / 23287`
- Bad frame ranges concentrated early:
  - `[0,425]`, `[427,547]`, `[549,567]`, `[569,575]`
- Bad rows concentrated in:
  - `reason=clean`
  - `quality_label=0`
  - `detection_source=0`
- Not concentrated in:
  - `manual_correction`
  - interpolated rows (`detection_source=1`)

### 3) Regenerated refined run is clean

Regeneration:
- `refine_keypoints --keypoint-run keypoints_2026-02-27_22-46-42`
- New run: `refined_keypoints_2026-03-03_23-45-34`

Re-audit:
- `Bad frames: 0`
- No coordinate-space mismatch detected.

## Likely Cause

Most likely, a subset of rows in the corrupted refined run had a coordinate-space mixup in the refined write/mutation path (ROI-local vs image-space), while the majority remained correct.

Why this is the most likely explanation:
- Source run is clean.
- Corruption exists in refined run only.
- Affected rows were tagged `clean` (not obviously associated with manual/edit tags).
- Corruption was concentrated in contiguous early row ranges, consistent with a localized write-path/chunk behavior.

## What This Is Not

- Not a base keypoint model inference issue (source run checks clean).
- Not primarily a manual correction issue (bad rows were not tagged `manual_correction`).
- Not an interpolated detection path issue (`detection_source=1` rows not affected).

## Recovery Performed

1. Re-refined from known-clean source run to produce a new refined run.
2. Audited the new refined run with Crimson coordinate-space tooling.
3. Confirmed new refined run is clean.
4. Planned/used review-status and artifact refresh steps so latest downstream views point to a clean refined run.

## Operational Guidance

When this pattern appears:
1. Audit source keypoint run and refined run separately.
2. If source is clean and refined is not, regenerate refined run from explicit source run.
3. Re-audit regenerated refined run before approval.
4. Mark known-bad refined run as rejected/superseded.
5. Re-finalize visual artifacts and refresh registry derived views.

## Follow-ups

1. Add/refine invariant checks in refined write paths:
   - `keypoints_img ~= keypoints_roi + roi_offset` (within tolerance)
   - `keypoints_roi` inside ROI bounds when finite
   - `keypoints_norm` in `[0,1]` when finite
2. Add sentinel-row debug logging for problematic row windows during refinement/mutation.
3. Consider adding a built-in Palette audit utility mirroring Crimson coordinate-space checks for immediate post-run validation.

