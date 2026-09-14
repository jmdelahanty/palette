# Versioned 192-pixel crops in recovered training Zarrs

This materialization adds `crop_runs/pose_head_center_192_recovered_v001` inside
each pose-bearing `*_recovered_training.zarr` under
`/groups/johnson/johnsonlab/jeremy/recordings/`. It uses all surviving
manually reviewed pose rows from the recovered cohort, and keeps the one
detection-only archive without an invented pose crop. The run is complete but
selector-ineligible; no crop selector, registry selection, or model has been
activated.

## Scientific and coordinate contract

Recipe `recovered_pose_roi_center_192_visibility_v1` copies the central
192×192 window (`x=160:352`, `y=160:352`) from each 512×512 recovered pose ROI.
Pixels remain mono `uint8` at 1:1 resolution, with no resize, rotation, or
padding. The source's three labels are translated by `(160,160)` into the
materialized window and stored in canonical order: swim bladder, left eye,
right eye. The run records a `traditional_v1` three-point schema and declares
mono replication to RGB and scaling by 1/255 for model input.

Every pose row is retained. A landmark outside the 192 window remains in the
coordinate array with visibility `0`, while visible landmarks have visibility
`2`. Read-only preflight found eight outside landmarks across four of 10,721
rows; the other 10,717 rows contain all three landmarks. The run stores each
crop's origin **in the recovered 512 ROI**, source pose and detector row IDs,
source merged-row IDs, source sampled-frame index, and the bound normalized
detection box. Each run binds the recovered source's per-array digest index,
source merged set/run/dataset IDs, historical manifest digests, and manual
pose-review snapshot. Per-output-array SHA-256 values are recorded in the run.

The deleted original full-resolution frames and per-row full-sensor origin of
the 512 ROI are unavailable. Therefore these crops are **not asserted to be
pixel-equivalent** to the orange device kernel's full-sensor crop centered on
the detector box. The recipe and coordinate system are separately versioned
from `pose_head_fixed_192_truncated_centroid_v1`, which requires full-sensor
frames. The sensor-pixel origin is explicitly unavailable in the run metadata.
These recovered crops can be used for a controlled training experiment, but
the deployment geometry difference must be evaluated before claiming exact
orange-pipeline equivalence.

## Publication and validation

The writer validates each recovered source's consolidated content hashes,
frame and box joins, approved pose-review snapshot, and selector-ineligible
state. It materializes a run locally, validates its complete pixels, labels,
visibility, row bindings, and content hashes, then uses Palette's existing
atomic run publisher. Root metadata is consolidated last. It reopens the
published run through consolidated metadata and compares every output row to
the recovered source. A failed or interrupted cohort pass can resume by
validating an already published run before continuing. A collection receipt
under `recordings/_index/pose_head_center_192_recovered_v001.collection.json`
binds every final run to the prior relocation receipt and is written only
after all pose-bearing recordings pass.

One affected recording, `2026-01-28T19-36-18Z_arena_2`, was published first.
Its 185 rows include two rows with four outside landmarks. Full post-write
source comparison passed, and a 12-image montage at
`/tmp/recovered_pose_head_192_canary_montage.png` was visually inspected for
landmark alignment and visibility. Cohort preflight passed across all 52
archives before bulk publication: 51 pose-bearing archives, 10,721 pose rows,
eight invisible landmarks in four rows, and one detection-only archive.

Bulk publication completed on 2026-09-14. The collection receipt is
`/groups/johnson/johnsonlab/jeremy/recordings/_index/pose_head_center_192_recovered_v001.collection.json`.
All 51 pose crop runs passed the writer's post-publication exact source-row
comparison. A separate read-only receipt/metadata audit confirmed 51 run
directories, one detect-only skip, 10,721 crop rows, eight invisible points in
four rows, matching source digest indexes and output array digests, complete
run markers, consolidated root visibility, and no crop selector activation.
The focused real-Zarr publication tests passed (2 tests), including a tampered
source refusal and an affected-row visibility case. Ruff, Python compilation,
the Zarr storage census write/check, file-size ratchet, Zarr metadata-mode
ratchet, and observed-metadata-literal check passed locally.

This is a scientific/schema/identity addition. Existing recovered pixels,
labels, row identities, source digests, and root selection state remain intact;
the new versioned materialized pixels, local coordinates, visibility, and
receipt are intentional additions. Existing training consumers have not yet
been adapted to select this recovered crop schema by run ID. The worktree is
`/tmp/palette-pose-head-crops-20260914`, branch
`agent/palette/pose-head-crops-20260914`, based on `origin/main` commit
`999358c3523fcd0cbf344eca669ded4e7b41e586`. Changes are uncommitted.
Required GitHub CI has not run for an exact commit; this branch is not
merge-ready. Nothing was pushed, integrated, or activated.
