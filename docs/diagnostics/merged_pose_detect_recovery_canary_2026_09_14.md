# Merged pose and detection recovery canary (2026-09-14)

The deleted January per-recording training archives were **not** restored. This
is a new training-only derivative of two surviving immutable merged exports:

- Pose: `pose_cedar_shadow_filtered_gray_latest_traditional_refresh_v001`
  (`merged_export_20260304T055134Z`), with 512×512 mono crops and the reviewed
  three-point traditional pose. The historical `bladder` label is canonicalized
  to `swim_bladder`.
- Detection: `detect_cedar_shadow_v007` (`merged_export_20260207T041604Z`),
  with 640×640 mono detector-training images and approved manual boxes.

Before writing, the full merged-row indexes were checked read-only. All 10,721
pose rows joined a detector row by recording ID and source sampled-frame index;
there were no duplicate keys, and all 10,721 joined boxes were byte-for-byte
equal. The detector export has 221 additional rows without pose labels. The
pose source has 51 Danio rerio recordings with manual-approved training review;
the detector source has 52 Danio rerio recordings. This recovery path uses no
Sleepyfish, Sickyfish, or Danionella sources.

The first published selector-ineligible canary is:

`/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/training/merged_pose_detect_recovery_v1/2026-01-28T19-22-28Z_arena_2_recovered_training.zarr`

It preserves **221 pose-source rows** and **231 detector-source rows**, including
**10 detector-only rows**. `recovered_sources/pose` retains the 512×512 pixels,
three-point labels, matched detector-row mapping, source sampled-frame indices,
and original merged-row IDs. `recovered_sources/detect` retains the 640×640
images, boxes, source sampled-frame indices, and merged-row IDs. Their row axes
remain distinct. `crop_runs/pose_head_center_192_from_merged_v001` contains
221 exact central 192×192 pixel crops, translated three-point labels, visibility,
and source-row and offset arrays. None of its three landmarks lies outside the
crop. No crop selector or registry selection was changed.

The root records both merged manifests' SHA-256 values, the historical `/groups`
copy-verification tree digests, the source set/run/dataset IDs, the current
registry species and pose-review snapshot, a producing-code invocation, and
content hashes for every output array. The historical copy receipt is not an
immutable-generation attestation of current source bytes; the canary's logical
array hashes bind the exact bytes actually copied. The destination was written
to a temporary sibling, consolidated after all writes, reopened through the
consolidated view, checked against every recorded array hash, and atomically
renamed to its final path.

Validation: the new row-join and crop-containment unit tests passed (3 tests)
outside the sandbox. Independent post-publication checks compared three pose
source images, three detector source images, three 192-pixel crops, and three
translated pose rows with their source arrays (12/12 checks). A 16-row
pose-crop/label montage at
`/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/training/merged_pose_detect_recovery_v1/pose_head_192_canary_montage.png`
was visually inspected and showed aligned head landmarks. Ruff, `py_compile`,
`git diff --check`, the Zarr storage census, the Zarr open-mode ratchet, the
file-size ratchet, and observed-metadata-literal checks passed locally.

This is a **scientific/schema/identity addition**. The original merged sets,
registry, and production selectors remain unchanged. The new output is a
limited derived training artifact; it does not recreate the original full
sensor frames, excluded source rows, acquisition-frame mapping, or review/edit
history. Its 192 crop recipe uses the center of the surviving 512 ROI and is
not asserted to be byte-equivalent to a crop regenerated from the deleted
original full frame. Existing training loaders do not yet consume this recovery
schema directly. The remaining 50 pose-source recordings have not been
materialized; across the full pose set four labels fall outside an unshifted
central 192 crop and need explicit exclusion or a separately named offset
recipe.

Implementation worktree: `/tmp/palette-pose-head-crops-20260914`, branch
`agent/palette/pose-head-crops-20260914`, base commit
`999358c3523fcd0cbf344eca669ded4e7b41e586` from `origin/main`. This work
is uncommitted and shares that worktree's prior pose-head implementation dirty
scope. The new recovery module, focused unit tests, this handoff, and refreshed
Zarr census artifacts are owned by the current agent. Required GitHub CI checks
have not run on this combined dirty worktree. Nothing was pushed, integrated,
registered, deployed to production, or activated; the branch is not merge-ready.
