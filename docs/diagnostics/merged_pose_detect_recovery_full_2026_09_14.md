# Full recoverable pose/detection training source cohort

This recovery restores the **surviving sampled training examples** from the
historical merged exports as one derived, source-only training Zarr per source
recording. It does not recreate deleted full-frame sensor data, the deleted
per-recording training archives, or their editable review history. New 192-pixel
head crops remain a separate, subsequent versioned materialization step.

## Inputs and identity

- Pose: `pose_cedar_shadow_filtered_gray_latest_traditional_refresh_v001`,
  10,721 three-keypoint rows from 51 source recordings. The exact refined pose
  source runs have manually approved training reviews in the registry snapshot.
- Detection: `detect_cedar_shadow_v007`, 10,942 rows from 52 source recordings,
  with manually approved source boxes in its merged export manifest.
- Every pose row joins one detection row by recording ID and sampled source
  frame, with byte-identical normalized crop boxes. The 221 remaining detector
  rows are retained as detector-only rows, including all 188 rows in one
  recording with no pose source.
- The historical copy-verification receipt, merged manifests, registry review
  snapshot, source run IDs, row indices, frame indices, and per-array content
  digests are bound into the recovered archives.

## Resulting storage contract

The collection receipt is retained at
`/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/training/merged_pose_detect_recovery_sources_v001/recovery_collection.json`.
After full validation, each archive is renamed to
`/groups/johnson/johnsonlab/jeremy/recordings/<manifest-recording-name>/zarr/<manifest-recording-name>_recovered_training.zarr`.
The relocation receipt is
`/groups/johnson/johnsonlab/jeremy/recordings/_index/merged_pose_detect_recovery_relocation_v001.json`;
it binds both paths and the original collection receipt hash.

Each source-only Zarr carries schema
`palette.training.merged_pose_detect_recovery_source.v1`, purpose `training`,
and `stage_selector_eligible=false`. At initial recovery it had no `crop_runs`;
a separately versioned 192-pixel run was added later to the 51 pose-bearing
archives (see `recovered_pose_head_192_materialization_2026_09_14.md`). Pose
and detection arrays
retain separate sampled-row axes under `recovered_sources/pose` and
`recovered_sources/detect`. Missing pose is represented by an absent pose group.
The filename deliberately includes `recovered`; these archives are derivatives
and do not claim the deleted originals' canonical identity. They have not been
registered, selected, or activated for training consumers.

## Validation and handoff

The producer validates each consolidated archive's array content digests,
sample counts, frame/box joins, detector-only rows, and selector-ineligible
state before atomic publication. Relocation validates against the collection
before moving and revalidates each destination after an inode-preserving
same-filesystem rename. The collection and relocation receipts bind the 52
per-recording destinations. Focused recovery and relocation unit tests cover
valid joins, duplicate and box-mismatch refusal, detector-only recovery,
manifest-name routing, conflicting collection refusal, and an inode-preserving
relocation. Exact executed results are recorded in the task handoff.

This is a scientific/schema/identity addition. Existing source arrays, row
coordinates, source review evidence, and source identities are preserved. The
new per-recording Zarr names, derivative schema, separate row axes, and
selector-ineligible status are intentional. Existing training loaders are not
yet adapted to consume this new recovered-source schema directly. CI and
integration remain required before any branch is described as merge-ready.

## Executed evidence and development status

On 2026-09-14, the 52-source recovery completed and wrote the collection
receipt: 51 pose-bearing recordings, 10,721 pose rows, 10,942 detector rows,
and 221 detector-only rows. All 52 archives passed the producer's consolidated
content/identity validation. Relocation subsequently revalidated all 52 at
their final recording paths and wrote the relocation receipt. A read-only
receipt audit found 52 final directories, zero remaining old archive paths,
and matching totals and source-collection SHA-256. One inspected archive has
`recovered_sources/detect/images_ds` shape `(231, 640, 640)` and
`recovered_sources/pose/roi_images` shape `(231, 512, 512)`; no `crop_runs`
group exists.

The worktree is `/tmp/palette-pose-head-crops-20260914`, branch
`agent/palette/pose-head-crops-20260914`, based on commit
`999358c3523fcd0cbf344eca669ded4e7b41e586` from `origin/main`.
Recovery, relocation, crop materialization, and export code and documents are
uncommitted in that worktree; this recovery work did not push, merge, deploy
code, register the archives, or activate a selector. The recovery and
relocation focused tests passed (9 tests total); Ruff, Python compilation,
file-size and metadata-mode ratchets, observed-metadata-literal check,
`git diff --check`, and Zarr storage census write/check passed. Required CI
checks have not run for an exact committed version of this work; the branch is
not merge-ready. Training-loader adaptation remains follow-up work; the
separate 192-pixel crop materialization was subsequently completed and is
documented in the linked handoff.
