# Batman keypoint recording allocation — 2026-08-07

Status: **frozen allocation; pigmentation classification applied and verified**

Machine-readable authority:
`docs/diagnostics/batman_keypoint_recording_allocation_20260807.json`, with
the exact recording/dataset/subject census in the adjacent
`batman_keypoint_recording_allocation_20260807.csv`.

## Allocation

The 36 analysis recordings form nine synchronized four-camera acquisition
sessions. Whole sessions, not frames or individual cameras, are assigned to a
single role. This prevents the same acquisition session from contributing to
both training and evaluation while preserving all-camera coverage.

| Session start (UTC) | Role | Cam 2010093 | Cam 2010094 | Cam 2010095 | Cam 2010096 |
|---|---|---|---|---|---|
| 2026-07-21 19:38:32 | train | arena 1 | arena 2 (181 reviewed rows already used) | arena 3 | arena 4 |
| 2026-07-21 20:12:57 | train | arena 1 | arena 2 | arena 3 | arena 4 |
| 2026-07-21 20:56:02 | train | arena 1 | arena 2 | arena 3 | arena 4 |
| 2026-07-21 23:29:15 | development | arena 1 | arena 2 | arena 3 | arena 4 |
| 2026-07-22 00:06:17/18 | train | arena 1 | arena 2 | arena 3 | arena 4 |
| 2026-07-22 00:40:31 | train | arena 1 | arena 2 | arena 3 | arena 4 |
| 2026-07-22 01:08:24 | train | arena 1 | arena 2 | arena 3 | arena 4 |
| 2026-07-22 15:44:40 | train | arena 1 | arena 2 | arena 3 | arena 4 |
| 2026-07-22 16:15:04 | sealed test | arena 1 | arena 2 | arena 3 | arena 4 |

Totals: 28 training recordings, four development recordings, and four sealed
test recordings. Registry subject IDs are unique across all 36 recordings.

## Intended workflow

1. Sample diverse pose labels from the 28 training recordings without allowing
   one recording to dominate.
2. Train the first multi-recording candidate.
3. Run full-video inference on the development session and correct selected
   failures in Crimson.
4. Publish accepted development corrections as a new immutable training-data
   version and train the successor candidate.
5. Compare the historical model and both candidates once on the sealed-test
   session. Sealed-test corrections must not enter training before that result
   is frozen.

Negative fish-absent frames remain detection evidence and do not create pose
crops. Pose correction targets are instances that contain a fish but have
missing or inaccurate landmarks.

## Biological metadata finding

The source metadata reports 36 unique *Danio rerio* subjects with the exact
line/genotype label `AB [AB IC] SEPT25`: 28 at 7 DPF and eight at 8 DPF. The
project operator confirmed that every Batman subject had normal, typical
wild-type pigmentation. Accordingly, all 36 observations are classified as:

- canonical strain `AB`;
- `pigmentation_phenotype=wild_type_pigmented`;
- normal melanophore, xanthophore, and iridophore status;
- wild-type pigment pattern; and
- normal optical transparency.

The value origin is `subject_observed`. AB also carries the same strain-level
expectations, but those defaults do not replace the observations. The complete
source label remains preserved, while `[AB IC] SEPT25` is explicitly
uninterpreted pending guidance from Aquatics; no stock or colony field is
published. The normalized resolution and override contract is
`docs/recording_subject_trait_contract.md`.

The shared registry backfill completed at `2026-08-07T18:34:58Z` under schema
version 65. Independent readback found one exact label mapping, six AB strain
expectations, 216 recording-subject observations, and 36 resolved
`subject_observed` values for each pigmentation axis.

## Sampling guidance

Prefer dispersed frames that cover posture, crop-edge placement, lighting,
contrast, bubbles, and partial/failed poses. Initial targets of roughly 50–100
accepted poses per training recording and 150–300 per development/test
recording are planning ranges, not contract constants. Exact frames and
instances must be persisted in each immutable training manifest.

## Next materialization batches

The next training-source batch is the synchronized four-camera session
`2026-07-21T20-12-57Z`. Create one independently reviewable source training
Zarr for each arena, preserving its recording/dataset/subject identity and
explicit negative detection frames. Aim initially for 50–100 accepted pose
instances per recording; do not merge these sources until review is complete.

The first Batman-domain evaluation batch is the synchronized four-camera
development session `2026-07-21T23-29-15Z`. Its source training Zarrs may be
created and labeled, but their accepted labels must remain outside training
merges until the historical and candidate models have been evaluated on the
same frozen development selection. The `2026-07-22T16-15-04Z` sealed-test
session remains untouched.

For every new immutable source or merged artifact:

- build and review through unconsolidated metadata while mutable;
- retain exact instance, crop, recording, camera, subject, and split lineage;
- persist the exact ordered pose labels and skeleton edges;
- finalize payload, review state, and provenance before consolidation; and
- consolidate metadata only as the final publication visibility step.

## Atomic source-base publication checkpoint

Before submitting the four-source batch, Palette retired direct sampled-
training destination construction from maintained operator workflows. The
PyNvVideoCodec importer remains an internal scratch constructor, but its CLI
fails closed. `import_recordings_training` now always requires bounded local
`--scratch-root` and routes through `publish_sampled_training_base`; overwrite
in place is rejected. The LSF submitter supplies a job-specific `$TMPDIR`,
validates the local artifact, checks the copied hidden sibling, and exposes the
destination only by final rename. Local-host and video-only intake use the same
publisher rather than retaining a non-atomic compatibility path.

The no-write four-recording plan resolved 139,385 source frames per camera,
`frame_step=696`, and exactly 200 sampled rows for cameras 2010093 through
2010096. At this checkpoint no training Zarr, registry row, selector, or LSF job
had been created.
