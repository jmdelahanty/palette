# Five-keypoint training-source census — 2026-08-06

Status: **READ-ONLY CENSUS COMPLETE; FULL MERGE DEFERRED PENDING TWO CONTRACT FIXES**

No source Zarr, registry row, selector, training artifact, or model was changed.
The census explicitly used direct, unconsolidated metadata because the source
set includes historical editable training archives.

## Ground truth

The earlier “seven eligible datasets” count described only the narrow
Sickyfish/Sleepyfish cohort used by the first task-specific merge. It was not a
corpus-wide count.

| Source class | Sources | Usable poses | Individual landmarks |
|---|---:|---:|---:|
| Registry-approved `pose_skel_traditional_v2` | 60 | 12,523 | 62,615 |
| Selector-ineligible reviewed Batman candidate | 1 | 181 | 905 |
| **Next five-point candidate pool** | **61** | **12,704** | **63,520** |
| Approved ten-point sources, intentionally separate | 5 | 845 | 8,450 |

The five-point pool contains 12,867 total pose rows; 163 are excluded by the
persisted usable-row gates. Its source-composition SHA-256 is:

`eb3baa8aeb58b619bc17f5283fa5764e01113983250028be368f91d745143941`

That digest covers each source dataset, recording, path, refined run, usable
row count, usable-row identity digest, pixel contract, and conservative split
group.

## Compatibility findings

All 61 candidate sources pass the pose-only compatibility gate:

| Surface | Census result | Merge consequence |
|---|---|---|
| Skeleton | 61/61 are `pose_skel_traditional_v2` | Exact match |
| Label order | swim bladder, left eye, right eye, snout tip, tail tip in all 61 | Exact match |
| Coordinate domain | ROI pixels in all 61 | Exact match |
| Usable values | All finite and inside their declared ROI | Exact match |
| Duplicate usable row identities | Zero | No source-row deduplication needed |
| Historical crops | 60 sources at 512×512 `uint8` grayscale | Copy directly |
| Batman crop | One source at 348×348 `uint8` grayscale | Center-zero-pad to 512×512; never resize |
| Historical keypoints | 60 sources are `float64` | Use the existing checked `float32` conversion receipt |
| Batman keypoints | One source is canonical `float32` | Copy exactly |
| Pixel contracts | 52 raw-video grayscale, 8 Orange NV12-luma grayscale, one immutable training materialization | Keep exact source contract and output-transform provenance |

Five crop pixel contracts are resolved without a flat
`roi_pixel_contract_name`: four use their persisted crop contract document and
Batman uses the immutable training-materialization binding. These are
contracted representations, not missing pixel provenance.

## The May merged artifact

The historical artifact is:

`/nvme1/training/datasets/pose_all_registry_reviewed_v2_keypoints_20260520_v001/zarr/pose_all_registry_reviewed_v2_keypoints_20260520_v001_merged.zarr`

It contains 12,292 rows from 59 recordings. The census proved:

- all 59 recordings remain in the current approved five-point pool;
- every per-recording usable-row count is unchanged;
- all 12,292 historical rows are therefore still recoverable;
- `source_roi_idx`, `source_detect_row_index`, and
  `source_refined_row_ids` match the current sources exactly;
- the current pool adds one 231-row January recording and 181 reviewed Batman
  rows.

The old artifact must not be reused as held-out evaluation evidence. Its
`global_random` split places all 59 source datasets in both training and
validation: source overlap is 59/59, or 100%.

It also exposed an ambiguous lineage field. `source_frame_idx` means a local
sample-row index for 51 sources and an acquisition-camera frame for eight
clipped sources. The values are internally recoverable, but one array name
cannot continue to carry two domains.

## Split-unit finding

The first merged-v2 canary improved on global-random splitting by grouping on
source dataset. That is still too narrow for the full corpus:

- 52 registry sources have registered subject identities;
- many subjects occur in paired or repeated recordings;
- eight Sickyfish/Sleepyfish sources lack subject rows but form two obvious
  acquisition-time cohorts;
- Batman currently falls back to its recording identity;
- the 61 sources collapse to 29 conservative leakage groups;
- 58 recordings belong to a group containing more than one recording.

The next merge must assign complete leakage groups—not individual datasets—to
one split. Use registered subject identity first, acquisition-start cohort for
historical multi-camera collections without subjects, and recording identity
only as the final fallback. Persist the chosen group ID and its source for every
input.

## Legacy detection-lineage finding

Pose supervision and upstream detection lineage have different eligibility
requirements. Forty-six historical sources contain at least one usable pose row
whose crop cannot be joined to either a stable refined-detection row or a raw
detection row. In total, 3,252 of 12,704 usable pose rows require the explicit
sample identity fallback `(recording, acquisition frame, crop row)`.

Those rows remain valid for pose-only training: their pixels, five keypoints,
recording, frame, and crop row are exact and non-duplicated. They are not valid
for a future joint detection/keypoint merge, detection-edit propagation, or a
claim of complete detection lineage. The full merge manifest must preserve
that distinction rather than silently calling `-1` a complete lineage join.

## Required contract changes before the full merge

1. Replace ambiguous `source_frame_idx` semantics with two exact arrays:
   `source_sample_row_index` and `source_acquisition_frame_index`.
2. Add a persisted `leakage_group_id` and `leakage_group_source`, then split by
   the complete group.
3. Keep `source_crop_row_index`, `source_detect_row_index`, and
   `source_refined_row_id` separate. Permit the latter two to be absent only
   under an explicit pose-only lineage policy.
4. Version this as a successor storage/manifest contract; do not rewrite the
   historical merge or the existing v2 canary in place.
5. Rematerialize all 60 approved historical sources plus the reviewed Batman
   candidate through checked `float64`→`float32` conversion and centered
   zero-padding without resizing.

## Implementation checklist

- [x] Census all approved training-intended keypoint runs.
- [x] Inspect actual keypoint array shapes rather than relying on incomplete
  historical registry skeleton profiles.
- [x] Include the unregistered reviewed Batman candidate explicitly.
- [x] Validate skeleton, label order, coordinate domain, dtype, ROI geometry,
  finiteness, bounds, and usable-row counts.
- [x] Check exact usable-row identities for duplicates.
- [x] Compare the current sources with the May 12,292-row merge.
- [x] Quantify old split leakage and mixed frame-index semantics.
- [x] Derive conservative subject/cohort split groups.
- [x] Separate pose-only usability from complete detection lineage.
- [ ] Add dual frame-domain arrays to the merged storage schema and writer.
- [ ] Add leakage-grouped splitting and hostile overlap tests.
- [ ] Add the explicit pose-only incomplete-detection-lineage policy.
- [ ] Publish a new selector-ineligible 61-source merged candidate.
- [ ] Validate exact decoded equality, group-disjoint splits, storage plans,
  direct/consolidated metadata, and the training reader.
- [ ] Visually sample every source family and Batman zero-padding.
- [ ] Train and evaluate a candidate model using the frozen group split.
- [ ] Register or activate the training artifact only after those gates pass.

## Reproduction

The read-only utility is
`fisheye.utils.census_keypoint_training_sources`. The exact invocation is:

```bash
scripts/py -m fisheye.utils.census_keypoint_training_sources \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --skeleton-id pose_skel_traditional_v2 \
  --keypoint-count 5 \
  --historical-merge /nvme1/training/datasets/pose_all_registry_reviewed_v2_keypoints_20260520_v001/zarr/pose_all_registry_reviewed_v2_keypoints_20260520_v001_merged.zarr \
  --reviewed-artifact /groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/training/batman_training_canary_20260806_v1/2026-07-21T19-38-32Z_arena_2_Batman_reviewed_keypoints_training_v1.zarr \
  --output /tmp/palette_keypoint_training_source_census_20260806.json
```

The output is strict JSON containing the complete 61-source inventory and the
historical comparison. Writing that diagnostic file does not modify any source
archive or registry.
