# Stage-Array Enforcement Census, 2026-07-02

<!-- contract-meta
status: active
created: 2026-07-02
owner: jeremy
related: docs/provenance_enforcement_roadmap.md,
         docs/diagnostics/codebase_review_2026-07-01.md
-->

## Purpose

This census supports Slice 1 of `docs/provenance_enforcement_roadmap.md`: promote
stage-array validation from shadow telemetry to hard completion-time enforcement only
for stages whose current writers already satisfy their declared `StageSpec`.

This is intentionally evidence-driven. A non-compliant stage is a latent contract bug,
not a reason to weaken its spec or silently broaden enforcement.

## Method

- Enumerated every `StageSpec` in `src/fisheye/shared/zarr/stage_arrays.py`.
- Compared those specs to finalized stages that flow through
  `src/fisheye/registry/stage_complete.py`.
- Excluded deprecated `eye_masks` and `refined_eye_masks`.
- Inspected the active registry read-only:
  `/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite`.
- Ran current `validate_run(...)` in shadow mode against available real stores under
  `/nvme1` and `/groups`. Transient pytest/temp rows outside those roots were ignored.
- Treated missing run groups in otherwise registered rows as registry hygiene/stale-row
  evidence, not as stage-array writer compliance evidence.

No real zarr stores were modified.

## Summary

| Classification | Count | Stages |
|---|---:|---|
| Already enforced | 1 | `detect_quality` |
| Promote now | 5 | `detect`, `crop`, `refined_keypoints`, `arena_assignment`, `tracking` |
| Do not promote: current data does not comply | 4 | `refined_detect`, `keypoints`, `subject_masks`, `refined_subject_masks` |
| Deprecated, never enforce | 2 | `eye_masks`, `refined_eye_masks` |
| Not finalized through `stage_complete` in this slice | 13 | `raw_video`, `stimulus`, `detection_profile`, `eye_mask_profile`, `keypoint_profile`, `subject_shape`, `tail_posture_view`, `bout_classification`, `tail_kinematics`, `track_kinematics`, `eye_angle`, `bout_kinematics`, `background` |

## Finalized Stage Census

| StageSpec | Completion parent(s) | Present run groups checked | Current validation result | Classification | Decision |
|---|---|---:|---|---|---|
| `detect` | `detect_runs` | 185 | 185 valid, 0 invalid | complies | Promote |
| `detect_quality` | nested `detect_runs/<run>/quality_reports` | 52 present groups; 2 stale missing groups | 52 valid, 0 invalid | complies | Already enforced |
| `refined_detect` | `refined_detect_runs`, `refined_runs` | 201 recording rows plus 2 non-recording scratch rows | recording rows validated except stale missing groups; 2 scratch rows invalid | does-not-comply | Do not promote |
| `crop` | `crop_runs` | 193 present groups; 3 stale missing groups | 193 valid, 0 invalid | complies | Promote |
| `keypoints` | `keypoints_runs` | sampled 120 of 176 real rows | 117 valid, 2 invalid, 1 stale missing group in sample | does-not-comply | Do not promote |
| `refined_keypoints` | `refined_keypoints_runs`, `keypoints_refined_runs` | 170 present groups; 1 stale missing group | 170 valid, 0 invalid | complies | Promote |
| `subject_masks` | `subject_mask_runs` | 81 | 29 valid, 52 invalid | does-not-comply | Do not promote |
| `refined_subject_masks` | `refined_subject_masks_runs` | 81 | 29 valid, 52 invalid | does-not-comply | Do not promote |
| `arena_assignment` | `arena_assignment_runs` | 84 | 84 valid, 0 invalid | complies | Promote |
| `tracking` | `tracking_runs` | 84 | 84 valid, 0 invalid | complies | Promote |

## Latent Contract Bugs Found

### `keypoints`

Some real RedScare training keypoint runs fail the current `StageSpec` because
`n_keypoints` has the frame-axis length of the source training dataset rather than the
run's `frame_counts` axis.

Examples:

- `/groups/johnson/johnsonlab/jeremy/training_data/red_scare/2026-06-23T20-56-03Z_arena_3_RedScare_training.zarr`,
  run `keypoints_training_review_red_scare_training_review_20260626_01`:
  `leading dimension mismatch for 'n_frames' (n_keypoints has 137809, expected 138505)`.
- `/groups/johnson/johnsonlab/jeremy/training_data/red_scare/2026-06-23T17-16-51Z_arena_3_RedScare_training.zarr`,
  same run family: `n_keypoints has 138007, expected 138704`.

Recommended follow-up: inspect the training-review keypoint writer and decide whether
`n_keypoints` should be per-frame for the run's frame domain, or whether the spec needs
a separate source-frame-domain array. Do not promote until the writer/spec contract is
resolved.

### `subject_masks`

Older real subject-mask runs fail because required lineage arrays are absent.

Observed error:

- `subject_masks: missing required array 'source_crop_row_ids'`.

Representative run family:

- `/nvme1/recordings/2026-01-28.../subject_mask_runs/subject_masks_unet_registry_batch_20260504`.

Recommended follow-up: either backfill/stamp `source_crop_row_ids` for the legacy
subject-mask writer path or establish a new strict epoch that separates historical
non-compliant runs from current compliant runs. Do not promote while the active registry
contains these accepted non-compliant runs.

### `refined_subject_masks`

Older real refined-subject-mask runs fail for the same lineage gap.

Observed error:

- `refined_subject_masks: missing required array 'source_crop_row_ids'`.

Representative run family:

- `/nvme1/recordings/2026-01-28.../refined_subject_masks_runs/refined_subject_masks_smart_finalizer_batch_20260504`.

Recommended follow-up: align the refined-subject-mask finalizer output with the
lineage contract, or split historical compatibility from current strict completion.

### `refined_detect`

Most recording refined-detection runs validate, but the active registry also contains
non-recording scratch rows marked `ok` under `/nvme1/dan*.zarr` whose run groups are
missing required subgroups.

Observed errors:

- `refined_detect: missing required subgroup 'source_detections'`.
- `refined_detect: missing required subgroup 'instances'`.

Recommended follow-up: remove or reclassify scratch rows from finalized production
status before promoting `refined_detect`, or enforce only after the registry no longer
contains accepted malformed refined-detection completions.

## Registry Hygiene Notes

Read-only validation also found stale registry rows whose registered run group no
longer exists in the store. These were not counted as writer contract failures, but they
are worth cleaning separately:

- `crop`: 3 stale missing run groups.
- `refined_keypoints`: 1 stale missing run group.
- `detect_quality`: 2 stale missing run groups.
- `refined_detect`: 2 stale recording run groups.

These rows should be reconciled independently from stage-array enforcement.

## Promotion Set

Promote these canonical stage names into
`_ENFORCE_STAGE_ARRAY_VALIDATION_FOR`:

- `detect`
- `detect_quality` (already present)
- `crop`
- `refined_keypoints`
- `arena_assignment`
- `tracking`

Do not promote aliases directly. The completion hook already maps aliases such as
`tracks` to canonical specs before enforcement.
