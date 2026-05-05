# Subject Mask Batch QC Addendum

Generated after `scripts/run_subject_mask_batch_pipeline --apply --run-label batch_20260504`.

## Execution Summary

- Batch report: `/tmp/subject_mask_batch_apply.json`
- Markdown report: `docs/diagnostics/subject_mask_batch_2026-05-04.md`
- Planned archives: 47
- Inference status: 45 `ok`, 2 `not_requested` (existing raw subject-mask runs)
- Finalization status: 47 `ok`
- Validation status: 47 `ok`
- Strict metadata JSON check: 27,290 `zarr.json` files checked, 0 parse failures
- Refined-run provenance check: 47/47 have runtime, Git/platform/environment, command, and Dask parameter provenance

The batch succeeded operationally. The table below is a row-level QC review list,
not a command-failure list.

## Gate Refresh Follow-up

After the drilldown below, the refined subject-mask finalizer was rerun over the
same 47 archives with the stricter eye-assignment keypoint gate. The refresh
overwrote `refined_subject_masks_smart_finalizer_batch_20260504` only; it reused
the existing raw subject-mask runs and did not rerun U-Net inference.

- Refresh report: `/tmp/subject_mask_gate_refresh_apply.json`
- Markdown report: `docs/diagnostics/subject_mask_gate_refresh_2026-05-04.md`
- Planned archives: 47
- Inference status: 47 `not_requested`
- Finalization status: 47 `ok`
- Validation status: 47 `ok`
- Errors: 0
- Refined-run metadata check: 47/47 use
  `assignment_keypoint_success_dataset="usable_keypoints"` at the run attrs and
  in `eyes_union_assignment_summary.keypoint_success_dataset`
- Refined-run provenance check: 47/47 include Dask/runtime finalizer parameters
  in `provenance.parameters`
- Strict metadata JSON check across refreshed archives: 0 parse failures
- Worst-outlier coverage diagnostic after refresh:
  `zarr_pass=1 zarr_fail=0 errors=0`

Current post-refresh QC hotspots by eye review / keypoint gate:

| Zarr | left eye review | right eye review | subject body review | swim bladder review | failed keypoint rows |
|---|---:|---:|---:|---:|---:|
| `2026-01-28T22-22-57Z_arena_3_Feeding_analysis.zarr` | 13,562 | 13,563 | 3,915 | 9,875 | 12,419 |
| `2026-01-28T22-15-04Z_arena_3_DefaultScreen_analysis.zarr` | 10,772 | 10,769 | 6,604 | 925 | 10,506 |
| `2026-01-28T19-36-18Z_arena_2_Feeding_analysis.zarr` | 6,822 | 6,894 | 14,603 | 7,231 | 2 |
| `2026-01-28T22-50-39Z_arena_4_Feeding_analysis.zarr` | 5,857 | 5,838 | 3,224 | 45 | 1,198 |
| `2026-01-28T22-50-39Z_arena_2_Feeding_analysis.zarr` | 3,049 | 3,051 | 5,474 | 666 | 3 |

Current post-refresh QC hotspots by subject-body review:

| Zarr | subject body review | left eye review | right eye review | swim bladder review | failed keypoint rows |
|---|---:|---:|---:|---:|---:|
| `2026-01-28T22-42-59Z_arena_4_DefaultScreen_analysis.zarr` | 22,020 | 392 | 403 | 356 | 162 |
| `2026-01-28T22-50-39Z_arena_1_Feeding_analysis.zarr` | 16,191 | 78 | 82 | 59 | 5 |
| `2026-01-28T19-36-18Z_arena_2_Feeding_analysis.zarr` | 14,603 | 6,822 | 6,894 | 7,231 | 2 |
| `2026-01-28T21-56-23Z_arena_2_Feeding_analysis.zarr` | 12,489 | 2,706 | 2,736 | 0 | 1,051 |
| `2026-01-28T23-07-24Z_arena_2_DefaultScreen_analysis.zarr` | 9,918 | 87 | 85 | 305 | 4 |

The refresh does not make these rows good data. It changes the persisted
semantics so rows with invalid keypoint geometry/confidence are explicitly
recorded as keypoint-gated failures rather than being counted as valid
anatomical anchors.

## Review Thresholds

Archives are listed when any of the following are true:

- `eye_failed > 100`
- `max(left_eye_needs_review, right_eye_needs_review) > 10%`
- `subject_body_needs_review > 25%`
- `swim_bladder_needs_review > 10%`

## QC Review Candidates

| Zarr | rows | eye failed | max eye review | body review | swim review | reason |
|---|---:|---:|---:|---:|---:|---|
| `2026-01-28T19-36-18Z_arena_1_Feeding_analysis.zarr` | 18647 | 0 | 2355 | 4295 | 358 | eye_review=2355 (12.6%) |
| `2026-01-28T19-36-18Z_arena_2_Feeding_analysis.zarr` | 18647 | 50 | 6892 | 14603 | 7231 | eye_review=6892 (37.0%); body_review=14603 (78.3%); swim_review=7231 (38.8%) |
| `2026-01-28T19-36-18Z_arena_3_Feeding_analysis.zarr` | 18647 | 38 | 481 | 6911 | 14 | body_review=6911 (37.1%) |
| `2026-01-28T20-41-59Z_arena_2_DefaultScreen_analysis.zarr` | 23028 | 198 | 236 | 4362 | 326 | eye_failed=198 |
| `2026-01-28T20-51-00Z_arena_2_Feeding_analysis.zarr` | 19208 | 0 | 9 | 6104 | 0 | body_review=6104 (31.8%) |
| `2026-01-28T21-18-51Z_arena_1_DefaultScreen_analysis.zarr` | 23047 | 443 | 746 | 5754 | 16 | eye_failed=443 |
| `2026-01-28T21-27-20Z_arena_2_Feeding_analysis.zarr` | 18554 | 2374 | 2380 | 7086 | 46 | eye_failed=2374; eye_review=2380 (12.8%); body_review=7086 (38.2%) |
| `2026-01-28T21-47-47Z_arena_3_DefaultScreen_analysis.zarr` | 23218 | 4 | 101 | 6981 | 4 | body_review=6981 (30.1%) |
| `2026-01-28T21-56-23Z_arena_2_Feeding_analysis.zarr` | 18870 | 524 | 2212 | 12489 | 0 | eye_failed=524; eye_review=2212 (11.7%); body_review=12489 (66.2%) |
| `2026-01-28T21-56-23Z_arena_3_Feeding_analysis.zarr` | 18870 | 0 | 9 | 7102 | 13 | body_review=7102 (37.6%) |
| `2026-01-28T22-15-03Z_arena_1_DefaultScreen_analysis.zarr` | 22876 | 3 | 334 | 7725 | 272 | body_review=7725 (33.8%) |
| `2026-01-28T22-15-03Z_arena_2_DefaultScreen_analysis.zarr` | 22877 | 50 | 670 | 9010 | 2 | body_review=9010 (39.4%) |
| `2026-01-28T22-15-04Z_arena_3_DefaultScreen_analysis.zarr` | 22877 | 98 | 9566 | 6604 | 925 | eye_review=9566 (41.8%); body_review=6604 (28.9%) |
| `2026-01-28T22-22-57Z_arena_2_Feeding_analysis.zarr` | 18963 | 81 | 275 | 5911 | 40 | body_review=5911 (31.2%) |
| `2026-01-28T22-22-57Z_arena_3_Feeding_analysis.zarr` | 18963 | 12330 | 13512 | 3915 | 9875 | eye_failed=12330; eye_review=13512 (71.3%); swim_review=9875 (52.1%) |
| `2026-01-28T22-42-59Z_arena_1_DefaultScreen_analysis.zarr` | 23204 | 1295 | 1415 | 6158 | 1455 | eye_failed=1295; body_review=6158 (26.5%) |
| `2026-01-28T22-42-59Z_arena_3_DefaultScreen_analysis.zarr` | 23204 | 0 | 359 | 6145 | 488 | body_review=6145 (26.5%) |
| `2026-01-28T22-42-59Z_arena_4_DefaultScreen_analysis.zarr` | 23204 | 67 | 359 | 22020 | 356 | body_review=22020 (94.9%) |
| `2026-01-28T22-50-39Z_arena_1_Feeding_analysis.zarr` | 21160 | 2 | 79 | 16191 | 59 | body_review=16191 (76.5%) |
| `2026-01-28T22-50-39Z_arena_2_Feeding_analysis.zarr` | 21160 | 6 | 3050 | 5474 | 666 | eye_review=3050 (14.4%); body_review=5474 (25.9%) |
| `2026-01-28T22-50-39Z_arena_4_Feeding_analysis.zarr` | 21160 | 780 | 5494 | 3224 | 45 | eye_failed=780; eye_review=5494 (26.0%) |
| `2026-01-28T23-07-24Z_arena_2_DefaultScreen_analysis.zarr` | 23190 | 3 | 86 | 9918 | 305 | body_review=9918 (42.8%) |
| `2026-01-28T23-07-24Z_arena_3_DefaultScreen_analysis.zarr` | 23190 | 0 | 125 | 6322 | 654 | body_review=6322 (27.3%) |
| `2026-01-28T23-15-10Z_arena_3_Feeding_analysis.zarr` | 19235 | 0 | 35 | 6556 | 918 | body_review=6556 (34.1%) |

## Worst-Outlier Drilldown

Archive:
`/nvme1/recordings/2026-01-28T22-22-57Z_arena_3_Feeding/zarr/2026-01-28T22-22-57Z_arena_3_Feeding_analysis.zarr`

Runs inspected:

- Raw subject-mask run: `subject_mask_runs/subject_masks_unet_registry_batch_20260504`
- Refined subject-mask run: `refined_subject_masks_runs/refined_subject_masks_smart_finalizer_batch_20260504`
- Keypoint run: `refined_keypoints_runs/refined_keypoints_2026-03-02_13-45-11`

Findings:

- Raw `eyes_union` masks are present in 18,652/18,963 rows, so this is not primarily a raw U-Net eye-channel outage.
- Refined LR eyes are present in 6,633/18,963 rows.
- 11,952 rows have `refined_success=True` and raw `eyes_union=True`, but fail LR splitting because the keypoint row is marked `low_confidence|geometry_issue`.
- Those 11,952 rows have `usable_keypoints=False`, `geometry_valid=False`, and `confidence_valid=False`, while `heading_usable=True`.
- Sampled failing rows show raw eye-union masks near ROI y ~250, but the assignment keypoints near ROI y ~360 and y ~493. The LR splitter is fail-closing correctly; the issue is that `refined_success` is too broad for anatomical eye assignment.

Code action taken:

- Updated keypoint success selection for subject-mask eye assignment and coverage diagnostics to prefer `usable_keypoints` over broader success flags.
- Existing broad flags (`detection_success`, `refined_success`, `source_success`) remain as legacy fallbacks when `usable_keypoints` is absent.

Validation:

- Before the gate change, `check_subject_mask_keypoint_coverage` failed this archive because it treated `refined_success` rows as keypoint-valid.
- After the gate change, the same diagnostic passes with `zarr_pass=1`, because geometry/confidence-invalid keypoint rows are no longer counted as valid anchors for LR eye splitting.

Interpretation:

This archive still has a real keypoint-quality problem over long frame ranges. The corrected diagnostic now classifies that as invalid keypoint support rather than as a subject-mask coverage failure. Downstream eye-geometry or LR eye-mask analyses should not use those rows without repairing/re-running keypoints.
