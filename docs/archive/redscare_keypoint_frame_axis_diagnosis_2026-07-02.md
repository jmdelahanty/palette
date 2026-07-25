# RedScare keypoint frame-axis mismatch diagnosis

<!-- contract-meta
status: diagnostic
created: 2026-07-02
owner: jeremy
related: docs/diagnostics/stage_array_enforcement_census_2026-07-02.md,
         docs/identity_lineage_staleness_review.md
-->

## Summary

The RedScare keypoint mismatch found by the stage-array enforcement census is low severity for the current SAM3-teacher training surface. The short array is the per-frame `n_keypoints` summary/count array only. The actual keypoint labels and their row-level lineage arrays are internally aligned in the affected runs:

- `keypoints_roi`, `keypoints_img`, `keypoints_norm`, `keypoint_confidences`, `frame_indices`, `detection_indices`, `detection_success`, `heading`, `heading_finite`, and `heading_usable` all have 200 rows in both affected runs.
- `frame_counts` is the authoritative frame-axis array in these runs, has the expected full sparse frame domain, sums to 200 rows, and exactly matches `np.bincount(frame_indices, minlength=len(frame_counts))`.
- `n_rois` matches `frame_counts` length in both affected runs.
- `n_keypoints` is the only required frame-axis array that is short.

Conclusion: RedScare keypoint training data is safe to use as-is for SAM3 teacher training if consumers use the row-axis keypoint arrays and lineage (`keypoints_roi`, `frame_indices`, `detection_success`, crop lineage) and do not depend on the stale `n_keypoints` summary array. Do not promote hard stage-array enforcement for `keypoints` until the writer/backfill fix lands.

## Investigation Method

The read-only diagnostic script is committed as `scripts/diagnose_redscare_keypoint_frame_axis.py`. It opens zarrs with `mode="r"` and `use_consolidated=False`; it does not write to stores.

Commands run:

```bash
scripts/py -m py_compile scripts/diagnose_redscare_keypoint_frame_axis.py
```

```bash
PYTHONPATH=/tmp/palette-redscare-keypoint-frame-axis-diagnosis/src \
  scripts/py scripts/diagnose_redscare_keypoint_frame_axis.py \
  --output /tmp/redscare_keypoint_frame_axis_report.json
```

Focused check of the two census failures:

```bash
PYTHONPATH=/tmp/palette-redscare-keypoint-frame-axis-diagnosis/src \
  scripts/py scripts/diagnose_redscare_keypoint_frame_axis.py \
  /groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T17-16-51Z_arena_3_RedScare/zarr/2026-06-23T17-16-51Z_arena_3_RedScare_training.zarr \
  /groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T20-56-03Z_arena_3_RedScare/zarr/2026-06-23T20-56-03Z_arena_3_RedScare_training.zarr \
  --output /tmp/redscare_keypoint_frame_axis_affected_report.json
```

The family-wide pass inspected 28 RedScare training zarrs and 31 keypoint runs discovered under `/groups/johnson/johnsonlab/jeremy/recordings`.

## Severity Fork

The mismatch is summary-only, not label-reaching.

| zarr | keypoint run | `frame_counts` | `n_keypoints` | gap | `frame_counts` sum | `keypoints_roi` | row arrays aligned? | rows beyond `n_keypoints` axis | successful rows beyond axis |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| `2026-06-23T17-16-51Z_arena_3_RedScare_training.zarr` | `keypoints_training_review_red_scare_training_review_20260626_01` | 138,704 | 138,007 | 697 | 200 | `(200, 5, 2)` | yes | 1 | 0 |
| `2026-06-23T20-56-03Z_arena_3_RedScare_training.zarr` | `keypoints_training_review_red_scare_training_review_20260626_01` | 138,505 | 137,809 | 696 | 200 | `(200, 5, 2)` | yes | 1 | 0 |

Additional evidence:

- In both runs, `frame_counts_matches_bincount_frame_indices` is true.
- In both runs, `n_rois` has the same shape as `frame_counts`.
- The single row whose frame index is beyond the short `n_keypoints` axis is not a successful keypoint detection in either affected run. This explains why `n_keypoints` can be short without omitting a positive success count.
- The crop source run selected all 200 source samples in both affected zarrs: `source_sample_count=200`, `selected_sample_count=200`, and all recorded rejection counters are zero.

This rules out a row-label shift in the arrays that training would normally consume.

## Scope

Family-wide keypoint run summary:

| result | keypoint runs |
| --- | ---: |
| `n_keypoints` length matches `frame_counts` | 28 |
| `n_keypoints` short by 696 | 1 |
| `n_keypoints` short by 697 | 1 |
| `n_keypoints` absent | 1 |

The one run with absent `n_keypoints` is `keypoints_red_scare_may_pose_training_20260625_02` in `2026-06-23T16-01-09Z_arena_1_RedScare_training.zarr`. Its row arrays still have 200 rows and `frame_counts` matches `frame_indices`; it appears to be an older run that predates this summary array rather than the specific short-axis failure.

The two short-axis failures are isolated to the current YOLO-pose review run `keypoints_training_review_red_scare_training_review_20260626_01` for arena 3 recordings at `2026-06-23T17-16-51Z` and `2026-06-23T20-56-03Z`.

## Root Cause

Root cause is a writer-side frame-axis sizing bug in the `n_keypoints` summary array for the YOLO keypoint path. It is a concrete frame-domain boundary issue, but the data evidence does not support actual keypoint label corruption.

Why this is not spec overreach:

- `KEYPOINTS_SPEC` declares `frame_counts`, optional legacy alias `n_rois`, and `n_keypoints` on the same `("n_frames",)` axis, while keypoint coordinate arrays are on `("n_rois", "n_keypoints", 2)`.
- Most sampled RedScare keypoint runs already satisfy this contract: 28 of 31 keypoint runs have `n_keypoints` matching `frame_counts`.
- The semantic value stored in these YOLO runs is a per-frame success count (`0/1`), so a frame-axis summary array is legitimate.

Why this is not crop-video dropped-frame corruption:

- The affected crop runs report `source_sample_count=200`, `selected_sample_count=200`, and zero rejected samples.
- `source_crop_video_frame_indices`, `source_training_row_indices`, `source_crop_meta_row_indices`, and crop `frame_indices` all have 200 rows with 200 unique selected rows.
- The gap equals one sparse sampling interval (`frame_step=697` or `696`), not a count of rejected crop-video rows.

Most likely code seam:

- `src/fisheye/detection/detect_keypoints_yolo.py` copies or creates `frame_counts`/`n_rois` from the crop lineage, but later builds `n_keypoints` as `success_counts` using an independently resolved `total_frames`.
- The observed failing runs have `method="yolo_pose"` and `n_keypoints` values `[0, 1]`, matching a success-count summary.
- `src/fisheye/utils/export_acquisition_crop_pose_training_zarr.py` is unlikely to be the source of these two arrays because it writes `n_keypoints` as landmark-count values at selected frames (`0`/`5` for this schema), not `0`/`1`.
- `src/fisheye/utils/patch_keypoints_from_crops.py` already uses the safer rule: derive the summary length from `frame_counts` when present.

The exact historical reason `total_frames` resolved one sample interval short in these two stores is not recoverable from current attrs alone, but the bad shape is restricted to the summary-count array and matches the writer seam where `n_keypoints` can be sized independently of copied `frame_counts`.

## Recommended Fix Path

1. Fix the YOLO keypoint writer so `n_keypoints` is sized from the authoritative frame-axis array already in the run:
   - If `frame_counts_total` exists, use `len(frame_counts_total)` for `success_counts`.
   - Assert `n_keypoints.shape[0] == frame_counts_total.shape[0]` before completion.
   - Only fall back to resolved `total_frames` when no copied/created frame-count axis exists.
2. Add a focused test for acquisition-crop keypoint runs where `frame_counts` is copied from the crop lineage and `n_keypoints` must match it exactly.
3. Backfill only the `n_keypoints` summary array in the two affected runs with:

   ```python
   np.bincount(frame_indices[detection_success], minlength=len(frame_counts))
   ```

   Do not rewrite keypoint coordinate arrays or lineage arrays.
4. Leave `KEYPOINTS_SPEC` intact. The spec is correct for the current writer intent; the writer/backfill should satisfy it.
5. Treat this as another data point for the `FrameDomains` resolver proposed in `docs/identity_lineage_staleness_review.md`: every consumer should stop independently deriving frame-domain lengths.

## Training Decision

Proceeding with SAM3 teacher training from the current RedScare keypoint labels is reasonable, with one caveat:

- Safe to use: row-indexed labels and lineage (`keypoints_roi`, `keypoints_img`, `keypoints_norm`, `keypoint_confidences`, `frame_indices`, `detection_indices`, `detection_success`, crop row lineage).
- Do not use as authoritative until fixed/backfilled: the `n_keypoints` per-frame summary/count array in the two affected runs.

If a training/export path uses `n_keypoints` to select frames or validate frame-level success, fix/backfill first. If it uses the row-level keypoint arrays and `detection_success`, the mismatch is not a blocker.
