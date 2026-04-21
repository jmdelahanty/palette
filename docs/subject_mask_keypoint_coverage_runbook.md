# Subject-Mask Keypoint Coverage Runbook

Use this runbook when checking that modern subject-mask eye components cover
all rows that are still keypoint-usable.

The invariant is:

- If a row has usable/refined keypoints for the reviewed keypoint run, the
  subject-mask run must contain the required eye component mask(s).
- If a fish is present but no keypoints should be available, repair the
  refined keypoint row instead of forcing an eye-mask label.

## Run the Diagnostic

```bash
scripts/py -m fisheye.diagnostics.check_subject_mask_keypoint_coverage \
  /nvme1/recordings \
  --recursive \
  --zarr-use training \
  --log-dir /tmp/subject-mask-coverage \
  --write-frame-flag-file /tmp/subject-mask-coverage/eye-review-flags.json \
  --write-repair-plan /tmp/subject-mask-coverage/repair-plan.jsonl
```

The JSONL report goes in `--log-dir`. The frame flag file is overwritten on
each run and contains only current failures. The repair plan is JSONL with one
row per failing ROI/frame.

A clean run looks like:

```text
Subject-mask keypoint coverage summary: ...
zarr_fail=0 zarr_missing=0 errors=0
frame_flag_targets_written=0 repair_plan_rows_written=0 issues=no
```

## Inspect Failures

List failing zarr/ROI/frame targets:

```bash
jq -r 'to_entries[] | .key as $z | .value[] | [$z, .roi_idx, .frame_idx] | @tsv' \
  /tmp/subject-mask-coverage/eye-review-flags.json
```

List failing zarrs with the keypoint run that the diagnostic checked from the
repair plan:

```bash
jq -r '
  [.zarr, .target.roi_idx, .target.frame_idx, .keypoint_group, .keypoint_run]
  | @tsv
' /tmp/subject-mask-coverage/repair-plan.jsonl
```

Each repair-plan row also includes candidate commands:

```bash
jq -r '.repair_options.eye_mask_review.shell' \
  /tmp/subject-mask-coverage/repair-plan.jsonl

jq -r '.repair_options.keypoint_review.shell' \
  /tmp/subject-mask-coverage/repair-plan.jsonl
```

Use the `keypoint_run` from the repair plan when repairing keypoints. Do not
assume `refined_keypoints_runs/latest` is the same run; sampled/promoted runs
can make `latest` differ from the lineage run checked by the diagnostic.

## Classify Each Failure

Open the flagged ROI and decide which surface is wrong:

- Missing eye component mask: fish and keypoints are valid, but one or both eye
  component masks are missing or incorrect. Repair the subject/eye mask.
- Fish present but no keypoints: the fish is present, but the keypoints should
  be unavailable or unlabeled. Repair the refined keypoint row.
- Detection/crop issue: the ROI does not contain a valid fish instance or the
  crop is wrong. Mark it through the keypoint review detection-issue path so it
  can be routed back to detection/crop repair.

Do not patch the eye mask just to satisfy coverage when the keypoint row should
be invalid. That hides the real training-label intent.

## Repair Missing Eye Component Masks

Use `eye_mask_review` with the frame flag file:

```bash
jq -r '.repair_options.eye_mask_review.shell' \
  /tmp/subject-mask-coverage/repair-plan.jsonl
```

Run the row command after confirming the failure is an eye-component mask
problem. The generated command includes `--refined-run` when the diagnostic can
resolve a source or latest refined eye-mask run.

The review UI uses the frame flag file to start at the flagged ROI.

## Repair Fish Present But No Keypoints

Use `keypoint_review` against the refined keypoint run reported by the repair
plan:

```bash
jq -r '.repair_options.keypoint_review.shell' \
  /tmp/subject-mask-coverage/repair-plan.jsonl
```

In the UI, press `x` on the flagged ROI to mark
`fish_present_no_keypoints`. This writes the keypoint failure intent to the
refined keypoint run:

- `refined_success=false`
- `usable_keypoints=false`
- keypoint coordinates and derived keypoint fields cleared/invalidated
- reason labels include `fish_present_no_keypoints`
- heading fields are refreshed after close

For the policy behind this label, see
`docs/keypoint_review_policy.md`.

## Verify

Rerun the diagnostic with the same command:

```bash
scripts/py -m fisheye.diagnostics.check_subject_mask_keypoint_coverage \
  /nvme1/recordings \
  --recursive \
  --zarr-use training \
  --log-dir /tmp/subject-mask-coverage \
  --write-frame-flag-file /tmp/subject-mask-coverage/eye-review-flags.json \
  --write-repair-plan /tmp/subject-mask-coverage/repair-plan.jsonl
```

Expected result:

- `zarr_fail=0`
- `zarr_missing=0`
- `errors=0`
- `frame_flag_targets_written=0`
- `repair_plan_rows_written=0`
- `issues=no`

If failures remain, the refreshed
`/tmp/subject-mask-coverage/repair-plan.jsonl` is the next repair queue.

## Example From The Coverage Repair Pass

The diagnostic found one row where the eye mask needed repair and one row where
the correct fix was `fish_present_no_keypoints`.

For the keypoint repair case, the subject-mask lineage referenced
`refined_keypoints_2026-02-04_12-43-25`, while
`refined_keypoints_runs/latest` pointed to a different promoted run. The repair
therefore had to use:

```bash
recording=/nvme1/recordings/2026-01-28T19-22-28Z_arena_2_DefaultScreen
zarr="$recording/zarr/2026-01-28T19-22-28Z_arena_2_DefaultScreen_training.zarr"

scripts/py -m fisheye.tune.keypoint_review \
  "$zarr" \
  --manual \
  --refined-run refined_keypoints_2026-02-04_12-43-25 \
  --frames /tmp/subject-mask-coverage/eye-review-flags.json \
  --review-state approved \
  --review-method manual \
  --review-intended-use training
```

The final verification run reported:

```text
zarr_checked=52 zarr_pass=52 zarr_fail=0 zarr_missing=0 errors=0 frame_flag_targets_written=0 issues=no
```
