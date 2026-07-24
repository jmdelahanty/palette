# Deferred Clipped Training Visual Inspection
<!-- contract-meta
status: operator_checklist
last_verified: 2026-05-16
purpose: Record the exact next-day visual smoke for sleepyfish clipped training Zarrs before using them in final training exports.
-->

## Why This Is Deferred

The sleepyfish clipped training Zarrs were generated from Orange-style keyframe
aligned clips and copied cleaned detection labels only after exact parent-frame
map equality checks. The remaining caution is visual: confirm that the clipped
sample rows render the same biological frames as the original full-video
training Zarrs.

This check should happen before treating the clipped training Zarrs as the
source of record for the next exported detection dataset.

## Visual Smoke

Use the copied refined detect run and review all frames for one clipped camera:

```bash
scripts/py -m fisheye.tune.detect_review \
  /nvme1/recordings/sleepyfish_2026_05_05_17_45_30_cam2010093/zarr/sleepyfish_2026_05_05_17_45_30_cam2010093_clipped_training.zarr \
  --refined-run refined_detect_2026-05-13_16-00-31_review_migrated \
  --all \
  --max-frames 20
```

Then spot-check the other clipped cameras if the first camera looks correct:

```bash
for cam in 2010094 2010095 2010096; do
  scripts/py -m fisheye.tune.detect_review \
    /nvme1/recordings/sleepyfish_2026_05_05_17_45_30_cam${cam}/zarr/sleepyfish_2026_05_05_17_45_30_cam${cam}_clipped_training.zarr \
    --refined-run refined_detect_2026-05-13_16-00-31_review_migrated \
    --all \
    --max-frames 20
done
```

## What To Confirm

- The first, middle, and last sampled frames show the expected sleepyfish dish.
- Boxes and present/missing labels align with the fish in the rendered clipped
  frames.
- The dish mask overlays the same dish location across sampled rows.
- No frame appears to come from the wrong clip, wrong camera, or wrong parent
  time.
- If a detection is missing or outside the dish, treat that as a label problem
  only after confirming the source-frame mapping is correct.

## Source-Frame Sanity

The implementation already verified exact equality between the original
training Zarr's `raw_video/original_frame_indices` and the clipped training
Zarr's parent-frame samples before copying label groups. The important
semantics are:

- `raw_video/original_frame_indices` in clipped training Zarrs contains parent
  frame indexes.
- Stage `frame_indices` remain sample-local indices into `raw_video/images_*`.
- Exact source clip lookup comes from `source_frame_index.parquet`.

## After Passing

Register or rescan the clipped training Zarrs, then use the registry-driven
detection training export. The registry preparation wrapper should prefer
`source_layout="rolling_clips"` training Zarrs over original full-video sampled
Zarrs when they have identical parent-frame samples, preventing double-counted
labels.

Do not include both the original sleepyfish training Zarrs and their clipped
replacements in the same exported dataset unless double-counting is explicitly
intentional and documented.
