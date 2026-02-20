# Keypoint Late-Correction Contract

Purpose: define how to flag missed keypoint ROIs late in the workflow, apply
targeted corrections, and mark downstream eye-mask runs stale without rerunning
all keypoint tracking.

## Scope

- Applies to curated `_training.zarr` and `_analysis.zarr` archives.
- Applies to corrections written into `refined_keypoints_runs/<run>`.
- Does not mutate historical raw detections/crops by default.

## ROI Flag Contract (`keypoint_frame_flags.json`)

`keypoint_frame_flags.json` is a JSON object keyed by Zarr path:

```json
{
  "/path/to/recording_training.zarr": [
    {"frame_idx": 1234, "roi_idx": 56},
    {"frame_idx": 1300},
    {"roi_idx": 77}
  ]
}
```

Semantics:

- `frame_idx` only: include all ROIs mapped to that frame.
- `roi_idx` only: include that exact ROI row.
- both present: include that exact ROI row (frame is informational/provenance).
- Duplicate entries are ignored by writer tools.

Compatibility:

- Legacy frame-only lists (e.g. `[123, 456]`) remain valid.

## Keypoint Nudge Flag Contract (`keypoint_nudge_flags.json`)

When reviewing eye masks, operators can flag targeted keypoint nudges that are
intended to keep existing eye masks unchanged.

Writer:

- `fisheye.visualization.visualize_eye_mask_patches` hotkey `k`.

Default file:

- `keypoint_nudge_flags.json` (JSON object keyed by zarr path).

Entry shape:

```json
{
  "/path/to/recording_training.zarr": [
    {
      "frame_idx": 1234,
      "roi_idx": 56,
      "action": "keypoint_nudge",
      "preserve_eye_masks": true
    }
  ]
}
```

Semantics:

- `frame_idx`/`roi_idx` targeting matches the main flag contract.
- `action="keypoint_nudge"` + `preserve_eye_masks=true` declare operator intent:
  keypoints should be nudged without regenerating masks by default.
- Extra metadata keys are advisory for operators/tools and do not change target
  resolution semantics.

## Targeted Manual Correction Contract

`fisheye.tune.keypoint_review --manual` accepts `--frames` with:

- comma/space frame list, or
- JSON/text file path.

When JSON entries include `roi_idx`, manual review targets exact ROI rows.
Nudge files using the shape above are accepted as valid target input.

## Automatic Downstream Stale Marking Contract

When keypoints are edited (manual save/mark actions or patch utility apply), the
system marks dependent eye-mask runs stale:

- `eye_masks_runs/<run>.attrs["source_keypoint_stale"]`
- `refined_eye_masks_runs/<run>.attrs["source_keypoint_stale"]`

Payload shape:

```json
{
  "state": "stale",
  "timestamp": "2026-02-12T20:30:00.000000+00:00",
  "reason": "keypoint_manual_correction",
  "source_keypoint_group": "refined_keypoints_runs",
  "source_keypoints_run": "refined_keypoints_...",
  "roi_indices": [56, 77],
  "frame_indices": [1234, 1300]
}
```

Notes:

- Marking is run-local and only affects eye-mask runs whose source keypoint
  lineage matches (`source_keypoint_group` + `source_keypoints_run`).
- Index history lists are deduplicated and bounded in size.

## Operational Consequences

- Keypoint approval should be reset to `needs_review` or `pending` before
  correction, then re-approved after correction/audit.
- Existing eye-mask runs sourced from the edited keypoint run should be treated
  as stale until regenerated or manually corrected.
- Training exports/models generated before correction will not update
  automatically; re-export/retrain is required when corrected rows matter.
