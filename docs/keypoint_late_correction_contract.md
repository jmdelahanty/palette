# Keypoint Late-Correction Contract
<!-- contract-meta
version: 1
status: draft
implementation: implemented
last_verified: 2026-04-24
stage_arrays_spec: REFINED_KEYPOINTS_SPEC
-->

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
    {
      "frame_idx": 1234,
      "roi_idx": 56,
      "source_refined_row_id": 8821,
      "source_detect_row_index": 4410
    },
    {"frame_idx": 1300},
    {"roi_idx": 77}
  ]
}
```

Semantics:

- `source_refined_row_id` and `source_detect_row_index` are optional stable
  row-lineage fields copied from the active crop run when available.
- ROI-aware consumers must prefer `source_refined_row_id`, then
  `source_detect_row_index`, before falling back to legacy `frame_idx`/`roi_idx`
  targeting.
- If a flag carries a stable ID and the current crop run exposes the matching
  identity array, an unresolved ID means the row is no longer present; consumers
  should not silently retarget by stale ROI position.
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
      "source_refined_row_id": 8821,
      "source_detect_row_index": 4410,
      "action": "keypoint_nudge",
      "preserve_eye_masks": true
    }
  ]
}
```

Semantics:

- Stable row-lineage targeting and `frame_idx`/`roi_idx` fallback match the main
  flag contract.
- `action="keypoint_nudge"` + `preserve_eye_masks=true` declare operator intent:
  keypoints should be nudged without regenerating masks by default.
- Extra metadata keys are advisory for operators/tools and do not change target
  resolution semantics, except the stable row-lineage fields defined above.

## Targeted Manual Correction Contract

`fisheye.tune.keypoint_review --manual` accepts `--frames` with:

- comma/space frame list, or
- JSON/text file path.

When JSON entries include `roi_idx`, manual review targets exact ROI rows.
When JSON entries include stable row-lineage fields, manual review resolves
those fields against the active/source crop run before falling back to legacy
ROI rows. Nudge files using the shape above are accepted as valid target input.

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

## Explicit Stale Resolution Contract (Preserve Masks)

When operators intentionally preserve curated eye masks after small keypoint
nudges, stale markers can be explicitly resolved instead of regenerating masks.

CLI:

```bash
scripts/py -m fisheye.utils.resolve_eye_mask_stale <zarr_or_root> \
  --zarr-use training \
  --apply \
  --resolution manual_accept_after_keypoint_nudge_preserve_masks \
  --reviewer "$USER"
```

Resolution payload updates (on matching `eye_masks_runs` / `refined_eye_masks_runs`):

- `source_keypoint_stale.state = "resolved"`
- `source_keypoint_stale.resolved_at_utc`
- `source_keypoint_stale.resolution`
- optional: `source_keypoint_stale.resolved_by`, `resolved_notes`

Safety:

- Original stale evidence is preserved (`reason`, `roi_indices`,
  `frame_indices`, and original stale timestamp copied to
  `stale_timestamp_utc` when needed).
- This is an explicit operator acknowledgment path and should be used only when
  mask geometry is intentionally preserved.
