# Swim-Bladder Tuning Metadata Audit

Purpose: verify that swim-bladder tuning propagated by camera is stored
cleanly, and distinguish the original hand-tuned source zarr from cleaned
propagated targets.

This is an operator runbook for
`scripts/py -m fisheye.utils.audit_swim_bladder_tuning_metadata`.

## Why This Exists

Older swim-bladder tuning propagation copied the full
`analysis_metadata.subject_mask_tuning.components["swim_bladder"]` entry into
target zarrs, including source-specific `context` fields such as:

- `crop_run`
- `keypoint_run`
- `keypoint_source`
- ROI-local frame / detection / patch-center details

That copied metadata did not break existing materialized masks, but it could
make manual inspection misleading because propagated targets appeared to refer
to the original source archive's local crop/keypoint lineage.

The cleanup workflow now:

- preserves the hand-tuned source zarr with its real local context
- rewrites propagated targets with a cleaned swim-bladder component entry
- marks those cleaned targets with:
  - `propagated_by_camera = true`
  - `propagated_component = "swim_bladder"`

## What The Audit Checks

The audit reads `analysis_metadata/zarr.json` directly and classifies each zarr
for the swim-bladder component:

- `source_like`
  The swim-bladder tuning entry still has source-specific context, and the
  referenced `crop_run` and `keypoint_run` both exist locally in that same
  zarr. This is the expected hand-tuned source.
- `clean_propagated`
  The tuning entry is marked `propagated_by_camera`, and the stale
  source-specific context keys have been removed. This is the expected cleaned
  propagated target.
- `stale_source_context`
  The entry still contains source-specific context keys, but the referenced
  crop/keypoint lineage does not exist locally. This is the old misleading
  propagated state and should be cleaned up.
- `clean_unmarked`
  The entry has no stale source-specific context, but it is also not marked as
  propagated. This is unusual and should be inspected.
- `missing_swim_tuning`
  No swim-bladder tuning entry is present.

Strict mode also checks camera-level consistency:

- exactly one `source_like` zarr per camera
- zero `stale_source_context`
- zero `missing_swim_tuning`
- one shared swim-bladder `method` per camera
- one shared swim-bladder `tuned_timestamp` per camera

## Recommended Command

For the current first swim-bladder training canaries:

```bash
scripts/py -m fisheye.utils.audit_swim_bladder_tuning_metadata \
  /nvme1/recordings \
  --recursive \
  --camera-id 2010093 \
  --camera-id 2010094 \
  --camera-id 2010095 \
  --camera-id 2010096 \
  --strict
```

Use `--show-all` if you want every scanned row instead of only the interesting
statuses.

## Expected Healthy Output

For a cleaned camera propagation set, the summary should look like:

- one `source_like`
- the remaining zarrs `clean_propagated`
- zero `stale_source_context`
- zero `missing_swim_tuning`

Example interpretation:

- `camera_id=2010094 total=14 source_like=1 clean_propagated=13`
  one hand-tuned source zarr and thirteen cleaned propagated training targets

## Cleanup Command

If the audit reports `stale_source_context`, rerun propagation from the actual
hand-tuned source zarr:

```bash
scripts/py -m fisheye.utils.apply_tuning_by_camera \
  /nvme1/recordings \
  --source <hand_tuned_training>.zarr \
  --recursive \
  --apply \
  --keys subject_mask_tuning \
  --subject-mask-components swim_bladder \
  --overwrite
```

Important:

- use `--overwrite`
- do not use `--merge-dicts` for this cleanup pass

Reason:

- `--overwrite` replaces only
  `subject_mask_tuning.components["swim_bladder"]`
- unrelated subject-mask components such as `subject_body` are preserved
- eye-mask tuning is untouched
- existing materialized mask runs are untouched
- `--merge-dicts` would preserve stale nested `context` fields on the target

## Notes

- This audit is metadata-only. It does not inspect or rewrite
  `subject_mask_runs`.
- A camera can legitimately have different swim-bladder `tuned_timestamp`
  values across cameras. Strict mode checks consistency within each camera, not
  across all cameras.
- The audit is intentionally based on metadata-file reads rather than sync
  `zarr.open_group(...)`, so it remains usable in sandbox situations where
  normal zarr access may hang.
