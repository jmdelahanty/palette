# Documentation Staleness Audit: 2026-05-17

## Scope

This pass reviewed the docs most likely to mislead current cluster/training
work:

- `traditional_v2` keypoint migration and multi-skeleton rollout docs
- PyNvVC-luma crop representation docs
- operator training-data docs
- clipped/cluster migration checklist docs

## Corrections Applied

- Updated `traditional_v2` docs to state the current recommended path:
  reliable 3-point inference, refined-keypoint seed promotion to
  `traditional_v2`, manual completion of `snout_tip` and `tail_tip`, then
  retraining.
- Marked direct 5-point YOLO inference as runtime-supported but not currently
  approved as an automatic label source for new PyNvVC-luma crops.
- Updated multi-skeleton docs to reflect that the YOLO keypoint writer can now
  write schema-stamped dynamic-`K` outputs.
- Updated training-selection docs to reflect that effective annotation source
  metadata is now implemented for the main registry/export path.
- Replaced the stale "8 sampled zarrs are detection-only and have no crop runs"
  inventory note with a dynamic inspection command, because those archives are
  now being promoted into crop/keypoint training sources.
- Added operator guidance for manually completing 5-point seed runs over
  SSH/tmux with the existing Matplotlib keypoint reviewer.

## Still Open Or Intentionally Deferred

- `docs/keypoint_training_refined_run_tie_fix_todo.md` remains valid. The
  repair utility exists, but the underlying preflight tie-break logic still
  sorts candidate refined runs primarily by `created_utc`.
- Raw `traditional_pose` detection remains a 3-point producer by design. This
  is not a stale doc issue; it is a producer-contract limitation.
- Existing low-confidence direct `traditional_v2` prediction runs should remain
  experimental unless visually corrected and promoted. They should not be
  treated as training labels just because the schema and arrays exist.
- The sleepyfish clipped-training visual inspection checklist remains
  intentionally deferred until operator review confirms the rendered clipped
  frames and copied labels align.
- The web keypoint reviewer work is not documented here yet because it is still
  an implementation prototype. The durable contract should be added after the
  backend/web UI semantics are finalized.

## Metadata-Only Inventory Checks

Use metadata-file reads when auditing current local zarr state from Codex or
over SSH:

```bash
for z in /nvme1/recordings/{sickyfish_*,sleepyfish_*}/zarr/*_training.zarr \
         /nvme1/recordings/{sickyfish_*,sleepyfish_*}/zarr/*_clipped_training.zarr; do
  [ -d "$z" ] || continue
  echo "$z"
  jq -r '.attributes.latest // "no crop_runs latest"' "$z/crop_runs/zarr.json" 2>/dev/null || true
  jq -r '.attributes.latest // "no keypoints_runs latest"' "$z/keypoints_runs/zarr.json" 2>/dev/null || true
  jq -r '.attributes.latest // "no refined_keypoints_runs latest"' "$z/refined_keypoints_runs/zarr.json" 2>/dev/null || true
done
```
