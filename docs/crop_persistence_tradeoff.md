# Crop Persistence Tradeoffs

## Why crops are persisted today
- High-resolution frames (e.g., 4512×4512) make on-the-fly cropping expensive.
- Persisted crops make downstream steps deterministic and fast:
  - keypoints, eye masks, refinement, QC, and tuning
- Decouples processing from codec/video IO differences.
- Enables rapid iteration without re-reading full videos.

## Why this can feel inefficient
- Labeling tools typically crop on demand and do not store ROIs.
- Persisting crops trades storage for speed and stability.

## When on-the-fly cropping makes sense
- Lower-resolution frames (cheap random access).
- Detection is stable (nearly all frames have a correct bbox).
- Training/analysis workflows rarely need repeated ROI access.

## Proposed transition path
- Keep persisted crops for:
  - failure/QC frames
  - manual correction sets
  - training snapshots
- Allow an on-demand crop mode for routine runs.
- Promote to full crop persistence only when needed.

## Future options
- Add a pipeline flag: crop_mode = persist | on_demand.
- Cache only selected frames (e.g., misses or interpolated detections).
- Support lite Zarr with detections + metadata; compute crops on demand.

## Related Discussion
- See `crop_live_view_vs_materialized_stream_design.md` for a fuller architecture and registry-focused treatment of live-crop-first + optional materialization.
