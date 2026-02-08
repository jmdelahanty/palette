# Detection Dataset Statistics (Future Work)

## Goal
Define a lightweight, repeatable way to inspect detection training data representation before/after curation changes.

This is not required for current progress, but it is useful when:
- adding new rigs/canvases/protocols,
- changing label pipelines,
- investigating unexpected metric shifts.

## Why This Matters
- Helps catch distribution drift between training sets.
- Makes train/val mismatch visible early.
- Provides objective context for model performance changes.

## Recommended Minimal Scope (Phase 1)
For each detection training set, compute and store:

- Dataset size:
  - image count,
  - bbox count,
  - bboxes per image.
- Box geometry (at training resolution):
  - bbox area distribution (quantiles),
  - bbox width/height distributions,
  - aspect ratio distribution,
  - tiny-box rate (below threshold).
- Spatial distribution:
  - bbox center heatmap summary,
  - edge-proximity rate.
- Image-level summary:
  - intensity mean/std,
  - saturation/clipping rate,
  - blur/sharpness proxy.
- Source composition:
  - counts by rig, dish design, canvas, protocol/session group.
- Split parity:
  - compare train vs val for key geometry/intensity metrics.

## Nice-to-Have (Phase 2)
- Duplicate/near-duplicate checks across train/val.
- Explicit drift checks vs last accepted training set.
- Threshold-based warn/fail gating for CI/pipeline.

## Suggested Storage
Write a `dataset_stats.json` (or `dataset_stats.yaml`) next to the training set artifacts, then optionally upsert a summary into the registry.

Recommended location pattern:
- `/nvme1/training/datasets/<set_id>/dataset_stats.json`

## Practical Review Checklist
Before training:
- Are train/val bbox size distributions similar?
- Is source composition unexpectedly skewed?
- Are intensity/blur stats materially different from prior set?
- Is tiny-box fraction unusually high?

After training:
- If metrics regress, do stats indicate shift in geometry/intensity/source mix?

## Decision for Now
Keep as documentation and defer implementation until data diversity increases or regressions become harder to explain.
