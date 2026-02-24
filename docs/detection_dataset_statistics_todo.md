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

Canonical schema contract (defined):
- `docs/detection_data_profile_schema_contract.md`

## Status

- [x] Define canonical `v1` schema for:
  - dataset profile artifact payload,
  - registry projection,
  - training data card aggregate.
- [x] Define linkage policy:
  - profile rows link to production stage output via `source_detection_path`,
  - defer stage reverse pointer (`profile_latest_ref`) for now.
- [x] Add operator-facing finalized detect visualization workflow for approved refined runs.
  - `scripts/py -m fisheye.utils.finalize_refinement_artifacts` writes:
    - `detect_quality_overview_png`
    - `refinement_pipeline_overview_png`
  - `scripts/py -m fisheye.utils.export_detect_quality_overview --artifact ... --view`
    supports recursive direct viewing from Zarr.
- [x] Implement profile writer + backfill utility.
  - `scripts/py -m fisheye.utils.backfill_detection_profiles`
  - writes canonical profile payloads under
    `analysis/detection_profile_runs/<run>/attrs["profile_summary"]`
- [x] Add registry table/view migration + query surface.
  - `detection_data_profile`, `detection_data_profile_latest`,
    `recording_detection_data_profile_latest`
  - query surface:
    `scripts/py -m fisheye.utils.registry_query --detection-data-profile-latest`
    and
    `scripts/py -m fisheye.utils.registry_query --recording-detection-data-profile-latest`
- [x] Add registry sync utility for profile rows from Zarr profile runs.
  - `scripts/py -m fisheye.utils.sync_detection_profile_registry`
- [x] Add training data card aggregation command.
  - `scripts/py -m fisheye.utils.aggregate_detection_training_data_card --manifest <set>.manifest.json --registry <registry.sqlite>`
  - optional pipeline integration:
    `scripts/py -m fisheye.utils.run_detect_training_pipeline ... --aggregate-training-data-card`

## Execution Evidence (2026-02-24)

Detection profile registry surfaces were validated on production training data
with the following sequence and outcomes:

1. Populate registry dataset rows:
   - `scripts/py -m fisheye.utils.registry_rescan /nvme1/recordings --recursive --registry /nvme1/registry.sqlite`
2. Backfill training profile runs:
   - `scripts/py -m fisheye.utils.backfill_detection_profiles /nvme1/recordings --recursive --zarr-use training --registry /nvme1/registry.sqlite --apply`
   - observed: `zarr_scanned=105`, `filtered_zarr_use=53`, `updated=52`, `errors=0`
3. Sync latest profile summaries into registry:
   - `scripts/py -m fisheye.utils.sync_detection_profile_registry --registry /nvme1/registry.sqlite --zarr-use training --apply`
   - initial observed: `datasets=52`, `updated=51`, `missing_profile=1`
4. Recover single missing-profile dataset:
   - `scripts/py -m fisheye.utils.sync_detection_profile_registry --registry /nvme1/registry.sqlite --dataset-id 2026-01-28T19-22-28Z_arena_1:zc66de17bea1b --apply`
   - observed: `updated=1`, `errors=0`
5. Verify operator-facing registry query view:
   - `scripts/py -m fisheye.utils.registry_query --registry /nvme1/registry.sqlite --recording-detection-data-profile-latest --profile-zarr-use training --json | jq 'length'`
   - observed: `52`

## Practical Review Checklist
Before training:
- Are train/val bbox size distributions similar?
- Is source composition unexpectedly skewed?
- Are intensity/blur stats materially different from prior set?
- Is tiny-box fraction unusually high?

After training:
- If metrics regress, do stats indicate shift in geometry/intensity/source mix?

## Decision for Now
Detection profile schema, writer/backfill, registry projection/query, sync
workflow, and training data card aggregation are implemented and validated.
