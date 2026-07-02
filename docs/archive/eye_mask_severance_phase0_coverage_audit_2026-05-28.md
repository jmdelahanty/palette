# Eye-Mask Severance Phase 0 Coverage Audit

> Archived execution note, 2026-07-01: this predecessor audit has been
> superseded by `docs/diagnostics/eye_mask_severance_census_2026-07-01.md` and
> the severance commits `6ab7843`, `cfc0d02`, `73ab164`, `4a85e5d`, `b983b38`,
> and `edb2534`. Retained for historical context only.

**Date:** 2026-05-28
**Amended:** 2026-05-30
**Mode:** original audit was read-only. On 2026-05-30, the smoke-only registry row below was marked `datasets.status = 'inactive_smoke'`; no recording zarrs were modified.
**Verdict:** **GO** for active production registry coverage and Palette Phase 1 consumer re-pointing. The only legacy-only gap was a smoke fixture and is now explicitly excluded from active registry coverage.

Phase 0 exists to answer one safety question: would removing the legacy `refined_eye_masks_runs` fallback silently remove a currently available eye-geometry source? The answer was yes for one registry path, and it was a smoke archive:

`/nvme1/recordings/smoke/2026-01-28T19-22-28Z_arena_1_DefaultScreen_geometry_smoke_analysis.zarr`

This archive has populated legacy `refined_eye_masks_runs/refined_threshold_validation_single_ref_v7` geometry but no resolver-ready `analysis/subject_shape_runs` or `refined_subject_masks_runs` eye geometry. It is not production data, so it was removed from active coverage by changing its registry row from `active` to `inactive_smoke` while preserving the row for traceability.

Registry mutation evidence:

- Active registry: `/nvme1/palette_registry.sqlite`
- Backup before edit: `/nvme1/palette_registry_before_smoke_exclusion_20260530.sqlite`
- Dataset row: `2026-01-28T19-22-28Z_arena_1:za40d85eb352f`
- Current row status: `inactive_smoke`
- Current status counts: `323 active`, `1 inactive_smoke`, `1 missing`

## Method

I verified the active code paths before scanning:

- `src/fisheye/shared/eye_geometry_source.py` resolves in this order when `prefer_subject_shape=True`: `analysis/subject_shape_runs`, then `refined_subject_masks_runs`, then `refined_eye_masks_runs`.
- Subject-shape geometry is `analysis/subject_shape_runs/<run>/components/{eye_left,eye_right}/ellipse_params`, `ellipse_success`, plus `relations/eye_pair/separation_px`.
- Refined-subject geometry is `refined_subject_masks_runs/<run>/components/{eye_left,eye_right}/geometry/ellipse_params`, `ellipse_success`, plus `relations/eye_pair/metrics/separation_px`.
- Legacy refined-eye geometry is `refined_eye_masks_runs/<run>/masks_roi`, `ellipse_params`, and `ellipse_success`.

The scan used metadata-file checks (`zarr.json` / `.zarray` / `.zattrs`) instead of sync `zarr.open_group(...)`, to avoid sandbox hangs and to keep the audit read-only. After the 2026-05-30 registry cleanup, the denominator is 58 active analysis zarrs from `/nvme1/palette_registry.sqlite`: 57 active `source_recording` paths plus one real `/groups/.../palette_smoke` derived analysis path. `/tmp`, in-memory, pytest, derived training zarrs, and `inactive_smoke` rows are excluded.

## Coverage Summary

| Status | Count |
|---|---:|
| Scanned real active analysis zarrs | 58 |
| Existing paths | 58 |
| Resolver-effective `subject_shape` source | 48 |
| Resolver-effective `refined_subject` source | 4 |
| Legacy refined-eye only | 0 |
| No eye-geometry source at all | 6 |
| Dangerous bucket: legacy yes / subject no | 0 |
| Excluded smoke-only legacy fixture | 1 |

The 6 `none` archives are not a severance data-loss risk because they have no legacy eye-geometry fallback either. They already cannot provide eye-angle geometry through the legacy path.

## Excluded Smoke Fixture

| recording_id | subject_shape | refined_subject | legacy_refined_eye | legacy run | path |
|---|---:|---:|---:|---|---|
| `2026-01-28T19-22-28Z_arena_1` | no | no | yes | `refined_threshold_validation_single_ref_v7` | `/nvme1/recordings/smoke/2026-01-28T19-22-28Z_arena_1_DefaultScreen_geometry_smoke_analysis.zarr` |

Metadata confirms the legacy run has `masks_roi` shape `(23287, 2, 512, 512)`, `ellipse_params`, `ellipse_success`, `detection_source`, `frame_indices`, `source_crop_run = crop_2026-03-06_08-45-57`, and anatomical `eye_labels = ["eye_left", "eye_right"]`. This is enough for the legacy-to-subject projection tool to plan a subject-mask backfill if we ever want to modernize the fixture, but it is not required for production severance.

## Optional Smoke-Fixture Backfill Plan

This is optional and should only be run if the excluded smoke fixture should become a modern subject-mask test archive. Dry-run the legacy-to-subject projection first:

```bash
ZARR=/nvme1/recordings/smoke/2026-01-28T19-22-28Z_arena_1_DefaultScreen_geometry_smoke_analysis.zarr
REFINED_EYE_RUN=refined_threshold_validation_single_ref_v7
SUBJECT_RUN=subject_masks_from_refined_threshold_validation_single_ref_v7
REFINED_SUBJECT_RUN=refined_subject_masks_from_refined_threshold_validation_single_ref_v7

scripts/py -m fisheye.utils.backfill_subject_mask_runs \
  "$ZARR" \
  --zarr-use analysis \
  --source-stage refined_eye_masks_runs \
  --source-run "$REFINED_EYE_RUN" \
  --run-name "$SUBJECT_RUN" \
  --label-schema auto \
  --batch-size 256
```

If that looks right, apply only the projected subject-mask run:

```bash
scripts/py -m fisheye.utils.backfill_subject_mask_runs \
  "$ZARR" \
  --zarr-use analysis \
  --source-stage refined_eye_masks_runs \
  --source-run "$REFINED_EYE_RUN" \
  --run-name "$SUBJECT_RUN" \
  --label-schema auto \
  --batch-size 256 \
  --apply
```

Then dry-run refined-subject finalization against the newly-created subject run:

```bash
scripts/py -m fisheye.refinement.finalize_subject_masks \
  "$ZARR" \
  --subject-run "$SUBJECT_RUN" \
  --refined-run "$REFINED_SUBJECT_RUN" \
  --components eye_left eye_right \
  --write-eye-geometry \
  --dry-run
```

Apply refined-subject finalization only after that dry-run looks right:

```bash
scripts/py -m fisheye.refinement.finalize_subject_masks \
  "$ZARR" \
  --subject-run "$SUBJECT_RUN" \
  --refined-run "$REFINED_SUBJECT_RUN" \
  --components eye_left eye_right \
  --write-eye-geometry
```

If a refined-subject run already exists but lacks geometry, use:

```bash
scripts/py -m fisheye.utils.backfill_refined_subject_eye_geometry \
  "$ZARR" \
  --zarr-use analysis \
  --all-runs
```

and add `--apply` only after the dry-run output shows the intended target run.

## Active Coverage Table

| recording_id | effective source | subject_shape | refined_subject | legacy_refined_eye | dangerous | path |
|---|---|---:|---:|---:|---:|---|
| `sleepyfish_2026_05_05_17_45_30_cam2010093` | none | no | no | no | no | `/groups/johnson/johnsonlab/jeremy/palette_smoke/sleepyfish_2026_05_05_17_45_30_cam2010093/zarr/sleepyfish_2026_05_05_17_45_30_cam2010093_analysis.zarr` |
| `sleepyfish_2026_05_05_17_45_30_cam2010095` | none | no | no | no | no | `/groups/johnson/johnsonlab/jeremy/recordings/sleepyfish_2026_05_05_17_45_30_cam2010095/zarr/sleepyfish_2026_05_05_17_45_30_cam2010095_analysis.zarr` |
| `2026-01-28T19-22-28Z_arena_1` | refined_subject | no | yes | no | no | `/nvme1/recordings/2026-01-28T19-22-28Z_arena_1_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_1_DefaultScreen_analysis.zarr` |
| `2026-01-28T19-22-28Z_arena_2` | refined_subject | no | yes | no | no | `/nvme1/recordings/2026-01-28T19-22-28Z_arena_2_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_2_DefaultScreen_analysis.zarr` |
| `2026-01-28T19-22-28Z_arena_3` | refined_subject | no | yes | no | no | `/nvme1/recordings/2026-01-28T19-22-28Z_arena_3_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_3_DefaultScreen_analysis.zarr` |
| `2026-01-28T19-22-28Z_arena_4` | refined_subject | no | yes | no | no | `/nvme1/recordings/2026-01-28T19-22-28Z_arena_4_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_4_DefaultScreen_analysis.zarr` |
| `2026-01-28T19-36-18Z_arena_1` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T19-36-18Z_arena_1_Feeding/zarr/2026-01-28T19-36-18Z_arena_1_Feeding_analysis.zarr` |
| `2026-01-28T19-36-18Z_arena_2` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T19-36-18Z_arena_2_Feeding/zarr/2026-01-28T19-36-18Z_arena_2_Feeding_analysis.zarr` |
| `2026-01-28T19-36-18Z_arena_3` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T19-36-18Z_arena_3_Feeding/zarr/2026-01-28T19-36-18Z_arena_3_Feeding_analysis.zarr` |
| `2026-01-28T19-36-18Z_arena_4` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T19-36-18Z_arena_4_Feeding/zarr/2026-01-28T19-36-18Z_arena_4_Feeding_analysis.zarr` |
| `2026-01-28T20-41-59Z_arena_1` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T20-41-59Z_arena_1_DefaultScreen/zarr/2026-01-28T20-41-59Z_arena_1_DefaultScreen_analysis.zarr` |
| `2026-01-28T20-41-59Z_arena_2` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T20-41-59Z_arena_2_DefaultScreen/zarr/2026-01-28T20-41-59Z_arena_2_DefaultScreen_analysis.zarr` |
| `2026-01-28T20-41-59Z_arena_3` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T20-41-59Z_arena_3_DefaultScreen/zarr/2026-01-28T20-41-59Z_arena_3_DefaultScreen_analysis.zarr` |
| `2026-01-28T20-41-59Z_arena_4` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T20-41-59Z_arena_4_DefaultScreen/zarr/2026-01-28T20-41-59Z_arena_4_DefaultScreen_analysis.zarr` |
| `2026-01-28T20-51-00Z_arena_1` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T20-51-00Z_arena_1_Feeding/zarr/2026-01-28T20-51-00Z_arena_1_Feeding_analysis.zarr` |
| `2026-01-28T20-51-00Z_arena_2` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T20-51-00Z_arena_2_Feeding/zarr/2026-01-28T20-51-00Z_arena_2_Feeding_analysis.zarr` |
| `2026-01-28T20-51-00Z_arena_3` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T20-51-00Z_arena_3_Feeding/zarr/2026-01-28T20-51-00Z_arena_3_Feeding_analysis.zarr` |
| `2026-01-28T20-51-00Z_arena_4` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T20-51-00Z_arena_4_Feeding/zarr/2026-01-28T20-51-00Z_arena_4_Feeding_analysis.zarr` |
| `2026-01-28T21-18-51Z_arena_1` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T21-18-51Z_arena_1_DefaultScreen/zarr/2026-01-28T21-18-51Z_arena_1_DefaultScreen_analysis.zarr` |
| `2026-01-28T21-18-51Z_arena_2` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T21-18-51Z_arena_2_DefaultScreen/zarr/2026-01-28T21-18-51Z_arena_2_DefaultScreen_analysis.zarr` |
| `2026-01-28T21-18-51Z_arena_4` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T21-18-51Z_arena_4_DefaultScreen/zarr/2026-01-28T21-18-51Z_arena_4_DefaultScreen_analysis.zarr` |
| `2026-01-28T21-27-20Z_arena_1` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T21-27-20Z_arena_1_Feeding/zarr/2026-01-28T21-27-20Z_arena_1_Feeding_analysis.zarr` |
| `2026-01-28T21-27-20Z_arena_2` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T21-27-20Z_arena_2_Feeding/zarr/2026-01-28T21-27-20Z_arena_2_Feeding_analysis.zarr` |
| `2026-01-28T21-27-20Z_arena_4` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T21-27-20Z_arena_4_Feeding/zarr/2026-01-28T21-27-20Z_arena_4_Feeding_analysis.zarr` |
| `2026-01-28T21-47-47Z_arena_1` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T21-47-47Z_arena_1_DefaultScreen/zarr/2026-01-28T21-47-47Z_arena_1_DefaultScreen_analysis.zarr` |
| `2026-01-28T21-47-47Z_arena_2` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T21-47-47Z_arena_2_DefaultScreen/zarr/2026-01-28T21-47-47Z_arena_2_DefaultScreen_analysis.zarr` |
| `2026-01-28T21-47-47Z_arena_3` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T21-47-47Z_arena_3_DefaultScreen/zarr/2026-01-28T21-47-47Z_arena_3_DefaultScreen_analysis.zarr` |
| `2026-01-28T21-47-47Z_arena_4` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T21-47-47Z_arena_4_DefaultScreen/zarr/2026-01-28T21-47-47Z_arena_4_DefaultScreen_analysis.zarr` |
| `2026-01-28T21-56-23Z_arena_1` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T21-56-23Z_arena_1_Feeding/zarr/2026-01-28T21-56-23Z_arena_1_Feeding_analysis.zarr` |
| `2026-01-28T21-56-23Z_arena_2` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T21-56-23Z_arena_2_Feeding/zarr/2026-01-28T21-56-23Z_arena_2_Feeding_analysis.zarr` |
| `2026-01-28T21-56-23Z_arena_3` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T21-56-23Z_arena_3_Feeding/zarr/2026-01-28T21-56-23Z_arena_3_Feeding_analysis.zarr` |
| `2026-01-28T21-56-23Z_arena_4` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T21-56-23Z_arena_4_Feeding/zarr/2026-01-28T21-56-23Z_arena_4_Feeding_analysis.zarr` |
| `2026-01-28T22-15-03Z_arena_1` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T22-15-03Z_arena_1_DefaultScreen/zarr/2026-01-28T22-15-03Z_arena_1_DefaultScreen_analysis.zarr` |
| `2026-01-28T22-15-03Z_arena_2` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T22-15-03Z_arena_2_DefaultScreen/zarr/2026-01-28T22-15-03Z_arena_2_DefaultScreen_analysis.zarr` |
| `2026-01-28T22-15-04Z_arena_3` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T22-15-04Z_arena_3_DefaultScreen/zarr/2026-01-28T22-15-04Z_arena_3_DefaultScreen_analysis.zarr` |
| `2026-01-28T22-15-04Z_arena_4` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T22-15-04Z_arena_4_DefaultScreen/zarr/2026-01-28T22-15-04Z_arena_4_DefaultScreen_analysis.zarr` |
| `2026-01-28T22-22-57Z_arena_1` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T22-22-57Z_arena_1_Feeding/zarr/2026-01-28T22-22-57Z_arena_1_Feeding_analysis.zarr` |
| `2026-01-28T22-22-57Z_arena_2` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T22-22-57Z_arena_2_Feeding/zarr/2026-01-28T22-22-57Z_arena_2_Feeding_analysis.zarr` |
| `2026-01-28T22-22-57Z_arena_3` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T22-22-57Z_arena_3_Feeding/zarr/2026-01-28T22-22-57Z_arena_3_Feeding_analysis.zarr` |
| `2026-01-28T22-22-57Z_arena_4` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T22-22-57Z_arena_4_Feeding/zarr/2026-01-28T22-22-57Z_arena_4_Feeding_analysis.zarr` |
| `2026-01-28T22-42-59Z_arena_1` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T22-42-59Z_arena_1_DefaultScreen/zarr/2026-01-28T22-42-59Z_arena_1_DefaultScreen_analysis.zarr` |
| `2026-01-28T22-42-59Z_arena_2` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T22-42-59Z_arena_2_DefaultScreen/zarr/2026-01-28T22-42-59Z_arena_2_DefaultScreen_analysis.zarr` |
| `2026-01-28T22-42-59Z_arena_3` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T22-42-59Z_arena_3_DefaultScreen/zarr/2026-01-28T22-42-59Z_arena_3_DefaultScreen_analysis.zarr` |
| `2026-01-28T22-42-59Z_arena_4` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T22-42-59Z_arena_4_DefaultScreen/zarr/2026-01-28T22-42-59Z_arena_4_DefaultScreen_analysis.zarr` |
| `2026-01-28T22-50-39Z_arena_1` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T22-50-39Z_arena_1_Feeding/zarr/2026-01-28T22-50-39Z_arena_1_Feeding_analysis.zarr` |
| `2026-01-28T22-50-39Z_arena_2` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T22-50-39Z_arena_2_Feeding/zarr/2026-01-28T22-50-39Z_arena_2_Feeding_analysis.zarr` |
| `2026-01-28T22-50-39Z_arena_3` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T22-50-39Z_arena_3_Feeding/zarr/2026-01-28T22-50-39Z_arena_3_Feeding_analysis.zarr` |
| `2026-01-28T22-50-39Z_arena_4` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T22-50-39Z_arena_4_Feeding/zarr/2026-01-28T22-50-39Z_arena_4_Feeding_analysis.zarr` |
| `2026-01-28T23-07-24Z_arena_2` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T23-07-24Z_arena_2_DefaultScreen/zarr/2026-01-28T23-07-24Z_arena_2_DefaultScreen_analysis.zarr` |
| `2026-01-28T23-07-24Z_arena_3` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T23-07-24Z_arena_3_DefaultScreen/zarr/2026-01-28T23-07-24Z_arena_3_DefaultScreen_analysis.zarr` |
| `2026-01-28T23-07-24Z_arena_4` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T23-07-24Z_arena_4_DefaultScreen/zarr/2026-01-28T23-07-24Z_arena_4_DefaultScreen_analysis.zarr` |
| `2026-01-28T23-15-10Z_arena_2` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T23-15-10Z_arena_2_Feeding/zarr/2026-01-28T23-15-10Z_arena_2_Feeding_analysis.zarr` |
| `2026-01-28T23-15-10Z_arena_3` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T23-15-10Z_arena_3_Feeding/zarr/2026-01-28T23-15-10Z_arena_3_Feeding_analysis.zarr` |
| `2026-01-28T23-15-10Z_arena_4` | subject_shape | yes | yes | no | no | `/nvme1/recordings/2026-01-28T23-15-10Z_arena_4_Feeding/zarr/2026-01-28T23-15-10Z_arena_4_Feeding_analysis.zarr` |
| `sickyfish_2026_02_23_16_23_35_cam2010093` | none | no | no | no | no | `/nvme1/recordings/sickyfish_2026_02_23_16_23_35_cam2010093/zarr/sickyfish_2026_02_23_16_23_35_cam2010093_analysis.zarr` |
| `sickyfish_2026_02_23_16_23_35_cam2010094` | none | no | no | no | no | `/nvme1/recordings/sickyfish_2026_02_23_16_23_35_cam2010094/zarr/sickyfish_2026_02_23_16_23_35_cam2010094_analysis.zarr` |
| `sickyfish_2026_02_23_16_23_35_cam2010095` | none | no | no | no | no | `/nvme1/recordings/sickyfish_2026_02_23_16_23_35_cam2010095/zarr/sickyfish_2026_02_23_16_23_35_cam2010095_analysis.zarr` |
| `sickyfish_2026_02_23_16_23_35_cam2010096` | none | no | no | no | no | `/nvme1/recordings/sickyfish_2026_02_23_16_23_35_cam2010096/zarr/sickyfish_2026_02_23_16_23_35_cam2010096_analysis.zarr` |

## Registry Enumeration

The active configured registry is `/nvme1/palette_registry.sqlite`.

Evidence:

- `configs/fisheye/registry.yaml` points at `/nvme1/palette_registry.sqlite`.
- `RegistryPaths.from_env(...)` first honors `PALETTE_REGISTRY_PATH`, then the config path, then falls back to `runs/registry/palette_registry.sqlite`.
- No `PALETTE_REGISTRY_PATH` environment variable was set in this shell during the audit.

Registry-like databases found:

| Path | Role |
|---|---|
| `/nvme1/palette_registry.sqlite` | active live registry; 325 datasets, 3533 step rows; status counts after cleanup: 323 active, 1 missing, 1 inactive_smoke |
| `/groups/ahrens/ahrenslab/jeremy/zebrobot/backups/palette_registry_*.sqlite` | 10 retained live-registry backups |
| `/nvme1/palette_registry*.sqlite` | 12 local live/historical/pre-migration copies, including the active DB |
| `/nvme1/registry.sqlite` | older/dev Palette-like DB; 55 datasets, 0 step rows |
| `/groups/johnson/johnsonlab/jeremy/palette_smoke/stage_completion_smoke/stage_completion_shared_storage_20260522T001445Z.sqlite` | smoke/test Palette-like DB |
| `/home/delahantyj@hhmi.org/gitrepos/metazebrobot/zebrobot.db`, `/nvme1/zebrobot.db` | not Palette registries by table set |

Migration implication: Phase 4 must add and run a new drop-migration on the active registry before removing old migration bodies. Backups do not need to be migrated for normal operation, but any registry copy that might be replayed as a live DB later must either be migrated first or treated as historical/read-only.

## Off-Repo And Compatibility Consumers

This is not clean enough for Phase 3 deletion.

In-repo live consumers remain by design until Phase 1, including:

- `src/fisheye/analysis/eye_angle_analysis.py`
- `src/fisheye/shared/eye_geometry_source.py`
- `src/fisheye/visualization/visualize_eye_angle_overlays.py`
- `src/fisheye/tune/keypoint_failure_review.py`
- `src/fisheye/tune/keypoint_review_backend.py`
- `src/fisheye/utils/patch_keypoints_from_crops.py`
- `src/fisheye/utils/materialize_refined_eye_masks_compat.py`
- `src/fisheye/utils/refined_eye_masks_compat.py`
- `src/fisheye/tune/refined_subject_mask_review.py`

External repo hits were found in Crimson:

- `/home/delahantyj@hhmi.org/gitrepos/crimson-ui-monolith/src/zarr_loader_eye_keypoint.cpp` still loads `refined_eye_masks_runs/<latest>/masks_roi` as a fallback overlay.
- `/home/delahantyj@hhmi.org/gitrepos/crimson-ui-monolith/src/refined_keypoint_repository.cpp` still marks `eye_masks_runs` and `refined_eye_masks_runs` stale after keypoint edits.
- `/home/delahantyj@hhmi.org/gitrepos/crimson-ui-monolith/docs/crimson_eye_mask_texture_and_editing_plan.md` documents legacy refined-eye fallback behavior.

So Phase 1 Palette consumer re-point can proceed for active production registry coverage, but deleting compatibility support or the `refined_eye_masks_runs` view is **NO-GO** until Crimson is re-pointed or explicitly agrees to drop that fallback.

## Source Of Truth

Effective source distribution for the 58 active scanned zarrs:

| Effective source | Count |
|---|---:|
| `analysis/subject_shape_runs` | 48 |
| `refined_subject_masks_runs` | 4 |
| `refined_eye_masks_runs` only | 0 |
| none | 6 |

For current production-style behavior recordings, re-pointed eye-angle analysis will mostly use `analysis/subject_shape_runs`. The first 4 `2026-01-28T19-22-28Z_arena_*` recordings currently resolve from `refined_subject_masks_runs` because subject-shape geometry was not materialized there. The sleepy/sicky clipped analysis zarrs have no eye-mask source yet, so severance does not remove an existing fallback for them. The smoke-only legacy fixture is now outside active registry coverage.

## Final Decision

**GO for Palette Phase 1 re-point** across active production registry coverage. The only legacy-refined-eye-only archive was a smoke fixture and has been marked `inactive_smoke` in `/nvme1/palette_registry.sqlite`.

**NO-GO for Phase 3 deletion of compatibility support** until Crimson no longer reads `refined_eye_masks_runs` and no external notebooks/tooling are confirmed to depend on the compatibility view.
