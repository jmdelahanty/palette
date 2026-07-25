# Recording Zarr Dish-Mask and Instance Readiness

Generated: 2026-05-04

## Scope

- Root scanned: `/nvme1/recordings`
- Normal recording pairs scanned: 52 `_training.zarr` / `_analysis.zarr` pairs
- Excluded artifact: `/nvme1/recordings/smoke/2026-01-28T19-22-28Z_arena_1_DefaultScreen_geometry_smoke_analysis.zarr`

The smoke Zarr is not a recording pair and is not required to carry a dish mask.

## Dish Masks

Policy:

- Copy the reviewed circle mask from each paired `_training.zarr` to the matching `_analysis.zarr`.
- Never overwrite an existing analysis mask.
- Existing circle masks pass automatically only when center-x, center-y, and radius differences are all `< 5` px.
- Anything else is listed for review.

Results:

- Normal paired recordings: 52
- Missing analysis masks filled from training masks: 50
- Metadata reconsolidated after writes: 50
- Existing analysis masks left untouched: 2
- Write errors: 0
- Review items: 1

Review item:

| Recording | Reason | Training Mask | Existing Analysis Mask |
|---|---|---|---|
| `2026-01-28T23-15-10Z_arena_2_Feeding` | Existing analysis mask radius differs by 6 px, exceeding the `< 5` px tolerance. It was not overwritten. | center `[310, 320]`, radius `310` | center `[310, 320]`, radius `304` |

## Detection and Keypoint Structure

Audit result across the 52 normal analysis Zarrs:

- `detect_runs` present: 52 / 52
- `refined_detect_runs` present: 52 / 52
- Latest refined detect run has canonical `instances/` arrays: 52 / 52
- `keypoints_runs` present: 52 / 52
- `refined_keypoints_runs` present: 52 / 52
- Latest crop run has `source_refined_row_ids` and `source_detect_row_index`: 52 / 52

No detection, keypoint, refined-detect instances, or crop row-lineage gaps were found.

## Notes

The detection/keypoint readiness audit matches the canary-style structural migration outcome: these archives already expose modern refined-detection `instances/` and crop row-lineage surfaces, so no additional migration was applied in this pass.
