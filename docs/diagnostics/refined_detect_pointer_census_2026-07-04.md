# Refined detect pointer census

Date: 2026-07-04

## Purpose

This read-only census checked whether rerouting crop's refined/auto detect-source
selection from `refined_detect_runs.attrs["latest"]` to authoritative run
resolution would change which refined detect run existing recordings consume.

The implementation premise was verified against real metadata before code was
changed because `detect_review_status_latest` is a legacy authority pointer,
while crop's refined path historically selected `latest`.

## Method

The census read `zarr.json` metadata only. It compared these parent attrs on
`refined_detect_runs` groups:

- `authoritative_run`
- `detect_review_status_latest`
- `latest`
- `latest_complete`

Scanned sources:

- active registry zarr roots from
  `/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite`
- historical local recording roots under `/nvme1/recordings`

No Zarr stores were opened for mutation.

## Results

After de-duplicating parent paths, the census found 186
`refined_detect_runs` parents.

| Classification | Count |
| --- | ---: |
| `MATCH` | 184 |
| `NO_POINTERS` | 2 |
| mismatch | 0 |

Pointer combinations:

| Parent attrs present | Count |
| --- | ---: |
| `detect_review_status_latest + latest` | 113 |
| `latest + latest_complete` | 70 |
| `latest` only | 1 |
| none | 2 |
| `authoritative_run` present | 0 |

No observed parent had:

- `detect_review_status_latest != latest`
- `latest != latest_complete` when both were present
- an `authoritative_run` conflict

The two no-pointer parents were old/smoke-style stores:

- `/groups/johnson/johnsonlab/jeremy/palette_smoke/sleepyfish_2026_05_05_17_45_30_cam2010093/zarr/sleepyfish_2026_05_05_17_45_30_cam2010093_analysis.zarr/refined_detect_runs`
- `/groups/johnson/johnsonlab/jeremy/recordings/sleepyfish_2026_05_05_17_45_30_cam2010095/zarr/sleepyfish_2026_05_05_17_45_30_cam2010095_analysis.zarr/refined_detect_runs`

## Decision

The census found zero cases where the legacy reviewed-run pointer disagreed with
the run that crop currently selects. Therefore the design-correct precedence is
behavior-preserving for observed data:

`authoritative_run -> detect_review_status_latest -> latest_complete`

This justifies rerouting crop's refined/auto detect-source path through
`RunResolution.AUTHORITATIVE` with a scoped legacy bridge for
`detect_review_status_latest` on refined-detect parents.

The broader writer surface that still writes `detect_review_status_latest`
remains a follow-up. This slice only bridges the legacy pointer for reads,
reroutes crop selection, and updates `utils/backfill_detect_review_status.py` so
that utility no longer directly writes the old bypass pointer.
