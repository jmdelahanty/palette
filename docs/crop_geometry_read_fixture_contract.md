# Crop Geometry Publication/Read Fixture

Date: 2026-07-29

Status: implemented; persistent 23k-frame publication pending

## Purpose

This package exercises the future-facing crop publication boundary without
modifying a production archive or overlapping production DAG integration. It
exists for Palette publication measurements and Crimson's backend-neutral
crop-v2 read benchmark.

The package is integration evidence, not storage-profile promotion evidence.
It contains geometry and lineage only; it never stores `roi_images`.

## Construction

`fisheye.diagnostics.publish_crop_geometry_read_fixture` performs the following
fail-closed sequence:

1. opens and fully validates one selector-ineligible refined-v2 canary;
2. content-hashes and copies only that small refined archive to node-local
   scratch;
3. adds an exact benchmark-local external-video acquisition authority;
4. makes the copied refined run authoritative only inside the benchmark archive;
5. copies that minimal seed into a hidden shared package;
6. invokes `publish_crop_geometry_production_candidate()` so the crop run is
   materialized locally, validated, atomically imported, reconsolidated, and
   validated again through the production-candidate path;
7. records per-phase and per-array write timing, copy timing, hashes, storage
   statistics, source bindings, and final validation in `handoff_manifest.json`;
8. verifies that the original refined canary did not change; and
9. atomically renames the hidden package into its final benchmark path.

Failures remain hidden under a unique `.partial.*` package with a failure
receipt. A complete destination is never overwritten.

## Selection Boundary

The archive root remains:

- `benchmark_only=true`;
- `selector_eligible=false`; and
- `registry_registered=false`.

The copied refined run is selector-eligible and approved only inside this
isolated archive because the strict crop binder requires an approved refined
authority. That local authority is provenance for fixture construction, not a
production selection. The published crop run remains
`stage_selector_eligible=false`, and `crop_runs` receives no `latest`,
`latest_complete`, or authoritative pointer.

## Planned Representative Publication

Seed archive:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/
coordinate_catalog/integration/
20260128_coordinate_catalog_crimson_20260728_v1/refined.zarr
```

Seed run:

```text
refined_detect_runs/refined_detect_coordinate_catalog_crimson_20260728_v2
```

The seed contains 23,287 camera frames and 22,926 refined rows. The crop policy
is fixed 512×512 geometry with zero padding outside the source frame. The source
video is referenced for authority verification but not copied.

Invocation template:

```bash
mkdir -p /tmp/palette-crop-read-fixture-work
scripts/py -m fisheye.diagnostics.publish_crop_geometry_read_fixture \
  --source-refined-zarr /groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/coordinate_catalog/integration/20260128_coordinate_catalog_crimson_20260728_v1/refined.zarr \
  --source-refined-run refined_detect_coordinate_catalog_crimson_20260728_v2 \
  --source-video /nvme1/recordings/2026-01-28T19-22-28Z_arena_1_DefaultScreen/cams/Cam2010093_2026-01-28T19-22-28Z_arena_1.mp4 \
  --recording-path /nvme1/recordings/2026-01-28T19-22-28Z_arena_1_DefaultScreen \
  --camera 2010093 \
  --fps 60 \
  --codec hevc \
  --pixel-format unknown \
  --destination /groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/crop_geometry/publication_reads/20260128_crop_v2_publication_read_20260729_v1 \
  --work-root /tmp/palette-crop-read-fixture-work \
  --crop-run-id crop_geometry_publication_read_crimson_20260729_v2 \
  --crop-size 512
```

## Crimson Workload

Crimson should use the explicit archive and run in the handoff and:

1. validate the consolidated v2 manifest and all 13 exact declarations without
   dtype probing or compatibility fallback;
2. retain `frame_row_offsets` after exactly one whole-array read;
3. benchmark cold open, first usable frame, warm random frames, 70-frame
   sequential windows, cancellation, and stale-result prevention;
4. read the hot geometry/identity columns concurrently;
5. report file reads, transferred bytes, latency percentiles, cache behavior,
   and peak RSS; and
6. prove that the geometry-only reader never requests `roi_images`.

The 23k-frame result validates publication/read integration and instrumentation.
A full-duration fixture should be requested only if the smaller checkpoint
reveals scaling questions that cannot be answered analytically.

