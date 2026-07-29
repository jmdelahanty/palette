# Crop Geometry Storage Contract v1

Status: implemented for selector-ineligible refined-source shadows; not a
production selector or writer default

Date: 2026-07-29

## Purpose

This contract freezes the immutable geometry and lineage needed to extract
downstream crop pixels without persisting `roi_images` in an analysis Zarr.
Detection identity remains upstream and unchanged. A single refined-detection
snapshot may produce multiple crop runs for different purposes or crop-size
policies.

## Exact Layout

The run lives at `crop_runs/<run>` and contains exactly these arrays:

| Array | Dtype and shape | Meaning |
| --- | --- | --- |
| `instance_key` | `uint64[N]` | Preserved observation identity; not subject identity. |
| `source_refined_row_ids` | `int64[N]` | Stable row identity in the bound refined lineage. |
| `frame_indices` | `int64[N]` | Sorted acquisition-camera frame per row. |
| `source_acquisition_frame_index` | `int64[N]` | Exact full-acquisition identity mapping. |
| `frame_row_offsets` | `int64[F+1]` | CSR frame-to-row index supporting zero/one/many rows. |
| `bbox_norm_coords` | `float32[N,4]` | Authoritative refined bbox in normalized `cx,cy,w,h`. |
| `bbox_img_xyxy` | `float32[N,4]` | Exact source-camera pixel projection. |
| `centers_img_xy` | `float32[N,2]` | Exact persisted source-camera center. |
| `roi_coordinates_full` | `int32[N,2]` | Exact integer source top-left `[x,y]`. |
| `roi_sizes_full` | `int32[N,2]` | Exact positive `[width,height]`; never assumes 512. |
| `source_crop_xywh` | `float32[N,4]` | Float32 projection of the integer extraction window. |
| `bbox_roi_xyxy` | `float32[N,4]` | Refined bbox translated into crop-local pixels. |
| `source_row_signature` | `uint8[N,32]` | Exact incremental reuse signature. |

`roi_images`, `roi_images_delta`, `frame_counts`, `n_detections`,
`detection_indices`, and `source_frame_indices` are forbidden. Dense pixels
belong to keyed work packages, caches, or immutable training artifacts.

Coordinate meanings are assigned by the shared v1 coordinate-surface catalog:
normalized authority, source-camera continuous pixels/half-open edges, integer
extraction origin and extent, and ROI-local half-open edges remain distinct.
The crop schema exposes this catalog separately without changing the already
frozen v1 run-manifest bytes. Persisting it in the run envelope requires a new
manifest version rather than silently revising v1.

## Crop Policy

Every run binds a canonical-JSON/SHA-256 crop policy containing:

- purpose;
- persisted-center placement;
- NumPy ties-to-even center rounding;
- `rounded_center - floor(size/2)` top-left derivation;
- `fixed_per_run` or `variable_per_row` sizing; and
- `require_fully_contained` or `zero_outside_source_frame` padding.

The policy digest is part of crop identity and row signatures. It is never
part of `instance_key` or the source refined-detection identity.

## Refined and Pixel Authorities

The first profile requires the complete `instances` rowset of one validated
full-acquisition refined-detection v1 snapshot. The crop manifest binds its:

- run ID and run-manifest digest;
- decoded logical-content digest;
- recording identity;
- lineage UUID and snapshot UUID; and
- exact `instance_key`, refined row ID, frame, offsets, and bbox arrays.

The source-pixel authority separately binds recording/camera identity,
acquisition frame domain, dimensions, decoded `uint8` grayscale semantics, and
the digest of the external authority manifest. The shadow publisher accepts a
typed, already-proven authority.

`bind_refined_crop_source_pixel_authority()` is the future-facing strict
authority binder for a single external full-frame source video. It reopens the
direct archive metadata, requires the mirrored acquisition publication state to be
`published_canonical_v1` in `external_video_v1` mode, reloads the sealed import
ownership and acquisition-camera frame, resolves the exact source-video
locator, and recomputes the live `stat_v1` fingerprint. It then binds the
`orange_mono_pynvvc_luma_uint8_v1` full-frame decode policy into the crop
authority digest. Recording identity, camera identity, `F`, width, and height
must exactly match the refined handoff. Materialized source arrays, acquisition
crop videos, and clipped collections intentionally require separate typed
authority binders. These are contract boundaries, not compatibility adapters:
they do not probe dtypes, translate aliases, infer identities, or fall back to
another source.

## Manifest and Publication Gate

The exact envelope is persisted at:

```text
crop_runs/<run>/zarr.json.attributes.run_manifest
```

It contains the logical schema, concrete dimensions, crop policy, storage
plans, refined and pixel authorities, signature spec, per-array decoded
digests, metadata-declaration digest, completion state, and selector
eligibility.

The gate fails closed unless:

1. the envelope and every nested field set match the frozen builders;
2. the outer, logical-content, policy, source, pixel, and signature digests
   agree;
3. decoded crop arrays satisfy all geometry, identity, and CSR invariants;
4. row signatures recompute exactly;
5. decoded refined source arrays and manifest match the crop binding;
6. direct and consolidated metadata are equivalent;
7. physical declarations match `StoragePlan` and the exact codec chain; and
8. every shard write has whole-shard single-writer ownership.

Metadata declaration digests retain attributes. Only the circular root
`run_manifest` attribute and representational `consolidated_metadata` envelope
are excluded.

## Physical Policy

The initial unpromoted crop profile is `published_http_v1`:

- approximately 1 MiB uncompressed inner chunks derived from bytes per row;
- approximately 32 MiB immutable indexed shards where the array is large
  enough;
- `frame_row_offsets` classified `EAGER`;
- all row-aligned identity/geometry/signature columns classified `WINDOWED`;
- Zarr v3 with the shared bytes + Zstd/CRC and indexed-sharding contract; and
- complete trailing row axes in every chunk and shard.

On a Sleepyfish-sized `N=1,187,087`, `F=1,188,000` plan, the 13 arrays are
estimated at 14 payload objects. `source_row_signature` is the only two-shard
column; the remaining arrays each fit in one large shard.

## Current Safety Boundary

The standalone writer can create only a fresh child below `/tmp`,
`.palette_scratch`, or `.palette_benchmarks`. The production-candidate boundary
`publish_crop_geometry_production_candidate()` additionally:

1. binds only the approved authoritative refined-detection snapshot;
2. binds and re-verifies the exact published external-video pixel authority;
3. materializes and fully validates a geometry-only crop run on bounded
   node-local scratch;
4. atomically imports the immutable run into `crop_runs/<run>`;
5. rebuilds the exact run manifest after publisher transaction metadata exists;
6. reconsolidates and revalidates the complete imported publication; and
7. proves root and crop-family selector attributes are unchanged.

It does not register an artifact, update `latest`, activate a selector, change
a production default, or replace an existing run. A post-import failure is
retained as an owner-bound selector-ineligible failed child rather than being
made authoritative.

Before production integration:

- [x] implement and validate the external full-frame source-pixel authority
      binder;
- [x] implement node-local materialization and atomic selector-ineligible
      production-candidate import;
- [x] implement a benchmark-only package builder that exercises the production
      candidate path and preserves publication/read handoff evidence;
- [ ] obtain parallel Palette producer/DAG review of this exact contract;
- [x] publish a small immutable canary outside production selectors;
- [x] pass the Crimson canonical-v3/refined-v2/crop-v2 coordinate archive gate;
- [ ] test Palette pixel materialization and downstream keypoint/mask readers;
- [x] benchmark selector-ineligible publication and record object counts;
- [ ] benchmark representative row/window reads in Crimson;
- [ ] add a typed purpose/profile selector with guarded activation; and
- [ ] migrate production writers only after downstream completeness passes.

The Crimson coordinate canary passed at implementation commit
`ce478c7d13d2f870e6c711308090e28364872602` and evidence commit `4100719`.
The supplied evidence SHA-256 is
`9918615e142a1f946eb98865f46e264cacff23a2885e008ce0030d87efc6fd7d`.
Crimson validated exact typed opens, CSR offsets, lineage, and ROI-to-source
camera transforms; this closes the coordinate-consumer gate but does not
activate a Palette crop selector.
