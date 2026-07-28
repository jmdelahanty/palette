# Crop Geometry Storage Contract v1

Status: implemented for selector-ineligible refined-source shadows; not a
production selector or writer default

Date: 2026-07-28

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
typed, already-proven authority. A production adapter must reopen and validate
the external recording/video manifest rather than accepting an arbitrary
caller-provided digest.

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

The implemented publisher can create only a fresh child below `/tmp`,
`.palette_scratch`, or `.palette_benchmarks`. It does not register an artifact,
update `latest`, activate a selector, or write into an analysis archive.

Before production integration:

- [ ] implement and validate the external source-pixel authority adapter;
- [ ] obtain parallel Palette producer/DAG review of this exact contract;
- [ ] publish a small immutable benchmark canary outside production selectors;
- [ ] test Palette pixel materialization and downstream keypoint/mask readers;
- [ ] benchmark publication and representative row/window reads;
- [ ] add a typed purpose/profile selector with guarded activation; and
- [ ] migrate production writers only after downstream completeness passes.
