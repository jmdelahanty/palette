# Coordinate Catalog Selector-Ineligible Canary

Date: 2026-07-28

Status: passed Palette and Crimson archive gates; selector-ineligible

## Result

Palette published the three-artifact coordinate-catalog canary accepted by the
Crimson cross-language review. The package is immutable, benchmark-only, and
selector-ineligible. It did not update a registry, selector, writer default, or
production archive.

Palette publication commit:
`7a710276beea3037a457f4bfbb5be9f0525de0dc`

Crimson review commit:
`ae211f0551f9d1a4b82da6b06b2202750725e6d1`

Crimson review document SHA-256:
`6663be812021589a7abf7cff2661e263524b7f3b2d1f276875f0c549b2f4d42d`

Server package:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/
coordinate_catalog/integration/
20260128_coordinate_catalog_crimson_20260728_v1
```

macOS package:

```text
/Volumes/johnsonlab/jeremy/recordings/.palette_benchmarks/
coordinate_catalog/integration/
20260128_coordinate_catalog_crimson_20260728_v1
```

Handoff manifest SHA-256:
`21ccf119dfe7910e6c2cce7b027a9318e7b3c8cec4702363fbc3d4460775c4d3`

Handoff payload digest:
`c4c6b186386ea8711b7d5517ae369d49d187b26228e0d6c2c6cc6228bab98692`

The payload digest was independently recomputed after publication and matches.

## Artifacts

| Artifact | Run | Manifest | Manifest digest | Catalog digest | Objects / apparent bytes |
| --- | --- | ---: | --- | --- | ---: |
| `canonical_source.zarr` | `detect_coordinate_catalog_crimson_20260728_v3` | canonical detection v3 | `960dc83e7fb47d27ed3577b3f9ea2bbe7d490b18b4c979b11f922f0523845e2d` | `337613bd6e5f283eef9d6a89c14766d50c5b6863dea584f7568b90bb1d936733` | 22 / 820,501 |
| `refined.zarr` | `refined_detect_coordinate_catalog_crimson_20260728_v2` | refined detection v2 | `df1e05e9a435fcbd3ff5403c666f6d622276e66dd32222300e1b358b47e86881` | `75656615ecd32a215f6b4148a01c9ef75e96b8d7aa6bf9fb8a7d21757fa7a2ed` | 59 / 1,984,826 |
| `crops.zarr` | `crop_geometry_coordinate_catalog_crimson_20260728_v2` | geometry-only crop v2 | `a4f42a823b1ca81cc69936fe0a374a59b75f42a8781a159a6a56f02e25a463f6` | `e9ce640761ee1de4a6edd72695968bd66ae2fcbdd09d7d2c902450904f6ddfec` | 30 / 1,833,315 |

The canonical and refined artifacts use
`detection_published_access_aware_v1`. The crop artifact uses
`published_http_v1`. Every root has inline consolidated metadata; every root
and run family declares `benchmark_only=true` and
`selector_eligible=false`. The run families have no `latest`,
`latest_complete`, or authority pointer.

## Source And Identity

The canary contains 23,287 camera frames at 4512×4512. It references, but does
not copy, camera `2010093` video content:

```text
/nvme1/recordings/2026-01-28T19-22-28Z_arena_1_DefaultScreen/
cams/Cam2010093_2026-01-28T19-22-28Z_arena_1.mp4
```

Video content SHA-256:
`b40ad6595dd32a910509dbbe55ae2179f2b462bb023cc97f690ca9161bd6171f`

Current video size: 11,860,222,428 bytes.

Pixel-authority document digest:
`b0c09c9b21d5f3dceaa63911758d680718c268e5e1cb067a918a75f1fea8025e`

The recording manifest SHA-256 is
`c47f17fa8bdf67688f77c00b39b8d34dd669e51dcc3d00b1a7797e6d9aec4336`.
Its embedded historical keyframe-check file size predates the current video
file, so the canary does not treat that old size as content identity. It binds
the current video through the independently computed content SHA-256, size,
and modification timestamp while retaining the recording manifest association.

This historical source-audit table predates durable `instance_key` values. The
first publication attempt failed closed before copying anything. Palette then
used the explicit historical migration option and records the resulting
operation in the handoff:

```text
source_detections/instance_key
mint_recording_frame_bbox_class_v1
row_count = 22,938
```

The migration is deterministic and belongs only to this historical conversion;
modern producers must already persist durable keys.

## Crop And Coordinate Samples

The geometry-only crop policy is fixed 512×512, centered on persisted
`centers_img_xy`, using NumPy ties-to-even rounding and zero padding outside the
source frame. Its policy digest is
`91ed7be0a0703a4f0fb50bbdc25e282e97a10910984dee1ea123f86833c76003`.

The handoff contains exact row-zero samples for both required conversions:

- normalized `[cx,cy,w,h]` detection to source-camera pixel box and center;
- ROI-local pixel box plus rowwise integer crop origin to source-camera pixel
  box.

Both samples reproduce the persisted derived arrays with exact float32 equality
and zero maximum absolute error. This real camera source is square. Crimson's
producer-generated cross-language fixture remains the non-square-dimension
evidence.

The real source has empty frames but no multi-row frames: canonical offsets
contain 349 empty frames; refined presentation offsets contain 361. The frozen
`[2,0,1,3]` Palette/Crimson fixture remains the authoritative multi-row and
manual-addition gate.

## Publication And Validation

The tracked publisher materialized all artifacts under local `/tmp`, validated
them, copied them to a unique partial package on `/groups`, compared every file
hash, reopened the copied consolidated stores, wrote the strict handoff, and
atomically renamed the package into place.

Validation completed with:

- exact v3/v2/v2 coordinate-catalog digests;
- zero canonical, refined, or crop publication errors;
- direct/consolidated metadata equivalence;
- exact local/copy/final artifact-tree equality;
- source metadata unchanged before and after publication;
- source video size and modification timestamp unchanged;
- strict finite JSON and matching handoff payload digest; and
- no partial package or local scratch residue.

## Crimson Archive Gate

Crimson passed the gate using implementation commit
`ce478c7d13d2f870e6c711308090e28364872602` and evidence commit `4100719`.
The supplied evidence SHA-256 is
`9918615e142a1f946eb98865f46e264cacff23a2885e008ce0030d87efc6fd7d`.

The consumer validated canonical-v3, refined-v2, and crop-v2 coordinate
manifests, exact typed TensorStore opens, CSR offsets, lineage, and
ROI-box-to-source-camera transforms while retaining the legacy crop adapter as
a separate compatibility surface. The normalized projection differed by
`0.000109` pixel after float32-to-double promotion, within the frozen
`0.001`-pixel tolerance; the ROI transformation was exact. This gate changes no
Palette selector or production writer default.
