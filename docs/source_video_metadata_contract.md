# Source-video metadata and locator contract

Schema ID: `palette.source_video_metadata.v2`

## Purpose

`source_video_metadata` is the canonical structured description and locator for
new single-video recording Zarrs. Its authoritative locator is recording-relative
whenever the video belongs to the recording directory. Absolute paths remain
compatibility mirrors during migration; they are not stable video identity.

This contract applies to new writes. It does not authorize rewriting historical
Zarrs in place.

## Single-video schema

The object is stored in the root Zarr attributes:

```json
{
  "schema_id": "palette.source_video_metadata.v2",
  "layout": "single_video",
  "locator": {
    "kind": "recording_relative",
    "relative_path": "cams/Cam2010093_recording.mp4"
  },
  "source_video": "Cam2010093_recording.mp4",
  "source_path": "/groups/.../recording/cams/Cam2010093_recording.mp4",
  "width": 4512,
  "height": 4512,
  "total_frames": 143447,
  "fps": 100.0,
  "duration_seconds": 1434.47,
  "codec": "hevc",
  "pix_fmt": "yuv420p"
}
```

For `locator.kind = recording_relative`:

- `relative_path` must be non-empty and relative;
- `..` traversal is forbidden;
- the resolved path must remain underneath root `recording_path`;
- consumers may derive `recording_path` from a canonical
  `<recording>/zarr/<archive>.zarr` location only when the root attribute is absent.

`locator.kind = absolute` is allowed only for a source genuinely outside the
recording directory. It requires an absolute `locator.path` and is less relocatable.

## Authority and compatibility

For metadata carrying this schema ID, `source_video_metadata.locator` is
authoritative. During migration, writers also emit these mirrors:

- root `source_video_path`;
- root `source_path`;
- `source_video_metadata.source_path`;
- `raw_video.source_path`.

Historical `video_source_path` and `raw_video.source_video_path` mirrors are
also validated when present, but new root writers do not need to create them.

Every populated mirror must resolve to the same file as the canonical locator.
Readers and validators must fail closed on disagreement rather than silently select
one path. The nested `source_path` is retained only for compatibility and may be
removed in a later schema after all consumers use the shared resolver.

Archives without `palette.source_video_metadata.v2` remain legacy-readable through
the existing mirror fields. A legacy resolver must also reject conflicts between
populated mirrors.

## Identity and relocation

Technical stream facts—geometry, FPS, frame count, codec, pixel format and
colorimetry—describe the video independently of its current storage root.

The current `stat_v1` fingerprint includes path and modification time. When embedded
as `file_fingerprint`, it is explicitly marked `relocation_stable = false`; it is a
cheap swapped-file diagnostic, not durable content identity. A future content hash
may provide relocation-stable identity without changing locator semantics.

Relocating a recording should update root `recording_path` and regenerate the
absolute compatibility mirrors atomically. A recording-relative canonical locator
does not change.

Historical environment, model and run-provenance paths are not operational video
locators and must not be rewritten by relocation maintenance.

## Multi-video and clipped layouts

A single root video object is not authoritative for clipped or multi-video
collections. Those layouts must resolve the source through their collection manifest
and recording frame index so every row maps to a specific clip/video. The
single-video resolver rejects non-`single_video` v2 layouts.

## Shared Palette API

`fisheye.shared.source_video_metadata.resolve_source_video` implements the contract:

- v2 canonical-locator precedence;
- legacy mirror fallback;
- path traversal protection;
- conflict detection;
- optional existence validation;
- explicit rejection of collection layouts.

Adoption is staged. Existing consumers continue using compatibility fields until
they are individually migrated and parity-tested against this resolver.
Current migration status is tracked in `docs/source_video_metadata_migration.md`.

## Zarr representation

The contract is attribute-based and works for Zarr v2 and v3. In Zarr v3 the root
attributes are stored inside `zarr.json`; this contract does not introduce `.zattrs`.
