# Crimson Detection Bounding-Box Read Contract
<!-- contract-meta
version: 7
status: future-normal
last_verified: 2026-07-19
-->

This document is Palette's implementation-facing mirror of the authoritative
cross-repository contract:

```text
agent-contracts/palette-crimson/detect_bbox_read.md
```

The future-normal Crimson path reads only sealed canonical raw detections from
`detect_runs/<run>`. Historical refined/manual layouts and unbound raw runs are
not normal fallbacks. A future recording must be readable without invoking a
legacy adapter.

Presentation viewport coordinates are ephemeral Crimson renderer state. They
must not be persisted as Palette coordinate authority.

## Candidate discovery and pinning

An explicit run name or `detect_runs.attrs["latest_complete"]`/`["latest"]` may
discover a candidate. The selector does not authorize it. Crimson must reopen
the exact `detect_runs/<run>` child and require:

```text
coordinate_contract == "canonical_v2"
palette_run_completion_contract is supported
palette_run_completion_status == "complete"
stage_selector_eligible is true
```

Lexicographic selection, refined-to-raw fallback, and use of a detached child
handle are forbidden. A selected child that is missing, replaced, incomplete,
ineligible, or changed during the read is unavailable.

## Required arrays and identity

One nonempty canonical run contains:

```text
instance_key                         uint64  (N,)
source_acquisition_frame_index       int64   (N,)
frame_indices                        int32   (N,)
bbox_norm_coords                     float64 (N,4)
bbox_img_xyxy                        float64 (N,4)
centers_img_xy                       float64 (N,2)
scores                               float32 (N,)
class_ids                            int32   (N,)
frame_counts                         int32   (F,)
n_detections                         int32   (F,)
```

`instance_key` is observation identity. Acquisition frame is a separate time
mapping; row offset is not frame number. `frame_indices` is an exact
compatibility alias and cannot replace the typed temporal authority.

`frame_counts` and `n_detections` must equal the bincount of the acquisition
frame mapping over the sealed full frame domain and sum to `N`. Zero-observation
runs require the canonical full-domain empty-observation proof; empty arrays by
themselves are insufficient.

## Coordinate surfaces

Every geometry array owns `coordinate_descriptor` and
`coordinate_descriptor_sha256`. Names and numerical ranges carry no coordinate
meaning.

### Source-camera boxes

`bbox_img_xyxy` is the preferred overlay surface and must declare:

```text
profile_id == "source_camera_image_px.top_left_y_down.v1"
geometry_type == "bbox_xyxy"
components == ["x_min", "y_min", "x_max", "y_max"]
component_units == ["px", "px", "px", "px"]
pixel_convention == "pixel_edge_half_open"
source_camera_overlay.status == "direct"
```

The box uses half-open edge geometry. `x_max == width` and
`y_max == height` are valid; scientific validation must not clamp them to
`width - 1` or `height - 1`.

The bbox authority is distinct from the continuous source-camera point frame.
Both frames bind the same exact acquisition extent, but a reader must not
substitute one convention for the other merely because their numeric extents
match.

### Normalized boxes

`bbox_norm_coords` is a source-camera-normalized `[cx, cy, width, height]`
sibling, not detector-model-input geometry. It must declare the controlled
source-camera-normalized profile, continuous convention, and an exact
direction-labelled normalized-to-source-camera transform. Crimson must not
multiply it by root, inference, video-widget, or viewport dimensions chosen
independently of that authority.

### Source-camera centers

`centers_img_xy` must declare source-camera pixels, `point_xy`, continuous
convention, and direct-overlay status. Its derivation proves exact midpoint
equality to `bbox_img_xyxy` under the same observation identity and explicitly
crosses from the half-open bbox-edge authority to the continuous point
authority.

### Detector result-space lineage

`detection_backend_result_projection` is required. It binds the exact
fingerprinted model artifact, decoder/backend identity, validated result count,
uniform `result.orig_shape`, persisted normalized bbox payload, and the exact
result-pixel-to-normalized/source-camera matrices. Palette validates that every
YOLO result's `orig_shape` equals its corresponding runtime input shape before
normalization.

This record is deliberately not a guessed network letterbox authority. When
the backend's internal network preprocessing is not persisted exactly, the
record labels it unavailable and excludes it from coordinate projection
authority. Crimson consumes the validated result-space projection and must not
reconstruct a model-input transform from `imgsz` or resolution ratios.

## Required validation boundary

Before returning values, Crimson must validate the logical equivalent of
`fisheye.shared.observation_coordinate_publication.load_persisted_detection_observation_geometry`,
including:

- acquisition camera, source-video identity, continuous point frame,
  half-open bbox-edge frame, and their common exact extent;
- the validated detector result-space projection and exact model fingerprint;
- the source-camera-normalized frame and directed transform;
- observation identity and acquisition-time records;
- instance-key derivation and exact observation cardinality;
- bbox projection and center derivations;
- every geometry payload, descriptor, digest, and referenced record;
- the full-domain declaration for a zero-observation run.

The reader copies the requested arrays, reopens/revalidates the pinned child
and exact source graph, and only then returns ordinary arrays to the renderer.
This boundary validation does not require rendering or scientific kernels to
carry metadata dictionaries internally.

## Refined and manual detections

Current `refined_detect_runs/<run>` layouts do not provide the immutable
successor, exact row identity, complete source-camera geometry, descriptors,
and publication seal required by the future-normal contract. Crimson therefore
reports canonical refined detections as unavailable. It must not silently fall
back to raw detections after an explicit refined selection.

The future write boundary is
`agent-contracts/palette-crimson/refined_detect_manual.md`: Crimson submits a
logical edit request; Palette owns validation, recomputation of image and
normalized siblings, immutable successor publication, sealing, and activation.
Direct in-place Crimson mutation cannot preserve canonical authority.

Historical `instances/`, sparse manual/interpolated/filtered groups, legacy
normalized boxes, reason arrays, and old chunk layouts may be exposed only by
an explicitly selected, visibly labelled read-only mode such as
`unverified_legacy_inspection_only`. That mode cannot enter normal selectors,
feed canonical scientific publication, or mint missing coordinate metadata.

## Failure behavior

Normal reading fails closed for missing or stale lifecycle gates, identity or
time evidence, arrays, descriptors, digests, frame extents, source-video
binding, transform direction, derivation, cardinality, or empty-run proof. It
also fails closed when model-input, ROI, normalized, or viewport geometry is
presented as direct source-camera geometry.

Renderer clipping and source-camera-to-viewport scaling are allowed only after
validation and remain ephemeral.

## Required tests

Focused Palette and Crimson fixtures cover:

1. multiple observations in one acquisition frame;
2. plausible ROI/model/viewport values rejected as source-camera boxes;
3. unequal source, model-input, and display extents;
4. a non-self-inverse transform used in the wrong direction;
5. stale descriptors, payloads, row keys, frame counts, or source records;
6. half-open boxes whose maximum edge equals the reference extent, with
   continuous centers bound to the separate point frame;
7. proved versus unproved zero-observation runs;
8. exact-child replacement or mutation during read;
9. missing, stale, or dimension-mismatched backend result-space lineage;
10. explicit refined/manual selection failing without raw fallback; and
11. a future recording read without any legacy adapter or persisted viewport
    coordinates.

## Related documents

- `docs/video_pixel_model_input_contract.md`
- `docs/coordinate_metadata_framework.md`
- `docs/keypoint_refined_coordinate_space_incident_2026-03-04.md`
- `docs/crimson_refined_detect_manual_contract.md`
- `docs/clipped_recording_consumer_mapping_contract.md`
