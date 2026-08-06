# Citrus stimulus coordinate output contract

This is the future-write contract accepted by Palette's strict stimulus H5
importer. Citrus should emit either a renderer-only snapshot or a canonical
coordinate-bearing chaser rowset. It must not use a group name to imply
scientific coordinates when it has only startup renderer metadata.

## Renderer-only sessions

When no scientific coordinate arrays exist, Citrus must omit
`/stimulus_coordinates` and write this exact attribute-only tree:

```text
/stimulus_renderer_snapshot
  attrs:
    schema_id = "citrus.stimulus_renderer_snapshot"
    schema_version = 1
    capture_phase = "experiment_start_after_arena_initialization"
  /arena_<positive integer>
    attrs:
      active_stimulus_mode = <non-empty string>
      texture_width_px = <positive integer>
      texture_height_px = <positive integer>
      texture_origin = "top_left"
    /custom_coordinates
      attrs:
        texture_center_x = <finite value within [0, width]>
        texture_center_y = <finite value within [0, height]>
```

The snapshot root has exactly the three listed attributes, exactly one arena
group, no datasets, and no extra children or attributes. This metadata is a
renderer description, not a trajectory or accepted scientific coordinate
surface.

## Coordinate-bearing chaser sessions

The relevant H5 tree is:

```text
/tracking_data/chaser_states                         structured dataset [N]
/tracking_data/stimulus_state_key                   <i8 [N,2]
/tracking_data/source_acquisition_frame_index       <i8 [N]
/tracking_data/target_source_acquisition_frame_index <i8 [N]
/tracking_data/target_source_acquisition_frame_valid bool [N]
/calibration_snapshot/arena_geometry                attribute-bearing group
/stimulus_renderer_snapshot                         renderer metadata only
```

`/tracking_data/chaser_states` must carry:

```text
schema_id = "citrus.tracking.chaser_states"
schema_version = 5
row_identity_contract = <closed record>
row_identity_contract_sha256 = <sha256>
coordinate_descriptor = <canonical descriptor-v2 record>
coordinate_descriptor_sha256 = <sha256>
coordinate_surface_manifest = <closed record>
coordinate_surface_manifest_sha256 = <sha256>
```

Do not also write the legacy attributes `coordinate_frame`,
`coordinate_units`, `coordinate_origin`, `position_fields`,
`x_axis_direction`, `y_axis_direction`, or `pixel_convention`.

### Surface manifest

The manifest is closed and must classify every field in the structured dtype
exactly once as `row_identity`, `coordinate_component`, or `non_spatial`:

```json
{
  "schema_id": "palette.columnar_coordinate_surface_manifest",
  "schema_version": 1,
  "coordinate_fields_complete": true,
  "field_classifications": {
    "chaser_index": "row_identity",
    "stimulus_frame_num": "row_identity",
    "chaser_pos_x": "coordinate_component",
    "chaser_pos_y": "coordinate_component",
    "target_pos_x": "coordinate_component",
    "target_pos_y": "coordinate_component",
    "target_clamped_pos_x": "coordinate_component",
    "target_clamped_pos_y": "coordinate_component"
  },
  "row_identity_fields": ["chaser_index", "stimulus_frame_num"],
  "surfaces": [
    {
      "array_name": "chaser_position_xy",
      "semantic_role": "chaser_position",
      "component_fields": ["chaser_pos_x", "chaser_pos_y"]
    },
    {
      "array_name": "target_position_xy",
      "semantic_role": "target_position",
      "component_fields": ["target_pos_x", "target_pos_y"]
    },
    {
      "array_name": "target_clamped_position_xy",
      "semantic_role": "target_clamped_position",
      "component_fields": ["target_clamped_pos_x", "target_clamped_pos_y"]
    }
  ]
}
```

The abbreviated `field_classifications` object above illustrates the spatial
and identity fields. The actual object must additionally list every other
structured field as `non_spatial`. Fields whose names end in `_mm` remain
non-spatial metadata unless a separate accepted physical-world authority is
implemented.

### Coordinate descriptor

Use Palette's canonical descriptor-v2 shape with these exact semantics:

- profile: `arena_relative_canvas_px.top_left_y_down.v1`
- geometry: `point_xy`, components `x`, `y`
- component units: `px`, `px`
- space: `arena_relative_canvas_px`
- origin: `arena_top_left`
- positive directions: `+x right`, `+y down`
- pixel convention: `continuous`
- reference extent: the sealed arena pixel-frame authority described below
- source-camera overlay status: `not_suitable`, with no direct overlay
  transform references

The coordinates are valid on the projected display/behavior surface. They are
not accepted `world_mm` coordinates. Camera-pixel transformation can be
performed through the separately sealed camera-to-canvas homography chain.

### Stable row identity

`/tracking_data/stimulus_state_key` must contain exact little-endian signed
int64 values:

```text
stimulus_state_key[i] = [
  chaser_states[i].chaser_index,
  chaser_states[i].stimulus_frame_num
]
```

The key must be unique, nonnegative, shape `[N,2]`, and have exactly these four
attributes:

```text
row_identity_key
row_identity_key_sha256
row_identity_contract_ref = "@row_identity_contract"
row_identity_contract_sha256
```

The owning row contract uses schema `palette.row_identity_contract` version 1,
domain `stimulus_state`, mode `stimulus_state_key`, leading dimension `N`, one
key-array record for `stimulus_state_key`, components
`["chaser_index", "stimulus_frame_num"]`, dtype `<i8`, shape `[N,2]`, and
`unique=true`. The key-array schema is `palette.row_identity_key_array`
version 1.

### State-to-acquisition mapping

`/tracking_data/source_acquisition_frame_index` must be exact `<i8 [N]`.
Values are zero-based indices in the authoritative full-camera recording,
nonnegative, and less than `source_total_frames`. Citrus must derive this from
Orange's explicit acquisition identity. It must never reinterpret
`triggering_camera_frame_id` as an acquisition index.

The array has only these two attributes:

```text
source_acquisition_mapping_record = <record below>
source_acquisition_mapping_record_sha256 = <canonical record sha256>
```

```json
{
  "schema_id": "citrus.stimulus_source_acquisition_mapping",
  "schema_version": 1,
  "mapping_method": "explicit_per_stimulus_state_v1",
  "source_rowset_ref": "/tracking_data/chaser_states",
  "source_row_identity_ref": "/tracking_data/stimulus_state_key",
  "source_row_identity_sha256": "<exact key-array content sha256>",
  "source_row_identity_contract_sha256": "<row contract sha256>",
  "acquisition_recording_id": "example_full_recording_name",
  "acquisition_camera_id": "example_camera_serial",
  "source_total_frames": 10,
  "target_domain": "acquisition_frame_index",
  "array_ref": "/tracking_data/source_acquisition_frame_index",
  "array_dtype": "<i8",
  "array_shape": [2],
  "array_content_sha256": "<exact array content sha256>",
  "canonicalization": "canonical_json_sort_keys_v1"
}
```

The example illustrates `N=2` and a ten-frame source. Emitted values must match
the actual arrays and acquisition authority. Orange `recording_frame_id` is
one-based, so its corresponding acquisition index is
`recording_frame_id - 1`.

### Held-target provenance

The target detection may come from an earlier acquisition than the state row.
It therefore has a separate mapping:

- `target_source_acquisition_frame_index`: exact `<i8 [N]`
- `target_source_acquisition_frame_valid`: exact `bool [N]`
- valid row: index is nonnegative and below `source_total_frames`
- invalid row: index is exactly `-1`

The index array carries a sealed
`target_source_acquisition_mapping_record` and matching `_sha256`. The record
uses:

```text
schema_id = "citrus.stimulus_target_source_acquisition_mapping"
schema_version = 1
mapping_method = "explicit_per_stimulus_state_target_provenance_v1"
source_target_frame_field = "/tracking_data/chaser_states#target_source_frame_id"
source_target_camera_field = "/tracking_data/chaser_states#target_source_camera_id"
array_ref = "/tracking_data/target_source_acquisition_frame_index"
array_dtype = "<i8"
validity_array_ref = "/tracking_data/target_source_acquisition_frame_valid"
validity_array_dtype = "|b1"
invalid_index_sentinel = -1
```

It also contains the same row-identity, acquisition recording/camera,
`source_total_frames`, array shape/content digest, target-domain, and
canonicalization fields as the state mapping, plus the validity array's shape
and content digest. The validity array carries only a record reference and the
same record digest.

### Arena geometry and frame authority

`/calibration_snapshot/arena_geometry` must preserve these direct attributes
as exact integers:

```text
arena_region_width_px
arena_region_height_px
arena_origin_in_canvas_x_px
arena_origin_in_canvas_y_px
```

Width and height are positive; origins are nonnegative. Seal them in an
`arena_geometry_record` (`palette.arena_geometry_reference`, version 1,
units `px`) and `arena_geometry_record_sha256`. Also attach a sealed
`pixel_frame_authority` (`palette.pixel_frame_authority`) and matching digest
for frame `citrus_arena_relative_canvas`, space
`arena_relative_canvas_px`, continuous pixels, and the arena geometry's exact
reference extent.

Keep the selected camera identity, display snapshot, directed camera-to-canvas
homography, calibration source records, and their checksums in the existing
calibration snapshot. Pixel transformations are supported only when that chain
is complete and digest-bound.

## Temporal and shutdown rules

- Write one canonical chaser row and one corresponding frame-metadata row for
  each stimulus state.
- Consume the Shaman v2 `recording_frame_id`/recording identity during logging
  and persist the explicit zero-based acquisition arrays above.
- Keep `triggering_camera_frame_id` as external provenance/join evidence.
- Do not log a shutdown-only `UpdateTargetState()` row without a corresponding
  rendered/frame-metadata state. Cleanup must not append scientific state.
- Keep camera-native bounding boxes outside the stimulus coordinate surfaces;
  they require their own camera-observation contract.

## Digests

JSON-record digests are SHA-256 over UTF-8 JSON with sorted keys, no NaN,
compact separators, and no trailing newline. Array content digests are SHA-256
over the compact canonical JSON header
`{"canonicalization":"numpy_dtype_shape_c_order_bytes_v1","dtype":...,"shape":...}`,
one zero byte, then the exact contiguous C-order bytes. A stale record, extra
field, wrong dtype/shape, or mismatched digest fails closed.

The executable reference implementation is
`fisheye.shared.stimulus_coordinate_contract`; the historical Batman migration
utility demonstrates production construction of every record and digest in
`fisheye.utils.migrate_legacy_batman_stimulus_h5`.
