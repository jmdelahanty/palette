# Canonical coordinate metadata framework

Palette uses one metadata framework for many scientifically distinct coordinate
spaces. It does not force all geometry into one space. Every persisted geometry
array owns a compact descriptor, and that descriptor points to exact typed,
digest-bound evidence for its frame, extent, transforms, row identity, time
mapping, and derivation.

This is the required contract for future normal producers. Historical labels and
unstamped arrays are migration inputs, not an alternate production contract.

## Array-owned descriptor

Each geometry array carries `coordinate_descriptor` plus its SHA-256 digest. The
descriptor answers only the questions needed to interpret that array:

- controlled `profile_id` and `space_id`;
- geometry type, component order, and component units;
- origin and positive X/Y directions;
- exact reference width and height with a digest-bound authority;
- pixel-center, pixel-edge, continuous, or not-applicable convention;
- digest-bound row-identity record;
- whether source-camera overlay is direct, requires a direction-labelled
  transform chain, or is not suitable;
- digest-bound lineage records and, where required, a typed frame record.

Large provenance structures are not duplicated into every descriptor. Crop
placement, calibration, transformation, and derivation records are persisted
once and referenced by canonical archive path plus digest. A descriptor is valid
only when a fresh loader resolves and revalidates every referenced record in the
same exact archive.

Run, subgroup, array name, shape, dimension name, value range, and historical
helper name are never coordinate authority.

## Controlled spaces and profiles

The current controlled `space_id` vocabulary is:

| Space | Meaning | Canonical profile direction |
|---|---|---|
| `source_camera_image_px` | acquisition/source camera pixels | top-left, +X right, +Y down |
| `source_camera_normalized_xy` | normalized source-camera coordinates | top-left, +X right, +Y down |
| `detector_model_input_px` | exact detector tensor/input pixels | top-left, +X right, +Y down |
| `detector_normalized_xy` | normalized detector-input coordinates | top-left, +X right, +Y down |
| `roi_local_px` | crop or ROI-local pixels | top-left, +X right, +Y down |
| `stimulus_texture_px` | stimulus texture pixels | top-left, +X right, +Y down |
| `stimulus_canvas_px` | selected stimulus canvas pixels | top-left, +X right, +Y down |
| `projector_px` | calibrated projector pixels | projector top-left, +X right, +Y down |
| `arena_relative_canvas_px` | arena-relative canvas pixels | arena top-left, +X right, +Y down |
| `physical_mm` | a typed physical frame in millimetres | profile-specific and explicit |
| `fish_anatomical_body_frame` | fish-local anatomical coordinates | +X anterior, +Y anatomical left |

`profile_id` selects an allowed combination of space, origin, axes, units,
extent policy, pixel convention, overlay status, and required frame-record kind.
Writers cannot invent field combinations merely by supplying a known
`space_id`.

Presentation viewport, renderer, screen, and display-window coordinates are not
persisted coordinate spaces. They are ephemeral Crimson renderer state.

## Typed evidence records

The compact descriptor is supported by separate records:

- pixel-frame authority: exact width, height, origin, axes, convention, and
  acquisition/crop/canvas ownership;
- reference-extent authority: exact selected width and height, never a guessed
  root or model dimension;
- directed transform: source frame, destination frame, numerical transform,
  direction, convention, and digest;
- directed transform chain: an ordered, composable source-to-destination chain;
- calibration/frame record: exact selected physical or anatomical frame and its
  lineage;
- coordinate derivation: exact input and output payloads, operation, selection,
  and record digests;
- row identity and temporal authority: exact row keys and acquisition-frame
  mapping.

Transforms are always labelled from source to destination. Historical helper
names such as `projector_to_camera_px` do not override a persisted direction.
Resolution ratios cannot replace calibration, crop placement, model-input
preprocessing, or a directed transform record.

## Identity domains

Coordinate metadata references row identity but does not collapse different
scientific identities into one key:

- observation rows use `instance_key`;
- track samples use `track_sample_key = (track_id,
  source_acquisition_frame_index)`;
- a track sample may also carry nullable `source_instance_key` as observation
  lineage, but it is not the track-sample primary key;
- stimulus states use `stimulus_state_key`, with acquisition-frame mapping stored
  separately;
- subject identity remains distinct from observation and track-sample identity.

Subset and reorder operations must persist the exact source-row selection and
mechanically derive output identity/time records. Positional equality is not
identity evidence.

## Producer publication boundary

A normal future producer must:

1. Load sealed evidence from the exact selected input nodes.
2. Validate numerical geometry and row/time alignment before publication.
3. Persist output arrays and exact derivation records.
4. Stamp array-owned descriptors from sealed evidence, never reconstructed
   dictionaries.
5. Reopen the authoritative path and freshly validate the complete binding.
6. Mark the run complete and update convenience pointers only after validation.

Node-local or sharded computation may write an explicitly unbound numeric stage.
It must not carry coordinate descriptors or identity claims. Binding happens
only after atomic placement at the final authoritative archive path; any failure
leaves the run incomplete and rolls back publication.

Future normal writers are canonical-only. A historical reader or migration tool
may expose an explicit `legacy_noncanonical` mode, but that mode cannot publish
canonical descriptors or silently enter a normal producer path.

## Palette and Crimson responsibilities

Palette must persist:

- the array-owned descriptor and digest;
- exact frame and reference-extent records;
- exact row identity and acquisition-time mapping;
- crop/model/calibration/transform lineage with direction-labelled transforms;
- exact derivation or subset/reorder records;
- enough source-camera overlay semantics to distinguish direct overlay,
  transform-required, and unsupported geometry.

Crimson may:

- validate a supported descriptor/profile and every referenced digest;
- draw `source_camera_image_px` arrays directly only when overlay status is
  `direct`;
- apply only the persisted ordered transform chain when status is
  `requires_transform`;
- create ephemeral viewport/display transforms after reaching its presentation
  camera frame.

Crimson must not infer a space from `positions_px`, dimensions, numeric ranges,
run type, or historical `camera`/`texture` labels. Missing, ambiguous,
unsupported, or stale evidence fails closed.

## Legacy compatibility and migration

`camera` and `texture` remain readable only through explicit compatibility rules
that supply exact authority, dimensions, and lineage. Future writers do not emit
those labels and future recordings do not require runtime legacy adapters.

Migration classes are:

- **metadata-only backfill**: existing numeric payload and exact lineage already
  prove one canonical descriptor without reconstructing coordinates;
- **recomputation required**: coordinates or identity were derived from guessed
  dimensions, simple ratios, missing crop placement, incomplete acquisition
  pixels, or an unsupported source;
- **ambiguous, fail closed**: available metadata cannot prove a unique frame,
  direction, reference extent, identity, or transform lineage.

A metadata-only backfill may add records and descriptors but cannot change array
values. Any value transformation creates a new derived run with fresh lineage.

The implementation authorities are
`fisheye.shared.coordinate_descriptor`,
`fisheye.shared.canonical_coordinate_publication`, the typed frame/transform
modules under `fisheye.shared`, and producer-specific publication modules.
