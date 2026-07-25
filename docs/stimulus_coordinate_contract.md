# Future stimulus coordinate contract

Normal stimulus import and refined-online processing are canonical-only. They
do not infer coordinate or time semantics from array names, dimensions,
numeric ranges, camera-frame identifiers, or resolution ratios. Historical
archives must be handled by explicit offline inventory/migration tooling.

## Metadata-and-calibration-only import

An H5 may contain scientifically useful events, protocol metadata, and selected
calibration while its coordinate arrays predate the canonical row/time identity
contract. Operators may explicitly request
`--metadata-and-calibration-only` from `import_stimulus_to_zarr`, or
`--stimulus-metadata-and-calibration-only` from the recording import wrappers.

This mode does not infer or bless missing coordinate semantics. It publishes the
event timeline, authored protocol definition and step timing, selected camera and
display calibration, and physical scale authority. It omits
`stimulus_coordinates`, H5 tracking bounding boxes, `chaser_states`, and their
row/time identity arrays. The completed run records the exact omitted H5 paths,
the explicit policy, and the omission reason in both attrs and run provenance.
Consumers requiring chaser positions must continue to fail closed until those
surfaces are handled by an exact offline migration.

The default remains strict canonical import.

## Row and time identity

`stimulus_state_key` is the primary identity of a `chaser_states` row. It is
not an observation `instance_key`, and `triggering_camera_frame_id` is not an
acquisition-frame index. Citrus must publish this exact source dataset:

`/tracking_data/source_acquisition_frame_index`

The dataset is signed int64, row-aligned with `stimulus_state_key`, and carries
a closed `citrus.stimulus_source_acquisition_mapping` record plus its digest.
That record names the recording, camera, acquisition-frame domain, source row
identity, array payload, and exact content digests. Palette copies the values
and record, binds them to the archive's acquisition-camera authority, and
publishes `source_row_temporal_authority` on the imported rowset. Missing or
conflicting evidence fails before canonical publication; Palette never falls
back to `triggering_camera_frame_id`.

The future temporal route is therefore:

`stimulus_state_key → source_acquisition_frame_index → track_sample_key`

External camera-frame IDs remain useful provenance and presentation joins, but
they may repeat and do not order smoothing, outlier detection, interpolation,
or tracking. A compatibility reader whose output is keyed by camera-frame ID
may reject duplicates; future acquisition-time consumers must use the sealed
stimulus-state/acquisition mapping instead.

## Spatial authority

Each imported coordinate surface carries a compact schema-v2 descriptor. The
descriptor binds:

- the exact `stimulus_state_key` row contract;
- a typed arena-relative canvas pixel frame;
- an explicit ordered transform chain from arena-relative canvas pixels to the
  selected canvas and then to source-camera pixels; and
- digest-bound surface, camera mapping, frame/transform, import, and output
  records.

The Citrus source descriptor must itself bind a digest-sealed
`pixel_frame_authority` record on `/calibration_snapshot/arena_geometry`; the
arena-geometry record is that frame's extent and placement lineage, not a
substitute for the typed frame record. Palette verifies both before opening the
destination and then publishes the destination-local frame and transform chain.

The selected-calibration homography is persisted in its declared
source-camera-to-selected-canvas direction. Palette persists and labels the
exact inverse separately. Arena placement is a direction-labelled translation,
not a width/height ratio. Presentation viewport coordinates are renderer state
and are not persisted as scientific coordinate authority.

## Refined-online publication

Refinement selects rows by `stimulus_state_key`, orders them only by sealed
`source_acquisition_frame_index`, and copies external camera IDs as provenance.
Filtered and interpolated arrays preserve the exact typed reference frame and
arena-to-source-camera transform chain from the selected source. Their source
mapping, processing record, surface manifest, output row identity, and output
temporal authority are digest-bound and reloadable through
`BoundRefinedOnlineCoordinateEvidence`.

Normal readers must reject missing lineage, unsupported spaces, stale extents,
wrong transform direction, ambiguous row identity, and source/destination
archive substitution. Compatibility rules belong in explicit migration or
inventory commands, never in future writers or ordinary readers.
