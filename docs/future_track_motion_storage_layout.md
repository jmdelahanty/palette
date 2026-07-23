# Future Track-Motion Storage Layout

Status: deferred follow-on design

This document records a future storage-layout change for Palette track motion.
It is intentionally **not** part of the active coordinate-contract remediation.
The current goal should first finish canonical coordinate publication, strict
reader validation, registry inventory, migration classification, and the
Palette–Crimson contract.  This design should begin only after that work is
stable and independently reviewed.

## Decision

Future-normal track-motion runs should stop materializing the same logical
motion series as both nested per-level arrays and flat compatibility aliases.
Palette should adopt one versioned physical layout in which repeated,
same-domain motion levels are columns of explicitly labelled arrays.

The logical reader remains the public software boundary.  Scientific
algorithms and Crimson should consume verified logical surfaces rather than
hard-code the physical Zarr layout.

This is a storage normalization.  It does not replace the coordinate metadata
framework and does not imply that all coordinate spaces become one space.

## Current layout and problem

Current runs can contain paths such as:

```text
tracks/id_0/movement/speed/filtered/mm
tracks/id_0/movement/speed/filtered/px
tracks/id_0/speed_filtered_mm
tracks/id_0/speed_filtered_px
```

The grouped and flat arrays can contain identical payloads.  This layout is
readable, but it has undesirable properties for a future canonical schema:

- the level and units are partly encoded in path segments;
- one logical family expands into many Zarr arrays and metadata nodes;
- aliases duplicate storage and increase sealing, validation, and maintenance
  work;
- direct consumers become coupled to a directory layout;
- a leaf such as `mm` does not state that the quantity is speed in `mm/s`;
- adding a level or derivative multiplies paths and compatibility rules.

Paths may remain useful discovery labels, but they must not establish units,
coordinate space, row identity, or derivation semantics.

## Proposed canonical layout

Keep one subgroup per track because tracks may have different row counts.  Keep
sample identity and position surfaces at the track root.  Store repeated motion
levels as the second axis of metric-specific arrays:

```text
analysis/track_kinematics_runs/<scope>/<run>/tracks/id_<track>/
  track_sample_key                       # [track_sample, 2]
  source_acquisition_frame_index         # [track_sample]
  source_instance_key                    # [track_sample, ...], nullable lineage
  positions_px                           # [track_sample, xy]
  positions_mm                           # optional [track_sample, xy]

  movement/
    speed_px_s                           # [track_transition_destination_sample, speed_level]
    speed_mm_s                           # optional, same shape
    frame_path_distance_px               # [track_transition_destination_sample, path_level]
    frame_path_distance_mm               # optional, same shape
    acceleration_px_s2                   # [track_transition_destination_sample, speed_level]
    acceleration_mm_s2                   # optional, same shape
    smoothed_acceleration_px_s2          # [track_transition_destination_sample, speed_level]
    smoothed_acceleration_mm_s2          # optional, same shape
```

Exact names and the next run-schema version must be selected during the
follow-on implementation.  The important decisions are one physical payload
per metric/unit profile, explicit dimensions, and no future flat aliases.

### Labelled collection axes

Each level-bearing array must bind a canonical, digest-bound collection-axis
record.  For example:

```text
axis_name: speed_level
axis_index: 1
cardinality: 4
ordered_labels: [raw, filtered, smoothed, averaged]
```

The record must also bind the exact array shape and, where applicable, a
per-label derivation profile.  A consumer must reject a reordered, missing, or
unrecognized level even when the numerical shape is unchanged.

Level labels must come from a controlled vocabulary.  They must not be inferred
from column position, a path name, or an algorithm's defaults.

### Array authority

Every canonical array still requires:

- exact dtype, rank, shape, and payload digest;
- an explicit axis-0 domain: positions use `track_sample`, while speed,
  distance, acceleration, and smoothed acceleration use
  `track_transition_destination_sample` and bind the destination
  `track_sample_key`, acquisition-frame identity, and transition validity;
- axis-1 collection-axis record and digest;
- units and semantic profile;
- exact input references and per-level operation/parameter records;
- publication manifest membership;
- optional physical-frame authority for millimetre outputs;
- fresh complete/eligible validation before normal reading.

`speed_mm_s` and other physical arrays are optional.  They may exist only when
an exact compatible typed physical calibration is bound.  Palette must not
synthesize them from an untyped scalar or resolution ratio.

## Chunking and performance decision

The follow-on team must benchmark the physical chunk shape rather than assume
that a rectangular logical array implies one full-width physical chunk.

Candidate layouts include:

```text
(track_row_chunk, all_levels)
(track_row_chunk, 1)
```

The first favors reading several levels together and reduces chunk count.  The
second permits efficient single-level reads.  The decision should use real
track lengths and the dominant Palette/Crimson access patterns.

Any parallel Zarr writer must follow `AGENTS.md`: workers may write only whole,
non-overlapping physical chunks.  Requested and effective worker chunking must
be persisted in provenance when adjusted for chunk safety.

## Reader and consumer boundary

`fisheye.analysis.track_kinematics_io` should remain the normal logical reader.
It should return named logical levels and immutable authority records regardless
of the physical schema version it was explicitly written to support.

For the future schema:

- normal readers support only the new canonical profile;
- Palette algorithms receive ordinary NumPy arrays after strict preflight;
- Crimson consumes the contract-defined logical surfaces, not Zarr paths;
- presentation viewport coordinates remain ephemeral Crimson state;
- historical layouts are available only to explicit read-only audit or
  migration commands;
- new writers do not emit grouped/flat compatibility copies.

The current coordinate-remediation work may use the existing grouped layout
while establishing the verified reader boundary.  That boundary is what makes
this later physical-layout change possible without rewriting scientific
algorithms.

## Migration policy

Do not rewrite historical runs in place merely to obtain the compact layout.

- Existing runs remain historical immutable artifacts.
- Audit and migration tooling may read them through an explicitly requested
  historical-inspection path.
- If a compact canonical copy is scientifically useful, create a new run and
  bind it to the exact verified source payload and manifest.
- Ambiguous or unsealed historical arrays must not be promoted by copying their
  numbers into a new path.
- Future acquisitions should write only the new schema once the cutover is
  activated; they should not require a legacy adapter.

## Ordered agent-team work package

Start this work only after the active coordinate-contract goal is complete.

1. **Schema and access-pattern audit**

   Inventory every remaining direct path consumer and measure actual
   single-level versus multi-level reads.  Freeze the controlled level
   vocabulary, array names, shapes, and schema-version boundary.

2. **Chunking benchmark**

   Compare candidate chunk shapes using representative long recordings.
   Measure metadata-node count, archive size, single-level reads, full-motion
   reads, publication hashing, and write behavior.

3. **Canonical writer implementation**

   Write only the compact arrays, collection-axis records, per-level
   derivations, and full payload manifest.  Do not create flat or per-level
   compatibility aliases.

4. **Logical reader implementation**

   Teach the strict reader to expose the same named logical mappings from the
   compact arrays.  Keep physical path knowledge inside the reader.

5. **Consumer cutover**

   Remove direct Zarr path reads from scientific, visualization, export, and
   Crimson-facing code.  Persist the source motion-manifest digest in every
   derived output.

6. **Contract update**

   Version the Palette–Crimson motion contract around logical fields and
   collection-axis labels.  State that Crimson must not infer meaning from the
   compact layout.

7. **Independent review and activation**

   Review schema closure, numerical equivalence, chunk safety, lifecycle,
   selector behavior, and unsupported-layout failures before enabling the new
   writer for future recordings.

## Acceptance criteria

The follow-on is complete when:

- future runs contain one physical array per metric/unit profile and no flat or
  per-level aliases;
- every level axis is ordered, controlled, digest-bound, and shape-bound;
- swapping or reordering equal-shaped columns fails validation;
- pixel and millimetre arrays share exact row/level identity and a typed
  calibration relationship;
- absence of physical calibration produces no millimetre arrays;
- the logical reader returns numerically equivalent named levels for an exact
  verified reference fixture;
- normal consumers contain no hard-coded old-layout paths;
- interrupted or concurrent publication cannot expose a partial selectable
  run;
- physical chunk ownership satisfies Palette's parallel-write rule;
- historical support is confined to explicit inspection/migration tooling;
- the Palette–Crimson contract and focused tests describe the same schema.

## Non-goals

- Do not rewrite speed, smoothing, acceleration, or path-distance algorithms.
- Do not conflate storage-level labels with coordinate-space identifiers.
- Do not persist Crimson viewport coordinates.
- Do not migrate production archives as part of the design phase.
- Do not preserve future write compatibility by emitting duplicate arrays.
