# Bout Morphology Collection Design Decision

<!-- decision-meta
status: accepted-design
created: 2026-07-10
owner: jeremy
scope: cross-recording swim-bout collections containing contours, trajectories,
  segmentation, distance, and tail kinematics
depends_on: docs/behavior_event_analysis_design_decision.md,
  docs/stable_identity_incremental_materialization_decision.md,
  docs/dataset_reporting_contract.md,
  docs/zarr_parquet_sidecar_exports_design.md
-->

## Decision Summary

Palette will represent large cross-recording deliveries of swim-bout contours
and synchronized kinematics as versioned, bout-centered multimodal
collections. These are not ordinary flat analytics exports.

The collection uses:

- immutable manifests for exact recording and run selection;
- Parquet indexes for recording-, bout-, sample-, and exclusion-level scalar
  metadata;
- per-recording Zarr shards for fixed-size and ragged contour geometry,
  frame-aligned tail posture, and other multidimensional payloads;
- explicit capability and validity fields so recordings without every modern
  modality remain visible rather than being silently removed.

Dense refined subject masks remain the pixel authority. Full and sampled
contours are regenerable derived caches. Recording Zarrs remain the scientific
source of truth; collection shards are portable, immutable derived products.

## Why This Decision Exists

A collaborator may need every valid subject contour across as many accepted
swim bouts and recordings as possible, together with position, velocity, speed,
distance, bout segmentation, and tail kinematics. This creates several nested
axes:

```text
recording
  track or fish
    swim bout
      frame or sample
        variable-length contour points
```

Flattening every contour point into one Parquet table would produce a very
large repeated scalar dataset, make synchronized multidimensional reads
awkward, and obscure the natural bout/sample hierarchy. Copying whole recording
archives would preserve too much unrelated data while failing to provide a
cohort-level index. Keeping only bout summaries would discard the requested
geometry.

The hybrid collection preserves exact selection and queryability while keeping
large array payloads in a chunked representation.

## Collection Layout

The portable collection layout is:

```text
bout_morphology_collection/
  collection_manifest.json
  recordings.parquet
  bouts.parquet
  samples.parquet
  exclusions.parquet
  shards/
    <dataset_id_1>.zarr
    <dataset_id_2>.zarr
    ...
```

Each recording owns a separate shard. Workers may build shards concurrently
because no two workers write the same physical Zarr chunks. The collection is
published only after every included shard, Parquet index, checksum, and
manifest has validated successfully.

The collection must not be published by incrementally appending rows into one
shared mutable Zarr store.

## Selection And Identity

The planner starts from active physical analysis archives selected through the
registry. Physical aliases are collapsed before source inspection.

One accepted/canonical swim-bout run and signal level is selected per concrete
track unless the collection explicitly targets segmentation-candidate
comparison. Candidate and accepted bouts must not be mixed silently.

Every bout receives a deterministic `global_bout_id` derived from at least:

```text
dataset_id
recording_id
source tracking run
track_id
source swim-bout run
speed/signal level
source bout_id
```

Every materialized sample receives a `collection_sample_index` scoped to one
collection version and retains its exact source frame, timestamp, observation
identity, and source row identities.

## Exact Cross-Stage Join

The materializer resolves this lineage for each bout frame:

```text
swim-bout track and frame
  -> tracking observation
  -> refined subject-mask row
  -> component contour cache or authoritative dense mask
  -> subject-shape row
  -> tail-kinematics and tail-posture row
```

Frame number alone is not a sufficient join key in multi-subject recordings.
The join uses stable observation identity, refined row identity, source detect
row identity, and exact tracking-run assignment where available.

Single-fish recordings may resolve to one track, but this does not authorize a
general track-0 assumption. Duplicate observations for one track/frame,
identity ambiguity, missing associations, and track gaps receive explicit
terminal status.

## Parquet Index Contracts

### `recordings.parquet`

One row per selected physical archive, including:

- dataset and recording identity;
- registry context and physical Zarr location;
- protocol/condition metadata;
- exact selected source runs and lineage hashes;
- available modality tiers;
- selected and excluded bout/sample counts;
- estimated and written payload bytes;
- overall inclusion status and reason.

### `bouts.parquet`

One row per canonical source bout, including:

```text
dataset_id
recording_id
track_run
track_id
swim_bout_run
speed_level
bout_id
global_bout_id

start_frame
core_start_frame
peak_frame
core_end_frame
end_frame
start_time_s
end_time_s
duration_s

protocol_hash
stimulus_run
stimulus_step_index
stimulus_mode
condition metadata

source_refined_subject_mask_run
source_subject_shape_run
source_tail_kinematics_run
source_tail_posture_run
source_track_kinematics_run

sample_start
sample_count
valid_contour_fraction
valid_tail_fraction
valid_track_fraction

contours_available
tail_kinematics_available
calibration_available
inclusion_status
exclusion_reason
source_lineage_hash
```

Segmentation boundaries, source signal, detector method, parameters, and
thresholds remain present even when the consumer initially asks only for
geometry.

### `samples.parquet`

One row per materialized bout frame/sample, including:

```text
global_bout_id
bout_sample_index
collection_sample_index
camera_frame
timestamp_ns
relative_time_s

position_x_mm
position_y_mm
velocity_x_mm_s
velocity_y_mm_s
speed_mm_s
step_distance_mm
cumulative_bout_distance_mm
heading_rad

track_valid
transition_valid
contour_valid
tail_valid

source_mask_row
source_shape_row
source_tail_row
instance_key
refined_row_id
```

Velocity is a signed vector and speed is its magnitude. Step distance is
computed only between valid adjacent samples. Cumulative bout distance does not
bridge invalid transitions or tracking gaps. Raw pixel coordinates may be
retained for diagnosis, but calibrated arena millimeters are the primary
cross-recording coordinates when available.

### `exclusions.parquet`

Every recording, bout, or requested sample that does not enter the payload has
a row with scope, identity, reason code, detail, failed prerequisite, and source
run context. Exclusions are part of the scientific product, not disposable
logs.

## Zarr Shard Contract

Each recording shard uses a flattened sample axis with bout ranges rather than
padding every bout to the longest duration:

```text
bout_samples/
  bout_indptr                    (n_bouts + 1,)
  source_frame                   (n_samples,)
  relative_time_s               (n_samples,)
  position_mm                    (n_samples, 2)
  velocity_mm_s                  (n_samples, 2)
  speed_mm_s                     (n_samples,)
  cumulative_bout_distance_mm   (n_samples,)
  heading_rad                    (n_samples,)
```

`bout_indptr[b]:bout_indptr[b+1]` selects the samples for bout `b`. A separate
fixed-window tensor may later be generated for machine learning, but it is a
derived view with explicit padding/validity and is not the collection
authority.

## Contour Payloads

The shard may contain both comparable sampled contours and exact packed
contours:

```text
contours/
  body_sampled_xy_roi            (n_samples, K_body, 2)
  body_sampled_xy_body           (n_samples, K_body, 2)
  body_valid                     (n_samples,)

  exact_points_xy                (n_contour_points, 2)
  exact_ptr                      (n_samples,)
  exact_len                      (n_samples,)
```

The exact contour uses a second ragged mapping:

```text
bout -> sample range through bout_indptr
sample -> contour-point range through exact_ptr/exact_len
```

Components are declared explicitly. `subject_body` is the initial required
contour family; swim bladder and eye contours may be included as separately
named optional components.

### Contour authority

Modern refined subject-mask `masks_roi` is the authoritative pixel surface.
Full component contours under `components/<component>/contours` and fixed-count
sampled contours under `sampled_contours` are derived caches.

If a contour cache is missing or stale and the exact authoritative dense mask
is available, the collection builder may regenerate the contour directly into
the export shard. It must not mutate the source recording merely to satisfy a
delivery. Historical compact masks may be read through compatibility tooling,
with the source encoding and materialization procedure recorded.

### Comparable contour parameterization

The existing fixed-count sampled contour cache guarantees uniform closed
arc-length sampling but does not guarantee that a point index is an anatomically
homologous location across frames and fish. It is a display cache until a
canonical start and winding contract is applied.

The comparable body contour therefore:

1. transforms into the persisted body frame;
2. normalizes distances by a declared body/reference length;
3. enforces a single winding direction;
4. anchors point 0 at a declared anatomical landmark, initially the snout;
5. traverses one declared anatomical side toward the tail and returns along the
   other side;
6. validates temporal continuity and flags ambiguous rows.

Boundary point index must not be interpreted as anatomy before this
canonicalization. A body-normalized mask or signed-distance representation may
be added when point correspondence is unreliable.

## Tail Payloads

Tail arrays remain aligned to the same flattened sample axis:

```text
tail/
  tail_sample_s                  (n_tail_positions,)
  tail_sample_xy_body            (n_samples, n_tail_positions, 2)
  tail_angle_rad                 (n_samples, n_tail_positions)
  tail_curvature_normalized      (n_samples, n_tail_positions)
  tail_lateral_deflection        (n_samples, n_tail_positions)
  valid                          (n_samples,)
```

Palette-native tangent angles and Megabouts-compatible cumulative segment
angles remain different named contracts. A classifier category may be included
as bout metadata, but it does not replace continuous tail posture and curvature
arrays.

## Availability Tiers

Maximizing coverage must not silently reduce the collection to recordings with
all modern outputs. The collection defines cumulative capability tiers:

- **Tier A — bout trajectory:** segmentation, position, velocity, speed,
  heading, and distance;
- **Tier B — contour morphology:** Tier A plus valid subject-body contours;
- **Tier C — tail enhanced:** Tier B plus subject shape and tail kinematics;
- **Tier D — calibrated/event annotated:** Tier C plus reliable stimulus and
  calibration metadata.

The master bout index includes all eligible Tier A bouts and availability flags
for higher tiers. A collaborator may select the Tier C complete-case
intersection, but the collection still exposes how many recordings and bouts
were lost and whether missingness differs by condition.

## Read-Only Census Before Materialization

No collection is materialized before a read-only census reports:

1. selected registry cohort and collapsed physical aliases;
2. exact source runs and completion/freshness state;
3. recording, track, bout, and requested sample counts;
4. full and sampled contour availability and valid-row coverage;
5. tail, track, calibration, and stimulus coverage;
6. stale or missing derived contour caches that are regenerable from dense
   masks;
7. estimated bytes for scalar indexes, sampled contours, exact contours, and
   tail arrays;
8. availability tier and exclusion counts by protocol/condition;
9. proposed immutable collection ID and source manifest hash.

Registry stage status is useful for candidate selection but is insufficient for
row-level coverage. The census inspects source Zarr metadata and required
validity arrays without modifying them.

## Validation And Publication

Before atomic publication, validate:

- global bout IDs are unique;
- every Parquet bout range agrees with shard `bout_indptr`;
- sample indexes and source frames are monotonic within each bout;
- all valid contour pointers and lengths remain within the point payload;
- sampled and exact contours resolve to the declared source rows;
- no valid transition crosses an invalid tracking gap;
- scalar Parquet values match the corresponding shard arrays;
- source run IDs and lineage hashes match the frozen collection manifest;
- file hashes and byte sizes match the final manifest;
- exclusions plus included rows reconcile with the census universe.

Canary visualizations overlay exported contours, tail splines, trajectories,
and bout boundaries on source frames for a deterministic sample of recordings
and failure cases.

## Statistical Use

Millions of frames do not change the experimental-unit hierarchy. Analyses
report fish/recording, bout, and frame counts separately. Bouts remain nested
within fish or recording; condition comparisons use experimental-unit-aware
models or resampling.

The collection keeps separate questions for bout occurrence, bout-category
selection, execution within a category, contour/tail morphology, and resulting
trajectory. Fish with more detected bouts must not dominate unless an
event-weighted estimand is explicitly intended.

## Registry And Versioning

Each collection version is immutable and registered by collection ID, manifest
SHA-256, manifest path, record/bout/sample counts, source recording count,
availability-tier counts, output root, status, and creation time. Detailed
source and exclusion lineage remains in collection files.

Changing source masks, tracking, bout segmentation, tail runs, run-selection
policy, canonical contour parameterization, or export code creates a new
collection version. Existing exports are not patched in place.

## Consequences

### Benefits

- A collaborator can query bouts and scalar kinematics without opening every
  source recording.
- Exact and normalized contours remain available at frame resolution.
- Multidimensional payloads retain efficient chunked access.
- Missing modalities and excluded data remain scientifically visible.
- Every sample is traceable to exact recording rows and source runs.
- Per-recording shards allow safe parallel materialization and partial reuse.

### Costs

- Cross-stage observation-to-track joins must be complete and validated.
- Exact contours may be large and require a census-based delivery decision.
- Comparable contour indexing requires a new anatomical anchor/winding
  contract beyond the current display cache.
- A portable collection duplicates derived data already present in recording
  Zarrs.
- Collection building and verification are more complex than a flat table
  export.

## Rejected Alternatives

### Put every contour point in the existing analytics Parquet export

Rejected because it explodes repeated scalar rows, weakens multidimensional
read locality, and obscures bout/sample/point hierarchy.

### Deliver only fixed-count contour points without the exact source contract

Rejected because current sampled point indexes are not yet anatomical
landmarks, and a fixed-count display cache cannot replace the authoritative
mask or exact contour.

### Require every modality and silently drop incomplete recordings

Rejected because complete-case selection may bias conditions and hide the true
available universe. Capability tiers and explicit exclusions are required.

### Copy complete recording directories

Rejected as the default because it includes unrelated video and analysis data,
does not create a cohort index, and is substantially less portable. Full
recordings remain available when the recipient needs original sources.

### Write one shared collection Zarr from parallel recording workers

Rejected because logical row separation does not guarantee physical Zarr chunk
ownership. Per-recording shards provide deterministic non-overlapping writes.

## Methodological References

- Di Santo et al., 2021, [Convergence of undulatory swimming kinematics across
  a diversity of fishes](https://pmc.ncbi.nlm.nih.gov/articles/PMC8670443/).
- Bernal et al., 2016, [FFT-based alignment of 2D closed curves with
  application to elastic shape analysis](https://www.nist.gov/publications/fft-based-alignment-2d-closed-curves-application-elastic-shape-analysis).
- Sridhar et al., 2024, [Uncovering multiscale structure in the variability of
  larval zebrafish navigation](https://pmc.ncbi.nlm.nih.gov/articles/PMC11588111/).
