# Citrus Arena Topology Design

This note summarizes the current downstream assumptions around Citrus `arena`
metadata and proposes a long-term topology model that can support:

- multiple cameras for one physical arena
- multiple sub-arenas under one camera view
- gradual migration of existing Palette consumers and recordings

## Problem

Today the word `arena` is overloaded.

- In acquisition/session metadata, `arena_id` is treated by downstream code as a
  recording context key.
- In derived analysis outputs, `arena_ids` are already used as ROI-level
  occupancy IDs for sub-dish or per-region assignment.

That is workable for the current single-camera, single-context recording model,
but it does not scale cleanly to:

- one physical arena observed by multiple cameras
- one camera stream containing multiple independently analyzed sub-arenas
- future cross-camera reasoning about the same physical sub-arena

The main design mistake to avoid is repurposing existing `arena_id` to mean the
parent physical container. Current Palette consumers already use `arena_id` as a
strong grouping, filtering, and model-selection key, so changing its meaning
would collapse contexts that are currently distinct.

## Key Current Constraints

The current Palette consumers behave as if a recording artifact has a single
context:

- one H5/session
- one `arena_id`
- one `camera_id`
- one protocol context

Relevant code paths:

- `src/fisheye/analysis/import_stimulus_to_zarr.py`
- `src/fisheye/utils/organize_recordings.py`
- `src/fisheye/utils/import_recordings_analysis.py`
- `src/fisheye/utils/import_recordings_training.py`
- `src/fisheye/registry/query.py`
- `src/fisheye/utils/resolve_detect_model.py`

Derived tracking/analysis already uses ROI-level arena semantics:

- `src/fisheye/tracking/arena_assignment.py`
- `src/fisheye/tracking/single_subject_per_arena.py`
- `src/fisheye/analysis/compute_speed.py`
- `src/fisheye/analysis/track_kinematics.py`

This split is the core reason to separate parent arena, child zone, camera, and
stream explicitly.

## Design Goals

- Make physical topology explicit instead of inferred from `arena_id`.
- Keep the smallest independently analyzed unit stable across training,
  registry, and analysis outputs.
- Support many-to-many relationships between cameras/streams and sub-arenas.
- Allow legacy recordings to continue working with minimal backfill.
- Keep naming separate from identity.

## Modularity Assessment

This design is more modular than the current workflow, but the important point is
that it makes meaning modular, not just code modular.

Today the pipeline is already stage-modular, but its metadata model is not.
`arena` currently carries multiple meanings across acquisition, registry, and
analysis. That ambiguity is manageable for the current single-camera workflow,
but it becomes a structural problem once the project needs:

- multiple cameras per physical arena
- multiple independently analyzed sub-arenas in one view
- future cross-camera reasoning about the same physical unit

In that setting, the cost of not modularizing the metadata model is higher than
the cost of introducing a few more explicit identifiers.

### Pros

- Aligns the data model with the actual project goals.
- Makes parent container, analysis unit, camera, and stream separable.
- Improves provenance and auditability.
- Reduces hidden assumptions currently embedded in `arena_id`.
- Creates a clean path for multi-camera and multi-zone support.
- Lets legacy recordings remain usable through compatibility aliases.

### Cons

- Adds conceptual complexity for developers and operators.
- Requires a compatibility period with both old and new fields.
- Increases registry/query/reporting surface area.
- Forces consumers to be explicit about whether they mean physical arena, zone,
  camera, or stream.
- Can become abstraction overhead if the model is generalized too far ahead of
  actual use cases.

### Recommendation

For this project, the advantages outweigh the disadvantages.

The roadmap already requires a distinction between parent arena, sub-arena, and
camera view. If that distinction is not made explicit in the metadata model, the
complexity still exists, but it remains implicit and harder to manage.

The recommended approach is:

1. modularize the identity and metadata model first
2. preserve backward compatibility with legacy `arena_id`
3. migrate consumers incrementally
4. only then generalize execution and analysis workflows for richer multi-camera
   behavior

In short: modularize the contract first, not the runtime architecture first.

## Recommended Canonical Model

Use separate identifiers for separate concepts.

### 1. Session

Represents one acquisition run.

- `session_id`
- `session_start_iso8601_utc`
- `rig_id`
- `metadata_schema_version`

### 2. Physical Arena

Represents the real dish/chamber/container.

- `physical_arena_id`
- `physical_arena_label`
- `physical_arena_type`

This is the parent object that may contain multiple independently analyzed
sub-regions and may be observed by multiple cameras.

### 3. Zone

Represents the smallest independently analyzed spatial unit.

- `zone_id`
- `physical_arena_id`
- `zone_label`
- `zone_type`
- `display_order`
- `parent_zone_id` (optional, if nested zones ever matter)

This is the concept that most closely matches what top-level `arena_id` means to
current downstream Palette consumers.

If a physical arena has no sub-arenas, it can still have exactly one zone.

### 4. Camera

Represents the hardware device.

- `camera_id`
- `camera_label`
- `camera_model`

### 5. Stream or View

Represents one recorded video/logging stream or one distinct camera view.

- `stream_id`
- `camera_id`
- `stream_label`
- `stream_kind`
- `primary_physical_arena_id` (optional shortcut)

This should become the unit that owns concrete media artifacts such as MP4, CSV,
and camera-side frame metadata.

`camera_id` and `stream_id` should not be treated as the same thing.

### 6. Zone Visibility / Zone View Mapping

Represents the fact that a given zone is visible in a given stream.

- `zone_id`
- `stream_id`
- `view_geometry`
- `homography_ref`
- `roi_id`
- `visibility_status`

This mapping is what allows the same physical sub-arena to appear in multiple
camera views without duplicating the zone identity itself.

### 7. Protocol Binding

Represents where the protocol logically applies.

- `protocol_id`
- `scope_type`
- `scope_id`

Recommended `scope_type` values:

- `session`
- `physical_arena`
- `zone`
- `stream`

Protocol scope should be explicit, not inferred from `arena_id`.

## Naming Recommendation

For the long-term model, avoid using plain `arena` as the canonical unit name.

Use:

- `physical_arena_id` for the parent container
- `zone_id` for the independently analyzed sub-unit

Retain `arena_id` only as a compatibility alias during migration.

## Recommended Metadata Surfaces

### Acquisition Session Context

Keep existing fields, but add:

- `physical_arena_id`
- `zone_id`
- `stream_id`
- `visible_zone_ids`
- `primary_zone_id`
- `protocol_scope_type`
- `protocol_scope_id`

For multi-camera session-level metadata, add:

- `camera_ids`
- `primary_camera_id`
- `stream_ids`

### Recording Manifest

Continue writing existing fields for compatibility, but extend the manifest with:

- `physical_arena_id`
- `zone_id`
- `stream_id`
- `camera_ids`
- `primary_camera_id`
- `visible_zone_ids`
- `primary_zone_id`
- `protocol_scope_type`
- `protocol_scope_id`
- `topology_snapshot`

### Derived Arrays and Run Metadata

Introduce new canonical names:

- `zone_ids`
- `track_zone_ids`
- `n_detections_per_zone`

During migration, dual-write legacy aliases:

- `arena_ids`
- `track_arena_ids`
- `n_detections_per_arena`

## Example Topology Snapshot

```json
{
  "metadata_schema_version": 2,
  "session_id": "2026-03-28T12-00-00Z_run_001",
  "rig_id": "omnifin0",
  "physical_arenas": [
    {
      "physical_arena_id": "pa_01",
      "physical_arena_label": "dish_bank_a"
    }
  ],
  "zones": [
    {
      "zone_id": "pa_01.z01",
      "physical_arena_id": "pa_01",
      "zone_label": "arena_1",
      "display_order": 1
    },
    {
      "zone_id": "pa_01.z02",
      "physical_arena_id": "pa_01",
      "zone_label": "arena_2",
      "display_order": 2
    }
  ],
  "cameras": [
    {
      "camera_id": "2010093"
    },
    {
      "camera_id": "2010094"
    }
  ],
  "streams": [
    {
      "stream_id": "cam2010093.main",
      "camera_id": "2010093"
    },
    {
      "stream_id": "cam2010094.main",
      "camera_id": "2010094"
    }
  ],
  "zone_views": [
    {
      "zone_id": "pa_01.z01",
      "stream_id": "cam2010093.main"
    },
    {
      "zone_id": "pa_01.z01",
      "stream_id": "cam2010094.main"
    },
    {
      "zone_id": "pa_01.z02",
      "stream_id": "cam2010093.main"
    }
  ],
  "protocol_bindings": [
    {
      "protocol_id": "DefaultScreen",
      "scope_type": "zone",
      "scope_id": "pa_01.z01"
    }
  ]
}
```

## Compatibility Strategy

### Rule 1

Do not redefine existing `arena_id` to mean the parent physical arena.

### Rule 2

For the migration period, treat:

- legacy `arena_id` == canonical `zone_id`

That preserves the expectations of current downstream consumers that group,
filter, and compare on `arena_id`.

### Rule 3

For old single-zone recordings without richer metadata, backfill:

- `physical_arena_id = arena_id`
- `zone_id = arena_id`
- `primary_zone_id = arena_id`
- `camera_ids = [camera_id]` when known
- `primary_camera_id = camera_id` when known

### Rule 4

Consumers should migrate in this order:

1. Read new fields when present.
2. Fall back to legacy `arena_id` and `camera_id`.
3. Prefer canonical `zone_ids` arrays when present.
4. Fall back to legacy `arena_ids`.

### Rule 5

Producers should dual-write for a while:

- legacy fields for current Palette compatibility
- new canonical fields for future Citrus/Orange topology support

## Recommended Consumer Migration Order

### Phase 1: Producer expansion

Add new metadata to Citrus/Orange outputs without changing existing field
meanings.

### Phase 2: Palette ingestion expansion

Update H5-to-Zarr/session-context import and manifest writing to mirror the new
fields.

### Phase 3: Registry normalization

Add normalized registry entities or columns for:

- session
- physical arena
- zone
- stream
- zone-to-stream mapping
- protocol binding scope

### Phase 4: Consumer preference switch

Update query, grouping, profile, and model-selection code to prefer:

- `zone_id` over legacy `arena_id`
- `stream_id` where stream-level identity matters
- `physical_arena_id` for parent-container grouping

### Phase 5: Derived analysis rename

Rename new derived outputs to `zone_*`, but continue writing `arena_*` aliases
until old consumers are retired.

## Decisions to Avoid

- Do not make `camera_id` the identity of the recording artifact.
- Do not make `arena_id` the parent physical container.
- Do not encode identity only in filenames like `arena_1`.
- Do not rely on ordering to infer semantic identity.
- Do not infer protocol scope from arena naming.

## Recommended Long-Term Direction

The long-term contract should be:

- parent container = `physical_arena_id`
- smallest independent unit = `zone_id`
- hardware device = `camera_id`
- concrete recorded view = `stream_id`

The migration contract should be:

- `arena_id` remains a compatibility alias for `zone_id`
- existing recordings are backfilled into the new model
- new recordings emit both legacy and canonical metadata until Palette
  consumers are updated

## How ROI-Local Segmentation Fits

The current and planned segmentation stages should be treated as row-local
artifacts, not as the place where cross-camera or cross-arena identity is
resolved.

That means stage families such as:

- `crop_runs`
- `keypoints_runs`
- `subject_mask_runs`
- `refined_subject_masks_runs`

should continue to bind to one exact source rowset and inherit topology through
their lineage rather than redefining it.

Recommended interpretation:

- one subject-mask run is one stream-local rowset artifact
- the rowset may include rows from multiple zones visible in the same stream
- if the same physical zone appears in multiple streams/cameras, each stream
  still has its own rowset and its own segmentation runs
- `zone_id`, `physical_arena_id`, and `stream_id` should come from upstream
  metadata/provenance surfaces, not from the mask tensor itself

In other words:

- topology decides what a row means
- segmentation stores the mask for that row
- tracking decides which rows belong to the same subject over time

This keeps `subject_mask_runs` and `refined_subject_masks_runs` compatible with
the current canonical design direction while avoiding any need to make mask
artifacts themselves cross-camera or cross-zone global objects.

## Multi-Subject Tracking Within One Arena

The current subject-mask layout is compatible with multiple subjects in one
arena as long as one row still means one candidate subject instance.

Recommended model:

- `zone_id` (or legacy `arena_id`) says which spatial container a row belongs
  to
- each detection/crop row still represents one putative fish/subject instance
- `subject_mask_runs` stores that row's `subject_body`, `eye_left`,
  `eye_right`, and `swim_bladder` masks
- `tracking_runs` assigns `track_id` to connect those per-row instances over
  time within the same zone

Under that model:

- multiple rows in the same frame may share one `zone_id`
- each row gets its own instance-local segmentation
- multi-subject semantics live in `track_id`, not in the subject-mask channel
  schema

Important non-goal:

- do not encode multiple subjects by creating channels like
  `subject_1_body`, `subject_2_body`, etc.

The thing that must generalize for multi-subject arenas is tracking, not the
subject-mask tensor layout.

### Key assumption

This only works if the upstream rowset remains instance-level.

If one ROI row regularly contains two fish, then the current mask contract
becomes ambiguous because it assumes one subject's components per row. In that
case the problem is upstream instance detection/cropping, not just tracking.

So for true multi-subject-in-one-arena support, the recommended layered model
is:

1. one zone may contain multiple simultaneous instance rows
2. each row gets its own per-instance subject masks
3. `tracking_runs` assigns stable `track_id`s across frames within that zone
4. downstream analysis joins mask-derived features to `track_id`, not to
   `arena_id` alone

Recommended conclusion for the current subject-mask contracts:

- keep `subject_mask_runs` and `refined_subject_masks_runs` single-instance per
  row
- do not try to encode multiple subjects by adding more component channels
- if one ROI contains multiple subjects, treat that as an ambiguous or
  upstream-instance-resolution problem first
- if multi-instance-in-one-ROI support becomes a real recurring requirement,
  introduce a separate instance-segmentation artifact rather than overloading
  the current subject-mask schema

## Short Recommendation for Citrus/Orange

If Citrus wants multi-camera arenas and multi-zone camera views, the safest path
is not to repurpose `arena`.

Instead:

- add `physical_arena_id`
- add `zone_id`
- add `stream_id`
- add explicit zone-to-stream topology metadata
- keep legacy `arena_id` stable as the current smallest independent unit until
  downstream consumers migrate

## Appendix: Current Palette Consumer Assumptions

This appendix captures the current-state assumptions that motivated the design
above. It is intentionally pragmatic: it summarizes what Palette consumers
actually read today, what they appear to assume `arena` means, and what would
break if that meaning changed.

### Current Output Surfaces That Read `arena`

#### H5 root attrs and Zarr session-context mirror

Current Citrus-to-Palette import paths read fixed root-attribute keys including:

- `session_uuid`
- `session_start_iso8601_utc`
- `rig_id`
- `arena_id`
- `camera_id`
- `canvas_name`
- `protocol_name_from_definition`
- `loaded_protocol_filepath`

Relevant code:

- `src/fisheye/analysis/import_stimulus_to_zarr.py:317`
- `src/fisheye/utils/organize_recordings.py:92`
- `src/fisheye/utils/import_recordings_analysis.py:72`
- `src/fisheye/utils/import_recordings_training.py:75`

#### Recording manifest JSON

Current downstream backfill and intake logic reads:

- `arena_id`
- `camera_id`
- `rig_id`
- `canvas_name`
- `protocol_name_from_definition`
- `session_uuid`
- `recording_name`
- `session_start_iso8601_utc`

Relevant code:

- `src/fisheye/utils/organize_recordings.py:482`
- `src/fisheye/utils/intake_video_only_recording.py:177`
- `src/fisheye/registry/maintenance.py:789`

#### Filenames and recording layout

Current recording organization and import code uses camera-centric filename
patterns:

- `Cam{camera_id}.mp4`
- `Cam{camera_id}_meta.csv`
- renamed forms such as `Cam{camera_id}_{session_tag}.mp4`

The recording folder name is usually derived from the H5 stem or recording name,
and examples commonly embed strings like `arena_1`, but most code does not parse
`arena_id` from the folder name.

Relevant code:

- `src/fisheye/utils/organize_recordings.py:155`
- `src/fisheye/utils/organize_recordings.py:351`
- `src/fisheye/utils/import_recordings_training.py:106`
- `docs/organize_recordings_logging_schema.md`

#### Registry, query, and reporting surfaces

Current registry/query/reporting consumers use exact `arena_id` fields for:

- filtering
- grouping
- display
- context completeness checks

Relevant code:

- `src/fisheye/registry/db.py:8447`
- `src/fisheye/registry/db.py:12541`
- `src/fisheye/registry/query.py:55`
- `src/fisheye/registry/query.py:280`
- `src/fisheye/utils/registry_query.py:616`
- `src/fisheye/utils/check_training_registry.py:3055`
- `src/zarr_inspector.py:158`

#### Model selection and training composition

Current model/profile consumers treat `arena_id` as part of dataset composition
and similarity matching.

Relevant code:

- `src/fisheye/utils/resolve_detect_model.py:15`
- `src/fisheye/utils/resolve_detect_model.py:181`
- `src/fisheye/utils/detection_profile.py:22`
- `src/fisheye/utils/detection_profile.py:282`
- `src/fisheye/utils/aggregate_keypoint_training_data_card.py:1482`

#### Derived ROI and tracking outputs

Analysis code already uses arena-like IDs at the ROI level via:

- `arena_ids`
- `n_detections_per_arena`
- `track_arena_ids`

Relevant code:

- `src/fisheye/tracking/arena_assignment.py:557`
- `src/fisheye/tracking/arena_assignment.py:595`
- `src/fisheye/tracking/single_subject_per_arena.py:114`
- `src/fisheye/tracking/single_subject_per_arena.py:223`
- `src/fisheye/analysis/compute_speed.py:753`
- `src/fisheye/analysis/track_kinematics.py:471`

### What Downstream Code Currently Assumes `arena` Means

Palette currently uses `arena` in several different ways.

#### 1. Recording context key

At the recording/session/registry level, `arena_id` behaves like a stable
context key that distinguishes one recording context from another.

Relevant code:

- `src/fisheye/registry/query.py:263`
- `src/fisheye/utils/resolve_detect_model.py:15`
- `src/fisheye/utils/registry_query.py:616`

#### 2. Physical area or dish/chamber

In calibration and overlay utilities, arena config refers to the physical
experimental region and its geometry.

Relevant code:

- `src/fisheye/visualization/overlay_arena_mask.py:31`
- `src/fisheye/visualization/overlay_arena_mask.py:69`
- `src/fisheye/analysis/import_stimulus_to_zarr.py:251`

#### 3. ROI or sub-region within a view

In arena assignment and ROI-based filtering, the effective analysis meaning is
already “which sub-region did this detection land in.”

Relevant code:

- `src/fisheye/tracking/arena_assignment.py:413`
- `src/fisheye/tracking/arena_assignment.py:561`
- `src/fisheye/analysis/chaser_phase_analysis.py:373`

#### 4. Track identity namespace

For current single-subject-per-arena tracking, track identity is derived from
occupied arena IDs. In practice this means the current tracking pipeline treats
arena as the smallest independently analyzed unit.

Relevant code:

- `src/fisheye/tracking/single_subject_per_arena.py:120`
- `src/fisheye/tracking/single_subject_per_arena.py:145`
- `src/fisheye/tracking/single_subject_per_arena.py:160`
- `src/fisheye/analysis/compute_speed.py:778`
- `src/fisheye/analysis/track_kinematics.py:1838`

### Current Concrete Assumptions

The current codebase behaves as if these statements are approximately true for a
single recording artifact:

- one recording artifact has one `arena_id`
- one recording artifact has one `camera_id`
- one recording artifact has one protocol context
- one recording directory corresponds to one H5/session context
- one analysis import target corresponds to one camera stream

Multi-camera import is not yet supported by the main batch analysis import path.

Relevant code:

- `src/fisheye/utils/import_recordings_analysis.py:101`
- `src/fisheye/utils/import_recordings_analysis.py:122`
- `src/fisheye/utils/import_recordings_training.py:106`
- `src/fisheye/utils/organize_recordings.py:545`

### Naming Assumptions

Most code treats `arena_id` as an opaque string, not a required `arena_N`
pattern. For example, documentation already includes non-`arena_N` values such
as `scope_stage_a`.

However, there is at least one helper that extracts a number from names like
`arena_1` using:

- `arena[_-]?(\\d+)`

Relevant code:

- `docs/recording_manifest_contract.md`
- `src/fisheye/diagnostics/prepare_detect_training.py:336`

So `arena_1` is mostly a convention, not a universal contract, but some
downstream logic does still recognize that pattern.

### What Does Not Currently Use `arena`

Event/timeline readers currently do not parse a dedicated `arena` field from the
event log payload. They work from general event fields such as event type,
context name, stimulus mode, and timestamps.

Relevant code:

- `src/fisheye/utils/read_h5_data.py:124`
- `src/fisheye/visualization/visualize_experiment_timeline_combined.py:146`

### Backward-Compatibility Risk if `arena_id` Is Repurposed

If Citrus changed top-level `arena_id` to mean “parent physical arena containing
multiple sub-arenas,” the main failure mode would be semantic collapse.

#### Likely breakage

- registry filters and reports would merge recordings that are currently treated
  as distinct contexts
- model selection and profile composition would match on the wrong level of
  granularity
- session metadata would no longer align cleanly with ROI-level `arena_ids` and
  `track_arena_ids`
- current single-camera import/layout assumptions would become an awkward fit
  for multi-camera-per-parent-arena acquisition

Relevant code:

- `src/fisheye/utils/resolve_detect_model.py:15`
- `src/fisheye/utils/detection_profile.py:22`
- `src/fisheye/utils/registry_query.py:616`
- `src/fisheye/tracking/single_subject_per_arena.py:163`
- `src/fisheye/utils/import_recordings_analysis.py:110`

### Additive Schema Tolerance

Current consumers are fairly tolerant of additive metadata, but only partially.

#### Generally tolerant

- JSON-based `session_context` readers typically load a dict and ignore unknown
  keys
- manifest validation does not enforce a closed schema for extra fields
- manifest backfill reads a fixed subset of known keys and ignores the rest

Relevant code:

- `src/fisheye/registry/db.py:1849`
- `src/fisheye/utils/validate_recording_manifest.py:173`
- `src/fisheye/registry/maintenance.py:789`

#### Not automatically propagated

H5/session import code uses fixed known-key lists, so newly added Citrus fields
will usually be ignored until Palette import code is updated to mirror them.

Relevant code:

- `src/fisheye/analysis/import_stimulus_to_zarr.py:317`
- `src/fisheye/utils/organize_recordings.py:97`

### Operational Conclusion

The current Palette side is compatible with additive fields such as:

- `physical_arena_id`
- `zone_id`
- `stream_id`
- `camera_ids`
- `primary_camera_id`
- `visible_zone_ids`

It is not compatible with silently changing the meaning of existing `arena_id`.

That is the reason this document recommends:

- keeping legacy `arena_id` stable
- treating it as a compatibility alias for `zone_id`
- introducing parent-arena and stream-level identifiers as new fields rather
  than redefining the old one
