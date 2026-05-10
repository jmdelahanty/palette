# V2 Tabular Identity Migration Checklist

<!-- design-meta
status: draft
last_updated: 2026-05-06
-->

## Purpose

This checklist consolidates the v2 direction for Palette analysis archives:

- reduce Zarr object count by moving group-per-variant schemas toward compact
  tabular/indexed layouts
- preserve the raw/refined/derived artifact policy
- make multi-subject identity safe before the pipeline needs true
  multi-subject tracking
- make downstream staleness and exports traceable to exact source revisions
- keep each recording archive as the canonical unit of provenance while
  representing projects/cohorts as reproducible virtual collections

This is not a migration script. It is the implementation order and contract
checklist for future code changes.

For a writer-by-writer status inventory and migration priority list, see
[analysis_writer_compact_layout_inventory.md](analysis_writer_compact_layout_inventory.md).

## Archive, Collection, And Project Model

Do not make a SLEAP-style mutable physical project container the canonical
unit of analysis. Palette's canonical unit is the recording archive. Projects,
cohorts, protocol groups, and training sets should be represented as manifests
or registry queries that point back to exact recording archives and run IDs.

| Layer | Role | Authority |
| --- | --- | --- |
| Recording analysis Zarr | Canonical per-recording archive for raw/model outputs, refined authoring surfaces, and derived analysis runs | Authoritative for that recording |
| Registry | Searchable index/cache over archives, protocols, runs, freshness, and provenance | Rebuildable; not authoritative |
| Manifest/collection | Virtual project or cohort definition with exact recording IDs, run selections, filters, and export parameters | Reproducible selection boundary |
| Export lake | Parquet/DuckDB analytics product for cross-session queries | Rebuildable sidecar |
| Training Zarr | Versioned derived training dataset assembled from recording archives | Derived artifact with source refs |

The design should borrow SLEAP's useful conceptual separation between raw
predictions, user-corrected instances, tracks, skeleton/schema metadata, and
exports, but not its assumption that a mutable multi-video project file is the
main source of truth.

Within one recording archive, the SLEAP-like mutable labeling container maps
to the refined authoring surfaces, not to the whole Zarr. The archive is a
provenance container with different mutability zones:

- `detect_runs/...`, raw model keypoints, and raw model masks are immutable
  model-output provenance artifacts.
- `refined_detect_runs/...`, `refined_keypoints_runs/...`, and
  `refined_subject_mask_runs/...` are editable or revisioned authoring
  surfaces for curated labels/instances.
- `analysis/...` contains rebuildable derived runs that must point to exact
  refined source refs and revisions.
- `exports` and external Parquet/DuckDB products are rebuildable query
  products, not labeling surfaces.

This means manual repair, identity correction, reprediction into a refined
surface, and user-authored additions should update refined revisions and edit
provenance. They must not rewrite the raw prediction artifacts they were
derived from, and they must not silently leave downstream analysis runs looking
fresh when their source refined revisions changed.

Virtual project manifests should record:

- exact recording/archive IDs
- exact source run IDs and source revisions/fingerprints
- query/filter criteria used to construct the collection
- run-selection policy, including whether `latest` was allowed during
  collection construction
- export schema/version and export parameters
- generated export artifact IDs, when materialized

Serious analytics should resolve `latest` into concrete run IDs before writing
export rows. A manifest may record that `latest` was the selection policy, but
the exported data should still carry the concrete resolved run IDs and source
fingerprints used at build time.

Avoid turning the registry or export lake into a competing authority. If the
registry database or Parquet lake is deleted, it must be possible to rebuild it
from the recording archives plus collection/export manifests. Conversely,
avoid copying large complete metadata blobs into every derived run when
compact `source_refs`, `source_fingerprints`, schema IDs, and manifest IDs are
sufficient.

Deferred future implementation: add protocol-derived trial descriptors as a
search/indexing layer, not as fuzzy semantic hashes. Exact protocol semantic
hashes should remain strict equality checks for semantically identical
protocols. A later Citrus/registry/Palette integration should add a versioned
trial-index payload, for example `protocol_trial_index_json`, with normalized
step descriptors such as `stimulus_family`, `trial_family`, duration,
direction, speed, spatial frequency, contrast, radial polarity, and center
metadata where applicable. The registry can then support faceted/range search,
bucketed trial hashes, and eventually example-based similarity queries such as
"all moving-grating trials like these protocols" without weakening the meaning
of the exact semantic hash.

## Core Identity Model

Keep these identifiers separate:

| Identifier | Meaning | Not |
| --- | --- | --- |
| `refined_row_id` | Stable artifact row identity for one curated instance row | Not a fish, arena, or track identity |
| `source_detect_row_index` | Raw detect row lineage inside one immutable raw detect run | Not stable across raw detect runs |
| `arena_id` | Spatial container assignment, such as dish/chamber/lane | Not temporal subject identity |
| `track_id` | Run-local temporal identity from one `tracking_runs/<run>` | Not a registry subject ID |
| `subject_id` | Optional biological/registry identity when known | Not inferred solely from image tracking |

Identity swaps should create a new tracking or identity-assignment revision.
They must not mutate detection rows or mask rows in place.

## Artifact Mutability Policy

| Artifact family | Policy |
| --- | --- |
| Raw/model outputs | Row-immutable provenance artifacts. Whole-run overwrite may exist only as explicit destructive replacement. |
| Refined surfaces | The mutable labeling/authoring layer for one recording, with stable row IDs, revisions, and edit provenance. |
| Derived analysis | Rebuildable outputs tied to exact source refs, source revisions, and method versions. |
| Compatibility surfaces | Readable legacy/adaptor layouts, not competing authorities. |
| Exports | Rebuildable sidecars or lakes tied to source run IDs and export run IDs. |

V2 compact layout is only a physical storage change. It must not weaken these
mutability rules.

## Shared V2 Metadata To Add

Add shared helpers and docs for these fields before rewriting individual
families:

- `artifact_mutability`: `raw_immutable | refined_authoring | derived_rebuildable | compatibility_cache`
- `schema_id`
- `schema_version`
- `method`
- `method_version`
- `source_refs`
- `source_fingerprints`
- `source_revision`
- `authoring_revision`
- `row_revision`
- `edit_log` or `edit_events`
- `row_tombstones` for deleted refined rows whose IDs must not be reused

Recommended fingerprint model:

- immutable raw source: run ID plus schema/method/model/config metadata and
  optional content hash
- refined source: run ID plus `authoring_revision`
- row-derived source: row IDs plus row/component revisions when available
- derived source: source refs plus method parameters and source lineage hash

## Implementation Phases

### Phase 1: Contract And Resolver Foundation

- [ ] Create shared v2 provenance/revision helper functions.
- [ ] Define JSON-safe attr normalization for non-finite values and nested
      source metadata.
- [ ] Add resolver helpers before changing writers:
  - `resolve_refined_detect_instances(...)`
  - `resolve_track_motion_arrays(...)`
  - `resolve_swim_bout_tables(...)`
  - `resolve_bout_kinematics_tables(...)`
  - `resolve_eye_angle_representation(...)`
  - `resolve_subject_shape_arrays(...)`
  - `resolve_refined_subject_mask_arrays(...)`
- [ ] Each resolver should prefer v2 compact layout and fall back to current
      v1 hierarchical layout.
- [ ] Add tests proving readers tolerate both layouts.

### Phase 2: Raw And Refined Detection

- [ ] Mark `detect_runs/<run>` as row-immutable raw model output.
- [ ] Define whether raw physical row index is the stable raw row ID within a
      run, or add explicit `raw_row_ids`.
- [ ] Add raw detect run fingerprint metadata.
- [ ] Extend `refined_detect_runs/<run>/instances` with:
  - `row_revision`
  - non-reused `refined_row_ids`
  - tombstones for deleted rows
  - edit operation history
  - source run/revision/fingerprint attrs
- [ ] Disable frame-singleton row-ID fallback for v2 multi-subject refined
      detection surfaces.
- [ ] Update refined-detect validation to check revision monotonicity, no ID
      reuse, valid tombstones, and source revision match.
- [ ] Replace fixed-slot detect review with sparse row editing that supports
      multiple curated instances in the same frame and arena.

### Phase 3: Spatial Assignment And Tracking

- [ ] Keep `arena_assignment_runs` spatial-only.
- [ ] Record source rowset path, source revision, and source fingerprint on
      arena assignment runs.
- [ ] Fix schema drift between `ARENA_ASSIGNMENT_SPEC` and the writer
      (`confidence` vs `n_detections_per_arena`).
- [ ] Keep `single_subject_per_arena` as the strict current workflow.
- [ ] Add a separate multi-subject tracking method with the same public
      `tracking_runs` contract.
- [ ] Tracking rows should include:
  - `track_ids`
  - `arena_ids`
  - `source_row_indices`
  - `source_refined_row_ids`
  - `source_detect_row_index`
  - optional confidence/status
  - swap/split/merge/reassignment events
- [ ] Identity swaps should create a new tracking run or identity-assignment
      revision and mark dependent analyses stale.

### Phase 4: Downstream Lineage And Staleness

- [ ] Update `track_kinematics` to carry exact per-sample lineage:
  - `source_refined_row_id`
  - `source_detect_row_index`
  - source rowset path
  - source tracking run
  - source revision/fingerprint
- [ ] Define detect/crop edit policy:
  - row-stable bbox edits may produce targeted stale payloads
  - add/delete/split/merge changes force rerun or broader invalidation
- [ ] Add `source_detect_stale` and/or `source_crop_stale` only if targeted
      stale repair is implemented. Otherwise explicitly document rerun-only.
- [ ] Keep stale state separate from review state.
- [ ] Project source revision/stale state into registry/query surfaces.
- [x] Add a read-only derived-analysis staleness audit CLI:
      `scripts/py -m fisheye.utils.audit_analysis_staleness <archive>.zarr`.
      The current implementation resolves same-archive `source_refs` and common
      `source_*_run` attrs, checks source fingerprint mismatches, reports
      explicit source stale payloads, and warns on unverifiable lineage or
      non-latest sources.

### Phase 5: Compact Analysis Layouts

Prefer fewer run groups with index tables and enum columns.

Common indexes:

```text
source_index/
track_index/
subject_index/
variant_index/
component_index/
```

Family-specific changes:

- [ ] `analysis/track_kinematics_runs`: move toward run-level ragged/CSR
      arrays instead of `tracks/id_<track_id>` subtrees as the only layout.
- [ ] `analysis/swim_bout_runs`: use `candidate_index`, `signal_index`,
      `bouts`, `peak_events`, and optional detector signals rather than
      one group per parameter/speed-level variant.
- [ ] `analysis/bout_kinematics_runs`: collapse heading/source variants into
      enum columns such as `heading_level_id`.
- [ ] `analysis/stimulus_response_runs`: replace `fish_id` language with
      `track_id` plus optional `subject_id`; store step/bout/window tables
      rather than step subtrees where practical.
- [ ] `analysis/eye_angle_runs`: keep canonical major/gaze/body-frame arrays
      and variant transforms; materialize aliases/smoothed/delta surfaces only
      as compatibility/cache outputs.
- [ ] `analysis/subject_shape_runs`: stack common component metrics by
      component axis; keep body-only geometry semantic.
- [ ] `analysis/tail_kinematics_runs`: preserve the compact dense shape, but
      normalize source index/revision fields.

### Phase 6: Exports And Registry

- [ ] Define a manifest/collection schema that acts as the virtual project
      boundary for cross-recording work. See
      [virtual_collection_manifest_schema.md](virtual_collection_manifest_schema.md).
- [ ] Export builders should resolve `latest` selections to concrete source
      run IDs before writing rows.
- [ ] Export tables should include source run IDs, source lineage hash,
      protocol signature hash, track ID, optional subject ID, and source
      revision.
- [ ] Registry should expose freshness/staleness for source refs and derived
      exports. Partial: first-pass derived-analysis presence is now visible in
      `recording_step_status` for track kinematics, swim bouts, bout
      kinematics, eye angles, subject shape, tail kinematics, tail posture
      views, bout classification, and stimulus response. Remaining work is
      broader semantic freshness: tail behavior, bout-kinematics, and
      stimulus-response run families now compare stored source refs against
      current upstream selections and render `STALE`/`UNVER` in the wide status
      view, but the other derived run families, source revisions/fingerprints,
      and export propagation still need the same treatment.
- [ ] Registry rows should remain rebuildable indexes over canonical Zarr
      archives and manifests, not the authoritative source of analysis truth.
- [ ] Parquet sidecars may be durable analytics products, but canonical Zarr
      source refs must remain sufficient to rebuild them.

## Compatibility Policy

- Do not break old archives while readers are migrating.
- New resolvers must read both v1 hierarchical and v2 compact layouts.
- Crimson and Marimo should consume resolver-level semantics rather than
  hardcoded physical paths.
- Compatibility arrays are allowed when needed by viewers, but they should be
  labeled as caches or compatibility surfaces.
- Broad parameter sweeps should live in scratch/sidecar outputs unless
  explicitly promoted to a canonical run.

## Open Decisions

1. Should row-stable refined-detect/crop edits produce targeted stale payloads,
   or should all crop-affecting edits force reruns?
2. Should raw detect runs add explicit `raw_row_ids`, or is physical row index
   stable enough inside an immutable run?
3. Are refined revisions run-level only, row-level only, or both?
4. Should biological `subject_id` assignment live in the registry only, or in a
   Zarr `subject_identity_runs` family that links tracks to subjects?
5. Which v2 compact layouts must still materialize compatibility arrays for
   Crimson before Crimson gets v2 readers?
6. Which accepted analysis products should be Zarr-native, and which should be
   Parquet sidecars only?
7. What is the minimal manifest schema for virtual projects/cohorts?
8. Should production exports forbid unresolved implicit `latest` selection by
   default?
9. What should the first version of `protocol_trial_index_json` contain, and
   should Citrus emit it directly or should Palette backfill it from Citrus H5
   protocol snapshots?

## References

- [current_pipeline_contract.md](current_pipeline_contract.md)
- [analysis_zarr_object_count_schema_direction.md](analysis_zarr_object_count_schema_direction.md)
- [analysis_writer_compact_layout_inventory.md](analysis_writer_compact_layout_inventory.md)
- [refined_detect_sparse_instances_schema.md](refined_detect_sparse_instances_schema.md)
- [refined_detect_row_identity_contract.md](refined_detect_row_identity_contract.md)
- [realtime_sparse_row_index_contract.md](realtime_sparse_row_index_contract.md)
- [track_identity_target_architecture.md](track_identity_target_architecture.md)
- [single_subject_per_arena_tracking_contract.md](single_subject_per_arena_tracking_contract.md)
- [repo_wide_staleness_policy.md](repo_wide_staleness_policy.md)
- [repo_wide_staleness_gap_matrix.md](repo_wide_staleness_gap_matrix.md)
- [derived_analysis_run_contract.md](derived_analysis_run_contract.md)
- [zarr_parquet_sidecar_exports_design.md](zarr_parquet_sidecar_exports_design.md)
- [virtual_collection_manifest_schema.md](virtual_collection_manifest_schema.md)
