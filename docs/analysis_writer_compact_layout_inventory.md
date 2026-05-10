# Analysis Writer Compact Layout Inventory

<!-- design-meta
status: draft
last_updated: 2026-05-10
-->

## Purpose

This inventory records which Palette writers currently use compact/tabular
layouts, which still use group-per-variant or component-per-group layouts, and
which families should be migrated first. It is a companion to
`analysis_zarr_object_count_schema_direction.md` and
`v2_tabular_identity_migration_checklist.md`.

The goal is not to make every writer look identical. Compact tabular layout is
valuable when the current physical tree fans out by candidate, signal,
component, step, or representation. Dense semantic arrays should stay dense when
that is the natural access pattern.

## Summary

`analysis/swim_bout_runs` and `analysis/bout_kinematics_runs` now have
compact-v2 writers behind explicit layout flags. Both defaults remain
hierarchical v1 until external-reader confidence is high enough to switch
defaults deliberately.

Most other writers are still hierarchical, but they are not equally urgent:

- `bout_classification_runs`, `tail_kinematics_runs`, and
  `tail_posture_view_runs` are already compact enough for their current scope.
- `track_kinematics_runs` has a grouped v2 speed layout, but still stores tracks
  as `tracks/id_<track_id>` subtrees and materializes compatibility arrays.
- `stimulus_response_runs`, `eye_angle_runs`, `subject_shape_runs`, and
  `refined_subject_masks_runs` are the main remaining future migration
  candidates.

## Writer Inventory

| Writer family | Current layout | Compact status | Migration priority | Recommendation |
| --- | --- | --- | --- | --- |
| `analysis/swim_bout_runs` | Hierarchical v1 by speed level, plus compact-v2 opt-in | Compact-v2 implemented, not default | High, nearly ready | Keep resolver-first policy. After Crimson/Marimo confidence, switch default to `compact_v2` and keep `--layout hierarchical_v1` as explicit compatibility. |
| `analysis/swim_bout_runs` legacy statistics writer | Flat legacy run written by `swim_bout_statistics.py` | Not compact-v2 and not the canonical detector writer | Low | Treat as historical/reporting output. Do not use it as the model for future bout-segmentation storage. |
| `analysis/track_kinematics_runs` | `online/offline/<run>/tracks/id_<track>/...`; also writes grouped `movement/speed/<level>/...` | Partial v2 grouping plus initial logical loader, not compact tabular | Medium | Do not rewrite immediately. Continue moving readers through `fisheye.analysis.track_kinematics_io` first. Future compact layout should use run-level track index plus ragged/CSR arrays instead of one subtree per track. |
| `analysis/bout_kinematics_runs` | Hierarchical v1 by domain/heading variant, plus compact-v2 opt-in tables `level_index`, `movement_metrics`, `heading_metrics`, and optional `eye_gaze_metrics` | Compact-v2 implemented, not default | High, reader validation in progress | Keep resolver-first policy. After Crimson/Marimo confidence, consider switching new canary/batch runs to `compact_tabular_v2`; keep hierarchical v1 as explicit compatibility. |
| `analysis/stimulus_response_runs` | Hierarchical v1 by step/family, plus compact-tabular-v2 opt-in summary/bout/window tables | Compact-v2 implemented, not default | Medium-high | `stimulus_response_io.resolve_stimulus_response_tables(...)` covers hierarchical-v1 and compact-v2. The first compact slice writes step/global/base/family per-fish/per-bout/window/trial tables and intentionally omits high-volume per-frame/time-series tables. See `stimulus_response_compact_v2_design.md`. |
| `analysis/eye_angle_runs` | `angles/roi`, `angles/frame`, `qa`, `support`; many persisted representations, aliases, smoothed and delta arrays | Logical resolver implemented; writer still materializes many arrays | Medium-high | Continue moving readers through `fisheye.analysis.eye_angle_io` before writer changes. Future migration should store canonical major/gaze/body-frame arrays plus transform metadata and keep accepted compatibility caches for established consumers. This is mostly a repack/derive migration, not a scientific recompute, when canonical arrays exist. |
| `analysis/subject_shape_runs` | `components/<component>/...`, `relations/...`, `body_frame/...`, body-specific centerline/tail geometry | Hierarchical by component, with many common metric mirrors | Medium | Stack common component metrics along a component axis. Keep specialized body-only geometry in semantic groups. Do not flatten centerline/tail geometry into generic component tables. |
| `refined_subject_masks_runs` | Dense `masks_roi` plus component-local metrics/QC/review groups | Canonical dense mask is appropriate; component mirrors fan out | Medium | Keep `masks_roi` dense. Future layout should stack common component metrics/QC as `(row, component)` arrays and reserve component groups for true component-specific authoring state. |
| `analysis/tail_kinematics_runs` | Run-level dense arrays such as `tail_angle_rad (N,K)`, `tail_lateral_deflection_px (N,K)`, row lineage | Already compact for current single source | Low | Do not migrate now. Add source revision/fingerprint consistency as v2 lineage work, not a physical layout rewrite. |
| `analysis/tail_posture_view_runs` | Run-level dense tool-compatible arrays such as `tail_keypoints_xy`, `tail_angle_rad`, row lineage | Already compact for current single view | Low | Do not migrate now. If multiple tool views are persisted later, add a `view_index` rather than one run per minor view. |
| `analysis/bout_classification_runs` | Single `per_bout` columnar table plus run attrs | Already compact | Low | Keep as-is. If multiple classifiers are compared, prefer classifier rows/attrs or separate promoted runs, not nested classifier subgroups. |
| `analysis/stimulus_runs` | `events`, `frame_alignment`, `steps/step_<n>/...`, calibration/stimulus-coordinate metadata | Hierarchical import/protocol metadata | Low-medium | Leave stable for now. A compact canonical `steps` table would help Crimson/query code, but this is not the main object-count source compared with derived response runs. |
| `analysis/chaser_fish_metrics` | Legacy/specialized metric run layout | Hierarchical legacy | Low | Do not prioritize. Fold future behavior into stimulus-response or explicit derived metric tables if the analysis is revived. |
| `analysis/speed_runs` from `compute_speed.py` | Legacy `tracks/id_<track>/...` speed arrays | Legacy | Low | Treat as historical. New work should use `analysis/track_kinematics_runs`. |
| Profile/quality/dashboard writers | Mostly per-run or per-latest summary tables and visualization arrays | Mixed | Low | Keep run-local dashboards compact. For cross-recording queries, rely on registry/profile tables and Parquet exports rather than expanding Zarr dashboards. |

## Current Compact-V2 Readiness

### Ready Or Nearly Ready

- `analysis/swim_bout_runs`
  - Writer exists behind `--layout compact_v2`.
  - Python resolver exists in `fisheye.analysis.swim_bout_io`.
  - Crimson compact loader smoke has passed on the audited canaries.
  - Marimo-backed discovery and main Palette consumers use the resolver for
    promoted compact runs; focused `test_interactive_track_kinematics.py`
    parity passed outside the sandbox on 2026-05-09.
  - Internal compatibility update, 2026-05-09:
    `run_movement_bout_batch_pipeline.py` validates compact logical bout
    tables, and the legacy `track_kinematics.py` swim-bout mirror reads
    compact and hierarchical runs through `swim_bout_io.py`.
  - Remaining steps before default switch:
    - complete the deferred Crimson visual check for rendered bout overlays;
    - decide whether this is enough external-reader confidence to switch the
      `detect_bouts_multi_level` default from hierarchical v1 to compact v2.

- `analysis/bout_classification_runs`
  - Already uses a single `per_bout` columnar table.
  - No physical-layout migration is needed for the current scope.

- `analysis/bout_kinematics_runs`
  - Writer exists behind `--layout compact_tabular_v2`.
  - Python resolver exists in `fisheye.analysis.bout_kinematics` as
    `resolve_bout_kinematics_tables(...)`.
  - Marimo-backed interactive loading and cross-recording Parquet export now
    use the resolver.
  - Focused tests passed outside the sandbox on 2026-05-10:
    `test_bout_kinematics.py` and `test_export_cross_recording_analytics.py`.
  - A real feeding canary run was written and resolver-validated on
    2026-05-10:
    `bk_tk_hyst4_low2_latch_s005_peak_event_prom4_w098_compact_v2_canary_20260510`.
  - Remaining steps before default switch:
    - complete Crimson smoke/visual checks for compact bout-kinematics reads;
    - decide whether visualization artifact specs need compact source paths
      before allowing `--write-zarr-artifacts` with compact-v2.

### Validated Compact Reader Target

- `analysis/stimulus_response_runs`
  - Writer exists behind `--layout compact_tabular_v2`.
  - A shared logical resolver reads hierarchical-v1 and compact-tabular-v2:
    `fisheye.analysis.stimulus_response_io.resolve_stimulus_response_tables(...)`.
  - The cross-recording exporter, the Marimo `track_kinematics_explorer.py`
    stimulus-response panels, and `plot_stimulus_response_omr.py` now use that
    resolver for the moving-grating OMR and concentric radial-OMR reader paths.
  - Focused tests passed outside the sandbox on 2026-05-10:
    `test_stimulus_response_io.py`, `test_stimulus_response.py`,
    `test_plot_stimulus_response_omr.py`, and
    `test_export_cross_recording_analytics.py`.
  - Real DefaultScreen canary validation, 2026-05-10:
    `stimulus_response_tk_hyst4_low2_latch_s005_omr_compact_v2_canary_20260510`
    passed strict JSON, resolver parity against the hierarchical canary,
    Parquet export, OMR plot generation, and `marimo check`.
  - The writer path now reads upstream track-kinematics inputs through
    `fisheye.analysis.track_kinematics_io.load_track_kinematics_track(...)`
    instead of hard-coding `tracks/id_<track>/...` arrays.
  - Remaining before default switch:
    - decide whether high-volume per-frame/time-series tables should stay
      omitted, become optional, or be written in a separate compact table family;
    - decide whether visualization artifact specs need logical table names before
      allowing compact runs with `--write-zarr-artifacts`.
  - Design details live in
    `docs/stimulus_response_compact_v2_design.md`.

### Resolver-First Track-Kinematics Work

- `analysis/track_kinematics_runs`
  - Initial logical loader exists in `fisheye.analysis.track_kinematics_io`.
  - The loader reads current hierarchical runs, prefers grouped
    `movement/speed/<level>` arrays, and falls back to compatibility flat arrays.
  - `detect_bouts_multi_level` now loads track speed/position inputs through that
    resolver, reducing one direct dependency on `tracks/id_<track>/...`.
  - `stimulus_response` now expands sparse track inputs through the same logical
    loader before computing dense stimulus-response metrics.
  - `plot_track_kinematics` now uses the logical loader for plot data while
    retaining existing physical source-path metadata for artifact specs.
  - Focused tests and a non-mutating DefaultScreen real-Zarr smoke passed on
    2026-05-10.

### Resolver-First Eye-Angle Work

- `analysis/eye_angle_runs`
  - Initial logical loader exists in `fisheye.analysis.eye_angle_io`.
  - The loader reads current hierarchical `angles/roi`, `angles/frame`,
    `qa/roi`, `qa/frame`, and `support` arrays and exposes run discovery,
    logical tables, frame alignment, and bout eye-gaze frame-series helpers.
  - `bout_kinematics` now loads optional bout eye-gaze inputs through this
    resolver.
  - `interactive_track_kinematics.py` now discovers eye-angle runs and builds
    Marimo eye-angle time-series data through the resolver.
  - Focused tests passed outside the sandbox on 2026-05-10:
    `test_eye_angle_io.py`, `test_interactive_track_kinematics.py`, and
    `test_bout_kinematics.py`.

### Needs Resolver Before Writer Changes

- `analysis/subject_shape_runs`
- `refined_subject_masks_runs`

For these families, first add resolver/helper APIs that return logical tables or
arrays without exposing physical paths. Then add compact writer support behind an
explicit opt-in. Only switch defaults after Palette, Marimo, Crimson, and export
consumers are verified against both layouts.

## Recommended Migration Order

1. **Finish swim-bout compact-v2 default migration.**
   Make `detect_bouts_multi_level --layout compact_v2` the default only after
   the current Crimson visual smoke is accepted. Keep `--layout hierarchical_v1`
   as an explicit compatibility option.

2. **Finish bout-kinematics compact-v2 external-reader validation.**
   The writer and Palette resolvers exist. The next decision is whether Crimson
   and visualization/export consumers are sufficiently validated to make
   compact-v2 the preferred canary/batch layout, while keeping hierarchical v1
   as compatibility.

3. **Stimulus response compact-v2 default decision.**
   The opt-in writer, resolver, export/Marimo/plot consumers, and a real canary
   validation now exist. The remaining design decision is whether high-volume
   per-frame/time-series outputs stay omitted, become optional, or move into a
   separate compact table family before any default switch.

4. **Eye-angle canonical/variant repack.**
   Keep canonical major/gaze/body-frame arrays and variant metadata. Materialize
   aliases, frame copies, smoothed arrays, and delta arrays only when they are
   accepted compatibility caches or expensive enough to justify persistence.

5. **Component metric stacking.**
   For `subject_shape_runs` and `refined_subject_masks_runs`, stack common
   component metrics into `(row, component, ...)` arrays while preserving dense
   masks and body-specific geometry.

6. **Track kinematics ragged run-level layout.**
   Defer until multi-track/multi-subject tracking pressure is real. The current
   grouped movement layout already improves semantic clarity, and a rushed
   track-layout rewrite would touch many readers.

## Non-Goals

- Do not force dense semantic arrays into row tables just to make everything
  look tabular.
- Do not remove v1 readers while historical canaries and Crimson still depend
  on them.
- Do not write compatibility mirrors under compact runs unless a named consumer
  requires them.
- Do not compact raw/model-output provenance artifacts in a way that weakens the
  raw/refined/derived mutability policy.

## Acceptance Criteria For Any Compact Migration

- A logical resolver reads both old and new layouts.
- The writer records `schema_id`, `schema_version`, `method`,
  `method_version`, `layout`, `source_refs`, and source revisions/fingerprints
  where available.
- A focused equivalence test compares old and new logical outputs on at least
  one canary-sized fixture.
- Marimo/Crimson/export consumers either use the resolver or have explicit
  compatibility tests.
- Strict JSON validation passes; attrs must not contain `NaN`, `Infinity`, or
  `-Infinity`.
- New layout reduces group/array object count for the target family or gives a
  clear reader/provenance benefit.
