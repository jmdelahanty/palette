<!-- ARCHIVED 2026-07-04: dated point-in-time snapshot / spent work ticket, retained for history only. -->

# Compact-V2 Readiness Audit - 2026-05-11

<!-- audit-meta
status: completed
last_verified: 2026-05-11
purpose: Concrete audit of compact-default readiness across writer defaults, readers, visualization artifacts, docs/contracts, and data-lifecycle policy.
-->

## Objective

Audit the five compact-readiness areas requested after the compact-v2 rollout:

1. Writers default to compact layouts where intended, with hierarchical layouts
   retained only as explicit legacy/debug options.
2. Readers, exporters, Marimo helpers, plotters, and batch validators use
   logical resolver APIs or equivalent layout-aware reads instead of assuming
   hierarchical physical paths.
3. Visualization artifacts work from compact runs and record layout-aware
   logical source metadata.
4. Contracts and docs accurately describe current compact defaults and
   intentional deferrals.
5. Data-lifecycle docs preserve the raw/refined/derived mutability policy.

## Checklist And Evidence

| Requirement | Evidence inspected | Status |
| --- | --- | --- |
| Swim-bout writer defaults to compact-v2 | `src/fisheye/analysis/detect_bouts_multi_level.py` sets `SWIM_BOUT_LAYOUT_DEFAULT = SWIM_BOUT_LAYOUT_COMPACT_V2`; `tests/unit/fisheye/test_swim_bout_layout_defaults.py` covers the function and CLI defaults. | Pass |
| Bout-kinematics writer defaults to compact-v2 | `src/fisheye/analysis/bout_kinematics.py` sets `BOUT_KINEMATICS_LAYOUT_DEFAULT = LAYOUT_COMPACT_TABULAR_V2`; `tests/unit/fisheye/test_bout_kinematics.py` covers defaults and compact artifact output. | Pass |
| Stimulus-response writer defaults to compact-v2 | `src/fisheye/analysis/stimulus_response.py` sets `STIMULUS_RESPONSE_LAYOUT_DEFAULT = STIMULUS_RESPONSE_LAYOUT_COMPACT_V2`; `tests/unit/fisheye/test_stimulus_response.py` covers the default. | Pass |
| Eye-angle writer defaults to compact dense v2 | `src/fisheye/analysis/eye_angle_analysis.py` sets `EYE_ANGLE_LAYOUT_DEFAULT = EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2`; `tests/unit/fisheye/test_eye_angle_lineage_attrs.py` covers the default. | Pass |
| Subject-shape and refined-subject-mask writer compaction is intentionally deferred | `docs/analysis_writer_compact_layout_inventory.md` records the deferral and explains why dense masks/body-specific geometry should not be flattened yet. | Pass |
| Swim-bout readers use logical resolver | `src/fisheye/analysis/swim_bout_io.py` exposes compact/hierarchical readers; `plot_track_kinematics.py`, `track_kinematics.py`, `megabouts_classifier_inputs.py`, `stimulus_response.py`, `export_cross_recording_analytics.py`, and `interactive_track_kinematics.py` import/use that resolver surface. | Pass |
| Bout-kinematics readers use logical resolver | `src/fisheye/analysis/bout_kinematics.py::resolve_bout_kinematics_tables`; Marimo discovery/loading and cross-recording export use the resolver. The batch pipeline validator now validates logical compact levels instead of `*/per_bout_metrics` paths. | Pass |
| Stimulus-response readers use logical resolver | `src/fisheye/analysis/stimulus_response_io.py::resolve_stimulus_response_tables`; exporter, Marimo stimulus-response panels, and `plot_stimulus_response_omr.py` use the resolver for compact-v2 summary tables. | Pass |
| Eye-angle readers use logical resolver | `src/fisheye/analysis/eye_angle_io.py::load_eye_angle_run_tables`; Marimo, eye-angle visualization scripts, overlays, and bout-kinematics eye-gaze loading use the logical API. | Pass |
| Track-kinematics source readers use logical resolver where practical | `detect_bouts_multi_level.py`, `stimulus_response.py`, `plot_track_kinematics.py`, and now `plot_stimulus_response_omr.py` use `track_kinematics_io` for source track arrays. Artifact specs may still record the current physical `tracks/id_<track>` path as source provenance because track-kinematics is not compact-tabular yet. | Pass with note |
| Bout-kinematics visualization artifacts support compact-v2 | `bout_kinematics.py` writes compact source paths such as `movement_metrics`, `heading_metrics`, `eye_gaze_metrics` and records `source_filters`; `test_bout_kinematics.py::test_compute_and_save_bout_kinematics_compact_v2_writes_zarr_artifacts` covers this. | Pass |
| Stimulus-response OMR visualization artifacts support compact-v2 | `plot_stimulus_response_omr.py` writes compact source paths and filters for moving-grating OMR tables; `test_plot_stimulus_response_omr.py` covers compact source paths and filters. | Pass |
| Eye-angle dashboard artifacts support compact-dense-v2 | `visualize_eye_angles.py::_eye_angle_source_paths` records compact dense backing arrays and channel-index groups when `layout == "compact_dense_v2"`. | Pass |
| Docs state current compact defaults | Updated `docs/stimulus_response_analysis_flow.md`, `docs/eye_angle_compact_v2_design.md`, and `docs/v2_tabular_identity_migration_checklist.md`; existing `docs/analysis_writer_compact_layout_inventory.md`, `docs/bout_kinematics_compact_v2_layout.md`, `docs/stimulus_response_compact_v2_design.md`, and `docs/swim_bout_runs_v2_compact_layout.md` already reflect the accepted/default state. | Pass |
| Data-lifecycle policy remains explicit | `docs/v2_tabular_identity_migration_checklist.md` defines the recording archive as the authority, registry/export lakes as rebuildable sidecars, raw/model outputs as immutable provenance, refined surfaces as mutable authoring layers, and derived analyses/exports as rebuildable. | Pass |

## Findings

- No remaining high-priority compact analytic reader was found that assumes
  hierarchical swim-bout, bout-kinematics, stimulus-response, or eye-angle
  physical paths.
- One stale reader pattern was found and fixed: OMR bout-trajectory plotting
  now loads track arrays through `track_kinematics_io` instead of walking
  `tracks/id_<track>` directly.
- Two stale docs were found and fixed: stimulus-response analysis flow no
  longer describes compact-v2 as planned, and the eye-angle compact design no
  longer says the writer is unchanged.
- The v2 migration checklist now marks the accepted compact writer families as
  completed while keeping subject-shape, refined-subject masks, track ragged
  layout, and full multi-subject identity work as future phases.

## Remaining Intentional Direct Paths

- Visualization specs can still record physical source paths for provenance,
  such as `analysis/track_kinematics_runs/<scope>/<run>/tracks/id_<track>`.
  This is acceptable because track-kinematics itself is still hierarchical and
  the path documents the current source artifact. Reader code should continue
  loading through `track_kinematics_io`.
- Subject-shape and refined-subject-mask physical layouts remain specialized
  and are not part of this compact-default gate.

## Validation Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/palette-pycache scripts/py -m py_compile \
  src/fisheye/analysis/plot_stimulus_response_omr.py \
  src/fisheye/utils/run_movement_bout_batch_pipeline.py \
  src/fisheye/utils/export_cross_recording_analytics.py
```

```bash
PYTHONPYCACHEPREFIX=/tmp/palette-pycache scripts/py -m pytest -p no:cacheprovider \
  tests/unit/fisheye/test_plot_stimulus_response_omr.py \
  tests/unit/fisheye/test_run_movement_bout_batch_pipeline.py \
  tests/unit/fisheye/test_export_cross_recording_analytics.py -q
```

```bash
git diff --check
```

Validation result on 2026-05-11:

- `py_compile`: passed.
- Focused pytest outside the sandbox:
  `tests/unit/fisheye/test_plot_stimulus_response_omr.py`,
  `tests/unit/fisheye/test_run_movement_bout_batch_pipeline.py`, and
  `tests/unit/fisheye/test_export_cross_recording_analytics.py` passed
  (`15 passed in 9.27s`).
- `git diff --check`: passed.
