# Analysis and Post-Detection Workflow Status

Date anchored: 2026-03-06

Last reviewed: 2026-07-02

Purpose: summarize the current state of Palette's analysis and post-detection
workflows, identify which repository docs are current versus aspirational or
stale, and recommend whether to overhaul `track_kinematics` or build a new
unified downstream analysis layer.

## Executive Summary

The repository already has substantial post-detection analysis code, but it is
not yet a single unified workflow. The current state is:

- The ingest + detect/refine path is clearly contracted and operator-facing.
- Downstream analysis exists as a set of real but mostly standalone tools.
- `track_kinematics` is no longer the main architectural gap; it already
  supports separate `online` and `offline` outputs under
  `analysis/track_kinematics_runs/`.
- A canonical downstream consumer now exists in `stimulus_response`; the main
  remaining work is hardening its movement-distance semantics, bout inputs, and
  stimulus-specific adapters.

Recommendation:

- Do not start with a wholesale overhaul of `track_kinematics`.
- Treat `track_kinematics` as an upstream producer that is already useful and
  mostly structurally aligned.
- Add a future skeleton-derived `pose_kinematics` layer for tail / fin / richer
  body geometry rather than expanding `track_kinematics` into a catch-all
  analysis stage.
- Continue hardening the unified `stimulus_response` analysis run on top of
  track kinematics, optional pose kinematics, stimulus, bouts, and optional eye
  angles.

## What Exists Today

### 1. Canonical ingest / detect / refine pipeline

The only clearly contracted operator workflow today is the recording analysis
pipeline:

- [recording_analysis_pipeline_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/recording_analysis_pipeline_contract.md)
- [run_recording_analysis_pipeline.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/utils/run_recording_analysis_pipeline.py)

This workflow covers:

1. analysis-archive creation / import
2. detect
3. detect-quality
4. refine-detect
5. optional keypoints / refine-keypoints
6. optional registry scan

This contract is explicit about stage order and failure semantics, but it stops
before a unified downstream analysis stage.

### 2. Analysis archive creation and stimulus ingest are real

The analysis-archive split is implemented and documented:

- [analysis_zarr_creation_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/analysis_zarr_creation_contract.md)
- [analysis_zarr_creation_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/analysis_zarr_creation_todo.md)
- [create_analysis_zarr.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/analysis/create_analysis_zarr.py)
- [import_stimulus_to_zarr.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/analysis/import_stimulus_to_zarr.py)

This gives the repo a real `analysis/stimulus_runs/<run>/` substrate, including:

- frame metadata
- events
- chaser states
- calibration data
- protocol JSON import

### 3. There is already real post-detection analysis code

Implemented downstream modules include:

- [track_kinematics.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/analysis/track_kinematics.py)
- [chaser_distance_runs.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/analysis/chaser_distance_runs.py)
- [chaser_egocentric_bearing.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/analysis/chaser_egocentric_bearing.py)
- [swim_bout_statistics.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/analysis/swim_bout_statistics.py)
- [detect_bouts_multi_level.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/analysis/detect_bouts_multi_level.py)
- [stimulus_response.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/analysis/stimulus_response.py)
- [eye_angle_analysis.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/analysis/eye_angle_analysis.py)
- [chaser_phase_analysis.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/analysis/chaser_phase_analysis.py)
- [plot_track_kinematics.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/analysis/plot_track_kinematics.py)
- [chaser_metrics_loader.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/analysis/chaser_metrics_loader.py)

These are not placeholders. They represent an actual analysis stack with
persisted outputs under `analysis/`.

## What Is Already Unified

### Movement online/offline split

The repo already has a partially unified movement architecture:

- [track_kinematics.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/analysis/track_kinematics.py)
  writes to:
  - `analysis/track_kinematics_runs/online/<run>/`
  - `analysis/track_kinematics_runs/offline/<run>/`
- [plot_track_kinematics.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/analysis/plot_track_kinematics.py)
  already resolves both online and offline runs.
- [chaser_metrics_loader.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/analysis/chaser_metrics_loader.py)
  provides a unified loader for online and offline chaser metrics.

This means the repo is past the point where "split online/offline movement
runs" is a hypothetical future state. That split is implemented.

### Shared storage pattern under `analysis/`

The codebase already follows a fairly consistent storage convention:

- `analysis/stimulus_runs`
- `analysis/track_kinematics_runs`
- `analysis/chaser_fish_metrics` *(legacy readable layout only)*
- `analysis/chaser_distance_runs`
- `analysis/swim_bout_runs`
- `analysis/eye_angle_runs`
- `analysis/*_profile_runs`

So the repo does have an emerging analysis platform. The main missing piece is
not storage naming but a higher-level consumer contract.

There is also a clear future extension point for richer skeletons:

- [`pose_kinematics_run_design.md`](/home/delahantyj@hhmi.org/gitrepos/palette/docs/pose_kinematics_run_design.md)

That design keeps `track_kinematics` focused on generic whole-animal motion and
places future tail / fin / segment geometry into a sibling downstream layer.

## What Is Not Unified Yet

### No canonical downstream analysis orchestrator

There is no single, operator-facing contract that says:

1. movement is ready
2. bouts are ready
3. eye angles are ready
4. stimulus-aware response analysis is ready

The current operator contract ends at detect/refine and optional keypoint
stages. Downstream analyzers are separate CLIs and scripts.

### Most downstream analyzers are stage-specific

The current analysis modules are still specialized:

- `track_kinematics` focuses on track-level movement summaries
- `chaser_distance_runs` and `chaser_egocentric_bearing` are chaser/fish geometry specific
- `swim_bout_statistics` focuses on bout segmentation and per-trial summaries
- `eye_angle_analysis` is subject-shape/refined-subject/refined-eye geometry
  plus keypoint-heading derived and not stimulus-aware
- `chaser_phase_analysis` is a specialized consumer/visualizer

This is useful, but it is not yet a unified framework for "post-detection
analysis by stimulus type."

### Multi-stimulus support has an implemented substrate

The general unification design in the repo is:

- [stimulus_response_run_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/stimulus_response_run_design.md)

That document defines the implemented
`analysis/stimulus_response_runs/<run>/` layout, which:

- consumes movement as the identity-resolved source of truth
- computes base per-step metrics for all stimulus types
- adds stimulus-specific subgroups only where needed
- can support moving gratings, concentric gratings, looming dots, chaser, and
  future stimulus families without changing the top-level run contract

Implementation now exists under `src/fisheye/analysis/stimulus_response.py`,
with helper modules for storage and stimulus-specific OMR/grating behavior. This
section should be read as current-status guidance plus remaining hardening work,
not as a pre-implementation design note.

## Documentation Status

### Current / useful docs

These reflect real architecture or active contracts:

- [recording_analysis_pipeline_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/recording_analysis_pipeline_contract.md)
- [analysis_zarr_creation_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/analysis_zarr_creation_contract.md)
- [analysis_zarr_creation_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/analysis_zarr_creation_todo.md)
- [stimulus_response_run_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/stimulus_response_run_design.md)
- [grating_analysis_acquisition_questions.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/grating_analysis_acquisition_questions.md)
- [protocol_parameter_registry_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/protocol_parameter_registry_todo.md)
- [experiment_types_reference.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/experiment_types_reference.md)

### Partially stale docs

#### `movement_online_offline_plan`

- [movement_online_offline_plan.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/notes/movement_online_offline_plan.md)

This doc describes the online/offline movement split as future work. That is
now out of date at the code level because:

- `track_kinematics.py` already writes `analysis/track_kinematics_runs/online` and
  `.../offline`
- `plot_track_kinematics.py` already consumes that structure

The doc is still useful as historical rationale, but not as an accurate plan.

#### Protocol import bug docs

Several docs still describe the stimulus importer as if it only looked for the
wrong H5 key:

- [protocol_parameter_registry_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/protocol_parameter_registry_todo.md)
- [experiment_types_reference.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/experiment_types_reference.md)
- [provenance_backfill_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/provenance_backfill_todo.md)

Code audit result:

- [import_stimulus_to_zarr.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/analysis/import_stimulus_to_zarr.py)
  now checks both:
  - `/protocol_snapshot/protocol_definition_json`
  - `/protocol_snapshot/protocol_json`
  and writes `run_group.attrs["protocol_json"]`.

Important distinction:

- the code path is fixed
- existing archives / registry rows may still lag if they were never backfilled

So those docs may still be correct about live data coverage, but they are stale
if read as statements about current importer behavior.

### Superseded narrow design

- [base_analysis_moving_grating_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/archive/base_analysis_moving_grating_design.md)

This is explicitly narrower and is effectively superseded by
`stimulus_response_run_design.md`.

## Multi-Stimulus Readiness

### What is present

The repository has the ingredients for multi-stimulus analysis:

- stimulus taxonomy and protocol reference:
  [experiment_types_reference.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/experiment_types_reference.md)
- a protocol registry design:
  [protocol_parameter_registry_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/protocol_parameter_registry_todo.md)
- a general stimulus-response storage design:
  [stimulus_response_run_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/stimulus_response_run_design.md)
- a stimulus ingest path that writes `analysis/stimulus_runs`
- an implemented `stimulus_response` module that consumes movement, stimulus,
  optional bout runs, and stimulus-specific adapters

### What is missing

The following gaps still block real multi-stimulus analysis:

1. Stimulus-response distance summaries still need to preserve the gap-aware
   distance semantics from `track_kinematics`.
2. Bout inputs need a single canonical producer/consumer split, with
   `detect_bouts_multi_level` preferred for per-track bout segmentation.
3. No canonical loader that extracts protocol steps and step parameters once
   for all downstream consumers.
4. No registry-backed protocol parameter tables yet.
5. Reactive stimulus parameter logging is incomplete for future dynamic
   analyses.
6. Grating orientation / projector-camera transform semantics are not settled
   enough for confident grating-response production analysis.

### Grating-specific blockers

[grating_analysis_acquisition_questions.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/grating_analysis_acquisition_questions.md)
shows that moving-grating analysis still has unresolved scientific/geometry
questions:

- projector vs camera rotation / flip
- meaning of `orientation_degrees`
- whether orientation is drift direction or bar orientation
- reactive module behavior and logging
- whether frame-level grating parameters are available

That means grating is architecturally attractive for the unified design, but it
is not the lowest-risk first implementation unless those questions are closed.

## Recommendation

### Short version

Do not start by overhauling `track_kinematics`.

Instead:

1. treat `track_kinematics` as an upstream producer
2. stabilize shared analysis-input loaders and protocol/stimulus metadata
3. implement `stimulus_response` as the new unified downstream consumer

### Why not overhaul `track_kinematics` first

`track_kinematics` already:

- has real users and outputs
- already split online/offline storage
- already carries provenance-rich run attrs
- already has at least one consumer path via `plot_track_kinematics`

Overhauling it now would likely mix three separate concerns:

- storage cleanup
- input resolution cleanup
- new stimulus-aware analysis semantics

That is high churn for limited architectural payoff.

### Better target: harden the unified consumer layer

The stronger architectural move is:

- keep `track_kinematics` producing identity-resolved tracks
- keep `eye_angle_analysis` as a specialized producer
- make `detect_bouts_multi_level` the canonical per-track bout segmentation
  producer and keep `swim_bout_statistics` as a summary/aggregation layer
- harden the `stimulus_response` layer that consumes:
  - movement
  - stimulus
  - bouts
  - optional eye angles

This matches the design direction already captured in
[stimulus_response_run_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/stimulus_response_run_design.md).

### Lowest-risk first implementation

If the goal is to start shipping a unified downstream analysis soon:

- use the `stimulus_response` architecture
- start with a stimulus family whose inputs are already trustworthy

Lowest-risk options:

- CHASER-like workflows, because the repo already has:
  - stimulus import
  - chaser/fish metrics
  - phase analysis
  - chaser-specific diagnostics
- non-reactive grating workflows only after the geometry/orientation questions in
  `grating_analysis_acquisition_questions.md` are resolved

### Suggested implementation order

1. Refresh status docs that are now partially stale.
   - especially `movement_online_offline_plan.md`
2. Add a small shared analysis-input layer:
   - stimulus run resolution
   - track kinematics run resolution
   - protocol step extraction
   - protocol parameter normalization
3. Decide the first production stimulus-response target:
   - CHASER first for lowest implementation risk
   - MOVING_GRATING first only if acquisition semantics are closed
4. Fix `stimulus_response` distance semantics so it consumes gap-aware
   displacement/cumulative-distance data from `track_kinematics`.
5. Expand with stimulus-specific adapters rather than rewriting the entire
   track kinematics stack.

## Recommended Doc Follow-Ups

1. Update [movement_online_offline_plan.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/notes/movement_online_offline_plan.md)
   to reflect that the split is implemented.
2. Re-audit [experiment_types_reference.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/experiment_types_reference.md)
   and [provenance_backfill_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/provenance_backfill_todo.md)
   so they distinguish code status from live-data backfill status.
3. Decide whether [stimulus_response_run_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/stimulus_response_run_design.md)
   is now the canonical downstream-analysis target. If yes, say so explicitly in
   the contract and related TODO docs.

## Bottom Line

The repository is no longer missing "analysis workflows" in general. It already
has a real analysis substrate and an implemented stimulus-aware consumer. The
missing piece is now hardening that consumer so movement, bout, and
stimulus-specific metrics all consume the same canonical source artifacts.

The best next architectural move is to fix those consumer contracts, not to
rebuild `track_kinematics` from scratch.
