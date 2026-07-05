# Recording Artifact Inventory Contract

Status: initial read-only surface.

This contract describes the per-recording artifact inventory produced by:

```bash
scripts/py -m fisheye.utils.recording_artifact_inventory <archive.zarr> --json
```

The inventory is a discovery surface. It does not choose scientific authority,
edit the Zarr store, refresh registry rows, or consolidate metadata.

## Purpose

Palette has several valid per-recording artifact surfaces:

- root stage run families such as `detect_runs`, `refined_detect_runs`,
  `crop_runs`, `keypoints_runs`, and subject-mask families
- derived analysis families under `analysis/*_runs`
- nested reports such as `detect_runs/<run>/quality_reports/<report>`
- run-local visualization artifacts under `<run>/visualizations/*`
- recording-level acquisition sidecar mirrors under
  `analysis/acquisition_video_streams`
- registry projections such as `recording_step_status` and optional artifact
  views

The inventory gives users, agents, and future UIs one read-only summary of
those surfaces for a single recording Zarr.

## Output Schema

The JSON payload has:

```text
schema_id = "palette.recording_artifact_inventory.v1"
```

Top-level fields:

- `zarr_path`: path passed to the CLI when available
- `zarr_use`: inferred via `shared.zarr_helpers.infer_zarr_use`
- `root_attrs`: compact identity/role attrs copied from the root
- `root_run_families`: run parents outside `analysis/`
- `analysis_run_families`: run parents under `analysis/*_runs`
- `nested_report_families`: nested run-like report parents discovered below
  other runs
- `run_family_count`
- `run_count`
- `visualization_artifact_count`
- `acquisition_video_streams`
- `registry_projection_names`

Each run family entry records:

- `family_path`: logical family path, for example
  `analysis/track_kinematics_runs`
- `run_parent_path`: concrete parent that contains runs, for example
  `analysis/track_kinematics_runs/offline`
- `family_kind`: `root_stage`, `analysis`, `analysis_scoped`, or
  `nested_quality_reports`
- `scope`: optional scope such as `offline`
- `parent`: compact pointer/completion summary
- `runs`: one entry per child run

Each run entry records:

- `name`
- `path`
- `complete`
- `completion_status`
- `has_completion_contract`
- compact lineage/schema attrs
- child array/group counts
- `visualizations`

Visualization entries are discovered from the run attrs `visualizations`
manifest first, then from direct children of `visualizations/` that are not in
the manifest.

## Authority Rules

The inventory reports pointer state but does not make an authority decision.

- `resolved_latest_complete` means the parent's latest-complete compatible
  child according to the run-completion compatibility rules.
- `resolved_authoritative` means the explicit `authoritative_run` when present
  and complete; otherwise it falls back to latest-complete for inventory
  visibility.
- Consumers that need a scientific source of truth should still use explicit
  `RunResolution` modes.

## Acquisition Video Streams

`analysis/acquisition_video_streams` is summarized separately from `crop_runs`.
The inventory reports stream keys, availability status, selected geometry/media
attrs, and whether common file references exist according to the mirrored
attrs.

This surface means acquisition media exists. It does not mean those videos were
used as model input or that their crop coordinates are Palette `crop_runs`.

## Registry Projection Names

`registry_projection_names` lists likely registry tables/views related to the
inventory. It is descriptive only. The inventory CLI does not open SQLite and
does not assert that registry rows are current.

Current core projections:

- `datasets`
- `recordings`
- `recording_step_status`

Optional projections are listed when the Zarr contains matching surfaces, such
as `acquisition_video_streams` or run-local visualization artifacts.

## Known Scope

The initial implementation intentionally focuses on individual recording Zarrs.
It does not inventory:

- cross-recording exports
- external training data-card file trees
- cluster transfer packages
- live web labeling task stores
- registry-only rows not reflected in the Zarr

Those can link to this inventory later, but should not be collapsed into the
recording Zarr authority model.

## Intended Uses

Use this inventory to:

- inspect a recording before deciding which artifact writer to refactor
- compare duplicate per-recording artifact actors
- feed future registry/UI artifact browsers
- generate read-only audit reports for missing visualizations, stale
  completion markers, or inconsistent run-family naming

Do not use this inventory as:

- the source of truth for authoritative run selection
- a replacement for completion markers
- a registry repair tool
- proof that external filesystem artifacts exist

