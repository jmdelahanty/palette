# Analytics Storage Schema Matrix

<!-- contract-meta
version: 1
status: active
last_verified: 2026-07-26
-->

This is the canonical index for Palette's recording-local derived analytics
storage. It answers which schema and logical reader own each analysis family;
it does not replace the detailed scientific contracts or the physical path
reference in `src/fisheye/docs/zarr_structure.md`.

## System boundary

- Recording-local scientific authorities and derived runs live in the
  recording analysis Zarr. Derived runs are immutable, versioned, source-pinned,
  and independently regenerable.
- Cross-recording tables are immutable Parquet exports under the shared
  analytics root. They are query products derived from selected recording-local
  runs, not competing authorities and not archive-local sidecars.
- Consumers use the family resolver or logical reader. Physical groups,
  compatibility arrays, chunks, and shards are not consumer contracts.
- Large immutable arrays use measured chunk/shard profiles. Parallel writers
  must own complete, non-overlapping physical chunks or shards.
- Production materializers compute in node-local Zarr storage and publish a
  validated run through `palette.atomic_run_group_publisher` version 1: hidden
  same-parent sibling, verified copy, atomic rename, completion, configured
  selector or eligibility activation, and rollback on failure.

## Current family matrix

The maintained array-bearing analysis families are also enumerated by the
executable writer-backed catalog in
`fisheye.analysis_workflows.storage_contract_catalog`. Schema values in this
table were last reconciled against that catalog on 2026-08-01.

| Family | Authority role | Schema and method | Default logical/physical layout | Axis and identity | Logical reader | Publication | Compatibility |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `analysis/arena_geometry_fit_runs` | Immutable pre-review evidence for the blind early/middle/late dish-rim fit; never an operational selector | `palette.arena_geometry_fit_review_run` v1 containing `palette.arena_geometry_fit_review_record` v1; JSON evidence uses checksummed UTF-8 byte arrays and images use `palette.visualization.png_bytes.v1` | Content-derived Zarr v3 run containing fit report, review receipt, optional post-freeze acquisition reveal, montage, and exactly three temporal source panels | Exact source-video/camera/native-pixel identity, frozen fit digest, artifact digests, and human-review-required state | `fisheye.analysis_workflows.materializers.arena_geometry_fit_review` plus the generic Zarr visualization reader | shared atomic publisher; final consolidation follows payload validation; run remains selector-ineligible and does not publish a candidate, selection, or gate | External probe packages are compatibility inputs and become disposable after verified embedded publication |
| `analysis/arena_geometry_runs` | Immutable geometry candidates for later comparison and explicit selection; publication alone is not operational authority | `palette.arena_geometry_candidate_run` v1 containing `palette.arena_geometry_candidate_record` v1; first candidate kind is `acquisition_registered_dish` | Metadata-only Zarr v3 run groups; canonical-JSON digest supplies the content-derived candidate ID | Exact rig, canvas, arena, camera, source-camera continuous pixel-frame authority, recovery receipt, and Orange contract checksums | `fisheye.analysis_workflows.materializers.arena_geometry_candidates` | shared atomic publisher; completion and eligibility do not set `latest`, project `dish_mask`, or activate gating | legacy `analysis_metadata.attrs["dish_mask"]` may only be a future derived projection of an explicitly selected candidate |
| `analysis/track_kinematics_runs/<online|offline>` | Identity-resolved framewise position, motion, heading, and derivatives | `analysis.track_kinematics_runs` v1; `track_kinematics.v1`; grouped movement subcontracts remain `palette.track_movement.v2` | Per-track semantic groups; node-local output converted to 262,144-row indexed shards | `track_samples`; each track persists exact `frame_indices`, detection lineage, and track ID | `fisheye.analysis.track_kinematics_io` | shared atomic publisher | flat speed/acceleration arrays and track-local `swim_bouts` are deprecated mirrors |
| `analysis/swim_bout_runs` | Authoritative event segmentation candidates | `palette.swim_bout_runs` v8 for compact runs; algorithm contract v1 | `compact_tabular_v2`; columnar tables plus signal-major detector trace; adaptive columnar sharding | `swim_bout_rows`; frame coordinate uses `palette.swim_bout_frame_axis_reference` v1 by default | `fisheye.analysis.swim_bout_io` | shared atomic publisher | schema-7 embedded frame axes and `hierarchical_v1` remain readable |
| `analysis/bout_kinematics_runs` | Per-bout movement, heading, and optional eye-gaze measurements | `analysis.bout_kinematics_runs` v7; `bout_kinematics.v7` | `compact_tabular_v2`; shared columnar sharding | source swim-bout rows plus exact track, segmentation, heading, and eye-run lineage | `fisheye.analysis.bout_kinematics_io` | shared atomic publisher | `hierarchical_v1` remains explicit legacy/debug output |
| `analysis/eye_angle_runs` | Framewise/ROI eye orientation, gaze, and convergence | exact compact run schema v7; output schema v9; array schema v1; algorithm contract v1 | `compact_dense_v2`; semantic channel order; roughly `(4096,16)` chunks and `(131072,32)` shards; byte-planner adoption remains false | keypoint-detection rows plus explicit frame projection and body-frame support | `fisheye.analysis.eye_angle_io` | shared atomic publisher | the closed run-schema v2-v6 legacy-layout allowlist remains explicit compatibility |
| `analysis/subject_shape_runs` | Body/eye/swim-bladder geometry, body frame, centerline, spline, and tail landmarks | historical v4/method 11; recording-bundle v5/method 12 | v4 retains its explicit materialized layout; bundle-v5 candidate and supported profiles share byte-planned row-aligned inner chunks near 128 KiB inside 8 MiB indexed shards with eager semantic axes near 1 MiB | exact recording subject-mask bundle rows for v5; explicit historical refined-mask rows for v4 | `fisheye.analysis.subject_shape_io` through the shared canonical resolver | shared atomic publisher; candidate v5 remains selector-ineligible while supported v5 activates through its guarded profile and final consolidated generation | no flattening requirement; specialized body geometry remains semantic |
| `analysis/tail_kinematics_runs` | Framewise tail-angle, lateral-deflection, curvature, and validity products | `analysis.tail_kinematics_runs` v2; method version 1 | Compact run-level dense arrays; 262,144-row process-owned shards | exact subject-shape row lineage | `fisheye.analysis.tail_kinematics_io` | shared atomic publisher | legacy/tool-facing projections belong in explicit view runs |
| `analysis/stimulus_response_runs` | Per-step, per-fish, per-bout, window, trial, and stimulus-adapter metrics | `palette.stimulus_response` v2 | `compact_tabular_v2`; dense upstream traces are referenced or reconstructed rather than copied | stimulus-step/fish/bout rows with exact track, stimulus, and optional bout lineage | `fisheye.analysis.stimulus_response_io` | shared atomic publisher | `hierarchical_v1` remains explicit compatibility/debug output |
| `analysis/*/visualizations` | Review snapshots and interactive rendering contracts | artifact schemas defined by `plot_visualization_artifact_contract.md` | PNG byte arrays and lightweight specs pointing to scientific source arrays | exact source run and plot-data paths | `fisheye.shared.plot_artifacts` and application-specific resolvers | published with the owning analysis run or explicit finalized update | full HTML and decoded image matrices are not canonical artifacts |
| shared analytics exports | Cross-recording query/report products | export schema v2 plus immutable manifest | partitioned Parquet tables and manifests under the shared analytics root | recording IDs plus exact source-run and lineage columns | registry/export resolver and Polars/PyArrow projection APIs | separate serialized manifest publication after recording runs complete | `/nvme1` exports and archive-local Parquet sidecars are non-production history |

## Axis rules

An exact coordinate axis may be referenced when the downstream data preserves
the upstream row domain, order, and length. The reference must pin a concrete
run and array path plus enough shape, dtype, and lineage information to fail
closed if the authority changes. It must never resolve through `latest`.

Identity and mapping arrays are different. `instance_key` identifies an
instance; `source_row_indices` maps local rows to upstream rows; frame offsets
index sparse frame groups. Filtering, reordering, duplication, or sparsification
requires a local mapping even when metadata also names an upstream authority.

## Versioning rules

- `schema_id` and `schema_version` describe the run family's logical contract.
- An output schema versions the named scientific fields when that vocabulary
  evolves independently, as with eye angles.
- Algorithm-contract versions describe the computation and its parameters.
- Physical-layout and publisher versions describe storage and transaction
  mechanics and must not be used as scientific schema versions.
- A new physical layout may retain the logical schema only when the resolver
  produces the same declared logical surface and validation proves parity.

## Canonical supporting contracts

- Physical hierarchy: `src/fisheye/docs/zarr_structure.md`
- Common derived-run attributes and lineage: `docs/derived_analysis_run_contract.md`
- Workflow staging/publication: `docs/analysis_workflow_dag.md`
- Completion and pointer semantics: `docs/zarr_run_completion_contract.md`
- Storage lifecycle and Zarr/Parquet boundary: `docs/zarr_storage_lifecycle_policy.md`
- Chunk-safe parallel writes: `docs/dask_zarr_write_safety.md`
- Cross-recording exports: `docs/cross_recording_analytics_export_design.md`
- Visualization/reporting boundary: `docs/dataset_reporting_contract.md`

Detailed family contracts remain authoritative for scientific fields and
algorithms. Dated measurements belong under `docs/diagnostics/`; implemented or
rejected proposals belong under `docs/archive/` and must carry an archived or
superseded banner.
