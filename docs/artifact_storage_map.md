# Artifact Storage Map

This doc clarifies where PNG/JSON artifacts are persisted today.

## Quick Answer

- **Training data card plots** are written as **filesystem files** (`*.png`) next to a
  `*.data_card.json`.
- **Profile/refinement visual artifacts** are written **inside zarr** under
  `visualizations/<artifact_name>`.
- Some tools export zarr-stored PNG artifacts back out to filesystem paths for viewing.
- Use `palette artifacts <recording>` for a read-only
  per-recording index of run families, run-local visualizations, nested
  reports, and acquisition stream mirrors.
- Major analysis run types should expose a run-local PNG summary writer; static
  plot snapshots and future interactive plot specs should follow
  `docs/plot_visualization_artifact_contract.md`.
- Cross-recording metric exports should be regenerated columnar views, not
  archive authorities. See
  [cross_recording_analytics_export_design.md](cross_recording_analytics_export_design.md).

## Storage Roots

The storage and registry state in this section is a snapshot reported by the
2026-09-03 review, not a fresh inventory or authorization to delete or migrate
data. Re-verify live references and applicable storage contracts before acting;
see [the measured review](diagnostics/store_measurements_selectors_and_training_membership_2026-09-03.md)
and [its second opinion](diagnostics/review_wave_second_opinion_2026-09-04.md).

All durable Palette data lives under `/groups/johnson/johnsonlab/jeremy/` (backed up and managed by
HPC/IT). `/nvme1` is workstation scratch: its per-recording archives were deleted on purpose in
2026-08/09, its merged training zarrs were copied to `/groups` on 2026-09-03, and no registry row
references it any more. Do not point new tooling at `/nvme1`.

| Root | Contents | Registry table(s) that point here |
|---|---|---|
| `recordings/<recording_id>/zarr/<recording_id>_{analysis,training}.zarr` | per-recording archives (run families, clips, sidecars) | `datasets` (`artifact_kind='source_recording'`) |
| `training/datasets/<set_id>/…/<set_id>_merged.zarr` | merged training zarrs, one per training set, plus `<set_id>.manifest.json`, config yaml, data cards; `_index/` holds the card index and copy-verification digests | `datasets` (`artifact_kind='derived_training_merge'`), `training_runs.{config,manifest}_path` |
| `models/<task>/<set_id>/<run_id>/` | trained model runs (layout below) | `training_runs`, `training_models`, `onnx_models`, `tensorrt_models` |
| `registries/palette_registry.sqlite` | the canonical registry; `registries/backups/` holds validated pre-write backups | — |
| `operations/` | operator records (run plans, snapshot manifests, canaries) | — |

### Models tree

`<task>` is one of `detect`, `pose`, `eye_masks`, `subject_masks`; `sam3/` holds base weights.
`<set_id>` matches `training_sets.set_id` and the merged-zarr directory name under
`training/datasets/`. `<run_id>` matches `training_runs.run_id`.

```text
models/
  detect/<set_id>/<run_id>/            # ultralytics YOLO run
    weights/best.pt                    # training_models.model_path
    weights/best.pt.content_v1.json    # content-hash sidecar (fingerprint_artifact)
    weights/last.pt
    exports/onnx/<run_id>.onnx         # onnx_models.path
    exports/tensorrt/<run_id>_fp16.engine   # tensorrt_models.path (per GPU class)
    inputs/                            # copied set manifest + effective training yaml + train_invocation.json
    args.yaml, results.csv, *_training_report.yaml, curve PNGs, batch previews
  pose/<set_id>/<run_id>/              # same layout; adds Pose*_curve.png
  subject_masks/<set_id>/<run_id>/     # UNet run
    best_model.pt (+ .content_v1.json), last_model.pt
    dataset_metadata.json, training_history.json, training_summary.json, validation_previews/
  eye_masks/<set_id>/<run_id>/         # legacy-compat UNet run: best_model.pt, validation_previews/;
                                       # the set dir also holds tensorboard/ and two empty sibling run dirs
  sam3/sam3.pt (+ .content_v1.json)    # base SAM weights, not a registry run
```

Rules that hold today and should keep holding:

- Every directory under `models/<task>/` is referenced by a `training_runs` row, and every
  registry model, ONNX, and TensorRT path resolves into `models/`. Orphan run directories and
  set-less runs were removed on 2026-09-03; keep it that way by registering before training
  (`training/training_run_shared.record_registry_training_run`) and deleting registry row and
  directory together.
- `metrics_path` points at the run's `results.csv` (YOLO) or `training_summary.json` (UNet).
- There is exactly one models tree. The `palette_models/` snapshot was deleted on 2026-09-03; its
  manifest is kept under `operations/palette_models_snapshot_20260514/`.
- `datasets_root` in the training-card tools means `training/datasets/`.

## Storage Matrix

| Artifact | Canonical location | In zarr | Producer |
|---|---|---:|---|
| Detect training data card JSON | `<dataset_dir>/<set_id>.data_card.json` | no | `fisheye.utils.aggregate_detection_training_data_card` |
| Detect training data card plots | `<dataset_dir>/<set_id>.data_card.plots/*.png` | no | `fisheye.utils.plot_detection_training_data_card` (or aggregate with plots enabled) |
| Keypoint training data card JSON | `<dataset_dir>/<set_id>.data_card.json` | no | `fisheye.utils.aggregate_keypoint_training_data_card` |
| Keypoint training data card plots | `<dataset_dir>/<set_id>.data_card.plots/*.png` | no | `fisheye.utils.plot_keypoint_training_data_card` (or aggregate with plots enabled) |
| Eye-mask training data card JSON | `<dataset_dir>/<set_id>.data_card.json` | no | `fisheye.utils.aggregate_eye_mask_training_data_card` |
| Eye-mask training data card plots | `<dataset_dir>/<set_id>.data_card.plots/*.png` | no | `fisheye.utils.plot_eye_mask_training_data_card` (or aggregate with plots enabled) |
| Detection profile summary | `analysis/detection_profile_runs/<run>/attrs["profile_summary"]` | yes | `fisheye.utils.detection_profile` |
| Detection occupancy overview PNG | `analysis/detection_occupancy_runs/<run>/visualizations/detection_occupancy_overview_png` | yes | `fisheye.analysis.detection_occupancy_runs` |
| Realtime/offline detection comparison PNG | `analysis/detection_comparison_runs/<run>/visualizations/realtime_offline_detection_comparison_png` | yes | `fisheye.diagnostics.compare_realtime_offline_detections` |
| Chaser distance time-series PNG | `analysis/chaser_distance_runs/<run>/visualizations/chaser_distance_timeseries_png` | yes | `fisheye.analysis.chaser_distance_runs` |
| Chaser distance epoch-median PNG | `analysis/chaser_distance_runs/<run>/visualizations/chaser_distance_epoch_median_png` | yes | `fisheye.analysis.chaser_distance_runs` |
| Chaser distance epoch-distribution PNG | `analysis/chaser_distance_runs/<run>/visualizations/chaser_distance_epoch_distribution_png` | yes | `fisheye.analysis.chaser_distance_runs` |
| Keypoint profile summary | `analysis/keypoint_profile_runs/<run>/attrs["profile_summary"]` | yes | `fisheye.utils.keypoint_profile` |
| Eye-mask profile summary | `analysis/eye_mask_profile_runs/<run>/attrs["profile_summary"]` | yes | `fisheye.utils.eye_mask_profile` |
| Eye-mask profile overview PNG | `analysis/eye_mask_profile_runs/<run>/visualizations/eye_mask_profile_overview_png` | yes | `fisheye.utils.finalize_eye_mask_profile_artifacts` |
| Refined detect quality PNGs | `refined_detect_runs/<run>/visualizations/{detect_quality_overview_png,refinement_pipeline_overview_png}` | yes | `fisheye.utils.finalize_refinement_artifacts` |
| Refined keypoint quality PNGs | `refined_keypoints_runs/<run>/visualizations/{keypoint_quality_overview_png,keypoint_refinement_pipeline_overview_png}` | yes | `fisheye.utils.finalize_keypoint_refinement_artifacts` |
| Analysis plot PNG snapshots | `analysis/<stage>_runs/<run>/visualizations/<artifact>_png` | yes | stage-specific plot/finalize helpers using `fisheye.shared.plot_artifacts` |
| Interactive plot specs | `analysis/<stage>_runs/<run>/visualizations/<artifact>/spec_json` | yes | stage-specific plot/finalize helpers using `fisheye.shared.plot_artifacts` |
| Source-profile HTML thumbnail cache | `<output_html_stem>.artifacts/*.png` | no | `fisheye.utils.index_source_recording_profiles --include-artifacts` |
| Training-card HTML index | `<datasets_root>/_index/training_data_cards_index.html` | no | `fisheye.utils.index_training_data_cards` |

## Notes

- A **profile summary** and a **profile PNG artifact** are different things.
  A run can have `attrs["profile_summary"]` and still have no `visualizations/*_png`
  artifacts at all.
- Detect/keypoint **profile runs** (`analysis/*_profile_runs`) primarily store metric summary attrs.
  Their images shown in the source-profile HTML may come from linked refined-run visualizations.
- Eye-mask profile runs are different: their finalized overview PNG is written directly into
  `analysis/eye_mask_profile_runs/<run>/visualizations/eye_mask_profile_overview_png`.
- The source-profile HTML indexer (`fisheye.utils.index_source_recording_profiles --include-artifacts`)
  first checks the profile run itself, then for detect/keypoint profiles also follows
  `profile_summary.source.refined_run` and related source paths to look for PNG arrays in the
  linked refined run's `visualizations/` group.
- Finalized PNG artifacts are not written automatically just because a profile/refined run exists.
  The finalize helpers are gated and dry-run by default:
  - `fisheye.utils.finalize_refinement_artifacts`
  - `fisheye.utils.finalize_keypoint_refinement_artifacts`
  - `fisheye.utils.finalize_eye_mask_profile_artifacts`
- For detect/keypoint finalize flows, the run generally must satisfy the expected review state
  (default `approved`) and the command must be run with `--apply` before the PNG artifacts are
  persisted into zarr.
- For eye-mask profile finalize flow, the same `--apply` rule applies, and review-state /
  intended-use filters may also exclude a run from artifact generation.
- Export/view helpers for zarr-stored artifacts:
  - per-recording artifact inventory:
    `palette artifacts <recording>` (also `fisheye.api`: `Recording.artifact_inventory()`)
  - generic visualization artifact viewer: `fisheye.utils.view_zarr_visualization`
  - combined detection occupancy + chaser distance viewer:
    `fisheye.utils.view_detection_chaser_overview`
  - detect: `fisheye.utils.export_detect_quality_overview`
  - keypoint: `fisheye.utils.export_keypoint_quality_overview`
  - eye-mask profile: `fisheye.utils.export_eye_mask_quality_overview`
- Training card plots can be disabled via `--no-plots` (`--data-card-no-plots` in pipeline wrappers).
- Rendered PNGs are review snapshots. Interactive plots should be represented by
  lightweight specs pointing back to source arrays, not by full HTML documents
  or decoded RGB image arrays in zarr.
- Some analysis visualizations have explicit plot-data arrays beside the PNG.
  For example, `analysis/chaser_distance_runs/<run>/epoch_distributions/*`
  stores reusable distance histogram bins and densities; consumers should read
  those arrays when they need an interactive distribution plot.
- Persisted visualization artifacts are expected for reviewable analysis runs,
  but generation may remain explicit via `--write-zarr-artifacts` or an
  equivalent finalize/apply command so heavy debug plots are not produced
  accidentally.

## Why A Profile May Have No Visible PNG Artifacts

Common cases:

- The profile run exists, but only the summary attrs were written; no finalize step has been run.
- A finalize command was run without `--apply`, so it stayed in dry-run mode.
- The run failed review-state gating, so the finalize command skipped it.
- For detect/keypoint profiles, the profile run exists but the linked refined run has no
  `visualizations/*_png` arrays to extract.
- The HTML index was generated with `--include-artifacts`, but there were no underlying PNG arrays
  in any resolved target run.

Practical implication:

- Eye-mask profiles can legitimately have profile-local PNGs.
- Detect/keypoint profiles often show thumbnails only when the corresponding refined run has already
  had its PNG artifacts finalized.

## Quick Checks

Check whether training-card plots are external files:

```bash
find /groups/johnson/johnsonlab/jeremy/training/datasets -type d -name '*.data_card.plots' -maxdepth 4
```

Check whether zarr visual artifacts exist:

```bash
find /groups/johnson/johnsonlab/jeremy/recordings -type d -path '*/visualizations/*_png' | head
```

Inventory one recording Zarr:

```bash
palette artifacts <archive.zarr>
palette artifacts <archive.zarr> --json
```
