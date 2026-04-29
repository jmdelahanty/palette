# Artifact Storage Map

This doc clarifies where PNG/JSON artifacts are persisted today.

## Quick Answer

- **Training data card plots** are written as **filesystem files** (`*.png`) next to a
  `*.data_card.json`.
- **Profile/refinement visual artifacts** are written **inside zarr** under
  `visualizations/<artifact_name>`.
- Some tools export zarr-stored PNG artifacts back out to filesystem paths for viewing.
- Major analysis run types should expose a run-local PNG summary writer; static
  plot snapshots and future interactive plot specs should follow
  `docs/plot_visualization_artifact_contract.md`.
- Cross-recording metric exports should be regenerated columnar views, not
  archive authorities. See
  [cross_recording_analytics_export_design.md](cross_recording_analytics_export_design.md).

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
  - generic visualization artifact viewer: `fisheye.utils.view_zarr_visualization`
  - detect: `fisheye.utils.export_detect_quality_overview`
  - keypoint: `fisheye.utils.export_keypoint_quality_overview`
  - eye-mask profile: `fisheye.utils.export_eye_mask_quality_overview`
- Training card plots can be disabled via `--no-plots` (`--data-card-no-plots` in pipeline wrappers).
- Rendered PNGs are review snapshots. Interactive plots should be represented by
  lightweight specs pointing back to source arrays, not by full HTML documents
  or decoded RGB image arrays in zarr.
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
find /nvme1/training/datasets -type d -name '*.data_card.plots' -maxdepth 4
```

Check whether zarr visual artifacts exist:

```bash
find /nvme1/recordings -type d -path '*/visualizations/*_png' | head
```
