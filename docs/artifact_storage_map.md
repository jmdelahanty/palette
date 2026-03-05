# Artifact Storage Map

This doc clarifies where PNG/JSON artifacts are persisted today.

## Quick Answer

- **Training data card plots** are written as **filesystem files** (`*.png`) next to a
  `*.data_card.json`.
- **Profile/refinement visual artifacts** are written **inside zarr** under
  `visualizations/<artifact_name>`.
- Some tools export zarr-stored PNG artifacts back out to filesystem paths for viewing.

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
| Source-profile HTML thumbnail cache | `<output_html_stem>.artifacts/*.png` | no | `fisheye.utils.index_source_recording_profiles --include-artifacts` |
| Training-card HTML index | `<datasets_root>/_index/training_data_cards_index.html` | no | `fisheye.utils.index_training_data_cards` |

## Notes

- Detect/keypoint **profile runs** (`analysis/*_profile_runs`) primarily store metric summary attrs.
  Their images shown in the source-profile HTML may come from linked refined-run visualizations.
- Export/view helpers for zarr-stored artifacts:
  - detect: `fisheye.utils.export_detect_quality_overview`
  - keypoint: `fisheye.utils.export_keypoint_quality_overview`
  - eye-mask profile: `fisheye.utils.export_eye_mask_quality_overview`
- Training card plots can be disabled via `--no-plots` (`--data-card-no-plots` in pipeline wrappers).

## Quick Checks

Check whether training-card plots are external files:

```bash
find /nvme1/training/datasets -type d -name '*.data_card.plots' -maxdepth 4
```

Check whether zarr visual artifacts exist:

```bash
find /nvme1/recordings -type d -path '*/visualizations/*_png' | head
```

