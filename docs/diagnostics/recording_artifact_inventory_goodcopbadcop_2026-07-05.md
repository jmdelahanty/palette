# Recording Artifact Inventory GoodCopBadCop Smoke

Status: read-only smoke, 2026-07-05.

This note records the first batch check of the per-recording artifact inventory
CLI against 12 GoodCopBadCop analysis Zarrs under:

```text
/groups/johnson/johnsonlab/jeremy/recordings/2026-06-14T*_GoodCopBadCop/zarr/*_analysis.zarr
```

Command shape:

```bash
scripts/py -m fisheye.utils.recording_artifact_inventory <analysis.zarr> --json
```

The transient combined JSON summary was written to:

```text
/tmp/palette_artifact_inventory_goodcopbadcop_summary.json
```

## Coverage

- Zarrs inspected: 12
- Acquisition stream mirrors present on all inspected Zarrs: `crop`, `full`
- Nested detect quality report families present on all inspected Zarrs.
- Visualization artifacts present on all inspected Zarrs: 8 per recording.

Root run-family presence:

```text
arena_assignment_runs: 12
crop_runs: 12
detect_runs: 12
keypoints_runs: 12
refined_detect_runs: 12
refined_keypoints_runs: 12
refined_subject_masks_runs: 1
subject_mask_runs: 1
tracking_runs: 12
```

Analysis run-family presence:

```text
analysis/chaser_distance_runs: 12
analysis/detection_occupancy_runs: 12
analysis/stimulus_epoch_runs: 12
analysis/stimulus_runs: 12
analysis/swim_bout_runs: 12
analysis/track_kinematics_runs/offline: 12
```

Visualization artifact presence:

```text
chaser_distance_epoch_distribution_png: 12
chaser_distance_epoch_median_png: 12
chaser_distance_timeseries_png: 12
cra_primary_endpoint_interactive: 12
detection_occupancy_overview_png: 12
goodcopbadcop_chaser_dashboard_interactive: 12
track_kinematics_summary_track_0_interactive: 12
track_kinematics_summary_track_0_png: 12
```

## Observations

The inventory surface successfully separates per-recording static artifacts from
run families and nested reports. The GoodCopBadCop analysis Zarrs have consistent
analysis visualization coverage, and the acquisition video stream mirrors are
present uniformly.

The notable asymmetry is subject-mask coverage: only the first inspected
recording has `subject_mask_runs` and `refined_subject_masks_runs`. This is
expected for the current production state, but the inventory makes that gap
visible without needing bespoke family-specific checks.

Run counts differ by recording. Most inspected recordings have 15 total runs in
14 run families; arena 1 has 30 runs in 16 families because it includes
subject-mask outputs, and one arena 3 recording has 17 total runs. This is a
useful signal for duplicate or partial actor audits, but the inventory does not
decide which runs are authoritative.

## Implication

This CLI is a good narrow waist for future artifact cleanup work:

- duplicate actor audits can start from `run_family_count`, `run_count`, and
  per-family `runs`
- visualization simplification can start from run-local `visualizations`
- registry/UI artifact browsers can use this read-only summary without learning
  every run-family-specific layout first

The next simplification step should compare the producers of the eight repeated
GoodCopBadCop visualization artifacts and decide which are canonical static
artifacts, which are interactive review aids, and which should be regenerated
through a shared visualization writer.
