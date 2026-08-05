# Chaser analytics meeting pack

These plots are representative outputs from the completed GoodCopBadCop canary at:

`/nvme1/recordings/chunking_canary_2026-06-24_heartrate/zarr/2026-06-14T21-12-08Z_arena_4_GoodCopBadCop_analysis.zarr`

They demonstrate the current chaser-analysis surface and are not Batman results.
The source Zarr was read only; the PNGs below are exported copies of its
analysis-owned visualization artifacts.

The Batman source-state sanity check is the one exception: it is a read-only
plot of the raw `tracking_data/chaser_states` H5 surface from one Batman
recording. It confirms the stimulus-side trajectory/state payload but is not a
fish-response analysis.

## Plot index

- [Contact sheet](./00_contact_sheet.png)
- [Distance time series](./01_distance_timeseries.png)
- [Distance by epoch](./02_distance_epoch_median.png)
- [Distance distributions](./03_distance_epoch_distribution.png)
- [Primary endpoint / trajectory](./04_cra_primary_endpoint.png)
- [Near-field summary](./05_cra_near_field_summary.png)
- [Near-field distance CDF](./06_cra_near_field_cdf.png)
- [Near-field radial density](./07_cra_near_field_radial.png)
- [Egocentric bearing](./08_egocentric_bearing.png)
- [Egocentric bearing point cloud](./09_egocentric_point_cloud.png)
- [Bout-response kinematics](./10_bout_response_kinematics.png)
- [Bout-response turn bias](./11_bout_response_turn_bias.png)
- [Radial occupancy density](./12_radial_occupancy_density.png)
- [Radial selection index](./13_radial_selection_index.png)
- [Escape gain](./14_response_escape_gain.png)
- [Freeze curve](./15_response_freeze_curve.png)
- [Batman source-state sanity check](./16_batman_source_chaser_overview.png)

## Current analysis surface

The dependency graph is:

```text
stimulus epochs
        |
        v
chaser distance ----> quadrant occupancy ----> near-field occupancy
        |\
        | +---------> epoch behavior summary
        | +---------> egocentric bearing ----> gaze tracking
        |                    |
        |                    +---------------> bout response ----> escape events
        |\
        | +---------> radial occupancy
        | +---------> response regimes
        +-----------> escape/freeze summary
```

The basic `chaser_behavior_v1` profile selects the distance, quadrant,
near-field, epoch-summary, egocentric, gaze, and escape/freeze modules. The
expanded `chaser_behavior_full_v2` profile additionally selects detection
occupancy, bout response, escape events, radial occupancy, and response
regimes.

| Analysis | Main question | Batman inputs |
| --- | --- | --- |
| Distance | How close is the fish to each chaser over time and by epoch? | Refined detection, canonical chaser states, arena geometry |
| Quadrant occupancy | Does the fish occupy the chaser's side of the arena? | Chaser distance plus arena bounds and configured chaser roles |
| Near-field occupancy | How often does the fish enter/stay near the chaser? | Quadrant occupancy plus distance/arena geometry |
| Epoch behavior | How do speed, bouts, and movement change pre/training/post? | Track kinematics and swim bouts, plus stimulus epochs |
| Egocentric bearing | Where is the chaser relative to the fish's heading? | Track heading plus chaser distance |
| Gaze tracking | Does the fish orient its eyes toward the chaser? | Egocentric bearing plus completed eye-angle run |
| Bout response | Do bouts approach, turn toward, or widen distance from the chaser? | Swim bouts, egocentric bearing, chaser distance |
| Escape events | Which chaser encounters produce successful escapes? | Bout response plus chaser distance |
| Radial occupancy | How does chaser position/occupancy vary with radius? | Chaser distance and arena geometry |
| Response regimes | Are responses approach, escape, or freeze-like across distance? | Chaser distance plus track speed/separation metrics |
| Escape/freeze summary | Per-chaser escape and immobility endpoints | Chaser distance and movement metrics |

## Batman readiness observed today

The recording archive contains 44 Batman analysis Zarrs. The migration is now
complete for all 44 recording directories: 36 canonical derivative H5s and
receipts were created in this run, while 8 pre-existing derivatives were left
untouched. A post-migration H5/receipt verification found zero issues.

The Zarr publications still need to be updated separately; this migration does
not import derivatives into Zarr. The current census found:

- 8 Zarr archives with a complete canonical migrated stimulus surface;
- 36 Zarr archives still exposing the older metadata-only stimulus publication,
  now with canonical derivative H5s ready for import;
- 36 archives with complete refined detection metadata and 8 without a refined
  detection run yet;
- 0 archives with keypoint or subject-mask runs.

The canonical stimulus surface is therefore the first gate for distance-based
chaser analytics. Keypoints are needed for heading/track kinematics and
egocentric/bout analyses; subject masks plus keypoints are needed for eye-angle
and gaze tracking.

## Run order for Batman

1. Preflight and, after explicit approval, create immutable canonical stimulus
   derivatives:

   ```bash
   scripts/py -m fisheye.utils.migrate_legacy_batman_stimulus_h5 \
     /groups/johnson/johnsonlab/jeremy/recordings/*Batman
   
   # Apply only after reviewing the dry-run output:
   scripts/py -m fisheye.utils.migrate_legacy_batman_stimulus_h5 \
     /groups/johnson/johnsonlab/jeremy/recordings/*Batman --apply
   ```

2. Import each `.canonical_stimulus_v1.h5` derivative into its analysis Zarr
   with a new immutable run name, then validate that the selected run contains
   `tracking_data/chaser_states`, canonical coordinate evidence, and complete
   frame mapping. The import entry point is:

   ```bash
   scripts/py -m fisheye.analysis.import_stimulus_to_zarr \
     /path/to/<recording>.canonical_stimulus_v1.h5 \
     /path/to/<recording>_analysis.zarr \
     --run-name stimulus_batman_coordinate_v1_YYYYMMDD
   ```

3. Finish refined detections, keypoints, and subject masks. Once those are
   complete, build movement/bout prerequisites and run the chaser batch. The
   batch launcher is the supported DAG runner:

   ```bash
   find /groups/johnson/johnsonlab/jeremy/recordings \
     -maxdepth 4 -type d -iname '*Batman*' -name '*_analysis.zarr' \
     -print > /tmp/batman_zarrs.txt

   scripts/submit_chaser_analytics_bsub.sh \
     --zarr-list /tmp/batman_zarrs.txt \
     --source filesystem \
     --palette-repo /groups/johnson/johnsonlab/jeremy/gitrepos/palette \
     --preset goodcopbadcop_v2 \
     --log-dir /groups/johnson/johnsonlab/jeremy/recordings/logs/chaser_analytics
   ```

   `goodcopbadcop_v2` is currently the closest full-surface preset: it selects
   all 13 chaser modules and the separate eye-angle/gaze targets. Use
   `--dry-run` first; the observed dry-run selected 44 targets with 4 cores,
   16 GB per task, and at most 8 concurrent array elements. It did not submit
   jobs.

4. For a first scientific smoke, run one archive through the basic surface,
   inspect the distance/epoch/egocentric plots, then enable the expanded v2
   modules across the cohort. Keep the exact stimulus, detection, keypoint,
   mask, track, and epoch run names in the resulting provenance.

## Interpretation cautions for the meeting

- The exported plots show the available measurement families and plot shapes;
  they do not establish Batman effects.
- Distance and occupancy are projected arena-surface quantities converted to
  millimetres using the sealed projector scale. They are not world-coordinate
  measurements.
- The pre/training/post windows are protocol-derived epoch windows. Any Batman
  summary should report the exact event/window run used.
- Gaze is an optional higher-level endpoint and should be presented only after
  eye-angle source provenance and keypoint/mask coverage pass validation.
