# Witness H5 files for admission revision 7 (video_overlay_v1, pose observation v3)

Generated on the Citrus branch `agent/citrus/video-tracking-overlay` at `20f2739`
(commit B: `kAdmissionSha256` and the preflight v5 `admission_policy_sha256` const are the revision 7
policy `543102f7…`) by the synthetic bound-pair fixture (no cameras, no encoder):

```
headless_h5_core_writer_fixture --case full_bound_pair_deferred_capacity_video_overlay \
  --out-dir <new dir> --experimental-bound-pair
```

The overlay rows are made by the production code path (`TrackingTarget::Build` +
`VideoTrackingOverlay::Decide` + `ToLoggedState`) on the file's own pose update and the
bounding-box row they join on.

| File | sha256 | What a consumer must do |
|---|---|---|
| `video_overlay_v1_complete.h5` | `8057641421531d4b12ba829f877ccebda0d2f614b59db653b92228049de082f8` | **Admit.** `capacity_preflight_json` schema_version 5, `catalog_id video_overlay_v1`; `/components/video_tracking_overlay/states` schema_version 1 with 4 rows = 4 `/frames/stimulus` rows; `/components/video_tracking_overlay/settings_json` (radius 1 mm = 10 px at `pixels_per_mm_projector` 10, green, held_policy draw, arrow active 3 mm = 30 px, yellow, 2 px). Row 0: current box (`input_used` 1) from pose update sequence 1, camera 0, recording frame 1, box (10, 20, 30, 40), payload time 5000: `position_camera_px` (25, 40), `position_arena_local_px` (25.5, 20.5) (integer rectangle x 10..40, y 20..60 under the identity homography, arena origin (0, 20)), heading status 7, 0 deg at (20, 10), arrow origin (20, -10), direction (1, 0). Rows 1..3: held box (`input_used` 2), same box identity and timestamp, no heading. No encoder: `video_frame_index_valid` 0, `drawn` and `arrow_drawn` 0 on every row, `decision_draw` 1. Pose v3: contract 3, sidecar at `/observations/pose/sidecars/2eb288ea7245aba076ae6eee56e6262420c7a104f4e156e9cab414957e5732db`, object statuses [7, 9, 7, 5, 6, 9, 3] (7; 9 eyes swapped; 7 confidence exactly at the 0.5 gate; 5 one ulp below; 6 tail on the fin midpoint; 9 eye_left absent; 3 fin_b absent). |
| `video_overlay_v1_rig_homography_complete.h5` | `0fcfd8aeec0ea591d8597a31671766370836bd7cf40acaa37549c85c6dd58c53` | **Admit.** The same case with the rig homography recorded for camera 2010096 in the 2026-08-25 redbat session (y reflection, rotation, perspective row (-7.445e-07, -2.962e-07, 1)) and the fish translated by (500, 500): no projected corner is an integer. Encoder acceptance simulated: `/frames/stimulus` frames 0..2 in the video (indices 0..2), frame 3 dropped, so rows 0..2 have `drawn` 1 and row 3 `drawn` 0; row 0 `arrow_drawn` 1 (current pose, midline valid). Row 0: `position_camera_px` (525, 540), `position_arena_local_px` (1514.5, 441.0), arrow origin (1514.0162, 443.51263), direction (0.99997574, -0.0069679441). |
| `video_overlay_v1_probe_horizon_complete.h5` | `55ef06963664c24adb2a22ab6ca27d0f44def05cd8546c09860bce8151b631ee` | **Admit.** A constructed homography (rotation and scale 1.25, perspective row (1/4096, -1/1024, 1)) whose line w = 0 passes exactly through the heading's left probe (512, 1152); the fish is rotated 180 degrees (forward (-1, 0), left axis (0, 1)) about (266, 571). Row 0: status 7 heading (-180 deg at (512, 1132)), `heading_valid` 1, `midline_valid` 0 (the left probe projects with w = 0), `drawn` 1, `arrow_drawn` 0; every box corner, the origin and the forward probe have w > 0 (position (-8100.5, 40095.5)). Encoder acceptance as the rig witness. |
| `video_overlay_v1_partial_adoption_refused.h5` | `262d9522768221d808a283eaa05593ecf58390e9a210c762d20aecfda660e62a` | **Refuse.** `/components/video_tracking_overlay/states` attribute `schema_version` rewritten to 2: `unsupported_table_version` (or `catalog_id_mismatch` where catalogs are selected first); never a fallback to `phototaxis_v1`. |
| `video_overlay_v1_catalog_id_phototaxis_v1_refused.h5` | `348f19afd227db6bbe719c704c776962ae3cea474883d6d1fc393a1050915c16` | **Refuse.** `capacity_preflight_json.catalog_id` rewritten to `phototaxis_v1` while the file declares and carries `/components/video_tracking_overlay/states`: `catalog_id_mismatch`. |
| `video_overlay_v1_states_position_mismatch_refused.h5` | `9378a79ef34176383373e345e53bc67b531f22e069c72f5362a188556254cd1b` | **Refuse.** row 0 `position_camera_px` rewritten to (26, 40): the bounding-box row with the same payload timestamp has rect centre (25, 40), which must match exactly. |
| `video_overlay_v1_states_missing_box_row_refused.h5` | `a0ac0dcc67362a9df7254295d591ebaceea7d624f2e571bed84dc7aed4d8f576` | **Refuse.** `box_recording_frame_id` rewritten to 77 on every row: no bounding_boxes v2 row has camera 0, recording frame 77, object 0. |
| `video_overlay_v1_states_coverage_gap_refused.h5` | `ebc255acd3e83b164e6fd4cc47d85a1a982b651740393a3f36dfc6a0a470d721` | **Refuse.** `/components/video_tracking_overlay/states` resized to 3 rows while `/frames/stimulus` has 4: one states row per frame row while the component is declared. |
| `video_overlay_v1_states_arrow_flag_refused.h5` | `ad3d67f080e1cc864245ece774da41c80e89256253e52bbda2544a9b5ba4ab65` | **Refuse.** row 0 `arrow_drawn` rewritten to 1 while `drawn` is 0: `arrow_drawn` iff `drawn` and `midline_valid` and the arrow is active. |
| `video_overlay_v1_pose_heading_not_reproduced_refused.h5` | `796a910a2467534495fabe659abf862fa66c4f4f4637429b4e8230482477676d` | **Refuse.** pose keypoint (object row 0, label 0 `tail`) moved from (10, 10) to (10, 12): the status-7 object's stored heading (0 deg, origin (20, 10)) is no longer what its keypoints give (pose v3 recomputation). |
| `video_overlay_v1_pose_sidecar_bytes_refused.h5` | `66cbf42be3c82641d4294c6d9c6a8fa4d69067922194bd5f89ff6e92bab404ed` | **Refuse.** `/observations/pose/sidecars/2eb288ea7245aba076ae6eee56e6262420c7a104f4e156e9cab414957e5732db` bytes changed (one trailing space): they no longer hash to `sidecar_sha256` (pose v3). |
| `video_overlay_v1_rig_drawn_without_video_frame_refused.h5` | `4b064cee9710441b56f5d4b46e2fb03309b36d8ca46898f997c2fbce4230bd92` | **Refuse.** from the rig-homography witness: row 3 `drawn` rewritten to 1 while its frame is not in the video (`video_frame_index_valid` 0): `drawn` iff `decision_draw` and `video_frame_index_valid`. |
| `video_overlay_v1_rig_position_off_by_one_refused.h5` | `611bad844e661ecc306a2f782bb97829edcad646e4235793f665c285d8e208b9` | **Refuse.** from the rig-homography witness: row 0 `position_arena_local_px` rewritten from (1514.5, 441.0) to (1515.5, 441.0): the exact integer-rectangle replication refuses a one-pixel error (no tolerance). |
| `video_overlay_v1_horizon_arrow_drawn_refused.h5` | `10ee0e457819d831771abe111ddd53442665f0e00e6524224d5c072723cd434c` | **Refuse.** from the probe-horizon witness: row 0 `arrow_drawn` rewritten to 1 while `midline_valid` is 0 (the left probe lies on w = 0): `arrow_drawn` iff `drawn` and `midline_valid` and active. |

Every refused file was edited with h5py after the logger sealed it, so a full pipeline refuses
it at integrity (`canonical_dependency_mismatch`) first; the named validator alone refuses it
with its own reason.

The three complete files cover: exact-integer corners (identity homography, the
floor-boundary case), general projective arithmetic with a y reflection (rig homography),
and a left-probe failure on the w = 0 horizon (probe horizon); the last two also exercise
`drawn` and `arrow_drawn` on true values. Every row of all three was recomputed by an
independent Python replication of the contract's arithmetic
(`../../video_tracking_overlay/check_overlay_arith.py`): zero differences.

Frozen bytes: the logger records real clock values, so a regeneration has the same structure
with different bytes. Pin these by digest.

First row (for reference): `{"input_used": 1, "position_arena_local_px": [25.5, 20.5], "position_camera_px": [25.0, 40.0], "box_payload_timestamp_ns_monotonic": 5000, "heading_status": 7, "arrow_origin_arena_local_px": [20.0, -10.0], "arrow_direction_canvas": [1.0, 0.0]}`
