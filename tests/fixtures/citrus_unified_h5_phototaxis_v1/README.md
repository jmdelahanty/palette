# Witness H5 files for admission revision 6 (phototaxis_v1)

Generated on the Citrus branch `agent/citrus/reactive-phototaxis-states` at
`d6754a3` (commit C: `kAdmissionSha256` and the preflight v4
`admission_policy_sha256` const are the revision 6 policy `012cd25f…`)
by the synthetic bound-pair fixture (no cameras):

```
headless_h5_core_writer_fixture --case full_bound_pair_deferred_capacity_phototaxis \
  --out-dir <new dir> --experimental-bound-pair
```

| File | sha256 | What a consumer must do |
|---|---|---|
| `phototaxis_v1_complete.h5` | `e9777bdacd4ac5aa1398b5ef6066abc2636407de74e187629e3e66522b6836dd` | **Admit.** `session_status COMPLETE`, finalization `complete`; `capacity_preflight_json` schema_version 4, `catalog_id phototaxis_v1`, `admission_policy_sha256 012cd25f…`; `/components/phototaxis/states` schema_version 1 with 4 rows, all on the heading-split step (recipe step 0, `protocol_step_index` 0): frame 0 is a pose update (`input_used` 1, `heading_status` 7, heading 0.0° at camera (20, 10), midline origin (20, 10) direction (1, 0) left (0, -1) angle 0 under the identity homography, `held_*` equal to it, `held_age_updates` 0) and frames 1..3 are held updates (`input_used` 2, `heading_status` 0, `heading_valid`/`midline_valid` 0 with zero payload, held midline unchanged, `held_age_updates` 1..3); `fallback_id` 0; `source_camera_id` 0 and `pose_sequence_id` 1 / `pose_recording_frame_id` 1 name the file's one status-7 pose update (camera_id 0, same heading and origin); `box_sequence_id_valid` 0 (the slot carries no sequenced box); `pose_model_hash64` / `pose_skeleton_hash64` are prefix64 of the file's own `contract_json` engine / sidecar sha256 (`pose_identity_valid` 1). `/protocol_execution` carries step 0 completed over `[0, 4)` (STEP_START at frame 0, STEP_END at frame 3); `/definitions/enums/phototaxis_*` present; every other table and attribute as the angle_v1 complete witness. |
| `phototaxis_v1_partial_adoption_refused.h5` | `bfb8c1618812ddf47bd395f94633512680e7bb0f7e3be99deae292082dac56ef` | **Refuse.** Byte-identical to the complete file except `/components/phototaxis/states` attribute `schema_version` rewritten to 2 (`h5py attrs.modify`). Its map matches no admitted catalog while `catalog_id` says `phototaxis_v1`. Refused as `catalog_id_mismatch` or, where a consumer checks table versions before catalog selection, as `unsupported_table_version:/components/phototaxis/states` (Palette revision 6 does). Either way it must not fall back to `angle_v1`. |
| `phototaxis_v1_catalog_id_angle_v1_refused.h5` | `0b4efec0a4de3bff09963b33bf1cad2fc68d360580ea109c449fc1be748811cc` | **Refuse.** Byte-identical to the complete file except `capacity_preflight_json.catalog_id` rewritten to `angle_v1` while the file declares and carries `/components/phototaxis/states`, a table whose version is in no admitted map of `angle_v1`: `catalog_id_mismatch` (the policy digest and preflight v4 also belong to `phototaxis_v1`, not `angle_v1`). |
| `phototaxis_v1_states_wrong_step_refused.h5` | `b201c091000debc3a2d60e833805f17e4bb68d418858e557a16ad3c1d7326446` | **Refuse.** `/components/phototaxis/states` rows' `protocol_step_index` rewritten to 1 (the `chaser` step, not the heading-split step 0): `phototaxis_step_not_active` (the row's step must be a heading-split step whose execution interval contains `stimulus_frame_num`). |
| `phototaxis_v1_states_hash_mismatch_refused.h5` | `b510722d846857ee1e5be1f53ea3e1059e5519d54fe01f4b0c8e9407058f1cb7` | **Refuse.** `pose_model_hash64` rewritten to `0x1234567890abcdef`: it must equal prefix64 of `/observations/pose/contract_json` `metadata_cameras.<serial>.engine_sha256`. |
| `phototaxis_v1_states_missing_pose_row_refused.h5` | `dcfaecf80186929565e88ed97d30cd3c18a750a185c71788955d281ee68a181e` | **Refuse.** `pose_sequence_id` rewritten to 99 on an `input_used` 1 row: `/observations/pose/updates` has no row with that sequence, so the status-7 object the heading came from cannot be found. |

The three `states_*_refused` files were edited with h5py after the logger
sealed the file, so a full admission pipeline that checks sealing dependencies
first refuses them at integrity (`canonical_dependency_mismatch`, Palette
revision 6) before reaching the states validator. Run on the states table
alone, each yields its own code: `phototaxis_step_not_active`,
`phototaxis_pose_identity_mismatch`, `phototaxis_pose_row_missing`. They are
witnesses for the states validator, not for the integrity stage; sealed
variants written through the logger are a follow-up if a consumer needs the
specific codes from the full pipeline.

Row/step frame semantics (real logger, `src/core/arena.cpp`): a step's execution
interval is half-open `[start, end)` where `start` is the stimulus frame the
STEP_START event carried and `end` is the STEP_END frame + 1. Protocol events
carry the frame of the previous arena update (events are processed before the
frame counter advances) while states rows carry the frame of the update that
produced them, so in a real file a step's first row is at `start + 1`, its last
row at `end - 1`, and consecutive intervals overlap by exactly one frame. A
row's step is `protocol_step_index`, checked as "that step's interval contains
`stimulus_frame_num`", never as "the unique interval containing it". The
fixture has no update before its first step, so its first row sits at `start`.

Other real-file facts a validator must allow for: the pose table evaluates
headings with the hard-coded `contract_json.minimum_keypoint_confidence` (0.5)
while the view uses the step's `minimum_keypoint_confidence`, so the pose-row
cross-check (same heading on a status-7 row) holds iff the step's value is at
least the contract's (tomorrow's protocol: 0.5); the first row of a heading-split
step may be a fallback row (`input_used` 3) when no heading is held yet;
`source_camera_id` is the slot's SHAMAN numeric camera id, cross-checked against
`binding_json.shaman_numeric_camera_id` and the pose update's `camera_id`.

Frozen bytes: the logger records real clock values, so a regeneration has the
same structure with different bytes. Pin these by digest.
