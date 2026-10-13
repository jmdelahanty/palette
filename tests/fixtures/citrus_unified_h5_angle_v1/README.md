# Witness H5 files for admission revision 5 (angle_v1)

Generated on the Citrus branch `agent/citrus/stimulus-angle-convention-20261012`
at `c9d52be` (commit B: `kAdmissionSha256` and the preflight v3
`admission_policy_sha256` const are the revision 5 policy `9af921ca…`) by the
synthetic bound-pair fixture (no cameras):

```
headless_h5_core_writer_fixture --case full_bound_pair_deferred_capacity_preflight \
  --out-dir <new dir> --experimental-bound-pair
```

| File | sha256 | What a consumer must do |
|---|---|---|
| `angle_v1_complete.h5` | `aee0217b984a7ad3c57d7a2a7a8db21f7bd746c05c77d6f769bb6cc157d54df1` | **Admit.** `session_status COMPLETE`, finalization `complete`; `capacity_preflight_json` schema_version 3, `catalog_id angle_v1`, `admission_policy_sha256 9af921ca…`; `/components/moving_grating/states` schema_version 3 with two rows at `motion_direction_degrees_canvas = 90` (up), `render_axis_degrees_canvas = 270`, `speed_px_per_sec = +100` (the drift rule: axis = motion + 180 at positive speed); `/metadata/session` `stimulus_angle_convention = atan2_negative_y_x_degrees`; `/components/chaser/states` carries no `angle_zero_direction` / `positive_rotation_direction`. Angles read as counter-clockwise (+90 = up). |
| `angle_v1_partial_adoption_refused.h5` | `db818b7aae5a6aa3babb9879ac530a9d3c98a8842347793b2c91a55e3565f482` | **Refuse.** Byte-identical to the complete file except `/components/moving_grating/states` attribute `schema_version` rewritten to 2 (`h5py attrs.modify`). Its map matches no admitted catalog while `catalog_id` says `angle_v1`: `catalog_id_mismatch`. It must not fall back to `frame_identity_v1` (which would read its angles as clockwise). |
| `angle_v1_missing_session_convention_refused.h5` | `6924a717fa130ecae3d90405f00d1488fa113a4d5b017d50fb67eaffe17a6ce8` | **Refuse.** Byte-identical to the complete file except the `/metadata/session` attribute `stimulus_angle_convention` deleted: binding `angle_v1` without its declaration, `stimulus_angle_convention_mismatch`. |
| `legacy_frame_identity_v1_with_session_convention_refused.h5` | `34025b7b4cf37e068726c612d1066563edd016019ee97d4476f57d25c24db313` | **Refuse.** Byte-identical to `../frame_identity_v1/frame_identity_v1_complete.h5` (revision 4 witness) except `/metadata/session` attribute `stimulus_angle_convention = atan2_negative_y_x_degrees` added. A revision-4 binding never carries the attribute: `stimulus_angle_convention_unexpected`. The sign is never taken from the attribute on a legacy binding. |

Frozen bytes: the logger records real clock values, so a regeneration has the
same structure with different bytes. Pin these by digest.
