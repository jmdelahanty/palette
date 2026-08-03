# Bout Classification and Tail-Posture Exact Schemas

Date: 2026-08-03

Status: implemented contract checkpoint; no production selector or physical
profile promotion.

## Outcome

Palette now has executable, exact logical array declarations for the two
derived companion families that were previously described mostly by writer
behavior:

- `analysis/bout_classification_runs/<run>` schema v2;
- `analysis/tail_posture_view_runs/<run>` schema v3.

Each declaration freezes the relative path, exact dtype, symbolic shape,
axes, units or coordinate domain, fill/null semantics, access class, immutable
write mode, authority role, and current physical-policy owner. Both manifests
state `byte_planner_adopted=false`; this checkpoint does not pretend that the
legacy chunk helpers have already migrated to the shared byte planner.

## Bout classification v2

The maintained `per_bout` table contains exactly 20 arrays in a frozen order.
Scalar numeric fields retain their existing `int64`, `int32`, `float32`, and
`bool` representations. The two text fields are now physically exact,
cross-language `uint8` matrices:

- `category_label_bytes`: `[n_bouts, 64]`;
- `failure_reason_bytes`: `[n_bouts, 128]`.

This fixes the old behavior where the generic columnar writer selected the
stored text width from the longest value observed in a particular run. The
logical structured-table metadata remains fixed as `S64` and `S128`, so the
existing columnar readback helper remains compatible.

The writer persists `bout_classification_array_schema` plus its SHA-256 and the
activation validator reconstructs the executable declaration. A stored
manifest that is altered and rehashed still fails. Historical v1 is accepted
only through the explicit `legacy_compatibility=True` API or
`--legacy-compatibility` CLI option; v1 cannot pass the maintained writer's
activation proof.

## Tail-posture view v3

The maintained view contains exactly ten direct arrays:

```text
instance_key
source_crop_row_ids
source_acquisition_frame_index
valid
failure_reason_bytes
head_xy
head_yaw_rad
tail_keypoints_xy
tail_angle_rad
tail_angle_deg
```

Lineage is normalized to `uint64`, `int64`, and `int64`; posture payloads are
`float32`; validity is `bool`; reasons are `uint8[n_rows,64]`. The keypoint and
angle widths remain manifest dimensions (`n_keypoints` and
`n_angles=n_keypoints-1`) rather than being assumed from a particular model.

The view is explicitly a derived tool-compatibility surface, not the
scientific authority for subject shape. Its existing coordinate-publication
manifest continues to bind exact upstream geometry and payload hashes. The new
`tail_posture_view_array_schema` manifest adds the missing logical dtype,
shape, access, fill, and authority contract before that coordinate publication
is sealed.

The current physical owner remains the subject-mask metric row-chunk helper for
compatibility, and the declaration records that honestly. Migrating this
compact view to the shared byte planner is a later physical-policy step and
must be benchmarked separately.

## Publication checks

- Writers create immutable owner-bound, selector-ineligible candidates.
- Exact manifests and their digests are written before completion.
- Executable validators reject missing, unexpected, wrong-rank, wrong-shape,
  and wrong-dtype arrays.
- Recomputed-digest manifest tampering is rejected by comparison with the
  executable declaration.
- Existing completion, tombstone, freshness, and atomic selector activation
  rules remain in place.

## Deferred work

- Adopt the shared byte planner and benchmark the resulting physical profiles.
- Add consumer-facing strict-vs-legacy selection APIs for every tail-posture
  reader; this checkpoint changes no production selection behavior.
- Promote neither schema nor a physical profile until the coordinated
  derived-analytics gate is complete.
