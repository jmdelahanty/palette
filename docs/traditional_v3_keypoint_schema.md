# Traditional V3 Keypoint Schema

Last verified: 2026-06-25

## Purpose

`traditional_v3` extends the current 5-point `traditional_v2` skeleton with
tail and pectoral-fin landmarks while preserving existing keypoint indices.

This schema is intended for new manual labeling and future pose-model training.
It should not mutate existing `traditional_v2` runs in place.

## Schema Identity

- pose schema name: `traditional_v3`
- skeleton id: `pose_skel_traditional_v3`
- runtime coordinate shape: `(N, 10, 2)`
- training/export `kpt_shape`: `[10, 3]`
- schema file:
  `configs/fisheye/pose_schemas/traditional_v3.json`
- derived metric schema file:
  `configs/fisheye/keypoint_metric_schemas/traditional_v3.json`

## Keypoint Order

The first five labels are identical to `traditional_v2`:

1. `swim_bladder`
2. `eye_left`
3. `eye_right`
4. `snout_tip`
5. `tail_tip`

New labels are appended:

6. `mid_tail`
7. `right_pectoral_fin_insertion`
8. `right_pectoral_fin_tip`
9. `left_pectoral_fin_insertion`
10. `left_pectoral_fin_tip`

Appending preserves all existing `traditional_v2` label indices and keeps
partial migration deterministic.

## Skeleton Edges

The head triangle and snout links remain:

- `swim_bladder -> eye_left`
- `swim_bladder -> eye_right`
- `eye_left -> eye_right`
- `snout_tip -> eye_left`
- `snout_tip -> eye_right`

The tail axis is routed through the new mid-tail point:

- `swim_bladder -> mid_tail`
- `mid_tail -> tail_tip`

The pectoral-fin topology is:

- `eye_right -> right_pectoral_fin_insertion`
- `swim_bladder -> right_pectoral_fin_insertion`
- `right_pectoral_fin_insertion -> right_pectoral_fin_tip`
- `eye_left -> left_pectoral_fin_insertion`
- `swim_bladder -> left_pectoral_fin_insertion`
- `left_pectoral_fin_insertion -> left_pectoral_fin_tip`

## Heading Policy

Heading remains defined by the stable core head triangle:

- `swim_bladder`
- `eye_left`
- `eye_right`

The new landmarks do not affect heading computation. This avoids changing
downstream heading behavior while the richer skeleton is being labeled and
validated.

## Migration Path

Create new runs rather than editing old runs in place:

```bash
scripts/py -m fisheye.utils.extend_keypoint_skeleton \
  /path/to/recording_training.zarr \
  --source-parent refined_keypoints_runs \
  --source-run <traditional_v2_refined_run> \
  --target-schema traditional_v3 \
  --target-run <new_traditional_v3_seed_run> \
  --apply
```

Expected behavior:

- existing five keypoints are copied into indices `0..4`
- new v3 keypoints are initialized missing
- the target run is marked as needing completion
- training/export still requires one skeleton identity per job

Do not use a generated `traditional_v3` seed as a training label source until
the new landmarks are manually completed and the refined run is reviewed.

## Derived Metrics

`traditional_v3` includes the `traditional_v2` distance metrics plus:

- anterior tail segment: `swim_bladder -> mid_tail`
- posterior tail segment: `mid_tail -> tail_tip`
- left/right pectoral fin insertion-to-tip lengths
- left/right eye-to-pectoral-insertion distances

These are skeleton-specific derived metrics. Cross-skeleton registry/query
surfaces should continue to treat derived metrics as schema-scoped.
