# Keypoint Quality Registry Operator Workflow

This workflow keeps keypoint training selection deterministic and fail-closed.

## 1. Review keypoints in Zarr

Run keypoint review so refined runs have:

- `keypoint_review_status.state`
- `keypoint_review_status.intended_use`
- `usable_keypoints` (or summary statistics with usable totals/rate)

## 2. Refresh registry quality rows

Use maintenance refresh to sync quality rows and remove stale entries:

```bash
scripts/py -m fisheye.registry.maintenance --registry /path/to/palette_registry.sqlite --refresh-keypoint-quality --dry-run
scripts/py -m fisheye.registry.maintenance --registry /path/to/palette_registry.sqlite --refresh-keypoint-quality
```

Backfill-only mode (do not touch datasets that already have quality rows):

```bash
scripts/py -m fisheye.registry.maintenance --registry /path/to/palette_registry.sqlite --backfill-keypoint-quality
```

## 3. Integrity check

Run integrity checks before training:

```bash
scripts/py -m fisheye.registry.maintenance --registry /path/to/palette_registry.sqlite --check-integrity
```

Keypoint quality checks include stale `zarr_mtime_ns`, missing refined runs, and row divergence.

Quick operator report (summary + optional details):

```bash
scripts/py -m fisheye.utils.check_training_registry --registry /path/to/palette_registry.sqlite --show-keypoint-quality
```

Current limitation: `quality_stale` in the overview report is `1` when stored
`zarr_mtime_ns` is missing, and `0` otherwise. It does not yet compare against
a dataset mtime snapshot column in the registry.

## 4. Build/train with SQL gates

Use quality gates in preflight:

```bash
scripts/py -m fisheye.utils.prepare_keypoint_training_from_registry \
  --registry /path/to/palette_registry.sqlite \
  --require-review-state approved \
  --require-review-intended-use training \
  --min-usable-keypoints-rate 0.70
```

## 5. Recovery for stale or divergent metadata

If preflight/integrity reports stale or divergent quality rows:

1. Re-run keypoint review/audit for affected archives.
2. Confirm refined-run attributes in archive metadata (`zarr.json` for refined groups).
3. Re-run `--refresh-keypoint-quality`.
4. Re-run `--check-integrity`.
