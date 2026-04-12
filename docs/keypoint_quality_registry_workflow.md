# Keypoint Quality Registry Operator Workflow

This workflow keeps keypoint training selection deterministic and fail-closed.

Related future quality heuristic:

- [keypoint_temporal_heading_heuristic_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/keypoint_temporal_heading_heuristic_todo.md)

## Temporal Heading Review

Temporal heading review is intended for temporally contiguous refined-keypoint
runs, such as analysis/full-recording review surfaces.

It is explicitly disabled for sampled imports, including typical training zarrs
created with `import_mode="sampled"` / `frame_step > 1`. In those archives:

- temporal-heading arrays are omitted
- `summary_statistics.postprocess.temporal_heading_status` is
  `disabled_sampled_import`
- `summary_statistics.postprocess.temporal_heading_disabled_reason` is
  `sampled_import`

### One-time backfill

When temporal-heading fields or policy metadata need to be refreshed on
existing refined-keypoint runs:

```bash
scripts/py -m fisheye.utils.backfill_keypoint_heading_fields \
  /nvme1/recordings \
  --recursive \
  --zarr-use training
```

Apply changes:

```bash
scripts/py -m fisheye.utils.backfill_keypoint_heading_fields \
  /nvme1/recordings \
  --recursive \
  --zarr-use training \
  --apply
```

### Review flagged temporal outliers

For contiguous archives where the heuristic is enabled, open only ROIs flagged
by `heading_temporal_outlier`:

```bash
scripts/py -m fisheye.utils.review_keypoints_batch \
  /nvme1/recordings \
  --recursive \
  --zarr-use analysis \
  --manual \
  --jump-temporal-outliers
```

The manual reviewer will use the flagged ROI subset as its queue, so `n` / `p`
walk only those targeted rows.

### Export and sort by temporal outliers

To inspect which archives have the most temporal-heading outliers:

```bash
scripts/py -m fisheye.utils.export_keypoint_quality_overview \
  /nvme1/recordings \
  --recursive \
  --zarr-use analysis \
  --list \
  --sort-by temporal-outliers
```

Or sort by rate instead of count:

```bash
scripts/py -m fisheye.utils.export_keypoint_quality_overview \
  /nvme1/recordings \
  --recursive \
  --zarr-use analysis \
  --list \
  --sort-by temporal-outlier-rate
```

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

## 3.1 Registry hygiene one-liner (recommended)

Use the maintenance orchestrator to reconcile missing dataset paths, delete dangling
dataset rows, and run integrity checks in one command:

```bash
scripts/py -m fisheye.registry.maintenance --registry /path/to/palette_registry.sqlite --reconcile-registry
```

Dry-run preview:

```bash
scripts/py -m fisheye.registry.maintenance --registry /path/to/palette_registry.sqlite --reconcile-registry --dry-run
```

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

## 4.1 Optional quality visualization export/view

Finalize artifacts, then export/view refined-keypoint quality dashboards
(detect-style operator UX):

```bash
scripts/py -m fisheye.utils.finalize_keypoint_refinement_artifacts /path/to/recordings --recursive --zarr-use training --required-intended-use training --apply

scripts/py -m fisheye.utils.export_keypoint_quality_overview /path/to/recordings --recursive --zarr-use training --artifact keypoint_quality_overview_png --view
scripts/py -m fisheye.utils.export_keypoint_quality_overview /path/to/recordings --recursive --zarr-use training --artifact keypoint_refinement_pipeline_overview_png --view
```

## 5. Recovery for stale or divergent metadata

If preflight/integrity reports stale or divergent quality rows:

1. Re-run keypoint review/audit for affected archives.
2. Confirm refined-run attributes in archive metadata (`zarr.json` for refined groups).
3. Re-run `--refresh-keypoint-quality`.
4. Re-run `--check-integrity`.
