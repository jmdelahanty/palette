# PRFS Registry Migration Status - 2026-06-18

## Summary

Palette's durable registry target is now:

```text
/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite
```

Future cluster-oriented workflows should use this PRFS registry and the PRFS
recording root:

```text
/groups/johnson/johnsonlab/jeremy/recordings
```

The local `/nvme1/palette_registry.sqlite` registry should be treated as a
legacy workstation snapshot. Do not use it as the authority for new cluster
runs, new recording imports, or long-lived status updates.

## What Changed

- Created the canonical PRFS registry from the latest groups-side snapshot:
  `/groups/johnson/johnsonlab/jeremy/registries/palette_registry_model_paths_groups_20260617T024552Z.sqlite`.
- Added apply-capable dataset deduplication in `fisheye.registry.dedupe`.
- Applied deduplication to the canonical PRFS registry.
- Refreshed recording step status for PRFS recordings under
  `/groups/johnson/johnsonlab/jeremy/recordings`.
- Updated registry defaults in config, cluster shell wrappers, analysis CLIs,
  and repair runbooks from `/nvme1/palette_registry.sqlite` to the PRFS registry.

## Applied Dedupe Results

Report:

```text
/groups/johnson/johnsonlab/jeremy/registries/palette_registry_dedupe_apply_20260618T1621.json
```

Observed results:

- Planned duplicate dataset groups: 73
- Duplicate `datasets` rows removed: 73
- Dependent rows repointed: 5,957
- Conflicting dependent rows deleted: 1,680
- Remaining duplicate active zarr paths: 0

The dedupe implementation keeps one canonical `datasets` row per duplicated
path, repoints non-conflicting dependent rows, deletes dependent rows that would
violate canonical uniqueness constraints, deletes lineage self-edges, and then
removes the duplicate dataset row.

## Recording Step Refresh

The PRFS registry was refreshed for analysis datasets under:

```text
/groups/johnson/johnsonlab/jeremy/recordings
```

Refresh scope and result:

- PRFS analysis datasets in scope: 21
- Source-recording datasets evaluated: 12
- `recording_step_status` rows inserted: 204
- `recording_step_status` rows updated: 156
- History rows written: 360

All 12 current GoodCopBadCop recordings have `refined_keypoints` status `ok` in
the PRFS registry.

## Validation

Completed validation:

- SQLite `PRAGMA quick_check`: `ok`
- SQLite `PRAGMA foreign_key_check`: no rows
- Duplicate active zarr path check: 0 duplicates
- Focused dedupe tests: passed
- Changed Python files: `py_compile` passed
- Changed shell scripts: `bash -n` passed
- Working-tree whitespace: `git diff --check` passed before this status note

## Remaining Caveats

The canonical PRFS registry still contains legacy `/nvme1` rows. That is
intentional for now: the registry has historical local data that has not all
been migrated to PRFS.

A broad registry integrity check still reports 82 pre-existing content issues.
These are not dedupe mechanics failures. They include old training lineage
mismatches, stale or missing keypoint-quality rows, missing `recording_id` values
on legacy source datasets, and missing keypoint-performance source-crop
projections.

Do not run broad destructive reconcile, prune, or filesystem-existence repair
from a cluster node while `/nvme1` rows remain in the canonical registry. Cluster
nodes cannot see workstation-local `/nvme1` paths, so a broad filesystem repair
from the cluster would misclassify local legacy rows as missing.

## Current Operating Rule

Use the PRFS registry for new work:

```bash
export PALETTE_REGISTRY_PATH=/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite
```

Use PRFS recording paths for cluster jobs:

```text
/groups/johnson/johnsonlab/jeremy/recordings/<recording_id>/zarr/<recording_id>_analysis.zarr
```

Use `/nvme1` only as a workstation-local legacy source while remaining datasets
are migrated or intentionally retired.

## Recommended Next Steps

1. Commit the PRFS registry migration code, tests, and documentation as one
   cohesive migration slice.
2. Audit subject-mask cluster submission next. The current subject-mask batch
   wrapper is serial and local-oriented; it does not yet have a dedicated LSF
   submitter equivalent to the detect, crop, or keypoint batch scripts.
3. For subject masks, prefer a PRFS-first submission wrapper that records the
   registry path, repo path, host, GPU, queue, job id, per-recording output JSON,
   and final refined-subject status.
