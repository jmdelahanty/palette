# Recording Store Relocation Runbook
<!-- runbook-meta
status: active
last_verified: 2026-05-28
purpose: Operator checklist for moving Palette recordings between storage roots while preserving registry, Zarr, sidecar, and review-proxy consistency.
-->

## Scope

Use this runbook when promoting a recording from workstation-local storage such
as `/nvme1/recordings` to durable storage such as
`/groups/johnson/johnsonlab/jeremy/recordings`.

The component policy is defined in
`docs/recording_store_relocation_components.md`. This runbook is the operator
sequence.

## Preconditions

- You know the source recording root and destination recording root.
- No training/export/inference jobs are writing to the source or destination
  roots.
- The registry path is known.
- You have enough storage quota at the destination.
- For clipped recordings, the source root has `recording_frame_index.parquet`
  and clip sidecars.

Recommended shell variables:

```bash
SRC=/nvme1/recordings/<recording_id>
DST=/groups/johnson/johnsonlab/jeremy/recordings/<recording_id>
REG=/nvme1/palette_registry.sqlite
```

## 1. Inspect Before Copying

Confirm the source root shape:

```bash
find "$SRC" -maxdepth 2 -type d | sort | head -80
find "$SRC" -maxdepth 2 -type f \( -name 'recording_*' -o -name '*manifest*.json' \) | sort
```

For clipped recordings, confirm clip count:

```bash
find "$SRC/clips" -maxdepth 1 -type d -name 'clip_*' | wc -l
```

Inspect registry rows before writing:

```bash
sqlite3 "$REG" "
SELECT dataset_id, zarr_path, zarr_use, source_layout
FROM datasets
WHERE zarr_path LIKE '$SRC/%' OR zarr_path LIKE '$SRC';
"
```

## 2. Back Up The Registry

Always make an SQLite backup before registry edits:

```bash
BACKUP=/groups/ahrens/ahrenslab/jeremy/zebrobot/backups/palette_registry_$(date +%Y%m%d_%H%M%S).sqlite
sqlite3 "$REG" ".backup '$BACKUP'"
sqlite3 "$BACKUP" "PRAGMA quick_check;"
```

Do not proceed if the backup quick check is not `ok`.

## 3. Copy The Physical Root

Use `rsync` so reruns are safe:

```bash
mkdir -p "$(dirname "$DST")"
rsync -aH --info=progress2 "$SRC/" "$DST/"
```

Run a second dry pass to check that the copy converged:

```bash
rsync -aH --dry-run --itemize-changes "$SRC/" "$DST/" | head -80
```

Expected result after convergence: no meaningful file changes. Directory mtime
noise is acceptable; missing files are not.

## 4. Validate The Destination Shape

For clipped recordings:

```bash
find "$DST/clips" -maxdepth 1 -type d -name 'clip_*' | wc -l
test -f "$DST/recording_frame_index.parquet"
test -f "$DST/recording_clip_index.json"
```

For training Zarrs, confirm the expected store exists:

```bash
find "$DST/zarr" -maxdepth 1 -type d -name '*.zarr' | sort
```

For review proxies copied with the root, check manifests and proxy files before
rewriting active paths.

## 5. Rewrite Active Sidecars And Zarr Attrs

Rewrite only active location pointers. Preserve historical provenance fields
that describe where an artifact was originally created.

Active path surfaces to inspect:

- Zarr root attrs such as `recording_path` and
  `source_recording_frame_index_path`;
- `recording_frame_index.parquet` path columns;
- training-local `source_frame_index.parquet` path columns;
- `*_clipped_training_manifest.json`;
- review proxy `manifest.json`;
- finalized collection `selected_runs[*].source.*` paths.

After rewriting, add an explicit relocation note where the artifact supports
attrs or manifest metadata.

Do not rewrite frozen exported datasets or model artifacts. Build a new export
from the migrated registry state instead.

## 6. Update Registry Pointers

Registry updates must target the canonical active dataset row. Do not create a
second active row for the same physical Zarr unless there is an intentional
dataset-id split, such as an old analysis dataset id and a new clipped-training
dataset id for the same recording.

Fields commonly requiring updates:

- `recordings.recording_path`
- `datasets.zarr_path`
- `datasets.path_hash`
- `datasets.source_recording_frame_index_path`
- path-bearing `detection_data_profile.profile_json`
- path-bearing `training_sets.query_filter`
- path-bearing `training_sets.invocation_json`

`datasets.path_hash` is path-derived. Recompute it with registry code or a
purpose-built migration helper; do not leave the old hash after changing
`zarr_path`.

After registry edits:

```bash
sqlite3 "$REG" "PRAGMA foreign_key_check;"
sqlite3 "$REG" "PRAGMA quick_check;"
```

Both must be clean.

## 7. Dedupe Duplicate Registry Rows

Run dedupe in read-only mode first:

```bash
scripts/py -m fisheye.registry.dedupe --registry "$REG"
```

Before deleting a duplicate `dataset_id`, search every registry table for that
id. Safe duplicate cleanup requires:

- no training-set membership under the duplicate id, or an intentional migration
  of that membership;
- no unique quality/profile data only attached to the duplicate id;
- current status rows either duplicated under the canonical id or explicitly
  stale;
- history rows preserved by moving them to the canonical id before deleting the
  duplicate.

Keep foreign keys enabled during cleanup. Re-run `foreign_key_check` and
`quick_check` after deletion.

## 8. Rebuild Or Relocate Review Proxies

If copied proxy manifests point to the old root, rewrite active paths and verify
every proxy exists.

For long clipped recordings without valid copied proxies, submit sharded proxy
generation from the cluster repo:

```bash
scripts/submit_review_proxy_videos_sharded_bsub.sh \
  "$DST" \
  --proxy-run-id <proxy_run_id> \
  --proxy-width 1024 \
  --proxy-height 1024 \
  --encoder h264_nvenc \
  --hwaccel cuda \
  --scale-flags bilinear \
  --shard-count 4 \
  --max-active 4 \
  --queue gpu_l4 \
  --gpus 1 \
  --walltime 2:00 \
  --overwrite \
  --submit
```

The sharded wrapper publishes `manifest.json` only from its finalizer. Do not
treat shard completion alone as a valid proxy run.

## 9. Final Validation

Minimum gates:

- destination root exists and has expected clip/Zarr sidecars;
- no active path columns or manifests still point at the old root;
- registry has exactly one canonical active dataset row per intended active
  Zarr path;
- registry `foreign_key_check` is clean;
- registry `quick_check` is `ok`;
- review proxy manifests have no missing proxy paths;
- a task-specific dry run selects the relocated Zarr.

Example task-specific dry runs:

```bash
scripts/py -m fisheye.tune.video_detect_review_web \
  "$DST/zarr/<recording_id>_analysis.zarr" \
  --review-proxy-manifest "$DST/derived/review_proxy/video_detect/<proxy_run_id>/manifest.json" \
  --help

scripts/py -m fisheye.registry.dedupe --registry "$REG"
```

For actual viewer validation, launch the reviewer normally and inspect at least
one known frame/clip.

## 10. Rollback

If validation fails before new jobs depend on the migrated root:

1. Restore the registry backup.
2. Leave the copied files in place for inspection or remove them only after
   explicit operator approval.
3. Document the failed migration and the validation gate that failed.

If downstream jobs already used the migrated root, do not blindly restore the
registry. First identify generated artifacts and whether they should be retained
or invalidated.

