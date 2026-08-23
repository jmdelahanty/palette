# Registry Repair Playbook

This playbook covers recovery after source recording Zarr path/name changes
(for example, renaming to `*_training.zarr`) when lineage or set membership
breaks.

For current production work, prefer:

```bash
REG=/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite
```

Use `/nvme1/palette_registry.sqlite` only for legacy/local snapshot repairs.

For stale registry rows that point at abandoned temp-root Zarr paths, use
`PYTHONPATH=src scripts/py -m fisheye.registry.prune_stale_datasets --registry "$REG" --json /path/to/dryrun.json`
first. The tool opens dry runs read-only, reports temp-root candidates and `/home`
rows needing maintainer review, plus active analysis rows without a normalized
`recording_id`. It only executes with `--execute --backup <path>
--include-temp-root all-temp`; exact `/home` or unowned-analysis dataset IDs must
be opted in with repeatable `--include-dataset-id`. The tool does not run
`VACUUM`; after a large reviewed prune, the maintainer may choose to vacuum the
NFS registry manually.

## 1) Backup First

```bash
sqlite3 /path/to/palette_registry.sqlite ".backup /path/to/palette_registry.backup.sqlite"
```

## 2) Rename Source Recording Zarrs (Optional, if not already done)

```bash
scripts/py -m fisheye.utils.rename_recording_zarrs_to_training \
  /nvme1/recordings \
  --recursive \
  --apply \
  --list-limit 100
```

## 3) Rescan Registry Paths

```bash
scripts/py -m fisheye.utils.registry_rescan \
  --registry "$REG" \
  /nvme1/recordings \
  --recursive
```

## 4) Remap Training Set Membership IDs

If source `dataset_id` values changed (for example to `session_uuid:z<hash>`),
remap `training_sets.dataset_ids_json` by session UUID:

```bash
scripts/py -m fisheye.registry.maintenance \
  --registry "$REG" \
  --remap-training-set-dataset-ids \
  --dry-run
```

Then apply:

```bash
scripts/py -m fisheye.registry.maintenance \
  --registry "$REG" \
  --remap-training-set-dataset-ids
```

## 5) Rebuild Dataset Lineage

```bash
scripts/py -m fisheye.registry.maintenance \
  --registry "$REG" \
  --backfill-dataset-lineage
```

## 6) Validate Integrity

```bash
scripts/py -m fisheye.registry.maintenance \
  --registry "$REG" \
  --check-integrity \
  --list-limit 100
```

Expected result:
- `Integrity check passed: no issues found.`

## 7) Duplicate Dataset Row Dry Run

After path or dataset-ID policy migrations, the registry may contain multiple
active `datasets` rows for the same physical Zarr path. Do not delete those
rows directly. First generate a dry-run dedupe plan:

For storage-root relocation, first identify which path references are active
location pointers and which are historical provenance. See
`docs/recording_store_relocation_components.md` before rewriting registry
paths, Zarr attrs, Parquet sidecars, or manifest fields.

```bash
scripts/py -m fisheye.registry.dedupe \
  --registry "$REG"
```

For the clipped training slice:

```bash
scripts/py -m fisheye.registry.dedupe \
  --registry "$REG" \
  --zarr-use training \
  --path-contains clipped_training.zarr \
  --json
```

The default report is read-only. It groups duplicate rows by exact `zarr_path`
and `path_hash`, proposes a canonical `dataset_id`, and lists every dependent
row that would need to be moved. `conflicting_rows > 0` means a direct update
would collide with an existing primary-key or unique-index row, most commonly
`recording_step_status(dataset_id, step_name)`.

Live stage-completion writes should now use the same effective dataset ID as
`register_from_root`: for source-recording zarrs under `/recordings/`, the
canonical form is `<session_uuid>:z<path_hash_prefix>`. If you see both a bare
session/recording ID and a `:z...` ID for the same exact `zarr_path`, the bare
ID is a legacy duplicate unless it points to a different path hash.

Quick check for one path:

```bash
sqlite3 -header -column "$REG" "
SELECT dataset_id, recording_id, zarr_use, artifact_kind, status
FROM datasets
WHERE zarr_path = '/path/to/recording_training.zarr'
ORDER BY dataset_id;

SELECT dataset_id, step_name, status, run_name, source, updated_utc
FROM recording_step_status_latest
WHERE zarr_path = '/path/to/recording_training.zarr'
ORDER BY step_name, dataset_id;
"
```

Expected after consolidation: exactly one active `datasets` row for the path,
no dependent rows referencing the duplicate ID, and all current
`recording_step_status_latest` rows under the canonical `:z...` ID.

After backup and review, apply the generic merge:

```bash
scripts/py -m fisheye.registry.dedupe --registry "$REG" --apply
```

Apply mode repoints non-conflicting dependent rows, drops duplicate dependent
rows that would collide with already-present canonical rows, deletes lineage
self-edges, and then removes the duplicate `datasets` rows.

## 8) Phase 2 Subject/Dish/Cross Backfill (Optional)

Preview without writes:

```bash
scripts/py -m fisheye.registry.maintenance \
  --registry "$REG" \
  --backfill-subject-dish-cross \
  --dry-run
```

Then apply:

```bash
scripts/py -m fisheye.registry.maintenance \
  --registry "$REG" \
  --backfill-subject-dish-cross
```

## 9) Phase 6 Subjects Backfill + Query View (Optional)

Preview without writes:

```bash
scripts/py -m fisheye.registry.maintenance \
  --registry "$REG" \
  --backfill-subjects \
  --dry-run
```

Then apply:

```bash
scripts/py -m fisheye.registry.maintenance \
  --registry "$REG" \
  --backfill-subjects
```

Verify normalized query view (cross/genotype + DPF filters):

```bash
sqlite3 -header -column "$REG" "
  SELECT DISTINCT recording_id
  FROM recording_subject_overview
  WHERE cross_id = :cross_id;
"
```

```bash
sqlite3 -header -column "$REG" "
  SELECT DISTINCT recording_id
  FROM recording_subject_overview
  WHERE dpf_at_acquisition = :dpf
    AND genotype = :genotype;
"
```

## Useful Diagnostics

```bash
sqlite3 -header -column "$REG" \
  "SELECT COUNT(*) AS lineage_edges FROM dataset_lineage_current;"
```

```bash
sqlite3 -header -column "$REG" \
  "SELECT set_id, dataset_ids_json FROM training_sets ORDER BY set_id;"
```

```bash
sqlite3 -header -column "$REG" \
  "SELECT dataset_id, session_uuid, artifact_kind, zarr_path
   FROM datasets
   WHERE artifact_kind='source_recording'
   ORDER BY session_uuid, dataset_id
   LIMIT 100;"
```

## Dataset ID Re-key Runbook (Phases A-D)

Use this when `dataset_id` values must be decoupled from `session_uuid`
(for example, to support multiple source Zarrs per recording).

Set your registry path once:

```bash
REG=/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite
```

### Dry-run checklist (no writes)

Run this checklist before any mutation:

```bash
# 1) Confirm current integrity baseline
scripts/py -m fisheye.registry.maintenance \
  --registry "$REG" \
  --check-integrity \
  --list-limit 100

# 2) Snapshot high-level dataset identity state
sqlite3 -header -column "$REG" "
  SELECT COUNT(*) AS datasets_total FROM datasets;
  SELECT artifact_kind, COUNT(*) AS n
  FROM datasets
  GROUP BY artifact_kind
  ORDER BY artifact_kind;
  SELECT COUNT(*) AS ids_equal_session_uuid
  FROM datasets
  WHERE session_uuid IS NOT NULL AND dataset_id=session_uuid;
"

# 3) Confirm no unresolved training-set members right now
sqlite3 -header -column "$REG" "
  WITH ids AS (
    SELECT ts.set_id, je.value AS id_value
    FROM training_sets ts, json_each(ts.dataset_ids_json) je
  ),
  src_map AS (
    SELECT session_uuid, MIN(dataset_id) AS dataset_id
    FROM datasets
    WHERE artifact_kind='source_recording' AND session_uuid IS NOT NULL
    GROUP BY session_uuid
  )
  SELECT COUNT(*) AS unresolved_training_set_members
  FROM ids i
  LEFT JOIN datasets d ON d.dataset_id=i.id_value
  LEFT JOIN src_map s ON s.session_uuid=i.id_value
  WHERE d.dataset_id IS NULL AND s.dataset_id IS NULL;
"

# 4) Preview JSON remap behavior (read-only)
scripts/py -m fisheye.registry.maintenance \
  --registry "$REG" \
  --remap-training-set-dataset-ids \
  --dry-run
```

Proceed to Phase A only if:
- integrity passes,
- unresolved training-set members = 0 (or you understand/accept known exceptions),
- dry-run remap output matches expected dataset ID migration behavior.

Optional: capture dry-run evidence to timestamped logs:

```bash
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="/tmp/registry_rekey_preflight_${TS}"
mkdir -p "$LOG_DIR"

scripts/py -m fisheye.registry.maintenance \
  --registry "$REG" \
  --check-integrity \
  --list-limit 100 | tee "$LOG_DIR/01_check_integrity.txt"

sqlite3 -header -column "$REG" "
  SELECT COUNT(*) AS datasets_total FROM datasets;
  SELECT artifact_kind, COUNT(*) AS n
  FROM datasets
  GROUP BY artifact_kind
  ORDER BY artifact_kind;
  SELECT COUNT(*) AS ids_equal_session_uuid
  FROM datasets
  WHERE session_uuid IS NOT NULL AND dataset_id=session_uuid;
" | tee "$LOG_DIR/02_dataset_identity_snapshot.txt"

sqlite3 -header -column "$REG" "
  WITH ids AS (
    SELECT ts.set_id, je.value AS id_value
    FROM training_sets ts, json_each(ts.dataset_ids_json) je
  ),
  src_map AS (
    SELECT session_uuid, MIN(dataset_id) AS dataset_id
    FROM datasets
    WHERE artifact_kind='source_recording' AND session_uuid IS NOT NULL
    GROUP BY session_uuid
  )
  SELECT COUNT(*) AS unresolved_training_set_members
  FROM ids i
  LEFT JOIN datasets d ON d.dataset_id=i.id_value
  LEFT JOIN src_map s ON s.session_uuid=i.id_value
  WHERE d.dataset_id IS NULL AND s.dataset_id IS NULL;
" | tee "$LOG_DIR/03_unresolved_members.txt"

scripts/py -m fisheye.registry.maintenance \
  --registry "$REG" \
  --remap-training-set-dataset-ids \
  --dry-run | tee "$LOG_DIR/04_remap_dry_run.txt"

echo "Saved preflight logs to: $LOG_DIR"
```

### Phase A: Preflight + safety snapshot

```bash
sqlite3 "$REG" ".backup ${REG%.sqlite}.pre_rekey.sqlite"
```

```bash
sqlite3 -header -column "$REG" "
  SELECT COUNT(*) AS datasets_total FROM datasets;
  SELECT artifact_kind, COUNT(*) AS n
  FROM datasets
  GROUP BY artifact_kind
  ORDER BY artifact_kind;
  SELECT COUNT(*) AS ids_equal_session_uuid
  FROM datasets
  WHERE session_uuid IS NOT NULL AND dataset_id=session_uuid;
"
```

```bash
sqlite3 -header -column "$REG" "
  WITH ids AS (
    SELECT ts.set_id, je.value AS id_value
    FROM training_sets ts, json_each(ts.dataset_ids_json) je
  ),
  src_map AS (
    SELECT session_uuid, MIN(dataset_id) AS dataset_id
    FROM datasets
    WHERE artifact_kind='source_recording' AND session_uuid IS NOT NULL
    GROUP BY session_uuid
  )
  SELECT COUNT(*) AS unresolved_training_set_members
  FROM ids i
  LEFT JOIN datasets d ON d.dataset_id=i.id_value
  LEFT JOIN src_map s ON s.session_uuid=i.id_value
  WHERE d.dataset_id IS NULL AND s.dataset_id IS NULL;
"
```

### Phase B: Build deterministic remap table

```bash
sqlite3 "$REG" "
  CREATE TABLE IF NOT EXISTS dataset_id_remap (
    old_dataset_id TEXT PRIMARY KEY,
    new_dataset_id TEXT NOT NULL UNIQUE,
    reason TEXT,
    created_utc TEXT NOT NULL
  );
  DELETE FROM dataset_id_remap;
  INSERT INTO dataset_id_remap(old_dataset_id, new_dataset_id, reason, created_utc)
  SELECT
    d.dataset_id,
    d.session_uuid || ':z' || substr(d.path_hash, 1, 12),
    'source_recording_rekey',
    datetime('now')
  FROM datasets d
  WHERE d.artifact_kind='source_recording'
    AND d.session_uuid IS NOT NULL
    AND d.dataset_id=d.session_uuid;
"
```

```bash
sqlite3 -header -column "$REG" "
  SELECT COUNT(*) AS remap_rows FROM dataset_id_remap;
  SELECT COUNT(*) AS duplicate_new_ids
  FROM (
    SELECT new_dataset_id, COUNT(*) AS n
    FROM dataset_id_remap
    GROUP BY new_dataset_id
    HAVING COUNT(*) > 1
  );
  SELECT old_dataset_id, new_dataset_id
  FROM dataset_id_remap
  ORDER BY old_dataset_id
  LIMIT 20;
"
```

### Phase C: Transaction-safe FK rewrite

Notes:
- Run only the table updates that exist in your DB.
- If `training_set_datasets` is absent in your schema, skip that update.

```bash
sqlite3 -header -column "$REG" "
  SELECT name
  FROM sqlite_master
  WHERE type='table'
    AND name IN (
      'provenance',
      'training_set_datasets',
      'keypoint_quality',
      'detect_quality',
      'dataset_lineage'
    )
  ORDER BY name;
"
```

```bash
sqlite3 "$REG" "
  PRAGMA foreign_keys=OFF;
  BEGIN;

  UPDATE datasets
  SET dataset_id = (
    SELECT r.new_dataset_id
    FROM dataset_id_remap r
    WHERE r.old_dataset_id = datasets.dataset_id
  )
  WHERE dataset_id IN (SELECT old_dataset_id FROM dataset_id_remap);

  UPDATE provenance
  SET dataset_id = (
    SELECT r.new_dataset_id
    FROM dataset_id_remap r
    WHERE r.old_dataset_id = provenance.dataset_id
  )
  WHERE dataset_id IN (SELECT old_dataset_id FROM dataset_id_remap);

  UPDATE keypoint_quality
  SET dataset_id = (
    SELECT r.new_dataset_id
    FROM dataset_id_remap r
    WHERE r.old_dataset_id = keypoint_quality.dataset_id
  )
  WHERE dataset_id IN (SELECT old_dataset_id FROM dataset_id_remap);

  UPDATE detect_quality
  SET dataset_id = (
    SELECT r.new_dataset_id
    FROM dataset_id_remap r
    WHERE r.old_dataset_id = detect_quality.dataset_id
  )
  WHERE dataset_id IN (SELECT old_dataset_id FROM dataset_id_remap);

  UPDATE dataset_lineage
  SET parent_dataset_id = (
    SELECT r.new_dataset_id
    FROM dataset_id_remap r
    WHERE r.old_dataset_id = dataset_lineage.parent_dataset_id
  )
  WHERE parent_dataset_id IN (SELECT old_dataset_id FROM dataset_id_remap);

  UPDATE dataset_lineage
  SET child_dataset_id = (
    SELECT r.new_dataset_id
    FROM dataset_id_remap r
    WHERE r.old_dataset_id = dataset_lineage.child_dataset_id
  )
  WHERE child_dataset_id IN (SELECT old_dataset_id FROM dataset_id_remap);

  COMMIT;
  PRAGMA foreign_keys=ON;
"
```

Run the runtime-bound complete SQLite acceptance gate after the transaction:

```bash
scripts/py -m fisheye.utils.registry_integrity \
  --registry "$REG" \
  --result-json /path/to/registry_integrity.json
```

Do not use a separately installed `sqlite3` binary as acceptance evidence; it
may embed a different SQLite engine from the one used by Palette writers.

### Phase D: Rewrite `training_sets.dataset_ids_json`

Preferred (built-in helper):

```bash
scripts/py -m fisheye.registry.maintenance \
  --registry "$REG" \
  --remap-training-set-dataset-ids \
  --dry-run
```

```bash
scripts/py -m fisheye.registry.maintenance \
  --registry "$REG" \
  --remap-training-set-dataset-ids
```

SQL equivalent:

```bash
sqlite3 "$REG" "
  BEGIN;
  WITH mapped AS (
    SELECT
      ts.set_id,
      CAST(je.key AS INTEGER) AS ord,
      COALESCE(r.new_dataset_id, je.value) AS new_id
    FROM training_sets ts
    JOIN json_each(ts.dataset_ids_json) je
    LEFT JOIN dataset_id_remap r ON r.old_dataset_id = je.value
  ),
  rebuilt AS (
    SELECT
      set_id,
      json_group_array(new_id) AS new_dataset_ids_json
    FROM (
      SELECT set_id, ord, new_id
      FROM mapped
      ORDER BY set_id, ord
    )
    GROUP BY set_id
  )
  UPDATE training_sets
  SET dataset_ids_json = (
    SELECT r.new_dataset_ids_json
    FROM rebuilt r
    WHERE r.set_id = training_sets.set_id
  );
  COMMIT;
"
```

Final validation:

```bash
scripts/py -m fisheye.registry.maintenance \
  --registry "$REG" \
  --backfill-dataset-lineage
```

```bash
scripts/py -m fisheye.registry.maintenance \
  --registry "$REG" \
  --check-integrity \
  --list-limit 100
```

## Operator Checklist: First Real Migration After Bootstrap

Use this when introducing the first non-noop schema migration after the current
baseline (`v1` initial schema, `v2` reserved noop).

### Preflight (required)

```bash
REG=/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite
```

```bash
# 1) Back up DB
sqlite3 "$REG" ".backup ${REG%.sqlite}.pre_migration.sqlite"

# 2) Verify baseline integrity
scripts/py -m fisheye.registry.maintenance \
  --registry "$REG" \
  --check-integrity \
  --list-limit 100

# 3) Record current schema versions
sqlite3 -header -column "$REG" "SELECT version, name, applied_utc FROM schema_version ORDER BY version;"
sqlite3 -header -column "$REG" "PRAGMA user_version;"
```

Go/no-go:
- Proceed only if integrity passes.
- Proceed only if backup exists and is readable.

### Apply migration

Migration application is automatic on registry open.
Run any registry command that opens the DB:

```bash
scripts/py -m fisheye.registry.maintenance \
  --registry "$REG" \
  --check-integrity \
  --list-limit 100
```

### Post-apply validation

```bash
# 1) Confirm schema_version advanced exactly as expected
sqlite3 -header -column "$REG" "SELECT version, name, applied_utc FROM schema_version ORDER BY version;"
sqlite3 -header -column "$REG" "PRAGMA user_version;"

# 2) Confirm integrity still passes
scripts/py -m fisheye.registry.maintenance \
  --registry "$REG" \
  --check-integrity \
  --list-limit 100

# 3) Optional smoke checks for key views
scripts/py -m fisheye.utils.check_training_registry \
  --registry "$REG" \
  --all \
  --limit 5
```

### Rollback

If migration fails or integrity regresses:

```bash
cp "${REG%.sqlite}.pre_migration.sqlite" "$REG"
scripts/py -m fisheye.registry.maintenance \
  --registry "$REG" \
  --check-integrity \
  --list-limit 100
```

Expected rollback result:
- DB returns to pre-migration state.
- Integrity matches preflight baseline.

## Completion Note (2026-02-09)

This runbook was executed end-to-end on live registry:

- Registry backup created before mutation:
  - `/nvme1/palette_registry.pre_drop_legacy_ids_20260209T065013Z.sqlite`
- Legacy duplicate source rows were remapped and removed.
- Post-migration checks:
  - `legacy_dupe_rows = 0` for `artifact_kind='source_recording' AND dataset_id=session_uuid`
  - no active duplicate `zarr_path` rows
  - `scripts/py -m fisheye.registry.maintenance --check-integrity` passed
  - `scripts/py -m fisheye.utils.registry_query --zarr-use training` returned deduplicated results

Follow-up guardrail applied in code:

- `Registry._resolve_effective_dataset_id()` now prefers canonical source IDs
  (`{session_uuid}:z<path-hash>`) for recording-source artifacts.
- Verified by rescanning `/nvme1/recordings`: no legacy IDs were recreated.

## Runtime Status Guardrail Note (2026-06-25)

The same canonical-ID rule now applies to live stage completion via
`emit_stage_completion()`. This closes the failure mode where runtime stage
writers inserted `recording_step_status` under `dataset_id=session_uuid` while
full scans and targeted refreshes used `dataset_id=session_uuid:z<path-hash>`.

RedScare training-zarr repair validated the operator pattern:

- canonical dataset:
  `2026-06-23T16-01-09Z_arena_1:z92f469b75d66`
- duplicate bare dataset:
  `2026-06-23T16-01-09Z_arena_1`
- all current step-status rows, status history, and keypoint-performance rows
  were consolidated under the canonical ID
- the duplicate dataset row was removed after verifying zero remaining
  dependent references
