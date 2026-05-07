# Registry Data Governance Policy

## Scope

This policy defines:
- immutable vs mutable registry fields
- deletion policy for source vs derived artifacts

It applies to the palette registry database and its filesystem-linked artifacts.

## Authority Model

The registry is the fast searchable index for available datasets, recording
metadata, run status, quality summaries, freshness/staleness, and operational
state. It is expected to make queries fast enough that analysis code does not
need to scan every Zarr archive for routine cohort construction.

The registry is not the scientific source of truth for raw/refined/derived
analysis values. Recording analysis Zarrs remain authoritative for
per-recording artifacts, virtual collection manifests freeze cross-recording
source selections, and Parquet/DuckDB exports are rebuildable products derived
from those manifests.

The registry is, however, the right place to maintain current and alternate
storage locators for datasets that move between hot compute storage, networked
storage, and cold archive/object storage. Immutable manifests should store the
stable dataset identity plus a `locator_at_selection` audit snapshot. They
should not duplicate every future alternate locator. When a source archive is
moved, update the registry locator state; the manifest remains scientifically
valid because its identity and source run selections did not change.

This means registry rows may be repaired, rebuilt, normalized, or refreshed as
indexes. Those changes must not silently change the meaning of an already
materialized analysis export, because exports should point to immutable
collection manifests with concrete source run IDs and source fingerprints.

## Immutable vs Mutable

### Immutable (must not be updated in normal operations)

Identity and provenance fields:
- `datasets.dataset_id`
- `datasets.session_uuid`
- `datasets.path_hash`
- `datasets.recording_id` for `artifact_kind='source_recording'`
- `dataset_lineage.*` edge identity:
  - `child_dataset_id`
  - `parent_dataset_id`
  - `relationship_type`
  - `source_set_id` (as originally recorded/backfilled)
- `training_runs.run_id`
- `training_sets.set_id`
- content hashes when present (`*_sha256`)

Rule:
- If an immutable value is wrong, create a corrective migration/repair entry path.
- Do not silently mutate immutable values during routine backfill/rescan.

### Mutable (can change as workflows progress)

Operational/state fields:
- `datasets.zarr_path` (path mobility / storage moves)
- `datasets.status` (`active`, `missing`, etc.)
- `datasets.last_seen_utc`
- `datasets.zarr_origin`, `datasets.zarr_use` (normalization/backfill refinement)
- quality tables current rows (`keypoint_quality`, `detect_quality`)
- run status/progress (`training_runs.status`)
- derived metadata and summary fields in `metadata_json` columns

Rule:
- Mutable updates must be idempotent and safe to replay.
- Maintenance commands should support `--dry-run` where destructive.

Current `zarr_use` semantics:
- Primary active uses: `training`, `analysis`
- `inference` is not currently used as a standalone dataset artifact in behavior workflows
  (inference outputs are kept inside analysis artifacts/runs)
- `export` is reserved for standalone collated/packaged outputs

## Deletion Policy

### Never delete by default

Source recording datasets:
- `datasets.artifact_kind='source_recording'`
- associated recording entities (`recordings`, `recording_artifacts`)

Default behavior:
- mark missing/stale state, do not hard-delete.

### Allowed deletions (explicit only)

Derived artifacts may be deleted with explicit commands:
- merged training datasets
- model exports and run-linked derived outputs
- training set / training run rows when user requests deletion

Requirements:
- command must be explicit (`--delete-run-id`, `--delete-set-id`, etc.)
- preview via dry-run where available
- filesystem delete must stay scoped to safe derived roots
- source recording paths must never be deleted by maintenance file-delete logic

### Cascades

Expected cascade behavior is allowed for dependent derived rows:
- deleting `training_runs` may cascade to `training_models`, `onnx_models`, `tensorrt_models`
- deleting derived dataset rows may cascade to lineage/quality rows

Source provenance should remain auditable after derived cleanup.

## Operational Guardrails

- Always create a DB backup before bulk mutation/deletion.
- Prefer:
  1. `--dry-run`
  2. apply
  3. `--check-integrity`
- For path/name migrations:
  1. rescan
  2. remap training-set dataset IDs
  3. backfill lineage
  4. integrity check

See `docs/registry_repair_playbook.md`.

## Scheduled Backups

Palette owns its registry backup script:

```bash
scripts/backup_palette_registry.sh
```

Default behavior:
- source registry: `/nvme1/palette_registry.sqlite`
- backup directory: `/groups/ahrens/ahrenslab/jeremy/zebrobot/backups`
- retention: delete `palette_registry_*.sqlite` files older than 7 days only
  after a new backup has been created and verified

Recommended cron entry:

```cron
0 2 * * * cd /home/delahantyj@hhmi.org/gitrepos/palette && scripts/backup_palette_registry.sh >> /home/delahantyj@hhmi.org/palette_registry_backup.log 2>&1
```

The backup script verifies that the source registry exists and is non-empty,
uses SQLite's `.backup` command, runs `PRAGMA quick_check` on the source and
backup, and refuses to report success for a missing or zero-byte backup.
