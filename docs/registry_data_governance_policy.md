# Registry Data Governance Policy

## Scope

This policy defines:
- immutable vs mutable registry fields
- deletion policy for source vs derived artifacts

It applies to the palette registry database and its filesystem-linked artifacts.

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
