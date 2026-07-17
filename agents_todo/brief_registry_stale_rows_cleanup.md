# Brief: registry stale-row cleanup tool (`agent/registry-stale-rows-cleanup`)

**From:** commander session, 2026-07-06
**Status: READY.** One agent, one slice, single checkpoint at the end.
**Do NOT push or merge — the commander verifies and merges.**
**This brief authorizes ZERO writes to the real registry.** The agent ships a
dry-run-default tool and runs ONLY the dry run against the real registry; the
maintainer runs `--execute` from their own session.
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

**Read first:** `docs/archive/HANDOFF_2026-07-05.md` operating notes;
`docs/diagnostics/detect_review_pointer_census_2026-07-06.md` (the row inventory
this cleans); `docs/registry_data_governance_policy.md` (authority model — registry
rows "may be repaired, rebuilt, normalized, or refreshed as indexes", and the
locator model that makes MISSING ≠ DELETABLE); `docs/registry_schema_reference.md`
(the dependent-table map); `docs/registry_repair_playbook.md` (house repair style).
Ground rules: local `sun` is ground truth; fresh worktree on
`agent/registry-stale-rows-cleanup` from CURRENT `sun`; env
`~/miniconda3/envs/palette-py311/bin/python`; sync code only. Gates: import-linter
via `scripts/py -m importlinter.cli --config pyproject.toml`,
`python scripts/check_file_size_ratchet.py` (**`registry/db.py` 8,518 and
`registry/maintenance.py` 9,733 are ratcheted — prefer a new module or accept the
maintenance ratchet math before writing there**), `git diff --check`, `py_compile`,
focused tests, full suite `PYTHONPATH=src ... -m pytest tests -m "not gpu" -q -n 16`.
Baseline: **3,454 passed / 2 skipped** at merge `8fa894b` (main-worktree count with
untracked parallel tests; recount in your worktree and report both).

## Why

The pointer census enumerated 2,280 registry store rows; **1,975 point at paths
that no longer exist and were never real recordings**: 1,938 under
`/tmp/pytest-of-delahantyj`, 27 under `/tmp`, 10 under `/home`. ~87% of the
registry's dataset rows are detritus from unguarded test/dev registration (the
guard closing that hole is the sibling slice
`brief_registry_temp_store_guard.md`; the two slices are independent — do not
coordinate branches).

## The safety line that shapes everything

**Prune ONLY rows whose `datasets.zarr_path` is under a temp root** (resolved
prefix under `tempfile.gettempdir()`, `/tmp`, `/var/tmp`, `/dev/shm`).
A missing path outside temp roots is NOT deletable — the governance policy's
locator model explicitly anticipates archives moving between hot/network/cold
storage, so "doesn't exist right now" can mean "unmounted", and `/groups` was
partially enumerable in the census. The 10 `/home` rows are NOT pruned by default:
list them in the dry-run output under a separate `NEEDS-MAINTAINER-REVIEW` section
with their dataset_ids; support `--include-dataset-id <id>` (repeatable, exact
match) so the maintainer can opt them in one by one at execute time.

## Scope

1. **Tool** — a maintenance-style CLI (follow the `registry/maintenance.py`
   argparse/reporting house style; new module
   `registry/prune_stale_datasets.py` is fine given the ratchets) with:
   - `--registry <path>` required; `--dry-run` the DEFAULT action.
   - Dry run prints, per temp-root class (`pytest-tmp`, `tmp`, `var-tmp`,
     `dev-shm`): dataset_id count, and per dependent table the exact row counts
     that WOULD be deleted; plus the `NEEDS-MAINTAINER-REVIEW` `/home` section.
     Machine-readable JSON alongside the human table (`--json <path>`).
   - `--execute` requires `--backup <path>`: the tool first copies the sqlite
     file via the sqlite3 backup API (`Connection.backup`, not `shutil.copy` — the
     registry lives on NFS with WAL off; take a `BEGIN IMMEDIATE` transaction for
     the whole prune so a concurrent writer blocks rather than interleaves, and
     keep `busy_timeout` at the house 30s).
   - Deletion order derived from `docs/registry_schema_reference.md`: build the
     dependent-table map (every table keyed by `dataset_id`, and `recordings`
     rows only when NO remaining dataset references that `recording_id`) in code
     as an explicit list with a test asserting it covers every table in the live
     schema that has a `dataset_id` column (query `sqlite_master` + `PRAGMA
     table_info` — the assertion is the drift guard, so a future table can't be
     silently orphaned).
   - Post-execute verification inside the tool: `PRAGMA integrity_check`,
     `PRAGMA foreign_key_check`, and a re-scan proving zero remaining temp-root
     `datasets.zarr_path` rows (excluding any not-opted-in `/home` rows).
2. **Tests** on a fixture registry built through the real `Registry` API
   (register fixture stores under a fake temp root via monkeypatched temp-root
   list): dry-run counts match; execute deletes exactly the classified rows and
   their dependents; a non-temp missing-path row SURVIVES; `/home`-class row
   survives unless `--include-dataset-id`; backup file is a valid sqlite DB with
   the pre-prune row counts; integrity/foreign-key checks pass post-prune.
3. **Real-registry dry run** (read-only by construction): run it against the real
   registry, commit the JSON output as
   `docs/diagnostics/registry_stale_rows_dryrun_2026-07-06.json` and summarize the
   counts in the report. Expect ≈1,965 temp-root dataset rows (census said 1,938
   + 27; `/var/tmp`/`/dev/shm` unmeasured) and 10 `/home` review rows — if the
   numbers disagree with the census materially, that is a FINDING to report, not
   to reconcile silently.
4. One paragraph added to `docs/registry_repair_playbook.md` describing the tool
   and when to reach for it.

## Out of scope — hard boundaries
- Running `--execute` against the real registry (maintainer-only, from their own
  session, after reviewing the dry run).
- Deleting or touching any row outside the temp-root classes and explicitly
  opted-in `/home` dataset_ids. Never delete on "path missing" alone.
- The registration guard (sibling brief), the reconcile-consolidation design
  (`register_from_root` collapse — separate maintainer-reviewed slice), schema
  changes, VACUUM (note in the playbook paragraph that the maintainer may want
  `VACUUM` after a ~87% row prune, but the tool must not auto-vacuum an NFS
  database).
- Any zarr store reads/writes — this slice is registry-only.

## Reporting
Branch + SHA, dry-run JSON path + headline counts vs census expectations, the
dependent-table map with its drift-guard test, fixture-test evidence for every
safety property above, suite counts (recount → final), premise discrepancies.
