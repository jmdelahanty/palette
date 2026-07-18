# Brief: registry reconcile — root sweep verb (`agent/registry-reconcile-root-sweep`)

**From:** commander session, 2026-07-07
**Status: READY.** One agent, one slice, single checkpoint at the end.
**Do NOT push or merge — the commander verifies and merges.**
**Zero writes to the real registry:** the agent runs only the dry run against it;
the maintainer runs `--apply` from their own session.
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

**Read first:** `docs/archive/HANDOFF_2026-07-05.md` operating notes;
`docs/diagnostics/registry_reconcile_collapse_audit_2026-06-18.md` (the design
audit — NOTE it is partially stale, see "Landed since the audit" below);
`docs/diagnostics/detect_review_pointer_recensus_2026-07-07.md` (the motivating
drift evidence). Ground rules: local `sun` is ground truth; fresh worktree on
`agent/registry-reconcile-root-sweep` from CURRENT `sun`; env
`~/miniconda3/envs/palette-py311/bin/python`; sync code only; re-locate line
numbers by content. Gates: import-linter via
`scripts/py -m importlinter.cli --config pyproject.toml`,
`python scripts/check_file_size_ratchet.py` (**`registry/db.py` 8,518 and
`registry/maintenance.py` 9,733 are ratcheted — the sweep orchestrator goes in a
new module, e.g. `registry/reconcile_sweep.py`; only the thin CLI wiring may
touch `maintenance.py` if its ratchet math allows**), `git diff --check`,
`py_compile`, focused tests, full suite
`PYTHONPATH=src ... -m pytest tests -m "not gpu" -q -n 16`.
Baseline: **3,472 passed / 2 skipped** (main-worktree count incl. parallel
untracked tests; recount in your worktree and report both).

## Landed since the audit (verified 2026-07-07 — build on it, don't rebuild it)

The audit's "smallest first step" is DONE in parallel work:
- `Registry.reconcile_dataset_from_root(root, zarr_path, *, include_step_status)`
  (`db.py:6749` area) runs `register_from_root` (all ~10 zarr-derived tables) PLUS
  the detection, keypoint, and **subject-mask** data-profile extractors (the
  audit's "subject_mask has no profile path" finding is fixed) in one
  transaction, composing existing `replace_*` primitives.
- The standalone `sync_*_profile_registry` scripts are already deleted.
- CLI `--reconcile-dataset <zarr>` exists in `registry/maintenance.py`.
- Discovery layer exists: `scan_paths(registry, paths, recursive=True)`
  (`db.py:8420` area) finds zarr roots via `_find_zarr_roots`, registers each via
  `scan_zarr` (SHALLOW — `register_from_root` level, no profiles/step-status),
  then `reconcile_missing_datasets(scope_paths=…)` marks vanished paths missing.
- The temp-store guard is live at `upsert_dataset` — a sweep over real roots is
  unaffected; a sweep over a temp root against the real registry will (correctly)
  refuse.

## The gap this slice fills

There is no single verb for "make the registry converge to what is on disk under
these roots." Evidence of the cost: the 2026-07-07 re-census found **13
filesystem-only stores** (sickyfish training zarrs + the chunking canary) on
`/nvme1/recordings` that the registry has never indexed. Today fixing that means
hand-running `--reconcile-dataset` 13 times, and nothing sweeps for the next one.

## Scope

1. **Sweep orchestrator** in a new module (`registry/reconcile_sweep.py`):
   `reconcile_roots(registry, roots, *, include_step_status=True, apply=False)`:
   - Enumerate zarr roots under each given root (reuse `_find_zarr_roots` /
     `_is_zarr_root` — import, don't copy).
   - Classify each store against the registry: `new` (path not in `datasets`),
     `known` (registered), `unreadable` (open failed — record error, keep going).
   - In apply mode: run `reconcile_dataset_from_root` per readable store (new AND
     known — the whole point is convergence, and every write path it composes is
     idempotent DELETE-then-INSERT), then `reconcile_missing_datasets`
     scoped to the swept roots.
   - In dry-run mode (DEFAULT): no registry writes at all — open the registry
     read-only (reuse the `mode=ro` + `query_only` pattern from
     `registry/prune_stale_datasets.py`), report per-store classification and
     what apply WOULD do, including which registered-but-vanished rows would be
     marked missing.
   - Return a structured report; support `--json <path>` like the prune tool.
2. **CLI**: `--reconcile-root <path>` (repeatable) + `--apply` on the maintenance
   CLI if ratchet math allows, otherwise a `python -m fisheye.registry.reconcile_sweep`
   entrypoint — state which and why in the report.
3. **Tests** on fixture stores + fixture registry (built through the real
   `Registry` API, temp-root guard satisfied since both live under tmp):
   new-store discovered and fully reconciled in apply mode (profile tables
   populated, not just dataset row — assert at least one `replace_*`-fed table);
   known store refreshed idempotently (second apply is a no-op diff);
   vanished-path row marked missing; unreadable store is a report row, not a
   crash; dry run writes nothing (assert registry file content hash unchanged).
4. **Real-registry dry run** (read-only): sweep `/nvme1/recordings` and
   `/groups/johnson/johnsonlab/jeremy/recordings`, commit the JSON as
   `docs/diagnostics/registry_reconcile_sweep_dryrun_2026-07-07.json`. Expect the
   13 known filesystem-only stores to classify `new`; material disagreement with
   the re-census is a FINDING, not something to reconcile silently.
5. **Audit-doc addendum**: append a short dated "Landed state" section to
   `docs/diagnostics/registry_reconcile_collapse_audit_2026-06-18.md` recording
   what exists now (orchestrator, profile extractors incl. subject_mask, deleted
   sync scripts, this sweep) so the audit stops going stale silently.

## Out of scope — hard boundaries
- Running `--apply` against the real registry (maintainer-only, after dry-run
  review).
- Model-manifest reconcile (audit category C) — future slice.
- The eye-mask profile stack — it is a DELETION target on the deprecation path
  (eye masks are legacy); do not extend, wire in, or "fix" anything eye-mask.
- Zarr-attr backfills and read-only audit scripts (audit's "should NOT collapse"
  list stands).
- Any zarr store writes; any change to `reconcile_dataset_from_root` internals or
  the extractor contract; deleting any existing script (retirement of redundant
  scripts is a follow-up once the sweep is proven on real data).

## Reporting
Branch + SHA, dry-run JSON path + classification counts vs the re-census's 13,
CLI placement decision (maintenance vs module entrypoint) with ratchet numbers,
idempotency-test evidence, suite counts (recount → final), premise discrepancies.
