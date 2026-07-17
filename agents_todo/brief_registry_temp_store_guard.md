# Brief: registry temp-store registration guard (`agent/registry-temp-store-guard`)

**From:** commander session, 2026-07-06
**Status: READY.** One agent, one slice, single checkpoint at the end.
**Do NOT push or merge — the commander verifies and merges.**
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

**Read first:** `docs/archive/HANDOFF_2026-07-05.md` operating notes;
`docs/diagnostics/detect_review_pointer_census_2026-07-06.md` (the motivating
evidence); `docs/registry_data_governance_policy.md` (authority model). Ground
rules: local `sun` is ground truth; fresh worktree on
`agent/registry-temp-store-guard` from CURRENT `sun`; env
`~/miniconda3/envs/palette-py311/bin/python` (conda; never uv, never `.venv`);
sync code only. Gates: import-linter via
`scripts/py -m importlinter.cli --config pyproject.toml`,
`python scripts/check_file_size_ratchet.py` (**note: `registry/db.py` is ratcheted
at 8,518 lines — if the guard would push it over, put the guard helper in a small
new module and import it**), `git diff --check`, `py_compile`, focused tests, full
suite `PYTHONPATH=src ... -m pytest tests -m "not gpu" -q -n 16`.
Baseline: **3,454 passed / 2 skipped** at merge `8fa894b` (main-worktree count,
includes parallel agents' untracked tests — your fresh-worktree recount will be
lower; recount and report both).

## Why

The pointer census found **1,938 rows in the real registry pointing at
`/tmp/pytest-of-delahantyj` store paths** (plus 27 `/tmp`, 10 `/home`): test/dev
runs have been registering temp stores into the production registry for months.
A cleanup slice removes the existing rows; THIS slice makes the detritus unable
to regrow.

## The policy to implement

Reject a dataset registration when **the store path is under a temp root but the
registry database itself is not**. Both directions of legitimacy stay intact:

- Registry under a temp root (the entire test suite's pattern: tmp registry + tmp
  stores) → **allowed, unconditionally.** The existing suite passing unchanged is
  the regression proof.
- Real registry + real store path → allowed (obviously).
- Real registry + temp store path → **`ValueError` with a loud, specific message**
  (name the store path, the registry path, and the env override), unless
  `PALETTE_REGISTRY_ALLOW_TEMP_STORES=1` is set.

Temp roots: `tempfile.gettempdir()` plus the literal `/tmp`, `/var/tmp`,
`/dev/shm` — resolve symlinks before comparing (`Path.resolve()`), compare with
`is_relative_to`, and treat "under a temp root" as a prefix relationship on the
RESOLVED path. Do NOT block `/home` paths in this slice — worktree stores are a
policy question, not mechanically decidable; see Reporting.

## Scope

1. **Find the funnel first, then guard the minimal chokepoint(s).** Registration
   flows through `Registry.register_from_root` →
   `_register_from_root_in_transaction` → `upsert_dataset(dataset_id, zarr_path=…)`
   (`registry/db.py:6609` area, re-locate by content), and there is also
   `reconcile_dataset_from_root` (used by `registry/maintenance.py`,
   `registry/inline_refresh.py`, `shared/subject_mask_profile.py`,
   `diagnostics/prepare_detect_training.py`). Census every write path that can
   introduce a NEW `datasets.zarr_path` value (grep `upsert_dataset`,
   `zarr_path` INSERT/UPDATE sites in `registry/db.py`) and put the check at the
   fewest places that cover all of them — ideally one helper called from
   `upsert_dataset`. Include the census (call-site table) in the report.
2. The check compares the resolved store path against temp roots and the resolved
   registry sqlite path (`self.path` / connection origin — find the attribute the
   Registry object actually holds) against the same roots.
3. **Tests, both directions:** (a) tmp registry + tmp store → registers fine
   (exists implicitly, but add one explicit test so the contract is pinned);
   (b) non-temp registry + tmp store → raises, message includes the override env
   var (simulate the non-temp registry with a monkeypatched temp-root list or a
   registry path fixture outside `tempfile.gettempdir()` — do not write outside
   the test sandbox to do it); (c) override env var set → allowed.
4. Document the policy in `docs/registry_data_governance_policy.md` — one short
   subsection under the authority model ("index hygiene: temp-store registrations
   are refused"), matching the doc's voice.

## Out of scope — hard boundaries
- No pruning/deleting of existing rows (that is the cleanup slice
  `brief_registry_stale_rows_cleanup.md`).
- No sanctioned-roots allowlist (`/nvme1/...`-only style) — too aggressive to land
  without maintainer policy. If the funnel census shows `/home` registrations came
  from an identifiable code path, REPORT it with a recommendation; don't block it.
- No schema changes, no registry writes in tests against any non-temp registry,
  no changes to `RegistryPaths` resolution order.

## Reporting
Branch + SHA, the funnel census table (every zarr_path write site → guarded via
which chokepoint), test evidence for all three directions, where the `/home`
registrations likely originate (best-effort code-path attribution, grep only),
suite counts (recount → final), premise discrepancies.
