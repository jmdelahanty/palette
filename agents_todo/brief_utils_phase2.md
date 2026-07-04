# Brief: utils Phase 2 — move the misfiled libraries down

**To:** provenance-finalization agent (next assignment)
**From:** commander session, 2026-07-04
**Read first:** `docs/utils_reorganization_strategy.md` (whole doc; you are executing "Phase 2"), `HANDOFF_2026-07-04.md` (operating notes section).

## Status of your last slice — merged, thank you

Your provenance finalization branch is merged into `sun` at `c0e111b` with the full
non-GPU suite green on the merged tree (3320 passed, 2 skipped, 2 deselected). The gate,
opt-in rollout, and shim structure all verified as specified. Two stale items from your
report, for your model of the world: the cluster checkout has since moved past
`257af7d` (another agent owns the origin push + cluster pull — do not touch either), and
`origin/sun` lags local `sun`; **local `sun` is ground truth**.

## Housekeeping before you start

From the main checkout (`~/gitrepos/palette`):
1. `git worktree remove /tmp/palette-provenance-finalization-enforcement`
2. `git branch -d agent/provenance-finalization-enforcement agent/provenance-finalization-design` (both fully merged; `-d` will confirm)
3. Create a fresh worktree on a new branch `agent/utils-phase2` **from current local `sun`**
   (must be `c0e111b` or later — you need your own `shared/system_metadata.py` merge).
   `sun` moves between turns: re-check fast-forward cleanliness against `sun` before you
   declare done, and rebase at each checkpoint if it moved.

## The task

Execute **Phase 2** of `docs/utils_reorganization_strategy.md`: move the ~8 misfiled
library modules out of `fisheye.utils` into their owning layers, with one-release
re-export shims. This severs ~120 upward edges and is the prerequisite for the
import-linter contract forbidding `fisheye.utils` (later slice — not yours).

### Scope, in order

1. **Finish `system.py`** (you started this — highest edge count, 75 upward edges):
   - 44 files still import via `fisheye.utils.system`. Retarget them to
     `fisheye.shared.system_metadata`. Per-file, not blind-sed.
   - The shim still *defines* `build_invocation_record` locally (~82-line file). Move it
     into `shared/system_metadata.py`; shim becomes pure re-export.
2. **Move to `shared/`** with same-shape re-export shims:
   `zarr_io` (23 edges), `zarr_metadata`, `calibration`, `encoder_tags`,
   `recording_preflight`, `import_video_metadata`, `zarr_recording_context`.
3. **`metadata.py`** (9 edges) is domain-flavored — the strategy doc says decide
   per-symbol between `shared/` and `io/`. Make the call, record the reasoning in your
   report.
4. **Model resolvers** (`resolve_detect_model`, `resolve_subject_mask_model`) →
   `registry/model_resolution.py`, promoting `_load_candidates` / `_load_target_profile` /
   `_resolve_recording_id` to public names in `registry`.
5. **Promote the ~10 borrowed `_private` symbols** (identified in the strategy doc's
   import-graph analysis) to public APIs in their owning layers, and retarget borrowers.

### Explicitly out of scope
- Deleting the shims (one-release policy; a later slice deletes them).
- The import-linter contracts / `__main__` predicate CI (Phase 0/forcing-function work).
- Runner moves into `apps/` (Phase 5, gated on other work).
- Any Layer-3 sediment deletion (backfills/migrations).
- Anything touching `origin/sun`, the cluster checkout, or other agents' worktrees.

## Constraints

- **Pure relocation — zero behavior change.** No signature changes, no "while I'm here"
  fixes. If you find a live bug during the moves, report it; don't fix it in this slice.
- One commit per module move (move + shim + retargeted importers together), so each is
  independently revertible.
- Watch for import cycles the moves can *create*: `shared/` may import nothing above it.
  If a module you're moving imports upward, that's a finding — stop and report rather
  than forcing it.

## Validation bar (all required before "done")

- `git diff --check` clean; `py_compile` clean.
- Focused tests for every touched surface.
- **Full non-GPU suite** (`-m "not gpu"`) green on your branch tip, rebased on current
  `sun`. Baseline: 3320 passed / 2 skipped / 2 deselected. Env:
  `~/miniconda3/envs/palette-py311/bin/python` (conda; do NOT use uv or create a `.venv`).
- Zero remaining `fisheye.utils.<moved-module>` imports under `src/` and `tests/` except
  the shims themselves (grep proof in your report).

## Reporting format (same as your last report — it was good)

Branch, commits, what landed, per-module edge counts severed, the `metadata.py`
per-symbol decision record, validation results, and any discrepancies you found between
the strategy doc's counts and reality. Push the agent branch to origin; do not push
`sun`. Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
