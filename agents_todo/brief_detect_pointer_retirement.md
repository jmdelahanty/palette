# Brief: retire the `detect_review_status_latest` writer surface

**Status: READY (hold cleared 2026-07-04, commander).** Both hold conditions merged to
`sun`: the `system.py` checkpoint (3f1f1f6) and the `metadata.py` move (72dcd68).
Verified: no model-resolver imports in the files below, so this slice is parallel-safe
with utils Phase 2 items 4–5. Branch from `sun` at 72dcd68 or later.

**From:** commander session, 2026-07-04
**Read first:** `docs/diagnostics/refined_detect_pointer_census_2026-07-04.md` (lines
77–80 scope this exact follow-up), `docs/run_resolution_semantics.md` §retire/subsume,
`HANDOFF_2026-07-04.md` operating notes only (status sections stale).

## Context — why this is now writer-only

A 2026-07-04 read-only audit (opus, against `sun`) established:
- The read side is DONE. `shared/run_resolution.py:294–306` consults the legacy pointer
  only as a fallback below `authoritative_run` (refined-detect parents only, completed
  runs only, `fallback_used=True`). Crop resolves via AUTHORITATIVE with no direct
  pointer reads. **Keep the bridge — it is out of scope.**
- `backfill_detect_review_status` already retired as a writer (routes through `approve`).
- Other readers (`labeling/task_generation.py:99`, the three batch pipelines'
  `_latest_group_name` chains) treat the pointer as one candidate among several with
  directory-listing fallback — none depend on freshness. Removing those read-fallback
  entries is a SEPARATE cleanup, not this slice.

## Scope — the writers

Branch from current `sun` (re-verify lines; `sun` moves between turns):

1. **`tune/detect_review_backend.py:501`** — redundant dual-write. This path already
   calls `approve(...)` (~line 416) and `set_authoritative_run` (~482). Delete the
   legacy write; update `tests/.../test_detect_review_backend.py:251` which asserts it.
2. **`tune/detect_review.py:1430`** (TUI review) — pure legacy writer, no authoritative
   call in the file. Convert to the same `approve`/`set_authoritative_run` path the
   backend uses, then drop the pointer write.
   **Added scope (2026-07-04, curation instance-key slice):** this file also calls
   `write_curated_refined_detect_surfaces` WITHOUT instance-key arguments, so its
   curated writes are stamped `instance_key_status="missing"` with a warning. While
   you are converting this file, wire the instance-key inputs through (mirror how
   `update_curated_refined_detect_rows` → `write_curated_refined_detect_root` does
   it) so TUI-curated rowsets carry keys. One agent, one pass over the file.
3. **`utils/accept_detect_review.py:329`** and **`utils/set_detect_review_status.py:130`**
   — `--no-latest`-guarded CLIs, pure legacy writers. Route approval through `approve`;
   stop writing the pointer. Update `--no-latest` help/flag semantics accordingly (a
   flag that gates a write that no longer exists should not silently no-op — remove or
   repurpose, your call, but say which in the report).
4. **Migrations `utils/migrate_legacy_detect_labels.py:524` and
   `utils/migrate_refined_detect_sparse.py:664`** — Layer-3 sediment. Do NOT convert.
   Leave the writes in place and add a one-line comment marking them legacy-pointer
   writes retained for archive repair. (Deleting these files is a future sediment
   slice's call, not yours.)

## Explicitly OUT of scope

- **`tune/detect_training_promotion_backend.py:652`** — writes the pointer to a
  *pending* run; it is a pending-marker, not an approval signal. The maintainer owns
  the decision on what replaces that signal. Do not touch it; do note in your report
  anything you learn about its consumers.
- The `run_resolution.py` legacy bridge (read compatibility stays for existing stores).
- Reader fallback chains in task_generation / batch pipelines.
- Schema/doc declarative mentions (`shared/zarr/schema.py:80` "(optional)" is accurate).

## Constraints

- Approval semantics must remain fail-closed — converting a writer must not weaken any
  check the `approve` path enforces.
- Existing stores keep their old pointers; nothing rewrites history. New sign-offs
  simply stop minting the legacy pointer.
- One commit per writer file; each independently revertible.
- If you find a consumer that DOES depend on pointer freshness (the audit found none),
  stop and report rather than working around it.

## Validation bar

- `git diff --check` + `py_compile` clean.
- Focused tests for every touched file, plus the review-approval fail-closed tests.
- Full non-GPU suite green on branch tip rebased on current `sun`
  (`~/miniconda3/envs/palette-py311/bin/python -m pytest tests -m "not gpu"`;
  baseline 3346 passed / 2 skipped / 2 deselected as of 182d1be — recount, it grows).
- Grep proof in report: remaining `detect_review_status_latest` assignment sites are
  exactly the two annotated migrations + nothing else in `src/`.

## Reporting

Branch `agent/detect-pointer-writer-retirement`, pushed to origin (never push `sun`).
Report: commits, per-writer conversion notes, the `--no-latest` decision, promotion-
backend consumer notes, validation results.
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
