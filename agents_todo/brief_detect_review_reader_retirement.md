# Brief: detect-review legacy reader retirement (`agent/detect-review-reader-retirement`)

**From:** commander session, 2026-07-07
**Status: READY.** One agent, one slice, single checkpoint at the end.
**Do NOT push or merge — the commander verifies and merges.**
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

**Read first:** `HANDOFF_2026-07-05.md` operating notes;
`docs/diagnostics/detect_review_pointer_recensus_2026-07-07.md` (the gate
evidence). Ground rules: local `sun` is ground truth; fresh worktree on
`agent/detect-review-reader-retirement` from CURRENT `sun`; env
`~/miniconda3/envs/palette-py311/bin/python` (conda; never uv, never `.venv`);
sync code only; re-locate cited line numbers by content. Gates: import-linter via
`scripts/py -m importlinter.cli --config pyproject.toml`,
`python scripts/check_file_size_ratchet.py`, `git diff --check`, `py_compile`,
focused tests, full suite `PYTHONPATH=src ... -m pytest tests -m "not gpu" -q -n 16`.
Baseline: **3,472 passed / 2 skipped** (main-worktree count incl. parallel
untracked tests; your fresh-worktree recount will differ — recount and report both).

## Why the gate is open (do not re-litigate; do cite)

Maintainer executed the full repair sequence 2026-07-07: registry prune (2,054
stale rows deleted, integrity clean), `authoritative_run` backfill (118/118
applied, 0 errors), and a post-repair re-census over all 293 enumerable stores:
**BACKFILLABLE = 0, fallback-would-fire = 0, winner-would-change = 0** (the 10
AMBIGUOUS rows are unreadable `/home/.../in-memory.zarr` registry detritus with no
pointer relevance). Every real store now resolves through `authoritative_run` or
never used the legacy pointer at all. Removing the legacy readers is now a
measured zero-behavior-change on real data.

## Scope — pure removal, one commit per site

1. **`shared/run_resolution.py`** — delete the legacy fallback block in
   `AUTHORITATIVE` mode (the `_is_refined_detect_parent(...)` branch reading
   `LEGACY_DETECT_REVIEW_AUTHORITY_ATTR`, ~lines 293–306). Then delete
   `_is_refined_detect_parent` and `_REFINED_DETECT_PARENT_NAMES` if the fallback
   was their only consumer (verify by grep, state the proof), and delete the
   `LEGACY_DETECT_REVIEW_AUTHORITY_ATTR` constant — **but see item 5 first: the
   census tool imports it from this module.** The `fallback_used`/`source_attr`
   fields on `RunResolutionResult` STAY (the latest_complete fallback still uses
   them).
2. **`labeling/task_generation.py`** — candidate tuple
   `("detect_review_status_latest", "latest")` → `("latest",)`. Pure removal
   only: do NOT add `authoritative_run` to the candidate list in this slice —
   that is a semantic enhancement the census never measured. Put it in the report
   as a recommended follow-up instead.
3. **Batch pipelines ×3** (`utils/run_megabouts_batch_pipeline.py`,
   `utils/run_movement_bout_batch_pipeline.py`,
   `utils/run_subject_mask_batch_pipeline.py`) — drop the third key from
   `("latest", "latest_materialized", "detect_review_status_latest")`.
4. **`shared/zarr/schema.py`** — do NOT delete the attr description: the two
   sanctioned migration writers still stamp `detect_review_status_latest` on
   stores they migrate, so the attr still exists in the wild. Reword the
   description to mark it historical: written by migration tools for lineage,
   no longer consulted by any reader.
5. **Constant ownership move** — `diagnostics/detect_review_pointer_census.py`
   imports `LEGACY_DETECT_REVIEW_AUTHORITY_ATTR` from `run_resolution`. Define
   the literal in the census module itself (it is a historical-diagnostic tool
   and rightfully owns the legacy name now) and update
   `utils/backfill_detect_review_authoritative_run.py` only if its import chain
   breaks (it imports from the census module, so it should be untouched —
   verify). The census and backfill tools are NOT retired — they are the
   instruments that measured and executed this migration and must keep working
   for any future re-census.

## Tests (per site, one equivalence module or extend existing)

Fixture-store matrix for `run_resolution` AUTHORITATIVE mode, before/after
pinned by keeping the OLD fallback logic verbatim in the test as oracle where
cheap, or by direct assertions:
- `authoritative_run` present → identical result (authoritative wins, no change).
- Neither attr → identical result (latest_complete fallback, `fallback_used=True`).
- **Legacy attr ONLY (the retired case)** → assert the NEW sanctioned behavior
  explicitly: resolution falls through to latest_complete and `source_attr` is no
  longer `detect_review_status_latest`. This is the one intentional semantics
  change; the test documents it as deliberate (on all real stores latest agreed
  with legacy, so the resolved run is the same — say so in the test docstring).
- Batch-pipeline and task-generation key orders: winner unchanged on fixtures
  with and without a legacy attr present alongside `latest`.
- Census tool still runs against a fixture store after the constant move (its
  existing tests keep passing unmodified is the bar).

## Out of scope — hard boundaries
- The two sanctioned pointer WRITERS and the batch wrapper help text (they keep
  writing the legacy attr; stopping them is a separate maintainer decision —
  note it in the report).
- The census/backfill/prune tools beyond the constant-ownership move.
- Adding `authoritative_run` to task-generation or batch-pipeline candidate
  orders (report-only recommendation).
- Any store or registry writes; eye-mask paths; `_REFINED_DETECT_PARENT_NAMES`
  consumers outside the deleted fallback (if grep finds any, keep the constant
  and report).

## Reporting
Branch + SHA per site, grep-proof that `detect_review_status_latest` no longer
appears in `src/` outside the sanctioned writers + diagnostics/backfill tools +
schema's historical description, the `_is_refined_detect_parent` consumer proof,
suite counts (recount → final), premise discrepancies.
