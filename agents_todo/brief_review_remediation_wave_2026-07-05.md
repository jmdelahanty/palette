# Brief: review-remediation wave — close the 07-01 review's remaining partials

**From:** commander session, 2026-07-05
**Context:** A 5-agent verification pass (2026-07-05) checked every finding of
`docs/diagnostics/codebase_review_2026-07-01.md` against `sun` HEAD `8af2443`. Findings
1, 2, 4 are closed; 3, 5, 6, 7 are partial with well-scoped remainders. This brief
parcels the remainders into **four independent slices (A–D)**, one agent each. They are
file-disjoint except where noted; dispatch in parallel.

**Also dispatch-ready, not covered here:** `agents_todo/brief_inventory_accessor_verb_port.md`
(the one open queue brief — module landed at `ca4444a`, accessor/api/verb port never started).
That makes five agents total for the wave.

**Read first (all agents):** `HANDOFF_2026-07-04.md` operating-notes section ONLY (its
status sections are stale; the review doc's remediation deltas are current). Ground
rules that bit previous agents:
- Local `sun` is ground truth; `origin/sun` is stale and unreachable from the sandbox.
  Do not push `sun`; push only your agent branch if origin is reachable, otherwise report.
- Work in a fresh worktree on `agent/<slice-name>` branched from **current** `sun`;
  `sun` moves between turns — re-check clean fast-forward before declaring done.
- Env: `~/miniconda3/envs/palette-py311/bin/python` (conda). Never uv, never a `.venv`.
- Full non-GPU suite before "done": `PYTHONPATH=src ... -m pytest tests -m "not gpu" -q -n 16`.
  Recount the baseline from current `sun` first (collect-only today: 3373 selected, 0 errors).
- Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

## Slice A — forcing function: import-linter layering + size ratchet in CI

**Why first-priority:** flagged twice in repo docs (action plan item 2 tail; the 07-05
proportion finding calls it the highest-leverage next move). Nothing enforces layering
or size today; `registry/db.py` is the one review metric moving the wrong way
(8,186 → 8,518 since 07-01) because nothing stops it.

**Read first:** `docs/utils_reorganization_strategy.md` (Phase 0 / forcing-function
sections), `docs/diagnostics/codebase_review_2026-07-01.md` §"Proportion finding",
`.github/workflows/ci.yml`, `pyproject.toml`.

Scope, in order:
1. **import-linter config** in `pyproject.toml` (`[tool.importlinter]`) with:
   - a layers contract: `shared` imports nothing above it (shared → nothing in
     registry/tracking/detection/labeling/apps/cli/utils);
   - a forbidden contract for the **already-severed** `fisheye.utils` modules
     (the Phase 2 moves: `system`, `zarr_io`, `zarr_metadata`, `calibration`,
     `encoder_tags`, `recording_preflight`, `import_video_metadata`, model resolvers) —
     forbid importing them from `src/` EXCEPT the shim files themselves. Do NOT forbid
     all of `fisheye.utils`; unmoved modules and one-release shims are still live.
   - Run it; whatever violations exist on current `sun`, either fix trivial ones
     (import-path retargets only, zero behavior change) or register them in an explicit
     ignore list with a `# RATCHET:` comment. The contract must pass on your branch.
2. **File-size ratchet:** a small `scripts/check_file_size_ratchet.py` + committed
   baseline (JSON: path → line count) covering at minimum `registry/db.py`,
   `registry/maintenance.py`, `labeling/web.py`, `cli/palette.py`. Failing condition:
   any listed file GROWS past baseline + 200 lines; shrinking auto-tightens the baseline
   (script rewrites it, CI checks it's committed). This freezes growth without demanding
   decomposition now.
3. **Wire both into `ci.yml`** as steps before the test run. Keep runtime trivial.

Constraints: zero behavior change to `src/` beyond import-path retargets. If a layering
violation is load-bearing (a real upward dependency, not a stale import), ratchet-ignore
it and report — do not restructure code in this slice.

Validation: import-linter green, ratchet green, full non-GPU suite green, CI workflow
syntax validated (`act` not available — eyeball + yaml parse). Report: contract list,
ignore-list contents with reasons, baseline table.

## Slice B — stage-graph: put the launcher on the catalog (Finding 5 remainder)

**The gap:** three hand-maintained stage graphs still exist. Pipeline↔catalog drift is
frozen by `tests/unit/fisheye/test_stage_catalog_drift.py` (explicit override table with
INTENT comments), but `cli/interactive_launcher.py:65` `STAGE_INFO[...].requires` has
**no edge-level guard** — `test_stage_catalog.py:183` checks only that stage *names*
resolve. Launcher edges can silently diverge; today they already differ from the catalog
on `detect`/`crop`/`keypoints`.

**Read first:** `docs/stage_catalog_reality_gaps.md`, `test_stage_catalog_drift.py`
(the pattern to mirror), `registry/stage_catalog.py` (`dependency_map()`, `canonical_stage_id`).

Scope:
1. Preferred: derive launcher `requires` from `stage_catalog.dependency_map()` at
   import time, mapping through `canonical_stage_id` (launcher vocabulary differs:
   `import`/`refine`/`keypoints_refine`/`track`). Keep launcher-only presentation
   fields hand-maintained; only the edges move.
2. If derivation is not clean (launcher edges intentionally differ for UX ordering —
   check git blame / gap doc before assuming), fall back to a
   `KNOWN_LAUNCHER_DEPENDENCY_OVERRIDES` drift test mirroring the pipeline one, with
   INTENT comments per divergent edge. Either way: after this slice, no stage-graph
   copy can silently drift.
3. Do NOT touch `core/pipeline.py` edges — legacy, already frozen, out of scope.

Validation: launcher still renders/launches (drive it: it's a Textual TUI — see the
sandbox asyncio caveat; a smoke import + unit test of the derived edges is the bar, not
an interactive session). Focused stage-catalog + launcher tests, then full suite.

## Slice C — provenance Phase 5: default epoch bump (Finding 3 remainder)

**State:** Slice 2 landed (merge `c0e111b`): fail-closed gate at `mark_run_complete`,
`COMPLETION_EPOCH_REQUIRE_PROVENANCE = 2` (`shared/zarr_run_completion.py:33`), five
production writers stamp epoch 2 (detect_yolo, detect_keypoints_yolo, crop,
run_sam_subject_masks, run_subject_mask_batch_pipeline). Remaining per the design doc's
own Phase 5: `require_runs_parent(...)` still stamps **epoch 1 by default**
(`zarr_run_completion.py:101-104`), so unpatched stages finalize without provenance.

**Read first:** `docs/archive/provenance_finalization_enforcement_design.md` (Phase 5 +
Acceptance Shape sections are the spec; do not re-litigate locked decisions),
`docs/archive/provenance_enforcement_roadmap.md`.

Scope, in order:
1. **Census first (the gate):** enumerate every `require_runs_parent` /
   run-finalizing writer in `src/` and classify: stamps epoch 2 / synthesizes
   provenance but stamps epoch 1 / no provenance at all. The design doc gates the bump
   on this census being green — if a live production writer would start failing at
   `mark_run_complete`, patch it to synthesize provenance (same pattern as `9a8100c`)
   BEFORE the bump, or report why it can't.
2. **Bump the default epoch to 2 for NEW parents.** Existing stores stay grandfathered
   (epoch read from the store, never rewritten) — that's a locked design decision.
3. Tests: new-parent-defaults-to-2; grandfathered epoch-1 store still finalizes;
   unprovenance'd writer against a new store fails loudly with the designed message.

**Explicitly out of scope:** content hashes for input/model artifacts (the SAM3
checkpoint canary) — that is the next slice after this, separately briefed. Do not
start it.

Checkpoint: report the census BEFORE making the bump commit; the bump is cheap, a wrong
census is not.

## Slice D — hygiene sweep: the persistent small untouched items

Small, verified-still-present items from the review's lower-severity list. One agent,
one commit per item, each independently revertible.

1. **TensorRT regexes** `training/export_shared.py:70,73,78`: raw strings contain `\\s`/`\\d`
   (literal backslash) so they can never match and `tensorrt_version` stays None. Fix the
   escapes; add a unit test with a captured trtexec output sample (there are trtexec
   artifacts referenced in the review — synthesize a representative line if none is
   committed).
2. **Stale `shared/__init__.py:10`** — `__all__ = ["ZARR_SCHEMA"]` names a symbol the
   module doesn't define. Fix to reflect reality (likely empty or the actual re-exports).
3. **Empty `src/fisheye/io/` package** — only `__init__.py`. Delete it unless something
   imports it (grep proof in report). If `utils_reorganization_strategy.md` reserves it
   as a Phase-2+ target (the metadata.py per-symbol decision mentioned `io/`), leave it
   and say so instead.
4. **Bare `except: pass` at `core/pipeline.py:368`** — module is legacy but the swallow
   was review-flagged. Minimal fix: `except Exception` + `logger.warning` with context.
   No other pipeline.py changes.
5. **Root scratch drawer (~26 tracked root `.py`/`.md`, incl. `CRITICAL_REIMPORT_NEEDED.md`
   from May, `speed_test*.py`, `verify_enum_format.py`, root `diagnostics/`):** produce a
   classification manifest (dead / superseded-by-doc-X / still-referenced), `git rm` only
   the ones that are (a) unimported from `src/` and `tests/`, (b) unreferenced from live
   docs, and (c) obviously spent one-off scripts. Anything ambiguous: leave, flag in the
   manifest. Everything is git-recoverable; err toward deleting the obvious, flagging
   the rest.
6. **README.md** (7-line stub): expand to a real minimal README — what Palette is, the
   conda env setup, `pip install -e ".[dev]"`, the palette CLI entry, pointer to
   `docs/`. One page, no marketing. **LICENSE: do NOT add one** — license choice is a
   maintainer/HHMI decision; flag it in your report as the remaining shareability gap.
   Leave `setup.py` alone (pyproject is authoritative).

Merge-order note: Slice D item 6 and Slice A both touch repo-root config surfaces
(README vs pyproject/ci.yml) — no file overlap, but whoever merges second rebases.

## Reporting (all slices)

Branch, commits, what landed vs. what was skipped and why, grep/test proof for every
claim, validation counts (full-suite numbers, rebased on current `sun`), and any
discrepancy between this brief's premises and reality — **verify the cited line numbers
and counts before acting on them; re-locate by content if drifted.** If a premise is
wrong (e.g., a "still present" item was fixed since 07-05), report and stop that item
rather than improvising.
