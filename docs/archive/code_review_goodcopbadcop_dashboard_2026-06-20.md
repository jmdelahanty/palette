# Code Review — GoodCopBadCop Interactive Dashboard + Component-Scoped RLE Refresh

- **Date:** 2026-06-20
- **Branch:** `sun`
- **Scope:** working-tree changes (uncommitted) plus new untracked files
- **Method:** high-effort multi-angle review (line-by-line, removed-behavior, cross-file, reuse, simplification, efficiency, altitude, conventions) with per-finding verification against the actual code paths.

## Files Under Review

Modified (working tree):

- `src/fisheye/analysis/chaser_distance_runs.py` — writes the new GoodCopBadCop interactive dashboard spec artifact
- `src/fisheye/shared/mask_store.py` — extracts `_write_encoded_component_rle_groups`; adds component-scoped RLE refresh
- `src/fisheye/utils/materialize_refined_subject_mask_store.py` — `--components` flag plumbing
- docs (`goodcopbadcop_analysis_surfaces.md`, mask/subject-mask contracts, migration checklist), `AGENTS.md`, tests

New (untracked):

- `src/fisheye/visualization/goodcopbadcop_interactive.py` — dashboard spec builder + discovery
- `apps/marimo/goodcopbadcop_explorer.py`, `apps/marimo/palette_explorer.py`
- `apps/marimo/components/{common,goodcopbadcop_chaser,provenance,registry,static_artifacts,__init__}.py`
- `tests/unit/fisheye/test_goodcopbadcop_interactive.py`, `test_marimo_palette_explorer_components.py`

## Findings

Ranked most-severe first. Correctness bugs outrank cleanup/altitude.

### 1. `--no-png` silently suppresses the interactive dashboard spec — CONFIRMED

- **File:** `src/fisheye/analysis/chaser_distance_runs.py:1073`
- The `if write_interactive_spec:` block is nested inside the `if write_png:` block (opened at line 1009).
- **Impact:** Running `chaser_distance_runs --apply --no-png` writes no `goodcopbadcop_chaser_dashboard_interactive` artifact even though `write_interactive_spec` defaults to `True` and a separate `--no-interactive-spec` flag exists for that purpose. The marimo explorer then finds no dashboard for the run.
- **Fix:** De-indent the `if write_interactive_spec:` block one level so it runs independently of `write_png`.

### 2. Whole-store stale marker never cleared by per-component refresh — CONFIRMED

- **File:** `src/fisheye/shared/mask_store.py:641`
- `mark_mask_rle_stale_attrs` writes `mask_rle_stale_component_names = []` when `updated_components is None` (its default). In `_clear_mask_rle_stale_for_refreshed_components`, the empty-list branch only clears when `refreshed == all_names`.
- **Impact:** If an edit marks the whole store stale with an empty component list, refreshing components one at a time (`--refresh-rle --components <name>`) never satisfies `refreshed == all_names`, so the stale flag is never cleared. The store stays permanently stale and `open_mask_store(prefer="rle")` raises `mask_rle is marked stale`, forcing dense fallback on a store whose RLE is actually current.
- **Fix:** When the recorded stale scope is the whole store (empty list), accumulate refreshed components and clear once all channels have been refreshed, or treat an empty list as "all components" for the remaining-scope computation.
- **Note:** Not covered by the two new tests (they exercise explicit component lists only).

### 3. marimo `palette_explorer.py` absolute imports break under `marimo run` — PLAUSIBLE (launch-time)

- **File:** `apps/marimo/palette_explorer.py:26`
- Uses `from apps.marimo.components... import ...`, but there is no `apps/__init__.py` or `apps/marimo/__init__.py`, and `apps` is not an installed package (only `fisheye` is).
- **Impact:** `scripts/py -m marimo run apps/marimo/palette_explorer.py` can raise `ModuleNotFoundError: No module named 'apps'` because marimo executes the script with its own directory on `sys.path`, not the repo root. The sibling `goodcopbadcop_explorer.py` is unaffected (it imports only `fisheye.*`). Pytest masks this because it runs from the repo root.
- **Fix:** Either add the `__init__.py` files and guarantee repo root on `sys.path`, or move the shared component code under the installed `fisheye` package and import it the same way `goodcopbadcop_explorer.py` does.

### 4. Empty occupancy cube → IndexError — PLAUSIBLE (edge crash)

- **File:** `apps/marimo/components/goodcopbadcop_chaser.py:289`
- A `(0, H, W)` detection-occupancy cube passes the `is None` guard (line 258) and the `len(windows_df)` guard (line 260, which checks the chaser run's windows, not the cube). Line 268 clamps `window_idx` to `max(0, min(0, -1)) = 0`; line 289 then indexes `occupancy_normalized[0]` on a zero-row array.
- **Impact:** Occupancy panel crashes instead of rendering the "no heatmap" message when the matched detection_occupancy run materialized zero windows.
- **Fix:** Guard on `occupancy_normalized.shape[0] == 0` alongside the `is None` check.

### 5. Cross-run positional window-index assumption — PLAUSIBLE (silent mislabel)

- **File:** `apps/marimo/components/goodcopbadcop_chaser.py:263`
- The selected window's **row position** within the chaser run's `windows_df` is used directly as the index into the detection-occupancy run's cube (lines 263-268), with no validation that the two runs share window ordering.
- **Impact:** If the occupancy cube materialized a different subset/order of windows than the chaser run's `epoch_summary`, the displayed heatmap belongs to a different epoch than its title label — silent, no error.
- **Fix:** Resolve the occupancy row by matching the occupancy cube's own window ids, not by chaser-window position; or validate equal ordering when building the spec.

### 6. Component-scoped encode re-reads each dense chunk per component — efficiency

- **File:** `src/fisheye/shared/mask_store.py:596`
- `_encode_dense_selected_component_rle_serial` issues `dense_masks[start:stop, component_idx]` separately per selected component per row-chunk. For a 4-D `masks_roi` whose row-axis chunk spans all channels, each underlying chunk is read and decompressed N times for N components.
- **Fix:** Read each chunk once with `dense_masks[start:stop, list(indices)]` and index channels in memory.

### 7. Scoped refresh validates the whole store — efficiency

- **File:** `src/fisheye/shared/mask_store.py:1130`
- After a component-scoped refresh, validation runs with the full `names` list, re-reading and re-scanning every untouched component's RLE arrays (counts/indptr/present/area_px/bbox_xyxy), defeating the purpose of scoping.
- **Fix:** Validate only the refreshed components.

### 8. `_artifact_signature` is a third canonical-JSON SHA implementation — reuse/correctness

- **File:** `src/fisheye/analysis/chaser_distance_runs.py:64`
- Hand-rolls `json_attr_safe -> json.dumps(sort_keys, separators) -> sha256` with `json.dumps` defaults (`ensure_ascii=True`, `default=str`), whereas `plot_artifacts` computes the stored `content_sha256` via `strict_json_dumps` (`allow_nan=False`).
- **Impact:** A non-ASCII recording id or a NaN value yields a stored content hash that no longer equals this signature, defeating any skip-if-unchanged / dedup comparison.
- **Fix:** Reuse `fisheye.shared.json_safety.strict_json_dumps` and the existing sha helper instead of a third serializer.

### 9. `goodcopbadcop_explorer.py` duplicates extracted panel rendering — altitude/duplication

- **File:** `apps/marimo/goodcopbadcop_explorer.py:49` (and the distance/arena/occupancy/debug cells)
- Inlines epoch overlays, distance figure, arena/occupancy heatmaps, and debug tables that already exist in `components/goodcopbadcop_chaser.py` and are consumed by `palette_explorer.py`. Two near-identical implementations of every panel.
- **Impact:** Any future change to a panel must be made twice or the two apps silently diverge.
- **Fix:** Make `goodcopbadcop_explorer.py` a thin wrapper over the shared components (pre-filtering discovery to the chaser renderer).

### 10. Component-scoped refresh is a parallel bolt-on — altitude

- **File:** `src/fisheye/shared/mask_store.py:1088`
- A second serial-only encoder plus an early-return block duplicate the full path's attr-stamping, validation, and `rle_refreshed` return-dict shape (lines 1088-1170 vs 1172-1206). The two return dicts already differ (`refreshed_component_names` exists only on the scoped path), forcing callers to branch on `refresh_scope`.
- **Fix:** Generalize `write_component_rle_mask_store_from_dense` to accept a component-index subset so both full and scoped refresh share one encode/write/stamp/validate sequence.

## Notes / Non-Issues

- **Conventions:** No `AGENTS.md` rule is violated. The `eye_left` / `eye_right` references are subject-mask **components** (of `refined_subject_masks_runs`), which the "Subject Mask Direction" section explicitly permits — not `eye_masks_runs`.
- **mask_store refactor parity:** The extraction of `_write_encoded_component_rle_groups` preserves all attrs, chunk computations, and the five per-component arrays; the full-write path is actually safer than before (it deletes `mask_rle` wholesale first).
- **Dropped candidate (refuted):** The epoch dropdown was flagged for label collisions, but the label includes start/end times to 0.1 s (`goodcopbadcop_chaser.py:133`), so distinct epochs get distinct keys.
- **Lower-confidence candidates not promoted:** non-atomic delete-then-recreate of a component group on mid-write failure; orphan component groups when the component set shrinks/renames (fixed component set in practice); shape-drift hard-error instead of self-heal; scoped path skips the dense round-trip validation the full path runs by default. Worth keeping in mind if the scoped path is extended.
