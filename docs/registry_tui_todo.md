# Registry TUI TODO

## Vision

A single terminal UI to browse, filter, and operate the full pipeline registry.
Replace repetitive CLI commands with discoverable, scrollable views where you can
select a dataset or recording, inspect its status, and spawn pipeline jobs directly.

## Current State (2026-02-28)

**Implemented:** `src/fisheye/utils/registry_tui.py` (single-file, 413 lines, Textual 6.1.0)

Phase 1 read-only browser is complete:
- [x] 3-pane layout: view selector (left), data table (center), details/lineage (right)
- [x] 11 curated views: `datasets`, `dataset_lineage_current`, `training_sets`,
      `training_runs`, `training_models`, `onnx_models`, `tensorrt_models`,
      `keypoint_quality_current`, `detect_quality_current`, `pose_skeleton_specs`,
      plus dynamic raw-table browser
- [x] Keyboard navigation: `q` quit, `r` refresh, `/` filter, `n`/`p` cycle views, `c` clear
- [x] Substring filter across visible columns
- [x] Relationship/details panel with smart lineage hints (dataset→sets→runs→models)
- [x] Zebra-striped row cursor, 500-row default limit

**Not yet implemented:**
- No `recording_step_status` view (the primary pipeline progress view)
- No interactive column sorting (only SQL ORDER BY)
- No write/maintenance actions
- No job spawning
- No provenance, performance, or data profile views
- No modular file structure or unit tests

## Non-Goals

- No silent destructive actions (all writes require dry-run + confirm).
- No direct filesystem deletion by default.
- No web UI dependency.

---

## Phase 2: Pipeline Status & Enhanced Browsing

Priority: make the TUI the primary tool for understanding what work has been done
and what's stale across all recordings.

### New views

- [ ] Add `recording_step_status_wide` view (pivoted: one row per dataset, columns per step).
  - This is the most important missing view — shows pipeline progress at a glance.
  - Color-code status cells: green=ok, red=missing, gray=absent/na, yellow=error.
- [ ] Add `recording_step_overview` view (step status with recording/dataset context).
- [ ] Add `provenance` view with provenance completeness indicators.
- [ ] Add `recording_overview` view (recording metadata: session, type, rig, arena, camera).
- [ ] Add performance views: `detect_performance_latest`, `keypoint_performance_latest`,
      `eye_mask_performance_latest`.
- [ ] Add data profile views: `keypoint_data_profile_latest`, `detection_data_profile_latest`,
      `eye_mask_data_profile_latest`.

### Table enhancements

- [ ] Interactive column sorting (click header or keybinding to sort asc/desc).
- [ ] Column-aware filtering (e.g., `status:missing`, `step:keypoints`, `dataset_id:cedar`).
- [ ] Scrollable detail panel (for rows with large JSON fields like `details_json`).
- [ ] Show row count and current cursor position in status bar.
- [ ] Copyable "equivalent SQL" preview for current view + filter.

### Step dependency visualization

- [ ] Show the `STEP_DEPENDENTS` graph in the details panel when a step row is highlighted.
  - Highlight which steps are downstream of the selected step.
  - Show cascade invalidation impact ("re-running detect would invalidate 9 steps").

### Acceptance criteria

- Can see full pipeline status for any recording without leaving the TUI.
- Stale/missing steps are immediately visible via color coding.
- Can filter to "only recordings with missing keypoints" or similar.

---

## Phase 3: Maintenance Actions

Wire the 60+ existing maintenance CLI actions into the TUI with safe dry-run previews.

### Quality/performance backfill actions

- [ ] Backfill keypoint quality (`--backfill-keypoint-quality` / `--refresh-keypoint-quality`).
- [ ] Backfill detect quality (`--backfill-detect-quality` / `--refresh-detect-quality`).
- [ ] Backfill eye mask quality (`--backfill-eye-mask-quality` / `--refresh-eye-mask-quality`).
- [ ] Backfill crop quality (`--backfill-crop-quality` / `--refresh-crop-quality`).
- [ ] Backfill performance metrics (detect, keypoint, eye mask).
- [ ] Backfill data profiles (keypoint, detection, eye mask).

### Recording step status actions

- [ ] Backfill recording step status (`--backfill-recording-step-status`).
  - Allow scoping to selected dataset(s) from the current view.
- [ ] Manual step status override (set a step to ok/missing/na for selected dataset).
  - Uses `upsert_recording_step_status` with source="tui_manual".

### Training management actions

- [ ] Delete training run (`--delete-run-id`) with cascade preview.
- [ ] Delete training set (`--delete-set-id`) with linked-run warning.
- [ ] Prune failed runs (`--prune-failed-runs`).
- [ ] Reconcile stale in-progress runs (`--reconcile-in-progress-runs`).
- [ ] Prune empty sets (`--prune-empty-sets`).

### Registry health actions

- [ ] Integrity check (`--check-integrity`) with inline results.
- [ ] Reconcile missing datasets (`--reconcile-registry`).
- [ ] Vacuum (`--vacuum`) with size before/after display.

### Action UX pattern

All actions share the same flow:
1. User selects action from action bar or keybinding.
2. TUI shows scope (selected row? all visible rows? entire registry?).
3. Dry-run executes and results display in a log pane (RichLog widget).
4. User confirms to apply, or cancels.
5. Apply executes and log pane shows results + affected counts.

- [ ] Implement action bar widget with categorized action list.
- [ ] Implement dry-run → confirm → apply flow with Textual Screen/Modal.
- [ ] Implement RichLog output pane for action results.
- [ ] Scope actions to selected row(s) when applicable.

### Acceptance criteria

- Every action supports dry-run preview before apply.
- Destructive actions (delete run/set) require double confirmation and show affected rows.
- Action output is scrollable and copyable in the log pane.

---

## Phase 4: Job Spawning

Spawn pipeline jobs directly from the TUI for selected datasets/recordings.
The existing `interactive_launcher.py` already does this via `subprocess.Popen()` —
this phase integrates that capability into the registry TUI.

### Job types (scoped to selected dataset)

- [ ] Run detection (`python -m fisheye <zarr> --stages detect`).
- [ ] Run refinement (`python -m fisheye <zarr> --stages refine`).
- [ ] Run crop (`python -m fisheye <zarr> --stages crop`).
- [ ] Run keypoint inference (`python -m fisheye <zarr> --stages keypoints`).
- [ ] Run keypoint refinement (`python -m fisheye <zarr> --stages keypoints_refine`).
- [ ] Run eye mask inference (`python -m fisheye <zarr> --stages eye_masks`).
- [ ] Run eye mask refinement (`python -m fisheye <zarr> --stages refined_eye_masks`).
- [ ] Run ID assignment + tracking (`python -m fisheye <zarr> --stages assign_ids track`).
- [ ] Run full pipeline (all stages in dependency order).
- [ ] Run "fix stale" — automatically determine which stages need re-running from
      `recording_step_status` and execute only those.

### Job configuration

- [ ] Argument editor: show required + optional args for the selected job type.
  - Pre-fill from registry (zarr_path, model paths from `model_exports`, run names).
  - Allow override of key params: `--batch-size`, `--device`, `--scheduler`, `--num-workers`.
- [ ] Model selector: pick from registered models in `onnx_models`/`tensorrt_models`.
- [ ] Config file selector: pick or generate pipeline config.

### Job execution & monitoring

- [ ] Spawn job via `subprocess.Popen()` with stdout/stderr streaming to log pane.
- [ ] Show live job status (running/completed/failed) in status bar.
- [ ] Support multiple concurrent jobs with a job list view.
- [ ] Auto-refresh the registry view when a job completes (step status will update).
- [ ] Job history: keep a session log of spawned jobs and their outcomes.

### Batch operations

- [ ] Multi-select datasets (checkbox or range select) for batch job spawning.
- [ ] Sequential batch: run same pipeline stage across N selected datasets.
- [ ] Show batch progress (N/M complete, current dataset).

### Acceptance criteria

- Can select a dataset with missing keypoints and spawn keypoint inference in 3 keystrokes.
- Job output streams in real-time to the log pane.
- Registry views auto-refresh to show updated step status after job completion.
- Model paths are auto-resolved from registry (no manual path entry for registered models).

---

## Phase 5: Power Features

- [ ] Saved filter presets (persist across sessions).
- [ ] "Issues only" dashboard: show only datasets with stale/error/missing steps.
- [ ] Step dependency graph visualization (ASCII tree in details panel).
- [ ] Keyboard-driven row selection for bulk operations (shift+arrow, ctrl+a).
- [ ] Export current view to CSV/JSON from within TUI.
- [ ] Session restore: reopen with same view, filter, and cursor position.

---

## Architecture

### Current: single file

`src/fisheye/utils/registry_tui.py` — adequate for Phase 1, will need splitting for
Phases 3–4.

### Target: modular package

```
src/fisheye/registry/tui/
├── __init__.py          # re-export RegistryTUI
├── app.py               # main App class, layout, bindings
├── client.py            # RegistryClient (DB access layer)
├── views.py             # curated view definitions (SQL + column config)
├── widgets/
│   ├── details_panel.py # relationship/lineage panel
│   ├── action_bar.py    # action palette with categorized commands
│   ├── log_pane.py      # RichLog for action/job output
│   └── job_monitor.py   # job status tracker
├── actions/
│   ├── maintenance.py   # wrappers around maintenance.py functions
│   └── jobs.py          # subprocess spawning for pipeline stages
└── css/
    └── registry_tui.tcss  # extracted stylesheet
```

Migration: Phase 2 can stay single-file with incremental additions. Split into
package at Phase 3 when action/job code would make the single file unwieldy.

### Entry point (unchanged)

```bash
scripts/py -m fisheye.utils.registry_tui --registry /path/to/palette_registry.sqlite
```

### Testing plan

- [ ] Unit tests for `RegistryClient` query helpers (no TUI dependency).
- [ ] Unit tests for filter parser (column-aware syntax).
- [ ] Textual `pilot` tests for view switching, filter apply, row highlight.
- [ ] Action tests: dry-run paths, confirmation flow, error propagation.
- [ ] Job spawn tests: command construction, subprocess lifecycle.

---

## Resolved Questions

- **Textual required?** Yes — already shipped with Textual 6.1.0, no plain-rich fallback.
- **Raw SQL in-app?** No — curated views + column-aware filters cover the use cases.
- **Delete-file flows?** CLI-only for now. TUI actions operate on registry rows only.
  Can add `--delete-files` as an opt-in flag on delete actions later.
