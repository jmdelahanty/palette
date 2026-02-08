# Registry TUI TODO

## Goals
- Provide a single terminal UI to browse, filter, and maintain the training registry.
- Replace repetitive command memorization with discoverable workflows.
- Make dataset/set/run/export lineage visible from any selected row.

## Non-Goals (v1)
- No silent destructive actions.
- No direct filesystem deletion workflows by default.
- No dependency on web UI.

## Core Requirements
- Show all primary registry tables/views in one app.
- Show links for highlighted row:
  - dataset -> set(s) -> run(s) -> onnx/trt
  - run -> set -> datasets -> exports
  - quality rows/current + gate metadata
- Support global and per-view filters.
- Support maintenance actions with dry-run preview and explicit confirmation.
- Display action output and status in-app.

## Proposed Layout
- Left pane: view selector (`datasets`, `training_sets`, `training_runs`, `onnx_models`, `tensorrt_models`, `keypoint_quality_current`, `detect_quality_current`, `pose_skeleton_specs`, raw table browser).
- Center pane: paginated table for active view.
- Right pane: relationship details for highlighted row.
- Bottom pane: filter bar + action bar + event log.

## Safety Model
- All write actions require:
  - preview (`dry-run`) first
  - explicit confirmation prompt
  - clear target summary (counts, IDs)
- Dangerous actions (delete run/set):
  - second confirmation
  - show derived rows affected
  - show filesystem impact separately (if enabled later)

## Implementation Phases

### Phase 1: Read-Only Browser (v1 milestone)
- Build app shell and keyboard navigation.
- Implement core views and column sorting.
- Add filter model:
  - status, date range, dish/canvas/rig, review/use, quality thresholds.
- Add relationship panel for highlighted rows.
- Add copyable “equivalent CLI/SQL” preview for current query.

Acceptance criteria:
- Can browse all target views without leaving TUI.
- Can apply/clear filters and see row counts update.
- Highlighting any row updates lineage panel correctly.

### Phase 2: Maintenance Actions
- Add actions:
  - backfill/refresh keypoint quality
  - backfill/refresh detect quality
  - integrity checks
  - delete run-id
  - delete set-id (optional include linked runs)
- Wire actions to existing maintenance code paths.
- Show dry-run report and apply report in log pane.

Acceptance criteria:
- Each action supports dry-run and apply.
- Confirmations block accidental writes.
- Action output includes counts and IDs.

### Phase 3: Power Features
- Saved filter presets.
- “Only issues” dashboards (stale/divergent/excluded).
- Cross-method fallback visibility widgets.
- Optional exported report snapshots.

Acceptance criteria:
- Presets persist and reload.
- Issue dashboards match existing CLI diagnostics.

## Data/Query Layer Tasks
- Add reusable query helpers for lineage joins.
- Add normalized filter parser shared by views.
- Add pagination helpers to prevent full-table blocking.
- Add typed row adapters for each view.

## Testing Plan
- Unit tests for query helpers and filter parser.
- TUI state tests:
  - view switching
  - filter apply/reset
  - row highlight -> relationship panel
- Maintenance action tests:
  - dry-run path
  - confirmation path
  - failure/error propagation

## Suggested File Structure
- `src/fisheye/utils/registry_tui.py` (entrypoint)
- `src/fisheye/registry/tui/` (views, widgets, controllers)
- `tests/unit/fisheye/test_registry_tui_*.py`

## Initial Command Contract
- `scripts/py -m fisheye.utils.registry_tui --registry /path/to/palette_registry.sqlite`
- Optional:
  - `--view`
  - `--filter`
  - `--readonly`

## Open Questions
- Should v1 require `textual`, or support a plain-rich fallback?
- Should raw SQL execution be allowed in-app (likely no for v1)?
- Should delete-file flows remain CLI-only indefinitely?
