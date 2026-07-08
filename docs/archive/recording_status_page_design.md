# Recording Status Page Design

Todo reference:
- `docs/recording_status_page_todo.md`
Deployment reference:
- `docs/recording_status_page_deployment.md`

## Goal

Provide a live, read-only web status page for pipeline progress across recordings,
using registry-backed step status as the single source of truth.

Primary user outcome:
- See a "wide" at-a-glance view of all recordings and steps (Import, Detect, Crop, Keypoints, Eye Masks, etc.).
- Filter quickly to missing/error/stale work.
- Drill into run/method/review details and status history per dataset/step.

## Why Now

- The registry already has normalized status writes (`upsert_recording_step_status`) and cascade invalidation.
- The schema already exposes the right views:
  - `recording_step_status_latest`
  - `recording_step_status_wide`
  - `recording_step_overview`
  - `recording_step_status_history`
- Current visibility is mostly CLI/TUI; a browser view is better for continuous monitoring and team sharing.

## Scope

In scope (MVP):
- Read-only web app.
- Live-ish updates via polling.
- Wide table + filters + details drawer.
- Per-row history timeline for step transitions.

Out of scope (MVP):
- Editing status rows from web.
- Triggering jobs from web.
- New registry schema changes.

## Existing Data Contract (No New Model)

Use existing registry semantics directly:
- Status enum: `ok | missing | absent | na | error`
- Latest snapshot: `recording_step_status` + `recording_step_status_latest`
- Audit trail: `recording_step_status_history`
- Operator-friendly wide render: `recording_step_status_wide`

This keeps parity with CLI/TUI behavior and avoids model drift.

## Proposed Architecture

### Backend

Use a dedicated package for this feature (recommended):
- `src/fisheye/status_page/`

Suggested structure:
- `src/fisheye/status_page/__init__.py`
- `src/fisheye/status_page/app.py` (server wiring)
- `src/fisheye/status_page/api.py` (endpoints)
- `src/fisheye/status_page/query.py` (registry SQL access)
- `src/fisheye/status_page/models.py` (response DTO helpers)
- `src/fisheye/status_page/static/` (HTML/CSS/JS)

Keep a thin CLI launcher for operator ergonomics:
- `src/fisheye/utils/serve_recording_status_page.py`

Suggested stack:
- FastAPI + Uvicorn (or Flask if we want minimal deps and no async features).
- SQLite read-only queries against the existing registry DB.

Run command (proposed):
- `scripts/py -m fisheye.utils.serve_recording_status_page --registry /nvme1/palette_registry.sqlite --host 127.0.0.1 --port 8765`

### Frontend

Single-page HTML/JS served by backend:
- Top summary cards (recordings total, rows with blocking statuses, stale/missing counts).
- Main table from `recording_step_status_wide`.
- Color-coded step cells (match TUI conventions).
- Filters:
  - Recording / camera text search
  - `Use` (`analysis`/`training`)
  - status toggles (`missing`, `error`, `ok`)
  - step-specific filters (e.g. "Keypoints contains MISS")
- Details drawer on row click:
  - underlying dataset id + zarr path/use/status
  - step-level metadata (`run_name`, `method`, `coverage_pct`, `review_status_json`, `details_json`)
  - history table for selected step from `recording_step_status_history`

## API Endpoints (MVP)

1. `GET /api/status/summary`
- Returns counts used by top cards.

2. `GET /api/status/wide`
- Returns rows from `recording_step_status_wide`.
- Query params:
  - `q` (text)
  - `zarr_use`
  - `only_blocking` (bool)
  - `limit`, `offset`

3. `GET /api/status/dataset/{dataset_id}`
- Returns detail rows from `recording_step_status_latest` for that dataset.

4. `GET /api/status/history`
- Query params: `dataset_id`, optional `step_name`, `limit`.
- Source: `recording_step_status_history`.

5. `GET /api/status/heartbeat`
- Returns `MAX(updated_utc)` and row counts for cheap poll checks.

## Live Update Strategy

Default MVP:
- Poll `heartbeat` every 10-15 seconds.
- If `max_updated_utc` changed, refresh visible data.

Optional phase 2:
- Add Server-Sent Events (SSE) for push-style updates.

Reasoning:
- Current registry size is small (~100s of datasets), so polling is simple and reliable.
- No materialized views needed at this scale.

## Performance Notes

- Query source should be `recording_step_status_wide` for table render.
- Details/history queries are on-demand only (row click), not preloaded.
- Add backend-side default limit (e.g. 500-1000) with pagination.

## Security + Access Model

Default local mode:
- bind `127.0.0.1`, no auth, for workstation use.

LAN mode (optional):
- bind `0.0.0.0` and allow other machines on same network to access the page.
- recommend network controls (firewall/VPN/SSH tunnel) before broad exposure.
- preferred durable deployment: keep the app on `127.0.0.1` and place auth/TLS
  in a reverse proxy rather than in the Python server.

This directly supports viewing from another computer when the storage host is reachable.

## UX Requirements

- Sticky header + horizontal scroll for wide columns.
- Stable step ordering matching registry/TUI language:
  - Import, BG Full, BG DS, Detect, Detect Quality, Refine Detect, Crop,
    Keypoints, Refined Keypoints, Eye Masks, Refined Eye Masks, Arena Assignment, Track, Stimulus, Calib, Tuning.
- Explicit legend for status colors and `N/A`.
- "Only issues" quick toggle.
- CSV export of current filtered table.

## Implementation Plan

### Phase 1: Read-Only Web MVP
- Build `/api/status/summary`, `/api/status/wide`, `/api/status/dataset/{dataset_id}`, `/api/status/history`, `/api/status/heartbeat`.
- Build static HTML table with filters and color coding.
- Add polling refresh.
- Add CLI args (`--registry`, `--host`, `--port`).

Acceptance:
- Opens in browser and shows live status without manual SQL.
- Can filter to rows missing keypoints or eye masks.

### Phase 2: Better Drilldown + History UX
- Step-focused detail panel with prettified JSON.
- History timeline grouped by step.
- Step-column click filtering.

Acceptance:
- Operator can identify exactly when/why a step became missing/error.

### Phase 3: Optional Integrations
- Link-outs to:
  - source profile index pages
  - training card/profile pages
  - related artifacts by dataset id
- Optional SSE refresh.

Acceptance:
- Status page acts as navigation hub for deeper diagnostics.

## Test Plan

Unit tests:
- API filter parsing and SQL parameterization.
- Summary counts and blocking-row logic.
- Dataset detail and history endpoint behavior.

Integration tests:
- Temporary sqlite fixture with seeded `recording_step_status*` rows.
- Verify filters and status rendering payloads.

Manual checks:
- Run page locally and confirm updates after pipeline writes new step status rows.

## Open Questions

- Should we keep this as pure read-only forever, or add guarded write actions later?
- Do we want auth in-process, or rely only on network perimeter controls?
- Should this supersede the status portion of `registry_tui`, or remain complementary?
