# Recording Status Page TODO

Purpose: implement a live, read-only web status page for recording pipeline progress using existing registry step-status tables/views.

Design reference:
- `docs/recording_status_page_design.md`

Date anchored: 2026-03-05.

## Decision Snapshot (Current)

- [x] Build this as a dedicated feature package, not a single large `utils` script.
- [x] Keep a thin launcher in `fisheye.utils` for command-line ergonomics.
- [x] Reuse existing registry views (`recording_step_status_wide`, `recording_step_status_latest`, `recording_step_status_history`) as the only data model.
- [x] Start read-only (no web-triggered writes/jobs in v1).

## Recommended Module Layout

- [x] Add `src/fisheye/status_page/__init__.py`.
- [x] Add `src/fisheye/status_page/app.py`.
- [x] Add `src/fisheye/status_page/api.py`.
- [x] Add `src/fisheye/status_page/query.py`.
- [x] Add `src/fisheye/status_page/models.py`.
- [x] Add `src/fisheye/status_page/static/index.html`.
- [x] Add `src/fisheye/status_page/static/status_page.js`.
- [x] Add `src/fisheye/status_page/static/status_page.css`.
- [x] Add CLI wrapper `src/fisheye/utils/serve_recording_status_page.py`.

## Phase 0: Skeleton + Wiring

- [x] Implement app factory with `--registry`, `--host`, `--port` configuration.
- [x] Add registry path validation and startup error handling.
- [x] Add static file serving and `/` route to status page UI.
- [x] Add `/healthz` endpoint with registry connectivity check.

## Phase 1: Query Layer (Read-Only Registry Access)

- [x] Implement summary query from status views/tables for dashboard cards.
- [x] Implement wide-row query from `recording_step_status_wide` with pagination.
- [x] Implement dataset detail query from `recording_step_status_latest`.
- [x] Implement history query from `recording_step_status_history`.
- [x] Implement safe filter parameterization (`q`, `zarr_use`, `only_blocking`, `limit`, `offset`).

## Phase 2: API Endpoints

- [x] `GET /api/status/summary`
- [x] `GET /api/status/wide`
- [x] `GET /api/status/dataset/{dataset_id}`
- [x] `GET /api/status/history`
- [x] `GET /api/status/heartbeat`
- [x] Standardize JSON error responses and bad-request validation.

## Phase 3: Frontend MVP

- [x] Render summary cards (total rows, blocking rows, missing rows, error rows, latest update).
- [x] Render wide status table with sticky header and horizontal scroll.
- [x] Implement color legend and cell styling for `ok/missing/absent/na/error`.
- [x] Add filters for text search, `Use`, and "Only issues".
- [x] Add row-click details panel with step metadata and JSON blocks.
- [x] Add per-dataset step history panel.
- [ ] Add CSV download for currently filtered rows.

## Phase 4: Live Refresh + Ops

- [x] Poll heartbeat every 10-15 seconds and refresh only when changed.
- [x] Add CLI examples to doc/runbook.
- [x] Document local-only mode (`127.0.0.1`) and LAN mode (`0.0.0.0`) with network safety notes.
- [x] Add reverse-proxy deployment examples for auth/TLS (`caddy` and `nginx`).
- [x] Warn at startup when binding to a non-loopback host.
- [x] Ensure the page tolerates missing optional fields in legacy rows.

## Phase 5: Tests

- [x] Unit tests for filter parsing and SQL parameter binding.
- [x] Unit tests for summary/wide/detail/history query behavior.
- [ ] API tests for success + bad-input responses.
- [ ] Frontend smoke test for table render with sample payload.
- [ ] Manual validation against real registry while pipeline updates step status.

## Phase 6: Nice-to-Have (After MVP)

- [ ] SSE-based push updates (optional replacement for polling).
- [ ] Deep links to profile/training-card artifact pages by dataset.
- [ ] Column click filters by pipeline step.
- [ ] Saved filter presets in browser local storage.

## Acceptance Criteria (v1)

- [ ] A user can open one URL and see live pipeline step status for all recordings.
- [ ] A user can quickly isolate rows with missing/error keypoints or eye masks.
- [ ] A user can inspect run/method/review/details and status history for any dataset.
- [ ] No new registry schema changes are required for v1.

## Open Questions

- [ ] Should this eventually include guarded write actions (manual status override / job launch)?
- [ ] Should authentication be added in-app, or remain network-perimeter controlled?
- [ ] Should this become the primary status UI over `registry_tui`, or remain complementary?
