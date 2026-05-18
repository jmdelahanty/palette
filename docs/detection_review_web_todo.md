# Detection Review Web TODO (Training Zarr Editing)

<!-- todo-meta
status: active
last_updated: 2026-05-18
-->

## Goal

Build a browser review flow for detection curation that writes to training/refinement zarr
surfaces with the same persistence semantics as `detect_review.py`, while keeping the current Matplotlib/manual reviewer untouched.

The first deliverable is a **single-frame MVP**: load one frame at a time, view/edit the box, save, and navigate.

## Current baseline (what exists today)

Editable detection artifacts are in `refined_detect_runs/<run>` and handled by:

- `src/fisheye/tune/detect_review.py`
  - `run_manual_review(...)` / `run_retune_review(...)`
  - `_select_refined_review_rows(...)`
  - `_load_dense_curated_edit_payload(...)`
  - `_write_dense_curated_edit_payload(...)`
  - frame-axis manual save/update logic in `run_manual_review`
- `src/fisheye/shared/refined_detect_curation.py`
  - `update_curated_refined_detect_rows(...)`
  - `write_curated_refined_detect_surfaces(...)`
  - curated/refined storage helpers used by both dense and sparse paths
- `src/fisheye/shared/refined_detect_resolution.py`
  - existing run/variant resolution helpers
- `src/fisheye/shared/detect_reason_codec.py`
  - reason read/write utilities

Current manual save semantics already include:

- `status_labels` toggles (`present` / `filtered_out`)
- `source_kind_labels` (`manual` / `none`)
- `source_detect_row_index` selection and retention
- `manual_edit_flags=True`
- `reason_labels` (`manual_correction`, `manual_clear`)
- `confidence_scores`, `class_ids`, `detection_source` updates
- optional source-detection sync (`source_surface_*` fields)
- curated provenance metadata via shared write helpers

## High-level web architecture

Add a dedicated backend and thin stdlib server:

- `src/fisheye/tune/detect_review_backend.py`
  - backend/session primitives
- `src/fisheye/tune/detect_review_web.py`
  - CLI + `BaseHTTPRequestHandler` + state router
- `src/fisheye/tune/detect_review_web/static/`
  - minimal canvas UI

This keeps existing manual review operational and lets web parity be built incrementally.

## Slice 1 MVP (recommended first implementation)

Keep scope strict:
- [x] canonical refined frame-axis review path only (`variant=refined`)
- [x] `review_all/targets` semantics only
- [x] one box per frame
- [x] no arena-aware mode
- [x] no status transition shortcuts beyond save+nav

### Backend functions to implement

1. [x] `resolve_review_context(...)`
   - open root with `open_zarr_group_direct(..., mode="a", use_consolidated=False)`
   - resolve refined run (`--refined-run` if provided)
   - resolve base payload from `_load_dense_curated_edit_payload(...)` and `raw_video/images_ds`

2. [x] `list_review_rows(...)`
   - preserve semantics from `_select_refined_review_rows`:
     - failure-only default
     - `--all` + target frame filtering
     - `--max-frames` cap

3. [x] `load_frame_payload(position)`
   - return:
     - `position`, `frame_idx`, current bbox, status, source/manual flags, reason
     - `source_detect_row_index`, `detection_source`
     - encoded frame image bytes for canvas rendering

4. [x] `apply_manual_edit(position, rect_norm|None, manual_class_id=..., manual_score=...)`
   - duplicate existing manual rules:
     - present edit => `present/manual/manual_correction`
     - clear => `filtered_out/none/manual_clear`
     - set manual flag and update numeric fields
   - write through `_write_dense_curated_edit_payload(...)` (initially dense path only)

### API MVP

- [x] `GET /api/state`
- [x] `GET /api/frame/current`
- [x] `POST /api/frame/current/save`
- [x] `POST /api/nav`
- [x] static `/` served from repo-local assets

### UI MVP

- [x] frame canvas with current bbox overlay
- [x] create/drag/clear box interaction
- [x] keys: `n` next, `p` previous, `s` save, `q` quit
- [x] compact status strip with frame + reason + flags

## Testing requirements (browser-free first)

Add/extend tests in `tests/unit/fisheye/test_detect_review_backend.py`:

- in-memory/fake-group coverage for both dense curated semantics and mocked read path
- present edit updates:
  - `status_labels`
  - `source_kind_labels`
  - `reason_labels`
  - `manual_edit_flags`
  - bbox and numeric fields (`confidence_scores`, `class_ids`)
- clear edit updates:
  - `filtered_out/none/manual_clear` + invalidation fields
- source row + source-surface sync behavior for curated dense runs
- reason/readable reason-byte consistency for edited rows
- verify mutable open path uses `use_consolidated=False`

## What to defer

- replacing `run_manual_review(...)` UI
- `frame_arena` and multi-slot per frame support
- retune path and batch save
- full review-status transitions (`a/N/R/P`) and follow-up flags
- any behavior changes outside the manual save path

## Current implementation status

Implemented in the first slice:

- `src/fisheye/tune/detect_review_backend.py`
- `src/fisheye/tune/detect_review_web.py`
- `src/fisheye/tune/detect_review_web/static/`
- `tests/unit/fisheye/test_detect_review_backend.py`

Validation completed:

- `scripts/py -m py_compile src/fisheye/tune/detect_review_backend.py src/fisheye/tune/detect_review_web.py tests/unit/fisheye/test_detect_review_backend.py`
- `node --check src/fisheye/tune/detect_review_web/static/app.js`
- `scripts/py -m pytest -p no:cacheprovider tests/unit/fisheye/test_detect_review_backend.py -q`
- `scripts/py -m fisheye.tune.detect_review_web --help`

Still required before calling the web detection reviewer production-ready:

- Real-zarr smoke against an actual training/analysis zarr with `raw_video/images_ds` and `refined_detect_runs`.
- Browser smoke through SSH port forwarding.
- Explicit acceptance/approval workflow integration.
- Arena-aware and multi-instance review support.

## Analysis-Zarr And Video-Backed Review Status

The current web detection reviewer is primarily useful for materialized training
Zarrs and any archive that already exposes `raw_video/images_ds`. It is not yet
a general analysis-video viewer.

For traditional single-video analysis Zarrs without persisted image arrays, a
future web reviewer would need to:

- resolve `source_video_path`;
- decode requested video frames on demand;
- overlay the selected refined-detect run;
- write edits back through the refined-detect curation contract.

For clipped analysis shells, the web reviewer would need the same resolver
layer Crimson now needs:

- finalized collection selection from
  `experiment_index/finalized_runs/<collection_id>`;
- `recording_frame_index.parquet` mapping from parent frame to clip-local
  frame;
- clip-video decode for the selected frame;
- run lookup under
  `clips/<clip_id>/cameras/<camera_serial>/refined_detect_runs/<run>`.

That is a separate implementation slice. For now, Crimson is the preferred
place to build analysis-video review for clipped recordings, while the web
reviewer remains focused on materialized training examples.

## Why this is feasible now

- Manual save logic is already concentrated in `detect_review.py` and mostly reusable through `_apply` and `shared` curation helpers.
- A backend surface can keep that write contract in one place and prevent accidental drift.
- Existing `keypoint_review_web` pattern is a proven low-dependency bootstrap for the server/asset split.

## Suggested follow-up slices

- arena-aware navigation (`frame_arena`) using `source_rows_by_slot` and `_load_arena_slot_curated_edit_payload(...)`
- batch navigation and richer shortcuts (`manual_clear`, status workflow)
- optional `image/full-res` toggle and zoom/pan quality-of-life controls
- optional backend reuse for `detect_review` command by delegating save path only
