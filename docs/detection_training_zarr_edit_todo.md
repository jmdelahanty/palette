# Detection Training Zarr Editing TODO

## Objective
Create a safe path to edit detection labels in training Zarrs (primarily `refined_detect_runs` surfaces) without changing existing matplotlib/manual workflows, so we can support faster web-based curation and maintain strict save parity.

## Scope
- Editing targets: detection review/edit artifacts in `refined_detect_runs` used by training generation.
- Initial launch surface: canonical curated detect surface (`variant=refined`) first.
- Primary review/edit client: browser web UI.
- Legacy/multi-instance compatibility is second-phase.

Out of scope (initially):
- Replacing `scripts/py -m fisheye.tune.detect_review --manual`.
- Retune workflows (`--retune`, `--retune-ui`).
- Full arena-aware `frame_arena` web parity.

## Why this matters
- Matplotlib UI is SSH-fragile; web flow enables low-latency local review.
- Current save behavior for detection edits is intricate (source linkage, reason/status encoding, lineage), so we should keep semantics centralized and test them once.

## Baseline audit references
- `src/fisheye/tune/detect_review.py` (current manual review + save semantics).
- `src/fisheye/shared/refined_detect_curation.py` (canonical write/update helpers).
- `src/fisheye/shared/refined_detect_resolution.py` (targeted run/group resolution).
- `src/fisheye/shared/detect_reason_codec.py` (reason columns encoding compatibility).
- `docs/analysis_to_training_promotion_contract.md` (analysis-zarr edit promotion into per-recording training zarrs).

## Phase 0 — Safety and non-invasive prep
- [x] Do not alter existing `detect_review.py` control flow.
- [x] Confirm mutable read/write requirement: use `open_zarr_group_direct(..., use_consolidated=False)`.
- [x] Add/extend focused tests against in-memory fake groups for:
  - payload load from `dense refined` data
  - save semantics for edit/clear
  - source linkage updates (source row indexes, source-detection reason transitions)
  - reason/status arrays remain aligned.

## Phase 1 — Backend extraction (shared minimal API)
Create `src/fisheye/tune/detect_review_backend.py` with only the minimum needed for web MVP.

- [x] `resolve_latest_refined_and_payload(...)`
  - resolve zarr root + latest refined detect run.
  - support explicit `--refined-run` override.
- [x] `resolve_review_rows(..., include_all, target_frames, max_items)`
  - preserve existing failure/all semantics from matrix in detect reviewer.
- [x] `load_frame_payload(position)`
  - returns:
    - frame index
    - current bbox (`np.ndarray` shape `(4,)` cxcywh)
    - status/source/reason/flags arrays values
    - source metadata needed for lineage (`source_detect_row_index`, source surface arrays if present)
    - ROI image payload data (for frame-based review it is full frame)
- [x] `apply_detection_correction(position, rect_norm, manual_score=1.0, manual_class_id=0)`
  - mirror existing manual-review semantics:
    - rect present -> `present`, `manual`, `manual_correction`
    - rect clear -> `filtered_out`, `none`, `manual_clear`
    - set `manual_edit_flags=True`
    - update `source_detect_row_index`, `confidence_scores`, `class_ids`, `detection_source`
    - preserve `source_context` provenance fields for the write helper.
  - persist via:
    - `write_curated_refined_detect_surfaces(...)` for canonical sparse-aware path, or
    - `update_curated_refined_detect_rows(...)` depending on active payload representation.

## Phase 2 — Web MVP (first runnable flow)
Add `src/fisheye/tune/detect_review_web.py` + minimal static assets.

- [x] Server endpoints:
  - [x] `GET /` serves static UI
  - [x] `GET /api/state`
  - [x] `GET /api/frame/current`
  - [x] `POST /api/frame/current/save`
  - [x] `POST /api/nav`
- [x] Canvas viewer features:
  - [x] pan/zoom
  - [x] draw, move, and clear box
  - [x] visible current state/flags
  - [x] keyboard controls aligned to existing detect/manual semantics: `n`, `p`, `s`.
- [x] Command line mirrors keypoint reviewer style:
  - `scripts/py -m fisheye.tune.detect_review_web <zarr> --manual --port 8787 --host 0.0.0.0`

## Phase 3 — Validation
- [x] Browser-free backend tests for save semantics (required):
  - [x] editing one frame updates `bbox_norm`, `status`, `source_kind`, `reason`, `manual_edit_flags`
  - [x] updates `source_detect_row_index` when present
  - [x] keeps confidence/class/source arrays consistent
  - [x] stale downstream markers not affected unless edit path says so.
- [ ] End-to-end smoke:
  - small fixture zarr run with mutable canonical refined surface.
  - open/next/save/clear and validate arrays.

Validation completed for the first slice:

- `scripts/py -m py_compile src/fisheye/tune/detect_review_backend.py src/fisheye/tune/detect_review_web.py tests/unit/fisheye/test_detect_review_backend.py`
- `node --check src/fisheye/tune/detect_review_web/static/app.js`
- `scripts/py -m pytest -p no:cacheprovider tests/unit/fisheye/test_detect_review_backend.py -q`
- `scripts/py -m fisheye.tune.detect_review_web --help`

## Phase 4 — Phase-in parity features
- [ ] Add `x/d/b/a/N/R/P` equivalents if and only when backend semantics are fully covered by tests.
- [ ] Add `frame_arena` support only after canonical frame workflow is stable.
- [ ] Add legacy sparse fallback (`filtered/interpolated/manual` subgroup compatibility) if required by archive mix.

## Engineering constraints
- Preserve existing matplotlib reviewer behavior unchanged.
- Keep the implementation dependency-light (stdlib server + numpy/zarr stack already in repo).
- Keep dask-safe chunked write discipline in mind; avoid multi-worker writes to overlapping chunks.
- Ensure every edit includes provenance attributes for traceability (editor/action/axis/status).

## Definition of done
- User can run web detection reviewer with SSH port-forward and save one-frame corrections.
- `keypoint_review` and `detect_review` CLI paths still work.
- Focused backend tests pass with zarr semantics equivalent to existing manual save path.
- No changes to default metadata schema layout without explicit migration tests.
