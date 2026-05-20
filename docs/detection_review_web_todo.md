# Detection Review Web TODO (Training Zarr Editing)

<!-- todo-meta
status: active
last_updated: 2026-05-20
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
- retune path
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

There are now two browser review surfaces with different backing data:

- `detect_review_web` is the materialized-image reviewer. It is primarily for
  training Zarrs and any archive that already exposes `raw_video/images_ds`.
- `video_detect_review_web` is the video-backed analysis reviewer. It resolves
  source media plus refined detections and can inspect/edit detection bboxes
  against traditional analysis Zarrs and clipped finalized collections.

For clipped analysis shells, `video_detect_review_web` uses the same resolver
shape Crimson needs:

- finalized collection selection from
  `experiment_index/finalized_runs/<collection_id>`;
- `recording_frame_index.parquet` mapping from parent frame to clip-local
  frame;
- clip-video or review-proxy media for the selected frame;
- run lookup under
  `clips/<clip_id>/cameras/<camera_serial>/refined_detect_runs/<run>`.

The recommended browser path for long clipped videos is to build review proxy
MP4s and pass `--review-proxy-manifest`. Directly serving original
high-resolution clips remains a diagnostic fallback, not the preferred operator
review path.

## Review Proxy Videos For Clipped Analysis

The first video-backed analysis viewer (`video_detect_review_web`) can resolve
clipped parent frames and refined boxes, but direct browser playback of the
original clip MP4s is a poor review substrate. The sleepyfish smoke clips are
`4512x4512` HEVC, roughly `14 GB` per 30-minute clip, and the MP4 `moov` atom is
at the end of the file. A browser `<video>` element may need multiple range
requests plus large HEVC decode work before it can display one exact review
frame. This is much less predictable than PyNvVideoCodec/Crimson decoding.

Preferred design: create derived browser-review proxy videos outside the
analysis Zarr and point the web viewer at a manifest. The proxy is cache data,
not canonical analysis truth.

Suggested layout:

```text
<recording>/
  clips/
  zarr/
  derived/
    review_proxy/
      video_detect/
        <proxy_run_id>/
          manifest.json
          clips/
            clip_000000/
              Cam2010093_1024_h264.mp4
            clip_000001/
              Cam2010093_1024_h264.mp4
```

Proxy contract:

- Same `clip_id`, `camera_serial`, frame count, FPS, and frame-index timeline as
  the source clip.
- Lower display resolution, initially `1024x1024` or `1280x1280`.
- Browser-friendly codec/container, initially H.264 MP4 with faststart metadata.
- One manifest entry per `(clip_id, camera_serial)`.
- Boxes remain stored/read in source-image coordinates; the frontend scales
  overlays to the displayed proxy dimensions. Proxy pixels are display-only.
- Proxies may be regenerated, cleaned by TTL, or copied to shared cache such as
  `/misc/public/palette_cache`; they should not be written into the canonical
  analysis Zarr.

Example manifest shape:

```json
{
  "schema_version": "palette.review_proxy.video.v1",
  "recording_id": "sleepyfish_2026_05_05_17_45_30_cam2010093",
  "source_recording_dir": "/groups/johnson/johnsonlab/jeremy/palette_smoke/sleepyfish_2026_05_05_17_45_30_cam2010093",
  "proxy_width": 1024,
  "proxy_height": 1024,
  "frame_count_policy": "same_as_source_clip",
  "timebase_policy": "same_fps_same_frame_index",
  "coordinate_policy": "scale_source_image_to_proxy_for_display_only",
  "clips": [
    {
      "clip_id": "clip_000000",
      "camera_serial": "2010093",
      "source_video_path": ".../clips/clip_000000/Cam2010093_....mp4",
      "proxy_video_path": ".../derived/review_proxy/video_detect/<proxy_run_id>/clips/clip_000000/Cam2010093_1024_h264.mp4",
      "source_width": 4512,
      "source_height": 4512,
      "proxy_width": 1024,
      "proxy_height": 1024,
      "fps": 30,
      "frame_count": 54000
    }
  ]
}
```

Planned viewer interface:

```bash
scripts/py -m fisheye.tune.video_detect_review_web \
  <analysis.zarr> \
  --review-proxy-manifest <recording>/derived/review_proxy/video_detect/<proxy_run_id>/manifest.json
```

Implementation status:

1. [x] Keep the existing source-video path for diagnostics.
2. [x] Add a builder that creates faststart H.264 proxy MP4s from clipped source
   videos and writes the manifest.
3. [x] Add proxy manifest resolution in `video_detect_review_backend.py`.
4. [ ] Add validation that proxy frame count/FPS matches the source clip.
5. [x] Prefer proxy media in the browser when a manifest is provided; fall back to
   source video only when no manifest is provided.

Builder status: `fisheye.utils.build_review_proxy_videos` creates the proxy
manifest and, with `--apply`, transcodes the selected clip-camera videos.
It is dry-run by default. The default `--encoder auto` prefers `libx264` when
available and falls back to NVENC H.264 encoders such as `h264_nvenc` on
FFmpeg builds that do not include GPL x264 support.

```bash
scripts/py -m fisheye.utils.build_review_proxy_videos \
  <recording_dir> \
  --proxy-run-id <proxy_run_id> \
  --proxy-width 1024 \
  --proxy-height 1024 \
  --limit 1
```

Full apply for all clips:

```bash
scripts/py -m fisheye.utils.build_review_proxy_videos \
  <recording_dir> \
  --proxy-run-id <proxy_run_id> \
  --proxy-width 1024 \
  --proxy-height 1024 \
  --apply
```

The default output directory is:

```text
<recording>/derived/review_proxy/video_detect/<proxy_run_id>/
```

After proxies are built, run the video-backed reviewer against the analysis
Zarr and pass the manifest explicitly:

```bash
scripts/py -m fisheye.tune.video_detect_review_web \
  <recording>/zarr/<recording>_analysis.zarr \
  --review-proxy-manifest <recording>/derived/review_proxy/video_detect/<proxy_run_id>/manifest.json \
  --host 0.0.0.0 \
  --port 8790
```

When a proxy manifest is present, the backend still resolves detections in
source-image coordinates and exposes both source and proxy dimensions to the
frontend. The proxy MP4 is used only as the media source for display.

### Video-Backed Review UI Status

`video_detect_review_web` now supports clipped analysis inspection against
finalized refined-detect collections and optional proxy videos. It is read-only
by default. Launch with `--edit` only when intentional manual writes back into
the analysis Zarr are desired.

Current review controls:

- Frame stepping by parent frame (`n` / `p`) and direct parent-frame seek.
- Native browser playback for fast screening, with overlay boxes following the
  browser's current media time.
- Issue navigation for frames with no present box or a non-`present` refined
  status (`m` next, `M` previous).
- Low-confidence navigation using the UI threshold (`l` next, `L` previous).
- Manual-edit navigation (`e` next, `E` previous).
- Save/clear controls are disabled in read-only mode and guarded server-side by
  the `editable` session flag.
- Unsaved bbox edits are buffered client-side by parent frame, remain visible
  when navigating back to that frame, and are saved together with `Save Pending`.

Save behavior:

- In editable mode, each saved frame updates the selected refined-detect state in
  the analysis Zarr.
- If there are unsaved edits on multiple frames, `Save` submits the whole pending
  set to `/api/frames/save_batch`; otherwise it saves the current frame.
- Saving promotes each edited frame into the recording's training Zarr only when
  the server is launched with `--promote-training-zarr <training.zarr>`.
- The post-save hook writes the analysis edit first, then calls
  `fisheye.tune.detect_training_promotion_backend`. Batch saves call the
  promotion backend once for all successfully saved frames instead of looping
  one frame at a time.
- For clipped sessions, the promotion backend groups appended training examples
  by source clip/video so one save batch can decode a clip once for all frames in
  that clip. Existing training rows are updated in-place without re-decoding the
  source video.
- Save responses and the server log report batch timing telemetry:
  `total_batch_s`, analysis write totals, promotion totals, clipped decode group
  counts, clipped decode total time, image-array write time, and promotion
  metadata write time.
- If promotion fails, the analysis edit remains valid and the UI reports
  `promotion_failed`; it does not silently roll back the analysis edit. Pending
  edits with promotion failures remain in the client buffer so the operator can
  retry.
- Without `--promote-training-zarr`, use
  `fisheye.utils.promote_analysis_detect_to_training` explicitly after a save
  when the edited frame should become a per-recording training example. For
  migrated/clipped stores, prefer `--use-registry-target` so the CLI writes the
  active registry `zarr_use='training'` store. For standard
  `<recording>_analysis.zarr` layouts without a registry target, the CLI infers
  the sibling `<recording>_training.zarr`; use `--training-zarr` for
  nonstandard targets.

Example editable analysis review with post-save promotion:

```bash
scripts/py -m fisheye.tune.video_detect_review_web \
  <recording>/zarr/<recording>_analysis.zarr \
  --edit \
  --promote-training-zarr <recording>/zarr/<recording>_training.zarr \
  --host 0.0.0.0 \
  --port 8790
```

Current limits:

- The hook supports traditional sessions and clipped finalized-collection
  sessions when the review resolver provides clip/camera/parent-frame mapping.
- The standalone promotion CLI supports clipped finalized-collection backfill
  from `recording_frame_index.parquet`; by default it promotes only manually
  edited frames and keeps manual clears as explicit negative examples. It can
  resolve the target training Zarr from the registry with
  `--use-registry-target`.
- Clipped promotion decodes the real source clip for training images. Review
  proxy videos are display-only and are not promoted into training Zarrs.

The issue-navigation target is intentionally broad: it includes both model
failures with no box and refined frames marked filtered/missing. This is the
right first pass for finding frames where the user may need to add a box, move a
box, or decide that no fish should be labeled.

### Browser Playback Semantics

`video_detect_review_web` uses the browser `<video>` element for continuous
playback and overlays boxes by resolving the current media time to a parent
frame. This is intended for fast visual screening, not proof that every source
frame has been presented to the reviewer.

At playback rates above the display refresh budget, the browser may skip
presenting source frames. For example, a `30 FPS` proxy on a `60 Hz` monitor can
usually present every source frame up to `2x` playback. At `4x`, the media stream
advances at `120` source frames per second, so the monitor cannot show every
source frame. The overlay follows the frame corresponding to the browser's
current media time; it does not force every source frame to be displayed.

Current policy:

- Use `1x` for ordinary inspection.
- Use `2x` as the practical upper bound for every-frame presentation on a
  `60 Hz` monitor with `30 FPS` proxies.
- Use `3x`/`4x`/`8x` only for fast screening where skipped visual frames are
  acceptable.
- When exact per-frame inspection matters, use manual `n`/`p` frame stepping or
  a future every-frame playback mode rather than native browser playback.

Deferred option: add an explicit "every-frame mode" that does not call
`video.play()`. Instead, it would advance parent frame `N, N+1, ...` on each
render tick, seek/render that exact proxy frame, and overlay the matching
detection payload. This guarantees no skipped source-frame presentation but caps
throughput at decode/seek/render speed and display refresh. A future optimized
version could use a server-side decoded-frame cache or a browser-side frame
stream, but the current browser `<video>` playback path should remain the simple
fast-screening mode.

Serving individual PyNvVideoCodec-decoded JPEG/WebP frames remains a useful
alternative for exact-frame labeling and can share the same clipped frame
resolver. For smooth browser playback and scrubbing, precomputed proxy MP4s are
the more debuggable first target than an on-the-fly PyNvVC downsampled stream.

## Why this is feasible now

- Manual save logic is already concentrated in `detect_review.py` and mostly reusable through `_apply` and `shared` curation helpers.
- A backend surface can keep that write contract in one place and prevent accidental drift.
- Existing `keypoint_review_web` pattern is a proven low-dependency bootstrap for the server/asset split.

## Suggested follow-up slices

- arena-aware navigation (`frame_arena`) using `source_rows_by_slot` and `_load_arena_slot_curated_edit_payload(...)`
- batch navigation and richer shortcuts (`manual_clear`, status workflow)
- optional `image/full-res` toggle and zoom/pan quality-of-life controls
- optional backend reuse for `detect_review` command by delegating save path only
