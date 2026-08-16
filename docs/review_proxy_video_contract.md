# Review Proxy Video Contract
<!-- contract-meta
version: 1
status: active
implementation: implemented
last_verified: 2026-05-28
purpose: Define derived browser-review proxy videos for Palette video-backed detection review.
-->

## Purpose

`video_detect_review_web` can resolve source clips directly, but direct browser
playback of full-resolution acquisition MP4s is a poor review substrate for
long clipped recordings. Sleepyfish clipped source videos are large
`4512x4512` HEVC files, and browser exact-frame review can require expensive
range requests and decode work before the first frame appears.

Review proxy videos are derived cache artifacts for browser display. They are
not canonical analysis truth.

## Layout

Proxy videos live under the recording root:

```text
<recording>/
  derived/
    review_proxy/
      video_detect/
        <proxy_run_id>/
          manifest.json
          clips/
            clip_000000/
              Cam2010093_1024x1024_h264.mp4
            clip_000001/
              Cam2010093_1024x1024_h264.mp4
```

## Proxy Contract

- Same `clip_id`, `camera_serial`, frame count, FPS, and frame-index timeline as
  the source clip.
- Lower display resolution, typically `1024x1024` or `1280x1280`.
- Browser-friendly codec/container, currently H.264 MP4 with faststart
  metadata.
- One manifest entry per `(clip_id, camera_serial)`.
- Proxy pixels are display-only.
- Canonical detections remain in source-image coordinates or normalized edit
  coordinates; frontends scale overlays to proxy dimensions for rendering.
- Proxies may be regenerated, cleaned by TTL, or copied with the recording.
- Proxies should not be written into the canonical analysis Zarr.

## Manifest Schema

Example:

```json
{
  "schema_version": "palette.review_proxy.video.v1",
  "recording_id": "sleepyfish_2026_05_05_17_45_30_cam2010093",
  "source_recording_dir": "/groups/johnson/johnsonlab/jeremy/recordings/sleepyfish_2026_05_05_17_45_30_cam2010093",
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
      "proxy_video_path": ".../derived/review_proxy/video_detect/<proxy_run_id>/clips/clip_000000/Cam2010093_1024x1024_h264.mp4",
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

The manifest is an active path surface. If the recording root moves, rewrite
active `source_*` and `proxy_*` paths in copied manifests or rebuild proxies in
the new root.

## Builder

Dry run one proxy plan:

```bash
scripts/py -m fisheye.utils.build_review_proxy_videos \
  <recording_dir> \
  --proxy-run-id <proxy_run_id> \
  --proxy-width 1024 \
  --proxy-height 1024 \
  --limit 1
```

Direct apply is useful for bounded smokes:

```bash
scripts/py -m fisheye.utils.build_review_proxy_videos \
  <recording_dir> \
  --proxy-run-id <proxy_run_id> \
  --proxy-width 1024 \
  --proxy-height 1024 \
  --clip-id clip_000000 \
  --hwaccel cuda \
  --scale-flags bilinear \
  --apply
```

Do not run full all-clip `--apply` transcodes on a cluster login node.

## Cluster Wrappers

Single sequential compute job:

```bash
scripts/submit_review_proxy_videos_bsub.sh \
  <recording_dir> \
  --proxy-run-id <proxy_run_id> \
  --proxy-width 1024 \
  --proxy-height 1024 \
  --encoder h264_nvenc \
  --hwaccel cuda \
  --scale-flags bilinear \
  --queue gpu_l4 \
  --gpus 1
```

For long clipped recordings, prefer the sharded LSF workflow:

```bash
scripts/submit_review_proxy_videos_sharded_bsub.sh \
  <recording_dir> \
  --proxy-run-id <proxy_run_id> \
  --proxy-width 1024 \
  --proxy-height 1024 \
  --encoder h264_nvenc \
  --hwaccel cuda \
  --scale-flags bilinear \
  --shard-count 4 \
  --max-active 4 \
  --queue gpu_l4 \
  --gpus 1 \
  --walltime 2:00 \
  --overwrite \
  --submit
```

The sharded wrapper submits bounded clip shards and a finalizer. Shards use
`--defer-manifest` and must not publish `manifest.json`. The finalizer verifies
all expected proxy files with `--write-manifest-only --require-existing-proxies`
and writes the single final manifest.

The sharded workflow uses `--skip-existing-valid` by default, so reruns with the
same `<proxy_run_id>` skip completed non-empty proxy MP4s and retry missing or
incomplete outputs.

Run directories:

- single-job wrapper: `<recording>/derived/review_proxy/video_detect/<proxy_run_id>/bsub_submission_<run_id>/`;
- sharded wrapper: `<recording>/derived/review_proxy/video_detect/<proxy_run_id>/bsub_sharded_<run_id>/`.

## Performance Notes

Cluster timing on `sleepyfish_2026_05_05_17_45_30_cam2010095` showed CPU
decode plus Lanczos scaling was too slow for full recordings, about `58s` wall
time for a 60s clip segment.

The current wrapper defaults to `--hwaccel cuda --scale-flags bilinear`, which
tested at about `22s` for the same 60s segment on an L4-class node. This is
acceptable for review proxies because proxy pixels are display-only.

An additional test with
`--hwaccel cuda --scale-flags bilinear --encoder libx264 --preset ultrafast`
was effectively tied with the NVENC encode path at about `22.3s`, with a
`2.8 MB` output. The fast path appears to come primarily from CUDA decode plus
cheaper bilinear scaling, not necessarily NVENC encoding. The wrapper default
remains `h264_nvenc` because that path is validated in cluster use.

## Reviewer Usage

Launch the video-backed reviewer with the proxy manifest:

```bash
scripts/py -m fisheye.tune.video_detect_review_web \
  <recording>/zarr/<recording>_analysis.zarr \
  --review-proxy-manifest <recording>/derived/review_proxy/video_detect/<proxy_run_id>/manifest.json \
  --host 0.0.0.0 \
  --port 8790
```

When a proxy manifest is present, the backend still resolves detections in
source-image coordinates and exposes both source and proxy dimensions to the
frontend. The proxy MP4 is only the media source for display.

## Validation Checklist

- Manifest schema is `palette.review_proxy.video.v1`.
- Every selected `(clip_id, camera_serial)` pair has one proxy entry.
- Every `proxy_video_path` exists and is non-empty.
- Proxy frame count and FPS match the source clip.
- Proxy dimensions match manifest `proxy_width` and `proxy_height`.
- The reviewer renders boxes by scaling source/normalized coordinates to proxy
  media dimensions.
