"""Build browser-friendly review proxy videos for clipped recordings.

The proxies are derived display artifacts, not canonical analysis data. They
preserve the source clip frame timeline while using a lower-resolution,
browser-friendly H.264 MP4 representation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


SCHEMA_VERSION = "palette.review_proxy.video.v1"
DEFAULT_PROXY_WIDTH = 1024
DEFAULT_PROXY_HEIGHT = 1024
DEFAULT_ENCODER = "auto"
DEFAULT_H264_ENCODER = "libx264"
DEFAULT_PRESET = "veryfast"
DEFAULT_CRF = 23
DEFAULT_SCALE_FLAGS = "lanczos"
H264_ENCODER_PRIORITY = ("libx264", "h264_nvenc", "nvenc_h264", "nvenc", "libopenh264")
NVENC_H264_ENCODERS = {"h264_nvenc", "nvenc_h264", "nvenc"}

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class ReviewProxyOptions:
    output_dir: Path
    proxy_run_id: str
    proxy_width: int = DEFAULT_PROXY_WIDTH
    proxy_height: int = DEFAULT_PROXY_HEIGHT
    encoder: str = DEFAULT_ENCODER
    preset: str = DEFAULT_PRESET
    crf: int = DEFAULT_CRF
    hwaccel: str | None = None
    scale_flags: str = DEFAULT_SCALE_FLAGS
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    apply: bool = False
    overwrite: bool = False
    probe: bool = True
    limit: int | None = None
    defer_manifest: bool = False
    write_manifest_only: bool = False
    require_existing_proxies: bool = False
    skip_existing_valid: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_proxy_run_id() -> str:
    return "video_detect_proxy_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON {path}: {exc}") from exc


def _atomic_write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _safe_component(value: Any, *, fallback: str) -> str:
    text = str(value or "").strip()
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return label or fallback


def _clip_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_rows = payload.get("clips")
    if raw_rows is None:
        raw_rows = payload.get("rows")
    if raw_rows is None:
        raw_rows = payload.get("camera_artifacts")
    if not isinstance(raw_rows, list):
        raise ValueError("recording_clip_index JSON must include clips, rows, or camera_artifacts list")

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("recording_clip_index row is not an object")
        if isinstance(raw.get("camera_artifacts"), list):
            base = {key: value for key, value in raw.items() if key != "camera_artifacts"}
            for artifact in raw["camera_artifacts"]:
                if not isinstance(artifact, Mapping):
                    raise ValueError("camera_artifacts item is not an object")
                rows.append({**base, **dict(artifact)})
        else:
            rows.append(dict(raw))
    return rows


def _resolve_recording_path(recording_dir: Path, value: Any) -> Path:
    if value is None:
        raise ValueError("Expected path value, got None")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = recording_dir.resolve()
    resolved = (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Recording-relative path escapes recording root: {value}")
    return resolved


def _fraction_to_float(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _nonnegative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _run_subprocess(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _ffmpeg_video_encoders(
    *,
    ffmpeg_bin: str,
    runner: CommandRunner = _run_subprocess,
) -> set[str]:
    command = [str(ffmpeg_bin), "-hide_banner", "-encoders"]
    result = runner(command)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg encoder discovery failed "
            f"(exit {result.returncode})\n"
            f"command: {' '.join(shlex.quote(part) for part in command)}\n"
            f"stderr:\n{result.stderr}"
        )
    encoders: set[str] = set()
    for line in (result.stdout or "").splitlines():
        stripped = line.strip()
        if not stripped or not stripped.startswith("V"):
            continue
        parts = stripped.split(None, 2)
        if len(parts) >= 2:
            encoders.add(parts[1])
    return encoders


def resolve_review_proxy_encoder(
    encoder: str,
    *,
    ffmpeg_bin: str = "ffmpeg",
    runner: CommandRunner = _run_subprocess,
) -> str:
    requested = str(encoder or DEFAULT_ENCODER).strip()
    if requested and requested != "auto":
        return requested
    encoders = _ffmpeg_video_encoders(ffmpeg_bin=ffmpeg_bin, runner=runner)
    for candidate in H264_ENCODER_PRIORITY:
        if candidate in encoders:
            return candidate
    available_preview = ", ".join(sorted(encoders)[:20])
    raise RuntimeError(
        "No supported H.264 encoder found for browser review proxies. "
        "Install an FFmpeg build with libx264 or run on a machine with h264_nvenc. "
        f"Available video encoders include: {available_preview}"
    )


def _nvenc_preset(preset: str) -> str:
    requested = str(preset or "").strip()
    if requested in {"p1", "p2", "p3", "p4", "p5", "p6", "p7", "slow", "medium", "fast", "hp", "hq"}:
        return requested
    if requested in {"ultrafast", "superfast", "veryfast", "faster"}:
        return "p3"
    return "p4"


def probe_video(
    video_path: Path,
    *,
    ffprobe_bin: str = "ffprobe",
    runner: CommandRunner = _run_subprocess,
) -> dict[str, Any]:
    command = [
        str(ffprobe_bin),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = runner(command)
    if result.returncode != 0:
        raise RuntimeError(
            "ffprobe failed for "
            f"{video_path} (exit {result.returncode})\n"
            f"command: {' '.join(shlex.quote(part) for part in command)}\n"
            f"stderr:\n{result.stderr}"
        )
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise RuntimeError(f"ffprobe found no video stream: {video_path}")
    stream = streams[0]
    fps = _fraction_to_float(stream.get("avg_frame_rate")) or _fraction_to_float(stream.get("r_frame_rate"))
    return {
        "codec_name": stream.get("codec_name"),
        "width": _positive_int(stream.get("width")),
        "height": _positive_int(stream.get("height")),
        "fps": fps,
        "frame_count": _positive_int(stream.get("nb_frames")),
        "duration_s": _positive_float(stream.get("duration")),
        "ffprobe_command": " ".join(shlex.quote(part) for part in command),
    }


def build_ffmpeg_review_proxy_command(
    *,
    ffmpeg_bin: str,
    source_video: Path,
    output_video: Path,
    proxy_width: int,
    proxy_height: int,
    encoder: str = DEFAULT_H264_ENCODER,
    preset: str = DEFAULT_PRESET,
    crf: int = DEFAULT_CRF,
    hwaccel: str | None = None,
    scale_flags: str = DEFAULT_SCALE_FLAGS,
    overwrite: bool = False,
) -> list[str]:
    if proxy_width <= 0 or proxy_height <= 0:
        raise ValueError(f"Proxy dimensions must be positive, got {proxy_width}x{proxy_height}")
    if not re.fullmatch(r"[A-Za-z0-9_+.-]+", str(scale_flags)):
        raise ValueError(f"Unsafe FFmpeg scale flags value: {scale_flags!r}")
    command = [
        str(ffmpeg_bin),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y" if overwrite else "-n",
    ]
    hwaccel_value = str(hwaccel or "").strip()
    if hwaccel_value and hwaccel_value.lower() not in {"none", "off", "false"}:
        if not re.fullmatch(r"[A-Za-z0-9_+.-]+", hwaccel_value):
            raise ValueError(f"Unsafe FFmpeg hwaccel value: {hwaccel!r}")
        command.extend(["-hwaccel", hwaccel_value])
    command.extend(
        [
            "-i",
            str(source_video),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            f"scale={int(proxy_width)}:{int(proxy_height)}:flags={scale_flags}",
            "-c:v",
            str(encoder),
        ]
    )
    if encoder in NVENC_H264_ENCODERS:
        command.extend(["-preset", _nvenc_preset(preset), "-cq", str(int(crf)), "-b:v", "0"])
    elif encoder == "libopenh264":
        command.extend(["-b:v", "4M"])
    else:
        command.extend(["-preset", str(preset), "-crf", str(int(crf))])
    command.extend(["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_video)])
    return command


def _write_proxy_video(
    *,
    command: Sequence[str],
    output_video: Path,
    overwrite: bool,
    runner: CommandRunner,
) -> dict[str, Any]:
    if output_video.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing proxy video: {output_video}")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    tmp_video = output_video.with_name(f".{output_video.stem}.tmp{output_video.suffix}")
    if tmp_video.exists():
        if overwrite:
            tmp_video.unlink()
        else:
            raise FileExistsError(f"Temporary proxy video already exists: {tmp_video}")

    tmp_command = list(command)
    tmp_command[-1] = str(tmp_video)
    started = time.perf_counter()
    result = runner(tmp_command)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg proxy transcode failed for "
            f"{output_video} (exit {result.returncode})\n"
            f"command: {' '.join(shlex.quote(part) for part in tmp_command)}\n"
            f"stderr:\n{result.stderr}"
        )
    if not tmp_video.exists():
        raise RuntimeError(f"ffmpeg reported success but did not create {tmp_video}")
    os.replace(tmp_video, output_video)
    return {
        "status": "written",
        "output_video": str(output_video),
        "elapsed_seconds": float(elapsed),
        "bytes": int(output_video.stat().st_size),
    }


def _existing_proxy_output(output_video: Path) -> dict[str, Any] | None:
    if not output_video.exists() or not output_video.is_file():
        return None
    size = int(output_video.stat().st_size)
    if size <= 0:
        return None
    return {
        "status": "existing",
        "output_video": str(output_video),
        "elapsed_seconds": 0.0,
        "bytes": size,
    }


def _require_existing_proxy_outputs(clips: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    missing: list[str] = []
    for clip in clips:
        output_video = Path(str(clip["proxy_video_path"]))
        existing = _existing_proxy_output(output_video)
        if existing is None:
            missing.append(str(output_video))
        else:
            outputs.append(existing)
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:20])
        if len(missing) > 20:
            preview += f"\n  ... {len(missing) - 20} more missing proxies"
        raise FileNotFoundError(f"Missing required review proxy videos:\n{preview}")
    return outputs


def _output_video_path(output_dir: Path, *, clip_id: str, source_video: Path, proxy_width: int, proxy_height: int) -> Path:
    source_component = _safe_component(source_video.stem, fallback="camera")
    name = f"{source_component}_{int(proxy_width)}x{int(proxy_height)}_h264.mp4"
    return output_dir / "clips" / _safe_component(clip_id, fallback="clip") / name


def _selected_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    clip_ids: Optional[Sequence[str]] = None,
    camera_serials: Optional[Sequence[str]] = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    selected_clip_ids = {str(value) for value in clip_ids or []}
    selected_cameras = {str(value) for value in camera_serials or []}
    selected: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        clip_id = str(row.get("clip_id") or f"clip_{int(row.get('clip_index') or 0):06d}")
        camera_serial = str(row.get("camera_serial") or "")
        if selected_clip_ids and clip_id not in selected_clip_ids:
            continue
        if selected_cameras and camera_serial not in selected_cameras:
            continue
        selected.append(row)
    if limit is not None:
        selected = selected[: int(limit)]
    if not selected:
        raise ValueError("No clip-camera rows matched the requested filters")
    return selected


def build_review_proxy_manifest(
    recording_dir: str | Path,
    *,
    options: ReviewProxyOptions,
    clip_ids: Optional[Sequence[str]] = None,
    camera_serials: Optional[Sequence[str]] = None,
    ffmpeg_runner: CommandRunner = _run_subprocess,
    ffprobe_runner: CommandRunner = _run_subprocess,
) -> dict[str, Any]:
    recording_path = Path(recording_dir).expanduser().resolve()
    clip_index_path = recording_path / "recording_clip_index.json"
    if not clip_index_path.exists():
        raise FileNotFoundError(f"recording_clip_index.json not found: {clip_index_path}")
    clip_index_payload = _read_json(clip_index_path)
    if not isinstance(clip_index_payload, Mapping):
        raise ValueError(f"recording_clip_index JSON is not an object: {clip_index_path}")

    rows = _selected_rows(
        _clip_rows(clip_index_payload),
        clip_ids=clip_ids,
        camera_serials=camera_serials,
        limit=options.limit,
    )
    resolved_encoder = resolve_review_proxy_encoder(
        options.encoder,
        ffmpeg_bin=options.ffmpeg_bin,
        runner=ffmpeg_runner,
    )
    clips: list[dict[str, Any]] = []
    for row in rows:
        clip_id = str(row.get("clip_id") or f"clip_{int(row.get('clip_index') or 0):06d}")
        clip_index_value = _nonnegative_int(row.get("clip_index"))
        if clip_index_value is None:
            try:
                clip_index_value = int(clip_id.rsplit("_", 1)[-1])
            except (IndexError, ValueError):
                clip_index_value = None
        camera_serial = str(row.get("camera_serial") or "")
        if not camera_serial:
            raise ValueError(f"Missing camera_serial for {clip_id}")
        source_video = _resolve_recording_path(recording_path, row.get("video_path"))
        proxy_video = _output_video_path(
            options.output_dir,
            clip_id=clip_id,
            source_video=source_video,
            proxy_width=options.proxy_width,
            proxy_height=options.proxy_height,
        )
        probe: dict[str, Any] = {}
        probe_error: str | None = None
        if options.probe and source_video.exists():
            try:
                probe = probe_video(source_video, ffprobe_bin=options.ffprobe_bin, runner=ffprobe_runner)
            except Exception as exc:  # noqa: BLE001 - surface probe failures in manifest
                probe_error = str(exc)

        source_width = _positive_int(row.get("source_width") or row.get("width")) or probe.get("width")
        source_height = _positive_int(row.get("source_height") or row.get("height")) or probe.get("height")
        fps = _positive_float(row.get("fps") or row.get("source_video_fps")) or probe.get("fps")
        frame_count = _positive_int(row.get("frame_count")) or probe.get("frame_count")
        command = build_ffmpeg_review_proxy_command(
            ffmpeg_bin=options.ffmpeg_bin,
            source_video=source_video,
            output_video=proxy_video,
            proxy_width=options.proxy_width,
            proxy_height=options.proxy_height,
            encoder=resolved_encoder,
            preset=options.preset,
            crf=options.crf,
            hwaccel=options.hwaccel,
            scale_flags=options.scale_flags,
            overwrite=options.overwrite,
        )
        clip_payload: dict[str, Any] = {
            "clip_id": clip_id,
            "clip_index": clip_index_value,
            "camera_serial": camera_serial,
            "source_video_path": str(source_video),
            "proxy_video_path": str(proxy_video),
            "source_width": source_width,
            "source_height": source_height,
            "proxy_width": int(options.proxy_width),
            "proxy_height": int(options.proxy_height),
            "fps": fps,
            "frame_count": frame_count,
            "encoder": str(resolved_encoder),
            "hwaccel": options.hwaccel,
            "scale_flags": str(options.scale_flags),
            "ffmpeg_command": " ".join(shlex.quote(str(part)) for part in command),
        }
        if probe_error:
            clip_payload["probe_error"] = probe_error
        clips.append(clip_payload)

    recording_id = str(clip_index_payload.get("recording_id") or recording_path.name)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "planned",
        "created_at_utc": _utc_now(),
        "proxy_run_id": options.proxy_run_id,
        "recording_id": recording_id,
        "source_recording_dir": str(recording_path),
        "recording_clip_index": str(clip_index_path),
        "proxy_width": int(options.proxy_width),
        "proxy_height": int(options.proxy_height),
        "encoder": str(options.encoder),
        "resolved_encoder": str(resolved_encoder),
        "preset": str(options.preset),
        "crf": int(options.crf),
        "hwaccel": options.hwaccel,
        "scale_flags": str(options.scale_flags),
        "frame_count_policy": "same_as_source_clip",
        "timebase_policy": "same_fps_same_frame_index",
        "coordinate_policy": "linear_scale_source_image_to_proxy_for_display_only",
        "artifact_kind": "derived_review_proxy_video",
        "clip_count": int(len(clips)),
        "clips": clips,
    }


def build_review_proxy_videos(
    recording_dir: str | Path,
    *,
    options: ReviewProxyOptions,
    clip_ids: Optional[Sequence[str]] = None,
    camera_serials: Optional[Sequence[str]] = None,
    ffmpeg_runner: CommandRunner = _run_subprocess,
    ffprobe_runner: CommandRunner = _run_subprocess,
) -> dict[str, Any]:
    started = time.perf_counter()
    if options.apply and options.write_manifest_only:
        raise ValueError("--apply and --write-manifest-only are mutually exclusive")
    if options.defer_manifest and options.write_manifest_only:
        raise ValueError("--defer-manifest and --write-manifest-only are mutually exclusive")
    manifest = build_review_proxy_manifest(
        recording_dir,
        options=options,
        clip_ids=clip_ids,
        camera_serials=camera_serials,
        ffmpeg_runner=ffmpeg_runner,
        ffprobe_runner=ffprobe_runner,
    )
    manifest_path = options.output_dir / "manifest.json"
    outputs: list[dict[str, Any]] = []
    if options.apply:
        for clip in manifest["clips"]:
            command = shlex.split(str(clip["ffmpeg_command"]))
            output_video = Path(str(clip["proxy_video_path"]))
            existing = _existing_proxy_output(output_video) if options.skip_existing_valid else None
            if existing is not None:
                outputs.append({**existing, "status": "skipped_existing"})
                continue
            outputs.append(
                _write_proxy_video(
                    command=command,
                    output_video=output_video,
                    overwrite=options.overwrite,
                    runner=ffmpeg_runner,
                )
            )
        manifest = {**manifest, "status": "ok", "materialized_at_utc": _utc_now(), "video_outputs": outputs}
        if options.defer_manifest:
            manifest = {**manifest, "manifest_deferred": True}
        else:
            _atomic_write_json(manifest_path, manifest, overwrite=options.overwrite)
            manifest = {**manifest, "manifest_written": True}
    elif options.write_manifest_only or options.require_existing_proxies:
        outputs = _require_existing_proxy_outputs(manifest["clips"])
        manifest = {**manifest, "status": "ok", "materialized_at_utc": _utc_now(), "video_outputs": outputs}
        _atomic_write_json(manifest_path, manifest, overwrite=options.overwrite)
        manifest = {**manifest, "manifest_written": True}
    else:
        manifest = {**manifest, "status": "dry_run", "dry_run_only": True}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": manifest["status"],
        "apply": bool(options.apply),
        "output_dir": str(options.output_dir),
        "manifest_path": str(manifest_path),
        "proxy_run_id": options.proxy_run_id,
        "clip_count": int(manifest["clip_count"]),
        "ffmpeg_bin": str(options.ffmpeg_bin),
        "ffmpeg_available": shutil.which(str(options.ffmpeg_bin)) is not None,
        "ffprobe_bin": str(options.ffprobe_bin),
        "ffprobe_available": shutil.which(str(options.ffprobe_bin)) is not None,
        "encoder": str(manifest.get("encoder")),
        "resolved_encoder": str(manifest.get("resolved_encoder")),
        "hwaccel": manifest.get("hwaccel"),
        "scale_flags": str(manifest.get("scale_flags")),
        "manifest_written": bool(manifest.get("manifest_written", False)),
        "manifest_deferred": bool(manifest.get("manifest_deferred", False)),
        "host": socket.gethostname(),
        "pid": int(os.getpid()),
        "duration_seconds": float(time.perf_counter() - started),
        "manifest": manifest,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build derived browser-review proxy videos for a clipped recording. "
            "Dry-run by default; pass --apply to transcode and write manifest.json."
        )
    )
    parser.add_argument("recording_dir", type=Path, help="Recording folder containing recording_clip_index.json")
    parser.add_argument("--output-dir", type=Path, help="Proxy run output directory")
    parser.add_argument("--proxy-run-id", default=None, help="Stable proxy run id; default is UTC timestamp")
    parser.add_argument("--proxy-width", type=int, default=DEFAULT_PROXY_WIDTH)
    parser.add_argument("--proxy-height", type=int, default=DEFAULT_PROXY_HEIGHT)
    parser.add_argument(
        "--encoder",
        default=DEFAULT_ENCODER,
        help="FFmpeg H.264 encoder to use, or 'auto' to prefer libx264 then NVENC.",
    )
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--crf", type=int, default=DEFAULT_CRF)
    parser.add_argument("--hwaccel", default=None, help="Optional FFmpeg input hardware acceleration, e.g. cuda.")
    parser.add_argument("--scale-flags", default=DEFAULT_SCALE_FLAGS, help=f"FFmpeg scale filter flags (default: {DEFAULT_SCALE_FLAGS}).")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--no-probe", action="store_true", help="Do not run ffprobe; use recording_clip_index metadata only.")
    parser.add_argument("--clip-id", action="append", help="Limit to one or more clip ids.")
    parser.add_argument("--camera-serial", action="append", help="Limit to one or more camera serials.")
    parser.add_argument("--limit", type=int, help="Limit selected clip-camera rows for smoke testing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing proxy artifacts.")
    parser.add_argument("--skip-existing-valid", action="store_true", help="Skip non-empty existing proxy MP4 outputs during --apply.")
    parser.add_argument("--defer-manifest", action="store_true", help="With --apply, transcode selected clips but do not write manifest.json.")
    parser.add_argument("--write-manifest-only", action="store_true", help="Verify expected proxies exist and write manifest.json without transcoding.")
    parser.add_argument("--require-existing-proxies", action="store_true", help="Require every expected proxy MP4 to exist before writing a manifest.")
    parser.add_argument("--apply", action="store_true", help="Transcode videos and write manifest.")
    parser.add_argument("--json", action="store_true", help="Print full JSON result.")
    return parser


def _print_summary(result: Mapping[str, Any]) -> None:
    print(f"status: {result.get('status')}")
    print(f"apply: {result.get('apply')}")
    print(f"output_dir: {result.get('output_dir')}")
    print(f"manifest_path: {result.get('manifest_path')}")
    print(f"proxy_run_id: {result.get('proxy_run_id')}")
    print(f"clip_count: {result.get('clip_count')}")
    print(f"encoder: {result.get('encoder')} resolved={result.get('resolved_encoder')}")
    print(f"hwaccel: {result.get('hwaccel')} scale_flags={result.get('scale_flags')}")
    print(f"manifest_written: {result.get('manifest_written')} manifest_deferred={result.get('manifest_deferred')}")
    print(f"ffmpeg_bin: {result.get('ffmpeg_bin')} available={result.get('ffmpeg_available')}")
    manifest = result.get("manifest")
    if isinstance(manifest, Mapping):
        clips = manifest.get("clips")
        if isinstance(clips, list):
            for clip in clips[:3]:
                if isinstance(clip, Mapping):
                    print(f"  {clip.get('clip_id')} cam{clip.get('camera_serial')}: {clip.get('proxy_video_path')}")
            if len(clips) > 3:
                print(f"  ... {len(clips) - 3} more proxy clips")
    if not result.get("apply"):
        print("dry_run: no files written; pass --apply to transcode proxies")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    recording_dir = Path(args.recording_dir).expanduser().resolve()
    proxy_run_id = args.proxy_run_id or _default_proxy_run_id()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else recording_dir / "derived" / "review_proxy" / "video_detect" / _safe_component(proxy_run_id, fallback="proxy")
    )
    result = build_review_proxy_videos(
        recording_dir,
        options=ReviewProxyOptions(
            output_dir=output_dir,
            proxy_run_id=proxy_run_id,
            proxy_width=int(args.proxy_width),
            proxy_height=int(args.proxy_height),
            encoder=str(args.encoder),
            preset=str(args.preset),
            crf=int(args.crf),
            hwaccel=str(args.hwaccel) if args.hwaccel else None,
            scale_flags=str(args.scale_flags),
            ffmpeg_bin=str(args.ffmpeg_bin),
            ffprobe_bin=str(args.ffprobe_bin),
            apply=bool(args.apply),
            overwrite=bool(args.overwrite),
            probe=not bool(args.no_probe),
            limit=args.limit,
            defer_manifest=bool(args.defer_manifest),
            write_manifest_only=bool(args.write_manifest_only),
            require_existing_proxies=bool(args.require_existing_proxies),
            skip_existing_valid=bool(args.skip_existing_valid),
        ),
        clip_ids=args.clip_id,
        camera_serials=args.camera_serial,
    )
    if args.json:
        print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    else:
        _print_summary(result)
    return 0 if result.get("status") in {"ok", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
