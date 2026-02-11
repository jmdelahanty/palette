#!/usr/bin/env python3
"""
Native decode adapter for benchmark_detect_decode_backends.

This script runs FFmpeg decode (NVDEC by default) and emits a JSON payload with
fields expected by the benchmark runner:
  - frames_processed
  - duration_seconds
  - decode_fps
  - batch_decode_ms
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _run_ffprobe(
    ffprobe_bin: str,
    video_path: Path,
) -> Dict[str, Any]:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-select_streams",
        "v:0",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError("ffprobe returned no video streams")
    stream = streams[0]

    fps_raw = str(stream.get("avg_frame_rate", "0/0"))
    fps = 0.0
    if "/" in fps_raw:
        num_str, den_str = fps_raw.split("/", 1)
        try:
            num = float(num_str)
            den = float(den_str)
            if den != 0:
                fps = num / den
        except ValueError:
            fps = 0.0
    else:
        try:
            fps = float(fps_raw)
        except ValueError:
            fps = 0.0

    frames_raw = stream.get("nb_frames")
    total_frames: Optional[int]
    if frames_raw is None:
        total_frames = None
    else:
        try:
            total_frames = int(frames_raw)
        except (TypeError, ValueError):
            total_frames = None

    width = int(stream.get("width", 0) or 0)
    height = int(stream.get("height", 0) or 0)

    return {
        "fps": float(fps),
        "total_frames": total_frames,
        "width": width,
        "height": height,
    }


def _resolve_frames_to_process(
    start_frame: int,
    max_frames: int,
    total_frames: Optional[int],
) -> int:
    if start_frame < 0:
        raise ValueError("--start-frame must be >= 0")
    if max_frames < 0:
        raise ValueError("--max-frames must be >= 0")

    if total_frames is not None and start_frame >= total_frames:
        raise ValueError(
            f"start frame {start_frame} is beyond total frames {total_frames}"
        )

    if max_frames > 0:
        if total_frames is None:
            return max_frames
        return max(0, min(max_frames, total_frames - start_frame))

    if total_frames is None:
        raise ValueError(
            "--max-frames must be > 0 when total frame count cannot be resolved from ffprobe"
        )
    return max(0, total_frames - start_frame)


def _parse_ffmpeg_progress(
    ffmpeg_cmd: Sequence[str],
) -> Tuple[int, List[Tuple[float, int]], str]:
    """
    Execute ffmpeg and parse -progress output.

    Returns:
      final_frame_count,
      progress_points [(elapsed_seconds, frame_count)],
      stderr_text
    """
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    start = time.perf_counter()
    last_frame = 0
    pending_frame: Optional[int] = None
    points: List[Tuple[float, int]] = [(0.0, 0)]

    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key == "frame":
            try:
                pending_frame = int(value)
                if pending_frame > last_frame:
                    last_frame = pending_frame
            except ValueError:
                continue
        elif key == "progress":
            if pending_frame is None:
                frame_value = last_frame
            else:
                frame_value = max(last_frame, pending_frame)
            points.append((time.perf_counter() - start, int(frame_value)))

    return_code = proc.wait()
    stderr_text = proc.stderr.read() if proc.stderr is not None else ""
    if return_code != 0:
        tail = stderr_text.strip().splitlines()[-20:]
        raise RuntimeError(
            "ffmpeg command failed (exit {}):\n{}".format(
                return_code, "\n".join(tail)
            )
        )

    return last_frame, points, stderr_text


def _interpolate_time_for_frame(
    points: Sequence[Tuple[float, int]],
    target_frame: int,
    fallback_total_seconds: float,
    fallback_total_frames: int,
) -> float:
    if target_frame <= 0:
        return 0.0

    previous_t, previous_f = points[0]
    for current_t, current_f in points[1:]:
        if current_f < previous_f:
            continue
        if target_frame <= current_f:
            if current_f == previous_f:
                return float(current_t)
            frac = (target_frame - previous_f) / float(current_f - previous_f)
            return float(previous_t + frac * (current_t - previous_t))
        previous_t, previous_f = current_t, current_f

    if fallback_total_frames > 0:
        return float(
            fallback_total_seconds * (target_frame / float(fallback_total_frames))
        )
    return float(fallback_total_seconds)


def _estimate_batch_decode_ms(
    raw_points: Sequence[Tuple[float, int]],
    frames_processed: int,
    batch_size: int,
    total_duration_seconds: float,
) -> List[float]:
    if frames_processed <= 0 or batch_size <= 0:
        return []

    points: List[Tuple[float, int]] = [(0.0, 0)]
    max_frame = 0
    max_time = 0.0
    for t_sec, frame in raw_points:
        t_sec = max(0.0, float(t_sec))
        frame = max(0, int(frame))
        if frame < max_frame:
            continue
        max_frame = frame
        max_time = max(max_time, t_sec)
        points.append((max_time, max_frame))

    if points[-1][1] < frames_processed:
        points.append((max(total_duration_seconds, points[-1][0]), frames_processed))
    elif points[-1][0] < total_duration_seconds:
        points.append((float(total_duration_seconds), points[-1][1]))

    num_batches = int(math.ceil(frames_processed / float(batch_size)))
    boundaries: List[int] = []
    for batch_index in range(1, num_batches + 1):
        boundary = min(batch_index * batch_size, frames_processed)
        boundaries.append(int(boundary))

    end_times: List[float] = []
    for boundary in boundaries:
        end_times.append(
            _interpolate_time_for_frame(
                points=points,
                target_frame=boundary,
                fallback_total_seconds=total_duration_seconds,
                fallback_total_frames=frames_processed,
            )
        )

    batch_decode_ms: List[float] = []
    prev = 0.0
    for current in end_times:
        delta = max(0.0, current - prev)
        batch_decode_ms.append(delta * 1000.0)
        prev = current
    return batch_decode_ms


def run_native_decode_benchmark(
    *,
    video_path: Path,
    start_frame: int,
    max_frames: int,
    batch_size: int,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    hwaccel: str,
    hwaccel_output_format: Optional[str],
    extra_ffmpeg_args: Sequence[str],
    repeat_index: Optional[int],
    phase: Optional[str],
) -> Dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    probe = _run_ffprobe(ffprobe_bin=ffprobe_bin, video_path=video_path)
    fps = float(probe["fps"])
    if fps <= 0:
        raise RuntimeError(
            f"Could not determine valid FPS from ffprobe ({ffprobe_bin}) for {video_path}"
        )

    frames_to_process = _resolve_frames_to_process(
        start_frame=start_frame,
        max_frames=max_frames,
        total_frames=probe["total_frames"],
    )
    if frames_to_process <= 0:
        raise RuntimeError("No frames selected for native decode benchmark")

    start_seconds = start_frame / fps
    ffmpeg_cmd: List[str] = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
    ]
    if hwaccel and hwaccel.lower() != "none":
        ffmpeg_cmd.extend(["-hwaccel", hwaccel])
        if hwaccel_output_format:
            ffmpeg_cmd.extend(["-hwaccel_output_format", hwaccel_output_format])
    ffmpeg_cmd.extend(
        [
            "-ss",
            f"{start_seconds:.6f}",
            "-i",
            str(video_path),
            "-frames:v",
            str(frames_to_process),
            "-progress",
            "pipe:1",
            "-f",
            "null",
            "-",
        ]
    )
    if extra_ffmpeg_args:
        insert_pos = len(ffmpeg_cmd) - 4
        ffmpeg_cmd[insert_pos:insert_pos] = list(extra_ffmpeg_args)

    run_start = time.perf_counter()
    final_frame, progress_points, _stderr_text = _parse_ffmpeg_progress(ffmpeg_cmd)
    duration_seconds = time.perf_counter() - run_start

    if final_frame <= 0:
        frames_processed = frames_to_process
    else:
        frames_processed = min(final_frame, frames_to_process)

    decode_fps = (
        float(frames_processed / duration_seconds) if duration_seconds > 0 else 0.0
    )
    batch_decode_ms = _estimate_batch_decode_ms(
        raw_points=progress_points,
        frames_processed=frames_processed,
        batch_size=batch_size,
        total_duration_seconds=duration_seconds,
    )

    return {
        "frames_processed": int(frames_processed),
        "duration_seconds": float(duration_seconds),
        "decode_fps": float(decode_fps),
        "batch_decode_ms": [float(v) for v in batch_decode_ms],
        "native_backend": "ffmpeg_nvdec" if hwaccel and hwaccel != "none" else "ffmpeg_cpu",
        "phase": phase,
        "repeat_index": repeat_index,
        "ffmpeg_command": " ".join(shlex.quote(token) for token in ffmpeg_cmd),
        "video_fps": fps,
        "video_width": int(probe["width"]),
        "video_height": int(probe["height"]),
    }


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run FFmpeg decode benchmark and emit JSON payload for "
            "benchmark_detect_decode_backends --native-cmd."
        )
    )
    parser.add_argument("--video", required=True, type=Path, help="Input video path.")
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start frame index (default: 0).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Frames to decode; 0 means to EOF if frame count is known.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used to derive batch_decode_ms (default: 64).",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="/opt/orange/lib/ffmpeg-nvidia/bin/ffmpeg",
        help="FFmpeg binary path (default: /opt/orange/lib/ffmpeg-nvidia/bin/ffmpeg).",
    )
    parser.add_argument(
        "--ffprobe-bin",
        default="/opt/orange/lib/ffmpeg-nvidia/bin/ffprobe",
        help="ffprobe binary path (default: /opt/orange/lib/ffmpeg-nvidia/bin/ffprobe).",
    )
    parser.add_argument(
        "--hwaccel",
        default="cuda",
        help="FFmpeg -hwaccel value; use 'none' to disable (default: cuda).",
    )
    parser.add_argument(
        "--hwaccel-output-format",
        default="cuda",
        help="FFmpeg -hwaccel_output_format value (default: cuda).",
    )
    parser.add_argument(
        "--ffmpeg-extra-arg",
        action="append",
        default=[],
        help="Extra FFmpeg arg token (repeatable).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path. If omitted, JSON is printed to stdout.",
    )
    parser.add_argument(
        "--repeat-index",
        type=int,
        default=None,
        help="Optional repetition index for provenance.",
    )
    parser.add_argument(
        "--phase",
        default=None,
        help="Optional phase label (e.g., warmup/timed).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    result = run_native_decode_benchmark(
        video_path=args.video.expanduser().resolve(),
        start_frame=int(args.start_frame),
        max_frames=int(args.max_frames),
        batch_size=int(args.batch_size),
        ffmpeg_bin=str(args.ffmpeg_bin),
        ffprobe_bin=str(args.ffprobe_bin),
        hwaccel=str(args.hwaccel),
        hwaccel_output_format=(
            None
            if str(args.hwaccel_output_format).lower() == "none"
            else str(args.hwaccel_output_format)
        ),
        extra_ffmpeg_args=list(args.ffmpeg_extra_arg),
        repeat_index=args.repeat_index,
        phase=args.phase,
    )

    payload = json.dumps(result, indent=2)
    if args.output_json is None:
        print(payload)
        return

    out = args.output_json.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
