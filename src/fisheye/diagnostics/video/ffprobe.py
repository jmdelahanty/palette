from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def is_ffprobe_tooling_issue(message: str) -> bool:
    lowered = message.lower()
    return "ffprobe executable not found" in lowered or "ffprobe executable is not accessible" in lowered


def _run_ffprobe_json(args: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe executable not found") from exc
    except PermissionError as exc:
        raise RuntimeError("ffprobe executable is not accessible") from exc
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(stderr or f"ffprobe failed with exit code {result.returncode}")
    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError("ffprobe returned no output")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("ffprobe returned a non-object JSON payload")
    return payload


def probe_stream_payload(video_path: Path) -> dict[str, Any]:
    return _run_ffprobe_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
    )


def probe_frame_payload(video_path: Path, *, max_frames: int | None = None) -> list[dict[str, Any]]:
    args = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "frame=pkt_pts,pkt_pts_time,pkt_dts,pkt_dts_time,"
            "pts_time,dts_time,best_effort_timestamp_time,pict_type,key_frame,pkt_size"
        ),
        "-of",
        "json",
    ]
    if max_frames is not None and max_frames > 0:
        args.extend(["-read_intervals", f"%+#{int(max_frames)}"])
    args.append(str(video_path))
    payload = _run_ffprobe_json(args)
    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        raise RuntimeError("ffprobe frame payload did not include a frame list")
    return [item for item in frames if isinstance(item, dict)]
