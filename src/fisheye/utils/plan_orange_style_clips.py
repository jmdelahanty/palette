"""Plan Orange-style rolling clips from a long camera video.

This utility is intentionally planning-only. It reads an existing camera MP4,
its ``Cam*_meta.csv`` sidecar, and its ``Cam*_keyframe.json`` sidecar, then
emits a keyframe-aligned clip plan compatible with Orange's rolling clip
semantics. It does not split video files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


MODULE_NAME = "fisheye.utils.plan_orange_style_clips"
DEFAULT_TARGET_DURATION_MINUTES = 30.0
SUPPORTED_SNAP_DIRECTIONS = ("next", "previous", "nearest")


@dataclass(frozen=True)
class MetadataTable:
    path: Path
    fieldnames: tuple[str, ...]
    frame_ids: np.ndarray
    timestamps: np.ndarray | None
    timestamps_sys: np.ndarray | None

    @property
    def row_count(self) -> int:
        return int(self.frame_ids.shape[0])


@dataclass(frozen=True)
class KeyframeSummary:
    path: Path
    total_frames: int
    fps: float
    keyframe_frames: np.ndarray
    payload: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _coerce_int(value: Any, *, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected integer {field}, got {value!r}") from exc


def _coerce_float(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected numeric {field}, got {value!r}") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"Expected positive finite {field}, got {value!r}")
    return parsed


def _camera_serial_from_name(path: Path) -> str | None:
    match = re.search(r"Cam(\d+)", path.name)
    return match.group(1) if match else None


def _infer_recording_id(video_path: Path) -> str:
    for parent in video_path.parents:
        if parent.name == "cams":
            return parent.parent.name
    return video_path.stem


def _infer_metadata_csv(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}_meta.csv")


def _infer_keyframe_json(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}_keyframe.json")


def _parse_optional_int(text: str | None) -> int | None:
    if text is None:
        return None
    stripped = str(text).strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


def load_native_metadata_csv(path: str | Path) -> MetadataTable:
    metadata_path = Path(path).expanduser().resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    frame_ids: list[int] = []
    timestamps: list[int | None] = []
    timestamps_sys: list[int | None] = []
    with metadata_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata CSV has no header: {metadata_path}")
        fields = tuple(str(name) for name in reader.fieldnames)
        if "frame_id" not in fields and "recording_frame_id" not in fields:
            raise ValueError(
                f"Metadata CSV must include frame_id or recording_frame_id: {metadata_path}"
            )
        frame_key = "recording_frame_id" if "recording_frame_id" in fields else "frame_id"
        has_timestamp = "timestamp" in fields
        has_timestamp_sys = "timestamp_sys" in fields
        for row_number, row in enumerate(reader, start=2):
            value = row.get(frame_key)
            try:
                frame_ids.append(int(str(value).strip()))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {frame_key} at {metadata_path}:{row_number}: {value!r}"
                ) from exc
            if has_timestamp:
                timestamps.append(_parse_optional_int(row.get("timestamp")))
            if has_timestamp_sys:
                timestamps_sys.append(_parse_optional_int(row.get("timestamp_sys")))

    timestamp_array = None
    timestamp_sys_array = None
    if timestamps:
        timestamp_array = np.asarray([-1 if value is None else value for value in timestamps], dtype=np.int64)
    if timestamps_sys:
        timestamp_sys_array = np.asarray(
            [-1 if value is None else value for value in timestamps_sys],
            dtype=np.int64,
        )

    return MetadataTable(
        path=metadata_path,
        fieldnames=fields,
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        timestamps=timestamp_array,
        timestamps_sys=timestamp_sys_array,
    )


def load_keyframe_summary(path: str | Path) -> KeyframeSummary:
    keyframe_path = Path(path).expanduser().resolve()
    if not keyframe_path.exists():
        raise FileNotFoundError(f"Keyframe JSON not found: {keyframe_path}")
    try:
        payload = json.loads(keyframe_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid keyframe JSON {keyframe_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected keyframe JSON object: {keyframe_path}")
    total_frames = _coerce_int(payload.get("total_frames"), field="total_frames")
    fps = _coerce_float(payload.get("fps"), field="fps")
    raw_keyframes = payload.get("keyframe_frames")
    if not isinstance(raw_keyframes, list) or not raw_keyframes:
        raise ValueError(f"Keyframe JSON has no keyframe_frames list: {keyframe_path}")
    keyframes = np.asarray([_coerce_int(v, field="keyframe_frames[]") for v in raw_keyframes], dtype=np.int64)
    keyframes = np.unique(keyframes[(keyframes >= 0) & (keyframes < total_frames)])
    if keyframes.size == 0 or int(keyframes[0]) != 0:
        raise ValueError(f"Keyframe JSON must include frame 0 as a keyframe: {keyframe_path}")
    return KeyframeSummary(
        path=keyframe_path,
        total_frames=int(total_frames),
        fps=float(fps),
        keyframe_frames=keyframes,
        payload=payload,
    )


def _snap_keyframe(
    *,
    keyframes: np.ndarray,
    target_frame: int,
    total_frames: int,
    direction: str,
) -> int:
    if direction not in SUPPORTED_SNAP_DIRECTIONS:
        raise ValueError(f"Unsupported snap direction: {direction}")
    target = int(max(0, min(int(target_frame), int(total_frames - 1))))
    insertion = int(np.searchsorted(keyframes, target, side="left"))
    previous_value = int(keyframes[max(0, insertion - 1)])
    next_value = int(keyframes[min(insertion, len(keyframes) - 1)])
    if direction == "next":
        return next_value
    if direction == "previous":
        return previous_value
    previous_delta = abs(target - previous_value)
    next_delta = abs(next_value - target)
    return next_value if next_delta <= previous_delta else previous_value


def choose_clip_boundaries(
    *,
    total_frames: int,
    fps: float,
    keyframe_frames: np.ndarray,
    target_duration_minutes: float,
    snap_direction: str = "next",
) -> list[dict[str, Any]]:
    if total_frames <= 0:
        raise ValueError("total_frames must be > 0")
    if fps <= 0:
        raise ValueError("fps must be > 0")
    target_frames = max(1, int(round(float(target_duration_minutes) * 60.0 * float(fps))))
    starts: list[int] = [0]
    targets: list[int] = [0]
    target = target_frames
    while target < int(total_frames):
        snapped = _snap_keyframe(
            keyframes=keyframe_frames,
            target_frame=target,
            total_frames=total_frames,
            direction=snap_direction,
        )
        if snapped <= starts[-1]:
            next_index = int(np.searchsorted(keyframe_frames, starts[-1] + 1, side="left"))
            if next_index >= len(keyframe_frames):
                break
            snapped = int(keyframe_frames[next_index])
        if snapped >= int(total_frames):
            break
        starts.append(int(snapped))
        targets.append(int(target))
        target += target_frames
    starts.append(int(total_frames))
    targets.append(int(total_frames))

    boundaries: list[dict[str, Any]] = []
    for index in range(len(starts) - 1):
        start = int(starts[index])
        end_exclusive = int(starts[index + 1])
        target_start = int(targets[index])
        boundaries.append(
            {
                "clip_index": index,
                "clip_id": f"clip_{index:06d}",
                "target_start_frame": target_start,
                "start_frame": start,
                "end_frame_exclusive": end_exclusive,
                "frame_count": end_exclusive - start,
                "snap_delta_frames": start - target_start,
                "target_start_time_s": target_start / float(fps),
                "start_time_s": start / float(fps),
                "duration_s": (end_exclusive - start) / float(fps),
                "final_clip": index == len(starts) - 2,
                "start_reason": "initial" if index == 0 else f"target_keyframe_snap_{snap_direction}",
            }
        )
    return boundaries


def _slice_gap_count(values: np.ndarray, start: int, end_exclusive: int) -> int:
    if end_exclusive - start <= 1:
        return 0
    diff = np.diff(values[start:end_exclusive])
    return int(np.count_nonzero(diff != 1))


def _array_value(values: np.ndarray | None, index: int) -> int | None:
    if values is None:
        return None
    if index < 0 or index >= int(values.shape[0]):
        return None
    value = int(values[index])
    return None if value < 0 else value


def build_clip_plan(
    *,
    video_path: str | Path,
    metadata_csv: str | Path | None = None,
    keyframe_json: str | Path | None = None,
    target_duration_minutes: float = DEFAULT_TARGET_DURATION_MINUTES,
    snap_direction: str = "next",
    recording_id: str | None = None,
    camera_serial: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    resolved_video = Path(video_path).expanduser().resolve()
    if not resolved_video.exists():
        raise FileNotFoundError(f"Video not found: {resolved_video}")
    resolved_metadata = Path(metadata_csv).expanduser().resolve() if metadata_csv else _infer_metadata_csv(resolved_video)
    resolved_keyframe = Path(keyframe_json).expanduser().resolve() if keyframe_json else _infer_keyframe_json(resolved_video)
    metadata = load_native_metadata_csv(resolved_metadata)
    keyframes = load_keyframe_summary(resolved_keyframe)
    resolved_recording_id = str(recording_id).strip() if recording_id else _infer_recording_id(resolved_video)
    resolved_camera_serial = str(camera_serial).strip() if camera_serial else _camera_serial_from_name(resolved_video)

    boundaries = choose_clip_boundaries(
        total_frames=keyframes.total_frames,
        fps=keyframes.fps,
        keyframe_frames=keyframes.keyframe_frames,
        target_duration_minutes=target_duration_minutes,
        snap_direction=snap_direction,
    )

    checks: list[dict[str, Any]] = []
    if metadata.row_count == keyframes.total_frames:
        checks.append({"status": "ok", "code": "metadata_rows_match_keyframe_total_frames"})
    else:
        checks.append(
            {
                "status": "fail",
                "code": "metadata_rows_mismatch_keyframe_total_frames",
                "metadata_rows": metadata.row_count,
                "keyframe_total_frames": keyframes.total_frames,
            }
        )
    global_gap_count = _slice_gap_count(metadata.frame_ids, 0, metadata.row_count)
    checks.append(
        {
            "status": "ok" if global_gap_count == 0 else "fail",
            "code": "recording_frame_id_continuity",
            "recording_frame_id_gaps": int(global_gap_count),
        }
    )

    clip_rows: list[dict[str, Any]] = []
    keyframe_set = set(int(v) for v in keyframes.keyframe_frames.tolist())
    for boundary in boundaries:
        start = int(boundary["start_frame"])
        end_exclusive = int(boundary["end_frame_exclusive"])
        if start < 0 or end_exclusive > metadata.row_count or end_exclusive <= start:
            checks.append(
                {
                    "status": "fail",
                    "code": "clip_boundary_outside_metadata_rows",
                    "clip_id": boundary["clip_id"],
                    "start_frame": start,
                    "end_frame_exclusive": end_exclusive,
                    "metadata_rows": metadata.row_count,
                }
            )
            continue
        frame_count = end_exclusive - start
        gaps = _slice_gap_count(metadata.frame_ids, start, end_exclusive)
        start_is_keyframe = start in keyframe_set
        checks.append(
            {
                "status": "ok" if start_is_keyframe else "fail",
                "code": "clip_start_is_keyframe",
                "clip_id": boundary["clip_id"],
                "start_frame": start,
            }
        )
        checks.append(
            {
                "status": "ok" if gaps == 0 else "fail",
                "code": "clip_recording_frame_id_continuity",
                "clip_id": boundary["clip_id"],
                "recording_frame_id_gaps": int(gaps),
            }
        )
        first_recording_frame_id = int(metadata.frame_ids[start])
        last_recording_frame_id = int(metadata.frame_ids[end_exclusive - 1])
        clip_rows.append(
            {
                "recording_id": resolved_recording_id,
                "session_id": resolved_recording_id,
                "producer": "palette_clip_planner_from_orange_style_long_video",
                "recording_backend_mode": "derived_stream_copy_plan",
                "camera_serial": resolved_camera_serial,
                "clip_index": int(boundary["clip_index"]),
                "clip_id": boundary["clip_id"],
                "clip_directory": f"clips/{boundary['clip_id']}",
                "video_path": f"clips/{boundary['clip_id']}/{resolved_video.name}",
                "metadata_path": f"clips/{boundary['clip_id']}/{resolved_video.stem}_meta.csv",
                "keyframe_path": f"clips/{boundary['clip_id']}/{resolved_video.stem}_keyframe.json",
                "clip_manifest_path": f"clips/{boundary['clip_id']}/clip_manifest.json",
                "source_video_path": str(resolved_video),
                "source_metadata_path": str(metadata.path),
                "source_keyframe_path": str(keyframes.path),
                "target_start_frame": int(boundary["target_start_frame"]),
                "actual_start_frame": start,
                "end_frame_exclusive": end_exclusive,
                "target_start_time_s": float(boundary["target_start_time_s"]),
                "actual_start_time_s": float(boundary["start_time_s"]),
                "duration_s": float(boundary["duration_s"]),
                "frame_count": int(frame_count),
                "first_recording_frame_id": first_recording_frame_id,
                "last_recording_frame_id": last_recording_frame_id,
                "first_clip_local_frame_index": 0,
                "last_clip_local_frame_index": int(frame_count - 1),
                "recording_frame_id_gaps": int(gaps),
                "first_timestamp": _array_value(metadata.timestamps, start),
                "last_timestamp": _array_value(metadata.timestamps, end_exclusive - 1),
                "first_timestamp_sys": _array_value(metadata.timestamps_sys, start),
                "last_timestamp_sys": _array_value(metadata.timestamps_sys, end_exclusive - 1),
                "packet_count": None,
                "packet_count_source": None,
                "rollover_at_recording_frame_id": (
                    first_recording_frame_id if int(boundary["clip_index"]) > 0 else None
                ),
                "status": "planned",
                "start_reason": boundary["start_reason"],
                "stop_reason": "rollover" if not boundary["final_clip"] else "source_eof",
                "final_clip": bool(boundary["final_clip"]),
                "start_is_keyframe": bool(start_is_keyframe),
                "snap_delta_frames": int(boundary["snap_delta_frames"]),
                "snap_delta_s": float(boundary["snap_delta_frames"] / keyframes.fps),
            }
        )

    status = "ok" if all(item.get("status") == "ok" for item in checks) else "fail"
    return {
        "status": status,
        "generated_by": MODULE_NAME,
        "generated_at_utc": _utc_now(),
        "host": socket.gethostname(),
        "pid": int(os.getpid()),
        "duration_seconds": float(time.perf_counter() - started),
        "mode": "dry_run",
        "video_path": str(resolved_video),
        "metadata_csv": str(metadata.path),
        "keyframe_json": str(keyframes.path),
        "recording_id": resolved_recording_id,
        "camera_serial": resolved_camera_serial,
        "target_duration_minutes": float(target_duration_minutes),
        "target_duration_frames": int(round(float(target_duration_minutes) * 60.0 * keyframes.fps)),
        "snap_direction": str(snap_direction),
        "source": {
            "total_frames": int(keyframes.total_frames),
            "metadata_rows": int(metadata.row_count),
            "fps": float(keyframes.fps),
            "keyframe_count": int(keyframes.keyframe_frames.shape[0]),
            "first_keyframe_frame": int(keyframes.keyframe_frames[0]),
            "last_keyframe_frame": int(keyframes.keyframe_frames[-1]),
            "metadata_fields": list(metadata.fieldnames),
        },
        "checks": checks,
        "clip_count": int(len(clip_rows)),
        "clips": clip_rows,
    }


def write_plan_artifacts(plan: Mapping[str, Any], output_dir: str | Path, *, prefix: str) -> dict[str, str]:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{prefix}.json"
    csv_path = out_dir / f"{prefix}.csv"
    json_path.write_text(json.dumps(_json_safe(plan), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    clips = list(plan.get("clips") or [])
    fieldnames: list[str] = []
    for row in clips:
        if isinstance(row, Mapping):
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(str(key))
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in clips:
            writer.writerow({key: _json_safe(row.get(key)) for key in fieldnames})
    return {"json": str(json_path), "csv": str(csv_path)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan keyframe-aligned Orange-style rolling clips for a long camera video. "
            "This command is dry-run only and does not write clipped videos."
        )
    )
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--metadata-csv", type=Path, help="Cam*_meta.csv path; inferred from video name if omitted.")
    parser.add_argument("--keyframe-json", type=Path, help="Cam*_keyframe.json path; inferred from video name if omitted.")
    parser.add_argument(
        "--target-duration-minutes",
        type=float,
        default=DEFAULT_TARGET_DURATION_MINUTES,
        help="Target clip duration before snapping starts to keyframes.",
    )
    parser.add_argument(
        "--snap-direction",
        choices=SUPPORTED_SNAP_DIRECTIONS,
        default="next",
        help="How target boundaries are snapped to keyframes.",
    )
    parser.add_argument("--recording-id")
    parser.add_argument("--camera-serial")
    parser.add_argument("--output-dir", type=Path, help="Directory for recording_clip_index JSON/CSV plan artifacts.")
    parser.add_argument("--output-prefix", default="recording_clip_index")
    parser.add_argument("--json", action="store_true", help="Print the full plan JSON.")
    parser.add_argument("--summary", action="store_true", help="Print a compact human summary.")
    parser.add_argument("--dry-run", action="store_true", help="Accepted for clarity; this tool is always dry-run.")
    return parser


def _print_summary(plan: Mapping[str, Any], artifacts: Mapping[str, str] | None = None) -> None:
    print(f"status: {plan.get('status')}")
    print(f"video: {plan.get('video_path')}")
    print(f"metadata_csv: {plan.get('metadata_csv')}")
    print(f"keyframe_json: {plan.get('keyframe_json')}")
    source = plan.get("source") if isinstance(plan.get("source"), Mapping) else {}
    print(
        "source: "
        f"frames={source.get('total_frames')} metadata_rows={source.get('metadata_rows')} "
        f"fps={source.get('fps')} keyframes={source.get('keyframe_count')}"
    )
    print(
        f"plan: clips={plan.get('clip_count')} "
        f"target_minutes={plan.get('target_duration_minutes')} snap={plan.get('snap_direction')}"
    )
    checks = list(plan.get("checks") or [])
    failures = [item for item in checks if isinstance(item, Mapping) and item.get("status") != "ok"]
    print(f"checks: total={len(checks)} failures={len(failures)}")
    for row in list(plan.get("clips") or [])[:5]:
        if not isinstance(row, Mapping):
            continue
        print(
            f"  {row.get('clip_id')}: frames {row.get('actual_start_frame')}.."
            f"{int(row.get('end_frame_exclusive')) - 1} "
            f"recording_frame_id {row.get('first_recording_frame_id')}.."
            f"{row.get('last_recording_frame_id')} count={row.get('frame_count')}"
        )
    if int(plan.get("clip_count") or 0) > 5:
        print(f"  ... {int(plan.get('clip_count') or 0) - 5} more clips")
    if artifacts:
        print(f"wrote_json: {artifacts.get('json')}")
        print(f"wrote_csv: {artifacts.get('csv')}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    plan = build_clip_plan(
        video_path=args.video_path,
        metadata_csv=args.metadata_csv,
        keyframe_json=args.keyframe_json,
        target_duration_minutes=args.target_duration_minutes,
        snap_direction=args.snap_direction,
        recording_id=args.recording_id,
        camera_serial=args.camera_serial,
    )
    artifacts = None
    if args.output_dir is not None:
        artifacts = write_plan_artifacts(plan, args.output_dir, prefix=args.output_prefix)
        plan = {**plan, "artifacts": artifacts}
    if args.json:
        print(json.dumps(_json_safe(plan), indent=2, sort_keys=True))
    else:
        _print_summary(plan, artifacts)
    return 0 if plan.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
