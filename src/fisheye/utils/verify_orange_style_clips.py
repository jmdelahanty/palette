"""Verify Orange-style rolling clip artifacts.

The verifier checks the root ``recording_clip_index.json`` plus each per-clip
video/metadata/keyframe/manifest artifact. Video packet counting is optional
because it requires ffprobe to scan each clip.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from fisheye.utils.plan_orange_style_clips import _json_safe


MODULE_NAME = "fisheye.utils.verify_orange_style_clips"
DEFAULT_INDEX_NAME = "recording_clip_index.json"

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
ProgressCallback = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class VerifyOptions:
    recording_dir: Path
    index_json: Path
    ffprobe_bin: str = "ffprobe"
    probe_video: bool = False
    max_clips: int | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _resolve_relative(root: Path, value: Any) -> Path:
    rel = Path(str(value))
    if rel.is_absolute():
        return rel
    resolved_root = root.resolve()
    resolved = (resolved_root / rel).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Path escapes recording root: {value}")
    return resolved


def _check(checks: list[dict[str, Any]], *, status: str, code: str, **details: Any) -> None:
    checks.append({"status": status, "code": code, **details})


def _coerce_int(value: Any) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metadata_stats(path: Path) -> dict[str, Any]:
    row_count = 0
    first_frame_id: int | None = None
    last_frame_id: int | None = None
    gap_count = 0
    previous_frame_id: int | None = None
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        if "frame_id" not in fieldnames and "recording_frame_id" not in fieldnames:
            raise ValueError(f"Metadata CSV has no frame_id or recording_frame_id column: {path}")
        frame_key = "recording_frame_id" if "recording_frame_id" in fieldnames else "frame_id"
        for row_number, row in enumerate(reader, start=2):
            frame_id = _coerce_int(row.get(frame_key))
            if frame_id is None:
                raise ValueError(f"Invalid {frame_key} at {path}:{row_number}: {row.get(frame_key)!r}")
            if first_frame_id is None:
                first_frame_id = frame_id
            if previous_frame_id is not None and frame_id != previous_frame_id + 1:
                gap_count += 1
            previous_frame_id = frame_id
            last_frame_id = frame_id
            row_count += 1
    return {
        "row_count": int(row_count),
        "first_frame_id": first_frame_id,
        "last_frame_id": last_frame_id,
        "recording_frame_id_gaps": int(gap_count),
        "fieldnames": fieldnames,
    }


def _run_subprocess(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _probe_packet_count(
    *,
    path: Path,
    ffprobe_bin: str,
    runner: CommandRunner,
) -> dict[str, Any]:
    command = [
        str(ffprobe_bin),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets,nb_frames,avg_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    result = runner(command)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffprobe failed with exit code {result.returncode}")
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise RuntimeError("ffprobe returned no video stream")
    stream = streams[0] if isinstance(streams[0], dict) else {}
    return {
        "command": command,
        "nb_read_packets": _coerce_int(stream.get("nb_read_packets")),
        "nb_frames": _coerce_int(stream.get("nb_frames")),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "duration": stream.get("duration"),
    }


def _verify_clip(
    *,
    root: Path,
    row: Mapping[str, Any],
    options: VerifyOptions,
    runner: CommandRunner,
    progress: ProgressCallback | None = None,
    ordinal: int | None = None,
    total: int | None = None,
) -> dict[str, Any]:
    clip_id = str(row.get("clip_id"))
    expected_frame_count = _coerce_int(row.get("frame_count"))
    checks: list[dict[str, Any]] = []
    paths = {
        "video": _resolve_relative(root, row.get("video_path")),
        "metadata": _resolve_relative(root, row.get("metadata_path")),
        "keyframe": _resolve_relative(root, row.get("keyframe_path")),
        "clip_manifest": _resolve_relative(root, row.get("clip_manifest_path")),
    }
    for kind, path in paths.items():
        _check(
            checks,
            status="ok" if path.exists() else "fail",
            code=f"{kind}_exists",
            path=str(path),
        )

    metadata_stats: dict[str, Any] | None = None
    if paths["metadata"].exists():
        try:
            metadata_stats = _metadata_stats(paths["metadata"])
            _check(
                checks,
                status="ok" if expected_frame_count == metadata_stats["row_count"] else "fail",
                code="metadata_row_count_matches_index",
                expected=expected_frame_count,
                observed=metadata_stats["row_count"],
            )
            _check(
                checks,
                status="ok" if metadata_stats["first_frame_id"] == _coerce_int(row.get("first_recording_frame_id")) else "fail",
                code="metadata_first_frame_id_matches_index",
                expected=_coerce_int(row.get("first_recording_frame_id")),
                observed=metadata_stats["first_frame_id"],
            )
            _check(
                checks,
                status="ok" if metadata_stats["last_frame_id"] == _coerce_int(row.get("last_recording_frame_id")) else "fail",
                code="metadata_last_frame_id_matches_index",
                expected=_coerce_int(row.get("last_recording_frame_id")),
                observed=metadata_stats["last_frame_id"],
            )
            _check(
                checks,
                status="ok" if metadata_stats["recording_frame_id_gaps"] == 0 else "fail",
                code="metadata_recording_frame_id_continuity",
                recording_frame_id_gaps=metadata_stats["recording_frame_id_gaps"],
            )
        except Exception as exc:
            _check(checks, status="fail", code="metadata_read_error", error=str(exc))

    keyframe_payload: dict[str, Any] | None = None
    if paths["keyframe"].exists():
        try:
            keyframe_payload = _load_json(paths["keyframe"])
            keyframes = keyframe_payload.get("keyframe_frames")
            total_frames = _coerce_int(keyframe_payload.get("total_frames"))
            keyframes_list = keyframes if isinstance(keyframes, list) else []
            _check(
                checks,
                status="ok" if total_frames == expected_frame_count else "fail",
                code="keyframe_total_frames_matches_index",
                expected=expected_frame_count,
                observed=total_frames,
            )
            _check(
                checks,
                status="ok" if keyframes_list and _coerce_int(keyframes_list[0]) == 0 else "fail",
                code="keyframe_starts_at_zero",
                first_keyframe=keyframes_list[0] if keyframes_list else None,
            )
            invalid_keyframes = [
                value
                for value in keyframes_list
                if _coerce_int(value) is None
                or expected_frame_count is None
                or _coerce_int(value) < 0
                or _coerce_int(value) >= expected_frame_count
            ]
            _check(
                checks,
                status="ok" if not invalid_keyframes else "fail",
                code="keyframes_within_clip_bounds",
                invalid_count=len(invalid_keyframes),
            )
        except Exception as exc:
            _check(checks, status="fail", code="keyframe_read_error", error=str(exc))

    if paths["clip_manifest"].exists():
        try:
            manifest = _load_json(paths["clip_manifest"])
            _check(
                checks,
                status="ok" if str(manifest.get("clip_id")) == clip_id else "fail",
                code="clip_manifest_clip_id_matches_index",
                expected=clip_id,
                observed=manifest.get("clip_id"),
            )
        except Exception as exc:
            _check(checks, status="fail", code="clip_manifest_read_error", error=str(exc))

    probe: dict[str, Any] | None = None
    if options.probe_video and paths["video"].exists():
        try:
            probe_started = time.perf_counter()
            if progress is not None:
                progress(
                    {
                        "event": "video_probe_start",
                        "clip_id": clip_id,
                        "clip_index": _coerce_int(row.get("clip_index")),
                        "ordinal": ordinal,
                        "total": total,
                        "path": str(paths["video"]),
                        "expected_frame_count": expected_frame_count,
                    }
                )
            probe = _probe_packet_count(path=paths["video"], ffprobe_bin=options.ffprobe_bin, runner=runner)
            probe_elapsed = time.perf_counter() - probe_started
            probe["elapsed_s"] = float(probe_elapsed)
            observed = probe.get("nb_read_packets")
            if observed is None:
                observed = probe.get("nb_frames")
            if progress is not None:
                progress(
                    {
                        "event": "video_probe_done",
                        "clip_id": clip_id,
                        "clip_index": _coerce_int(row.get("clip_index")),
                        "ordinal": ordinal,
                        "total": total,
                        "path": str(paths["video"]),
                        "expected_frame_count": expected_frame_count,
                        "observed_frame_count": observed,
                        "elapsed_s": float(probe_elapsed),
                    }
                )
            _check(
                checks,
                status="ok" if observed == expected_frame_count else "fail",
                code="video_packet_count_matches_index",
                expected=expected_frame_count,
                observed=observed,
            )
        except Exception as exc:
            _check(checks, status="fail", code="video_probe_error", error=str(exc))

    status = "ok" if all(check["status"] == "ok" for check in checks) else "fail"
    return {
        "status": status,
        "clip_id": clip_id,
        "clip_index": _coerce_int(row.get("clip_index")),
        "expected_frame_count": expected_frame_count,
        "paths": {key: str(value) for key, value in paths.items()},
        "metadata_stats": metadata_stats,
        "keyframe_total_frames": keyframe_payload.get("total_frames") if keyframe_payload else None,
        "video_probe": probe,
        "checks": checks,
    }


def verify_recording_clips(
    options: VerifyOptions,
    *,
    runner: CommandRunner = _run_subprocess,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(options.recording_dir).expanduser().resolve()
    index_path = Path(options.index_json).expanduser().resolve()
    checks: list[dict[str, Any]] = []
    _check(checks, status="ok" if root.exists() else "fail", code="recording_dir_exists", path=str(root))
    _check(checks, status="ok" if index_path.exists() else "fail", code="recording_clip_index_exists", path=str(index_path))
    if not index_path.exists():
        return {
            "status": "fail",
            "generated_by": MODULE_NAME,
            "generated_at_utc": _utc_now(),
            "recording_dir": str(root),
            "index_json": str(index_path),
            "checks": checks,
            "clips": [],
        }

    index = _load_json(index_path)
    rows = [row for row in list(index.get("clips") or []) if isinstance(row, Mapping)]
    if options.max_clips is not None:
        rows = rows[: int(options.max_clips)]
    expected_clip_count = _coerce_int(index.get("clip_count"))
    if options.max_clips is None:
        _check(
            checks,
            status="ok" if expected_clip_count == len(rows) else "fail",
            code="index_clip_count_matches_rows",
            expected=expected_clip_count,
            observed=len(rows),
        )

    clip_results = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        clip_results.append(
            _verify_clip(
                root=root,
                row=row,
                options=options,
                runner=runner,
                progress=progress,
                ordinal=index,
                total=total,
            )
        )
    failed_clips = [row for row in clip_results if row["status"] != "ok"]
    status = "ok" if all(check["status"] == "ok" for check in checks) and not failed_clips else "fail"
    return {
        "status": status,
        "generated_by": MODULE_NAME,
        "generated_at_utc": _utc_now(),
        "host": socket.gethostname(),
        "pid": int(os.getpid()),
        "recording_dir": str(root),
        "index_json": str(index_path),
        "probe_video": bool(options.probe_video),
        "ffprobe_bin": str(options.ffprobe_bin),
        "clip_count": int(len(clip_results)),
        "failed_clip_count": int(len(failed_clips)),
        "checks": checks,
        "clips": clip_results,
        "duration_seconds": float(time.perf_counter() - started),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify Orange-style rolling clip artifacts.")
    parser.add_argument("recording_dir", type=Path)
    parser.add_argument("--index-json", type=Path, help="Defaults to <recording_dir>/recording_clip_index.json.")
    parser.add_argument("--probe-video", action="store_true", help="Use ffprobe -count_packets to validate clip videos.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable stderr progress lines while probing video packets.",
    )
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--max-clips", type=int, help="Verify only the first N clips.")
    parser.add_argument("--output-json", type=Path, help="Write full verification JSON.")
    parser.add_argument("--json", action="store_true", help="Print full verification JSON.")
    return parser


def _print_summary(result: Mapping[str, Any]) -> None:
    print(f"status: {result.get('status')}")
    print(f"recording_dir: {result.get('recording_dir')}")
    print(f"index_json: {result.get('index_json')}")
    print(f"probe_video: {result.get('probe_video')}")
    print(f"clips: {result.get('clip_count')} failed={result.get('failed_clip_count')}")
    for row in list(result.get("clips") or [])[:10]:
        if not isinstance(row, Mapping):
            continue
        print(
            f"  {row.get('clip_id')}: status={row.get('status')} "
            f"expected_frames={row.get('expected_frame_count')}"
        )
        failures = [
            check for check in list(row.get("checks") or [])
            if isinstance(check, Mapping) and check.get("status") != "ok"
        ]
        for failure in failures[:3]:
            print(f"    fail {failure.get('code')}: {failure}")
    remaining = int(result.get("clip_count") or 0) - 10
    if remaining > 0:
        print(f"  ... {remaining} more clips")


def _print_progress(event: Mapping[str, Any]) -> None:
    kind = event.get("event")
    ordinal = event.get("ordinal")
    total = event.get("total")
    clip_id = event.get("clip_id")
    prefix = f"[{ordinal}/{total}] {clip_id}" if ordinal and total else str(clip_id)
    if kind == "video_probe_start":
        print(
            f"{prefix} ffprobe_count_packets start expected={event.get('expected_frame_count')}",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "video_probe_done":
        elapsed = float(event.get("elapsed_s") or 0.0)
        print(
            f"{prefix} ffprobe_count_packets done observed={event.get('observed_frame_count')} "
            f"expected={event.get('expected_frame_count')} elapsed_s={elapsed:.2f}",
            file=sys.stderr,
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    recording_dir = Path(args.recording_dir).expanduser().resolve()
    index_json = (
        Path(args.index_json).expanduser().resolve()
        if args.index_json is not None
        else recording_dir / DEFAULT_INDEX_NAME
    )
    result = verify_recording_clips(
        VerifyOptions(
            recording_dir=recording_dir,
            index_json=index_json,
            ffprobe_bin=str(args.ffprobe_bin),
            probe_video=bool(args.probe_video),
            max_clips=args.max_clips,
        ),
        progress=_print_progress if args.probe_video and not args.no_progress else None,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    else:
        _print_summary(result)
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
