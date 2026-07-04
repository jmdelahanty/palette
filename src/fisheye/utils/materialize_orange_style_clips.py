"""Materialize Orange-style rolling clips from a long camera video.

This utility consumes the same source bundle as ``plan_orange_style_clips`` and
creates a retroactive rolling-clip layout under the recording folder:

``clips/clip_000000/{Cam*.mp4,Cam*_meta.csv,Cam*_keyframe.json,clip_manifest.json}``

It is intentionally conservative: the command is dry-run by default and requires
``--apply`` before it writes clip artifacts.
"""

from __future__ import annotations

from fisheye.shared.batch_logging import utc_now as _utc_now
import argparse
import csv
import json
import os
import shlex
import shutil
import socket
import subprocess
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from fisheye.utils.plan_orange_style_clips import (
    DEFAULT_TARGET_DURATION_MINUTES,
    SUPPORTED_SNAP_DIRECTIONS,
    _json_safe,
    build_clip_plan,
    load_keyframe_summary,
    write_plan_artifacts,
)


MODULE_NAME = "fisheye.utils.materialize_orange_style_clips"
DEFAULT_INDEX_PREFIX = "recording_clip_index"

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class MaterializeOptions:
    output_recording_dir: Path
    ffmpeg_bin: str = "ffmpeg"
    apply: bool = False
    overwrite: bool = False
    max_clips: int | None = None


def _infer_recording_dir(video_path: Path) -> Path:
    for parent in video_path.resolve().parents:
        if parent.name == "cams":
            return parent.parent
    return video_path.resolve().parent


def _format_seconds(value: float) -> str:
    text = f"{float(value):.9f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _ensure_relative_path(root: Path, relative_path: str) -> Path:
    rel = Path(str(relative_path))
    if rel.is_absolute():
        raise ValueError(f"Clip artifact path must be relative: {relative_path}")
    resolved_root = root.resolve()
    resolved = (resolved_root / rel).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Clip artifact path escapes output root: {relative_path}")
    return resolved


def _atomic_write_text(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    _atomic_write_text(
        path,
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        overwrite=overwrite,
    )


def build_ffmpeg_stream_copy_command(
    *,
    ffmpeg_bin: str,
    source_video: Path,
    output_video: Path,
    start_time_s: float,
    frame_count: int,
    overwrite: bool,
) -> list[str]:
    """Build the stream-copy command for one keyframe-aligned clip."""
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    return [
        str(ffmpeg_bin),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y" if overwrite else "-n",
        "-ss",
        _format_seconds(start_time_s),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "copy",
        "-frames:v",
        str(int(frame_count)),
        "-reset_timestamps",
        "1",
        "-avoid_negative_ts",
        "make_zero",
        str(output_video),
    ]


def _run_subprocess(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _write_video_clip(
    *,
    command: Sequence[str],
    output_video: Path,
    overwrite: bool,
    runner: CommandRunner,
) -> dict[str, Any]:
    if output_video.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing video: {output_video}")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    tmp_video = output_video.with_name(f".{output_video.stem}.tmp{output_video.suffix}")
    if tmp_video.exists():
        if overwrite:
            tmp_video.unlink()
        else:
            raise FileExistsError(f"Temporary video already exists: {tmp_video}")

    tmp_command = list(command)
    tmp_command[-1] = str(tmp_video)
    started = time.perf_counter()
    result = runner(tmp_command)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg stream-copy failed for "
            f"{output_video} (exit {result.returncode})\n"
            f"command: {' '.join(shlex.quote(part) for part in tmp_command)}\n"
            f"stderr:\n{result.stderr}"
        )
    if not tmp_video.exists():
        raise RuntimeError(f"ffmpeg reported success but did not create {tmp_video}")
    os.replace(tmp_video, output_video)
    return {
        "output_video": str(output_video),
        "elapsed_s": float(elapsed),
        "returncode": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _clip_rows(plan: Mapping[str, Any], max_clips: int | None) -> list[dict[str, Any]]:
    rows = [dict(row) for row in list(plan.get("clips") or []) if isinstance(row, Mapping)]
    if max_clips is not None:
        rows = rows[: int(max_clips)]
    return rows


def _validate_plan_for_apply(plan: Mapping[str, Any], clip_rows: Sequence[Mapping[str, Any]]) -> None:
    if plan.get("status") != "ok":
        failures = [
            item
            for item in list(plan.get("checks") or [])
            if isinstance(item, Mapping) and item.get("status") != "ok"
        ]
        raise ValueError(f"Refusing to materialize failed clip plan: {failures[:5]}")
    if not clip_rows:
        raise ValueError("Clip plan contains no clips to materialize")
    for row in clip_rows:
        if not row.get("start_is_keyframe"):
            raise ValueError(f"Refusing non-keyframe clip start: {row.get('clip_id')}")


def _preflight_output_collisions(
    *,
    root: Path,
    clip_rows: Sequence[Mapping[str, Any]],
    overwrite: bool,
) -> None:
    if overwrite:
        return
    relative_keys = ("video_path", "metadata_path", "keyframe_path", "clip_manifest_path")
    targets: list[Path] = [
        root / f"{DEFAULT_INDEX_PREFIX}.json",
        root / f"{DEFAULT_INDEX_PREFIX}.csv",
    ]
    for row in clip_rows:
        for key in relative_keys:
            targets.append(_ensure_relative_path(root, str(row[key])))
    existing = [path for path in targets if path.exists()]
    if existing:
        sample = "\n".join(str(path) for path in existing[:10])
        suffix = "" if len(existing) <= 10 else f"\n... {len(existing) - 10} more"
        raise FileExistsError(
            "Refusing to overwrite existing clip artifacts without --overwrite:\n"
            f"{sample}{suffix}"
        )


def _write_metadata_slices(
    *,
    source_metadata_csv: Path,
    root: Path,
    clip_rows: Sequence[Mapping[str, Any]],
    overwrite: bool,
) -> dict[str, Any]:
    ordered = sorted(clip_rows, key=lambda row: int(row["actual_start_frame"]))
    outputs: list[dict[str, Any]] = []
    with source_metadata_csv.open("r", newline="", encoding="utf-8") as src_fh:
        reader = csv.DictReader(src_fh)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata CSV has no header: {source_metadata_csv}")
        with ExitStack() as stack:
            writers: list[tuple[Mapping[str, Any], csv.DictWriter[str], Any]] = []
            for row in ordered:
                path = _ensure_relative_path(root, str(row["metadata_path"]))
                if path.exists() and not overwrite:
                    raise FileExistsError(f"Refusing to overwrite existing metadata CSV: {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                fh = stack.enter_context(path.open("w", newline="", encoding="utf-8"))
                writer = csv.DictWriter(fh, fieldnames=reader.fieldnames)
                writer.writeheader()
                writers.append((row, writer, path))

            active_index = 0
            written_by_clip = {str(row["clip_id"]): 0 for row in ordered}
            for row_index, metadata_row in enumerate(reader):
                while (
                    active_index < len(writers)
                    and row_index >= int(writers[active_index][0]["end_frame_exclusive"])
                ):
                    active_index += 1
                if active_index >= len(writers):
                    break
                clip_row, writer, _path = writers[active_index]
                start = int(clip_row["actual_start_frame"])
                end_exclusive = int(clip_row["end_frame_exclusive"])
                if start <= row_index < end_exclusive:
                    writer.writerow(metadata_row)
                    written_by_clip[str(clip_row["clip_id"])] += 1

            for row, _writer, path in writers:
                clip_id = str(row["clip_id"])
                outputs.append(
                    {
                        "clip_id": clip_id,
                        "metadata_path": str(path),
                        "rows_written": int(written_by_clip[clip_id]),
                        "expected_rows": int(row["frame_count"]),
                    }
                )
                if written_by_clip[clip_id] != int(row["frame_count"]):
                    raise RuntimeError(
                        f"Metadata slice row count mismatch for {clip_id}: "
                        f"wrote {written_by_clip[clip_id]}, expected {row['frame_count']}"
                    )
    return {"outputs": outputs}


def _clip_keyframe_payload(
    *,
    source_payload: Mapping[str, Any],
    source_keyframes: np.ndarray,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    start = int(row["actual_start_frame"])
    end_exclusive = int(row["end_frame_exclusive"])
    shifted = source_keyframes[(source_keyframes >= start) & (source_keyframes < end_exclusive)] - start
    payload = dict(source_payload)
    payload["total_frames"] = int(row["frame_count"])
    payload["keyframe_frames"] = [int(value) for value in shifted.tolist()]
    payload["palette_retro_clip"] = {
        "created_at_utc": _utc_now(),
        "tool": MODULE_NAME,
        "source_keyframe_path": str(row.get("source_keyframe_path")),
        "source_start_frame": int(start),
        "source_end_frame_exclusive": int(end_exclusive),
        "recording_id": row.get("recording_id"),
        "clip_id": row.get("clip_id"),
        "clip_index": row.get("clip_index"),
    }
    return payload


def _write_keyframe_slices(
    *,
    source_keyframe_json: Path,
    root: Path,
    clip_rows: Sequence[Mapping[str, Any]],
    overwrite: bool,
) -> dict[str, Any]:
    summary = load_keyframe_summary(source_keyframe_json)
    outputs: list[dict[str, Any]] = []
    for row in clip_rows:
        path = _ensure_relative_path(root, str(row["keyframe_path"]))
        payload = _clip_keyframe_payload(
            source_payload=summary.payload,
            source_keyframes=summary.keyframe_frames,
            row=row,
        )
        if not payload["keyframe_frames"] or int(payload["keyframe_frames"][0]) != 0:
            raise RuntimeError(f"Clip keyframe sidecar does not start at frame 0: {row['clip_id']}")
        _atomic_write_json(path, payload, overwrite=overwrite)
        outputs.append(
            {
                "clip_id": row["clip_id"],
                "keyframe_path": str(path),
                "keyframe_count": len(payload["keyframe_frames"]),
                "total_frames": int(payload["total_frames"]),
            }
        )
    return {"outputs": outputs}


def _clip_manifest_payload(
    *,
    plan: Mapping[str, Any],
    row: Mapping[str, Any],
    root: Path,
    ffmpeg_command: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": "palette.orange_style_clip_manifest.v1",
        "generated_by": MODULE_NAME,
        "generated_at_utc": _utc_now(),
        "recording_id": row.get("recording_id"),
        "session_id": row.get("session_id"),
        "clip_index": row.get("clip_index"),
        "clip_id": row.get("clip_id"),
        "clip_directory": row.get("clip_directory"),
        "producer": "palette_retro_orange_style_stream_copy",
        "source": {
            "video_path": plan.get("video_path"),
            "metadata_csv": plan.get("metadata_csv"),
            "keyframe_json": plan.get("keyframe_json"),
            "source_total_frames": (plan.get("source") or {}).get("total_frames")
            if isinstance(plan.get("source"), Mapping)
            else None,
        },
        "camera_artifacts": [
            {
                "camera_serial": row.get("camera_serial"),
                "video_path": row.get("video_path"),
                "metadata_path": row.get("metadata_path"),
                "keyframe_path": row.get("keyframe_path"),
                "frame_count": row.get("frame_count"),
                "first_recording_frame_id": row.get("first_recording_frame_id"),
                "last_recording_frame_id": row.get("last_recording_frame_id"),
                "first_clip_local_frame_index": row.get("first_clip_local_frame_index"),
                "last_clip_local_frame_index": row.get("last_clip_local_frame_index"),
                "first_timestamp": row.get("first_timestamp"),
                "last_timestamp": row.get("last_timestamp"),
                "first_timestamp_sys": row.get("first_timestamp_sys"),
                "last_timestamp_sys": row.get("last_timestamp_sys"),
            }
        ],
        "stream_copy": {
            "ffmpeg_command": " ".join(shlex.quote(str(part)) for part in ffmpeg_command),
            "start_time_s": row.get("actual_start_time_s"),
            "frame_count": row.get("frame_count"),
            "keyframe_aligned": row.get("start_is_keyframe"),
        },
        "output_recording_dir": str(root),
    }


def _write_clip_manifests(
    *,
    plan: Mapping[str, Any],
    root: Path,
    clip_rows: Sequence[Mapping[str, Any]],
    ffmpeg_commands_by_clip: Mapping[str, Sequence[str]],
    overwrite: bool,
) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    for row in clip_rows:
        path = _ensure_relative_path(root, str(row["clip_manifest_path"]))
        payload = _clip_manifest_payload(
            plan=plan,
            row=row,
            root=root,
            ffmpeg_command=ffmpeg_commands_by_clip[str(row["clip_id"])],
        )
        _atomic_write_json(path, payload, overwrite=overwrite)
        outputs.append({"clip_id": row["clip_id"], "clip_manifest_path": str(path)})
    return {"outputs": outputs}


def _materialized_index(plan: Mapping[str, Any], clip_rows: Sequence[Mapping[str, Any]], root: Path) -> dict[str, Any]:
    clip_ids = {str(row["clip_id"]) for row in clip_rows}
    clips: list[dict[str, Any]] = []
    for row in list(plan.get("clips") or []):
        if not isinstance(row, Mapping) or str(row.get("clip_id")) not in clip_ids:
            continue
        materialized = dict(row)
        materialized["status"] = "materialized"
        materialized["materialized_at_utc"] = _utc_now()
        clips.append(materialized)
    index = dict(plan)
    index["generated_by"] = MODULE_NAME
    index["mode"] = "materialized_stream_copy"
    index["output_recording_dir"] = str(root)
    index["clip_count"] = len(clips)
    index["clips"] = clips
    return index


def materialize_clip_plan(
    plan: Mapping[str, Any],
    *,
    options: MaterializeOptions,
    runner: CommandRunner = _run_subprocess,
) -> dict[str, Any]:
    """Dry-run or materialize a clip plan.

    In dry-run mode no files are written. In apply mode, video clips are written
    first, then sidecars, then the root recording clip index.
    """
    started = time.perf_counter()
    root = Path(options.output_recording_dir).expanduser().resolve()
    rows = _clip_rows(plan, options.max_clips)
    _validate_plan_for_apply(plan, rows)
    source_video = Path(str(plan["video_path"])).expanduser().resolve()
    source_metadata = Path(str(plan["metadata_csv"])).expanduser().resolve()
    source_keyframe = Path(str(plan["keyframe_json"])).expanduser().resolve()
    if options.apply:
        _preflight_output_collisions(root=root, clip_rows=rows, overwrite=options.overwrite)

    ffmpeg_commands_by_clip: dict[str, list[str]] = {}
    video_outputs: list[dict[str, Any]] = []
    for row in rows:
        output_video = _ensure_relative_path(root, str(row["video_path"]))
        command = build_ffmpeg_stream_copy_command(
            ffmpeg_bin=options.ffmpeg_bin,
            source_video=source_video,
            output_video=output_video,
            start_time_s=float(row["actual_start_time_s"]),
            frame_count=int(row["frame_count"]),
            overwrite=options.overwrite,
        )
        ffmpeg_commands_by_clip[str(row["clip_id"])] = command
        if options.apply:
            video_outputs.append(
                _write_video_clip(
                    command=command,
                    output_video=output_video,
                    overwrite=options.overwrite,
                    runner=runner,
                )
            )

    metadata_result: dict[str, Any] | None = None
    keyframe_result: dict[str, Any] | None = None
    manifest_result: dict[str, Any] | None = None
    index_artifacts: dict[str, str] | None = None
    if options.apply:
        metadata_result = _write_metadata_slices(
            source_metadata_csv=source_metadata,
            root=root,
            clip_rows=rows,
            overwrite=options.overwrite,
        )
        keyframe_result = _write_keyframe_slices(
            source_keyframe_json=source_keyframe,
            root=root,
            clip_rows=rows,
            overwrite=options.overwrite,
        )
        manifest_result = _write_clip_manifests(
            plan=plan,
            root=root,
            clip_rows=rows,
            ffmpeg_commands_by_clip=ffmpeg_commands_by_clip,
            overwrite=options.overwrite,
        )
        index_plan = _materialized_index(plan, rows, root)
        index_artifacts = write_plan_artifacts(index_plan, root, prefix=DEFAULT_INDEX_PREFIX)

    return {
        "status": "ok",
        "generated_by": MODULE_NAME,
        "generated_at_utc": _utc_now(),
        "host": socket.gethostname(),
        "pid": int(os.getpid()),
        "apply": bool(options.apply),
        "overwrite": bool(options.overwrite),
        "output_recording_dir": str(root),
        "clip_count": int(len(rows)),
        "source_video": str(source_video),
        "source_metadata_csv": str(source_metadata),
        "source_keyframe_json": str(source_keyframe),
        "ffmpeg_bin": str(options.ffmpeg_bin),
        "ffmpeg_available": shutil.which(str(options.ffmpeg_bin)) is not None,
        "ffmpeg_commands": {
            clip_id: " ".join(shlex.quote(str(part)) for part in command)
            for clip_id, command in ffmpeg_commands_by_clip.items()
        },
        "video_outputs": video_outputs,
        "metadata_slices": metadata_result,
        "keyframe_slices": keyframe_result,
        "clip_manifests": manifest_result,
        "recording_clip_index": index_artifacts,
        "duration_seconds": float(time.perf_counter() - started),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create Orange-style rolling clip artifacts from a long camera MP4. "
            "Dry-run by default; pass --apply to write clips and sidecars."
        )
    )
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--metadata-csv", type=Path, help="Cam*_meta.csv path; inferred from video name if omitted.")
    parser.add_argument("--keyframe-json", type=Path, help="Cam*_keyframe.json path; inferred from video name if omitted.")
    parser.add_argument("--output-recording-dir", type=Path, help="Recording root to receive clips/. Defaults to parent of cams/.")
    parser.add_argument("--target-duration-minutes", type=float, default=DEFAULT_TARGET_DURATION_MINUTES)
    parser.add_argument("--snap-direction", choices=SUPPORTED_SNAP_DIRECTIONS, default="next")
    parser.add_argument("--recording-id")
    parser.add_argument("--camera-serial")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--max-clips", type=int, help="Materialize only the first N clips for smoke testing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing clip artifacts.")
    parser.add_argument("--apply", action="store_true", help="Write clips and sidecars. Without this, only prints the plan.")
    parser.add_argument("--json", action="store_true", help="Print full JSON result.")
    return parser


def _print_summary(result: Mapping[str, Any]) -> None:
    print(f"status: {result.get('status')}")
    print(f"apply: {result.get('apply')}")
    print(f"output_recording_dir: {result.get('output_recording_dir')}")
    print(f"clip_count: {result.get('clip_count')}")
    print(f"ffmpeg_bin: {result.get('ffmpeg_bin')} available={result.get('ffmpeg_available')}")
    if not result.get("apply"):
        print("dry_run: no files written; pass --apply to materialize clips")
    commands = result.get("ffmpeg_commands")
    if isinstance(commands, Mapping):
        for clip_id, command in list(commands.items())[:3]:
            print(f"  {clip_id}: {command}")
        remaining = len(commands) - 3
        if remaining > 0:
            print(f"  ... {remaining} more ffmpeg commands")
    index = result.get("recording_clip_index")
    if isinstance(index, Mapping):
        print(f"recording_clip_index_json: {index.get('json')}")
        print(f"recording_clip_index_csv: {index.get('csv')}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    video_path = Path(args.video_path).expanduser().resolve()
    output_root = (
        Path(args.output_recording_dir).expanduser().resolve()
        if args.output_recording_dir is not None
        else _infer_recording_dir(video_path)
    )
    plan = build_clip_plan(
        video_path=video_path,
        metadata_csv=args.metadata_csv,
        keyframe_json=args.keyframe_json,
        target_duration_minutes=args.target_duration_minutes,
        snap_direction=args.snap_direction,
        recording_id=args.recording_id,
        camera_serial=args.camera_serial,
    )
    result = materialize_clip_plan(
        plan,
        options=MaterializeOptions(
            output_recording_dir=output_root,
            ffmpeg_bin=str(args.ffmpeg_bin),
            apply=bool(args.apply),
            overwrite=bool(args.overwrite),
            max_clips=args.max_clips,
        ),
    )
    if args.json:
        print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    else:
        _print_summary(result)
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
