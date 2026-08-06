#!/usr/bin/env python3
"""Remove manifest frame-clock declarations whose target file is absent.

The repair is intentionally narrow.  It does not create timestamp data or
change import status.  Once the broken declaration is removed, the acquisition
clock loader may discover a real conventional camera CSV or recording-level
Parquet index.  If neither exists, import records that the acquisition clock is
unavailable while retaining recording-only analysis support.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional

from fisheye.shared.batch_logging import make_run_id, utc_now
from fisheye.shared.json_safety import write_json_atomic


REPAIR_TYPE = "missing_frame_clock_declaration_v1"
TOOL_NAME = "fisheye.utils.repair_missing_frame_clock_declarations"


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload, sha256(raw).hexdigest()


def _discover_recording_dirs(paths: list[Path], *, recursive: bool) -> list[Path]:
    recordings: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if path.is_file() and path.name == "recording_manifest.json":
            candidates = [path.parent]
        elif (path / "recording_manifest.json").is_file():
            candidates = [path]
        elif path.is_dir() and recursive:
            candidates = sorted(
                manifest.parent for manifest in path.rglob("recording_manifest.json")
            )
        elif path.is_dir():
            candidates = sorted(
                child
                for child in path.iterdir()
                if (child / "recording_manifest.json").is_file()
            )
        else:
            candidates = []
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            recordings.append(resolved)
    return recordings


def _resolve_recording_path(recording_dir: Path, raw_value: object) -> Path:
    text = str(raw_value or "").strip()
    if not text:
        raise ValueError("frame-clock path is empty")
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    resolved = (recording_dir / candidate).resolve()
    if resolved != recording_dir and recording_dir not in resolved.parents:
        raise ValueError(f"recording-relative path escapes recording root: {raw_value!r}")
    return resolved


def _fallback_clock_source(
    recording_dir: Path,
    stream: Mapping[str, Any],
) -> dict[str, object]:
    raw_video = stream.get("video")
    if raw_video not in (None, ""):
        try:
            video = _resolve_recording_path(recording_dir, raw_video)
        except ValueError:
            video = None
        if video is not None:
            conventional = video.with_name(f"{video.stem}_meta.csv")
            if conventional.is_file():
                return {
                    "kind": "conventional_camera_metadata_csv",
                    "path": _recording_locator(recording_dir, conventional),
                }

    frame_index = recording_dir / "recording_frame_index.parquet"
    if frame_index.is_file():
        return {
            "kind": "recording_frame_index_parquet",
            "path": "recording_frame_index.parquet",
        }
    return {
        "kind": "none",
        "acquisition_frame_clock_status": "unavailable_no_camera_clock_source",
    }


def _recording_locator(recording_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(recording_dir).as_posix()
    except ValueError:
        return str(path.resolve())


def repair_recording_manifest(
    recording_dir: Path,
    *,
    apply: bool,
    repair_id: str,
    reason: str,
) -> dict[str, object]:
    root = recording_dir.expanduser().resolve()
    manifest_path = root / "recording_manifest.json"
    payload, manifest_sha256 = _load_manifest(manifest_path)

    video_streams = payload.get("video_streams")
    streams = video_streams.get("streams") if isinstance(video_streams, Mapping) else None
    if not isinstance(streams, dict):
        return {
            "recording": root.name,
            "recording_dir": str(root),
            "status": "unchanged_no_video_streams",
            "repaired_stream_count": 0,
        }

    repaired_streams: list[dict[str, object]] = []
    for stream_name, raw_stream in sorted(streams.items(), key=lambda item: str(item[0])):
        if not isinstance(raw_stream, dict):
            continue
        raw_declaration = raw_stream.get("frame_clock_metadata")
        if raw_declaration in (None, ""):
            continue
        declared_path = _resolve_recording_path(root, raw_declaration)
        if declared_path.is_file():
            continue
        repaired_streams.append(
            {
                "stream": str(stream_name),
                "removed_frame_clock_metadata": str(raw_declaration),
                "resolved_missing_path": str(declared_path),
                "fallback": _fallback_clock_source(root, raw_stream),
            }
        )
        del raw_stream["frame_clock_metadata"]

    if not repaired_streams:
        return {
            "recording": root.name,
            "recording_dir": str(root),
            "status": "unchanged_no_missing_declarations",
            "repaired_stream_count": 0,
        }

    status = "repair_planned"
    if apply:
        existing_repairs = payload.get("metadata_repairs")
        if existing_repairs is None:
            existing_repairs = []
            payload["metadata_repairs"] = existing_repairs
        if not isinstance(existing_repairs, list):
            raise ValueError(f"metadata_repairs is not a list: {manifest_path}")
        existing_repairs.append(
            {
                "repair_type": REPAIR_TYPE,
                "repair_id": repair_id,
                "created_at_utc": utc_now(),
                "tool": TOOL_NAME,
                "reason": reason,
                "manifest_sha256_before": manifest_sha256,
                "streams": repaired_streams,
            }
        )
        write_json_atomic(manifest_path, payload)
        status = "repaired"

    return {
        "recording": root.name,
        "recording_dir": str(root),
        "status": status,
        "repaired_stream_count": len(repaired_streams),
        "streams": repaired_streams,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recording_dirs",
        nargs="+",
        type=Path,
        help="Recording directories, manifest paths, or roots containing recordings.",
    )
    parser.add_argument("--apply", action="store_true", help="Atomically patch matching manifests.")
    parser.add_argument("--dry-run", action="store_true", help="Report matching repairs without writing.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover recording manifests below directory inputs.",
    )
    parser.add_argument(
        "--repair-id",
        default=f"missing_frame_clock_declaration_{make_run_id()}",
        help="Repair identifier written to manifest audit metadata.",
    )
    parser.add_argument(
        "--reason",
        default="declared frame-clock metadata was absent from the transferred recording",
        help="Human-readable reason written to manifest audit metadata.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if bool(args.apply) == bool(args.dry_run):
        print("Specify exactly one of --dry-run or --apply.", file=sys.stderr)
        return 1

    recordings = _discover_recording_dirs(args.recording_dirs, recursive=bool(args.recursive))
    if not recordings:
        print("No recording directories matched.")
        return 0

    exit_code = 0
    changed = 0
    for recording_dir in recordings:
        try:
            result = repair_recording_manifest(
                recording_dir,
                apply=bool(args.apply),
                repair_id=str(args.repair_id),
                reason=str(args.reason),
            )
        except Exception as exc:
            result = {
                "recording": recording_dir.name,
                "recording_dir": str(recording_dir),
                "status": "error",
                "error": str(exc),
            }
            exit_code = 1
        else:
            if result["status"] in {"repair_planned", "repaired"}:
                changed += 1
        print(json.dumps(result, sort_keys=True))

    print(
        json.dumps(
            {
                "exit_code": exit_code,
                "mode": "apply" if args.apply else "dry_run",
                "recordings_changed": changed,
                "recordings_scanned": len(recordings),
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
