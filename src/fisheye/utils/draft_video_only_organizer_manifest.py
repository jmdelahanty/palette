#!/usr/bin/env python3
"""Draft the CSV consumed by organize_recordings --video-only.

The output is an operator-reviewed manifest: one row per camera-video recording.
It is intentionally a CSV so users can fill known metadata before applying the
organizer.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Optional


VIDEO_ONLY_ORGANIZER_MANIFEST_FIELDS = (
    "source_video",
    "source_camera_metadata_csv",
    "camera_id",
    "session_uuid",
    "recording_id",
    "recording_name",
    "session_start_iso8601_utc",
    "recording_type",
    "recording_subtype",
    "behavior_mode",
    "artifact_schema_id",
    "dish_design",
    "rig_id",
    "arena_id",
    "canvas_name",
    "protocol_name",
    "protocol_name_from_definition",
    "genotype",
    "dpf_at_acquisition",
    "num_dishes",
    "fish_per_dish",
)

PROMPTABLE_FIELDS = (
    "dish_design",
    "rig_id",
    "arena_id",
    "canvas_name",
    "protocol_name",
    "genotype",
    "dpf_at_acquisition",
    "num_dishes",
    "fish_per_dish",
)


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "video_only_recording"


def _derive_camera_id(path: Path) -> Optional[str]:
    match = re.search(r"Cam(\d+)", path.name, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    digits = re.findall(r"\d+", path.stem)
    return digits[-1] if digits else None


def _load_snapshot(source_root: Path) -> dict[str, Any]:
    snapshot_path = source_root / "recording_snapshot.json"
    if not snapshot_path.exists():
        return {}
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _snapshot_session_start(snapshot: dict[str, Any]) -> str:
    timestamp = snapshot.get("timestamp_utc")
    if isinstance(timestamp, str) and timestamp.strip():
        return timestamp.strip()
    sync = snapshot.get("sync")
    if isinstance(sync, dict):
        captured_at = sync.get("captured_at_utc")
        if isinstance(captured_at, str) and captured_at.strip():
            return captured_at.strip()
    return ""


def _snapshot_recording_id(snapshot: dict[str, Any], source_root: Path) -> str:
    recording_id = snapshot.get("recording_id")
    if isinstance(recording_id, str) and recording_id.strip():
        return _slugify(recording_id)
    return _slugify(source_root.name)


def _discover_videos(source_root: Path, *, pattern: str, recursive: bool) -> list[Path]:
    iterator: Iterable[Path] = source_root.rglob(pattern) if recursive else source_root.glob(pattern)
    videos = sorted(path for path in iterator if path.is_file())
    return videos


def _relative_or_absolute(path: Path, *, source_root: Path, absolute_paths: bool) -> str:
    resolved = path.resolve()
    if absolute_paths:
        return str(resolved)
    try:
        return resolved.relative_to(source_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _find_camera_metadata_csv(video_path: Path, camera_id: Optional[str]) -> Optional[Path]:
    candidates: list[Path] = []
    if camera_id:
        candidates.append(video_path.with_name(f"Cam{camera_id}_meta.csv"))
    candidates.append(video_path.with_name(f"{video_path.stem}_meta.csv"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _format_template(template: str, *, base_recording_id: str, camera_id: str, video_path: Path) -> str:
    return _slugify(
        template.format(
            recording_id=base_recording_id,
            camera_id=camera_id,
            video_stem=video_path.stem,
        )
    )


def _prompt_for_defaults(defaults: dict[str, str]) -> dict[str, str]:
    if not sys.stdin.isatty():
        raise RuntimeError("--prompt-metadata requires an interactive terminal")
    prompted = dict(defaults)
    for field in PROMPTABLE_FIELDS:
        if prompted.get(field):
            continue
        value = input(f"{field} (blank if unknown): ").strip()
        if value:
            prompted[field] = value
    return prompted


def _build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    source_root = args.source_root.expanduser().resolve()
    snapshot = _load_snapshot(source_root)
    base_recording_id = _slugify(args.recording_id or _snapshot_recording_id(snapshot, source_root))
    session_start = args.session_start_utc or _snapshot_session_start(snapshot)

    defaults = {
        "recording_type": args.recording_type,
        "recording_subtype": args.recording_subtype,
        "behavior_mode": args.behavior_mode,
        "artifact_schema_id": args.artifact_schema_id,
        "dish_design": args.dish_design or "",
        "rig_id": args.rig_id or "",
        "arena_id": args.arena_id or "",
        "canvas_name": args.canvas_name or "",
        "protocol_name": args.protocol_name or "",
        "protocol_name_from_definition": args.protocol_name_from_definition or args.protocol_name or "",
        "genotype": args.genotype or "",
        "dpf_at_acquisition": str(args.dpf_at_acquisition) if args.dpf_at_acquisition is not None else "",
        "num_dishes": str(args.num_dishes) if args.num_dishes is not None else "",
        "fish_per_dish": str(args.fish_per_dish) if args.fish_per_dish is not None else "",
    }
    if args.prompt_metadata:
        defaults = _prompt_for_defaults(defaults)
        if defaults.get("protocol_name") and not defaults.get("protocol_name_from_definition"):
            defaults["protocol_name_from_definition"] = defaults["protocol_name"]

    videos = _discover_videos(source_root, pattern=args.video_glob, recursive=bool(args.recursive))
    if not videos:
        raise FileNotFoundError(f"No videos matched {args.video_glob!r} under {source_root}")

    rows: list[dict[str, str]] = []
    for video_path in videos:
        camera_id = args.camera_id or _derive_camera_id(video_path) or ""
        metadata_csv = _find_camera_metadata_csv(video_path, camera_id)
        if args.require_camera_metadata_csv and metadata_csv is None:
            raise FileNotFoundError(f"No camera metadata CSV found for {video_path}")

        template_context_camera = camera_id or _slugify(video_path.stem)
        session_uuid = (
            _format_template(
                args.session_uuid_template,
                base_recording_id=base_recording_id,
                camera_id=template_context_camera,
                video_path=video_path,
            )
            if args.session_uuid_template
            else f"{base_recording_id}_cam{template_context_camera}"
        )
        recording_name = (
            _format_template(
                args.recording_name_template,
                base_recording_id=base_recording_id,
                camera_id=template_context_camera,
                video_path=video_path,
            )
            if args.recording_name_template
            else session_uuid
        )

        row = {field: "" for field in VIDEO_ONLY_ORGANIZER_MANIFEST_FIELDS}
        row.update(defaults)
        row["source_video"] = _relative_or_absolute(
            video_path,
            source_root=source_root,
            absolute_paths=bool(args.absolute_paths),
        )
        if metadata_csv is not None:
            row["source_camera_metadata_csv"] = _relative_or_absolute(
                metadata_csv,
                source_root=source_root,
                absolute_paths=bool(args.absolute_paths),
            )
        row["camera_id"] = camera_id
        row["session_uuid"] = session_uuid
        row["recording_id"] = base_recording_id
        row["recording_name"] = recording_name
        row["session_start_iso8601_utc"] = session_start
        rows.append(row)
    return rows


def _write_rows(rows: list[dict[str, str]], output: Optional[Path], *, overwrite: bool) -> None:
    if output is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=VIDEO_ONLY_ORGANIZER_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        return

    output = output.expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}. Use --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=VIDEO_ONLY_ORGANIZER_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} row(s): {output}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path, help="Staging directory containing camera videos.")
    parser.add_argument("--output", type=Path, help="CSV path to write. Defaults to stdout.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite --output if it exists.")
    parser.add_argument("--video-glob", default="Cam*.mp4", help="Video filename glob, relative to source root.")
    parser.add_argument("--recursive", action="store_true", help="Discover matching videos recursively.")
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute source paths instead of paths relative to source_root.",
    )
    parser.add_argument(
        "--require-camera-metadata-csv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail if a matching Cam<id>_meta.csv sidecar is missing.",
    )
    parser.add_argument(
        "--prompt-metadata",
        action="store_true",
        help="Interactively prompt for optional shared metadata not provided by flags.",
    )
    parser.add_argument("--recording-id", help="Base recording ID. Defaults to recording_snapshot.json or source folder.")
    parser.add_argument(
        "--session-uuid-template",
        default="{recording_id}_cam{camera_id}",
        help="Template for per-row session_uuid. Variables: recording_id, camera_id, video_stem.",
    )
    parser.add_argument(
        "--recording-name-template",
        default="{recording_id}_cam{camera_id}",
        help="Template for per-row recording_name. Variables: recording_id, camera_id, video_stem.",
    )
    parser.add_argument("--session-start-utc", help="ISO 8601 UTC start time. Defaults to recording_snapshot.json when present.")
    parser.add_argument("--recording-type", default="behavior")
    parser.add_argument("--recording-subtype", default="free")
    parser.add_argument("--behavior-mode", default="free")
    parser.add_argument("--artifact-schema-id", default="video_only_v1")
    parser.add_argument("--dish-design")
    parser.add_argument("--rig-id")
    parser.add_argument("--arena-id")
    parser.add_argument("--camera-id", help="Override camera_id for every row. Usually leave unset.")
    parser.add_argument("--canvas-name")
    parser.add_argument("--protocol-name")
    parser.add_argument("--protocol-name-from-definition")
    parser.add_argument("--genotype")
    parser.add_argument("--dpf-at-acquisition", type=int)
    parser.add_argument("--num-dishes", type=int)
    parser.add_argument("--fish-per-dish", type=int)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        rows = _build_rows(args)
        _write_rows(rows, args.output, overwrite=bool(args.overwrite))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
