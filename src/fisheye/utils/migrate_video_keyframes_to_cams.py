#!/usr/bin/env python3
"""Move video-only keyframe summaries from derived/ to cams/."""

from __future__ import annotations

from fisheye.shared.json_safety import write_json_atomic as _write_json
import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

from fisheye.shared.batch_logging import make_run_id, utc_now


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def _discover_recording_dirs(paths: list[Path], *, name_prefix: Optional[str]) -> list[Path]:
    recordings: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        candidates: list[Path]
        if (path / "recording_manifest.json").exists():
            candidates = [path]
        elif path.is_dir():
            candidates = [child for child in sorted(path.iterdir()) if (child / "recording_manifest.json").exists()]
        else:
            candidates = []
        for candidate in candidates:
            if name_prefix and not candidate.name.startswith(name_prefix):
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            recordings.append(candidate)
    return recordings


def _expected_keyframe_name(recording_dir: Path) -> str | None:
    videos = sorted((recording_dir / "cams").glob("*.mp4"))
    if len(videos) != 1:
        return None
    return f"{videos[0].stem}_keyframe.json"


def _select_keyframe(recording_dir: Path) -> Path | None:
    expected_name = _expected_keyframe_name(recording_dir)
    if expected_name:
        expected_derived = recording_dir / "derived" / expected_name
        if expected_derived.exists():
            return expected_derived
    candidates = sorted((recording_dir / "derived").glob("*_keyframe.json"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _replace_manifest_path(payload: dict[str, Any], *, old_rel: str, new_rel: str) -> str:
    files = payload.get("files")
    if not isinstance(files, dict):
        files = {}
        payload["files"] = files
    derived = files.get("derived")
    if not isinstance(derived, list):
        derived = []
        files["derived"] = derived
    cams = files.get("cams")
    if not isinstance(cams, list):
        cams = []
        files["cams"] = cams

    changed = False
    if old_rel in derived:
        derived.remove(old_rel)
        changed = True
    if new_rel not in cams:
        cams.append(new_rel)
        changed = True
    return "manifest_patched" if changed else "manifest_already_current"


def _append_migration(payload: dict[str, Any], *, migration_id: str, old_rel: str, new_rel: str) -> None:
    migrations = payload.get("metadata_migrations")
    if not isinstance(migrations, list):
        migrations = []
        payload["metadata_migrations"] = migrations
    migrations.append(
        {
            "migration_type": "video_keyframe_to_cams_v1",
            "migration_id": migration_id,
            "created_at_utc": utc_now(),
            "tool": "fisheye.utils.migrate_video_keyframes_to_cams",
            "old_path": old_rel,
            "new_path": new_rel,
        }
    )


def _migrate_recording(recording_dir: Path, *, dry_run: bool, overwrite: bool, migration_id: str) -> dict[str, Any]:
    source = _select_keyframe(recording_dir)
    if source is None:
        return {"recording": recording_dir.name, "status": "skip_no_derived_keyframe"}
    dest = recording_dir / "cams" / source.name
    old_rel = source.relative_to(recording_dir).as_posix()
    new_rel = dest.relative_to(recording_dir).as_posix()
    if dest.exists() and not overwrite:
        manifest_status = "manifest_not_touched"
        if dest.resolve() == source.resolve():
            manifest_status = "manifest_already_current"
        return {
            "recording": recording_dir.name,
            "status": "destination_exists",
            "old_path": old_rel,
            "new_path": new_rel,
            "manifest_status": manifest_status,
        }
    if dry_run:
        return {
            "recording": recording_dir.name,
            "status": "move_planned",
            "old_path": old_rel,
            "new_path": new_rel,
            "manifest_status": "manifest_patch_planned",
        }

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and overwrite:
        dest.unlink()
    shutil.move(str(source), str(dest))

    manifest_path = recording_dir / "recording_manifest.json"
    manifest_status = "manifest_missing"
    if manifest_path.exists():
        payload = _load_json(manifest_path)
        manifest_status = _replace_manifest_path(payload, old_rel=old_rel, new_rel=new_rel)
        _append_migration(payload, migration_id=migration_id, old_rel=old_rel, new_rel=new_rel)
        _write_json(manifest_path, payload)

    return {
        "recording": recording_dir.name,
        "status": "moved",
        "old_path": old_rel,
        "new_path": new_rel,
        "manifest_status": manifest_status,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_dirs", nargs="+", type=Path, help="Recording directories or roots.")
    parser.add_argument("--apply", action="store_true", help="Move keyframe files and patch manifests.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned moves only.")
    parser.add_argument("--name-prefix", help="Only process recording directories with this name prefix.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cams keyframe destinations.")
    parser.add_argument(
        "--migration-id",
        default=f"video_keyframe_to_cams_{make_run_id()}",
        help="Stable migration id recorded in manifests.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if bool(args.apply) == bool(args.dry_run):
        print("Specify exactly one of --dry-run or --apply.", file=sys.stderr)
        return 1
    recordings = _discover_recording_dirs(args.recording_dirs, name_prefix=args.name_prefix)
    if not recordings:
        print("No recording directories matched.")
        return 0

    counts: dict[str, int] = {}
    for recording_dir in recordings:
        try:
            result = _migrate_recording(
                recording_dir,
                dry_run=bool(args.dry_run),
                overwrite=bool(args.overwrite),
                migration_id=str(args.migration_id),
            )
        except Exception as exc:
            result = {"recording": recording_dir.name, "status": "error", "error": str(exc)}
        counts[str(result.get("status"))] = counts.get(str(result.get("status")), 0) + 1
        print(json.dumps(result, sort_keys=True))
    print("summary:")
    print(json.dumps({"recordings": len(recordings), "statuses": counts}, sort_keys=True))
    return 1 if counts.get("error") else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
