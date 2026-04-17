#!/usr/bin/env python3
"""Backfill hevc_keyframe_flags metadata in recording manifests.

Defaults to dry-run; pass --apply to write changes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from fisheye.shared.batch_logging import utc_now

try:
    from fisheye.diagnostics.video.container import check_hevc_keyframe_flags
except ModuleNotFoundError:
    import sys

    _THIS_DIR = Path(__file__).resolve().parent
    _SRC_DIR = _THIS_DIR.parent.parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    from fisheye.diagnostics.video.container import check_hevc_keyframe_flags


_utc_now = utc_now


def _iter_manifests(
    roots: Sequence[Path],
    *,
    recursive: bool,
    manifest_name: str,
) -> Iterable[Path]:
    seen: set[str] = set()
    for root in roots:
        path = root.expanduser()
        if path.is_file() and path.name == manifest_name:
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                yield path
            continue
        if not path.exists():
            continue
        if path.is_dir():
            direct = path / manifest_name
            if direct.is_file():
                key = str(direct.resolve())
                if key not in seen:
                    seen.add(key)
                    yield direct
            if recursive:
                candidates = path.rglob(manifest_name)
            else:
                candidates = path.glob(f"*/{manifest_name}")
            for manifest in candidates:
                if not manifest.is_file():
                    continue
                key = str(manifest.resolve())
                if key in seen:
                    continue
                seen.add(key)
                yield manifest


def _collect_manifest_mp4_paths(recording_dir: Path, payload: Dict[str, Any]) -> List[str]:
    rel_paths: List[str] = []
    seen: set[str] = set()
    files = payload.get("files")
    if isinstance(files, dict):
        for section in ("raw", "cams", "derived"):
            values = files.get(section)
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, str):
                    continue
                rel = value.strip().replace("\\", "/")
                if not rel or not rel.lower().endswith(".mp4"):
                    continue
                if rel in seen:
                    continue
                seen.add(rel)
                rel_paths.append(rel)

    if rel_paths:
        return rel_paths

    # Fallback for legacy manifests lacking file maps.
    for section in ("raw", "cams", "derived"):
        folder = recording_dir / section
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.mp4")):
            rel = f"{section}/{path.name}"
            if rel in seen:
                continue
            seen.add(rel)
            rel_paths.append(rel)
    return rel_paths


def _scan_video(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = dict(check_hevc_keyframe_flags(path))
    result["checked_at_utc"] = _utc_now()
    try:
        stat = path.stat()
        result["file_size_bytes"] = int(stat.st_size)
        result["file_mtime_ns"] = int(stat.st_mtime_ns)
    except OSError as exc:
        result["fingerprint_error"] = str(exc)
    return result


def _build_keyframe_payload(recording_dir: Path, manifest_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for rel_path in _collect_manifest_mp4_paths(recording_dir, manifest_payload):
        abs_path = recording_dir / rel_path
        out[rel_path] = _scan_video(abs_path)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "Recording folders, roots, or recording_manifest.json paths "
            "(default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings)."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan for recording manifests under path roots.",
    )
    parser.add_argument(
        "--manifest-name",
        default="recording_manifest.json",
        help="Manifest filename to backfill (default: recording_manifest.json).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing non-empty hevc_keyframe_flags payloads.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write updates to manifests (default is dry-run).",
    )
    parser.add_argument(
        "--show-needs-fix",
        type=int,
        default=20,
        help="Maximum number of needs-fix rows to print (default: 20).",
    )
    args = parser.parse_args(argv)

    if args.paths:
        roots = list(args.paths)
    else:
        env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
        roots = [Path(env_root)] if env_root else [Path("/nvme1/recordings")]

    manifests = list(
        _iter_manifests(
            roots,
            recursive=bool(args.recursive),
            manifest_name=str(args.manifest_name),
        )
    )
    if not manifests:
        print("No manifests found.")
        return 1

    updated = 0
    skipped_existing = 0
    failed = 0
    total_videos = 0
    total_needs_fix = 0
    needs_fix_rows: List[str] = []

    for manifest_path in manifests:
        recording_dir = manifest_path.parent
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"ERROR {manifest_path}: failed to read JSON ({exc})")
            failed += 1
            continue

        if not isinstance(payload, dict):
            print(f"ERROR {manifest_path}: top-level JSON is not an object")
            failed += 1
            continue

        existing = payload.get("hevc_keyframe_flags")
        if (
            not args.overwrite
            and isinstance(existing, dict)
            and len(existing) > 0
        ):
            print(f"SKIP {manifest_path}: hevc_keyframe_flags already populated")
            skipped_existing += 1
            continue

        scan_payload = _build_keyframe_payload(recording_dir, payload)
        videos_scanned = len(scan_payload)
        needs_fix = 0
        for rel_path, result in scan_payload.items():
            if bool(result.get("needs_fix", False)):
                needs_fix += 1
                if len(needs_fix_rows) < max(0, int(args.show_needs_fix)):
                    needs_fix_rows.append(f"{recording_dir / rel_path}")

        total_videos += videos_scanned
        total_needs_fix += needs_fix

        action = "UPDATED" if args.apply else "WOULD UPDATE"
        print(f"{action} {manifest_path}: videos={videos_scanned} needs_fix={needs_fix}")

        if args.apply:
            payload["hevc_keyframe_flags"] = scan_payload if scan_payload else None
            try:
                manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except Exception as exc:
                print(f"ERROR {manifest_path}: failed to write JSON ({exc})")
                failed += 1
                continue
        updated += 1

    print("\nSummary:")
    print(f"  manifests_found: {len(manifests)}")
    print(f"  updated: {updated}")
    print(f"  skipped_existing: {skipped_existing}")
    print(f"  failed: {failed}")
    print(f"  videos_scanned: {total_videos}")
    print(f"  needs_fix: {total_needs_fix}")
    if needs_fix_rows:
        print("  needs_fix_examples:")
        for row in needs_fix_rows:
            print(f"    {row}")
    if not args.apply:
        print("Dry-run only. Re-run with --apply to write changes.")

    return 0 if failed == 0 else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
