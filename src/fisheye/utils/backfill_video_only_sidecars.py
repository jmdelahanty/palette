#!/usr/bin/env python3
"""Backfill optional video-only sidecars into already organized recordings."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fisheye.utils.organize_recordings import (
    PlannedFile,
    _build_video_only_plan,
    _load_video_only_rows,
)


@dataclass(frozen=True)
class SidecarOperation:
    recording_name: str
    section: str
    planned: PlannedFile
    dest: Path


def _relative_manifest_path(plan_dest_dir: Path, dest: Path) -> str:
    return dest.relative_to(plan_dest_dir).as_posix()


def _operation_status(op: SidecarOperation, *, overwrite: bool) -> str:
    if not op.planned.source.exists():
        if op.dest.exists():
            return "destination_exists_source_missing"
        return "missing_optional_source"
    if op.dest.exists() and not overwrite:
        return "exists_skip"
    return "ready"


def _build_operations(args: argparse.Namespace) -> list[SidecarOperation]:
    source_root = args.source_root.expanduser().resolve()
    rows = _load_video_only_rows(args.metadata_csv.expanduser().resolve(), source_root=source_root)
    plans = [
        _build_video_only_plan(row, dest_root=args.dest_root.expanduser().resolve(), rename_cams=bool(args.rename_cams))
        for row in rows
    ]

    operations: list[SidecarOperation] = []
    for plan in plans:
        for section, files in (
            ("raw", plan.raw_files),
            ("cams", plan.cam_files),
            ("derived", plan.derived_files),
        ):
            for planned in files:
                if section == "cams" and planned.dest_name.lower().endswith((".mp4", "_meta.csv")):
                    continue
                operations.append(
                    SidecarOperation(
                        recording_name=plan.name,
                        section=section,
                        planned=planned,
                        dest=plan.dest_dir / section / planned.dest_name,
                    )
                )
    return operations


def _patch_manifest(
    *,
    recording_dir: Path,
    section: str,
    rel_path: str,
    dry_run: bool,
) -> str:
    manifest_path = recording_dir / "recording_manifest.json"
    if not manifest_path.exists():
        return "manifest_missing"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"manifest_read_error:{exc}"
    if not isinstance(payload, dict):
        return "manifest_invalid_root"

    files = payload.get("files")
    if not isinstance(files, dict):
        files = {}
        payload["files"] = files
    entries = files.get(section)
    if not isinstance(entries, list):
        entries = []
        files[section] = entries
    if rel_path not in entries:
        entries.append(rel_path)
    else:
        return "manifest_already_listed"

    if dry_run:
        return "manifest_patch_planned"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return "manifest_patched"


def _apply_operation(op: SidecarOperation, *, dry_run: bool, overwrite: bool, patch_manifest: bool) -> tuple[str, str]:
    status = _operation_status(op, overwrite=overwrite)
    rel_path = _relative_manifest_path(op.dest.parent.parent, op.dest)
    if status == "exists_skip" or status == "destination_exists_source_missing":
        manifest_status = "manifest_not_requested"
        if patch_manifest:
            manifest_status = _patch_manifest(
                recording_dir=op.dest.parent.parent,
                section=op.section,
                rel_path=rel_path,
                dry_run=dry_run,
            )
        return status, manifest_status
    if status != "ready":
        return status, "manifest_not_touched"

    manifest_status = "manifest_not_requested"
    if dry_run:
        action = "copy" if op.planned.action == "copy" else "move"
        if patch_manifest:
            manifest_status = _patch_manifest(
                recording_dir=op.dest.parent.parent,
                section=op.section,
                rel_path=rel_path,
                dry_run=True,
            )
        return f"{action}_planned", manifest_status

    op.dest.parent.mkdir(parents=True, exist_ok=True)
    if op.dest.exists() and overwrite:
        op.dest.unlink()
    if op.planned.action == "copy":
        shutil.copy2(op.planned.source, op.dest)
        file_status = "copied"
    elif op.planned.action == "move":
        shutil.move(str(op.planned.source), str(op.dest))
        file_status = "moved"
    else:
        return f"unknown_action:{op.planned.action}", "manifest_not_touched"

    if patch_manifest:
        manifest_status = _patch_manifest(
            recording_dir=op.dest.parent.parent,
            section=op.section,
            rel_path=rel_path,
            dry_run=False,
        )
    return file_status, manifest_status


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path, help="Original video-only staging directory.")
    parser.add_argument("--metadata-csv", type=Path, required=True, help="Video-only organizer manifest CSV.")
    parser.add_argument("--dest-root", type=Path, required=True, help="Organized recordings root.")
    parser.add_argument("--apply", action="store_true", help="Copy/move files and patch manifests.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned repair only.")
    parser.add_argument(
        "--rename-cams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the same camera renaming policy as organize_recordings.",
    )
    parser.add_argument(
        "--patch-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Patch recording_manifest.json files.raw/files.derived entries.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sidecar destinations.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if bool(args.apply) == bool(args.dry_run):
        print("Specify exactly one of --dry-run or --apply.", file=sys.stderr)
        return 1
    if args.overwrite and not args.apply:
        print("--overwrite only has an effect with --apply.", file=sys.stderr)
        return 1

    try:
        operations = _build_operations(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not operations:
        print("No sidecar operations found.")
        return 0

    counts: dict[str, int] = {}
    manifest_counts: dict[str, int] = {}
    for op in operations:
        file_status, manifest_status = _apply_operation(
            op,
            dry_run=bool(args.dry_run),
            overwrite=bool(args.overwrite),
            patch_manifest=bool(args.patch_manifest),
        )
        counts[file_status] = counts.get(file_status, 0) + 1
        manifest_counts[manifest_status] = manifest_counts.get(manifest_status, 0) + 1
        print(
            f"{file_status}: {op.recording_name} {op.section}/"
            f"{op.planned.dest_name} <- {op.planned.source} "
            f"(action={op.planned.action}, manifest={manifest_status})"
        )

    print("summary:")
    print(json.dumps({"files": counts, "manifests": manifest_counts}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
