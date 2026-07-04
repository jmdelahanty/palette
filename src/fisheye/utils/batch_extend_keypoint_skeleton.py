from __future__ import annotations

from fisheye.shared.zarr_helpers import infer_zarr_use as _infer_zarr_use
from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import zarr

from fisheye.utils.extend_keypoint_skeleton import (
    KEYPOINT_PARENT_CHOICES,
    TARGET_SCHEMA_DEFAULT,
    _default_target_run_name,
    extend_keypoint_skeleton_run,
)


DEFAULT_RECORDINGS_ROOT = Path("/nvme1/recordings")


@dataclass(frozen=True)
class BatchRow:
    zarr_path: Path
    status: str
    reason: Optional[str] = None
    source_parent: Optional[str] = None
    source_run: Optional[str] = None
    target_parent: Optional[str] = None
    target_run: Optional[str] = None


def _resolve_roots(paths: list[Path]) -> list[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [DEFAULT_RECORDINGS_ROOT]


def _normalize_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _select_latest_run(parent: zarr.Group) -> Optional[str]:
    latest = _normalize_text(parent.attrs.get("latest"))
    if latest and latest in parent:
        return latest
    try:
        names = sorted(list(parent.group_keys()))
    except Exception:
        names = sorted(list(parent.keys()))
    return names[-1] if names else None


def _resolve_source_choice(
    root: zarr.Group,
    *,
    source_parent: str,
    source_run: Optional[str],
) -> tuple[str, str]:
    parent_names: Sequence[str]
    if source_parent == "auto":
        parent_names = ("refined_keypoints_runs", "keypoints_runs")
    else:
        parent_names = (source_parent,)

    errors: list[str] = []
    for parent_name in parent_names:
        parent = root.get(parent_name)
        if parent is None:
            errors.append(f"{parent_name}: missing")
            continue
        chosen_run = _normalize_text(source_run)
        if chosen_run is None:
            chosen_run = _select_latest_run(parent)
            if chosen_run is None:
                errors.append(f"{parent_name}: no runs")
                continue
        if chosen_run not in parent:
            errors.append(f"{parent_name}: missing run {chosen_run}")
            continue
        return parent_name, chosen_run
    raise RuntimeError("; ".join(errors) if errors else "no source run resolved")


def _source_already_matches_target(root: zarr.Group, *, parent_name: str, run_name: str, target_schema: str) -> bool:
    try:
        group = root[f"{parent_name}/{run_name}"]
    except Exception:
        return False
    pose_schema = group.attrs.get("pose_schema")
    if isinstance(pose_schema, dict):
        name = _normalize_text(pose_schema.get("name"))
        if name == target_schema:
            return True
    return False


def _format_row(row: BatchRow) -> str:
    source = (
        f"{row.source_parent}/{row.source_run}"
        if row.source_parent is not None and row.source_run is not None
        else "-"
    )
    target = (
        f"{row.target_parent}/{row.target_run}"
        if row.target_parent is not None and row.target_run is not None
        else "-"
    )
    reason = row.reason or "-"
    return f"{row.status}\t{row.zarr_path}\t{source}\t{target}\t{reason}"


def _process_zarr_path(
    zarr_path: Path,
    *,
    apply: bool,
    zarr_use_filter: str,
    source_parent: str,
    source_run: Optional[str],
    target_run: Optional[str],
    target_schema: str,
    overwrite: bool,
    set_latest: bool,
) -> BatchRow:
    mode = "r+" if apply else "r"
    root = zarr.open_group(str(zarr_path), mode=mode)

    observed_use = _infer_zarr_use(root, zarr_path)
    if zarr_use_filter != "any" and observed_use != zarr_use_filter:
        return BatchRow(
            zarr_path=zarr_path,
            status="filtered_zarr_use",
            reason=f"wanted={zarr_use_filter} found={observed_use or 'unknown'}",
        )

    try:
        resolved_parent, resolved_run = _resolve_source_choice(
            root,
            source_parent=source_parent,
            source_run=source_run,
        )
    except Exception as exc:
        return BatchRow(
            zarr_path=zarr_path,
            status="missing_source",
            reason=str(exc),
        )
    if _source_already_matches_target(
        root,
        parent_name=resolved_parent,
        run_name=resolved_run,
        target_schema=target_schema,
    ):
        return BatchRow(
            zarr_path=zarr_path,
            status="skipped_existing",
            reason=f"{resolved_parent}/{resolved_run} already uses {target_schema}",
            source_parent=resolved_parent,
            source_run=resolved_run,
        )

    computed_target_run = target_run or _default_target_run_name(resolved_run, target_schema)
    target_parent = resolved_parent
    if not apply and target_parent in root and computed_target_run in root[target_parent] and not overwrite:
        return BatchRow(
            zarr_path=zarr_path,
            status="skipped_existing",
            reason=f"{target_parent}/{computed_target_run} already exists",
            source_parent=resolved_parent,
            source_run=resolved_run,
            target_parent=target_parent,
            target_run=computed_target_run,
        )

    try:
        summary = extend_keypoint_skeleton_run(
            root,
            source_run=resolved_run,
            source_parent=resolved_parent,
            target_run=computed_target_run,
            target_schema=target_schema,
            overwrite=overwrite,
            set_latest=set_latest,
            apply=apply,
        )
    except Exception as exc:
        return BatchRow(
            zarr_path=zarr_path,
            status="error",
            reason=str(exc),
            source_parent=resolved_parent,
            source_run=resolved_run,
            target_parent=target_parent,
            target_run=computed_target_run,
        )

    return BatchRow(
        zarr_path=zarr_path,
        status="updated" if apply else "planned",
        source_parent=str(summary["source_parent"]),
        source_run=str(summary["source_run"]),
        target_parent=str(summary["target_parent"]),
        target_run=str(summary["target_run"]),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch seed extended keypoint skeleton runs across recording zarrs."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=("analysis", "training", "any"),
        default="analysis",
        help="Filter zarr archives by purpose (default: analysis).",
    )
    parser.add_argument(
        "--source-parent",
        choices=list(KEYPOINT_PARENT_CHOICES),
        default="auto",
        help="Parent group containing the source run (default: auto).",
    )
    parser.add_argument(
        "--source-run",
        type=str,
        help="Optional source run name override. Defaults to latest in the chosen parent.",
    )
    parser.add_argument("--target-run", type=str, help="Optional explicit target run name.")
    parser.add_argument(
        "--target-schema",
        default=TARGET_SCHEMA_DEFAULT,
        help=f"Target pose schema name from configs/fisheye/pose_schemas (default: {TARGET_SCHEMA_DEFAULT}).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing target run.")
    parser.add_argument("--set-latest", action="store_true", help="Update parent latest to the new run.")
    parser.add_argument("--apply", action="store_true", help="Write the extended seed runs.")
    args = parser.parse_args(argv)

    roots = _resolve_roots(list(args.paths))
    rows: list[BatchRow] = []
    for zarr_path in _iter_zarr(roots, recursive=bool(args.recursive)):
        row = _process_zarr_path(
            zarr_path,
            apply=bool(args.apply),
            zarr_use_filter=str(args.zarr_use),
            source_parent=str(args.source_parent),
            source_run=str(args.source_run) if args.source_run else None,
            target_run=str(args.target_run) if args.target_run else None,
            target_schema=str(args.target_schema),
            overwrite=bool(args.overwrite),
            set_latest=bool(args.set_latest),
        )
        rows.append(row)
        print(_format_row(row))

    if not rows:
        print("No zarr files found.")
        return 1

    counts = {
        "zarr_scanned": len(rows),
        "planned": 0,
        "updated": 0,
        "filtered_zarr_use": 0,
        "missing_source": 0,
        "skipped_existing": 0,
        "error": 0,
    }
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1

    mode = "Applied" if args.apply else "Dry run"
    print(
        "Batch keypoint skeleton extend: "
        f"mode={mode} scope={args.zarr_use} zarr_scanned={counts['zarr_scanned']} "
        f"filtered_zarr_use={counts['filtered_zarr_use']} missing_source={counts['missing_source']} "
        f"errors={counts['error']}"
    )
    print(
        f"Results: updated={counts['updated']} planned={counts['planned']} "
        f"skipped_existing={counts['skipped_existing']}"
    )
    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
