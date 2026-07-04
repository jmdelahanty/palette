from __future__ import annotations

from fisheye.shared.zarr_helpers import infer_zarr_use as _infer_zarr_use
from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional

import zarr

from fisheye.cli.shared_args import add_log_args
from fisheye.pose.schema import normalize_kpt_shape, resolve_skeleton_identity_from_attrs
from fisheye.shared.batch_logging import JsonLogger as SharedJsonLogger
from fisheye.shared.batch_logging import make_run_id
from fisheye.shared.environment import resolve_log_dir as resolve_shared_log_dir
from fisheye.shared.zarr_helpers import _direct_group_names, _group_names, _open_group_direct, _open_mode, _root_fs_path
from fisheye.shared.zarr_io import open_zarr_root

JsonLogger = SharedJsonLogger
_run_id = make_run_id


def _iter_run_groups(
    root: zarr.Group,
    *,
    all_runs: bool,
    zarr_path: Optional[Path] = None,
    open_mode: Optional[str] = None,
) -> Iterable[tuple[str, str, zarr.Group]]:
    root_fs_path = zarr_path.expanduser().resolve() if zarr_path is not None else _root_fs_path(root)
    resolved_open_mode = open_mode or _open_mode(root)
    for parent_name in ("keypoints_runs", "refined_keypoints_runs", "keypoints_refined_runs"):
        parent_fs_path = root_fs_path
        if parent_fs_path is not None:
            parent_fs_path = parent_fs_path / parent_name

        parent = root.get(parent_name)
        if parent is None and parent_fs_path is not None and parent_fs_path.is_dir():
            try:
                parent = _open_group_direct(parent_fs_path, mode=resolved_open_mode)
            except Exception:
                parent = None
        if parent is None:
            continue
        available = sorted(set(_group_names(parent)) | set(_direct_group_names(parent_fs_path)))
        if not available:
            continue
        if all_runs:
            run_names = available
        else:
            latest = str(parent.attrs.get("latest") or "").strip()
            run_names = [latest] if latest and latest in available else [available[-1]]
        for run_name in run_names:
            direct_path = (parent_fs_path / run_name) if parent_fs_path is not None else None
            if direct_path is not None and run_name in available:
                try:
                    run_group = _open_group_direct(direct_path, mode=resolved_open_mode)
                except Exception:
                    if run_name in parent:
                        yield parent_name, run_name, parent[run_name]
                    continue
                yield parent_name, run_name, run_group
            elif run_name in parent:
                yield parent_name, run_name, parent[run_name]


def _explicit_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text or None


def _keypoint_count(run_group: zarr.Group) -> Optional[int]:
    points = run_group.get("keypoints_roi")
    if points is not None and len(points.shape) == 3:
        return int(points.shape[1])
    return None


def _resolve_log_dir(arg_log_dir: Optional[Path], roots: list[Path]) -> Path:
    return resolve_shared_log_dir(arg_log_dir, roots, log_subdir="audit_keypoint_skeleton_attrs")


def audit_keypoint_skeleton_attrs(
    roots: list[Path],
    *,
    recursive: bool,
    zarr_use: str = "all",
    all_runs: bool = False,
    only_missing: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for zarr_path in _iter_zarr(roots, recursive):
        root = open_zarr_root(zarr_path, mode="r")
        inferred_use = _infer_zarr_use(root, zarr_path)
        if zarr_use != "all" and inferred_use != zarr_use:
            continue
        for parent_name, run_name, run_group in _iter_run_groups(
            root,
            all_runs=all_runs,
            zarr_path=zarr_path,
            open_mode="r",
        ):
            resolved = resolve_skeleton_identity_from_attrs(
                dict(run_group.attrs),
                keypoint_count=_keypoint_count(run_group),
            )
            missing: list[str] = []
            if _explicit_text(run_group.attrs.get("skeleton_id")) is None:
                missing.append("skeleton_id")
            if normalize_kpt_shape(run_group.attrs.get("kpt_shape")) is None:
                missing.append("kpt_shape")
            if not isinstance(run_group.attrs.get("pose_schema"), Mapping):
                missing.append("pose_schema")
            if only_missing and not missing:
                continue
            rows.append(
                {
                    "zarr_path": str(zarr_path),
                    "run_path": f"{parent_name}/{run_name}",
                    "status": "missing_explicit_attrs" if missing else "ok",
                    "missing_attrs": missing,
                    "pose_schema_name": resolved.pose_schema_name,
                    "resolved_skeleton_id": resolved.skeleton_id,
                    "resolved_kpt_shape": list(resolved.kpt_shape) if resolved.kpt_shape is not None else None,
                    "zarr_use": inferred_use,
                }
            )
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit explicit skeleton identity attrs on keypoint and refined-keypoint runs."
    )
    parser.add_argument("roots", nargs="+", help="Recording roots or .zarr paths to inspect.")
    parser.add_argument("--recursive", action="store_true", help="Recurse under roots for zarr/*.zarr.")
    parser.add_argument(
        "--zarr-use",
        choices=["all", "analysis", "training"],
        default="all",
        help="Restrict to one zarr scope (default: all).",
    )
    parser.add_argument("--all-runs", action="store_true", help="Inspect all run groups, not just latest.")
    parser.add_argument("--only-missing", action="store_true", help="Print only rows missing explicit attrs.")
    parser.add_argument("--json", action="store_true", help="Emit full JSON instead of line output.")
    parser.add_argument("--strict", action="store_true", help="Return exit code 2 when missing attrs are found.")
    add_log_args(
        parser,
        log_dir_help=(
            "Directory for JSONL logs. Defaults to $PALETTE_LOG_ROOT/audit_keypoint_skeleton_attrs "
            "or <root>/logs/audit_keypoint_skeleton_attrs."
        ),
    )
    args = parser.parse_args(argv)
    roots = [Path(item) for item in args.roots]

    logger: Optional[JsonLogger] = None
    log_path: Optional[Path] = None
    if not args.no_log:
        log_dir = _resolve_log_dir(args.log_dir, roots)
        log_dir.mkdir(parents=True, exist_ok=True)
        run_id = _run_id()
        log_path = log_dir / f"audit_keypoint_skeleton_attrs_{run_id}.jsonl"
        logger = JsonLogger(log_path, run_id)
        print_target = sys.stderr if args.json else sys.stdout
        print(f"Log file: {log_path}", file=print_target)
        logger.log(
            "run_start",
            roots=[str(path) for path in roots],
            recursive=bool(args.recursive),
            zarr_use=str(args.zarr_use),
            all_runs=bool(args.all_runs),
            only_missing=bool(args.only_missing),
            strict=bool(args.strict),
            json=bool(args.json),
        )

    try:
        rows = audit_keypoint_skeleton_attrs(
            roots,
            recursive=bool(args.recursive),
            zarr_use=str(args.zarr_use),
            all_runs=bool(args.all_runs),
            only_missing=bool(args.only_missing),
        )
        missing_count = sum(1 for row in rows if row["status"] != "ok")
        summary = {
            "rows": len(rows),
            "missing": missing_count,
            "ok": len(rows) - missing_count,
            "zarr_use": str(args.zarr_use),
        }

        if logger is not None:
            for row in rows:
                logger.log("run_group_checked", **row)

        if args.json:
            print(json.dumps({"summary": summary, "rows": rows}, indent=2))
        else:
            print(
                "Keypoint skeleton attr audit: "
                f"scope={summary['zarr_use']} rows={summary['rows']} "
                f"ok={summary['ok']} missing={summary['missing']}"
            )
            for row in rows:
                if row["status"] == "ok":
                    continue
                print(
                    f"- {row['run_path']} @ {row['zarr_path']}: "
                    f"missing={','.join(row['missing_attrs'])}"
                )

        rc = 2 if args.strict and missing_count else 0
        if logger is not None:
            logger.log(
                "run_end",
                status="ok",
                mode="json" if args.json else "text",
                strict=bool(args.strict),
                exit_code=rc,
                **summary,
            )
        return rc
    finally:
        if logger is not None:
            logger.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
