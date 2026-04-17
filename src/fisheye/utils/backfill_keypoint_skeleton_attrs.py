from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import zarr

from ..cli.shared_args import add_log_args
from ..pose.schema import normalize_kpt_shape, resolve_skeleton_identity_from_attrs
from ..shared.batch_logging import JsonLogger as SharedJsonLogger
from ..shared.batch_logging import make_run_id
from ..shared.environment import resolve_log_dir as resolve_shared_log_dir
from ..shared.zarr_helpers import _direct_group_names, _group_names, _open_group_direct, _open_mode, _root_fs_path
from .zarr_io import open_zarr_root


@dataclass
class BackfillResult:
    status: str
    changed_fields: tuple[str, ...] = ()
    reason: Optional[str] = None
    resolved_skeleton_id: Optional[str] = None
    resolved_kpt_shape: Optional[tuple[int, int]] = None


JsonLogger = SharedJsonLogger
_run_id = make_run_id


def _iter_zarr(roots: list[Path], recursive: bool) -> Iterable[Path]:
    for root in roots:
        root = root.expanduser()
        if root.is_file() and root.suffix == ".zarr":
            yield root
            continue
        if not root.exists():
            continue
        if recursive:
            yield from root.rglob("zarr/*.zarr")
        else:
            yield from root.glob("*/zarr/*.zarr")


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> Optional[str]:
    purpose = root.attrs.get("zarr_purpose")
    if purpose is not None:
        value = str(purpose).strip().lower()
        if value in {"analysis", "training"}:
            return value
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return None


def _coerce_mapping(value: object) -> Optional[dict[str, object]]:
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _explicit_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text or None


def _keypoint_count(run_group: zarr.Group) -> Optional[int]:
    for array_name in ("keypoints_roi", "keypoints_img"):
        points = run_group.get(array_name)
        if points is not None and len(points.shape) == 3:
            return int(points.shape[1])
    return None


def _resolve_log_dir(arg_log_dir: Optional[Path], roots: list[Path]) -> Path:
    return resolve_shared_log_dir(arg_log_dir, roots, log_subdir="backfill_keypoint_skeleton_attrs")


def _iter_run_groups(
    root: zarr.Group,
    all_runs: bool,
    *,
    zarr_path: Optional[Path] = None,
    open_mode: Optional[str] = None,
) -> Iterable[tuple[str, zarr.Group]]:
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

        available_names = sorted(set(_group_names(parent)) | set(_direct_group_names(parent_fs_path)))
        if all_runs:
            run_names = available_names
        else:
            latest = parent.attrs.get("latest")
            latest_name = str(latest) if latest else ""
            if latest_name and latest_name in available_names:
                run_names = [latest_name]
            else:
                run_names = [available_names[-1]] if available_names else []

        for run_name in run_names:
            direct_path = (parent_fs_path / run_name) if parent_fs_path is not None else None
            if direct_path is not None and run_name in available_names:
                try:
                    run_group = _open_group_direct(direct_path, mode=resolved_open_mode)
                except Exception:
                    if run_name in parent:
                        yield f"{parent_name}/{run_name}", parent[run_name]
                    continue
                yield f"{parent_name}/{run_name}", run_group
            elif run_name in parent:
                yield f"{parent_name}/{run_name}", parent[run_name]


def _backfill_run_group(run_group: zarr.Group, *, apply: bool) -> BackfillResult:
    pose_schema = _coerce_mapping(run_group.attrs.get("pose_schema"))
    if pose_schema is None:
        return BackfillResult(status="no_pose_schema", reason="pose_schema attr missing or invalid")

    resolved = resolve_skeleton_identity_from_attrs(
        dict(run_group.attrs),
        keypoint_count=_keypoint_count(run_group),
    )
    if resolved.skeleton_id is None:
        return BackfillResult(status="no_skeleton_id", reason="unable to resolve skeleton_id from attrs")
    if resolved.kpt_shape is None:
        return BackfillResult(status="no_kpt_shape", reason="unable to resolve kpt_shape from attrs")

    canonical_shape = [int(resolved.kpt_shape[0]), int(resolved.kpt_shape[1])]
    pose_schema_out = deepcopy(pose_schema)
    changed_fields: list[str] = []

    if _explicit_text(run_group.attrs.get("skeleton_id")) != resolved.skeleton_id:
        changed_fields.append("skeleton_id")
    if normalize_kpt_shape(run_group.attrs.get("kpt_shape")) != resolved.kpt_shape:
        changed_fields.append("kpt_shape")
    if _explicit_text(pose_schema_out.get("skeleton_id")) != resolved.skeleton_id:
        pose_schema_out["skeleton_id"] = resolved.skeleton_id
        changed_fields.append("pose_schema.skeleton_id")
    if normalize_kpt_shape(pose_schema_out.get("kpt_shape")) != resolved.kpt_shape:
        pose_schema_out["kpt_shape"] = canonical_shape
        changed_fields.append("pose_schema.kpt_shape")

    if not changed_fields:
        return BackfillResult(
            status="skipped_existing",
            resolved_skeleton_id=resolved.skeleton_id,
            resolved_kpt_shape=resolved.kpt_shape,
        )

    if apply:
        run_group.attrs["skeleton_id"] = resolved.skeleton_id
        run_group.attrs["kpt_shape"] = canonical_shape
        run_group.attrs["pose_schema"] = pose_schema_out

    return BackfillResult(
        status="ok",
        changed_fields=tuple(changed_fields),
        resolved_skeleton_id=resolved.skeleton_id,
        resolved_kpt_shape=resolved.kpt_shape,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill explicit skeleton identity attrs on keypoint and refined-keypoint runs. "
            "Writes run-level skeleton_id/kpt_shape and normalizes pose_schema.skeleton_id/"
            "pose_schema.kpt_shape from the resolved runtime identity."
        )
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="any",
        help="Filter zarr archives by purpose (default: any).",
    )
    parser.add_argument("--all-runs", action="store_true", help="Backfill all run groups (default: latest only).")
    add_log_args(
        parser,
        log_dir_help=(
            "Directory for JSONL logs (default: $PALETTE_LOG_ROOT/backfill_keypoint_skeleton_attrs "
            "or <root>/logs/backfill_keypoint_skeleton_attrs)."
        ),
    )
    parser.add_argument("--apply", action="store_true", help="Apply updates (default: dry-run).")

    args = parser.parse_args(argv)
    roots = list(args.paths) if args.paths else [Path("/nvme1/recordings")]
    logger: Optional[JsonLogger] = None
    log_path: Optional[Path] = None
    run_id = _run_id()

    if not args.no_log:
        log_dir = _resolve_log_dir(args.log_dir, roots)
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"backfill_keypoint_skeleton_attrs_{run_id}.jsonl"
            logger = JsonLogger(log_path, run_id)
            print(f"Log file: {log_path}")
        except Exception as exc:
            logger = None
            print(f"Warning: logging disabled ({exc})")

    counts = {
        "zarr_scanned": 0,
        "runs_considered": 0,
        "ok": 0,
        "skipped_existing": 0,
        "no_pose_schema": 0,
        "no_skeleton_id": 0,
        "no_kpt_shape": 0,
        "missing_runs": 0,
        "filtered_zarr_use": 0,
        "errors": 0,
    }

    if logger is not None:
        logger.log(
            "run_start",
            roots=[str(root) for root in roots],
            recursive=bool(args.recursive),
            zarr_use=str(args.zarr_use),
            all_runs=bool(args.all_runs),
            apply=bool(args.apply),
            dry_run=not bool(args.apply),
        )

    any_zarr = False
    for zarr_path in _iter_zarr(roots, recursive=bool(args.recursive)):
        any_zarr = True
        counts["zarr_scanned"] += 1
        try:
            root = open_zarr_root(zarr_path, mode="a" if args.apply else "r")
            if args.zarr_use != "any":
                observed_use = _infer_zarr_use(root, zarr_path)
                if observed_use != args.zarr_use:
                    counts["filtered_zarr_use"] += 1
                    if logger is not None:
                        logger.log(
                            "zarr_skipped_use_filter",
                            zarr=str(zarr_path),
                            observed_zarr_use=observed_use,
                            requested_zarr_use=str(args.zarr_use),
                        )
                    continue
            else:
                observed_use = _infer_zarr_use(root, zarr_path)

            run_info = list(
                _iter_run_groups(
                    root,
                    all_runs=bool(args.all_runs),
                    zarr_path=zarr_path,
                    open_mode="a" if args.apply else "r",
                )
            )
            if not run_info:
                counts["missing_runs"] += 1
                if logger is not None:
                    logger.log("zarr_missing_runs", zarr=str(zarr_path))
                continue

            for run_path, run_group in run_info:
                counts["runs_considered"] += 1
                result = _backfill_run_group(run_group, apply=bool(args.apply))
                counts[result.status] += 1
                if logger is not None:
                    logger.log(
                        "run_group_checked",
                        zarr=str(zarr_path),
                        zarr_use=observed_use,
                        run_path=run_path,
                        status=result.status,
                        changed=result.status == "ok",
                        changed_fields=list(result.changed_fields),
                        reason=result.reason,
                        resolved_skeleton_id=result.resolved_skeleton_id,
                        resolved_kpt_shape=list(result.resolved_kpt_shape)
                        if result.resolved_kpt_shape is not None
                        else None,
                        apply=bool(args.apply),
                        dry_run=not bool(args.apply),
                    )
        except Exception as exc:
            counts["errors"] += 1
            print(f"error: {zarr_path}: {exc}")
            if logger is not None:
                logger.log("zarr_error", zarr=str(zarr_path), error=str(exc))

    if not any_zarr:
        print("No zarr files found.")
        if logger is not None:
            logger.log("run_end", status="no_zarr_found", exit_code=1)
            logger.close()
        return 1

    mode = "Applied" if args.apply else "Dry run"
    print(
        "Keypoint skeleton-attr backfill: "
        f"scope={args.zarr_use} zarr_scanned={counts['zarr_scanned']} "
        f"runs_considered={counts['runs_considered']} filtered_zarr_use={counts['filtered_zarr_use']} "
        f"missing_runs={counts['missing_runs']} errors={counts['errors']}"
    )
    print(
        f"{mode}: ok={counts['ok']} skipped_existing={counts['skipped_existing']} "
        f"no_pose_schema={counts['no_pose_schema']} no_skeleton_id={counts['no_skeleton_id']} "
        f"no_kpt_shape={counts['no_kpt_shape']}"
    )

    exit_code = 0 if counts["errors"] == 0 else 1
    if logger is not None:
        logger.log(
            "run_end",
            status="ok" if exit_code == 0 else "error",
            mode="apply" if args.apply else "dry-run",
            exit_code=exit_code,
            **counts,
        )
        logger.close()
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
