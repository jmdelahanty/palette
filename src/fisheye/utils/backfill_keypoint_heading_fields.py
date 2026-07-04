from __future__ import annotations

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import zarr

from ..cli.shared_args import add_log_args
from ..pose.heading import compute_heading_from_spec, dependent_keypoints, resolve_heading_computation_from_attrs
from ..shared.batch_logging import JsonLogger as SharedJsonLogger
from ..shared.batch_logging import make_run_id
from ..shared.environment import resolve_log_dir as resolve_shared_log_dir
from ..shared.zarr_helpers import _direct_group_names, _group_names, _open_group_direct, _open_mode, _root_fs_path
from ..tune.keypoint_review import _update_postprocess_summary
from .backfill_keypoint_label_names import _canonicalize_label_seq
from .zarr_io import open_zarr_root


@dataclass
class BackfillResult:
    status: str
    reason: Optional[str] = None


JsonLogger = SharedJsonLogger
_run_id = make_run_id


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


def _resolve_log_dir(arg_log_dir: Optional[Path], roots: list[Path]) -> Path:
    return resolve_shared_log_dir(arg_log_dir, roots, log_subdir="backfill_keypoint_heading_fields")


def _iter_run_groups(
    root: zarr.Group,
    all_runs: bool,
    *,
    zarr_path: Optional[Path] = None,
    open_mode: Optional[str] = None,
) -> Iterable[tuple[str, str, zarr.Group]]:
    candidates = [
        ("keypoints_runs", "detection_success"),
        ("refined_keypoints_runs", "refined_success"),
        ("keypoints_refined_runs", "refined_success"),
    ]
    root_fs_path = zarr_path.expanduser().resolve() if zarr_path is not None else _root_fs_path(root)
    resolved_open_mode = open_mode or _open_mode(root)
    for parent_name, success_name in candidates:
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
                        yield f"{parent_name}/{run_name}", success_name, parent[run_name]
                    continue
                yield f"{parent_name}/{run_name}", success_name, run_group
            elif run_name in parent:
                yield f"{parent_name}/{run_name}", success_name, parent[run_name]


def _compute_chunks(run_group: zarr.Group, size: int) -> tuple[int, ...]:
    heading = run_group.get("heading")
    if heading is not None and heading.chunks:
        return heading.chunks
    success = run_group.get("detection_success") or run_group.get("refined_success")
    if success is not None and success.chunks:
        return success.chunks
    return (max(1, min(1024, int(size))),)


def _normalize_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text or None


def _coerce_mapping(value: object) -> Optional[dict[str, object]]:
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _normalize_text_sequence(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    labels: list[str] = []
    for item in value:
        text = _normalize_text(item)
        if text:
            labels.append(text)
    canonical_labels, _ = _canonicalize_label_seq(labels)
    return canonical_labels


def _extract_pose_schema_labels(value: object) -> list[str]:
    pose_schema = _coerce_mapping(value)
    if pose_schema is None:
        return []

    nodes = pose_schema.get("nodes")
    if isinstance(nodes, Sequence) and not isinstance(nodes, (str, bytes, bytearray)):
        labels: list[str] = []
        for node in nodes:
            if isinstance(node, Mapping):
                text = _normalize_text(node.get("name"))
            else:
                text = _normalize_text(node)
            if text:
                labels.append(text)
        if labels:
            return labels

    return _normalize_text_sequence(pose_schema.get("keypoint_labels"))


def _resolve_keypoint_labels(run_group: zarr.Group) -> list[str]:
    labels = _normalize_text_sequence(run_group.attrs.get("keypoint_labels"))
    if labels:
        return labels
    return _extract_pose_schema_labels(run_group.attrs.get("pose_schema"))


def _load_heading_point_rows(run_group: zarr.Group) -> tuple[Optional[str], Optional[np.ndarray]]:
    for array_name in ("keypoints_roi", "keypoints_img"):
        points_arr = run_group.get(array_name)
        if points_arr is None:
            continue
        points_vals = np.asarray(points_arr[:], dtype=np.float64)
        return array_name, points_vals
    return None, None


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape:
        return False
    return bool(np.array_equal(left, right, equal_nan=True))


def _has_temporal_heading_arrays(run_group: zarr.Group) -> bool:
    return all(
        name in run_group
        for name in ("heading_delta_prev_deg", "heading_delta_next_deg", "heading_temporal_outlier")
    )


def _has_temporal_heading_summary(run_group: zarr.Group) -> bool:
    summary_raw = run_group.attrs.get("summary_statistics", {})
    if not isinstance(summary_raw, dict):
        return False
    postprocess = summary_raw.get("postprocess")
    source = postprocess if isinstance(postprocess, dict) else summary_raw
    return all(
        key in source
        for key in (
            "heading_temporal_evaluable",
            "heading_temporal_outlier",
            "heading_temporal_outlier_rate_percent",
        )
    )


def _temporal_heading_status(run_group: zarr.Group) -> Optional[str]:
    summary_raw = run_group.attrs.get("summary_statistics", {})
    if not isinstance(summary_raw, dict):
        return None
    postprocess = summary_raw.get("postprocess")
    source = postprocess if isinstance(postprocess, dict) else summary_raw
    value = source.get("temporal_heading_status")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _backfill_heading_columns(
    run_group: zarr.Group,
    *,
    root: Optional[zarr.Group],
    success_array_name: str,
    overwrite_existing: bool,
    apply: bool,
) -> BackfillResult:
    success_arr = run_group.get(success_array_name)
    if success_arr is None:
        return BackfillResult(status="no_success", reason=f"{success_array_name} array missing")

    success_vals = np.asarray(success_arr[:], dtype=bool)
    total = int(success_vals.shape[0])

    points_name, points_vals = _load_heading_point_rows(run_group)
    if points_vals is None:
        return BackfillResult(status="no_points", reason="keypoints_roi/keypoints_img array missing")
    if points_vals.ndim != 3 or points_vals.shape[2] < 2:
        return BackfillResult(
            status="shape_mismatch",
            reason=f"{points_name} shape {points_vals.shape} is not NxKxD with D>=2",
        )
    if int(points_vals.shape[0]) != total:
        return BackfillResult(
            status="shape_mismatch",
            reason=f"{points_name} len {points_vals.shape[0]} != {success_array_name} len {success_vals.shape[0]}",
        )

    resolved = resolve_heading_computation_from_attrs(run_group.attrs)
    spec = resolved.spec
    if spec is None:
        return BackfillResult(status="no_heading_spec", reason="heading computation metadata missing")

    labels = _resolve_keypoint_labels(run_group)
    required_labels = dependent_keypoints(spec)
    if required_labels:
        label_set = {label for label in labels}
        missing = [label for label in required_labels if label not in label_set]
        if missing:
            return BackfillResult(
                status="invalid_labels",
                reason=f"missing heading labels: {', '.join(missing)}",
            )

    heading_vals = np.full(total, np.nan, dtype=np.float64)
    for idx in range(total):
        heading_vals[idx] = compute_heading_from_spec(
            spec,
            labels=labels,
            points=points_vals[idx],
        )

    source_arr = run_group.get("detection_source")
    if source_arr is not None:
        source_vals = np.asarray(source_arr[:], dtype=np.int8)
        if source_vals.shape[0] != total:
            return BackfillResult(
                status="shape_mismatch",
                reason=f"heading len {total} != detection_source len {source_vals.shape[0]}",
            )
    else:
        source_vals = np.zeros(total, dtype=np.int8)

    is_refined_run = success_array_name == "refined_success"
    heading = run_group.get("heading")
    existing_heading: Optional[np.ndarray] = None
    if heading is not None:
        existing_heading = np.asarray(heading[:], dtype=np.float64)
        if existing_heading.shape[0] != total:
            return BackfillResult(
                status="shape_mismatch",
                reason=f"heading len {existing_heading.shape[0]} != {success_array_name} len {total}",
            )
    has_heading_finite = "heading_finite" in run_group
    has_heading_usable = "heading_usable" in run_group
    has_legacy = "heading_valid" in run_group
    if is_refined_run:
        temporal_status = _temporal_heading_status(run_group)
        has_temporal_arrays = _has_temporal_heading_arrays(run_group)
        has_temporal_summary = _has_temporal_heading_summary(run_group)
        temporal_ready = (
            temporal_status == "disabled_sampled_import"
            if temporal_status == "disabled_sampled_import"
            else has_temporal_arrays and has_temporal_summary and temporal_status == "enabled"
        )
    else:
        temporal_status = None
        has_temporal_arrays = True
        has_temporal_summary = True
        temporal_ready = True
    heading_finite = np.isfinite(heading_vals)
    heading_usable = success_vals & (source_vals == 0) & heading_finite
    existing_heading_finite: Optional[np.ndarray] = None
    if has_heading_finite:
        existing_heading_finite = np.asarray(run_group["heading_finite"][:], dtype=bool)
        if existing_heading_finite.shape[0] != total:
            return BackfillResult(
                status="shape_mismatch",
                reason=f"heading_finite len {existing_heading_finite.shape[0]} != {success_array_name} len {total}",
            )
    existing_heading_usable: Optional[np.ndarray] = None
    if has_heading_usable:
        existing_heading_usable = np.asarray(run_group["heading_usable"][:], dtype=bool)
        if existing_heading_usable.shape[0] != total:
            return BackfillResult(
                status="shape_mismatch",
                reason=f"heading_usable len {existing_heading_usable.shape[0]} != {success_array_name} len {total}",
            )
    fields_match = (
        existing_heading is not None
        and _arrays_equal(existing_heading, heading_vals)
        and (existing_heading_finite is not None and _arrays_equal(existing_heading_finite, heading_finite))
        and (existing_heading_usable is not None and _arrays_equal(existing_heading_usable, heading_usable))
    )
    if (
        fields_match
        and not has_legacy
        and temporal_ready
        and not overwrite_existing
    ):
        return BackfillResult(status="skipped_existing")

    if apply:
        if has_legacy:
            del run_group["heading_valid"]
        chunks = _compute_chunks(run_group, total)
        if heading is not None:
            run_group["heading"][:] = heading_vals
        else:
            run_group.create_array(
                "heading",
                data=heading_vals,
                chunks=chunks,
                overwrite=True,
            )
        if is_refined_run:
            try:
                _update_postprocess_summary(run_group, root=root, print_summary=False)
            except Exception as exc:
                return BackfillResult(status="summary_error", reason=str(exc))
        else:
            if has_heading_finite:
                run_group["heading_finite"][:] = heading_finite
            else:
                run_group.create_array(
                    "heading_finite",
                    data=heading_finite,
                    chunks=chunks,
                    overwrite=True,
                )
            if has_heading_usable:
                run_group["heading_usable"][:] = heading_usable
            else:
                run_group.create_array(
                    "heading_usable",
                    data=heading_usable,
                    chunks=chunks,
                    overwrite=True,
                )

    return BackfillResult(status="ok")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute keypoint heading arrays from stored points plus metadata-driven "
            "heading semantics. Raw runs get refreshed heading/heading_finite/heading_usable "
            "and legacy heading_valid removal; refined runs also refresh temporal heading "
            "arrays and postprocess summary statistics."
        )
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="analysis",
        help="Filter zarr archives by purpose (default: analysis).",
    )
    parser.add_argument("--all-runs", action="store_true", help="Backfill all run groups (default: latest only).")
    add_log_args(
        parser,
        log_dir_help=(
            "Directory for JSONL logs (default: $PALETTE_LOG_ROOT/backfill_keypoint_heading_fields "
            "or <root>/logs/backfill_keypoint_heading_fields)."
        ),
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Force rewrites even when heading and derived fields already match.",
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
            log_path = log_dir / f"backfill_keypoint_heading_fields_{run_id}.jsonl"
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
        "no_heading_spec": 0,
        "no_success": 0,
        "no_points": 0,
        "invalid_labels": 0,
        "shape_mismatch": 0,
        "summary_error": 0,
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
            overwrite_existing=bool(args.overwrite_existing),
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
            for run_path, success_name, run_group in run_info:
                counts["runs_considered"] += 1
                result = _backfill_heading_columns(
                    run_group,
                    root=root,
                    success_array_name=success_name,
                    overwrite_existing=bool(args.overwrite_existing),
                    apply=bool(args.apply),
                )
                counts[result.status] += 1
                if logger is not None:
                    logger.log(
                        "run_group_checked",
                        zarr=str(zarr_path),
                        run_path=run_path,
                        success_array_name=success_name,
                        status=result.status,
                        reason=result.reason,
                        apply=bool(args.apply),
                        dry_run=not bool(args.apply),
                        changed=(result.status == "ok"),
                    )
        except Exception as exc:
            counts["errors"] += 1
            print(f"error: {zarr_path}: {exc}")
            if logger is not None:
                logger.log("zarr_error", zarr=str(zarr_path), error=str(exc))

    if not any_zarr:
        if logger is not None:
            logger.log("run_end", status="no_zarr", **counts)
            logger.close()
        print("No zarr files found.")
        return 1

    mode = "Applied" if args.apply else "Dry run"
    print(
        "Keypoint heading fields backfill: "
        f"scope={args.zarr_use} zarr_scanned={counts['zarr_scanned']} "
        f"runs_considered={counts['runs_considered']} filtered_zarr_use={counts['filtered_zarr_use']} "
        f"missing_runs={counts['missing_runs']} errors={counts['errors']}"
    )
    print(
        f"{mode}: ok={counts['ok']} skipped_existing={counts['skipped_existing']} "
        f"no_heading_spec={counts['no_heading_spec']} no_success={counts['no_success']} "
        f"no_points={counts['no_points']} invalid_labels={counts['invalid_labels']} "
        f"shape_mismatch={counts['shape_mismatch']} summary_error={counts['summary_error']}"
    )
    if logger is not None:
        logger.log(
            "run_end",
            status="ok" if counts["errors"] == 0 else "error",
            mode="apply" if args.apply else "dry-run",
            **counts,
        )
        logger.close()
    return 0 if counts["errors"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
