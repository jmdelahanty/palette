#!/usr/bin/env python3
"""Materialize or remove dense refined-subject mask compatibility caches."""

from __future__ import annotations

from fisheye.shared.batch_logging import utc_now as _utc_now
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import zarr

from fisheye.shared.mask_store import (
    is_mask_rle_stale,
    materialize_dense_masks_roi_from_store,
    refresh_bitpacked_mask_store_from_dense,
    refresh_component_rle_mask_store_from_dense,
    update_mask_storage_attrs,
)
from fisheye.shared.zarr_io import open_zarr_root


def _resolve_refined_run(root: zarr.Group, refined_run: str | None) -> tuple[str, zarr.Group]:
    parent = root.get("refined_subject_masks_runs")
    if parent is None:
        raise ValueError("Archive has no refined_subject_masks_runs group.")
    run_name = str(refined_run or parent.attrs.get("latest") or "")
    if not run_name:
        candidates = sorted(str(name) for name in parent.keys())
        if not candidates:
            raise ValueError("refined_subject_masks_runs has no runs.")
        run_name = candidates[-1]
    if run_name not in parent:
        raise ValueError(f"refined_subject_masks_runs/{run_name} not found.")
    return run_name, parent[run_name]


def _has_child(group: zarr.Group, name: str) -> bool:
    try:
        return group.get(name) is not None
    except Exception:
        return name in group


def _run_source_path(run_group: zarr.Group) -> str:
    return str(getattr(run_group, "name", "") or "")


def _create_dense_array_from_store(
    run_group: zarr.Group,
    *,
    chunk_size: int,
) -> dict[str, object]:
    return materialize_dense_masks_roi_from_store(
        run_group,
        chunk_size=max(1, int(chunk_size)),
        overwrite=True,
        source_path=_run_source_path(run_group),
    )


def materialize_refined_subject_mask_store(
    zarr_path: Path,
    *,
    refined_run: str | None = None,
    apply: bool = False,
    overwrite: bool = False,
    delete_dense: bool = False,
    refresh_rle: bool = False,
    refresh_bitpacked: bool = False,
    components: Sequence[str] | None = None,
    rows: Sequence[int] | None = None,
    allow_delete_last_mask_store: bool = False,
    chunk_size: int = 256,
) -> dict[str, object]:
    """Materialize dense masks, refresh compact RLE, or remove dense caches."""

    action_count = int(bool(delete_dense)) + int(bool(refresh_rle)) + int(bool(refresh_bitpacked))
    if action_count > 1:
        raise ValueError("--delete-dense, --refresh-rle, and --refresh-bitpacked are mutually exclusive.")
    if components and not (refresh_rle or refresh_bitpacked):
        raise ValueError("--components is only valid with --refresh-rle or --refresh-bitpacked.")
    if rows and not refresh_bitpacked:
        raise ValueError("--rows is only valid with --refresh-bitpacked.")

    root = open_zarr_root(zarr_path, mode="a" if apply else "r")
    run_name, run_group = _resolve_refined_run(root, refined_run)
    has_dense = _has_child(run_group, "masks_roi")
    has_bitpacked = _has_child(run_group, "mask_bitpacked")
    has_rle = _has_child(run_group, "mask_rle")

    summary: dict[str, object] = {
        "zarr_path": str(zarr_path),
        "refined_run": str(run_name),
        "apply": bool(apply),
        "delete_dense": bool(delete_dense),
        "refresh_rle": bool(refresh_rle),
        "refresh_bitpacked": bool(refresh_bitpacked),
        "components": [str(value) for value in components] if components else [],
        "rows": [int(value) for value in rows] if rows else [],
        "overwrite": bool(overwrite),
        "has_dense_before": bool(has_dense),
        "has_bitpacked": bool(has_bitpacked),
        "has_rle": bool(has_rle),
    }

    if refresh_rle:
        if not has_dense:
            raise ValueError("Cannot refresh compact mask_rle because refined run has no dense masks_roi source.")
        if not apply:
            summary.update(
                {
                    "status": "would_refresh_rle",
                    "has_dense_after": True,
                    "has_rle_after": True,
                    "mask_rle_stale_after": bool(run_group.attrs.get("mask_rle_stale", False)),
                }
            )
            return summary
        write_summary = refresh_component_rle_mask_store_from_dense(
            run_group,
            encode_row_chunk_size=max(1, int(chunk_size)),
            source_path=_run_source_path(run_group),
            clear_stale=True,
            refresh_components=components,
        )
        summary.update(write_summary)
        summary["has_dense_after"] = True
        summary["has_rle_after"] = True
        return summary

    if refresh_bitpacked:
        if not has_dense:
            raise ValueError("Cannot refresh compact mask_bitpacked because refined run has no dense masks_roi source.")
        if not apply:
            summary.update(
                {
                    "status": "would_refresh_bitpacked" if has_bitpacked else "would_write_bitpacked",
                    "has_dense_after": True,
                    "has_bitpacked_after": True,
                }
            )
            return summary
        write_summary = refresh_bitpacked_mask_store_from_dense(
            run_group,
            encode_row_chunk_size=max(1, int(chunk_size)),
            source_path=_run_source_path(run_group),
            refresh_components=components,
            refresh_rows=rows,
        )
        summary.update(write_summary)
        summary["has_dense_after"] = True
        summary["has_bitpacked_after"] = True
        return summary

    if delete_dense:
        if not has_dense:
            summary.update({"status": "already_absent", "has_dense_after": False})
            return summary
        if not (has_bitpacked or has_rle) and not allow_delete_last_mask_store:
            raise ValueError(
                "Refusing to delete masks_roi because no compact mask_bitpacked or mask_rle store exists. "
                "Pass allow_delete_last_mask_store=True only for destructive repair workflows."
            )
        if has_rle and not has_bitpacked and is_mask_rle_stale(run_group) and not allow_delete_last_mask_store:
            raise ValueError(
                "Refusing to delete masks_roi because compact mask_rle is marked stale. "
                "Run with --refresh-rle --apply first, or pass allow_delete_last_mask_store=True "
                "only for destructive repair workflows."
            )
        if not apply:
            summary.update({"status": "would_delete", "has_dense_after": bool(has_dense)})
            return summary
        del run_group["masks_roi"]
        run_group.attrs["masks_roi_deleted_at_utc"] = _utc_now()
        run_group.attrs["masks_roi_deleted_reason"] = "materialize_refined_subject_mask_store --delete-dense"
        update_mask_storage_attrs(run_group, has_dense=False, has_bitpacked=has_bitpacked, has_rle=has_rle)
        summary.update({"status": "deleted", "has_dense_after": False})
        return summary

    if has_dense and not overwrite:
        summary.update({"status": "existing", "has_dense_after": True})
        return summary

    if not (has_bitpacked or has_rle):
        if has_dense:
            summary.update(
                {
                    "status": "missing_compact_source",
                    "reason": "masks_roi exists, but no compact mask_bitpacked or mask_rle source is available",
                    "has_dense_after": True,
                }
            )
            return summary
        raise ValueError("Cannot materialize masks_roi because refined run has no mask_bitpacked or mask_rle store.")

    if not apply:
        action = "would_refresh" if has_dense else "would_materialize"
        summary.update({"status": action, "has_dense_after": bool(has_dense)})
        return summary

    if has_dense:
        del run_group["masks_roi"]
    write_summary = _create_dense_array_from_store(run_group, chunk_size=max(1, int(chunk_size)))
    summary.update(write_summary)
    summary["has_dense_after"] = True
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Palette analysis or training Zarr archive.")
    parser.add_argument("--refined-run", help="Target refined_subject_masks_runs/<run>. Defaults to latest.")
    parser.add_argument("--apply", action="store_true", help="Apply changes. Default is dry-run.")
    parser.add_argument("--overwrite", action="store_true", help="Refresh an existing dense masks_roi cache.")
    parser.add_argument(
        "--delete-dense",
        action="store_true",
        help="Delete the dense masks_roi compatibility cache instead of materializing it.",
    )
    parser.add_argument(
        "--refresh-rle",
        action="store_true",
        help="Regenerate compact mask_rle from the current dense masks_roi and clear stale compact markers.",
    )
    parser.add_argument(
        "--refresh-bitpacked",
        action="store_true",
        help="Regenerate or update compact mask_bitpacked from the current dense masks_roi.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        help="With --refresh-rle or --refresh-bitpacked, refresh only these mask component names.",
    )
    parser.add_argument(
        "--rows",
        nargs="+",
        type=int,
        help="With --refresh-bitpacked, refresh only these row indices.",
    )
    parser.add_argument(
        "--allow-delete-last-mask-store",
        action="store_true",
        help="Allow deleting masks_roi even when no compact mask_rle store exists. Dangerous; default refuses.",
    )
    parser.add_argument("--chunk-size", type=int, default=256, help="Rows to materialize per read/write batch.")
    parser.add_argument("--json", action="store_true", help="Emit compact JSON.")
    args = parser.parse_args(argv)

    summary = materialize_refined_subject_mask_store(
        args.zarr_path,
        refined_run=args.refined_run,
        apply=bool(args.apply),
        overwrite=bool(args.overwrite),
        delete_dense=bool(args.delete_dense),
        refresh_rle=bool(args.refresh_rle),
        refresh_bitpacked=bool(args.refresh_bitpacked),
        components=args.components,
        rows=args.rows,
        allow_delete_last_mask_store=bool(args.allow_delete_last_mask_store),
        chunk_size=max(1, int(args.chunk_size)),
    )
    print(json.dumps(summary, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
