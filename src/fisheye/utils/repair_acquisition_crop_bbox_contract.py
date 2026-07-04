"""Repair acquisition crop-video crop-run bbox coordinate semantics.

Dry-run by default. The repair preserves existing crop-frame-normalized
``bbox_norm_coords`` values as ``bbox_crop_norm_coords`` and rewrites
``bbox_norm_coords`` to the canonical full-frame-normalized convention.
"""

from __future__ import annotations

from fisheye.shared.json_safety import write_json_atomic as _write_json
import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import zarr

from fisheye.registry.db import RegistryPaths
from fisheye.shared.crop_geometry import (
    bbox_img_xyxy_to_norm_cxcywh,
    bbox_roi_xyxy_to_img_xyxy,
    resolve_full_frame_shape,
)
from fisheye.shared.zarr_discovery import discover_filesystem_zarrs, discover_registry_zarrs, load_path_list
from fisheye.shared.zarr_helpers import open_zarr_group_direct


SCHEMA_VERSION = "palette.acquisition_crop_bbox_contract_repair.v1"
OLD_CROP_FRAME_SEMANTICS = {
    "realtime_detection_bbox_xywh_normalized_to_crop_video_frame",
    "pose_bbox_from_keypoint_extents_xywh_normalized_to_crop_video_frame",
}
CANONICAL_SEMANTICS = "bbox_xywh_normalized_to_full_frame"
LOCAL_SEMANTICS = "bbox_xywh_normalized_to_crop_roi_frame"


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _child_group_names(group: zarr.Group) -> list[str]:
    try:
        return sorted(str(name) for name in group.group_keys())
    except Exception:
        names: list[str] = []
        try:
            items = list(group.items())
        except Exception:
            return names
        for name, value in items:
            if hasattr(value, "keys") or hasattr(value, "group_keys"):
                names.append(str(name))
        return sorted(names)


def _array_or_none(group: zarr.Group, name: str) -> np.ndarray | None:
    if name not in group:
        return None
    return np.asarray(group[name][:])


def _chunks_for_bbox(group: zarr.Group, fallback_rows: int) -> tuple[int, int]:
    for name in ("bbox_norm_coords", "bbox_roi_xyxy", "bbox_img_xyxy", "bbox_crop_norm_coords"):
        arr = group.get(name)
        chunks = getattr(arr, "chunks", None)
        if chunks is not None and len(chunks) == 2:
            return int(chunks[0]), int(chunks[1])
    return max(1, min(8192, int(fallback_rows))), 4


def _write_array(group: zarr.Group, name: str, data: np.ndarray, *, chunks: tuple[int, ...]) -> None:
    old_attrs: dict[str, Any] = {}
    if name in group:
        try:
            old_attrs = dict(group[name].attrs)
        except Exception:
            old_attrs = {}
        del group[name]
    group.create_array(name, data=np.asarray(data), chunks=chunks, overwrite=True)
    for key, value in old_attrs.items():
        group[name].attrs[key] = value


def _arrays_equal_with_nan(left: np.ndarray | None, right: np.ndarray) -> bool:
    if left is None or left.shape != right.shape:
        return False
    left_nan = np.isnan(left)
    right_nan = np.isnan(right)
    if not np.array_equal(left_nan, right_nan):
        return False
    finite = np.isfinite(left) & np.isfinite(right)
    if not np.any(finite):
        return True
    return bool(np.allclose(left[finite], right[finite], rtol=0.0, atol=1e-6))


def _max_abs_diff(left: np.ndarray | None, right: np.ndarray) -> float | None:
    if left is None or left.shape != right.shape:
        return None
    finite = np.isfinite(left) & np.isfinite(right)
    if not np.any(finite):
        return 0.0
    return float(np.max(np.abs(left[finite] - right[finite])))


def _resolve_roi_shape(crop_group: zarr.Group, *, row_count: int) -> tuple[int, int]:
    roi_images = crop_group.get("roi_images")
    shape = getattr(roi_images, "shape", None)
    if shape is not None and len(shape) >= 3:
        return int(shape[1]), int(shape[2])
    roi_size = crop_group.attrs.get("roi_size")
    if isinstance(roi_size, (list, tuple)) and len(roi_size) == 2:
        return int(roi_size[0]), int(roi_size[1])
    source_crop_xywh = _array_or_none(crop_group, "source_crop_xywh")
    if source_crop_xywh is not None and source_crop_xywh.shape == (row_count, 4):
        width = float(np.nanmedian(source_crop_xywh[:, 2]))
        height = float(np.nanmedian(source_crop_xywh[:, 3]))
        if np.isfinite(width) and np.isfinite(height) and width > 0 and height > 0:
            return int(round(height)), int(round(width))
    raise ValueError("could not resolve crop/ROI frame shape")


def _should_repair(crop_group: zarr.Group) -> tuple[bool, str]:
    semantics = str(crop_group.attrs.get("bbox_norm_coords_semantics") or "")
    source_type = str(crop_group.attrs.get("source_type") or crop_group.attrs.get("source_pixels") or "")
    if semantics in OLD_CROP_FRAME_SEMANTICS:
        return True, "old_crop_frame_bbox_norm_semantics"
    if source_type == "acquisition_crop_video" and semantics not in {"", CANONICAL_SEMANTICS}:
        return True, "ambiguous_acquisition_crop_video_bbox_semantics"
    return False, "not_affected"


def _repair_crop_group(
    root: zarr.Group,
    crop_group: zarr.Group,
    *,
    crop_run: str,
    apply: bool,
) -> dict[str, Any]:
    should_repair, reason = _should_repair(crop_group)
    report: dict[str, Any] = {
        "crop_run": crop_run,
        "status": "skipped",
        "reason": reason,
        "apply": bool(apply),
        "rows": 0,
        "attrs_before": {
            "source_type": crop_group.attrs.get("source_type"),
            "source_pixels": crop_group.attrs.get("source_pixels"),
            "bbox_norm_coords_semantics": crop_group.attrs.get("bbox_norm_coords_semantics"),
        },
        "errors": [],
    }
    if not should_repair:
        return report

    required = ("bbox_norm_coords", "bbox_roi_xyxy", "source_crop_xywh")
    missing = [name for name in required if name not in crop_group]
    if missing:
        report["status"] = "blocked"
        report["errors"].append(f"missing required arrays: {missing}")
        return report

    try:
        local_norm_existing = np.asarray(crop_group["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4)
        bbox_roi = np.asarray(crop_group["bbox_roi_xyxy"][:], dtype=np.float64).reshape(-1, 4)
        source_crop_xywh = np.asarray(crop_group["source_crop_xywh"][:], dtype=np.float64).reshape(-1, 4)
        row_count = int(local_norm_existing.shape[0])
        if bbox_roi.shape[0] != row_count or source_crop_xywh.shape[0] != row_count:
            raise ValueError(
                "row count mismatch among bbox_norm_coords, bbox_roi_xyxy, and source_crop_xywh "
                f"({row_count}, {bbox_roi.shape[0]}, {source_crop_xywh.shape[0]})"
            )
        frame_height, frame_width = resolve_full_frame_shape(root)
        roi_height, roi_width = _resolve_roi_shape(crop_group, row_count=row_count)
        bbox_img = bbox_roi_xyxy_to_img_xyxy(
            bbox_roi,
            source_crop_xywh,
            roi_width=int(roi_width),
            roi_height=int(roi_height),
        )
        bbox_norm = bbox_img_xyxy_to_norm_cxcywh(
            bbox_img,
            width=int(frame_width),
            height=int(frame_height),
        )
    except Exception as exc:
        report["status"] = "blocked"
        report["errors"].append(str(exc))
        return report

    existing_crop_norm = _array_or_none(crop_group, "bbox_crop_norm_coords")
    existing_img = _array_or_none(crop_group, "bbox_img_xyxy")
    existing_norm = np.asarray(crop_group["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4)
    expected_attrs = {
        "bbox_norm_coords_semantics": CANONICAL_SEMANTICS,
        "bbox_img_xyxy_semantics": "bbox_xyxy_full_frame_pixels",
        "bbox_roi_xyxy_semantics": str(crop_group.attrs.get("bbox_roi_xyxy_semantics") or "bbox_xyxy_crop_roi_pixels"),
        "bbox_crop_norm_coords_semantics": str(crop_group.attrs.get("bbox_crop_norm_coords_semantics") or LOCAL_SEMANTICS),
        "bbox_norm_reference_width": int(frame_width),
        "bbox_norm_reference_height": int(frame_height),
        "bbox_norm_reference_space": "source_image",
        "bbox_img_xyxy_reference_width": int(frame_width),
        "bbox_img_xyxy_reference_height": int(frame_height),
    }
    crop_norm_ok = _arrays_equal_with_nan(
        None if existing_crop_norm is None else np.asarray(existing_crop_norm, dtype=np.float64).reshape(-1, 4),
        local_norm_existing,
    )
    img_ok = _arrays_equal_with_nan(
        None if existing_img is None else np.asarray(existing_img, dtype=np.float64).reshape(-1, 4),
        bbox_img,
    )
    norm_ok = _arrays_equal_with_nan(existing_norm, bbox_norm)
    attrs_ok = all(crop_group.attrs.get(key) == value for key, value in expected_attrs.items())

    report.update(
        {
            "status": "ok" if crop_norm_ok and img_ok and norm_ok and attrs_ok else ("updated" if apply else "would_update"),
            "rows": row_count,
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "roi_width": int(roi_width),
            "roi_height": int(roi_height),
            "bbox_crop_norm_status": "ok" if crop_norm_ok else ("updated" if apply else "would_write"),
            "bbox_img_xyxy_status": "ok" if img_ok else ("updated" if apply else "would_write"),
            "bbox_norm_coords_status": "ok" if norm_ok else ("updated" if apply else "would_rewrite"),
            "attrs_status": "ok" if attrs_ok else ("updated" if apply else "would_update"),
            "bbox_img_xyxy_max_abs_diff": _max_abs_diff(
                None if existing_img is None else np.asarray(existing_img, dtype=np.float64).reshape(-1, 4),
                bbox_img,
            ),
            "bbox_norm_coords_max_abs_diff": _max_abs_diff(existing_norm, bbox_norm),
            "attrs_expected": expected_attrs,
        }
    )
    if apply and report["status"] == "updated":
        chunks = _chunks_for_bbox(crop_group, row_count)
        _write_array(crop_group, "bbox_crop_norm_coords", local_norm_existing.astype(np.float32), chunks=chunks)
        _write_array(crop_group, "bbox_img_xyxy", bbox_img.astype(np.float32), chunks=chunks)
        _write_array(crop_group, "bbox_norm_coords", bbox_norm.astype(np.float32), chunks=chunks)
        for key, value in expected_attrs.items():
            crop_group.attrs[key] = value
    return report


def repair_acquisition_crop_bbox_contract(
    zarr_path: str | Path,
    *,
    apply: bool = False,
) -> dict[str, Any]:
    archive_path = Path(zarr_path).expanduser().resolve()
    root = open_zarr_group_direct(archive_path, mode="a" if apply else "r")
    crop_parent = root.get("crop_runs")
    reports: list[dict[str, Any]] = []
    if crop_parent is not None:
        for crop_run in _child_group_names(crop_parent):
            try:
                crop_group = crop_parent[crop_run]
            except Exception:
                continue
            reports.append(_repair_crop_group(root, crop_group, crop_run=crop_run, apply=apply))
    blocked = sum(1 for item in reports if item.get("status") == "blocked")
    changed = sum(1 for item in reports if item.get("status") in {"would_update", "updated"})
    affected = sum(1 for item in reports if item.get("reason") != "not_affected")
    return {
        "status": "failed" if blocked else "ok",
        "schema_version": SCHEMA_VERSION,
        "zarr_path": str(archive_path),
        "apply": bool(apply),
        "crop_run_count": len(reports),
        "affected_crop_run_count": int(affected),
        "changed_crop_run_count": int(changed),
        "blocked_crop_run_count": int(blocked),
        "crop_runs": reports,
    }


def _discover_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = [Path(value) for value in args.zarr_path]
    for list_path in args.path_list:
        paths.extend(load_path_list(Path(list_path)))
    if args.source == "none":
        return sorted({path.expanduser().resolve() for path in paths})
    if args.source == "filesystem":
        paths.extend(discover_filesystem_zarrs(args.scope, recursive=bool(args.recursive)))
    else:
        registry_path = (args.registry or RegistryPaths.from_env(Path.cwd()).path).expanduser().resolve()
        zarr_uses = ("analysis", "training") if args.zarr_use == "both" else (args.zarr_use,)
        for zarr_use in zarr_uses:
            suffix = None if zarr_use == "both" else f"_{zarr_use}.zarr"
            if zarr_use == "analysis":
                suffix = "_analysis.zarr"
            elif zarr_use == "training":
                suffix = "_training.zarr"
            paths.extend(
                discover_registry_zarrs(
                    registry_path=registry_path,
                    scope_paths=args.scope,
                    zarr_use=zarr_use,
                    path_contains=args.path_contains,
                    zarr_suffix=suffix,
                )
            )
    return sorted({path.expanduser().resolve() for path in paths})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", nargs="*", type=Path, help="Specific Zarr path(s) to inspect/repair.")
    parser.add_argument("--path-list", action="append", default=[], type=Path, help="File containing Zarr paths.")
    parser.add_argument("--scope", action="append", default=[], type=Path, help="Filesystem or registry scope path.")
    parser.add_argument("--recursive", action="store_true", help="Recursively discover filesystem Zarrs under --scope.")
    parser.add_argument("--source", choices=("none", "filesystem", "registry"), default="none")
    parser.add_argument("--registry", type=Path, help="Palette registry path for --source registry.")
    parser.add_argument("--zarr-use", choices=("analysis", "training", "both"), default="both")
    parser.add_argument("--path-contains", type=str, help="Registry zarr_path substring filter.")
    parser.add_argument("--apply", action="store_true", help="Apply repairs. Default is dry-run.")
    parser.add_argument("--output-json", type=Path, help="Write aggregate JSON report.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    paths = _discover_paths(args)
    if not paths:
        parser.error("No Zarr paths provided or discovered.")
    reports = [repair_acquisition_crop_bbox_contract(path, apply=bool(args.apply)) for path in paths]
    failed = sum(1 for report in reports if report.get("status") == "failed")
    result = {
        "status": "failed" if failed else "ok",
        "schema_version": SCHEMA_VERSION,
        "apply": bool(args.apply),
        "zarr_count": len(reports),
        "failed_zarr_count": int(failed),
        "affected_crop_run_count": int(sum(report.get("affected_crop_run_count", 0) for report in reports)),
        "changed_crop_run_count": int(sum(report.get("changed_crop_run_count", 0) for report in reports)),
        "blocked_crop_run_count": int(sum(report.get("blocked_crop_run_count", 0) for report in reports)),
        "zarrs": reports,
    }
    if args.output_json is not None:
        _write_json(args.output_json.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
