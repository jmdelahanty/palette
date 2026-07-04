"""Backfill refined-detect pixel-space bbox arrays from normalized boxes.

The canonical refined-detect contract is:

- ``bbox_norm_coords``: normalized ``cxcywh`` relative to the detector/inference
  reference dimensions recorded in attrs.
- ``bbox_img_xyxy``: source-image pixel-space ``xyxy`` for display/editing.

Older sparse refined runs could materialize ``bbox_img_xyxy`` in inference-space
pixels when the source detect run carried both inference and source dimensions.
This utility is dry-run by default and rewrites only ``bbox_img_xyxy`` plus
coordinate metadata when ``--apply`` is provided.
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

from fisheye.shared.refined_detect_curation import (
    _bbox_norm_to_img_xyxy_with_missing,
    _resolve_bbox_norm_reference_dimensions,
    _resolve_image_dimensions,
    _write_bbox_coordinate_attrs,
    _write_common_array,
)
from fisheye.shared.zarr_helpers import open_zarr_group_direct


SCHEMA_VERSION = "palette.refined_detect_bbox_img_xyxy_backfill.v1"
SURFACE_GROUP_NAMES = ("instances", "source_detections")
COORDINATE_ATTR_KEYS = (
    "bbox_img_xyxy_coordinate_space",
    "bbox_img_xyxy_reference_width",
    "bbox_img_xyxy_reference_height",
    "bbox_norm_coords_format",
    "bbox_norm_coords_coordinate_space",
    "bbox_norm_reference_width",
    "bbox_norm_reference_height",
    "bbox_norm_reference_space",
)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _normalize_group_path(path: str) -> str:
    value = str(path or "").strip().strip("/")
    if not value:
        raise ValueError("group path must be non-empty")
    if any(part == ".." for part in Path(value).parts):
        raise ValueError(f"group path must not contain '..': {path!r}")
    return value


def _is_group_like(value: Any) -> bool:
    return hasattr(value, "group_keys") or hasattr(value, "keys")


def _child_group_names(group: zarr.Group) -> list[str]:
    keys_fn = getattr(group, "group_keys", None)
    if callable(keys_fn):
        try:
            return sorted(str(key) for key in keys_fn())
        except Exception:
            pass
    names: list[str] = []
    try:
        keys = list(group.keys())
    except Exception:
        return names
    for key in keys:
        try:
            child = group[key]
        except Exception:
            continue
        if _is_group_like(child):
            names.append(str(key))
    return sorted(names)


def _iter_refined_detect_group_paths(root: zarr.Group, *, prefix: str = "") -> list[str]:
    paths: list[str] = []
    for name in _child_group_names(root):
        path = f"{prefix}/{name}".strip("/")
        try:
            child = root[name]
        except Exception:
            continue
        if _is_refined_detect_group(child):
            paths.append(path)
        paths.extend(_iter_refined_detect_group_paths(child, prefix=path))
    return sorted(set(paths))


def _is_refined_detect_group(group: zarr.Group) -> bool:
    for surface_name in SURFACE_GROUP_NAMES:
        surface = group.get(surface_name)
        if surface is not None and "bbox_norm_coords" in surface:
            return True
    return False


def _collection_group(root: zarr.Group, collection_id: str | None) -> tuple[str, zarr.Group]:
    resolved_id = str(collection_id or "").strip()
    if not resolved_id:
        refined_parent = root.get("refined_detect_runs")
        if refined_parent is not None:
            resolved_id = str(refined_parent.attrs.get("latest_collection") or "").strip()
    if not resolved_id:
        raise ValueError("No collection id provided and refined_detect_runs.latest_collection is unset")
    path = f"experiment_index/finalized_runs/{resolved_id}"
    try:
        return resolved_id, root[path]
    except Exception as exc:
        raise ValueError(f"Finalized collection not found: {path}") from exc


def _target_paths_from_collection(root: zarr.Group, collection_id: str | None) -> tuple[str, list[str]]:
    resolved_id, collection = _collection_group(root, collection_id)
    paths: list[str] = []
    selected_runs = collection.attrs.get("selected_runs", [])
    for row in selected_runs:
        if not isinstance(row, dict):
            continue
        path = str(row.get("refined_group_path") or "").strip()
        if path:
            paths.append(_normalize_group_path(path))
    return resolved_id, sorted(set(paths))


def _expected_coordinate_attrs(
    *,
    bbox_img_width: int,
    bbox_img_height: int,
    bbox_norm_reference_width: Optional[int],
    bbox_norm_reference_height: Optional[int],
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "bbox_img_xyxy_coordinate_space": "source_image_xyxy",
        "bbox_img_xyxy_reference_width": int(bbox_img_width),
        "bbox_img_xyxy_reference_height": int(bbox_img_height),
        "bbox_norm_coords_format": "cxcywh",
        "bbox_norm_coords_coordinate_space": "normalized",
    }
    if bbox_norm_reference_width is not None and bbox_norm_reference_height is not None:
        attrs["bbox_norm_reference_width"] = int(bbox_norm_reference_width)
        attrs["bbox_norm_reference_height"] = int(bbox_norm_reference_height)
        attrs["bbox_norm_reference_space"] = (
            "inference_image"
            if (
                int(bbox_norm_reference_width) != int(bbox_img_width)
                or int(bbox_norm_reference_height) != int(bbox_img_height)
            )
            else "source_image"
        )
    return attrs


def _attrs_need_update(group: zarr.Group, expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if group.attrs.get(key) != value:
            return True
    return False


def _max_abs_diff(existing: np.ndarray | None, desired: np.ndarray) -> float | None:
    if existing is None or existing.shape != desired.shape:
        return None
    finite = np.isfinite(existing) & np.isfinite(desired)
    if not np.any(finite):
        return 0.0
    return float(np.max(np.abs(existing[finite] - desired[finite])))


def _arrays_equal_with_nan(existing: np.ndarray | None, desired: np.ndarray) -> bool:
    if existing is None or existing.shape != desired.shape:
        return False
    existing_nan = np.isnan(existing)
    desired_nan = np.isnan(desired)
    if not np.array_equal(existing_nan, desired_nan):
        return False
    finite = np.isfinite(existing) & np.isfinite(desired)
    if not np.any(finite):
        return True
    return bool(np.array_equal(existing[finite], desired[finite]))


def _rewrite_bbox_img_xyxy(group: zarr.Group, desired: np.ndarray) -> None:
    old_attrs: dict[str, Any] = {}
    if "bbox_img_xyxy" in group:
        try:
            old_attrs = dict(group["bbox_img_xyxy"].attrs)
        except Exception:
            old_attrs = {}
    _write_common_array(group, "bbox_img_xyxy", desired)
    for key, value in old_attrs.items():
        group["bbox_img_xyxy"].attrs[key] = value


def _surface_report(
    surface: zarr.Group,
    *,
    surface_name: str,
    bbox_img_width: int,
    bbox_img_height: int,
    bbox_norm_reference_width: Optional[int],
    bbox_norm_reference_height: Optional[int],
    apply: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "surface": surface_name,
        "status": "ok",
        "rows": 0,
        "array_status": "ok",
        "attrs_status": "ok",
    }
    if "bbox_norm_coords" not in surface:
        report["status"] = "skipped"
        report["reason"] = "missing bbox_norm_coords"
        return report

    bbox_norm = np.asarray(surface["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4)
    desired = _bbox_norm_to_img_xyxy_with_missing(
        bbox_norm,
        width=int(bbox_img_width),
        height=int(bbox_img_height),
    )
    existing = (
        np.asarray(surface["bbox_img_xyxy"][:], dtype=np.float64).reshape(-1, 4)
        if "bbox_img_xyxy" in surface
        else None
    )
    expected_attrs = _expected_coordinate_attrs(
        bbox_img_width=bbox_img_width,
        bbox_img_height=bbox_img_height,
        bbox_norm_reference_width=bbox_norm_reference_width,
        bbox_norm_reference_height=bbox_norm_reference_height,
    )
    array_equal = _arrays_equal_with_nan(existing, desired)
    attrs_current = {key: surface.attrs.get(key) for key in COORDINATE_ATTR_KEYS if key in surface.attrs}
    attrs_need_update = _attrs_need_update(surface, expected_attrs)
    report.update(
        {
            "rows": int(bbox_norm.shape[0]),
            "bbox_img_reference_width": int(bbox_img_width),
            "bbox_img_reference_height": int(bbox_img_height),
            "bbox_norm_reference_width": int(bbox_norm_reference_width)
            if bbox_norm_reference_width is not None
            else None,
            "bbox_norm_reference_height": int(bbox_norm_reference_height)
            if bbox_norm_reference_height is not None
            else None,
            "max_abs_diff": _max_abs_diff(existing, desired),
            "attrs_before": attrs_current,
            "attrs_expected": expected_attrs,
        }
    )

    if not array_equal:
        report["array_status"] = "rewritten" if apply else "would_rewrite"
    if attrs_need_update:
        report["attrs_status"] = "updated" if apply else "would_update"
    if not array_equal or attrs_need_update:
        report["status"] = "updated" if apply else "would_update"

    if apply and (not array_equal or attrs_need_update):
        if not array_equal:
            _rewrite_bbox_img_xyxy(surface, desired)
        _write_bbox_coordinate_attrs(
            surface,
            bbox_img_width=int(bbox_img_width),
            bbox_img_height=int(bbox_img_height),
            bbox_norm_reference_width=bbox_norm_reference_width,
            bbox_norm_reference_height=bbox_norm_reference_height,
        )
    return report


def _sync_run_coordinate_attrs(refined: zarr.Group) -> bool:
    for surface_name in SURFACE_GROUP_NAMES:
        surface = refined.get(surface_name)
        if surface is None:
            continue
        copied = False
        for key in COORDINATE_ATTR_KEYS:
            if key in surface.attrs:
                refined.attrs[key] = surface.attrs[key]
                copied = True
        if copied:
            refined.attrs["bbox_coordinate_contract_version"] = "refined_detect_bbox_coordinates_v2"
            return True
    return False


def _run_attrs_need_update(refined: zarr.Group) -> bool:
    for surface_name in SURFACE_GROUP_NAMES:
        surface = refined.get(surface_name)
        if surface is None:
            continue
        expected = {key: surface.attrs.get(key) for key in COORDINATE_ATTR_KEYS if key in surface.attrs}
        if not expected:
            continue
        if refined.attrs.get("bbox_coordinate_contract_version") != "refined_detect_bbox_coordinates_v2":
            return True
        for key, value in expected.items():
            if refined.attrs.get(key) != value:
                return True
        return False
    return False


def _backfill_refined_group(
    root: zarr.Group,
    target_group_path: str,
    *,
    apply: bool,
) -> dict[str, Any]:
    path = _normalize_group_path(target_group_path)
    report: dict[str, Any] = {
        "target_group_path": path,
        "status": "ok",
        "surfaces": [],
        "run_attrs_status": "ok",
        "errors": [],
    }
    try:
        refined = root[path]
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(f"refined group not found: {exc}")
        return report

    width, height = _resolve_image_dimensions(root, refined_run=refined)
    norm_width, norm_height = _resolve_bbox_norm_reference_dimensions(root, refined_run=refined)
    if width is None or height is None or width <= 0 or height <= 0:
        report["status"] = "failed"
        report["errors"].append("could not resolve positive source-image dimensions")
        return report

    for surface_name in SURFACE_GROUP_NAMES:
        surface = refined.get(surface_name)
        if surface is None:
            continue
        report["surfaces"].append(
            _surface_report(
                surface,
                surface_name=surface_name,
                bbox_img_width=int(width),
                bbox_img_height=int(height),
                bbox_norm_reference_width=norm_width,
                bbox_norm_reference_height=norm_height,
                apply=apply,
            )
        )

    surface_attrs_will_update = any(
        surface.get("attrs_status") in {"would_update", "updated"}
        for surface in report["surfaces"]
    )
    run_attrs_need_update = _run_attrs_need_update(refined) or surface_attrs_will_update
    if run_attrs_need_update:
        report["run_attrs_status"] = "updated" if apply else "would_update"
        if apply:
            _sync_run_coordinate_attrs(refined)

    if any(surface.get("status") in {"would_update", "updated"} for surface in report["surfaces"]) or run_attrs_need_update:
        report["status"] = "updated" if apply else "would_update"
    return report


def backfill_refined_detect_bbox_img_xyxy(
    zarr_path: str | Path,
    *,
    target_group_paths: Sequence[str] = (),
    collection_id: str | None = None,
    discover: bool = False,
    apply: bool = False,
) -> dict[str, Any]:
    archive_path = Path(zarr_path).expanduser().resolve()
    root = open_zarr_group_direct(archive_path, mode="a" if apply else "r")
    resolved_collection_id = None
    paths = [_normalize_group_path(path) for path in target_group_paths]
    if collection_id is not None:
        resolved_collection_id, collection_paths = _target_paths_from_collection(root, collection_id)
        paths.extend(collection_paths)
    if discover:
        paths.extend(_iter_refined_detect_group_paths(root))
    if not paths:
        if discover:
            return {
                "status": "ok",
                "schema_version": SCHEMA_VERSION,
                "zarr_path": str(archive_path),
                "collection_id": resolved_collection_id or collection_id,
                "apply": bool(apply),
                "target_group_count": 0,
                "changed_group_count": 0,
                "changed_surface_count": 0,
                "failed_group_count": 0,
                "target_groups": [],
            }
        raise ValueError("Provide --target-group-path, --collection-id, or --discover")
    unique_paths = sorted(set(paths))
    reports = [_backfill_refined_group(root, path, apply=apply) for path in unique_paths]
    failed = sum(1 for report in reports if report["status"] == "failed")
    changed_surfaces = sum(
        1
        for report in reports
        for surface in report.get("surfaces", [])
        if surface.get("status") in {"would_update", "updated"}
    )
    changed_groups = sum(1 for report in reports if report["status"] in {"would_update", "updated"})
    return {
        "status": "failed" if failed else "ok",
        "schema_version": SCHEMA_VERSION,
        "zarr_path": str(archive_path),
        "collection_id": resolved_collection_id or collection_id,
        "apply": bool(apply),
        "target_group_count": len(unique_paths),
        "changed_group_count": int(changed_groups),
        "changed_surface_count": int(changed_surfaces),
        "failed_group_count": int(failed),
        "target_groups": reports,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run or apply source-image bbox_img_xyxy backfill for refined-detect runs."
    )
    parser.add_argument("zarr_path", type=Path, help="Analysis Zarr path")
    parser.add_argument(
        "--target-group-path",
        action="append",
        default=[],
        help="Refined run group path. May be provided multiple times.",
    )
    parser.add_argument(
        "--collection-id",
        default=None,
        help="Finalized clipped collection id; reads selected refined_group_path attrs.",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Discover all refined-detect groups that have instances/source_detections bbox_norm_coords.",
    )
    parser.add_argument("--apply", action="store_true", help="Rewrite bbox_img_xyxy and coordinate attrs.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON report path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = backfill_refined_detect_bbox_img_xyxy(
        args.zarr_path,
        target_group_paths=args.target_group_path,
        collection_id=args.collection_id,
        discover=bool(args.discover),
        apply=bool(args.apply),
    )
    if args.output_json is not None:
        _write_json(args.output_json.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
