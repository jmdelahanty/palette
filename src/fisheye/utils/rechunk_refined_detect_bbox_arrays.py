"""Rechunk refined-detect bbox arrays to keep fixed-width bbox rows together."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import zarr

from fisheye.shared.refined_detect_curation import DEFAULT_REFINED_DETECT_ROW_CHUNK


SCHEMA_VERSION = "palette.refined_detect_bbox_rechunk.v1"
BBOX_ARRAY_NAMES = ("bbox_img_xyxy", "bbox_norm_coords")
BBOX_GROUP_NAMES = ("instances", "source_detections")


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _open_root(zarr_path: Path, *, apply: bool) -> zarr.Group:
    try:
        return zarr.open_group(str(zarr_path), mode="a" if apply else "r", use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(zarr_path), mode="a" if apply else "r")


def _normalize_group_path(path: str) -> str:
    value = str(path or "").strip().strip("/")
    if not value:
        raise ValueError("group path must be non-empty")
    if any(part == ".." for part in Path(value).parts):
        raise ValueError(f"group path must not contain '..': {path!r}")
    return value


def _array_chunks(array: Any) -> tuple[int, ...] | None:
    chunks = getattr(array, "chunks", None)
    if chunks is None:
        return None
    try:
        return tuple(int(value) for value in chunks)
    except TypeError:
        return None


def _target_chunks(shape: tuple[int, ...]) -> tuple[int, int]:
    return (max(1, min(int(shape[0]), DEFAULT_REFINED_DETECT_ROW_CHUNK)), 4)


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
    selected_runs = collection.attrs.get("selected_runs", [])
    paths = []
    for row in selected_runs:
        if not isinstance(row, dict):
            continue
        path = str(row.get("refined_group_path") or "").strip()
        if path:
            paths.append(_normalize_group_path(path))
    return resolved_id, sorted(set(paths))


def _rechunk_array(group: zarr.Group, name: str, *, apply: bool) -> dict[str, Any]:
    array = group[name]
    shape = tuple(int(value) for value in getattr(array, "shape", ()))
    chunks = _array_chunks(array)
    report: dict[str, Any] = {
        "array": name,
        "shape": list(shape),
        "chunks_before": list(chunks) if chunks is not None else None,
        "status": "ok",
    }
    if len(shape) != 2 or shape[1] != 4:
        report["status"] = "skipped"
        report["reason"] = "not a fixed-width [N, 4] bbox array"
        return report
    target = _target_chunks(shape)
    report["target_chunks"] = list(target)
    if chunks == target:
        return report
    if not apply:
        report["status"] = "would_rechunk"
        return report

    attrs = dict(array.attrs)
    data = np.asarray(array[:])
    del group[name]
    replacement = group.create_array(name, data=data, chunks=target, overwrite=True)
    for key, value in attrs.items():
        replacement.attrs[key] = value
    report["status"] = "rechunked"
    report["chunks_after"] = list(_array_chunks(replacement) or ())
    return report


def _rechunk_refined_group(root: zarr.Group, target_group_path: str, *, apply: bool) -> dict[str, Any]:
    path = _normalize_group_path(target_group_path)
    report: dict[str, Any] = {
        "target_group_path": path,
        "status": "ok",
        "groups": [],
        "errors": [],
    }
    try:
        refined = root[path]
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(f"refined group not found: {exc}")
        return report

    for group_name in BBOX_GROUP_NAMES:
        group = refined.get(group_name)
        if group is None:
            continue
        group_report = {"group": group_name, "arrays": []}
        for array_name in BBOX_ARRAY_NAMES:
            if array_name in group:
                group_report["arrays"].append(_rechunk_array(group, array_name, apply=apply))
        if group_report["arrays"]:
            report["groups"].append(group_report)
    return report


def rechunk_refined_detect_bbox_arrays(
    zarr_path: str | Path,
    *,
    target_group_paths: Sequence[str] = (),
    collection_id: str | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    archive_path = Path(zarr_path).expanduser().resolve()
    root = _open_root(archive_path, apply=apply)
    resolved_collection_id = None
    paths = [_normalize_group_path(path) for path in target_group_paths]
    if collection_id is not None:
        resolved_collection_id, collection_paths = _target_paths_from_collection(root, collection_id)
        paths.extend(collection_paths)
    if not paths:
        raise ValueError("Provide --target-group-path or --collection-id")
    unique_paths = sorted(set(paths))
    reports = [_rechunk_refined_group(root, path, apply=apply) for path in unique_paths]
    failed = sum(1 for report in reports if report["status"] == "failed")
    changed = sum(
        1
        for report in reports
        for group in report.get("groups", [])
        for array in group.get("arrays", [])
        if array.get("status") in {"would_rechunk", "rechunked"}
    )
    return {
        "status": "failed" if failed else "ok",
        "schema_version": SCHEMA_VERSION,
        "zarr_path": str(archive_path),
        "collection_id": resolved_collection_id or collection_id,
        "apply": bool(apply),
        "target_group_count": len(unique_paths),
        "changed_array_count": int(changed),
        "failed_group_count": int(failed),
        "target_groups": reports,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run or apply bbox-array rechunking for refined-detect instances/source_detections."
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
        help="Finalized clipped collection id; defaults are not inferred unless this option is provided.",
    )
    parser.add_argument("--apply", action="store_true", help="Rewrite arrays. Default is dry-run.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON report path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = rechunk_refined_detect_bbox_arrays(
        args.zarr_path,
        target_group_paths=args.target_group_path,
        collection_id=args.collection_id,
        apply=args.apply,
    )
    if args.output_json is not None:
        _write_json(args.output_json.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
