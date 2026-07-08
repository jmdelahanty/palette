"""Create crop-run-compatible proxies for clipped collection ROI caches."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
import zarr

from fisheye.shared.clipped_collection_flat_roi_cache import CLIPPED_COLLECTION_ROW_INDEX_SCHEMA
from fisheye.shared.flat_roi_cache import load_flat_roi_cache_manifest, open_flat_roi_cache
from fisheye.shared.row_lineage import direct_source_crop_row_ids
from fisheye.shared.zarr.chunk_profiles import create_geometry_preload_array
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    require_runs_parent,
)


PROXY_CROP_RUN_SCHEMA = "palette_clipped_collection_proxy_crop_run_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_component(value: object, *, default: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text or default


def _resolve_relative_or_absolute(value: object, *, base: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Expected a non-empty path value.")
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    return base / path


def _relative_or_absolute(path: Path, *, base: Path) -> str:
    try:
        return os.path.relpath(path, start=base)
    except ValueError:
        return str(path)


def _column_np(table: Any, name: str, dtype: Any) -> np.ndarray:
    if name not in table.column_names:
        raise ValueError(f"Row-index parquet is missing required column {name!r}.")
    return np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=dtype)


def _column_pylist(table: Any, name: str) -> list[Any]:
    if name not in table.column_names:
        raise ValueError(f"Row-index parquet is missing required column {name!r}.")
    return table[name].to_pylist()


def _unique_nonempty(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _single_unique(values: Sequence[Any], *, name: str) -> str:
    unique = _unique_nonempty(values)
    if len(unique) != 1:
        raise ValueError(f"Expected exactly one unique {name}; found {unique!r}.")
    return unique[0]


def _default_proxy_run_name(manifest: Mapping[str, Any], table: Any) -> str:
    source = manifest.get("source") if isinstance(manifest.get("source"), Mapping) else {}
    collection_id = _sanitize_component(source.get("collection_id"), default="collection")
    clip_id = _sanitize_component(_single_unique(_column_pylist(table, "clip_id"), name="clip_id"), default="clip")
    cache_key = _sanitize_component(str(manifest.get("cache_key") or "")[:10], default="cache")
    return f"crop_proxy_{collection_id}_{clip_id}_{cache_key}"


def _write_array(group: zarr.Group, name: str, data: np.ndarray) -> None:
    create_geometry_preload_array(group, name, data=np.asarray(data), overwrite=True)


def _write_alias_manifest(
    *,
    source_manifest_path: Path,
    source_manifest: Mapping[str, Any],
    proxy_run_name: str,
    alias_manifest_path: Path,
    row_index_path: Path,
) -> dict[str, Any]:
    alias_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    alias = json.loads(json.dumps(dict(source_manifest)))
    source = alias.setdefault("source", {})
    if not isinstance(source, dict):
        raise ValueError("Flat ROI cache manifest source must be a mapping.")
    source["crop_run_name"] = proxy_run_name
    source["proxy_crop_run_name"] = proxy_run_name
    source["proxy_crop_run_schema"] = PROXY_CROP_RUN_SCHEMA
    source["source_manifest_path"] = str(source_manifest_path)

    array = alias.get("array")
    if not isinstance(array, dict):
        raise ValueError("Flat ROI cache manifest array must be a mapping.")
    source_bin = _resolve_relative_or_absolute(array.get("bin_path"), base=source_manifest_path.parent).resolve()
    array["bin_path"] = _relative_or_absolute(source_bin, base=alias_manifest_path.parent)

    row_index = alias.get("row_index")
    if isinstance(row_index, dict):
        row_index["path"] = _relative_or_absolute(row_index_path.resolve(), base=alias_manifest_path.parent)
        row_index["source_manifest_path"] = str(source_manifest_path)

    alias["manifest_path"] = str(alias_manifest_path)
    alias["proxy_alias"] = {
        "schema": PROXY_CROP_RUN_SCHEMA,
        "created_at_utc": _utc_now(),
        "source_manifest_path": str(source_manifest_path),
        "proxy_crop_run_name": proxy_run_name,
    }
    tmp = alias_manifest_path.with_name(f"{alias_manifest_path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(alias, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, alias_manifest_path)
    return alias


def create_clipped_collection_proxy_crop_run(
    *,
    zarr_path: str | Path,
    manifest_path: str | Path,
    proxy_run_name: str | None = None,
    alias_manifest_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a geometry-only proxy crop run and cache-manifest alias.

    The proxy preserves normal Palette crop-run row contracts while adapting a
    clipped collection flat cache for explicit model-runner consumption.
    """

    archive_path = Path(zarr_path).expanduser().resolve()
    source_manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = load_flat_roi_cache_manifest(source_manifest_path)

    row_index_payload = manifest.get("row_index")
    if not isinstance(row_index_payload, Mapping):
        raise ValueError(f"Flat ROI cache manifest is missing row_index metadata: {source_manifest_path}")
    if row_index_payload.get("schema") != CLIPPED_COLLECTION_ROW_INDEX_SCHEMA:
        raise ValueError(
            f"Unsupported row index schema {row_index_payload.get('schema')!r}; "
            f"expected {CLIPPED_COLLECTION_ROW_INDEX_SCHEMA!r}."
        )
    row_index_path = _resolve_relative_or_absolute(row_index_payload.get("path"), base=source_manifest_path.parent).resolve()
    if not row_index_path.exists():
        raise FileNotFoundError(f"Row-index parquet not found: {row_index_path}")

    table = pq.read_table(row_index_path)
    if table.num_rows <= 0:
        raise ValueError(f"Row-index parquet has no rows: {row_index_path}")
    array_payload = manifest.get("array")
    if not isinstance(array_payload, Mapping):
        raise ValueError("Flat ROI cache manifest is missing array metadata.")
    shape = [int(value) for value in array_payload.get("shape") or []]
    if len(shape) != 3:
        raise ValueError(f"Flat ROI cache array.shape must be [N,H,W]; got {shape!r}.")
    if int(table.num_rows) != int(shape[0]):
        raise ValueError(f"Row-index row count {table.num_rows} does not match cache shape[0] {shape[0]}.")

    roi_x = _column_np(table, "roi_x", np.int32)
    roi_y = _column_np(table, "roi_y", np.int32)
    roi_w = _column_np(table, "roi_w", np.int32)
    roi_h = _column_np(table, "roi_h", np.int32)
    if np.any(roi_w != int(shape[2])) or np.any(roi_h != int(shape[1])):
        raise ValueError("Row-index roi_w/roi_h values do not match flat-cache array shape.")
    roi_coordinates_full = np.stack((roi_x, roi_y), axis=1).astype(np.int32, copy=False)

    frame_indices = _column_np(table, "parent_frame_index", np.int64)
    clip_indices = _column_np(table, "clip_index", np.int64)
    clip_local_frame_indices = _column_np(table, "clip_local_frame_index", np.int64)
    refined_row_ids = (
        _column_np(table, "refined_row_id", np.int64)
        if "refined_row_id" in table.column_names
        else _column_np(table, "refined_instance_row_index", np.int64)
    )
    source_detect_row_index = _column_np(table, "source_detect_row_index", np.int64)
    roi_row_index = _column_np(table, "roi_row_index", np.int64)
    clip_id = _single_unique(_column_pylist(table, "clip_id"), name="clip_id")
    video_path = _single_unique(_column_pylist(table, "video_path"), name="video_path")

    resolved_proxy_run = proxy_run_name or _default_proxy_run_name(manifest, table)
    resolved_proxy_run = _sanitize_component(resolved_proxy_run, default="crop_proxy")
    if "/" in resolved_proxy_run:
        raise ValueError("proxy_run_name must be a single Zarr group name, not a path.")

    if alias_manifest_path is None:
        alias_manifest = source_manifest_path.with_name(
            f"{source_manifest_path.stem}__{resolved_proxy_run}.alias.json"
        )
    else:
        alias_manifest = Path(alias_manifest_path).expanduser().resolve()

    root = zarr.open_group(str(archive_path), mode="a", use_consolidated=False)
    crop_parent = require_runs_parent(root, "crop_runs")
    if resolved_proxy_run in crop_parent:
        if not overwrite:
            raise ValueError(f"crop_runs/{resolved_proxy_run} already exists. Pass --overwrite to replace it.")
        del crop_parent[resolved_proxy_run]
    crop_group = crop_parent.create_group(resolved_proxy_run)

    _write_array(crop_group, "frame_indices", frame_indices)
    _write_array(crop_group, "source_frame_indices", frame_indices)
    _write_array(crop_group, "source_clip_indices", clip_indices)
    _write_array(crop_group, "source_clip_local_frame_indices", clip_local_frame_indices)
    _write_array(crop_group, "source_refined_row_ids", refined_row_ids)
    _write_array(crop_group, "source_detect_row_index", source_detect_row_index)
    _write_array(crop_group, "detection_indices", roi_row_index)
    _write_array(crop_group, "source_crop_row_ids", direct_source_crop_row_ids(int(table.num_rows)))
    _write_array(crop_group, "roi_coordinates_full", roi_coordinates_full)

    source = manifest.get("source") if isinstance(manifest.get("source"), Mapping) else {}
    source_archive = str(source.get("archive_path") or "")
    if source_archive and Path(source_archive).expanduser().resolve() != archive_path:
        raise ValueError(f"Manifest archive_path {source_archive!r} does not match zarr_path {str(archive_path)!r}.")

    crop_group.attrs.update(
        {
            "schema": PROXY_CROP_RUN_SCHEMA,
            "crop_proxy_schema": PROXY_CROP_RUN_SCHEMA,
            "proxy_crop_complete": True,
            "stage_selector_eligible": False,
            RUN_COMPLETION_CONTRACT_ATTR: RUN_COMPLETION_CONTRACT,
            RUN_COMPLETION_STATUS_ATTR: "auxiliary",
            "status": "completed",
            "run_name": resolved_proxy_run,
            "stage": "crop_proxy",
            "crop_storage_mode": "geometry_only",
            "source_kind": "finalized_clipped_refined_detect_collection_proxy",
            "source_collection_id": source.get("collection_id"),
            "source_collection_path": source.get("collection_path"),
            "source_clip_id": clip_id,
            "source_clip_index": int(clip_indices[0]),
            "source_video_path": video_path,
            "source_roi_cache_manifest": str(source_manifest_path),
            "source_roi_cache_alias_manifest": str(alias_manifest),
            "source_roi_cache_row_index_path": str(row_index_path),
            "source_roi_cache_required": True,
            "crop_policy": "centered_refined_bbox",
            "roi_size": [int(shape[1]), int(shape[2])],
            "roi_shape": [int(shape[1]), int(shape[2])],
            "height": int(root.attrs["height"]) if root.attrs.get("height") is not None else None,
            "width": int(root.attrs["width"]) if root.attrs.get("width") is not None else None,
            "created_at_utc": _utc_now(),
            "row_count": int(table.num_rows),
        }
    )

    alias_manifest_payload = _write_alias_manifest(
        source_manifest_path=source_manifest_path,
        source_manifest=manifest,
        proxy_run_name=resolved_proxy_run,
        alias_manifest_path=alias_manifest,
        row_index_path=row_index_path,
    )
    cache = open_flat_roi_cache(
        alias_manifest,
        expected_archive_path=archive_path,
        expected_crop_run=resolved_proxy_run,
        expected_shape=shape,
    )
    try:
        cache_shape = tuple(int(value) for value in cache.shape)
    finally:
        cache.close()

    return {
        "ok": True,
        "zarr_path": str(archive_path),
        "proxy_crop_run": resolved_proxy_run,
        "proxy_crop_run_path": f"crop_runs/{resolved_proxy_run}",
        "alias_manifest_path": str(alias_manifest),
        "source_manifest_path": str(source_manifest_path),
        "row_index_path": str(row_index_path),
        "row_count": int(table.num_rows),
        "cache_shape": list(cache_shape),
        "source_clip_id": clip_id,
        "source_clip_index": int(clip_indices[0]),
        "source_video_path": video_path,
        "source_collection_id": source.get("collection_id"),
        "source_crop_run_name_in_alias": alias_manifest_payload["source"]["crop_run_name"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a geometry-only proxy crop run for a clipped collection flat ROI cache."
    )
    parser.add_argument("zarr_path", type=Path, help="Analysis Zarr path.")
    parser.add_argument("manifest_path", type=Path, help="Clipped collection flat ROI cache manifest.")
    parser.add_argument("--proxy-run", help="Output crop_runs/<name>. Defaults to a collection/clip/cache-derived name.")
    parser.add_argument("--alias-manifest", type=Path, help="Output manifest alias path.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing proxy crop run.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args(argv)

    result = create_clipped_collection_proxy_crop_run(
        zarr_path=args.zarr_path,
        manifest_path=args.manifest_path,
        proxy_run_name=args.proxy_run,
        alias_manifest_path=args.alias_manifest,
        overwrite=bool(args.overwrite),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            "created proxy crop run "
            f"{result['proxy_crop_run_path']} rows={result['row_count']} "
            f"alias={result['alias_manifest_path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
