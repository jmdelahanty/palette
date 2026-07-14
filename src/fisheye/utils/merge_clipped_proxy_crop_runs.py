#!/usr/bin/env python3
"""Merge per-clip proxy crop runs into one collection-level proxy crop run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq
import zarr

from fisheye.shared.row_lineage import direct_source_crop_row_ids
from fisheye.shared.run_provenance import json_ready
from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr.chunk_profiles import create_geometry_preload_array
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    require_runs_parent,
)
from fisheye.utils.create_clipped_collection_proxy_crop_run import (
    CLIPPED_COLLECTION_PROXY_DETECTION_SOURCE_TYPE,
    PROXY_CROP_RUN_SCHEMA,
    bbox_norm_from_clipped_collection_row_index,
    clipped_collection_proxy_source_detect_label,
)

MERGED_PROXY_CROP_RUN_SCHEMA = "palette_clipped_collection_merged_proxy_crop_run_v1"

REQUIRED_PROXY_ARRAYS: tuple[str, ...] = (
    "frame_indices",
    "source_frame_indices",
    "source_clip_indices",
    "source_clip_local_frame_indices",
    "source_refined_row_ids",
    "source_detect_row_index",
    "detection_indices",
    "source_crop_row_ids",
    "roi_coordinates_full",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_array(group: zarr.Group, name: str) -> np.ndarray:
    return np.asarray(group[name][:])


def _resolve_existing_path(value: Any, *, relative_to: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Expected non-empty path.")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _load_run_names(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict):
        for key in ("source_crop_runs", "proxy_crop_runs", "crop_runs", "runs"):
            value = payload.get(key)
            if isinstance(value, list):
                return [str(item) for item in value]
    raise ValueError(f"Could not read crop-run list from {path}; expected list or mapping.")


def _unique_attr_values(groups: Sequence[tuple[str, zarr.Group]], name: str) -> list[Any]:
    values: list[Any] = []
    for _run_name, group in groups:
        value = group.attrs.get(name)
        if value is None:
            continue
        if isinstance(value, list):
            candidates = value
        else:
            candidates = [value]
        for candidate in candidates:
            if candidate is None or candidate in values:
                continue
            values.append(candidate)
    return values


def _single_common_attr(groups: Sequence[tuple[str, zarr.Group]], name: str) -> Any:
    values = _unique_attr_values(groups, name)
    if len(values) > 1:
        raise ValueError(f"Source proxy crop runs disagree on {name}: {values!r}")
    return values[0] if values else None


def _bbox_norm_coords_for_source(
    run_name: str,
    group: zarr.Group,
    *,
    total_rows: int,
    zarr_path: Path,
) -> tuple[np.ndarray, bool]:
    if "bbox_norm_coords" in group:
        bbox = _as_array(group, "bbox_norm_coords")
        if bbox.ndim != 2 or int(bbox.shape[1]) != 4:
            raise ValueError(
                f"crop_runs/{run_name}/bbox_norm_coords must have shape (N, 4); got {tuple(bbox.shape)}."
            )
        if int(bbox.shape[0]) != int(total_rows):
            raise ValueError(
                f"crop_runs/{run_name}/bbox_norm_coords has {int(bbox.shape[0])} rows; expected {total_rows}."
            )
        return bbox.astype(np.float32, copy=False), False

    row_index_attr = group.attrs.get("source_roi_cache_row_index_path")
    if row_index_attr is None:
        raise ValueError(
            f"crop_runs/{run_name} missing bbox_norm_coords and source_roi_cache_row_index_path; "
            "cannot repair legacy proxy crop geometry."
        )
    row_index_path = _resolve_existing_path(row_index_attr, relative_to=zarr_path.parent)
    if not row_index_path.exists():
        raise FileNotFoundError(
            f"crop_runs/{run_name} references missing source_roi_cache_row_index_path: {row_index_path}"
        )
    table = pq.read_table(row_index_path)
    if int(table.num_rows) != int(total_rows):
        raise ValueError(
            f"crop_runs/{run_name} row-index parquet has {int(table.num_rows)} rows; expected {total_rows}."
        )
    return bbox_norm_from_clipped_collection_row_index(table), True


def _source_detect_run_for_merge(sources: Sequence[tuple[str, zarr.Group]]) -> str:
    source_detect_run = _single_common_attr(sources, "source_detect_run")
    if source_detect_run:
        return str(source_detect_run)
    collection_id = _single_common_attr(sources, "source_collection_id")
    return clipped_collection_proxy_source_detect_label(collection_id)


def _resolve_source_runs(root: zarr.Group, names: Sequence[str]) -> list[tuple[str, zarr.Group]]:
    if not names:
        raise ValueError("At least one --source-crop-run is required.")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate source crop run names were supplied.")
    parent = root.get("crop_runs")
    if parent is None:
        raise ValueError("Missing crop_runs group.")
    runs: list[tuple[str, zarr.Group]] = []
    for name in names:
        if name not in parent:
            raise ValueError(f"crop_runs/{name} not found.")
        group = parent[name]
        missing = [array_name for array_name in REQUIRED_PROXY_ARRAYS if array_name not in group]
        if missing:
            raise ValueError(f"crop_runs/{name} missing required arrays: {missing}")
        total_rows = int(group["frame_indices"].shape[0])
        for array_name in REQUIRED_PROXY_ARRAYS:
            rows = int(group[array_name].shape[0])
            if rows != total_rows:
                raise ValueError(f"crop_runs/{name}/{array_name} has {rows} rows; expected {total_rows}.")
        runs.append((str(name), group))

    def sort_key(item: tuple[str, zarr.Group]) -> tuple[int, str]:
        value = item[1].attrs.get("source_clip_index")
        try:
            return int(value), item[0]
        except (TypeError, ValueError):
            return 10**12, item[0]

    return sorted(runs, key=sort_key)


def merge_clipped_proxy_crop_runs(
    *,
    zarr_path: str | Path,
    source_crop_runs: Sequence[str],
    output_run: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Merge geometry-only per-clip proxy crop runs into one proxy crop run."""

    archive = Path(zarr_path).expanduser().resolve()
    root = open_zarr_root(archive, mode="a")
    sources = _resolve_source_runs(root, [str(name) for name in source_crop_runs])
    if "/" in output_run or not str(output_run).strip():
        raise ValueError("output_run must be a single non-empty Zarr group name.")

    parent = require_runs_parent(root, "crop_runs")
    if output_run in parent:
        if not overwrite:
            raise ValueError(f"crop_runs/{output_run} already exists. Pass --overwrite to replace it.")
        del parent[output_run]
    target = parent.create_group(output_run)

    row_counts = [int(group["frame_indices"].shape[0]) for _name, group in sources]
    total_rows = int(sum(row_counts))
    concatenated: dict[str, np.ndarray] = {
        name: np.concatenate([_as_array(group, name) for _run_name, group in sources], axis=0)
        for name in REQUIRED_PROXY_ARRAYS
    }
    bbox_payloads = [
        _bbox_norm_coords_for_source(
            run_name,
            group,
            total_rows=int(group["frame_indices"].shape[0]),
            zarr_path=archive,
        )
        for run_name, group in sources
    ]
    concatenated["bbox_norm_coords"] = np.concatenate([payload[0] for payload in bbox_payloads], axis=0)
    repaired_bbox_source_count = int(sum(1 for _bbox, repaired in bbox_payloads if repaired))
    concatenated["source_crop_row_ids"] = direct_source_crop_row_ids(total_rows)
    concatenated["detection_indices"] = direct_source_crop_row_ids(total_rows)

    source_proxy_crop_run_index = np.concatenate(
        [np.full(count, idx, dtype=np.int32) for idx, count in enumerate(row_counts)],
        axis=0,
    )
    source_proxy_crop_row_ids = np.concatenate(
        [direct_source_crop_row_ids(count) for count in row_counts],
        axis=0,
    )

    for name, data in concatenated.items():
        create_geometry_preload_array(target, name, data=data, overwrite=True)
    create_geometry_preload_array(target, "source_proxy_crop_run_index", data=source_proxy_crop_run_index, overwrite=True)
    create_geometry_preload_array(target, "source_proxy_crop_row_ids", data=source_proxy_crop_row_ids, overwrite=True)

    first_attrs = sources[0][1].attrs
    source_detect_run = _source_detect_run_for_merge(sources)
    detection_source_type = (
        _single_common_attr(sources, "detection_source_type")
        or CLIPPED_COLLECTION_PROXY_DETECTION_SOURCE_TYPE
    )
    source_run_names = [name for name, _group in sources]
    source_clip_ids = [normalize_attr(group.attrs.get("source_clip_id")) for _name, group in sources]
    source_clip_indices = [group.attrs.get("source_clip_index") for _name, group in sources]
    source_manifests = [normalize_attr(group.attrs.get("source_roi_cache_manifest")) for _name, group in sources]
    source_alias_manifests = [normalize_attr(group.attrs.get("source_roi_cache_alias_manifest")) for _name, group in sources]
    source_video_width = (
        _single_common_attr(sources, "source_video_width")
        or _single_common_attr(sources, "width")
        or root.attrs.get("source_video_width")
        or root.attrs.get("width")
    )
    source_video_height = (
        _single_common_attr(sources, "source_video_height")
        or _single_common_attr(sources, "height")
        or root.attrs.get("source_video_height")
        or root.attrs.get("height")
    )
    if source_video_width is None or source_video_height is None:
        raise ValueError("Source proxy crop runs do not provide source-video dimensions.")
    source_video_dimension_sources = _unique_attr_values(
        sources,
        "source_video_dimensions_source",
    )
    created_at = _utc_now()
    target.attrs.update(
        json_ready(
            {
                "schema": MERGED_PROXY_CROP_RUN_SCHEMA,
                "crop_proxy_schema": MERGED_PROXY_CROP_RUN_SCHEMA,
                "proxy_crop_complete": True,
                "stage_selector_eligible": False,
                RUN_COMPLETION_CONTRACT_ATTR: RUN_COMPLETION_CONTRACT,
                RUN_COMPLETION_STATUS_ATTR: "auxiliary",
                "status": "completed",
                "run_name": output_run,
                "stage": "crop_proxy",
                "crop_storage_mode": "geometry_only",
                "source_kind": "merged_clipped_collection_proxy_crop_run",
                "detection_source_type": detection_source_type,
                "source_detect_run": source_detect_run,
                "source_detect_run_semantics": "synthetic_collection_rowset_label_not_detect_runs_child",
                "source_refined_runs": _unique_attr_values(sources, "source_refined_runs"),
                "source_refined_run_paths": _unique_attr_values(sources, "source_refined_run_paths"),
                "source_collection_id": first_attrs.get("source_collection_id"),
                "source_collection_path": first_attrs.get("source_collection_path"),
                "source_proxy_crop_runs": source_run_names,
                "source_proxy_crop_run_count": len(source_run_names),
                "source_clip_ids": source_clip_ids,
                "source_clip_indices": source_clip_indices,
                "source_roi_cache_manifests": source_manifests,
                "source_roi_cache_alias_manifests": source_alias_manifests,
                "source_roi_cache_required": True,
                "crop_policy": first_attrs.get("crop_policy") or "centered_refined_bbox",
                "bbox_norm_coords_semantics": first_attrs.get("bbox_norm_coords_semantics")
                or "bbox_xywh_normalized_to_full_frame",
                "bbox_norm_coords_source": (
                    "source_proxy_crop_runs.bbox_norm_coords_or_repaired_from_source_roi_cache_row_index_path"
                ),
                "legacy_bbox_norm_coords_repair_count": repaired_bbox_source_count,
                "roi_size": first_attrs.get("roi_size"),
                "roi_shape": first_attrs.get("roi_shape"),
                "source_video_width": int(source_video_width),
                "source_video_height": int(source_video_height),
                "source_video_dimensions_source": source_video_dimension_sources,
                "height": int(source_video_height),
                "width": int(source_video_width),
                "created_at_utc": created_at,
                "row_count": total_rows,
            }
        )
    )
    return {
        "ok": True,
        "zarr_path": str(archive),
        "merged_proxy_crop_run": output_run,
        "merged_proxy_crop_run_path": f"crop_runs/{output_run}",
        "source_proxy_crop_runs": source_run_names,
        "source_proxy_crop_run_count": len(source_run_names),
        "row_count": total_rows,
        "source_row_counts": row_counts,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Analysis Zarr path.")
    parser.add_argument("--source-crop-run", action="append", default=[], help="Per-clip proxy crop run to merge. Repeatable.")
    parser.add_argument("--source-crop-runs-file", type=Path, help="JSON list or mapping containing source crop runs.")
    parser.add_argument("--output-run", required=True, help="Output crop_runs/<name>.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output run.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    source_runs = list(args.source_crop_run or [])
    if args.source_crop_runs_file is not None:
        source_runs.extend(_load_run_names(args.source_crop_runs_file.expanduser()))
    result = merge_clipped_proxy_crop_runs(
        zarr_path=args.zarr_path,
        source_crop_runs=source_runs,
        output_run=args.output_run,
        overwrite=bool(args.overwrite),
    )
    if args.json:
        print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    else:
        print(f"created {result['merged_proxy_crop_run_path']} rows={result['row_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
