"""Repair clipped-collection proxy crop-run contract metadata.

Dry-run by default. The repair targets geometry-only proxy crop runs created
for clipped finalized detect collections before they carried canonical
``bbox_norm_coords`` and a stable synthetic ``source_detect_run`` label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq
import zarr

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.run_provenance import json_ready
from fisheye.shared.zarr.chunk_profiles import create_geometry_preload_array
from fisheye.utils.create_clipped_collection_proxy_crop_run import (
    CLIPPED_COLLECTION_PROXY_DETECTION_SOURCE_TYPE,
    PROXY_CROP_RUN_SCHEMA,
    bbox_norm_from_clipped_collection_row_index,
    clipped_collection_proxy_source_detect_label,
)
from fisheye.utils.merge_clipped_proxy_crop_runs import MERGED_PROXY_CROP_RUN_SCHEMA


SCHEMA_VERSION = "palette.clipped_proxy_crop_contract_repair.v1"
BBOX_NORM_SEMANTICS = "bbox_xywh_normalized_to_full_frame"
SOURCE_DETECT_SEMANTICS = "synthetic_collection_rowset_label_not_detect_runs_child"
_UNKNOWN_TEXT = {"", "none", "null", "unknown"}


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _clean_text(value: Any) -> str | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _UNKNOWN_TEXT:
        return None
    return text


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


def _resolve_existing_path(value: Any, *, relative_to: Path) -> Path:
    text = _clean_text(value)
    if not text:
        raise ValueError("Expected a non-empty path.")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _unique_nonempty(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = _clean_text(value)
        if text and text not in out:
            out.append(text)
    return out


def _row_count(group: zarr.Group) -> int:
    if "frame_indices" not in group:
        raise ValueError("missing required array frame_indices")
    return int(group["frame_indices"].shape[0])


def _is_clipped_proxy_crop_run(group: zarr.Group) -> bool:
    schema = _clean_text(group.attrs.get("crop_proxy_schema") or group.attrs.get("schema"))
    source_kind = _clean_text(group.attrs.get("source_kind"))
    source_type = _clean_text(group.attrs.get("detection_source_type") or group.attrs.get("source_type"))
    if schema in {PROXY_CROP_RUN_SCHEMA, MERGED_PROXY_CROP_RUN_SCHEMA}:
        return True
    if source_kind in {
        "finalized_clipped_refined_detect_collection_proxy",
        "merged_clipped_collection_proxy_crop_run",
    }:
        return True
    if source_type and source_type.startswith("finalized_clipped_refined_detect_collection"):
        return True
    if group.attrs.get("source_proxy_crop_runs") is not None:
        return True
    if group.attrs.get("source_roi_cache_row_index_path") is not None and str(source_kind or "").startswith("finalized_clipped"):
        return True
    return False


def _is_merged_proxy_crop_run(group: zarr.Group) -> bool:
    schema = _clean_text(group.attrs.get("crop_proxy_schema") or group.attrs.get("schema"))
    source_kind = _clean_text(group.attrs.get("source_kind"))
    return (
        schema == MERGED_PROXY_CROP_RUN_SCHEMA
        or source_kind == "merged_clipped_collection_proxy_crop_run"
        or group.attrs.get("source_proxy_crop_runs") is not None
    )


def _read_row_index(group: zarr.Group, *, zarr_path: Path) -> tuple[Path, Any]:
    row_index_path = _resolve_existing_path(
        group.attrs.get("source_roi_cache_row_index_path"),
        relative_to=zarr_path.parent,
    )
    if not row_index_path.exists():
        raise FileNotFoundError(f"source_roi_cache_row_index_path does not exist: {row_index_path}")
    return row_index_path, pq.read_table(row_index_path)


def _bbox_from_per_clip_proxy(group: zarr.Group, *, crop_run: str, zarr_path: Path) -> tuple[np.ndarray, str]:
    rows = _row_count(group)
    if "bbox_norm_coords" in group:
        bbox = np.asarray(group["bbox_norm_coords"][:], dtype=np.float32)
        if bbox.ndim != 2 or int(bbox.shape[1]) != 4:
            raise ValueError(f"crop_runs/{crop_run}/bbox_norm_coords must have shape (N, 4); got {tuple(bbox.shape)}")
        if int(bbox.shape[0]) != rows:
            raise ValueError(f"crop_runs/{crop_run}/bbox_norm_coords has {int(bbox.shape[0])} rows; expected {rows}")
        return bbox, "existing_bbox_norm_coords"

    row_index_path, table = _read_row_index(group, zarr_path=zarr_path)
    if int(table.num_rows) != rows:
        raise ValueError(f"{row_index_path} has {int(table.num_rows)} rows; expected {rows}")
    return bbox_norm_from_clipped_collection_row_index(table), str(row_index_path)


def _source_crop_runs(group: zarr.Group) -> list[str]:
    value = group.attrs.get("source_proxy_crop_runs")
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    text = _clean_text(value)
    return [text] if text else []


def _bbox_from_merged_proxy(
    crop_parent: zarr.Group,
    group: zarr.Group,
    *,
    crop_run: str,
    zarr_path: Path,
) -> tuple[np.ndarray, str, int]:
    source_runs = _source_crop_runs(group)
    if not source_runs:
        return _bbox_from_per_clip_proxy(group, crop_run=crop_run, zarr_path=zarr_path)[0], "merged_row_index", 1

    bbox_parts: list[np.ndarray] = []
    repaired_count = 0
    for source_run in source_runs:
        if source_run not in crop_parent:
            raise ValueError(f"crop_runs/{crop_run} references missing source proxy crop run {source_run!r}")
        source_group = crop_parent[source_run]
        bbox, source = _bbox_from_per_clip_proxy(source_group, crop_run=source_run, zarr_path=zarr_path)
        bbox_parts.append(bbox)
        if source != "existing_bbox_norm_coords":
            repaired_count += 1
    bbox = np.concatenate(bbox_parts, axis=0) if bbox_parts else np.empty((0, 4), dtype=np.float32)
    rows = _row_count(group)
    if int(bbox.shape[0]) != rows:
        raise ValueError(f"merged bbox rows from source proxies has {int(bbox.shape[0])} rows; expected {rows}")
    return bbox.astype(np.float32, copy=False), "source_proxy_crop_runs", repaired_count


def _row_index_unique(group: zarr.Group, column: str, *, zarr_path: Path) -> list[str]:
    try:
        _path, table = _read_row_index(group, zarr_path=zarr_path)
    except Exception:
        return []
    if column not in table.column_names:
        return []
    return _unique_nonempty(table[column].to_pylist())


def _aggregate_source_attrs(
    crop_parent: zarr.Group,
    group: zarr.Group,
    attr_name: str,
    *,
    row_index_column: str,
    zarr_path: Path,
) -> list[str]:
    existing = group.attrs.get(attr_name)
    if isinstance(existing, list) and existing:
        return _unique_nonempty(existing)

    out: list[str] = []
    for source_run in _source_crop_runs(group):
        if source_run not in crop_parent:
            continue
        source_group = crop_parent[source_run]
        values = source_group.attrs.get(attr_name)
        if isinstance(values, list):
            out.extend(values)
        elif values is not None:
            out.append(values)
        else:
            out.extend(_row_index_unique(source_group, row_index_column, zarr_path=zarr_path))
    if not out and not _source_crop_runs(group):
        out.extend(_row_index_unique(group, row_index_column, zarr_path=zarr_path))
    return _unique_nonempty(out)


def _expected_attrs(
    crop_parent: zarr.Group,
    group: zarr.Group,
    *,
    zarr_path: Path,
) -> dict[str, Any]:
    collection_id = group.attrs.get("source_collection_id")
    source_detect_run = _clean_text(group.attrs.get("source_detect_run")) or clipped_collection_proxy_source_detect_label(
        collection_id
    )
    attrs: dict[str, Any] = {
        "detection_source_type": CLIPPED_COLLECTION_PROXY_DETECTION_SOURCE_TYPE,
        "source_detect_run": source_detect_run,
        "source_detect_run_semantics": SOURCE_DETECT_SEMANTICS,
        "bbox_norm_coords_semantics": BBOX_NORM_SEMANTICS,
    }
    if _is_merged_proxy_crop_run(group):
        attrs["bbox_norm_coords_source"] = (
            "source_proxy_crop_runs.bbox_norm_coords_or_repaired_from_source_roi_cache_row_index_path"
        )
        attrs["source_refined_runs"] = _aggregate_source_attrs(
            crop_parent,
            group,
            "source_refined_runs",
            row_index_column="refined_detect_run",
            zarr_path=zarr_path,
        )
        attrs["source_refined_run_paths"] = _aggregate_source_attrs(
            crop_parent,
            group,
            "source_refined_run_paths",
            row_index_column="refined_group_path",
            zarr_path=zarr_path,
        )
    else:
        attrs["bbox_norm_coords_source"] = "clipped_collection_row_index.bbox_norm_cxcywh"
        source_refined_runs = _row_index_unique(group, "refined_detect_run", zarr_path=zarr_path)
        source_refined_paths = _row_index_unique(group, "refined_group_path", zarr_path=zarr_path)
        if source_refined_runs:
            attrs["source_refined_runs"] = source_refined_runs
        if source_refined_paths:
            attrs["source_refined_run_paths"] = source_refined_paths
    return attrs


def _attrs_need_update(group: zarr.Group, expected: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in expected.items() if group.attrs.get(key) != value}


def _bbox_needs_update(group: zarr.Group, bbox: np.ndarray) -> bool:
    if "bbox_norm_coords" not in group:
        return True
    existing = np.asarray(group["bbox_norm_coords"][:], dtype=np.float32)
    if existing.shape != bbox.shape:
        return True
    if not np.array_equal(np.isnan(existing), np.isnan(bbox)):
        return True
    finite = np.isfinite(existing) & np.isfinite(bbox)
    if not np.any(finite):
        return False
    return not bool(np.allclose(existing[finite], bbox[finite], rtol=0.0, atol=1e-6))


def _repair_crop_group(
    crop_parent: zarr.Group,
    group: zarr.Group,
    *,
    crop_run: str,
    zarr_path: Path,
    apply: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "crop_run": crop_run,
        "apply": bool(apply),
        "status": "skipped",
        "reason": "not_clipped_proxy_crop_run",
        "rows": 0,
        "errors": [],
    }
    if not _is_clipped_proxy_crop_run(group):
        return report

    try:
        rows = _row_count(group)
        if _is_merged_proxy_crop_run(group):
            bbox, bbox_source, repaired_count = _bbox_from_merged_proxy(
                crop_parent,
                group,
                crop_run=crop_run,
                zarr_path=zarr_path,
            )
        else:
            bbox, bbox_source = _bbox_from_per_clip_proxy(group, crop_run=crop_run, zarr_path=zarr_path)
            repaired_count = int(bbox_source != "existing_bbox_norm_coords")
        expected_attrs = _expected_attrs(crop_parent, group, zarr_path=zarr_path)
        if _is_merged_proxy_crop_run(group):
            expected_attrs["legacy_bbox_norm_coords_repair_count"] = repaired_count
        attr_updates = _attrs_need_update(group, expected_attrs)
        bbox_update = _bbox_needs_update(group, bbox)
    except Exception as exc:
        report.update({"status": "blocked", "reason": "repair_inputs_unavailable", "errors": [str(exc)]})
        return report

    report.update(
        {
            "reason": "clipped_proxy_crop_run",
            "rows": rows,
            "bbox_norm_coords_status": "would_write" if bbox_update and not apply else ("updated" if bbox_update else "ok"),
            "attrs_status": "would_update" if attr_updates and not apply else ("updated" if attr_updates else "ok"),
            "attr_updates": attr_updates,
            "bbox_norm_coords_source": bbox_source,
            "legacy_bbox_norm_coords_repair_count": repaired_count,
            "status": "would_update" if (bbox_update or attr_updates) and not apply else ("updated" if (bbox_update or attr_updates) else "ok"),
        }
    )

    if apply and (bbox_update or attr_updates):
        if bbox_update:
            create_geometry_preload_array(group, "bbox_norm_coords", data=bbox.astype(np.float32), overwrite=True)
        if attr_updates:
            group.attrs.update(json_ready(attr_updates))
    return report


def repair_clipped_proxy_crop_contract(
    zarr_path: str | Path,
    *,
    crop_runs: Sequence[str] | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    archive_path = Path(zarr_path).expanduser().resolve()
    root = zarr.open_group(str(archive_path), mode="a" if apply else "r", use_consolidated=False)
    crop_parent = root.get("crop_runs")
    if crop_parent is None:
        return {
            "status": "ok",
            "schema_version": SCHEMA_VERSION,
            "zarr_path": str(archive_path),
            "apply": bool(apply),
            "crop_run_count": 0,
            "affected_crop_run_count": 0,
            "changed_crop_run_count": 0,
            "blocked_crop_run_count": 0,
            "crop_runs": [],
        }

    selected = [str(name) for name in crop_runs] if crop_runs else _child_group_names(crop_parent)
    reports: list[dict[str, Any]] = []
    for crop_run in selected:
        if crop_run not in crop_parent:
            reports.append(
                {
                    "crop_run": crop_run,
                    "apply": bool(apply),
                    "status": "blocked",
                    "reason": "crop_run_not_found",
                    "rows": 0,
                    "errors": [f"crop_runs/{crop_run} not found"],
                }
            )
            continue
        reports.append(
            _repair_crop_group(
                crop_parent,
                crop_parent[crop_run],
                crop_run=crop_run,
                zarr_path=archive_path,
                apply=apply,
            )
        )

    blocked = sum(1 for item in reports if item.get("status") == "blocked")
    changed = sum(1 for item in reports if item.get("status") in {"would_update", "updated"})
    affected = sum(1 for item in reports if item.get("reason") == "clipped_proxy_crop_run")
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", nargs="+", type=Path, help="Analysis Zarr path(s) to inspect or repair.")
    parser.add_argument("--crop-run", action="append", default=[], help="Specific crop_runs/<run> to repair. Repeatable.")
    parser.add_argument("--apply", action="store_true", help="Apply repairs. Default is dry-run.")
    parser.add_argument("--output-json", type=Path, help="Write aggregate JSON report.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    reports = [
        repair_clipped_proxy_crop_contract(
            path,
            crop_runs=args.crop_run or None,
            apply=bool(args.apply),
        )
        for path in args.zarr_path
    ]
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
        write_json_atomic(args.output_json.expanduser().resolve(), json_ready(result))
    print(json.dumps(json_ready(result), indent=2, sort_keys=True, default=_json_default))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
