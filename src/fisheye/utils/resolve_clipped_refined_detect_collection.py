"""Resolve finalized clipped refined-detect collections to frame/run mappings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import zarr


RESOLVER_SCHEMA = "palette.clipped_refined_detect_collection_resolver.v1"


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON {path}: {exc}") from exc


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _open_root(zarr_path: Path) -> zarr.Group:
    try:
        return zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(zarr_path), mode="r")


def _resolve_collection(root: zarr.Group, collection_id: Optional[str]) -> tuple[str, zarr.Group]:
    resolved_id = str(collection_id or "").strip()
    if not resolved_id:
        refined_parent = root.get("refined_detect_runs")
        if refined_parent is not None:
            resolved_id = str(refined_parent.attrs.get("latest_collection") or "").strip()
    if not resolved_id:
        raise ValueError("No collection id provided and refined_detect_runs.latest_collection is not set")
    path = f"experiment_index/finalized_runs/{resolved_id}"
    try:
        return resolved_id, root[path]
    except Exception as exc:
        raise ValueError(f"Finalized collection not found: {path}") from exc


def _resolve_recording_frame_index(
    collection_attrs: Mapping[str, Any],
    *,
    explicit_path: str | Path | None,
) -> Path:
    if explicit_path is not None:
        return Path(explicit_path).expanduser().resolve()
    plan_path_value = collection_attrs.get("plan_path")
    if not plan_path_value:
        raise ValueError("Collection has no plan_path; provide --recording-frame-index")
    plan_path = Path(str(plan_path_value)).expanduser().resolve()
    plan = _read_json(plan_path)
    if not isinstance(plan, Mapping) or not plan.get("recording_dir"):
        raise ValueError("Collection plan has no recording_dir; provide --recording-frame-index")
    recording_dir = Path(str(plan["recording_dir"])).expanduser().resolve()
    manifest_path = recording_dir / "recording_frame_index_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"recording_frame_index_manifest.json not found: {manifest_path}")
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"recording_frame_index_manifest.json is not an object: {manifest_path}")
    raw_path = manifest.get("recording_frame_index_path") or "recording_frame_index.parquet"
    frame_index_path = Path(str(raw_path)).expanduser()
    if not frame_index_path.is_absolute():
        frame_index_path = recording_dir / frame_index_path
    return frame_index_path.resolve()


def _read_frame_index(path: Path) -> pa.Table:
    if not path.exists():
        raise FileNotFoundError(f"recording_frame_index.parquet not found: {path}")
    parquet_file = pq.ParquetFile(path)
    names = set(parquet_file.schema_arrow.names)
    required = {"camera_serial", "clip_id", "clip_local_frame_index", "recording_frame_id"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"recording_frame_index.parquet missing required columns: {missing}")
    optional = ["parent_frame_index", "timestamp", "timestamp_sys"]
    columns = ["camera_serial", "clip_id", "clip_local_frame_index", "recording_frame_id"]
    columns.extend(name for name in optional if name in names)
    return pq.read_table(path, columns=columns).combine_chunks()


def build_collection_frame_map(
    zarr_path: str | Path,
    *,
    collection_id: Optional[str] = None,
    recording_frame_index: str | Path | None = None,
) -> tuple[dict[str, Any], pa.Table]:
    archive_path = Path(zarr_path).expanduser().resolve()
    root = _open_root(archive_path)
    resolved_id, collection = _resolve_collection(root, collection_id)
    attrs = dict(collection.attrs)
    selected_runs = [dict(row) for row in attrs.get("selected_runs", [])]
    if not selected_runs:
        raise ValueError(f"Collection has no selected_runs: {resolved_id}")

    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in selected_runs:
        key = (str(row.get("camera_serial") or ""), str(row.get("clip_id") or ""))
        if not all(key):
            raise ValueError(f"Selected run is missing camera_serial or clip_id: {row}")
        if key in lookup:
            raise ValueError(f"Duplicate selected run for camera/clip pair: {key}")
        lookup[key] = row

    frame_index_path = _resolve_recording_frame_index(
        attrs,
        explicit_path=recording_frame_index,
    )
    frame_index = _read_frame_index(frame_index_path)
    camera_values = pc.cast(frame_index["camera_serial"], "string").to_pylist()
    clip_values = pc.cast(frame_index["clip_id"], "string").to_pylist()
    columns = {name: frame_index[name].to_pylist() for name in frame_index.column_names}

    rows: list[dict[str, Any]] = []
    missing_pairs: set[tuple[str, str]] = set()
    for row_index, (camera_serial, clip_id) in enumerate(zip(camera_values, clip_values)):
        selected = lookup.get((str(camera_serial), str(clip_id)))
        if selected is None:
            missing_pairs.add((str(camera_serial), str(clip_id)))
            continue
        row = {
            "camera_serial": str(camera_serial),
            "clip_id": str(clip_id),
            "recording_frame_id": columns["recording_frame_id"][row_index],
            "clip_local_frame_index": columns["clip_local_frame_index"][row_index],
            "work_unit_id": selected.get("work_unit_id"),
            "detect_run": selected.get("detect_run"),
            "detect_quality_run": selected.get("detect_quality_run"),
            "refined_detect_run": selected.get("refined_detect_run"),
            "detect_group_path": selected.get("detect_group_path"),
            "refined_group_path": selected.get("refined_group_path"),
        }
        for optional in ("parent_frame_index", "timestamp", "timestamp_sys"):
            if optional in columns:
                row[optional] = columns[optional][row_index]
        rows.append(row)

    table = pa.Table.from_pylist(rows)
    summary = {
        "status": "ok",
        "schema_version": RESOLVER_SCHEMA,
        "analysis_zarr": str(archive_path),
        "collection_id": resolved_id,
        "collection_path": f"experiment_index/finalized_runs/{resolved_id}",
        "recording_frame_index": str(frame_index_path),
        "selected_run_count": len(selected_runs),
        "mapped_frame_count": int(table.num_rows),
        "unselected_frame_pair_count": len(missing_pairs),
        "unselected_frame_pairs": [
            {"camera_serial": camera, "clip_id": clip}
            for camera, clip in sorted(missing_pairs)[:20]
        ],
        "selected_runs": selected_runs,
    }
    return summary, table


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve a finalized clipped refined-detect collection to frame/run mappings."
    )
    parser.add_argument("zarr_path", type=Path, help="Analysis Zarr path")
    parser.add_argument("--collection-id", default=None, help="Collection id; default uses refined_detect_runs.latest_collection")
    parser.add_argument("--recording-frame-index", type=Path, default=None, help="Override recording_frame_index.parquet path")
    parser.add_argument("--output-parquet", type=Path, default=None, help="Optional frame/run mapping parquet output")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional summary JSON output")
    parser.add_argument("--include-rows", action="store_true", help="Include frame rows in printed/output JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary, table = build_collection_frame_map(
        args.zarr_path,
        collection_id=args.collection_id,
        recording_frame_index=args.recording_frame_index,
    )
    if args.output_parquet is not None:
        args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, args.output_parquet)
        summary["output_parquet"] = str(args.output_parquet.expanduser().resolve())
    payload = dict(summary)
    if args.include_rows:
        payload["rows"] = table.to_pylist()
    if args.output_json is not None:
        _write_json(args.output_json.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
