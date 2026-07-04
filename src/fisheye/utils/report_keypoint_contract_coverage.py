"""Report keypoint ROI pixel-contract coverage from the registry."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Optional, Sequence

from fisheye.registry.db import RegistryPaths
from fisheye.shared.json_safety import write_jsonl_atomic

MISSING = "<missing>"

DEFAULT_CANDIDATE_COMPATIBLE_CONTRACTS = (
    "orange_mono_pynvvc_luma_uint8_v1",
    "nv12_luma_plane_uint8",
)

SOURCE_RELATIONS = {
    "all": "keypoint_performance",
    "latest-dataset": "keypoint_performance_latest",
    "latest-recording": "recording_keypoint_performance_latest",
}

GROUP_BY_VALUES = ("all", "recording", "dataset")

KEYPOINT_FIELDS = (
    "dataset_id",
    "recording_id",
    "keypoint_run",
    "keypoint_created_utc",
    "zarr_use",
    "keypoint_method",
    "model_run_id",
    "model_set_id",
    "model_path",
    "model_name",
    "source_crop_run",
    "source_detect_run",
    "source_refined_run",
    "source_crop_storage_mode",
    "source_crop_signature",
    "source_crop_revision",
    "source_roi_image_representation",
    "source_roi_pixel_contract_name",
    "source_roi_read_mode",
    "roi_cache_policy",
    "source_roi_cache_used",
    "source_roi_cache_backend",
    "source_roi_live_acceleration_effective",
    "source_roi_live_gpu_chunk_frames",
    "input_mode_requested",
    "input_mode_effective",
    "total_rois",
    "successful_detections",
    "failed_detections",
    "success_rate_percent",
    "frames_with_keypoints",
    "mean_confidence",
    "duration_seconds",
    "inference_duration_seconds",
    "keypoints_per_second",
    "inference_average_fps",
    "batch_size",
    "imgsz",
    "conf_threshold",
    "iou_threshold",
    "summary_statistics_json",
    "zarr_mtime_ns",
    "updated_utc",
)


def _registry_path_from_arg(value: Optional[str]) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return RegistryPaths.from_env(Path.cwd()).path.expanduser().resolve()


def _object_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE name = ? AND type IN ('table', 'view')
        LIMIT 1;
        """,
        (name,),
    ).fetchone()
    return row is not None


def _relation_columns(conn: sqlite3.Connection, relation: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({relation});").fetchall()
    return {str(row["name"]) for row in rows}


def _select_expr(columns: set[str], name: str, *, alias: str) -> str:
    if name in columns:
        return f"{alias}.{name} AS {name}"
    return f"NULL AS {name}"


def _normalize(value: object) -> str:
    if value is None:
        return MISSING
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "ignore")
    text = str(value).strip()
    return text if text else MISSING


def _parse_contracts(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_CANDIDATE_COMPATIBLE_CONTRACTS
    contracts: list[str] = []
    for value in values:
        for part in str(value).split(","):
            contract = part.strip()
            if contract:
                contracts.append(contract)
    return tuple(dict.fromkeys(contracts))


def _build_source_sql(
    conn: sqlite3.Connection,
    *,
    source_relation: str,
    path_contains: Optional[str],
    recording_contains: Optional[str],
    dataset_contains: Optional[str],
) -> tuple[str, list[object]]:
    if source_relation not in SOURCE_RELATIONS:
        raise ValueError(f"Unknown source relation: {source_relation!r}")
    relation = SOURCE_RELATIONS[source_relation]
    if not _object_exists(conn, relation):
        raise RuntimeError(f"registry is missing {relation}")

    columns = _relation_columns(conn, relation)
    select_cols = [_select_expr(columns, field, alias="kp") for field in KEYPOINT_FIELDS]
    params: list[object] = []

    if source_relation == "latest-recording":
        select_cols.extend(
            [
                _select_expr(columns, "zarr_path", alias="kp"),
                _select_expr(columns, "artifact_kind", alias="kp"),
                _select_expr(columns, "dataset_status", alias="kp"),
                _select_expr(columns, "rig_id", alias="kp"),
                _select_expr(columns, "arena_id", alias="kp"),
                _select_expr(columns, "camera_id", alias="kp"),
            ]
        )
        sql = f"SELECT {', '.join(select_cols)} FROM {relation} kp"
        zarr_path_filter_expr = "kp.zarr_path"
    else:
        dataset_columns = _relation_columns(conn, "datasets") if _object_exists(conn, "datasets") else set()
        select_cols.extend(
            [
                _select_expr(dataset_columns, "zarr_path", alias="d"),
                _select_expr(dataset_columns, "artifact_kind", alias="d"),
                _select_expr(dataset_columns, "status", alias="d").replace(" AS status", " AS dataset_status"),
                "NULL AS rig_id",
                "NULL AS arena_id",
                "NULL AS camera_id",
            ]
        )
        sql = f"SELECT {', '.join(select_cols)} FROM {relation} kp LEFT JOIN datasets d ON d.dataset_id = kp.dataset_id"
        zarr_path_filter_expr = "d.zarr_path"

    clauses: list[str] = []
    if path_contains:
        clauses.append(f"COALESCE({zarr_path_filter_expr}, '') LIKE ?")
        params.append(f"%{path_contains}%")
    if recording_contains:
        clauses.append("COALESCE(kp.recording_id, '') LIKE ?")
        params.append(f"%{recording_contains}%")
    if dataset_contains:
        clauses.append("COALESCE(kp.dataset_id, '') LIKE ?")
        params.append(f"%{dataset_contains}%")
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY recording_id, dataset_id, keypoint_run"
    return sql, params


def _iter_keypoint_rows(
    conn: sqlite3.Connection,
    *,
    source_relation: str,
    path_contains: Optional[str],
    recording_contains: Optional[str],
    dataset_contains: Optional[str],
) -> Iterable[sqlite3.Row]:
    sql, params = _build_source_sql(
        conn,
        source_relation=source_relation,
        path_contains=path_contains,
        recording_contains=recording_contains,
        dataset_contains=dataset_contains,
    )
    yield from conn.execute(sql, params)


def _row_record(row: sqlite3.Row, *, source_relation: str) -> dict[str, Any]:
    contract = _normalize(row["source_roi_pixel_contract_name"])
    read_mode = _normalize(row["source_roi_read_mode"])
    cache_backend = _normalize(row["source_roi_cache_backend"])
    input_mode = _normalize(row["input_mode_effective"])
    return {
        "record_type": "keypoint_run",
        "source_relation": source_relation,
        "dataset_id": row["dataset_id"],
        "recording_id": row["recording_id"],
        "keypoint_run": row["keypoint_run"],
        "keypoint_created_utc": row["keypoint_created_utc"],
        "keypoint_method": row["keypoint_method"],
        "model_run_id": row["model_run_id"],
        "model_set_id": row["model_set_id"],
        "model_name": row["model_name"],
        "source_crop_run": row["source_crop_run"],
        "source_detect_run": row["source_detect_run"],
        "source_refined_run": row["source_refined_run"],
        "source_crop_storage_mode": row["source_crop_storage_mode"],
        "source_crop_signature": row["source_crop_signature"],
        "source_crop_revision": row["source_crop_revision"],
        "source_roi_image_representation": row["source_roi_image_representation"],
        "source_roi_pixel_contract_name": contract,
        "source_roi_read_mode": read_mode,
        "roi_cache_policy": row["roi_cache_policy"],
        "source_roi_cache_used": row["source_roi_cache_used"],
        "source_roi_cache_backend": cache_backend,
        "source_roi_live_acceleration_effective": row["source_roi_live_acceleration_effective"],
        "source_roi_live_gpu_chunk_frames": row["source_roi_live_gpu_chunk_frames"],
        "input_mode_requested": _normalize(row["input_mode_requested"]),
        "input_mode_effective": input_mode,
        "total_rois": row["total_rois"],
        "successful_detections": row["successful_detections"],
        "failed_detections": row["failed_detections"],
        "success_rate_percent": row["success_rate_percent"],
        "frames_with_keypoints": row["frames_with_keypoints"],
        "batch_size": row["batch_size"],
        "imgsz": row["imgsz"],
        "zarr_path": row["zarr_path"],
        "artifact_kind": row["artifact_kind"],
        "dataset_status": row["dataset_status"],
        "rig_id": row["rig_id"],
        "arena_id": row["arena_id"],
        "camera_id": row["camera_id"],
        "updated_utc": row["updated_utc"],
        "contract_presence": "missing" if contract == MISSING else "explicit",
    }


def _group_key(row: Mapping[str, Any], *, group_by: str) -> tuple[str, str]:
    if group_by == "all":
        return ("all", "all")
    if group_by == "recording":
        return ("recording", _normalize(row.get("recording_id")))
    if group_by == "dataset":
        return ("dataset", _normalize(row.get("dataset_id")))
    raise ValueError(f"Unknown group_by: {group_by!r}")


def _group_status(
    contracts: Sequence[str],
    *,
    candidate_compatible_contracts: set[str],
) -> tuple[str, str]:
    non_missing = {contract for contract in contracts if contract != MISSING}
    has_missing = any(contract == MISSING for contract in contracts)
    if not non_missing:
        return "unknown_only", "unknown"
    if len(non_missing) == 1 and not has_missing:
        return "explicit_single", "single_contract"
    if len(non_missing) == 1 and has_missing:
        return "explicit_with_unknown", "unknown"
    if has_missing:
        return "mixed_with_unknown", "unknown"
    if non_missing <= candidate_compatible_contracts:
        return "mixed_explicit", "candidate_compatible"
    return "mixed_explicit", "needs_review"


def _summarize_groups(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_by: str,
    candidate_compatible_contracts: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row, group_by=group_by)].append(row)

    candidate_set = set(candidate_compatible_contracts)
    summaries: list[dict[str, Any]] = []
    for (group_type, group_id), group_rows in sorted(grouped.items()):
        contracts = [str(row["source_roi_pixel_contract_name"]) for row in group_rows]
        status, compatibility_status = _group_status(
            contracts,
            candidate_compatible_contracts=candidate_set,
        )
        summaries.append(
            {
                "record_type": "contract_group",
                "group_by": group_by,
                "group_type": group_type,
                "group_id": group_id,
                "row_count": len(group_rows),
                "contract_counts": dict(sorted(Counter(contracts).items())),
                "read_mode_counts": dict(
                    sorted(Counter(str(row["source_roi_read_mode"]) for row in group_rows).items())
                ),
                "cache_backend_counts": dict(
                    sorted(Counter(str(row["source_roi_cache_backend"]) for row in group_rows).items())
                ),
                "input_mode_counts": dict(
                    sorted(Counter(str(row["input_mode_effective"]) for row in group_rows).items())
                ),
                "contract_group_status": status,
                "compatibility_status": compatibility_status,
                "candidate_compatible_contracts": list(candidate_compatible_contracts),
            }
        )
    return summaries


def build_keypoint_contract_coverage_report(
    registry_path: Path,
    *,
    source_relation: str = "all",
    group_by: str = "recording",
    candidate_compatible_contracts: Sequence[str] = DEFAULT_CANDIDATE_COMPATIBLE_CONTRACTS,
    path_contains: Optional[str] = None,
    recording_contains: Optional[str] = None,
    dataset_contains: Optional[str] = None,
) -> dict[str, Any]:
    if group_by not in GROUP_BY_VALUES:
        raise ValueError(f"group_by must be one of {GROUP_BY_VALUES}: {group_by!r}")

    conn = sqlite3.connect(str(registry_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = [
            _row_record(row, source_relation=source_relation)
            for row in _iter_keypoint_rows(
                conn,
                source_relation=source_relation,
                path_contains=path_contains,
                recording_contains=recording_contains,
                dataset_contains=dataset_contains,
            )
        ]
    finally:
        conn.close()

    groups = _summarize_groups(
        rows,
        group_by=group_by,
        candidate_compatible_contracts=candidate_compatible_contracts,
    )
    return {
        "schema_version": "palette.keypoint_contract_coverage_report.v1",
        "registry_path": str(registry_path),
        "source_relation": source_relation,
        "group_by": group_by,
        "filters": {
            "path_contains": path_contains,
            "recording_contains": recording_contains,
            "dataset_contains": dataset_contains,
        },
        "candidate_compatible_contracts": list(candidate_compatible_contracts),
        "row_count": len(rows),
        "group_count": len(groups),
        "contract_counts": dict(
            sorted(Counter(str(row["source_roi_pixel_contract_name"]) for row in rows).items())
        ),
        "read_mode_counts": dict(sorted(Counter(str(row["source_roi_read_mode"]) for row in rows).items())),
        "cache_backend_counts": dict(
            sorted(Counter(str(row["source_roi_cache_backend"]) for row in rows).items())
        ),
        "input_mode_counts": dict(sorted(Counter(str(row["input_mode_effective"]) for row in rows).items())),
        "group_status_counts": dict(
            sorted(Counter(str(group["contract_group_status"]) for group in groups).items())
        ),
        "compatibility_status_counts": dict(
            sorted(Counter(str(group["compatibility_status"]) for group in groups).items())
        ),
        "groups": groups,
        "rows": rows,
    }


def _format_counts(counts: Mapping[str, Any]) -> str:
    if not counts:
        return "{}"
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def print_text_report(report: Mapping[str, Any], *, limit: int = 40) -> None:
    print("keypoint_contract_coverage_report")
    print(f"registry: {report['registry_path']}")
    print(f"source_relation: {report['source_relation']}")
    print(f"group_by: {report['group_by']}")
    print(f"filters: {report['filters']}")
    print(f"candidate_compatible_contracts: {report['candidate_compatible_contracts']}")
    print(f"row_count: {report['row_count']}")
    print(f"group_count: {report['group_count']}")
    print(f"contract_counts: {_format_counts(report['contract_counts'])}")
    print(f"read_mode_counts: {_format_counts(report['read_mode_counts'])}")
    print(f"cache_backend_counts: {_format_counts(report['cache_backend_counts'])}")
    print(f"input_mode_counts: {_format_counts(report['input_mode_counts'])}")
    print(f"group_status_counts: {_format_counts(report['group_status_counts'])}")
    print(f"compatibility_status_counts: {_format_counts(report['compatibility_status_counts'])}")

    groups = report.get("groups")
    if isinstance(groups, list) and groups:
        print()
        print("sample_groups:")
        for group in groups[: max(int(limit), 0)]:
            if not isinstance(group, Mapping):
                continue
            print(
                f"  {group.get('contract_group_status', '-'):<22} "
                f"{group.get('compatibility_status', '-'):<22} "
                f"{group.get('group_type', '-')}:"
                f"{group.get('group_id', '-')} "
                f"rows={group.get('row_count')} "
                f"contracts={group.get('contract_counts')}"
            )


def _write_contract_report_jsonl(path: Path, report: Mapping[str, Any]) -> None:
    write_jsonl_atomic(
        path,
        [*report.get("groups", []), *report.get("rows", [])],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report keypoint ROI pixel-contract coverage from keypoint_performance. "
            "This is read-only telemetry for deciding training/export compatibility."
        )
    )
    parser.add_argument(
        "--registry",
        help="Registry SQLite path. Defaults to RegistryPaths.from_env(Path.cwd()).",
    )
    parser.add_argument(
        "--source-relation",
        choices=tuple(SOURCE_RELATIONS),
        default="all",
        help="Registry keypoint surface to inspect. Default: all.",
    )
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="Alias for --source-relation latest-recording.",
    )
    parser.add_argument(
        "--group-by",
        choices=GROUP_BY_VALUES,
        default="recording",
        help="Grouping used to flag mixed versus unknown contracts. Default: recording.",
    )
    parser.add_argument("--path-contains", help="Filter rows whose zarr_path contains this text.")
    parser.add_argument("--recording-contains", help="Filter rows whose recording_id contains this text.")
    parser.add_argument("--dataset-contains", help="Filter rows whose dataset_id contains this text.")
    parser.add_argument(
        "--candidate-compatible-contract",
        action="append",
        help=(
            "Contract name to treat as part of the candidate-compatible set for mixed-explicit "
            "group labeling. Repeatable and comma-separated values are accepted. Defaults to the "
            "current Orange/luma candidate pair."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    parser.add_argument("--output-json", type=Path, help="Optional path to write the JSON report.")
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        help="Optional path to write one JSONL record per group followed by one per keypoint run.",
    )
    parser.add_argument("--limit", type=int, default=40, help="Maximum sample groups to print in text mode.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    source_relation = "latest-recording" if args.current_only else args.source_relation
    report = build_keypoint_contract_coverage_report(
        _registry_path_from_arg(args.registry),
        source_relation=source_relation,
        group_by=args.group_by,
        candidate_compatible_contracts=_parse_contracts(args.candidate_compatible_contract),
        path_contains=args.path_contains,
        recording_contains=args.recording_contains,
        dataset_contains=args.dataset_contains,
    )

    if args.output_json:
        args.output_json.expanduser().resolve().write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.output_jsonl:
        _write_contract_report_jsonl(args.output_jsonl.expanduser().resolve(), report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text_report(report, limit=max(int(args.limit), 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
