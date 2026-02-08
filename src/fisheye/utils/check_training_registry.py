#!/usr/bin/env python3
"""Summarize training sets or training runs from the registry."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fisheye.registry.db import Registry, RegistryPaths

try:  # optional rich dependency
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover - optional
    Console = None
    Table = None


@dataclass
class SetRow:
    set_id: str
    name: Optional[str]
    created_utc: Optional[str]
    dataset_count: Optional[int]
    skeleton_id: Optional[str]
    manifest_path: Optional[str]
    config_path: Optional[str]
    run_id: Optional[str]


@dataclass
class ModelRow:
    run_id: str
    set_id: Optional[str]
    set_name: Optional[str]
    run_status: Optional[str]
    created_utc: Optional[str]
    skeleton_id: Optional[str]
    model_path: Optional[str]
    metrics_path: Optional[str]
    manifest_path: Optional[str]
    config_path: Optional[str]
    onnx_path: Optional[str]
    onnx_requires_plugins: Optional[int]
    onnx_plugin_ops_json: Optional[str]
    onnx_plugin_versions_json: Optional[str]
    trt_path: Optional[str]
    model_source: Optional[str]
    onnx_source: Optional[str]
    trt_source: Optional[str]
    roi_count: Optional[int]
    metrics_summary: Optional[str]


@dataclass
class OnnxRow:
    run_id: str
    set_id: Optional[str]
    set_name: Optional[str]
    detection_model_run_id: Optional[str]
    created_utc: Optional[str]
    skeleton_id: Optional[str]
    path: Optional[str]
    opset: Optional[int]
    img_h: Optional[int]
    img_w: Optional[int]
    max_batch: Optional[int]
    dynamic_shapes: Optional[int]
    exporter_torch_version: Optional[str]
    exporter_cuda_version: Optional[str]
    exporter_hostname: Optional[str]
    requires_plugins: Optional[int]
    plugin_ops_json: Optional[str]
    plugin_versions_json: Optional[str]


@dataclass
class TensorRTRow:
    run_id: str
    set_id: Optional[str]
    set_name: Optional[str]
    detection_model_run_id: Optional[str]
    onnx_run_id: Optional[str]
    onnx_path: Optional[str]
    created_utc: Optional[str]
    skeleton_id: Optional[str]
    precision: Optional[str]
    path: Optional[str]
    trt_version: Optional[str]
    cuda_version: Optional[str]
    img_h: Optional[int]
    img_w: Optional[int]
    max_batch: Optional[int]
    dynamic_shapes: Optional[int]
    gpu_name: Optional[str]
    gpu_uuid: Optional[str]
    compute_capability: Optional[str]
    system_hostname: Optional[str]
    requires_plugins: Optional[int]
    plugin_ops_json: Optional[str]
    plugin_versions_json: Optional[str]


@dataclass
class KeypointQualitySummary:
    total_rows: int
    passing_rows: int
    excluded_rows: int
    exclusion_reasons: Dict[str, int]


KEYPOINT_GATE_REVIEW_STATE = "approved"
KEYPOINT_GATE_REVIEW_INTENDED_USE = "training"
KEYPOINT_GATE_MIN_RATE = 0.70


def _status_text(value: Optional[bool]) -> str:
    if value is None:
        return "—"
    return "OK" if value else "MISS"


def _status_rich(value: Optional[bool]) -> str:
    if value is None:
        return "—"
    return "[chartreuse1]OK[/chartreuse1]" if value else "[red]MISS[/red]"


def _status_with_details(value: Optional[bool], details: Optional[str], *, rich: bool) -> str:
    if value is None:
        return "—"
    if not value:
        return "[red]MISS[/red]" if rich else "MISS"
    base = "[chartreuse1]OK[/chartreuse1]" if rich else "OK"
    if details:
        return f"{base} ({details})"
    return base


def _source_value(value: Optional[str]) -> str:
    if value == "new":
        return "new"
    return "—"


def _source_value_rich(value: Optional[str]) -> str:
    if value == "new":
        return "[chartreuse1]new[/chartreuse1]"
    return "—"


def _source_summary(
    *,
    model_source: Optional[str],
    onnx_source: Optional[str],
    trt_source: Optional[str],
    rich: bool,
) -> str:
    formatter = _source_value_rich if rich else _source_value
    return " ".join(
        [
            f"M:{formatter(model_source)}",
            f"O:{formatter(onnx_source)}",
            f"T:{formatter(trt_source)}",
        ]
    )


def _path_ok(path: Optional[str]) -> Optional[bool]:
    if not path:
        return None
    return Path(path).exists()


def _parse_count(payload: Optional[str]) -> Optional[int]:
    if not payload:
        return None
    try:
        items = json.loads(payload)
        if isinstance(items, list):
            return len(items)
    except Exception:
        return None
    return None


def _sum_manifest_rois(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        return None
    total = 0
    found = False
    for item in datasets:
        if not isinstance(item, dict):
            continue
        # Detect manifests store dataset sample totals as total_bboxes.
        # Pose manifests store dataset sample totals as keypoints_total.
        for key in ("total_bboxes", "keypoints_total", "total_keypoints", "total_rois"):
            value = item.get(key)
            if isinstance(value, (int, float)):
                total += int(value)
                found = True
                break
    return total if found else None


def _parse_json_dict(payload: Optional[str]) -> Optional[Dict[str, Any]]:
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _metric_value(payload: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _format_metric(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return f"{value:.3f}"


def _metrics_summary_from_json(payload: Optional[str]) -> Optional[str]:
    data = _parse_json_dict(payload)
    if not data:
        return None
    parts: List[str] = []
    map50 = _format_metric(_metric_value(data, ["mAP50", "map50"]))
    map50_95 = _format_metric(_metric_value(data, ["mAP50_95", "mAP50-95", "map50_95", "map50-95"]))
    precision = _format_metric(_metric_value(data, ["precision", "metrics/precision(B)"]))
    recall = _format_metric(_metric_value(data, ["recall", "metrics/recall(B)"]))
    if map50 is not None:
        parts.append(f"mAP50={map50}")
    if map50_95 is not None:
        parts.append(f"mAP50-95={map50_95}")
    if precision is not None:
        parts.append(f"P={precision}")
    if recall is not None:
        parts.append(f"R={recall}")
    if not parts:
        return None
    return ", ".join(parts)


def _fetch_latest_run(registry: Registry, set_id: str) -> Dict[str, Optional[str]]:
    row = registry.conn.execute(
        """
        SELECT
            tr.run_id,
            dm.status AS status,
            tr.created_utc,
            tr.skeleton_id,
            dm.model_path AS model_path,
            dm.metrics_path AS metrics_path,
            tr.manifest_path,
            tr.config_path,
            dm.final_metrics_json AS final_metrics_json,
            CASE WHEN dm.run_id IS NOT NULL THEN 'new' ELSE NULL END AS model_source
        FROM training_runs tr
        LEFT JOIN detection_models dm ON dm.run_id = tr.run_id
        WHERE tr.set_id = ?
        ORDER BY tr.created_utc DESC, tr.run_id DESC
        LIMIT 1;
        """,
        (set_id,),
    ).fetchone()
    if not row:
        return {}
    return {
        "run_id": row["run_id"],
        "run_status": row["status"],
        "run_created_utc": row["created_utc"],
        "skeleton_id": row["skeleton_id"],
        "model_path": row["model_path"],
        "metrics_path": row["metrics_path"],
        "manifest_path": row["manifest_path"],
        "config_path": row["config_path"],
        "final_metrics_json": row["final_metrics_json"],
        "model_source": row["model_source"],
    }


def _fetch_exports(registry: Registry, run_id: str) -> Dict[str, Any]:
    onnx_row = registry.conn.execute(
        """
        SELECT path, requires_plugins, plugin_ops_json, plugin_versions_json
        FROM onnx_models
        WHERE run_id = ?;
        """,
        (run_id,),
    ).fetchone()
    trt_row = registry.conn.execute(
        """
        SELECT path
        FROM tensorrt_models
        WHERE run_id = ?
        ORDER BY CASE WHEN precision = 'fp16' THEN 0 ELSE 1 END, created_utc DESC
        LIMIT 1;
        """,
        (run_id,),
    ).fetchone()
    onnx_new = onnx_row["path"] if onnx_row else None
    trt_new = trt_row["path"] if trt_row else None
    return {
        "onnx_path": onnx_new,
        "onnx_requires_plugins": onnx_row["requires_plugins"] if onnx_row else None,
        "onnx_plugin_ops_json": onnx_row["plugin_ops_json"] if onnx_row else None,
        "onnx_plugin_versions_json": onnx_row["plugin_versions_json"] if onnx_row else None,
        "trt_path": trt_new,
        "onnx_source": "new" if onnx_new else None,
        "trt_source": "new" if trt_new else None,
    }


def _onnx_plugin_details(
    requires_plugins: Optional[int],
    plugin_ops_json: Optional[str],
    plugin_versions_json: Optional[str],
) -> Optional[str]:
    if not requires_plugins:
        return None

    ops: list[str] = []
    versions: dict[str, str] = {}
    if plugin_ops_json:
        try:
            parsed_ops = json.loads(plugin_ops_json)
            if isinstance(parsed_ops, list):
                ops = [str(item) for item in parsed_ops if item]
        except Exception:
            ops = []
    if plugin_versions_json:
        try:
            parsed_versions = json.loads(plugin_versions_json)
            if isinstance(parsed_versions, dict):
                versions = {
                    str(key): str(value)
                    for key, value in parsed_versions.items()
                    if key and value is not None
                }
        except Exception:
            versions = {}

    if ops:
        rendered: list[str] = []
        for op in ops:
            version = versions.get(op)
            rendered.append(f"{op}@{version}" if version else op)
        return "plugins=" + ",".join(rendered)

    return "plugins=true"


def _shape_summary(
    *,
    img_h: Optional[int],
    img_w: Optional[int],
    max_batch: Optional[int],
    dynamic_shapes: Optional[int],
) -> str:
    size = f"{img_w}x{img_h}" if img_h and img_w else "—"
    batch = str(max_batch) if max_batch is not None else "—"
    dyn = "dyn" if dynamic_shapes else "fix"
    return f"{batch}:{size} ({dyn})"


def _load_set_rows(registry: Registry, set_filter: Optional[str], limit: Optional[int]) -> List[SetRow]:
    sql = [
        "SELECT set_id, name, dataset_ids_json, created_utc",
        "FROM training_sets",
    ]
    params: List[Any] = []
    if set_filter:
        sql.append("WHERE set_id = ?")
        params.append(set_filter)
    sql.append("ORDER BY created_utc DESC, set_id DESC")
    if limit and limit > 0:
        sql.append("LIMIT ?")
        params.append(limit)
    rows = registry.conn.execute(" ".join(sql), params).fetchall()

    results: List[SetRow] = []
    for row in rows:
        latest = _fetch_latest_run(registry, row["set_id"])
        results.append(
            SetRow(
                set_id=row["set_id"],
                name=row["name"],
                created_utc=row["created_utc"],
                dataset_count=_parse_count(row["dataset_ids_json"]),
                skeleton_id=latest.get("skeleton_id"),
                manifest_path=latest.get("manifest_path"),
                config_path=latest.get("config_path"),
                run_id=latest.get("run_id"),
            )
        )
    return results


def _load_onnx_rows(
    registry: Registry,
    set_filter: Optional[str],
    limit: Optional[int],
    *,
    hide_unlinked: bool,
) -> List[OnnxRow]:
    sql = [
        "SELECT",
        "om.run_id,",
        "om.set_id,",
        "ts.name AS set_name,",
        "om.detection_model_run_id,",
        "om.created_utc,",
        "om.skeleton_id,",
        "om.path,",
        "om.opset,",
        "om.img_h,",
        "om.img_w,",
        "om.max_batch,",
        "om.dynamic_shapes,",
        "om.exporter_torch_version,",
        "om.exporter_cuda_version,",
        "om.exporter_hostname,",
        "om.requires_plugins,",
        "om.plugin_ops_json,",
        "om.plugin_versions_json",
        "FROM onnx_models om",
        "LEFT JOIN training_sets ts ON ts.set_id = om.set_id",
    ]
    params: List[Any] = []
    where_clauses: List[str] = []
    if set_filter:
        where_clauses.append("om.set_id = ?")
        params.append(set_filter)
    elif hide_unlinked:
        where_clauses.append(
            "om.set_id IS NOT NULL AND TRIM(om.set_id) <> '' AND om.set_id IN (SELECT set_id FROM training_sets)"
        )
    if where_clauses:
        sql.append("WHERE " + " AND ".join(where_clauses))
    sql.append("ORDER BY om.created_utc DESC, om.run_id DESC")
    if limit and limit > 0:
        sql.append("LIMIT ?")
        params.append(limit)
    rows = registry.conn.execute(" ".join(sql), params).fetchall()
    return [
        OnnxRow(
            run_id=row["run_id"],
            set_id=row["set_id"],
            set_name=row["set_name"],
            detection_model_run_id=row["detection_model_run_id"],
            created_utc=row["created_utc"],
            skeleton_id=row["skeleton_id"],
            path=row["path"],
            opset=row["opset"],
            img_h=row["img_h"],
            img_w=row["img_w"],
            max_batch=row["max_batch"],
            dynamic_shapes=row["dynamic_shapes"],
            exporter_torch_version=row["exporter_torch_version"],
            exporter_cuda_version=row["exporter_cuda_version"],
            exporter_hostname=row["exporter_hostname"],
            requires_plugins=row["requires_plugins"],
            plugin_ops_json=row["plugin_ops_json"],
            plugin_versions_json=row["plugin_versions_json"],
        )
        for row in rows
    ]


def _load_tensorrt_rows(
    registry: Registry,
    set_filter: Optional[str],
    limit: Optional[int],
    *,
    hide_unlinked: bool,
) -> List[TensorRTRow]:
    sql = [
        "SELECT",
        "tm.run_id,",
        "tm.set_id,",
        "ts.name AS set_name,",
        "tm.detection_model_run_id,",
        "tm.onnx_run_id,",
        "om.path AS onnx_path,",
        "tm.created_utc,",
        "tm.skeleton_id,",
        "tm.precision,",
        "tm.path,",
        "tm.trt_version,",
        "tm.cuda_version,",
        "tm.img_h,",
        "tm.img_w,",
        "tm.max_batch,",
        "tm.dynamic_shapes,",
        "tm.gpu_name,",
        "tm.gpu_uuid,",
        "tm.compute_capability,",
        "tm.system_hostname,",
        "tm.requires_plugins,",
        "tm.plugin_ops_json,",
        "tm.plugin_versions_json",
        "FROM tensorrt_models tm",
        "LEFT JOIN training_sets ts ON ts.set_id = tm.set_id",
        "LEFT JOIN onnx_models om ON om.run_id = tm.onnx_run_id",
    ]
    params: List[Any] = []
    where_clauses: List[str] = []
    if set_filter:
        where_clauses.append("tm.set_id = ?")
        params.append(set_filter)
    elif hide_unlinked:
        where_clauses.append(
            "tm.set_id IS NOT NULL AND TRIM(tm.set_id) <> '' AND tm.set_id IN (SELECT set_id FROM training_sets)"
        )
    if where_clauses:
        sql.append("WHERE " + " AND ".join(where_clauses))
    sql.append("ORDER BY tm.created_utc DESC, tm.run_id DESC")
    if limit and limit > 0:
        sql.append("LIMIT ?")
        params.append(limit)
    rows = registry.conn.execute(" ".join(sql), params).fetchall()
    return [
        TensorRTRow(
            run_id=row["run_id"],
            set_id=row["set_id"],
            set_name=row["set_name"],
            detection_model_run_id=row["detection_model_run_id"],
            onnx_run_id=row["onnx_run_id"],
            onnx_path=row["onnx_path"],
            created_utc=row["created_utc"],
            skeleton_id=row["skeleton_id"],
            precision=row["precision"],
            path=row["path"],
            trt_version=row["trt_version"],
            cuda_version=row["cuda_version"],
            img_h=row["img_h"],
            img_w=row["img_w"],
            max_batch=row["max_batch"],
            dynamic_shapes=row["dynamic_shapes"],
            gpu_name=row["gpu_name"],
            gpu_uuid=row["gpu_uuid"],
            compute_capability=row["compute_capability"],
            system_hostname=row["system_hostname"],
            requires_plugins=row["requires_plugins"],
            plugin_ops_json=row["plugin_ops_json"],
            plugin_versions_json=row["plugin_versions_json"],
        )
        for row in rows
    ]


def _load_model_rows(
    registry: Registry,
    set_filter: Optional[str],
    limit: Optional[int],
    *,
    hide_unlinked: bool,
) -> List[ModelRow]:
    sql = [
        "SELECT",
        "tr.run_id,",
        "tr.set_id,",
        "ts.name AS set_name,",
        "COALESCE(dm.status, tr.status) AS status,",
        "tr.created_utc,",
        "tr.skeleton_id,",
        "COALESCE(dm.model_path, tr.model_path) AS model_path,",
        "COALESCE(dm.metrics_path, tr.metrics_path) AS metrics_path,",
        "tr.manifest_path,",
        "tr.config_path,",
        "COALESCE(dm.final_metrics_json, tr.final_metrics_json) AS final_metrics_json,",
        "CASE WHEN dm.run_id IS NOT NULL THEN 'new' ELSE NULL END AS model_source",
        "FROM training_runs tr",
        "LEFT JOIN detection_models dm ON dm.run_id = tr.run_id",
        "LEFT JOIN training_sets ts ON ts.set_id = tr.set_id",
    ]
    params: List[Any] = []
    where_clauses: List[str] = []
    if set_filter:
        where_clauses.append("tr.set_id = ?")
        params.append(set_filter)
    elif hide_unlinked:
        where_clauses.append(
            "tr.set_id IS NOT NULL AND TRIM(tr.set_id) <> '' AND tr.set_id IN (SELECT set_id FROM training_sets)"
        )
    if where_clauses:
        sql.append("WHERE " + " AND ".join(where_clauses))
    sql.append("ORDER BY tr.created_utc DESC, tr.run_id DESC")
    if limit and limit > 0:
        sql.append("LIMIT ?")
        params.append(limit)
    rows = registry.conn.execute(" ".join(sql), params).fetchall()
    results: List[ModelRow] = []
    for row in rows:
        exports = _fetch_exports(registry, row["run_id"])
        roi_count = _sum_manifest_rois(row["manifest_path"])
        results.append(
            ModelRow(
                run_id=row["run_id"],
                set_id=row["set_id"],
                set_name=row["set_name"],
                run_status=row["status"],
                created_utc=row["created_utc"],
                skeleton_id=row["skeleton_id"],
                model_path=row["model_path"],
                metrics_path=row["metrics_path"],
                manifest_path=row["manifest_path"],
                config_path=row["config_path"],
                onnx_path=exports.get("onnx_path"),
                onnx_requires_plugins=exports.get("onnx_requires_plugins"),
                onnx_plugin_ops_json=exports.get("onnx_plugin_ops_json"),
                onnx_plugin_versions_json=exports.get("onnx_plugin_versions_json"),
                trt_path=exports.get("trt_path"),
                model_source=row["model_source"],
                onnx_source=exports.get("onnx_source"),
                trt_source=exports.get("trt_source"),
                roi_count=roi_count,
                metrics_summary=_metrics_summary_from_json(row["final_metrics_json"]),
            )
        )
    return results


def _format_time(value: Optional[str]) -> str:
    if not value:
        return "—"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return value


_RUN_ID_HASH_RE = re.compile(r"^[0-9a-f]{8}$")
_RUN_ID_STAMP_RE = re.compile(r"^\d{8}-\d{6}$")
_RUN_ID_VER_RE = re.compile(r"^v\d{3,}$")


def _run_id_style(run_id: Optional[str]) -> str:
    text = str(run_id or "").strip()
    if not text:
        return "legacy"
    parts = text.split("_")
    if len(parts) < 7:
        return "legacy"
    hash_token = parts[-1]
    stamp_token = parts[-2]
    task_token = parts[-3]
    version_token = parts[-4]
    if task_token not in {"detect", "pose"}:
        return "legacy"
    if _RUN_ID_HASH_RE.fullmatch(hash_token) is None:
        return "legacy"
    if _RUN_ID_STAMP_RE.fullmatch(stamp_token) is None:
        return "legacy"
    if _RUN_ID_VER_RE.fullmatch(version_token) is None:
        return "legacy"
    return "contract"


def _dataset_ids_for_set(registry: Registry, set_id: Optional[str]) -> Optional[List[str]]:
    if not set_id:
        return None
    row = registry.conn.execute(
        "SELECT dataset_ids_json FROM training_sets WHERE set_id = ?;",
        (set_id,),
    ).fetchone()
    if not row:
        return []
    try:
        payload = json.loads(row["dataset_ids_json"] or "[]")
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload if item is not None]


def _load_keypoint_quality_rows(
    registry: Registry,
    *,
    set_filter: Optional[str],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    sql = ["SELECT * FROM keypoint_quality_overview WHERE 1=1"]
    params: List[Any] = []
    dataset_ids = _dataset_ids_for_set(registry, set_filter)
    if dataset_ids is not None:
        if not dataset_ids:
            return []
        placeholders = ", ".join("?" for _ in dataset_ids)
        sql.append(f"AND dataset_id IN ({placeholders})")
        params.extend(dataset_ids)
    sql.append("ORDER BY dataset_id, keypoint_method")
    if limit and limit > 0:
        sql.append("LIMIT ?")
        params.append(int(limit))
    rows = registry.conn.execute(" ".join(sql), params).fetchall()
    return [dict(row) for row in rows]


def _keypoint_row_passes_default_gate(row: Dict[str, Any]) -> bool:
    rate = row.get("usable_keypoints_rate")
    return (
        row.get("review_state") == KEYPOINT_GATE_REVIEW_STATE
        and row.get("review_intended_use") == KEYPOINT_GATE_REVIEW_INTENDED_USE
        and isinstance(rate, (int, float))
        and float(rate) >= KEYPOINT_GATE_MIN_RATE
    )


def _keypoint_exclusion_reason(row: Dict[str, Any]) -> Optional[str]:
    if _keypoint_row_passes_default_gate(row):
        return None
    state = row.get("review_state")
    intended_use = row.get("review_intended_use")
    rate = row.get("usable_keypoints_rate")
    if not state or not intended_use:
        return "missing review"
    if state != KEYPOINT_GATE_REVIEW_STATE or intended_use != KEYPOINT_GATE_REVIEW_INTENDED_USE:
        return "wrong state/use"
    if rate is None:
        return "missing rate"
    if float(rate) < KEYPOINT_GATE_MIN_RATE:
        return "low rate"
    return "other"


def _summarize_keypoint_quality_rows(rows: List[Dict[str, Any]]) -> KeypointQualitySummary:
    passing = 0
    reasons: Dict[str, int] = {}
    for row in rows:
        reason = _keypoint_exclusion_reason(row)
        if reason is None:
            passing += 1
            continue
        reasons[reason] = reasons.get(reason, 0) + 1
    total = len(rows)
    excluded = total - passing
    ordered = dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0])))
    return KeypointQualitySummary(
        total_rows=total,
        passing_rows=passing,
        excluded_rows=excluded,
        exclusion_reasons=ordered,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")
    parser.add_argument("--set-id", type=str, help="Filter to a specific training set id.")
    parser.add_argument(
        "--view",
        choices=["sets", "models", "onnx", "tensorrt", "keypoint-quality"],
        default="sets",
        help="Select output view: sets, models, onnx, tensorrt, or keypoint-quality (default: sets).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all registry views (sets, models, onnx, tensorrt, and keypoint quality).",
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--show-keypoint-quality",
        action="store_true",
        help="Print detailed keypoint quality rows (summary is always shown).",
    )
    parser.add_argument(
        "--hide-unlinked",
        action="store_true",
        help="Hide rows with no matching training_set row (applies to models/onnx/tensorrt views).",
    )
    parser.add_argument("--no-rich", action="store_true", help="Disable rich table output.")
    args = parser.parse_args(argv)

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    show_sets = args.all or args.view == "sets"
    show_models = args.all or args.view == "models"
    show_onnx = args.all or args.view == "onnx"
    show_tensorrt = args.all or args.view == "tensorrt"
    show_keypoint_view = args.view == "keypoint-quality"
    show_keypoint_details = args.show_keypoint_quality or show_keypoint_view

    set_rows = _load_set_rows(registry, args.set_id, args.limit) if show_sets else []
    model_rows = (
        _load_model_rows(
            registry,
            args.set_id,
            args.limit,
            hide_unlinked=args.hide_unlinked,
        )
        if show_models
        else []
    )
    onnx_rows = (
        _load_onnx_rows(
            registry,
            args.set_id,
            args.limit,
            hide_unlinked=args.hide_unlinked,
        )
        if show_onnx
        else []
    )
    trt_rows = (
        _load_tensorrt_rows(
            registry,
            args.set_id,
            args.limit,
            hide_unlinked=args.hide_unlinked,
        )
        if show_tensorrt
        else []
    )
    keypoint_quality_limit = args.limit if (show_keypoint_details or show_keypoint_view) else None
    keypoint_quality_rows = _load_keypoint_quality_rows(
        registry,
        set_filter=args.set_id,
        limit=keypoint_quality_limit,
    )
    keypoint_quality_summary = _summarize_keypoint_quality_rows(keypoint_quality_rows)
    registry.close()

    if not args.all and args.view == "sets" and not set_rows:
        print("No training sets found.")
        return 1
    if not args.all and args.view == "models" and not model_rows:
        print("No training runs found.")
        return 1
    if not args.all and args.view == "onnx" and not onnx_rows:
        print("No ONNX rows found.")
        return 1
    if not args.all and args.view == "tensorrt" and not trt_rows:
        print("No TensorRT rows found.")
        return 1
    if not args.all and args.view == "keypoint-quality" and not keypoint_quality_rows:
        print("No keypoint quality rows found.")
        return 1

    use_rich = not args.no_rich and Console is not None and Table is not None
    if use_rich:
        console = Console()
        if show_sets:
            table = Table(title="Training Sets", show_lines=False)
            table.add_column("Set ID", style="cyan")
            table.add_column("Name", style="magenta")
            table.add_column("Created")
            table.add_column("Datasets", justify="right")
            table.add_column("Latest Training Run")
            table.add_column("Skeleton")
            table.add_column("Manifest")
            table.add_column("Config")
            for row in set_rows:
                table.add_row(
                    row.set_id,
                    row.name or "—",
                    _format_time(row.created_utc),
                    str(row.dataset_count) if row.dataset_count is not None else "—",
                    row.run_id or "—",
                    row.skeleton_id or "—",
                    _status_rich(_path_ok(row.manifest_path)),
                    _status_rich(_path_ok(row.config_path)),
                )
            console.print(table)
        if show_models:
            table = Table(title="Training Runs / Models", show_lines=False)
            table.add_column("Run ID", style="cyan")
            table.add_column("ID Style")
            table.add_column("Set ID")
            table.add_column("Name", style="magenta")
            table.add_column("Status")
            table.add_column("Created")
            table.add_column("Skeleton")
            table.add_column("ROIs", justify="right")
            table.add_column("Model")
            table.add_column("Metrics")
            table.add_column("ONNX")
            table.add_column("TRT")
            for row in model_rows:
                run_id = row.run_id
                status = row.run_status or "—"
                if row.run_status == "failed":
                    run_id = f"[red]{run_id}[/red]"
                    status = "[red]failed[/red]"
                elif row.run_status == "in_progress":
                    run_id = f"[yellow]{run_id}[/yellow]"
                    status = "[yellow]in_progress[/yellow]"
                elif row.run_status == "success":
                    run_id = f"[chartreuse1]{run_id}[/chartreuse1]"
                    status = "[chartreuse1]success[/chartreuse1]"
                table.add_row(
                    run_id,
                    _run_id_style(row.run_id),
                    row.set_id or "—",
                    row.set_name or "—",
                    status,
                    _format_time(row.created_utc),
                    row.skeleton_id or "—",
                    str(row.roi_count) if row.roi_count is not None else "—",
                    _status_rich(_path_ok(row.model_path)),
                    _status_with_details(_path_ok(row.metrics_path), row.metrics_summary, rich=True),
                    _status_with_details(
                        _path_ok(row.onnx_path),
                        _onnx_plugin_details(
                            row.onnx_requires_plugins,
                            row.onnx_plugin_ops_json,
                            row.onnx_plugin_versions_json,
                        ),
                        rich=True,
                    ),
                    _status_rich(_path_ok(row.trt_path)),
                )
            console.print(table)
        if show_onnx:
            table = Table(title="ONNX Models", show_lines=False)
            table.add_column("Run ID", style="cyan")
            table.add_column("Set ID")
            table.add_column("Name", style="magenta")
            table.add_column("From Model")
            table.add_column("Created")
            table.add_column("Skeleton")
            table.add_column("ONNX")
            table.add_column("Shape")
            table.add_column("Plugins")
            table.add_column("Opset")
            table.add_column("Torch/CUDA")
            table.add_column("Host")
            for row in onnx_rows:
                table.add_row(
                    row.run_id,
                    row.set_id or "—",
                    row.set_name or "—",
                    row.detection_model_run_id or row.run_id,
                    _format_time(row.created_utc),
                    row.skeleton_id or "—",
                    _status_rich(_path_ok(row.path)),
                    _shape_summary(
                        img_h=row.img_h,
                        img_w=row.img_w,
                        max_batch=row.max_batch,
                        dynamic_shapes=row.dynamic_shapes,
                    ),
                    _onnx_plugin_details(
                        row.requires_plugins,
                        row.plugin_ops_json,
                        row.plugin_versions_json,
                    )
                    or "—",
                    str(row.opset) if row.opset is not None else "—",
                    f"{row.exporter_torch_version or '—'} / {row.exporter_cuda_version or '—'}",
                    row.exporter_hostname or "—",
                )
            console.print(table)
        if show_tensorrt:
            table = Table(title="TensorRT Models", show_lines=False)
            table.add_column("Run ID", style="cyan")
            table.add_column("Set ID")
            table.add_column("Name", style="magenta")
            table.add_column("From ONNX Run")
            table.add_column("Created")
            table.add_column("Skeleton")
            table.add_column("Engine")
            table.add_column("Shape")
            table.add_column("Plugins")
            table.add_column("Precision")
            table.add_column("TRT/CUDA")
            table.add_column("GPU")
            table.add_column("CC")
            table.add_column("Host")
            table.add_column("ONNX")
            for row in trt_rows:
                table.add_row(
                    row.run_id,
                    row.set_id or "—",
                    row.set_name or "—",
                    row.onnx_run_id or row.run_id,
                    _format_time(row.created_utc),
                    row.skeleton_id or "—",
                    _status_rich(_path_ok(row.path)),
                    _shape_summary(
                        img_h=row.img_h,
                        img_w=row.img_w,
                        max_batch=row.max_batch,
                        dynamic_shapes=row.dynamic_shapes,
                    ),
                    _onnx_plugin_details(
                        row.requires_plugins,
                        row.plugin_ops_json,
                        row.plugin_versions_json,
                    )
                    or "—",
                    row.precision or "—",
                    f"{row.trt_version or '—'} / {row.cuda_version or '—'}",
                    row.gpu_name or "—",
                    row.compute_capability or "—",
                    row.system_hostname or "—",
                    _status_rich(_path_ok(row.onnx_path)),
                )
            console.print(table)
        summary_lines = [
            f"total rows: {keypoint_quality_summary.total_rows}",
            (
                "passing rows "
                f"({KEYPOINT_GATE_REVIEW_STATE}/{KEYPOINT_GATE_REVIEW_INTENDED_USE}, "
                f"usable_rate>={KEYPOINT_GATE_MIN_RATE:.2f}): {keypoint_quality_summary.passing_rows}"
            ),
            f"excluded rows: {keypoint_quality_summary.excluded_rows}",
        ]
        if keypoint_quality_summary.exclusion_reasons:
            reason_text = ", ".join(
                f"{name}={count}"
                for name, count in keypoint_quality_summary.exclusion_reasons.items()
            )
        else:
            reason_text = "none"
        summary_lines.append(f"top exclusion reasons: {reason_text}")
        summary_lines.append(
            "quality_stale note: currently flags only rows missing stored zarr_mtime_ns."
        )
        console.print("[bold]Keypoint Quality[/bold]")
        for line in summary_lines:
            console.print(f"- {line}")
        if show_keypoint_details and keypoint_quality_rows:
            kq_table = Table(title="Keypoint Quality Details", show_lines=False)
            kq_table.add_column("Dataset", style="cyan")
            kq_table.add_column("Purpose")
            kq_table.add_column("Method")
            kq_table.add_column("Review")
            kq_table.add_column("Usable/Total")
            kq_table.add_column("Rate")
            kq_table.add_column("Stale")
            kq_table.add_column("Gate")
            kq_table.add_column("Reason")
            for row in keypoint_quality_rows:
                passes = _keypoint_row_passes_default_gate(row)
                reason = _keypoint_exclusion_reason(row) or "—"
                usable = row.get("usable_keypoints")
                total = row.get("total_keypoints")
                rate = row.get("usable_keypoints_rate")
                review = f"{row.get('review_state') or '—'}/{row.get('review_intended_use') or '—'}"
                kq_table.add_row(
                    str(row.get("dataset_id") or "—"),
                    str(row.get("zarr_purpose") or "—"),
                    str(row.get("keypoint_method") or "—"),
                    review,
                    f"{usable if usable is not None else '—'}/{total if total is not None else '—'}",
                    f"{float(rate):.3f}" if isinstance(rate, (int, float)) else "—",
                    "1" if int(row.get("quality_stale") or 0) == 1 else "0",
                    "[chartreuse1]PASS[/chartreuse1]" if passes else "[red]EXCLUDE[/red]",
                    reason,
                )
            console.print(kq_table)
    else:
        if show_sets:
            for row in set_rows:
                print(row.set_id)
                print(f"  name: {row.name or '—'}")
                print(f"  created: {_format_time(row.created_utc)}")
                print(f"  datasets: {row.dataset_count if row.dataset_count is not None else '—'}")
                print(f"  latest_run: {row.run_id or '—'}")
                print(f"  skeleton: {row.skeleton_id or '—'}")
                print(f"  manifest: {_status_text(_path_ok(row.manifest_path))}")
                print(f"  config: {_status_text(_path_ok(row.config_path))}")
        if show_models:
            for row in model_rows:
                print(row.run_id)
                print(f"  id_style: {_run_id_style(row.run_id)}")
                print(f"  set_id: {row.set_id or '—'}")
                print(f"  name: {row.set_name or '—'}")
                print(f"  status: {row.run_status or '—'}")
                print(f"  created: {_format_time(row.created_utc)}")
                print(f"  skeleton: {row.skeleton_id or '—'}")
                print(f"  rois: {row.roi_count if row.roi_count is not None else '—'}")
                print(f"  model: {_status_text(_path_ok(row.model_path))}")
                print(
                    f"  metrics: {_status_with_details(_path_ok(row.metrics_path), row.metrics_summary, rich=False)}"
                )
                print(
                    "  onnx: "
                    + _status_with_details(
                        _path_ok(row.onnx_path),
                        _onnx_plugin_details(
                            row.onnx_requires_plugins,
                            row.onnx_plugin_ops_json,
                            row.onnx_plugin_versions_json,
                        ),
                        rich=False,
                    )
                )
                print(f"  trt: {_status_text(_path_ok(row.trt_path))}")
        if show_onnx:
            for row in onnx_rows:
                print(row.run_id)
                print(f"  set_id: {row.set_id or '—'}")
                print(f"  name: {row.set_name or '—'}")
                print(f"  from_model: {row.detection_model_run_id or row.run_id}")
                print(f"  created: {_format_time(row.created_utc)}")
                print(f"  skeleton: {row.skeleton_id or '—'}")
                print(f"  onnx: {_status_text(_path_ok(row.path))}")
                print(
                    "  shape: "
                    + _shape_summary(
                        img_h=row.img_h,
                        img_w=row.img_w,
                        max_batch=row.max_batch,
                        dynamic_shapes=row.dynamic_shapes,
                    )
                )
                print(f"  opset: {row.opset if row.opset is not None else '—'}")
                print(
                    "  plugins: "
                    + (
                        _onnx_plugin_details(
                            row.requires_plugins,
                            row.plugin_ops_json,
                            row.plugin_versions_json,
                        )
                        or "—"
                    )
                )
                print(f"  torch/cuda: {row.exporter_torch_version or '—'} / {row.exporter_cuda_version or '—'}")
                print(f"  host: {row.exporter_hostname or '—'}")
        if show_tensorrt:
            for row in trt_rows:
                print(row.run_id)
                print(f"  set_id: {row.set_id or '—'}")
                print(f"  name: {row.set_name or '—'}")
                print(f"  from_onnx_run: {row.onnx_run_id or row.run_id}")
                print(f"  created: {_format_time(row.created_utc)}")
                print(f"  skeleton: {row.skeleton_id or '—'}")
                print(f"  engine: {_status_text(_path_ok(row.path))}")
                print(
                    "  shape: "
                    + _shape_summary(
                        img_h=row.img_h,
                        img_w=row.img_w,
                        max_batch=row.max_batch,
                        dynamic_shapes=row.dynamic_shapes,
                    )
                )
                print(
                    "  plugins: "
                    + (
                        _onnx_plugin_details(
                            row.requires_plugins,
                            row.plugin_ops_json,
                            row.plugin_versions_json,
                        )
                        or "—"
                    )
                )
                print(f"  precision: {row.precision or '—'}")
                print(f"  trt/cuda: {row.trt_version or '—'} / {row.cuda_version or '—'}")
                print(f"  gpu: {row.gpu_name or '—'}")
                print(f"  cc: {row.compute_capability or '—'}")
                print(f"  host: {row.system_hostname or '—'}")
                print(f"  onnx: {_status_text(_path_ok(row.onnx_path))}")
        print("Keypoint Quality")
        print(f"  total rows: {keypoint_quality_summary.total_rows}")
        print(
            "  passing rows "
            f"({KEYPOINT_GATE_REVIEW_STATE}/{KEYPOINT_GATE_REVIEW_INTENDED_USE}, "
            f"usable_rate>={KEYPOINT_GATE_MIN_RATE:.2f}): {keypoint_quality_summary.passing_rows}"
        )
        print(f"  excluded rows: {keypoint_quality_summary.excluded_rows}")
        if keypoint_quality_summary.exclusion_reasons:
            reason_text = ", ".join(
                f"{name}={count}"
                for name, count in keypoint_quality_summary.exclusion_reasons.items()
            )
        else:
            reason_text = "none"
        print(f"  top exclusion reasons: {reason_text}")
        print("  quality_stale note: currently flags only rows missing stored zarr_mtime_ns.")
        if show_keypoint_details and keypoint_quality_rows:
            print("  details:")
            for row in keypoint_quality_rows:
                reason = _keypoint_exclusion_reason(row)
                usable = row.get("usable_keypoints")
                total = row.get("total_keypoints")
                rate = row.get("usable_keypoints_rate")
                print(f"    {row.get('dataset_id')}")
                print(f"      purpose: {row.get('zarr_purpose') or '—'}")
                print(f"      method: {row.get('keypoint_method') or '—'}")
                print(
                    "      review: "
                    f"{row.get('review_state') or '—'}/{row.get('review_intended_use') or '—'}"
                )
                print(f"      usable/total: {usable if usable is not None else '—'}/{total if total is not None else '—'}")
                print(f"      usable_rate: {float(rate):.3f}" if isinstance(rate, (int, float)) else "      usable_rate: —")
                print(f"      quality_stale: {int(row.get('quality_stale') or 0)}")
                print(f"      gate: {'PASS' if reason is None else 'EXCLUDE'}")
                print(f"      reason: {reason or '—'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
