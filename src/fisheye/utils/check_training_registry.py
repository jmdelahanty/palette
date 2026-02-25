#!/usr/bin/env python3
"""Summarize training sets or training runs from the registry."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    nms_conf: Optional[float]
    nms_iou: Optional[float]
    nms_topk: Optional[int]
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
    nms_conf: Optional[float]
    nms_iou: Optional[float]
    nms_topk: Optional[int]
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


@dataclass
class DetectQualitySummary:
    total_rows: int
    passing_rows: int
    excluded_rows: int
    exclusion_reasons: Dict[str, int]


@dataclass
class KeypointProfileSummary:
    total_rows: int
    stale_rows: int
    exclusion_reasons: Dict[str, int]


@dataclass
class EyeMaskPerformanceSummary:
    total_rows: int
    passing_rows: int
    excluded_rows: int
    stale_rows: int
    exclusion_reasons: Dict[str, int]
    review_rollups: Dict[str, int]


@dataclass
class EyeMaskProfileSummary:
    total_rows: int
    stale_rows: int
    exclusion_reasons: Dict[str, int]
    review_rollups: Dict[str, int]


@dataclass
class DatasetRow:
    dataset_id: str
    session_uuid: Optional[str]
    recording_id: Optional[str]
    artifact_kind: Optional[str]
    status: Optional[str]
    zarr_path: Optional[str]
    zarr_origin: Optional[str]
    zarr_use: Optional[str]
    rig_id: Optional[str]
    arena_id: Optional[str]
    camera_id: Optional[str]
    parent_count: int
    child_count: int
    last_seen_utc: Optional[str]


@dataclass
class RecordingRow:
    recording_id: str
    session_uuid: Optional[str]
    recording_name: Optional[str]
    recording_path: Optional[str]
    started_utc: Optional[str]
    recording_type: Optional[str]
    recording_subtype: Optional[str]
    behavior_mode: Optional[str]
    artifact_schema_id: Optional[str]
    rig_id: Optional[str]
    arena_id: Optional[str]
    camera_id: Optional[str]
    canvas_name: Optional[str]
    protocol_name: Optional[str]
    dish_design: Optional[str]
    updated_utc: Optional[str]
    artifact_count: int
    missing_required_artifacts: List[str]


@dataclass
class RecordingSummaryRow:
    recording_type: Optional[str]
    recording_subtype: Optional[str]
    count: int


@dataclass
class RecordingOverviewRow:
    recording_id: str
    recording_type: Optional[str]
    recording_subtype: Optional[str]
    behavior_mode: Optional[str]
    dish_design: Optional[str]
    rig_id: Optional[str]
    arena_id: Optional[str]
    camera_id: Optional[str]
    dataset_count: int
    training_dataset_count: int
    analysis_dataset_count: int
    inference_dataset_count: int
    export_dataset_count: int
    active_dataset_count: int
    missing_dataset_count: int
    last_seen_utc: Optional[str]


@dataclass
class RecordingStepOverviewRow:
    recording_id: str
    dataset_count: int
    step_rows_total: int
    ok_rows: int
    missing_rows: int
    absent_rows: int
    na_rows: int
    error_rows: int
    blocking_steps_csv: Optional[str]
    latest_step_update_utc: Optional[str]


@dataclass
class RecordingStepDetailRow:
    recording_id: Optional[str]
    dataset_id: str
    zarr_use: Optional[str]
    step_name: str
    status: str
    run_name: Optional[str]
    method: Optional[str]
    coverage_pct: Optional[float]
    source: Optional[str]
    updated_utc: Optional[str]


@dataclass
class RecordingVocabRow:
    recording_type: str
    recording_subtype: str


@dataclass
class DatasetLineageRow:
    child_dataset_id: str
    parent_dataset_id: str
    relationship_type: str
    source_set_id: Optional[str]
    updated_utc: Optional[str]


@dataclass
class DatasetLineageSummary:
    edge_count: int
    merged_dataset_count: int
    merged_missing_lineage_count: int
    set_lineage_mismatch_count: int
    relationship_type_counts: Dict[str, int]


KEYPOINT_GATE_REVIEW_STATE = "approved"
KEYPOINT_GATE_REVIEW_INTENDED_USE = "training"
KEYPOINT_GATE_MIN_RATE = 0.70
DETECT_GATE_REVIEW_STATE = "approved"
DETECT_GATE_REVIEW_INTENDED_USE = "training"
DETECT_GATE_MAX_INTERPOLATED_RATE = 0.25
KEYPOINT_PROFILE_METHOD_KEYS = ("keypoint_method", "method")
EYE_MASK_GATE_REVIEW_STATE = "approved"
EYE_MASK_GATE_REVIEW_INTENDED_USE = "training"
EYE_MASK_PROFILE_METHOD_KEYS = ("eye_mask_method", "method", "source_eye_masks_method")
EYE_MASK_PROFILE_STALE_STATE_KEYS = (
    "source_keypoint_stale_state",
    "profile_stale_state",
    "stale_state",
)
EYE_MASK_SYNC_COMMAND = "scripts/py -m fisheye.utils.sync_eye_mask_profile_registry"
EYE_MASK_REFRESH_COMMAND = "scripts/py -m fisheye.registry.maintenance --refresh-eye-mask-profiles"
BEHAVIOR_V1_REQUIRED_ARTIFACT_TYPES = {
    "h5_log",
    "camera_video",
    "camera_metadata_csv",
    "timing_profile_csv",
}


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


def _step_status_rich(status: Optional[str]) -> str:
    normalized = str(status or "").strip().lower()
    if normalized == "ok":
        return "[chartreuse1]ok[/chartreuse1]"
    if normalized in {"missing", "error"}:
        return f"[red]{normalized}[/red]"
    if normalized in {"absent", "na"}:
        return f"[yellow]{normalized}[/yellow]"
    return normalized or "—"


def _step_status_text(status: Optional[str]) -> str:
    normalized = str(status or "").strip().lower()
    return normalized or "—"


_RATIO_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")
_PCT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)%\s*(?:\(|$)")


def _wide_cell_rich(value: Optional[str]) -> str:
    text = str(value or "—").strip() or "—"
    upper = text.upper()
    if text == "—":
        return text
    if upper == "OK":
        return "[chartreuse1]OK[/chartreuse1]"
    if upper == "MISS":
        return "[red]MISS[/red]"
    if upper in {"N/A", "NA"}:
        return "[yellow]N/A[/yellow]"
    if upper.startswith("OK"):
        return f"[chartreuse1]{text}[/chartreuse1]"
    if upper.startswith("APPROVED"):
        return f"[chartreuse1]{text}[/chartreuse1]"
    if upper.startswith("FAILED") or upper.startswith("ERROR"):
        return f"[red]{text}[/red]"
    if upper.startswith("PENDING") or upper.startswith("REVIEW"):
        return f"[yellow]{text}[/yellow]"
    ratio_match = _RATIO_RE.fullmatch(text)
    if ratio_match:
        ok_count = int(ratio_match.group(1))
        total_count = int(ratio_match.group(2))
        if total_count <= 0:
            return text
        if ok_count == total_count:
            return f"[chartreuse1]{text}[/chartreuse1]"
        if ok_count == 0:
            return f"[red]{text}[/red]"
        return f"[yellow]{text}[/yellow]"
    pct_match = _PCT_RE.match(text)
    if pct_match:
        pct = float(pct_match.group(1))
        if pct >= 99.9:
            return f"[chartreuse1]{text}[/chartreuse1]"
        return f"[yellow]{text}[/yellow]"
    if "(MISS)" in upper:
        return f"[red]{text}[/red]"
    if "(OK)" in upper:
        return f"[chartreuse1]{text}[/chartreuse1]"
    return text


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


def _json_text_list(payload: Optional[str]) -> List[str]:
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [str(item) for item in data if item is not None]


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
        LEFT JOIN training_models dm ON dm.run_id = tr.run_id
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


def _nms_summary(
    *,
    nms_conf: Optional[float],
    nms_iou: Optional[float],
    nms_topk: Optional[int],
) -> str:
    if nms_conf is None and nms_iou is None and nms_topk is None:
        return "—"
    conf = f"{float(nms_conf):.3f}" if nms_conf is not None else "—"
    iou = f"{float(nms_iou):.3f}" if nms_iou is not None else "—"
    topk = str(int(nms_topk)) if nms_topk is not None else "—"
    return f"c={conf} i={iou} k={topk}"


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
        "om.nms_conf,",
        "om.nms_iou,",
        "om.nms_topk,",
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
            nms_conf=row["nms_conf"],
            nms_iou=row["nms_iou"],
            nms_topk=row["nms_topk"],
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
        "tm.nms_conf,",
        "tm.nms_iou,",
        "tm.nms_topk,",
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
            nms_conf=row["nms_conf"],
            nms_iou=row["nms_iou"],
            nms_topk=row["nms_topk"],
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
        "LEFT JOIN training_models dm ON dm.run_id = tr.run_id",
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


def _load_dataset_rows(
    registry: Registry,
    *,
    set_filter: Optional[str],
    limit: Optional[int] = None,
) -> List[DatasetRow]:
    sql = [
        "SELECT",
        "d.dataset_id,",
        "d.session_uuid,",
        "d.recording_id,",
        "d.artifact_kind,",
        "d.status,",
        "d.zarr_path,",
        "d.zarr_origin,",
        "d.zarr_use,",
        "d.last_seen_utc,",
        "p.rig_id,",
        "p.arena_id,",
        "p.camera_id,",
        "COALESCE(lp.parent_count, 0) AS parent_count,",
        "COALESCE(lc.child_count, 0) AS child_count",
        "FROM datasets d",
        "LEFT JOIN provenance p ON p.dataset_id = d.dataset_id",
        "LEFT JOIN (",
        "  SELECT child_dataset_id, COUNT(*) AS parent_count",
        "  FROM dataset_lineage_current",
        "  GROUP BY child_dataset_id",
        ") lp ON lp.child_dataset_id = d.dataset_id",
        "LEFT JOIN (",
        "  SELECT parent_dataset_id, COUNT(*) AS child_count",
        "  FROM dataset_lineage_current",
        "  GROUP BY parent_dataset_id",
        ") lc ON lc.parent_dataset_id = d.dataset_id",
        "WHERE 1=1",
    ]
    params: List[Any] = []
    dataset_ids = _dataset_ids_for_set(registry, set_filter)
    if dataset_ids is not None:
        if not dataset_ids:
            return []
        placeholders = ", ".join("?" for _ in dataset_ids)
        sql.append(f"AND d.dataset_id IN ({placeholders})")
        params.extend(dataset_ids)
    sql.append("ORDER BY COALESCE(d.last_seen_utc, '') DESC, d.dataset_id ASC")
    if limit and limit > 0:
        sql.append("LIMIT ?")
        params.append(int(limit))
    rows = registry.conn.execute(" ".join(sql), params).fetchall()
    return [
        DatasetRow(
            dataset_id=row["dataset_id"],
            session_uuid=row["session_uuid"],
            recording_id=row["recording_id"],
            artifact_kind=row["artifact_kind"],
            status=row["status"],
            zarr_path=row["zarr_path"],
            zarr_origin=row["zarr_origin"],
            zarr_use=row["zarr_use"],
            rig_id=row["rig_id"],
            arena_id=row["arena_id"],
            camera_id=row["camera_id"],
            parent_count=int(row["parent_count"] or 0),
            child_count=int(row["child_count"] or 0),
            last_seen_utc=row["last_seen_utc"],
        )
        for row in rows
    ]


def _load_dataset_lineage_rows(
    registry: Registry,
    *,
    set_filter: Optional[str],
    limit: Optional[int] = None,
) -> List[DatasetLineageRow]:
    sql = [
        "SELECT",
        "child_dataset_id,",
        "parent_dataset_id,",
        "relationship_type,",
        "source_set_id,",
        "updated_utc",
        "FROM dataset_lineage_current",
        "WHERE 1=1",
    ]
    params: List[Any] = []
    dataset_ids = _dataset_ids_for_set(registry, set_filter)
    if dataset_ids is not None:
        if not dataset_ids:
            return []
        placeholders = ", ".join("?" for _ in dataset_ids)
        sql.append(
            f"AND (child_dataset_id IN ({placeholders}) OR parent_dataset_id IN ({placeholders}))"
        )
        params.extend(dataset_ids)
        params.extend(dataset_ids)
    sql.append("ORDER BY child_dataset_id, relationship_type, parent_dataset_id")
    if limit and limit > 0:
        sql.append("LIMIT ?")
        params.append(int(limit))
    rows = registry.conn.execute(" ".join(sql), params).fetchall()
    return [
        DatasetLineageRow(
            child_dataset_id=str(row["child_dataset_id"]),
            parent_dataset_id=str(row["parent_dataset_id"]),
            relationship_type=str(row["relationship_type"]),
            source_set_id=row["source_set_id"],
            updated_utc=row["updated_utc"],
        )
        for row in rows
    ]


def _summarize_dataset_lineage(
    registry: Registry,
    *,
    set_filter: Optional[str],
) -> DatasetLineageSummary:
    dataset_ids = _dataset_ids_for_set(registry, set_filter)
    params: List[Any] = []
    if dataset_ids is None:
        edge_count_sql = "SELECT COUNT(*) FROM dataset_lineage_current;"
        rel_count_sql = (
            "SELECT relationship_type, COUNT(*) AS n "
            "FROM dataset_lineage_current GROUP BY relationship_type ORDER BY n DESC, relationship_type ASC;"
        )
        merged_count_sql = (
            "SELECT COUNT(*) FROM datasets WHERE artifact_kind='derived_training_merge' OR dataset_id LIKE '%_merged';"
        )
        merged_missing_sql = (
            "SELECT COUNT(*) "
            "FROM datasets d "
            "WHERE (d.artifact_kind='derived_training_merge' OR d.dataset_id LIKE '%_merged') "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM dataset_lineage_current dl "
            "  WHERE dl.child_dataset_id=d.dataset_id "
            "    AND dl.relationship_type='training_merge_source'"
            ");"
        )
    else:
        if not dataset_ids:
            return DatasetLineageSummary(
                edge_count=0,
                merged_dataset_count=0,
                merged_missing_lineage_count=0,
                set_lineage_mismatch_count=0,
                relationship_type_counts={},
            )
        placeholders = ", ".join("?" for _ in dataset_ids)
        edge_count_sql = (
            "SELECT COUNT(*) "
            "FROM dataset_lineage_current "
            f"WHERE child_dataset_id IN ({placeholders}) OR parent_dataset_id IN ({placeholders});"
        )
        rel_count_sql = (
            "SELECT relationship_type, COUNT(*) AS n "
            "FROM dataset_lineage_current "
            f"WHERE child_dataset_id IN ({placeholders}) OR parent_dataset_id IN ({placeholders}) "
            "GROUP BY relationship_type ORDER BY n DESC, relationship_type ASC;"
        )
        merged_count_sql = (
            "SELECT COUNT(*) FROM datasets "
            f"WHERE dataset_id IN ({placeholders}) "
            "AND (artifact_kind='derived_training_merge' OR dataset_id LIKE '%_merged');"
        )
        merged_missing_sql = (
            "SELECT COUNT(*) "
            "FROM datasets d "
            f"WHERE d.dataset_id IN ({placeholders}) "
            "AND (d.artifact_kind='derived_training_merge' OR d.dataset_id LIKE '%_merged') "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM dataset_lineage_current dl "
            "  WHERE dl.child_dataset_id=d.dataset_id "
            "    AND dl.relationship_type='training_merge_source'"
            ");"
        )
        params = list(dataset_ids)

    if dataset_ids is None:
        edge_count = int(registry.conn.execute(edge_count_sql).fetchone()[0])
        rel_rows = registry.conn.execute(rel_count_sql).fetchall()
        merged_dataset_count = int(registry.conn.execute(merged_count_sql).fetchone()[0])
        merged_missing_lineage_count = int(registry.conn.execute(merged_missing_sql).fetchone()[0])
    else:
        edge_count = int(registry.conn.execute(edge_count_sql, params + params).fetchone()[0])
        rel_rows = registry.conn.execute(rel_count_sql, params + params).fetchall()
        merged_dataset_count = int(registry.conn.execute(merged_count_sql, params).fetchone()[0])
        merged_missing_lineage_count = int(registry.conn.execute(merged_missing_sql, params).fetchone()[0])

    rel_counts: Dict[str, int] = {}
    for row in rel_rows:
        rel = str(row["relationship_type"] or "").strip()
        if not rel:
            rel = "—"
        rel_counts[rel] = int(row["n"] or 0)

    set_lineage_mismatch_count = 0
    set_rows = registry.conn.execute(
        "SELECT set_id, dataset_ids_json FROM training_sets;"
    ).fetchall()

    dataset_rows = registry.conn.execute(
        "SELECT dataset_id, artifact_kind FROM datasets;"
    ).fetchall()
    dataset_kind: Dict[str, str] = {
        str(row["dataset_id"]): str(row["artifact_kind"] or "")
        for row in dataset_rows
        if row["dataset_id"] is not None
    }
    dataset_scope = set(dataset_ids) if dataset_ids is not None else None
    for row in set_rows:
        ids = sorted(set(_json_text_list(row["dataset_ids_json"])))
        if dataset_scope is not None:
            ids = [item for item in ids if item in dataset_scope]
        merged_ids = [
            dataset_id
            for dataset_id in ids
            if dataset_id in dataset_kind
            and (dataset_kind.get(dataset_id) == "derived_training_merge" or dataset_id.endswith("_merged"))
        ]
        if not merged_ids:
            continue
        expected_parents = {
            dataset_id
            for dataset_id in ids
            if dataset_id in dataset_kind and dataset_id not in merged_ids
        }
        for child_dataset_id in merged_ids:
            expected = {item for item in expected_parents if item != child_dataset_id}
            if not expected:
                continue
            actual_rows = registry.conn.execute(
                """
                SELECT parent_dataset_id
                FROM dataset_lineage_current
                WHERE child_dataset_id = ? AND relationship_type = 'training_merge_source';
                """,
                (child_dataset_id,),
            ).fetchall()
            actual = {
                str(item["parent_dataset_id"])
                for item in actual_rows
                if item["parent_dataset_id"] is not None
            }
            if actual != expected:
                set_lineage_mismatch_count += 1

    return DatasetLineageSummary(
        edge_count=edge_count,
        merged_dataset_count=merged_dataset_count,
        merged_missing_lineage_count=merged_missing_lineage_count,
        set_lineage_mismatch_count=set_lineage_mismatch_count,
        relationship_type_counts=rel_counts,
    )


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


def _load_keypoint_profile_rows(
    registry: Registry,
    *,
    set_filter: Optional[str],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    dataset_ids = _dataset_ids_for_set(registry, set_filter)
    if dataset_ids is not None and not dataset_ids:
        return []

    rows: List[Dict[str, Any]] = []
    query_latest = getattr(registry, "query_keypoint_data_profile_latest", None)
    if callable(query_latest):
        try:
            if dataset_ids is None:
                raw_rows = query_latest()
            else:
                raw_rows = query_latest(dataset_ids=dataset_ids)
        except TypeError:
            # Defensive fallback for older/alternate query signatures.
            try:
                raw_rows = query_latest()
            except Exception:
                raw_rows = []
        except Exception:
            raw_rows = []
        rows = [dict(row) for row in raw_rows]
    elif _view_exists(registry, "keypoint_data_profile_latest"):
        # SQL fallback keeps this report usable even if helper API is absent.
        sql = ["SELECT * FROM keypoint_data_profile_latest WHERE 1=1"]
        params: List[Any] = []
        if dataset_ids is not None:
            placeholders = ", ".join("?" for _ in dataset_ids)
            sql.append(f"AND dataset_id IN ({placeholders})")
            params.extend(dataset_ids)
        sql.append("ORDER BY dataset_id, COALESCE(keypoint_method, ''), COALESCE(profile_run, '')")
        if limit and limit > 0:
            sql.append("LIMIT ?")
            params.append(int(limit))
        rows = [dict(row) for row in registry.conn.execute(" ".join(sql), params).fetchall()]
    else:
        return []

    dataset_ids_in_rows = sorted(
        {
            str(row.get("dataset_id")).strip()
            for row in rows
            if str(row.get("dataset_id") or "").strip()
        }
    )
    if dataset_ids_in_rows:
        placeholders = ", ".join("?" for _ in dataset_ids_in_rows)
        dataset_sql = (
            "SELECT dataset_id, zarr_path, zarr_use FROM datasets "
            f"WHERE dataset_id IN ({placeholders})"
        )
        dataset_rows = registry.conn.execute(dataset_sql, dataset_ids_in_rows).fetchall()
        dataset_lookup: Dict[str, Dict[str, Any]] = {
            str(row["dataset_id"]): dict(row)
            for row in dataset_rows
            if row["dataset_id"] is not None
        }
        for row in rows:
            dataset_id = str(row.get("dataset_id") or "").strip()
            if not dataset_id:
                continue
            dataset_payload = dataset_lookup.get(dataset_id)
            if not dataset_payload:
                continue
            if not str(row.get("zarr_path") or "").strip():
                row["zarr_path"] = dataset_payload.get("zarr_path")
            if not str(row.get("zarr_use") or "").strip():
                row["zarr_use"] = dataset_payload.get("zarr_use")

    rows = sorted(
        rows,
        key=lambda row: (
            str(row.get("dataset_id") or ""),
            str(row.get("keypoint_method") or row.get("method") or ""),
            str(row.get("profile_run") or ""),
        ),
    )
    if limit and limit > 0:
        rows = rows[: int(limit)]
    return rows


def _load_eye_mask_performance_rows(
    registry: Registry,
    *,
    set_filter: Optional[str],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    params: List[Any] = []
    dataset_ids = _dataset_ids_for_set(registry, set_filter)
    if dataset_ids is not None:
        if not dataset_ids:
            return []
    if _view_exists(registry, "eye_mask_quality_overview"):
        sql = [
            "SELECT",
            "  emqo.dataset_id AS dataset_id,",
            "  emqo.zarr_path AS zarr_path,",
            "  emqo.zarr_use AS zarr_use,",
            "  emqo.stage_group AS stage_group,",
            "  emqo.run_name AS run_name,",
            "  emqo.run_created_utc AS run_created_utc,",
            "  emqo.recording_id AS recording_id,",
            "  emqo.eye_mask_method AS method,",
            "  emqo.source_crop_run AS source_crop_run,",
            "  emqo.source_keypoint_group AS source_keypoint_group,",
            "  emqo.source_keypoints_run AS source_keypoints_run,",
            "  emqo.source_eye_masks_run AS source_eye_masks_run,",
            "  emqo.source_eye_masks_method AS source_eye_masks_method,",
            "  emqo.total_rois AS total_rois,",
            "  emqo.successful_eyes AS successful_eyes,",
            "  emqo.successful_roi_pairs AS successful_roi_pairs,",
            "  emqo.successful_roi_pair_rate AS successful_roi_pair_rate,",
            "  emp.duration_seconds AS duration_seconds,",
            "  emp.rois_per_second AS rois_per_second,",
            "  emp.inference_duration_seconds AS inference_duration_seconds,",
            "  emp.inference_average_fps AS inference_average_fps,",
            "  emqo.review_state AS review_state,",
            "  emqo.review_method AS review_method,",
            "  emqo.review_intended_use AS review_intended_use,",
            "  emqo.review_reviewer AS review_reviewer,",
            "  emqo.review_timestamp_utc AS review_timestamp_utc,",
            "  emqo.source_keypoint_stale_state AS source_keypoint_stale_state,",
            "  emqo.source_keypoint_stale_reason AS source_keypoint_stale_reason,",
            "  emqo.source_keypoint_stale_timestamp_utc AS source_keypoint_stale_timestamp_utc,",
            "  emqo.lifecycle_state AS lifecycle_state,",
            "  emqo.lifecycle_reason AS lifecycle_reason,",
            "  emqo.zarr_mtime_ns AS zarr_mtime_ns,",
            "  emqo.quality_updated_utc AS quality_updated_utc",
            "FROM eye_mask_quality_overview emqo",
            "LEFT JOIN eye_mask_performance emp",
            "  ON emp.dataset_id = emqo.dataset_id",
            " AND emp.stage_group = emqo.stage_group",
            " AND emp.run_name = emqo.run_name",
            "WHERE 1=1",
        ]
        quality_params: List[Any] = []
        if dataset_ids is not None:
            placeholders = ", ".join("?" for _ in dataset_ids)
            sql.append(f"AND emqo.dataset_id IN ({placeholders})")
            quality_params.extend(dataset_ids)
        sql.append("ORDER BY emqo.dataset_id, COALESCE(emqo.stage_group, ''), COALESCE(emqo.run_created_utc, '')")
        if limit and limit > 0:
            sql.append("LIMIT ?")
            quality_params.append(int(limit))
        quality_rows = registry.conn.execute(" ".join(sql), quality_params).fetchall()
        if quality_rows:
            return [dict(row) for row in quality_rows]

    if not _view_exists(registry, "eye_mask_performance_latest"):
        return []
    sql = [
        "SELECT",
        "  empl.dataset_id AS dataset_id,",
        "  d.zarr_path AS zarr_path,",
        "  d.zarr_use AS zarr_use,",
        "  empl.stage_group AS stage_group,",
        "  empl.run_name AS run_name,",
        "  empl.run_created_utc AS run_created_utc,",
        "  empl.recording_id AS recording_id,",
        "  empl.method AS method,",
        "  empl.source_crop_run AS source_crop_run,",
        "  empl.source_keypoint_group AS source_keypoint_group,",
        "  empl.source_keypoints_run AS source_keypoints_run,",
        "  empl.source_eye_masks_run AS source_eye_masks_run,",
        "  empl.source_eye_masks_method AS source_eye_masks_method,",
        "  empl.total_rois AS total_rois,",
        "  empl.successful_eyes AS successful_eyes,",
        "  empl.successful_roi_pairs AS successful_roi_pairs,",
        "  empl.successful_roi_pair_rate AS successful_roi_pair_rate,",
        "  empl.duration_seconds AS duration_seconds,",
        "  empl.rois_per_second AS rois_per_second,",
        "  empl.inference_duration_seconds AS inference_duration_seconds,",
        "  empl.inference_average_fps AS inference_average_fps,",
        "  empl.review_state AS review_state,",
        "  empl.review_method AS review_method,",
        "  empl.review_intended_use AS review_intended_use,",
        "  empl.review_reviewer AS review_reviewer,",
        "  empl.review_timestamp_utc AS review_timestamp_utc,",
        "  empl.source_keypoint_stale_state AS source_keypoint_stale_state,",
        "  empl.source_keypoint_stale_reason AS source_keypoint_stale_reason,",
        "  empl.source_keypoint_stale_timestamp_utc AS source_keypoint_stale_timestamp_utc,",
        "  empl.lifecycle_state AS lifecycle_state,",
        "  empl.lifecycle_reason AS lifecycle_reason,",
        "  empl.zarr_mtime_ns AS zarr_mtime_ns",
        "FROM eye_mask_performance_latest empl",
        "LEFT JOIN datasets d ON d.dataset_id = empl.dataset_id",
        "WHERE 1=1",
    ]
    if dataset_ids is not None:
        placeholders = ", ".join("?" for _ in dataset_ids)
        sql.append(f"AND empl.dataset_id IN ({placeholders})")
        params.extend(dataset_ids)
    sql.append("ORDER BY empl.dataset_id, COALESCE(empl.stage_group, ''), COALESCE(empl.run_created_utc, '')")
    if limit and limit > 0:
        sql.append("LIMIT ?")
        params.append(int(limit))
    rows = registry.conn.execute(" ".join(sql), params).fetchall()
    return [dict(row) for row in rows]


def _load_eye_mask_profile_rows(
    registry: Registry,
    *,
    set_filter: Optional[str],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    dataset_ids = _dataset_ids_for_set(registry, set_filter)
    if dataset_ids is not None and not dataset_ids:
        return []

    rows: List[Dict[str, Any]] = []
    query_latest = getattr(registry, "query_eye_mask_data_profile_latest", None)
    if callable(query_latest):
        try:
            if dataset_ids is None:
                raw_rows = query_latest()
            else:
                raw_rows = query_latest(dataset_ids=dataset_ids)
        except TypeError:
            try:
                raw_rows = query_latest()
            except Exception:
                raw_rows = []
        except Exception:
            raw_rows = []
        rows = [dict(row) for row in raw_rows]
    if not rows and _view_exists(registry, "eye_mask_data_profile_latest"):
        sql = ["SELECT * FROM eye_mask_data_profile_latest WHERE 1=1"]
        params: List[Any] = []
        if dataset_ids is not None:
            placeholders = ", ".join("?" for _ in dataset_ids)
            sql.append(f"AND dataset_id IN ({placeholders})")
            params.extend(dataset_ids)
        sql.append(
            "ORDER BY dataset_id, COALESCE(eye_mask_method, COALESCE(method, '')), COALESCE(profile_run, '')"
        )
        if limit and limit > 0:
            sql.append("LIMIT ?")
            params.append(int(limit))
        try:
            rows = [dict(row) for row in registry.conn.execute(" ".join(sql), params).fetchall()]
        except Exception:
            fallback_sql = ["SELECT * FROM eye_mask_data_profile_latest WHERE 1=1"]
            fallback_params: List[Any] = []
            if dataset_ids is not None:
                placeholders = ", ".join("?" for _ in dataset_ids)
                fallback_sql.append(f"AND dataset_id IN ({placeholders})")
                fallback_params.extend(dataset_ids)
            fallback_sql.append("ORDER BY dataset_id")
            if limit and limit > 0:
                fallback_sql.append("LIMIT ?")
                fallback_params.append(int(limit))
            rows = [dict(row) for row in registry.conn.execute(" ".join(fallback_sql), fallback_params).fetchall()]
    if not rows:
        return []

    dataset_ids_in_rows = sorted(
        {
            str(row.get("dataset_id")).strip()
            for row in rows
            if str(row.get("dataset_id") or "").strip()
        }
    )
    if dataset_ids_in_rows:
        placeholders = ", ".join("?" for _ in dataset_ids_in_rows)
        dataset_sql = (
            "SELECT dataset_id, zarr_path, zarr_use FROM datasets "
            f"WHERE dataset_id IN ({placeholders})"
        )
        dataset_rows = registry.conn.execute(dataset_sql, dataset_ids_in_rows).fetchall()
        dataset_lookup: Dict[str, Dict[str, Any]] = {
            str(row["dataset_id"]): dict(row)
            for row in dataset_rows
            if row["dataset_id"] is not None
        }
        for row in rows:
            dataset_id = str(row.get("dataset_id") or "").strip()
            if not dataset_id:
                continue
            dataset_payload = dataset_lookup.get(dataset_id)
            if not dataset_payload:
                continue
            if not str(row.get("zarr_path") or "").strip():
                row["zarr_path"] = dataset_payload.get("zarr_path")
            if not str(row.get("zarr_use") or "").strip():
                row["zarr_use"] = dataset_payload.get("zarr_use")

    rows = sorted(
        rows,
        key=lambda row: (
            str(row.get("dataset_id") or ""),
            str(row.get("eye_mask_method") or row.get("method") or row.get("source_eye_masks_method") or ""),
            str(row.get("profile_run") or row.get("run_name") or ""),
        ),
    )
    if limit and limit > 0:
        rows = rows[: int(limit)]
    return rows


def _load_detect_quality_rows(
    registry: Registry,
    *,
    set_filter: Optional[str],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not _view_exists(registry, "detect_quality_current"):
        return []
    sql = [
        "SELECT",
        "  dqc.dataset_id AS dataset_id,",
        "  d.zarr_path AS zarr_path,",
        "  d.zarr_origin AS zarr_origin,",
        "  d.zarr_use AS zarr_use,",
        "  dqc.detect_method AS detect_method,",
        "  dqc.refined_run AS refined_run,",
        "  dqc.source_detect_run AS source_detect_run,",
        "  dqc.review_state AS review_state,",
        "  dqc.review_intended_use AS review_intended_use,",
        "  dqc.review_resolved_group AS review_resolved_group,",
        "  dqc.total_detections AS total_detections,",
        "  dqc.real_detections AS real_detections,",
        "  dqc.interpolated_detections AS interpolated_detections,",
        "  dqc.interpolated_detections_rate AS interpolated_detections_rate,",
        "  dqc.review_timestamp_utc AS review_timestamp_utc,",
        "  dqc.quality_updated_utc AS quality_updated_utc,",
        "  dqc.zarr_mtime_ns AS zarr_mtime_ns",
        "FROM detect_quality_current dqc",
        "LEFT JOIN datasets d ON d.dataset_id = dqc.dataset_id",
        "WHERE 1=1",
    ]
    params: List[Any] = []
    dataset_ids = _dataset_ids_for_set(registry, set_filter)
    if dataset_ids is not None:
        if not dataset_ids:
            return []
        placeholders = ", ".join("?" for _ in dataset_ids)
        sql.append(f"AND dqc.dataset_id IN ({placeholders})")
        params.extend(dataset_ids)
    sql.append("ORDER BY dqc.dataset_id, COALESCE(dqc.detect_method, ''), dqc.refined_run")
    if limit and limit > 0:
        sql.append("LIMIT ?")
        params.append(int(limit))
    rows = registry.conn.execute(" ".join(sql), params).fetchall()
    return [dict(row) for row in rows]


def _load_recording_rows(
    registry: Registry,
    *,
    set_filter: Optional[str],
    limit: Optional[int] = None,
) -> List[RecordingRow]:
    sql = [
        "SELECT",
        "r.recording_id,",
        "r.session_uuid,",
        "r.recording_name,",
        "r.recording_path,",
        "r.started_utc,",
        "r.recording_type,",
        "r.recording_subtype,",
        "r.behavior_mode,",
        "r.artifact_schema_id,",
        "r.rig_id,",
        "r.arena_id,",
        "r.camera_id,",
        "r.canvas_name,",
        "r.protocol_name,",
        "COALESCE(NULLIF(TRIM(r.dish_design), ''), COALESCE(pd.dish_design_csv, '')) AS dish_design_csv,",
        "r.updated_utc,",
        "COALESCE(a.artifact_count, 0) AS artifact_count,",
        "COALESCE(a.artifact_types_csv, '') AS artifact_types_csv",
        "FROM recordings r",
        "LEFT JOIN (",
        "  SELECT",
        "    d.recording_id,",
        "    GROUP_CONCAT(DISTINCT NULLIF(TRIM(p.dish_design), '')) AS dish_design_csv",
        "  FROM datasets d",
        "  LEFT JOIN provenance p ON p.dataset_id = d.dataset_id",
        "  GROUP BY d.recording_id",
        ") pd ON pd.recording_id = r.recording_id",
        "LEFT JOIN (",
        "  SELECT",
        "    recording_id,",
        "    COUNT(*) AS artifact_count,",
        "    GROUP_CONCAT(DISTINCT artifact_type) AS artifact_types_csv",
        "  FROM recording_artifacts",
        "  GROUP BY recording_id",
        ") a ON a.recording_id = r.recording_id",
        "WHERE 1=1",
    ]
    params: List[Any] = []
    dataset_ids = _dataset_ids_for_set(registry, set_filter)
    if dataset_ids is not None:
        if not dataset_ids:
            return []
        placeholders = ", ".join("?" for _ in dataset_ids)
        sql.append(
            "AND r.recording_id IN ("
            "SELECT DISTINCT recording_id FROM datasets "
            f"WHERE dataset_id IN ({placeholders}) AND recording_id IS NOT NULL AND TRIM(recording_id) <> ''"
            ")"
        )
        params.extend(dataset_ids)
    sql.append(
        "ORDER BY COALESCE(r.started_utc, r.updated_utc, r.created_utc, '') DESC, r.recording_id DESC"
    )
    if limit and limit > 0:
        sql.append("LIMIT ?")
        params.append(int(limit))
    rows = registry.conn.execute(" ".join(sql), params).fetchall()

    result: List[RecordingRow] = []
    for row in rows:
        artifact_types = {
            item.strip()
            for item in str(row["artifact_types_csv"] or "").split(",")
            if item and item.strip()
        }
        missing_required: List[str] = []
        if row["artifact_schema_id"] == "behavior_v1":
            missing_required = sorted(BEHAVIOR_V1_REQUIRED_ARTIFACT_TYPES - artifact_types)
        result.append(
            RecordingRow(
                recording_id=row["recording_id"],
                session_uuid=row["session_uuid"],
                recording_name=row["recording_name"],
                recording_path=row["recording_path"],
                started_utc=row["started_utc"],
                recording_type=row["recording_type"],
                recording_subtype=row["recording_subtype"],
                behavior_mode=row["behavior_mode"],
                artifact_schema_id=row["artifact_schema_id"],
                rig_id=row["rig_id"],
                arena_id=row["arena_id"],
                camera_id=row["camera_id"],
                canvas_name=row["canvas_name"],
                protocol_name=row["protocol_name"],
                dish_design=row["dish_design_csv"] or None,
                updated_utc=row["updated_utc"],
                artifact_count=int(row["artifact_count"] or 0),
                missing_required_artifacts=missing_required,
            )
        )
    return result


def _load_recording_summary_rows(registry: Registry) -> List[RecordingSummaryRow]:
    rows = registry.conn.execute(
        """
        SELECT
            recording_type,
            recording_subtype,
            COUNT(*) AS n
        FROM recordings
        GROUP BY recording_type, recording_subtype
        ORDER BY recording_type, recording_subtype;
        """
    ).fetchall()
    return [
        RecordingSummaryRow(
            recording_type=row["recording_type"],
            recording_subtype=row["recording_subtype"],
            count=int(row["n"] or 0),
        )
        for row in rows
    ]


def _load_recording_overview_rows(
    registry: Registry,
    *,
    set_filter: Optional[str],
    limit: Optional[int] = None,
) -> List[RecordingOverviewRow]:
    overview_columns = {
        str(row["name"])
        for row in registry.conn.execute("PRAGMA table_info(recording_overview);").fetchall()
        if row["name"] is not None
    }
    has_dish_design = "dish_design" in overview_columns
    sql = [
        "SELECT",
        "recording_id,",
        "recording_type,",
        "recording_subtype,",
        "behavior_mode,",
        ("dish_design," if has_dish_design else "'' AS dish_design,"),
        "rig_id,",
        "arena_id,",
        "camera_id,",
        "dataset_count,",
        "training_dataset_count,",
        "analysis_dataset_count,",
        "inference_dataset_count,",
        "export_dataset_count,",
        "active_dataset_count,",
        "missing_dataset_count,",
        "last_seen_utc",
        "FROM recording_overview",
        "WHERE 1=1",
    ]
    params: List[Any] = []
    dataset_ids = _dataset_ids_for_set(registry, set_filter)
    if dataset_ids is not None:
        if not dataset_ids:
            return []
        placeholders = ", ".join("?" for _ in dataset_ids)
        sql.append(
            "AND recording_id IN ("
            "SELECT DISTINCT recording_id FROM datasets "
            f"WHERE dataset_id IN ({placeholders}) AND recording_id IS NOT NULL AND TRIM(recording_id) <> ''"
            ")"
        )
        params.extend(dataset_ids)
    sql.append("ORDER BY COALESCE(last_seen_utc, '') DESC, recording_id DESC")
    if limit and limit > 0:
        sql.append("LIMIT ?")
        params.append(int(limit))
    rows = registry.conn.execute(" ".join(sql), params).fetchall()
    return [
        RecordingOverviewRow(
            recording_id=str(row["recording_id"]),
            recording_type=row["recording_type"],
            recording_subtype=row["recording_subtype"],
            behavior_mode=row["behavior_mode"],
            dish_design=row["dish_design"],
            rig_id=row["rig_id"],
            arena_id=row["arena_id"],
            camera_id=row["camera_id"],
            dataset_count=int(row["dataset_count"] or 0),
            training_dataset_count=int(row["training_dataset_count"] or 0),
            analysis_dataset_count=int(row["analysis_dataset_count"] or 0),
            inference_dataset_count=int(row["inference_dataset_count"] or 0),
            export_dataset_count=int(row["export_dataset_count"] or 0),
            active_dataset_count=int(row["active_dataset_count"] or 0),
            missing_dataset_count=int(row["missing_dataset_count"] or 0),
            last_seen_utc=row["last_seen_utc"],
        )
        for row in rows
    ]


def _view_exists(registry: Registry, view_name: str) -> bool:
    row = registry.conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'view' AND name = ? LIMIT 1;",
        (view_name,),
    ).fetchone()
    return row is not None


def _load_recording_step_overview_rows(
    registry: Registry,
    *,
    set_filter: Optional[str],
    limit: Optional[int] = None,
) -> List[RecordingStepOverviewRow]:
    if not _view_exists(registry, "recording_step_overview"):
        return []
    sql = [
        "SELECT",
        "recording_id,",
        "dataset_count,",
        "step_rows_total,",
        "ok_rows,",
        "missing_rows,",
        "absent_rows,",
        "na_rows,",
        "error_rows,",
        "blocking_steps_csv,",
        "latest_step_update_utc",
        "FROM recording_step_overview",
        "WHERE 1=1",
    ]
    params: List[Any] = []
    dataset_ids = _dataset_ids_for_set(registry, set_filter)
    if dataset_ids is not None:
        if not dataset_ids:
            return []
        placeholders = ", ".join("?" for _ in dataset_ids)
        sql.append(
            "AND recording_id IN ("
            "SELECT DISTINCT recording_id FROM recording_step_status_latest "
            f"WHERE dataset_id IN ({placeholders}) "
            "AND recording_id IS NOT NULL AND TRIM(recording_id) <> ''"
            ")"
        )
        params.extend(dataset_ids)
    sql.append("ORDER BY COALESCE(latest_step_update_utc, '') DESC, recording_id ASC")
    if limit and limit > 0:
        sql.append("LIMIT ?")
        params.append(int(limit))
    rows = registry.conn.execute(" ".join(sql), params).fetchall()
    return [
        RecordingStepOverviewRow(
            recording_id=str(row["recording_id"]),
            dataset_count=int(row["dataset_count"] or 0),
            step_rows_total=int(row["step_rows_total"] or 0),
            ok_rows=int(row["ok_rows"] or 0),
            missing_rows=int(row["missing_rows"] or 0),
            absent_rows=int(row["absent_rows"] or 0),
            na_rows=int(row["na_rows"] or 0),
            error_rows=int(row["error_rows"] or 0),
            blocking_steps_csv=row["blocking_steps_csv"],
            latest_step_update_utc=row["latest_step_update_utc"],
        )
        for row in rows
    ]


def _load_recording_step_detail_rows(
    registry: Registry,
    *,
    set_filter: Optional[str],
    zarr_use_filter: str = "all",
    limit: Optional[int] = None,
) -> List[RecordingStepDetailRow]:
    if not _view_exists(registry, "recording_step_status_latest"):
        return []
    sql = [
        "SELECT",
        "recording_id,",
        "dataset_id,",
        "zarr_use,",
        "step_name,",
        "status,",
        "run_name,",
        "method,",
        "coverage_pct,",
        "source,",
        "updated_utc",
        "FROM recording_step_status_latest",
        "WHERE 1=1",
    ]
    params: List[Any] = []
    dataset_ids = _dataset_ids_for_set(registry, set_filter)
    if dataset_ids is not None:
        if not dataset_ids:
            return []
        placeholders = ", ".join("?" for _ in dataset_ids)
        sql.append(f"AND dataset_id IN ({placeholders})")
        params.extend(dataset_ids)
    if zarr_use_filter != "all":
        sql.append("AND lower(COALESCE(zarr_use, '')) = ?")
        params.append(zarr_use_filter.lower())
    sql.append("ORDER BY COALESCE(updated_utc, '') DESC, recording_id ASC, dataset_id ASC, step_name ASC")
    if limit and limit > 0:
        sql.append("LIMIT ?")
        params.append(int(limit))
    rows = registry.conn.execute(" ".join(sql), params).fetchall()
    return [
        RecordingStepDetailRow(
            recording_id=row["recording_id"],
            dataset_id=str(row["dataset_id"]),
            zarr_use=row["zarr_use"],
            step_name=str(row["step_name"] or ""),
            status=str(row["status"] or ""),
            run_name=row["run_name"],
            method=row["method"],
            coverage_pct=float(row["coverage_pct"]) if row["coverage_pct"] is not None else None,
            source=row["source"],
            updated_utc=row["updated_utc"],
        )
        for row in rows
    ]


def _load_recording_step_wide_rows(
    registry: Registry,
    *,
    set_filter: Optional[str],
    zarr_use_filter: str = "all",
    limit: Optional[int] = None,
) -> Tuple[List[str], List[Dict[str, str]]]:
    if not _view_exists(registry, "recording_step_status_wide"):
        return [], []

    columns = [
        str(row["name"])
        for row in registry.conn.execute("PRAGMA table_info(recording_step_status_wide);").fetchall()
        if row["name"] is not None
    ]
    if not columns:
        return [], []

    sql = ["SELECT * FROM recording_step_status_wide WHERE 1=1"]
    params: List[Any] = []
    dataset_ids = _dataset_ids_for_set(registry, set_filter)
    if dataset_ids is not None:
        if not dataset_ids:
            return columns, []
        placeholders = ", ".join("?" for _ in dataset_ids)
        sql.append(
            "AND [Recording] IN ("
            "SELECT DISTINCT recording_id "
            "FROM recording_step_status_latest "
            f"WHERE dataset_id IN ({placeholders}) "
            "AND recording_id IS NOT NULL AND TRIM(recording_id) <> ''"
            ")"
        )
        params.extend(dataset_ids)
    if zarr_use_filter != "all":
        sql.append("AND lower(COALESCE([Use], '')) = ?")
        params.append(zarr_use_filter.lower())
    sql.append("ORDER BY [Recording] DESC, [Camera] ASC")
    if limit and limit > 0:
        sql.append("LIMIT ?")
        params.append(int(limit))
    rows = registry.conn.execute(" ".join(sql), params).fetchall()

    rendered_rows: List[Dict[str, str]] = []
    for row in rows:
        rendered_row: Dict[str, str] = {}
        for column in columns:
            value = row[column]
            if value is None:
                rendered_row[column] = "—"
                continue
            text = str(value).strip()
            rendered_row[column] = text if text else "—"
        rendered_rows.append(rendered_row)
    return columns, rendered_rows


def _load_recording_vocab_rows(registry: Registry) -> List[RecordingVocabRow]:
    rows = registry.conn.execute(
        """
        SELECT recording_type, recording_subtype
        FROM recording_subtype_vocab
        WHERE active = 1
        ORDER BY recording_type, recording_subtype;
        """
    ).fetchall()
    results: List[RecordingVocabRow] = []
    for row in rows:
        if row["recording_type"] is None or row["recording_subtype"] is None:
            continue
        recording_type = str(row["recording_type"]).strip()
        recording_subtype = str(row["recording_subtype"]).strip()
        if not recording_type or not recording_subtype:
            continue
        results.append(
            RecordingVocabRow(
                recording_type=recording_type,
                recording_subtype=recording_subtype,
            )
        )
    return results


def _group_recording_vocab_rows(
    rows: List[RecordingVocabRow],
) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for vocab_row in rows:
        recording_type = (vocab_row.recording_type or "").strip()
        recording_subtype = (vocab_row.recording_subtype or "").strip()
        if not recording_type or not recording_subtype:
            continue
        grouped.setdefault(recording_type, [])
        if recording_subtype not in grouped[recording_type]:
            grouped[recording_type].append(recording_subtype)
    for subtypes in grouped.values():
        subtypes.sort()
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _format_recording_vocab_inline(grouped: Dict[str, List[str]]) -> str:
    return "; ".join(
        f"{recording_type}:{','.join(subtypes)}"
        for recording_type, subtypes in grouped.items()
        if subtypes
    )


def _present_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _keypoint_profile_method_info(row: Dict[str, Any]) -> Tuple[Optional[str], bool]:
    for key in KEYPOINT_PROFILE_METHOD_KEYS:
        if key not in row:
            continue
        raw = row.get(key)
        if raw is None:
            return None, True
        text = str(raw).strip()
        return (text or None), True
    return None, False


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


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _zarr_mtime_ns(path_text: Optional[str], *, mtime_cache: Dict[str, Optional[int]]) -> Optional[int]:
    normalized = str(path_text or "").strip()
    if not normalized:
        return None
    if normalized in mtime_cache:
        return mtime_cache[normalized]
    try:
        mtime = int(Path(normalized).stat().st_mtime_ns)
    except OSError:
        mtime = None
    mtime_cache[normalized] = mtime
    return mtime


def _keypoint_profile_stale_reason(
    row: Dict[str, Any],
    *,
    mtime_cache: Dict[str, Optional[int]],
) -> Optional[str]:
    expected_mtime = _coerce_int(row.get("zarr_mtime_ns"))
    if expected_mtime is None:
        return "stale row: missing zarr_mtime_ns"
    zarr_path = str(row.get("zarr_path") or "").strip()
    if not zarr_path:
        return "stale row: missing zarr_path"
    actual_mtime = _zarr_mtime_ns(zarr_path, mtime_cache=mtime_cache)
    if actual_mtime is None:
        return "stale row: zarr missing on disk"
    if int(actual_mtime) != int(expected_mtime):
        return "stale row: mtime mismatch"
    return None


def _keypoint_profile_exclusion_reason(row: Dict[str, Any]) -> Optional[str]:
    profile_missing = not _present_value(row.get("profile_json"))
    method_value, method_detectable = _keypoint_profile_method_info(row)
    method_missing = method_detectable and not _present_value(method_value)
    if not profile_missing and not method_missing:
        return None
    if profile_missing and method_missing:
        return "missing profile_json+method"
    if profile_missing:
        return "missing profile_json"
    return "missing method"


def _summarize_keypoint_profile_rows(rows: List[Dict[str, Any]]) -> KeypointProfileSummary:
    stale_rows = 0
    reasons: Dict[str, int] = {}
    mtime_cache: Dict[str, Optional[int]] = {}
    for row in rows:
        if _keypoint_profile_stale_reason(row, mtime_cache=mtime_cache) is not None:
            stale_rows += 1
        reason = _keypoint_profile_exclusion_reason(row)
        if reason is None:
            continue
        reasons[reason] = reasons.get(reason, 0) + 1
    ordered = dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0])))
    return KeypointProfileSummary(
        total_rows=len(rows),
        stale_rows=stale_rows,
        exclusion_reasons=ordered,
    )


def _review_rollups(
    rows: List[Dict[str, Any]],
    *,
    state_key: str = "review_state",
    intended_use_key: str = "review_intended_use",
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        state = str(row.get(state_key) or "").strip() or "—"
        intended_use = str(row.get(intended_use_key) or "").strip() or "—"
        label = f"{state}/{intended_use}"
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _eye_mask_profile_method_info(row: Dict[str, Any]) -> Tuple[Optional[str], bool]:
    for key in EYE_MASK_PROFILE_METHOD_KEYS:
        if key not in row:
            continue
        raw = row.get(key)
        if raw is None:
            return None, True
        text = str(raw).strip()
        return (text or None), True
    return None, False


def _eye_mask_profile_stale_state(row: Dict[str, Any]) -> Optional[str]:
    for key in EYE_MASK_PROFILE_STALE_STATE_KEYS:
        if key not in row:
            continue
        value = str(row.get(key) or "").strip().lower()
        if value:
            return value
    return None


def _eye_mask_profile_stale_reason(
    row: Dict[str, Any],
    *,
    mtime_cache: Dict[str, Optional[int]],
) -> Optional[str]:
    stale_state = _eye_mask_profile_stale_state(row)
    if stale_state == "stale":
        return "stale row: source keypoint stale"

    expected_mtime = _coerce_int(row.get("zarr_mtime_ns"))
    if expected_mtime is None:
        return "stale row: missing zarr_mtime_ns"
    zarr_path = str(row.get("zarr_path") or "").strip()
    if not zarr_path:
        return "stale row: missing zarr_path"
    actual_mtime = _zarr_mtime_ns(zarr_path, mtime_cache=mtime_cache)
    if actual_mtime is None:
        return "stale row: zarr missing on disk"
    if int(actual_mtime) != int(expected_mtime):
        return "stale row: mtime mismatch"
    return None


def _eye_mask_profile_exclusion_reason(
    row: Dict[str, Any],
    *,
    mtime_cache: Dict[str, Optional[int]],
) -> Optional[str]:
    stale_reason = _eye_mask_profile_stale_reason(row, mtime_cache=mtime_cache)
    if stale_reason is not None:
        return stale_reason

    review_detectable = "review_state" in row or "review_intended_use" in row
    if review_detectable:
        state = row.get("review_state")
        intended_use = row.get("review_intended_use")
        if not state or not intended_use:
            return "missing review"
        if state != EYE_MASK_GATE_REVIEW_STATE or intended_use != EYE_MASK_GATE_REVIEW_INTENDED_USE:
            return "wrong state/use"

    profile_missing = not _present_value(row.get("profile_json"))
    method_value, method_detectable = _eye_mask_profile_method_info(row)
    method_missing = method_detectable and not _present_value(method_value)
    if not profile_missing and not method_missing:
        return None
    if profile_missing and method_missing:
        return "missing profile_json+method"
    if profile_missing:
        return "missing profile_json"
    return "missing method"


def _summarize_eye_mask_profile_rows(rows: List[Dict[str, Any]]) -> EyeMaskProfileSummary:
    stale_rows = 0
    reasons: Dict[str, int] = {}
    mtime_cache: Dict[str, Optional[int]] = {}
    for row in rows:
        if _eye_mask_profile_stale_reason(row, mtime_cache=mtime_cache) is not None:
            stale_rows += 1
        reason = _eye_mask_profile_exclusion_reason(
            row,
            mtime_cache=mtime_cache,
        )
        if reason is None:
            continue
        reasons[reason] = reasons.get(reason, 0) + 1
    ordered = dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0])))
    return EyeMaskProfileSummary(
        total_rows=len(rows),
        stale_rows=stale_rows,
        exclusion_reasons=ordered,
        review_rollups=_review_rollups(rows),
    )


def _eye_mask_performance_stale_reason(
    row: Dict[str, Any],
    *,
    mtime_cache: Dict[str, Optional[int]],
) -> Optional[str]:
    stale_state = str(row.get("source_keypoint_stale_state") or "").strip().lower()
    if stale_state == "stale":
        return "stale source keypoint"
    lifecycle_state = str(row.get("lifecycle_state") or "").strip().lower()
    if lifecycle_state == "stale":
        return "stale lifecycle"

    expected_mtime = _coerce_int(row.get("zarr_mtime_ns"))
    zarr_path = str(row.get("zarr_path") or "").strip()
    if expected_mtime is None:
        return "stale row: missing zarr_mtime_ns"
    if not zarr_path:
        return "stale row: missing zarr_path"
    actual_mtime = _zarr_mtime_ns(zarr_path, mtime_cache=mtime_cache)
    if actual_mtime is None:
        return "stale row: zarr missing on disk"
    if int(actual_mtime) != int(expected_mtime):
        return "stale row: mtime mismatch"
    return None


def _eye_mask_performance_row_passes_default_gate(
    row: Dict[str, Any],
    *,
    mtime_cache: Dict[str, Optional[int]],
) -> bool:
    if _eye_mask_performance_stale_reason(row, mtime_cache=mtime_cache) is not None:
        return False
    return (
        row.get("review_state") == EYE_MASK_GATE_REVIEW_STATE
        and row.get("review_intended_use") == EYE_MASK_GATE_REVIEW_INTENDED_USE
    )


def _eye_mask_performance_exclusion_reason(
    row: Dict[str, Any],
    *,
    mtime_cache: Dict[str, Optional[int]],
) -> Optional[str]:
    if _eye_mask_performance_row_passes_default_gate(row, mtime_cache=mtime_cache):
        return None
    stale_reason = _eye_mask_performance_stale_reason(row, mtime_cache=mtime_cache)
    if stale_reason is not None:
        return stale_reason
    state = row.get("review_state")
    intended_use = row.get("review_intended_use")
    if not state or not intended_use:
        return "missing review"
    if state != EYE_MASK_GATE_REVIEW_STATE or intended_use != EYE_MASK_GATE_REVIEW_INTENDED_USE:
        return "wrong state/use"
    return "other"


def _summarize_eye_mask_performance_rows(rows: List[Dict[str, Any]]) -> EyeMaskPerformanceSummary:
    passing = 0
    stale_rows = 0
    reasons: Dict[str, int] = {}
    mtime_cache: Dict[str, Optional[int]] = {}
    for row in rows:
        stale_reason = _eye_mask_performance_stale_reason(row, mtime_cache=mtime_cache)
        if stale_reason is not None:
            stale_rows += 1
        reason = _eye_mask_performance_exclusion_reason(row, mtime_cache=mtime_cache)
        if reason is None:
            passing += 1
            continue
        reasons[reason] = reasons.get(reason, 0) + 1
    total = len(rows)
    ordered = dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0])))
    return EyeMaskPerformanceSummary(
        total_rows=total,
        passing_rows=passing,
        excluded_rows=total - passing,
        stale_rows=stale_rows,
        exclusion_reasons=ordered,
        review_rollups=_review_rollups(rows),
    )


def _eye_mask_profile_remediation_lines(
    summary: EyeMaskProfileSummary,
    *,
    registry_path: Path,
) -> List[str]:
    sync_cmd = f"{EYE_MASK_SYNC_COMMAND} --registry {registry_path} --apply"
    refresh_cmd = f"{EYE_MASK_REFRESH_COMMAND} --registry {registry_path} --apply"
    lines: List[str] = []
    if summary.total_rows <= 0:
        lines.append(f"remediation: no eye-mask profile rows found; run `{sync_cmd}`")
    if summary.stale_rows > 0:
        lines.append(f"remediation: stale eye-mask profile rows detected; run `{refresh_cmd}`")
        lines.append(f"remediation: re-sync latest profile rows with `{sync_cmd}`")
    return lines


def _detect_quality_stale_reason(
    row: Dict[str, Any],
    *,
    mtime_cache: Dict[str, Optional[int]],
) -> Optional[str]:
    expected_mtime = _coerce_int(row.get("zarr_mtime_ns"))
    if expected_mtime is None:
        return "stale row: missing zarr_mtime_ns"
    zarr_path = str(row.get("zarr_path") or "").strip()
    if not zarr_path:
        return "stale row: missing zarr_path"
    actual_mtime = _zarr_mtime_ns(zarr_path, mtime_cache=mtime_cache)
    if actual_mtime is None:
        return "stale row: zarr missing on disk"
    if int(actual_mtime) != int(expected_mtime):
        return "stale row: mtime mismatch"
    return None


def _detect_quality_row_passes_default_gate(
    row: Dict[str, Any],
    *,
    mtime_cache: Dict[str, Optional[int]],
) -> bool:
    if _detect_quality_stale_reason(row, mtime_cache=mtime_cache) is not None:
        return False
    rate = _coerce_float(row.get("interpolated_detections_rate"))
    return (
        row.get("review_state") == DETECT_GATE_REVIEW_STATE
        and row.get("review_intended_use") == DETECT_GATE_REVIEW_INTENDED_USE
        and rate is not None
        and float(rate) <= DETECT_GATE_MAX_INTERPOLATED_RATE
    )


def _detect_quality_exclusion_reason(
    row: Dict[str, Any],
    *,
    mtime_cache: Dict[str, Optional[int]],
) -> Optional[str]:
    if _detect_quality_row_passes_default_gate(row, mtime_cache=mtime_cache):
        return None
    stale_reason = _detect_quality_stale_reason(row, mtime_cache=mtime_cache)
    if stale_reason is not None:
        return stale_reason
    state = row.get("review_state")
    intended_use = row.get("review_intended_use")
    rate = _coerce_float(row.get("interpolated_detections_rate"))
    if not state or not intended_use:
        return "missing review"
    if state != DETECT_GATE_REVIEW_STATE or intended_use != DETECT_GATE_REVIEW_INTENDED_USE:
        return "wrong state/use"
    if rate is None:
        return "missing interpolated rate"
    if float(rate) > DETECT_GATE_MAX_INTERPOLATED_RATE:
        return "high interpolation"
    return "other"


def _summarize_detect_quality_rows(rows: List[Dict[str, Any]]) -> DetectQualitySummary:
    passing = 0
    reasons: Dict[str, int] = {}
    mtime_cache: Dict[str, Optional[int]] = {}
    for row in rows:
        reason = _detect_quality_exclusion_reason(row, mtime_cache=mtime_cache)
        if reason is None:
            passing += 1
            continue
        reasons[reason] = reasons.get(reason, 0) + 1
    total = len(rows)
    excluded = total - passing
    ordered = dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0])))
    return DetectQualitySummary(
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
        choices=[
            "sets",
            "datasets",
            "recordings",
            "recording-overview",
            "recording-steps",
            "recording-steps-wide",
            "models",
            "onnx",
            "tensorrt",
            "detect-quality",
            "keypoint-quality",
            "keypoint-profile",
            "eye-mask-quality",
            "eye-mask-performance",
            "eye-mask-profile",
            "lineage",
        ],
        default="sets",
        help=(
            "Select output view: sets, datasets, recordings, recording-overview, "
            "recording-steps, recording-steps-wide, models, onnx, tensorrt, "
            "detect-quality, keypoint-quality, keypoint-profile, eye-mask-quality, "
            "eye-mask-performance, "
            "eye-mask-profile, or lineage (default: sets)."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Show all registry views (sets, datasets, recordings, recording-overview, "
            "models, onnx, tensorrt, lineage, detect quality, keypoint quality, "
            "keypoint profile, eye-mask quality/performance, and eye-mask profile)."
        ),
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--show-detect-quality",
        action="store_true",
        help="Print detailed detect quality rows (summary is always shown).",
    )
    parser.add_argument(
        "--show-keypoint-quality",
        action="store_true",
        help="Print detailed keypoint quality rows (summary is always shown).",
    )
    parser.add_argument(
        "--show-eye-mask-quality",
        action="store_true",
        help="Print detailed eye-mask quality rows (summary is always shown).",
    )
    parser.add_argument(
        "--show-eye-mask-performance",
        action="store_true",
        help="Alias of --show-eye-mask-quality.",
    )
    parser.add_argument(
        "--show-eye-mask-profile",
        action="store_true",
        help="Print detailed eye-mask profile rows (summary is always shown).",
    )
    parser.add_argument(
        "--recording-summary",
        action="store_true",
        help="Print recording_type/recording_subtype distribution summary.",
    )
    parser.add_argument(
        "--show-recording-step-details",
        action="store_true",
        help="Show step-detail rows from recording_step_status_latest (recording-steps view).",
    )
    parser.add_argument(
        "--zarr-use",
        choices=["all", "training", "analysis"],
        default="all",
        help=(
            "Filter recording-step rows by zarr_use for recording-steps detail rows "
            "and recording-steps-wide view (default: all)."
        ),
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
    summary_only_mode = bool(args.recording_summary and not args.all and args.view == "sets")
    show_sets = args.all or args.view == "sets"
    show_datasets = args.all or args.view == "datasets"
    show_recordings = args.all or args.view == "recordings"
    show_recording_overview = args.all or args.view == "recording-overview"
    show_recording_steps = args.view == "recording-steps"
    show_recording_steps_wide = args.view == "recording-steps-wide"
    show_recording_step_details = bool(show_recording_steps and args.show_recording_step_details)
    show_models = args.all or args.view == "models"
    show_onnx = args.all or args.view == "onnx"
    show_tensorrt = args.all or args.view == "tensorrt"
    show_lineage = args.all or args.view == "lineage"
    show_detect_view = args.view == "detect-quality"
    show_detect_details = args.show_detect_quality or show_detect_view
    show_keypoint_view = args.view == "keypoint-quality"
    show_keypoint_details = args.show_keypoint_quality or show_keypoint_view
    show_keypoint_profile = args.view == "keypoint-profile"
    show_eye_mask_performance_view = args.view in {"eye-mask-quality", "eye-mask-performance"}
    show_eye_mask_performance_details = (
        args.show_eye_mask_quality
        or args.show_eye_mask_performance
        or show_eye_mask_performance_view
    )
    show_eye_mask_profile_view = args.view == "eye-mask-profile"
    show_eye_mask_profile_details = args.show_eye_mask_profile or show_eye_mask_profile_view
    if summary_only_mode:
        show_sets = False
        show_datasets = False
        show_recordings = False
        show_recording_overview = False
        show_recording_steps = False
        show_recording_steps_wide = False
        show_recording_step_details = False
        show_models = False
        show_onnx = False
        show_tensorrt = False
        show_lineage = False
        show_detect_view = False
        show_detect_details = False
        show_keypoint_view = False
        show_keypoint_details = False
        show_keypoint_profile = False
        show_eye_mask_performance_view = False
        show_eye_mask_performance_details = False
        show_eye_mask_profile_view = False
        show_eye_mask_profile_details = False

    set_rows = _load_set_rows(registry, args.set_id, args.limit) if show_sets else []
    dataset_rows = _load_dataset_rows(
        registry,
        set_filter=args.set_id,
        limit=args.limit,
    ) if show_datasets else []
    recording_rows = _load_recording_rows(
        registry,
        set_filter=args.set_id,
        limit=args.limit,
    ) if show_recordings else []
    recording_overview_rows = _load_recording_overview_rows(
        registry,
        set_filter=args.set_id,
        limit=args.limit,
    ) if show_recording_overview else []
    recording_step_overview_rows = _load_recording_step_overview_rows(
        registry,
        set_filter=args.set_id,
        limit=args.limit,
    ) if show_recording_steps else []
    recording_step_detail_rows = _load_recording_step_detail_rows(
        registry,
        set_filter=args.set_id,
        zarr_use_filter=args.zarr_use,
        limit=args.limit,
    ) if show_recording_step_details else []
    recording_step_wide_columns, recording_step_wide_rows = _load_recording_step_wide_rows(
        registry,
        set_filter=args.set_id,
        zarr_use_filter=args.zarr_use,
        limit=args.limit,
    ) if show_recording_steps_wide else ([], [])
    recording_summary_rows = _load_recording_summary_rows(registry) if args.recording_summary else []
    recording_vocab_rows = _load_recording_vocab_rows(registry) if args.recording_summary else []
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
    lineage_rows = (
        _load_dataset_lineage_rows(
            registry,
            set_filter=args.set_id,
            limit=args.limit,
        )
        if show_lineage
        else []
    )
    detect_quality_limit = args.limit if (show_detect_details or show_detect_view) else None
    detect_quality_rows = _load_detect_quality_rows(
        registry,
        set_filter=args.set_id,
        limit=detect_quality_limit,
    )
    detect_quality_summary = _summarize_detect_quality_rows(detect_quality_rows)
    keypoint_quality_limit = args.limit if (show_keypoint_details or show_keypoint_view) else None
    keypoint_quality_rows = _load_keypoint_quality_rows(
        registry,
        set_filter=args.set_id,
        limit=keypoint_quality_limit,
    )
    keypoint_quality_summary = _summarize_keypoint_quality_rows(keypoint_quality_rows)
    keypoint_profile_limit = args.limit if show_keypoint_profile else None
    keypoint_profile_rows = _load_keypoint_profile_rows(
        registry,
        set_filter=args.set_id,
        limit=keypoint_profile_limit,
    )
    keypoint_profile_summary = _summarize_keypoint_profile_rows(keypoint_profile_rows)
    eye_mask_performance_limit = args.limit if (show_eye_mask_performance_view or show_eye_mask_performance_details) else None
    eye_mask_performance_rows = _load_eye_mask_performance_rows(
        registry,
        set_filter=args.set_id,
        limit=eye_mask_performance_limit,
    )
    eye_mask_performance_summary = _summarize_eye_mask_performance_rows(eye_mask_performance_rows)
    eye_mask_profile_limit = args.limit if (show_eye_mask_profile_view or show_eye_mask_profile_details) else None
    eye_mask_profile_rows = _load_eye_mask_profile_rows(
        registry,
        set_filter=args.set_id,
        limit=eye_mask_profile_limit,
    )
    eye_mask_profile_summary = _summarize_eye_mask_profile_rows(eye_mask_profile_rows)
    eye_mask_profile_remediation = _eye_mask_profile_remediation_lines(
        eye_mask_profile_summary,
        registry_path=registry_path,
    )
    dataset_lineage_summary = _summarize_dataset_lineage(
        registry,
        set_filter=args.set_id,
    ) if (show_lineage or args.all) else None
    registry.close()

    if not summary_only_mode and not args.all and args.view == "sets" and not set_rows:
        print("No training sets found.")
        return 1
    if not args.all and args.view == "datasets" and not dataset_rows:
        print("No dataset rows found.")
        return 1
    if not args.all and args.view == "models" and not model_rows:
        print("No training runs found.")
        return 1
    if not args.all and args.view == "recordings" and not recording_rows:
        print("No recording rows found.")
        return 1
    if not args.all and args.view == "recording-overview" and not recording_overview_rows:
        print("No recording overview rows found.")
        return 1
    if not args.all and args.view == "recording-steps" and not recording_step_overview_rows:
        print("No recording step rows found.")
        return 1
    if not args.all and args.view == "recording-steps-wide" and not recording_step_wide_rows:
        print("No recording step wide rows found.")
        return 1
    if not args.all and args.view == "onnx" and not onnx_rows:
        print("No ONNX rows found.")
        return 1
    if not args.all and args.view == "tensorrt" and not trt_rows:
        print("No TensorRT rows found.")
        return 1
    if not args.all and args.view == "lineage" and not lineage_rows:
        print("No dataset lineage rows found.")
        return 1
    if not args.all and args.view == "detect-quality" and not detect_quality_rows:
        print("No detect quality rows found.")
        return 1
    if not args.all and args.view == "keypoint-quality" and not keypoint_quality_rows:
        print("No keypoint quality rows found.")
        return 1
    if not args.all and args.view == "keypoint-profile" and not keypoint_profile_rows:
        print("No keypoint profile rows found.")
        return 1
    if not args.all and args.view in {"eye-mask-quality", "eye-mask-performance"} and not eye_mask_performance_rows:
        print("No eye-mask quality rows found.")
        return 1
    if not args.all and args.view == "eye-mask-profile" and not eye_mask_profile_rows:
        print("No eye-mask profile rows found.")
        for line in eye_mask_profile_remediation:
            print(f"  {line}")
        return 1

    show_inference_col = any(row.inference_dataset_count > 0 for row in recording_overview_rows)
    show_export_col = any(row.export_dataset_count > 0 for row in recording_overview_rows)

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
        if show_datasets:
            table = Table(title="Datasets", show_lines=False)
            table.add_column("Dataset ID", style="cyan")
            table.add_column("Recording ID")
            table.add_column("Kind")
            table.add_column("Status")
            table.add_column("Path")
            table.add_column("Origin")
            table.add_column("Use")
            table.add_column("Rig")
            table.add_column("Arena")
            table.add_column("Camera")
            table.add_column("Parents", justify="right")
            table.add_column("Children", justify="right")
            table.add_column("Last Seen")
            table.add_column("Zarr Path")
            for row in dataset_rows:
                table.add_row(
                    row.dataset_id,
                    row.recording_id or "—",
                    row.artifact_kind or "—",
                    row.status or "—",
                    _status_rich(_path_ok(row.zarr_path)),
                    row.zarr_origin or "—",
                    row.zarr_use or "—",
                    row.rig_id or "—",
                    row.arena_id or "—",
                    row.camera_id or "—",
                    str(row.parent_count),
                    str(row.child_count),
                    _format_time(row.last_seen_utc),
                    row.zarr_path or "—",
                )
            console.print(table)
        if show_lineage:
            table = Table(title="Dataset Lineage", show_lines=False)
            table.add_column("Child Dataset", style="cyan")
            table.add_column("Parent Dataset")
            table.add_column("Relationship")
            table.add_column("Source Set")
            table.add_column("Updated")
            for row in lineage_rows:
                table.add_row(
                    row.child_dataset_id,
                    row.parent_dataset_id,
                    row.relationship_type,
                    row.source_set_id or "—",
                    _format_time(row.updated_utc),
                )
            console.print(table)
            if dataset_lineage_summary is not None:
                console.print("[bold]Dataset Lineage Summary[/bold]")
                console.print(f"- total edges: {dataset_lineage_summary.edge_count}")
                console.print(f"- merged datasets: {dataset_lineage_summary.merged_dataset_count}")
                console.print(
                    "- merged datasets missing lineage: "
                    f"{dataset_lineage_summary.merged_missing_lineage_count}"
                )
                console.print(
                    "- training-set lineage mismatches: "
                    f"{dataset_lineage_summary.set_lineage_mismatch_count}"
                )
                if dataset_lineage_summary.relationship_type_counts:
                    rel_text = ", ".join(
                        f"{k}={v}"
                        for k, v in dataset_lineage_summary.relationship_type_counts.items()
                    )
                else:
                    rel_text = "none"
                console.print(f"- relationship types: {rel_text}")
        if show_recordings:
            table = Table(title="Recordings", show_lines=False)
            table.add_column("Recording ID", style="cyan")
            table.add_column("Type")
            table.add_column("Subtype")
            table.add_column("Behavior")
            table.add_column("Dish")
            table.add_column("Schema")
            table.add_column("Rig")
            table.add_column("Arena")
            table.add_column("Camera")
            table.add_column("Artifacts", justify="right")
            table.add_column("Required")
            table.add_column("Started")
            table.add_column("Path")
            for row in recording_rows:
                missing = row.missing_required_artifacts
                required = (
                    "[chartreuse1]OK[/chartreuse1]"
                    if not missing
                    else "[red]MISS[/red] " + ",".join(missing)
                )
                table.add_row(
                    row.recording_id,
                    row.recording_type or "—",
                    row.recording_subtype or "—",
                    row.behavior_mode or "—",
                    row.dish_design or "—",
                    row.artifact_schema_id or "—",
                    row.rig_id or "—",
                    row.arena_id or "—",
                    row.camera_id or "—",
                    str(row.artifact_count),
                    required,
                    _format_time(row.started_utc or row.updated_utc),
                    row.recording_path or "—",
                )
            console.print(table)
        if show_recording_overview:
            table = Table(title="Recording Overview", show_lines=False)
            table.add_column("Recording ID", style="cyan")
            table.add_column("Type")
            table.add_column("Subtype")
            table.add_column("Behavior")
            table.add_column("Dish")
            table.add_column("Rig")
            table.add_column("Arena")
            table.add_column("Camera")
            table.add_column("Datasets", justify="right")
            table.add_column("Train", justify="right")
            table.add_column("Analysis", justify="right")
            if show_inference_col:
                table.add_column("Infer", justify="right")
            if show_export_col:
                table.add_column("Export", justify="right")
            table.add_column("Active", justify="right")
            table.add_column("Missing", justify="right")
            table.add_column("Last Seen")
            for row in recording_overview_rows:
                cells = [
                    row.recording_id,
                    row.recording_type or "—",
                    row.recording_subtype or "—",
                    row.behavior_mode or "—",
                    row.dish_design or "—",
                    row.rig_id or "—",
                    row.arena_id or "—",
                    row.camera_id or "—",
                    str(row.dataset_count),
                    str(row.training_dataset_count),
                    str(row.analysis_dataset_count),
                    str(row.active_dataset_count),
                    str(row.missing_dataset_count),
                    _format_time(row.last_seen_utc),
                ]
                if show_inference_col:
                    cells.insert(10, str(row.inference_dataset_count))
                if show_export_col:
                    export_index = 11 if show_inference_col else 10
                    cells.insert(export_index, str(row.export_dataset_count))
                table.add_row(*cells)
            console.print(table)
        if show_recording_steps:
            table = Table(title="Recording Step Overview", show_lines=False)
            table.add_column("Recording ID", style="cyan")
            table.add_column("Datasets", justify="right")
            table.add_column("Rows", justify="right")
            table.add_column("OK", justify="right")
            table.add_column("Missing", justify="right")
            table.add_column("Absent", justify="right")
            table.add_column("N/A", justify="right")
            table.add_column("Error", justify="right")
            table.add_column("Blocking Steps")
            table.add_column("Latest Update")
            for row in recording_step_overview_rows:
                table.add_row(
                    row.recording_id,
                    str(row.dataset_count),
                    str(row.step_rows_total),
                    str(row.ok_rows),
                    str(row.missing_rows),
                    str(row.absent_rows),
                    str(row.na_rows),
                    str(row.error_rows),
                    row.blocking_steps_csv or "—",
                    _format_time(row.latest_step_update_utc),
                )
            console.print(table)
            if show_recording_step_details and recording_step_detail_rows:
                detail_table = Table(title="Recording Step Details", show_lines=False)
                detail_table.add_column("Recording ID", style="cyan")
                detail_table.add_column("Dataset ID")
                detail_table.add_column("Use")
                detail_table.add_column("Step")
                detail_table.add_column("Status")
                detail_table.add_column("Run")
                detail_table.add_column("Method")
                detail_table.add_column("Coverage")
                detail_table.add_column("Source")
                detail_table.add_column("Updated")
                for row in recording_step_detail_rows:
                    detail_table.add_row(
                        row.recording_id or "—",
                        row.dataset_id,
                        row.zarr_use or "—",
                        row.step_name or "—",
                        _step_status_rich(row.status),
                        row.run_name or "—",
                        row.method or "—",
                        (f"{float(row.coverage_pct):.1f}%" if row.coverage_pct is not None else "—"),
                        row.source or "—",
                        _format_time(row.updated_utc),
                    )
                console.print(detail_table)
        if show_recording_steps_wide:
            table = Table(title="Recording Step Status (Wide)", show_lines=False)
            for column in recording_step_wide_columns:
                style = "cyan" if column == "Recording" else None
                table.add_column(column, style=style)
            for row in recording_step_wide_rows:
                table.add_row(*[_wide_cell_rich(row.get(column)) for column in recording_step_wide_columns])
            console.print(table)
        if args.recording_summary:
            summary_table = Table(title="Recording Type/Subtype Summary", show_lines=False)
            summary_table.add_column("Type")
            summary_table.add_column("Subtype")
            summary_table.add_column("Count", justify="right")
            for row in recording_summary_rows:
                summary_table.add_row(
                    row.recording_type or "—",
                    row.recording_subtype or "—",
                    str(row.count),
                )
            console.print(summary_table)
            if recording_vocab_rows:
                grouped = _group_recording_vocab_rows(recording_vocab_rows)
                if grouped:
                    vocab_text = _format_recording_vocab_inline(grouped)
                    console.print(f"Allowed subtype vocab: {vocab_text}")
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
            table.add_column("NMS")
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
                    _nms_summary(
                        nms_conf=row.nms_conf,
                        nms_iou=row.nms_iou,
                        nms_topk=row.nms_topk,
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
            table.add_column("NMS")
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
                    _nms_summary(
                        nms_conf=row.nms_conf,
                        nms_iou=row.nms_iou,
                        nms_topk=row.nms_topk,
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
        if not summary_only_mode:
            detect_lines = [
                f"total rows: {detect_quality_summary.total_rows}",
                (
                    "passing rows "
                    f"({DETECT_GATE_REVIEW_STATE}/{DETECT_GATE_REVIEW_INTENDED_USE}, "
                    f"interpolated_rate<={DETECT_GATE_MAX_INTERPOLATED_RATE:.2f}, fresh mtime): "
                    f"{detect_quality_summary.passing_rows}"
                ),
                f"excluded rows: {detect_quality_summary.excluded_rows}",
            ]
            if detect_quality_summary.exclusion_reasons:
                detect_reason_text = ", ".join(
                    f"{name}={count}"
                    for name, count in detect_quality_summary.exclusion_reasons.items()
                )
            else:
                detect_reason_text = "none"
            detect_lines.append(f"top exclusion reasons: {detect_reason_text}")
            console.print("[bold]Detect Quality[/bold]")
            for line in detect_lines:
                console.print(f"- {line}")
            if show_detect_details and detect_quality_rows:
                detect_table = Table(title="Detect Quality Details", show_lines=False)
                detect_table.add_column("Dataset", style="cyan")
                detect_table.add_column("Use")
                detect_table.add_column("Method")
                detect_table.add_column("Review")
                detect_table.add_column("Interp/Total")
                detect_table.add_column("InterpRate")
                detect_table.add_column("Stale")
                detect_table.add_column("Gate")
                detect_table.add_column("Reason")
                detect_mtime_cache: Dict[str, Optional[int]] = {}
                for row in detect_quality_rows:
                    passes = _detect_quality_row_passes_default_gate(
                        row,
                        mtime_cache=detect_mtime_cache,
                    )
                    reason = _detect_quality_exclusion_reason(
                        row,
                        mtime_cache=detect_mtime_cache,
                    ) or "—"
                    total = row.get("total_detections")
                    interpolated = row.get("interpolated_detections")
                    interp_rate = _coerce_float(row.get("interpolated_detections_rate"))
                    stale = _detect_quality_stale_reason(row, mtime_cache=detect_mtime_cache)
                    review = f"{row.get('review_state') or '—'}/{row.get('review_intended_use') or '—'}"
                    detect_table.add_row(
                        str(row.get("dataset_id") or "—"),
                        str(row.get("zarr_use") or "—"),
                        str(row.get("detect_method") or "—"),
                        review,
                        f"{interpolated if interpolated is not None else '—'}/{total if total is not None else '—'}",
                        f"{float(interp_rate):.3f}" if interp_rate is not None else "—",
                        "1" if stale is not None else "0",
                        "[chartreuse1]PASS[/chartreuse1]" if passes else "[red]EXCLUDE[/red]",
                        reason,
                    )
                console.print(detect_table)

            keypoint_lines = [
                f"total rows: {keypoint_quality_summary.total_rows}",
                (
                    "passing rows "
                    f"({KEYPOINT_GATE_REVIEW_STATE}/{KEYPOINT_GATE_REVIEW_INTENDED_USE}, "
                    f"usable_rate>={KEYPOINT_GATE_MIN_RATE:.2f}): {keypoint_quality_summary.passing_rows}"
                ),
                f"excluded rows: {keypoint_quality_summary.excluded_rows}",
            ]
            if keypoint_quality_summary.exclusion_reasons:
                keypoint_reason_text = ", ".join(
                    f"{name}={count}"
                    for name, count in keypoint_quality_summary.exclusion_reasons.items()
                )
            else:
                keypoint_reason_text = "none"
            keypoint_lines.append(f"top exclusion reasons: {keypoint_reason_text}")
            keypoint_lines.append(
                "quality_stale note: currently flags only rows missing stored zarr_mtime_ns."
            )
            console.print("[bold]Keypoint Quality[/bold]")
            for line in keypoint_lines:
                console.print(f"- {line}")
            if show_keypoint_details and keypoint_quality_rows:
                kq_table = Table(title="Keypoint Quality Details", show_lines=False)
                kq_table.add_column("Dataset", style="cyan")
                kq_table.add_column("Use")
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
                        str(row.get("zarr_use") or "—"),
                        str(row.get("keypoint_method") or "—"),
                        review,
                        f"{usable if usable is not None else '—'}/{total if total is not None else '—'}",
                        f"{float(rate):.3f}" if isinstance(rate, (int, float)) else "—",
                        "1" if int(row.get("quality_stale") or 0) == 1 else "0",
                        "[chartreuse1]PASS[/chartreuse1]" if passes else "[red]EXCLUDE[/red]",
                        reason,
                    )
                console.print(kq_table)

            keypoint_profile_lines = [
                f"total rows: {keypoint_profile_summary.total_rows}",
                f"stale rows (mtime mismatch/missing): {keypoint_profile_summary.stale_rows}",
            ]
            if keypoint_profile_summary.exclusion_reasons:
                keypoint_profile_reason_text = ", ".join(
                    f"{name}={count}"
                    for name, count in keypoint_profile_summary.exclusion_reasons.items()
                )
            else:
                keypoint_profile_reason_text = "none"
            keypoint_profile_lines.append(
                f"top exclusion-ish reasons: {keypoint_profile_reason_text}"
            )
            console.print("[bold]Keypoint Profile[/bold]")
            for line in keypoint_profile_lines:
                console.print(f"- {line}")

            eye_mask_lines = [
                f"total rows: {eye_mask_performance_summary.total_rows}",
                (
                    "passing rows "
                    f"({EYE_MASK_GATE_REVIEW_STATE}/{EYE_MASK_GATE_REVIEW_INTENDED_USE}, non-stale source): "
                    f"{eye_mask_performance_summary.passing_rows}"
                ),
                f"excluded rows: {eye_mask_performance_summary.excluded_rows}",
                f"stale rows: {eye_mask_performance_summary.stale_rows}",
            ]
            if eye_mask_performance_summary.exclusion_reasons:
                eye_mask_reason_text = ", ".join(
                    f"{name}={count}"
                    for name, count in eye_mask_performance_summary.exclusion_reasons.items()
                )
            else:
                eye_mask_reason_text = "none"
            eye_mask_lines.append(f"top exclusion reasons: {eye_mask_reason_text}")
            if eye_mask_performance_summary.review_rollups:
                rollup_text = ", ".join(
                    f"{name}={count}"
                    for name, count in eye_mask_performance_summary.review_rollups.items()
                )
            else:
                rollup_text = "none"
            eye_mask_lines.append(f"review rollups: {rollup_text}")
            console.print("[bold]Eye-Mask Quality[/bold]")
            for line in eye_mask_lines:
                console.print(f"- {line}")
            if show_eye_mask_performance_details and eye_mask_performance_rows:
                ep_table = Table(title="Eye-Mask Quality Details", show_lines=False)
                ep_table.add_column("Dataset", style="cyan")
                ep_table.add_column("Use")
                ep_table.add_column("Stage")
                ep_table.add_column("Method")
                ep_table.add_column("Review")
                ep_table.add_column("Stale")
                ep_table.add_column("Lifecycle")
                ep_table.add_column("Rate")
                ep_table.add_column("Gate")
                ep_table.add_column("Reason")
                eye_mask_mtime_cache: Dict[str, Optional[int]] = {}
                for row in eye_mask_performance_rows:
                    reason = _eye_mask_performance_exclusion_reason(
                        row,
                        mtime_cache=eye_mask_mtime_cache,
                    ) or "—"
                    stale_reason = _eye_mask_performance_stale_reason(
                        row,
                        mtime_cache=eye_mask_mtime_cache,
                    )
                    passes = _eye_mask_performance_row_passes_default_gate(
                        row,
                        mtime_cache=eye_mask_mtime_cache,
                    )
                    review = f"{row.get('review_state') or '—'}/{row.get('review_intended_use') or '—'}"
                    rate = _coerce_float(row.get("successful_roi_pair_rate"))
                    ep_table.add_row(
                        str(row.get("dataset_id") or "—"),
                        str(row.get("zarr_use") or "—"),
                        str(row.get("stage_group") or "—"),
                        str(row.get("method") or "—"),
                        review,
                        "1" if stale_reason is not None else "0",
                        str(row.get("lifecycle_state") or "—"),
                        f"{float(rate):.3f}" if rate is not None else "—",
                        "[chartreuse1]PASS[/chartreuse1]" if passes else "[red]EXCLUDE[/red]",
                        reason,
                    )
                console.print(ep_table)

            eye_mask_profile_lines = [
                f"total rows: {eye_mask_profile_summary.total_rows}",
                f"stale rows (mtime mismatch/missing): {eye_mask_profile_summary.stale_rows}",
            ]
            if eye_mask_profile_summary.exclusion_reasons:
                eye_mask_profile_reason_text = ", ".join(
                    f"{name}={count}"
                    for name, count in eye_mask_profile_summary.exclusion_reasons.items()
                )
            else:
                eye_mask_profile_reason_text = "none"
            eye_mask_profile_lines.append(f"top exclusion reasons: {eye_mask_profile_reason_text}")
            if eye_mask_profile_summary.review_rollups:
                eye_mask_profile_rollups = ", ".join(
                    f"{name}={count}"
                    for name, count in eye_mask_profile_summary.review_rollups.items()
                )
            else:
                eye_mask_profile_rollups = "none"
            eye_mask_profile_lines.append(f"review rollups: {eye_mask_profile_rollups}")
            console.print("[bold]Eye-Mask Profile[/bold]")
            for line in eye_mask_profile_lines:
                console.print(f"- {line}")
            for line in eye_mask_profile_remediation:
                console.print(f"- {line}")
            if show_eye_mask_profile_details and eye_mask_profile_rows:
                epf_table = Table(title="Eye-Mask Profile Details", show_lines=False)
                epf_table.add_column("Dataset", style="cyan")
                epf_table.add_column("Use")
                epf_table.add_column("Method")
                epf_table.add_column("Review")
                epf_table.add_column("Stale")
                epf_table.add_column("Lifecycle")
                epf_table.add_column("Profile JSON")
                epf_table.add_column("Gate")
                epf_table.add_column("Reason")
                eye_mask_profile_mtime_cache: Dict[str, Optional[int]] = {}
                for row in eye_mask_profile_rows:
                    reason = _eye_mask_profile_exclusion_reason(
                        row,
                        mtime_cache=eye_mask_profile_mtime_cache,
                    ) or "—"
                    stale = _eye_mask_profile_stale_reason(
                        row,
                        mtime_cache=eye_mask_profile_mtime_cache,
                    )
                    epf_table.add_row(
                        str(row.get("dataset_id") or "—"),
                        str(row.get("zarr_use") or "—"),
                        str(
                            row.get("eye_mask_method")
                            or row.get("method")
                            or row.get("source_eye_masks_method")
                            or "—"
                        ),
                        f"{row.get('review_state') or '—'}/{row.get('review_intended_use') or '—'}",
                        "1" if stale is not None else "0",
                        str(row.get("lifecycle_state") or "—"),
                        "1" if _present_value(row.get("profile_json")) else "0",
                        "[chartreuse1]PASS[/chartreuse1]" if reason == "—" else "[red]EXCLUDE[/red]",
                        reason,
                    )
                console.print(epf_table)
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
        if show_datasets:
            for row in dataset_rows:
                print(row.dataset_id)
                print(f"  recording_id: {row.recording_id or '—'}")
                print(f"  kind: {row.artifact_kind or '—'}")
                print(f"  status: {row.status or '—'}")
                print(f"  path: {_status_text(_path_ok(row.zarr_path))}")
                print(f"  origin: {row.zarr_origin or '—'}")
                print(f"  use: {row.zarr_use or '—'}")
                print(f"  rig: {row.rig_id or '—'}")
                print(f"  arena: {row.arena_id or '—'}")
                print(f"  camera: {row.camera_id or '—'}")
                print(f"  parents: {row.parent_count}")
                print(f"  children: {row.child_count}")
                print(f"  last_seen: {_format_time(row.last_seen_utc)}")
                print(f"  zarr_path: {row.zarr_path or '—'}")
        if show_lineage:
            for row in lineage_rows:
                print(row.child_dataset_id)
                print(f"  parent_dataset_id: {row.parent_dataset_id}")
                print(f"  relationship: {row.relationship_type}")
                print(f"  source_set_id: {row.source_set_id or '—'}")
                print(f"  updated: {_format_time(row.updated_utc)}")
            if dataset_lineage_summary is not None:
                print("Dataset Lineage Summary")
                print(f"  total edges: {dataset_lineage_summary.edge_count}")
                print(f"  merged datasets: {dataset_lineage_summary.merged_dataset_count}")
                print(
                    "  merged datasets missing lineage: "
                    f"{dataset_lineage_summary.merged_missing_lineage_count}"
                )
                print(
                    "  training-set lineage mismatches: "
                    f"{dataset_lineage_summary.set_lineage_mismatch_count}"
                )
                if dataset_lineage_summary.relationship_type_counts:
                    rel_text = ", ".join(
                        f"{k}={v}"
                        for k, v in dataset_lineage_summary.relationship_type_counts.items()
                    )
                else:
                    rel_text = "none"
                print(f"  relationship types: {rel_text}")
        if show_recordings:
            for row in recording_rows:
                print(row.recording_id)
                print(f"  type: {row.recording_type or '—'}")
                print(f"  subtype: {row.recording_subtype or '—'}")
                print(f"  behavior_mode: {row.behavior_mode or '—'}")
                print(f"  dish_design: {row.dish_design or '—'}")
                print(f"  schema: {row.artifact_schema_id or '—'}")
                print(f"  rig: {row.rig_id or '—'}")
                print(f"  arena: {row.arena_id or '—'}")
                print(f"  camera: {row.camera_id or '—'}")
                print(f"  artifacts: {row.artifact_count}")
                if row.missing_required_artifacts:
                    print("  required: MISS " + ",".join(row.missing_required_artifacts))
                else:
                    print("  required: OK")
                print(f"  started: {_format_time(row.started_utc or row.updated_utc)}")
                print(f"  path: {row.recording_path or '—'}")
        if show_recording_overview:
            print("Recording Overview")
            header = [
                "recording_id",
                "type",
                "subtype",
                "behavior",
                "dish",
                "rig",
                "arena",
                "camera",
                "datasets",
                "training",
                "analysis",
            ]
            if show_inference_col:
                header.append("inference")
            if show_export_col:
                header.append("export")
            header.extend(["active", "missing", "last_seen"])
            print("\t".join(header))
            for row in recording_overview_rows:
                cells = [
                    row.recording_id,
                    row.recording_type or "—",
                    row.recording_subtype or "—",
                    row.behavior_mode or "—",
                    row.dish_design or "—",
                    row.rig_id or "—",
                    row.arena_id or "—",
                    row.camera_id or "—",
                    str(row.dataset_count),
                    str(row.training_dataset_count),
                    str(row.analysis_dataset_count),
                ]
                if show_inference_col:
                    cells.append(str(row.inference_dataset_count))
                if show_export_col:
                    cells.append(str(row.export_dataset_count))
                cells.extend(
                    [
                        str(row.active_dataset_count),
                        str(row.missing_dataset_count),
                        _format_time(row.last_seen_utc),
                    ]
                )
                print("\t".join(cells))
        if show_recording_steps:
            print("Recording Step Overview")
            header = [
                "recording_id",
                "datasets",
                "rows",
                "ok",
                "missing",
                "absent",
                "na",
                "error",
                "blocking_steps",
                "latest_update",
            ]
            print("\t".join(header))
            for row in recording_step_overview_rows:
                cells = [
                    row.recording_id,
                    str(row.dataset_count),
                    str(row.step_rows_total),
                    str(row.ok_rows),
                    str(row.missing_rows),
                    str(row.absent_rows),
                    str(row.na_rows),
                    str(row.error_rows),
                    row.blocking_steps_csv or "—",
                    _format_time(row.latest_step_update_utc),
                ]
                print("\t".join(cells))
            if show_recording_step_details and recording_step_detail_rows:
                print("Recording Step Details")
                detail_header = [
                    "recording_id",
                    "dataset_id",
                    "use",
                    "step_name",
                    "status",
                    "run_name",
                    "method",
                    "coverage",
                    "source",
                    "updated",
                ]
                print("\t".join(detail_header))
                for row in recording_step_detail_rows:
                    detail_cells = [
                        row.recording_id or "—",
                        row.dataset_id,
                        row.zarr_use or "—",
                        row.step_name or "—",
                        _step_status_text(row.status),
                        row.run_name or "—",
                        row.method or "—",
                        (f"{float(row.coverage_pct):.1f}%" if row.coverage_pct is not None else "—"),
                        row.source or "—",
                        _format_time(row.updated_utc),
                    ]
                    print("\t".join(detail_cells))
        if show_recording_steps_wide:
            print("Recording Step Status (Wide)")
            print("\t".join(recording_step_wide_columns))
            for row in recording_step_wide_rows:
                print(
                    "\t".join(
                        row.get(column, "—")
                        for column in recording_step_wide_columns
                    )
                )
        if args.recording_summary:
            print("Recording Type/Subtype Summary")
            for row in recording_summary_rows:
                print(
                    f"  {row.recording_type or '—'} / {row.recording_subtype or '—'}: {row.count}"
                )
            if recording_vocab_rows:
                grouped = _group_recording_vocab_rows(recording_vocab_rows)
                if grouped:
                    print("  allowed subtype vocab:")
                    for rtype, subtypes in grouped.items():
                        print(f"    {rtype}: {', '.join(subtypes)}")
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
                print(
                    "  nms: "
                    + _nms_summary(
                        nms_conf=row.nms_conf,
                        nms_iou=row.nms_iou,
                        nms_topk=row.nms_topk,
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
                    "  nms: "
                    + _nms_summary(
                        nms_conf=row.nms_conf,
                        nms_iou=row.nms_iou,
                        nms_topk=row.nms_topk,
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
        if not summary_only_mode:
            print("Detect Quality")
            print(f"  total rows: {detect_quality_summary.total_rows}")
            print(
                "  passing rows "
                f"({DETECT_GATE_REVIEW_STATE}/{DETECT_GATE_REVIEW_INTENDED_USE}, "
                f"interpolated_rate<={DETECT_GATE_MAX_INTERPOLATED_RATE:.2f}, fresh mtime): "
                f"{detect_quality_summary.passing_rows}"
            )
            print(f"  excluded rows: {detect_quality_summary.excluded_rows}")
            if detect_quality_summary.exclusion_reasons:
                detect_reason_text = ", ".join(
                    f"{name}={count}"
                    for name, count in detect_quality_summary.exclusion_reasons.items()
                )
            else:
                detect_reason_text = "none"
            print(f"  top exclusion reasons: {detect_reason_text}")
            if show_detect_details and detect_quality_rows:
                print("  details:")
                detect_mtime_cache: Dict[str, Optional[int]] = {}
                for row in detect_quality_rows:
                    reason = _detect_quality_exclusion_reason(
                        row,
                        mtime_cache=detect_mtime_cache,
                    )
                    passes = _detect_quality_row_passes_default_gate(
                        row,
                        mtime_cache=detect_mtime_cache,
                    )
                    stale = _detect_quality_stale_reason(
                        row,
                        mtime_cache=detect_mtime_cache,
                    )
                    total = row.get("total_detections")
                    interpolated = row.get("interpolated_detections")
                    rate = _coerce_float(row.get("interpolated_detections_rate"))
                    print(f"    {row.get('dataset_id')}")
                    print(f"      use: {row.get('zarr_use') or '—'}")
                    print(f"      method: {row.get('detect_method') or '—'}")
                    print(
                        "      review: "
                        f"{row.get('review_state') or '—'}/{row.get('review_intended_use') or '—'}"
                    )
                    print(
                        "      interpolated/total: "
                        f"{interpolated if interpolated is not None else '—'}/"
                        f"{total if total is not None else '—'}"
                    )
                    print(
                        f"      interpolated_rate: {float(rate):.3f}"
                        if rate is not None
                        else "      interpolated_rate: —"
                    )
                    print(f"      quality_stale: {1 if stale is not None else 0}")
                    print(f"      gate: {'PASS' if passes else 'EXCLUDE'}")
                    print(f"      reason: {reason or '—'}")

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
                    print(f"      use: {row.get('zarr_use') or '—'}")
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

            print("Keypoint Profile")
            print(f"  total rows: {keypoint_profile_summary.total_rows}")
            print(
                "  stale rows (mtime mismatch/missing): "
                f"{keypoint_profile_summary.stale_rows}"
            )
            if keypoint_profile_summary.exclusion_reasons:
                keypoint_profile_reason_text = ", ".join(
                    f"{name}={count}"
                    for name, count in keypoint_profile_summary.exclusion_reasons.items()
                )
            else:
                keypoint_profile_reason_text = "none"
            print(f"  top exclusion-ish reasons: {keypoint_profile_reason_text}")

            print("Eye-Mask Quality")
            print(f"  total rows: {eye_mask_performance_summary.total_rows}")
            print(
                "  passing rows "
                f"({EYE_MASK_GATE_REVIEW_STATE}/{EYE_MASK_GATE_REVIEW_INTENDED_USE}, non-stale source): "
                f"{eye_mask_performance_summary.passing_rows}"
            )
            print(f"  excluded rows: {eye_mask_performance_summary.excluded_rows}")
            print(f"  stale rows: {eye_mask_performance_summary.stale_rows}")
            if eye_mask_performance_summary.exclusion_reasons:
                eye_mask_reason_text = ", ".join(
                    f"{name}={count}"
                    for name, count in eye_mask_performance_summary.exclusion_reasons.items()
                )
            else:
                eye_mask_reason_text = "none"
            print(f"  top exclusion reasons: {eye_mask_reason_text}")
            if eye_mask_performance_summary.review_rollups:
                eye_mask_rollup_text = ", ".join(
                    f"{name}={count}"
                    for name, count in eye_mask_performance_summary.review_rollups.items()
                )
            else:
                eye_mask_rollup_text = "none"
            print(f"  review rollups: {eye_mask_rollup_text}")
            if show_eye_mask_performance_details and eye_mask_performance_rows:
                print("  details:")
                eye_mask_mtime_cache: Dict[str, Optional[int]] = {}
                for row in eye_mask_performance_rows:
                    reason = _eye_mask_performance_exclusion_reason(
                        row,
                        mtime_cache=eye_mask_mtime_cache,
                    )
                    stale_reason = _eye_mask_performance_stale_reason(
                        row,
                        mtime_cache=eye_mask_mtime_cache,
                    )
                    passes = _eye_mask_performance_row_passes_default_gate(
                        row,
                        mtime_cache=eye_mask_mtime_cache,
                    )
                    rate = _coerce_float(row.get("successful_roi_pair_rate"))
                    print(f"    {row.get('dataset_id')}")
                    print(f"      use: {row.get('zarr_use') or '—'}")
                    print(f"      stage: {row.get('stage_group') or '—'}")
                    print(f"      method: {row.get('method') or '—'}")
                    print(
                        "      review: "
                        f"{row.get('review_state') or '—'}/{row.get('review_intended_use') or '—'}"
                    )
                    print(f"      stale: {1 if stale_reason is not None else 0}")
                    print(f"      lifecycle: {row.get('lifecycle_state') or '—'}")
                    print(f"      success_rate: {float(rate):.3f}" if rate is not None else "      success_rate: —")
                    print(f"      gate: {'PASS' if passes else 'EXCLUDE'}")
                    print(f"      reason: {reason or '—'}")

            print("Eye-Mask Profile")
            print(f"  total rows: {eye_mask_profile_summary.total_rows}")
            print(
                "  stale rows (mtime mismatch/missing): "
                f"{eye_mask_profile_summary.stale_rows}"
            )
            if eye_mask_profile_summary.exclusion_reasons:
                eye_mask_profile_reason_text = ", ".join(
                    f"{name}={count}"
                    for name, count in eye_mask_profile_summary.exclusion_reasons.items()
                )
            else:
                eye_mask_profile_reason_text = "none"
            print(f"  top exclusion reasons: {eye_mask_profile_reason_text}")
            if eye_mask_profile_summary.review_rollups:
                eye_mask_profile_rollup_text = ", ".join(
                    f"{name}={count}"
                    for name, count in eye_mask_profile_summary.review_rollups.items()
                )
            else:
                eye_mask_profile_rollup_text = "none"
            print(f"  review rollups: {eye_mask_profile_rollup_text}")
            for line in eye_mask_profile_remediation:
                print(f"  {line}")
            if show_eye_mask_profile_details and eye_mask_profile_rows:
                print("  details:")
                eye_mask_profile_mtime_cache: Dict[str, Optional[int]] = {}
                for row in eye_mask_profile_rows:
                    reason = _eye_mask_profile_exclusion_reason(
                        row,
                        mtime_cache=eye_mask_profile_mtime_cache,
                    )
                    stale = _eye_mask_profile_stale_reason(
                        row,
                        mtime_cache=eye_mask_profile_mtime_cache,
                    )
                    method = (
                        row.get("eye_mask_method")
                        or row.get("method")
                        or row.get("source_eye_masks_method")
                        or "—"
                    )
                    print(f"    {row.get('dataset_id')}")
                    print(f"      use: {row.get('zarr_use') or '—'}")
                    print(f"      method: {method}")
                    print(
                        "      review: "
                        f"{row.get('review_state') or '—'}/{row.get('review_intended_use') or '—'}"
                    )
                    print(f"      stale: {1 if stale is not None else 0}")
                    print(f"      lifecycle: {row.get('lifecycle_state') or '—'}")
                    print(f"      profile_json: {1 if _present_value(row.get('profile_json')) else 0}")
                    print(f"      gate: {'PASS' if reason is None else 'EXCLUDE'}")
                    print(f"      reason: {reason or '—'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
