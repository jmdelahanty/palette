import argparse
from decimal import Decimal, ROUND_HALF_UP
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import h5py
import numpy as np
import zarr

from fisheye.registry.db import Registry
from fisheye.shared.experiment_setup import subdish_required
from fisheye.shared.refined_detect_review import (
    DEFAULT_DETECT_GROUP_PREFERENCE,
    resolve_refined_detect_group,
)
from fisheye.tracking.single_subject_per_arena import build_tracking_qc_fields
from fisheye.shared.type_conversions import normalize_attr as _normalize_attr
from fisheye.shared.type_conversions import status_text as _status_text
from fisheye.utils.crop_quality_freshness import is_crop_quality_row_fresh
try:
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover - rich is optional
    Console = None  # type: ignore
    Table = None  # type: ignore


DEFAULT_TUNING_KEYS = [
    "dish_mask",
    "detection_tuning",
    "keypoint_tuning",
    "subject_mask_tuning",
    "eye_mask_tuning",
    "subdish_mask_tuning",
]


_STEP_STATUS_ALLOWED = {"ok", "missing", "absent", "na", "error"}
_STEP_STATUS_PRIORITY = {"error": 4, "missing": 3, "absent": 2, "na": 1, "ok": 0}
_STEP_NAME_ALIASES = {
    "raw": "raw",
    "background": "background",
    "detect": "detect",
    "refined_detect": "refined_detect",
    "crop": "crop",
    "keypoints": "keypoints",
    "refined_keypoints": "refined_keypoints",
    "eye_masks": "eye_masks",
    "refined_eye_masks": "refined_eye_masks",
    "subject_masks": "subject_masks",
    "refined_subject_masks": "refined_subject_masks",
    "arena_assignment": "arena_assignment",
    "tracks": "tracks",
    "track": "tracks",
    "stimulus": "stimulus",
    "calibration": "calibration",
    "dish_mask": "dish_mask",
    "detection_tuning": "detection_tuning",
    "keypoint_tuning": "keypoint_tuning",
    "subject_mask_tuning": "subject_mask_tuning",
    "eye_mask_tuning": "eye_mask_tuning",
    "subdish_mask_tuning": "subdish_mask_tuning",
}
_OVERVIEW_STEP_PREFIX = {
    "raw": "raw",
    "background": "background",
    "detect": "detect",
    "refined_detect": "refined_detect",
    "crop": "crop",
    "keypoints": "keypoints",
    "refined_keypoints": "refined_keypoints",
    "eye_masks": "eye_masks",
    "refined_eye_masks": "refined_eye_masks",
    "subject_masks": "subject_masks",
    "refined_subject_masks": "refined_subject_masks",
    "arena_assignment": "arena_assignment",
    "tracks": "tracks",
    "stimulus": "stimulus",
    "calibration": "calibration",
    "dish_mask": "dish_mask",
    "detection_tuning": "detection_tuning",
    "keypoint_tuning": "keypoint_tuning",
    "subject_mask_tuning": "subject_mask_tuning",
    "eye_mask_tuning": "eye_mask_tuning",
    "subdish_mask_tuning": "subdish_mask_tuning",
}

_DISPLAY_FIELD_LABELS = {
    "eye_masks": "eye_masks (legacy compat)",
    "refined_eye_masks": "refined_eye_masks (legacy compat)",
    "eye_mask_review_status": "eye_mask_review_status (legacy compat)",
    "subject_mask_components": "subject_mask_components (unified)",
    "refined_subject_mask_components": "refined_subject_mask_components (unified)",
}


@dataclass
class RecordingStatus:
    recording_dir: Path
    h5_path: Path
    camera_id: Optional[str]
    recording_id: Optional[str]
    zarr_path: Path
    zarr_use: Optional[str]
    zarr_exists: bool
    pipeline_type: Optional[str]
    zarr_purpose: Optional[str]
    has_raw_video_attr: Optional[bool]
    raw_present: bool
    full_present: bool
    ds_present: bool
    sampled_present: bool
    background_full_present: bool
    background_ds_present: bool
    detect_present: bool
    detect_method: Optional[str]
    detect_coverage: Optional[float]
    detect_coverage_basis: Optional[str]
    detect_quality_present: bool
    detect_quality_run: Optional[str]
    detect_quality_grade: Optional[str]
    detect_quality_score: Optional[float]
    detect_quality_clean_percent: Optional[float]
    detect_quality_artifacts: Optional[int]
    refined_detect_present: bool
    refined_detect_coverage: Optional[float]
    refined_detect_method: Optional[str]
    refined_detect_resolved_group: Optional[str]
    detect_review_status: Optional[Dict[str, object]]
    crop_present: bool
    crop_status: Optional[str]
    crop_review_status: Optional[Dict[str, object]]
    keypoints_present: bool
    refined_keypoints_present: bool
    refined_keypoints_coverage: Optional[float]
    refined_keypoints_success: Optional[float]
    keypoint_review_status: Optional[Dict[str, object]]
    eye_masks_present: bool
    refined_eye_masks_present: bool
    eye_mask_review_status: Optional[Dict[str, object]]
    subject_masks_present: bool
    subject_masks_coverage: Optional[float]
    subject_mask_review_status: Optional[Dict[str, object]]
    subject_mask_available_components: List[str]
    subject_mask_unavailable_components: List[str]
    subject_mask_component_review_states: Dict[str, str]
    refined_subject_masks_present: bool
    refined_subject_masks_coverage: Optional[float]
    refined_subject_mask_review_status: Optional[Dict[str, object]]
    refined_subject_mask_available_components: List[str]
    refined_subject_mask_unavailable_components: List[str]
    refined_subject_mask_component_review_states: Dict[str, str]
    arena_assignment_present: bool
    track_present: bool
    track_qc_state: Optional[str]
    track_unassigned_rows: Optional[int]
    track_unassigned_rate_percent: Optional[float]
    stimulus_runs: int
    calibration_present: bool
    tuning_present: int
    tuning_total: int
    tuning_missing: List[str]
    tuning_status: Dict[str, str]


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, bytes):
        text = value.decode("utf-8", "ignore").strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return int(value)
    if isinstance(value, bytes):
        text = value.decode("utf-8", "ignore").strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return bool(int(value))
    if isinstance(value, bytes):
        text = value.decode("utf-8", "ignore").strip().lower()
    elif isinstance(value, str):
        text = value.strip().lower()
    else:
        text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    return None


def _coerce_mapping(value: object) -> Optional[Dict[str, object]]:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.flat[0]
        else:
            try:
                return dict(value.tolist())  # type: ignore[arg-type]
            except Exception:
                return None
    if isinstance(value, dict):
        return value
    try:
        if isinstance(value, np.generic):
            value = value.item()
    except Exception:
        pass
    if hasattr(value, "items"):
        try:
            return dict(value)  # type: ignore[arg-type]
        except Exception:
            pass
    if isinstance(value, bytes):
        text = value.decode("utf-8", "ignore").strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    if isinstance(value, (list, tuple)):
        try:
            return dict(value)
        except Exception:
            return None
    return None


def _coerce_text_list(value: object) -> List[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, bytes):
        text = value.decode("utf-8", "ignore").strip()
        return [text] if text else []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            text = _normalize_attr(item)
            if text:
                out.append(text)
        return out
    return []


def _extract_subject_mask_component_fields(
    *,
    mask_labels: List[str],
    available_flags: Optional[List[bool]] = None,
    component_review_statuses: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    if not mask_labels:
        return {
            "available_components": [],
            "unavailable_components": [],
            "component_review_states": {},
        }

    flags = list(available_flags or [])
    if not flags:
        flags = [True] * len(mask_labels)
    if len(flags) < len(mask_labels):
        flags.extend([False] * (len(mask_labels) - len(flags)))
    flags = flags[: len(mask_labels)]

    available_components = [label for label, flag in zip(mask_labels, flags) if flag]
    unavailable_components = [label for label, flag in zip(mask_labels, flags) if not flag]

    review_states: Dict[str, str] = {}
    for label in mask_labels:
        payload = _coerce_mapping((component_review_statuses or {}).get(label))
        if not payload:
            continue
        state = _normalize_attr(payload.get("state") or payload.get("review_state"))
        if state:
            review_states[label] = state

    return {
        "available_components": available_components,
        "unavailable_components": unavailable_components,
        "component_review_states": review_states,
    }


def _subject_mask_component_fields_from_details(details: Optional[Dict[str, object]]) -> Dict[str, object]:
    details = details or {}
    review_states_raw = _coerce_mapping(details.get("component_review_states")) or {}
    review_states: Dict[str, str] = {}
    for component_name, raw_state in review_states_raw.items():
        component = _normalize_attr(component_name)
        state = _normalize_attr(raw_state)
        if component and state:
            review_states[component] = state
    return {
        "available_components": _coerce_text_list(details.get("available_components")),
        "unavailable_components": _coerce_text_list(details.get("unavailable_components")),
        "component_review_states": review_states,
    }


def _extract_subject_mask_coverage(group: Optional[zarr.Group]) -> Optional[float]:
    if group is None:
        return None
    for key in ("frame_counts", "n_detections"):
        counts_arr = group.get(key)
        if counts_arr is None:
            continue
        try:
            values = np.asarray(counts_arr[:])
        except Exception:
            continue
        if values.size == 0:
            continue
        return float(np.sum(values > 0)) / float(values.shape[0]) * 100.0

    metrics = group.get("metrics")
    if metrics is None:
        return None
    mask_present = metrics.get("mask_present")
    if mask_present is None:
        return None
    try:
        values = np.asarray(mask_present[:], dtype=bool)
    except Exception:
        return None
    if values.size == 0 or values.shape[0] == 0:
        return None
    return float(np.sum(np.any(values, axis=1))) / float(values.shape[0]) * 100.0


def _extract_subject_mask_run_summary(
    *,
    zarr_path: Path,
    group_path: str,
    group: zarr.Group,
    review_attr_names: List[str],
) -> Dict[str, object]:
    attrs = dict(group.attrs)
    disk_attrs = _load_group_attrs(zarr_path, group_path)
    for key, value in disk_attrs.items():
        attrs.setdefault(key, value)

    review_status = None
    for attr_name in review_attr_names:
        review_status = _coerce_mapping(attrs.get(attr_name))
        if review_status:
            break

    available_flags: List[bool] = []
    available_channels = group.get("available_channels")
    if available_channels is not None:
        try:
            available_flags = [bool(value) for value in np.asarray(available_channels[:]).tolist()]
        except Exception:
            available_flags = []

    component_fields = _extract_subject_mask_component_fields(
        mask_labels=_coerce_text_list(attrs.get("mask_labels")),
        available_flags=available_flags,
        component_review_statuses=_coerce_mapping(attrs.get("component_review_statuses")) or {},
    )
    return {
        "coverage": _extract_subject_mask_coverage(group),
        "review_status": review_status,
        **component_fields,
    }


def _base_status_payload(*, tuning_keys: List[str], zarr_exists: bool) -> Dict[str, object]:
    return {
        "zarr_exists": zarr_exists,
        "pipeline_type": None,
        "zarr_purpose": None,
        "has_raw_video_attr": None,
        "raw_present": False,
        "full_present": False,
        "ds_present": False,
        "sampled_present": False,
        "background_full_present": False,
        "background_ds_present": False,
        "detect_present": False,
        "detect_method": None,
        "detect_coverage": None,
        "detect_coverage_basis": None,
        "detect_quality_present": False,
        "detect_quality_run": None,
        "detect_quality_grade": None,
        "detect_quality_score": None,
        "detect_quality_clean_percent": None,
        "detect_quality_artifacts": None,
        "refined_detect_present": False,
        "refined_detect_coverage": None,
        "refined_detect_method": None,
        "refined_detect_resolved_group": None,
        "detect_review_status": None,
        "crop_present": False,
        "crop_status": None,
        "crop_review_status": None,
        "keypoints_present": False,
        "refined_keypoints_present": False,
        "refined_keypoints_coverage": None,
        "refined_keypoints_success": None,
        "keypoint_review_status": None,
        "eye_masks_present": False,
        "refined_eye_masks_present": False,
        "eye_mask_review_status": None,
        "subject_masks_present": False,
        "subject_masks_coverage": None,
        "subject_mask_review_status": None,
        "subject_mask_available_components": [],
        "subject_mask_unavailable_components": [],
        "subject_mask_component_review_states": {},
        "refined_subject_masks_present": False,
        "refined_subject_masks_coverage": None,
        "refined_subject_mask_review_status": None,
        "refined_subject_mask_available_components": [],
        "refined_subject_mask_unavailable_components": [],
        "refined_subject_mask_component_review_states": {},
        "arena_assignment_present": False,
        "track_present": False,
        "track_qc_state": None,
        "track_unassigned_rows": None,
        "track_unassigned_rate_percent": None,
        "stimulus_runs": 0,
        "calibration_present": False,
        "tuning_present": 0,
        "tuning_total": len(tuning_keys),
        "tuning_missing": list(tuning_keys),
        "tuning_status": {key: "miss" for key in tuning_keys},
    }


def _normalize_step_status(value: object) -> Optional[str]:
    normalized = _normalize_attr(value)
    if normalized is None:
        return None
    lowered = normalized.strip().lower()
    if lowered not in _STEP_STATUS_ALLOWED:
        return None
    return lowered


def _step_row_status_ok(row: Optional[Dict[str, object]]) -> bool:
    return _normalize_step_status(row.get("status") if row else None) == "ok"


def _parse_step_json(value: object) -> Optional[Dict[str, object]]:
    mapping = _coerce_mapping(value)
    if mapping:
        return mapping
    return None


def _normalize_tracking_qc_state(value: object) -> Optional[str]:
    normalized = _normalize_attr(value)
    if normalized is None:
        return None
    lowered = normalized.strip().lower()
    if lowered == "block":
        return "warn"
    if lowered in {"ok", "warn"}:
        return lowered
    return None


def _extract_tracking_qc_details(details: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not details:
        return {
            "track_qc_state": None,
            "track_unassigned_rows": None,
            "track_unassigned_rate_percent": None,
        }

    summary = _coerce_mapping(details.get("summary_statistics")) or {}

    n_unassigned_rows = _coerce_int(details.get("n_unassigned_rows"))
    if n_unassigned_rows is None:
        n_unassigned_rows = _coerce_int(summary.get("n_unassigned_rows"))

    unassigned_row_rate_percent = _coerce_float(details.get("unassigned_row_rate_percent"))
    if unassigned_row_rate_percent is None:
        unassigned_row_rate_percent = _coerce_float(summary.get("unassigned_row_rate_percent"))

    if unassigned_row_rate_percent is None:
        n_rows = _coerce_int(details.get("n_rows"))
        if n_rows is None:
            n_rows = _coerce_int(summary.get("n_rows"))
        if n_unassigned_rows is not None and n_rows is not None and n_rows > 0:
            unassigned_row_rate_percent = float(n_unassigned_rows) / float(n_rows) * 100.0

    track_qc_state = _normalize_tracking_qc_state(details.get("tracking_qc_state"))
    if track_qc_state is None:
        track_qc_state = _normalize_tracking_qc_state(summary.get("tracking_qc_state"))

    if n_unassigned_rows is not None:
        computed_qc = build_tracking_qc_fields(
            n_unassigned_rows=int(n_unassigned_rows),
            unassigned_row_rate_percent=unassigned_row_rate_percent,
        )
        if track_qc_state is None:
            track_qc_state = str(computed_qc["tracking_qc_state"])

    return {
        "track_qc_state": track_qc_state,
        "track_unassigned_rows": n_unassigned_rows,
        "track_unassigned_rate_percent": unassigned_row_rate_percent,
    }


def _select_step_row(rows: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not rows:
        return None

    def _key(row: Dict[str, object]) -> tuple[int, str, str, str]:
        status = _normalize_step_status(row.get("status"))
        priority = _STEP_STATUS_PRIORITY.get(status or "", -1)
        return (
            str(row.get("updated_utc") or ""),
            priority,
            str(row.get("run_name") or ""),
            str(row.get("dataset_id") or ""),
        )

    return sorted(rows, key=_key, reverse=True)[0]


def _overview_step_status(overview_row: Dict[str, object], prefix: str) -> Optional[str]:
    ok_count = _coerce_int(overview_row.get(f"{prefix}_ok_count")) or 0
    non_ok_count = _coerce_int(overview_row.get(f"{prefix}_non_ok_count")) or 0
    if non_ok_count > 0:
        return "missing"
    if ok_count > 0:
        return "ok"
    return "absent"


def _registry_step_rows_for_zarr(
    registry: Registry,
    zarr_path: Path,
    *,
    recording_id: Optional[str],
) -> List[Dict[str, object]]:
    try:
        rows = registry.conn.execute(
            """
            SELECT
                recording_id,
                dataset_id,
                zarr_use,
                step_name,
                status,
                run_name,
                method,
                coverage_pct,
                review_status_json,
                details_json,
                source,
                updated_utc
            FROM recording_step_status_latest
            WHERE zarr_path = ?
            ORDER BY COALESCE(updated_utc, '') DESC, dataset_id, step_name;
            """,
            (str(zarr_path),),
        ).fetchall()
    except Exception:
        return []
    normalized_rows = [dict(row) for row in rows]
    wanted_recording_id = _normalize_attr(recording_id)
    if wanted_recording_id is None:
        return normalized_rows
    return [
        row
        for row in normalized_rows
        if _normalize_attr(row.get("recording_id")) == wanted_recording_id
    ]


def _registry_step_overview_for_recording(
    registry: Registry,
    recording_id: Optional[str],
) -> Optional[Dict[str, object]]:
    if not recording_id:
        return None
    try:
        row = registry.conn.execute(
            """
            SELECT *
            FROM recording_step_overview
            WHERE recording_id = ?;
            """,
            (recording_id,),
        ).fetchone()
    except Exception:
        return None
    return dict(row) if row is not None else None


def _registry_status_payload(
    *,
    registry: Registry,
    zarr_path: Path,
    recording_id: Optional[str],
    tuning_keys: List[str],
) -> Dict[str, object]:
    payload = _base_status_payload(tuning_keys=tuning_keys, zarr_exists=zarr_path.exists())
    step_rows = _registry_step_rows_for_zarr(
        registry,
        zarr_path,
        recording_id=recording_id,
    )

    selected: Dict[str, List[Dict[str, object]]] = {}
    for row in step_rows:
        step_name = _normalize_attr(row.get("step_name"))
        if not step_name:
            continue
        canonical = _STEP_NAME_ALIASES.get(step_name.strip().lower())
        if not canonical:
            continue
        selected.setdefault(canonical, [])
        selected[canonical].append(row)

    selected_rows: Dict[str, Dict[str, object]] = {}
    for canonical, rows in selected.items():
        row = _select_step_row(rows)
        if row is not None:
            selected_rows[canonical] = row

    if not selected_rows:
        overview = _registry_step_overview_for_recording(registry, recording_id)
        if overview is not None:
            for canonical, prefix in _OVERVIEW_STEP_PREFIX.items():
                status = _overview_step_status(overview, prefix)
                if status is None:
                    continue
                selected_rows[canonical] = {"status": status}

    for row in selected_rows.values():
        details = _parse_step_json(row.get("details_json")) or {}
        if not details:
            continue
        if payload["pipeline_type"] is None:
            payload["pipeline_type"] = _normalize_attr(details.get("pipeline_type"))
        if payload["zarr_purpose"] is None:
            payload["zarr_purpose"] = _normalize_attr(details.get("zarr_purpose"))
        if payload["has_raw_video_attr"] is None:
            payload["has_raw_video_attr"] = _coerce_bool(details.get("has_raw_video_attr"))
        if (
            payload["pipeline_type"] is not None
            and payload["zarr_purpose"] is not None
            and payload["has_raw_video_attr"] is not None
        ):
            break

    raw_row = selected_rows.get("raw")
    raw_ok = _step_row_status_ok(raw_row)
    raw_details = _parse_step_json(raw_row.get("details_json") if raw_row else None) or {}
    raw_present_flag = _coerce_bool(raw_details.get("raw_present"))
    full_present_flag = _coerce_bool(raw_details.get("full_present"))
    ds_present_flag = _coerce_bool(raw_details.get("ds_present"))
    sampled_present_flag = _coerce_bool(raw_details.get("sampled_present"))
    payload["raw_present"] = raw_ok if raw_present_flag is None else raw_present_flag
    payload["full_present"] = raw_ok if full_present_flag is None else full_present_flag
    payload["ds_present"] = raw_ok if ds_present_flag is None else ds_present_flag
    payload["sampled_present"] = False if sampled_present_flag is None else sampled_present_flag

    background_row = selected_rows.get("background")
    background_ok = _step_row_status_ok(background_row)
    background_details = _parse_step_json(background_row.get("details_json") if background_row else None) or {}
    bg_full_present = _coerce_bool(background_details.get("full_present"))
    bg_ds_present = _coerce_bool(background_details.get("ds_present"))
    payload["background_full_present"] = background_ok if bg_full_present is None else bg_full_present
    payload["background_ds_present"] = background_ok if bg_ds_present is None else bg_ds_present

    detect_row = selected_rows.get("detect")
    detect_ok = _step_row_status_ok(detect_row)
    payload["detect_present"] = detect_ok
    payload["detect_method"] = _normalize_attr(detect_row.get("method") if detect_row else None)
    payload["detect_coverage"] = _coerce_float(detect_row.get("coverage_pct") if detect_row else None)
    payload["detect_coverage_basis"] = "registry" if detect_ok else None

    detect_details = _parse_step_json(detect_row.get("details_json") if detect_row else None) or {}
    detect_grade = _normalize_attr(
        detect_details.get("detect_quality_grade") or detect_details.get("grade")
    )
    detect_score = _coerce_float(
        detect_details.get("detect_quality_score") or detect_details.get("score")
    )
    detect_clean = _coerce_float(
        detect_details.get("detect_quality_clean_percent")
        or detect_details.get("clean_percent")
        or detect_details.get("clean_percentage")
    )
    detect_artifacts = _coerce_int(
        detect_details.get("detect_quality_artifacts")
        or detect_details.get("artifact_count")
    )
    payload["detect_quality_grade"] = detect_grade
    payload["detect_quality_score"] = detect_score
    payload["detect_quality_clean_percent"] = detect_clean
    payload["detect_quality_artifacts"] = detect_artifacts
    payload["detect_quality_present"] = any(
        value is not None for value in (detect_grade, detect_score, detect_clean, detect_artifacts)
    )
    payload["detect_quality_run"] = (
        _normalize_attr(detect_row.get("run_name") if detect_row else None)
        if payload["detect_quality_present"]
        else None
    )

    refined_detect_row = selected_rows.get("refined_detect")
    refined_detect_ok = _step_row_status_ok(refined_detect_row)
    refined_detect_coverage = _coerce_float(
        refined_detect_row.get("coverage_pct") if refined_detect_row else None
    )
    if refined_detect_ok and refined_detect_coverage is None:
        refined_detect_coverage = 100.0
    payload["refined_detect_present"] = refined_detect_ok
    payload["refined_detect_coverage"] = refined_detect_coverage
    payload["refined_detect_method"] = _normalize_attr(
        refined_detect_row.get("method") if refined_detect_row else None
    )
    refined_detect_review = _parse_step_json(
        refined_detect_row.get("review_status_json") if refined_detect_row else None
    )
    payload["detect_review_status"] = refined_detect_review
    resolved_group = _normalize_attr(
        (refined_detect_review or {}).get("resolved_group")
    )
    if not resolved_group:
        refined_detect_details = _parse_step_json(
            refined_detect_row.get("details_json") if refined_detect_row else None
        ) or {}
        resolved_group = _normalize_attr(refined_detect_details.get("resolved_group"))
    payload["refined_detect_resolved_group"] = resolved_group

    crop_row = selected_rows.get("crop")
    crop_status = _normalize_step_status(crop_row.get("status") if crop_row else None)
    crop_details = _parse_step_json(crop_row.get("details_json") if crop_row else None) or {}
    crop_run_state = _normalize_attr(crop_details.get("run_state"))
    if crop_status == "ok":
        payload["crop_present"] = True
        payload["crop_status"] = crop_run_state
    elif crop_status == "error":
        payload["crop_present"] = True
        payload["crop_status"] = "failed"
    elif crop_status == "na":
        payload["crop_present"] = True
        payload["crop_status"] = "na"
    else:
        payload["crop_present"] = False
        payload["crop_status"] = None
    payload["crop_review_status"] = _parse_step_json(
        crop_row.get("review_status_json") if crop_row else None
    )

    keypoints_row = selected_rows.get("keypoints")
    payload["keypoints_present"] = _step_row_status_ok(keypoints_row)

    refined_keypoints_row = selected_rows.get("refined_keypoints")
    refined_keypoints_ok = _step_row_status_ok(refined_keypoints_row)
    refined_keypoints_success = _coerce_float(
        refined_keypoints_row.get("coverage_pct") if refined_keypoints_row else None
    )
    refined_keypoints_details = _parse_step_json(
        refined_keypoints_row.get("details_json") if refined_keypoints_row else None
    ) or {}
    refined_keypoints_usable = _coerce_float(
        refined_keypoints_details.get("usable_keypoints_pct")
        or refined_keypoints_details.get("usable_percent")
        or refined_keypoints_details.get("train_usable_pct")
    )
    if refined_keypoints_ok and refined_keypoints_success is None:
        refined_keypoints_success = 100.0
    payload["refined_keypoints_present"] = refined_keypoints_ok
    payload["refined_keypoints_success"] = refined_keypoints_success
    payload["refined_keypoints_coverage"] = refined_keypoints_usable
    payload["keypoint_review_status"] = _parse_step_json(
        refined_keypoints_row.get("review_status_json") if refined_keypoints_row else None
    )

    eye_masks_row = selected_rows.get("eye_masks")
    payload["eye_masks_present"] = _step_row_status_ok(eye_masks_row)

    refined_eye_masks_row = selected_rows.get("refined_eye_masks")
    payload["refined_eye_masks_present"] = _step_row_status_ok(refined_eye_masks_row)
    payload["eye_mask_review_status"] = _parse_step_json(
        refined_eye_masks_row.get("review_status_json") if refined_eye_masks_row else None
    ) or _parse_step_json(
        eye_masks_row.get("review_status_json") if eye_masks_row else None
    )

    subject_masks_row = selected_rows.get("subject_masks")
    payload["subject_masks_present"] = _step_row_status_ok(subject_masks_row)
    subject_masks_coverage = _coerce_float(
        subject_masks_row.get("coverage_pct") if subject_masks_row else None
    )
    if payload["subject_masks_present"] and subject_masks_coverage is None:
        subject_masks_coverage = 100.0
    payload["subject_masks_coverage"] = subject_masks_coverage
    payload["subject_mask_review_status"] = _parse_step_json(
        subject_masks_row.get("review_status_json") if subject_masks_row else None
    )
    subject_details = _parse_step_json(
        subject_masks_row.get("details_json") if subject_masks_row else None
    ) or {}
    subject_component_fields = _subject_mask_component_fields_from_details(subject_details)
    payload["subject_mask_available_components"] = subject_component_fields["available_components"]
    payload["subject_mask_unavailable_components"] = subject_component_fields["unavailable_components"]
    payload["subject_mask_component_review_states"] = subject_component_fields["component_review_states"]

    refined_subject_masks_row = selected_rows.get("refined_subject_masks")
    payload["refined_subject_masks_present"] = _step_row_status_ok(refined_subject_masks_row)
    refined_subject_masks_coverage = _coerce_float(
        refined_subject_masks_row.get("coverage_pct") if refined_subject_masks_row else None
    )
    if payload["refined_subject_masks_present"] and refined_subject_masks_coverage is None:
        refined_subject_masks_coverage = 100.0
    payload["refined_subject_masks_coverage"] = refined_subject_masks_coverage
    payload["refined_subject_mask_review_status"] = _parse_step_json(
        refined_subject_masks_row.get("review_status_json") if refined_subject_masks_row else None
    )
    refined_subject_details = _parse_step_json(
        refined_subject_masks_row.get("details_json") if refined_subject_masks_row else None
    ) or {}
    refined_subject_component_fields = _subject_mask_component_fields_from_details(refined_subject_details)
    payload["refined_subject_mask_available_components"] = refined_subject_component_fields["available_components"]
    payload["refined_subject_mask_unavailable_components"] = refined_subject_component_fields["unavailable_components"]
    payload["refined_subject_mask_component_review_states"] = refined_subject_component_fields["component_review_states"]

    payload["arena_assignment_present"] = _step_row_status_ok(selected_rows.get("arena_assignment"))
    tracks_row = selected_rows.get("tracks")
    payload["track_present"] = _step_row_status_ok(tracks_row)
    track_details = _parse_step_json(tracks_row.get("details_json") if tracks_row else None) or {}
    tracking_qc_details = _extract_tracking_qc_details(track_details)
    payload["track_qc_state"] = tracking_qc_details["track_qc_state"]
    payload["track_unassigned_rows"] = tracking_qc_details["track_unassigned_rows"]
    payload["track_unassigned_rate_percent"] = tracking_qc_details["track_unassigned_rate_percent"]
    stimulus_row = selected_rows.get("stimulus")
    stimulus_details = _parse_step_json(stimulus_row.get("details_json") if stimulus_row else None) or {}
    stimulus_runs = _coerce_int(stimulus_details.get("stimulus_runs"))
    if stimulus_runs is None:
        stimulus_runs = 1 if _step_row_status_ok(stimulus_row) else 0
    payload["stimulus_runs"] = max(0, int(stimulus_runs))
    payload["calibration_present"] = _step_row_status_ok(selected_rows.get("calibration"))

    tuning_present = 0
    tuning_total = 0
    tuning_missing: List[str] = []
    tuning_status: Dict[str, str] = {}
    for key in tuning_keys:
        status = _normalize_step_status(selected_rows.get(key, {}).get("status"))
        if status == "ok":
            tuning_present += 1
            tuning_total += 1
            tuning_status[key] = "ok"
        elif status in {"na", "absent"}:
            tuning_status[key] = "na"
        else:
            tuning_total += 1
            tuning_status[key] = "miss"
            tuning_missing.append(key)
    payload["tuning_present"] = tuning_present
    payload["tuning_total"] = tuning_total
    payload["tuning_missing"] = tuning_missing
    payload["tuning_status"] = tuning_status

    return payload


def _build_recording_status(
    *,
    recording_dir: Path,
    h5_path: Path,
    camera_id: Optional[str],
    recording_id: Optional[str],
    zarr_path: Path,
    zarr_use: Optional[str],
    zarr_info: Dict[str, object],
) -> RecordingStatus:
    return RecordingStatus(
        recording_dir=recording_dir,
        h5_path=h5_path,
        camera_id=camera_id,
        recording_id=recording_id,
        zarr_path=zarr_path,
        zarr_use=zarr_use,
        zarr_exists=bool(zarr_info["zarr_exists"]),
        pipeline_type=zarr_info["pipeline_type"],  # type: ignore[arg-type]
        zarr_purpose=zarr_info["zarr_purpose"],  # type: ignore[arg-type]
        has_raw_video_attr=zarr_info["has_raw_video_attr"],  # type: ignore[arg-type]
        raw_present=bool(zarr_info["raw_present"]),
        full_present=bool(zarr_info["full_present"]),
        ds_present=bool(zarr_info["ds_present"]),
        sampled_present=bool(zarr_info["sampled_present"]),
        background_full_present=bool(zarr_info["background_full_present"]),
        background_ds_present=bool(zarr_info["background_ds_present"]),
        detect_present=bool(zarr_info["detect_present"]),
        detect_method=zarr_info["detect_method"],  # type: ignore[arg-type]
        detect_coverage=zarr_info["detect_coverage"],  # type: ignore[arg-type]
        detect_coverage_basis=zarr_info["detect_coverage_basis"],  # type: ignore[arg-type]
        detect_quality_present=bool(zarr_info["detect_quality_present"]),
        detect_quality_run=zarr_info["detect_quality_run"],  # type: ignore[arg-type]
        detect_quality_grade=zarr_info["detect_quality_grade"],  # type: ignore[arg-type]
        detect_quality_score=zarr_info["detect_quality_score"],  # type: ignore[arg-type]
        detect_quality_clean_percent=zarr_info["detect_quality_clean_percent"],  # type: ignore[arg-type]
        detect_quality_artifacts=zarr_info["detect_quality_artifacts"],  # type: ignore[arg-type]
        refined_detect_present=bool(zarr_info["refined_detect_present"]),
        refined_detect_coverage=zarr_info["refined_detect_coverage"],  # type: ignore[arg-type]
        refined_detect_method=zarr_info["refined_detect_method"],  # type: ignore[arg-type]
        refined_detect_resolved_group=zarr_info["refined_detect_resolved_group"],  # type: ignore[arg-type]
        detect_review_status=zarr_info["detect_review_status"],  # type: ignore[arg-type]
        crop_present=bool(zarr_info["crop_present"]),
        crop_status=zarr_info["crop_status"],  # type: ignore[arg-type]
        crop_review_status=zarr_info["crop_review_status"],  # type: ignore[arg-type]
        keypoints_present=bool(zarr_info["keypoints_present"]),
        refined_keypoints_present=bool(zarr_info["refined_keypoints_present"]),
        refined_keypoints_coverage=zarr_info["refined_keypoints_coverage"],  # type: ignore[arg-type]
        refined_keypoints_success=zarr_info["refined_keypoints_success"],  # type: ignore[arg-type]
        keypoint_review_status=zarr_info["keypoint_review_status"],  # type: ignore[arg-type]
        eye_masks_present=bool(zarr_info["eye_masks_present"]),
        refined_eye_masks_present=bool(zarr_info["refined_eye_masks_present"]),
        eye_mask_review_status=zarr_info["eye_mask_review_status"],  # type: ignore[arg-type]
        subject_masks_present=bool(zarr_info["subject_masks_present"]),
        subject_masks_coverage=zarr_info["subject_masks_coverage"],  # type: ignore[arg-type]
        subject_mask_review_status=zarr_info["subject_mask_review_status"],  # type: ignore[arg-type]
        subject_mask_available_components=list(zarr_info["subject_mask_available_components"]),  # type: ignore[arg-type]
        subject_mask_unavailable_components=list(zarr_info["subject_mask_unavailable_components"]),  # type: ignore[arg-type]
        subject_mask_component_review_states=dict(zarr_info["subject_mask_component_review_states"]),  # type: ignore[arg-type]
        refined_subject_masks_present=bool(zarr_info["refined_subject_masks_present"]),
        refined_subject_masks_coverage=zarr_info["refined_subject_masks_coverage"],  # type: ignore[arg-type]
        refined_subject_mask_review_status=zarr_info["refined_subject_mask_review_status"],  # type: ignore[arg-type]
        refined_subject_mask_available_components=list(zarr_info["refined_subject_mask_available_components"]),  # type: ignore[arg-type]
        refined_subject_mask_unavailable_components=list(zarr_info["refined_subject_mask_unavailable_components"]),  # type: ignore[arg-type]
        refined_subject_mask_component_review_states=dict(zarr_info["refined_subject_mask_component_review_states"]),  # type: ignore[arg-type]
        arena_assignment_present=bool(zarr_info["arena_assignment_present"]),
        track_present=bool(zarr_info["track_present"]),
        track_qc_state=zarr_info["track_qc_state"],  # type: ignore[arg-type]
        track_unassigned_rows=zarr_info["track_unassigned_rows"],  # type: ignore[arg-type]
        track_unassigned_rate_percent=zarr_info["track_unassigned_rate_percent"],  # type: ignore[arg-type]
        stimulus_runs=int(zarr_info["stimulus_runs"]),
        calibration_present=bool(zarr_info["calibration_present"]),
        tuning_present=int(zarr_info["tuning_present"]),
        tuning_total=int(zarr_info["tuning_total"]),
        tuning_missing=list(zarr_info["tuning_missing"]),  # type: ignore[arg-type]
        tuning_status=dict(zarr_info["tuning_status"]),  # type: ignore[arg-type]
    )


def _plan_compare_snapshot(plan: RecordingStatus, tuning_keys: List[str]) -> Dict[str, str]:
    is_production = (
        (plan.zarr_purpose == "production")
        or (plan.pipeline_type == "yolo_inference")
        or (plan.has_raw_video_attr is False and not (plan.full_present or plan.ds_present))
    )
    import_ok = None if is_production else (plan.raw_present and (plan.full_present or plan.ds_present))
    background_full_ok = None if is_production else plan.background_full_present
    background_ds_ok = None if is_production else plan.background_ds_present
    snapshot = {
        "zarr": _status_text(plan.zarr_exists),
        "import": _status_text(import_ok),
        "bg_full": _status_text(background_full_ok),
        "bg_ds": _status_text(background_ds_ok),
        "detect": _status_text(plan.detect_present),
        "refined_detect": _status_text(plan.refined_detect_coverage is not None),
        "crop": _crop_status_text(plan.crop_present, plan.crop_status),
        "keypoints": _status_text(plan.keypoints_present),
        "refined_keypoints": _status_text(plan.refined_keypoints_present),
        "eye_masks": _status_text(plan.eye_masks_present),
        "refined_eye_masks": _status_text(plan.refined_eye_masks_present),
        "subject_masks": _subject_mask_stage_text(plan.subject_masks_present, plan.subject_masks_coverage),
        "subject_mask_components": _subject_mask_component_summary_text(
            plan.subject_mask_available_components,
            plan.subject_mask_component_review_states,
        ),
        "refined_subject_masks": _subject_mask_stage_text(
            plan.refined_subject_masks_present,
            plan.refined_subject_masks_coverage,
        ),
        "refined_subject_mask_components": _subject_mask_component_summary_text(
            plan.refined_subject_mask_available_components,
            plan.refined_subject_mask_component_review_states,
        ),
        "arena_assignment": _status_text(plan.arena_assignment_present),
        "track": _track_status_text(
            plan.track_present,
            plan.track_qc_state,
            plan.track_unassigned_rows,
            plan.track_unassigned_rate_percent,
        ),
        "stimulus": _status_text(plan.stimulus_runs > 0),
        "calibration": _status_text(plan.calibration_present),
        "tuning": "N/A" if is_production else f"{plan.tuning_present}/{plan.tuning_total}",
    }
    for key in tuning_keys:
        status = "na" if is_production else plan.tuning_status.get(key, "miss")
        snapshot[f"tuning:{key}"] = _tuning_status_text(status)
    return snapshot


def _extract_coverage_from_group(group: zarr.Group) -> Optional[float]:
    frame_counts = group.get("frame_counts") or group.get("n_detections")
    if frame_counts is not None:
        try:
            counts = frame_counts[:]
        except Exception:
            counts = None
        if counts is not None:
            total = counts.shape[0]
            if total > 0:
                present = (counts > 0).sum()
                return float(present) / float(total) * 100.0
    return None


def _dataset_len(group: Optional[zarr.Group], key: str) -> Optional[int]:
    if group is None:
        return None
    arr = group.get(key)
    if arr is None:
        return None
    try:
        return int(arr.shape[0])
    except Exception:
        return None


def _extract_detect_coverage(
    detect_group: zarr.Group,
    root: Optional[zarr.Group] = None,
) -> tuple[Optional[float], Optional[str]]:
    frame_counts = detect_group.get("frame_counts") or detect_group.get("n_detections")
    if frame_counts is not None:
        try:
            counts = np.asarray(frame_counts[:])
        except Exception:
            counts = None
        if counts is not None and counts.size > 0:
            total = int(counts.shape[0])
            present = int(np.sum(counts > 0))
            basis = "full"
            raw = root.get("raw_video") if root is not None else None
            sampled_total = _dataset_len(raw, "original_frame_indices")
            if sampled_total is not None and sampled_total == total:
                basis = "sampled"
            return (float(present) / float(total) * 100.0), basis

    frame_indices = detect_group.get("frame_indices")
    if frame_indices is not None:
        try:
            values = np.asarray(frame_indices[:])
        except Exception:
            values = None
        if values is not None and values.size > 0:
            try:
                flat = values.reshape(-1).astype(np.int64, copy=False)
                total = int(flat.max()) + 1
                if total > 0:
                    present = int(np.unique(flat).size)
                    return (float(present) / float(total) * 100.0), "inferred"
            except Exception:
                return None, None

    return None, None


def _extract_detect_method(detect_group: zarr.Group) -> Optional[str]:
    direct = _normalize_attr(detect_group.attrs.get("detection_method"))
    if direct:
        return direct
    provenance = _coerce_mapping(detect_group.attrs.get("provenance"))
    if provenance is not None:
        prov_method = _normalize_attr(provenance.get("method"))
        if prov_method:
            return prov_method
    return None


def _extract_detect_quality(detect_group: zarr.Group) -> Dict[str, object]:
    quality_parent = detect_group.get("quality_reports")
    if quality_parent is None:
        return {
            "present": False,
            "run": None,
            "grade": None,
            "score": None,
            "clean_percent": None,
            "artifacts": None,
        }

    latest_quality = _normalize_attr(quality_parent.attrs.get("latest"))
    quality_candidate = None
    if latest_quality and latest_quality in quality_parent:
        quality_candidate = latest_quality
    else:
        if hasattr(quality_parent, "group_keys"):
            names = list(quality_parent.group_keys())
        else:
            names = list(quality_parent.keys())
        if names:
            quality_candidate = sorted(names)[-1]

    if not quality_candidate:
        return {
            "present": False,
            "run": None,
            "grade": None,
            "score": None,
            "clean_percent": None,
            "artifacts": None,
        }

    quality_group = quality_parent[quality_candidate]
    quality_score = _coerce_mapping(quality_group.attrs.get("quality_score")) or {}
    quality_summary = _coerce_mapping(quality_group.attrs.get("detection_quality_summary")) or {}
    blip = _coerce_int(quality_summary.get("blip_detections"))
    jump = _coerce_int(quality_summary.get("jump_detections"))
    multi = _coerce_int(quality_summary.get("multi_detections"))
    artifact_total = None
    if blip is not None or jump is not None or multi is not None:
        artifact_total = int((blip or 0) + (jump or 0) + (multi or 0))

    return {
        "present": True,
        "run": str(quality_candidate),
        "grade": _normalize_attr(quality_score.get("grade")),
        "score": _coerce_float(quality_score.get("overall_score")),
        "clean_percent": _coerce_float(quality_summary.get("clean_percentage")),
        "artifacts": artifact_total,
    }


def _sampled_total_frames(root: Optional[zarr.Group]) -> Optional[int]:
    if root is None:
        return None
    raw = root.get("raw_video")
    if raw is not None:
        if "original_frame_indices" in raw:
            return int(raw["original_frame_indices"].shape[0])
        if "images_ds" in raw:
            return int(raw["images_ds"].shape[0])
        if "images_full" in raw:
            return int(raw["images_full"].shape[0])
    return None


def _extract_refined_coverage(
    refined_group: zarr.Group,
    root: Optional[zarr.Group] = None,
) -> tuple[Optional[float], Optional[str]]:
    manual_group_name = _normalize_attr(refined_group.attrs.get("manual_review_latest"))
    if manual_group_name and manual_group_name in refined_group:
        manual_group = refined_group[manual_group_name]
        coverage = _extract_coverage_from_group(manual_group)
        if coverage is not None:
            return coverage, "manual"

    parameters = _coerce_mapping(refined_group.attrs.get("parameters"))
    refine_mode = _normalize_attr(parameters.get("refine_mode")) if parameters is not None else None
    sampled_import = bool(parameters.get("sampled_import")) if parameters is not None else False
    operations = refined_group.attrs.get("operations")
    if refine_mode == "passthrough" or operations == ["passthrough"]:
        if sampled_import:
            total_frames = refined_group.attrs.get("coverage_frames_total")
            if total_frames is None:
                total_frames = _sampled_total_frames(root)
            try:
                total_frames = int(total_frames) if total_frames is not None else None
            except Exception:
                total_frames = None
            base_group = refined_group.get("interpolated") or refined_group.get("filtered")
            if base_group is not None:
                frame_counts = base_group.get("frame_counts") or base_group.get("n_detections")
                if frame_counts is not None:
                    counts = frame_counts[:]
                    if total_frames is not None:
                        if counts.shape[0] < total_frames:
                            counts = np.pad(counts, (0, total_frames - counts.shape[0]), mode="constant")
                        elif counts.shape[0] > total_frames:
                            counts = counts[:total_frames]
                        coverage = (float(np.sum(counts > 0)) / float(total_frames)) * 100.0
                        return coverage, "passthrough"
                    if counts.shape[0] > 0:
                        coverage = (float(np.sum(counts > 0)) / float(counts.shape[0])) * 100.0
                        return coverage, "passthrough"
        coverage = None
        comparison = _coerce_mapping(refined_group.attrs.get("coverage_comparison"))
        if comparison is not None:
            original = comparison.get("original")
            if isinstance(original, dict):
                coverage = _coerce_float(original.get("coverage_percent"))
        if coverage is None:
            interp_group = refined_group.get("interpolated")
            if interp_group is not None:
                coverage = _coerce_float(interp_group.attrs.get("coverage_percent"))
        return coverage, "passthrough"

    comparison = _coerce_mapping(refined_group.attrs.get("coverage_comparison"))
    if comparison is not None:
        filtered = comparison.get("filtered")
        interpolated = comparison.get("interpolated")
        if isinstance(filtered, dict) and isinstance(interpolated, dict):
            removed = _coerce_float(filtered.get("detections_removed"))
            added = _coerce_float(interpolated.get("detections_added"))
            if removed == 0 and added == 0:
                original = comparison.get("original")
                if isinstance(original, dict):
                    coverage = _coerce_float(original.get("coverage_percent"))
                    if coverage is not None:
                        return coverage, "unchanged"

        interpolated = comparison.get("interpolated")
        if isinstance(interpolated, dict):
            coverage = _coerce_float(interpolated.get("coverage_percent"))
            if coverage is not None:
                return coverage, "interpolated"
        filtered = comparison.get("filtered")
        if isinstance(filtered, dict):
            coverage = _coerce_float(filtered.get("coverage_percent"))
            if coverage is not None:
                return coverage, "filtered"

    stats = _coerce_mapping(refined_group.attrs.get("coverage_stats"))
    if stats is not None:
        final = stats.get("final")
        if isinstance(final, dict):
            coverage = _coerce_float(final.get("coverage_percent"))
            if coverage is not None:
                return coverage, "interpolated"
        clean = stats.get("clean")
        if isinstance(clean, dict):
            coverage = _coerce_float(clean.get("coverage_percent"))
            if coverage is not None:
                return coverage, "filtered"

    interp_group = refined_group.get("interpolated")
    if interp_group is not None:
        coverage = _coerce_float(interp_group.attrs.get("coverage_percent"))
        if coverage is not None:
            return coverage, "interpolated"

    return None, None


def _pick_keypoint_summary_block(summary: Dict[str, object]) -> Dict[str, object]:
    block = summary.get("postprocess")
    if isinstance(block, dict):
        return block
    block = summary.get("refine")
    if isinstance(block, dict):
        return block
    return summary


def _extract_refined_keypoints_stats(
    refined_group: zarr.Group,
) -> tuple[Optional[float], Optional[float]]:
    summary = _coerce_mapping(refined_group.attrs.get("summary_statistics"))
    total: Optional[float] = None
    refined_success: Optional[float] = None
    usable: Optional[float] = None
    success_percent: Optional[float] = None

    if summary is not None:
        summary_block = _pick_keypoint_summary_block(summary)
        total = _coerce_float(summary_block.get("total_rois")) or _coerce_float(summary_block.get("total"))
        refined_success = _coerce_float(summary_block.get("refined_success"))
        usable = _coerce_float(summary_block.get("usable_keypoints")) or _coerce_float(summary_block.get("usable"))
        success_percent = _coerce_float(summary_block.get("success_rate_percent")) or _coerce_float(
            summary_block.get("pass_rate_percent")
        )

    if total is None:
        for key in ("refined_success", "usable_keypoints", "keypoints_roi"):
            arr = refined_group.get(key)
            if arr is not None:
                try:
                    total = float(arr.shape[0])
                    break
                except Exception:
                    continue

    if refined_success is None:
        refined_arr = refined_group.get("refined_success")
        if refined_arr is not None:
            try:
                refined_vals = np.asarray(refined_arr[:], dtype=bool)
                refined_success = float(np.sum(refined_vals))
            except Exception:
                refined_success = None

    if usable is None:
        usable_arr = refined_group.get("usable_keypoints")
        if usable_arr is not None:
            try:
                usable_vals = np.asarray(usable_arr[:], dtype=bool)
                usable = float(np.sum(usable_vals))
            except Exception:
                usable = None

    usable_percent: Optional[float] = None
    if total and total > 0:
        if success_percent is None and refined_success is not None:
            success_percent = float(refined_success) / float(total) * 100.0
        if usable is not None:
            usable_percent = float(usable) / float(total) * 100.0

    return success_percent, usable_percent


def _derive_camera_id(ipc_source_name: object) -> Optional[str]:
    if ipc_source_name is None:
        return None
    text = _normalize_attr(ipc_source_name)
    if text is None:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits if digits else None


def _read_camera_id(h5_path: Path) -> Optional[str]:
    with h5py.File(h5_path, "r") as h5:
        root = h5.attrs
        if "camera_id" in root:
            cam = _normalize_attr(root.get("camera_id"))
            if cam:
                return cam
        ipc = _normalize_attr(root.get("ipc_source_name"))
        return _derive_camera_id(ipc)


def _read_recording_id(h5_path: Path) -> Optional[str]:
    with h5py.File(h5_path, "r") as h5:
        root = h5.attrs
        for key in ("session_uuid", "session_id"):
            value = _normalize_attr(root.get(key))
            if value:
                return value
    return None


def _resolve_root(paths: Optional[List[Path]]) -> List[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [Path("/nvme1/recordings")]


def _iter_h5(paths: List[Path], recursive: bool) -> Iterable[Path]:
    for path in paths:
        path = path.expanduser()
        if path.is_file():
            if path.suffix.lower() in {".h5", ".hdf5"}:
                yield path
            continue
        if not path.exists():
            continue
        if recursive:
            yield from path.rglob("raw/*.h5")
            yield from path.rglob("raw/*.hdf5")
        else:
            yield from path.glob("*/raw/*.h5")
            yield from path.glob("*/raw/*.hdf5")


def _infer_use_from_path(zarr_path: Path) -> Optional[str]:
    name = zarr_path.name
    if name.endswith("_training.zarr"):
        return "training"
    if name.endswith("_analysis.zarr"):
        return "analysis"
    return None


def _resolve_zarr_candidates(
    *,
    recording_dir: Path,
    recording_id: Optional[str],
    requested_use: str,
    registry: Optional[Registry],
) -> List[tuple[Path, Optional[str]]]:
    candidates: Dict[str, tuple[Path, Optional[str]]] = {}

    def _accept_use(use_value: Optional[str]) -> bool:
        if requested_use == "all":
            return True
        return (use_value or "").strip().lower() == requested_use

    if registry is not None and recording_id:
        rows = registry.conn.execute(
            """
            SELECT zarr_path, zarr_use
            FROM datasets
            WHERE recording_id = ?
              AND zarr_path IS NOT NULL
              AND TRIM(zarr_path) <> '';
            """,
            (recording_id,),
        ).fetchall()
        for row in rows:
            path_text = str(row["zarr_path"]).strip()
            if not path_text:
                continue
            path = Path(path_text).expanduser()
            zarr_use = _normalize_attr(row["zarr_use"])
            if not _accept_use(zarr_use):
                continue
            candidates[str(path)] = (path, zarr_use)

    zarr_dir = recording_dir / "zarr"
    if zarr_dir.exists():
        for path in sorted(zarr_dir.glob("*.zarr")):
            inferred_use = _infer_use_from_path(path)
            if not _accept_use(inferred_use):
                continue
            key = str(path)
            if key in candidates:
                existing = candidates[key]
                candidates[key] = (existing[0], existing[1] or inferred_use)
            else:
                candidates[key] = (path, inferred_use)

    if candidates:
        return sorted(candidates.values(), key=lambda item: str(item[0]))

    recording_name = recording_dir.name
    if requested_use == "analysis":
        return [(recording_dir / "zarr" / f"{recording_name}_analysis.zarr", "analysis")]
    if requested_use == "training":
        return [(recording_dir / "zarr" / f"{recording_name}_training.zarr", "training")]
    return [
        (recording_dir / "zarr" / f"{recording_name}_analysis.zarr", "analysis"),
        (recording_dir / "zarr" / f"{recording_name}_training.zarr", "training"),
    ]


def _registry_crop_review_status_for_zarr(
    *,
    registry: Registry,
    zarr_path: Path,
) -> Optional[Dict[str, object]]:
    rows = registry.conn.execute(
        """
        SELECT
            cqc.crop_run,
            cqc.review_state,
            cqc.review_method,
            cqc.review_intended_use,
            cqc.review_reviewer,
            cqc.review_timestamp_utc,
            cqc.review_notes,
            cqc.zarr_mtime_ns
        FROM crop_quality_current cqc
        JOIN datasets d ON d.dataset_id = cqc.dataset_id
        WHERE d.zarr_path = ?
        ORDER BY
            COALESCE(cqc.review_timestamp_utc, cqc.crop_created_utc, cqc.updated_utc) DESC,
            COALESCE(cqc.crop_created_utc, '') DESC,
            cqc.crop_run DESC
        LIMIT 1;
        """,
        (str(zarr_path),),
    ).fetchall()
    if not rows:
        return None
    row = rows[0]
    is_fresh, _stale_reason = is_crop_quality_row_fresh(
        zarr_path=zarr_path,
        zarr_mtime_ns=row["zarr_mtime_ns"],
    )
    if not is_fresh:
        return None

    status: Dict[str, object] = {}
    state = _normalize_attr(row["review_state"])
    method = _normalize_attr(row["review_method"])
    intended_use = _normalize_attr(row["review_intended_use"])
    reviewer = _normalize_attr(row["review_reviewer"])
    timestamp = _normalize_attr(row["review_timestamp_utc"])
    notes = _normalize_attr(row["review_notes"])
    if state:
        status["state"] = state
    if method:
        status["method"] = method
    if intended_use:
        status["intended_use"] = intended_use
    if reviewer:
        status["reviewer"] = reviewer
    if timestamp:
        status["timestamp_utc"] = timestamp
    if notes:
        status["notes"] = notes

    return {
        "crop_run": _normalize_attr(row["crop_run"]),
        "crop_review_status": status or None,
    }


def _load_group_attrs(zarr_path: Path, group_path: str) -> Dict[str, object]:
    group_dir = zarr_path / group_path
    zarr_json = group_dir / "zarr.json"
    attrs: Dict[str, object] = {}
    if zarr_json.exists():
        try:
            data = json.loads(zarr_json.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        attrs_raw = data.get("attributes") if isinstance(data, dict) else None
        if isinstance(attrs_raw, dict):
            attrs = dict(attrs_raw)

    parent_zarr = group_dir.parent / "zarr.json"
    if parent_zarr.exists():
        try:
            parent_data = json.loads(parent_zarr.read_text(encoding="utf-8"))
        except Exception:
            parent_data = {}
        meta = None
        if isinstance(parent_data, dict):
            consolidated = parent_data.get("consolidated_metadata")
            if isinstance(consolidated, dict):
                meta = consolidated.get("metadata")
        if isinstance(meta, dict):
            entry = meta.get(group_dir.name)
            if isinstance(entry, dict):
                child_attrs = entry.get("attributes")
                if isinstance(child_attrs, dict):
                    for key, value in child_attrs.items():
                        attrs.setdefault(key, value)

    if attrs:
        return attrs
    zattrs = group_dir / ".zattrs"
    if zattrs.exists():
        try:
            data = json.loads(zattrs.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        return data if isinstance(data, dict) else {}
    return {}


def _open_root_live(zarr_path: Path) -> zarr.Group:
    try:
        return zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(zarr_path), mode="r")


def _check_zarr(zarr_path: Path, tuning_keys: List[str]) -> Dict[str, object]:
    if not zarr_path.exists():
        return _base_status_payload(tuning_keys=tuning_keys, zarr_exists=False)

    root = _open_root_live(zarr_path)
    pipeline_type = _normalize_attr(root.attrs.get("pipeline_type"))
    zarr_purpose = _normalize_attr(root.attrs.get("zarr_purpose"))
    has_raw_video_attr = root.attrs.get("has_raw_video")
    if isinstance(has_raw_video_attr, (bytes, bytearray)):
        has_raw_video_attr = has_raw_video_attr.decode("utf-8", "ignore")
    if isinstance(has_raw_video_attr, str):
        if has_raw_video_attr.lower() in {"true", "1", "yes"}:
            has_raw_video_attr = True
        elif has_raw_video_attr.lower() in {"false", "0", "no"}:
            has_raw_video_attr = False
        else:
            has_raw_video_attr = None

    raw = root.get("raw_video")
    raw_present = raw is not None
    full_present = raw_present and "images_full" in raw
    ds_present = raw_present and "images_ds" in raw
    sampled_present = raw_present and "original_frame_indices" in raw

    background_full_present = False
    background_ds_present = False
    bg_runs = root.get("background_runs")
    if bg_runs is not None:
        latest_bg = bg_runs.attrs.get("latest")
        if latest_bg and latest_bg in bg_runs:
            latest_group = bg_runs[latest_bg]
            background_full_present = "background_full" in latest_group
            background_ds_present = "background_ds" in latest_group
    if not (background_full_present and background_ds_present):
        legacy_bg = root.get("background")
        if legacy_bg is not None:
            background_full_present = "background_full" in legacy_bg
            background_ds_present = "background_ds" in legacy_bg

    detect_present = False
    detect_method: Optional[str] = None
    detect_coverage: Optional[float] = None
    detect_coverage_basis: Optional[str] = None
    detect_quality_present = False
    detect_quality_run: Optional[str] = None
    detect_quality_grade: Optional[str] = None
    detect_quality_score: Optional[float] = None
    detect_quality_clean_percent: Optional[float] = None
    detect_quality_artifacts: Optional[int] = None
    detect_parent = root.get("detect_runs")
    if detect_parent is not None:
        latest_detect = detect_parent.attrs.get("latest")
        detect_candidate = None
        if latest_detect and latest_detect in detect_parent:
            detect_candidate = latest_detect
        else:
            if hasattr(detect_parent, "group_keys"):
                names = list(detect_parent.group_keys())
            else:
                names = list(detect_parent.keys())
            if names:
                detect_candidate = sorted(names)[-1]
        if detect_candidate:
            detect_present = True
            detect_group = detect_parent[detect_candidate]
            detect_method = _extract_detect_method(detect_group)
            detect_coverage, detect_coverage_basis = _extract_detect_coverage(detect_group, root)
            quality_info = _extract_detect_quality(detect_group)
            detect_quality_present = bool(quality_info["present"])
            detect_quality_run = _normalize_attr(quality_info["run"])
            detect_quality_grade = _normalize_attr(quality_info["grade"])
            detect_quality_score = _coerce_float(quality_info["score"])
            detect_quality_clean_percent = _coerce_float(quality_info["clean_percent"])
            detect_quality_artifacts = _coerce_int(quality_info["artifacts"])

    refined_detect_present = False
    refined_detect_coverage: Optional[float] = None
    refined_detect_method: Optional[str] = None
    refined_detect_resolved_group: Optional[str] = None
    detect_review_status: Optional[Dict[str, object]] = None
    refined_parent = root.get("refined_detect_runs") or root.get("refined_runs")
    if refined_parent is not None:
        latest_refined = refined_parent.attrs.get("latest")
        candidate_run = None
        if latest_refined and latest_refined in refined_parent:
            candidate_run = latest_refined
        else:
            if hasattr(refined_parent, "group_keys"):
                names = list(refined_parent.group_keys())
            else:
                names = list(refined_parent.keys())
            if names:
                candidate_run = sorted(names)[-1]
        if candidate_run:
            refined_detect_present = True
            refined_group = refined_parent[candidate_run]
            refined_detect_coverage, refined_detect_method = _extract_refined_coverage(refined_group, root)
            resolution = resolve_refined_detect_group(
                refined_group, preference=DEFAULT_DETECT_GROUP_PREFERENCE
            )
            refined_detect_resolved_group = resolution.group or resolution.label
            detect_review_status = _coerce_mapping(refined_group.attrs.get("detect_review_status"))

    crop_present = False
    crop_status: Optional[str] = None
    crop_review_status: Optional[Dict[str, object]] = None
    crop_parent = root.get("crop_runs")
    if crop_parent is not None:
        latest_crop = crop_parent.attrs.get("latest")
        if latest_crop and latest_crop in crop_parent:
            crop_present = True
            crop_status = _normalize_attr(crop_parent[latest_crop].attrs.get("status"))
            crop_review_status = _coerce_mapping(
                crop_parent[latest_crop].attrs.get("crop_review_status")
            )
        else:
            if hasattr(crop_parent, "group_keys"):
                names = list(crop_parent.group_keys())
            else:
                names = list(crop_parent.keys())
            crop_present = len(names) > 0
            if names:
                fallback_crop = sorted(names)[-1]
                try:
                    crop_status = _normalize_attr(crop_parent[fallback_crop].attrs.get("status"))
                except Exception:
                    crop_status = None

    keypoints_present = False
    keypoints_parent = root.get("keypoints_runs")
    if keypoints_parent is not None:
        latest_keypoints = _normalize_attr(keypoints_parent.attrs.get("latest"))
        if latest_keypoints and latest_keypoints in keypoints_parent:
            keypoints_present = True
        else:
            if hasattr(keypoints_parent, "group_keys"):
                keypoints_present = len(list(keypoints_parent.group_keys())) > 0
            else:
                keypoints_present = len(list(keypoints_parent.keys())) > 0

    refined_keypoints_present = False
    refined_keypoints_coverage: Optional[float] = None
    refined_keypoints_success: Optional[float] = None
    keypoint_review_status: Optional[Dict[str, object]] = None
    refined_keypoints_parent = root.get("refined_keypoints_runs") or root.get("keypoints_refined_runs")
    refined_keypoints_group_name = (
        "refined_keypoints_runs" if "refined_keypoints_runs" in root else "keypoints_refined_runs"
    )
    if refined_keypoints_parent is not None:
        latest_refined_keypoints = _normalize_attr(refined_keypoints_parent.attrs.get("latest"))
        candidate_run = None
        if latest_refined_keypoints and latest_refined_keypoints in refined_keypoints_parent:
            candidate_run = latest_refined_keypoints
        else:
            if hasattr(refined_keypoints_parent, "group_keys"):
                names = list(refined_keypoints_parent.group_keys())
            else:
                names = list(refined_keypoints_parent.keys())
            if names:
                candidate_run = sorted(names)[-1]
        if candidate_run:
            refined_keypoints_present = True
            refined_kp_group = refined_keypoints_parent[candidate_run]
            refined_keypoints_success, refined_keypoints_coverage = _extract_refined_keypoints_stats(refined_kp_group)
            keypoint_review_status = _coerce_mapping(
                refined_kp_group.attrs.get("keypoint_review_status")
            )
            if not keypoint_review_status:
                attrs = _load_group_attrs(
                    zarr_path, f"{refined_keypoints_group_name}/{candidate_run}"
                )
                keypoint_review_status = _coerce_mapping(attrs.get("keypoint_review_status"))
            if not keypoint_review_status:
                review_latest = _normalize_attr(
                    refined_keypoints_parent.attrs.get("keypoint_review_status_latest")
                )
                if review_latest:
                    attrs = _load_group_attrs(
                        zarr_path, f"{refined_keypoints_group_name}/{review_latest}"
                    )
                    keypoint_review_status = _coerce_mapping(attrs.get("keypoint_review_status"))
        else:
            if hasattr(refined_keypoints_parent, "group_keys"):
                refined_keypoints_present = len(list(refined_keypoints_parent.group_keys())) > 0
            else:
                refined_keypoints_present = len(list(refined_keypoints_parent.keys())) > 0

    eye_masks_present = False
    eye_masks_parent = root.get("eye_masks_runs")
    if eye_masks_parent is not None:
        latest_eye_masks = eye_masks_parent.attrs.get("latest")
        if latest_eye_masks and latest_eye_masks in eye_masks_parent:
            eye_masks_present = True
        else:
            if hasattr(eye_masks_parent, "group_keys"):
                eye_masks_present = len(list(eye_masks_parent.group_keys())) > 0
            else:
                eye_masks_present = len(list(eye_masks_parent.keys())) > 0

    refined_eye_masks_present = False
    eye_mask_review_status: Optional[Dict[str, object]] = None
    refined_eye_masks_parent = root.get("refined_eye_masks_runs")
    if refined_eye_masks_parent is not None:
        latest_refined_eye_masks = _normalize_attr(refined_eye_masks_parent.attrs.get("latest"))
        candidate_run = None
        if latest_refined_eye_masks and latest_refined_eye_masks in refined_eye_masks_parent:
            candidate_run = latest_refined_eye_masks
        else:
            if hasattr(refined_eye_masks_parent, "group_keys"):
                names = list(refined_eye_masks_parent.group_keys())
            else:
                names = list(refined_eye_masks_parent.keys())
            if names:
                candidate_run = sorted(names)[-1]
        if candidate_run:
            refined_eye_masks_present = True
            refined_eye_masks_group = refined_eye_masks_parent[candidate_run]
            eye_mask_review_status = _coerce_mapping(
                refined_eye_masks_group.attrs.get("eye_mask_review_status")
            )
            if not eye_mask_review_status:
                review_latest = _normalize_attr(
                    refined_eye_masks_parent.attrs.get("eye_mask_review_status_latest")
                )
                if review_latest and review_latest in refined_eye_masks_parent:
                    eye_mask_review_status = _coerce_mapping(
                        refined_eye_masks_parent[review_latest].attrs.get("eye_mask_review_status")
                    )
        else:
            if hasattr(refined_eye_masks_parent, "group_keys"):
                refined_eye_masks_present = len(list(refined_eye_masks_parent.group_keys())) > 0
            else:
                refined_eye_masks_present = len(list(refined_eye_masks_parent.keys())) > 0

    subject_masks_present = False
    subject_masks_coverage: Optional[float] = None
    subject_mask_review_status: Optional[Dict[str, object]] = None
    subject_mask_available_components: List[str] = []
    subject_mask_unavailable_components: List[str] = []
    subject_mask_component_review_states: Dict[str, str] = {}
    subject_masks_parent = root.get("subject_mask_runs")
    if subject_masks_parent is not None:
        latest_subject_masks = _normalize_attr(subject_masks_parent.attrs.get("latest"))
        candidate_run = None
        if latest_subject_masks and latest_subject_masks in subject_masks_parent:
            candidate_run = latest_subject_masks
        else:
            if hasattr(subject_masks_parent, "group_keys"):
                names = list(subject_masks_parent.group_keys())
            else:
                names = list(subject_masks_parent.keys())
            if names:
                candidate_run = sorted(names)[-1]
        if candidate_run:
            subject_masks_present = True
            subject_masks_group = subject_masks_parent[candidate_run]
            subject_summary = _extract_subject_mask_run_summary(
                zarr_path=zarr_path,
                group_path=f"subject_mask_runs/{candidate_run}",
                group=subject_masks_group,
                review_attr_names=["subject_mask_review_status"],
            )
            subject_masks_coverage = subject_summary["coverage"]  # type: ignore[assignment]
            subject_mask_review_status = subject_summary["review_status"]  # type: ignore[assignment]
            subject_mask_available_components = list(subject_summary["available_components"])  # type: ignore[arg-type]
            subject_mask_unavailable_components = list(subject_summary["unavailable_components"])  # type: ignore[arg-type]
            subject_mask_component_review_states = dict(subject_summary["component_review_states"])  # type: ignore[arg-type]
        else:
            if hasattr(subject_masks_parent, "group_keys"):
                subject_masks_present = len(list(subject_masks_parent.group_keys())) > 0
            else:
                subject_masks_present = len(list(subject_masks_parent.keys())) > 0

    refined_subject_masks_present = False
    refined_subject_masks_coverage: Optional[float] = None
    refined_subject_mask_review_status: Optional[Dict[str, object]] = None
    refined_subject_mask_available_components: List[str] = []
    refined_subject_mask_unavailable_components: List[str] = []
    refined_subject_mask_component_review_states: Dict[str, str] = {}
    refined_subject_masks_parent = root.get("refined_subject_masks_runs")
    if refined_subject_masks_parent is not None:
        latest_refined_subject_masks = _normalize_attr(refined_subject_masks_parent.attrs.get("latest"))
        candidate_run = None
        if latest_refined_subject_masks and latest_refined_subject_masks in refined_subject_masks_parent:
            candidate_run = latest_refined_subject_masks
        else:
            if hasattr(refined_subject_masks_parent, "group_keys"):
                names = list(refined_subject_masks_parent.group_keys())
            else:
                names = list(refined_subject_masks_parent.keys())
            if names:
                candidate_run = sorted(names)[-1]
        if candidate_run:
            refined_subject_masks_present = True
            refined_subject_masks_group = refined_subject_masks_parent[candidate_run]
            refined_subject_summary = _extract_subject_mask_run_summary(
                zarr_path=zarr_path,
                group_path=f"refined_subject_masks_runs/{candidate_run}",
                group=refined_subject_masks_group,
                review_attr_names=["refined_subject_mask_review_status", "subject_mask_review_status"],
            )
            refined_subject_masks_coverage = refined_subject_summary["coverage"]  # type: ignore[assignment]
            refined_subject_mask_review_status = refined_subject_summary["review_status"]  # type: ignore[assignment]
            refined_subject_mask_available_components = list(refined_subject_summary["available_components"])  # type: ignore[arg-type]
            refined_subject_mask_unavailable_components = list(refined_subject_summary["unavailable_components"])  # type: ignore[arg-type]
            refined_subject_mask_component_review_states = dict(refined_subject_summary["component_review_states"])  # type: ignore[arg-type]
        else:
            if hasattr(refined_subject_masks_parent, "group_keys"):
                refined_subject_masks_present = len(list(refined_subject_masks_parent.group_keys())) > 0
            else:
                refined_subject_masks_present = len(list(refined_subject_masks_parent.keys())) > 0

    arena_assignment_present = False
    arena_assignment_parent = root.get("arena_assignment_runs")
    if arena_assignment_parent is not None:
        latest_assign = arena_assignment_parent.attrs.get("latest")
        if latest_assign and latest_assign in arena_assignment_parent:
            arena_assignment_present = True
        else:
            if hasattr(arena_assignment_parent, "group_keys"):
                arena_assignment_present = len(list(arena_assignment_parent.group_keys())) > 0
            else:
                arena_assignment_present = len(list(arena_assignment_parent.keys())) > 0

    track_present = False
    track_qc_state: Optional[str] = None
    track_unassigned_rows: Optional[int] = None
    track_unassigned_rate_percent: Optional[float] = None
    track_parent = root.get("tracking_runs")
    if track_parent is not None:
        latest_track = _normalize_attr(track_parent.attrs.get("latest"))
        candidate_run = None
        if latest_track and latest_track in track_parent:
            candidate_run = latest_track
        else:
            if hasattr(track_parent, "group_keys"):
                names = list(track_parent.group_keys())
            else:
                names = list(track_parent.keys())
            if names:
                candidate_run = sorted(names)[-1]
        if candidate_run:
            track_present = True
            track_group = track_parent[candidate_run]
            track_attrs = _coerce_mapping(track_group.attrs) or {}
            tracking_qc_details = _extract_tracking_qc_details(track_attrs)
            track_qc_state = tracking_qc_details["track_qc_state"]  # type: ignore[assignment]
            track_unassigned_rows = tracking_qc_details["track_unassigned_rows"]  # type: ignore[assignment]
            track_unassigned_rate_percent = tracking_qc_details["track_unassigned_rate_percent"]  # type: ignore[assignment]
        else:
            if hasattr(track_parent, "group_keys"):
                track_present = len(list(track_parent.group_keys())) > 0
            else:
                track_present = len(list(track_parent.keys())) > 0

    stim_runs = 0
    analysis = root.get("analysis")
    if analysis is not None and "stimulus_runs" in analysis:
        stim_group = analysis["stimulus_runs"]
        if hasattr(stim_group, "group_keys"):
            stim_runs = len(list(stim_group.group_keys()))
        else:
            stim_runs = len(list(stim_group.keys()))

    calibration_present = "calibration" in root

    subdish_needed = subdish_required(root.attrs)

    tuning_missing: List[str] = []
    tuning_present = 0
    tuning_total = 0
    tuning_status: Dict[str, str] = {}
    analysis_meta = root.get("analysis_metadata")
    attrs = analysis_meta.attrs if analysis_meta is not None else {}
    for key in tuning_keys:
        if key in attrs:
            tuning_present += 1
            tuning_total += 1
            tuning_status[key] = "ok"
            continue
        if key == "subdish_mask_tuning" and not subdish_needed:
            tuning_status[key] = "na"
            continue
        tuning_total += 1
        tuning_status[key] = "miss"
        if tuning_status[key] == "miss":
            tuning_missing.append(key)

    return {
        "zarr_exists": True,
        "pipeline_type": pipeline_type,
        "zarr_purpose": zarr_purpose,
        "has_raw_video_attr": has_raw_video_attr,
        "raw_present": raw_present,
        "full_present": full_present,
        "ds_present": ds_present,
        "sampled_present": sampled_present,
        "background_full_present": background_full_present,
        "background_ds_present": background_ds_present,
        "detect_present": detect_present,
        "detect_method": detect_method,
        "detect_coverage": detect_coverage,
        "detect_coverage_basis": detect_coverage_basis,
        "detect_quality_present": detect_quality_present,
        "detect_quality_run": detect_quality_run,
        "detect_quality_grade": detect_quality_grade,
        "detect_quality_score": detect_quality_score,
        "detect_quality_clean_percent": detect_quality_clean_percent,
        "detect_quality_artifacts": detect_quality_artifacts,
        "refined_detect_present": refined_detect_present,
        "refined_detect_coverage": refined_detect_coverage,
        "refined_detect_method": refined_detect_method,
        "refined_detect_resolved_group": refined_detect_resolved_group,
        "detect_review_status": detect_review_status,
        "crop_present": crop_present,
        "crop_status": crop_status,
        "crop_review_status": crop_review_status,
        "keypoints_present": keypoints_present,
        "refined_keypoints_present": refined_keypoints_present,
        "refined_keypoints_coverage": refined_keypoints_coverage,
        "refined_keypoints_success": refined_keypoints_success,
        "keypoint_review_status": keypoint_review_status,
        "eye_masks_present": eye_masks_present,
        "refined_eye_masks_present": refined_eye_masks_present,
        "eye_mask_review_status": eye_mask_review_status,
        "subject_masks_present": subject_masks_present,
        "subject_masks_coverage": subject_masks_coverage,
        "subject_mask_review_status": subject_mask_review_status,
        "subject_mask_available_components": subject_mask_available_components,
        "subject_mask_unavailable_components": subject_mask_unavailable_components,
        "subject_mask_component_review_states": subject_mask_component_review_states,
        "refined_subject_masks_present": refined_subject_masks_present,
        "refined_subject_masks_coverage": refined_subject_masks_coverage,
        "refined_subject_mask_review_status": refined_subject_mask_review_status,
        "refined_subject_mask_available_components": refined_subject_mask_available_components,
        "refined_subject_mask_unavailable_components": refined_subject_mask_unavailable_components,
        "refined_subject_mask_component_review_states": refined_subject_mask_component_review_states,
        "arena_assignment_present": arena_assignment_present,
        "track_present": track_present,
        "track_qc_state": track_qc_state,
        "track_unassigned_rows": track_unassigned_rows,
        "track_unassigned_rate_percent": track_unassigned_rate_percent,
        "stimulus_runs": stim_runs,
        "calibration_present": calibration_present,
        "tuning_present": tuning_present,
        "tuning_total": tuning_total,
        "tuning_missing": tuning_missing,
        "tuning_status": tuning_status,
    }


def _tuning_status_text(value: str) -> str:
    if value == "ok":
        return "OK"
    if value == "na":
        return "N/A"
    return "MISS"


def _status_rich(value: Optional[bool]) -> str:
    if value is None:
        return "N/A"
    return "[chartreuse1]OK[/chartreuse1]" if value else "[red]MISS[/red]"


def _crop_status_text(present: bool, status: Optional[str]) -> str:
    if not present:
        return "MISS"
    normalized = str(status or "").strip().lower()
    return normalized or "OK"


def _format_one_decimal(value: float) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _track_status_text(
    present: bool,
    track_qc_state: Optional[str],
    n_unassigned_rows: Optional[int],
    unassigned_row_rate_percent: Optional[float],
) -> str:
    if not present:
        return "MISS"
    normalized_state = _normalize_tracking_qc_state(track_qc_state)
    if normalized_state == "warn":
        if n_unassigned_rows is not None and unassigned_row_rate_percent is not None:
            return f"WARN ({n_unassigned_rows} unassigned, {_format_one_decimal(unassigned_row_rate_percent)}%)"
        if n_unassigned_rows is not None:
            return f"WARN ({n_unassigned_rows} unassigned)"
        return "WARN"
    if n_unassigned_rows is None or n_unassigned_rows <= 0:
        return "OK"
    if unassigned_row_rate_percent is not None:
        return f"WARN ({n_unassigned_rows} unassigned, {_format_one_decimal(unassigned_row_rate_percent)}%)"
    return f"WARN ({n_unassigned_rows} unassigned)"


def _track_status_rich(
    present: bool,
    track_qc_state: Optional[str],
    n_unassigned_rows: Optional[int],
    unassigned_row_rate_percent: Optional[float],
) -> str:
    if not present:
        return "[red]MISS[/red]"
    normalized_state = _normalize_tracking_qc_state(track_qc_state)
    if normalized_state == "warn":
        if n_unassigned_rows is not None and unassigned_row_rate_percent is not None:
            return (
                f"[yellow]WARN[/yellow] ({n_unassigned_rows} unassigned, "
                f"{_format_one_decimal(unassigned_row_rate_percent)}%)"
            )
        if n_unassigned_rows is not None:
            return f"[yellow]WARN[/yellow] ({n_unassigned_rows} unassigned)"
        return "[yellow]WARN[/yellow]"
    if n_unassigned_rows is None or n_unassigned_rows <= 0:
        return "[chartreuse1]OK[/chartreuse1]"
    if unassigned_row_rate_percent is not None:
        return (
            f"[yellow]WARN[/yellow] ({n_unassigned_rows} unassigned, "
            f"{_format_one_decimal(unassigned_row_rate_percent)}%)"
        )
    return f"[yellow]WARN[/yellow] ({n_unassigned_rows} unassigned)"


def _crop_status_rich(present: bool, status: Optional[str]) -> str:
    if not present:
        return "[red]MISS[/red]"
    normalized = str(status or "").strip().lower()
    if normalized == "completed":
        return "[chartreuse1]completed[/chartreuse1]"
    if normalized == "running":
        return "[yellow]running[/yellow]"
    if normalized == "failed":
        return "[red]failed[/red]"
    if normalized:
        return f"[dim]{normalized}[/dim]"
    return "[chartreuse1]OK[/chartreuse1]"


def _tuning_status_rich(value: str) -> str:
    if value == "ok":
        return "[chartreuse1]OK[/chartreuse1]"
    if value == "na":
        return "N/A"
    return "[red]MISS[/red]"


def _percent_text(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    if value >= 99.999:
        return "100%"
    return f"{value:.1f}%"


def _percent_rich(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    if value >= 99.999:
        return "[chartreuse1]100%[/chartreuse1]"
    return f"[yellow]{value:.1f}%[/yellow]"


def _refined_status_text(coverage: Optional[float], method: Optional[str]) -> str:
    if coverage is None:
        return "MISS"
    percent = _percent_text(coverage) or "—"
    if method:
        return f"{percent} ({method})"
    return percent


def _detect_status_text(
    present: bool,
    method: Optional[str],
    coverage: Optional[float],
    basis: Optional[str],
) -> str:
    if not present:
        return "MISS"
    percent = _percent_text(coverage)
    if percent is None and method:
        return f"OK ({method})"
    if percent is None:
        return "OK"
    if basis and method:
        return f"OK ({percent}, {basis}, {method})"
    if method:
        return f"OK ({percent}, {method})"
    if basis:
        return f"OK ({percent}, {basis})"
    return f"OK ({percent})"


def _detect_status_rich(
    present: bool,
    method: Optional[str],
    coverage: Optional[float],
    basis: Optional[str],
) -> str:
    if not present:
        return "[red]MISS[/red]"
    percent = _percent_rich(coverage)
    method_text = f"[dim]{method}[/dim]" if method else None
    if percent is None and method_text is not None:
        return f"[chartreuse1]OK[/chartreuse1] ({method_text})"
    if percent is None:
        return "[chartreuse1]OK[/chartreuse1]"
    if basis and method_text is not None:
        return f"[chartreuse1]OK[/chartreuse1] ({percent}, [dim]{basis}[/dim], {method_text})"
    if method_text is not None:
        return f"[chartreuse1]OK[/chartreuse1] ({percent}, {method_text})"
    if basis:
        return f"[chartreuse1]OK[/chartreuse1] ({percent}, [dim]{basis}[/dim])"
    return f"[chartreuse1]OK[/chartreuse1] ({percent})"


def _refined_status_rich(coverage: Optional[float], method: Optional[str]) -> str:
    if coverage is None:
        return "[red]MISS[/red]"
    percent = _percent_rich(coverage) or "[dim]—[/dim]"
    if method:
        return f"{percent} [dim]({method})[/dim]"
    return percent


def _detect_quality_status_text(
    present: bool,
    grade: Optional[str],
    score: Optional[float],
    clean_percent: Optional[float],
    artifacts: Optional[int],
) -> str:
    if not present:
        return "MISS"
    parts: List[str] = []
    if grade and score is not None:
        parts.append(f"{grade} {score:.1f}")
    elif grade:
        parts.append(grade)
    elif score is not None:
        parts.append(f"{score:.1f}")
    if clean_percent is not None:
        parts.append(f"clean {clean_percent:.1f}%")
    if artifacts is not None:
        parts.append(f"art {artifacts}")
    if parts:
        return f"OK ({', '.join(parts)})"
    return "OK"


def _detect_quality_status_rich(
    present: bool,
    grade: Optional[str],
    score: Optional[float],
    clean_percent: Optional[float],
    artifacts: Optional[int],
) -> str:
    text = _detect_quality_status_text(
        present=present,
        grade=grade,
        score=score,
        clean_percent=clean_percent,
        artifacts=artifacts,
    )
    if not present:
        return "[red]MISS[/red]"
    if grade == "A":
        return f"[chartreuse1]{text}[/chartreuse1]"
    if grade in {"B", "C"}:
        return f"[yellow]{text}[/yellow]"
    if grade in {"D", "F"}:
        return f"[red]{text}[/red]"
    return f"[chartreuse1]{text}[/chartreuse1]"


def _keypoint_status_text(success: Optional[float], usable: Optional[float]) -> str:
    if success is None and usable is None:
        return "MISS"
    success_text = _percent_text(success) or "—"
    if usable is None:
        return success_text
    usable_text = _percent_text(usable) or "—"
    return f"{success_text} (train {usable_text})"


def _keypoint_status_rich(success: Optional[float], usable: Optional[float]) -> str:
    if success is None and usable is None:
        return "[red]MISS[/red]"
    success_text = _percent_rich(success) or "[dim]—[/dim]"
    if usable is None:
        return success_text
    usable_text = _percent_rich(usable) or "[dim]—[/dim]"
    return f"{success_text} (train {usable_text})"


def _subject_mask_stage_text(present: bool, coverage: Optional[float]) -> str:
    if not present:
        return "MISS"
    percent = _percent_text(coverage)
    if percent is None:
        return "OK"
    return f"OK ({percent})"


def _subject_mask_stage_rich(present: bool, coverage: Optional[float]) -> str:
    if not present:
        return "[red]MISS[/red]"
    percent = _percent_rich(coverage)
    if percent is None:
        return "[chartreuse1]OK[/chartreuse1]"
    return f"[chartreuse1]OK[/chartreuse1] ({percent})"


_SUBJECT_COMPONENT_LABELS = {
    "subject_body": "body",
    "eye_left": "eye_l",
    "eye_right": "eye_r",
    "eyes_union": "eyes",
    "swim_bladder": "swim",
}
_SUBJECT_REVIEW_LABELS = {
    "approved": "appr",
    "needs_review": "review",
    "pending": "pend",
    "rejected": "rej",
}


def _subject_mask_component_summary_text(
    available_components: List[str],
    review_states: Dict[str, str],
) -> str:
    if not available_components:
        return "—"
    parts: List[str] = []
    for component_name in available_components:
        label = _SUBJECT_COMPONENT_LABELS.get(component_name, component_name)
        state = _normalize_attr(review_states.get(component_name))
        if state:
            label = f"{label}={_SUBJECT_REVIEW_LABELS.get(state, state)}"
        parts.append(label)
    return ", ".join(parts)


def _subject_mask_component_summary_rich(
    available_components: List[str],
    review_states: Dict[str, str],
) -> str:
    text = _subject_mask_component_summary_text(available_components, review_states)
    if text == "—":
        return "[dim]—[/dim]"
    return text


def _review_status_text(status: Optional[Dict[str, object]]) -> str:
    if not status:
        return "—"
    state = str(status.get("state", "")).strip()
    method = str(status.get("method", "")).strip()
    intended_use = str(status.get("intended_use", "")).strip()
    resolved_group = str(status.get("resolved_group", "")).strip()
    parts: List[str] = []
    if method:
        parts.append(method)
    if intended_use:
        parts.append(intended_use)
    if resolved_group:
        parts.append(f"group={resolved_group}")
    label = state or "review"
    if parts:
        return f"{label} ({', '.join(parts)})"
    if state:
        return state
    return "—"


def _review_status_rich(status: Optional[Dict[str, object]]) -> str:
    if not status:
        return "[dim]—[/dim]"
    state = str(status.get("state", "")).strip().lower()
    label = _review_status_text(status)
    if state == "approved":
        return f"[chartreuse1]{label}[/chartreuse1]"
    if state in ("rejected", "fail", "failed"):
        return f"[red]{label}[/red]"
    if state in ("pending", "needs_review", "review"):
        return f"[yellow]{label}[/yellow]"
    return f"[dim]{label}[/dim]"


def _resolved_group_text(group: Optional[str]) -> str:
    return group or "—"


def _resolved_group_rich(group: Optional[str]) -> str:
    if not group:
        return "[dim]—[/dim]"
    return group


def _display_field_label(field: str) -> str:
    return _DISPLAY_FIELD_LABELS.get(field, field)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check which processing steps have been completed for recordings. "
            "Legacy eye stages are reported separately for transition diagnostics; "
            "unified eye/body/swim availability lives in the subject-mask component summaries."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording root(s) to scan (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan for recordings under each root.",
    )
    parser.add_argument(
        "--tuning-keys",
        type=str,
        help="Comma-separated tuning keys to check (default: dish_mask,detection_tuning,keypoint_tuning,eye_mask_tuning,subdish_mask_tuning).",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable rich table output.",
    )
    parser.add_argument(
        "--zarr-use",
        choices=("all", "training", "analysis"),
        default="all",
        help="Filter status rows by zarr use type (default: all).",
    )
    parser.add_argument(
        "--status-source",
        choices=("filesystem", "registry", "compare"),
        default="filesystem",
        help="Status source: filesystem (default), registry, or compare.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Optional registry SQLite path to resolve recording zarr artifacts.",
    )
    parser.add_argument(
        "--registry-prefer-crop-review",
        action="store_true",
        help="Prefer crop review status from crop_quality_current when --registry is provided.",
    )

    args = parser.parse_args(argv)
    roots = _resolve_root(args.paths)
    tuning_keys = (
        [key.strip() for key in args.tuning_keys.split(",") if key.strip()]
        if args.tuning_keys
        else DEFAULT_TUNING_KEYS
    )
    registry: Optional[Registry] = None
    if args.registry is not None:
        registry_path = args.registry.expanduser().resolve()
        if not registry_path.exists():
            print(f"Registry not found: {registry_path}")
            return 1
        registry = Registry(registry_path)
    if args.registry_prefer_crop_review and registry is None:
        print("--registry-prefer-crop-review requires --registry.")
        return 1
    if args.status_source in {"registry", "compare"} and registry is None:
        print(f"--status-source {args.status_source} requires --registry.")
        return 1

    plans: List[RecordingStatus] = []
    compare_rows: List[Dict[str, object]] = []
    compare_checked = 0
    try:
        for h5_path in _iter_h5(roots, args.recursive):
            recording_dir = h5_path.parent.parent
            camera_id = _read_camera_id(h5_path)
            recording_id = _read_recording_id(h5_path)
            zarr_candidates = _resolve_zarr_candidates(
                recording_dir=recording_dir,
                recording_id=recording_id,
                requested_use=str(args.zarr_use),
                registry=registry,
            )
            for zarr_path, zarr_use in zarr_candidates:
                if args.status_source == "compare":
                    compare_checked += 1
                    fs_info = _check_zarr(zarr_path, tuning_keys)
                    fs_plan = _build_recording_status(
                        recording_dir=recording_dir,
                        h5_path=h5_path,
                        camera_id=camera_id,
                        recording_id=recording_id,
                        zarr_path=zarr_path,
                        zarr_use=zarr_use,
                        zarr_info=fs_info,
                    )
                    reg_info = _registry_status_payload(
                        registry=registry,  # type: ignore[arg-type]
                        zarr_path=zarr_path,
                        recording_id=recording_id,
                        tuning_keys=tuning_keys,
                    )
                    reg_plan = _build_recording_status(
                        recording_dir=recording_dir,
                        h5_path=h5_path,
                        camera_id=camera_id,
                        recording_id=recording_id,
                        zarr_path=zarr_path,
                        zarr_use=zarr_use,
                        zarr_info=reg_info,
                    )
                    fs_snapshot = _plan_compare_snapshot(fs_plan, tuning_keys)
                    reg_snapshot = _plan_compare_snapshot(reg_plan, tuning_keys)
                    for field in sorted(set(fs_snapshot) | set(reg_snapshot)):
                        fs_value = fs_snapshot.get(field, "—")
                        reg_value = reg_snapshot.get(field, "—")
                        if fs_value == reg_value:
                            continue
                        compare_rows.append(
                            {
                                "recording": recording_dir.name,
                                "camera_id": camera_id or "unknown",
                                "zarr_path": str(zarr_path),
                                "zarr_use": zarr_use or "—",
                                "field": field,
                                "filesystem": fs_value,
                                "registry": reg_value,
                            }
                        )
                    continue

                if args.status_source == "registry":
                    zarr_info = _registry_status_payload(
                        registry=registry,  # type: ignore[arg-type]
                        zarr_path=zarr_path,
                        recording_id=recording_id,
                        tuning_keys=tuning_keys,
                    )
                else:
                    zarr_info = _check_zarr(zarr_path, tuning_keys)
                    if args.registry_prefer_crop_review and registry is not None:
                        crop_registry = _registry_crop_review_status_for_zarr(
                            registry=registry,
                            zarr_path=zarr_path,
                        )
                        if crop_registry is not None:
                            zarr_info["crop_present"] = True
                            if crop_registry.get("crop_review_status") is not None:
                                zarr_info["crop_review_status"] = crop_registry["crop_review_status"]
                plans.append(
                    _build_recording_status(
                        recording_dir=recording_dir,
                        h5_path=h5_path,
                        camera_id=camera_id,
                        recording_id=recording_id,
                        zarr_path=zarr_path,
                        zarr_use=zarr_use,
                        zarr_info=zarr_info,
                    )
                )
    finally:
        if registry is not None:
            registry.close()

    if args.status_source == "compare":
        if compare_checked == 0:
            print("No recordings found.")
            return 1
        if not compare_rows:
            print("No status mismatches found.")
            return 0
        use_rich = not args.no_rich and Console is not None and Table is not None
        if use_rich:
            console = Console()
            table = Table(title="Recording Step Status Mismatches", show_lines=False)
            table.add_column("Recording", style="cyan")
            table.add_column("Camera", style="magenta")
            table.add_column("Use")
            table.add_column("Field")
            table.add_column("Filesystem")
            table.add_column("Registry")
            table.add_column("Zarr Path", style="dim")
            for row in compare_rows:
                table.add_row(
                    str(row["recording"]),
                    str(row["camera_id"]),
                    str(row["zarr_use"]),
                    _display_field_label(str(row["field"])),
                    str(row["filesystem"]),
                    str(row["registry"]),
                    str(row["zarr_path"]),
                )
            console.print(table)
        else:
            print("Recording Step Status Mismatches")
            for row in compare_rows:
                print(str(row["recording"]))
                print(f"  camera_id: {row['camera_id']}")
                print(f"  use: {row['zarr_use']}")
                print(f"  field: {_display_field_label(str(row['field']))}")
                print(f"  filesystem: {row['filesystem']}")
                print(f"  registry: {row['registry']}")
                print(f"  zarr_path: {row['zarr_path']}")
        return 0

    if not plans:
        print("No recordings found.")
        return 1

    use_rich = not args.no_rich and Console is not None and Table is not None
    if use_rich:
        console = Console()
        table = Table(title="Recording Step Status", show_lines=False)
        table.add_column("Recording", style="cyan")
        table.add_column("Camera", style="magenta")
        table.add_column("Zarr")
        table.add_column("Use")
        table.add_column("Purpose")
        table.add_column("Import")
        table.add_column("BG Full")
        table.add_column("BG DS")
        table.add_column("Detect")
        table.add_column("Detect Quality")
        table.add_column("Refine Detect")
        table.add_column("Detect Group")
        table.add_column("Detect Review")
        table.add_column("Crop")
        table.add_column("Crop Review")
        table.add_column("Keypoints")
        table.add_column("Refined Keypoints (analysis/train)")
        table.add_column("Keypoint Review")
        table.add_column("Eye Masks (legacy)")
        table.add_column("Refined Eye Masks (legacy)")
        table.add_column("Eye Review (legacy)")
        table.add_column("Subject Masks")
        table.add_column("Subject Components (unified)")
        table.add_column("Refined Subject Masks")
        table.add_column("Refined Subject Components (unified)")
        table.add_column("Arena Assignment")
        table.add_column("Track")
        table.add_column("Stimulus")
        table.add_column("Calib")
        table.add_column("Tuning")
        for key in tuning_keys:
            table.add_column(key, style="dim")
        for plan in plans:
            is_production = (
                (plan.zarr_purpose == "production")
                or (plan.pipeline_type == "yolo_inference")
                or (plan.has_raw_video_attr is False and not (plan.full_present or plan.ds_present))
            )
            import_ok = None if is_production else (plan.raw_present and (plan.full_present or plan.ds_present))
            stimulus_ok = plan.stimulus_runs > 0
            background_full_ok = None if is_production else plan.background_full_present
            background_ds_ok = None if is_production else plan.background_ds_present
            tuning_text = f"{plan.tuning_present}/{plan.tuning_total}"
            stimulus_text = f"{plan.stimulus_runs} ({_status_rich(stimulus_ok)})"
            row = [
                plan.recording_dir.name,
                plan.camera_id or "unknown",
                _status_rich(plan.zarr_exists),
                plan.zarr_use or "—",
                plan.zarr_purpose or "—",
                _status_rich(import_ok),
                _status_rich(background_full_ok),
                _status_rich(background_ds_ok),
                _detect_status_rich(
                    plan.detect_present,
                    plan.detect_method,
                    plan.detect_coverage,
                    plan.detect_coverage_basis,
                ),
                _detect_quality_status_rich(
                    plan.detect_quality_present,
                    plan.detect_quality_grade,
                    plan.detect_quality_score,
                    plan.detect_quality_clean_percent,
                    plan.detect_quality_artifacts,
                ),
                _refined_status_rich(plan.refined_detect_coverage, plan.refined_detect_method),
                _resolved_group_rich(plan.refined_detect_resolved_group),
                _review_status_rich(plan.detect_review_status),
                _crop_status_rich(plan.crop_present, plan.crop_status),
                _review_status_rich(plan.crop_review_status),
                _status_rich(plan.keypoints_present),
                _keypoint_status_rich(plan.refined_keypoints_success, plan.refined_keypoints_coverage),
                _review_status_rich(plan.keypoint_review_status),
                _status_rich(plan.eye_masks_present),
                _status_rich(plan.refined_eye_masks_present),
                _review_status_rich(plan.eye_mask_review_status),
                _subject_mask_stage_rich(plan.subject_masks_present, plan.subject_masks_coverage),
                _subject_mask_component_summary_rich(
                    plan.subject_mask_available_components,
                    plan.subject_mask_component_review_states,
                ),
                _subject_mask_stage_rich(
                    plan.refined_subject_masks_present,
                    plan.refined_subject_masks_coverage,
                ),
                _subject_mask_component_summary_rich(
                    plan.refined_subject_mask_available_components,
                    plan.refined_subject_mask_component_review_states,
                ),
                _status_rich(plan.arena_assignment_present),
                _track_status_rich(
                    plan.track_present,
                    plan.track_qc_state,
                    plan.track_unassigned_rows,
                    plan.track_unassigned_rate_percent,
                ),
                stimulus_text,
                _status_rich(plan.calibration_present),
                "N/A" if is_production else tuning_text,
            ]
            for key in tuning_keys:
                status = "na" if is_production else plan.tuning_status.get(key, "miss")
                row.append(_tuning_status_rich(status))
            table.add_row(*row)
        console.print(table)
    else:
        for plan in plans:
            is_production = (
                (plan.zarr_purpose == "production")
                or (plan.pipeline_type == "yolo_inference")
                or (plan.has_raw_video_attr is False and not (plan.full_present or plan.ds_present))
            )
            import_ok = None if is_production else (plan.raw_present and (plan.full_present or plan.ds_present))
            stimulus_ok = plan.stimulus_runs > 0
            background_full_ok = None if is_production else plan.background_full_present
            background_ds_ok = None if is_production else plan.background_ds_present
            print(plan.recording_dir.name)
            print(f"  camera_id: {plan.camera_id or 'unknown'}")
            print(f"  recording_id: {plan.recording_id or 'unknown'}")
            print(f"  zarr: {_status_text(plan.zarr_exists)}")
            print(f"  use: {plan.zarr_use or '—'}")
            if plan.zarr_purpose:
                print(f"  purpose: {plan.zarr_purpose}")
            print(f"  import: {_status_text(import_ok)}")
            print(f"  background_full: {_status_text(background_full_ok)}")
            print(f"  background_ds: {_status_text(background_ds_ok)}")
            print(
                "  detect: "
                f"{_detect_status_text(plan.detect_present, plan.detect_method, plan.detect_coverage, plan.detect_coverage_basis)}"
            )
            print(
                "  detect_quality: "
                f"{_detect_quality_status_text(plan.detect_quality_present, plan.detect_quality_grade, plan.detect_quality_score, plan.detect_quality_clean_percent, plan.detect_quality_artifacts)}"
            )
            if plan.detect_quality_run:
                print(f"  detect_quality_run: {plan.detect_quality_run}")
            print(
                f"  refined_detect: {_refined_status_text(plan.refined_detect_coverage, plan.refined_detect_method)}"
            )
            print(f"  detect_group: {_resolved_group_text(plan.refined_detect_resolved_group)}")
            print(f"  detect_review_status: {_review_status_text(plan.detect_review_status)}")
            print(f"  crop: {_crop_status_text(plan.crop_present, plan.crop_status)}")
            print(f"  crop_review_status: {_review_status_text(plan.crop_review_status)}")
            print(f"  keypoints: {_status_text(plan.keypoints_present)}")
            print(
                f"  refined_keypoints: "
                f"{_keypoint_status_text(plan.refined_keypoints_success, plan.refined_keypoints_coverage)}"
            )
            print(f"  keypoint_review_status: {_review_status_text(plan.keypoint_review_status)}")
            print(f"  eye_masks (legacy compat): {_status_text(plan.eye_masks_present)}")
            print(f"  refined_eye_masks (legacy compat): {_status_text(plan.refined_eye_masks_present)}")
            print(
                "  eye_mask_review_status (legacy compat): "
                f"{_review_status_text(plan.eye_mask_review_status)}"
            )
            print(f"  subject_masks: {_subject_mask_stage_text(plan.subject_masks_present, plan.subject_masks_coverage)}")
            print(
                "  subject_mask_components (unified): "
                f"{_subject_mask_component_summary_text(plan.subject_mask_available_components, plan.subject_mask_component_review_states)}"
            )
            print(f"  subject_mask_review_status: {_review_status_text(plan.subject_mask_review_status)}")
            print(
                "  refined_subject_masks: "
                f"{_subject_mask_stage_text(plan.refined_subject_masks_present, plan.refined_subject_masks_coverage)}"
            )
            print(
                "  refined_subject_mask_components (unified): "
                f"{_subject_mask_component_summary_text(plan.refined_subject_mask_available_components, plan.refined_subject_mask_component_review_states)}"
            )
            print(
                "  refined_subject_mask_review_status: "
                f"{_review_status_text(plan.refined_subject_mask_review_status)}"
            )
            print(f"  arena_assignment: {_status_text(plan.arena_assignment_present)}")
            print(
                "  track: "
                f"{_track_status_text(plan.track_present, plan.track_qc_state, plan.track_unassigned_rows, plan.track_unassigned_rate_percent)}"
            )
            print(f"  stimulus_runs: {plan.stimulus_runs} ({_status_text(stimulus_ok)})")
            print(f"  calibration: {_status_text(plan.calibration_present)}")
            if is_production:
                print("  tuning: N/A (production)")
            else:
                print(
                    f"  tuning: {plan.tuning_present}/{plan.tuning_total} "
                    f"(missing: {', '.join(plan.tuning_missing) if plan.tuning_missing else 'none'})"
                )
            for key in tuning_keys:
                status = "na" if is_production else plan.tuning_status.get(key, "miss")
                print(f"    {key}: {_tuning_status_text(status)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
