from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.shared.refined_detect_review import DEFAULT_DETECT_GROUP_PREFERENCE, resolve_refined_detect_group


SCHEMA_NAME = "detection_dataset_profile"
SCHEMA_VERSION = "v1"

GEOMETRY_PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)

COMPOSITION_FIELDS = (
    "rig_id",
    "camera_id",
    "arena_id",
    "dish_design",
    "canvas_name",
    "protocol_name",
    "genotype",
    "dpf_at_acquisition",
)

DEFAULT_PROFILE_CONFIG: dict[str, Any] = {
    "edge_margin_norm": 0.05,
    "center_heatmap": {"grid_h": 32, "grid_w": 32},
    "histograms": {
        "w_norm": {"range": [0.0, 1.0], "bins": 50},
        "h_norm": {"range": [0.0, 1.0], "bins": 50},
        "area_norm": {"range": [0.0, 1.0], "bins": 50},
        "aspect_ratio": {"range": [0.0, 4.0], "bins": 80},
    },
}

REVIEW_TIMESTAMP_KEYS = ("review_timestamp_utc", "timestamp", "updated_utc")


class DetectionProfileError(RuntimeError):
    """Base error for detection profile read/write failures."""


class DetectionSourceError(DetectionProfileError):
    """Raised when no usable detection source can be resolved."""


@dataclass(frozen=True)
class ResolvedDetectionSource:
    detection_path: str
    detection_type: str
    detect_run: Optional[str]
    refined_run: Optional[str]
    manual_group: Optional[str]
    review_state: Optional[str]
    review_method: Optional[str]
    review_intended_use: Optional[str]
    review_timestamp_utc: Optional[str]


@dataclass(frozen=True)
class DetectionProfileWriteResult:
    run_name: str
    source_detection_path: str
    source_detection_type: str
    profile_summary: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_profile_run_name(created_at_utc: str) -> str:
    try:
        stamp = datetime.fromisoformat(str(created_at_utc))
    except Exception:
        stamp = datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    stamp = stamp.astimezone(timezone.utc)
    return f"detection_profile_{stamp.strftime('%Y-%m-%d_%H-%M-%S')}"


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text or None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_mapping(value: Any) -> Optional[dict[str, Any]]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (bytes, bytearray)):
        raw = value.decode("utf-8", "ignore")
    elif isinstance(value, str):
        raw = value
    else:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if isinstance(payload, Mapping):
        return dict(payload)
    return None


def _select_latest_run(parent: zarr.Group) -> Optional[str]:
    latest = _normalize_text(parent.attrs.get("latest"))
    if latest and latest in parent:
        return latest
    try:
        names = sorted(str(name) for name in parent.group_keys())
    except Exception:
        names = sorted(str(name) for name in parent.keys())
    return names[-1] if names else None


def infer_zarr_use(root: zarr.Group, zarr_path: Optional[Path] = None) -> Optional[str]:
    for key in ("zarr_use", "zarr_purpose"):
        value = _normalize_text(root.attrs.get(key))
        if value is None:
            continue
        lowered = value.lower()
        if lowered in {"analysis", "training"}:
            return lowered
    if zarr_path is None:
        return None
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return None


def _merge_profile_config(profile_config: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_PROFILE_CONFIG)
    if not profile_config:
        return merged

    edge_margin = profile_config.get("edge_margin_norm")
    if edge_margin is not None:
        merged["edge_margin_norm"] = float(edge_margin)

    center_cfg = profile_config.get("center_heatmap")
    if isinstance(center_cfg, Mapping):
        for key in ("grid_h", "grid_w"):
            if key in center_cfg and center_cfg[key] is not None:
                merged["center_heatmap"][key] = int(center_cfg[key])

    hist_cfg = profile_config.get("histograms")
    if isinstance(hist_cfg, Mapping):
        for name in ("w_norm", "h_norm", "area_norm", "aspect_ratio"):
            override = hist_cfg.get(name)
            if not isinstance(override, Mapping):
                continue
            for key in ("range", "bins"):
                if key in override and override[key] is not None:
                    merged["histograms"][name][key] = override[key]
            range_value = merged["histograms"][name].get("range")
            if isinstance(range_value, Sequence) and len(range_value) >= 2:
                merged["histograms"][name]["range"] = [float(range_value[0]), float(range_value[1])]
            merged["histograms"][name]["bins"] = int(merged["histograms"][name]["bins"])

    return merged


def _quantile(values: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(values, q, method="linear"))
    except TypeError:  # numpy < 1.22
        return float(np.quantile(values, q, interpolation="linear"))


def _metric_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    stats: dict[str, Any] = {"count": int(values.size)}
    if values.size == 0:
        stats.update(
            {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
            }
        )
        for pct in GEOMETRY_PERCENTILES:
            stats[f"p{pct:02d}"] = None
        return stats

    stats.update(
        {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    )
    for pct in GEOMETRY_PERCENTILES:
        stats[f"p{pct:02d}"] = _quantile(values, pct / 100.0)
    return stats


def _detections_per_frame_stats(frame_counts: np.ndarray) -> dict[str, Any]:
    values = np.asarray(frame_counts, dtype=np.float64)
    if values.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": int(np.min(values)),
        "max": int(np.max(values)),
        "p10": _quantile(values, 0.10),
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
    }


def _make_bin_edges(low: float, high: float, bins: int) -> np.ndarray:
    bins = max(1, int(bins))
    high = float(high)
    low = float(low)
    if not np.isfinite(low) or not np.isfinite(high):
        raise DetectionProfileError(f"non-finite histogram range: low={low} high={high}")
    if high <= low:
        raise DetectionProfileError(f"invalid histogram range: low={low} high={high}")
    return np.linspace(low, high, bins + 1, dtype=np.float64)


def _histogram_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros(len(edges) - 1, dtype=np.int64)
    clipped = np.clip(values, edges[0], np.nextafter(edges[-1], edges[0]))
    counts, _ = np.histogram(clipped, bins=edges)
    return counts.astype(np.int64, copy=False)


def _extract_subject_snapshot(root: zarr.Group) -> dict[str, Any]:
    analysis_meta = root.get("analysis_metadata")
    if analysis_meta is None:
        return {}
    for key in ("subject_metadata", "zebrobot_snapshot"):
        payload = _coerce_mapping(analysis_meta.attrs.get(key))
        if payload:
            return payload
    return {}


def _extract_composition(root: zarr.Group) -> dict[str, Any]:
    session_context: dict[str, Any] = {}
    analysis_meta = root.get("analysis_metadata")
    if analysis_meta is not None:
        payload = _coerce_mapping(analysis_meta.attrs.get("session_context"))
        if payload:
            session_context = payload
    subject_snapshot = _extract_subject_snapshot(root)
    dish_map = _coerce_mapping(subject_snapshot.get("dish")) if subject_snapshot else None
    dish_map = dish_map or {}

    protocol_from_context = _normalize_text(
        session_context.get("protocol_name") or session_context.get("protocol_name_from_definition")
    )
    genotype_from_snapshot = _normalize_text(
        dish_map.get("genotype") or subject_snapshot.get("genotype")
    )
    dpf_from_snapshot = _as_int(
        subject_snapshot.get("dpf_at_acquisition")
        or subject_snapshot.get("days_post_fertilization")
    )

    composition: dict[str, Any] = {}
    for key in COMPOSITION_FIELDS:
        if key == "protocol_name":
            value = _normalize_text(root.attrs.get("protocol_name")) or protocol_from_context
        elif key == "genotype":
            value = (
                _normalize_text(root.attrs.get("genotype"))
                or _normalize_text(session_context.get("genotype"))
                or genotype_from_snapshot
            )
        elif key == "dpf_at_acquisition":
            value = _as_int(root.attrs.get("dpf_at_acquisition"))
            if value is None:
                value = _as_int(session_context.get("dpf_at_acquisition"))
            if value is None:
                value = _as_int(session_context.get("days_post_fertilization"))
            if value is None:
                value = dpf_from_snapshot
        else:
            value = _normalize_text(root.attrs.get(key)) or _normalize_text(session_context.get(key))
        if value is not None:
            composition[key] = value
    return composition


def _resolve_review_fields(refined_group: Optional[zarr.Group]) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    if refined_group is None:
        return None, None, None, None
    review = _coerce_mapping(refined_group.attrs.get("detect_review_status"))
    if not review:
        return None, None, None, None
    timestamp = None
    for key in REVIEW_TIMESTAMP_KEYS:
        timestamp = _normalize_text(review.get(key))
        if timestamp is not None:
            break
    return (
        _normalize_text(review.get("state")),
        _normalize_text(review.get("method")),
        _normalize_text(review.get("intended_use")),
        timestamp,
    )


def _path_exists(root: zarr.Group, path: str) -> bool:
    try:
        _ = root[path]
        return True
    except Exception:
        return False


def resolve_detection_source(
    root: zarr.Group,
    *,
    source_detection_path: Optional[str] = None,
    detect_run: Optional[str] = None,
    refined_run: Optional[str] = None,
    preference: Sequence[str] = DEFAULT_DETECT_GROUP_PREFERENCE,
) -> ResolvedDetectionSource:
    detect_parent = root.get("detect_runs")
    refined_parent_name = "refined_detect_runs" if root.get("refined_detect_runs") is not None else "refined_runs"
    refined_parent = root.get(refined_parent_name)

    detect_run = _normalize_text(detect_run)
    refined_run = _normalize_text(refined_run)

    def _detect_only(run_name: str) -> ResolvedDetectionSource:
        if detect_parent is None or run_name not in detect_parent:
            raise DetectionSourceError(f"detect run not found: {run_name}")
        return ResolvedDetectionSource(
            detection_path=f"detect_runs/{run_name}",
            detection_type="detect",
            detect_run=run_name,
            refined_run=None,
            manual_group=None,
            review_state=None,
            review_method=None,
            review_intended_use=None,
            review_timestamp_utc=None,
        )

    if source_detection_path:
        source_path = source_detection_path.strip("/")
        if not _path_exists(root, source_path):
            raise DetectionSourceError(f"source detection path missing: {source_path}")
        parts = source_path.split("/")
        if len(parts) >= 2 and parts[0] == "detect_runs":
            return _detect_only(parts[1])
        if len(parts) >= 3 and parts[0] in {"refined_detect_runs", "refined_runs"}:
            parent_name, refined_name, group_name = parts[0], parts[1], parts[2]
            parent = root.get(parent_name)
            if parent is None or refined_name not in parent:
                raise DetectionSourceError(f"refined run not found: {refined_name}")
            refined_group = parent[refined_name]
            manual_latest = _normalize_text(refined_group.attrs.get("manual_review_latest"))
            source_detect = _normalize_text(refined_group.attrs.get("source_detect_run"))
            review_state, review_method, review_use, review_timestamp = _resolve_review_fields(refined_group)
            detection_type = "manual" if group_name in {"manual", manual_latest} else group_name
            return ResolvedDetectionSource(
                detection_path=source_path,
                detection_type=detection_type,
                detect_run=source_detect,
                refined_run=refined_name,
                manual_group=group_name if detection_type == "manual" else None,
                review_state=review_state,
                review_method=review_method,
                review_intended_use=review_use,
                review_timestamp_utc=review_timestamp,
            )
        raise DetectionSourceError(f"unsupported source_detection_path format: {source_path}")

    if refined_parent is not None:
        refined_name = refined_run or _select_latest_run(refined_parent)
        if refined_name is not None:
            if refined_name not in refined_parent:
                raise DetectionSourceError(f"refined run not found: {refined_name}")
            refined_group = refined_parent[refined_name]
            resolution = resolve_refined_detect_group(
                refined_group,
                preference=preference,
                override_group=None,
            )
            source_detect = _normalize_text(resolution.source_detect_run) or _normalize_text(
                refined_group.attrs.get("source_detect_run")
            )
            review_state, review_method, review_use, review_timestamp = _resolve_review_fields(refined_group)
            if resolution.group is not None:
                source_path = f"{refined_parent_name}/{refined_name}/{resolution.group}"
                if not _path_exists(root, source_path):
                    raise DetectionSourceError(f"resolved refined group missing: {source_path}")
                detection_type = "manual" if (resolution.label or "").lower() == "manual" else str(
                    resolution.label or resolution.group
                ).lower()
                return ResolvedDetectionSource(
                    detection_path=source_path,
                    detection_type=detection_type,
                    detect_run=source_detect,
                    refined_run=refined_name,
                    manual_group=resolution.group if detection_type == "manual" else None,
                    review_state=review_state,
                    review_method=review_method,
                    review_intended_use=review_use,
                    review_timestamp_utc=review_timestamp,
                )
            if source_detect:
                detect_choice = source_detect
            elif detect_run:
                detect_choice = detect_run
            elif detect_parent is not None:
                detect_choice = _select_latest_run(detect_parent)
            else:
                detect_choice = None
            if detect_choice:
                detect_source = _detect_only(detect_choice)
                return ResolvedDetectionSource(
                    detection_path=detect_source.detection_path,
                    detection_type=detect_source.detection_type,
                    detect_run=detect_source.detect_run,
                    refined_run=refined_name,
                    manual_group=None,
                    review_state=review_state,
                    review_method=review_method,
                    review_intended_use=review_use,
                    review_timestamp_utc=review_timestamp,
                )

    if detect_parent is None:
        raise DetectionSourceError("detect_runs group missing")
    detect_choice = detect_run or _select_latest_run(detect_parent)
    if detect_choice is None:
        raise DetectionSourceError("no detect run available")
    return _detect_only(detect_choice)


def _load_detection_arrays(root: zarr.Group, source: ResolvedDetectionSource) -> tuple[np.ndarray, np.ndarray]:
    group = root[source.detection_path]
    if "bbox_norm_coords" not in group:
        raise DetectionProfileError(f"{source.detection_path}: missing bbox_norm_coords")
    if "frame_indices" not in group:
        raise DetectionProfileError(f"{source.detection_path}: missing frame_indices")
    bbox = np.asarray(group["bbox_norm_coords"][:], dtype=np.float64)
    frame_indices = np.asarray(group["frame_indices"][:], dtype=np.int64)
    if bbox.ndim != 2 or int(bbox.shape[1]) != 4:
        raise DetectionProfileError(f"{source.detection_path}: bbox_norm_coords must have shape (N,4)")
    if frame_indices.ndim != 1:
        raise DetectionProfileError(f"{source.detection_path}: frame_indices must be 1D")
    if int(frame_indices.shape[0]) != int(bbox.shape[0]):
        raise DetectionProfileError(
            f"{source.detection_path}: frame_indices length {frame_indices.shape[0]} "
            f"!= bbox_norm_coords length {bbox.shape[0]}"
        )
    finite_mask = np.isfinite(bbox).all(axis=1)
    finite_mask &= np.isfinite(frame_indices.astype(np.float64, copy=False))
    finite_mask &= frame_indices >= 0
    if not np.all(finite_mask):
        bbox = bbox[finite_mask]
        frame_indices = frame_indices[finite_mask]
    return bbox, frame_indices.astype(np.int64, copy=False)


def _resolve_frame_universe(
    root: zarr.Group,
    source: ResolvedDetectionSource,
    frame_indices: np.ndarray,
) -> tuple[int, str, int]:
    group = root[source.detection_path]

    frames_total = None
    if "frame_counts" in group:
        frames_total = _as_int(group["frame_counts"].shape[0])

    frame_source = "full"
    frames_full = None

    if source.refined_run:
        parent_name = "refined_detect_runs" if _path_exists(root, "refined_detect_runs") else "refined_runs"
        refined_path = f"{parent_name}/{source.refined_run}"
        if _path_exists(root, refined_path):
            refined_group = root[refined_path]
            refined_total = _as_int(refined_group.attrs.get("coverage_frames_total"))
            refined_source = _normalize_text(refined_group.attrs.get("coverage_frame_source"))
            refined_full = _as_int(refined_group.attrs.get("coverage_frames_full"))
            if refined_total is not None and refined_total > 0:
                frames_total = refined_total
            if refined_source in {"full", "sampled"}:
                frame_source = refined_source
            if refined_full is not None and refined_full > 0:
                frames_full = refined_full

    if frames_total is None:
        frames_total = _as_int(root.attrs.get("total_frames"))

    raw_video = root.get("raw_video")
    if raw_video is not None:
        sampled_import = (
            _normalize_text(raw_video.attrs.get("import_mode")) == "sampled"
            or _normalize_text(raw_video.attrs.get("import_purpose")) == "training_data"
            or "original_frame_indices" in raw_video
        )
        if sampled_import and frame_source != "sampled":
            frame_source = "sampled"
        if "original_frame_indices" in raw_video:
            frames_full = _as_int(raw_video["original_frame_indices"].shape[0]) or frames_full
        if frames_total is None and "images_ds" in raw_video:
            frames_total = _as_int(raw_video["images_ds"].shape[0])
        if frames_total is None and "images_full" in raw_video:
            frames_total = _as_int(raw_video["images_full"].shape[0])

    if frames_total is None:
        frames_total = int(frame_indices.max() + 1) if frame_indices.size else 0
    if frame_indices.size:
        observed = int(frame_indices.max() + 1)
        if observed > int(frames_total):
            frames_total = observed

    frames_total = max(0, int(frames_total))
    if frame_source == "sampled":
        if frames_full is None:
            frames_full = frames_total
    else:
        frames_full = frames_total

    return frames_total, frame_source, int(frames_full)


def _build_histograms(
    *,
    profile_config: Mapping[str, Any],
    w: np.ndarray,
    h: np.ndarray,
    area: np.ndarray,
    aspect_ratio: np.ndarray,
) -> dict[str, dict[str, list[Any]]]:
    histogram_values = {
        "w_norm": w,
        "h_norm": h,
        "area_norm": area,
        "aspect_ratio": aspect_ratio,
    }
    hist_payload: dict[str, dict[str, list[Any]]] = {}
    for name, values in histogram_values.items():
        hist_cfg = profile_config["histograms"][name]
        bounds = hist_cfg.get("range") or [0.0, 1.0]
        bins = int(hist_cfg.get("bins") or 1)
        edges = _make_bin_edges(float(bounds[0]), float(bounds[1]), bins)
        counts = _histogram_counts(values, edges)
        hist_payload[name] = {
            "bin_edges": [float(x) for x in edges.tolist()],
            "counts": [int(x) for x in counts.tolist()],
        }
    return hist_payload


def _build_heatmap_density(cx: np.ndarray, cy: np.ndarray, *, grid_h: int, grid_w: int) -> np.ndarray:
    grid_h = max(1, int(grid_h))
    grid_w = max(1, int(grid_w))
    heatmap = np.zeros((grid_h, grid_w), dtype=np.float64)
    if cx.size == 0:
        return heatmap

    cx_clip = np.clip(cx, 0.0, np.nextafter(1.0, 0.0))
    cy_clip = np.clip(cy, 0.0, np.nextafter(1.0, 0.0))
    x_idx = np.floor(cx_clip * grid_w).astype(np.int64, copy=False)
    y_idx = np.floor(cy_clip * grid_h).astype(np.int64, copy=False)
    np.add.at(heatmap, (y_idx, x_idx), 1.0)
    heatmap /= float(cx.size)
    return heatmap


def build_detection_profile_summary(
    root: zarr.Group,
    *,
    zarr_path: Optional[Path] = None,
    source_detection_path: Optional[str] = None,
    detect_run: Optional[str] = None,
    refined_run: Optional[str] = None,
    dataset_id: Optional[str] = None,
    recording_id: Optional[str] = None,
    zarr_use: Optional[str] = None,
    created_at_utc: Optional[str] = None,
    profile_config: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    profile_config = _merge_profile_config(profile_config)
    created_at_utc = created_at_utc or _utc_now()
    source = resolve_detection_source(
        root,
        source_detection_path=source_detection_path,
        detect_run=detect_run,
        refined_run=refined_run,
    )
    bbox, frame_indices = _load_detection_arrays(root, source)
    frames_total, frame_source, frames_full = _resolve_frame_universe(root, source, frame_indices)

    if frames_total > 0:
        frame_counts = np.bincount(frame_indices, minlength=frames_total).astype(np.int64, copy=False)
    else:
        frame_counts = np.zeros((0,), dtype=np.int64)
    frames_with_detections = int(np.sum(frame_counts > 0))

    cx = bbox[:, 0] if bbox.size else np.empty((0,), dtype=np.float64)
    cy = bbox[:, 1] if bbox.size else np.empty((0,), dtype=np.float64)
    w = bbox[:, 2] if bbox.size else np.empty((0,), dtype=np.float64)
    h = bbox[:, 3] if bbox.size else np.empty((0,), dtype=np.float64)
    area = w * h
    h_safe = np.where(h > 0.0, h, 1e-12)
    aspect_ratio = w / h_safe

    detections_total = int(bbox.shape[0])
    if frames_total > 0:
        coverage_percent = float(100.0 * frames_with_detections / frames_total)
    else:
        coverage_percent = 0.0

    edge_margin = float(profile_config["edge_margin_norm"])
    left = cx - (w * 0.5)
    right = cx + (w * 0.5)
    top = cy - (h * 0.5)
    bottom = cy + (h * 0.5)
    near_edge = (left <= edge_margin) | (top <= edge_margin) | (right >= (1.0 - edge_margin)) | (
        bottom >= (1.0 - edge_margin)
    )
    edge_rate = float(np.mean(near_edge)) if detections_total > 0 else 0.0

    grid_h = int(profile_config["center_heatmap"]["grid_h"])
    grid_w = int(profile_config["center_heatmap"]["grid_w"])
    heatmap_density = _build_heatmap_density(cx, cy, grid_h=grid_h, grid_w=grid_w)

    dataset_id = _normalize_text(dataset_id) or _normalize_text(root.attrs.get("dataset_id"))
    recording_id = (
        _normalize_text(recording_id)
        or _normalize_text(root.attrs.get("recording_id"))
        or _normalize_text(root.attrs.get("session_uuid"))
    )
    zarr_use = _normalize_text(zarr_use) or infer_zarr_use(root, zarr_path)
    zarr_path_text = str(zarr_path) if zarr_path is not None else None

    summary: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": created_at_utc,
        "dataset": {
            "dataset_id": dataset_id,
            "recording_id": recording_id,
            "zarr_use": zarr_use,
            "zarr_path": zarr_path_text,
        },
        "source": {
            "detection_path": source.detection_path,
            "detection_type": source.detection_type,
            "detect_run": source.detect_run,
            "refined_run": source.refined_run,
            "manual_group": source.manual_group,
            "review_state": source.review_state,
            "review_method": source.review_method,
            "review_intended_use": source.review_intended_use,
            "review_timestamp_utc": source.review_timestamp_utc,
        },
        "coverage": {
            "frames_total": int(frames_total),
            "frames_with_detections": int(frames_with_detections),
            "coverage_percent": float(coverage_percent),
            "frame_source": frame_source,
            "frames_full": int(frames_full),
        },
        "counts": {
            "detections_total": int(detections_total),
            "detections_per_frame": _detections_per_frame_stats(frame_counts),
        },
        "geometry_norm": {
            "cx": _metric_stats(cx),
            "cy": _metric_stats(cy),
            "w": _metric_stats(w),
            "h": _metric_stats(h),
            "area": _metric_stats(area),
            "aspect_ratio": _metric_stats(aspect_ratio),
        },
        "spatial": {
            "edge_margin_norm": float(edge_margin),
            "edge_proximity_rate": float(edge_rate),
            "center_heatmap": {
                "grid_h": int(grid_h),
                "grid_w": int(grid_w),
                "density": [float(x) for x in heatmap_density.reshape(-1).tolist()],
            },
        },
        "histograms": _build_histograms(
            profile_config=profile_config,
            w=w,
            h=h,
            area=area,
            aspect_ratio=aspect_ratio,
        ),
    }

    composition = _extract_composition(root)
    if composition:
        summary["composition"] = composition

    return summary


def write_detection_profile(
    root: zarr.Group,
    *,
    zarr_path: Optional[Path] = None,
    run_name: Optional[str] = None,
    overwrite: bool = False,
    source_detection_path: Optional[str] = None,
    detect_run: Optional[str] = None,
    refined_run: Optional[str] = None,
    dataset_id: Optional[str] = None,
    recording_id: Optional[str] = None,
    zarr_use: Optional[str] = None,
    created_at_utc: Optional[str] = None,
    profile_config: Optional[Mapping[str, Any]] = None,
) -> DetectionProfileWriteResult:
    merged_config = _merge_profile_config(profile_config)
    summary = build_detection_profile_summary(
        root,
        zarr_path=zarr_path,
        source_detection_path=source_detection_path,
        detect_run=detect_run,
        refined_run=refined_run,
        dataset_id=dataset_id,
        recording_id=recording_id,
        zarr_use=zarr_use,
        created_at_utc=created_at_utc,
        profile_config=merged_config,
    )
    created = _normalize_text(summary.get("created_at_utc")) or _utc_now()
    run_name = _normalize_text(run_name) or _default_profile_run_name(created)

    def _open_child_group(parent: zarr.Group, name: str) -> Optional[zarr.Group]:
        store = getattr(parent, "store", None)
        if store is None:
            return None
        parent_path = _normalize_text(getattr(parent, "path", None))
        child_path = f"{parent_path}/{name}" if parent_path else name
        try:
            return zarr.open_group(store=store, path=child_path, mode="a")
        except TypeError:
            # Older zarr signatures may not support kw-only store/path.
            try:
                return zarr.open_group(store, mode="a", path=child_path)
            except Exception:
                return None
        except Exception:
            return None

    def _get_or_create_group(parent: zarr.Group, name: str) -> zarr.Group:
        # Some stores can intermittently fail lookup via get() for an existing
        # group. Fall back to keyed access before attempting creation.
        existing = parent.get(name)
        if existing is None:
            try:
                existing = parent[name]
            except Exception:
                existing = None
        if existing is None:
            existing = _open_child_group(parent, name)
        if existing is not None:
            return existing
        try:
            return parent.create_group(name)
        except Exception:
            # If creation says the group already exists, retry direct access.
            existing = parent.get(name)
            if existing is None:
                try:
                    existing = parent[name]
                except Exception:
                    existing = None
            if existing is None:
                existing = _open_child_group(parent, name)
            if existing is not None:
                return existing
            raise

    analysis_group = _get_or_create_group(root, "analysis")
    runs_parent = _get_or_create_group(analysis_group, "detection_profile_runs")
    if run_name in runs_parent:
        if not overwrite:
            raise FileExistsError(f"analysis/detection_profile_runs/{run_name} already exists")
        del runs_parent[run_name]
    run_group = runs_parent.create_group(run_name)

    coverage = summary.get("coverage", {})
    dataset = summary.get("dataset", {})
    source = summary.get("source", {})
    frame_source = _normalize_text(coverage.get("frame_source")) or "full"

    run_group.attrs.update(
        {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": created,
            "source_dataset_id": dataset.get("dataset_id"),
            "source_recording_id": dataset.get("recording_id"),
            "source_zarr_use": dataset.get("zarr_use"),
            "source_detection_path": source.get("detection_path"),
            "source_detection_type": source.get("detection_type"),
            "source_resolution": frame_source,
            "source_frame_count": int(coverage.get("frames_total") or 0),
            "source_frame_count_full": (
                int(coverage.get("frames_full"))
                if frame_source == "sampled" and coverage.get("frames_full") is not None
                else None
            ),
            "profile_config": merged_config,
            "profile_summary": summary,
        }
    )
    runs_parent.attrs["latest"] = run_name

    return DetectionProfileWriteResult(
        run_name=run_name,
        source_detection_path=str(source.get("detection_path") or ""),
        source_detection_type=str(source.get("detection_type") or ""),
        profile_summary=summary,
    )


__all__ = [
    "DEFAULT_PROFILE_CONFIG",
    "DetectionProfileError",
    "DetectionProfileWriteResult",
    "DetectionSourceError",
    "ResolvedDetectionSource",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "build_detection_profile_summary",
    "infer_zarr_use",
    "resolve_detection_source",
    "write_detection_profile",
]
