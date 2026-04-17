"""
Keypoint refinement pipeline.

Detects and corrects left/right eye swaps in keypoint detections and
records diagnostic metrics without re-running the geometry filters that
already execute during detection.
"""

from __future__ import annotations

import time
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple

import numpy as np
import zarr
from rich.console import Console
import argparse
import yaml

import dask
from dask import delayed

try:
    from dask.distributed import Client, LocalCluster

    HAVE_DISTRIBUTED = True
except ImportError:
    LocalCluster = None  # type: ignore
    Client = None  # type: ignore
    HAVE_DISTRIBUTED = False

from .keypoint_quality import KeypointGeometryMetrics, compute_geometry_metrics
from ..registry.db import Registry, RegistryPaths
from ..pose.heading import compute_heading_from_attrs
from ..pose.heuristics import (
    heuristic_profile_from_package,
    maybe_flip_detection_from_attrs,
    maybe_geometry_qc_from_attrs,
    require_flip_detection,
    require_geometry_qc,
)
from ..pose.metric_schema import (
    DerivedMetricStorage,
    ensure_derived_metric_storage,
    resolve_metric_schema_for_group,
    update_derived_metric_rows,
)
from ..shared.detect_reason_codec import write_reason_columns
from ..shared.derived_metrics_schema import build_refined_keypoint_derived_metrics_schema
from ..shared.keypoint_temporal_heading import refresh_refined_keypoint_heading_fields
from ..shared.provenance_attrs import build_source_crop_snapshot_attrs
from ..shared.registry_stage_complete import (
    DatasetMetadata,
    emit_stage_completion,
    extract_dataset_metadata,
)
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.type_conversions import as_float, normalize_attr
from ..utils.system import get_environment_info, get_git_info

REFINED_KEYPOINT_GROUP = "refined_keypoints_runs"
LEGACY_KEYPOINT_GROUP = "keypoints_refined_runs"
_STEP_NAME_REFINED_KEYPOINTS = "refined_keypoints"
_STATUS_SOURCE = "runtime_refine_keypoints"
_TRADITIONAL_HEURISTIC_METHOD = "traditional_pose"
_TRADITIONAL_FLIP_FAMILY = "traditional_eye_flip_v1"
_DEFAULT_CONFIDENCE_THRESHOLD = 0.3

TRADITIONAL_REFINEMENT_HEURISTIC_PROFILE = heuristic_profile_from_package(
    _TRADITIONAL_HEURISTIC_METHOD,
    "traditional_v1",
)
TRADITIONAL_REFINEMENT_GEOMETRY_QC = require_geometry_qc(
    TRADITIONAL_REFINEMENT_HEURISTIC_PROFILE
)
require_flip_detection(
    TRADITIONAL_REFINEMENT_HEURISTIC_PROFILE,
    family=_TRADITIONAL_FLIP_FAMILY,
)


@dataclass(frozen=True)
class _StatusContext:
    registry_path: Path
    dataset_id: str
    recording_id: Optional[str]
    zarr_path: Path


def _as_float(value: object) -> Optional[float]:
    return as_float(value)


def _as_mapping(value: object) -> Optional[Dict[str, object]]:
    if isinstance(value, Mapping):
        return dict(value)
    payload: Optional[str] = None
    if isinstance(value, bytes):
        payload = value.decode("utf-8", "ignore").strip()
    elif isinstance(value, str):
        payload = value.strip()
    if not payload:
        return None
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if isinstance(decoded, Mapping):
        return dict(decoded)
    return None


def _required_geometry_default(value: Optional[float], field_name: str) -> float:
    if value is None:
        raise RuntimeError(
            "Traditional pose heuristic profile is missing "
            f"'geometry_qc.{field_name}'."
        )
    return float(value)


DEFAULT_MIN_TRIANGLE_ANGLE = _required_geometry_default(
    TRADITIONAL_REFINEMENT_GEOMETRY_QC.min_triangle_angle_deg,
    "min_triangle_angle_deg",
)
DEFAULT_MIN_TRIANGLE_AREA = _required_geometry_default(
    TRADITIONAL_REFINEMENT_GEOMETRY_QC.min_triangle_area_px,
    "min_triangle_area_px",
)
DEFAULT_MAX_TRIANGLE_AREA = (
    float(TRADITIONAL_REFINEMENT_GEOMETRY_QC.max_triangle_area_px)
    if TRADITIONAL_REFINEMENT_GEOMETRY_QC.max_triangle_area_px is not None
    else None
)


def _resolve_refinement_geometry_defaults(
    attrs: Mapping[str, object] | None,
) -> tuple[float, float, Optional[float]]:
    if attrs is None:
        return (
            DEFAULT_MIN_TRIANGLE_ANGLE,
            DEFAULT_MIN_TRIANGLE_AREA,
            DEFAULT_MAX_TRIANGLE_AREA,
        )
    geometry_qc = maybe_geometry_qc_from_attrs(_TRADITIONAL_HEURISTIC_METHOD, attrs)
    if geometry_qc is None:
        return (
            DEFAULT_MIN_TRIANGLE_ANGLE,
            DEFAULT_MIN_TRIANGLE_AREA,
            DEFAULT_MAX_TRIANGLE_AREA,
        )
    maybe_flip_detection_from_attrs(
        _TRADITIONAL_HEURISTIC_METHOD,
        attrs,
        family=_TRADITIONAL_FLIP_FAMILY,
    )
    min_angle = _required_geometry_default(
        geometry_qc.min_triangle_angle_deg,
        "min_triangle_angle_deg",
    )
    min_area = _required_geometry_default(
        geometry_qc.min_triangle_area_px,
        "min_triangle_area_px",
    )
    max_area = (
        float(geometry_qc.max_triangle_area_px)
        if geometry_qc.max_triangle_area_px is not None
        else None
    )
    return (min_angle, min_area, max_area)


def _resolve_status_dataset_id(
    registry: Registry,
    *,
    base_dataset_id: str,
    session_uuid: Optional[str],
    zarr_path: Path,
) -> str:
    path_text = str(zarr_path).replace("\\", "/").lower()
    is_source_recording = bool(session_uuid) and "/recordings/" in path_text
    if not is_source_recording:
        return base_dataset_id

    assert session_uuid is not None
    path_hash = hashlib.sha256(str(zarr_path).encode("utf-8")).hexdigest()
    candidate = f"{session_uuid}:z{path_hash[:12]}"
    for extra in ("", path_hash[12:16], path_hash[16:20], path_hash[20:24]):
        resolved = candidate if not extra else f"{candidate}{extra}"
        row = registry.conn.execute(
            "SELECT path_hash FROM datasets WHERE dataset_id = ?;",
            (resolved,),
        ).fetchone()
        if row is None:
            return resolved
        if str(row["path_hash"] or "") == path_hash:
            return resolved
    return f"{session_uuid}:z{path_hash}"


def _resolve_status_context(zarr_path: str) -> Optional[_StatusContext]:
    resolved_zarr = Path(zarr_path).expanduser().resolve()
    registry_path = RegistryPaths.from_env(Path.cwd()).path.expanduser().resolve()
    if not registry_path.exists():
        return None

    registry = Registry(registry_path)
    try:
        path_hash = hashlib.sha256(str(resolved_zarr).encode("utf-8")).hexdigest()
        row = registry.conn.execute(
            """
            SELECT dataset_id, recording_id
            FROM datasets
            WHERE path_hash = ?
            ORDER BY COALESCE(last_seen_utc, '') DESC, dataset_id
            LIMIT 1;
            """,
            (path_hash,),
        ).fetchone()

        if row is None:
            path_variants = tuple(
                dict.fromkeys(
                    [
                        str(resolved_zarr),
                        str(Path(zarr_path).expanduser()),
                        str(Path(zarr_path)),
                    ]
                )
            )
            for candidate in path_variants:
                row = registry.conn.execute(
                    """
                    SELECT dataset_id, recording_id
                    FROM datasets
                    WHERE zarr_path = ?
                    ORDER BY COALESCE(last_seen_utc, '') DESC, dataset_id
                    LIMIT 1;
                    """,
                    (candidate,),
                ).fetchone()
                if row is not None:
                    break

        if row is None:
            row = registry.conn.execute(
                """
                SELECT dataset_id, recording_id
                FROM datasets
                WHERE lower(COALESCE(zarr_path, '')) = lower(?)
                ORDER BY COALESCE(last_seen_utc, '') DESC, dataset_id
                LIMIT 1;
                """,
                (str(resolved_zarr),),
            ).fetchone()

        if row is None:
            return None

        dataset_id = normalize_attr(row["dataset_id"])
        if dataset_id is None:
            return None

        return _StatusContext(
            registry_path=registry_path,
            dataset_id=dataset_id,
            recording_id=normalize_attr(row["recording_id"]),
            zarr_path=resolved_zarr,
        )
    finally:
        registry.close()


def _resolve_status_context_from_root(
    root: zarr.Group,
    zarr_path: str,
) -> Optional[_StatusContext]:
    resolved_zarr = Path(zarr_path).expanduser().resolve()
    registry_path = RegistryPaths.from_env(Path.cwd()).path.expanduser().resolve()
    try:
        metadata = extract_dataset_metadata(root, resolved_zarr)
    except Exception:
        return None

    registry = Registry(registry_path)
    try:
        dataset_id = _resolve_status_dataset_id(
            registry,
            base_dataset_id=metadata.dataset_id,
            session_uuid=metadata.session_uuid,
            zarr_path=resolved_zarr,
        )
        registry.upsert_dataset(
            dataset_id,
            session_uuid=metadata.session_uuid,
            zarr_path=resolved_zarr,
            recording_id=metadata.recording_id,
            zarr_use=metadata.zarr_use,
            zarr_purpose=metadata.zarr_purpose,
        )
        row = registry.conn.execute(
            """
            SELECT recording_id
            FROM datasets
            WHERE dataset_id = ?
            LIMIT 1;
            """,
            (dataset_id,),
        ).fetchone()
        resolved_recording_id = normalize_attr(row["recording_id"]) if row is not None else metadata.recording_id
    except Exception:
        return None
    finally:
        registry.close()

    return _StatusContext(
        registry_path=registry_path,
        dataset_id=dataset_id,
        recording_id=resolved_recording_id,
        zarr_path=resolved_zarr,
    )


def _emit_refined_keypoint_status(
    *,
    context: Optional[_StatusContext],
    status: str,
    run_name: Optional[str],
    method: Optional[str],
    coverage_pct: Optional[float],
    review_status_json: Optional[Dict[str, object]],
    details: Dict[str, object],
    console: Optional[Console],
) -> bool:
    if context is None:
        return False

    registry = Registry(context.registry_path)
    try:
        metadata = DatasetMetadata(
            dataset_id=context.dataset_id,
            session_uuid=None,
            recording_id=context.recording_id,
            zarr_use=None,
            zarr_purpose=None,
        )
        return emit_stage_completion(
            None,
            context.zarr_path,
            step_name=_STEP_NAME_REFINED_KEYPOINTS,
            status=status,
            source=_STATUS_SOURCE,
            run_name=run_name,
            method=method,
            coverage_pct=coverage_pct,
            review_status_json=review_status_json,
            details_json=details,
            console=console,
            warning_label=_STEP_NAME_REFINED_KEYPOINTS,
            registry=registry,
            auto_registry_from_env=False,
            invalidate_on_ok=True,
            trigger_run_name=run_name,
            metadata=metadata,
            upsert_dataset_row=False,
        )
    finally:
        registry.close()


def _classify_refined_keypoint_failure(exc: Exception) -> Tuple[str, str]:
    message = str(exc).lower()
    if isinstance(exc, RuntimeError):
        if "no keypoint runs found" in message:
            return "absent", "keypoints_missing"
        if "contains zero rois" in message:
            return "missing", "no_rois"
    if isinstance(exc, KeyError) and "keypoints_runs/" in str(exc):
        return "absent", "keypoint_run_missing"
    return "error", "refine_failed"


def _write_reason_arrays(group: zarr.Group, reason: np.ndarray, chunk_size: int) -> None:
    """Write reason labels in both text and Crimson-compatible byte formats."""
    write_reason_columns(
        group,
        np.asarray(reason, dtype=object),
        chunk_size,
        include_reason_text=True,
        overwrite=True,
    )


@dataclass
class KeypointRefinementParams:
    """Configuration values controlling keypoint refinement."""

    chunk_size: int = 1024
    scheduler: str = "processes"
    num_workers: Optional[int] = None
    memory_limit: Optional[str] = None
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD
    min_triangle_angle: float = DEFAULT_MIN_TRIANGLE_ANGLE
    min_triangle_area: float = DEFAULT_MIN_TRIANGLE_AREA
    max_triangle_area: Optional[float] = DEFAULT_MAX_TRIANGLE_AREA

    @classmethod
    def from_config(
        cls,
        config: Optional[Dict[str, Any]],
        *,
        source_attrs: Optional[Mapping[str, object]] = None,
    ) -> Tuple["KeypointRefinementParams", str]:
        """
        Instantiate parameters from pipeline config subtree.
        Returns (params, source_label).
        """
        source = "defaults"
        min_triangle_angle, min_triangle_area, max_triangle_area = (
            _resolve_refinement_geometry_defaults(source_attrs)
        )
        params = cls(
            min_triangle_angle=min_triangle_angle,
            min_triangle_area=min_triangle_area,
            max_triangle_area=max_triangle_area,
        )
        if config:
            if config.get("chunk_size") is not None:
                params.chunk_size = int(config["chunk_size"])
            if config.get("scheduler") is not None:
                params.scheduler = str(config["scheduler"])
            if config.get("num_workers") is not None:
                params.num_workers = int(config["num_workers"])
            if config.get("memory_limit") is not None:
                params.memory_limit = str(config["memory_limit"])
            if config.get("confidence_threshold") is not None:
                params.confidence_threshold = float(config["confidence_threshold"])
            if config.get("min_triangle_angle") is not None:
                params.min_triangle_angle = float(config["min_triangle_angle"])
            if config.get("min_triangle_area") is not None:
                params.min_triangle_area = float(config["min_triangle_area"])
            if config.get("max_triangle_area") is not None:
                params.max_triangle_area = float(config["max_triangle_area"])
            source = config.get("parameter_source", "config")
        return params, source


def _ensure_group(root: zarr.Group, name: str) -> zarr.Group:
    if name in root:
        return root[name]
    return root.create_group(name)


def _copy_array(src: zarr.Array, dst_group: zarr.Group, name: str) -> None:
    """Shallow copy helper for metadata arrays."""
    dst_group.create_array(
        name,
        data=src[:],
        chunks=src.chunks,
        overwrite=True,
    )


def _hash_parameters(params: object) -> Optional[str]:
    if params is None:
        return None
    try:
        payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    except (TypeError, ValueError):
        payload = str(params).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_keypoint_signature(
    attrs: Dict[str, object],
    parameters: Optional[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "signature_version": 2,
        "source_keypoints_run": attrs.get("source_keypoints_run"),
        "source_crop_run": attrs.get("source_crop_run"),
        "source_crop_storage_mode": attrs.get("source_crop_storage_mode"),
        "source_crop_signature": attrs.get("source_crop_signature"),
        "source_crop_revision": attrs.get("source_crop_revision"),
        "source_detect_run": attrs.get("source_detect_run"),
        "source_refined_run": attrs.get("source_refined_run"),
        "source_detect_review_status_ref": attrs.get("source_detect_review_status_ref"),
        "parameter_source": parameters.get("parameter_source") if parameters else None,
        "parameters_hash": _hash_parameters(parameters),
    }


def _analyze_coordinate_space(
    *,
    zarr_path: str,
    run_name: str,
    crop_run: Optional[str],
    tol_px: float,
    pad_px: float,
) -> Dict[str, Any]:
    from ..utils.audit_keypoint_coordinate_spaces import analyze as _analyze

    return _analyze(
        zarr_path=zarr_path,
        run_name=run_name,
        crop_run=crop_run,
        tol_px=tol_px,
        pad_px=pad_px,
    )


def _analyze_bad_row_overlap(
    *,
    zarr_path: str,
    run_name: str,
    crop_run: Optional[str],
    tol_px: float,
    bad_ratio_threshold: float,
    pad_px: float,
) -> Dict[str, Any]:
    from ..utils.analyze_bad_keypoint_row_overlap import analyze as _analyze

    return _analyze(
        zarr_path=zarr_path,
        run_name=run_name,
        crop_run=crop_run,
        tol_px=tol_px,
        bad_ratio_threshold=bad_ratio_threshold,
        pad_px=pad_px,
    )


def _dataset_report_stem(zarr_path: str) -> str:
    name = Path(zarr_path).name
    if name.endswith(".zarr"):
        name = name[:-5]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return stem or "dataset"


def _run_post_refinement_diagnostics(
    *,
    zarr_path: str,
    run_name: str,
    source_crop_run: Optional[str],
    config: Optional[Dict[str, Any]],
    run_attrs: MutableMapping[str, Any],
    console: Optional[Console],
) -> Dict[str, Any]:
    refine_cfg: Dict[str, Any] = {}
    if config:
        cfg = config.get("refine_keypoints")
        if isinstance(cfg, Mapping):
            refine_cfg = dict(cfg)

    enabled = bool(refine_cfg.get("post_refinement_audit", True))
    overlap_enabled = bool(refine_cfg.get("post_refinement_overlap", False))
    tol_px = float(refine_cfg.get("post_refinement_tol_px", 2.0))
    pad_px = float(refine_cfg.get("post_refinement_pad_px", 5.0))
    bad_ratio_threshold = float(refine_cfg.get("post_refinement_bad_ratio_threshold", 0.5))
    output_dir = Path(str(refine_cfg.get("post_refinement_output_dir", "/tmp"))).expanduser()
    dataset_stem = _dataset_report_stem(zarr_path)

    result: Dict[str, Any] = {
        "enabled": enabled,
        "overlap_enabled": overlap_enabled,
        "output_dir": str(output_dir),
        "audit_json": None,
        "overlap_json": None,
        "audit_error": None,
        "overlap_error": None,
    }
    if not enabled:
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_json_path = output_dir / f"{dataset_stem}_audit.json"

    try:
        audit_report = _analyze_coordinate_space(
            zarr_path=zarr_path,
            run_name=run_name,
            crop_run=source_crop_run,
            tol_px=tol_px,
            pad_px=pad_px,
        )
        audit_json_path.write_text(json.dumps(audit_report, indent=2), encoding="utf-8")
        result["audit_json"] = str(audit_json_path)
        run_attrs["post_refinement_audit_json"] = str(audit_json_path)
        run_attrs["post_refinement_audit_generated_utc"] = datetime.now(timezone.utc).isoformat()
        if isinstance(audit_report.get("status_counts"), Mapping):
            run_attrs["post_refinement_audit_status_counts"] = dict(audit_report["status_counts"])
        if console is not None:
            console.print(f"[green]✓[/green] Coordinate-space audit JSON: [cyan]{audit_json_path}[/cyan]")
    except Exception as exc:
        result["audit_error"] = str(exc)
        if console is not None:
            console.print(f"[yellow]Warning:[/yellow] coordinate-space audit failed: {exc}")
        return result

    if not overlap_enabled:
        return result

    overlap_json_path = output_dir / f"{dataset_stem}_overlap.json"
    try:
        overlap_report = _analyze_bad_row_overlap(
            zarr_path=zarr_path,
            run_name=run_name,
            crop_run=source_crop_run,
            tol_px=tol_px,
            bad_ratio_threshold=bad_ratio_threshold,
            pad_px=pad_px,
        )
        overlap_json_path.write_text(json.dumps(overlap_report, indent=2), encoding="utf-8")
        result["overlap_json"] = str(overlap_json_path)
        run_attrs["post_refinement_overlap_json"] = str(overlap_json_path)
        run_attrs["post_refinement_overlap_generated_utc"] = datetime.now(timezone.utc).isoformat()
        if "bad_row_count" in overlap_report:
            run_attrs["post_refinement_overlap_bad_row_count"] = int(overlap_report["bad_row_count"])
        if console is not None:
            console.print(f"[green]✓[/green] Bad-row overlap JSON: [cyan]{overlap_json_path}[/cyan]")
    except Exception as exc:
        result["overlap_error"] = str(exc)
        if console is not None:
            console.print(f"[yellow]Warning:[/yellow] bad-row overlap analysis failed: {exc}")

    return result


def _normalize_keypoint_labels(
    kp_source: zarr.Group,
    *,
    n_keypoints: int,
) -> List[str]:
    labels_raw = kp_source.attrs.get("keypoint_labels")
    labels: List[str] = []
    if isinstance(labels_raw, (list, tuple)):
        labels = [str(item).strip() for item in labels_raw if str(item).strip()]
    if len(labels) != n_keypoints:
        pose_schema = _as_mapping(kp_source.attrs.get("pose_schema")) or {}
        nodes = pose_schema.get("nodes")
        parsed_nodes: List[str] = []
        if isinstance(nodes, (list, tuple)):
            for node in nodes:
                if isinstance(node, Mapping):
                    text = normalize_attr(node.get("name")) or normalize_attr(node.get("id"))
                    if text:
                        parsed_nodes.append(text)
                else:
                    text = normalize_attr(node)
                    if text:
                        parsed_nodes.append(text)
        if len(parsed_nodes) == n_keypoints:
            labels = parsed_nodes
    if len(labels) != n_keypoints:
        labels = [f"k{i}" for i in range(n_keypoints)]
    return labels


def _extract_skeleton_edges(
    kp_source: zarr.Group,
    *,
    n_keypoints: int,
) -> Tuple[np.ndarray, str]:
    raw_edges: Optional[object] = None
    source = "none"
    pose_schema = _as_mapping(kp_source.attrs.get("pose_schema"))
    if pose_schema:
        raw_edges = pose_schema.get("edges")
        if raw_edges is None:
            raw_edges = pose_schema.get("skeleton")
        if raw_edges is not None:
            source = "pose_schema"

    if raw_edges is None:
        raw_edges = kp_source.attrs.get("keypoint_skeleton")
        if raw_edges is not None:
            source = "keypoint_skeleton"

    if raw_edges is None and n_keypoints == 3:
        raw_edges = ((0, 1), (0, 2), (1, 2))
        source = "default_triangle"

    pairs: List[Tuple[int, int]] = []
    if isinstance(raw_edges, (list, tuple)):
        for edge in raw_edges:
            a_raw: Optional[object] = None
            b_raw: Optional[object] = None
            if isinstance(edge, Mapping):
                a_raw = edge.get("from")
                if a_raw is None:
                    a_raw = edge.get("source")
                if a_raw is None:
                    a_raw = edge.get("a")
                b_raw = edge.get("to")
                if b_raw is None:
                    b_raw = edge.get("target")
                if b_raw is None:
                    b_raw = edge.get("b")
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                a_raw = edge[0]
                b_raw = edge[1]
            if a_raw is None or b_raw is None:
                continue
            try:
                a = int(a_raw)
                b = int(b_raw)
            except (TypeError, ValueError):
                continue
            if a < 0 or b < 0 or a >= n_keypoints or b >= n_keypoints or a == b:
                continue
            left, right = (a, b) if a < b else (b, a)
            pair = (left, right)
            if pair not in pairs:
                pairs.append(pair)

    if not pairs:
        return np.zeros((0, 2), dtype=np.int16), source

    max_index = max(max(pair) for pair in pairs)
    dtype = np.int16 if max_index <= np.iinfo(np.int16).max else np.int32
    return np.asarray(pairs, dtype=dtype), source


def _resolve_roi_diagonal(
    root: zarr.Group,
    *,
    source_crop_run: Optional[str],
) -> Optional[float]:
    if not source_crop_run:
        return None
    crop_group = root.get(f"crop_runs/{source_crop_run}")
    if crop_group is None or "roi_images" not in crop_group:
        return None
    try:
        shape = crop_group["roi_images"].shape
        roi_h = float(shape[1])
        roi_w = float(shape[2])
    except Exception:
        return None
    diagonal = float(np.hypot(roi_w, roi_h))
    return diagonal if np.isfinite(diagonal) and diagonal > 0 else None


def _compute_heading_from_points(
    bladder: np.ndarray, eye_left: np.ndarray, eye_right: np.ndarray
) -> float:
    """Compute heading (degrees) from keypoint geometry."""
    eye_mean = (eye_left + eye_right) / 2.0
    head_vec = eye_mean - bladder
    if not np.all(np.isfinite(head_vec)) or np.linalg.norm(head_vec) == 0:
        return float("nan")
    return float(np.rad2deg(np.arctan2(-head_vec[1], head_vec[0])))


def _detect_eye_flip(
    bladder: np.ndarray,
    eye_left: np.ndarray,
    eye_right: np.ndarray,
    heading_deg: float,
) -> bool:
    """Return True if left/right eye assignments appear swapped."""
    if not np.isfinite(heading_deg):
        heading_deg = _compute_heading_from_points(bladder, eye_left, eye_right)
    if not np.isfinite(heading_deg):
        return False

    theta = np.deg2rad(heading_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    eye_mean = (eye_left + eye_right) / 2.0
    pts = np.vstack([bladder, eye_left, eye_right])
    rotated = (pts - eye_mean) @ rotation.T

    left_y = rotated[1, 1]
    right_y = rotated[2, 1]
    if not (np.isfinite(left_y) and np.isfinite(right_y)):
        return False
    # In the detector, the right eye ends up with the larger y value in this frame.
    return left_y > right_y


def _process_refinement_chunk(
    zarr_path: str,
    keypoint_run: str,
    start: int,
    end: int,
    params_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Process a slice of keypoints and return refinement outputs."""
    root = zarr.open(zarr_path, mode="r")
    kp_source = root[f"keypoints_runs/{keypoint_run}"]
    min_triangle_angle_default, min_triangle_area_default, max_triangle_area_default = (
        _resolve_refinement_geometry_defaults(kp_source.attrs)
    )

    idx = slice(start, end)

    kp_roi_src = kp_source["keypoints_roi"][idx]
    kp_img_src = kp_source["keypoints_img"][idx]
    kp_norm_src = kp_source["keypoints_norm"][idx]
    heading_src = kp_source["heading"][idx]
    confidence_src = kp_source["confidence"][idx]
    kp_conf_src = (
        kp_source["keypoint_confidences"][idx]
        if "keypoint_confidences" in kp_source
        else None
    )
    thresh_src = (
        kp_source["effective_threshold"][idx]
        if "effective_threshold" in kp_source
        else None
    )
    se2_src = (
        kp_source["effective_se2_radius"][idx]
        if "effective_se2_radius" in kp_source
        else None
    )
    success_chunk = kp_source["detection_success"][idx]
    keypoint_labels = _normalize_keypoint_labels(kp_source, n_keypoints=kp_roi_src.shape[1])

    length = end - start

    roi_out = np.full_like(kp_roi_src, np.nan)
    img_out = np.full_like(kp_img_src, np.nan)
    norm_out = np.full_like(kp_norm_src, np.nan)
    heading_out = np.full_like(heading_src, np.nan)
    confidence_out = np.full_like(confidence_src, np.nan)
    kp_conf_out = (
        np.full_like(kp_conf_src, np.nan) if kp_conf_src is not None else None
    )
    thresh_out = (
        np.full_like(thresh_src, np.nan) if thresh_src is not None else None
    )
    se2_out = np.full_like(se2_src, np.nan) if se2_src is not None else None

    area_out = np.full(length, np.nan, dtype=np.float64)
    min_angle_out = np.full(length, np.nan, dtype=np.float64)
    triangle_angles_out = np.full((length, 3), np.nan, dtype=np.float64)
    quality_out = np.zeros(length, dtype=np.int8)
    refined_success_out = np.zeros(length, dtype=bool)
    flip_flags_out = np.zeros(length, dtype=bool)
    confidence_valid_out = np.zeros(length, dtype=bool)
    geometry_valid_out = np.zeros(length, dtype=bool)
    usable_out = np.zeros(length, dtype=bool)
    reason_out = np.full(length, "", dtype=object)

    confidence_threshold = float(
        params_dict.get("confidence_threshold", _DEFAULT_CONFIDENCE_THRESHOLD)
    )
    min_triangle_angle = float(
        params_dict.get("min_triangle_angle", min_triangle_angle_default)
    )
    min_triangle_area = float(
        params_dict.get("min_triangle_area", min_triangle_area_default)
    )
    max_tri_val = params_dict.get("max_triangle_area")
    try:
        max_triangle_area = (
            float(max_tri_val)
            if max_tri_val is not None
            else max_triangle_area_default
        )
    except (TypeError, ValueError):
        max_triangle_area = max_triangle_area_default

    stats = {
        "refined_success": 0,
        "source_success": int(np.sum(success_chunk)),
        "source_failures": int(len(success_chunk) - int(np.sum(success_chunk))),
        "flips_corrected": 0,
        "low_confidence": 0,
        "confidence_missing": 0,
        "geometry_issues": 0,
        "clean": 0,
        "usable": 0,
    }

    for i in range(length):
        if not success_chunk[i]:
            quality_out[i] = 4  # source detection failed
            reason_out[i] = "detection_failed"
            continue

        metrics: KeypointGeometryMetrics = compute_geometry_metrics(kp_roi_src[i])
        area_out[i] = metrics.area
        min_angle_out[i] = metrics.min_angle
        triangle_angles_out[i] = metrics.angles

        roi_out[i] = kp_roi_src[i]
        img_out[i] = kp_img_src[i]
        norm_out[i] = kp_norm_src[i]
        confidence_out[i] = confidence_src[i]
        if kp_conf_out is not None and kp_conf_src is not None:
            kp_conf_out[i] = kp_conf_src[i]

        heading_for_flip = _compute_heading_from_points(
            roi_out[i][0], roi_out[i][1], roi_out[i][2]
        )

        if thresh_out is not None and thresh_src is not None:
            thresh_out[i] = thresh_src[i]
        if se2_out is not None and se2_src is not None:
            se2_out[i] = se2_src[i]

        flip_detected = _detect_eye_flip(
            roi_out[i][0], roi_out[i][1], roi_out[i][2], heading_for_flip
        )
        if flip_detected:
            # Swap left/right eyes in all coordinate spaces
            roi_out[i][[1, 2]] = roi_out[i][[2, 1]]
            img_out[i][[1, 2]] = img_out[i][[2, 1]]
            norm_out[i][[1, 2]] = norm_out[i][[2, 1]]
            if kp_conf_out is not None:
                kp_conf_out[i][[1, 2]] = kp_conf_out[i][[2, 1]]
            flip_flags_out[i] = True
            quality_out[i] = 6  # Flag flip correction
            stats["flips_corrected"] += 1
        else:
            quality_out[i] = 0

        heading_out[i] = compute_heading_from_attrs(
            kp_source.attrs,
            labels=keypoint_labels,
            points=roi_out[i],
        )

        conf_missing = False
        conf_ok = False
        if kp_conf_out is None:
            conf_missing = True
        else:
            conf_vals = kp_conf_out[i]
            if not np.all(np.isfinite(conf_vals)):
                conf_missing = True
            else:
                conf_ok = bool(np.all(conf_vals >= confidence_threshold))
        confidence_valid_out[i] = conf_ok

        max_ok = max_triangle_area is None or metrics.area <= max_triangle_area
        geom_ok = bool(
            np.isfinite(metrics.min_angle)
            and np.isfinite(metrics.area)
            and metrics.min_angle >= min_triangle_angle
            and metrics.area >= min_triangle_area
            and max_ok
        )
        geometry_valid_out[i] = geom_ok

        tags: List[str] = []
        if flip_detected:
            tags.append("flip_corrected")
        if conf_missing:
            tags.append("confidence_missing")
            stats["confidence_missing"] += 1
        elif not conf_ok:
            tags.append("low_confidence")
            stats["low_confidence"] += 1
        if not geom_ok:
            tags.append("geometry_issue")
            stats["geometry_issues"] += 1

        if tags:
            reason_out[i] = "|".join(tags)
        else:
            reason_out[i] = "clean"
            stats["clean"] += 1

        usable = conf_ok and geom_ok
        usable_out[i] = usable
        if usable:
            stats["usable"] += 1

        refined_success_out[i] = True
        stats["refined_success"] += 1

    return {
        "start": start,
        "end": end,
        "quality": quality_out,
        "refined_success": refined_success_out,
        "roi": roi_out,
        "img": img_out,
        "norm": norm_out,
        "heading": heading_out,
        "confidence": confidence_out,
        "kp_conf": kp_conf_out,
        "thresh": thresh_out,
        "se2": se2_out,
        "flip_flags": flip_flags_out,
        "area": area_out,
        "min_angle": min_angle_out,
        "triangle_angles": triangle_angles_out,
        "confidence_valid": confidence_valid_out,
        "geometry_valid": geometry_valid_out,
        "usable": usable_out,
        "reason": reason_out,
        "stats": stats,
    }


def create_refined_keypoint_run(
    zarr_path: str,
    keypoint_run: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    console: Optional[Console] = None,
    *,
    command: Optional[str] = None,
    created_at_utc: Optional[str] = None,
) -> str:
    """
    Create a refined keypoint run with geometry-based validation.

    Returns name of the created run.
    """
    if console is None:
        console = Console()

    console.rule("[bold]Keypoint Refinement[/bold]")
    start_time = time.perf_counter()

    root = zarr.open(zarr_path, mode="a")
    status_context = _resolve_status_context_from_root(root, zarr_path)
    if status_context is None:
        status_context = _resolve_status_context(zarr_path)
    if "keypoints_runs" not in root or root["keypoints_runs"].attrs.get("latest") is None:
        _emit_refined_keypoint_status(
            context=status_context,
            status="absent",
            run_name=None,
            method="refine_keypoints",
            coverage_pct=None,
            review_status_json=None,
            details={
                "reason": "keypoints_missing",
                "error": "No keypoint runs found. Run keypoint detection first.",
            },
            console=console,
        )
        raise RuntimeError("No keypoint runs found. Run keypoint detection first.")

    if keypoint_run is None:
        keypoint_run = root["keypoints_runs"].attrs["latest"]

    kp_source = root[f"keypoints_runs/{keypoint_run}"]
    console.print(f"Source keypoint run: [cyan]{keypoint_run}[/cyan]")

    params_config = (config or {}).get("refine_keypoints", {}) if config else {}
    params, param_source = KeypointRefinementParams.from_config(
        params_config,
        source_attrs=kp_source.attrs,
    )

    console.print(f"Parameter source: [cyan]{param_source}[/cyan]")
    console.print(f"  Chunk size: {params.chunk_size}")
    console.print(f"  Scheduler: {params.scheduler}")
    if params.num_workers is not None:
        console.print(f"  Num workers: {params.num_workers}")
    console.print(f"  Confidence threshold: {params.confidence_threshold}")
    console.print(f"  Min triangle angle: {params.min_triangle_angle}")
    console.print(f"  Min triangle area: {params.min_triangle_area}")
    if params.max_triangle_area is not None:
        console.print(f"  Max triangle area: {params.max_triangle_area}")

    total_rois = kp_source["keypoints_roi"].shape[0]
    n_keypoints = int(kp_source["keypoints_roi"].shape[1])
    console.print(f"Total ROI keypoints: {total_rois}")
    if total_rois == 0:
        _emit_refined_keypoint_status(
            context=status_context,
            status="missing",
            run_name=None,
            method="refine_keypoints",
            coverage_pct=None,
            review_status_json=None,
            details={
                "reason": "no_rois",
                "source_keypoints_run": keypoint_run,
            },
            console=console,
        )
        raise RuntimeError("Keypoint run contains zero ROIs; nothing to refine.")

    keypoint_labels = _normalize_keypoint_labels(kp_source, n_keypoints=n_keypoints)
    edge_pairs, edge_source = _extract_skeleton_edges(kp_source, n_keypoints=n_keypoints)
    edge_count = int(edge_pairs.shape[0])
    edge_labels = [f"{keypoint_labels[int(src)]}-{keypoint_labels[int(dst)]}" for src, dst in edge_pairs.tolist()]
    console.print(f"Edge-distance pairs: {edge_count} ({edge_source})")
    derived_metric_schema = resolve_metric_schema_for_group(kp_source, required=False)
    if derived_metric_schema is not None:
        console.print(
            "Derived metric schema: "
            f"[cyan]{derived_metric_schema.schema_name}[/cyan] "
            f"({len(derived_metric_schema.metrics)} metrics)"
        )

    # Prepare destination group
    kp_refined_root = _ensure_group(root, REFINED_KEYPOINT_GROUP)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"refined_keypoints_{timestamp}"
    kp_refined = kp_refined_root.create_group(run_name)
    kp_refined_root.attrs["latest"] = run_name

    created_timestamp = created_at_utc or datetime.now(timezone.utc).isoformat()

    source_crop_run = kp_source.attrs.get("source_crop_run")
    source_detect_run = kp_source.attrs.get("source_detect_run")
    source_refined_run = kp_source.attrs.get("source_refined_run")
    source_crop_group = (
        root.get(f"crop_runs/{source_crop_run}") if isinstance(source_crop_run, str) and source_crop_run else None
    )
    source_crop_snapshot_attrs = build_source_crop_snapshot_attrs(
        source_crop_group.attrs if source_crop_group is not None else None,
        source_crop_storage_mode=(
            (
                source_crop_group.attrs.get("crop_storage_mode")
                or ("materialized" if source_crop_group.get("roi_images") is not None else "geometry_only")
            )
            if source_crop_group is not None
            else None
        ),
    )

    kp_refined.attrs.update(
        {
            "source_keypoints_run": keypoint_run,
            "chunk_size": params.chunk_size,
            "scheduler": params.scheduler,
            "num_workers": params.num_workers,
            "memory_limit": params.memory_limit,
            "refinement_role": "left_right_eye_check",
            "created_utc": created_timestamp,
            **source_crop_snapshot_attrs,
        }
    )
    if source_crop_run:
        kp_refined.attrs["source_crop_run"] = source_crop_run
    if source_detect_run:
        kp_refined.attrs["source_detect_run"] = source_detect_run
    if source_refined_run:
        kp_refined.attrs["source_refined_run"] = source_refined_run

    if "keypoint_labels" in kp_source.attrs:
        kp_refined.attrs["keypoint_labels"] = kp_source.attrs["keypoint_labels"]
    else:
        kp_refined.attrs["keypoint_labels"] = keypoint_labels

    # Copy pose schema if present in source run
    if "pose_schema" in kp_source.attrs:
        kp_refined.attrs["pose_schema"] = kp_source.attrs["pose_schema"]

    if edge_count > 0:
        edge_chunks = (max(1, min(128, edge_count)), 2)
        kp_refined.create_array(
            "edge_pairs",
            data=edge_pairs,
            chunks=edge_chunks,
            overwrite=True,
        )
        kp_refined.attrs["edge_distance_labels"] = edge_labels
    kp_refined.attrs["edge_distance_source"] = edge_source

    # Copy metadata arrays (frame indices, counts, etc.)
    for meta_name in ("frame_indices", "n_rois", "frame_counts", "detection_indices"):
        if meta_name in kp_source:
            _copy_array(kp_source[meta_name], kp_refined, meta_name)

    heading_chunks = kp_source["heading"].chunks or (min(1024, total_rois),)
    if "detection_source" in kp_source:
        source_detection_array = kp_source["detection_source"]
        detection_source_values = source_detection_array[:].astype("i1", copy=False)
        det_chunks = source_detection_array.chunks or heading_chunks
    else:
        detection_source_values = np.zeros(total_rois, dtype="i1")
        det_chunks = heading_chunks
    detection_source_dst = kp_refined.create_array(
        "detection_source",
        shape=(total_rois,),
        chunks=det_chunks,
        dtype="i1",
        fill_value=0,
        overwrite=True,
    )
    detection_source_dst[:] = detection_source_values
    kp_refined.create_array(
        "retune_id",
        shape=(total_rois,),
        chunks=heading_chunks,
        dtype="i4",
        fill_value=-1,
        overwrite=True,
    )

    # Prepare output arrays
    kp_roi_dst = kp_refined.create_array(
        "keypoints_roi",
        shape=kp_source["keypoints_roi"].shape,
        chunks=kp_source["keypoints_roi"].chunks,
        dtype="f8",
        fill_value=np.nan,
        overwrite=True,
    )
    kp_img_dst = kp_refined.create_array(
        "keypoints_img",
        shape=kp_source["keypoints_img"].shape,
        chunks=kp_source["keypoints_img"].chunks,
        dtype="f8",
        fill_value=np.nan,
        overwrite=True,
    )
    kp_norm_dst = kp_refined.create_array(
        "keypoints_norm",
        shape=kp_source["keypoints_norm"].shape,
        chunks=kp_source["keypoints_norm"].chunks,
        dtype="f8",
        fill_value=np.nan,
        overwrite=True,
    )

    heading_dst = kp_refined.create_array(
        "heading",
        shape=kp_source["heading"].shape,
        chunks=kp_source["heading"].chunks,
        dtype="f8",
        fill_value=np.nan,
        overwrite=True,
    )
    confidence_dst = kp_refined.create_array(
        "confidence",
        shape=kp_source["confidence"].shape,
        chunks=kp_source["confidence"].chunks,
        dtype="f8",
        fill_value=np.nan,
        overwrite=True,
    )
    kp_conf_dst = None
    if "keypoint_confidences" in kp_source:
        kp_conf_dst = kp_refined.create_array(
            "keypoint_confidences",
            shape=kp_source["keypoint_confidences"].shape,
            chunks=kp_source["keypoint_confidences"].chunks,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
        )
    thresh_dst = None
    if "effective_threshold" in kp_source:
        thresh_dst = kp_refined.create_array(
            "effective_threshold",
            shape=kp_source["effective_threshold"].shape,
            chunks=kp_source["effective_threshold"].chunks,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
        )
    se2_dst = None
    if "effective_se2_radius" in kp_source:
        se2_dst = kp_refined.create_array(
            "effective_se2_radius",
            shape=kp_source["effective_se2_radius"].shape,
            chunks=kp_source["effective_se2_radius"].chunks,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
        )

    # Diagnostics
    geom_area_dst = kp_refined.create_array(
        "triangle_area",
        shape=(total_rois,),
        chunks=kp_source["heading"].chunks,
        dtype="f8",
        fill_value=np.nan,
        overwrite=True,
    )
    geom_min_angle_dst = kp_refined.create_array(
        "min_angle",
        shape=(total_rois,),
        chunks=kp_source["heading"].chunks,
        dtype="f8",
        fill_value=np.nan,
        overwrite=True,
    )
    geom_angles_dst = kp_refined.create_array(
        "triangle_angles",
        shape=(total_rois, 3),
        chunks=(
            kp_source["heading"].chunks[0]
            if kp_source["heading"].chunks
            else min(1024, total_rois),
            3,
        ),
        dtype="f8",
        fill_value=np.nan,
        overwrite=True,
    )

    quality_dst = kp_refined.create_array(
        "quality_labels",
        shape=(total_rois,),
        chunks=kp_source["heading"].chunks,
        dtype="i1",
        fill_value=0,
        overwrite=True,
    )
    refined_success_dst = kp_refined.create_array(
        "refined_success",
        shape=(total_rois,),
        chunks=kp_source["heading"].chunks,
        dtype="bool",
        fill_value=False,
        overwrite=True,
    )
    confidence_valid_dst = kp_refined.create_array(
        "confidence_valid",
        shape=(total_rois,),
        chunks=kp_source["heading"].chunks,
        dtype="bool",
        fill_value=False,
        overwrite=True,
    )
    geometry_valid_dst = kp_refined.create_array(
        "geometry_valid",
        shape=(total_rois,),
        chunks=kp_source["heading"].chunks,
        dtype="bool",
        fill_value=False,
        overwrite=True,
    )
    usable_dst = kp_refined.create_array(
        "usable_keypoints",
        shape=(total_rois,),
        chunks=kp_source["heading"].chunks,
        dtype="bool",
        fill_value=False,
        overwrite=True,
    )
    reason_chunk = int(heading_chunks[0]) if heading_chunks else max(1, min(1024, total_rois))
    reason_values = np.full(int(total_rois), "", dtype=object)
    kp_refined.create_array(
        "edit_applied",
        shape=(total_rois,),
        chunks=heading_chunks,
        dtype="bool",
        fill_value=False,
        overwrite=True,
    )
    source_success = kp_source["detection_success"][:]
    kp_refined.create_array(
        "source_success",
        data=source_success,
        chunks=kp_source["detection_success"].chunks,
        overwrite=True,
    )
    flip_dst = kp_refined.create_array(
        "flip_corrected",
        shape=(total_rois,),
        chunks=kp_source["heading"].chunks,
        dtype="bool",
        fill_value=False,
        overwrite=True,
    )
    heading_finite_dst = kp_refined.create_array(
        "heading_finite",
        shape=(total_rois,),
        chunks=heading_chunks,
        dtype="bool",
        fill_value=False,
        overwrite=True,
    )
    heading_usable_dst = kp_refined.create_array(
        "heading_usable",
        shape=(total_rois,),
        chunks=heading_chunks,
        dtype="bool",
        fill_value=False,
        overwrite=True,
    )

    edge_distances_dst = None
    edge_distances_norm_dst = None
    edge_distance_valid_dst = None
    derived_metric_storage: DerivedMetricStorage | None = None
    roi_diagonal = _resolve_roi_diagonal(root, source_crop_run=normalize_attr(source_crop_run))
    if edge_count > 0:
        edge_chunks = (heading_chunks[0] if heading_chunks else max(1, min(1024, total_rois)), edge_count)
        edge_distances_dst = kp_refined.create_array(
            "edge_distances",
            shape=(total_rois, edge_count),
            chunks=edge_chunks,
            dtype="f4",
            fill_value=np.nan,
            overwrite=True,
        )
        edge_distances_norm_dst = kp_refined.create_array(
            "edge_distances_norm",
            shape=(total_rois, edge_count),
            chunks=edge_chunks,
            dtype="f4",
            fill_value=np.nan,
            overwrite=True,
        )
        edge_distance_valid_dst = kp_refined.create_array(
            "edge_distance_valid",
            shape=(total_rois, edge_count),
            chunks=edge_chunks,
            dtype="bool",
            fill_value=False,
            overwrite=True,
        )
        kp_refined.attrs["edge_distance_normalization"] = {
            "mode": "roi_diagonal",
            "roi_diagonal": roi_diagonal,
        }
    if derived_metric_schema is not None:
        derived_metric_storage = ensure_derived_metric_storage(
            kp_refined,
            schema=derived_metric_schema,
            row_count=total_rois,
            chunk_len=int(heading_chunks[0]) if heading_chunks else max(1, min(1024, total_rois)),
            roi_diagonal=roi_diagonal,
            overwrite=True,
        )

    stats = {
        "total": int(total_rois),
        "source_success": int(np.sum(source_success)),
        "refined_success": 0,
        "source_failures": int(total_rois - np.sum(source_success)),
        "flips_corrected": 0,
        "low_confidence": 0,
        "confidence_missing": 0,
        "geometry_issues": 0,
        "clean": 0,
        "usable": 0,
    }

    console.print("\nEvaluating keypoints for eye flips...")
    console.print(
        f"  Scheduler: {params.scheduler}"
        + (f" | Workers: {params.num_workers or 'default'}" if params.scheduler != "single-threaded" else "")
    )

    chunk_len = max(1, params.chunk_size)
    chunk_tasks = []
    for start in range(0, total_rois, chunk_len):
        end = min(start + chunk_len, total_rois)
        chunk_tasks.append(
            delayed(_process_refinement_chunk)(
                zarr_path,
                keypoint_run,
                int(start),
                int(end),
                {
                    "confidence_threshold": params.confidence_threshold,
                    "min_triangle_angle": params.min_triangle_angle,
                    "min_triangle_area": params.min_triangle_area,
                    "max_triangle_area": params.max_triangle_area,
                },
            )
        )

    cluster = None
    client = None
    results_list: List[Dict[str, Any]] = []
    if chunk_tasks:
        try:
            if params.scheduler == "distributed":
                if not HAVE_DISTRIBUTED:
                    raise RuntimeError(
                        "Dask distributed is not available. Install dask[distributed] "
                        "or choose a different scheduler (e.g. 'processes' or 'threads')."
                    )
                cluster_kwargs: Dict[str, Any] = {}
                if params.num_workers is not None:
                    cluster_kwargs["n_workers"] = params.num_workers
                if params.memory_limit is not None:
                    cluster_kwargs["memory_limit"] = params.memory_limit
                cluster = LocalCluster(**cluster_kwargs)
                client = Client(cluster)
                futures = client.compute(chunk_tasks)
                results_list = list(client.gather(futures))
            else:
                compute_kwargs: Dict[str, Any] = {"scheduler": params.scheduler}
                if params.num_workers is not None:
                    compute_kwargs["num_workers"] = params.num_workers
                compute_result = dask.compute(*chunk_tasks, **compute_kwargs)
                results_list = list(compute_result) if isinstance(compute_result, tuple) else list(compute_result)
        finally:
            if client is not None:
                client.close()
            if cluster is not None:
                cluster.close()

    for result in results_list:
        start = result["start"]
        end = result["end"]
        idx = slice(start, end)

        kp_roi_dst[idx] = result["roi"]
        kp_img_dst[idx] = result["img"]
        kp_norm_dst[idx] = result["norm"]
        heading_dst[idx] = result["heading"]
        confidence_dst[idx] = result["confidence"]
        if kp_conf_dst is not None and result.get("kp_conf") is not None:
            kp_conf_dst[idx] = result["kp_conf"]
        if thresh_dst is not None and result["thresh"] is not None:
            thresh_dst[idx] = result["thresh"]
        if se2_dst is not None and result["se2"] is not None:
            se2_dst[idx] = result["se2"]
        geom_area_dst[idx] = result["area"]
        geom_min_angle_dst[idx] = result["min_angle"]
        geom_angles_dst[idx] = result["triangle_angles"]
        quality_dst[idx] = result["quality"]
        refined_success_dst[idx] = result["refined_success"]
        confidence_valid_dst[idx] = result["confidence_valid"]
        geometry_valid_dst[idx] = result["geometry_valid"]
        usable_dst[idx] = result["usable"]
        reason_values[idx] = np.asarray(result["reason"], dtype=object)
        flip_dst[idx] = result["flip_flags"]
        if edge_distances_dst is not None and edge_distances_norm_dst is not None and edge_distance_valid_dst is not None:
            roi_chunk = np.asarray(result["roi"], dtype=np.float64)
            src_points = roi_chunk[:, edge_pairs[:, 0], :]
            dst_points = roi_chunk[:, edge_pairs[:, 1], :]
            valid = (
                np.asarray(result["refined_success"], dtype=bool)[:, None]
                & np.all(np.isfinite(src_points), axis=2)
                & np.all(np.isfinite(dst_points), axis=2)
            )
            deltas = src_points - dst_points
            distances = np.sqrt(np.sum(np.square(deltas), axis=2, dtype=np.float64))
            distances[~valid] = np.nan
            if roi_diagonal is not None and np.isfinite(roi_diagonal) and roi_diagonal > 0:
                distances_norm = distances / float(roi_diagonal)
            else:
                distances_norm = np.full_like(distances, np.nan)
            edge_distances_dst[idx] = distances.astype(np.float32, copy=False)
            edge_distances_norm_dst[idx] = distances_norm.astype(np.float32, copy=False)
            edge_distance_valid_dst[idx] = valid
        if derived_metric_storage is not None:
            metric_points = np.asarray(result["roi"], dtype=np.float64)
            metric_points[~np.asarray(result["refined_success"], dtype=bool)] = np.nan
            update_derived_metric_rows(
                derived_metric_storage,
                row_indexer=idx,
                keypoints_roi=metric_points,
                keypoint_labels=keypoint_labels,
                roi_diagonal=roi_diagonal,
            )

        stats["refined_success"] += result["stats"]["refined_success"]
        stats["flips_corrected"] += result["stats"]["flips_corrected"]
        stats["low_confidence"] += result["stats"]["low_confidence"]
        stats["confidence_missing"] += result["stats"]["confidence_missing"]
        stats["geometry_issues"] += result["stats"]["geometry_issues"]
        stats["clean"] += result["stats"]["clean"]
        stats["usable"] += result["stats"]["usable"]

    _write_reason_arrays(kp_refined, reason_values, reason_chunk)

    duration = time.perf_counter() - start_time

    pass_rate = stats["refined_success"] / stats["total"] * 100 if stats["total"] else 0.0
    summary_statistics = {
        "total_rois": stats["total"],
        "source_success": stats["source_success"],
        "source_failures": stats["source_failures"],
        "refined_success": stats["refined_success"],
        "flips_corrected": stats["flips_corrected"],
        "low_confidence": stats["low_confidence"],
        "confidence_missing": stats["confidence_missing"],
        "geometry_issues": stats["geometry_issues"],
        "clean": stats["clean"],
        "usable_keypoints": stats["usable"],
        "confidence_threshold": params.confidence_threshold,
        "min_triangle_angle": params.min_triangle_angle,
        "min_triangle_area": params.min_triangle_area,
        "max_triangle_area": params.max_triangle_area,
        "edge_distance_count": edge_count,
        "edge_distance_source": edge_source,
        "edge_distance_normalization": "roi_diagonal",
        "edge_distance_roi_diagonal": roi_diagonal,
        "derived_metric_schema_id": derived_metric_schema.schema_name if derived_metric_schema is not None else None,
        "derived_metric_count": len(derived_metric_schema.metrics) if derived_metric_schema is not None else 0,
        "pass_rate_percent": pass_rate,
        "duration_seconds": duration,
    }

    failure_indices = np.where(~source_success)[0].astype("i4", copy=False)
    failure_chunk = (max(1, min(10000, failure_indices.size)),)
    kp_refined.create_array(
        "failure_indices",
        data=failure_indices,
        chunks=failure_chunk,
        overwrite=True,
    )

    git_info = get_git_info()
    env_info = get_environment_info(
        include_all_packages=False,
        disk_path=str(zarr_path),
        collect_ip=False,
        capture_env_vars=False,
    )
    environment_summary = env_info.get("environment")

    scheduler_info: Optional[Dict[str, object]] = None
    if params.scheduler:
        scheduler_info = {"type": params.scheduler}
        if params.num_workers is not None:
            scheduler_info["num_workers"] = int(params.num_workers)
        if params.memory_limit is not None:
            scheduler_info["memory_limit"] = params.memory_limit

    platform_info = {
        "hostname": env_info["platform"].get("hostname", "unknown"),
        "python_version": env_info["platform"].get("python_version", "unknown"),
        "system": env_info["platform"].get("system", "unknown"),
        "release": env_info["platform"].get("release", "unknown"),
        "machine": env_info["platform"].get("machine", "unknown"),
    }

    parameters_info = {
        "chunk_size": params.chunk_size,
        "scheduler": params.scheduler,
        "num_workers": params.num_workers,
        "memory_limit": params.memory_limit,
        "parameter_source": param_source,
        "confidence_threshold": params.confidence_threshold,
        "min_triangle_angle": params.min_triangle_angle,
        "min_triangle_area": params.min_triangle_area,
        "max_triangle_area": params.max_triangle_area,
    }
    kp_refined.attrs["parameter_source"] = param_source
    kp_refined.attrs["parameters"] = parameters_info

    artifact_keys = [
        "model_checkpoint",
        "model_name",
        "model_version",
        "source_checkpoint",
        "inference_device",
        "inference_batch_size",
        "dataset_meta",
    ]
    artifact_info = {key: kp_source.attrs[key] for key in artifact_keys if key in kp_source.attrs}

    frame_source = "zarr"
    source_video_path = root.attrs.get("source_video_path")
    if source_crop_group is not None:
        frame_source = source_crop_group.attrs.get("video_source_type", frame_source)
        source_video_path = source_crop_group.attrs.get("video_source_path") or source_video_path

    provenance_record = build_stage_provenance(
        stage="refine_keypoints",
        command=command or "unknown",
        created_at_utc=created_timestamp,
        version=git_info.get("short_hash") or git_info.get("commit_hash"),
        git={
            "commit": git_info.get("commit_hash"),
            "short": git_info.get("short_hash"),
            "branch": git_info.get("branch"),
            "is_dirty": git_info.get("is_dirty"),
            "remote": git_info.get("remote_url"),
        },
        environment=environment_summary,
        platform=platform_info,
        scheduler=scheduler_info,
        parameters=parameters_info,
        inputs={
            "keypoints_run": keypoint_run,
            "source_crop_run": source_crop_run,
            **source_crop_snapshot_attrs,
            "source_refined_run": source_refined_run,
            "frame_source": frame_source,
            "source_video_path": source_video_path,
        },
        artifacts=artifact_info,
    )

    write_stage_provenance(kp_refined, provenance_record)
    kp_refined.attrs["keypoint_signature"] = _build_keypoint_signature(
        kp_refined.attrs, parameters_info
    )

    temporal_heading_summary = refresh_refined_keypoint_heading_fields(kp_refined, root=root)
    summary_statistics.update(
        {
            "heading_temporal_evaluable": int(temporal_heading_summary["heading_temporal_evaluable"]),
            "heading_temporal_outlier": int(temporal_heading_summary["heading_temporal_outlier_count"]),
            "heading_temporal_outlier_rate_percent": float(
                temporal_heading_summary["heading_temporal_outlier_rate_percent"]
            ),
            "temporal_heading_threshold_deg": float(
                temporal_heading_summary["temporal_heading_threshold_deg"]
            ),
            "temporal_heading_max_frame_gap": int(
                temporal_heading_summary["temporal_heading_max_frame_gap"]
            ),
            "temporal_heading_status": str(temporal_heading_summary.get("temporal_heading_status", "enabled")),
            "temporal_heading_disabled_reason": temporal_heading_summary.get("temporal_heading_disabled_reason"),
        }
    )
    kp_refined.attrs["summary_statistics"] = summary_statistics
    kp_refined.attrs["derived_metrics_schema"] = build_refined_keypoint_derived_metrics_schema(
        keypoint_labels=keypoint_labels
    )

    report_lines = [
        "[bold]Results[/bold]",
        f"  Total ROIs: {stats['total']}",
        f"  Source success: {stats['source_success']}",
        f"  Refined success: {stats['refined_success']}",
        f"  Source failures: {stats['source_failures']}",
        f"  Flips corrected: {stats['flips_corrected']}",
        f"  Low confidence: {stats['low_confidence']}",
        f"  Confidence missing: {stats['confidence_missing']}",
        f"  Geometry issues: {stats['geometry_issues']}",
        f"  Clean: {stats['clean']}",
        f"  Usable keypoints: {stats['usable']}",
        (
            "  Heading temporal heuristic: disabled "
            f"({temporal_heading_summary.get('temporal_heading_disabled_reason') or temporal_heading_summary.get('temporal_heading_status')})"
            if str(temporal_heading_summary.get("temporal_heading_status", "enabled")) != "enabled"
            else "  Heading temporal outliers: "
            f"{int(temporal_heading_summary['heading_temporal_outlier_count'])}/"
            f"{int(temporal_heading_summary['heading_temporal_evaluable'])}"
        ),
        f"  Pass rate: {pass_rate:.1f}%",
        f"  Duration: {duration:.2f}s",
    ]

    console.print("\n".join(report_lines))
    console.print(f"[green]✓[/green] Saved refined keypoints run: [cyan]{run_name}[/cyan]")

    _emit_refined_keypoint_status(
        context=status_context,
        status="ok",
        run_name=run_name,
        method=normalize_attr(kp_refined.attrs.get("method")) or "refine_keypoints",
        coverage_pct=pass_rate,
        review_status_json=_as_mapping(kp_refined.attrs.get("keypoint_review_status")),
        details={
            "reason": "present",
            "source_keypoints_run": keypoint_run,
            "source_crop_run": normalize_attr(kp_refined.attrs.get("source_crop_run")),
            "source_detect_run": normalize_attr(kp_refined.attrs.get("source_detect_run")),
            "source_refined_run": normalize_attr(kp_refined.attrs.get("source_refined_run")),
            "total_rois": stats["total"],
            "refined_success": stats["refined_success"],
            "usable_keypoints": stats["usable"],
            "usable_keypoints_pct": (float(stats["usable"]) / float(stats["total"]) * 100.0) if stats["total"] else None,
            "pass_rate_percent": pass_rate,
        },
        console=console,
    )

    _run_post_refinement_diagnostics(
        zarr_path=zarr_path,
        run_name=run_name,
        source_crop_run=normalize_attr(source_crop_run),
        config=config,
        run_attrs=kp_refined.attrs,
        console=console,
    )

    return run_name


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Refine keypoint detections to correct eye swaps and produce diagnostics."
    )
    parser.add_argument("zarr_path", help="Path to the Palette Zarr archive.")
    parser.add_argument(
        "--keypoint-run",
        help="Keypoint run to refine (defaults to latest in keypoints_runs).",
    )
    parser.add_argument(
        "--config",
        default="configs/fisheye/default.yaml",
        help="Pipeline configuration file (default: configs/fisheye/default.yaml).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Override refinement chunk size.",
    )
    parser.add_argument(
        "--scheduler",
        choices={"processes", "threads", "distributed"},
        help="Override Dask scheduler.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Override number of workers.",
    )
    parser.add_argument(
        "--memory-limit",
        help="Override Dask worker memory limit (e.g. '32GB').",
    )
    parser.add_argument(
        "--review-failures",
        action="store_true",
        help="Launch the manual keypoint failure review tool after refinement.",
    )
    parser.add_argument(
        "--no-post-audit",
        action="store_true",
        help="Disable post-refinement coordinate-space audit JSON export.",
    )
    parser.add_argument(
        "--post-overlap",
        action="store_true",
        help="Enable post-refinement bad-row overlap analysis JSON export.",
    )
    parser.add_argument(
        "--post-audit-output-dir",
        type=str,
        help="Directory for post-refinement audit/overlap JSON files (default: /tmp).",
    )

    args = parser.parse_args(argv)
    console = Console()

    config: Dict[str, Any] = {}
    overrides_applied = False

    cfg_path = Path(args.config) if args.config else None
    if cfg_path and cfg_path.exists():
        with cfg_path.open("r") as f:
            loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                config = loaded
            else:
                console.print(
                    f"[yellow]Warning:[/yellow] Config file '{cfg_path}' did not contain a mapping; ignoring."
                )
    elif cfg_path:
        console.print(
            f"[yellow]Warning:[/yellow] Config file '{cfg_path}' not found; using defaults."
        )

    refine_cfg = config.setdefault("refine_keypoints", {})

    if args.chunk_size is not None:
        refine_cfg["chunk_size"] = args.chunk_size
        overrides_applied = True
    if args.scheduler is not None:
        refine_cfg["scheduler"] = args.scheduler
        overrides_applied = True
    if args.num_workers is not None:
        refine_cfg["num_workers"] = args.num_workers
        overrides_applied = True
    if args.memory_limit is not None:
        refine_cfg["memory_limit"] = args.memory_limit
        overrides_applied = True
    if args.no_post_audit:
        refine_cfg["post_refinement_audit"] = False
        overrides_applied = True
    if args.post_overlap:
        refine_cfg["post_refinement_overlap"] = True
        overrides_applied = True
    if args.post_audit_output_dir is not None:
        refine_cfg["post_refinement_output_dir"] = args.post_audit_output_dir
        overrides_applied = True

    config_to_use = config if (config or overrides_applied) else None

    run_name = create_refined_keypoint_run(
        zarr_path=args.zarr_path,
        keypoint_run=args.keypoint_run,
        config=config_to_use,
        console=console,
    )

    console.print(
        f"[green]✓[/green] Refined keypoints written to "
        f"[bold]refined_keypoints_runs/{run_name}[/bold]"
    )

    if args.review_failures:
        try:
            from ..tune.keypoint_review import run_manual_review
        except Exception as exc:  # pragma: no cover - optional UI dependency
            console.print(f"[yellow]Warning:[/yellow] Review tool unavailable: {exc}")
            return
        console.print("[blue]Launching keypoint review (manual)...[/blue]")
        run_manual_review(args.zarr_path, refined_run=run_name)


if __name__ == "__main__":  # pragma: no cover
    main()
