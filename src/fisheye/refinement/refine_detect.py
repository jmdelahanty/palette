# src/fisheye/refinement/refine_detect.py
"""
Detection Refinement Pipeline

Builds canonical refined-detect surfaces from raw detections.

Workflow:
1. Load detection data and quality labels
2. Filter: Remove jumps/artifacts from the curated surface
3. Write sparse curated `instances/` and `source_detections/`
4. Refresh the dense compatibility root and save metadata
"""

import numpy as np
import zarr
import sys
from json import JSONDecodeError, loads as json_loads
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Any
from rich.console import Console

from ..shared.frame_domains import FrameDomain, FrameDomainError, FrameDomains
from ..shared.experiment_setup import (
    MissingExperimentSetupError,
    ResolvedExperimentSetup,
    resolve_experiment_setup,
)
from ..shared.dish_mask_boundary import (
    DEFAULT_DISH_MASK_BOUNDARY_TOLERANCE_MM,
    apply_dish_mask_boundary_tolerance,
    resolve_dish_mask_boundary_tolerance,
)
from ..shared.detection_tables import (
    read_detection_frame_counts,
    resolve_detection_instance_table,
)
from ..shared.metadata import get_total_frames, get_detection_method
from ..shared.system_metadata import get_environment_info, get_git_info
from ..shared.refined_detect_curation import write_curated_refined_detect_surfaces
from ..registry.stage_complete import emit_stage_completion
from ..shared.run_provenance import build_run_provenance_from_stage_record
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.type_conversions import normalize_attr
from ..shared.zarr_run_completion import (
    is_run_complete,
    mark_run_complete,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
)
from ..shared.zarr_helpers import open_zarr_group_direct
from ..shared.zarr.canonical_detection_manifest import (
    require_active_coordinate_canonical_detection,
    resolve_expected_canonical_detection_manifest_digest,
)

REFINED_DETECT_GROUP = "refined_detect_runs"
LEGACY_REFINED_DETECT_GROUP = "refined_runs"
DEFAULT_DETECT_FAMILY_PATH = "detect_runs"
_REFINED_DETECT_STATUS_SOURCE = "runtime_refine_detect"
_DISH_MASK_QUALITY_LABEL = 5
_REGISTERED_DISH_GATE_QUALITY_LABEL = 6
_REGISTERED_GATE_REQUIREMENTS = frozenset({"off", "if_available", "required"})
_DEPRECATED_INTERPOLATION_OVERRIDE_MESSAGE = (
    "Interpolation overrides are deprecated and unsupported for refine_detect. "
    "The current sparse-first refine_detect workflow always runs with interpolation disabled."
)


def _validate_quality_experiment_setup_binding(
    setup: ResolvedExperimentSetup,
    quality_attrs: Mapping[str, Any],
) -> None:
    """Reject quality output computed under a different acquisition plan."""

    quality_expected = quality_attrs.get("expected_subject_count")
    if quality_expected is not None and int(quality_expected) != setup.expected_subject_count:
        raise ValueError(
            "Detection quality expected-subject count contradicts current experiment setup: "
            f"quality={quality_expected}, setup={setup.expected_subject_count}"
        )
    quality_setup_sha = normalize_attr(quality_attrs.get("experiment_setup_sha256"))
    if quality_setup_sha is None:
        if not setup.legacy:
            raise ValueError(
                "Detection quality run is not bound to the canonical experiment setup"
            )
    elif quality_setup_sha != setup.record_sha256:
        raise ValueError(
            "Detection quality experiment-setup binding is stale or contradictory: "
            f"quality={quality_setup_sha}, current={setup.record_sha256}"
        )


def _normalize_group_path(path: str) -> str:
    value = str(path or "").strip().strip("/")
    if not value:
        raise ValueError("group path must be non-empty")
    if ".." in Path(value).parts:
        raise ValueError(f"group path must not contain '..': {path!r}")
    return value


def _join_group_path(*parts: str) -> str:
    return "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))


def _reject_deprecated_interpolation_overrides(
    *,
    max_gap: Optional[int] = None,
    interpolation_method: Optional[str] = None,
) -> None:
    if max_gap is None and interpolation_method is None:
        return
    flags: list[str] = []
    if max_gap is not None:
        flags.append("--max-gap")
    if interpolation_method is not None:
        flags.append("--method")
    raise ValueError(f"{_DEPRECATED_INTERPOLATION_OVERRIDE_MESSAGE} Remove {' and '.join(flags)}.")


def _emit_refined_detect_status(
    *,
    root: zarr.Group,
    zarr_path: Path,
    status: str,
    run_name: Optional[str],
    method: Optional[str],
    coverage_pct: Optional[float],
    review_status: Optional[Dict[str, object]],
    details: Dict[str, object],
    console: Optional[Console],
) -> None:
    emit_stage_completion(
        root,
        zarr_path,
        step_name="refined_detect",
        status=status,
        source=_REFINED_DETECT_STATUS_SOURCE,
        run_name=run_name,
        method=method,
        coverage_pct=coverage_pct,
        review_status_json=review_status,
        details_json=details,
        console=console,
        warning_label="refined_detect",
        auto_registry_from_env=True,
        require_env_registry_exists=False,
        invalidate_on_ok=True,
        trigger_run_name=run_name,
    )

def _parse_mapping(value: object) -> Optional[Dict[str, object]]:
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "ignore")
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        payload = json_loads(text)
    except JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _read_sampled_import_meta(root: zarr.Group) -> Tuple[bool, Dict[str, Any]]:
    raw = root.get("raw_video")
    if raw is None:
        return False, {}
    attrs = raw.attrs
    import_mode = normalize_attr(attrs.get("import_mode"))
    import_purpose = normalize_attr(attrs.get("import_purpose"))
    frame_step = attrs.get("frame_step")
    has_mapping = "original_frame_indices" in raw

    sampled = False
    if import_mode == "sampled":
        sampled = True
    if import_purpose == "training_data":
        sampled = True
    try:
        if frame_step is not None and int(frame_step) > 1:
            sampled = True
    except Exception:
        pass
    if has_mapping:
        sampled = True

    meta = {
        "import_mode": import_mode,
        "import_purpose": import_purpose,
        "frame_step": int(frame_step) if isinstance(frame_step, (int, np.integer)) else frame_step,
        "has_original_frame_indices": bool(has_mapping),
    }
    return sampled, meta


def _get_sampled_frame_count(root: zarr.Group, detect_group: Optional[zarr.Group]) -> Optional[int]:
    raw = root.get("raw_video")
    if raw is not None:
        if "original_frame_indices" in raw:
            try:
                return int(FrameDomains(root=root).count(FrameDomain.STORED_ZARR))
            except FrameDomainError:
                return int(raw["original_frame_indices"].shape[0])
        if "images_ds" in raw:
            return int(raw["images_ds"].shape[0])
        if "images_full" in raw:
            return int(raw["images_full"].shape[0])
    if detect_group is not None:
        detect_table = resolve_detection_instance_table(detect_group)
        if "frame_row_offsets" in detect_table:
            return int(detect_table["frame_row_offsets"].shape[0]) - 1
        if "frame_counts" in detect_table:
            try:
                return int(
                    FrameDomains(root=root, run_group=detect_table).count(
                        FrameDomain.RUN_FRAME
                    )
                )
            except FrameDomainError:
                return int(detect_table["frame_counts"].shape[0])
    return None


def _coerce_mapping(value: object) -> Optional[Mapping[str, object]]:
    parsed = _parse_mapping(value)
    if parsed is not None:
        return parsed
    return value if isinstance(value, Mapping) else None


def _shape_hw_from_sequence(value: object) -> Optional[Tuple[float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        # Frame/source shapes are stored as [n, h, w] or [h, w].
        if len(value) >= 3:
            return float(value[-2]), float(value[-1])
        return float(value[0]), float(value[1])
    except Exception:
        return None


def _detect_frame_shape_hw(root: zarr.Group, detect_group: zarr.Group) -> Optional[Tuple[float, float]]:
    frame_source_shape = detect_group.attrs.get("frame_source_shape")
    shape = _shape_hw_from_sequence(frame_source_shape)
    if shape is not None:
        return shape

    inference_h = detect_group.attrs.get("inference_height")
    inference_w = detect_group.attrs.get("inference_width")
    try:
        if inference_h is not None and inference_w is not None:
            return float(inference_h), float(inference_w)
    except Exception:
        pass

    raw = root.get("raw_video")
    if raw is not None:
        for key in ("images_ds", "images_ds_rgb", "images_full"):
            if key in raw:
                shape = _shape_hw_from_sequence(raw[key].shape)
                if shape is not None:
                    return shape
    return None


def _read_dish_mask_spec(root: zarr.Group, detect_group: zarr.Group) -> Optional[Dict[str, object]]:
    analysis_meta = root.get("analysis_metadata")
    if analysis_meta is None or "dish_mask" not in analysis_meta.attrs:
        return None
    mask_data = _coerce_mapping(analysis_meta.attrs.get("dish_mask"))
    if not mask_data:
        return None

    shape = str(
        mask_data.get("shape")
        or ("circle" if "detected_circle" in mask_data else "")
        or ("rectangle" if "rectangle" in mask_data else "")
    ).strip().lower()
    if shape not in {"circle", "rectangle"}:
        return None

    metrics = _coerce_mapping(mask_data.get("metrics")) or {}
    mask_hw = _shape_hw_from_sequence(metrics.get("image_shape"))
    if mask_hw is None:
        mask_hw = _detect_frame_shape_hw(root, detect_group)
    if mask_hw is None:
        return None
    mask_h, mask_w = mask_hw
    if mask_h <= 0 or mask_w <= 0:
        return None

    if shape == "circle":
        circle = _coerce_mapping(mask_data.get("detected_circle"))
        if circle is not None:
            center = circle.get("center")
            radius = circle.get("radius")
            if isinstance(center, (list, tuple)) and len(center) >= 2 and radius is not None:
                try:
                    cx_px = float(center[0])
                    cy_px = float(center[1])
                    r_px = float(radius)
                except Exception:
                    cx_px = cy_px = r_px = float("nan")
                if np.isfinite([cx_px, cy_px, r_px]).all() and r_px > 0:
                    return {
                        "enabled": True,
                        "shape": "circle",
                        "center_norm": [cx_px / mask_w, cy_px / mask_h],
                        "radius_norm_x": r_px / mask_w,
                        "radius_norm_y": r_px / mask_h,
                        "mask_image_shape_hw": [mask_h, mask_w],
                        "source": "analysis_metadata.dish_mask",
                    }

        center_norm = metrics.get("center_norm")
        radius_norm = metrics.get("radius_norm")
        if isinstance(center_norm, (list, tuple)) and len(center_norm) >= 2 and radius_norm is not None:
            try:
                cx = float(center_norm[0])
                cy = float(center_norm[1])
                r = float(radius_norm)
                if np.isfinite([cx, cy, r]).all() and r > 0:
                    return {
                        "enabled": True,
                        "shape": "circle",
                        "center_norm": [cx, cy],
                        "radius_norm_x": r,
                        "radius_norm_y": r,
                        "mask_image_shape_hw": [mask_h, mask_w],
                        "source": "analysis_metadata.dish_mask",
                    }
            except Exception:
                pass
        return None

    rectangle = _coerce_mapping(mask_data.get("rectangle"))
    if rectangle is None:
        return None
    roi = rectangle.get("roi")
    if not isinstance(roi, (list, tuple)) or len(roi) < 4:
        return None
    try:
        x, y, w, h = [float(v) for v in roi[:4]]
    except Exception:
        return None
    if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0:
        return None
    return {
        "enabled": True,
        "shape": "rectangle",
        "x_min_norm": x / mask_w,
        "y_min_norm": y / mask_h,
        "x_max_norm": (x + w) / mask_w,
        "y_max_norm": (y + h) / mask_h,
        "mask_image_shape_hw": [mask_h, mask_w],
        "source": "analysis_metadata.dish_mask",
    }


def _dish_mask_inside_bbox_centers(bbox_coords: np.ndarray, spec: Mapping[str, object]) -> np.ndarray:
    bboxes = np.asarray(bbox_coords, dtype=np.float64).reshape(-1, 4)
    if bboxes.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    centers = bboxes[:, :2]
    finite = np.isfinite(centers).all(axis=1)
    shape = str(spec.get("shape") or "")
    if shape == "circle":
        center = np.asarray(spec.get("center_norm"), dtype=np.float64).reshape(2)
        rx = float(spec.get("radius_norm_x") or 0)
        ry = float(spec.get("radius_norm_y") or 0)
        if rx <= 0 or ry <= 0:
            return np.zeros((bboxes.shape[0],), dtype=bool)
        dx = (centers[:, 0] - float(center[0])) / rx
        dy = (centers[:, 1] - float(center[1])) / ry
        return finite & ((dx * dx + dy * dy) <= 1.0)
    if shape == "rectangle":
        x_min = float(spec.get("x_min_norm") or 0)
        y_min = float(spec.get("y_min_norm") or 0)
        x_max = float(spec.get("x_max_norm") or 0)
        y_max = float(spec.get("y_max_norm") or 0)
        return (
            finite
            & (centers[:, 0] >= x_min)
            & (centers[:, 0] <= x_max)
            & (centers[:, 1] >= y_min)
            & (centers[:, 1] <= y_max)
        )
    return finite


def _apply_dish_mask_quality_gate(
    *,
    bbox_coords: np.ndarray,
    detection_quality_labels: np.ndarray,
    mask_spec: Optional[Mapping[str, object]],
) -> Tuple[np.ndarray, Dict[str, object]]:
    labels = np.asarray(detection_quality_labels, dtype=np.int8).reshape(-1).copy()
    stats: Dict[str, object] = {
        "enabled": False,
        "source": None,
        "shape": None,
        "candidate_rows": int(labels.shape[0]),
        "outside_rows": 0,
        "outside_clean_rows": 0,
    }
    if not mask_spec or not bool(mask_spec.get("enabled", False)):
        return labels, stats

    inside = _dish_mask_inside_bbox_centers(bbox_coords, mask_spec)
    outside = ~inside
    clean_outside = outside & (labels == 0)
    labels[clean_outside] = np.int8(_DISH_MASK_QUALITY_LABEL)
    shape = str(mask_spec.get("shape") or "")
    if shape == "circle":
        base_geometry: Dict[str, object] = {
            "center_norm": mask_spec.get("center_norm"),
            "radius_norm_x": mask_spec.get(
                "base_radius_norm_x",
                mask_spec.get("radius_norm_x"),
            ),
            "radius_norm_y": mask_spec.get(
                "base_radius_norm_y",
                mask_spec.get("radius_norm_y"),
            ),
        }
        effective_geometry: Dict[str, object] = {
            "center_norm": mask_spec.get("center_norm"),
            "radius_norm_x": mask_spec.get("radius_norm_x"),
            "radius_norm_y": mask_spec.get("radius_norm_y"),
        }
    else:
        base_geometry = dict(
            mask_spec.get("base_rectangle_norm")
            if isinstance(mask_spec.get("base_rectangle_norm"), Mapping)
            else {
                name: mask_spec.get(name)
                for name in ("x_min_norm", "y_min_norm", "x_max_norm", "y_max_norm")
            }
        )
        effective_geometry = {
            name: mask_spec.get(name)
            for name in ("x_min_norm", "y_min_norm", "x_max_norm", "y_max_norm")
        }
    stats.update(
        {
            "enabled": True,
            "source": mask_spec.get("source"),
            "shape": shape,
            "boundary_tolerance": mask_spec.get("boundary_tolerance"),
            "base_geometry": base_geometry,
            "effective_geometry": effective_geometry,
            "candidate_rows": int(labels.shape[0]),
            "outside_rows": int(np.sum(outside)),
            "outside_clean_rows": int(np.sum(clean_outside)),
        }
    )
    return labels, stats


def _apply_registered_detection_gate(
    *,
    zarr_path: str | Path,
    source_detect_path: str,
    raw_instance_keys: Optional[np.ndarray],
    detection_quality_labels: np.ndarray,
    requirement: str,
    gate_run: Optional[str],
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Join one exact keyed gate or preserve an explicit ungated disposition."""

    mode = str(requirement or "off").strip()
    if mode not in _REGISTERED_GATE_REQUIREMENTS:
        raise ValueError(
            "registered_gate_requirement must be off, if_available, or required."
        )
    labels = np.asarray(detection_quality_labels, dtype=np.int8).reshape(-1).copy()
    stats: Dict[str, object] = {
        "requirement": mode,
        "status": "off" if mode == "off" else "unavailable",
        "applied": False,
        "gate_run": str(gate_run).strip() if gate_run is not None else None,
        "source_detection_path": source_detect_path,
        "row_count": int(labels.shape[0]),
        "rejected_count": 0,
        "rejection_reason": "outside_registered_detection_gate",
    }
    if mode == "off":
        stats["reason"] = (
            "configured_off_gate_run_ignored"
            if gate_run is not None
            else "configured_off"
        )
        return labels, stats
    if raw_instance_keys is None:
        message = "Modern registered gating requires source detection instance_key."
        if mode == "required":
            raise ValueError(message)
        stats.update({"status": "rejected_invalid", "reason": message})
        return labels, stats
    if gate_run is None or not str(gate_run).strip():
        message = "No exact registered detection gate run was configured."
        if mode == "required":
            raise ValueError(message)
        stats["reason"] = message
        return labels, stats
    try:
        from ..analysis_workflows.materializers.registered_detection_gate import (
            validate_registered_detection_gate_consumption,
        )

        evidence = validate_registered_detection_gate_consumption(
            zarr_path,
            source_group_path=source_detect_path,
            gate_run=str(gate_run),
            expected_instance_keys=raw_instance_keys,
            require_modern_operational_selection=True,
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if mode == "required":
            raise ValueError(
                f"Required registered detection gate is invalid: {message}"
            ) from exc
        stats.update({"status": "rejected_invalid", "reason": message})
        return labels, stats
    inside = np.asarray(evidence.pop("inside"), dtype=bool).reshape(-1)
    if inside.shape != labels.shape:
        raise ValueError("Registered gate row count differs from quality labels.")
    outside = ~inside
    labels[outside] = np.int8(_REGISTERED_DISH_GATE_QUALITY_LABEL)
    stats.update(evidence)
    stats.update(
        {
            "status": "applied",
            "applied": True,
            "reason": "exact_keyed_gate_join_validated",
            "rejected_count": int(np.count_nonzero(outside)),
        }
    )
    return labels, stats


def _quality_guardrail_error(detect_run: str, reason: str, quality_run: Optional[str] = None) -> ValueError:
    target = f"quality run '{quality_run}'" if quality_run else "latest quality run"
    return ValueError(
        f"Missing usable detect_quality context for detect run '{detect_run}' ({target}): {reason}. "
        "Run `python -m fisheye.refinement.detect_quality <zarr_path>` for this detect run, "
        "or pass --allow-missing-quality to opt out."
    )


def _require_quality_matches_active_canonical_manifest(
    quality_group: Any | None,
    active_source_manifest: Mapping[str, Any] | None,
) -> None:
    """Fail before refinement writes when quality used another raw authority."""

    if quality_group is None or active_source_manifest is None:
        raise ValueError(
            "Active canonical refinement requires exact detection-quality evidence."
        )
    quality_manifest_digest = str(
        quality_group.attrs.get("source_detect_run_manifest_digest") or ""
    ).strip()
    if quality_manifest_digest != active_source_manifest.get("payload_digest"):
        raise ValueError(
            "Detection quality was computed from a different canonical "
            "manifest digest."
        )


def _group_at(root: zarr.Group, group_path: str) -> zarr.Group:
    group = root
    for part in _normalize_group_path(group_path).split("/"):
        group = group[part]
    return group


def _modern_quality_slice(
    root: zarr.Group,
    quality_group: zarr.Group,
    *,
    quality_group_path: str,
    source_detect_path: str,
    detect_group: zarr.Group,
    total_detections: int,
) -> np.ndarray:
    detect_table = resolve_detection_instance_table(detect_group)
    if not is_run_complete(quality_group):
        raise ValueError(f"modern quality run is incomplete: {quality_group_path}")
    missing = [
        name
        for name in ("detection_quality_labels", "instance_key")
        if name not in quality_group
    ]
    if missing:
        raise ValueError(f"modern quality run is missing arrays: {missing}")
    if "instance_key" not in detect_table:
        raise ValueError("modern keyed quality requires source detect instance_key")
    if np.dtype(detect_table["instance_key"].dtype) != np.dtype(np.uint64):
        raise ValueError("source detect instance_key must be uint64 for modern quality")

    source_group_path = normalize_attr(
        quality_group.attrs.get("source_detection_group_path")
    )
    if not source_group_path:
        raise ValueError("modern quality run has no source_detection_group_path")
    start = 0
    stop = int(quality_group["instance_key"].shape[0])
    if source_group_path != source_detect_path:
        source_group = _group_at(root, source_group_path)
        raw_slices = source_group.attrs.get("source_slices")
        slices = [
            row
            for row in raw_slices
            if isinstance(row, Mapping)
            and normalize_attr(row.get("detect_group_path")) == source_detect_path
        ] if isinstance(raw_slices, list) else []
        if len(slices) != 1:
            raise ValueError(
                "modern collection quality source does not declare exactly one "
                f"slice for {source_detect_path!r}"
            )
        start = int(slices[0].get("start", -1))
        stop = int(slices[0].get("stop", -1))
    if start < 0 or stop < start or stop - start != int(total_detections):
        raise ValueError(
            f"modern quality slice {start}:{stop} does not match detections "
            f"{int(total_detections)}"
        )
    if stop > int(quality_group["detection_quality_labels"].shape[0]):
        raise ValueError("modern quality slice exceeds detection_quality_labels")
    if stop > int(quality_group["instance_key"].shape[0]):
        raise ValueError("modern quality slice exceeds instance_key")
    quality_keys = np.asarray(
        quality_group["instance_key"][start:stop], dtype=np.uint64
    ).reshape(-1)
    detect_keys = np.asarray(detect_table["instance_key"][:], dtype=np.uint64).reshape(-1)
    if not np.array_equal(quality_keys, detect_keys):
        raise ValueError(
            "modern quality instance_key slice does not exactly match the source detect run"
        )
    return np.asarray(
        quality_group["detection_quality_labels"][start:stop], dtype=np.int8
    ).reshape(-1)


def _resolve_modern_quality_group(
    root: zarr.Group,
    *,
    explicit_group_path: Optional[str],
) -> tuple[Optional[str], Optional[zarr.Group]]:
    if explicit_group_path:
        path = _normalize_group_path(explicit_group_path)
        return path, _group_at(root, path)
    parent = root.get("detect_quality_runs")
    if parent is None:
        return None, None
    selected = normalize_attr(parent.attrs.get("latest_complete")) or normalize_attr(
        parent.attrs.get("latest")
    )
    if not selected or selected not in parent:
        return None, None
    return f"detect_quality_runs/{selected}", parent[selected]


def _resolve_detection_quality_labels(
    root: zarr.Group,
    detect_group: zarr.Group,
    *,
    detect_run: str,
    source_detect_path: str,
    quality_run: Optional[str],
    quality_group_path: Optional[str],
    total_detections: int,
    require_quality: bool,
    allow_missing_reason: str,
    console: Optional[Console],
) -> Tuple[np.ndarray, Optional[str], Optional[zarr.Group]]:
    """Resolve per-detection quality labels and fail closed when required."""
    total = int(total_detections)
    modern_path: Optional[str] = None
    modern_group: Optional[zarr.Group] = None
    modern_error: Optional[str] = None
    if quality_group_path or quality_run is None:
        try:
            modern_path, modern_group = _resolve_modern_quality_group(
                root,
                explicit_group_path=quality_group_path,
            )
            if modern_group is not None and modern_path is not None:
                labels = _modern_quality_slice(
                    root,
                    modern_group,
                    quality_group_path=modern_path,
                    source_detect_path=source_detect_path,
                    detect_group=detect_group,
                    total_detections=total,
                )
                resolved = modern_path.rsplit("/", 1)[-1]
                if console is not None:
                    console.print(
                        f"Source quality run: [cyan]{resolved}[/cyan] "
                        f"([dim]{modern_path}[/dim])"
                    )
                return labels, resolved, modern_group
        except Exception as exc:
            modern_error = str(exc)
            if quality_group_path:
                if require_quality:
                    raise _quality_guardrail_error(
                        detect_run=detect_run,
                        reason=modern_error,
                        quality_run=quality_group_path,
                    ) from exc
                modern_group = None

    quality_reports = detect_group.get("quality_reports")
    requested_quality_run = normalize_attr(quality_run)
    resolved_quality_run = requested_quality_run
    quality_group: Optional[zarr.Group] = None

    if quality_reports is not None and resolved_quality_run is None:
        resolved_quality_run = normalize_attr(quality_reports.attrs.get("latest"))

    missing_reason: Optional[str] = None
    if quality_reports is None:
        missing_reason = modern_error or "quality_reports group is missing"
    elif not resolved_quality_run:
        missing_reason = "quality_reports/latest is missing and no --quality-run was provided"
    elif resolved_quality_run not in quality_reports:
        missing_reason = f"quality run '{resolved_quality_run}' was not found"
    else:
        quality_group = quality_reports[resolved_quality_run]
        if "detection_quality_labels" not in quality_group:
            missing_reason = "missing detection_quality_labels array"
        else:
            labels = np.asarray(quality_group["detection_quality_labels"][:], dtype="i1")
            if labels.shape[0] != total:
                missing_reason = (
                    "detection_quality_labels length "
                    f"{int(labels.shape[0])} does not match detections {total}"
                )
            else:
                if console is not None:
                    console.print(f"Source quality run: [cyan]{resolved_quality_run}[/cyan]")
                return labels, resolved_quality_run, quality_group

    if require_quality:
        raise _quality_guardrail_error(
            detect_run=detect_run,
            reason=missing_reason or "quality report is missing",
            quality_run=resolved_quality_run,
        )

    if console is not None:
        console.print(
            "[yellow]⚠ No usable detection quality report found; proceeding with all "
            f"detections marked clean ({allow_missing_reason}).[/yellow]"
        )
    return np.zeros(total, dtype="i1"), None, None

def filter_detections(
    bbox_coords: np.ndarray,
    scores: np.ndarray,
    frame_indices: np.ndarray,
    class_ids: np.ndarray,
    detection_quality_labels: np.ndarray,
    num_frames: int,
    filters: List[str] = ['remove_jumps']
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Filter detections based on quality labels.
    
    Args:
        bbox_coords: Original bounding boxes (N, 4)
        scores: Original confidence scores (N,)
        frame_indices: Frame index for each detection (N,)
        detection_quality_labels: Quality label per detection (N,)
        num_frames: Total frame count for the source video
        filters: List of filters to apply
        
    Returns:
        Tuple of (filtered_bboxes, filtered_scores, frame_counts,
                  frame_indices, class_ids, drop_stats)
    """
    # Determine which detections to keep
    keep_mask = np.ones(len(detection_quality_labels), dtype=bool)
    drop_reasons = {}
    
    if 'remove_jumps' in filters:
        jump_mask = detection_quality_labels == 3
        keep_mask &= ~jump_mask
        drop_reasons['jumps'] = int(np.sum(jump_mask))
    
    if 'remove_blips' in filters:
        blip_mask = detection_quality_labels == 2
        keep_mask &= ~blip_mask
        drop_reasons['blips'] = int(np.sum(blip_mask))
    dish_mask = detection_quality_labels == _DISH_MASK_QUALITY_LABEL
    if np.any(dish_mask):
        drop_reasons['outside_dish_mask'] = int(np.sum(dish_mask))
    registered_gate = detection_quality_labels == _REGISTERED_DISH_GATE_QUALITY_LABEL
    if np.any(registered_gate):
        drop_reasons['outside_registered_detection_gate'] = int(
            np.sum(registered_gate)
        )
    
    # Only keep clean detections (label=0)
    # This also excludes multi-detections (label=4) if present
    keep_mask = detection_quality_labels == 0
    
    # Apply filter
    filtered_bboxes = bbox_coords[keep_mask]
    filtered_scores = scores[keep_mask]
    filtered_frame_indices = frame_indices[keep_mask].astype('i4', copy=False)
    filtered_class_ids = class_ids[keep_mask].astype('i4', copy=False)
    
    # Update per-frame detection counts
    filtered_counts = np.bincount(filtered_frame_indices, minlength=num_frames).astype('i4', copy=False)
    
    # Stats
    drop_stats = {
        'total_dropped': int(np.sum(~keep_mask)),
        'reasons': drop_reasons,
        'kept': int(np.sum(keep_mask)),
        'original': len(detection_quality_labels)
    }
    
    return (
        filtered_bboxes,
        filtered_scores,
        filtered_counts,
        filtered_frame_indices,
        filtered_class_ids,
        drop_stats,
    )


def _select_per_frame_top_k_raw_indices(
    *,
    raw_frame_indices: np.ndarray,
    raw_scores: np.ndarray,
    candidate_raw_indices: np.ndarray,
    per_frame_top_k: Optional[int],
    score_field: str = "scores",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Select the top scoring raw candidates per frame.

    Returns selected raw row indices, duplicate raw row indices, and a small
    provenance/statistics payload. Candidate rows are expected to already be
    eligible for refinement, e.g. clean after quality filtering.
    """

    candidates = np.asarray(candidate_raw_indices, dtype=np.int32).reshape(-1)
    if per_frame_top_k is None:
        return (
            candidates,
            np.empty((0,), dtype=np.int32),
            {
                "enabled": False,
                "per_frame_top_k": None,
                "score_field": str(score_field),
                "candidate_rows": int(candidates.shape[0]),
                "selected_rows": int(candidates.shape[0]),
                "duplicate_rows": 0,
                "frames_with_duplicates": 0,
            },
        )
    if int(per_frame_top_k) < 1:
        raise ValueError("--per-frame-top-k must be a positive integer.")
    if str(score_field) != "scores":
        raise ValueError("Only --top-k-score-field scores is currently supported.")

    k = int(per_frame_top_k)
    frame_indices = np.asarray(raw_frame_indices, dtype=np.int32).reshape(-1)
    scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
    if frame_indices.shape[0] != scores.shape[0]:
        raise ValueError("raw_frame_indices and raw_scores must agree on row count.")
    if candidates.size == 0:
        return (
            candidates,
            np.empty((0,), dtype=np.int32),
            {
                "enabled": True,
                "per_frame_top_k": k,
                "score_field": str(score_field),
                "candidate_rows": 0,
                "selected_rows": 0,
                "duplicate_rows": 0,
                "frames_with_duplicates": 0,
            },
        )
    if int(np.min(candidates)) < 0 or int(np.max(candidates)) >= frame_indices.shape[0]:
        raise ValueError("candidate_raw_indices contains out-of-range rows.")

    selected: list[int] = []
    duplicate: list[int] = []
    candidate_frames = frame_indices[candidates]
    frames_with_duplicates = 0
    for frame in np.unique(candidate_frames):
        rows = candidates[candidate_frames == frame]
        row_scores = scores[rows]
        rank_scores = np.where(np.isfinite(row_scores), row_scores, -np.inf)
        # Primary key: score descending. Tie-breaker: source row ascending for determinism.
        order = np.lexsort((rows, -rank_scores))
        ranked = rows[order]
        keep = ranked[:k]
        drop = ranked[k:]
        selected.extend(int(row) for row in keep.tolist())
        duplicate.extend(int(row) for row in drop.tolist())
        if drop.size:
            frames_with_duplicates += 1

    selected_arr = np.asarray(sorted(selected), dtype=np.int32)
    duplicate_arr = np.asarray(sorted(duplicate), dtype=np.int32)
    return (
        selected_arr,
        duplicate_arr,
        {
            "enabled": True,
            "per_frame_top_k": k,
            "score_field": str(score_field),
            "candidate_rows": int(candidates.shape[0]),
            "selected_rows": int(selected_arr.shape[0]),
            "duplicate_rows": int(duplicate_arr.shape[0]),
            "frames_with_duplicates": int(frames_with_duplicates),
        },
    )


def find_gaps_to_interpolate(
    frame_indices: np.ndarray,
    max_gap: int = 50
) -> List[Dict]:
    """
    Find gaps between detections suitable for interpolation.
    
    Args:
        frame_indices: Frame index for each detection
        max_gap: Maximum gap size to interpolate
        
    Returns:
        List of gap dictionaries with start, end, size
    """
    if frame_indices.size == 0:
        return []
    
    detected_frames = np.unique(frame_indices.astype('i4', copy=False))
    
    if detected_frames.size < 2:
        return []
    
    gaps = []
    for i in range(detected_frames.size - 1):
        gap_start = int(detected_frames[i])
        gap_end = int(detected_frames[i + 1])
        gap_size = gap_end - gap_start - 1
        
        if 0 < gap_size <= max_gap:
            gaps.append({
                'start_frame': gap_start,
                'end_frame': gap_end,
                'size': gap_size,
                'fill_frames': list(range(gap_start + 1, gap_end))
            })
    
    return gaps


def interpolate_gap(
    bbox_start: np.ndarray,
    bbox_end: np.ndarray,
    score_start: float,
    score_end: float,
    gap_frames: List[int],
    method: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate bounding boxes and scores across a gap.
    
    Args:
        bbox_start: Starting bbox [cx, cy, w, h]
        bbox_end: Ending bbox [cx, cy, w, h]
        score_start: Starting confidence score
        score_end: Ending confidence score
        gap_frames: Frame indices to fill
        method: Interpolation method ('linear')
        
    Returns:
        Tuple of (interpolated_bboxes, interpolated_scores)
    """
    n_interp = len(gap_frames)
    
    if method == 'linear':
        # Linear interpolation for each bbox component
        t = np.linspace(0, 1, n_interp + 2)[1:-1]  # Exclude endpoints
        
        interp_bboxes = np.zeros((n_interp, 4), dtype='f8')
        for i in range(4):
            interp_bboxes[:, i] = bbox_start[i] + t * (bbox_end[i] - bbox_start[i])
        
        # Score interpolation with decay toward gap center
        # Score is lowest at gap center, higher near endpoints
        gap_fraction = np.abs(t - 0.5) * 2  # 0 at center, 1 at edges
        min_score = min(score_start, score_end) * 0.5  # Minimum score
        interp_scores = min_score + gap_fraction * (min(score_start, score_end) - min_score)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    return interp_bboxes, interp_scores


def interpolate_detections(
    filtered_bboxes: np.ndarray,
    filtered_scores: np.ndarray,
    filtered_frame_indices: np.ndarray,
    filtered_class_ids: np.ndarray,
    num_frames: int,
    max_gap: int = 50,
    method: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Interpolate gaps in filtered detections.
    
    Args:
        filtered_bboxes: Filtered bounding boxes
        filtered_scores: Filtered scores
        filtered_frame_indices: Frame index for each filtered detection
        num_frames: Total frame count for the source video
        max_gap: Maximum gap size to interpolate
        method: Interpolation method
        
    Returns:
        Tuple of (interp_bboxes, interp_scores, frame_counts,
                  frame_indices, class_ids, detection_source, interp_stats)
    """
    # Find gaps
    gaps = find_gaps_to_interpolate(filtered_frame_indices, max_gap)
    
    if len(gaps) == 0:
        # No gaps to fill - return filtered data with source labels
        detection_source = np.zeros(len(filtered_bboxes), dtype='i1')
        stats = {
            'gaps_filled': 0,
            'interpolated_detections': 0,
            'mean_gap_size': 0.0,
            'max_gap_size': 0
        }
        filtered_counts = np.bincount(filtered_frame_indices, minlength=num_frames).astype('i4', copy=False)
        return (
            filtered_bboxes,
            filtered_scores,
            filtered_counts,
            filtered_frame_indices.astype('i4', copy=False),
            filtered_class_ids.astype('i4', copy=False),
            detection_source,
            stats,
        )
    
    # Build index mapping for filtered data
    frame_to_idx = {}
    for idx, frame in enumerate(filtered_frame_indices):
        frame = int(frame)
        if frame not in frame_to_idx:
            frame_to_idx[frame] = idx
    
    # Collect all interpolated detections
    all_bboxes = [filtered_bboxes]
    all_scores = [filtered_scores]
    all_frame_indices = [filtered_frame_indices.astype('i4', copy=False)]
    all_class_ids = [filtered_class_ids.astype('i4', copy=False)]
    all_sources = [np.zeros(len(filtered_bboxes), dtype='i1')]  # 0 = original
    
    total_interpolated = 0
    gap_sizes = []
    
    for gap in gaps:
        start_frame = gap['start_frame']
        end_frame = gap['end_frame']
        fill_frames = gap['fill_frames']
        
        # Get start and end detections
        start_idx = frame_to_idx[start_frame]
        end_idx = frame_to_idx[end_frame]
        
        bbox_start = filtered_bboxes[start_idx]
        bbox_end = filtered_bboxes[end_idx]
        score_start = filtered_scores[start_idx]
        score_end = filtered_scores[end_idx]
        
        # Interpolate
        interp_bboxes, interp_scores = interpolate_gap(
            bbox_start, bbox_end,
            score_start, score_end,
            fill_frames,
            method=method
        )

        start_class = int(filtered_class_ids[start_idx])
        end_class = int(filtered_class_ids[end_idx])
        if start_class == end_class:
            gap_class_ids = np.full(len(fill_frames), start_class, dtype='i4')
        else:
            gap_class_ids = np.full(len(fill_frames), start_class, dtype='i4')
        
        all_bboxes.append(interp_bboxes)
        all_scores.append(interp_scores)
        all_frame_indices.append(np.array(fill_frames, dtype='i4'))
        all_class_ids.append(gap_class_ids)
        all_sources.append(np.ones(len(fill_frames), dtype='i1'))  # 1 = interpolated
        
        total_interpolated += len(fill_frames)
        gap_sizes.append(gap['size'])
    
    # Concatenate all detections
    interp_bboxes = np.concatenate(all_bboxes, axis=0)
    interp_scores = np.concatenate(all_scores)
    interp_frame_indices = np.concatenate(all_frame_indices)
    interp_class_ids = np.concatenate(all_class_ids)
    detection_source = np.concatenate(all_sources)
    
    # Sort by frame to maintain temporal order
    sort_idx = np.argsort(interp_frame_indices)
    interp_bboxes = interp_bboxes[sort_idx]
    interp_scores = interp_scores[sort_idx]
    interp_frame_indices = interp_frame_indices[sort_idx]
    interp_class_ids = interp_class_ids[sort_idx]
    detection_source = detection_source[sort_idx]
    
    # Update n_detections
    interp_counts = np.bincount(interp_frame_indices, minlength=num_frames).astype('i4', copy=False)
    
    # Stats
    stats = {
        'gaps_filled': len(gaps),
        'interpolated_detections': int(total_interpolated),
        'mean_gap_size': float(np.mean(gap_sizes)) if gap_sizes else 0.0,
        'max_gap_size': int(max(gap_sizes)) if gap_sizes else 0,
        'min_gap_size': int(min(gap_sizes)) if gap_sizes else 0
    }
    
    return (
        interp_bboxes,
        interp_scores,
        interp_counts,
        interp_frame_indices.astype('i4', copy=False),
        interp_class_ids.astype('i4', copy=False),
        detection_source,
        stats,
    )


def _filtered_reason_from_quality_label(label: int) -> str:
    if label == _REGISTERED_DISH_GATE_QUALITY_LABEL:
        return "outside_registered_detection_gate"
    if label == _DISH_MASK_QUALITY_LABEL:
        return "outside_dish_mask"
    if label == 4:
        return "multi_detection"
    if label == 2:
        return "filtered_blip"
    if label == 3:
        return "filtered_jump"
    return f"filtered_quality_{int(label)}"


def _build_sparse_refined_inputs_from_filtered(
    *,
    raw_bboxes: np.ndarray,
    raw_scores: np.ndarray,
    raw_frame_indices: np.ndarray,
    raw_class_ids: np.ndarray,
    raw_instance_keys: Optional[np.ndarray] = None,
    detection_quality_labels: np.ndarray,
    interp_bboxes: np.ndarray,
    interp_scores: np.ndarray,
    interp_frame_indices: np.ndarray,
    interp_class_ids: np.ndarray,
    selected_source_detect_row_index: Optional[np.ndarray] = None,
    duplicate_source_detect_row_index: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    raw_bboxes_arr = np.asarray(raw_bboxes, dtype=np.float64).reshape(-1, 4)
    raw_scores_arr = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
    raw_frame_indices_arr = np.asarray(raw_frame_indices, dtype=np.int32).reshape(-1)
    raw_class_ids_arr = np.asarray(raw_class_ids, dtype=np.int32).reshape(-1)
    raw_instance_keys_arr = (
        np.asarray(raw_instance_keys, dtype=np.uint64).reshape(-1)
        if raw_instance_keys is not None
        else None
    )
    quality_labels_arr = np.asarray(detection_quality_labels, dtype=np.int8).reshape(-1)
    filtered_bboxes_arr = np.asarray(interp_bboxes, dtype=np.float64).reshape(-1, 4)
    filtered_scores_arr = np.asarray(interp_scores, dtype=np.float32).reshape(-1)
    filtered_frame_indices_arr = np.asarray(interp_frame_indices, dtype=np.int32).reshape(-1)
    filtered_class_ids_arr = np.asarray(interp_class_ids, dtype=np.int32).reshape(-1)

    if not (
        raw_bboxes_arr.shape[0]
        == raw_scores_arr.shape[0]
        == raw_frame_indices_arr.shape[0]
        == raw_class_ids_arr.shape[0]
        == quality_labels_arr.shape[0]
    ):
        raise ValueError("Raw detect arrays and quality labels must agree on row count.")
    if raw_instance_keys_arr is not None and raw_instance_keys_arr.shape[0] != raw_frame_indices_arr.shape[0]:
        raise ValueError("raw_instance_keys length does not match raw detect row count.")
    if not (
        filtered_bboxes_arr.shape[0]
        == filtered_scores_arr.shape[0]
        == filtered_frame_indices_arr.shape[0]
        == filtered_class_ids_arr.shape[0]
    ):
        raise ValueError("Filtered detect arrays must agree on row count.")

    kept_raw_indices = np.flatnonzero(quality_labels_arr == 0).astype(np.int32, copy=False)
    selected_raw_indices = (
        np.asarray(selected_source_detect_row_index, dtype=np.int32).reshape(-1)
        if selected_source_detect_row_index is not None
        else kept_raw_indices
    )
    duplicate_raw_indices = (
        np.asarray(duplicate_source_detect_row_index, dtype=np.int32).reshape(-1)
        if duplicate_source_detect_row_index is not None
        else np.empty((0,), dtype=np.int32)
    )
    if selected_raw_indices.shape[0] != filtered_frame_indices_arr.shape[0]:
        raise ValueError(
            "Filtered detection rows do not match selected source-detect rows."
        )
    if selected_raw_indices.size:
        if int(np.min(selected_raw_indices)) < 0 or int(np.max(selected_raw_indices)) >= raw_bboxes_arr.shape[0]:
            raise ValueError("selected_source_detect_row_index contains out-of-range rows.")
    if duplicate_raw_indices.size:
        if int(np.min(duplicate_raw_indices)) < 0 or int(np.max(duplicate_raw_indices)) >= raw_bboxes_arr.shape[0]:
            raise ValueError("duplicate_source_detect_row_index contains out-of-range rows.")

    source_decision_labels = np.full(raw_frame_indices_arr.shape[0], "filtered", dtype=object)
    source_reason_labels = np.asarray(
        [_filtered_reason_from_quality_label(int(label)) for label in quality_labels_arr.tolist()],
        dtype=object,
    )
    if duplicate_raw_indices.size:
        source_decision_labels[duplicate_raw_indices] = "duplicate"
        source_reason_labels[duplicate_raw_indices] = "per_frame_top_k_excluded"
    if selected_raw_indices.size:
        source_decision_labels[selected_raw_indices] = "accepted"
        source_reason_labels[selected_raw_indices] = "clean"

    payload = {
        "instance_frame_indices": filtered_frame_indices_arr,
        "instance_bbox_norm_coords": filtered_bboxes_arr,
        "instance_source_kind_labels": np.full(filtered_frame_indices_arr.shape[0], "raw_detect", dtype=object),
        "instance_reason_labels": np.full(filtered_frame_indices_arr.shape[0], "clean", dtype=object),
        "instance_source_detect_row_index": selected_raw_indices,
        "instance_manual_edit_flags": np.zeros(filtered_frame_indices_arr.shape[0], dtype=bool),
        "instance_confidence_scores": filtered_scores_arr,
        "instance_class_ids": filtered_class_ids_arr,
        "source_detection_source_detect_row_index": np.arange(raw_frame_indices_arr.shape[0], dtype=np.int32),
        "source_detection_frame_indices": raw_frame_indices_arr,
        "source_detection_bbox_norm_coords": raw_bboxes_arr,
        "source_detection_decision_labels": source_decision_labels,
        "source_detection_reason_labels": source_reason_labels,
        "source_detection_confidence_scores": raw_scores_arr,
        "source_detection_class_ids": raw_class_ids_arr,
    }
    if raw_instance_keys_arr is not None:
        payload["instance_key"] = raw_instance_keys_arr[selected_raw_indices]
        payload["source_detection_instance_key"] = raw_instance_keys_arr
    return payload


def get_refinement_parameters(
    config: Dict[str, Any],
    cli_overrides: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], str]:
    """
    Get refinement parameters from config (no tuning step needed).
    
    Args:
        config: Loaded config dictionary
        cli_overrides: Optional CLI parameter overrides
        
    Returns:
        Tuple of (parameters dict, source string)
    """
    # Start with config defaults
    refine_params = config.get('refine_detect', {}).copy()
    refine_params.pop('max_gap', None)
    refine_params.pop('interpolation_method', None)
    refine_params.setdefault('filters', {'remove_jumps': True, 'remove_blips': False})
    refine_params.setdefault('max_gap', 0)
    refine_params.setdefault('interpolation_method', 'disabled')
    
    # Apply CLI overrides if provided
    if cli_overrides:
        refine_params.update(cli_overrides)
        return refine_params, 'cli_override'
    
    return refine_params, 'config'


def create_refined_run(
    zarr_path: str,
    detect_run: Optional[str] = None,
    quality_run: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    max_gap: Optional[int] = None,
    interpolation_method: Optional[str] = None,
    remove_jumps: Optional[bool] = None,
    remove_blips: Optional[bool] = None,
    console: Optional[Console] = None,
    *,
    quality_group_path: Optional[str] = None,
    command: Optional[str] = None,
    created_at_utc: Optional[str] = None,
    save_visuals: bool = False,
    show_visuals: bool = False,
    visuals_dpi: int = 150,
    require_detect_quality: bool = True,
    per_frame_top_k: Optional[int] = None,
    top_k_score_field: str = "scores",
    detect_family_path: str = DEFAULT_DETECT_FAMILY_PATH,
    refined_family_path: str = REFINED_DETECT_GROUP,
    refined_run_name: Optional[str] = None,
    dish_mask_boundary_tolerance_mm: Optional[float] = None,
    pixels_per_mm_camera: Optional[float] = None,
    registered_gate_requirement: str = "off",
    registered_gate_run: Optional[str] = None,
    stage_selector_eligible: bool = True,
    emit_completion_status: bool = True,
    require_active_canonical_source: bool = False,
    expected_source_manifest_digest: str | None = None,
    source_publication_receipt: str | Path | None = None,
) -> str:
    """
    Create a refined detection run with sparse-first curated detect surfaces.
    
    Args:
        zarr_path: Path to zarr file
        detect_run: Source detect run (default: latest)
        quality_run: Historical nested source quality run (default: latest)
        quality_group_path: Modern root quality group path. Collection quality
            slices are resolved and validated by exact ``instance_key``.
        config: Config dictionary (optional, will load if not provided)
        max_gap: Deprecated compatibility argument; ignored because interpolation is disabled
        interpolation_method: Deprecated compatibility argument; ignored because interpolation is disabled
        remove_jumps: Remove jump artifacts (overrides config)
        remove_blips: Remove blip artifacts (overrides config)
        console: Rich console for output
        require_detect_quality: Require usable detect_quality labels for refinement
        per_frame_top_k: Optional max accepted detections per frame after quality filtering
        top_k_score_field: Score field used for top-k ranking; currently only "scores"
        detect_family_path: Group path containing source detect runs
        refined_family_path: Group path where refined detect runs are written
        refined_run_name: Optional explicit refined run name. Used by cluster
            planners when downstream jobs need deterministic paths.
        dish_mask_boundary_tolerance_mm: Physical expansion around the fitted
            dish boundary. Defaults to 0.5 mm when a dish mask is present.
        pixels_per_mm_camera: Explicit camera-space calibration override. The
            override is recorded in provenance and does not mutate Zarr calibration.
        registered_gate_requirement: Explicit modern geometry mode: off,
            if_available, or required.
        registered_gate_run: Exact immutable detection-gate run to join. Mutable
            latest pointers are never resolved by refinement.
        stage_selector_eligible: Whether completion may update the refined-run
            selectors. Set false for explicitly selected canaries and review seeds.
        emit_completion_status: Whether to project the completed stage into the
            registry/status system. Set false for unregistered benchmark canaries.
        require_active_canonical_source: Require the exact source run to remain
            the selected canonical-v3 detection authority.
        expected_source_manifest_digest: Exact planner-bound manifest digest for
            an already-published canonical source.
        source_publication_receipt: Immutable native-publication receipt used to
            resolve the expected digest when inference is an upstream job.
        
    Returns:
        Name of created refined run
    """
    if console is None:
        console = Console()

    console.rule("[bold]Detection Refinement[/bold]")
    
    import time
    start_time = time.perf_counter()
    
    # Load config if not provided
    if config is None:
        import yaml
        config_path = Path("pipeline_config.yaml")
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            config = {}

    _reject_deprecated_interpolation_overrides(
        max_gap=max_gap,
        interpolation_method=interpolation_method,
    )

    refine_config = config.get("refine_detect", {}) if isinstance(config, dict) else {}
    resolved_dish_tolerance_mm = (
        dish_mask_boundary_tolerance_mm
        if dish_mask_boundary_tolerance_mm is not None
        else refine_config.get(
            "dish_mask_boundary_tolerance_mm",
            DEFAULT_DISH_MASK_BOUNDARY_TOLERANCE_MM,
        )
    )
    resolved_pixels_per_mm_camera = (
        pixels_per_mm_camera
        if pixels_per_mm_camera is not None
        else refine_config.get("pixels_per_mm_camera")
    )
    configured_max_gap = refine_config.get("max_gap")
    configured_method = refine_config.get("interpolation_method")
    if configured_max_gap not in (None, 0):
        console.print(
            "[yellow]Warning:[/yellow] refine_detect.max_gap is deprecated and ignored; "
            "the sparse-first refine_detect workflow no longer interpolates gaps."
        )
    if configured_method not in (None, "disabled"):
        console.print(
            "[yellow]Warning:[/yellow] refine_detect.interpolation_method is deprecated and ignored; "
            "the sparse-first refine_detect workflow no longer interpolates gaps."
        )
    
    # Get parameters
    params, param_source = get_refinement_parameters(config, None)
    if dish_mask_boundary_tolerance_mm is not None or pixels_per_mm_camera is not None:
        param_source = "cli_override"
    
    # Handle filter overrides
    filters_config = params.get('filters', {'remove_jumps': True, 'remove_blips': False})
    if remove_jumps is not None:
        filters_config['remove_jumps'] = remove_jumps
        param_source = 'cli_override'
    if remove_blips is not None:
        filters_config['remove_blips'] = remove_blips
        param_source = 'cli_override'
    if type(require_active_canonical_source) is not bool:
        raise TypeError("require_active_canonical_source must be an exact bool.")
    
    # Build filter list
    filters = []
    if filters_config.get('remove_jumps', True):
        filters.append('remove_jumps')
    if filters_config.get('remove_blips', False):
        filters.append('remove_blips')
    
    refine_mode = "standard"
    
    # Open mutable stores directly so a newly written run is not hidden by stale consolidated metadata.
    root = open_zarr_group_direct(zarr_path, mode='a')
    detect_family_path = _normalize_group_path(detect_family_path)
    refined_family_path = _normalize_group_path(refined_family_path)
    
    # Get detect run
    if detect_run is None:
        detect_run = root[detect_family_path].attrs['latest']
    
    source_detect_path = _join_group_path(detect_family_path, detect_run)
    active_source_manifest = None
    bound_manifest_digest: str | None = None
    if require_active_canonical_source:
        bound_manifest_digest = resolve_expected_canonical_detection_manifest_digest(
            expected_group_path=source_detect_path,
            expected_manifest_digest=expected_source_manifest_digest,
            publication_receipt_path=source_publication_receipt,
        )
        active_source_manifest = require_active_coordinate_canonical_detection(
            root,
            group_path=source_detect_path,
            expected_manifest_digest=bound_manifest_digest,
        )
    elif expected_source_manifest_digest is not None or source_publication_receipt is not None:
        raise ValueError(
            "Canonical source expectations require require_active_canonical_source."
        )
    detect_group = root[source_detect_path]
    detect_table = resolve_detection_instance_table(detect_group)
    console.print(f"Source detect run: [cyan]{detect_run}[/cyan]")
    if detect_family_path != DEFAULT_DETECT_FAMILY_PATH:
        console.print(f"Source detect family: [cyan]{detect_family_path}[/cyan]")
    
    # Load detection data
    console.print("\nLoading detection data...")
    bbox_coords = detect_table['bbox_norm_coords'][:]
    frame_indices = detect_table['frame_indices'][:]
    raw_instance_keys = (
        np.asarray(detect_table["instance_key"][:], dtype=np.uint64).reshape(-1)
        if "instance_key" in detect_table
        else None
    )
    if raw_instance_keys is not None and raw_instance_keys.shape[0] != len(bbox_coords):
        raise ValueError("detect instance_key length does not match detection row count.")

    sampled_import, sampled_meta = _read_sampled_import_meta(root)
    require_quality_for_run = bool(require_detect_quality and not sampled_import)
    if not require_detect_quality:
        allow_missing_reason = "explicit opt-out (--allow-missing-quality)"
    elif sampled_import:
        allow_missing_reason = "sampled import passthrough mode"
    else:
        allow_missing_reason = "guardrail disabled"
    (detection_quality_labels, resolved_quality_run, quality_group) = _resolve_detection_quality_labels(
        root,
        detect_group,
        detect_run=detect_run,
        source_detect_path=source_detect_path,
        quality_run=quality_run,
        quality_group_path=quality_group_path,
        total_detections=len(bbox_coords),
        require_quality=require_quality_for_run,
        allow_missing_reason=allow_missing_reason,
        console=console,
    )
    resolved_quality_group_path = (
        normalize_attr(getattr(quality_group, "path", None))
        if quality_group is not None
        else None
    )
    if require_active_canonical_source:
        _require_quality_matches_active_canonical_manifest(
            quality_group,
            active_source_manifest,
        )
    experiment_setup = None
    try:
        experiment_setup = resolve_experiment_setup(root, allow_legacy=True)
    except MissingExperimentSetupError:
        if require_quality_for_run:
            raise ValueError(
                "Production detection refinement requires experiment setup metadata"
            )
    if experiment_setup is not None and quality_group is not None:
        _validate_quality_experiment_setup_binding(
            experiment_setup,
            quality_group.attrs,
        )
    if sampled_import:
        console.print("[yellow]⚠ Sampled training import detected; disabling refine filters.[/yellow]")
        detection_quality_labels = np.zeros_like(detection_quality_labels)
        filters = []
        refine_mode = "passthrough"
        param_source = "sampled_import_guard"

    detection_quality_labels, registered_detection_gate = (
        _apply_registered_detection_gate(
            zarr_path=zarr_path,
            source_detect_path=source_detect_path,
            raw_instance_keys=raw_instance_keys,
            detection_quality_labels=detection_quality_labels,
            requirement=registered_gate_requirement,
            gate_run=registered_gate_run,
        )
    )
    dish_mask_spec = None
    if registered_gate_requirement == "off":
        dish_mask_spec = _read_dish_mask_spec(root, detect_group)
        if dish_mask_spec is not None:
            dish_mask_tolerance = resolve_dish_mask_boundary_tolerance(
                root,
                source_group=detect_group,
                tolerance_mm=resolved_dish_tolerance_mm,
                pixels_per_mm_camera=resolved_pixels_per_mm_camera,
            )
            dish_mask_spec = apply_dish_mask_boundary_tolerance(
                dish_mask_spec,
                dish_mask_tolerance,
            )
    detection_quality_labels, dish_mask_gate = _apply_dish_mask_quality_gate(
        bbox_coords=bbox_coords,
        detection_quality_labels=detection_quality_labels,
        mask_spec=dish_mask_spec,
    )

    console.print(f"Parameters source: [cyan]{param_source}[/cyan]")
    console.print("  Interpolation: disabled")
    console.print(f"  Filters: {filters}")
    if sampled_import:
        console.print("  Refine mode: passthrough (sampled import)")
    console.print(
        "  Registered detection gate: "
        f"{registered_detection_gate.get('status')} "
        f"(requirement={registered_detection_gate.get('requirement')})"
    )
    if dish_mask_gate.get("enabled"):
        console.print(
            "  Dish mask gate: "
            f"{dish_mask_gate.get('outside_clean_rows')} clean detections outside "
            f"{dish_mask_gate.get('shape')} mask"
        )

    # Get total frames using unified metadata helper
    num_frames = get_total_frames(root, detect_group)
    full_frame_count = num_frames
    coverage_frame_source = "full"
    if sampled_import:
        sampled_frames = _get_sampled_frame_count(root, detect_group)
        if sampled_frames is not None:
            if full_frame_count is not None and sampled_frames != full_frame_count:
                console.print(
                    f"[yellow]⚠ Using sampled frame count ({sampled_frames}) "
                    f"for coverage stats (full={full_frame_count}).[/yellow]"
                )
            num_frames = sampled_frames
            coverage_frame_source = "sampled"

    if num_frames is None:
        # Infer from detections as last resort
        num_frames = int(frame_indices.max() + 1)
        console.print(f"[yellow]⚠ No 'total_frames' in metadata, inferred {num_frames} from detections[/yellow]")
        
        # Log detection method for context
        detect_method = get_detection_method(detect_group)
        console.print(f"[yellow]  Detection method: {detect_method}[/yellow]")
    else:
        console.print(f"Using {num_frames} frames from metadata")
    
    # Get frame counts
    frame_counts = read_detection_frame_counts(detect_table, n_frames=num_frames)
    
    # Scores may not exist for blob detection - create placeholder if missing
    if 'scores' in detect_table:
        scores = detect_table['scores'][:]
    else:
        # Create placeholder scores (all 1.0 for blob detections)
        scores = np.ones(len(bbox_coords), dtype='f4')
        console.print("  [yellow]Note: No scores array found, using placeholder values[/yellow]")

    if 'class_ids' in detect_table:
        class_ids = detect_table['class_ids'][:].astype('i4', copy=False)
    else:
        class_ids = np.zeros(len(bbox_coords), dtype='i4')
        console.print("  [yellow]Note: No class_ids array found, defaulting to 0[/yellow]")
    console.print(f"  Total detections: {len(bbox_coords)}")
    console.print(f"  Total frames: {num_frames}")
    
    # Step 1: Filter
    console.print("\n[bold]Step 1: Filtering[/bold]")
    console.print(f"  Filters: {filters}")
    
    (filtered_bboxes, filtered_scores, filtered_counts,
     filtered_frame_indices, filtered_class_ids, drop_stats) = filter_detections(
        bbox_coords,
        scores,
        frame_indices,
        class_ids,
        detection_quality_labels,
        num_frames,
        filters,
    )

    selected_source_detect_row_index = np.flatnonzero(detection_quality_labels == 0).astype(np.int32, copy=False)
    duplicate_source_detect_row_index = np.empty((0,), dtype=np.int32)
    top_k_selection = {
        "enabled": False,
        "per_frame_top_k": None,
        "score_field": str(top_k_score_field),
        "candidate_rows": int(selected_source_detect_row_index.shape[0]),
        "selected_rows": int(selected_source_detect_row_index.shape[0]),
        "duplicate_rows": 0,
        "frames_with_duplicates": 0,
    }
    if per_frame_top_k is not None:
        (
            selected_source_detect_row_index,
            duplicate_source_detect_row_index,
            top_k_selection,
        ) = _select_per_frame_top_k_raw_indices(
            raw_frame_indices=frame_indices,
            raw_scores=scores,
            candidate_raw_indices=np.flatnonzero(detection_quality_labels == 0).astype(np.int32, copy=False),
            per_frame_top_k=per_frame_top_k,
            score_field=top_k_score_field,
        )
        filtered_bboxes = bbox_coords[selected_source_detect_row_index]
        filtered_scores = scores[selected_source_detect_row_index]
        filtered_frame_indices = frame_indices[selected_source_detect_row_index].astype("i4", copy=False)
        filtered_class_ids = class_ids[selected_source_detect_row_index].astype("i4", copy=False)
        filtered_counts = np.bincount(filtered_frame_indices, minlength=num_frames).astype("i4", copy=False)
        nonclean_dropped = int(np.sum(detection_quality_labels != 0))
        duplicate_dropped = int(duplicate_source_detect_row_index.shape[0])
        drop_stats["kept"] = int(selected_source_detect_row_index.shape[0])
        drop_stats["total_dropped"] = nonclean_dropped + duplicate_dropped
        drop_stats.setdefault("reasons", {})["per_frame_top_k_duplicate"] = duplicate_dropped
        console.print(
            "  Top-k selection: "
            f"kept {top_k_selection['selected_rows']}/{top_k_selection['candidate_rows']} "
            f"clean candidates with top {top_k_selection['per_frame_top_k']} by {top_k_selection['score_field']}"
        )
    
    console.print(f"  Kept: {drop_stats['kept']} detections")
    console.print(f"  Dropped: {drop_stats['total_dropped']} detections")
    for reason, count in drop_stats['reasons'].items():
        console.print(f"    - {reason}: {count}")
    
    # Step 2: Build sparse curated surfaces
    console.print("\n[bold]Step 2: Sparse Curated Surfaces[/bold]")
    console.print("  Interpolation: disabled")

    sparse_refined = _build_sparse_refined_inputs_from_filtered(
        raw_bboxes=bbox_coords,
        raw_scores=scores,
        raw_frame_indices=frame_indices,
        raw_class_ids=class_ids,
        raw_instance_keys=raw_instance_keys,
        detection_quality_labels=detection_quality_labels,
        interp_bboxes=filtered_bboxes,
        interp_scores=filtered_scores,
        interp_frame_indices=filtered_frame_indices,
        interp_class_ids=filtered_class_ids,
        selected_source_detect_row_index=selected_source_detect_row_index,
        duplicate_source_detect_row_index=duplicate_source_detect_row_index,
    )

    # Calculate coverage comparison
    original_coverage = (np.sum(frame_counts > 0) / num_frames) * 100
    refined_coverage = (np.sum(filtered_counts > 0) / num_frames) * 100
    
    comparison_stats = {
        'original': {
            'total_detections': int(len(bbox_coords)),
            'frames_with_detections': int(np.sum(frame_counts > 0)),
            'coverage_percent': float(original_coverage)
        },
        'refined': {
            'total_detections': int(len(filtered_bboxes)),
            'frames_with_detections': int(np.sum(filtered_counts > 0)),
            'coverage_percent': float(refined_coverage),
            'detections_removed': int(drop_stats['total_dropped']),
            'coverage_delta': float(refined_coverage - original_coverage),
        }
    }
    
    # Step 3: Save
    console.print("\n[bold]Step 3: Saving Refined Run[/bold]")
    
    # Calculate processing time
    duration = time.perf_counter() - start_time
    
    # Create refined detect group
    refined_runs = require_runs_parent(root, refined_family_path)
    
    # Create timestamped or explicitly named run.
    if refined_run_name is not None:
        run_name = str(refined_run_name).strip()
        if not run_name or "/" in run_name or run_name in {".", ".."}:
            raise ValueError(f"Invalid refined run name: {refined_run_name!r}")
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_name = f"refined_detect_{timestamp}"
    if run_name in refined_runs:
        raise ValueError(f"{refined_runs.path}/{run_name} already exists")
    refined_group = refined_runs.create_group(run_name)
    mark_run_started(refined_group, run_name=run_name, stage="refined_detect")
    refined_group.attrs["stage_selector_eligible"] = bool(stage_selector_eligible)
    if stage_selector_eligible:
        note_pending_latest(refined_runs, run_name)
    
    # Store root metadata
    created_timestamp = created_at_utc or datetime.now(timezone.utc).isoformat()

    parameters_payload = {
        'filters_applied': filters,
        'parameter_source': param_source,
        'refine_mode': refine_mode,
        'interpolation_enabled': False,
        'interpolation_method': 'disabled',
        'max_gap': 0,
        'sampled_import': sampled_import,
        'sampled_import_meta': sampled_meta,
        'detect_quality_guardrail_requested': bool(require_detect_quality),
        'detect_quality_guardrail_enforced': bool(require_quality_for_run),
        'require_active_canonical_source': require_active_canonical_source,
        'expected_source_manifest_digest': bound_manifest_digest,
        'source_publication_receipt': (
            str(Path(source_publication_receipt).expanduser().resolve())
            if source_publication_receipt is not None
            else None
        ),
        'registered_detection_gate': registered_detection_gate,
        'dish_mask_gate': dish_mask_gate,
        'top_k_selection': top_k_selection,
        'experiment_setup': (
            {
                'path': experiment_setup.group_path,
                'sha256': experiment_setup.record_sha256,
                'expected_subject_count': experiment_setup.expected_subject_count,
                'legacy': experiment_setup.legacy,
            }
            if experiment_setup is not None
            else None
        ),
    }

    refined_group.attrs['source_detect_run'] = detect_run
    refined_group.attrs['source_detect_family_path'] = detect_family_path
    refined_group.attrs['source_detect_path'] = source_detect_path
    refined_group.attrs['source_quality_run'] = resolved_quality_run or 'N/A'
    refined_group.attrs['source_quality_group_path'] = (
        resolved_quality_group_path or "N/A"
    )
    refined_group.attrs['registered_detection_gate'] = registered_detection_gate
    refined_group.attrs['registered_detection_gate_requirement'] = (
        registered_gate_requirement
    )
    refined_group.attrs['registered_detection_gate_consumed'] = bool(
        registered_detection_gate.get("applied")
    )
    if experiment_setup is not None:
        refined_group.attrs['experiment_setup_path'] = experiment_setup.group_path
        refined_group.attrs['experiment_setup_sha256'] = experiment_setup.record_sha256
        refined_group.attrs['expected_subject_count'] = experiment_setup.expected_subject_count
    refined_group.attrs['refined_family_path'] = refined_family_path
    refined_group.attrs['refinement_timestamp'] = created_timestamp
    refined_group.attrs['processing_time_seconds'] = float(duration)
    refined_group.attrs['operations'] = ['filter'] if refine_mode == "standard" else ['passthrough']
    refined_group.attrs['parameters'] = parameters_payload
    refined_group.attrs['coverage_comparison'] = comparison_stats
    refined_group.attrs['coverage_frames_total'] = int(num_frames)
    refined_group.attrs['coverage_frame_source'] = coverage_frame_source
    if full_frame_count is not None:
        refined_group.attrs['coverage_frames_full'] = int(full_frame_count)
    refined_group.attrs['inputs'] = {
        'detect_run': detect_run,
        'detect_family_path': detect_family_path,
        'detect_path': source_detect_path,
        'quality_run': resolved_quality_run or 'N/A',
        'quality_group_path': resolved_quality_group_path or 'N/A',
        'registered_detection_gate': registered_detection_gate,
        'experiment_setup_path': (
            experiment_setup.group_path if experiment_setup is not None else 'N/A'
        ),
        'experiment_setup_sha256': (
            experiment_setup.record_sha256 if experiment_setup is not None else 'N/A'
        ),
        'refined_family_path': refined_family_path,
    }

    git_info = get_git_info()
    env_info = get_environment_info()
    environment_info = {
        "hostname": env_info["platform"].get("hostname", "unknown"),
        "python_version": env_info["platform"].get("python_version", "unknown"),
        "system": env_info["platform"].get("system", "unknown"),
        "release": env_info["platform"].get("release", "unknown"),
    }

    scheduler_info = None

    artifact_keys = [
        'model_path',
        'model_name',
        'model_version',
        'detection_method',
        'pipeline_type',
        'training_run',
        'checkpoint_path',
        'quality_source',
    ]
    artifact_info = {key: detect_group.attrs[key] for key in artifact_keys if key in detect_group.attrs}
    if quality_group is not None and 'artifact_detection_params' in quality_group.attrs:
        artifact_info['quality_detection_params'] = quality_group.attrs['artifact_detection_params']

    provenance_record = build_stage_provenance(
        stage='refine_detect',
        command=command or ' '.join(sys.argv),
        created_at_utc=created_timestamp,
        version=git_info.get('short_hash') or git_info.get('commit_hash'),
        git={
            'commit': git_info.get('commit_hash'),
            'short': git_info.get('short_hash'),
            'branch': git_info.get('branch'),
            'is_dirty': git_info.get('is_dirty'),
            'remote': git_info.get('remote_url'),
        },
        environment=environment_info,
        scheduler=scheduler_info,
        parameters=parameters_payload,
        inputs={
            'detect_run': detect_run,
            'detect_family_path': detect_family_path,
            'detect_path': source_detect_path,
            'quality_run': resolved_quality_run or 'N/A',
            'frame_source': 'zarr' if root.attrs.get('has_raw_video', True) else 'external',
            'source_video_path': root.attrs.get('source_video_path'),
            'refined_family_path': refined_family_path,
        },
        artifacts=artifact_info,
    )
    write_stage_provenance(refined_group, provenance_record)
    
    write_curated_refined_detect_surfaces(
        root,
        zarr_path=Path(zarr_path).expanduser().resolve(),
        refined_run_name=run_name,
        instance_frame_indices=np.asarray(sparse_refined["instance_frame_indices"], dtype=np.int32),
        instance_bbox_norm_coords=np.asarray(sparse_refined["instance_bbox_norm_coords"], dtype=np.float64),
        instance_source_kind_labels=np.asarray(sparse_refined["instance_source_kind_labels"], dtype=object),
        instance_reason_labels=np.asarray(sparse_refined["instance_reason_labels"], dtype=object),
        instance_source_detect_row_index=np.asarray(sparse_refined["instance_source_detect_row_index"], dtype=np.int32),
        instance_manual_edit_flags=np.asarray(sparse_refined["instance_manual_edit_flags"], dtype=bool),
        instance_confidence_scores=np.asarray(sparse_refined["instance_confidence_scores"], dtype=np.float32),
        instance_class_ids=np.asarray(sparse_refined["instance_class_ids"], dtype=np.int32),
        instance_key=(
            np.asarray(sparse_refined["instance_key"], dtype=np.uint64)
            if "instance_key" in sparse_refined
            else None
        ),
        source_detection_source_detect_row_index=np.asarray(
            sparse_refined["source_detection_source_detect_row_index"], dtype=np.int32
        ),
        source_detection_frame_indices=np.asarray(sparse_refined["source_detection_frame_indices"], dtype=np.int32),
        source_detection_bbox_norm_coords=np.asarray(sparse_refined["source_detection_bbox_norm_coords"], dtype=np.float64),
        source_detection_decision_labels=np.asarray(sparse_refined["source_detection_decision_labels"], dtype=object),
        source_detection_reason_labels=np.asarray(sparse_refined["source_detection_reason_labels"], dtype=object),
        source_detection_confidence_scores=np.asarray(
            sparse_refined["source_detection_confidence_scores"], dtype=np.float32
        ),
        source_detection_class_ids=np.asarray(sparse_refined["source_detection_class_ids"], dtype=np.int32),
        source_detection_instance_key=(
            np.asarray(sparse_refined["source_detection_instance_key"], dtype=np.uint64)
            if "source_detection_instance_key" in sparse_refined
            else None
        ),
        command=command or ' '.join(sys.argv),
        env_info=env_info,
        source_context={
            "source_detect_run": detect_run,
            "source_detect_family_path": detect_family_path,
            "source_detect_path": source_detect_path,
            "quality_run": resolved_quality_run or "N/A",
            "experiment_setup_path": (
                experiment_setup.group_path if experiment_setup is not None else "N/A"
            ),
            "experiment_setup_sha256": (
                experiment_setup.record_sha256 if experiment_setup is not None else "N/A"
            ),
            "selection_policy": (
                "quality_filtered_per_frame_top_k_sparse_instances_no_interpolation"
                if top_k_selection.get("enabled")
                else "quality_filtered_sparse_instances_no_interpolation"
            ),
            "registered_detection_gate": registered_detection_gate,
            "dish_mask_gate": dish_mask_gate,
            "top_k_selection": top_k_selection,
        },
        refined_family_path=refined_family_path,
    )

    console.print(f"[green]✓[/green] Refined run saved: {refined_group.path}")
    console.print(f"[green]✓[/green] Processing completed in {duration:.2f} seconds")
    
    console.print("\n[bold green]Coverage Comparison:[/bold green]")
    console.print(f"  Original:     {comparison_stats['original']['frames_with_detections']:5d} frames ({comparison_stats['original']['coverage_percent']:.2f}%)")
    console.print(f"  Refined:      {comparison_stats['refined']['frames_with_detections']:5d} frames ({comparison_stats['refined']['coverage_percent']:.2f}%) "
                 f"[red]{comparison_stats['refined']['coverage_delta']:+.2f}%[/red]")
    
    console.print("\n[bold green]Detection Summary:[/bold green]")
    console.print(f"  Refined present detections: {len(filtered_bboxes)}")
    console.print(f"  Filtered out detections: {drop_stats['total_dropped']}")

    if save_visuals or show_visuals:
        if quality_group is None:
            console.print("[yellow]Visualizations requested, but no quality report is available; skipping.[/yellow]")
        elif detect_family_path != DEFAULT_DETECT_FAMILY_PATH:
            console.print(
                "[yellow]Visualizations requested, but clip-local detect quality "
                "rendering is not path-aware yet; skipping.[/yellow]"
            )
        else:
            try:
                from ..visualization.visualize_detect_quality import render_quality_png

                png_bytes, quality_meta = render_quality_png(
                    zarr_path,
                    detect_run=detect_run,
                    quality_run=resolved_quality_run,
                    dpi=visuals_dpi,
                    show=show_visuals,
                )
                if save_visuals:
                    vis_group = refined_group.require_group('visualizations')
                    array_name = 'detect_quality_overview_png'
                    if array_name in vis_group:
                        del vis_group[array_name]
                    data = np.frombuffer(png_bytes, dtype=np.uint8)
                    chunk = max(1, min(len(data), 1_048_576))
                    ds = vis_group.create_array(
                        array_name,
                        data=data,
                        chunks=(chunk,),
                        overwrite=True,
                    )
                    ds.attrs.update({
                        'mime': 'image/png',
                        'description': 'Detection quality overview visualization',
                        'source_detect_run': detect_run,
                        'source_quality_run': resolved_quality_run,
                        'quality_grade': quality_meta['quality_score'].get('grade'),
                    })
                    manifest = dict(refined_group.attrs.get('visualizations', {}))
                    manifest['detect_quality_overview_png'] = {
                        'path': 'visualizations/detect_quality_overview_png',
                        'description': 'Detection quality overview PNG',
                    }
                    refined_group.attrs['visualizations'] = manifest
                    console.print("[green]✓[/green] Detection quality visualization stored in refined run.")
            except Exception as exc:
                console.print(f"[yellow]Warning:[/yellow] Failed to render detection visualization: {exc}")

    mark_run_complete(
        refined_group,
        parent_group=refined_runs,
        run_name=run_name,
        run_provenance=build_run_provenance_from_stage_record(provenance_record),
    )

    detect_review_status = _parse_mapping(refined_group.attrs.get("detect_review_status"))
    if emit_completion_status:
        _emit_refined_detect_status(
            root=root,
            zarr_path=Path(zarr_path).expanduser().resolve(),
            status="ok",
            run_name=run_name,
            method=(
                (
                    normalize_attr(refined_group.attrs.get("method"))
                    if hasattr(refined_group, "attrs")
                    else None
                )
                or refine_mode
            ),
            coverage_pct=comparison_stats.get("refined", {}).get(
                "coverage_percent"
            ),
            review_status=detect_review_status,
            details={
                "reason": "present",
                "source_detect_run": detect_run,
                "source_detect_family_path": detect_family_path,
                "source_detect_path": source_detect_path,
                "source_quality_run": resolved_quality_run,
                "refined_family_path": refined_family_path,
                "refine_mode": refine_mode,
                "parameter_source": param_source,
                "filters_applied": filters,
                "coverage_frame_source": coverage_frame_source,
                "coverage_frames_total": num_frames,
                "coverage_frames_full": full_frame_count,
                "refined_present_detections": int(len(filtered_bboxes)),
                "filtered_out_detections": int(drop_stats["total_dropped"]),
                "interpolation_enabled": False,
            },
            console=console,
        )
    
    return run_name


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Refine detection data into a sparse curated detect surface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic refinement (uses config defaults)
  scripts/py -m fisheye.refinement.refine_detect data.zarr

  # Remove both jumps and blips
  scripts/py -m fisheye.refinement.refine_detect data.zarr --remove-jumps --remove-blips

  # Keep jumps (don't filter them)
  scripts/py -m fisheye.refinement.refine_detect data.zarr --no-remove-jumps

  # Specify source runs
  scripts/py -m fisheye.refinement.refine_detect data.zarr --detect-run detect_2025-10-03_20-28-11

  # Refine a clip-local detect run
  scripts/py -m fisheye.refinement.refine_detect data.zarr \\
    --detect-family-path clips/clip_000000/cameras/2010093/detect_runs \\
    --refined-family-path clips/clip_000000/cameras/2010093/refined_detect_runs

  # Keep only the top confidence candidate per frame while preserving all raw candidates
  # in source_detections.
  scripts/py -m fisheye.refinement.refine_detect data.zarr --per-frame-top-k 1
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    parser.add_argument('--detect-run', help='Source detect run (default: latest)')
    parser.add_argument(
        '--detect-family-path',
        default=DEFAULT_DETECT_FAMILY_PATH,
        help=(
            'Group path containing detect runs (default: detect_runs). '
            'Use clips/<clip>/cameras/<serial>/detect_runs for clip-local runs.'
        ),
    )
    parser.add_argument(
        '--refined-family-path',
        default=REFINED_DETECT_GROUP,
        help=(
            'Group path where refined detect runs are written (default: refined_detect_runs). '
            'Use clips/<clip>/cameras/<serial>/refined_detect_runs for clip-local runs.'
        ),
    )
    parser.add_argument('--quality-run', help='Source quality run (default: latest)')
    parser.add_argument(
        '--quality-group-path',
        default=None,
        help=(
            'Modern root quality group path, for example '
            'detect_quality_runs/<run>. Collection rows are selected through '
            'the source manifest and verified by instance_key.'
        ),
    )
    parser.add_argument('--max-gap', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--method', default=None, choices=['linear'], help=argparse.SUPPRESS)
    parser.add_argument('--remove-jumps', action='store_true', default=None,
                       help='Remove jump artifacts (overrides config)')
    parser.add_argument('--no-remove-jumps', action='store_false', dest='remove_jumps',
                       help='Keep jump artifacts (overrides config)')
    parser.add_argument('--remove-blips', action='store_true', default=None,
                       help='Remove blip artifacts (overrides config)')
    parser.add_argument('--no-remove-blips', action='store_false', dest='remove_blips',
                       help='Keep blip artifacts (overrides config)')
    parser.add_argument('--config', default='pipeline_config.yaml',
                       help='Path to config file (default: pipeline_config.yaml)')
    parser.add_argument('--save-visuals', action='store_true',
                       help='Store detection quality visualization inside the refined run.')
    parser.add_argument('--show-visuals', action='store_true',
                       help='Display detection quality visualization interactively after refinement.')
    parser.add_argument('--visuals-dpi', type=int, default=150,
                       help='DPI to use when rendering saved visualizations (default: 150).')
    parser.add_argument(
        '--allow-missing-quality',
        action='store_true',
        help='Allow refinement without detect_quality labels (legacy fallback).',
    )
    parser.add_argument(
        '--per-frame-top-k',
        type=int,
        default=None,
        help='Accept only the top K clean detections per frame into instances; non-top clean rows remain in source_detections as duplicate.',
    )
    parser.add_argument(
        '--top-k-score-field',
        default='scores',
        choices=['scores'],
        help='Score field used for --per-frame-top-k ranking (default: scores).',
    )
    parser.add_argument(
        '--run-name',
        default=None,
        help='Optional explicit refined run name (default: timestamped refined_detect_<local_time>).',
    )
    parser.add_argument(
        '--dish-mask-boundary-tolerance-mm',
        type=float,
        default=None,
        help=(
            'Physical tolerance outside the fitted dish boundary. Default: 0.5 mm '
            '(or refine_detect.dish_mask_boundary_tolerance_mm from config).'
        ),
    )
    parser.add_argument(
        '--pixels-per-mm-camera',
        type=float,
        default=None,
        help=(
            'Explicit raw camera pixels/mm calibration override for the dish tolerance. '
            'Used only when supplied and recorded in provenance.'
        ),
    )
    parser.add_argument(
        '--registered-gate-requirement',
        choices=tuple(sorted(_REGISTERED_GATE_REQUIREMENTS)),
        default='off',
        help=(
            'Modern registered geometry policy: off, if_available, or required. '
            'This is independent of the legacy dish-mask configuration.'
        ),
    )
    parser.add_argument(
        '--registered-gate-run',
        help=(
            'Exact analysis/detection_gate_runs child to consume. Refinement never '
            'resolves a mutable latest gate pointer.'
        ),
    )
    parser.add_argument(
        '--selector-ineligible',
        action='store_true',
        help=(
            'Complete the refined run without updating refined-run selectors. '
            'Use for explicitly selected canaries and review seeds.'
        ),
    )
    parser.add_argument(
        '--skip-completion-status',
        action='store_true',
        help=(
            'Do not project completion into registry/stage status. '
            'Use for unregistered benchmark canaries.'
        ),
    )
    parser.add_argument(
        '--require-active-canonical-source',
        action='store_true',
        help='Fail unless the source is the selected canonical-v3 detection authority.',
    )
    parser.add_argument('--expected-source-manifest-digest')
    parser.add_argument('--source-publication-receipt', type=Path)
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    from pathlib import Path
    
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    
    try:
        run_name = create_refined_run(
            zarr_path=args.zarr_path,
            detect_run=args.detect_run,
            quality_run=args.quality_run,
            quality_group_path=args.quality_group_path,
            config=config,
            max_gap=args.max_gap,
            interpolation_method=args.method,
            remove_jumps=args.remove_jumps,
            remove_blips=args.remove_blips,
            command=' '.join(sys.argv),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            save_visuals=args.save_visuals,
            show_visuals=args.show_visuals,
            visuals_dpi=args.visuals_dpi,
            require_detect_quality=not args.allow_missing_quality,
            per_frame_top_k=args.per_frame_top_k,
            top_k_score_field=args.top_k_score_field,
            detect_family_path=args.detect_family_path,
            refined_family_path=args.refined_family_path,
            refined_run_name=args.run_name,
            dish_mask_boundary_tolerance_mm=args.dish_mask_boundary_tolerance_mm,
            pixels_per_mm_camera=args.pixels_per_mm_camera,
            registered_gate_requirement=args.registered_gate_requirement,
            registered_gate_run=args.registered_gate_run,
            stage_selector_eligible=not args.selector_ineligible,
            emit_completion_status=not args.skip_completion_status,
            require_active_canonical_source=args.require_active_canonical_source,
            expected_source_manifest_digest=args.expected_source_manifest_digest,
            source_publication_receipt=args.source_publication_receipt,
        )
        
        print(f"\n✓ Created refined run: {run_name}")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
