#!/usr/bin/env python3
"""
Prepare a YOLO detection training config + manifest from Palette Zarr archives.

Validates crop provenance, bbox integrity, and downsample frame availability.
Optionally captures experiment provenance (arena/camera/calibration) for auditing.
Current refined-detect training should point at the canonical curated surface
on `refined_detect_runs/<run>/instances`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml
import zarr
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..utils.zarr_metadata import get_downsample_array_path, get_downsample_shape
from ..registry.db import Registry, RegistryPaths
from ..registry.stage_complete import extract_dataset_metadata
from ..shared.refined_detect_curation import (
    build_source_detection_decision_summary,
    extract_present_curated_rows,
    has_curated_refined_source_detections_projection,
    has_curated_refined_detect_surface,
)
from ..shared.zarr_helpers import open_zarr_group_direct
from ..utils.system import build_invocation_record


class CameraParameters(BaseModel):
    camera_id: Optional[str] = None
    native_width_px: Optional[int] = None
    native_height_px: Optional[int] = None
    pixels_per_mm_camera: Optional[float] = None
    pixels_per_mm_projector: Optional[float] = None
    homography_matrix: Optional[List[List[float]]] = None
    stimulus_offset_x: Optional[float] = None
    stimulus_offset_y: Optional[float] = None

    model_config = ConfigDict(extra="allow")


class ArenaParameters(BaseModel):
    arena_id: Optional[str] = None
    arena_number: Optional[str] = None
    dish_design: Optional[str] = None
    arena_config_name: Optional[str] = None
    shape: Optional[str] = None
    center_x_px: Optional[float] = None
    center_y_px: Optional[float] = None
    radius_px: Optional[float] = None
    width_px: Optional[float] = None
    height_px: Optional[float] = None
    corner_radius_px: Optional[float] = None
    diameter_mm: Optional[float] = None

    model_config = ConfigDict(extra="allow")


class CalibrationSummary(BaseModel):
    pixel_to_mm: Optional[float] = None
    pixels_per_mm: Optional[float] = None
    pixels_per_mm_camera: Optional[float] = None
    pixels_per_mm_projector: Optional[float] = None

    model_config = ConfigDict(extra="allow")


class ProvenanceInfo(BaseModel):
    source_h5: Optional[str] = None
    stimulus_run: Optional[str] = None
    arena: Optional[ArenaParameters] = None
    camera: Optional[CameraParameters] = None
    calibration: Optional[CalibrationSummary] = None
    rig_info: Optional[Dict[str, Any]] = None
    arena_config: Optional[Dict[str, Any]] = None
    camera_metadata: Optional[Dict[str, Any]] = None
    camera_config_hash: Optional[str] = None
    video_codec: Optional[str] = None
    video_pix_fmt: Optional[str] = None
    video_source: Optional[str] = None
    format_title: Optional[str] = None
    format_comment: Optional[str] = None
    format_encoder: Optional[str] = None
    encoder_name: Optional[str] = None
    encoder_codec: Optional[str] = None
    encoder_preset: Optional[str] = None
    encoder_tuning: Optional[str] = None
    encoder_rc: Optional[str] = None
    encoder_bpp: Optional[float] = None
    encoder_target_bps: Optional[int] = None
    encoder_res: Optional[str] = None
    encoder_res_width: Optional[int] = None
    encoder_res_height: Optional[int] = None
    encoder_fps: Optional[float] = None
    encoder_color: Optional[int] = None
    encoder_params: Optional[Dict[str, Any]] = None
    provenance_source: str = "missing"
    missing_fields: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    override_fields: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class DatasetManifest(BaseModel):
    name: str
    zarr_path: str
    dataset_id: Optional[str] = None
    session_uuid: Optional[str] = None
    crop_run: Optional[str] = None
    bbox_array_path: str
    detection_source_type: str
    detection_source_path: Optional[str] = None
    includes_interpolated: bool = False
    input_format: str
    images_ds_shape: List[int]
    total_bboxes: int
    manual_edited_bboxes: int = 0
    source_detection_decision_counts: Dict[str, int] = Field(default_factory=dict)
    invalid_bboxes: int
    invalid_bbox_sample: List[int] = Field(default_factory=list)
    detection_source_counts: Dict[str, int] = Field(default_factory=dict)
    frame_indices_present: bool = False
    detection_source_present: bool = False
    provenance: Optional[ProvenanceInfo] = None

    model_config = ConfigDict(extra="allow")


class TrainingManifest(BaseModel):
    created_at_utc: str
    task: str
    source_type: str
    input_format: str
    imgsz: List[int]
    datasets: List[DatasetManifest]
    base_config_path: Optional[str] = None
    output_config_path: Optional[str] = None
    output_manifest_path: Optional[str] = None
    project: Optional[str] = None
    run_name: Optional[str] = None
    provenance_policy: str
    registry_path: Optional[str] = None
    set_name: Optional[str] = None
    set_version: Optional[int] = None
    set_id: Optional[str] = None
    query_filter: Optional[Dict[str, Any]] = None
    invocation: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")


def _sanitize_name(value: str) -> str:
    cleaned = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned)


def _next_version(prefix: str, config_dir: Path, manifest_dir: Path) -> int:
    versions: List[int] = []
    config_re = re.compile(rf"^{re.escape(prefix)}_v(\d+)\.ya?ml$")
    manifest_re = re.compile(rf"^{re.escape(prefix)}_v(\d+)\.manifest\.json$")
    for path in config_dir.glob(f"{prefix}_v*.y*ml"):
        match = config_re.match(path.name)
        if match:
            versions.append(int(match.group(1)))
    for path in manifest_dir.glob(f"{prefix}_v*.manifest.json"):
        match = manifest_re.match(path.name)
        if match:
            versions.append(int(match.group(1)))
    return max(versions) + 1 if versions else 1


def _build_training_set_id(set_name: str, set_version: int) -> str:
    return f"detect_{set_name}_v{set_version:03d}"


def _build_query_filter_payload(
    args: argparse.Namespace,
    *,
    require_approved: bool,
    prefer_manual: bool,
    set_name: Optional[str],
    set_version: Optional[int],
    selected_paths: List[Path],
) -> Dict[str, Any]:
    return {
        "tool": "fisheye.diagnostics.prepare_detect_training",
        "task": "detect",
        "source_type": args.source_type,
        "input_format": args.input_format,
        "provenance_policy": args.provenance_policy,
        "require_approved": bool(require_approved),
        "prefer_manual": bool(prefer_manual),
        "allow_source_mismatch": bool(args.allow_source_mismatch),
        "base_config_path": str(args.base_config),
        "imgsz_override": int(args.imgsz) if args.imgsz is not None else None,
        "set_name": set_name,
        "set_version": set_version,
        "selected_zarr_paths": [str(path) for path in selected_paths],
    }


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, np.generic):
        try:
            return float(value.item())
        except Exception:
            return None
    return None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        return int(value)
    except Exception:
        return None


def _lookup_registry_dataset_id(registry: Registry, zarr_path: Path) -> Optional[str]:
    candidates = [str(zarr_path)]
    try:
        resolved = str(zarr_path.resolve())
    except OSError:
        resolved = None
    if resolved and resolved not in candidates:
        candidates.append(resolved)
    for candidate in candidates:
        row = registry.conn.execute(
            "SELECT dataset_id FROM datasets WHERE zarr_path = ? LIMIT 1;",
            (candidate,),
        ).fetchone()
        if row is not None and row["dataset_id"] is not None:
            text = str(row["dataset_id"]).strip()
            if text:
                return text
    return None


def _resolve_manifest_dataset_identity(
    *,
    root: zarr.Group,
    zarr_path: Path,
    registry: Optional[Registry],
) -> Tuple[str, Optional[str]]:
    metadata = extract_dataset_metadata(root, zarr_path)
    dataset_id = metadata.dataset_id
    session_uuid = metadata.session_uuid
    if registry is not None:
        canonical_dataset_id = _lookup_registry_dataset_id(registry, zarr_path)
        if canonical_dataset_id:
            dataset_id = canonical_dataset_id
    return dataset_id, session_uuid


def _parse_arena_config(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except Exception:
            return {}
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if isinstance(raw, dict):
        return raw
    return {}


def _normalize_state(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text.lower() if text else None


def _review_state(status: Any) -> Optional[str]:
    if not isinstance(status, dict):
        return None
    return _normalize_state(status.get("state"))


def _resolve_refined_group(root: zarr.Group, crop_group: Optional[zarr.Group]) -> Optional[zarr.Group]:
    if crop_group is not None:
        ref = crop_group.attrs.get("detect_review_status_ref")
        if isinstance(ref, (bytes, bytearray)):
            ref = ref.decode("utf-8", "ignore")
        if isinstance(ref, str):
            ref = ref.strip("/")
            if ref in root:
                return root[ref]
    refined_parent = root.get("refined_detect_runs") or root.get("refined_runs")
    if refined_parent is None:
        return None
    latest = refined_parent.attrs.get("latest")
    if isinstance(latest, (bytes, bytearray)):
        latest = latest.decode("utf-8", "ignore")
    if isinstance(latest, str) and latest in refined_parent:
        return refined_parent[latest]
    return None


def _manual_group_name(refined_group: Optional[zarr.Group]) -> Optional[str]:
    if refined_group is None:
        return None
    manual_label = refined_group.attrs.get("manual_review_latest")
    if isinstance(manual_label, (bytes, bytearray)):
        manual_label = manual_label.decode("utf-8", "ignore")
    if isinstance(manual_label, str) and manual_label in refined_group:
        return manual_label
    if "manual" in refined_group:
        return "manual"
    return None


def _group_path(group: zarr.Group) -> str:
    path = getattr(group, "path", "")
    if isinstance(path, (bytes, bytearray)):
        path = path.decode("utf-8", "ignore")
    return str(path).strip("/")


_ARENA_NUMBER_RE = re.compile(r"arena[_-]?(\d+)", re.IGNORECASE)


def _infer_arena_number(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8", "ignore")
        text = str(value).strip()
        if not text:
            continue
        if text.isdigit():
            return text
        match = _ARENA_NUMBER_RE.search(text)
        if match:
            return match.group(1)
    return None


def _build_arena_info(arena_config: Dict[str, Any], root: zarr.Group) -> ArenaParameters:
    arena_id = arena_config.get("arena_id") or root.attrs.get("arena_id")
    arena_config_name = arena_config.get("arena_config_name") or arena_config.get("config_name")
    arena_number = (
        arena_config.get("arena_number")
        or root.attrs.get("arena_number")
        or _infer_arena_number(arena_id, arena_config_name, arena_config.get("arena_id"))
    )
    dish_design = (
        arena_config.get("selected_dish_type_name")
        or arena_config.get("selected_dish_name")
        or arena_config.get("dish_design")
        or arena_config.get("arena_design")
        or root.attrs.get("dish_design")
    )
    shape = arena_config.get("experimental_area_shape") or arena_config.get("swimmable_area_shape")
    center_x = arena_config.get("experimental_area_center_x_px") or arena_config.get(
        "swimmable_area_center_x_px"
    )
    center_y = arena_config.get("experimental_area_center_y_px") or arena_config.get(
        "swimmable_area_center_y_px"
    )
    radius = arena_config.get("experimental_area_radius_px") or arena_config.get("swimmable_area_radius_px")
    width = arena_config.get("experimental_area_width_px") or arena_config.get("swimmable_area_width_px")
    height = arena_config.get("experimental_area_height_px") or arena_config.get("swimmable_area_height_px")
    corner_radius = arena_config.get("experimental_area_corner_radius_px")

    diameter_mm = None
    if "calibration" in root and "arena" in root["calibration"]:
        arena_group = root["calibration"]["arena"]
        diameter_mm = _as_float(arena_group.attrs.get("diameter_mm"))

    return ArenaParameters(
        arena_id=str(arena_id) if arena_id is not None else None,
        arena_number=str(arena_number) if arena_number is not None else None,
        dish_design=str(dish_design) if dish_design is not None else None,
        arena_config_name=str(arena_config_name) if arena_config_name is not None else None,
        shape=str(shape) if shape is not None else None,
        center_x_px=_as_float(center_x),
        center_y_px=_as_float(center_y),
        radius_px=_as_float(radius),
        width_px=_as_float(width),
        height_px=_as_float(height),
        corner_radius_px=_as_float(corner_radius),
        diameter_mm=_as_float(diameter_mm),
    )


def _load_camera_from_stimulus(stim_group: zarr.Group) -> Optional[CameraParameters]:
    calib_group = stim_group.get("calibration")
    if calib_group is None:
        return None
    camera_ids = list(calib_group.keys())
    if not camera_ids:
        return None
    camera_id = camera_ids[0]
    cam_group = calib_group.get(camera_id)
    if cam_group is None:
        return None
    homography = None
    if "homography_matrix" in cam_group:
        try:
            homography = np.asarray(cam_group["homography_matrix"][:], dtype=float).tolist()
        except Exception:
            homography = None
    return CameraParameters(
        camera_id=str(camera_id),
        native_width_px=_as_int(cam_group.attrs.get("camera_width_px")),
        native_height_px=_as_int(cam_group.attrs.get("camera_height_px")),
        pixels_per_mm_camera=_as_float(cam_group.attrs.get("pixels_per_mm_camera")),
        pixels_per_mm_projector=_as_float(cam_group.attrs.get("pixels_per_mm_projector")),
        stimulus_offset_x=_as_float(cam_group.attrs.get("stimulus_offset_x")),
        stimulus_offset_y=_as_float(cam_group.attrs.get("stimulus_offset_y")),
        homography_matrix=homography,
    )


def _load_camera_from_root(root: zarr.Group) -> Optional[CameraParameters]:
    calib_group = root.get("calibration")
    if calib_group is None:
        return None
    camera_group = calib_group.get("cameras")
    if camera_group is None:
        return None
    camera_ids = list(camera_group.keys())
    if not camera_ids:
        return None
    camera_id = calib_group.attrs.get("primary_camera_id") or camera_ids[0]
    cam_group = camera_group.get(str(camera_id))
    if cam_group is None:
        return None
    homography = None
    if "homography_matrix" in cam_group:
        try:
            homography = np.asarray(cam_group["homography_matrix"][:], dtype=float).tolist()
        except Exception:
            homography = None
    return CameraParameters(
        camera_id=str(camera_id),
        native_width_px=_as_int(cam_group.attrs.get("native_width_px")),
        native_height_px=_as_int(cam_group.attrs.get("native_height_px")),
        pixels_per_mm_camera=_as_float(cam_group.attrs.get("pixels_per_mm_camera")),
        pixels_per_mm_projector=_as_float(cam_group.attrs.get("pixels_per_mm_projector")),
        stimulus_offset_x=_as_float(cam_group.attrs.get("stimulus_offset_x")),
        stimulus_offset_y=_as_float(cam_group.attrs.get("stimulus_offset_y")),
        homography_matrix=homography,
    )


def _load_calibration_summary(root: zarr.Group) -> CalibrationSummary:
    summary = CalibrationSummary()
    calib_group = root.get("calibration")
    if calib_group is None:
        return summary
    summary.pixel_to_mm = _as_float(calib_group.attrs.get("pixel_to_mm"))
    summary.pixels_per_mm = _as_float(calib_group.attrs.get("pixels_per_mm"))
    summary.pixels_per_mm_camera = _as_float(calib_group.attrs.get("pixels_per_mm_camera"))
    summary.pixels_per_mm_projector = _as_float(calib_group.attrs.get("pixels_per_mm_projector"))
    return summary


def _load_rig_info(root: zarr.Group) -> Optional[Dict[str, Any]]:
    calib_group = root.get("calibration")
    if calib_group is None:
        return None
    rig_group = calib_group.get("rig_info")
    if rig_group is None:
        return None
    return {k: rig_group.attrs.get(k) for k in rig_group.attrs.keys()}


def _hash_json(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(canonical.encode("utf-8")).hexdigest()


def _load_camera_metadata(root: zarr.Group) -> Optional[Dict[str, Any]]:
    analysis = root.get("analysis_metadata")
    if analysis is None:
        return None
    raw = analysis.attrs.get("camera_metadata")
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except Exception:
            return None
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    if isinstance(raw, dict):
        return raw
    return None


def _load_video_metadata(root: zarr.Group) -> Dict[str, Any]:
    raw = root.get("raw_video")
    if raw is None:
        return {
            "video_codec": None,
            "video_pix_fmt": None,
            "video_source": None,
            "format_title": None,
            "format_comment": None,
            "format_encoder": None,
            "encoder_name": None,
            "encoder_codec": None,
            "encoder_preset": None,
            "encoder_tuning": None,
            "encoder_rc": None,
            "encoder_bpp": None,
            "encoder_target_bps": None,
            "encoder_res": None,
            "encoder_res_width": None,
            "encoder_res_height": None,
            "encoder_fps": None,
            "encoder_color": None,
            "encoder_params": None,
        }
    return {
        "video_codec": raw.attrs.get("video_codec") or raw.attrs.get("codec"),
        "video_pix_fmt": raw.attrs.get("video_pix_fmt") or raw.attrs.get("pix_fmt"),
        "video_source": raw.attrs.get("source_video"),
        "format_title": raw.attrs.get("format_title"),
        "format_comment": raw.attrs.get("format_comment"),
        "format_encoder": raw.attrs.get("format_encoder"),
        "encoder_name": raw.attrs.get("encoder_name"),
        "encoder_codec": raw.attrs.get("encoder_codec"),
        "encoder_preset": raw.attrs.get("encoder_preset"),
        "encoder_tuning": raw.attrs.get("encoder_tuning"),
        "encoder_rc": raw.attrs.get("encoder_rc"),
        "encoder_bpp": raw.attrs.get("encoder_bpp"),
        "encoder_target_bps": raw.attrs.get("encoder_target_bps"),
        "encoder_res": raw.attrs.get("encoder_res"),
        "encoder_res_width": raw.attrs.get("encoder_res_width"),
        "encoder_res_height": raw.attrs.get("encoder_res_height"),
        "encoder_fps": raw.attrs.get("encoder_fps"),
        "encoder_color": raw.attrs.get("encoder_color"),
        "encoder_params": raw.attrs.get("encoder_params"),
    }


def _extract_provenance(
    root: zarr.Group,
    override: Optional[Dict[str, Any]],
    provenance_policy: str,
) -> ProvenanceInfo:
    provenance = ProvenanceInfo()

    stim_group = None
    stimulus_run = None
    if "analysis" in root and "stimulus_runs" in root["analysis"]:
        stim_parent = root["analysis"]["stimulus_runs"]
        stimulus_run = stim_parent.attrs.get("latest")
        if stimulus_run and stimulus_run in stim_parent:
            stim_group = stim_parent[stimulus_run]
            provenance.stimulus_run = str(stimulus_run)
            provenance.source_h5 = stim_group.attrs.get("source_h5")

    arena_config = {}
    if stim_group is not None:
        arena_config = _parse_arena_config(stim_group.attrs.get("arena_config_json"))
        if arena_config:
            provenance.arena_config = arena_config
            provenance.arena = _build_arena_info(arena_config, root)
        provenance.camera = _load_camera_from_stimulus(stim_group)

    if provenance.arena is None:
        provenance.arena = _build_arena_info({}, root)
    if provenance.camera is None:
        provenance.camera = _load_camera_from_root(root)

    provenance.calibration = _load_calibration_summary(root)
    provenance.rig_info = _load_rig_info(root)
    camera_metadata = _load_camera_metadata(root)
    if camera_metadata:
        provenance.camera_metadata = camera_metadata
        provenance.camera_config_hash = _hash_json(camera_metadata)
    video_meta = _load_video_metadata(root)
    provenance.video_codec = video_meta.get("video_codec")
    provenance.video_pix_fmt = video_meta.get("video_pix_fmt")
    provenance.video_source = video_meta.get("video_source")
    provenance.format_title = video_meta.get("format_title")
    provenance.format_comment = video_meta.get("format_comment")
    provenance.format_encoder = video_meta.get("format_encoder")
    provenance.encoder_name = video_meta.get("encoder_name")
    provenance.encoder_codec = video_meta.get("encoder_codec")
    provenance.encoder_preset = video_meta.get("encoder_preset")
    provenance.encoder_tuning = video_meta.get("encoder_tuning")
    provenance.encoder_rc = video_meta.get("encoder_rc")
    provenance.encoder_bpp = video_meta.get("encoder_bpp")
    provenance.encoder_target_bps = video_meta.get("encoder_target_bps")
    provenance.encoder_res = video_meta.get("encoder_res")
    provenance.encoder_res_width = video_meta.get("encoder_res_width")
    provenance.encoder_res_height = video_meta.get("encoder_res_height")
    provenance.encoder_fps = video_meta.get("encoder_fps")
    provenance.encoder_color = video_meta.get("encoder_color")
    provenance.encoder_params = video_meta.get("encoder_params")

    if stim_group is not None:
        provenance.provenance_source = "stimulus_import"
    elif "calibration" in root:
        provenance.provenance_source = "zarr"

    if override:
        override_fields = []
        try:
            override_model = ProvenanceInfo.model_validate(override)
        except ValidationError:
            override_model = ProvenanceInfo.model_validate(override, strict=False)
        for field in ("arena", "camera", "calibration", "rig_info", "arena_config", "source_h5", "stimulus_run"):
            override_value = getattr(override_model, field)
            if override_value is not None:
                setattr(provenance, field, override_value)
                override_fields.append(field)
        provenance.override_fields = override_fields
        provenance.provenance_source = "user_override" if provenance.provenance_source == "missing" else "mixed"

    missing = []
    arena = provenance.arena
    camera = provenance.camera
    if not arena or not arena.dish_design:
        missing.append("dish_design")
    if not arena or not arena.arena_number:
        missing.append("arena_number")
    if not camera or not camera.camera_id:
        missing.append("camera_id")
    if not camera or camera.pixels_per_mm_camera is None:
        missing.append("pixels_per_mm_camera")

    provenance.missing_fields = missing

    if missing and provenance_policy == "warn":
        provenance.warnings.append(
            "Missing provenance fields: " + ", ".join(missing)
        )
    if missing and provenance_policy == "strict":
        raise ValueError(
            "Missing required provenance fields: "
            + ", ".join(missing)
            + ". Provide --metadata-json or run import_stimulus_to_zarr."
        )

    return provenance


def _load_override(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Metadata override file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yml", ".yaml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _ensure_dummy_paths(config: Dict[str, Any], config_path: Path) -> List[Path]:
    created: List[Path] = []
    data_root = config.get("path")
    base_dir = Path(data_root) if data_root else config_path.parent
    for key in ("train", "val"):
        value = config.get(key)
        if not isinstance(value, str) or not value:
            continue
        target = Path(value)
        if not target.is_absolute():
            target = (base_dir / target).resolve()
        if target.exists():
            continue
        if target.suffix.lower() not in {".txt"} and "dummy" not in target.name:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
        created.append(target)
    return created


def _validate_bboxes(bbox: np.ndarray) -> Tuple[int, List[int]]:
    if bbox.size == 0:
        return 0, []
    nan_mask = np.isnan(bbox).any(axis=1)
    out_of_range = (bbox[:, 0] < 0) | (bbox[:, 0] > 1) | (bbox[:, 1] < 0) | (bbox[:, 1] > 1)
    invalid_dims = (bbox[:, 2] <= 0) | (bbox[:, 3] <= 0)
    invalid = nan_mask | out_of_range | invalid_dims
    invalid_indices = np.where(invalid)[0].astype(int).tolist()
    return int(np.sum(invalid)), invalid_indices[:10]


def _ensure_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("Base config is empty or invalid.")
    if "datasets" not in config:
        raise ValueError("Base config missing 'datasets' section.")
    if "training_params" not in config:
        raise ValueError("Base config missing 'training_params' section.")
    return config


def _choose_dataset_name(existing: set, path: Path, index: int) -> str:
    stem = path.stem
    name = stem
    if name in existing:
        name = f"{stem}_{index}"
    existing.add(name)
    return name


def _log_timing(enabled: bool, label: str, started_at: float) -> float:
    ended_at = perf_counter()
    if enabled:
        print(f"[timing] {label}: {ended_at - started_at:.3f}s")
    return ended_at


def _print_summary(manifest: TrainingManifest) -> None:
    print("\nDetection Training Preflight")
    print(f"  Task: {manifest.task}")
    print(f"  Source type: {manifest.source_type}")
    print(f"  Input format: {manifest.input_format}")
    print(f"  imgsz: {manifest.imgsz}")
    if manifest.set_id:
        print(f"  Set ID: {manifest.set_id}")
    for dataset in manifest.datasets:
        print(f"\nDataset: {dataset.name}")
        print(f"  Zarr: {dataset.zarr_path}")
        print(f"  Crop run: {dataset.crop_run}")
        print(f"  Detection source: {dataset.detection_source_type}")
        print(f"  Detection source path: {dataset.detection_source_path}")
        print(f"  Total bboxes: {dataset.total_bboxes}")
        if dataset.manual_edited_bboxes:
            print(f"  Manual-edited bboxes: {dataset.manual_edited_bboxes}")
        if dataset.source_detection_decision_counts:
            print(f"  Source detection decisions: {dataset.source_detection_decision_counts}")
        print(f"  Invalid bboxes: {dataset.invalid_bboxes}")
        if dataset.invalid_bbox_sample:
            print(f"  Invalid bbox sample: {dataset.invalid_bbox_sample}")
        if dataset.detection_source_counts:
            print(f"  Detection source counts: {dataset.detection_source_counts}")
        if dataset.provenance and dataset.provenance.warnings:
            for warning in dataset.provenance.warnings:
                print(f"  ⚠ {warning}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a YOLO detection training config and manifest from Palette Zarr data."
    )
    parser.add_argument("zarr_paths", nargs="+", type=Path, help="One or more Palette Zarr paths.")
    parser.add_argument(
        "--source-type",
        choices=["detect", "filtered", "interpolated", "manual", "refined"],
        default="refined",
        help=(
            "Detection source type to train on (default: refined canonical "
            "curated surface on refined_detect_runs/<run>/instances)."
        ),
    )
    parser.add_argument(
        "--input-format",
        choices=["gray", "rgb"],
        default="gray",
        help="Downsample frame format (images_ds or images_ds_rgb).",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/fisheye/detect_config.yaml"),
        help="Base detect training config to use for hyperparameters.",
    )
    parser.add_argument("--out-config", type=Path, help="Output path for the generated config YAML.")
    parser.add_argument("--out-manifest", type=Path, help="Output path for the manifest JSON.")
    parser.add_argument(
        "--set-name",
        type=str,
        help="Auto-generate versioned output paths under runs/{configs,manifests}/detect/.",
    )
    parser.add_argument(
        "--set-version",
        type=int,
        help="Optional explicit version number to use with --set-name.",
    )
    parser.add_argument("--project", type=str, help="Ultralytics project directory for outputs.")
    parser.add_argument("--run-name", type=str, help="Suggested run name for training.")
    parser.add_argument("--imgsz", type=int, help="Override training image size.")
    parser.add_argument(
        "--provenance-policy",
        choices=["warn", "strict", "ignore"],
        default="warn",
        help="How to handle missing provenance fields.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        help="Optional JSON/YAML file with provenance overrides.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Optional registry SQLite path to update.",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Write scanned datasets into the registry.",
    )
    parser.add_argument(
        "--allow-source-mismatch",
        action="store_true",
        help="Allow crop source type to differ from requested source_type.",
    )
    parser.add_argument(
        "--require-approved",
        action="store_true",
        help="Require approved detect + crop reviews (default).",
    )
    parser.add_argument(
        "--allow-unapproved",
        action="store_true",
        help="Include datasets even if detect/crop reviews are missing or unapproved.",
    )
    parser.add_argument(
        "--prefer-manual",
        action="store_true",
        help="Legacy sparse-mode policy: prefer manual subgroups when available.",
    )
    parser.add_argument(
        "--no-prefer-manual",
        action="store_true",
        help="Disable legacy sparse manual-subgroup preference checks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the config + manifest without writing files.",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Print timing logs for major preflight phases.",
    )

    cli_argv = [str(token) for token in (list(argv) if argv is not None else list(sys.argv[1:]))]
    args = parser.parse_args(cli_argv)
    run_started = perf_counter()

    phase_started = perf_counter()
    override = _load_override(args.metadata_json)
    _log_timing(args.timing, "load metadata override", phase_started)
    require_approved = True if args.require_approved else not args.allow_unapproved
    prefer_manual = True if args.prefer_manual else not args.no_prefer_manual

    resolved_set_name: Optional[str] = None
    set_version: Optional[int] = None
    set_id: Optional[str] = None
    if args.set_name:
        safe_name = _sanitize_name(args.set_name)
        resolved_set_name = safe_name
        config_dir = Path("runs") / "configs" / "detect"
        manifest_dir = Path("runs") / "manifests" / "detect"
        if args.set_version is not None:
            if args.set_version < 1:
                raise ValueError("--set-version must be >= 1")
            set_version = args.set_version
        else:
            set_version = _next_version(safe_name, config_dir, manifest_dir)
        suffix = f"_v{set_version:03d}"
        set_id = _build_training_set_id(safe_name, set_version)
        if args.out_config is None:
            args.out_config = config_dir / f"{safe_name}{suffix}.yaml"
        if args.out_manifest is None:
            args.out_manifest = (
                Path(args.out_config).with_suffix(".manifest.json")
                if args.out_config is not None
                else (manifest_dir / f"{safe_name}{suffix}.manifest.json")
            )

    phase_started = perf_counter()
    invocation_payload = build_invocation_record(
        tool="fisheye.diagnostics.prepare_detect_training",
        args=args,
        argv=cli_argv,
    )
    _log_timing(args.timing, "capture invocation metadata", phase_started)

    if not args.base_config.exists():
        raise FileNotFoundError(f"Base config not found: {args.base_config}")
    phase_started = perf_counter()
    base_config = _ensure_config(yaml.safe_load(args.base_config.read_text(encoding="utf-8")))
    _log_timing(args.timing, "load base config", phase_started)

    dataset_entries: Dict[str, Dict[str, Any]] = {}
    manifests: List[DatasetManifest] = []
    skipped: List[str] = []
    registry: Optional[Registry] = None
    registry_path: Optional[str] = None
    should_write_training_set = bool(set_id and not args.dry_run)
    should_open_registry = bool(args.register or args.registry or should_write_training_set)
    should_register_datasets = bool(args.register or args.registry)
    if should_open_registry:
        resolved_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
        registry_path = str(resolved_path)
        phase_started = perf_counter()
        registry = Registry(resolved_path)
        _log_timing(args.timing, f"open registry ({resolved_path})", phase_started)
    seen_names: set = set()
    reference_shape: Optional[Tuple[int, int]] = None

    for idx, zarr_path in enumerate(args.zarr_paths, start=1):
        dataset_started = perf_counter()
        dataset_label = zarr_path.name
        if not zarr_path.exists():
            raise FileNotFoundError(f"Zarr path not found: {zarr_path}")
        phase_started = perf_counter()
        root = open_zarr_group_direct(zarr_path, mode="r")
        _log_timing(args.timing, f"{dataset_label}: open zarr group", phase_started)

        array_path = get_downsample_array_path(root, format_hint=args.input_format)
        if array_path is None:
            raise ValueError(
                f"Downsampled frames for format '{args.input_format}' not found in {zarr_path.name}."
            )
        ds_shape = get_downsample_shape(root, format_hint=args.input_format)
        if ds_shape is None or len(ds_shape) < 2:
            raise ValueError(f"Downsampled frame shape missing for {zarr_path.name}.")
        height, width = int(ds_shape[0]), int(ds_shape[1])
        if reference_shape is None:
            reference_shape = (height, width)
        elif reference_shape != (height, width) and args.imgsz is None:
            raise ValueError(
                f"Downsample sizes differ across datasets ({reference_shape} vs {(height, width)}). "
                "Use --imgsz to override."
            )

        crop_run = None
        crop_group = None
        bbox_array_path = ""
        detection_source_type = "detect"
        detection_source_path = None
        includes_interpolated = False
        detection_source_counts: Dict[str, int] = {}
        frame_indices_present = False
        detection_source_present = False
        manual_edited_bboxes = 0
        source_detection_decision_counts: Dict[str, int] = {}
        refined_group: Optional[zarr.Group] = None

        if "crop_runs" in root and root["crop_runs"].attrs.get("latest"):
            crop_run = root["crop_runs"].attrs.get("latest")
            crop_group = root["crop_runs"][crop_run]
            if "bbox_norm_coords" not in crop_group:
                raise ValueError(f"crop_runs/{crop_run} missing bbox_norm_coords in {zarr_path.name}.")
            bbox_array_path = f"crop_runs/{crop_run}/bbox_norm_coords"
            detection_source_type = crop_group.attrs.get("detection_source_type", "detect")
            detection_source_path = crop_group.attrs.get("detection_source_path")
            includes_interpolated = bool(crop_group.attrs.get("includes_interpolated", False))
            frame_indices_present = "frame_indices" in crop_group
            detection_source_present = "detection_source" in crop_group
            if detection_source_present:
                phase_started = perf_counter()
                detection_source = np.asarray(crop_group["detection_source"][:], dtype=np.int8)
                unique, counts = np.unique(detection_source, return_counts=True)
                detection_source_counts = {str(int(k)): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
                _log_timing(args.timing, f"{dataset_label}: read detection_source", phase_started)
        elif "detect_runs" in root and root["detect_runs"].attrs.get("latest"):
            detect_run = root["detect_runs"].attrs.get("latest")
            detect_group = root["detect_runs"][detect_run]
            if "bbox_norm_coords" not in detect_group:
                raise ValueError(f"detect_runs/{detect_run} missing bbox_norm_coords in {zarr_path.name}.")
            bbox_array_path = f"detect_runs/{detect_run}/bbox_norm_coords"
            frame_indices_present = "frame_indices" in detect_group
        elif args.source_type == "refined":
            refined_group = _resolve_refined_group(root, None)
            if refined_group is None or not has_curated_refined_detect_surface(refined_group):
                raise ValueError(f"No crop_runs, detect_runs, or curated refined detect surface found in {zarr_path.name}.")
            detection_source_type = "refined"
        else:
            raise ValueError(f"No crop_runs or detect_runs found in {zarr_path.name}.")

        if refined_group is None:
            refined_group = _resolve_refined_group(
                root,
                None if args.source_type == "refined" else crop_group,
            )
        manual_group = _manual_group_name(refined_group)
        if args.source_type == "refined":
            if refined_group is None or not has_curated_refined_detect_surface(refined_group):
                skipped.append(f"{zarr_path.name}: refined detect surface missing")
                _log_timing(args.timing, f"{dataset_label}: skipped (refined detect surface missing)", dataset_started)
                continue
            refined_path = _group_path(refined_group)
            if not refined_path:
                raise ValueError(f"{zarr_path.name}: refined detect group path could not be resolved.")
            instances_path = f"{refined_path}/instances"
            instances_group = root[instances_path]
            if "bbox_norm_coords" not in instances_group:
                raise ValueError(f"{instances_path} missing bbox_norm_coords in {zarr_path.name}.")
            bbox_array_path = f"{instances_path}/bbox_norm_coords"
            detection_source_type = "refined"
            detection_source_path = instances_path
            # Refined training labels come from the canonical curated surface,
            # not from historical crop metadata that may point at an older run.
            crop_run = None
            crop_group = None
            frame_indices_present = "frame_indices" in instances_group
            detection_source_present = False
            if "source_kind_codes" in instances_group:
                phase_started = perf_counter()
                source_kind_codes = np.asarray(instances_group["source_kind_codes"][:], dtype=np.int16)
                unique, counts = np.unique(source_kind_codes, return_counts=True)
                detection_source_counts = {str(int(k)): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
                includes_interpolated = bool(np.any(source_kind_codes == 2))
                _log_timing(args.timing, f"{dataset_label}: read source_kind_codes", phase_started)
        if (
            detection_source_type == "refined"
            and refined_group is not None
            and has_curated_refined_detect_surface(refined_group)
        ):
            try:
                present_rows = extract_present_curated_rows(refined_group)
                manual_flags = present_rows.get("manual_edit_flags")
                if manual_flags is not None:
                    manual_edited_bboxes = int(np.count_nonzero(np.asarray(manual_flags, dtype=bool)))
            except Exception:
                manual_edited_bboxes = 0
            if has_curated_refined_source_detections_projection(refined_group):
                try:
                    source_summary = build_source_detection_decision_summary(refined_group)
                    if source_summary:
                        source_detection_decision_counts = {
                            str(key): int(value)
                            for key, value in source_summary.items()
                            if value is not None
                        }
                except Exception:
                    source_detection_decision_counts = {}
        detect_review_status = None
        if crop_group is not None:
            detect_review_status = crop_group.attrs.get("detect_review_status")
        if not isinstance(detect_review_status, dict) and refined_group is not None:
            detect_review_status = refined_group.attrs.get("detect_review_status")
        detect_state = _review_state(detect_review_status)
        crop_state = _review_state(crop_group.attrs.get("crop_review_status")) if crop_group is not None else None

        if require_approved:
            if detect_state != "approved":
                skipped.append(f"{zarr_path.name}: detect review not approved")
                _log_timing(args.timing, f"{dataset_label}: skipped (detect review not approved)", dataset_started)
                continue
            if crop_group is not None and crop_state != "approved":
                skipped.append(f"{zarr_path.name}: crop review not approved")
                _log_timing(args.timing, f"{dataset_label}: skipped (crop review not approved)", dataset_started)
                continue

        if prefer_manual and args.source_type != "refined" and manual_group and crop_group is not None:
            if detection_source_type != "manual":
                skipped.append(
                    f"{zarr_path.name}: manual detections available but crop source is '{detection_source_type}'"
                )
                _log_timing(args.timing, f"{dataset_label}: skipped (manual preference)", dataset_started)
                continue

        if not args.allow_source_mismatch and crop_group is not None and args.source_type != "manual":
            if detection_source_type != args.source_type:
                raise ValueError(
                    f"{zarr_path.name}: crop source type is '{detection_source_type}', "
                    f"requested '{args.source_type}'. Re-run crop or pass --allow-source-mismatch."
                )

        if detection_source_type in {"filtered", "interpolated", "refined"} and detection_source_path:
            if detection_source_path not in root:
                raise ValueError(
                    f"{zarr_path.name}: detection_source_path '{detection_source_path}' not found in Zarr."
                )
        if args.source_type in {"filtered", "detect"} and includes_interpolated and not detection_source_present:
            raise ValueError(
                f"{zarr_path.name}: crop run includes interpolated ROIs but detection_source is missing."
            )

        phase_started = perf_counter()
        bboxes = np.asarray(root[bbox_array_path][:], dtype=np.float32)
        invalid_count, invalid_sample = _validate_bboxes(bboxes)
        _log_timing(args.timing, f"{dataset_label}: read+validate bbox_norm_coords", phase_started)

        dataset_name = _choose_dataset_name(seen_names, zarr_path, idx)
        dataset_entries[dataset_name] = {
            "zarr_path": str(zarr_path),
            "source_type": detection_source_type,
            "input_format": args.input_format,
        }

        phase_started = perf_counter()
        provenance = _extract_provenance(root, override, args.provenance_policy)
        if args.provenance_policy == "ignore":
            provenance.warnings = []
        _log_timing(args.timing, f"{dataset_label}: extract provenance", phase_started)

        dataset_id, session_uuid = _resolve_manifest_dataset_identity(
            root=root,
            zarr_path=zarr_path,
            registry=registry,
        )
        if registry is not None and should_register_datasets:
            phase_started = perf_counter()
            registered_dataset_id = registry.register_from_root(root, zarr_path)
            if registered_dataset_id:
                dataset_id = str(registered_dataset_id)
            _log_timing(args.timing, f"{dataset_label}: registry dataset upsert", phase_started)
        manifests.append(
            DatasetManifest(
                name=dataset_name,
                zarr_path=str(zarr_path),
                dataset_id=dataset_id,
                session_uuid=session_uuid,
                crop_run=crop_run,
                bbox_array_path=bbox_array_path,
                detection_source_type=detection_source_type,
                detection_source_path=detection_source_path,
                includes_interpolated=includes_interpolated,
                input_format=args.input_format,
                images_ds_shape=[height, width],
                total_bboxes=int(bboxes.shape[0]),
                manual_edited_bboxes=manual_edited_bboxes,
                source_detection_decision_counts=source_detection_decision_counts,
                invalid_bboxes=invalid_count,
                invalid_bbox_sample=invalid_sample,
                detection_source_counts=detection_source_counts,
                frame_indices_present=frame_indices_present,
                detection_source_present=detection_source_present,
                provenance=provenance,
            )
        )
        _log_timing(args.timing, f"{dataset_label}: total preflight", dataset_started)

    if skipped:
        print(f"\nSkipped {len(skipped)} datasets:")
        for reason in skipped:
            print(f"  - {reason}")
    if not manifests:
        raise ValueError("No datasets matched the requested filters (approved/manual).")

    imgsz = args.imgsz if args.imgsz is not None else int(reference_shape[0])
    if args.imgsz is None and reference_shape and reference_shape[0] != reference_shape[1]:
        imgsz = list(reference_shape)
    elif args.imgsz is not None:
        imgsz = int(args.imgsz)
    manifest_imgsz = (
        list(reference_shape)
        if reference_shape is not None
        else [int(imgsz), int(imgsz)]
    )
    if isinstance(imgsz, int):
        manifest_imgsz = [int(imgsz), int(imgsz)]
    elif isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
        manifest_imgsz = [int(imgsz[0]), int(imgsz[1])]

    base_config["datasets"] = dataset_entries
    base_config["task"] = "detect"
    if "training_params" not in base_config:
        base_config["training_params"] = {}
    base_config["training_params"]["imgsz"] = imgsz
    if args.project:
        base_config["training_params"]["project"] = args.project

    query_filter_payload = _build_query_filter_payload(
        args,
        require_approved=require_approved,
        prefer_manual=prefer_manual,
        set_name=resolved_set_name,
        set_version=set_version,
        selected_paths=args.zarr_paths,
    )

    if registry is not None and should_write_training_set and set_id:
        dataset_ids = [item.dataset_id for item in manifests if item.dataset_id]
        phase_started = perf_counter()
        registry.upsert_training_set(
            set_id=set_id,
            name=resolved_set_name,
            task_type="detect",
            query_filter=query_filter_payload,
            dataset_ids=dataset_ids,
            invocation=invocation_payload,
        )
        print(f"Recorded training set: {set_id}")
        _log_timing(args.timing, f"upsert training set ({set_id})", phase_started)

    if registry is not None:
        phase_started = perf_counter()
        registry.close()
        _log_timing(args.timing, "close registry", phase_started)
    manifest = TrainingManifest(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        task="detect",
        source_type=args.source_type,
        input_format=args.input_format,
        imgsz=manifest_imgsz,
        datasets=manifests,
        base_config_path=str(args.base_config),
        output_config_path=str(args.out_config) if args.out_config else None,
        output_manifest_path=str(args.out_manifest) if args.out_manifest else None,
        project=args.project,
        run_name=args.run_name,
        provenance_policy=args.provenance_policy,
        registry_path=registry_path,
        set_name=resolved_set_name,
        set_version=set_version,
        set_id=set_id,
        query_filter=query_filter_payload,
        invocation=invocation_payload,
    )

    _print_summary(manifest)

    phase_started = perf_counter()
    config_yaml = yaml.safe_dump(base_config, sort_keys=False)
    manifest_json = json.dumps(manifest.model_dump(exclude_none=True), indent=2)
    _log_timing(args.timing, "serialize config + manifest", phase_started)

    if args.dry_run:
        print("\n--- Generated Config (YAML) ---")
        print(config_yaml.strip())
        print("\n--- Training Manifest (JSON) ---")
        print(manifest_json)
        _log_timing(args.timing, "total runtime", run_started)
        return

    if args.out_config is None:
        raise ValueError("--out-config is required unless --dry-run is set.")
    out_manifest = args.out_manifest
    if out_manifest is None:
        out_manifest = args.out_config.with_suffix(".manifest.json")

    args.out_config.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    phase_started = perf_counter()
    args.out_config.write_text(config_yaml, encoding="utf-8")
    out_manifest.write_text(manifest_json, encoding="utf-8")
    _log_timing(args.timing, "write config + manifest", phase_started)

    print(f"\nWrote config: {args.out_config}")
    print(f"Wrote manifest: {out_manifest}")
    phase_started = perf_counter()
    created = _ensure_dummy_paths(base_config, args.out_config)
    for path in created:
        print(f"Created dummy dataset file: {path}")
    _log_timing(args.timing, "ensure dummy dataset paths", phase_started)
    if args.run_name:
        print(
            "Next: python -m fisheye.training.train_detection "
            f"{args.out_config} --run-name {args.run_name}"
        )
    else:
        print(f"Next: python -m fisheye.training.train_detection {args.out_config}")
    _log_timing(args.timing, "total runtime", run_started)


if __name__ == "__main__":  # pragma: no cover
    main()
