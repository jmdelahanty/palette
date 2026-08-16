"""Packaged heuristic profile loaders for method-specific pose policy."""

from __future__ import annotations

from collections.abc import Mapping
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fisheye.shared.runtime_config import runtime_config_dirs


@dataclass(frozen=True)
class LabelRequirements:
    required_labels: tuple[str, ...]
    core_assignment_labels: tuple[str, ...]


@dataclass(frozen=True)
class BlobAssignmentHeuristic:
    family: str
    bladder_vertex_rule: Optional[str] = None
    left_right_rule: Optional[str] = None
    fallback_rule: Optional[str] = None


@dataclass(frozen=True)
class GeometryQcHeuristic:
    min_triangle_angle_deg: Optional[float] = None
    max_triangle_angle_deg: Optional[float] = None
    min_triangle_area_px: Optional[float] = None
    max_triangle_area_px: Optional[float] = None


@dataclass(frozen=True)
class HeadingQcHeuristic:
    temporal_outlier_threshold_deg: Optional[float] = None
    max_frame_gap: Optional[int] = None


@dataclass(frozen=True)
class FlipDetectionHeuristic:
    family: str


@dataclass(frozen=True)
class PoseHeuristicProfile:
    profile_name: str
    profile_version: str
    method: str
    source_pose_schema: str
    skeleton_id: str
    label_requirements: Optional[LabelRequirements]
    blob_assignment: Optional[BlobAssignmentHeuristic]
    geometry_qc: Optional[GeometryQcHeuristic]
    heading_qc: Optional[HeadingQcHeuristic]
    flip_detection: Optional[FlipDetectionHeuristic]
    notes: Any = None


def _coerce_mapping(value: object) -> Optional[dict[str, object]]:
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _require_text(raw: dict[str, object], key: str) -> str:
    value = str(raw.get(key, "")).strip()
    if not value:
        raise ValueError(f"Heuristic profile missing required field '{key}'.")
    return value


def _coerce_str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    labels: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            labels.append(text)
    return tuple(labels)


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _label_requirements_from_raw(raw: object) -> Optional[LabelRequirements]:
    if not isinstance(raw, dict):
        return None
    return LabelRequirements(
        required_labels=_coerce_str_tuple(raw.get("required_labels")),
        core_assignment_labels=_coerce_str_tuple(raw.get("core_assignment_labels")),
    )


def _blob_assignment_from_raw(raw: object) -> Optional[BlobAssignmentHeuristic]:
    if not isinstance(raw, dict):
        return None
    family = str(raw.get("family", "")).strip()
    if not family:
        return None
    return BlobAssignmentHeuristic(
        family=family,
        bladder_vertex_rule=str(raw.get("bladder_vertex_rule", "")).strip() or None,
        left_right_rule=str(raw.get("left_right_rule", "")).strip() or None,
        fallback_rule=str(raw.get("fallback_rule", "")).strip() or None,
    )


def _geometry_qc_from_raw(raw: object) -> Optional[GeometryQcHeuristic]:
    if not isinstance(raw, dict):
        return None
    return GeometryQcHeuristic(
        min_triangle_angle_deg=_coerce_float(raw.get("min_triangle_angle_deg")),
        max_triangle_angle_deg=_coerce_float(raw.get("max_triangle_angle_deg")),
        min_triangle_area_px=_coerce_float(raw.get("min_triangle_area_px")),
        max_triangle_area_px=_coerce_float(raw.get("max_triangle_area_px")),
    )


def _heading_qc_from_raw(raw: object) -> Optional[HeadingQcHeuristic]:
    if not isinstance(raw, dict):
        return None
    return HeadingQcHeuristic(
        temporal_outlier_threshold_deg=_coerce_float(
            raw.get("temporal_outlier_threshold_deg")
        ),
        max_frame_gap=_coerce_int(raw.get("max_frame_gap")),
    )


def _flip_detection_from_raw(raw: object) -> Optional[FlipDetectionHeuristic]:
    if not isinstance(raw, dict):
        return None
    family = str(raw.get("family", "")).strip()
    if not family:
        return None
    return FlipDetectionHeuristic(family=family)


def load_heuristic_profile(profile_path: Path) -> PoseHeuristicProfile:
    with open(profile_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Heuristic profile must decode to an object: {profile_path}")

    return PoseHeuristicProfile(
        profile_name=_require_text(raw, "profile_name"),
        profile_version=_require_text(raw, "profile_version"),
        method=_require_text(raw, "method"),
        source_pose_schema=_require_text(raw, "source_pose_schema"),
        skeleton_id=_require_text(raw, "skeleton_id"),
        label_requirements=_label_requirements_from_raw(raw.get("label_requirements")),
        blob_assignment=_blob_assignment_from_raw(raw.get("blob_assignment")),
        geometry_qc=_geometry_qc_from_raw(raw.get("geometry_qc")),
        heading_qc=_heading_qc_from_raw(raw.get("heading_qc")),
        flip_detection=_flip_detection_from_raw(raw.get("flip_detection")),
        notes=raw.get("notes"),
    )


def heuristic_profile_from_package(
    method: str,
    pose_schema_name: str,
    base_dir: Optional[Path] = None,
) -> PoseHeuristicProfile:
    search_dirs: list[Path] = []
    if base_dir is not None:
        search_dirs.append(Path(base_dir))
    search_dirs.extend(runtime_config_dirs("pose_heuristics"))

    tried_paths: list[Path] = []
    for directory in search_dirs:
        profile_path = directory / method / f"{pose_schema_name}.json"
        tried_paths.append(profile_path)
        if profile_path.exists():
            return load_heuristic_profile(profile_path)

    tried_str = ", ".join(str(path) for path in tried_paths)
    raise FileNotFoundError(
        f"Heuristic profile '{method}/{pose_schema_name}' not found. Tried: {tried_str}"
    )


def pose_schema_name_from_value(pose_schema: object) -> Optional[str]:
    pose_schema_map = _coerce_mapping(pose_schema)
    if pose_schema_map is None:
        return None
    name = str(pose_schema_map.get("name", "")).strip()
    return name or None


def pose_schema_name_from_attrs(attrs: Mapping[str, object]) -> Optional[str]:
    return pose_schema_name_from_value(attrs.get("pose_schema"))


def maybe_heuristic_profile_from_attrs(
    method: str,
    attrs: Mapping[str, object],
    *,
    base_dir: Optional[Path] = None,
) -> Optional[PoseHeuristicProfile]:
    pose_schema_name = pose_schema_name_from_attrs(attrs)
    if not pose_schema_name:
        return None
    try:
        return heuristic_profile_from_package(method, pose_schema_name, base_dir=base_dir)
    except FileNotFoundError:
        return None


def require_blob_assignment(
    profile: PoseHeuristicProfile,
    *,
    family: Optional[str] = None,
) -> BlobAssignmentHeuristic:
    blob = profile.blob_assignment
    if blob is None:
        raise RuntimeError(
            f"Heuristic profile '{profile.profile_name}' is missing 'blob_assignment'."
        )
    if family is not None and blob.family != family:
        raise RuntimeError(
            f"Heuristic profile '{profile.profile_name}' has unsupported "
            f"blob_assignment.family='{blob.family}' (expected '{family}')."
        )
    return blob


def require_geometry_qc(profile: PoseHeuristicProfile) -> GeometryQcHeuristic:
    geometry = profile.geometry_qc
    if geometry is None:
        raise RuntimeError(
            f"Heuristic profile '{profile.profile_name}' is missing 'geometry_qc'."
        )
    return geometry


def require_heading_qc(profile: PoseHeuristicProfile) -> HeadingQcHeuristic:
    heading = profile.heading_qc
    if heading is None:
        raise RuntimeError(
            f"Heuristic profile '{profile.profile_name}' is missing 'heading_qc'."
        )
    return heading


def require_flip_detection(
    profile: PoseHeuristicProfile,
    *,
    family: Optional[str] = None,
) -> FlipDetectionHeuristic:
    flip = profile.flip_detection
    if flip is None:
        raise RuntimeError(
            f"Heuristic profile '{profile.profile_name}' is missing 'flip_detection'."
        )
    if family is not None and flip.family != family:
        raise RuntimeError(
            f"Heuristic profile '{profile.profile_name}' has unsupported "
            f"flip_detection.family='{flip.family}' (expected '{family}')."
        )
    return flip


def maybe_geometry_qc_from_attrs(
    method: str,
    attrs: Mapping[str, object],
    *,
    base_dir: Optional[Path] = None,
) -> Optional[GeometryQcHeuristic]:
    profile = maybe_heuristic_profile_from_attrs(method, attrs, base_dir=base_dir)
    if profile is None:
        return None
    return require_geometry_qc(profile)


def maybe_flip_detection_from_attrs(
    method: str,
    attrs: Mapping[str, object],
    *,
    family: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> Optional[FlipDetectionHeuristic]:
    profile = maybe_heuristic_profile_from_attrs(method, attrs, base_dir=base_dir)
    if profile is None:
        return None
    return require_flip_detection(profile, family=family)
