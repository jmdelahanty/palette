"""Inventory persisted coordinate contracts without opening or mutating Zarr.

The scanner deliberately has no apply mode.  Registry access uses SQLite's
``mode=ro`` URI together with ``PRAGMA query_only`` and Zarr inspection reads
only ``zarr.json`` / Zarr-v2 metadata files.  Array payloads, consolidated
metadata, registry models, and processing code are never opened by this module.

The JSONL output contains one ``coordinate_dataset`` record for every row in
``datasets`` (including rows whose path is missing) followed by zero or more
``coordinate_surface`` records.  That invariant makes the output suitable for
full-registry reconciliation rather than only reporting archives that happened
to be reachable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from fisheye.shared.coordinate_descriptor import validate_coordinate_descriptor


AUDIT_SCHEMA_ID = "palette.coordinate_contract_inventory"
AUDIT_SCHEMA_VERSION = 1

STATUSES = (
    "compatible",
    "compatible_via_explicit_legacy_rule",
    "metadata_backfill_candidate",
    "numerical_validation_required",
    "recompute_required",
    "ambiguous_fail_closed",
    "missing_or_unreadable",
)

# Worst status wins when surface results are rolled up to a registry row.
_STATUS_PRIORITY = {
    "compatible": 0,
    "compatible_via_explicit_legacy_rule": 1,
    "metadata_backfill_candidate": 2,
    "numerical_validation_required": 3,
    "ambiguous_fail_closed": 4,
    "recompute_required": 5,
    "missing_or_unreadable": 6,
}

_PIXEL_OR_NORMALIZED_SURFACES = {
    "track_positions_px",
    "refined_online_positions_px",
    "detect_bbox",
    "refined_detect_bbox",
    "crop_geometry",
    "keypoint_roi",
    "keypoint_source_image",
    "keypoint_normalized",
    "stimulus_chaser",
    "subject_shape_geometry",
    "subject_mask_geometry",
}

_DESCRIPTOR_ATTRS = (
    "coordinate_descriptor",
    "coordinate_contract",
)

_SPACE_KEYS = (
    "space_id",
    "coordinate_space_id",
    "coordinate_space",
    "coordinate_frame",
    "reference_space",
)
_UNITS_KEYS = ("units", "coordinate_units")
_ORIGIN_KEYS = ("origin", "coordinate_origin")
_X_AXIS_KEYS = ("positive_x_direction", "x_axis_direction", "x_direction")
_Y_AXIS_KEYS = ("positive_y_direction", "y_axis_direction", "y_direction")
_WIDTH_KEYS = (
    "reference_width",
    "coordinate_reference_width",
    "source_width",
    "image_width",
    "video_width",
    "texture_width",
    "canvas_width",
    "roi_width",
)
_HEIGHT_KEYS = (
    "reference_height",
    "coordinate_reference_height",
    "source_height",
    "image_height",
    "video_height",
    "texture_height",
    "canvas_height",
    "roi_height",
)
_PIXEL_CONVENTION_KEYS = (
    "pixel_convention",
    "pixel_center_convention",
    "pixel_coordinate_convention",
)
_GEOMETRY_CONVENTION_KEYS = (
    "geometry_convention",
    "coordinate_format",
    "bbox_format",
    "component_order",
)
_ROW_IDENTITY_KEYS = (
    "row_identity_ref",
    "row_identity",
    "row_axis",
    "frame_index_path",
    "frame_indices_path",
    "source_row_ids_path",
    "row_identity",
)
_SOURCE_REF_KEYS = (
    "source_ref",
    "source_coordinate_ref",
    "source_coordinate_descriptor_ref",
    "source_path",
    "position_source_path",
    "source_rowset_path",
    "source_crop_run",
    "source_detect_run",
    "source_keypoints_run",
    "source_keypoint_run",
    "lineage_refs",
)
_TRANSFORM_REF_KEYS = (
    "transform_ref",
    "coordinate_transform_ref",
    "transform_lineage_ref",
    "calibration_ref",
    "calibration_path",
    "transform_refs",
)
_TRANSFORM_DIRECTION_KEYS = (
    "transform_direction",
    "homography_direction",
    "coordinate_transform_direction",
)
_TRANSFORM_FROM_KEYS = ("from_space_id", "source_space_id", "transform_from_space")
_TRANSFORM_TO_KEYS = ("to_space_id", "target_space_id", "transform_to_space")
_OVERLAY_KEYS = (
    "source_camera_overlay_suitable",
    "suitable_for_source_camera_overlay",
    "camera_overlay_compatible",
    "source_camera_overlay",
)


@dataclass(frozen=True)
class MetadataNode:
    """One node discovered from an on-disk Zarr metadata file."""

    relative_path: str
    node_type: str | None
    metadata_format: str
    shape: Any
    data_type: Any
    attributes: dict[str, Any]
    metadata_error: str | None = None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        if isinstance(parsed, Mapping):
            return {str(key): item for key, item in parsed.items()}
    return {}


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def open_registry_readonly(registry_path: Path) -> sqlite3.Connection:
    """Open *registry_path* through SQLite's immutable read-only boundary.

    ``query_only`` is intentionally redundant with ``mode=ro``.  It protects
    callers if SQLite URI handling changes and is straightforward to assert in
    focused tests.
    """

    resolved = registry_path.expanduser().resolve()
    uri = f"{resolved.as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    except sqlite3.Error:
        return set()
    return {str(row[1]) for row in rows}


def read_registry_dataset_rows(registry_path: Path) -> list[dict[str, Any]]:
    """Return every ``datasets`` row in deterministic order."""

    conn = open_registry_readonly(registry_path)
    try:
        columns = _table_columns(conn, "datasets")
        if not columns:
            raise ValueError(f"registry has no datasets table: {registry_path}")
        try:
            raw_rows = conn.execute("SELECT rowid AS _registry_rowid, d.* FROM datasets d").fetchall()
        except sqlite3.Error:
            raw_rows = conn.execute("SELECT d.* FROM datasets d").fetchall()
    finally:
        conn.close()

    rows = [{str(key): _json_safe(row[key]) for key in row.keys()} for row in raw_rows]
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("dataset_id") or ""),
            str(row.get("zarr_path") or ""),
            str(row.get("_registry_rowid") or ""),
            _canonical_json(row),
        ),
    )


def _read_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # metadata corruption and filesystem errors are audit data
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, "metadata JSON is not an object"
    return payload, None


def _read_metadata_node(path: Path, relative_path: str) -> MetadataNode | None:
    zarr_json = path / "zarr.json"
    if zarr_json.is_file():
        payload, error = _read_json_object(zarr_json)
        payload = payload or {}
        attrs = payload.get("attributes")
        return MetadataNode(
            relative_path=relative_path,
            node_type=str(payload.get("node_type")) if payload.get("node_type") is not None else None,
            metadata_format="zarr.json",
            shape=payload.get("shape"),
            data_type=payload.get("data_type"),
            attributes=dict(attrs) if isinstance(attrs, Mapping) else {},
            metadata_error=error,
        )

    zgroup = path / ".zgroup"
    zarray = path / ".zarray"
    zattrs = path / ".zattrs"
    if zgroup.is_file() or zarray.is_file() or zattrs.is_file():
        attrs_payload: dict[str, Any] = {}
        errors: list[str] = []
        if zattrs.is_file():
            attrs, error = _read_json_object(zattrs)
            attrs_payload = attrs or {}
            if error:
                errors.append(f".zattrs: {error}")
        array_payload: dict[str, Any] = {}
        if zarray.is_file():
            array, error = _read_json_object(zarray)
            array_payload = array or {}
            if error:
                errors.append(f".zarray: {error}")
        return MetadataNode(
            relative_path=relative_path,
            node_type="array" if zarray.is_file() else "group",
            metadata_format="zarr_v2",
            shape=array_payload.get("shape"),
            data_type=array_payload.get("dtype"),
            attributes=attrs_payload,
            metadata_error="; ".join(errors) or None,
        )
    return None


def _root_metadata_fingerprint(zarr_path: Path) -> str | None:
    """Digest root metadata so resume cannot rely on registry identity alone."""

    digest = hashlib.sha256()
    found = False
    for name in (".zarray", ".zattrs", ".zgroup", "zarr.json"):
        path = zarr_path / name
        try:
            if not path.is_file():
                continue
            payload = path.read_bytes()
        except OSError:
            return None
        found = True
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest() if found else None


def _metadata_inventory_fingerprint(nodes: Sequence[MetadataNode]) -> str:
    """Digest every metadata node actually used for classification."""

    digest = hashlib.sha256()
    for node in sorted(nodes, key=lambda item: item.relative_path):
        payload = {
            "relative_path": node.relative_path,
            "node_type": node.node_type,
            "metadata_format": node.metadata_format,
            "shape": _json_safe(node.shape),
            "data_type": _json_safe(node.data_type),
            "attributes": _json_safe(node.attributes),
            "metadata_error": node.metadata_error,
        }
        digest.update(_canonical_json(payload).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def iter_metadata_nodes(zarr_path: Path) -> Iterable[MetadataNode]:
    """Yield nodes found by directory metadata only, never by a Zarr API."""

    stack: list[tuple[Path, str]] = [(zarr_path, ".")]
    while stack:
        path, relative_path = stack.pop()
        node = _read_metadata_node(path, relative_path)
        if node is None:
            continue
        yield node
        try:
            children = [child for child in path.iterdir() if child.is_dir() and not child.name.startswith(".")]
        except OSError:
            children = []
        discovered: list[tuple[Path, str]] = []
        for child in children:
            if not any((child / name).is_file() for name in ("zarr.json", ".zgroup", ".zarray", ".zattrs")):
                continue
            child_relative = child.name if relative_path == "." else f"{relative_path}/{child.name}"
            discovered.append((child, child_relative))
        # Reverse push order so iteration itself is lexical.
        stack.extend(reversed(sorted(discovered, key=lambda item: item[1])))


def _path_parts(relative_path: str) -> tuple[str, ...]:
    if relative_path in ("", "."):
        return ()
    return tuple(part.lower() for part in PurePosixPath(relative_path).parts)


def classify_surface(relative_path: str, node: MetadataNode) -> str | None:
    """Map an important persisted geometry node to a stable surface family."""

    parts = _path_parts(relative_path)
    if not parts:
        return None
    leaf = parts[-1]
    part_set = set(parts)

    if "track_kinematics_runs" in part_set and leaf == "positions_px":
        return "track_positions_px"
    if "track_kinematics_runs" in part_set and leaf == "positions_mm":
        return "track_positions_mm"
    if any(
        part in {
            "refined_online_runs",
            "refined_online_detect_runs",
            "refined_online_detection_runs",
        }
        for part in parts
    ):
        if leaf == "positions_px":
            return "refined_online_positions_px"
        if leaf == "positions_mm":
            return "refined_online_positions_mm"

    if leaf == "chaser_states" and "tracking_data" in part_set:
        return "stimulus_chaser"
    if (
        "tracking_data" in part_set
        and "chaser_states" in part_set
        and leaf
        in {
            "chaser_pos_x",
            "chaser_pos_y",
            "target_pos_x",
            "target_pos_y",
            "target_clamped_pos_x",
            "target_clamped_pos_y",
        }
    ):
        return "stimulus_chaser"

    if "homograph" in leaf and ("calibration" in part_set or "calibration_runs" in part_set):
        return "calibration_homography"
    if ("calibration" in part_set or "calibration_runs" in part_set) and any(
        "homograph" in str(key).lower() for key in node.attributes
    ):
        return "calibration_homography"

    bbox_like = (
        leaf.startswith("bbox")
        or leaf.startswith("bboxes")
        or leaf in {"boxes", "roi_bounds", "crop_bounds", "bounds_xyxy"}
    )
    if bbox_like and "refined_detect_runs" in part_set:
        return "refined_detect_bbox"
    if bbox_like and any(part in {"detect_runs", "detection_runs"} for part in parts):
        return "detect_bbox"

    if "crop_runs" in part_set:
        crop_index = parts.index("crop_runs")
        if len(parts) == crop_index + 2 and node.node_type == "group":
            return "crop_geometry"
        if bbox_like or leaf in {
            "roi_coordinates_full",
            "roi_coordinates_ds",
            "roi_centers",
            "crop_centers",
            "source_centers_px",
        }:
            return "crop_geometry"

    if any(part in {"keypoints_runs", "refined_keypoints_runs"} for part in parts):
        if leaf in {"keypoints_roi", "keypoint_roi"}:
            return "keypoint_roi"
        if leaf in {"keypoints_img", "keypoints_image", "keypoints_source_image"}:
            return "keypoint_source_image"
        if leaf in {"keypoints_norm", "keypoints_normalized"}:
            return "keypoint_normalized"

    subject_shape = any(part in {"subject_shape_runs", "subject_shapes_runs"} for part in parts)
    geometry_tokens = (
        "centerline",
        "spline",
        "control_point",
        "landmark",
        "tail",
        "tangent",
        "normal",
        "principal_axis",
        "origin",
        "forward_axis",
        "left_axis",
        "ellipse",
        "caudal",
    )
    geometry_like = (
        "contour" in leaf
        or "centroid" in leaf
        or bbox_like
        or "body_axis" in leaf
        or any(token in leaf for token in geometry_tokens)
    )
    if subject_shape and geometry_like:
        return "subject_shape_geometry"

    subject_mask = any(
        part in {
            "subject_mask_runs",
            "refined_subject_masks_runs",
            "eye_masks_runs",
            "refined_eye_masks_runs",
        }
        for part in parts
    )
    if subject_mask and (leaf in {"masks_roi", "mask_bitpacked", "mask_rle"} or geometry_like):
        return "subject_mask_geometry"

    # Keep plausible geometry arrays visible even when a new producer surface
    # has not yet been assigned a controlled family.  This is intentionally
    # conservative and limited to array nodes with geometry-specific names.
    generic_geometry_tokens = (
        "bbox",
        "bound",
        "centerline",
        "centroid",
        "contour",
        "control_point",
        "ellipse",
        "homograph",
        "keypoints_",
        "landmark",
        "roi_coordinates",
        "spline",
    )
    if node.node_type == "array" and any(token in leaf for token in generic_geometry_tokens):
        return "unclassified_geometry_candidate"
    return None


def _ancestor_paths(relative_path: str) -> list[str]:
    if relative_path in ("", "."):
        return ["."]
    path = PurePosixPath(relative_path)
    result: list[str] = []
    parent = path.parent
    while str(parent) not in ("", "."):
        result.append(parent.as_posix())
        parent = parent.parent
    result.append(".")
    return result


def _deep_find(mapping: Mapping[str, Any], keys: Sequence[str], *, prefix: str = "", depth: int = 0) -> tuple[Any, str] | None:
    if depth > 8:
        return None
    wanted = {str(key).lower() for key in keys}
    for key in sorted(mapping, key=lambda item: str(item)):
        value = mapping[key]
        key_text = str(key)
        location = f"{prefix}.{key_text}" if prefix else key_text
        if key_text.lower() in wanted and value not in (None, ""):
            return value, location
    for key in sorted(mapping, key=lambda item: str(item)):
        nested = _as_mapping(mapping[key])
        if not nested:
            continue
        key_text = str(key)
        location = f"{prefix}.{key_text}" if prefix else key_text
        found = _deep_find(nested, keys, prefix=location, depth=depth + 1)
        if found:
            return found
    return None


def _find_declared(
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
    keys: Sequence[str],
    *,
    include_nested: bool = True,
) -> tuple[Any, str] | None:
    paths = [node.relative_path, *_ancestor_paths(node.relative_path)]
    seen: set[str] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        candidate = nodes.get(path)
        if candidate is None:
            continue
        direct = _deep_find(candidate.attributes, keys) if include_nested else None
        if not include_nested:
            for key in keys:
                value = candidate.attributes.get(key)
                if value not in (None, ""):
                    direct = (value, key)
                    break
        if direct:
            value, attr_path = direct
            return value, f"{path}:{attr_path}"
    return None


def _surface_prefixed_keys(node: MetadataNode, keys: Sequence[str]) -> tuple[str, ...]:
    leaf = PurePosixPath(node.relative_path).name
    prefixed = [f"{leaf}_{key}" for key in keys]
    # Persisted bbox contracts commonly use a shorter stem.
    if leaf.startswith("bbox_norm"):
        prefixed.extend(f"bbox_norm_{key.removeprefix('coordinate_')}" for key in keys)
    if leaf.startswith("bbox_img"):
        prefixed.extend(f"bbox_img_xyxy_{key.removeprefix('coordinate_')}" for key in keys)
    return tuple(dict.fromkeys([*prefixed, *keys]))


def _find_descriptor(node: MetadataNode, nodes: Mapping[str, MetadataNode]) -> tuple[dict[str, Any], str, bool] | None:
    leaf = PurePosixPath(node.relative_path).name
    attrs = node.attributes
    for name in (f"{leaf}_coordinate_descriptor", *_DESCRIPTOR_ATTRS):
        descriptor = _as_mapping(attrs.get(name))
        if descriptor:
            return descriptor, f"{node.relative_path}:{name}", True
    descriptors = _as_mapping(attrs.get("coordinate_descriptors"))
    for key in (leaf, node.relative_path):
        descriptor = _as_mapping(descriptors.get(key))
        if descriptor:
            return descriptor, f"{node.relative_path}:coordinate_descriptors.{key}", True

    for ancestor in _ancestor_paths(node.relative_path):
        parent = nodes.get(ancestor)
        if parent is None:
            continue
        for name in (f"{leaf}_coordinate_descriptor", *_DESCRIPTOR_ATTRS):
            descriptor = _as_mapping(parent.attributes.get(name))
            if descriptor:
                return descriptor, f"{ancestor}:{name}", False
        descriptors = _as_mapping(parent.attributes.get("coordinate_descriptors"))
        for key in (leaf, node.relative_path):
            descriptor = _as_mapping(descriptors.get(key))
            if descriptor:
                return descriptor, f"{ancestor}:coordinate_descriptors.{key}", False
    return None


def _descriptor_value(descriptor: Mapping[str, Any], keys: Sequence[str]) -> Any:
    found = _deep_find(descriptor, keys)
    return found[0] if found else None


def _descriptor_extent(descriptor: Mapping[str, Any]) -> tuple[Any, Any]:
    extent = _as_mapping(descriptor.get("reference_extent"))
    width = _descriptor_value(extent, ("width", "reference_width"))
    height = _descriptor_value(extent, ("height", "reference_height"))
    if width is None:
        width = _descriptor_value(descriptor, _WIDTH_KEYS)
    if height is None:
        height = _descriptor_value(descriptor, _HEIGHT_KEYS)
    return width, height


def _issue(code: str, severity: str, message: str, **evidence: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"code": code, "severity": severity, "message": message}
    if evidence:
        result["evidence"] = _json_safe(evidence)
    return result


def _has_row_identity(node: MetadataNode, nodes: Mapping[str, MetadataNode]) -> tuple[bool, str | None]:
    declared = _find_declared(node, nodes, _ROW_IDENTITY_KEYS)
    if declared:
        return True, declared[1]
    parent = str(PurePosixPath(node.relative_path).parent)
    sibling_names = {
        PurePosixPath(path).name
        for path in nodes
        if str(PurePosixPath(path).parent) == parent
    }
    for name in (
        "frame_indices",
        "source_frame_ids",
        "source_row_ids",
        "refined_row_ids",
        "stimulus_frame_num",
        "detection_indices",
    ):
        if name in sibling_names:
            return True, f"{parent}/{name}" if parent != "." else name
    if PurePosixPath(node.relative_path).name == "chaser_states":
        fields = node.attributes.get("field_names")
        if isinstance(fields, list) and any(name in fields for name in ("stimulus_frame_num", "source_frame_id")):
            return True, f"{node.relative_path}:field_names"
    return False, None


def _surface_evidence(
    surface_type: str,
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None, bool]:
    descriptor_match = _find_descriptor(node, nodes)
    descriptor = descriptor_match[0] if descriptor_match else None
    descriptor_source = descriptor_match[1] if descriptor_match else None
    descriptor_is_array_specific = bool(descriptor_match and descriptor_match[2])

    evidence: dict[str, Any] = {}
    field_specs = {
        "space_id": _SPACE_KEYS,
        "units": _UNITS_KEYS,
        "origin": _ORIGIN_KEYS,
        "x_axis_direction": _X_AXIS_KEYS,
        "y_axis_direction": _Y_AXIS_KEYS,
        "reference_width": _WIDTH_KEYS,
        "reference_height": _HEIGHT_KEYS,
        "pixel_convention": _PIXEL_CONVENTION_KEYS,
        "geometry_convention": _GEOMETRY_CONVENTION_KEYS,
        "source_ref": _SOURCE_REF_KEYS,
        "transform_ref": _TRANSFORM_REF_KEYS,
        "transform_direction": _TRANSFORM_DIRECTION_KEYS,
        "transform_from_space": _TRANSFORM_FROM_KEYS,
        "transform_to_space": _TRANSFORM_TO_KEYS,
        "source_camera_overlay_suitable": _OVERLAY_KEYS,
    }
    for field, keys in field_specs.items():
        declared = _find_declared(node, nodes, _surface_prefixed_keys(node, keys))
        if declared:
            evidence[field] = {"value": _json_safe(declared[0]), "source": declared[1]}

    row_identity, row_identity_source = _has_row_identity(node, nodes)
    if row_identity:
        evidence["row_identity"] = {"value": True, "source": row_identity_source}

    if descriptor:
        descriptor_fields = {
            "space_id": ("space_id",),
            "origin": ("origin",),
            "pixel_convention": ("pixel_convention",),
            "geometry_convention": ("geometry_type",),
        }
        for field, keys in descriptor_fields.items():
            value = _descriptor_value(descriptor, keys)
            if value is not None:
                evidence[field] = {"value": _json_safe(value), "source": descriptor_source}
        component_units = descriptor.get("component_units")
        if isinstance(component_units, (list, tuple)) and component_units:
            distinct_units = tuple(dict.fromkeys(str(unit) for unit in component_units))
            evidence["units"] = {
                "value": distinct_units[0] if len(distinct_units) == 1 else list(distinct_units),
                "source": descriptor_source,
            }
        directions = _as_mapping(descriptor.get("positive_directions"))
        if directions.get("x") not in (None, ""):
            evidence["x_axis_direction"] = {
                "value": _json_safe(directions["x"]),
                "source": descriptor_source,
            }
        if directions.get("y") not in (None, ""):
            evidence["y_axis_direction"] = {
                "value": _json_safe(directions["y"]),
                "source": descriptor_source,
            }
        width, height = _descriptor_extent(descriptor)
        if width is not None:
            evidence["reference_width"] = {"value": _json_safe(width), "source": descriptor_source}
        if height is not None:
            evidence["reference_height"] = {"value": _json_safe(height), "source": descriptor_source}
        row_identity = _as_mapping(descriptor.get("row_identity"))
        if row_identity:
            evidence["row_identity"] = {
                "value": True,
                "source": f"{descriptor_source}:row_identity",
                "descriptor_value": _json_safe(row_identity),
            }
        if descriptor.get("source_camera_overlay") not in (None, ""):
            evidence["source_camera_overlay_suitable"] = {
                "value": _json_safe(descriptor["source_camera_overlay"]),
                "source": f"{descriptor_source}:source_camera_overlay",
            }
        lineage_refs = descriptor.get("lineage_refs")
        if isinstance(lineage_refs, (list, tuple)) and lineage_refs:
            evidence["source_ref"] = {
                "value": _json_safe(lineage_refs),
                "source": f"{descriptor_source}:lineage_refs",
            }
        transform_refs = descriptor.get("transform_refs")
        if isinstance(transform_refs, (list, tuple)) and transform_refs:
            evidence["transform_ref"] = {
                "value": _json_safe(transform_refs),
                "source": f"{descriptor_source}:transform_refs",
            }
    return evidence, descriptor, descriptor_source, descriptor_is_array_specific


def _value(evidence: Mapping[str, Any], field: str) -> Any:
    item = evidence.get(field)
    return item.get("value") if isinstance(item, Mapping) else None


def _is_direct_source(node: MetadataNode, evidence: Mapping[str, Any], field: str) -> bool:
    item = evidence.get(field)
    if not isinstance(item, Mapping):
        return False
    return str(item.get("source") or "").startswith(f"{node.relative_path}:")


def _legacy_online_mm_requires_recompute(
    surface_type: str,
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
    descriptor_is_array_specific: bool,
) -> tuple[bool, dict[str, Any]]:
    if surface_type not in {"track_positions_mm", "refined_online_positions_mm"}:
        return False, {}
    if descriptor_is_array_specific:
        return False, {}
    path_is_online = "online" in node.relative_path.lower()
    method = _find_declared(node, nodes, ("method", "stage"))
    method_is_online = bool(method and "online" in str(method[0]).lower())
    if not (path_is_online or method_is_online):
        return False, {}
    pixel_to_mm = _find_declared(node, nodes, ("pixel_to_mm", "calibration_used"))
    ppm = _find_declared(node, nodes, ("pixels_per_mm_projector", "pixels_per_mm"))
    if pixel_to_mm and ppm:
        try:
            same_declared_value = float(pixel_to_mm[0]) == float(ppm[0])
        except (TypeError, ValueError):
            same_declared_value = False
        if same_declared_value:
            return True, {
                "pixel_to_mm_source": pixel_to_mm[1],
                "pixels_per_mm_source": ppm[1],
                "declared_value": _json_safe(pixel_to_mm[0]),
            }
    # Historical online track writers are an explicit compatibility class.  We
    # do not infer correctness from a numerical range.
    if surface_type == "track_positions_mm":
        return True, {"online_evidence": method[1] if method_is_online and method else node.relative_path}
    return False, {}


def _offline_crop_reconstruction_requires_recompute(
    surface_type: str,
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
    descriptor_is_array_specific: bool,
) -> tuple[bool, dict[str, Any]]:
    if surface_type != "track_positions_px" or descriptor_is_array_specific:
        return False, {}
    position_kind = _find_declared(node, nodes, ("position_source_kind",))
    space = _find_declared(node, nodes, _SPACE_KEYS)
    if not position_kind or str(position_kind[0]) != "crop_rows":
        return False, {}
    if not space or str(space[0]).lower() not in {"camera", "source_camera", "source_camera_px"}:
        return False, {}
    return True, {
        "position_source_kind_source": position_kind[1],
        "declared_space_source": space[1],
    }


def classify_surface_contract(
    *,
    surface_type: str,
    node: MetadataNode,
    nodes: Mapping[str, MetadataNode],
) -> dict[str, Any]:
    """Classify a surface from declarations and linked metadata only."""

    evidence, descriptor, descriptor_source, descriptor_is_array_specific = _surface_evidence(
        surface_type, node, nodes
    )
    issues: list[dict[str, Any]] = []

    if surface_type == "unclassified_geometry_candidate":
        issues.append(
            _issue(
                "UNCLASSIFIED_GEOMETRY_CANDIDATE",
                "error",
                "A geometry-like persisted array is not assigned to a controlled audit surface family.",
                surface_path=node.relative_path,
            )
        )

    if node.metadata_error:
        issues.append(
            _issue(
                "INVALID_ZARR_METADATA",
                "error",
                "The surface metadata file could not be parsed completely.",
                error=node.metadata_error,
            )
        )
        return {
            "status": "ambiguous_fail_closed",
            "issues": issues,
            "evidence": evidence,
            "coordinate_descriptor": descriptor,
            "descriptor_source": descriptor_source,
            "descriptor_is_array_specific": descriptor_is_array_specific,
        }

    online_bad, online_bad_evidence = _legacy_online_mm_requires_recompute(
        surface_type, node, nodes, descriptor_is_array_specific
    )
    if surface_type == "unclassified_geometry_candidate":
        status = "ambiguous_fail_closed"
    elif online_bad:
        issues.append(
            _issue(
                "ONLINE_MM_CONVERSION_RECOMPUTATION_REQUIRED",
                "critical",
                "Historical online millimetre positions may have multiplied by pixels/mm; metadata cannot repair values.",
                **online_bad_evidence,
            )
        )

    offline_crop_bad, offline_crop_evidence = _offline_crop_reconstruction_requires_recompute(
        surface_type, node, nodes, descriptor_is_array_specific
    )
    if offline_crop_bad:
        issues.append(
            _issue(
                "OFFLINE_CROP_SOURCE_RECONSTRUCTION_NUMERICAL_VALIDATION_REQUIRED",
                "error",
                "Crop-row positions declared as camera pixels need targeted validation against exact source reference dimensions.",
                **offline_crop_evidence,
            )
        )

    space = _value(evidence, "space_id")
    units = _value(evidence, "units")
    width = _value(evidence, "reference_width")
    height = _value(evidence, "reference_height")
    origin = _value(evidence, "origin")
    x_axis = _value(evidence, "x_axis_direction")
    y_axis = _value(evidence, "y_axis_direction")
    row_identity = _value(evidence, "row_identity")
    source_ref = _value(evidence, "source_ref")
    transform_ref = _value(evidence, "transform_ref")
    overlay = _value(evidence, "source_camera_overlay_suitable")
    pixel_convention = _value(evidence, "pixel_convention")
    geometry_convention = _value(evidence, "geometry_convention")

    if descriptor is None:
        issues.append(
            _issue(
                "ARRAY_COORDINATE_DESCRIPTOR_MISSING",
                "warning",
                "No compact array-specific coordinate descriptor is persisted.",
            )
        )
    elif not descriptor_is_array_specific:
        issues.append(
            _issue(
                "COORDINATE_DESCRIPTOR_INHERITED",
                "warning",
                "The descriptor is inherited from an ancestor instead of attached to this surface.",
                descriptor_source=descriptor_source,
            )
        )

    descriptor_validation_issues: list[dict[str, str]] = []
    if descriptor is not None:
        descriptor_validation_issues = [
            {"code": issue.code, "path": issue.path, "message": issue.message}
            for issue in validate_coordinate_descriptor(descriptor)
        ]
        if descriptor_validation_issues:
            issues.append(
                _issue(
                    "COORDINATE_DESCRIPTOR_INVALID",
                    "error",
                    "The compact coordinate descriptor does not satisfy the canonical schema.",
                    validation_issues=descriptor_validation_issues,
                    descriptor_source=descriptor_source,
                )
            )

    if space in (None, ""):
        issues.append(_issue("COORDINATE_SPACE_MISSING", "error", "Coordinate space is not declared."))
    elif str(space).lower() in {"camera", "texture"}:
        issues.append(
            _issue(
                "LEGACY_SPACE_LABEL_REQUIRES_COMPATIBILITY_RULE",
                "warning",
                "The legacy camera/texture label needs an explicit compatibility mapping.",
                declared_space=space,
            )
        )
    if units in (None, ""):
        issues.append(_issue("COORDINATE_UNITS_MISSING", "error", "Coordinate units are not declared."))
    if origin in (None, "") or x_axis in (None, "") or y_axis in (None, ""):
        issues.append(
            _issue(
                "ORIGIN_OR_AXES_MISSING",
                "error",
                "Origin and positive X/Y directions are not fully declared.",
                origin=origin,
                x_axis_direction=x_axis,
                y_axis_direction=y_axis,
            )
        )
    if surface_type in _PIXEL_OR_NORMALIZED_SURFACES and (width in (None, "") or height in (None, "")):
        issues.append(
            _issue(
                "REFERENCE_EXTENT_MISSING",
                "error",
                "Pixel or normalized coordinates lack exact reference width/height.",
                reference_width=width,
                reference_height=height,
            )
        )
    if surface_type in _PIXEL_OR_NORMALIZED_SURFACES and pixel_convention in (None, ""):
        issues.append(
            _issue(
                "PIXEL_CONVENTION_MISSING",
                "error",
                "Pixel-center, pixel-edge, or continuous-coordinate convention is not declared.",
            )
        )
    if geometry_convention in (None, ""):
        issues.append(
            _issue(
                "GEOMETRY_CONVENTION_MISSING",
                "error",
                "Component order and geometry convention are not declared.",
            )
        )
    if not row_identity and surface_type != "calibration_homography":
        issues.append(_issue("ROW_IDENTITY_MISSING", "error", "Frame/row identity is not linked."))

    if surface_type == "calibration_homography":
        direction = _value(evidence, "transform_direction")
        from_space = _value(evidence, "transform_from_space")
        to_space = _value(evidence, "transform_to_space")
        if direction in (None, "") and (from_space in (None, "") or to_space in (None, "")):
            issues.append(
                _issue(
                    "HOMOGRAPHY_DIRECTION_MISSING",
                    "critical",
                    "Homography direction is not explicitly labelled; its historical name is not evidence.",
                )
            )
        if transform_ref in (None, "") and not descriptor_is_array_specific:
            issues.append(
                _issue(
                    "CALIBRATION_LINEAGE_MISSING",
                    "error",
                    "Homography calibration lineage is not linked from this surface.",
                )
            )
    else:
        if source_ref in (None, "") and transform_ref in (None, ""):
            issues.append(
                _issue(
                    "SOURCE_OR_TRANSFORM_LINEAGE_MISSING",
                    "error",
                    "The selected source or transform lineage is not linked.",
                )
            )
        if overlay in (None, ""):
            issues.append(
                _issue(
                    "SOURCE_CAMERA_OVERLAY_SUITABILITY_UNDECLARED",
                    "warning",
                    "Suitability for source-camera overlay is not declared.",
                )
            )

    if online_bad:
        status = "recompute_required"
    elif offline_crop_bad:
        status = "numerical_validation_required"
    elif surface_type == "calibration_homography" and any(
        issue["code"] == "HOMOGRAPHY_DIRECTION_MISSING" for issue in issues
    ):
        status = "ambiguous_fail_closed"
    elif descriptor_is_array_specific and not descriptor_validation_issues and not any(
        issue["severity"] in {"error", "critical"} for issue in issues
    ):
        status = "compatible"
    else:
        critical_missing = {
            "COORDINATE_SPACE_MISSING",
            "REFERENCE_EXTENT_MISSING",
            "ORIGIN_OR_AXES_MISSING",
            "SOURCE_OR_TRANSFORM_LINEAGE_MISSING",
        }
        issue_codes = {str(issue["code"]) for issue in issues}
        has_direct_legacy_core = all(
            _is_direct_source(node, evidence, field)
            for field in ("space_id", "units", "origin", "x_axis_direction", "y_axis_direction")
        )
        exact_link_available = source_ref not in (None, "") or transform_ref not in (None, "")
        if issue_codes & critical_missing and not exact_link_available:
            status = "ambiguous_fail_closed"
        elif descriptor is not None or exact_link_available or space not in (None, ""):
            status = "metadata_backfill_candidate"
        elif has_direct_legacy_core:
            status = "compatible_via_explicit_legacy_rule"
        else:
            status = "ambiguous_fail_closed"

        # A fully explicit direct legacy declaration remains readable under a
        # testable compatibility rule even though a canonical descriptor is
        # absent.  Reference extent and row identity must still be present.
        if (
            descriptor is None
            and has_direct_legacy_core
            and (surface_type not in _PIXEL_OR_NORMALIZED_SURFACES or (width is not None and height is not None))
            and (surface_type == "calibration_homography" or row_identity)
            and (surface_type not in _PIXEL_OR_NORMALIZED_SURFACES or pixel_convention not in (None, ""))
            and geometry_convention not in (None, "")
            and not any(issue["severity"] == "critical" for issue in issues)
        ):
            status = "compatible_via_explicit_legacy_rule"

        # These historical surfaces need value-level confirmation even when
        # their legacy declarations are sufficient to construct metadata.  The
        # scanner never samples array payloads, so it records that work rather
        # than guessing correctness from names or ranges.
        if (
            status in {
                "metadata_backfill_candidate",
                "compatible_via_explicit_legacy_rule",
            }
            and surface_type in {
                "refined_online_positions_px",
                "refined_online_positions_mm",
            }
        ):
            status = "numerical_validation_required"

    return {
        "status": status,
        "issues": issues,
        "evidence": evidence,
        "coordinate_descriptor": descriptor,
        "descriptor_source": descriptor_source,
        "descriptor_is_array_specific": descriptor_is_array_specific,
    }


def _dataset_key(row: Mapping[str, Any], ordinal: int) -> str:
    dataset_id = row.get("dataset_id")
    if dataset_id not in (None, ""):
        return str(dataset_id)
    registry_rowid = row.get("_registry_rowid")
    if registry_rowid not in (None, ""):
        return f"registry_rowid:{registry_rowid}"
    return f"row:{ordinal:08d}:{_fingerprint(row)[:16]}"


def _registry_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in sorted(row.items())}


def audit_dataset_row(
    row: Mapping[str, Any],
    *,
    ordinal: int = 0,
    _preloaded_nodes: Sequence[MetadataNode] | None = None,
) -> list[dict[str, Any]]:
    """Audit one registry dataset row and always return a dataset record."""

    registry = _registry_projection(row)
    key = _dataset_key(registry, ordinal)
    raw_path = registry.get("zarr_path")
    root_metadata_fingerprint = (
        _root_metadata_fingerprint(Path(str(raw_path)).expanduser())
        if raw_path not in (None, "")
        else None
    )
    base = {
        "audit_schema_id": AUDIT_SCHEMA_ID,
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "dataset_key": key,
        "dataset_id": registry.get("dataset_id"),
        "recording_id": registry.get("recording_id"),
        "zarr_path": raw_path,
        "registry_status": registry.get("status"),
        "zarr_use": registry.get("zarr_use"),
        "artifact_kind": registry.get("artifact_kind"),
        "registry": registry,
        "registry_fingerprint": _fingerprint(registry),
        "root_metadata_fingerprint": root_metadata_fingerprint,
        "metadata_inventory_fingerprint": None,
    }

    dataset_issues: list[dict[str, Any]] = []
    if raw_path in (None, ""):
        dataset_issues.append(
            _issue("DATASET_ZARR_PATH_MISSING", "critical", "Registry row has no zarr_path.")
        )
        return [
            {
                **base,
                "record_type": "coordinate_dataset",
                "status": "missing_or_unreadable",
                "issues": dataset_issues,
                "issue_codes": [issue["code"] for issue in dataset_issues],
                "surface_count": 0,
                "scan_complete": True,
            }
        ]

    zarr_path = Path(str(raw_path)).expanduser()
    registry_status = str(registry.get("status") or "").lower()
    if registry_status == "missing":
        dataset_issues.append(
            _issue("REGISTRY_STATUS_MISSING", "critical", "Registry marks this dataset missing.")
        )
    try:
        path_exists = zarr_path.exists()
        path_is_dir = zarr_path.is_dir() if path_exists else False
    except OSError as exc:
        path_exists = False
        path_is_dir = False
        dataset_issues.append(
            _issue("DATASET_PATH_STAT_FAILED", "critical", "Dataset path could not be inspected.", error=str(exc))
        )
    if not path_exists or not path_is_dir:
        dataset_issues.append(
            _issue("DATASET_PATH_UNREACHABLE", "critical", "Dataset path is missing or not a directory.")
        )
        return [
            {
                **base,
                "record_type": "coordinate_dataset",
                "status": "missing_or_unreadable",
                "issues": dataset_issues,
                "issue_codes": sorted({issue["code"] for issue in dataset_issues}),
                "surface_count": 0,
                "scan_complete": True,
            }
        ]

    nodes_list = (
        list(_preloaded_nodes)
        if _preloaded_nodes is not None
        else list(iter_metadata_nodes(zarr_path))
    )
    base["metadata_inventory_fingerprint"] = _metadata_inventory_fingerprint(nodes_list)
    nodes = {node.relative_path: node for node in nodes_list}
    if "." not in nodes:
        dataset_issues.append(
            _issue(
                "ZARR_ROOT_METADATA_MISSING",
                "critical",
                "Dataset directory has no root zarr.json/.zgroup/.zarray metadata.",
            )
        )
        return [
            {
                **base,
                "record_type": "coordinate_dataset",
                "status": "missing_or_unreadable",
                "issues": dataset_issues,
                "issue_codes": sorted({issue["code"] for issue in dataset_issues}),
                "surface_count": 0,
                "scan_complete": True,
            }
        ]

    surface_records: list[dict[str, Any]] = []
    for node in nodes_list:
        surface_type = classify_surface(node.relative_path, node)
        if surface_type is None:
            continue
        result = classify_surface_contract(surface_type=surface_type, node=node, nodes=nodes)
        surface_records.append(
            {
                **base,
                "record_type": "coordinate_surface",
                "surface_type": surface_type,
                "surface_path": node.relative_path,
                "node_type": node.node_type,
                "metadata_format": node.metadata_format,
                "shape": _json_safe(node.shape),
                "data_type": _json_safe(node.data_type),
                "status": result["status"],
                "issues": result["issues"],
                "issue_codes": sorted({str(issue["code"]) for issue in result["issues"]}),
                "evidence": result["evidence"],
                "coordinate_descriptor": _json_safe(result["coordinate_descriptor"]),
                "descriptor_source": result["descriptor_source"],
                "descriptor_is_array_specific": result["descriptor_is_array_specific"],
            }
        )
    surface_records.sort(key=lambda item: (str(item["surface_path"]), str(item["surface_type"])))

    # A second metadata-only walk closes the time-of-check/time-of-use gap.  A
    # mixed snapshot is never advertised as resumable or complete.
    post_scan_nodes = list(iter_metadata_nodes(zarr_path))
    post_scan_fingerprint = _metadata_inventory_fingerprint(post_scan_nodes)
    source_changed_during_scan = (
        post_scan_fingerprint != base["metadata_inventory_fingerprint"]
    )
    if source_changed_during_scan:
        dataset_issues.append(
            _issue(
                "SOURCE_CHANGED_DURING_SCAN",
                "critical",
                "Zarr metadata changed between the audit snapshot and post-scan verification.",
                initial_metadata_inventory_fingerprint=base["metadata_inventory_fingerprint"],
                post_scan_metadata_inventory_fingerprint=post_scan_fingerprint,
            )
        )
    for surface_record in surface_records:
        surface_record["scan_snapshot_valid"] = not source_changed_during_scan

    if not surface_records:
        dataset_issues.append(
            _issue(
                "NO_IMPORTANT_COORDINATE_SURFACES_DETECTED",
                "info",
                "No coordinate surface covered by this audit was found.",
            )
        )
    statuses = [str(record["status"]) for record in surface_records]
    dataset_status = max(statuses, key=lambda status: _STATUS_PRIORITY[status]) if statuses else "compatible"
    if registry_status == "missing":
        dataset_status = "missing_or_unreadable"
    if source_changed_during_scan:
        dataset_status = "missing_or_unreadable"
    dataset_record = {
        **base,
        "record_type": "coordinate_dataset",
        "status": dataset_status,
        "issues": dataset_issues,
        "issue_codes": sorted({issue["code"] for issue in dataset_issues}),
        "surface_count": len(surface_records),
        "metadata_node_count": len(nodes_list),
        "scan_complete": not source_changed_during_scan,
        "post_scan_metadata_inventory_fingerprint": post_scan_fingerprint,
    }
    return [dataset_record, *surface_records]


def _load_resume_records(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None or not path.is_file():
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(record, dict) or record.get("audit_schema_id") != AUDIT_SCHEMA_ID:
            continue
        key = record.get("dataset_key")
        if key not in (None, ""):
            grouped.setdefault(str(key), []).append(record)
    reusable: dict[str, list[dict[str, Any]]] = {}
    for key, records in grouped.items():
        dataset_records = [record for record in records if record.get("record_type") == "coordinate_dataset"]
        if len(dataset_records) == 1 and dataset_records[0].get("scan_complete") is True:
            reusable[key] = records
    return reusable


def audit_registry(registry_path: Path, *, resume_jsonl: Path | None = None) -> list[dict[str, Any]]:
    """Audit every registry dataset row; optionally reuse explicit prior rows."""

    rows = read_registry_dataset_rows(registry_path)
    resumed = _load_resume_records(resume_jsonl)
    records: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        key = _dataset_key(row, ordinal)
        registry_fingerprint = _fingerprint(_registry_projection(row))
        raw_path = row.get("zarr_path")
        root_metadata_fingerprint = (
            _root_metadata_fingerprint(Path(str(raw_path)).expanduser())
            if raw_path not in (None, "")
            else None
        )
        prior = resumed.get(key, [])
        prior_dataset = next(
            (record for record in prior if record.get("record_type") == "coordinate_dataset"), None
        )
        preloaded_nodes: list[MetadataNode] | None = None
        metadata_inventory_fingerprint: str | None = None
        if prior_dataset and raw_path not in (None, ""):
            candidate_path = Path(str(raw_path)).expanduser()
            try:
                candidate_is_dir = candidate_path.is_dir()
            except OSError:
                candidate_is_dir = False
            if candidate_is_dir:
                preloaded_nodes = list(iter_metadata_nodes(candidate_path))
                metadata_inventory_fingerprint = _metadata_inventory_fingerprint(preloaded_nodes)
        resume_matches = bool(
            prior_dataset
            and prior_dataset.get("registry_fingerprint") == registry_fingerprint
            and prior_dataset.get("root_metadata_fingerprint")
            == root_metadata_fingerprint
            and metadata_inventory_fingerprint is not None
            and prior_dataset.get("metadata_inventory_fingerprint")
            == metadata_inventory_fingerprint
        )
        if resume_matches:
            # Verify the prospective reuse snapshot just as strictly as a new
            # scan.  If it moved, fall through to a fresh classification using
            # the most recent complete metadata snapshot.
            candidate_path = Path(str(raw_path)).expanduser()
            verified_nodes = list(iter_metadata_nodes(candidate_path))
            verified_fingerprint = _metadata_inventory_fingerprint(verified_nodes)
            verified_root_fingerprint = _root_metadata_fingerprint(candidate_path)
            if (
                verified_fingerprint != metadata_inventory_fingerprint
                or verified_root_fingerprint != root_metadata_fingerprint
            ):
                resume_matches = False
                preloaded_nodes = verified_nodes
        if resume_matches:
            records.extend(
                sorted(
                    prior,
                    key=lambda record: (
                        0 if record.get("record_type") == "coordinate_dataset" else 1,
                        str(record.get("surface_path") or ""),
                    ),
                )
            )
            continue
        records.extend(
            audit_dataset_row(
                row,
                ordinal=ordinal,
                _preloaded_nodes=preloaded_nodes,
            )
        )
    return records


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    datasets = [record for record in records if record.get("record_type") == "coordinate_dataset"]
    surfaces = [record for record in records if record.get("record_type") == "coordinate_surface"]
    return {
        "audit_schema_id": AUDIT_SCHEMA_ID,
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "record_type": "coordinate_inventory_summary",
        "dataset_row_count": len(datasets),
        "distinct_recording_count": len(
            {str(record.get("recording_id")) for record in datasets if record.get("recording_id") not in (None, "")}
        ),
        "surface_count": len(surfaces),
        "dataset_status_counts": dict(sorted(Counter(str(record.get("status")) for record in datasets).items())),
        "surface_status_counts": dict(sorted(Counter(str(record.get("status")) for record in surfaces).items())),
        "surface_type_counts": dict(sorted(Counter(str(record.get("surface_type")) for record in surfaces).items())),
        "issue_code_counts": dict(
            sorted(
                Counter(
                    str(code)
                    for record in records
                    for code in (record.get("issue_codes") or [])
                ).items()
            )
        ),
    }


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(_json_safe(record), sort_keys=True) + "\n" for record in records)
    path.write_text(text, encoding="utf-8")


_CSV_COLUMNS = (
    "record_type",
    "dataset_key",
    "dataset_id",
    "recording_id",
    "zarr_path",
    "registry_status",
    "zarr_use",
    "artifact_kind",
    "surface_type",
    "surface_path",
    "node_type",
    "metadata_format",
    "shape",
    "data_type",
    "status",
    "issue_codes",
    "descriptor_source",
    "descriptor_is_array_specific",
    "evidence",
    "coordinate_descriptor",
)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (Mapping, list, tuple, set)):
        return _canonical_json(value)
    return value


def write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_CSV_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow({column: _csv_value(record.get(column)) for column in _CSV_COLUMNS})
    path.write_text(stream.getvalue(), encoding="utf-8")


def _markdown_escape(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def write_markdown(path: Path, records: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        record
        for record in records
        if record.get("record_type") == "coordinate_surface"
        or record.get("status") == "missing_or_unreadable"
    ]
    lines = [
        "# Coordinate contract inventory",
        "",
        f"- Registry dataset rows: {summary.get('dataset_row_count', 0)}",
        f"- Distinct recordings: {summary.get('distinct_recording_count', 0)}",
        f"- Important geometry surfaces: {summary.get('surface_count', 0)}",
        "",
        "## Status counts",
        "",
        "```json",
        json.dumps(_json_safe(summary.get("surface_status_counts", {})), indent=2, sort_keys=True),
        "```",
        "",
        "## Inventory",
        "",
        "| Dataset | Recording | Surface | Path | Status | Issues |",
        "|---|---|---|---|---|---|",
    ]
    for record in rows:
        lines.append(
            "| "
            + " | ".join(
                _markdown_escape(value)
                for value in (
                    record.get("dataset_id") or record.get("dataset_key"),
                    record.get("recording_id"),
                    record.get("surface_type") or "dataset",
                    record.get("surface_path") or record.get("zarr_path"),
                    record.get("status"),
                    ", ".join(str(code) for code in (record.get("issue_codes") or [])),
                )
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only registry-wide audit of persisted coordinate contracts."
    )
    parser.add_argument("--registry", type=Path, required=True, help="Palette SQLite registry (opened mode=ro).")
    parser.add_argument("--output-jsonl", type=Path, help="Deterministic detailed inventory JSONL.")
    parser.add_argument("--output-csv", type=Path, help="Deterministic flattened inventory CSV.")
    parser.add_argument("--output-markdown", type=Path, help="Deterministic human-readable report.")
    parser.add_argument("--summary-json", type=Path, help="Deterministic summary JSON.")
    parser.add_argument(
        "--resume-jsonl",
        type=Path,
        help="Reuse complete rows whose registry fingerprint matches this prior JSONL (for immutable archives).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    records = audit_registry(args.registry, resume_jsonl=args.resume_jsonl)
    summary = summarize(records)
    if args.output_jsonl:
        write_jsonl(args.output_jsonl, records)
    else:
        for record in records:
            print(json.dumps(_json_safe(record), sort_keys=True))
    if args.output_csv:
        write_csv(args.output_csv, records)
    if args.output_markdown:
        write_markdown(args.output_markdown, records, summary)
    if args.summary_json:
        write_summary(args.summary_json, summary)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
