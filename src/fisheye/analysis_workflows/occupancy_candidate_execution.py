"""Exact external-authority binding for occupancy storage candidates.

The persisted occupancy arrays are not a sufficient source identity.  Their
scientific meaning also depends on an external detection table, the exact
coordinate representation selected from that table, the recording time axis,
and (for epoch occupancy) one immutable stimulus-window run.  This module
reconstructs those dependencies from the live archive and produces one closed
identity that both the parent invocation and fresh child must agree on.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis.detection_occupancy_runs import IMAGE_QUADRANTS_ZONE_SET_ID
from fisheye.analysis_workflows.analysis_candidate_execution import (
    CoordinateEvidenceStatus,
)
from fisheye.shared.json_safety import decode_null_terminated_text
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)

OCCUPANCY_SOURCE_IDENTITY_SCHEMA_ID = "palette.occupancy_candidate_source_identity"
OCCUPANCY_SOURCE_IDENTITY_SCHEMA_VERSION = 1
OCCUPANCY_INVOCATION_CONTRACT_ID = "occupancy_v1"

_OCCUPANCY_STAGES = frozenset({"detection_occupancy", "session_occupancy"})
_DETECTION_ARRAY_CANDIDATES = (
    ("bbox_img_xyxy", "source_image_pixels_xyxy", "xyxy_midpoint_float64_v1"),
    (
        "bbox_norm_coords",
        "source_camera_normalized_cxcywh",
        "normalized_center_scaled_by_resolved_image_dimensions_float64_v1",
    ),
)
_SCORE_ARRAY_CANDIDATES = ("confidence_scores", "scores")
_STIMULUS_WINDOW_ARRAYS = (
    "window_id",
    "label_bytes",
    "start_frame",
    "end_frame",
    "start_time_s",
    "end_time_s",
    "duration_s",
)


def _strict_json_copy(value: object) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("occupancy authority must be strict JSON") from exc


def _attrs(group: Any) -> dict[str, Any]:
    value = group.attrs
    return dict(value.asdict() if hasattr(value, "asdict") else dict(value))


def _group_at_path(root: Any, path: str) -> Any:
    node = root
    for component in path.split("/"):
        node = node[component]
    return node


def _canonical_relative_path(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value.startswith("/") or "\\" in value:
        raise ValueError(f"{label} is not one canonical relative Zarr path")
    parts = value.split("/")
    if any(part in {"", ".", "..", "latest", "latest_complete"} for part in parts):
        raise ValueError(f"{label} is not one immutable canonical Zarr path")
    return value


def _array_values_sha256(array: Any) -> str:
    digest = hashlib.sha256()
    if int(array.ndim) == 0:
        digest.update(np.ascontiguousarray(array[...]).tobytes(order="C"))
        return digest.hexdigest()
    rows = int(array.shape[0])
    block_rows = max(1, min(rows or 1, 65_536))
    for start in range(0, rows, block_rows):
        values = np.ascontiguousarray(array[start : start + block_rows])
        if values.dtype.hasobject:
            raise ValueError("occupancy authority arrays cannot contain objects")
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _coordinate_descriptor_sha256(array: Any) -> str | None:
    attrs = _attrs(array)
    descriptor = attrs.get("coordinate_descriptor")
    if descriptor is None:
        return None
    if not isinstance(descriptor, Mapping):
        raise ValueError("detection coordinate_descriptor is not one JSON object")
    return canonical_json_sha256(_strict_json_copy(descriptor))


def _array_record(group: Any, path: str, *, coordinate: bool = False) -> dict[str, Any]:
    array = _group_at_path(group, path)
    dtype = np.dtype(array.dtype)
    if dtype.hasobject:
        raise ValueError(f"occupancy authority array {path!r} has object dtype")
    return {
        "path": path,
        "dtype": dtype.str,
        "shape": [int(value) for value in array.shape],
        "values_sha256": _array_values_sha256(array),
        "coordinate_descriptor_sha256": (
            _coordinate_descriptor_sha256(array) if coordinate else None
        ),
    }


def _first_numeric(
    attrs: Mapping[str, Any],
    names: Sequence[str],
    *,
    cast: type[int] | type[float],
) -> int | float | None:
    for name in names:
        value = attrs.get(name)
        if value is None:
            continue
        try:
            parsed = cast(value)
        except (TypeError, ValueError):
            continue
        if parsed:
            return parsed
    return None


def _resolved_dimensions(root: Any, detection: Any) -> dict[str, Any]:
    root_attrs = _attrs(root)
    detection_attrs = _attrs(detection)
    width = (
        _first_numeric(
            detection_attrs,
            ("source_full_width", "source_video_width", "width", "video_width"),
            cast=int,
        )
        or _first_numeric(
            root_attrs,
            ("width", "video_width", "source_video_width", "palette_video_width"),
            cast=int,
        )
        or 4512
    )
    height = (
        _first_numeric(
            detection_attrs,
            ("source_full_height", "source_video_height", "height", "video_height"),
            cast=int,
        )
        or _first_numeric(
            root_attrs,
            ("height", "video_height", "source_video_height", "palette_video_height"),
            cast=int,
        )
        or 4512
    )
    fps = (
        _first_numeric(root_attrs, ("fps", "video_fps"), cast=float)
        or _first_numeric(detection_attrs, ("fps", "video_fps"), cast=float)
        or 30.0
    )
    total_frames = (
        _first_numeric(
            root_attrs,
            ("total_frames", "n_frames", "source_video_total_frames"),
            cast=int,
        )
        or _first_numeric(detection_attrs, ("total_frames", "n_frames"), cast=int)
        or 0
    )
    result = {
        "width_px": int(width),
        "height_px": int(height),
        "fps": float(fps),
        "total_frames": int(total_frames),
        "resolver_id": "occupancy_writer_dimension_precedence_v1",
    }
    if (
        result["width_px"] <= 0
        or result["height_px"] <= 0
        or not math.isfinite(result["fps"])
        or result["fps"] <= 0.0
        or result["total_frames"] <= 0
    ):
        raise ValueError("occupancy source dimensions are not positive and finite")
    return result


def _require_source_refs(attrs: Mapping[str, Any], *, stage_id: str) -> dict[str, Any]:
    refs = attrs.get("source_refs")
    base = {
        "source_detection_path",
        "source_detection_kind",
        "source_segment_kind",
        "source_segment_id",
        "source_segment_path",
    }
    expected = (
        base | {"source_stimulus_epoch_run", "source_stimulus_epoch_path"}
        if stage_id == "detection_occupancy"
        else base
    )
    if not isinstance(refs, Mapping) or set(refs) != expected:
        raise ValueError("occupancy source_refs field set differs")
    record = _strict_json_copy(refs)
    for field in base:
        if attrs.get(field) != record[field]:
            raise ValueError(f"occupancy top-level {field} differs from source_refs")
    if stage_id == "detection_occupancy":
        for field in ("source_stimulus_epoch_run", "source_stimulus_epoch_path"):
            if attrs.get(field) != record[field]:
                raise ValueError(
                    f"occupancy top-level {field} differs from source_refs"
                )
    return record


def _scientific_parameters(attrs: Mapping[str, Any]) -> dict[str, Any]:
    parameters = attrs.get("parameters")
    expected = {
        "bin_size",
        "smooth_sigma",
        "min_score",
        "spatial_occupancy_zone_sets",
    }
    if not isinstance(parameters, Mapping) or set(parameters) != expected:
        raise ValueError("occupancy scientific parameter field set differs")
    result = _strict_json_copy(parameters)
    if type(result["bin_size"]) is not int or result["bin_size"] <= 0:
        raise ValueError("occupancy bin_size must be one positive integer")
    sigma = result["smooth_sigma"]
    if type(sigma) not in {int, float} or not math.isfinite(float(sigma)) or sigma < 0:
        raise ValueError("occupancy smooth_sigma must be finite and nonnegative")
    score = result["min_score"]
    if score is not None and (
        type(score) not in {int, float} or not math.isfinite(float(score))
    ):
        raise ValueError("occupancy min_score must be null or finite")
    if result["spatial_occupancy_zone_sets"] != [IMAGE_QUADRANTS_ZONE_SET_ID]:
        raise ValueError("occupancy zone-set authority differs")
    return result


def _detection_authority(
    root: Any,
    *,
    refs: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    path = _canonical_relative_path(
        refs["source_detection_path"], label="source_detection_path"
    )
    kind = refs["source_detection_kind"]
    if type(kind) is not str or not kind:
        raise ValueError("source_detection_kind must be one nonempty string")
    try:
        group = _group_at_path(root, path)
    except (KeyError, TypeError) as exc:
        raise ValueError("occupancy detection source does not resolve") from exc
    if "frame_indices" not in group:
        raise ValueError("occupancy detection source lacks frame_indices")
    coordinate_name = None
    coordinate_space = None
    conversion = None
    for name, space, rule in _DETECTION_ARRAY_CANDIDATES:
        if name in group:
            coordinate_name = name
            coordinate_space = space
            conversion = rule
            break
    if coordinate_name is None:
        raise ValueError("occupancy detection source lacks a supported bbox array")
    score_name = next((name for name in _SCORE_ARRAY_CANDIDATES if name in group), None)
    frames = group["frame_indices"]
    coordinates = group[coordinate_name]
    if (
        len(frames.shape) != 1
        or len(coordinates.shape) != 2
        or int(coordinates.shape[1]) != 4
    ):
        raise ValueError("occupancy detection lineage or bbox shape differs")
    if int(frames.shape[0]) != int(coordinates.shape[0]):
        raise ValueError("occupancy frame_indices and bbox row counts differ")
    if score_name is not None:
        scores = group[score_name]
        if len(scores.shape) != 1 or int(scores.shape[0]) != int(frames.shape[0]):
            raise ValueError("occupancy score row count differs")
    elif parameters["min_score"] is not None:
        raise ValueError("occupancy min_score cannot bind a source without scores")
    dimensions = _resolved_dimensions(root, group)
    records = [
        _array_record(group, "frame_indices"),
        _array_record(group, coordinate_name, coordinate=True),
    ]
    if score_name is not None:
        records.append(_array_record(group, score_name))
    records.sort(key=lambda record: str(record["path"]))
    return {
        "schema_id": "palette.occupancy_detection_source_authority",
        "schema_version": 1,
        "source_detection_path": path,
        "source_detection_kind": kind,
        "coordinate_representation": coordinate_space,
        "center_conversion_rule": conversion,
        "selected_score_path": score_name,
        "dimensions": dimensions,
        "arrays": records,
    }


def _decode_labels(array: Any) -> list[str]:
    values = np.asarray(array[:])
    if values.ndim == 1:
        rows = values.reshape(-1)
    elif values.ndim == 2:
        rows = values
    else:
        raise ValueError("occupancy label_bytes must be rank one or two")
    return [decode_null_terminated_text(row) for row in rows]


def _require_array_equal(left: Any, right: Any, *, label: str) -> None:
    if not np.array_equal(np.asarray(left[:]), np.asarray(right[:])):
        raise ValueError(f"occupancy {label} differs from its bound source")


def _stimulus_authority(
    root: Any,
    run_group: Any,
    *,
    refs: Mapping[str, Any],
) -> dict[str, Any]:
    path = _canonical_relative_path(
        refs["source_stimulus_epoch_path"],
        label="source_stimulus_epoch_path",
    )
    if refs["source_segment_path"] != path:
        raise ValueError("occupancy segment and stimulus paths differ")
    if refs["source_segment_kind"] != "stimulus_epoch":
        raise ValueError("detection occupancy segment kind differs")
    run_name = refs["source_stimulus_epoch_run"]
    if (
        type(run_name) is not str
        or not run_name
        or refs["source_segment_id"] != run_name
    ):
        raise ValueError("detection occupancy stimulus run identity differs")
    if path != f"analysis/stimulus_epoch_runs/{run_name}":
        raise ValueError("detection occupancy stimulus path is not canonical")
    try:
        group = _group_at_path(root, path)
        windows = group["windows"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "occupancy stimulus-window authority does not resolve"
        ) from exc
    group_attrs = _attrs(group)
    if (
        group_attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or group_attrs.get("stage_selector_eligible") is not True
    ):
        raise ValueError(
            "occupancy stimulus-window authority is not complete and eligible"
        )
    missing = [name for name in _STIMULUS_WINDOW_ARRAYS if name not in windows]
    if missing:
        raise ValueError(f"occupancy stimulus-window authority lacks {missing!r}")
    output = run_group["windows"]
    mapping = {
        "window_id": "window_id",
        "start_frame": "start_frame",
        "end_frame": "end_frame",
        "start_time_s": "start_time_s",
        "end_time_s": "end_time_s",
        "duration_s": "duration_s",
    }
    for output_name, source_name in mapping.items():
        _require_array_equal(
            output[output_name], windows[source_name], label=f"window {output_name}"
        )
    _require_array_equal(
        output["source_stimulus_epoch_window_id"],
        windows["window_id"],
        label="source stimulus window IDs",
    )
    _require_array_equal(
        output["source_segment_id"],
        windows["window_id"],
        label="source segment IDs",
    )
    if _decode_labels(output["label_bytes"]) != _decode_labels(windows["label_bytes"]):
        raise ValueError("occupancy window labels differ from stimulus authority")
    records = [_array_record(windows, name) for name in _STIMULUS_WINDOW_ARRAYS]
    records.sort(key=lambda record: str(record["path"]))
    return {
        "schema_id": "palette.occupancy_stimulus_window_authority",
        "schema_version": 1,
        "source_stimulus_epoch_path": path,
        "source_stimulus_epoch_run": run_name,
        "source_schema_id": group_attrs.get("schema_id"),
        "source_schema_version": group_attrs.get("schema_version"),
        "arrays": records,
    }


def _session_temporal_authority(
    run_group: Any,
    *,
    refs: Mapping[str, Any],
    dimensions: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        refs["source_segment_kind"] != "full_session"
        or refs["source_segment_id"] != "full_session"
        or refs["source_segment_path"] != "recording/full_session"
    ):
        raise ValueError("session occupancy full-session authority differs")
    windows = run_group["windows"]
    total_frames = int(dimensions["total_frames"])
    fps = float(dimensions["fps"])
    expected = {
        "window_id": np.asarray([0], dtype=np.int32),
        "source_segment_id": np.asarray([0], dtype=np.int32),
        "start_frame": np.asarray([0], dtype=np.int64),
        "end_frame": np.asarray([max(0, total_frames - 1)], dtype=np.int64),
        "start_time_s": np.asarray([0.0], dtype=np.float64),
        "end_time_s": np.asarray([float(total_frames) / fps], dtype=np.float64),
        "duration_s": np.asarray([float(total_frames) / fps], dtype=np.float64),
    }
    for name, values in expected.items():
        if not np.array_equal(np.asarray(windows[name][:]), values):
            raise ValueError(f"session occupancy {name} differs from recording axis")
    if _decode_labels(windows["label_bytes"]) != ["full_session"]:
        raise ValueError("session occupancy label differs from full_session")
    return {
        "schema_id": "palette.occupancy_full_session_temporal_authority",
        "schema_version": 1,
        "source_segment_path": "recording/full_session",
        "fps": fps,
        "total_frames": total_frames,
        "start_frame": 0,
        "end_frame": max(0, total_frames - 1),
        "duration_s": float(total_frames) / fps,
    }


def build_occupancy_source_identity(
    root: Any,
    run_group: Any,
    *,
    stage_id: str,
    logical_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Recompute the complete logical and external occupancy source identity."""

    if stage_id not in _OCCUPANCY_STAGES:
        raise ValueError(f"unsupported occupancy stage {stage_id!r}")
    attrs = _attrs(run_group)
    expected_schema = (
        "palette.detection_occupancy.v1"
        if stage_id == "detection_occupancy"
        else "palette.session_occupancy.v1"
    )
    if (
        attrs.get("schema_id") != expected_schema
        or attrs.get("schema_version") != 1
        or attrs.get("coordinate_space") != "source_image_pixels"
    ):
        raise ValueError("occupancy run schema or coordinate-space identity differs")
    refs = _require_source_refs(attrs, stage_id=stage_id)
    parameters = _scientific_parameters(attrs)
    detection = _detection_authority(root, refs=refs, parameters=parameters)
    dimensions = detection["dimensions"]
    for field, expected in (
        ("width", dimensions["width_px"]),
        ("height", dimensions["height_px"]),
        ("fps", dimensions["fps"]),
        ("total_frames", dimensions["total_frames"]),
    ):
        if attrs.get(field) != expected:
            raise ValueError(f"occupancy persisted {field} differs from live source")
    if stage_id == "detection_occupancy":
        segment = _stimulus_authority(root, run_group, refs=refs)
    else:
        segment = _session_temporal_authority(
            run_group,
            refs=refs,
            dimensions=dimensions,
        )
    hashes = {str(path): str(digest) for path, digest in logical_hashes.items()}
    if not hashes or any(
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in hashes.values()
    ):
        raise ValueError("occupancy logical hashes are empty or invalid")
    result = {
        "schema_id": OCCUPANCY_SOURCE_IDENTITY_SCHEMA_ID,
        "schema_version": OCCUPANCY_SOURCE_IDENTITY_SCHEMA_VERSION,
        "stage_id": stage_id,
        "logical_arrays_sha256": canonical_json_sha256(hashes),
        "detection_authority": detection,
        "segment_authority": segment,
        "scientific_parameters": parameters,
    }
    return _strict_json_copy(result)


def occupancy_source_identity_sha256(identity: Mapping[str, Any]) -> str:
    if (
        not isinstance(identity, Mapping)
        or identity.get("schema_id") != OCCUPANCY_SOURCE_IDENTITY_SCHEMA_ID
        or identity.get("schema_version") != OCCUPANCY_SOURCE_IDENTITY_SCHEMA_VERSION
        or identity.get("stage_id") not in _OCCUPANCY_STAGES
    ):
        raise ValueError("occupancy source identity envelope differs")
    return canonical_json_sha256(_strict_json_copy(identity))


def build_occupancy_coordinate_evidence(
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    source_digest = occupancy_source_identity_sha256(identity)
    segment_role = (
        "stimulus_epoch_windows"
        if identity["stage_id"] == "detection_occupancy"
        else "recording_temporal_axis"
    )
    authorities = [
        {
            "role": "detection_geometry",
            "sha256": canonical_json_sha256(identity["detection_authority"]),
        },
        {
            "role": segment_role,
            "sha256": canonical_json_sha256(identity["segment_authority"]),
        },
    ]
    authorities.sort(key=lambda item: str(item["role"]))
    validation = {
        "stage_id": identity["stage_id"],
        "source_identity_sha256": source_digest,
        "source_authority_digests": authorities,
    }
    return {
        "role": "bound_derivative",
        "status": CoordinateEvidenceStatus.VERIFIED_BOUND_SOURCE.value,
        "source_authority_digests": authorities,
        "published_authority_sha256": None,
        "published_authority_ref": None,
        "temporal_axis_sha256": None,
        "temporal_axis_ref": None,
        "validator_ref": (
            "fisheye.analysis_workflows.occupancy_candidate_execution:"
            "build_occupancy_source_identity"
        ),
        "validation_receipt_sha256": canonical_json_sha256(validation),
        "coordinate_gate_passed": True,
    }


def require_occupancy_invocation_parameters(value: object) -> Mapping[str, Any]:
    expected = {
        "source_spatiotemporal_identity_sha256",
        "storage_profile_id",
        "copy_backend",
        "keep_scratch",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("occupancy invocation parameter field set differs")
    digest = value["source_spatiotemporal_identity_sha256"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("source_spatiotemporal_identity_sha256 is invalid")
    if value["storage_profile_id"] != "published_http_v1":
        raise ValueError("occupancy storage_profile_id differs")
    if value["copy_backend"] not in {"python", "rsync"}:
        raise ValueError("occupancy copy_backend differs")
    if type(value["keep_scratch"]) is not bool:
        raise TypeError("occupancy keep_scratch must be an exact bool")
    return value


__all__ = [
    "OCCUPANCY_INVOCATION_CONTRACT_ID",
    "OCCUPANCY_SOURCE_IDENTITY_SCHEMA_ID",
    "OCCUPANCY_SOURCE_IDENTITY_SCHEMA_VERSION",
    "build_occupancy_coordinate_evidence",
    "build_occupancy_source_identity",
    "occupancy_source_identity_sha256",
    "require_occupancy_invocation_parameters",
]
