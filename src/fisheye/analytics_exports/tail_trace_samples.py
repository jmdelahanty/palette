"""Fail-closed sources and bounded projection for long-form tail traces.

The recording-local tail-kinematics, subject-shape, and track-kinematics Zarr
publications remain authority.  This module binds those exact publications and
projects one bounded tail-observation window into primitive long-form columns;
publication is deliberately a separate boundary so no caller can mistake a
projection helper for selector or registry authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import socket
from typing import Any, Mapping
import uuid

import numpy as np

from fisheye.analysis.tail_kinematics_runs import (
    TAIL_KINEMATICS_SCHEMA_ID,
    TAIL_KINEMATICS_SCHEMA_VERSION,
)
from fisheye.analysis.tail_kinematics_schema import (
    TAIL_KINEMATICS_ARRAY_SCHEMA_ATTR,
    TAIL_KINEMATICS_ARRAY_SCHEMA_DIGEST_ATTR,
    validate_tail_kinematics_array_schema,
)
from fisheye.analytics_exports import kinematics_samples as track_export
from fisheye.analytics_exports.arrow_contracts import (
    ARROW_TABLE_CONTRACTS,
    arrow_contract_envelope,
    exact_arrow_schema,
    validate_arrow_schema,
)
from fisheye.analytics_exports.capabilities import resolve_capabilities
from fisheye.analytics_exports.contracts import (
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
    TABLE_CONTRACTS,
    TAIL_TRACE_SAMPLES_TABLE,
    contract_snapshot,
)
from fisheye.analytics_exports.publication import (
    PUBLICATION_SCHEMA_ID,
    PUBLICATION_SCHEMA_VERSION,
    commit_staged_publication,
    export_manifest_path,
    generation_relative_path,
    manifest_identity,
    manifest_selected_part_files_from_payload,
    publication_generation_root,
    publication_staging_root,
    safe_component,
    sha256_file,
)
from fisheye.analytics_exports.runtime_telemetry import ExportRuntimePhaseRecorder
from fisheye.shared.coordinate_frame_record import (
    ARRAY_PAYLOAD_CANONICALIZATION,
    array_values_sha256,
)
from fisheye.shared.coordinate_identity import (
    TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE,
)
from fisheye.shared.tail_coordinate_publication import (
    load_tail_kinematics_coordinate_publication,
)
from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr_io import open_zarr_root

TAIL_TRACE_EXPORT_SCHEMA_ID = "palette.analytics_export.tail_trace_samples"
TAIL_TRACE_EXPORT_SCHEMA_VERSION = 1
TAIL_TRACE_SOURCE_BINDING_SCHEMA_ID = "palette.tail_trace_samples.source_binding"
TAIL_TRACE_SOURCE_BINDING_SCHEMA_VERSION = 1
TAIL_TRACE_PROJECTION_SCHEMA_ID = "palette.tail_trace_samples.projection"
TAIL_TRACE_PROJECTION_SCHEMA_VERSION = 1
TAIL_TRACE_PROJECTED_PAYLOAD_SCHEMA_ID = "palette.tail_trace_samples.projected_payload"
TAIL_TRACE_PROJECTED_PAYLOAD_SCHEMA_VERSION = 1
TAIL_TRACE_PARQUET_POLICY_SCHEMA_ID = "palette.tail_trace_samples.parquet_policy"
TAIL_TRACE_PARQUET_POLICY_SCHEMA_VERSION = 1

TAIL_TRACE_REASON_VALID = np.uint16(0)
TAIL_TRACE_REASON_SOURCE_INVALID = np.uint16(1)
TAIL_TRACE_REASON_REFERENCE_INVALID = np.uint16(2)
TAIL_TRACE_REASON_GEOMETRY_NONFINITE = np.uint16(3)

TAIL_TRACE_SCIENTIFIC_DTYPES: Mapping[str, np.dtype[Any]] = {
    "source_tail_row_index": np.dtype("<i8"),
    "track_id": np.dtype("<i8"),
    "instance_key": np.dtype("<u8"),
    "source_crop_row_id": np.dtype("<i8"),
    "source_acquisition_frame_index": np.dtype("<i8"),
    "time_seconds": np.dtype("<f8"),
    "tail_sample_index": np.dtype("<i4"),
    "normalized_tail_position": np.dtype("<f4"),
    "reference_length_px": np.dtype("<f4"),
    "body_longitudinal_fraction": np.dtype("<f4"),
    "body_lateral_fraction": np.dtype("<f4"),
    "tangent_angle_rad": np.dtype("<f4"),
    "body_curvature_dimensionless": np.dtype("<f4"),
    "source_camera_x_px": np.dtype("<f4"),
    "source_camera_y_px": np.dtype("<f4"),
    "source_camera_curvature_px_inv": np.dtype("<f4"),
    "source_lateral_deflection_px": np.dtype("<f4"),
    "source_tail_row_valid": np.dtype("bool"),
    "reference_length_valid": np.dtype("bool"),
    "sample_valid": np.dtype("bool"),
    "sample_reason_code": np.dtype("<u2"),
}

_TAIL_WINDOW_ARRAYS: Mapping[str, tuple[np.dtype[Any], tuple[int | str, ...]]] = {
    "instance_key": (np.dtype("<u8"), ("rows",)),
    "source_crop_row_ids": (np.dtype("<i8"), ("rows",)),
    "source_acquisition_frame_index": (np.dtype("<i8"), ("rows",)),
    "valid": (np.dtype("bool"), ("rows",)),
    "failure_reason_bytes": (np.dtype("u1"), ("rows", 64)),
    "tail_angle_sample_xy": (np.dtype("<f4"), ("rows", "samples", 2)),
    "tail_angle_rad": (np.dtype("<f4"), ("rows", "samples")),
    "tail_curvature_px_inv": (np.dtype("<f4"), ("rows", "samples")),
    "tail_lateral_deflection_px": (np.dtype("<f4"), ("rows", "samples")),
}

_SHAPE_WINDOW_ARRAYS: Mapping[str, tuple[np.dtype[Any], tuple[int | str, ...]]] = {
    "instance_key": (np.dtype("<u8"), ("rows",)),
    "source_crop_row_ids": (np.dtype("<i8"), ("rows",)),
    "source_acquisition_frame_index": (np.dtype("<i8"), ("rows",)),
    "components/subject_body/tail_base_xy": (
        np.dtype("<f4"),
        ("rows", 2),
    ),
    "components/subject_body/tail_segment_arclength_px": (
        np.dtype("<f4"),
        ("rows",),
    ),
    "components/subject_body/tail_base_valid": (
        np.dtype("bool"),
        ("rows",),
    ),
    "body_frame/forward_axis_xy": (np.dtype("<f4"), ("rows", 2)),
    "body_frame/left_axis_xy": (np.dtype("<f4"), ("rows", 2)),
    "body_frame/axis_valid": (np.dtype("bool"), ("rows",)),
}

_ARRAY_RECORD_FIELDS = {
    "relative_ref",
    "dtype",
    "shape",
    "content_sha256",
    "canonicalization",
}
_SUBJECT_SHAPE_ARRAY_RECORD_FIELDS = _ARRAY_RECORD_FIELDS | {"array_ref"}


@dataclass(frozen=True)
class TailTrackIdentityIndex:
    """Sorted exact observation-to-track identity map."""

    instance_keys: np.ndarray
    track_ids: np.ndarray
    frame_indices: np.ndarray
    record: Mapping[str, Any]


@dataclass(frozen=True)
class BoundTailTraceSources:
    """Verified handles plus the canonical source-binding receipt."""

    binding: Mapping[str, Any]
    tail_run: Any
    subject_shape_run: Any
    track_source: Any
    track_index: TailTrackIdentityIndex
    tail_sample_s: np.ndarray
    reference_length_node: Any
    reference_validity_node: Any
    subject_shape_publication: Any | None = None


def _attrs(group: Any) -> dict[str, Any]:
    values = getattr(group, "attrs", {})
    return dict(values.asdict() if hasattr(values, "asdict") else dict(values))


def _child(group: Any, path: str) -> Any:
    node = group
    for component in str(path).strip("/").split("/"):
        if component:
            node = node[component]
    return node


def _exact_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} must be one lowercase SHA-256 digest.")
    return text


def _recording_id(path: Path) -> str:
    name = path.name.removesuffix(".zarr").removesuffix("_analysis")
    if not name:
        raise ValueError("Cannot derive recording ID from an empty archive name.")
    return name


def _array_manifest_record(
    publication: Any,
    path: str,
    *,
    label: str,
    array_ref_root: str | None = None,
) -> Mapping[str, Any]:
    arrays = publication.manifest.record.get("arrays")
    record = arrays.get(path) if isinstance(arrays, Mapping) else None
    expected_fields = (
        _SUBJECT_SHAPE_ARRAY_RECORD_FIELDS
        if array_ref_root is not None
        else _ARRAY_RECORD_FIELDS
    )
    if (
        not isinstance(record, Mapping)
        or set(record) != expected_fields
        or record.get("relative_ref") != path
        or record.get("canonicalization") != ARRAY_PAYLOAD_CANONICALIZATION
    ):
        raise ValueError(f"{label} manifest does not bind exact array {path!r}.")
    if array_ref_root is not None and record.get("array_ref") != (
        f"/{array_ref_root.strip('/')}/{path}"
    ):
        raise ValueError(f"{label} manifest has an invalid array_ref for {path!r}.")
    raw_dtype = record.get("dtype")
    if not isinstance(raw_dtype, str):
        raise ValueError(f"{label} manifest has an invalid dtype for {path!r}.")
    try:
        parsed_dtype = np.dtype(raw_dtype)
    except TypeError as exc:
        raise ValueError(
            f"{label} manifest has an invalid dtype for {path!r}."
        ) from exc
    if raw_dtype != parsed_dtype.str:
        raise ValueError(f"{label} manifest dtype is not canonical for {path!r}.")
    shape = record.get("shape")
    if not isinstance(shape, list) or any(
        type(value) is not int or value < 0 for value in shape
    ):
        raise ValueError(f"{label} manifest has an invalid shape for {path!r}.")
    _exact_sha256(record.get("content_sha256"), label=f"{label} {path} content")
    return dict(record)


def _tail_array_schema_adoption(run: Any) -> tuple[bool, Mapping[str, Any]]:
    attrs = _attrs(run)
    manifest = attrs.get(TAIL_KINEMATICS_ARRAY_SCHEMA_ATTR)
    if not isinstance(manifest, Mapping):
        raise ValueError("Tail source lacks its exact array-schema manifest.")
    payload = manifest.get("payload")
    declarations = payload.get("declarations") if isinstance(payload, Mapping) else None
    if not isinstance(declarations, list) or not declarations:
        raise ValueError("Tail source array-schema declarations are invalid.")
    adoption = {
        declaration.get("byte_planner_adopted")
        for declaration in declarations
        if isinstance(declaration, Mapping)
    }
    if adoption not in ({True}, {False}):
        raise ValueError("Tail source has mixed or invalid byte-planner adoption.")
    digest = canonical_json_sha256(dict(manifest))
    if attrs.get(TAIL_KINEMATICS_ARRAY_SCHEMA_DIGEST_ATTR) != manifest.get(
        "payload_digest"
    ):
        raise ValueError("Tail source array-schema digest attribute is stale.")
    errors = validate_tail_kinematics_array_schema(
        run,
        byte_planner_adopted=bool(next(iter(adoption))),
    )
    if errors:
        raise ValueError(f"Tail source exact array schema is invalid: {errors!r}.")
    return bool(next(iter(adoption))), {
        "manifest_sha256": digest,
        "payload_sha256": manifest["payload_digest"],
    }


def _source_fps_from_subject_shape_authority(
    publication: Any,
    *,
    expected_recording_id: str,
) -> tuple[float, str]:
    """Resolve FPS only through the sealed subject-shape temporal authority."""

    temporal = getattr(publication, "temporal_authority", None)
    acquisition = getattr(temporal, "acquisition_frame", None)
    record = getattr(acquisition, "record", None)
    metadata = getattr(record, "source_video_metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError(
            "Tail trace export requires canonical subject-shape acquisition authority."
        )
    if getattr(record, "recording_id", None) != expected_recording_id:
        raise ValueError(
            "Tail subject-shape acquisition authority binds another recording."
        )
    value = metadata.get("fps")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "Tail subject-shape acquisition authority lacks exact source FPS."
        )
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(
            "Tail subject-shape acquisition authority FPS must be positive and finite."
        )
    record_ref = getattr(acquisition, "record_ref", None)
    record_sha256 = getattr(acquisition, "record_sha256", None)
    if not isinstance(record_ref, str) or not record_ref:
        raise ValueError("Tail acquisition authority lacks its canonical record ref.")
    _exact_sha256(record_sha256, label="tail acquisition authority record")
    return rate, f"{record_ref}.source_video_metadata.fps"


class _BoundedPayloadHasher:
    def __init__(self, *, dtype: np.dtype[Any], shape: tuple[int, ...]) -> None:
        header = {
            "canonicalization": ARRAY_PAYLOAD_CANONICALIZATION,
            "dtype": np.lib.format.dtype_to_descr(dtype),
            "shape": [int(value) for value in shape],
        }
        self._hasher = hashlib.sha256()
        self._hasher.update(canonical_json_bytes(header))
        self._hasher.update(b"\x00")

    def update(self, values: np.ndarray) -> None:
        self._hasher.update(np.ascontiguousarray(values).tobytes(order="C"))

    def hexdigest(self) -> str:
        return self._hasher.hexdigest()


def _build_track_identity_index(
    track_source: Any,
    *,
    source_window_rows: int,
) -> TailTrackIdentityIndex:
    if type(source_window_rows) is not int or source_window_rows <= 0:
        raise ValueError("source_window_rows must be a positive exact integer.")
    key_parts: list[np.ndarray] = []
    track_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    for track in track_source.binding["tracks"]:
        track_id = int(track["track_id"])
        count = int(track["sample_count"])
        group = track_source.run_group["tracks"][f"id_{track_id}"]
        records = track["selected_surfaces"]
        nodes = {
            "track_sample_key": _child(group, "track_sample_key"),
            "source_acquisition_frame_index": _child(
                group, "source_acquisition_frame_index"
            ),
            "source_instance_key": _child(group, "source_instance_key"),
        }
        expected = {
            "track_sample_key": (np.dtype("<i8"), (count, 2)),
            "source_acquisition_frame_index": (np.dtype("<i8"), (count,)),
            "source_instance_key": (
                TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE,
                (count,),
            ),
        }
        hashers: dict[str, _BoundedPayloadHasher] = {}
        for name, (dtype, shape) in expected.items():
            node = nodes[name]
            if (
                np.dtype(node.dtype) != dtype
                or tuple(int(v) for v in node.shape) != shape
            ):
                raise ValueError(f"Track {track_id} identity surface {name!r} changed.")
            hashers[name] = _BoundedPayloadHasher(dtype=dtype, shape=shape)
        previous_frame: int | None = None
        for start in range(0, count, source_window_rows):
            stop = min(count, start + source_window_rows)
            values = {
                name: np.asarray(node[start:stop]) for name, node in nodes.items()
            }
            for name, value in values.items():
                hashers[name].update(value)
            sample_key = values["track_sample_key"]
            frames = values["source_acquisition_frame_index"]
            instance = values["source_instance_key"]
            if (
                np.any(sample_key[:, 0] != track_id)
                or not np.array_equal(sample_key[:, 1], frames)
                or np.any(frames < 0)
                or (frames.size > 1 and np.any(np.diff(frames) <= 0))
                or (
                    previous_frame is not None
                    and frames.size
                    and int(frames[0]) <= previous_frame
                )
            ):
                raise ValueError(f"Track {track_id} temporal identity is invalid.")
            if frames.size:
                previous_frame = int(frames[-1])
            valid = np.asarray(instance["valid"], dtype=bool)
            keys = np.asarray(instance["instance_key"], dtype=np.uint64)
            if np.any((~valid) & (keys != 0)) or np.any(valid & (keys == 0)):
                raise ValueError(
                    f"Track {track_id} instance-key null semantics are invalid."
                )
            if np.any(valid):
                key_parts.append(keys[valid].copy())
                track_parts.append(
                    np.full(np.count_nonzero(valid), track_id, dtype=np.int64)
                )
                frame_parts.append(frames[valid].copy())
        for name, hasher in hashers.items():
            expected_digest = records[name]["content_sha256"]
            if hasher.hexdigest() != expected_digest:
                raise ValueError(
                    f"Track {track_id} identity surface {name!r} differs from its manifest."
                )

    keys = np.concatenate(key_parts) if key_parts else np.empty(0, dtype=np.uint64)
    tracks = np.concatenate(track_parts) if track_parts else np.empty(0, dtype=np.int64)
    frames = np.concatenate(frame_parts) if frame_parts else np.empty(0, dtype=np.int64)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    tracks = tracks[order]
    frames = frames[order]
    if keys.size > 1 and np.any(keys[1:] == keys[:-1]):
        raise ValueError(
            "Track source maps one instance_key to multiple track samples."
        )
    body: dict[str, Any] = {
        "schema_id": "palette.tail_trace_samples.track_identity_index",
        "schema_version": 1,
        "row_count": int(keys.size),
        "sort_order": "instance_key_ascending",
        "instance_key_sha256": array_values_sha256(keys),
        "track_id_sha256": array_values_sha256(tracks),
        "source_acquisition_frame_index_sha256": array_values_sha256(frames),
        "uniqueness": "one_valid_instance_key_to_exactly_one_track_sample",
    }
    return TailTrackIdentityIndex(
        instance_keys=keys,
        track_ids=tracks,
        frame_indices=frames,
        record={**body, "payload_sha256": canonical_json_sha256(body)},
    )


def _join_track_identities(
    index: TailTrackIdentityIndex,
    *,
    instance_keys: np.ndarray,
    frame_indices: np.ndarray,
) -> np.ndarray:
    keys = np.asarray(instance_keys)
    frames = np.asarray(frame_indices)
    if keys.dtype != np.dtype("<u8") or frames.dtype != np.dtype("<i8"):
        raise ValueError("Tail-to-track join identities changed dtype.")
    positions = np.searchsorted(index.instance_keys, keys)
    in_bounds = positions < index.instance_keys.size
    matched = np.zeros(keys.shape, dtype=bool)
    matched[in_bounds] = index.instance_keys[positions[in_bounds]] == keys[in_bounds]
    if not np.all(matched):
        missing = keys[~matched][:8].tolist()
        raise ValueError(f"Tail rows lack exact track membership: {missing!r}.")
    matched_frames = index.frame_indices[positions]
    if not np.array_equal(matched_frames, frames):
        raise ValueError("Tail and track instance identities disagree on camera frame.")
    return index.track_ids[positions].copy()


def tail_trace_projection_contract() -> dict[str, Any]:
    """Return the closed v1 body-frame projection contract."""

    body: dict[str, Any] = {
        "schema_id": TAIL_TRACE_PROJECTION_SCHEMA_ID,
        "schema_version": TAIL_TRACE_PROJECTION_SCHEMA_VERSION,
        "source_grain": "one_tail_kinematics_observation_with_fixed_run_sample_axis",
        "output_grain": "one_observation_by_normalized_tail_position",
        "source_position_space": "source_camera_pixels",
        "reference_length_kind": "tail_base_to_tail_tip_centerline_arclength_px",
        "reference_length_path": "components/subject_body/tail_segment_arclength_px",
        "origin_path": "components/subject_body/tail_base_xy",
        "longitudinal_axis_convention": "caudal_axis_equals_negative_body_forward_axis;positive_tail_base_to_tip",
        "lateral_axis_convention": "positive_anatomical_left_axis",
        "angle_convention": "source_tail_angle_rad_about_caudal_axis;positive_anatomical_left",
        "curvature_convention": "gradient_of_unwrapped_tangent_angle_rad_over_normalized_s;nonuniform_numpy_gradient_edge_order_1",
        "axis_validation": "finite_unit_orthogonal_axes_absolute_tolerance_1e-4",
        "invalid_value_policy": "identity_and_sample_coordinates_retained;invalid_scientific_floats_are_ieee_nan",
        "reason_registry": {
            "0": "valid",
            "1": "source_tail_row_invalid",
            "2": "reference_length_or_body_frame_invalid",
            "3": "derived_geometry_nonfinite",
        },
        "source_failure_reason_policy": "strict_utf8_nul_terminated_row_reason_repeated_per_sample",
    }
    return {**body, "payload_sha256": canonical_json_sha256(body)}


def _validate_projection_inputs(
    *,
    source_tail_row_indices: np.ndarray,
    track_ids: np.ndarray,
    instance_keys: np.ndarray,
    source_crop_row_ids: np.ndarray,
    source_acquisition_frame_indices: np.ndarray,
    source_tail_row_valid: np.ndarray,
    source_failure_reasons: np.ndarray,
    tail_sample_s: np.ndarray,
    tail_sample_xy: np.ndarray,
    tail_angle_rad: np.ndarray,
    tail_curvature_px_inv: np.ndarray,
    tail_lateral_deflection_px: np.ndarray,
    tail_base_xy: np.ndarray,
    reference_length_px: np.ndarray,
    reference_length_source_valid: np.ndarray,
    body_forward_axis_xy: np.ndarray,
    body_left_axis_xy: np.ndarray,
    body_frame_valid: np.ndarray,
) -> tuple[int, int]:
    row_count = int(source_tail_row_indices.shape[0])
    if source_tail_row_indices.dtype != np.dtype(
        "<i8"
    ) or source_tail_row_indices.shape != (row_count,):
        raise ValueError("source_tail_row_indices must be exact int64 rows.")
    exact_rows = {
        "track_ids": (track_ids, np.dtype("<i8")),
        "instance_keys": (instance_keys, np.dtype("<u8")),
        "source_crop_row_ids": (source_crop_row_ids, np.dtype("<i8")),
        "source_acquisition_frame_indices": (
            source_acquisition_frame_indices,
            np.dtype("<i8"),
        ),
        "source_tail_row_valid": (source_tail_row_valid, np.dtype("bool")),
        "reference_length_px": (reference_length_px, np.dtype("<f4")),
        "reference_length_source_valid": (
            reference_length_source_valid,
            np.dtype("bool"),
        ),
        "body_frame_valid": (body_frame_valid, np.dtype("bool")),
    }
    for name, (values, dtype) in exact_rows.items():
        if values.dtype != dtype or values.shape != (row_count,):
            raise ValueError(f"{name} must have exact dtype {dtype} and row shape.")
    reasons = np.asarray(source_failure_reasons)
    if reasons.shape != (row_count,) or reasons.dtype.kind not in {"O", "U"}:
        raise ValueError("source_failure_reasons must be one row-aligned text vector.")
    sample_count = int(tail_sample_s.shape[0])
    if (
        tail_sample_s.dtype != np.dtype("<f4")
        or tail_sample_s.shape != (sample_count,)
        or sample_count < 2
        or not np.all(np.isfinite(tail_sample_s))
        or np.any(np.diff(tail_sample_s.astype(np.float64)) <= 0)
        or float(tail_sample_s[0]) < 0.0
        or float(tail_sample_s[-1]) > 1.0
    ):
        raise ValueError(
            "tail_sample_s must be exact finite increasing float32 in [0, 1]."
        )
    profiles = {
        "tail_sample_xy": (tail_sample_xy, (row_count, sample_count, 2)),
        "tail_angle_rad": (tail_angle_rad, (row_count, sample_count)),
        "tail_curvature_px_inv": (
            tail_curvature_px_inv,
            (row_count, sample_count),
        ),
        "tail_lateral_deflection_px": (
            tail_lateral_deflection_px,
            (row_count, sample_count),
        ),
    }
    for name, (values, shape) in profiles.items():
        if values.dtype != np.dtype("<f4") or values.shape != shape:
            raise ValueError(f"{name} must be exact float32 with shape {shape!r}.")
    for name, values in {
        "tail_base_xy": tail_base_xy,
        "body_forward_axis_xy": body_forward_axis_xy,
        "body_left_axis_xy": body_left_axis_xy,
    }.items():
        if values.dtype != np.dtype("<f4") or values.shape != (row_count, 2):
            raise ValueError(f"{name} must be exact float32 row XY.")
    if (
        np.any(instance_keys == 0)
        or np.any(source_crop_row_ids < 0)
        or np.any(source_acquisition_frame_indices < 0)
    ):
        raise ValueError("Tail row identity contains forbidden negative/zero values.")
    return row_count, sample_count


def project_tail_trace_window(
    *,
    source_tail_row_indices: np.ndarray,
    track_ids: np.ndarray,
    instance_keys: np.ndarray,
    source_crop_row_ids: np.ndarray,
    source_acquisition_frame_indices: np.ndarray,
    source_tail_row_valid: np.ndarray,
    source_failure_reasons: np.ndarray,
    tail_sample_s: np.ndarray,
    tail_sample_xy: np.ndarray,
    tail_angle_rad: np.ndarray,
    tail_curvature_px_inv: np.ndarray,
    tail_lateral_deflection_px: np.ndarray,
    tail_base_xy: np.ndarray,
    reference_length_px: np.ndarray,
    reference_length_source_valid: np.ndarray,
    body_forward_axis_xy: np.ndarray,
    body_left_axis_xy: np.ndarray,
    body_frame_valid: np.ndarray,
    source_sample_rate_hz: float,
) -> dict[str, np.ndarray]:
    """Project one exact bounded row window into primitive long-form columns."""

    row_count, sample_count = _validate_projection_inputs(
        source_tail_row_indices=source_tail_row_indices,
        track_ids=track_ids,
        instance_keys=instance_keys,
        source_crop_row_ids=source_crop_row_ids,
        source_acquisition_frame_indices=source_acquisition_frame_indices,
        source_tail_row_valid=source_tail_row_valid,
        source_failure_reasons=source_failure_reasons,
        tail_sample_s=tail_sample_s,
        tail_sample_xy=tail_sample_xy,
        tail_angle_rad=tail_angle_rad,
        tail_curvature_px_inv=tail_curvature_px_inv,
        tail_lateral_deflection_px=tail_lateral_deflection_px,
        tail_base_xy=tail_base_xy,
        reference_length_px=reference_length_px,
        reference_length_source_valid=reference_length_source_valid,
        body_forward_axis_xy=body_forward_axis_xy,
        body_left_axis_xy=body_left_axis_xy,
        body_frame_valid=body_frame_valid,
    )
    if isinstance(source_sample_rate_hz, bool) or not isinstance(
        source_sample_rate_hz, (int, float)
    ):
        raise ValueError("source_sample_rate_hz must be positive and finite.")
    rate = float(source_sample_rate_hz)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("source_sample_rate_hz must be positive and finite.")

    forward64 = body_forward_axis_xy.astype(np.float64)
    left64 = body_left_axis_xy.astype(np.float64)
    axis_finite = np.all(np.isfinite(forward64), axis=1) & np.all(
        np.isfinite(left64), axis=1
    )
    axis_unit = np.isclose(
        np.linalg.norm(forward64, axis=1), 1.0, rtol=0.0, atol=1e-4
    ) & np.isclose(np.linalg.norm(left64, axis=1), 1.0, rtol=0.0, atol=1e-4)
    axis_orthogonal = np.isclose(
        np.einsum("ij,ij->i", forward64, left64),
        0.0,
        rtol=0.0,
        atol=1e-4,
    )
    reference_valid = (
        reference_length_source_valid
        & body_frame_valid
        & np.isfinite(reference_length_px)
        & (reference_length_px > 0)
        & np.all(np.isfinite(tail_base_xy), axis=1)
        & axis_finite
        & axis_unit
        & axis_orthogonal
    )
    delta = (
        tail_sample_xy.astype(np.float64) - tail_base_xy.astype(np.float64)[:, None, :]
    )
    caudal = -forward64
    length64 = reference_length_px.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        longitudinal = np.einsum("nkj,nj->nk", delta, caudal) / length64[:, None]
        lateral = np.einsum("nkj,nj->nk", delta, left64) / length64[:, None]
        unwrapped = np.unwrap(tail_angle_rad.astype(np.float64), axis=1)
        body_curvature = np.gradient(
            unwrapped,
            tail_sample_s.astype(np.float64),
            axis=1,
            edge_order=1,
        )
    derived_finite = (
        np.all(np.isfinite(tail_sample_xy), axis=2)
        & np.isfinite(tail_angle_rad)
        & np.isfinite(tail_curvature_px_inv)
        & np.isfinite(tail_lateral_deflection_px)
        & np.isfinite(longitudinal)
        & np.isfinite(lateral)
        & np.isfinite(body_curvature)
    )
    sample_valid_2d = (
        source_tail_row_valid[:, None] & reference_valid[:, None] & derived_finite
    )
    reasons = np.full(
        (row_count, sample_count),
        TAIL_TRACE_REASON_VALID,
        dtype=np.uint16,
    )
    reasons[~source_tail_row_valid, :] = TAIL_TRACE_REASON_SOURCE_INVALID
    reasons[source_tail_row_valid & ~reference_valid, :] = (
        TAIL_TRACE_REASON_REFERENCE_INVALID
    )
    reasons[
        source_tail_row_valid[:, None] & reference_valid[:, None] & ~derived_finite
    ] = TAIL_TRACE_REASON_GEOMETRY_NONFINITE

    repeat = sample_count
    tiled_sample_index = np.tile(np.arange(sample_count, dtype=np.int32), row_count)
    scientific = {
        "reference_length_px": np.repeat(reference_length_px, repeat),
        "body_longitudinal_fraction": longitudinal.astype(np.float32).reshape(-1),
        "body_lateral_fraction": lateral.astype(np.float32).reshape(-1),
        "tangent_angle_rad": tail_angle_rad.reshape(-1).copy(),
        "body_curvature_dimensionless": body_curvature.astype(np.float32).reshape(-1),
        "source_camera_x_px": tail_sample_xy[:, :, 0].reshape(-1).copy(),
        "source_camera_y_px": tail_sample_xy[:, :, 1].reshape(-1).copy(),
        "source_camera_curvature_px_inv": tail_curvature_px_inv.reshape(-1).copy(),
        "source_lateral_deflection_px": tail_lateral_deflection_px.reshape(-1).copy(),
    }
    flat_valid = sample_valid_2d.reshape(-1)
    for values in scientific.values():
        values[~flat_valid] = np.nan

    output: dict[str, np.ndarray] = {
        "source_tail_row_index": np.repeat(source_tail_row_indices, repeat),
        "track_id": np.repeat(track_ids, repeat),
        "instance_key": np.repeat(instance_keys, repeat),
        "source_crop_row_id": np.repeat(source_crop_row_ids, repeat),
        "source_acquisition_frame_index": np.repeat(
            source_acquisition_frame_indices, repeat
        ),
        "time_seconds": np.repeat(
            source_acquisition_frame_indices.astype(np.float64) / rate, repeat
        ),
        "tail_sample_index": tiled_sample_index,
        "normalized_tail_position": np.tile(tail_sample_s, row_count),
        **scientific,
        "source_tail_row_valid": np.repeat(source_tail_row_valid, repeat),
        "reference_length_valid": np.repeat(reference_valid, repeat),
        "sample_valid": flat_valid,
        "sample_reason_code": reasons.reshape(-1),
        "source_failure_reason": np.repeat(
            np.asarray(source_failure_reasons, dtype=object), repeat
        ),
    }
    for name, dtype in TAIL_TRACE_SCIENTIFIC_DTYPES.items():
        if output[name].dtype != dtype:
            output[name] = output[name].astype(dtype, copy=False)
    return output


def _decode_reason_rows(values: np.ndarray) -> np.ndarray:
    if values.dtype != np.dtype("u1") or values.ndim != 2 or values.shape[1] != 64:
        raise ValueError("Tail failure_reason_bytes changed from exact uint8[N,64].")
    labels: list[str] = []
    for row in values:
        zeros = np.flatnonzero(row == 0)
        stop = int(zeros[0]) if zeros.size else int(row.size)
        try:
            label = bytes(row[:stop]).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("Tail failure reason is not strict UTF-8.") from exc
        if not label:
            raise ValueError("Tail failure reason cannot be empty.")
        labels.append(label)
    return np.asarray(labels, dtype=object)


def _read_exact_window(
    group: Any,
    declarations: Mapping[str, tuple[np.dtype[Any], tuple[int | str, ...]]],
    *,
    start: int,
    stop: int,
    total_rows: int,
    sample_count: int,
) -> dict[str, np.ndarray]:
    row_count = stop - start
    result: dict[str, np.ndarray] = {}
    for path, (dtype, shape_spec) in declarations.items():
        node = _child(group, path)
        expected_full = tuple(
            (
                total_rows
                if value == "rows"
                else sample_count if value == "samples" else int(value)
            )
            for value in shape_spec
        )
        if (
            np.dtype(node.dtype) != dtype
            or tuple(int(value) for value in node.shape) != expected_full
        ):
            raise ValueError(f"Source array {path!r} changed dtype or shape.")
        values = np.asarray(node[start:stop])
        expected_window = (row_count, *expected_full[1:])
        if values.dtype != dtype or values.shape != expected_window:
            raise ValueError(f"Bounded source read {path!r} changed dtype or shape.")
        result[path] = values
    return result


def read_projected_tail_trace_window(
    bound: BoundTailTraceSources,
    *,
    start_row: int,
    stop_row: int,
) -> dict[str, np.ndarray]:
    """Read and project one half-open source-row interval."""

    row_count = int(bound.binding["tail_row_count"])
    if (
        type(start_row) is not int
        or type(stop_row) is not int
        or start_row < 0
        or stop_row < start_row
        or stop_row > row_count
    ):
        raise ValueError("Tail source-row interval is outside the bound run.")
    samples = int(bound.binding["source_tail_sample_count"])
    tail = _read_exact_window(
        bound.tail_run,
        _TAIL_WINDOW_ARRAYS,
        start=start_row,
        stop=stop_row,
        total_rows=row_count,
        sample_count=samples,
    )
    shape = _read_exact_window(
        bound.subject_shape_run,
        _SHAPE_WINDOW_ARRAYS,
        start=start_row,
        stop=stop_row,
        total_rows=row_count,
        sample_count=samples,
    )
    reference = np.asarray(bound.reference_length_node[start_row:stop_row])
    reference_valid = np.asarray(bound.reference_validity_node[start_row:stop_row])
    if reference.dtype != np.dtype("<f4") or reference.shape != (stop_row - start_row,):
        raise ValueError("Bound reference-length surface changed dtype or shape.")
    if reference_valid.dtype != np.dtype("bool") or reference_valid.shape != (
        stop_row - start_row,
    ):
        raise ValueError("Bound reference-length validity changed dtype or shape.")
    for name in (
        "instance_key",
        "source_crop_row_ids",
        "source_acquisition_frame_index",
    ):
        if not np.array_equal(tail[name], shape[name]):
            raise ValueError(f"Tail and subject-shape {name} lineage differs.")
    track_ids = _join_track_identities(
        bound.track_index,
        instance_keys=tail["instance_key"],
        frame_indices=tail["source_acquisition_frame_index"],
    )
    return project_tail_trace_window(
        source_tail_row_indices=np.arange(start_row, stop_row, dtype=np.int64),
        track_ids=track_ids,
        instance_keys=tail["instance_key"],
        source_crop_row_ids=tail["source_crop_row_ids"],
        source_acquisition_frame_indices=tail["source_acquisition_frame_index"],
        source_tail_row_valid=tail["valid"],
        source_failure_reasons=_decode_reason_rows(tail["failure_reason_bytes"]),
        tail_sample_s=bound.tail_sample_s,
        tail_sample_xy=tail["tail_angle_sample_xy"],
        tail_angle_rad=tail["tail_angle_rad"],
        tail_curvature_px_inv=tail["tail_curvature_px_inv"],
        tail_lateral_deflection_px=tail["tail_lateral_deflection_px"],
        tail_base_xy=shape["components/subject_body/tail_base_xy"],
        reference_length_px=reference,
        reference_length_source_valid=reference_valid,
        body_forward_axis_xy=shape["body_frame/forward_axis_xy"],
        body_left_axis_xy=shape["body_frame/left_axis_xy"],
        body_frame_valid=shape["body_frame/axis_valid"],
        source_sample_rate_hz=float(bound.binding["source_sample_rate_hz"]),
    )


def bind_tail_trace_sources(
    root: Any,
    *,
    zarr_path: Path,
    tail_kinematics_run: str,
    subject_shape_run: str,
    track_kinematics_run: str,
    track_scope: str,
    source_window_rows: int = 65_536,
    prebound_track_source: Any | None = None,
) -> BoundTailTraceSources:
    """Bind exact tail/body/track authorities without publishing an export."""

    tail_name = str(tail_kinematics_run).strip()
    if not tail_name or "/" in tail_name:
        raise ValueError("tail_kinematics_run must be one explicit child name.")
    tail_path = f"analysis/tail_kinematics_runs/{tail_name}"
    publication = load_tail_kinematics_coordinate_publication(root, tail_path)
    tail_run = publication._run
    tail_attrs = _attrs(tail_run)
    if (
        tail_attrs.get("schema_id") != TAIL_KINEMATICS_SCHEMA_ID
        or tail_attrs.get("schema_version") != TAIL_KINEMATICS_SCHEMA_VERSION
        or tail_attrs.get("palette_run_completion_status") != "complete"
        or tail_attrs.get("stage_selector_eligible") is not True
    ):
        raise ValueError("Tail source must be one exact completed eligible v2 run.")
    adopted, array_schema = _tail_array_schema_adoption(tail_run)
    tail_sample_s = np.asarray(tail_run["tail_angle_sample_s"][:])
    if (
        tail_sample_s.dtype != np.dtype("<f4")
        or tail_sample_s.ndim != 1
        or tail_sample_s.size < 2
        or not np.all(np.isfinite(tail_sample_s))
        or np.any(np.diff(tail_sample_s.astype(np.float64)) <= 0)
        or float(tail_sample_s[0]) < 0.0
        or float(tail_sample_s[-1]) > 1.0
    ):
        raise ValueError("Tail sample axis is not exact increasing float32 in [0,1].")
    tail_rows = int(tail_run["valid"].shape[0])
    tail_samples = int(tail_sample_s.size)

    shape_publication = publication.source
    expected_shape_name = str(subject_shape_run).strip()
    if not expected_shape_name or "/" in expected_shape_name:
        raise ValueError("subject_shape_run must be one explicit child name.")
    if shape_publication.run_path != (
        f"analysis/subject_shape_runs/{expected_shape_name}"
    ):
        raise ValueError(
            "Tail publication subject-shape authority differs from the explicit "
            "workflow dependency."
        )
    shape_run = shape_publication._run
    shape_attrs = _attrs(shape_run)
    if shape_publication.manifest.record_sha256 != tail_attrs.get(
        "source_subject_shape_publication_manifest_sha256"
    ):
        raise ValueError("Tail source does not bind the loaded subject-shape manifest.")
    reference = shape_publication.require_scalar_surface(
        "components/subject_body/tail_segment_arclength_px",
        units="px",
        surface_kind="row_scalar",
    )
    if int(reference.array_node.shape[0]) != tail_rows:
        raise ValueError("Subject-shape reference length is not tail-row aligned.")
    if tuple(int(value) for value in reference.validity_node.shape) != (tail_rows,):
        raise ValueError("Subject-shape reference validity is not tail-row aligned.")
    body_frame_sha = _exact_sha256(
        shape_publication.body_frame.record_sha256,
        label="subject-shape body-frame record",
    )

    recording_id = _recording_id(Path(zarr_path))
    track_source = prebound_track_source
    if track_source is None:
        track_source = track_export.bind_kinematics_samples_source(
            root,
            zarr_path=Path(zarr_path),
            expected_recording_id=recording_id,
            track_kinematics_run=track_kinematics_run,
            track_scope=track_scope,
        )
    elif (
        track_source.binding.get("recording_id") != recording_id
        or track_source.binding.get("zarr_path") != str(Path(zarr_path).resolve())
        or track_source.binding.get("run_name") != track_kinematics_run
        or track_source.binding.get("scope") != track_scope
    ):
        raise ValueError(
            "Prebound track source differs from the explicit tail dependency."
        )
    track_index = _build_track_identity_index(
        track_source,
        source_window_rows=source_window_rows,
    )
    if int(track_index.instance_keys.size) != tail_rows:
        raise ValueError(
            "Tail source row count differs from valid track instance membership."
        )
    source_fps, fps_source = _source_fps_from_subject_shape_authority(
        shape_publication,
        expected_recording_id=recording_id,
    )
    track_fps = float(track_source.binding["source_sample_rate_hz"])
    if source_fps != track_fps:
        raise ValueError("Tail and track sources disagree on exact source FPS.")

    tail_axis_record = _array_manifest_record(
        publication, "tail_angle_sample_s", label="tail publication"
    )
    reference_record = _array_manifest_record(
        shape_publication,
        "components/subject_body/tail_segment_arclength_px",
        label="subject-shape publication",
        array_ref_root=shape_publication.run_path,
    )
    selected_tail_paths = tuple(_TAIL_WINDOW_ARRAYS)
    selected_shape_paths = tuple(_SHAPE_WINDOW_ARRAYS)
    payload: dict[str, Any] = {
        "schema_id": TAIL_TRACE_SOURCE_BINDING_SCHEMA_ID,
        "schema_version": TAIL_TRACE_SOURCE_BINDING_SCHEMA_VERSION,
        "stage_id": "tail_traces",
        "recording_id": recording_id,
        "zarr_path": str(Path(zarr_path).resolve()),
        "source_sample_rate_hz": source_fps,
        "source_sample_rate_authority": fps_source,
        "tail_run_name": tail_name,
        "tail_run_path": tail_path,
        "tail_schema_id": tail_attrs["schema_id"],
        "tail_schema_version": tail_attrs["schema_version"],
        "tail_publication_manifest_sha256": publication.manifest.record_sha256,
        "tail_array_schema_manifest_sha256": array_schema["manifest_sha256"],
        "tail_array_schema_payload_sha256": array_schema["payload_sha256"],
        "tail_byte_planner_adopted": adopted,
        "tail_row_count": tail_rows,
        "source_tail_sample_count": tail_samples,
        "source_tail_sample_axis_sha256": tail_axis_record["content_sha256"],
        "subject_shape_run_name": shape_publication.run_path.rsplit("/", 1)[-1],
        "subject_shape_run_path": shape_publication.run_path,
        "subject_shape_schema_id": shape_attrs.get("schema_id"),
        "subject_shape_schema_version": shape_attrs.get("schema_version"),
        "subject_shape_publication_manifest_sha256": shape_publication.manifest.record_sha256,
        "body_frame_record_sha256": body_frame_sha,
        "reference_length_semantics_sha256": reference.semantics.record_sha256,
        "reference_length_content_sha256": reference_record["content_sha256"],
        "track_source_binding": track_source.binding,
        "track_identity_index": track_index.record,
        "selected_tail_arrays": {
            path: _array_manifest_record(publication, path, label="tail publication")
            for path in selected_tail_paths
        },
        "selected_subject_shape_arrays": {
            path: _array_manifest_record(
                shape_publication,
                path,
                label="subject-shape publication",
                array_ref_root=shape_publication.run_path,
            )
            for path in selected_shape_paths
        },
        "completion_snapshot": {
            "tail_status": tail_attrs.get("palette_run_completion_status"),
            "tail_completed_at_utc": tail_attrs.get("palette_run_completed_at_utc"),
            "tail_selector_eligible": tail_attrs.get("stage_selector_eligible"),
            "subject_shape_selector_eligible": shape_publication.selector_eligible,
            "track_status": track_source.binding["completion_snapshot"]["status"],
            "track_completed_at_utc": track_source.binding["completion_snapshot"][
                "completed_at_utc"
            ],
            "track_selector_eligible": track_source.binding["completion_snapshot"][
                "selector_eligible"
            ],
        },
    }
    binding = {**payload, "payload_sha256": canonical_json_sha256(payload)}
    return BoundTailTraceSources(
        binding=binding,
        tail_run=tail_run,
        subject_shape_run=shape_run,
        subject_shape_publication=shape_publication,
        track_source=track_source,
        track_index=track_index,
        tail_sample_s=tail_sample_s.copy(),
        reference_length_node=reference.array_node,
        reference_validity_node=reference.validity_node,
    )


class _ProjectedPayloadHasher:
    def __init__(self) -> None:
        self._hashers = {
            name: hashlib.sha256() for name in TAIL_TRACE_SCIENTIFIC_DTYPES
        }
        self._hashers["source_failure_reason"] = hashlib.sha256()
        self.row_count = 0

    def update(self, columns: Mapping[str, np.ndarray]) -> None:
        expected = {*TAIL_TRACE_SCIENTIFIC_DTYPES, "source_failure_reason"}
        if set(columns) != expected:
            raise ValueError("Tail projected payload column inventory changed.")
        lengths = {int(np.asarray(value).shape[0]) for value in columns.values()}
        if len(lengths) != 1:
            raise ValueError("Tail projected columns have unequal row counts.")
        count = lengths.pop()
        for name, dtype in TAIL_TRACE_SCIENTIFIC_DTYPES.items():
            values = np.asarray(columns[name])
            if values.ndim != 1 or values.shape != (count,):
                raise ValueError(f"{name}: tail projected column must be 1D.")
            encoded = values.astype(dtype, copy=False)
            self._hashers[name].update(np.ascontiguousarray(encoded).tobytes(order="C"))
        reasons = np.asarray(columns["source_failure_reason"], dtype=object)
        if reasons.shape != (count,):
            raise ValueError("source_failure_reason must be one text value per row.")
        for value in reasons.tolist():
            if not isinstance(value, str):
                raise ValueError("source_failure_reason must contain only text.")
            payload = value.encode("utf-8", errors="strict")
            self._hashers["source_failure_reason"].update(
                len(payload).to_bytes(8, "little", signed=False)
            )
            self._hashers["source_failure_reason"].update(payload)
        self.row_count += count

    def finish(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema_id": TAIL_TRACE_PROJECTED_PAYLOAD_SCHEMA_ID,
            "schema_version": TAIL_TRACE_PROJECTED_PAYLOAD_SCHEMA_VERSION,
            "row_count": self.row_count,
            "column_sha256": {
                name: self._hashers[name].hexdigest()
                for name in (*TAIL_TRACE_SCIENTIFIC_DTYPES, "source_failure_reason")
            },
        }
        return {**body, "payload_sha256": canonical_json_sha256(body)}


def tail_trace_parquet_policy(
    *,
    source_window_rows: int,
    source_rows_per_part: int,
    row_group_rows: int,
) -> dict[str, Any]:
    for name, value in (
        ("source_window_rows", source_window_rows),
        ("source_rows_per_part", source_rows_per_part),
        ("row_group_rows", row_group_rows),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive exact integer.")
    if source_window_rows > source_rows_per_part:
        raise ValueError("source_window_rows cannot exceed source_rows_per_part.")
    body: dict[str, Any] = {
        "schema_id": TAIL_TRACE_PARQUET_POLICY_SCHEMA_ID,
        "schema_version": TAIL_TRACE_PARQUET_POLICY_SCHEMA_VERSION,
        "compression": "zstd",
        "compression_level": 3,
        "dictionary_columns": [
            field.name
            for field in ARROW_TABLE_CONTRACTS[TAIL_TRACE_SAMPLES_TABLE].fields
            if field.arrow_type == "string"
        ],
        "source_window_rows": source_window_rows,
        "source_rows_per_part": source_rows_per_part,
        "row_group_rows": row_group_rows,
        "part_boundary_policy": "contiguous_source_tail_row_ranges_v1",
        "row_order": "source_tail_row_index_then_tail_sample_index",
    }
    return {**body, "payload_sha256": canonical_json_sha256(body)}


def _footer_metadata() -> dict[bytes, bytes]:
    return {
        b"palette.export_schema_id": EXPORT_SCHEMA_ID.encode("utf-8"),
        b"palette.export_schema_version": str(EXPORT_SCHEMA_VERSION).encode("ascii"),
        b"palette.table_contract": json.dumps(
            TABLE_CONTRACTS[TAIL_TRACE_SAMPLES_TABLE].to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    }


def _source_lineage_sha256(
    source_binding: Mapping[str, Any], projection: Mapping[str, Any]
) -> str:
    return canonical_json_sha256(
        {
            "source_binding_sha256": source_binding["payload_sha256"],
            "projection_contract_sha256": projection["payload_sha256"],
        }
    )


def tail_trace_constant_values(
    source: Mapping[str, Any], projection: Mapping[str, Any]
) -> dict[str, Any]:
    """Return exact standalone provenance constants for one tail projection."""

    track = source["track_source_binding"]
    return {
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "table_name": TAIL_TRACE_SAMPLES_TABLE,
        "recording_id": source["recording_id"],
        "zarr_path": source["zarr_path"],
        "source_lineage_hash": _source_lineage_sha256(source, projection),
        "source_tail_kinematics_run": source["tail_run_name"],
        "source_tail_kinematics_path": source["tail_run_path"],
        "source_tail_kinematics_schema_id": source["tail_schema_id"],
        "source_tail_kinematics_schema_version": source["tail_schema_version"],
        "source_tail_publication_manifest_sha256": source[
            "tail_publication_manifest_sha256"
        ],
        "source_subject_shape_run": source["subject_shape_run_name"],
        "source_subject_shape_path": source["subject_shape_run_path"],
        "source_subject_shape_schema_id": source["subject_shape_schema_id"],
        "source_subject_shape_schema_version": source["subject_shape_schema_version"],
        "source_subject_shape_publication_manifest_sha256": source[
            "subject_shape_publication_manifest_sha256"
        ],
        "source_track_kinematics_scope": track["scope"],
        "source_track_kinematics_run": track["run_name"],
        "source_track_kinematics_path": track["run_path"],
        "source_track_motion_manifest_sha256": track["source_manifest_sha256"],
        "source_binding_sha256": source["payload_sha256"],
        "projection_contract_sha256": projection["payload_sha256"],
        "source_sample_rate_hz": source["source_sample_rate_hz"],
        "source_tail_sample_count": source["source_tail_sample_count"],
        "source_tail_sample_axis_sha256": source["source_tail_sample_axis_sha256"],
        "body_frame_record_sha256": source["body_frame_record_sha256"],
        "reference_length_kind": projection["reference_length_kind"],
        "longitudinal_axis_convention": projection["longitudinal_axis_convention"],
        "lateral_axis_convention": projection["lateral_axis_convention"],
        "angle_convention": projection["angle_convention"],
        "curvature_convention": projection["curvature_convention"],
    }


def _arrow_batch(
    columns: Mapping[str, np.ndarray],
    *,
    source_binding: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> Any:
    import pyarrow as pa

    count = int(np.asarray(columns["source_tail_row_index"]).shape[0])
    constants = tail_trace_constant_values(source_binding, projection)
    schema = exact_arrow_schema(TAIL_TRACE_SAMPLES_TABLE, metadata=_footer_metadata())
    arrays = [
        pa.array(
            (
                columns[field.name]
                if field.name in columns
                else [constants[field.name]] * count
            ),
            type=field.type,
        )
        for field in schema
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def projected_tail_trace_sample_batch(
    bound: BoundTailTraceSources,
    *,
    start_row: int,
    stop_row: int,
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Read one bounded tail window with the exact standalone column roster."""

    columns = read_projected_tail_trace_window(
        bound,
        start_row=start_row,
        stop_row=stop_row,
    )
    constants = tail_trace_constant_values(bound.binding, projection)
    count = int(columns["source_tail_row_index"].shape[0])
    return {
        **{name: [value] * count for name, value in constants.items()},
        **columns,
    }


def _write_streaming_parts(
    bound: BoundTailTraceSources,
    *,
    table_root: Path,
    projection: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    row_count = int(bound.binding["tail_row_count"])
    part_rows = int(policy["source_rows_per_part"])
    window_rows = int(policy["source_window_rows"])
    row_group_rows = int(policy["row_group_rows"])
    dictionary_columns = list(policy["dictionary_columns"])
    table_root.mkdir(parents=True, exist_ok=False)
    spans = list(range(0, row_count, part_rows)) or [0]
    projected = _ProjectedPayloadHasher()
    receipts: list[dict[str, Any]] = []
    for part_index, part_start in enumerate(spans):
        part_stop = min(row_count, part_start + part_rows)
        part_path = table_root / f"part-{part_index:05d}.parquet"
        writer = pq.ParquetWriter(
            part_path,
            exact_arrow_schema(
                TAIL_TRACE_SAMPLES_TABLE,
                metadata=_footer_metadata(),
            ),
            compression="zstd",
            compression_level=3,
            use_dictionary=dictionary_columns,
        )
        written = 0
        try:
            if part_start == part_stop:
                writer.write_table(
                    _arrow_batch(
                        {
                            name: np.empty(0, dtype=dtype)
                            for name, dtype in TAIL_TRACE_SCIENTIFIC_DTYPES.items()
                        }
                        | {"source_failure_reason": np.empty(0, dtype=object)},
                        source_binding=bound.binding,
                        projection=projection,
                    )
                )
            for start in range(part_start, part_stop, window_rows):
                stop = min(part_stop, start + window_rows)
                columns = read_projected_tail_trace_window(
                    bound,
                    start_row=start,
                    stop_row=stop,
                )
                projected.update(columns)
                table = _arrow_batch(
                    columns,
                    source_binding=bound.binding,
                    projection=projection,
                )
                writer.write_table(table, row_group_size=row_group_rows)
                written += int(table.num_rows)
        finally:
            writer.close()
        receipts.append(
            {
                "scratch_path": part_path,
                "source_row_start": part_start,
                "source_row_stop": part_stop,
                "row_count": written,
            }
        )
    return projected.finish(), receipts


_SOURCE_BINDING_FIELDS = {
    "schema_id",
    "schema_version",
    "stage_id",
    "recording_id",
    "zarr_path",
    "source_sample_rate_hz",
    "source_sample_rate_authority",
    "tail_run_name",
    "tail_run_path",
    "tail_schema_id",
    "tail_schema_version",
    "tail_publication_manifest_sha256",
    "tail_array_schema_manifest_sha256",
    "tail_array_schema_payload_sha256",
    "tail_byte_planner_adopted",
    "tail_row_count",
    "source_tail_sample_count",
    "source_tail_sample_axis_sha256",
    "subject_shape_run_name",
    "subject_shape_run_path",
    "subject_shape_schema_id",
    "subject_shape_schema_version",
    "subject_shape_publication_manifest_sha256",
    "body_frame_record_sha256",
    "reference_length_semantics_sha256",
    "reference_length_content_sha256",
    "track_source_binding",
    "track_identity_index",
    "selected_tail_arrays",
    "selected_subject_shape_arrays",
    "completion_snapshot",
    "payload_sha256",
}


def _validate_source_binding(source: Mapping[str, Any]) -> None:
    if set(source) != _SOURCE_BINDING_FIELDS:
        raise ValueError("Tail source binding has an unexpected field set.")
    body = dict(source)
    digest = body.pop("payload_sha256", None)
    if digest != canonical_json_sha256(body):
        raise ValueError("Tail source-binding digest is invalid.")
    if (
        body.get("schema_id") != TAIL_TRACE_SOURCE_BINDING_SCHEMA_ID
        or body.get("schema_version") != TAIL_TRACE_SOURCE_BINDING_SCHEMA_VERSION
        or body.get("stage_id") != "tail_traces"
        or body.get("tail_schema_id") != TAIL_KINEMATICS_SCHEMA_ID
        or body.get("tail_schema_version") != TAIL_KINEMATICS_SCHEMA_VERSION
        or type(body.get("tail_byte_planner_adopted")) is not bool
    ):
        raise ValueError("Tail source-binding schema is invalid.")
    for name in (
        "recording_id",
        "zarr_path",
        "tail_run_name",
        "tail_run_path",
        "subject_shape_run_name",
        "subject_shape_run_path",
        "source_sample_rate_authority",
    ):
        if not isinstance(body.get(name), str) or not body[name]:
            raise ValueError(f"Tail source field {name} is invalid.")
    if (
        body["tail_run_path"]
        != f"analysis/tail_kinematics_runs/{body['tail_run_name']}"
    ):
        raise ValueError("Tail source run path is invalid.")
    if body["subject_shape_run_path"] != (
        f"analysis/subject_shape_runs/{body['subject_shape_run_name']}"
    ):
        raise ValueError("Subject-shape source run path is invalid.")
    if (
        not isinstance(body.get("subject_shape_schema_id"), str)
        or not body["subject_shape_schema_id"]
        or type(body.get("subject_shape_schema_version")) is not int
        or body["subject_shape_schema_version"] <= 0
    ):
        raise ValueError("Subject-shape source schema is invalid.")
    for name in (
        "tail_publication_manifest_sha256",
        "tail_array_schema_manifest_sha256",
        "tail_array_schema_payload_sha256",
        "source_tail_sample_axis_sha256",
        "subject_shape_publication_manifest_sha256",
        "body_frame_record_sha256",
        "reference_length_semantics_sha256",
        "reference_length_content_sha256",
    ):
        _exact_sha256(body.get(name), label=name)
    if (
        type(body.get("tail_row_count")) is not int
        or body["tail_row_count"] < 0
        or type(body.get("source_tail_sample_count")) is not int
        or body["source_tail_sample_count"] < 2
    ):
        raise ValueError("Tail source cardinalities are invalid.")
    rate = body.get("source_sample_rate_hz")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise ValueError("Tail source FPS is invalid.")
    if not math.isfinite(float(rate)) or float(rate) <= 0:
        raise ValueError("Tail source FPS must be positive and finite.")
    track = body.get("track_source_binding")
    if not isinstance(track, Mapping):
        raise ValueError("Tail track-source binding is invalid.")
    track_export._validate_source_binding(track)
    if float(track["source_sample_rate_hz"]) != float(rate):
        raise ValueError("Tail and track bindings disagree on source FPS.")
    index = body.get("track_identity_index")
    if not isinstance(index, Mapping):
        raise ValueError("Tail track identity index is invalid.")
    index_body = dict(index)
    index_digest = index_body.pop("payload_sha256", None)
    if (
        index_digest != canonical_json_sha256(index_body)
        or index_body.get("schema_id")
        != "palette.tail_trace_samples.track_identity_index"
        or index_body.get("schema_version") != 1
        or index_body.get("row_count") != body["tail_row_count"]
    ):
        raise ValueError("Tail track identity index receipt is invalid.")
    for name in (
        "instance_key_sha256",
        "track_id_sha256",
        "source_acquisition_frame_index_sha256",
    ):
        _exact_sha256(index_body.get(name), label=f"track index {name}")
    if set(body.get("selected_tail_arrays", {})) != set(_TAIL_WINDOW_ARRAYS):
        raise ValueError("Tail selected-array binding is incomplete.")
    if set(body.get("selected_subject_shape_arrays", {})) != set(_SHAPE_WINDOW_ARRAYS):
        raise ValueError("Subject-shape selected-array binding is incomplete.")
    for inventory_name, declarations, expected_fields, array_ref_root in (
        (
            "selected_tail_arrays",
            _TAIL_WINDOW_ARRAYS,
            _ARRAY_RECORD_FIELDS,
            None,
        ),
        (
            "selected_subject_shape_arrays",
            _SHAPE_WINDOW_ARRAYS,
            _SUBJECT_SHAPE_ARRAY_RECORD_FIELDS,
            body["subject_shape_run_path"],
        ),
    ):
        inventory = body[inventory_name]
        for path, (dtype, shape_spec) in declarations.items():
            record = inventory[path]
            expected_shape = [
                (
                    body["tail_row_count"]
                    if value == "rows"
                    else (
                        body["source_tail_sample_count"]
                        if value == "samples"
                        else int(value)
                    )
                )
                for value in shape_spec
            ]
            if (
                not isinstance(record, Mapping)
                or set(record) != expected_fields
                or record.get("relative_ref") != path
                or record.get("dtype") != dtype.str
                or record.get("shape") != expected_shape
                or record.get("canonicalization") != ARRAY_PAYLOAD_CANONICALIZATION
            ):
                raise ValueError(f"{inventory_name} record {path!r} is invalid.")
            if array_ref_root is not None and record.get("array_ref") != (
                f"/{array_ref_root.strip('/')}/{path}"
            ):
                raise ValueError(f"{inventory_name} record {path!r} is invalid.")
            _exact_sha256(record.get("content_sha256"), label=f"{path} content")
    completion = body.get("completion_snapshot")
    if not isinstance(completion, Mapping) or completion != {
        "tail_status": "complete",
        "tail_completed_at_utc": completion.get("tail_completed_at_utc"),
        "tail_selector_eligible": True,
        "subject_shape_selector_eligible": True,
        "track_status": "complete",
        "track_completed_at_utc": completion.get("track_completed_at_utc"),
        "track_selector_eligible": True,
    }:
        raise ValueError("Tail completion snapshot is invalid.")
    for name in ("tail_completed_at_utc", "track_completed_at_utc"):
        if not isinstance(completion.get(name), str) or not completion[name]:
            raise ValueError(f"Tail completion field {name} is invalid.")


def _validate_tail_envelope(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    envelope = payload.get("tail_trace_export")
    required = {
        "schema_id",
        "schema_version",
        "source_binding",
        "projection_contract",
        "projected_payload",
        "parquet_policy",
        "payload_sha256",
    }
    if not isinstance(envelope, Mapping) or set(envelope) != required:
        raise ValueError("Tail export envelope has an unexpected field set.")
    body = {name: envelope[name] for name in required - {"payload_sha256"}}
    if envelope.get("payload_sha256") != canonical_json_sha256(body):
        raise ValueError("Tail export envelope digest is invalid.")
    if (
        envelope.get("schema_id") != TAIL_TRACE_EXPORT_SCHEMA_ID
        or envelope.get("schema_version") != TAIL_TRACE_EXPORT_SCHEMA_VERSION
    ):
        raise ValueError("Tail export envelope schema is invalid.")
    source = envelope["source_binding"]
    if not isinstance(source, Mapping):
        raise ValueError("Tail source binding is invalid.")
    _validate_source_binding(source)
    projection = envelope["projection_contract"]
    if (
        not isinstance(projection, Mapping)
        or dict(projection) != tail_trace_projection_contract()
    ):
        raise ValueError("Tail projection differs from the installed contract.")
    projected = envelope["projected_payload"]
    if not isinstance(projected, Mapping):
        raise ValueError("Tail projected-payload receipt is invalid.")
    projected_body = dict(projected)
    projected_digest = projected_body.pop("payload_sha256", None)
    if (
        projected_digest != canonical_json_sha256(projected_body)
        or projected_body.get("schema_id") != TAIL_TRACE_PROJECTED_PAYLOAD_SCHEMA_ID
        or projected_body.get("schema_version")
        != TAIL_TRACE_PROJECTED_PAYLOAD_SCHEMA_VERSION
        or projected_body.get("row_count")
        != source["tail_row_count"] * source["source_tail_sample_count"]
        or set(projected_body.get("column_sha256", {}))
        != {*TAIL_TRACE_SCIENTIFIC_DTYPES, "source_failure_reason"}
    ):
        raise ValueError("Tail projected-payload receipt is invalid.")
    for digest in projected_body["column_sha256"].values():
        _exact_sha256(digest, label="tail projected column")
    policy = envelope["parquet_policy"]
    if not isinstance(policy, Mapping):
        raise ValueError("Tail Parquet policy is invalid.")
    expected_policy = tail_trace_parquet_policy(
        source_window_rows=policy.get("source_window_rows"),
        source_rows_per_part=policy.get("source_rows_per_part"),
        row_group_rows=policy.get("row_group_rows"),
    )
    if dict(policy) != expected_policy:
        raise ValueError("Tail Parquet policy differs from the installed contract.")
    return envelope


def _decoded_part_validation(
    parts: list[Path],
    *,
    source: Mapping[str, Any],
    projection: Mapping[str, Any],
    expected_payload: Mapping[str, Any],
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    hasher = _ProjectedPayloadHasher()
    constants = tail_trace_constant_values(source, projection)
    sample_count = int(source["source_tail_sample_count"])
    expected_axis = np.full(sample_count, np.nan, dtype=np.float32)
    axis_seen = np.zeros(sample_count, dtype=bool)
    seen = 0
    scientific_float_names = (
        "reference_length_px",
        "body_longitudinal_fraction",
        "body_lateral_fraction",
        "tangent_angle_rad",
        "body_curvature_dimensionless",
        "source_camera_x_px",
        "source_camera_y_px",
        "source_camera_curvature_px_inv",
        "source_lateral_deflection_px",
    )
    for part in parts:
        parquet = pq.ParquetFile(part)
        validate_arrow_schema(TAIL_TRACE_SAMPLES_TABLE, parquet.schema_arrow)
        for batch in parquet.iter_batches():
            table = batch.to_pydict()
            count = len(table["source_tail_row_index"])
            dynamic = {
                name: np.asarray(table[name], dtype=dtype)
                for name, dtype in TAIL_TRACE_SCIENTIFIC_DTYPES.items()
            }
            dynamic["source_failure_reason"] = np.asarray(
                table["source_failure_reason"], dtype=object
            )
            flat = np.arange(seen, seen + count, dtype=np.int64)
            expected_rows = flat // sample_count
            expected_samples = (flat % sample_count).astype(np.int32)
            if not np.array_equal(dynamic["source_tail_row_index"], expected_rows):
                raise ValueError("Tail Parquet source-row order is not contiguous.")
            if not np.array_equal(dynamic["tail_sample_index"], expected_samples):
                raise ValueError("Tail Parquet sample order is not contiguous.")
            axis_values = dynamic["normalized_tail_position"]
            for sample_index in range(sample_count):
                selected = expected_samples == sample_index
                if not np.any(selected):
                    continue
                observed_values = axis_values[selected]
                if not axis_seen[sample_index]:
                    expected_axis[sample_index] = observed_values[0]
                    axis_seen[sample_index] = True
                if not np.all(observed_values == expected_axis[sample_index]):
                    raise ValueError(
                        "Tail normalized sample axis changes between rows."
                    )
            expected_time = dynamic["source_acquisition_frame_index"].astype(
                np.float64
            ) / float(source["source_sample_rate_hz"])
            if not np.array_equal(dynamic["time_seconds"], expected_time):
                raise ValueError("Tail Parquet time differs from frame/FPS authority.")
            reason = dynamic["sample_reason_code"]
            source_valid = dynamic["source_tail_row_valid"]
            reference_valid = dynamic["reference_length_valid"]
            sample_valid = dynamic["sample_valid"]
            expected_validity = (
                ((reason == 0) & source_valid & reference_valid & sample_valid)
                | ((reason == 1) & ~source_valid & ~sample_valid)
                | ((reason == 2) & source_valid & ~reference_valid & ~sample_valid)
                | ((reason == 3) & source_valid & reference_valid & ~sample_valid)
            )
            if not np.all(expected_validity):
                raise ValueError(
                    "Tail Parquet validity/reason registry is inconsistent."
                )
            floats = np.column_stack([dynamic[name] for name in scientific_float_names])
            if np.any(~np.all(np.isfinite(floats[sample_valid]), axis=1)) or np.any(
                ~np.all(np.isnan(floats[~sample_valid]), axis=1)
            ):
                raise ValueError(
                    "Tail Parquet valid/invalid float semantics are inconsistent."
                )
            reasons = dynamic["source_failure_reason"]
            if any(
                not isinstance(value, str) or not value for value in reasons.tolist()
            ):
                raise ValueError("Tail Parquet source failure reasons are invalid.")
            if np.any(source_valid & (reasons != "ok")) or np.any(
                (~source_valid) & (reasons == "ok")
            ):
                raise ValueError("Tail source validity and failure reason disagree.")
            for name, expected in constants.items():
                if any(value != expected for value in table[name]):
                    raise ValueError(f"Tail Parquet constant field {name} changed.")
            hasher.update(dynamic)
            seen += count
    observed = hasher.finish()
    if observed["row_count"] and (
        not np.all(axis_seen)
        or array_values_sha256(expected_axis)
        != source["source_tail_sample_axis_sha256"]
    ):
        raise ValueError("Tail Parquet normalized axis differs from source.")
    if observed != expected_payload:
        raise ValueError("Tail decoded payload differs from its projected receipt.")
    return observed


def validate_tail_trace_export_payload(
    export_root: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    envelope = _validate_tail_envelope(payload)
    parts = manifest_selected_part_files_from_payload(
        export_root,
        payload,
        TAIL_TRACE_SAMPLES_TABLE,
        allow_legacy_layout=False,
    )
    source = envelope["source_binding"]
    projection = envelope["projection_contract"]
    projected = envelope["projected_payload"]
    assert isinstance(source, Mapping)
    assert isinstance(projection, Mapping)
    assert isinstance(projected, Mapping)
    observed = _decoded_part_validation(
        parts,
        source=source,
        projection=projection,
        expected_payload=projected,
    )
    return {
        "valid": True,
        "row_count": observed["row_count"],
        "part_count": len(parts),
        "projected_payload_sha256": observed["payload_sha256"],
        "source_binding_sha256": source["payload_sha256"],
    }


def _path_is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def export_tail_trace_samples(
    zarr_path: str | Path,
    *,
    tail_kinematics_run: str,
    subject_shape_run: str,
    track_kinematics_run: str,
    track_scope: str,
    output_root: str | Path,
    export_run_id: str,
    scratch_root: str | Path,
    source_window_rows: int = 16_384,
    source_rows_per_part: int = 131_072,
    row_group_rows: int = 65_536,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Publish one exact selector-ineligible long-form tail query product."""

    source_path = Path(zarr_path).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    if (
        destination == scratch
        or _path_is_within(destination, scratch)
        or _path_is_within(scratch, destination)
    ):
        raise ValueError("Export and scratch roots must not overlap.")
    if _path_is_within(destination, source_path) or _path_is_within(
        scratch, source_path
    ):
        raise ValueError(
            "Export and scratch roots must not be inside the source archive."
        )
    run_id = safe_component(export_run_id, label="export run ID")
    tail_run = safe_component(tail_kinematics_run, label="tail-kinematics run ID")
    shape_run = safe_component(subject_shape_run, label="subject-shape run ID")
    track_run = safe_component(track_kinematics_run, label="track-kinematics run ID")
    if track_scope not in {"online", "offline"}:
        raise ValueError("track_scope must be 'online' or 'offline'.")
    policy = tail_trace_parquet_policy(
        source_window_rows=source_window_rows,
        source_rows_per_part=source_rows_per_part,
        row_group_rows=row_group_rows,
    )
    baseline = manifest_identity(export_manifest_path(destination, run_id))
    if baseline is not None and not overwrite:
        raise FileExistsError(
            f"Export manifest already exists: {export_manifest_path(destination, run_id)}"
        )
    runtime = ExportRuntimePhaseRecorder()
    with runtime.measure("source_binding_before"):
        root = open_zarr_root(source_path, mode="r")
        before = bind_tail_trace_sources(
            root,
            zarr_path=source_path,
            tail_kinematics_run=tail_run,
            subject_shape_run=shape_run,
            track_kinematics_run=track_run,
            track_scope=track_scope,
            source_window_rows=source_window_rows,
        )
    projection = tail_trace_projection_contract()
    generation_id = uuid.uuid4().hex
    final_generation_path = generation_relative_path(run_id, generation_id)
    staging = publication_staging_root(destination, run_id, generation_id)
    final_generation = publication_generation_root(destination, run_id, generation_id)
    scratch_generation = scratch / f"palette_tail_trace_{run_id}_{generation_id}"
    if staging.exists() or final_generation.exists() or scratch_generation.exists():
        raise FileExistsError("Tail export generation identity already exists.")
    try:
        with runtime.measure("scratch_parquet_write"):
            projected, scratch_receipts = _write_streaming_parts(
                before,
                table_root=scratch_generation / "tables" / TAIL_TRACE_SAMPLES_TABLE,
                projection=projection,
                policy=policy,
            )
        with runtime.measure("source_binding_after"):
            after = bind_tail_trace_sources(
                open_zarr_root(source_path, mode="r"),
                zarr_path=source_path,
                tail_kinematics_run=tail_run,
                subject_shape_run=shape_run,
                track_kinematics_run=track_run,
                track_scope=track_scope,
                source_window_rows=source_window_rows,
            )
            if after.binding != before.binding:
                raise RuntimeError(
                    "Tail, subject-shape, or track source binding changed during "
                    "extraction."
                )
        staged_table = staging / "tables" / TAIL_TRACE_SAMPLES_TABLE
        inventory_entries: list[dict[str, Any]] = []
        relative_parts: list[str] = []
        staged_parts: list[Path] = []
        with runtime.measure("scratch_to_staging_copy"):
            staged_table.mkdir(parents=True, exist_ok=False)
            for receipt in scratch_receipts:
                scratch_part = Path(receipt["scratch_path"])
                staged_part = staged_table / scratch_part.name
                shutil.copy2(scratch_part, staged_part)
                staged_sha256 = sha256_file(staged_part)
                if staged_sha256 != sha256_file(scratch_part):
                    raise RuntimeError(
                        "Tail scratch-to-publication copy digest mismatch."
                    )
                relative = (
                    final_generation_path
                    / "tables"
                    / TAIL_TRACE_SAMPLES_TABLE
                    / staged_part.name
                ).as_posix()
                relative_parts.append(relative)
                staged_parts.append(staged_part)
                inventory_entries.append(
                    {
                        "path": relative,
                        "sha256": staged_sha256,
                        "size_bytes": int(staged_part.stat().st_size),
                        "row_count": int(receipt["row_count"]),
                    }
                )
        columns = tuple(
            field.name
            for field in ARROW_TABLE_CONTRACTS[TAIL_TRACE_SAMPLES_TABLE].fields
        )
        capability_statuses = resolve_capabilities({TAIL_TRACE_SAMPLES_TABLE: columns})
        envelope_body: dict[str, Any] = {
            "schema_id": TAIL_TRACE_EXPORT_SCHEMA_ID,
            "schema_version": TAIL_TRACE_EXPORT_SCHEMA_VERSION,
            "source_binding": before.binding,
            "projection_contract": projection,
            "projected_payload": projected,
            "parquet_policy": policy,
        }
        envelope = {
            **envelope_body,
            "payload_sha256": canonical_json_sha256(envelope_body),
        }
        git = get_git_info(Path(__file__).resolve().parents[3])
        manifest: dict[str, Any] = {
            "export_run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "schema_id": EXPORT_SCHEMA_ID,
            "schema_version": EXPORT_SCHEMA_VERSION,
            "tool": "fisheye.analytics_exports.tail_trace_samples",
            "hostname": socket.gethostname(),
            "palette_git_commit": git.get("commit_hash"),
            "palette_git_dirty": git.get("is_dirty"),
            "source_recording_count": 1,
            "source_zarrs": [str(source_path)],
            "tables_requested": [TAIL_TRACE_SAMPLES_TABLE],
            "table_contracts": contract_snapshot((TAIL_TRACE_SAMPLES_TABLE,)),
            "arrow_schema_contracts": arrow_contract_envelope(
                (TAIL_TRACE_SAMPLES_TABLE,)
            ),
            "capabilities": [
                item.capability_id for item in capability_statuses if item.available
            ],
            "capability_statuses": [item.to_dict() for item in capability_statuses],
            "row_counts_by_table": {
                TAIL_TRACE_SAMPLES_TABLE: int(projected["row_count"])
            },
            "part_files_by_table": {TAIL_TRACE_SAMPLES_TABLE: relative_parts},
            "publication": {
                "schema_id": PUBLICATION_SCHEMA_ID,
                "schema_version": PUBLICATION_SCHEMA_VERSION,
                "state": "complete",
                "generation_id": generation_id,
                "generation_path": final_generation_path.as_posix(),
                "parts_by_table": {TAIL_TRACE_SAMPLES_TABLE: inventory_entries},
            },
            "diagnostics": [],
            "collection_manifest": None,
            "export_parameters": {
                "registry_indexing": False,
                "selector_activation": False,
                "source_mutation": False,
                "scratch_root": str(scratch),
                "overwrite": bool(overwrite),
            },
            "tail_trace_export": envelope,
        }
        with runtime.measure("staged_decoded_validation"):
            _decoded_part_validation(
                staged_parts,
                source=before.binding,
                projection=projection,
                expected_payload=projected,
            )
        with runtime.measure("manifest_validation"):
            _validate_tail_envelope(manifest)
        committed = commit_staged_publication(
            destination,
            staging,
            manifest,
            baseline_manifest_identity=baseline,
            runtime_recorder=runtime,
        )
        with runtime.measure("published_payload_validation"):
            published = json.loads(committed.read_text(encoding="utf-8"))
            validation = validate_tail_trace_export_payload(destination, published)
        return {
            **published,
            "manifest_path": str(committed),
            "tail_trace_validation": validation,
            "runtime_telemetry": runtime.snapshot(),
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        if scratch_generation.exists():
            shutil.rmtree(scratch_generation)


__all__ = [
    "BoundTailTraceSources",
    "TAIL_TRACE_EXPORT_SCHEMA_ID",
    "TAIL_TRACE_EXPORT_SCHEMA_VERSION",
    "TAIL_TRACE_PROJECTION_SCHEMA_ID",
    "TAIL_TRACE_PROJECTION_SCHEMA_VERSION",
    "TAIL_TRACE_PROJECTED_PAYLOAD_SCHEMA_ID",
    "TAIL_TRACE_PROJECTED_PAYLOAD_SCHEMA_VERSION",
    "TAIL_TRACE_REASON_GEOMETRY_NONFINITE",
    "TAIL_TRACE_REASON_REFERENCE_INVALID",
    "TAIL_TRACE_REASON_SOURCE_INVALID",
    "TAIL_TRACE_REASON_VALID",
    "TAIL_TRACE_SCIENTIFIC_DTYPES",
    "TailTrackIdentityIndex",
    "bind_tail_trace_sources",
    "export_tail_trace_samples",
    "projected_tail_trace_sample_batch",
    "project_tail_trace_window",
    "read_projected_tail_trace_window",
    "tail_trace_parquet_policy",
    "tail_trace_constant_values",
    "tail_trace_projection_contract",
    "validate_tail_trace_export_payload",
]
