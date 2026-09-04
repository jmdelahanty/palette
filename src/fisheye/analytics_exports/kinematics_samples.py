"""Exact bounded export of generic multi-track kinematic samples.

The recording-local track-kinematics Zarr publication remains scientific
authority.  This module emits one selector-ineligible, manifest-selected
Parquet query product.  It binds the completed source publication metadata,
validates the selected live array declarations, and reads payloads only in
bounded first-axis windows; it deliberately does not rehash all 69 source
surfaces during export.
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

from fisheye.analysis.track_kinematics import (
    TRACK_KINEMATICS_RUN_SCHEMA_ID,
    TRACK_KINEMATICS_RUN_SCHEMA_VERSION,
    TRACK_MOTION_PUBLICATION_COMMIT_ATTR,
    TRACK_MOTION_PUBLICATION_COMMIT_SCHEMA_ID,
    TRACK_MOTION_PUBLICATION_COMMIT_SCHEMA_VERSION,
    TRACK_MOTION_PUBLICATION_COMMIT_SCHEMA_VERSION_V2,
    TRACK_MOTION_PUBLICATION_MANIFEST_ATTR,
    TRACK_MOTION_PUBLICATION_MANIFEST_DIGEST_ATTR,
    TRACK_MOTION_PUBLICATION_MANIFEST_SCHEMA_ID,
    TRACK_MOTION_PUBLICATION_MANIFEST_SCHEMA_VERSION,
    TRACK_MOTION_PUBLICATION_MANIFEST_SCHEMA_VERSION_V2,
)
from fisheye.analysis.track_kinematics_io import resolve_track_kinematics_run
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
    KINEMATICS_SAMPLES_TABLE,
    TABLE_CONTRACTS,
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
from fisheye.shared.coordinate_identity import (
    TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE,
)
from fisheye.shared.coordinate_frame_record import ARRAY_PAYLOAD_CANONICALIZATION
from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr_io import open_zarr_root

KINEMATICS_EXPORT_SCHEMA_ID = "palette.analytics_export.kinematics_samples"
KINEMATICS_EXPORT_SCHEMA_VERSION = 1
KINEMATICS_SOURCE_BINDING_SCHEMA_ID = "palette.kinematics_samples.source_binding"
KINEMATICS_SOURCE_BINDING_SCHEMA_VERSION = 1
KINEMATICS_PROJECTION_SCHEMA_ID = "palette.kinematics_samples.projection"
KINEMATICS_PROJECTION_SCHEMA_VERSION = 1
KINEMATICS_PROJECTION_SCHEMA_VERSION_V2 = 2
KINEMATICS_PROJECTED_PAYLOAD_SCHEMA_ID = "palette.kinematics_samples.projected_payload"
KINEMATICS_PROJECTED_PAYLOAD_SCHEMA_VERSION = 1
KINEMATICS_PARQUET_POLICY_SCHEMA_ID = "palette.kinematics_samples.parquet_policy"
KINEMATICS_PARQUET_POLICY_SCHEMA_VERSION = 1

KINEMATICS_SPEED_LEVEL = "filtered"
KINEMATICS_SAMPLING_POLICY = "global_acquisition_frame_modulo_stride_v1"
KINEMATICS_FRAME_SELECTION_POLICY = "half_open_acquisition_frame_range_v1"
KINEMATICS_POSITION_SPACE = "physical_mm"

_SELECTED_SURFACES = (
    "track_sample_key",
    "source_acquisition_frame_index",
    "time_seconds",
    "source_row_index",
    "source_instance_key",
    "detection_source",
    "positions_mm",
    "movement/speed/filtered/mm",
    "movement/speed/filtered/frame_path_distance_mm",
    "heading_degrees",
    "smoothed_heading_degrees",
    "angular_velocity_smoothed_deg_s",
    "source_observed",
    "sample_observed",
    "position_finite",
    "heading_usable",
    "sample_valid",
    "transition_valid",
    "sample_reason_code",
    "transition_reason_code",
)
_PHYSICAL_SURFACES = frozenset(
    {
        "positions_mm",
        "movement/speed/filtered/mm",
        "movement/speed/filtered/frame_path_distance_mm",
    }
)
_TRANSITION_SURFACES = frozenset(
    {
        "angular_velocity_smoothed_deg_s",
        "transition_valid",
        "transition_reason_code",
    }
)
_SOURCE_DTYPES: Mapping[str, np.dtype[Any]] = {
    "track_sample_key": np.dtype("<i8"),
    "source_acquisition_frame_index": np.dtype("<i8"),
    "time_seconds": np.dtype("<f4"),
    "source_row_index": np.dtype("<i8"),
    "source_instance_key": TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE,
    "detection_source": np.dtype("i1"),
    "positions_mm": np.dtype("<f4"),
    "movement/speed/filtered/mm": np.dtype("<f4"),
    "movement/speed/filtered/frame_path_distance_mm": np.dtype("<f4"),
    "heading_degrees": np.dtype("<f4"),
    "smoothed_heading_degrees": np.dtype("<f4"),
    "angular_velocity_smoothed_deg_s": np.dtype("<f4"),
    "source_observed": np.dtype("bool"),
    "sample_observed": np.dtype("bool"),
    "position_finite": np.dtype("bool"),
    "heading_usable": np.dtype("bool"),
    "sample_valid": np.dtype("bool"),
    "transition_valid": np.dtype("bool"),
    "sample_reason_code": np.dtype("<i2"),
    "transition_reason_code": np.dtype("<i2"),
}
_SOURCE_TRAILING_SHAPES: Mapping[str, tuple[int, ...]] = {
    **{name: () for name in _SELECTED_SURFACES},
    "track_sample_key": (2,),
    "positions_mm": (2,),
}

KINEMATICS_SCIENTIFIC_DTYPES: Mapping[str, str] = {
    "track_id": "int64",
    "track_sample_index": "int64",
    "source_acquisition_frame_index": "int64",
    "time_seconds": "float32",
    "source_row_index": "int64",
    "source_instance_key_valid": "bool",
    "source_instance_key": "uint64",
    "detection_source": "int8",
    "position_x_mm": "float32",
    "position_y_mm": "float32",
    "speed_mm_s": "float32",
    "frame_path_distance_mm": "float32",
    "motion_heading_degrees": "float32",
    "smoothed_motion_heading_degrees": "float32",
    "smoothed_angular_velocity_deg_s": "float32",
    "source_observed": "bool",
    "sample_observed": "bool",
    "position_finite": "bool",
    "heading_usable": "bool",
    "sample_valid": "bool",
    "transition_valid": "bool",
    "sample_reason_code": "int16",
    "transition_reason_code": "int16",
}
_NUMPY_DTYPES: Mapping[str, np.dtype[Any]] = {
    "int64": np.dtype("<i8"),
    "float64": np.dtype("<f8"),
    "float32": np.dtype("<f4"),
    "uint64": np.dtype("<u8"),
    "int16": np.dtype("<i2"),
    "int8": np.dtype("i1"),
    "bool": np.dtype("u1"),
}

_MANIFEST_V1_FIELDS = {
    "schema_id",
    "schema_version",
    "run_ref",
    "run_type",
    "run_name",
    "coordinate_binding_status",
    "source_authority",
    "input_authority",
    "physical_authority",
    "run_derivation",
    "run_root_attrs",
    "run_group_inventory",
    "tracks_group_inventory",
    "run_arrays",
    "track_count",
    "tracks",
}
_TRACK_RECORD_FIELDS = {
    "track_id",
    "track_ref",
    "track_sample_count",
    "second_bin_count",
    "row_identity_ref",
    "row_identity_sha256",
    "track_time_lineage_ref",
    "track_time_lineage_sha256",
    "position_derivation_ref",
    "position_derivation_sha256",
    "groups",
    "surfaces",
}
_SURFACE_FIELDS = {
    "relative_ref",
    "dtype",
    "dtype_fields",
    "itemsize",
    "shape",
    "content_sha256",
    "attrs",
    "attrs_sha256",
    "authority_scope",
    "axis0_domain",
    "units",
    "semantic_profile",
    "operation_id",
    "input_refs",
    "alias_of",
    "axis0_identity",
}
_PHYSICAL_SURFACE_FIELDS = {
    "pixel_source_ref",
    "physical_authority_sha256",
    "physical_value_comparison",
}
_SOURCE_BINDING_FIELDS = {
    "schema_id",
    "schema_version",
    "stage_id",
    "recording_id",
    "zarr_path",
    "scope",
    "run_name",
    "run_path",
    "source_schema_id",
    "source_schema_version",
    "source_manifest_schema_id",
    "source_manifest_schema_version",
    "source_manifest_sha256",
    "source_publication_commit_sha256",
    "source_sample_rate_hz",
    "position_coordinate_space",
    "position_coordinate_descriptor_sha256",
    "physical_authority_sha256",
    "selection_snapshot",
    "completion_snapshot",
    "track_count",
    "tracks",
    "payload_sha256",
}
_SELECTION_SNAPSHOT_FIELDS = {
    "mode",
    "parent_latest",
    "parent_latest_complete",
    "parent_latest_scope",
    "scope_latest",
    "parent_completion_epoch",
    "scope_completion_epoch",
}
_COMPLETION_SNAPSHOT_FIELDS = {
    "status",
    "completed_at_utc",
    "selector_eligible",
}
_BOUND_TRACK_FIELDS = {
    "track_id",
    "track_ref",
    "sample_count",
    "track_record_sha256",
    "selected_surfaces",
}
_BOUND_SURFACE_FIELDS = {
    "relative_ref",
    "dtype",
    "dtype_fields",
    "itemsize",
    "shape",
    "content_sha256",
    "attrs_sha256",
    "record_sha256",
}
_PROJECTED_PAYLOAD_FIELDS = {
    "schema_id",
    "schema_version",
    "row_count",
    "column_sha256",
    "payload_sha256",
}
_SHA256_LENGTH = 64


@dataclass(frozen=True)
class BoundKinematicsSamplesSource:
    """Strictly validated track source used by standalone and cohort exports."""

    binding: Mapping[str, Any]
    run_group: Any


# Keep the private spelling for internal type annotations while exposing one
# supported source interface to sibling export profiles.
_BoundSource = BoundKinematicsSamplesSource


def _recording_id(path: Path) -> str:
    name = path.name.removesuffix(".zarr").removesuffix("_analysis")
    if not name:
        raise ValueError("Cannot derive recording ID from an empty archive name.")
    return name


def _json_object(value: object, *, label: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} is not strict JSON.") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be one exact JSON object.")
    normalized = json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))
    if not isinstance(normalized, dict):  # pragma: no cover
        raise TypeError(f"{label} did not normalize to an object.")
    return normalized


def _sha256_text(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} must be one lowercase SHA-256 digest.")
    return text


def _group_attrs(group: Any) -> dict[str, Any]:
    attrs = getattr(group, "attrs", {})
    return dict(attrs.asdict() if hasattr(attrs, "asdict") else dict(attrs))


def _child(group: Any, path: str) -> Any:
    node = group
    for part in str(path).strip("/").split("/"):
        if part:
            node = node[part]
    return node


def _dtype_fields(dtype: np.dtype[Any]) -> list[dict[str, Any]] | None:
    if dtype.fields is None:
        return None
    return [
        {
            "name": str(name),
            "dtype": np.dtype(field[0]).str,
            "offset": int(field[1]),
        }
        for name, field in dtype.fields.items()
    ]


def _expected_publication_commit(manifest: Mapping[str, Any]) -> dict[str, Any]:
    tracks = manifest.get("tracks")
    source = manifest.get("source_authority")
    derivation = manifest.get("run_derivation")
    input_authority = manifest.get("input_authority")
    if not all(
        isinstance(value, Mapping)
        for value in (tracks, source, derivation, input_authority)
    ):
        raise ValueError("Track-motion manifest cannot mint a publication commit.")
    version = manifest.get("schema_version")
    if version not in {
        TRACK_MOTION_PUBLICATION_MANIFEST_SCHEMA_VERSION,
        TRACK_MOTION_PUBLICATION_MANIFEST_SCHEMA_VERSION_V2,
    }:
        raise ValueError("Track-motion manifest schema version is unsupported.")
    commit_version = (
        TRACK_MOTION_PUBLICATION_COMMIT_SCHEMA_VERSION
        if version == TRACK_MOTION_PUBLICATION_MANIFEST_SCHEMA_VERSION
        else TRACK_MOTION_PUBLICATION_COMMIT_SCHEMA_VERSION_V2
    )
    position_derivations = {
        str(name): {
            "record_ref": record.get("position_derivation_ref"),
            "record_sha256": record.get("position_derivation_sha256"),
        }
        for name, record in tracks.items()
        if isinstance(record, Mapping)
    }
    commit: dict[str, Any] = {
        "schema_id": TRACK_MOTION_PUBLICATION_COMMIT_SCHEMA_ID,
        "schema_version": commit_version,
        "run_ref": manifest.get("run_ref"),
        "manifest_sha256": canonical_json_sha256(manifest),
        "source_authority_sha256": canonical_json_sha256(source),
        "input_authority_sha256": input_authority.get("record_sha256"),
        "run_derivation_sha256": derivation.get("record_sha256"),
        "position_derivations": position_derivations,
    }
    if commit_version == TRACK_MOTION_PUBLICATION_COMMIT_SCHEMA_VERSION_V2:
        commit["manifest_schema_version"] = version
    return _json_object(commit, label="expected track-motion publication commit")


def _sampling_stride(source_rate_hz: float, requested_rate_hz: float) -> int:
    if not math.isfinite(source_rate_hz) or source_rate_hz <= 0:
        raise ValueError("source sample rate must be positive and finite.")
    if not math.isfinite(requested_rate_hz) or requested_rate_hz <= 0:
        raise ValueError("requested sample rate must be positive and finite.")
    return max(1, int(math.floor((source_rate_hz / requested_rate_hz) + 0.5)))


def kinematics_projection_contract(
    *,
    source_sample_rate_hz: float,
    requested_sample_rate_hz: float,
    source_frame_start: int | None = None,
    source_frame_stop_exclusive: int | None = None,
) -> dict[str, Any]:
    """Return the closed acquisition-frame-aligned sample projection."""

    source_rate = float(source_sample_rate_hz)
    requested_rate = float(requested_sample_rate_hz)
    stride = _sampling_stride(source_rate, requested_rate)
    if (source_frame_start is None) != (source_frame_stop_exclusive is None):
        raise ValueError(
            "source frame start and stop must either both be omitted or both be set."
        )
    frame_range: tuple[int, int] | None = None
    if source_frame_start is not None and source_frame_stop_exclusive is not None:
        if (
            isinstance(source_frame_start, bool)
            or type(source_frame_start) is not int
            or source_frame_start < 0
            or isinstance(source_frame_stop_exclusive, bool)
            or type(source_frame_stop_exclusive) is not int
            or source_frame_stop_exclusive <= source_frame_start
        ):
            raise ValueError(
                "source frame range must be one nonempty half-open interval of "
                "nonnegative exact integers."
            )
        frame_range = (source_frame_start, source_frame_stop_exclusive)
    payload: dict[str, Any] = {
        "schema_id": KINEMATICS_PROJECTION_SCHEMA_ID,
        "schema_version": (
            KINEMATICS_PROJECTION_SCHEMA_VERSION
            if frame_range is None
            else KINEMATICS_PROJECTION_SCHEMA_VERSION_V2
        ),
        "table_name": KINEMATICS_SAMPLES_TABLE,
        "source_speed_level": KINEMATICS_SPEED_LEVEL,
        "source_sample_rate_hz": source_rate,
        "requested_sample_rate_hz": requested_rate,
        "sampling_stride_frames": stride,
        "nominal_sample_rate_hz": source_rate / stride,
        "sampling_policy": KINEMATICS_SAMPLING_POLICY,
        "selection_expression": "source_acquisition_frame_index % stride == 0",
        "row_order": "track_id_then_source_acquisition_frame_index",
        "source_logical_paths": list(_SELECTED_SURFACES),
        "source_dtypes": {
            name: {
                "dtype": dtype.str,
                "dtype_fields": _dtype_fields(dtype),
                "trailing_shape": list(_SOURCE_TRAILING_SHAPES[name]),
            }
            for name, dtype in _SOURCE_DTYPES.items()
        },
        "scientific_dtypes": dict(KINEMATICS_SCIENTIFIC_DTYPES),
        "arrow_schema_sha256": ARROW_TABLE_CONTRACTS[
            KINEMATICS_SAMPLES_TABLE
        ].payload_sha256,
        "invalid_float_semantics": "source_ieee_nan_not_arrow_null",
        "position_authority": "source_camera_physical_mm",
    }
    if frame_range is not None:
        payload.update(
            {
                "frame_selection_policy": KINEMATICS_FRAME_SELECTION_POLICY,
                "source_frame_start": frame_range[0],
                "source_frame_stop_exclusive": frame_range[1],
                "selection_expression": (
                    "source_frame_start <= source_acquisition_frame_index < "
                    "source_frame_stop_exclusive and "
                    "source_acquisition_frame_index % stride == 0"
                ),
            }
        )
    return {**payload, "payload_sha256": canonical_json_sha256(payload)}


def kinematics_parquet_policy(*, row_group_rows: int) -> dict[str, Any]:
    if type(row_group_rows) is not int or row_group_rows <= 0:
        raise ValueError("row_group_rows must be a positive exact integer.")
    payload: dict[str, Any] = {
        "schema_id": KINEMATICS_PARQUET_POLICY_SCHEMA_ID,
        "schema_version": KINEMATICS_PARQUET_POLICY_SCHEMA_VERSION,
        "compression": "zstd",
        "compression_level": 3,
        "row_group_rows": row_group_rows,
        "part_policy": "one_part_per_recording",
        "dictionary_columns": [
            field.name
            for field in ARROW_TABLE_CONTRACTS[KINEMATICS_SAMPLES_TABLE].fields
            if field.arrow_type == "string"
        ],
    }
    return {**payload, "payload_sha256": canonical_json_sha256(payload)}


def _surface_binding(
    track_group: Any,
    *,
    relative_path: str,
    record: Mapping[str, Any],
    sample_count: int,
) -> dict[str, Any]:
    expected_fields = (
        _SURFACE_FIELDS
        | (_PHYSICAL_SURFACE_FIELDS if relative_path in _PHYSICAL_SURFACES else set())
        | ({"transition_anchor"} if relative_path in _TRANSITION_SURFACES else set())
    )
    if set(record) != expected_fields:
        raise ValueError(
            f"Track surface {relative_path!r} has an unexpected manifest field set."
        )
    if record.get("relative_ref") != relative_path:
        raise ValueError(f"Track surface {relative_path!r} has a mismatched reference.")
    if record.get("authority_scope") != "public_derived_motion":
        raise ValueError(f"Track surface {relative_path!r} is not public authority.")
    if (
        relative_path in _TRANSITION_SURFACES
        and record.get("transition_anchor") != "destination_track_sample"
    ):
        raise ValueError(
            f"Track transition surface {relative_path!r} has an invalid anchor."
        )
    shape = [sample_count, *_SOURCE_TRAILING_SHAPES[relative_path]]
    dtype = _SOURCE_DTYPES[relative_path]
    if (
        record.get("shape") != shape
        or record.get("dtype") != dtype.str
        or record.get("dtype_fields") != _dtype_fields(dtype)
        or record.get("itemsize") != dtype.itemsize
    ):
        raise ValueError(
            f"Track surface {relative_path!r} differs from its exact dtype/shape contract."
        )
    _sha256_text(record.get("content_sha256"), label=f"{relative_path} content")
    _sha256_text(record.get("attrs_sha256"), label=f"{relative_path} attrs")
    attrs = _json_object(record.get("attrs"), label=f"{relative_path} manifest attrs")
    if canonical_json_sha256(attrs) != record["attrs_sha256"]:
        raise ValueError(f"Track surface {relative_path!r} attr digest is invalid.")
    node = _child(track_group, relative_path)
    node_dtype = np.dtype(getattr(node, "dtype"))
    node_shape = [int(value) for value in getattr(node, "shape")]
    node_attrs = _json_object(
        _group_attrs(node), label=f"/{getattr(node, 'path', relative_path)} attrs"
    )
    if (
        node_dtype != dtype
        or node_shape != shape
        or node_attrs != attrs
        or canonical_json_sha256(node_attrs) != record["attrs_sha256"]
    ):
        raise ValueError(
            f"Track surface {relative_path!r} live declaration differs from its manifest."
        )
    return {
        "relative_ref": relative_path,
        "dtype": dtype.str,
        "dtype_fields": _dtype_fields(dtype),
        "itemsize": dtype.itemsize,
        "shape": shape,
        "content_sha256": record["content_sha256"],
        "attrs_sha256": record["attrs_sha256"],
        "record_sha256": canonical_json_sha256(record),
    }


def _source_binding(
    root: Any,
    *,
    zarr_path: Path,
    recording_id: str,
    run_name: str,
    scope: str,
) -> _BoundSource:
    run, resolved_name, run_path = resolve_track_kinematics_run(
        root,
        run_name=run_name,
        scope=scope,
        historical_inspection=False,
    )
    attrs = _group_attrs(run)
    if (
        attrs.get("schema_id") != TRACK_KINEMATICS_RUN_SCHEMA_ID
        or attrs.get("schema_version") != TRACK_KINEMATICS_RUN_SCHEMA_VERSION
    ):
        raise ValueError("Kinematic export source run schema is invalid.")
    if attrs.get("palette_run_completion_status") != "complete":
        raise ValueError("Kinematic export source must be complete.")
    if attrs.get("stage_selector_eligible") is not True:
        raise ValueError("Kinematic export source must be selector-eligible.")
    manifest = _json_object(
        attrs.get(TRACK_MOTION_PUBLICATION_MANIFEST_ATTR),
        label="track-motion publication manifest",
    )
    manifest_version = manifest.get("schema_version")
    expected_manifest_fields = _MANIFEST_V1_FIELDS | (
        {"position_lineage_mode"}
        if manifest_version == TRACK_MOTION_PUBLICATION_MANIFEST_SCHEMA_VERSION_V2
        else set()
    )
    if (
        set(manifest) != expected_manifest_fields
        or manifest.get("schema_id") != TRACK_MOTION_PUBLICATION_MANIFEST_SCHEMA_ID
        or manifest_version
        not in {
            TRACK_MOTION_PUBLICATION_MANIFEST_SCHEMA_VERSION,
            TRACK_MOTION_PUBLICATION_MANIFEST_SCHEMA_VERSION_V2,
        }
        or manifest.get("run_ref") != f"/{run_path}"
        or manifest.get("run_type") != scope
        or manifest.get("run_name") != resolved_name
    ):
        raise ValueError("Track-motion manifest schema or run identity is invalid.")
    manifest_sha = canonical_json_sha256(manifest)
    if attrs.get(TRACK_MOTION_PUBLICATION_MANIFEST_DIGEST_ATTR) != manifest_sha:
        raise ValueError("Track-motion manifest digest is invalid.")
    commit = _json_object(
        attrs.get(TRACK_MOTION_PUBLICATION_COMMIT_ATTR),
        label="track-motion publication commit",
    )
    if commit != _expected_publication_commit(manifest):
        raise ValueError("Track-motion publication commit is invalid.")
    physical = manifest.get("physical_authority")
    if not isinstance(physical, Mapping):
        raise ValueError("Kinematic sample export requires physical-mm authority.")
    physical_sha = canonical_json_sha256(physical)
    root_attrs = manifest.get("run_root_attrs")
    record = root_attrs.get("record") if isinstance(root_attrs, Mapping) else None
    immutable = record.get("immutable_attrs") if isinstance(record, Mapping) else None
    if not isinstance(immutable, Mapping):
        raise ValueError("Track-motion manifest lacks immutable root attributes.")
    source_rate = immutable.get("fps")
    if isinstance(source_rate, bool) or not isinstance(source_rate, (int, float)):
        raise ValueError("Track-motion manifest lacks its exact source FPS.")
    source_rate = float(source_rate)
    if not math.isfinite(source_rate) or source_rate <= 0:
        raise ValueError("Track-motion source FPS must be positive and finite.")

    raw_tracks = manifest.get("tracks")
    track_count = manifest.get("track_count")
    if (
        not isinstance(raw_tracks, Mapping)
        or type(track_count) is not int
        or track_count <= 0
        or len(raw_tracks) != track_count
    ):
        raise ValueError("Track-motion manifest track inventory is invalid.")
    ordered: list[dict[str, Any]] = []
    descriptor_sha: str | None = None
    descriptor_space: str | None = None
    live_tracks = run["tracks"]
    expected_track_names: list[str] = []
    for name, track_record_value in sorted(
        raw_tracks.items(),
        key=lambda item: int(str(item[0]).removeprefix("id_")),
    ):
        if not isinstance(track_record_value, Mapping):
            raise ValueError(f"Track-motion manifest record {name!r} is invalid.")
        track_record = dict(track_record_value)
        if set(track_record) != _TRACK_RECORD_FIELDS:
            raise ValueError(f"Track-motion manifest record {name!r} is not closed.")
        track_id = track_record.get("track_id")
        if type(track_id) is not int or name != f"id_{track_id}":
            raise ValueError("Track-motion track name and numeric identity disagree.")
        expected_track_names.append(name)
        sample_count = track_record.get("track_sample_count")
        if type(sample_count) is not int or sample_count < 0:
            raise ValueError(f"Track {track_id} sample count is invalid.")
        expected_ref = f"/{run_path}/tracks/{name}"
        if track_record.get("track_ref") != expected_ref:
            raise ValueError(f"Track {track_id} reference is invalid.")
        surfaces = track_record.get("surfaces")
        if not isinstance(surfaces, Mapping):
            raise ValueError(f"Track {track_id} surface inventory is invalid.")
        track_group = live_tracks[name]
        selected = {
            relative_path: _surface_binding(
                track_group,
                relative_path=relative_path,
                record=surfaces.get(relative_path, {}),
                sample_count=sample_count,
            )
            for relative_path in _SELECTED_SURFACES
        }
        position_record = surfaces["positions_mm"]
        position_attrs = position_record["attrs"]
        current_descriptor_sha = position_attrs.get("coordinate_descriptor_sha256")
        descriptor = position_attrs.get("coordinate_descriptor")
        current_space = (
            descriptor.get("space_id") if isinstance(descriptor, Mapping) else None
        )
        _sha256_text(
            current_descriptor_sha,
            label=f"track {track_id} position coordinate descriptor",
        )
        if current_space != KINEMATICS_POSITION_SPACE:
            raise ValueError(
                f"Track {track_id} position space must be {KINEMATICS_POSITION_SPACE!r}."
            )
        if position_record.get("physical_authority_sha256") != physical_sha:
            raise ValueError(f"Track {track_id} physical authority binding is invalid.")
        if descriptor_sha is None:
            descriptor_sha = str(current_descriptor_sha)
            descriptor_space = str(current_space)
        elif (
            descriptor_sha != current_descriptor_sha
            or descriptor_space != current_space
        ):
            raise ValueError(
                "All exported tracks must share one physical coordinate frame."
            )
        ordered.append(
            {
                "track_id": track_id,
                "track_ref": expected_ref,
                "sample_count": sample_count,
                "track_record_sha256": canonical_json_sha256(track_record),
                "selected_surfaces": selected,
            }
        )
    live_names = sorted(str(value) for value in live_tracks.group_keys())
    if live_names != sorted(expected_track_names):
        raise ValueError("Live track groups differ from the sealed manifest inventory.")
    if descriptor_sha is None or descriptor_space is None:  # pragma: no cover
        raise ValueError("Track-motion source has no physical coordinate descriptor.")

    parent = root["analysis"]["track_kinematics_runs"]
    scope_group = parent[scope]
    payload: dict[str, Any] = {
        "schema_id": KINEMATICS_SOURCE_BINDING_SCHEMA_ID,
        "schema_version": KINEMATICS_SOURCE_BINDING_SCHEMA_VERSION,
        "stage_id": "track_kinematics",
        "recording_id": recording_id,
        "zarr_path": str(zarr_path),
        "scope": scope,
        "run_name": resolved_name,
        "run_path": run_path,
        "source_schema_id": attrs.get("schema_id"),
        "source_schema_version": attrs.get("schema_version"),
        "source_manifest_schema_id": manifest["schema_id"],
        "source_manifest_schema_version": manifest_version,
        "source_manifest_sha256": manifest_sha,
        "source_publication_commit_sha256": canonical_json_sha256(commit),
        "source_sample_rate_hz": source_rate,
        "position_coordinate_space": descriptor_space,
        "position_coordinate_descriptor_sha256": descriptor_sha,
        "physical_authority_sha256": physical_sha,
        "selection_snapshot": {
            "mode": "explicit_run",
            "parent_latest": parent.attrs.get("latest"),
            "parent_latest_complete": parent.attrs.get("latest_complete"),
            "parent_latest_scope": parent.attrs.get(f"latest_{scope}"),
            "scope_latest": scope_group.attrs.get("latest"),
            "parent_completion_epoch": parent.attrs.get("palette_completion_epoch"),
            "scope_completion_epoch": scope_group.attrs.get("palette_completion_epoch"),
        },
        "completion_snapshot": {
            "status": attrs.get("palette_run_completion_status"),
            "completed_at_utc": attrs.get("palette_run_completed_at_utc"),
            "selector_eligible": attrs.get("stage_selector_eligible"),
        },
        "track_count": track_count,
        "tracks": ordered,
    }
    return _BoundSource(
        binding={**payload, "payload_sha256": canonical_json_sha256(payload)},
        run_group=run,
    )


def bind_kinematics_samples_source(
    root: Any,
    *,
    zarr_path: str | Path,
    track_kinematics_run: str,
    track_scope: str,
    expected_recording_id: str | None = None,
) -> BoundKinematicsSamplesSource:
    """Bind one explicit canonical track-kinematics publication.

    This is the shared admission boundary used by both the standalone
    ``kinematics_samples`` exporter and composable cohort profiles.  It does
    not discover a selector or publish any output.
    """

    source = Path(zarr_path).expanduser().resolve()
    recording_id = _recording_id(source)
    if expected_recording_id is not None and recording_id != str(expected_recording_id):
        raise ValueError("Kinematic source archive has another recording identity.")
    run_name = safe_component(
        track_kinematics_run,
        label="track-kinematics run ID",
    )
    if track_scope not in {"online", "offline"}:
        raise ValueError("track_scope must be 'online' or 'offline'.")
    return _source_binding(
        root,
        zarr_path=source,
        recording_id=recording_id,
        run_name=run_name,
        scope=track_scope,
    )


class _ProjectedPayloadHasher:
    def __init__(self) -> None:
        self._hashers = {
            name: hashlib.sha256() for name in KINEMATICS_SCIENTIFIC_DTYPES
        }
        self.row_count = 0

    def update(self, columns: Mapping[str, np.ndarray[Any, Any]]) -> None:
        missing = set(KINEMATICS_SCIENTIFIC_DTYPES) - set(columns)
        unexpected = set(columns) - set(KINEMATICS_SCIENTIFIC_DTYPES)
        if missing or unexpected:
            raise ValueError(
                "Kinematic payload column set differs: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        lengths = {int(np.asarray(value).shape[0]) for value in columns.values()}
        if len(lengths) != 1:
            raise ValueError("Kinematic projected columns have unequal row counts.")
        count = lengths.pop()
        for name, dtype_name in KINEMATICS_SCIENTIFIC_DTYPES.items():
            values = np.asarray(columns[name])
            if values.ndim != 1 or int(values.shape[0]) != count:
                raise ValueError(f"{name}: projected kinematic column must be 1D.")
            dtype = _NUMPY_DTYPES[dtype_name]
            if dtype_name == "bool":
                values = values.astype(bool, copy=False).astype(dtype, copy=False)
            else:
                values = values.astype(dtype, copy=False)
            self._hashers[name].update(np.ascontiguousarray(values).tobytes(order="C"))
        self.row_count += count

    def finish(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_id": KINEMATICS_PROJECTED_PAYLOAD_SCHEMA_ID,
            "schema_version": KINEMATICS_PROJECTED_PAYLOAD_SCHEMA_VERSION,
            "row_count": self.row_count,
            "column_sha256": {
                name: self._hashers[name].hexdigest()
                for name in KINEMATICS_SCIENTIFIC_DTYPES
            },
        }
        return {**payload, "payload_sha256": canonical_json_sha256(payload)}


class _SelectedSourcePayloadHasher:
    """Stream the canonical source-array hashes already sealed by publication."""

    def __init__(self, track_binding: Mapping[str, Any]) -> None:
        surfaces = track_binding["selected_surfaces"]
        self._expected = {
            path: str(record["content_sha256"]) for path, record in surfaces.items()
        }
        self._hashers: dict[str, Any] = {}
        for path in _SELECTED_SURFACES:
            dtype = _SOURCE_DTYPES[path]
            shape = [
                int(track_binding["sample_count"]),
                *_SOURCE_TRAILING_SHAPES[path],
            ]
            header = {
                "canonicalization": ARRAY_PAYLOAD_CANONICALIZATION,
                "dtype": np.lib.format.dtype_to_descr(dtype),
                "shape": shape,
            }
            digest = hashlib.sha256()
            digest.update(canonical_json_bytes(header))
            digest.update(b"\x00")
            self._hashers[path] = digest

    def update(self, source: Mapping[str, np.ndarray[Any, Any]]) -> None:
        if set(source) != set(_SELECTED_SURFACES):
            raise ValueError("Selected source payload inventory changed while reading.")
        for path in _SELECTED_SURFACES:
            self._hashers[path].update(
                np.ascontiguousarray(source[path]).tobytes(order="C")
            )

    def finish(self) -> None:
        changed = [
            path
            for path in _SELECTED_SURFACES
            if self._hashers[path].hexdigest() != self._expected[path]
        ]
        if changed:
            raise ValueError(
                "Selected track-motion payload differs from its publication "
                f"manifest: {changed!r}."
            )


def _footer_metadata() -> dict[bytes, bytes]:
    return {
        b"palette.export_schema_id": EXPORT_SCHEMA_ID.encode("utf-8"),
        b"palette.export_schema_version": str(EXPORT_SCHEMA_VERSION).encode("ascii"),
        b"palette.table_contract": json.dumps(
            TABLE_CONTRACTS[KINEMATICS_SAMPLES_TABLE].to_dict(),
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


def _read_projected_window(
    track_group: Any,
    *,
    track_id: int,
    start: int,
    stop: int,
    stride: int,
    source_rate_hz: float,
    source_hasher: _SelectedSourcePayloadHasher,
    source_frame_start: int | None = None,
    source_frame_stop_exclusive: int | None = None,
) -> tuple[dict[str, np.ndarray[Any, Any]], np.ndarray[Any, Any]]:
    source: dict[str, np.ndarray[Any, Any]] = {}
    for path in _SELECTED_SURFACES:
        values = np.asarray(_child(track_group, path)[start:stop])
        expected_shape = (stop - start, *_SOURCE_TRAILING_SHAPES[path])
        if values.dtype != _SOURCE_DTYPES[path] or values.shape != expected_shape:
            raise ValueError(
                f"Track {track_id} source {path!r} changed dtype or shape while reading."
            )
        source[path] = values
    source_hasher.update(source)
    keys = source["track_sample_key"]
    frames = source["source_acquisition_frame_index"]
    if (
        np.any(keys[:, 0] != track_id)
        or not np.array_equal(keys[:, 1], frames)
        or np.any(frames < 0)
        or (frames.size > 1 and np.any(np.diff(frames) <= 0))
    ):
        raise ValueError(f"Track {track_id} row identity is invalid.")
    expected_time = np.asarray(
        frames.astype(np.float64) / source_rate_hz, dtype=np.float32
    )
    if not np.array_equal(source["time_seconds"], expected_time):
        raise ValueError(
            f"Track {track_id} time values differ from frame/FPS authority."
        )
    instance = source["source_instance_key"]
    valid = np.asarray(instance["valid"], dtype=bool)
    values = np.asarray(instance["instance_key"], dtype=np.uint64)
    if np.any((~valid) & (values != 0)):
        raise ValueError(f"Track {track_id} nullable instance keys violate zero-fill.")
    finite = np.all(np.isfinite(source["positions_mm"]), axis=1)
    if not np.array_equal(source["position_finite"], finite):
        raise ValueError(f"Track {track_id} position-finite flags are inconsistent.")
    selection = np.asarray(frames % stride == 0, dtype=bool)
    if source_frame_start is not None or source_frame_stop_exclusive is not None:
        if source_frame_start is None or source_frame_stop_exclusive is None:
            raise ValueError("Projected source frame range is incomplete.")
        selection &= frames >= source_frame_start
        selection &= frames < source_frame_stop_exclusive
    positions = source["positions_mm"][selection]
    columns = {
        "track_id": np.full(np.count_nonzero(selection), track_id, dtype=np.int64),
        "track_sample_index": np.arange(start, stop, dtype=np.int64)[selection],
        "source_acquisition_frame_index": frames[selection],
        "time_seconds": source["time_seconds"][selection],
        "source_row_index": source["source_row_index"][selection],
        "source_instance_key_valid": valid[selection],
        "source_instance_key": values[selection],
        "detection_source": source["detection_source"][selection],
        "position_x_mm": positions[:, 0],
        "position_y_mm": positions[:, 1],
        "speed_mm_s": source["movement/speed/filtered/mm"][selection],
        "frame_path_distance_mm": source[
            "movement/speed/filtered/frame_path_distance_mm"
        ][selection],
        "motion_heading_degrees": source["heading_degrees"][selection],
        "smoothed_motion_heading_degrees": source["smoothed_heading_degrees"][
            selection
        ],
        "smoothed_angular_velocity_deg_s": source["angular_velocity_smoothed_deg_s"][
            selection
        ],
        "source_observed": source["source_observed"][selection],
        "sample_observed": source["sample_observed"][selection],
        "position_finite": source["position_finite"][selection],
        "heading_usable": source["heading_usable"][selection],
        "sample_valid": source["sample_valid"][selection],
        "transition_valid": source["transition_valid"][selection],
        "sample_reason_code": source["sample_reason_code"][selection],
        "transition_reason_code": source["transition_reason_code"][selection],
    }
    return columns, frames


def kinematics_sample_constant_values(
    source_binding: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact standalone provenance constants for one projection."""

    return {
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "table_name": KINEMATICS_SAMPLES_TABLE,
        "recording_id": source_binding["recording_id"],
        "zarr_path": source_binding["zarr_path"],
        "source_lineage_hash": _source_lineage_sha256(source_binding, projection),
        "source_track_kinematics_scope": source_binding["scope"],
        "source_track_kinematics_run": source_binding["run_name"],
        "source_track_kinematics_path": source_binding["run_path"],
        "source_track_motion_manifest_schema_id": source_binding[
            "source_manifest_schema_id"
        ],
        "source_track_motion_manifest_schema_version": source_binding[
            "source_manifest_schema_version"
        ],
        "source_track_motion_manifest_sha256": source_binding["source_manifest_sha256"],
        "source_binding_sha256": source_binding["payload_sha256"],
        "projection_contract_sha256": projection["payload_sha256"],
        "source_speed_level": projection["source_speed_level"],
        "source_sample_rate_hz": projection["source_sample_rate_hz"],
        "requested_sample_rate_hz": projection["requested_sample_rate_hz"],
        "sampling_stride_frames": projection["sampling_stride_frames"],
        "nominal_sample_rate_hz": projection["nominal_sample_rate_hz"],
        "sampling_policy": projection["sampling_policy"],
        "position_coordinate_space": source_binding["position_coordinate_space"],
        "position_coordinate_descriptor_sha256": source_binding[
            "position_coordinate_descriptor_sha256"
        ],
        "physical_authority_sha256": source_binding["physical_authority_sha256"],
    }


def _arrow_batch(
    columns: Mapping[str, Any],
    *,
    source_binding: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> Any:
    import pyarrow as pa

    count = int(np.asarray(columns["track_id"]).shape[0])
    constants = kinematics_sample_constant_values(source_binding, projection)
    schema = exact_arrow_schema(KINEMATICS_SAMPLES_TABLE, metadata=_footer_metadata())
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


def iter_projected_kinematics_sample_batches(
    bound: BoundKinematicsSamplesSource,
    *,
    projection: Mapping[str, Any],
    source_window_rows: int,
) -> Any:
    """Yield bounded exact standalone-column batches from a bound source.

    The iterator verifies every selected source surface against the manifest
    while it is already resident for projection.  Consumers may add outer
    dataset provenance columns, but must not alter these returned values.
    """

    if type(source_window_rows) is not int or source_window_rows <= 0:
        raise ValueError("source_window_rows must be a positive exact integer.")
    if projection.get("table_name") != KINEMATICS_SAMPLES_TABLE or float(
        projection.get("source_sample_rate_hz", float("nan"))
    ) != float(bound.binding["source_sample_rate_hz"]):
        raise ValueError("Kinematic projection differs from its bound source.")
    constants = kinematics_sample_constant_values(bound.binding, projection)
    for track in bound.binding["tracks"]:
        track_id = int(track["track_id"])
        track_group = bound.run_group["tracks"][f"id_{track_id}"]
        sample_count = int(track["sample_count"])
        source_hasher = _SelectedSourcePayloadHasher(track)
        last_frame: int | None = None
        for start in range(0, sample_count, source_window_rows):
            stop = min(sample_count, start + source_window_rows)
            columns, source_frames = _read_projected_window(
                track_group,
                track_id=track_id,
                start=start,
                stop=stop,
                stride=int(projection["sampling_stride_frames"]),
                source_rate_hz=float(projection["source_sample_rate_hz"]),
                source_hasher=source_hasher,
                source_frame_start=projection.get("source_frame_start"),
                source_frame_stop_exclusive=projection.get(
                    "source_frame_stop_exclusive"
                ),
            )
            if source_frames.size:
                if last_frame is not None and int(source_frames[0]) <= last_frame:
                    raise ValueError(
                        f"Track {track_id} frame identity is not globally increasing."
                    )
                last_frame = int(source_frames[-1])
            count = int(columns["track_id"].shape[0])
            if count:
                yield {
                    **{name: [value] * count for name, value in constants.items()},
                    **columns,
                }
        source_hasher.finish()


def _write_streaming_part(
    bound: _BoundSource,
    *,
    part_path: Path,
    projection: Mapping[str, Any],
    source_window_rows: int,
    row_group_rows: int,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    if type(source_window_rows) is not int or source_window_rows <= 0:
        raise ValueError("source_window_rows must be a positive exact integer.")
    schema = exact_arrow_schema(KINEMATICS_SAMPLES_TABLE, metadata=_footer_metadata())
    dictionary_columns = [
        field.name
        for field in ARROW_TABLE_CONTRACTS[KINEMATICS_SAMPLES_TABLE].fields
        if field.arrow_type == "string"
    ]
    hasher = _ProjectedPayloadHasher()
    part_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(
        part_path,
        schema,
        compression="zstd",
        compression_level=3,
        use_dictionary=dictionary_columns,
    )
    try:
        for columns in iter_projected_kinematics_sample_batches(
            bound,
            projection=projection,
            source_window_rows=source_window_rows,
        ):
            hasher.update(
                {name: columns[name] for name in KINEMATICS_SCIENTIFIC_DTYPES}
            )
            writer.write_table(
                _arrow_batch(
                    columns,
                    source_binding=bound.binding,
                    projection=projection,
                ),
                row_group_size=row_group_rows,
            )
    finally:
        writer.close()
    return hasher.finish()


def _validate_source_binding(source: Mapping[str, Any]) -> None:
    if set(source) != _SOURCE_BINDING_FIELDS:
        raise ValueError("Kinematic source binding has an unexpected field set.")
    body = dict(source)
    digest = body.pop("payload_sha256", None)
    if digest != canonical_json_sha256(body):
        raise ValueError("Kinematic source-binding digest is invalid.")
    if (
        body.get("schema_id") != KINEMATICS_SOURCE_BINDING_SCHEMA_ID
        or body.get("schema_version") != KINEMATICS_SOURCE_BINDING_SCHEMA_VERSION
        or body.get("stage_id") != "track_kinematics"
        or body.get("source_manifest_schema_id")
        != TRACK_MOTION_PUBLICATION_MANIFEST_SCHEMA_ID
        or body.get("source_manifest_schema_version")
        not in {
            TRACK_MOTION_PUBLICATION_MANIFEST_SCHEMA_VERSION,
            TRACK_MOTION_PUBLICATION_MANIFEST_SCHEMA_VERSION_V2,
        }
        or body.get("source_schema_id") != TRACK_KINEMATICS_RUN_SCHEMA_ID
        or body.get("source_schema_version") != TRACK_KINEMATICS_RUN_SCHEMA_VERSION
    ):
        raise ValueError("Kinematic source-binding schema is invalid.")
    for field in ("recording_id", "zarr_path", "scope", "run_name", "run_path"):
        if not isinstance(body.get(field), str) or not body[field]:
            raise ValueError(f"Kinematic source field {field} is invalid.")
    if body["scope"] not in {"online", "offline"} or body["run_path"] != (
        f"analysis/track_kinematics_runs/{body['scope']}/{body['run_name']}"
    ):
        raise ValueError("Kinematic source run path is invalid.")
    for field in (
        "source_manifest_sha256",
        "source_publication_commit_sha256",
        "position_coordinate_descriptor_sha256",
        "physical_authority_sha256",
    ):
        _sha256_text(body.get(field), label=field)
    if body.get("position_coordinate_space") != KINEMATICS_POSITION_SPACE:
        raise ValueError("Kinematic source position coordinate space is invalid.")
    rate = body.get("source_sample_rate_hz")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise ValueError("Kinematic source sample rate is invalid.")
    if not math.isfinite(float(rate)) or float(rate) <= 0:
        raise ValueError("Kinematic source sample rate must be positive and finite.")
    selection = body.get("selection_snapshot")
    if (
        not isinstance(selection, Mapping)
        or set(selection) != _SELECTION_SNAPSHOT_FIELDS
    ):
        raise ValueError("Kinematic source selection snapshot is invalid.")
    if selection.get("mode") != "explicit_run":
        raise ValueError("Kinematic source selection mode is invalid.")
    for field, value in selection.items():
        if field == "mode":
            continue
        if field.endswith("epoch"):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"Kinematic selection field {field} is invalid.")
        elif value is not None and not isinstance(value, str):
            raise ValueError(f"Kinematic selection field {field} is invalid.")
    completion = body.get("completion_snapshot")
    if (
        not isinstance(completion, Mapping)
        or set(completion) != _COMPLETION_SNAPSHOT_FIELDS
        or completion.get("status") != "complete"
        or completion.get("selector_eligible") is not True
        or not isinstance(completion.get("completed_at_utc"), str)
        or not completion["completed_at_utc"]
    ):
        raise ValueError("Kinematic source completion snapshot is invalid.")
    tracks = body.get("tracks")
    if (
        not isinstance(tracks, list)
        or type(body.get("track_count")) is not int
        or body["track_count"] <= 0
        or len(tracks) != body["track_count"]
    ):
        raise ValueError("Kinematic source bound-track inventory is invalid.")
    previous_id: int | None = None
    for track in tracks:
        if not isinstance(track, Mapping) or set(track) != _BOUND_TRACK_FIELDS:
            raise ValueError("Kinematic bound-track record is invalid.")
        track_id = track.get("track_id")
        if type(track_id) is not int or (
            previous_id is not None and track_id <= previous_id
        ):
            raise ValueError("Kinematic bound-track IDs must be strictly increasing.")
        previous_id = track_id
        if (
            track.get("track_ref") != f"/{body['run_path']}/tracks/id_{track_id}"
            or type(track.get("sample_count")) is not int
            or track["sample_count"] < 0
        ):
            raise ValueError("Kinematic bound-track identity is invalid.")
        _sha256_text(track.get("track_record_sha256"), label="track record")
        surfaces = track.get("selected_surfaces")
        if not isinstance(surfaces, Mapping) or set(surfaces) != set(
            _SELECTED_SURFACES
        ):
            raise ValueError("Kinematic selected-surface inventory is invalid.")
        for path, surface in surfaces.items():
            if (
                not isinstance(surface, Mapping)
                or set(surface) != _BOUND_SURFACE_FIELDS
            ):
                raise ValueError(f"Kinematic surface binding {path!r} is invalid.")
            expected_dtype = _SOURCE_DTYPES[path]
            if (
                surface.get("relative_ref") != path
                or surface.get("dtype") != expected_dtype.str
                or surface.get("dtype_fields") != _dtype_fields(expected_dtype)
                or surface.get("itemsize") != expected_dtype.itemsize
                or surface.get("shape")
                != [track["sample_count"], *_SOURCE_TRAILING_SHAPES[path]]
            ):
                raise ValueError(f"Kinematic surface binding {path!r} is inconsistent.")
            for field in ("content_sha256", "attrs_sha256", "record_sha256"):
                _sha256_text(surface.get(field), label=f"{path} {field}")


def _validate_kinematics_envelope(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    envelope = payload.get("kinematics_samples_export")
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
        raise ValueError("Kinematic export envelope has an unexpected field set.")
    if (
        envelope.get("schema_id") != KINEMATICS_EXPORT_SCHEMA_ID
        or envelope.get("schema_version") != KINEMATICS_EXPORT_SCHEMA_VERSION
    ):
        raise ValueError("Kinematic export schema is invalid.")
    body = {key: envelope[key] for key in required - {"payload_sha256"}}
    if envelope.get("payload_sha256") != canonical_json_sha256(body):
        raise ValueError("Kinematic export envelope digest is invalid.")
    source = envelope["source_binding"]
    if not isinstance(source, Mapping):
        raise ValueError("Kinematic source binding is invalid.")
    _validate_source_binding(source)
    projection = envelope["projection_contract"]
    if not isinstance(projection, Mapping):
        raise ValueError("Kinematic projection contract is invalid.")
    projection_version = projection.get("schema_version")
    if projection_version == KINEMATICS_PROJECTION_SCHEMA_VERSION:
        frame_start = None
        frame_stop = None
    elif projection_version == KINEMATICS_PROJECTION_SCHEMA_VERSION_V2:
        frame_start = projection.get("source_frame_start")
        frame_stop = projection.get("source_frame_stop_exclusive")
    else:
        raise ValueError("Kinematic projection schema version is unsupported.")
    expected_projection = kinematics_projection_contract(
        source_sample_rate_hz=float(source["source_sample_rate_hz"]),
        requested_sample_rate_hz=float(projection.get("requested_sample_rate_hz")),
        source_frame_start=frame_start,
        source_frame_stop_exclusive=frame_stop,
    )
    if dict(projection) != expected_projection:
        raise ValueError("Kinematic projection differs from the installed contract.")
    projected = envelope["projected_payload"]
    if (
        not isinstance(projected, Mapping)
        or set(projected) != _PROJECTED_PAYLOAD_FIELDS
    ):
        raise ValueError("Kinematic projected-payload receipt is invalid.")
    projected_body = dict(projected)
    projected_digest = projected_body.pop("payload_sha256", None)
    if projected_digest != canonical_json_sha256(projected_body):
        raise ValueError("Kinematic projected-payload digest is invalid.")
    if (
        projected_body.get("schema_id") != KINEMATICS_PROJECTED_PAYLOAD_SCHEMA_ID
        or projected_body.get("schema_version")
        != KINEMATICS_PROJECTED_PAYLOAD_SCHEMA_VERSION
        or type(projected_body.get("row_count")) is not int
        or projected_body["row_count"] < 0
        or set(projected_body.get("column_sha256", {}))
        != set(KINEMATICS_SCIENTIFIC_DTYPES)
    ):
        raise ValueError("Kinematic projected-payload schema is invalid.")
    for digest in projected_body["column_sha256"].values():
        _sha256_text(digest, label="projected column digest")
    policy = envelope["parquet_policy"]
    if not isinstance(policy, Mapping):
        raise ValueError("Kinematic Parquet policy is invalid.")
    policy_body = dict(policy)
    policy_digest = policy_body.pop("payload_sha256", None)
    if policy_digest != canonical_json_sha256(policy_body) or dict(policy) != (
        kinematics_parquet_policy(row_group_rows=policy_body.get("row_group_rows"))
    ):
        raise ValueError(
            "Kinematic Parquet policy differs from the installed contract."
        )
    return envelope


def validate_kinematics_samples_export_payload(
    export_root: Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Rehash and validate the exact manifest-selected decoded projection."""

    import pyarrow.parquet as pq

    envelope = _validate_kinematics_envelope(payload)
    parts = manifest_selected_part_files_from_payload(
        export_root,
        payload,
        KINEMATICS_SAMPLES_TABLE,
        allow_legacy_layout=False,
    )
    if len(parts) != 1:
        raise ValueError("Kinematic sample export must select exactly one part.")
    source = envelope["source_binding"]
    projection = envelope["projection_contract"]
    assert isinstance(source, Mapping) and isinstance(projection, Mapping)
    track_counts = {
        int(track["track_id"]): int(track["sample_count"]) for track in source["tracks"]
    }
    hasher = _ProjectedPayloadHasher()
    last_key: tuple[int, int] | None = None
    parquet_file = pq.ParquetFile(parts[0])
    validate_arrow_schema(KINEMATICS_SAMPLES_TABLE, parquet_file.schema_arrow)
    policy = envelope["parquet_policy"]
    assert isinstance(policy, Mapping)
    max_row_group_rows = int(policy["row_group_rows"])
    dictionary_columns = set(policy["dictionary_columns"])
    arrow_names = parquet_file.schema_arrow.names
    for group_index in range(parquet_file.metadata.num_row_groups):
        row_group = parquet_file.metadata.row_group(group_index)
        if row_group.num_rows > max_row_group_rows:
            raise ValueError("Kinematic Parquet row group exceeds its frozen policy.")
        for column_index, field_name in enumerate(arrow_names):
            column = row_group.column(column_index)
            if column.compression != "ZSTD":
                raise ValueError(
                    "Kinematic Parquet column is not Zstandard-compressed."
                )
            if field_name in dictionary_columns and not any(
                encoding in {"PLAIN_DICTIONARY", "RLE_DICTIONARY"}
                for encoding in column.encodings
            ):
                raise ValueError(
                    f"Kinematic Parquet string column {field_name!r} is not dictionary-encoded."
                )
    constant_values = {
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "table_name": KINEMATICS_SAMPLES_TABLE,
        "recording_id": source["recording_id"],
        "zarr_path": source["zarr_path"],
        "source_lineage_hash": _source_lineage_sha256(source, projection),
        "source_track_kinematics_scope": source["scope"],
        "source_track_kinematics_run": source["run_name"],
        "source_track_kinematics_path": source["run_path"],
        "source_track_motion_manifest_schema_id": source["source_manifest_schema_id"],
        "source_track_motion_manifest_schema_version": source[
            "source_manifest_schema_version"
        ],
        "source_track_motion_manifest_sha256": source["source_manifest_sha256"],
        "source_binding_sha256": source["payload_sha256"],
        "projection_contract_sha256": projection["payload_sha256"],
        "source_speed_level": projection["source_speed_level"],
        "source_sample_rate_hz": projection["source_sample_rate_hz"],
        "requested_sample_rate_hz": projection["requested_sample_rate_hz"],
        "sampling_stride_frames": projection["sampling_stride_frames"],
        "nominal_sample_rate_hz": projection["nominal_sample_rate_hz"],
        "sampling_policy": projection["sampling_policy"],
        "position_coordinate_space": source["position_coordinate_space"],
        "position_coordinate_descriptor_sha256": source[
            "position_coordinate_descriptor_sha256"
        ],
        "physical_authority_sha256": source["physical_authority_sha256"],
    }
    for batch in parquet_file.iter_batches():
        table = batch.to_pydict()
        columns = {
            name: np.asarray(table[name], dtype=_NUMPY_DTYPES[dtype_name])
            for name, dtype_name in KINEMATICS_SCIENTIFIC_DTYPES.items()
        }
        for field, expected_value in constant_values.items():
            if any(value != expected_value for value in table[field]):
                raise ValueError(
                    f"Kinematic Parquet field {field} changed within the part."
                )
        ids = columns["track_id"]
        indices = columns["track_sample_index"]
        frames = columns["source_acquisition_frame_index"]
        stride = int(projection["sampling_stride_frames"])
        if np.any(frames < 0) or np.any(frames % stride != 0):
            raise ValueError("Kinematic Parquet frame sampling is invalid.")
        if projection["schema_version"] == KINEMATICS_PROJECTION_SCHEMA_VERSION_V2:
            if np.any(frames < int(projection["source_frame_start"])) or np.any(
                frames >= int(projection["source_frame_stop_exclusive"])
            ):
                raise ValueError("Kinematic Parquet frame range is invalid.")
        expected_time = np.asarray(
            frames.astype(np.float64) / float(projection["source_sample_rate_hz"]),
            dtype=np.float32,
        )
        if not np.array_equal(columns["time_seconds"], expected_time):
            raise ValueError("Kinematic Parquet time values are invalid.")
        if np.any(
            (~columns["source_instance_key_valid"].astype(bool))
            & (columns["source_instance_key"] != 0)
        ):
            raise ValueError("Kinematic Parquet nullable instance keys are invalid.")
        for track_id, sample_index, frame in zip(ids, indices, frames, strict=True):
            key = (int(track_id), int(frame))
            if int(track_id) not in track_counts:
                raise ValueError("Kinematic Parquet contains an unknown track ID.")
            if (
                int(sample_index) < 0
                or int(sample_index) >= track_counts[int(track_id)]
            ):
                raise ValueError("Kinematic Parquet sample index is out of bounds.")
            if last_key is not None and key <= last_key:
                raise ValueError(
                    "Kinematic Parquet primary keys are not strictly ordered."
                )
            last_key = key
        hasher.update(columns)
    observed = hasher.finish()
    if observed != envelope["projected_payload"]:
        raise ValueError("Kinematic decoded payload differs from its receipt.")
    return {
        "valid": True,
        "row_count": observed["row_count"],
        "projected_payload_sha256": observed["payload_sha256"],
        "source_binding_sha256": source["payload_sha256"],
    }


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def export_kinematics_samples(
    zarr_path: str | Path,
    *,
    track_kinematics_run: str,
    track_scope: str,
    output_root: str | Path,
    export_run_id: str,
    scratch_root: str | Path,
    requested_sample_rate_hz: float | None = None,
    source_window_rows: int = 131_072,
    row_group_rows: int = 65_536,
    source_frame_start: int | None = None,
    source_frame_stop_exclusive: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Stream all tracks into one exact immutable portable sample table."""

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
    source_run = safe_component(track_kinematics_run, label="track-kinematics run ID")
    if track_scope not in {"online", "offline"}:
        raise ValueError("track_scope must be 'online' or 'offline'.")
    policy = kinematics_parquet_policy(row_group_rows=row_group_rows)
    recording_id = _recording_id(source_path)
    manifest_path = export_manifest_path(destination, run_id)
    baseline_identity = manifest_identity(manifest_path)
    if baseline_identity is not None and not overwrite:
        raise FileExistsError(f"Export manifest already exists: {manifest_path}")

    runtime = ExportRuntimePhaseRecorder()
    with runtime.measure("source_binding_before"):
        root = open_zarr_root(source_path, mode="r")
        before = bind_kinematics_samples_source(
            root,
            zarr_path=source_path,
            expected_recording_id=recording_id,
            track_kinematics_run=source_run,
            track_scope=track_scope,
        )
    source_sample_rate_hz = float(before.binding["source_sample_rate_hz"])
    projection = kinematics_projection_contract(
        source_sample_rate_hz=source_sample_rate_hz,
        requested_sample_rate_hz=(
            source_sample_rate_hz
            if requested_sample_rate_hz is None
            else float(requested_sample_rate_hz)
        ),
        source_frame_start=source_frame_start,
        source_frame_stop_exclusive=source_frame_stop_exclusive,
    )
    generation_id = uuid.uuid4().hex
    final_generation_path = generation_relative_path(run_id, generation_id)
    staging = publication_staging_root(destination, run_id, generation_id)
    final_generation = publication_generation_root(destination, run_id, generation_id)
    if staging.exists() or final_generation.exists():
        raise FileExistsError(
            f"Analytics export generation already exists: {generation_id}"
        )
    scratch_generation = scratch / f"palette_kinematics_{run_id}_{generation_id}"
    if scratch_generation.exists():
        raise FileExistsError(
            f"Kinematic scratch generation already exists: {scratch_generation}"
        )
    source_hash = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
    part_name = f"part-00000-{source_hash}.parquet"
    scratch_part = scratch_generation / "tables" / KINEMATICS_SAMPLES_TABLE / part_name
    try:
        with runtime.measure("scratch_parquet_write"):
            projected_payload = _write_streaming_part(
                before,
                part_path=scratch_part,
                projection=projection,
                source_window_rows=source_window_rows,
                row_group_rows=row_group_rows,
            )
        with runtime.measure("source_binding_after"):
            after_root = open_zarr_root(source_path, mode="r")
            after = bind_kinematics_samples_source(
                after_root,
                zarr_path=source_path,
                expected_recording_id=recording_id,
                track_kinematics_run=source_run,
                track_scope=track_scope,
            )
            if after.binding != before.binding:
                raise RuntimeError(
                    "Track-kinematics selection, completion, or manifest binding "
                    "changed during extraction."
                )
        staged_part = staging / "tables" / KINEMATICS_SAMPLES_TABLE / part_name
        with runtime.measure("scratch_to_staging_copy"):
            staged_part.parent.mkdir(parents=True, exist_ok=False)
            shutil.copy2(scratch_part, staged_part)
            staged_sha256 = sha256_file(staged_part)
            if staged_sha256 != sha256_file(scratch_part):
                raise RuntimeError(
                    "Kinematic scratch-to-publication copy digest mismatch."
                )

        relative_part = (
            final_generation_path / "tables" / KINEMATICS_SAMPLES_TABLE / part_name
        ).as_posix()
        row_count = int(projected_payload["row_count"])
        inventory = {
            KINEMATICS_SAMPLES_TABLE: [
                {
                    "path": relative_part,
                    "sha256": staged_sha256,
                    "size_bytes": int(staged_part.stat().st_size),
                    "row_count": row_count,
                }
            ]
        }
        columns = tuple(
            field.name
            for field in ARROW_TABLE_CONTRACTS[KINEMATICS_SAMPLES_TABLE].fields
        )
        capability_statuses = resolve_capabilities({KINEMATICS_SAMPLES_TABLE: columns})
        envelope_body: dict[str, Any] = {
            "schema_id": KINEMATICS_EXPORT_SCHEMA_ID,
            "schema_version": KINEMATICS_EXPORT_SCHEMA_VERSION,
            "source_binding": before.binding,
            "projection_contract": projection,
            "projected_payload": projected_payload,
            "parquet_policy": policy,
        }
        kinematics_envelope = {
            **envelope_body,
            "payload_sha256": canonical_json_sha256(envelope_body),
        }
        git = get_git_info(Path(__file__).resolve().parents[3])
        manifest: dict[str, Any] = {
            "export_run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "schema_id": EXPORT_SCHEMA_ID,
            "schema_version": EXPORT_SCHEMA_VERSION,
            "tool": "fisheye.analytics_exports.kinematics_samples",
            "hostname": socket.gethostname(),
            "palette_git_commit": git.get("commit_hash"),
            "palette_git_dirty": git.get("is_dirty"),
            "source_recording_count": 1,
            "source_zarrs": [str(source_path)],
            "tables_requested": [KINEMATICS_SAMPLES_TABLE],
            "table_contracts": contract_snapshot((KINEMATICS_SAMPLES_TABLE,)),
            "arrow_schema_contracts": arrow_contract_envelope(
                (KINEMATICS_SAMPLES_TABLE,)
            ),
            "capabilities": [
                item.capability_id for item in capability_statuses if item.available
            ],
            "capability_statuses": [item.to_dict() for item in capability_statuses],
            "row_counts_by_table": {KINEMATICS_SAMPLES_TABLE: row_count},
            "part_files_by_table": {KINEMATICS_SAMPLES_TABLE: [relative_part]},
            "publication": {
                "schema_id": PUBLICATION_SCHEMA_ID,
                "schema_version": PUBLICATION_SCHEMA_VERSION,
                "state": "complete",
                "generation_id": generation_id,
                "generation_path": final_generation_path.as_posix(),
                "parts_by_table": inventory,
            },
            "diagnostics": [],
            "collection_manifest": None,
            "export_parameters": {
                "registry_indexing": False,
                "selector_activation": False,
                "source_mutation": False,
                "scratch_root": str(scratch),
                "source_window_rows": source_window_rows,
                "source_frame_start": source_frame_start,
                "source_frame_stop_exclusive": source_frame_stop_exclusive,
                "overwrite": bool(overwrite),
            },
            "kinematics_samples_export": kinematics_envelope,
        }
        with runtime.measure("staged_decoded_validation"):
            staged_hasher = _ProjectedPayloadHasher()
            import pyarrow.parquet as pq

            staged_file = pq.ParquetFile(staged_part)
            for batch in staged_file.iter_batches():
                values = batch.to_pydict()
                staged_hasher.update(
                    {
                        name: np.asarray(values[name], dtype=_NUMPY_DTYPES[dtype_name])
                        for name, dtype_name in KINEMATICS_SCIENTIFIC_DTYPES.items()
                    }
                )
            if staged_hasher.finish() != projected_payload:
                raise RuntimeError(
                    "Kinematic staged decoded payload differs from scratch."
                )
        with runtime.measure("manifest_validation"):
            _validate_kinematics_envelope(manifest)
        committed = commit_staged_publication(
            destination,
            staging,
            manifest,
            baseline_manifest_identity=baseline_identity,
            runtime_recorder=runtime,
        )
        with runtime.measure("published_payload_validation"):
            published = json.loads(committed.read_text(encoding="utf-8"))
            validation = validate_kinematics_samples_export_payload(
                destination, published
            )
        return {
            **published,
            "manifest_path": str(committed),
            "kinematics_samples_validation": validation,
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
    "BoundKinematicsSamplesSource",
    "KINEMATICS_EXPORT_SCHEMA_ID",
    "KINEMATICS_EXPORT_SCHEMA_VERSION",
    "KINEMATICS_FRAME_SELECTION_POLICY",
    "KINEMATICS_PROJECTION_SCHEMA_VERSION_V2",
    "KINEMATICS_SAMPLING_POLICY",
    "KINEMATICS_SCIENTIFIC_DTYPES",
    "bind_kinematics_samples_source",
    "export_kinematics_samples",
    "iter_projected_kinematics_sample_batches",
    "kinematics_sample_constant_values",
    "kinematics_parquet_policy",
    "kinematics_projection_contract",
    "validate_kinematics_samples_export_payload",
]
