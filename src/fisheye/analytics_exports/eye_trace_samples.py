"""Exact bounded export of compact-v7 framewise eye traces.

The recording-local Zarr run remains scientific authority.  This module emits
one immutable, manifest-selected Parquet query product without mutating source
selectors, completion markers, registries, or storage profiles.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import socket
from typing import Any, Mapping
import uuid
from datetime import datetime, timezone

import numpy as np

from fisheye.analysis.eye_angle_io import (
    EyeAngleIOError,
    catalog_eye_angle_series,
    load_eye_angle_series_rows,
    resolve_eye_angle_run,
)
from fisheye.analysis.eye_angle_schema import (
    EYE_ANGLE_ARRAY_SCHEMA_ATTR,
    EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2,
    EYE_ANGLE_RUN_PARENT,
    EYE_ANGLE_RUN_SCHEMA_ID,
    EYE_ANGLE_RUN_SCHEMA_VERSION,
)
from fisheye.analytics_exports.arrow_contracts import (
    ARROW_TABLE_CONTRACTS,
    arrow_contract_envelope,
    exact_arrow_schema,
    validate_arrow_schema,
)
from fisheye.analytics_exports.capabilities import resolve_capabilities
from fisheye.analytics_exports.contracts import (
    EYE_TRACE_SAMPLES_TABLE,
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
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
from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr_io import open_zarr_root

EYE_TRACE_EXPORT_SCHEMA_ID = "palette.analytics_export.eye_trace_samples"
EYE_TRACE_EXPORT_SCHEMA_VERSION = 1
EYE_TRACE_SOURCE_BINDING_SCHEMA_ID = "palette.eye_trace.source_binding"
EYE_TRACE_SOURCE_BINDING_SCHEMA_VERSION = 1
EYE_TRACE_PROJECTION_SCHEMA_ID = "palette.eye_trace.projection"
EYE_TRACE_PROJECTION_SCHEMA_VERSION = 1
EYE_TRACE_PROJECTED_PAYLOAD_SCHEMA_ID = "palette.eye_trace.projected_payload"
EYE_TRACE_PROJECTED_PAYLOAD_SCHEMA_VERSION = 1
EYE_TRACE_PARQUET_POLICY_SCHEMA_ID = "palette.eye_trace.parquet_policy"
EYE_TRACE_PARQUET_POLICY_SCHEMA_VERSION = 1

EYE_TRACE_ANGLE_CHANNELS = (
    "left_eye_angle_deg",
    "right_eye_angle_deg",
    "vergence_eye_angle_deg",
    "left_eye_angle_deg_smoothed",
    "right_eye_angle_deg_smoothed",
    "vergence_eye_angle_deg_smoothed",
    "left_gaze_signed_deg",
    "right_gaze_signed_deg",
    "left_gaze_signed_deg_smoothed",
    "right_gaze_signed_deg_smoothed",
    "mean_eye_vergence_gaze_deg",
    "mean_eye_vergence_gaze_deg_smoothed",
)
EYE_TRACE_QA_CHANNELS = (
    "valid_frame",
    "major_axis_marginal",
    "reason_codes",
)
EYE_TRACE_SCIENTIFIC_DTYPES = {
    "source_acquisition_frame_index": "int64",
    "time_seconds": "float32",
    **{name: "float32" for name in EYE_TRACE_ANGLE_CHANNELS},
    "valid_frame": "bool",
    "major_axis_marginal": "bool",
    "reason_codes": "uint16",
}
_NUMPY_DTYPES = {
    "int64": np.dtype("<i8"),
    "float32": np.dtype("<f4"),
    "bool": np.dtype("u1"),
    "uint16": np.dtype("<u2"),
}
_REQUIRED_SOURCE_MANIFEST_ATTRS = {
    "eye_angle_array_schema_sha256": EYE_ANGLE_ARRAY_SCHEMA_ATTR,
    "eye_angle_source_contracts_sha256": "eye_angle_source_contracts",
    "eye_angle_algorithm_contract_sha256": "eye_angle_algorithm_contract",
    "eye_angle_output_schema_sha256": "eye_angle_output_schema",
    "eye_angle_variant_schema_sha256": "eye_angle_variant_schema",
}
_SOURCE_BINDING_FIELDS = {
    "schema_id",
    "schema_version",
    "stage_id",
    "recording_id",
    "zarr_path",
    "run_name",
    "run_path",
    "source_schema_id",
    "source_schema_version",
    "source_layout",
    "source_method",
    "source_method_version",
    "frame_count",
    "selection_snapshot",
    "completion_snapshot",
    "manifest_digests",
    "payload_sha256",
}
_SELECTION_SNAPSHOT_FIELDS = {
    "mode",
    "parent_latest",
    "parent_latest_complete",
    "parent_completion_epoch",
}
_COMPLETION_SNAPSHOT_FIELDS = {
    "status",
    "completed_at_utc",
    "selector_eligible",
}
_SOURCE_MANIFEST_DIGEST_FIELDS = {
    *_REQUIRED_SOURCE_MANIFEST_ATTRS,
    "eye_angle_storage_plan_sha256",
}
_PROJECTED_PAYLOAD_FIELDS = {
    "schema_id",
    "schema_version",
    "row_count",
    "column_sha256",
    "payload_sha256",
}
_SHA256_LENGTH = 64


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
    if not isinstance(normalized, dict):  # pragma: no cover - guarded above.
        raise TypeError(f"{label} did not normalize to an object.")
    return normalized


def _sha256_text(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != _SHA256_LENGTH or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{label} must be one lowercase SHA-256 digest.")
    return text


def eye_trace_projection_contract() -> dict[str, Any]:
    """Return the closed semantic projection independently of batch sizing."""

    arrow_contract = ARROW_TABLE_CONTRACTS[EYE_TRACE_SAMPLES_TABLE]
    source_paths = {
        "source_acquisition_frame_index": "implicit_camera_frame_axis",
        "time_seconds": "support/frame_time_seconds",
        **{name: f"frame_angles/{name}" for name in EYE_TRACE_ANGLE_CHANNELS},
        **{name: f"frame_qa/{name}" for name in EYE_TRACE_QA_CHANNELS},
    }
    payload: dict[str, Any] = {
        "schema_id": EYE_TRACE_PROJECTION_SCHEMA_ID,
        "schema_version": EYE_TRACE_PROJECTION_SCHEMA_VERSION,
        "table_name": EYE_TRACE_SAMPLES_TABLE,
        "row_axis": "camera_frame",
        "angle_channels": list(EYE_TRACE_ANGLE_CHANNELS),
        "qa_channels": list(EYE_TRACE_QA_CHANNELS),
        "scientific_dtypes": dict(EYE_TRACE_SCIENTIFIC_DTYPES),
        "source_logical_paths": source_paths,
        "arrow_schema_sha256": arrow_contract.payload_sha256,
        "invalid_float_semantics": "source_ieee_nan_not_arrow_null",
    }
    return {**payload, "payload_sha256": canonical_json_sha256(payload)}


def eye_trace_parquet_policy(*, row_group_rows: int) -> dict[str, Any]:
    if type(row_group_rows) is not int or row_group_rows <= 0:
        raise ValueError("row_group_rows must be a positive exact integer.")
    payload: dict[str, Any] = {
        "schema_id": EYE_TRACE_PARQUET_POLICY_SCHEMA_ID,
        "schema_version": EYE_TRACE_PARQUET_POLICY_SCHEMA_VERSION,
        "compression": "zstd",
        "compression_level": 3,
        "row_group_rows": row_group_rows,
        "part_policy": "one_part_per_recording",
        "dictionary_columns": [
            field.name
            for field in ARROW_TABLE_CONTRACTS[EYE_TRACE_SAMPLES_TABLE].fields
            if field.arrow_type == "string"
        ],
    }
    return {**payload, "payload_sha256": canonical_json_sha256(payload)}


def _source_binding(
    root: Any,
    *,
    zarr_path: Path,
    recording_id: str,
    run_name: str,
) -> dict[str, Any]:
    run, resolved_name, run_path = resolve_eye_angle_run(
        root,
        run_name,
        legacy_compatibility=False,
    )
    catalog = catalog_eye_angle_series(
        root,
        run_name=resolved_name,
        prefer_frame=True,
        legacy_compatibility=False,
    )
    if catalog.row_axis != "frame":
        raise EyeAngleIOError("Eye-trace export requires the exact frame axis.")
    missing_angles = sorted(set(EYE_TRACE_ANGLE_CHANNELS) - set(catalog.angle_channels))
    missing_qa = sorted(set(EYE_TRACE_QA_CHANNELS) - set(catalog.qa_channels))
    if missing_angles or missing_qa:
        raise EyeAngleIOError(
            "Eye-trace projection is unavailable; missing "
            f"angles={missing_angles}, qa={missing_qa}."
        )
    attrs = dict(run.attrs)
    if attrs.get("layout") != EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2:
        raise EyeAngleIOError("Eye-trace export supports only compact-dense-v2.")
    parent = root[EYE_ANGLE_RUN_PARENT]
    manifest_digests: dict[str, str | None] = {}
    for digest_name, attr_name in _REQUIRED_SOURCE_MANIFEST_ATTRS.items():
        manifest_digests[digest_name] = canonical_json_sha256(
            _json_object(attrs.get(attr_name), label=attr_name)
        )
    storage_plan = attrs.get("eye_angle_storage_plan")
    manifest_digests["eye_angle_storage_plan_sha256"] = (
        canonical_json_sha256(
            _json_object(storage_plan, label="eye_angle_storage_plan")
        )
        if storage_plan is not None
        else None
    )
    payload: dict[str, Any] = {
        "schema_id": EYE_TRACE_SOURCE_BINDING_SCHEMA_ID,
        "schema_version": EYE_TRACE_SOURCE_BINDING_SCHEMA_VERSION,
        "stage_id": "eye_angles",
        "recording_id": recording_id,
        "zarr_path": str(zarr_path),
        "run_name": resolved_name,
        "run_path": run_path,
        "source_schema_id": attrs.get("schema_id"),
        "source_schema_version": attrs.get("schema_version"),
        "source_layout": attrs.get("layout"),
        "source_method": attrs.get("method"),
        "source_method_version": attrs.get("method_version"),
        "frame_count": catalog.row_count,
        "selection_snapshot": {
            "mode": "explicit_run",
            "parent_latest": parent.attrs.get("latest"),
            "parent_latest_complete": parent.attrs.get("latest_complete"),
            "parent_completion_epoch": parent.attrs.get("palette_completion_epoch"),
        },
        "completion_snapshot": {
            "status": attrs.get("palette_run_completion_status"),
            "completed_at_utc": attrs.get("palette_run_completed_at_utc"),
            "selector_eligible": attrs.get("stage_selector_eligible"),
        },
        "manifest_digests": manifest_digests,
    }
    return {**payload, "payload_sha256": canonical_json_sha256(payload)}


class _ProjectedPayloadHasher:
    def __init__(self) -> None:
        self._hashers = {name: hashlib.sha256() for name in EYE_TRACE_SCIENTIFIC_DTYPES}
        self.row_count = 0

    def update(self, columns: Mapping[str, np.ndarray]) -> None:
        missing = set(EYE_TRACE_SCIENTIFIC_DTYPES) - set(columns)
        unexpected = set(columns) - set(EYE_TRACE_SCIENTIFIC_DTYPES)
        if missing or unexpected:
            raise ValueError(
                f"Eye-trace payload column set differs: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        lengths = {int(np.asarray(value).shape[0]) for value in columns.values()}
        if len(lengths) != 1:
            raise ValueError("Eye-trace projected columns have unequal row counts.")
        count = lengths.pop()
        for name, dtype_name in EYE_TRACE_SCIENTIFIC_DTYPES.items():
            dtype = _NUMPY_DTYPES[dtype_name]
            values = np.asarray(columns[name])
            if values.ndim != 1 or int(values.shape[0]) != count:
                raise ValueError(f"{name}: projected eye-trace column must be 1D.")
            if dtype_name == "bool":
                values = values.astype(bool, copy=False).astype(dtype, copy=False)
            else:
                values = values.astype(dtype, copy=False)
            self._hashers[name].update(np.ascontiguousarray(values).tobytes(order="C"))
        self.row_count += count

    def finish(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_id": EYE_TRACE_PROJECTED_PAYLOAD_SCHEMA_ID,
            "schema_version": EYE_TRACE_PROJECTED_PAYLOAD_SCHEMA_VERSION,
            "row_count": self.row_count,
            "column_sha256": {
                name: self._hashers[name].hexdigest()
                for name in EYE_TRACE_SCIENTIFIC_DTYPES
            },
        }
        return {**payload, "payload_sha256": canonical_json_sha256(payload)}


def _footer_metadata() -> dict[bytes, bytes]:
    return {
        b"palette.export_schema_id": EXPORT_SCHEMA_ID.encode("utf-8"),
        b"palette.export_schema_version": str(EXPORT_SCHEMA_VERSION).encode("ascii"),
        b"palette.table_contract": json.dumps(
            TABLE_CONTRACTS[EYE_TRACE_SAMPLES_TABLE].to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    }


def _source_lineage_sha256(
    source_binding: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> str:
    return canonical_json_sha256(
        {
            "source_binding_sha256": source_binding["payload_sha256"],
            "projection_contract_sha256": projection["payload_sha256"],
        }
    )


def _batch_columns(window: Any) -> dict[str, np.ndarray]:
    qa = {
        "valid_frame": np.asarray(window.qa["valid_frame"], dtype=bool),
        "major_axis_marginal": np.asarray(window.qa["major_axis_marginal"], dtype=bool),
        "reason_codes": np.asarray(window.qa["reason_codes"], dtype=np.uint16),
    }
    for name in ("valid_frame", "major_axis_marginal"):
        raw = np.asarray(window.qa[name])
        if np.any((raw != 0) & (raw != 1)):
            raise ValueError(
                f"{name}: compact-v7 boolean QA contains values outside 0/1."
            )
    return {
        "source_acquisition_frame_index": np.asarray(
            window.frame_indices, dtype=np.int64
        ),
        "time_seconds": np.asarray(window.time_seconds, dtype=np.float32),
        **{
            name: np.asarray(window.angles[name], dtype=np.float32)
            for name in EYE_TRACE_ANGLE_CHANNELS
        },
        **qa,
    }


def _arrow_batch(
    columns: Mapping[str, np.ndarray],
    *,
    recording_id: str,
    zarr_path: Path,
    source_binding: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> Any:
    import pyarrow as pa

    count = int(np.asarray(columns["source_acquisition_frame_index"]).shape[0])
    binding_sha = str(source_binding["payload_sha256"])
    projection_sha = str(projection["payload_sha256"])
    lineage_sha = _source_lineage_sha256(source_binding, projection)
    constants: dict[str, Any] = {
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "table_name": EYE_TRACE_SAMPLES_TABLE,
        "recording_id": recording_id,
        "zarr_path": str(zarr_path),
        "source_lineage_hash": lineage_sha,
        "source_eye_angle_run": source_binding["run_name"],
        "source_eye_angle_path": source_binding["run_path"],
        "source_eye_angle_schema_id": source_binding["source_schema_id"],
        "source_eye_angle_schema_version": source_binding["source_schema_version"],
        "source_eye_angle_layout": source_binding["source_layout"],
        "source_eye_angle_method": source_binding["source_method"],
        "source_eye_angle_method_version": source_binding["source_method_version"],
        "source_binding_sha256": binding_sha,
        "projection_contract_sha256": projection_sha,
    }
    schema = exact_arrow_schema(
        EYE_TRACE_SAMPLES_TABLE,
        metadata=_footer_metadata(),
    )
    arrays = []
    for field in schema:
        if field.name in columns:
            arrays.append(pa.array(columns[field.name], type=field.type))
        else:
            arrays.append(pa.array([constants[field.name]] * count, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def _write_streaming_part(
    root: Any,
    *,
    part_path: Path,
    zarr_path: Path,
    recording_id: str,
    source_binding: Mapping[str, Any],
    projection: Mapping[str, Any],
    row_group_rows: int,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    schema = exact_arrow_schema(EYE_TRACE_SAMPLES_TABLE, metadata=_footer_metadata())
    dictionary_columns = [
        field.name
        for field in ARROW_TABLE_CONTRACTS[EYE_TRACE_SAMPLES_TABLE].fields
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
        frame_count = int(source_binding["frame_count"])
        for start in range(0, frame_count, row_group_rows):
            stop = min(frame_count, start + row_group_rows)
            window = load_eye_angle_series_rows(
                root,
                run_name=str(source_binding["run_name"]),
                start_row=start,
                stop_row=stop,
                angle_channels=EYE_TRACE_ANGLE_CHANNELS,
                qa_channels=EYE_TRACE_QA_CHANNELS,
                max_rows=row_group_rows,
            )
            columns = _batch_columns(window)
            expected_frames = np.arange(start, stop, dtype=np.int64)
            if not np.array_equal(
                columns["source_acquisition_frame_index"], expected_frames
            ):
                raise ValueError(
                    "Eye-trace bounded reader returned a noncontiguous frame axis."
                )
            hasher.update(columns)
            writer.write_table(
                _arrow_batch(
                    columns,
                    recording_id=recording_id,
                    zarr_path=zarr_path,
                    source_binding=source_binding,
                    projection=projection,
                ),
                row_group_size=row_group_rows,
            )
    finally:
        writer.close()
    return hasher.finish()


def _validate_eye_trace_envelope(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    envelope = payload.get("eye_trace_export")
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
        raise ValueError("Eye-trace export envelope has an unexpected field set.")
    if envelope.get("schema_id") != EYE_TRACE_EXPORT_SCHEMA_ID:
        raise ValueError("Eye-trace export schema ID is invalid.")
    if envelope.get("schema_version") != EYE_TRACE_EXPORT_SCHEMA_VERSION:
        raise ValueError("Eye-trace export schema version is invalid.")
    body = {key: envelope[key] for key in required - {"payload_sha256"}}
    if envelope.get("payload_sha256") != canonical_json_sha256(body):
        raise ValueError("Eye-trace export envelope digest is invalid.")
    projection = envelope["projection_contract"]
    if projection != eye_trace_projection_contract():
        raise ValueError("Eye-trace projection differs from the installed contract.")
    source = envelope["source_binding"]
    if not isinstance(source, Mapping) or set(source) != _SOURCE_BINDING_FIELDS:
        raise ValueError("Eye-trace source binding is invalid.")
    source_body = dict(source)
    source_digest = source_body.pop("payload_sha256", None)
    if source_digest != canonical_json_sha256(source_body):
        raise ValueError("Eye-trace source-binding digest is invalid.")
    if source_body.get("schema_id") != EYE_TRACE_SOURCE_BINDING_SCHEMA_ID:
        raise ValueError("Eye-trace source-binding schema ID is invalid.")
    if source_body.get("schema_version") != EYE_TRACE_SOURCE_BINDING_SCHEMA_VERSION:
        raise ValueError("Eye-trace source-binding schema version is invalid.")
    if source_body.get("stage_id") != "eye_angles":
        raise ValueError("Eye-trace source stage is invalid.")
    if source_body.get("source_schema_id") != EYE_ANGLE_RUN_SCHEMA_ID:
        raise ValueError("Eye-trace source schema ID is invalid.")
    if source_body.get("source_schema_version") != EYE_ANGLE_RUN_SCHEMA_VERSION:
        raise ValueError("Eye-trace source schema version is invalid.")
    if source_body.get("source_layout") != EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2:
        raise ValueError("Eye-trace source layout is invalid.")
    for field in ("recording_id", "zarr_path", "run_name", "run_path"):
        if not isinstance(source_body.get(field), str) or not source_body[field]:
            raise ValueError(f"Eye-trace source field {field} is invalid.")
    if source_body["run_path"] != f"{EYE_ANGLE_RUN_PARENT}/{source_body['run_name']}":
        raise ValueError("Eye-trace source run path does not match its run identity.")
    if (
        type(source_body.get("frame_count")) is not int
        or source_body["frame_count"] <= 0
    ):
        raise ValueError(
            "Eye-trace source frame count must be a positive exact integer."
        )
    for field in ("source_method", "source_method_version"):
        if source_body.get(field) is not None and not isinstance(
            source_body[field], str
        ):
            raise ValueError(f"Eye-trace source field {field} is invalid.")
    selection = source_body.get("selection_snapshot")
    if (
        not isinstance(selection, Mapping)
        or set(selection) != _SELECTION_SNAPSHOT_FIELDS
    ):
        raise ValueError("Eye-trace source selection snapshot is invalid.")
    if selection.get("mode") != "explicit_run":
        raise ValueError("Eye-trace source selection mode is invalid.")
    for field in ("parent_latest", "parent_latest_complete"):
        if selection.get(field) is not None and not isinstance(selection[field], str):
            raise ValueError(f"Eye-trace source selection field {field} is invalid.")
    if (
        type(selection.get("parent_completion_epoch")) is not int
        or selection["parent_completion_epoch"] < 0
    ):
        raise ValueError("Eye-trace source completion epoch is invalid.")
    completion = source_body.get("completion_snapshot")
    if (
        not isinstance(completion, Mapping)
        or set(completion) != _COMPLETION_SNAPSHOT_FIELDS
    ):
        raise ValueError("Eye-trace source completion snapshot is invalid.")
    if completion.get("status") != "complete":
        raise ValueError("Eye-trace source completion status is invalid.")
    if (
        not isinstance(completion.get("completed_at_utc"), str)
        or not completion["completed_at_utc"]
    ):
        raise ValueError("Eye-trace source completion time is invalid.")
    if completion.get("selector_eligible") is not True:
        raise ValueError("Eye-trace source must be selector-eligible.")
    manifest_digests = source_body.get("manifest_digests")
    if (
        not isinstance(manifest_digests, Mapping)
        or set(manifest_digests) != _SOURCE_MANIFEST_DIGEST_FIELDS
    ):
        raise ValueError("Eye-trace source manifest-digest inventory is invalid.")
    for digest in manifest_digests.values():
        if digest is not None:
            _sha256_text(digest, label="source manifest digest")
    projected = envelope["projected_payload"]
    if (
        not isinstance(projected, Mapping)
        or set(projected) != _PROJECTED_PAYLOAD_FIELDS
    ):
        raise ValueError("Eye-trace projected-payload receipt is invalid.")
    projected_body = dict(projected)
    projected_digest = projected_body.pop("payload_sha256", None)
    if projected_digest != canonical_json_sha256(projected_body):
        raise ValueError("Eye-trace projected-payload digest is invalid.")
    if projected_body.get("schema_id") != EYE_TRACE_PROJECTED_PAYLOAD_SCHEMA_ID:
        raise ValueError("Eye-trace projected-payload schema ID is invalid.")
    if (
        projected_body.get("schema_version")
        != EYE_TRACE_PROJECTED_PAYLOAD_SCHEMA_VERSION
    ):
        raise ValueError("Eye-trace projected-payload schema version is invalid.")
    if (
        type(projected_body.get("row_count")) is not int
        or projected_body["row_count"] < 0
    ):
        raise ValueError("Eye-trace projected-payload row count is invalid.")
    if set(projected_body.get("column_sha256", {})) != set(EYE_TRACE_SCIENTIFIC_DTYPES):
        raise ValueError("Eye-trace projected-payload column inventory is invalid.")
    for digest in projected_body["column_sha256"].values():
        _sha256_text(digest, label="projected column digest")
    policy = envelope["parquet_policy"]
    if not isinstance(policy, Mapping):
        raise ValueError("Eye-trace Parquet policy is invalid.")
    policy_body = dict(policy)
    policy_digest = policy_body.pop("payload_sha256", None)
    if policy_digest != canonical_json_sha256(policy_body):
        raise ValueError("Eye-trace Parquet-policy digest is invalid.")
    expected_policy = eye_trace_parquet_policy(
        row_group_rows=policy_body.get("row_group_rows")
    )
    if dict(policy) != expected_policy:
        raise ValueError(
            "Eye-trace Parquet policy differs from the installed contract."
        )
    return envelope


def validate_eye_trace_export_payload(
    export_root: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Rehash the exact decoded frame projection selected by one manifest."""

    import pyarrow.parquet as pq

    envelope = _validate_eye_trace_envelope(payload)
    parts = manifest_selected_part_files_from_payload(
        export_root,
        payload,
        EYE_TRACE_SAMPLES_TABLE,
        allow_legacy_layout=False,
    )
    if len(parts) != 1:
        raise ValueError("Eye-trace export must select exactly one recording part.")
    source = envelope["source_binding"]
    projection = envelope["projection_contract"]
    assert isinstance(source, Mapping) and isinstance(projection, Mapping)
    hasher = _ProjectedPayloadHasher()
    expected_frame = 0
    parquet_file = pq.ParquetFile(parts[0])
    validate_arrow_schema(EYE_TRACE_SAMPLES_TABLE, parquet_file.schema_arrow)
    for batch in parquet_file.iter_batches():
        table = batch.to_pydict()
        columns = {
            name: np.asarray(table[name], dtype=_NUMPY_DTYPES[dtype_name])
            for name, dtype_name in EYE_TRACE_SCIENTIFIC_DTYPES.items()
        }
        count = int(columns["source_acquisition_frame_index"].shape[0])
        expected = np.arange(expected_frame, expected_frame + count, dtype=np.int64)
        if not np.array_equal(columns["source_acquisition_frame_index"], expected):
            raise ValueError("Eye-trace Parquet frame identity is not contiguous.")
        expected_frame += count
        for field, expected_value in (
            ("export_schema_version", EXPORT_SCHEMA_VERSION),
            ("table_name", EYE_TRACE_SAMPLES_TABLE),
            ("recording_id", source["recording_id"]),
            ("zarr_path", source["zarr_path"]),
            ("source_lineage_hash", _source_lineage_sha256(source, projection)),
            ("source_eye_angle_run", source["run_name"]),
            ("source_eye_angle_path", source["run_path"]),
            ("source_eye_angle_schema_id", source["source_schema_id"]),
            ("source_eye_angle_schema_version", source["source_schema_version"]),
            ("source_eye_angle_layout", source["source_layout"]),
            ("source_eye_angle_method", source["source_method"]),
            ("source_eye_angle_method_version", source["source_method_version"]),
            ("source_binding_sha256", source["payload_sha256"]),
            ("projection_contract_sha256", projection["payload_sha256"]),
        ):
            if any(value != expected_value for value in table[field]):
                raise ValueError(
                    f"Eye-trace Parquet field {field} changed within the part."
                )
        hasher.update(columns)
    observed = hasher.finish()
    if observed != envelope["projected_payload"]:
        raise ValueError(
            "Eye-trace decoded payload differs from its projected receipt."
        )
    if observed["row_count"] != source["frame_count"]:
        raise ValueError(
            "Eye-trace projected row count differs from its source binding."
        )
    return {
        "valid": True,
        "row_count": observed["row_count"],
        "projected_payload_sha256": observed["payload_sha256"],
        "source_binding_sha256": source["payload_sha256"],
    }


def export_eye_trace_samples(
    zarr_path: str | Path,
    *,
    eye_angle_run: str,
    output_root: str | Path,
    export_run_id: str,
    scratch_root: str | Path,
    row_group_rows: int = 65_536,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Stream one exact eye-angle frame projection and publish it atomically."""

    source_path = Path(zarr_path).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    run_id = safe_component(export_run_id, label="export run ID")
    source_run = safe_component(eye_angle_run, label="eye-angle run ID")
    policy = eye_trace_parquet_policy(row_group_rows=row_group_rows)
    recording_id = _recording_id(source_path)
    manifest_path = export_manifest_path(destination, run_id)
    baseline_identity = manifest_identity(manifest_path)
    if baseline_identity is not None and not overwrite:
        raise FileExistsError(f"Export manifest already exists: {manifest_path}")

    runtime = ExportRuntimePhaseRecorder()
    with runtime.measure("source_binding_before"):
        root = open_zarr_root(source_path, mode="r")
        before = _source_binding(
            root,
            zarr_path=source_path,
            recording_id=recording_id,
            run_name=source_run,
        )
    projection = eye_trace_projection_contract()
    generation_id = uuid.uuid4().hex
    final_generation_path = generation_relative_path(run_id, generation_id)
    staging = publication_staging_root(destination, run_id, generation_id)
    final_generation = publication_generation_root(destination, run_id, generation_id)
    if staging.exists() or final_generation.exists():
        raise FileExistsError(
            f"Analytics export generation already exists: {generation_id}"
        )

    scratch_generation = scratch / f"palette_eye_trace_{run_id}_{generation_id}"
    if scratch_generation.exists():
        raise FileExistsError(
            f"Eye-trace scratch generation already exists: {scratch_generation}"
        )
    source_hash = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
    part_name = f"part-00000-{source_hash}.parquet"
    scratch_part = scratch_generation / "tables" / EYE_TRACE_SAMPLES_TABLE / part_name
    try:
        with runtime.measure("scratch_parquet_write"):
            projected_payload = _write_streaming_part(
                root,
                part_path=scratch_part,
                zarr_path=source_path,
                recording_id=recording_id,
                source_binding=before,
                projection=projection,
                row_group_rows=row_group_rows,
            )
        # Reopen direct metadata so the after-snapshot cannot be satisfied by
        # cached group attributes from the pre-extraction handle.
        with runtime.measure("source_binding_after"):
            after_root = open_zarr_root(source_path, mode="r")
            after = _source_binding(
                after_root,
                zarr_path=source_path,
                recording_id=recording_id,
                run_name=source_run,
            )
            if after != before:
                raise RuntimeError(
                    "Eye-angle source selection, completion, or manifest binding "
                    "changed during extraction."
                )
        staged_part = staging / "tables" / EYE_TRACE_SAMPLES_TABLE / part_name
        with runtime.measure("scratch_to_staging_copy"):
            staged_part.parent.mkdir(parents=True, exist_ok=False)
            shutil.copy2(scratch_part, staged_part)
            staged_sha256 = sha256_file(staged_part)
            if staged_sha256 != sha256_file(scratch_part):
                raise RuntimeError(
                    "Eye-trace scratch-to-publication copy digest mismatch."
                )

        relative_part = (
            final_generation_path / "tables" / EYE_TRACE_SAMPLES_TABLE / part_name
        ).as_posix()
        row_count = int(projected_payload["row_count"])
        inventory = {
            EYE_TRACE_SAMPLES_TABLE: [
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
            for field in ARROW_TABLE_CONTRACTS[EYE_TRACE_SAMPLES_TABLE].fields
        )
        capability_statuses = resolve_capabilities({EYE_TRACE_SAMPLES_TABLE: columns})
        envelope_body: dict[str, Any] = {
            "schema_id": EYE_TRACE_EXPORT_SCHEMA_ID,
            "schema_version": EYE_TRACE_EXPORT_SCHEMA_VERSION,
            "source_binding": before,
            "projection_contract": projection,
            "projected_payload": projected_payload,
            "parquet_policy": policy,
        }
        eye_envelope = {
            **envelope_body,
            "payload_sha256": canonical_json_sha256(envelope_body),
        }
        git = get_git_info(Path(__file__).resolve().parents[3])
        manifest: dict[str, Any] = {
            "export_run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "schema_id": EXPORT_SCHEMA_ID,
            "schema_version": EXPORT_SCHEMA_VERSION,
            "tool": "fisheye.analytics_exports.eye_trace_samples",
            "hostname": socket.gethostname(),
            "palette_git_commit": git.get("commit_hash"),
            "palette_git_dirty": git.get("is_dirty"),
            "source_recording_count": 1,
            "source_zarrs": [str(source_path)],
            "tables_requested": [EYE_TRACE_SAMPLES_TABLE],
            "table_contracts": contract_snapshot((EYE_TRACE_SAMPLES_TABLE,)),
            "arrow_schema_contracts": arrow_contract_envelope(
                (EYE_TRACE_SAMPLES_TABLE,)
            ),
            "capabilities": [
                item.capability_id for item in capability_statuses if item.available
            ],
            "capability_statuses": [item.to_dict() for item in capability_statuses],
            "row_counts_by_table": {EYE_TRACE_SAMPLES_TABLE: row_count},
            "part_files_by_table": {EYE_TRACE_SAMPLES_TABLE: [relative_part]},
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
                "overwrite": bool(overwrite),
            },
            "eye_trace_export": eye_envelope,
        }
        # Re-read staged bytes, not the scratch source, before visibility.
        with runtime.measure("staged_decoded_validation"):
            staged_projection = _ProjectedPayloadHasher()
            import pyarrow.parquet as pq

            staged_file = pq.ParquetFile(staged_part)
            for batch in staged_file.iter_batches():
                values = batch.to_pydict()
                staged_projection.update(
                    {
                        name: np.asarray(values[name], dtype=_NUMPY_DTYPES[dtype_name])
                        for name, dtype_name in EYE_TRACE_SCIENTIFIC_DTYPES.items()
                    }
                )
            if staged_projection.finish() != projected_payload:
                raise RuntimeError(
                    "Eye-trace staged decoded payload differs from scratch."
                )
        with runtime.measure("manifest_validation"):
            _validate_eye_trace_envelope(manifest)
        committed = commit_staged_publication(
            destination,
            staging,
            manifest,
            baseline_manifest_identity=baseline_identity,
            runtime_recorder=runtime,
        )
        with runtime.measure("published_payload_validation"):
            published = json.loads(committed.read_text(encoding="utf-8"))
            validation = validate_eye_trace_export_payload(destination, published)
        return {
            **published,
            "manifest_path": str(committed),
            "eye_trace_validation": validation,
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
    "EYE_TRACE_ANGLE_CHANNELS",
    "EYE_TRACE_EXPORT_SCHEMA_ID",
    "EYE_TRACE_EXPORT_SCHEMA_VERSION",
    "EYE_TRACE_QA_CHANNELS",
    "EYE_TRACE_SCIENTIFIC_DTYPES",
    "export_eye_trace_samples",
    "eye_trace_parquet_policy",
    "eye_trace_projection_contract",
    "validate_eye_trace_export_payload",
]
