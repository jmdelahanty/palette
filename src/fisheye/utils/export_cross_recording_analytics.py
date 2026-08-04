"""Export cross-recording Palette analytics tables to Parquet.

The exporter treats Zarr archives as the source of truth and writes derived,
regenerable Parquet parts for cohort/population-level analysis. Extraction is
parallelized by recording. Final Parquet/manifest writes are coordinated in the
parent process so workers never append to the same file.
"""

from __future__ import annotations

from fisheye.shared.batch_logging import utc_now_compact
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import socket
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analytics_exports.capabilities import resolve_capabilities
from fisheye.analytics_exports.contracts import (
    ALL_TABLES,
    BASELINE_BEHAVIOR_SUMMARY_TABLE,
    BASELINE_BEHAVIOR_TIME_BINS_TABLE,
    BASELINE_KINEMATIC_SAMPLES_TABLE,
    BOUT_KINEMATICS_METRICS_TABLE,
    CHASER_BOUT_EVENTS_TABLE,
    CHASER_BOUT_HISTOGRAM_TABLE,
    CHASER_CENTER_DISTANCE_HISTOGRAM_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_DISTANCE_CDF_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_CHASER_PHASE_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_RADIAL_DENSITY_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_SUMMARY_TABLE,
    CHASER_QUADRANT_OCCUPANCY_CHASER_PHASE_TABLE,
    CHASER_QUADRANT_OCCUPANCY_DENSITY_TABLE,
    CHASER_QUADRANT_OCCUPANCY_SUMMARY_TABLE,
    CHASER_DISTANCE_HISTOGRAM_TABLE,
    CHASER_DISTANCE_SUMMARY_TABLE,
    CHASER_EGOCENTRIC_HISTOGRAM_TABLE,
    CHASER_EGOCENTRIC_SUMMARY_TABLE,
    CHASER_EPOCH_BEHAVIOR_TABLE,
    CHASER_IBI_HISTOGRAM_TABLE,
    CHASER_SPATIAL_TABLE,
    CHASER_SPEED_DISTANCE_TABLE,
    DEFAULT_TABLES,
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
    POSITION_OCCUPANCY_HISTOGRAM_TABLE,
    STIMULUS_RESPONSE_TABLE,
    SWIM_BOUT_METRICS_TABLE,
    TABLE_CONTRACTS,
    TRACE_TABLES,
    canonicalize_export_row,
    contract_snapshot,
    validate_table_columns,
)
from fisheye.analytics_exports.arrow_contracts import (
    ARROW_TABLE_CONTRACTS,
    arrow_contract_envelope,
    exact_arrow_schema,
)
from fisheye.analytics_exports.publication import (
    PUBLICATION_SCHEMA_ID,
    PUBLICATION_SCHEMA_VERSION,
    commit_staged_publication,
    export_manifest_path,
    generation_relative_path,
    manifest_identity,
    publication_generation_root,
    publication_staging_root,
    safe_component,
    sha256_file,
)
from fisheye.analytics_exports.baseline import (
    BaselineArrays,
    BaselineWindow,
    build_sample_metrics,
    build_summary_metrics,
    build_time_bin_metrics,
    is_baseline_label,
)
from fisheye.analysis.bout_kinematics import resolve_bout_kinematics_tables
from fisheye.analysis.chaser_behavior import canonical_behavior_label
from fisheye.analysis.chaser_distance_io import (
    ChaserDistanceReadError,
    load_chaser_distance_run,
)
from fisheye.analysis.chaser_quadrant_occupancy import (
    QUADRANT_LABELS,
    quadrant_code_for_xy,
)
from fisheye.shared.zarr.columnar import load_structured_dataset
from fisheye.shared.zarr.stimulus_response_schema import (
    CONCENTRIC_PER_FISH,
    GRATING_PER_FISH,
    MOVING_OMR_PER_FISH,
    RADIAL_PER_FISH,
    STEP_BOUT_SUMMARY,
    STEP_PER_FISH_BASE,
)
from fisheye.analysis.swim_bout_io import (
    SwimBoutIOError,
    load_default_swim_bout_tables,
    structured_records_to_dicts,
)
from fisheye.analysis.stimulus_response_io import resolve_stimulus_response_tables
from fisheye.analysis.track_kinematics_io import load_track_kinematics_track
from fisheye.registry.db import Registry, RegistryPaths
from fisheye.shared.json_safety import decode_null_terminated_text, json_attr_safe, strict_json_dumps
from fisheye.shared.zarr_helpers import resolve_zarr_run
from fisheye.shared.zarr_run_completion import (
    is_run_complete_in_parent,
    is_run_selector_eligible,
    resolve_latest_complete_run_name,
)
from fisheye.utils.index_analytics_manifests import index_export_manifest
from fisheye.shared.system_metadata import get_git_info
from fisheye.utils.virtual_collection_manifest import load_manifest, verify_manifest_sha256
from fisheye.shared.zarr_io import open_zarr_root


# Full-duration framewise trace products use dedicated bounded streaming
# exporters.  The compact cross-recording exporter must not accept them and
# silently construct an empty generation through its row-accumulating path.
AVAILABLE_TABLES = tuple(table for table in ALL_TABLES if table not in TRACE_TABLES)

# The recording-Zarr extraction functions predate the protocol-neutral V2
# Parquet contract. These names are internal source adapters only; callers and
# written exports accept canonical V2 names exclusively.
_SOURCE_TABLE_BY_V2 = {
    CHASER_SPATIAL_TABLE: "goodcopbadcop_spatial_occupancy_zones",
    CHASER_DISTANCE_SUMMARY_TABLE: "goodcopbadcop_chaser_epoch_summary",
    CHASER_EPOCH_BEHAVIOR_TABLE: "goodcopbadcop_epoch_behavior_summary",
    CHASER_BOUT_EVENTS_TABLE: "goodcopbadcop_epoch_bout_distribution",
    CHASER_BOUT_HISTOGRAM_TABLE: "goodcopbadcop_epoch_bout_histogram",
    CHASER_IBI_HISTOGRAM_TABLE: "goodcopbadcop_epoch_inter_bout_interval_histogram",
    CHASER_CENTER_DISTANCE_HISTOGRAM_TABLE: "goodcopbadcop_epoch_center_distance_histogram",
    CHASER_SPEED_DISTANCE_TABLE: "goodcopbadcop_speed_distance_bins",
    CHASER_DISTANCE_HISTOGRAM_TABLE: "goodcopbadcop_chaser_distance_histogram",
    CHASER_QUADRANT_OCCUPANCY_SUMMARY_TABLE: "goodcopbadcop_cra_primary_endpoint_summary",
    CHASER_QUADRANT_OCCUPANCY_CHASER_PHASE_TABLE: "goodcopbadcop_cra_primary_endpoint_object_phase",
    CHASER_QUADRANT_OCCUPANCY_DENSITY_TABLE: "goodcopbadcop_cra_quadrant_occupancy",
    CHASER_NEAR_FIELD_OCCUPANCY_SUMMARY_TABLE: "goodcopbadcop_cra_near_field_summary",
    CHASER_NEAR_FIELD_OCCUPANCY_CHASER_PHASE_TABLE: "goodcopbadcop_cra_near_field_object_phase",
    CHASER_NEAR_FIELD_OCCUPANCY_RADIAL_DENSITY_TABLE: "goodcopbadcop_cra_near_field_radial_density",
    CHASER_NEAR_FIELD_OCCUPANCY_DISTANCE_CDF_TABLE: "goodcopbadcop_cra_near_field_distance_cdf",
    CHASER_EGOCENTRIC_SUMMARY_TABLE: "goodcopbadcop_egocentric_epoch_summary",
    CHASER_EGOCENTRIC_HISTOGRAM_TABLE: "goodcopbadcop_egocentric_distance_bearing_histogram",
}
_V2_TABLE_BY_SOURCE = {source: target for target, source in _SOURCE_TABLE_BY_V2.items()}

# These tables are read from persisted chaser-distance summaries or nested
# derived components.  The canonical chaser-distance seal does not currently
# protect those derived semantics.  Normal cohort export must therefore report
# them unavailable instead of reopening the selected child as a raw Zarr group.
# The spatial occupancy table is intentionally absent: it is sourced from the
# separately selected detection-occupancy family, not chaser-distance data.
_UNSEALED_CHASER_EXPORT_V2_TABLES = frozenset(
    {
        CHASER_DISTANCE_SUMMARY_TABLE,
        CHASER_EPOCH_BEHAVIOR_TABLE,
        CHASER_BOUT_EVENTS_TABLE,
        CHASER_BOUT_HISTOGRAM_TABLE,
        CHASER_IBI_HISTOGRAM_TABLE,
        CHASER_CENTER_DISTANCE_HISTOGRAM_TABLE,
        CHASER_SPEED_DISTANCE_TABLE,
        CHASER_DISTANCE_HISTOGRAM_TABLE,
        CHASER_QUADRANT_OCCUPANCY_SUMMARY_TABLE,
        CHASER_QUADRANT_OCCUPANCY_CHASER_PHASE_TABLE,
        CHASER_QUADRANT_OCCUPANCY_DENSITY_TABLE,
        CHASER_NEAR_FIELD_OCCUPANCY_SUMMARY_TABLE,
        CHASER_NEAR_FIELD_OCCUPANCY_CHASER_PHASE_TABLE,
        CHASER_NEAR_FIELD_OCCUPANCY_RADIAL_DENSITY_TABLE,
        CHASER_NEAR_FIELD_OCCUPANCY_DISTANCE_CDF_TABLE,
        CHASER_EGOCENTRIC_SUMMARY_TABLE,
        CHASER_EGOCENTRIC_HISTOGRAM_TABLE,
    }
)
_UNSEALED_CHASER_EXPORT_SOURCE_TABLES = frozenset(
    _SOURCE_TABLE_BY_V2[table] for table in _UNSEALED_CHASER_EXPORT_V2_TABLES
)
_UNSEALED_CHASER_EXPORT_REASON = (
    "requested chaser-distance summary/component has no independently verified "
    "sealed semantic authority; raw Zarr export is unavailable"
)

CRA_PRIMARY_ENDPOINT_COMPONENT_PARENT = "chaser_quadrant_occupancy"
CRA_PRIMARY_ENDPOINT_ALLOWED_STATUSES = {"computed", "complete"}
CRA_NEAR_FIELD_COMPONENT_PARENT = "chaser_near_field_occupancy"
CRA_NEAR_FIELD_ALLOWED_STATUSES = {"computed", "complete"}
EPOCH_BEHAVIOR_COMPONENT_PARENT = "epoch_behavior_summary"
EPOCH_BEHAVIOR_SCHEMA_ID = "palette.chaser.epoch_behavior_summary.v1"


@dataclass(frozen=True)
class StepSpan:
    step_index: int
    stimulus_mode: str | None
    step_name: str | None
    start_frame: int | None
    end_frame: int | None


@dataclass(frozen=True)
class ProtocolSignature:
    schema: str
    hash: str
    mode_sequence: str | None
    duration_sequence_s: str | None
    step_count: int


@dataclass
class SourceExportResult:
    zarr_path: str
    recording_id: str
    rows_by_table: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class CollectionManifestSummary:
    path: str
    collection_id: str
    collection_name: str | None
    manifest_sha256: str
    schema_id: str | None
    schema_version: int | None
    record_count: int
    included_record_count: int


def _recording_id_from_path(zarr_path: Path) -> str:
    name = zarr_path.name
    if name.endswith(".zarr"):
        name = name[:-5]
    if name.endswith("_analysis"):
        name = name[:-9]
    return name


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _scalar_for_parquet(value: Any) -> Any:
    """Convert NumPy/Zarr values into strict scalar Parquet-friendly values."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return decode_null_terminated_text(value, errors="ignore")
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _scalar_for_parquet(value.item())
        return _json_dumps_safe(value.tolist())
    if isinstance(value, Mapping) or isinstance(value, (list, tuple)):
        return _json_dumps_safe(value)
    return str(value)


_json_safe = json_attr_safe


def _json_dumps_safe(value: Any) -> str:
    return strict_json_dumps(value)


def _hash_payload(payload: Mapping[str, Any]) -> str:
    blob = _json_dumps_safe(payload).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _load_collection_manifest(path: Path) -> dict[str, Any]:
    manifest = load_manifest(path.expanduser())
    if not verify_manifest_sha256(manifest):
        raise ValueError(f"collection manifest hash check failed: {path}")
    return manifest


def _collection_manifest_summary(path: Path) -> CollectionManifestSummary:
    manifest = _load_collection_manifest(path)
    records = manifest.get("records", [])
    if not isinstance(records, list):
        records = []
    included = [
        record
        for record in records
        if isinstance(record, Mapping)
        and isinstance(record.get("status"), Mapping)
        and record["status"].get("included") is True
    ]
    return CollectionManifestSummary(
        path=str(path.expanduser().resolve()),
        collection_id=str(manifest["collection_id"]),
        collection_name=(
            str(manifest["collection_name"])
            if manifest.get("collection_name") is not None
            else None
        ),
        manifest_sha256=str(manifest["manifest_sha256"]),
        schema_id=str(manifest.get("schema_id")) if manifest.get("schema_id") is not None else None,
        schema_version=_safe_int(manifest.get("schema_version")),
        record_count=len(records),
        included_record_count=len(included),
    )


def _source_paths_from_collection_manifest(path: Path) -> list[Path]:
    manifest = _load_collection_manifest(path)
    paths: list[Path] = []
    for record in manifest.get("records", []):
        if not isinstance(record, Mapping):
            continue
        status = record.get("status")
        if not isinstance(status, Mapping) or status.get("included") is not True:
            continue
        locator = record.get("locator_at_selection")
        if not isinstance(locator, Mapping):
            continue
        uri = locator.get("uri")
        if isinstance(uri, str) and uri:
            paths.append(Path(uri).expanduser().resolve())
    return paths


def _parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _protocol_signature_row(protocol_signature: ProtocolSignature | None) -> dict[str, Any]:
    if protocol_signature is None:
        return {
            "protocol_signature_schema": None,
            "protocol_signature_hash": None,
            "derived_protocol_hash": None,
            "protocol_mode_sequence": None,
            "protocol_duration_sequence_s": None,
            "protocol_step_count": None,
        }
    return {
        "protocol_signature_schema": protocol_signature.schema,
        "protocol_signature_hash": protocol_signature.hash,
        # Temporary alias for the ad hoc analysis naming used before the
        # exporter persisted protocol signatures directly.
        "derived_protocol_hash": protocol_signature.hash,
        "protocol_mode_sequence": protocol_signature.mode_sequence,
        "protocol_duration_sequence_s": protocol_signature.duration_sequence_s,
        "protocol_step_count": protocol_signature.step_count,
    }


def _group_names(group: Any) -> list[str]:
    if hasattr(group, "group_keys"):
        return sorted(str(name) for name in group.group_keys())
    return sorted(str(name) for name in group.keys() if hasattr(group[name], "attrs"))


def _array_names(group: Any) -> list[str]:
    if hasattr(group, "array_keys"):
        return sorted(str(name) for name in group.array_keys())
    names: list[str] = []
    for name in group.keys():
        try:
            item = group[name]
            if not hasattr(item, "attrs") or not hasattr(item, "keys"):
                names.append(str(name))
        except Exception:
            continue
    return sorted(names)


def _has_child(group: Any, name: str) -> bool:
    try:
        return name in group
    except Exception:
        return False


def _attrs_dict(group: Any) -> dict[str, Any]:
    try:
        return {str(key): _scalar_for_parquet(value) for key, value in dict(group.attrs).items()}
    except Exception:
        return {}


def _row_count_from_group(group: Any, names: Sequence[str] | None = None) -> int:
    candidates = list(names) if names is not None else _array_names(group)
    for name in candidates:
        if not _has_child(group, name):
            continue
        try:
            arr = group[name]
            shape = tuple(arr.shape)
            if shape:
                return int(shape[0])
        except Exception:
            continue
    return 0


def _read_1d_array(group: Any, name: str) -> np.ndarray | None:
    if not _has_child(group, name):
        return None
    try:
        arr = np.asarray(group[name][:])
    except Exception:
        return None
    if arr.ndim != 1:
        return None
    return arr


def _read_array(group: Any, name: str) -> np.ndarray | None:
    if not _has_child(group, name):
        return None
    try:
        return np.asarray(group[name][:])
    except Exception:
        return None


def _decode_text_column(values: Any, *, fallback_count: int = 0) -> list[str | None]:
    if values is None:
        return [None] * int(fallback_count)
    arr = np.asarray(values)
    if arr.ndim == 0:
        return [decode_null_terminated_text(arr.item(), errors="ignore")]
    if arr.ndim == 2 and arr.dtype.kind in ("u", "i"):
        return [decode_null_terminated_text(row, errors="ignore") for row in arr]
    flat = arr.reshape(-1)
    return [decode_null_terminated_text(value, errors="ignore") for value in flat]


def _read_canonical_behavior_labels(objects: Any) -> list[str | None]:
    values = _read_array(objects, "behavior_class_label_bytes")
    if values is None:
        values = _read_array(objects, "object_role_label_bytes")
    return [canonical_behavior_label(value) if value else None for value in _decode_text_column(values)]


def _array_scalar(values: np.ndarray | None, *indices: int) -> Any:
    if values is None:
        return None
    arr = np.asarray(values)
    if len(indices) > arr.ndim:
        return None
    for axis, index in enumerate(indices):
        if index < 0 or index >= int(arr.shape[axis]):
            return None
    try:
        return _scalar_for_parquet(arr[indices])
    except Exception:
        return None


def _array_float(values: np.ndarray | None, *indices: int) -> float | None:
    return _safe_float(_array_scalar(values, *indices))


def _array_int(values: np.ndarray | None, *indices: int) -> int | None:
    return _safe_int(_array_scalar(values, *indices))


def _json_mapping_attr(attrs: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = _parse_jsonish(attrs.get(name))
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    return {}


def _source_refs_json(source_refs: Mapping[str, Any]) -> str:
    return _json_dumps_safe(dict(source_refs))


def _window_rows_from_group(group: Any) -> list[dict[str, Any]]:
    if not _has_child(group, "windows"):
        return []
    windows = group["windows"]
    n_rows = _row_count_from_group(
        windows,
        names=("window_id", "label_bytes", "start_frame", "end_frame", "start_time_s", "end_time_s", "duration_s"),
    )
    if n_rows <= 0:
        return []
    labels = _decode_text_column(_read_array(windows, "label_bytes"), fallback_count=n_rows)
    window_id = _read_1d_array(windows, "window_id")
    start_frame = _read_1d_array(windows, "start_frame")
    end_frame = _read_1d_array(windows, "end_frame")
    start_time = _read_1d_array(windows, "start_time_s")
    end_time = _read_1d_array(windows, "end_time_s")
    duration = _read_1d_array(windows, "duration_s")

    rows: list[dict[str, Any]] = []
    for idx in range(n_rows):
        start_time_s = _array_float(start_time, idx)
        end_time_s = _array_float(end_time, idx)
        duration_s = _array_float(duration, idx)
        if duration_s is None and start_time_s is not None and end_time_s is not None:
            duration_s = end_time_s - start_time_s
        rows.append(
            {
                "window_index": idx,
                "window_id": _array_int(window_id, idx) if window_id is not None else idx,
                "window_label": labels[idx] if idx < len(labels) else None,
                "start_frame": _array_int(start_frame, idx),
                "end_frame": _array_int(end_frame, idx),
                "start_time_s": start_time_s,
                "end_time_s": end_time_s,
                "duration_s": duration_s,
            }
        )
    return rows


def _epoch_summary_window_rows(group: Any) -> list[dict[str, Any]]:
    if not _has_child(group, "epoch_summary"):
        return []
    summary = group["epoch_summary"]
    n_rows = _row_count_from_group(
        summary,
        names=("window_id", "label_bytes", "start_frame", "end_frame"),
    )
    if n_rows <= 0:
        return []
    labels = _decode_text_column(_read_array(summary, "label_bytes"), fallback_count=n_rows)
    window_id = _read_1d_array(summary, "window_id")
    start_frame = _read_1d_array(summary, "start_frame")
    end_frame = _read_1d_array(summary, "end_frame")
    start_time = _read_1d_array(summary, "start_time_s")
    end_time = _read_1d_array(summary, "end_time_s")
    duration = _read_1d_array(summary, "duration_s")

    rows: list[dict[str, Any]] = []
    for idx in range(n_rows):
        start_time_s = _array_float(start_time, idx)
        end_time_s = _array_float(end_time, idx)
        duration_s = _array_float(duration, idx)
        if duration_s is None and start_time_s is not None and end_time_s is not None:
            duration_s = end_time_s - start_time_s
        rows.append(
            {
                "window_index": idx,
                "window_id": _array_int(window_id, idx) if window_id is not None else idx,
                "window_label": labels[idx] if idx < len(labels) else None,
                "start_frame": _array_int(start_frame, idx),
                "end_frame": _array_int(end_frame, idx),
                "start_time_s": start_time_s,
                "end_time_s": end_time_s,
                "duration_s": duration_s,
            }
        )
    return rows


def _read_table_rows(
    group: Any,
    *,
    include_arrays: Sequence[str] | None = None,
    exclude_arrays: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Read scalar 1D arrays from a Zarr group into row dictionaries."""

    names = list(include_arrays) if include_arrays is not None else _array_names(group)
    excluded = {str(name) for name in exclude_arrays}
    arrays: dict[str, np.ndarray] = {}
    n_rows: int | None = None
    for name in names:
        if name in excluded:
            continue
        arr = _read_1d_array(group, name)
        if arr is None:
            continue
        if n_rows is None:
            n_rows = int(arr.shape[0])
        if int(arr.shape[0]) != n_rows:
            continue
        arrays[name] = arr

    if not arrays or n_rows is None:
        return []

    rows: list[dict[str, Any]] = []
    for idx in range(n_rows):
        rows.append({name: _scalar_for_parquet(arr[idx]) for name, arr in arrays.items()})
    return rows


def _array_mapping_rows(mapping: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    """Read equal-length 1D arrays from a logical table mapping into rows."""

    arrays: dict[str, np.ndarray] = {}
    n_rows: int | None = None
    for name, value in mapping.items():
        arr = np.asarray(value)
        if arr.ndim != 1:
            continue
        if n_rows is None:
            n_rows = int(arr.shape[0])
        if int(arr.shape[0]) != n_rows:
            continue
        arrays[str(name)] = arr

    if not arrays or n_rows is None:
        return []

    return [
        {name: _scalar_for_parquet(arr[idx]) for name, arr in arrays.items()}
        for idx in range(n_rows)
    ]


def _latest_run(root: Any, parent_path: str, requested: str | None = None) -> tuple[Any | None, str | None, str | None]:
    if str(parent_path).strip("/") == "analysis/swim_bout_runs":
        try:
            parent = root[parent_path]
            selector = str(requested or "latest").strip() or "latest"
            if selector == "latest":
                name = resolve_latest_complete_run_name(parent, legacy_default=False)
                if name is None:
                    raise ValueError(
                        "no complete, selector-eligible swim-bout run is selected"
                    )
            else:
                name = selector
                child = parent[name]
                if not is_run_selector_eligible(child) or not is_run_complete_in_parent(
                    parent, child, legacy_default=False
                ):
                    raise ValueError(
                        f"swim-bout run {name!r} is incomplete or selector-ineligible"
                    )
            return parent[name], name, None
        except Exception as exc:
            return None, None, f"canonical swim-bout selection failed closed: {exc}"
    if str(parent_path).strip("/") == "analysis/chaser_distance_runs":
        # This compatibility-shaped helper must never return a raw canonical
        # chaser-distance child.  Direct callers still cross the exact typed
        # selector boundary, then receive an explicit unavailability result for
        # the unsealed derived surface they were about to inspect.
        try:
            snapshot = load_chaser_distance_run(
                root,
                run_name=str(requested or "latest").strip() or "latest",
            )
        except ChaserDistanceReadError as exc:
            return None, None, f"canonical chaser-distance preflight failed closed: {exc}"
        return (
            None,
            snapshot.run_name,
            f"{_UNSEALED_CHASER_EXPORT_REASON} ({snapshot.run_path})",
        )
    try:
        group, name = resolve_zarr_run(
            root,
            parent_path,
            run_name=requested,
            fallback_to_latest=True,
            fallback_to_sorted="last",
        )
        return group, name, None
    except Exception as exc:
        return None, None, str(exc)


def _preflight_unsealed_chaser_exports(
    root: Any,
    *,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> set[str]:
    """Remove unsupported chaser-derived exports after one exact preflight."""

    requested = sorted(tables & _UNSEALED_CHASER_EXPORT_SOURCE_TABLES)
    if not requested:
        return tables
    run_name: str | None = None
    run_path: str | None = None
    try:
        snapshot = load_chaser_distance_run(root, run_name="latest")
    except ChaserDistanceReadError as exc:
        reason = f"canonical chaser-distance preflight failed closed: {exc}"
    else:
        run_name = snapshot.run_name
        run_path = snapshot.run_path
        reason = f"{_UNSEALED_CHASER_EXPORT_REASON} ({run_path})"
    for table in requested:
        diagnostic: dict[str, Any] = {
            "table": table,
            "status": "unavailable",
            "reason": reason,
        }
        if run_name is not None:
            diagnostic["chaser_distance_run"] = run_name
        if run_path is not None:
            diagnostic["chaser_distance_path"] = run_path
        diagnostics.append(diagnostic)
    return tables - _UNSEALED_CHASER_EXPORT_SOURCE_TABLES


def _common_row(
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    table: str,
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "table_name": table,
        "recording_id": recording_id,
        "zarr_path": str(zarr_path),
        "source_lineage_hash": _hash_payload(lineage),
    }


def _summarize_numeric(values: np.ndarray | None, op: str) -> float | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    if op == "sum":
        return float(np.sum(arr))
    if op == "mean":
        return float(np.mean(arr))
    if op == "median":
        return float(np.median(arr))
    raise ValueError(f"Unsupported summary op: {op}")


def _load_stimulus_steps(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> tuple[str | None, list[StepSpan], list[dict[str, Any]], ProtocolSignature | None]:
    rows: list[dict[str, Any]] = []
    spans: list[StepSpan] = []
    signature_steps: list[dict[str, Any]] = []
    stim_group, stim_run, error = _latest_run(root, "analysis/stimulus_runs")
    if stim_group is None or stim_run is None:
        diagnostics.append({"table": "stimulus_steps", "status": "skipped", "reason": error})
        return None, spans, rows, None

    if not _has_child(stim_group, "steps"):
        diagnostics.append({"table": "stimulus_steps", "status": "skipped", "reason": "missing steps group"})
        return stim_run, spans, rows, None

    steps_group = stim_group["steps"]
    step_names = sorted(
        (name for name in _group_names(steps_group) if re.fullmatch(r"step_\d+", name)),
        key=lambda item: int(item.split("_")[1]),
    )

    for name in step_names:
        step_group = steps_group[name]
        attrs = _attrs_dict(step_group)
        idx = _safe_int(attrs.get("step_index"))
        if idx is None:
            idx = int(name.split("_")[1])
        start_frame = _safe_int(attrs.get("start_frame"))
        if start_frame is None:
            start_frame = _safe_int(attrs.get("start_camera_frame"))
        end_frame = _safe_int(attrs.get("end_frame"))
        if end_frame is None:
            end_frame = _safe_int(attrs.get("end_camera_frame"))
        stimulus_mode = attrs.get("stimulus_mode")
        step_name = attrs.get("step_name") or attrs.get("name")
        spans.append(StepSpan(
            step_index=idx,
            stimulus_mode=str(stimulus_mode) if stimulus_mode is not None else None,
            step_name=str(step_name) if step_name is not None else None,
            start_frame=start_frame,
            end_frame=end_frame,
        ))

        signature_step: dict[str, Any] = {
            "step_index": idx,
            "step_name": step_name,
            "stimulus_mode": stimulus_mode,
            "stimulus_mode_id": attrs.get("stimulus_mode_id"),
            "duration_s": _safe_float(attrs.get("duration_s")),
            "stimulus_params": _parse_jsonish(
                attrs.get("stimulus_params") or attrs.get("raw_protocol_params_json")
            ),
        }
        for child_name in ("moving_grating", "concentric_grating", "looming_dot"):
            if _has_child(step_group, child_name):
                signature_step[child_name] = {
                    key: _parse_jsonish(value)
                    for key, value in sorted(_attrs_dict(step_group[child_name]).items())
                }
        signature_steps.append(signature_step)

        if "stimulus_steps" not in tables:
            continue

        lineage = {
            "zarr_path": str(zarr_path),
            "stimulus_run": stim_run,
            "step_index": idx,
            "stimulus_run_schema_version": stim_group.attrs.get("schema_version"),
        }
        row = _common_row(
            export_run_id=export_run_id,
            zarr_path=zarr_path,
            recording_id=recording_id,
            table="stimulus_steps",
            lineage=lineage,
        )
        row.update({
            "stimulus_run": stim_run,
            "step_index": idx,
            "step_group": name,
            "step_name": step_name,
            "stimulus_mode": stimulus_mode,
            "stimulus_mode_id": attrs.get("stimulus_mode_id"),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_camera_frame": _safe_int(attrs.get("start_camera_frame")) or start_frame,
            "end_camera_frame": _safe_int(attrs.get("end_camera_frame")) or end_frame,
            "duration_s": _safe_float(attrs.get("duration_s")),
            "stimulus_params_json": attrs.get("stimulus_params") or attrs.get("raw_protocol_params_json"),
        })
        if isinstance(row["stimulus_params_json"], str):
            # Already serialized by _attrs_dict when attrs contained a dict/list.
            pass
        elif row["stimulus_params_json"] is not None:
            row["stimulus_params_json"] = _json_dumps_safe(row["stimulus_params_json"])

        for child_name in ("moving_grating", "concentric_grating", "looming_dot"):
            if not _has_child(step_group, child_name):
                continue
            prefix = child_name
            for key, value in _attrs_dict(step_group[child_name]).items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)

    if signature_steps:
        payload = {
            "schema": "palette_protocol_signature_v1",
            "stimulus_run_schema_version": _scalar_for_parquet(stim_group.attrs.get("schema_version")),
            "steps": signature_steps,
        }
        step_modes = [
            str(step["stimulus_mode"])
            for step in signature_steps
            if step.get("stimulus_mode") is not None
        ]
        step_durations = [
            str(step["duration_s"])
            for step in signature_steps
            if step.get("duration_s") is not None
        ]
        protocol_signature = ProtocolSignature(
            schema="palette_protocol_signature_v1",
            hash=_hash_payload(payload),
            mode_sequence=" -> ".join(step_modes) if step_modes else None,
            duration_sequence_s=",".join(step_durations) if step_durations else None,
            step_count=len(signature_steps),
        )
    else:
        protocol_signature = None

    signature_row = _protocol_signature_row(protocol_signature)
    for row in rows:
        row.update(signature_row)

    return stim_run, spans, rows, protocol_signature


def _assign_step(start_frame: int | None, end_frame: int | None, spans: Sequence[StepSpan]) -> StepSpan | None:
    if start_frame is None and end_frame is None:
        return None
    best: tuple[int, StepSpan] | None = None
    bout_start = start_frame if start_frame is not None else end_frame
    bout_end = end_frame if end_frame is not None else start_frame
    if bout_start is None or bout_end is None:
        return None
    for span in spans:
        if span.start_frame is None or span.end_frame is None:
            continue
        overlap = max(0, min(bout_end, span.end_frame) - max(bout_start, span.start_frame))
        if overlap > 0 and (best is None or overlap > best[0]):
            best = (overlap, span)
    if best is not None:
        return best[1]
    for span in spans:
        if span.start_frame is not None and span.end_frame is not None and span.start_frame <= bout_start < span.end_frame:
            return span
    return None


def _load_recording_summary(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    stimulus_run: str | None,
    protocol_signature: ProtocolSignature | None,
    step_count: int,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if "recording_summary" not in tables:
        return []

    stim_resp_group, stim_resp_run, stim_resp_error = _latest_run(root, "analysis/stimulus_response_runs")
    swim_group, swim_run, _swim_error = _latest_run(root, "analysis/swim_bout_runs")

    lineage = {
        "zarr_path": str(zarr_path),
        "stimulus_run": stimulus_run,
        "stimulus_response_run": stim_resp_run,
        "swim_bout_run": swim_run,
    }
    row = _common_row(
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        table="recording_summary",
        lineage=lineage,
    )
    row.update({
        "stimulus_run": stimulus_run,
        "stimulus_response_run": stim_resp_run,
        "swim_bout_run": swim_run,
        "stimulus_step_count": step_count,
    })
    row.update(_protocol_signature_row(protocol_signature))

    if stim_resp_group is not None:
        attrs = _attrs_dict(stim_resp_group)
        row.update({
            "source_track_kinematics_run": attrs.get("source_track_kinematics_run"),
            "source_track_kinematics_type": attrs.get("source_track_kinematics_type"),
            "source_bout_run": attrs.get("source_bout_run"),
            "n_fish": attrs.get("n_fish"),
            "n_steps": attrs.get("n_steps"),
        })
        try:
            stim_resp_tables = resolve_stimulus_response_tables(stim_resp_group)
            global_metrics = stim_resp_tables.global_per_fish
            fish_ids = np.asarray(global_metrics.get("fish_id", []))
            row["global_fish_count"] = int(fish_ids.size) if fish_ids.size else None
            row["total_distance_mm_sum"] = _summarize_numeric(global_metrics.get("total_distance_mm"), "sum")
            row["mean_speed_mm_s_mean"] = _summarize_numeric(global_metrics.get("mean_speed_mm_s"), "mean")
            row["fraction_moving_mean"] = _summarize_numeric(global_metrics.get("fraction_moving"), "mean")
            row["total_active_s_sum"] = _summarize_numeric(global_metrics.get("total_active_s"), "sum")
        except Exception as exc:
            diagnostics.append({
                "table": "recording_summary",
                "status": "partial",
                "reason": f"failed to resolve stimulus-response global metrics: {exc}",
            })
    elif stim_resp_error:
        diagnostics.append({"table": "recording_summary", "status": "partial", "reason": stim_resp_error})

    if swim_group is not None:
        default_level = str(swim_group.attrs.get("default_level", ""))
        row["swim_bout_default_level"] = default_level or None
        if default_level and _has_child(swim_group, default_level):
            level = swim_group[default_level]
            row["swim_bout_default_n_bouts"] = _safe_int(level.attrs.get("n_bouts"))
            row["swim_bout_default_mean_duration_s"] = _safe_float(level.attrs.get("mean_bout_duration_s"))
            row["swim_bout_default_total_path_length_mm"] = _safe_float(level.attrs.get("total_path_length_mm"))

    return [row]


def _load_stimulus_response_tables(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    protocol_signature: ProtocolSignature | None,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    step_summary_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    wanted = {"stimulus_step_summary", "stimulus_response_per_fish_step"} & tables
    if not wanted:
        return step_summary_rows, response_rows

    response_group, response_run, error = _latest_run(root, "analysis/stimulus_response_runs")
    if response_group is None or response_run is None:
        for table in wanted:
            diagnostics.append({"table": table, "status": "skipped", "reason": error})
        return step_summary_rows, response_rows

    try:
        response_tables = resolve_stimulus_response_tables(response_group)
    except Exception as exc:
        for table in wanted:
            diagnostics.append({
                "table": table,
                "status": "skipped",
                "reason": f"failed to resolve stimulus-response tables: {exc}",
            })
        return step_summary_rows, response_rows

    response_attrs = _attrs_dict(response_group)
    for step in response_tables.steps:
        idx = step.step_index
        if not step.per_fish:
            continue
        _validate_logical_mapping_fields(
            step.per_fish,
            allowed={name for name, _field in (*STEP_PER_FISH_BASE, *STEP_BOUT_SUMMARY)},
            context=f"stimulus-response step {idx} per_fish",
        )
        _validate_logical_mapping_fields(
            step.grating_per_fish,
            allowed={name for name, _field in GRATING_PER_FISH},
            context=f"stimulus-response step {idx} grating_per_fish",
        )
        _validate_logical_mapping_fields(
            step.concentric_per_fish,
            allowed={name for name, _field in CONCENTRIC_PER_FISH},
            context=f"stimulus-response step {idx} concentric_per_fish",
        )
        if step.moving_grating_omr is not None:
            _validate_logical_mapping_fields(
                step.moving_grating_omr.per_fish,
                allowed={name for name, _field in MOVING_OMR_PER_FISH},
                context=f"stimulus-response step {idx} moving OMR per_fish",
            )
        if step.concentric_radial_omr is not None:
            _validate_logical_mapping_fields(
                step.concentric_radial_omr.per_fish,
                allowed={name for name, _field in RADIAL_PER_FISH},
                context=f"stimulus-response step {idx} radial OMR per_fish",
            )
        attrs = dict(step.attrs)
        base_rows = _array_mapping_rows(step.per_fish)
        for base in base_rows:
            fish_id = _safe_int(base.get("fish_id"))
            lineage = {
                "zarr_path": str(zarr_path),
                "stimulus_response_run": response_run,
                "source_stimulus_run": response_attrs.get("source_stimulus_run"),
                "source_track_kinematics_run": response_attrs.get("source_track_kinematics_run"),
                "source_bout_run": response_attrs.get("source_bout_run"),
                "step_index": idx,
                "fish_id": fish_id,
            }
            common = {
                "stimulus_response_run": response_run,
                "source_stimulus_run": response_attrs.get("source_stimulus_run"),
                "source_track_kinematics_run": response_attrs.get("source_track_kinematics_run"),
                "source_track_kinematics_type": response_attrs.get("source_track_kinematics_type"),
                "source_bout_run": response_attrs.get("source_bout_run"),
                "step_index": idx,
                "step_name": step.step_name,
                "stimulus_mode": step.stimulus_mode,
                "stimulus_mode_id": step.stimulus_mode_id,
                "start_frame": step.start_frame,
                "end_frame": step.end_frame,
                "start_camera_frame": _safe_int(attrs.get("start_camera_frame")) or step.start_frame,
                "end_camera_frame": _safe_int(attrs.get("end_camera_frame")) or step.end_frame,
                "duration_s": step.duration_s,
            }
            common.update(_protocol_signature_row(protocol_signature))

            if "stimulus_step_summary" in tables:
                row = _common_row(
                    export_run_id=export_run_id,
                    zarr_path=zarr_path,
                    recording_id=recording_id,
                    table="stimulus_step_summary",
                    lineage=lineage,
                )
                row.update(common)
                row.update(base)
                step_summary_rows.append(row)

            if "stimulus_response_per_fish_step" not in tables:
                continue

            row = _common_row(
                export_run_id=export_run_id,
                zarr_path=zarr_path,
                recording_id=recording_id,
                table="stimulus_response_per_fish_step",
                lineage=lineage,
            )
            row.update(common)
            row.update(base)
            row["omr_family"] = None

            if step.grating_per_fish:
                grating_rows = _array_mapping_rows(step.grating_per_fish)
                _merge_matching_fish_row(
                    row,
                    grating_rows,
                    fish_id,
                    prefix="grating_",
                    table=STIMULUS_RESPONSE_TABLE,
                )
            if step.moving_grating_omr is not None:
                row["omr_family"] = "moving_grating_omr"
                row["omr_attr_method_version"] = _scalar_for_parquet(
                    step.moving_grating_omr.attrs.get("method_version")
                )
                omr_rows = _array_mapping_rows(step.moving_grating_omr.per_fish)
                _merge_matching_fish_row(
                    row,
                    omr_rows,
                    fish_id,
                    prefix="",
                    table=STIMULUS_RESPONSE_TABLE,
                )

            if step.concentric_per_fish:
                conc_rows = _array_mapping_rows(step.concentric_per_fish)
                _merge_matching_fish_row(
                    row,
                    conc_rows,
                    fish_id,
                    prefix="concentric_",
                    table=STIMULUS_RESPONSE_TABLE,
                )
            if step.concentric_radial_omr is not None:
                row["omr_family"] = "concentric_radial_omr"
                row["radial_omr_attr_method_version"] = _scalar_for_parquet(
                    step.concentric_radial_omr.attrs.get("method_version")
                )
                radial_rows = _array_mapping_rows(step.concentric_radial_omr.per_fish)
                _merge_matching_fish_row(
                    row,
                    radial_rows,
                    fish_id,
                    prefix="",
                    table=STIMULUS_RESPONSE_TABLE,
                )

            response_rows.append(row)

    return step_summary_rows, response_rows


def _validate_logical_mapping_fields(
    mapping: Mapping[str, np.ndarray],
    *,
    allowed: set[str],
    context: str,
) -> None:
    unexpected = sorted(set(str(name) for name in mapping) - allowed)
    if unexpected:
        raise ValueError(f"{context} has undeclared logical fields: {unexpected}")


def _merge_matching_fish_row(
    row: dict[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    fish_id: int | None,
    *,
    prefix: str,
    table: str,
) -> None:
    exact_names = {field.name for field in ARROW_TABLE_CONTRACTS[table].fields}
    matches = [
        candidate
        for candidate in candidate_rows
        if _safe_int(candidate.get("fish_id")) == fish_id
    ]
    if len(matches) > 1:
        raise ValueError(f"{table}: fish_id {fish_id!r} has multiple matching rows")
    for candidate in matches:
        for key, value in candidate.items():
            if key == "fish_id":
                continue
            out_key = f"{prefix}{key}" if prefix else str(key)
            if out_key not in exact_names:
                raise ValueError(
                    f"{table}: source field {key!r} projects to undeclared field "
                    f"{out_key!r}"
                )
            if out_key in row:
                if row[out_key] == value:
                    continue
                if row[out_key] is not None and value is not None:
                    raise ValueError(
                        f"{table}: conflicting projected field {out_key!r}: "
                        f"{row[out_key]!r} != {value!r}"
                    )
            row[out_key] = value
        return


def _load_swim_bout_metrics(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    stimulus_run: str | None,
    protocol_signature: ProtocolSignature | None,
    steps: Sequence[StepSpan],
    tables: set[str],
    diagnostics: list[dict[str, Any]],
    legacy_compatibility: bool = False,
) -> list[dict[str, Any]]:
    if "swim_bout_metrics" not in tables:
        return []

    _swim_group, selected_swim_run, selection_error = _latest_run(
        root,
        "analysis/swim_bout_runs",
    )
    if selected_swim_run is None:
        diagnostics.append({
            "table": "swim_bout_metrics",
            "status": "skipped",
            "reason": selection_error or "no canonical swim-bout run is selected",
        })
        return []
    try:
        swim_payload = load_default_swim_bout_tables(
            root,
            run_name=selected_swim_run,
            legacy_compatibility=legacy_compatibility,
        )
    except SwimBoutIOError as exc:
        diagnostics.append({
            "table": "swim_bout_metrics",
            "status": "skipped",
            "reason": str(exc),
        })
        return []
    except Exception as exc:
        diagnostics.append({
            "table": "swim_bout_metrics",
            "status": "skipped",
            "reason": f"failed to load swim-bout metrics: {exc}",
        })
        return []

    swim_run = swim_payload.run_name
    swim_attrs = dict(swim_payload.run_attrs)
    level_attrs = dict(swim_payload.signal_attrs)
    default_level = swim_payload.signal.speed_level
    bout_rows = structured_records_to_dicts(swim_payload.bouts)
    rows: list[dict[str, Any]] = []
    for raw_bout in bout_rows:
        bout = _closed_metric_record(
            raw_bout,
            table=SWIM_BOUT_METRICS_TABLE,
            context="swim-bout payload",
        )
        bout_id = _safe_int(bout.get("bout_id"))
        start_frame = _safe_int(bout.get("start_frame"))
        end_frame = _safe_int(bout.get("end_frame"))
        step = _assign_step(start_frame, end_frame, steps)
        lineage = {
            "zarr_path": str(zarr_path),
            "swim_bout_run": swim_run,
            "speed_level": default_level,
            "source_track_kinematics_run": swim_attrs.get("source_track_kinematics_run"),
            "track_id": swim_attrs.get("track_id"),
            "candidate_id": swim_payload.candidate.candidate_id,
            "signal_id": swim_payload.signal.signal_id,
            "bout_id": bout_id,
        }
        row = _common_row(
            export_run_id=export_run_id,
            zarr_path=zarr_path,
            recording_id=recording_id,
            table="swim_bout_metrics",
            lineage=lineage,
        )
        row.update({
            "stimulus_run": stimulus_run,
            "swim_bout_run": swim_run,
            "source_track_kinematics_run": swim_attrs.get("source_track_kinematics_run"),
            "source_track_kinematics_type": swim_attrs.get("source_track_kinematics_type"),
            "track_id": _safe_int(swim_attrs.get("track_id")),
            "speed_level": default_level,
            "candidate_id": int(swim_payload.candidate.candidate_id),
            "signal_id": int(swim_payload.signal.signal_id),
            "signal_role": swim_payload.signal.role,
            "signal_source_level": swim_payload.signal.source_level,
            "detection_method": swim_attrs.get("detection_method") or level_attrs.get("detection_method"),
            "detection_signal_transform_type": level_attrs.get("detection_signal_transform_type"),
            "detection_signal_source_level": level_attrs.get("detection_signal_source_level"),
            "movement_metric_source_level": level_attrs.get("movement_metric_source_level"),
            "threshold_mm_s": _safe_float(level_attrs.get("threshold_mm_s") or swim_attrs.get("threshold_mm_s")),
            "peak_prominence_mm_s": _safe_float(level_attrs.get("peak_prominence_mm_s") or swim_attrs.get("peak_prominence_mm_s")),
            "step_index": step.step_index if step else None,
            "step_name": step.step_name if step else None,
            "stimulus_mode": step.stimulus_mode if step else None,
        })
        row.update(_protocol_signature_row(protocol_signature))
        _merge_identity_checked(row, bout, context="swim-bout payload")
        rows.append(row)
    return rows


def _closed_metric_record(
    record: Mapping[str, Any],
    *,
    table: str,
    context: str,
    semantic_fixed_text_names: bool = False,
) -> dict[str, Any]:
    """Normalize one metric record without permitting schema-by-observation."""

    exact_names = {field.name for field in ARROW_TABLE_CONTRACTS[table].fields}
    out: dict[str, Any] = {}
    for raw_name, raw_value in record.items():
        name = str(raw_name)
        if semantic_fixed_text_names and name.endswith("_bytes"):
            name = name[: -len("_bytes")]
        if name in out:
            raise ValueError(f"{context} maps multiple fields to {name!r}")
        if name not in exact_names:
            raise ValueError(f"{context} has undeclared logical field {raw_name!r}")
        out[name] = _scalar_for_parquet(raw_value)
    return out


def _merge_identity_checked(
    row: dict[str, Any],
    values: Mapping[str, Any],
    *,
    context: str,
) -> None:
    for name, value in values.items():
        if name in row and row[name] is not None and value is not None and row[name] != value:
            raise ValueError(
                f"{context} conflicts with projected {name!r}: "
                f"{row[name]!r} != {value!r}"
            )
        row[name] = value


def _measurement_family_for_level(level_name: str) -> str:
    if level_name.startswith("heading_"):
        return "heading"
    if level_name == "movement":
        return "movement"
    if level_name == "eye_gaze":
        return "eye_gaze"
    return "unknown"


def _load_bout_kinematics_metrics(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    stimulus_run: str | None,
    protocol_signature: ProtocolSignature | None,
    steps: Sequence[StepSpan],
    tables: set[str],
    diagnostics: list[dict[str, Any]],
    legacy_compatibility: bool = False,
) -> list[dict[str, Any]]:
    if "bout_kinematics_metrics" not in tables:
        return []

    bout_kin_group, bout_kin_run, error = _latest_run(root, "analysis/bout_kinematics_runs")
    if bout_kin_group is None or bout_kin_run is None:
        diagnostics.append({"table": "bout_kinematics_metrics", "status": "skipped", "reason": error})
        return []

    run_attrs = _attrs_dict(bout_kin_group)
    source_refs = _parse_jsonish(run_attrs.get("source_refs"))
    if not isinstance(source_refs, Mapping):
        source_refs = {}
    source_swim_bout_run = run_attrs.get("source_swim_bout_run") or source_refs.get(
        "source_swim_bout_run"
    )
    source_swim_bout_speed_level = run_attrs.get("source_swim_bout_speed_level") or source_refs.get(
        "source_swim_bout_speed_level"
    )
    source_track_kinematics_run = run_attrs.get("source_track_kinematics_run") or source_refs.get(
        "source_track_kinematics_run"
    )
    source_track_id = run_attrs.get("source_track_id")
    if source_track_id is None:
        source_track_id = source_refs.get("source_track_id")
    track_id = _safe_int(source_track_id)
    default_heading_level = run_attrs.get("default_heading_level")
    records_by_level, level_attrs_by_level, metrics_attrs_by_level = resolve_bout_kinematics_tables(
        bout_kin_group,
        legacy_compatibility=legacy_compatibility,
    )

    if not records_by_level:
        diagnostics.append({
            "table": "bout_kinematics_metrics",
            "status": "skipped",
            "reason": "no bout-kinematics measurement tables",
            "bout_kinematics_run": bout_kin_run,
        })
        return []

    rows: list[dict[str, Any]] = []
    for level_name in sorted(records_by_level):
        metrics_attrs = metrics_attrs_by_level.get(level_name, {})
        metric_rows = structured_records_to_dicts(records_by_level[level_name])
        if not metric_rows:
            diagnostics.append({
                "table": "bout_kinematics_metrics",
                "status": "partial",
                "reason": "empty bout-kinematics measurement table",
                "bout_kinematics_run": bout_kin_run,
                "measurement_level": level_name,
            })
            continue

        level_attrs = level_attrs_by_level.get(level_name, {})
        measurement_family = _measurement_family_for_level(level_name)
        if measurement_family == "unknown":
            raise ValueError(
                f"bout-kinematics level {level_name!r} has no declared measurement family"
            )
        for raw_metric in metric_rows:
            metric = _closed_metric_record(
                raw_metric,
                table=BOUT_KINEMATICS_METRICS_TABLE,
                context=f"bout-kinematics {level_name!r} payload",
                semantic_fixed_text_names=True,
            )
            bout_id = _safe_int(metric.get("bout_id"))
            start_frame = _safe_int(metric.get("source_start_frame"))
            end_frame = _safe_int(metric.get("source_end_frame"))
            if start_frame is None:
                start_frame = _safe_int(metric.get("source_core_start_frame"))
            if end_frame is None:
                end_frame = _safe_int(metric.get("source_core_end_frame"))
            step = _assign_step(start_frame, end_frame, steps)
            lineage = {
                "zarr_path": str(zarr_path),
                "bout_kinematics_run": bout_kin_run,
                "measurement_level": level_name,
                "source_swim_bout_run": source_swim_bout_run,
                "source_swim_bout_speed_level": source_swim_bout_speed_level,
                "source_track_kinematics_run": source_track_kinematics_run,
                "track_id": track_id,
                "bout_id": bout_id,
            }
            row = _common_row(
                export_run_id=export_run_id,
                zarr_path=zarr_path,
                recording_id=recording_id,
                table="bout_kinematics_metrics",
                lineage=lineage,
            )
            row.update({
                "stimulus_run": stimulus_run,
                "bout_kinematics_run": bout_kin_run,
                "measurement_level": level_name,
                "measurement_family": measurement_family,
                "is_default_heading_level": (
                    bool(level_attrs.get("is_default_heading_level"))
                    if level_attrs.get("is_default_heading_level") is not None
                    else level_name == default_heading_level
                ),
                "source_swim_bout_run": source_swim_bout_run,
                "source_swim_bout_speed_level": source_swim_bout_speed_level,
                "source_track_kinematics_run": source_track_kinematics_run,
                "track_id": track_id,
                "schema_version": _safe_int(run_attrs.get("schema_version") or metrics_attrs.get("schema_version")),
                "method": run_attrs.get("method"),
                "method_version": run_attrs.get("method_version"),
                "step_index": step.step_index if step else None,
                "step_name": step.step_name if step else None,
                "stimulus_mode": step.stimulus_mode if step else None,
            })
            row.update(_protocol_signature_row(protocol_signature))
            _merge_identity_checked(
                row,
                metric,
                context=f"bout-kinematics {level_name!r} payload",
            )
            rows.append(row)
    return rows


def _load_position_occupancy_histogram_2d(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    table = POSITION_OCCUPANCY_HISTOGRAM_TABLE
    if table not in tables:
        return []

    run_group, run_name, error = _latest_run(
        root,
        "analysis/detection_occupancy_runs",
    )
    if run_group is None or run_name is None:
        diagnostics.append({"table": table, "status": "skipped", "reason": error})
        return []
    if not _has_child(run_group, "heatmaps"):
        diagnostics.append(
            {
                "table": table,
                "status": "skipped",
                "reason": "detection occupancy run has no heatmaps group",
                "position_occupancy_run": run_name,
            }
        )
        return []

    heatmaps = run_group["heatmaps"]
    counts = _read_array(heatmaps, "counts")
    x_edges = _read_1d_array(heatmaps, "x_edges")
    y_edges = _read_1d_array(heatmaps, "y_edges")
    if counts is None or np.asarray(counts).ndim != 3:
        diagnostics.append(
            {
                "table": table,
                "status": "skipped",
                "reason": "heatmaps/counts is missing or not 3D",
                "position_occupancy_run": run_name,
            }
        )
        return []
    counts = np.asarray(counts)
    if x_edges is None or y_edges is None:
        diagnostics.append(
            {
                "table": table,
                "status": "skipped",
                "reason": "heatmap x_edges or y_edges is missing",
                "position_occupancy_run": run_name,
            }
        )
        return []
    x_edges = np.asarray(x_edges, dtype=np.float64).reshape(-1)
    y_edges = np.asarray(y_edges, dtype=np.float64).reshape(-1)
    n_windows, n_y_bins, n_x_bins = map(int, counts.shape)
    if x_edges.size != n_x_bins + 1 or y_edges.size != n_y_bins + 1:
        diagnostics.append(
            {
                "table": table,
                "status": "skipped",
                "reason": "heatmap edge lengths disagree with counts shape",
                "position_occupancy_run": run_name,
                "counts_shape": list(counts.shape),
                "x_edge_count": int(x_edges.size),
                "y_edge_count": int(y_edges.size),
            }
        )
        return []

    run_attrs = _attrs_dict(run_group)
    heatmap_attrs = _attrs_dict(heatmaps)
    windows = _window_rows_from_group(run_group)
    coverage = run_group["coverage"] if _has_child(run_group, "coverage") else None
    detection_count = _read_1d_array(coverage, "detection_count") if coverage is not None else None
    covered_frame_count = (
        _read_1d_array(coverage, "covered_frame_count") if coverage is not None else None
    )
    total_span_frames = (
        _read_1d_array(coverage, "total_span_frames") if coverage is not None else None
    )
    coverage_pct = _read_1d_array(coverage, "coverage_pct") if coverage is not None else None
    image_width = _safe_float(run_attrs.get("width"))
    image_height = _safe_float(run_attrs.get("height"))
    if image_width is None or image_width <= 0.0:
        image_width = float(x_edges[-1])
    if image_height is None or image_height <= 0.0:
        image_height = float(y_edges[-1])
    if image_width <= 0.0 or image_height <= 0.0:
        diagnostics.append(
            {
                "table": table,
                "status": "skipped",
                "reason": "source image width or height is not positive",
                "position_occupancy_run": run_name,
            }
        )
        return []

    x_fraction_edges = x_edges / float(image_width)
    y_fraction_edges = y_edges / float(image_height)
    normalized_grid_id = _hash_payload(
        {
            "coordinate_frame": "source_image_fraction",
            "x_edges": [round(float(value), 12) for value in x_fraction_edges],
            "y_edges": [round(float(value), 12) for value in y_fraction_edges],
        }
    )[:16]
    source_refs = _json_mapping_attr(run_attrs, "source_refs")
    run_path = f"analysis/detection_occupancy_runs/{run_name}"
    rows: list[dict[str, Any]] = []
    for window_index in range(n_windows):
        window = (
            windows[window_index]
            if window_index < len(windows)
            else {
                "window_index": window_index,
                "window_id": window_index,
                "window_label": f"window_{window_index}",
                "start_frame": None,
                "end_frame": None,
                "start_time_s": None,
                "end_time_s": None,
                "duration_s": None,
            }
        )
        window_has_counts = bool(np.any(counts[window_index] > 0))
        for y_bin_index in range(n_y_bins):
            for x_bin_index in range(n_x_bins):
                hist_count = int(counts[window_index, y_bin_index, x_bin_index])
                if hist_count <= 0 and (
                    window_has_counts or y_bin_index != 0 or x_bin_index != 0
                ):
                    continue
                lineage = {
                    "zarr_path": str(zarr_path),
                    "position_occupancy_run": run_name,
                    "window_id": window.get("window_id"),
                    "y_bin_index": y_bin_index,
                    "x_bin_index": x_bin_index,
                }
                row = _common_row(
                    export_run_id=export_run_id,
                    zarr_path=zarr_path,
                    recording_id=recording_id,
                    table=table,
                    lineage=lineage,
                )
                x_left_px = float(x_edges[x_bin_index])
                x_right_px = float(x_edges[x_bin_index + 1])
                y_left_px = float(y_edges[y_bin_index])
                y_right_px = float(y_edges[y_bin_index + 1])
                row.update(
                    {
                        "position_occupancy_run": run_name,
                        "position_occupancy_path": run_path,
                        "position_occupancy_schema_id": run_attrs.get("schema_id"),
                        "position_occupancy_schema_version": _safe_int(
                            run_attrs.get("schema_version")
                        ),
                        "source_detection_path": run_attrs.get("source_detection_path"),
                        "source_detection_kind": run_attrs.get("source_detection_kind"),
                        "source_segment_kind": run_attrs.get("source_segment_kind"),
                        "source_segment_path": run_attrs.get("source_segment_path"),
                        "source_refs_json": _source_refs_json(source_refs),
                        "window_index": window_index,
                        "window_id": _safe_int(window.get("window_id")),
                        "window_label": window.get("window_label"),
                        "start_frame": _safe_int(window.get("start_frame")),
                        "end_frame": _safe_int(window.get("end_frame")),
                        "start_time_s": _safe_float(window.get("start_time_s")),
                        "end_time_s": _safe_float(window.get("end_time_s")),
                        "duration_s": _safe_float(window.get("duration_s")),
                        "coordinate_frame": "source_image_fraction",
                        "source_coordinate_frame": run_attrs.get("coordinate_space")
                        or "source_image_pixels",
                        "coordinate_origin": "top_left",
                        "x_axis_direction": "right",
                        "y_axis_direction": "down",
                        "image_width_px": float(image_width),
                        "image_height_px": float(image_height),
                        "normalized_grid_id": normalized_grid_id,
                        "normalized_grid_uniform": True,
                        "sparse_zero_bins_omitted": True,
                        "x_bin_count": n_x_bins,
                        "y_bin_count": n_y_bins,
                        "x_bin_index": x_bin_index,
                        "x_bin_left_px": x_left_px,
                        "x_bin_right_px": x_right_px,
                        "x_bin_center_px": (x_left_px + x_right_px) / 2.0,
                        "x_bin_width_px": x_right_px - x_left_px,
                        "x_bin_left_fraction": float(x_fraction_edges[x_bin_index]),
                        "x_bin_right_fraction": float(x_fraction_edges[x_bin_index + 1]),
                        "x_bin_center_fraction": float(
                            (x_fraction_edges[x_bin_index] + x_fraction_edges[x_bin_index + 1])
                            / 2.0
                        ),
                        "x_bin_width_fraction": float(
                            x_fraction_edges[x_bin_index + 1]
                            - x_fraction_edges[x_bin_index]
                        ),
                        "y_bin_index": y_bin_index,
                        "y_bin_left_px": y_left_px,
                        "y_bin_right_px": y_right_px,
                        "y_bin_center_px": (y_left_px + y_right_px) / 2.0,
                        "y_bin_width_px": y_right_px - y_left_px,
                        "y_bin_left_fraction": float(y_fraction_edges[y_bin_index]),
                        "y_bin_right_fraction": float(y_fraction_edges[y_bin_index + 1]),
                        "y_bin_center_fraction": float(
                            (y_fraction_edges[y_bin_index] + y_fraction_edges[y_bin_index + 1])
                            / 2.0
                        ),
                        "y_bin_width_fraction": float(
                            y_fraction_edges[y_bin_index + 1]
                            - y_fraction_edges[y_bin_index]
                        ),
                        "hist_count": hist_count,
                        "window_detection_count": _array_int(
                            detection_count, window_index
                        ),
                        "covered_frame_count": _array_int(
                            covered_frame_count, window_index
                        ),
                        "total_span_frames": _array_int(
                            total_span_frames, window_index
                        ),
                        "coverage_pct": _array_float(coverage_pct, window_index),
                        "axis_order": heatmap_attrs.get("axis_order")
                        or ["window", "y_bin", "x_bin"],
                        "source_bin_size_px": _safe_float(
                            heatmap_attrs.get("bin_size_px")
                        ),
                    }
                )
                rows.append(row)
    return rows


def _load_goodcopbadcop_spatial_occupancy_zones(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    table = "goodcopbadcop_spatial_occupancy_zones"
    if table not in tables:
        return []

    run_group, run_name, error = _latest_run(root, "analysis/detection_occupancy_runs")
    if run_group is None or run_name is None:
        diagnostics.append({"table": table, "status": "skipped", "reason": error})
        return []
    if not _has_child(run_group, "spatial_occupancy"):
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "missing spatial_occupancy group",
            "detection_occupancy_run": run_name,
        })
        return []

    run_attrs = _attrs_dict(run_group)
    source_refs = _json_mapping_attr(run_attrs, "source_refs")
    windows = _window_rows_from_group(run_group)
    spatial_parent = run_group["spatial_occupancy"]
    run_path = f"analysis/detection_occupancy_runs/{run_name}"
    rows: list[dict[str, Any]] = []

    for zone_set_name in _group_names(spatial_parent):
        zone_group = spatial_parent[zone_set_name]
        if not _has_child(zone_group, "zone_spec") or not _has_child(zone_group, "summary"):
            diagnostics.append({
                "table": table,
                "status": "partial",
                "reason": "zone set missing zone_spec or summary group",
                "detection_occupancy_run": run_name,
                "zone_set_id": zone_set_name,
            })
            continue

        zone_attrs = _attrs_dict(zone_group)
        zone_spec = zone_group["zone_spec"]
        summary = zone_group["summary"]
        frame_count = _read_array(summary, "frame_count")
        if frame_count is None or np.asarray(frame_count).ndim != 2:
            diagnostics.append({
                "table": table,
                "status": "partial",
                "reason": "zone set summary/frame_count missing or not 2D",
                "detection_occupancy_run": run_name,
                "zone_set_id": zone_set_name,
            })
            continue
        frame_count = np.asarray(frame_count)
        n_windows, n_zones = int(frame_count.shape[0]), int(frame_count.shape[1])
        zone_ids = _decode_text_column(_read_array(zone_spec, "zone_id"), fallback_count=n_zones)
        zone_labels = _decode_text_column(_read_array(zone_spec, "label_bytes"), fallback_count=n_zones)
        geometry_types = _decode_text_column(_read_array(zone_spec, "geometry_type"), fallback_count=n_zones)
        display_order = _read_1d_array(zone_spec, "display_order")
        bounds_xyxy = _read_array(zone_spec, "bounds_xyxy")
        time_s = _read_array(summary, "time_s")
        fraction_of_epoch = _read_array(summary, "fraction_of_epoch")
        fraction_of_detected = _read_array(summary, "fraction_of_detected")
        detected_frame_count = _read_1d_array(summary, "detected_frame_count")
        missing_frame_count = _read_1d_array(summary, "missing_frame_count")
        total_span_frames = _read_1d_array(summary, "total_span_frames")
        coverage_pct = _read_1d_array(summary, "coverage_pct")
        zone_set_id = str(zone_attrs.get("zone_set_id") or zone_set_name)

        for window_index in range(n_windows):
            window = (
                windows[window_index]
                if window_index < len(windows)
                else {
                    "window_index": window_index,
                    "window_id": window_index,
                    "window_label": None,
                    "start_frame": None,
                    "end_frame": None,
                    "start_time_s": None,
                    "end_time_s": None,
                    "duration_s": None,
                }
            )
            for zone_index in range(n_zones):
                zone_id = (
                    str(zone_ids[zone_index])
                    if zone_index < len(zone_ids) and zone_ids[zone_index]
                    else str(zone_index)
                )
                lineage = {
                    "zarr_path": str(zarr_path),
                    "detection_occupancy_run": run_name,
                    "source_detection_path": run_attrs.get("source_detection_path"),
                    "source_stimulus_epoch_run": run_attrs.get("source_stimulus_epoch_run"),
                    "zone_set_id": zone_set_id,
                    "window_id": window.get("window_id"),
                    "zone_id": zone_id,
                }
                row = _common_row(
                    export_run_id=export_run_id,
                    zarr_path=zarr_path,
                    recording_id=recording_id,
                    table=table,
                    lineage=lineage,
                )
                row.update({
                    "detection_occupancy_run": run_name,
                    "detection_occupancy_path": run_path,
                    "detection_occupancy_schema_id": run_attrs.get("schema_id"),
                    "detection_occupancy_schema_version": _safe_int(run_attrs.get("schema_version")),
                    "detection_occupancy_method": run_attrs.get("method"),
                    "detection_occupancy_method_version": run_attrs.get("method_version"),
                    "source_detection_path": run_attrs.get("source_detection_path") or zone_attrs.get("source_detection_path"),
                    "source_detection_kind": run_attrs.get("source_detection_kind"),
                    "source_stimulus_epoch_run": run_attrs.get("source_stimulus_epoch_run"),
                    "source_stimulus_epoch_path": run_attrs.get("source_stimulus_epoch_path") or zone_attrs.get("source_stimulus_epoch_path"),
                    "source_refs_json": _source_refs_json(source_refs),
                    "zone_schema_id": zone_attrs.get("schema_id"),
                    "zone_schema_version": _safe_int(zone_attrs.get("schema_version")),
                    "zone_set_id": zone_set_id,
                    "zone_set_source": zone_attrs.get("zone_set_source"),
                    "zone_set_source_ref": zone_attrs.get("zone_set_source_ref"),
                    "coordinate_frame": zone_attrs.get("coordinate_frame"),
                    "coordinate_origin": zone_attrs.get("coordinate_origin"),
                    "x_axis_direction": zone_attrs.get("x_axis_direction"),
                    "y_axis_direction": zone_attrs.get("y_axis_direction"),
                    "width_px": _safe_int(zone_attrs.get("width") or run_attrs.get("width")),
                    "height_px": _safe_int(zone_attrs.get("height") or run_attrs.get("height")),
                    "fps": _safe_float(zone_attrs.get("fps") or run_attrs.get("fps")),
                    "detection_selection_policy": zone_attrs.get("detection_selection_policy"),
                    "zone_overlap_policy": zone_attrs.get("zone_overlap_policy"),
                    "time_basis": zone_attrs.get("time_basis"),
                    "window_index": _safe_int(window.get("window_index")),
                    "window_id": _safe_int(window.get("window_id")),
                    "window_label": window.get("window_label"),
                    "start_frame": _safe_int(window.get("start_frame")),
                    "end_frame": _safe_int(window.get("end_frame")),
                    "start_time_s": _safe_float(window.get("start_time_s")),
                    "end_time_s": _safe_float(window.get("end_time_s")),
                    "duration_s": _safe_float(window.get("duration_s")),
                    "zone_index": zone_index,
                    "zone_id": zone_id,
                    "zone_label": (
                        str(zone_labels[zone_index])
                        if zone_index < len(zone_labels) and zone_labels[zone_index]
                        else zone_id
                    ),
                    "display_order": _array_int(display_order, zone_index),
                    "geometry_type": (
                        str(geometry_types[zone_index])
                        if zone_index < len(geometry_types) and geometry_types[zone_index]
                        else None
                    ),
                    "x_min": _array_float(bounds_xyxy, zone_index, 0),
                    "y_min": _array_float(bounds_xyxy, zone_index, 1),
                    "x_max": _array_float(bounds_xyxy, zone_index, 2),
                    "y_max": _array_float(bounds_xyxy, zone_index, 3),
                    "frame_count": _array_int(frame_count, window_index, zone_index),
                    "time_s": _array_float(time_s, window_index, zone_index),
                    "fraction_of_epoch": _array_float(fraction_of_epoch, window_index, zone_index),
                    "fraction_of_detected": _array_float(fraction_of_detected, window_index, zone_index),
                    "detected_frame_count": _array_int(detected_frame_count, window_index),
                    "missing_frame_count": _array_int(missing_frame_count, window_index),
                    "total_span_frames": _array_int(total_span_frames, window_index),
                    "coverage_pct": _array_float(coverage_pct, window_index),
                })
                rows.append(row)
    return rows


def _chaser_indices_for_run(run_group: Any, *, fallback_count: int) -> list[int]:
    if _has_child(run_group, "chasers"):
        arr = _read_1d_array(run_group["chasers"], "chaser_index")
        if arr is not None and arr.size:
            return [_safe_int(value) if _safe_int(value) is not None else int(idx) for idx, value in enumerate(arr)]
    if _has_child(run_group, "epoch_distributions"):
        arr = _read_1d_array(run_group["epoch_distributions"], "chaser_index")
        if arr is not None and arr.size:
            return [_safe_int(value) if _safe_int(value) is not None else int(idx) for idx, value in enumerate(arr)]
    return list(range(int(fallback_count)))


def _chaser_behaviors_for_run(
    run_group: Any,
    chaser_indices: Sequence[int],
) -> list[tuple[int, str]]:
    by_index: dict[int, tuple[int, str]] = {}
    chasers = run_group.get("chasers") if _has_child(run_group, "chasers") else None
    if chasers is not None:
        indices = _read_1d_array(chasers, "chaser_index")
        class_ids = _read_1d_array(chasers, "behavior_class_id")
        labels = _decode_text_column(_read_array(chasers, "behavior_class_label_bytes"))
        for column, raw_index in enumerate(indices if indices is not None else []):
            index = _safe_int(raw_index)
            if index is None:
                continue
            class_id = _array_int(class_ids, column) or 0
            label = canonical_behavior_label(labels[column]) if column < len(labels) and labels[column] else "unknown"
            by_index[index] = (class_id, label)
    return [by_index.get(int(index), (0, "unknown")) for index in chaser_indices]


def _chaser_common_run_fields(run_group: Any, run_name: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    attrs = _attrs_dict(run_group)
    source_refs = _json_mapping_attr(attrs, "source_refs")
    parameters = _json_mapping_attr(attrs, "parameters")
    common = {
        "chaser_distance_run": run_name,
        "chaser_distance_path": f"analysis/chaser_distance_runs/{run_name}",
        "chaser_distance_schema_id": attrs.get("schema_id"),
        "chaser_distance_schema_version": _safe_int(attrs.get("schema_version")),
        "chaser_distance_method": attrs.get("method"),
        "chaser_distance_method_version": attrs.get("method_version"),
        "source_detection_path": attrs.get("source_detection_path") or source_refs.get("source_detection_path"),
        "source_detection_kind": attrs.get("source_detection_kind") or source_refs.get("source_detection_kind"),
        "source_stimulus_run": attrs.get("source_stimulus_run") or source_refs.get("source_stimulus_run"),
        "source_stimulus_path": attrs.get("source_stimulus_path") or source_refs.get("source_stimulus_path"),
        "source_stimulus_epoch_run": attrs.get("source_stimulus_epoch_run") or source_refs.get("source_stimulus_epoch_run"),
        "source_stimulus_epoch_path": attrs.get("source_stimulus_epoch_path") or source_refs.get("source_stimulus_epoch_path"),
        "source_refs_json": _source_refs_json(source_refs),
        "coordinate_frame": attrs.get("coordinate_frame"),
        "coordinate_origin": attrs.get("coordinate_origin"),
        "fps": _safe_float(attrs.get("fps")),
        "total_frames": _safe_int(attrs.get("total_frames")),
        "pixels_per_mm_projector": _safe_float(attrs.get("pixels_per_mm_projector")),
    }
    return common, source_refs, parameters


def _fill_chaser_window_times(window: dict[str, Any], *, fps: float | None) -> dict[str, Any]:
    out = dict(window)
    if fps is None or fps <= 0:
        return out
    start_frame = _safe_int(out.get("start_frame"))
    end_frame = _safe_int(out.get("end_frame"))
    if start_frame is not None and out.get("start_time_s") is None:
        out["start_time_s"] = float(start_frame) / fps
    if end_frame is not None and out.get("end_time_s") is None:
        out["end_time_s"] = float(end_frame + 1) / fps
    if out.get("duration_s") is None and start_frame is not None and end_frame is not None:
        out["duration_s"] = float(max(0, end_frame - start_frame + 1)) / fps
    return out


def _load_goodcopbadcop_chaser_epoch_summary(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    table = "goodcopbadcop_chaser_epoch_summary"
    if table not in tables:
        return []

    run_group, run_name, error = _latest_run(root, "analysis/chaser_distance_runs")
    if run_group is None or run_name is None:
        diagnostics.append({"table": table, "status": "skipped", "reason": error})
        return []
    if not _has_child(run_group, "epoch_summary"):
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "missing epoch_summary group",
            "chaser_distance_run": run_name,
        })
        return []

    run_common, source_refs, parameters = _chaser_common_run_fields(run_group, run_name)
    summary = run_group["epoch_summary"]
    valid_frame_count = _read_array(summary, "valid_frame_count")
    if valid_frame_count is None or np.asarray(valid_frame_count).ndim != 2:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "epoch_summary/valid_frame_count missing or not 2D",
            "chaser_distance_run": run_name,
        })
        return []

    valid_frame_count = np.asarray(valid_frame_count)
    n_windows, n_chasers = int(valid_frame_count.shape[0]), int(valid_frame_count.shape[1])
    chaser_indices = _chaser_indices_for_run(run_group, fallback_count=n_chasers)
    chaser_behaviors = _chaser_behaviors_for_run(run_group, chaser_indices)
    windows = _epoch_summary_window_rows(run_group)
    fps = _safe_float(run_common.get("fps"))
    mean_distance = _read_array(summary, "mean_distance_mm")
    min_distance = _read_array(summary, "min_distance_mm")
    p05_distance = _read_array(summary, "p05_distance_mm")
    p50_distance = _read_array(summary, "p50_distance_mm")
    p95_distance = _read_array(summary, "p95_distance_mm")
    fraction_within_threshold = _read_array(summary, "fraction_within_threshold")
    summary_attrs = _attrs_dict(summary)
    threshold_mm = (
        _safe_float(summary_attrs.get("threshold_mm"))
        or _safe_float(parameters.get("threshold_mm"))
    )

    rows: list[dict[str, Any]] = []
    for window_index in range(n_windows):
        raw_window = (
            windows[window_index]
            if window_index < len(windows)
            else {
                "window_index": window_index,
                "window_id": window_index,
                "window_label": None,
                "start_frame": None,
                "end_frame": None,
                "start_time_s": None,
                "end_time_s": None,
                "duration_s": None,
            }
        )
        window = _fill_chaser_window_times(raw_window, fps=fps)
        for chaser_column_index in range(n_chasers):
            chaser_index = (
                chaser_indices[chaser_column_index]
                if chaser_column_index < len(chaser_indices)
                else chaser_column_index
            )
            behavior_class_id, behavior_class = chaser_behaviors[chaser_column_index]
            lineage = {
                "zarr_path": str(zarr_path),
                "chaser_distance_run": run_name,
                "source_detection_path": source_refs.get("source_detection_path"),
                "source_stimulus_run": source_refs.get("source_stimulus_run"),
                "source_stimulus_epoch_run": source_refs.get("source_stimulus_epoch_run"),
                "window_id": window.get("window_id"),
                "chaser_index": chaser_index,
            }
            row = _common_row(
                export_run_id=export_run_id,
                zarr_path=zarr_path,
                recording_id=recording_id,
                table=table,
                lineage=lineage,
            )
            row.update(run_common)
            row.update({
                "window_index": _safe_int(window.get("window_index")),
                "window_id": _safe_int(window.get("window_id")),
                "window_label": window.get("window_label"),
                "start_frame": _safe_int(window.get("start_frame")),
                "end_frame": _safe_int(window.get("end_frame")),
                "start_time_s": _safe_float(window.get("start_time_s")),
                "end_time_s": _safe_float(window.get("end_time_s")),
                "duration_s": _safe_float(window.get("duration_s")),
                "chaser_column_index": chaser_column_index,
                "chaser_index": _safe_int(chaser_index),
                "behavior_class_id": behavior_class_id,
                "behavior_class": behavior_class,
                "threshold_mm": threshold_mm,
                "valid_frame_count": _array_int(valid_frame_count, window_index, chaser_column_index),
                "mean_distance_mm": _array_float(mean_distance, window_index, chaser_column_index),
                "min_distance_mm": _array_float(min_distance, window_index, chaser_column_index),
                "p05_distance_mm": _array_float(p05_distance, window_index, chaser_column_index),
                "p50_distance_mm": _array_float(p50_distance, window_index, chaser_column_index),
                "p95_distance_mm": _array_float(p95_distance, window_index, chaser_column_index),
                "fraction_within_threshold": _array_float(
                    fraction_within_threshold,
                    window_index,
                    chaser_column_index,
                ),
            })
            rows.append(row)
    return rows


def _latest_epoch_behavior_component(
    run_group: Any,
    *,
    run_name: str,
) -> tuple[Any | None, str | None, str | None]:
    if not _has_child(run_group, EPOCH_BEHAVIOR_COMPONENT_PARENT):
        return None, None, "missing epoch_behavior_summary component parent"
    parent = run_group[EPOCH_BEHAVIOR_COMPONENT_PARENT]
    keys = set(_group_names(parent))
    for attr_name in ("latest_complete", "latest"):
        candidate = str(_attrs_dict(parent).get(attr_name) or "").strip()
        if candidate and candidate in keys:
            component = parent[candidate]
            attrs = _attrs_dict(component)
            if str(attrs.get("schema_id") or "") in {"", EPOCH_BEHAVIOR_SCHEMA_ID}:
                return (
                    component,
                    candidate,
                    f"analysis/chaser_distance_runs/{run_name}/{EPOCH_BEHAVIOR_COMPONENT_PARENT}/{candidate}",
                )

    complete_candidates: list[str] = []
    for name in sorted(keys):
        try:
            component = parent[name]
        except Exception:
            continue
        attrs = _attrs_dict(component)
        if str(attrs.get("status") or "").strip() == "complete" and str(attrs.get("schema_id") or "") in {
            "",
            EPOCH_BEHAVIOR_SCHEMA_ID,
        }:
            complete_candidates.append(name)
    if not complete_candidates:
        return None, None, "no complete epoch_behavior_summary component"
    selected = complete_candidates[-1]
    return (
        parent[selected],
        selected,
        f"analysis/chaser_distance_runs/{run_name}/{EPOCH_BEHAVIOR_COMPONENT_PARENT}/{selected}",
    )


def _group_at_path(root: Any, path: str | None) -> Any | None:
    normalized = "/".join(part for part in str(path or "").strip("/").split("/") if part)
    if not normalized:
        return root
    current = root
    for part in normalized.split("/"):
        if not _has_child(current, part):
            return None
        current = current[part]
    return current


def _baseline_arena_geometry(root: Any, chaser_run: Any) -> dict[str, float] | None:
    """Resolve the arena-relative coordinate geometry used by chaser positions."""

    candidates: list[Any] = []
    source_stimulus_path = str(_attrs_dict(chaser_run).get("source_stimulus_path") or "").strip()
    if source_stimulus_path:
        source_geometry = _group_at_path(
            root,
            f"{source_stimulus_path}/calibration/arena_geometry",
        )
        if source_geometry is not None:
            candidates.append(source_geometry)
    analysis_calibration = _group_at_path(root, "analysis/calibration")
    if analysis_calibration is not None:
        candidates.append(analysis_calibration)

    pixels_per_mm = _safe_float(_attrs_dict(chaser_run).get("pixels_per_mm_projector"))
    if pixels_per_mm is None or pixels_per_mm <= 0:
        return None
    for group in candidates:
        attrs = _attrs_dict(group)
        shape = str(
            attrs.get("experimental_area_shape") or attrs.get("arena_shape") or ""
        ).strip().lower()
        center_x = _safe_float(attrs.get("experimental_area_center_x_px"))
        center_y = _safe_float(attrs.get("experimental_area_center_y_px"))
        radius_px = _safe_float(attrs.get("experimental_area_radius_px"))
        if radius_px is None:
            radius_mm = _safe_float(attrs.get("experimental_area_radius_mm"))
            if radius_mm is not None:
                radius_px = radius_mm * pixels_per_mm
        if (
            shape == "circle"
            and center_x is not None
            and center_y is not None
            and radius_px is not None
            and radius_px > 0
        ):
            return {
                "center_x_px": center_x,
                "center_y_px": center_y,
                "radius_px": radius_px,
                "pixels_per_mm": pixels_per_mm,
            }
    return None


def _load_baseline_tables(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
    time_bin_s: float,
    sample_rate_hz: float,
    full_resolution_samples: bool,
    spatial_grid_size: int,
) -> dict[str, list[dict[str, Any]]]:
    requested = {
        BASELINE_BEHAVIOR_SUMMARY_TABLE,
        BASELINE_BEHAVIOR_TIME_BINS_TABLE,
        BASELINE_KINEMATIC_SAMPLES_TABLE,
    } & tables
    output = {table: [] for table in requested}
    if not requested:
        return output

    chaser_run, chaser_run_name, error = _latest_run(root, "analysis/chaser_distance_runs")
    if chaser_run is None or chaser_run_name is None:
        for table in sorted(requested):
            diagnostics.append({"table": table, "status": "skipped", "reason": error})
        return output
    component, component_name, component_path_or_error = _latest_epoch_behavior_component(
        chaser_run,
        run_name=chaser_run_name,
    )
    if component is None or component_name is None:
        for table in sorted(requested):
            diagnostics.append(
                {
                    "table": table,
                    "status": "skipped",
                    "reason": component_path_or_error,
                }
            )
        return output
    component_path = str(component_path_or_error)
    try:
        summary_records, _summary_attrs = load_structured_dataset(component, "per_epoch_fish")
    except Exception as exc:
        for table in sorted(requested):
            diagnostics.append(
                {
                    "table": table,
                    "status": "skipped",
                    "reason": f"failed to load per_epoch_fish: {exc}",
                }
            )
        return output
    summary_rows = [
        {name: _scalar_for_parquet(record[name]) for name in summary_records.dtype.names or ()}
        for record in summary_records
    ]
    baseline_rows = [row for row in summary_rows if is_baseline_label(row.get("window_label"))]
    if not baseline_rows:
        for table in sorted(requested):
            diagnostics.append(
                {
                    "table": table,
                    "status": "skipped",
                    "reason": "no canonical baseline window in per_epoch_fish",
                }
            )
        return output

    component_attrs = _attrs_dict(component)
    component_source_refs = _json_mapping_attr(component_attrs, "source_refs")
    component_parameters = _json_mapping_attr(component_attrs, "parameters")
    track_run_name = str(
        component_source_refs.get("source_track_kinematics_run") or ""
    ).strip()
    track_scope = str(
        component_source_refs.get("source_track_kinematics_scope") or "offline"
    ).strip()
    track_id = _safe_int(component_source_refs.get("source_track_kinematics_track_id"))
    if track_id is None:
        track_id = 0
    speed_level = str(component_parameters.get("speed_level") or "filtered").strip()
    try:
        track = load_track_kinematics_track(
            root,
            run_name=track_run_name or "latest",
            scope=track_scope or "offline",
            track_id=track_id,
            required_speed_levels=(speed_level,),
        )
    except Exception as exc:
        for table in sorted(requested):
            diagnostics.append(
                {
                    "table": table,
                    "status": "skipped",
                    "reason": f"failed to load source track kinematics: {exc}",
                }
            )
        return output

    if not _has_child(chaser_run, "positions"):
        reason = "chaser-distance run has no positions group"
        for table in sorted(requested):
            diagnostics.append({"table": table, "status": "skipped", "reason": reason})
        return output
    positions = chaser_run["positions"]
    arena_xy = _read_array(positions, "fish_centroid_arena_xy")
    position_valid = _read_array(positions, "fish_valid")
    geometry = _baseline_arena_geometry(root, chaser_run)
    if arena_xy is None or position_valid is None or geometry is None:
        reason = "missing arena-relative fish positions, validity, or circular arena geometry"
        for table in sorted(requested):
            diagnostics.append({"table": table, "status": "skipped", "reason": reason})
        return output

    fps = _safe_float(_attrs_dict(chaser_run).get("fps"))
    if fps is None:
        fps = _safe_float(track.run_attrs.get("fps"))
    if fps is None or fps <= 0:
        reason = "missing positive fps"
        for table in sorted(requested):
            diagnostics.append({"table": table, "status": "skipped", "reason": reason})
        return output

    speed_values = track.speed_mm_by_level.get(speed_level)
    if speed_values is None:
        speed_values = track.speed_mm_by_level.get("filtered")
    path_values = track.frame_path_distance_mm_by_level.get(speed_level)
    if path_values is None:
        path_values = track.frame_path_distance_mm_by_level.get("filtered")
    heading_values = (
        track.smoothed_heading_degrees
        if track.smoothed_heading_degrees is not None
        else track.heading_degrees
    )
    bout_event_frames: np.ndarray | None = None
    try:
        bout_records, _bout_attrs = load_structured_dataset(component, "per_epoch_bouts")
        if bout_records.dtype.names:
            for field_name in ("bout_event_frame", "bout_start_frame", "bout_source_row"):
                if field_name in bout_records.dtype.names:
                    bout_event_frames = np.asarray(bout_records[field_name], dtype=np.int64)
                    break
    except Exception:
        bout_event_frames = None

    wall_band_mm = _safe_float(component_parameters.get("wall_band_mm"))
    if wall_band_mm is None:
        wall_band_mm = _safe_float(baseline_rows[0].get("wall_band_mm"))
    if wall_band_mm is None:
        wall_band_mm = 5.0
    arrays = BaselineArrays(
        fps=fps,
        arena_xy_px=np.asarray(arena_xy),
        position_valid=np.asarray(position_valid),
        arena_center_x_px=geometry["center_x_px"],
        arena_center_y_px=geometry["center_y_px"],
        arena_radius_px=geometry["radius_px"],
        pixels_per_mm=geometry["pixels_per_mm"],
        wall_band_mm=wall_band_mm,
        track_frames=np.asarray(track.frame_indices),
        track_time_s=track.time_seconds,
        speed_mm_s=speed_values,
        frame_path_distance_mm=path_values,
        heading_deg=heading_values,
        sample_valid=track.sample_valid,
        bout_event_frames=bout_event_frames,
    )

    run_common, chaser_source_refs, _chaser_parameters = _chaser_common_run_fields(
        chaser_run,
        chaser_run_name,
    )
    source_epoch_run = component_source_refs.get("source_stimulus_epoch_run") or chaser_source_refs.get(
        "source_stimulus_epoch_run"
    )
    source_epoch_path = component_source_refs.get("source_stimulus_epoch_path") or chaser_source_refs.get(
        "source_stimulus_epoch_path"
    )
    common_source = {
        **run_common,
        "source_chaser_distance_run": chaser_run_name,
        "source_chaser_distance_path": f"analysis/chaser_distance_runs/{chaser_run_name}",
        "source_epoch_behavior_component": component_name,
        "source_epoch_behavior_path": component_path,
        "source_stimulus_epoch_run": source_epoch_run,
        "source_stimulus_epoch_path": source_epoch_path,
        "source_track_kinematics_run": track.run_name,
        "source_track_kinematics_scope": track.scope,
        "source_track_kinematics_path": track.run_path,
        "source_track_kinematics_track_path": track.track_path,
        "source_speed_level": speed_level,
        "source_swim_bout_run": component_source_refs.get("source_swim_bout_run"),
        "source_swim_bout_path": component_source_refs.get("source_swim_bout_path"),
        "track_id": track_id,
        "arena_center_x_px": geometry["center_x_px"],
        "arena_center_y_px": geometry["center_y_px"],
        "arena_radius_px": geometry["radius_px"],
        "pixels_per_mm_projector": geometry["pixels_per_mm"],
    }

    for source_summary in baseline_rows:
        start_frame = _safe_int(source_summary.get("start_frame"))
        end_frame = _safe_int(source_summary.get("end_frame"))
        window_id = _safe_int(source_summary.get("window_id"))
        if start_frame is None or end_frame is None or window_id is None:
            continue
        start_time_s = _safe_float(source_summary.get("start_time_s"))
        end_time_s = _safe_float(source_summary.get("end_time_s"))
        duration_s = _safe_float(source_summary.get("duration_s"))
        if start_time_s is None:
            start_time_s = start_frame / fps
        if end_time_s is None:
            end_time_s = (end_frame + 1) / fps
        if duration_s is None:
            duration_s = max(0.0, end_time_s - start_time_s)
        window = BaselineWindow(
            window_id=window_id,
            label=str(source_summary.get("window_label") or "baseline"),
            start_frame=start_frame,
            end_frame=end_frame,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            duration_s=duration_s,
        )
        builders: list[tuple[str, list[dict[str, Any]]]] = []
        if BASELINE_BEHAVIOR_SUMMARY_TABLE in requested:
            builders.append(
                (
                    BASELINE_BEHAVIOR_SUMMARY_TABLE,
                    [
                        build_summary_metrics(
                            arrays,
                            window,
                            spatial_grid_size=spatial_grid_size,
                            source_summary=source_summary,
                        )
                    ],
                )
            )
        if BASELINE_BEHAVIOR_TIME_BINS_TABLE in requested:
            builders.append(
                (
                    BASELINE_BEHAVIOR_TIME_BINS_TABLE,
                    build_time_bin_metrics(arrays, window, time_bin_s=time_bin_s),
                )
            )
        if BASELINE_KINEMATIC_SAMPLES_TABLE in requested:
            builders.append(
                (
                    BASELINE_KINEMATIC_SAMPLES_TABLE,
                    build_sample_metrics(
                        arrays,
                        window,
                        target_sample_rate_hz=sample_rate_hz,
                        full_resolution=full_resolution_samples,
                    ),
                )
            )
        for table, metric_rows in builders:
            for metrics in metric_rows:
                lineage = {
                    "zarr_path": str(zarr_path),
                    "source_chaser_distance_run": chaser_run_name,
                    "source_epoch_behavior_component": component_name,
                    "source_track_kinematics_run": track.run_name,
                    "source_track_kinematics_scope": track.scope,
                    "track_id": track_id,
                    "baseline_window_id": window.window_id,
                    "time_bin_index": metrics.get("time_bin_index"),
                    "source_sample_index": metrics.get("source_sample_index"),
                    "time_bin_s": time_bin_s,
                    "sample_rate_hz": None if full_resolution_samples else sample_rate_hz,
                    "full_resolution_samples": full_resolution_samples,
                    "spatial_grid_size": spatial_grid_size,
                }
                row = _common_row(
                    export_run_id=export_run_id,
                    zarr_path=zarr_path,
                    recording_id=recording_id,
                    table=table,
                    lineage=lineage,
                )
                row.update(common_source)
                row.update(metrics)
                output[table].append(row)
    return output


def _load_goodcopbadcop_epoch_behavior_summary(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    table = "goodcopbadcop_epoch_behavior_summary"
    if table not in tables:
        return []

    run_group, run_name, error = _latest_run(root, "analysis/chaser_distance_runs")
    if run_group is None or run_name is None:
        diagnostics.append({"table": table, "status": "skipped", "reason": error})
        return []

    component, component_name, component_path_or_error = _latest_epoch_behavior_component(
        run_group,
        run_name=run_name,
    )
    if component is None or component_name is None:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": component_path_or_error,
            "chaser_distance_run": run_name,
        })
        return []
    component_path = str(component_path_or_error)
    try:
        records, _records_attrs = load_structured_dataset(component, "per_epoch_fish")
    except Exception as exc:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": f"failed to load per_epoch_fish: {exc}",
            "chaser_distance_run": run_name,
            "epoch_behavior_component": component_name,
        })
        return []
    if records.size == 0 or records.dtype.names is None:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "empty per_epoch_fish table",
            "chaser_distance_run": run_name,
            "epoch_behavior_component": component_name,
        })
        return []

    run_common, source_refs, _run_parameters = _chaser_common_run_fields(run_group, run_name)
    component_attrs = _attrs_dict(component)
    component_source_refs = _json_mapping_attr(component_attrs, "source_refs")
    component_parameters = _json_mapping_attr(component_attrs, "parameters")

    rows: list[dict[str, Any]] = []
    for record in records:
        record_row = {
            name: _scalar_for_parquet(record[name])
            for name in records.dtype.names
        }
        lineage = {
            "zarr_path": str(zarr_path),
            "chaser_distance_run": run_name,
            "epoch_behavior_component": component_name,
            "source_detection_path": source_refs.get("source_detection_path"),
            "source_stimulus_run": source_refs.get("source_stimulus_run"),
            "source_stimulus_epoch_run": source_refs.get("source_stimulus_epoch_run"),
            "window_id": record_row.get("window_id"),
        }
        row = _common_row(
            export_run_id=export_run_id,
            zarr_path=zarr_path,
            recording_id=recording_id,
            table=table,
            lineage=lineage,
        )
        row.update(run_common)
        row.update({
            "epoch_behavior_component": component_name,
            "epoch_behavior_path": component_path,
            "epoch_behavior_schema_id": component_attrs.get("schema_id"),
            "epoch_behavior_schema_version": _safe_int(component_attrs.get("schema_version")),
            "epoch_behavior_method": component_attrs.get("method"),
            "epoch_behavior_method_version": component_attrs.get("method_version"),
            "epoch_behavior_status": component_attrs.get("status"),
            "epoch_behavior_created_at_utc": component_attrs.get("created_at_utc"),
            "epoch_behavior_source_refs_json": _source_refs_json(component_source_refs),
            "epoch_behavior_parameters_json": _json_dumps_safe(component_parameters),
            "source_track_kinematics_run": component_source_refs.get("source_track_kinematics_run"),
            "source_track_kinematics_scope": component_source_refs.get("source_track_kinematics_scope"),
            "source_track_kinematics_track_id": _safe_int(
                component_source_refs.get("source_track_kinematics_track_id")
            ),
            "source_track_kinematics_track_path": component_source_refs.get("source_track_kinematics_track_path"),
            "source_swim_bout_run": component_source_refs.get("source_swim_bout_run"),
            "source_swim_bout_path": component_source_refs.get("source_swim_bout_path"),
            "source_swim_bout_level_path": component_source_refs.get("source_swim_bout_level_path"),
            "source_speed_level": component_parameters.get("speed_level"),
            "swim_bout_signal_level": component_parameters.get("swim_bout_signal_level"),
        })
        row.update(record_row)
        rows.append(row)
    return rows


def _load_goodcopbadcop_epoch_bout_distribution(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    table = "goodcopbadcop_epoch_bout_distribution"
    if table not in tables:
        return []

    run_group, run_name, error = _latest_run(root, "analysis/chaser_distance_runs")
    if run_group is None or run_name is None:
        diagnostics.append({"table": table, "status": "skipped", "reason": error})
        return []

    component, component_name, component_path_or_error = _latest_epoch_behavior_component(
        run_group,
        run_name=run_name,
    )
    if component is None or component_name is None:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": component_path_or_error,
            "chaser_distance_run": run_name,
        })
        return []
    component_path = str(component_path_or_error)
    try:
        records, _records_attrs = load_structured_dataset(component, "per_epoch_bouts")
    except Exception as exc:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": f"failed to load per_epoch_bouts: {exc}",
            "chaser_distance_run": run_name,
            "epoch_behavior_component": component_name,
        })
        return []
    if records.size == 0 or records.dtype.names is None:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "empty per_epoch_bouts table",
            "chaser_distance_run": run_name,
            "epoch_behavior_component": component_name,
        })
        return []

    run_common, source_refs, _run_parameters = _chaser_common_run_fields(run_group, run_name)
    component_attrs = _attrs_dict(component)
    component_source_refs = _json_mapping_attr(component_attrs, "source_refs")
    component_parameters = _json_mapping_attr(component_attrs, "parameters")

    rows: list[dict[str, Any]] = []
    for record in records:
        record_row = {
            name: _scalar_for_parquet(record[name])
            for name in records.dtype.names
        }
        lineage = {
            "zarr_path": str(zarr_path),
            "chaser_distance_run": run_name,
            "epoch_behavior_component": component_name,
            "source_detection_path": source_refs.get("source_detection_path"),
            "source_stimulus_run": source_refs.get("source_stimulus_run"),
            "source_stimulus_epoch_run": source_refs.get("source_stimulus_epoch_run"),
            "window_id": record_row.get("window_id"),
            "bout_source_row": record_row.get("bout_source_row"),
            "bout_id": record_row.get("bout_id"),
        }
        row = _common_row(
            export_run_id=export_run_id,
            zarr_path=zarr_path,
            recording_id=recording_id,
            table=table,
            lineage=lineage,
        )
        row.update(run_common)
        row.update({
            "epoch_behavior_component": component_name,
            "epoch_behavior_path": component_path,
            "epoch_behavior_schema_id": component_attrs.get("schema_id"),
            "epoch_behavior_schema_version": _safe_int(component_attrs.get("schema_version")),
            "epoch_behavior_method": component_attrs.get("method"),
            "epoch_behavior_method_version": component_attrs.get("method_version"),
            "epoch_behavior_status": component_attrs.get("status"),
            "epoch_behavior_created_at_utc": component_attrs.get("created_at_utc"),
            "epoch_behavior_source_refs_json": _source_refs_json(component_source_refs),
            "epoch_behavior_parameters_json": _json_dumps_safe(component_parameters),
            "source_track_kinematics_run": component_source_refs.get("source_track_kinematics_run"),
            "source_track_kinematics_scope": component_source_refs.get("source_track_kinematics_scope"),
            "source_track_kinematics_track_id": _safe_int(
                component_source_refs.get("source_track_kinematics_track_id")
            ),
            "source_track_kinematics_track_path": component_source_refs.get("source_track_kinematics_track_path"),
            "source_swim_bout_run": component_source_refs.get("source_swim_bout_run"),
            "source_swim_bout_path": component_source_refs.get("source_swim_bout_path"),
            "source_swim_bout_level_path": component_source_refs.get("source_swim_bout_level_path"),
            "source_speed_level": component_parameters.get("speed_level"),
            "swim_bout_signal_level": component_parameters.get("swim_bout_signal_level"),
        })
        row.update(record_row)
        rows.append(row)
    return rows


def _load_goodcopbadcop_epoch_center_distance_histogram(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    table = "goodcopbadcop_epoch_center_distance_histogram"
    if table not in tables:
        return []

    run_group, run_name, error = _latest_run(root, "analysis/chaser_distance_runs")
    if run_group is None or run_name is None:
        diagnostics.append({"table": table, "status": "skipped", "reason": error})
        return []

    component, component_name, component_path_or_error = _latest_epoch_behavior_component(
        run_group,
        run_name=run_name,
    )
    if component is None or component_name is None:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": component_path_or_error,
            "chaser_distance_run": run_name,
        })
        return []
    component_path = str(component_path_or_error)
    try:
        records, _records_attrs = load_structured_dataset(component, "center_distance_histogram")
    except Exception as exc:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": f"failed to load center_distance_histogram: {exc}",
            "chaser_distance_run": run_name,
            "epoch_behavior_component": component_name,
        })
        return []
    if records.size == 0 or records.dtype.names is None:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "empty center_distance_histogram table",
            "chaser_distance_run": run_name,
            "epoch_behavior_component": component_name,
        })
        return []

    run_common, source_refs, _run_parameters = _chaser_common_run_fields(run_group, run_name)
    component_attrs = _attrs_dict(component)
    component_source_refs = _json_mapping_attr(component_attrs, "source_refs")
    component_parameters = _json_mapping_attr(component_attrs, "parameters")

    rows: list[dict[str, Any]] = []
    for record in records:
        record_row = {
            name: _scalar_for_parquet(record[name])
            for name in records.dtype.names
        }
        lineage = {
            "zarr_path": str(zarr_path),
            "chaser_distance_run": run_name,
            "epoch_behavior_component": component_name,
            "source_detection_path": source_refs.get("source_detection_path"),
            "source_stimulus_run": source_refs.get("source_stimulus_run"),
            "source_stimulus_epoch_run": source_refs.get("source_stimulus_epoch_run"),
            "window_id": record_row.get("window_id"),
            "bin_index": record_row.get("bin_index"),
        }
        row = _common_row(
            export_run_id=export_run_id,
            zarr_path=zarr_path,
            recording_id=recording_id,
            table=table,
            lineage=lineage,
        )
        row.update(run_common)
        row.update({
            "epoch_behavior_component": component_name,
            "epoch_behavior_path": component_path,
            "epoch_behavior_schema_id": component_attrs.get("schema_id"),
            "epoch_behavior_schema_version": _safe_int(component_attrs.get("schema_version")),
            "epoch_behavior_method": component_attrs.get("method"),
            "epoch_behavior_method_version": component_attrs.get("method_version"),
            "epoch_behavior_status": component_attrs.get("status"),
            "epoch_behavior_created_at_utc": component_attrs.get("created_at_utc"),
            "epoch_behavior_source_refs_json": _source_refs_json(component_source_refs),
            "epoch_behavior_parameters_json": _json_dumps_safe(component_parameters),
            "source_track_kinematics_run": component_source_refs.get("source_track_kinematics_run"),
            "source_swim_bout_run": component_source_refs.get("source_swim_bout_run"),
        })
        row.update(record_row)
        rows.append(row)
    return rows


def _load_goodcopbadcop_epoch_behavior_structured_histogram(
    root: Any,
    *,
    table: str,
    dataset_name: str,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if table not in tables:
        return []

    run_group, run_name, error = _latest_run(root, "analysis/chaser_distance_runs")
    if run_group is None or run_name is None:
        diagnostics.append({"table": table, "status": "skipped", "reason": error})
        return []

    component, component_name, component_path_or_error = _latest_epoch_behavior_component(
        run_group,
        run_name=run_name,
    )
    if component is None or component_name is None:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": component_path_or_error,
            "chaser_distance_run": run_name,
        })
        return []
    component_path = str(component_path_or_error)
    try:
        records, records_attrs = load_structured_dataset(component, dataset_name)
    except Exception as exc:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": f"failed to load {dataset_name}: {exc}",
            "chaser_distance_run": run_name,
            "epoch_behavior_component": component_name,
        })
        return []
    if records.size == 0 or records.dtype.names is None:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": f"empty {dataset_name} table",
            "chaser_distance_run": run_name,
            "epoch_behavior_component": component_name,
        })
        return []

    run_common, source_refs, _run_parameters = _chaser_common_run_fields(run_group, run_name)
    component_attrs = _attrs_dict(component)
    component_source_refs = _json_mapping_attr(component_attrs, "source_refs")
    component_parameters = _json_mapping_attr(component_attrs, "parameters")
    bin_contract = records_attrs.get("bin_contract")

    rows: list[dict[str, Any]] = []
    for record in records:
        record_row = {
            name: _scalar_for_parquet(record[name])
            for name in records.dtype.names
        }
        lineage = {
            "zarr_path": str(zarr_path),
            "chaser_distance_run": run_name,
            "epoch_behavior_component": component_name,
            "source_detection_path": source_refs.get("source_detection_path"),
            "source_stimulus_run": source_refs.get("source_stimulus_run"),
            "source_stimulus_epoch_run": source_refs.get("source_stimulus_epoch_run"),
            "window_id": record_row.get("window_id"),
            "metric_name": record_row.get("metric_name"),
            "bin_index": record_row.get("bin_index"),
        }
        row = _common_row(
            export_run_id=export_run_id,
            zarr_path=zarr_path,
            recording_id=recording_id,
            table=table,
            lineage=lineage,
        )
        row.update(run_common)
        row.update({
            "epoch_behavior_component": component_name,
            "epoch_behavior_path": component_path,
            "epoch_behavior_schema_id": component_attrs.get("schema_id"),
            "epoch_behavior_schema_version": _safe_int(component_attrs.get("schema_version")),
            "epoch_behavior_method": component_attrs.get("method"),
            "epoch_behavior_method_version": component_attrs.get("method_version"),
            "epoch_behavior_status": component_attrs.get("status"),
            "epoch_behavior_created_at_utc": component_attrs.get("created_at_utc"),
            "epoch_behavior_source_refs_json": _source_refs_json(component_source_refs),
            "epoch_behavior_parameters_json": _json_dumps_safe(component_parameters),
            "histogram_dataset": dataset_name,
            "histogram_bin_contract_json": _json_dumps_safe(bin_contract),
            "source_track_kinematics_run": component_source_refs.get("source_track_kinematics_run"),
            "source_track_kinematics_scope": component_source_refs.get("source_track_kinematics_scope"),
            "source_track_kinematics_track_id": _safe_int(
                component_source_refs.get("source_track_kinematics_track_id")
            ),
            "source_track_kinematics_track_path": component_source_refs.get("source_track_kinematics_track_path"),
            "source_swim_bout_run": component_source_refs.get("source_swim_bout_run"),
            "source_swim_bout_path": component_source_refs.get("source_swim_bout_path"),
            "source_swim_bout_level_path": component_source_refs.get("source_swim_bout_level_path"),
            "source_speed_level": component_parameters.get("speed_level"),
            "swim_bout_signal_level": component_parameters.get("swim_bout_signal_level"),
        })
        row.update(record_row)
        rows.append(row)
    return rows


def _epoch_speed_values_mm_s(
    fish_xy: np.ndarray,
    fish_valid: np.ndarray,
    *,
    start_frame: int | None,
    end_frame: int | None,
    fps: float | None,
    pixels_per_mm: float | None,
) -> np.ndarray:
    if fps is None or fps <= 0 or pixels_per_mm is None or pixels_per_mm <= 0:
        return np.asarray([], dtype=np.float64)
    if start_frame is None or end_frame is None:
        return np.asarray([], dtype=np.float64)
    xy = np.asarray(fish_xy, dtype=np.float64)
    valid = np.asarray(fish_valid, dtype=bool).reshape(-1)
    if xy.ndim != 2 or xy.shape[1] != 2 or valid.shape[0] != xy.shape[0]:
        return np.asarray([], dtype=np.float64)
    start = max(0, int(start_frame))
    end = min(int(end_frame), int(xy.shape[0]) - 1)
    if end <= start:
        return np.asarray([], dtype=np.float64)
    segment = xy[start : end + 1]
    segment_valid = valid[start : end + 1] & np.isfinite(segment).all(axis=1)
    pair_valid = segment_valid[:-1] & segment_valid[1:]
    if not np.any(pair_valid):
        return np.asarray([], dtype=np.float64)
    displacement_px = np.linalg.norm(np.diff(segment, axis=0), axis=1)
    speeds = displacement_px[pair_valid] / float(pixels_per_mm) * float(fps)
    return speeds[np.isfinite(speeds)]


def _load_goodcopbadcop_epoch_speed_summary(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    table = "goodcopbadcop_epoch_speed_summary"
    if table not in tables:
        return []

    run_group, run_name, error = _latest_run(root, "analysis/chaser_distance_runs")
    if run_group is None or run_name is None:
        diagnostics.append({"table": table, "status": "skipped", "reason": error})
        return []
    if not _has_child(run_group, "epoch_summary") or not _has_child(run_group, "positions"):
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "missing epoch_summary or positions group",
            "chaser_distance_run": run_name,
        })
        return []

    positions = run_group["positions"]
    fish_xy = _read_array(positions, "fish_centroid_arena_xy")
    fish_valid = _read_array(positions, "fish_valid")
    if fish_xy is None or fish_valid is None:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "missing positions/fish_centroid_arena_xy or positions/fish_valid",
            "chaser_distance_run": run_name,
        })
        return []

    run_common, source_refs, _parameters = _chaser_common_run_fields(run_group, run_name)
    fps = _safe_float(run_common.get("fps"))
    pixels_per_mm = _safe_float(run_common.get("pixels_per_mm_projector"))
    windows = _epoch_summary_window_rows(run_group)
    if not windows:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "no epoch_summary windows",
            "chaser_distance_run": run_name,
        })
        return []

    fish_xy = np.asarray(fish_xy)
    fish_valid = np.asarray(fish_valid, dtype=bool).reshape(-1)
    finite_position = np.isfinite(fish_xy).all(axis=1) if fish_xy.ndim == 2 else np.zeros(fish_valid.shape, dtype=bool)
    usable_fish = fish_valid & finite_position
    rows: list[dict[str, Any]] = []
    for window_index, raw_window in enumerate(windows):
        window = _fill_chaser_window_times(raw_window, fps=fps)
        start_frame = _safe_int(window.get("start_frame"))
        end_frame = _safe_int(window.get("end_frame"))
        start = max(0, int(start_frame)) if start_frame is not None else 0
        end = min(int(end_frame), int(fish_valid.shape[0]) - 1) if end_frame is not None else -1
        total_span_frames = max(0, end - start + 1) if end >= start else 0
        valid_frame_count = int(np.count_nonzero(usable_fish[start : end + 1])) if total_span_frames else 0
        speeds = _epoch_speed_values_mm_s(
            fish_xy,
            fish_valid,
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps,
            pixels_per_mm=pixels_per_mm,
        )
        lineage = {
            "zarr_path": str(zarr_path),
            "chaser_distance_run": run_name,
            "source_detection_path": source_refs.get("source_detection_path"),
            "source_stimulus_run": source_refs.get("source_stimulus_run"),
            "source_stimulus_epoch_run": source_refs.get("source_stimulus_epoch_run"),
            "window_id": window.get("window_id"),
        }
        row = _common_row(
            export_run_id=export_run_id,
            zarr_path=zarr_path,
            recording_id=recording_id,
            table=table,
            lineage=lineage,
        )
        row.update(run_common)
        row.update({
            "window_index": _safe_int(window.get("window_index")),
            "window_id": _safe_int(window.get("window_id")),
            "window_label": window.get("window_label"),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time_s": _safe_float(window.get("start_time_s")),
            "end_time_s": _safe_float(window.get("end_time_s")),
            "duration_s": _safe_float(window.get("duration_s")),
            "source_position_path": f"analysis/chaser_distance_runs/{run_name}/positions/fish_centroid_arena_xy",
            "speed_definition": "consecutive valid fish_centroid_arena_xy displacement / pixels_per_mm_projector * fps; pairs must remain within epoch",
            "total_span_frames": total_span_frames,
            "valid_frame_count": valid_frame_count,
            "missing_frame_count": max(0, total_span_frames - valid_frame_count),
            "tracking_dropout_fraction": (
                float(max(0, total_span_frames - valid_frame_count)) / float(total_span_frames)
                if total_span_frames > 0
                else None
            ),
            "speed_sample_count": int(speeds.size),
            "mean_speed_mm_s": float(np.mean(speeds)) if speeds.size else None,
            "median_speed_mm_s": float(np.median(speeds)) if speeds.size else None,
            "p05_speed_mm_s": float(np.percentile(speeds, 5)) if speeds.size else None,
            "p95_speed_mm_s": float(np.percentile(speeds, 95)) if speeds.size else None,
            "max_speed_mm_s": float(np.max(speeds)) if speeds.size else None,
            "total_path_mm": float(np.sum(speeds) / float(fps)) if speeds.size and fps is not None and fps > 0 else None,
        })
        rows.append(row)
    return rows


def _frame_speed_mm_s(
    fish_xy: np.ndarray,
    fish_valid: np.ndarray,
    *,
    fps: float | None,
    pixels_per_mm: float | None,
) -> np.ndarray:
    xy = np.asarray(fish_xy, dtype=np.float64)
    valid = np.asarray(fish_valid, dtype=bool).reshape(-1)
    speeds = np.full(valid.shape[0], np.nan, dtype=np.float64)
    if fps is None or fps <= 0 or pixels_per_mm is None or pixels_per_mm <= 0:
        return speeds
    if xy.ndim != 2 or xy.shape[1] != 2 or valid.shape[0] != xy.shape[0] or xy.shape[0] < 2:
        return speeds
    finite = valid & np.isfinite(xy).all(axis=1)
    pair_valid = finite[:-1] & finite[1:]
    if not np.any(pair_valid):
        return speeds
    displacement_px = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    speeds[:-1][pair_valid] = displacement_px[pair_valid] / float(pixels_per_mm) * float(fps)
    return speeds


def _load_goodcopbadcop_speed_distance_bins(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    table = "goodcopbadcop_speed_distance_bins"
    if table not in tables:
        return []

    run_group, run_name, error = _latest_run(root, "analysis/chaser_distance_runs")
    if run_group is None or run_name is None:
        diagnostics.append({"table": table, "status": "skipped", "reason": error})
        return []
    required = ("epoch_summary", "positions", "distances")
    missing = [name for name in required if not _has_child(run_group, name)]
    if missing:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": f"missing chaser-distance group(s): {missing}",
            "chaser_distance_run": run_name,
        })
        return []

    positions = run_group["positions"]
    distances_group = run_group["distances"]
    fish_xy = _read_array(positions, "fish_centroid_arena_xy")
    fish_valid = _read_array(positions, "fish_valid")
    distance_mm = _read_array(distances_group, "distance_mm")
    if fish_xy is None or fish_valid is None or distance_mm is None:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "missing positions/fish_centroid_arena_xy, positions/fish_valid, or distances/distance_mm",
            "chaser_distance_run": run_name,
        })
        return []

    distance_mm = np.asarray(distance_mm, dtype=np.float64)
    if distance_mm.ndim != 2:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "distances/distance_mm is not 2D",
            "chaser_distance_run": run_name,
        })
        return []

    run_common, source_refs, _parameters = _chaser_common_run_fields(run_group, run_name)
    fps = _safe_float(run_common.get("fps"))
    pixels_per_mm = _safe_float(run_common.get("pixels_per_mm_projector"))
    frame_speed = _frame_speed_mm_s(fish_xy, fish_valid, fps=fps, pixels_per_mm=pixels_per_mm)
    windows = _epoch_summary_window_rows(run_group)
    if not windows:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "no epoch_summary windows",
            "chaser_distance_run": run_name,
        })
        return []

    bin_edges = None
    if _has_child(run_group, "epoch_distributions"):
        bin_edges = _read_1d_array(run_group["epoch_distributions"], "bin_edges_mm")
    if bin_edges is None or np.asarray(bin_edges).size < 2:
        finite = distance_mm[np.isfinite(distance_mm)]
        max_distance = float(np.nanmax(finite)) if finite.size else 2.0
        max_edge = max(2.0, float(math.ceil(max_distance / 2.0) * 2.0))
        bin_edges = np.arange(0.0, max_edge + 1.0, 2.0, dtype=np.float32)
    bin_edges = np.asarray(bin_edges, dtype=np.float64).reshape(-1)
    if bin_edges.size < 2:
        return []
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    chaser_indices = _chaser_indices_for_run(run_group, fallback_count=int(distance_mm.shape[1]))

    rows: list[dict[str, Any]] = []
    for window_index, raw_window in enumerate(windows):
        window = _fill_chaser_window_times(raw_window, fps=fps)
        start_frame = _safe_int(window.get("start_frame"))
        end_frame = _safe_int(window.get("end_frame"))
        start = max(0, int(start_frame)) if start_frame is not None else 0
        # Speeds are stored at the first frame in a t->t+1 movement pair, so
        # the last inclusive epoch frame cannot start a within-epoch speed.
        end = min(int(end_frame) - 1, int(distance_mm.shape[0]) - 1) if end_frame is not None else -1
        base_window = {
            "window_index": _safe_int(window.get("window_index")),
            "window_id": _safe_int(window.get("window_id")),
            "window_label": window.get("window_label"),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time_s": _safe_float(window.get("start_time_s")),
            "end_time_s": _safe_float(window.get("end_time_s")),
            "duration_s": _safe_float(window.get("duration_s")),
        }
        for chaser_column_index in range(int(distance_mm.shape[1])):
            chaser_index = (
                chaser_indices[chaser_column_index]
                if chaser_column_index < len(chaser_indices)
                else chaser_column_index
            )
            if end >= start:
                distances = distance_mm[start : end + 1, chaser_column_index]
                speeds = frame_speed[start : end + 1]
                valid = np.isfinite(distances) & np.isfinite(speeds)
            else:
                distances = np.asarray([], dtype=np.float64)
                speeds = np.asarray([], dtype=np.float64)
                valid = np.asarray([], dtype=bool)
            for bin_index in range(int(bin_edges.size) - 1):
                left = float(bin_edges[bin_index])
                right = float(bin_edges[bin_index + 1])
                if bin_index == int(bin_edges.size) - 2:
                    bin_mask = valid & (distances >= left) & (distances <= right)
                else:
                    bin_mask = valid & (distances >= left) & (distances < right)
                values = speeds[bin_mask]
                lineage = {
                    "zarr_path": str(zarr_path),
                    "chaser_distance_run": run_name,
                    "source_detection_path": source_refs.get("source_detection_path"),
                    "source_stimulus_run": source_refs.get("source_stimulus_run"),
                    "source_stimulus_epoch_run": source_refs.get("source_stimulus_epoch_run"),
                    "window_id": window.get("window_id"),
                    "chaser_index": chaser_index,
                    "distance_bin_index": bin_index,
                }
                row = _common_row(
                    export_run_id=export_run_id,
                    zarr_path=zarr_path,
                    recording_id=recording_id,
                    table=table,
                    lineage=lineage,
                )
                row.update(run_common)
                row.update(base_window)
                row.update({
                    "source_position_path": f"analysis/chaser_distance_runs/{run_name}/positions/fish_centroid_arena_xy",
                    "source_distance_path": f"analysis/chaser_distance_runs/{run_name}/distances/distance_mm",
                    "speed_distance_definition": "speed at frame t from fish position t->t+1, distance to chaser at frame t, pairs constrained within epoch",
                    "chaser_column_index": chaser_column_index,
                    "chaser_index": _safe_int(chaser_index),
                    "distance_bin_index": bin_index,
                    "distance_bin_left_mm": left,
                    "distance_bin_right_mm": right,
                    "distance_bin_center_mm": float(bin_centers[bin_index]),
                    "distance_bin_width_mm": right - left,
                    "speed_sample_count": int(values.size),
                    "speed_sum_mm_s": float(np.sum(values)) if values.size else 0.0,
                    "mean_speed_mm_s": float(np.mean(values)) if values.size else None,
                    "median_speed_mm_s": float(np.median(values)) if values.size else None,
                    "p05_speed_mm_s": float(np.percentile(values, 5)) if values.size else None,
                    "p95_speed_mm_s": float(np.percentile(values, 95)) if values.size else None,
                })
                rows.append(row)
    return rows


def _load_goodcopbadcop_chaser_distance_histogram(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    table = "goodcopbadcop_chaser_distance_histogram"
    if table not in tables:
        return []

    run_group, run_name, error = _latest_run(root, "analysis/chaser_distance_runs")
    if run_group is None or run_name is None:
        diagnostics.append({"table": table, "status": "skipped", "reason": error})
        return []
    if not _has_child(run_group, "epoch_distributions"):
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "missing epoch_distributions group",
            "chaser_distance_run": run_name,
        })
        return []

    run_common, source_refs, parameters = _chaser_common_run_fields(run_group, run_name)
    distributions = run_group["epoch_distributions"]
    hist_counts = _read_array(distributions, "hist_counts")
    if hist_counts is None or np.asarray(hist_counts).ndim != 3:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "epoch_distributions/hist_counts missing or not 3D",
            "chaser_distance_run": run_name,
        })
        return []

    hist_counts = np.asarray(hist_counts)
    n_windows, n_chasers, n_bins = (
        int(hist_counts.shape[0]),
        int(hist_counts.shape[1]),
        int(hist_counts.shape[2]),
    )
    hist_density = _read_array(distributions, "hist_density")
    valid_sample_count = _read_array(distributions, "valid_sample_count")
    bin_edges = _read_1d_array(distributions, "bin_edges_mm")
    bin_centers = _read_1d_array(distributions, "bin_centers_mm")
    dist_chaser_indices = _read_1d_array(distributions, "chaser_index")
    if dist_chaser_indices is not None and dist_chaser_indices.size:
        chaser_indices = [
            _safe_int(value) if _safe_int(value) is not None else int(idx)
            for idx, value in enumerate(dist_chaser_indices)
        ]
    else:
        chaser_indices = _chaser_indices_for_run(run_group, fallback_count=n_chasers)
    chaser_behaviors = _chaser_behaviors_for_run(run_group, chaser_indices)
    windows = _epoch_summary_window_rows(run_group)
    fps = _safe_float(run_common.get("fps"))
    dist_attrs = _attrs_dict(distributions)
    bin_width_mm = (
        _safe_float(dist_attrs.get("bin_width_mm"))
        or _safe_float(parameters.get("distribution_bin_width_mm"))
    )

    rows: list[dict[str, Any]] = []
    for window_index in range(n_windows):
        raw_window = (
            windows[window_index]
            if window_index < len(windows)
            else {
                "window_index": window_index,
                "window_id": window_index,
                "window_label": None,
                "start_frame": None,
                "end_frame": None,
                "start_time_s": None,
                "end_time_s": None,
                "duration_s": None,
            }
        )
        window = _fill_chaser_window_times(raw_window, fps=fps)
        for chaser_column_index in range(n_chasers):
            chaser_index = (
                chaser_indices[chaser_column_index]
                if chaser_column_index < len(chaser_indices)
                else chaser_column_index
            )
            behavior_class_id, behavior_class = chaser_behaviors[chaser_column_index]
            for bin_index in range(n_bins):
                bin_left = _array_float(bin_edges, bin_index)
                bin_right = _array_float(bin_edges, bin_index + 1)
                bin_center = _array_float(bin_centers, bin_index)
                if bin_center is None and bin_left is not None and bin_right is not None:
                    bin_center = (bin_left + bin_right) / 2.0
                lineage = {
                    "zarr_path": str(zarr_path),
                    "chaser_distance_run": run_name,
                    "source_detection_path": source_refs.get("source_detection_path"),
                    "source_stimulus_run": source_refs.get("source_stimulus_run"),
                    "source_stimulus_epoch_run": source_refs.get("source_stimulus_epoch_run"),
                    "window_id": window.get("window_id"),
                    "chaser_index": chaser_index,
                    "distance_bin_index": bin_index,
                    "bin_left_mm": bin_left,
                    "bin_right_mm": bin_right,
                }
                row = _common_row(
                    export_run_id=export_run_id,
                    zarr_path=zarr_path,
                    recording_id=recording_id,
                    table=table,
                    lineage=lineage,
                )
                row.update(run_common)
                row.update({
                    "window_index": _safe_int(window.get("window_index")),
                    "window_id": _safe_int(window.get("window_id")),
                    "window_label": window.get("window_label"),
                    "start_frame": _safe_int(window.get("start_frame")),
                    "end_frame": _safe_int(window.get("end_frame")),
                    "start_time_s": _safe_float(window.get("start_time_s")),
                    "end_time_s": _safe_float(window.get("end_time_s")),
                    "duration_s": _safe_float(window.get("duration_s")),
                    "chaser_column_index": chaser_column_index,
                    "chaser_index": _safe_int(chaser_index),
                    "behavior_class_id": behavior_class_id,
                    "behavior_class": behavior_class,
                    "distance_bin_index": bin_index,
                    "bin_left_mm": bin_left,
                    "bin_right_mm": bin_right,
                    "bin_center_mm": bin_center,
                    "bin_width_mm": bin_width_mm,
                    "hist_count": _array_int(hist_counts, window_index, chaser_column_index, bin_index),
                    "hist_density": _array_float(hist_density, window_index, chaser_column_index, bin_index),
                    "valid_sample_count": _array_int(valid_sample_count, window_index, chaser_column_index),
                    "density_normalization": dist_attrs.get("density_normalization"),
                })
                rows.append(row)
    return rows


def _latest_egocentric_component(
    run_group: Any,
) -> tuple[Any | None, str | None, str | None]:
    if not _has_child(run_group, "egocentric_bearing"):
        return None, None, "missing egocentric_bearing group"
    parent = run_group["egocentric_bearing"]
    parent_attrs = _attrs_dict(parent)
    candidate = (
        str(parent_attrs.get("latest_complete") or "").strip()
        or str(parent_attrs.get("latest") or "").strip()
    )
    group_names = _group_names(parent)
    if not candidate and group_names:
        candidate = group_names[-1]
    if not candidate or candidate not in group_names:
        return None, None, "no egocentric_bearing component found"
    component = parent[candidate]
    component_attrs = _attrs_dict(component)
    status = str(component_attrs.get("status") or "").strip().lower()
    if status and status != "complete":
        return None, None, f"egocentric_bearing component is not complete: {candidate}"
    return component, candidate, None


def _egocentric_chaser_indices(component: Any, *, fallback_count: int) -> list[int]:
    if _has_child(component, "per_chaser"):
        arr = _read_1d_array(component["per_chaser"], "chaser_index")
        if arr is not None and arr.size:
            return [_safe_int(value) if _safe_int(value) is not None else int(idx) for idx, value in enumerate(arr)]
    if _has_child(component, "distance_bearing_histogram"):
        arr = _read_1d_array(component["distance_bearing_histogram"], "chaser_index")
        if arr is not None and arr.size:
            return [_safe_int(value) if _safe_int(value) is not None else int(idx) for idx, value in enumerate(arr)]
    return list(range(int(fallback_count)))


def _egocentric_common_fields(
    *,
    component: Any,
    component_name: str,
    run_name: str,
    run_path: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    attrs = _attrs_dict(component)
    source_refs = _json_mapping_attr(attrs, "source_refs")
    parameters = _json_mapping_attr(attrs, "parameters")
    common = {
        "egocentric_component_name": component_name,
        "egocentric_component_path": f"{run_path}/egocentric_bearing/{component_name}",
        "egocentric_schema_id": attrs.get("schema_id"),
        "egocentric_schema_version": _safe_int(attrs.get("schema_version")),
        "egocentric_method": attrs.get("method"),
        "egocentric_method_version": attrs.get("method_version"),
        "egocentric_created_at_utc": attrs.get("created_at_utc"),
        "egocentric_source_refs_json": _source_refs_json(source_refs),
        "egocentric_parameters_json": _source_refs_json(parameters),
        "source_chaser_distance_run": source_refs.get("source_chaser_distance_run") or run_name,
        "source_chaser_distance_path": source_refs.get("source_chaser_distance_path") or run_path,
        "source_track_kinematics_run": source_refs.get("source_track_kinematics_run"),
        "source_track_kinematics_scope": source_refs.get("source_track_kinematics_scope"),
        "source_track_kinematics_track_id": _safe_int(source_refs.get("source_track_kinematics_track_id")),
        "source_track_kinematics_track_path": source_refs.get("source_track_kinematics_track_path"),
        "source_heading_array": source_refs.get("source_heading_array"),
        "heading_level": parameters.get("heading_level"),
        "angle_convention": parameters.get("angle_convention"),
        "distance_bin_width_mm": _safe_float(parameters.get("distance_bin_width_mm")),
        "bearing_bin_width_deg": _safe_float(parameters.get("bearing_bin_width_deg")),
    }
    return common, source_refs, parameters


def _latest_cra_primary_endpoint_component(
    run_group: Any,
) -> tuple[Any | None, str | None, str | None]:
    if not _has_child(run_group, CRA_PRIMARY_ENDPOINT_COMPONENT_PARENT):
        return None, None, f"missing {CRA_PRIMARY_ENDPOINT_COMPONENT_PARENT} group"
    parent = run_group[CRA_PRIMARY_ENDPOINT_COMPONENT_PARENT]
    parent_attrs = _attrs_dict(parent)
    candidate = (
        str(parent_attrs.get("latest_complete") or "").strip()
        or str(parent_attrs.get("latest") or "").strip()
    )
    group_names = _group_names(parent)
    if not candidate and group_names:
        candidate = group_names[-1]
    if not candidate or candidate not in group_names:
        return None, None, "no chaser quadrant occupancy component found"
    component = parent[candidate]
    component_attrs = _attrs_dict(component)
    status = str(component_attrs.get("status") or "").strip().lower()
    if status and status not in CRA_PRIMARY_ENDPOINT_ALLOWED_STATUSES:
        return (
            None,
            None,
            f"chaser quadrant occupancy component is not computed: {candidate}",
        )
    if component_attrs.get("schema_id") != "palette.chaser.quadrant_occupancy.v1":
        return (
            None,
            None,
            "chaser quadrant occupancy component has an unsupported schema",
        )
    return component, candidate, None


def _latest_cra_near_field_component(
    run_group: Any,
) -> tuple[Any | None, str | None, str | None]:
    if not _has_child(run_group, CRA_NEAR_FIELD_COMPONENT_PARENT):
        return None, None, f"missing {CRA_NEAR_FIELD_COMPONENT_PARENT} group"
    parent = run_group[CRA_NEAR_FIELD_COMPONENT_PARENT]
    parent_attrs = _attrs_dict(parent)
    candidate = (
        str(parent_attrs.get("latest_complete") or "").strip()
        or str(parent_attrs.get("latest") or "").strip()
    )
    group_names = _group_names(parent)
    if not candidate and group_names:
        candidate = group_names[-1]
    if not candidate or candidate not in group_names:
        return None, None, "no chaser near-field occupancy component found"
    component = parent[candidate]
    component_attrs = _attrs_dict(component)
    status = str(component_attrs.get("status") or "").strip().lower()
    if status and status not in CRA_NEAR_FIELD_ALLOWED_STATUSES:
        return (
            None,
            None,
            f"chaser near-field occupancy component is not computed: {candidate}",
        )
    if component_attrs.get("schema_id") != "palette.chaser.near_field_occupancy.v1":
        return (
            None,
            None,
            "chaser near-field occupancy component has an unsupported schema",
        )
    return component, candidate, None


def _cra_primary_endpoint_common_fields(
    *,
    component: Any,
    component_name: str,
    run_name: str,
    run_path: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    attrs = _attrs_dict(component)
    source_refs = _json_mapping_attr(attrs, "source_refs")
    parameters = _json_mapping_attr(attrs, "parameters")
    component_path = f"{run_path}/{CRA_PRIMARY_ENDPOINT_COMPONENT_PARENT}/{component_name}"
    fingerprint = (
        attrs.get("source_fingerprint")
        or attrs.get("source_lineage_hash")
        or attrs.get("lineage_hash")
    )
    common = {
        "cra_primary_endpoint_component": component_name,
        "cra_primary_endpoint_path": component_path,
        "source_cra_primary_endpoint_component": component_name,
        "source_cra_primary_endpoint_path": component_path,
        "source_component_schema_id": attrs.get("schema_id"),
        "source_component_schema_version": _safe_int(attrs.get("schema_version")),
        "source_component_fingerprint": fingerprint,
        "source_component_fingerprint_status": attrs.get("fingerprint_status"),
        "cra_primary_endpoint_schema_id": attrs.get("schema_id"),
        "cra_primary_endpoint_schema_version": _safe_int(attrs.get("schema_version")),
        "cra_primary_endpoint_method": attrs.get("method"),
        "cra_primary_endpoint_method_version": attrs.get("method_version"),
        "cra_primary_endpoint_created_at_utc": attrs.get("created_at_utc"),
        "endpoint_status": attrs.get("status"),
        "cra_primary_endpoint_source_refs_json": _source_refs_json(source_refs),
        "cra_primary_endpoint_parameters_json": _source_refs_json(parameters),
        "qc_warnings_json": _json_dumps_safe(_parse_jsonish(attrs.get("qc_warnings")) or []),
        "diagnostics_json": _json_dumps_safe(_parse_jsonish(attrs.get("diagnostics")) or {}),
        "source_chaser_distance_run": source_refs.get("source_chaser_distance_run") or run_name,
        "source_chaser_distance_path": source_refs.get("source_chaser_distance_path") or run_path,
        "source_stimulus_run": source_refs.get("source_stimulus_run"),
        "source_stimulus_path": source_refs.get("source_stimulus_path"),
        "source_stimulus_epoch_run": source_refs.get("source_stimulus_epoch_run"),
        "source_stimulus_epoch_path": source_refs.get("source_stimulus_epoch_path"),
        "coordinate_frame": attrs.get("coordinate_frame"),
        "coordinate_origin": attrs.get("coordinate_origin"),
        "x_axis_direction": attrs.get("x_axis_direction"),
        "y_axis_direction": attrs.get("y_axis_direction"),
        "quadrant_bounds_source": attrs.get("quadrant_bounds_source"),
        "quadrant_width_px": _safe_float(attrs.get("quadrant_width_px")),
        "quadrant_height_px": _safe_float(attrs.get("quadrant_height_px")),
        "pixels_per_mm_projector": _safe_float(attrs.get("pixels_per_mm_projector")),
    }
    return common, source_refs, parameters


def _cra_near_field_common_fields(
    *,
    component: Any,
    component_name: str,
    run_name: str,
    run_path: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    attrs = _attrs_dict(component)
    source_refs = _json_mapping_attr(attrs, "source_refs")
    parameters = _json_mapping_attr(attrs, "parameters")
    component_path = f"{run_path}/{CRA_NEAR_FIELD_COMPONENT_PARENT}/{component_name}"
    fingerprint = (
        attrs.get("source_fingerprint")
        or attrs.get("source_lineage_hash")
        or attrs.get("lineage_hash")
    )
    common = {
        "cra_near_field_component": component_name,
        "cra_near_field_path": component_path,
        "source_cra_near_field_component": component_name,
        "source_cra_near_field_path": component_path,
        "source_component_schema_id": attrs.get("schema_id"),
        "source_component_schema_version": _safe_int(attrs.get("schema_version")),
        "source_component_fingerprint": fingerprint,
        "source_component_fingerprint_status": attrs.get("fingerprint_status"),
        "cra_near_field_schema_id": attrs.get("schema_id"),
        "cra_near_field_schema_version": _safe_int(attrs.get("schema_version")),
        "cra_near_field_method": attrs.get("method"),
        "cra_near_field_method_version": attrs.get("method_version"),
        "cra_near_field_created_at_utc": attrs.get("created_at_utc"),
        "endpoint_status": attrs.get("status"),
        "cra_near_field_source_refs_json": _source_refs_json(source_refs),
        "cra_near_field_parameters_json": _source_refs_json(parameters),
        "qc_warnings_json": _json_dumps_safe(
            _parse_jsonish(attrs.get("qc_warnings")) or []
        ),
        "diagnostics_json": _json_dumps_safe(
            _parse_jsonish(attrs.get("diagnostics")) or {}
        ),
        "source_chaser_distance_run": source_refs.get("source_chaser_distance_run")
        or run_name,
        "source_chaser_distance_path": source_refs.get("source_chaser_distance_path")
        or run_path,
        "source_quadrant_occupancy_component": source_refs.get(
            "source_quadrant_occupancy_component"
        ),
        "source_quadrant_occupancy_path": source_refs.get(
            "source_quadrant_occupancy_path"
        ),
        "source_stimulus_run": source_refs.get("source_stimulus_run"),
        "source_stimulus_path": source_refs.get("source_stimulus_path"),
        "source_stimulus_epoch_run": source_refs.get("source_stimulus_epoch_run"),
        "source_stimulus_epoch_path": source_refs.get("source_stimulus_epoch_path"),
        "coordinate_frame": attrs.get("coordinate_frame"),
        "coordinate_origin": attrs.get("coordinate_origin"),
        "x_axis_direction": attrs.get("x_axis_direction"),
        "y_axis_direction": attrs.get("y_axis_direction"),
        "pixels_per_mm_projector": _safe_float(attrs.get("pixels_per_mm_projector")),
        "geometry_status": attrs.get("geometry_status") or parameters.get("geometry_status") or parameters.get("geometry_mode"),
        "arena_shape": attrs.get("arena_shape") or parameters.get("arena_shape"),
        "arena_geometry_source": attrs.get("arena_geometry_source") or parameters.get("arena_geometry_source"),
        "arena_center_x_px": _safe_float(attrs.get("arena_center_x_px") or parameters.get("arena_center_x_px")),
        "arena_center_y_px": _safe_float(attrs.get("arena_center_y_px") or parameters.get("arena_center_y_px")),
        "arena_radius_px": _safe_float(attrs.get("arena_radius_px") or parameters.get("arena_radius_px")),
        "arena_width_px": _safe_float(attrs.get("arena_width_px")),
        "arena_height_px": _safe_float(attrs.get("arena_height_px")),
        "r_zone_mm": _safe_float(parameters.get("r_zone_mm")),
        "r_in_mm": _safe_float(parameters.get("r_in_mm")),
        "r_out_mm": _safe_float(parameters.get("r_out_mm")),
        "perimeter_band_mm": _safe_float(parameters.get("perimeter_band_mm")),
    }
    return common, source_refs, parameters


def _cra_summary_mapping_from_component(component: Any) -> dict[str, Any]:
    attrs = _attrs_dict(component)
    summary = _json_mapping_attr(attrs, "summary")
    if summary:
        return {str(key): _scalar_for_parquet(value) for key, value in summary.items()}
    if not _has_child(component, "summary"):
        return {}
    group = component["summary"]
    out: dict[str, Any] = {}
    for name in _array_names(group):
        values = _read_array(group, name)
        if values is None or np.asarray(values).shape[0] < 1:
            continue
        key = str(name)
        if key.endswith("_bytes"):
            decoded = _decode_text_column(values[:1], fallback_count=1)
            out[key[: -len("_bytes")]] = decoded[0] if decoded else None
        else:
            out[key] = _array_scalar(values, 0)
    return out


def _load_goodcopbadcop_cra_primary_endpoint_summary(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    table = "goodcopbadcop_cra_primary_endpoint_summary"
    if table not in tables:
        return []

    run_group, run_name, error = _latest_run(root, "analysis/chaser_distance_runs")
    if run_group is None or run_name is None:
        diagnostics.append({"table": table, "status": "skipped", "reason": error})
        return []
    component, component_name, component_error = _latest_cra_primary_endpoint_component(run_group)
    if component is None or component_name is None:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": component_error,
            "chaser_distance_run": run_name,
        })
        return []

    run_common, _source_refs, _parameters = _chaser_common_run_fields(run_group, run_name)
    run_path = str(run_common["chaser_distance_path"])
    component_common, cra_refs, _cra_parameters = _cra_primary_endpoint_common_fields(
        component=component,
        component_name=component_name,
        run_name=run_name,
        run_path=run_path,
    )
    summary = _cra_summary_mapping_from_component(component)
    if not summary:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "missing CRA primary endpoint summary",
            "chaser_distance_run": run_name,
            "cra_primary_endpoint_component": component_name,
        })
        return []

    lineage = {
        "zarr_path": str(zarr_path),
        "chaser_distance_run": run_name,
        "cra_primary_endpoint_component": component_name,
        "source_component_fingerprint": component_common.get("source_component_fingerprint"),
        "source_detection_path": run_common.get("source_detection_path"),
        "source_stimulus_run": cra_refs.get("source_stimulus_run"),
        "source_stimulus_epoch_run": cra_refs.get("source_stimulus_epoch_run"),
        "fish_id": summary.get("fish_id"),
    }
    row = _common_row(
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        table=table,
        lineage=lineage,
    )
    row.update(run_common)
    row.update(component_common)
    summary_for_row = dict(summary)
    component_recording_id = summary_for_row.pop("recording_id", None)
    row.update(summary_for_row)
    row["cra_summary_recording_id"] = component_recording_id
    return [row]


def _load_goodcopbadcop_cra_primary_endpoint_object_phase(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    table = "goodcopbadcop_cra_primary_endpoint_object_phase"
    if table not in tables:
        return []

    run_group, run_name, error = _latest_run(root, "analysis/chaser_distance_runs")
    if run_group is None or run_name is None:
        diagnostics.append({"table": table, "status": "skipped", "reason": error})
        return []
    component, component_name, component_error = _latest_cra_primary_endpoint_component(run_group)
    if component is None or component_name is None:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": component_error,
            "chaser_distance_run": run_name,
        })
        return []
    required = ("chasers", "phases", "chaser_phase", "per_chaser_phase")
    missing = [name for name in required if not _has_child(component, name)]
    if missing:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": f"missing CRA primary endpoint group(s): {missing}",
            "chaser_distance_run": run_name,
            "cra_primary_endpoint_component": component_name,
        })
        return []

    run_common, _source_refs, _parameters = _chaser_common_run_fields(run_group, run_name)
    run_path = str(run_common["chaser_distance_path"])
    component_common, cra_refs, _cra_parameters = _cra_primary_endpoint_common_fields(
        component=component,
        component_name=component_name,
        run_name=run_name,
        run_path=run_path,
    )
    objects = component["chasers"]
    phases = component["phases"]
    object_phase = component["chaser_phase"]
    per_object = component["per_chaser_phase"]

    object_indices = _read_1d_array(objects, "chaser_index")
    object_roles = _read_canonical_behavior_labels(objects)
    raw_colors = _decode_text_column(_read_array(objects, "raw_color_hex_bytes"))
    enable_chase = _read_1d_array(objects, "enable_chase")
    behavior_mode = _read_1d_array(objects, "behavior_mode")
    start_preset = _decode_text_column(_read_array(objects, "start_position_preset_bytes"))
    end_preset = _decode_text_column(_read_array(objects, "end_position_preset_bytes"))

    phase_indices = _read_1d_array(phases, "phase_index")
    phase_labels = _decode_text_column(_read_array(phases, "phase_label_bytes"))
    source_window_labels = _decode_text_column(_read_array(phases, "source_window_label_bytes"))
    source_start = _read_1d_array(phases, "source_start_frame")
    source_end = _read_1d_array(phases, "source_end_frame")
    effective_start = _read_1d_array(phases, "effective_start_frame")
    effective_end = _read_1d_array(phases, "effective_end_frame")
    settle_excluded = _read_1d_array(phases, "settle_excluded_frame_count")

    object_x_px = _read_array(object_phase, "chaser_x_px")
    if object_x_px is None or np.asarray(object_x_px).ndim != 2:
        diagnostics.append(
            {
                "table": table,
                "status": "skipped",
                "reason": "chaser_phase/chaser_x_px missing or not 2D",
                "chaser_distance_run": run_name,
                "cra_primary_endpoint_component": component_name,
            }
        )
        return []
    object_x_px = np.asarray(object_x_px)
    n_phases, n_objects = int(object_x_px.shape[0]), int(object_x_px.shape[1])
    object_y_px = _read_array(object_phase, "chaser_y_px")
    object_x_mm = _read_array(object_phase, "chaser_x_mm")
    object_y_mm = _read_array(object_phase, "chaser_y_mm")
    quadrant_code = _read_array(object_phase, "chaser_quadrant_code")
    quadrant_labels = _decode_text_column(
        _read_array(object_phase, "chaser_quadrant_label_bytes")
    )
    sample_count = _read_array(object_phase, "chaser_position_sample_count")
    max_drift = _read_array(object_phase, "chaser_max_drift_mm")
    median_drift = _read_array(object_phase, "chaser_median_drift_mm")

    median_distance = _read_array(per_object, "median_distance_mm")
    mean_distance = _read_array(per_object, "mean_distance_mm")
    occupancy = _read_array(per_object, "occupancy_fraction")
    occupancy_epoch = _read_array(per_object, "occupancy_fraction_of_epoch")
    valid_count = _read_array(per_object, "valid_frame_count")
    distance_valid_count = _read_array(per_object, "distance_valid_frame_count")
    total_count = _read_array(per_object, "total_frame_count")
    missing_count = _read_array(per_object, "missing_frame_count")
    dropout = _read_array(per_object, "tracking_dropout_fraction")

    rows: list[dict[str, Any]] = []
    for phase_idx in range(n_phases):
        for object_col in range(n_objects):
            q_code = _array_int(quadrant_code, phase_idx, object_col)
            q_label = (
                quadrant_labels[q_code]
                if q_code is not None and 0 <= q_code < len(quadrant_labels)
                else None
            )
            object_index = _array_int(object_indices, object_col)
            object_role = object_roles[object_col] if object_col < len(object_roles) else None
            lineage = {
                "zarr_path": str(zarr_path),
                "chaser_distance_run": run_name,
                "cra_primary_endpoint_component": component_name,
                "source_component_fingerprint": component_common.get("source_component_fingerprint"),
                "source_detection_path": run_common.get("source_detection_path"),
                "source_stimulus_run": cra_refs.get("source_stimulus_run"),
                "source_stimulus_epoch_run": cra_refs.get("source_stimulus_epoch_run"),
                "phase_index": phase_idx,
                "object_index": object_index,
                "object_role": object_role,
                "behavior_class": object_role,
            }
            row = _common_row(
                export_run_id=export_run_id,
                zarr_path=zarr_path,
                recording_id=recording_id,
                table=table,
                lineage=lineage,
            )
            row.update(run_common)
            row.update(component_common)
            row.update({
                "phase_axis_index": phase_idx,
                "phase_index": _array_int(phase_indices, phase_idx),
                "phase_label": phase_labels[phase_idx] if phase_idx < len(phase_labels) else None,
                "source_window_label": (
                    source_window_labels[phase_idx]
                    if phase_idx < len(source_window_labels)
                    else None
                ),
                "source_start_frame": _array_int(source_start, phase_idx),
                "source_end_frame": _array_int(source_end, phase_idx),
                "effective_start_frame": _array_int(effective_start, phase_idx),
                "effective_end_frame": _array_int(effective_end, phase_idx),
                "settle_excluded_frame_count": _array_int(settle_excluded, phase_idx),
                "object_column_index": object_col,
                "object_index": object_index,
                "object_role": object_role,
                "behavior_class": object_role,
                "raw_color_hex": raw_colors[object_col] if object_col < len(raw_colors) else None,
                "enable_chase": _scalar_for_parquet(_array_scalar(enable_chase, object_col)),
                "behavior_mode": _array_int(behavior_mode, object_col),
                "start_position_preset": start_preset[object_col] if object_col < len(start_preset) else None,
                "end_position_preset": end_preset[object_col] if object_col < len(end_preset) else None,
                "object_x_px": _array_float(object_x_px, phase_idx, object_col),
                "object_y_px": _array_float(object_y_px, phase_idx, object_col),
                "object_x_mm": _array_float(object_x_mm, phase_idx, object_col),
                "object_y_mm": _array_float(object_y_mm, phase_idx, object_col),
                "object_quadrant_code": q_code,
                "object_quadrant_label": q_label,
                "object_position_sample_count": _array_int(sample_count, phase_idx, object_col),
                "object_max_drift_mm": _array_float(max_drift, phase_idx, object_col),
                "object_median_drift_mm": _array_float(median_drift, phase_idx, object_col),
                "median_distance_mm": _array_float(median_distance, phase_idx, object_col),
                "mean_distance_mm": _array_float(mean_distance, phase_idx, object_col),
                "occupancy_fraction": _array_float(occupancy, phase_idx, object_col),
                "occupancy_fraction_of_epoch": _array_float(occupancy_epoch, phase_idx, object_col),
                "valid_frame_count": _array_int(valid_count, phase_idx, object_col),
                "distance_valid_frame_count": _array_int(distance_valid_count, phase_idx, object_col),
                "total_frame_count": _array_int(total_count, phase_idx, object_col),
                "missing_frame_count": _array_int(missing_count, phase_idx, object_col),
                "tracking_dropout_fraction": _array_float(dropout, phase_idx, object_col),
            })
            rows.append(row)
    return rows


def _load_goodcopbadcop_cra_quadrant_occupancy(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    table = "goodcopbadcop_cra_quadrant_occupancy"
    if table not in tables:
        return []

    run_group, run_name, error = _latest_run(root, "analysis/chaser_distance_runs")
    if run_group is None or run_name is None:
        diagnostics.append({"table": table, "status": "skipped", "reason": error})
        return []
    component, component_name, component_error = _latest_cra_primary_endpoint_component(run_group)
    if component is None or component_name is None:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": component_error,
            "chaser_distance_run": run_name,
        })
        return []
    if not _has_child(run_group, "positions"):
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "chaser-distance run missing positions group",
            "chaser_distance_run": run_name,
            "cra_primary_endpoint_component": component_name,
        })
        return []
    required_component = ("chasers", "phases", "chaser_phase")
    missing_component = [
        name for name in required_component if not _has_child(component, name)
    ]
    required_positions = ("fish_centroid_arena_xy", "fish_valid")
    missing_positions = [name for name in required_positions if not _has_child(run_group["positions"], name)]
    if missing_component or missing_positions:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": f"missing group/array(s): component={missing_component}, positions={missing_positions}",
            "chaser_distance_run": run_name,
            "cra_primary_endpoint_component": component_name,
        })
        return []

    run_common, _source_refs, _parameters = _chaser_common_run_fields(run_group, run_name)
    run_path = str(run_common["chaser_distance_path"])
    component_common, cra_refs, _cra_parameters = _cra_primary_endpoint_common_fields(
        component=component,
        component_name=component_name,
        run_name=run_name,
        run_path=run_path,
    )
    width_px = _safe_float(component_common.get("quadrant_width_px"))
    height_px = _safe_float(component_common.get("quadrant_height_px"))
    if width_px is None or height_px is None or width_px <= 0 or height_px <= 0:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "CRA primary endpoint lacks positive quadrant dimensions",
            "chaser_distance_run": run_name,
            "cra_primary_endpoint_component": component_name,
        })
        return []

    objects = component["chasers"]
    phases = component["phases"]
    object_phase = component["chaser_phase"]
    positions = run_group["positions"]

    object_indices = _read_1d_array(objects, "chaser_index")
    object_roles = _read_canonical_behavior_labels(objects)
    raw_colors = _decode_text_column(_read_array(objects, "raw_color_hex_bytes"))
    aggressive_cols = [idx for idx, role in enumerate(object_roles) if role == "aggressive"]
    if len(aggressive_cols) != 1:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": f"expected exactly one aggressive object role, found {len(aggressive_cols)}",
            "chaser_distance_run": run_name,
            "cra_primary_endpoint_component": component_name,
        })
        return []
    aggressive_col = int(aggressive_cols[0])

    phase_indices = _read_1d_array(phases, "phase_index")
    phase_labels = _decode_text_column(_read_array(phases, "phase_label_bytes"))
    source_window_labels = _decode_text_column(_read_array(phases, "source_window_label_bytes"))
    source_start = _read_1d_array(phases, "source_start_frame")
    source_end = _read_1d_array(phases, "source_end_frame")
    effective_start = _read_1d_array(phases, "effective_start_frame")
    effective_end = _read_1d_array(phases, "effective_end_frame")
    settle_excluded = _read_1d_array(phases, "settle_excluded_frame_count")
    object_x_px = _read_array(object_phase, "chaser_x_px")
    object_y_px = _read_array(object_phase, "chaser_y_px")
    object_quadrant_code = _read_array(object_phase, "chaser_quadrant_code")
    if object_quadrant_code is None or np.asarray(object_quadrant_code).ndim != 2:
        diagnostics.append(
            {
                "table": table,
                "status": "skipped",
                "reason": "chaser_phase/chaser_quadrant_code missing or not 2D",
                "chaser_distance_run": run_name,
                "cra_primary_endpoint_component": component_name,
            }
        )
        return []

    try:
        fish_xy = np.asarray(positions["fish_centroid_arena_xy"][:], dtype=np.float64)
        fish_valid = np.asarray(positions["fish_valid"][:], dtype=bool).reshape(-1)
    except Exception as exc:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": f"failed reading fish positions: {exc}",
            "chaser_distance_run": run_name,
            "cra_primary_endpoint_component": component_name,
        })
        return []
    if fish_xy.ndim != 2 or fish_xy.shape[1] != 2 or fish_xy.shape[0] != fish_valid.shape[0]:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "fish_centroid_arena_xy/fish_valid shape mismatch",
            "chaser_distance_run": run_name,
            "cra_primary_endpoint_component": component_name,
        })
        return []

    def _label_for_code(code: int | None) -> str | None:
        if code is None or code < 0 or code >= len(QUADRANT_LABELS):
            return None
        return QUADRANT_LABELS[int(code)]

    total_frames = int(fish_xy.shape[0])
    rows: list[dict[str, Any]] = []
    n_phases = len(phase_labels) if phase_labels else int(np.asarray(object_quadrant_code).shape[0])
    for phase_idx in range(n_phases):
        start = _array_int(effective_start, phase_idx)
        end = _array_int(effective_end, phase_idx)
        if start is None or end is None:
            continue
        start = max(0, int(start))
        end = min(total_frames - 1, int(end))
        if end < start:
            phase_len = 0
            phase_xy = np.zeros((0, 2), dtype=np.float64)
            phase_valid = np.zeros(0, dtype=bool)
        else:
            phase_len = int(end - start + 1)
            phase_xy = fish_xy[start : end + 1]
            phase_valid = fish_valid[start : end + 1] & np.isfinite(phase_xy).all(axis=1)

        quadrant_codes = np.asarray(
            [
                quadrant_code_for_xy(float(x), float(y), width_px=float(width_px), height_px=float(height_px))
                for x, y in phase_xy
            ],
            dtype=np.int16,
        )
        valid_in_arena = phase_valid & (quadrant_codes >= 0)
        valid_frame_count = int(np.count_nonzero(phase_valid))
        quadrant_valid_frame_count = int(np.count_nonzero(valid_in_arena))
        out_of_bounds_frame_count = max(0, valid_frame_count - quadrant_valid_frame_count)
        missing_frame_count = max(0, phase_len - valid_frame_count)
        counts = np.bincount(quadrant_codes[valid_in_arena].astype(np.int64), minlength=len(QUADRANT_LABELS))[
            : len(QUADRANT_LABELS)
        ]
        occupancy = np.divide(
            counts.astype(np.float64),
            float(quadrant_valid_frame_count),
            out=np.full(len(QUADRANT_LABELS), np.nan, dtype=np.float64),
            where=quadrant_valid_frame_count > 0,
        )
        occupancy_epoch = np.divide(
            counts.astype(np.float64),
            float(phase_len),
            out=np.full(len(QUADRANT_LABELS), np.nan, dtype=np.float64),
            where=phase_len > 0,
        )

        chaser_quadrant_code = _array_int(object_quadrant_code, phase_idx, aggressive_col)
        chaser_quadrant_label = _label_for_code(chaser_quadrant_code)
        chaser_quadrant_occ = (
            float(occupancy[int(chaser_quadrant_code)])
            if chaser_quadrant_code is not None
            and 0 <= int(chaser_quadrant_code) < len(QUADRANT_LABELS)
            and math.isfinite(float(occupancy[int(chaser_quadrant_code)]))
            else None
        )
        for quadrant_code, quadrant_label in enumerate(QUADRANT_LABELS):
            lineage = {
                "zarr_path": str(zarr_path),
                "chaser_distance_run": run_name,
                "cra_primary_endpoint_component": component_name,
                "source_component_fingerprint": component_common.get("source_component_fingerprint"),
                "source_detection_path": run_common.get("source_detection_path"),
                "source_stimulus_run": cra_refs.get("source_stimulus_run"),
                "source_stimulus_epoch_run": cra_refs.get("source_stimulus_epoch_run"),
                "phase_index": phase_idx,
                "quadrant_code": quadrant_code,
                "object_role": "aggressive",
                "behavior_class": "aggressive",
            }
            row = _common_row(
                export_run_id=export_run_id,
                zarr_path=zarr_path,
                recording_id=recording_id,
                table=table,
                lineage=lineage,
            )
            row.update(run_common)
            row.update(component_common)
            value = float(occupancy[quadrant_code])
            epoch_value = float(occupancy_epoch[quadrant_code])
            row.update({
                "fish_id": "0",
                "phase_axis_index": phase_idx,
                "phase_index": _array_int(phase_indices, phase_idx),
                "phase_label": phase_labels[phase_idx] if phase_idx < len(phase_labels) else None,
                "source_window_label": (
                    source_window_labels[phase_idx]
                    if phase_idx < len(source_window_labels)
                    else None
                ),
                "source_start_frame": _array_int(source_start, phase_idx),
                "source_end_frame": _array_int(source_end, phase_idx),
                "effective_start_frame": _array_int(effective_start, phase_idx),
                "effective_end_frame": _array_int(effective_end, phase_idx),
                "settle_excluded_frame_count": _array_int(settle_excluded, phase_idx),
                "quadrant_code": quadrant_code,
                "quadrant_id": quadrant_label,
                "quadrant_label": quadrant_label.replace("_", " ").title(),
                "display_order": quadrant_code,
                "frame_count": int(counts[quadrant_code]),
                "occupancy_fraction": value if math.isfinite(value) else None,
                "fraction_of_detected": value if math.isfinite(value) else None,
                "occupancy_fraction_of_epoch": epoch_value if math.isfinite(epoch_value) else None,
                "fraction_of_epoch": epoch_value if math.isfinite(epoch_value) else None,
                "total_frame_count": int(phase_len),
                "valid_frame_count": valid_frame_count,
                "quadrant_valid_frame_count": quadrant_valid_frame_count,
                "missing_frame_count": missing_frame_count,
                "out_of_bounds_frame_count": out_of_bounds_frame_count,
                "tracking_dropout_fraction": (
                    float(missing_frame_count) / float(phase_len)
                    if phase_len > 0
                    else None
                ),
                "chaser_object_index": _array_int(object_indices, aggressive_col),
                "chaser_object_role": "aggressive",
                "chaser_raw_color_hex": (
                    raw_colors[aggressive_col] if aggressive_col < len(raw_colors) else None
                ),
                "chaser_x_px": _array_float(object_x_px, phase_idx, aggressive_col),
                "chaser_y_px": _array_float(object_y_px, phase_idx, aggressive_col),
                "chaser_quadrant_code": chaser_quadrant_code,
                "chaser_quadrant_label": chaser_quadrant_label,
                "chaser_quadrant_occ": chaser_quadrant_occ,
                "is_chaser_quadrant": bool(chaser_quadrant_code == quadrant_code),
                "series_role": "chaser" if chaser_quadrant_code == quadrant_code else "non_chaser",
            })
            rows.append(row)
    return rows


def _load_goodcopbadcop_cra_near_field_summary(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    table = "goodcopbadcop_cra_near_field_summary"
    if table not in tables:
        return []

    run_group, run_name, error = _latest_run(root, "analysis/chaser_distance_runs")
    if run_group is None or run_name is None:
        diagnostics.append({"table": table, "status": "skipped", "reason": error})
        return []
    component, component_name, component_error = _latest_cra_near_field_component(run_group)
    if component is None or component_name is None:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": component_error,
            "chaser_distance_run": run_name,
        })
        return []

    run_common, _source_refs, _parameters = _chaser_common_run_fields(run_group, run_name)
    run_path = str(run_common["chaser_distance_path"])
    component_common, near_refs, _near_parameters = _cra_near_field_common_fields(
        component=component,
        component_name=component_name,
        run_name=run_name,
        run_path=run_path,
    )
    summary = _cra_summary_mapping_from_component(component)
    if not summary:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "missing CRA near-field summary",
            "chaser_distance_run": run_name,
            "cra_near_field_component": component_name,
        })
        return []

    lineage = {
        "zarr_path": str(zarr_path),
        "chaser_distance_run": run_name,
        "cra_near_field_component": component_name,
        "source_component_fingerprint": component_common.get(
            "source_component_fingerprint"
        ),
        "source_quadrant_occupancy_path": near_refs.get(
            "source_quadrant_occupancy_path"
        ),
        "source_detection_path": run_common.get("source_detection_path"),
        "source_stimulus_run": near_refs.get("source_stimulus_run"),
        "source_stimulus_epoch_run": near_refs.get("source_stimulus_epoch_run"),
        "fish_id": summary.get("fish_id"),
    }
    row = _common_row(
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        table=table,
        lineage=lineage,
    )
    row.update(run_common)
    row.update(component_common)
    summary_for_row = dict(summary)
    component_recording_id = summary_for_row.pop("recording_id", None)
    row.update(summary_for_row)
    row["cra_near_field_summary_recording_id"] = component_recording_id
    return [row]


def _near_field_percentile_columns(component: Any, approach: np.ndarray | None) -> list[tuple[int, str, float | None]]:
    percentiles = None
    if _has_child(component, "config"):
        percentiles = _read_1d_array(component["config"], "percentile_values")
    out: list[tuple[int, str, float | None]] = []
    n_percentiles = int(np.asarray(approach).shape[2]) if approach is not None and np.asarray(approach).ndim == 3 else 0
    for index in range(n_percentiles):
        value = _array_float(percentiles, index)
        if value is None:
            key = f"p{index:02d}"
        elif math.isclose(float(value), round(float(value))):
            key = f"p{int(round(float(value))):02d}"
        else:
            text = f"{float(value):g}".replace(".", "_").replace("-", "m")
            key = f"p{text}"
        out.append((index, f"approach_{key}_mm", value))
    return out


def _load_goodcopbadcop_cra_near_field_object_phase(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    table = "goodcopbadcop_cra_near_field_object_phase"
    if table not in tables:
        return []

    run_group, run_name, error = _latest_run(root, "analysis/chaser_distance_runs")
    if run_group is None or run_name is None:
        diagnostics.append({"table": table, "status": "skipped", "reason": error})
        return []
    component, component_name, component_error = _latest_cra_near_field_component(run_group)
    if component is None or component_name is None:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": component_error,
            "chaser_distance_run": run_name,
        })
        return []
    required = ("chasers", "phases", "per_chaser_phase")
    missing = [name for name in required if not _has_child(component, name)]
    if missing:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": f"missing CRA near-field group(s): {missing}",
            "chaser_distance_run": run_name,
            "cra_near_field_component": component_name,
        })
        return []

    run_common, _source_refs, _parameters = _chaser_common_run_fields(run_group, run_name)
    run_path = str(run_common["chaser_distance_path"])
    component_common, near_refs, _near_parameters = _cra_near_field_common_fields(
        component=component,
        component_name=component_name,
        run_name=run_name,
        run_path=run_path,
    )
    objects = component["chasers"]
    phases = component["phases"]
    per_object = component["per_chaser_phase"]

    object_indices = _read_1d_array(objects, "chaser_index")
    object_roles = _read_canonical_behavior_labels(objects)
    raw_colors = _decode_text_column(_read_array(objects, "raw_color_hex_bytes"))
    object_role_code = _read_1d_array(objects, "behavior_class_id")

    phase_indices = _read_1d_array(phases, "phase_index")
    phase_labels = _decode_text_column(_read_array(phases, "phase_label_bytes"))
    effective_start = _read_1d_array(phases, "effective_start_frame")
    effective_end = _read_1d_array(phases, "effective_end_frame")
    total_frame_count = _read_1d_array(phases, "total_frame_count")

    approach = _read_array(per_object, "approach_percentile_mm")
    near_zone_occupancy = _read_array(per_object, "near_zone_occupancy_fraction")
    if near_zone_occupancy is None or np.asarray(near_zone_occupancy).ndim != 2:
        diagnostics.append(
            {
                "table": table,
                "status": "skipped",
                "reason": "per_chaser_phase/near_zone_occupancy_fraction missing or not 2D",
                "chaser_distance_run": run_name,
                "cra_near_field_component": component_name,
            }
        )
        return []
    near_zone_occupancy = np.asarray(near_zone_occupancy)
    n_phases, n_objects = int(near_zone_occupancy.shape[0]), int(near_zone_occupancy.shape[1])
    percentile_columns = _near_field_percentile_columns(component, approach)
    near_zone_epoch = _read_array(per_object, "near_zone_occupancy_fraction_of_epoch")
    near_zone_dwell_s = _read_array(per_object, "near_zone_dwell_s")
    near_zone_density = _read_array(per_object, "near_zone_density_per_mm2")
    near_zone_area = _read_array(per_object, "near_zone_available_area_mm2")
    entry_count = _read_array(per_object, "near_zone_entry_count")
    entry_rate = _read_array(per_object, "near_zone_entry_rate_per_min")
    visit_median_dwell = _read_array(per_object, "near_zone_visit_median_dwell_s")
    visit_total_dwell = _read_array(per_object, "near_zone_visit_total_dwell_s")
    valid_count = _read_array(per_object, "valid_distance_count")
    missing_count = _read_array(per_object, "missing_frame_count")
    dropout = _read_array(per_object, "tracking_dropout_fraction")

    rows: list[dict[str, Any]] = []
    for phase_idx in range(n_phases):
        for object_col in range(n_objects):
            object_index = _array_int(object_indices, object_col)
            object_role = object_roles[object_col] if object_col < len(object_roles) else None
            lineage = {
                "zarr_path": str(zarr_path),
                "chaser_distance_run": run_name,
                "cra_near_field_component": component_name,
                "source_component_fingerprint": component_common.get(
                    "source_component_fingerprint"
                ),
                "source_quadrant_occupancy_path": near_refs.get(
                    "source_quadrant_occupancy_path"
                ),
                "source_detection_path": run_common.get("source_detection_path"),
                "source_stimulus_run": near_refs.get("source_stimulus_run"),
                "source_stimulus_epoch_run": near_refs.get("source_stimulus_epoch_run"),
                "phase_index": phase_idx,
                "object_index": object_index,
                "object_role": object_role,
                "behavior_class": object_role,
            }
            row = _common_row(
                export_run_id=export_run_id,
                zarr_path=zarr_path,
                recording_id=recording_id,
                table=table,
                lineage=lineage,
            )
            row.update(run_common)
            row.update(component_common)
            row.update({
                "phase_axis_index": phase_idx,
                "phase_index": _array_int(phase_indices, phase_idx),
                "phase_label": phase_labels[phase_idx] if phase_idx < len(phase_labels) else None,
                "effective_start_frame": _array_int(effective_start, phase_idx),
                "effective_end_frame": _array_int(effective_end, phase_idx),
                "total_frame_count": _array_int(total_frame_count, phase_idx),
                "object_column_index": object_col,
                "object_index": object_index,
                "object_role": object_role,
                "behavior_class": object_role,
                "object_role_code": _array_int(object_role_code, object_col),
                "raw_color_hex": raw_colors[object_col] if object_col < len(raw_colors) else None,
                "near_zone_occupancy_fraction": _array_float(near_zone_occupancy, phase_idx, object_col),
                "near_zone_occupancy_fraction_of_epoch": _array_float(near_zone_epoch, phase_idx, object_col),
                "near_zone_dwell_s": _array_float(near_zone_dwell_s, phase_idx, object_col),
                "near_zone_density_per_mm2": _array_float(near_zone_density, phase_idx, object_col),
                "near_zone_available_area_mm2": _array_float(near_zone_area, phase_idx, object_col),
                "near_zone_entry_count": _array_int(entry_count, phase_idx, object_col),
                "near_zone_entry_rate_per_min": _array_float(entry_rate, phase_idx, object_col),
                "near_zone_visit_median_dwell_s": _array_float(visit_median_dwell, phase_idx, object_col),
                "near_zone_visit_total_dwell_s": _array_float(visit_total_dwell, phase_idx, object_col),
                "valid_distance_count": _array_int(valid_count, phase_idx, object_col),
                "missing_frame_count": _array_int(missing_count, phase_idx, object_col),
                "tracking_dropout_fraction": _array_float(dropout, phase_idx, object_col),
            })
            for percentile_axis, column_name, percentile_value in percentile_columns:
                row[column_name] = _array_float(approach, phase_idx, object_col, percentile_axis)
                row[f"{column_name}_percentile"] = percentile_value
            rows.append(row)
    return rows


def _load_goodcopbadcop_cra_near_field_radial_density(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    table = "goodcopbadcop_cra_near_field_radial_density"
    if table not in tables:
        return []

    run_group, run_name, error = _latest_run(root, "analysis/chaser_distance_runs")
    if run_group is None or run_name is None:
        diagnostics.append({"table": table, "status": "skipped", "reason": error})
        return []
    component, component_name, component_error = _latest_cra_near_field_component(run_group)
    if component is None or component_name is None:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": component_error,
            "chaser_distance_run": run_name,
        })
        return []
    required = ("chasers", "phases", "config", "radial_density")
    missing = [name for name in required if not _has_child(component, name)]
    if missing:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": f"missing CRA near-field radial group(s): {missing}",
            "chaser_distance_run": run_name,
            "cra_near_field_component": component_name,
        })
        return []

    run_common, _source_refs, _parameters = _chaser_common_run_fields(run_group, run_name)
    run_path = str(run_common["chaser_distance_path"])
    component_common, near_refs, _near_parameters = _cra_near_field_common_fields(
        component=component,
        component_name=component_name,
        run_name=run_name,
        run_path=run_path,
    )
    objects = component["chasers"]
    phases = component["phases"]
    config = component["config"]
    radial = component["radial_density"]

    object_indices = _read_1d_array(objects, "chaser_index")
    object_roles = _read_canonical_behavior_labels(objects)
    raw_colors = _decode_text_column(_read_array(objects, "raw_color_hex_bytes"))
    phase_indices = _read_1d_array(phases, "phase_index")
    phase_labels = _decode_text_column(_read_array(phases, "phase_label_bytes"))
    effective_start = _read_1d_array(phases, "effective_start_frame")
    effective_end = _read_1d_array(phases, "effective_end_frame")
    total_frame_count = _read_1d_array(phases, "total_frame_count")

    bin_edges = _read_1d_array(config, "radial_bin_edges_mm")
    bin_centers = _read_1d_array(config, "radial_bin_centers_mm")
    density = _read_array(radial, "radial_density_per_mm2")
    if density is None or np.asarray(density).ndim != 3:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "radial_density/radial_density_per_mm2 missing or not 3D",
            "chaser_distance_run": run_name,
            "cra_near_field_component": component_name,
        })
        return []
    density = np.asarray(density)
    n_phases, n_objects, n_bins = int(density.shape[0]), int(density.shape[1]), int(density.shape[2])
    radial_count = _read_array(radial, "radial_count")
    radial_fraction = _read_array(radial, "radial_fraction")
    radial_area = _read_array(radial, "radial_available_area_mm2")
    radial_count_wall_excluded = _read_array(radial, "radial_count_wall_excluded")
    radial_fraction_wall_excluded = _read_array(radial, "radial_fraction_wall_excluded")
    radial_density_wall_excluded = _read_array(radial, "radial_density_wall_excluded_per_mm2")
    radial_area_wall_excluded = _read_array(radial, "radial_available_area_wall_excluded_mm2")
    radial_wall_excluded_valid_count = _read_array(radial, "radial_wall_excluded_valid_count")

    rows: list[dict[str, Any]] = []
    for phase_idx in range(n_phases):
        for object_col in range(n_objects):
            object_index = _array_int(object_indices, object_col)
            object_role = object_roles[object_col] if object_col < len(object_roles) else None
            for bin_idx in range(n_bins):
                left = _array_float(bin_edges, bin_idx)
                right = _array_float(bin_edges, bin_idx + 1)
                center = _array_float(bin_centers, bin_idx)
                if center is None and left is not None and right is not None:
                    center = (float(left) + float(right)) / 2.0
                width = None if left is None or right is None else float(right) - float(left)
                lineage = {
                    "zarr_path": str(zarr_path),
                    "chaser_distance_run": run_name,
                    "cra_near_field_component": component_name,
                    "source_component_fingerprint": component_common.get(
                        "source_component_fingerprint"
                    ),
                    "source_quadrant_occupancy_path": near_refs.get(
                        "source_quadrant_occupancy_path"
                    ),
                    "source_detection_path": run_common.get("source_detection_path"),
                    "source_stimulus_run": near_refs.get("source_stimulus_run"),
                    "source_stimulus_epoch_run": near_refs.get("source_stimulus_epoch_run"),
                    "phase_index": phase_idx,
                    "object_index": object_index,
                    "object_role": object_role,
                    "behavior_class": object_role,
                    "radial_bin_index": bin_idx,
                }
                row = _common_row(
                    export_run_id=export_run_id,
                    zarr_path=zarr_path,
                    recording_id=recording_id,
                    table=table,
                    lineage=lineage,
                )
                row.update(run_common)
                row.update(component_common)
                row.update({
                    "phase_axis_index": phase_idx,
                    "phase_index": _array_int(phase_indices, phase_idx),
                    "phase_label": phase_labels[phase_idx] if phase_idx < len(phase_labels) else None,
                    "effective_start_frame": _array_int(effective_start, phase_idx),
                    "effective_end_frame": _array_int(effective_end, phase_idx),
                    "total_frame_count": _array_int(total_frame_count, phase_idx),
                    "object_column_index": object_col,
                    "object_index": object_index,
                    "object_role": object_role,
                    "behavior_class": object_role,
                    "raw_color_hex": raw_colors[object_col] if object_col < len(raw_colors) else None,
                    "radial_bin_index": bin_idx,
                    "radial_bin_left_mm": left,
                    "radial_bin_right_mm": right,
                    "radial_bin_center_mm": center,
                    "radial_bin_width_mm": width,
                    "radial_count": _array_int(radial_count, phase_idx, object_col, bin_idx),
                    "radial_fraction": _array_float(radial_fraction, phase_idx, object_col, bin_idx),
                    "radial_density_per_mm2": _array_float(density, phase_idx, object_col, bin_idx),
                    "radial_available_area_mm2": _array_float(radial_area, phase_idx, object_col, bin_idx),
                    "radial_count_wall_excluded": _array_int(radial_count_wall_excluded, phase_idx, object_col, bin_idx),
                    "radial_fraction_wall_excluded": _array_float(radial_fraction_wall_excluded, phase_idx, object_col, bin_idx),
                    "radial_density_wall_excluded_per_mm2": _array_float(radial_density_wall_excluded, phase_idx, object_col, bin_idx),
                    "radial_available_area_wall_excluded_mm2": _array_float(radial_area_wall_excluded, phase_idx, object_col, bin_idx),
                    "radial_wall_excluded_valid_count": _array_int(radial_wall_excluded_valid_count, phase_idx, object_col, bin_idx),
                })
                rows.append(row)
    return rows


def _load_goodcopbadcop_cra_near_field_distance_cdf(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    table = "goodcopbadcop_cra_near_field_distance_cdf"
    if table not in tables:
        return []

    run_group, run_name, error = _latest_run(root, "analysis/chaser_distance_runs")
    if run_group is None or run_name is None:
        diagnostics.append({"table": table, "status": "skipped", "reason": error})
        return []
    component, component_name, component_error = _latest_cra_near_field_component(run_group)
    if component is None or component_name is None:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": component_error,
            "chaser_distance_run": run_name,
        })
        return []
    required = ("chasers", "phases", "config", "distance_cdf")
    missing = [name for name in required if not _has_child(component, name)]
    if missing:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": f"missing CRA near-field CDF group(s): {missing}",
            "chaser_distance_run": run_name,
            "cra_near_field_component": component_name,
        })
        return []

    run_common, _source_refs, _parameters = _chaser_common_run_fields(run_group, run_name)
    run_path = str(run_common["chaser_distance_path"])
    component_common, near_refs, _near_parameters = _cra_near_field_common_fields(
        component=component,
        component_name=component_name,
        run_name=run_name,
        run_path=run_path,
    )
    objects = component["chasers"]
    phases = component["phases"]
    config = component["config"]
    cdf_group = component["distance_cdf"]

    object_indices = _read_1d_array(objects, "chaser_index")
    object_roles = _read_canonical_behavior_labels(objects)
    raw_colors = _decode_text_column(_read_array(objects, "raw_color_hex_bytes"))
    phase_indices = _read_1d_array(phases, "phase_index")
    phase_labels = _decode_text_column(_read_array(phases, "phase_label_bytes"))
    effective_start = _read_1d_array(phases, "effective_start_frame")
    effective_end = _read_1d_array(phases, "effective_end_frame")
    total_frame_count = _read_1d_array(phases, "total_frame_count")

    thresholds = _read_1d_array(config, "cdf_thresholds_mm")
    cdf = _read_array(cdf_group, "cdf_fraction")
    if cdf is None or np.asarray(cdf).ndim != 3:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "distance_cdf/cdf_fraction missing or not 3D",
            "chaser_distance_run": run_name,
            "cra_near_field_component": component_name,
        })
        return []
    cdf = np.asarray(cdf)
    n_phases, n_objects, n_thresholds = int(cdf.shape[0]), int(cdf.shape[1]), int(cdf.shape[2])

    rows: list[dict[str, Any]] = []
    for phase_idx in range(n_phases):
        for object_col in range(n_objects):
            object_index = _array_int(object_indices, object_col)
            object_role = object_roles[object_col] if object_col < len(object_roles) else None
            for threshold_idx in range(n_thresholds):
                threshold = _array_float(thresholds, threshold_idx)
                lineage = {
                    "zarr_path": str(zarr_path),
                    "chaser_distance_run": run_name,
                    "cra_near_field_component": component_name,
                    "source_component_fingerprint": component_common.get(
                        "source_component_fingerprint"
                    ),
                    "source_quadrant_occupancy_path": near_refs.get(
                        "source_quadrant_occupancy_path"
                    ),
                    "source_detection_path": run_common.get("source_detection_path"),
                    "source_stimulus_run": near_refs.get("source_stimulus_run"),
                    "source_stimulus_epoch_run": near_refs.get("source_stimulus_epoch_run"),
                    "phase_index": phase_idx,
                    "object_index": object_index,
                    "object_role": object_role,
                    "behavior_class": object_role,
                    "cdf_threshold_index": threshold_idx,
                }
                row = _common_row(
                    export_run_id=export_run_id,
                    zarr_path=zarr_path,
                    recording_id=recording_id,
                    table=table,
                    lineage=lineage,
                )
                row.update(run_common)
                row.update(component_common)
                row.update({
                    "phase_axis_index": phase_idx,
                    "phase_index": _array_int(phase_indices, phase_idx),
                    "phase_label": phase_labels[phase_idx] if phase_idx < len(phase_labels) else None,
                    "effective_start_frame": _array_int(effective_start, phase_idx),
                    "effective_end_frame": _array_int(effective_end, phase_idx),
                    "total_frame_count": _array_int(total_frame_count, phase_idx),
                    "object_column_index": object_col,
                    "object_index": object_index,
                    "object_role": object_role,
                    "behavior_class": object_role,
                    "raw_color_hex": raw_colors[object_col] if object_col < len(raw_colors) else None,
                    "cdf_threshold_index": threshold_idx,
                    "distance_threshold_mm": threshold,
                    "cdf_fraction": _array_float(cdf, phase_idx, object_col, threshold_idx),
                })
                rows.append(row)
    return rows


def _load_goodcopbadcop_egocentric_epoch_summary(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    table = "goodcopbadcop_egocentric_epoch_summary"
    if table not in tables:
        return []

    run_group, run_name, error = _latest_run(root, "analysis/chaser_distance_runs")
    if run_group is None or run_name is None:
        diagnostics.append({"table": table, "status": "skipped", "reason": error})
        return []
    component, component_name, component_error = _latest_egocentric_component(run_group)
    if component is None or component_name is None:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": component_error,
            "chaser_distance_run": run_name,
        })
        return []
    if not _has_child(component, "epoch_summary"):
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "missing egocentric epoch_summary group",
            "chaser_distance_run": run_name,
            "egocentric_component_name": component_name,
        })
        return []

    run_common, source_refs, _parameters = _chaser_common_run_fields(run_group, run_name)
    run_path = str(run_common["chaser_distance_path"])
    component_common, egocentric_refs, _egocentric_parameters = _egocentric_common_fields(
        component=component,
        component_name=component_name,
        run_name=run_name,
        run_path=run_path,
    )
    summary = component["epoch_summary"]
    valid_frame_count = _read_array(summary, "valid_frame_count")
    if valid_frame_count is None or np.asarray(valid_frame_count).ndim != 2:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "egocentric epoch_summary/valid_frame_count missing or not 2D",
            "chaser_distance_run": run_name,
            "egocentric_component_name": component_name,
        })
        return []

    valid_frame_count = np.asarray(valid_frame_count)
    n_windows, n_chasers = int(valid_frame_count.shape[0]), int(valid_frame_count.shape[1])
    chaser_indices = _egocentric_chaser_indices(component, fallback_count=n_chasers)
    chaser_behaviors = _chaser_behaviors_for_run(run_group, chaser_indices)
    windows = _epoch_summary_window_rows(component)
    fps = _safe_float(run_common.get("fps"))
    circular_mean = _read_array(summary, "circular_mean_bearing_deg")
    resultant = _read_array(summary, "circular_resultant_length")
    mean_alignment = _read_array(summary, "mean_alignment_cos")
    mean_lateral = _read_array(summary, "mean_lateral_sin")
    fraction_front = _read_array(summary, "fraction_front_45")
    fraction_lateral = _read_array(summary, "fraction_lateral_45")
    fraction_behind = _read_array(summary, "fraction_behind_45")
    summary_attrs = _attrs_dict(summary)

    rows: list[dict[str, Any]] = []
    for window_index in range(n_windows):
        raw_window = (
            windows[window_index]
            if window_index < len(windows)
            else {
                "window_index": window_index,
                "window_id": window_index,
                "window_label": None,
                "start_frame": None,
                "end_frame": None,
                "start_time_s": None,
                "end_time_s": None,
                "duration_s": None,
            }
        )
        window = _fill_chaser_window_times(raw_window, fps=fps)
        for chaser_column_index in range(n_chasers):
            chaser_index = (
                chaser_indices[chaser_column_index]
                if chaser_column_index < len(chaser_indices)
                else chaser_column_index
            )
            behavior_class_id, behavior_class = chaser_behaviors[chaser_column_index]
            lineage = {
                "zarr_path": str(zarr_path),
                "chaser_distance_run": run_name,
                "egocentric_component_name": component_name,
                "source_detection_path": source_refs.get("source_detection_path"),
                "source_stimulus_run": source_refs.get("source_stimulus_run"),
                "source_stimulus_epoch_run": source_refs.get("source_stimulus_epoch_run"),
                "source_track_kinematics_run": egocentric_refs.get("source_track_kinematics_run"),
                "source_track_kinematics_track_id": egocentric_refs.get("source_track_kinematics_track_id"),
                "source_heading_array": egocentric_refs.get("source_heading_array"),
                "window_id": window.get("window_id"),
                "chaser_index": chaser_index,
            }
            row = _common_row(
                export_run_id=export_run_id,
                zarr_path=zarr_path,
                recording_id=recording_id,
                table=table,
                lineage=lineage,
            )
            row.update(run_common)
            row.update(component_common)
            row.update({
                "window_index": _safe_int(window.get("window_index")),
                "window_id": _safe_int(window.get("window_id")),
                "window_label": window.get("window_label"),
                "start_frame": _safe_int(window.get("start_frame")),
                "end_frame": _safe_int(window.get("end_frame")),
                "start_time_s": _safe_float(window.get("start_time_s")),
                "end_time_s": _safe_float(window.get("end_time_s")),
                "duration_s": _safe_float(window.get("duration_s")),
                "chaser_column_index": chaser_column_index,
                "chaser_index": _safe_int(chaser_index),
                "behavior_class_id": behavior_class_id,
                "behavior_class": behavior_class,
                "valid_frame_count": _array_int(valid_frame_count, window_index, chaser_column_index),
                "circular_mean_bearing_deg": _array_float(circular_mean, window_index, chaser_column_index),
                "circular_resultant_length": _array_float(resultant, window_index, chaser_column_index),
                "mean_alignment_cos": _array_float(mean_alignment, window_index, chaser_column_index),
                "mean_lateral_sin": _array_float(mean_lateral, window_index, chaser_column_index),
                "fraction_front_45": _array_float(fraction_front, window_index, chaser_column_index),
                "fraction_lateral_45": _array_float(fraction_lateral, window_index, chaser_column_index),
                "fraction_behind_45": _array_float(fraction_behind, window_index, chaser_column_index),
                "front_definition": summary_attrs.get("front_definition"),
                "lateral_definition": summary_attrs.get("lateral_definition"),
                "behind_definition": summary_attrs.get("behind_definition"),
            })
            rows.append(row)
    return rows


def _load_goodcopbadcop_egocentric_distance_bearing_histogram(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    table = "goodcopbadcop_egocentric_distance_bearing_histogram"
    if table not in tables:
        return []

    run_group, run_name, error = _latest_run(root, "analysis/chaser_distance_runs")
    if run_group is None or run_name is None:
        diagnostics.append({"table": table, "status": "skipped", "reason": error})
        return []
    component, component_name, component_error = _latest_egocentric_component(run_group)
    if component is None or component_name is None:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": component_error,
            "chaser_distance_run": run_name,
        })
        return []
    if not _has_child(component, "distance_bearing_histogram"):
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "missing distance_bearing_histogram group",
            "chaser_distance_run": run_name,
            "egocentric_component_name": component_name,
        })
        return []

    run_common, source_refs, _parameters = _chaser_common_run_fields(run_group, run_name)
    run_path = str(run_common["chaser_distance_path"])
    component_common, egocentric_refs, _egocentric_parameters = _egocentric_common_fields(
        component=component,
        component_name=component_name,
        run_name=run_name,
        run_path=run_path,
    )
    hist = component["distance_bearing_histogram"]
    hist_counts = _read_array(hist, "hist_counts")
    if hist_counts is None or np.asarray(hist_counts).ndim != 4:
        diagnostics.append({
            "table": table,
            "status": "skipped",
            "reason": "distance_bearing_histogram/hist_counts missing or not 4D",
            "chaser_distance_run": run_name,
            "egocentric_component_name": component_name,
        })
        return []

    hist_counts = np.asarray(hist_counts)
    n_windows, n_chasers, n_distance_bins, n_bearing_bins = (
        int(hist_counts.shape[0]),
        int(hist_counts.shape[1]),
        int(hist_counts.shape[2]),
        int(hist_counts.shape[3]),
    )
    hist_probability = _read_array(hist, "hist_probability")
    valid_frame_count = None
    if _has_child(component, "epoch_summary"):
        valid_frame_count = _read_array(component["epoch_summary"], "valid_frame_count")
    chaser_indices = _egocentric_chaser_indices(component, fallback_count=n_chasers)
    chaser_behaviors = _chaser_behaviors_for_run(run_group, chaser_indices)
    windows = _epoch_summary_window_rows(component)
    fps = _safe_float(run_common.get("fps"))
    distance_edges = _read_1d_array(hist, "distance_bin_edges_mm")
    distance_centers = _read_1d_array(hist, "distance_bin_centers_mm")
    bearing_edges = _read_1d_array(hist, "bearing_bin_edges_deg")
    bearing_centers = _read_1d_array(hist, "bearing_bin_centers_deg")
    hist_attrs = _attrs_dict(hist)
    distance_bin_width = _safe_float(component_common.get("distance_bin_width_mm"))
    bearing_bin_width = _safe_float(component_common.get("bearing_bin_width_deg"))

    rows: list[dict[str, Any]] = []
    for window_index in range(n_windows):
        raw_window = (
            windows[window_index]
            if window_index < len(windows)
            else {
                "window_index": window_index,
                "window_id": window_index,
                "window_label": None,
                "start_frame": None,
                "end_frame": None,
                "start_time_s": None,
                "end_time_s": None,
                "duration_s": None,
            }
        )
        window = _fill_chaser_window_times(raw_window, fps=fps)
        for chaser_column_index in range(n_chasers):
            chaser_index = (
                chaser_indices[chaser_column_index]
                if chaser_column_index < len(chaser_indices)
                else chaser_column_index
            )
            behavior_class_id, behavior_class = chaser_behaviors[chaser_column_index]
            valid_count = _array_int(valid_frame_count, window_index, chaser_column_index)
            for distance_bin_index in range(n_distance_bins):
                distance_left = _array_float(distance_edges, distance_bin_index)
                distance_right = _array_float(distance_edges, distance_bin_index + 1)
                distance_center = _array_float(distance_centers, distance_bin_index)
                if distance_center is None and distance_left is not None and distance_right is not None:
                    distance_center = (distance_left + distance_right) / 2.0
                effective_distance_width = distance_bin_width
                if effective_distance_width is None and distance_left is not None and distance_right is not None:
                    effective_distance_width = distance_right - distance_left
                for bearing_bin_index in range(n_bearing_bins):
                    bearing_left = _array_float(bearing_edges, bearing_bin_index)
                    bearing_right = _array_float(bearing_edges, bearing_bin_index + 1)
                    bearing_center = _array_float(bearing_centers, bearing_bin_index)
                    if bearing_center is None and bearing_left is not None and bearing_right is not None:
                        bearing_center = (bearing_left + bearing_right) / 2.0
                    effective_bearing_width = bearing_bin_width
                    if effective_bearing_width is None and bearing_left is not None and bearing_right is not None:
                        effective_bearing_width = bearing_right - bearing_left
                    lineage = {
                        "zarr_path": str(zarr_path),
                        "chaser_distance_run": run_name,
                        "egocentric_component_name": component_name,
                        "source_detection_path": source_refs.get("source_detection_path"),
                        "source_stimulus_run": source_refs.get("source_stimulus_run"),
                        "source_stimulus_epoch_run": source_refs.get("source_stimulus_epoch_run"),
                        "source_track_kinematics_run": egocentric_refs.get("source_track_kinematics_run"),
                        "source_track_kinematics_track_id": egocentric_refs.get("source_track_kinematics_track_id"),
                        "source_heading_array": egocentric_refs.get("source_heading_array"),
                        "window_id": window.get("window_id"),
                        "chaser_index": chaser_index,
                        "distance_bin_index": distance_bin_index,
                        "bearing_bin_index": bearing_bin_index,
                    }
                    row = _common_row(
                        export_run_id=export_run_id,
                        zarr_path=zarr_path,
                        recording_id=recording_id,
                        table=table,
                        lineage=lineage,
                    )
                    row.update(run_common)
                    row.update(component_common)
                    row.update({
                        "window_index": _safe_int(window.get("window_index")),
                        "window_id": _safe_int(window.get("window_id")),
                        "window_label": window.get("window_label"),
                        "start_frame": _safe_int(window.get("start_frame")),
                        "end_frame": _safe_int(window.get("end_frame")),
                        "start_time_s": _safe_float(window.get("start_time_s")),
                        "end_time_s": _safe_float(window.get("end_time_s")),
                        "duration_s": _safe_float(window.get("duration_s")),
                        "chaser_column_index": chaser_column_index,
                        "chaser_index": _safe_int(chaser_index),
                        "behavior_class_id": behavior_class_id,
                        "behavior_class": behavior_class,
                        "distance_bin_index": distance_bin_index,
                        "distance_bin_left_mm": distance_left,
                        "distance_bin_right_mm": distance_right,
                        "distance_bin_center_mm": distance_center,
                        "distance_bin_width_mm": effective_distance_width,
                        "bearing_bin_index": bearing_bin_index,
                        "bearing_bin_left_deg": bearing_left,
                        "bearing_bin_right_deg": bearing_right,
                        "bearing_bin_center_deg": bearing_center,
                        "bearing_bin_width_deg": effective_bearing_width,
                        "hist_count": _array_int(
                            hist_counts,
                            window_index,
                            chaser_column_index,
                            distance_bin_index,
                            bearing_bin_index,
                        ),
                        "hist_probability": _array_float(
                            hist_probability,
                            window_index,
                            chaser_column_index,
                            distance_bin_index,
                            bearing_bin_index,
                        ),
                        "valid_sample_count": valid_count,
                        "probability_normalization": hist_attrs.get("probability_normalization"),
                    })
                    rows.append(row)
    return rows


def export_one_zarr(
    zarr_path: str | Path,
    *,
    tables: Sequence[str],
    export_run_id: str,
    baseline_time_bin_s: float = 5.0,
    baseline_sample_rate_hz: float = 10.0,
    baseline_full_resolution_samples: bool = False,
    baseline_spatial_grid_size: int = 12,
    legacy_compatibility: bool = False,
) -> SourceExportResult:
    zarr_path = Path(zarr_path).expanduser().resolve()
    recording_id = _recording_id_from_path(zarr_path)
    result = SourceExportResult(zarr_path=str(zarr_path), recording_id=recording_id)
    requested_table_set = set(tables)
    table_set = {_SOURCE_TABLE_BY_V2.get(table, table) for table in requested_table_set}

    try:
        root = open_zarr_root(zarr_path, mode="r")
    except Exception as exc:
        result.diagnostics.append({"table": "*", "status": "failed", "reason": f"open_failed: {exc}"})
        return result

    stimulus_run, steps, step_rows, protocol_signature = _load_stimulus_steps(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if step_rows:
        result.rows_by_table.setdefault("stimulus_steps", []).extend(step_rows)

    recording_rows = _load_recording_summary(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        stimulus_run=stimulus_run,
        protocol_signature=protocol_signature,
        step_count=len(steps),
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if recording_rows:
        result.rows_by_table.setdefault("recording_summary", []).extend(recording_rows)

    step_summary_rows, response_rows = _load_stimulus_response_tables(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        protocol_signature=protocol_signature,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if step_summary_rows:
        result.rows_by_table.setdefault("stimulus_step_summary", []).extend(step_summary_rows)
    if response_rows:
        result.rows_by_table.setdefault("stimulus_response_per_fish_step", []).extend(response_rows)

    bout_rows = _load_swim_bout_metrics(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        stimulus_run=stimulus_run,
        protocol_signature=protocol_signature,
        steps=steps,
        tables=table_set,
        diagnostics=result.diagnostics,
        legacy_compatibility=legacy_compatibility,
    )
    if bout_rows:
        result.rows_by_table.setdefault("swim_bout_metrics", []).extend(bout_rows)

    bout_kinematics_rows = _load_bout_kinematics_metrics(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        stimulus_run=stimulus_run,
        protocol_signature=protocol_signature,
        steps=steps,
        tables=table_set,
        diagnostics=result.diagnostics,
        legacy_compatibility=legacy_compatibility,
    )
    if bout_kinematics_rows:
        result.rows_by_table.setdefault("bout_kinematics_metrics", []).extend(bout_kinematics_rows)

    position_occupancy_rows = _load_position_occupancy_histogram_2d(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if position_occupancy_rows:
        result.rows_by_table.setdefault(
            POSITION_OCCUPANCY_HISTOGRAM_TABLE,
            [],
        ).extend(position_occupancy_rows)

    baseline_rows_by_table = _load_baseline_tables(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
        time_bin_s=baseline_time_bin_s,
        sample_rate_hz=baseline_sample_rate_hz,
        full_resolution_samples=baseline_full_resolution_samples,
        spatial_grid_size=baseline_spatial_grid_size,
    )
    for baseline_table, baseline_rows in baseline_rows_by_table.items():
        if baseline_rows:
            result.rows_by_table.setdefault(baseline_table, []).extend(baseline_rows)

    # Canonical chaser selection is verified once per source.  Until each
    # persisted summary/component publishes its own sealed semantic authority,
    # remove those requests before the legacy extraction helpers can navigate
    # the selected child as raw Zarr.  Each requested table receives an explicit
    # unavailable diagnostic above.
    table_set = _preflight_unsealed_chaser_exports(
        root,
        tables=table_set,
        diagnostics=result.diagnostics,
    )

    goodcopbadcop_spatial_rows = _load_goodcopbadcop_spatial_occupancy_zones(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if goodcopbadcop_spatial_rows:
        result.rows_by_table.setdefault("goodcopbadcop_spatial_occupancy_zones", []).extend(
            goodcopbadcop_spatial_rows
        )

    goodcopbadcop_chaser_summary_rows = _load_goodcopbadcop_chaser_epoch_summary(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if goodcopbadcop_chaser_summary_rows:
        result.rows_by_table.setdefault("goodcopbadcop_chaser_epoch_summary", []).extend(
            goodcopbadcop_chaser_summary_rows
        )

    goodcopbadcop_epoch_behavior_rows = _load_goodcopbadcop_epoch_behavior_summary(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if goodcopbadcop_epoch_behavior_rows:
        result.rows_by_table.setdefault("goodcopbadcop_epoch_behavior_summary", []).extend(
            goodcopbadcop_epoch_behavior_rows
        )

    goodcopbadcop_epoch_bout_distribution_rows = _load_goodcopbadcop_epoch_bout_distribution(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if goodcopbadcop_epoch_bout_distribution_rows:
        result.rows_by_table.setdefault("goodcopbadcop_epoch_bout_distribution", []).extend(
            goodcopbadcop_epoch_bout_distribution_rows
        )

    goodcopbadcop_epoch_bout_histogram_rows = _load_goodcopbadcop_epoch_behavior_structured_histogram(
        root,
        table="goodcopbadcop_epoch_bout_histogram",
        dataset_name="per_epoch_bout_histograms",
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if goodcopbadcop_epoch_bout_histogram_rows:
        result.rows_by_table.setdefault("goodcopbadcop_epoch_bout_histogram", []).extend(
            goodcopbadcop_epoch_bout_histogram_rows
        )

    goodcopbadcop_epoch_inter_bout_interval_histogram_rows = (
        _load_goodcopbadcop_epoch_behavior_structured_histogram(
            root,
            table="goodcopbadcop_epoch_inter_bout_interval_histogram",
            dataset_name="per_epoch_inter_bout_interval_histograms",
            export_run_id=export_run_id,
            zarr_path=zarr_path,
            recording_id=recording_id,
            tables=table_set,
            diagnostics=result.diagnostics,
        )
    )
    if goodcopbadcop_epoch_inter_bout_interval_histogram_rows:
        result.rows_by_table.setdefault("goodcopbadcop_epoch_inter_bout_interval_histogram", []).extend(
            goodcopbadcop_epoch_inter_bout_interval_histogram_rows
        )

    goodcopbadcop_epoch_center_distance_histogram_rows = _load_goodcopbadcop_epoch_center_distance_histogram(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if goodcopbadcop_epoch_center_distance_histogram_rows:
        result.rows_by_table.setdefault("goodcopbadcop_epoch_center_distance_histogram", []).extend(
            goodcopbadcop_epoch_center_distance_histogram_rows
        )

    goodcopbadcop_epoch_speed_rows = _load_goodcopbadcop_epoch_speed_summary(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if goodcopbadcop_epoch_speed_rows:
        result.rows_by_table.setdefault("goodcopbadcop_epoch_speed_summary", []).extend(
            goodcopbadcop_epoch_speed_rows
        )

    goodcopbadcop_speed_distance_rows = _load_goodcopbadcop_speed_distance_bins(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if goodcopbadcop_speed_distance_rows:
        result.rows_by_table.setdefault("goodcopbadcop_speed_distance_bins", []).extend(
            goodcopbadcop_speed_distance_rows
        )

    goodcopbadcop_chaser_histogram_rows = _load_goodcopbadcop_chaser_distance_histogram(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if goodcopbadcop_chaser_histogram_rows:
        result.rows_by_table.setdefault("goodcopbadcop_chaser_distance_histogram", []).extend(
            goodcopbadcop_chaser_histogram_rows
        )

    goodcopbadcop_cra_summary_rows = _load_goodcopbadcop_cra_primary_endpoint_summary(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if goodcopbadcop_cra_summary_rows:
        result.rows_by_table.setdefault("goodcopbadcop_cra_primary_endpoint_summary", []).extend(
            goodcopbadcop_cra_summary_rows
        )

    goodcopbadcop_cra_object_phase_rows = _load_goodcopbadcop_cra_primary_endpoint_object_phase(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if goodcopbadcop_cra_object_phase_rows:
        result.rows_by_table.setdefault("goodcopbadcop_cra_primary_endpoint_object_phase", []).extend(
            goodcopbadcop_cra_object_phase_rows
        )

    goodcopbadcop_cra_quadrant_rows = _load_goodcopbadcop_cra_quadrant_occupancy(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if goodcopbadcop_cra_quadrant_rows:
        result.rows_by_table.setdefault("goodcopbadcop_cra_quadrant_occupancy", []).extend(
            goodcopbadcop_cra_quadrant_rows
        )

    goodcopbadcop_cra_near_field_summary_rows = _load_goodcopbadcop_cra_near_field_summary(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if goodcopbadcop_cra_near_field_summary_rows:
        result.rows_by_table.setdefault("goodcopbadcop_cra_near_field_summary", []).extend(
            goodcopbadcop_cra_near_field_summary_rows
        )

    goodcopbadcop_cra_near_field_object_phase_rows = _load_goodcopbadcop_cra_near_field_object_phase(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if goodcopbadcop_cra_near_field_object_phase_rows:
        result.rows_by_table.setdefault("goodcopbadcop_cra_near_field_object_phase", []).extend(
            goodcopbadcop_cra_near_field_object_phase_rows
        )

    goodcopbadcop_cra_near_field_radial_rows = _load_goodcopbadcop_cra_near_field_radial_density(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if goodcopbadcop_cra_near_field_radial_rows:
        result.rows_by_table.setdefault("goodcopbadcop_cra_near_field_radial_density", []).extend(
            goodcopbadcop_cra_near_field_radial_rows
        )

    goodcopbadcop_cra_near_field_cdf_rows = _load_goodcopbadcop_cra_near_field_distance_cdf(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if goodcopbadcop_cra_near_field_cdf_rows:
        result.rows_by_table.setdefault("goodcopbadcop_cra_near_field_distance_cdf", []).extend(
            goodcopbadcop_cra_near_field_cdf_rows
        )

    goodcopbadcop_egocentric_summary_rows = _load_goodcopbadcop_egocentric_epoch_summary(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if goodcopbadcop_egocentric_summary_rows:
        result.rows_by_table.setdefault("goodcopbadcop_egocentric_epoch_summary", []).extend(
            goodcopbadcop_egocentric_summary_rows
        )

    goodcopbadcop_egocentric_histogram_rows = _load_goodcopbadcop_egocentric_distance_bearing_histogram(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if goodcopbadcop_egocentric_histogram_rows:
        result.rows_by_table.setdefault("goodcopbadcop_egocentric_distance_bearing_histogram", []).extend(
            goodcopbadcop_egocentric_histogram_rows
        )

    canonical_rows: dict[str, list[dict[str, Any]]] = {}
    for source_table, rows in result.rows_by_table.items():
        table = _V2_TABLE_BY_SOURCE.get(source_table, source_table)
        if table not in requested_table_set:
            continue
        canonical_rows[table] = [canonicalize_export_row(table, row) for row in rows]
    for table in requested_table_set:
        canonical_rows.setdefault(table, [])
    result.rows_by_table = canonical_rows
    for diagnostic in result.diagnostics:
        source_table = diagnostic.get("table")
        if isinstance(source_table, str):
            diagnostic["table"] = _V2_TABLE_BY_SOURCE.get(source_table, source_table)
    return result


def discover_analysis_zarrs(recordings_root: Path) -> list[Path]:
    recordings_root = recordings_root.expanduser().resolve()
    if not recordings_root.exists():
        raise FileNotFoundError(f"recordings root does not exist: {recordings_root}")
    return sorted(path for path in recordings_root.rglob("*_analysis.zarr") if path.is_dir())


def _parse_tables(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_TABLES
    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",")]
    else:
        raw = []
        for item in value:
            raw.extend(part.strip() for part in str(item).split(","))
    tables = tuple(item for item in raw if item)
    trace_tables = sorted(set(tables) & set(TRACE_TABLES))
    if trace_tables:
        raise ValueError(
            "Framewise trace table(s) require their bounded streaming exporter: "
            + ", ".join(trace_tables)
        )
    unknown = sorted(set(tables) - set(AVAILABLE_TABLES))
    if unknown:
        expected = ", ".join(AVAILABLE_TABLES)
        raise ValueError(f"Unknown table(s): {', '.join(unknown)}. Expected subset of: {expected}")
    return tables or DEFAULT_TABLES


def _collect_sources(args: argparse.Namespace) -> list[Path]:
    if args.collection_manifest is not None and (args.zarr or args.recordings_root is not None):
        raise ValueError("--collection-manifest cannot be combined with --zarr or --recordings-root")

    sources: list[Path] = []
    if args.collection_manifest is not None:
        sources.extend(_source_paths_from_collection_manifest(Path(args.collection_manifest)))
    for path in args.zarr or []:
        sources.append(Path(path).expanduser().resolve())
    if args.recordings_root is not None:
        sources.extend(discover_analysis_zarrs(Path(args.recordings_root)))
    deduped: dict[str, Path] = {}
    for path in sources:
        deduped[str(path)] = path
    out = sorted(deduped.values())
    if args.limit is not None:
        out = out[: int(args.limit)]
    return out


def _normalize_rows(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> list[dict[str, Any]]:
    return [{column: row.get(column) for column in columns} for row in rows]


def _arrow_footer_metadata(*, table: str) -> dict[bytes, bytes]:
    return {
        b"palette.export_schema_id": EXPORT_SCHEMA_ID.encode("utf-8"),
        b"palette.export_schema_version": str(EXPORT_SCHEMA_VERSION).encode("utf-8"),
        b"palette.table_contract": json.dumps(
            TABLE_CONTRACTS[table].to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    }


def _build_arrow_schema(rows: Sequence[Mapping[str, Any]], *, table: str):
    import pyarrow as pa

    metadata = _arrow_footer_metadata(table=table)
    exact_contract = ARROW_TABLE_CONTRACTS.get(table)
    if exact_contract is not None:
        expected_names = tuple(field.name for field in exact_contract.fields)
        expected_set = set(expected_names)
        unexpected = sorted({key for row in rows for key in row} - expected_set)
        if unexpected:
            raise ValueError(f"{table}: unexpected fields for exact Arrow schema: {unexpected}")
        nonnullable = {
            field.name for field in exact_contract.fields if not field.nullable
        }
        for row_index, row in enumerate(rows):
            missing = sorted(name for name in nonnullable if row.get(name) is None)
            if missing:
                raise ValueError(
                    f"{table}: row {row_index} has null/missing non-nullable fields: {missing}"
                )
        return exact_arrow_schema(table, metadata=metadata)

    schema = pa.Table.from_pylist([dict(row) for row in rows]).schema
    return schema.with_metadata(
        {
            **metadata,
            b"palette.arrow_schema_mode": b"inferred_v2_compatibility",
        }
    )


def _write_table_parts(
    *,
    generation_root: Path,
    table: str,
    rows_by_source: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> tuple[int, list[str]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table_dir = generation_root / "tables" / table
    table_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[Mapping[str, Any]] = []
    for _source, rows in rows_by_source:
        all_rows.extend(rows)
    if not all_rows:
        return 0, []

    exact_contract = ARROW_TABLE_CONTRACTS.get(table)
    if exact_contract is not None:
        exact_names = {field.name for field in exact_contract.fields}
        unexpected = sorted({key for row in all_rows for key in row} - exact_names)
        if unexpected:
            raise ValueError(
                f"{table}: unexpected fields for exact Arrow schema: {unexpected}"
            )
    primary_key_first = table in {
        STIMULUS_RESPONSE_TABLE,
        SWIM_BOUT_METRICS_TABLE,
        BOUT_KINEMATICS_METRICS_TABLE,
    }
    if primary_key_first:
        # Preserve the established diagnostic contract for the first three
        # exact scientific tables, whose callers distinguish malformed keys
        # from other missing required fields.
        _validate_exact_primary_keys(table, all_rows)
    columns = (
        [field.name for field in exact_contract.fields]
        if exact_contract is not None
        else sorted({key for row in all_rows for key in row.keys()})
    )
    missing = validate_table_columns(table, columns)
    if missing:
        raise ValueError(
            f"{table} does not satisfy its V2 table contract; missing columns: {list(missing)}"
        )
    normalized_all = _normalize_rows(all_rows, columns)
    schema = _build_arrow_schema(normalized_all, table=table)
    if exact_contract is not None and not primary_key_first:
        # Field-set and nullability errors are more fundamental than key
        # uniqueness and retain their precise fail-closed diagnostics.  Once
        # the row shape is proven exact, every persisted key must be complete
        # and unique across all source parts.
        _validate_exact_primary_keys(table, normalized_all)

    row_count = 0
    part_paths: list[str] = []
    part_index = 0
    for source_name, rows in rows_by_source:
        if not rows:
            continue
        part_rows = _normalize_rows(rows, columns)
        arrow_table = pa.Table.from_pylist(part_rows, schema=schema)
        source_hash = hashlib.sha1(source_name.encode("utf-8")).hexdigest()[:10]
        part_path = table_dir / f"part-{part_index:05d}-{source_hash}.parquet"
        tmp_path = table_dir / f".{part_path.name}.tmp"
        if tmp_path.exists():
            tmp_path.unlink()
        pq.write_table(arrow_table, tmp_path)
        os.replace(tmp_path, part_path)
        row_count += len(rows)
        part_paths.append(str(part_path))
        part_index += 1

    return row_count, part_paths


def _validate_exact_primary_keys(
    table: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    primary_key = TABLE_CONTRACTS[table].primary_key
    seen: dict[tuple[Any, ...], int] = {}
    for row_index, row in enumerate(rows):
        key = tuple(row.get(name) for name in primary_key)
        if any(value is None for value in key):
            raise ValueError(
                f"{table}: row {row_index} has a null/missing primary key {primary_key!r}"
            )
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(
                f"{table}: duplicate primary key {key!r} at rows "
                f"{previous} and {row_index}"
            )
        seen[key] = row_index


def _staged_part_inventory(
    *,
    staging_root: Path,
    final_generation_path: Path,
    tables: Sequence[str],
) -> tuple[
    dict[str, list[str]],
    dict[str, list[dict[str, Any]]],
]:
    """Build the exact manifest-exclusive staged inventory."""

    import pyarrow.parquet as pq

    logical_parts: dict[str, list[str]] = {}
    inventory: dict[str, list[dict[str, Any]]] = {}
    for table in tables:
        table_root = staging_root / "tables" / table
        staged_parts = tuple(sorted(table_root.glob("*.parquet")))
        entries: list[dict[str, Any]] = []
        paths: list[str] = []
        for staged_part in staged_parts:
            relative = final_generation_path / "tables" / table / staged_part.name
            parquet_file = pq.ParquetFile(staged_part)
            row_count = int(parquet_file.metadata.num_rows)
            entry = {
                "path": relative.as_posix(),
                "sha256": sha256_file(staged_part),
                "size_bytes": int(staged_part.stat().st_size),
                "row_count": row_count,
            }
            entries.append(entry)
            paths.append(relative.as_posix())
        logical_parts[table] = paths
        inventory[table] = entries
    return logical_parts, inventory


def export_sources(
    zarr_paths: Sequence[Path],
    *,
    output_root: Path,
    tables: Sequence[str] = DEFAULT_TABLES,
    jobs: int = 1,
    export_run_id: str | None = None,
    overwrite: bool = False,
    collection_manifest_path: Path | None = None,
    baseline_time_bin_s: float = 5.0,
    baseline_sample_rate_hz: float = 10.0,
    baseline_full_resolution_samples: bool = False,
    baseline_spatial_grid_size: int = 12,
    legacy_compatibility: bool = False,
) -> dict[str, Any]:
    tables = _parse_tables(tables)
    output_root = Path(output_root).expanduser().resolve()
    export_run_id = safe_component(
        export_run_id or "run_" + utc_now_compact(),
        label="export run ID",
    )
    zarr_paths = [Path(path).expanduser().resolve() for path in zarr_paths]
    if not zarr_paths:
        raise ValueError("No analysis Zarr sources were provided or discovered.")
    if not math.isfinite(float(baseline_time_bin_s)) or float(baseline_time_bin_s) <= 0:
        raise ValueError("baseline_time_bin_s must be positive and finite")
    if (
        not baseline_full_resolution_samples
        and (
            not math.isfinite(float(baseline_sample_rate_hz))
            or float(baseline_sample_rate_hz) <= 0
        )
    ):
        raise ValueError("baseline_sample_rate_hz must be positive and finite")
    if int(baseline_spatial_grid_size) < 2:
        raise ValueError("baseline_spatial_grid_size must be at least 2")
    manifest_path = export_manifest_path(output_root, export_run_id)
    baseline_manifest_identity = manifest_identity(manifest_path)
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Export manifest already exists: {manifest_path}")
    collection_summary = (
        _collection_manifest_summary(Path(collection_manifest_path))
        if collection_manifest_path is not None
        else None
    )

    results: list[SourceExportResult] = []
    if jobs <= 1:
        for path in zarr_paths:
            results.append(
                export_one_zarr(
                    path,
                    tables=tables,
                    export_run_id=export_run_id,
                    baseline_time_bin_s=baseline_time_bin_s,
                    baseline_sample_rate_hz=baseline_sample_rate_hz,
                    baseline_full_resolution_samples=baseline_full_resolution_samples,
                    baseline_spatial_grid_size=baseline_spatial_grid_size,
                    legacy_compatibility=legacy_compatibility,
                )
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [
                pool.submit(
                    export_one_zarr,
                    path,
                    tables=tables,
                    export_run_id=export_run_id,
                    baseline_time_bin_s=baseline_time_bin_s,
                    baseline_sample_rate_hz=baseline_sample_rate_hz,
                    baseline_full_resolution_samples=baseline_full_resolution_samples,
                    baseline_spatial_grid_size=baseline_spatial_grid_size,
                    legacy_compatibility=legacy_compatibility,
                )
                for path in zarr_paths
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item.zarr_path)

    rows_by_table_source: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {
        table: [] for table in tables
    }
    diagnostics: list[dict[str, Any]] = []
    for result in results:
        diagnostics.extend(
            {"zarr_path": result.zarr_path, "recording_id": result.recording_id, **diag}
            for diag in result.diagnostics
        )
        for table in tables:
            rows = result.rows_by_table.get(table, [])
            if collection_summary is not None:
                for row in rows:
                    row["collection_id"] = collection_summary.collection_id
                    row["collection_manifest_sha256"] = collection_summary.manifest_sha256
                    row["collection_manifest_path"] = collection_summary.path
            rows_by_table_source[table].append((result.zarr_path, rows))

    columns_by_table = {
        table: sorted(
            {
                key
                for _source, rows in rows_by_table_source[table]
                for row in rows
                for key in row
            }
        )
        for table in tables
        if any(rows for _source, rows in rows_by_table_source[table])
    }
    capability_statuses = resolve_capabilities(columns_by_table)

    output_root.mkdir(parents=True, exist_ok=True)
    generation_id = uuid.uuid4().hex
    final_generation_path = generation_relative_path(export_run_id, generation_id)
    staging_root = publication_staging_root(
        output_root,
        export_run_id,
        generation_id,
    )
    final_generation_root = publication_generation_root(
        output_root,
        export_run_id,
        generation_id,
    )
    if staging_root.exists() or final_generation_root.exists():
        raise FileExistsError(f"Analytics export generation already exists: {generation_id}")
    row_counts: dict[str, int] = {}
    try:
        for table in tables:
            count, _parts = _write_table_parts(
                generation_root=staging_root,
                table=table,
                rows_by_source=rows_by_table_source[table],
            )
            row_counts[table] = count
        part_files, part_inventory = _staged_part_inventory(
            staging_root=staging_root,
            final_generation_path=final_generation_path,
            tables=tables,
        )
        inventory_rows = {
            table: sum(int(entry["row_count"]) for entry in entries)
            for table, entries in part_inventory.items()
        }
        if inventory_rows != row_counts:
            raise ValueError(
                "Staged analytics export row counts differ from the exact part inventory"
            )
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    git = get_git_info(Path(__file__).resolve().parents[3])
    manifest = {
        "export_run_id": export_run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_id": EXPORT_SCHEMA_ID,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "tool": "fisheye.utils.export_cross_recording_analytics",
        "hostname": socket.gethostname(),
        "palette_git_commit": git.get("commit_hash"),
        "palette_git_dirty": git.get("is_dirty"),
        "source_recording_count": len(zarr_paths),
        "source_zarrs": [str(path) for path in zarr_paths],
        "tables_requested": list(tables),
        "table_contracts": contract_snapshot(tables),
        "arrow_schema_contracts": arrow_contract_envelope(tables),
        "capabilities": [
            status.capability_id for status in capability_statuses if status.available
        ],
        "capability_statuses": [status.to_dict() for status in capability_statuses],
        "row_counts_by_table": row_counts,
        "part_files_by_table": part_files,
        "publication": {
            "schema_id": PUBLICATION_SCHEMA_ID,
            "schema_version": PUBLICATION_SCHEMA_VERSION,
            "state": "complete",
            "generation_id": generation_id,
            "generation_path": final_generation_path.as_posix(),
            "parts_by_table": part_inventory,
        },
        "diagnostics": diagnostics,
        "collection_manifest": (
            {
                "path": collection_summary.path,
                "collection_id": collection_summary.collection_id,
                "collection_name": collection_summary.collection_name,
                "manifest_sha256": collection_summary.manifest_sha256,
                "schema_id": collection_summary.schema_id,
                "schema_version": collection_summary.schema_version,
                "record_count": collection_summary.record_count,
                "included_record_count": collection_summary.included_record_count,
            }
            if collection_summary is not None
            else None
        ),
        "export_parameters": {
            "jobs": jobs,
            "overwrite": overwrite,
            "legacy_compatibility": bool(legacy_compatibility),
            "collection_manifest_path": (
                collection_summary.path if collection_summary is not None else None
            ),
            "baseline": {
                "time_bin_s": float(baseline_time_bin_s),
                "spatial_grid_size": int(baseline_spatial_grid_size),
                "kinematic_samples_requested": BASELINE_KINEMATIC_SAMPLES_TABLE in tables,
                "sample_rate_hz": (
                    None
                    if baseline_full_resolution_samples
                    else float(baseline_sample_rate_hz)
                ),
                "full_resolution_samples": bool(baseline_full_resolution_samples),
            },
        },
    }
    safe_manifest = _json_safe(manifest)
    if not isinstance(safe_manifest, Mapping):
        raise TypeError("Analytics export manifest did not normalize to an object")
    committed_manifest_path = commit_staged_publication(
        output_root,
        staging_root,
        safe_manifest,
        baseline_manifest_identity=baseline_manifest_identity,
    )
    result_manifest = dict(safe_manifest)
    result_manifest["manifest_path"] = str(committed_manifest_path)
    return result_manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export cross-recording Palette analytics tables to Parquet.",
    )
    parser.add_argument("--zarr", action="append", type=Path, help="Analysis Zarr path. May be repeated.")
    parser.add_argument("--recordings-root", type=Path, help="Root to scan recursively for *_analysis.zarr archives.")
    parser.add_argument(
        "--collection-manifest",
        type=Path,
        help=(
            "Virtual collection manifest. Included records provide the source Zarr list, "
            "and collection_id/manifest_sha256 are written to export metadata and rows."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root, e.g. /nvme1/exports/palette_analytics.",
    )
    parser.add_argument(
        "--tables",
        default=",".join(DEFAULT_TABLES),
        help=f"Comma-separated table list. Available: {', '.join(AVAILABLE_TABLES)}.",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Parallel extraction workers by recording.")
    parser.add_argument(
        "--baseline-time-bin-s",
        type=float,
        default=5.0,
        help="Fixed baseline behavior time-bin width in seconds (default: 5).",
    )
    parser.add_argument(
        "--baseline-sample-rate-hz",
        type=float,
        default=10.0,
        help=(
            "Requested rate for optional baseline_kinematic_samples (default: 10 Hz). "
            "The effective rate is recorded after integer-frame stride selection."
        ),
    )
    parser.add_argument(
        "--baseline-full-resolution-samples",
        action="store_true",
        help="Export every source kinematic sample when baseline_kinematic_samples is selected.",
    )
    parser.add_argument(
        "--baseline-spatial-grid-size",
        type=int,
        default=12,
        help="Per-axis grid size used for normalized baseline spatial entropy (default: 12).",
    )
    parser.add_argument("--limit", type=int, help="Limit discovered sources, useful for canaries.")
    parser.add_argument("--export-run-id", help="Explicit export run id. Defaults to current UTC timestamp.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing export_run_id directory/manifest.")
    parser.add_argument(
        "--legacy-compatibility",
        action="store_true",
        help=(
            "Permit explicitly legacy/unmanifested swim-bout and bout-kinematics "
            "sources. Current exact schemas remain required by default."
        ),
    )
    parser.add_argument("--registry", type=Path, help="Palette registry SQLite path for optional export indexing.")
    parser.add_argument(
        "--index-registry",
        action="store_true",
        help="Index the written export manifest in the Palette registry after a successful export.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    tables = _parse_tables(args.tables)
    sources = _collect_sources(args)
    manifest = export_sources(
        sources,
        output_root=args.output_root,
        tables=tables,
        jobs=max(1, int(args.jobs)),
        export_run_id=args.export_run_id,
        overwrite=bool(args.overwrite),
        collection_manifest_path=args.collection_manifest,
        baseline_time_bin_s=float(args.baseline_time_bin_s),
        baseline_sample_rate_hz=float(args.baseline_sample_rate_hz),
        baseline_full_resolution_samples=bool(args.baseline_full_resolution_samples),
        baseline_spatial_grid_size=int(args.baseline_spatial_grid_size),
        legacy_compatibility=bool(args.legacy_compatibility),
    )
    print(f"export_run_id\t{manifest['export_run_id']}")
    print(f"manifest\t{manifest['manifest_path']}")
    for table, count in manifest["row_counts_by_table"].items():
        print(f"rows\t{table}\t{count}")
    if manifest["diagnostics"]:
        print(f"diagnostics\t{len(manifest['diagnostics'])}")
    if args.index_registry:
        registry_path = (
            args.registry.expanduser().resolve()
            if args.registry is not None
            else RegistryPaths.from_env(Path.cwd()).path
        )
        registry = Registry(registry_path)
        try:
            export_run_id = index_export_manifest(registry, Path(manifest["manifest_path"]))
        finally:
            registry.close()
        print(f"indexed_registry\t{registry_path}\t{export_run_id}")


if __name__ == "__main__":
    main()
