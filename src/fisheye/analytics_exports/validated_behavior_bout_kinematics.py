"""Exact source binding for bout-kinematics cohort projections.

This consumer binds the report-selected bout run to the already admitted core
track, swim-bout, and eye suppliers. It does not select a run, change a source
publication, or reinterpret a source metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fisheye.shared.run_provenance import validate_run_provenance
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .validated_behavior_bout_kinematics_contracts import source_dtype


LEVELS = ("movement", "heading_raw", "heading_smoothed", "eye_gaze")
SOURCE_SCHEMA_ID = "analysis.bout_kinematics_runs"
SOURCE_SCHEMA_VERSION = 7
SOURCE_LAYOUT = "compact_tabular_v2"
BINDING_SCHEMA_ID = "palette.validated_behavior.bout_kinematics_source_binding"


class BoutKinematicsExportSourceError(ValueError):
    """The selected bout run cannot supply this validated export."""


def _fail(message: str) -> None:
    raise BoutKinematicsExportSourceError(message)


def _mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field_name} must be one object.")
    return value


def validate_bout_kinematics_source_refs(
    refs: Mapping[str, Any], *, expected: Mapping[str, Any]
) -> None:
    """Prove that the derived run names the three exact admitted suppliers."""

    pairs = {
        "zarr_path": "zarr_path",
        "source_track_kinematics_run": "track_run",
        "source_track_kinematics_path": "track_path",
        "source_track_id": "track_id",
        "source_swim_bout_run": "swim_bout_run",
        "source_swim_bout_candidate_id": "swim_bout_candidate_id",
        "source_swim_bout_signal_id": "swim_bout_signal_id",
        "source_swim_bout_speed_level": "swim_bout_speed_level",
        "source_eye_angle_run": "eye_run",
        "source_eye_angle_path": "eye_path",
    }
    for source_field, expected_field in pairs.items():
        if refs.get(source_field) != expected[expected_field]:
            _fail(f"Bout-kinematics {source_field} differs from its admitted source.")
    if refs.get("source_track_kinematics_scope") != "offline":
        _fail("Bout-kinematics track scope is not the admitted offline scope.")
    track_path = str(expected["track_path"])
    track_id = int(expected["track_id"])
    if refs.get("source_track_kinematics_track_path") != (
        f"{track_path}/tracks/id_{track_id}"
    ):
        _fail("Bout-kinematics track path differs from its admitted track.")
    track_authority = _mapping(
        refs.get("source_track_motion_authority"),
        field_name="bout track-motion authority",
    )
    bout_track_authority = _mapping(
        refs.get("source_swim_bout_track_motion_authority"),
        field_name="swim-bout track-motion authority",
    )
    if track_authority != bout_track_authority or (
        track_authority.get("motion_manifest_sha256")
        != expected["track_manifest_sha256"]
    ):
        _fail("Bout-kinematics track-motion authority differs from its source.")
    path = (
        f"analysis/swim_bout_runs/{expected['swim_bout_run']}/tables/bouts"
        f"?candidate_id={expected['swim_bout_candidate_id']}"
        f"&signal_id={expected['swim_bout_signal_id']}"
    )
    if refs.get("source_swim_bout_path") != path:
        _fail("Bout-kinematics selected swim-bout query path differs.")


def validate_bout_kinematics_run_provenance(
    provenance: Mapping[str, Any], *, refs: Mapping[str, Any], git_commit: object
) -> None:
    """Require recorded input identifiers and producing code to match the run."""

    check = validate_run_provenance(provenance)
    if not check.valid:
        _fail("Bout-kinematics run provenance is invalid: " + "; ".join(check.errors))
    if provenance.get("input_run_ids") != refs:
        _fail("Bout-kinematics run provenance names different source inputs.")
    if provenance.get("git_sha") != git_commit:
        _fail("Bout-kinematics run provenance names different producing code.")


def validate_bout_kinematics_metric_rows(
    records_by_level: Mapping[str, np.ndarray],
    canonical_bouts: np.ndarray,
) -> dict[str, int]:
    """Require one correctly keyed row per selected bout at every level."""

    if set(records_by_level) != set(LEVELS):
        _fail("Bout-kinematics run lacks the required movement, heading, or eye level.")
    source = np.asarray(canonical_bouts)
    source_fields = {
        "bout_id": "bout_id",
        "source_start_frame": "start_frame",
        "source_end_frame": "end_frame",
        "source_core_start_frame": "core_start_frame",
        "source_core_end_frame": "core_end_frame",
    }
    if not set(source_fields.values()).issubset(source.dtype.names or ()):
        _fail("Selected canonical bouts lack required boundary identities.")
    order = np.argsort(source["bout_id"], kind="stable")
    selected = source[order]
    bout_ids = np.asarray(selected["bout_id"], dtype=np.int64)
    if bout_ids.size > 1 and np.any(np.diff(bout_ids) <= 0):
        _fail("Selected canonical bout IDs are not unique and increasing.")
    counts: dict[str, int] = {}
    for level in LEVELS:
        records = np.asarray(records_by_level[level])
        if not set(source_fields).issubset(records.dtype.names or ()):
            _fail(f"Bout-kinematics {level} lacks exact source row identities.")
        if len(records) != len(selected):
            _fail(f"Bout-kinematics {level} does not cover every canonical bout.")
        for metric_field, bout_field in source_fields.items():
            if not np.array_equal(records[metric_field], selected[bout_field]):
                _fail(
                    f"Bout-kinematics {level}.{metric_field} disagrees with "
                    "the selected canonical bouts."
                )
        counts[level] = len(records)
    return counts


def _content_sha256(records: np.ndarray) -> str:
    value = np.ascontiguousarray(records)
    digest = hashlib.sha256()
    digest.update(canonical_json_sha256(value.dtype.descr).encode("ascii"))
    digest.update(canonical_json_sha256(list(value.shape)).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def pack_bout_kinematics_metric_rows(
    records_by_level: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Discard compact layout columns and padding before sealing metric bytes."""

    if set(records_by_level) != set(LEVELS):
        _fail("Bout-kinematics run lacks its four required logical levels.")
    packed_by_level: dict[str, np.ndarray] = {}
    for level in LEVELS:
        source = np.asarray(records_by_level[level])
        native_level = "heading" if level.startswith("heading_") else level
        dtype = source_dtype(native_level)
        if source.ndim != 1 or source.dtype.names != dtype.names:
            _fail(f"Bout-kinematics {level} metric field roster is not native.")
        if any(
            source.dtype.fields[name][0] != dtype.fields[name][0]
            for name in dtype.names
        ):
            _fail(f"Bout-kinematics {level} metric field dtypes are not native.")
        packed = np.empty(source.shape[0], dtype=dtype)
        for name in dtype.names:
            packed[name] = source[name]
        packed_by_level[level] = packed
    return packed_by_level


@dataclass(frozen=True)
class BoundBoutKinematicsMetricsSource:
    binding: Mapping[str, Any]
    records_by_level: Mapping[str, np.ndarray] = field(repr=False, compare=False)


def bind_bout_kinematics_metrics_source(
    root: Any,
    *,
    run_name: str,
    run_path: str,
    zarr_path: str | Path,
    track_binding: Mapping[str, Any],
    bout_binding: Mapping[str, Any],
    eye_binding: Mapping[str, Any],
    canonical_bouts: np.ndarray,
    require_selector_eligible: bool,
) -> BoundBoutKinematicsMetricsSource:
    """Validate one report-selected source and seal its exact logical values."""

    from fisheye.analysis.bout_kinematics import resolve_bout_kinematics_tables
    from fisheye.analysis.bout_kinematics_schema import (
        validate_bout_kinematics_array_manifest,
    )

    expected_path = f"analysis/bout_kinematics_runs/{run_name}"
    if run_path != expected_path or run_path not in root:
        _fail("Execution report does not name one present bout-kinematics run.")
    run = root[run_path]
    attrs = dict(run.attrs)
    if (
        attrs.get("palette_run_name") != run_name
        or attrs.get("palette_run_stage") != "bout_kinematics"
        or attrs.get("palette_run_completion_status") != "complete"
        or attrs.get("palette_run_completion_contract")
        != "palette.zarr_run_completion.v1"
        or attrs.get("stage_selector_eligible") is not require_selector_eligible
        or attrs.get("schema_id") != SOURCE_SCHEMA_ID
        or attrs.get("schema_version") != SOURCE_SCHEMA_VERSION
        or attrs.get("layout") != SOURCE_LAYOUT
    ):
        _fail("Bout-kinematics run fails its exact completion and schema contract.")
    provenance = _mapping(attrs.get("run_provenance"), field_name="bout run provenance")
    manifest_errors = validate_bout_kinematics_array_manifest(run)
    if manifest_errors:
        _fail(
            "Bout-kinematics array manifest is invalid: " + "; ".join(manifest_errors)
        )
    refs = _mapping(attrs.get("source_refs"), field_name="bout source_refs")
    validate_bout_kinematics_run_provenance(
        provenance, refs=refs, git_commit=attrs.get("git_commit")
    )
    track_id = int(bout_binding["track_id"])
    expected = {
        "zarr_path": str(Path(zarr_path).expanduser().resolve()),
        "track_run": track_binding["run_name"],
        "track_path": track_binding["run_path"],
        "track_id": track_id,
        "track_manifest_sha256": track_binding["source_manifest_sha256"],
        "swim_bout_run": bout_binding["run_name"],
        "swim_bout_candidate_id": bout_binding["candidate_id"],
        "swim_bout_signal_id": bout_binding["signal_id"],
        "swim_bout_speed_level": bout_binding["speed_level"],
        "eye_run": eye_binding["run_name"],
        "eye_path": eye_binding["run_path"],
    }
    validate_bout_kinematics_source_refs(refs, expected=expected)
    if (
        attrs.get("source_track_kinematics_run") != expected["track_run"]
        or attrs.get("source_swim_bout_run") != expected["swim_bout_run"]
        or attrs.get("source_swim_bout_speed_level")
        != expected["swim_bout_speed_level"]
        or attrs.get("source_track_id") != track_id
    ):
        _fail("Bout-kinematics run attrs disagree with its sealed source refs.")
    logical_rows, _level_attrs, _table_attrs = resolve_bout_kinematics_tables(run)
    records_by_level = pack_bout_kinematics_metric_rows(logical_rows)
    row_counts = validate_bout_kinematics_metric_rows(records_by_level, canonical_bouts)
    array_manifest = _mapping(
        attrs.get("array_schema_manifest"), field_name="bout array manifest"
    )
    binding_body = {
        "schema_id": BINDING_SCHEMA_ID,
        "schema_version": 1,
        "zarr_path": expected["zarr_path"],
        "run_name": run_name,
        "run_path": run_path,
        "source_schema_id": SOURCE_SCHEMA_ID,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "source_layout": SOURCE_LAYOUT,
        "completion_snapshot": {
            "status": attrs["palette_run_completion_status"],
            "completed_at_utc": attrs.get("palette_run_completed_at_utc"),
            "selector_eligible": attrs["stage_selector_eligible"],
        },
        "source_refs_sha256": canonical_json_sha256(refs),
        "source_array_manifest_sha256": array_manifest["payload_digest"],
        "source_run_provenance_sha256": canonical_json_sha256(provenance),
        "track_source_binding_sha256": track_binding["payload_sha256"],
        "swim_bout_source_binding_sha256": bout_binding["payload_sha256"],
        "eye_source_binding_sha256": eye_binding["payload_sha256"],
        "source_track_id": track_id,
        "source_signal_id": int(bout_binding["signal_id"]),
        "row_counts_by_level": row_counts,
        "content_sha256_by_level": {
            level: _content_sha256(np.asarray(records_by_level[level]))
            for level in LEVELS
        },
    }
    binding = {
        **binding_body,
        "payload_sha256": canonical_json_sha256(binding_body),
    }
    return BoundBoutKinematicsMetricsSource(
        binding=binding, records_by_level=records_by_level
    )


__all__ = [
    "BINDING_SCHEMA_ID",
    "LEVELS",
    "BoundBoutKinematicsMetricsSource",
    "BoutKinematicsExportSourceError",
    "bind_bout_kinematics_metrics_source",
    "pack_bout_kinematics_metric_rows",
    "validate_bout_kinematics_metric_rows",
    "validate_bout_kinematics_run_provenance",
    "validate_bout_kinematics_source_refs",
]
