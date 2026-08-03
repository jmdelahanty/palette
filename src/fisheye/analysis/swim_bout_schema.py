"""Closed exact-array contract for maintained compact swim-bout runs."""

from __future__ import annotations

from typing import Any

import numpy as np

from fisheye.shared.zarr.analysis_array_contracts import AnalysisAuthorityRole
from fisheye.shared.zarr.storage_intent import AccessPattern

from ._exact_tabular_run_schema import (
    ColumnSpec,
    MANIFEST_ATTRIBUTE,
    build_exact_manifest,
    collect_run_arrays,
    prefixed_specs,
    validate_exact_manifest,
)


SWIM_BOUT_RUN_SCHEMA_ID = "palette.swim_bout_runs"
SWIM_BOUT_RUN_SCHEMA_VERSION = 8
SWIM_BOUT_LAYOUT = "compact_tabular_v2"
SWIM_BOUT_ARRAY_MANIFEST_SCHEMA_ID = "palette.swim_bout.array_schema_manifest"
_COLUMNAR_TABLE_PATHS = (
    "indexes/candidates",
    "indexes/signal_variants",
    "tables/bouts",
    "tables/peak_events",
    "tables/inter_bout_intervals",
    "tables/summary_metrics",
    "tables/histograms",
    "tables/bout_points",
)


def _prepend(dtype: np.dtype, fields: list[tuple[str, str]]) -> np.dtype:
    return np.dtype([*fields, *[(name, dtype.fields[name][0]) for name in dtype.names or ()]])


def _required_specs() -> dict[str, ColumnSpec]:
    # Import lazily so the producer may install this contract after completing
    # its module initialization without a circular import.
    from fisheye.analysis import detect_bouts_multi_level as writer

    empty_bouts = writer._empty_bouts()
    intervals, _metrics, histogram = writer._compute_inter_bout_intervals(
        empty_bouts, 1.0
    )
    points = writer._create_bout_points(
        empty_bouts,
        None,
        None,
        np.asarray([], dtype=np.int64),
        1.0,
    )

    candidate_dtype = np.dtype(
        [
            ("candidate_id", "i4"),
            ("candidate_name", "S256"),
            ("is_default", "?"),
            ("detection_method", "S32"),
            ("boundary_mode", "S32"),
            ("boundary_window_s", "f8"),
            ("boundary_constraint", "S32"),
            ("gap_merge_policy", "S32"),
            ("min_bout_duration_s", "f8"),
            ("min_gap_duration_s", "f8"),
            ("min_gap_frames", "i4"),
            ("parameter_hash", "S64"),
            ("parameters_json", "S8192"),
            ("provenance_json", "S1"),
        ]
    )
    signal_dtype = np.dtype(
        [
            ("signal_id", "i4"),
            ("speed_level", "S32"),
            ("signal_name", "S32"),
            ("role", "S32"),
            ("source_level", "S32"),
            ("transform_type", "S32"),
            ("transform_source_signal_id", "i4"),
            ("tau_s", "f8"),
            ("units", "S16"),
            ("path_distance_source_level", "S32"),
            ("parameters_json", "S8192"),
        ]
    )
    bout_dtype = _prepend(
        writer._bout_dtype(),
        [
            ("candidate_id", "i4"),
            ("signal_id", "i4"),
            ("estimator_signal_id", "i4"),
            ("track_id", "i4"),
            ("mean_speed_px_s", "f8"),
            ("peak_detection_signal_px_s", "f8"),
            ("peak_frame", "i8"),
            ("peak_time_s", "f8"),
            ("threshold_crossing_valid", "?"),
        ],
    )
    peak_dtype = _prepend(
        writer._peak_event_dtype(),
        [
            ("peak_event_id", "i8"),
            ("candidate_id", "i4"),
            ("signal_id", "i4"),
            ("accepted", "?"),
            ("rejection_reason", "S16"),
        ],
    )
    interval_dtype = _prepend(
        intervals.dtype,
        [
            ("interval_id", "i8"),
            ("candidate_id", "i4"),
            ("signal_id", "i4"),
            ("valid", "?"),
        ],
    )
    histogram_dtype = _prepend(
        histogram.dtype,
        [
            ("candidate_id", "i4"),
            ("signal_id", "i4"),
            ("metric_name", "S32"),
            ("bin_left", "f8"),
            ("bin_right", "f8"),
            ("density", "f8"),
            ("units", "S8"),
        ],
    )
    point_dtype = _prepend(
        points.dtype,
        [
            ("candidate_id", "i4"),
            ("signal_id", "i4"),
            ("point_role", "S8"),
        ],
    )
    summary_dtype = np.dtype(
        [
            ("candidate_id", "i4"),
            ("signal_id", "i4"),
            ("metric_name", "S64"),
            ("value", "f8"),
            ("units", "S8"),
            ("source_table", "S32"),
        ]
    )
    specs: dict[str, ColumnSpec] = {}
    for prefix, dtype, authority, access in (
        ("indexes/candidates", candidate_dtype, AnalysisAuthorityRole.SEMANTIC_METADATA, AccessPattern.EAGER),
        ("indexes/signal_variants", signal_dtype, AnalysisAuthorityRole.SEMANTIC_METADATA, AccessPattern.EAGER),
        ("tables/bouts", bout_dtype, AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY, AccessPattern.INDEXED),
        ("tables/peak_events", peak_dtype, AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY, AccessPattern.INDEXED),
        ("tables/inter_bout_intervals", interval_dtype, AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY, AccessPattern.INDEXED),
        ("tables/summary_metrics", summary_dtype, AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY, AccessPattern.EAGER),
        ("tables/histograms", histogram_dtype, AnalysisAuthorityRole.DERIVED_CACHE, AccessPattern.EAGER),
        ("tables/bout_points", point_dtype, AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY, AccessPattern.INDEXED),
    ):
        specs.update(prefixed_specs(prefix, dtype, access=access, authority=authority))

    specs["signals/detector_signal_mm_s"] = ColumnSpec(
        path="signals/detector_signal_mm_s",
        dtype=np.dtype("float32").str,
        axes=("detector_signal", "frame"),
        units="mm/s",
        access=AccessPattern.WINDOWED,
    )
    specs["signals/detector_signal_signal_ids"] = ColumnSpec(
        path="signals/detector_signal_signal_ids",
        dtype=np.dtype("int32").str,
        axes=("detector_signal",),
        authority=AnalysisAuthorityRole.LINEAGE_INDEX,
        access=AccessPattern.EAGER,
        fill="every row stores one nonnegative signal identifier",
        null="no null sentinel",
    )
    return specs


def _optional_bundles() -> dict[str, dict[str, ColumnSpec]]:
    return {
        "embedded_frame_axis": {
            "signals/frame_indices": ColumnSpec(
                path="signals/frame_indices",
                dtype=np.dtype("int64").str,
                axes=("frame",),
                units="acquisition_frame_index",
                authority=AnalysisAuthorityRole.LINEAGE_INDEX,
                access=AccessPattern.WINDOWED,
                fill="every row stores an authoritative acquisition-frame index",
                null="negative frame indices are forbidden",
            )
        }
    }


def build_swim_bout_array_manifest(run_group: Any) -> dict[str, Any]:
    return build_exact_manifest(
        run_group,
        collect_run_arrays(run_group),
        manifest_schema_id=SWIM_BOUT_ARRAY_MANIFEST_SCHEMA_ID,
        run_schema_id=SWIM_BOUT_RUN_SCHEMA_ID,
        run_schema_version=SWIM_BOUT_RUN_SCHEMA_VERSION,
        layout=SWIM_BOUT_LAYOUT,
        schema_prefix="palette.array.swim_bout",
        required=_required_specs(),
        optional_bundles=_optional_bundles(),
        columnar_table_paths=_COLUMNAR_TABLE_PATHS,
    )


def validate_swim_bout_array_manifest(run_group: Any) -> tuple[str, ...]:
    attrs = dict(run_group.attrs)
    errors: list[str] = []
    if attrs.get("schema_id") != SWIM_BOUT_RUN_SCHEMA_ID:
        errors.append("swim-bout schema_id mismatch")
    if type(attrs.get("schema_version")) is not int or attrs.get("schema_version") != SWIM_BOUT_RUN_SCHEMA_VERSION:
        errors.append("swim-bout schema_version mismatch")
    if attrs.get("layout") != SWIM_BOUT_LAYOUT:
        errors.append("swim-bout layout mismatch")
    errors.extend(
        validate_exact_manifest(
            run_group,
            collect_run_arrays(run_group),
            attrs.get(MANIFEST_ATTRIBUTE),
            manifest_schema_id=SWIM_BOUT_ARRAY_MANIFEST_SCHEMA_ID,
            run_schema_id=SWIM_BOUT_RUN_SCHEMA_ID,
            run_schema_version=SWIM_BOUT_RUN_SCHEMA_VERSION,
            layout=SWIM_BOUT_LAYOUT,
            schema_prefix="palette.array.swim_bout",
            required=_required_specs(),
            optional_bundles=_optional_bundles(),
            columnar_table_paths=_COLUMNAR_TABLE_PATHS,
        )
    )
    return tuple(errors)


def write_swim_bout_array_manifest(run_group: Any) -> dict[str, Any]:
    manifest = build_swim_bout_array_manifest(run_group)
    run_group.attrs[MANIFEST_ATTRIBUTE] = manifest
    errors = validate_swim_bout_array_manifest(run_group)
    if errors:
        raise ValueError("Invalid swim-bout exact array manifest: " + "; ".join(errors))
    return manifest


__all__ = [
    "SWIM_BOUT_ARRAY_MANIFEST_SCHEMA_ID",
    "SWIM_BOUT_LAYOUT",
    "SWIM_BOUT_RUN_SCHEMA_ID",
    "SWIM_BOUT_RUN_SCHEMA_VERSION",
    "build_swim_bout_array_manifest",
    "validate_swim_bout_array_manifest",
    "write_swim_bout_array_manifest",
]
