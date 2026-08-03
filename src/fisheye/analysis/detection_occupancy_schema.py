"""Exact logical arrays for epoch and full-session detection occupancy.

The two run families share one payload shape, but they are different
scientific authorities: epoch occupancy is bound to a stimulus-window run,
whereas session occupancy contains exactly one full-recording segment and has
no stimulus dependency.  This module freezes that distinction without
changing either writer's physical layout.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fisheye.shared.zarr.analysis_array_contracts import AnalysisAuthorityRole
from fisheye.shared.zarr.storage_intent import AccessPattern

from ._exact_tabular_run_schema import (
    ColumnSpec,
    MANIFEST_ATTRIBUTE,
    build_exact_array_declarations,
    build_exact_manifest,
    collect_run_arrays,
    validate_exact_manifest,
)


DETECTION_OCCUPANCY_ARRAY_MANIFEST_SCHEMA_ID = (
    "palette.detection_occupancy.array_schema_manifest"
)
SESSION_OCCUPANCY_ARRAY_MANIFEST_SCHEMA_ID = (
    "palette.session_occupancy.array_schema_manifest"
)
OCCUPANCY_ARRAY_LAYOUT = "fixed_image_quadrants_v2"
OCCUPANCY_TEXT_WIDTH = 96


def _spec(
    path: str,
    dtype: str,
    axes: tuple[str, ...],
    *,
    units: str | None = None,
    authority: AnalysisAuthorityRole = AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY,
    access: AccessPattern = AccessPattern.EAGER,
    fill: str = "every persisted value is authoritative; no implicit fill",
    null: str = "no null sentinel",
) -> ColumnSpec:
    return ColumnSpec(
        path=path,
        dtype=np.dtype(dtype).str,
        axes=axes,
        units=units,
        authority=authority,
        access=access,
        fill=fill,
        null=null,
    )


def _shared_specs() -> dict[str, ColumnSpec]:
    lineage = AnalysisAuthorityRole.LINEAGE_INDEX
    semantic = AnalysisAuthorityRole.SEMANTIC_METADATA
    quality = AnalysisAuthorityRole.QUALITY_DIAGNOSTIC
    cache = AnalysisAuthorityRole.DERIVED_CACHE
    segment = ("row",)
    zone = ("zone",)
    summary = ("row", "zone")
    heatmap = ("row", "y_bin", "x_bin")
    prefix = "spatial_occupancy/image_quadrants_v1"
    return {
        "windows/window_id": _spec(
            "windows/window_id", "int32", segment, authority=lineage
        ),
        "windows/label_bytes": _spec(
            "windows/label_bytes",
            "uint8",
            ("row", "utf8_byte"),
            authority=semantic,
            fill="NUL-padded fixed UTF-8 bytes",
            null="all-zero row means empty text and is forbidden for a segment label",
        ),
        "windows/start_frame": _spec(
            "windows/start_frame", "int64", segment, units="acquisition_frame_index"
        ),
        "windows/end_frame": _spec(
            "windows/end_frame", "int64", segment, units="acquisition_frame_index"
        ),
        "windows/start_time_s": _spec(
            "windows/start_time_s", "float64", segment, units="s"
        ),
        "windows/end_time_s": _spec(
            "windows/end_time_s", "float64", segment, units="s"
        ),
        "windows/duration_s": _spec(
            "windows/duration_s", "float64", segment, units="s"
        ),
        "windows/source_segment_id": _spec(
            "windows/source_segment_id", "int32", segment, authority=lineage
        ),
        "coverage/detection_count": _spec(
            "coverage/detection_count", "int64", segment, authority=quality
        ),
        "coverage/covered_frame_count": _spec(
            "coverage/covered_frame_count", "int64", segment, authority=quality
        ),
        "coverage/total_span_frames": _spec(
            "coverage/total_span_frames", "int64", segment, authority=quality
        ),
        "coverage/coverage_pct": _spec(
            "coverage/coverage_pct", "float32", segment, units="percent", authority=quality
        ),
        "heatmaps/counts": _spec(
            "heatmaps/counts", "uint32", heatmap, authority=cache
        ),
        "heatmaps/normalized": _spec(
            "heatmaps/normalized",
            "float32",
            heatmap,
            units="fraction_of_segment_max",
            authority=cache,
        ),
        "heatmaps/x_edges": _spec(
            "heatmaps/x_edges", "float32", ("x_bin_edge",), units="px", authority=semantic
        ),
        "heatmaps/y_edges": _spec(
            "heatmaps/y_edges", "float32", ("y_bin_edge",), units="px", authority=semantic
        ),
        f"{prefix}/zone_spec/zone_id": _spec(
            f"{prefix}/zone_spec/zone_id",
            "uint8",
            ("zone", "utf8_byte"),
            authority=semantic,
            fill="NUL-padded fixed UTF-8 bytes",
            null="all-zero row is forbidden for a zone identifier",
        ),
        f"{prefix}/zone_spec/label_bytes": _spec(
            f"{prefix}/zone_spec/label_bytes",
            "uint8",
            ("zone", "utf8_byte"),
            authority=semantic,
            fill="NUL-padded fixed UTF-8 bytes",
            null="all-zero row means an absent display label",
        ),
        f"{prefix}/zone_spec/geometry_type": _spec(
            f"{prefix}/zone_spec/geometry_type",
            "uint8",
            ("zone", "utf8_byte"),
            authority=semantic,
            fill="NUL-padded fixed UTF-8 bytes",
            null="all-zero row is forbidden for a geometry type",
        ),
        f"{prefix}/zone_spec/display_order": _spec(
            f"{prefix}/zone_spec/display_order", "int16", zone, authority=semantic
        ),
        f"{prefix}/zone_spec/bounds_xyxy": _spec(
            f"{prefix}/zone_spec/bounds_xyxy",
            "float32",
            ("zone", "xyxy"),
            units="px",
            authority=semantic,
        ),
        f"{prefix}/summary/frame_count": _spec(
            f"{prefix}/summary/frame_count", "int64", summary
        ),
        f"{prefix}/summary/time_s": _spec(
            f"{prefix}/summary/time_s", "float32", summary, units="s"
        ),
        f"{prefix}/summary/fraction_of_epoch": _spec(
            f"{prefix}/summary/fraction_of_epoch",
            "float32",
            summary,
            units="fraction",
        ),
        f"{prefix}/summary/fraction_of_detected": _spec(
            f"{prefix}/summary/fraction_of_detected",
            "float32",
            summary,
            units="fraction",
        ),
        f"{prefix}/summary/detected_frame_count": _spec(
            f"{prefix}/summary/detected_frame_count", "int64", segment
        ),
        f"{prefix}/summary/missing_frame_count": _spec(
            f"{prefix}/summary/missing_frame_count", "int64", segment, authority=quality
        ),
        f"{prefix}/summary/total_span_frames": _spec(
            f"{prefix}/summary/total_span_frames", "int64", segment
        ),
        f"{prefix}/summary/coverage_pct": _spec(
            f"{prefix}/summary/coverage_pct",
            "float32",
            segment,
            units="percent",
            authority=quality,
        ),
    }


def _required_specs(*, session: bool) -> dict[str, ColumnSpec]:
    result = _shared_specs()
    if not session:
        result["windows/source_stimulus_epoch_window_id"] = _spec(
            "windows/source_stimulus_epoch_window_id",
            "int32",
            ("row",),
            authority=AnalysisAuthorityRole.LINEAGE_INDEX,
        )
    return result


def _identities(*, session: bool) -> tuple[str, int, str, str]:
    # Lazy import prevents the scientific writer importing itself during module
    # initialization while keeping one authority for its public constants.
    from fisheye.analysis import detection_occupancy_runs as writer

    if session:
        return (
            writer.SESSION_SCHEMA_ID,
            writer.SCHEMA_VERSION,
            SESSION_OCCUPANCY_ARRAY_MANIFEST_SCHEMA_ID,
            "palette.array.session_occupancy",
        )
    return (
        writer.SCHEMA_ID,
        writer.SCHEMA_VERSION,
        DETECTION_OCCUPANCY_ARRAY_MANIFEST_SCHEMA_ID,
        "palette.array.detection_occupancy",
    )


def build_occupancy_array_declarations(
    run_group: Any,
    *,
    session: bool,
    byte_planner_adopted: bool = False,
) -> tuple[Any, ...]:
    _run_schema_id, _version, _manifest_id, schema_prefix = _identities(
        session=session
    )
    return build_exact_array_declarations(
        collect_run_arrays(run_group),
        schema_prefix=schema_prefix,
        required=_required_specs(session=session),
        optional_bundles={},
        byte_planner_adopted=byte_planner_adopted,
    )


def build_occupancy_array_manifest(
    run_group: Any,
    *,
    session: bool,
    byte_planner_adopted: bool = False,
) -> dict[str, Any]:
    run_schema_id, version, manifest_id, schema_prefix = _identities(session=session)
    arrays = collect_run_arrays(run_group)
    return build_exact_manifest(
        run_group,
        arrays,
        manifest_schema_id=manifest_id,
        run_schema_id=run_schema_id,
        run_schema_version=version,
        layout=OCCUPANCY_ARRAY_LAYOUT,
        schema_prefix=schema_prefix,
        required=_required_specs(session=session),
        optional_bundles={},
        columnar_table_paths=(),
        byte_planner_adopted=byte_planner_adopted,
    )


def write_occupancy_array_manifest(
    run_group: Any,
    *,
    session: bool,
    byte_planner_adopted: bool = False,
) -> dict[str, Any]:
    manifest = build_occupancy_array_manifest(
        run_group,
        session=session,
        byte_planner_adopted=byte_planner_adopted,
    )
    run_group.attrs[MANIFEST_ATTRIBUTE] = manifest
    return manifest


def validate_occupancy_array_manifest(
    run_group: Any,
    *,
    session: bool,
    byte_planner_adopted: bool = False,
) -> tuple[str, ...]:
    run_schema_id, version, manifest_id, schema_prefix = _identities(session=session)
    arrays = collect_run_arrays(run_group)
    return validate_exact_manifest(
        run_group,
        arrays,
        run_group.attrs.get(MANIFEST_ATTRIBUTE),
        manifest_schema_id=manifest_id,
        run_schema_id=run_schema_id,
        run_schema_version=version,
        layout=OCCUPANCY_ARRAY_LAYOUT,
        schema_prefix=schema_prefix,
        required=_required_specs(session=session),
        optional_bundles={},
        columnar_table_paths=(),
        byte_planner_adopted=byte_planner_adopted,
    )


__all__ = [
    "DETECTION_OCCUPANCY_ARRAY_MANIFEST_SCHEMA_ID",
    "OCCUPANCY_ARRAY_LAYOUT",
    "SESSION_OCCUPANCY_ARRAY_MANIFEST_SCHEMA_ID",
    "build_occupancy_array_declarations",
    "build_occupancy_array_manifest",
    "validate_occupancy_array_manifest",
    "write_occupancy_array_manifest",
]
