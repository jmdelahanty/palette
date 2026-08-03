"""Closed exact-array contract for maintained compact bout-kinematics runs."""

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


BOUT_KINEMATICS_RUN_SCHEMA_ID = "analysis.bout_kinematics_runs"
BOUT_KINEMATICS_RUN_SCHEMA_VERSION = 7
BOUT_KINEMATICS_LAYOUT = "compact_tabular_v2"
BOUT_KINEMATICS_ARRAY_MANIFEST_SCHEMA_ID = (
    "palette.bout_kinematics.array_schema_manifest"
)
_COLUMNAR_TABLE_PATHS = (
    "level_index",
    "movement_metrics",
    "heading_metrics",
    "eye_gaze_metrics",
)


_LEVEL_FIELDS = [
    ("analysis_level_id", "i2"),
    ("analysis_level_bytes", "S64"),
    ("heading_level_id", "i2"),
    ("heading_level_bytes", "S64"),
]


def _prepend(dtype: np.dtype) -> np.dtype:
    return np.dtype(
        [*_LEVEL_FIELDS, *[(name, dtype.fields[name][0]) for name in dtype.names or ()]]
    )


def _required_specs() -> dict[str, ColumnSpec]:
    from fisheye.analysis import bout_kinematics as writer

    level_index = np.dtype(
        [
            ("analysis_level_id", "i2"),
            ("analysis_level_bytes", "S64"),
            ("measurement_family_bytes", "S64"),
            ("heading_level_id", "i2"),
            ("heading_level_bytes", "S64"),
            ("is_default_heading_level", "?"),
            ("row_count", "i8"),
        ]
    )
    specs: dict[str, ColumnSpec] = {}
    specs.update(
        prefixed_specs(
            "level_index",
            level_index,
            access=AccessPattern.EAGER,
            authority=AnalysisAuthorityRole.SEMANTIC_METADATA,
        )
    )
    specs.update(
        prefixed_specs(
            "movement_metrics",
            _prepend(writer._movement_metrics_dtype()),
            access=AccessPattern.INDEXED,
        )
    )
    specs.update(
        prefixed_specs(
            "heading_metrics",
            _prepend(writer._metrics_dtype()),
            access=AccessPattern.INDEXED,
        )
    )
    return specs


def _optional_bundles() -> dict[str, dict[str, ColumnSpec]]:
    from fisheye.analysis import bout_kinematics as writer

    return {
        "eye_gaze_metrics": prefixed_specs(
            "eye_gaze_metrics",
            _prepend(writer._eye_gaze_metrics_dtype()),
            access=AccessPattern.INDEXED,
            authority=AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY,
        )
    }


def build_bout_kinematics_array_manifest(run_group: Any) -> dict[str, Any]:
    return build_exact_manifest(
        run_group,
        collect_run_arrays(run_group),
        manifest_schema_id=BOUT_KINEMATICS_ARRAY_MANIFEST_SCHEMA_ID,
        run_schema_id=BOUT_KINEMATICS_RUN_SCHEMA_ID,
        run_schema_version=BOUT_KINEMATICS_RUN_SCHEMA_VERSION,
        layout=BOUT_KINEMATICS_LAYOUT,
        schema_prefix="palette.array.bout_kinematics",
        required=_required_specs(),
        optional_bundles=_optional_bundles(),
        columnar_table_paths=_COLUMNAR_TABLE_PATHS,
    )


def validate_bout_kinematics_array_manifest(run_group: Any) -> tuple[str, ...]:
    attrs = dict(run_group.attrs)
    errors: list[str] = []
    if attrs.get("schema_id") != BOUT_KINEMATICS_RUN_SCHEMA_ID:
        errors.append("bout-kinematics schema_id mismatch")
    if type(attrs.get("schema_version")) is not int or attrs.get("schema_version") != BOUT_KINEMATICS_RUN_SCHEMA_VERSION:
        errors.append("bout-kinematics schema_version mismatch")
    if attrs.get("layout") != BOUT_KINEMATICS_LAYOUT:
        errors.append("bout-kinematics layout mismatch")
    errors.extend(
        validate_exact_manifest(
            run_group,
            collect_run_arrays(run_group),
            attrs.get(MANIFEST_ATTRIBUTE),
            manifest_schema_id=BOUT_KINEMATICS_ARRAY_MANIFEST_SCHEMA_ID,
            run_schema_id=BOUT_KINEMATICS_RUN_SCHEMA_ID,
            run_schema_version=BOUT_KINEMATICS_RUN_SCHEMA_VERSION,
            layout=BOUT_KINEMATICS_LAYOUT,
            schema_prefix="palette.array.bout_kinematics",
            required=_required_specs(),
            optional_bundles=_optional_bundles(),
            columnar_table_paths=_COLUMNAR_TABLE_PATHS,
        )
    )
    return tuple(errors)


def write_bout_kinematics_array_manifest(run_group: Any) -> dict[str, Any]:
    manifest = build_bout_kinematics_array_manifest(run_group)
    run_group.attrs[MANIFEST_ATTRIBUTE] = manifest
    errors = validate_bout_kinematics_array_manifest(run_group)
    if errors:
        raise ValueError(
            "Invalid bout-kinematics exact array manifest: " + "; ".join(errors)
        )
    return manifest


__all__ = [
    "BOUT_KINEMATICS_ARRAY_MANIFEST_SCHEMA_ID",
    "BOUT_KINEMATICS_LAYOUT",
    "BOUT_KINEMATICS_RUN_SCHEMA_ID",
    "BOUT_KINEMATICS_RUN_SCHEMA_VERSION",
    "build_bout_kinematics_array_manifest",
    "validate_bout_kinematics_array_manifest",
    "write_bout_kinematics_array_manifest",
]
