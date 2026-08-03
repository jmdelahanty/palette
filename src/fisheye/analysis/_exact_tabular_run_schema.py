"""Exact whole-run manifests for maintained compact columnar analyses.

This helper is intentionally private to the analysis schema modules.  It binds
every physical column array to an ``AnalysisArrayDeclaration`` while leaving
chunking and sharding to the storage-policy owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.shared.zarr.analysis_array_contracts import (
    AnalysisArrayDeclaration,
    AnalysisAuthorityRole,
)
from fisheye.shared.zarr.array_contracts import ArrayContract, DTypeContract
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode


MANIFEST_ATTRIBUTE = "array_schema_manifest"
EXCLUDED_REPORT_PREFIXES = ("visualizations", "report_tables")
LEGACY_COLUMNAR_PHYSICAL_POLICY_OWNER = "shared_columnar_storage_policy"
BYTE_PLANNER_PHYSICAL_POLICY_OWNER = "analysis_storage_planning_v1"


@dataclass(frozen=True)
class ColumnSpec:
    """One closed compact column declaration."""

    path: str
    dtype: str | None
    logical_dtype: str | None = None
    axes: tuple[str, ...] = ("row",)
    units: str | None = None
    authority: AnalysisAuthorityRole = AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY
    access: AccessPattern = AccessPattern.WINDOWED
    fill: str = "no implicit fill; every persisted row is authoritative"
    null: str = "floating NaN or a field-specific sentinel represents unavailable data"


def collect_run_arrays(group: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}

    def visit(node: Any, prefix: str = "") -> None:
        for name, array in sorted(node.arrays(), key=lambda item: item[0]):
            path = f"{prefix}/{name}" if prefix else str(name)
            result[path] = array
        for name, child in sorted(node.groups(), key=lambda item: item[0]):
            child_prefix = f"{prefix}/{name}" if prefix else str(name)
            if child_prefix.split("/", 1)[0] in EXCLUDED_REPORT_PREFIXES:
                continue
            visit(child, child_prefix)

    visit(group)
    return result


def _fixed_dtype_contract(dtype: Any) -> DTypeContract:
    observed = np.dtype(dtype)
    return DTypeContract(
        dtype_id=(
            f"fixed_bytes_{observed.itemsize}"
            if observed.kind == "S"
            else str(observed)
        ),
        numpy_dtype=observed.str,
    )


def _require_array_specs(
    arrays: Mapping[str, Any],
    *,
    required: Mapping[str, ColumnSpec],
    optional_bundles: Mapping[str, Mapping[str, ColumnSpec]],
) -> tuple[dict[str, ColumnSpec], tuple[str, ...]]:
    paths = set(arrays)
    expected = dict(required)
    enabled: list[str] = []
    for bundle_name, bundle in optional_bundles.items():
        present = paths.intersection(bundle)
        if present and present != set(bundle):
            missing = sorted(set(bundle) - present)
            raise ValueError(
                f"Optional bundle {bundle_name!r} is partial; missing {missing!r}."
            )
        if present:
            expected.update(bundle)
            enabled.append(bundle_name)
    missing = sorted(set(required) - paths)
    unexpected = sorted(paths - set(expected))
    if missing:
        raise ValueError(f"Missing required compact arrays: {missing!r}.")
    if unexpected:
        raise ValueError(f"Unexpected compact arrays: {unexpected!r}.")
    return expected, tuple(sorted(enabled))


def _get_child(group: Any, path: str) -> Any:
    current = group
    for component in path.split("/"):
        current = current[component]
    return current


def _columnar_table_declarations(
    run_group: Any,
    *,
    specs: Mapping[str, ColumnSpec],
    table_paths: Sequence[str],
) -> list[dict[str, object]]:
    declarations: list[dict[str, object]] = []
    for table_path in table_paths:
        prefix = table_path + "/"
        fields = [
            (path[len(prefix) :], spec)
            for path, spec in specs.items()
            if path.startswith(prefix) and "/" not in path[len(prefix) :]
        ]
        if not fields:
            continue
        group = _get_child(run_group, table_path)
        expected_names = [name for name, _spec in fields]
        expected_dtypes = {
            name: spec.logical_dtype
            for name, spec in fields
            if spec.logical_dtype is not None
        }
        observed_names = list(group.attrs.get("field_names", []))
        observed_dtypes = dict(group.attrs.get("field_dtypes", {}))
        if observed_names != expected_names:
            raise ValueError(
                f"{table_path}: field_names mismatch; expected {expected_names!r}, "
                f"got {observed_names!r}."
            )
        if observed_dtypes != expected_dtypes:
            raise ValueError(
                f"{table_path}: field_dtypes mismatch; expected {expected_dtypes!r}, "
                f"got {observed_dtypes!r}."
            )
        declarations.append(
            {
                "path": table_path,
                "storage_layout": "columnar",
                "field_names": expected_names,
                "field_dtypes": expected_dtypes,
            }
        )
    return declarations


def _shape_contract(
    spec: ColumnSpec, shape: tuple[int, ...]
) -> tuple[tuple[str | int, ...], dict[str, int]]:
    template: list[str | int] = []
    dimensions: dict[str, int] = {}
    parent = spec.path.rsplit("/", 1)[0] if "/" in spec.path else spec.path
    for axis_index, (axis_name, extent) in enumerate(zip(spec.axes, shape)):
        dimension_name: str | None = None
        if axis_index == 0 and axis_name == "row":
            dimension_name = "n_" + parent.replace("/", "__") + "_rows"
        elif axis_name in {"frame", "detector_signal"}:
            dimension_name = "n_" + axis_name
        if dimension_name is None:
            template.append(int(extent))
        else:
            template.append(dimension_name)
            dimensions[dimension_name] = int(extent)
    return tuple(template), dimensions


def _declaration(
    spec: ColumnSpec,
    array: Any,
    *,
    schema_prefix: str,
    required: bool,
    byte_planner_adopted: bool = False,
) -> AnalysisArrayDeclaration:
    dtype = np.dtype(array.dtype)
    if spec.dtype is not None and dtype != np.dtype(spec.dtype):
        raise ValueError(
            f"{spec.path}: dtype mismatch; expected {np.dtype(spec.dtype)}, got {dtype}."
        )
    if dtype.hasobject:
        raise ValueError(f"{spec.path}: object dtype is forbidden.")
    shape = tuple(int(value) for value in array.shape)
    if len(shape) != len(spec.axes):
        raise ValueError(
            f"{spec.path}: rank mismatch; expected {len(spec.axes)}, got {len(shape)}."
        )
    shape_template, _dimensions = _shape_contract(spec, shape)
    return AnalysisArrayDeclaration(
        path=spec.path,
        contract=ArrayContract(
            schema_id=f"{schema_prefix}.{spec.path.replace('/', '.')}",
            schema_version=1,
            dtype=_fixed_dtype_contract(dtype),
            shape_template=shape_template,
            axis_names=spec.axes,
            description=f"Exact maintained compact column {spec.path}.",
            units=spec.units,
        ),
        required=required,
        access_pattern=spec.access,
        write_mode=WriteMode.IMMUTABLE,
        authority_role=spec.authority,
        fill_semantics=spec.fill,
        null_semantics=spec.null,
        physical_policy_owner=(
            BYTE_PLANNER_PHYSICAL_POLICY_OWNER
            if byte_planner_adopted
            else LEGACY_COLUMNAR_PHYSICAL_POLICY_OWNER
        ),
        byte_planner_adopted=byte_planner_adopted,
    )


def build_exact_array_declarations(
    arrays: Mapping[str, Any],
    *,
    schema_prefix: str,
    required: Mapping[str, ColumnSpec],
    optional_bundles: Mapping[str, Mapping[str, ColumnSpec]],
    byte_planner_adopted: bool = False,
) -> tuple[AnalysisArrayDeclaration, ...]:
    """Return the closed declaration set for one observed compact run.

    The logical schema remains identical between the established columnar
    layout and an opt-in byte-planned candidate.  Only the physical-policy
    owner and adoption flag differ.  Callers must still derive concrete
    chunks/shards from the returned declarations and the observed fixed-width
    array facts; this helper never chooses a storage profile.
    """

    specs, _enabled = _require_array_specs(
        arrays,
        required=required,
        optional_bundles=optional_bundles,
    )
    return tuple(
        _declaration(
            specs[path],
            arrays[path],
            schema_prefix=schema_prefix,
            required=path in required,
            byte_planner_adopted=byte_planner_adopted,
        )
        for path in sorted(specs)
    )


def build_exact_manifest(
    run_group: Any,
    arrays: Mapping[str, Any],
    *,
    manifest_schema_id: str,
    run_schema_id: str,
    run_schema_version: int,
    layout: str,
    schema_prefix: str,
    required: Mapping[str, ColumnSpec],
    optional_bundles: Mapping[str, Mapping[str, ColumnSpec]],
    columnar_table_paths: Sequence[str],
    byte_planner_adopted: bool = False,
) -> dict[str, Any]:
    specs, enabled = _require_array_specs(
        arrays, required=required, optional_bundles=optional_bundles
    )
    declarations = build_exact_array_declarations(
        arrays,
        schema_prefix=schema_prefix,
        required=required,
        optional_bundles=optional_bundles,
        byte_planner_adopted=byte_planner_adopted,
    )
    declaration_by_path = {
        declaration.path: declaration for declaration in declarations
    }
    declaration_manifests: list[dict[str, object]] = []
    dimensions: dict[str, int] = {}
    for path in sorted(specs):
        declaration_manifests.append(declaration_by_path[path].as_manifest())
        shape = tuple(int(value) for value in arrays[path].shape)
        _shape_template, item_dimensions = _shape_contract(specs[path], shape)
        for name, value in item_dimensions.items():
            previous = dimensions.get(name)
            if previous is not None and previous != value:
                raise ValueError(
                    f"Shared dimension {name!r} disagrees: {previous} != {value} "
                    f"at {path}."
                )
            dimensions[name] = value
    payload = {
        "run_schema_id": run_schema_id,
        "run_schema_version": run_schema_version,
        "layout": layout,
        "enabled_optional_bundles": list(enabled),
        "dimensions": dimensions,
        "arrays": declaration_manifests,
        "columnar_tables": _columnar_table_declarations(
            run_group,
            specs=specs,
            table_paths=columnar_table_paths,
        ),
        "forbidden_layouts": ["hierarchical_v1"],
        "excluded_non_scientific_report_namespaces": list(EXCLUDED_REPORT_PREFIXES),
        "physical_policy_owner": (
            BYTE_PLANNER_PHYSICAL_POLICY_OWNER
            if byte_planner_adopted
            else LEGACY_COLUMNAR_PHYSICAL_POLICY_OWNER
        ),
        "byte_planner_adopted": byte_planner_adopted,
    }
    return {
        "schema_id": manifest_schema_id,
        "schema_version": 1,
        "persisted_attribute": MANIFEST_ATTRIBUTE,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }


def validate_exact_manifest(
    run_group: Any,
    arrays: Mapping[str, Any],
    manifest: Any,
    *,
    manifest_schema_id: str,
    run_schema_id: str,
    run_schema_version: int,
    layout: str,
    schema_prefix: str,
    required: Mapping[str, ColumnSpec],
    optional_bundles: Mapping[str, Mapping[str, ColumnSpec]],
    columnar_table_paths: Sequence[str],
    byte_planner_adopted: bool = False,
) -> tuple[str, ...]:
    errors: list[str] = []
    if type(manifest) is not dict:
        return ("array_schema_manifest must be an exact object",)
    if set(manifest) != {
        "schema_id",
        "schema_version",
        "persisted_attribute",
        "digest_algorithm",
        "payload",
        "payload_digest",
    }:
        errors.append("array_schema_manifest envelope has unexpected fields")
    if manifest.get("schema_id") != manifest_schema_id:
        errors.append("array_schema_manifest schema_id mismatch")
    if type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1:
        errors.append("array_schema_manifest schema_version mismatch")
    if manifest.get("persisted_attribute") != MANIFEST_ATTRIBUTE:
        errors.append("array_schema_manifest persisted_attribute mismatch")
    if manifest.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM:
        errors.append("array_schema_manifest digest_algorithm mismatch")
    payload = manifest.get("payload")
    if type(payload) is not dict:
        return (*errors, "array_schema_manifest payload must be an exact object")
    try:
        if manifest.get("payload_digest") != canonical_json_sha256(payload):
            errors.append("array_schema_manifest payload_digest mismatch")
        expected = build_exact_manifest(
            run_group,
            arrays,
            manifest_schema_id=manifest_schema_id,
            run_schema_id=run_schema_id,
            run_schema_version=run_schema_version,
            layout=layout,
            schema_prefix=schema_prefix,
            required=required,
            optional_bundles=optional_bundles,
            columnar_table_paths=columnar_table_paths,
            byte_planner_adopted=byte_planner_adopted,
        )
        if canonical_json_bytes(manifest) != canonical_json_bytes(expected):
            errors.append("array_schema_manifest does not equal the executable schema")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    return tuple(errors)


def prefixed_specs(
    prefix: str,
    dtype: np.dtype,
    *,
    access: AccessPattern = AccessPattern.WINDOWED,
    authority: AnalysisAuthorityRole = AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY,
) -> dict[str, ColumnSpec]:
    if dtype.names is None:
        raise ValueError(f"{prefix}: structured dtype is required.")
    result: dict[str, ColumnSpec] = {}
    for name in dtype.names:
        logical_dtype = dtype.fields[name][0]
        path = f"{prefix}/{name}"
        if logical_dtype.kind in {"S", "U", "O"}:
            result[path] = ColumnSpec(
                path=path,
                dtype=np.dtype("uint8").str,
                logical_dtype=logical_dtype.str,
                axes=("row", "utf8_byte"),
                access=access,
                authority=authority,
                units=None,
                fill="NUL-padded fixed UTF-8 bytes",
                null="all-zero row means empty string",
            )
        else:
            result[path] = ColumnSpec(
                path=path,
                dtype=logical_dtype.str,
                logical_dtype=str(logical_dtype),
                access=access,
                authority=authority,
                units=_infer_units(name),
            )
    return result


def _infer_units(name: str) -> str | None:
    if name.endswith("_frame") or name.endswith("_frames"):
        return "acquisition_frame_index"
    if name.endswith("_time_s") or name.endswith("_duration_s") or name.endswith("_s"):
        return "s"
    if name.endswith("_mm_s") or "speed_mm_s" in name:
        return "mm/s"
    if name.endswith("_px_s"):
        return "px/s"
    if name.endswith("_mm"):
        return "mm"
    if name.endswith("_px"):
        return "px"
    if name.endswith("_deg_s"):
        return "deg/s"
    if name.endswith("_deg"):
        return "deg"
    if name.endswith("_hz"):
        return "Hz"
    if name.endswith("_fraction"):
        return "fraction"
    return None


__all__ = [
    "BYTE_PLANNER_PHYSICAL_POLICY_OWNER",
    "ColumnSpec",
    "LEGACY_COLUMNAR_PHYSICAL_POLICY_OWNER",
    "MANIFEST_ATTRIBUTE",
    "build_exact_array_declarations",
    "build_exact_manifest",
    "collect_run_arrays",
    "prefixed_specs",
    "validate_exact_manifest",
]
