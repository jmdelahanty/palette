"""Exact logical array schema for maintained tail-kinematics snapshots.

The scientific writer owns the meaning of these arrays.  This module closes
their path, dtype, shape, null, lifecycle, and access declarations without
selecting a physical storage profile.  The optional refined-mask revision
snapshot is one atomic bundle: candidate writers either publish both arrays or
neither array.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fisheye.shared.zarr.analysis_array_contracts import (
    AnalysisArrayDeclaration,
    AnalysisAuthorityRole,
)
from fisheye.shared.zarr.array_contracts import (
    BOOL,
    FLOAT32,
    INT64,
    UINT8,
    UINT64,
    ArrayContract,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode


TAIL_KINEMATICS_ARRAY_SCHEMA_ID = "palette.analysis.tail_kinematics.arrays"
TAIL_KINEMATICS_ARRAY_SCHEMA_VERSION = 1
TAIL_KINEMATICS_ARRAY_SCHEMA_ATTR = "tail_kinematics_array_schema"
TAIL_KINEMATICS_ARRAY_SCHEMA_DIGEST_ATTR = f"{TAIL_KINEMATICS_ARRAY_SCHEMA_ATTR}_sha256"
TAIL_KINEMATICS_LEGACY_PHYSICAL_POLICY_OWNER = "tail_kinematics_process_shards_v1"
TAIL_KINEMATICS_BYTE_PLANNER_OWNER = "analysis_storage_planning_v1"
TAIL_KINEMATICS_REASON_BYTES_WIDTH = 64
TAIL_KINEMATICS_SOURCE_REVISION_BUNDLE = "source_refined_subject_masks_revision"


@dataclass(frozen=True)
class TailKinematicsDimensions:
    """Concrete cardinalities shared by one exact tail-kinematics snapshot."""

    n_rows: int
    n_tail_samples: int
    n_components: int | None = None

    def __post_init__(self) -> None:
        if type(self.n_rows) is not int or self.n_rows < 0:
            raise ValueError("n_rows must be an exact nonnegative integer.")
        if type(self.n_tail_samples) is not int or self.n_tail_samples < 2:
            raise ValueError("n_tail_samples must be an exact integer >= 2.")
        if self.n_components is not None and (
            type(self.n_components) is not int or self.n_components <= 0
        ):
            raise ValueError("n_components must be null or an exact positive integer.")

    @property
    def include_source_revision_bundle(self) -> bool:
        return self.n_components is not None

    @property
    def contract_dimensions(self) -> dict[str, int]:
        dimensions = {
            "n_rows": self.n_rows,
            "n_tail_samples": self.n_tail_samples,
        }
        if self.n_components is not None:
            dimensions["n_components"] = self.n_components
        return dimensions


def _contract(
    path: str,
    dtype: Any,
    shape: tuple[str | int, ...],
    axes: tuple[str, ...],
    *,
    units: str | None = None,
    coordinate_space: str | None = None,
) -> ArrayContract:
    return ArrayContract(
        schema_id="palette.array.tail_kinematics." + path.replace("/", "."),
        schema_version=1,
        dtype=dtype,
        shape_template=shape,
        axis_names=axes,
        description=f"Maintained tail-kinematics array {path}.",
        units=units,
        coordinate_space=coordinate_space,
    )


def _declaration(
    path: str,
    dtype: Any,
    shape: tuple[str | int, ...],
    axes: tuple[str, ...],
    *,
    authority: AnalysisAuthorityRole,
    fill: str,
    null: str,
    access: AccessPattern = AccessPattern.WINDOWED,
    required: bool = True,
    units: str | None = None,
    coordinate_space: str | None = None,
    byte_planner_adopted: bool = False,
) -> AnalysisArrayDeclaration:
    return AnalysisArrayDeclaration(
        path=path,
        contract=_contract(
            path,
            dtype,
            shape,
            axes,
            units=units,
            coordinate_space=coordinate_space,
        ),
        required=required,
        access_pattern=access,
        write_mode=WriteMode.IMMUTABLE,
        authority_role=authority,
        fill_semantics=fill,
        null_semantics=null,
        physical_policy_owner=(
            TAIL_KINEMATICS_BYTE_PLANNER_OWNER
            if byte_planner_adopted
            else TAIL_KINEMATICS_LEGACY_PHYSICAL_POLICY_OWNER
        ),
        byte_planner_adopted=byte_planner_adopted,
    )


def build_tail_kinematics_array_declarations(
    *,
    include_source_revision_bundle: bool,
    byte_planner_adopted: bool = False,
) -> tuple[AnalysisArrayDeclaration, ...]:
    """Return the closed declaration set for one supported bundle selection."""

    lineage = AnalysisAuthorityRole.LINEAGE_INDEX
    quality = AnalysisAuthorityRole.QUALITY_DIAGNOSTIC
    scientific = AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY
    row = ("n_rows",)
    observation = ("observation",)
    declarations = [
        _declaration(
            "instance_key",
            UINT64,
            row,
            observation,
            authority=lineage,
            fill="physical zero is overwritten for every row; zero is not a null sentinel",
            null="all rows store one unique upstream observation identity",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "source_crop_row_ids",
            INT64,
            row,
            observation,
            authority=lineage,
            fill="physical zero is overwritten for every row; zero is not a null sentinel",
            null="all rows store one nonnegative canonical crop-row identity",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "source_acquisition_frame_index",
            INT64,
            row,
            observation,
            authority=lineage,
            fill="physical zero is overwritten for every row; zero is not a null sentinel",
            null="all rows store one nonnegative source-camera frame index",
            units="camera_frame",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "valid",
            BOOL,
            row,
            observation,
            authority=quality,
            fill="false until the bounded serial writer resolves the row",
            null="false requires a nonempty failure reason and NaN floating payloads",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "failure_reason_bytes",
            UINT8,
            ("n_rows", TAIL_KINEMATICS_REASON_BYTES_WIDTH),
            ("observation", "utf8_byte"),
            authority=quality,
            fill="zero bytes encode an initially empty NUL-terminated reason buffer",
            null="valid rows store ok; invalid rows store a nonempty UTF-8 reason",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "tail_angle_sample_s",
            FLOAT32,
            ("n_tail_samples",),
            ("tail_sample",),
            authority=scientific,
            fill="physical zero is overwritten by the complete finite sample grid",
            null="no sample-axis positions are nullable",
            access=AccessPattern.EAGER,
            units="normalized_tail_arclength",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "tail_angle_sample_xy",
            FLOAT32,
            ("n_rows", "n_tail_samples", 2),
            ("observation", "tail_sample", "xy"),
            authority=scientific,
            fill="NaN represents unavailable geometry before or after an invalid row",
            null="valid is the row validity bitmap",
            units="px",
            coordinate_space="source_camera_pixels",
            byte_planner_adopted=byte_planner_adopted,
        ),
    ]
    for path, units in (
        ("tail_angle_rad", "rad"),
        ("tail_angle_deg", "deg"),
        ("tail_lateral_deflection_px", "px"),
        ("tail_curvature_px_inv", "px^-1"),
    ):
        declarations.append(
            _declaration(
                path,
                FLOAT32,
                ("n_rows", "n_tail_samples"),
                ("observation", "tail_sample"),
                authority=scientific,
                fill="NaN represents an unavailable metric for an invalid row",
                null="valid is the row validity bitmap",
                units=units,
                byte_planner_adopted=byte_planner_adopted,
            )
        )
    for path, units in (
        ("tail_tip_angle_rad", "rad"),
        ("tail_tip_angle_deg", "deg"),
        ("tail_tip_lateral_deflection_px", "px"),
        ("max_abs_tail_angle_rad", "rad"),
        ("max_abs_tail_angle_deg", "deg"),
        ("tail_angle_rms_rad", "rad"),
        ("tail_angle_rms_deg", "deg"),
        ("integrated_abs_tail_angle_rad", "rad"),
        ("max_abs_tail_curvature_px_inv", "px^-1"),
        ("integrated_abs_tail_curvature", "px^-1"),
    ):
        declarations.append(
            _declaration(
                path,
                FLOAT32,
                row,
                observation,
                authority=scientific,
                fill="NaN represents an unavailable metric for an invalid row",
                null="valid is the row validity bitmap",
                units=units,
                byte_planner_adopted=byte_planner_adopted,
            )
        )
    if include_source_revision_bundle:
        declarations.extend(
            (
                _declaration(
                    "source_refined_subject_masks/row_revision",
                    INT64,
                    ("n_rows", "n_components"),
                    ("observation", "component"),
                    authority=lineage,
                    fill="zero is the exact initial revision and is overwritten from the source snapshot",
                    null="availability is declared independently per component",
                    required=False,
                    byte_planner_adopted=byte_planner_adopted,
                ),
                _declaration(
                    "source_refined_subject_masks/row_revision_available",
                    BOOL,
                    ("n_components",),
                    ("component",),
                    authority=lineage,
                    fill="false means the source did not expose a revision for that component",
                    null="all declared components have one availability value",
                    access=AccessPattern.EAGER,
                    required=False,
                    byte_planner_adopted=byte_planner_adopted,
                ),
            )
        )
    return tuple(sorted(declarations, key=lambda declaration: declaration.path))


def tail_kinematics_array_shapes_and_dtypes(
    dimensions: TailKinematicsDimensions,
) -> dict[str, tuple[tuple[int, ...], Any]]:
    """Resolve exact concrete shape/dtype facts from the frozen declarations."""

    declarations = build_tail_kinematics_array_declarations(
        include_source_revision_bundle=dimensions.include_source_revision_bundle,
        byte_planner_adopted=True,
    )
    resolved = dimensions.contract_dimensions
    result: dict[str, tuple[tuple[int, ...], Any]] = {}
    for declaration in declarations:
        shape = tuple(
            resolved[extent] if isinstance(extent, str) else int(extent)
            for extent in declaration.contract.shape_template
        )
        dtype = declaration.contract.dtype.numpy_dtype
        if dtype is None:
            raise ValueError(f"{declaration.path}: fixed-width dtype is required.")
        result[declaration.path] = (shape, dtype)
    return result


def tail_kinematics_fill_values(
    *, include_source_revision_bundle: bool
) -> dict[str, object]:
    """Return exact physical fills matching the logical null declarations."""

    values: dict[str, object] = {
        "instance_key": 0,
        "source_crop_row_ids": 0,
        "source_acquisition_frame_index": 0,
        "valid": False,
        "failure_reason_bytes": 0,
        "tail_angle_sample_s": 0.0,
        "tail_angle_sample_xy": float("nan"),
        "tail_angle_rad": float("nan"),
        "tail_angle_deg": float("nan"),
        "tail_lateral_deflection_px": float("nan"),
        "tail_curvature_px_inv": float("nan"),
        "tail_tip_angle_rad": float("nan"),
        "tail_tip_angle_deg": float("nan"),
        "tail_tip_lateral_deflection_px": float("nan"),
        "max_abs_tail_angle_rad": float("nan"),
        "max_abs_tail_angle_deg": float("nan"),
        "tail_angle_rms_rad": float("nan"),
        "tail_angle_rms_deg": float("nan"),
        "integrated_abs_tail_angle_rad": float("nan"),
        "max_abs_tail_curvature_px_inv": float("nan"),
        "integrated_abs_tail_curvature": float("nan"),
    }
    if include_source_revision_bundle:
        values.update(
            {
                "source_refined_subject_masks/row_revision": 0,
                "source_refined_subject_masks/row_revision_available": False,
            }
        )
    return values


def tail_kinematics_access_unit_semantics(
    *, include_source_revision_bundle: bool
) -> dict[str, str]:
    """Name the full trailing record preserved by the byte planner."""

    declarations = build_tail_kinematics_array_declarations(
        include_source_revision_bundle=include_source_revision_bundle,
        byte_planner_adopted=True,
    )
    return {
        declaration.path: (
            "one complete tail-kinematics observation including every trailing "
            "sample/vector/byte field; eager axes preserve one scalar position"
        )
        for declaration in declarations
    }


def _array_at_path(run_group: Any, path: str) -> Any | None:
    node = run_group
    for component in path.split("/"):
        node = node.get(component)
        if node is None:
            return None
    return node


def infer_tail_kinematics_dimensions(run_group: Any) -> TailKinematicsDimensions:
    """Infer dimensions while enforcing the optional bundle atomically."""

    sample_s = _array_at_path(run_group, "tail_angle_sample_s")
    valid = _array_at_path(run_group, "valid")
    if sample_s is None or valid is None:
        raise ValueError("Tail-kinematics core arrays are absent.")
    revision = _array_at_path(run_group, "source_refined_subject_masks/row_revision")
    available = _array_at_path(
        run_group, "source_refined_subject_masks/row_revision_available"
    )
    if (revision is None) != (available is None):
        raise ValueError("Tail-kinematics source-revision optional bundle is partial.")
    n_components: int | None = None
    if revision is not None:
        if len(revision.shape) != 2 or len(available.shape) != 1:
            raise ValueError(
                "Tail-kinematics source-revision bundle ranks are invalid."
            )
        n_components = int(revision.shape[1])
        if int(available.shape[0]) != n_components:
            raise ValueError("Tail-kinematics source-revision component axes disagree.")
    return TailKinematicsDimensions(
        n_rows=int(valid.shape[0]),
        n_tail_samples=int(sample_s.shape[0]),
        n_components=n_components,
    )


def build_tail_kinematics_array_schema_manifest(
    dimensions: TailKinematicsDimensions,
    *,
    byte_planner_adopted: bool,
) -> dict[str, object]:
    declarations = build_tail_kinematics_array_declarations(
        include_source_revision_bundle=dimensions.include_source_revision_bundle,
        byte_planner_adopted=byte_planner_adopted,
    )
    payload = {
        "run_schema_id": "analysis.tail_kinematics_runs",
        "run_schema_version": 2,
        "dimensions": dimensions.contract_dimensions,
        "enabled_optional_bundles": (
            [TAIL_KINEMATICS_SOURCE_REVISION_BUNDLE]
            if dimensions.include_source_revision_bundle
            else []
        ),
        "declarations": [declaration.as_manifest() for declaration in declarations],
        "closed_array_inventory": True,
    }
    return {
        "schema_id": TAIL_KINEMATICS_ARRAY_SCHEMA_ID,
        "schema_version": TAIL_KINEMATICS_ARRAY_SCHEMA_VERSION,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }


def stamp_tail_kinematics_array_schema(
    run_group: Any,
    dimensions: TailKinematicsDimensions,
    *,
    byte_planner_adopted: bool,
) -> dict[str, object]:
    manifest = build_tail_kinematics_array_schema_manifest(
        dimensions,
        byte_planner_adopted=byte_planner_adopted,
    )
    run_group.attrs[TAIL_KINEMATICS_ARRAY_SCHEMA_ATTR] = manifest
    run_group.attrs[TAIL_KINEMATICS_ARRAY_SCHEMA_DIGEST_ATTR] = manifest[
        "payload_digest"
    ]
    return manifest


def validate_tail_kinematics_array_schema(
    run_group: Any,
    *,
    byte_planner_adopted: bool,
) -> tuple[str, ...]:
    """Validate the exact live inventory and its persisted declaration receipt."""

    errors: list[str] = []
    try:
        dimensions = infer_tail_kinematics_dimensions(run_group)
    except Exception as exc:
        return (str(exc),)
    declarations = build_tail_kinematics_array_declarations(
        include_source_revision_bundle=dimensions.include_source_revision_bundle,
        byte_planner_adopted=byte_planner_adopted,
    )
    expected_paths = {declaration.path for declaration in declarations}

    observed_paths: set[str] = set()

    def visit(group: Any, prefix: str = "") -> None:
        for name, _array in group.arrays():
            observed_paths.add(f"{prefix}/{name}" if prefix else str(name))
        for name, child in group.groups():
            child_prefix = f"{prefix}/{name}" if prefix else str(name)
            visit(child, child_prefix)

    visit(run_group)
    missing = sorted(expected_paths - observed_paths)
    unexpected = sorted(observed_paths - expected_paths)
    if missing:
        errors.append(f"missing exact tail-kinematics arrays: {missing!r}")
    if unexpected:
        errors.append(f"unexpected tail-kinematics arrays: {unexpected!r}")
    for declaration in declarations:
        node = _array_at_path(run_group, declaration.path)
        if node is None:
            continue
        errors.extend(
            f"{declaration.path}: {message}"
            for message in declaration.contract.validate_observation(
                node,
                dimensions=dimensions.contract_dimensions,
            )
        )

    expected_manifest = build_tail_kinematics_array_schema_manifest(
        dimensions,
        byte_planner_adopted=byte_planner_adopted,
    )
    if run_group.attrs.get(TAIL_KINEMATICS_ARRAY_SCHEMA_ATTR) != expected_manifest:
        errors.append("persisted tail-kinematics array schema differs from live arrays")
    if (
        run_group.attrs.get(TAIL_KINEMATICS_ARRAY_SCHEMA_DIGEST_ATTR)
        != expected_manifest["payload_digest"]
    ):
        errors.append("tail-kinematics array schema digest binding differs")
    return tuple(errors)


__all__ = [
    "TAIL_KINEMATICS_ARRAY_SCHEMA_ATTR",
    "TAIL_KINEMATICS_ARRAY_SCHEMA_DIGEST_ATTR",
    "TAIL_KINEMATICS_ARRAY_SCHEMA_ID",
    "TAIL_KINEMATICS_ARRAY_SCHEMA_VERSION",
    "TAIL_KINEMATICS_SOURCE_REVISION_BUNDLE",
    "TailKinematicsDimensions",
    "build_tail_kinematics_array_declarations",
    "build_tail_kinematics_array_schema_manifest",
    "infer_tail_kinematics_dimensions",
    "stamp_tail_kinematics_array_schema",
    "tail_kinematics_access_unit_semantics",
    "tail_kinematics_array_shapes_and_dtypes",
    "tail_kinematics_fill_values",
    "validate_tail_kinematics_array_schema",
]
