"""Exact logical array contract for maintained tail-posture compatibility views."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
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
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode

TAIL_POSTURE_VIEW_RUN_SCHEMA_ID = "analysis.tail_posture_view_runs"
TAIL_POSTURE_VIEW_RUN_SCHEMA_VERSION = 3
TAIL_POSTURE_VIEW_LEGACY_SCHEMA_VERSION = 2
TAIL_POSTURE_VIEW_ARRAY_SCHEMA_ID = "palette.analysis.tail_posture_view.arrays"
TAIL_POSTURE_VIEW_ARRAY_SCHEMA_VERSION = 1
TAIL_POSTURE_VIEW_ARRAY_SCHEMA_ATTR = "tail_posture_view_array_schema"
TAIL_POSTURE_VIEW_ARRAY_SCHEMA_DIGEST_ATTR = (
    f"{TAIL_POSTURE_VIEW_ARRAY_SCHEMA_ATTR}_sha256"
)
TAIL_POSTURE_VIEW_PHYSICAL_POLICY_OWNER = (
    "refined_subject_mask_metric_row_chunk_compatibility"
)
TAIL_POSTURE_VIEW_BYTE_PLANNER_OWNER = "analysis_storage_planning_v1"
TAIL_POSTURE_FAILURE_REASON_BYTES_WIDTH = 64


@dataclass(frozen=True)
class TailPostureViewDimensions:
    n_rows: int
    n_keypoints: int
    n_angles: int

    def __post_init__(self) -> None:
        if type(self.n_rows) is not int or self.n_rows < 0:
            raise ValueError("n_rows must be an exact nonnegative integer.")
        if type(self.n_keypoints) is not int or self.n_keypoints < 2:
            raise ValueError("n_keypoints must be an exact integer >= 2.")
        if type(self.n_angles) is not int or self.n_angles != self.n_keypoints - 1:
            raise ValueError("n_angles must exactly equal n_keypoints - 1.")

    @property
    def contract_dimensions(self) -> dict[str, int]:
        return {
            "n_rows": self.n_rows,
            "n_keypoints": self.n_keypoints,
            "n_angles": self.n_angles,
        }


@dataclass(frozen=True)
class TailPostureViewSchemaIssue:
    code: str
    path: str
    message: str


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
        schema_id="palette.array.tail_posture_view." + path.replace("/", "."),
        schema_version=1,
        dtype=dtype,
        shape_template=shape,
        axis_names=axes,
        description=f"Maintained tail-posture view array {path}.",
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
        required=True,
        access_pattern=AccessPattern.WINDOWED,
        write_mode=WriteMode.IMMUTABLE,
        authority_role=authority,
        fill_semantics=fill,
        null_semantics=null,
        physical_policy_owner=(
            TAIL_POSTURE_VIEW_BYTE_PLANNER_OWNER
            if byte_planner_adopted
            else TAIL_POSTURE_VIEW_PHYSICAL_POLICY_OWNER
        ),
        byte_planner_adopted=byte_planner_adopted,
    )


def build_tail_posture_view_array_declarations(
    *, byte_planner_adopted: bool = False
) -> tuple[AnalysisArrayDeclaration, ...]:
    lineage = AnalysisAuthorityRole.LINEAGE_INDEX
    view = AnalysisAuthorityRole.COMPATIBILITY_ALIAS
    quality = AnalysisAuthorityRole.QUALITY_DIAGNOSTIC
    rows = ("n_rows",)
    observation = ("observation",)
    declarations = (
        _declaration(
            "instance_key",
            UINT64,
            rows,
            observation,
            authority=lineage,
            fill="every row stores one unique upstream observation identity; no sentinel",
            null="all rows are present",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "source_crop_row_ids",
            INT64,
            rows,
            observation,
            authority=lineage,
            fill="every row stores a nonnegative canonical crop-row identity; no sentinel",
            null="all rows are present",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "source_acquisition_frame_index",
            INT64,
            rows,
            observation,
            authority=lineage,
            fill="every row stores a nonnegative acquisition-frame index; no sentinel",
            null="all rows are present",
            units="camera_frame",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "valid",
            BOOL,
            rows,
            observation,
            authority=quality,
            fill="false means every floating posture payload for the row is NaN",
            null="all rows are present",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "failure_reason_bytes",
            UINT8,
            ("n_rows", TAIL_POSTURE_FAILURE_REASON_BYTES_WIDTH),
            ("observation", "utf8_byte"),
            authority=quality,
            fill="UTF-8 reason followed by NUL and zero padding to exactly 64 bytes",
            null="valid rows store ok; invalid rows store a nonempty reason",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "head_xy",
            FLOAT32,
            ("n_rows", 2),
            ("observation", "xy"),
            authority=view,
            fill="both coordinates are NaN exactly when valid is false",
            null="valid is the validity bitmap",
            units="px",
            coordinate_space="source_camera_pixels",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "head_yaw_rad",
            FLOAT32,
            rows,
            observation,
            authority=view,
            fill="NaN exactly when valid is false",
            null="valid is the validity bitmap",
            units="rad",
            coordinate_space="source_camera_image_xy_heading",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "tail_keypoints_xy",
            FLOAT32,
            ("n_rows", "n_keypoints", 2),
            ("observation", "tail_keypoint", "xy"),
            authority=view,
            fill="all coordinates are NaN exactly when valid is false",
            null="valid is the validity bitmap",
            units="px",
            coordinate_space="source_camera_pixels",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "tail_angle_rad",
            FLOAT32,
            ("n_rows", "n_angles"),
            ("observation", "cumulative_segment_angle"),
            authority=view,
            fill="all angles are NaN exactly when valid is false",
            null="valid is the validity bitmap",
            units="rad",
            coordinate_space="megabouts_cumulative_segment_angle",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "tail_angle_deg",
            FLOAT32,
            ("n_rows", "n_angles"),
            ("observation", "cumulative_segment_angle"),
            authority=view,
            fill="all angles are NaN exactly when valid is false",
            null="valid is the validity bitmap",
            units="deg",
            coordinate_space="megabouts_cumulative_segment_angle",
            byte_planner_adopted=byte_planner_adopted,
        ),
    )
    paths = tuple(item.path for item in declarations)
    if len(paths) != 10 or len(paths) != len(set(paths)):
        raise RuntimeError("Tail-posture view schema must contain 10 unique arrays.")
    return declarations


TAIL_POSTURE_VIEW_ARRAY_DECLARATIONS = build_tail_posture_view_array_declarations()
TAIL_POSTURE_VIEW_CANDIDATE_ARRAY_DECLARATIONS = (
    build_tail_posture_view_array_declarations(byte_planner_adopted=True)
)
TAIL_POSTURE_VIEW_ACCESS_UNIT_SEMANTICS = {
    declaration.path: "one complete observation row including all trailing fields"
    for declaration in TAIL_POSTURE_VIEW_CANDIDATE_ARRAY_DECLARATIONS
}
TAIL_POSTURE_VIEW_FILL_VALUES = {
    "instance_key": 0,
    "source_crop_row_ids": 0,
    "source_acquisition_frame_index": 0,
    "valid": False,
    "failure_reason_bytes": 0,
    "head_xy": float("nan"),
    "head_yaw_rad": float("nan"),
    "tail_keypoints_xy": float("nan"),
    "tail_angle_rad": float("nan"),
    "tail_angle_deg": float("nan"),
}


def tail_posture_view_array_schema_manifest(
    dimensions: TailPostureViewDimensions,
    *,
    byte_planner_adopted: bool = False,
) -> dict[str, object]:
    return {
        "schema_id": TAIL_POSTURE_VIEW_ARRAY_SCHEMA_ID,
        "schema_version": TAIL_POSTURE_VIEW_ARRAY_SCHEMA_VERSION,
        "run_schema_id": TAIL_POSTURE_VIEW_RUN_SCHEMA_ID,
        "run_schema_version": TAIL_POSTURE_VIEW_RUN_SCHEMA_VERSION,
        "dimensions": dimensions.contract_dimensions,
        "arrays": [
            item.as_manifest()
            for item in build_tail_posture_view_array_declarations(
                byte_planner_adopted=byte_planner_adopted
            )
        ],
        "byte_planner_adopted": byte_planner_adopted,
    }


def canonical_exact_json_bytes(value: Any, *, path: str = "$") -> bytes:
    def require_exact_json(item: Any, item_path: str) -> None:
        if item is None or type(item) in {str, bool, int}:
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError(f"{item_path} contains a non-finite float.")
            return
        if type(item) is list:
            for index, child in enumerate(item):
                require_exact_json(child, f"{item_path}[{index}]")
            return
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError(f"{item_path} contains a non-string key.")
                require_exact_json(child, f"{item_path}.{key}")
            return
        raise ValueError(
            f"{item_path} contains non-canonical JSON type {type(item).__name__}."
        )

    require_exact_json(value, path)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def tail_posture_view_manifest_digest(manifest: Any) -> str:
    return hashlib.sha256(canonical_exact_json_bytes(manifest)).hexdigest()


def collect_tail_posture_view_arrays(run_group: Any) -> dict[str, Any]:
    return {str(name): run_group[name] for name in run_group.array_keys()}


def validate_tail_posture_view_arrays(
    run_group: Any,
    *,
    dimensions: TailPostureViewDimensions,
) -> tuple[TailPostureViewSchemaIssue, ...]:
    issues: list[TailPostureViewSchemaIssue] = []
    persisted = run_group.attrs.get(TAIL_POSTURE_VIEW_ARRAY_SCHEMA_ATTR)
    persisted_adopted = (
        persisted.get("byte_planner_adopted") if isinstance(persisted, dict) else None
    )
    byte_planner_adopted = persisted_adopted is True
    expected = {
        item.path: item
        for item in build_tail_posture_view_array_declarations(
            byte_planner_adopted=byte_planner_adopted
        )
    }
    arrays = collect_tail_posture_view_arrays(run_group)
    for path in sorted(set(arrays) - set(expected)):
        issues.append(
            TailPostureViewSchemaIssue(
                "unexpected_array",
                path,
                "Direct array is outside maintained schema v3.",
            )
        )
    for path, declaration in expected.items():
        array = arrays.get(path)
        if array is None:
            issues.append(
                TailPostureViewSchemaIssue(
                    "missing_required_array", path, "Required direct array is absent."
                )
            )
            continue
        for error in declaration.contract.validate_observation(
            array, dimensions=dimensions.contract_dimensions
        ):
            issues.append(
                TailPostureViewSchemaIssue("array_contract_violation", path, error)
            )

    expected_manifest = tail_posture_view_array_schema_manifest(
        dimensions,
        byte_planner_adopted=byte_planner_adopted,
    )
    try:
        manifest_matches = canonical_exact_json_bytes(persisted) == (
            canonical_exact_json_bytes(expected_manifest)
        )
    except (TypeError, ValueError):
        manifest_matches = False
    if not manifest_matches:
        issues.append(
            TailPostureViewSchemaIssue(
                "array_schema_manifest_mismatch",
                TAIL_POSTURE_VIEW_ARRAY_SCHEMA_ATTR,
                "Persisted declaration must exactly equal the executable schema.",
            )
        )
    expected_digest = tail_posture_view_manifest_digest(expected_manifest)
    if (
        run_group.attrs.get(TAIL_POSTURE_VIEW_ARRAY_SCHEMA_DIGEST_ATTR)
        != expected_digest
    ):
        issues.append(
            TailPostureViewSchemaIssue(
                "array_schema_digest_mismatch",
                TAIL_POSTURE_VIEW_ARRAY_SCHEMA_DIGEST_ATTR,
                "Persisted digest must bind the exact executable declaration.",
            )
        )
    from .direct_writer_storage import (
        ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
        validate_direct_writer_storage_receipt,
    )

    if byte_planner_adopted:
        for message in validate_direct_writer_storage_receipt(
            run_group,
            declarations=TAIL_POSTURE_VIEW_CANDIDATE_ARRAY_DECLARATIONS,
            access_unit_semantics=TAIL_POSTURE_VIEW_ACCESS_UNIT_SEMANTICS,
            fill_values=TAIL_POSTURE_VIEW_FILL_VALUES,
            dimensions=dimensions.contract_dimensions,
        ):
            issues.append(
                TailPostureViewSchemaIssue(
                    "storage_plan_mismatch",
                    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
                    message,
                )
            )
    elif run_group.attrs.get(ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR) is not None:
        issues.append(
            TailPostureViewSchemaIssue(
                "storage_plan_mismatch",
                ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
                "Legacy physical declaration cannot carry a candidate receipt.",
            )
        )
    return tuple(issues)


def write_tail_posture_view_array_schema_manifest(
    run_group: Any,
    *,
    n_rows: int,
    n_keypoints: int,
    n_angles: int,
    byte_planner_adopted: bool = False,
) -> dict[str, object]:
    manifest = tail_posture_view_array_schema_manifest(
        TailPostureViewDimensions(
            n_rows=n_rows,
            n_keypoints=n_keypoints,
            n_angles=n_angles,
        ),
        byte_planner_adopted=byte_planner_adopted,
    )
    run_group.attrs[TAIL_POSTURE_VIEW_ARRAY_SCHEMA_ATTR] = manifest
    run_group.attrs[TAIL_POSTURE_VIEW_ARRAY_SCHEMA_DIGEST_ATTR] = (
        tail_posture_view_manifest_digest(manifest)
    )
    return manifest
