"""Exact logical array contract for maintained bout-classification runs."""

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
    INT32,
    INT64,
    UINT8,
    ArrayContract,
)
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode

BOUT_CLASSIFICATION_RUN_SCHEMA_ID = "analysis.bout_classification_runs"
BOUT_CLASSIFICATION_RUN_SCHEMA_VERSION = 2
BOUT_CLASSIFICATION_LEGACY_SCHEMA_VERSION = 1
BOUT_CLASSIFICATION_ARRAY_SCHEMA_ID = (
    "palette.analysis.bout_classification.per_bout_arrays"
)
BOUT_CLASSIFICATION_ARRAY_SCHEMA_VERSION = 1
BOUT_CLASSIFICATION_ARRAY_SCHEMA_ATTR = "bout_classification_array_schema"
BOUT_CLASSIFICATION_ARRAY_SCHEMA_DIGEST_ATTR = (
    f"{BOUT_CLASSIFICATION_ARRAY_SCHEMA_ATTR}_sha256"
)
BOUT_CLASSIFICATION_PHYSICAL_POLICY_OWNER = "columnar_store_array_v1"
BOUT_CLASSIFICATION_BYTE_PLANNER_OWNER = "analysis_storage_planning_v1"
CATEGORY_LABEL_BYTES_WIDTH = 64
FAILURE_REASON_BYTES_WIDTH = 128


@dataclass(frozen=True)
class BoutClassificationDimensions:
    n_bouts: int

    def __post_init__(self) -> None:
        if type(self.n_bouts) is not int or self.n_bouts < 0:
            raise ValueError("n_bouts must be an exact nonnegative integer.")

    @property
    def contract_dimensions(self) -> dict[str, int]:
        return {"n_bouts": self.n_bouts}


@dataclass(frozen=True)
class BoutClassificationSchemaIssue:
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
) -> ArrayContract:
    return ArrayContract(
        schema_id="palette.array.bout_classification." + path.replace("/", "."),
        schema_version=1,
        dtype=dtype,
        shape_template=shape,
        axis_names=axes,
        description=f"Maintained bout-classification array {path}.",
        units=units,
    )


def _declaration(
    name: str,
    dtype: Any,
    shape: tuple[str | int, ...],
    axes: tuple[str, ...],
    *,
    authority: AnalysisAuthorityRole,
    fill: str,
    null: str,
    units: str | None = None,
    byte_planner_adopted: bool = False,
) -> AnalysisArrayDeclaration:
    path = f"per_bout/{name}"
    return AnalysisArrayDeclaration(
        path=path,
        contract=_contract(path, dtype, shape, axes, units=units),
        required=True,
        access_pattern=AccessPattern.EAGER,
        write_mode=WriteMode.IMMUTABLE,
        authority_role=authority,
        fill_semantics=fill,
        null_semantics=null,
        physical_policy_owner=(
            BOUT_CLASSIFICATION_BYTE_PLANNER_OWNER
            if byte_planner_adopted
            else BOUT_CLASSIFICATION_PHYSICAL_POLICY_OWNER
        ),
        byte_planner_adopted=byte_planner_adopted,
    )


def build_bout_classification_array_declarations(
    *, byte_planner_adopted: bool = False
) -> tuple[AnalysisArrayDeclaration, ...]:
    lineage = AnalysisAuthorityRole.LINEAGE_INDEX
    authority = AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY
    quality = AnalysisAuthorityRole.QUALITY_DIAGNOSTIC
    rows = ("n_bouts",)
    axis = ("swim_bout",)
    declarations = (
        _declaration(
            "source_bout_id",
            INT64,
            rows,
            axis,
            authority=lineage,
            fill="every row stores the exact source swim-bout identity; no numeric sentinel",
            null="all source rows are present",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "start_frame",
            INT64,
            rows,
            axis,
            authority=lineage,
            fill="inclusive nonnegative source start frame; no numeric sentinel",
            null="all source rows are present",
            units="camera_frame",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "end_frame",
            INT64,
            rows,
            axis,
            authority=lineage,
            fill="inclusive nonnegative source end frame; no numeric sentinel",
            null="all source rows are present",
            units="camera_frame",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "window_start_frame",
            INT64,
            rows,
            axis,
            authority=lineage,
            fill="inclusive classifier-window start frame; no numeric sentinel",
            null="all source rows are present",
            units="camera_frame",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "window_end_frame",
            INT64,
            rows,
            axis,
            authority=lineage,
            fill="inclusive classifier-window end frame; no numeric sentinel",
            null="all source rows are present",
            units="camera_frame",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "HB1_frame",
            INT64,
            rows,
            axis,
            authority=authority,
            fill="-1 exactly when classified is false; otherwise absolute first-half-beat frame",
            null="classified is the validity bitmap",
            units="camera_frame",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "HB1_offset_frames",
            INT32,
            rows,
            axis,
            authority=authority,
            fill="-1 exactly when classified is false; otherwise offset from window_start_frame",
            null="classified is the validity bitmap",
            units="frame",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "category_id",
            INT32,
            rows,
            axis,
            authority=authority,
            fill="-1 exactly when classified is false; nonnegative classifier category otherwise",
            null="classified is the validity bitmap",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "category_label_bytes",
            UINT8,
            ("n_bouts", CATEGORY_LABEL_BYTES_WIDTH),
            ("swim_bout", "utf8_byte"),
            authority=authority,
            fill="UTF-8 label followed by NUL and zero padding to exactly 64 bytes",
            null="empty labels are forbidden; skipped rows use skipped_invalid_window",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "subcategory_id",
            INT32,
            rows,
            axis,
            authority=authority,
            fill="-1 exactly when classified is false; classifier subcategory otherwise",
            null="classified is the validity bitmap",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "sign",
            INT32,
            rows,
            axis,
            authority=authority,
            fill="zero when classified is false; classifier sign otherwise",
            null="classified is the validity bitmap",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "probability",
            FLOAT32,
            rows,
            axis,
            authority=authority,
            fill="NaN exactly when classified is false; finite classifier probability otherwise",
            null="classified is the validity bitmap",
            units="probability",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "tail_valid_fraction",
            FLOAT32,
            rows,
            axis,
            authority=quality,
            fill="finite fraction in [0,1] for every source window",
            null="all source rows are present",
            units="fraction",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "traj_valid_fraction",
            FLOAT32,
            rows,
            axis,
            authority=quality,
            fill="finite fraction in [0,1] for every source window",
            null="all source rows are present",
            units="fraction",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "max_consecutive_tail_invalid",
            INT32,
            rows,
            axis,
            authority=quality,
            fill="nonnegative invalid-frame run length; no numeric sentinel",
            null="all source rows are present",
            units="frame",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "max_consecutive_traj_invalid",
            INT32,
            rows,
            axis,
            authority=quality,
            fill="nonnegative invalid-frame run length; no numeric sentinel",
            null="all source rows are present",
            units="frame",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "source_window_valid",
            BOOL,
            rows,
            axis,
            authority=quality,
            fill="false means the source window failed the declared eligibility policy",
            null="all source rows are present",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "classified",
            BOOL,
            rows,
            axis,
            authority=quality,
            fill="false means the classifier was not run for this source row",
            null="all source rows are present",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "valid",
            BOOL,
            rows,
            axis,
            authority=quality,
            fill="true means the classification is usable downstream",
            null="all source rows are present",
            byte_planner_adopted=byte_planner_adopted,
        ),
        _declaration(
            "failure_reason_bytes",
            UINT8,
            ("n_bouts", FAILURE_REASON_BYTES_WIDTH),
            ("swim_bout", "utf8_byte"),
            authority=quality,
            fill="UTF-8 reason followed by NUL and zero padding to exactly 128 bytes",
            null="empty reasons are forbidden; usable rows store ok",
            byte_planner_adopted=byte_planner_adopted,
        ),
    )
    paths = tuple(item.path for item in declarations)
    if len(paths) != 20 or len(paths) != len(set(paths)):
        raise RuntimeError("Bout-classification schema must contain 20 unique arrays.")
    return declarations


BOUT_CLASSIFICATION_ARRAY_DECLARATIONS = build_bout_classification_array_declarations()
BOUT_CLASSIFICATION_CANDIDATE_ARRAY_DECLARATIONS = (
    build_bout_classification_array_declarations(byte_planner_adopted=True)
)
BOUT_CLASSIFICATION_ACCESS_UNIT_SEMANTICS = {
    declaration.path: "one complete swim-bout row including all trailing fields"
    for declaration in BOUT_CLASSIFICATION_CANDIDATE_ARRAY_DECLARATIONS
}
BOUT_CLASSIFICATION_FILL_VALUES = {
    "per_bout/source_bout_id": 0,
    "per_bout/start_frame": 0,
    "per_bout/end_frame": 0,
    "per_bout/window_start_frame": 0,
    "per_bout/window_end_frame": 0,
    "per_bout/HB1_frame": -1,
    "per_bout/HB1_offset_frames": -1,
    "per_bout/category_id": -1,
    "per_bout/category_label_bytes": 0,
    "per_bout/subcategory_id": -1,
    "per_bout/sign": 0,
    "per_bout/probability": float("nan"),
    "per_bout/tail_valid_fraction": 0.0,
    "per_bout/traj_valid_fraction": 0.0,
    "per_bout/max_consecutive_tail_invalid": 0,
    "per_bout/max_consecutive_traj_invalid": 0,
    "per_bout/source_window_valid": False,
    "per_bout/classified": False,
    "per_bout/valid": False,
    "per_bout/failure_reason_bytes": 0,
}
BOUT_CLASSIFICATION_FIELD_NAMES = tuple(
    item.path.removeprefix("per_bout/")
    for item in BOUT_CLASSIFICATION_ARRAY_DECLARATIONS
)
BOUT_CLASSIFICATION_FIELD_DTYPES = {
    "source_bout_id": "int64",
    "start_frame": "int64",
    "end_frame": "int64",
    "window_start_frame": "int64",
    "window_end_frame": "int64",
    "HB1_frame": "int64",
    "HB1_offset_frames": "int32",
    "category_id": "int32",
    "category_label_bytes": f"|S{CATEGORY_LABEL_BYTES_WIDTH}",
    "subcategory_id": "int32",
    "sign": "int32",
    "probability": "float32",
    "tail_valid_fraction": "float32",
    "traj_valid_fraction": "float32",
    "max_consecutive_tail_invalid": "int32",
    "max_consecutive_traj_invalid": "int32",
    "source_window_valid": "bool",
    "classified": "bool",
    "valid": "bool",
    "failure_reason_bytes": f"|S{FAILURE_REASON_BYTES_WIDTH}",
}


def bout_classification_array_schema_manifest(
    dimensions: BoutClassificationDimensions,
    *,
    byte_planner_adopted: bool = False,
) -> dict[str, object]:
    return {
        "schema_id": BOUT_CLASSIFICATION_ARRAY_SCHEMA_ID,
        "schema_version": BOUT_CLASSIFICATION_ARRAY_SCHEMA_VERSION,
        "run_schema_id": BOUT_CLASSIFICATION_RUN_SCHEMA_ID,
        "run_schema_version": BOUT_CLASSIFICATION_RUN_SCHEMA_VERSION,
        "dimensions": dimensions.contract_dimensions,
        "arrays": [
            item.as_manifest()
            for item in build_bout_classification_array_declarations(
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


def bout_classification_manifest_digest(manifest: Any) -> str:
    return hashlib.sha256(canonical_exact_json_bytes(manifest)).hexdigest()


def collect_bout_classification_arrays(run_group: Any) -> dict[str, Any]:
    per_bout = run_group.get("per_bout")
    if per_bout is None:
        return {}
    return {f"per_bout/{name}": per_bout[name] for name in per_bout.array_keys()}


def validate_bout_classification_arrays(
    run_group: Any,
    *,
    dimensions: BoutClassificationDimensions,
) -> tuple[BoutClassificationSchemaIssue, ...]:
    issues: list[BoutClassificationSchemaIssue] = []
    persisted = run_group.attrs.get(BOUT_CLASSIFICATION_ARRAY_SCHEMA_ATTR)
    persisted_adopted = (
        persisted.get("byte_planner_adopted") if isinstance(persisted, dict) else None
    )
    byte_planner_adopted = persisted_adopted is True
    expected = {
        item.path: item
        for item in build_bout_classification_array_declarations(
            byte_planner_adopted=byte_planner_adopted
        )
    }
    arrays = collect_bout_classification_arrays(run_group)
    for path in sorted(set(arrays) - set(expected)):
        issues.append(
            BoutClassificationSchemaIssue(
                "unexpected_array", path, "Array is outside maintained schema v2."
            )
        )
    for path, declaration in expected.items():
        array = arrays.get(path)
        if array is None:
            issues.append(
                BoutClassificationSchemaIssue(
                    "missing_required_array", path, "Required array is absent."
                )
            )
            continue
        for error in declaration.contract.validate_observation(
            array, dimensions=dimensions.contract_dimensions
        ):
            issues.append(
                BoutClassificationSchemaIssue("array_contract_violation", path, error)
            )

    per_bout = run_group.get("per_bout")
    if per_bout is not None:
        observed_names = per_bout.attrs.get("field_names")
        if type(observed_names) is not list or tuple(observed_names) != (
            BOUT_CLASSIFICATION_FIELD_NAMES
        ):
            issues.append(
                BoutClassificationSchemaIssue(
                    "field_names_mismatch",
                    "per_bout",
                    "field_names must exactly match the frozen ordered inventory.",
                )
            )
        observed_dtypes = per_bout.attrs.get("field_dtypes")
        if type(observed_dtypes) is not dict or observed_dtypes != (
            BOUT_CLASSIFICATION_FIELD_DTYPES
        ):
            issues.append(
                BoutClassificationSchemaIssue(
                    "field_dtypes_mismatch",
                    "per_bout",
                    "field_dtypes must exactly match the frozen logical table dtypes.",
                )
            )

    expected_manifest = bout_classification_array_schema_manifest(
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
            BoutClassificationSchemaIssue(
                "array_schema_manifest_mismatch",
                BOUT_CLASSIFICATION_ARRAY_SCHEMA_ATTR,
                "Persisted declaration must exactly equal the executable schema.",
            )
        )
    expected_digest = bout_classification_manifest_digest(expected_manifest)
    if (
        run_group.attrs.get(BOUT_CLASSIFICATION_ARRAY_SCHEMA_DIGEST_ATTR)
        != expected_digest
    ):
        issues.append(
            BoutClassificationSchemaIssue(
                "array_schema_digest_mismatch",
                BOUT_CLASSIFICATION_ARRAY_SCHEMA_DIGEST_ATTR,
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
            declarations=BOUT_CLASSIFICATION_CANDIDATE_ARRAY_DECLARATIONS,
            access_unit_semantics=BOUT_CLASSIFICATION_ACCESS_UNIT_SEMANTICS,
            fill_values=BOUT_CLASSIFICATION_FILL_VALUES,
            dimensions=dimensions.contract_dimensions,
        ):
            issues.append(
                BoutClassificationSchemaIssue(
                    "storage_plan_mismatch",
                    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
                    message,
                )
            )
    elif run_group.attrs.get(ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR) is not None:
        issues.append(
            BoutClassificationSchemaIssue(
                "storage_plan_mismatch",
                ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
                "Legacy physical declaration cannot carry a candidate receipt.",
            )
        )
    return tuple(issues)


def write_bout_classification_array_schema_manifest(
    run_group: Any,
    *,
    n_bouts: int,
    byte_planner_adopted: bool = False,
) -> dict[str, object]:
    manifest = bout_classification_array_schema_manifest(
        BoutClassificationDimensions(n_bouts=n_bouts),
        byte_planner_adopted=byte_planner_adopted,
    )
    run_group.attrs[BOUT_CLASSIFICATION_ARRAY_SCHEMA_ATTR] = manifest
    run_group.attrs[BOUT_CLASSIFICATION_ARRAY_SCHEMA_DIGEST_ATTR] = (
        bout_classification_manifest_digest(manifest)
    )
    return manifest
