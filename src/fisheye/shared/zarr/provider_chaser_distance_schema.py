"""Logical contract for the provider-aware chaser-distance successor.

This module is intentionally independent of Zarr I/O.  It defines the exact
flat row surface that a later materializer may publish.  Rows are always
acquisition-frame major and chaser-axis minor; the ``n_rows`` dimension is
therefore exactly ``n_frames * n_chasers``.

The schema keeps provider row identities and validity/reason arrays explicit.
It does not identify a provider as authoritative, resolve a selector, or
assign a physical interpretation to a temporal proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.array_contracts import (
    BOOL,
    FLOAT32,
    INT64,
    UINT8,
    UINT16,
    ArrayContract,
    ArrayContractBinding,
    ArrayContractCatalog,
)


PROVIDER_CHASER_DISTANCE_SCHEMA_ID = "palette.analysis.provider_chaser_distance"
PROVIDER_CHASER_DISTANCE_SCHEMA_VERSION = 1
PROVIDER_CHASER_DISTANCE_LAYOUT = "frame_x_chaser_flat_rows_v1"

_N = ("n_rows",)
_XY = ("n_rows", 2)
_ROW_AXIS = ("relative_frame_row",)
_XY_AXIS = ("relative_frame_row", "xy")


def _contract(
    name: str,
    *,
    dtype: Any,
    shape: tuple[str | int, ...],
    axes: tuple[str, ...],
    description: str,
    units: str | None = None,
    coordinate_space: str | None = None,
) -> ArrayContract:
    return ArrayContract(
        schema_id=f"palette.array.provider_chaser_distance.{name}",
        schema_version=1,
        dtype=dtype,
        shape_template=shape,
        axis_names=axes,
        description=description,
        units=units,
        coordinate_space=coordinate_space,
    )


_PIXEL_SPACE = "source_camera_continuous_pixel_xy"

_ARRAYS: tuple[tuple[str, ArrayContract], ...] = (
    (
        "acquisition_frame_id",
        _contract(
            "acquisition_frame_id",
            dtype=INT64,
            shape=_N,
            axes=_ROW_AXIS,
            description="Acquisition-camera frame identity repeated on each chaser row.",
            units="acquisition_frame_index",
        ),
    ),
    (
        "track_sample_id",
        _contract(
            "track_sample_id",
            dtype=INT64,
            shape=_N,
            axes=_ROW_AXIS,
            description="Fish tracking sample identity repeated on each chaser row.",
            units="track_sample_index",
        ),
    ),
    (
        "timestamp_ns",
        _contract(
            "timestamp_ns",
            dtype=INT64,
            shape=_N,
            axes=_ROW_AXIS,
            description="Camera acquisition timestamp, if the source timing authority provides one.",
            units="nanoseconds",
        ),
    ),
    (
        "timestamp_valid",
        _contract(
            "timestamp_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether timestamp_ns is semantically valid.",
        ),
    ),
    (
        "timestamp_reason_code",
        _contract(
            "timestamp_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with timestamp_valid.",
        ),
    ),
    (
        "source_position_row_id",
        _contract(
            "source_position_row_id",
            dtype=INT64,
            shape=_N,
            axes=_ROW_AXIS,
            description=(
                "Original row identity in the bound source-position provider; "
                "not an output row number or inferred instance key."
            ),
            units="source_provider_row_index",
        ),
    ),
    (
        "source_position_row_valid",
        _contract(
            "source_position_row_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether source_position_row_id identifies a source row.",
        ),
    ),
    (
        "source_position_row_reason_code",
        _contract(
            "source_position_row_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with source-position row validity.",
        ),
    ),
    (
        "source_position_xy_px",
        _contract(
            "source_position_xy_px",
            dtype=FLOAT32,
            shape=_XY,
            axes=_XY_AXIS,
            description="Source-position provider coordinates in source-camera pixels.",
            units="px",
            coordinate_space=_PIXEL_SPACE,
        ),
    ),
    (
        "source_position_valid",
        _contract(
            "source_position_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether source_position_xy_px is valid.",
        ),
    ),
    (
        "source_position_reason_code",
        _contract(
            "source_position_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with source-position validity.",
        ),
    ),
    (
        "fish_identity_code",
        _contract(
            "fish_identity_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Stable fish identity code repeated on each chaser row.",
            units="identity_code",
        ),
    ),
    (
        "selection_member",
        _contract(
            "selection_member",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Temporal selection membership repeated on each chaser row.",
        ),
    ),
    (
        "acquisition_frame_delta",
        _contract(
            "acquisition_frame_delta",
            dtype=INT64,
            shape=_N,
            axes=_ROW_AXIS,
            description="Frame-index delta from the preceding source frame.",
            units="acquisition_frames",
        ),
    ),
    (
        "timestamp_delta_ns",
        _contract(
            "timestamp_delta_ns",
            dtype=INT64,
            shape=_N,
            axes=_ROW_AXIS,
            description="Camera timestamp delta from the preceding source frame.",
            units="nanoseconds",
        ),
    ),
    (
        "chaser_position_row_id",
        _contract(
            "chaser_position_row_id",
            dtype=INT64,
            shape=_N,
            axes=_ROW_AXIS,
            description="Original row identity in the bound chaser-position provider.",
            units="source_provider_row_index",
        ),
    ),
    (
        "chaser_position_row_valid",
        _contract(
            "chaser_position_row_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether chaser_position_row_id identifies a source row.",
        ),
    ),
    (
        "chaser_position_row_reason_code",
        _contract(
            "chaser_position_row_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with chaser-position row validity.",
        ),
    ),
    (
        "chaser_position_xy_px",
        _contract(
            "chaser_position_xy_px",
            dtype=FLOAT32,
            shape=_XY,
            axes=_XY_AXIS,
            description="Chaser-position provider coordinates in source-camera pixels.",
            units="px",
            coordinate_space=_PIXEL_SPACE,
        ),
    ),
    (
        "chaser_position_valid",
        _contract(
            "chaser_position_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether chaser_position_xy_px is valid.",
        ),
    ),
    (
        "chaser_position_reason_code",
        _contract(
            "chaser_position_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with chaser-position validity.",
        ),
    ),
    (
        "chaser_identity_code",
        _contract(
            "chaser_identity_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Stable identity code for the chaser-axis column.",
            units="identity_code",
        ),
    ),
    (
        "chaser_behavior_role_code",
        _contract(
            "chaser_behavior_role_code",
            dtype=UINT8,
            shape=_N,
            axes=_ROW_AXIS,
            description="Declared behavior-role code for the chaser-axis column.",
            units="behavior_role_code",
        ),
    ),
    (
        "chaser_behavior_role_valid",
        _contract(
            "chaser_behavior_role_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether the chaser behavior-role code is valid.",
        ),
    ),
    (
        "chaser_behavior_role_reason_code",
        _contract(
            "chaser_behavior_role_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with behavior-role validity.",
        ),
    ),
    (
        "chaser_occurrence_member",
        _contract(
            "chaser_occurrence_member",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether the source row belongs to the declared chaser occurrence.",
        ),
    ),
    (
        "nearest_chaser_member",
        _contract(
            "nearest_chaser_member",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether this row is the nearest-chaser relation for its frame.",
        ),
    ),
    (
        "row_valid",
        _contract(
            "row_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether the source relative relation row is valid.",
        ),
    ),
    (
        "row_reason_code",
        _contract(
            "row_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with row_valid.",
        ),
    ),
    (
        "relative_transition_valid",
        _contract(
            "relative_transition_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether the source relative transition is valid.",
        ),
    ),
    (
        "relative_transition_reason_code",
        _contract(
            "relative_transition_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with relative-transition validity.",
        ),
    ),
    (
        "relative_vector_px_xy",
        _contract(
            "relative_vector_px_xy",
            dtype=FLOAT32,
            shape=_XY,
            axes=_XY_AXIS,
            description="Source-to-chaser relative vector in source-camera pixels.",
            units="px",
            coordinate_space=_PIXEL_SPACE,
        ),
    ),
    (
        "distance_px",
        _contract(
            "distance_px",
            dtype=FLOAT32,
            shape=_N,
            axes=_ROW_AXIS,
            description="Euclidean source-to-chaser distance in pixels.",
            units="px",
            coordinate_space=_PIXEL_SPACE,
        ),
    ),
    (
        "distance_px_valid",
        _contract(
            "distance_px_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether distance_px is valid.",
        ),
    ),
    (
        "distance_px_reason_code",
        _contract(
            "distance_px_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with distance_px_valid.",
        ),
    ),
    (
        "distance_mm",
        _contract(
            "distance_mm",
            dtype=FLOAT32,
            shape=_N,
            axes=_ROW_AXIS,
            description="Euclidean source-to-chaser distance derived from pixels_per_unit in millimetres.",
            units="mm",
            coordinate_space="source_camera_calibrated_xy",
        ),
    ),
    (
        "distance_mm_valid",
        _contract(
            "distance_mm_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Whether distance_mm is valid; requires an authoritative mm scale.",
        ),
    ),
    (
        "distance_mm_reason_code",
        _contract(
            "distance_mm_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Versioned reason code paired with distance_mm_valid.",
        ),
    ),
    (
        "trial_id",
        _contract(
            "trial_id",
            dtype=INT64,
            shape=_N,
            axes=_ROW_AXIS,
            description="Optional source trial identity repeated on each chaser row.",
            units="trial_id",
        ),
    ),
    (
        "trial_valid",
        _contract(
            "trial_valid",
            dtype=BOOL,
            shape=_N,
            axes=_ROW_AXIS,
            description="Optional explicit validity for trial_id.",
        ),
    ),
    (
        "trial_reason_code",
        _contract(
            "trial_reason_code",
            dtype=UINT16,
            shape=_N,
            axes=_ROW_AXIS,
            description="Optional versioned reason code paired with trial_valid.",
        ),
    ),
)

_ARRAY_NAMES = frozenset(name for name, _ in _ARRAYS)
_OPTIONAL_TRIAL = frozenset({"trial_id", "trial_valid", "trial_reason_code"})
_REQUIRED_ARRAY_NAMES = _ARRAY_NAMES - _OPTIONAL_TRIAL

PROVIDER_CHASER_DISTANCE_ARRAY_CONTRACTS = ArrayContractCatalog(
    contract for _, contract in _ARRAYS
)
PROVIDER_CHASER_DISTANCE_BINDINGS = tuple(
    ArrayContractBinding(
        path=name,
        contract_id=contract.schema_id,
        contract_version=contract.schema_version,
        required=name in _REQUIRED_ARRAY_NAMES,
    )
    for name, contract in _ARRAYS
)

_FRAME_REPEATED_FIELDS = frozenset(
    {
        "acquisition_frame_id",
        "track_sample_id",
        "timestamp_ns",
        "timestamp_valid",
        "timestamp_reason_code",
        "source_position_row_id",
        "source_position_row_valid",
        "source_position_row_reason_code",
        "source_position_xy_px",
        "source_position_valid",
        "source_position_reason_code",
        "fish_identity_code",
        "selection_member",
        "acquisition_frame_delta",
        "timestamp_delta_ns",
    }
)


@dataclass(frozen=True, slots=True)
class ProviderChaserDistanceDimensions:
    """Concrete dimensions for one frame-by-chaser flat row table."""

    n_frames: int
    n_chasers: int

    def __post_init__(self) -> None:
        if type(self.n_frames) is not int or self.n_frames < 0:
            raise TypeError("n_frames must be a nonnegative exact integer.")
        if type(self.n_chasers) is not int or self.n_chasers <= 0:
            raise TypeError("n_chasers must be a positive exact integer.")

    @property
    def n_rows(self) -> int:
        return self.n_frames * self.n_chasers

    @property
    def contract_dimensions(self) -> dict[str, int]:
        return {"n_rows": self.n_rows}

    def as_manifest(self) -> dict[str, int]:
        return {
            "n_frames": self.n_frames,
            "n_chasers": self.n_chasers,
            "n_rows": self.n_rows,
        }


@dataclass(frozen=True, slots=True)
class ProviderChaserDistanceSchemaIssue:
    code: str
    path: str
    message: str


class ProviderChaserDistanceSchemaError(ValueError):
    """Raised when provider chaser-distance rows violate the exact contract."""

    def __init__(self, issues: tuple[ProviderChaserDistanceSchemaIssue, ...]):
        self.issues = issues
        detail = "; ".join(
            f"{issue.code} at {issue.path}: {issue.message}" for issue in issues
        )
        super().__init__(
            "Provider chaser-distance schema validation failed with "
            f"{len(issues)} issue(s): {detail}"
        )


def _issue(code: str, path: str, message: str) -> ProviderChaserDistanceSchemaIssue:
    return ProviderChaserDistanceSchemaIssue(code=code, path=path, message=message)


def _frame_chaser(array: np.ndarray, dimensions: ProviderChaserDistanceDimensions) -> np.ndarray:
    return array.reshape((dimensions.n_frames, dimensions.n_chasers) + array.shape[1:])


def _validate_validity_reason(
    arrays: Mapping[str, np.ndarray],
    *,
    valid_path: str,
    reason_path: str,
    issues: list[ProviderChaserDistanceSchemaIssue],
) -> None:
    valid = arrays[valid_path]
    reason = arrays[reason_path]
    if np.any(valid & (reason != 0)):
        issues.append(_issue("valid_reason_mismatch", reason_path, "valid rows require reason code zero."))
    if np.any(~valid & (reason == 0)):
        issues.append(_issue("missing_invalid_reason", reason_path, "invalid rows require a nonzero reason code."))


def _validate_float_validity(
    arrays: Mapping[str, np.ndarray],
    *,
    value_path: str,
    valid_path: str,
    issues: list[ProviderChaserDistanceSchemaIssue],
) -> None:
    values = arrays[value_path]
    valid = arrays[valid_path]
    if values.ndim > 1:
        finite = np.isfinite(values).all(axis=tuple(range(1, values.ndim)))
        nan = np.isnan(values).all(axis=tuple(range(1, values.ndim)))
    else:
        finite = np.isfinite(values)
        nan = np.isnan(values)
    if np.any(valid & ~finite):
        issues.append(_issue("valid_nonfinite", value_path, "valid rows require finite floating values."))
    if np.any(~valid & ~nan):
        issues.append(_issue("invalid_value_not_nan", value_path, "invalid rows require NaN floating values."))


def _validate_row_ids(
    arrays: Mapping[str, np.ndarray],
    *,
    row_path: str,
    valid_path: str,
    issues: list[ProviderChaserDistanceSchemaIssue],
) -> None:
    row = arrays[row_path]
    valid = arrays[valid_path]
    if np.any(valid & (row < 0)):
        issues.append(_issue("negative_source_row", row_path, "valid source rows require nonnegative identities."))
    if np.any(~valid & (row != -1)):
        issues.append(_issue("invalid_source_row_sentinel", row_path, "invalid source rows require the -1 sentinel."))


def _validate_invariants(
    arrays: Mapping[str, np.ndarray],
    *,
    dimensions: ProviderChaserDistanceDimensions,
) -> tuple[ProviderChaserDistanceSchemaIssue, ...]:
    issues: list[ProviderChaserDistanceSchemaIssue] = []
    if dimensions.n_rows:
        grouped = {
            name: _frame_chaser(array, dimensions) for name, array in arrays.items()
        }
        for name in _FRAME_REPEATED_FIELDS:
            values = grouped[name]
            reference = np.broadcast_to(values[:, :1, ...], values.shape)
            if not np.array_equal(values, reference, equal_nan=True):
                issues.append(_issue("frame_evidence_mismatch", name, "frame-level evidence must repeat across chaser rows."))
        frame_ids = grouped["acquisition_frame_id"][:, 0]
        if np.unique(frame_ids).size != dimensions.n_frames:
            issues.append(_issue("duplicate_frame_id", "acquisition_frame_id", "one frame identity must identify one frame group."))
        if np.any(frame_ids < 0):
            issues.append(_issue("negative_frame_id", "acquisition_frame_id", "frame identities must be nonnegative."))
        values = grouped["chaser_identity_code"]
        reference = np.broadcast_to(values[0:1, ...], values.shape)
        if not np.array_equal(values, reference):
            issues.append(_issue("unstable_chaser_axis", "chaser_identity_code", "chaser identity codes must be stable across frames."))
        if np.unique(grouped["chaser_identity_code"][0]).size != dimensions.n_chasers:
            issues.append(_issue("duplicate_chaser_identity", "chaser_identity_code", "chaser columns require distinct identity codes."))
    timestamp = arrays["timestamp_ns"]
    timestamp_valid = arrays["timestamp_valid"]
    if np.any(timestamp_valid & (timestamp < 0)):
        issues.append(_issue("negative_timestamp", "timestamp_ns", "valid timestamps must be nonnegative."))
    if np.any(~timestamp_valid & (timestamp != -1)):
        issues.append(_issue("invalid_timestamp_sentinel", "timestamp_ns", "invalid timestamps require the -1 sentinel."))

    for valid_path, reason_path in (
        ("timestamp_valid", "timestamp_reason_code"),
        ("source_position_row_valid", "source_position_row_reason_code"),
        ("source_position_valid", "source_position_reason_code"),
        ("chaser_position_row_valid", "chaser_position_row_reason_code"),
        ("chaser_position_valid", "chaser_position_reason_code"),
        ("chaser_behavior_role_valid", "chaser_behavior_role_reason_code"),
        ("row_valid", "row_reason_code"),
        ("relative_transition_valid", "relative_transition_reason_code"),
        ("distance_px_valid", "distance_px_reason_code"),
        ("distance_mm_valid", "distance_mm_reason_code"),
    ):
        _validate_validity_reason(arrays, valid_path=valid_path, reason_path=reason_path, issues=issues)
    if "trial_id" in arrays:
        _validate_validity_reason(arrays, valid_path="trial_valid", reason_path="trial_reason_code", issues=issues)

    _validate_row_ids(arrays, row_path="source_position_row_id", valid_path="source_position_row_valid", issues=issues)
    _validate_row_ids(arrays, row_path="chaser_position_row_id", valid_path="chaser_position_row_valid", issues=issues)
    for value_path, valid_path in (
        ("source_position_xy_px", "source_position_valid"),
        ("chaser_position_xy_px", "chaser_position_valid"),
        ("relative_vector_px_xy", "distance_px_valid"),
        ("distance_px", "distance_px_valid"),
        ("distance_mm", "distance_mm_valid"),
    ):
        _validate_float_validity(arrays, value_path=value_path, valid_path=valid_path, issues=issues)
    if np.any(arrays["source_position_valid"] & ~arrays["source_position_row_valid"]):
        issues.append(_issue("position_without_source_row", "source_position_valid", "valid source positions require valid source row identities."))
    if np.any(arrays["chaser_position_valid"] & ~arrays["chaser_position_row_valid"]):
        issues.append(_issue("position_without_chaser_row", "chaser_position_valid", "valid chaser positions require valid chaser row identities."))
    if np.any(arrays["distance_px_valid"] & (~arrays["source_position_valid"] | ~arrays["chaser_position_valid"])):
        issues.append(_issue("distance_without_positions", "distance_px_valid", "valid distances require valid source and chaser positions."))
    if not np.array_equal(arrays["distance_mm_valid"], arrays["distance_px_valid"]):
        issues.append(_issue("mm_validity_mismatch", "distance_mm_valid", "mm validity must match pixel validity after an authoritative mm scale is bound."))
    if not np.array_equal(arrays["distance_mm_reason_code"], arrays["distance_px_reason_code"]):
        issues.append(_issue("mm_reason_mismatch", "distance_mm_reason_code", "mm and pixel distance reason codes must agree."))
    comparable = arrays["distance_px_valid"]
    if np.any(comparable):
        expected = np.linalg.norm(
            arrays["chaser_position_xy_px"].astype(np.float64)
            - arrays["source_position_xy_px"].astype(np.float64),
            axis=1,
        )
        if not np.allclose(arrays["distance_px"][comparable], expected[comparable], atol=5e-4, rtol=0.0):
            issues.append(_issue("distance_derivation_mismatch", "distance_px", "valid pixel distances must equal the source-to-chaser Euclidean norm."))
    return tuple(issues)


@dataclass(frozen=True, slots=True)
class ProviderChaserDistanceSchema:
    schema_id: str
    schema_version: int
    bindings: tuple[ArrayContractBinding, ...]
    contracts: ArrayContractCatalog

    @property
    def binding_paths(self) -> tuple[str, ...]:
        return tuple(binding.path for binding in self.bindings)

    def validate(
        self,
        arrays: Mapping[str, Any],
        *,
        dimensions: ProviderChaserDistanceDimensions,
    ) -> tuple[ProviderChaserDistanceSchemaIssue, ...]:
        issues: list[ProviderChaserDistanceSchemaIssue] = []
        if not isinstance(arrays, Mapping):
            return (_issue("array_mapping", "arrays", "arrays must be a mapping."),)
        observed = set(arrays)
        unknown = observed - _ARRAY_NAMES
        if unknown:
            issues.append(_issue("unknown_array", "arrays", f"unknown arrays: {sorted(unknown)!r}."))
        missing = _REQUIRED_ARRAY_NAMES - observed
        if missing:
            issues.append(_issue("missing_array", "arrays", f"missing required arrays: {sorted(missing)!r}."))
        optional_present = observed & _OPTIONAL_TRIAL
        if optional_present and optional_present != _OPTIONAL_TRIAL:
            issues.append(_issue("partial_trial", "trial_id", "trial_id, trial_valid, and trial_reason_code must be supplied together."))
        for name in sorted(observed & _ARRAY_NAMES):
            try:
                errors = self.contracts.resolve(
                    f"palette.array.provider_chaser_distance.{name}", 1
                ).validate_observation(arrays[name], dimensions=dimensions.contract_dimensions)
            except (AttributeError, TypeError, ValueError) as exc:
                errors = (str(exc),)
            for error in errors:
                issues.append(_issue("array_contract", name, error))
        if not issues:
            issues.extend(_validate_invariants(arrays, dimensions=dimensions))
        return tuple(issues)

    def require(
        self,
        arrays: Mapping[str, Any],
        *,
        dimensions: ProviderChaserDistanceDimensions,
    ) -> None:
        issues = self.validate(arrays, dimensions=dimensions)
        if issues:
            raise ProviderChaserDistanceSchemaError(issues)

    def as_manifest(self, *, dimensions: ProviderChaserDistanceDimensions) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "layout": PROVIDER_CHASER_DISTANCE_LAYOUT,
            "dimensions": dimensions.as_manifest(),
            "bindings": [binding.as_manifest() for binding in self.bindings],
            "array_contracts": self.contracts.as_manifest(),
            "invariants": {
                "row_axis": "frame_x_chaser",
                "row_order": "acquisition_frame_major_chaser_axis_minor",
                "n_rows": "n_frames_times_n_chasers",
                "frame_evidence": "frame-level fields repeat across chaser rows",
                "source_position_row_id": "original_provider_row_identity_not_output_row_number",
                "coordinate_space": _PIXEL_SPACE,
                "invalid_float_values": "NaN",
                "distance_mm": "distance_px_divided_by_authoritative_pixels_per_mm",
                "optional_trial": "trial_id_trial_valid_trial_reason_code_are_one_atomic_triple",
                "temporal": "controller_input_provenance_proxy_is_not_display_presentation",
                "production": "selector_ineligible_and_non_production",
            },
        }


PROVIDER_CHASER_DISTANCE_SCHEMA_V1 = ProviderChaserDistanceSchema(
    schema_id=PROVIDER_CHASER_DISTANCE_SCHEMA_ID,
    schema_version=PROVIDER_CHASER_DISTANCE_SCHEMA_VERSION,
    bindings=PROVIDER_CHASER_DISTANCE_BINDINGS,
    contracts=PROVIDER_CHASER_DISTANCE_ARRAY_CONTRACTS,
)


__all__ = [
    "PROVIDER_CHASER_DISTANCE_ARRAY_CONTRACTS",
    "PROVIDER_CHASER_DISTANCE_BINDINGS",
    "PROVIDER_CHASER_DISTANCE_LAYOUT",
    "PROVIDER_CHASER_DISTANCE_SCHEMA_ID",
    "PROVIDER_CHASER_DISTANCE_SCHEMA_V1",
    "PROVIDER_CHASER_DISTANCE_SCHEMA_VERSION",
    "ProviderChaserDistanceDimensions",
    "ProviderChaserDistanceSchema",
    "ProviderChaserDistanceSchemaError",
    "ProviderChaserDistanceSchemaIssue",
]
