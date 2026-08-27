"""Exact logical schema for reusable observation-level body-frame runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.array_contracts import (
    BODY_FRAME_ARRAY_CONTRACTS,
    BODY_FRAME_AXIS_VALID_V1,
    BODY_FRAME_FORWARD_AXIS_XY_V1,
    BODY_FRAME_HEADING_DEG_V1,
    BODY_FRAME_LEFT_AXIS_XY_V1,
    BODY_FRAME_ORIGIN_XY_V1,
    BODY_FRAME_SOURCE_KEYPOINT_ROW_IDS_V1,
    BODY_FRAME_SOURCE_KEYPOINT_ROW_SIGNATURE_V1,
    FRAME_ROW_OFFSETS_V1,
    KEYPOINT_FRAME_INDICES_V1,
    KEYPOINT_INSTANCE_KEY_V1,
    ArrayContract,
    ArrayContractBinding,
    ArrayContractCatalog,
)
from fisheye.shared.zarr.keypoint_schema import derive_frame_row_offsets

BODY_FRAME_RUN_SCHEMA_ID = "palette.analysis.body_frame"
BODY_FRAME_RUN_SCHEMA_VERSION = 1
BODY_FRAME_RUN_LAYOUT = "sparse_observation_body_frames_v1"
BODY_FRAME_ANGLE_CONVENTION = "atan2_negative_y_x_degrees"
# NumPy/libm implementations can differ by a few float32 ULPs when the cached
# angle is reproduced from the authoritative axes on another machine.  This
# bound matches the downstream relative-frame schema while remaining far below
# any scientifically meaningful orientation change.
BODY_FRAME_HEADING_VALIDATION_ATOL_DEG = 5e-5

_CONTRACTS: tuple[tuple[str, ArrayContract], ...] = (
    ("instance_key", KEYPOINT_INSTANCE_KEY_V1),
    ("source_keypoint_row_ids", BODY_FRAME_SOURCE_KEYPOINT_ROW_IDS_V1),
    (
        "source_keypoint_row_signature",
        BODY_FRAME_SOURCE_KEYPOINT_ROW_SIGNATURE_V1,
    ),
    ("frame_indices", KEYPOINT_FRAME_INDICES_V1),
    ("frame_row_offsets", FRAME_ROW_OFFSETS_V1),
    ("origin_xy", BODY_FRAME_ORIGIN_XY_V1),
    ("forward_axis_xy", BODY_FRAME_FORWARD_AXIS_XY_V1),
    ("left_axis_xy", BODY_FRAME_LEFT_AXIS_XY_V1),
    ("axis_valid", BODY_FRAME_AXIS_VALID_V1),
    ("heading_deg", BODY_FRAME_HEADING_DEG_V1),
)

BODY_FRAME_BINDINGS = tuple(
    ArrayContractBinding(
        path=name,
        contract_id=contract.schema_id,
        contract_version=contract.schema_version,
        required=True,
    )
    for name, contract in _CONTRACTS
)


@dataclass(frozen=True)
class BodyFrameDimensions:
    n_frames: int
    n_instances: int

    def __post_init__(self) -> None:
        if type(self.n_frames) is not int or self.n_frames <= 0:
            raise ValueError("n_frames must be a positive exact integer.")
        if type(self.n_instances) is not int or self.n_instances < 0:
            raise ValueError("n_instances must be a nonnegative exact integer.")

    @property
    def contract_dimensions(self) -> dict[str, int]:
        return {
            "n_frames": self.n_frames,
            "n_frame_boundaries": self.n_frames + 1,
            "n_instances": self.n_instances,
        }

    def as_manifest(self) -> dict[str, int]:
        return dict(self.contract_dimensions)


@dataclass(frozen=True)
class BodyFrameSchemaIssue:
    code: str
    path: str
    message: str

    def as_manifest(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


class BodyFrameSchemaError(ValueError):
    def __init__(self, issues: tuple[BodyFrameSchemaIssue, ...]) -> None:
        self.issues = issues
        detail = "; ".join(
            f"{issue.code} at {issue.path}: {issue.message}" for issue in issues
        )
        super().__init__(
            f"Body-frame schema validation failed with {len(issues)} issue(s): {detail}"
        )


def _issue(code: str, path: str, message: str) -> BodyFrameSchemaIssue:
    return BodyFrameSchemaIssue(code=code, path=path, message=message)


def _materialize(array: Any) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    try:
        return np.asarray(array[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(array)


@dataclass(frozen=True)
class BodyFrameSchema:
    schema_id: str
    schema_version: int
    bindings: tuple[ArrayContractBinding, ...]
    contracts: ArrayContractCatalog

    def __post_init__(self) -> None:
        if not self.schema_id.strip():
            raise ValueError("schema_id cannot be empty.")
        if type(self.schema_version) is not int or self.schema_version <= 0:
            raise ValueError("schema_version must be a positive exact integer.")
        paths = [binding.path for binding in self.bindings]
        if len(paths) != len(set(paths)):
            raise ValueError("Body-frame binding paths must be unique.")
        for binding in self.bindings:
            if not binding.required:
                raise ValueError("Every body-frame-v1 binding must be required.")
            self.contracts.resolve(binding.contract_id, binding.contract_version)

    @property
    def binding_paths(self) -> tuple[str, ...]:
        return tuple(binding.path for binding in self.bindings)

    def coordinate_contract_manifest(self) -> dict[str, object]:
        from fisheye.shared.zarr.coordinate_contracts import (
            array_coordinate_catalog_manifest,
        )

        return array_coordinate_catalog_manifest(self.contracts)

    def validate(
        self,
        arrays: Mapping[str, Any],
        *,
        dimensions: BodyFrameDimensions,
        source_keypoint_arrays: Mapping[str, Any] | None,
    ) -> tuple[BodyFrameSchemaIssue, ...]:
        issues: list[BodyFrameSchemaIssue] = []
        invalid_paths: set[str] = set()
        expected = set(self.binding_paths)
        for path in sorted(set(arrays) - expected):
            issues.append(
                _issue(
                    "unexpected_array",
                    path,
                    "The exact body-frame-v1 schema does not declare this array.",
                )
            )
        for binding in self.bindings:
            path = binding.path
            if path not in arrays:
                invalid_paths.add(path)
                issues.append(
                    _issue(
                        "missing_required_array",
                        path,
                        "Required body-frame array is absent.",
                    )
                )
                continue
            contract = self.contracts.resolve(
                binding.contract_id,
                binding.contract_version,
            )
            try:
                errors = contract.validate_observation(
                    arrays[path],
                    dimensions=dimensions.contract_dimensions,
                )
            except Exception as exc:
                errors = (f"array metadata is unreadable: {exc}",)
            if errors:
                invalid_paths.add(path)
                issues.extend(
                    _issue("array_contract_violation", path, error) for error in errors
                )

        values: dict[str, np.ndarray] = {}
        for path in self.binding_paths:
            if path in invalid_paths:
                continue
            try:
                values[path] = _materialize(arrays[path])
            except Exception as exc:
                invalid_paths.add(path)
                issues.append(
                    _issue(
                        "array_read_failure", path, f"Array could not be read: {exc}"
                    )
                )

        frames = values.get("frame_indices")
        frames_valid = False
        if frames is not None:
            frames_valid = bool(
                np.all(frames >= 0) and np.all(frames < dimensions.n_frames)
            )
            if not frames_valid:
                issues.append(
                    _issue(
                        "frame_index_out_of_bounds",
                        "frame_indices",
                        "Every row frame must be in [0, n_frames).",
                    )
                )
            if frames.size > 1 and np.any(np.diff(frames) < 0):
                frames_valid = False
                issues.append(
                    _issue(
                        "frame_indices_not_sorted",
                        "frame_indices",
                        "Rows must be contiguous in nondecreasing frame order.",
                    )
                )

        offsets = values.get("frame_row_offsets")
        if offsets is not None:
            if int(offsets[0]) != 0:
                issues.append(
                    _issue(
                        "offset_start_mismatch",
                        "frame_row_offsets",
                        "Offsets must start at zero.",
                    )
                )
            if np.any(np.diff(offsets) < 0):
                issues.append(
                    _issue(
                        "offsets_not_monotonic",
                        "frame_row_offsets",
                        "Offsets must be nondecreasing.",
                    )
                )
            if int(offsets[-1]) != dimensions.n_instances:
                issues.append(
                    _issue(
                        "offset_end_mismatch",
                        "frame_row_offsets",
                        "The final offset must equal n_instances.",
                    )
                )
        if frames is not None and frames_valid and offsets is not None:
            expected_offsets = derive_frame_row_offsets(
                frames,
                n_frames=dimensions.n_frames,
            )
            if not np.array_equal(offsets, expected_offsets):
                issues.append(
                    _issue(
                        "frame_row_offsets_mismatch",
                        "frame_row_offsets",
                        "Offsets must exactly index contiguous body-frame rows.",
                    )
                )

        keys = values.get("instance_key")
        if keys is not None and np.unique(keys).shape[0] != keys.shape[0]:
            issues.append(
                _issue(
                    "duplicate_instance_key",
                    "instance_key",
                    "instance_key values must be unique.",
                )
            )
        source_rows = values.get("source_keypoint_row_ids")
        if source_rows is not None:
            if np.any(source_rows < 0):
                issues.append(
                    _issue(
                        "invalid_source_keypoint_row_id",
                        "source_keypoint_row_ids",
                        "Source keypoint row IDs must be nonnegative.",
                    )
                )
            if np.unique(source_rows).shape[0] != source_rows.shape[0]:
                issues.append(
                    _issue(
                        "duplicate_source_keypoint_row_id",
                        "source_keypoint_row_ids",
                        "A body-frame run cannot reuse a source keypoint row.",
                    )
                )

        if source_keypoint_arrays is None:
            issues.append(
                _issue(
                    "missing_source_keypoint_evidence",
                    "source_keypoint_row_ids",
                    "Exact bound keypoint arrays are required for lineage validation.",
                )
            )
        elif source_rows is not None and np.all(source_rows >= 0):
            issues.extend(
                self._validate_source_binding(
                    values,
                    source_keypoint_arrays=source_keypoint_arrays,
                    source_rows=source_rows,
                )
            )

        origin = values.get("origin_xy")
        forward = values.get("forward_axis_xy")
        left = values.get("left_axis_xy")
        valid = values.get("axis_valid")
        heading = values.get("heading_deg")
        if all(item is not None for item in (origin, forward, left, valid, heading)):
            assert origin is not None
            assert forward is not None
            assert left is not None
            assert valid is not None
            assert heading is not None
            finite_geometry = (
                np.all(np.isfinite(origin), axis=1)
                & np.all(np.isfinite(forward), axis=1)
                & np.all(np.isfinite(left), axis=1)
                & np.isfinite(heading)
            )
            if not np.array_equal(valid, finite_geometry):
                issues.append(
                    _issue(
                        "axis_valid_mismatch",
                        "axis_valid",
                        "axis_valid must exactly mark rows with complete finite geometry.",
                    )
                )
            invalid = ~valid
            if np.any(
                np.isfinite(origin[invalid])
                | np.isfinite(forward[invalid])
                | np.isfinite(left[invalid])
            ) or np.any(np.isfinite(heading[invalid])):
                issues.append(
                    _issue(
                        "invalid_axis_not_nan",
                        "axis_valid",
                        "Invalid body-frame rows must use NaN for every geometry value.",
                    )
                )
            if np.any(valid):
                forward_norm = np.linalg.norm(forward[valid].astype(np.float64), axis=1)
                left_norm = np.linalg.norm(left[valid].astype(np.float64), axis=1)
                dot = np.einsum(
                    "ij,ij->i",
                    forward[valid].astype(np.float64),
                    left[valid].astype(np.float64),
                )
                determinant = forward[valid, 0].astype(np.float64) * left[
                    valid, 1
                ].astype(np.float64) - forward[valid, 1].astype(np.float64) * left[
                    valid, 0
                ].astype(
                    np.float64
                )
                if not (
                    np.allclose(forward_norm, 1.0, atol=5e-6, rtol=0.0)
                    and np.allclose(left_norm, 1.0, atol=5e-6, rtol=0.0)
                    and np.allclose(dot, 0.0, atol=5e-6, rtol=0.0)
                    and np.allclose(determinant, -1.0, atol=5e-6, rtol=0.0)
                ):
                    issues.append(
                        _issue(
                            "invalid_body_axes",
                            "forward_axis_xy",
                            "Valid axes must be orthonormal with anatomical-left orientation.",
                        )
                    )
                expected_heading = np.rad2deg(
                    np.arctan2(-forward[:, 1], forward[:, 0])
                ).astype(np.float32)
                expected_heading[invalid] = np.nan
                if np.any(
                    valid
                    & ~np.isclose(
                        heading,
                        expected_heading,
                        atol=BODY_FRAME_HEADING_VALIDATION_ATOL_DEG,
                        rtol=0.0,
                    )
                ):
                    issues.append(
                        _issue(
                            "heading_derivation_mismatch",
                            "heading_deg",
                            "heading_deg must equal atan2(-forward_y, forward_x) "
                            f"within {BODY_FRAME_HEADING_VALIDATION_ATOL_DEG:g} degrees.",
                        )
                    )
        return tuple(issues)

    def _validate_source_binding(
        self,
        values: Mapping[str, np.ndarray],
        *,
        source_keypoint_arrays: Mapping[str, Any],
        source_rows: np.ndarray,
    ) -> tuple[BodyFrameSchemaIssue, ...]:
        required = ("instance_key", "frame_indices", "keypoint_row_signature")
        missing = [path for path in required if path not in source_keypoint_arrays]
        if missing:
            return (
                _issue(
                    "incomplete_source_keypoint_evidence",
                    "source_keypoint_row_ids",
                    f"Bound keypoint evidence is missing {missing!r}.",
                ),
            )
        try:
            source = {
                path: _materialize(source_keypoint_arrays[path]) for path in required
            }
        except Exception as exc:
            return (
                _issue(
                    "source_keypoint_read_failure",
                    "source_keypoint_row_ids",
                    f"Bound keypoint evidence could not be read: {exc}",
                ),
            )
        source_count = int(source["instance_key"].shape[0])
        if np.any(source_rows >= source_count):
            return (
                _issue(
                    "source_keypoint_row_out_of_bounds",
                    "source_keypoint_row_ids",
                    "A source keypoint row ID is outside the bound snapshot.",
                ),
            )
        selected = source_rows.astype(np.int64, copy=False)
        issues: list[BodyFrameSchemaIssue] = []
        comparisons = (
            ("instance_key", "instance_key"),
            ("frame_indices", "frame_indices"),
            ("source_keypoint_row_signature", "keypoint_row_signature"),
        )
        for target_path, source_path in comparisons:
            target = values.get(target_path)
            if target is not None and not np.array_equal(
                target, source[source_path][selected]
            ):
                issues.append(
                    _issue(
                        "source_keypoint_binding_mismatch",
                        target_path,
                        f"Values do not equal bound keypoint rows from {source_path}.",
                    )
                )
        return tuple(issues)

    def require(
        self,
        arrays: Mapping[str, Any],
        *,
        dimensions: BodyFrameDimensions,
        source_keypoint_arrays: Mapping[str, Any] | None,
    ) -> None:
        issues = self.validate(
            arrays,
            dimensions=dimensions,
            source_keypoint_arrays=source_keypoint_arrays,
        )
        if issues:
            raise BodyFrameSchemaError(issues)

    def as_manifest(self, *, dimensions: BodyFrameDimensions) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "stage": "body_frame",
            "layout": BODY_FRAME_RUN_LAYOUT,
            "base_path": "analysis/body_frame_runs/<run>",
            "dimensions": dimensions.as_manifest(),
            "bindings": [binding.as_manifest() for binding in self.bindings],
            "array_contracts": self.contracts.as_manifest(),
            "invariants": {
                "row_order": "frame_indices_nondecreasing",
                "row_identity": "instance_key",
                "source_row_identity": "source_keypoint_row_ids",
                "frame_lookup": "frame_row_offsets_csr",
                "coordinate_space": "source_camera_pixels",
                "axis_handedness": "forward_cross_left_negative_camera_xy",
                "invalid_geometry": "axis_valid_false_all_geometry_nan",
                "heading_derivation": BODY_FRAME_ANGLE_CONVENTION,
                "instance_key_semantics": "observation_identity_not_subject_identity",
            },
        }


BODY_FRAME_SCHEMA_V1 = BodyFrameSchema(
    schema_id=BODY_FRAME_RUN_SCHEMA_ID,
    schema_version=BODY_FRAME_RUN_SCHEMA_VERSION,
    bindings=BODY_FRAME_BINDINGS,
    contracts=BODY_FRAME_ARRAY_CONTRACTS,
)


__all__ = [
    "BODY_FRAME_ANGLE_CONVENTION",
    "BODY_FRAME_BINDINGS",
    "BODY_FRAME_HEADING_VALIDATION_ATOL_DEG",
    "BODY_FRAME_RUN_LAYOUT",
    "BODY_FRAME_RUN_SCHEMA_ID",
    "BODY_FRAME_RUN_SCHEMA_VERSION",
    "BODY_FRAME_SCHEMA_V1",
    "BodyFrameDimensions",
    "BodyFrameSchema",
    "BodyFrameSchemaError",
    "BodyFrameSchemaIssue",
]
