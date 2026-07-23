"""Versioned logical array contracts shared by writers, readers, and benchmarks.

This module does not create or open Zarr arrays. It defines exact logical
schemas and can produce an :class:`ArrayIntent` for the physical storage
planner after validating a concrete shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from fisheye.shared.zarr.storage_intent import (
    AccessPattern,
    ArrayIntent,
    WriteMode,
)


ShapeDimension = str | int


@dataclass(frozen=True)
class DTypeContract:
    """Cross-language logical dtype identity with NumPy validation support."""

    dtype_id: str
    numpy_dtype: str | None
    variable_length: bool = False

    def __post_init__(self) -> None:
        dtype_id = str(self.dtype_id).strip()
        if not dtype_id:
            raise ValueError("dtype_id cannot be empty.")
        object.__setattr__(self, "dtype_id", dtype_id)
        if self.variable_length and self.numpy_dtype is not None:
            raise ValueError(
                "Variable-length dtype contracts cannot claim a fixed NumPy dtype."
            )
        if self.numpy_dtype is not None:
            dtype = np.dtype(self.numpy_dtype)
            if dtype.hasobject:
                raise ValueError("Fixed dtype contracts cannot use an object dtype.")
            object.__setattr__(self, "numpy_dtype", str(dtype))

    @property
    def itemsize_bytes(self) -> int | None:
        if self.numpy_dtype is None:
            return None
        return int(np.dtype(self.numpy_dtype).itemsize)

    def matches(self, observed_dtype: Any) -> bool:
        """Return whether an observed array dtype matches this exact contract."""

        if self.variable_length:
            dtype_name = type(observed_dtype).__name__.lower()
            dtype_text = str(observed_dtype).lower()
            return (
                "variablelengthutf8" in dtype_name
                or "variablelengthutf8" in dtype_text
                or dtype_text in {"string", "utf8"}
            )
        try:
            return np.dtype(observed_dtype) == np.dtype(self.numpy_dtype)
        except TypeError:
            return False

    def as_manifest(self) -> dict[str, object]:
        return {
            "dtype_id": self.dtype_id,
            "numpy_dtype": self.numpy_dtype,
            "variable_length": self.variable_length,
            "itemsize_bytes": self.itemsize_bytes,
        }


BOOL = DTypeContract("bool", "bool")
INT8 = DTypeContract("int8", "int8")
INT16 = DTypeContract("int16", "int16")
INT32 = DTypeContract("int32", "int32")
INT64 = DTypeContract("int64", "int64")
UINT8 = DTypeContract("uint8", "uint8")
UINT16 = DTypeContract("uint16", "uint16")
UINT32 = DTypeContract("uint32", "uint32")
UINT64 = DTypeContract("uint64", "uint64")
FLOAT16 = DTypeContract("float16", "float16")
FLOAT32 = DTypeContract("float32", "float32")
FLOAT64 = DTypeContract("float64", "float64")
UTF8 = DTypeContract("utf8", None, variable_length=True)


@dataclass(frozen=True)
class ArrayContract:
    """Exact logical contract for one semantic array family."""

    schema_id: str
    schema_version: int
    dtype: DTypeContract
    shape_template: tuple[ShapeDimension, ...]
    axis_names: tuple[str, ...]
    description: str
    units: str | None = None
    coordinate_space: str | None = None

    def __post_init__(self) -> None:
        schema_id = str(self.schema_id).strip()
        if not schema_id:
            raise ValueError("schema_id cannot be empty.")
        object.__setattr__(self, "schema_id", schema_id)
        if type(self.schema_version) is not int or self.schema_version <= 0:
            raise ValueError("schema_version must be a positive exact integer.")

        shape_template = tuple(self.shape_template)
        for dimension in shape_template:
            if isinstance(dimension, bool) or not isinstance(dimension, (str, int)):
                raise TypeError(
                    "shape_template dimensions must be symbolic strings or integers."
                )
            if isinstance(dimension, str) and not dimension.strip():
                raise ValueError("Symbolic shape dimensions cannot be empty.")
            if isinstance(dimension, int) and dimension <= 0:
                raise ValueError("Fixed shape dimensions must be positive.")
        object.__setattr__(self, "shape_template", shape_template)

        axis_names = tuple(str(axis).strip() for axis in self.axis_names)
        if len(axis_names) != len(shape_template):
            raise ValueError(
                "axis_names must have the same rank as shape_template; "
                f"got axes={axis_names!r}, shape_template={shape_template!r}."
            )
        if any(not axis for axis in axis_names):
            raise ValueError("axis_names cannot contain empty values.")
        if len(set(axis_names)) != len(axis_names):
            raise ValueError(f"axis_names must be unique; got {axis_names!r}.")
        object.__setattr__(self, "axis_names", axis_names)

    @property
    def key(self) -> tuple[str, int]:
        return self.schema_id, self.schema_version

    def validate_shape(
        self,
        shape: Iterable[int],
        *,
        dimensions: Mapping[str, int] | None = None,
    ) -> tuple[str, ...]:
        """Return exact rank/fixed/symbolic-dimension validation errors."""

        observed = tuple(int(value) for value in shape)
        errors: list[str] = []
        if len(observed) != len(self.shape_template):
            return (
                f"rank mismatch: expected {len(self.shape_template)}, got "
                f"{len(observed)} with shape {observed!r}",
            )
        for axis, (actual, expected) in enumerate(
            zip(observed, self.shape_template)
        ):
            if actual < 0:
                errors.append(f"axis {axis} ({self.axis_names[axis]}) is negative")
            elif isinstance(expected, int) and actual != expected:
                errors.append(
                    f"axis {axis} ({self.axis_names[axis]}) expected {expected}, "
                    f"got {actual}"
                )
            elif (
                isinstance(expected, str)
                and dimensions is not None
                and expected in dimensions
                and actual != int(dimensions[expected])
            ):
                errors.append(
                    f"axis {axis} ({self.axis_names[axis]}) expected symbolic "
                    f"dimension {expected}={int(dimensions[expected])}, got {actual}"
                )
        return tuple(errors)

    def validate_observation(
        self,
        array: Any,
        *,
        dimensions: Mapping[str, int] | None = None,
    ) -> tuple[str, ...]:
        """Return logical shape and exact-dtype errors for an array-like object."""

        errors = list(self.validate_shape(array.shape, dimensions=dimensions))
        if not self.dtype.matches(array.dtype):
            errors.append(
                f"dtype mismatch: expected {self.dtype.dtype_id}, got {array.dtype}"
            )
        return tuple(errors)

    def require_observation(
        self,
        array: Any,
        *,
        dimensions: Mapping[str, int] | None = None,
    ) -> None:
        """Raise when an observed array violates this logical contract."""

        errors = self.validate_observation(array, dimensions=dimensions)
        if errors:
            joined = "; ".join(errors)
            raise ValueError(
                f"Array contract {self.schema_id}@{self.schema_version} failed: "
                f"{joined}."
            )

    def storage_intent(
        self,
        *,
        shape: tuple[int, ...],
        access: AccessPattern,
        write_mode: WriteMode,
        access_unit_shape: tuple[int, ...] | None = None,
        growth_axis: int = 0,
        shard_axes: tuple[int, ...] | None = None,
        logical_itemsize_bytes: int | None = None,
        whole_shard_writes: bool = False,
        name: str | None = None,
        dimensions: Mapping[str, int] | None = None,
    ) -> ArrayIntent:
        """Create a planner intent after validating shape and dtype identity."""

        errors = self.validate_shape(shape, dimensions=dimensions)
        if errors:
            raise ValueError(
                f"Array contract {self.schema_id}@{self.schema_version} shape "
                f"failed: {'; '.join(errors)}."
            )
        dtype: Any = self.dtype.numpy_dtype or object
        itemsize = logical_itemsize_bytes
        if itemsize is None:
            itemsize = self.dtype.itemsize_bytes
        return ArrayIntent(
            shape=shape,
            dtype=dtype,
            access=access,
            write_mode=write_mode,
            logical_schema_id=self.schema_id,
            logical_schema_version=self.schema_version,
            access_unit_shape=access_unit_shape,
            growth_axis=growth_axis,
            shard_axes=shard_axes,
            logical_itemsize_bytes=itemsize,
            whole_shard_writes=whole_shard_writes,
            name=name,
        )

    def as_manifest(self) -> dict[str, object]:
        """Return a JSON-safe logical schema record for archive manifests."""

        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "dtype": self.dtype.as_manifest(),
            "shape_template": list(self.shape_template),
            "axis_names": list(self.axis_names),
            "description": self.description,
            "units": self.units,
            "coordinate_space": self.coordinate_space,
        }


@dataclass(frozen=True)
class ArrayContractBinding:
    """Bind a concrete archive path to a reusable logical array contract."""

    path: str
    contract_id: str
    contract_version: int
    required: bool

    def as_manifest(self) -> dict[str, object]:
        return {
            "path": self.path,
            "contract_id": self.contract_id,
            "contract_version": self.contract_version,
            "required": self.required,
        }


class ArrayContractCatalog:
    """Immutable lookup collection for versioned logical contracts."""

    def __init__(self, contracts: Iterable[ArrayContract]) -> None:
        by_key: dict[tuple[str, int], ArrayContract] = {}
        for contract in contracts:
            if contract.key in by_key:
                raise ValueError(
                    f"Duplicate array contract {contract.schema_id}@"
                    f"{contract.schema_version}."
                )
            by_key[contract.key] = contract
        self._by_key = by_key

    @property
    def contracts(self) -> tuple[ArrayContract, ...]:
        return tuple(self._by_key[key] for key in sorted(self._by_key))

    def resolve(self, schema_id: str, schema_version: int) -> ArrayContract:
        key = (str(schema_id), int(schema_version))
        try:
            return self._by_key[key]
        except KeyError as exc:
            raise KeyError(
                f"Unknown array contract {schema_id}@{schema_version}."
            ) from exc

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": "palette.array_contract_catalog",
            "schema_version": 1,
            "contracts": [contract.as_manifest() for contract in self.contracts],
        }


FRAME_COUNTS_V1 = ArrayContract(
    schema_id="palette.array.frame_counts",
    schema_version=1,
    dtype=INT32,
    shape_template=("n_frames",),
    axis_names=("camera_frame",),
    description="Number of row-aligned observations belonging to each camera frame.",
    units="rows",
)

FRAME_OFFSETS_V1 = ArrayContract(
    schema_id="palette.array.frame_offsets",
    schema_version=1,
    dtype=INT64,
    shape_template=("n_frame_offsets",),
    axis_names=("camera_frame_boundary",),
    description="Exclusive row offsets with length n_frames + 1.",
    units="rows",
)

KEYPOINTS_ROI_V1 = ArrayContract(
    schema_id="palette.array.keypoints_roi",
    schema_version=1,
    dtype=FLOAT64,
    shape_template=("n_rois", "n_keypoints", 2),
    axis_names=("roi", "keypoint", "xy"),
    description="Keypoint coordinates in ROI pixel space.",
    units="pixels",
    coordinate_space="roi_pixel",
)

KEYPOINTS_IMG_V1 = ArrayContract(
    schema_id="palette.array.keypoints_img",
    schema_version=1,
    dtype=FLOAT64,
    shape_template=("n_rois", "n_keypoints", 2),
    axis_names=("roi", "keypoint", "xy"),
    description="Keypoint coordinates in full-image pixel space.",
    units="pixels",
    coordinate_space="image_pixel",
)

KEYPOINTS_NORM_V1 = ArrayContract(
    schema_id="palette.array.keypoints_norm",
    schema_version=1,
    dtype=FLOAT64,
    shape_template=("n_rois", "n_keypoints", 2),
    axis_names=("roi", "keypoint", "xy"),
    description="Keypoint coordinates normalized to the declared image domain.",
    units="normalized",
    coordinate_space="image_normalized",
)

DENSE_SUBJECT_MASKS_ROI_V1 = ArrayContract(
    schema_id="palette.array.subject_masks_roi_dense",
    schema_version=1,
    dtype=UINT8,
    shape_template=("n_rois", "n_channels", "H", "W"),
    axis_names=("roi", "component", "y", "x"),
    description="Dense authoritative subject-component labels in ROI pixel space.",
    units="binary_label",
    coordinate_space="roi_pixel",
)

CONTOUR_POINTS_XY_V1 = ArrayContract(
    schema_id="palette.array.contour_points_xy",
    schema_version=1,
    dtype=FLOAT32,
    shape_template=("n_points", 2),
    axis_names=("point", "xy"),
    description="Flat CSR contour coordinate values in ROI pixel space.",
    units="pixels",
    coordinate_space="roi_pixel",
)


CORE_ARRAY_CONTRACTS = ArrayContractCatalog(
    (
        FRAME_COUNTS_V1,
        FRAME_OFFSETS_V1,
        KEYPOINTS_ROI_V1,
        KEYPOINTS_IMG_V1,
        KEYPOINTS_NORM_V1,
        DENSE_SUBJECT_MASKS_ROI_V1,
        CONTOUR_POINTS_XY_V1,
    )
)
