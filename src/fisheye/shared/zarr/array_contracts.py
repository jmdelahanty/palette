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

FRAME_ROW_OFFSETS_V1 = ArrayContract(
    schema_id="palette.array.frame_row_offsets",
    schema_version=1,
    dtype=INT64,
    shape_template=("n_frame_boundaries",),
    axis_names=("camera_frame_boundary",),
    description=(
        "Exclusive offsets into a frame-contiguous sparse instance table; "
        "length is n_frames + 1."
    ),
    units="instance_rows",
)

DETECTION_FRAME_INDICES_V1 = ArrayContract(
    schema_id="palette.array.detection.frame_indices",
    schema_version=1,
    dtype=INT32,
    shape_template=("n_instances",),
    axis_names=("instance",),
    description="Run-local camera frame index for each detection instance.",
    units="camera_frame_index",
)

DETECTION_SOURCE_ACQUISITION_FRAME_INDEX_V1 = ArrayContract(
    schema_id="palette.array.detection.source_acquisition_frame_index",
    schema_version=1,
    dtype=INT64,
    shape_template=("n_instances",),
    axis_names=("instance",),
    description=(
        "Sealed acquisition-camera frame index for each detection instance."
    ),
    units="acquisition_frame_index",
)

DETECTION_INSTANCE_KEY_V1 = ArrayContract(
    schema_id="palette.array.detection.instance_key",
    schema_version=1,
    dtype=UINT64,
    shape_template=("n_instances",),
    axis_names=("instance",),
    description="Stable content-derived identity for each detection instance.",
    units="identity_key",
)

DETECTION_BBOX_NORM_COORDS_V1 = ArrayContract(
    schema_id="palette.array.detection.bbox_norm_coords",
    schema_version=1,
    dtype=FLOAT32,
    shape_template=("n_instances", 4),
    axis_names=("instance", "cxcywh"),
    description=(
        "Authoritative source-camera-normalized detection boxes in cx,cy,w,h "
        "component order."
    ),
    units="normalized",
    coordinate_space="source_camera_normalized",
)

DETECTION_BBOX_IMG_XYXY_V1 = ArrayContract(
    schema_id="palette.array.detection.bbox_img_xyxy",
    schema_version=1,
    dtype=FLOAT32,
    shape_template=("n_instances", 4),
    axis_names=("instance", "xyxy"),
    description=(
        "Exact source-camera half-open pixel-edge projection of the normalized "
        "detection box."
    ),
    units="pixels",
    coordinate_space="source_camera_pixel_edges",
)

DETECTION_CENTERS_IMG_XY_V1 = ArrayContract(
    schema_id="palette.array.detection.centers_img_xy",
    schema_version=1,
    dtype=FLOAT32,
    shape_template=("n_instances", 2),
    axis_names=("instance", "xy"),
    description=(
        "Exact source-camera continuous-pixel midpoint of bbox_img_xyxy."
    ),
    units="pixels",
    coordinate_space="source_camera_continuous_pixel",
)

DETECTION_SCORES_V1 = ArrayContract(
    schema_id="palette.array.detection.scores",
    schema_version=1,
    dtype=FLOAT32,
    shape_template=("n_instances",),
    axis_names=("instance",),
    description="Finite model confidence for each detection instance.",
    units="probability",
)

DETECTION_CLASS_IDS_V1 = ArrayContract(
    schema_id="palette.array.detection.class_ids",
    schema_version=1,
    dtype=INT32,
    shape_template=("n_instances",),
    axis_names=("instance",),
    description="Nonnegative model taxonomy index for each detection instance.",
    units="class_index",
)

REFINED_DETECTION_REFINED_ROW_IDS_V1 = ArrayContract(
    schema_id="palette.array.refined_detection.refined_row_ids",
    schema_version=1,
    dtype=INT64,
    shape_template=("n_instances",),
    axis_names=("instance",),
    description=(
        "Stable non-reused logical row identity within one refined-detection "
        "lineage."
    ),
    units="refined_row_identity",
)

REFINED_DETECTION_SCORES_V1 = ArrayContract(
    schema_id="palette.array.refined_detection.scores",
    schema_version=1,
    dtype=FLOAT32,
    shape_template=("n_instances",),
    axis_names=("instance",),
    description=(
        "Canonical model confidence value paired with score_valid; invalid "
        "manual scores use the exact physical value zero."
    ),
    units="probability",
)

REFINED_DETECTION_SCORE_VALID_V1 = ArrayContract(
    schema_id="palette.array.refined_detection.score_valid",
    schema_version=1,
    dtype=BOOL,
    shape_template=("n_instances",),
    axis_names=("instance",),
    description="Whether scores contains a semantically valid model confidence.",
)

REFINED_DETECTION_SOURCE_KIND_CODES_V1 = ArrayContract(
    schema_id="palette.array.refined_detection.source_kind_codes",
    schema_version=1,
    dtype=UINT8,
    shape_template=("n_instances",),
    axis_names=("instance",),
    description="Versioned refined-instance origin code: raw-backed or manual.",
)

REFINED_DETECTION_MANUAL_EDIT_FLAGS_V1 = ArrayContract(
    schema_id="palette.array.refined_detection.manual_edit_flags",
    schema_version=1,
    dtype=BOOL,
    shape_template=("n_instances",),
    axis_names=("instance",),
    description=(
        "Whether a human changed or created the persisted refined instance."
    ),
)

REFINED_DETECTION_SOURCE_DETECT_ROW_INDEX_V1 = ArrayContract(
    schema_id="palette.array.refined_detection.source_detect_row_index",
    schema_version=1,
    dtype=INT64,
    shape_template=("n_instances",),
    axis_names=("instance",),
    description=(
        "Row in the bound source_detections audit table, or exact sentinel -1 "
        "for a manual instance without a raw candidate."
    ),
    units="source_detection_row_index",
)

REFINED_DETECTION_REASON_CODES_V1 = ArrayContract(
    schema_id="palette.array.refined_detection.reason_codes",
    schema_version=1,
    dtype=UINT16,
    shape_template=("n_instances",),
    axis_names=("instance",),
    description=(
        "Versioned compact reason code; code zero means no additional reason."
    ),
)

REFINED_SOURCE_DETECT_ROW_INDEX_V1 = ArrayContract(
    schema_id="palette.array.refined_detection.source.source_detect_row_index",
    schema_version=1,
    dtype=INT64,
    shape_template=("n_source_detections",),
    axis_names=("source_detection",),
    description="Contiguous row identity in the bound source-candidate audit table.",
    units="source_detection_row_index",
)

REFINED_SOURCE_FRAME_INDICES_V1 = ArrayContract(
    schema_id="palette.array.refined_detection.source.frame_indices",
    schema_version=1,
    dtype=INT32,
    shape_template=("n_source_detections",),
    axis_names=("source_detection",),
    description="Run-local camera frame index for each source candidate.",
    units="camera_frame_index",
)

REFINED_SOURCE_ACQUISITION_FRAME_INDEX_V1 = ArrayContract(
    schema_id=(
        "palette.array.refined_detection.source.source_acquisition_frame_index"
    ),
    schema_version=1,
    dtype=INT64,
    shape_template=("n_source_detections",),
    axis_names=("source_detection",),
    description="Sealed acquisition-camera frame index for each source candidate.",
    units="acquisition_frame_index",
)

REFINED_SOURCE_INSTANCE_KEY_V1 = ArrayContract(
    schema_id="palette.array.refined_detection.source.instance_key",
    schema_version=1,
    dtype=UINT64,
    shape_template=("n_source_detections",),
    axis_names=("source_detection",),
    description="Stable identity copied from the bound canonical raw candidate.",
    units="identity_key",
)

REFINED_SOURCE_BBOX_NORM_COORDS_V1 = ArrayContract(
    schema_id="palette.array.refined_detection.source.bbox_norm_coords",
    schema_version=1,
    dtype=FLOAT32,
    shape_template=("n_source_detections", 4),
    axis_names=("source_detection", "cxcywh"),
    description="Authoritative normalized source-candidate bbox in cx,cy,w,h order.",
    units="normalized",
    coordinate_space="source_camera_normalized",
)

REFINED_SOURCE_BBOX_IMG_XYXY_V1 = ArrayContract(
    schema_id="palette.array.refined_detection.source.bbox_img_xyxy",
    schema_version=1,
    dtype=FLOAT32,
    shape_template=("n_source_detections", 4),
    axis_names=("source_detection", "xyxy"),
    description="Exact pixel-edge projection of the source candidate bbox.",
    units="pixels",
    coordinate_space="source_camera_pixel_edges",
)

REFINED_SOURCE_CENTERS_IMG_XY_V1 = ArrayContract(
    schema_id="palette.array.refined_detection.source.centers_img_xy",
    schema_version=1,
    dtype=FLOAT32,
    shape_template=("n_source_detections", 2),
    axis_names=("source_detection", "xy"),
    description="Exact continuous-pixel midpoint of the source candidate bbox.",
    units="pixels",
    coordinate_space="source_camera_continuous_pixel",
)

REFINED_SOURCE_SCORES_V1 = ArrayContract(
    schema_id="palette.array.refined_detection.source.scores",
    schema_version=1,
    dtype=FLOAT32,
    shape_template=("n_source_detections",),
    axis_names=("source_detection",),
    description="Finite model confidence copied from the bound raw candidate.",
    units="probability",
)

REFINED_SOURCE_CLASS_IDS_V1 = ArrayContract(
    schema_id="palette.array.refined_detection.source.class_ids",
    schema_version=1,
    dtype=INT32,
    shape_template=("n_source_detections",),
    axis_names=("source_detection",),
    description="Nonnegative taxonomy index copied from the bound raw candidate.",
    units="class_index",
)

REFINED_SOURCE_DECISION_CODES_V1 = ArrayContract(
    schema_id="palette.array.refined_detection.source.decision_codes",
    schema_version=1,
    dtype=UINT8,
    shape_template=("n_source_detections",),
    axis_names=("source_detection",),
    description="Versioned accepted, filtered, duplicate, or cleared decision.",
)

REFINED_SOURCE_RESOLVED_REFINED_ROW_ID_V1 = ArrayContract(
    schema_id=(
        "palette.array.refined_detection.source.resolved_refined_row_id"
    ),
    schema_version=1,
    dtype=INT64,
    shape_template=("n_source_detections",),
    axis_names=("source_detection",),
    description=(
        "Resolved refined row identity for an accepted source candidate, or "
        "exact sentinel -1 for an unaccepted candidate."
    ),
    units="refined_row_identity",
)

REFINED_SOURCE_REASON_CODES_V1 = ArrayContract(
    schema_id="palette.array.refined_detection.source.reason_codes",
    schema_version=1,
    dtype=UINT16,
    shape_template=("n_source_detections",),
    axis_names=("source_detection",),
    description="Versioned compact source-candidate decision reason code.",
)

REFINED_SOURCE_FRAME_ROW_OFFSETS_V1 = ArrayContract(
    schema_id="palette.array.refined_detection.source.frame_row_offsets",
    schema_version=1,
    dtype=INT64,
    shape_template=("n_frame_boundaries",),
    axis_names=("camera_frame_boundary",),
    description=(
        "Exclusive offsets into the frame-contiguous source-candidate audit table."
    ),
    units="source_detection_rows",
)

REFINED_INSTANCE_SOURCE_RECORDING_FRAME_IDS_V1 = ArrayContract(
    schema_id="palette.array.refined_detection.lineage.source_recording_frame_ids",
    schema_version=1,
    dtype=INT64,
    shape_template=("n_instances",),
    axis_names=("instance",),
    description="One-based acquisition recording-frame ID for a clipped snapshot row.",
    units="recording_frame_id",
)

REFINED_INSTANCE_SOURCE_CLIP_INDICES_V1 = ArrayContract(
    schema_id="palette.array.refined_detection.lineage.source_clip_indices",
    schema_version=1,
    dtype=INT32,
    shape_template=("n_instances",),
    axis_names=("instance",),
    description="Zero-based selected clip ordinal for a clipped snapshot row.",
    units="clip_index",
)

REFINED_INSTANCE_SOURCE_CLIP_LOCAL_FRAME_INDICES_V1 = ArrayContract(
    schema_id=(
        "palette.array.refined_detection.lineage.source_clip_local_frame_indices"
    ),
    schema_version=1,
    dtype=INT32,
    shape_template=("n_instances",),
    axis_names=("instance",),
    description="Zero-based decoded frame within the selected source clip.",
    units="clip_local_frame_index",
)

REFINED_INSTANCE_SOURCE_CLIP_DETECT_ROW_INDEX_V1 = ArrayContract(
    schema_id=(
        "palette.array.refined_detection.lineage.source_clip_detect_row_index"
    ),
    schema_version=1,
    dtype=INT64,
    shape_template=("n_instances",),
    axis_names=("instance",),
    description=(
        "Clip-local raw detection row, or -1 for a manual row without raw lineage."
    ),
    units="clip_detection_row_index",
)

REFINED_INSTANCE_SOURCE_REFINED_ROW_IDS_V1 = ArrayContract(
    schema_id="palette.array.refined_detection.lineage.source_refined_row_ids",
    schema_version=1,
    dtype=INT64,
    shape_template=("n_instances",),
    axis_names=("instance",),
    description="Original clip-local refined row identity.",
    units="clip_refined_row_identity",
)

REFINED_SOURCE_RECORDING_FRAME_IDS_V1 = ArrayContract(
    schema_id=(
        "palette.array.refined_detection.source.lineage.source_recording_frame_ids"
    ),
    schema_version=1,
    dtype=INT64,
    shape_template=("n_source_detections",),
    axis_names=("source_detection",),
    description="One-based acquisition recording-frame ID for a source candidate.",
    units="recording_frame_id",
)

REFINED_SOURCE_CLIP_INDICES_V1 = ArrayContract(
    schema_id="palette.array.refined_detection.source.lineage.source_clip_indices",
    schema_version=1,
    dtype=INT32,
    shape_template=("n_source_detections",),
    axis_names=("source_detection",),
    description="Zero-based selected clip ordinal for a source candidate.",
    units="clip_index",
)

REFINED_SOURCE_CLIP_LOCAL_FRAME_INDICES_V1 = ArrayContract(
    schema_id=(
        "palette.array.refined_detection.source.lineage."
        "source_clip_local_frame_indices"
    ),
    schema_version=1,
    dtype=INT32,
    shape_template=("n_source_detections",),
    axis_names=("source_detection",),
    description="Zero-based decoded frame within the selected source clip.",
    units="clip_local_frame_index",
)

REFINED_SOURCE_CLIP_DETECT_ROW_INDEX_V1 = ArrayContract(
    schema_id=(
        "palette.array.refined_detection.source.lineage."
        "source_clip_detect_row_index"
    ),
    schema_version=1,
    dtype=INT64,
    shape_template=("n_source_detections",),
    axis_names=("source_detection",),
    description="Original clip-local raw candidate row index.",
    units="clip_detection_row_index",
)

REFINED_SOURCE_RESOLVED_SOURCE_REFINED_ROW_ID_V1 = ArrayContract(
    schema_id=(
        "palette.array.refined_detection.source.lineage."
        "source_resolved_refined_row_id"
    ),
    schema_version=1,
    dtype=INT64,
    shape_template=("n_source_detections",),
    axis_names=("source_detection",),
    description=(
        "Original clip-local resolved refined row ID, or -1 when unaccepted."
    ),
    units="clip_refined_row_identity",
)

CROP_FRAME_INDICES_V1 = ArrayContract(
    schema_id="palette.array.crop.frame_indices",
    schema_version=1,
    dtype=INT64,
    shape_template=("n_instances",),
    axis_names=("instance",),
    description="Acquisition-camera frame index for each crop observation row.",
    units="camera_frame_index",
)

CROP_SOURCE_REFINED_ROW_IDS_V1 = ArrayContract(
    schema_id="palette.array.crop.source_refined_row_ids",
    schema_version=1,
    dtype=INT64,
    shape_template=("n_instances",),
    axis_names=("instance",),
    description=(
        "Stable refined-row identity copied from the exact bound refined "
        "detection snapshot."
    ),
    units="refined_row_identity",
)

CROP_ROI_COORDINATES_FULL_V1 = ArrayContract(
    schema_id="palette.array.crop.roi_coordinates_full",
    schema_version=1,
    dtype=INT32,
    shape_template=("n_instances", 2),
    axis_names=("instance", "xy"),
    description=(
        "Exact integer source-camera top-left of each crop extraction window."
    ),
    units="pixels",
    coordinate_space="source_camera_pixel_index",
)

CROP_ROI_SIZES_FULL_V1 = ArrayContract(
    schema_id="palette.array.crop.roi_sizes_full",
    schema_version=1,
    dtype=INT32,
    shape_template=("n_instances", 2),
    axis_names=("instance", "width_height"),
    description=(
        "Exact positive integer width and height of each source extraction window."
    ),
    units="pixels",
    coordinate_space="source_camera_pixel_extent",
)

CROP_SOURCE_CROP_XYWH_V1 = ArrayContract(
    schema_id="palette.array.crop.source_crop_xywh",
    schema_version=1,
    dtype=FLOAT32,
    shape_template=("n_instances", 4),
    axis_names=("instance", "xywh"),
    description=(
        "Exact float32 projection of integer crop top-left and size in source "
        "camera pixels."
    ),
    units="pixels",
    coordinate_space="source_camera_pixel_edges",
)

CROP_BBOX_ROI_XYXY_V1 = ArrayContract(
    schema_id="palette.array.crop.bbox_roi_xyxy",
    schema_version=1,
    dtype=FLOAT32,
    shape_template=("n_instances", 4),
    axis_names=("instance", "xyxy"),
    description=(
        "Exact float32 detection box translated into the crop-local pixel frame."
    ),
    units="pixels",
    coordinate_space="crop_pixel_edges",
)

CROP_SOURCE_ROW_SIGNATURE_V1 = ArrayContract(
    schema_id="palette.array.crop.source_row_signature",
    schema_version=1,
    dtype=UINT8,
    shape_template=("n_instances", 32),
    axis_names=("instance", "sha256_byte"),
    description=(
        "Per-row source compatibility signature for incremental crop "
        "materialization."
    ),
    units="sha256_digest_byte",
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


DETECTION_ARRAY_CONTRACTS = ArrayContractCatalog(
    (
        DETECTION_FRAME_INDICES_V1,
        DETECTION_SOURCE_ACQUISITION_FRAME_INDEX_V1,
        DETECTION_INSTANCE_KEY_V1,
        DETECTION_BBOX_NORM_COORDS_V1,
        DETECTION_BBOX_IMG_XYXY_V1,
        DETECTION_CENTERS_IMG_XY_V1,
        DETECTION_SCORES_V1,
        DETECTION_CLASS_IDS_V1,
        FRAME_ROW_OFFSETS_V1,
    )
)


REFINED_DETECTION_ARRAY_CONTRACTS = ArrayContractCatalog(
    (
        DETECTION_FRAME_INDICES_V1,
        DETECTION_SOURCE_ACQUISITION_FRAME_INDEX_V1,
        DETECTION_INSTANCE_KEY_V1,
        DETECTION_BBOX_NORM_COORDS_V1,
        DETECTION_BBOX_IMG_XYXY_V1,
        DETECTION_CENTERS_IMG_XY_V1,
        DETECTION_CLASS_IDS_V1,
        FRAME_ROW_OFFSETS_V1,
        REFINED_DETECTION_REFINED_ROW_IDS_V1,
        REFINED_DETECTION_SCORES_V1,
        REFINED_DETECTION_SCORE_VALID_V1,
        REFINED_DETECTION_SOURCE_KIND_CODES_V1,
        REFINED_DETECTION_MANUAL_EDIT_FLAGS_V1,
        REFINED_DETECTION_SOURCE_DETECT_ROW_INDEX_V1,
        REFINED_DETECTION_REASON_CODES_V1,
        REFINED_SOURCE_DETECT_ROW_INDEX_V1,
        REFINED_SOURCE_FRAME_INDICES_V1,
        REFINED_SOURCE_ACQUISITION_FRAME_INDEX_V1,
        REFINED_SOURCE_INSTANCE_KEY_V1,
        REFINED_SOURCE_BBOX_NORM_COORDS_V1,
        REFINED_SOURCE_BBOX_IMG_XYXY_V1,
        REFINED_SOURCE_CENTERS_IMG_XY_V1,
        REFINED_SOURCE_SCORES_V1,
        REFINED_SOURCE_CLASS_IDS_V1,
        REFINED_SOURCE_DECISION_CODES_V1,
        REFINED_SOURCE_RESOLVED_REFINED_ROW_ID_V1,
        REFINED_SOURCE_REASON_CODES_V1,
        REFINED_SOURCE_FRAME_ROW_OFFSETS_V1,
        REFINED_INSTANCE_SOURCE_RECORDING_FRAME_IDS_V1,
        REFINED_INSTANCE_SOURCE_CLIP_INDICES_V1,
        REFINED_INSTANCE_SOURCE_CLIP_LOCAL_FRAME_INDICES_V1,
        REFINED_INSTANCE_SOURCE_CLIP_DETECT_ROW_INDEX_V1,
        REFINED_INSTANCE_SOURCE_REFINED_ROW_IDS_V1,
        REFINED_SOURCE_RECORDING_FRAME_IDS_V1,
        REFINED_SOURCE_CLIP_INDICES_V1,
        REFINED_SOURCE_CLIP_LOCAL_FRAME_INDICES_V1,
        REFINED_SOURCE_CLIP_DETECT_ROW_INDEX_V1,
        REFINED_SOURCE_RESOLVED_SOURCE_REFINED_ROW_ID_V1,
    )
)


CROP_ARRAY_CONTRACTS = ArrayContractCatalog(
    (
        DETECTION_INSTANCE_KEY_V1,
        CROP_SOURCE_REFINED_ROW_IDS_V1,
        CROP_FRAME_INDICES_V1,
        DETECTION_SOURCE_ACQUISITION_FRAME_INDEX_V1,
        FRAME_ROW_OFFSETS_V1,
        DETECTION_BBOX_NORM_COORDS_V1,
        DETECTION_BBOX_IMG_XYXY_V1,
        DETECTION_CENTERS_IMG_XY_V1,
        CROP_ROI_COORDINATES_FULL_V1,
        CROP_ROI_SIZES_FULL_V1,
        CROP_SOURCE_CROP_XYWH_V1,
        CROP_BBOX_ROI_XYXY_V1,
        CROP_SOURCE_ROW_SIGNATURE_V1,
    )
)


CORE_ARRAY_CONTRACTS = ArrayContractCatalog(
    (
        FRAME_COUNTS_V1,
        FRAME_OFFSETS_V1,
        FRAME_ROW_OFFSETS_V1,
        DETECTION_FRAME_INDICES_V1,
        DETECTION_SOURCE_ACQUISITION_FRAME_INDEX_V1,
        DETECTION_INSTANCE_KEY_V1,
        DETECTION_BBOX_NORM_COORDS_V1,
        DETECTION_BBOX_IMG_XYXY_V1,
        DETECTION_CENTERS_IMG_XY_V1,
        DETECTION_SCORES_V1,
        DETECTION_CLASS_IDS_V1,
        REFINED_DETECTION_REFINED_ROW_IDS_V1,
        REFINED_DETECTION_SCORES_V1,
        REFINED_DETECTION_SCORE_VALID_V1,
        REFINED_DETECTION_SOURCE_KIND_CODES_V1,
        REFINED_DETECTION_MANUAL_EDIT_FLAGS_V1,
        REFINED_DETECTION_SOURCE_DETECT_ROW_INDEX_V1,
        REFINED_DETECTION_REASON_CODES_V1,
        REFINED_SOURCE_DETECT_ROW_INDEX_V1,
        REFINED_SOURCE_FRAME_INDICES_V1,
        REFINED_SOURCE_ACQUISITION_FRAME_INDEX_V1,
        REFINED_SOURCE_INSTANCE_KEY_V1,
        REFINED_SOURCE_BBOX_NORM_COORDS_V1,
        REFINED_SOURCE_BBOX_IMG_XYXY_V1,
        REFINED_SOURCE_CENTERS_IMG_XY_V1,
        REFINED_SOURCE_SCORES_V1,
        REFINED_SOURCE_CLASS_IDS_V1,
        REFINED_SOURCE_DECISION_CODES_V1,
        REFINED_SOURCE_RESOLVED_REFINED_ROW_ID_V1,
        REFINED_SOURCE_REASON_CODES_V1,
        REFINED_SOURCE_FRAME_ROW_OFFSETS_V1,
        REFINED_INSTANCE_SOURCE_RECORDING_FRAME_IDS_V1,
        REFINED_INSTANCE_SOURCE_CLIP_INDICES_V1,
        REFINED_INSTANCE_SOURCE_CLIP_LOCAL_FRAME_INDICES_V1,
        REFINED_INSTANCE_SOURCE_CLIP_DETECT_ROW_INDEX_V1,
        REFINED_INSTANCE_SOURCE_REFINED_ROW_IDS_V1,
        REFINED_SOURCE_RECORDING_FRAME_IDS_V1,
        REFINED_SOURCE_CLIP_INDICES_V1,
        REFINED_SOURCE_CLIP_LOCAL_FRAME_INDICES_V1,
        REFINED_SOURCE_CLIP_DETECT_ROW_INDEX_V1,
        REFINED_SOURCE_RESOLVED_SOURCE_REFINED_ROW_ID_V1,
        CROP_FRAME_INDICES_V1,
        CROP_SOURCE_REFINED_ROW_IDS_V1,
        CROP_ROI_COORDINATES_FULL_V1,
        CROP_ROI_SIZES_FULL_V1,
        CROP_SOURCE_CROP_XYWH_V1,
        CROP_BBOX_ROI_XYXY_V1,
        CROP_SOURCE_ROW_SIGNATURE_V1,
        KEYPOINTS_ROI_V1,
        KEYPOINTS_IMG_V1,
        KEYPOINTS_NORM_V1,
        DENSE_SUBJECT_MASKS_ROI_V1,
        CONTOUR_POINTS_XY_V1,
    )
)
