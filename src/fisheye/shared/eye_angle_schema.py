"""Shared exact maintained compact-dense-v2 eye-angle array contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from fisheye.shared.zarr.analysis_array_contracts import (
    AnalysisArrayDeclaration,
    AnalysisAuthorityRole,
)
from fisheye.shared.zarr.array_contracts import (
    BOOL,
    FLOAT32,
    INT64,
    UINT8,
    UINT16,
    UINT64,
    ArrayContract,
)
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode

EYE_ANGLE_ARRAY_SCHEMA_ID = "palette.analysis.eye_angle.compact_dense_arrays"
EYE_ANGLE_ARRAY_SCHEMA_VERSION = 1
EYE_ANGLE_RUN_PARENT = "analysis/eye_angle_runs"
EYE_ANGLE_RUN_SCHEMA_ID = "analysis.eye_angle_runs"
EYE_ANGLE_RUN_SCHEMA_VERSION = 7
EYE_ANGLE_LEGACY_RUN_SCHEMA_VERSION = 6
EYE_ANGLE_LAYOUT_HIERARCHICAL_V1 = "hierarchical_v1"
EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2 = "compact_dense_v2"
EYE_ANGLE_LAYOUT = EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2
EYE_ANGLE_LAYOUT_CHOICES = (
    EYE_ANGLE_LAYOUT_HIERARCHICAL_V1,
    EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2,
)
EYE_ANGLE_LAYOUT_DEFAULT = EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2
EYE_ANGLE_LEGACY_RUN_CONTRACTS = frozenset(
    {
        (EYE_ANGLE_RUN_SCHEMA_ID, 2, None),
        (EYE_ANGLE_RUN_SCHEMA_ID, 2, EYE_ANGLE_LAYOUT_HIERARCHICAL_V1),
        (EYE_ANGLE_RUN_SCHEMA_ID, 3, None),
        (EYE_ANGLE_RUN_SCHEMA_ID, 3, EYE_ANGLE_LAYOUT_HIERARCHICAL_V1),
        (EYE_ANGLE_RUN_SCHEMA_ID, 4, None),
        (EYE_ANGLE_RUN_SCHEMA_ID, 4, EYE_ANGLE_LAYOUT_HIERARCHICAL_V1),
        (EYE_ANGLE_RUN_SCHEMA_ID, 5, None),
        (EYE_ANGLE_RUN_SCHEMA_ID, 5, EYE_ANGLE_LAYOUT_HIERARCHICAL_V1),
        (EYE_ANGLE_RUN_SCHEMA_ID, 5, EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2),
        (
            EYE_ANGLE_RUN_SCHEMA_ID,
            EYE_ANGLE_LEGACY_RUN_SCHEMA_VERSION,
            EYE_ANGLE_LAYOUT_HIERARCHICAL_V1,
        ),
        (
            EYE_ANGLE_RUN_SCHEMA_ID,
            EYE_ANGLE_LEGACY_RUN_SCHEMA_VERSION,
            EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2,
        ),
    }
)
EYE_ANGLE_ARRAY_SCHEMA_ATTR = "eye_angle_array_schema"
EYE_ANGLE_PHYSICAL_POLICY_OWNER = "eye_angle_semantic_dense_v2"
EYE_ANGLE_COLUMN_ORDER_SCHEMA_ID = "palette.eye_angle_semantic_column_order.v1"
EYE_ANGLE_COLUMN_ORDER_PROFILE = "semantic_bundles_v1"
EYE_ANGLE_TEXT_ENCODING = "uint8_fixed_width_null_terminated_utf8"

ROI_ANGLE_BASE_OUTPUTS = (
    "left_deg",
    "right_deg",
    "left_signed_deg",
    "right_signed_deg",
    "left_major_signed_deg",
    "right_major_signed_deg",
    "left_eye_angle_deg",
    "right_eye_angle_deg",
    "vergence_eye_angle_deg",
    "vergence_deg",
    "vergence_signed_deg",
    "vergence_major_signed_deg",
    "version_deg",
    "version_major_deg",
    "left_minor_signed_deg",
    "right_minor_signed_deg",
    "vergence_minor_signed_deg",
    "version_minor_deg",
    "left_gaze_deg",
    "right_gaze_deg",
    "left_gaze_signed_deg",
    "right_gaze_signed_deg",
    "vergence_gaze_deg",
    "vergence_gaze_signed_deg",
    "left_nasal_gaze_deg",
    "right_nasal_gaze_deg",
    "mean_eye_vergence_gaze_deg",
    "version_gaze_deg",
    "heading_deg",
    "left_centroid_deg",
    "right_centroid_deg",
    "vergence_centroid_deg",
)
FRAME_ANGLE_BASE_OUTPUTS = (
    "left_deg",
    "right_deg",
    "vergence_deg",
    "vergence_signed_deg",
    "vergence_major_signed_deg",
    "left_eye_angle_deg",
    "right_eye_angle_deg",
    "vergence_eye_angle_deg",
    "version_deg",
    "version_major_deg",
    "vergence_minor_signed_deg",
    "version_minor_deg",
    "left_gaze_deg",
    "right_gaze_deg",
    "left_gaze_signed_deg",
    "right_gaze_signed_deg",
    "vergence_gaze_deg",
    "vergence_gaze_signed_deg",
    "left_nasal_gaze_deg",
    "right_nasal_gaze_deg",
    "mean_eye_vergence_gaze_deg",
    "version_gaze_deg",
    "left_centroid_deg",
    "right_centroid_deg",
    "vergence_centroid_deg",
)
ROI_DERIVATIVE_OUTPUTS = (
    "left_speed_deg_s",
    "right_speed_deg_s",
    "vergence_speed_deg_s",
    "vergence_signed_speed_deg_s",
    "version_speed_deg_s",
    "left_gaze_speed_deg_s",
    "right_gaze_speed_deg_s",
    "vergence_gaze_speed_deg_s",
    "vergence_gaze_signed_speed_deg_s",
    "version_gaze_speed_deg_s",
    "mean_eye_vergence_gaze_speed_deg_s",
    "left_accel_deg_s2",
    "right_accel_deg_s2",
    "vergence_accel_deg_s2",
    "vergence_signed_accel_deg_s2",
    "version_accel_deg_s2",
    "left_gaze_accel_deg_s2",
    "right_gaze_accel_deg_s2",
    "vergence_gaze_accel_deg_s2",
    "vergence_gaze_signed_accel_deg_s2",
    "version_gaze_accel_deg_s2",
    "mean_eye_vergence_gaze_accel_deg_s2",
)
ROI_VECTOR_CHANNELS = ("left_gaze_xy", "right_gaze_xy")
ROI_QA_CHANNELS = (
    "left_major_axis_marginal",
    "major_axis_marginal",
    "reason_codes",
    "right_major_axis_marginal",
    "valid_frame",
    "valid_left",
    "valid_right",
)
FRAME_QA_CHANNELS = ("major_axis_marginal", "reason_codes", "valid_frame")

_PRIMARY = (
    "left_eye_angle_deg",
    "right_eye_angle_deg",
    "vergence_eye_angle_deg",
    "left_eye_angle_deg_smoothed",
    "right_eye_angle_deg_smoothed",
    "vergence_eye_angle_deg_smoothed",
    "left_gaze_signed_deg",
    "right_gaze_signed_deg",
    "vergence_gaze_deg",
    "left_gaze_signed_deg_smoothed",
    "right_gaze_signed_deg_smoothed",
    "vergence_gaze_deg_smoothed",
    "left_nasal_gaze_deg",
    "right_nasal_gaze_deg",
    "mean_eye_vergence_gaze_deg",
    "mean_eye_vergence_gaze_deg_smoothed",
)
_BASE_BUNDLES = (
    ("left_eye_angle_deg", "right_eye_angle_deg", "vergence_eye_angle_deg"),
    ("left_gaze_signed_deg", "right_gaze_signed_deg", "vergence_gaze_deg"),
    ("left_nasal_gaze_deg", "right_nasal_gaze_deg", "mean_eye_vergence_gaze_deg"),
    ("left_major_signed_deg", "right_major_signed_deg", "vergence_major_signed_deg"),
    ("left_centroid_deg", "right_centroid_deg", "vergence_centroid_deg"),
    ("left_deg", "right_deg", "vergence_deg"),
    ("left_signed_deg", "right_signed_deg", "vergence_signed_deg"),
    ("left_minor_signed_deg", "right_minor_signed_deg", "vergence_minor_signed_deg"),
    ("left_gaze_deg", "right_gaze_deg", "vergence_gaze_signed_deg"),
)
_KINEMATIC_BUNDLES = (
    ("left_speed_deg_s", "right_speed_deg_s", "vergence_speed_deg_s"),
    ("left_gaze_speed_deg_s", "right_gaze_speed_deg_s", "vergence_gaze_speed_deg_s"),
    ("left_accel_deg_s2", "right_accel_deg_s2", "vergence_accel_deg_s2"),
    ("left_gaze_accel_deg_s2", "right_gaze_accel_deg_s2", "vergence_gaze_accel_deg_s2"),
)


def _variant(base: str, variant: str) -> str:
    if variant == "raw":
        return base
    if variant == "smoothed":
        return f"{base}_smoothed"
    stem = base[:-4] if base.endswith("_deg") else base
    return f"{stem}_{'delta_deg' if variant == 'delta' else 'delta_deg_smoothed'}"


def semantic_angle_channel_order(
    names: Sequence[str], *, block_width: int = 16
) -> tuple[str, ...]:
    width = int(block_width)
    if width <= 0:
        raise ValueError("Eye-angle semantic block width must be positive.")
    available = {str(name) for name in names}
    primary = tuple(name for name in _PRIMARY if name in available)
    bundles: list[tuple[str, ...]] = [primary] if 2 <= len(primary) <= width else []
    for base_bundle in _BASE_BUNDLES:
        for variant in ("raw", "smoothed", "delta", "delta_smoothed"):
            bundle = tuple(
                _variant(base, variant)
                for base in base_bundle
                if _variant(base, variant) in available
            )
            if len(bundle) >= 2:
                bundles.append(bundle)
    bundles.extend(
        tuple(name for name in bundle if name in available)
        for bundle in _KINEMATIC_BUNDLES
    )
    bundles = [bundle for bundle in bundles if len(bundle) >= 2]
    priority = {name for bundle in bundles for name in bundle}
    filler = [name for name in sorted(available) if name not in priority]
    ordered: list[str] = []
    used: set[str] = set()
    for bundle in bundles:
        fresh = [name for name in bundle if name not in used]
        remaining = width - (len(ordered) % width) if ordered else width
        if remaining != width and len(fresh) > remaining:
            while filler and len(ordered) % width:
                name = filler.pop(0)
                ordered.append(name)
                used.add(name)
        for name in fresh:
            ordered.append(name)
            used.add(name)
    ordered.extend(name for name in filler if name not in used)
    ordered.extend(
        name for name in sorted(available) if name not in used and name not in filler
    )
    if len(ordered) != len(available) or set(ordered) != available:
        raise RuntimeError("Semantic eye-angle ordering lost or duplicated channels.")
    return tuple(ordered)


def _expanded(
    base_names: Sequence[str],
    *,
    raw_only: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    result: list[str] = []
    for name in base_names:
        result.append(name)
        if name not in raw_only:
            result.extend(
                (
                    _variant(name, "smoothed"),
                    _variant(name, "delta"),
                    _variant(name, "delta_smoothed"),
                )
            )
    return tuple(result)


_ROI_RAW_ONLY = frozenset(
    {
        "heading_deg",
        "left_major_signed_deg",
        "right_major_signed_deg",
        "vergence_major_signed_deg",
        "version_major_deg",
    }
)
CANONICAL_ROI_ANGLE_CHANNELS = (
    _expanded(ROI_ANGLE_BASE_OUTPUTS, raw_only=_ROI_RAW_ONLY) + ROI_DERIVATIVE_OUTPUTS
)
CANONICAL_FRAME_ANGLE_CHANNELS = _expanded(FRAME_ANGLE_BASE_OUTPUTS)


def canonical_angle_channels(block_width: int = 16) -> tuple[str, ...]:
    return semantic_angle_channel_order(
        tuple(
            sorted(
                set(CANONICAL_ROI_ANGLE_CHANNELS) | set(CANONICAL_FRAME_ANGLE_CHANNELS)
            )
        ),
        block_width=block_width,
    )


CANONICAL_ANGLE_CHANNELS = canonical_angle_channels()


def is_supported_legacy_eye_angle_run(attrs: Mapping[str, Any]) -> bool:
    """Return whether attrs name one exact, closed v2-v6 compatibility layout."""

    identity = (
        attrs.get("schema_id"),
        attrs.get("schema_version"),
        attrs.get("layout"),
    )
    return (
        type(identity[0]) is str
        and type(identity[1]) is int
        and (identity[2] is None or type(identity[2]) is str)
        and identity in EYE_ANGLE_LEGACY_RUN_CONTRACTS
    )


def is_current_eye_angle_run_contract(attrs: Mapping[str, Any]) -> bool:
    """Return whether attrs name the exact maintained compact-v7 contract."""

    return (
        type(attrs.get("schema_id")) is str
        and attrs.get("schema_id") == EYE_ANGLE_RUN_SCHEMA_ID
        and type(attrs.get("schema_version")) is int
        and attrs.get("schema_version") == EYE_ANGLE_RUN_SCHEMA_VERSION
        and type(attrs.get("layout")) is str
        and attrs.get("layout") == EYE_ANGLE_LAYOUT
    )


def _eye_for_channel(name: str) -> str:
    if name.startswith("left_"):
        return "left"
    if name.startswith("right_"):
        return "right"
    if name.startswith(("vergence_", "version_", "mean_eye_vergence_")):
        return "binocular"
    return "none"


def _value_kind_for_angle_channel(name: str) -> str:
    if name.endswith("_accel_deg_s2"):
        return "acceleration"
    if name.endswith("_speed_deg_s"):
        return "speed"
    if "delta_deg" in name:
        return "delta"
    if name.startswith("vergence_") or name.startswith("mean_eye_vergence_"):
        return "vergence"
    if name.startswith("version_"):
        return "version"
    if name == "heading_deg":
        return "heading"
    return "angle"


def _units_for_angle_channel(name: str) -> str:
    if name.endswith("_accel_deg_s2"):
        return "deg/s2"
    if name.endswith("_speed_deg_s"):
        return "deg/s"
    return "deg"


def _representation_for_angle_channel(name: str) -> str:
    if name == "heading_deg":
        return "body_frame_compatibility_alias"
    if "centroid" in name:
        return "centroid"
    if "nasal_gaze" in name or name.startswith("mean_eye_vergence_gaze"):
        return "nasal_gaze"
    if "eye_angle" in name:
        return "eye_frame"
    if "gaze" in name:
        return "gaze"
    if "major" in name:
        return "major"
    if "minor" in name:
        return "legacy_minor"
    if name in {
        "left_deg",
        "right_deg",
        "left_signed_deg",
        "right_signed_deg",
        "vergence_deg",
        "vergence_signed_deg",
    }:
        return "legacy"
    return "major" if name == "version_deg" else "legacy"


def _alias_target_for_angle_channel(name: str) -> str:
    aliases = {
        "heading_deg": "support/body_frame/heading_deg",
        "left_signed_deg": "left_major_signed_deg",
        "right_signed_deg": "right_major_signed_deg",
        "left_minor_signed_deg": "left_gaze_signed_deg",
        "right_minor_signed_deg": "right_gaze_signed_deg",
        "vergence_minor_signed_deg": "vergence_gaze_deg",
        "version_minor_deg": "version_gaze_deg",
        "vergence_deg": "vergence_major_signed_deg",
        "vergence_signed_deg": "vergence_major_signed_deg",
        "version_deg": "version_major_deg",
    }
    return aliases.get(name, "")


def _angle_channel_from_stem(stem: str) -> str:
    return stem if stem.endswith("_deg") else f"{stem}_deg"


def _source_channel_for_angle_channel(name: str) -> str:
    if name.endswith("_delta_deg_smoothed"):
        return _angle_channel_from_stem(name[: -len("_delta_deg_smoothed")])
    if name.endswith("_delta_deg"):
        return _angle_channel_from_stem(name[: -len("_delta_deg")])
    if name.endswith("_smoothed"):
        return name[: -len("_smoothed")]
    if name.endswith("_speed_deg_s"):
        return _angle_channel_from_stem(name[: -len("_speed_deg_s")])
    if name.endswith("_accel_deg_s2"):
        return f"{name[: -len('_accel_deg_s2')]}_speed_deg_s"
    return _alias_target_for_angle_channel(name)


def _formula_for_angle_channel(name: str) -> str:
    if name.endswith("_delta_deg_smoothed"):
        return "abs(smoothed_source_channel[row] - smoothed_source_channel[row - 1])"
    if name.endswith("_delta_deg"):
        return "abs(source_channel[row] - source_channel[row - 1])"
    if name.endswith("_smoothed"):
        return "nan_aware_centered_boxcar(source_channel)"
    if name.endswith("_speed_deg_s"):
        return "backward_difference_to_previous_valid(source_channel, time_seconds)"
    if name.endswith("_accel_deg_s2"):
        return "backward_difference_to_previous_valid(speed_channel, time_seconds)"
    formulas = {
        "heading_deg": "exact_value_alias(support/body_frame/heading_deg)",
        "left_eye_angle_deg": "-left_major_signed_deg",
        "right_eye_angle_deg": "right_major_signed_deg",
        "vergence_eye_angle_deg": "left_eye_angle_deg + right_eye_angle_deg",
        "mean_eye_vergence_gaze_deg": (
            "0.5 * (left_nasal_gaze_deg + right_nasal_gaze_deg)"
        ),
    }
    return formulas.get(name, "")


def eye_angle_channel_metadata(
    channel_names: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Return writer/validator-shared scalar-angle channel metadata."""

    names = tuple(channel_names)
    return {
        "name": names,
        "representation": tuple(
            _representation_for_angle_channel(name) for name in names
        ),
        "eye": tuple(_eye_for_channel(name) for name in names),
        "value_kind": tuple(_value_kind_for_angle_channel(name) for name in names),
        "units": tuple(_units_for_angle_channel(name) for name in names),
        "source_channel": tuple(
            _source_channel_for_angle_channel(name) for name in names
        ),
        "formula": tuple(_formula_for_angle_channel(name) for name in names),
        "compatibility_alias_of": tuple(
            _alias_target_for_angle_channel(name) for name in names
        ),
    }


def eye_vector_channel_metadata(
    channel_names: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Return writer/validator-shared gaze-vector channel metadata."""

    names = tuple(channel_names)
    return {
        "name": names,
        "representation": tuple(
            "gaze" if "gaze" in name else "support" for name in names
        ),
        "eye": tuple(_eye_for_channel(name) for name in names),
        "value_kind": tuple("unit_vector_xy" for _name in names),
        "units": tuple("unitless" for _name in names),
    }


def eye_qa_channel_metadata(
    channel_names: Sequence[str],
    *,
    dtype_by_name: Mapping[str, str],
) -> dict[str, tuple[str, ...]]:
    """Return writer/validator-shared QA channel metadata."""

    names = tuple(channel_names)
    return {
        "name": names,
        "value_kind": tuple(
            (
                "reason_code"
                if name == "reason_codes"
                else "warning_flag" if "marginal" in name else "validity_flag"
            )
            for name in names
        ),
        "dtype": tuple(dtype_by_name.get(name, "uint16") for name in names),
    }


def expected_eye_angle_channel_index_content(
    *, angle_block_width: int
) -> dict[str, tuple[str, ...] | tuple[bool, ...]]:
    """Return the exact semantic payload of all three channel-index groups."""

    angle_names = canonical_angle_channels(angle_block_width)
    roi_sets = {
        "angle_channel_index": set(CANONICAL_ROI_ANGLE_CHANNELS),
        "vector_channel_index": set(ROI_VECTOR_CHANNELS),
        "qa_channel_index": set(ROI_QA_CHANNELS),
    }
    frame_sets = {
        "angle_channel_index": set(CANONICAL_FRAME_ANGLE_CHANNELS),
        "vector_channel_index": set(),
        "qa_channel_index": set(FRAME_QA_CHANNELS),
    }
    names = {
        "angle_channel_index": angle_names,
        "vector_channel_index": ROI_VECTOR_CHANNELS,
        "qa_channel_index": ROI_QA_CHANNELS,
    }
    content: dict[str, tuple[str, ...] | tuple[bool, ...]] = {}
    family_metadata = {
        "angle_channel_index": eye_angle_channel_metadata(angle_names),
        "vector_channel_index": eye_vector_channel_metadata(ROI_VECTOR_CHANNELS),
        "qa_channel_index": eye_qa_channel_metadata(
            ROI_QA_CHANNELS,
            dtype_by_name={
                name: "uint16" if name == "reason_codes" else "bool"
                for name in ROI_QA_CHANNELS
            },
        ),
    }
    for group, metadata in family_metadata.items():
        for field, values in metadata.items():
            content[f"{group}/{field}"] = values
    for group, group_names in names.items():
        content[f"{group}/roi_available"] = tuple(
            name in roi_sets[group] for name in group_names
        )
        content[f"{group}/frame_available"] = tuple(
            name in frame_sets[group] for name in group_names
        )
    return content


def expected_eye_angle_channel_index_attrs(
    *, angle_block_width: int
) -> dict[str, dict[str, object]]:
    """Return exact group attributes for all compact-v7 channel indexes."""

    counts = {
        "angle_channel_index": len(canonical_angle_channels(angle_block_width)),
        "vector_channel_index": len(ROI_VECTOR_CHANNELS),
        "qa_channel_index": len(ROI_QA_CHANNELS),
    }
    return {
        group_name: eye_angle_channel_index_attrs(
            group_name,
            channel_count=channel_count,
        )
        for group_name, channel_count in counts.items()
    }


def eye_angle_channel_index_attrs(
    group_name: str,
    *,
    channel_count: int,
) -> dict[str, object]:
    """Return the exact attrs written for one semantic index group."""

    if type(channel_count) is not int or channel_count < 0:
        raise ValueError("channel_count must be a nonnegative exact integer.")
    attrs: dict[str, object] = {
        "channel_count": channel_count,
        "encoding": EYE_ANGLE_TEXT_ENCODING,
        "axis": 1,
    }
    if group_name == "angle_channel_index":
        attrs.update(
            {
                "logical_lookup": "name",
                "physical_order_schema_id": EYE_ANGLE_COLUMN_ORDER_SCHEMA_ID,
                "physical_order_profile": EYE_ANGLE_COLUMN_ORDER_PROFILE,
            }
        )
    elif group_name == "vector_channel_index":
        attrs["component_axis"] = 2
    elif group_name != "qa_channel_index":
        raise ValueError(f"Unknown eye-angle channel-index group {group_name!r}.")
    return attrs


@dataclass(frozen=True)
class EyeAngleDimensions:
    n_roi_rows: int
    n_frames: int
    angle_block_width: int = 16

    def __post_init__(self) -> None:
        if type(self.n_roi_rows) is not int or self.n_roi_rows < 0:
            raise ValueError("n_roi_rows must be a nonnegative exact integer.")
        if type(self.n_frames) is not int or self.n_frames <= 0:
            raise ValueError(
                "Maintained eye-angle v7 runs require a positive frame count."
            )
        if type(self.angle_block_width) is not int or self.angle_block_width < 3:
            raise ValueError(
                "angle_block_width must be an exact integer of at least three."
            )

    @property
    def contract_dimensions(self) -> dict[str, int]:
        return {
            "n_roi_rows": self.n_roi_rows,
            "n_frames": self.n_frames,
            "n_angle_channels": len(CANONICAL_ANGLE_CHANNELS),
            "n_vector_channels": len(ROI_VECTOR_CHANNELS),
            "n_qa_channels": len(ROI_QA_CHANNELS),
            "angle_block_width": self.angle_block_width,
        }


def eye_angle_dimensions_from_run_attrs(
    attrs: Mapping[str, Any],
) -> EyeAngleDimensions:
    """Parse exact compact-v7 dimensions without numeric coercion."""

    order_contract = attrs.get("angle_column_order_contract")
    if type(order_contract) is not dict:
        raise ValueError("angle_column_order_contract is missing or invalid.")
    n_roi_rows = attrs.get("num_detections")
    n_frames = attrs.get("num_frames")
    if type(n_roi_rows) is not int or n_roi_rows < 0:
        raise ValueError("num_detections must be an exact nonnegative integer.")
    if type(n_frames) is not int or n_frames <= 0:
        raise ValueError("num_frames must be an exact positive integer.")
    return EyeAngleDimensions(
        n_roi_rows=n_roi_rows,
        n_frames=n_frames,
        angle_block_width=order_contract.get("semantic_bundle_width"),
    )


@dataclass(frozen=True)
class EyeAngleSchemaIssue:
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
        schema_id="palette.array.eye_angle." + path.replace("/", "."),
        schema_version=1,
        dtype=dtype,
        shape_template=shape,
        axis_names=axes,
        description=f"Maintained compact eye-angle array {path}.",
        units=units,
        coordinate_space=coordinate_space,
    )


def _declaration(
    path: str,
    dtype: Any,
    shape: tuple[str | int, ...],
    axes: tuple[str, ...],
    *,
    access: AccessPattern,
    authority: str,
    fill: str,
    null: str = "none",
    units: str | None = None,
    coordinate_space: str | None = None,
    physical_owner: str = EYE_ANGLE_PHYSICAL_POLICY_OWNER,
    byte_planner_adopted: bool = False,
) -> AnalysisArrayDeclaration:
    if authority == "compatibility_alias":
        authority_role = AnalysisAuthorityRole.COMPATIBILITY_ALIAS
    elif authority.startswith("semantic_channel"):
        authority_role = AnalysisAuthorityRole.SEMANTIC_METADATA
    elif (
        "quality" in authority
        or "validity" in authority
        or "failure_reason" in authority
    ):
        authority_role = AnalysisAuthorityRole.QUALITY_DIAGNOSTIC
    elif authority in {
        "observation_identity",
        "source_frame_identity",
        "row_time_coordinate",
        "frame_time_coordinate",
    }:
        authority_role = AnalysisAuthorityRole.LINEAGE_INDEX
    else:
        authority_role = AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY
    return AnalysisArrayDeclaration(
        path=path,
        contract=_contract(
            path, dtype, shape, axes, units=units, coordinate_space=coordinate_space
        ),
        required=True,
        access_pattern=access,
        write_mode=WriteMode.IMMUTABLE,
        authority_role=authority_role,
        fill_semantics=fill,
        null_semantics=null,
        physical_policy_owner=physical_owner,
        byte_planner_adopted=byte_planner_adopted,
    )


def build_eye_angle_array_declarations(
    *, byte_planner_adopted: bool = False
) -> tuple[AnalysisArrayDeclaration, ...]:
    """Return the exact 41-array inventory for one physical-policy mode.

    The default retains the established production declaration unchanged.
    Candidate byte-planned runs opt in explicitly and differ only in the
    physical-policy ownership fields; logical paths, shapes, dtypes, authority,
    fill, and null semantics remain identical.
    """

    if type(byte_planner_adopted) is not bool:
        raise TypeError("byte_planner_adopted must be an exact bool.")

    def declared(*args: Any, **kwargs: Any) -> AnalysisArrayDeclaration:
        kwargs["byte_planner_adopted"] = byte_planner_adopted
        if byte_planner_adopted:
            kwargs["physical_owner"] = "eye_angle_byte_planner_candidate_v1"
        return _declaration(*args, **kwargs)

    declarations = [
        declared(
            "roi_angles",
            FLOAT32,
            ("n_roi_rows", "n_angle_channels"),
            ("observation", "angle_channel"),
            access=AccessPattern.WINDOWED,
            authority="derived_analysis_payload",
            fill="NaN is the only invalid or axis-unavailable sentinel; finite values follow each channel-index declaration",
            null="no separate null bitmap",
            units="channel_index_declared",
            coordinate_space="mixed_angle_domain_declared_by_angle_channel_index",
            physical_owner="eye_angle_materializer_explicit_semantic_shards",
        ),
        declared(
            "frame_angles",
            FLOAT32,
            ("n_frames", "n_angle_channels"),
            ("camera_frame", "angle_channel"),
            access=AccessPattern.WINDOWED,
            authority="derived_analysis_payload",
            fill="NaN is the only invalid or axis-unavailable sentinel; finite values follow each channel-index declaration",
            null="no separate null bitmap",
            units="channel_index_declared",
            coordinate_space="mixed_angle_domain_declared_by_angle_channel_index",
            physical_owner="eye_angle_materializer_explicit_semantic_shards",
        ),
        declared(
            "roi_vectors",
            FLOAT32,
            ("n_roi_rows", "n_vector_channels", 2),
            ("observation", "vector_channel", "xy"),
            access=AccessPattern.WINDOWED,
            authority="derived_analysis_payload",
            fill="both components are NaN when the row or declared vector is invalid; no finite numeric sentinel",
            null="no separate null bitmap",
            units="unitless",
            coordinate_space="roi_image_xy_unit_vector",
        ),
        declared(
            "roi_qa",
            UINT16,
            ("n_roi_rows", "n_qa_channels"),
            ("observation", "qa_channel"),
            access=AccessPattern.WINDOWED,
            authority="derived_quality_payload",
            fill="zero means false for boolean channels and no reason bits for reason_codes; no missing sentinel",
            null="all rows are present",
        ),
        declared(
            "frame_qa",
            UINT16,
            ("n_frames", "n_qa_channels"),
            ("camera_frame", "qa_channel"),
            access=AccessPattern.WINDOWED,
            authority="derived_quality_payload",
            fill="zero means false for boolean channels and no reason bits for reason_codes; no missing sentinel",
            null="all frames are present",
        ),
    ]
    text_specs = {
        "angle_channel_index": (
            ("name", 256),
            ("representation", 256),
            ("eye", 64),
            ("value_kind", 64),
            ("units", 64),
            ("source_channel", 256),
            ("formula", 512),
            ("compatibility_alias_of", 256),
        ),
        "vector_channel_index": (
            ("name", 256),
            ("representation", 256),
            ("eye", 64),
            ("value_kind", 64),
            ("units", 64),
        ),
        "qa_channel_index": (("name", 256), ("value_kind", 256), ("dtype", 64)),
    }
    dimension = {
        "angle_channel_index": "n_angle_channels",
        "vector_channel_index": "n_vector_channels",
        "qa_channel_index": "n_qa_channels",
    }
    for group, specs in text_specs.items():
        for name, width in specs:
            declarations.append(
                declared(
                    f"{group}/{name}",
                    UINT8,
                    (dimension[group], width),
                    ("channel", "utf8_byte"),
                    access=AccessPattern.EAGER,
                    authority="semantic_channel_index",
                    fill="zero_padded_null_terminated_utf8",
                    null="empty_string_is_all_zero_bytes",
                )
            )
        for name in ("roi_available", "frame_available"):
            declarations.append(
                declared(
                    f"{group}/{name}",
                    BOOL,
                    (dimension[group],),
                    ("channel",),
                    access=AccessPattern.EAGER,
                    authority="semantic_channel_availability",
                    fill="false_means_channel_unavailable_on_axis",
                )
            )
    support_specs = (
        (
            "instance_key",
            UINT64,
            "identity_key",
            "observation_identity",
            "every row stores its upstream observation identity; no fill or missing sentinel",
            "all rows are present",
        ),
        (
            "source_acquisition_frame_index",
            INT64,
            "acquisition_frame_index",
            "source_frame_identity",
            "every row stores a nonnegative acquisition-frame index; negative sentinels are forbidden",
            "all rows are present",
        ),
        (
            "frame_indices",
            INT64,
            "acquisition_frame_index",
            "compatibility_alias",
            "every value exactly equals support/source_acquisition_frame_index; negative sentinels are forbidden",
            "all rows are present",
        ),
        (
            "time_seconds",
            FLOAT32,
            "s",
            "row_time_coordinate",
            "finite source_acquisition_frame_index divided by the positive run fps; no numeric sentinel",
            "all rows are present",
        ),
        (
            "ellipse_major",
            FLOAT32,
            "px",
            "derived_geometry_support",
            "finite positive fitted major-axis length when available; NaN is the only unavailable or invalid sentinel",
            "no separate null bitmap",
        ),
        (
            "ellipse_minor",
            FLOAT32,
            "px",
            "derived_geometry_support",
            "finite positive fitted minor-axis length when available; NaN is the only unavailable or invalid sentinel",
            "no separate null bitmap",
        ),
        (
            "ellipse_ratio",
            FLOAT32,
            "ratio",
            "derived_geometry_support",
            "finite minor divided by major when available; NaN is the only unavailable or invalid sentinel",
            "no separate null bitmap",
        ),
    )
    for name, dtype, units, authority, fill, null in support_specs:
        declarations.append(
            declared(
                f"support/{name}",
                dtype,
                ("n_roi_rows",),
                ("observation",),
                access=AccessPattern.WINDOWED,
                authority=authority,
                fill=fill,
                null=null,
                units=units,
            )
        )
    declarations.append(
        declared(
            "support/frame_time_seconds",
            FLOAT32,
            ("n_frames",),
            ("camera_frame",),
            access=AccessPattern.WINDOWED,
            authority="frame_time_coordinate",
            fill="finite camera-frame index divided by the positive run fps; no numeric sentinel",
            null="all frames are present",
            units="s",
        )
    )
    for name in ("origin_xy", "forward_axis_xy", "left_axis_xy"):
        declarations.append(
            declared(
                f"support/body_frame/{name}",
                FLOAT32,
                ("n_roi_rows", 2),
                ("observation", "xy"),
                access=AccessPattern.WINDOWED,
                authority="derived_body_frame_support",
                fill="both components are NaN exactly when support/body_frame/valid is false; no finite numeric sentinel",
                null="no separate null bitmap",
                coordinate_space=(
                    "roi_pixels" if name == "origin_xy" else "roi_image_xy_unit_vector"
                ),
            )
        )
    declarations.extend(
        (
            declared(
                "support/body_frame/heading_deg",
                FLOAT32,
                ("n_roi_rows",),
                ("observation",),
                access=AccessPattern.WINDOWED,
                authority="derived_body_frame_support",
                fill="NaN exactly when support/body_frame/valid is false; no finite numeric sentinel",
                null="no separate null bitmap",
                units="deg",
                coordinate_space="roi_image_xy_heading_math_ccw_after_y_flip",
            ),
            declared(
                "support/body_frame/valid",
                BOOL,
                ("n_roi_rows",),
                ("observation",),
                access=AccessPattern.WINDOWED,
                authority="derived_body_frame_validity",
                fill="false means body-frame origin axes and heading are invalid; true means all are finite",
                null="all rows are present",
            ),
            declared(
                "support/body_frame/failure_reason_bytes",
                UINT8,
                ("n_roi_rows", 64),
                ("observation", "utf8_byte"),
                access=AccessPattern.WINDOWED,
                authority="derived_body_frame_failure_reason",
                fill="exact UTF-8 reason tag followed by NUL and zero padding; valid rows store ok and invalid rows store a nonempty failure reason",
                null="empty strings are forbidden",
            ),
        )
    )
    paths = [item.path for item in declarations]
    if len(paths) != 41 or len(paths) != len(set(paths)):
        raise RuntimeError(
            "Eye-angle compact-v7 declaration inventory must contain 41 unique arrays."
        )
    return tuple(declarations)


EYE_ANGLE_ARRAY_DECLARATIONS = build_eye_angle_array_declarations()


def eye_angle_array_schema_manifest(
    dimensions: EyeAngleDimensions,
    *,
    byte_planner_adopted: bool = False,
) -> dict[str, object]:
    declarations = build_eye_angle_array_declarations(
        byte_planner_adopted=byte_planner_adopted
    )
    return {
        "schema_id": EYE_ANGLE_ARRAY_SCHEMA_ID,
        "schema_version": EYE_ANGLE_ARRAY_SCHEMA_VERSION,
        "run_schema_id": EYE_ANGLE_RUN_SCHEMA_ID,
        "run_schema_version": EYE_ANGLE_RUN_SCHEMA_VERSION,
        "layout": EYE_ANGLE_LAYOUT,
        "dimensions": dimensions.contract_dimensions,
        "arrays": [item.as_manifest() for item in declarations],
        "forbidden_arrays": ["frame_vectors"],
        "byte_planner_adopted": byte_planner_adopted,
    }


def _materialize(array: Any) -> np.ndarray:
    try:
        return np.asarray(array[:])
    except (IndexError, TypeError):
        return np.asarray(array[...])


def canonical_exact_json_bytes(value: Any, *, path: str = "$") -> bytes:
    """Encode exact JSON types canonically, preserving bool/int/float identity."""

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


def _encoded_text_rows(values: Sequence[str], *, width: int) -> np.ndarray:
    encoded = np.zeros((len(values), int(width)), dtype=np.uint8)
    for row_index, value in enumerate(values):
        payload = value.encode("utf-8")
        if len(payload) >= int(width):
            raise ValueError(
                f"Canonical channel metadata exceeds fixed width {width}: {value!r}."
            )
        encoded[row_index, : len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    return encoded


def collect_eye_angle_channel_index_attrs(
    run_group: Any,
) -> dict[str, dict[str, Any]]:
    """Collect direct attrs for the three exact semantic index groups."""

    result: dict[str, dict[str, Any]] = {}
    for group_name in (
        "angle_channel_index",
        "vector_channel_index",
        "qa_channel_index",
    ):
        try:
            group = run_group[group_name]
        except (KeyError, TypeError):
            continue
        result[group_name] = dict(group.attrs)
    return result


def validate_eye_angle_compact_arrays(
    arrays: Mapping[str, Any],
    *,
    dimensions: EyeAngleDimensions,
    persisted_manifest: Any | None = None,
    channel_index_attrs: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[EyeAngleSchemaIssue, ...]:
    issues: list[EyeAngleSchemaIssue] = []
    expected = {item.path: item for item in build_eye_angle_array_declarations()}
    for path in sorted(set(arrays) - set(expected)):
        issues.append(
            EyeAngleSchemaIssue(
                "unexpected_array", path, "Array is outside compact eye-angle v7."
            )
        )
    for path, declaration in expected.items():
        array = arrays.get(path)
        if array is None:
            if not declaration.required:
                continue
            issues.append(
                EyeAngleSchemaIssue(
                    "missing_required_array",
                    path,
                    "Required compact eye-angle array is absent.",
                )
            )
            continue
        for error in declaration.contract.validate_observation(
            array, dimensions=dimensions.contract_dimensions
        ):
            issues.append(EyeAngleSchemaIssue("array_contract_violation", path, error))
    adopted = (
        isinstance(persisted_manifest, Mapping)
        and persisted_manifest.get("byte_planner_adopted") is True
    )
    expected_manifest = eye_angle_array_schema_manifest(
        dimensions,
        byte_planner_adopted=adopted,
    )
    try:
        manifest_matches = canonical_exact_json_bytes(
            persisted_manifest,
            path=f"$.{EYE_ANGLE_ARRAY_SCHEMA_ATTR}",
        ) == canonical_exact_json_bytes(expected_manifest)
    except (TypeError, ValueError):
        manifest_matches = False
    if not manifest_matches:
        issues.append(
            EyeAngleSchemaIssue(
                "array_schema_manifest_mismatch",
                EYE_ANGLE_ARRAY_SCHEMA_ATTR,
                "Persisted declaration must exactly equal the executable schema.",
            )
        )
    content = expected_eye_angle_channel_index_content(
        angle_block_width=dimensions.angle_block_width
    )
    for path, expected_values in content.items():
        if path not in arrays:
            continue
        try:
            observed = _materialize(arrays[path])
            if expected_values and type(expected_values[0]) is bool:
                expected_values_array = np.asarray(expected_values, dtype=bool)
            else:
                declaration = expected[path]
                width = declaration.contract.shape_template[1]
                if type(width) is not int:
                    raise TypeError(f"Text metadata width is not fixed for {path}.")
                expected_values_array = _encoded_text_rows(
                    expected_values,
                    width=width,
                )
        except Exception as exc:
            issues.append(
                EyeAngleSchemaIssue("channel_index_read_failure", path, str(exc))
            )
            continue
        if not np.array_equal(observed, expected_values_array):
            issue_code = (
                "channel_availability_mismatch"
                if path.endswith("_available")
                else "channel_index_content_mismatch"
            )
            issues.append(
                EyeAngleSchemaIssue(
                    issue_code,
                    path,
                    "Channel metadata is not the exact canonical bytes/order/content.",
                )
            )

    expected_attrs = expected_eye_angle_channel_index_attrs(
        angle_block_width=dimensions.angle_block_width
    )
    if channel_index_attrs is None:
        for group_name in expected_attrs:
            issues.append(
                EyeAngleSchemaIssue(
                    "channel_index_attrs_missing",
                    group_name,
                    "Exact channel-index group attrs were not supplied for validation.",
                )
            )
    else:
        for group_name, group_expected_attrs in expected_attrs.items():
            observed_attrs = channel_index_attrs.get(group_name)
            try:
                attrs_match = canonical_exact_json_bytes(
                    observed_attrs,
                    path=f"$.{group_name}.attrs",
                ) == canonical_exact_json_bytes(group_expected_attrs)
            except (TypeError, ValueError):
                attrs_match = False
            if not attrs_match:
                issues.append(
                    EyeAngleSchemaIssue(
                        "channel_index_attrs_mismatch",
                        group_name,
                        "Channel-index attrs must exactly equal the executable contract.",
                    )
                )
    return tuple(issues)


def collect_eye_angle_arrays(run_group: Any) -> dict[str, Any]:
    arrays: dict[str, Any] = {}

    def visit(group: Any, prefix: str = "") -> None:
        for name in group.array_keys():
            path = f"{prefix}/{name}" if prefix else str(name)
            arrays[path] = group[name]
        for name in group.group_keys():
            child_prefix = f"{prefix}/{name}" if prefix else str(name)
            visit(group[name], child_prefix)

    visit(run_group)
    return arrays


def validate_eye_angle_value_aliases(
    run_group: Any,
) -> tuple[EyeAngleSchemaIssue, ...]:
    """Chunk-scan exact compatibility aliases for publication/audit gates."""

    try:
        dimensions = eye_angle_dimensions_from_run_attrs(run_group.attrs)
        roi_angles = run_group["roi_angles"]
        heading = run_group["support/body_frame/heading_deg"]
        acquisition = run_group["support/source_acquisition_frame_index"]
        frame_alias = run_group["support/frame_indices"]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return (
            EyeAngleSchemaIssue(
                "alias_validation_unavailable",
                "attrs",
                str(exc),
            ),
        )
    issues: list[EyeAngleSchemaIssue] = []
    try:
        heading_column = canonical_angle_channels(
            dimensions.angle_block_width
        ).index("heading_deg")
        row_chunk = max(1, int(roi_angles.chunks[0]))
        for start in range(0, dimensions.n_roi_rows, row_chunk):
            stop = min(dimensions.n_roi_rows, start + row_chunk)
            if not np.array_equal(
                np.asarray(roi_angles[start:stop, heading_column]),
                np.asarray(heading[start:stop]),
                equal_nan=True,
            ):
                issues.append(
                    EyeAngleSchemaIssue(
                        "heading_alias_mismatch",
                        "roi_angles/heading_deg",
                        "Dense heading compatibility values differ from support/body_frame/heading_deg.",
                    )
                )
                break
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        issues.append(
            EyeAngleSchemaIssue(
                "heading_alias_read_failure",
                "roi_angles/heading_deg",
                str(exc),
            )
        )
    try:
        alias_chunk = max(1, int(acquisition.chunks[0]))
        for start in range(0, dimensions.n_roi_rows, alias_chunk):
            stop = min(dimensions.n_roi_rows, start + alias_chunk)
            if not np.array_equal(
                np.asarray(acquisition[start:stop]),
                np.asarray(frame_alias[start:stop]),
            ):
                issues.append(
                    EyeAngleSchemaIssue(
                        "frame_alias_mismatch",
                        "support/frame_indices",
                        "Frame compatibility values differ from support/source_acquisition_frame_index.",
                    )
                )
                break
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        issues.append(
            EyeAngleSchemaIssue(
                "frame_alias_read_failure",
                "support/frame_indices",
                str(exc),
            )
        )
    return tuple(issues)


def validate_eye_angle_compact_run(
    run_group: Any,
    *,
    candidate_storage_validator: Callable[..., Sequence[Any]] | None = None,
) -> tuple[EyeAngleSchemaIssue, ...]:
    attrs = run_group.attrs
    issues: list[EyeAngleSchemaIssue] = []
    expected_attrs = {
        "schema_id": EYE_ANGLE_RUN_SCHEMA_ID,
        "schema_version": EYE_ANGLE_RUN_SCHEMA_VERSION,
        "layout": EYE_ANGLE_LAYOUT,
    }
    for name, expected in expected_attrs.items():
        observed = attrs.get(name)
        if type(observed) is not type(expected) or observed != expected:
            issues.append(
                EyeAngleSchemaIssue(
                    "run_contract_mismatch",
                    name,
                    f"Expected exact {type(expected).__name__} {expected!r}, "
                    f"got {type(observed).__name__} {observed!r}.",
                )
            )
    fps = attrs.get("fps")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not np.isfinite(float(fps))
        or float(fps) <= 0.0
    ):
        issues.append(
            EyeAngleSchemaIssue(
                "invalid_fps",
                "fps",
                "Maintained compact eye-angle v7 requires a positive finite fps and frame_time_seconds.",
            )
        )
    try:
        order_contract = attrs.get("angle_column_order_contract")
        dimensions = eye_angle_dimensions_from_run_attrs(attrs)
    except (TypeError, ValueError) as exc:
        issues.append(EyeAngleSchemaIssue("invalid_dimensions", "attrs", str(exc)))
        return tuple(issues)

    expected_order_keys = {
        "schema_id",
        "profile",
        "logical_lookup",
        "physical_index_semantics",
        "semantic_bundle_width",
        "requested_dense_inner_chunks",
        "effective_roi_chunks",
        "effective_frame_chunks",
        "first_angle_chunk_channels",
    }
    order_errors: list[str] = []
    if set(order_contract) != expected_order_keys:
        order_errors.append(
            "field set differs from the exact compact-v7 order envelope"
        )
    exact_constants = {
        "schema_id": EYE_ANGLE_COLUMN_ORDER_SCHEMA_ID,
        "profile": EYE_ANGLE_COLUMN_ORDER_PROFILE,
        "logical_lookup": "angle_channel_index/name",
        "physical_index_semantics": False,
        "semantic_bundle_width": dimensions.angle_block_width,
    }
    for name, expected in exact_constants.items():
        observed = order_contract.get(name)
        if type(observed) is not type(expected) or observed != expected:
            order_errors.append(
                f"{name} must be exact {type(expected).__name__} {expected!r}"
            )

    requested = order_contract.get("requested_dense_inner_chunks")
    if not (
        type(requested) is list
        and len(requested) == 2
        and all(type(value) is int and value > 0 for value in requested)
        and requested[1] == dimensions.angle_block_width
    ):
        order_errors.append(
            "requested_dense_inner_chunks must be two exact positive integers "
            "whose column width equals semantic_bundle_width"
        )

    for field_name, array_name in (
        ("effective_roi_chunks", "roi_angles"),
        ("effective_frame_chunks", "frame_angles"),
    ):
        observed = order_contract.get(field_name)
        try:
            expected = [int(value) for value in run_group[array_name].chunks]
        except (AttributeError, KeyError, TypeError):
            expected = None
        if expected is not None and (
            type(observed) is not list
            or any(type(value) is not int for value in observed)
            or observed != expected
        ):
            order_errors.append(f"{field_name} must exactly match {array_name}.chunks")

    first_channels = order_contract.get("first_angle_chunk_channels")
    expected_first_channels = list(
        canonical_angle_channels(dimensions.angle_block_width)[
            : dimensions.angle_block_width
        ]
    )
    if (
        type(first_channels) is not list
        or any(type(value) is not str for value in first_channels)
        or first_channels != expected_first_channels
    ):
        order_errors.append(
            "first_angle_chunk_channels must exactly match the executable order"
        )
    if order_errors:
        issues.append(
            EyeAngleSchemaIssue(
                "column_order_contract_mismatch",
                "angle_column_order_contract",
                "; ".join(order_errors),
            )
        )
    issues.extend(
        validate_eye_angle_compact_arrays(
            collect_eye_angle_arrays(run_group),
            dimensions=dimensions,
            persisted_manifest=attrs.get(EYE_ANGLE_ARRAY_SCHEMA_ATTR),
            channel_index_attrs=collect_eye_angle_channel_index_attrs(run_group),
        )
    )
    persisted_array_schema = attrs.get(EYE_ANGLE_ARRAY_SCHEMA_ATTR)
    if (
        isinstance(persisted_array_schema, Mapping)
        and persisted_array_schema.get("byte_planner_adopted") is True
    ):
        if candidate_storage_validator is None:
            issues.append(
                EyeAngleSchemaIssue(
                    "candidate_storage_validator_missing",
                    EYE_ANGLE_ARRAY_SCHEMA_ATTR,
                    "Byte-planned eye-angle validation requires the "
                    "analysis-owned physical-storage validator.",
                )
            )
        else:
            issues.extend(
                EyeAngleSchemaIssue(item.code, item.path, item.message)
                for item in candidate_storage_validator(
                    run_group,
                    dimensions=dimensions,
                )
            )
    return tuple(issues)
