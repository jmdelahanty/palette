"""Exact logical array census for maintained track-kinematics motion runs.

This module freezes the current writer-owned array vocabulary without claiming
that the shared byte-planned array factory can publish it yet.  Two required
lineage arrays use NumPy structured dtypes, which the current shared
``DTypeContract``/``StoragePlan`` boundary cannot round-trip.  Callers can turn
every other declaration into an ``AnalysisArrayDeclaration``; attempting that
conversion for either structured array fails closed.  The separately versioned
flat-lineage candidate replaces those two records with five primitive arrays;
it does not reinterpret or silently upgrade the v1 authority.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

from fisheye.shared.coordinate_identity import (
    TRACK_SAMPLE_INTERPOLATION_DTYPE,
    TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE,
)
from fisheye.shared.zarr.analysis_array_contracts import (
    AnalysisArrayDeclaration,
    AnalysisAuthorityRole,
)
from fisheye.shared.zarr.array_contracts import ArrayContract, DTypeContract
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode

TRACK_KINEMATICS_ARRAY_SCHEMA_ID = "palette.analysis.track_kinematics.motion_arrays"
TRACK_KINEMATICS_ARRAY_SCHEMA_VERSION = 1
TRACK_KINEMATICS_FLAT_LINEAGE_SCHEMA_VERSION = 2
TRACK_KINEMATICS_PHYSICAL_POLICY_OWNER = "track_kinematics_rechunk_v3"
TRACK_KINEMATICS_FLAT_PHYSICAL_POLICY_OWNER = "analysis_storage_planning_v1"

TRACK_KINEMATICS_CORE_BUNDLE = "core_motion_v1"
TRACK_KINEMATICS_PHYSICAL_BUNDLE = "physical_motion_v1"
TRACK_KINEMATICS_ARENA_BUNDLE = "track_arena_inventory_v1"
TRACK_KINEMATICS_FLAT_LINEAGE_BUNDLE = "flat_source_lineage_v2"

TRACK_KINEMATICS_FLAT_LINEAGE_PATHS = (
    "source_frame_interpolation/left_source_frame_index",
    "source_frame_interpolation/right_source_frame_index",
    "source_frame_interpolation/right_weight",
    "source_instance_key/valid",
    "source_instance_key/value",
)

TRACK_KINEMATICS_LEGACY_EXCLUDED_PREFIXES = ("tracks/id_<track_id>/swim_bouts/",)
TRACK_KINEMATICS_LEGACY_EXCLUDED_RUN_ARRAYS = (
    "camera_frame_ids",
    "stimulus_frame_nums",
    "timestamp_ns",
    "trial_state",
    "metadata_mask",
    "angle_unsigned_deg",
    "angle_signed_deg",
    "heading_deg",
    "has_offline",
)


class TrackKinematicsStructuredDTypeBlockedError(ValueError):
    """Raised when a structured declaration reaches the simple dtype boundary."""


def _dtype_fields(dtype: np.dtype[Any]) -> tuple[tuple[str, str, int], ...]:
    if dtype.fields is None:
        return ()
    return tuple(
        (
            str(name),
            str(np.dtype(dtype.fields[name][0])),
            int(dtype.fields[name][1]),
        )
        for name in dtype.names or ()
    )


@dataclass(frozen=True)
class TrackKinematicsExactArrayDeclaration:
    """One exact current track-kinematics array declaration.

    ``relative_path`` is relative either to one ``tracks/id_<N>`` group or to
    the run group, as recorded by ``scope``.  Structured field names, dtypes,
    offsets, and total itemsize are first-class so the checkpoint does not hide
    those identities behind an opaque void dtype.
    """

    relative_path: str
    scope: str
    dtype: np.dtype[Any]
    shape_template: tuple[str | int, ...]
    axis_names: tuple[str, ...]
    required: bool
    bundle: str
    access_pattern: AccessPattern
    authority_role: AnalysisAuthorityRole
    fill_semantics: str
    null_semantics: str
    units: str | None
    coordinate_space: str | None = None

    def __post_init__(self) -> None:
        path = self.relative_path
        if (
            type(path) is not str
            or not path
            or path != path.strip()
            or path.startswith("/")
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise ValueError(f"Invalid track-kinematics array path {path!r}.")
        if self.scope not in {"track", "run"}:
            raise ValueError("Track-kinematics array scope must be track or run.")
        dtype = np.dtype(self.dtype)
        if dtype.hasobject or dtype.itemsize <= 0:
            raise ValueError("Track arrays require exact fixed-width dtypes.")
        object.__setattr__(self, "dtype", dtype)
        if len(self.shape_template) != len(self.axis_names):
            raise ValueError("Track array shape and axis-name ranks differ.")
        object.__setattr__(self, "access_pattern", AccessPattern(self.access_pattern))
        object.__setattr__(
            self,
            "authority_role",
            AnalysisAuthorityRole(self.authority_role),
        )

    @property
    def is_structured(self) -> bool:
        return self.dtype.fields is not None

    @property
    def structured_fields(self) -> tuple[tuple[str, str, int], ...]:
        return _dtype_fields(self.dtype)

    def bind_track(self, track_id: int) -> TrackKinematicsExactArrayDeclaration:
        if self.scope != "track":
            raise ValueError("Only track-scoped declarations bind a track ID.")
        if isinstance(track_id, bool) or not isinstance(track_id, int):
            raise TypeError("track_id must be an exact integer.")
        return replace(
            self,
            relative_path=f"tracks/id_{track_id}/{self.relative_path}",
            scope="run",
        )

    def bind_track_dimensions(
        self,
        track_id: int,
    ) -> TrackKinematicsExactArrayDeclaration:
        """Bind path and per-track dimensions for one run-wide plan receipt."""

        bound = self.bind_track(track_id)
        suffix = str(track_id).replace("-", "neg_")
        dimensions = {
            "n_track_samples": f"n_track_samples_id_{suffix}",
            "n_track_seconds": f"n_track_seconds_id_{suffix}",
        }
        return replace(
            bound,
            shape_template=tuple(
                dimensions.get(value, value) for value in bound.shape_template
            ),
        )

    def as_manifest(self) -> dict[str, object]:
        dtype_record: dict[str, object] = {
            "dtype_id": (
                "numpy_structured_v1" if self.is_structured else str(self.dtype)
            ),
            "numpy_dtype": None if self.is_structured else str(self.dtype),
            "itemsize_bytes": int(self.dtype.itemsize),
            "structured_fields": [
                {"name": name, "numpy_dtype": field_dtype, "offset_bytes": offset}
                for name, field_dtype, offset in self.structured_fields
            ],
        }
        return {
            "path": self.relative_path,
            "scope": self.scope,
            "required": self.required,
            "bundle": self.bundle,
            "logical_contract": {
                "schema_id": (
                    f"{TRACK_KINEMATICS_ARRAY_SCHEMA_ID}."
                    f"{self.relative_path.replace('/', '.')}"
                ),
                "schema_version": TRACK_KINEMATICS_ARRAY_SCHEMA_VERSION,
                "dtype": dtype_record,
                "shape_template": list(self.shape_template),
                "axis_names": list(self.axis_names),
                "description": (
                    f"Maintained exact track-kinematics array {self.relative_path}."
                ),
                "units": self.units,
                "coordinate_space": self.coordinate_space,
            },
            "access_pattern": self.access_pattern.value,
            "write_mode": WriteMode.IMMUTABLE.value,
            "authority_role": self.authority_role.value,
            "fill_semantics": self.fill_semantics,
            "null_semantics": self.null_semantics,
            "physical_policy_owner": TRACK_KINEMATICS_PHYSICAL_POLICY_OWNER,
            "byte_planner_adopted": False,
        }

    def to_analysis_array_declaration(
        self,
        *,
        schema_version: int = TRACK_KINEMATICS_ARRAY_SCHEMA_VERSION,
        byte_planner_adopted: bool = False,
        physical_policy_owner: str = TRACK_KINEMATICS_PHYSICAL_POLICY_OWNER,
    ) -> AnalysisArrayDeclaration:
        """Convert simple fixed dtypes; fail closed for structured lineage."""

        if self.is_structured:
            raise TrackKinematicsStructuredDTypeBlockedError(
                f"{self.relative_path}: structured dtype {self.dtype.descr!r} "
                "cannot be represented by the current shared DTypeContract/"
                "StoragePlan/array-factory boundary."
            )
        dtype = DTypeContract(str(self.dtype), str(self.dtype))
        contract = ArrayContract(
            schema_id=(
                f"{TRACK_KINEMATICS_ARRAY_SCHEMA_ID}."
                f"{self.relative_path.replace('/', '.')}"
            ),
            schema_version=schema_version,
            dtype=dtype,
            shape_template=self.shape_template,
            axis_names=self.axis_names,
            description=(
                f"Maintained exact track-kinematics array {self.relative_path}."
            ),
            units=self.units,
            coordinate_space=self.coordinate_space,
        )
        return AnalysisArrayDeclaration(
            path=self.relative_path,
            contract=contract,
            required=self.required,
            access_pattern=self.access_pattern,
            write_mode=WriteMode.IMMUTABLE,
            authority_role=self.authority_role,
            fill_semantics=self.fill_semantics,
            null_semantics=self.null_semantics,
            physical_policy_owner=physical_policy_owner,
            byte_planner_adopted=byte_planner_adopted,
        )


_CORE_TRACK_PATHS = frozenset(
    {
        "acceleration_px",
        "angular_speed_raw_deg_s",
        "angular_speed_smoothed_deg_s",
        "angular_velocity_deg_s",
        "angular_velocity_raw_deg_s",
        "angular_velocity_smoothed_deg_s",
        "cumulative_path_distance_px",
        "delta_frames",
        "delta_heading_degrees",
        "delta_heading_smoothed_degrees",
        "delta_seconds",
        "detection_source",
        "frame_indices",
        "frame_path_distance_filtered_px",
        "frame_path_distance_raw_px",
        "frame_path_distance_smoothed_px",
        "heading_degrees",
        "heading_per_second_degrees",
        "heading_per_second_resultant",
        "heading_radians",
        "heading_usable",
        "keypoint_success",
        "keypoint_usable",
        "movement/speed/averaged/acceleration_px",
        "movement/speed/averaged/px",
        "movement/speed/averaged/smoothed_acceleration_px",
        "movement/speed/filtered/acceleration_px",
        "movement/speed/filtered/frame_path_distance_px",
        "movement/speed/filtered/px",
        "movement/speed/filtered/smoothed_acceleration_px",
        "movement/speed/raw/acceleration_px",
        "movement/speed/raw/frame_path_distance_px",
        "movement/speed/raw/px",
        "movement/speed/raw/smoothed_acceleration_px",
        "movement/speed/smoothed/acceleration_px",
        "movement/speed/smoothed/frame_path_distance_px",
        "movement/speed/smoothed/px",
        "movement/speed/smoothed/smoothed_acceleration_px",
        "position_finite",
        "positions_px",
        "sample_observed",
        "sample_reason_code",
        "sample_valid",
        "second_indices",
        "smoothed_acceleration_px",
        "smoothed_heading_degrees",
        "smoothed_heading_radians",
        "source_acquisition_frame_index",
        "source_frame_interpolation",
        "source_instance_key",
        "source_observed",
        "source_row_index",
        "speed_averaged_px",
        "speed_derivatives/speed_averaged/acceleration_px",
        "speed_derivatives/speed_averaged/smoothed_acceleration_px",
        "speed_derivatives/speed_filtered/acceleration_px",
        "speed_derivatives/speed_filtered/smoothed_acceleration_px",
        "speed_derivatives/speed_raw/acceleration_px",
        "speed_derivatives/speed_raw/smoothed_acceleration_px",
        "speed_derivatives/speed_smoothed/acceleration_px",
        "speed_derivatives/speed_smoothed/smoothed_acceleration_px",
        "speed_filtered_px",
        "speed_per_second_px",
        "speed_raw_px",
        "speed_smoothed_px",
        "time_seconds",
        "track_sample_key",
        "transition_reason_code",
        "transition_valid",
    }
)

_SECOND_PATHS = frozenset(
    {
        "second_indices",
        "speed_per_second_px",
        "speed_per_second_mm",
        "heading_per_second_degrees",
        "heading_per_second_resultant",
    }
)
_PAIR_PATHS = frozenset({"track_sample_key", "positions_px", "positions_mm"})
_STRUCTURED_DTYPES = {
    "source_frame_interpolation": TRACK_SAMPLE_INTERPOLATION_DTYPE,
    "source_instance_key": TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE,
}
_FLAT_INT64_PATHS = frozenset(
    {
        "source_frame_interpolation/left_source_frame_index",
        "source_frame_interpolation/right_source_frame_index",
    }
)
_FLAT_FLOAT64_PATHS = frozenset({"source_frame_interpolation/right_weight"})
_FLAT_BOOL_PATHS = frozenset({"source_instance_key/valid"})
_FLAT_UINT64_PATHS = frozenset({"source_instance_key/value"})
_BOOL_PATHS = frozenset(
    {
        "heading_usable",
        "keypoint_success",
        "keypoint_usable",
        "position_finite",
        "sample_observed",
        "sample_valid",
        "source_observed",
        "transition_valid",
    }
)
_INT64_PATHS = frozenset(
    {
        "frame_indices",
        "source_acquisition_frame_index",
        "source_row_index",
        "second_indices",
        "track_sample_key",
    }
)
_INT16_PATHS = frozenset({"sample_reason_code", "transition_reason_code"})
_INT8_PATHS = frozenset({"detection_source"})
_INT32_PATHS = frozenset({"delta_frames"})
_FLOAT32_POSITION_PATHS = frozenset({"positions_px", "positions_mm"})

_LINEAGE_PATHS = frozenset(
    {
        "frame_indices",
        "track_sample_key",
        "source_acquisition_frame_index",
        "source_frame_interpolation",
        "source_instance_key",
        "source_row_index",
        "time_seconds",
        "second_indices",
        "detection_source",
    }
)
_QUALITY_PATHS = _BOOL_PATHS | _INT16_PATHS


def _physical_peer(path: str) -> str | None:
    parts = path.split("/")
    leaf = parts[-1]
    if leaf == "px":
        parts[-1] = "mm"
    elif leaf.endswith("_px"):
        parts[-1] = f"{leaf[:-3]}_mm"
    else:
        return None
    return "/".join(parts)


_PHYSICAL_TRACK_PATHS = frozenset(
    peer for path in _CORE_TRACK_PATHS if (peer := _physical_peer(path)) is not None
)


def _alias_target(path: str) -> str | None:
    if path == "frame_indices":
        return "source_acquisition_frame_index"
    if path == "angular_velocity_deg_s":
        return "angular_velocity_raw_deg_s"
    movement = {
        "speed_raw": "raw",
        "speed_filtered": "filtered",
        "speed_smoothed": "smoothed",
        "speed_averaged": "averaged",
    }
    for source, group in movement.items():
        for suffix in ("px", "mm"):
            if path == f"{source}_{suffix}":
                return f"movement/speed/{group}/{suffix}"
    for level in ("raw", "filtered", "smoothed"):
        for suffix in ("px", "mm"):
            if path == f"frame_path_distance_{level}_{suffix}":
                return f"movement/speed/{level}/frame_path_distance_{suffix}"
    for suffix in ("px", "mm"):
        if path == f"acceleration_{suffix}":
            return f"speed_derivatives/speed_smoothed/acceleration_{suffix}"
        if path == f"smoothed_acceleration_{suffix}":
            return f"speed_derivatives/speed_smoothed/smoothed_acceleration_{suffix}"
    parts = path.split("/")
    if (
        len(parts) == 4
        and parts[:2] == ["movement", "speed"]
        and parts[2] in set(movement.values())
        and parts[3]
        in {
            "acceleration_px",
            "acceleration_mm",
            "smoothed_acceleration_px",
            "smoothed_acceleration_mm",
        }
    ):
        source = next(name for name, group in movement.items() if group == parts[2])
        return f"speed_derivatives/{source}/{parts[3]}"
    return None


def _dtype_for_path(path: str) -> np.dtype[Any]:
    if path in _STRUCTURED_DTYPES:
        return np.dtype(_STRUCTURED_DTYPES[path])
    if path in _BOOL_PATHS or path in _FLAT_BOOL_PATHS:
        return np.dtype(bool)
    if path in _INT64_PATHS or path in _FLAT_INT64_PATHS:
        return np.dtype("int64")
    if path in _FLAT_UINT64_PATHS:
        return np.dtype("uint64")
    if path in _INT32_PATHS:
        return np.dtype("int32")
    if path in _INT16_PATHS:
        return np.dtype("int16")
    if path in _INT8_PATHS:
        return np.dtype("int8")
    if path in _FLAT_FLOAT64_PATHS:
        return np.dtype("float64")
    if path in _FLOAT32_POSITION_PATHS:
        return np.dtype("float32")
    return np.dtype("float32")


def _units_for_path(path: str) -> str | None:
    leaf = path.rsplit("/", 1)[-1]
    if leaf in {"positions_px"}:
        return "px"
    if leaf in {"positions_mm"}:
        return "mm"
    if "acceleration" in leaf:
        return "mm/s^2" if leaf.endswith("_mm") else "px/s^2"
    if leaf.startswith("speed_") or leaf in {"px", "mm"}:
        return "mm/s" if leaf.endswith("_mm") or leaf == "mm" else "px/s"
    if "path_distance" in leaf:
        return "mm" if leaf.endswith("_mm") else "px"
    if leaf.endswith("_deg_s"):
        return "deg/s"
    if leaf.endswith("_degrees") or leaf.endswith("_deg"):
        return "deg"
    if leaf.endswith("_radians"):
        return "rad"
    if leaf in {"time_seconds", "delta_seconds"}:
        return "s"
    if "frame" in leaf or leaf == "second_indices":
        return "index"
    return None


def _fill_and_null(path: str, dtype: np.dtype[Any]) -> tuple[str, str]:
    if path == "source_instance_key/valid":
        return (
            "false",
            "false is the only null source-observation discriminator",
        )
    if path == "source_instance_key/value":
        return (
            "zero",
            "value must be zero whenever sibling valid is false",
        )
    if path.startswith("source_frame_interpolation/"):
        return (
            "zero_all_rows_written",
            "no null; every row carries one exact interpolation field",
        )
    if path == "source_instance_key":
        return (
            "structured_null_record_valid_false_instance_key_zero",
            "valid=false with instance_key=0 is the only null source observation",
        )
    if path == "source_frame_interpolation":
        return (
            "structured_zero_record_all_rows_written",
            "no null; every row carries exact left/right acquisition frames and weight",
        )
    if dtype == np.dtype(bool):
        return "false", "false is the conservative unavailable or invalid state"
    if path == "track_arena_ids":
        return "minus_one", "-1 means no arena assignment"
    if dtype.kind in "iu":
        return "zero", "no null; every identity, index, or code row is mandatory"
    if path in {"time_seconds", "heading_per_second_resultant"}:
        return "zero", "no null for time; zero resultant means no directional coherence"
    return "nan", "NaN marks unavailable or invalid floating-point motion"


def _shape_for_track_path(path: str) -> tuple[tuple[str | int, ...], tuple[str, ...]]:
    if path in _SECOND_PATHS:
        return ("n_track_seconds",), ("track_second",)
    if path in _PAIR_PATHS:
        return ("n_track_samples", 2), ("track_sample", "component")
    return ("n_track_samples",), ("track_sample",)


def _role_for_path(path: str) -> AnalysisAuthorityRole:
    if _alias_target(path) is not None:
        return AnalysisAuthorityRole.COMPATIBILITY_ALIAS
    if path in _LINEAGE_PATHS or path in TRACK_KINEMATICS_FLAT_LINEAGE_PATHS:
        return AnalysisAuthorityRole.LINEAGE_INDEX
    if path in _QUALITY_PATHS:
        return AnalysisAuthorityRole.QUALITY_DIAGNOSTIC
    return AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY


def _track_declaration(
    path: str,
    *,
    bundle: str,
    required: bool,
) -> TrackKinematicsExactArrayDeclaration:
    dtype = _dtype_for_path(path)
    shape, axes = _shape_for_track_path(path)
    fill, null = _fill_and_null(path, dtype)
    coordinate_space = None
    if path == "positions_px":
        coordinate_space = "source_camera_pixels"
    elif path == "positions_mm":
        coordinate_space = "source_camera_physical_mm"
    return TrackKinematicsExactArrayDeclaration(
        relative_path=path,
        scope="track",
        dtype=dtype,
        shape_template=shape,
        axis_names=axes,
        required=required,
        bundle=bundle,
        access_pattern=(
            AccessPattern.EAGER if path in _SECOND_PATHS else AccessPattern.WINDOWED
        ),
        authority_role=_role_for_path(path),
        fill_semantics=fill,
        null_semantics=null,
        units=_units_for_path(path),
        coordinate_space=coordinate_space,
    )


TRACK_KINEMATICS_CORE_TRACK_DECLARATIONS = tuple(
    _track_declaration(
        path,
        bundle=TRACK_KINEMATICS_CORE_BUNDLE,
        required=True,
    )
    for path in sorted(_CORE_TRACK_PATHS)
)

TRACK_KINEMATICS_PHYSICAL_TRACK_DECLARATIONS = tuple(
    _track_declaration(
        path,
        bundle=TRACK_KINEMATICS_PHYSICAL_BUNDLE,
        required=False,
    )
    for path in sorted(_PHYSICAL_TRACK_PATHS)
)

TRACK_KINEMATICS_FLAT_LINEAGE_TRACK_DECLARATIONS = tuple(
    _track_declaration(
        path,
        bundle=TRACK_KINEMATICS_FLAT_LINEAGE_BUNDLE,
        required=True,
    )
    for path in TRACK_KINEMATICS_FLAT_LINEAGE_PATHS
)


def _run_declaration(
    path: str,
    *,
    required: bool,
    bundle: str,
) -> TrackKinematicsExactArrayDeclaration:
    fill = "minus_one" if path == "track_arena_ids" else "zero"
    return TrackKinematicsExactArrayDeclaration(
        relative_path=path,
        scope="run",
        dtype=np.dtype("int32"),
        shape_template=("n_tracks",),
        axis_names=("track",),
        required=required,
        bundle=bundle,
        access_pattern=AccessPattern.EAGER,
        authority_role=AnalysisAuthorityRole.LINEAGE_INDEX,
        fill_semantics=fill,
        null_semantics=(
            "-1 means no arena assignment"
            if path == "track_arena_ids"
            else "no null; sorted unique track inventory"
        ),
        units="identifier",
    )


TRACK_KINEMATICS_RUN_DECLARATIONS = (
    _run_declaration(
        "track_ids",
        required=True,
        bundle=TRACK_KINEMATICS_CORE_BUNDLE,
    ),
    _run_declaration(
        "track_arena_ids",
        required=False,
        bundle=TRACK_KINEMATICS_ARENA_BUNDLE,
    ),
)


def build_track_kinematics_track_declarations(
    *,
    include_physical: bool,
) -> tuple[TrackKinematicsExactArrayDeclaration, ...]:
    """Return the closed all-or-none declaration inventory for one track."""

    return TRACK_KINEMATICS_CORE_TRACK_DECLARATIONS + (
        TRACK_KINEMATICS_PHYSICAL_TRACK_DECLARATIONS if include_physical else ()
    )


def build_track_kinematics_flat_lineage_declarations(
    *,
    track_ids: Iterable[int],
    include_physical: bool,
    include_arena_inventory: bool,
) -> tuple[AnalysisArrayDeclaration, ...]:
    """Return the closed v2 primitive inventory for one complete candidate."""

    normalized_ids: list[int] = []
    for raw_track_id in track_ids:
        if isinstance(raw_track_id, bool) or not isinstance(raw_track_id, int):
            raise TypeError("track_ids must contain exact integers.")
        normalized_ids.append(raw_track_id)
    if normalized_ids != sorted(set(normalized_ids)) or not normalized_ids:
        raise ValueError("track_ids must be nonempty, strictly increasing, and unique.")

    run_declarations = tuple(
        declaration.to_analysis_array_declaration(
            schema_version=TRACK_KINEMATICS_FLAT_LINEAGE_SCHEMA_VERSION,
            byte_planner_adopted=True,
            physical_policy_owner=TRACK_KINEMATICS_FLAT_PHYSICAL_POLICY_OWNER,
        )
        for declaration in TRACK_KINEMATICS_RUN_DECLARATIONS
        if declaration.relative_path != "track_arena_ids" or include_arena_inventory
    )
    primitive_core = (
        tuple(
            declaration
            for declaration in TRACK_KINEMATICS_CORE_TRACK_DECLARATIONS
            if declaration.relative_path not in _STRUCTURED_DTYPES
        )
        + TRACK_KINEMATICS_FLAT_LINEAGE_TRACK_DECLARATIONS
    )
    track_declarations = primitive_core + (
        TRACK_KINEMATICS_PHYSICAL_TRACK_DECLARATIONS if include_physical else ()
    )
    return run_declarations + tuple(
        declaration.bind_track_dimensions(track_id).to_analysis_array_declaration(
            schema_version=TRACK_KINEMATICS_FLAT_LINEAGE_SCHEMA_VERSION,
            byte_planner_adopted=True,
            physical_policy_owner=TRACK_KINEMATICS_FLAT_PHYSICAL_POLICY_OWNER,
        )
        for track_id in normalized_ids
        for declaration in track_declarations
    )


def declaration_paths(
    declarations: Iterable[TrackKinematicsExactArrayDeclaration],
) -> frozenset[str]:
    return frozenset(declaration.relative_path for declaration in declarations)


__all__ = [
    "TRACK_KINEMATICS_ARENA_BUNDLE",
    "TRACK_KINEMATICS_ARRAY_SCHEMA_ID",
    "TRACK_KINEMATICS_ARRAY_SCHEMA_VERSION",
    "TRACK_KINEMATICS_CORE_BUNDLE",
    "TRACK_KINEMATICS_CORE_TRACK_DECLARATIONS",
    "TRACK_KINEMATICS_FLAT_LINEAGE_BUNDLE",
    "TRACK_KINEMATICS_FLAT_LINEAGE_PATHS",
    "TRACK_KINEMATICS_FLAT_LINEAGE_SCHEMA_VERSION",
    "TRACK_KINEMATICS_FLAT_LINEAGE_TRACK_DECLARATIONS",
    "TRACK_KINEMATICS_FLAT_PHYSICAL_POLICY_OWNER",
    "TRACK_KINEMATICS_LEGACY_EXCLUDED_PREFIXES",
    "TRACK_KINEMATICS_LEGACY_EXCLUDED_RUN_ARRAYS",
    "TRACK_KINEMATICS_PHYSICAL_BUNDLE",
    "TRACK_KINEMATICS_PHYSICAL_POLICY_OWNER",
    "TRACK_KINEMATICS_PHYSICAL_TRACK_DECLARATIONS",
    "TRACK_KINEMATICS_RUN_DECLARATIONS",
    "TrackKinematicsExactArrayDeclaration",
    "TrackKinematicsStructuredDTypeBlockedError",
    "build_track_kinematics_flat_lineage_declarations",
    "build_track_kinematics_track_declarations",
    "declaration_paths",
]
