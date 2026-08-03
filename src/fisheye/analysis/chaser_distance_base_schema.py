"""Exact logical arrays in the sealed chaser-distance base authority.

The v1 writer contains additional protocol-role, count, visualization, and
compatibility surfaces.  They are deliberately absent here: the canonical
publication seal does not make them scientific authority.  This schema is the
closed projection jointly protected by the v1 publication seal, epoch-window
authority, and surface manifest.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fisheye.analysis._exact_tabular_run_schema import (
    BYTE_PLANNER_PHYSICAL_POLICY_OWNER,
    ColumnSpec,
)
from fisheye.shared.zarr.analysis_array_contracts import (
    AnalysisArrayDeclaration,
    AnalysisAuthorityRole,
)
from fisheye.shared.zarr.array_contracts import ArrayContract, DTypeContract
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode


CHASER_DISTANCE_BASE_CANDIDATE_SCHEMA_ID = (
    "palette.chaser_distance.sealed_base_storage_candidate.v2"
)
CHASER_DISTANCE_BASE_CANDIDATE_SCHEMA_VERSION = 2


def _spec(
    path: str,
    dtype: str,
    axes: tuple[str, ...],
    *,
    units: str | None = None,
    coordinate_space: str | None = None,
    authority: AnalysisAuthorityRole = AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY,
    access: AccessPattern = AccessPattern.EAGER,
    fill: str = "every persisted value is authoritative; no implicit fill",
    null: str = "no null sentinel",
) -> ColumnSpec:
    # coordinate_space is kept beside the frozen declaration below because the
    # compact ColumnSpec helper has no coordinate-space field.  The exact value
    # is emitted in ARRAY_SEMANTICS and the candidate manifest.
    del coordinate_space
    return ColumnSpec(
        path=path,
        dtype=np.dtype(dtype).str,
        axes=axes,
        units=units,
        authority=authority,
        access=access,
        fill=fill,
        null=null,
    )


_LINEAGE = AnalysisAuthorityRole.LINEAGE_INDEX
_SEMANTIC = AnalysisAuthorityRole.SEMANTIC_METADATA
_CACHE = AnalysisAuthorityRole.DERIVED_CACHE
_WINDOWED = AccessPattern.WINDOWED


def _specs() -> dict[str, ColumnSpec]:
    frame = ("camera_frame",)
    frame_chaser = ("camera_frame", "chaser")
    window_chaser = ("stimulus_epoch_window", "chaser")
    return {
        "stimulus_state_key": _spec(
            "stimulus_state_key", "int64", frame, authority=_LINEAGE, access=_WINDOWED,
            units="acquisition_frame_index",
        ),
        "frames/camera_frame_id": _spec(
            "frames/camera_frame_id", "int64", frame, authority=_LINEAGE,
            access=_WINDOWED, units="acquisition_frame_index",
        ),
        "frames/stimulus_frame_num": _spec(
            "frames/stimulus_frame_num", "int64", frame, authority=_LINEAGE,
            access=_WINDOWED, units="stimulus_frame_index",
            fill="-1 exactly when the stimulus frame is unavailable",
            null="-1 is the only unavailable sentinel",
        ),
        "frames/timestamp_ns": _spec(
            "frames/timestamp_ns", "int64", frame, authority=_LINEAGE,
            access=_WINDOWED, units="ns",
            fill="-1 exactly when the timestamp is unavailable",
            null="-1 is the only unavailable sentinel",
        ),
        "frames/stimulus_epoch_window_id": _spec(
            "frames/stimulus_epoch_window_id", "int32", frame, authority=_LINEAGE,
            access=_WINDOWED,
            fill="-1 outside a declared stimulus epoch window",
            null="-1 is the only no-window sentinel",
        ),
        "chasers/chaser_index": _spec(
            "chasers/chaser_index", "int16", ("chaser",), authority=_LINEAGE,
        ),
        "chasers/stimulus_instance_id_bytes": _spec(
            "chasers/stimulus_instance_id_bytes", "uint8", ("chaser", "utf8_byte"),
            authority=_SEMANTIC, fill="NUL-padded UTF-8 in exactly 96 bytes",
            null="empty identities are forbidden",
        ),
        "chasers/source_track_key_bytes": _spec(
            "chasers/source_track_key_bytes", "uint8", ("chaser", "utf8_byte"),
            authority=_LINEAGE, fill="NUL-padded UTF-8 in exactly 96 bytes",
            null="empty source-track keys are forbidden",
        ),
        "positions/source_detection_row_index": _spec(
            "positions/source_detection_row_index", "int64", frame,
            authority=_LINEAGE, access=_WINDOWED,
            fill="-1 when no source detection was selected",
            null="-1 is the only no-source-row sentinel",
        ),
        "positions/fish_centroid_img_xy": _spec(
            "positions/fish_centroid_img_xy", "float32", ("camera_frame", "xy"),
            units="px", access=_WINDOWED,
            fill="NaN pair when fish_valid is false",
            null="both coordinates are NaN exactly when unavailable",
        ),
        "positions/fish_centroid_arena_xy": _spec(
            "positions/fish_centroid_arena_xy", "float32", ("camera_frame", "xy"),
            units="px", access=_WINDOWED,
            fill="NaN pair when fish_valid is false",
            null="both coordinates are NaN exactly when unavailable",
        ),
        "positions/chaser_arena_xy": _spec(
            "positions/chaser_arena_xy", "float32", ("camera_frame", "chaser", "xy"),
            units="px", access=_WINDOWED,
            fill="NaN pair when the corresponding chaser_valid is false",
            null="both coordinates are NaN exactly when unavailable",
        ),
        "positions/fish_valid": _spec(
            "positions/fish_valid", "bool", frame, access=_WINDOWED,
            fill="false when no usable fish centroid exists",
        ),
        "positions/chaser_valid": _spec(
            "positions/chaser_valid", "bool", frame_chaser, access=_WINDOWED,
            fill="false when no usable chaser position exists",
        ),
        "distances/distance_px": _spec(
            "distances/distance_px", "float32", frame_chaser, units="px",
            access=_WINDOWED, fill="NaN unless fish and chaser positions are valid",
            null="NaN means unavailable; validity arrays remain authoritative",
        ),
        "distances/distance_mm": _spec(
            "distances/distance_mm", "float32", frame_chaser, units="mm",
            access=_WINDOWED, fill="NaN unless distance_px is finite",
            null="NaN means unavailable; validity arrays remain authoritative",
        ),
        "distances/nearest_chaser_index": _spec(
            "distances/nearest_chaser_index", "int16", frame, authority=_LINEAGE,
            access=_WINDOWED, fill="-1 when no finite chaser distance exists",
            null="-1 is the only unavailable sentinel",
        ),
        "distances/nearest_distance_mm": _spec(
            "distances/nearest_distance_mm", "float32", frame, units="mm",
            access=_WINDOWED, fill="NaN when no finite chaser distance exists",
            null="NaN is the unavailable sentinel",
        ),
        "epoch_summary/window_id": _spec(
            "epoch_summary/window_id", "int32", ("stimulus_epoch_window",),
            authority=_LINEAGE,
        ),
        "epoch_summary/label_bytes": _spec(
            "epoch_summary/label_bytes", "uint8", ("stimulus_epoch_window", "utf8_byte"),
            authority=_SEMANTIC, fill="NUL-padded UTF-8 in exactly 96 bytes",
            null="empty labels are forbidden",
        ),
        "epoch_summary/start_frame": _spec(
            "epoch_summary/start_frame", "int64", ("stimulus_epoch_window",),
            authority=_LINEAGE, units="acquisition_frame_index",
        ),
        "epoch_summary/end_frame": _spec(
            "epoch_summary/end_frame", "int64", ("stimulus_epoch_window",),
            authority=_LINEAGE, units="acquisition_frame_index",
        ),
        **{
            f"epoch_summary/{name}": _spec(
                f"epoch_summary/{name}", "float32", window_chaser, units="mm",
                fill="NaN when the epoch/chaser has no finite samples",
                null="NaN is the unavailable sentinel",
            )
            for name in (
                "mean_distance_mm", "min_distance_mm", "p05_distance_mm",
                "p50_distance_mm", "p95_distance_mm",
            )
        },
        "epoch_distributions/bin_edges_mm": _spec(
            "epoch_distributions/bin_edges_mm", "float32", ("distance_bin_edge",),
            units="mm", authority=_SEMANTIC,
        ),
        "epoch_distributions/bin_centers_mm": _spec(
            "epoch_distributions/bin_centers_mm", "float32", ("distance_bin",),
            units="mm", authority=_SEMANTIC,
        ),
        "epoch_distributions/hist_density": _spec(
            "epoch_distributions/hist_density", "float32",
            ("stimulus_epoch_window", "chaser", "distance_bin"), units="per_mm",
            authority=_CACHE, fill="zero for an empty epoch/chaser distribution",
        ),
    }


SEALED_CHASER_DISTANCE_BASE_PATHS = tuple(sorted(_specs()))

ARRAY_COORDINATE_SPACES: dict[str, str | None] = {
    path: None for path in SEALED_CHASER_DISTANCE_BASE_PATHS
}
ARRAY_COORDINATE_SPACES.update(
    {
        "positions/fish_centroid_img_xy": "source_camera_image_px",
        "positions/fish_centroid_arena_xy": "arena_relative_canvas_px",
        "positions/chaser_arena_xy": "arena_relative_canvas_px",
    }
)


def _array(group: Any, path: str) -> Any:
    node = group
    for component in path.split("/"):
        node = node[component]
    return node


def sealed_base_arrays(group: Any) -> dict[str, Any]:
    return {path: _array(group, path) for path in SEALED_CHASER_DISTANCE_BASE_PATHS}


def build_chaser_distance_base_declarations(
    group: Any,
) -> tuple[Any, ...]:
    arrays = sealed_base_arrays(group)
    dimensions: dict[str, str | int] = {
        "camera_frame": "n_camera_frames",
        "chaser": "n_chasers",
        "stimulus_epoch_window": "n_stimulus_epoch_windows",
        "distance_bin": "n_distance_bins",
        "distance_bin_edge": "n_distance_bin_edges",
        "xy": 2,
        "utf8_byte": 96,
    }
    declarations: list[AnalysisArrayDeclaration] = []
    for path, spec in sorted(_specs().items()):
        array = arrays[path]
        dtype = np.dtype(array.dtype)
        if dtype != np.dtype(spec.dtype):
            raise ValueError(
                f"{path}: dtype mismatch; expected {np.dtype(spec.dtype)}, got {dtype}."
            )
        if int(array.ndim) != len(spec.axes):
            raise ValueError(
                f"{path}: rank mismatch; expected {len(spec.axes)}, got {array.ndim}."
            )
        declarations.append(
            AnalysisArrayDeclaration(
                path=path,
                contract=ArrayContract(
                    schema_id=(
                        "palette.array.chaser_distance.sealed_base."
                        + path.replace("/", ".")
                    ),
                    schema_version=1,
                    dtype=DTypeContract(str(dtype), dtype.str),
                    shape_template=tuple(dimensions[axis] for axis in spec.axes),
                    axis_names=spec.axes,
                    description=f"Exact sealed chaser-distance base array {path}.",
                    units=spec.units,
                    coordinate_space=ARRAY_COORDINATE_SPACES[path],
                ),
                required=True,
                access_pattern=spec.access,
                write_mode=WriteMode.IMMUTABLE,
                authority_role=spec.authority,
                fill_semantics=spec.fill,
                null_semantics=spec.null,
                physical_policy_owner=BYTE_PLANNER_PHYSICAL_POLICY_OWNER,
                byte_planner_adopted=True,
            )
        )
    return tuple(declarations)


def validate_chaser_distance_base_semantics(group: Any) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        arrays = sealed_base_arrays(group)
        declarations = build_chaser_distance_base_declarations(group)
    except Exception as exc:
        return (str(exc),)
    dimensions: dict[str, int] = {}
    for declaration in declarations:
        errors.extend(
            f"{declaration.path}: {error}"
            for error in declaration.contract.validate_observation(
                arrays[declaration.path], dimensions=dimensions
            )
        )
    if errors:
        return tuple(errors)
    frames = int(arrays["stimulus_state_key"].shape[0])
    chasers = int(arrays["chasers/chaser_index"].shape[0])
    windows = int(arrays["epoch_summary/window_id"].shape[0])
    frame_paths = (
        "stimulus_state_key", "frames/camera_frame_id", "frames/stimulus_frame_num",
        "frames/timestamp_ns", "frames/stimulus_epoch_window_id",
        "positions/source_detection_row_index", "positions/fish_centroid_img_xy",
        "positions/fish_centroid_arena_xy", "positions/chaser_arena_xy",
        "positions/fish_valid", "positions/chaser_valid", "distances/distance_px",
        "distances/distance_mm", "distances/nearest_chaser_index",
        "distances/nearest_distance_mm",
    )
    if any(int(arrays[path].shape[0]) != frames for path in frame_paths):
        errors.append("camera-frame axes do not share one exact extent")
    if tuple(arrays["chasers/stimulus_instance_id_bytes"].shape) != (chasers, 96):
        errors.append("stimulus_instance_id_bytes must be exact uint8[n_chasers,96]")
    if tuple(arrays["chasers/source_track_key_bytes"].shape) != (chasers, 96):
        errors.append("source_track_key_bytes must be exact uint8[n_chasers,96]")
    for path in (
        "positions/chaser_arena_xy", "positions/chaser_valid",
        "distances/distance_px", "distances/distance_mm",
    ):
        if int(arrays[path].shape[1]) != chasers:
            errors.append(f"{path}: chaser axis differs from chaser_index")
    if tuple(arrays["epoch_summary/label_bytes"].shape) != (windows, 96):
        errors.append("epoch label_bytes must be exact uint8[n_windows,96]")
    for path in (
        "epoch_summary/start_frame", "epoch_summary/end_frame",
        "epoch_summary/mean_distance_mm", "epoch_summary/min_distance_mm",
        "epoch_summary/p05_distance_mm", "epoch_summary/p50_distance_mm",
        "epoch_summary/p95_distance_mm",
    ):
        if int(arrays[path].shape[0]) != windows:
            errors.append(f"{path}: epoch-window axis differs from window_id")
    for path in (
        "epoch_summary/mean_distance_mm", "epoch_summary/min_distance_mm",
        "epoch_summary/p05_distance_mm", "epoch_summary/p50_distance_mm",
        "epoch_summary/p95_distance_mm",
    ):
        if tuple(arrays[path].shape) != (windows, chasers):
            errors.append(f"{path}: expected exact [n_windows,n_chasers]")
    bins = int(arrays["epoch_distributions/bin_centers_mm"].shape[0])
    if tuple(arrays["epoch_distributions/bin_edges_mm"].shape) != (bins + 1,):
        errors.append("bin_edges_mm must have exactly one more row than bin_centers_mm")
    if tuple(arrays["epoch_distributions/hist_density"].shape) != (
        windows, chasers, bins
    ):
        errors.append("hist_density must be exact [n_windows,n_chasers,n_bins]")
    state = np.asarray(arrays["stimulus_state_key"][:])
    camera = np.asarray(arrays["frames/camera_frame_id"][:])
    expected = np.arange(frames, dtype=np.int64)
    if not np.array_equal(state, expected) or not np.array_equal(camera, expected):
        errors.append("stimulus_state_key and camera_frame_id must equal arange(n_frames)")
    indices = np.asarray(arrays["chasers/chaser_index"][:])
    if not indices.size or not np.array_equal(indices, np.sort(np.unique(indices))):
        errors.append("chaser_index must be nonempty, unique, and strictly increasing")
    return tuple(errors)


__all__ = [
    "ARRAY_COORDINATE_SPACES",
    "CHASER_DISTANCE_BASE_CANDIDATE_SCHEMA_ID",
    "CHASER_DISTANCE_BASE_CANDIDATE_SCHEMA_VERSION",
    "SEALED_CHASER_DISTANCE_BASE_PATHS",
    "build_chaser_distance_base_declarations",
    "sealed_base_arrays",
    "validate_chaser_distance_base_semantics",
]
