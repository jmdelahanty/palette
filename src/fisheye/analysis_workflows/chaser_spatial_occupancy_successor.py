"""Paired-provider fish occupancy on exact protocol-semantic chaser epochs.

The successor stores two-dimensional arena occupancy before plotting it.  It
accepts explicit immutable keypoint and detection relative-frame runs, one
exact protocol-semantic selection, and the reviewed geometry already sealed by
their radial successors.  No selector, interpolation, or nominal-FPS timebase
is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis.provider_chaser_position_suite import PositionSuiteEpoch
from fisheye.analysis_workflows.chaser_relative_distance_view import (
    load_chaser_relative_distance_view,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection import (
    CHASER_WINDOW_ROLES,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


SCHEMA_ID = "palette.analysis.chaser_spatial_occupancy_successor"
SCHEMA_VERSION = 1
METHOD_ID = "paired_provider_exact_epoch_arena_occupancy_v1"
GRID_POLICY_ID = "reviewed_circle_centered_physical_square_grid_v1"
BIN_POLICY_ID = "half_open_except_final_closed_numpy_histogram2d_v1"
NORMALIZATION_POLICY_ID = "valid_in_arena_and_candidate_epoch_denominators_v1"
PROVIDER_ROLES = ("keypoint", "detection")


class ChaserSpatialOccupancySuccessorError(ValueError):
    """Raised when paired spatial occupancy cannot remain exact."""


def _fail(message: str) -> None:
    raise ChaserSpatialOccupancySuccessorError(message)


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field} must be one exact non-empty string.")
    return value


def _digest(value: object, *, field: str) -> str:
    result = _text(value, field=field)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        _fail(f"{field} must be one lowercase SHA-256 digest.")
    return result


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _array_declarations(arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    return [
        {
            "path": name,
            "dtype": np.asarray(value).dtype.str,
            "shape": list(np.asarray(value).shape),
            "content_sha256": array_values_sha256(np.asarray(value)),
        }
        for name, value in sorted(arrays.items())
    ]


@dataclass(frozen=True, slots=True)
class SpatialPositionProviderInput:
    provider_role: str
    relative_frame_run_path: str
    relative_frame_manifest_sha256: str
    radial_run_path: str
    radial_manifest_sha256: str
    fish_position_authority: Mapping[str, Any]
    fish_xy_px: np.ndarray
    fish_valid: np.ndarray
    relative_frame_verification_mode: str = "direct_prepared_input_no_receipt"
    relative_frame_validation_receipt_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ChaserSpatialOccupancyInput:
    recording_id: str
    semantic_selection_run_path: str
    semantic_selection_manifest_sha256: str
    acquisition_frame_id: np.ndarray
    selection_member: np.ndarray
    epochs: Sequence[PositionSuiteEpoch]
    providers: Sequence[SpatialPositionProviderInput]
    arena_center_x_px: float
    arena_center_y_px: float
    arena_radius_px: float
    mm_per_pixel: float
    arena_geometry_authority: Mapping[str, Any]
    arena_boundary_role: str
    arena_observed_feature: str
    bin_width_mm: float = 2.0


@dataclass(frozen=True, slots=True)
class PreparedChaserSpatialOccupancy:
    recording_id: str
    n_providers: int
    n_epochs: int
    grid_rows: int
    grid_columns: int
    arrays: Mapping[str, np.ndarray]
    manifest: Mapping[str, Any]

    def array(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise KeyError(f"Unknown spatial-occupancy array {name!r}.") from exc

    @property
    def payload_digest(self) -> str:
        return str(self.manifest["payload_digest"])


def _grid_edges(radius_mm: float, bin_width_mm: float) -> np.ndarray:
    if not math.isfinite(bin_width_mm) or bin_width_mm <= 0:
        _fail("bin_width_mm must be finite and positive.")
    extent = math.ceil(radius_mm / bin_width_mm) * bin_width_mm
    bin_count = int(round((2.0 * extent) / bin_width_mm))
    if bin_count <= 0 or bin_count > 4096:
        _fail("Spatial occupancy grid dimensions are unsafe.")
    return np.linspace(-extent, extent, bin_count + 1, dtype=np.float64)


def prepare_chaser_spatial_occupancy_successor(
    inputs: ChaserSpatialOccupancyInput,
) -> PreparedChaserSpatialOccupancy:
    """Persist paired-provider epoch occupancy on one shared physical grid."""

    if type(inputs) is not ChaserSpatialOccupancyInput:
        raise TypeError("inputs must be one ChaserSpatialOccupancyInput.")
    recording_id = _text(inputs.recording_id, field="recording_id")
    _text(inputs.semantic_selection_run_path, field="semantic selection run path")
    _digest(
        inputs.semantic_selection_manifest_sha256,
        field="semantic selection manifest digest",
    )
    geometry = _plain(inputs.arena_geometry_authority)
    _digest(geometry.get("selection_record_sha256"), field="geometry selection digest")
    _digest(
        geometry.get("physical_authority_sha256"),
        field="physical authority digest",
    )
    _text(geometry.get("pixel_frame_record_ref"), field="geometry pixel frame")

    center_x = float(inputs.arena_center_x_px)
    center_y = float(inputs.arena_center_y_px)
    radius_px = float(inputs.arena_radius_px)
    mm_per_pixel = float(inputs.mm_per_pixel)
    if not all(math.isfinite(value) for value in (center_x, center_y, radius_px, mm_per_pixel)):
        _fail("Arena center, radius, and scale must be finite.")
    if radius_px <= 0 or mm_per_pixel <= 0:
        _fail("Arena radius and mm_per_pixel must be positive.")
    radius_mm = radius_px * mm_per_pixel
    edges = _grid_edges(radius_mm, float(inputs.bin_width_mm))
    grid_rows = edges.size - 1
    grid_columns = edges.size - 1

    frame = np.asarray(inputs.acquisition_frame_id)
    selected = np.asarray(inputs.selection_member)
    if (
        frame.dtype != np.dtype(np.int64)
        or selected.dtype != np.dtype(bool)
        or frame.ndim != 1
        or selected.shape != frame.shape
        or frame.size == 0
        or np.any(np.diff(frame) <= 0)
    ):
        _fail("Frame and selection axes must be aligned int64/bool exact vectors.")

    epochs = tuple(inputs.epochs)
    if tuple(epoch.analysis_role for epoch in epochs) != CHASER_WINDOW_ROLES:
        _fail("Spatial occupancy requires exact chaser pre/training/post epoch order.")
    epoch_masks: list[np.ndarray] = []
    for previous, epoch in zip((None, *epochs[:-1]), epochs):
        if (
            type(epoch) is not PositionSuiteEpoch
            or epoch.start_frame < 0
            or epoch.end_frame <= epoch.start_frame
        ):
            _fail("Spatial occupancy epoch bounds are invalid.")
        if previous is not None and epoch.start_frame < previous.end_frame:
            _fail("Spatial occupancy epochs overlap.")
        epoch_masks.append(
            (frame >= int(epoch.start_frame)) & (frame < int(epoch.end_frame))
        )
    expected_selection = np.logical_or.reduce(epoch_masks)
    if np.any(expected_selection & ~selected):
        _fail("An exact semantic epoch row is absent from relative-frame selection.")
    selected_outside_epoch_count = int(
        np.count_nonzero(selected & ~expected_selection)
    )

    providers = tuple(inputs.providers)
    if tuple(provider.provider_role for provider in providers) != PROVIDER_ROLES:
        _fail("Spatial occupancy requires ordered keypoint and detection providers.")
    provider_ids: list[str] = []
    provider_digests: list[str] = []
    for provider in providers:
        _text(provider.relative_frame_run_path, field="relative-frame run path")
        _digest(
            provider.relative_frame_manifest_sha256,
            field="relative-frame manifest digest",
        )
        _text(provider.radial_run_path, field="radial run path")
        _digest(provider.radial_manifest_sha256, field="radial manifest digest")
        authority = _plain(provider.fish_position_authority)
        provider_ids.append(
            _text(authority.get("provider_id"), field="position provider ID")
        )
        provider_digests.append(
            _digest(
                authority.get("provider_digest"),
                field="position provider digest",
            )
        )
        if authority.get("coordinate_authority_id") != geometry.get(
            "pixel_frame_record_ref"
        ):
            _fail("Fish position and reviewed geometry use different pixel frames.")
    if len(set(provider_ids)) != len(provider_ids):
        _fail("Paired occupancy providers must have distinct provider identities.")

    shape = (len(providers), len(epochs), grid_rows, grid_columns)
    counts = np.zeros(shape, dtype=np.int64)
    density_valid = np.zeros(shape, dtype=np.float64)
    fraction_candidate = np.zeros(shape, dtype=np.float64)
    candidate_count = np.zeros((len(providers), len(epochs)), dtype=np.int64)
    declared_valid_count = np.zeros_like(candidate_count)
    finite_valid_count = np.zeros_like(candidate_count)
    in_arena_count = np.zeros_like(candidate_count)
    invalid_count = np.zeros_like(candidate_count)
    out_of_arena_count = np.zeros_like(candidate_count)
    coverage = np.zeros(candidate_count.shape, dtype=np.float64)
    in_arena_fraction_valid = np.zeros(candidate_count.shape, dtype=np.float64)

    for provider_index, provider in enumerate(providers):
        xy = np.asarray(provider.fish_xy_px)
        declared = np.asarray(provider.fish_valid)
        if (
            xy.shape != (frame.size, 2)
            or xy.dtype.kind != "f"
            or declared.shape != frame.shape
            or declared.dtype != np.dtype(bool)
        ):
            _fail("Provider fish position arrays do not match the exact frame axis.")
        finite = np.all(np.isfinite(xy), axis=1)
        valid = declared & finite
        x_mm = (xy[:, 0] - center_x) * mm_per_pixel
        y_mm = (xy[:, 1] - center_y) * mm_per_pixel
        inside = valid & (np.hypot(x_mm, y_mm) <= radius_mm)
        for epoch_index, epoch_mask in enumerate(epoch_masks):
            candidate = epoch_mask & selected
            candidate_total = int(np.count_nonzero(candidate))
            declared_total = int(np.count_nonzero(candidate & declared))
            finite_total = int(np.count_nonzero(candidate & valid))
            in_arena = candidate & inside
            in_arena_total = int(np.count_nonzero(in_arena))
            out_total = int(np.count_nonzero(candidate & valid & ~inside))
            invalid_total = int(np.count_nonzero(candidate & ~valid))
            if candidate_total == 0:
                _fail("An exact semantic epoch has no relative-frame rows.")
            histogram, _, _ = np.histogram2d(
                y_mm[in_arena],
                x_mm[in_arena],
                bins=(edges, edges),
            )
            integer_histogram = histogram.astype(np.int64, copy=False)
            if int(integer_histogram.sum()) != in_arena_total:
                _fail("Spatial histogram does not conserve in-arena position rows.")
            counts[provider_index, epoch_index] = integer_histogram
            if in_arena_total:
                density_valid[provider_index, epoch_index] = (
                    integer_histogram / float(in_arena_total)
                )
            fraction_candidate[provider_index, epoch_index] = (
                integer_histogram / float(candidate_total)
            )
            candidate_count[provider_index, epoch_index] = candidate_total
            declared_valid_count[provider_index, epoch_index] = declared_total
            finite_valid_count[provider_index, epoch_index] = finite_total
            in_arena_count[provider_index, epoch_index] = in_arena_total
            invalid_count[provider_index, epoch_index] = invalid_total
            out_of_arena_count[provider_index, epoch_index] = out_total
            coverage[provider_index, epoch_index] = in_arena_total / candidate_total
            in_arena_fraction_valid[provider_index, epoch_index] = (
                in_arena_total / finite_total if finite_total else 0.0
            )

    centers = (edges[:-1] + edges[1:]) / 2.0
    center_x_grid, center_y_grid = np.meshgrid(centers, centers)
    arena_mask = np.hypot(center_x_grid, center_y_grid) <= radius_mm
    epoch_role_registry = {
        str(index): role for index, role in enumerate(CHASER_WINDOW_ROLES)
    }
    provider_role_registry = {
        str(index): role for index, role in enumerate(PROVIDER_ROLES)
    }
    arrays = {
        "provider_role_code": _readonly(np.arange(len(providers)), dtype=np.uint8),
        "epoch_role_code": _readonly(np.arange(len(epochs)), dtype=np.uint8),
        "epoch_window_id": _readonly([epoch.window_id for epoch in epochs], dtype=np.int64),
        "epoch_start_frame": _readonly([epoch.start_frame for epoch in epochs], dtype=np.int64),
        "epoch_end_frame_exclusive": _readonly(
            [epoch.end_frame for epoch in epochs], dtype=np.int64
        ),
        "x_bin_edges_mm": _readonly(edges, dtype=np.float64),
        "y_bin_edges_mm": _readonly(edges, dtype=np.float64),
        "arena_bin_center_mask": _readonly(arena_mask, dtype=bool),
        "occupancy_count": _readonly(counts, dtype=np.int64),
        "occupancy_density_valid_in_arena": _readonly(
            density_valid, dtype=np.float64
        ),
        "occupancy_fraction_candidate_epoch": _readonly(
            fraction_candidate, dtype=np.float64
        ),
        "candidate_frame_count": _readonly(candidate_count, dtype=np.int64),
        "declared_valid_position_frame_count": _readonly(
            declared_valid_count, dtype=np.int64
        ),
        "finite_valid_position_frame_count": _readonly(
            finite_valid_count, dtype=np.int64
        ),
        "in_arena_position_frame_count": _readonly(in_arena_count, dtype=np.int64),
        "invalid_position_frame_count": _readonly(invalid_count, dtype=np.int64),
        "out_of_arena_position_frame_count": _readonly(
            out_of_arena_count, dtype=np.int64
        ),
        "in_arena_coverage_fraction_candidate": _readonly(
            coverage, dtype=np.float64
        ),
        "in_arena_fraction_finite_valid": _readonly(
            in_arena_fraction_valid, dtype=np.float64
        ),
    }
    readonly = MappingProxyType(arrays)
    provider_records = []
    for index, provider in enumerate(providers):
        authority = _plain(provider.fish_position_authority)
        provider_records.append(
            {
                "provider_role_code": index,
                "provider_role": provider.provider_role,
                "provider_id": provider_ids[index],
                "provider_digest": provider_digests[index],
                "fish_position_authority": authority,
                "relative_frame": {
                    "run_path": provider.relative_frame_run_path,
                    "manifest_sha256": provider.relative_frame_manifest_sha256,
                    "verification_mode": _text(
                        provider.relative_frame_verification_mode,
                        field="relative-frame verification mode",
                    ),
                    "validation_receipt_sha256": (
                        _digest(
                            provider.relative_frame_validation_receipt_sha256,
                            field="relative-frame validation receipt digest",
                        )
                        if provider.relative_frame_validation_receipt_sha256
                        is not None
                        else None
                    ),
                },
                "radial_near_field": {
                    "run_path": provider.radial_run_path,
                    "manifest_sha256": provider.radial_manifest_sha256,
                },
            }
        )
    body = {
        "scientific_schema": {"schema_id": SCHEMA_ID, "schema_version": SCHEMA_VERSION},
        "method_id": METHOD_ID,
        "recording_id": recording_id,
        "dimensions": {
            "n_providers": len(providers),
            "n_epochs": len(epochs),
            "grid_rows": grid_rows,
            "grid_columns": grid_columns,
        },
        "sources": {
            "protocol_semantic_selection": {
                "run_path": inputs.semantic_selection_run_path,
                "manifest_sha256": inputs.semantic_selection_manifest_sha256,
            },
            "arena_geometry_and_scale": geometry,
            "position_providers": provider_records,
        },
        "arena": {
            "center_x_px": center_x,
            "center_y_px": center_y,
            "radius_px": radius_px,
            "radius_mm": radius_mm,
            "mm_per_pixel": mm_per_pixel,
            "boundary_role": _text(
                inputs.arena_boundary_role, field="arena boundary role"
            ),
            "observed_feature": _text(
                inputs.arena_observed_feature, field="arena observed feature"
            ),
            "coordinate_space": "arena_centered_source_camera_mm_xy_top_left_y_down",
        },
        "grid": {
            "policy_id": GRID_POLICY_ID,
            "bin_policy_id": BIN_POLICY_ID,
            "normalization_policy_id": NORMALIZATION_POLICY_ID,
            "bin_width_mm": float(inputs.bin_width_mm),
            "x_min_mm": float(edges[0]),
            "x_max_mm": float(edges[-1]),
            "y_min_mm": float(edges[0]),
            "y_max_mm": float(edges[-1]),
            "outside_reviewed_circle": "excluded_before_histogram",
            "coordinate_orientation": "+x_right_+y_down",
        },
        "identity_registries": {
            "provider_role": provider_role_registry,
            "epoch_role": epoch_role_registry,
        },
        "epoch_records": [
            {
                "analysis_role": epoch.analysis_role,
                "window_id": epoch.window_id,
                "source_label": epoch.source_label,
                "start_frame": epoch.start_frame,
                "end_frame_exclusive": epoch.end_frame,
                "source_interval_sha256": epoch.source_interval_sha256,
            }
            for epoch in epochs
        ],
        "denominators": {
            "occupancy_density_valid_in_arena": (
                "in_arena_position_frame_count_per_provider_epoch"
            ),
            "occupancy_fraction_candidate_epoch": (
                "all_exact_selected_relative_frames_per_epoch"
            ),
            "missing_position_policy": "retained_in_coverage_not_spatially_imputed",
            "interpolation": "prohibited",
            "relative_selection_rows_outside_semantic_epochs": (
                selected_outside_epoch_count
            ),
            "relative_selection_rows_outside_semantic_epoch_policy": (
                "retained_as_source_evidence_and_excluded_from_epoch_heatmaps"
            ),
        },
        "array_declarations": _array_declarations(readonly),
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
        "registry_update": False,
    }
    manifest = _freeze({**body, "payload_digest": canonical_json_sha256(body)})
    return PreparedChaserSpatialOccupancy(
        recording_id=recording_id,
        n_providers=len(providers),
        n_epochs=len(epochs),
        grid_rows=grid_rows,
        grid_columns=grid_columns,
        arrays=readonly,
        manifest=manifest,
    )


def chaser_spatial_occupancy_input_from_handles(
    relative_keypoint: Any,
    relative_detection: Any,
    semantic_selection: Any,
    radial_keypoint: Any,
    radial_detection: Any,
    *,
    bin_width_mm: float = 2.0,
) -> ChaserSpatialOccupancyInput:
    """Bind strict current handles and prove paired-provider comparability."""

    from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
        ChaserRelativeFrameSourceHandle,
    )
    from fisheye.analysis_workflows.chaser_relative_frame_validation_receipt import (
        ChaserRelativeFrameTargetedSourceHandle,
    )
    from fisheye.analysis_workflows.composable_chaser_successor_publication import (
        ComposableChaserSuccessorSourceHandle,
    )
    from fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication import (
        ProtocolSemanticChaserSelectionSourceHandle,
    )

    relatives = (relative_keypoint, relative_detection)
    radials = (radial_keypoint, radial_detection)
    deep_handles = all(
        type(value) is ChaserRelativeFrameSourceHandle for value in relatives
    )
    targeted_handles = all(
        type(value) is ChaserRelativeFrameTargetedSourceHandle
        for value in relatives
    )
    if not (deep_handles or targeted_handles):
        raise TypeError(
            "relative providers must be a matched pair of strict deep-audit or "
            "receipt-bound targeted handles."
        )
    if type(semantic_selection) is not ProtocolSemanticChaserSelectionSourceHandle:
        raise TypeError("semantic_selection must be one strict semantic handle.")
    if any(type(value) is not ComposableChaserSuccessorSourceHandle for value in radials):
        raise TypeError("radial providers must be strict composable handles.")
    for handle in (*relatives, semantic_selection, *radials):
        handle.assert_current()
    if any(radial.successor_kind != "chaser_radial_near_field" for radial in radials):
        _fail("Spatial occupancy requires radial/near-field source handles.")
    recording_ids = {
        relative_keypoint.recording_id,
        relative_detection.recording_id,
        semantic_selection.recording_id,
        radial_keypoint.recording_id,
        radial_detection.recording_id,
    }
    archives = {
        relative_keypoint.analysis_zarr_path,
        relative_detection.analysis_zarr_path,
        semantic_selection.analysis_zarr,
        radial_keypoint.analysis_zarr,
        radial_detection.analysis_zarr,
    }
    if len(recording_ids) != 1 or len(archives) != 1:
        _fail("Spatial occupancy sources belong to different recordings or archives.")

    views = (
        tuple(load_chaser_relative_distance_view(value) for value in relatives)
        if deep_handles
        else relatives
    )
    for name in (
        "acquisition_frame_id",
        "timestamp_ns",
        "timestamp_valid",
        "selection_member",
    ):
        if not np.array_equal(views[0].frame_array(name), views[1].frame_array(name)):
            _fail(f"Paired providers differ on the exact shared {name!r} axis.")

    semantic_source = {
        "run_path": semantic_selection.run_path,
        "manifest_sha256": semantic_selection.manifest_sha256,
    }
    radial_scientific = tuple(radial.scientific_manifest for radial in radials)
    geometry = _plain(radial_scientific[0]["sources"]["arena_geometry_and_scale"])
    arena = _plain(radial_scientific[0]["arena"])
    epoch_records = _plain(radial_scientific[0]["epoch_records"])
    for index, (relative, radial, scientific) in enumerate(
        zip(relatives, radials, radial_scientific, strict=True)
    ):
        sources = scientific.get("sources")
        if not isinstance(sources, Mapping):
            _fail("Radial successor source bindings are absent.")
        if sources.get("relative_frame") != {
            "run_path": relative.run_path,
            "manifest_sha256": relative.manifest_sha256,
        }:
            _fail("Radial successor does not bind its exact relative-frame source.")
        if sources.get("protocol_semantic_selection") != semantic_source:
            _fail("Radial successor binds a different semantic selection.")
        if _plain(sources.get("arena_geometry_and_scale")) != geometry:
            _fail("Paired radial successors bind different reviewed geometry.")
        if _plain(scientific.get("arena")) != arena:
            _fail("Paired radial successors expose different arena geometry.")
        if _plain(scientific.get("epoch_records")) != epoch_records:
            _fail("Paired radial successors expose different semantic epochs.")
        position_provider = scientific.get("position_provider")
        fish_authority = relative.source_authorities["fish_position"]
        if (
            not isinstance(position_provider, Mapping)
            or position_provider.get("provider_id") != fish_authority.get("provider_id")
            or position_provider.get("provider_digest")
            != fish_authority.get("provider_digest")
        ):
            _fail(f"Radial provider {index} differs from relative-frame authority.")

    scales = tuple(_plain(relative.run_manifest.get("scale_policy")) for relative in relatives)
    if scales[0] != scales[1] or scales[0].get("unit") != "mm":
        _fail("Paired providers do not share one millimetre scale policy.")
    pixels_per_mm = float(scales[0].get("pixels_per_unit", math.nan))
    if not math.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
        _fail("Relative-frame millimetre scale is invalid.")
    mm_per_pixel = 1.0 / pixels_per_mm
    radius_px = float(arena["radius_px"])
    radius_mm = float(arena["radius_mm"])
    if not math.isclose(
        radius_px * mm_per_pixel, radius_mm, rel_tol=1e-6, abs_tol=1e-9
    ):
        _fail("Arena and relative-frame physical scales disagree.")

    providers = tuple(
        SpatialPositionProviderInput(
            provider_role=role,
            relative_frame_run_path=relative.run_path,
            relative_frame_manifest_sha256=relative.manifest_sha256,
            radial_run_path=radial.run_path,
            radial_manifest_sha256=radial.manifest_sha256,
            relative_frame_verification_mode=relative.verification_mode,
            relative_frame_validation_receipt_sha256=(
                relative.receipt_digest if targeted_handles else None
            ),
            fish_position_authority=relative.source_authorities["fish_position"],
            fish_xy_px=view.frame_array("fish_position_xy_px"),
            fish_valid=view.frame_array("fish_position_valid"),
        )
        for role, relative, radial, view in zip(
            PROVIDER_ROLES, relatives, radials, views, strict=True
        )
    )
    return ChaserSpatialOccupancyInput(
        recording_id=relative_keypoint.recording_id,
        semantic_selection_run_path=semantic_selection.run_path,
        semantic_selection_manifest_sha256=semantic_selection.manifest_sha256,
        acquisition_frame_id=views[0].frame_array("acquisition_frame_id"),
        selection_member=views[0].frame_array("selection_member"),
        epochs=semantic_selection.position_suite_epochs(),
        providers=providers,
        arena_center_x_px=float(arena["center_x_px"]),
        arena_center_y_px=float(arena["center_y_px"]),
        arena_radius_px=radius_px,
        mm_per_pixel=mm_per_pixel,
        arena_geometry_authority=geometry,
        arena_boundary_role=str(arena["boundary_role"]),
        arena_observed_feature=str(arena["observed_feature"]),
        bin_width_mm=bin_width_mm,
    )


def prepare_chaser_spatial_occupancy_successor_from_handles(
    relative_keypoint: Any,
    relative_detection: Any,
    semantic_selection: Any,
    radial_keypoint: Any,
    radial_detection: Any,
    *,
    bin_width_mm: float = 2.0,
) -> PreparedChaserSpatialOccupancy:
    return prepare_chaser_spatial_occupancy_successor(
        chaser_spatial_occupancy_input_from_handles(
            relative_keypoint,
            relative_detection,
            semantic_selection,
            radial_keypoint,
            radial_detection,
            bin_width_mm=bin_width_mm,
        )
    )


__all__ = [
    "BIN_POLICY_ID",
    "GRID_POLICY_ID",
    "METHOD_ID",
    "NORMALIZATION_POLICY_ID",
    "PROVIDER_ROLES",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "ChaserSpatialOccupancyInput",
    "ChaserSpatialOccupancySuccessorError",
    "PreparedChaserSpatialOccupancy",
    "SpatialPositionProviderInput",
    "chaser_spatial_occupancy_input_from_handles",
    "prepare_chaser_spatial_occupancy_successor",
    "prepare_chaser_spatial_occupancy_successor_from_handles",
]
