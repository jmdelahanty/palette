"""Deterministic display projections for sealed chaser spatial occupancy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

ROBUST_COLOR_QUANTILE = 0.98
ROBUST_COLOR_QUANTILE_METHOD = "linear"
COARSEN_POLICY_ID = "aligned_integer_factor_count_sum_preserve_denominators_v1"
COLOR_SCALE_POLICY_ID = "shared_positive_bin_quantile_with_full_range_reference_v1"


class ChaserSpatialOccupancyDisplayError(ValueError):
    """A display projection cannot preserve the sealed occupancy evidence."""


def _fail(message: str) -> None:
    raise ChaserSpatialOccupancyDisplayError(message)


def _readonly(values: Any, *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class SpatialOccupancyDisplaySurface:
    """One exact-grid or aligned-coarsened display surface."""

    label: str
    source_bin_width_mm: float
    display_bin_width_mm: float
    coarsen_factor: int
    counts: np.ndarray
    density_valid_in_arena: np.ndarray
    fraction_candidate_epoch: np.ndarray
    x_edges_mm: np.ndarray
    y_edges_mm: np.ndarray
    arena_bin_center_mask: np.ndarray

    def provenance_record(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "source_bin_width_mm": self.source_bin_width_mm,
            "display_bin_width_mm": self.display_bin_width_mm,
            "coarsen_factor": self.coarsen_factor,
            "coarsen_policy_id": (
                "identity_exact_persisted_grid"
                if self.coarsen_factor == 1
                else COARSEN_POLICY_ID
            ),
            "count_aggregation": (
                "none"
                if self.coarsen_factor == 1
                else f"exact_{self.coarsen_factor}x{self.coarsen_factor}_sum"
            ),
            "density_denominator": (
                "persisted_in_arena_position_frame_count_per_provider_epoch"
            ),
            "candidate_fraction_denominator": (
                "persisted_candidate_frame_count_per_provider_epoch"
            ),
            "grid_shape": [
                int(self.counts.shape[-2]),
                int(self.counts.shape[-1]),
            ],
            "x_bin_edges_mm": self.x_edges_mm.tolist(),
            "y_bin_edges_mm": self.y_edges_mm.tolist(),
            "interpolation": "prohibited",
            "scientific_authority": False,
        }


@dataclass(frozen=True, slots=True)
class SharedColorScale:
    """A robust default paired with an exact full-range reference."""

    quantile: float
    quantile_method: str
    robust_limit: float
    full_limit: float
    positive_bin_count: int
    bins_above_robust_limit: int
    empty_fallback_limit: float | None = None

    def provenance_record(self) -> dict[str, Any]:
        return {
            "policy_id": COLOR_SCALE_POLICY_ID,
            "quantile": self.quantile,
            "quantile_method": self.quantile_method,
            "robust_limit": self.robust_limit,
            "full_limit": self.full_limit,
            "positive_bin_count": self.positive_bin_count,
            "bins_above_robust_limit": self.bins_above_robust_limit,
            "empty_fallback_limit": self.empty_fallback_limit,
            "default": "robust",
            "full_range_reference_available": True,
            "values_above_robust_limit": (
                "color_saturated_only_exact_hover_values_retained"
            ),
        }


def _uniform_bin_width(edges: np.ndarray, *, label: str) -> float:
    values = np.asarray(edges, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
        _fail(f"{label} edges must be one finite vector.")
    differences = np.diff(values)
    if np.any(differences <= 0):
        _fail(f"{label} edges must be strictly increasing.")
    width = float(differences[0])
    tolerance = max(1e-12, abs(width) * 1e-10)
    if not np.allclose(differences, width, rtol=0.0, atol=tolerance):
        _fail(f"{label} bins must be uniformly spaced.")
    return width


def exact_spatial_occupancy_display_surface(
    *,
    counts: np.ndarray,
    density_valid_in_arena: np.ndarray,
    fraction_candidate_epoch: np.ndarray,
    x_edges_mm: np.ndarray,
    y_edges_mm: np.ndarray,
    arena_bin_center_mask: np.ndarray,
) -> SpatialOccupancyDisplaySurface:
    """Snapshot the persisted grid as the canonical display surface."""

    integer_counts = np.asarray(counts)
    density = np.asarray(density_valid_in_arena, dtype=np.float64)
    candidate = np.asarray(fraction_candidate_epoch, dtype=np.float64)
    x_edges = np.asarray(x_edges_mm, dtype=np.float64)
    y_edges = np.asarray(y_edges_mm, dtype=np.float64)
    mask = np.asarray(arena_bin_center_mask, dtype=bool)
    if integer_counts.ndim != 4 or integer_counts.dtype.kind not in "iu":
        _fail(
            "Persisted occupancy counts must be one integer provider/epoch/grid array."
        )
    if density.shape != integer_counts.shape or candidate.shape != integer_counts.shape:
        _fail("Persisted occupancy surfaces have inconsistent shapes.")
    grid_shape = integer_counts.shape[-2:]
    if x_edges.shape != (grid_shape[1] + 1,) or y_edges.shape != (grid_shape[0] + 1,):
        _fail("Persisted occupancy edges do not match the grid.")
    if mask.shape != grid_shape:
        _fail("Persisted arena center mask does not match the grid.")
    if (
        np.any(integer_counts < 0)
        or np.any(~np.isfinite(density))
        or np.any(~np.isfinite(candidate))
        or np.any(density < 0)
        or np.any(candidate < 0)
    ):
        _fail("Persisted occupancy surfaces contain invalid values.")
    x_width = _uniform_bin_width(x_edges, label="x")
    y_width = _uniform_bin_width(y_edges, label="y")
    if not math.isclose(x_width, y_width, rel_tol=0.0, abs_tol=1e-10):
        _fail("Persisted occupancy bins must be square.")
    return SpatialOccupancyDisplaySurface(
        label=f"canonical {x_width:g} mm",
        source_bin_width_mm=x_width,
        display_bin_width_mm=x_width,
        coarsen_factor=1,
        counts=_readonly(integer_counts),
        density_valid_in_arena=_readonly(density),
        fraction_candidate_epoch=_readonly(candidate),
        x_edges_mm=_readonly(x_edges),
        y_edges_mm=_readonly(y_edges),
        arena_bin_center_mask=_readonly(mask, dtype=bool),
    )


def aligned_coarsened_spatial_occupancy_display_surface(
    source: SpatialOccupancyDisplaySurface,
    *,
    factor: int,
    in_arena_denominator: np.ndarray,
    candidate_denominator: np.ndarray,
    arena_radius_mm: float,
) -> SpatialOccupancyDisplaySurface:
    """Sum whole aligned source bins and retain the sealed denominators."""

    if type(factor) is not int or factor <= 1:
        _fail("Coarsen factor must be one integer greater than one.")
    if source.coarsen_factor != 1:
        _fail("Aligned coarsening must start from the canonical persisted grid.")
    counts = np.asarray(source.counts)
    rows, columns = counts.shape[-2:]
    if rows % factor or columns % factor:
        _fail("Canonical grid cannot be partitioned into whole aligned coarse bins.")
    in_arena = np.asarray(in_arena_denominator)
    candidate = np.asarray(candidate_denominator)
    expected_denominator_shape = counts.shape[:2]
    if (
        in_arena.shape != expected_denominator_shape
        or candidate.shape != expected_denominator_shape
        or in_arena.dtype.kind not in "iu"
        or candidate.dtype.kind not in "iu"
        or np.any(in_arena < 0)
        or np.any(candidate < 0)
        or np.any(in_arena > candidate)
    ):
        _fail("Persisted occupancy denominators are invalid for coarsening.")
    if not math.isfinite(float(arena_radius_mm)) or float(arena_radius_mm) <= 0:
        _fail("Reviewed arena radius must be finite and positive.")

    reshaped = counts.reshape(
        (*counts.shape[:2], rows // factor, factor, columns // factor, factor)
    )
    coarse_counts = reshaped.sum(axis=(3, 5), dtype=np.int64)
    if not np.array_equal(
        coarse_counts.sum(axis=(-2, -1), dtype=np.int64),
        counts.sum(axis=(-2, -1), dtype=np.int64),
    ):
        _fail("Aligned occupancy coarsening does not conserve exact counts.")
    if not np.array_equal(coarse_counts.sum(axis=(-2, -1), dtype=np.int64), in_arena):
        _fail("Coarsened occupancy counts differ from persisted in-arena denominators.")

    density = np.zeros(coarse_counts.shape, dtype=np.float64)
    fraction = np.zeros(coarse_counts.shape, dtype=np.float64)
    np.divide(
        coarse_counts,
        in_arena[..., None, None],
        out=density,
        where=in_arena[..., None, None] > 0,
    )
    np.divide(
        coarse_counts,
        candidate[..., None, None],
        out=fraction,
        where=candidate[..., None, None] > 0,
    )
    x_edges = np.asarray(source.x_edges_mm)[::factor]
    y_edges = np.asarray(source.y_edges_mm)[::factor]
    if (
        x_edges.size != columns // factor + 1
        or y_edges.size != rows // factor + 1
        or x_edges[-1] != source.x_edges_mm[-1]
        or y_edges[-1] != source.y_edges_mm[-1]
    ):
        _fail("Aligned coarsening does not preserve the canonical grid extent.")
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    center_x, center_y = np.meshgrid(x_centers, y_centers)
    arena_mask = np.hypot(center_x, center_y) <= float(arena_radius_mm)
    display_width = source.source_bin_width_mm * factor
    return SpatialOccupancyDisplaySurface(
        label=f"aligned {display_width:g} mm display",
        source_bin_width_mm=source.source_bin_width_mm,
        display_bin_width_mm=display_width,
        coarsen_factor=factor,
        counts=_readonly(coarse_counts, dtype=np.int64),
        density_valid_in_arena=_readonly(density),
        fraction_candidate_epoch=_readonly(fraction),
        x_edges_mm=_readonly(x_edges),
        y_edges_mm=_readonly(y_edges),
        arena_bin_center_mask=_readonly(arena_mask, dtype=bool),
    )


def shared_robust_color_scale(
    values: np.ndarray,
    *,
    quantile: float = ROBUST_COLOR_QUANTILE,
    empty_fallback_limit: float | None = None,
) -> SharedColorScale:
    """Return a robust default and the exact full positive range."""

    array = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(array)) or np.any(array < 0):
        _fail("Color-scale values must be finite and nonnegative.")
    if not 0.0 < float(quantile) < 1.0:
        _fail("Robust color quantile must lie strictly between zero and one.")
    positive = array[array > 0]
    if not positive.size:
        if (
            empty_fallback_limit is None
            or not math.isfinite(float(empty_fallback_limit))
            or float(empty_fallback_limit) <= 0
        ):
            _fail("Color-scale values contain no positive bins.")
        return SharedColorScale(
            quantile=float(quantile),
            quantile_method=ROBUST_COLOR_QUANTILE_METHOD,
            robust_limit=float(empty_fallback_limit),
            full_limit=float(empty_fallback_limit),
            positive_bin_count=0,
            bins_above_robust_limit=0,
            empty_fallback_limit=float(empty_fallback_limit),
        )
    robust = float(
        np.quantile(positive, float(quantile), method=ROBUST_COLOR_QUANTILE_METHOD)
    )
    full = float(np.max(positive))
    if not math.isfinite(robust) or robust <= 0 or robust > full:
        _fail("Robust color limit is invalid.")
    return SharedColorScale(
        quantile=float(quantile),
        quantile_method=ROBUST_COLOR_QUANTILE_METHOD,
        robust_limit=robust,
        full_limit=full,
        positive_bin_count=int(positive.size),
        bins_above_robust_limit=int(np.count_nonzero(positive > robust)),
        empty_fallback_limit=None,
    )


__all__ = [
    "COARSEN_POLICY_ID",
    "COLOR_SCALE_POLICY_ID",
    "ROBUST_COLOR_QUANTILE",
    "ROBUST_COLOR_QUANTILE_METHOD",
    "ChaserSpatialOccupancyDisplayError",
    "SharedColorScale",
    "SpatialOccupancyDisplaySurface",
    "aligned_coarsened_spatial_occupancy_display_surface",
    "exact_spatial_occupancy_display_surface",
    "shared_robust_color_scale",
]
