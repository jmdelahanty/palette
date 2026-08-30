"""Shared display projection for exact body-bearing by chaser distance views.

The helpers in this module are intentionally plotting-library agnostic.  Both
the read-only Marimo explorer and the receipt-sealed static publisher use this
one validity, binning, normalization, and display-sampling contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

DISPLAY_RECIPE_ID = "accepted_body_axis_bearing_distance_display_v1"
DISTANCE_BIN_WIDTH_MM = 5.0
BEARING_BIN_WIDTH_DEG = 30.0
DENSITY_COLOR_CMAX_QUANTILE = 0.98
INTERACTIVE_POINT_CLOUD_MAX_ROWS_PER_PANEL_CHASER = 4_000
STATIC_POINT_CLOUD_MAX_ROWS_PER_PANEL_CHASER = 20_000


@dataclass(frozen=True)
class BodyBearingDistanceHistogram:
    """One panel/chaser display histogram and its exact denominator."""

    distance_bin_edges_mm: np.ndarray
    bearing_bin_edges_deg: np.ndarray
    counts: np.ndarray
    probability: np.ndarray
    denominator: int


def body_bearing_distance_valid_mask(
    distance_mm: np.ndarray,
    bearing_deg: np.ndarray,
    distance_valid: np.ndarray,
    bearing_valid: np.ndarray,
    occurrence_member: np.ndarray,
    panel_member: np.ndarray,
) -> np.ndarray:
    """Return the exact intersection used by every joint display.

    A declared-valid non-finite or out-of-contract scientific value is an
    error.  It is not silently normalized into missing display evidence.
    """

    distance = np.asarray(distance_mm, dtype=np.float64)
    bearing = np.asarray(bearing_deg, dtype=np.float64)
    distance_ok = np.asarray(distance_valid, dtype=bool)
    bearing_ok = np.asarray(bearing_valid, dtype=bool)
    occurrence = np.asarray(occurrence_member, dtype=bool)
    panel = np.asarray(panel_member, dtype=bool)
    shapes = {
        distance.shape,
        bearing.shape,
        distance_ok.shape,
        bearing_ok.shape,
        occurrence.shape,
        panel.shape,
    }
    if len(shapes) != 1 or distance.ndim == 0:
        raise ValueError(
            "Body-bearing distance values, validity, occurrence, and panel "
            "membership must have one identical non-scalar shape."
        )
    if np.any(distance_ok & (~np.isfinite(distance) | (distance < 0.0))):
        raise ValueError(
            "Valid physical-distance values must be finite and nonnegative."
        )
    if np.any(
        bearing_ok & (~np.isfinite(bearing) | (bearing < -180.0) | (bearing > 180.0))
    ):
        raise ValueError(
            "Valid anatomical body-bearing values must be finite in [-180, 180]."
        )
    return panel & occurrence & distance_ok & bearing_ok


def distance_bin_edges_mm(
    distance_mm: np.ndarray,
    valid: np.ndarray,
    *,
    bin_width_mm: float = DISTANCE_BIN_WIDTH_MM,
) -> np.ndarray:
    """Return zero-anchored fixed-width edges covering every selected row."""

    width = float(bin_width_mm)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("Distance bin width must be finite and positive.")
    distance = np.asarray(distance_mm, dtype=np.float64)
    selected = np.asarray(valid, dtype=bool)
    if distance.shape != selected.shape or distance.ndim == 0:
        raise ValueError("Distance values and validity must have one identical shape.")
    values = distance[selected]
    if np.any(~np.isfinite(values) | (values < 0.0)):
        raise ValueError("Selected physical-distance values are invalid.")
    maximum = float(np.max(values)) if values.size else 0.0
    bin_count = max(1, int(np.ceil(maximum / width)))
    edges = np.linspace(0.0, float(bin_count) * width, bin_count + 1)
    edges.setflags(write=False)
    return edges


def bearing_bin_edges_deg(
    *, bin_width_deg: float = BEARING_BIN_WIDTH_DEG
) -> np.ndarray:
    """Return fixed whole-circle bearing edges from -180 through +180 degrees."""

    width = float(bin_width_deg)
    if (
        not np.isfinite(width)
        or width <= 0.0
        or not np.isclose(360.0 / width, round(360.0 / width))
    ):
        raise ValueError("Bearing bin width must divide 360 degrees exactly.")
    count = int(round(360.0 / width))
    edges = np.linspace(-180.0, 180.0, count + 1)
    edges.setflags(write=False)
    return edges


def body_bearing_distance_histogram(
    distance_mm: np.ndarray,
    bearing_deg: np.ndarray,
    valid: np.ndarray,
    *,
    distance_edges_mm: np.ndarray,
    bearing_edges_deg: np.ndarray | None = None,
) -> BodyBearingDistanceHistogram:
    """Bin exact valid rows and normalize within this panel and chaser."""

    distance = np.asarray(distance_mm, dtype=np.float64)
    bearing = np.asarray(bearing_deg, dtype=np.float64)
    selected = np.asarray(valid, dtype=bool)
    if len({distance.shape, bearing.shape, selected.shape}) != 1 or distance.ndim == 0:
        raise ValueError(
            "Distance, bearing, and validity must have one identical shape."
        )
    if np.any(
        selected
        & (
            ~np.isfinite(distance)
            | (distance < 0.0)
            | ~np.isfinite(bearing)
            | (bearing < -180.0)
            | (bearing > 180.0)
        )
    ):
        raise ValueError(
            "Selected body-bearing distance rows are outside the contract."
        )
    distance_edges = np.asarray(distance_edges_mm, dtype=np.float64)
    bearing_edges = np.asarray(
        bearing_bin_edges_deg() if bearing_edges_deg is None else bearing_edges_deg,
        dtype=np.float64,
    )
    if (
        distance_edges.ndim != 1
        or distance_edges.size < 2
        or distance_edges[0] != 0.0
        or np.any(~np.isfinite(distance_edges))
        or np.any(np.diff(distance_edges) <= 0.0)
    ):
        raise ValueError(
            "Distance histogram edges are not zero-anchored and increasing."
        )
    if (
        bearing_edges.ndim != 1
        or bearing_edges.size < 2
        or not np.isclose(bearing_edges[0], -180.0)
        or not np.isclose(bearing_edges[-1], 180.0)
        or np.any(~np.isfinite(bearing_edges))
        or np.any(np.diff(bearing_edges) <= 0.0)
    ):
        raise ValueError("Bearing histogram edges do not cover one increasing circle.")
    counts = np.histogram2d(
        distance[selected],
        bearing[selected],
        bins=(distance_edges, bearing_edges),
    )[0].astype(np.int64)
    denominator = int(np.count_nonzero(selected))
    if int(np.sum(counts)) != denominator:
        raise ValueError("Histogram edges do not cover every selected exact row.")
    probability = (
        counts.astype(np.float64) / float(denominator)
        if denominator
        else np.zeros(counts.shape, dtype=np.float64)
    )
    for values in (counts, probability):
        values.setflags(write=False)
    return BodyBearingDistanceHistogram(
        distance_bin_edges_mm=distance_edges,
        bearing_bin_edges_deg=bearing_edges,
        counts=counts,
        probability=probability,
        denominator=denominator,
    )


def positive_probability_color_max(
    histograms: Iterable[BodyBearingDistanceHistogram],
    *,
    quantile: float = DENSITY_COLOR_CMAX_QUANTILE,
) -> float:
    """Return one robust positive color maximum shared by all density panels."""

    q = float(quantile)
    if not np.isfinite(q) or not 0.0 < q <= 1.0:
        raise ValueError("Density color quantile must be in (0, 1].")
    positive = [
        histogram.probability[histogram.probability > 0.0]
        for histogram in histograms
        if np.any(histogram.probability > 0.0)
    ]
    if not positive:
        return 1.0
    return max(
        float(np.quantile(np.concatenate(positive), q)),
        float(np.finfo(np.float64).eps),
    )


def uniformly_sample_indices(indices: np.ndarray, *, maximum: int) -> np.ndarray:
    """Bound a display-only point cloud while retaining source order and endpoints."""

    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    limit = int(maximum)
    if limit <= 0:
        raise ValueError("Point-cloud display maximum must be positive.")
    if values.size <= limit:
        return values.copy()
    positions = np.linspace(0, values.size - 1, limit, dtype=np.int64)
    return values[positions]


__all__ = [
    "BEARING_BIN_WIDTH_DEG",
    "BodyBearingDistanceHistogram",
    "DENSITY_COLOR_CMAX_QUANTILE",
    "DISPLAY_RECIPE_ID",
    "DISTANCE_BIN_WIDTH_MM",
    "INTERACTIVE_POINT_CLOUD_MAX_ROWS_PER_PANEL_CHASER",
    "STATIC_POINT_CLOUD_MAX_ROWS_PER_PANEL_CHASER",
    "bearing_bin_edges_deg",
    "body_bearing_distance_histogram",
    "body_bearing_distance_valid_mask",
    "distance_bin_edges_mm",
    "positive_probability_color_max",
    "uniformly_sample_indices",
]
