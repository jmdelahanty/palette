"""Pure provider-neutral spatial occupancy calculations.

This module deliberately contains no Zarr, plotting, interpolation, or
provider-selection code.  Callers provide one row per scientific sample and
must bind those rows to an exact track/selection product before calling
``calculate_provider_occupancy_v2``.

The result uses ``(y_bin, x_bin)`` array order.  Bins are left-closed and
right-open, except that the final outer edge is inclusive on each axis.
Undefined occupancy fractions for a selection with no valid in-grid samples
are represented by NaN rather than by a misleading all-zero distribution.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Final

import numpy as np


PROVIDER_OCCUPANCY_V2_SCHEMA_ID: Final = "palette.provider_occupancy.v2"
PROVIDER_OCCUPANCY_V2_SCHEMA_VERSION: Final = 2
EDGE_POLICY_ID: Final = "left_closed_right_open_final_outer_edge_inclusive_v1"
TIMING_POLICY_ID: Final = "valid_in_grid_sample_count_divided_by_fps_v1"
EMPTY_FRACTION_POLICY_ID: Final = "nan_when_no_valid_in_grid_samples_v1"
BIN_ARRAY_ORDER_ID: Final = "y_bin_x_bin_v1"


def _readonly_copy(values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _validate_edges(values: Sequence[float] | np.ndarray, *, axis: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{axis}_edges must be one-dimensional.")
    if raw.shape[0] < 2:
        raise ValueError(f"{axis}_edges must contain at least two edges.")
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{axis}_edges must contain real numeric values.")
    edges = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(edges).all():
        raise ValueError(f"{axis}_edges must contain only finite values.")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError(f"{axis}_edges must be strictly increasing.")
    return _readonly_copy(edges, dtype=np.float64)


@dataclass(frozen=True)
class OccupancyGrid:
    """Fixed scientific grid in arena millimetres."""

    x_edges: Sequence[float] | np.ndarray
    y_edges: Sequence[float] | np.ndarray
    edge_policy_id: str = EDGE_POLICY_ID

    def __post_init__(self) -> None:
        if self.edge_policy_id != EDGE_POLICY_ID:
            raise ValueError(
                f"Unsupported edge policy {self.edge_policy_id!r}; "
                f"only {EDGE_POLICY_ID!r} is implemented."
            )
        object.__setattr__(
            self,
            "x_edges",
            _validate_edges(self.x_edges, axis="x"),
        )
        object.__setattr__(
            self,
            "y_edges",
            _validate_edges(self.y_edges, axis="y"),
        )

    @property
    def bin_shape(self) -> tuple[int, int]:
        """Return the result shape in ``(y_bin, x_bin)`` order."""

        return (self.y_edges.size - 1, self.x_edges.size - 1)


@dataclass(frozen=True)
class OccupancyTimingPolicy:
    """Explicit timing policy used to convert valid samples into seconds."""

    fps_hz: float
    timing_policy_id: str = TIMING_POLICY_ID

    def __post_init__(self) -> None:
        fps = float(self.fps_hz)
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("fps_hz must be finite and greater than zero.")
        if self.timing_policy_id != TIMING_POLICY_ID:
            raise ValueError(
                f"Unsupported timing policy {self.timing_policy_id!r}; "
                f"only {TIMING_POLICY_ID!r} is implemented."
            )
        object.__setattr__(self, "fps_hz", fps)


def _coerce_position(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values.")
    return _readonly_copy(raw, dtype=np.float64)


def _coerce_state(values: Sequence[bool] | np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if raw.dtype.kind != "b":
        raise TypeError(f"{name} must contain boolean values.")
    return _readonly_copy(raw, dtype=np.bool_)


def _canonical_occurrence_id(value: object) -> str:
    if value is None:
        raise ValueError("occurrence_ids cannot contain None.")
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        raise ValueError("occurrence_ids cannot contain non-finite numbers.")
    text = str(value)
    if not text:
        raise ValueError("occurrence_ids cannot contain empty identifiers.")
    return text


def _coerce_occurrence_ids(
    values: Sequence[object] | np.ndarray,
) -> tuple[tuple[str, ...], ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("occurrence_ids must contain one identity set per row.")
    try:
        rows = tuple(values)
    except TypeError as exc:
        raise ValueError(
            "occurrence_ids must contain one identity set per row."
        ) from exc
    result: list[tuple[str, ...]] = []
    for index, row in enumerate(rows):
        try:
            row_values = (
                (row,)
                if isinstance(row, (str, np.str_))
                else tuple(row)  # type: ignore[arg-type]
            )
        except TypeError as exc:
            raise ValueError(
                f"occurrence_ids[{index}] must be an identity sequence."
            ) from exc
        identities = tuple(_canonical_occurrence_id(value) for value in row_values)
        if len(set(identities)) != len(identities):
            raise ValueError(
                f"occurrence_ids[{index}] contains duplicate identities."
            )
        result.append(identities)
    return tuple(result)


@dataclass(frozen=True)
class ProviderOccupancySamples:
    """In-memory provider rows and their independent validity states.

    The arrays have one row per track sample in the caller's exact input
    domain.  ``selected`` identifies the resolved stimulus selection.  The
    other booleans remain separate so coverage diagnostics distinguish missing
    provider output, provider rejection, failed transforms, non-finite values,
    and finite positions outside the fixed grid.
    """

    x_mm: Sequence[float] | np.ndarray
    y_mm: Sequence[float] | np.ndarray
    selected: Sequence[bool] | np.ndarray
    provider_present: Sequence[bool] | np.ndarray
    provider_valid: Sequence[bool] | np.ndarray
    transform_valid: Sequence[bool] | np.ndarray
    occurrence_ids: Sequence[object] | np.ndarray
    expected_occurrence_ids: Sequence[object] | np.ndarray | None = None

    def __post_init__(self) -> None:
        x_mm = _coerce_position(self.x_mm, name="x_mm")
        y_mm = _coerce_position(self.y_mm, name="y_mm")
        selected = _coerce_state(self.selected, name="selected")
        provider_present = _coerce_state(
            self.provider_present,
            name="provider_present",
        )
        provider_valid = _coerce_state(self.provider_valid, name="provider_valid")
        transform_valid = _coerce_state(
            self.transform_valid,
            name="transform_valid",
        )
        occurrence_ids = _coerce_occurrence_ids(self.occurrence_ids)
        arrays = (
            y_mm,
            selected,
            provider_present,
            provider_valid,
            transform_valid,
        )
        if any(array.shape != x_mm.shape for array in arrays):
            raise ValueError(
                "x_mm, y_mm, state arrays, and occurrence_ids must have equal cardinality."
            )
        if len(occurrence_ids) != x_mm.size:
            raise ValueError(
                "x_mm, y_mm, state arrays, and occurrence_ids must have equal cardinality."
            )
        if np.any(provider_valid & ~provider_present):
            raise ValueError("provider_valid cannot be true when provider_present is false.")
        if np.any(transform_valid & ~provider_valid):
            raise ValueError("transform_valid cannot be true when provider_valid is false.")
        if self.expected_occurrence_ids is None:
            expected_occurrence_ids = tuple(
                occurrence_ids[index]
                for index in np.flatnonzero(selected).tolist()
            )
        else:
            expected_occurrence_ids = _coerce_occurrence_ids(
                self.expected_occurrence_ids
            )
        object.__setattr__(self, "x_mm", x_mm)
        object.__setattr__(self, "y_mm", y_mm)
        object.__setattr__(self, "selected", selected)
        object.__setattr__(self, "provider_present", provider_present)
        object.__setattr__(self, "provider_valid", provider_valid)
        object.__setattr__(self, "transform_valid", transform_valid)
        object.__setattr__(self, "occurrence_ids", occurrence_ids)
        object.__setattr__(self, "expected_occurrence_ids", expected_occurrence_ids)

    @property
    def row_count(self) -> int:
        return int(self.x_mm.size)


@dataclass(frozen=True)
class ProviderOccupancySummary:
    """Raw and normalized occupancy for one occurrence or the pooled input."""

    occurrence_id: str | None
    counts: np.ndarray
    occupancy_fraction: np.ndarray
    expected_selected_frames: int
    provider_present_count: int
    provider_valid_count: int
    transform_invalid_count: int
    nonfinite_count: int
    out_of_grid_count: int
    valid_in_grid_sample_count: int
    occupancy_time_s: float

    def __post_init__(self) -> None:
        counts = np.asarray(self.counts)
        fraction = np.asarray(self.occupancy_fraction)
        if counts.ndim != 2 or fraction.ndim != 2 or counts.shape != fraction.shape:
            raise ValueError("counts and occupancy_fraction must be equal-shaped 2-D arrays.")
        if counts.dtype.kind not in "iu" or np.any(counts < 0):
            raise ValueError("counts must be non-negative integer values.")
        if not np.isfinite(fraction[~np.isnan(fraction)]).all():
            raise ValueError("occupancy_fraction may contain only finite values or NaN.")
        if any(
            int(value) < 0
            for value in (
                self.expected_selected_frames,
                self.provider_present_count,
                self.provider_valid_count,
                self.transform_invalid_count,
                self.nonfinite_count,
                self.out_of_grid_count,
                self.valid_in_grid_sample_count,
            )
        ):
            raise ValueError("occupancy coverage counts cannot be negative.")
        if not (
            self.expected_selected_frames
            >= self.provider_present_count
            >= self.provider_valid_count
            >= self.valid_in_grid_sample_count
        ):
            raise ValueError(
                "occupancy coverage counts must satisfy expected >= present >= "
                "provider-valid >= valid-in-grid."
            )
        if not np.isfinite(float(self.occupancy_time_s)) or self.occupancy_time_s < 0.0:
            raise ValueError("occupancy_time_s must be finite and non-negative.")
        object.__setattr__(self, "counts", _readonly_copy(counts, dtype=np.int64))
        object.__setattr__(
            self,
            "occupancy_fraction",
            _readonly_copy(fraction, dtype=np.float64),
        )
        object.__setattr__(self, "expected_selected_frames", int(self.expected_selected_frames))
        object.__setattr__(self, "provider_present_count", int(self.provider_present_count))
        object.__setattr__(self, "provider_valid_count", int(self.provider_valid_count))
        object.__setattr__(self, "transform_invalid_count", int(self.transform_invalid_count))
        object.__setattr__(self, "nonfinite_count", int(self.nonfinite_count))
        object.__setattr__(self, "out_of_grid_count", int(self.out_of_grid_count))
        object.__setattr__(
            self,
            "valid_in_grid_sample_count",
            int(self.valid_in_grid_sample_count),
        )
        object.__setattr__(self, "occupancy_time_s", float(self.occupancy_time_s))

    @property
    def provider_missing_count(self) -> int:
        return self.expected_selected_frames - self.provider_present_count

    @property
    def provider_invalid_count(self) -> int:
        return self.provider_present_count - self.provider_valid_count

    def validate_conservation(self, *, fraction_tolerance: float = 1e-12) -> None:
        """Fail if raw counts or non-empty fractions are not conserved."""

        if int(self.counts.sum(dtype=np.int64)) != self.valid_in_grid_sample_count:
            raise ValueError("Occupancy count conservation failed.")
        if self.valid_in_grid_sample_count:
            if not np.isfinite(self.occupancy_fraction).all():
                raise ValueError("Non-empty occupancy fractions must be finite.")
            if not np.isclose(
                float(self.occupancy_fraction.sum()),
                1.0,
                rtol=0.0,
                atol=float(fraction_tolerance),
            ):
                raise ValueError("Occupancy fraction conservation failed.")
        elif not np.isnan(self.occupancy_fraction).all():
            raise ValueError(
                "Empty occupancy fractions must be NaN under "
                f"{EMPTY_FRACTION_POLICY_ID}."
            )
        classified_valid = (
            self.nonfinite_count
            + self.transform_invalid_count
            + self.out_of_grid_count
            + self.valid_in_grid_sample_count
        )
        if classified_valid != self.provider_valid_count:
            raise ValueError("Occupancy provider-valid coverage classification failed.")

    @property
    def occupancy_time_by_bin_s(self) -> np.ndarray:
        """Return per-bin time under the same explicit timing policy.

        ``occupancy_time_s`` is the total time represented by valid in-grid
        samples.  The per-bin view is derived directly from raw counts and is
        therefore never a display-normalized quantity.
        """

        if self.valid_in_grid_sample_count == 0:
            values = np.zeros_like(self.counts, dtype=np.float64)
        else:
            values = self.occupancy_fraction * self.occupancy_time_s
        return _readonly_copy(values, dtype=np.float64)


@dataclass(frozen=True)
class ProviderOccupancyV2Result:
    """Pure occupancy output for per-occurrence and pooled aggregation."""

    schema_id: str
    schema_version: int
    config_digest: str
    edge_policy_id: str
    timing_policy_id: str
    fps_hz: float
    x_edges: np.ndarray
    y_edges: np.ndarray
    per_occurrence: tuple[ProviderOccupancySummary, ...]
    pooled: ProviderOccupancySummary

    def __post_init__(self) -> None:
        if self.schema_id != PROVIDER_OCCUPANCY_V2_SCHEMA_ID:
            raise ValueError("Unexpected provider occupancy schema ID.")
        if int(self.schema_version) != PROVIDER_OCCUPANCY_V2_SCHEMA_VERSION:
            raise ValueError("Unexpected provider occupancy schema version.")
        if len(self.config_digest) != hashlib.sha256().digest_size * 2:
            raise ValueError("config_digest must be a SHA-256 hexadecimal digest.")
        if self.edge_policy_id != EDGE_POLICY_ID:
            raise ValueError("Unexpected provider occupancy edge policy.")
        if self.timing_policy_id != TIMING_POLICY_ID:
            raise ValueError("Unexpected provider occupancy timing policy.")
        fps = float(self.fps_hz)
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("fps_hz must be finite and greater than zero.")
        x_edges = _validate_edges(self.x_edges, axis="x")
        y_edges = _validate_edges(self.y_edges, axis="y")
        expected_shape = (y_edges.size - 1, x_edges.size - 1)
        if self.pooled.counts.shape != expected_shape:
            raise ValueError("Pooled occupancy arrays disagree with grid cardinality.")
        if any(summary.counts.shape != expected_shape for summary in self.per_occurrence):
            raise ValueError("Per-occurrence occupancy arrays disagree with grid cardinality.")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "fps_hz", fps)
        object.__setattr__(self, "x_edges", x_edges)
        object.__setattr__(self, "y_edges", y_edges)

    def validate_conservation(self, *, fraction_tolerance: float = 1e-12) -> None:
        for summary in (*self.per_occurrence, self.pooled):
            summary.validate_conservation(fraction_tolerance=fraction_tolerance)


def build_provider_occupancy_config_digest(
    grid: OccupancyGrid,
    timing: OccupancyTimingPolicy,
) -> str:
    """Return a deterministic digest for the numerical configuration."""

    def edge_bytes(edges: np.ndarray) -> str:
        return np.asarray(edges, dtype="<f8").tobytes().hex()

    payload = {
        "bin_array_order_id": BIN_ARRAY_ORDER_ID,
        "edge_policy_id": grid.edge_policy_id,
        "empty_fraction_policy_id": EMPTY_FRACTION_POLICY_ID,
        "schema_id": PROVIDER_OCCUPANCY_V2_SCHEMA_ID,
        "schema_version": PROVIDER_OCCUPANCY_V2_SCHEMA_VERSION,
        "timing_policy_id": timing.timing_policy_id,
        "fps_hz": timing.fps_hz,
        "x_edges_float64_le_hex": edge_bytes(grid.x_edges),
        "x_edge_count": int(grid.x_edges.size),
        "y_edges_float64_le_hex": edge_bytes(grid.y_edges),
        "y_edge_count": int(grid.y_edges.size),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _summarize(
    samples: ProviderOccupancySamples,
    grid: OccupancyGrid,
    timing: OccupancyTimingPolicy,
    row_mask: np.ndarray,
    *,
    occurrence_id: str | None,
    expected_selected_frames: int,
) -> ProviderOccupancySummary:
    selected = np.asarray(row_mask, dtype=np.bool_) & samples.selected
    present = selected & samples.provider_present
    provider_valid = present & samples.provider_valid
    finite = np.isfinite(samples.x_mm) & np.isfinite(samples.y_mm)
    nonfinite = provider_valid & ~finite
    transform_invalid = provider_valid & finite & ~samples.transform_valid
    transformed_finite = provider_valid & samples.transform_valid & finite
    inside = np.zeros(samples.row_count, dtype=np.bool_)
    inside[transformed_finite] = (
        (samples.x_mm[transformed_finite] >= grid.x_edges[0])
        & (samples.x_mm[transformed_finite] <= grid.x_edges[-1])
        & (samples.y_mm[transformed_finite] >= grid.y_edges[0])
        & (samples.y_mm[transformed_finite] <= grid.y_edges[-1])
    )
    in_grid = transformed_finite & inside
    out_of_grid = transformed_finite & ~inside

    counts = np.zeros(grid.bin_shape, dtype=np.int64)
    if np.any(in_grid):
        x_indices = np.searchsorted(grid.x_edges, samples.x_mm[in_grid], side="right") - 1
        y_indices = np.searchsorted(grid.y_edges, samples.y_mm[in_grid], side="right") - 1
        # The explicit inside mask excludes out-of-grid values.  Correct the
        # only intentional searchsorted overflow: a point exactly on the
        # final outer edge belongs to the final bin.
        x_indices = np.where(
            samples.x_mm[in_grid] == grid.x_edges[-1],
            grid.x_edges.size - 2,
            x_indices,
        )
        y_indices = np.where(
            samples.y_mm[in_grid] == grid.y_edges[-1],
            grid.y_edges.size - 2,
            y_indices,
        )
        np.add.at(counts, (y_indices, x_indices), 1)

    valid_count = int(in_grid.sum())
    if valid_count:
        fraction = counts.astype(np.float64) / float(valid_count)
    else:
        fraction = np.full(grid.bin_shape, np.nan, dtype=np.float64)
    summary = ProviderOccupancySummary(
        occurrence_id=occurrence_id,
        counts=counts,
        occupancy_fraction=fraction,
        expected_selected_frames=int(expected_selected_frames),
        provider_present_count=int(present.sum()),
        provider_valid_count=int(provider_valid.sum()),
        transform_invalid_count=int(transform_invalid.sum()),
        nonfinite_count=int(nonfinite.sum()),
        out_of_grid_count=int(out_of_grid.sum()),
        valid_in_grid_sample_count=valid_count,
        occupancy_time_s=float(valid_count / timing.fps_hz),
    )
    summary.validate_conservation()
    return summary


def calculate_provider_occupancy_v2(
    samples: ProviderOccupancySamples,
    grid: OccupancyGrid,
    timing: OccupancyTimingPolicy,
) -> ProviderOccupancyV2Result:
    """Calculate occupancy without selecting, transforming, or publishing data."""

    if not isinstance(samples, ProviderOccupancySamples):
        raise TypeError("samples must be a ProviderOccupancySamples instance.")
    if not isinstance(grid, OccupancyGrid):
        raise TypeError("grid must be an OccupancyGrid instance.")
    if not isinstance(timing, OccupancyTimingPolicy):
        raise TypeError("timing must be an OccupancyTimingPolicy instance.")

    row_mask = np.ones(samples.row_count, dtype=np.bool_)
    occurrence_order = tuple(
        dict.fromkeys(
            occurrence
            for memberships in samples.expected_occurrence_ids
            for occurrence in memberships
        )
    )
    per_occurrence = tuple(
        _summarize(
            samples,
            grid,
            timing,
            np.asarray(
                [occurrence in memberships for memberships in samples.occurrence_ids],
                dtype=np.bool_,
            ),
            occurrence_id=occurrence,
            expected_selected_frames=sum(
                occurrence in memberships
                for memberships in samples.expected_occurrence_ids
            ),
        )
        for occurrence in occurrence_order
    )
    pooled = _summarize(
        samples,
        grid,
        timing,
        row_mask,
        occurrence_id=None,
        expected_selected_frames=len(samples.expected_occurrence_ids),
    )
    result = ProviderOccupancyV2Result(
        schema_id=PROVIDER_OCCUPANCY_V2_SCHEMA_ID,
        schema_version=PROVIDER_OCCUPANCY_V2_SCHEMA_VERSION,
        config_digest=build_provider_occupancy_config_digest(grid, timing),
        edge_policy_id=grid.edge_policy_id,
        timing_policy_id=timing.timing_policy_id,
        fps_hz=timing.fps_hz,
        x_edges=grid.x_edges,
        y_edges=grid.y_edges,
        per_occurrence=per_occurrence,
        pooled=pooled,
    )
    result.validate_conservation()
    return result


# A descriptive alias for callers that use “compute” for pure numerical APIs.
compute_provider_occupancy_v2 = calculate_provider_occupancy_v2


def occupancy_samples_from_trajectory(trajectory: object) -> ProviderOccupancySamples:
    """Adapt one verified pure trajectory without losing missing-frame exposure.

    The import is intentionally local so the numerical occupancy kernel remains
    independently importable.  Expected exposure comes from the resolved
    selection's complete frame membership, while spatial rows come from the
    provider's exact track-sample domain.
    """

    from fisheye.analysis.provider_spatial_trajectory import ProviderSpatialTrajectory

    if type(trajectory) is not ProviderSpatialTrajectory:
        raise TypeError("trajectory must be one exact ProviderSpatialTrajectory.")
    return ProviderOccupancySamples(
        x_mm=trajectory.arena_position_xy[:, 0],
        y_mm=trajectory.arena_position_xy[:, 1],
        selected=trajectory.in_selection,
        provider_present=trajectory.provider_present,
        provider_valid=trajectory.provider_valid,
        transform_valid=trajectory.transform_valid,
        occurrence_ids=trajectory.selection_occurrence_id,
        expected_occurrence_ids=trajectory.selection.occurrence_ids,
    )


__all__ = [
    "BIN_ARRAY_ORDER_ID",
    "EDGE_POLICY_ID",
    "EMPTY_FRACTION_POLICY_ID",
    "OccupancyGrid",
    "OccupancyTimingPolicy",
    "PROVIDER_OCCUPANCY_V2_SCHEMA_ID",
    "PROVIDER_OCCUPANCY_V2_SCHEMA_VERSION",
    "ProviderOccupancySamples",
    "ProviderOccupancySummary",
    "ProviderOccupancyV2Result",
    "TIMING_POLICY_ID",
    "build_provider_occupancy_config_digest",
    "calculate_provider_occupancy_v2",
    "compute_provider_occupancy_v2",
    "occupancy_samples_from_trajectory",
]
