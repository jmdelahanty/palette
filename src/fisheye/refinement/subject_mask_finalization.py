"""Component-aware subject-mask finalization helpers.

This module intentionally has no Zarr write path. Its production API finalizes
one block of ROI-local component surfaces into fixed-shape numeric buffers.
The single-surface API remains as a compatibility wrapper. Callers decide where
and how to persist the result or decode human-readable reason text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..shared.mask_geometry import fill_holes_with_change as _fill_holes_with_change
from ..shared.mask_geometry import hole_stats as _hole_stats


QUALITY_CLEAN = 0
QUALITY_CLEANUP_APPLIED = 10
QUALITY_NEEDS_REVIEW = 50

REVIEW_PENDING = 0
REVIEW_NEEDS_REVIEW = 1

FINALIZATION_COMPUTE_KERNEL = "numeric_struct_of_arrays_spatial_reuse_v2"
FINALIZATION_METRIC_LAYOUT = "float32_n_by_metric_v1"
FINALIZATION_REASON_ENCODING = "uint32_bitflags_v1"
FINALIZATION_REVIEW_ENCODING = "uint8_review_code_v1"

FINALIZATION_METRIC_NAMES = (
    "added_area_px",
    "area_px_after",
    "area_px_before",
    "changed_area_fraction",
    "changed_area_px",
    "component_count_after",
    "component_count_before",
    "hole_area_fraction_after",
    "hole_area_fraction_before",
    "hole_count_after",
    "hole_count_before",
    "largest_component_fraction_after",
    "largest_component_fraction_before",
    "removed_area_fraction",
    "removed_area_px",
    "removed_component_count",
    "removed_high_prob_area_px",
    "removed_prob_mass",
    "removed_prob_mass_fraction",
)
FINALIZATION_METRIC_INDEX = {
    name: index for index, name in enumerate(FINALIZATION_METRIC_NAMES)
}

_FINALIZATION_REASON_TAG_BITS = (
    ("cleanup_closed_gaps", 1 << 0),
    ("cleanup_filled_holes", 1 << 1),
    ("cleanup_removed_small_islands", 1 << 2),
    ("cleanup_kept_largest_component", 1 << 3),
    ("needs_review_empty_mask", 1 << 4),
    ("needs_review_removed_high_prob_island", 1 << 5),
    ("needs_review_large_cleanup_delta", 1 << 6),
    ("needs_review_multiple_components", 1 << 7),
)
FINALIZATION_REASON_BITS = {
    tag: np.uint32(bit) for tag, bit in _FINALIZATION_REASON_TAG_BITS
}
_CLEANUP_REASON_MASK = np.uint32(sum(bit for tag, bit in _FINALIZATION_REASON_TAG_BITS if tag.startswith("cleanup_")))
_NEEDS_REVIEW_REASON_MASK = np.uint32(
    sum(bit for tag, bit in _FINALIZATION_REASON_TAG_BITS if tag.startswith("needs_review"))
)


@dataclass(frozen=True)
class ComponentFinalizationPolicy:
    """Configuration for one component finalization pass."""

    component_name: str
    threshold: float = 0.5
    low_threshold: Optional[float] = None
    high_threshold: Optional[float] = None
    closing_radius: int = 0
    fill_holes: bool = False
    min_component_area_px: int = 0
    keep_largest_component: bool = False
    max_component_count: Optional[int] = 1
    max_removed_high_prob_mass_fraction: float = 0.01
    max_changed_area_fraction: float = 0.20


@dataclass(frozen=True)
class ComponentFinalizationResult:
    """Finalized component mask plus machine-readable review routing data."""

    mask: np.ndarray
    source_mask: np.ndarray
    metrics: dict[str, float]
    reason_tags: tuple[str, ...]
    review_recommendation: str
    quality_code: int
    quality_score: float


@dataclass(frozen=True)
class ComponentFinalizationBatchResult:
    """Struct-of-arrays output for one component block.

    No field has object dtype. Reason text and review labels are decoded only at
    compatibility or persistence boundaries.
    """

    masks: np.ndarray
    source_masks: np.ndarray
    metrics: np.ndarray
    reason_flags: np.ndarray
    quality_code: np.ndarray
    quality_score: np.ndarray
    review_code: np.ndarray
    centroid_xy: np.ndarray
    bbox_xyxy: np.ndarray

    def metric(self, name: str) -> np.ndarray:
        try:
            column = FINALIZATION_METRIC_INDEX[str(name)]
        except KeyError as exc:
            raise KeyError(f"Unknown finalization metric {name!r}.") from exc
        return np.asarray(self.metrics[:, int(column)], dtype=np.float32)


@dataclass(frozen=True)
class _MaskComponentStats:
    labels: np.ndarray
    component_labels: np.ndarray
    areas: np.ndarray
    bounding_boxes_xywh: np.ndarray
    centroids_xy: np.ndarray
    total_area: int

    @property
    def component_count(self) -> int:
        return int(self.areas.shape[0])

    @property
    def largest_area(self) -> int:
        return int(self.areas.max()) if self.areas.size else 0

    @property
    def largest_component_fraction(self) -> float:
        return float(self.largest_area / self.total_area) if self.total_area > 0 else 0.0

    @property
    def combined_centroid_xy(self) -> np.ndarray:
        if self.total_area <= 0 or self.areas.size == 0:
            return np.zeros((2,), dtype=np.float32)
        weighted = (
            np.asarray(self.centroids_xy, dtype=np.float64)
            * np.asarray(self.areas, dtype=np.float64).reshape(-1, 1)
        ).sum(axis=0) / float(self.total_area)
        return np.asarray(weighted, dtype=np.float32)

    @property
    def combined_bbox_xyxy(self) -> np.ndarray:
        if self.total_area <= 0 or self.bounding_boxes_xywh.size == 0:
            return np.zeros((4,), dtype=np.float32)
        boxes = np.asarray(self.bounding_boxes_xywh, dtype=np.int32)
        x0 = int(np.min(boxes[:, 0]))
        y0 = int(np.min(boxes[:, 1]))
        x1 = int(np.max(boxes[:, 0] + boxes[:, 2] - 1))
        y1 = int(np.max(boxes[:, 1] + boxes[:, 3] - 1))
        return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def default_subject_body_policy() -> ComponentFinalizationPolicy:
    """Return conservative defaults for body-mask candidate finalization."""

    return ComponentFinalizationPolicy(
        component_name="subject_body",
        threshold=0.5,
        high_threshold=0.8,
        closing_radius=1,
        fill_holes=True,
        min_component_area_px=8,
        keep_largest_component=True,
        max_component_count=1,
        max_removed_high_prob_mass_fraction=0.01,
        max_changed_area_fraction=0.20,
    )


def default_swim_bladder_policy() -> ComponentFinalizationPolicy:
    """Return conservative defaults for swim-bladder candidate finalization."""

    return ComponentFinalizationPolicy(
        component_name="swim_bladder",
        threshold=0.5,
        high_threshold=0.8,
        closing_radius=0,
        fill_holes=True,
        min_component_area_px=4,
        keep_largest_component=True,
        max_component_count=1,
        max_removed_high_prob_mass_fraction=0.02,
        max_changed_area_fraction=0.25,
    )


def default_eyes_union_policy() -> ComponentFinalizationPolicy:
    """Return defaults that preserve up to two plausible eye components."""

    return ComponentFinalizationPolicy(
        component_name="eyes_union",
        threshold=0.5,
        high_threshold=0.8,
        closing_radius=0,
        fill_holes=False,
        min_component_area_px=4,
        keep_largest_component=False,
        max_component_count=2,
        max_removed_high_prob_mass_fraction=0.02,
        max_changed_area_fraction=0.25,
    )


def finalize_component_mask(
    component_name: str,
    surface: np.ndarray,
    *,
    policy: Optional[ComponentFinalizationPolicy] = None,
    surface_is_probability: bool = True,
) -> ComponentFinalizationResult:
    """Finalize one ROI-local surface through the numeric block kernel."""

    arr = np.asarray(surface)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D component surface, got shape {arr.shape}")
    batch = finalize_component_masks_batch(
        component_name,
        arr[np.newaxis, ...],
        policy=policy,
        surface_is_probability=surface_is_probability,
    )
    metrics = {
        name: float(batch.metrics[0, metric_index])
        for metric_index, name in enumerate(FINALIZATION_METRIC_NAMES)
    }
    reason_tags = reason_tags_from_flags(batch.reason_flags[0])
    return ComponentFinalizationResult(
        mask=np.asarray(batch.masks[0], dtype=np.uint8),
        source_mask=np.asarray(batch.source_masks[0], dtype=np.uint8),
        metrics=metrics,
        reason_tags=reason_tags,
        review_recommendation=review_recommendation_from_code(batch.review_code[0]),
        quality_code=int(batch.quality_code[0]),
        quality_score=float(batch.quality_score[0]),
    )


def finalize_component_masks_batch(
    component_name: str,
    surfaces: np.ndarray,
    *,
    policy: Optional[ComponentFinalizationPolicy] = None,
    surface_is_probability: bool = True,
    probabilities_are_normalized: bool = False,
) -> ComponentFinalizationBatchResult:
    """Finalize one ``(N,H,W)`` block into fixed-shape numeric buffers.

    The row loop is restricted to morphology, connected components, and hole
    measurement. Area/change/probability metrics and review routing are computed
    across the completed block.
    """

    resolved_policy = policy or _default_policy_for_component(component_name)
    if resolved_policy.component_name != component_name:
        raise ValueError(
            f"Policy component {resolved_policy.component_name!r} does not match {component_name!r}"
        )
    probabilities = _coerce_probability_surfaces(
        surfaces,
        surface_is_probability=surface_is_probability,
        probabilities_are_normalized=probabilities_are_normalized,
    )
    total_rows = int(probabilities.shape[0])
    masks = np.zeros(probabilities.shape, dtype=np.uint8)
    source_masks = np.zeros(probabilities.shape, dtype=np.uint8)
    metrics = np.zeros(
        (total_rows, len(FINALIZATION_METRIC_NAMES)),
        dtype=np.float32,
    )
    reason_flags = np.zeros((total_rows,), dtype=np.uint32)
    centroid_xy = np.zeros((total_rows, 2), dtype=np.float32)
    bbox_xyxy = np.zeros((total_rows, 4), dtype=np.float32)
    post_filter_component_count = np.zeros((total_rows,), dtype=np.int32)
    removed_probability_mass_fraction = np.zeros((total_rows,), dtype=np.float64)
    changed_area_fraction = np.zeros((total_rows,), dtype=np.float64)
    pixel_scratch = np.empty(probabilities.shape[1:], dtype=bool)
    closing_kernel = _closing_kernel(resolved_policy.closing_radius)

    uses_simple_threshold = resolved_policy.low_threshold is None or resolved_policy.high_threshold is None
    if uses_simple_threshold:
        np.greater_equal(
            probabilities,
            np.float32(resolved_policy.threshold),
            out=source_masks,
        )

    for row_idx in range(total_rows):
        if uses_simple_threshold:
            initial_mask = source_masks[row_idx].view(np.bool_)
            initial_stats = _component_stats(initial_mask)
        else:
            initial_mask, initial_stats = _threshold_surface_with_stats(
                probabilities[row_idx],
                resolved_policy,
            )
            source_masks[row_idx] = np.asarray(initial_mask, dtype=np.uint8)

        closed_mask = _binary_close(initial_mask, kernel=closing_kernel)
        closing_changed = not np.array_equal(initial_mask, closed_mask)
        if resolved_policy.fill_holes:
            filled_mask, fill_changed = _fill_holes_with_change(closed_mask)
        else:
            filled_mask, fill_changed = closed_mask, False
        min_area_mask, min_area_stats, min_area_changed = _remove_small_components_with_stats(
            filled_mask,
            resolved_policy.min_component_area_px,
        )
        selected_mask, final_stats = _select_components_with_stats(
            min_area_mask,
            resolved_policy,
            stats=min_area_stats,
        )
        selection_changed = final_stats.total_area != min_area_stats.total_area
        masks[row_idx] = np.asarray(selected_mask, dtype=np.uint8)
        centroid_xy[row_idx] = final_stats.combined_centroid_xy
        bbox_xyxy[row_idx] = final_stats.combined_bbox_xyxy
        post_filter_component_count[row_idx] = np.int32(min_area_stats.component_count)
        initial_area = int(initial_stats.total_area)
        final_area = int(final_stats.total_area)
        initial_prob_mass = (
            float(probabilities[row_idx][initial_mask].sum())
            if initial_area > 0
            else 0.0
        )
        np.logical_not(selected_mask, out=pixel_scratch)
        np.logical_and(initial_mask, pixel_scratch, out=pixel_scratch)
        removed_area = int(np.count_nonzero(pixel_scratch))
        removed_probabilities = probabilities[row_idx][pixel_scratch]
        removed_prob_mass = float(removed_probabilities.sum()) if removed_area > 0 else 0.0
        high_threshold = float(
            resolved_policy.high_threshold
            if resolved_policy.high_threshold is not None
            else resolved_policy.threshold
        )
        removed_high_prob_area = int(np.count_nonzero(removed_probabilities >= high_threshold))
        np.logical_not(initial_mask, out=pixel_scratch)
        np.logical_and(selected_mask, pixel_scratch, out=pixel_scratch)
        added_area = int(np.count_nonzero(pixel_scratch))
        changed_area = int(removed_area + added_area)
        removed_area_fraction = float(removed_area / max(1, initial_area))
        removed_prob_fraction = float(removed_prob_mass / max(1.0, initial_prob_mass))
        changed_fraction = float(changed_area / max(1, initial_area))
        removed_probability_mass_fraction[row_idx] = removed_prob_fraction
        changed_area_fraction[row_idx] = changed_fraction

        metric_row = metrics[row_idx]
        metric_row[FINALIZATION_METRIC_INDEX["area_px_before"]] = np.float32(initial_area)
        metric_row[FINALIZATION_METRIC_INDEX["area_px_after"]] = np.float32(final_area)
        metric_row[FINALIZATION_METRIC_INDEX["removed_component_count"]] = np.float32(
            max(0, initial_stats.component_count - final_stats.component_count)
        )
        metric_row[FINALIZATION_METRIC_INDEX["removed_area_px"]] = np.float32(removed_area)
        metric_row[FINALIZATION_METRIC_INDEX["removed_area_fraction"]] = np.float32(
            removed_area_fraction
        )
        metric_row[FINALIZATION_METRIC_INDEX["removed_prob_mass"]] = np.float32(
            removed_prob_mass
        )
        metric_row[FINALIZATION_METRIC_INDEX["removed_prob_mass_fraction"]] = np.float32(
            removed_prob_fraction
        )
        metric_row[FINALIZATION_METRIC_INDEX["removed_high_prob_area_px"]] = np.float32(
            removed_high_prob_area
        )
        metric_row[FINALIZATION_METRIC_INDEX["changed_area_px"]] = np.float32(changed_area)
        metric_row[FINALIZATION_METRIC_INDEX["changed_area_fraction"]] = np.float32(
            changed_fraction
        )
        metric_row[FINALIZATION_METRIC_INDEX["added_area_px"]] = np.float32(added_area)

        metrics[row_idx, FINALIZATION_METRIC_INDEX["component_count_before"]] = np.float32(
            initial_stats.component_count
        )
        metrics[row_idx, FINALIZATION_METRIC_INDEX["component_count_after"]] = np.float32(
            final_stats.component_count
        )
        metrics[row_idx, FINALIZATION_METRIC_INDEX["largest_component_fraction_before"]] = np.float32(
            initial_stats.largest_component_fraction
        )
        metrics[row_idx, FINALIZATION_METRIC_INDEX["largest_component_fraction_after"]] = np.float32(
            final_stats.largest_component_fraction
        )
        hole_count_before, hole_fraction_before, _hole_area_before = _hole_stats(initial_mask)
        metrics[row_idx, FINALIZATION_METRIC_INDEX["hole_count_before"]] = np.float32(hole_count_before)
        metrics[row_idx, FINALIZATION_METRIC_INDEX["hole_area_fraction_before"]] = np.float32(
            hole_fraction_before
        )
        if resolved_policy.fill_holes:
            hole_count_after, hole_fraction_after = 0, 0.0
        elif not (closing_changed or fill_changed or min_area_changed or selection_changed):
            hole_count_after, hole_fraction_after = hole_count_before, hole_fraction_before
        else:
            hole_count_after, hole_fraction_after, _hole_area_after = _hole_stats(selected_mask)
        metrics[row_idx, FINALIZATION_METRIC_INDEX["hole_count_after"]] = np.float32(hole_count_after)
        metrics[row_idx, FINALIZATION_METRIC_INDEX["hole_area_fraction_after"]] = np.float32(
            hole_fraction_after
        )

        if closing_changed:
            reason_flags[row_idx] |= FINALIZATION_REASON_BITS["cleanup_closed_gaps"]
        if fill_changed:
            reason_flags[row_idx] |= FINALIZATION_REASON_BITS["cleanup_filled_holes"]
        if min_area_changed:
            reason_flags[row_idx] |= FINALIZATION_REASON_BITS["cleanup_removed_small_islands"]
        if selection_changed:
            reason_flags[row_idx] |= FINALIZATION_REASON_BITS["cleanup_kept_largest_component"]

    _populate_vectorized_reason_flags(
        metrics=metrics,
        reason_flags=reason_flags,
        post_filter_component_count=post_filter_component_count,
        removed_probability_mass_fraction=removed_probability_mass_fraction,
        changed_area_fraction=changed_area_fraction,
        policy=resolved_policy,
    )
    quality_code, quality_score, review_code = _vectorized_review_routing(
        reason_flags,
        metrics,
        removed_probability_mass_fraction=removed_probability_mass_fraction,
        changed_area_fraction=changed_area_fraction,
    )
    return ComponentFinalizationBatchResult(
        masks=masks,
        source_masks=source_masks,
        metrics=metrics,
        reason_flags=reason_flags,
        quality_code=quality_code,
        quality_score=quality_score,
        review_code=review_code,
        centroid_xy=centroid_xy,
        bbox_xyxy=bbox_xyxy,
    )


def _default_policy_for_component(component_name: str) -> ComponentFinalizationPolicy:
    if component_name == "subject_body":
        return default_subject_body_policy()
    if component_name == "swim_bladder":
        return default_swim_bladder_policy()
    if component_name == "eyes_union":
        return default_eyes_union_policy()
    raise NotImplementedError(f"No default finalization policy for {component_name!r}")


def _coerce_probability_surfaces(
    surfaces: np.ndarray,
    *,
    surface_is_probability: bool,
    probabilities_are_normalized: bool,
) -> np.ndarray:
    arr = np.asarray(surfaces, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected component surfaces with shape (N,H,W), got {arr.shape}")
    if surface_is_probability:
        if probabilities_are_normalized:
            return arr
        return np.clip(arr, 0.0, 1.0)
    return (arr > 0).astype(np.float32)


def _threshold_surface_with_stats(
    probabilities: np.ndarray,
    policy: ComponentFinalizationPolicy,
) -> tuple[np.ndarray, _MaskComponentStats]:
    if policy.low_threshold is None or policy.high_threshold is None:
        mask = probabilities >= float(policy.threshold)
        return mask, _component_stats(mask)

    low_mask = probabilities >= float(policy.low_threshold)
    high_mask = probabilities >= float(policy.high_threshold)
    if not np.any(high_mask):
        empty = _empty_component_stats(tuple(high_mask.shape))
        return empty.labels > 0, empty

    low_stats = _component_stats(low_mask)
    if low_stats.component_count == 0:
        return low_stats.labels > 0, low_stats
    high_labels = np.unique(low_stats.labels[high_mask])
    keep_labels = high_labels[high_labels > 0].astype(np.int32, copy=False)
    selected_stats = _subset_component_stats(low_stats, keep_labels)
    return selected_stats.labels > 0, selected_stats


def _closing_kernel(radius: int) -> np.ndarray | None:
    if radius <= 0:
        return None
    kernel_size = int(radius) * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))


def _binary_close(mask: np.ndarray, *, kernel: np.ndarray | None) -> np.ndarray:
    if kernel is None or not np.any(mask):
        return mask.astype(bool, copy=True)
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed.astype(bool, copy=False)


def _component_stats(mask: np.ndarray) -> _MaskComponentStats:
    mask_bool = np.asarray(mask).astype(bool, copy=False)
    if not np.any(mask_bool):
        return _empty_component_stats(mask_bool.shape)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_bool.astype(np.uint8),
        connectivity=8,
    )
    component_labels = np.arange(1, int(num_labels), dtype=np.int32)
    areas = np.asarray(stats[1:, cv2.CC_STAT_AREA], dtype=np.int64)
    return _MaskComponentStats(
        labels=labels.astype(np.int32, copy=False),
        component_labels=component_labels,
        areas=areas,
        bounding_boxes_xywh=np.asarray(
            stats[1:, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]],
            dtype=np.int32,
        ),
        centroids_xy=np.asarray(centroids[1:], dtype=np.float64),
        total_area=int(areas.sum()),
    )


def _empty_component_stats(shape: tuple[int, ...]) -> _MaskComponentStats:
    return _MaskComponentStats(
        labels=np.zeros(tuple(int(dim) for dim in shape), dtype=np.int32),
        component_labels=np.zeros((0,), dtype=np.int32),
        areas=np.zeros((0,), dtype=np.int64),
        bounding_boxes_xywh=np.zeros((0, 4), dtype=np.int32),
        centroids_xy=np.zeros((0, 2), dtype=np.float64),
        total_area=0,
    )


def _subset_component_stats(stats: _MaskComponentStats, keep_labels: np.ndarray) -> _MaskComponentStats:
    labels_to_keep = np.asarray(keep_labels, dtype=np.int32).reshape(-1)
    if labels_to_keep.size == 0 or stats.component_count == 0:
        return _empty_component_stats(tuple(stats.labels.shape))
    max_label = int(max(int(stats.component_labels.max()), int(labels_to_keep.max())))
    keep = np.zeros((max_label + 1,), dtype=bool)
    keep[labels_to_keep] = True
    labels = np.where(keep[stats.labels], stats.labels, 0).astype(np.int32, copy=False)

    area_by_label = np.zeros((max_label + 1,), dtype=np.int64)
    area_by_label[np.asarray(stats.component_labels, dtype=np.int32)] = np.asarray(stats.areas, dtype=np.int64)
    areas = area_by_label[labels_to_keep]
    present = areas > 0
    component_labels = labels_to_keep[present].astype(np.int32, copy=False)
    component_areas = areas[present].astype(np.int64, copy=False)
    position_by_label = {
        int(label): int(position)
        for position, label in enumerate(np.asarray(stats.component_labels, dtype=np.int32))
    }
    component_positions = np.asarray(
        [position_by_label[int(label)] for label in component_labels],
        dtype=np.int64,
    )
    return _MaskComponentStats(
        labels=labels,
        component_labels=component_labels,
        areas=component_areas,
        bounding_boxes_xywh=np.asarray(
            stats.bounding_boxes_xywh[component_positions],
            dtype=np.int32,
        ),
        centroids_xy=np.asarray(stats.centroids_xy[component_positions], dtype=np.float64),
        total_area=int(component_areas.sum()),
    )


def _component_areas(mask: np.ndarray) -> List[int]:
    return [int(area) for area in _component_stats(mask).areas]


def _remove_small_components_with_stats(
    mask: np.ndarray,
    min_area_px: int,
) -> tuple[np.ndarray, _MaskComponentStats, bool]:
    stats = _component_stats(mask)
    if min_area_px <= 1 or stats.component_count == 0:
        return stats.labels > 0, stats, False
    keep_labels = stats.component_labels[stats.areas >= int(min_area_px)]
    filtered = _subset_component_stats(stats, keep_labels)
    return filtered.labels > 0, filtered, filtered.total_area != stats.total_area


def _remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    cleaned, _stats, _changed = _remove_small_components_with_stats(mask, min_area_px)
    return cleaned


def _select_components_with_stats(
    mask: np.ndarray,
    policy: ComponentFinalizationPolicy,
    *,
    stats: _MaskComponentStats | None = None,
) -> tuple[np.ndarray, _MaskComponentStats]:
    resolved_stats = stats or _component_stats(mask)
    if policy.keep_largest_component:
        if resolved_stats.component_count <= 1:
            return resolved_stats.labels > 0, resolved_stats
        max_area = int(resolved_stats.areas.max())
        keep_label = int(resolved_stats.component_labels[resolved_stats.areas == max_area].max())
        selected = _subset_component_stats(resolved_stats, np.asarray([keep_label], dtype=np.int32))
        return selected.labels > 0, selected
    if policy.max_component_count is not None:
        count = int(policy.max_component_count)
        if count <= 0 or resolved_stats.component_count == 0:
            empty = _empty_component_stats(tuple(resolved_stats.labels.shape))
            return empty.labels > 0, empty
        if resolved_stats.component_count <= count:
            return resolved_stats.labels > 0, resolved_stats
        areas = [
            (int(area), int(label))
            for area, label in zip(resolved_stats.areas, resolved_stats.component_labels)
        ]
        keep_labels = [
            label
            for _area, label in sorted(areas, reverse=True)[:count]
        ]
        selected = _subset_component_stats(resolved_stats, np.asarray(keep_labels, dtype=np.int32))
        return selected.labels > 0, selected
    return mask.astype(bool, copy=True), resolved_stats


def _select_components(
    mask: np.ndarray,
    policy: ComponentFinalizationPolicy,
    *,
    stats: _MaskComponentStats | None = None,
) -> np.ndarray:
    selected, _selected_stats = _select_components_with_stats(mask, policy, stats=stats)
    return selected


def _largest_component_fraction(mask: np.ndarray, *, stats: _MaskComponentStats | None = None) -> float:
    return (stats or _component_stats(mask)).largest_component_fraction


def _populate_vectorized_reason_flags(
    *,
    metrics: np.ndarray,
    reason_flags: np.ndarray,
    post_filter_component_count: np.ndarray,
    removed_probability_mass_fraction: np.ndarray,
    changed_area_fraction: np.ndarray,
    policy: ComponentFinalizationPolicy,
) -> None:
    """Apply review predicates across the numeric row summaries."""

    area_after = metrics[:, FINALIZATION_METRIC_INDEX["area_px_after"]]
    component_count_after = metrics[:, FINALIZATION_METRIC_INDEX["component_count_after"]]
    removed_high_prob_area = metrics[:, FINALIZATION_METRIC_INDEX["removed_high_prob_area_px"]]

    reason_flags[area_after <= 0] |= FINALIZATION_REASON_BITS["needs_review_empty_mask"]
    reason_flags[
        (removed_high_prob_area > 0)
        & (
            removed_probability_mass_fraction
            > float(policy.max_removed_high_prob_mass_fraction)
        )
    ] |= FINALIZATION_REASON_BITS["needs_review_removed_high_prob_island"]
    reason_flags[changed_area_fraction > float(policy.max_changed_area_fraction)] |= (
        FINALIZATION_REASON_BITS["needs_review_large_cleanup_delta"]
    )
    if policy.max_component_count is not None:
        max_component_count = int(policy.max_component_count)
        multiple = post_filter_component_count > max_component_count
        multiple |= (component_count_after > float(max_component_count)) & (area_after > 0)
        reason_flags[multiple] |= FINALIZATION_REASON_BITS["needs_review_multiple_components"]


def _vectorized_review_routing(
    reason_flags: np.ndarray,
    metrics: np.ndarray,
    *,
    removed_probability_mass_fraction: np.ndarray,
    changed_area_fraction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flags = np.asarray(reason_flags, dtype=np.uint32).reshape(-1)
    needs_review = (flags & _NEEDS_REVIEW_REASON_MASK) != 0
    cleanup_applied = (flags & _CLEANUP_REASON_MASK) != 0
    severity = (
        np.asarray(removed_probability_mass_fraction, dtype=np.float64) * 100.0
        + np.asarray(changed_area_fraction, dtype=np.float64) * 50.0
        + metrics[:, FINALIZATION_METRIC_INDEX["removed_high_prob_area_px"]].astype(np.float64)
    )
    quality_code = np.full(flags.shape, QUALITY_CLEAN, dtype=np.int16)
    quality_score = severity.astype(np.float32)
    review_code = np.full(flags.shape, REVIEW_PENDING, dtype=np.uint8)
    cleanup_only = cleanup_applied & ~needs_review
    quality_code[cleanup_only] = np.int16(QUALITY_CLEANUP_APPLIED)
    quality_score[cleanup_only] = (severity[cleanup_only] + 10.0).astype(np.float32)
    quality_code[needs_review] = np.int16(QUALITY_NEEDS_REVIEW)
    quality_score[needs_review] = (severity[needs_review] + 100.0).astype(np.float32)
    review_code[needs_review] = np.uint8(REVIEW_NEEDS_REVIEW)
    return quality_code, quality_score, review_code


def reason_tags_from_flags(value: object) -> tuple[str, ...]:
    flags = int(np.uint32(value))
    tags = tuple(tag for tag, bit in _FINALIZATION_REASON_TAG_BITS if flags & int(bit))
    return tags or ("clean",)


def decode_reason_flags(
    values: np.ndarray,
    *,
    probability_source: bool = False,
) -> np.ndarray:
    """Decode numeric reason flags at a persistence or compatibility boundary."""

    flags = np.asarray(values, dtype=np.uint32).reshape(-1)
    labels = np.empty(flags.shape, dtype=object)
    for row_idx, value in enumerate(flags):
        tags = list(reason_tags_from_flags(value))
        if probability_source:
            tags = [tag for tag in tags if tag != "clean"]
            tags.insert(0, "cleanup_thresholded_probability")
        labels[row_idx] = "|".join(tags) if tags else "clean"
    return labels


def review_recommendation_from_code(value: object) -> str:
    code = int(np.uint8(value))
    if code == REVIEW_PENDING:
        return "pending"
    if code == REVIEW_NEEDS_REVIEW:
        return "needs_review"
    raise ValueError(f"Unknown finalization review code {code}.")
