"""Component-aware subject-mask finalization helpers.

This module intentionally has no Zarr write path. It turns one ROI-local
component probability/mask surface into a finalized binary candidate plus QC
metrics and review-routing reasons. Callers decide where and how to persist the
result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes


QUALITY_CLEAN = 0
QUALITY_CLEANUP_APPLIED = 10
QUALITY_NEEDS_REVIEW = 50


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
    metrics: Dict[str, float]
    reason_tags: tuple[str, ...]
    review_recommendation: str
    quality_code: int
    quality_score: float


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
    """Finalize one ROI-local component surface.

    The policy is component-specific. Body and swim-bladder finalization keep one
    dominant component; eyes-union finalization can preserve two components so
    left/right assignment can happen downstream.
    """

    resolved_policy = policy or _default_policy_for_component(component_name)
    if resolved_policy.component_name != component_name:
        raise ValueError(
            f"Policy component {resolved_policy.component_name!r} does not match {component_name!r}"
        )
    probabilities = _coerce_probability_surface(surface, surface_is_probability=surface_is_probability)
    initial_mask = _threshold_surface(probabilities, resolved_policy)
    closed_mask = _binary_close(initial_mask, resolved_policy.closing_radius)
    filled_mask = _fill_holes(closed_mask) if resolved_policy.fill_holes else closed_mask
    min_area_mask = _remove_small_components(filled_mask, resolved_policy.min_component_area_px)
    selected_mask = _select_components(min_area_mask, resolved_policy)
    final_mask = selected_mask

    metrics = _build_metrics(
        initial_mask=initial_mask,
        final_mask=final_mask,
        probabilities=probabilities,
        policy=resolved_policy,
    )
    reason_tags = _build_reason_tags(
        initial_mask=initial_mask,
        closed_mask=closed_mask,
        filled_mask=filled_mask,
        min_area_mask=min_area_mask,
        selected_mask=selected_mask,
        final_mask=final_mask,
        metrics=metrics,
        policy=resolved_policy,
    )
    quality_code, quality_score, review_recommendation = _review_routing(reason_tags, metrics)
    return ComponentFinalizationResult(
        mask=final_mask.astype(np.uint8, copy=False),
        source_mask=initial_mask.astype(np.uint8, copy=False),
        metrics=metrics,
        reason_tags=tuple(reason_tags),
        review_recommendation=review_recommendation,
        quality_code=int(quality_code),
        quality_score=float(quality_score),
    )


def _default_policy_for_component(component_name: str) -> ComponentFinalizationPolicy:
    if component_name == "subject_body":
        return default_subject_body_policy()
    if component_name == "swim_bladder":
        return default_swim_bladder_policy()
    if component_name == "eyes_union":
        return default_eyes_union_policy()
    raise NotImplementedError(f"No default finalization policy for {component_name!r}")


def _coerce_probability_surface(surface: np.ndarray, *, surface_is_probability: bool) -> np.ndarray:
    arr = np.asarray(surface, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D component surface, got shape {arr.shape}")
    if surface_is_probability:
        return np.clip(arr, 0.0, 1.0)
    return (arr > 0).astype(np.float32)


def _threshold_surface(probabilities: np.ndarray, policy: ComponentFinalizationPolicy) -> np.ndarray:
    if policy.low_threshold is None or policy.high_threshold is None:
        return probabilities >= float(policy.threshold)

    low_mask = probabilities >= float(policy.low_threshold)
    high_mask = probabilities >= float(policy.high_threshold)
    if not np.any(high_mask):
        return high_mask

    labeled, count = _connected_component_labels(low_mask)
    selected = np.zeros_like(low_mask, dtype=bool)
    for label_idx in range(1, count + 1):
        component = labeled == label_idx
        if np.any(component & high_mask):
            selected |= component
    return selected


def _binary_close(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not np.any(mask):
        return mask.astype(bool, copy=True)
    kernel_size = int(radius) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed.astype(bool, copy=False)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool, copy=False)
    if not np.any(mask_bool):
        return mask_bool.copy()
    return np.asarray(binary_fill_holes(mask_bool), dtype=bool)


def _connected_component_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return labels.astype(np.int32, copy=False), int(num_labels - 1)


def _component_areas(mask: np.ndarray) -> List[int]:
    labeled, count = _connected_component_labels(mask)
    return [int(np.count_nonzero(labeled == label_idx)) for label_idx in range(1, count + 1)]


def _remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    mask_bool = mask.astype(bool, copy=False)
    if min_area_px <= 1 or not np.any(mask_bool):
        return mask_bool.copy()
    labeled, count = _connected_component_labels(mask_bool)
    cleaned = np.zeros_like(mask_bool, dtype=bool)
    for label_idx in range(1, count + 1):
        component = labeled == label_idx
        if int(np.count_nonzero(component)) >= int(min_area_px):
            cleaned |= component
    return cleaned


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool, copy=False)
    if not np.any(mask_bool):
        return mask_bool.copy()
    labeled, count = _connected_component_labels(mask_bool)
    if count <= 1:
        return mask_bool.copy()
    areas = [(int(np.count_nonzero(labeled == label_idx)), label_idx) for label_idx in range(1, count + 1)]
    _area, keep_label = max(areas)
    return labeled == keep_label


def _keep_largest_components(mask: np.ndarray, count: int) -> np.ndarray:
    mask_bool = mask.astype(bool, copy=False)
    if count <= 0 or not np.any(mask_bool):
        return np.zeros_like(mask_bool, dtype=bool)
    labeled, component_count = _connected_component_labels(mask_bool)
    if component_count <= int(count):
        return mask_bool.copy()
    areas = [
        (int(np.count_nonzero(labeled == label_idx)), label_idx)
        for label_idx in range(1, component_count + 1)
    ]
    keep_labels = {label for _area, label in sorted(areas, reverse=True)[: int(count)]}
    return np.isin(labeled, list(keep_labels))


def _select_components(mask: np.ndarray, policy: ComponentFinalizationPolicy) -> np.ndarray:
    if policy.keep_largest_component:
        return _keep_largest_component(mask)
    if policy.max_component_count is not None:
        return _keep_largest_components(mask, int(policy.max_component_count))
    return mask.astype(bool, copy=True)


def _largest_component_fraction(mask: np.ndarray) -> float:
    total = int(np.count_nonzero(mask))
    if total <= 0:
        return 0.0
    areas = _component_areas(mask)
    return float(max(areas) / total) if areas else 0.0


def _hole_stats(mask: np.ndarray) -> tuple[int, float, int]:
    mask_bool = mask.astype(bool, copy=False)
    if not np.any(mask_bool):
        return 0, 0.0, 0
    filled = _fill_holes(mask_bool)
    holes = filled & ~mask_bool
    hole_area = int(np.count_nonzero(holes))
    if hole_area == 0:
        return 0, 0.0, 0
    _labeled, count = _connected_component_labels(holes)
    denom = max(1, int(np.count_nonzero(filled)))
    return int(count), float(hole_area / denom), hole_area


def _build_metrics(
    *,
    initial_mask: np.ndarray,
    final_mask: np.ndarray,
    probabilities: np.ndarray,
    policy: ComponentFinalizationPolicy,
) -> Dict[str, float]:
    initial_area = int(np.count_nonzero(initial_mask))
    final_area = int(np.count_nonzero(final_mask))
    removed = initial_mask & ~final_mask
    added = final_mask & ~initial_mask
    changed = removed | added
    removed_area = int(np.count_nonzero(removed))
    changed_area = int(np.count_nonzero(changed))
    initial_prob_mass = float(probabilities[initial_mask].sum()) if initial_area else 0.0
    removed_prob_mass = float(probabilities[removed].sum()) if removed_area else 0.0
    high_threshold = float(policy.high_threshold if policy.high_threshold is not None else policy.threshold)
    removed_high_prob_area = int(np.count_nonzero(removed & (probabilities >= high_threshold)))
    hole_count_before, hole_fraction_before, _hole_area_before = _hole_stats(initial_mask)
    hole_count_after, hole_fraction_after, _hole_area_after = _hole_stats(final_mask)
    component_count_before = int(len(_component_areas(initial_mask)))
    component_count_after = int(len(_component_areas(final_mask)))
    return {
        "area_px_before": float(initial_area),
        "area_px_after": float(final_area),
        "component_count_before": float(component_count_before),
        "component_count_after": float(component_count_after),
        "largest_component_fraction_before": float(_largest_component_fraction(initial_mask)),
        "largest_component_fraction_after": float(_largest_component_fraction(final_mask)),
        "removed_component_count": float(max(0, component_count_before - component_count_after)),
        "removed_area_px": float(removed_area),
        "removed_area_fraction": float(removed_area / max(1, initial_area)),
        "removed_prob_mass": float(removed_prob_mass),
        "removed_prob_mass_fraction": float(removed_prob_mass / max(1.0, initial_prob_mass)),
        "removed_high_prob_area_px": float(removed_high_prob_area),
        "changed_area_px": float(changed_area),
        "changed_area_fraction": float(changed_area / max(1, initial_area)),
        "added_area_px": float(int(np.count_nonzero(added))),
        "hole_count_before": float(hole_count_before),
        "hole_count_after": float(hole_count_after),
        "hole_area_fraction_before": float(hole_fraction_before),
        "hole_area_fraction_after": float(hole_fraction_after),
    }


def _build_reason_tags(
    *,
    initial_mask: np.ndarray,
    closed_mask: np.ndarray,
    filled_mask: np.ndarray,
    min_area_mask: np.ndarray,
    selected_mask: np.ndarray,
    final_mask: np.ndarray,
    metrics: Dict[str, float],
    policy: ComponentFinalizationPolicy,
) -> List[str]:
    tags: List[str] = []
    if not np.array_equal(initial_mask, closed_mask):
        tags.append("cleanup_closed_gaps")
    if not np.array_equal(closed_mask, filled_mask):
        tags.append("cleanup_filled_holes")
    if not np.array_equal(filled_mask, min_area_mask):
        tags.append("cleanup_removed_small_islands")
    if not np.array_equal(min_area_mask, selected_mask):
        tags.append("cleanup_kept_largest_component")
    if metrics["area_px_after"] <= 0:
        tags.append("needs_review_empty_mask")
    if metrics["removed_high_prob_area_px"] > 0 and (
        metrics["removed_prob_mass_fraction"] > float(policy.max_removed_high_prob_mass_fraction)
    ):
        tags.append("needs_review_removed_high_prob_island")
    if metrics["changed_area_fraction"] > float(policy.max_changed_area_fraction):
        tags.append("needs_review_large_cleanup_delta")
    max_component_count = policy.max_component_count
    post_filter_component_count = float(len(_component_areas(min_area_mask)))
    if max_component_count is not None and post_filter_component_count > float(max_component_count):
        tags.append("needs_review_multiple_components")
    if (
        max_component_count is not None
        and metrics["component_count_after"] > float(max_component_count)
        and metrics["area_px_after"] > 0
    ):
        tags.append("needs_review_multiple_components")
    if not tags:
        tags.append("clean")
    return _dedupe(tags)


def _dedupe(tags: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        result.append(str(tag))
    return result


def _review_routing(
    reason_tags: Sequence[str],
    metrics: Dict[str, float],
) -> tuple[int, float, str]:
    needs_review = any(str(tag).startswith("needs_review") for tag in reason_tags)
    cleanup_applied = any(str(tag).startswith("cleanup_") for tag in reason_tags)
    severity = 0.0
    severity += float(metrics.get("removed_prob_mass_fraction", 0.0)) * 100.0
    severity += float(metrics.get("changed_area_fraction", 0.0)) * 50.0
    severity += float(metrics.get("removed_high_prob_area_px", 0.0))
    if needs_review:
        return QUALITY_NEEDS_REVIEW, 100.0 + severity, "needs_review"
    if cleanup_applied:
        return QUALITY_CLEANUP_APPLIED, 10.0 + severity, "pending"
    return QUALITY_CLEAN, severity, "pending"
