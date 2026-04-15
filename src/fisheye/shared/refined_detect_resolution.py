from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import zarr

from .refined_detect_curation import (
    has_curated_refined_detect_surface,
    has_sparse_curated_refined_detect_instances_arrays,
    resolve_curated_refined_detect_run,
)
from .refined_detect_review import DEFAULT_DETECT_GROUP_PREFERENCE, resolve_refined_detect_group
from .type_conversions import normalize_attr


REVIEW_STATUS_DETECT_GROUP_PREFERENCE = ("refined", "manual", "interpolated", "filtered", "raw")


@dataclass(frozen=True)
class RefinedDetectionReadResolution:
    detection_path: Optional[str]
    detection_kind: Optional[str]
    refined_detect_run: Optional[str]
    refined_sparse_group: Optional[str]
    source_detect_run: Optional[str]
    curated_root: bool


@dataclass(frozen=True)
class RefinedDetectReviewTargetResolution:
    resolved_group: Optional[str]
    preference_chain: Sequence[str]


def _sorted_group_names(group: Optional[zarr.Group]) -> List[str]:
    if group is None:
        return []
    keys_fn = getattr(group, "group_keys", None)
    try:
        names = list(keys_fn()) if callable(keys_fn) else []
    except Exception:
        names = []
    return sorted(name for name in names if isinstance(name, str))


def _resolve_refined_parent(root: zarr.Group) -> tuple[Optional[zarr.Group], Optional[str]]:
    refined_parent = root.get("refined_detect_runs")
    if refined_parent is not None:
        return refined_parent, "refined_detect_runs"
    refined_parent = root.get("refined_runs")
    if refined_parent is not None:
        return refined_parent, "refined_runs"
    return None, None


def resolve_active_curated_refined_run_name(
    root: zarr.Group,
    *,
    run_name: Optional[str] = None,
) -> Optional[str]:
    refined_parent, _ = _resolve_refined_parent(root)
    if refined_parent is None:
        return None

    requested = normalize_attr(run_name)
    if requested:
        if requested not in refined_parent:
            raise ValueError(f"Refined detect run not found: {requested}")
        return requested if has_curated_refined_detect_surface(refined_parent[requested]) else None

    latest = normalize_attr(refined_parent.attrs.get("latest"))
    if latest and latest in refined_parent and has_curated_refined_detect_surface(refined_parent[latest]):
        return latest

    for candidate in reversed(_sorted_group_names(refined_parent)):
        if has_curated_refined_detect_surface(refined_parent[candidate]):
            return candidate
    return None


def resolve_detection_read_source(
    root: zarr.Group,
    *,
    prefer_curated: bool = True,
    refined_run: Optional[str] = None,
    preference: Sequence[str] = DEFAULT_DETECT_GROUP_PREFERENCE,
    allow_sparse_fallback: bool = True,
) -> RefinedDetectionReadResolution:
    refined_parent, parent_name = _resolve_refined_parent(root)

    if prefer_curated and refined_parent is not None:
        curated_run = resolve_active_curated_refined_run_name(root, run_name=refined_run)
        if curated_run is not None:
            curated_group, _ = resolve_curated_refined_detect_run(root, run_name=curated_run)
            if has_sparse_curated_refined_detect_instances_arrays(curated_group):
                detection_path = f"{parent_name}/{curated_run}/instances"
            else:
                detection_path = f"{parent_name}/{curated_run}"
            return RefinedDetectionReadResolution(
                detection_path=detection_path,
                detection_kind="refined",
                refined_detect_run=curated_run,
                refined_sparse_group=normalize_attr(curated_group.attrs.get("active_sparse_group")),
                source_detect_run=normalize_attr(curated_group.attrs.get("source_detect_run")),
                curated_root=True,
            )
        if not allow_sparse_fallback:
            raise ValueError("No curated refined detect run available and sparse fallback is disabled.")

    if refined_parent is not None:
        resolved_refined_run = normalize_attr(refined_run)
        if resolved_refined_run is None:
            resolved_refined_run = normalize_attr(refined_parent.attrs.get("latest"))
            if resolved_refined_run is None:
                names = _sorted_group_names(refined_parent)
                resolved_refined_run = names[-1] if names else None
        if resolved_refined_run is not None:
            if resolved_refined_run not in refined_parent:
                raise ValueError(f"Refined detect run not found: {resolved_refined_run}")
            refined_group = refined_parent[resolved_refined_run]
            resolution = resolve_refined_detect_group(
                refined_group,
                preference=preference,
                override_group=None,
            )
            source_detect_run = normalize_attr(resolution.source_detect_run) or normalize_attr(
                refined_group.attrs.get("source_detect_run")
            )
            if resolution.group is not None:
                detection_kind = (
                    "manual"
                    if (normalize_attr(resolution.label) or "").lower() == "manual"
                    else str(normalize_attr(resolution.label) or resolution.group).lower()
                )
                return RefinedDetectionReadResolution(
                    detection_path=f"{parent_name}/{resolved_refined_run}/{resolution.group}",
                    detection_kind=detection_kind,
                    refined_detect_run=resolved_refined_run,
                    refined_sparse_group=resolution.group,
                    source_detect_run=source_detect_run,
                    curated_root=False,
                )
            if source_detect_run and root.get("detect_runs") is not None and source_detect_run in root["detect_runs"]:
                return RefinedDetectionReadResolution(
                    detection_path=f"detect_runs/{source_detect_run}",
                    detection_kind="raw",
                    refined_detect_run=resolved_refined_run,
                    refined_sparse_group=None,
                    source_detect_run=source_detect_run,
                    curated_root=False,
                )

    detect_parent = root.get("detect_runs")
    if detect_parent is not None:
        detect_run_name = normalize_attr(detect_parent.attrs.get("latest"))
        if detect_run_name is None:
            names = _sorted_group_names(detect_parent)
            detect_run_name = names[-1] if names else None
        if detect_run_name is not None and detect_run_name in detect_parent:
            return RefinedDetectionReadResolution(
                detection_path=f"detect_runs/{detect_run_name}",
                detection_kind="raw",
                refined_detect_run=None,
                refined_sparse_group=None,
                source_detect_run=detect_run_name,
                curated_root=False,
            )

    raise ValueError("No detection source available.")


def resolve_detect_review_target(
    root: zarr.Group,
    *,
    refined_run_name: str,
    refined_run: zarr.Group,
    override_group: Optional[str] = None,
) -> RefinedDetectReviewTargetResolution:
    preference_chain = REVIEW_STATUS_DETECT_GROUP_PREFERENCE
    requested = normalize_attr(override_group)
    requested = requested.lower() if requested else None

    if requested:
        if requested == "refined":
            resolved_group = "refined" if has_curated_refined_detect_surface(refined_run) else None
            return RefinedDetectReviewTargetResolution(
                resolved_group=resolved_group,
                preference_chain=preference_chain,
            )
        resolution = resolve_refined_detect_group(
            refined_run,
            preference=DEFAULT_DETECT_GROUP_PREFERENCE,
            override_group=requested,
        )
        return RefinedDetectReviewTargetResolution(
            resolved_group=resolution.label or resolution.group,
            preference_chain=preference_chain,
        )

    resolution = resolve_detection_read_source(
        root,
        prefer_curated=True,
        refined_run=refined_run_name,
        allow_sparse_fallback=True,
    )
    if resolution.curated_root:
        resolved_group = "refined"
    else:
        resolved_group = resolution.detection_kind or resolution.refined_sparse_group
    return RefinedDetectReviewTargetResolution(
        resolved_group=resolved_group,
        preference_chain=preference_chain,
    )


__all__ = [
    "REVIEW_STATUS_DETECT_GROUP_PREFERENCE",
    "RefinedDetectReviewTargetResolution",
    "RefinedDetectionReadResolution",
    "resolve_detect_review_target",
    "resolve_active_curated_refined_run_name",
    "resolve_detection_read_source",
]
