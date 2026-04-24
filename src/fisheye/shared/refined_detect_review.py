from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import zarr

from .refined_detect_curation import has_sparse_curated_refined_detect_instances_arrays
from .type_conversions import normalize_attr as _normalize_attr


DEFAULT_DETECT_GROUP_PREFERENCE = ("refined", "manual", "interpolated", "filtered", "raw")


@dataclass(frozen=True)
class RefinedDetectResolution:
    label: Optional[str]
    group: Optional[str]
    source_detect_run: Optional[str]


def resolve_refined_detect_group(
    refined_run: zarr.Group,
    preference: Optional[Iterable[str]] = None,
    override_group: Optional[str] = None,
) -> RefinedDetectResolution:
    source_detect_run = _normalize_attr(refined_run.attrs.get("source_detect_run"))
    manual_label = _normalize_attr(refined_run.attrs.get("manual_review_latest"))

    if override_group:
        requested = _normalize_attr(override_group) or ""
        token = requested.lower()
        if token == "raw":
            return RefinedDetectResolution("raw", None, source_detect_run)
        if token in {"refined", "instances"} and has_sparse_curated_refined_detect_instances_arrays(
            refined_run
        ):
            return RefinedDetectResolution("refined", "instances", source_detect_run)
        if requested in refined_run:
            label = "manual" if manual_label and requested == manual_label else requested
            if requested == "instances":
                label = "refined"
            return RefinedDetectResolution(label, requested, source_detect_run)
        return RefinedDetectResolution(None, None, source_detect_run)

    pref_list = [str(item).lower() for item in (preference or DEFAULT_DETECT_GROUP_PREFERENCE)]
    for token in pref_list:
        if token in {"refined", "instances"}:
            if has_sparse_curated_refined_detect_instances_arrays(refined_run):
                return RefinedDetectResolution("refined", "instances", source_detect_run)
        elif token == "manual":
            if manual_label and manual_label in refined_run:
                return RefinedDetectResolution("manual", manual_label, source_detect_run)
            if "manual" in refined_run:
                return RefinedDetectResolution("manual", "manual", source_detect_run)
        elif token in ("interpolated", "filtered"):
            if token in refined_run:
                return RefinedDetectResolution(token, token, source_detect_run)
        elif token == "raw":
            return RefinedDetectResolution("raw", None, source_detect_run)
        else:
            if token in refined_run:
                return RefinedDetectResolution(token, token, source_detect_run)

    return RefinedDetectResolution(None, None, source_detect_run)
