from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import zarr


DEFAULT_DETECT_GROUP_PREFERENCE = ("manual", "interpolated", "filtered", "raw")


@dataclass(frozen=True)
class RefinedDetectResolution:
    label: Optional[str]
    group: Optional[str]
    source_detect_run: Optional[str]


def _normalize_attr(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    text = str(value).strip()
    return text or None


def resolve_refined_detect_group(
    refined_run: zarr.Group,
    preference: Optional[Iterable[str]] = None,
    override_group: Optional[str] = None,
) -> RefinedDetectResolution:
    source_detect_run = _normalize_attr(refined_run.attrs.get("source_detect_run"))
    manual_label = _normalize_attr(refined_run.attrs.get("manual_review_latest"))

    if override_group:
        if override_group == "raw":
            return RefinedDetectResolution("raw", None, source_detect_run)
        if override_group in refined_run:
            label = "manual" if manual_label and override_group == manual_label else override_group
            return RefinedDetectResolution(label, override_group, source_detect_run)
        return RefinedDetectResolution(None, None, source_detect_run)

    pref_list = [str(item).lower() for item in (preference or DEFAULT_DETECT_GROUP_PREFERENCE)]
    for token in pref_list:
        if token == "manual":
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
