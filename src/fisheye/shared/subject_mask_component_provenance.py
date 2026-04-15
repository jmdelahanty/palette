"""Helpers for canonical component-scoped subject-mask provenance attrs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .provenance_attrs import extract_source_crop_snapshot_attrs


def _normalize_source_channels(value: object) -> list[str]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def write_subject_mask_component_provenance(
    run_group: Any,
    *,
    component_name: str,
    source_stage: str,
    source_run: str,
    source_method: str,
    source_channels: Sequence[str] | np.ndarray | str,
    source_label_schema_id: str | None = None,
    projection_mode: str | None = None,
    source_created_at_utc: str | None = None,
    source_crop_run: str | None = None,
    source_crop_snapshot: Mapping[str, Any] | None = None,
) -> Any:
    """Write canonical component provenance attrs under components/<name>/provenance."""

    component_group = run_group.require_group("components").require_group(str(component_name))
    provenance_group = component_group.require_group("provenance")
    provenance_group.attrs["source_stage"] = str(source_stage)
    provenance_group.attrs["source_run"] = str(source_run)
    provenance_group.attrs["source_method"] = str(source_method)
    provenance_group.attrs["source_channels"] = _normalize_source_channels(source_channels)
    if source_crop_run is not None:
        provenance_group.attrs["source_crop_run"] = str(source_crop_run)
    for key, value in extract_source_crop_snapshot_attrs(source_crop_snapshot).items():
        provenance_group.attrs[key] = value
    if source_label_schema_id is not None:
        provenance_group.attrs["source_label_schema_id"] = str(source_label_schema_id)
    if projection_mode is not None:
        provenance_group.attrs["projection_mode"] = str(projection_mode)
    if source_created_at_utc is not None:
        provenance_group.attrs["source_created_at_utc"] = str(source_created_at_utc)
    return provenance_group
