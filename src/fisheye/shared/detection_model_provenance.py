"""Persist registry model-selection provenance on a detection candidate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import zarr


def write_detect_model_resolution_provenance(
    *,
    zarr_path: Path,
    run_name: str,
    payload: Mapping[str, Any],
) -> None:
    root = zarr.open_group(str(zarr_path), mode="r+", use_consolidated=False)
    detect_parent = root.get("detect_runs")
    if detect_parent is None or run_name not in detect_parent:
        raise RuntimeError(
            f"detect run not found for provenance annotation: detect_runs/{run_name}"
        )

    detect_group = detect_parent[run_name]
    selected_value = payload.get("selected")
    selected = selected_value if isinstance(selected_value, Mapping) else {}
    attrs = dict(detect_group.attrs)
    attrs["model_resolution_mode"] = "registry"
    attrs["model_resolution_task"] = "detect"
    attrs["model_resolution_registry_path"] = payload.get("registry_path")
    attrs["model_resolution_recording_id"] = payload.get("recording_id")
    attrs["model_resolution_selected_run_id"] = selected.get("run_id")
    attrs["model_resolution_selected_set_id"] = selected.get("set_id")
    attrs["model_resolution_selected_model_path"] = selected.get("model_path")
    attrs["model_resolution_selected_score"] = selected.get("score")
    attrs["model_resolution_selected_created_utc"] = selected.get("created_utc")
    attrs["model_resolution_resolved_at_utc"] = payload.get("resolved_at_utc")
    attrs["model_resolution_candidates_json"] = json.dumps(
        payload.get("candidates", []),
        sort_keys=True,
    )

    provenance = attrs.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    provenance["model_resolution"] = dict(payload)
    attrs["provenance"] = provenance
    detect_group.attrs.put(attrs)


__all__ = ["write_detect_model_resolution_provenance"]
