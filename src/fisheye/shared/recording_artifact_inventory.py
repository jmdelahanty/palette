"""Read-only inventory of per-recording Palette Zarr artifacts.

This module intentionally does not decide which run is scientifically
authoritative. It reports the run families, completion pointers, visualization
manifests, and acquisition sidecar mirrors that already exist in a single
recording Zarr so callers can reason about duplicate artifact surfaces without
reimplementing Zarr traversal.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import zarr

from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.stage_run_groups import STAGE_RUN_PARENTS
from fisheye.shared.zarr_helpers import infer_zarr_use, normalize_zarr_path
from fisheye.shared.zarr_run_completion import (
    AUTHORITATIVE_RUN_ATTR,
    COMPLETION_EPOCH_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_LATEST_COMPLETE_ATTR,
    RUN_LATEST_PENDING_ATTR,
    effective_legacy_default,
    has_run_completion_contract,
    is_run_complete,
)


INVENTORY_SCHEMA_ID = "palette.recording_artifact_inventory.v1"

_ROOT_RUN_PARENT_NAMES = {
    parent
    for parents in STAGE_RUN_PARENTS.values()
    for parent in parents
    if parent
}

_RUN_ATTR_SUMMARY_KEYS = (
    "schema_id",
    "schema_version",
    "method",
    "method_version",
    "created_at_utc",
    "row_axis",
    "source_refs",
    "source_fingerprint",
    "source_lineage_hash",
    "lineage_hash",
    "fingerprint_status",
    "source_detect_run",
    "source_refined_detect_run",
    "source_crop_run",
    "source_keypoint_run",
    "source_refined_keypoint_run",
    "source_subject_mask_run",
    "source_refined_subject_mask_run",
    "source_stimulus_run",
    "source_stimulus_epoch_run",
    "source_tracking_run",
    "source_track_kinematics_run",
)

_PARENT_POINTER_ATTRS = (
    "latest",
    RUN_LATEST_COMPLETE_ATTR,
    RUN_LATEST_PENDING_ATTR,
    AUTHORITATIVE_RUN_ATTR,
    COMPLETION_EPOCH_ATTR,
)

_ACQUISITION_STREAM_ATTRS = (
    "availability_status",
    "output_kind",
    "stream_id",
    "frame_count",
    "frames_received",
    "frames_encoded",
    "frames_dropped",
    "width",
    "height",
    "frame_rate",
    "codec",
    "container",
    "encoded_format",
    "pixel_source_format",
)

_ANALYSIS_SCOPE_NAMES = frozenset({"online", "offline"})


def _group_names(group: Any | None) -> list[str]:
    if group is None:
        return []
    keys_fn = getattr(group, "group_keys", None)
    if callable(keys_fn):
        try:
            return sorted(str(name) for name in keys_fn())
        except Exception:
            return []
    try:
        return sorted(str(name) for name, value in group.items() if _is_group(value))
    except Exception:
        return []


def _array_names(group: Any | None) -> list[str]:
    if group is None:
        return []
    keys_fn = getattr(group, "array_keys", None)
    if callable(keys_fn):
        try:
            return sorted(str(name) for name in keys_fn())
        except Exception:
            return []
    try:
        return sorted(str(name) for name, value in group.items() if _is_array(value))
    except Exception:
        return []


def _child(group: Any | None, path: str) -> Any | None:
    if group is None:
        return None
    current = group
    for part in normalize_zarr_path(path).split("/"):
        if not part:
            continue
        try:
            if part not in current:
                return None
            current = current[part]
        except Exception:
            return None
    return current


def _attrs(group: Any | None) -> dict[str, Any]:
    if group is None:
        return {}
    raw_attrs = getattr(group, "attrs", {})
    try:
        return {str(key): value for key, value in raw_attrs.items()}
    except Exception:
        return {}


def _safe_attrs_subset(group: Any | None, keys: tuple[str, ...]) -> dict[str, Any]:
    attrs = _attrs(group)
    return json_attr_safe({key: attrs[key] for key in keys if key in attrs})


def _is_group(value: Any) -> bool:
    return isinstance(value, zarr.Group) or hasattr(value, "group_keys")


def _is_array(value: Any) -> bool:
    return isinstance(value, zarr.Array) or (hasattr(value, "shape") and hasattr(value, "dtype"))


def _node_kind(value: Any | None) -> str | None:
    if value is None:
        return None
    if _is_array(value):
        return "array"
    if _is_group(value):
        return "group"
    return type(value).__name__


def _join_path(*parts: str | None) -> str:
    return "/".join(normalize_zarr_path(part) for part in parts if part and normalize_zarr_path(part))


def _is_complete(parent_group: Any, run_group: Any) -> bool:
    return is_run_complete(run_group, legacy_default=effective_legacy_default(parent_group))


def _completion_status(parent_group: Any, run_group: Any) -> str:
    attrs = _attrs(run_group)
    status = attrs.get(RUN_COMPLETION_STATUS_ATTR)
    if status is not None:
        return str(status)
    complete = _is_complete(parent_group, run_group)
    return "legacy_complete" if complete else "legacy_incomplete"


def _first_complete_child(parent_group: Any, candidates: list[str]) -> str | None:
    for name in candidates:
        child = _child(parent_group, name)
        if child is not None and _is_group(child) and _is_complete(parent_group, child):
            return name
    return None


def _resolved_latest_complete(parent_group: Any) -> str | None:
    attrs = _attrs(parent_group)
    candidates = []
    for key in ("latest", RUN_LATEST_COMPLETE_ATTR):
        value = attrs.get(key)
        if value not in (None, ""):
            candidates.append(str(value))
    candidates.extend(reversed(_group_names(parent_group)))
    return _first_complete_child(parent_group, candidates)


def _resolved_authoritative(parent_group: Any) -> str | None:
    attrs = _attrs(parent_group)
    authoritative = attrs.get(AUTHORITATIVE_RUN_ATTR)
    if authoritative not in (None, ""):
        name = str(authoritative)
        child = _child(parent_group, name)
        return name if child is not None and _is_group(child) and _is_complete(parent_group, child) else None
    return _resolved_latest_complete(parent_group)


def _parent_summary(parent_group: Any) -> dict[str, Any]:
    return {
        **_safe_attrs_subset(parent_group, _PARENT_POINTER_ATTRS),
        "run_count": len(_group_names(parent_group)),
        "resolved_latest_complete": _resolved_latest_complete(parent_group),
        "resolved_authoritative": _resolved_authoritative(parent_group),
    }


def _visualization_artifact_entry(
    *,
    run_group: Any,
    artifact_name: str,
    manifest_entry: Mapping[str, Any] | None,
    source: str,
) -> dict[str, Any]:
    path_from_manifest = None
    if isinstance(manifest_entry, Mapping):
        raw_path = manifest_entry.get("path")
        if raw_path not in (None, ""):
            path_from_manifest = str(raw_path)
    relative_path = normalize_zarr_path(path_from_manifest or f"visualizations/{artifact_name}")
    node = _child(run_group, relative_path)
    attrs = _attrs(node)
    payload: dict[str, Any] = {
        "artifact_name": str(artifact_name),
        "path": relative_path,
        "source": source,
        "present": node is not None,
        "node_type": _node_kind(node),
    }
    for key in (
        "artifact_schema_id",
        "artifact_type",
        "artifact_role",
        "media_type",
        "mime",
        "description",
        "renderer",
        "snapshot_artifact",
        "content_sha256",
        "byte_length",
        "created_at_utc",
        "created_by",
    ):
        value = attrs.get(key)
        if value is not None:
            payload[key] = value
    if isinstance(manifest_entry, Mapping):
        payload["manifest_entry"] = json_attr_safe(dict(manifest_entry))
    return json_attr_safe(payload)


def _run_visualizations(run_group: Any) -> list[dict[str, Any]]:
    attrs = _attrs(run_group)
    manifest = attrs.get("visualizations")
    artifacts: list[dict[str, Any]] = []
    seen: set[str] = set()
    if isinstance(manifest, Mapping):
        for artifact_name, manifest_entry in sorted(manifest.items()):
            name = str(artifact_name)
            artifacts.append(
                _visualization_artifact_entry(
                    run_group=run_group,
                    artifact_name=name,
                    manifest_entry=manifest_entry if isinstance(manifest_entry, Mapping) else None,
                    source="manifest",
                )
            )
            seen.add(name)
    vis_group = _child(run_group, "visualizations")
    if _is_group(vis_group):
        for artifact_name in _group_names(vis_group) + _array_names(vis_group):
            if artifact_name in seen:
                continue
            artifacts.append(
                _visualization_artifact_entry(
                    run_group=run_group,
                    artifact_name=artifact_name,
                    manifest_entry=None,
                    source="visualizations_group",
                )
            )
    return artifacts


def _run_summary(parent_group: Any, run_name: str, run_group: Any, *, run_path: str) -> dict[str, Any]:
    visualizations = _run_visualizations(run_group)
    return {
        "name": str(run_name),
        "path": run_path,
        "complete": _is_complete(parent_group, run_group),
        "completion_status": _completion_status(parent_group, run_group),
        "has_completion_contract": has_run_completion_contract(run_group),
        "attrs": _safe_attrs_subset(run_group, _RUN_ATTR_SUMMARY_KEYS),
        "array_count": len(_array_names(run_group)),
        "group_count": len(_group_names(run_group)),
        "visualization_count": len(visualizations),
        "visualizations": visualizations,
    }


def _looks_like_scope_group(name: str, group: Any) -> bool:
    if str(name) not in _ANALYSIS_SCOPE_NAMES:
        return False
    if not _is_group(group):
        return False
    if _array_names(group):
        return False
    if _child(group, "visualizations") is not None:
        return False
    return bool(_group_names(group))


def _summarize_run_parent(
    *,
    family_path: str,
    parent_path: str,
    parent_group: Any,
    family_kind: str,
    scope: str | None = None,
) -> dict[str, Any]:
    runs = [
        _run_summary(
            parent_group,
            run_name,
            _child(parent_group, run_name),
            run_path=_join_path(parent_path, run_name),
        )
        for run_name in _group_names(parent_group)
        if _is_group(_child(parent_group, run_name))
    ]
    return {
        "family_path": family_path,
        "run_parent_path": parent_path,
        "family_kind": family_kind,
        "scope": scope,
        "parent": _parent_summary(parent_group),
        "run_count": len(runs),
        "runs": runs,
    }


def _run_parent_entries(
    *,
    family_path: str,
    family_group: Any,
    family_kind: str,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    child_names = _group_names(family_group)
    scope_names = [
        name
        for name in child_names
        if family_kind == "analysis" and _looks_like_scope_group(name, _child(family_group, name))
    ]
    direct_run_names = [
        name
        for name in child_names
        if name not in scope_names and _is_group(_child(family_group, name))
    ]
    if direct_run_names:
        entries.append(
            _summarize_run_parent(
                family_path=family_path,
                parent_path=family_path,
                parent_group=family_group,
                family_kind=family_kind,
            )
        )
    for scope_name in scope_names:
        scope_group = _child(family_group, scope_name)
        if scope_group is None:
            continue
        entries.append(
            _summarize_run_parent(
                family_path=family_path,
                parent_path=_join_path(family_path, scope_name),
                parent_group=scope_group,
                family_kind=f"{family_kind}_scoped",
                scope=scope_name,
            )
        )
    if not entries:
        entries.append(
            _summarize_run_parent(
                family_path=family_path,
                parent_path=family_path,
                parent_group=family_group,
                family_kind=family_kind,
            )
        )
    return entries


def _root_run_family_paths(root: Any) -> list[str]:
    paths: list[str] = []
    for name in _group_names(root):
        if name == "analysis":
            continue
        if name.endswith("_runs") or name in _ROOT_RUN_PARENT_NAMES:
            paths.append(name)
    return sorted(set(paths))


def _analysis_run_family_paths(root: Any) -> list[str]:
    analysis = _child(root, "analysis")
    if not _is_group(analysis):
        return []
    paths: list[str] = []

    def walk(group: Any, path: str, depth: int) -> None:
        if depth > 5:
            return
        for name in _group_names(group):
            child = _child(group, name)
            if not _is_group(child):
                continue
            child_path = _join_path(path, name)
            if name.endswith("_runs"):
                paths.append(child_path)
                continue
            if name in {"visualizations", "streams"}:
                continue
            walk(child, child_path, depth + 1)

    walk(analysis, "analysis", 0)
    return sorted(set(paths))


def _nested_quality_report_parents(root: Any, run_family_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for family in run_family_entries:
        for run in family.get("runs", []):
            run_path = str(run.get("path") or "")
            run_group = _child(root, run_path)
            quality_parent = _child(run_group, "quality_reports")
            if not _is_group(quality_parent):
                continue
            quality_parent_path = _join_path(run_path, "quality_reports")
            entries.append(
                _summarize_run_parent(
                    family_path=quality_parent_path,
                    parent_path=quality_parent_path,
                    parent_group=quality_parent,
                    family_kind="nested_quality_reports",
                )
            )
    return entries


def _acquisition_video_streams(root: Any) -> dict[str, Any]:
    parent = _child(root, "analysis/acquisition_video_streams")
    if not _is_group(parent):
        return {"available": False}
    attrs = _attrs(parent)
    streams_group = _child(parent, "streams")
    stream_entries: list[dict[str, Any]] = []
    if _is_group(streams_group):
        for stream_key in _group_names(streams_group):
            stream_group = _child(streams_group, stream_key)
            stream_attrs = _attrs(stream_group)
            entry = {
                "stream_key": stream_key,
                **json_attr_safe({key: stream_attrs[key] for key in _ACQUISITION_STREAM_ATTRS if key in stream_attrs}),
            }
            files = stream_attrs.get("files")
            if isinstance(files, Mapping):
                entry["file_keys"] = sorted(str(key) for key in files)
                for file_key in ("video", "metadata", "frame_clock_metadata", "summary", "status"):
                    file_entry = files.get(file_key)
                    if isinstance(file_entry, Mapping):
                        entry[f"{file_key}_exists"] = bool(file_entry.get("exists"))
            stream_entries.append(entry)
    return json_attr_safe(
        {
            "available": True,
            "path": "analysis/acquisition_video_streams",
            "schema_id": attrs.get("schema_id"),
            "schema_version": attrs.get("schema_version"),
            "source_schema_id": attrs.get("source_schema_id"),
            "inventory_status": attrs.get("inventory_status"),
            "stream_count": attrs.get("stream_count", len(stream_entries)),
            "stream_keys": attrs.get("stream_keys") or [entry["stream_key"] for entry in stream_entries],
            "crop_stream_available": attrs.get("crop_stream_available"),
            "streams": stream_entries,
        }
    )


def _registry_projection_names(inventory: Mapping[str, Any]) -> dict[str, list[str]]:
    optional: list[str] = []
    acquisition = inventory.get("acquisition_video_streams")
    if isinstance(acquisition, Mapping) and acquisition.get("available"):
        optional.extend(
            [
                "acquisition_video_streams",
                "dataset_acquisition_video_streams_current",
                "recording_acquisition_video_streams_current",
            ]
        )
    if int(inventory.get("visualization_artifact_count") or 0) > 0:
        optional.append("recording_artifacts")
    return {
        "core": ["datasets", "recordings", "recording_step_status"],
        "optional": sorted(set(optional)),
    }


def build_recording_artifact_inventory(
    root: zarr.Group,
    *,
    zarr_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable, read-only artifact inventory for one Zarr root."""

    root_run_families: list[dict[str, Any]] = []
    for family_path in _root_run_family_paths(root):
        group = _child(root, family_path)
        if _is_group(group):
            root_run_families.extend(
                _run_parent_entries(
                    family_path=family_path,
                    family_group=group,
                    family_kind="root_stage",
                )
            )

    analysis_run_families: list[dict[str, Any]] = []
    for family_path in _analysis_run_family_paths(root):
        group = _child(root, family_path)
        if _is_group(group):
            analysis_run_families.extend(
                _run_parent_entries(
                    family_path=family_path,
                    family_group=group,
                    family_kind="analysis",
                )
            )

    nested_report_families = _nested_quality_report_parents(
        root,
        root_run_families + analysis_run_families,
    )
    all_run_families = root_run_families + analysis_run_families + nested_report_families
    visualization_artifact_count = sum(
        int(run.get("visualization_count") or 0)
        for family in all_run_families
        for run in family.get("runs", [])
    )
    run_count = sum(int(family.get("run_count") or 0) for family in all_run_families)

    inventory: dict[str, Any] = {
        "schema_id": INVENTORY_SCHEMA_ID,
        "zarr_path": str(zarr_path) if zarr_path is not None else None,
        "zarr_use": infer_zarr_use(root, zarr_path),
        "root_attrs": _safe_attrs_subset(
            root,
            (
                "recording_id",
                "recording_name",
                "session_uuid",
                "dataset_id",
                "zarr_purpose",
                "zarr_use",
                "schema_id",
                "schema_version",
                "acquisition_video_streams_available",
                "acquisition_video_streams_path",
            ),
        ),
        "root_run_families": root_run_families,
        "analysis_run_families": analysis_run_families,
        "nested_report_families": nested_report_families,
        "run_family_count": len(all_run_families),
        "run_count": run_count,
        "visualization_artifact_count": visualization_artifact_count,
        "acquisition_video_streams": _acquisition_video_streams(root),
    }
    inventory["registry_projection_names"] = _registry_projection_names(inventory)
    return json_attr_safe(inventory)
