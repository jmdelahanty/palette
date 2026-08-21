"""Create a clipped-recording analysis Zarr shell.

This utility creates only structure and metadata. It does not run analysis,
write model outputs, update a registry, or set stage ``latest`` aliases.
"""

from __future__ import annotations

from fisheye.shared.json_safety import write_json_atomic as _write_json
from fisheye.shared.batch_logging import utc_now as _utc_now
import argparse
import json
import os
import re
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.compute as pc
import pyarrow.parquet as pq
import zarr

from fisheye.shared.clipped_video_collection import (
    build_clipped_video_collection_metadata,
    clipped_video_collection_summary,
)
from fisheye.shared.import_video_metadata import (
    publish_clipped_video_collection_acquisition_authority,
)
from fisheye.shared.zarr_run_completion import require_runs_parent
from fisheye.shared.system_metadata import (
    get_environment_summary,
    get_git_info,
    get_platform_info,
)


MODULE_NAME = "fisheye.utils.create_clipped_analysis_zarr"
SHELL_SCHEMA_VERSION = "palette.clipped_analysis_zarr_shell.v1"
DISH_MASK_POLICY_SCHEMA_VERSION = "palette.dish_mask_recording_camera_policy.v1"

CLIP_RUN_FAMILIES = (
    "detect_runs",
    "refined_detect_runs",
    "crop_runs",
    "keypoints_runs",
    "refined_keypoints_runs",
    "eye_masks_runs",
    "refined_eye_masks_runs",
    "subject_mask_runs",
    "refined_subject_masks_runs",
)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON {path}: {exc}") from exc


def _resolve_path(recording_dir: Path, value: Any) -> Path:
    if value is None:
        raise ValueError("Expected path value, got None")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    resolved_root = recording_dir.resolve()
    resolved = (resolved_root / path).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Recording-relative path escapes recording root: {value}")
    return resolved


def _optional_resolve_path(recording_dir: Path, value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return _resolve_path(recording_dir, text)


def _safe_group_component(value: Any, *, fallback: str) -> str:
    text = str(value or "").strip()
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return label or fallback


def _default_output_zarr(recording_dir: Path) -> Path:
    return recording_dir / "zarr" / f"{recording_dir.name}_analysis.zarr"


def _clip_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_rows = payload.get("clips")
    if raw_rows is None:
        raw_rows = payload.get("rows")
    if raw_rows is None:
        raw_rows = payload.get("camera_artifacts")
    if not isinstance(raw_rows, list):
        raise ValueError(
            "recording_clip_index JSON must include clips, rows, or camera_artifacts list"
        )

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("recording_clip_index row is not an object")
        if isinstance(raw.get("camera_artifacts"), list):
            base = {
                key: value for key, value in raw.items() if key != "camera_artifacts"
            }
            for artifact in raw["camera_artifacts"]:
                if not isinstance(artifact, Mapping):
                    raise ValueError("camera_artifacts item is not an object")
                rows.append({**base, **dict(artifact)})
        else:
            rows.append(dict(raw))
    return rows


def _metadata_summary_from_frame_index(frame_index_path: Path) -> dict[str, Any]:
    if not frame_index_path.exists():
        raise FileNotFoundError(
            f"recording_frame_index.parquet not found: {frame_index_path}"
        )
    parquet_file = pq.ParquetFile(frame_index_path)
    row_count = int(parquet_file.metadata.num_rows)
    table = pq.read_table(
        frame_index_path,
        columns=[
            "camera_serial",
            "recording_frame_id",
            "parent_frame_index",
            "clip_id",
            "clip_local_frame_index",
        ],
    ).combine_chunks()
    if table.num_rows == 0:
        raise ValueError(
            f"recording_frame_index.parquet has zero rows: {frame_index_path}"
        )
    first = {name: table[name][0].as_py() for name in table.column_names}
    last_index = table.num_rows - 1
    last = {name: table[name][last_index].as_py() for name in table.column_names}
    camera_serials = sorted({str(value.as_py()) for value in table["camera_serial"]})
    frame_range = pc.min_max(table["recording_frame_id"]).as_py()
    return {
        "path": str(frame_index_path),
        "row_count": row_count,
        "camera_serials": camera_serials,
        "first_row": first,
        "last_row": last,
        "recording_frame_id_min": int(frame_range["min"]),
        "recording_frame_id_max": int(frame_range["max"]),
    }


def _open_new_zarr(path: Path, *, overwrite: bool) -> zarr.Group:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing analysis Zarr: {path}"
            )
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return zarr.open_group(str(path), mode="w", zarr_format=3)


def _set_attrs(group: zarr.Group, attrs: Mapping[str, Any]) -> None:
    for key, value in attrs.items():
        group.attrs[key] = value


def _dish_mask_policy_attrs(camera_serials: Sequence[str]) -> dict[str, Any]:
    return {
        "dish_mask_policy_schema": DISH_MASK_POLICY_SCHEMA_VERSION,
        "dish_mask_scope": "recording_camera",
        "dish_mask_clip_policy": "single_camera_mask_applies_to_all_clips",
        "dish_mask_source_policy": "derive_once_from_first_usable_clip_or_parent_source",
        "dish_mask_coordinate_system": "recording_camera_geometry",
        "dish_mask_value_coordinate_policy": "preserve_source_tuned_array_and_normalized_metrics",
        "dish_mask_required_per_clip": False,
        "dish_mask_camera_serials": list(camera_serials),
        "orange_fixed_camera_geometry_invariant": True,
        "orange_fixed_dish_location_invariant": True,
    }


def _read_node_attrs(node_path: Path) -> dict[str, Any]:
    """Read Zarr node attrs without opening the complete store."""

    metadata_path = node_path / "zarr.json"
    if metadata_path.exists():
        payload = _read_json(metadata_path)
        attrs = payload.get("attributes") if isinstance(payload, Mapping) else None
        if attrs is None:
            return {}
        if not isinstance(attrs, Mapping):
            raise ValueError(f"Zarr attributes are not an object: {metadata_path}")
        return dict(attrs)

    legacy_attrs_path = node_path / ".zattrs"
    if legacy_attrs_path.exists():
        payload = _read_json(legacy_attrs_path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Zarr attributes are not an object: {legacy_attrs_path}")
        return dict(payload)
    return {}


def _analysis_context_payload(source_zarr: Path) -> dict[str, Any]:
    resolved = source_zarr.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Analysis-metadata source Zarr not found: {resolved}")
    analysis_attrs = _read_node_attrs(resolved / "analysis_metadata")
    root_attrs = _read_node_attrs(resolved)
    dish_mask = analysis_attrs.get("dish_mask")
    experiment_setup = root_attrs.get("experiment_setup")
    if dish_mask is not None and not isinstance(dish_mask, Mapping):
        raise ValueError(
            f"analysis_metadata.attrs['dish_mask'] is not an object: {resolved}"
        )
    if experiment_setup is not None and not isinstance(experiment_setup, Mapping):
        raise ValueError(f"root attrs['experiment_setup'] is not an object: {resolved}")
    return {
        "source_zarr": str(resolved),
        "dish_mask": dict(dish_mask) if isinstance(dish_mask, Mapping) else None,
        "experiment_setup": (
            dict(experiment_setup) if isinstance(experiment_setup, Mapping) else None
        ),
    }


def _canonical_value(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=_json_default
    )


def _discover_analysis_context_source(
    *,
    recording_path: Path,
    output_zarr: Path,
    camera_serials: Sequence[str],
    explicit_source: Path | None,
    auto_discover: bool,
    require_dish_mask: bool,
) -> dict[str, Any] | None:
    """Resolve one unambiguous sibling source for mask/setup metadata."""

    if explicit_source is not None:
        selected = _analysis_context_payload(explicit_source)
        if require_dish_mask and selected["dish_mask"] is None:
            raise ValueError(
                "--require-dish-mask was set, but the explicit analysis-metadata "
                f"source has no dish_mask: {selected['source_zarr']}"
            )
        return selected

    if not auto_discover:
        if require_dish_mask:
            raise ValueError(
                "--require-dish-mask was set while analysis-metadata discovery was disabled; "
                "pass --copy-analysis-metadata-from."
            )
        return None

    # One top-level dish_mask represents one recording-camera geometry. Do not
    # silently choose a camera-specific source for a multi-camera shell.
    if len(camera_serials) != 1:
        if require_dish_mask:
            raise ValueError(
                "Automatic dish-mask discovery requires exactly one camera; pass an explicit "
                "source after defining the multi-camera mask contract."
            )
        return None

    zarr_dir = recording_path / "zarr"
    preferred = [
        zarr_dir / f"{recording_path.name}_training.zarr",
        zarr_dir / f"{recording_path.name}_clipped_training.zarr",
    ]
    candidates: list[Path] = []
    if output_zarr.exists():
        candidates.append(output_zarr)
    candidates.extend(preferred)
    if zarr_dir.exists():
        candidates.extend(sorted(zarr_dir.glob("*.zarr")))

    payloads: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        payload = _analysis_context_payload(resolved)
        if payload["dish_mask"] is None and payload["experiment_setup"] is None:
            continue
        payloads.append(payload)

    if not payloads:
        if require_dish_mask:
            raise ValueError(
                "--require-dish-mask was set, but no sibling Zarr supplies "
                "analysis_metadata.attrs['dish_mask']; pass --copy-analysis-metadata-from."
            )
        return None

    distinct_masks = {
        _canonical_value(payload["dish_mask"])
        for payload in payloads
        if payload["dish_mask"] is not None
    }
    if len(distinct_masks) > 1:
        sources = ", ".join(str(payload["source_zarr"]) for payload in payloads)
        raise ValueError(
            "Conflicting sibling dish masks were discovered; pass "
            f"--copy-analysis-metadata-from explicitly. Candidates: {sources}"
        )
    distinct_setups = {
        _canonical_value(payload["experiment_setup"])
        for payload in payloads
        if payload["experiment_setup"] is not None
    }
    if len(distinct_setups) > 1:
        sources = ", ".join(str(payload["source_zarr"]) for payload in payloads)
        raise ValueError(
            "Conflicting sibling experiment_setup values were discovered; pass "
            f"--copy-analysis-metadata-from explicitly. Candidates: {sources}"
        )

    with_mask = [payload for payload in payloads if payload["dish_mask"] is not None]
    if require_dish_mask and not with_mask:
        raise ValueError(
            "--require-dish-mask was set, but discovered sibling Zarrs have no dish_mask."
        )
    selection_pool = with_mask or payloads
    # Prefer a source that carries both fields, then the original training Zarr
    # over a derived clipped-training copy.
    selection_pool.sort(
        key=lambda payload: (
            payload["experiment_setup"] is not None,
            str(payload["source_zarr"]).endswith("_training.zarr")
            and not str(payload["source_zarr"]).endswith("_clipped_training.zarr"),
        ),
        reverse=True,
    )
    selected = dict(selection_pool[0])

    # If the chosen mask source lacks setup but another equivalent sibling has
    # it, carry that independently with explicit provenance.
    if selected["experiment_setup"] is None:
        setup_source = next(
            (
                payload
                for payload in payloads
                if payload["experiment_setup"] is not None
            ),
            None,
        )
        if setup_source is not None:
            selected["experiment_setup"] = setup_source["experiment_setup"]
            selected["experiment_setup_source_zarr"] = setup_source["source_zarr"]
    return selected


def _copy_analysis_context(
    root: zarr.Group,
    context: Mapping[str, Any] | None,
) -> None:
    if context is None:
        return
    copied_at = _utc_now()
    source_zarr = str(context["source_zarr"])
    dish_mask = context.get("dish_mask")
    if isinstance(dish_mask, Mapping):
        analysis_metadata = root["analysis_metadata"]
        analysis_metadata.attrs["dish_mask"] = dict(dish_mask)
        analysis_metadata.attrs["dish_mask_source_zarr"] = source_zarr
        analysis_metadata.attrs["dish_mask_source_key"] = (
            "analysis_metadata.attrs.dish_mask"
        )
        analysis_metadata.attrs["dish_mask_copied_at_utc"] = copied_at
        analysis_metadata.attrs["dish_mask_copy_tool"] = MODULE_NAME

    experiment_setup = context.get("experiment_setup")
    if isinstance(experiment_setup, Mapping):
        setup_source = str(context.get("experiment_setup_source_zarr") or source_zarr)
        root.attrs["experiment_setup"] = dict(experiment_setup)
        root.attrs["experiment_setup_source_zarr"] = setup_source
        root.attrs["experiment_setup_copied_at_utc"] = copied_at
        root.attrs["experiment_setup_copy_tool"] = MODULE_NAME


def _create_common_groups(root: zarr.Group, *, camera_serials: Sequence[str]) -> None:
    raw_video = root.require_group("raw_video")
    _set_attrs(
        raw_video,
        {
            "storage_mode": "external_clips",
            "has_images_full": False,
            "has_images_ds": False,
            "frame_index_source": "recording_frame_index.parquet",
        },
    )
    root.require_group("pipeline_params")
    root.require_group("analysis")
    analysis_metadata = root.require_group("analysis_metadata")
    _set_attrs(analysis_metadata, _dish_mask_policy_attrs(camera_serials))
    root.require_group("calibration")
    experiment_index = root.require_group("experiment_index")
    experiment_index.require_group("clip_table")
    experiment_index.require_group("workflow_manifests")
    experiment_index.require_group("finalized_runs")
    root.require_group("clips")
    for family in CLIP_RUN_FAMILIES:
        family_group = require_runs_parent(root, family)
        family_group.attrs["latest"] = None
        family_group.attrs["scope"] = "parent_finalized_or_aggregated"


def _create_clip_group(
    *,
    root: zarr.Group,
    recording_dir: Path,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    clip_id = str(row.get("clip_id") or f"clip_{int(row.get('clip_index') or 0):06d}")
    camera_serial = str(row.get("camera_serial") or "")
    clip_index = int(row.get("clip_index") or 0)
    clip_group = root.require_group("clips").require_group(clip_id)
    _set_attrs(
        clip_group,
        {
            "clip_id": clip_id,
            "clip_index": clip_index,
            "granularity": "clip",
        },
    )
    camera_group_name = _safe_group_component(camera_serial, fallback="unknown_camera")
    camera_group = clip_group.require_group("cameras").require_group(camera_group_name)
    _set_attrs(
        camera_group,
        {
            "clip_id": clip_id,
            "clip_index": clip_index,
            "camera_serial": camera_serial,
            "granularity": "clip_camera",
            "recording_frame_id_first": row.get("first_recording_frame_id"),
            "recording_frame_id_last": row.get("last_recording_frame_id"),
            "frame_count": row.get("frame_count"),
            "frame_geometry_scope": "recording_camera",
            "frame_geometry_invariant_source": "orange_acquisition",
            "dish_mask_scope": "recording_camera",
            "dish_mask_clip_policy": "single_camera_mask_applies_to_all_clips",
        },
    )
    source = camera_group.require_group("source")
    video_path = _resolve_path(recording_dir, row.get("video_path"))
    metadata_path = _resolve_path(recording_dir, row.get("metadata_path"))
    keyframe_path = _optional_resolve_path(recording_dir, row.get("keyframe_path"))
    clip_manifest_path = _optional_resolve_path(
        recording_dir, row.get("clip_manifest_path")
    )
    clip_directory = str(row.get("clip_directory") or f"clips/{clip_id}")
    clip_recording_folder = _resolve_path(recording_dir, clip_directory)
    source_attrs = {
        "clip_id": clip_id,
        "clip_index": clip_index,
        "camera_serial": camera_serial,
        "camera_group": camera_group_name,
        "video_path": str(video_path),
        "metadata_path": str(metadata_path),
        "keyframe_path": str(keyframe_path) if keyframe_path is not None else None,
        "clip_manifest_path": str(clip_manifest_path)
        if clip_manifest_path is not None
        else None,
        "clip_directory": clip_directory,
        "clip_recording_folder": str(clip_recording_folder),
        "first_recording_frame_id": row.get("first_recording_frame_id"),
        "last_recording_frame_id": row.get("last_recording_frame_id"),
        "first_clip_local_frame_index": row.get("first_clip_local_frame_index"),
        "last_clip_local_frame_index": row.get("last_clip_local_frame_index"),
        "frame_count": row.get("frame_count"),
        "timestamp_first": row.get("first_timestamp"),
        "timestamp_last": row.get("last_timestamp"),
        "timestamp_sys_first": row.get("first_timestamp_sys"),
        "timestamp_sys_last": row.get("last_timestamp_sys"),
    }
    _set_attrs(source, source_attrs)
    frame_map = source.require_group("frame_map")
    _set_attrs(
        frame_map,
        {
            "source": "recording_frame_index.parquet",
            "clip_id": clip_id,
            "camera_serial": camera_serial,
            "clip_local_frame_index_semantics": "zero_based_row_number_in_per_clip_metadata_csv",
            "recording_frame_id_semantics": "session_continuous_recording_frame_id",
        },
    )
    for family in CLIP_RUN_FAMILIES:
        family_group = camera_group.require_group(family)
        family_group.attrs["latest"] = None
        family_group.attrs["clip_local"] = True
        family_group.attrs["scope"] = "clip_camera"
    return source_attrs


def create_clipped_analysis_zarr(
    recording_dir: str | Path,
    *,
    output_zarr: str | Path | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    write_manifest: bool = True,
    copy_analysis_metadata_from: str | Path | None = None,
    require_dish_mask: bool = False,
    auto_discover_analysis_metadata: bool = True,
) -> dict[str, Any]:
    """Create a clipped analysis-Zarr shell from recording sidecars."""
    started = datetime.now(timezone.utc)
    recording_path = Path(recording_dir).expanduser().resolve()
    if not recording_path.exists():
        raise FileNotFoundError(f"Recording folder not found: {recording_path}")
    clip_index_path = recording_path / "recording_clip_index.json"
    if not clip_index_path.exists():
        raise FileNotFoundError(
            f"recording_clip_index.json not found: {clip_index_path}"
        )
    frame_manifest_path = recording_path / "recording_frame_index_manifest.json"
    if not frame_manifest_path.exists():
        raise FileNotFoundError(
            f"recording_frame_index_manifest.json not found: {frame_manifest_path}"
        )

    clip_index = _read_json(clip_index_path)
    if not isinstance(clip_index, Mapping):
        raise ValueError(
            f"recording_clip_index.json is not an object: {clip_index_path}"
        )
    frame_manifest = _read_json(frame_manifest_path)
    if not isinstance(frame_manifest, Mapping):
        raise ValueError(
            f"recording_frame_index_manifest.json is not an object: {frame_manifest_path}"
        )
    if frame_manifest.get("status") != "ok":
        raise ValueError(
            f"recording frame index manifest is not ok: {frame_manifest_path}"
        )

    frame_index_path = _resolve_path(
        recording_path,
        frame_manifest.get("recording_frame_index_path")
        or "recording_frame_index.parquet",
    )
    frame_summary = _metadata_summary_from_frame_index(frame_index_path)
    rows = _clip_rows(clip_index)
    if not rows:
        raise ValueError(f"No clip rows found in {clip_index_path}")
    clip_ids = sorted(
        {
            str(row.get("clip_id") or f"clip_{int(row.get('clip_index') or 0):06d}")
            for row in rows
        }
    )
    camera_serials = sorted(
        {
            str(row.get("camera_serial") or "")
            for row in rows
            if row.get("camera_serial")
        }
    )
    recording_id = str(
        clip_index.get("recording_id")
        or frame_manifest.get("recording_id")
        or recording_path.name
    )
    session_id = str(
        clip_index.get("session_id") or frame_manifest.get("session_id") or recording_id
    )
    zarr_path = (
        Path(output_zarr).expanduser().resolve()
        if output_zarr
        else _default_output_zarr(recording_path)
    )
    shell_manifest_path = zarr_path.parent / f"{zarr_path.name}_shell_manifest.json"
    acquisition_metadata = build_clipped_video_collection_metadata(
        recording_path,
        clip_index_path=clip_index_path,
        frame_index_path=frame_index_path,
        frame_manifest_path=frame_manifest_path,
    )
    analysis_context = _discover_analysis_context_source(
        recording_path=recording_path,
        output_zarr=zarr_path,
        camera_serials=camera_serials,
        explicit_source=(
            Path(copy_analysis_metadata_from).expanduser().resolve()
            if copy_analysis_metadata_from is not None
            else None
        ),
        auto_discover=bool(auto_discover_analysis_metadata),
        require_dish_mask=bool(require_dish_mask),
    )

    planned = {
        "status": "ok",
        "generated_by": MODULE_NAME,
        "schema_version": SHELL_SCHEMA_VERSION,
        "dry_run": bool(dry_run),
        "recording_folder": str(recording_path),
        "output_zarr": str(zarr_path),
        "recording_id": recording_id,
        "session_id": session_id,
        "source_layout": "rolling_clips",
        "clip_count": len(clip_ids),
        "clip_camera_row_count": len(rows),
        "camera_serials": camera_serials,
        "recording_clip_index_json": str(clip_index_path),
        "recording_frame_index_manifest": str(frame_manifest_path),
        "recording_frame_index_path": str(frame_index_path),
        "recording_frame_index_row_count": frame_summary["row_count"],
        "recording_frame_id_min": frame_summary["recording_frame_id_min"],
        "recording_frame_id_max": frame_summary["recording_frame_id_max"],
        "clip_ids": clip_ids,
        "shell_manifest_path": str(shell_manifest_path) if write_manifest else None,
        "dish_mask_policy": _dish_mask_policy_attrs(camera_serials),
        "acquisition_source": clipped_video_collection_summary(acquisition_metadata),
        "analysis_context_source": (
            {
                "source_zarr": analysis_context["source_zarr"],
                "has_dish_mask": analysis_context.get("dish_mask") is not None,
                "has_experiment_setup": analysis_context.get("experiment_setup")
                is not None,
                "experiment_setup_source_zarr": analysis_context.get(
                    "experiment_setup_source_zarr"
                ),
            }
            if analysis_context is not None
            else None
        ),
        "require_dish_mask": bool(require_dish_mask),
    }

    if dry_run:
        return {**planned, "wrote_zarr": False, "wrote_manifest": False}

    root = _open_new_zarr(zarr_path, overwrite=overwrite)
    env_summary = get_environment_summary()
    _set_attrs(
        root,
        {
            "schema_version": "3.0.0",
            "zarr_format": 3,
            "zarr_purpose": "analysis",
            "analysis_layout": "clipped_recording_shell",
            "clipped_analysis_shell_schema": SHELL_SCHEMA_VERSION,
            "created_at": _utc_now(),
            "created_by": MODULE_NAME,
            "recording_id": recording_id,
            "session_id": session_id,
            "camera_id": acquisition_metadata["camera_id"],
            "recording_name": recording_path.name,
            "recording_path": str(recording_path),
            "source_layout": "rolling_clips",
            "recording_clip_index_json": str(clip_index_path),
            "recording_frame_index_path": str(frame_index_path),
            "recording_frame_index_manifest_path": str(frame_manifest_path),
            "recording_frame_index_schema": frame_manifest.get(
                "frame_index_schema_version"
            ),
            "source_recording_frame_index_path": str(frame_index_path),
            "source_frame_index_schema": frame_manifest.get(
                "frame_index_schema_version"
            ),
            "recording_frame_index_row_count": frame_summary["row_count"],
            "recording_frame_id_min": frame_summary["recording_frame_id_min"],
            "recording_frame_id_max": frame_summary["recording_frame_id_max"],
            "clip_count": len(clip_ids),
            "clip_camera_row_count": len(rows),
            "camera_serials": camera_serials,
            "has_raw_video": False,
            "raw_video_storage": "external_clips",
            "source_video_metadata": acquisition_metadata,
            "experiment_context_status": "absent",
            "experiment_context_source": "none",
            "stimulus_runs_available": False,
            "git_info": get_git_info(),
            "platform_info": get_platform_info(
                collect_ip=False, disk_path=str(zarr_path)
            ),
            "software_versions": env_summary.get("key_packages", {}),
            "environment": {
                "type": env_summary.get("environment_type", "unknown"),
                "name": env_summary.get("environment_name", "none"),
                "python_version": env_summary.get("python_version"),
            },
        },
    )
    _create_common_groups(root, camera_serials=camera_serials)
    _copy_analysis_context(root, analysis_context)

    clip_sources: list[dict[str, Any]] = []
    for row in sorted(
        rows,
        key=lambda item: (
            int(item.get("clip_index") or 0),
            str(item.get("camera_serial") or ""),
        ),
    ):
        clip_sources.append(
            _create_clip_group(root=root, recording_dir=recording_path, row=row)
        )

    clip_table_group = root["experiment_index"]["clip_table"]
    _set_attrs(
        clip_table_group,
        {
            "source": "recording_clip_index.json",
            "row_count": len(rows),
            "clip_count": len(clip_ids),
            "camera_serials": camera_serials,
            "recording_clip_index_json": str(clip_index_path),
        },
    )
    acquisition_publication = publish_clipped_video_collection_acquisition_authority(
        root
    )

    shell_manifest = {
        **planned,
        "dry_run": False,
        "wrote_zarr": True,
        "wrote_manifest": bool(write_manifest),
        "created_at_utc": _utc_now(),
        "host": socket.gethostname(),
        "pid": int(os.getpid()),
        "clip_sources": clip_sources,
        "acquisition_authority": acquisition_publication,
        "frame_index_summary": frame_summary,
        "duration_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
    }
    if write_manifest:
        _write_json(shell_manifest_path, shell_manifest, overwrite=overwrite)
    return shell_manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a clipped analysis Zarr shell from recording_clip_index and recording_frame_index sidecars."
    )
    parser.add_argument("recording_dir", type=Path)
    parser.add_argument("--output-zarr", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--copy-analysis-metadata-from",
        type=Path,
        help=(
            "Explicit source Zarr for analysis_metadata.dish_mask and root "
            "experiment_setup. By default, unambiguous sibling training Zarrs are discovered."
        ),
    )
    parser.add_argument(
        "--require-dish-mask",
        action="store_true",
        help="Fail before shell creation unless a dish mask can be copied.",
    )
    parser.add_argument(
        "--no-auto-discover-analysis-metadata",
        action="store_true",
        help="Disable sibling training-Zarr discovery.",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write the sibling shell manifest JSON.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def _print_summary(result: Mapping[str, Any]) -> None:
    print(f"status: {result.get('status')}")
    print(f"recording_folder: {result.get('recording_folder')}")
    print(f"output_zarr: {result.get('output_zarr')}")
    print(f"source_layout: {result.get('source_layout')}")
    print(f"clip_count: {result.get('clip_count')}")
    print(f"clip_camera_row_count: {result.get('clip_camera_row_count')}")
    print(f"camera_serials: {result.get('camera_serials')}")
    print(
        f"recording_frame_index_row_count: {result.get('recording_frame_index_row_count')}"
    )
    print(
        "recording_frame_id_range: "
        f"{result.get('recording_frame_id_min')}..{result.get('recording_frame_id_max')}"
    )
    print(f"wrote_zarr: {result.get('wrote_zarr')}")
    print(f"wrote_manifest: {result.get('wrote_manifest')}")
    if result.get("shell_manifest_path"):
        print(f"shell_manifest_path: {result.get('shell_manifest_path')}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = create_clipped_analysis_zarr(
        args.recording_dir,
        output_zarr=args.output_zarr,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        write_manifest=not args.no_manifest,
        copy_analysis_metadata_from=args.copy_analysis_metadata_from,
        require_dish_mask=args.require_dish_mask,
        auto_discover_analysis_metadata=not args.no_auto_discover_analysis_metadata,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    else:
        _print_summary(result)
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
