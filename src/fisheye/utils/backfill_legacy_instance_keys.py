"""Repair or backfill stable observation keys in selected analysis lineages.

The migration does not rerun inference or change run selectors. Keys are
minted at legacy detection origins and copied through row-lineage mappings
exactly as modern writers do. Existing arrays are replaced only when they
exactly reproduce a known legacy clipped-local identity recipe.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pyarrow.parquet as pq
import zarr

from fisheye.shared.instance_keys import (
    INSTANCE_KEY_ORIGIN_ARRAY,
    INSTANCE_KEY_ORIGIN_CODE_MAP,
    instance_key_attrs,
    mint_detection_instance_keys,
    resolve_recording_identity,
)
from fisheye.shared.refined_detect_curation import resolve_curated_instance_keys
from fisheye.shared.rowset_fingerprint import (
    build_rowset_fingerprint,
    resolve_rowset_edit_revision,
)


MIGRATION_ID = "palette.instance_key_lineage_repair.v2"
MIGRATION_TOOL = "fisheye.utils.backfill_legacy_instance_keys"
DEFAULT_CHUNK_ROWS = 16_384
DEFAULT_TABULAR_SHARD_ROWS = 131_072


@dataclass(frozen=True)
class PlannedArray:
    group_path: str
    name: str
    values: np.ndarray
    policy: str
    shard_rows: int | None = None
    action: str = "add"

    @property
    def row_count(self) -> int:
        return int(self.values.shape[0])


@dataclass
class InstanceKeyBackfillPlan:
    zarr_path: Path
    recording_identity: str
    selected_runs: dict[str, str]
    arrays: list[PlannedArray] = field(default_factory=list)
    attrs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def array(self, group_path: str, name: str) -> PlannedArray:
        for item in self.arrays:
            if item.group_path == group_path and item.name == name:
                return item
        raise KeyError(f"No planned array {group_path}/{name}.")

    def summary(self) -> dict[str, Any]:
        return {
            "zarr_path": str(self.zarr_path),
            "recording_identity": self.recording_identity,
            "selected_runs": dict(self.selected_runs),
            "arrays": [
                {
                    "path": f"{item.group_path}/{item.name}",
                    "rows": item.row_count,
                    "dtype": str(item.values.dtype),
                    "policy": item.policy,
                    "action": item.action,
                    "storage_layout": (
                        "indexed_sharding_v1" if item.shard_rows is not None else "regular_chunks_v1"
                    ),
                    "shard_rows": item.shard_rows,
                }
                for item in self.arrays
            ],
            "attr_groups": sorted(self.attrs),
        }


def _selected_run(root: Any, parent_name: str) -> tuple[str, Any]:
    if parent_name not in root:
        raise ValueError(f"Archive is missing required parent {parent_name}.")
    parent = root[parent_name]
    run_name = parent.attrs.get("latest_complete") or parent.attrs.get("latest")
    if not run_name or str(run_name) not in parent:
        raise ValueError(f"{parent_name} has no resolvable selected complete run.")
    run_name = str(run_name)
    run = parent[run_name]
    completion = str(run.attrs.get("palette_run_completion_status") or "").strip().lower()
    if completion != "complete":
        raise ValueError(
            f"Refusing to migrate incomplete selected run {parent_name}/{run_name}: "
            f"palette_run_completion_status={completion or '<missing>'}."
        )
    return run_name, run


def _array(group: Any, name: str, *, dtype: np.dtype[Any] | str) -> np.ndarray:
    if name not in group:
        raise ValueError(f"{getattr(group, 'path', '<group>')} is missing required array {name}.")
    return np.asarray(group[name][:], dtype=dtype)


def _row_array(group: Any, name: str, *, dtype: np.dtype[Any] | str) -> np.ndarray:
    return _array(group, name, dtype=dtype).reshape(-1)


def _require_unique(keys: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(keys, dtype=np.uint64).reshape(-1)
    if int(np.unique(values).shape[0]) != int(values.shape[0]):
        raise ValueError(f"{label} contains duplicate instance_key values.")
    return values


def _mapped_keys(
    source_keys: np.ndarray,
    row_indices: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    keys = np.asarray(source_keys, dtype=np.uint64).reshape(-1)
    rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= int(keys.shape[0])):
        raise ValueError(f"{label} contains source rows outside 0..{int(keys.shape[0]) - 1}.")
    return _require_unique(keys[rows], label=label)


def _check_frame_mapping(
    *,
    source_frames: np.ndarray,
    source_rows: np.ndarray,
    target_frames: np.ndarray,
    label: str,
) -> None:
    rows = np.asarray(source_rows, dtype=np.int64).reshape(-1)
    source = np.asarray(source_frames, dtype=np.int64).reshape(-1)
    target = np.asarray(target_frames, dtype=np.int64).reshape(-1)
    if int(rows.shape[0]) != int(target.shape[0]):
        raise ValueError(f"{label} row mapping length does not match target frame count.")
    if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= int(source.shape[0])):
        raise ValueError(f"{label} source row mapping is out of range.")
    if not np.array_equal(source[rows], target):
        raise ValueError(f"{label} source rows do not map to matching frame_indices.")


def _copy_or_mint_crop_keys(
    *,
    recording_identity: str,
    crop_group: Any,
    detect_keys: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    frames = _row_array(crop_group, "frame_indices", dtype=np.int64)
    bboxes = _array(crop_group, "bbox_norm_coords", dtype=np.float64).reshape(-1, 4)
    classes = (
        _row_array(crop_group, "class_ids", dtype=np.int64)
        if "class_ids" in crop_group
        else np.zeros(frames.shape[0], dtype=np.int64)
    )
    keys = mint_detection_instance_keys(
        recording_identity=recording_identity,
        frame_indices=frames,
        bbox_norm_coords=bboxes,
        class_ids=classes,
    )
    copied = 0
    if "source_detect_row_index" in crop_group:
        rows = _row_array(crop_group, "source_detect_row_index", dtype=np.int64)
        if int(rows.shape[0]) != int(frames.shape[0]):
            raise ValueError("crop source_detect_row_index length does not match crop rows.")
        valid = (rows >= 0) & (rows < int(detect_keys.shape[0]))
        keys[valid] = np.asarray(detect_keys, dtype=np.uint64)[rows[valid]]
        copied = int(np.count_nonzero(valid))
    keys = _require_unique(keys, label="crop rowset")
    return keys, {"copied_from_detect": copied, "minted_legacy_detection_origin": int(keys.shape[0]) - copied}


def _planned_existing_or_new(
    group: Any,
    *,
    group_path: str,
    name: str,
    values: np.ndarray,
    policy: str,
    shard_rows_override: int | None = None,
    replace_existing_values: np.ndarray | None = None,
) -> PlannedArray:
    shard_rows: int | None = None
    if group_path.startswith("keypoints_runs/"):
        if str(group.attrs.get("keypoint_storage_layout") or "") == "indexed_sharding_v1":
            value = group.attrs.get("keypoint_roi_shard_rows")
            shard_rows = int(value) if value is not None else None
    elif group_path.startswith("detect_runs/"):
        if str(group.attrs.get("detect_storage_layout") or "") == "indexed_sharding_v1":
            value = group.attrs.get("detect_row_shard_rows")
            shard_rows = int(value) if value is not None else None
    if shard_rows is not None and shard_rows <= 0:
        raise ValueError(f"Invalid sharding metadata on {group_path}: shard_rows={shard_rows}.")
    if shard_rows_override is not None:
        if int(shard_rows_override) <= 0:
            raise ValueError("shard_rows_override must be positive.")
        shard_rows = int(shard_rows_override)
    action = "add"
    if name in group:
        target_values = np.asarray(values)
        existing = np.asarray(group[name][:], dtype=target_values.dtype)
        if np.array_equal(existing, target_values):
            action = "verify_existing"
        elif replace_existing_values is not None and np.array_equal(
            existing,
            np.asarray(replace_existing_values, dtype=target_values.dtype),
        ):
            action = "replace_verified_legacy"
        else:
            raise ValueError(f"Existing {group_path}/{name} disagrees with deterministic backfill.")
    return PlannedArray(
        group_path=group_path,
        name=name,
        values=np.asarray(values),
        policy=policy,
        shard_rows=shard_rows,
        action=action,
    )


def _build_dense_plan(root: Any, *, zarr_path: Path) -> InstanceKeyBackfillPlan:
    """Build and fully validate an additive migration plan for one archive."""

    recording_identity = resolve_recording_identity(root.attrs, fallback_path=zarr_path)
    plan = InstanceKeyBackfillPlan(
        zarr_path=Path(zarr_path),
        recording_identity=recording_identity,
        selected_runs={},
    )

    detect_name, detect = _selected_run(root, "detect_runs")
    detect_path = f"detect_runs/{detect_name}"
    plan.selected_runs["detect"] = detect_name
    detect_frames = _row_array(detect, "frame_indices", dtype=np.int64)
    detect_bboxes = _array(detect, "bbox_norm_coords", dtype=np.float64).reshape(-1, 4)
    detect_classes = (
        _row_array(detect, "class_ids", dtype=np.int64)
        if "class_ids" in detect
        else np.zeros(detect_frames.shape[0], dtype=np.int64)
    )
    detect_keys = _require_unique(
        mint_detection_instance_keys(
            recording_identity=recording_identity,
            frame_indices=detect_frames,
            bbox_norm_coords=detect_bboxes,
            class_ids=detect_classes,
        ),
        label="detect rowset",
    )
    plan.arrays.append(
        _planned_existing_or_new(
            detect,
            group_path=detect_path,
            name="instance_key",
            values=detect_keys,
            policy="minted_at_legacy_detect_origin",
        )
    )
    plan.attrs[detect_path] = {
        **instance_key_attrs(recording_identity),
        "instance_key_status": "present",
        "instance_key_policy": "minted_at_legacy_detect_origin",
    }

    refined_name, refined = _selected_run(root, "refined_detect_runs")
    plan.selected_runs["refined_detect"] = refined_name
    if "instances" not in refined:
        raise ValueError(f"refined_detect_runs/{refined_name} lacks instances.")
    instances = refined["instances"]
    instances_path = f"refined_detect_runs/{refined_name}/instances"
    instance_keys, origin_codes = resolve_curated_instance_keys(
        root,
        zarr_path=zarr_path,
        instance_frame_indices=_row_array(instances, "frame_indices", dtype=np.int64),
        instance_bbox_norm_coords=_array(instances, "bbox_norm_coords", dtype=np.float64).reshape(-1, 4),
        instance_class_ids=(
            _row_array(instances, "class_ids", dtype=np.int64)
            if "class_ids" in instances
            else None
        ),
        instance_source_detect_row_index=_row_array(
            instances, "source_detect_row_index", dtype=np.int64
        ),
        source_detection_instance_key=detect_keys,
    )
    plan.arrays.extend(
        (
            _planned_existing_or_new(
                instances,
                group_path=instances_path,
                name="instance_key",
                values=instance_keys,
                policy="copied_from_detect_or_minted_at_curation",
            ),
            _planned_existing_or_new(
                instances,
                group_path=instances_path,
                name=INSTANCE_KEY_ORIGIN_ARRAY,
                values=origin_codes,
                policy="refined_instance_key_origin_codes",
            ),
        )
    )
    plan.attrs[instances_path] = {
        "instance_key_status": "present",
        "instance_key_origin_code_map": dict(INSTANCE_KEY_ORIGIN_CODE_MAP),
    }
    if "source_detections" in refined:
        source_detections = refined["source_detections"]
        source_path = f"refined_detect_runs/{refined_name}/source_detections"
        source_keys = _mapped_keys(
            detect_keys,
            _row_array(source_detections, "source_detect_row_index", dtype=np.int64),
            label="refined source_detections",
        )
        plan.arrays.append(
            _planned_existing_or_new(
                source_detections,
                group_path=source_path,
                name="instance_key",
                values=source_keys,
                policy="copied_from_detect",
            )
        )
        plan.attrs[source_path] = {
            "instance_key_status": "present",
            "instance_key_policy": "copied_from_detect",
        }

    crop_name, crop = _selected_run(root, "crop_runs")
    crop_path = f"crop_runs/{crop_name}"
    plan.selected_runs["crop"] = crop_name
    crop_keys, crop_key_counts = _copy_or_mint_crop_keys(
        recording_identity=recording_identity,
        crop_group=crop,
        detect_keys=detect_keys,
    )
    crop_frames = _row_array(crop, "frame_indices", dtype=np.int64)
    plan.arrays.append(
        _planned_existing_or_new(
            crop,
            group_path=crop_path,
            name="instance_key",
            values=crop_keys,
            policy="mixed_legacy_detection_origin_backfill",
        )
    )
    plan.attrs[crop_path] = {
        "instance_key_available": True,
        "instance_key_policy": "mixed_legacy_detection_origin_backfill",
        "instance_key_backfill_counts": crop_key_counts,
        **instance_key_attrs(recording_identity),
    }

    for parent_name, stage_name in (
        ("keypoints_runs", "keypoints"),
        ("refined_keypoints_runs", "refined_keypoints"),
    ):
        run_name, group = _selected_run(root, parent_name)
        group_path = f"{parent_name}/{run_name}"
        plan.selected_runs[stage_name] = run_name
        source_rows = _row_array(group, "source_crop_row_ids", dtype=np.int64)
        target_frames = _row_array(group, "frame_indices", dtype=np.int64)
        _check_frame_mapping(
            source_frames=crop_frames,
            source_rows=source_rows,
            target_frames=target_frames,
            label=stage_name,
        )
        keys = _mapped_keys(crop_keys, source_rows, label=stage_name)
        plan.arrays.append(
            _planned_existing_or_new(
                group,
                group_path=group_path,
                name="instance_key",
                values=keys,
                policy="copied_from_source_crop_row_ids",
            )
        )
        plan.attrs[group_path] = {
            "instance_key_available": True,
            "instance_key_policy": "copied_from_source_crop_row_ids",
        }

    crop_fingerprint = build_rowset_fingerprint(
        source_rowset_path=crop_path,
        row_count=int(crop_keys.shape[0]),
        instance_keys=crop_keys,
        source_edit_revision=resolve_rowset_edit_revision(crop.attrs),
    )
    if "arena_assignment_runs" in root:
        arena_name, arena = _selected_run(root, "arena_assignment_runs")
        arena_path = f"arena_assignment_runs/{arena_name}"
        plan.selected_runs["arena_assignment"] = arena_name
        source_path = str(arena.attrs.get("source_rowset_path") or "")
        if source_path != crop_path:
            raise ValueError(
                f"Selected arena assignment binds {source_path!r}, expected selected crop {crop_path!r}."
            )
        if int(arena["arena_ids"].shape[0]) != int(crop_keys.shape[0]):
            raise ValueError("arena assignment row count does not match its crop source rowset.")
        plan.attrs[arena_path] = crop_fingerprint.to_attrs()

    if "tracking_runs" in root:
        tracking_name, tracking = _selected_run(root, "tracking_runs")
        tracking_path = f"tracking_runs/{tracking_name}"
        plan.selected_runs["tracking"] = tracking_name
        source_path = str(tracking.attrs.get("source_rowset_path") or "")
        if source_path != crop_path:
            raise ValueError(
                f"Selected tracking run binds {source_path!r}, expected selected crop {crop_path!r}."
            )
        source_rows = _row_array(tracking, "source_row_indices", dtype=np.int64)
        target_frames = _row_array(tracking, "frame_indices", dtype=np.int64)
        _check_frame_mapping(
            source_frames=crop_frames,
            source_rows=source_rows,
            target_frames=target_frames,
            label="tracking",
        )
        tracking_keys = _mapped_keys(crop_keys, source_rows, label="tracking")
        plan.arrays.append(
            _planned_existing_or_new(
                tracking,
                group_path=tracking_path,
                name="instance_key",
                values=tracking_keys,
                policy="copied_from_source_row_indices",
            )
        )
        summary = dict(tracking.attrs.get("summary_statistics") or {})
        summary["tracking_identity_mode"] = "instance_key"
        summary["source_rowset_fingerprint_status"] = crop_fingerprint.status
        plan.attrs[tracking_path] = {
            **crop_fingerprint.to_attrs(),
            "tracking_identity_mode": "instance_key",
            "summary_statistics": summary,
        }

    return plan


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_clipped_parent_frame_maps(
    path: Path,
    *,
    requested: set[tuple[str, str]],
) -> dict[tuple[str, str], np.ndarray]:
    required = (
        "camera_serial",
        "clip_id",
        "clip_local_frame_index",
        "parent_frame_index",
    )
    table = pq.read_table(path, columns=list(required))
    missing = [name for name in required if name not in table.column_names]
    if missing:
        raise ValueError(f"Recording frame index is missing columns: {missing}.")

    camera_values = [str(value) for value in table["camera_serial"].to_pylist()]
    clip_values = [str(value) for value in table["clip_id"].to_pylist()]
    local_values = np.asarray(table["clip_local_frame_index"].to_numpy(), dtype=np.int64)
    parent_values = np.asarray(table["parent_frame_index"].to_numpy(), dtype=np.int64)
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = {key: [] for key in requested}
    for camera, clip, local, parent in zip(
        camera_values,
        clip_values,
        local_values.tolist(),
        parent_values.tolist(),
        strict=True,
    ):
        key = (camera, clip)
        if key in grouped:
            grouped[key].append((int(local), int(parent)))

    result: dict[tuple[str, str], np.ndarray] = {}
    for key, rows in grouped.items():
        if not rows:
            raise ValueError(f"Recording frame index has no rows for camera/clip {key}.")
        rows.sort()
        local = np.asarray([row[0] for row in rows], dtype=np.int64)
        parent = np.asarray([row[1] for row in rows], dtype=np.int64)
        if not np.array_equal(local, np.arange(local.shape[0], dtype=np.int64)):
            raise ValueError(f"Recording frame index is not contiguous from zero for {key}.")
        if np.any(parent < 0) or int(np.unique(parent).shape[0]) != int(parent.shape[0]):
            raise ValueError(f"Recording frame index parent frames are invalid for {key}.")
        result[key] = parent
    return result


def _mapped_parent_frames(
    local_frames: np.ndarray,
    parent_frames: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    local = np.asarray(local_frames, dtype=np.int64).reshape(-1)
    if local.size and (int(local.min()) < 0 or int(local.max()) >= int(parent_frames.shape[0])):
        raise ValueError(f"{label} contains clip-local frames outside its frame-index mapping.")
    return np.asarray(parent_frames, dtype=np.int64)[local]


def _keys_for_stable_row_ids(
    source_keys: np.ndarray,
    source_row_ids: np.ndarray,
    requested_row_ids: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    keys = np.asarray(source_keys, dtype=np.uint64).reshape(-1)
    row_ids = np.asarray(source_row_ids, dtype=np.int64).reshape(-1)
    requested = np.asarray(requested_row_ids, dtype=np.int64).reshape(-1)
    if int(row_ids.shape[0]) != int(keys.shape[0]):
        raise ValueError(f"{label} source row IDs do not align with source keys.")
    if int(np.unique(row_ids).shape[0]) != int(row_ids.shape[0]):
        raise ValueError(f"{label} source refined_row_ids are not unique.")
    order = np.argsort(row_ids, kind="stable")
    sorted_ids = row_ids[order]
    positions = np.searchsorted(sorted_ids, requested)
    valid = positions < sorted_ids.shape[0]
    if bool(np.any(valid)):
        valid[valid] &= sorted_ids[positions[valid]] == requested[valid]
    if not bool(np.all(valid)):
        missing = requested[~valid]
        raise ValueError(
            f"{label} references {int(missing.shape[0])} unknown stable refined row IDs, "
            f"e.g. {[int(value) for value in missing[:5].tolist()]}"
        )
    return _require_unique(keys[order[positions]], label=label)


def _build_clipped_plan(root: Any, *, zarr_path: Path) -> InstanceKeyBackfillPlan:
    """Plan stable keys through one finalized clipped-analysis collection."""

    recording_identity = resolve_recording_identity(root.attrs, fallback_path=zarr_path)
    refined_parent = root.get("refined_detect_runs")
    if refined_parent is None:
        raise ValueError("Clipped archive is missing refined_detect_runs.")
    collection_path = str(refined_parent.attrs.get("latest_collection_path") or "").strip("/")
    if not collection_path:
        raise ValueError("Clipped archive has no refined detection latest_collection_path.")
    collection = _group_at(root, collection_path)
    selected = list(collection.attrs.get("selected_runs") or [])
    if not selected:
        raise ValueError(f"Clipped collection {collection_path} has no selected_runs.")

    requested = {
        (str(item["camera_serial"]), str(item["clip_id"]))
        for item in selected
    }
    if len(requested) != len(selected):
        raise ValueError("Clipped collection repeats a camera/clip work unit.")
    frame_index_path = Path(
        str(root.attrs.get("recording_frame_index_path") or root.attrs.get("source_recording_frame_index_path") or "")
    )
    if not frame_index_path.is_file():
        raise ValueError(f"Recording frame index does not exist: {frame_index_path}.")
    frame_mapping_sha256 = _file_sha256(frame_index_path)
    parent_frame_maps = _load_clipped_parent_frame_maps(frame_index_path, requested=requested)

    plan = InstanceKeyBackfillPlan(
        zarr_path=Path(zarr_path),
        recording_identity=recording_identity,
        selected_runs={"refined_detect_collection": collection_path},
    )
    refined_keys_by_work_unit: dict[tuple[str, str], np.ndarray] = {}
    refined_row_ids_by_work_unit: dict[tuple[str, str], np.ndarray] = {}
    raw_keysets: list[np.ndarray] = []
    refined_keysets: list[np.ndarray] = []

    for item in selected:
        camera = str(item["camera_serial"])
        clip = str(item["clip_id"])
        work_unit = (camera, clip)
        parent_frames = parent_frame_maps[work_unit]
        detect_path = str(item["detect_group_path"]).strip("/")
        refined_path = str(item["refined_group_path"]).strip("/")
        detect = _group_at(root, detect_path)
        refined = _group_at(root, refined_path)
        if "instances" not in refined:
            raise ValueError(f"{refined_path} lacks instances.")

        detect_local_frames = _row_array(detect, "frame_indices", dtype=np.int64)
        detect_parent_frames = _mapped_parent_frames(
            detect_local_frames,
            parent_frames,
            label=detect_path,
        )
        detect_keys = _require_unique(
            mint_detection_instance_keys(
                recording_identity=recording_identity,
                frame_indices=detect_parent_frames,
                bbox_norm_coords=_array(detect, "bbox_norm_coords", dtype=np.float64).reshape(-1, 4),
                class_ids=(
                    _row_array(detect, "class_ids", dtype=np.int64)
                    if "class_ids" in detect
                    else None
                ),
            ),
            label=detect_path,
        )
        legacy_recording_identity = str(
            detect.attrs.get("instance_key_recording_identity") or ""
        )
        legacy_frame_domain = str(detect.attrs.get("instance_key_frame_domain") or "")
        legacy_detect_keys: np.ndarray | None = None
        if (
            legacy_recording_identity
            and (
                legacy_recording_identity != recording_identity
                or legacy_frame_domain != "recording_parent_frame_index"
            )
        ):
            legacy_detect_keys = _require_unique(
                mint_detection_instance_keys(
                    recording_identity=legacy_recording_identity,
                    frame_indices=detect_local_frames,
                    bbox_norm_coords=_array(
                        detect, "bbox_norm_coords", dtype=np.float64
                    ).reshape(-1, 4),
                    class_ids=(
                        _row_array(detect, "class_ids", dtype=np.int64)
                        if "class_ids" in detect
                        else None
                    ),
                ),
                label=f"{detect_path} verified legacy keys",
            )
        raw_keysets.append(detect_keys)
        plan.arrays.append(
            _planned_existing_or_new(
                detect,
                group_path=detect_path,
                name="instance_key",
                values=detect_keys,
                policy="minted_at_legacy_clipped_detect_origin",
                shard_rows_override=int(detect.attrs.get("detect_row_shard_rows") or DEFAULT_TABULAR_SHARD_ROWS),
                replace_existing_values=legacy_detect_keys,
            )
        )
        plan.attrs[detect_path] = {
            **instance_key_attrs(recording_identity),
            "instance_key_frame_domain": "recording_parent_frame_index",
            "instance_key_frame_mapping_source": str(frame_index_path),
            "instance_key_frame_mapping_sha256": frame_mapping_sha256,
            "instance_key_status": "present",
            "instance_key_policy": "minted_at_legacy_clipped_detect_origin",
        }

        instances = refined["instances"]
        instances_path = f"{refined_path}/instances"
        instance_local_frames = _row_array(instances, "frame_indices", dtype=np.int64)
        instance_parent_frames = _mapped_parent_frames(
            instance_local_frames,
            parent_frames,
            label=instances_path,
        )
        instance_keys, origin_codes = resolve_curated_instance_keys(
            root,
            zarr_path=zarr_path,
            instance_frame_indices=instance_parent_frames,
            instance_bbox_norm_coords=_array(instances, "bbox_norm_coords", dtype=np.float64).reshape(-1, 4),
            instance_class_ids=(
                _row_array(instances, "class_ids", dtype=np.int64)
                if "class_ids" in instances
                else None
            ),
            instance_source_detect_row_index=_row_array(
                instances, "source_detect_row_index", dtype=np.int64
            ),
            source_detection_instance_key=detect_keys,
        )
        legacy_instance_keys: np.ndarray | None = None
        if legacy_detect_keys is not None:
            legacy_instance_keys, legacy_origin_codes = resolve_curated_instance_keys(
                root,
                zarr_path=zarr_path,
                instance_frame_indices=instance_local_frames,
                instance_bbox_norm_coords=_array(
                    instances, "bbox_norm_coords", dtype=np.float64
                ).reshape(-1, 4),
                instance_class_ids=(
                    _row_array(instances, "class_ids", dtype=np.int64)
                    if "class_ids" in instances
                    else None
                ),
                instance_source_detect_row_index=_row_array(
                    instances, "source_detect_row_index", dtype=np.int64
                ),
                source_detection_instance_key=legacy_detect_keys,
            )
            if not np.array_equal(legacy_origin_codes, origin_codes):
                raise ValueError(f"{instances_path} legacy origin codes changed unexpectedly.")
        refined_keys_by_work_unit[work_unit] = instance_keys
        refined_row_ids_by_work_unit[work_unit] = _row_array(
            instances, "refined_row_ids", dtype=np.int64
        )
        refined_keysets.append(instance_keys)
        plan.arrays.extend(
            (
                _planned_existing_or_new(
                    instances,
                    group_path=instances_path,
                    name="instance_key",
                    values=instance_keys,
                    policy="copied_from_clipped_detect_or_minted_at_curation",
                    shard_rows_override=DEFAULT_TABULAR_SHARD_ROWS,
                    replace_existing_values=legacy_instance_keys,
                ),
                _planned_existing_or_new(
                    instances,
                    group_path=instances_path,
                    name=INSTANCE_KEY_ORIGIN_ARRAY,
                    values=origin_codes,
                    policy="refined_instance_key_origin_codes",
                    shard_rows_override=DEFAULT_TABULAR_SHARD_ROWS,
                ),
            )
        )
        plan.attrs[instances_path] = {
            "instance_key_status": "present",
            "instance_key_origin_code_map": dict(INSTANCE_KEY_ORIGIN_CODE_MAP),
            "instance_key_frame_domain": "recording_parent_frame_index",
        }
        if "source_detections" in refined:
            source = refined["source_detections"]
            source_path = f"{refined_path}/source_detections"
            source_keys = _mapped_keys(
                detect_keys,
                _row_array(source, "source_detect_row_index", dtype=np.int64),
                label=source_path,
            )
            legacy_source_keys = (
                _mapped_keys(
                    legacy_detect_keys,
                    _row_array(source, "source_detect_row_index", dtype=np.int64),
                    label=f"{source_path} legacy",
                )
                if legacy_detect_keys is not None
                else None
            )
            plan.arrays.append(
                _planned_existing_or_new(
                    source,
                    group_path=source_path,
                    name="instance_key",
                    values=source_keys,
                    policy="copied_from_clipped_detect",
                    shard_rows_override=DEFAULT_TABULAR_SHARD_ROWS,
                    replace_existing_values=legacy_source_keys,
                )
            )
            plan.attrs[source_path] = {
                "instance_key_status": "present",
                "instance_key_policy": "copied_from_clipped_detect",
            }

    _require_unique(np.concatenate(raw_keysets), label="clipped raw detection collection")
    _require_unique(np.concatenate(refined_keysets), label="clipped refined detection collection")

    keypoint_name, keypoints = _selected_run(root, "keypoints_runs")
    keypoint_path = f"keypoints_runs/{keypoint_name}"
    plan.selected_runs["keypoints"] = keypoint_name
    merged_crop_name = str(keypoints.attrs.get("source_crop_run") or "")
    if not merged_crop_name:
        raise ValueError(f"{keypoint_path} does not name its source_crop_run.")
    merged_crop_path = f"crop_runs/{merged_crop_name}"
    merged_crop = _group_at(root, merged_crop_path)
    proxy_names = [str(value) for value in merged_crop.attrs.get("source_proxy_crop_runs") or []]
    if not proxy_names:
        raise ValueError(f"{merged_crop_path} has no source_proxy_crop_runs.")
    selected_by_clip = {str(item["clip_id"]): item for item in selected}
    proxy_keys: list[np.ndarray] = []
    proxy_key_by_name: dict[str, np.ndarray] = {}
    for proxy_name in proxy_names:
        proxy_path = f"crop_runs/{proxy_name}"
        proxy = _group_at(root, proxy_path)
        clip = str(proxy.attrs.get("source_clip_id") or "")
        if clip not in selected_by_clip:
            raise ValueError(f"{proxy_path} has unresolved source_clip_id={clip!r}.")
        item = selected_by_clip[clip]
        work_unit = (str(item["camera_serial"]), clip)
        keys = _keys_for_stable_row_ids(
            refined_keys_by_work_unit[work_unit],
            refined_row_ids_by_work_unit[work_unit],
            _row_array(proxy, "source_refined_row_ids", dtype=np.int64),
            label=proxy_path,
        )
        proxy_keys.append(keys)
        proxy_key_by_name[proxy_name] = keys
        plan.arrays.append(
            _planned_existing_or_new(
                proxy,
                group_path=proxy_path,
                name="instance_key",
                values=keys,
                policy="copied_from_clipped_refined_detection",
                shard_rows_override=DEFAULT_TABULAR_SHARD_ROWS,
            )
        )
        plan.attrs[proxy_path] = {
            "instance_key_available": True,
            "instance_key_policy": "copied_from_clipped_refined_detection",
        }

    proxy_run_index = _row_array(merged_crop, "source_proxy_crop_run_index", dtype=np.int64)
    proxy_row_index = _row_array(merged_crop, "source_proxy_crop_row_ids", dtype=np.int64)
    if int(proxy_run_index.shape[0]) != int(proxy_row_index.shape[0]):
        raise ValueError(f"{merged_crop_path} proxy lineage arrays have different lengths.")
    merged_keys = np.empty(proxy_run_index.shape[0], dtype=np.uint64)
    for proxy_index, keys in enumerate(proxy_keys):
        selected_rows = np.flatnonzero(proxy_run_index == proxy_index)
        source_rows = proxy_row_index[selected_rows]
        if source_rows.size and (int(source_rows.min()) < 0 or int(source_rows.max()) >= int(keys.shape[0])):
            raise ValueError(f"{merged_crop_path} has invalid rows for proxy index {proxy_index}.")
        merged_keys[selected_rows] = keys[source_rows]
    if np.any((proxy_run_index < 0) | (proxy_run_index >= len(proxy_keys))):
        raise ValueError(f"{merged_crop_path} contains invalid source proxy indexes.")
    merged_keys = _require_unique(merged_keys, label=merged_crop_path)
    plan.arrays.append(
        _planned_existing_or_new(
            merged_crop,
            group_path=merged_crop_path,
            name="instance_key",
            values=merged_keys,
            policy="copied_from_source_proxy_crop_rows",
            shard_rows_override=DEFAULT_TABULAR_SHARD_ROWS,
        )
    )
    plan.attrs[merged_crop_path] = {
        "instance_key_available": True,
        "instance_key_policy": "copied_from_source_proxy_crop_rows",
    }
    plan.selected_runs["crop"] = merged_crop_name

    for shard_path_value in keypoints.attrs.get("source_keypoint_shard_run_paths") or []:
        shard_path = str(shard_path_value).strip("/")
        shard = _group_at(root, shard_path)
        source_proxy_name = str(shard.attrs.get("source_crop_run") or "")
        if source_proxy_name not in proxy_key_by_name:
            raise ValueError(f"{shard_path} has unresolved source_crop_run={source_proxy_name!r}.")
        keys = _mapped_keys(
            proxy_key_by_name[source_proxy_name],
            _row_array(shard, "source_crop_row_ids", dtype=np.int64),
            label=shard_path,
        )
        plan.arrays.append(
            _planned_existing_or_new(
                shard,
                group_path=shard_path,
                name="instance_key",
                values=keys,
                policy="copied_from_source_proxy_crop_rows",
                shard_rows_override=int(shard.attrs.get("keypoint_roi_shard_rows") or DEFAULT_TABULAR_SHARD_ROWS),
            )
        )
        plan.attrs[shard_path] = {
            "instance_key_available": True,
            "instance_key_policy": "copied_from_source_proxy_crop_rows",
        }

    keypoint_keys = _mapped_keys(
        merged_keys,
        _row_array(keypoints, "source_crop_row_ids", dtype=np.int64),
        label=keypoint_path,
    )
    plan.arrays.append(
        _planned_existing_or_new(
            keypoints,
            group_path=keypoint_path,
            name="instance_key",
            values=keypoint_keys,
            policy="copied_from_merged_proxy_crop_rows",
            shard_rows_override=DEFAULT_TABULAR_SHARD_ROWS,
        )
    )
    plan.attrs[keypoint_path] = {
        "instance_key_available": True,
        "instance_key_policy": "copied_from_merged_proxy_crop_rows",
    }

    refined_keypoint_name, refined_keypoints = _selected_run(root, "refined_keypoints_runs")
    refined_keypoint_path = f"refined_keypoints_runs/{refined_keypoint_name}"
    plan.selected_runs["refined_keypoints"] = refined_keypoint_name
    refined_keypoint_keys = _mapped_keys(
        merged_keys,
        _row_array(refined_keypoints, "source_crop_row_ids", dtype=np.int64),
        label=refined_keypoint_path,
    )
    plan.arrays.append(
        _planned_existing_or_new(
            refined_keypoints,
            group_path=refined_keypoint_path,
            name="instance_key",
            values=refined_keypoint_keys,
            policy="copied_from_merged_proxy_crop_rows",
            shard_rows_override=DEFAULT_TABULAR_SHARD_ROWS,
        )
    )
    plan.attrs[refined_keypoint_path] = {
        "instance_key_available": True,
        "instance_key_policy": "copied_from_merged_proxy_crop_rows",
    }
    return plan


def _uses_clipped_collection_lineage(root: Any) -> bool:
    """Resolve clipped shells from either modern or historical metadata.

    Historical finalized clipped archives can predate the root-level
    ``analysis_layout=clipped_recording_shell`` marker. The finalized
    collection pointer is itself an unambiguous structural contract and must
    take precedence over unrelated legacy root ``detect_runs`` groups.
    """

    if str(root.attrs.get("analysis_layout") or "") == "clipped_recording_shell":
        return True
    refined_parent = root.get("refined_detect_runs")
    if refined_parent is None:
        return False
    return bool(str(refined_parent.attrs.get("latest_collection_path") or "").strip("/"))


def build_plan(root: Any, *, zarr_path: Path) -> InstanceKeyBackfillPlan:
    """Build and fully validate an additive migration plan for one archive."""

    if _uses_clipped_collection_lineage(root):
        return _build_clipped_plan(root, zarr_path=zarr_path)
    return _build_dense_plan(root, zarr_path=zarr_path)


def _group_at(root: Any, group_path: str) -> Any:
    group = root
    for part in group_path.split("/"):
        group = group[part]
    return group


def _atomic_add_array(
    *,
    zarr_path: Path,
    group_path: str,
    name: str,
    values: np.ndarray,
    chunk_rows: int,
    shard_rows: int | None,
    replace_existing: bool = False,
) -> str:
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    group = _group_at(root, group_path)
    if name in group:
        existing = np.asarray(group[name][:], dtype=values.dtype)
        if np.array_equal(existing, values):
            return "verified_existing"
        if not replace_existing:
            raise ValueError(f"Existing {group_path}/{name} differs from migration payload.")

    temp_name = f"_{name}_backfill_{uuid4().hex}"
    rows = int(values.shape[0])
    chunks = (max(1, min(int(chunk_rows), max(1, rows))),)
    create_kwargs: dict[str, Any] = {"chunks": chunks, "overwrite": False}
    if shard_rows is not None:
        effective_shard_rows = max(
            chunks[0],
            ((int(shard_rows) + chunks[0] - 1) // chunks[0]) * chunks[0],
        )
        create_kwargs["shards"] = (effective_shard_rows,)
    group.create_array(temp_name, data=values, **create_kwargs)
    written = np.asarray(group[temp_name][:], dtype=values.dtype)
    if not np.array_equal(written, values):
        raise ValueError(f"Reread validation failed for temporary {group_path}/{temp_name}.")
    del group
    del root

    group_dir = zarr_path.joinpath(*group_path.split("/"))
    temp_dir = group_dir / temp_name
    destination = group_dir / name
    backup: Path | None = None
    if destination.exists():
        if not replace_existing:
            raise FileExistsError(f"Destination appeared during migration: {destination}")
        backup = group_dir / f"_{name}_verified_legacy_{uuid4().hex}"
        os.replace(destination, backup)
    try:
        os.replace(temp_dir, destination)
        verify_root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
        verify_group = _group_at(verify_root, group_path)
        reread = np.asarray(verify_group[name][:], dtype=values.dtype)
        if not np.array_equal(reread, values):
            raise ValueError(f"Published array validation failed for {group_path}/{name}.")
    except Exception:
        if backup is not None and backup.exists():
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(backup, destination)
        raise
    if backup is not None:
        shutil.rmtree(backup)
        return "replaced_verified_legacy"
    return "written"


def apply_plan(plan: InstanceKeyBackfillPlan, *, chunk_rows: int = DEFAULT_CHUNK_ROWS) -> dict[str, Any]:
    """Apply a validated plan using same-directory temporary array publication."""

    if int(chunk_rows) <= 0:
        raise ValueError("chunk_rows must be positive.")
    started_at = datetime.now(timezone.utc).isoformat()
    root = zarr.open_group(str(plan.zarr_path), mode="a", use_consolidated=False)
    root.attrs.update(
        {
            "instance_key_backfill_status": "in_progress",
            "instance_key_backfill_migration_id": MIGRATION_ID,
            "instance_key_backfill_started_at_utc": started_at,
        }
    )
    del root

    outcomes: dict[str, str] = {}
    try:
        for item in plan.arrays:
            key = f"{item.group_path}/{item.name}"
            outcomes[key] = _atomic_add_array(
                zarr_path=plan.zarr_path,
                group_path=item.group_path,
                name=item.name,
                values=item.values,
                chunk_rows=chunk_rows,
                shard_rows=item.shard_rows,
                replace_existing=item.action == "replace_verified_legacy",
            )

        completed_at = datetime.now(timezone.utc).isoformat()
        root = zarr.open_group(str(plan.zarr_path), mode="a", use_consolidated=False)
        migration_attrs = {
            "instance_key_backfill_status": "complete",
            "instance_key_backfill_migration_id": MIGRATION_ID,
            "instance_key_backfill_tool": MIGRATION_TOOL,
            "instance_key_backfill_recording_identity": plan.recording_identity,
            "instance_key_backfill_completed_at_utc": completed_at,
        }
        for group_path, attrs in plan.attrs.items():
            group = _group_at(root, group_path)
            group.attrs.update({**attrs, **migration_attrs})
        root.attrs.update(
            {
                **migration_attrs,
                "instance_key_backfill_selected_runs": dict(plan.selected_runs),
            }
        )
        return {
            **plan.summary(),
            "status": "complete",
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "outcomes": outcomes,
        }
    except Exception:
        root = zarr.open_group(str(plan.zarr_path), mode="a", use_consolidated=False)
        root.attrs.update(
            {
                "instance_key_backfill_status": "error",
                "instance_key_backfill_migration_id": MIGRATION_ID,
                "instance_key_backfill_failed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        raise


def validate_applied_plan(plan: InstanceKeyBackfillPlan) -> dict[str, Any]:
    root = zarr.open_group(str(plan.zarr_path), mode="r", use_consolidated=False)
    mismatches: list[str] = []
    for item in plan.arrays:
        group = _group_at(root, item.group_path)
        if item.name not in group:
            mismatches.append(f"missing:{item.group_path}/{item.name}")
            continue
        values = np.asarray(group[item.name][:], dtype=item.values.dtype)
        if not np.array_equal(values, item.values):
            mismatches.append(f"mismatch:{item.group_path}/{item.name}")
    status = str(root.attrs.get("instance_key_backfill_status") or "")
    if status != "complete":
        mismatches.append(f"root_status:{status or '<missing>'}")
    return {
        "zarr_path": str(plan.zarr_path),
        "status": "ok" if not mismatches else "error",
        "arrays_checked": len(plan.arrays),
        "mismatches": mismatches,
    }


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_paths", nargs="+", type=Path)
    parser.add_argument("--apply", action="store_true", help="Apply the additive migration (default: dry-run).")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    reports: list[dict[str, Any]] = []
    for raw_path in args.zarr_paths:
        path = raw_path.expanduser().resolve()
        root = zarr.open_group(str(path), mode="r", use_consolidated=False)
        plan = build_plan(root, zarr_path=path)
        if args.apply:
            report = apply_plan(plan, chunk_rows=int(args.chunk_rows))
            validation_root = zarr.open_group(str(path), mode="r", use_consolidated=False)
            validation_plan = build_plan(validation_root, zarr_path=path)
            report["validation"] = validate_applied_plan(validation_plan)
        else:
            report = {**plan.summary(), "status": "dry_run"}
        reports.append(report)
        print(json.dumps(report, allow_nan=False, sort_keys=True))

    payload = {
        "migration_id": MIGRATION_ID,
        "apply": bool(args.apply),
        "archives": reports,
        "archives_ok": sum(
            1
            for report in reports
            if report.get("status") in {"dry_run", "complete"}
            and (not args.apply or report.get("validation", {}).get("status") == "ok")
        ),
    }
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
