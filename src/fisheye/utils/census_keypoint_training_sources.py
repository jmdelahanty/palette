#!/usr/bin/env python3
"""Read-only census of reviewed keypoint-training source Zarrs.

This diagnostic distinguishes a registry row from a source that can safely
participate in one exact-skeleton merged training artifact.  It deliberately
opens source archives with direct metadata because historical training Zarrs
may still be editable and may have stale consolidated metadata.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from fisheye.registry.db import RegistryPaths
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.training_leakage_groups import resolve_training_leakage_group
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.utils.export_keypoint_training_zarr import _resolve_roi_pixel_contract

CENSUS_SCHEMA_ID = "palette.keypoint_training_source_census"
CENSUS_SCHEMA_VERSION = 1
DEFAULT_SKELETON_ID = "pose_skel_traditional_v2"
DEFAULT_KEYPOINT_LABELS = (
    "swim_bladder",
    "eye_left",
    "eye_right",
    "snout_tip",
    "tail_tip",
)


def _readonly_connection(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve()
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _source_mount(path: str) -> str:
    if path.startswith("/nvme1/"):
        return "/nvme1"
    if path.startswith("/groups/"):
        return "/groups"
    return "other"


_leakage_group_id = resolve_training_leakage_group


def _frame_domain(
    historical: np.ndarray,
    *,
    local: np.ndarray,
    acquisition: np.ndarray,
) -> str:
    if np.array_equal(historical, local):
        return "source_sample_row"
    if np.array_equal(historical, acquisition):
        return "source_acquisition_frame"
    return "mismatch"


def _stable_array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        normalized = np.ascontiguousarray(np.asarray(array))
        digest.update(str(normalized.dtype).encode("ascii"))
        digest.update(
            json.dumps(list(normalized.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _subject_ids(connection: sqlite3.Connection, recording_id: str) -> tuple[str, ...]:
    return tuple(
        str(row["subject_id"])
        for row in connection.execute(
            "SELECT subject_id FROM recording_subjects WHERE recording_id = ? ORDER BY subject_id",
            (recording_id,),
        )
    )


def _recording_started_utc(
    connection: sqlite3.Connection, recording_id: str
) -> str | None:
    row = connection.execute(
        "SELECT started_utc FROM recordings WHERE recording_id = ?",
        (recording_id,),
    ).fetchone()
    if row is None or row["started_utc"] is None:
        return None
    value = str(row["started_utc"]).strip()
    return value or None


def _performance_contract(
    connection: sqlite3.Connection,
    *,
    dataset_id: str,
    keypoint_run: str | None,
) -> dict[str, Any]:
    if not keypoint_run:
        return {}
    row = connection.execute(
        """
        SELECT
            source_roi_pixel_contract_name,
            source_roi_image_representation,
            source_roi_read_mode,
            source_crop_storage_mode,
            input_mode_effective,
            updated_utc
        FROM keypoint_performance
        WHERE dataset_id = ? AND keypoint_run = ?
        ORDER BY COALESCE(keypoint_created_utc, updated_utc) DESC
        LIMIT 1
        """,
        (dataset_id, keypoint_run),
    ).fetchone()
    return dict(row) if row is not None else {}


def _query_reviewed_rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(connection.execute("""
            SELECT
                q.dataset_id,
                d.recording_id,
                d.zarr_path,
                q.refined_run,
                q.source_keypoint_run,
                q.keypoint_method,
                q.review_state,
                q.review_intended_use,
                q.review_method,
                q.review_reviewer,
                q.review_policy_id,
                q.review_policy_version,
                q.review_timestamp_utc,
                q.usable_keypoints,
                q.total_keypoints
            FROM keypoint_quality_current q
            JOIN datasets d ON d.dataset_id = q.dataset_id
            WHERE q.review_state = 'approved'
              AND q.review_intended_use = 'training'
            ORDER BY d.recording_id, q.dataset_id
            """))


def _reviewed_artifact_row(path: Path) -> dict[str, Any]:
    """Project one immutable reviewed artifact into the registry-row shape."""

    resolved = path.expanduser().resolve()
    root = open_zarr_group_direct(resolved, mode="r")
    publication = root.attrs.get("reviewed_keypoint_training_artifact")
    if not isinstance(publication, Mapping):
        raise ValueError(f"Reviewed keypoint artifact envelope is missing: {resolved}")
    if (
        publication.get("schema_id") != "palette.reviewed_keypoint_training_artifact"
        or publication.get("schema_version") != 1
    ):
        raise ValueError(
            f"Reviewed keypoint artifact envelope is unsupported: {resolved}"
        )
    payload = publication.get("payload")
    included = (
        payload.get("included_run_paths") if isinstance(payload, Mapping) else None
    )
    if not isinstance(included, Mapping) or set(included) != {
        "crop",
        "raw_keypoints",
        "keypoint_quality",
        "refined_keypoints",
    }:
        raise ValueError(
            f"Reviewed keypoint artifact run roles are invalid: {resolved}"
        )
    refined_path = str(included["refined_keypoints"])
    raw_path = str(included["raw_keypoints"])
    if not refined_path.startswith(
        "refined_keypoints_runs/"
    ) or not raw_path.startswith("keypoints_runs/"):
        raise ValueError(
            f"Reviewed keypoint artifact run paths are invalid: {resolved}"
        )
    refined = root[refined_path]
    usable = np.asarray(refined["usable_keypoints"][:], dtype=np.bool_)
    recording_id = str(
        root.attrs.get("recording_id") or root.attrs.get("session_uuid") or ""
    )
    if not recording_id:
        raise ValueError(
            f"Reviewed keypoint artifact recording identity is missing: {resolved}"
        )
    payload_digest = str(publication.get("payload_digest") or "")
    return {
        "dataset_id": f"{recording_id}:reviewed_keypoints:{payload_digest[:12]}",
        "recording_id": recording_id,
        "zarr_path": str(resolved),
        "refined_run": refined_path.split("/", 1)[1],
        "source_keypoint_run": raw_path.split("/", 1)[1],
        "keypoint_method": "reviewed_keypoint_artifact",
        "review_state": "approved",
        "review_intended_use": "training",
        "review_method": "manual_web_review_compaction",
        "review_reviewer": None,
        "review_policy_id": "immutable_reviewed_keypoint_artifact_v1",
        "review_policy_version": 1,
        "review_timestamp_utc": (
            payload.get("created_at_utc") if isinstance(payload, Mapping) else None
        ),
        "usable_keypoints": int(usable.sum()),
        "total_keypoints": int(usable.shape[0]),
        "source_origin": "reviewed_artifact_candidate",
        "reviewed_artifact_payload_digest": payload_digest,
    }


def _int_array(group: Any, path: str, *, dtype: Any = np.int64) -> np.ndarray:
    return np.asarray(group[path][:], dtype=dtype)


def _source_record(
    row: Mapping[str, Any],
    *,
    connection: sqlite3.Connection,
    skeleton_id: str,
    keypoint_count: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    dataset_id = str(row["dataset_id"])
    recording_id = str(row["recording_id"])
    zarr_path = Path(str(row["zarr_path"]))
    refined_run = str(row["refined_run"])
    root = open_zarr_group_direct(zarr_path, mode="r")
    refined_path = f"refined_keypoints_runs/{refined_run}"
    if refined_path not in root:
        raise ValueError(f"Missing reviewed refined run: {zarr_path}:{refined_path}")
    refined = root[refined_path]
    keypoints_array = refined["keypoints_roi"]
    shape = tuple(int(value) for value in keypoints_array.shape)
    if len(shape) != 3:
        raise ValueError(f"{zarr_path}:{refined_path}/keypoints_roi is not rank 3")

    attrs = dict(refined.attrs)
    source_bindings = attrs.get("source_bindings")
    modern_bindings = source_bindings if isinstance(source_bindings, Mapping) else {}
    skeleton_binding = modern_bindings.get("skeleton")
    if not isinstance(skeleton_binding, Mapping):
        skeleton_binding = {}
    skeleton_semantics = skeleton_binding.get("semantics")
    if not isinstance(skeleton_semantics, Mapping):
        skeleton_semantics = {}
    pose_schema = attrs.get("pose_schema")
    if not isinstance(pose_schema, Mapping):
        pose_schema = skeleton_semantics
    labels = tuple(str(value) for value in pose_schema.get("keypoint_labels", ()))
    effective_skeleton = str(
        attrs.get("skeleton_id")
        or skeleton_binding.get("skeleton_id")
        or pose_schema.get("skeleton_id")
        or ""
    )
    target_shape = (keypoint_count, 2)
    selected = shape[1:] == target_shape and effective_skeleton == skeleton_id

    base_record: dict[str, Any] = {
        "dataset_id": dataset_id,
        "recording_id": recording_id,
        "zarr_path": str(zarr_path),
        "storage_mount": _source_mount(str(zarr_path)),
        "source_origin": row.get("source_origin", "registry_approved"),
        "refined_run": refined_run,
        "source_keypoint_run": row["source_keypoint_run"],
        "keypoint_method": row["keypoint_method"],
        "keypoints_shape": list(shape),
        "keypoints_dtype": str(keypoints_array.dtype),
        "skeleton_id": effective_skeleton or None,
        "keypoint_labels": list(labels),
        "coordinate_system": (
            pose_schema.get("metadata", {}).get("coordinate_system")
            if isinstance(pose_schema.get("metadata"), Mapping)
            else ("roi" if modern_bindings else None)
        ),
        "units": (
            pose_schema.get("metadata", {}).get("units")
            if isinstance(pose_schema.get("metadata"), Mapping)
            else ("pixels" if modern_bindings else None)
        ),
        "selected_for_target_skeleton": bool(selected),
        "registry_row_counts": {
            "usable": int(row["usable_keypoints"] or 0),
            "total": int(row["total_keypoints"] or 0),
        },
        "review": {
            "state": row["review_state"],
            "intended_use": row["review_intended_use"],
            "method": row["review_method"],
            "reviewer": row["review_reviewer"],
            "policy_id": row["review_policy_id"],
            "policy_version": row["review_policy_version"],
            "timestamp_utc": row["review_timestamp_utc"],
            "reviewed_artifact_payload_digest": row.get(
                "reviewed_artifact_payload_digest"
            ),
        },
    }
    if not selected:
        return base_record, {}

    expected_labels = DEFAULT_KEYPOINT_LABELS if keypoint_count == 5 else labels
    problems: list[str] = []
    if labels != expected_labels:
        problems.append("keypoint_labels_mismatch")
    if base_record["coordinate_system"] != "roi":
        problems.append("coordinate_system_not_roi")

    crop_snapshot = modern_bindings.get("crop_snapshot")
    if not isinstance(crop_snapshot, Mapping):
        crop_snapshot = {}
    crop_run = str(attrs.get("source_crop_run") or crop_snapshot.get("run_id") or "")
    crop_path = f"crop_runs/{crop_run}"
    if not crop_run or crop_path not in root:
        raise ValueError(
            f"{zarr_path}:{refined_path} has no resolvable source crop run"
        )
    crop = root[crop_path]
    crop_attrs = dict(crop.attrs)
    roi_images = crop["roi_images"]
    roi_shape = tuple(int(value) for value in roi_images.shape)
    if len(roi_shape) != 3:
        problems.append("roi_images_not_rank_3")

    performance = _performance_contract(
        connection,
        dataset_id=dataset_id,
        keypoint_run=(
            str(row["source_keypoint_run"]) if row["source_keypoint_run"] else None
        ),
    )
    resolved_contract_document, resolved_crop_contract = _resolve_roi_pixel_contract(
        crop
    )
    direct_contract = crop_attrs.get("roi_pixel_contract_name")
    registry_contract = performance.get("source_roi_pixel_contract_name")
    if direct_contract:
        resolved_contract = str(direct_contract)
        contract_source = "crop_run_attribute"
        if registry_contract and str(registry_contract) != resolved_contract:
            problems.append("pixel_contract_registry_disagrees")
    elif resolved_crop_contract:
        resolved_contract = str(resolved_crop_contract)
        contract_source = (
            "crop_run_contract_document"
            if isinstance(crop_attrs.get("roi_pixel_contract"), Mapping)
            else "training_crop_materialization_binding"
        )
    elif registry_contract:
        resolved_contract = str(registry_contract)
        contract_source = "registry_performance_fallback"
    else:
        resolved_contract = None
        contract_source = "missing"
        problems.append("pixel_contract_missing")

    usable = np.asarray(refined["usable_keypoints"][:], dtype=np.bool_)
    keypoints = np.asarray(keypoints_array[:])
    frame_local = _int_array(refined, "frame_indices")
    if "source_crop_row_ids" in refined:
        roi_index = _int_array(refined, "source_crop_row_ids")
        roi_index_path = f"{refined_path}/source_crop_row_ids"
    elif "detection_indices" in refined:
        roi_index = _int_array(refined, "detection_indices")
        roi_index_path = f"{refined_path}/detection_indices"
    else:
        roi_index = np.arange(shape[0], dtype=np.int64)
        roi_index_path = "implicit_row_order"
    for name, array in (
        ("usable_keypoints", usable),
        ("frame_indices", frame_local),
        ("detection_indices", roi_index),
    ):
        if array.shape != (shape[0],):
            raise ValueError(f"{zarr_path}:{refined_path}/{name} is not length N")

    direct_acquisition_path = f"{refined_path}/source_acquisition_frame_index"
    original_frame_path = "raw_video/original_frame_indices"
    if "source_acquisition_frame_index" in refined:
        acquisition_frame = _int_array(refined, "source_acquisition_frame_index")
        if acquisition_frame.shape != (shape[0],):
            problems.append("source_acquisition_frame_index_length_mismatch")
            acquisition_frame = frame_local.copy()
            acquisition_source = "unresolved_source_acquisition_frame_index"
        else:
            acquisition_source = direct_acquisition_path
    elif original_frame_path in root:
        original_frames = _int_array(root, original_frame_path)
        if frame_local.size and (
            int(frame_local.min()) < 0
            or int(frame_local.max()) >= int(original_frames.shape[0])
        ):
            problems.append("frame_indices_outside_original_frame_map")
            acquisition_frame = frame_local.copy()
            acquisition_source = "unresolved_frame_indices"
        else:
            acquisition_frame = original_frames[frame_local]
            acquisition_source = original_frame_path
    else:
        acquisition_frame = frame_local.copy()
        acquisition_source = "frame_indices_fallback"
        problems.append("original_frame_map_missing")

    if roi_index.size and (
        int(roi_index.min()) < 0 or int(roi_index.max()) >= int(roi_shape[0])
    ):
        problems.append("roi_index_out_of_bounds")
        source_detect = np.full(shape[0], -1, dtype=np.int64)
        source_refined = np.full(shape[0], -1, dtype=np.int64)
    else:
        source_detect = (
            _int_array(crop, "source_detect_row_index")[roi_index]
            if "source_detect_row_index" in crop
            else np.full(shape[0], -1, dtype=np.int64)
        )
        source_refined = (
            _int_array(crop, "source_refined_row_ids")[roi_index]
            if "source_refined_row_ids" in crop
            else np.full(shape[0], -1, dtype=np.int64)
        )

    usable_keypoints = keypoints[usable]
    finite_rows = (
        np.isfinite(usable_keypoints).all(axis=(1, 2))
        if usable_keypoints.size
        else np.ones(0, dtype=np.bool_)
    )
    roi_height = int(roi_shape[1])
    roi_width = int(roi_shape[2])
    in_bounds_rows = (
        (
            (usable_keypoints[:, :, 0] >= 0)
            & (usable_keypoints[:, :, 0] < roi_width)
            & (usable_keypoints[:, :, 1] >= 0)
            & (usable_keypoints[:, :, 1] < roi_height)
        ).all(axis=1)
        if usable_keypoints.size
        else np.ones(0, dtype=np.bool_)
    )
    if not bool(finite_rows.all()):
        problems.append("usable_keypoints_nonfinite")
    if not bool(in_bounds_rows.all()):
        problems.append("usable_keypoints_out_of_roi_bounds")
    if int(usable.sum()) != int(row["usable_keypoints"] or 0):
        problems.append("registry_usable_count_mismatch")
    if int(shape[0]) != int(row["total_keypoints"] or 0):
        problems.append("registry_total_count_mismatch")
    if str(keypoints_array.dtype) not in {"float32", "float64"}:
        problems.append("unsupported_keypoint_dtype")
    if str(roi_images.dtype) != "uint8":
        problems.append("roi_images_not_uint8")

    subject_ids = _subject_ids(connection, recording_id)
    started_utc = _recording_started_utc(connection, recording_id)
    leakage_group_id, leakage_group_source = _leakage_group_id(
        recording_id=recording_id,
        subject_ids=subject_ids,
        started_utc=started_utc,
    )
    selected_detect = source_detect[usable]
    selected_refined = source_refined[usable]
    selected_local = frame_local[usable]
    selected_acquisition = acquisition_frame[usable]
    selected_roi = roi_index[usable]

    base_record.update(
        {
            "source_crop_run": crop_run,
            "crop_storage_mode": crop_attrs.get("crop_storage_mode"),
            "roi_images_shape": list(roi_shape),
            "roi_images_dtype": str(roi_images.dtype),
            "pixel_contract": {
                "resolved_name": resolved_contract,
                "source": contract_source,
                "direct_name": direct_contract,
                "registry_name": registry_contract,
                "image_representation": (
                    resolved_contract_document.get("image_representation")
                    if isinstance(resolved_contract_document, Mapping)
                    else performance.get("source_roi_image_representation")
                ),
                "padding": (
                    resolved_contract_document.get("padding")
                    if isinstance(resolved_contract_document, Mapping)
                    else None
                ),
            },
            "frame_identity": {
                "local_path": f"{refined_path}/frame_indices",
                "acquisition_source": acquisition_source,
                "source_crop_row_path": roi_index_path,
                "usable_identity_sha256": _stable_array_digest(
                    selected_acquisition,
                    selected_roi,
                    selected_detect,
                    selected_refined,
                ),
                "repeated_acquisition_frame_rows": int(
                    selected_acquisition.size - np.unique(selected_acquisition).size
                ),
            },
            "lineage": {
                "usable_source_detect_rows_missing": int(
                    np.count_nonzero(selected_detect < 0)
                ),
                "usable_source_refined_rows_missing": int(
                    np.count_nonzero(selected_refined < 0)
                ),
                "usable_rows_without_detection_or_refined_identity": int(
                    np.count_nonzero((selected_detect < 0) & (selected_refined < 0))
                ),
                "crop_backfill": crop_attrs.get("row_lineage_backfill"),
            },
            "split_group": {
                "id": leakage_group_id,
                "source": leakage_group_source,
                "subject_ids": list(subject_ids),
                "recording_started_utc": started_utc,
            },
            "rows": {
                "total": int(shape[0]),
                "usable": int(usable.sum()),
                "unusable": int(shape[0] - int(usable.sum())),
                "usable_finite": int(finite_rows.sum()),
                "usable_in_bounds": int(in_bounds_rows.sum()),
            },
            "problems": sorted(set(problems)),
            "safe_for_pose_merge": not problems,
            "exact_detection_or_refined_lineage_complete": bool(
                np.all((selected_detect >= 0) | (selected_refined >= 0))
            ),
        }
    )
    internal = {
        "frame_local": selected_local,
        "frame_acquisition": selected_acquisition,
        "roi_index": selected_roi,
        "source_detect_row_index": selected_detect,
        "source_refined_row_ids": selected_refined,
    }
    return base_record, internal


def _historical_comparison(
    historical_path: Path,
    *,
    sources: Sequence[Mapping[str, Any]],
    internals: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    root = open_zarr_group_direct(historical_path, mode="r")
    export = root.attrs.get("training_export")
    if not isinstance(export, Mapping):
        raise ValueError("Historical merge is missing training_export metadata")
    source_ids = [str(value) for value in export.get("source_dataset_ids", ())]
    dataset_index = _int_array(root, "source_index/source_dataset_idx")
    source_arrays = {
        "source_frame_idx": _int_array(root, "source_index/source_frame_idx"),
        "source_roi_idx": _int_array(root, "source_index/source_roi_idx"),
        "source_detect_row_index": _int_array(
            root, "source_index/source_detect_row_index"
        ),
        "source_refined_row_ids": _int_array(
            root, "source_index/source_refined_row_ids"
        ),
    }
    old_recordings = [dataset_id.split(":", 1)[0] for dataset_id in source_ids]
    old_counts: dict[str, int] = {}
    frame_domain_counts: Counter[str] = Counter()
    lineage_mismatches: list[dict[str, Any]] = []
    for ordinal, recording_id in enumerate(old_recordings):
        mask = dataset_index == ordinal
        old_counts[recording_id] = int(np.count_nonzero(mask))
        current = internals.get(recording_id)
        if current is None:
            continue
        frame_domain_counts[
            _frame_domain(
                source_arrays["source_frame_idx"][mask],
                local=current["frame_local"],
                acquisition=current["frame_acquisition"],
            )
        ] += 1
        for old_name, current_name in (
            ("source_roi_idx", "roi_index"),
            ("source_detect_row_index", "source_detect_row_index"),
            ("source_refined_row_ids", "source_refined_row_ids"),
        ):
            old_values = source_arrays[old_name][mask]
            current_values = current[current_name]
            if not np.array_equal(old_values, current_values):
                lineage_mismatches.append(
                    {
                        "recording_id": recording_id,
                        "array": old_name,
                        "old_rows": int(old_values.shape[0]),
                        "current_rows": int(current_values.shape[0]),
                    }
                )

    current_counts = {
        str(source["recording_id"]): int(source["rows"]["usable"]) for source in sources
    }
    common = set(old_counts) & set(current_counts)
    row_count_mismatches = {
        recording_id: {
            "historical": old_counts[recording_id],
            "current": current_counts[recording_id],
        }
        for recording_id in sorted(common)
        if old_counts[recording_id] != current_counts[recording_id]
    }

    train = _int_array(root, "splits/train_indices")
    validation = _int_array(root, "splits/val_indices")
    train_sources = set(dataset_index[train].tolist())
    validation_sources = set(dataset_index[validation].tolist())
    split_overlap = train_sources & validation_sources
    split = export.get("split") if isinstance(export.get("split"), Mapping) else {}
    return {
        "path": str(historical_path),
        "source_recordings": len(old_counts),
        "rows": int(dataset_index.shape[0]),
        "current_common_recordings": len(common),
        "current_only_recordings": {
            key: current_counts[key]
            for key in sorted(set(current_counts) - set(old_counts))
        },
        "historical_only_recordings": {
            key: old_counts[key]
            for key in sorted(set(old_counts) - set(current_counts))
        },
        "row_count_mismatches": row_count_mismatches,
        "common_historical_rows": int(sum(old_counts[key] for key in common)),
        "common_current_rows": int(sum(current_counts[key] for key in common)),
        "lineage_array_mismatches": lineage_mismatches,
        "source_frame_idx_domain_counts": dict(sorted(frame_domain_counts.items())),
        "source_frame_idx_semantics_are_uniform": len(frame_domain_counts) <= 1,
        "split": {
            "declared": dict(split),
            "train_source_count": len(train_sources),
            "validation_source_count": len(validation_sources),
            "source_overlap_count": len(split_overlap),
            "source_overlap_fraction": (
                float(len(split_overlap) / len(set(dataset_index.tolist())))
                if dataset_index.size
                else 0.0
            ),
            "leakage_safe": not split_overlap,
        },
    }


def _count(sources: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(source[key]) for source in sources).items()))


def build_census(
    registry_path: Path,
    *,
    skeleton_id: str = DEFAULT_SKELETON_ID,
    keypoint_count: int = 5,
    historical_merge: Path | None = None,
    reviewed_artifacts: Sequence[Path] = (),
) -> dict[str, Any]:
    connection = _readonly_connection(registry_path)
    try:
        source_records: list[dict[str, Any]] = []
        internals: dict[str, dict[str, np.ndarray]] = {}
        excluded: list[dict[str, Any]] = []
        rows: list[Mapping[str, Any]] = []
        for row in _query_reviewed_rows(connection):
            value = dict(row)
            value["source_origin"] = "registry_approved"
            rows.append(value)
        rows.extend(_reviewed_artifact_row(path) for path in reviewed_artifacts)
        seen_paths: set[str] = set()
        for row in rows:
            zarr_path = str(row["zarr_path"])
            if zarr_path in seen_paths:
                raise ValueError(f"Duplicate source Zarr in census inputs: {zarr_path}")
            seen_paths.add(zarr_path)
            record, internal = _source_record(
                row,
                connection=connection,
                skeleton_id=skeleton_id,
                keypoint_count=keypoint_count,
            )
            if record["selected_for_target_skeleton"]:
                source_records.append(record)
                internals[str(record["recording_id"])] = internal
            else:
                excluded.append(record)
    finally:
        connection.close()

    source_records.sort(
        key=lambda item: (str(item["recording_id"]), str(item["dataset_id"]))
    )
    identity_digests = [
        source["frame_identity"]["usable_identity_sha256"] for source in source_records
    ]
    duplicate_digest_count = len(identity_digests) - len(set(identity_digests))
    row_identities: Counter[tuple[Any, ...]] = Counter()
    for source in source_records:
        recording_id = str(source["recording_id"])
        internal = internals[recording_id]
        for acquisition, roi_index, detect_row, refined_row in zip(
            internal["frame_acquisition"].tolist(),
            internal["roi_index"].tolist(),
            internal["source_detect_row_index"].tolist(),
            internal["source_refined_row_ids"].tolist(),
        ):
            if int(refined_row) >= 0:
                identity = (recording_id, "refined", int(refined_row))
            elif int(detect_row) >= 0:
                identity = (recording_id, "detect", int(detect_row))
            else:
                identity = (
                    recording_id,
                    "sample",
                    int(acquisition),
                    int(roi_index),
                )
            row_identities[identity] += 1
    duplicate_row_identities = {
        "|".join(str(value) for value in identity): count
        for identity, count in row_identities.items()
        if count > 1
    }
    split_groups = Counter(source["split_group"]["id"] for source in source_records)
    problems = Counter(
        problem for source in source_records for problem in source["problems"]
    )
    source_composition = [
        {
            "dataset_id": source["dataset_id"],
            "recording_id": source["recording_id"],
            "zarr_path": source["zarr_path"],
            "refined_run": source["refined_run"],
            "usable_rows": source["rows"]["usable"],
            "identity_sha256": source["frame_identity"]["usable_identity_sha256"],
            "pixel_contract": source["pixel_contract"]["resolved_name"],
            "split_group_id": source["split_group"]["id"],
        }
        for source in source_records
    ]
    summary = {
        "source_count": len(source_records),
        "source_composition_sha256": _canonical_digest(source_composition),
        "excluded_other_skeleton_count": len(excluded),
        "excluded_other_skeleton_usable_rows": int(
            sum(source["registry_row_counts"]["usable"] for source in excluded)
        ),
        "excluded_shape_counts": dict(
            sorted(
                Counter(
                    str(tuple(source["keypoints_shape"][1:])) for source in excluded
                ).items()
            )
        ),
        "usable_pose_rows": int(
            sum(source["rows"]["usable"] for source in source_records)
        ),
        "total_pose_rows": int(
            sum(source["rows"]["total"] for source in source_records)
        ),
        "individual_landmark_locations": int(
            keypoint_count * sum(source["rows"]["usable"] for source in source_records)
        ),
        "storage_mount_counts": _count(source_records, "storage_mount"),
        "source_origin_counts": _count(source_records, "source_origin"),
        "keypoint_dtype_counts": _count(source_records, "keypoints_dtype"),
        "skeleton_counts": _count(source_records, "skeleton_id"),
        "roi_shape_counts": dict(
            sorted(
                Counter(
                    str(tuple(source["roi_images_shape"][1:]))
                    for source in source_records
                ).items()
            )
        ),
        "roi_dtype_counts": dict(
            sorted(
                Counter(
                    str(source["roi_images_dtype"]) for source in source_records
                ).items()
            )
        ),
        "pixel_contract_counts": dict(
            sorted(
                Counter(
                    str(source["pixel_contract"]["resolved_name"])
                    for source in source_records
                ).items()
            )
        ),
        "pixel_contract_fallback_count": int(
            sum(
                source["pixel_contract"]["source"] != "crop_run_attribute"
                for source in source_records
            )
        ),
        "pixel_contract_source_counts": dict(
            sorted(
                Counter(
                    str(source["pixel_contract"]["source"]) for source in source_records
                ).items()
            )
        ),
        "safe_for_pose_merge_count": int(
            sum(bool(source["safe_for_pose_merge"]) for source in source_records)
        ),
        "problem_counts": dict(sorted(problems.items())),
        "duplicate_source_identity_digest_count": duplicate_digest_count,
        "duplicate_usable_row_identity_count": len(duplicate_row_identities),
        "duplicate_usable_row_excess_count": int(
            sum(count - 1 for count in duplicate_row_identities.values())
        ),
        "usable_source_detect_rows_missing": int(
            sum(
                source["lineage"]["usable_source_detect_rows_missing"]
                for source in source_records
            )
        ),
        "usable_source_refined_rows_missing": int(
            sum(
                source["lineage"]["usable_source_refined_rows_missing"]
                for source in source_records
            )
        ),
        "usable_rows_without_detection_or_refined_identity": int(
            sum(
                source["lineage"]["usable_rows_without_detection_or_refined_identity"]
                for source in source_records
            )
        ),
        "sources_with_incomplete_detection_or_refined_lineage": int(
            sum(
                not bool(source["exact_detection_or_refined_lineage_complete"])
                for source in source_records
            )
        ),
        "split_group_count": len(split_groups),
        "split_groups_with_multiple_recordings": int(
            sum(count > 1 for count in split_groups.values())
        ),
        "recordings_in_multi_recording_split_groups": int(
            sum(count for count in split_groups.values() if count > 1)
        ),
        "split_group_source_counts": dict(
            sorted(
                Counter(
                    source["split_group"]["source"] for source in source_records
                ).items()
            )
        ),
    }
    report: dict[str, Any] = {
        "schema_id": CENSUS_SCHEMA_ID,
        "schema_version": CENSUS_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "registry_path": str(registry_path.expanduser().resolve()),
        "reviewed_artifacts": [
            str(path.expanduser().resolve()) for path in reviewed_artifacts
        ],
        "metadata_mode": "direct_unconsolidated",
        "selection": {
            "review_state": "approved",
            "review_intended_use": "training",
            "skeleton_id": skeleton_id,
            "keypoint_array_shape_tail": [keypoint_count, 2],
            "expected_keypoint_labels": (
                list(DEFAULT_KEYPOINT_LABELS) if keypoint_count == 5 else None
            ),
        },
        "summary": summary,
        "sources": source_records,
        "excluded_other_skeletons": excluded,
    }
    if historical_merge is not None:
        report["historical_merge"] = _historical_comparison(
            historical_merge.expanduser().resolve(),
            sources=source_records,
            internals=internals,
        )
    return report


def _summary_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_id": report["schema_id"],
        "schema_version": report["schema_version"],
        "generated_at_utc": report["generated_at_utc"],
        "registry_path": report["registry_path"],
        "metadata_mode": report["metadata_mode"],
        "selection": report["selection"],
        "summary": report["summary"],
    }
    if "historical_merge" in report:
        payload["historical_merge"] = report["historical_merge"]
    return payload


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--skeleton-id", default=DEFAULT_SKELETON_ID)
    parser.add_argument("--keypoint-count", type=int, default=5)
    parser.add_argument("--historical-merge", type=Path)
    parser.add_argument("--reviewed-artifact", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    report = build_census(
        registry_path,
        skeleton_id=str(args.skeleton_id),
        keypoint_count=int(args.keypoint_count),
        historical_merge=args.historical_merge,
        reviewed_artifacts=args.reviewed_artifact,
    )
    if args.output is not None:
        write_json_atomic(args.output.expanduser().resolve(), report)
    print(json.dumps(_summary_payload(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
