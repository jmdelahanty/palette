"""Recover one training-only recording from matched legacy merged exports.

The output is a new, selector-ineligible derivative. It keeps detector images
and pose ROIs on their separate sampled-row axes; it never claims to recreate
the deleted source recording archive or its review/edit history.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any, Sequence

import numpy as np
import zarr

from fisheye.shared.pose_schema import canonicalize_keypoint_label
from fisheye.shared.system_metadata import build_invocation_record


SCHEMA_ID = "palette.training.merged_pose_detect_recovery.v1"
SOURCE_ONLY_SCHEMA_ID = "palette.training.merged_pose_detect_recovery_source.v1"
CROP_RUN_ID = "pose_head_center_192_from_merged_v001"
CROP_RECIPE_ID = "merged_pose_roi_center_192_exclude_outside_v1"
HEAD_LABELS = ("swim_bladder", "eye_left", "eye_right")
CROP_SIZE = 192


@dataclass(frozen=True)
class RecordingRowJoin:
    pose_source_dataset_id: str | None
    detect_source_dataset_id: str
    pose_merged_rows: np.ndarray
    detect_merged_rows: np.ndarray
    pose_to_detect_local: np.ndarray
    detect_only_local: np.ndarray
    pose_frame_idx: np.ndarray
    detect_frame_idx: np.ndarray


def _source_recording_id(dataset_id: str) -> str:
    return str(dataset_id).split(":", 1)[0]


def build_recording_row_join(
    *,
    recording_id: str,
    pose_source_ids: Sequence[str],
    pose_dataset_idx: np.ndarray,
    pose_frame_idx: np.ndarray,
    pose_boxes: np.ndarray,
    detect_source_ids: Sequence[str],
    detect_dataset_idx: np.ndarray,
    detect_frame_idx: np.ndarray,
    detect_boxes: np.ndarray,
    allow_no_pose: bool = False,
) -> RecordingRowJoin:
    """Join exact sampled-frame identities and refuse conflicting boxes."""
    pose_sources = [
        i
        for i, value in enumerate(pose_source_ids)
        if _source_recording_id(value) == recording_id
    ]
    detect_sources = [
        i
        for i, value in enumerate(detect_source_ids)
        if _source_recording_id(value) == recording_id
    ]
    if (len(pose_sources) != 1 and not (allow_no_pose and not pose_sources)) or len(
        detect_sources
    ) != 1:
        raise ValueError(
            f"Expected one pose and one detect source for {recording_id}; "
            f"found {len(pose_sources)} and {len(detect_sources)}"
        )
    pi = np.asarray(pose_dataset_idx)
    pf = np.asarray(pose_frame_idx)
    pb = np.asarray(pose_boxes)
    di = np.asarray(detect_dataset_idx)
    df = np.asarray(detect_frame_idx)
    db = np.asarray(detect_boxes)
    if (
        pi.ndim != 1
        or pf.shape != pi.shape
        or pb.shape != (len(pi), 4)
        or di.ndim != 1
        or df.shape != di.shape
        or db.shape != (len(di), 4)
    ):
        raise ValueError("Merged source-index or bounding-box shapes disagree")
    pose_rows = (
        np.flatnonzero(pi == pose_sources[0]).astype(np.int64)
        if pose_sources
        else np.empty(0, dtype=np.int64)
    )
    detect_rows = np.flatnonzero(di == detect_sources[0]).astype(np.int64)
    if (pose_sources and len(pose_rows) == 0) or len(detect_rows) == 0:
        raise ValueError("The selected recording has no pose or detection rows")
    pose_frames = pf[pose_rows].astype(np.int64, copy=False)
    detect_frames = df[detect_rows].astype(np.int64, copy=False)
    if len(np.unique(pose_frames)) != len(pose_frames):
        raise ValueError("duplicate pose frame identity in selected recording")
    if len(np.unique(detect_frames)) != len(detect_frames):
        raise ValueError("duplicate detection frame identity in selected recording")
    detect_local_by_frame = {int(frame): i for i, frame in enumerate(detect_frames)}
    if any(int(frame) not in detect_local_by_frame for frame in pose_frames):
        raise ValueError("A pose row has no matching detection frame")
    pose_to_detect = np.array(
        [detect_local_by_frame[int(frame)] for frame in pose_frames], dtype=np.int32
    )
    if not np.array_equal(pb[pose_rows], db[detect_rows[pose_to_detect]]):
        raise ValueError("pose crop provenance and detection box mismatch")
    matched = set(pose_to_detect.tolist())
    detect_only = np.array(
        [i for i in range(len(detect_rows)) if i not in matched], dtype=np.int32
    )
    return RecordingRowJoin(
        pose_source_dataset_id=(
            str(pose_source_ids[pose_sources[0]]) if pose_sources else None
        ),
        detect_source_dataset_id=str(detect_source_ids[detect_sources[0]]),
        pose_merged_rows=pose_rows,
        detect_merged_rows=detect_rows,
        pose_to_detect_local=pose_to_detect,
        detect_only_local=detect_only,
        pose_frame_idx=pose_frames,
        detect_frame_idx=detect_frames,
    )


def select_centered_head_crops(
    points_roi: np.ndarray, *, image_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select rows with all three points inside an unshifted central 192 ROI."""
    points = np.asarray(points_roi)
    height, width = image_shape
    if points.ndim != 3 or points.shape[1:] != (3, 2):
        raise ValueError("Pose points must be (rows, 3, 2)")
    if min(height, width) < CROP_SIZE or height % 2 or width % 2:
        raise ValueError(
            "Centered crop requires an even source ROI at least 192 pixels"
        )
    origin = np.array([(width - CROP_SIZE) // 2, (height - CROP_SIZE) // 2])
    local = points.astype(np.float64, copy=False) - origin
    inside = np.isfinite(local).all(axis=(1, 2))
    inside &= np.logical_and(local >= 0, local < CROP_SIZE).all(axis=(1, 2))
    kept = np.flatnonzero(inside).astype(np.int64)
    excluded = np.flatnonzero(~inside).astype(np.int64)
    return kept, local[kept], excluded


def _read_array(root: Path, path: str) -> zarr.Array:
    return zarr.open_array(str(root / path), mode="r")


def _read_small(root: Path, path: str) -> np.ndarray:
    return np.asarray(_read_array(root, path)[:])


def _read_rows(array: zarr.Array, rows: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(array[int(i)]) for i in rows], axis=0)


def _sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_receipt_entry(receipt: dict[str, Any], path: Path) -> dict[str, Any]:
    marker = "training/datasets/"
    text = str(path)
    if marker not in text:
        raise ValueError(f"Merged source is outside the training datasets root: {path}")
    key = text.split(marker, 1)[1]
    entry = receipt.get(key)
    if not isinstance(entry, dict) or entry.get("match") is not True:
        raise ValueError(f"No successful historical copy receipt for {key}")
    return {"receipt_key": key, "tree_sha256_at_copy": entry["dst"][0]}


def _check_registry_source(
    *,
    registry_path: Path,
    recording_id: str,
    pose_dataset_id: str,
    refined_run: str,
    required_species: str,
) -> dict[str, str]:
    connection = sqlite3.connect(f"file:{registry_path}?mode=ro", uri=True)
    try:
        species = [
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT species FROM recording_subjects WHERE dataset_id = ?",
                (pose_dataset_id,),
            )
        ]
        if species != [required_species]:
            raise ValueError(
                f"Recording {recording_id} does not have the required species "
                f"{required_species!r}: {species!r}"
            )
        review = connection.execute(
            "SELECT review_state, review_intended_use, review_method "
            "FROM keypoint_quality WHERE dataset_id = ? AND refined_run = ?",
            (pose_dataset_id, refined_run),
        ).fetchone()
        if review != ("approved", "training", "manual"):
            raise ValueError(
                f"Pose source lacks approved manual training review: {review!r}"
            )
        return {
            "species": species[0],
            "pose_review_state": review[0],
            "pose_review_intended_use": review[1],
            "pose_review_method": review[2],
            "pose_refined_run": refined_run,
        }
    finally:
        connection.close()


def _check_recording_species(
    *, registry_path: Path, recording_id: str, required_species: str
) -> dict[str, str]:
    """A detection-only recovery still needs an exact species assignment."""
    connection = sqlite3.connect(f"file:{registry_path}?mode=ro", uri=True)
    try:
        species = [
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT species FROM recording_subjects WHERE recording_id = ?",
                (recording_id,),
            )
        ]
        if species != [required_species]:
            raise ValueError(
                f"Recording {recording_id} does not have the required species "
                f"{required_species!r}: {species!r}"
            )
        return {"species": species[0], "pose_review_state": "not_applicable_no_pose"}
    finally:
        connection.close()


def _create_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    data = np.ascontiguousarray(values)
    chunks = (max(1, min(16, len(data))), *data.shape[1:])
    array = group.create_array(
        name,
        shape=data.shape,
        dtype=data.dtype,
        chunks=chunks,
        compressors=[
            zarr.codecs.BloscCodec(cname="lz4", clevel=1, shuffle="bitshuffle")
        ],
    )
    array[:] = data


def recover_recording(
    *,
    recording_id: str,
    pose_merged: Path,
    detect_merged: Path,
    pose_manifest: Path,
    detect_manifest: Path,
    registry_path: Path,
    copy_receipt: Path,
    output: Path,
    required_species: str = "Danio rerio",
    source_only: bool = False,
) -> dict[str, Any]:
    """Publish one new training-only derivative after exact source validation.

    ``source_only`` preserves all surviving detector and pose rows for later
    versioned crop runs. It permits a detector-only recording but never invents
    missing pose rows or an original full-sensor frame surface.
    """
    pose_meta = json.loads(pose_manifest.read_text())
    detect_meta = json.loads(detect_manifest.read_text())
    if pose_meta.get("task") != "pose" or detect_meta.get("task") != "detect":
        raise ValueError("Expected one pose and one detection source manifest")
    if detect_meta.get("query_filter", {}).get("require_approved") is not True:
        raise ValueError("Detection source manifest did not require approved boxes")
    if detect_meta.get("source_type") != "manual":
        raise ValueError("This recovery requires the reviewed manual detection export")
    labels = tuple(
        canonicalize_keypoint_label(value)
        for value in pose_meta.get("pose_schema", {}).get("keypoint_labels", ())
    )
    if labels != HEAD_LABELS:
        raise ValueError(f"Pose source labels do not match the head triangle: {labels}")
    pose_run = pose_meta["merged_export"]["run_name"]
    detect_run = detect_meta["merged_export"]["run_name"]
    pose_source_ids = tuple(
        str(x) for x in _read_small(pose_merged, "source_index/source_dataset_id")
    )
    detect_source_ids = tuple(
        str(x) for x in _read_small(detect_merged, "source_index/source_dataset_id")
    )
    pose_box_path = f"crop_runs/{pose_run}/crop_bbox_norm_coords"
    detect_box_path = f"crop_runs/{detect_run}/bbox_norm_coords"
    join = build_recording_row_join(
        recording_id=recording_id,
        pose_source_ids=pose_source_ids,
        pose_dataset_idx=_read_small(pose_merged, "source_index/source_dataset_idx"),
        pose_frame_idx=_read_small(pose_merged, "source_index/source_frame_idx"),
        pose_boxes=_read_small(pose_merged, pose_box_path),
        detect_source_ids=detect_source_ids,
        detect_dataset_idx=_read_small(
            detect_merged, "source_index/source_dataset_idx"
        ),
        detect_frame_idx=_read_small(detect_merged, "source_index/source_frame_idx"),
        detect_boxes=_read_small(detect_merged, detect_box_path),
        allow_no_pose=source_only,
    )
    pose_source = [
        row
        for row in pose_meta["merged_export"]["source_datasets"]
        if row.get("dataset_id") == join.pose_source_dataset_id
    ]
    detect_source = [
        row
        for row in detect_meta["merged_export"]["source_datasets"]
        if row.get("dataset_id") == join.detect_source_dataset_id
    ]
    if (
        len(pose_source) != int(join.pose_source_dataset_id is not None)
        or len(detect_source) != 1
    ):
        raise ValueError("Merged source manifests do not bind the selected recording")
    review = (
        _check_registry_source(
            registry_path=registry_path,
            recording_id=recording_id,
            pose_dataset_id=join.pose_source_dataset_id,
            refined_run=str(pose_source[0].get("row_gate_refined_run") or ""),
            required_species=required_species,
        )
        if join.pose_source_dataset_id is not None
        else _check_recording_species(
            registry_path=registry_path,
            recording_id=recording_id,
            required_species=required_species,
        )
    )
    receipt = json.loads(copy_receipt.read_text())
    pose_receipt = (
        _source_receipt_entry(receipt, pose_merged)
        if join.pose_source_dataset_id is not None
        else None
    )
    detect_receipt = _source_receipt_entry(receipt, detect_merged)

    detect_pixels = _read_rows(
        _read_array(detect_merged, "raw_video/images_ds"),
        join.detect_merged_rows,
    )
    detect_boxes = _read_small(detect_merged, detect_box_path)[join.detect_merged_rows]
    if (
        detect_pixels.dtype != np.uint8
        or detect_pixels.ndim != 3
        or not np.isfinite(detect_boxes).all()
    ):
        raise ValueError("Surviving merged detector pixels or boxes are invalid")
    source_arrays = {
        "recovered_sources/detect/images_ds": detect_pixels,
        "recovered_sources/detect/bbox_norm_coords": detect_boxes,
        "recovered_sources/detect/source_merged_row": join.detect_merged_rows,
        "recovered_sources/detect/source_frame_idx": join.detect_frame_idx,
        "recovered_sources/detect/detect_only_local_row": join.detect_only_local,
    }
    kept = np.empty(0, dtype=np.int64)
    excluded = np.empty(0, dtype=np.int64)
    if join.pose_source_dataset_id is not None:
        pose_pixels = _read_rows(
            _read_array(pose_merged, f"crop_runs/{pose_run}/roi_images"),
            join.pose_merged_rows,
        )
        pose_points = _read_rows(
            _read_array(pose_merged, f"keypoints_runs/{pose_run}/keypoints_roi"),
            join.pose_merged_rows,
        )
        pose_boxes = _read_small(pose_merged, pose_box_path)[join.pose_merged_rows]
        if (
            pose_pixels.dtype != np.uint8
            or pose_pixels.ndim != 3
            or pose_points.shape != (len(pose_pixels), 3, 2)
            or not np.isfinite(pose_points).all()
            or not np.isfinite(pose_boxes).all()
        ):
            raise ValueError("Surviving merged pose pixels or labels are invalid")
        source_arrays.update(
            {
                "recovered_sources/pose/roi_images": pose_pixels,
                "recovered_sources/pose/keypoints_roi": pose_points,
                "recovered_sources/pose/source_merged_row": join.pose_merged_rows,
                "recovered_sources/pose/source_frame_idx": join.pose_frame_idx,
                "recovered_sources/pose/detect_local_row": join.pose_to_detect_local,
                "recovered_sources/pose/crop_bbox_norm_coords": pose_boxes,
            }
        )
        if not source_only:
            kept, head_points, excluded = select_centered_head_crops(
                pose_points, image_shape=tuple(pose_pixels.shape[1:])
            )
            if len(kept) == 0:
                raise ValueError("No head pose rows remain after the 192-pixel gate")
            top = (pose_pixels.shape[1] - CROP_SIZE) // 2
            left = (pose_pixels.shape[2] - CROP_SIZE) // 2
            head_pixels = np.ascontiguousarray(
                pose_pixels[kept, top : top + CROP_SIZE, left : left + CROP_SIZE]
            )
            source_arrays.update(
                {
                    f"crop_runs/{CROP_RUN_ID}/roi_images": head_pixels,
                    f"crop_runs/{CROP_RUN_ID}/keypoints_roi": head_points.astype(
                        np.float32
                    ),
                    f"crop_runs/{CROP_RUN_ID}/keypoint_visibility": np.full(
                        (len(kept), 3), 2, dtype=np.uint8
                    ),
                    f"crop_runs/{CROP_RUN_ID}/source_pose_local_row": kept,
                    f"crop_runs/{CROP_RUN_ID}/source_detect_local_row": join.pose_to_detect_local[
                        kept
                    ],
                    f"crop_runs/{CROP_RUN_ID}/source_frame_idx": join.pose_frame_idx[
                        kept
                    ],
                    f"crop_runs/{CROP_RUN_ID}/roi_origin_xy_in_pose_512": np.tile(
                        [left, top], (len(kept), 1)
                    ).astype(np.int32),
                    f"crop_runs/{CROP_RUN_ID}/excluded_pose_local_row": excluded,
                }
            )
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing recovery archive: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent)
    )
    hashes = {path: _sha256_array(data) for path, data in source_arrays.items()}
    attrs = {
        "schema_id": SOURCE_ONLY_SCHEMA_ID if source_only else SCHEMA_ID,
        "schema_version": 1,
        "zarr_purpose": "training",
        "training_artifact_status": "complete",
        "stage_selector_eligible": False,
        "recording_id": recording_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_pose": (
            {
                "merged_zarr": str(pose_merged),
                "set_id": pose_meta["set_id"],
                "run_id": pose_run,
                "dataset_id": join.pose_source_dataset_id,
                "manifest_sha256": _sha256_file(pose_manifest),
                **pose_receipt,
            }
            if pose_receipt is not None
            else None
        ),
        "source_detect": {
            "merged_zarr": str(detect_merged),
            "set_id": detect_meta["set_id"],
            "run_id": detect_run,
            "dataset_id": join.detect_source_dataset_id,
            "manifest_sha256": _sha256_file(detect_manifest),
            **detect_receipt,
            "review_gate": "require_approved/manual_from_historical_manifest",
        },
        "review_snapshot": review,
        "pose_source_row_count": len(join.pose_merged_rows),
        "detect_source_row_count": len(join.detect_merged_rows),
        "detect_only_row_count": len(join.detect_only_local),
        "recovery_mode": "source_only" if source_only else "source_and_head_crop",
        "array_sha256": hashes,
        "invocation": build_invocation_record(
            tool="fisheye.training.recover_merged_training_recording",
            args={
                "recording_id": recording_id,
                "output": str(output),
                "pose_merged": str(pose_merged),
                "detect_merged": str(detect_merged),
            },
        ),
    }
    if not source_only:
        attrs.update(
            {
                "head_crop_row_count": len(kept),
                "head_crop_excluded_row_count": len(excluded),
                "head_crop_run": CROP_RUN_ID,
                "head_crop_recipe": CROP_RECIPE_ID,
                "head_keypoint_labels": list(HEAD_LABELS),
            }
        )
    try:
        root = zarr.open_group(str(temporary), mode="w", use_consolidated=False)
        for path, values in source_arrays.items():
            parent_path, name = path.rsplit("/", 1)
            group = root.require_group(parent_path)
            _create_array(group, name, values)
        root.attrs.update(attrs)
        if not source_only:
            crop = root[f"crop_runs/{CROP_RUN_ID}"]
            crop.attrs.update(
                {
                    "schema_id": "palette.training.recovered_pose_head_crop.v1",
                    "crop_storage_mode": "materialized",
                    "stage_selector_eligible": False,
                    "recipe_id": CROP_RECIPE_ID,
                    "roi_size": [CROP_SIZE, CROP_SIZE],
                    "pose_schema": {
                        "skeleton_id": "pose_schema:traditional_v1",
                        "keypoint_labels": list(HEAD_LABELS),
                        "kpt_shape": [3, 3],
                    },
                    "source_pose_run": pose_run,
                    "source_detect_run": detect_run,
                }
            )
        zarr.consolidate_metadata(str(temporary))
        published_attrs = validate_recovered_recording(
            temporary,
            expected_recording_id=recording_id,
            require_source_only=source_only,
            require_no_crop_runs=source_only,
        )
        if published_attrs.get("array_sha256") != hashes:
            raise ValueError("Consolidated recovery metadata lost array digests")
        os.rename(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return attrs


def validate_recovered_recording(
    path: Path,
    *,
    expected_recording_id: str | None = None,
    require_source_only: bool = False,
    require_no_crop_runs: bool = False,
) -> dict[str, Any]:
    """Reopen a published derivative and verify content and row identities."""
    root = zarr.open_group(str(path), mode="r", use_consolidated=True)
    attrs = dict(root.attrs)
    if attrs.get("schema_id") not in {SCHEMA_ID, SOURCE_ONLY_SCHEMA_ID}:
        raise ValueError("Unsupported recovered training schema")
    if (
        attrs.get("training_artifact_status") != "complete"
        or attrs.get("stage_selector_eligible") is not False
    ):
        raise ValueError("Recovered training source is incomplete or selector-eligible")
    if (
        expected_recording_id is not None
        and attrs.get("recording_id") != expected_recording_id
    ):
        raise ValueError("Recovered training recording identity mismatch")
    if require_source_only and attrs.get("recovery_mode") != "source_only":
        raise ValueError("Recovery does not preserve the full source-only row axes")
    if require_no_crop_runs and "crop_runs" in root:
        raise ValueError("Recovery unexpectedly includes a crop run")
    hashes = attrs.get("array_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Recovered training source lacks array hashes")
    for array_path, expected in hashes.items():
        if array_path not in root or len(str(expected)) != 64:
            raise ValueError(f"Recovered array declaration is incomplete: {array_path}")
        if _sha256_array(np.asarray(root[array_path][:])) != expected:
            raise ValueError(f"Recovered array content digest mismatch: {array_path}")
    detect = root["recovered_sources/detect"]
    detect_n = int(detect["images_ds"].shape[0])
    if detect_n != int(attrs.get("detect_source_row_count", -1)):
        raise ValueError("Recovered detector row count mismatch")
    detect_frames = np.asarray(detect["source_frame_idx"][:])
    detect_boxes = np.asarray(detect["bbox_norm_coords"][:])
    if len(np.unique(detect_frames)) != detect_n or detect_boxes.shape != (detect_n, 4):
        raise ValueError("Recovered detector frame identity or boxes are invalid")
    if attrs.get("source_pose") is None:
        if (
            int(attrs.get("pose_source_row_count", -1)) != 0
            or "pose" in root["recovered_sources"]
        ):
            raise ValueError("Detect-only recovery has an invented pose surface")
        if int(attrs.get("detect_only_row_count", -1)) != detect_n:
            raise ValueError("Detect-only recovery has missing detector-only rows")
    else:
        pose = root["recovered_sources/pose"]
        pose_n = int(pose["roi_images"].shape[0])
        if pose_n != int(attrs.get("pose_source_row_count", -1)):
            raise ValueError("Recovered pose row count mismatch")
        pose_frames = np.asarray(pose["source_frame_idx"][:])
        pose_boxes = np.asarray(pose["crop_bbox_norm_coords"][:])
        matched = np.asarray(pose["detect_local_row"][:], dtype=np.int64)
        if (
            len(np.unique(pose_frames)) != pose_n
            or matched.shape != (pose_n,)
            or np.any((matched < 0) | (matched >= detect_n))
            or not np.array_equal(pose_frames, detect_frames[matched])
            or not np.array_equal(pose_boxes, detect_boxes[matched])
        ):
            raise ValueError("Recovered pose-to-detection row binding is invalid")
        only = np.asarray(detect["detect_only_local_row"][:], dtype=np.int64)
        expected_only = np.setdiff1d(np.arange(detect_n), matched)
        if not np.array_equal(only, expected_only):
            raise ValueError("Recovered detection-only row declaration is invalid")
    return attrs


def recover_full_source_cohort(
    *,
    pose_merged: Path,
    detect_merged: Path,
    pose_manifest: Path,
    detect_manifest: Path,
    registry_path: Path,
    copy_receipt: Path,
    output_dir: Path,
    required_species: str = "Danio rerio",
    resume: bool = False,
) -> dict[str, Any]:
    """Recover every detector source, preserving absent pose as absent."""
    pose_meta = json.loads(pose_manifest.read_text())
    detect_meta = json.loads(detect_manifest.read_text())
    pose_recordings = {
        _source_recording_id(str(row["dataset_id"]))
        for row in pose_meta["merged_export"]["source_datasets"]
    }
    detect_recordings = {
        _source_recording_id(str(row["dataset_id"]))
        for row in detect_meta["merged_export"]["source_datasets"]
    }
    if not detect_recordings or not pose_recordings.issubset(detect_recordings):
        raise ValueError("Pose and detection merged source cohorts disagree")
    expected_pose_rows = int(pose_meta["merged_export"]["counts"]["total_samples"])
    expected_detect_rows = int(detect_meta["merged_export"]["counts"]["total_samples"])
    results: list[dict[str, Any]] = []
    for ordinal, recording_id in enumerate(sorted(detect_recordings), start=1):
        path = output_dir / f"{recording_id}_recovered_training.zarr"
        if path.exists():
            if not resume:
                raise FileExistsError(f"Recovery output already exists: {path}")
            attrs = validate_recovered_recording(
                path, expected_recording_id=recording_id, require_source_only=True
            )
            if (
                attrs["source_detect"]["manifest_sha256"]
                != _sha256_file(detect_manifest)
                or attrs["source_detect"]["merged_zarr"] != str(detect_merged)
                or (
                    attrs.get("source_pose") is not None
                    and (
                        attrs["source_pose"]["manifest_sha256"]
                        != _sha256_file(pose_manifest)
                        or attrs["source_pose"]["merged_zarr"] != str(pose_merged)
                    )
                )
            ):
                raise ValueError(f"Existing recovery binds different sources: {path}")
        else:
            attrs = recover_recording(
                recording_id=recording_id,
                pose_merged=pose_merged,
                detect_merged=detect_merged,
                pose_manifest=pose_manifest,
                detect_manifest=detect_manifest,
                registry_path=registry_path,
                copy_receipt=copy_receipt,
                output=path,
                required_species=required_species,
                source_only=True,
            )
        result = {
            "recording_id": recording_id,
            "path": str(path),
            "pose_rows": int(attrs["pose_source_row_count"]),
            "detect_rows": int(attrs["detect_source_row_count"]),
            "detect_only_rows": int(attrs["detect_only_row_count"]),
            "array_digest_index_sha256": hashlib.sha256(
                json.dumps(
                    attrs["array_sha256"], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
        }
        results.append(result)
        print(
            f"[{ordinal}/{len(detect_recordings)}] {recording_id}: "
            f"pose={result['pose_rows']} detect={result['detect_rows']} "
            f"detect_only={result['detect_only_rows']}",
            flush=True,
        )
    pose_total = sum(row["pose_rows"] for row in results)
    detect_total = sum(row["detect_rows"] for row in results)
    detect_only_total = sum(row["detect_only_rows"] for row in results)
    if (
        pose_total != expected_pose_rows
        or detect_total != expected_detect_rows
        or sum(row["pose_rows"] > 0 for row in results) != len(pose_recordings)
    ):
        raise ValueError("Recovered cohort row totals differ from the merged manifests")
    collection = {
        "schema_id": "palette.training.merged_recovery_collection.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_pose_set_id": pose_meta["set_id"],
        "source_detect_set_id": detect_meta["set_id"],
        "source_pose_manifest_sha256": _sha256_file(pose_manifest),
        "source_detect_manifest_sha256": _sha256_file(detect_manifest),
        "recording_count": len(results),
        "recordings_with_pose": len(pose_recordings),
        "pose_rows": pose_total,
        "detect_rows": detect_total,
        "detect_only_rows": detect_only_total,
        "stage_selector_eligible": False,
        "recordings": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "recovery_collection.json"
    temporary_receipt = output_dir / ".recovery_collection.json.partial"
    temporary_receipt.write_text(
        json.dumps(collection, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary_receipt, receipt_path)
    return collection


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "pose-merged",
        "detect-merged",
        "pose-manifest",
        "detect-manifest",
        "registry",
        "copy-receipt",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--recording-id")
    parser.add_argument("--output")
    parser.add_argument("--all-recordings", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--required-species", default="Danio rerio")
    args = parser.parse_args(argv)
    common = dict(
        pose_merged=Path(args.pose_merged),
        detect_merged=Path(args.detect_merged),
        pose_manifest=Path(args.pose_manifest),
        detect_manifest=Path(args.detect_manifest),
        registry_path=Path(args.registry),
        copy_receipt=Path(args.copy_receipt),
        required_species=args.required_species,
    )
    if args.all_recordings:
        if (
            not args.source_only
            or not args.output_dir
            or args.recording_id
            or args.output
        ):
            parser.error(
                "--all-recordings requires --source-only and --output-dir only"
            )
        result = recover_full_source_cohort(
            **common, output_dir=Path(args.output_dir), resume=args.resume
        )
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "recordings"},
                indent=2,
            )
        )
        return
    if not args.recording_id or not args.output or args.output_dir or args.resume:
        parser.error("single-recording mode requires --recording-id and --output")
    result = recover_recording(
        recording_id=args.recording_id,
        output=Path(args.output),
        source_only=args.source_only,
        **common,
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "recording_id": result["recording_id"],
                "pose_rows": result["pose_source_row_count"],
                "detect_rows": result["detect_source_row_count"],
                "detect_only_rows": result["detect_only_row_count"],
                "recovery_mode": result["recovery_mode"],
                "head_crop_rows": result.get("head_crop_row_count"),
                "head_crop_excluded_rows": result.get("head_crop_excluded_row_count"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
