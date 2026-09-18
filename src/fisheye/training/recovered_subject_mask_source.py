"""Identity-checked adapter from merged masks to recovered per-recording ROIs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import zarr

from fisheye.training.recover_merged_training_recording import (
    _sha256_array,
    validate_recovered_recording,
)
from fisheye.training.materialize_recovered_pose_head_crops import _digest_index_sha256

BOX_ATOL = 1e-7


def join_mask_rows(recording_id, dataset_ids, dataset_idx, frame_idx, pose_frame_idx):
    sources = [
        i
        for i, name in enumerate(dataset_ids)
        if str(name).split(":", 1)[0] == recording_id
    ]
    if len(sources) != 1:
        raise ValueError("Expected exactly one merged mask source for the recording")
    dataset_idx, frame_idx, pose_frame_idx = map(
        np.asarray, (dataset_idx, frame_idx, pose_frame_idx)
    )
    if (
        dataset_idx.ndim != 1
        or frame_idx.shape != dataset_idx.shape
        or pose_frame_idx.ndim != 1
        or any(
            a.dtype.kind not in "iu" for a in (dataset_idx, frame_idx, pose_frame_idx)
        )
    ):
        raise ValueError("Source row identity arrays must be aligned integer vectors")
    if np.any(dataset_idx < 0) or np.any(dataset_idx >= len(dataset_ids)):
        raise ValueError("Invalid source dataset index")
    rows = np.flatnonzero(dataset_idx == sources[0])
    frames = frame_idx[rows]
    if (
        not len(rows)
        or len(np.unique(frames)) != len(frames)
        or len(np.unique(pose_frame_idx)) != len(pose_frame_idx)
    ):
        raise ValueError("Missing or ambiguous source frame identity")
    lookup = {int(f): i for i, f in enumerate(pose_frame_idx)}
    if any(int(f) not in lookup for f in frames):
        raise ValueError("Mask frame has no recovered pose crop")
    return rows.astype(np.int64), np.array(
        [lookup[int(f)] for f in frames], dtype=np.int64
    )


def verify_crop_pair(mask_image, pose_image, mask_box, pose_box):
    if (
        mask_image.shape != pose_image.shape
        or mask_image.dtype != pose_image.dtype
        or not np.array_equal(mask_image, pose_image)
    ):
        raise ValueError("Mask and recovered pose crop pixels differ")
    if (
        np.shape(mask_box) != (4,)
        or np.shape(pose_box) != (4,)
        or not np.allclose(mask_box, pose_box, atol=BOX_ATOL, rtol=0, equal_nan=False)
    ):
        raise ValueError("Mask and recovered pose crop box mismatch")


def read_recovered_mask_source(
    archive: Path,
    merged: Path,
    *,
    mask_run: str,
    crop_run: str,
    legacy_unconsolidated_source: bool = False,
):
    """Read one recording with bounded resident payload and exact pixel checks.

    The legacy frame key is a sampled training-row identity, not an acquisition
    frame claim. Duplicate images are expected and never used as join keys.
    """
    attrs = validate_recovered_recording(archive, require_source_only=True)
    root = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    source = zarr.open_group(
        str(merged), mode="r", use_consolidated=not legacy_unconsolidated_source
    )
    pose, index = root["recovered_sources/pose"], source["source_index"]
    crop, masks = (
        source[f"crop_runs/{crop_run}"],
        source[f"subject_mask_runs/{mask_run}"],
    )
    ids = index["source_dataset_id"][:].tolist()
    rows, pose_rows = join_mask_rows(
        str(attrs["recording_id"]),
        ids,
        index["source_dataset_idx"][:],
        index["source_frame_idx"][:],
        pose["source_frame_idx"][:],
    )
    shape = tuple(masks["masks_roi"].shape)
    if (
        len(shape) != 4
        or len(rows) * (np.prod(shape[1:]) + np.prod(crop["roi_images"].shape[1:]))
        > 512 * 1024**2
    ):
        raise ValueError("One recording exceeds the 512 MiB recovery payload budget")
    labels = tuple(masks.attrs["mask_labels"])
    if len(labels) != shape[1] or len(set(labels)) != len(labels):
        raise ValueError("Mask channels disagree with their labels")
    images, mask_values, points, boxes = [], [], [], []
    for source_row, pose_row in zip(rows, pose_rows, strict=True):
        image = np.asarray(crop["roi_images"][int(source_row)])
        box = np.asarray(crop["crop_bbox_norm_coords"][int(source_row)])
        verify_crop_pair(
            image,
            np.asarray(pose["roi_images"][int(pose_row)]),
            box,
            np.asarray(pose["crop_bbox_norm_coords"][int(pose_row)]),
        )
        images.append(image)
        mask_values.append(np.asarray(masks["masks_roi"][int(source_row)]))
        points.append(np.asarray(pose["keypoints_roi"][int(pose_row)]))
        boxes.append(box)
    arrays = {
        "roi_images": np.stack(images),
        "masks_roi": np.stack(mask_values),
        "head_keypoints_roi": np.stack(points),
        "source_bbox_norm_coords": np.stack(boxes),
        "source_merged_row": rows,
        "source_pose_local_row": pose_rows,
        "frame_indices": np.asarray(index["source_frame_idx"][:])[rows],
    }
    arrays["detection_source"] = np.asarray(crop["detection_source"][:])[rows]
    for name in (
        "source_roi_idx",
        "source_refined_row_ids",
        "source_detect_row_index",
        "label_origin_codes",
        "supervision_mode_codes",
    ):
        if name in index:
            arrays[name] = np.asarray(index[name][:])[rows]
    arrays["target_valid_channels"] = np.asarray(masks["target_valid_channels"][:])[
        rows
    ]
    if (
        arrays["target_valid_channels"].shape != (len(rows), len(labels))
        or arrays["target_valid_channels"].dtype != np.dtype(bool)
        or not arrays["target_valid_channels"].all()
    ):
        raise ValueError("Recovery requires valid supervision in every mask channel")
    if (
        arrays["masks_roi"].dtype != np.uint8
        or arrays["roi_images"].dtype != np.uint8
        or arrays["masks_roi"].shape[2:] != arrays["roi_images"].shape[1:]
    ):
        raise ValueError("Expected aligned dense uint8 masks and mono crops")
    source_id = int(np.asarray(index["source_dataset_idx"][:])[rows[0]])
    binding = {
        "recording_id": attrs["recording_id"],
        "merged_archive": str(merged.resolve()),
        "mask_run": mask_run,
        "crop_run": crop_run,
        "source_dataset_id": ids[source_id],
        "source_metadata_mode": (
            "legacy_unconsolidated" if legacy_unconsolidated_source else "consolidated"
        ),
        "source_metadata_sha256": hashlib.sha256(
            (merged / "zarr.json").read_bytes()
        ).hexdigest(),
        "recovery_source_digest_index_sha256": _digest_index_sha256(
            attrs["array_sha256"]
        ),
        "selected_array_sha256": {k: _sha256_array(v) for k, v in arrays.items()},
        "mask_run_attrs": dict(masks.attrs),
        "crop_box_atol": BOX_ATOL,
        "frame_index_domain": "legacy_training_sample_row",
        "historical_edit_records_recovered": False,
    }
    for name in (
        "source_zarr_path",
        "source_run_name",
        "source_stage_group",
        "source_label_schema_id",
        "source_projection_mode",
    ):
        if name in index:
            binding[name] = str(index[name][source_id])
    return arrays, labels, binding


def use_refined_mask_snapshot(archive, run_name, arrays, labels, binding):
    """Bind a fresh version to corrected dense masks, preserving the old seed.

    Capture identity before and after reading to refuse concurrent editing.
    The resulting new raw mask copy seals these exact pixels for derivation.
    """
    if not run_name or "/" in run_name or run_name.startswith("."):
        raise ValueError("Refined mask run must be one safe path component")
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    group = root[f"refined_subject_masks_runs/{run_name}"]
    attrs_before = dict(group.attrs)
    crop = root[f"crop_runs/{attrs_before['source_crop_run']}"]
    if (
        tuple(group.attrs["mask_labels"]) != labels
        or group.attrs.get("source_bindings", {}).get("recording_id")
        != binding["recording_id"]
        or group.attrs.get("source_bindings", {}).get(
            "recovery_source_digest_index_sha256"
        )
        != binding["recovery_source_digest_index_sha256"]
        or not np.array_equal(
            group["source_merged_row"][:], arrays["source_merged_row"]
        )
        or not np.array_equal(crop["roi_images"][:], arrays["roi_images"])
    ):
        raise ValueError("Refined masks do not bind these recovered crop rows")
    values = np.asarray(group["masks_roi"][:])
    digest = _sha256_array(values)
    # Re-open attrs because Zarr group attrs are a snapshot.
    after = zarr.open_group(
        str(archive / f"refined_subject_masks_runs/{run_name}"),
        mode="r",
        use_consolidated=False,
    )
    if (
        values.dtype != np.uint8
        or values.shape != arrays["masks_roi"].shape
        or attrs_before != dict(after.attrs)
        or digest != _sha256_array(np.asarray(after["masks_roi"][:]))
    ):
        raise ValueError("Refined mask source changed while taking a snapshot")
    binding = {
        **binding,
        "refined_mask_snapshot": {
            "run_path": f"refined_subject_masks_runs/{run_name}",
            "masks_roi_sha256": digest,
            "attrs": attrs_before,
        },
        "derivation_mask_sha256": digest,
    }
    return {**arrays, "masks_roi": values}, binding
