"""Seed versioned ROI annotations from reviewed native masks and named keypoints.

The source archive and its selectors remain intact. Outputs are independent,
selector-ineligible annotation snapshots using the same tail geometry and editor
as recovered datasets, with native frame/crop identities explicitly preserved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import zarr

from fisheye.shared.keypoint_motion_authority import (
    keypoint_source_crop_run_from_attributes,
)
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_run_completion import is_run_complete
from fisheye.training.recover_merged_training_recording import _sha256_array
from fisheye.training.recover_merged_subject_masks import (
    publish_review_payload,
    review_tasks,
    validate_initial_payload,
)
from fisheye.training.recovered_mask_review_payload import (
    build_review_payload,
    run_paths,
)
from fisheye.training.recovered_subject_mask_source import use_refined_mask_snapshot

MAX_SOURCE_BYTES = 512 * 1024 * 1024
HEAD_LABELS = ("swim_bladder", "eye_left", "eye_right")


def _approved(value):
    return isinstance(value, dict) and all(
        value.get(k) == v
        for k, v in (
            ("state", "approved"),
            ("method", "manual"),
            ("intended_use", "training"),
        )
    )


def _row_ids(group, crop_frames):
    ids = np.asarray(group["source_crop_row_ids"][:])
    if (
        ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or not len(ids)
        or len(np.unique(ids)) != len(ids)
        or np.any(ids < 0)
        or np.any(ids >= len(crop_frames))
        or not np.array_equal(group["frame_indices"][:], crop_frames[ids])
    ):
        raise ValueError("Invalid or conflicting source crop row/frame identities")
    return ids.astype(np.int64)


def read_native_source(archive, *, mask_run, keypoint_run):
    # The native archive contains mutable reviewed labels; consolidated metadata
    # must not hide new groups or current review attrs while taking the snapshot.
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    if root.attrs.get("zarr_purpose") != "training" or not root.attrs.get(
        "recording_id"
    ):
        raise ValueError(
            "A native training archive with recording identity is required"
        )
    for name in (mask_run, keypoint_run):
        if not name or name.startswith(".") or "/" in name:
            raise ValueError("Exact non-hidden run names are required")
    mask = root[f"refined_subject_masks_runs/{mask_run}"]
    pose = root[f"refined_keypoints_runs/{keypoint_run}"]
    crop_run = keypoint_source_crop_run_from_attributes(mask.attrs)
    if keypoint_source_crop_run_from_attributes(pose.attrs) != crop_run:
        raise ValueError("Masks and keypoints must bind the same source crop run")
    crop = root[f"crop_runs/{crop_run}"]
    labels = tuple(mask.attrs.get("mask_labels", []))
    point_labels = list(pose.attrs.get("keypoint_labels", []))
    reviews = mask.attrs.get("component_review_statuses", {})
    if (
        not {"subject_body", "swim_bladder"}.issubset(labels)
        or len(set(labels)) != len(labels)
        or not mask.attrs.get("label_schema_id")
        or any(not _approved(reviews.get(name)) for name in labels)
        or not _approved(pose.attrs.get("keypoint_review_status"))
        or not all(is_run_complete(g, legacy_default=False) for g in (mask, pose))
    ):
        raise ValueError(
            "Complete, manually approved training masks and keypoints are required"
        )
    if not set(HEAD_LABELS).issubset(point_labels) or len(set(point_labels)) != len(
        point_labels
    ):
        raise ValueError(
            "Unique named keypoints including all three head landmarks are required"
        )
    frames = np.asarray(crop["frame_indices"][:])
    mask_ids, pose_ids = (_row_ids(g, frames) for g in (mask, pose))
    lookup = {int(row): i for i, row in enumerate(pose_ids)}
    if any(int(row) not in lookup for row in mask_ids):
        raise ValueError("A mask crop row has no matching keypoint row")
    pose_rows = np.array([lookup[int(row)] for row in mask_ids], np.int64)
    n = len(mask_ids)
    images, masks, points = crop["roi_images"], mask["masks_roi"], pose["keypoints_roi"]
    if (
        images.ndim != 3
        or images.dtype != np.uint8
        or images.shape[0] != len(frames)
        or masks.shape != (n, len(labels), *images.shape[1:])
        or masks.dtype != np.uint8
        or points.shape != (len(pose_ids), len(point_labels), 2)
        or points.dtype.kind != "f"
    ):
        raise ValueError(
            "Dense masks, grayscale images, and named keypoint axes disagree"
        )
    size = n * (
        np.prod(images.shape[1:])
        + np.prod(masks.shape[1:])
        + np.prod(points.shape[1:]) * points.dtype.itemsize
    )
    if size > MAX_SOURCE_BYTES:
        raise ValueError(
            "Native snapshot exceeds the bounded 512 MiB source payload limit"
        )
    mask_values = np.asarray(masks[:])
    if np.any(mask_values > 1):
        raise ValueError("Native multilabel masks must contain only binary 0/1 pixels")
    available = np.asarray(mask["available_channels"][:], dtype=bool)
    if available.shape != (len(labels),) or not available.all():
        raise ValueError("Every declared mask channel must be available")
    target_valid = np.ones((n, len(labels)), bool)
    if "target_valid_channels" in mask:
        target_valid = np.asarray(mask["target_valid_channels"][:], dtype=bool)
        if target_valid.shape != (n, len(labels)) or not target_valid.all():
            raise ValueError("Partially supervised mask rows are not accepted")
    source_points = np.asarray(points.oindex[pose_rows])
    arrays = {
        "roi_images": np.asarray(images.oindex[mask_ids]),
        "masks_roi": mask_values,
        "head_keypoints_roi": source_points[
            :, [point_labels.index(name) for name in HEAD_LABELS]
        ],
        "source_keypoints_roi": source_points,
        "source_pose_local_row": pose_rows,
        "source_training_crop_row_ids": mask_ids,
        "frame_indices": frames[mask_ids],
        "source_bbox_norm_coords": np.asarray(
            crop["bbox_norm_coords"].oindex[mask_ids]
        ),
        "target_valid_channels": target_valid,
        "detection_source": np.asarray(crop["detection_source"].oindex[mask_ids]),
    }
    if arrays["source_bbox_norm_coords"].shape != (n, 4) or arrays[
        "detection_source"
    ].shape != (n,):
        raise ValueError("Native crop metadata axes disagree")
    if "roi_coordinates_full" in crop:
        arrays["source_roi_coordinates_full"] = np.asarray(
            crop["roi_coordinates_full"].oindex[mask_ids]
        )
        if arrays["source_roi_coordinates_full"].shape != (n, 2):
            raise ValueError("Native crop origins must have one xy pair per row")
    binding = {
        "source_kind": "native_reviewed_training_masks_v1",
        "source_archive": str(Path(archive).resolve()),
        "source_metadata_mode": "unconsolidated_mutable_snapshot",
        "recording_id": root.attrs["recording_id"],
        "mask_run": mask_run,
        "keypoint_run": keypoint_run,
        "crop_run": crop_run,
        "mask_run_attrs": dict(mask.attrs),
        "keypoint_run_attrs": dict(pose.attrs),
        "crop_run_attrs": dict(crop.attrs),
        "source_keypoint_labels": point_labels,
        "source_array_sha256": {
            name: _sha256_array(value) for name, value in arrays.items()
        },
        "historical_edit_records": "source_run_retained; no invented per-point edit history",
        "coordinate_projection": "identity_roi_copy; sensor origins retained only as sealed source lineage",
    }
    return arrays, labels, binding


def generate_native_mask_review(
    *,
    archive,
    mask_run,
    keypoint_run,
    version,
    apply=False,
    scratch_root=Path("/tmp"),
    refined_mask_run=None,
):
    archive = Path(archive).resolve()
    paths = run_paths(version, native=True)
    if any((archive / path).exists() for path in paths.values()):
        raise FileExistsError("Native annotation version exists; use a fresh version")

    def read_source():
        arrays, labels, binding = read_native_source(
            archive, mask_run=mask_run, keypoint_run=keypoint_run
        )
        if refined_mask_run is not None:
            arrays, binding = use_refined_mask_snapshot(
                archive, refined_mask_run, arrays, labels, binding
            )
        return arrays, labels, binding

    arrays, labels, binding = read_source()

    def check_source(_root=None):
        _, _, current = read_source()
        if current != binding:
            raise ValueError(
                "Native source identity or reviewed content changed during publication"
            )

    check_source()
    if not apply:
        return {
            "status": "planned",
            "recording_id": binding["recording_id"],
            "version": version,
            "source_bindings": binding,
            "row_count": len(arrays["roi_images"]),
            "paths": paths,
        }
    with tempfile.TemporaryDirectory(
        prefix="palette-native-mask-review-", dir=scratch_root
    ) as temp:
        local = Path(temp) / "review.zarr"
        root = zarr.open_group(str(local), mode="w", use_consolidated=False)
        root.attrs.update(zarr_purpose="training", recording_id=binding["recording_id"])
        result = build_review_payload(
            root, arrays, labels, binding, version=version, native=True
        )
        # Release the source buffers before the publisher rechecks their hashes.
        del arrays
        publications = publish_review_payload(
            archive, local, paths, binding, check_source
        )
    check_source()
    with archive_metadata_publication_lock(archive):
        consolidate_metadata_capture_expected_warnings(archive)
    published = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    if any(
        path not in published or not validate_initial_payload(archive / path)["valid"]
        for path in paths.values()
    ):
        raise ValueError("Incomplete native annotation publication")
    tasks = review_tasks(archive, binding["recording_id"], result, version)
    for task in tasks:
        task["dataset_id"] = f"{binding['recording_id']}:native_mask_tail:{version}"
        if task["workflow_kind"] == "keypoints":
            task["title"] = (
                "Review mask-derived tail and retained head, snout, and fin labels (19 points)"
            )
            task["notes"] = (
                "Existing head, snout, and fin landmarks retained by name where finite. Tail11 re-derived as one arc-length sequence. Complete missing labels and review every crop."
            )
    return {
        "status": "generated",
        "archive": str(archive),
        "recording_id": binding["recording_id"],
        "version": version,
        "source_bindings": binding,
        **result,
        "publications": publications,
        "tasks": tasks,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--mask-run", required=True)
    parser.add_argument("--keypoint-run", required=True)
    parser.add_argument(
        "--refined-mask-run",
        help="Use corrected dense masks from an earlier native annotation version",
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--scratch-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = vars(parser.parse_args(argv))
    report = args.pop("report")
    if report.exists():
        raise FileExistsError(report)
    result = generate_native_mask_review(**args)
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({k: result.get(k) for k in ("status", "recording_id", "row_count")})
    )


if __name__ == "__main__":
    main()
