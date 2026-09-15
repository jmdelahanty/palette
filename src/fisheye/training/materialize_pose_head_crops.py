"""Materialize a selector-ineligible 192-pixel pose dataset in a training Zarr.

The new crop run owns its pixels and projected labels together. Existing crop,
keypoint, and refined-keypoint runs are read-only sources; their selectors are
never changed. One invocation publishes one recording and one named version.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import zarr

from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.crop_roi_layout import (
    build_canonical_crop_roi_layout,
    build_crop_roi_create_kwargs,
    crop_roi_layout_attrs,
)
from fisheye.shared.keypoint_motion_authority import (
    keypoint_source_crop_run_from_attributes,
)
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
    open_zarr_group_direct,
)
from fisheye.shared.zarr_run_completion import (
    is_run_complete,
    is_run_complete_in_parent,
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)
from fisheye.training.pose_head_crop_geometry import (
    POSE_HEAD_CROP_RECIPE_ID,
    POSE_HEAD_CROP_SIZE_PX,
    POSE_HEAD_KEYPOINT_LABELS,
    fixed_pose_head_origins,
    project_pose_head_keypoints,
)


POSE_HEAD_DATASET_SCHEMA_ID = "palette.training.pose_head_materialized_crop"
# v1 used an implicit refined-keypoint source. v2 records the source family
# explicitly so reviewed and older selected keypoint runs share one crop recipe.
POSE_HEAD_DATASET_SCHEMA_VERSION = 2
POSE_HEAD_LEGACY_BOX_SCHEMA_VERSION = 3
_COMMAND = "fisheye.training.materialize_pose_head_crops"


def _safe_run_id(value: str) -> str:
    name = str(value).strip()
    if not name or name.startswith(".") or "/" in name or name in {".", ".."}:
        raise ValueError("run ID must be one non-hidden Zarr path component")
    return name


def _array_digest(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(json.dumps(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _required_array(
    group: zarr.Group, name: str, shape_suffix: tuple[int, ...]
) -> np.ndarray:
    if name not in group:
        raise ValueError(f"Required source array missing: {group.path}/{name}")
    values = np.asarray(group[name][:])
    if values.ndim != len(shape_suffix) + 1 or values.shape[1:] != shape_suffix:
        raise ValueError(f"Source array has invalid shape: {group.path}/{name}")
    return values


@dataclass(frozen=True)
class PoseHeadCropPlan:
    source_keypoint_group: str
    source_box_mode: str
    source_detection_run: str | None
    source_keypoint_run: str
    source_crop_run: str
    source_keypoint_rows: np.ndarray
    source_crop_rows: np.ndarray
    source_training_rows: np.ndarray
    source_frame_indices: np.ndarray
    source_boxes_img_xyxy: np.ndarray
    origins_xy: np.ndarray
    keypoints_roi: np.ndarray
    keypoint_visibility: np.ndarray
    source_labels: tuple[str, ...]
    source_skeleton_id: str
    raw_image_shape: tuple[int, int, int]
    source_h5_fingerprint: str | None

    @property
    def row_count(self) -> int:
        return len(self.source_keypoint_rows)

    def summary(self) -> dict[str, Any]:
        sides = self.source_boxes_img_xyxy[:, 2:] - self.source_boxes_img_xyxy[:, :2]
        longest = np.max(sides, axis=1)
        result = {
            "row_count": self.row_count,
            "crop_size_px": POSE_HEAD_CROP_SIZE_PX,
            "source_keypoint_run": self.source_keypoint_run,
            "source_keypoint_group": self.source_keypoint_group,
            "source_crop_run": self.source_crop_run,
            "source_skeleton_id": self.source_skeleton_id,
            "source_labels": list(self.source_labels),
            "target_labels": list(POSE_HEAD_KEYPOINT_LABELS),
            "keypoints_outside_crop": int(
                np.count_nonzero(self.keypoint_visibility == 0)
            ),
            "box_longer_side_px_percentiles": {
                "p5": float(np.percentile(longest, 5)),
                "p50": float(np.percentile(longest, 50)),
                "p95": float(np.percentile(longest, 95)),
            },
            "source_boxes_sha256": _array_digest(self.source_boxes_img_xyxy),
            "source_keypoint_rows_sha256": _array_digest(self.source_keypoint_rows),
        }
        if self.source_box_mode != "bound_crop_img_xyxy":
            result["source_box_mode"] = self.source_box_mode
            result["source_detection_run"] = self.source_detection_run
        return result


def build_pose_head_crop_plan(
    root: zarr.Group,
    *,
    source_keypoint_group: str = "refined_keypoints_runs",
    source_box_mode: str = "bound_crop_img_xyxy",
    source_keypoint_run: str,
    source_crop_run: str,
) -> PoseHeadCropPlan:
    """Preflight exact source row, skeleton, box, and frame bindings."""

    keypoint_id = _safe_run_id(source_keypoint_run)
    crop_id = _safe_run_id(source_crop_run)
    if source_keypoint_group not in {"refined_keypoints_runs", "keypoints_runs"}:
        raise ValueError("Unsupported source keypoint run family")
    if source_box_mode not in {
        "bound_crop_img_xyxy",
        "legacy_authoritative_refined_detection_norm_cxcywh",
    }:
        raise ValueError("Unsupported source box mode")
    if str(root.attrs.get("zarr_purpose") or "").lower() != "training":
        raise ValueError("Source must be a training-purpose Zarr")
    keypoint_parent = root[source_keypoint_group]
    crop_parent = root["crop_runs"]
    keypoints = keypoint_parent[keypoint_id]
    crop = crop_parent[crop_id]
    if not is_run_complete_in_parent(
        keypoint_parent, keypoints
    ) or not is_run_complete_in_parent(crop_parent, crop):
        raise ValueError("Both source runs must be complete")
    if keypoint_source_crop_run_from_attributes(keypoints.attrs) != crop_id:
        raise ValueError("Keypoint run is not bound to the requested source crop run")
    labels = tuple(str(label) for label in keypoints.attrs.get("keypoint_labels", ()))
    source_skeleton_id = str(keypoints.attrs.get("skeleton_id") or "")
    if not labels or not source_skeleton_id:
        raise ValueError(
            "Source keypoint run must declare labels and skeleton identity"
        )
    if (
        "raw_video/images_full" not in root
        or "raw_video/original_frame_indices" not in root
    ):
        raise ValueError(
            "Training archive needs embedded full frames and frame identities"
        )
    images = root["raw_video/images_full"]
    if len(images.shape) != 3 or np.dtype(images.dtype) != np.dtype("uint8"):
        raise ValueError(
            "raw_video/images_full must be [frame,height,width] uint8 mono"
        )
    frame_shape = tuple(int(value) for value in images.shape)
    n_keypoints = int(keypoints["keypoints_img"].shape[0])
    success_name = (
        "usable_keypoints"
        if source_keypoint_group == "refined_keypoints_runs"
        else "detection_success"
    )
    usable = _required_array(keypoints, success_name, ())
    if usable.shape != (n_keypoints,) or usable.dtype != np.dtype(bool):
        raise ValueError(f"{success_name} must be one boolean per source row")
    selected = np.flatnonzero(usable).astype(np.int64)
    if not len(selected):
        raise ValueError("No usable source keypoint rows")
    points_img = _required_array(keypoints, "keypoints_img", (len(labels), 2))[selected]
    if any(labels.count(label) != 1 for label in POSE_HEAD_KEYPOINT_LABELS):
        raise ValueError(
            "Source skeleton must contain each required label exactly once"
        )
    if not np.isfinite(
        points_img[:, [labels.index(label) for label in POSE_HEAD_KEYPOINT_LABELS], :]
    ).all():
        raise ValueError("Usable source rows contain nonfinite required points")
    if source_box_mode == "bound_crop_img_xyxy":
        source_crop_rows = _required_array(keypoints, "source_crop_row_ids", ())[
            selected
        ].astype(np.int64)
        n_crop = int(crop["bbox_img_xyxy"].shape[0])
    else:
        keypoint_local_frames = _required_array(keypoints, "frame_indices", ())
        keypoint_detection_ids = _required_array(keypoints, "detection_indices", ())
        crop_local_frames = _required_array(crop, "frame_indices", ())
        crop_detection_ids = _required_array(crop, "detection_indices", ())
        if (
            keypoint_local_frames.shape != keypoint_detection_ids.shape
            or crop_local_frames.shape != crop_detection_ids.shape
        ):
            raise ValueError("Legacy frame and detection row axes disagree")
        crop_keys = list(zip(crop_local_frames.tolist(), crop_detection_ids.tolist()))
        if len(set(crop_keys)) != len(crop_keys):
            raise ValueError("Legacy source crop frame/detection keys are not unique")
        crop_lookup = dict(zip(crop_keys, range(len(crop_keys))))
        try:
            source_crop_rows = np.asarray(
                [
                    crop_lookup[
                        (
                            int(keypoint_local_frames[row]),
                            int(keypoint_detection_ids[row]),
                        )
                    ]
                    for row in selected
                ],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(
                "Legacy keypoint row lacks a matching source crop frame/detection key"
            ) from exc
        n_crop = len(crop_keys)
    if np.any(source_crop_rows < 0) or np.any(source_crop_rows >= n_crop):
        raise ValueError("Keypoint source_crop_row_ids leave the bound crop run")
    if len(np.unique(source_crop_rows)) != len(source_crop_rows):
        raise ValueError("Multiple usable keypoint rows bind the same source crop row")
    source_detection_run: str | None = None
    if source_box_mode == "bound_crop_img_xyxy":
        boxes = _required_array(crop, "bbox_img_xyxy", (4,))[source_crop_rows]
        source_training_rows = _required_array(crop, "source_training_row_indices", ())[
            source_crop_rows
        ].astype(np.int64)
    else:
        source_detection_run = str(keypoints.attrs.get("source_refined_run") or "")
        if (
            not source_detection_run
            or crop.attrs.get("source_refined_run") != source_detection_run
        ):
            raise ValueError(
                "Legacy crop and keypoints disagree on refined detection source"
            )
        detect_parent = root["refined_detect_runs"]
        if detect_parent.attrs.get("authoritative_run") != source_detection_run:
            raise ValueError("Legacy refined detection source is not authoritative")
        detection = detect_parent[source_detection_run]
        if not is_run_complete_in_parent(detect_parent, detection):
            raise ValueError("Legacy refined detection run is incomplete")
        table = detection["instances"] if "instances" in detection else detection
        refined_ids = _required_array(table, "refined_row_ids", ())
        detection_frames = _required_array(table, "frame_indices", ())
        if len(set(refined_ids.tolist())) != len(refined_ids):
            raise ValueError("Legacy refined detection row IDs are not unique")
        refined_lookup = {
            int(row_id): index for index, row_id in enumerate(refined_ids)
        }
        crop_refined_ids = _required_array(crop, "source_refined_row_ids", ())[
            source_crop_rows
        ]
        try:
            detection_rows = np.asarray(
                [refined_lookup[int(row_id)] for row_id in crop_refined_ids],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(
                "Legacy source crop lacks a matching refined detection row"
            ) from exc
        if not np.array_equal(
            detection_frames[detection_rows],
            _required_array(crop, "frame_indices", ())[source_crop_rows],
        ):
            raise ValueError(
                "Legacy refined detection and crop frame identities disagree"
            )
        norm_boxes = _required_array(table, "bbox_norm_coords", (4,))[detection_rows]
        if (
            not np.isfinite(norm_boxes).all()
            or np.any(norm_boxes[:, 2:] <= 0)
            or np.any(norm_boxes < 0)
            or np.any(norm_boxes > 1)
        ):
            raise ValueError("Legacy normalized refined boxes are malformed")
        _, source_height, source_width = frame_shape
        centers = norm_boxes[:, :2] * (source_width, source_height)
        sizes = norm_boxes[:, 2:] * (source_width, source_height)
        boxes = np.concatenate((centers - sizes * 0.5, centers + sizes * 0.5), axis=1)
        source_training_rows = _required_array(crop, "frame_indices", ())[
            source_crop_rows
        ].astype(np.int64)
    original_frames = _required_array(root["raw_video"], "original_frame_indices", ())
    if np.any(source_training_rows < 0) or np.any(
        source_training_rows >= frame_shape[0]
    ):
        raise ValueError("Crop source_training_row_indices leave the raw frame axis")
    source_frames = _required_array(crop, "source_frame_indices", ())[
        source_crop_rows
    ].astype(np.int64)
    keypoint_frames = _required_array(keypoints, "source_frame_indices", ())[
        selected
    ].astype(np.int64)
    if not np.array_equal(source_frames, keypoint_frames) or not np.array_equal(
        source_frames, original_frames[source_training_rows]
    ):
        raise ValueError("Source crop, keypoint, and raw frame identities disagree")
    source_origins = _required_array(crop, "roi_coordinates_full", (2,))[
        source_crop_rows
    ]
    source_roi = _required_array(keypoints, "keypoints_roi", (len(labels), 2))[selected]
    required_indices = [labels.index(label) for label in POSE_HEAD_KEYPOINT_LABELS]
    if not np.allclose(
        points_img[:, required_indices, :],
        source_roi[:, required_indices, :] + source_origins[:, None, :],
        rtol=0,
        atol=1e-3,
    ):
        raise ValueError(
            "Source keypoint sensor coordinates disagree with crop offsets"
        )
    origins = fixed_pose_head_origins(
        boxes, frame_shape_hw=(frame_shape[1], frame_shape[2])
    )
    projected, visibility = project_pose_head_keypoints(
        points_img, source_labels=labels, origins_xy=origins
    )
    return PoseHeadCropPlan(
        source_keypoint_group=source_keypoint_group,
        source_box_mode=source_box_mode,
        source_detection_run=source_detection_run,
        source_keypoint_run=keypoint_id,
        source_crop_run=crop_id,
        source_keypoint_rows=selected,
        source_crop_rows=source_crop_rows,
        source_training_rows=source_training_rows,
        source_frame_indices=source_frames,
        source_boxes_img_xyxy=boxes.astype(np.float32),
        origins_xy=origins.astype(np.int32),
        keypoints_roi=projected.astype(np.float32),
        keypoint_visibility=visibility,
        source_labels=labels,
        source_skeleton_id=source_skeleton_id,
        raw_image_shape=frame_shape,
        source_h5_fingerprint=root.attrs.get("source_h5_fingerprint"),
    )


def _materialize_pixels(
    source_images: zarr.Array, plan: PoseHeadCropPlan
) -> np.ndarray:
    """Read each physical source frame chunk once; output is at most 37 KiB/row."""

    size = POSE_HEAD_CROP_SIZE_PX
    pixels = np.empty((plan.row_count, size, size), dtype=np.uint8)
    source_chunk_len = int(source_images.chunks[0])
    if source_chunk_len <= 0:
        raise ValueError("Invalid source frame chunk length")
    chunk_ids = plan.source_training_rows // source_chunk_len
    for chunk_id in np.unique(chunk_ids):
        chunk_start = int(chunk_id) * source_chunk_len
        chunk_end = min(chunk_start + source_chunk_len, plan.raw_image_shape[0])
        source_chunk = np.asarray(source_images[chunk_start:chunk_end], dtype=np.uint8)
        if source_chunk.shape != (chunk_end - chunk_start, *plan.raw_image_shape[1:]):
            raise RuntimeError(
                "Source frame chunk shape changed during materialization"
            )
        for output_row in np.flatnonzero(chunk_ids == chunk_id):
            local_frame = int(plan.source_training_rows[output_row]) - chunk_start
            x, y = (int(value) for value in plan.origins_xy[output_row])
            crop = source_chunk[local_frame, y : y + size, x : x + size]
            if crop.shape != (size, size):
                raise RuntimeError("Crop geometry escaped source frame")
            pixels[output_row] = crop
    return pixels


def _write_local_run(
    root: zarr.Group,
    *,
    run_id: str,
    plan: PoseHeadCropPlan,
    pixels: np.ndarray,
) -> zarr.Group:
    parent = require_runs_parent(root, "crop_runs")
    run = parent.create_group(run_id)
    mark_run_started(run, run_name=run_id, stage="crop")
    layout = build_canonical_crop_roi_layout(
        total_rois=plan.row_count, preferred_chunk_len=32
    )
    roi = run.create_array(
        "roi_images",
        **build_crop_roi_create_kwargs(
            total_rois=plan.row_count,
            roi_sz=(POSE_HEAD_CROP_SIZE_PX, POSE_HEAD_CROP_SIZE_PX),
            layout=layout,
            overwrite=False,
        ),
    )
    roi[:] = pixels
    array_values = {
        "keypoints_roi": plan.keypoints_roi,
        "keypoint_visibility": plan.keypoint_visibility,
        "roi_coordinates_full": plan.origins_xy,
        "source_keypoint_row_ids": plan.source_keypoint_rows,
        "source_crop_row_ids": plan.source_crop_rows,
        "source_training_row_indices": plan.source_training_rows,
        "source_frame_indices": plan.source_frame_indices,
        "bbox_img_xyxy": plan.source_boxes_img_xyxy,
    }
    for name, values in array_values.items():
        run.create_array(
            name,
            data=np.ascontiguousarray(values),
            chunks=(min(1024, plan.row_count), *values.shape[1:]),
        )
    run.attrs.update(
        {
            "schema_id": POSE_HEAD_DATASET_SCHEMA_ID,
            "schema_version": (
                POSE_HEAD_LEGACY_BOX_SCHEMA_VERSION
                if plan.source_box_mode != "bound_crop_img_xyxy"
                else POSE_HEAD_DATASET_SCHEMA_VERSION
            ),
            "status": "completed",
            "stage_selector_eligible": False,
            "crop_storage_mode": "materialized",
            "source_pixels": "raw_video/images_full",
            "source_pixel_range": "uint8_0_255",
            "model_input_preprocessing": "replicate_mono_to_rgb_then_scale_1_over_255_no_mean_std_v1",
            "roi_size": [POSE_HEAD_CROP_SIZE_PX, POSE_HEAD_CROP_SIZE_PX],
            "crop_recipe": {
                "id": POSE_HEAD_CROP_RECIPE_ID,
                "size_px": POSE_HEAD_CROP_SIZE_PX,
                "size_selection": "user_fixed_override_of_p95_1p3_ceil32_v1",
                "center_rule": "int_truncate_toward_zero_of_xyxy_midpoint",
                "origin_rule": "clamp_center_minus_half_size_to_frame",
                "padding": "none",
                "resize": "none",
                "rotation": "none",
            },
            "pose_schema": {
                "name": "traditional_v1",
                "skeleton_id": "pose_schema:traditional_v1",
                "keypoint_labels": list(POSE_HEAD_KEYPOINT_LABELS),
                "source": "configs/fisheye/pose_schemas/traditional_v1.json",
            },
            "source_bindings": {
                "source_keypoint_group": plan.source_keypoint_group,
                "source_keypoint_run": plan.source_keypoint_run,
                **(
                    {
                        "source_box_mode": plan.source_box_mode,
                        "source_detection_run": plan.source_detection_run,
                    }
                    if plan.source_box_mode != "bound_crop_img_xyxy"
                    else {}
                ),
                "source_crop_run": plan.source_crop_run,
                "source_skeleton_id": plan.source_skeleton_id,
                "source_keypoint_labels": list(plan.source_labels),
                "source_h5_fingerprint": plan.source_h5_fingerprint,
                "source_images_path": "raw_video/images_full",
                "source_images_shape": list(plan.raw_image_shape),
            },
            "pixel_sha256": _array_digest(pixels),
            "keypoints_roi_sha256": _array_digest(plan.keypoints_roi),
            "keypoint_visibility_sha256": _array_digest(plan.keypoint_visibility),
            "summary_statistics": plan.summary(),
            **crop_roi_layout_attrs(layout),
        }
    )
    mark_run_complete(
        run,
        parent_group=parent,
        run_name=run_id,
        run_provenance=build_writer_run_provenance(
            command=_COMMAND,
            params={
                "crop_recipe": POSE_HEAD_CROP_RECIPE_ID,
                "size_px": POSE_HEAD_CROP_SIZE_PX,
            },
            input_run_ids={
                "source_crop_run": plan.source_crop_run,
                "source_keypoint_run": f"{plan.source_keypoint_group}/{plan.source_keypoint_run}",
                **(
                    {"source_detection_run": plan.source_detection_run}
                    if plan.source_detection_run is not None
                    else {}
                ),
            },
        ),
    )
    return run


def _validate_run(path: Path) -> dict[str, Any]:
    try:
        run = zarr.open_group(str(path), mode="r", use_consolidated=False)
        n = int(run["roi_images"].shape[0])
        required = (
            "keypoints_roi",
            "keypoint_visibility",
            "roi_coordinates_full",
            "source_keypoint_row_ids",
            "source_crop_row_ids",
            "source_training_row_indices",
            "source_frame_indices",
            "bbox_img_xyxy",
        )
        valid = (
            run.attrs.get("schema_id") == POSE_HEAD_DATASET_SCHEMA_ID
            and run.attrs.get("schema_version")
            in (
                1,
                POSE_HEAD_DATASET_SCHEMA_VERSION,
                POSE_HEAD_LEGACY_BOX_SCHEMA_VERSION,
            )
            and run.attrs.get("stage_selector_eligible") is False
            and is_run_complete(run, legacy_default=False)
            and tuple(run["roi_images"].shape) == (n, 192, 192)
            and all(int(run[name].shape[0]) == n for name in required)
            and tuple(run["keypoints_roi"].shape) == (n, 3, 2)
            and tuple(run["keypoint_visibility"].shape) == (n, 3)
            and _array_digest(np.asarray(run["roi_images"][:]))
            == run.attrs.get("pixel_sha256")
            and _array_digest(np.asarray(run["keypoints_roi"][:]))
            == run.attrs.get("keypoints_roi_sha256")
            and _array_digest(np.asarray(run["keypoint_visibility"][:]))
            == run.attrs.get("keypoint_visibility_sha256")
        )
        return {"valid": bool(valid), "row_count": n}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def materialize_pose_head_crops(
    *,
    zarr_path: Path,
    source_keypoint_group: str = "refined_keypoints_runs",
    source_box_mode: str = "bound_crop_img_xyxy",
    source_keypoint_run: str,
    source_crop_run: str,
    run_id: str,
    scratch_root: Path,
    apply: bool = False,
) -> dict[str, Any]:
    archive = zarr_path.expanduser().resolve()
    candidate = _safe_run_id(run_id)
    if not archive.is_dir():
        raise FileNotFoundError(archive)
    source = open_zarr_group_direct(archive, mode="r")
    plan = build_pose_head_crop_plan(
        source,
        source_keypoint_group=source_keypoint_group,
        source_box_mode=source_box_mode,
        source_keypoint_run=source_keypoint_run,
        source_crop_run=source_crop_run,
    )
    result = {"archive": str(archive), "run_id": candidate, **plan.summary()}
    target = archive / "crop_runs" / candidate
    if target.exists():
        raise FileExistsError(target)
    if not apply:
        return {"status": "planned", **result}
    scratch = scratch_root.expanduser().resolve()
    if not scratch.is_dir():
        raise FileNotFoundError(scratch)
    with tempfile.TemporaryDirectory(
        prefix="palette-pose-head-", dir=scratch
    ) as temporary:
        local_archive = Path(temporary) / "training.zarr"
        local_root = zarr.open_group(
            str(local_archive), mode="w", zarr_format=3, use_consolidated=False
        )
        local_root.attrs["zarr_purpose"] = "training"
        pixels = _materialize_pixels(source["raw_video/images_full"], plan)
        _write_local_run(local_root, run_id=candidate, plan=plan, pixels=pixels)
        local_run = local_archive / "crop_runs" / candidate

        def prepare_parents(current_root: zarr.Group) -> tuple[zarr.Group, ...]:
            return (current_root, require_runs_parent(current_root, "crop_runs"))

        def complete_run(
            _root: zarr.Group, parent: zarr.Group, run: zarr.Group
        ) -> None:
            run.attrs["stage_selector_eligible"] = False
            mark_run_complete(
                run,
                parent_group=parent,
                run_name=candidate,
                run_provenance=run.attrs.get("run_provenance"),
            )

        def verify_unselected(current_root: zarr.Group) -> None:
            parent = current_root["crop_runs"]
            if any(
                parent.attrs.get(name) == candidate
                for name in ("latest", "latest_complete", "authoritative_run")
            ):
                raise RuntimeError("Pose-head crop run became selected unexpectedly")

        publication = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=archive,
                local_run_path=local_run,
                target_run_path=target,
                run_name=candidate,
                lock_suffix="pose_head_crop_materialization",
                publish_schema_id=POSE_HEAD_DATASET_SCHEMA_ID,
                policy="local_materialize_then_atomic_selector_ineligible_import_v1",
                rollback_policy="retain_failed_selector_ineligible_child_v1",
                content_checksum=True,
            ),
            copy_backend="python",
            validate_run=_validate_run,
            prepare_parents=prepare_parents,
            complete_run=complete_run,
            verify_pointers=verify_unselected,
            payload_metadata={
                "source_crop_run": plan.source_crop_run,
                "source_keypoint_run": f"{plan.source_keypoint_group}/{plan.source_keypoint_run}",
            },
        )
    with archive_metadata_publication_lock(archive):
        consolidate_metadata_capture_expected_warnings(archive)
    final = _validate_run(target)
    if not final["valid"]:
        raise RuntimeError(f"Published pose-head crop run failed validation: {final}")
    published = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    if (
        f"crop_runs/{candidate}" not in published
        or published[f"crop_runs/{candidate}"].attrs.get("stage_selector_eligible")
        is not False
        or any(
            published["crop_runs"].attrs.get(name) == candidate
            for name in ("latest", "latest_complete", "authoritative_run")
        )
    ):
        raise RuntimeError("Consolidated metadata omits the unselected published crop")
    return {"status": "materialized", **result, "publication": publication}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument(
        "--source-keypoint-group",
        choices=("refined_keypoints_runs", "keypoints_runs"),
        default="refined_keypoints_runs",
    )
    parser.add_argument(
        "--source-box-mode",
        choices=(
            "bound_crop_img_xyxy",
            "legacy_authoritative_refined_detection_norm_cxcywh",
        ),
        default="bound_crop_img_xyxy",
    )
    parser.add_argument("--source-keypoint-run", required=True)
    parser.add_argument("--source-crop-run", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scratch-root", type=Path, default=Path("/tmp"))
    parser.add_argument(
        "--apply", action="store_true", help="Publish the unselected run"
    )
    args = parser.parse_args(argv)
    result = materialize_pose_head_crops(
        zarr_path=args.zarr_path,
        source_keypoint_group=args.source_keypoint_group,
        source_box_mode=args.source_box_mode,
        source_keypoint_run=args.source_keypoint_run,
        source_crop_run=args.source_crop_run,
        run_id=args.run_id,
        scratch_root=args.scratch_root,
        apply=args.apply,
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "publication"},
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
