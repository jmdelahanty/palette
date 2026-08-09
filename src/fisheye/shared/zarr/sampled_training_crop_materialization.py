"""Materialize reviewed sampled detections from embedded full-frame pixels.

This provider is deliberately separate from recording-level crop-v2.  A
sampled training archive owns a compact local frame axis, while
``raw_video/original_frame_indices`` preserves acquisition-frame identity.
Only positive refined-detection rows produce crops; reviewed-negative frames
remain first-class frame supervision and never create placeholder pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.shared.crop_roi_layout import (
    build_canonical_crop_roi_layout,
    build_crop_roi_create_kwargs,
    crop_roi_layout_attrs,
)
from fisheye.shared.row_source_signature import (
    RowSourceSignatureSpec,
    build_row_source_signatures,
)
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr.array_contracts import (
    DETECTION_SOURCE_ACQUISITION_FRAME_INDEX_V1,
)
from fisheye.shared.zarr.array_factory import create_array_from_plan
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.crop_schema import (
    CROP_GEOMETRY_SCHEMA_V1,
    CropDimensions,
    derive_crop_placement_geometry,
)
from fisheye.shared.zarr.crop_storage import plan_crop_geometry_storage
from fisheye.shared.zarr.detect_frame_decisions import (
    FRAME_REVIEW_CONTRACT_ATTR,
    FRAME_REVIEW_CONTRACT_ID,
)
from fisheye.shared.zarr.detection_schema import (
    derive_canonical_detection_geometry,
)
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import TRAINING_IMMUTABLE_V1
from fisheye.shared.zarr.training_crop_materialization import (
    SAMPLED_TRAINING_CROP_GEOMETRY_SCHEMA_ID,
    SAMPLED_TRAINING_CROP_GEOMETRY_SCHEMA_VERSION,
    SAMPLED_TRAINING_IMAGES_FULL_MATERIALIZATION_PROVIDER,
    TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE,
    TRAINING_CROP_MATERIALIZATION_PROVIDERS,
    TRAINING_CROP_MATERIALIZATION_SCHEMA_ID,
    build_training_crop_materialization_binding,
)
from fisheye.shared.zarr_run_completion import (
    is_run_complete,
    mark_run_complete,
    mark_run_started,
)
from fisheye.training.detection_frame_supervision import (
    DetectionFrameSupervisionPlan,
    build_detection_frame_supervision_plan,
)

SAMPLED_TRAINING_CROP_WRITER_SCHEMA_ID = (
    "palette.sampled_training_images_full_crop_writer"
)
SAMPLED_TRAINING_CROP_WRITER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SampledTrainingCropPlan:
    """Complete immutable crop rows derived from one reviewed sampled source."""

    refined_run_id: str
    dimensions: CropDimensions
    arrays: Mapping[str, np.ndarray]
    source_frame_indices: np.ndarray
    source_images_shape: tuple[int, int, int]
    source_frame_decision_path: str
    source_frame_decision_digest: str
    row_signature_spec: RowSourceSignatureSpec
    supervision: DetectionFrameSupervisionPlan
    roi_size_wh: tuple[int, int]

    @property
    def row_count(self) -> int:
        return self.dimensions.n_instances


def _safe_run_id(value: str, *, label: str) -> str:
    candidate = str(value).strip()
    if not candidate or "/" in candidate or candidate in {".", ".."}:
        raise ValueError(f"{label} must be one nonempty Zarr path component.")
    return candidate


def _values(node: Any) -> np.ndarray:
    return np.asarray(node[:])


def _require_integer_vector(
    table: zarr.Group,
    name: str,
    *,
    row_count: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
    values = _values(table[name])
    if values.shape != (row_count,) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must be one integer vector of length {row_count}.")
    return np.asarray(values, dtype=dtype)


def build_sampled_training_crop_plan(
    root: zarr.Group,
    *,
    refined_run_id: str,
    roi_size_wh: tuple[int, int] = (348, 348),
) -> SampledTrainingCropPlan:
    """Bind reviewed rows and derive exact local/acquisition crop geometry."""

    run_id = _safe_run_id(refined_run_id, label="refined_run_id")
    if str(root.attrs.get("zarr_purpose") or "").strip().lower() != "training":
        raise ValueError("Sampled images_full crops require a training-purpose Zarr.")
    if (
        not isinstance(roi_size_wh, tuple)
        or len(roi_size_wh) != 2
        or any(type(value) is not int or value <= 0 for value in roi_size_wh)
    ):
        raise ValueError("roi_size_wh must be (positive width, positive height).")
    roi_width, roi_height = roi_size_wh

    raw = root.get("raw_video")
    if raw is None or "images_full" not in raw or "original_frame_indices" not in raw:
        raise ValueError(
            "Sampled provider requires raw_video/images_full and original_frame_indices."
        )
    images = raw["images_full"]
    if len(images.shape) != 3 or np.dtype(images.dtype) != np.dtype(np.uint8):
        raise ValueError("raw_video/images_full must have exact uint8 shape [F,H,W].")
    n_frames, source_height, source_width = (int(value) for value in images.shape)
    if n_frames <= 0 or source_height <= 0 or source_width <= 0:
        raise ValueError("raw_video/images_full dimensions must be positive.")
    original_frames = _values(raw["original_frame_indices"])
    if original_frames.shape != (n_frames,) or not np.issubdtype(
        original_frames.dtype, np.integer
    ):
        raise ValueError(
            "raw_video/original_frame_indices must be one integer per sampled frame."
        )
    original_frames = np.asarray(original_frames, dtype=np.int64)
    if np.any(original_frames < 0) or (
        original_frames.size > 1 and np.any(np.diff(original_frames) <= 0)
    ):
        raise ValueError(
            "raw_video/original_frame_indices must be nonnegative and strictly increasing."
        )

    refined_parent = root.get("refined_detect_runs")
    if refined_parent is None or run_id not in refined_parent:
        raise ValueError(f"Refined detection review run not found: {run_id!r}.")
    refined = refined_parent[run_id]
    legacy_status_complete = str(refined.attrs.get("status") or "").strip().lower() in {
        "complete",
        "completed",
    }
    if not legacy_status_complete and not is_run_complete(
        refined, legacy_default=False
    ):
        raise ValueError("Refined detection review run must be complete.")
    if refined.attrs.get(FRAME_REVIEW_CONTRACT_ATTR) != FRAME_REVIEW_CONTRACT_ID:
        raise ValueError(
            "Refined detection review must declare the frame-review contract."
        )
    table = refined.get("instances")
    if table is None:
        table = refined
    required = ("frame_indices", "bbox_norm_coords", "instance_key", "refined_row_ids")
    missing = [name for name in required if name not in table]
    if missing:
        raise ValueError(f"Refined detection review lacks required arrays: {missing}.")

    raw_boxes = _values(table["bbox_norm_coords"])
    if raw_boxes.ndim != 2 or tuple(raw_boxes.shape[1:]) != (4,):
        raise ValueError("Refined bbox_norm_coords must have shape [N,4].")
    row_count = int(raw_boxes.shape[0])
    frames = _require_integer_vector(
        table,
        "frame_indices",
        row_count=row_count,
        dtype=np.dtype(np.int64),
    )
    keys_node = table["instance_key"]
    if np.dtype(keys_node.dtype) != np.dtype(np.uint64):
        raise ValueError("Refined instance_key must have exact uint64 dtype.")
    keys = np.asarray(keys_node[:], dtype=np.uint64).reshape(-1)
    if keys.shape != (row_count,) or np.unique(keys).shape[0] != row_count:
        raise ValueError("Refined instance_key must be one unique key per row.")
    refined_row_ids = _require_integer_vector(
        table,
        "refined_row_ids",
        row_count=row_count,
        dtype=np.dtype(np.int64),
    )
    if np.any(refined_row_ids < 0) or np.unique(refined_row_ids).shape[0] != row_count:
        raise ValueError("refined_row_ids must be unique and nonnegative.")
    if row_count:
        expected_order = np.lexsort((keys, frames))
        if not np.array_equal(expected_order, np.arange(row_count, dtype=np.int64)):
            raise ValueError(
                "Refined rows must be ordered by frame_indices then instance_key."
            )

    bbox_path = f"refined_detect_runs/{run_id}/instances/bbox_norm_coords"
    frame_path = f"refined_detect_runs/{run_id}/instances/frame_indices"
    supervision = build_detection_frame_supervision_plan(
        root,
        bbox_path=bbox_path,
        frame_indices_path=frame_path,
        n_frames=n_frames,
    )
    if (
        supervision.source_decision_run_path is None
        or supervision.source_decision_digest is None
        or supervision.frame_count != n_frames
        or not np.array_equal(
            supervision.source_frame_indices,
            np.arange(n_frames, dtype=np.int64),
        )
    ):
        raise ValueError(
            "Sampled crop publication requires complete bound positive/negative frame decisions."
        )
    source_rows = supervision.source_instance_row_indices
    frames = np.asarray(
        supervision.instance_output_frame_indices, dtype=np.int64
    ).reshape(-1)
    keys = keys[source_rows]
    refined_row_ids = refined_row_ids[source_rows]
    boxes = np.asarray(raw_boxes[source_rows], dtype=np.float32)
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= 0):
        raise ValueError(
            "Refined boxes must remain finite and positive after float32 conversion."
        )
    if row_count:
        half = np.float32(0.5)
        if np.any(boxes[:, :2] - boxes[:, 2:] * half < 0) or np.any(
            boxes[:, :2] + boxes[:, 2:] * half > 1
        ):
            raise ValueError(
                "Refined boxes must be fully contained in normalized source coordinates."
            )
    bbox_img, centers = derive_canonical_detection_geometry(
        boxes,
        source_width=source_width,
        source_height=source_height,
    )
    sizes = np.repeat(
        np.asarray([[roi_width, roi_height]], dtype=np.int32),
        row_count,
        axis=0,
    )
    coordinates, source_crop, bbox_roi = derive_crop_placement_geometry(
        centers,
        bbox_img,
        sizes,
    )
    acquisition_frames = original_frames[frames]
    arrays: dict[str, np.ndarray] = {
        "instance_key": np.array(keys, copy=True, order="C"),
        "source_refined_row_ids": np.array(refined_row_ids, copy=True, order="C"),
        "frame_indices": np.array(frames, copy=True, order="C"),
        "source_acquisition_frame_index": np.array(
            acquisition_frames, copy=True, order="C"
        ),
        "frame_row_offsets": np.array(
            supervision.frame_row_offsets, copy=True, order="C"
        ),
        "bbox_norm_coords": np.array(boxes, copy=True, order="C"),
        "bbox_img_xyxy": np.array(bbox_img, copy=True, order="C"),
        "centers_img_xy": np.array(centers, copy=True, order="C"),
        "roi_coordinates_full": np.array(coordinates, copy=True, order="C"),
        "roi_sizes_full": np.array(sizes, copy=True, order="C"),
        "source_crop_xywh": np.array(source_crop, copy=True, order="C"),
        "bbox_roi_xyxy": np.array(bbox_roi, copy=True, order="C"),
    }
    signatures = build_row_source_signatures(
        stage="sampled_training_crop_materialization",
        instance_keys=keys,
        content_components={
            "source_refined_row_ids": refined_row_ids,
            "frame_indices": frames,
            "bbox_norm_coords": boxes,
            "roi_coordinates_full": coordinates,
            "roi_sizes_full": sizes,
        },
        compatibility_context={
            "geometry_schema_id": SAMPLED_TRAINING_CROP_GEOMETRY_SCHEMA_ID,
            "geometry_schema_version": SAMPLED_TRAINING_CROP_GEOMETRY_SCHEMA_VERSION,
            "provider": SAMPLED_TRAINING_IMAGES_FULL_MATERIALIZATION_PROVIDER,
            "source_refined_detect_run": run_id,
            "source_frame_decision_digest": supervision.source_decision_digest,
            "source_images_path": "raw_video/images_full",
            "source_images_shape": [n_frames, source_height, source_width],
            "source_original_frame_indices_sha256": sha256_array(original_frames),
            "roi_size_wh": [roi_width, roi_height],
            "padding_mode": "zero_outside_source_frame",
        },
    )
    arrays["source_row_signature"] = signatures.signatures
    dimensions = CropDimensions(
        n_frames=n_frames,
        n_instances=row_count,
        source_width=source_width,
        source_height=source_height,
    )
    return SampledTrainingCropPlan(
        refined_run_id=run_id,
        dimensions=dimensions,
        arrays=arrays,
        source_frame_indices=np.array(acquisition_frames, copy=True, order="C"),
        source_images_shape=(n_frames, source_height, source_width),
        source_frame_decision_path=supervision.source_decision_run_path,
        source_frame_decision_digest=supervision.source_decision_digest,
        row_signature_spec=signatures.spec,
        supervision=supervision,
        roi_size_wh=(roi_width, roi_height),
    )


def write_by_physical_units(destination: Any, values: np.ndarray, *, plan: Any) -> None:
    if plan.chunk_shape is None:
        raise ValueError("Sampled crop arrays cannot be scalars.")
    unit_rows = int(
        plan.shard_shape[0] if plan.shard_shape is not None else plan.chunk_shape[0]
    )
    trailing = (slice(None),) * (values.ndim - 1)
    for start in range(0, int(values.shape[0]), unit_rows):
        stop = min(start + unit_rows, int(values.shape[0]))
        selection = (slice(start, stop), *trailing)
        destination[selection] = values[selection]


def zero_padded_crop(
    frame: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    output = np.zeros((height, width), dtype=np.uint8)
    frame_height, frame_width = frame.shape
    source_x0 = max(0, x)
    source_y0 = max(0, y)
    source_x1 = min(frame_width, x + width)
    source_y1 = min(frame_height, y + height)
    if source_x0 >= source_x1 or source_y0 >= source_y1:
        return output
    destination_x0 = source_x0 - x
    destination_y0 = source_y0 - y
    output[
        destination_y0 : destination_y0 + (source_y1 - source_y0),
        destination_x0 : destination_x0 + (source_x1 - source_x0),
    ] = frame[source_y0:source_y1, source_x0:source_x1]
    return output


def write_sampled_training_crops_from_images_full(
    archive_path: str | Path,
    *,
    run_id: str,
    refined_run_id: str,
    roi_size_wh: tuple[int, int] = (348, 348),
    published_archive_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write one complete selector-ineligible crop run into a local copy."""

    archive = Path(archive_path).expanduser().resolve()
    archive_identity = (
        archive
        if published_archive_path is None
        else Path(published_archive_path).expanduser().resolve()
    )
    candidate = _safe_run_id(run_id, label="run_id")
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    plan = build_sampled_training_crop_plan(
        root,
        refined_run_id=refined_run_id,
        roi_size_wh=roi_size_wh,
    )
    parent = root.require_group("crop_runs")
    if candidate in parent:
        raise FileExistsError(
            f"Training crop run already exists: crop_runs/{candidate}."
        )
    run = parent.create_group(candidate)
    mark_run_started(run, run_name=candidate, stage="crop")
    storage = plan_crop_geometry_storage(
        plan.dimensions,
        profile=TRAINING_IMMUTABLE_V1,
    )
    roi_width, roi_height = plan.roi_size_wh
    layout = build_canonical_crop_roi_layout(
        total_rois=plan.row_count,
        preferred_chunk_len=1,
        roi_storage="compressed",
        use_sharding=False,
    )
    run.attrs.update(
        {
            "status": "running",
            "stage_selector_eligible": False,
            "immutable_training_materialization": True,
            "artifact_class": "sampled_training_materialized_crops",
            "logical_schema_id": SAMPLED_TRAINING_CROP_GEOMETRY_SCHEMA_ID,
            "logical_schema_version": SAMPLED_TRAINING_CROP_GEOMETRY_SCHEMA_VERSION,
            "storage_plan": storage.as_manifest(),
            "training_materialization_schema": TRAINING_CROP_MATERIALIZATION_SCHEMA_ID,
            "training_materialization_provider": (
                SAMPLED_TRAINING_IMAGES_FULL_MATERIALIZATION_PROVIDER
            ),
            "training_materialization_provider_contract": list(
                TRAINING_CROP_MATERIALIZATION_PROVIDERS
            ),
            "source_crop_archive_path": str(archive_identity),
            "source_crop_run": plan.refined_run_id,
            "source_crop_path": (
                f"refined_detect_runs/{plan.refined_run_id}/instances"
            ),
            "source_crop_manifest_binding": {
                "authority_kind": "sampled_detection_review",
                "source_refined_detect_run": plan.refined_run_id,
                "source_frame_decision_digest": plan.source_frame_decision_digest,
                "source_row_signature_spec_digest": (
                    plan.row_signature_spec.spec_digest
                ),
            },
            "source_refined_detect_run": plan.refined_run_id,
            "source_frame_decision_path": plan.source_frame_decision_path,
            "source_frame_decision_digest": plan.source_frame_decision_digest,
            "source_images_path": "raw_video/images_full",
            "source_images_dtype": "uint8",
            "source_images_shape": list(plan.source_images_shape),
            "source_pixels": "sampled_embedded_full_frame_uint8",
            "source_pixel_range": "0_255",
            "crop_storage_mode": "materialized",
            "coordinate_contract": "sampled_training_local_to_acquisition_v1",
            "height": plan.dimensions.source_height,
            "width": plan.dimensions.source_width,
            "roi_size": [roi_height, roi_width],
            "padding_mode": "zero_outside_source_frame",
            "pixel_verification": "all_rows_byte_equal_to_source_window_v1",
            **crop_roi_layout_attrs(layout),
            **plan.row_signature_spec.to_attrs(),
        }
    )

    bindings = {binding.path: binding for binding in CROP_GEOMETRY_SCHEMA_V1.bindings}
    for entry in storage.entries:
        name = entry.rule.path
        binding = bindings[name]
        contract = CROP_GEOMETRY_SCHEMA_V1.contracts.resolve(
            binding.contract_id,
            binding.contract_version,
        )
        destination = create_array_from_plan(
            run,
            name=name,
            contract=contract,
            plan=entry.plan,
            fill_value=0,
            attributes={"artifact_class": "sampled_training_crop_geometry"},
        )
        write_by_physical_units(
            destination,
            np.asarray(plan.arrays[name]),
            plan=entry.plan,
        )

    source_frame_contract = DETECTION_SOURCE_ACQUISITION_FRAME_INDEX_V1
    source_frame_intent = source_frame_contract.storage_intent(
        shape=plan.source_frame_indices.shape,
        access=AccessPattern.WINDOWED,
        write_mode=WriteMode.IMMUTABLE,
        access_unit_shape=(1,),
        growth_axis=0,
        shard_axes=(0,),
        name="source_frame_indices",
        dimensions={"n_instances": plan.row_count},
    )
    source_frame_plan = plan_storage(source_frame_intent, TRAINING_IMMUTABLE_V1)
    source_frame_array = create_array_from_plan(
        run,
        name="source_frame_indices",
        contract=source_frame_contract,
        plan=source_frame_plan,
        fill_value=0,
        attributes={"identity_semantics": "source_acquisition_frame_index"},
    )
    write_by_physical_units(
        source_frame_array,
        plan.source_frame_indices,
        plan=source_frame_plan,
    )

    roi_images = run.create_array(
        "roi_images",
        **build_crop_roi_create_kwargs(
            total_rois=plan.row_count,
            roi_sz=(roi_height, roi_width),
            layout=layout,
            overwrite=False,
        ),
    )
    source_images = root["raw_video/images_full"]
    local_frames = np.asarray(plan.arrays["frame_indices"], dtype=np.int64)
    coordinates = np.asarray(plan.arrays["roi_coordinates_full"], dtype=np.int32)
    row_start = 0
    while row_start < plan.row_count:
        frame_index = int(local_frames[row_start])
        row_stop = row_start + 1
        while row_stop < plan.row_count and int(local_frames[row_stop]) == frame_index:
            row_stop += 1
        frame = np.asarray(source_images[frame_index], dtype=np.uint8)
        if frame.shape != plan.source_images_shape[1:]:
            raise RuntimeError(
                "Decoded sampled source frame shape changed during write."
            )
        for row_index in range(row_start, row_stop):
            x, y = (int(value) for value in coordinates[row_index])
            crop = zero_padded_crop(
                frame,
                x=x,
                y=y,
                width=roi_width,
                height=roi_height,
            )
            roi_images[row_index] = crop
            if not np.array_equal(np.asarray(roi_images[row_index]), crop):
                raise RuntimeError(
                    f"Persisted crop row {row_index} differs from its source window."
                )
        row_start = row_stop

    run.attrs[TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE] = (
        build_training_crop_materialization_binding(run)
    )
    run.attrs["training_crop_materialization_binding_status"] = "strict_v1"
    run.attrs["summary_statistics"] = {
        "total_rois_cropped": plan.row_count,
        "roi_size": [roi_height, roi_width],
        "positive_frames": plan.supervision.positive_frame_count,
        "negative_frames": plan.supervision.negative_frame_count,
        "padded_rows": int(
            np.count_nonzero(
                (coordinates[:, 0] < 0)
                | (coordinates[:, 1] < 0)
                | (coordinates[:, 0] + roi_width > plan.dimensions.source_width)
                | (coordinates[:, 1] + roi_height > plan.dimensions.source_height)
            )
        ),
        "pixel_rows_verified": plan.row_count,
    }
    run.attrs["status"] = "completed"
    mark_run_complete(
        run,
        run_name=candidate,
        run_provenance=build_writer_run_provenance(
            command=("fisheye.shared.zarr.sampled_training_crop_materialization"),
            params={
                "writer_schema_id": SAMPLED_TRAINING_CROP_WRITER_SCHEMA_ID,
                "writer_schema_version": (SAMPLED_TRAINING_CROP_WRITER_SCHEMA_VERSION),
                "materialization_provider": (
                    SAMPLED_TRAINING_IMAGES_FULL_MATERIALIZATION_PROVIDER
                ),
                "roi_size_wh": [roi_width, roi_height],
                "padding_mode": "zero_outside_source_frame",
                "pixel_verification": ("all_rows_byte_equal_to_source_window_v1"),
            },
            input_run_ids={
                "source_refined_detect_run": plan.refined_run_id,
                "source_frame_decision_path": plan.source_frame_decision_path,
                "source_images_path": "raw_video/images_full",
            },
        ),
    )
    # mark_run_complete owns the standard completion attrs; this provider's
    # consumer contract uses the historical crop status spelling.
    run.attrs["status"] = "completed"
    run.attrs[TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE] = (
        build_training_crop_materialization_binding(run)
    )
    return {
        "schema_id": SAMPLED_TRAINING_CROP_WRITER_SCHEMA_ID,
        "schema_version": SAMPLED_TRAINING_CROP_WRITER_SCHEMA_VERSION,
        "status": "complete",
        "run_id": candidate,
        "row_count": plan.row_count,
        "frame_count": plan.dimensions.n_frames,
        "positive_frame_count": plan.supervision.positive_frame_count,
        "negative_frame_count": plan.supervision.negative_frame_count,
        "roi_shape": [roi_height, roi_width],
        "binding_digest": run.attrs[TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE][
            "payload_digest"
        ],
        "storage_plan": storage.as_manifest(),
        "pixel_rows_verified": plan.row_count,
        "stage_selector_eligible": False,
    }


__all__ = [
    "SAMPLED_TRAINING_CROP_WRITER_SCHEMA_ID",
    "SAMPLED_TRAINING_CROP_WRITER_SCHEMA_VERSION",
    "SampledTrainingCropPlan",
    "build_sampled_training_crop_plan",
    "write_sampled_training_crops_from_images_full",
    "write_by_physical_units",
    "zero_padded_crop",
]
