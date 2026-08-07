"""Exact logical arrays and byte-planned storage for merged keypoint training v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Mapping

import zarr

from fisheye.shared.zarr.array_contracts import (
    BOOL,
    FLOAT32,
    INT8,
    INT32,
    INT64,
    UINT8,
    ArrayContract,
    DTypeContract,
)
from fisheye.shared.zarr.array_factory import (
    create_array_from_plan,
    validate_array_metadata_declaration_from_plan,
)
from fisheye.shared.zarr.storage_intent import AccessPattern, StoragePlan, WriteMode
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import TRAINING_IMMUTABLE_V1, StorageProfile


SCHEMA_ID = "palette.merged_keypoint_training"
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class PlannedTrainingArray:
    path: str
    contract: ArrayContract
    plan: StoragePlan
    fill_value: Any

    def as_manifest(self) -> dict[str, object]:
        return {
            "path": self.path,
            "logical_contract": self.contract.as_manifest(),
            "storage_plan": self.plan.as_dict(),
            "fill_value": self.fill_value,
        }


def _contract(
    path: str,
    *,
    dtype: DTypeContract,
    shape_template: tuple[str | int, ...],
    axis_names: tuple[str, ...],
    description: str,
    coordinate_space: str | None = None,
) -> ArrayContract:
    return ArrayContract(
        schema_id=f"{SCHEMA_ID}.{path.replace('/', '.')}",
        schema_version=SCHEMA_VERSION,
        dtype=dtype,
        shape_template=shape_template,
        axis_names=axis_names,
        description=description,
        coordinate_space=coordinate_space,
    )


def plan_merged_keypoint_training_arrays(
    *,
    run_name: str,
    n_samples: int,
    roi_shape: tuple[int, ...],
    keypoint_shape: tuple[int, int],
    n_sources: int,
    split_counts: Mapping[str, int],
    profile: StorageProfile = TRAINING_IMMUTABLE_V1,
) -> dict[str, PlannedTrainingArray]:
    if len(roi_shape) not in {2, 3}:
        raise ValueError(f"roi_shape must have rank 2 or 3, got {roi_shape!r}.")
    if len(keypoint_shape) != 2 or int(keypoint_shape[1]) != 2:
        raise ValueError(f"keypoint_shape must be (K,2), got {keypoint_shape!r}.")
    dimensions = {
        "N": int(n_samples),
        "K": int(keypoint_shape[0]),
        "S": int(n_sources),
        "N_train": int(split_counts.get("train", 0)),
        "N_val": int(split_counts.get("val", 0)),
        "N_test": int(split_counts.get("test", 0)),
    }
    if any(value < 0 for value in dimensions.values()):
        raise ValueError("Merged keypoint training dimensions cannot be negative.")

    roi_axes = ("sample", "roi_y", "roi_x") + (("channel",) if len(roi_shape) == 3 else ())
    roi_template: tuple[str | int, ...] = ("N", *roi_shape)
    specs: list[tuple[str, ArrayContract, tuple[int, ...], AccessPattern, Any]] = [
        (
            f"crop_runs/{run_name}/roi_images",
            _contract(
                "crop_runs.roi_images",
                dtype=UINT8,
                shape_template=roi_template,
                axis_names=roi_axes,
                description="Zero-padded, never-resized keypoint training pixels.",
                coordinate_space="training_roi_pixels",
            ),
            (n_samples, *roi_shape),
            AccessPattern.PER_ROW,
            0,
        ),
        (
            f"crop_runs/{run_name}/bbox_norm_coords",
            _contract(
                "crop_runs.bbox_norm_coords",
                dtype=FLOAT32,
                shape_template=("N", 4),
                axis_names=("sample", "xywh"),
                description="Pose-training bounding boxes on the output ROI canvas.",
                coordinate_space="training_roi_normalized",
            ),
            (n_samples, 4),
            AccessPattern.WINDOWED,
            0.0,
        ),
        (
            f"crop_runs/{run_name}/crop_bbox_norm_coords",
            _contract(
                "crop_runs.crop_bbox_norm_coords",
                dtype=FLOAT32,
                shape_template=("N", 4),
                axis_names=("sample", "xywh"),
                description="Source crop-stage bounding-box provenance.",
            ),
            (n_samples, 4),
            AccessPattern.WINDOWED,
            0.0,
        ),
        (
            f"crop_runs/{run_name}/frame_indices",
            _contract(
                "crop_runs.frame_indices",
                dtype=INT64,
                shape_template=("N",),
                axis_names=("sample",),
                description="Dense merged sample indices.",
            ),
            (n_samples,),
            AccessPattern.WINDOWED,
            0,
        ),
        (
            f"crop_runs/{run_name}/detection_source",
            _contract(
                "crop_runs.detection_source",
                dtype=INT8,
                shape_template=("N",),
                axis_names=("sample",),
                description="Detection-source code retained from crop lineage.",
            ),
            (n_samples,),
            AccessPattern.WINDOWED,
            0,
        ),
        (
            f"keypoints_runs/{run_name}/keypoints_roi",
            _contract(
                "keypoints_runs.keypoints_roi",
                dtype=FLOAT32,
                shape_template=("N", "K", 2),
                axis_names=("sample", "keypoint", "xy"),
                description="Reviewed keypoints translated onto the output ROI canvas.",
                coordinate_space="training_roi_pixels",
            ),
            (n_samples, *keypoint_shape),
            AccessPattern.WINDOWED,
            0.0,
        ),
        (
            f"keypoints_runs/{run_name}/detection_success",
            _contract(
                "keypoints_runs.detection_success",
                dtype=BOOL,
                shape_template=("N",),
                axis_names=("sample",),
                description="Whether the row has usable full keypoint supervision.",
            ),
            (n_samples,),
            AccessPattern.WINDOWED,
            False,
        ),
        (
            f"keypoints_runs/{run_name}/keypoint_box_only",
            _contract(
                "keypoints_runs.keypoint_box_only",
                dtype=BOOL,
                shape_template=("N",),
                axis_names=("sample",),
                description="Whether the row contributes box-only supervision.",
            ),
            (n_samples,),
            AccessPattern.WINDOWED,
            False,
        ),
    ]
    for name, dtype, description in (
        ("source_dataset_idx", INT32, "Source-dataset ordinal per merged row."),
        ("source_frame_idx", INT64, "Source frame index per merged row."),
        ("source_roi_idx", INT64, "Source ROI row per merged row."),
        ("source_refined_row_ids", INT64, "Stable refined-row identity per merged row."),
        ("source_detect_row_index", INT32, "Source detection-row lineage per merged row."),
    ):
        specs.append(
            (
                f"source_index/{name}",
                _contract(
                    f"source_index.{name}",
                    dtype=dtype,
                    shape_template=("N",),
                    axis_names=("sample",),
                    description=description,
                ),
                (n_samples,),
                AccessPattern.WINDOWED,
                -1 if name == "source_detect_row_index" else 0,
            )
        )
    for split_name, dimension_name in (
        ("train_indices", "N_train"),
        ("val_indices", "N_val"),
        ("test_indices", "N_test"),
    ):
        specs.append(
            (
                f"splits/{split_name}",
                _contract(
                    f"splits.{split_name}",
                    dtype=INT64,
                    shape_template=(dimension_name,),
                    axis_names=("split_row",),
                    description=f"Fixed merged-row indices for {split_name.removesuffix('_indices')}.",
                ),
                (dimensions[dimension_name],),
                AccessPattern.EAGER,
                0,
            )
        )

    planned: dict[str, PlannedTrainingArray] = {}
    for path, contract, shape, access, fill_value in specs:
        plan = plan_storage(
            contract.storage_intent(
                shape=shape,
                access=access,
                write_mode=WriteMode.IMMUTABLE,
                dimensions=dimensions,
                name=path,
            ),
            profile,
        )
        planned[path] = PlannedTrainingArray(
            path=path,
            contract=contract,
            plan=plan,
            fill_value=fill_value,
        )
    return planned


def create_planned_training_array(
    group: zarr.Group,
    *,
    name: str,
    planned: PlannedTrainingArray,
) -> zarr.Array:
    return create_array_from_plan(
        group,
        name=name,
        contract=planned.contract,
        plan=planned.plan,
        fill_value=planned.fill_value,
    )


def validate_merged_keypoint_training_storage(
    archive: str | Path,
    *,
    plans: Mapping[str, PlannedTrainingArray],
) -> tuple[str, ...]:
    root = Path(archive).expanduser().resolve()
    errors: list[str] = []
    for path, planned in plans.items():
        metadata_path = root.joinpath(*path.split("/"), "zarr.json")
        if not metadata_path.is_file():
            errors.append(f"{path}: direct zarr.json is missing")
            continue
        try:
            declaration = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: cannot read direct metadata: {exc}")
            continue
        for error in validate_array_metadata_declaration_from_plan(
            declaration,
            contract=planned.contract,
            plan=planned.plan,
            fill_value=planned.fill_value,
        ):
            errors.append(f"{path}: {error}")
    return tuple(errors)


def storage_plan_manifest(plans: Mapping[str, PlannedTrainingArray]) -> dict[str, object]:
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "profile_id": TRAINING_IMMUTABLE_V1.profile_id,
        "arrays": {
            path: plans[path].as_manifest()
            for path in sorted(plans)
        },
        "variable_width_metadata_arrays": [
            "source_index/source_dataset_id",
            "source_index/source_zarr_path",
            "source_index/source_roi_transform_json",
        ],
    }


__all__ = [
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "PlannedTrainingArray",
    "create_planned_training_array",
    "plan_merged_keypoint_training_arrays",
    "storage_plan_manifest",
    "validate_merged_keypoint_training_storage",
]
