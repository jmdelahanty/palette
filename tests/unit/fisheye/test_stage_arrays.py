from __future__ import annotations

import asyncio
import threading
from typing import Dict, Tuple

import numpy as np
import pytest
import zarr
import zarr.api.synchronous as zarr_sync_api
import zarr.core.sync as zarr_sync
from zarr.core.dtype import VariableLengthUTF8
from zarr.storage import MemoryStore

from fisheye.shared.mask_store import (
    write_bitpacked_mask_store_from_dense,
    write_component_rle_mask_store_from_dense,
)
from fisheye.shared.row_lineage import (
    ROW_IDENTITY_MODE_INSTANCE_KEY,
    ROW_IDENTITY_MODE_LEGACY_POSITIONAL,
    ROW_IDENTITY_MODE_SCHEMA,
)
from fisheye.shared.zarr.stage_arrays import (
    ARENA_ASSIGNMENT_SPEC,
    BOUT_KINEMATICS_SPEC,
    BOUT_CLASSIFICATION_SPEC,
    CROP_SPEC,
    DETECTION_PROFILE_SPEC,
    DETECT_SPEC,
    EYE_ANGLE_SPEC,
    EYE_MASK_PROFILE_SPEC,
    EYE_MASKS_SPEC,
    KEYPOINT_PROFILE_SPEC,
    KEYPOINTS_SPEC,
    REFINED_DETECT_SPEC,
    REFINED_EYE_MASKS_SPEC,
    REFINED_KEYPOINTS_SPEC,
    REFINED_SUBJECT_COMPONENT_METRICS,
    REFINED_SUBJECT_EYE_PAIR_METRICS,
    REFINED_SUBJECT_MASKS_SPEC,
    STAGES,
    STIMULUS_SPEC,
    SUBJECT_MASKS_SPEC,
    SUBJECT_SHAPE_SPEC,
    TAIL_POSTURE_VIEW_SPEC,
    TAIL_KINEMATICS_SPEC,
    TRACK_KINEMATICS_SPEC,
    ArraySpec,
    StageSpec,
    array_specs_by_name,
    validate_run,
)


DEFAULT_DIMS: Dict[str, int] = {
    "n_frames": 3,
    "n_detections": 5,
    "n_refined": 5,
    "n_rows": 5,
    "n_source_detections": 5,
    "n_instances": 3,
    "n_frame_offsets": 4,
    "n_rois": 4,
    "n_channels": 4,
    "n_keypoints": 5,
    "n_import_frames": 2,
    "n_samples": 3,
    "n_failures": 2,
    "n_points": 7,
    "n_bouts": 4,
    "n_tail_keypoints": 11,
    "n_tail_angles": 10,
    "n_tail_samples": 10,
    "n_tracks": 2,
    "n_arenas": 4,
    "n_metadata_frames": 6,
    "n_camera_frames": 4,
    "H": 8,
    "W": 8,
    "H_ds": 4,
    "W_ds": 4,
    "h": 8,
    "w": 8,
    "width": 16,
}


@pytest.fixture(autouse=True)
def _patch_zarr_sync(monkeypatch):
    def _sync_via_asyncio_run(coro, loop=None, timeout=None):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result = {}
        error = {}

        def _runner():
            try:
                result["value"] = asyncio.run(coro)
            except Exception as exc:  # pragma: no cover - defensive
                error["exc"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()

        if "exc" in error:
            raise error["exc"]
        return result.get("value")

    monkeypatch.setattr(zarr_sync, "sync", _sync_via_asyncio_run)
    monkeypatch.setattr(zarr_sync_api, "sync", _sync_via_asyncio_run)
    monkeypatch.setattr(zarr, "group", lambda *args, **kwargs: zarr.open_group(store=MemoryStore(), mode="a"))


def _shape_from_template(shape_template: Tuple[str | int, ...]) -> Tuple[int, ...]:
    shape: list[int] = []
    for dim in shape_template:
        if isinstance(dim, int):
            shape.append(dim)
        else:
            shape.append(DEFAULT_DIMS[dim])
    return tuple(shape)


def _dtype_for_spec(dtype_text: str):
    token = dtype_text.split("/", maxsplit=1)[0].strip().lower()
    if token.startswith("uint"):
        return np.uint8 if token == "uint8" else np.uint32
    if token.startswith("int"):
        if token == "int64":
            return np.int64
        return np.int8 if token == "int8" else np.int32
    if token.startswith("float"):
        if token == "float16":
            return np.float16
        if token == "float64":
            return np.float64
        return np.float32
    if token == "bool":
        return np.bool_
    if token == "string":
        return VariableLengthUTF8()
    raise ValueError(f"Unsupported dtype token: {dtype_text}")


def _data_for_spec(spec: ArraySpec):
    shape = _shape_from_template(spec.shape_template)
    dtype = _dtype_for_spec(spec.dtype)
    if spec.dtype == "string":
        out = np.empty(shape, dtype=object)
        out[...] = "ok"
        return out
    if np.dtype(dtype).kind == "b":
        return np.ones(shape, dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def _write_required_specs(group: zarr.Group, specs: Tuple[ArraySpec, ...]) -> None:
    for spec in specs:
        if not spec.required:
            continue
        data = _data_for_spec(spec)
        if spec.dtype == "string":
            shape = tuple(int(dim) for dim in data.shape)
            chunks = tuple(max(1, dim) for dim in shape) if shape else (1,)
            arr = group.create_array(
                spec.name,
                shape=shape,
                chunks=chunks,
                dtype=VariableLengthUTF8(),
                fill_value="",
                overwrite=True,
            )
            arr[:] = data
            continue
        group.create_array(spec.name, data=data, overwrite=True)


def _write_required_arrays(group: zarr.Group, stage_spec: StageSpec) -> None:
    for attr_name in stage_spec.required_attrs:
        group.attrs[attr_name] = "ok"
    for attr_name, attr_value in stage_spec.required_attr_values.items():
        group.attrs[attr_name] = attr_value

    _write_required_specs(group, stage_spec.specs)

    for subgroup_name, subgroup_specs in stage_spec.subgroups.items():
        subgroup = group.require_group(subgroup_name)
        _write_required_specs(subgroup, subgroup_specs)


def test_all_stage_specs_define_arrays() -> None:
    for name, stage_spec in STAGES.items():
        assert stage_spec.stage_name == name
        assert (
            stage_spec.specs
            or stage_spec.subgroups
            or stage_spec.required_attrs
            or stage_spec.required_attr_values
        )


def test_validate_run_accepts_detect_crop_keypoints_and_eye_masks_groups() -> None:
    for stage_spec in (
        DETECT_SPEC,
        CROP_SPEC,
        KEYPOINTS_SPEC,
        EYE_MASKS_SPEC,
    ):
        group = zarr.group()
        _write_required_arrays(group, stage_spec)

        result = validate_run(group, stage_spec)
        assert result.valid, f"{stage_spec.stage_name} errors: {result.errors}"


def test_validate_run_accepts_minimal_stimulus_import_surface() -> None:
    group = zarr.group()
    _write_required_arrays(group, STIMULUS_SPEC)

    result = validate_run(group, STIMULUS_SPEC)

    assert result.valid, result.errors
    assert not result.errors


def test_validate_run_accepts_attrs_only_keypoint_profile_surface() -> None:
    group = zarr.group()
    _write_required_arrays(group, KEYPOINT_PROFILE_SPEC)

    result = validate_run(group, KEYPOINT_PROFILE_SPEC)

    assert result.valid, result.errors
    assert not result.errors


def test_validate_run_accepts_attrs_only_detection_profile_surface() -> None:
    group = zarr.group()
    _write_required_arrays(group, DETECTION_PROFILE_SPEC)

    result = validate_run(group, DETECTION_PROFILE_SPEC)

    assert result.valid, result.errors
    assert not result.errors


def test_validate_run_accepts_attrs_only_eye_mask_profile_surface() -> None:
    group = zarr.group()
    _write_required_arrays(group, EYE_MASK_PROFILE_SPEC)

    result = validate_run(group, EYE_MASK_PROFILE_SPEC)

    assert result.valid, result.errors
    assert not result.errors


def test_validate_run_accepts_minimal_subject_shape_surface() -> None:
    group = zarr.group()
    _write_required_arrays(group, SUBJECT_SHAPE_SPEC)

    result = validate_run(group, SUBJECT_SHAPE_SPEC)

    assert result.valid, result.errors
    assert not result.errors


def test_validate_run_accepts_minimal_tail_posture_view_surface() -> None:
    group = zarr.group()
    _write_required_arrays(group, TAIL_POSTURE_VIEW_SPEC)

    result = validate_run(group, TAIL_POSTURE_VIEW_SPEC)

    assert result.valid, result.errors
    assert not result.errors


def test_validate_run_accepts_minimal_bout_classification_surface() -> None:
    group = zarr.group()
    _write_required_arrays(group, BOUT_CLASSIFICATION_SPEC)

    result = validate_run(group, BOUT_CLASSIFICATION_SPEC)

    assert result.valid, result.errors
    assert not result.errors


def test_validate_run_accepts_minimal_tail_kinematics_surface() -> None:
    group = zarr.group()
    _write_required_arrays(group, TAIL_KINEMATICS_SPEC)

    result = validate_run(group, TAIL_KINEMATICS_SPEC)

    assert result.valid, result.errors
    assert not result.errors


def test_validate_run_accepts_minimal_track_kinematics_surface() -> None:
    group = zarr.group()
    _write_required_arrays(group, TRACK_KINEMATICS_SPEC)

    result = validate_run(group, TRACK_KINEMATICS_SPEC)

    assert result.valid, result.errors
    assert not result.errors


def test_validate_run_accepts_minimal_eye_angle_surface() -> None:
    group = zarr.group()
    _write_required_arrays(group, EYE_ANGLE_SPEC)

    result = validate_run(group, EYE_ANGLE_SPEC)

    assert result.valid, result.errors
    assert not result.errors


def test_validate_run_accepts_minimal_bout_kinematics_surface() -> None:
    group = zarr.group()
    _write_required_arrays(group, BOUT_KINEMATICS_SPEC)

    result = validate_run(group, BOUT_KINEMATICS_SPEC)

    assert result.valid, result.errors
    assert not result.errors


def test_validate_run_reports_missing_required_attrs() -> None:
    group = zarr.group()
    group.attrs["schema_name"] = "keypoint_dataset_profile"

    result = validate_run(group, KEYPOINT_PROFILE_SPEC)

    assert not result.valid
    assert any("missing required attr 'profile_summary'" in msg for msg in result.errors)


def test_validate_run_reports_wrong_required_attr_value() -> None:
    group = zarr.group()
    _write_required_arrays(group, EYE_ANGLE_SPEC)
    group.attrs["status"] = "running"

    result = validate_run(group, EYE_ANGLE_SPEC)

    assert not result.valid
    assert any("attr 'status' expected 'complete', got 'running'" in msg for msg in result.errors)


def test_refined_mask_stage_specs_cover_writer_metric_surfaces() -> None:
    refined_eye_metrics = array_specs_by_name(REFINED_EYE_MASKS_SPEC, subgroup="metrics")
    for name in (
        "area_ratio_vs_source",
        "area_union_refined",
        "area_union_source",
        "area_union_ratio",
        "separation_keypoint",
        "separation_delta",
        "probability_mean",
        "probability_high_fraction",
        "reason_bytes",
    ):
        assert name in refined_eye_metrics
    assert refined_eye_metrics["filter_flags"].dtype == "bool"

    refined_subject_metrics = array_specs_by_name(REFINED_SUBJECT_MASKS_SPEC, subgroup="metrics")
    for name in ("mask_present", "area_px", "centroid_xy", "centroid_valid", "bbox_xyxy", "bbox_valid"):
        assert name in refined_subject_metrics

    component_metric_names = {spec.name for spec in REFINED_SUBJECT_COMPONENT_METRICS}
    assert {
        "component_count",
        "largest_component_fraction",
        "hole_count",
        "hole_area_fraction",
        "sigma_noise",
        "curvature_var",
        "ipr",
        "solidity",
    }.issubset(component_metric_names)

    eye_pair_metric_names = {spec.name for spec in REFINED_SUBJECT_EYE_PAIR_METRICS}
    assert eye_pair_metric_names == {"separation_px", "separation_valid"}


def test_validate_run_accepts_current_refined_mask_specs() -> None:
    for stage_spec in (REFINED_EYE_MASKS_SPEC, REFINED_SUBJECT_MASKS_SPEC):
        group = zarr.group()
        _write_required_arrays(group, stage_spec)

        result = validate_run(group, stage_spec)
        assert result.valid, f"{stage_spec.stage_name} errors: {result.errors}"


def test_validate_run_requires_source_crop_row_ids_for_subject_mask_stages() -> None:
    for stage_spec in (SUBJECT_MASKS_SPEC, REFINED_SUBJECT_MASKS_SPEC):
        group = zarr.group()
        _write_required_arrays(group, stage_spec)
        del group["source_crop_row_ids"]

        result = validate_run(group, stage_spec)

        assert not result.valid
        assert any("missing required array 'source_crop_row_ids'" in message for message in result.errors)


def test_validate_run_accepts_refined_subject_masks_with_rle_store_without_dense_masks() -> None:
    group = zarr.group()
    _write_required_arrays(group, REFINED_SUBJECT_MASKS_SPEC)
    group.attrs["mask_labels"] = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    write_component_rle_mask_store_from_dense(
        group,
        group["masks_roi"],
        component_names=tuple(str(value) for value in group.attrs["mask_labels"]),
        encode_row_chunk_size=2,
    )
    del group["masks_roi"]

    result = validate_run(group, REFINED_SUBJECT_MASKS_SPEC)

    assert result.valid, result.errors


def test_validate_run_accepts_refined_subject_masks_with_bitpacked_store_without_dense_masks() -> None:
    group = zarr.group()
    _write_required_arrays(group, REFINED_SUBJECT_MASKS_SPEC)
    group.attrs["mask_labels"] = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    write_bitpacked_mask_store_from_dense(
        group,
        group["masks_roi"],
        component_names=tuple(str(value) for value in group.attrs["mask_labels"]),
        encode_row_chunk_size=2,
        validation_mode="invariants",
    )
    del group["masks_roi"]

    result = validate_run(group, REFINED_SUBJECT_MASKS_SPEC)

    assert result.valid, result.errors


def test_validate_run_rejects_refined_subject_rle_with_bad_indptr_terminal() -> None:
    group = zarr.group()
    _write_required_arrays(group, REFINED_SUBJECT_MASKS_SPEC)
    group.attrs["mask_labels"] = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    write_component_rle_mask_store_from_dense(
        group,
        group["masks_roi"],
        component_names=tuple(str(value) for value in group.attrs["mask_labels"]),
        encode_row_chunk_size=2,
    )
    del group["masks_roi"]
    component = group["mask_rle"]["components"]["00_subject_body"]
    component["indptr"][-1] = int(component["indptr"][-1]) + 1

    result = validate_run(group, REFINED_SUBJECT_MASKS_SPEC)

    assert not result.valid
    assert any("indptr terminates" in message and "counts has" in message for message in result.errors)


def test_validate_run_rejects_refined_subject_rle_with_bad_shape_attr() -> None:
    group = zarr.group()
    _write_required_arrays(group, REFINED_SUBJECT_MASKS_SPEC)
    group.attrs["mask_labels"] = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    write_component_rle_mask_store_from_dense(
        group,
        group["masks_roi"],
        component_names=tuple(str(value) for value in group.attrs["mask_labels"]),
        encode_row_chunk_size=2,
    )
    del group["masks_roi"]
    group["mask_rle"].attrs["encoded_shape_hw"] = [0, DEFAULT_DIMS["W"]]

    result = validate_run(group, REFINED_SUBJECT_MASKS_SPEC)

    assert not result.valid
    assert any("encoded_shape_hw" in message and "positive" in message for message in result.errors)


def test_validate_run_reports_missing_required_arrays() -> None:
    group = zarr.group()
    group.create_array("frame_indices", data=np.array([0, 1], dtype=np.int32), overwrite=True)

    result = validate_run(group, DETECT_SPEC)
    assert not result.valid
    assert any("missing required array 'bbox_norm_coords'" in msg for msg in result.errors)


def test_validate_run_reports_wrong_dtype_kind() -> None:
    group = zarr.group()
    _write_required_arrays(group, DETECT_SPEC)
    group.create_array(
        "scores",
        data=np.array([1, 2, 3, 4, 5], dtype=np.int32),
        overwrite=True,
    )

    result = validate_run(group, DETECT_SPEC)
    assert not result.valid
    assert any("scores" in msg and "dtype kind mismatch" in msg for msg in result.errors)


def test_validate_run_missing_optional_arrays_are_warnings() -> None:
    group = zarr.group()
    _write_required_arrays(group, DETECT_SPEC)

    result = validate_run(group, DETECT_SPEC)
    assert result.valid
    assert not result.errors
    assert any("optional array 'n_detections'" in msg for msg in result.warnings)
    assert any("optional array 'centers_px'" in msg for msg in result.warnings)


def test_validate_run_requires_arena_assignment_counts_surface() -> None:
    group = zarr.group()
    group.create_array(
        "arena_ids",
        data=np.array([0, 1, -1], dtype=np.int32),
        overwrite=True,
    )

    result = validate_run(group, ARENA_ASSIGNMENT_SPEC)

    assert not result.valid
    assert any("missing required array 'n_detections_per_arena'" in msg for msg in result.errors)


def test_validate_run_accepts_arena_assignment_counts_surface() -> None:
    group = zarr.group()
    _write_required_arrays(group, ARENA_ASSIGNMENT_SPEC)

    result = validate_run(group, ARENA_ASSIGNMENT_SPEC)

    assert result.valid, result.errors
    assert not result.errors


def test_legacy_count_aliases_are_not_required_for_current_specs() -> None:
    for stage_spec, alias in (
        (DETECT_SPEC, "n_detections"),
        (KEYPOINTS_SPEC, "n_rois"),
        (REFINED_KEYPOINTS_SPEC, "n_rois"),
    ):
        group = zarr.group()
        _write_required_arrays(group, stage_spec)

        result = validate_run(group, stage_spec)
        assert result.valid, f"{stage_spec.stage_name} errors: {result.errors}"
        assert any(f"optional array '{alias}'" in msg for msg in result.warnings)


def test_validate_run_accepts_geometry_only_crop_group() -> None:
    group = zarr.group()
    _write_required_arrays(group, CROP_SPEC)

    result = validate_run(group, CROP_SPEC)
    assert result.valid
    assert not result.errors
    assert any("optional array 'roi_images'" in msg for msg in result.warnings)
    assert any("optional array 'roi_coordinates_ds'" in msg for msg in result.warnings)
    assert any("optional array 'detection_source'" in msg for msg in result.warnings)


def test_validate_run_refined_detect_dense_root_happy_path() -> None:
    group = zarr.group()
    for subgroup_name, subgroup_specs in REFINED_DETECT_SPEC.subgroups.items():
        _write_required_arrays(
            group.require_group(subgroup_name),
            StageSpec(
                stage_name=f"refined_detect/{subgroup_name}",
                zarr_group=subgroup_name,
                specs=subgroup_specs,
            ),
        )

    result = validate_run(group, REFINED_DETECT_SPEC)
    assert result.valid
    assert not result.errors


def test_validate_run_refined_detect_reports_missing_root_array() -> None:
    group = zarr.group()
    for subgroup_name, subgroup_specs in REFINED_DETECT_SPEC.subgroups.items():
        _write_required_arrays(
            group.require_group(subgroup_name),
            StageSpec(
                stage_name=f"refined_detect/{subgroup_name}",
                zarr_group=subgroup_name,
                specs=subgroup_specs,
            ),
        )
    del group["instances"]["source_kind_codes"]

    result = validate_run(group, REFINED_DETECT_SPEC)
    assert not result.valid
    assert any("missing required array 'source_kind_codes'" in msg for msg in result.errors)


@pytest.mark.parametrize(
    ("stage_spec", "identity_subgroup"),
    (
        (REFINED_DETECT_SPEC, "instances"),
        (REFINED_KEYPOINTS_SPEC, None),
        (REFINED_SUBJECT_MASKS_SPEC, None),
    ),
)
def test_refined_identity_validation_requires_keys_for_modern_mode(
    stage_spec: StageSpec,
    identity_subgroup: str | None,
) -> None:
    group = zarr.group()
    _write_required_arrays(group, stage_spec)
    group.attrs["row_identity_mode"] = ROW_IDENTITY_MODE_INSTANCE_KEY
    group.attrs["row_identity_mode_schema"] = ROW_IDENTITY_MODE_SCHEMA

    result = validate_run(group, stage_spec)

    assert not result.valid
    assert any("requires instance_key" in message for message in result.errors)

    identity_group = group[identity_subgroup] if identity_subgroup is not None else group
    row_axis_array = (
        identity_group["frame_indices"]
        if "frame_indices" in identity_group
        else identity_group["source_crop_row_ids"]
    )
    row_count = int(row_axis_array.shape[0])
    identity_group.create_array(
        "instance_key",
        data=np.arange(1, row_count + 1, dtype=np.uint64),
        overwrite=True,
    )
    result = validate_run(group, stage_spec)
    assert result.valid, result.errors


def test_refined_identity_validation_rejects_duplicate_modern_keys() -> None:
    group = zarr.group()
    _write_required_arrays(group, REFINED_KEYPOINTS_SPEC)
    group.attrs["row_identity_mode"] = ROW_IDENTITY_MODE_INSTANCE_KEY
    group.attrs["row_identity_mode_schema"] = ROW_IDENTITY_MODE_SCHEMA
    row_count = int(group["frame_indices"].shape[0])
    group.create_array(
        "instance_key",
        data=np.ones((row_count,), dtype=np.uint64),
        overwrite=True,
    )

    result = validate_run(group, REFINED_KEYPOINTS_SPEC)

    assert not result.valid
    assert any("duplicate values" in message for message in result.errors)


def test_refined_identity_validation_labels_historical_keyless_mode() -> None:
    group = zarr.group()
    _write_required_arrays(group, REFINED_KEYPOINTS_SPEC)
    group.attrs["row_identity_mode"] = ROW_IDENTITY_MODE_LEGACY_POSITIONAL
    group.attrs["row_identity_mode_schema"] = ROW_IDENTITY_MODE_SCHEMA

    result = validate_run(group, REFINED_KEYPOINTS_SPEC)

    assert result.valid, result.errors
    assert any("explicit legacy_positional compatibility mode" in message for message in result.warnings)
    assert not any(
        "missing optional array 'instance_key'" in message for message in result.warnings
    )


def test_refined_identity_validation_rejects_explicit_identity_downgrade() -> None:
    group = zarr.group()
    _write_required_arrays(group, REFINED_KEYPOINTS_SPEC)
    group.attrs["row_identity_mode"] = ROW_IDENTITY_MODE_LEGACY_POSITIONAL
    group.attrs["row_identity_mode_schema"] = ROW_IDENTITY_MODE_SCHEMA
    row_count = int(group["frame_indices"].shape[0])
    group.create_array(
        "instance_key",
        data=np.arange(1, row_count + 1, dtype=np.uint64),
        overwrite=True,
    )

    result = validate_run(group, REFINED_KEYPOINTS_SPEC)

    assert not result.valid
    assert any("refusing an identity downgrade" in message for message in result.errors)
