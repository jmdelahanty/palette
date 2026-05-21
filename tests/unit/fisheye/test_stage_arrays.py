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

from fisheye.shared.zarr.stage_arrays import (
    CROP_SPEC,
    DETECT_SPEC,
    EYE_MASKS_SPEC,
    KEYPOINTS_SPEC,
    REFINED_DETECT_SPEC,
    REFINED_EYE_MASKS_SPEC,
    REFINED_KEYPOINTS_SPEC,
    REFINED_SUBJECT_COMPONENT_METRICS,
    REFINED_SUBJECT_EYE_PAIR_METRICS,
    REFINED_SUBJECT_MASKS_SPEC,
    STAGES,
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
    _write_required_specs(group, stage_spec.specs)

    for subgroup_name, subgroup_specs in stage_spec.subgroups.items():
        subgroup = group.require_group(subgroup_name)
        _write_required_specs(subgroup, subgroup_specs)


def test_all_stage_specs_define_arrays() -> None:
    for name, stage_spec in STAGES.items():
        assert stage_spec.stage_name == name
        assert stage_spec.specs or stage_spec.subgroups


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
