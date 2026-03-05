from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import zarr
from zarr.core.dtype import VariableLengthUTF8

from fisheye.shared.zarr.stage_arrays import (
    CROP_SPEC,
    DETECT_SPEC,
    EYE_MASKS_SPEC,
    KEYPOINTS_SPEC,
    STAGES,
    ArraySpec,
    StageSpec,
    validate_run,
)


DEFAULT_DIMS: Dict[str, int] = {
    "n_frames": 3,
    "n_detections": 5,
    "n_refined": 5,
    "n_rois": 4,
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


def _write_required_arrays(group: zarr.Group, stage_spec: StageSpec) -> None:
    for spec in stage_spec.specs:
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


def test_all_stage_specs_define_arrays() -> None:
    for name, stage_spec in STAGES.items():
        assert stage_spec.stage_name == name
        assert stage_spec.specs


def test_validate_run_accepts_detect_crop_keypoints_and_eye_masks_groups() -> None:
    for stage_spec in (DETECT_SPEC, CROP_SPEC, KEYPOINTS_SPEC, EYE_MASKS_SPEC):
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
    assert any("optional array 'centers_px'" in msg for msg in result.warnings)
