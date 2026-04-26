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

from fisheye.utils import audit_refined_mask_metrics as audit_mod
from fisheye.shared.zarr.stage_arrays import (
    REFINED_EYE_MASKS_SPEC,
    REFINED_SUBJECT_COMPONENT_ARRAYS,
    REFINED_SUBJECT_COMPONENT_METRICS,
    REFINED_SUBJECT_EYE_CONTOURS,
    REFINED_SUBJECT_EYE_GEOMETRY,
    REFINED_SUBJECT_EYE_PAIR_METRICS,
    REFINED_SUBJECT_MASKS_SPEC,
    ArraySpec,
    StageSpec,
)
from fisheye.utils.audit_refined_mask_metrics import audit_refined_mask_metrics


DEFAULT_DIMS: Dict[str, int] = {
    "n_rois": 4,
    "n_channels": 4,
    "n_points": 7,
    "n_frames": 3,
    "H": 8,
    "W": 8,
    "width": 16,
}


@pytest.fixture(autouse=True)
def _patch_zarr_sync(monkeypatch):
    def _sync_via_asyncio_run(coro, loop=None, timeout=None):
        del loop, timeout
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


def _root() -> zarr.Group:
    return zarr.open_group(store=MemoryStore(), mode="w")


def _shape_from_template(shape_template: Tuple[str | int, ...]) -> Tuple[int, ...]:
    shape: list[int] = []
    for dim in shape_template:
        shape.append(int(dim) if isinstance(dim, int) else DEFAULT_DIMS[dim])
    return tuple(shape)


def _dtype_for_spec(dtype_text: str):
    token = dtype_text.split("/", maxsplit=1)[0].strip().lower()
    if token.startswith("uint"):
        return np.uint8 if token == "uint8" else np.uint32
    if token.startswith("int"):
        return np.int64 if token == "int64" else np.int32
    if token.startswith("float"):
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
        out[...] = "clean"
        return out
    if np.dtype(dtype).kind == "b":
        return np.ones(shape, dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def _write_specs(group: zarr.Group, specs: Tuple[ArraySpec, ...]) -> None:
    for spec in specs:
        if not spec.required:
            continue
        data = _data_for_spec(spec)
        if spec.dtype == "string":
            shape = tuple(int(dim) for dim in data.shape)
            arr = group.create_array(
                spec.name,
                shape=shape,
                chunks=tuple(max(1, dim) for dim in shape),
                dtype=VariableLengthUTF8(),
                fill_value="",
                overwrite=True,
            )
            arr[:] = data
            continue
        group.create_array(spec.name, data=data, overwrite=True)


def _write_stage(group: zarr.Group, stage_spec: StageSpec) -> None:
    _write_specs(group, stage_spec.specs)
    for subgroup_name, specs in stage_spec.subgroups.items():
        _write_specs(group.require_group(subgroup_name), specs)


def _write_refined_subject_component(parent: zarr.Group, component_name: str) -> None:
    component = parent.require_group(component_name)
    _write_specs(component, REFINED_SUBJECT_COMPONENT_ARRAYS)
    _write_specs(component.require_group("metrics"), REFINED_SUBJECT_COMPONENT_METRICS)


def _write_refined_subject_eye_geometry(parent: zarr.Group, component_name: str) -> None:
    component = parent.require_group(component_name)
    _write_specs(component.require_group("geometry"), REFINED_SUBJECT_EYE_GEOMETRY)
    _write_specs(component.require_group("contours"), REFINED_SUBJECT_EYE_CONTOURS)


def test_audit_refined_eye_mask_metrics_accepts_current_surface() -> None:
    root = _root()
    parent = root.create_group("refined_eye_masks_runs")
    parent.attrs["latest"] = "refined_eye_masks_001"
    run = parent.create_group("refined_eye_masks_001")
    _write_stage(run, REFINED_EYE_MASKS_SPEC)

    summary = audit_refined_mask_metrics(root, stage="refined_eye_masks")

    assert summary["valid"] is True
    assert summary["audited_runs"]["refined_eye_masks"] == ["refined_eye_masks_001"]
    assert summary["errors"] == []


def test_audit_refined_subject_mask_metrics_accepts_components_and_eye_relations() -> None:
    root = _root()
    parent = root.create_group("refined_subject_masks_runs")
    parent.attrs["latest"] = "refined_subject_masks_001"
    run = parent.create_group("refined_subject_masks_001")
    run.attrs["mask_labels"] = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    _write_stage(run, REFINED_SUBJECT_MASKS_SPEC)

    components = run.require_group("components")
    for component_name in ("subject_body", "eye_left", "eye_right", "swim_bladder"):
        _write_refined_subject_component(components, component_name)
    for component_name in ("eye_left", "eye_right"):
        _write_refined_subject_eye_geometry(components, component_name)
    _write_specs(run.require_group("relations/eye_pair/metrics"), REFINED_SUBJECT_EYE_PAIR_METRICS)

    summary = audit_refined_mask_metrics(root, stage="refined_subject_masks")

    assert summary["valid"] is True
    assert summary["audited_runs"]["refined_subject_masks"] == ["refined_subject_masks_001"]
    assert summary["errors"] == []


def test_audit_refined_subject_mask_metrics_reports_missing_component_metric() -> None:
    root = _root()
    parent = root.create_group("refined_subject_masks_runs")
    run = parent.create_group("refined_subject_masks_001")
    run.attrs["mask_labels"] = ["subject_body"]
    _write_stage(run, REFINED_SUBJECT_MASKS_SPEC)
    components = run.require_group("components")
    _write_refined_subject_component(components, "subject_body")
    del components["subject_body/metrics/sigma_noise"]

    summary = audit_refined_mask_metrics(root, stage="refined_subject_masks")

    assert summary["valid"] is False
    assert any("sigma_noise" in message for message in summary["errors"])


def test_audit_refined_mask_metrics_rejects_ambiguous_run_name() -> None:
    with pytest.raises(ValueError, match="run_name requires"):
        audit_refined_mask_metrics(_root(), stage="all", run_name="refined_subject_masks_001")


def test_audit_refined_mask_metrics_zarr_uses_palette_zarr_opener(monkeypatch, tmp_path) -> None:
    root = _root()
    parent = root.create_group("refined_subject_masks_runs")
    parent.attrs["latest"] = "refined_subject_masks_001"
    run = parent.create_group("refined_subject_masks_001")
    run.attrs["mask_labels"] = ["subject_body"]
    _write_stage(run, REFINED_SUBJECT_MASKS_SPEC)
    _write_refined_subject_component(run.require_group("components"), "subject_body")

    calls = []

    def _fake_open_zarr_root(path, mode="r"):
        calls.append((path, mode))
        return root

    monkeypatch.setattr(audit_mod, "open_zarr_root", _fake_open_zarr_root)
    summary = audit_mod.audit_refined_mask_metrics_zarr(
        tmp_path / "analysis.zarr",
        stage="refined_subject_masks",
        latest_only=True,
    )

    assert summary["valid"] is True
    assert summary["audited_runs"]["refined_subject_masks"] == ["refined_subject_masks_001"]
    assert calls == [((tmp_path / "analysis.zarr").resolve(), "r")]
