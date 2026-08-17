"""Fake-backed contract tests for the strict subject-mask position adapter."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import fisheye.shared.subject_position_mask_source as subject_source
from fisheye.shared.subject_position_expression import (
    MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
    SUBJECT_BODY_MASK_CENTROID_ESTIMATOR_ID,
)


PROFILE_PATH = (
    "configs/fisheye/anatomy_profiles/zebrafish_larva_v1.json"
)
BINDING_ID = "zebrafish_larva_subject_mask_lr_v1"


class _Node:
    def __init__(self, value=None, *, attrs=None):
        self._value = value
        self.attrs = {} if attrs is None else attrs

    def __getitem__(self, key):
        if key is Ellipsis:
            return self._value
        raise KeyError(key)


class _Group:
    def __init__(self, children=None, *, attrs=None):
        self.children = {} if children is None else children
        self.attrs = {} if attrs is None else attrs

    def __getitem__(self, key):
        if key in self.children:
            return self.children[key]
        prefix = f"{key}/"
        nested = {
            name[len(prefix) :]: value
            for name, value in self.children.items()
            if name.startswith(prefix) and "/" not in name[len(prefix) :]
        }
        if nested:
            return _Group(nested)
        raise KeyError(key)


def _profile():
    return subject_source.load_anatomy_profile(PROFILE_PATH)


def _fake_source(*, labels=None, available=None, rows=2, source_kind="raw"):
    labels = tuple(labels or ("subject_body", "eye_left", "eye_right", "swim_bladder"))
    components = len(labels)
    centroid = np.zeros((rows, components, 2), dtype=np.float32)
    centroid[:, :, 0] = np.arange(rows, dtype=np.float32)[:, None]
    centroid[:, :, 1] = np.arange(components, dtype=np.float32)[None, :]
    valid = np.ones((rows, components), dtype=bool)
    if rows:
        valid[-1, -1] = False
        centroid[-1, -1] = 0
    available = np.ones(components, dtype=bool) if available is None else np.asarray(available, dtype=bool)
    identity = SimpleNamespace(
        leading_dimension=rows,
        record_sha256="row-identity",
        contract=SimpleNamespace(mode="instance_key"),
    )
    chain = SimpleNamespace(
        descriptor_pixel_convention="continuous",
        source_camera_pixel_convention="continuous",
        row_identity=identity,
        source_camera_frame_authority="camera-continuous",
    )
    descriptor = SimpleNamespace(coordinate_node=_Node(centroid))
    context = SimpleNamespace(
        run_path=("subject_mask_runs/" if source_kind == "raw" else "refined_subject_masks_runs/") + "run-a",
        row_identity=identity,
        labels=labels,
        continuous_chain=chain,
        context_record=SimpleNamespace(record_sha256="context"),
    )
    surfaces = SimpleNamespace(
        centroid_xy=descriptor,
        context=context,
        inventory=SimpleNamespace(record_sha256="inventory"),
        derivation=SimpleNamespace(record_sha256="derivation"),
    )
    children = {
        f"{context.run_path}/metrics/centroid_valid": _Node(valid),
        f"{context.run_path}/available_channels": _Node(available),
        f"{context.run_path}/instance_key": _Node(np.arange(rows, dtype=np.uint64)),
        f"{context.run_path}/source_acquisition_frame_index": _Node(
            np.arange(rows, dtype=np.int64) + 100
        ),
    }
    root = _Group(
        children,
        attrs={},
    )
    family = "subject_mask_runs" if source_kind == "raw" else "refined_subject_masks_runs"
    root.children[family] = _Group(attrs={"latest": "run-a", "latest_complete": "run-a"})
    return root, surfaces, chain


def _patch(monkeypatch, *, source_kind="raw", labels=None, available=None):
    direct, surfaces, chain = _fake_source(
        source_kind=source_kind, labels=labels, available=available
    )
    consolidated, _, _ = _fake_source(
        source_kind=source_kind, labels=labels, available=available
    )
    monkeypatch.setattr(subject_source, "open_zarr_root", lambda *args, **kwargs: direct if not kwargs.get("use_consolidated") else consolidated)
    monkeypatch.setattr(subject_source, "validate_direct_consolidated_subtree", lambda *args, **kwargs: None)
    monkeypatch.setattr(subject_source, _loader_name(source_kind), lambda root, path: surfaces)
    monkeypatch.setattr(subject_source, "require_bound_directed_transform_chain", lambda value: value)
    monkeypatch.setattr(subject_source, "apply_bound_directed_transform_chain", _row_varying_projection)
    return chain


def _loader_name(source_kind):
    return (
        "load_persisted_subject_mask_coordinate_surfaces"
        if source_kind == "raw"
        else "load_persisted_refined_subject_mask_coordinate_surfaces"
    )


def _row_varying_projection(points, chain, *, row_identity):
    assert row_identity is chain.row_identity
    result = np.asarray(points, dtype=np.float64).copy()
    result[:, :, 0] += np.arange(result.shape[0], dtype=np.float64)[:, None]
    return result


def test_reordered_labels_fail_exact_declared_schema(monkeypatch):
    _patch(
        monkeypatch,
        labels=("eye_right", "swim_bladder", "subject_body", "eye_left"),
    )
    with pytest.raises(
        subject_source.SubjectMaskPositionSourceError,
        match="label order differs",
    ):
        subject_source.load_subject_mask_position_source(
            "/tmp/fake.zarr",
            run_path="subject_mask_runs/run-a",
            source_kind=subject_source.RAW_SUBJECT_MASK_SOURCE_KIND,
            anatomy_profile=_profile(),
            binding_id=BINDING_ID,
        )


def test_row_varying_projection_is_applied_without_reading_masks(monkeypatch):
    _patch(monkeypatch)
    bound = subject_source.load_subject_mask_position_source(
        "/tmp/fake.zarr",
        run_path="subject_mask_runs/run-a",
        source_kind="raw",
        anatomy_profile=_profile(),
        binding_id=BINDING_ID,
    )
    assert bound.centroid_xy_source_camera[1, 0, 0] == 2.0


def test_unavailable_required_channel_is_structural(monkeypatch):
    _patch(monkeypatch, available=(True, True, False, True))
    with pytest.raises(subject_source.SubjectMaskPositionSourceError, match="unavailable"):
        subject_source.load_subject_mask_position_source(
            "/tmp/fake.zarr",
            run_path="subject_mask_runs/run-a",
            source_kind="raw",
            anatomy_profile=_profile(),
            binding_id=BINDING_ID,
        )


def test_empty_component_rows_remain_valid_source_rows(monkeypatch):
    _patch(monkeypatch)
    bound = subject_source.load_subject_mask_position_source(
        "/tmp/fake.zarr",
        run_path="subject_mask_runs/run-a",
        source_kind="raw",
        anatomy_profile=_profile(),
        binding_id=BINDING_ID,
    )
    assert bound.centroid_valid[-1, -1] is np.False_
    assert bound.expression_bindings.components["swim_bladder"].valid[-1] is np.False_
    assert bound.source_row_index.tolist() == [0, 1]


def test_stale_family_selector_fails_closed(monkeypatch):
    _patch(monkeypatch)
    original = subject_source.open_zarr_root

    def stale(*args, **kwargs):
        root = original(*args, **kwargs)
        root.children["subject_mask_runs"].attrs["latest"] = "other-run"
        return root

    monkeypatch.setattr(subject_source, "open_zarr_root", stale)
    with pytest.raises(subject_source.SubjectMaskPositionSourceError, match="does not select"):
        subject_source.load_subject_mask_position_source(
            "/tmp/fake.zarr",
            run_path="subject_mask_runs/run-a",
            source_kind="raw",
            anatomy_profile=_profile(),
            binding_id=BINDING_ID,
        )


def test_raw_and_refined_source_kinds_use_distinct_families(monkeypatch):
    _patch(monkeypatch, source_kind="refined")
    bound = subject_source.load_subject_mask_position_source(
        "/tmp/fake.zarr",
        run_path="refined_subject_masks_runs/run-a",
        source_kind=subject_source.REFINED_SUBJECT_MASK_SOURCE_KIND,
        anatomy_profile=_profile(),
        binding_id=BINDING_ID,
    )
    assert bound.source_kind == "refined"
    with pytest.raises(subject_source.SubjectMaskPositionSourceError, match="cross-family"):
        subject_source.load_subject_mask_position_source(
            "/tmp/fake.zarr",
            run_path="subject_mask_runs/run-a",
            source_kind=subject_source.REFINED_SUBJECT_MASK_SOURCE_KIND,
            anatomy_profile=_profile(),
            binding_id=BINDING_ID,
        )


@pytest.mark.parametrize(
    ("estimator_id", "available"),
    (
        (
            MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
            (False, True, True, True),
        ),
        (
            SUBJECT_BODY_MASK_CENTROID_ESTIMATOR_ID,
            (True, False, False, False),
        ),
    ),
)
def test_estimator_loader_requires_only_its_declared_roles(
    monkeypatch,
    estimator_id,
    available,
):
    _patch(monkeypatch, available=available)
    bound = subject_source.load_subject_mask_position_source_for_estimator(
        "/tmp/fake.zarr",
        run_path="subject_mask_runs/run-a",
        source_kind=subject_source.RAW_SUBJECT_MASK_SOURCE_KIND,
        anatomy_profile=_profile(),
        binding_id=BINDING_ID,
        estimator_id=estimator_id,
    )
    assert bound._required_role_ids == (
        ("swim_bladder", "eye_left", "eye_right")
        if estimator_id == MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID
        else ("subject_body",)
    )


def test_estimator_loader_rejects_unregistered_mask_formula(monkeypatch):
    _patch(monkeypatch)
    with pytest.raises(
        subject_source.SubjectMaskPositionSourceError,
        match="Unsupported subject-mask position estimator",
    ):
        subject_source.load_subject_mask_position_source_for_estimator(
            "/tmp/fake.zarr",
            run_path="subject_mask_runs/run-a",
            source_kind=subject_source.RAW_SUBJECT_MASK_SOURCE_KIND,
            anatomy_profile=_profile(),
            binding_id=BINDING_ID,
            estimator_id="mask_union_centroid.v1",
        )
