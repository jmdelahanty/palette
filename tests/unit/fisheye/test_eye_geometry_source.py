from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fisheye.shared.eye_geometry_source import EYE_GEOMETRY_STAGE_REFINED_SUBJECT
from fisheye.shared.eye_geometry_source import resolve_eye_geometry_source
from fisheye.utils import export_eye_mask_training_zarr as export_mod


class _FakeArray:
    def __init__(self, data: Any) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.ndim = self._data.ndim
        self.dtype = self._data.dtype
        self.chunks = self.shape

    def __getitem__(self, key):
        return self._data[key]


class _FakeGroup(dict):
    def __init__(self, *args, attrs: dict[str, Any] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def _resolve_parent(self, path: str) -> tuple["_FakeGroup", str]:
        tokens = [token for token in path.split("/") if token]
        if not tokens:
            raise KeyError(path)
        parent: _FakeGroup = self
        for token in tokens[:-1]:
            child = parent.get(token)
            if not isinstance(child, _FakeGroup):
                child = _FakeGroup()
                parent[token] = child
            parent = child
        return parent, tokens[-1]

    def require_group(self, path: str) -> "_FakeGroup":
        parent, name = self._resolve_parent(path)
        child = parent.get(name)
        if isinstance(child, _FakeGroup):
            return child
        child = _FakeGroup()
        parent[name] = child
        return child

    def create_array(self, path: str, *, data: Any, **_kwargs) -> _FakeArray:
        parent, name = self._resolve_parent(path)
        array = _FakeArray(data)
        parent[name] = array
        return array

    def group_keys(self) -> list[str]:
        return [str(key) for key, value in self.items() if isinstance(value, _FakeGroup)]

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key: str):
        tokens = [token for token in key.split("/") if token]
        if not tokens:
            raise KeyError(key)
        current: Any = self
        for token in tokens:
            if not isinstance(current, _FakeGroup):
                raise KeyError(key)
            current = dict.__getitem__(current, token)
        return current

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        try:
            self[key]
            return True
        except KeyError:
            return False


def _add_refined_eye_run(root: _FakeGroup, run_name: str = "refined_eye_001") -> _FakeGroup:
    parent = root.require_group("refined_eye_masks_runs")
    parent.attrs["latest"] = run_name
    run = parent.require_group(run_name)
    run.attrs.update(
        {
            "source_keypoints_run": "refined_kp_001",
            "source_keypoint_run": "refined_kp_001",
            "source_keypoint_group": "refined_keypoints_runs",
        }
    )
    run.create_array("masks_roi", data=np.arange(2 * 2 * 3 * 4, dtype=np.uint8).reshape(2, 2, 3, 4))
    run.create_array("ellipse_params", data=np.arange(2 * 2 * 5, dtype=np.float32).reshape(2, 2, 5))
    run.create_array("ellipse_success", data=np.asarray([[True, False], [True, True]], dtype=bool))
    run.create_array("eye_separation", data=np.asarray([4.0, 5.0], dtype=np.float32))
    return run


def _add_refined_subject_run(root: _FakeGroup, run_name: str = "refined_subject_001") -> _FakeGroup:
    parent = root.require_group("refined_subject_masks_runs")
    parent.attrs["latest"] = run_name
    run = parent.require_group(run_name)
    run.attrs.update(
        {
            "mask_labels": ["subject_body", "eye_left", "eye_right", "swim_bladder"],
            "source_refined_eye_masks_run": "refined_eye_001",
        }
    )
    masks = np.zeros((2, 4, 3, 4), dtype=np.uint8)
    masks[:, 1] = 11
    masks[:, 2] = 22
    run.create_array("masks_roi", data=masks)
    run.create_array(
        "components/eye_left/geometry/ellipse_params",
        data=np.full((2, 5), 1.0, dtype=np.float32),
    )
    run.create_array(
        "components/eye_right/geometry/ellipse_params",
        data=np.full((2, 5), 2.0, dtype=np.float32),
    )
    run.create_array("components/eye_left/geometry/ellipse_success", data=np.asarray([True, False], dtype=bool))
    run.create_array("components/eye_right/geometry/ellipse_success", data=np.asarray([True, True], dtype=bool))
    run.create_array("relations/eye_pair/metrics/separation_px", data=np.asarray([7.0, 8.0], dtype=np.float32))
    return run


def test_resolver_prefers_latest_refined_subject_geometry() -> None:
    root = _FakeGroup()
    _add_refined_eye_run(root)
    _add_refined_subject_run(root)

    source = resolve_eye_geometry_source(root)

    assert source.stage_group == EYE_GEOMETRY_STAGE_REFINED_SUBJECT
    assert source.run_name == "refined_subject_001"
    assert source.source_refined_eye_run == "refined_eye_001"
    expected_masks = np.stack(
        [
            np.full((3, 4), 11, dtype=np.uint8),
            np.full((3, 4), 22, dtype=np.uint8),
        ],
        axis=0,
    )
    np.testing.assert_array_equal(np.asarray(source.masks_roi[0]), expected_masks)
    assert source.ellipse_params.shape == (2, 2, 5)
    np.testing.assert_array_equal(np.asarray(source.ellipse_params[:, :, 0]), np.asarray([[1.0, 2.0], [1.0, 2.0]]))
    np.testing.assert_array_equal(np.asarray(source.ellipse_success[:]), np.asarray([[True, True], [False, True]]))


def test_explicit_refined_subject_run_resolves_canonical_subject() -> None:
    root = _FakeGroup()
    eye = _add_refined_eye_run(root)
    eye.attrs["source_refined_subject_masks_run"] = "refined_subject_001"
    _add_refined_subject_run(root)

    source = resolve_eye_geometry_source(root, refined_subject_run="refined_subject_001")

    assert source.stage_group == EYE_GEOMETRY_STAGE_REFINED_SUBJECT
    assert source.run_name == "refined_subject_001"
    assert source.source_refined_eye_run == "refined_eye_001"


def test_resolver_rejects_legacy_only_refined_eye_masks() -> None:
    root = _FakeGroup()
    _add_refined_eye_run(root)

    with pytest.raises(ValueError, match="No canonical subject-shape or refined-subject eye geometry found"):
        resolve_eye_geometry_source(root)


def test_eye_mask_training_export_auto_selects_refined_subject_geometry() -> None:
    root = _FakeGroup()
    _add_refined_eye_run(root)
    subject = _add_refined_subject_run(root)
    subject.attrs["source_crop_run"] = "crop_001"
    subject.attrs["source_subject_mask_run"] = "subject_masks_001"

    crop_parent = root.require_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    crop = crop_parent.require_group("crop_001")
    crop.create_array("roi_images", data=np.zeros((2, 3, 4), dtype=np.uint8))
    crop.create_array("bbox_norm_coords", data=np.zeros((2, 4), dtype=np.float32))

    selection, _crop_group, _eye_group, geometry = export_mod._select_source_runs(
        root,
        crop_run=None,
        eye_stage="auto",
        eye_run=None,
    )

    assert selection.eye_stage == EYE_GEOMETRY_STAGE_REFINED_SUBJECT
    assert selection.eye_stage_role == "canonical"
    assert selection.authority_stage_group == EYE_GEOMETRY_STAGE_REFINED_SUBJECT
    assert selection.eye_run == "refined_subject_001"
    assert selection.source_refined_subject_masks_run == "refined_subject_001"
    assert selection.source_subject_mask_run == "subject_masks_001"
    assert geometry is not None
    np.testing.assert_array_equal(np.asarray(geometry.masks_roi[0, :, 0, 0]), np.asarray([11, 22], dtype=np.uint8))
