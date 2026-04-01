from __future__ import annotations

import io

import numpy as np
import pytest
from rich.console import Console

from fisheye.segmentation import swim_bladder_segmentation as mod


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.dtype = self._data.dtype
        self.ndim = self._data.ndim

    def __getitem__(self, key):
        return self._data[key]

    def __array__(self, dtype=None):
        return np.asarray(self._data, dtype=dtype)


class _FakeGroup(dict):
    def __init__(self, *args, attrs: dict | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def get(self, key: str, default=None):
        return super().get(key, default)

    def group_keys(self) -> list[str]:
        return [str(key) for key, value in self.items() if isinstance(value, _FakeGroup)]

    def create_group(self, name: str):
        group = _FakeGroup()
        self[name] = group
        return group

    def require_group(self, name: str):
        value = self.get(name)
        if isinstance(value, _FakeGroup):
            return value
        value = _FakeGroup()
        self[name] = value
        return value

    def create_array(self, name: str, data, **_kwargs):
        array = _FakeArray(np.asarray(data))
        self[name] = array
        return array


def _make_root() -> _FakeGroup:
    root = _FakeGroup(attrs={"width": 8, "height": 8})
    crop_parent = root.require_group("crop_runs")
    crop_parent.attrs["latest_materialized"] = "crop_001"
    crop = crop_parent.create_group("crop_001")
    crop.attrs["crop_storage_mode"] = "materialized"
    crop.create_array(
        "roi_images",
        data=np.asarray(
            [
                [
                    [180, 180, 180, 180],
                    [180, 20, 20, 180],
                    [180, 20, 20, 180],
                    [180, 180, 180, 180],
                ],
                [
                    [180, 180, 180, 180],
                    [180, 30, 30, 180],
                    [180, 30, 30, 180],
                    [180, 180, 180, 180],
                ],
            ],
            dtype=np.uint8,
        ),
    )
    crop.create_array(
        "roi_coordinates_full",
        data=np.asarray([[0, 0], [0, 0]], dtype=np.int32),
    )
    crop.create_array("frame_indices", data=np.asarray([0, 1], dtype=np.int32))
    crop.create_array("frame_counts", data=np.asarray([1, 1], dtype=np.int32))
    crop.create_array("detection_indices", data=np.asarray([10, 11], dtype=np.int32))
    crop.create_array("detection_source", data=np.asarray([0, 1], dtype=np.int8))

    analysis = root.require_group("analysis_metadata")
    analysis.attrs[mod.subject_tuning.TUNING_KEY] = {
        "version": "2.0",
        "components": {
            "swim_bladder": {
                "method": "global_threshold_otsu",
                "subject_method_family": "swim_bladder_patch_threshold_v1",
                "version": "1.0",
                "tuned_timestamp": "2026-03-12T10:00:00+00:00",
                "tuned_parameters": {
                    "roi_padding": 1,
                    "pre_threshold": 50,
                    "sobel_strength": 0.0,
                    "min_area": 1,
                    "max_area": None,
                    "min_circularity": None,
                    "closing_radius": 0,
                    "opening_radius": 0,
                },
                "context": {"storage_component_name": "swim_bladder"},
            }
        },
    }
    return root


@pytest.fixture(autouse=True)
def _stub_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod,
        "get_git_info",
        lambda: {
            "commit_hash": "b" * 40,
            "short_hash": "bbbbbbbb",
            "branch": "main",
            "is_dirty": False,
            "remote_url": "origin",
        },
    )
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda **_kwargs: {
            "environment": {},
            "platform": {
                "hostname": "test-host",
                "system": "Linux",
                "release": "test",
                "python_version": "3.11",
                "machine": "x86_64",
            },
        },
    )


def test_segment_swim_bladder_masks_from_root_writes_swim_channel_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _make_root()
    console = Console(file=io.StringIO(), force_terminal=False)
    keypoint_source = mod.subject_tuning.EyeKeypointSource(
        group_name="refined_keypoints_runs",
        run_name="refined_keypoints_canary_001",
        group=_FakeGroup(attrs={"keypoint_labels": ["swim_bladder", "eye_left", "eye_right"]}),
        keypoints_roi=_FakeArray(
            np.asarray(
                [
                    [[1.5, 1.5], [0.0, 0.0], [0.0, 0.0]],
                    [[np.nan, np.nan], [0.0, 0.0], [0.0, 0.0]],
                ],
                dtype=np.float32,
            )
        ),
        success_flags=np.asarray([True, True], dtype=bool),
        heading_values=None,
    )
    monkeypatch.setattr(mod.subject_tuning, "_resolve_eye_keypoint_source", lambda *_args, **_kwargs: keypoint_source)

    run_name = mod.segment_swim_bladder_masks_from_root(
        root,
        zarr_path="/tmp/fake_training.zarr",
        console=console,
        output_run="swim_bladder_masks_canary_001",
    )

    assert run_name == "swim_bladder_masks_canary_001"
    run = root["subject_mask_runs"][run_name]
    assert run.attrs["run_semantics"] == "traditional_swim_bladder_inference"
    assert run.attrs["probability_semantics"] == "normalized_patch_darkness"
    assert run.attrs["source_keypoints_run"] == "refined_keypoints_canary_001"
    assert run.attrs["source_keypoint_group"] == "refined_keypoints_runs"
    assert run.attrs["tuning_source"] == "analysis_metadata.subject_mask_tuning.components.swim_bladder"
    assert run.attrs["tuning_entry_snapshot"]["method"] == "global_threshold_otsu"
    assert run.attrs["tuning_entry_snapshot"]["subject_method_family"] == "swim_bladder_patch_threshold_v1"
    assert run.attrs["tuning_entry_snapshot"]["tuned_timestamp"] == "2026-03-12T10:00:00+00:00"
    assert run.attrs["tuning_entry_snapshot"]["tuned_parameters"]["roi_padding"] == 1

    np.testing.assert_array_equal(
        run["available_channels"][:],
        np.asarray([False, False, True], dtype=bool),
    )
    masks = run["masks_roi"][:]
    probs = run["mask_probs_roi"][:]
    assert masks.shape == (2, 3, 4, 4)
    assert probs.shape == (2, 3, 4, 4)
    assert int(masks[0, 2].sum()) > 0
    assert int(masks[1, 2].sum()) == 0
    assert not np.any(masks[:, 0] > 0)
    assert not np.any(masks[:, 1] > 0)

    metrics = run["metrics"]
    assert metrics["mask_present"][:].tolist() == [[False, False, True], [False, False, False]]
    assert run.attrs["summary_statistics"]["rows_with_nonempty_masks"] == 1
    assert run.attrs["summary_statistics"]["rows_skipped_missing_keypoint"] == 1
    assert run.attrs["summary_statistics"]["rows_skipped_unsuccessful_keypoint"] == 0
    swim_provenance = run["components"]["swim_bladder"]["provenance"].attrs
    assert swim_provenance["source_stage"] == "subject_mask_runs"
    assert swim_provenance["source_run"] == "swim_bladder_masks_canary_001"
    assert swim_provenance["source_method"] == "global_threshold_otsu"
    assert swim_provenance["source_channels"] == ["swim_bladder"]
    assert swim_provenance["source_label_schema_id"] == "subject_v1_union"
    assert run.attrs["provenance"]["parameters"]["run_semantics"] == "traditional_swim_bladder_inference"
    assert run.attrs["provenance"]["parameters"]["tuning_timestamp"] == "2026-03-12T10:00:00+00:00"
    assert run.attrs["provenance"]["parameters"]["tuning_entry_snapshot"]["subject_method_family"] == (
        "swim_bladder_patch_threshold_v1"
    )


def test_segment_swim_bladder_masks_skips_unsuccessful_keypoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _make_root()
    keypoint_source = mod.subject_tuning.EyeKeypointSource(
        group_name="refined_keypoints_runs",
        run_name="refined_keypoints_canary_001",
        group=_FakeGroup(attrs={"keypoint_labels": ["swim_bladder", "eye_left", "eye_right"]}),
        keypoints_roi=_FakeArray(
            np.asarray(
                [
                    [[1.5, 1.5], [0.0, 0.0], [0.0, 0.0]],
                    [[1.5, 1.5], [0.0, 0.0], [0.0, 0.0]],
                ],
                dtype=np.float32,
            )
        ),
        success_flags=np.asarray([True, False], dtype=bool),
        heading_values=None,
    )
    monkeypatch.setattr(mod.subject_tuning, "_resolve_eye_keypoint_source", lambda *_args, **_kwargs: keypoint_source)

    run_name = mod.segment_swim_bladder_masks_from_root(
        root,
        zarr_path="/tmp/fake_training.zarr",
        output_run="swim_bladder_masks_canary_001",
    )

    run = root["subject_mask_runs"][run_name]
    assert run.attrs["summary_statistics"]["rows_skipped_missing_keypoint"] == 0
    assert run.attrs["summary_statistics"]["rows_skipped_unsuccessful_keypoint"] == 1
    assert run["metrics"]["mask_present"][:].tolist() == [[False, False, True], [False, False, False]]


def test_segment_swim_bladder_masks_uses_open_zarr_root_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _make_root()
    keypoint_source = mod.subject_tuning.EyeKeypointSource(
        group_name="refined_keypoints_runs",
        run_name="refined_keypoints_canary_001",
        group=_FakeGroup(attrs={"keypoint_labels": ["swim_bladder", "eye_left", "eye_right"]}),
        keypoints_roi=_FakeArray(
            np.asarray([[[1.5, 1.5], [0.0, 0.0], [0.0, 0.0]]], dtype=np.float32).repeat(2, axis=0)
        ),
        success_flags=np.asarray([True, True], dtype=bool),
        heading_values=None,
    )
    monkeypatch.setattr(mod.subject_tuning, "_resolve_eye_keypoint_source", lambda *_args, **_kwargs: keypoint_source)
    monkeypatch.setattr(mod, "open_zarr_root", lambda *_args, **_kwargs: root)

    run_name = mod.segment_swim_bladder_masks(
        "/tmp/fake_training.zarr",
        output_run="swim_bladder_masks_cli_001",
    )

    assert run_name == "swim_bladder_masks_cli_001"
    assert "swim_bladder_masks_cli_001" in root["subject_mask_runs"]
