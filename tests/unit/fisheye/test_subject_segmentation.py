from __future__ import annotations

import io

import numpy as np
import pytest
from rich.console import Console

from fisheye.segmentation import subject_segmentation as mod


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.dtype = self._data.dtype
        self.ndim = self._data.ndim

    def __getitem__(self, key):
        return self._data[key]


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


def _make_root(
    *,
    use_background_ds: bool = False,
    dish_mask: dict | None = None,
    roi_coordinates_full: np.ndarray | None = None,
) -> _FakeGroup:
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
                    [180, 40, 40, 180],
                    [180, 40, 40, 180],
                    [180, 180, 180, 180],
                ],
            ],
            dtype=np.uint8,
        ),
    )
    crop.create_array(
        "roi_coordinates_full",
        data=np.asarray(
            roi_coordinates_full
            if roi_coordinates_full is not None
            else [[0, 0], [2, 2]],
            dtype=np.int32,
        ),
    )
    crop.create_array("frame_indices", data=np.asarray([0, 1], dtype=np.int32))
    crop.create_array("frame_counts", data=np.asarray([1, 1], dtype=np.int32))
    crop.create_array("detection_indices", data=np.asarray([10, 11], dtype=np.int32))
    crop.create_array("detection_source", data=np.asarray([0, 1], dtype=np.int8))

    background_parent = root.require_group("background_runs")
    background_parent.attrs["latest"] = "background_001"
    background = background_parent.create_group("background_001")
    if use_background_ds:
        background.create_array(
            "background_ds",
            data=np.full((4, 4), 180, dtype=np.uint8),
        )
        raw_video = root.require_group("raw_video")
        raw_video.create_array("images_full", data=np.zeros((1, 8, 8), dtype=np.uint8))
    else:
        background.create_array(
            "background_full",
            data=np.full((8, 8), 180, dtype=np.uint8),
        )

    analysis = root.require_group("analysis_metadata")
    analysis.attrs[mod.tuning.TUNING_KEY] = {
        "version": "2.0",
        "components": {
            "subject_body": {
                "method": "traditional_subject_mask_seed",
                "version": "1.0",
                "tuned_timestamp": "2026-03-11T01:23:45+00:00",
                "tuned_parameters": {
                    "diff_threshold": 60,
                    "gaussian_blur_kernel": 4,
                    "closing_radius": 0,
                    "opening_radius": 0,
                    "min_area": 2,
                    "keep_largest_component": True,
                },
                "context": {"storage_component_name": "subject_body"},
            }
        },
    }
    if dish_mask is not None:
        analysis.attrs["dish_mask"] = dish_mask
    return root


@pytest.fixture(autouse=True)
def _stub_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod,
        "get_git_info",
        lambda: {
            "commit_hash": "a" * 40,
            "short_hash": "aaaaaaaa",
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


def test_segment_subject_masks_from_root_writes_body_only_run_using_saved_tuning() -> None:
    root = _make_root()
    console = Console(file=io.StringIO(), force_terminal=False)

    run_name = mod.segment_subject_masks_from_root(
        root,
        zarr_path="/tmp/fake_training.zarr",
        console=console,
        output_run="subject_masks_canary_001",
    )

    assert run_name == "subject_masks_canary_001"
    parent = root["subject_mask_runs"]
    assert parent.attrs["latest"] == "subject_masks_canary_001"
    run = parent["subject_masks_canary_001"]

    assert run.attrs["label_schema_id"] == "subject_v1_union"
    assert run.attrs["source_crop_run"] == "crop_001"
    assert run.attrs["source_background_run"] == "background_001"
    assert run.attrs["source_background_array"] == "background_full"
    assert run.attrs["run_semantics"] == "traditional_subject_body_inference"
    assert run.attrs["probability_semantics"] == "normalized_background_diff"
    assert run.attrs["tuning_source"] == "analysis_metadata.subject_mask_tuning.components.subject_body"
    assert run.attrs["config"]["gaussian_blur_kernel"] == 5
    assert run.attrs["tuning_entry_snapshot"]["method"] == "traditional_subject_mask_seed"
    assert run.attrs["tuning_entry_snapshot"]["tuned_timestamp"] == "2026-03-11T01:23:45+00:00"
    assert run.attrs["tuning_entry_snapshot"]["tuned_parameters"]["gaussian_blur_kernel"] == 4
    assert run.attrs["tuning_entry_snapshot"]["context"]["storage_component_name"] == "subject_body"

    np.testing.assert_array_equal(
        run["available_channels"][:],
        np.asarray([True, False, False], dtype=bool),
    )
    masks = run["masks_roi"][:]
    probs = run["mask_probs_roi"][:]
    assert masks.shape == (2, 3, 4, 4)
    assert probs.shape == (2, 3, 4, 4)
    assert np.any(masks[:, 0] > 0)
    assert not np.any(masks[:, 1] > 0)
    assert not np.any(masks[:, 2] > 0)

    metrics = run["metrics"]
    assert metrics["prob_max"][:].shape == (2, 3)
    assert metrics["mask_present"][:].tolist() == [[True, False, False], [True, False, False]]
    assert run.attrs["summary_statistics"]["rows_with_nonempty_masks"] == 2
    body_provenance = run["components"]["subject_body"]["provenance"].attrs
    assert body_provenance["source_stage"] == "subject_mask_runs"
    assert body_provenance["source_run"] == "subject_masks_canary_001"
    assert body_provenance["source_method"] == "traditional_subject_mask_seed"
    assert body_provenance["source_channels"] == ["subject_body"]
    assert body_provenance["source_label_schema_id"] == "subject_v1_union"
    assert "provenance" in run.attrs
    assert run.attrs["provenance"]["parameters"]["method"] == "traditional_subject_mask_seed"
    assert run.attrs["provenance"]["parameters"]["run_semantics"] == "traditional_subject_body_inference"
    assert (
        run.attrs["provenance"]["parameters"]["tuning_source"]
        == "analysis_metadata.subject_mask_tuning.components.subject_body"
    )
    assert run.attrs["provenance"]["parameters"]["tuning_timestamp"] == "2026-03-11T01:23:45+00:00"
    assert (
        run.attrs["provenance"]["parameters"]["tuning_entry_snapshot"]["tuned_parameters"]["gaussian_blur_kernel"]
        == 4
    )


def test_segment_subject_masks_from_root_supports_background_ds_and_overwrite() -> None:
    root = _make_root(use_background_ds=True)
    parent = root.require_group("subject_mask_runs")
    existing = parent.create_group("subject_masks_canary_001")
    existing.attrs["sentinel"] = "keep"

    with pytest.raises(ValueError, match="already exists"):
        mod.segment_subject_masks_from_root(
            root,
            zarr_path="/tmp/fake_training.zarr",
            output_run="subject_masks_canary_001",
            overwrite=False,
        )

    run_name = mod.segment_subject_masks_from_root(
        root,
        zarr_path="/tmp/fake_training.zarr",
        output_run="subject_masks_canary_001",
        overwrite=True,
        config_dict={"diff_threshold": 50, "keep_largest_component": False},
    )

    assert run_name == "subject_masks_canary_001"
    run = root["subject_mask_runs"]["subject_masks_canary_001"]
    assert "sentinel" not in run.attrs
    assert run.attrs["source_background_array"] == "background_ds"
    assert run.attrs["config"]["diff_threshold"] == 50
    assert run.attrs["config"]["keep_largest_component"] is False
    assert run.attrs["summary_statistics"]["background_array"] == "background_ds"
    np.testing.assert_array_equal(run["frame_indices"][:], np.asarray([0, 1], dtype=np.int32))


def test_segment_subject_masks_uses_open_zarr_root_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _make_root()
    monkeypatch.setattr(mod, "open_zarr_root", lambda *_args, **_kwargs: root)

    run_name = mod.segment_subject_masks(
        "/tmp/fake_training.zarr",
        output_run="subject_masks_cli_001",
    )

    assert run_name == "subject_masks_cli_001"
    assert "subject_masks_cli_001" in root["subject_mask_runs"]


def test_segment_subject_masks_applies_saved_dish_mask_projection() -> None:
    root = _make_root(
        dish_mask={
            "shape": "rectangle",
            "tuned_on_array": "images_full",
            "rectangle": {"roi": [0, 0, 3, 3]},
            "metrics": {"image_shape": [8, 8]},
        },
        roi_coordinates_full=np.asarray([[0, 0], [5, 5]], dtype=np.int32),
    )

    run_name = mod.segment_subject_masks_from_root(
        root,
        zarr_path="/tmp/fake_training.zarr",
        output_run="subject_masks_dish_gated_001",
    )

    assert run_name == "subject_masks_dish_gated_001"
    run = root["subject_mask_runs"][run_name]
    masks = run["masks_roi"][:]
    metrics = run["metrics"]

    assert int(masks[0, 0].sum()) > 0
    assert int(masks[1, 0].sum()) == 0
    assert metrics["mask_present"][:].tolist() == [[True, False, False], [False, False, False]]
    assert run.attrs["source_dish_mask_array"] == "images_full"
    assert run.attrs["summary_statistics"]["rows_with_nonempty_masks"] == 1
