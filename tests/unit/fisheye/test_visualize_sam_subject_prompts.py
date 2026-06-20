from __future__ import annotations

import numpy as np
import zarr

from fisheye.shared.mask_store import write_component_rle_mask_store_from_dense
from fisheye.visualization import visualize_sam_subject_prompts as mod


class FakeArray:
    def __init__(self, data: np.ndarray):
        self._data = np.asarray(data)
        self.shape = self._data.shape

    def __getitem__(self, item):
        return self._data[item]


def test_draw_prompt_overlay_supports_point_box_and_mask() -> None:
    roi = np.zeros((12, 16), dtype=np.uint8)
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[2:5, 3:7] = 1

    panel = mod.draw_prompt_overlay(
        roi,
        point_xy=np.asarray([5.0, 4.0], dtype=np.float32),
        box_xyxy=np.asarray([2.0, 1.0, 10.0, 8.0], dtype=np.float32),
        mask=mask,
        title="Prompt",
        footer_lines=("line1", "line2"),
    )

    assert panel.shape == (12, 16, 3)
    assert panel.dtype == np.uint8
    assert int(panel[:, :, 1].max()) > 0


def test_draw_prompt_overlay_supports_multiple_labeled_points() -> None:
    roi = np.zeros((12, 16), dtype=np.uint8)

    panel = mod.draw_prompt_overlay(
        roi,
        point_coords=np.asarray([[5.0, 4.0], [1.0, 1.0], [14.0, 10.0]], dtype=np.float32),
        point_labels=np.asarray([1, 0, 0], dtype=np.int32),
        title="Prompt",
    )

    assert panel.shape == (12, 16, 3)
    assert panel.dtype == np.uint8
    assert int(panel[:, :, 2].max()) > 0


def test_compose_prompt_grid_returns_2x2_layout() -> None:
    roi = np.zeros((10, 14), dtype=np.uint8)

    grid = mod._compose_prompt_grid(
        roi,
        point_coords=np.asarray([[3.0, 4.0], [1.0, 1.0]], dtype=np.float32),
        point_labels=np.asarray([1, 0], dtype=np.int32),
        box_xyxy=np.asarray([1.0, 2.0, 9.0, 7.0], dtype=np.float32),
        footer_lines=("eligible=True",),
    )

    assert grid.shape == (20, 28, 3)
    assert grid.dtype == np.uint8


def test_subject_body_mask_for_row_returns_none_when_channel_unavailable() -> None:
    loaded = mod.LoadedSubjectRun(
        run_name="subject_run_001",
        mask_labels=("subject_body", "eyes_union"),
        available_channels=np.asarray([False, True], dtype=bool),
        masks_roi=FakeArray(np.ones((2, 2, 5, 6), dtype=np.uint8)),
    )

    assert mod._subject_body_mask_for_row(loaded, 0) is None


def test_subject_body_mask_for_row_reads_available_channel() -> None:
    masks = np.zeros((2, 2, 5, 6), dtype=np.uint8)
    masks[1, 0, 1:3, 2:5] = 1
    loaded = mod.LoadedSubjectRun(
        run_name="subject_run_001",
        mask_labels=("subject_body", "eyes_union"),
        available_channels=np.asarray([True, False], dtype=bool),
        masks_roi=FakeArray(masks),
    )

    mask = mod._subject_body_mask_for_row(loaded, 1)

    assert mask is not None
    assert mask.shape == (5, 6)
    assert int(np.count_nonzero(mask)) == 6


def test_load_subject_run_reads_compact_mask_store(tmp_path) -> None:
    zarr_path = tmp_path / "subject_prompt_compact.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    parent = root.create_group("subject_mask_runs")
    parent.attrs["latest"] = "subject_compact_001"
    run = parent.create_group("subject_compact_001")
    run.attrs["mask_labels"] = ["subject_body", "eyes_union"]
    run.create_array("available_channels", data=np.asarray([True, False], dtype=bool), overwrite=True)
    masks = np.zeros((2, 2, 5, 6), dtype=np.uint8)
    masks[1, 0, 1:3, 2:5] = 1
    dense = run.create_array("masks_roi", data=masks, overwrite=True)
    write_component_rle_mask_store_from_dense(
        run,
        dense,
        component_names=("subject_body", "eyes_union"),
        encode_row_chunk_size=1,
    )
    del run["masks_roi"]
    run.attrs["masks_roi_materialized"] = False

    loaded = mod._load_subject_run(root, "subject_compact_001")
    mask = mod._subject_body_mask_for_row(loaded, 1)

    assert loaded is not None
    assert loaded.masks_roi is None
    assert loaded.mask_store is not None
    assert mask is not None
    assert mask.shape == (5, 6)
    assert int(np.count_nonzero(mask)) == 6
