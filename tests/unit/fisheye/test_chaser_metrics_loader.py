from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.analysis.chaser_metrics_loader import load_chaser_metrics
from fisheye.shared.zarr.columnar import write_columnar_dataset


def _write_minimal_stimulus_run(
    zarr_path: Path,
    *,
    chaser_state_attrs: dict[str, object] | None = None,
) -> None:
    root = zarr.open_group(str(zarr_path), mode="w")
    analysis = root.create_group("analysis")
    stimulus_parent = analysis.create_group("stimulus_runs")
    stimulus_parent.attrs["latest"] = "stim_1"
    stim = stimulus_parent.create_group("stim_1")

    frame_dtype = np.dtype(
        [
            ("stimulus_frame_num", np.int64),
            ("triggering_camera_frame_id", np.int64),
            ("timestamp_ns", np.int64),
        ]
    )
    frame_metadata = np.zeros(3, dtype=frame_dtype)
    frame_metadata["stimulus_frame_num"] = [0, 1, 2]
    frame_metadata["triggering_camera_frame_id"] = [10, 11, 12]
    frame_metadata["timestamp_ns"] = [0, 10_000_000, 20_000_000]
    write_columnar_dataset(stim.create_group("video_metadata"), "frame_metadata", frame_metadata)

    chaser_dtype = np.dtype(
        [
            ("stimulus_frame_num", np.int64),
            ("chaser_index", np.int16),
            ("trial_state", np.int16),
            ("chaser_pos_x", np.float32),
            ("chaser_pos_y", np.float32),
        ]
    )
    chaser_states = np.zeros(3, dtype=chaser_dtype)
    chaser_states["stimulus_frame_num"] = [0, 1, 2]
    chaser_states["chaser_index"] = 0
    chaser_states["trial_state"] = [1, 1, 1]
    chaser_states["chaser_pos_x"] = [1.0, 2.0, 3.0]
    chaser_states["chaser_pos_y"] = [4.0, 5.0, 6.0]
    write_columnar_dataset(
        stim.create_group("tracking_data"),
        "chaser_states",
        chaser_states,
        chaser_state_attrs,
    )


def test_load_chaser_metrics_missing_legacy_metrics_group_is_read_only_safe(tmp_path: Path) -> None:
    zarr_path = tmp_path / "missing_metrics.zarr"
    _write_minimal_stimulus_run(zarr_path)

    bundle = load_chaser_metrics(zarr_path)

    np.testing.assert_array_equal(bundle.camera_frame_ids, np.asarray([10, 11, 12], dtype=np.int64))
    np.testing.assert_allclose(bundle.online["chaser_pos_x"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(bundle.offline["has_offline"], np.zeros(3, dtype=bool))
    assert bundle.provenance["metrics_run"] is None
    assert bundle.online_coordinate_metadata["source_path"] == (
        "analysis/stimulus_runs/stim_1/tracking_data/chaser_states"
    )
    source_attrs = bundle.online_coordinate_metadata["source_attrs"]
    assert isinstance(source_attrs, dict)
    assert "coordinate_frame" not in source_attrs
    assert "position_fields" not in source_attrs

    root = zarr.open_group(str(zarr_path), mode="r")
    assert "chaser_fish_metrics" not in root["analysis"]


def test_load_chaser_metrics_preserves_exact_online_coordinate_metadata(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "coordinate_metadata.zarr"
    coordinate_attrs = {
        "coordinate_frame": "arena_relative_canvas_px",
        "coordinate_origin": "top_left_of_active_arena",
        "position_fields": "chaser_pos_x,chaser_pos_y",
        "x_axis_direction": "right",
        "y_axis_direction": "down",
    }
    _write_minimal_stimulus_run(
        zarr_path,
        chaser_state_attrs=coordinate_attrs,
    )

    bundle = load_chaser_metrics(zarr_path)

    source_path = "analysis/stimulus_runs/stim_1/tracking_data/chaser_states"
    root = zarr.open_group(str(zarr_path), mode="r")
    source_group = root[source_path]
    assert bundle.online_coordinate_metadata == {
        "source_path": source_path,
        "source_attrs": dict(source_group.attrs),
    }
    assert bundle.online_coordinate_metadata["source_attrs"]["position_fields"] == (
        "chaser_pos_x,chaser_pos_y"
    )
