from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.analysis.chaser_metrics_loader import load_chaser_metrics
from fisheye.analysis.chaser_state_interpolator import write_columnar_dataset


def _write_minimal_stimulus_run(zarr_path: Path) -> None:
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
    write_columnar_dataset(stim.create_group("tracking_data"), "chaser_states", chaser_states)


def test_load_chaser_metrics_missing_legacy_metrics_group_is_read_only_safe(tmp_path: Path) -> None:
    zarr_path = tmp_path / "missing_metrics.zarr"
    _write_minimal_stimulus_run(zarr_path)

    bundle = load_chaser_metrics(zarr_path)

    np.testing.assert_array_equal(bundle.camera_frame_ids, np.asarray([10, 11, 12], dtype=np.int64))
    np.testing.assert_allclose(bundle.online["chaser_pos_x"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(bundle.offline["has_offline"], np.zeros(3, dtype=bool))
    assert bundle.provenance["metrics_run"] is None

    root = zarr.open_group(str(zarr_path), mode="r")
    assert "chaser_fish_metrics" not in root["analysis"]
