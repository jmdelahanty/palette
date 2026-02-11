from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import zarr

from fisheye.analysis import import_stimulus_to_zarr as mod


def _write_minimal_stimulus_h5(path: Path) -> None:
    dtype = np.dtype(
        [
            ("stimulus_frame_num", np.uint64),
            ("triggering_camera_frame_id", np.uint64),
            ("timestamp_ns", np.int64),
        ]
    )
    frame_metadata = np.array(
        [
            (1000, 2000, 1_000_000_000),
            (1001, 2001, 1_008_333_333),
        ],
        dtype=dtype,
    )
    with h5py.File(path, "w") as h5:
        video_metadata = h5.create_group("video_metadata")
        video_metadata.create_dataset("frame_metadata", data=frame_metadata)


def test_import_sets_source_stimulus_video_path_when_rendered_mp4_exists(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    rendered_mp4 = tmp_path / "session.mp4"
    zarr_path = tmp_path / "sample_analysis.zarr"

    _write_minimal_stimulus_h5(h5_path)
    rendered_mp4.touch()
    zarr.open_group(str(zarr_path), mode="w")

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_test",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run_group = root["analysis"]["stimulus_runs"][run_name]
    assert run_group.attrs.get("source_h5") == str(h5_path.resolve())
    assert run_group.attrs.get("source_stimulus_video_path") == str(rendered_mp4.resolve())


def test_import_omits_source_stimulus_video_path_when_rendered_mp4_missing(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"

    _write_minimal_stimulus_h5(h5_path)
    zarr.open_group(str(zarr_path), mode="w")

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_test_no_video",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run_group = root["analysis"]["stimulus_runs"][run_name]
    assert run_group.attrs.get("source_h5") == str(h5_path.resolve())
    assert "source_stimulus_video_path" not in run_group.attrs
