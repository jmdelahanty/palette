from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.tracking import crop as crop_mod


class _FakeVideoCapture:
    def __init__(self, frames: dict[int, np.ndarray] | None = None) -> None:
        self.frames = frames or {}
        self.position = 0

    def set(self, _prop: int, value: float) -> bool:
        self.position = int(value)
        return True

    def read(self):  # noqa: ANN001
        frame = self.frames.get(self.position)
        if frame is None:
            return False, None
        return True, frame.copy()

    def release(self) -> None:
        pass


def test_crop_batch_cpu_rejects_out_of_bounds_frame_index() -> None:
    with pytest.raises(ValueError, match=r"Out-of-range frame index.*detect_runs/source.*frame_indices\[0\]=2"):
        crop_mod.crop_batch_cpu(
            "video.mp4",
            np.array([2], dtype=np.int64),
            np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32),
            (4, 4),
            (10, 10),
            total_frames=2,
            source_label="detect_runs/source",
        )


def test_crop_batch_cpu_decode_failure_raises_instead_of_black_crop(monkeypatch) -> None:
    monkeypatch.setattr(crop_mod.cv2, "VideoCapture", lambda _path: _FakeVideoCapture())

    with pytest.raises(RuntimeError, match=r"CPU crop decode failed for frame\(s\) 0.*detect_runs/source"):
        crop_mod.crop_batch_cpu(
            "video.mp4",
            np.array([0], dtype=np.int64),
            np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32),
            (4, 4),
            (10, 10),
            total_frames=1,
            source_label="detect_runs/source",
        )


def test_crop_batch_cpu_from_top_left_decode_failure_raises_instead_of_black_crop(monkeypatch) -> None:
    monkeypatch.setattr(crop_mod.cv2, "VideoCapture", lambda _path: _FakeVideoCapture())

    with pytest.raises(RuntimeError, match=r"CPU crop decode failed for frame\(s\) 0.*crop_runs/source"):
        crop_mod.crop_batch_cpu_from_top_left(
            "video.mp4",
            np.array([0], dtype=np.int64),
            np.array([[2, 3]], dtype=np.int32),
            (4, 4),
            (10, 10),
            total_frames=1,
            source_label="crop_runs/source",
        )


def test_materialized_zarr_worker_rejects_unsorted_source_rows(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["current_crop_group_path"] = "crop_runs/crop_001"
    raw = root.create_group("raw_video")
    raw.create_array("images_full", data=np.zeros((3, 8, 8), dtype=np.uint8), overwrite=True)
    crop_parent = root.create_group("crop_runs")
    crop_parent.create_group("crop_001")
    detect_parent = root.create_group("detect_runs")
    detect = detect_parent.create_group("detect_001")
    detect.create_array("frame_indices", data=np.array([0, 2, 1], dtype=np.int32), overwrite=True)
    detect.create_array(
        "bbox_norm_coords",
        data=np.array(
            [[0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]],
            dtype=np.float32,
        ),
        overwrite=True,
    )

    task = crop_mod.crop_and_store_chunk_delayed(
        str(zarr_path),
        slice(0, 3),
        (0, 3),
        (4, 4),
        1.0,
        "detect_runs/detect_001",
    )
    with pytest.raises(ValueError, match=r"detect_runs/detect_001.*first out-of-order row at position 2"):
        task.compute()


def test_save_crop_metadata_rejects_out_of_bounds_frame_counts(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    source = root.create_group("detect_runs/detect_001")
    source.create_array("frame_indices", data=np.array([0, 3], dtype=np.int32), overwrite=True)
    source.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]], dtype=np.float32),
        overwrite=True,
    )
    crop_group = root.create_group("crop_runs/crop_001")

    with pytest.raises(ValueError, match=r"Out-of-range frame index.*detect_runs/detect_001.*frame_indices\[1\]=3"):
        crop_mod.save_crop_metadata(
            crop_group=crop_group,
            source_group=source,
            source_path="detect_runs/detect_001",
            source_type="detect",
            detection_source=None,
            total_detections=2,
            num_frames=3,
        )
