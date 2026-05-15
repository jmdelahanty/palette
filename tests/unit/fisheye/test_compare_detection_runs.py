from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fisheye.diagnostics import compare_detection_runs as mod


class _Array:
    def __init__(self, data: object) -> None:
        self._data = np.asarray(data)

    def __getitem__(self, item: object) -> np.ndarray:
        return self._data[item]


class _Group(dict):
    def __init__(self, path: str = "") -> None:
        super().__init__()
        self.path = path
        self.attrs: dict[str, object] = {}


def _detect_group(path: str, *, bbox_offset: float = 0.0) -> _Group:
    group = _Group(path)
    group["n_detections"] = _Array([1, 0, 1])
    group["frame_indices"] = _Array([0, 2])
    group["bbox_norm_coords"] = _Array(
        [
            [0.1 + bbox_offset, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.1, 0.2],
        ]
    )
    group["scores"] = _Array([0.9, 0.8])
    group["class_ids"] = _Array([0, 0])
    return group


def test_compare_detection_runs_reports_box_and_count_parity(monkeypatch) -> None:
    root = _Group("/")
    parent = _Group("detect_runs")
    parent["run_a"] = _detect_group("detect_runs/run_a")
    parent["run_b"] = _detect_group("detect_runs/run_b", bbox_offset=0.01)
    root["detect_runs"] = parent

    monkeypatch.setattr(mod.zarr, "open_group", lambda *_args, **_kwargs: root)

    result = mod.compare_detection_runs(
        zarr_path=Path("/tmp/fake.zarr"),
        run_a="run_a",
        run_b="run_b",
        frames=[0, 1, 2],
    )

    assert result["frames_compared"] == 3
    assert result["detections_a"] == 2
    assert result["detections_b"] == 2
    assert result["count_mismatch_frames"] == 0
    assert result["bbox_abs_diff_max"] == pytest.approx(0.01)
