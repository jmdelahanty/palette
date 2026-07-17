from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.diagnostics.benchmark_eye_angle_materialization_layout import (
    _compare_digests,
    logical_run_digests,
)


def _encoded_names(names: list[str], width: int = 32) -> np.ndarray:
    values = np.zeros((len(names), width), dtype=np.uint8)
    for index, name in enumerate(names):
        encoded = name.encode("utf-8")
        values[index, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return values


def _write_run(path: Path, names: list[str], values: np.ndarray) -> None:
    run = zarr.open_group(str(path), mode="w", zarr_format=3)
    index = run.create_group("angle_channel_index")
    index.create_array("name", data=_encoded_names(names), chunks=(len(names), 32))
    run.create_array("roi_angles", data=values, chunks=(2, len(names)))
    support = run.create_group("support")
    support.create_array(
        "frame_indices",
        data=np.arange(values.shape[0], dtype=np.int64),
        chunks=(2,),
    )


def test_logical_run_digests_normalize_named_column_order(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    names = ["left_eye_angle_deg", "right_eye_angle_deg", "vergence_eye_angle_deg"]
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    permutation = [2, 0, 1]
    _write_run(left, names, values)
    _write_run(
        right,
        [names[index] for index in permutation],
        values[:, permutation],
    )

    left_digests = logical_run_digests(left, row_step=2)
    right_digests = logical_run_digests(right, row_step=2)

    assert left_digests == right_digests
    assert _compare_digests(left_digests, right_digests)["all_arrays_exact"] is True


def test_logical_run_digests_detect_value_changes(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    names = ["left_eye_angle_deg", "right_eye_angle_deg", "vergence_eye_angle_deg"]
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    changed = values.copy()
    changed[2, 1] += 1.0
    _write_run(left, names, values)
    _write_run(right, names, changed)

    comparison = _compare_digests(
        logical_run_digests(left, row_step=2),
        logical_run_digests(right, row_step=2),
    )

    assert comparison["all_arrays_exact"] is False
    assert comparison["digest_mismatches"] == ["roi_angles"]
