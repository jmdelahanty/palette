from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fisheye.diagnostics import benchmark_canonical_detection_storage as benchmark
from fisheye.diagnostics.benchmark_canonical_detection_storage import (
    build_canonical_detection_benchmark_input,
)
from fisheye.shared.zarr.detection_schema import CANONICAL_DETECTION_SCHEMA_V1


class _Array:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def __getitem__(self, selection):
        return self.values[selection]


def _source_arrays() -> dict[str, _Array]:
    return {
        "frame_indices": _Array(np.asarray([0, 0, 2, 4], dtype=np.int32)),
        "bbox_norm_coords": _Array(
            np.asarray(
                [
                    [0.5, 0.5, 0.2, 0.2],
                    [0.4, 0.4, 0.1, 0.1],
                    [0.6, 0.6, 0.2, 0.2],
                    [0.3, 0.3, 0.1, 0.1],
                ],
                dtype=np.float64,
            )
        ),
        "scores": _Array(np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32)),
        "class_ids": _Array(np.asarray([0, 1, 0, 1], dtype=np.int32)),
    }


def test_fixture_conversion_builds_exact_short_canonical_run() -> None:
    benchmark_input = build_canonical_detection_benchmark_input(
        _source_arrays(),
        recording_identity="recording-a",
        frame_count=5,
        source_width=640,
        source_height=480,
        frame_limit=3,
    )

    assert benchmark_input.dimensions.n_frames == 3
    assert benchmark_input.dimensions.n_instances == 3
    assert (
        CANONICAL_DETECTION_SCHEMA_V1.validate(
            benchmark_input.arrays,
            dimensions=benchmark_input.dimensions,
        )
        == ()
    )
    assert benchmark_input.arrays["instances/bbox_norm_coords"].dtype == np.float32
    assert benchmark_input.arrays["instances/frame_row_offsets"].tolist() == [
        0,
        2,
        2,
        3,
    ]
    assert benchmark_input.source_identity["conversion"]["bbox_norm_coords"] == (
        "float64->float32"
    )


def test_fixture_conversion_rejects_unsorted_source_rows() -> None:
    source = _source_arrays()
    source["frame_indices"] = _Array(np.asarray([0, 2, 1, 4], dtype=np.int32))

    with pytest.raises(ValueError, match="frame sorted"):
        build_canonical_detection_benchmark_input(
            source,
            recording_identity="recording-a",
            frame_count=5,
            source_width=640,
            source_height=480,
        )


def test_destination_guard_allows_only_fresh_tmp_benchmark_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark, "BENCHMARK_OUTPUT_ROOT", tmp_path)
    safe = tmp_path / "test-fixture" / "new-candidate.zarr"
    assert benchmark._require_safe_destination(safe) == safe.resolve()

    with pytest.raises(ValueError, match="must be below"):
        benchmark._require_safe_destination(
            tmp_path.parent / f"{tmp_path.name}-outside.zarr"
        )
    with pytest.raises(ValueError, match="must be below"):
        benchmark._require_safe_destination(tmp_path)
    existing = tmp_path / "existing.zarr"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        benchmark._require_safe_destination(existing)
