from __future__ import annotations

from pathlib import Path

import numpy as np

from fisheye.utils import census_keypoint_training_sources as census


class _Array:
    def __init__(self, values: object) -> None:
        self._values = np.asarray(values)

    def __getitem__(self, key: object) -> np.ndarray:
        return self._values[key]


class _Group:
    def __init__(self, arrays: dict[str, object], attrs: dict[str, object]) -> None:
        self._arrays = {name: _Array(values) for name, values in arrays.items()}
        self.attrs = attrs

    def __getitem__(self, path: str) -> _Array:
        return self._arrays[path]


def test_leakage_group_prefers_subject_then_acquisition_cohort() -> None:
    assert census._leakage_group_id(
        recording_id="recording-a",
        subject_ids=("subject-2", "subject-1"),
        started_utc="2026-01-01T00:00:00Z",
    ) == ("subjects:subject-1,subject-2", "registered_subject")
    assert census._leakage_group_id(
        recording_id="recording-a",
        subject_ids=(),
        started_utc="2026-01-01T00:00:00Z",
    ) == (
        "acquisition_cohort:2026-01-01T00:00:00Z",
        "acquisition_start_fallback",
    )
    assert census._leakage_group_id(
        recording_id="recording-a",
        subject_ids=(),
        started_utc=None,
    ) == ("recording:recording-a", "recording_fallback")


def test_frame_domain_distinguishes_local_and_acquisition_indices() -> None:
    local = np.asarray([0, 1, 3], dtype=np.int64)
    acquisition = np.asarray([0, 5000, 15000], dtype=np.int64)

    assert (
        census._frame_domain(local, local=local, acquisition=acquisition)
        == "source_sample_row"
    )
    assert (
        census._frame_domain(acquisition, local=local, acquisition=acquisition)
        == "source_acquisition_frame"
    )
    assert (
        census._frame_domain(
            np.asarray([9, 10, 11]),
            local=local,
            acquisition=acquisition,
        )
        == "mismatch"
    )


def test_historical_comparison_reports_split_leakage_and_mixed_frame_domains(
    monkeypatch,
) -> None:
    root = _Group(
        {
            "source_index/source_dataset_idx": [0, 0, 1, 1],
            "source_index/source_frame_idx": [0, 1, 100, 200],
            "source_index/source_roi_idx": [0, 1, 0, 1],
            "source_index/source_detect_row_index": [10, 11, 20, 21],
            "source_index/source_refined_row_ids": [30, 31, 40, 41],
            "splits/train_indices": [0, 2],
            "splits/val_indices": [1, 3],
        },
        {
            "training_export": {
                "source_dataset_ids": ["recording-a:dataset", "recording-b:dataset"],
                "split": {"strategy": "global_random"},
            }
        },
    )
    monkeypatch.setattr(
        census, "open_zarr_group_direct", lambda *_args, **_kwargs: root
    )
    sources = [
        {"recording_id": "recording-a", "rows": {"usable": 2}},
        {"recording_id": "recording-b", "rows": {"usable": 2}},
    ]
    internals = {
        "recording-a": {
            "frame_local": np.asarray([0, 1]),
            "frame_acquisition": np.asarray([0, 10]),
            "roi_index": np.asarray([0, 1]),
            "source_detect_row_index": np.asarray([10, 11]),
            "source_refined_row_ids": np.asarray([30, 31]),
        },
        "recording-b": {
            "frame_local": np.asarray([0, 1]),
            "frame_acquisition": np.asarray([100, 200]),
            "roi_index": np.asarray([0, 1]),
            "source_detect_row_index": np.asarray([20, 21]),
            "source_refined_row_ids": np.asarray([40, 41]),
        },
    }

    result = census._historical_comparison(
        Path("historical.zarr"),
        sources=sources,
        internals=internals,
    )

    assert result["row_count_mismatches"] == {}
    assert result["lineage_array_mismatches"] == []
    assert result["source_frame_idx_domain_counts"] == {
        "source_acquisition_frame": 1,
        "source_sample_row": 1,
    }
    assert result["source_frame_idx_semantics_are_uniform"] is False
    assert result["split"]["source_overlap_count"] == 2
    assert result["split"]["source_overlap_fraction"] == 1.0
    assert result["split"]["leakage_safe"] is False
