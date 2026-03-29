from __future__ import annotations

import numpy as np
import pytest

import fisheye.tracking.single_subject_per_arena as mod
from fisheye.tracking.single_subject_per_arena import (
    TRACKING_METHOD_SINGLE_SUBJECT_PER_ARENA,
    TrackingConflictError,
    build_single_subject_per_arena_tracking,
    load_tracking_ids,
    write_single_subject_per_arena_tracking_run,
)


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data)

    def __getitem__(self, key):
        return self._data[key]

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape


class _FakeGroup(dict):
    def __init__(self, *args, attrs: dict | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def create_group(self, name: str):
        group = _FakeGroup()
        self[name] = group
        return group

    def create_array(self, name: str, data, **_kwargs):
        array = _FakeArray(np.asarray(data))
        self[name] = array
        return array

    def group_keys(self) -> list[str]:
        return [key for key, value in self.items() if isinstance(value, _FakeGroup)]

    def get(self, key: str, default=None):
        return super().get(key, default)


def _memory_root() -> _FakeGroup:
    return _FakeGroup()


@pytest.fixture(autouse=True)
def _stub_environment_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda: {
            "git": {
                "commit_hash": "a" * 40,
                "short_hash": "aaaaaaaa",
                "branch": "main",
                "is_dirty": False,
                "remote_url": "origin",
            },
            "platform": {
                "hostname": "test-host",
                "system": "Linux",
                "release": "test",
                "python_version": "3.11",
                "machine": "x86_64",
            },
            "environment": {},
        },
    )


def test_build_single_subject_per_arena_tracking_assigns_deterministic_track_ids() -> None:
    result = build_single_subject_per_arena_tracking(
        arena_ids=np.array([9, -1, 5, 9, 12, 5], dtype=np.int32),
        frame_indices=np.array([0, 0, 1, 2, 1, 3], dtype=np.int32),
    )

    assert result.track_ids.tolist() == [1, -1, 0, 1, 2, 0]
    assert result.track_ids_present.tolist() == [0, 1, 2]
    assert result.track_arena_ids.tolist() == [5, 9, 12]
    assert result.n_unassigned_rows == 1


def test_build_single_subject_per_arena_tracking_fails_on_same_frame_same_arena_conflict() -> None:
    with pytest.raises(TrackingConflictError, match="frame 10"):
        build_single_subject_per_arena_tracking(
            arena_ids=np.array([5, 5], dtype=np.int32),
            frame_indices=np.array([10, 10], dtype=np.int32),
        )


def test_build_tracking_qc_fields_stays_warn_while_recording_future_thresholds() -> None:
    warn_qc = mod.build_tracking_qc_fields(
        n_unassigned_rows=1,
        unassigned_row_rate_percent=0.25,
    )
    high_rate_qc = mod.build_tracking_qc_fields(
        n_unassigned_rows=1,
        unassigned_row_rate_percent=25.0,
    )

    assert warn_qc["tracking_qc_state"] == "warn"
    assert int(warn_qc["tracking_warn_threshold_rows"]) == 1
    assert float(warn_qc["tracking_block_threshold_percent"]) == pytest.approx(1.0)
    assert high_rate_qc["tracking_qc_state"] == "warn"


def test_load_tracking_ids_resolves_matching_lineage_not_latest() -> None:
    root = _memory_root()

    first_run, first_group, _ = write_single_subject_per_arena_tracking_run(
        root=root,
        arena_ids=np.array([2, -1, 4, 2], dtype=np.int32),
        frame_indices=np.array([0, 1, 2, 3], dtype=np.int32),
        source_detect_run="detect_a",
        source_arena_assignment_run="arena_assignment_a",
        source_rowset_path="detect_runs/detect_a",
    )

    second_run, _second_group, _ = write_single_subject_per_arena_tracking_run(
        root=root,
        arena_ids=np.array([7, 7], dtype=np.int32),
        frame_indices=np.array([0, 1], dtype=np.int32),
        source_detect_run="detect_b",
        source_arena_assignment_run="arena_assignment_b",
        source_rowset_path="detect_runs/detect_b",
    )

    assert root["tracking_runs"].attrs["latest"] == second_run

    track_ids, metadata = load_tracking_ids(
        root,
        4,
        expected_detect_run="detect_a",
        return_metadata=True,
    )

    assert track_ids.tolist() == [0, -1, 1, 0]
    assert metadata["track_run"] == first_run
    assert metadata["tracking_method"] == TRACKING_METHOD_SINGLE_SUBJECT_PER_ARENA
    assert metadata["source_arena_assignment_run"] == "arena_assignment_a"
    assert metadata["track_id_to_arena_id"] == {0: 2, 1: 4}
    assert first_group["track_arena_ids"][:].tolist() == [2, 4]


def test_load_tracking_ids_requires_matching_refined_lineage() -> None:
    root = _memory_root()

    write_single_subject_per_arena_tracking_run(
        root=root,
        arena_ids=np.array([0, 0, 0], dtype=np.int32),
        frame_indices=np.array([0, 1, 2], dtype=np.int32),
        source_detect_run="detect_a",
        source_refined_run="refined_a",
        source_arena_assignment_run="arena_assignment_a",
        source_rowset_path="refined_detect_runs/refined_a/interpolated",
    )

    with pytest.raises(ValueError, match="source_refined_run=refined_b"):
        load_tracking_ids(
            root,
            3,
            expected_detect_run="detect_a",
            expected_refined_run="refined_b",
        )
