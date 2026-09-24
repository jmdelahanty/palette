"""Presentation filters must never expand an assigned keypoint row scope."""

from types import SimpleNamespace

import numpy as np
import pytest

from fisheye.labeling.web_runtimes import (
    KeypointRuntimeSession,
    _refresh_keypoint_queue,
)
from fisheye.tune import keypoint_review_backend as backend


def _runtime(scope, mode="all"):
    session = SimpleNamespace(
        frame_indices=np.arange(6),
        failures=np.asarray([1, 4], dtype=np.int32),
        refined={"failure_indices": np.asarray([0, 1, 3, 4], dtype=np.int32)},
        edit_applied_arr=np.asarray([True, True, False, False, True, False]),
        usable_arr=np.asarray([False, True, False, False, True, True]),
        refined_success_arr=np.asarray([False, False, True, False, False, True]),
        reason_arr=np.asarray(
            ["manual_correction", "manual_correction", "", "", "manual_correction", ""]
        ),
    )
    return KeypointRuntimeSession(
        session_id="session",
        task_id="task",
        recording_id="recording",
        user="reviewer",
        review_session=session,
        position=5,
        filter_mode=mode,
        task_roi_indices=None if scope is None else np.asarray(scope, dtype=np.int64),
    )


@pytest.mark.parametrize("mode", ["all", "edited", "manual", "usable", "raw_failed", "failed"])
def test_filter_keeps_only_task_rows_and_clamps_navigation(mode):
    runtime = _runtime([1, 4], mode)

    _refresh_keypoint_queue(runtime, backend)

    np.testing.assert_array_equal(runtime.review_session.failures, [1, 4])
    np.testing.assert_array_equal(runtime.task_roi_indices, [1, 4])
    assert runtime.position == 1


def test_empty_task_scope_stays_empty():
    runtime = _runtime([])
    _refresh_keypoint_queue(runtime, backend)
    assert runtime.review_session.failures.size == 0
    assert runtime.position == 0


def test_unscoped_legacy_session_retains_all_rows():
    runtime = _runtime(None)
    _refresh_keypoint_queue(runtime, backend)
    np.testing.assert_array_equal(runtime.review_session.failures, np.arange(6))
    assert runtime.position == 5


def test_filter_cannot_reintroduce_out_of_scope_search_match(monkeypatch):
    runtime = _runtime([1, 4])
    runtime.search = "roi=3"
    monkeypatch.setattr(
        backend,
        "summarize_roi",
        lambda session, row: {"frame_idx": row, "reason": "", "status": {}},
    )
    _refresh_keypoint_queue(runtime, backend)
    assert runtime.review_session.failures.size == 0
    runtime.search = ""
    _refresh_keypoint_queue(runtime, backend)
    np.testing.assert_array_equal(runtime.review_session.failures, [1, 4])
