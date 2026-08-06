from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from fisheye.labeling.web import _keypoint_session_html
from fisheye.labeling.web_runtimes import (
    KeypointRuntimeSession,
    _keypoint_runtime_state,
)


class _Backend:
    @staticmethod
    def review_session_summary(_session: object) -> dict[str, object]:
        return {"status": "ready"}


def _runtime(*, immutable_base: bool) -> KeypointRuntimeSession:
    review_session = SimpleNamespace(
        failures=np.asarray([0], dtype=np.int32),
        frame_indices=np.asarray([17], dtype=np.int32),
        refined=SimpleNamespace(attrs={}),
        zarr_path="/groups/private/review.zarr",
        refined_run="refined_v2",
        crop_run="crop_v2",
        keypoint_labels=["snout_tip"],
        immutable_base=immutable_base,
        delta_run="manual_edits" if immutable_base else None,
        delta_generation="generation_000001" if immutable_base else None,
    )
    return KeypointRuntimeSession(
        session_id="session-a",
        task_id="task-a",
        recording_id="recording-a",
        user="reviewer",
        review_session=review_session,
    )


def test_keypoint_runtime_declares_immutable_delta_edit_storage() -> None:
    state = _keypoint_runtime_state(_runtime(immutable_base=True), _Backend)

    assert state["immutable_base"] is True
    assert state["edit_storage"] == "delta_generation"
    assert state["delta_run"] == "manual_edits"
    assert state["delta_generation"] == "generation_000001"


def test_keypoint_runtime_declares_legacy_mutable_edit_storage() -> None:
    state = _keypoint_runtime_state(_runtime(immutable_base=False), _Backend)

    assert state["immutable_base"] is False
    assert state["edit_storage"] == "in_place"
    assert state["delta_run"] is None
    assert state["delta_generation"] is None


def test_keypoint_web_explains_delta_review_completion_boundary() -> None:
    html = _keypoint_session_html(
        {
            "session_id": "session-a",
            "task_id": "task-a",
            "title": "Keypoint task",
            "expires_at_utc": "2026-08-06T12:30:00+00:00",
        }
    ).decode("utf-8")

    assert 'id="mutable-review-controls"' in html
    assert 'id="immutable-delta-review-note"' in html
    assert "freeze, validate, and compact" in html
    assert 'state.edit_storage === "delta_generation"' in html

