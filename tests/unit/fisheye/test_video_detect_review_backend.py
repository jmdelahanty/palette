from __future__ import annotations

import numpy as np
import pytest

from fisheye.tune import video_detect_review_backend as backend
from fisheye.tune import video_detect_review_web as web


def test_parse_range_header_supports_standard_byte_ranges() -> None:
    assert web._parse_range_header("bytes=0-99", file_size=1000) == (0, 99)
    assert web._parse_range_header("bytes=100-", file_size=1000) == (100, 999)
    assert web._parse_range_header("bytes=-25", file_size=1000) == (975, 999)
    assert web._parse_range_header(None, file_size=1000) is None


def test_parse_range_header_rejects_unsatisfiable_ranges() -> None:
    with pytest.raises(ValueError):
        web._parse_range_header("bytes=1000-1001", file_size=1000)
    with pytest.raises(ValueError):
        web._parse_range_header("items=0-1", file_size=1000)


def test_parse_save_frame_path() -> None:
    assert web._parse_save_frame_from_path("/api/frame/42/save") == 42
    assert web._parse_save_frame_from_path("/api/frame/current/save") is None
    assert web._parse_save_frame_from_path("/api/frame/nope/save") is None


def test_normalize_bbox_or_none_clips_center_and_requires_positive_extent() -> None:
    bbox = backend._normalize_bbox_or_none([1.2, -0.2, 0.3, 0.4])
    np.testing.assert_allclose(bbox, [1.0, 0.0, 0.3, 0.4])
    assert backend._normalize_bbox_or_none(None) is None
    with pytest.raises(ValueError):
        backend._normalize_bbox_or_none([0.5, 0.5, 0.0, 0.4])


def test_norm_to_xyxy_uses_image_dimensions() -> None:
    assert backend._norm_to_xyxy([0.5, 0.25, 0.2, 0.1], width=1000, height=500) == [
        400.0,
        100.0,
        600.0,
        150.0,
    ]


def _single_frame_session(*, editable: bool) -> backend.VideoDetectReviewSession:
    payload = {
        "frame_indices": np.asarray([0], dtype=np.int32),
        "bbox_norm_coords": np.full((1, 4), np.nan, dtype=np.float64),
        "confidence_scores": np.asarray([np.nan], dtype=np.float32),
        "class_ids": np.asarray([-1], dtype=np.int32),
        "status_labels": np.asarray(["missing"], dtype=object),
        "source_kind_labels": np.asarray(["none"], dtype=object),
        "manual_edit_flags": np.asarray([False], dtype=bool),
        "reason_labels": np.asarray(["missing_detection"], dtype=object),
        "source_detect_row_index": np.asarray([-1], dtype=np.int32),
        "detection_source": np.asarray([0], dtype=np.int8),
        "frame_to_row": {0: 0},
    }
    record = backend.FrameRecord(
        parent_frame_index=0,
        source_frame_index=0,
        video_id="video",
        refined_family_path=None,
        refined_run_name="refined",
        refined_group_path="refined_detect_runs/refined",
    )
    session = backend.VideoDetectReviewSession(
        zarr_path="/tmp/fake.zarr",
        root=object(),  # type: ignore[arg-type]
        mode="traditional",
        editable=editable,
        videos={},
        frame_records=[record],
        manual_score=0.75,
        manual_class_id=2,
    )
    session.refined_cache[record.refined_group_path] = backend.RefinedPayloadCacheEntry(
        group=object(),  # type: ignore[arg-type]
        payload=payload,
        total_frames=1,
        family_path=None,
        run_name="refined",
    )
    return session


def test_apply_manual_edit_requires_editable() -> None:
    session = _single_frame_session(editable=False)
    with pytest.raises(RuntimeError, match="read-only"):
        backend.apply_manual_edit(session, parent_frame_index=0, bbox_norm=[0.5, 0.5, 0.2, 0.2])


def test_apply_manual_edit_writes_manual_correction(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _single_frame_session(editable=True)
    captured: dict[str, object] = {}

    def fake_write_payload(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        captured["payload"] = args[2]
        captured["added"] = kwargs["added"]
        captured["removed"] = kwargs["removed"]

    monkeypatch.setattr(backend, "_write_payload", fake_write_payload)
    monkeypatch.setattr(backend, "_reload_refined_payload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        backend,
        "load_frame_payload",
        lambda *_args, **_kwargs: {
            "bbox_norm": [0.5, 0.5, 0.2, 0.2],
            "status": {"status_label": "present", "reason_label": "manual_correction"},
        },
    )

    result = backend.apply_manual_edit(session, parent_frame_index=0, bbox_norm=[0.5, 0.5, 0.2, 0.2])
    payload = captured["payload"]
    assert captured["added"] == 1
    assert captured["removed"] == 0
    np.testing.assert_allclose(np.asarray(payload["bbox_norm_coords"])[0], [0.5, 0.5, 0.2, 0.2])
    assert np.asarray(payload["status_labels"], dtype=object)[0] == "present"
    assert np.asarray(payload["source_kind_labels"], dtype=object)[0] == "manual"
    assert np.asarray(payload["manual_edit_flags"], dtype=bool)[0]
    assert result["action"] == "manual_correction"
