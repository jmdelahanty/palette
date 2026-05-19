from __future__ import annotations

import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr
from pathlib import Path

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


def test_load_frame_payload_reports_media_space_bbox_for_proxy_video() -> None:
    record = backend.FrameRecord(
        parent_frame_index=0,
        source_frame_index=0,
        video_id="proxy",
        refined_family_path="clips/clip_000000/cameras/2010093/refined_detect_runs",
        refined_run_name="refined_a",
        refined_group_path="clips/clip_000000/cameras/2010093/refined_detect_runs/refined_a",
        clip_id="clip_000000",
        camera_serial="2010093",
        recording_frame_id=1,
    )
    session = backend.VideoDetectReviewSession(
        zarr_path="/tmp/fake.zarr",
        root=object(),  # type: ignore[arg-type]
        mode="clipped",
        editable=False,
        videos={
            "proxy": backend.VideoSource(
                video_id="proxy",
                path=Path("/tmp/proxy.mp4"),
                fps=30.0,
                width=4512,
                height=4512,
                media_width=1024,
                media_height=1024,
                media_kind="review_proxy_video",
            )
        },
        frame_records=[record],
    )
    session.refined_cache[record.refined_group_path] = backend.RefinedPayloadCacheEntry(
        group=object(),  # type: ignore[arg-type]
        payload={
            "frame_indices": np.asarray([0], dtype=np.int32),
            "bbox_norm_coords": np.asarray([[0.25, 0.5, 0.1, 0.2]], dtype=np.float64),
            "confidence_scores": np.asarray([0.8], dtype=np.float32),
            "class_ids": np.asarray([0], dtype=np.int32),
            "status_labels": np.asarray(["present"], dtype=object),
            "source_kind_labels": np.asarray(["raw_detect"], dtype=object),
            "manual_edit_flags": np.asarray([False], dtype=bool),
            "reason_labels": np.asarray(["clean"], dtype=object),
            "source_detect_row_index": np.asarray([0], dtype=np.int32),
            "frame_to_row": {0: 0},
        },
        total_frames=1,
        family_path=record.refined_family_path,
        run_name=record.refined_run_name,
    )

    payload = backend.load_frame_payload(session, 0)

    assert payload["source_width"] == 4512
    assert payload["media_width"] == 1024
    np.testing.assert_allclose(payload["bbox_img_xyxy"], [902.4, 1804.8, 1353.6, 2707.2])
    np.testing.assert_allclose(payload["bbox_media_xyxy"], [204.8, 409.6, 307.2, 614.4])


def test_clipped_session_uses_review_proxy_manifest(tmp_path: Path) -> None:
    recording_dir = tmp_path / "recording"
    source_video = recording_dir / "clips" / "clip_000000" / "Cam2010093.mp4"
    source_video.parent.mkdir(parents=True)
    source_video.write_bytes(b"source")
    proxy_video = recording_dir / "derived" / "review_proxy" / "video_detect" / "proxy_a" / "clips" / "clip_000000" / "Cam2010093_1024x1024_h264.mp4"
    proxy_video.parent.mkdir(parents=True)
    proxy_video.write_bytes(b"proxy")
    proxy_manifest = proxy_video.parents[2] / "manifest.json"
    proxy_manifest.write_text(
        json.dumps(
            {
                "schema_version": "palette.review_proxy.video.v1",
                "clips": [
                    {
                        "clip_id": "clip_000000",
                        "camera_serial": "2010093",
                        "source_video_path": str(source_video),
                        "proxy_video_path": str(proxy_video),
                        "source_width": 4512,
                        "source_height": 4512,
                        "proxy_width": 1024,
                        "proxy_height": 1024,
                        "fps": 30,
                        "frame_count": 10,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    frame_index = recording_dir / "recording_frame_index.parquet"
    frame_index.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "parent_frame_index": [0],
                "camera_serial": ["2010093"],
                "clip_id": ["clip_000000"],
                "clip_local_frame_index": [0],
                "recording_frame_id": [1],
                "video_path": [str(source_video)],
            }
        ),
        frame_index,
    )
    zarr_path = recording_dir / "zarr" / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs.update({"width": 4512, "height": 4512, "fps": 30})
    collection = root.create_group("experiment_index").create_group("finalized_runs").create_group("collection_a")
    collection.attrs["selected_runs"] = [
        {
            "clip_id": "clip_000000",
            "camera_serial": "2010093",
            "refined_group_path": "clips/clip_000000/cameras/2010093/refined_detect_runs/refined_a",
            "refined_detect_run": "refined_a",
            "source": {"video_path": str(source_video)},
        }
    ]
    root.create_group("clips").create_group("clip_000000").create_group("cameras").create_group("2010093").create_group("refined_detect_runs").create_group("refined_a")

    session = backend._clipped_session(  # noqa: SLF001
        zarr_path,
        root,
        collection_id="collection_a",
        recording_frame_index=frame_index,
        review_proxy_manifest=proxy_manifest,
        editable=False,
        manual_score=1.0,
        manual_class_id=0,
    )

    assert session.review_proxy_manifest == str(proxy_manifest.resolve())
    video = next(iter(session.videos.values()))
    assert video.path == proxy_video.resolve()
    assert video.source_path == source_video.resolve()
    assert video.media_kind == "review_proxy_video"
    assert video.width == 4512
    assert video.height == 4512
    assert video.media_width == 1024
    assert video.media_height == 1024
    assert video.frame_count == 10
    sources = backend.video_sources_payload(session)
    assert sources[0]["media_kind"] == "review_proxy_video"
    assert sources[0]["path"] == str(proxy_video.resolve())
    assert sources[0]["source_path"] == str(source_video.resolve())


def test_clipped_session_requires_matching_proxy_entry(tmp_path: Path) -> None:
    source_video = tmp_path / "recording" / "clips" / "clip_000000" / "Cam2010093.mp4"
    source_video.parent.mkdir(parents=True)
    source_video.write_bytes(b"source")
    proxy_manifest = tmp_path / "proxy" / "manifest.json"
    proxy_manifest.parent.mkdir()
    proxy_manifest.write_text(
        json.dumps(
            {
                "schema_version": "palette.review_proxy.video.v1",
                "clips": [
                    {
                        "clip_id": "clip_000001",
                        "camera_serial": "2010093",
                        "proxy_video_path": str(tmp_path / "proxy.mp4"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "proxy.mp4").write_bytes(b"proxy")
    frame_index = tmp_path / "recording" / "recording_frame_index.parquet"
    pq.write_table(
        pa.table(
            {
                "parent_frame_index": [0],
                "camera_serial": ["2010093"],
                "clip_id": ["clip_000000"],
                "clip_local_frame_index": [0],
                "recording_frame_id": [1],
                "video_path": [str(source_video)],
            }
        ),
        frame_index,
    )
    zarr_path = tmp_path / "recording" / "zarr" / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs.update({"width": 4512, "height": 4512, "fps": 30})
    collection = root.create_group("experiment_index").create_group("finalized_runs").create_group("collection_a")
    collection.attrs["selected_runs"] = [
        {
            "clip_id": "clip_000000",
            "camera_serial": "2010093",
            "refined_group_path": "clips/clip_000000/cameras/2010093/refined_detect_runs/refined_a",
            "source": {"video_path": str(source_video)},
        }
    ]
    root.create_group("clips").create_group("clip_000000").create_group("cameras").create_group("2010093").create_group("refined_detect_runs").create_group("refined_a")

    with pytest.raises(RuntimeError, match="no entry"):
        backend._clipped_session(  # noqa: SLF001
            zarr_path,
            root,
            collection_id="collection_a",
            recording_frame_index=frame_index,
            review_proxy_manifest=proxy_manifest,
            editable=False,
            manual_score=1.0,
            manual_class_id=0,
        )
