from __future__ import annotations

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pytest


_PLAYGROUND = (
    Path(__file__).resolve().parents[3] / "playgrounds" / "heartrate_stabilization"
)
sys.path.insert(0, str(_PLAYGROUND))

import select_fixed_video_roi_web as selector  # noqa: E402
import select_fixed_video_chambers_web as chamber_selector  # noqa: E402


def _state(tmp_path: Path) -> selector.SelectorState:
    video = tmp_path / "top_camera.mp4"
    video.write_bytes(b"video provenance fixture")
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    frame[:, :, 1] = 80

    def read_frame(path: Path, frame_index: int) -> np.ndarray:
        assert path == video
        assert 0 <= frame_index < 20
        return frame.copy()

    return selector.SelectorState(
        video=video,
        output_json=tmp_path / "top_camera_roi.json",
        preview_output=tmp_path / "top_camera_roi_preview.png",
        metadata=selector.VideoMetadata(
            width=16,
            height=12,
            frame_count=20,
            fps=200.0,
        ),
        initial_frame_index=0,
        initial_roi=None,
        frame_reader=read_frame,
    )


def test_validate_roi_uses_whole_source_pixels_and_frame_bounds() -> None:
    assert selector._validate_roi([2, 3, 8, 5], width=16, height=12) == (
        2,
        3,
        8,
        5,
    )
    with pytest.raises(ValueError, match="whole source pixels"):
        selector._validate_roi([2.5, 3, 8, 5], width=16, height=12)
    with pytest.raises(ValueError, match="exceeds frame bounds"):
        selector._validate_roi([10, 3, 8, 5], width=16, height=12)


def test_save_writes_provenance_json_and_annotated_preview(tmp_path: Path) -> None:
    state = _state(tmp_path)

    payload = state.save({"frame_index": 7, "roi_xywh": [2, 3, 8, 5]})

    persisted = json.loads(state.output_json.read_text())
    assert persisted == payload
    assert persisted["schema_id"] == "palette.playground.fixed_video_roi.v1"
    assert persisted["coordinate_space"] == "source_video_frame_pixels"
    assert persisted["source_video"] == str(state.video.resolve())
    assert persisted["frame_index"] == 7
    assert persisted["roi_xywh"] == [2, 3, 8, 5]
    preview = cv2.imread(str(state.preview_output), cv2.IMREAD_COLOR)
    assert preview.shape == (12, 16, 3)
    assert int(preview[3, 2, 2]) > int(preview[3, 2, 1])


def test_existing_selection_reloads_only_for_matching_video(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.save({"frame_index": 7, "roi_xywh": [2, 3, 8, 5]})

    loaded = selector._load_existing_roi(
        state.output_json,
        video=state.video,
        metadata=state.metadata,
    )
    assert loaded == (7, (2, 3, 8, 5))

    other_video = tmp_path / "other.mp4"
    other_video.write_bytes(b"other")
    with pytest.raises(ValueError, match="different video"):
        selector._load_existing_roi(
            state.output_json,
            video=other_video,
            metadata=state.metadata,
        )


def test_http_state_frame_and_save_endpoints(tmp_path: Path) -> None:
    state = _state(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), selector._handler_class(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(f"{base_url}/api/state", timeout=3) as response:
            status = json.load(response)
        assert status["width"] == 16
        assert status["frame_count"] == 20

        with urlopen(f"{base_url}/api/frame?index=4", timeout=3) as response:
            frame_png = response.read()
        assert frame_png.startswith(b"\x89PNG\r\n\x1a\n")

        request = Request(
            f"{base_url}/api/save",
            data=json.dumps(
                {"frame_index": 4, "roi_xywh": [1, 2, 10, 7]}
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=3) as response:
            saved = json.load(response)
        assert saved["saved"] is True
        assert json.loads(state.output_json.read_text())["roi_xywh"] == [1, 2, 10, 7]

        with pytest.raises(HTTPError) as error:
            urlopen(f"{base_url}/api/frame?index=20", timeout=3)
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def _chamber_state(tmp_path: Path) -> chamber_selector.ChamberSelectorState:
    video = tmp_path / "side_camera.avi"
    video.write_bytes(b"side camera provenance fixture")
    frame = np.full((20, 24, 3), 70, dtype=np.uint8)

    def read_frame(path: Path, frame_index: int) -> np.ndarray:
        assert path == video
        assert 0 <= frame_index < 30
        return frame.copy()

    return chamber_selector.ChamberSelectorState(
        video=video,
        output_json=tmp_path / "side_camera_chambers.json",
        preview_output=tmp_path / "side_camera_chambers_preview.png",
        metadata=selector.VideoMetadata(width=24, height=20, frame_count=30, fps=100.0),
        initial_frame_index=0,
        initial_chambers=None,
        frame_reader=read_frame,
    )


def test_validate_chamber_polygon_requires_bounded_nonzero_area() -> None:
    assert chamber_selector._validate_polygon(
        [[2, 3], [8, 3], [6, 9]],
        width=24,
        height=20,
    ) == ((2, 3), (8, 3), (6, 9))
    with pytest.raises(ValueError, match="whole source pixels"):
        chamber_selector._validate_polygon(
            [[2.5, 3], [8, 3], [6, 9]], width=24, height=20
        )
    with pytest.raises(ValueError, match="area"):
        chamber_selector._validate_polygon(
            [[2, 3], [4, 3], [6, 3]], width=24, height=20
        )


def test_chamber_save_writes_two_polygons_and_preview(tmp_path: Path) -> None:
    state = _chamber_state(tmp_path)
    request = {
        "frame_index": 12,
        "chambers": {
            "chamber_a": [[2, 3], [8, 3], [6, 9]],
            "chamber_b": [[12, 5], [20, 5], [18, 14], [13, 12]],
        },
    }

    payload = state.save(request)

    persisted = json.loads(state.output_json.read_text())
    assert persisted == payload
    assert persisted["schema_id"] == "palette.playground.fixed_video_chambers.v1"
    assert persisted["anatomical_identity_status"] == "unassigned_chamber_a_chamber_b"
    assert persisted["chambers"]["chamber_a"]["polygon_xy"] == request["chambers"]["chamber_a"]
    assert persisted["chambers"]["chamber_b"]["polygon_xy"] == request["chambers"]["chamber_b"]
    preview = cv2.imread(str(state.preview_output), cv2.IMREAD_COLOR)
    assert preview.shape == (20, 24, 3)
    assert not np.all(preview == 70)
