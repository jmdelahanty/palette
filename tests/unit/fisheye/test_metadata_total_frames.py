from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fisheye.shared.metadata import get_total_frames, get_video_source_path
from fisheye.shared.source_video_metadata import SourceVideoMetadataConflictError


class _Array:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _Group:
    def __init__(self, *, attrs: dict[str, Any] | None = None, children: dict[str, Any] | None = None) -> None:
        self.attrs = attrs or {}
        self._children = children or {}

    def __contains__(self, key: str) -> bool:
        return key in self._children

    def __getitem__(self, key: str) -> Any:
        return self._children[key]

    def get(self, key: str) -> Any:
        return self._children.get(key)


def test_get_total_frames_prefers_detect_frame_counts_before_root_total() -> None:
    root = _Group(attrs={"total_frames": 1_188_000})
    detect_group = _Group(children={"frame_counts": _Array((54_000,))})

    assert get_total_frames(root, detect_group) == 54_000


def test_get_total_frames_keeps_detect_total_frames_highest_priority() -> None:
    root = _Group(attrs={"total_frames": 1_188_000})
    detect_group = _Group(
        attrs={"total_frames": 12},
        children={"frame_counts": _Array((54_000,))},
    )

    assert get_total_frames(root, detect_group) == 12


def test_get_total_frames_uses_sampled_raw_video_before_root_total() -> None:
    raw_video = _Group(
        attrs={"import_mode": "sampled"},
        children={"images_ds": np.zeros((7, 2, 2), dtype=np.uint8)},
    )
    root = _Group(attrs={"total_frames": 1_188_000}, children={"raw_video": raw_video})

    assert get_total_frames(root) == 7


def test_get_video_source_path_resolves_v2_recording_relative_locator(tmp_path) -> None:
    recording = tmp_path / "recording"
    video = recording / "cams" / "source.mp4"
    root = _Group(
        attrs={
            "recording_path": str(recording),
            "source_video_path": str(video),
            "source_path": str(video),
            "source_video_metadata": {
                "schema_id": "palette.source_video_metadata.v2",
                "layout": "single_video",
                "locator": {
                    "kind": "recording_relative",
                    "relative_path": "cams/source.mp4",
                },
                "source_path": str(video),
            },
        },
        children={"raw_video": _Group(attrs={"source_path": str(video)})},
    )

    assert get_video_source_path(root) == str(video.resolve())


def test_get_video_source_path_keeps_missing_legacy_metadata_optional() -> None:
    assert get_video_source_path(_Group()) is None


def test_get_video_source_path_fails_closed_on_v2_mirror_conflict(tmp_path) -> None:
    recording = tmp_path / "recording"
    root = _Group(
        attrs={
            "recording_path": str(recording),
            "source_video_path": str(tmp_path / "stale.mp4"),
            "source_video_metadata": {
                "schema_id": "palette.source_video_metadata.v2",
                "layout": "single_video",
                "locator": {
                    "kind": "recording_relative",
                    "relative_path": "cams/source.mp4",
                },
            },
        }
    )

    with pytest.raises(SourceVideoMetadataConflictError):
        get_video_source_path(root)
