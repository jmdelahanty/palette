from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fisheye.shared.metadata import get_fps, get_total_frames, get_video_source_path
from fisheye.shared.source_video_metadata import (
    SourceVideoMetadataConflictError,
    SourceVideoMetadataError,
)


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


@pytest.mark.parametrize(
    ("schema_id", "layout"),
    (
        ("palette.source_video_metadata.v2", "single_video"),
        (
            "palette.source_video_collection_metadata.v1",
            "clipped_video_collection",
        ),
    ),
)
def test_get_fps_reads_supported_canonical_source_metadata(
    schema_id: str,
    layout: str,
) -> None:
    metadata = {
        "schema_id": schema_id,
        "layout": layout,
        "fps": 30.0,
    }
    if layout == "clipped_video_collection":
        metadata["collection"] = {"members": [{"fps": 30.0}, {"fps": 30.0}]}
    root = _Group(
        attrs={
            "source_video_metadata": metadata,
        }
    )

    assert get_fps(root) == 30.0


def test_get_fps_accepts_matching_legacy_mirror() -> None:
    root = _Group(
        attrs={
            "fps": 30,
            "source_video_metadata": {
                "schema_id": "palette.source_video_collection_metadata.v1",
                "layout": "clipped_video_collection",
                "fps": 30.0,
                "collection": {"members": [{"fps": 30.0}]},
            },
        }
    )

    assert get_fps(root) == 30.0


def test_get_fps_fails_closed_on_canonical_legacy_conflict() -> None:
    root = _Group(
        attrs={
            "fps": 29.0,
            "source_video_metadata": {
                "schema_id": "palette.source_video_collection_metadata.v1",
                "layout": "clipped_video_collection",
                "fps": 30.0,
                "collection": {"members": [{"fps": 30.0}]},
            },
        }
    )

    with pytest.raises(SourceVideoMetadataConflictError, match="root.fps differs"):
        get_fps(root)


@pytest.mark.parametrize("fps", (None, True, 0, -1, float("nan"), float("inf"), "30"))
def test_get_fps_rejects_invalid_canonical_values(fps: object) -> None:
    root = _Group(
        attrs={
            "source_video_metadata": {
                "schema_id": "palette.source_video_collection_metadata.v1",
                "layout": "clipped_video_collection",
                "fps": fps,
                "collection": {"members": [{"fps": fps}]},
            }
        }
    )

    with pytest.raises(SourceVideoMetadataError, match="positive finite number"):
        get_fps(root)


def test_get_fps_rejects_unsupported_versioned_metadata() -> None:
    root = _Group(
        attrs={
            "fps": 30.0,
            "source_video_metadata": {
                "schema_id": "palette.source_video_metadata.v999",
                "layout": "single_video",
                "fps": 30.0,
            },
        }
    )

    with pytest.raises(SourceVideoMetadataError, match="Unsupported"):
        get_fps(root)


def test_get_fps_fails_closed_on_clipped_member_conflict() -> None:
    root = _Group(
        attrs={
            "source_video_metadata": {
                "schema_id": "palette.source_video_collection_metadata.v1",
                "layout": "clipped_video_collection",
                "fps": 30.0,
                "collection": {
                    "members": [{"fps": 30.0}, {"fps": 60.0}],
                },
            }
        }
    )

    with pytest.raises(SourceVideoMetadataConflictError, match="member FPS differs"):
        get_fps(root)


def test_get_fps_preserves_legacy_root_fallback_and_missing_value() -> None:
    assert get_fps(_Group(attrs={"fps": 24.0})) == 24.0
    assert get_fps(_Group()) is None
