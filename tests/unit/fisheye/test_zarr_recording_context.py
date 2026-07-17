from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.shared.source_video_metadata import SourceVideoMetadataConflictError
from fisheye.shared.zarr_recording_context import infer_recording_context


def _write_v3_attrs(group_path: Path, attrs: dict[str, object]) -> None:
    group_path.mkdir(parents=True, exist_ok=True)
    (group_path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": attrs,
            }
        ),
        encoding="utf-8",
    )


def test_infer_recording_context_uses_v2_relative_locator(tmp_path: Path) -> None:
    recording = tmp_path / "recording"
    zarr_path = recording / "zarr" / "analysis.zarr"
    video = recording / "cams" / "source.mp4"
    _write_v3_attrs(
        zarr_path,
        {
            "recording_id": "recording-1",
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
    )
    _write_v3_attrs(zarr_path / "raw_video", {"source_path": str(video)})

    context = infer_recording_context(zarr_path)

    assert context.recording_dir == recording.resolve()
    assert context.recording_id == "recording-1"
    assert context.source_video_path == video.resolve()


def test_infer_recording_context_can_derive_recording_root_from_zarr_location(
    tmp_path: Path,
) -> None:
    recording = tmp_path / "recording"
    zarr_path = recording / "zarr" / "analysis.zarr"
    video = recording / "cams" / "source.mp4"
    _write_v3_attrs(
        zarr_path,
        {
            "source_video_path": str(video),
            "source_video_metadata": {
                "schema_id": "palette.source_video_metadata.v2",
                "layout": "single_video",
                "locator": {
                    "kind": "recording_relative",
                    "relative_path": "cams/source.mp4",
                },
            },
        },
    )

    context = infer_recording_context(zarr_path)

    assert context.recording_dir == recording.resolve()
    assert context.source_video_path == video.resolve()


def test_infer_recording_context_fails_closed_on_video_path_conflict(
    tmp_path: Path,
) -> None:
    recording = tmp_path / "recording"
    zarr_path = recording / "zarr" / "analysis.zarr"
    _write_v3_attrs(
        zarr_path,
        {
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
        },
    )

    with pytest.raises(SourceVideoMetadataConflictError):
        infer_recording_context(zarr_path)


def test_infer_recording_context_preserves_legacy_path_fallback(tmp_path: Path) -> None:
    recording = tmp_path / "recording"
    zarr_path = recording / "zarr" / "analysis.zarr"
    video = recording / "cams" / "source.mp4"
    _write_v3_attrs(zarr_path, {"source_video_path": str(video)})

    context = infer_recording_context(zarr_path)

    assert context.recording_dir == recording.resolve()
    assert context.source_video_path == video.resolve()
