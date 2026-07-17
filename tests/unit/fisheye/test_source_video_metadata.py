from __future__ import annotations

from pathlib import Path

import pytest
import zarr

from fisheye.shared.source_video_metadata import (
    SOURCE_VIDEO_LAYOUT_SINGLE,
    SOURCE_VIDEO_LOCATOR_ABSOLUTE,
    SOURCE_VIDEO_LOCATOR_RECORDING_RELATIVE,
    SOURCE_VIDEO_METADATA_SCHEMA_ID,
    SourceVideoMetadataConflictError,
    SourceVideoMetadataError,
    SourceVideoMetadataMissingError,
    build_source_video_metadata_v2,
    resolve_source_video,
    resolve_source_video_from_attrs,
)


class _Group:
    def __init__(self, *, attrs: dict[str, object] | None = None) -> None:
        self.attrs = dict(attrs or {})
        self.children: dict[str, _Group] = {}

    def get(self, name: str):
        return self.children.get(name)


def _versioned_metadata(relative_path: str, absolute_path: Path) -> dict[str, object]:
    return {
        "schema_id": SOURCE_VIDEO_METADATA_SCHEMA_ID,
        "layout": SOURCE_VIDEO_LAYOUT_SINGLE,
        "locator": {
            "kind": SOURCE_VIDEO_LOCATOR_RECORDING_RELATIVE,
            "relative_path": relative_path,
        },
        "source_path": str(absolute_path),
        "width": 4512,
        "height": 4512,
        "fps": 100.0,
    }


def _single_video_root(recording: Path, video: Path) -> _Group:
    metadata = _versioned_metadata("cams/source.mp4", video)
    root = _Group(
        attrs={
            "recording_path": str(recording),
            "source_video_path": str(video),
            "source_path": str(video),
            "source_video_metadata": metadata,
        }
    )
    root.children["raw_video"] = _Group(attrs={"source_path": str(video)})
    return root


def test_resolve_versioned_recording_relative_source(tmp_path: Path) -> None:
    recording = tmp_path / "recording"
    video = recording / "cams" / "source.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    root = _single_video_root(recording, video)

    resolved = resolve_source_video(root, require_exists=True)

    assert resolved.path == video.resolve()
    assert resolved.schema_id == SOURCE_VIDEO_METADATA_SCHEMA_ID
    assert resolved.locator_kind == SOURCE_VIDEO_LOCATOR_RECORDING_RELATIVE
    assert resolved.source == "source_video_metadata.locator"
    assert set(resolved.compatibility_sources) == {
        "root.source_video_path",
        "root.source_path",
        "source_video_metadata.source_path",
        "raw_video.source_path",
    }


def test_resolve_versioned_source_from_attribute_mappings(tmp_path: Path) -> None:
    recording = tmp_path / "recording"
    video = recording / "cams" / "source.mp4"
    root = _single_video_root(recording, video)

    resolved = resolve_source_video_from_attrs(
        root.attrs,
        raw_video_attrs=root.children["raw_video"].attrs,
    )

    assert resolved.path == video.resolve()
    assert resolved.source == "source_video_metadata.locator"


def test_resolve_versioned_source_from_real_zarr_group(tmp_path: Path) -> None:
    recording = tmp_path / "recording"
    video = recording / "cams" / "source.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    root = zarr.open_group(str(recording / "zarr" / "analysis.zarr"), mode="w")
    root.attrs.update(_single_video_root(recording, video).attrs)
    raw = root.require_group("raw_video")
    raw.attrs["source_path"] = str(video)

    resolved = resolve_source_video(
        root,
        zarr_path=recording / "zarr" / "analysis.zarr",
        require_exists=True,
    )

    assert resolved.path == video.resolve()
    assert resolved.schema_id == SOURCE_VIDEO_METADATA_SCHEMA_ID


def test_recording_relative_locator_survives_recording_relocation(tmp_path: Path) -> None:
    recording = tmp_path / "relocated" / "recording"
    video = recording / "cams" / "source.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    root = _single_video_root(recording, video)

    resolved = resolve_source_video(root, require_exists=True)

    assert resolved.path == video.resolve()
    metadata = root.attrs["source_video_metadata"]
    assert isinstance(metadata, dict)
    assert metadata["locator"] == {
        "kind": SOURCE_VIDEO_LOCATOR_RECORDING_RELATIVE,
        "relative_path": "cams/source.mp4",
    }


def test_versioned_resolver_fails_closed_on_compatibility_conflict(tmp_path: Path) -> None:
    recording = tmp_path / "recording"
    video = recording / "cams" / "source.mp4"
    stale = tmp_path / "stale" / "source.mp4"
    root = _single_video_root(recording, video)
    root.attrs["source_video_path"] = str(stale)

    with pytest.raises(SourceVideoMetadataConflictError, match="root.source_video_path"):
        resolve_source_video(root)


def test_legacy_resolver_accepts_consistent_mirrors(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    root = _Group(
        attrs={
            "source_video_path": str(video),
            "source_path": str(video),
            "source_video_metadata": {"source_path": str(video)},
        }
    )

    resolved = resolve_source_video(root)

    assert resolved.path == video.resolve()
    assert resolved.schema_id is None
    assert resolved.locator_kind == "legacy"
    assert resolved.source == "root.source_video_path"


def test_legacy_resolver_preserves_video_source_path_fallback(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"

    resolved = resolve_source_video(_Group(attrs={"video_source_path": str(video)}))

    assert resolved.path == video.resolve()
    assert resolved.source == "root.video_source_path"


def test_resolver_distinguishes_missing_locator() -> None:
    with pytest.raises(SourceVideoMetadataMissingError, match="No source-video locator"):
        resolve_source_video_from_attrs({})


def test_resolver_rejects_collection_layout() -> None:
    root = _Group(
        attrs={
            "source_video_metadata": {
                "schema_id": SOURCE_VIDEO_METADATA_SCHEMA_ID,
                "layout": "clipped_collection",
                "locator": {},
            }
        }
    )

    with pytest.raises(SourceVideoMetadataError, match="collection/frame-index resolver"):
        resolve_source_video(root)


def test_resolver_rejects_unknown_versioned_schema() -> None:
    root = _Group(
        attrs={
            "source_video_metadata": {
                "schema_id": "palette.source_video_metadata.v999",
                "source_path": "/tmp/source.mp4",
            }
        }
    )

    with pytest.raises(SourceVideoMetadataError, match="Unsupported source-video metadata schema"):
        resolve_source_video(root)


def test_resolver_rejects_recording_relative_traversal(tmp_path: Path) -> None:
    recording = tmp_path / "recording"
    root = _Group(
        attrs={
            "recording_path": str(recording),
            "source_video_metadata": _versioned_metadata(
                "../outside.mp4",
                tmp_path / "outside.mp4",
            ),
        }
    )

    with pytest.raises(SourceVideoMetadataError, match="Invalid recording-relative"):
        resolve_source_video(root)


def test_build_metadata_uses_recording_relative_locator(tmp_path: Path) -> None:
    recording = tmp_path / "recording"
    video = recording / "cams" / "source.mp4"
    payload = build_source_video_metadata_v2(
        {
            "source_path": str(video),
            "source_video": video.name,
            "width": 4512,
            "height": 4512,
        },
        recording_path=recording,
        fingerprint_attrs={
            "source_video_fingerprint": "abc123",
            "source_video_fingerprint_strategy": "stat_v1",
            "source_video_size_bytes": 123,
            "source_video_mtime_ns": 456,
        },
    )

    assert payload["schema_id"] == SOURCE_VIDEO_METADATA_SCHEMA_ID
    assert payload["locator"] == {
        "kind": SOURCE_VIDEO_LOCATOR_RECORDING_RELATIVE,
        "relative_path": "cams/source.mp4",
    }
    assert payload["source_path"] == str(video.resolve())
    assert payload["file_fingerprint"] == {
        "strategy": "stat_v1",
        "value": "abc123",
        "size_bytes": 123,
        "mtime_ns": 456,
        "relocation_stable": False,
    }


def test_build_metadata_uses_absolute_locator_for_external_video(tmp_path: Path) -> None:
    recording = tmp_path / "recording"
    video = tmp_path / "external" / "source.mp4"

    payload = build_source_video_metadata_v2(
        {"source_path": str(video)},
        recording_path=recording,
    )

    assert payload["locator"] == {
        "kind": SOURCE_VIDEO_LOCATOR_ABSOLUTE,
        "path": str(video.resolve()),
    }
