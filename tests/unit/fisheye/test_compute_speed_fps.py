from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from fisheye.analysis.compute_speed import find_fps
from fisheye.shared.source_video_metadata import SourceVideoMetadataConflictError


class _Group:
    def __init__(
        self,
        *,
        attrs: dict[str, object] | None = None,
        children: dict[str, object] | None = None,
    ) -> None:
        self.attrs = dict(attrs or {})
        self._children = dict(children or {})

    def __contains__(self, key: object) -> bool:
        return key in self._children

    def __getitem__(self, key: str) -> object:
        return self._children[key]


def _console() -> Console:
    return Console(file=StringIO(), force_terminal=False, color_system=None)


def _clipped_source_metadata(*, fps: float = 30.0) -> dict[str, object]:
    return {
        "schema_id": "palette.source_video_collection_metadata.v1",
        "layout": "clipped_video_collection",
        "fps": fps,
        "collection": {"members": [{"fps": fps}]},
    }


def test_find_fps_prefers_canonical_clipped_source_metadata() -> None:
    root = _Group(attrs={"source_video_metadata": _clipped_source_metadata()})

    assert find_fps(root, _console()) == 30.0


def test_find_fps_fails_closed_on_canonical_legacy_conflict() -> None:
    root = _Group(
        attrs={
            "fps": 60.0,
            "source_video_metadata": _clipped_source_metadata(),
        }
    )

    with pytest.raises(SourceVideoMetadataConflictError):
        find_fps(root, _console())


def test_find_fps_retains_raw_video_compatibility_source() -> None:
    root = _Group(children={"raw_video": SimpleNamespace(attrs={"fps": 24.0})})

    assert find_fps(root, _console()) == 24.0


def test_find_fps_uses_explicit_fallback_only_when_metadata_is_absent() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)

    assert find_fps(_Group(), console, fallback=48.0) == 48.0
    assert "defaulting to 48.0" in output.getvalue()
