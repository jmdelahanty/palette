"""Stable, compact handles for editable recording-exploration sessions.

The Marimo recording explorer owns the reactive UI state.  This module exposes
that state through one intentionally small object so users and pairing agents
do not need to depend on the notebook's implementation-cell names.  The source
Zarr is always opened through the read-only Palette helper; deployment-level
write prevention is provided separately by the workspace launcher's mount
namespace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from fisheye.shared.zarr_io import open_zarr_root


@dataclass(frozen=True, repr=False)
class RecordingExplorationWorkspace:
    """Live read handles made available below the rendered recording app."""

    zarr_path: Path
    selected_recording: Any
    selected_provider: Any | None
    selected_spec: Any | None
    selected_analysis: Any | None
    core_source: Any | None
    core_projection: Any | None
    chaser_view: Any | None

    @property
    def provider_id(self) -> str | None:
        value = getattr(self.selected_provider, "provider_id", None)
        return str(value) if value is not None else None

    @property
    def analysis_id(self) -> str | None:
        value = getattr(self.selected_analysis, "analysis_id", None)
        return str(value) if value is not None else None

    @property
    def core_frame(self) -> Any | None:
        """Currently selected core projection as a Polars LazyFrame, if any."""

        return getattr(self.core_projection, "frame", None)

    @property
    def related_core_frames(self) -> Mapping[str, Any]:
        """Related projected Polars LazyFrames, such as bout-event tables."""

        frames = getattr(self.core_projection, "related_frames", None)
        return frames if isinstance(frames, Mapping) else {}

    @property
    def chaser_tables(self) -> Mapping[str, Any]:
        """Already-loaded chaser tables for the selected analysis.

        These retain their viewer-native Pandas or Polars representation.  No
        additional arrays are loaded merely by inspecting this mapping.
        """

        if self.chaser_view is None:
            return {}
        names = (
            "distance_df",
            "position_df",
            "windows_df",
            "chaser_position_df",
            "spatial_occupancy_df",
            "egocentric_bearing_df",
            "egocentric_alignment_df",
            "egocentric_heading_df",
            "epoch_summary_df",
        )
        return {
            name: getattr(self.chaser_view, name)
            for name in names
            if getattr(self.chaser_view, name, None) is not None
        }

    def open_zarr(self) -> Any:
        """Open the selected recording Zarr in read-only mode."""

        return open_zarr_root(self.zarr_path, mode="r")

    def summary(self) -> dict[str, Any]:
        """Return a small description safe to display in a notebook output."""

        return {
            "zarr_path": str(self.zarr_path),
            "recording_id": getattr(self.selected_recording, "recording_id", None),
            "provider_id": self.provider_id,
            "analysis_id": self.analysis_id,
            "core_frame_available": self.core_frame is not None,
            "related_core_frames": tuple(sorted(self.related_core_frames)),
            "chaser_tables": tuple(sorted(self.chaser_tables)),
        }

    def __repr__(self) -> str:
        summary = self.summary()
        return (
            "RecordingExplorationWorkspace("
            f"recording_id={summary['recording_id']!r}, "
            f"provider_id={summary['provider_id']!r}, "
            f"analysis_id={summary['analysis_id']!r}, "
            f"zarr_path={summary['zarr_path']!r})"
        )
