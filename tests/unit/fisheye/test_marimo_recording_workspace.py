from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import apps.marimo.components.recording_workspace as workspace_module
from apps.marimo.components.recording_workspace import RecordingExplorationWorkspace


def _workspace(tmp_path: Path) -> tuple[RecordingExplorationWorkspace, object, object]:
    core_frame = object()
    bouts_frame = object()
    core_projection = SimpleNamespace(
        frame=core_frame,
        related_frames={"swim_bouts": bouts_frame},
    )
    chaser_view = SimpleNamespace(
        distance_df=object(),
        position_df=object(),
        windows_df=object(),
        chaser_position_df=None,
        spatial_occupancy_df=None,
        egocentric_bearing_df=None,
        egocentric_alignment_df=None,
        egocentric_heading_df=None,
        epoch_summary_df=object(),
    )
    workspace = RecordingExplorationWorkspace(
        zarr_path=tmp_path / "recording_analysis.zarr",
        selected_recording=SimpleNamespace(recording_id="recording-17"),
        selected_provider=SimpleNamespace(provider_id="core_behavior"),
        selected_spec=SimpleNamespace(run_path="analysis/example"),
        selected_analysis=SimpleNamespace(analysis_id="speed"),
        core_source=object(),
        core_projection=core_projection,
        chaser_view=chaser_view,
    )
    return workspace, core_frame, bouts_frame


def test_recording_workspace_exposes_compact_live_handles(tmp_path: Path) -> None:
    workspace, core_frame, bouts_frame = _workspace(tmp_path)

    assert workspace.provider_id == "core_behavior"
    assert workspace.analysis_id == "speed"
    assert workspace.core_frame is core_frame
    assert workspace.related_core_frames == {"swim_bouts": bouts_frame}
    assert set(workspace.chaser_tables) == {
        "distance_df",
        "position_df",
        "windows_df",
        "epoch_summary_df",
    }
    assert workspace.summary() == {
        "zarr_path": str(tmp_path / "recording_analysis.zarr"),
        "recording_id": "recording-17",
        "provider_id": "core_behavior",
        "analysis_id": "speed",
        "core_frame_available": True,
        "related_core_frames": ("swim_bouts",),
        "chaser_tables": (
            "distance_df",
            "epoch_summary_df",
            "position_df",
            "windows_df",
        ),
        "persisted_pngs": (),
    }
    assert "recording-17" in repr(workspace)
    assert "object at" not in repr(workspace)


def test_recording_workspace_opens_zarr_read_only(monkeypatch, tmp_path: Path) -> None:
    workspace, _, _ = _workspace(tmp_path)
    sentinel = object()
    calls: list[tuple[Path, str]] = []

    def _open(path: Path, *, mode: str):
        calls.append((path, mode))
        return sentinel

    monkeypatch.setattr(workspace_module, "open_zarr_root", _open)

    assert workspace.open_zarr() is sentinel
    assert calls == [(tmp_path / "recording_analysis.zarr", "r")]


def test_recording_workspace_exposes_gaze_tables_and_persisted_png(tmp_path: Path) -> None:
    payload = b"\x89PNG\r\n\x1a\nsummary"
    gaze_view = SimpleNamespace(
        recording_summary_df=object(),
        object_vs_virtual_df=object(),
        summary_png_path="analysis/chaser/gaze/visualizations/summary_png",
        summary_png_bytes=payload,
    )
    workspace = RecordingExplorationWorkspace(
        zarr_path=tmp_path / "recording_analysis.zarr",
        selected_recording=SimpleNamespace(recording_id="recording-17"),
        selected_provider=SimpleNamespace(provider_id="stimulus_chaser"),
        selected_spec=SimpleNamespace(run_path="analysis/chaser"),
        selected_analysis=SimpleNamespace(analysis_id="gaze_tracking"),
        core_source=None,
        core_projection=None,
        chaser_view=gaze_view,
    )

    assert set(workspace.chaser_tables) == {
        "recording_summary_df",
        "object_vs_virtual_df",
    }
    assert workspace.persisted_pngs == {
        "chaser_gaze_tracking_summary_png": {
            "path": "analysis/chaser/gaze/visualizations/summary_png",
            "media_type": "image/png",
            "bytes": payload,
        }
    }
