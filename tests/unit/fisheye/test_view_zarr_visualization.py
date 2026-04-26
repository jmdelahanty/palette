from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.utils import view_zarr_visualization as mod


PNG_BYTES = b"\x89PNG\r\n\x1a\nFAKEPNG"


def _make_archive(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    run = root.create_group("analysis").create_group("example_runs").create_group("example_1")
    vis = run.create_group("visualizations")
    png = vis.create_array(
        "example_summary_png",
        data=np.frombuffer(PNG_BYTES, dtype=np.uint8),
        chunks=(len(PNG_BYTES),),
    )
    png.attrs["media_type"] = "image/png"
    png.attrs["artifact_role"] = "snapshot"
    png.attrs["description"] = "Example summary"

    interactive = vis.create_group("example_summary_interactive")
    interactive.attrs["artifact_role"] = "interactive_spec"
    interactive.attrs["media_type"] = "application/vnd.palette.plot-spec+json"
    interactive.attrs["snapshot_artifact"] = "example_summary_png"
    return zarr_path


def test_load_png_artifact_bytes_from_direct_path(tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="r")

    resolved, payload = mod.load_png_artifact_bytes(
        root,
        "analysis/example_runs/example_1/visualizations/example_summary_png",
    )

    assert resolved == "analysis/example_runs/example_1/visualizations/example_summary_png"
    assert payload == PNG_BYTES


def test_load_png_artifact_bytes_resolves_interactive_snapshot(tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="r")

    resolved, payload = mod.load_png_artifact_bytes(
        root,
        "analysis/example_runs/example_1/visualizations/example_summary_interactive",
    )

    assert resolved == "analysis/example_runs/example_1/visualizations/example_summary_png"
    assert payload == PNG_BYTES


def test_main_view_uses_run_path_and_artifact_without_writing(monkeypatch, tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path)
    viewed: list[tuple[int, str, tuple[float, float]]] = []

    def _fake_view(png_bytes: bytes, *, title: str, figsize: tuple[float, float]) -> None:
        viewed.append((len(png_bytes), title, figsize))

    monkeypatch.setattr(mod, "_view_png_bytes", _fake_view)

    rc = mod.main(
        [
            str(zarr_path),
            "--run-path",
            "analysis/example_runs/example_1",
            "--artifact",
            "example_summary_png",
            "--title",
            "Example",
            "--figsize",
            "4,3",
        ]
    )

    assert rc == 0
    assert viewed == [(len(PNG_BYTES), "Example", (4.0, 3.0))]


def test_main_list_prints_visualization_artifacts(capsys, tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path)

    rc = mod.main([str(zarr_path), "--list"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "analysis/example_runs/example_1/visualizations/example_summary_png" in output
    assert "image/png" in output
    assert "analysis/example_runs/example_1/visualizations/example_summary_interactive" in output
    assert "snapshot=example_summary_png" in output
