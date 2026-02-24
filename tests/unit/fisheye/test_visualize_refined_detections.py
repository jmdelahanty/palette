from __future__ import annotations

from pathlib import Path

from fisheye.visualization import visualize_refined_detections as mod


def test_render_refinement_pipeline_png_uses_noninteractive_save(monkeypatch) -> None:
    fake_png = b"\x89PNG\r\n\x1a\nFAKE_REFINED"
    calls = {"count": 0}

    def _fake_visualize(
        zarr_path: str,
        refined_run=None,
        save_path=None,
        frame_range=None,
        show=True,
    ):
        calls["count"] += 1
        assert zarr_path == "/tmp/fake.zarr"
        assert refined_run == "refined_detect_1"
        assert frame_range is None
        assert show is False
        assert save_path is not None
        Path(save_path).write_bytes(fake_png)

    monkeypatch.setattr(mod, "visualize_refinement_pipeline", _fake_visualize)

    png_bytes, meta = mod.render_refinement_pipeline_png(
        "/tmp/fake.zarr",
        refined_run="refined_detect_1",
    )
    assert calls["count"] == 1
    assert png_bytes == fake_png
    assert meta["refined_run"] == "refined_detect_1"
