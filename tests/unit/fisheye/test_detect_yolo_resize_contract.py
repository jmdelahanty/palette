from __future__ import annotations

import pytest

from fisheye.detection import detect_yolo as mod


def test_normalize_legacy_video_resize_maps_width_height_to_height_width() -> None:
    assert mod._normalize_legacy_video_resize([1280, 768]) == [768, 1280]  # noqa: SLF001


def test_resize_dims_to_imgsz_returns_scalar_for_square_and_list_for_rectangular() -> None:
    assert mod._resize_dims_to_imgsz([640, 640]) == 640  # noqa: SLF001
    assert mod._resize_dims_to_imgsz([768, 1280]) == [768, 1280]  # noqa: SLF001


def test_normalize_decode_backend_defaults_to_auto() -> None:
    assert mod._normalize_decode_backend(None) == "auto"  # noqa: SLF001
    assert mod._normalize_decode_backend("") == "auto"  # noqa: SLF001
    assert mod._normalize_decode_backend("pynvvc_nv12_rgb") == "pynvvc_nv12_rgb"  # noqa: SLF001


def test_normalize_decode_backend_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported decode backend"):
        mod._normalize_decode_backend("not_a_backend")  # noqa: SLF001


def test_record_timing_accumulates_perf_counter_elapsed(monkeypatch) -> None:
    timings = {"read_decode_seconds_total": 1.0}
    monkeypatch.setattr(mod.time, "perf_counter", lambda: 12.5)

    elapsed = mod._record_timing(timings, "read_decode_seconds_total", 10.0)  # noqa: SLF001

    assert elapsed == pytest.approx(2.5)
    assert timings["read_decode_seconds_total"] == pytest.approx(3.5)


def test_detect_yolo_rejects_conflicting_resize_dims_and_imgsz() -> None:
    with pytest.raises(ValueError, match="Conflicting CLI overrides"):
        mod.detect_yolo(
            video_path="missing_video.mp4",
            model_path="missing_model.pt",
            output_zarr="missing_output.zarr",
            resize_dims=[768, 1280],
            imgsz=[640, 640],
        )
