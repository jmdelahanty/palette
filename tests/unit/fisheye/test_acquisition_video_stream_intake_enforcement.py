"""Refusal at the real inventory writer, before any Zarr mutation."""

import pytest

from fisheye.shared import acquisition_video_streams as mod


class NoWrites:
    def require_group(self, name):
        pytest.fail(f"invalid source reached inventory mutation: {name}")


@pytest.mark.parametrize("manifest", [{}, {"video_streams": None}])
def test_undeclared_stream_inventory_is_optional(tmp_path, manifest):
    assert mod.write_acquisition_video_stream_inventory(NoWrites(), tmp_path, manifest) is None


@pytest.mark.parametrize("damage", [
    "malformed_streams", "malformed_optional_stream", "missing_summary",
    "invalid_summary_path", "unreadable_summary", "duplicate_summary",
    "failed_summary", "row_count_mismatch", "missing_clock", "color_mismatch",
    "undeclared_conventional_clock",
])
def test_inventory_writer_refuses_declared_component_failures(tmp_path, monkeypatch, damage):
    (tmp_path / "full.mp4").write_bytes(b"video")
    (tmp_path / "clock.csv").write_text("frame\n0\n", encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    full = {"video": "full.mp4", "output_kind": "full", "frame_count": 1,
            "frame_clock_metadata": "clock.csv", "summary": "summary.json"}
    manifest = {"video_streams": {"streams": {"full": full}}}
    observed = {}
    if damage == "malformed_streams":
        manifest["video_streams"]["streams"] = []
    elif damage == "malformed_optional_stream":
        manifest["video_streams"]["streams"]["crop"] = False
    elif damage == "missing_summary":
        full["summary"] = "missing.json"
    elif damage == "invalid_summary_path":
        full["summary"] = False
    elif damage == "unreadable_summary":
        (tmp_path / "summary.json").write_text("bad json", encoding="utf-8")
    elif damage == "duplicate_summary":
        (tmp_path / "summary.json").write_text('{"status":"failed","status":"completed"}', encoding="utf-8")
    elif damage == "failed_summary":
        (tmp_path / "summary.json").write_text('{"status":"failed"}', encoding="utf-8")
    elif damage == "row_count_mismatch":
        full["frame_count"] = 2
    elif damage == "missing_clock":
        full["frame_clock_metadata"] = "missing.csv"
    elif damage == "color_mismatch":
        full["color_range"] = "pc"
        observed = {"video_color_range": "tv"}
    elif damage == "undeclared_conventional_clock":
        full.pop("frame_clock_metadata")
        (tmp_path / "full_meta.csv").write_text("frame\n0\n", encoding="utf-8")
    monkeypatch.setattr(mod, "probe_video_colorimetry_attrs", lambda _: observed)
    with pytest.raises(ValueError, match="acquisition video stream"):
        mod.write_acquisition_video_stream_inventory(NoWrites(), tmp_path, manifest)
