from __future__ import annotations

from fisheye.shared.inference_timing import InferenceTimingProfiler


def test_inference_timing_profiler_summary_tracks_calls_items_and_shares() -> None:
    profiler = InferenceTimingProfiler(enabled=True)
    profiler.record("roi_read", 0.5, items=100)
    profiler.record("roi_read", 0.25, items=50)
    profiler.record("output_write", 0.25, items=150)

    summary = profiler.summary(total_items=150, wall_seconds=1.5, notes=["example note"])

    assert summary["enabled"] is True
    assert summary["total_items"] == 150
    assert summary["wall_seconds"] == 1.5
    assert summary["accounted_seconds"] == 1.0
    assert summary["unaccounted_seconds"] == 0.5
    assert summary["notes"] == ["example note"]
    assert summary["stages"]["roi_read"]["calls"] == 2
    assert summary["stages"]["roi_read"]["items"] == 150
    assert summary["stages"]["output_write"]["calls"] == 1
    assert summary["stages"]["output_write"]["items"] == 150
    assert summary["stages"]["roi_read"]["share_of_wall_time_percent"] == 50.0


def test_inference_timing_profiler_render_lines_orders_by_total_seconds() -> None:
    profiler = InferenceTimingProfiler(enabled=True)
    profiler.record("output_write", 0.1, items=20)
    profiler.record("roi_read", 0.5, items=20)

    lines = profiler.render_lines(total_items=20, wall_seconds=1.0)

    assert lines
    assert lines[0].startswith("roi_read:")
    assert "ms/call" in lines[0]
