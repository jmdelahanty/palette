from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import fisheye.visualization.plot_detection_epoch_heatmaps as heatmap_module
from fisheye.visualization.plot_detection_epoch_heatmaps import EpochWindow
from fisheye.visualization.plot_detection_epoch_heatmaps import HeatmapResult
from fisheye.visualization.plot_detection_epoch_heatmaps import (
    build_chaser_event_windows,
)
from fisheye.visualization.plot_detection_epoch_heatmaps import compute_heatmap
from fisheye.visualization.plot_detection_epoch_heatmaps import default_windows
from fisheye.visualization.plot_detection_epoch_heatmaps import parse_args
from fisheye.visualization.plot_detection_epoch_heatmaps import parse_window_specs
from fisheye.visualization.plot_detection_epoch_heatmaps import render_heatmap_panel
from fisheye.visualization.plot_detection_epoch_heatmaps import write_summary


def test_default_windows_match_pre_training_post_durations() -> None:
    windows = default_windows(
        first_minutes=10.0, training_minutes=2.5, post_minutes=10.0
    )

    assert [(w.label, w.start_s, w.end_s) for w in windows] == [
        ("first_10min", 0.0, 600.0),
        ("training_2p5min", 600.0, 750.0),
        ("post_10min", 750.0, 1350.0),
    ]


def test_parse_window_specs_requires_ordered_seconds() -> None:
    windows = parse_window_specs(["baseline:0:60", "stim:60:90"])
    assert windows == [
        EpochWindow("baseline", 0.0, 60.0),
        EpochWindow("stim", 60.0, 90.0),
    ]

    with pytest.raises(ValueError, match="end must be after start"):
        parse_window_specs(["bad:20:10"])


def test_compute_heatmap_filters_by_frame_and_reports_coverage() -> None:
    frames = np.array([0, 1, 2, 5, 10], dtype=np.int64)
    centers = np.array(
        [
            [5.0, 5.0],
            [15.0, 5.0],
            [15.0, 5.0],
            [25.0, 25.0],
            [35.0, 35.0],
        ],
        dtype=np.float64,
    )

    heatmap, start, end, span, detections, covered, coverage = compute_heatmap(
        frames=frames,
        centers=centers,
        window=EpochWindow("test", 0.0, 0.4),
        width=40,
        height=40,
        fps=10.0,
        total_frames=20,
        bin_size=10,
        smooth_sigma=0.0,
        normalize="count",
    )

    assert (start, end, span) == (0, 3, 4)
    assert detections == 3
    assert covered == 3
    assert coverage == pytest.approx(75.0)
    assert heatmap.sum() == pytest.approx(3.0)


def test_compute_heatmap_prefers_explicit_event_frame_bounds() -> None:
    frames = np.array([9, 10, 11, 12, 13], dtype=np.int64)
    centers = np.array([[5.0, 5.0]] * len(frames), dtype=np.float64)

    _, start, end, span, detections, covered, coverage = compute_heatmap(
        frames=frames,
        centers=centers,
        window=EpochWindow(
            "event", 0.0, 100.0, start_frame=10, end_frame=12, source="stimulus_events"
        ),
        width=20,
        height=20,
        fps=10.0,
        total_frames=20,
        bin_size=10,
        smooth_sigma=0.0,
        normalize="count",
    )

    assert (start, end, span) == (10, 12, 3)
    assert detections == 3
    assert covered == 3
    assert coverage == pytest.approx(100.0)


def test_detection_payload_resolves_canonical_instances_layout() -> None:
    instances = {
        "frame_indices": object(),
        "bbox_img_xyxy": object(),
    }
    group, path = heatmap_module._resolve_detection_payload_group(
        {"instances": instances},
        "refined_detect_runs/finalized",
    )

    assert group is instances
    assert path == "refined_detect_runs/finalized/instances"


def test_build_chaser_event_windows_uses_camera_frame_events() -> None:
    windows = build_chaser_event_windows(
        {
            "CHASER_PRE_PERIOD_START": 731,
            "CHASER_TRAINING_START": 60_731,
            "CHASER_POST_PERIOD_START": 78_731,
            "PROTOCOL_FINISH": 138_731,
        },
        fps=100.0,
        total_frames=143_447,
    )

    assert [(w.label, w.start_frame, w.end_frame, w.source) for w in windows] == [
        ("pre_event", 731, 60_730, "stimulus_events"),
        ("training_event", 60_731, 78_730, "stimulus_events"),
        ("post_event", 78_731, 138_730, "stimulus_events"),
    ]
    assert windows[0].start_s == pytest.approx(7.31)
    assert windows[2].end_s == pytest.approx(1387.31)


def test_overlay_chasers_option_has_been_removed() -> None:
    with pytest.raises(SystemExit):
        parse_args(["recording.zarr", "--output", "heatmap.png", "--overlay-chasers"])


def test_render_and_summary_have_no_legacy_chaser_overlay(tmp_path: Path) -> None:
    result = HeatmapResult(
        recording_id="recording-test",
        arena_id="arena-1",
        camera_id="camera-asymmetric",
        zarr_path="/recording.zarr",
        source_path="refined_detect_runs/final",
        source_kind="refined",
        width=100,
        height=100,
        fps=10.0,
        window=EpochWindow("pre_event", 0.0, 1.0, start_frame=0, end_frame=9),
        start_frame=0,
        end_frame=9,
        total_span_frames=10,
        detection_count=5,
        covered_frame_count=5,
        coverage_pct=50.0,
        heatmap=np.ones((10, 10), dtype=np.float64),
    )
    png_path = tmp_path / "heatmap.png"
    summary_path = tmp_path / "heatmap.json"

    render_heatmap_panel(
        [[result]],
        output=png_path,
        title="Overlay",
        origin="upper",
        cmap="inferno",
        normalize="max",
    )
    write_summary([[result]], summary_path)

    assert png_path.stat().st_size > 1000
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "chaser_overlay" not in payload[0]
