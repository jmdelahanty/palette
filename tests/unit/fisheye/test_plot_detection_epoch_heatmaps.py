from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fisheye.analysis.chaser_behavior import ConfiguredChaserBehavior
import fisheye.visualization.plot_detection_epoch_heatmaps as heatmap_module
from fisheye.visualization.plot_detection_epoch_heatmaps import ChaserPositionMarker
from fisheye.visualization.plot_detection_epoch_heatmaps import ChaserPositionOverlay
from fisheye.visualization.plot_detection_epoch_heatmaps import EpochWindow
from fisheye.visualization.plot_detection_epoch_heatmaps import HeatmapResult
from fisheye.visualization.plot_detection_epoch_heatmaps import (
    build_chaser_event_windows,
)
from fisheye.visualization.plot_detection_epoch_heatmaps import (
    compute_chaser_position_overlays,
)
from fisheye.visualization.plot_detection_epoch_heatmaps import compute_heatmap
from fisheye.visualization.plot_detection_epoch_heatmaps import default_windows
from fisheye.visualization.plot_detection_epoch_heatmaps import parse_window_specs
from fisheye.visualization.plot_detection_epoch_heatmaps import render_heatmap_panel
from fisheye.visualization.plot_detection_epoch_heatmaps import write_summary


def _behavior(
    chaser_index: int,
    behavior_class_id: int,
    behavior_class: str,
    raw_color_rgba: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
) -> ConfiguredChaserBehavior:
    return ConfiguredChaserBehavior(
        chaser_index=chaser_index,
        behavior_class_id=behavior_class_id,
        behavior_class=behavior_class,
        enable_chase=behavior_class == "aggressive",
        enable_random_movement=behavior_class == "random_non_chasing",
        behavior_mode=None,
        raw_color_rgba=raw_color_rgba,
        start_position_preset="start",
        end_position_preset="end",
    )


def _overlay_kwargs() -> dict[str, object]:
    return {
        "source_camera_width": 1000,
        "source_camera_height": 1000,
        "fps": 10.0,
        "post_settle_duration_s": 1.0,
        "stimulus_run": "stimulus_canonical_v1_test",
        "source_chaser_path": (
            "analysis/stimulus_runs/stimulus_canonical_v1_test/"
            "tracking_data/chaser_states"
        ),
        "source_camera_id": "camera-asymmetric",
        "coordinate_descriptor_sha256": "a" * 64,
        "frame_transform_manifest_ref": "/stimulus@transform",
        "frame_transform_manifest_sha256": "b" * 64,
        "protocol_profile_id": "chaser_event_windows_v1",
        "protocol_profile_version": 1,
        "protocol_profile_sha256": "c" * 64,
    }


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


def test_compute_chaser_overlays_labels_static_pre_and_post_without_flip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows = [
        EpochWindow(
            "pre_event", 0.0, 1.0, start_frame=0, end_frame=9, source="stimulus_events"
        ),
        EpochWindow(
            "training_event",
            1.0,
            2.0,
            start_frame=10,
            end_frame=19,
            source="stimulus_events",
        ),
        EpochWindow(
            "post_event",
            2.0,
            4.0,
            start_frame=20,
            end_frame=39,
            source="stimulus_events",
        ),
    ]
    frames = np.array([0, 0, 5, 5, 20, 20, 30, 30, 35, 35], dtype=np.int64)
    indices = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    positions = np.array(
        [
            [10.0, 20.0],
            [30.0, 40.0],
            [10.0, 20.0],
            [30.0, 40.0],
            [700.0, 700.0],
            [650.0, 650.0],
            [50.0, 60.0],
            [70.0, 80.0],
            [50.0, 60.0],
            [70.0, 80.0],
        ],
        dtype=np.float64,
    )
    class_ids = np.array([1, 3, 1, 3, 1, 3, 1, 3, 1, 3], dtype=np.int8)
    transformed_inputs: list[np.ndarray] = []

    def asymmetric_transform(points: np.ndarray, _chain: object) -> np.ndarray:
        transformed_inputs.append(np.asarray(points).copy())
        return np.asarray(points, dtype=np.float64) + np.array([100.0, 200.0])

    monkeypatch.setattr(
        heatmap_module,
        "apply_bound_directed_transform_chain",
        asymmetric_transform,
    )
    overlays = compute_chaser_position_overlays(
        windows=windows,
        source_acquisition_frame_index=frames,
        chaser_index=indices,
        chaser_position_arena_xy=positions,
        chaser_behavior_class_id=class_ids,
        configured_behaviors=(
            _behavior(0, 1, "aggressive", (0.0, 0.0, 1.0, 1.0)),
            _behavior(1, 3, "inert"),
        ),
        transform_chain=object(),
        **_overlay_kwargs(),
    )

    assert [overlay.window_label for overlay in overlays] == [
        "pre_event",
        "post_event",
    ]
    assert overlays[1].effective_start_frame == 30
    assert [marker.behavior_class for marker in overlays[0].markers] == [
        "aggressive",
        "inert",
    ]
    assert [marker.behavior_marker for marker in overlays[0].markers] == ["*", "o"]
    assert [marker.experimental_color_hex for marker in overlays[0].markers] == [
        "#0000ff",
        "#ff0000",
    ]
    assert [
        (marker.camera_x_px, marker.camera_y_px) for marker in overlays[0].markers
    ] == [(110.0, 220.0), (130.0, 240.0)]
    assert [
        (marker.camera_x_px, marker.camera_y_px) for marker in overlays[1].markers
    ] == [(150.0, 260.0), (170.0, 280.0)]
    assert all(
        marker.sample_count == 2 for overlay in overlays for marker in overlay.markers
    )
    assert all(
        marker.max_drift_px == pytest.approx(0.0)
        for overlay in overlays
        for marker in overlay.markers
    )
    assert not any(np.any(points == 700.0) for points in transformed_inputs)


def test_compute_chaser_overlays_fails_closed_on_role_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        heatmap_module,
        "apply_bound_directed_transform_chain",
        lambda points, _chain: points,
    )
    windows = [
        EpochWindow("pre_event", 0.0, 0.1, start_frame=0, end_frame=0),
        EpochWindow("post_event", 0.1, 0.2, start_frame=1, end_frame=1),
    ]
    with pytest.raises(ValueError, match="classification disagrees with protocol"):
        compute_chaser_position_overlays(
            windows=windows,
            source_acquisition_frame_index=np.array([0, 1]),
            chaser_index=np.array([0, 0]),
            chaser_position_arena_xy=np.array([[1.0, 2.0], [1.0, 2.0]]),
            chaser_behavior_class_id=np.array([3, 3]),
            configured_behaviors=(_behavior(0, 1, "aggressive"),),
            transform_chain=object(),
            **{**_overlay_kwargs(), "post_settle_duration_s": 0.0},
        )


def test_compute_chaser_overlays_fails_closed_outside_camera_extent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        heatmap_module,
        "apply_bound_directed_transform_chain",
        lambda points, _chain: np.asarray(points) + np.array([2000.0, 0.0]),
    )
    windows = [
        EpochWindow("pre_event", 0.0, 0.1, start_frame=0, end_frame=0),
        EpochWindow("post_event", 0.1, 0.2, start_frame=1, end_frame=1),
    ]
    with pytest.raises(ValueError, match="outside the source-camera extent"):
        compute_chaser_position_overlays(
            windows=windows,
            source_acquisition_frame_index=np.array([0, 1]),
            chaser_index=np.array([0, 0]),
            chaser_position_arena_xy=np.array([[1.0, 2.0], [1.0, 2.0]]),
            chaser_behavior_class_id=np.array([1, 1]),
            configured_behaviors=(_behavior(0, 1, "aggressive"),),
            transform_chain=object(),
            **{**_overlay_kwargs(), "post_settle_duration_s": 0.0},
        )


def test_render_and_summary_include_behavior_labeled_overlay(tmp_path: Path) -> None:
    overlay = ChaserPositionOverlay(
        window_label="pre_event",
        effective_start_frame=0,
        effective_end_frame=9,
        stimulus_run="stimulus_canonical_v1_test",
        source_chaser_path="analysis/stimulus_runs/test/tracking_data/chaser_states",
        source_camera_id="camera-asymmetric",
        source_camera_width=100,
        source_camera_height=100,
        coordinate_descriptor_sha256="a" * 64,
        frame_transform_manifest_ref="/stimulus@transform",
        frame_transform_manifest_sha256="b" * 64,
        protocol_profile_id="chaser_event_windows_v1",
        protocol_profile_version=1,
        protocol_profile_sha256="c" * 64,
        post_settle_duration_s=0.0,
        markers=(
            ChaserPositionMarker(
                chaser_index=0,
                behavior_class_id=1,
                behavior_class="aggressive",
                behavior_marker="*",
                experimental_color_rgba=(0.0, 0.0, 0.0, 1.0),
                experimental_color_hex="#000000",
                camera_x_px=25.0,
                camera_y_px=75.0,
                sample_count=10,
                median_drift_px=0.0,
                max_drift_px=0.0,
            ),
        ),
    )
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
        chaser_overlay=overlay,
    )
    png_path = tmp_path / "overlay.png"
    summary_path = tmp_path / "overlay.json"

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
    assert payload[0]["chaser_overlay"]["markers"] == [
        {
            "behavior_class": "aggressive",
            "behavior_class_id": 1,
            "behavior_marker": "*",
            "camera_x_px": 25.0,
            "camera_y_px": 75.0,
            "chaser_index": 0,
            "experimental_color_hex": "#000000",
            "experimental_color_rgba": [0.0, 0.0, 0.0, 1.0],
            "max_drift_px": 0.0,
            "median_drift_px": 0.0,
            "sample_count": 10,
        }
    ]
    assert payload[0]["chaser_overlay"]["frame_transform_manifest_sha256"] == "b" * 64
