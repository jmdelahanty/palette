from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from fisheye.diagnostics.probe_recording_dish_rim_fit import (
    CircleFit,
    _validate_probe_source_args,
    build_clipped_keyframe_window_specs,
    build_keyframe_window_specs,
    build_parser,
    consensus_circle,
    fit_dish_circle,
    load_clipped_recording_source,
    load_declared_keyframes,
    render_acquisition_reveal,
    temporal_median,
    write_review_package,
)


def _write_clipped_recording_fixture(
    tmp_path: Path,
    *,
    clip_count: int = 5,
) -> Path:
    recording_dir = tmp_path / "session_cam2010093"
    rows = []
    for clip_index in range(clip_count):
        clip_id = f"clip_{clip_index:06d}"
        clip_dir = recording_dir / "clips" / clip_id
        clip_dir.mkdir(parents=True)
        video_path = clip_dir / "camera.mp4"
        keyframe_path = clip_dir / "camera_keyframe.json"
        video_path.write_bytes(b"video")
        keyframe_path.write_text(
            json.dumps(
                {
                    "total_frames": 100,
                    "fps": 10.0,
                    "keyframe_frames": list(range(0, 100, 5)),
                }
            )
        )
        first = clip_index * 100 + 1
        rows.append(
            {
                "session_id": "session",
                "recording_id": recording_dir.name,
                "source_layout": "rolling_clips",
                "clip_id": clip_id,
                "clip_index": clip_index,
                "camera_serial": "2010093",
                "status": "completed",
                "recording_frame_id_gaps": 0,
                "first_recording_frame_id": first,
                "last_recording_frame_id": first + 99,
                "frame_count": 100,
                "video_path": str(video_path.relative_to(recording_dir)),
                "keyframe_path": str(keyframe_path.relative_to(recording_dir)),
            }
        )
    (recording_dir / "recording_clip_index.json").write_text(
        json.dumps(
            {
                "schema_id": "palette.orange_external_ipc_recording_clip_index.v1",
                "schema_version": 1,
                "recording_id": recording_dir.name,
                "session_id": "session",
                "source_layout": "rolling_clips",
                "cameras": ["2010093"],
                "row_count": len(rows),
                "clip_count": len(rows),
                "rows": rows,
            }
        )
    )
    geometry_dir = recording_dir / "raw" / "recording_geometry_bundle"
    geometry_dir.mkdir(parents=True)
    (geometry_dir / "recording_snapshot.json").write_text(
        json.dumps(
            {
                "recording_id": "session",
                "camera_runtime": {
                    "2010093": {
                        "coordinate_frame": {
                            "image_shape": {"height": 4512, "width": 4512}
                        },
                        "runtime": {
                            "frame_rate": 10.0,
                            "height": 4512,
                            "width": 4512,
                        },
                    }
                },
            }
        )
    )
    return recording_dir


def _synthetic_rim(
    *,
    shape: tuple[int, int] = (512, 512),
    circle: tuple[int, int, int] = (251, 258, 220),
) -> np.ndarray:
    image = np.full(shape, 42, dtype=np.uint8)
    cv2.circle(image, circle[:2], circle[2], 205, 5, cv2.LINE_AA)
    cv2.circle(image, circle[:2], circle[2] - 13, 115, 3, cv2.LINE_AA)
    gradient = np.linspace(0, 18, shape[1], dtype=np.uint8)
    return cv2.add(image, np.broadcast_to(gradient, shape))


def test_keyframe_windows_use_only_declared_keyframes() -> None:
    keyframes = tuple(range(0, 140_000, 25))
    specs = build_keyframe_window_specs(
        frame_count=140_000,
        fps=100.0,
        keyframe_frames=keyframes,
        max_keyframes_per_window=21,
    )

    assert [spec.name for spec in specs] == ["early", "middle", "late"]
    assert all(
        3 <= len(spec.frame_indices) <= 21 and len(spec.frame_indices) % 2 == 1
        for spec in specs
    )
    assert all(set(spec.frame_indices) <= set(keyframes) for spec in specs)
    assert all(
        tuple(sorted(spec.frame_indices)) == spec.frame_indices for spec in specs
    )
    assert not set(specs[0].frame_indices) & set(specs[1].frame_indices)
    assert not set(specs[1].frame_indices) & set(specs[2].frame_indices)


def test_clipped_windows_use_recording_clock_and_declared_clip_keyframes(
    tmp_path: Path,
) -> None:
    recording_dir = _write_clipped_recording_fixture(tmp_path)

    source = load_clipped_recording_source(recording_dir)
    specs = build_clipped_keyframe_window_specs(
        source,
        max_keyframes_per_window=5,
        span_seconds=3.0,
    )

    assert source.recording_id == recording_dir.name
    assert source.camera_serial == "2010093"
    assert source.frame_count == 500
    assert source.fps == 10.0
    assert (source.height, source.width) == (4512, 4512)
    assert [spec.name for spec in specs] == ["early", "middle", "late"]
    assert [spec.center_recording_frame_id for spec in specs] == [51, 251, 450]
    assert [sorted({frame.clip_id for frame in spec.frames}) for spec in specs] == [
        ["clip_000000"],
        ["clip_000002"],
        ["clip_000004"],
    ]
    assert all(
        3 <= len(spec.frames) <= 5 and len(spec.frames) % 2 == 1 for spec in specs
    )
    for spec in specs:
        for frame in spec.frames:
            clip = source.clips[frame.clip_index]
            assert frame.clip_local_frame_index in clip.keyframe_frames
            assert (
                frame.recording_frame_id
                == clip.first_recording_frame_id + frame.clip_local_frame_index
            )


def test_clipped_source_rejects_recording_frame_discontinuity(tmp_path: Path) -> None:
    recording_dir = _write_clipped_recording_fixture(tmp_path)
    index_path = recording_dir / "recording_clip_index.json"
    index = json.loads(index_path.read_text())
    index["rows"][1]["first_recording_frame_id"] = 102
    index["rows"][1]["last_recording_frame_id"] = 201
    index_path.write_text(json.dumps(index))

    with pytest.raises(ValueError, match="not continuous"):
        load_clipped_recording_source(recording_dir)


def test_clipped_source_rejects_multiple_camera_streams(tmp_path: Path) -> None:
    recording_dir = _write_clipped_recording_fixture(tmp_path)
    index_path = recording_dir / "recording_clip_index.json"
    index = json.loads(index_path.read_text())
    index["rows"][1]["camera_serial"] = "2010094"
    index_path.write_text(json.dumps(index))

    with pytest.raises(ValueError, match="exactly one camera stream"):
        load_clipped_recording_source(recording_dir)


def test_clipped_source_rejects_unrelated_geometry_snapshot(tmp_path: Path) -> None:
    recording_dir = _write_clipped_recording_fixture(tmp_path)
    snapshot_path = (
        recording_dir / "raw" / "recording_geometry_bundle" / "recording_snapshot.json"
    )
    snapshot = json.loads(snapshot_path.read_text())
    snapshot["recording_id"] = "another_session"
    snapshot_path.write_text(json.dumps(snapshot))

    with pytest.raises(ValueError, match="recording_id is inconsistent"):
        load_clipped_recording_source(recording_dir)


def test_probe_parser_enforces_source_specific_metadata(tmp_path: Path) -> None:
    parser = build_parser()
    clipped = parser.parse_args(
        ["--recording-dir", str(tmp_path / "recording"), "--output-dir", "out"]
    )
    assert _validate_probe_source_args(clipped) == "clipped_recording"

    incomplete_video = parser.parse_args(
        ["--video", "video.mp4", "--output-dir", "out"]
    )
    with pytest.raises(ValueError, match="requires both"):
        _validate_probe_source_args(incomplete_video)


def test_declared_keyframe_summary_must_match_video_summary(tmp_path: Path) -> None:
    path = tmp_path / "keyframes.json"
    path.write_text(
        json.dumps(
            {
                "total_frames": 100,
                "fps": 100.0,
                "keyframe_frames": [0, 25, 50, 75],
            }
        )
    )

    assert load_declared_keyframes(
        path, expected_frame_count=100, expected_fps=100.0
    ) == (0, 25, 50, 75)
    with pytest.raises(ValueError, match="frame count"):
        load_declared_keyframes(path, expected_frame_count=101, expected_fps=100.0)
    with pytest.raises(ValueError, match="fps"):
        load_declared_keyframes(path, expected_frame_count=100, expected_fps=99.0)


def test_temporal_median_rejects_non_uint8_and_suppresses_transient() -> None:
    frames = np.zeros((5, 32, 32), dtype=np.uint8)
    frames[0, 4:20, 4:20] = 255
    assert not np.any(temporal_median(frames))

    with pytest.raises(ValueError, match="uint8"):
        temporal_median(np.zeros((5, 32, 32), dtype=np.float32))


def test_fit_dish_circle_uses_radial_evidence() -> None:
    expected = (251.0, 258.0, 220.0)
    fit, edge = fit_dish_circle(_synthetic_rim(), coarse_max_dimension_px=512)

    assert fit.center_x_px == pytest.approx(expected[0], abs=4.0)
    assert fit.center_y_px == pytest.approx(expected[1], abs=4.0)
    assert fit.radius_px == pytest.approx(expected[2], abs=6.0)
    assert fit.angular_support_fraction > 0.70
    assert fit.candidate_count >= 1
    assert fit.radial_residual_px >= 0.0
    assert len(fit.frozen_candidates) == fit.candidate_count
    assert fit.selected_candidate_id in {
        candidate.candidate_id for candidate in fit.frozen_candidates
    }
    assert fit.selection_reason == "highest_frozen_radial_evidence_score_v1"
    assert edge.shape == (512, 512)
    assert edge.dtype == np.uint8


def test_consensus_uses_medians() -> None:
    fits = [
        CircleFit(10, 20, 30, 0.8, 100, 2),
        CircleFit(11, 19, 31, 0.9, 110, 3),
        CircleFit(400, 500, 600, 0.1, 1, 4),
    ]
    result = consensus_circle(fits)
    assert (result.center_x_px, result.center_y_px, result.radius_px) == (11, 20, 31)
    assert result.candidate_count == 9


def test_acquisition_reveal_does_not_modify_frozen_fit_report(tmp_path: Path) -> None:
    fit_report = {
        "windows": {
            name: {
                "fit": CircleFit(251, 258, 220, 0.9, 100, 1).to_json(),
            }
            for name in ("early", "middle", "late")
        }
    }
    fit_report_path = tmp_path / "fit_report.json"
    fit_report_path.write_text(json.dumps(fit_report, sort_keys=True))
    before = fit_report_path.read_bytes()
    observation = {
        "artifact_id": "dishrim_test",
        "camera": {"height": 512, "width": 512},
        "accepted_inner_rim_boundary": {
            "geometry": {
                "type": "circle",
                "center_px": {"x": 250.0, "y": 257.0},
                "radius_px": 219.0,
            }
        },
    }
    observation_path = tmp_path / "observation.json"
    observation_path.write_text(json.dumps(observation))
    composites = {name: _synthetic_rim() for name in ("early", "middle", "late")}

    reveal_path = render_acquisition_reveal(
        output_dir=tmp_path,
        observation_path=observation_path,
        fit_report_path=fit_report_path,
        composites=composites,
    )

    assert fit_report_path.read_bytes() == before
    reveal = json.loads(reveal_path.read_text())
    assert reveal["purpose"] == "visual_reveal_only_after_blind_palette_fit_was_frozen"
    assert reveal["files"]["early"]["delta_radius_px"] == pytest.approx(1.0)
    support = reveal["acquisition_boundary_edge_support"]
    assert support["fit_frozen_before_measurement"] is True
    assert support["minimum_angular_edge_support_fraction"] > 0.5
    assert support["geometry"] == observation["accepted_inner_rim_boundary"]["geometry"]
    assert (tmp_path / "early_acquisition_reveal.png").is_file()


def test_review_package_binds_exactly_three_panels_and_stops_for_review(
    tmp_path: Path,
) -> None:
    fit_report = tmp_path / "fit_report.json"
    fit_report.write_text('{"status":"provisional_visual_review_required"}\n')
    for name in ("early", "middle", "late"):
        assert cv2.imwrite(
            str(tmp_path / f"{name}_palette_fit.png"),
            np.full((40, 60, 3), 10, dtype=np.uint8),
        )

    receipt_path = write_review_package(tmp_path, acquisition_revealed=False)

    receipt = json.loads(receipt_path.read_text())
    montage = cv2.imread(str(tmp_path / "dish_rim_review_montage.png"))
    assert receipt["status"] == "awaiting_explicit_human_review"
    assert len(receipt["source_panels"]) == 3
    assert montage.shape == (40, 180, 3)
    assert "palette_candidate_publication" in receipt["human_review_required_before"]
