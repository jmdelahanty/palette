from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.analysis import chaser_egocentric_bearing as bearing_module
from fisheye.analysis.chaser_distance_runs import write_chaser_distance_run
from fisheye.analysis.chaser_egocentric_bearing import (
    PRE_POST_POLAR_POINT_CLOUD_PNG_ARTIFACT_NAME,
    PRE_POST_POLAR_POINT_CLOUD_VISUALIZATION_CONTRACT_ID,
    PRE_POST_POLAR_PNG_ARTIFACT_NAME,
    PRE_POST_POLAR_VISUALIZATION_CONTRACT_ID,
    STATIC_VISUALIZATION_RENDERER,
    STATIC_VISUALIZATION_RENDERER_VERSION,
    build_chaser_egocentric_bearing_result,
    compute_egocentric_chaser_bearing,
    write_chaser_egocentric_bearing_component,
)
from fisheye.visualization.goodcopbadcop_interactive import (
    DEFAULT_GOODCOPBADCOP_INTERACTIVE_ARTIFACT,
    load_goodcopbadcop_interactive_data,
    to_egocentric_bearing_dataframe,
    to_egocentric_distance_alignment_dataframe,
    to_egocentric_heading_dataframe,
)
from tests.unit.fisheye.test_goodcopbadcop_interactive import (
    _make_archive_with_detection_occupancy,
    _make_chaser_result,
)


def _write_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    values = np.asarray(values)
    chunks = values.shape if values.shape else (1,)
    group.create_array(name, data=values, chunks=chunks, overwrite=True)


def _add_track_kinematics_run(
    zarr_path: Path,
    *,
    sample_valid: np.ndarray | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> None:
    """Install a minimal storage fixture and, when requested, a typed read stub.

    The downstream egocentric tests exercise heading densification and derived
    presentation, not the track-motion publication boundary.  That boundary is
    covered by the dedicated hostile track-motion suites, so these tests patch
    the already-verified logical reader instead of fabricating a legacy archive
    that the future-normal reader must (correctly) reject.
    """

    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    analysis = root.require_group("analysis")
    parent = analysis.require_group("track_kinematics_runs")
    offline = parent.require_group("offline")
    offline.attrs["latest"] = "tk_1"
    run = offline.create_group("tk_1")
    tracks = run.create_group("tracks")
    track = tracks.create_group("id_0")
    frames = np.asarray([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.int64)
    headings = np.zeros(frames.shape[0], dtype=np.float32)
    _write_array(run, "track_ids", np.asarray([0], dtype=np.int32))
    _write_array(track, "frame_indices", frames)
    _write_array(track, "heading_degrees", headings)
    _write_array(track, "smoothed_heading_degrees", headings)
    valid = (
        np.ones(frames.shape[0], dtype=bool)
        if sample_valid is None
        else np.asarray(sample_valid, dtype=bool)
    )
    _write_array(
        track,
        "sample_valid",
        valid,
    )
    if monkeypatch is not None:
        track_path = "analysis/track_kinematics_runs/offline/tk_1/tracks/id_0"

        def load_verified_fixture(
            fixture_root,
            *,
            run_name: str,
            scope: str,
            track_id: int,
            required_speed_levels=(),
        ):
            requested_levels = tuple(required_speed_levels)
            if (
                run_name not in {"tk_1", "latest"}
                or scope != "offline"
                or int(track_id) != 0
                or any(
                    level not in {"raw", "filtered", "smoothed", "averaged"}
                    for level in requested_levels
                )
            ):
                raise ValueError("Requested track differs from the sealed test fixture.")
            fixture_run = fixture_root[
                "analysis/track_kinematics_runs/offline/tk_1"
            ]
            fixture_track = fixture_run["tracks/id_0"]
            fixture_frames = np.asarray(
                fixture_track["frame_indices"][:],
                dtype=np.int64,
            )
            fixture_valid = np.asarray(
                fixture_track["sample_valid"][:],
                dtype=bool,
            )
            fixture_headings = np.asarray(
                fixture_track["heading_degrees"][:],
                dtype=np.float32,
            )
            fps = float(fixture_run.attrs.get("fps", 10.0))
            speed_by_level: dict[str, np.ndarray] = {}
            path_by_level: dict[str, np.ndarray] = {}
            if "speed_filtered_mm" in fixture_track:
                filtered = np.asarray(
                    fixture_track["speed_filtered_mm"][:],
                    dtype=np.float32,
                )
                speed_by_level["filtered"] = filtered
                path_by_level["filtered"] = filtered / fps
            if any(level not in speed_by_level for level in requested_levels):
                raise ValueError(
                    "Requested speed level is absent from the sealed test fixture."
                )
            return SimpleNamespace(
                run_name="tk_1",
                scope="offline",
                run_path="analysis/track_kinematics_runs/offline/tk_1",
                track_path=track_path,
                run_attrs=dict(fixture_run.attrs),
                frame_indices=fixture_frames,
                source_acquisition_frame_index=fixture_frames.copy(),
                time_seconds=fixture_frames.astype(np.float64) / fps,
                heading_degrees=fixture_headings,
                smoothed_heading_degrees=fixture_headings.copy(),
                sample_valid=fixture_valid,
                speed_mm_by_level=speed_by_level,
                speed_px_by_level={},
                frame_path_distance_mm_by_level=path_by_level,
                frame_path_distance_px_by_level={},
            )

        reader_modules = [bearing_module]
        for module_name in (
            "fisheye.utils.export_cross_recording_analytics",
            "fisheye.analysis.goodcopbadcop_common",
            "fisheye.analysis.chaser_epoch_behavior_summary",
            "apps.marimo.components.goodcopbadcop_chaser",
        ):
            reader_module = sys.modules.get(module_name)
            if reader_module is not None:
                reader_modules.append(reader_module)
        for reader_module in reader_modules:
            monkeypatch.setattr(
                reader_module,
                "load_track_kinematics_track",
                load_verified_fixture,
            )


def test_compute_egocentric_chaser_bearing_uses_image_y_down_to_math_y_up() -> None:
    fish = np.zeros((4, 2), dtype=np.float32)
    chaser = np.asarray(
        [
            [[10.0, 0.0]],
            [[0.0, -10.0]],
            [[-10.0, 0.0]],
            [[0.0, 10.0]],
        ],
        dtype=np.float32,
    )
    distance = np.linalg.norm(chaser[:, 0, :] - fish, axis=1).reshape(-1, 1)

    _vector, bearing, alignment, lateral, valid = compute_egocentric_chaser_bearing(
        fish_arena_xy=fish,
        chaser_arena_xy=chaser,
        fish_heading_deg=np.zeros(4, dtype=np.float32),
        distance_mm=distance,
    )

    np.testing.assert_array_equal(valid[:, 0], np.ones(4, dtype=bool))
    np.testing.assert_allclose(bearing[:, 0], [0.0, 90.0, -180.0, -90.0], atol=1e-6)
    np.testing.assert_allclose(alignment[:, 0], [1.0, 0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(lateral[:, 0], [0.0, 1.0, 0.0, -1.0], atol=1e-6)


def test_compute_egocentric_chaser_bearing_requires_finite_distance() -> None:
    _vector, bearing, _alignment, _lateral, valid = compute_egocentric_chaser_bearing(
        fish_arena_xy=np.asarray([[0.0, 0.0]], dtype=np.float32),
        chaser_arena_xy=np.asarray([[[10.0, 0.0]]], dtype=np.float32),
        fish_heading_deg=np.asarray([0.0], dtype=np.float32),
        distance_mm=np.asarray([[np.nan]], dtype=np.float32),
    )

    assert bool(valid[0, 0]) is False
    assert np.isnan(bearing[0, 0])


def test_compute_egocentric_chaser_bearing_gates_validity_inputs() -> None:
    _vector, bearing, _alignment, _lateral, valid = compute_egocentric_chaser_bearing(
        fish_arena_xy=np.zeros((4, 2), dtype=np.float32),
        chaser_arena_xy=np.full((4, 1, 2), [10.0, 0.0], dtype=np.float32),
        fish_heading_deg=np.asarray([0.0, 0.0, 0.0, np.nan], dtype=np.float32),
        fish_valid=np.asarray([True, False, True, True], dtype=bool),
        chaser_valid=np.asarray([[True], [True], [False], [True]], dtype=bool),
        fish_heading_valid=np.asarray([True, True, True, True], dtype=bool),
        distance_mm=np.ones((4, 1), dtype=np.float32),
    )

    np.testing.assert_array_equal(valid[:, 0], np.asarray([True, False, False, False]))
    np.testing.assert_allclose(bearing[:, 0], np.asarray([0.0, np.nan, np.nan, np.nan]), atol=1e-6)


def test_build_chaser_egocentric_bearing_requires_offline_track_kinematics(tmp_path: Path) -> None:
    zarr_path = _make_archive_with_detection_occupancy(tmp_path)
    write_chaser_distance_run(
        zarr_path,
        _make_chaser_result(zarr_path),
        overwrite=True,
        legacy_compatibility=True,
    )

    with pytest.raises(ValueError, match="requires an offline track-kinematics run"):
        build_chaser_egocentric_bearing_result(zarr_path)


def test_build_chaser_egocentric_bearing_falls_back_to_protocol_behavior_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = _make_archive_with_detection_occupancy(tmp_path)
    write_chaser_distance_run(
        zarr_path,
        _make_chaser_result(zarr_path),
        overwrite=True,
        legacy_compatibility=True,
    )
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    del root["analysis/chaser_distance_runs/chaser_distance_1/chasers/behavior_class_label_bytes"]
    _add_track_kinematics_run(zarr_path, monkeypatch=monkeypatch)

    result = build_chaser_egocentric_bearing_result(
        zarr_path,
        chaser_distance_run="chaser_distance_1",
        track_kinematics_run="tk_1",
    )

    assert result.chaser_behavior_labels == ("aggressive", "inert")


def test_build_chaser_egocentric_bearing_densifies_sparse_track_validity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = _make_archive_with_detection_occupancy(tmp_path)
    write_chaser_distance_run(
        zarr_path,
        _make_chaser_result(zarr_path),
        overwrite=True,
        legacy_compatibility=True,
    )
    sample_valid = np.ones(8, dtype=bool)
    sample_valid[3] = False
    _add_track_kinematics_run(
        zarr_path,
        sample_valid=sample_valid,
        monkeypatch=monkeypatch,
    )

    result = build_chaser_egocentric_bearing_result(
        zarr_path,
        chaser_distance_run="chaser_distance_1",
        track_kinematics_run="tk_1",
    )

    assert bool(result.fish_heading_valid[3]) is False
    assert bool(result.fish_heading_valid[4]) is False
    assert bool(result.fish_heading_valid[5]) is True
    np.testing.assert_array_equal(result.valid[3], np.asarray([False, False]))
    np.testing.assert_array_equal(result.valid[4], np.asarray([False, False]))
    np.testing.assert_array_equal(result.valid[5], np.asarray([True, True]))


def test_chaser_egocentric_bearing_writer_refreshes_interactive_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = _make_archive_with_detection_occupancy(tmp_path)
    write_chaser_distance_run(
        zarr_path,
        _make_chaser_result(zarr_path),
        overwrite=True,
        legacy_compatibility=True,
    )
    _add_track_kinematics_run(zarr_path, monkeypatch=monkeypatch)

    result = build_chaser_egocentric_bearing_result(
        zarr_path,
        chaser_distance_run="chaser_distance_1",
        track_kinematics_run="tk_1",
        distance_bin_width_mm=2.0,
        bearing_bin_width_deg=90.0,
    )
    component_path = write_chaser_egocentric_bearing_component(zarr_path, result, overwrite=True)

    assert component_path.endswith("/egocentric_bearing/track_offline_tk_1_id_0_smoothed")
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    component = root[component_path]
    assert component.attrs["schema_id"] == "palette.chaser_egocentric_bearing.v1"
    assert component.attrs["source_refs"]["source_track_kinematics_track_path"].endswith("/tracks/id_0")
    assert component["per_chaser/bearing_deg"].shape == (9, 2)
    assert component["per_chaser/distance_mm"].shape == (9, 2)
    assert component["distance_bearing_histogram/hist_counts"].shape[:2] == (3, 2)
    png = component["visualizations"][PRE_POST_POLAR_PNG_ARTIFACT_NAME]
    png_bytes = np.asarray(png[:], dtype=np.uint8).tobytes()
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert png.attrs["media_type"] == "image/png"
    assert png.attrs["artifact_role"] == "analysis_distribution"
    assert png.attrs["visualization_contract_id"] == PRE_POST_POLAR_VISUALIZATION_CONTRACT_ID
    assert png.attrs["renderer"] == STATIC_VISUALIZATION_RENDERER
    assert png.attrs["renderer_version"] == STATIC_VISUALIZATION_RENDERER_VERSION
    point_cloud_png = component["visualizations"][PRE_POST_POLAR_POINT_CLOUD_PNG_ARTIFACT_NAME]
    point_cloud_png_bytes = np.asarray(point_cloud_png[:], dtype=np.uint8).tobytes()
    assert point_cloud_png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert point_cloud_png.attrs["source_paths"]["distance_mm"].endswith("/per_chaser/distance_mm")
    assert (
        point_cloud_png.attrs["visualization_contract_id"]
        == PRE_POST_POLAR_POINT_CLOUD_VISUALIZATION_CONTRACT_ID
    )
    assert component.attrs["visualizations"][PRE_POST_POLAR_PNG_ARTIFACT_NAME]["media_type"] == "image/png"
    assert component.attrs["visualizations"][PRE_POST_POLAR_POINT_CLOUD_PNG_ARTIFACT_NAME]["media_type"] == "image/png"

    spec_group = root[
        "analysis/chaser_distance_runs/chaser_distance_1/visualizations/"
        f"{DEFAULT_GOODCOPBADCOP_INTERACTIVE_ARTIFACT}"
    ]
    spec = json.loads(np.asarray(spec_group["spec_json"][:], dtype=np.uint8).tobytes().decode("utf-8"))
    assert spec["source_runs"]["egocentric_bearing"] == "track_offline_tk_1_id_0_smoothed"
    assert spec["source_paths"]["egocentric_bearing_deg"].endswith("/per_chaser/bearing_deg")
    assert spec["source_paths"]["egocentric_hist_probability"].endswith(
        "/distance_bearing_histogram/hist_probability"
    )
    assert spec["static_artifacts"]["egocentric_bearing_pre_post_polar"].endswith(
        f"/visualizations/{PRE_POST_POLAR_PNG_ARTIFACT_NAME}"
    )
    assert spec["static_artifacts"]["egocentric_bearing_pre_post_polar_point_cloud"].endswith(
        f"/visualizations/{PRE_POST_POLAR_POINT_CLOUD_PNG_ARTIFACT_NAME}"
    )

    data = load_goodcopbadcop_interactive_data(
        zarr_path,
        run_path="analysis/chaser_distance_runs/chaser_distance_1",
    )
    assert data.egocentric_component_name == "track_offline_tk_1_id_0_smoothed"
    heading_df = to_egocentric_heading_dataframe(data)
    assert heading_df.height == int(np.sum(result.fish_heading_valid))
    assert set(["fish_heading_deg", "fish_heading_valid", "window_label"]).issubset(set(heading_df.columns))
    bearing_df = to_egocentric_bearing_dataframe(data)
    assert bearing_df.height == int(np.sum(result.valid))
    assert set(["bearing_deg", "alignment_cos", "window_label"]).issubset(set(bearing_df.columns))
    alignment_df = to_egocentric_distance_alignment_dataframe(data)
    assert not alignment_df.is_empty()
    assert "mean_alignment_cos" in alignment_df.columns
