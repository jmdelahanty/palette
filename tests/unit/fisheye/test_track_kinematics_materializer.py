from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows.materializers import track_kinematics as mod


def _build_source(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "track-materializer-fixture"


def _fake_track_writer(argv) -> None:
    output = Path(argv[argv.index("--output-zarr-path") + 1])
    run_name = argv[argv.index("--offline-run-name") + 1]
    root = zarr.open_group(str(output), mode="a", zarr_format=3)
    parent = root.require_group("analysis").require_group("track_kinematics_runs")
    offline = parent.require_group("offline")
    run = offline.create_group(run_name)
    run.attrs.update(
        {
            "schema_id": "analysis.track_kinematics_runs",
            "schema_version": 1,
            "method": "track_kinematics_offline",
            "method_version": "track_kinematics.v1",
            "row_axis": "track_samples",
            "source_refs": {"source_keypoints_path": "refined_keypoints_runs/kp_1"},
            "parameters": {"smoothing_seconds": 0.05},
            "palette_run_completion_status": "complete",
            "provenance": {
                "stage": "track_kinematics",
                "command": "unit-test-track-writer",
                "parameters": {},
                "inputs": {"keypoint_run": "refined/kp_1"},
            },
        }
    )
    run.create_array("track_ids", data=np.asarray([0, 1], dtype=np.int32), chunks=(2,))
    tracks = run.create_group("tracks")
    for track_id, row_count, chunk_rows in ((0, 11, 3), (1, 7, 2)):
        track = tracks.create_group(f"id_{track_id}")
        track.attrs["num_samples"] = row_count
        vector = np.arange(row_count, dtype=np.float32)
        track.create_array("frame_indices", data=np.arange(row_count), chunks=(chunk_rows,))
        track.create_array(
            "positions_px",
            data=np.column_stack([vector, vector]),
            chunks=(chunk_rows, 2),
        )
        for name in (
            "speed_raw_px",
            "speed_filtered_px",
            "speed_smoothed_px",
            "acceleration_px",
            "heading_degrees",
            "delta_seconds",
        ):
            track.create_array(name, data=vector, chunks=(chunk_rows,))
        track.create_array(
            "sample_valid",
            data=np.ones(row_count, dtype=bool),
            chunks=(chunk_rows,),
        )


def test_plan_is_read_only_and_refuses_existing_target(tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)

    result = mod.materialize_track_kinematics(
        source,
        scratch_root=scratch,
        keypoint_run="refined/kp_1",
        run_name="track_1",
        apply=False,
    )

    assert result["status"] == "planned"
    assert result["mutates_archive"] is False
    assert not scratch.exists()
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    assert "analysis" not in root


def test_materializer_computes_locally_shards_and_atomically_publishes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)
    monkeypatch.setattr(mod.track_writer, "main", _fake_track_writer)

    result = mod.materialize_track_kinematics(
        source,
        scratch_root=scratch,
        keypoint_run="refined/kp_1",
        run_name="track_1",
        output_shard_rows=5,
        shard_workers=2,
        copy_backend="python",
        apply=True,
        keep_scratch=True,
    )

    assert result["status"] == "complete"
    assert result["publish"]["pre_pointer_validation"]["valid"] is True
    assert result["publish"]["final_validation"]["valid"] is True
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    parent = root["analysis/track_kinematics_runs"]
    offline = parent["offline"]
    run = offline["track_1"]
    assert parent.attrs["latest"] == "offline/track_1"
    assert parent.attrs["latest_complete"] == "offline/track_1"
    assert parent.attrs["latest_offline"] == "track_1"
    assert offline.attrs["latest"] == "track_1"
    assert tuple(run["tracks/id_0/speed_raw_px"].shards) == (6,)
    assert tuple(run["tracks/id_1/speed_raw_px"].shards) == (6,)
    staging = run.attrs["cluster_output_staging"]
    assert staging["serialization_policy"] == (
        "per_recording_advisory_file_lock"
    )
    assert staging["publisher_contract"] == {
        "schema_id": "palette.atomic_run_group_publisher",
        "schema_version": 1,
    }
    assert set(staging["parent_attrs_before"]) == {
        "analysis/track_kinematics_runs",
        "analysis/track_kinematics_runs/offline",
    }
    final_track_attrs = staging["parent_attrs_after"]["analysis/track_kinematics_runs"]
    assert final_track_attrs["latest"] == "offline/track_1"
    assert final_track_attrs["latest_complete"] == "offline/track_1"
    assert final_track_attrs["latest_offline"] == "track_1"
    assert staging["parent_attrs_after"]["analysis/track_kinematics_runs/offline"][
        "latest"
    ] == "track_1"
    assert (tmp_path / ".source.zarr.track-kinematics-publish.lock").is_file()


def test_track_writer_rejects_no_write_with_separate_output(tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    _build_source(source)

    try:
        mod.track_writer.main(
            [
                str(source),
                "--output-zarr-path",
                str(tmp_path / "output.zarr"),
                "--no-write",
            ]
        )
    except ValueError as exc:
        assert "--no-write cannot be combined" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected contradictory output flags to be rejected.")


def test_publish_rolls_back_run_and_both_pointer_parents(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)
    root = zarr.open_group(str(source), mode="a", use_consolidated=False)
    track_parent = root.require_group("analysis").require_group("track_kinematics_runs")
    offline_parent = track_parent.require_group("offline")
    track_parent.attrs["latest"] = "offline/previous"
    track_parent.attrs["latest_complete"] = "offline/previous"
    track_parent.attrs["latest_offline"] = "previous"
    offline_parent.attrs["latest"] = "previous"
    monkeypatch.setattr(mod.track_writer, "main", _fake_track_writer)
    mark_complete = mod.track_writer.mark_track_kinematics_run_complete

    def fail_after_pointer_update(*args, **kwargs):
        mark_complete(*args, **kwargs)
        raise RuntimeError("injected post-pointer failure")

    monkeypatch.setattr(
        mod.track_writer,
        "mark_track_kinematics_run_complete",
        fail_after_pointer_update,
    )

    with pytest.raises(RuntimeError, match="injected post-pointer failure"):
        mod.materialize_track_kinematics(
            source,
            scratch_root=scratch,
            keypoint_run="refined/kp_1",
            run_name="track_1",
            output_shard_rows=5,
            copy_backend="python",
            apply=True,
        )

    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    track_parent = root["analysis/track_kinematics_runs"]
    offline_parent = track_parent["offline"]
    assert "track_1" not in offline_parent
    assert track_parent.attrs["latest"] == "offline/previous"
    assert track_parent.attrs["latest_complete"] == "offline/previous"
    assert track_parent.attrs["latest_offline"] == "previous"
    assert offline_parent.attrs["latest"] == "previous"
