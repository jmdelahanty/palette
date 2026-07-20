from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows.materializers import track_kinematics as mod


def _build_source(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "recording_id": "track-materializer-fixture",
            "source_revision": "source-revision-1",
        }
    )


def _populate_unbound_stage(
    staging_zarr: Path,
    *,
    run_name: str,
    detached_descriptor: bool = False,
) -> zarr.Group:
    root = zarr.open_group(
        str(staging_zarr),
        mode="w",
        zarr_format=3,
        use_consolidated=False,
    )
    parent = root.require_group("analysis").require_group("track_kinematics_runs")
    run = parent.require_group("offline").create_group(run_name)
    staging_manifest = {
        "schema_id": "palette.track_kinematics_staging_manifest.v1",
        "schema_version": 1,
        "run_name": run_name,
        "keypoint_run": "refined/kp_1",
        "source_coordinate_node_ref": "/crop_runs/crop_1/centers_img_xy",
        "source_coordinate_descriptor_sha256": "a" * 64,
        "source_row_identity_sha256": "b" * 64,
        "numeric_payload_manifest_sha256": "c" * 64,
    }
    run.attrs.update(
        {
            "schema_id": "analysis.track_kinematics_runs",
            "schema_version": 1,
            "method": "track_kinematics_offline",
            "method_version": "track_kinematics.v1",
            "row_axis": "track_samples",
            "source_refs": {
                "source_keypoints_path": "refined_keypoints_runs/kp_1",
            },
            "parameters": {"smoothing_seconds": 0.05},
            "palette_run_completion_contract": "palette.zarr_run_completion.v1",
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
            mod.COORDINATE_BINDING_STATUS_ATTR: mod.UNBOUND_STAGE_STATUS,
            mod.STAGING_MANIFEST_ATTR: staging_manifest,
            mod.STAGING_MANIFEST_DIGEST_ATTR: mod._canonical_mapping_sha256(
                staging_manifest
            ),
            "provenance": {
                "stage": "track_kinematics",
                "command": "unit-test-track-staging-writer",
                "parameters": {},
                "inputs": {"keypoint_run": "refined/kp_1"},
            },
        }
    )
    run.create_array(
        "track_ids",
        data=np.asarray([0, 1], dtype=np.int32),
        chunks=(2,),
    )
    tracks = run.create_group("tracks")
    for track_id, row_count, chunk_rows in ((0, 11, 3), (1, 7, 2)):
        track = tracks.create_group(f"id_{track_id}")
        track.attrs["num_samples"] = row_count
        frames = np.arange(row_count, dtype=np.int64) + track_id * 100
        vector = np.arange(row_count, dtype=np.float32) + track_id * 100.0
        track.create_array("frame_indices", data=frames, chunks=(chunk_rows,))
        track.create_array(
            "track_sample_key",
            data=np.column_stack(
                [np.full(row_count, track_id, dtype=np.int64), frames]
            ),
            chunks=(chunk_rows, 2),
        )
        track.create_array(
            "source_acquisition_frame_index",
            data=frames,
            chunks=(chunk_rows,),
        )
        interpolation = np.zeros(
            row_count,
            dtype=np.dtype(
                [
                    ("left_source_frame_index", "<i8"),
                    ("right_source_frame_index", "<i8"),
                    ("right_weight", "<f8"),
                ]
            ),
        )
        interpolation["left_source_frame_index"] = frames
        interpolation["right_source_frame_index"] = frames
        track.create_array(
            "source_frame_interpolation",
            data=interpolation,
            chunks=(chunk_rows,),
        )
        source_instances = np.zeros(
            row_count,
            dtype=np.dtype([("valid", "?"), ("instance_key", "<u8")]),
        )
        source_instances["valid"] = True
        source_instances["instance_key"] = np.arange(row_count, dtype=np.uint64)
        track.create_array(
            "source_instance_key",
            data=source_instances,
            chunks=(chunk_rows,),
        )
        track.create_array(
            "source_row_index",
            data=np.arange(row_count, dtype=np.int64) + track_id * 100,
            chunks=(chunk_rows,),
        )
        positions = track.create_array(
            "positions_px",
            data=np.column_stack([vector, vector]),
            chunks=(chunk_rows, 2),
        )
        if detached_descriptor:
            positions.attrs["coordinate_descriptor"] = {
                "schema_id": "palette.coordinate_descriptor",
                "schema_version": 2,
                "space_id": "source_camera_image_px",
            }
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
    return run


def _install_writer_api(
    monkeypatch: pytest.MonkeyPatch,
    events: list[tuple[Any, ...]],
    *,
    detached_descriptor: bool = False,
) -> None:
    def stage(
        source_zarr,
        staging_zarr,
        *,
        keypoint_run,
        run_name,
        writer_arguments=(),
    ):
        source_path = Path(source_zarr).resolve()
        staging_path = Path(staging_zarr).resolve()
        source_root = zarr.open_group(
            str(source_path), mode="r", use_consolidated=False
        )
        assert str(source_root.attrs["source_revision"]) == "source-revision-1"
        assert keypoint_run == "refined/kp_1"
        events.append(
            (
                "stage",
                source_path,
                staging_path,
                tuple(writer_arguments),
            )
        )
        _populate_unbound_stage(
            staging_path,
            run_name=run_name,
            detached_descriptor=detached_descriptor,
        )
        return {
            "valid": True,
            "status": mod.UNBOUND_STAGE_STATUS,
            "run_name": run_name,
            "track_count": 2,
            "staging_manifest_sha256": "c" * 64,
        }

    def bind(
        authoritative_root,
        final_run_group,
        *,
        expected_keypoint_run,
        expected_run_name,
    ):
        assert expected_keypoint_run == "refined/kp_1"
        assert expected_run_name == "track_1"
        assert final_run_group.path == (
            "analysis/track_kinematics_runs/offline/track_1"
        )
        assert (
            str(final_run_group.attrs[mod.COORDINATE_BINDING_STATUS_ATTR])
            == mod.PUBLISHING_BINDING_STATUS
        )
        assert str(final_run_group.attrs["palette_run_completion_status"]) == "running"
        assert final_run_group.attrs["stage_selector_eligible"] is False
        assert str(authoritative_root.attrs["source_revision"]) == "source-revision-1"
        events.append(
            (
                "bind",
                final_run_group.path,
                str(final_run_group.attrs["palette_run_completion_status"]),
            )
        )
        for track_name in final_run_group["tracks"].group_keys():
            track = final_run_group[f"tracks/{track_name}"]
            position = track["positions_px"]
            position.attrs["coordinate_descriptor"] = {
                "schema_id": "palette.coordinate_descriptor",
                "schema_version": 2,
                "node_ref": f"/{position.path}",
            }
            track.attrs["row_identity_contract"] = {
                "schema_id": "palette.row_identity_contract",
                "schema_version": 1,
            }
            track.attrs["track_position_derivation"] = {
                "schema_id": "palette.track_position_derivation",
                "schema_version": 1,
            }
        final_run_group.attrs[mod.COORDINATE_BINDING_STATUS_ATTR] = (
            mod.BOUND_CANONICAL_STATUS
        )
        return {
            "valid": True,
            "status": mod.BOUND_CANONICAL_STATUS,
            "track_count": 2,
            "binding_manifest_sha256": "d" * 64,
        }

    def validate_before_selection(
        authoritative_root,
        final_run_group,
        *,
        expected_keypoint_run,
        expected_run_name,
        require_complete,
    ):
        assert expected_keypoint_run == "refined/kp_1"
        assert expected_run_name == "track_1"
        expected_completion = "complete" if require_complete else "running"
        assert (
            str(final_run_group.attrs["palette_run_completion_status"])
            == expected_completion
        )
        assert (
            str(final_run_group.attrs[mod.COORDINATE_BINDING_STATUS_ATTR])
            == mod.BOUND_CANONICAL_STATUS
        )
        assert final_run_group.attrs["stage_selector_eligible"] is False
        assert str(authoritative_root.attrs["source_revision"]) == "source-revision-1"
        for track_name in final_run_group["tracks"].group_keys():
            position = final_run_group[f"tracks/{track_name}/positions_px"]
            assert position.attrs["coordinate_descriptor"]["node_ref"] == (
                f"/{position.path}"
            )
        events.append(
            ("validate_before", bool(require_complete), expected_completion)
        )
        return {
            "valid": True,
            "status": "canonical_track_run_valid",
            "track_count": 2,
            "binding_manifest_sha256": "d" * 64,
        }

    def validate_public(
        authoritative_root,
        final_run_group,
    ):
        assert final_run_group.attrs["palette_run_completion_status"] == "complete"
        assert final_run_group.attrs["stage_selector_eligible"] is True
        assert str(authoritative_root.attrs["source_revision"]) == "source-revision-1"
        events.append(("validate_public", True, "complete"))
        return SimpleNamespace(
            run_name="track_1",
            track_positions=((0, object()), (1, object())),
        )

    monkeypatch.setattr(
        mod.track_writer,
        "stage_offline_track_kinematics_run",
        stage,
        raising=False,
    )
    monkeypatch.setattr(
        mod.track_writer,
        "bind_staged_offline_track_kinematics_run",
        bind,
        raising=False,
    )
    monkeypatch.setattr(
        mod.track_writer,
        "_validate_bound_offline_track_kinematics_run_before_selection",
        validate_before_selection,
        raising=False,
    )
    monkeypatch.setattr(
        mod.track_writer,
        "validate_bound_track_position_bindings",
        validate_public,
        raising=False,
    )

    def legacy_main_was_used(_argv):  # pragma: no cover - defensive assertion
        raise AssertionError("materializer must not use legacy --output-zarr-path")

    monkeypatch.setattr(mod.track_writer, "main", legacy_main_was_used)


def _seed_previous_pointers(source: Path) -> None:
    root = zarr.open_group(str(source), mode="a", use_consolidated=False)
    track_parent = root.require_group("analysis").require_group(
        "track_kinematics_runs"
    )
    offline_parent = track_parent.require_group("offline")
    track_parent.attrs.update(
        {
            "latest": "offline/previous",
            "latest_complete": "offline/previous",
            "latest_offline": "previous",
        }
    )
    offline_parent.attrs["latest"] = "previous"


def _assert_previous_pointers_restored(source: Path) -> None:
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    track_parent = root["analysis/track_kinematics_runs"]
    offline_parent = track_parent["offline"]
    assert "track_1" not in offline_parent
    assert track_parent.attrs["latest"] == "offline/previous"
    assert track_parent.attrs["latest_complete"] == "offline/previous"
    assert track_parent.attrs["latest_offline"] == "previous"
    assert offline_parent.attrs["latest"] == "previous"


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
    assert result["plan"]["staging_zarr"].endswith("track-staging.zarr")
    assert not scratch.exists()
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    assert "analysis" not in root


def test_materializer_stages_unbound_then_binds_only_at_final_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)
    events: list[tuple[Any, ...]] = []
    _install_writer_api(monkeypatch, events)

    result = mod.materialize_track_kinematics(
        source,
        scratch_root=scratch,
        keypoint_run="refined/kp_1",
        run_name="track_1",
        output_shard_rows=5,
        shard_workers=2,
        writer_arguments=("--smooth-seconds", "0.1"),
        copy_backend="python",
        apply=True,
        keep_scratch=True,
    )

    assert result["status"] == "complete"
    assert result["publish"]["pre_pointer_validation"]["valid"] is True
    assert result["publish"]["final_validation"]["valid"] is True
    assert result["publish"]["physical_copy"]["verification"] == (
        "sha256_all_physical_files"
    )
    assert events[0] == (
        "stage",
        source.resolve(),
        (scratch / "track-staging.zarr").resolve(),
        ("--smooth-seconds", "0.1"),
    )
    assert events[1] == (
        "bind",
        "analysis/track_kinematics_runs/offline/track_1",
        "running",
    )
    assert events[2:] == [
        ("validate_before", False, "running"),
        ("validate_before", True, "complete"),
        ("validate_public", True, "complete"),
    ]

    local_run = zarr.open_group(
        str(scratch / "track-staging.zarr/analysis/track_kinematics_runs/offline/track_1"),
        mode="r",
        use_consolidated=False,
    )
    assert local_run.attrs[mod.COORDINATE_BINDING_STATUS_ATTR] == (
        mod.UNBOUND_STAGE_STATUS
    )
    assert local_run.attrs["palette_run_completion_status"] == "complete"
    assert local_run.attrs["stage_selector_eligible"] is False
    assert "coordinate_descriptor" not in local_run["tracks/id_0/positions_px"].attrs

    sharded = zarr.open_group(
        str(scratch / "track-run-sharded"),
        mode="r",
        use_consolidated=False,
    )
    assert sharded.attrs[mod.COORDINATE_BINDING_STATUS_ATTR] == (
        mod.PUBLISHING_BINDING_STATUS
    )
    assert sharded.attrs["palette_run_completion_status"] == "running"
    assert sharded.attrs["stage_selector_eligible"] is False
    assert "coordinate_descriptor" not in sharded["tracks/id_0/positions_px"].attrs

    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    parent = root["analysis/track_kinematics_runs"]
    offline = parent["offline"]
    run = offline["track_1"]
    assert parent.attrs["latest"] == "offline/track_1"
    assert parent.attrs["latest_complete"] == "offline/track_1"
    assert parent.attrs["latest_offline"] == "track_1"
    assert offline.attrs["latest"] == "track_1"
    assert run.attrs[mod.COORDINATE_BINDING_STATUS_ATTR] == mod.BOUND_CANONICAL_STATUS
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert run.attrs["stage_selector_eligible"] is True
    assert tuple(run["tracks/id_0/speed_raw_px"].shards) == (6,)
    assert tuple(run["tracks/id_1/speed_raw_px"].shards) == (6,)
    assert run["tracks/id_0/source_frame_interpolation"].shards is None
    assert run["tracks/id_0/source_instance_key"].shards is None
    assert tuple(run["tracks/id_0/source_frame_interpolation"].chunks) == (11,)
    assert tuple(run["tracks/id_0/source_instance_key"].chunks) == (11,)
    layouts = run.attrs["physical_storage_layout"][
        "effective_overridden_array_layouts"
    ]
    assert layouts["tracks/id_0/source_frame_interpolation"]["layout_profile"] == (
        "structured_dtype_single_chunk_zarr_v3_sharding_codec_workaround_v1"
    )
    position = run["tracks/id_0/positions_px"]
    assert position.attrs["coordinate_descriptor"]["node_ref"] == f"/{position.path}"
    staging = run.attrs["cluster_output_staging"]
    assert staging["serialization_policy"] == "per_recording_advisory_file_lock"
    assert staging["canonical_binding"] == {
        "valid": True,
        "status": mod.BOUND_CANONICAL_STATUS,
        "track_count": 2,
        "binding_manifest_sha256": "d" * 64,
    }
    assert staging["materialization"]["stage_result"] == {
        "valid": True,
        "status": mod.UNBOUND_STAGE_STATUS,
        "run_name": "track_1",
        "track_count": 2,
        "staging_manifest_sha256": "c" * 64,
    }
    assert set(staging["parent_attrs_before"]) == {
        "analysis/track_kinematics_runs",
        "analysis/track_kinematics_runs/offline",
    }
    assert (tmp_path / ".source.zarr.track-kinematics-publish.lock").is_file()


def test_unbound_stage_rejects_detached_coordinate_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)
    events: list[tuple[Any, ...]] = []
    _install_writer_api(monkeypatch, events, detached_descriptor=True)

    with pytest.raises(
        RuntimeError,
        match="Local unbound numerical track stage is invalid",
    ):
        mod.materialize_track_kinematics(
            source,
            scratch_root=scratch,
            keypoint_run="refined/kp_1",
            run_name="track_1",
            copy_backend="python",
            apply=True,
        )

    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    assert "analysis" not in root
    assert not (scratch / "track-run-sharded").exists()


def test_final_path_binding_failure_rolls_back_target_and_pointers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)
    _seed_previous_pointers(source)
    events: list[tuple[Any, ...]] = []
    _install_writer_api(monkeypatch, events)

    def source_drift(*_args, **_kwargs):
        raise RuntimeError("source coordinate evidence changed before binding")

    monkeypatch.setattr(
        mod.track_writer,
        "bind_staged_offline_track_kinematics_run",
        source_drift,
    )

    with pytest.raises(RuntimeError, match="source coordinate evidence changed"):
        mod.materialize_track_kinematics(
            source,
            scratch_root=scratch,
            keypoint_run="refined/kp_1",
            run_name="track_1",
            output_shard_rows=5,
            copy_backend="python",
            apply=True,
        )

    _assert_previous_pointers_restored(source)


def test_post_bind_canonical_validation_failure_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)
    _seed_previous_pointers(source)
    events: list[tuple[Any, ...]] = []
    _install_writer_api(monkeypatch, events)

    def invalid_binding(*_args, require_complete, **_kwargs):
        assert require_complete is False
        return {"valid": False, "errors": ["wrong source row selection"]}

    monkeypatch.setattr(
        mod.track_writer,
        "_validate_bound_offline_track_kinematics_run_before_selection",
        invalid_binding,
    )

    with pytest.raises(RuntimeError, match="Published pre-pointer validation failed"):
        mod.materialize_track_kinematics(
            source,
            scratch_root=scratch,
            keypoint_run="refined/kp_1",
            run_name="track_1",
            output_shard_rows=5,
            copy_backend="python",
            apply=True,
        )

    _assert_previous_pointers_restored(source)


def test_completion_failure_after_pointer_update_rolls_back_both_parents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)
    _seed_previous_pointers(source)
    events: list[tuple[Any, ...]] = []
    _install_writer_api(monkeypatch, events)
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

    _assert_previous_pointers_restored(source)


@pytest.mark.parametrize(
    "argument",
    (
        "--output-zarr-path=/tmp/detached.zarr",
        "--offline-run-name=other",
        "--keypoint-run=other",
        "--no-write",
    ),
)
def test_plan_rejects_materializer_owned_legacy_writer_arguments(
    tmp_path: Path,
    argument: str,
) -> None:
    source = tmp_path / "source.zarr"
    _build_source(source)

    with pytest.raises(ValueError, match="materializer owns"):
        mod.build_track_kinematics_materialization_plan(
            source,
            scratch_root=tmp_path / "scratch",
            keypoint_run="refined/kp_1",
            run_name="track_1",
            writer_arguments=(argument,),
        )
