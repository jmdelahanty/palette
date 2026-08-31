from pathlib import Path
import json

import h5py
import pytest

from fisheye.shared.source_recording_identity import (
    SOURCE_RECORDING_ID_MAPPING_PROFILE,
    SOURCE_RECORDING_ID_MAPPING_PROFILE_ATTR,
    recording_id_from_session_camera,
)
from fisheye.utils import organize_recordings


def _write_h5(path: Path, *, camera: str, session_uuid: str, arena: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["session_uuid"] = session_uuid
        h5.attrs["session_start_iso8601_utc"] = "2026-05-29T18:11:16Z"
        h5.attrs["arena_id"] = arena
        h5.attrs["ipc_source_name"] = f"/shm_cam_{camera}"
        h5.attrs["protocol_name_from_definition"] = "GoodCopBadCop"


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _make_external_ipc_batch(tmp_path: Path) -> Path:
    batch = tmp_path / "2026_05_29_14_11_07"
    camera = "2010093"
    h5_path = batch / "citrus" / "2026-05-29T18-11-16Z_arena_1_GoodCopBadCop.h5"
    _write_h5(
        h5_path,
        camera=camera,
        session_uuid="2026-05-29T18-11-16Z_arena_1",
        arena="arena_1",
    )
    _touch(h5_path.with_suffix(".mp4"), "stimulus")
    _touch(batch / "recording_snapshot.json", "{}")
    _touch(batch / "recording_session.json", "{}")
    _touch(batch / "ptp_sync_summary.json", "{}")
    _touch(batch / "_citrus_transfer_complete.json", "{}")
    _touch(batch / "orange_local_control.events.jsonl", "{}\n")
    _touch(batch / "citrus" / "2026-05-29T18-11-16Z_threading_startup.json", "{}")

    full_dir = batch / "external_recorder"
    crop_dir = batch / "external_crop_recorder"
    _touch(full_dir / f"Cam{camera}_external.mp4", "full")
    _touch(full_dir / f"Cam{camera}_external_summary.json", "{}")
    _touch(full_dir / f"Cam{camera}_external_keyframes.json", "{}")
    _touch(full_dir / f"Cam{camera}_external_detach.csv", "frame_index\n0\n")
    _touch(full_dir / f"Cam{camera}_external_gop_routing.csv", "gop\n0\n")
    _touch(full_dir / f"Cam{camera}_external_shard0_gpu3.mp4", "ignore")
    _touch(full_dir / f"Cam{camera}_external_encode_shard0_gpu3.csv", "ignore\n")
    _touch(full_dir / "external_recorder_finalization.json", "{}")

    _touch(crop_dir / f"Cam{camera}_crop_external.mp4", "crop")
    _touch(crop_dir / f"Cam{camera}_crop_external_summary.json", "{}")
    _touch(crop_dir / f"Cam{camera}_crop_external_keyframe.json", "{}")
    _touch(crop_dir / f"Cam{camera}_crop_external_detach.csv", "frame_index\n0\n")
    _touch(crop_dir / f"Cam{camera}_crop_external_encode.csv", "frame_index\n0\n")
    _touch(crop_dir / f"Cam{camera}_crop_external_gop_routing.csv", "gop\n0\n")
    _touch(
        batch / f"Cam{camera}_crop_meta.csv",
        "recording_frame_id,timestamp,timestamp_sys\n1,10,20\n",
    )
    _touch(batch / f"Cam{camera}_crop_perf.csv", "metric,value\n")
    _touch(batch / f"Cam{camera}_crop_sidecar_perf.csv", "metric,value\n")
    _touch(batch / f"Cam{camera}_yolo_perf.csv", "metric,value\n")
    _touch(batch / f"Cam{camera}_yolo_events.jsonl", "{}\n")

    stale_root = Path("/home/jeremy/orange_data/exp/unsorted") / batch.name
    session = {
        "schema_version": 1,
        "session_id": "2026-05-29T18-11-16Z_arena_1",
        "status": "completed",
        "mode": "single_clip",
        "producer": "orange_gui_external_ipc",
        "recording_backend": "external_ipc",
        "recording_outputs": {
            camera: {
                "full": {
                    "backend": "external_ipc",
                    "role": "ingest_authoritative",
                    "output_kind": "full",
                    "video": str(
                        stale_root / "external_recorder" / f"Cam{camera}_external.mp4"
                    ),
                    "metadata": str(
                        stale_root
                        / "external_recorder"
                        / f"Cam{camera}_external_summary.json"
                    ),
                    "keyframes": str(
                        stale_root
                        / "external_recorder"
                        / f"Cam{camera}_external_keyframes.json"
                    ),
                    "frame_count": 1,
                    "coordinate_space": "full_frame_pixels",
                },
                "crop": {
                    "backend": "external_ipc",
                    "role": "sidecar",
                    "output_kind": "crop",
                    "video": str(
                        stale_root
                        / "external_crop_recorder"
                        / f"Cam{camera}_crop_external.mp4"
                    ),
                    "metadata": f"Cam{camera}_crop_meta.csv",
                    "summary": str(
                        stale_root
                        / "external_crop_recorder"
                        / f"Cam{camera}_crop_external_summary.json"
                    ),
                    "keyframes": str(
                        stale_root
                        / "external_crop_recorder"
                        / f"Cam{camera}_crop_external_keyframe.json"
                    ),
                    "perf": f"Cam{camera}_crop_perf.csv",
                    "sidecar_perf": f"Cam{camera}_crop_sidecar_perf.csv",
                    "frame_count": 1,
                    "width": 256,
                    "height": 256,
                    "encoded_format": "nv12",
                    "pixel_source_format": "mono8",
                    "details": {
                        "stream_id": f"{camera}_crop",
                        "blank_frame_policy": "encode_black_frame_when_no_detection",
                        "selection_policy": "largest_detection_by_confidence",
                    },
                },
            }
        },
    }
    (batch / "recording_session.json").write_text(json.dumps(session), encoding="utf-8")
    return batch


def test_external_ipc_plan_maps_full_and_crop_outputs_without_shards(
    tmp_path: Path,
) -> None:
    batch = _make_external_ipc_batch(tmp_path)
    plans = organize_recordings._build_external_ipc_plans(
        batch,
        dest_root=tmp_path / "recordings",
        rename_cams=True,
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.missing == []
    assert plan.camera_id == "2010093"
    assert plan.meta["artifact_schema_id"] == "orange_external_ipc_single_clip_v1"
    assert plan.meta["recording_backend"] == "external_ipc"
    assert plan.meta["recording_id"] == recording_id_from_session_camera(
        session_uuid="2026-05-29T18-11-16Z_arena_1",
        camera_id="2010093",
    )
    assert plan.meta["recording_id"] != plan.name
    assert plan.meta[SOURCE_RECORDING_ID_MAPPING_PROFILE_ATTR] == (
        SOURCE_RECORDING_ID_MAPPING_PROFILE
    )

    raw_names = [item.dest_name for item in plan.raw_files]
    assert "transfer_complete.json" in raw_names
    assert "orange_local_control.events.jsonl" in raw_names

    cam_names = [item.dest_name for item in plan.cam_files]
    assert cam_names == [
        "Cam2010093_2026-05-29T18-11-16Z_arena_1.mp4",
        "Cam2010093_2026-05-29T18-11-16Z_arena_1_meta.csv",
        "Cam2010093_2026-05-29T18-11-16Z_arena_1_keyframe.json",
        "Cam2010093_2026-05-29T18-11-16Z_arena_1_external_summary.json",
    ]

    derived_names = [item.dest_name for item in plan.derived_files]
    assert (
        "external_crop_recorder/Cam2010093_2026-05-29T18-11-16Z_arena_1_crop_external.mp4"
        in derived_names
    )
    assert (
        "external_crop_recorder/Cam2010093_2026-05-29T18-11-16Z_arena_1_crop_meta.csv"
        in derived_names
    )
    assert (
        "external_recorder/Cam2010093_2026-05-29T18-11-16Z_arena_1_external_detach.csv"
        in derived_names
    )
    assert "citrus/2026-05-29T18-11-16Z_threading_startup.json" in derived_names
    assert all(
        "shard" not in item.source.name for item in plan.cam_files + plan.derived_files
    )
    assert all(
        "shard" not in item.dest_name for item in plan.cam_files + plan.derived_files
    )

    video_streams = plan.meta["video_streams"]
    assert video_streams["schema_id"] == "orange_runtime_video_streams_v1"
    assert video_streams["streams"]["full"]["role"] == "ingest_authoritative_full_frame"
    assert video_streams["streams"]["full"]["video"] == (
        "cams/Cam2010093_2026-05-29T18-11-16Z_arena_1.mp4"
    )
    assert video_streams["streams"]["full"]["frame_clock_metadata"] == (
        "cams/Cam2010093_2026-05-29T18-11-16Z_arena_1_meta.csv"
    )
    assert (
        video_streams["streams"]["crop"]["role"] == "runtime_derived_acquisition_input"
    )
    assert (
        video_streams["streams"]["crop"]["video_pixel_coordinate_space"]
        == "crop_frame_pixels"
    )
    assert (
        video_streams["streams"]["crop"]["blank_frame_policy"]
        == "encode_black_frame_when_no_detection"
    )


def test_external_ipc_plan_rejects_h5_and_session_identity_disagreement(
    tmp_path: Path,
) -> None:
    batch = _make_external_ipc_batch(tmp_path)
    session_path = batch / "recording_session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session["session_id"] = "different_session"
    session_path.write_text(json.dumps(session), encoding="utf-8")

    with pytest.raises(ValueError, match="must exactly match"):
        organize_recordings._build_external_ipc_plans(
            batch,
            dest_root=tmp_path / "recordings",
            rename_cams=True,
        )


def test_external_ipc_plan_accepts_missing_h5_session_identity(
    tmp_path: Path,
) -> None:
    batch = _make_external_ipc_batch(tmp_path)
    [h5_path] = list(batch.rglob("*.h5"))
    with h5py.File(h5_path, "r+") as h5:
        del h5.attrs["session_uuid"]

    [plan] = organize_recordings._build_external_ipc_plans(
        batch,
        dest_root=tmp_path / "recordings",
        rename_cams=True,
    )

    assert plan.meta["session_uuid"] == "2026-05-29T18-11-16Z_arena_1"
    assert plan.meta["recording_id"] == recording_id_from_session_camera(
        session_uuid="2026-05-29T18-11-16Z_arena_1",
        camera_id="2010093",
    )


def test_external_ipc_plan_keeps_full_metadata_summary_and_status_distinct(
    tmp_path: Path,
) -> None:
    batch = _make_external_ipc_batch(tmp_path)
    camera = "2010093"
    full_dir = batch / "external_recorder"
    full_metadata = _touch(
        full_dir / f"Cam{camera}_external_meta.csv",
        "frame_id,recording_frame_id\n1,1\n",
    )
    _touch(
        full_dir / f"Cam{camera}_external_status.json",
        json.dumps(
            {
                "status": "completed",
                "frames_received": 1,
                "frames_encoded": 1,
                "frames_dropped": 0,
            }
        ),
    )
    session_path = batch / "recording_session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session["recording_outputs"][camera]["full"]["metadata"] = str(full_metadata)
    session_path.write_text(json.dumps(session), encoding="utf-8")

    [plan] = organize_recordings._build_external_ipc_plans(
        batch,
        dest_root=tmp_path / "recordings",
        rename_cams=True,
    )

    cam_base = "Cam2010093_2026-05-29T18-11-16Z_arena_1"
    cam_sources = {item.dest_name: item.source for item in plan.cam_files}
    assert cam_sources[f"{cam_base}_external_meta.csv"] == full_metadata.resolve()
    assert (
        cam_sources[f"{cam_base}_external_summary.json"]
        == (full_dir / f"Cam{camera}_external_summary.json").resolve()
    )
    full = plan.meta["video_streams"]["streams"]["full"]
    assert full["frame_clock_metadata"] == f"cams/{cam_base}_external_meta.csv"
    assert full["summary"] == f"cams/{cam_base}_external_summary.json"
    assert full["status"] == (
        f"derived/external_recorder/{cam_base}_external_status.json"
    )


def test_external_ipc_recording_only_plan_maps_full_and_crop_outputs(
    tmp_path: Path,
) -> None:
    batch = _make_external_ipc_batch(tmp_path)
    for h5_path in batch.rglob("*.h5"):
        h5_path.unlink()

    plans = organize_recordings._build_external_ipc_recording_only_plans(
        batch,
        dest_root=tmp_path / "recordings",
        rename_cams=True,
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.name == "2026-05-29T18-11-16Z_arena_1_Cam2010093"
    assert plan.missing == []
    assert plan.camera_id == "2010093"
    assert plan.meta["artifact_schema_id"] == "orange_external_ipc_video_only_v1"
    assert plan.meta["recording_backend"] == "external_ipc"
    assert plan.meta["recording_id"] == recording_id_from_session_camera(
        session_uuid="2026-05-29T18-11-16Z_arena_1",
        camera_id="2010093",
    )

    raw_names = [item.dest_name for item in plan.raw_files]
    assert "recording_session.json" in raw_names
    assert "transfer_complete.json" in raw_names
    assert not any(name.endswith(".h5") for name in raw_names)

    cam_names = [item.dest_name for item in plan.cam_files]
    assert cam_names == [
        "Cam2010093_2026-05-29T18-11-16Z_arena_1.mp4",
        "Cam2010093_2026-05-29T18-11-16Z_arena_1_meta.csv",
        "Cam2010093_2026-05-29T18-11-16Z_arena_1_keyframe.json",
        "Cam2010093_2026-05-29T18-11-16Z_arena_1_external_summary.json",
    ]

    derived_names = [item.dest_name for item in plan.derived_files]
    assert (
        "external_crop_recorder/Cam2010093_2026-05-29T18-11-16Z_arena_1_crop_external.mp4"
        in derived_names
    )
    assert (
        "external_crop_recorder/Cam2010093_2026-05-29T18-11-16Z_arena_1_crop_meta.csv"
        in derived_names
    )

    video_streams = plan.meta["video_streams"]
    assert video_streams["streams"]["full"]["frame_clock_metadata"] == (
        "cams/Cam2010093_2026-05-29T18-11-16Z_arena_1_meta.csv"
    )
    assert (
        video_streams["streams"]["crop"]["video_pixel_coordinate_space"]
        == "crop_frame_pixels"
    )


def test_external_ipc_recording_only_rolling_clips_fail_closed(tmp_path: Path) -> None:
    batch = _make_external_ipc_batch(tmp_path)
    session_path = batch / "recording_session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    outputs = session.pop("recording_outputs")
    session["mode"] = "rolling_clips"
    session["clips"] = [
        {"clip_id": "clip_000000", "clip_index": 0, "recording_outputs": outputs},
        {"clip_id": "clip_000001", "clip_index": 1, "recording_outputs": outputs},
    ]
    session_path.write_text(json.dumps(session), encoding="utf-8")

    with pytest.raises(ValueError, match="one parent recording per camera stream"):
        organize_recordings._build_external_ipc_recording_only_plans(
            batch,
            dest_root=tmp_path / "recordings",
            rename_cams=True,
        )


def test_external_ipc_clip_start_never_uses_finalize_or_session_end() -> None:
    session = {
        "created_at_utc": "2026-08-06T23:13:38Z",
        "updated_at_utc": "2026-08-08T02:25:38Z",
        "stream": {"finished_at_utc": "2026-08-08T02:25:38Z"},
    }
    clip = {"finalized_at_utc": "2026-08-08T02:25:38Z"}

    assert (
        organize_recordings._external_ipc_clip_start_utc(session, clip)
        == "2026-08-06T23:13:38Z"
    )


def test_external_ipc_recording_only_full_video_without_crop_meta_is_valid(
    tmp_path: Path,
) -> None:
    batch = _make_external_ipc_batch(tmp_path)
    session_path = batch / "recording_session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    del session["recording_outputs"]["2010093"]["crop"]
    session_path.write_text(json.dumps(session), encoding="utf-8")

    plans = organize_recordings._build_external_ipc_recording_only_plans(
        batch,
        dest_root=tmp_path / "recordings",
        rename_cams=True,
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.missing == []
    assert [item.dest_name for item in plan.cam_files] == [
        "Cam2010093_2026-05-29T18-11-16Z_arena_1.mp4",
        "Cam2010093_2026-05-29T18-11-16Z_arena_1_keyframe.json",
        "Cam2010093_2026-05-29T18-11-16Z_arena_1_external_summary.json",
    ]
    assert "frame_clock_metadata" not in plan.meta["video_streams"]["streams"]["full"]
    assert "crop" not in plan.meta["video_streams"]["streams"]


def test_external_ipc_h5_plan_does_not_declare_missing_compat_clock_csv(
    tmp_path: Path,
) -> None:
    batch = _make_external_ipc_batch(tmp_path)
    (batch / "Cam2010093_crop_meta.csv").unlink()

    plans = organize_recordings._build_external_ipc_plans(
        batch,
        dest_root=tmp_path / "recordings",
        rename_cams=True,
    )

    assert len(plans) == 1
    plan = plans[0]
    assert "Cam2010093_crop_meta.csv (compatibility camera metadata)" in plan.missing
    assert not any(item.dest_name.endswith("_meta.csv") for item in plan.cam_files)
    assert "frame_clock_metadata" not in plan.meta["video_streams"]["streams"]["full"]


def test_external_ipc_apply_writes_nested_sidecars_and_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    batch = _make_external_ipc_batch(tmp_path)
    plans = organize_recordings._build_external_ipc_plans(
        batch,
        dest_root=tmp_path / "recordings",
        rename_cams=True,
    )
    monkeypatch.setattr(
        organize_recordings,
        "check_hevc_keyframe_flags",
        lambda _, **__: {
            "codec": "hevc",
            "has_stss": True,
            "needs_fix": False,
            "message": "ok",
        },
    )

    warnings = organize_recordings._apply_plan(
        plans,
        create_empty=False,
        write_manifest=True,
        snapshot={},
        snapshot_mode="copy",
        logger=None,
        run_id="test_run",
        log_path=None,
    )

    assert warnings == []
    dest = tmp_path / "recordings" / "2026-05-29T18-11-16Z_arena_1_GoodCopBadCop"
    assert (dest / "cams" / "Cam2010093_2026-05-29T18-11-16Z_arena_1.mp4").exists()
    assert (dest / "cams" / "Cam2010093_2026-05-29T18-11-16Z_arena_1_meta.csv").exists()
    assert (dest / "raw" / "transfer_complete.json").exists()
    assert (dest / "raw" / "orange_local_control.events.jsonl").exists()
    assert (
        dest
        / "derived"
        / "external_crop_recorder"
        / "Cam2010093_2026-05-29T18-11-16Z_arena_1_crop_meta.csv"
    ).exists()
    assert (
        dest / "derived" / "citrus" / "2026-05-29T18-11-16Z_threading_startup.json"
    ).exists()

    manifest = json.loads(
        (dest / "recording_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifact_schema_id"] == "orange_external_ipc_single_clip_v1"
    assert manifest["recording_backend"] == "external_ipc"
    assert manifest["orange_session_id"] == "2026-05-29T18-11-16Z_arena_1"
    assert manifest[SOURCE_RECORDING_ID_MAPPING_PROFILE_ATTR] == (
        SOURCE_RECORDING_ID_MAPPING_PROFILE
    )
    assert (
        "cams/Cam2010093_2026-05-29T18-11-16Z_arena_1_meta.csv"
        in manifest["files"]["cams"]
    )
    assert (
        "derived/external_crop_recorder/Cam2010093_2026-05-29T18-11-16Z_arena_1_crop_external.mp4"
        in manifest["files"]["derived"]
    )
    assert manifest["video_streams"]["streams"]["crop"]["stream_id"] == "2010093_crop"


def test_external_ipc_plan_lifts_manifest_context_from_h5_and_runtime_snapshot(
    tmp_path: Path,
) -> None:
    batch_root = tmp_path / "staging" / "2026_06_14_17_11_56"
    citrus_root = batch_root / "citrus"
    citrus_root.mkdir(parents=True)

    h5_path = citrus_root / "2026-06-14T21-12-08Z_arena_1_GoodCopBadCop.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["session_uuid"] = "2026-06-14T21-12-08Z_arena_1"
        h5.attrs["session_start_iso8601_utc"] = "2026-06-14T21-12-08Z"
        h5.attrs["rig_id"] = "omnifin0"
        h5.attrs["arena_id"] = "arena_1"
        h5.attrs["canvas_name"] = "shadow"
        h5.attrs["ipc_source_name"] = "/shm_cam_2010093"
        h5.attrs["software_version"] = ""

        protocol = h5.create_group("protocol_snapshot")
        protocol.attrs["protocol_name"] = "GoodCopBadCop"
        protocol.create_dataset(
            "protocol_definition_json",
            data=json.dumps({"protocol_name": "GoodCopBadCop"}).encode("utf-8"),
        )

        subjects = h5.create_group("subject_metadata")
        subjects.attrs["genotype"] = "EXPERIMENTAL"
        subjects.attrs["days_post_fertilization"] = 9
        subjects.attrs["fish_count"] = 8
        subjects.attrs["subject_count"] = 1

        calibration = h5.create_group("calibration_snapshot")
        calibration.create_dataset(
            "arena_config_json",
            data=json.dumps(
                {
                    "selected_dish_type_name": "palm1",
                    "dish_config": {"dish_name": "palm1"},
                }
            ).encode("utf-8"),
        )

    (batch_root / "recording_snapshot.json").write_text(
        json.dumps(
            {
                "producer_version": "951196b",
                "source_version": {
                    "describe": "v1.2.1-1444-g951196b-dirty",
                    "commit_short": "951196b",
                },
            }
        ),
        encoding="utf-8",
    )
    session = {
        "session_id": "2026-06-14T21-12-08Z_arena_1",
        "producer": "orange_gui_external_ipc",
        "mode": "single_clip",
        "recording_outputs": {},
    }

    plan = organize_recordings._build_external_ipc_plan(
        h5_path,
        batch_root=batch_root,
        session=session,
        dest_root=tmp_path / "recordings",
        rename_cams=True,
    )

    assert plan.camera_id == "2010093"
    assert plan.meta["protocol_name"] == "GoodCopBadCop"
    assert plan.meta["protocol_name_from_definition"] == "GoodCopBadCop"
    assert plan.meta["dish_design"] == "palm1"
    assert plan.meta["genotype"] == "EXPERIMENTAL"
    assert plan.meta["dpf_at_acquisition"] == "9"
    assert plan.meta["software_version"] == "v1.2.1-1444-g951196b-dirty"
    assert "fish_per_dish" not in plan.meta
    assert "num_dishes" not in plan.meta
