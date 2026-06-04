from pathlib import Path
import json

import h5py

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
    _touch(batch / f"Cam{camera}_crop_meta.csv", "recording_frame_id,timestamp,timestamp_sys\n1,10,20\n")
    _touch(batch / f"Cam{camera}_crop_perf.csv", "metric,value\n")
    _touch(batch / f"Cam{camera}_crop_sidecar_perf.csv", "metric,value\n")
    _touch(batch / f"Cam{camera}_yolo_perf.csv", "metric,value\n")
    _touch(batch / f"Cam{camera}_yolo_events.jsonl", "{}\n")

    stale_root = Path("/home/jeremy/orange_data/exp/unsorted") / batch.name
    session = {
        "schema_version": 1,
        "session_id": batch.name,
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
                    "video": str(stale_root / "external_recorder" / f"Cam{camera}_external.mp4"),
                    "metadata": str(stale_root / "external_recorder" / f"Cam{camera}_external_summary.json"),
                    "keyframes": str(stale_root / "external_recorder" / f"Cam{camera}_external_keyframes.json"),
                    "frame_count": 1,
                },
                "crop": {
                    "backend": "external_ipc",
                    "role": "sidecar",
                    "output_kind": "crop",
                    "video": str(stale_root / "external_crop_recorder" / f"Cam{camera}_crop_external.mp4"),
                    "metadata": f"Cam{camera}_crop_meta.csv",
                    "summary": str(stale_root / "external_crop_recorder" / f"Cam{camera}_crop_external_summary.json"),
                    "keyframes": str(stale_root / "external_crop_recorder" / f"Cam{camera}_crop_external_keyframe.json"),
                    "perf": f"Cam{camera}_crop_perf.csv",
                    "sidecar_perf": f"Cam{camera}_crop_sidecar_perf.csv",
                    "frame_count": 1,
                },
            }
        },
    }
    (batch / "recording_session.json").write_text(json.dumps(session), encoding="utf-8")
    return batch


def test_external_ipc_plan_maps_full_and_crop_outputs_without_shards(tmp_path: Path) -> None:
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

    cam_names = [item.dest_name for item in plan.cam_files]
    assert cam_names == [
        "Cam2010093_2026-05-29T18-11-16Z_arena_1.mp4",
        "Cam2010093_2026-05-29T18-11-16Z_arena_1_meta.csv",
        "Cam2010093_2026-05-29T18-11-16Z_arena_1_keyframe.json",
        "Cam2010093_2026-05-29T18-11-16Z_arena_1_external_summary.json",
    ]

    derived_names = [item.dest_name for item in plan.derived_files]
    assert "external_crop_recorder/Cam2010093_2026-05-29T18-11-16Z_arena_1_crop_external.mp4" in derived_names
    assert "external_crop_recorder/Cam2010093_2026-05-29T18-11-16Z_arena_1_crop_meta.csv" in derived_names
    assert "external_recorder/Cam2010093_2026-05-29T18-11-16Z_arena_1_external_detach.csv" in derived_names
    assert all("shard" not in item.source.name for item in plan.cam_files + plan.derived_files)
    assert all("shard" not in item.dest_name for item in plan.cam_files + plan.derived_files)


def test_external_ipc_apply_writes_nested_sidecars_and_manifest(tmp_path: Path, monkeypatch) -> None:
    batch = _make_external_ipc_batch(tmp_path)
    plans = organize_recordings._build_external_ipc_plans(
        batch,
        dest_root=tmp_path / "recordings",
        rename_cams=True,
    )
    monkeypatch.setattr(
        organize_recordings,
        "check_hevc_keyframe_flags",
        lambda _: {
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
    assert (
        dest
        / "derived"
        / "external_crop_recorder"
        / "Cam2010093_2026-05-29T18-11-16Z_arena_1_crop_meta.csv"
    ).exists()

    manifest = json.loads((dest / "recording_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_schema_id"] == "orange_external_ipc_single_clip_v1"
    assert manifest["recording_backend"] == "external_ipc"
    assert manifest["orange_session_id"] == "2026_05_29_14_11_07"
    assert "cams/Cam2010093_2026-05-29T18-11-16Z_arena_1_meta.csv" in manifest["files"]["cams"]
    assert (
        "derived/external_crop_recorder/Cam2010093_2026-05-29T18-11-16Z_arena_1_crop_external.mp4"
        in manifest["files"]["derived"]
    )
