from pathlib import Path
import json

import h5py

from fisheye.utils import refresh_recording_manifest_metadata as refresh


def _write_external_ipc_recording(recording_dir: Path, *, protocol_name: str = "GoodCopBadCop") -> Path:
    raw_dir = recording_dir / "raw"
    raw_dir.mkdir(parents=True)
    h5_path = raw_dir / f"{recording_dir.name}.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["session_uuid"] = "2026-06-14T21-12-08Z_arena_1"
        h5.attrs["ipc_source_name"] = "/shm_cam_2010093"
        h5.attrs["software_version"] = ""

        protocol = h5.create_group("protocol_snapshot")
        protocol.attrs["protocol_name"] = protocol_name
        protocol.create_dataset(
            "protocol_definition_json",
            data=json.dumps({"protocol_name": protocol_name}).encode("utf-8"),
        )

        subjects = h5.create_group("subject_metadata")
        subjects.attrs["genotype"] = "EXPERIMENTAL"
        subjects.attrs["days_post_fertilization"] = 9

        calibration = h5.create_group("calibration_snapshot")
        calibration.create_dataset(
            "arena_config_json",
            data=json.dumps({"selected_dish_type_name": "palm1"}).encode("utf-8"),
        )

    (raw_dir / "recording_snapshot_runtime.json").write_text(
        json.dumps(
            {
                "producer_version": "951196b",
                "source_version": {"describe": "v1.2.1-1444-g951196b-dirty"},
            }
        ),
        encoding="utf-8",
    )
    return h5_path


def _write_manifest(recording_dir: Path, payload: dict) -> Path:
    manifest_path = recording_dir / "recording_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def test_refresh_manifest_metadata_apply_fills_empty_derivable_fields(tmp_path: Path) -> None:
    recording_dir = tmp_path / "2026-06-14T21-12-08Z_arena_1_GoodCopBadCop"
    h5_path = _write_external_ipc_recording(recording_dir)
    manifest_path = _write_manifest(
        recording_dir,
        {
            "recording_name": recording_dir.name,
            "files": {"raw": [f"raw/{h5_path.name}"]},
            "protocol_name": None,
            "dish_design": None,
            "genotype": None,
            "dpf_at_acquisition": None,
            "software_version": None,
            "num_dishes": None,
            "fish_per_dish": None,
        },
    )

    rc = refresh.main([str(recording_dir), "--apply"])

    assert rc == 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["protocol_name"] == "GoodCopBadCop"
    assert payload["protocol_name_from_definition"] == "GoodCopBadCop"
    assert payload["dish_design"] == "palm1"
    assert payload["genotype"] == "EXPERIMENTAL"
    assert payload["dpf_at_acquisition"] == "9"
    assert payload["software_version"] == "v1.2.1-1444-g951196b-dirty"
    assert payload["num_dishes"] is None
    assert payload["fish_per_dish"] is None


def test_refresh_manifest_metadata_dry_run_does_not_mutate(tmp_path: Path) -> None:
    recording_dir = tmp_path / "recording"
    h5_path = _write_external_ipc_recording(recording_dir)
    manifest_path = _write_manifest(
        recording_dir,
        {
            "recording_name": recording_dir.name,
            "files": {"raw": [f"raw/{h5_path.name}"]},
            "protocol_name": None,
        },
    )
    before = manifest_path.read_text(encoding="utf-8")

    rc = refresh.main([str(recording_dir)])

    assert rc == 0
    assert manifest_path.read_text(encoding="utf-8") == before


def test_refresh_manifest_metadata_preserves_non_empty_values_by_default(tmp_path: Path) -> None:
    recording_dir = tmp_path / "recording"
    h5_path = _write_external_ipc_recording(recording_dir, protocol_name="H5Protocol")
    manifest_path = _write_manifest(
        recording_dir,
        {
            "recording_name": recording_dir.name,
            "files": {"raw": [f"raw/{h5_path.name}"]},
            "protocol_name": "OperatorProtocol",
        },
    )

    rc = refresh.main([str(recording_dir), "--apply"])

    assert rc == 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["protocol_name"] == "OperatorProtocol"
    assert payload["protocol_name_from_definition"] == "H5Protocol"


def test_refresh_manifest_metadata_overwrite_existing_replaces_different_values(tmp_path: Path) -> None:
    recording_dir = tmp_path / "recording"
    h5_path = _write_external_ipc_recording(recording_dir, protocol_name="H5Protocol")
    manifest_path = _write_manifest(
        recording_dir,
        {
            "recording_name": recording_dir.name,
            "files": {"raw": [f"raw/{h5_path.name}"]},
            "protocol_name": "OperatorProtocol",
        },
    )

    rc = refresh.main([str(recording_dir), "--apply", "--overwrite-existing"])

    assert rc == 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["protocol_name"] == "H5Protocol"


def test_refresh_manifest_metadata_treats_unknown_software_version_as_missing(tmp_path: Path) -> None:
    recording_dir = tmp_path / "recording"
    raw_dir = recording_dir / "raw"
    raw_dir.mkdir(parents=True)
    h5_path = raw_dir / "recording.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["ipc_source_name"] = "/shm_cam_2010093"
        h5.attrs["software_version"] = "unknown"
    manifest_path = _write_manifest(
        recording_dir,
        {
            "recording_name": recording_dir.name,
            "files": {"raw": [f"raw/{h5_path.name}"]},
            "software_version": "unknown",
        },
    )

    rc = refresh.main([str(recording_dir), "--apply"])

    assert rc == 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["software_version"] is None


def test_refresh_external_ipc_artifacts_copies_context_and_adds_video_streams(
    tmp_path: Path,
) -> None:
    recording_dir = tmp_path / "2026-06-14T21-12-08Z_arena_1_GoodCopBadCop"
    h5_path = _write_external_ipc_recording(recording_dir)
    (recording_dir / "cams").mkdir()
    (recording_dir / "derived" / "external_crop_recorder").mkdir(parents=True)
    (recording_dir / "cams" / "Cam2010093_2026-06-14T21-12-08Z_arena_1.mp4").write_text(
        "full",
        encoding="utf-8",
    )
    (
        recording_dir
        / "cams"
        / "Cam2010093_2026-06-14T21-12-08Z_arena_1_external_meta.csv"
    ).write_text("frame_id,recording_frame_id\n0,0\n", encoding="utf-8")
    (
        recording_dir
        / "cams"
        / "Cam2010093_2026-06-14T21-12-08Z_arena_1_external_summary.json"
    ).write_text("{}", encoding="utf-8")
    (recording_dir / "derived" / "external_recorder").mkdir(parents=True)
    (
        recording_dir
        / "derived"
        / "external_recorder"
        / "Cam2010093_2026-06-14T21-12-08Z_arena_1_external_status.json"
    ).write_text('{"status":"completed"}', encoding="utf-8")
    (
        recording_dir
        / "derived"
        / "external_crop_recorder"
        / "Cam2010093_2026-06-14T21-12-08Z_arena_1_crop_external.mp4"
    ).write_text("crop", encoding="utf-8")

    source_dir = tmp_path / "staging" / "2026_06_14_17_11_56"
    (source_dir / "citrus").mkdir(parents=True)
    (source_dir / "_citrus_transfer_complete.json").write_text("{}", encoding="utf-8")
    (source_dir / "orange_local_control.events.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    (source_dir / "citrus" / "startup_threading_startup_1.json").write_text(
        "{}",
        encoding="utf-8",
    )

    session = {
        "session_id": "2026_06_14_17_11_56",
        "producer": "orange_gui_external_ipc",
        "recording_backend": "external_ipc",
        "recording_outputs": {
            "2010093": {
                "full": {
                    "role": "ingest_authoritative",
                    "output_kind": "full",
                    "frame_count": 10,
                    "coordinate_space": "full_frame_pixels",
                },
                "crop": {
                    "role": "sidecar",
                    "output_kind": "crop",
                    "frame_count": 10,
                    "width": 256,
                    "height": 256,
                    "details": {
                        "blank_frame_policy": "encode_black_frame_when_no_detection",
                        "selection_policy": "largest_detection_by_confidence",
                    },
                },
            }
        },
    }
    (recording_dir / "raw" / "recording_session.json").write_text(
        json.dumps(session),
        encoding="utf-8",
    )
    manifest_path = _write_manifest(
        recording_dir,
        {
            "recording_name": recording_dir.name,
            "recording_backend": "external_ipc",
            "source_dir": str(source_dir),
            "camera_id": "2010093",
            "files": {
                "raw": [f"raw/{h5_path.name}", "raw/recording_session.json"],
                "cams": ["cams/Cam2010093_2026-06-14T21-12-08Z_arena_1.mp4"],
                "derived": [
                    (
                        "derived/external_crop_recorder/"
                        "Cam2010093_2026-06-14T21-12-08Z_arena_1_crop_external.mp4"
                    )
                ],
            },
        },
    )

    rc = refresh.main(
        [
            str(recording_dir),
            "--refresh-external-ipc-artifacts",
            "--apply",
        ]
    )

    assert rc == 0
    assert (recording_dir / "raw" / "transfer_complete.json").exists()
    assert (recording_dir / "raw" / "orange_local_control.events.jsonl").exists()
    assert (
        recording_dir / "derived" / "citrus" / "startup_threading_startup_1.json"
    ).exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "raw/transfer_complete.json" in payload["files"]["raw"]
    assert "raw/orange_local_control.events.jsonl" in payload["files"]["raw"]
    assert (
        "derived/citrus/startup_threading_startup_1.json" in payload["files"]["derived"]
    )
    assert payload["video_streams"]["schema_id"] == "orange_runtime_video_streams_v1"
    full_stream = payload["video_streams"]["streams"]["full"]
    assert full_stream["frame_clock_metadata"].endswith("_external_meta.csv")
    assert full_stream["summary"].endswith("_external_summary.json")
    assert full_stream["status"].endswith("_external_status.json")
    crop_stream = payload["video_streams"]["streams"]["crop"]
    assert crop_stream["role"] == "runtime_derived_acquisition_input"
    assert crop_stream["video_pixel_coordinate_space"] == "crop_frame_pixels"
    assert crop_stream["blank_frame_policy"] == "encode_black_frame_when_no_detection"
