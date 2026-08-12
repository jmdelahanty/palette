from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from fisheye.shared import stimulus_coordinate_contract as contract
from fisheye.utils import migrate_legacy_goodbatbadbat_stimulus_h5 as migration


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _chaser_dtype() -> np.dtype:
    integer_types = {
        "stimulus_frame_num": "<u8",
        "timestamp_ns_session": "<i8",
        "chaser_index": "u1",
        "is_chasing": "u1",
        "target_source_frame_id": "<u8",
        "target_source_camera_id": "<u4",
        "target_source_box_index_in_payload": "u1",
        "target_freshness_state": "u1",
        "target_area_state": "u1",
        "behavior_program_active": "u1",
        "behavior_episode_active": "u1",
        "behavior_episode_id": "<u8",
        "behavior_phase_index": "<i4",
        "behavior_motion_type_id": "u1",
        "behavior_retreat_plan_active": "u1",
        "loom_mode": "u1",
        "loom_phase": "u1",
        "chaser_behavior_class_id": "u1",
        "trial_state": "u1",
        "chase_sequence_active": "u1",
        "chase_trial_id": "<u8",
    }
    return np.dtype(
        [
            (name, integer_types.get(name, "<f4"))
            for name in migration._CHASER_FIELDS
        ]
    )


def _legacy_attrs(camera_id: str) -> dict[str, object]:
    attrs: dict[str, object] = {
        name: "fixture" for name in migration._LEGACY_CHASER_ATTRS
    }
    attrs.update(
        {
            "camera_id": camera_id,
            "coordinate_frame": "arena_relative_canvas_px",
            "coordinate_origin": "top_left_of_active_arena",
            "physical_frame_status": (
                "arena_relative_canvas_px_not_accepted_world_mm"
            ),
            "runtime_behavior_plane_id": "projected_surface",
            "runtime_render_plane_id": "projected_surface",
            "schema_version": np.int32(4),
            "units": "px",
            "x_axis_direction": "right",
            "y_axis_direction": "down",
        }
    )
    return attrs


def _write_source_h5(
    path: Path,
    *,
    duplicate_key: bool = False,
    target_camera_index: int = 0,
) -> None:
    states = np.zeros(4, dtype=_chaser_dtype())
    states["stimulus_frame_num"] = [0, 0, 1, 1]
    states["chaser_index"] = [0, 1, 0, 1]
    if duplicate_key:
        states["chaser_index"][1] = 0
    states["timestamp_ns_session"] = [10, 10, 20, 20]
    for offset, name in enumerate(
        (
            "chaser_pos_x",
            "chaser_pos_y",
            "target_pos_x",
            "target_pos_y",
            "target_clamped_pos_x",
            "target_clamped_pos_y",
        ),
        start=1,
    ):
        states[name] = np.arange(4, dtype=np.float32) + offset
    states["target_source_frame_id"] = [1, 1, 2, 2]
    states["target_source_camera_id"] = target_camera_index

    frame_dtype = np.dtype(
        [
            ("stimulus_frame_num", "<u8"),
            ("triggering_camera_frame_id", "<u8"),
            ("timestamp_ns_epoch", "<i8"),
            ("video_frame_index", "<i8"),
        ]
    )
    frames = np.asarray([(0, 1, 100, 0), (1, 2, 200, 1)], dtype=frame_dtype)
    with h5py.File(path, "w") as h5:
        tracking = h5.create_group("tracking_data")
        chaser = tracking.create_dataset(
            "chaser_states",
            data=states,
            maxshape=(None,),
        )
        chaser.attrs.update(_legacy_attrs("2010093"))
        tracking.create_dataset("bounding_boxes", data=np.arange(6, dtype=np.int64))

        video = h5.create_group("video_metadata")
        video.create_dataset("frame_metadata", data=frames)

        calibration = h5.create_group("calibration_snapshot")
        arena = calibration.create_group("arena_geometry")
        arena.attrs.update(
            {
                "arena_region_width_px": 344,
                "arena_region_height_px": 344,
                "arena_origin_in_canvas_x_px": 270,
                "arena_origin_in_canvas_y_px": 520,
            }
        )

        renderer = h5.create_group("stimulus_renderer_snapshot")
        renderer.attrs.update(
            {
                "schema_id": "citrus.stimulus_renderer_snapshot",
                "schema_version": 1,
                "capture_phase": "experiment_start_after_arena_initialization",
            }
        )
        renderer_arena = renderer.create_group("arena_1")
        renderer_arena.attrs.update(
            {
                "active_stimulus_mode": "CHASER",
                "texture_width_px": 344,
                "texture_height_px": 344,
                "texture_origin": "top_left",
            }
        )
        custom = renderer_arena.create_group("custom_coordinates")
        custom.attrs.update({"texture_center_x": 172.0, "texture_center_y": 172.0})


def _write_recording(
    root: Path,
    *,
    duplicate_key: bool = False,
    target_camera_index: int = 0,
) -> tuple[Path, Path]:
    recording = root / "recording_goodbatbadbat"
    raw = recording / "raw"
    cams = recording / "cams"
    orange = recording / "derived" / "external_crop_recorder"
    derived = recording / "derived"
    for path in (raw, cams, orange):
        path.mkdir(parents=True, exist_ok=True)

    source_h5 = raw / "recording_goodbatbadbat.h5"
    _write_source_h5(
        source_h5,
        duplicate_key=duplicate_key,
        target_camera_index=target_camera_index,
    )
    csv_path = cams / "Cam2010093_recording_meta.csv"
    csv_path.write_text(
        "recording_frame_id,camera_frame_id\n"
        "1,1001\n"
        "2,1002\n"
        "3,1003\n"
        "4,1004\n",
        encoding="utf-8",
    )
    jsonl_path = orange / "Cam2010093_recording_yolo_events.jsonl"
    events = []
    for frame_id in range(1, 5):
        events.append(
            json.dumps(
                {
                    "schema_id": "orange.yolo_event",
                    "event_kind": "yolo_result",
                    "camera_serial": "2010093",
                    "recording_id": "orange-session",
                    "camera_id": 0,
                    "frame": {
                        "ipc_frame_id": frame_id,
                        "recording_frame_id": frame_id,
                        "camera_frame_id": 1000 + frame_id,
                    },
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    jsonl_path.write_text("\n".join(events) + "\n", encoding="utf-8")

    session = {
        "camera_artifacts": {
            "2010093": {
                "first_recording_frame_id": 1,
                "last_recording_frame_id": 4,
                "frame_count": 4,
                "recording_frame_id_gaps": 0,
            }
        }
    }
    (raw / "recording_session.json").write_text(
        json.dumps(session),
        encoding="utf-8",
    )
    snapshot = {
        "camera_runtime": {
            "2010093": {
                "coordinate_frame": {
                    "coordinate_space": "camera_native_pixels",
                    "extent": {"width_px": 4512, "height_px": 4512},
                }
            }
        }
    }
    snapshot_path = derived / "recording_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    manifest = {
        "protocol_name": migration.SUPPORTED_PROTOCOL,
        "software_version": migration.SUPPORTED_CITRUS_VERSION,
        "camera_id": "2010093",
        "recording_name": recording.name,
        "orange_session_id": "orange-session",
        "recording_snapshot": "derived/recording_snapshot.json",
        "files": {"raw": ["raw/recording_goodbatbadbat.h5"]},
        "video_streams": {
            "frame_clock": "recording_frame_id",
            "streams": {
                "full": {
                    "camera_id": "2010093",
                    "frame_clock": "recording_frame_id",
                    "role": "ingest_authoritative_full_frame",
                    "frame_count": 4,
                    "frame_clock_metadata": "cams/Cam2010093_recording_meta.csv",
                }
            },
        },
    }
    (recording / "recording_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return recording, source_h5


def _bypass_selected_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        contract,
        "_preflight_selected_calibration",
        lambda _h5, *, source_h5: SimpleNamespace(
            source_evidence_sha256="fixture-selected-calibration"
        ),
    )


def test_migration_materializes_two_chaser_v5_derivative_without_touching_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording, source_h5 = _write_recording(tmp_path)
    source_file_digest = _file_sha256(source_h5)
    with h5py.File(source_h5, "r") as source:
        source_chaser_digest = contract._h5_dataset_content_digest(
            source["/tracking_data/chaser_states"]
        )
        source_frames = np.asarray(source["/video_metadata/frame_metadata"][:])

    dry_run = migration.migrate_recording(recording)
    assert dry_run["status"] == "would_migrate"
    assert dry_run["source_row_count"] == 4
    assert dry_run["stimulus_frame_count"] == 2
    assert dry_run["chaser_indices"] == [0, 1]
    assert not Path(dry_run["output_h5"]).exists()

    _bypass_selected_calibration(monkeypatch)
    result = migration.migrate_recording(recording, apply=True)
    derivative_path = Path(result["output_h5"])
    receipt_path = Path(result["external_receipt"])
    assert result["status"] == "migrated"
    assert result["raw_source_unchanged"] is True
    assert derivative_path.is_file()
    assert receipt_path.is_file()
    assert _file_sha256(source_h5) == source_file_digest

    with h5py.File(source_h5, "r") as source:
        assert "/tracking_data/bounding_boxes" in source
        assert source["/tracking_data/chaser_states"].attrs["schema_version"] == 4
    with h5py.File(derivative_path, "r") as derivative:
        assert "/tracking_data/bounding_boxes" not in derivative
        assert derivative["/tracking_data/chaser_states"].attrs["schema_version"] == 5
        assert contract._h5_dataset_content_digest(
            derivative["/tracking_data/chaser_states"]
        ) == source_chaser_digest
        assert np.array_equal(
            derivative["/video_metadata/frame_metadata"][:],
            source_frames,
        )
        assert np.array_equal(
            derivative["/tracking_data/stimulus_state_key"][:],
            np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64),
        )
        assert np.array_equal(
            derivative["/tracking_data/source_acquisition_frame_index"][:],
            np.asarray([0, 0, 1, 1], dtype=np.int64),
        )
        assert np.array_equal(
            derivative["/tracking_data/target_source_acquisition_frame_index"][:],
            np.asarray([0, 0, 1, 1], dtype=np.int64),
        )
        preflight = contract.preflight_stimulus_coordinate_contract(
            derivative,
            source_h5=derivative_path,
        )
        assert preflight.has_chaser_states is True
        assert preflight.row_identity_values is not None
        assert preflight.row_identity_values.shape == (4, 2)


@pytest.mark.parametrize(
    ("duplicate_key", "target_camera_index", "message"),
    [
        (True, 0, "composite stimulus-state identity is not unique"),
        (False, 1, "matching Orange camera evidence"),
    ],
)
def test_migration_rejects_ambiguous_row_or_target_identity(
    tmp_path: Path,
    duplicate_key: bool,
    target_camera_index: int,
    message: str,
) -> None:
    recording, _source_h5 = _write_recording(
        tmp_path,
        duplicate_key=duplicate_key,
        target_camera_index=target_camera_index,
    )
    with pytest.raises(migration.GoodBatBadBatMigrationError, match=message):
        migration.plan_migration(recording)
