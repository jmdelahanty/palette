from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.acquisition_frame_clock import (
    AcquisitionFrameClockError,
    import_acquisition_frame_clock,
    load_acquisition_frame_clock_source,
    resolve_acquisition_frame_clock,
)


def _write_clock_csv(
    path: Path,
    *,
    frame_count: int,
    frame_id_base: int = 1,
    system_start_ns: int = 2_000_000_000,
    camera_minus_system_ns: int = -1_000_000_000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["recording_frame_id", "timestamp", "timestamp_sys"],
        )
        writer.writeheader()
        for index in range(frame_count):
            writer.writerow(
                {
                    "recording_frame_id": index + frame_id_base,
                    "timestamp": system_start_ns
                    + camera_minus_system_ns
                    + index * 10_000_000,
                    "timestamp_sys": system_start_ns + index * 10_000_000,
                }
            )


def _write_ptp_summary(recording: Path, *, camera_id: str) -> None:
    path = recording / "raw" / "ptp_sync_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sync": {"camera_sync_enabled": True, "mode": "ptp_local"},
                "cameras": {
                    camera_id: {
                        "camera_serial": camera_id,
                        "sync_camera_enabled": True,
                        "ptp_register_reads": 8,
                        "ptp_offset_ns": {
                            "samples": 8,
                            "min": 400,
                            "max": 800,
                        },
                        "latch_minus_frame_ns": {
                            "samples": 8,
                            "min": 8_000_000,
                            "max": 10_000_000,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_import_acquisition_frame_clock_publishes_immutable_numeric_authority(
    tmp_path: Path,
) -> None:
    recording = tmp_path / "recording"
    video = recording / "cams" / "Cam2010093_recording.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    _write_clock_csv(video.with_name(f"{video.stem}_meta.csv"), frame_count=4)
    root = zarr.open_group(
        str(recording / "zarr" / "recording_analysis.zarr"),
        mode="w",
        zarr_format=3,
    )

    resolved = import_acquisition_frame_clock(
        root,
        recording_dir=recording,
        camera_id="2010093",
        video_path=video,
        expected_frame_count=4,
    )

    assert resolved is not None
    assert resolved.row_count == 4
    assert resolved.camera_id == "2010093"
    run = root[resolved.group_path]
    np.testing.assert_array_equal(run["recording_frame_id"][:], [1, 2, 3, 4])
    np.testing.assert_array_equal(run["parent_frame_index"][:], [0, 1, 2, 3])
    np.testing.assert_array_equal(
        run["camera_timestamp_ns"][:],
        [1_000_000_000, 1_010_000_000, 1_020_000_000, 1_030_000_000],
    )
    assert np.all(run["camera_timestamp_valid"][:])
    assert root.attrs["acquisition_frame_clock_ref"] == resolved.group_path
    assert root.attrs["acquisition_frame_clock_sha256"] == resolved.record_sha256
    camera_semantics = resolved.record["clock_surfaces"]["camera_timestamp_ns"]
    system_semantics = resolved.record["clock_surfaces"]["system_timestamp_ns"]
    assert camera_semantics["time_reference_kind"] == "device_defined_unknown_epoch"
    assert camera_semantics["origin"] == "unspecified"
    assert system_semantics["time_reference_kind"] == "absolute_epoch"
    assert system_semantics["origin"] == "1970-01-01T00:00:00_UTC"
    assert system_semantics["timescale"] == "POSIX_UTC_excluding_leap_seconds"

    repeated = import_acquisition_frame_clock(
        root,
        recording_dir=recording,
        camera_id="2010093",
        video_path=video,
        expected_frame_count=4,
    )
    assert repeated == resolved
    parent = root["analysis/acquisition_frame_clock_runs"]
    assert list(parent.group_keys()) == [resolved.run_name]
    assert resolve_acquisition_frame_clock(root) == resolved


def test_acquisition_frame_clock_labels_operationally_confirmed_ptp_tai(
    tmp_path: Path,
) -> None:
    recording = tmp_path / "recording"
    video = recording / "cams" / "Cam2010093_recording.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    _write_clock_csv(
        video.with_name(f"{video.stem}_meta.csv"),
        frame_count=4,
        system_start_ns=1_784_662_711_195_121_161,
        camera_minus_system_ns=36_990_853_682,
    )
    _write_ptp_summary(recording, camera_id="2010093")
    root = zarr.open_group(
        str(recording / "zarr" / "recording_analysis.zarr"),
        mode="w",
        zarr_format=3,
    )

    resolved = import_acquisition_frame_clock(
        root,
        recording_dir=recording,
        camera_id="2010093",
        video_path=video,
        expected_frame_count=4,
    )

    assert resolved is not None
    camera_semantics = resolved.record["clock_surfaces"]["camera_timestamp_ns"]
    assert camera_semantics["time_reference_kind"] == "absolute_epoch"
    assert camera_semantics["origin"] == "1970-01-01T00:00:00_TAI"
    assert camera_semantics["timescale"] == "IEEE-1588_PTP_TAI"
    assert (
        camera_semantics["semantic_status"]
        == "inferred_from_recording_evidence_not_sdk_declared"
    )
    evidence = resolved.record["clock_semantic_evidence"]
    assert evidence["camera_ptp_semantics_inferred"] is True
    assert evidence["camera_system_delta_ns"]["median_ns"] == 36_990_853_682
    assert evidence["ptp_offsets_indicate_synchronization"] is True
    assert evidence["ptp_latch_agrees_with_embedded_frame_time"] is True
    assert evidence["ptp_sync_summary"]["locator"] == "raw/ptp_sync_summary.json"


def test_acquisition_frame_clock_does_not_treat_ptp_enablement_as_sync(
    tmp_path: Path,
) -> None:
    recording = tmp_path / "recording"
    video = recording / "cams" / "Cam2010093_recording.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    _write_clock_csv(
        video.with_name(f"{video.stem}_meta.csv"),
        frame_count=3,
        system_start_ns=1_784_662_711_195_121_161,
        camera_minus_system_ns=37_000_000_000,
    )
    summary_path = recording / "raw" / "ptp_sync_summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(
            {
                "sync": {"camera_sync_enabled": True, "mode": "ptp_local"},
                "cameras": {
                    "2010093": {
                        "camera_serial": "2010093",
                        "sync_camera_enabled": True,
                        "ptp_register_reads": 8,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    source = load_acquisition_frame_clock_source(
        recording,
        camera_id="2010093",
        video_path=video,
        expected_frame_count=3,
    )

    assert source is not None
    camera_semantics = source.clock_surfaces["camera_timestamp_ns"]
    assert camera_semantics["time_reference_kind"] == "device_defined_unknown_epoch"
    assert source.clock_semantic_evidence["camera_ptp_semantics_inferred"] is False
    assert (
        source.clock_semantic_evidence["ptp_offsets_indicate_synchronization"]
        is False
    )


def test_acquisition_frame_clock_rejects_row_count_mismatch(tmp_path: Path) -> None:
    recording = tmp_path / "recording"
    video = recording / "cams" / "Cam2010093_recording.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    _write_clock_csv(video.with_name(f"{video.stem}_meta.csv"), frame_count=3)

    with pytest.raises(AcquisitionFrameClockError, match="row count does not match"):
        load_acquisition_frame_clock_source(
            recording,
            camera_id="2010093",
            video_path=video,
            expected_frame_count=4,
        )


def test_acquisition_frame_clock_marks_unavailable_without_camera_clock_source(
    tmp_path: Path,
) -> None:
    recording = tmp_path / "recording"
    video = recording / "cams" / "Cam2010093_recording.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    root = zarr.open_group(
        str(recording / "zarr" / "recording_analysis.zarr"),
        mode="w",
        zarr_format=3,
    )

    resolved = import_acquisition_frame_clock(
        root,
        recording_dir=recording,
        camera_id="2010093",
        video_path=video,
        expected_frame_count=4,
    )

    assert resolved is None
    assert root.attrs["acquisition_frame_clock_available"] is False
    assert (
        root.attrs["acquisition_frame_clock_status"]
        == "unavailable_no_camera_clock_source"
    )


def test_acquisition_frame_clock_resolver_detects_array_tampering(tmp_path: Path) -> None:
    recording = tmp_path / "recording"
    video = recording / "cams" / "Cam2010093_recording.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    _write_clock_csv(video.with_name(f"{video.stem}_meta.csv"), frame_count=3)
    root = zarr.open_group(
        str(recording / "zarr" / "recording_analysis.zarr"),
        mode="w",
        zarr_format=3,
    )
    resolved = import_acquisition_frame_clock(
        root,
        recording_dir=recording,
        camera_id="2010093",
        video_path=video,
        expected_frame_count=3,
    )
    assert resolved is not None

    root[f"{resolved.group_path}/camera_timestamp_ns"][1] = 123

    with pytest.raises(AcquisitionFrameClockError, match="bound digest"):
        resolve_acquisition_frame_clock(root)
