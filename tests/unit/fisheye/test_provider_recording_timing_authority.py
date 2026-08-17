from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows.provider_recording_timing_authority import (
    NOMINAL_FRAME_TIME_POLICY_ID,
    ProviderRecordingTimingAuthorityError,
    load_provider_recording_timing_authority,
)
from fisheye.shared.acquisition_frame_clock import import_acquisition_frame_clock
from fisheye.shared.source_video_metadata import build_source_video_metadata_v2


def _install_clock_authority(
    archive: Path,
    tmp_path: Path,
    *,
    frame_count: int,
    fps: float,
    recording_id: str = "recording-001",
) -> None:
    recording = tmp_path / "recording_clock_source"
    video = recording / "cams" / "Cam2010093_recording.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    clock_csv = video.with_name(f"{video.stem}_meta.csv")
    with clock_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["recording_frame_id", "timestamp", "timestamp_sys"],
        )
        writer.writeheader()
        for index in range(frame_count):
            writer.writerow(
                {
                    "recording_frame_id": index + 1,
                    "timestamp": 1_000_000_000 + round(index * 1_000_000_000 / fps),
                    "timestamp_sys": 2_000_000_000 + round(index * 1_000_000_000 / fps),
                }
            )

    root = zarr.open_group(
        str(archive), mode="a", zarr_format=3, use_consolidated=False
    )
    metadata = build_source_video_metadata_v2(
        {
            "source_path": str(video),
            "camera_id": "2010093",
            "width": 640,
            "height": 640,
            "fps": fps,
            "total_frames": frame_count,
        },
        recording_path=recording,
    )
    root.attrs.update(
        {
            "recording_id": recording_id,
            "recording_path": str(recording),
            "camera_id": "2010093",
            "fps": fps,
            "total_frames": frame_count,
            "source_video_metadata": metadata,
        }
    )
    raw = root.require_group("raw_video")
    raw.attrs.update({"fps": fps, "total_frames": frame_count})
    imported = import_acquisition_frame_clock(
        root,
        recording_dir=recording,
        camera_id="2010093",
        video_path=video,
        expected_frame_count=frame_count,
    )
    assert imported is not None
    zarr.consolidate_metadata(str(archive))


def _clock_archive(tmp_path: Path, *, frame_count: int = 4) -> Path:
    archive = tmp_path / "archive" / "recording_analysis.zarr"
    zarr.open_group(str(archive), mode="w", zarr_format=3)
    _install_clock_authority(
        archive,
        tmp_path,
        frame_count=frame_count,
        fps=100.0,
    )
    return archive


def test_authority_binds_existing_clock_without_copying_timestamp_values(
    tmp_path: Path,
) -> None:
    archive = _clock_archive(tmp_path)

    authority = load_provider_recording_timing_authority(archive)

    assert authority is not None
    assert authority.recording_id == "recording-001"
    assert authority.camera_id == "2010093"
    assert authority.nominal_fps == 100.0
    assert authority.frame_count == 4
    assert authority.record["policy_id"] == NOMINAL_FRAME_TIME_POLICY_ID
    clock = authority.record["acquisition_frame_clock"]
    assert "camera_timestamp_ns" in clock["array_sha256"]
    assert all(isinstance(value, str) for value in clock["array_sha256"].values())
    assert authority.record["numerical_semantics"] == {
        "timebase": "nominal_fps",
        "frame_delta_rule": "acquisition_frame_index_difference_divided_by_fps",
        "camera_timestamp_arrays_copied": False,
    }
    authority.validate_source_frame_indices(
        np.asarray([0, 2, 3], dtype=np.int64), name="provider frames"
    )
    authority.assert_current()


def test_authority_rejects_out_of_domain_or_non_int64_indices(tmp_path: Path) -> None:
    authority = load_provider_recording_timing_authority(_clock_archive(tmp_path))
    assert authority is not None

    with pytest.raises(ProviderRecordingTimingAuthorityError, match="outside"):
        authority.validate_source_frame_indices(
            np.asarray([0, 4], dtype=np.int64), name="provider frames"
        )
    with pytest.raises(ProviderRecordingTimingAuthorityError, match="int64"):
        authority.validate_source_frame_indices(
            np.asarray([0, 1], dtype=np.int32), name="provider frames"
        )


def test_authority_rejects_stale_direct_metadata_and_expected_digest(
    tmp_path: Path,
) -> None:
    archive = _clock_archive(tmp_path)
    authority = load_provider_recording_timing_authority(archive)
    assert authority is not None
    with pytest.raises(ProviderRecordingTimingAuthorityError, match="expected digest"):
        load_provider_recording_timing_authority(
            archive,
            use_consolidated=False,
            expected_sha256="f" * 64,
        )
    direct = zarr.open_group(
        str(archive), mode="a", zarr_format=3, use_consolidated=False
    )
    direct.attrs["fps"] = 99.0

    with pytest.raises(
        ProviderRecordingTimingAuthorityError,
        match="consolidated recording timing metadata",
    ):
        load_provider_recording_timing_authority(archive)


def test_authority_rejects_camera_and_frame_count_mismatch(tmp_path: Path) -> None:
    archive = _clock_archive(tmp_path)
    root = zarr.open_group(
        str(archive), mode="a", zarr_format=3, use_consolidated=False
    )
    metadata = dict(root.attrs["source_video_metadata"])
    metadata["camera_id"] = "other-camera"
    root.attrs["source_video_metadata"] = metadata

    with pytest.raises(
        ProviderRecordingTimingAuthorityError, match="camera identities"
    ):
        load_provider_recording_timing_authority(archive, use_consolidated=False)

    metadata["camera_id"] = "2010093"
    metadata["total_frames"] = 5
    root.attrs["source_video_metadata"] = metadata
    with pytest.raises(ProviderRecordingTimingAuthorityError, match="frame count"):
        load_provider_recording_timing_authority(archive, use_consolidated=False)


def test_authority_rejects_malformed_source_video_locator(tmp_path: Path) -> None:
    archive = _clock_archive(tmp_path)
    root = zarr.open_group(
        str(archive), mode="a", zarr_format=3, use_consolidated=False
    )
    metadata = dict(root.attrs["source_video_metadata"])
    locator = dict(metadata["locator"])
    locator["relative_path"] = "../outside.mp4"
    metadata["locator"] = locator
    root.attrs["source_video_metadata"] = metadata

    with pytest.raises(
        ProviderRecordingTimingAuthorityError,
        match="recording-relative source-video path",
    ):
        load_provider_recording_timing_authority(
            archive,
            use_consolidated=False,
        )


def test_missing_clock_is_explicit_legacy_or_required_failure(tmp_path: Path) -> None:
    archive = tmp_path / "legacy.zarr"
    zarr.open_group(str(archive), mode="w", zarr_format=3)
    zarr.consolidate_metadata(str(archive))

    assert load_provider_recording_timing_authority(archive, required=False) is None
    with pytest.raises(ProviderRecordingTimingAuthorityError, match="no acquisition"):
        load_provider_recording_timing_authority(archive, required=True)
