from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.acquisition_video_streams import write_acquisition_video_stream_inventory
from fisheye.shared.acquisition_crop_stream_ledger import (
    validate_current_acquisition_crop_stream_ledger,
)
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings


def _fixture(tmp_path: Path, *, frame_count: int = 2) -> tuple[Path, dict[str, object]]:
    recording_dir = tmp_path / "recording"
    crop_dir = recording_dir / "derived" / "external_crop_recorder"
    crop_dir.mkdir(parents=True)
    video = crop_dir / "crop.mp4"
    video.write_bytes(b"immutable crop video")
    metadata = crop_dir / "crop_meta.csv"
    metadata.write_text(
        "recording_frame_id,local_frame_id,camera_frame_id,timestamp,timestamp_sys,"
        "has_detection,blank_frame,detection_confidence,crop_x,crop_y,crop_w,crop_h,"
        "detection_x,detection_y,detection_w,detection_h,crop_video_frame_index,"
        "crop_state,crop_rect_valid,crop_rect_coordinate_space,crop_rect_layout,"
        "crop_rect_semantics,detection_rect_valid,detection_rect_coordinate_space,"
        "detection_rect_layout,detection_rect_semantics,detection_source,selection_policy,"
        "session_crop_video_frame_index\n"
        "1,10,10,100,90,1,0,0.8,5,6,384,384,100,120,20,30,0,detected_crop,1,"
        "full_frame_pixels,xywh_top_left,actual_clamped_source_roi,1,full_frame_pixels,"
        "xywh_top_left,selected_postprocessed_model_detection,model,largest,0\n"
        "2,11,11,110,100,0,1,0,0,0,0,0,0,0,0,0,1,blank_no_detection,0,"
        "full_frame_pixels,xywh_top_left,actual_clamped_source_roi,0,full_frame_pixels,"
        "xywh_top_left,selected_postprocessed_model_detection,none,largest,1\n",
        encoding="utf-8",
    )
    manifest: dict[str, object] = {
        "camera_id": "2010093",
        "video_streams": {
            "schema_id": "orange_runtime_video_streams_v1",
            "frame_clock": "recording_frame_id",
            "streams": {
                "crop": {
                    "output_kind": "crop",
                    "camera_id": "2010093",
                    "stream_id": "2010093_crop",
                    "video": str(video.relative_to(recording_dir)),
                    "metadata": str(metadata.relative_to(recording_dir)),
                    "frame_clock": "recording_frame_id",
                    "video_pixel_coordinate_space": "crop_frame_pixels",
                    "source_geometry_coordinate_space": "full_frame_pixels",
                    "frame_count": frame_count,
                    "width": 384,
                    "height": 384,
                }
            },
        },
    }
    (recording_dir / "recording_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return recording_dir, manifest


def test_complete_ledger_preserves_blank_rows_and_is_idempotent(tmp_path: Path) -> None:
    recording_dir, manifest = _fixture(tmp_path)
    root = zarr.open_group(str(tmp_path / "analysis.zarr"), mode="w", zarr_format=3)

    first = write_acquisition_video_stream_inventory(
        root, recording_dir, manifest, imported_at_utc="2026-08-15T12:00:00+00:00"
    )
    second = write_acquisition_video_stream_inventory(
        root, recording_dir, manifest, imported_at_utc="2026-08-15T13:00:00+00:00"
    )

    assert first is not None and second is not None
    stream = root["analysis/acquisition_video_streams/streams/crop"]
    assert stream.attrs["canonical_ledger_status"] == "complete"
    assert stream.attrs["canonical_ledger_run"].startswith("crop_ledger_20260815T120000Z_")
    assert second["streams"]["crop"]["canonical_ledger"]["canonical_ledger_idempotent"] is True
    run = stream[stream.attrs["canonical_ledger_path"]]
    np.testing.assert_array_equal(run["source_recording_frame_ids"][:], [1, 2])
    np.testing.assert_array_equal(run["source_crop_video_frame_indices"][:], [0, 1])
    np.testing.assert_array_equal(run["has_detection"][:], [True, False])
    np.testing.assert_array_equal(run["blank_frame"][:], [False, True])
    assert run.attrs["publication_status"] == "complete"
    assert len(list(stream["ledger_runs"].group_keys())) == 1

    consolidate_metadata_capture_expected_warnings(tmp_path / "analysis.zarr")
    consolidated = zarr.open_group(
        str(tmp_path / "analysis.zarr"), mode="r", use_consolidated=True
    )
    validated = validate_current_acquisition_crop_stream_ledger(
        consolidated,
        expected_record_sha256=stream.attrs["canonical_ledger_record_sha256"],
    )
    assert validated.row_count == 2


def test_cardinality_mismatch_fails_without_current_pointer(tmp_path: Path) -> None:
    recording_dir, manifest = _fixture(tmp_path, frame_count=3)
    root = zarr.open_group(str(tmp_path / "analysis.zarr"), mode="w", zarr_format=3)

    with pytest.raises(ValueError, match="row count 2 does not match frame_count 3"):
        write_acquisition_video_stream_inventory(root, recording_dir, manifest)

    stream = root["analysis/acquisition_video_streams/streams/crop"]
    assert stream.attrs["canonical_ledger_status"] == "failed"
    assert "canonical_ledger_run" not in stream.attrs


def test_changed_source_refuses_to_replace_immutable_current_ledger(tmp_path: Path) -> None:
    recording_dir, manifest = _fixture(tmp_path)
    root = zarr.open_group(str(tmp_path / "analysis.zarr"), mode="w", zarr_format=3)
    write_acquisition_video_stream_inventory(root, recording_dir, manifest)
    metadata = recording_dir / "derived" / "external_crop_recorder" / "crop_meta.csv"
    text = metadata.read_text(encoding="utf-8").replace("100,120,20,30", "101,120,20,30")
    metadata.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="different immutable canonical ledger"):
        write_acquisition_video_stream_inventory(root, recording_dir, manifest)
