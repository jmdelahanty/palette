from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from fisheye.shared.acquisition_crop_stream_ledger import (
    ACQUISITION_CROP_SOURCE_PROFILE_COLLECTION,
    validate_current_acquisition_crop_stream_ledger,
)
from fisheye.shared.acquisition_video_streams import (
    write_acquisition_video_stream_inventory,
)


def _write_clip(
    recording_dir: Path,
    *,
    clip_index: int,
    first_frame_id: int,
) -> dict[str, object]:
    clip_id = f"clip_{clip_index:06d}"
    clip_dir = recording_dir / "clips" / clip_id
    crop_dir = clip_dir / "crop"
    crop_dir.mkdir(parents=True)
    full_video = clip_dir / "Cam123.mp4"
    crop_video = crop_dir / "Cam123_crop_external.mp4"
    crop_meta = crop_dir / "Cam123_crop_meta.csv"
    full_video.write_bytes(f"full-{clip_index}".encode())
    crop_video.write_bytes(f"crop-{clip_index}".encode())
    second_frame_id = first_frame_id + 1
    crop_meta.write_text(
        "recording_frame_id,has_detection,blank_frame,crop_x,crop_y,crop_w,crop_h,"
        "detection_x,detection_y,detection_w,detection_h,crop_video_frame_index,"
        "session_crop_video_frame_index\n"
        f"{first_frame_id},1,0,10,20,384,384,100,120,20,30,0,{first_frame_id - 1}\n"
        f"{second_frame_id},0,1,,,,,,,,,1,{second_frame_id - 1}\n",
        encoding="utf-8",
    )
    manifest_path = clip_dir / "clip_manifest.json"

    def relative(path: Path) -> str:
        return str(path.relative_to(recording_dir))

    manifest_path.write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "clip_index": clip_index,
                "recording_outputs": {
                    "123": {
                        "crop": {
                            "output_kind": "crop",
                            "first_recording_frame_id": first_frame_id,
                            "last_recording_frame_id": second_frame_id,
                            "frame_count": 2,
                            "video": relative(crop_video),
                            "metadata": relative(crop_meta),
                        },
                        "full": {
                            "output_kind": "full",
                            "first_recording_frame_id": first_frame_id,
                            "last_recording_frame_id": second_frame_id,
                            "frame_count": 2,
                            "video": relative(full_video),
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "camera_serial": "123",
        "clip_id": clip_id,
        "clip_index": clip_index,
        "clip_manifest_path": relative(manifest_path),
        "first_recording_frame_id": first_frame_id,
        "last_recording_frame_id": second_frame_id,
        "frame_count": 2,
    }


def test_inventory_publishes_one_collection_ledger_for_two_clips(
    tmp_path: Path,
) -> None:
    recording_dir = tmp_path / "recording"
    rows = [
        _write_clip(recording_dir, clip_index=0, first_frame_id=1),
        _write_clip(recording_dir, clip_index=1, first_frame_id=3),
    ]
    clip_index_path = recording_dir / "recording_clip_index.json"
    clip_index_path.write_text(
        json.dumps(
            {
                "schema_id": "palette.orange_external_ipc_recording_clip_index.v1",
                "mode": "rolling_clips",
                "rows": rows,
                "camera_ranges": {
                    "123": {
                        "clip_count": 2,
                        "total_frame_count": 4,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "camera_id": "123",
        "recording_clip_index": "recording_clip_index.json",
        "rolling_clip_streams": {
            "schema_id": "palette.orange_rolling_clip_streams.v1",
            "frame_clock": "recording_frame_id",
            "recording_clip_index": "recording_clip_index.json",
        },
    }
    root = zarr.open_group(str(tmp_path / "analysis.zarr"), mode="w")

    inventory = write_acquisition_video_stream_inventory(
        root,
        recording_dir,
        manifest,
        imported_at_utc="2026-08-21T12:00:00+00:00",
    )
    publication = validate_current_acquisition_crop_stream_ledger(root)
    run = root[
        "analysis/acquisition_video_streams/streams/crop/ledger_runs/"
        + publication.run_name
    ]

    assert inventory is not None
    assert inventory["source_profile"] == ACQUISITION_CROP_SOURCE_PROFILE_COLLECTION
    assert publication.source_profile == ACQUISITION_CROP_SOURCE_PROFILE_COLLECTION
    assert publication.row_count == 4
    assert len(run.attrs["source_media_members"]) == 2
    np.testing.assert_array_equal(
        run["source_recording_frame_ids"][:], np.array([1, 2, 3, 4])
    )
    np.testing.assert_array_equal(
        run["source_crop_video_frame_indices"][:], np.array([0, 1, 0, 1])
    )
    np.testing.assert_array_equal(
        run["source_crop_video_member_indices"][:], np.array([0, 0, 1, 1])
    )
    np.testing.assert_array_equal(
        run["source_full_video_member_indices"][:], np.array([0, 0, 1, 1])
    )
