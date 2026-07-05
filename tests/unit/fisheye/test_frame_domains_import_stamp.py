from __future__ import annotations

from pathlib import Path

import zarr

from fisheye.capture.import_video import stamp_stored_zarr_frame_identity_mapping
from fisheye.shared.frame_domains import FrameDomain
from fisheye.shared.recording import open_recording

from .frame_domain_fixtures import build_full_missing_mapping_store


def test_import_identity_stamp_round_trips_with_resolver(tmp_path: Path) -> None:
    zarr_path = build_full_missing_mapping_store(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    stamp_stored_zarr_frame_identity_mapping(root["raw_video"], 5)

    domains = open_recording(zarr_path).frame_domains()

    assert "original_frame_indices" not in root["raw_video"]
    assert domains.convert([0, 4], FrameDomain.STORED_ZARR, FrameDomain.ACQUISITION).tolist() == [0, 4]
