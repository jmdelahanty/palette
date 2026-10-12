"""camera_tai_minus_utc: the offset and its authority come from the published clock."""

from __future__ import annotations

import pytest

from fisheye.shared.acquisition_frame_clock import ResolvedAcquisitionFrameClock, camera_tai_minus_utc


def _resolved(version: int, camera: dict) -> ResolvedAcquisitionFrameClock:
    return ResolvedAcquisitionFrameClock(
        run_name="r",
        group_path="analysis/acquisition_frame_clock_runs/r",
        record={"schema_version": version, "clock_surfaces": {"camera_timestamp_ns": camera}},
        record_sha256="0" * 64,
        camera_id="1",
        row_count=0,
    )


@pytest.mark.parametrize(
    ("version", "camera", "expected"),
    [
        (2, {"semantic_authority": "orange_declared", "tai_minus_utc_seconds": 37}, (37, "orange_declared")),
        (2, {"semantic_authority": "palette_inferred", "tai_minus_utc_seconds": 37}, (37, "palette_inferred")),
        (2, {"semantic_authority": "palette_inferred", "tai_minus_utc_seconds": None}, (None, None)),
        (1, {"timescale": "IEEE-1588_PTP_TAI"}, (37, "palette_inferred")),
        (1, {"timescale": "device_defined"}, (None, None)),
    ],
)
def test_offset_and_authority(version, camera, expected):
    result = camera_tai_minus_utc(_resolved(version, camera))
    assert (result["seconds"], result["authority"]) == expected
