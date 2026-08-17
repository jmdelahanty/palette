from __future__ import annotations

import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows.position_body_frame_motion import (
    PositionBodyFrameMotionError,
    bind_position_body_frame_to_tracking,
    compose_position_body_frame_motion_authority,
)
from fisheye.analysis_workflows.tracking_source_handle import (
    TrackingSourceHandleError,
    load_tracking_source_handle,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.tracking.single_subject_per_arena import (
    write_single_subject_per_arena_tracking_run,
)
from tests.unit.fisheye.test_position_body_frame_motion import (
    _handles,
    _tracking_handle,
)


def test_loader_mints_readonly_exact_keyed_tracking_handle(tmp_path) -> None:
    position, body_frame = _handles(tmp_path)
    source = compose_position_body_frame_motion_authority(position, body_frame)

    handle = _tracking_handle(source)

    assert handle.run_path.startswith("tracking_runs/tracking_")
    assert handle.selector_eligible is True
    assert handle.instance_key.dtype == np.dtype("uint64")
    assert handle.track_ids.dtype == np.dtype("int64")
    assert handle.instance_key.flags.writeable is False
    assert handle.track_ids.flags.writeable is False
    handle.assert_current()


def test_loader_rejects_keyless_legacy_tracking_as_modern_authority(tmp_path) -> None:
    position, body_frame = _handles(tmp_path)
    source = compose_position_body_frame_motion_authority(position, body_frame)
    root = zarr.open_group(
        str(source.analysis_zarr_path),
        mode="r+",
        zarr_format=3,
        use_consolidated=False,
    )
    run_name, _run, _summary = write_single_subject_per_arena_tracking_run(
        root=root,
        arena_ids=np.asarray([0, 0, 0], dtype=np.int32),
        frame_indices=np.asarray([0, 1, 2], dtype=np.int64),
        source_detect_run="detect_legacy",
        source_arena_assignment_run="arena_legacy",
        source_rowset_path="detect_runs/detect_legacy",
    )

    with pytest.raises(TrackingSourceHandleError, match="instance_key"):
        load_tracking_source_handle(
            source.analysis_zarr_path,
            f"tracking_runs/{run_name}",
            expected_selector_eligible=True,
            use_consolidated=False,
        )


def test_handle_reopens_and_rejects_post_seal_array_mutation(tmp_path) -> None:
    position, body_frame = _handles(tmp_path)
    source = compose_position_body_frame_motion_authority(position, body_frame)
    handle = _tracking_handle(source)
    root = zarr.open_group(
        str(source.analysis_zarr_path),
        mode="r+",
        zarr_format=3,
        use_consolidated=False,
    )
    root[f"{handle.run_path}/track_ids"][0] = np.int32(99)

    with pytest.raises(TrackingSourceHandleError, match="manifest"):
        handle.assert_current()


def test_consolidated_loader_rejects_stale_direct_tracking_payload(tmp_path) -> None:
    position, body_frame = _handles(tmp_path)
    source = compose_position_body_frame_motion_authority(position, body_frame)
    direct_handle = _tracking_handle(source)
    consolidate_metadata_capture_expected_warnings(source.analysis_zarr_path)
    published = load_tracking_source_handle(
        source.analysis_zarr_path,
        direct_handle.run_path,
        expected_selector_eligible=True,
        use_consolidated=True,
    )
    published.assert_current()
    root = zarr.open_group(
        str(source.analysis_zarr_path),
        mode="r+",
        zarr_format=3,
        use_consolidated=False,
    )
    root[f"{published.run_path}/track_ids"][0] = np.int32(99)

    with pytest.raises(TrackingSourceHandleError, match="manifest"):
        published.assert_current()


def test_tracking_and_provider_authorities_must_share_archive(tmp_path) -> None:
    first_position, first_body = _handles(tmp_path / "first")
    second_position, second_body = _handles(tmp_path / "second")
    first = compose_position_body_frame_motion_authority(first_position, first_body)
    second = compose_position_body_frame_motion_authority(second_position, second_body)
    tracking = _tracking_handle(first)

    with pytest.raises(PositionBodyFrameMotionError, match="same analysis archive"):
        bind_position_body_frame_to_tracking(second, tracking)
