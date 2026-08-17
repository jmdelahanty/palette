from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import zarr

import fisheye.analysis_workflows.materializers.provider_track_motion as provider_mod
from fisheye.analysis_workflows.materializers.provider_track_motion import (
    PROVIDER_TRACK_MOTION_PARENT_PATH,
    ProviderTrackMotionError,
    plan_provider_track_motion_run,
    prepare_provider_track_motion,
    publish_provider_track_motion_run,
    validate_provider_track_motion_run,
)
from fisheye.analysis_workflows.position_body_frame_motion import (
    bind_position_body_frame_to_tracking,
    compose_position_body_frame_motion_authority,
)
from fisheye.analysis_workflows.tracking_source_handle import (
    TrackingSourceHandleError,
)
from tests.unit.fisheye.test_position_body_frame_motion import (
    _handles,
    _tracking_handle,
)


def _tracked(tmp_path):  # type: ignore[no-untyped-def]
    position, body_frame = _handles(tmp_path)
    source = compose_position_body_frame_motion_authority(position, body_frame)
    return bind_position_body_frame_to_tracking(source, _tracking_handle(source))


def _install_fake_physical_authority(monkeypatch, *, mm_per_pixel: float = 0.25):
    source_camera = SimpleNamespace(
        record_ref="/analysis/coordinate_frames/source_camera/camera-a/continuous",
        record_sha256="a" * 64,
    )
    selected = SimpleNamespace(
        record_ref="/analysis/calibration/coordinate_frames/selected_camera_evidence",
        record_sha256="b" * 64,
    )
    physical_frame = SimpleNamespace(
        record_ref="/analysis/calibration/coordinate_frames/source_camera_physical_mm",
        record_sha256="c" * 64,
        source_camera_pixels=source_camera,
        selected_camera_evidence=selected,
    )
    physical = SimpleNamespace(
        camera_id="camera-a",
        source_kind="recording_calibration",
        manifest=SimpleNamespace(
            record_ref="/analysis/calibration/coordinate_frames@source_camera_physical_authority",
            record_sha256="d" * 64,
        ),
        physical_frame=physical_frame,
        mm_per_pixel=mm_per_pixel,
    )
    monkeypatch.setattr(
        provider_mod,
        "load_source_camera_physical_authority",
        lambda _root: physical,
    )
    monkeypatch.setattr(
        provider_mod,
        "require_bound_source_camera_physical_authority",
        lambda value: value,
    )
    return physical


def test_prepared_provider_motion_keeps_lineages_and_validity_independent(
    tmp_path,
) -> None:
    prepared = prepare_provider_track_motion(
        _tracked(tmp_path),
        fps=10.0,
        smooth_seconds=0.0,
        allow_pixel_only=True,
    )

    arrays = prepared.arrays
    assert arrays["track_ids"].tolist() == [0, 1]
    assert arrays["track_row_offsets"].tolist() == [0, 2, 3]
    assert arrays["source_provider_row_index"].tolist() == [0, 1, 2]
    assert arrays["source_tracking_row_index"].tolist() == [1, 2, 0]
    assert arrays["linear_sample_valid"].all()
    assert arrays["angular_sample_valid"].all()
    assert "sample_valid" not in arrays
    assert prepared.computation_record["validity_profile"] == (
        "explicit_position_body_frame_independent_validity.v1"
    )


def test_publishes_selector_ineligible_successor_without_pointer_mutation(
    tmp_path,
    monkeypatch,
) -> None:
    _install_fake_physical_authority(monkeypatch)
    tracked = _tracked(tmp_path)
    prepared = prepare_provider_track_motion(tracked, fps=10.0, smooth_seconds=0.0)
    archive = tracked.source_authority.analysis_zarr_path
    root = zarr.open_group(
        str(archive), mode="r+", zarr_format=3, use_consolidated=False
    )
    analysis = root["analysis"]
    track_family = analysis.require_group("track_kinematics_runs")
    track_family.attrs["latest"] = "offline/legacy_selected"
    plan = plan_provider_track_motion_run(
        archive,
        prepared,
        run_name="provider_motion_canary_001",
        scratch_root=tmp_path / "provider_motion_scratch",
    )

    result = publish_provider_track_motion_run(plan, keep_scratch=True)

    assert result["acceptance"]["consolidated_validation"]["valid"] is True
    validated = validate_provider_track_motion_run(
        archive,
        plan.run_path,
        use_consolidated=True,
        expected_manifest_sha256=plan.manifest_sha256,
    )
    assert validated["row_count"] == 3
    assert validated["track_count"] == 2
    direct = zarr.open_group(
        str(archive), mode="r", zarr_format=3, use_consolidated=False
    )
    assert direct["analysis/track_kinematics_runs"].attrs["latest"] == (
        "offline/legacy_selected"
    )
    run = direct[plan.run_path]
    assert run.attrs["stage_selector_eligible"] is False
    assert "sample_valid" not in set(run.array_keys())
    np.testing.assert_array_equal(
        np.asarray(run["positions_mm"][:]),
        np.asarray(run["positions_px"][:]) * np.float32(0.25),
    )
    assert run["positions_mm"].attrs["units"] == "mm"
    assert run["speed_raw_mm"].attrs["units"] == "mm/s"
    assert run["acceleration_mm"].attrs["units"] == "mm/s^2"
    assert set(direct[PROVIDER_TRACK_MOTION_PARENT_PATH].attrs).isdisjoint(
        {"latest", "latest_complete", "latest_pending", "authoritative_run"}
    )


def test_validation_rejects_tampered_independent_validity(tmp_path) -> None:
    tracked = _tracked(tmp_path)
    prepared = prepare_provider_track_motion(
        tracked,
        fps=10.0,
        smooth_seconds=0.0,
        allow_pixel_only=True,
    )
    archive = tracked.source_authority.analysis_zarr_path
    plan = plan_provider_track_motion_run(
        archive,
        prepared,
        run_name="provider_motion_canary_tamper",
        scratch_root=tmp_path / "provider_motion_tamper_scratch",
    )
    publish_provider_track_motion_run(plan, keep_scratch=True)
    root = zarr.open_group(
        str(archive), mode="r+", zarr_format=3, use_consolidated=False
    )
    root[f"{plan.run_path}/linear_sample_valid"][0] = False

    with pytest.raises(ProviderTrackMotionError, match="stale"):
        validate_provider_track_motion_run(
            archive,
            plan.run_path,
            use_consolidated=False,
            expected_manifest_sha256=plan.manifest_sha256,
        )


def test_physical_authority_is_required_by_default(tmp_path) -> None:
    with pytest.raises(ProviderTrackMotionError, match="physical authority"):
        prepare_provider_track_motion(
            _tracked(tmp_path),
            fps=10.0,
            smooth_seconds=0.0,
        )


def test_publication_reopens_tracking_authority_and_fails_before_write(
    tmp_path,
) -> None:
    tracked = _tracked(tmp_path)
    prepared = prepare_provider_track_motion(
        tracked,
        fps=10.0,
        smooth_seconds=0.0,
        allow_pixel_only=True,
    )
    archive = tracked.source_authority.analysis_zarr_path
    plan = plan_provider_track_motion_run(
        archive,
        prepared,
        run_name="provider_motion_tracking_stale",
        scratch_root=tmp_path / "provider_motion_tracking_stale_scratch",
    )
    root = zarr.open_group(
        str(archive), mode="r+", zarr_format=3, use_consolidated=False
    )
    root[f"{tracked.tracking_run_path}/track_ids"][0] = np.int32(99)

    with pytest.raises(TrackingSourceHandleError, match="manifest"):
        publish_provider_track_motion_run(plan, keep_scratch=True)

    assert not plan.local_zarr.exists()
    assert not plan.target_run_path.exists()
