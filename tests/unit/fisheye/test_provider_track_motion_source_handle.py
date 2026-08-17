from __future__ import annotations

import copy

import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows.materializers.provider_track_motion import (
    plan_provider_track_motion_run,
    prepare_provider_track_motion,
    publish_provider_track_motion_run,
)
from fisheye.analysis_workflows.provider_track_motion_source_handle import (
    ProviderTrackMotionSourceHandleError,
    load_provider_track_motion_source_handle,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from tests.unit.fisheye.test_provider_track_motion_publication import (
    _install_fake_physical_authority,
    _tracked,
)


def _publish_fixture(tmp_path, monkeypatch, *, physical: bool = False):  # type: ignore[no-untyped-def]
    if physical:
        _install_fake_physical_authority(monkeypatch)
    tracked = _tracked(tmp_path)
    prepared = prepare_provider_track_motion(
        tracked,
        fps=10.0,
        smooth_seconds=0.0,
        allow_pixel_only=not physical,
    )
    plan = plan_provider_track_motion_run(
        tracked.source_authority.analysis_zarr_path,
        prepared,
        run_name="provider_motion_reader_fixture",
        scratch_root=tmp_path / "provider_motion_reader_scratch",
    )
    publish_provider_track_motion_run(plan, keep_scratch=True)
    return tracked.source_authority.analysis_zarr_path, plan


def _load(archive, plan, **kwargs):  # type: ignore[no-untyped-def]
    return load_provider_track_motion_source_handle(
        archive,
        plan.run_path,
        **kwargs,
    )


def test_reads_exact_current_phase3_fixture_as_read_only_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    archive, plan = _publish_fixture(tmp_path, monkeypatch)

    handle = _load(archive, plan)

    assert handle.analysis_zarr_path == archive.resolve()
    assert handle.source_path == archive.resolve()
    assert handle.run_path == plan.run_path
    assert handle.run_name == "provider_motion_reader_fixture"
    assert handle.row_count == 3
    assert handle.track_count == 2
    assert handle.per_second_count == 2
    assert handle.selector_eligible is False
    assert handle.provider_manifest_sha256 == plan.manifest_sha256
    assert handle.source_authority_sha256
    assert handle.tracked_input_sha256
    assert handle.computation_sha256
    assert handle.physical_authority_status == ("omitted_explicit_pixel_only_canary")
    assert handle.temporal_authority_status == "compatibility_caller_fps_only"
    assert handle.timing_is_authoritative is False
    assert handle.temporal_authority_record is None
    assert "sample_valid" not in handle.arrays
    assert {
        "position_source_valid",
        "body_frame_source_valid",
        "linear_sample_valid",
        "angular_sample_valid",
        "transition_valid",
        "linear_sample_reason_code",
        "angular_sample_reason_code",
        "transition_reason_code",
        "time_seconds",
        "delta_seconds",
        "source_provider_row_index",
        "source_position_row_index",
        "source_body_frame_row_index",
        "source_tracking_row_index",
    }.issubset(handle.arrays)
    np.testing.assert_array_equal(
        handle.position_source_valid, handle.arrays["position_source_valid"]
    )
    np.testing.assert_array_equal(
        handle.linear_sample_valid, handle.arrays["linear_sample_valid"]
    )
    assert not np.shares_memory(
        handle.array("positions_px"), handle.array("time_seconds")
    )
    assert handle.array("linear_sample_valid") is not handle.array(
        "angular_sample_valid"
    )
    with pytest.raises(ValueError):
        handle.array("positions_px")[0, 0] = 99.0
    with pytest.raises((TypeError, AttributeError)):
        handle.arrays["positions_px"] = handle.array("positions_px")  # type: ignore[index]
    with pytest.raises((TypeError, AttributeError)):
        handle.provider_manifest["payload"] = {}  # type: ignore[index]

    handle.assert_verified()


def test_reads_optional_physical_arrays_without_changing_lineage(
    tmp_path,
    monkeypatch,
) -> None:
    archive, plan = _publish_fixture(tmp_path, monkeypatch, physical=True)

    handle = _load(archive, plan)

    assert handle.physical_authority_status == "bound"
    assert handle.physical_authority_record is not None
    assert handle.physical_authority_sha256
    assert "positions_mm" in handle.arrays
    np.testing.assert_array_equal(
        handle.array("positions_mm"),
        handle.array("positions_px") * np.float32(0.25),
    )
    np.testing.assert_array_equal(
        handle.array("source_provider_row_index"),
        np.asarray([0, 1, 2], dtype=np.int64),
    )


@pytest.mark.parametrize(
    "run_path",
    [
        "analysis/track_kinematics_runs/provider/latest",
        "analysis/track_kinematics_runs/provider/fallback",
        "analysis/track_kinematics_runs/provider/provider_motion_reader_fixture/extra",
        "analysis/track_kinematics_runs/provider",
        "analysis/track_kinematics_runs/provider/../provider_motion_reader_fixture",
    ],
)
def test_rejects_selectors_fallbacks_and_ambiguous_paths(
    tmp_path,
    monkeypatch,
    run_path: str,
) -> None:
    archive, _plan = _publish_fixture(tmp_path, monkeypatch)

    with pytest.raises(ProviderTrackMotionSourceHandleError, match="run_path"):
        load_provider_track_motion_source_handle(
            archive,
            run_path,
            use_consolidated=False,
        )


def test_rejects_selector_attrs_in_provider_namespace(
    tmp_path,
    monkeypatch,
) -> None:
    archive, plan = _publish_fixture(tmp_path, monkeypatch)
    root = zarr.open_group(
        str(archive), mode="r+", zarr_format=3, use_consolidated=False
    )
    root["analysis/track_kinematics_runs/provider"].attrs["latest"] = plan.run_name

    with pytest.raises(ProviderTrackMotionSourceHandleError, match="selector"):
        _load(archive, plan, use_consolidated=False)


def test_rejects_tampered_manifest_and_array_digest(
    tmp_path,
    monkeypatch,
) -> None:
    archive, plan = _publish_fixture(tmp_path, monkeypatch)
    root = zarr.open_group(
        str(archive), mode="r+", zarr_format=3, use_consolidated=False
    )
    run = root[plan.run_path]

    manifest = copy.deepcopy(dict(run.attrs["provider_track_motion_manifest"]))
    manifest["payload"]["arrays"][0]["sha256"] = "0" * 64
    run.attrs["provider_track_motion_manifest"] = manifest
    with pytest.raises(ProviderTrackMotionSourceHandleError, match="digest"):
        _load(archive, plan, use_consolidated=False)

    # Restore the immutable fixture by rebuilding the publication in a separate
    # archive, then alter only one payload array without changing its manifest.
    archive2, plan2 = _publish_fixture(tmp_path / "array", monkeypatch)
    root2 = zarr.open_group(
        str(archive2), mode="r+", zarr_format=3, use_consolidated=False
    )
    values = np.asarray(root2[f"{plan2.run_path}/positions_px"][:])
    root2[f"{plan2.run_path}/positions_px"][...] = values + np.float32(1.0)
    with pytest.raises(ProviderTrackMotionSourceHandleError, match="stale"):
        _load(archive2, plan2, use_consolidated=False)


def test_authoritative_timing_is_required_for_phase4_consumers(
    tmp_path,
    monkeypatch,
) -> None:
    archive, plan = _publish_fixture(tmp_path, monkeypatch)

    with pytest.raises(
        ProviderTrackMotionSourceHandleError,
        match="authoritative temporal authority",
    ):
        _load(
            archive,
            plan,
            require_authoritative_timing=True,
        )


def test_digest_bound_temporal_record_is_not_upgraded_without_clock_revalidation(
    tmp_path,
    monkeypatch,
) -> None:
    archive, plan = _publish_fixture(tmp_path, monkeypatch)
    root = zarr.open_group(
        str(archive), mode="r+", zarr_format=3, use_consolidated=False
    )
    run = root[plan.run_path]
    manifest = copy.deepcopy(dict(run.attrs["provider_track_motion_manifest"]))
    temporal_record = {
        "schema_id": "palette.source_row_temporal_authority",
        "schema_version": 1,
        "record_ref": "/analysis/detection@source_row_temporal_authority",
        "source_frame_index_sha256": "1" * 64,
    }
    computation = copy.deepcopy(manifest["payload"]["computation"]["record"])
    computation["temporal_authority"] = {
        "record": temporal_record,
        "sha256": canonical_json_sha256(temporal_record),
    }
    manifest["payload"]["computation"] = {
        "record": computation,
        "sha256": canonical_json_sha256(computation),
    }
    manifest["payload_digest"] = canonical_json_sha256(manifest["payload"])
    run.attrs["provider_track_motion_manifest"] = manifest
    run.attrs["provider_track_motion_manifest_sha256"] = manifest["payload_digest"]

    handle = _load(archive, plan, use_consolidated=False)

    assert handle.temporal_authority_status == (
        "bound_record_unverified_against_source_clock"
    )
    assert handle.timing_is_authoritative is False
    assert handle.temporal_authority_sha256 == canonical_json_sha256(temporal_record)

    with pytest.raises(
        ProviderTrackMotionSourceHandleError,
        match="authoritative temporal authority",
    ):
        _load(
            archive,
            plan,
            use_consolidated=False,
            require_authoritative_timing=True,
        )
