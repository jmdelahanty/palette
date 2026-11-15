from __future__ import annotations

import copy

import numpy as np
import pytest
import zarr

import fisheye.analysis_workflows.materializers.provider_track_motion as provider_writer
from fisheye.analysis_workflows.materializers.provider_track_motion import (
    PROVIDER_TRACK_MOTION_PAYLOAD_INTEGRITY_RECEIPT_ATTR,
    PROVIDER_TRACK_MOTION_PAYLOAD_VALIDATION_RECEIPT_ATTR,
    plan_provider_track_motion_run,
    prepare_provider_track_motion,
    publish_provider_track_motion_run,
)
from fisheye.analysis_workflows.provider_track_motion_source_handle import (
    PROVIDER_TRACK_MOTION_ATOMIC_COMPATIBILITY_RECEIPT_PROFILE,
    PROVIDER_TRACK_MOTION_NATIVE_RECEIPT_PROFILE,
    ProviderTrackMotionSourceHandleError,
    load_receipt_bound_provider_track_motion_source_handle,
    load_provider_track_motion_source_handle,
)
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from tests.unit.fisheye.test_provider_track_motion_publication import (
    _install_fake_physical_authority,
    _timed_tracked,
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


def _publish_timed_fixture(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    tracked, timing = _timed_tracked(tmp_path)
    prepared = prepare_provider_track_motion(
        tracked,
        fps=10.0,
        smooth_seconds=0.0,
        allow_pixel_only=True,
        temporal_authority=timing,
    )
    plan = plan_provider_track_motion_run(
        tracked.source_authority.analysis_zarr_path,
        prepared,
        run_name="provider_motion_timed_reader_fixture",
        scratch_root=tmp_path / "provider_motion_timed_reader_scratch",
    )
    publish_provider_track_motion_run(plan, keep_scratch=True)
    return tracked.source_authority.analysis_zarr_path, plan, timing


def _load(archive, plan, **kwargs):  # type: ignore[no-untyped-def]
    return load_provider_track_motion_source_handle(
        archive,
        plan.run_path,
        **kwargs,
    )


def _load_receipt_bound(archive, plan, **kwargs):  # type: ignore[no-untyped-def]
    return load_receipt_bound_provider_track_motion_source_handle(
        archive,
        plan.run_path,
        **kwargs,
    )


def test_receipt_bound_loader_preserves_eager_identity_and_reads_bounded_arrays(
    tmp_path,
    monkeypatch,
) -> None:
    archive, plan, timing = _publish_timed_fixture(tmp_path, monkeypatch)
    eager = _load(
        archive,
        plan,
        require_authoritative_timing=True,
    )
    deep_calls: list[object] = []

    def _forbid_deep_validation(*args, **kwargs):  # type: ignore[no-untyped-def]
        deep_calls.append((args, kwargs))
        raise AssertionError("receipt-bound load must not run the exhaustive validator")

    monkeypatch.setattr(provider_writer, "_validate_run_group", _forbid_deep_validation)

    handle = _load_receipt_bound(
        archive,
        plan,
        expected_manifest_sha256=plan.manifest_sha256,
        require_authoritative_timing=True,
    )

    assert deep_calls == []
    assert handle.receipt_profile == PROVIDER_TRACK_MOTION_NATIVE_RECEIPT_PROFILE
    assert handle.payload_integrity_receipt_sha256
    assert handle.payload_validation_receipt_sha256
    assert handle.verification_digest == eager.verification_digest
    assert handle.provider_manifest_sha256 == eager.provider_manifest_sha256
    assert handle.temporal_authority_sha256 == timing.sha256
    assert handle.metadata_evidence["timing_authority"]["profile"] == (
        "provider_bound_selected_immutable_clock_metadata_v1"
    )
    assert handle.available_arrays == tuple(sorted(eager.arrays))
    np.testing.assert_array_equal(handle.track_ids, eager.track_ids)
    np.testing.assert_array_equal(handle.track_row_offsets, eager.track_row_offsets)
    np.testing.assert_array_equal(
        handle.array_slice("positions_px", slice(1, 3)),
        eager.positions_px[1:3],
    )
    with pytest.raises(ValueError):
        handle.array_slice("positions_px", slice(0, 1))[0, 0] = 99.0
    with pytest.raises(ProviderTrackMotionSourceHandleError, match="contiguous"):
        handle.array_slice("positions_px", slice(0, 2, 2))

    handle.assert_current()
    assert deep_calls == []


def test_receipt_bound_loader_rejects_changed_clock_selector_generation(
    tmp_path,
    monkeypatch,
) -> None:
    archive, plan, _timing = _publish_timed_fixture(tmp_path, monkeypatch)
    clock_parent = zarr.open_group(
        str(archive / "analysis" / "acquisition_frame_clock_runs"),
        mode="r+",
        zarr_format=3,
        use_consolidated=False,
    )
    clock_parent.attrs["latest_complete"] = "another_clock"

    with pytest.raises(
        ProviderTrackMotionSourceHandleError,
        match="selector generation",
    ):
        _load_receipt_bound(
            archive,
            plan,
            require_authoritative_timing=True,
        )


def test_receipt_bound_loader_accepts_atomic_full_validation_compatibility(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        provider_writer,
        "_stamp_provider_track_motion_payload_receipts",
        lambda *_args, **_kwargs: {},
    )
    archive, plan, timing = _publish_timed_fixture(tmp_path, monkeypatch)

    handle = _load_receipt_bound(
        archive,
        plan,
        require_authoritative_timing=True,
        timing_authority=timing,
    )

    assert handle.receipt_profile == (
        PROVIDER_TRACK_MOTION_ATOMIC_COMPATIBILITY_RECEIPT_PROFILE
    )
    assert handle.payload_integrity_receipt_sha256 is None
    assert handle.payload_validation_receipt_sha256 is None
    assert handle.atomic_publication_receipt_sha256 == handle.receipt_digest


def test_receipt_bound_loader_rejects_stale_metadata_and_atomic_receipt(
    tmp_path,
    monkeypatch,
) -> None:
    archive, plan, timing = _publish_timed_fixture(tmp_path, monkeypatch)
    direct = zarr.open_group(
        str(archive), mode="r+", zarr_format=3, use_consolidated=False
    )
    run = direct[plan.run_path]
    receipt = copy.deepcopy(dict(run.attrs["cluster_output_staging"]))
    receipt["manifest_sha256"] = "0" * 64
    run.attrs["cluster_output_staging"] = receipt

    with pytest.raises(
        ProviderTrackMotionSourceHandleError,
        match="atomic publication receipt identity",
    ):
        _load_receipt_bound(
            archive,
            plan,
            require_authoritative_timing=True,
            timing_authority=timing,
        )


def test_receipt_bound_loader_rejects_incomplete_or_tampered_native_receipts(
    tmp_path,
    monkeypatch,
) -> None:
    archive, plan, timing = _publish_timed_fixture(tmp_path, monkeypatch)
    direct = zarr.open_group(
        str(archive), mode="r+", zarr_format=3, use_consolidated=False
    )
    run = direct[plan.run_path]
    original_integrity = copy.deepcopy(
        dict(run.attrs[PROVIDER_TRACK_MOTION_PAYLOAD_INTEGRITY_RECEIPT_ATTR])
    )

    del run.attrs[PROVIDER_TRACK_MOTION_PAYLOAD_INTEGRITY_RECEIPT_ATTR]
    with pytest.raises(
        ProviderTrackMotionSourceHandleError,
        match="receipt pair is incomplete",
    ):
        _load_receipt_bound(
            archive,
            plan,
            require_authoritative_timing=True,
            timing_authority=timing,
        )

    run.attrs[PROVIDER_TRACK_MOTION_PAYLOAD_INTEGRITY_RECEIPT_ATTR] = original_integrity
    validation = copy.deepcopy(
        dict(run.attrs[PROVIDER_TRACK_MOTION_PAYLOAD_VALIDATION_RECEIPT_ATTR])
    )
    validation["scientific_manifest"]["sha256"] = "0" * 64
    run.attrs[PROVIDER_TRACK_MOTION_PAYLOAD_VALIDATION_RECEIPT_ATTR] = validation
    with pytest.raises(
        ProviderTrackMotionSourceHandleError,
        match="payload receipt validation failed",
    ):
        _load_receipt_bound(
            archive,
            plan,
            require_authoritative_timing=True,
            timing_authority=timing,
        )


def test_receipt_bound_loader_rejects_changed_provider_namespace_generation(
    tmp_path,
    monkeypatch,
) -> None:
    archive, plan, timing = _publish_timed_fixture(tmp_path, monkeypatch)
    direct = zarr.open_group(
        str(archive), mode="r+", zarr_format=3, use_consolidated=False
    )
    direct[provider_writer.PROVIDER_TRACK_MOTION_PARENT_PATH].attrs[
        "post_publication_note"
    ] = "changed"
    consolidate_metadata_capture_expected_warnings(archive)

    with pytest.raises(
        ProviderTrackMotionSourceHandleError,
        match="namespace generation",
    ):
        _load_receipt_bound(
            archive,
            plan,
            require_authoritative_timing=True,
            timing_authority=timing,
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
            use_consolidated=False,
            require_authoritative_timing=True,
        )


def test_live_digest_bound_temporal_record_is_authoritative(
    tmp_path,
    monkeypatch,
) -> None:
    archive, plan, timing = _publish_timed_fixture(tmp_path, monkeypatch)

    handle = _load(
        archive,
        plan,
        require_authoritative_timing=True,
    )

    assert handle.temporal_authority_status == "bound_live_recording_timing_authority"
    assert handle.timing_is_authoritative is True
    assert handle.temporal_authority_record == timing.record
    assert handle.temporal_authority_sha256 == timing.sha256


def test_bound_temporal_record_rejects_stale_live_clock(
    tmp_path,
    monkeypatch,
) -> None:
    archive, plan, _timing = _publish_timed_fixture(tmp_path, monkeypatch)
    direct = zarr.open_group(
        str(archive), mode="r+", zarr_format=3, use_consolidated=False
    )
    direct.attrs["fps"] = 9.0

    with pytest.raises(
        ProviderTrackMotionSourceHandleError,
        match="recording timing",
    ):
        _load(
            archive,
            plan,
            use_consolidated=False,
            require_authoritative_timing=True,
        )
