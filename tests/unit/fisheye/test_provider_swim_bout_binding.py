from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from fisheye.analysis.swim_bout_frame_axis import build_frame_axis_contract
from fisheye.analysis.track_kinematics_io import (
    TRACK_KINEMATICS_PUBLICATION_PROFILE_SELECTOR_INELIGIBLE_CANARY_V1,
)
from fisheye.analysis_workflows.provider_swim_bout_binding import (
    PROVIDER_SWIM_BOUT_VALIDATION_PROFILE_CURRENT_STRICT_V1,
    ProviderSwimBoutBindingError,
    validate_provider_swim_bout_binding,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _fixture() -> tuple[SimpleNamespace, SimpleNamespace]:
    frames = np.asarray([10, 11, 13, 20, 21], dtype=np.int64)
    provider = SimpleNamespace(
        run_name="motion_v2",
        run_path="analysis/track_kinematics_runs/provider/motion_v2",
        provider_manifest_sha256="a" * 64,
        verification_digest="b" * 64,
        track_ids=np.asarray([0, 1], dtype=np.int64),
        track_row_offsets=np.asarray([0, 3, 5], dtype=np.int64),
        source_acquisition_frame_index=frames,
        physical_authority_sha256="c" * 64,
        temporal_authority_status="bound_live_recording_timing_authority",
        timing_is_authoritative=True,
        temporal_authority_record={"nominal_fps": 100.0},
        computation_record={"parameters": {"fps": 100.0}},
    )
    selected_frames = frames[:3]
    frame_contract = build_frame_axis_contract(
        selected_frames,
        authoritative_path=(
            "analysis/track_kinematics_runs/provider/motion_v2/"
            "source_acquisition_frame_index"
        ),
        source_track_kinematics_run="motion_v2",
        track_id=0,
        source_track_motion_manifest_sha256="a" * 64,
    )
    authority = {
        "schema_id": "palette.provider_track_motion_read_authority",
        "schema_version": 1,
        "run_ref": "/analysis/track_kinematics_runs/provider/motion_v2",
        "track_ref": "/analysis/track_kinematics_runs/provider/motion_v2",
        "track_id": 0,
        "track_row_start": 0,
        "track_row_stop": 3,
        "motion_manifest_ref": (
            "/analysis/track_kinematics_runs/provider/motion_v2"
            "@provider_track_motion_manifest"
        ),
        "motion_manifest_sha256": "a" * 64,
        "positions_px_ref": (
            "/analysis/track_kinematics_runs/provider/motion_v2/positions_px"
        ),
        "positions_mm_ref": (
            "/analysis/track_kinematics_runs/provider/motion_v2/positions_mm"
        ),
        "track_sample_key_ref": (
            "/analysis/track_kinematics_runs/provider/motion_v2/track_sample_key"
        ),
        "source_acquisition_frame_index_ref": (
            "/analysis/track_kinematics_runs/provider/motion_v2/"
            "source_acquisition_frame_index"
        ),
        "provider_verification_digest": "b" * 64,
        "physical_authority_sha256": "c" * 64,
        "temporal_authority_status": "bound_live_recording_timing_authority",
        "timing_is_authoritative": True,
    }
    tables = SimpleNamespace(
        run_name="bouts_v2",
        run_path="analysis/swim_bout_runs/bouts_v2",
        run_attrs={
            "source_track_kinematics_scope": "provider",
            "source_track_kinematics_run": "motion_v2",
            "source_track_kinematics_publication_profile_id": (
                TRACK_KINEMATICS_PUBLICATION_PROFILE_SELECTOR_INELIGIBLE_CANARY_V1
            ),
            "track_id": 0,
            "fps": 100.0,
            "source_track_motion_manifest_sha256": "a" * 64,
            "source_track_motion_authority": authority,
            "frame_axis_contract": frame_contract,
            "frame_axis_contract_sha256": canonical_json_sha256(frame_contract),
            "lineage_hash": "d" * 64,
        },
        candidate=SimpleNamespace(candidate_id=0),
        signal=SimpleNamespace(signal_id=4, speed_level="speed_exponential"),
        bouts=np.zeros(0, dtype=[("bout_id", np.int64)]),
        peak_events=np.zeros(0, dtype=[("peak_id", np.int64)]),
        inter_bout_intervals=np.zeros(0, dtype=[("interval_id", np.int64)]),
        inter_bout_interval_histogram=np.zeros(
            0,
            dtype=[("bin_id", np.int64)],
        ),
        global_metrics=np.zeros(0, dtype=[("metric_id", np.int64)]),
        trials=np.zeros(0, dtype=[("trial_id", np.int64)]),
        bout_points=np.zeros(0, dtype=[("point_id", np.int64)]),
        series={"frame_indices": selected_frames.copy()},
    )
    return provider, tables


def _recording_root() -> SimpleNamespace:
    return SimpleNamespace(attrs={"fps": 100.0})


def test_provider_swim_bout_binding_accepts_one_exact_whole_track() -> None:
    provider, tables = _fixture()

    binding, lineage, frame_sha256 = validate_provider_swim_bout_binding(
        tables,
        provider=provider,
        track_id=0,
        rows=slice(0, 3),
        recording_root=_recording_root(),
        validation_profile=(PROVIDER_SWIM_BOUT_VALIDATION_PROFILE_CURRENT_STRICT_V1),
    )

    assert binding["schema_id"] == "palette.selector_ineligible_swim_bout_binding.v1"
    assert binding["run_path"] == "analysis/swim_bout_runs/bouts_v2"
    assert binding["source_track_motion_manifest_sha256"] == "a" * 64
    assert binding["source_track_motion_verification_digest"] == "b" * 64
    assert binding["track_row_start"] == 0
    assert binding["track_row_stop"] == 3
    assert binding["sha256"] == canonical_json_sha256(
        {key: value for key, value in binding.items() if key != "sha256"}
    )
    assert lineage == "d" * 64
    assert frame_sha256 == tables.run_attrs["frame_axis_contract"]["content_sha256"]


def test_provider_swim_bout_binding_accepts_receipt_bound_timing_without_root() -> None:
    provider, tables = _fixture()

    binding, _lineage, _frame_sha256 = validate_provider_swim_bout_binding(
        tables,
        provider=provider,
        track_id=0,
        rows=slice(0, 3),
        validation_profile=(PROVIDER_SWIM_BOUT_VALIDATION_PROFILE_CURRENT_STRICT_V1),
    )

    assert binding["run_name"] == "bouts_v2"


def test_provider_swim_bout_binding_rootless_timing_rejects_local_fps_tamper() -> None:
    provider, tables = _fixture()
    tables.run_attrs["fps"] = 99.0

    with pytest.raises(ProviderSwimBoutBindingError, match="FPS"):
        validate_provider_swim_bout_binding(
            tables,
            provider=provider,
            track_id=0,
            validation_profile=(
                PROVIDER_SWIM_BOUT_VALIDATION_PROFILE_CURRENT_STRICT_V1
            ),
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("source_track_kinematics_scope",), "offline", "provider motion"),
        (("source_track_kinematics_run",), "other", "identities"),
        (("source_track_motion_manifest_sha256",), "e" * 64, "manifest"),
        (("source_track_motion_authority", "run_ref"), "/wrong", "run_ref"),
        (
            ("source_track_motion_authority", "provider_verification_digest"),
            "f" * 64,
            "provider_verification_digest",
        ),
        (("source_track_motion_authority", "track_row_stop"), 2, "track_row_stop"),
        (("frame_axis_contract", "content_sha256"), "0" * 64, "frame axis"),
        (("frame_axis_contract_sha256",), "1" * 64, "contract digest"),
        (("fps",), 99.0, "FPS"),
        (("lineage_hash",), "malformed", "lineage"),
    ),
)
def test_provider_swim_bout_binding_rejects_tampered_or_wrong_source_fields(
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    provider, original = _fixture()
    tables = copy.deepcopy(original)
    target = tables.run_attrs
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value

    with pytest.raises(ProviderSwimBoutBindingError, match=message):
        validate_provider_swim_bout_binding(
            tables,
            provider=provider,
            track_id=0,
            recording_root=_recording_root(),
            validation_profile=(
                PROVIDER_SWIM_BOUT_VALIDATION_PROFILE_CURRENT_STRICT_V1
            ),
        )


def test_provider_swim_bout_binding_rejects_partial_or_wrong_track_slice() -> None:
    provider, tables = _fixture()

    with pytest.raises(ProviderSwimBoutBindingError, match="whole provider track"):
        validate_provider_swim_bout_binding(
            tables,
            provider=provider,
            track_id=0,
            rows=slice(1, 3),
        )

    with pytest.raises(ProviderSwimBoutBindingError, match="track identities"):
        validate_provider_swim_bout_binding(
            tables,
            provider=provider,
            track_id=1,
        )


def test_provider_swim_bout_binding_rejects_noncanonical_bout_path() -> None:
    provider, tables = _fixture()
    tables.run_path = "analysis/swim_bout_runs/other"

    with pytest.raises(ProviderSwimBoutBindingError, match="run path"):
        validate_provider_swim_bout_binding(
            tables,
            provider=provider,
            track_id=0,
            recording_root=_recording_root(),
            validation_profile=(
                PROVIDER_SWIM_BOUT_VALIDATION_PROFILE_CURRENT_STRICT_V1
            ),
        )


def test_provider_swim_bout_binding_preserves_legacy_provider_compatibility() -> None:
    provider, tables = _fixture()
    provider.physical_authority_sha256 = None
    provider.temporal_authority_status = "compatibility_caller_fps_only"
    provider.timing_is_authoritative = False
    tables.run_attrs.pop("source_track_kinematics_publication_profile_id")
    tables.run_attrs.pop("frame_axis_contract_sha256")
    tables.run_attrs.pop("fps")
    tables.run_attrs["source_track_motion_authority"] = {
        key: tables.run_attrs["source_track_motion_authority"][key]
        for key in (
            "motion_manifest_sha256",
            "provider_verification_digest",
            "track_id",
            "track_row_start",
            "track_row_stop",
        )
    }
    tables.run_attrs["frame_axis_contract"] = {
        "content_sha256": tables.run_attrs["frame_axis_contract"]["content_sha256"]
    }

    binding, _lineage, _frame = validate_provider_swim_bout_binding(
        tables,
        provider=provider,
        track_id=0,
    )

    assert binding["source_track_motion_manifest_sha256"] == "a" * 64
    with pytest.raises(ProviderSwimBoutBindingError, match="physical authority"):
        validate_provider_swim_bout_binding(
            tables,
            provider=provider,
            track_id=0,
            validation_profile=(
                PROVIDER_SWIM_BOUT_VALIDATION_PROFILE_CURRENT_STRICT_V1
            ),
        )
