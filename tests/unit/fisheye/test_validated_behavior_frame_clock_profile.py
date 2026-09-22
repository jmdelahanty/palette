from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fisheye.analysis_workflows.validated_behavior_cohort_adapters import sha256_file
from fisheye.analytics_exports.validated_behavior_core_behavior_adapters import (
    _acquisition_frame_clock_samples,
    _recording_clock_metadata,
)
from fisheye.analytics_exports.validated_behavior_bout_kinematics_contracts import (
    BOUT_KINEMATICS_CAPABILITY_KEYS,
    BOUT_KINEMATICS_EXPORT_PROFILE_ID,
)
from fisheye.analytics_exports.validated_behavior_frame_clock import (
    ValidatedBehaviorFrameClockError,
    acquisition_frame_clock_projection_contract,
    bind_validated_behavior_frame_clock,
)
from fisheye.analytics_exports.validated_behavior_frame_clock_contracts import (
    ACQUISITION_FRAME_CLOCK_CAPABILITY,
    ACQUISITION_FRAME_CLOCK_SAMPLES_TABLE,
    FRAME_CLOCK_CAPABILITY_KEYS,
    FRAME_CLOCK_EXPORT_PROFILE_ID,
    RECORDING_CLOCK_METADATA_TABLE,
)
from fisheye.analytics_exports.validated_behavior_profiles import (
    resolve_validated_behavior_profile,
)
from fisheye.analytics_exports.validated_behavior_contracts import (
    validate_table_specs,
)
from fisheye.shared.acquisition_frame_clock import AcquisitionFrameClockSource
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def test_frame_clock_profile_adds_two_grains_without_changing_bout_v1() -> None:
    old = resolve_validated_behavior_profile(BOUT_KINEMATICS_EXPORT_PROFILE_ID)
    new = resolve_validated_behavior_profile(FRAME_CLOCK_EXPORT_PROFILE_ID)

    assert len(BOUT_KINEMATICS_CAPABILITY_KEYS) == 7
    assert FRAME_CLOCK_CAPABILITY_KEYS == (
        *BOUT_KINEMATICS_CAPABILITY_KEYS,
        ACQUISITION_FRAME_CLOCK_CAPABILITY,
    )
    assert set(new.table_specs) == set(old.table_specs) | {
        RECORDING_CLOCK_METADATA_TABLE,
        ACQUISITION_FRAME_CLOCK_SAMPLES_TABLE,
    }
    assert set(new.row_extractors()) == set(old.row_extractors()) | {
        RECORDING_CLOCK_METADATA_TABLE,
        ACQUISITION_FRAME_CLOCK_SAMPLES_TABLE,
    }
    assert len(validate_table_specs(old.table_specs)) == 11
    assert len(validate_table_specs(new.table_specs)) == 13


def _write_raw_clock(recording_dir: Path) -> Path:
    frame_index = recording_dir / "recording_frame_index.parquet"
    pq.write_table(
        pa.table(
            {
                "recording_id": ["recording-a"] * 3,
                "session_id": ["session-a"] * 3,
                "camera_serial": ["2010093", "2010093", "2010093"],
                "recording_frame_id": [100, 101, 102],
                "parent_frame_index": [0, 1, 2],
                "timestamp": [1_000, 2_000, 3_000],
                "timestamp_sys": [101_000, 102_000, 103_000],
            }
        ),
        frame_index,
    )
    return frame_index


def _binding_fixture(tmp_path: Path) -> tuple[SimpleNamespace, SimpleNamespace, Path]:
    recording_dir = tmp_path / "recording-a"
    recording_dir.mkdir()
    frame_index = _write_raw_clock(recording_dir)
    manifest = {
        "recording_id": "recording-a",
        "camera_id": "2010093",
        "orange_session_id": "session-a",
        "session_start_iso8601_utc": "2026-08-06T23:13:38Z",
    }
    (recording_dir / "recording_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    metadata = {
        "layout": "clipped_video_collection",
        "camera_id": "2010093",
        "fps": 30.0,
        "total_frames": 3,
        "collection": {
            "recording_frame_index": {
                "relative_path": frame_index.name,
                "sha256": sha256_file(frame_index),
            }
        },
    }
    record = SimpleNamespace(
        recording_id="recording-a",
        camera_id="2010093",
        source_total_frames=3,
        source_video_metadata=metadata,
        source_video_metadata_sha256=canonical_json_sha256(metadata),
    )
    acquisition = SimpleNamespace(
        record=record,
        record_ref="/analysis/acquisition_camera_frames/2010093@record",
        record_sha256="a" * 64,
    )
    root = SimpleNamespace(
        attrs={
            "recording_path": str(recording_dir),
            "recording_id": "recording-a",
            "session_id": "session-a",
        }
    )
    return root, acquisition, frame_index


def test_clock_binding_seals_raw_digest_session_and_join_limits(
    tmp_path: Path,
) -> None:
    root, acquisition, frame_index = _binding_fixture(tmp_path)

    bound = bind_validated_behavior_frame_clock(
        root,
        analysis_zarr=tmp_path / "recording-a" / "analysis.zarr",
        expected_recording_id="recording-a",
        acquisition=acquisition,
    )

    binding = bound.source_binding
    assert binding["session_id"] == "session-a"
    assert binding["frame_clock_source_file_sha256"] == sha256_file(frame_index)
    assert binding["acquisition_camera_frame_sha256"] == "a" * 64
    assert binding["within_session_alignment_status"] == (
        "recording_local_only_no_validated_shared_clock"
    )
    assert binding["cross_session_alignment_status"].startswith("not_validated")
    assert binding["equal_frame_rate_alignment_valid"] is False
    projection = acquisition_frame_clock_projection_contract()
    assert projection["within_session_cross_camera_join"]["coordinate"] == (
        "camera_timestamp_ns"
    )
    assert projection["equal_frame_rate_alignment_valid"] is False


def test_clock_binding_rejects_frame_index_that_differs_from_acquisition_digest(
    tmp_path: Path,
) -> None:
    root, acquisition, _frame_index = _binding_fixture(tmp_path)
    acquisition.record.source_video_metadata["collection"]["recording_frame_index"][
        "sha256"
    ] = ("f" * 64)

    with pytest.raises(
        ValidatedBehaviorFrameClockError,
        match="digest differs from acquisition metadata",
    ):
        bind_validated_behavior_frame_clock(
            root,
            analysis_zarr=tmp_path / "recording-a" / "analysis.zarr",
            expected_recording_id="recording-a",
            acquisition=acquisition,
        )


def test_clock_binding_rejects_frame_index_session_mismatch(tmp_path: Path) -> None:
    root, acquisition, frame_index = _binding_fixture(tmp_path)
    table = pq.read_table(frame_index)
    columns = table.to_pydict()
    columns["session_id"][1] = "another-session"
    pq.write_table(pa.table(columns), frame_index)
    acquisition.record.source_video_metadata["collection"]["recording_frame_index"][
        "sha256"
    ] = sha256_file(frame_index)

    with pytest.raises(
        ValidatedBehaviorFrameClockError,
        match="row identity disagrees",
    ):
        bind_validated_behavior_frame_clock(
            root,
            analysis_zarr=tmp_path / "recording-a" / "analysis.zarr",
            expected_recording_id="recording-a",
            acquisition=acquisition,
        )


class _ProjectionContext:
    row_group_rows = 2

    def __init__(self) -> None:
        surfaces = {
            "camera_timestamp_ns": {
                "clock_domain": "camera_hardware_ptp_clock",
                "time_reference_kind": "absolute_epoch",
                "origin": "1970-01-01T00:00:00_TAI",
                "timescale": "IEEE-1588_PTP_TAI",
                "semantic_status": "inferred_from_recording_evidence_not_sdk_declared",
            },
            "system_timestamp_ns": {
                "clock_domain": "host_CLOCK_REALTIME",
                "time_reference_kind": "absolute_epoch",
                "origin": "1970-01-01T00:00:00_UTC",
                "timescale": "POSIX_UTC_excluding_leap_seconds",
                "semantic_status": "producer_declared_by_orange_code",
            },
        }
        self.source = AcquisitionFrameClockSource(
            source_path=Path("/raw/recording_frame_index.parquet"),
            source_kind="recording_frame_index_parquet",
            source_locator="recording_frame_index.parquet",
            camera_id="2010093",
            recording_frame_id=np.asarray([9, 10, 11], dtype=np.int64),
            parent_frame_index=np.asarray([0, 1, 2], dtype=np.int64),
            camera_timestamp_ns=np.asarray([100, 200, 300], dtype=np.int64),
            system_timestamp_ns=np.asarray([110, 210, 310], dtype=np.int64),
            camera_timestamp_valid=np.asarray([True, True, False]),
            system_timestamp_valid=np.asarray([True, True, True]),
            clock_surfaces=surfaces,
            clock_semantic_evidence={},
        )
        self.binding = {
            "payload_sha256": "b" * 64,
            "session_id": "session-a",
            "session_start_iso8601_utc": "2026-08-06T23:13:38Z",
            "camera_id": "2010093",
            "row_count": 3,
            "raw_recording_path": "/raw",
            "frame_clock_source_path": "/raw/recording_frame_index.parquet",
            "frame_clock_source_file_sha256": "c" * 64,
            "recording_manifest_path": "/raw/recording_manifest.json",
            "recording_manifest_file_sha256": "d" * 64,
            "ptp_sync_summary_path": None,
            "ptp_sync_summary_file_sha256": None,
            "acquisition_camera_frame_sha256": "e" * 64,
            "source_video_metadata_sha256": "f" * 64,
            "acquisition_frame_clock_source_sha256": "1" * 64,
            "clock_semantics": {"clock_surfaces": surfaces},
            "clock_semantics_sha256": "2" * 64,
            "within_session_alignment_status": "supported_by_inferred_ptp_evidence",
            "cross_session_alignment_status": (
                "not_validated_requires_traceable_absolute_clock_or_external_anchor"
            ),
            "equal_frame_rate_alignment_valid": False,
        }
        self.projection = {"payload_sha256": "3" * 64}
        self.bound = SimpleNamespace(
            acquisition_frame_clock=SimpleNamespace(
                source=self.source,
                source_binding=self.binding,
                projection_contract=self.projection,
            )
        )

    def capability_binding(self, _capability_id: str) -> dict[str, object]:
        return {
            "source_binding": self.binding,
            "projection_contract": self.projection,
        }

    def common_columns(self, count: int) -> dict[str, list[str]]:
        return {
            name: [name] * count
            for name in (
                "export_run_id",
                "recording_id",
                "membership_member_sha256",
                "bundle_set_member_sha256",
                "bundle_record_sha256",
                "cross_grain_join_authority_sha256",
            )
        }


def test_clock_projection_emits_exact_metadata_and_bounded_sample_batches() -> None:
    context = _ProjectionContext()
    metadata, reason = _recording_clock_metadata(context)
    batches = list(_acquisition_frame_clock_samples(context).batches)
    profile = resolve_validated_behavior_profile(FRAME_CLOCK_EXPORT_PROFILE_ID)

    assert reason is None
    assert set(metadata[0]) == {
        item.name
        for item in profile.table_specs[RECORDING_CLOCK_METADATA_TABLE].contract.fields
    }
    assert metadata[0]["session_id"] == "session-a"
    assert metadata[0]["equal_frame_rate_alignment_valid"] is False
    assert [len(batch["recording_frame_id"]) for batch in batches] == [2, 1]
    assert np.array_equal(
        batches[1]["source_acquisition_frame_index"],
        np.asarray([2], dtype=np.int64),
    )
    assert set(batches[0]) == {
        item.name
        for item in profile.table_specs[
            ACQUISITION_FRAME_CLOCK_SAMPLES_TABLE
        ].contract.fields
    }
