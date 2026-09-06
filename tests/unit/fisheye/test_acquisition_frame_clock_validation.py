"""Preservation and adversarial tests for ingestion, without a Zarr backend."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from fisheye.shared import acquisition_frame_clock as clock


def _source(tmp_path: Path) -> clock.AcquisitionFrameClockSource:
    return clock.AcquisitionFrameClockSource(
        source_path=tmp_path / "clock.csv",
        source_kind="orange_camera_metadata_csv",
        source_locator="clock.csv",
        camera_id="2010093",
        recording_frame_id=np.array([1, 2, 3], dtype=np.int64),
        parent_frame_index=np.array([0, 1, 2], dtype=np.int64),
        camera_timestamp_ns=np.array([100, 110, 120], dtype=np.int64),
        system_timestamp_ns=np.array([200, 210, 220], dtype=np.int64),
        camera_timestamp_valid=np.ones(3, dtype=bool),
        system_timestamp_valid=np.ones(3, dtype=bool),
        clock_surfaces={},
        clock_semantic_evidence={},
    )


def _ptp_semantics(tmp_path, monkeypatch, status, field="ptp_status"):
    summary = {
        "sync": {"camera_sync_enabled": True, "mode": "ptp_local"},
        "cameras": {
            "2010093": {
                "sync_camera_enabled": True,
                "ptp_register_reads": 8,
                "ptp_offset_ns": {"samples": 8, "min": 400, "max": 800},
                "latch_minus_frame_ns": {"samples": 8, "min": 8, "max": 10},
            }
        },
    }
    if status is not None:
        summary["cameras"]["2010093"][field] = status
    monkeypatch.setattr(clock, "_safe_json_object", lambda _: summary)
    summary_path = tmp_path / "ptp_sync_summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(clock, "_clock_summary_path", lambda _: summary_path)
    return clock._clock_semantics(
        tmp_path,
        camera_id="2010093",
        camera=np.array([37_000_000_100, 37_000_000_110], dtype=np.int64),
        camera_valid=np.ones(2, dtype=bool),
        system=np.array([100, 110], dtype=np.int64),
        system_valid=np.ones(2, dtype=bool),
    )


@pytest.mark.parametrize("field", ["ptp_status", "ptp_state", "gev_ieee1588_status"])
@pytest.mark.parametrize(
    "status",
    [
        "unlocked",
        "unsynchronized",
        "not synchronized",
        "not locked",
        "not slave",
        "master disabled",
        "slave_fault",
        "faulty",
        "listening",
    ],
)
def test_ptp_negative_and_unknown_states_do_not_establish_epoch(
    tmp_path,
    monkeypatch,
    field,
    status,
):
    surfaces, evidence = _ptp_semantics(tmp_path, monkeypatch, status, field)
    assert evidence["explicit_ptp_status"] == status
    assert evidence["explicit_ptp_status_does_not_contradict_synchronization"] is False
    assert evidence["camera_ptp_semantics_inferred"] is False
    assert surfaces["camera_timestamp_ns"]["time_reference_kind"] == (
        "device_defined_unknown_epoch"
    )


@pytest.mark.parametrize(
    "status", [None, "", "slave", "master", "locked", "synchronized", " SLAVE "]
)
def test_ptp_supported_statuses_preserve_existing_inference(
    tmp_path, monkeypatch, status
):
    surfaces, evidence = _ptp_semantics(tmp_path, monkeypatch, status)
    assert evidence["camera_ptp_semantics_inferred"] is True
    assert surfaces["camera_timestamp_ns"]["timescale"] == "IEEE-1588_PTP_TAI"


def test_valid_clock_record_preserves_golden_digest(tmp_path, monkeypatch):
    surfaces, evidence = _ptp_semantics(tmp_path, monkeypatch, "slave")
    # Freeze only external stat evidence; exercise the real record/array grammar.
    evidence.pop("ptp_sync_summary")
    source = _source(tmp_path)
    source = replace(
        source, **{name: getattr(source, name)[:2] for name in clock._ARRAY_NAMES}
    )
    source = replace(
        source,
        camera_timestamp_ns=np.array([37_000_000_100, 37_000_000_110]),
        system_timestamp_ns=np.array([100, 110]),
        clock_surfaces=surfaces,
        clock_semantic_evidence=evidence,
    )
    monkeypatch.setattr(
        clock,
        "_source_evidence",
        lambda _: {
            "kind": "orange_camera_metadata_csv",
            "locator": "clock.csv",
            "size_bytes": 123,
            "mtime_ns": 456,
        },
    )
    clock._validate_source(source, expected_frame_count=2)
    assert clock.acquisition_frame_clock_sha256(clock._build_record(source)) == (
        "621faa07511174f0bbc02f0e6703430061b8710cebcb3c2451a7ed7ffc204771"
    )


def _clock_table(**changes):
    columns = {
        "camera_serial": ["2010093"] * 3,
        "recording_frame_id": [1, 2, 3],
        "parent_frame_index": [0, 1, 2],
        "timestamp": [100, 110, 120],
        "timestamp_sys": [200, 210, 220],
    }
    columns.update(changes)
    return pa.table(columns)


def _load_table(tmp_path, monkeypatch, **changes):
    table = _clock_table(**changes)
    monkeypatch.setattr(clock.pq, "read_table", lambda *a, **kw: table)
    source = clock._load_parquet_source(
        tmp_path / "index.parquet",
        recording_dir=tmp_path,
        camera_id="2010093",
    )
    return clock._validate_source(source, expected_frame_count=3)


@pytest.mark.parametrize(
    "field", ["recording_frame_id", "parent_frame_index", "timestamp", "timestamp_sys"]
)
def test_public_import_rejects_fractional_parquet_before_any_publication(
    tmp_path, field
):
    clock.pq.write_table(
        _clock_table(**{field: [0.9, 1.9, 2.9]}),
        tmp_path / "recording_frame_index.parquet",
    )
    # No group API is needed: invalid input must fail before reaching a writer.
    root = SimpleNamespace(attrs={})
    with pytest.raises(clock.AcquisitionFrameClockError, match=field):
        clock.import_acquisition_frame_clock(
            root,
            recording_dir=tmp_path,
            camera_id="2010093",
            video_path=tmp_path / "absent_parent.mp4",
            expected_frame_count=3,
        )
    assert root.attrs == {}


@pytest.mark.parametrize(
    "field", ["recording_frame_id", "parent_frame_index", "timestamp", "timestamp_sys"]
)
@pytest.mark.parametrize(
    "values",
    [
        [1.9, 2.9, 3.9],
        [0.0, 1.0, 2.0],
        [False, True, True],
        ["0", "1", "2"],
        [float("nan"), 1.0, 2.0],
        [float("inf"), 1.0, 2.0],
        pa.array([2**63, 2**63 + 1, 2**63 + 2], type=pa.uint64()),
    ],
)
def test_parquet_clock_rejects_non_integer_or_out_of_range_values(
    tmp_path,
    monkeypatch,
    field,
    values,
):
    with pytest.raises(clock.AcquisitionFrameClockError, match=field):
        _load_table(tmp_path, monkeypatch, **{field: values})


@pytest.mark.parametrize("field", ["recording_frame_id", "parent_frame_index"])
def test_parquet_clock_rejects_null_identifiers(tmp_path, monkeypatch, field):
    with pytest.raises(clock.AcquisitionFrameClockError, match=field):
        _load_table(tmp_path, monkeypatch, **{field: [None, 1, 2]})


def test_parquet_clock_preserves_large_exact_integers_and_null_timestamps(
    tmp_path, monkeypatch
):
    values = [2**53 + 1, None, 2**53 + 3]
    source = _load_table(
        tmp_path,
        monkeypatch,
        timestamp=pa.array(values, type=pa.uint64()),
        timestamp_sys=[None, None, None],
    )
    np.testing.assert_array_equal(
        source.camera_timestamp_ns, [values[0], -(2**63), values[2]]
    )
    np.testing.assert_array_equal(source.camera_timestamp_valid, [True, False, True])
    assert not source.system_timestamp_valid.any()
    assert source.camera_timestamp_ns.dtype == np.dtype("int64")


@pytest.mark.parametrize("start", [0, 1, 400, -3])
def test_parquet_source_identifier_origin_is_preserved_not_rebased(
    tmp_path, monkeypatch, start
):
    # Origin policy is a separate compatibility decision; retain the source ID.
    source = _load_table(
        tmp_path, monkeypatch, recording_frame_id=[start, start + 1, start + 2]
    )
    np.testing.assert_array_equal(
        source.recording_frame_id, [start, start + 1, start + 2]
    )
    np.testing.assert_array_equal(source.parent_frame_index, [0, 1, 2])


@pytest.mark.parametrize(
    "field", ["recording_frame_id", "camera_timestamp_ns", "system_timestamp_ns"]
)
def test_source_rejects_wrapped_int64_progression(tmp_path, field):
    source = replace(
        _source(tmp_path),
        **{
            field: np.array([2**63 - 1, -(2**63), -(2**63) + 1], dtype=np.int64),
        },
    )
    with pytest.raises(clock.AcquisitionFrameClockError, match=field):
        clock._validate_source(source, expected_frame_count=3)


def test_source_accepts_increasing_timestamps_spanning_int64_range(tmp_path):
    source = replace(
        _source(tmp_path), camera_timestamp_ns=np.array([-(2**63), 0, 2**63 - 1])
    )
    assert clock._validate_source(source, expected_frame_count=3) is source


@pytest.mark.parametrize("field", clock._ARRAY_NAMES)
def test_source_rejects_coercible_but_wrong_array_dtypes(tmp_path, field):
    source = _source(tmp_path)
    source = replace(source, **{field: getattr(source, field).astype(float)})
    with pytest.raises(clock.AcquisitionFrameClockError, match=field):
        clock._validate_source(source, expected_frame_count=3)


@pytest.mark.parametrize("value", [str(2**63), str(-(2**63) - 1), "1.9"])
def test_csv_integer_errors_are_labelled_and_fail_closed(value):
    with pytest.raises(clock.AcquisitionFrameClockError, match="timestamp.*row 2"):
        clock._parse_optional_int(value, label="timestamp", row_number=2)
