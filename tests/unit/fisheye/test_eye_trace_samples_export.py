from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fisheye.analytics_exports.arrow_contracts import (
    ARROW_TABLE_CONTRACTS,
    exact_arrow_schema,
)
from fisheye.analytics_exports.contracts import EYE_TRACE_SAMPLES_TABLE
from fisheye.analytics_exports.eye_trace_samples import (
    EYE_TRACE_ANGLE_CHANNELS,
    EYE_TRACE_QA_CHANNELS,
    export_eye_trace_samples,
    eye_trace_projection_contract,
)
from fisheye.analytics_exports.publication import sha256_file
from fisheye.analytics_exports.validation import (
    ExportValidationError,
    validate_export_payload,
    validate_export_run,
)
from fisheye.analysis.eye_angle_io import load_eye_angle_series_rows
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils.export_cross_recording_analytics import _parse_tables


class _Group(dict):
    def __init__(self, *args: object, attrs: dict[str, object] | None = None) -> None:
        super().__init__(*args)
        self.attrs = dict(attrs or {})


def _source_root(*, frame_count: int = 7) -> tuple[_Group, _Group, SimpleNamespace]:
    parent = _Group(
        attrs={
            "latest": "eye_1",
            "latest_complete": "eye_1",
            "palette_completion_epoch": 2,
        }
    )
    run = _Group(
        attrs={
            "schema_id": "analysis.eye_angle_runs",
            "schema_version": 7,
            "layout": "compact_dense_v2",
            "method": "ellipse_geometry",
            "method_version": "eye_angle.v7",
            "palette_run_completion_status": "complete",
            "palette_run_completed_at_utc": "2026-08-04T12:00:00+00:00",
            "stage_selector_eligible": True,
            "eye_angle_array_schema": {"schema_id": "array-schema", "version": 1},
            "eye_angle_source_contracts": {"schema_id": "sources", "version": 1},
            "eye_angle_algorithm_contract": {"schema_id": "algorithm", "version": 1},
            "eye_angle_output_schema": {"schema_id": "output", "version": 9},
            "eye_angle_variant_schema": {"schema_id": "variants", "version": 1},
        }
    )
    parent["eye_1"] = run
    root = _Group({"analysis/eye_angle_runs": parent})
    catalog = SimpleNamespace(
        run_name="eye_1",
        run_path="analysis/eye_angle_runs/eye_1",
        row_axis="frame",
        row_count=frame_count,
        angle_channels=EYE_TRACE_ANGLE_CHANNELS,
        qa_channels=EYE_TRACE_QA_CHANNELS,
    )
    return root, run, catalog


def _install_fake_source(
    monkeypatch: pytest.MonkeyPatch,
    *,
    frame_count: int = 7,
    mutate_after_last_batch: bool = False,
) -> tuple[_Group, _Group]:
    import fisheye.analytics_exports.eye_trace_samples as mod

    root, run, catalog = _source_root(frame_count=frame_count)
    monkeypatch.setattr(mod, "open_zarr_root", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(
        mod,
        "resolve_eye_angle_run",
        lambda *_args, **_kwargs: (
            run,
            "eye_1",
            "analysis/eye_angle_runs/eye_1",
        ),
    )
    monkeypatch.setattr(
        mod, "catalog_eye_angle_series", lambda *_args, **_kwargs: catalog
    )

    def rows(*_args: object, **kwargs: object) -> SimpleNamespace:
        start = int(kwargs["start_row"])
        stop = int(kwargs["stop_row"])
        frames = np.arange(start, stop, dtype=np.int64)
        base = frames.astype(np.float32)
        if mutate_after_last_batch and stop == frame_count:
            root["analysis/eye_angle_runs"].attrs["latest"] = "changed"
        return SimpleNamespace(
            frame_indices=frames,
            time_seconds=base / np.float32(100.0),
            angles={
                name: base + np.float32(index / 10.0)
                for index, name in enumerate(EYE_TRACE_ANGLE_CHANNELS)
            },
            qa={
                "valid_frame": frames % 2 == 0,
                "major_axis_marginal": frames % 3 == 0,
                "reason_codes": (frames % 5).astype(np.uint16),
            },
        )

    monkeypatch.setattr(mod, "load_eye_angle_series_rows", rows)
    return root, run


def test_eye_trace_arrow_contract_preserves_exact_source_dtypes() -> None:
    contract = ARROW_TABLE_CONTRACTS[EYE_TRACE_SAMPLES_TABLE]
    fields = {field.name: field for field in contract.fields}
    assert fields["time_seconds"].arrow_type == "float32"
    assert fields["left_eye_angle_deg"].arrow_type == "float32"
    assert fields["reason_codes"].arrow_type == "uint16"
    assert fields["valid_frame"].arrow_type == "bool"
    assert all(
        not field.nullable for field in contract.fields if "angle_deg" in field.name
    )

    schema = exact_arrow_schema(EYE_TRACE_SAMPLES_TABLE, metadata={})
    assert schema.field("time_seconds").type == pa.float32()
    assert schema.field("reason_codes").type == pa.uint16()


def test_eye_trace_projection_is_closed_and_digest_bound() -> None:
    projection = eye_trace_projection_contract()
    assert projection["row_axis"] == "camera_frame"
    assert tuple(projection["angle_channels"]) == EYE_TRACE_ANGLE_CHANNELS
    assert tuple(projection["qa_channels"]) == EYE_TRACE_QA_CHANNELS
    body = dict(projection)
    digest = body.pop("payload_sha256")
    assert digest == canonical_json_sha256(body)
    assert (
        "frame_angles/left_eye_angle_deg" in projection["source_logical_paths"].values()
    )


def test_compact_exporter_rejects_framewise_trace_table() -> None:
    with pytest.raises(ValueError, match="bounded streaming exporter"):
        _parse_tables((EYE_TRACE_SAMPLES_TABLE,))


def test_exact_eye_angle_row_reader_uses_half_open_frame_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fisheye.analysis.eye_angle_io as eye_io

    angle_names = EYE_TRACE_ANGLE_CHANNELS + ("unused",)
    qa_names = EYE_TRACE_QA_CHANNELS
    frame_angles = np.arange(5 * len(angle_names), dtype=np.float32).reshape(
        5, len(angle_names)
    )
    frame_qa = np.asarray(
        [[1, 0, 0], [1, 1, 2], [0, 0, 4], [1, 0, 0], [1, 1, 8]],
        dtype=np.uint16,
    )
    angle_index = _Group()
    qa_index = _Group()
    support = _Group(
        {"frame_time_seconds": np.arange(5, dtype=np.float32) / np.float32(100)}
    )
    run = _Group(
        {
            "frame_angles": frame_angles,
            "frame_qa": frame_qa,
            "angle_channel_index": angle_index,
            "qa_channel_index": qa_index,
            "support": support,
        }
    )
    catalog = SimpleNamespace(
        run_name="eye_1",
        run_path="analysis/eye_angle_runs/eye_1",
        row_axis="frame",
        row_count=5,
        angle_channels=angle_names,
        qa_channels=qa_names,
    )
    monkeypatch.setattr(eye_io, "catalog_eye_angle_series", lambda *_a, **_k: catalog)
    monkeypatch.setattr(
        eye_io,
        "resolve_eye_angle_run",
        lambda *_a, **_k: (run, "eye_1", catalog.run_path),
    )
    monkeypatch.setattr(
        eye_io,
        "_channel_names",
        lambda group, **_kwargs: list(
            angle_names if group is angle_index else qa_names
        ),
    )

    window = load_eye_angle_series_rows(
        _Group(),
        run_name="eye_1",
        start_row=1,
        stop_row=4,
        angle_channels=EYE_TRACE_ANGLE_CHANNELS,
        qa_channels=EYE_TRACE_QA_CHANNELS,
        max_rows=3,
    )
    assert window.frame_indices.tolist() == [1, 2, 3]
    assert window.time_seconds.tolist() == pytest.approx([0.01, 0.02, 0.03])
    np.testing.assert_array_equal(
        window.angles["left_eye_angle_deg"], frame_angles[1:4, 0]
    )
    np.testing.assert_array_equal(window.qa["reason_codes"], frame_qa[1:4, 2])


def test_eye_trace_export_streams_batches_and_is_batch_boundary_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, run = _install_fake_source(monkeypatch, frame_count=7)
    source_attrs_before = deepcopy(run.attrs)
    source = tmp_path / "recording_analysis.zarr"

    first = export_eye_trace_samples(
        source,
        eye_angle_run="eye_1",
        output_root=tmp_path / "exports_a",
        export_run_id="eye_trace_a",
        scratch_root=tmp_path / "scratch_a",
        row_group_rows=3,
    )
    second = export_eye_trace_samples(
        source,
        eye_angle_run="eye_1",
        output_root=tmp_path / "exports_b",
        export_run_id="eye_trace_b",
        scratch_root=tmp_path / "scratch_b",
        row_group_rows=4,
    )

    assert run.attrs == source_attrs_before
    assert first["row_counts_by_table"] == {EYE_TRACE_SAMPLES_TABLE: 7}
    assert first["eye_trace_validation"]["valid"] is True
    assert (
        first["eye_trace_export"]["projected_payload"]["payload_sha256"]
        == second["eye_trace_export"]["projected_payload"]["payload_sha256"]
    )
    validated = validate_export_run(tmp_path / "exports_a", "eye_trace_a")
    assert validated["status"] == "valid"
    assert validated["row_count"] == 7
    part = (
        tmp_path
        / "exports_a"
        / first["part_files_by_table"][EYE_TRACE_SAMPLES_TABLE][0]
    )
    parquet_file = pq.ParquetFile(part)
    assert parquet_file.metadata.num_row_groups == 3
    table = parquet_file.read()
    assert table.column("source_acquisition_frame_index").to_pylist() == list(range(7))
    assert table.schema.field("left_eye_angle_deg").type == pa.float32()
    assert table.schema.field("reason_codes").type == pa.uint16()


def test_eye_trace_export_fails_before_visibility_when_source_binding_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_source(
        monkeypatch,
        frame_count=7,
        mutate_after_last_batch=True,
    )
    output = tmp_path / "exports"
    scratch = tmp_path / "scratch"
    with pytest.raises(RuntimeError, match="changed during extraction"):
        export_eye_trace_samples(
            tmp_path / "recording_analysis.zarr",
            eye_angle_run="eye_1",
            output_root=output,
            export_run_id="eye_trace_changed",
            scratch_root=scratch,
            row_group_rows=3,
        )
    assert not (output / "v1" / "manifests").exists()
    assert not any(scratch.glob("palette_eye_trace_*"))


def test_eye_trace_validator_rejects_rehashed_projection_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_source(monkeypatch, frame_count=4)
    output = tmp_path / "exports"
    result = export_eye_trace_samples(
        tmp_path / "recording_analysis.zarr",
        eye_angle_run="eye_1",
        output_root=output,
        export_run_id="eye_trace_tamper",
        scratch_root=tmp_path / "scratch",
        row_group_rows=2,
    )
    payload = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    tampered = deepcopy(payload)
    projection = tampered["eye_trace_export"]["projection_contract"]
    projection["angle_channels"] = list(reversed(projection["angle_channels"]))
    projection_body = dict(projection)
    projection_body.pop("payload_sha256")
    projection["payload_sha256"] = canonical_json_sha256(projection_body)
    envelope = tampered["eye_trace_export"]
    envelope_body = dict(envelope)
    envelope_body.pop("payload_sha256")
    envelope["payload_sha256"] = canonical_json_sha256(envelope_body)

    with pytest.raises(ExportValidationError, match="installed contract"):
        validate_export_payload(output, "eye_trace_tamper", tampered)


@pytest.mark.parametrize(
    ("nested_field", "error_match"),
    (
        ("source_binding", "source binding"),
        ("projected_payload", "projected-payload receipt"),
        ("parquet_policy", "installed contract"),
    ),
)
def test_eye_trace_validator_rejects_rehashed_nested_extra_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested_field: str,
    error_match: str,
) -> None:
    _install_fake_source(monkeypatch, frame_count=4)
    output = tmp_path / "exports"
    result = export_eye_trace_samples(
        tmp_path / "recording_analysis.zarr",
        eye_angle_run="eye_1",
        output_root=output,
        export_run_id="eye_trace_nested_tamper",
        scratch_root=tmp_path / "scratch",
        row_group_rows=2,
    )
    payload = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    tampered = deepcopy(payload)
    nested = tampered["eye_trace_export"][nested_field]
    nested["unexpected"] = "adversarial"
    nested_body = dict(nested)
    nested_body.pop("payload_sha256")
    nested["payload_sha256"] = canonical_json_sha256(nested_body)
    envelope = tampered["eye_trace_export"]
    envelope_body = dict(envelope)
    envelope_body.pop("payload_sha256")
    envelope["payload_sha256"] = canonical_json_sha256(envelope_body)

    with pytest.raises(ExportValidationError, match=error_match):
        validate_export_payload(output, "eye_trace_nested_tamper", tampered)


def test_eye_trace_validator_rejects_rehashed_constant_column_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_source(monkeypatch, frame_count=4)
    output = tmp_path / "exports"
    result = export_eye_trace_samples(
        tmp_path / "recording_analysis.zarr",
        eye_angle_run="eye_1",
        output_root=output,
        export_run_id="eye_trace_constant_tamper",
        scratch_root=tmp_path / "scratch",
        row_group_rows=2,
    )
    payload = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    relative_part = payload["part_files_by_table"][EYE_TRACE_SAMPLES_TABLE][0]
    part = output / relative_part
    # Read the file directly so PyArrow does not infer the surrounding
    # ``export_run_id=``/``generation=`` directories as partition columns.
    table = pq.ParquetFile(part).read()
    field_index = table.schema.get_field_index("recording_id")
    table = table.set_column(
        field_index,
        table.schema.field(field_index),
        pa.array(["different-recording"] * table.num_rows, type=pa.string()),
    )
    pq.write_table(table, part, compression="zstd", compression_level=3)
    inventory = payload["publication"]["parts_by_table"][EYE_TRACE_SAMPLES_TABLE][0]
    inventory["sha256"] = sha256_file(part)
    inventory["size_bytes"] = int(part.stat().st_size)

    with pytest.raises(ExportValidationError, match="field recording_id changed"):
        validate_export_payload(output, "eye_trace_constant_tamper", payload)
