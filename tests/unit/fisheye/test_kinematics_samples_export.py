from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fisheye.analytics_exports import kinematics_samples as mod
from fisheye.analytics_exports.contracts import KINEMATICS_SAMPLES_TABLE
from fisheye.analytics_exports.validation import (
    ExportValidationError,
    validate_export_run,
)
from fisheye.analytics_exports.publication import sha256_file
from fisheye.analytics_exports.runtime_telemetry import (
    validate_export_runtime_telemetry,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils.export_kinematics_samples import build_arg_parser
from fisheye.diagnostics.validate_kinematics_query_window_equivalence import (
    validate_kinematics_query_window_equivalence,
)
from tests.unit.fisheye.test_track_motion_publication import (
    _clone_canonical_physical_motion_run,
    _clone_physical_motion_run,
    _fresh_full_motion_run,
)


def _mark_eligible_source(root: Any, run: Any) -> None:
    run.attrs["stage_selector_eligible"] = True
    run.attrs["palette_run_completed_at_utc"] = "2026-08-04T12:00:00+00:00"
    parent = root["analysis"]["track_kinematics_runs"]
    scope = parent["offline"]
    parent.attrs["palette_completion_epoch"] = 3
    scope.attrs["palette_completion_epoch"] = 2
    parent.attrs["latest"] = "offline/motion_physical"
    parent.attrs["latest_complete"] = "offline/motion_physical"
    parent.attrs["latest_offline"] = "motion_physical"
    scope.attrs["latest"] = "motion_physical"


def _eligible_source(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, Any]:
    root, run, track, _sealed, _physical = _clone_canonical_physical_motion_run(
        monkeypatch
    )
    _mark_eligible_source(root, run)
    monkeypatch.setattr(mod, "open_zarr_root", lambda *_args, **_kwargs: root)
    return root, run, track


def _export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    export_run_id: str,
    output_name: str,
    source_window_rows: int,
    row_group_rows: int,
    requested_sample_rate_hz: float = 0.5,
    source_frame_start: int | None = None,
    source_frame_stop_exclusive: int | None = None,
) -> dict[str, Any]:
    _eligible_source(monkeypatch)
    return mod.export_kinematics_samples(
        tmp_path / "recording_analysis.zarr",
        track_kinematics_run="motion_physical",
        track_scope="offline",
        requested_sample_rate_hz=requested_sample_rate_hz,
        output_root=tmp_path / output_name,
        export_run_id=export_run_id,
        scratch_root=tmp_path / f"scratch_{output_name}",
        source_window_rows=source_window_rows,
        row_group_rows=row_group_rows,
        source_frame_start=source_frame_start,
        source_frame_stop_exclusive=source_frame_stop_exclusive,
    )


def test_projection_contract_uses_global_acquisition_frame_stride() -> None:
    contract = mod.kinematics_projection_contract(
        source_sample_rate_hz=700.0,
        requested_sample_rate_hz=10.0,
    )
    assert contract["sampling_stride_frames"] == 70
    assert contract["nominal_sample_rate_hz"] == 10.0
    assert contract["sampling_policy"] == ("global_acquisition_frame_modulo_stride_v1")
    assert contract["selection_expression"] == (
        "source_acquisition_frame_index % stride == 0"
    )
    assert contract["source_speed_level"] == "filtered"
    assert contract["invalid_float_semantics"] == ("source_ieee_nan_not_arrow_null")
    assert contract["payload_sha256"] == canonical_json_sha256(
        {key: value for key, value in contract.items() if key != "payload_sha256"}
    )

    clamped = mod.kinematics_projection_contract(
        source_sample_rate_hz=5.0,
        requested_sample_rate_hz=20.0,
    )
    assert clamped["sampling_stride_frames"] == 1
    assert clamped["nominal_sample_rate_hz"] == 5.0


def test_cli_defaults_to_full_resolution_and_sampling_is_explicit() -> None:
    required = [
        "/tmp/recording_analysis.zarr",
        "--track-kinematics-run",
        "motion_physical",
        "--output-root",
        "/tmp/exports",
        "--export-run-id",
        "motion_export",
        "--scratch-root",
        "/tmp/scratch",
    ]
    default_args = build_arg_parser().parse_args(required)
    sampled_args = build_arg_parser().parse_args(
        [*required, "--sample-rate-hz", "10"]
    )

    assert default_args.sample_rate_hz is None
    assert sampled_args.sample_rate_hz == 10.0


def test_projection_contract_versions_exact_half_open_frame_range() -> None:
    contract = mod.kinematics_projection_contract(
        source_sample_rate_hz=30.0,
        requested_sample_rate_hz=10.0,
        source_frame_start=0,
        source_frame_stop_exclusive=200_000,
    )

    assert contract["schema_version"] == (
        mod.KINEMATICS_PROJECTION_SCHEMA_VERSION_V2
    )
    assert contract["frame_selection_policy"] == (
        mod.KINEMATICS_FRAME_SELECTION_POLICY
    )
    assert contract["source_frame_start"] == 0
    assert contract["source_frame_stop_exclusive"] == 200_000
    assert contract["payload_sha256"] == canonical_json_sha256(
        {key: value for key, value in contract.items() if key != "payload_sha256"}
    )


@pytest.mark.parametrize(
    ("start", "stop"),
    ((0, None), (None, 1), (-1, 1), (1, 1), (2, 1), (False, 1), (0, True)),
)
def test_projection_contract_rejects_invalid_frame_ranges(
    start: int | None,
    stop: int | None,
) -> None:
    with pytest.raises(ValueError, match="source frame"):
        mod.kinematics_projection_contract(
            source_sample_rate_hz=30.0,
            requested_sample_rate_hz=10.0,
            source_frame_start=start,
            source_frame_stop_exclusive=stop,
        )


@pytest.mark.parametrize("invalid", (0.0, -1.0, float("inf"), float("nan")))
def test_projection_contract_rejects_invalid_requested_rates(invalid: float) -> None:
    with pytest.raises(ValueError, match="requested sample rate"):
        mod.kinematics_projection_contract(
            source_sample_rate_hz=700.0,
            requested_sample_rate_hz=invalid,
        )


def test_exact_source_binding_accepts_float32_and_rejects_float64_positions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, _run, track = _eligible_source(monkeypatch)
    assert track.children["positions_mm"].data.dtype == np.dtype("float32")
    bound = mod._source_binding(
        root,
        zarr_path=(tmp_path / "recording_analysis.zarr").resolve(),
        recording_id="recording",
        run_name="motion_physical",
        scope="offline",
    )
    assert (
        bound.binding["tracks"][0]["selected_surfaces"]["positions_mm"]["dtype"]
        == "<f4"
    )

    track.children["positions_mm"].dtype = np.dtype("float64")
    with pytest.raises(ValueError, match="live declaration differs from its manifest"):
        mod._source_binding(
            root,
            zarr_path=(tmp_path / "recording_analysis.zarr").resolve(),
            recording_id="recording",
            run_name="motion_physical",
            scope="offline",
        )


def test_exact_source_binding_rejects_legacy_float64_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, run, track, _sealed, _physical = _clone_physical_motion_run(monkeypatch)
    assert track.children["positions_mm"].dtype == np.dtype("float64")
    _mark_eligible_source(root, run)
    with pytest.raises(ValueError, match="differs from its exact dtype/shape contract"):
        mod._source_binding(
            root,
            zarr_path=(tmp_path / "recording_analysis.zarr").resolve(),
            recording_id="recording",
            run_name="motion_physical",
            scope="offline",
        )


def test_kinematics_export_is_bounded_and_batch_boundary_independent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _export(
        monkeypatch,
        tmp_path,
        export_run_id="kinematics_a",
        output_name="exports_a",
        source_window_rows=1,
        row_group_rows=1,
    )
    second = _export(
        monkeypatch,
        tmp_path,
        export_run_id="kinematics_b",
        output_name="exports_b",
        source_window_rows=2,
        row_group_rows=2,
    )
    first_envelope = first["kinematics_samples_export"]
    second_envelope = second["kinematics_samples_export"]
    assert first_envelope["projected_payload"] == second_envelope["projected_payload"]
    assert first["kinematics_samples_validation"]["valid"] is True
    validate_export_runtime_telemetry(first["runtime_telemetry"])
    assert "runtime_telemetry" not in json.loads(
        Path(first["manifest_path"]).read_text(encoding="utf-8")
    )
    assert first["row_counts_by_table"] == {KINEMATICS_SAMPLES_TABLE: 1}
    report = validate_export_run(tmp_path / "exports_a", "kinematics_a")
    assert report["status"] == "valid"

    import pyarrow.parquet as pq

    part = next((tmp_path / "exports_a").rglob("*.parquet"))
    table = pq.read_table(part).to_pydict()
    assert table["track_id"] == [7]
    assert table["track_sample_index"] == [0]
    assert table["source_acquisition_frame_index"] == [0]
    assert table["source_track_kinematics_scope"] == ["offline"]
    assert table["source_track_kinematics_run"] == ["motion_physical"]
    assert table["position_coordinate_space"] == ["physical_mm"]
    # PyArrow's Python conversion widens scalar lists. The exact physical
    # Arrow declarations below prove the persisted source projection remains
    # float32 rather than introducing an export-only widening.
    assert np.asarray(table["position_x_mm"]).dtype == np.dtype("float64")
    assert np.asarray(table["speed_mm_s"]).dtype == np.dtype("float64")
    assert pq.ParquetFile(part).schema_arrow.field("position_x_mm").type == (
        __import__("pyarrow").float32()
    )
    assert pq.ParquetFile(part).schema_arrow.field("speed_mm_s").type == (
        __import__("pyarrow").float32()
    )


def test_kinematics_export_defaults_to_every_source_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _eligible_source(monkeypatch)
    result = mod.export_kinematics_samples(
        tmp_path / "recording_analysis.zarr",
        track_kinematics_run="motion_physical",
        track_scope="offline",
        output_root=tmp_path / "exports_full_rate",
        export_run_id="full_rate_default",
        scratch_root=tmp_path / "scratch_full_rate",
        source_window_rows=1,
        row_group_rows=1,
    )

    projection = result["kinematics_samples_export"]["projection_contract"]
    assert projection["requested_sample_rate_hz"] == (
        projection["source_sample_rate_hz"]
    )
    assert projection["sampling_stride_frames"] == 1
    assert result["row_counts_by_table"] == {KINEMATICS_SAMPLES_TABLE: 2}


def test_kinematics_export_persists_and_enforces_frame_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _export(
        monkeypatch,
        tmp_path,
        export_run_id="kinematics_windowed",
        output_name="exports_windowed",
        source_window_rows=1,
        row_group_rows=1,
        requested_sample_rate_hz=1.0,
        source_frame_start=1,
        source_frame_stop_exclusive=2,
    )

    projection = result["kinematics_samples_export"]["projection_contract"]
    assert projection["schema_version"] == 2
    assert projection["source_frame_start"] == 1
    assert projection["source_frame_stop_exclusive"] == 2
    assert result["row_counts_by_table"] == {KINEMATICS_SAMPLES_TABLE: 1}
    assert result["kinematics_samples_validation"]["valid"] is True

    import pyarrow.parquet as pq

    part = next((tmp_path / "exports_windowed").rglob("*.parquet"))
    table = pq.read_table(part).to_pydict()
    assert table["source_acquisition_frame_index"] == [1]
    assert table["track_sample_index"] == [1]


def test_bounded_export_equals_exact_unbounded_frame_slice(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _export(
        monkeypatch,
        tmp_path,
        export_run_id="kinematics_full",
        output_name="exports_full",
        source_window_rows=1,
        row_group_rows=1,
        requested_sample_rate_hz=1.0,
    )
    _export(
        monkeypatch,
        tmp_path,
        export_run_id="kinematics_window",
        output_name="exports_window",
        source_window_rows=2,
        row_group_rows=2,
        requested_sample_rate_hz=1.0,
        source_frame_start=0,
        source_frame_stop_exclusive=1,
    )

    evidence = validate_kinematics_query_window_equivalence(
        full_export_root=tmp_path / "exports_full",
        full_export_run_id="kinematics_full",
        bounded_export_root=tmp_path / "exports_window",
        bounded_export_run_id="kinematics_window",
        output=tmp_path
        / "palette_benchmarks"
        / "kinematics_window_equivalence.json",
    )

    assert evidence["payload"]["status"] == "passed"
    assert evidence["payload"]["frame_interval"] == {
        "start": 0,
        "stop_exclusive": 1,
        "frame_count": 1,
    }
    assert evidence["payload"]["logical_equality"]["equal"] is True
    assert evidence["payload"]["promotion_authorized"] is False


def test_streaming_writer_preserves_multiple_tracks_as_distinct_primary_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, run, track = _eligible_source(monkeypatch)
    source_path = (tmp_path / "recording_analysis.zarr").resolve()
    bound = mod._source_binding(
        root,
        zarr_path=source_path,
        recording_id="recording",
        run_name="motion_physical",
        scope="offline",
    )
    second_track = copy.deepcopy(track)
    second_track.children["track_sample_key"].data[:, 0] = 8
    run["tracks"].children["id_8"] = second_track
    binding = copy.deepcopy(bound.binding)
    second_binding = copy.deepcopy(binding["tracks"][0])
    second_binding["track_id"] = 8
    second_binding["track_ref"] = (
        "/analysis/track_kinematics_runs/offline/motion_physical/tracks/id_8"
    )
    second_binding["selected_surfaces"]["track_sample_key"]["content_sha256"] = (
        array_values_sha256(second_track.children["track_sample_key"].data)
    )
    binding["tracks"].append(second_binding)
    binding["track_count"] = 2
    binding["payload_sha256"] = canonical_json_sha256(
        {key: value for key, value in binding.items() if key != "payload_sha256"}
    )
    projection = mod.kinematics_projection_contract(
        source_sample_rate_hz=1.0,
        requested_sample_rate_hz=0.5,
    )
    part = tmp_path / "two_tracks.parquet"
    projected = mod._write_streaming_part(
        mod._BoundSource(binding=binding, run_group=run),
        part_path=part,
        projection=projection,
        source_window_rows=1,
        row_group_rows=1,
    )
    assert projected["row_count"] == 2

    import pyarrow.parquet as pq

    table = pq.read_table(part).to_pydict()
    assert list(
        zip(
            table["track_id"],
            table["source_acquisition_frame_index"],
            strict=True,
        )
    ) == [(7, 0), (8, 0)]


def test_kinematics_export_fails_before_visibility_when_source_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _root, run, _track = _eligible_source(monkeypatch)
    original = mod._write_streaming_part

    def changing_writer(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        run.attrs["stage_selector_eligible"] = False
        return result

    monkeypatch.setattr(mod, "_write_streaming_part", changing_writer)
    output = tmp_path / "exports"
    scratch = tmp_path / "scratch"
    with pytest.raises(ValueError, match="must be selector-eligible"):
        mod.export_kinematics_samples(
            tmp_path / "recording_analysis.zarr",
            track_kinematics_run="motion_physical",
            track_scope="offline",
            requested_sample_rate_hz=0.5,
            output_root=output,
            export_run_id="changed",
            scratch_root=scratch,
            source_window_rows=1,
            row_group_rows=1,
        )
    assert not (output / "v2" / "manifests" / "changed.json").exists()
    assert not any(scratch.glob("palette_kinematics_*"))


def test_kinematics_export_rejects_manifest_mismatched_unsampled_source_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _root, _run, track = _eligible_source(monkeypatch)
    # At 1 Hz -> requested 0.5 Hz, frame 1 is not exported. The bounded source
    # verifier must still bind every decoded source row, not only selected rows.
    track.children["positions_mm"].data[1, 0] += 0.25
    with pytest.raises(ValueError, match="payload differs from its publication"):
        mod.export_kinematics_samples(
            tmp_path / "recording_analysis.zarr",
            track_kinematics_run="motion_physical",
            track_scope="offline",
            requested_sample_rate_hz=0.5,
            output_root=tmp_path / "exports",
            export_run_id="tampered_source_bytes",
            scratch_root=tmp_path / "scratch",
            source_window_rows=1,
            row_group_rows=1,
        )
    assert not (
        tmp_path / "exports" / "v2" / "manifests" / "tampered_source_bytes.json"
    ).exists()


def test_kinematics_validator_rejects_rehashed_projection_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _export(
        monkeypatch,
        tmp_path,
        export_run_id="tamper_projection",
        output_name="exports",
        source_window_rows=1,
        row_group_rows=1,
    )
    manifest_path = Path(result["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    envelope = payload["kinematics_samples_export"]
    projection = envelope["projection_contract"]
    projection["selection_expression"] = "source_row_index % stride == 0"
    projection["payload_sha256"] = canonical_json_sha256(
        {key: value for key, value in projection.items() if key != "payload_sha256"}
    )
    envelope["payload_sha256"] = canonical_json_sha256(
        {key: value for key, value in envelope.items() if key != "payload_sha256"}
    )
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ExportValidationError, match="projection differs"):
        validate_export_run(tmp_path / "exports", "tamper_projection")


def test_kinematics_validator_rejects_rehashed_nested_source_extra(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _export(
        monkeypatch,
        tmp_path,
        export_run_id="tamper_source",
        output_name="exports",
        source_window_rows=1,
        row_group_rows=1,
    )
    manifest_path = Path(result["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    envelope = payload["kinematics_samples_export"]
    source = envelope["source_binding"]
    source["tracks"][0]["selected_surfaces"]["positions_mm"]["unexpected"] = True
    source["payload_sha256"] = canonical_json_sha256(
        {key: value for key, value in source.items() if key != "payload_sha256"}
    )
    envelope["payload_sha256"] = canonical_json_sha256(
        {key: value for key, value in envelope.items() if key != "payload_sha256"}
    )
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ExportValidationError, match="surface binding"):
        validate_export_run(tmp_path / "exports", "tamper_source")


def test_kinematics_validator_rejects_rehashed_constant_column_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _export(
        monkeypatch,
        tmp_path,
        export_run_id="tamper_constant",
        output_name="exports",
        source_window_rows=1,
        row_group_rows=1,
    )
    manifest_path = Path(result["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    part_path = next((tmp_path / "exports").rglob("*.parquet"))

    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(part_path)
    original_schema = parquet_file.schema_arrow
    table = parquet_file.read()
    column_index = table.schema.get_field_index("source_speed_level")
    arrays = [table.column(index) for index in range(table.num_columns)]
    arrays[column_index] = pa.chunked_array(
        [pa.array(["raw"] * table.num_rows, type=pa.string())]
    )
    table = pa.Table.from_arrays(arrays, schema=original_schema)
    writer = pq.ParquetWriter(
        part_path,
        original_schema,
        compression="zstd",
        compression_level=3,
        use_dictionary=payload["kinematics_samples_export"]["parquet_policy"][
            "dictionary_columns"
        ],
    )
    try:
        writer.write_table(table, row_group_size=1)
    finally:
        writer.close()
    entry = payload["publication"]["parts_by_table"][KINEMATICS_SAMPLES_TABLE][0]
    entry["sha256"] = sha256_file(part_path)
    entry["size_bytes"] = part_path.stat().st_size
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ExportValidationError, match="source_speed_level changed"):
        validate_export_run(tmp_path / "exports", "tamper_constant")


def test_failed_overwrite_preserves_previous_manifest_selected_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _root, run, _track = _eligible_source(monkeypatch)
    output = tmp_path / "exports"
    source_path = tmp_path / "recording_analysis.zarr"
    first = mod.export_kinematics_samples(
        source_path,
        track_kinematics_run="motion_physical",
        track_scope="offline",
        requested_sample_rate_hz=0.5,
        output_root=output,
        export_run_id="stable",
        scratch_root=tmp_path / "scratch_first",
        source_window_rows=1,
        row_group_rows=1,
    )
    manifest_path = Path(first["manifest_path"])
    baseline = manifest_path.read_bytes()
    original = mod._write_streaming_part

    def failing_replacement(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        run.attrs["stage_selector_eligible"] = False
        return result

    monkeypatch.setattr(mod, "_write_streaming_part", failing_replacement)
    with pytest.raises(ValueError, match="must be selector-eligible"):
        mod.export_kinematics_samples(
            source_path,
            track_kinematics_run="motion_physical",
            track_scope="offline",
            requested_sample_rate_hz=0.5,
            output_root=output,
            export_run_id="stable",
            scratch_root=tmp_path / "scratch_second",
            source_window_rows=1,
            row_group_rows=1,
            overwrite=True,
        )
    assert manifest_path.read_bytes() == baseline
    assert validate_export_run(output, "stable")["status"] == "valid"


def test_kinematics_export_rejects_track_run_without_physical_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, run, _track, _sealed, _physical = _fresh_full_motion_run(
        monkeypatch,
        physical=False,
    )
    run.attrs["stage_selector_eligible"] = True
    run.attrs["palette_run_completed_at_utc"] = "2026-08-04T12:00:00+00:00"
    monkeypatch.setattr(mod, "open_zarr_root", lambda *_args, **_kwargs: root)
    with pytest.raises(ValueError, match="requires physical-mm authority"):
        mod.export_kinematics_samples(
            tmp_path / "recording_analysis.zarr",
            track_kinematics_run="motion_pixel",
            track_scope="offline",
            requested_sample_rate_hz=0.5,
            output_root=tmp_path / "exports",
            export_run_id="no_physical",
            scratch_root=tmp_path / "scratch",
        )


def test_kinematics_export_rejects_overlapping_scratch_and_output_roots(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        mod.export_kinematics_samples(
            tmp_path / "recording_analysis.zarr",
            track_kinematics_run="motion_physical",
            track_scope="offline",
            requested_sample_rate_hz=0.5,
            output_root=tmp_path / "exports",
            export_run_id="overlap",
            scratch_root=tmp_path / "exports" / "scratch",
        )
