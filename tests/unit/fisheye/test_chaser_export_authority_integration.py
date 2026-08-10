from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
import zarr

from fisheye.analytics_exports.chaser_authority import (
    EGOCENTRIC_BEARING_FAMILY,
    EPOCH_BEHAVIOR_FAMILY,
    NEAR_FIELD_OCCUPANCY_FAMILY,
    QUADRANT_OCCUPANCY_FAMILY,
    build_chaser_export_authority_set,
    build_chaser_export_source_authority,
    write_chaser_export_authority_set,
)
from fisheye.analytics_exports.contracts import (
    CHASER_BOUT_EVENTS_TABLE,
    CHASER_BOUT_HISTOGRAM_TABLE,
    CHASER_CENTER_DISTANCE_HISTOGRAM_TABLE,
    CHASER_EGOCENTRIC_HISTOGRAM_TABLE,
    CHASER_EGOCENTRIC_SUMMARY_TABLE,
    CHASER_EPOCH_BEHAVIOR_TABLE,
    CHASER_IBI_HISTOGRAM_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_CHASER_PHASE_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_DISTANCE_CDF_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_RADIAL_DENSITY_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_SUMMARY_TABLE,
    CHASER_QUADRANT_OCCUPANCY_CHASER_PHASE_TABLE,
    CHASER_QUADRANT_OCCUPANCY_DENSITY_TABLE,
    CHASER_QUADRANT_OCCUPANCY_SUMMARY_TABLE,
    CHASER_SPEED_DISTANCE_TABLE,
)
from fisheye.analytics_exports.publication import manifest_selected_part_files
from fisheye.analytics_exports.registry_identity import (
    build_registry_identity_receipt,
    build_registry_identity_source,
)
from fisheye.analytics_exports.validation import validate_export_run
from fisheye.analysis.chaser_component_publication import (
    build_chaser_component_handle,
)
from fisheye.analysis.chaser_distance_io import load_chaser_distance_run
from fisheye.analysis.chaser_distance_runs import write_chaser_distance_run
from fisheye.analysis.chaser_epoch_behavior_summary import (
    ChaserEpochBehaviorSummaryResult,
    LEGACY_EXECUTION_MODE,
    LEGACY_METHOD_VERSION,
    LEGACY_SCHEMA_ID,
    LEGACY_SCHEMA_VERSION,
    _load_windows,
    _make_center_distance_histogram,
    _make_per_epoch_bout_histograms,
    _make_per_epoch_bouts,
    _make_per_epoch_chaser,
    _make_per_epoch_fish,
    _make_per_epoch_inter_bout_interval_histograms,
    _resolve_arena_geometry,
    write_legacy_chaser_epoch_behavior_summary_component,
)
from fisheye.analysis.chaser_egocentric_bearing import (
    build_chaser_egocentric_bearing_result,
    write_chaser_egocentric_bearing_component,
)
from fisheye.analysis.chaser_near_field_occupancy import (
    build_chaser_near_field_occupancy_result,
    write_chaser_near_field_occupancy_component,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils.export_cross_recording_analytics import (
    _required_near_field_v2_phase_array,
    export_sources,
)
from tests.unit.fisheye.test_cra_near_field import (
    _add_circle_geometry,
    _quadrant_handle,
    _write_sources,
)
from tests.unit.fisheye.test_cra_primary_endpoint import _make_archive
from tests.unit.fisheye.test_chaser_egocentric_bearing import (
    _add_track_kinematics_run,
)
from tests.unit.fisheye.test_goodcopbadcop_interactive import (
    _make_archive_with_detection_occupancy,
    _make_chaser_result,
)
from tests.unit.fisheye.test_export_cross_recording_analytics import (
    _RegistryReceiptTestDouble,
)


pytestmark = pytest.mark.usefixtures("logical_chaser_distance_reader")


def _write_authority(
    tmp_path: Path,
    source: Path,
    *,
    component_handles: dict[str, dict[str, object]],
    name: str = "chaser-authority.json",
) -> Path:
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    snapshot = load_chaser_distance_run(root, run_name="chaser_distance_1")
    source_record = build_chaser_export_source_authority(
        zarr_path=source,
        recording_id=snapshot.recording_id,
        base_run_name=snapshot.run_name,
        base_publication_seal_sha256=snapshot.publication_seal_sha256,
        component_handles=component_handles,
    )
    return write_chaser_export_authority_set(
        tmp_path / name,
        build_chaser_export_authority_set([source_record]),
    )


def _read_rows(output_root: Path, export_run_id: str, table: str) -> list[dict]:
    parts = manifest_selected_part_files(output_root, export_run_id, table)
    assert parts, f"expected a persisted Parquet part for {table}"
    return pq.read_table([str(path) for path in parts]).to_pylist()


def _component_handle(source: Path, component_path: str) -> dict[str, object]:
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    snapshot = load_chaser_distance_run(root, run_name="chaser_distance_1")
    relative_path = component_path.removeprefix(f"{snapshot.run_path}/")
    return build_chaser_component_handle(
        root[component_path],
        snapshot=snapshot,
        relative_path=relative_path,
    )


def _registry_identity_receipt(*sources: Path) -> dict[str, object]:
    bindings: list[dict[str, object]] = []
    for source in sources:
        resolved = source.expanduser().resolve()
        if source.exists():
            root = zarr.open_group(str(source), mode="r", use_consolidated=False)
            if "analysis/chaser_distance_runs" in root:
                recording_id = load_chaser_distance_run(
                    root,
                    run_name="chaser_distance_1",
                ).recording_id
            else:
                recording_id = source.name.removesuffix("_analysis.zarr")
        else:
            recording_id = source.name.removesuffix("_analysis.zarr")
        bindings.append(
            build_registry_identity_source(
                zarr_path=resolved,
                rows=[
                    {
                        "dataset_id": len(bindings) + 1,
                        "recording_id": recording_id,
                        "experimental_session_id": "session-test",
                        "experimental_session_snapshot_id": (
                            "00000000-0000-4000-8000-000000000001"
                        ),
                        "experimental_session_schema_id": (
                            "palette.registry.experimental_session.v1"
                        ),
                        "experimental_session_creation_registry_schema_version": 1,
                        "experimental_session_identity_status": "explicit",
                        "experimental_session_assignment_snapshot_id": (
                            "00000000-0000-4000-8000-000000000002"
                        ),
                        "experimental_session_assignment_batch_id": (
                            "00000000-0000-4000-8000-000000000003"
                        ),
                        "experimental_session_assignment_revision": 1,
                        "experimental_session_supersedes_assignment_snapshot_id": None,
                        "experimental_session_assignment_schema_id": (
                            "palette.registry.experimental_session_assignment.v1"
                        ),
                        "experimental_session_assignment_registry_schema_version": 1,
                        "experimental_session_assignment_method": "manual_test",
                        "experimental_session_assigned_by": "test",
                        "experimental_session_assigned_at_utc": (
                            "2026-08-10T00:00:00+00:00"
                        ),
                        "fish_id": f"subject_{recording_id}",
                        "subject_count": 1,
                        "subject_ids_json": None,
                    }
                ],
            )
        )
    return build_registry_identity_receipt(
        registry_path=Path("/registry/test.sqlite"),
        sources=bindings,
    )


def test_sealed_base_speed_distance_exports_with_exact_authority_in_process_pool(
    tmp_path: Path,
) -> None:
    source = _make_archive_with_detection_occupancy(tmp_path)
    write_chaser_distance_run(
        source,
        _make_chaser_result(source),
        overwrite=True,
        legacy_compatibility=True,
    )
    authority_path = _write_authority(
        tmp_path,
        source,
        component_handles={},
    )
    authority_file_sha256 = hashlib.sha256(authority_path.read_bytes()).hexdigest()
    output = tmp_path / "exports"

    manifest = export_sources(
        [source],
        output_root=output,
        export_run_id="sealed_base_speed",
        tables=(CHASER_SPEED_DISTANCE_TABLE,),
        jobs=2,
        chaser_authority_manifest_path=authority_path,
        chaser_authority_sha256=authority_file_sha256,
        registry=_RegistryReceiptTestDouble(_registry_identity_receipt(source)),
    )

    assert manifest["row_counts_by_table"][CHASER_SPEED_DISTANCE_TABLE] > 0
    assert manifest["chaser_export_authority"]["file_sha256"] == authority_file_sha256
    source_binding = manifest["chaser_export_authority"]["resolved_sources"][str(source)]
    assert source_binding["base_run_name"] == "chaser_distance_1"
    assert source_binding["component_handles"] == {}
    rows = _read_rows(output, "sealed_base_speed", CHASER_SPEED_DISTANCE_TABLE)
    assert rows[0]["chaser_distance_run"] == "chaser_distance_1"
    assert rows[0]["distance_bin_left_mm"] == 0.0
    assert rows[0]["distance_bin_right_mm"] == 2.0
    assert validate_export_run(output, "sealed_base_speed")["status"] == "valid"


def test_rehashed_chaser_authority_receipt_tamper_fails_validation(
    tmp_path: Path,
) -> None:
    source = _make_archive_with_detection_occupancy(tmp_path)
    write_chaser_distance_run(
        source,
        _make_chaser_result(source),
        overwrite=True,
        legacy_compatibility=True,
    )
    authority_path = _write_authority(tmp_path, source, component_handles={})
    output = tmp_path / "exports"
    manifest = export_sources(
        [source],
        output_root=output,
        export_run_id="tampered-authority-receipt",
        tables=(CHASER_SPEED_DISTANCE_TABLE,),
        chaser_authority_manifest_path=authority_path,
        registry=_RegistryReceiptTestDouble(_registry_identity_receipt(source)),
    )
    manifest_path = Path(manifest["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = payload["chaser_export_authority"]
    receipt["resolved_sources"][str(source)]["base_run_name"] = "forged-run"
    receipt["payload_sha256"] = canonical_json_sha256(
        {key: value for key, value in receipt.items() if key != "payload_sha256"}
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from authority"):
        validate_export_run(output, "tampered-authority-receipt")


def test_quadrant_component_exports_all_three_tables_from_one_explicit_handle(
    tmp_path: Path,
) -> None:
    source = _make_archive(tmp_path)
    component_path = _write_sources(source)
    handle = _quadrant_handle(source, component_path)
    authority_path = _write_authority(
        tmp_path,
        source,
        component_handles={QUADRANT_OCCUPANCY_FAMILY: handle},
        name="quadrant-authority.json",
    )
    output = tmp_path / "exports"
    tables = (
        CHASER_QUADRANT_OCCUPANCY_SUMMARY_TABLE,
        CHASER_QUADRANT_OCCUPANCY_CHASER_PHASE_TABLE,
        CHASER_QUADRANT_OCCUPANCY_DENSITY_TABLE,
    )

    manifest = export_sources(
        [source],
        output_root=output,
        export_run_id="sealed_quadrant",
        tables=tables,
        jobs=1,
        chaser_authority_manifest_path=authority_path,
        registry=_RegistryReceiptTestDouble(_registry_identity_receipt(source)),
    )

    assert all(manifest["row_counts_by_table"][table] > 0 for table in tables)
    binding = manifest["chaser_export_authority"]["resolved_sources"][str(source)]
    assert (
        binding["component_handles"][QUADRANT_OCCUPANCY_FAMILY]["component_path"]
        == component_path
    )
    for table in tables:
        rows = _read_rows(output, "sealed_quadrant", table)
        assert rows
        assert rows[0]["cra_primary_endpoint_path"] == component_path
    assert validate_export_run(output, "sealed_quadrant")["status"] == "valid"


def test_near_field_component_exports_all_four_tables_from_one_explicit_handle(
    tmp_path: Path,
) -> None:
    source = _make_archive(tmp_path)
    quadrant_path = _write_sources(source)
    quadrant_handle = _quadrant_handle(source, quadrant_path)
    result = build_chaser_near_field_occupancy_result(
        source,
        chaser_distance_run="chaser_distance_1",
        quadrant_occupancy_dependency_handle=quadrant_handle,
        r_zone_mm=2.0,
        r_in_mm=2.0,
        r_out_mm=3.0,
        percentile_values=(5.0, 10.0),
        radial_bin_edges_mm=(0.0, 2.0, 4.0, 8.0),
        cdf_thresholds_mm=(2.0, 4.0),
        perimeter_band_mm=2.0,
        immobility_signal_mode="raw_centroid_explicit",
    )
    component_path = str(
        write_chaser_near_field_occupancy_component(
            source,
            result,
            overwrite=True,
            write_png=False,
            write_interactive_spec=False,
        )
    )
    handle = _component_handle(source, component_path)
    authority_path = _write_authority(
        tmp_path,
        source,
        component_handles={NEAR_FIELD_OCCUPANCY_FAMILY: handle},
        name="near-field-authority.json",
    )
    output = tmp_path / "exports"
    tables = (
        CHASER_NEAR_FIELD_OCCUPANCY_SUMMARY_TABLE,
        CHASER_NEAR_FIELD_OCCUPANCY_CHASER_PHASE_TABLE,
        CHASER_NEAR_FIELD_OCCUPANCY_RADIAL_DENSITY_TABLE,
        CHASER_NEAR_FIELD_OCCUPANCY_DISTANCE_CDF_TABLE,
    )

    manifest = export_sources(
        [source],
        output_root=output,
        export_run_id="sealed_near_field",
        tables=tables,
        jobs=1,
        chaser_authority_manifest_path=authority_path,
        registry=_RegistryReceiptTestDouble(_registry_identity_receipt(source)),
    )

    assert all(manifest["row_counts_by_table"][table] > 0 for table in tables)
    binding = manifest["chaser_export_authority"]["resolved_sources"][str(source)]
    assert (
        binding["component_handles"][NEAR_FIELD_OCCUPANCY_FAMILY]["component_path"]
        == component_path
    )
    exported = {
        table: _read_rows(output, "sealed_near_field", table) for table in tables
    }
    assert all(exported.values())
    assert (
        exported[CHASER_NEAR_FIELD_OCCUPANCY_SUMMARY_TABLE][0]["speed_source"]
        == "raw_centroid_explicit"
    )
    summary_row = exported[CHASER_NEAR_FIELD_OCCUPANCY_SUMMARY_TABLE][0]
    assert summary_row["visit_policy_version"] == (
        "valid_tracked_observed_transitions_v2"
    )
    assert summary_row["near_zone_entry_rate_denominator_semantics"] == (
        "valid_fish_to_chaser_distance_samples_divided_by_fps"
    )
    phase_rows = exported[CHASER_NEAR_FIELD_OCCUPANCY_CHASER_PHASE_TABLE]
    for row in phase_rows:
        assert row["near_zone_entry_rate_numerator_count"] >= 0
        assert row["near_zone_valid_tracked_duration_s"] >= 0.0
        assert (
            row["near_zone_entry_rate_denominator_duration_s"]
            == row["near_zone_valid_tracked_duration_s"]
        )
        assert row["near_zone_invalid_gap_count"] >= 0
        assert row["near_zone_censor_event_count"] >= 0
        assert row["near_zone_boundary_censor_event_count"] >= 0
        assert row["near_zone_invalid_gap_censor_event_count"] >= 0
    assert validate_export_run(output, "sealed_near_field")["status"] == "valid"


def test_near_field_v2_phase_array_requires_exact_shape_and_dtype() -> None:
    exact = {"metric": np.zeros((2, 3), dtype=np.int64)}
    values = _required_near_field_v2_phase_array(
        exact,
        "metric",
        shape=(2, 3),
        dtype=np.int64,
    )
    assert values.dtype == np.dtype(np.int64)

    with pytest.raises(ValueError, match="requires per_chaser_phase/missing"):
        _required_near_field_v2_phase_array(
            exact,
            "missing",
            shape=(2, 3),
            dtype=np.int64,
        )
    with pytest.raises(ValueError, match="has shape"):
        _required_near_field_v2_phase_array(
            {"metric": np.zeros((2, 2), dtype=np.int64)},
            "metric",
            shape=(2, 3),
            dtype=np.int64,
        )
    with pytest.raises(ValueError, match="has dtype"):
        _required_near_field_v2_phase_array(
            {"metric": np.zeros((2, 3), dtype=np.int32)},
            "metric",
            shape=(2, 3),
            dtype=np.int64,
        )


def test_egocentric_component_exports_both_tables_from_one_explicit_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_archive_with_detection_occupancy(tmp_path)
    write_chaser_distance_run(
        source,
        _make_chaser_result(source),
        overwrite=True,
        legacy_compatibility=True,
    )
    _add_track_kinematics_run(source, monkeypatch=monkeypatch)
    result = build_chaser_egocentric_bearing_result(
        source,
        chaser_distance_run="chaser_distance_1",
        track_kinematics_run="tk_1",
        distance_bin_width_mm=2.0,
        bearing_bin_width_deg=90.0,
    )
    component_path = str(
        write_chaser_egocentric_bearing_component(
            source,
            result,
            overwrite=True,
            write_png=False,
            write_interactive_spec=False,
        )
    )
    handle = _component_handle(source, component_path)
    authority_path = _write_authority(
        tmp_path,
        source,
        component_handles={EGOCENTRIC_BEARING_FAMILY: handle},
        name="egocentric-authority.json",
    )
    output = tmp_path / "exports"
    tables = (
        CHASER_EGOCENTRIC_SUMMARY_TABLE,
        CHASER_EGOCENTRIC_HISTOGRAM_TABLE,
    )

    manifest = export_sources(
        [source],
        output_root=output,
        export_run_id="sealed_egocentric",
        tables=tables,
        jobs=1,
        chaser_authority_manifest_path=authority_path,
        registry=_RegistryReceiptTestDouble(_registry_identity_receipt(source)),
    )

    assert all(manifest["row_counts_by_table"][table] > 0 for table in tables)
    binding = manifest["chaser_export_authority"]["resolved_sources"][str(source)]
    assert (
        binding["component_handles"][EGOCENTRIC_BEARING_FAMILY]["component_path"]
        == component_path
    )
    assert all(
        _read_rows(output, "sealed_egocentric", table)
        for table in tables
    )
    assert validate_export_run(output, "sealed_egocentric")["status"] == "valid"


def test_epoch_behavior_component_exports_all_five_tables_from_one_explicit_handle(
    tmp_path: Path,
) -> None:
    source = _make_archive_with_detection_occupancy(tmp_path)
    write_chaser_distance_run(
        source,
        _make_chaser_result(source),
        overwrite=True,
        legacy_compatibility=True,
    )
    _add_circle_geometry(source)
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    snapshot = load_chaser_distance_run(root, run_name="chaser_distance_1")
    run = root[snapshot.run_path]
    windows = _load_windows(run, fps=snapshot.fps)
    geometry, _geometry_notes = _resolve_arena_geometry(root, run)
    empty_bouts = _make_per_epoch_bouts(
        windows=windows,
        swim_tables=None,
        track=None,
        fps=snapshot.fps,
    )
    per_epoch_bouts = np.zeros(1, dtype=empty_bouts.dtype)
    per_epoch_bouts["window_id"] = windows[0].window_id
    per_epoch_bouts["window_index"] = 0
    per_epoch_bouts["window_label"] = windows[0].label.encode("utf-8")
    per_epoch_bouts["start_frame"] = windows[0].start_frame
    per_epoch_bouts["end_frame"] = windows[0].end_frame
    per_epoch_bouts["start_time_s"] = windows[0].start_time_s
    per_epoch_bouts["end_time_s"] = windows[0].end_time_s
    per_epoch_bouts["duration_s"] = windows[0].duration_s
    per_epoch_bouts["bout_source_row"] = 0
    per_epoch_bouts["bout_id"] = 7
    per_epoch_bouts["bout_event_frame"] = windows[0].start_frame
    per_epoch_bouts["bout_event_time_s"] = windows[0].start_time_s
    per_epoch_bouts["bout_start_frame"] = windows[0].start_frame
    per_epoch_bouts["bout_end_frame"] = windows[0].start_frame + 1
    per_epoch_bouts["bout_start_time_s"] = windows[0].start_time_s
    per_epoch_bouts["bout_end_time_s"] = windows[0].start_time_s + 0.1
    per_epoch_bouts["bout_duration_s"] = 0.1
    per_epoch_bouts["bout_path_length_mm"] = 0.5
    per_epoch_bouts["bout_net_heading_change_deg"] = 5.0
    per_epoch_bouts["abs_bout_net_heading_change_deg"] = 5.0
    per_epoch_bouts["bout_heading_path_deg"] = 5.0
    interval_rows = np.asarray(
        [(0.1, windows[0].start_frame, windows[0].start_frame + 1, True)],
        dtype=[
            ("interval_s", np.float64),
            ("prev_end_frame", np.int64),
            ("next_start_frame", np.int64),
            ("valid", np.bool_),
        ],
    )
    interval_source = SimpleNamespace(inter_bout_intervals=interval_rows)
    result = ChaserEpochBehaviorSummaryResult(
        zarr_path=str(source),
        recording_id=snapshot.recording_id,
        component_name="export_fixture_v1",
        execution_mode=LEGACY_EXECUTION_MODE,
        schema_id=LEGACY_SCHEMA_ID,
        schema_version=LEGACY_SCHEMA_VERSION,
        method_version=LEGACY_METHOD_VERSION,
        chaser_distance_run_name=snapshot.run_name,
        chaser_distance_run_path=snapshot.run_path,
        source_track_kinematics_run=None,
        source_track_kinematics_scope=None,
        source_track_kinematics_track_id=None,
        source_track_kinematics_track_path=None,
        source_speed_level="filtered",
        source_speed_level_selection="sealed_export_fixture",
        source_swim_bout_run="fixture_bouts",
        source_swim_bout_path="analysis/swim_bout_runs/fixture_bouts",
        source_swim_bout_level_path="levels/filtered",
        source_swim_bout_signal_level="filtered",
        fps=snapshot.fps,
        windows=windows,
        per_epoch_fish=_make_per_epoch_fish(
            windows=windows,
            run_group=run,
            swim_tables=None,
            track=None,
            source_speed_level=None,
            geometry=geometry,
            wall_band_mm=5.0,
            fps=snapshot.fps,
            execution_mode=LEGACY_EXECUTION_MODE,
        ),
        per_epoch_chaser=_make_per_epoch_chaser(
            windows=windows,
            run_group=run,
        ),
        per_epoch_bouts=per_epoch_bouts,
        per_epoch_bout_histograms=_make_per_epoch_bout_histograms(
            windows=windows,
            per_epoch_bouts=per_epoch_bouts,
        ),
        per_epoch_inter_bout_interval_histograms=(
            _make_per_epoch_inter_bout_interval_histograms(
                windows=windows,
                swim_tables=interval_source,
            )
        ),
        center_distance_histogram=_make_center_distance_histogram(
            windows=windows,
            run_group=run,
            geometry=geometry,
            bin_width_mm=2.5,
            wall_band_mm=5.0,
        ),
        arena_geometry=geometry,
        center_distance_bin_width_mm=2.5,
        wall_band_mm=5.0,
        warnings=(),
    )
    component_path = str(
        write_legacy_chaser_epoch_behavior_summary_component(
            source,
            result,
            overwrite=True,
        )
    )
    handle = _component_handle(source, component_path)
    authority_path = _write_authority(
        tmp_path,
        source,
        component_handles={EPOCH_BEHAVIOR_FAMILY: handle},
        name="epoch-behavior-authority.json",
    )
    output = tmp_path / "exports"
    tables = (
        CHASER_EPOCH_BEHAVIOR_TABLE,
        CHASER_BOUT_EVENTS_TABLE,
        CHASER_BOUT_HISTOGRAM_TABLE,
        CHASER_IBI_HISTOGRAM_TABLE,
        CHASER_CENTER_DISTANCE_HISTOGRAM_TABLE,
    )

    manifest = export_sources(
        [source],
        output_root=output,
        export_run_id="sealed_epoch_behavior",
        tables=tables,
        jobs=1,
        chaser_authority_manifest_path=authority_path,
        registry=_RegistryReceiptTestDouble(_registry_identity_receipt(source)),
    )

    assert all(manifest["row_counts_by_table"][table] > 0 for table in tables)
    binding = manifest["chaser_export_authority"]["resolved_sources"][str(source)]
    assert (
        binding["component_handles"][EPOCH_BEHAVIOR_FAMILY]["component_path"]
        == component_path
    )
    assert all(
        _read_rows(output, "sealed_epoch_behavior", table)
        for table in tables
    )
    assert validate_export_run(output, "sealed_epoch_behavior")["status"] == "valid"


def test_supported_chaser_table_requires_authority_before_staging(
    tmp_path: Path,
) -> None:
    source = _make_archive_with_detection_occupancy(tmp_path)

    with pytest.raises(ValueError, match="exact chaser authority manifest"):
        export_sources(
            [source],
            output_root=tmp_path / "exports",
            export_run_id="missing_authority",
            tables=(CHASER_SPEED_DISTANCE_TABLE,),
            registry=_RegistryReceiptTestDouble(_registry_identity_receipt(source)),
        )

    assert not (tmp_path / "exports").exists()


def test_chaser_authority_source_set_must_match_before_staging(
    tmp_path: Path,
) -> None:
    source = _make_archive_with_detection_occupancy(tmp_path)
    write_chaser_distance_run(
        source,
        _make_chaser_result(source),
        overwrite=True,
        legacy_compatibility=True,
    )
    authority_path = _write_authority(
        tmp_path,
        source,
        component_handles={},
    )
    extra_source = tmp_path / "extra_analysis.zarr"

    with pytest.raises(ValueError, match="source set must exactly match"):
        export_sources(
            [source, extra_source],
            output_root=tmp_path / "exports",
            export_run_id="source_mismatch",
            tables=(CHASER_SPEED_DISTANCE_TABLE,),
            chaser_authority_manifest_path=authority_path,
            registry=_RegistryReceiptTestDouble(
                _registry_identity_receipt(source, extra_source)
            ),
        )

    assert not (tmp_path / "exports").exists()
