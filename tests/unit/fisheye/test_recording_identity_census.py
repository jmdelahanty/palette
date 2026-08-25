from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import fisheye.registry.recording_identity_census as census_module
from fisheye.registry.recording_identity_census import (
    CensusError,
    _scan_parquet_identity,
    _scan_recording_directory,
    _scan_zarr_metadata,
    _stable_read_json,
    main,
    run_census,
    write_report_read_only_safe,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _make_fixture(tmp_path: Path) -> dict[str, Path]:
    """Create a small registry plus path-bound direct Zarr metadata.

    The fixture deliberately keeps the three identity fields separate.  In
    particular, ``session_id`` is a legacy alias and must not be promoted to
    ``session_uuid`` by a diagnostic.
    """

    recording_dir = tmp_path / "recordings" / "rec_a"
    zarr_path = recording_dir / "zarr" / "rec_a_analysis.zarr"
    manifest_path = recording_dir / "recording_manifest.json"
    recording_session_path = recording_dir / "recording_session.json"
    zarr_metadata_path = zarr_path / "zarr.json"

    _write_json(
        manifest_path,
        {
            "recording_id": "rec_a",
            "session_uuid": "session-a",
            "session_id": "legacy-session-a",
            "recording_name": "rec_a",
        },
    )
    _write_json(
        zarr_metadata_path,
        {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "recording_id": "rec_a",
                "session_uuid": "session-a",
                "session_id": "legacy-session-a",
            },
        },
    )
    _write_json(
        recording_session_path,
        {
            "session_id": "legacy-session-a",
            "acquisition_index_mapping": {"recording_id": "session-a"},
        },
    )

    registry_path = tmp_path / "palette_registry.sqlite"
    conn = sqlite3.connect(registry_path)
    try:
        conn.executescript(
            """
            CREATE TABLE recordings (
                recording_id TEXT PRIMARY KEY,
                session_uuid TEXT,
                recording_name TEXT,
                recording_path TEXT
            );
            CREATE TABLE datasets (
                dataset_id TEXT PRIMARY KEY,
                session_uuid TEXT,
                recording_id TEXT,
                zarr_path TEXT,
                artifact_kind TEXT,
                zarr_use TEXT,
                status TEXT,
                source_layout TEXT,
                source_frame_index_schema TEXT
            );
            CREATE TABLE recording_step_status (
                dataset_id TEXT,
                recording_id TEXT,
                session_uuid TEXT,
                step_name TEXT
            );
            CREATE VIEW recording_identity_projection AS
                SELECT dataset_id, recording_id, session_uuid
                FROM recording_step_status;
            """
        )
        conn.executemany(
            "INSERT INTO recordings(recording_id,session_uuid,recording_name,recording_path) "
            "VALUES(?,?,?,?)",
            (
                ("rec_a", "session-a", "rec_a", str(recording_dir)),
                # No dataset points here: this is an intentional orphan.
                ("rec_orphan", "session-orphan", "orphan", str(tmp_path / "orphan")),
            ),
        )
        conn.executemany(
            "INSERT INTO datasets(dataset_id,session_uuid,recording_id,zarr_path,artifact_kind,zarr_use,status,source_layout,source_frame_index_schema) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                ("d-analysis", "session-a", "rec_a", str(zarr_path), "source_recording", "analysis", "active", "analysis_zarr", None),
                # The dataset claims a different session from its recordings row.
                ("d-conflict", "session-dataset", "rec_a", str(tmp_path / "missing.zarr"), "source_recording", "analysis", "active", "analysis_zarr", None),
                # A null recording_id must remain visible as a projection defect.
                ("d-null", "session-null", None, str(tmp_path / "null.zarr"), "source_recording", "analysis", "active", "analysis_zarr", None),
            ),
        )
        conn.executemany(
            "INSERT INTO recording_step_status(dataset_id,recording_id,session_uuid,step_name) "
            "VALUES(?,?,?,?)",
            (
                ("d-analysis", "rec_a", "session-a", "analysis"),
                ("d-conflict", "rec_a", "session-projection", "analysis"),
                ("d-null", None, "session-null", "analysis"),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "registry": registry_path,
        "recording_dir": recording_dir,
        "zarr": zarr_path,
        "manifest": manifest_path,
        "recording_session": recording_session_path,
        "zarr_metadata": zarr_metadata_path,
    }


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _contains(report: object, *tokens: str) -> bool:
    rendered = _canonical(report)
    return all(token in rendered for token in tokens)


def _objects(report: object) -> list[dict[str, Any]]:
    """Return all dictionaries that look like registry-object descriptors."""

    found: list[dict[str, Any]] = []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            if any(key in value for key in ("object_name", "table_name", "relation_name")):
                found.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(report)
    return found


def test_run_census_reports_all_identity_surfaces_without_promoting_aliases(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)

    report = run_census(
        fixture["registry"],
        scan_artifacts=True,
        max_zarr_nodes=100,
    )

    census = report["census"]
    assert census["schema_id"] == "palette.recording_identity_census.v1"
    assert census["read_only"] is True
    assert census["authorizes_mutation"] is False
    assert census["artifacts"]["metadata_view"] == "direct_unconsolidated"

    # Dynamic introspection must cover both the table and the derived view,
    # rather than treating only datasets/recordings as the registry surface.
    names = {
        str(item.get(key))
        for item in _objects(census)
        for key in ("object_name", "table_name", "relation_name")
        if item.get(key) is not None
    }
    assert {"recordings", "datasets", "recording_step_status", "recording_identity_projection"} <= names

    # All three fields are observable independently.  The legacy session_id
    # must not silently become a session_uuid observation.
    assert _contains(
        census,
        "recording_id",
        "rec_a",
        "session_uuid",
        "session-a",
        "session_id",
        "legacy-session-a",
    )
    assert _contains(census, "conflict", "session-dataset", "session-a")
    assert _contains(census, "null", "d-null")
    assert _contains(census, "orphan", "rec_orphan")


def test_run_census_binds_artifact_evidence_to_registered_paths(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)

    report = run_census(fixture["registry"], scan_artifacts=True, max_zarr_nodes=100)
    rendered = _canonical(report)

    # The direct metadata scan is path-bound to the dataset's exact zarr_path;
    # a basename/context hint is not sufficient evidence.
    assert str(fixture["zarr"]) in rendered
    assert str(fixture["manifest"]) in rendered
    assert str(fixture["zarr_metadata"]) in rendered
    assert "zarr_v3_group_attrs" in rendered
    assert "direct" in rendered

    # The report is observational: it may identify a candidate value, but it
    # must never emit an effective identity or a mutation plan.
    assert "effective_recording_id" not in rendered
    assert "mutation_plan" not in rendered
    assert report["census"]["effective_identity_values_emitted"] is False


def test_run_census_is_deterministic_and_does_not_modify_inputs(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    input_paths = [
        fixture["registry"],
        fixture["manifest"],
        fixture["recording_session"],
        fixture["zarr_metadata"],
    ]
    before = {path: path.read_bytes() for path in input_paths}

    first = run_census(fixture["registry"], scan_artifacts=True, max_zarr_nodes=100)
    second = run_census(fixture["registry"], scan_artifacts=True, max_zarr_nodes=100)

    # Reports must be stable enough to hash and compare.  Wall-clock fields,
    # temporary snapshot paths, and unordered SQL traversal do not belong in
    # the canonical result.
    assert _canonical(first["census"]) == _canonical(second["census"])
    assert first["census"]["report_digest"] == second["census"]["report_digest"]

    for path, expected in before.items():
        assert path.read_bytes() == expected, path


def test_main_writes_a_report_without_touching_registry_or_metadata(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    output = tmp_path / "census.json"
    input_paths = [
        fixture["registry"],
        fixture["manifest"],
        fixture["recording_session"],
        fixture["zarr_metadata"],
    ]
    before = {path: path.read_bytes() for path in input_paths}

    exit_code = main(
        [
            "--registry",
            str(fixture["registry"]),
            "--output",
            str(output),
            "--max-zarr-nodes",
            "100",
        ]
    )

    # The fixture intentionally contains conflicts/missing bindings, so the
    # diagnostic should complete but return its findings status.
    assert exit_code == 1
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["census"]["read_only"] is True
    assert written["census"]["authorizes_mutation"] is False
    assert _contains(written["census"], "rec_a", "rec_orphan", "d-null")
    for path, expected in before.items():
        assert path.read_bytes() == expected, path


def test_cross_artifact_conflict_is_reported_but_nested_alias_is_not_promoted(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    manifest["session_uuid"] = "manifest-conflict"
    _write_json(fixture["manifest"], manifest)

    census = run_census(fixture["registry"], scan_artifacts=True)["census"]
    scope = next(
        item
        for item in census["artifacts"]["zarr_scopes"]
        if item["path"] == str(fixture["zarr"])
    )
    conflicts = {
        item["semantic_fact"]: item["distinct_values"]
        for item in scope["classifications"]
        if item["status"] == "conflict"
    }
    assert conflicts["session_uuid"] == ["manifest-conflict", "session-a"]
    assert "recording_id" not in conflicts
    recording_scope = next(
        item
        for item in census["artifacts"]["recording_directory_scopes"]
        if item["path"] == str(fixture["recording_dir"])
    )
    assert _contains(recording_scope["observations"], "nested_recording_id", "session-a")


def test_parquet_row_count_is_scale_not_a_finding(tmp_path: Path) -> None:
    path = tmp_path / "recording_frame_index.parquet"
    pq.write_table(
        pa.table(
            {
                "recording_id": ["rec-a"] * 20_000,
                "session_id": ["legacy-session"] * 20_000,
                "camera_serial": ["cam-1"] * 20_000,
            }
        ),
        path,
        row_group_size=2_000,
    )

    report = _scan_parquet_identity(path)

    assert report["status"] == "complete"
    assert report["row_count"] == 20_000
    assert report["findings"] == []
    assert report["distinct_counts"] == {
        "camera_serial": 1,
        "recording_id": 1,
        "session_id": 1,
    }
    assert report["binding_kind"] == "identity_projection_digest_with_pre_post_stat_fence"


def test_json_duplicate_keys_and_non_finite_values_fail_closed(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"recording_id":"a","recording_id":"b"}', encoding="utf-8")
    non_finite = tmp_path / "nan.json"
    non_finite.write_text('{"recording_id":NaN}', encoding="utf-8")

    with pytest.raises(CensusError, match="duplicate object key"):
        _stable_read_json(duplicate, max_bytes=1_024)
    with pytest.raises(CensusError, match="non-finite JSON number"):
        _stable_read_json(non_finite, max_bytes=1_024)


def test_output_cannot_overwrite_or_enter_an_observed_input(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    report = run_census(fixture["registry"], scan_artifacts=True)

    with pytest.raises(CensusError, match="observed input"):
        write_report_read_only_safe(
            report,
            fixture["zarr_metadata"],
            registry_path=fixture["registry"],
        )
    with pytest.raises(CensusError, match="observed input"):
        write_report_read_only_safe(
            report,
            fixture["zarr"] / "new-report.json",
            registry_path=fixture["registry"],
        )


def test_default_scope_defers_unmarked_dataset_even_with_root_schema_marker(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    unmarked_zarr = tmp_path / "unmarked" / "unmarked.zarr"
    _write_json(
        unmarked_zarr / "zarr.json",
        {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "recording_id": "rec_a",
                "session_uuid": "session-a",
                "artifact_schema_id": "recording_analysis_v1",
            },
        },
    )
    conn = sqlite3.connect(fixture["registry"])
    try:
        conn.execute(
            "INSERT INTO datasets(dataset_id,session_uuid,recording_id,zarr_path,artifact_kind,zarr_use,status,source_layout,source_frame_index_schema) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                "d-unmarked",
                "session-a",
                "rec_a",
                str(unmarked_zarr),
                "source_recording",
                "analysis",
                "active",
                None,
                None,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    census = run_census(fixture["registry"], scan_artifacts=True)["census"]

    assert census["artifacts"]["scope_policy"] == "explicit_source_layout"
    assert census["artifacts"]["selected_dataset_count"] == 3
    assert str(unmarked_zarr) not in _canonical(census["artifacts"])
    assert census["registry"]["finding_scope"]["unmarked_outside_scope_deferred"] is True


def test_run_projection_and_arbitrary_analytics_metadata_do_not_become_root_conflicts(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    root_payload = json.loads(fixture["zarr_metadata"].read_text(encoding="utf-8"))
    root_payload["attributes"]["deep_metrics"] = {
        "provenance": {
            "recording_id": "not-an-artifact-identity",
            "source_zarr": "/unrelated/provenance/path.zarr",
        }
    }
    _write_json(fixture["zarr_metadata"], root_payload)
    _write_json(
        fixture["zarr"] / "analysis" / "zarr.json",
        {"zarr_format": 3, "node_type": "group", "attributes": {}},
    )
    _write_json(
        fixture["zarr"] / "analysis" / "example_runs" / "zarr.json",
        {"zarr_format": 3, "node_type": "group", "attributes": {}},
    )
    _write_json(
        fixture["zarr"] / "analysis" / "example_runs" / "run-a" / "zarr.json",
        {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {"recording_id": "legacy-run-recording"},
        },
    )

    census = run_census(
        fixture["registry"],
        scan_artifacts=True,
        dataset_ids=["d-analysis"],
    )["census"]
    scope = census["artifacts"]["zarr_scopes"][0]

    assert scope["scan_status"] == "complete"
    assert not any(item["code"] == "artifact_coverage_capped" for item in scope["findings"])
    assert not any(item["code"] == "artifact_identity_conflict" for item in scope["findings"])
    assert census["artifacts"]["donor_declaration_summary"]["unique_declaration_count"] == 0
    assert any(
        item["value"] == "legacy-run-recording"
        and str(item["comparison_domain"]).startswith("run:")
        for item in scope["observations"]
    )


def test_parquet_identity_cardinality_is_bounded_independently_of_row_count(
    tmp_path: Path,
) -> None:
    path = tmp_path / "recording_frame_index.parquet"
    pq.write_table(
        pa.table(
            {
                "recording_id": [f"rec-{index}" for index in range(100)],
                "session_id": ["legacy-session"] * 100,
                "camera_serial": ["cam-1"] * 100,
            }
        ),
        path,
        row_group_size=100,
    )

    report = _scan_parquet_identity(path, max_distinct_values=8)

    assert report["status"] == "capped"
    assert report["row_count"] == 100
    assert report["distinct_counts"]["recording_id"] == 8
    assert report["distinct_overflow_counts"]["recording_id"] == 92
    assert any(
        item["code"] == "parquet_identity_cardinality_capped"
        for item in report["findings"]
    )


def test_oversized_inline_root_metadata_uses_bounded_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "oversized.zarr"
    zarr_path.mkdir()
    prefix = (
        '{"zarr_format":3,"node_type":"group","attributes":'
        '{"recording_id":"rec-a","session_uuid":"session-a",'
        '"artifact_schema_id":"recording_analysis_v1"},'
        '"consolidated_metadata":'
    )
    (zarr_path / "zarr.json").write_text(
        prefix + json.dumps({"kind": "inline", "metadata": "x" * 8_000}) + "}",
        encoding="utf-8",
    )
    monkeypatch.setattr(census_module, "ROOT_PREFIX_READ_THRESHOLD_BYTES", 256)
    monkeypatch.setattr(census_module, "OVERSIZED_ROOT_METADATA_BYTES", 512)

    report = _scan_zarr_metadata(
        zarr_path,
        max_json_bytes=4_096,
        max_zarr_nodes=10,
        max_observations=100,
    )

    assert report["status"] == "complete"
    assert report["root_prefix_only"] is True
    assert report["contract_markers"]["artifact_schema_id"] == "recording_analysis_v1"
    assert any(item["code"] == "oversized_inline_root_metadata" for item in report["findings"])
    assert report["source_fences"]["binding_kinds"] == {"prefix_sha256": 1}


def test_registry_only_mode_is_explicitly_incomplete(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    report = run_census(fixture["registry"], scan_artifacts=False)["census"]

    assert report["declared_scope_complete"] is False
    assert any(item["code"] == "artifact_scan_deferred" for item in report["findings"])


def test_shared_zarr_path_is_an_ambiguous_binding(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    conn = sqlite3.connect(fixture["registry"])
    try:
        conn.execute(
            "INSERT INTO datasets(dataset_id,session_uuid,recording_id,zarr_path,artifact_kind,zarr_use,status,source_layout,source_frame_index_schema) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                "d-alias",
                "session-a",
                "rec_a",
                str(fixture["zarr"]),
                "source_recording",
                "analysis",
                "active",
                "analysis_zarr",
                None,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    census = run_census(
        fixture["registry"],
        scan_artifacts=True,
        dataset_ids=["d-analysis", "d-alias"],
    )["census"]
    scope = census["artifacts"]["zarr_scopes"][0]

    assert scope["scan_status"] == "ambiguous_binding"
    assert scope["coverage_complete"] is False
    assert any(item["code"] == "ambiguous_zarr_path_binding" for item in scope["findings"])


def test_explicit_null_required_artifact_identities_are_findings(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    manifest["recording_id"] = None
    _write_json(fixture["manifest"], manifest)
    root = json.loads(fixture["zarr_metadata"].read_text(encoding="utf-8"))
    root["attributes"]["session_uuid"] = None
    _write_json(fixture["zarr_metadata"], root)

    census = run_census(
        fixture["registry"],
        scan_artifacts=True,
        dataset_ids=["d-analysis"],
    )["census"]

    missing = [
        finding
        for finding in census["findings"]
        if finding["code"] == "artifact_identity_field_missing"
    ]
    assert {finding["locator"]["field"] for finding in missing} == {
        "recording_id",
        "session_uuid",
    }
    assert all(finding["detail"]["presence"] == "present_but_empty" for finding in missing)


def test_missing_recording_manifest_makes_declared_scope_incomplete(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    fixture["manifest"].unlink()

    census = run_census(
        fixture["registry"],
        scan_artifacts=True,
        dataset_ids=["d-analysis"],
    )["census"]
    recording_scope = census["artifacts"]["recording_directory_scopes"][0]

    assert recording_scope["scan_status"] == "incomplete"
    assert recording_scope["coverage_complete"] is False
    assert census["declared_scope_complete"] is False
    assert any(
        finding["code"] == "recording_manifest_missing"
        for finding in recording_scope["findings"]
    )


def test_missing_recording_path_makes_linked_scope_incomplete(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    conn = sqlite3.connect(fixture["registry"])
    try:
        conn.execute("UPDATE recordings SET recording_path = NULL WHERE recording_id = ?", ("rec_a",))
        conn.commit()
    finally:
        conn.close()

    census = run_census(
        fixture["registry"],
        scan_artifacts=True,
        dataset_ids=["d-analysis"],
    )["census"]
    scope = census["artifacts"]["zarr_scopes"][0]

    assert scope["scan_status"] == "incomplete_linkage"
    assert scope["coverage_complete"] is False
    assert census["declared_scope_complete"] is False
    assert any(
        finding["code"] == "recording_directory_binding_unavailable"
        and finding["detail"]["reason"] == "recording_path_missing"
        for finding in scope["findings"]
    )


def test_relative_zarr_locator_is_rejected_not_resolved_from_cwd(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    conn = sqlite3.connect(fixture["registry"])
    try:
        conn.execute(
            "UPDATE datasets SET zarr_path = ? WHERE dataset_id = ?",
            ("relative/archive.zarr", "d-analysis"),
        )
        conn.commit()
    finally:
        conn.close()

    census = run_census(
        fixture["registry"],
        scan_artifacts=True,
        dataset_ids=["d-analysis"],
    )["census"]
    scope = census["artifacts"]["zarr_scopes"][0]

    assert scope["scan_status"] == "invalid_locator"
    assert scope["coverage_complete"] is False
    finding = next(item for item in scope["findings"] if item["code"] == "invalid_zarr_locator")
    assert finding["detail"]["defects"] == ["relative_path"]


def test_shared_recording_directory_is_an_ambiguous_binding(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    second_zarr = tmp_path / "recordings" / "rec_b_analysis.zarr"
    _write_json(
        second_zarr / "zarr.json",
        {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "recording_id": "rec_b",
                "session_uuid": "session-b",
            },
        },
    )
    conn = sqlite3.connect(fixture["registry"])
    try:
        conn.execute(
            "INSERT INTO recordings(recording_id,session_uuid,recording_name,recording_path) "
            "VALUES(?,?,?,?)",
            ("rec_b", "session-b", "rec_b", str(fixture["recording_dir"])),
        )
        conn.execute(
            "INSERT INTO datasets(dataset_id,session_uuid,recording_id,zarr_path,artifact_kind,zarr_use,status,source_layout,source_frame_index_schema) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                "d-b",
                "session-b",
                "rec_b",
                str(second_zarr),
                "source_recording",
                "analysis",
                "active",
                "analysis_zarr",
                None,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    census = run_census(
        fixture["registry"],
        scan_artifacts=True,
        dataset_ids=["d-analysis", "d-b"],
    )["census"]
    recording_scope = census["artifacts"]["recording_directory_scopes"][0]

    assert recording_scope["scan_status"] == "ambiguous_binding"
    assert recording_scope["coverage_complete"] is False
    assert recording_scope["recording_ids"] == ["rec_a", "rec_b"]
    assert any(
        finding["code"] == "ambiguous_recording_path_binding"
        for finding in recording_scope["findings"]
    )


def test_declared_clip_manifest_observation_cap_makes_scope_incomplete(
    tmp_path: Path,
) -> None:
    recording_dir = tmp_path / "recording"
    _write_json(
        recording_dir / "recording_manifest.json",
        {"recording_id": "rec-a", "session_uuid": "session-a"},
    )
    _write_json(
        recording_dir / "recording_clip_index.json",
        {"clips": [{"clip_id": "clip-a"}]},
    )
    clip_manifest = recording_dir / "clips" / "clip-a" / "clip_manifest.json"
    _write_json(
        clip_manifest,
        {"recording_ids": [f"rec-{index}" for index in range(20)]},
    )

    report = _scan_recording_directory(
        recording_dir,
        max_json_bytes=64 * 1024,
        max_observations=8,
        scan_parquet=False,
    )

    source = next(item for item in report["sources"] if item["path"] == str(clip_manifest))
    assert source["truncated"] is True
    assert report["status"] == "incomplete"
    assert report["coverage_complete"] is False
    assert any(
        finding["code"] == "artifact_coverage_capped"
        and finding["locator"]["path"] == str(clip_manifest)
        for finding in report["findings"]
    )
