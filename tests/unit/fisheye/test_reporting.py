from __future__ import annotations

from io import BytesIO
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import zarr
from PIL import Image
from zarr.storage import MemoryStore

from fisheye.reporting import (
    PROVIDERS,
    PlanStatus,
    build_semantic_visualization_montages,
    check_report_manifest,
    export_report_bundle,
    build_report_plan,
    plan_recording_report,
    query_report_recordings,
    report_plan_to_dict,
    verify_report_manifest_sha256,
    index_report_manifest,
    query_indexed_reports,
    report_output_dir,
    resolve_analytics_export_binding,
)
from fisheye.registry.db import Registry
from fisheye.reporting import execution as reporting_execution
from fisheye.reporting.models import SelectedRecording
from fisheye.reporting.models import ReportPlan


def _root() -> zarr.Group:
    return zarr.open_group(store=MemoryStore(), mode="w")


def _recording(recording_id: str = "rec") -> SelectedRecording:
    return SelectedRecording(
        dataset_id=f"dataset_{recording_id}",
        recording_id=recording_id,
        zarr_path=f"/tmp/{recording_id}_analysis.zarr",
        protocol_name=None,
    )


def _run(root: zarr.Group, parent_path: str, run_id: str) -> zarr.Group:
    parent = root.require_group(parent_path)
    run = parent.require_group(run_id)
    parent.attrs["latest"] = run_id
    return run


def _artifact(
    run: zarr.Group,
    relative_path: str,
    *,
    contract: str | None,
    renderer: str | None,
    renderer_version: str | None,
) -> None:
    parts = relative_path.split("/")
    parent = run
    for part in parts[:-1]:
        parent = parent.require_group(part)
    array = parent.create_array(
        parts[-1],
        data=np.asarray([137, 80, 78, 71], dtype=np.uint8),
        overwrite=True,
    )
    array.attrs.update(
        {
            "artifact_type": "visualization",
            "artifact_role": "snapshot",
            "content_sha256": "abc",
        }
    )
    if contract is not None:
        array.attrs["visualization_contract_id"] = contract
    if renderer is not None:
        array.attrs["renderer"] = renderer
    if renderer_version is not None:
        array.attrs["renderer_version"] = renderer_version


def _track_run(root: zarr.Group, track_ids: tuple[int, ...]) -> zarr.Group:
    run = _run(root, "analysis/track_kinematics_runs/offline", "tk")
    tracks = run.require_group("tracks")
    for track_id in track_ids:
        tracks.require_group(f"id_{track_id}")
    return run


def _stimulus_run(root: zarr.Group, modes: tuple[str, ...]) -> zarr.Group:
    run = _run(root, "analysis/stimulus_runs", "stimulus")
    steps = run.require_group("steps")
    for index, mode in enumerate(modes):
        step = steps.require_group(f"step_{index}")
        step.attrs.update(
            {
                "step_index": index,
                "step_name": f"step {index}",
                "stimulus_mode": mode,
                "start_camera_frame": index * 100,
                "end_camera_frame": (index + 1) * 100,
                "duration_s": 10.0,
            }
        )
    return run


def _configure_chasers(run: zarr.Group, count: int) -> None:
    run.attrs["protocol_json"] = {
        "protocol_name": "Chaser example",
        "steps": [
            {
                "parameters": {
                    "chasers": [
                        {
                            "chaser_index": index,
                            "enable_chase": index == 0,
                            "enable_random_movement": index == 1,
                        }
                        for index in range(count)
                    ]
                }
            }
        ],
    }


def _item(plan, visualization_id: str, entity_id: str | None = None):
    return next(
        item
        for item in plan.items
        if item.visualization_id == visualization_id and item.entity_id == entity_id
    )


def test_no_stimulus_plan_uses_core_provider_and_full_track_cardinality() -> None:
    root = _root()
    track = _track_run(root, (0, 1))
    _artifact(
        track,
        "visualizations/track_kinematics_summary_track_0_png",
        contract="palette.core.track_kinematics.summary.v1",
        renderer="palette-track-kinematics-summary-v1",
        renderer_version="1",
    )

    plan = plan_recording_report(_recording(), root_opener=lambda _path: root)

    assert plan.stimulus_steps == ()
    assert plan.track_ids == ("0", "1")
    assert [provider.provider_id for provider in plan.providers if provider.applicable] == [
        "core_behavior.v1"
    ]
    assert _item(plan, "core.track_kinematics.overview", "0").status == PlanStatus.READY
    assert _item(plan, "core.track_kinematics.overview", "1").status == PlanStatus.NEEDS_RENDER
    assert _item(plan, "core.position.xy_trace", "0").status == PlanStatus.NEEDS_RENDER
    assert _item(plan, "core.swim_bouts.overview", "0").status == PlanStatus.NEEDS_ANALYSIS
    assert _item(plan, "core.bout_kinematics.heading", "0").status == (
        PlanStatus.BLOCKED_MISSING_SOURCE
    )


def test_historical_core_artifact_is_reported_as_contract_mismatch() -> None:
    root = _root()
    track = _track_run(root, (0,))
    _artifact(
        track,
        "visualizations/track_kinematics_summary_track_0_png",
        contract=None,
        renderer="palette-track-kinematics-summary-v1",
        renderer_version=None,
    )

    plan = plan_recording_report(_recording(), root_opener=lambda _path: root)
    item = _item(plan, "core.track_kinematics.overview", "0")

    assert item.status == PlanStatus.CONTRACT_MISMATCH
    assert "expected 'palette.core.track_kinematics.summary.v1'" in item.reason
    assert item.proposed_actions == ("render:core.track_kinematics.overview",)


def test_mixed_protocol_activates_only_registered_matching_stimulus_pack() -> None:
    root = _root()
    _track_run(root, (0,))
    _stimulus_run(
        root,
        ("SOLID_BLACK", "MOVING_GRATING", "LOOMING_DOT", "DARK_FLASH"),
    )
    response = _run(root, "analysis/stimulus_response_runs", "response")
    _artifact(
        response,
        "visualizations/stimulus_response_omr_summary_png",
        contract="palette.stimulus.moving_grating.omr_summary.v1",
        renderer="palette-stimulus-response-omr-summary-v1",
        renderer_version="1",
    )

    plan = plan_recording_report(_recording(), root_opener=lambda _path: root)
    applicable = {provider.provider_id for provider in plan.providers if provider.applicable}

    assert plan.stimulus_modes == (
        "DARK_FLASH",
        "LOOMING_DOT",
        "MOVING_GRATING",
        "SOLID_BLACK",
    )
    assert applicable == {
        "core_behavior.v1",
        "stimulus.moving_grating.v1",
        "stimulus.looming.v1",
        "stimulus.flash.v1",
    }
    assert not any(item.provider_id == "stimulus.chaser.v1" for item in plan.items)
    assert _item(plan, "stimulus.moving_grating.omr_summary").status == PlanStatus.READY


def test_requested_nonapplicable_provider_emits_not_applicable_items() -> None:
    root = _root()
    _track_run(root, (0,))
    _stimulus_run(root, ("MOVING_GRATING",))

    plan = plan_recording_report(
        _recording(),
        requested_provider_ids=("stimulus.chaser.v1",),
        root_opener=lambda _path: root,
    )

    assert plan.items
    assert {item.status for item in plan.items} == {PlanStatus.NOT_APPLICABLE.value}


def test_chaser_provider_discovers_nested_contracted_bearing_artifact() -> None:
    root = _root()
    _track_run(root, (0,))
    stimulus = _stimulus_run(root, ("CHASER",))
    _configure_chasers(stimulus, 3)
    distance = _run(root, "analysis/chaser_distance_runs", "distance")
    _artifact(
        distance,
        "egocentric_bearing/component/visualizations/egocentric_bearing_pre_post_polar_png",
        contract="palette.chaser_egocentric_bearing.pre_post_polar_density.v2",
        renderer="fisheye.analysis.chaser_egocentric_bearing",
        renderer_version="2",
    )

    plan = plan_recording_report(_recording(), root_opener=lambda _path: root)
    bearing = _item(plan, "stimulus.chaser.egocentric_bearing")

    assert bearing.status == PlanStatus.READY
    assert bearing.artifact is not None
    assert "/egocentric_bearing/component/" in bearing.artifact.path
    assert [(chaser.chaser_index, chaser.behavior_class) for chaser in plan.chasers] == [
        (0, "aggressive"),
        (1, "random_non_chasing"),
        (2, "inert"),
    ]


def test_missing_eye_geometry_is_blocked_as_missing_source() -> None:
    root = _root()
    _track_run(root, (0,))

    plan = plan_recording_report(_recording(), root_opener=lambda _path: root)
    eye = _item(plan, "core.eye_angles.overview")

    assert eye.status == PlanStatus.BLOCKED_MISSING_SOURCE
    assert "refined_subject_masks" in eye.reason
    assert "resolve_source:refined_subject_masks" in eye.proposed_actions


def _write_registry(path: Path, zarr_path: str) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE dataset_context_current (
                dataset_id TEXT,
                recording_id TEXT,
                zarr_path TEXT,
                protocol_name TEXT,
                protocol_hash TEXT,
                arena_id TEXT,
                recording_started_utc TEXT,
                zarr_use TEXT,
                dataset_status TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO dataset_context_current VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "dataset_rec",
                "rec",
                zarr_path,
                "Example",
                "hash",
                "arena_0",
                "2026-01-01T00:00:00Z",
                "analysis",
                "active",
            ),
        )


def test_registry_plan_is_serializable_and_counts_statuses(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _write_registry(registry_path, "/tmp/rec_analysis.zarr")
    root = _root()
    _track_run(root, (0,))

    plan = build_report_plan(
        registry_path=registry_path,
        protocol_name="example",
        root_opener=lambda _path: root,
    )
    payload = report_plan_to_dict(plan)

    assert payload["schema_id"] == "palette.dataset_report_plan.v1"
    assert payload["recordings"][0]["recording"]["recording_id"] == "rec"
    assert sum(plan.status_counts.values()) == len(plan.recordings[0].items)


def test_registry_query_requires_scope_and_is_read_only(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _write_registry(registry_path, "/tmp/rec_analysis.zarr")

    with pytest.raises(ValueError, match="cohort selector"):
        query_report_recordings(registry_path)

    rows = query_report_recordings(registry_path, protocol_name="example")
    assert [row.recording_id for row in rows] == ["rec"]
    assert rows[0].protocol_hash == "hash"
    with sqlite3.connect(registry_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM dataset_context_current").fetchone()[0] == 1


def test_catalog_has_core_and_initial_stimulus_providers() -> None:
    assert set(PROVIDERS) == {
        "core_behavior.v1",
        "stimulus.chaser.v1",
        "stimulus.moving_grating.v1",
        "stimulus.concentric_grating.v1",
        "stimulus.looming.v1",
        "stimulus.flash.v1",
    }


def test_execution_requires_explicit_mode() -> None:
    root = _root()
    _track_run(root, (0,))
    recording_plan = plan_recording_report(_recording(), root_opener=lambda _path: root)
    report = reporting_execution.ReportPlan(
        schema_id="palette.dataset_report_plan.v1",
        schema_version=1,
        created_at_utc="2026-01-01T00:00:00Z",
        tool="test",
        registry_path="/tmp/registry.sqlite",
        query={},
        requested_provider_ids=("core_behavior.v1",),
        recordings=(recording_plan,),
    )

    with pytest.raises(ValueError, match="Enable --render-missing"):
        reporting_execution.execute_report_plan(
            report,
            render_missing=False,
            apply_analysis=False,
        )


def test_execution_deduplicates_shared_track_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root()
    _track_run(root, (0,))
    recording_plan = plan_recording_report(_recording(), root_opener=lambda _path: root)
    report = reporting_execution.ReportPlan(
        schema_id="palette.dataset_report_plan.v1",
        schema_version=1,
        created_at_utc="2026-01-01T00:00:00Z",
        tool="test",
        registry_path="/tmp/registry.sqlite",
        query={},
        requested_provider_ids=("core_behavior.v1",),
        recordings=(recording_plan,),
    )
    calls: list[str] = []

    def fake_executor(context) -> str:
        calls.append(context.item.visualization_id)
        return "rendered"

    monkeypatch.setitem(
        reporting_execution.RENDER_EXECUTORS,
        "core.track_kinematics.overview",
        ("shared_track", fake_executor),
    )
    monkeypatch.setitem(
        reporting_execution.RENDER_EXECUTORS,
        "core.position.xy_trace",
        ("shared_track", fake_executor),
    )

    results = reporting_execution.execute_report_plan(
        report,
        render_missing=True,
        apply_analysis=False,
        visualization_ids=(
            "core.track_kinematics.overview",
            "core.position.xy_trace",
        ),
    )

    assert calls == ["core.track_kinematics.overview"]
    assert [result.status for result in results] == ["executed", "deduplicated"]


def test_semantic_montage_uses_ready_contracted_artifact(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording.zarr"
    root = zarr.open_group(zarr_path, mode="w")
    track = _track_run(root, (0,))
    image = Image.new("RGB", (40, 20), (20, 80, 140))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    artifact = track.require_group("visualizations").create_array(
        "track_kinematics_summary_track_0_png",
        data=np.frombuffer(buffer.getvalue(), dtype=np.uint8),
    )
    artifact.attrs.update(
        {
            "artifact_type": "visualization",
            "artifact_role": "snapshot",
            "visualization_contract_id": "palette.core.track_kinematics.summary.v1",
            "renderer": "palette-track-kinematics-summary-v1",
            "renderer_version": "1",
            "content_sha256": "fixture",
        }
    )
    selected = SelectedRecording(
        dataset_id="dataset_rec",
        recording_id="rec",
        zarr_path=str(zarr_path),
        protocol_name=None,
    )
    recording_plan = plan_recording_report(selected, root_opener=lambda _path: root)
    report = ReportPlan(
        schema_id="palette.dataset_report_plan.v1",
        schema_version=1,
        created_at_utc="2026-01-01T00:00:00Z",
        tool="test",
        registry_path="/tmp/registry.sqlite",
        query={"protocol_name": "fixture"},
        requested_provider_ids=("core_behavior.v1",),
        recordings=(recording_plan,),
    )

    result = build_semantic_visualization_montages(
        plan=report,
        output_dir=tmp_path / "montages",
        visualization_ids=("core.track_kinematics.overview",),
        columns=1,
        tile_width=200,
        max_image_height=120,
    )

    output = Path(result["outputs"][0]["path"])
    assert output.is_file()
    assert Image.open(output).format == "PNG"
    assert result["outputs"][0]["tiles"][0]["artifact_path"].endswith(
        "track_kinematics_summary_track_0_png"
    )
    assert result["nonready_count"] == 0


@pytest.mark.parametrize("materialization_policy", ["reference", "copy"])
def test_report_export_is_immutable_and_content_addressed(
    tmp_path: Path,
    materialization_policy: str,
) -> None:
    zarr_path = tmp_path / "export_source.zarr"
    root = zarr.open_group(zarr_path, mode="w")
    track = _track_run(root, (0,))
    image = Image.new("RGB", (30, 18), (140, 60, 30))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()
    content_sha256 = hashlib.sha256(png_bytes).hexdigest()
    artifact = track.require_group("visualizations").create_array(
        "track_kinematics_summary_track_0_png",
        data=np.frombuffer(png_bytes, dtype=np.uint8),
    )
    artifact.attrs.update(
        {
            "artifact_type": "visualization",
            "artifact_role": "snapshot",
            "visualization_contract_id": "palette.core.track_kinematics.summary.v1",
            "renderer": "palette-track-kinematics-summary-v1",
            "renderer_version": "1",
            "content_sha256": content_sha256,
        }
    )
    selected = SelectedRecording(
        dataset_id="dataset_export",
        recording_id="export",
        zarr_path=str(zarr_path),
        protocol_name=None,
    )
    recording_plan = plan_recording_report(selected, root_opener=lambda _path: root)
    report = ReportPlan(
        schema_id="palette.dataset_report_plan.v1",
        schema_version=1,
        created_at_utc="2026-01-01T00:00:00Z",
        tool="test",
        registry_path="/tmp/registry.sqlite",
        query={"recording_id": "export"},
        requested_provider_ids=("core_behavior.v1",),
        recordings=(recording_plan,),
    )
    collection_manifest = tmp_path / "collection.json"
    collection_manifest.write_text('{"collection_id":"fixture"}\n', encoding="utf-8")
    output_dir = tmp_path / f"report_{materialization_policy}"

    result = export_report_bundle(
        plan=report,
        output_dir=output_dir,
        materialization_policy=materialization_policy,
        visualization_ids=("core.track_kinematics.overview",),
        source_collection_manifest=collection_manifest,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert verify_report_manifest_sha256(manifest)
    assert manifest["artifact_count"] == 1
    assert manifest["source_collection_manifest"]["content_sha256"] == hashlib.sha256(
        collection_manifest.read_bytes()
    ).hexdigest()
    materialized = manifest["artifacts"][0]["materialized"]
    if materialization_policy == "copy":
        assert (output_dir / materialized["relative_path"]).read_bytes() == png_bytes
    else:
        assert materialized["zarr_path"] == str(zarr_path)
    with pytest.raises(FileExistsError, match="Immutable report output"):
        export_report_bundle(
            plan=report,
            output_dir=output_dir,
            materialization_policy=materialization_policy,
            visualization_ids=("core.track_kinematics.overview",),
        )


def test_bound_report_manifest_has_canonical_layout_and_compact_registry_index(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    analytics_root = tmp_path / "analytics"
    export_manifest_path = analytics_root / "v1" / "manifests" / "export.json"
    export_manifest_path.parent.mkdir(parents=True)
    export_manifest_path.write_text(
        json.dumps({"export_run_id": "export_001"}) + "\n",
        encoding="utf-8",
    )
    registry = Registry(registry_path)
    try:
        registry.upsert_analytics_export(
            export_run_id="export_001",
            export_manifest_path=export_manifest_path,
            output_root=analytics_root,
            row_counts_by_table={"recording_summary": 2},
            part_files_by_table={
                "recording_summary": [
                    str(
                        analytics_root
                        / "v1"
                        / "recording_summary"
                        / "export_run_id=export_001"
                        / "part-000.parquet"
                    )
                ]
            },
        )
    finally:
        registry.close()

    binding = resolve_analytics_export_binding(registry_path, "export_001")
    assert binding.available_tables == ("recording_summary",)
    output_dir = report_output_dir(binding, "protocol-overview")
    assert output_dir == (
        analytics_root
        / "v1"
        / "reports"
        / "export_run_id=export_001"
        / "report_id=protocol-overview"
    )
    output_dir.mkdir(parents=True)
    manifest: dict[str, object] = {
        "schema_id": "palette.dataset_report_export.v1",
        "schema_version": 1,
        "created_at_utc": "2026-07-11T12:00:00Z",
        "report_id": "protocol-overview",
        "analytics_export": binding.to_dict(),
        "materialization_policy": "reference",
        "source_backends": ["zarr"],
        "source_tables": [],
        "visualization_ids": ["core.track_kinematics.overview"],
        "artifact_count": 2,
        "nonready_count": 0,
        "nonready": [],
        "artifacts": [
            {
                "visualization_id": "core.track_kinematics.overview",
                "provider_id": "core_behavior.v1",
                "source_backend": "zarr",
                "visualization_contract_id": "palette.core.track_kinematics.summary.v1",
                "renderer": "palette-track-kinematics-summary-v1",
                "renderer_version": "1",
                "materialized": {"zarr_path": "/tmp/a.zarr", "artifact_path": "a"},
            },
            {
                "visualization_id": "core.track_kinematics.overview",
                "provider_id": "core_behavior.v1",
                "source_backend": "zarr",
                "visualization_contract_id": "palette.core.track_kinematics.summary.v1",
                "renderer": "palette-track-kinematics-summary-v1",
                "renderer_version": "1",
                "materialized": {"zarr_path": "/tmp/b.zarr", "artifact_path": "b"},
            },
        ],
        "source_report_plan_sha256": "plan-hash",
        "manifest_relative_path": "report_manifest.json",
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    manifest_path = output_dir / "report_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    check = check_report_manifest(manifest_path)
    assert check["ok"] is True
    assert check["export_run_id"] == "export_001"

    registry = Registry(registry_path)
    try:
        assert index_report_manifest(registry, manifest_path) == (
            "export_001",
            "protocol-overview",
        )
        summary = registry.conn.execute(
            "SELECT * FROM analytics_report_visualizations"
        ).fetchone()
        assert summary["artifact_count"] == 2
        assert summary["visualization_id"] == "core.track_kinematics.overview"
    finally:
        registry.close()

    rows = query_indexed_reports(
        registry_path,
        export_run_id="export_001",
        visualization_id="core.track_kinematics.overview",
    )
    assert len(rows) == 1
    assert rows[0]["report_id"] == "protocol-overview"
    assert rows[0]["visualization_count"] == 1
