from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_NAME_ATTR,
    RUN_STATUS_COMPLETE,
)
from fisheye.utils import export_provider_epoch_behavior_cohort as exporter


class _Group:
    def __init__(self, *, attrs=None, children=None, tables=None, table_attrs=None):
        self.attrs = dict(attrs or {})
        self.children = dict(children or {})
        self.tables = dict(tables or {})
        self.table_attrs = dict(table_attrs or {})

    def get(self, name):
        return self.children.get(name)


def _columnar_attrs(spec):
    return {
        "storage_layout": "columnar",
        "field_names": [name for name, _ in spec],
        "field_dtypes": {
            name: str(np.dtype(dtype))
            for name, dtype in spec
        },
    }


def _source_fixture(tmp_path: Path, *, semantic: bool = False):
    zarr_path = tmp_path / "recording-1_analysis.zarr"
    zarr_path.mkdir()
    recording_id = "recording-1"
    summary_run = "summary_semantic_v2" if semantic else "summary_linear_only_v1"
    refs = {
        "epoch_selection": {
            "record": {
                "schema_id": "palette.resolved_epoch_selection.v1",
                "source_timeline": {"recording_id": recording_id},
            },
            "sha256": "a" * 64,
        },
        "provider_motion": {
            "run_path": "analysis/track_kinematics_runs/provider/motion_v1",
            "manifest_sha256": "b" * 64,
            "verification_digest": "c" * 64,
            "track_id": 0,
        },
        "swim_bouts": {
            "run_path": "analysis/swim_bout_runs/bouts_v1",
            "lineage_hash": "d" * 64,
            "frame_axis_sha256": "e" * 64,
            "track_id": 0,
        },
    }
    semantic_run = "semantic_selection_v2"
    semantic_manifest_sha256 = "f" * 64
    protocol_semantic_hash = f"sha256:{'9' * 64}"
    role_records = [
        {
            "analysis_role": role,
            "source_window_id": window_id,
            "source_interval_sha256": str(index + 1) * 64,
            "selected_start_frame": index * 10,
            "selected_end_frame_exclusive": (index + 1) * 10,
            "protocol_semantic_hash": protocol_semantic_hash,
            "protocol_semantic_step_index": 1,
            "protocol_semantic_step_ref": (
                "protocol_semantic_snapshot@recipe.steps[1]"
            ),
        }
        for index, (role, window_id) in enumerate(
            zip(exporter.SEMANTIC_ROLES, (4, 8, 12))
        )
    ]
    position_epochs = [
        {
            "analysis_role": record["analysis_role"],
            "window_id": record["source_window_id"],
            "source_label": f"source_{record['analysis_role']}",
            "start_frame": record["selected_start_frame"],
            "end_frame_exclusive": record["selected_end_frame_exclusive"],
            "source_interval_sha256": record["source_interval_sha256"],
        }
        for record in role_records
    ]
    if semantic:
        refs.update(
            {
                "epoch_binding_mode": exporter.SEMANTIC_EPOCH_BINDING_MODE,
                "protocol_semantic_selection": {
                    "run_name": semantic_run,
                    "run_path": (
                        "analysis/protocol_semantic_chaser_selection_runs/"
                        f"{semantic_run}"
                    ),
                    "manifest_sha256": semantic_manifest_sha256,
                    "protocol_semantic_hash": protocol_semantic_hash,
                    "source_epoch_selection": refs["epoch_selection"]["record"],
                    "roles": list(exporter.SEMANTIC_ROLES),
                    "position_suite_epochs": position_epochs,
                    "position_suite_epochs_sha256": canonical_json_sha256(
                        position_epochs
                    ),
                    "semantic_role_bindings": role_records,
                    "semantic_role_bindings_sha256": canonical_json_sha256(
                        role_records
                    ),
                    "selector_eligible": False,
                    "production_authority": False,
                },
            }
        )
    attrs = {
        "schema_id": "palette.stimulus_epoch_behavior_summary",
        "schema_version": 2 if semantic else 1,
        RUN_COMPLETION_CONTRACT_ATTR: RUN_COMPLETION_CONTRACT,
        RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE,
        RUN_NAME_ATTR: summary_run,
        "stage_selector_eligible": False,
        "recording_id": recording_id,
        "track_id": 0,
        "source_refs": refs,
        "source_refs_sha256": canonical_json_sha256(refs),
        "analysis_offer": {
            "readiness": {"scientific": "ready"},
            "selector_eligible": False,
        },
        "analysis_offer_sha256": canonical_json_sha256(
            {
                "readiness": {"scientific": "ready"},
                "selector_eligible": False,
            }
        ),
        "run_provenance": {"command": "test"},
    }
    if semantic:
        attrs.update(
            {
                "method_version": 2,
                "epoch_binding_mode": exporter.SEMANTIC_EPOCH_BINDING_MODE,
            }
        )
    fish_spec = exporter._source_fields(
        exporter._FISH_SOURCE_FIELDS,
        schema_version=2 if semantic else 1,
    )
    bout_spec = exporter._source_fields(
        exporter._BOUT_SOURCE_FIELDS,
        schema_version=2 if semantic else 1,
    )
    fish = np.zeros(3 if semantic else 1, dtype=exporter._source_dtype(fish_spec))
    fish["track_id"] = 0
    fish["window_id"] = [4, 8, 12] if semantic else 4
    fish["window_index"] = [0, 1, 2] if semantic else 1
    fish["window_label"] = (
        [role.encode("utf-8") for role in exporter.SEMANTIC_ROLES]
        if semantic
        else b"chaser"
    )
    fish["start_frame"] = [0, 10, 20] if semantic else 10
    fish["end_frame"] = [9, 19, 29] if semantic else 19
    fish["duration_s"] = 1.0
    fish["valid_tracked_duration_s"] = 0.9
    fish["mean_speed_mm_s"] = 2.5
    fish["bout_count"] = 1
    fish["bout_rate_per_min"] = 60.0
    fish["mean_bout_net_heading_change_deg"] = 90.0
    if semantic:
        for index, record in enumerate(role_records):
            fish["analysis_role"][index] = str(record["analysis_role"]).encode()
            fish["protocol_semantic_hash"][index] = protocol_semantic_hash.encode()
            fish["protocol_semantic_step_index"][index] = 1
            fish["protocol_semantic_step_ref"][index] = (
                b"protocol_semantic_snapshot@recipe.steps[1]"
            )
    bouts = np.zeros(1, dtype=exporter._source_dtype(bout_spec))
    bouts["track_id"] = 0
    bouts["window_id"] = 4
    bouts["window_index"] = 0 if semantic else 1
    bouts["window_label"] = b"chaser_pre" if semantic else b"chaser"
    bouts["start_frame"] = 0 if semantic else 10
    bouts["end_frame"] = 9 if semantic else 19
    bouts["bout_source_row"] = 7
    bouts["bout_id"] = 42
    bouts["bout_event_frame"] = 14
    bouts["bout_duration_s"] = 0.1
    bouts["bout_path_length_mm"] = 0.25
    bouts["bout_net_heading_change_deg"] = 90.0
    if semantic:
        bouts["analysis_role"] = b"chaser_pre"
        bouts["protocol_semantic_hash"] = protocol_semantic_hash.encode()
        bouts["protocol_semantic_step_index"] = 1
        bouts["protocol_semantic_step_ref"] = (
            b"protocol_semantic_snapshot@recipe.steps[1]"
        )
    run = _Group(
        attrs=attrs,
        tables={"per_epoch_fish": fish, "per_epoch_bouts": bouts},
        table_attrs={
            "per_epoch_fish": {
                **_columnar_attrs(fish_spec),
                **(
                    {"epoch_binding_mode": exporter.SEMANTIC_EPOCH_BINDING_MODE}
                    if semantic
                    else {}
                ),
            },
            "per_epoch_bouts": {
                **_columnar_attrs(bout_spec),
                **(
                    {"epoch_binding_mode": exporter.SEMANTIC_EPOCH_BINDING_MODE}
                    if semantic
                    else {}
                ),
            },
        },
    )
    root = _Group(
        children={
            "analysis": _Group(
                children={
                    "stimulus_epoch_behavior_summary_runs": _Group(
                        children={summary_run: run}
                    )
                }
            )
        }
    )
    return zarr_path, root, recording_id, summary_run


def _write_manifest(
    path: Path,
    *,
    zarr_path: Path,
    recording_id: str = "recording-1",
    summary_run: str = "summary_linear_only_v1",
    metric_disposition: str = "linear_only",
    reason: str = "heading source superseded",
    schema_version: int = exporter.INPUT_SCHEMA_VERSION,
):
    entry = {
        "recording_id": recording_id,
        "analysis_zarr": str(zarr_path),
        "summary_run": summary_run,
        "track_id": 0,
        "subject_id": "fish-001",
    }
    if schema_version == exporter.SEMANTIC_INPUT_SCHEMA_VERSION:
        entry.update(
            {
                "protocol_semantic_selection_run": "semantic_selection_v2",
                "protocol_semantic_selection_manifest_sha256": "f" * 64,
                "protocol_semantic_hash": f"sha256:{'9' * 64}",
            }
        )
    unsigned = {
        "schema_id": exporter.INPUT_SCHEMA_ID,
        "schema_version": schema_version,
        "cohort_id": "goodbatbadbat_talk_canary",
        "metric_disposition": metric_disposition,
        "metric_disposition_reason": reason,
        "entries": [entry],
    }
    if schema_version == exporter.SEMANTIC_INPUT_SCHEMA_VERSION:
        unsigned["epoch_binding_mode"] = exporter.SEMANTIC_EPOCH_BINDING_MODE
    path.write_text(
        json.dumps(
            {**unsigned, "manifest_payload_sha256": canonical_json_sha256(unsigned)},
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _patch_source(monkeypatch, zarr_path, root):
    monkeypatch.setattr(exporter, "open_zarr_root", lambda *args, **kwargs: root)
    monkeypatch.setattr(
        exporter,
        "load_structured_dataset",
        lambda group, name: (group.tables[name], group.table_attrs[name]),
    )


def test_linear_only_export_omits_heading_columns_and_records_disposition(
    tmp_path: Path, monkeypatch
):
    zarr_path, root, recording_id, summary_run = _source_fixture(tmp_path)
    _patch_source(monkeypatch, zarr_path, root)
    manifest = tmp_path / "cohort.json"
    _write_manifest(manifest, zarr_path=zarr_path)

    planned = exporter.export_provider_epoch_behavior_cohort(
        manifest,
        output_root=tmp_path / "exports",
        analysis_run_id="talk-linear-only-v1",
    )
    assert planned["metric_disposition"] == "linear_only"
    assert planned["selector_eligible"] is False
    assert "bout_net_heading_change_deg" in planned["excluded_metrics"]
    linear_fields = {
        item.name
        for item in exporter.table_contracts_for_disposition("linear_only")[
            exporter.TABLE_BOUTS
        ].fields
    }
    assert "bout_net_heading_change_deg" not in linear_fields
    assert "mean_bout_net_heading_change_deg" not in {
        item.name
        for item in exporter.table_contracts_for_disposition("linear_only")[
            exporter.TABLE_FISH
        ].fields
    }

    result = exporter.export_provider_epoch_behavior_cohort(
        manifest,
        output_root=tmp_path / "exports",
        analysis_run_id="talk-linear-only-v1",
        apply=True,
        generation_id="generation-1",
    )
    publication_manifest = Path(result["publication"]["manifest_path"])
    payload = json.loads(publication_manifest.read_text(encoding="utf-8"))
    assert payload["metric_disposition"] == "linear_only"
    assert payload["metric_disposition_reason"] == "heading source superseded"
    assert payload["publication"]["selector_eligible"] is False
    assert "bout_net_heading_change_deg" in payload["excluded_metrics"]

    import pyarrow.parquet as pq

    part = (
        tmp_path
        / "exports"
        / "v2"
        / ".generations"
        / "analysis_run_id=talk-linear-only-v1"
        / "generation=generation-1"
        / "tables"
        / exporter.TABLE_BOUTS
        / "part-00000.parquet"
    )
    assert "bout_net_heading_change_deg" not in pq.ParquetFile(part).schema_arrow.names
    assert "subject_id" in pq.ParquetFile(part).schema_arrow.names


def test_manifest_requires_explicit_metric_disposition(tmp_path: Path):
    unsigned = {
        "schema_id": exporter.INPUT_SCHEMA_ID,
        "schema_version": exporter.INPUT_SCHEMA_VERSION,
        "cohort_id": "cohort",
        "entries": [],
    }
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps({**unsigned, "manifest_payload_sha256": canonical_json_sha256(unsigned)}),
        encoding="utf-8",
    )
    with pytest.raises(exporter.ProviderEpochBehaviorCohortError, match="field set"):
        exporter.load_cohort_manifest(path)


def test_duplicate_manifest_mapping_fails_closed(tmp_path: Path):
    zarr_path = tmp_path / "recording.zarr"
    zarr_path.mkdir()
    unsigned = {
        "schema_id": exporter.INPUT_SCHEMA_ID,
        "schema_version": exporter.INPUT_SCHEMA_VERSION,
        "cohort_id": "cohort",
        "metric_disposition": "linear_only",
        "metric_disposition_reason": "test",
        "entries": [
            {
                "recording_id": "one",
                "analysis_zarr": str(zarr_path),
                "summary_run": "summary-1",
                "track_id": 0,
                "subject_id": None,
            },
            {
                "recording_id": "one",
                "analysis_zarr": str(zarr_path),
                "summary_run": "summary-1",
                "track_id": 0,
                "subject_id": None,
            },
        ],
    }
    path = tmp_path / "duplicate.json"
    path.write_text(
        json.dumps({**unsigned, "manifest_payload_sha256": canonical_json_sha256(unsigned)}),
        encoding="utf-8",
    )
    with pytest.raises(exporter.ProviderEpochBehaviorCohortError, match="mapping"):
        exporter.load_cohort_manifest(path)


def test_incomplete_summary_is_rejected(tmp_path: Path, monkeypatch):
    zarr_path, root, _recording_id, _summary_run = _source_fixture(tmp_path)
    run = root.children["analysis"].children[
        "stimulus_epoch_behavior_summary_runs"
    ].children["summary_linear_only_v1"]
    run.attrs[RUN_COMPLETION_STATUS_ATTR] = "running"
    _patch_source(monkeypatch, zarr_path, root)
    manifest = tmp_path / "cohort.json"
    _write_manifest(manifest, zarr_path=zarr_path)
    with pytest.raises(exporter.ProviderEpochBehaviorCohortError, match="incomplete"):
        exporter.build_provider_epoch_behavior_cohort_plan(
            manifest,
            output_root=tmp_path / "exports",
            analysis_run_id="talk-linear-only-v1",
        )


def test_source_binding_digest_mismatch_is_rejected(tmp_path: Path, monkeypatch):
    zarr_path, root, _recording_id, _summary_run = _source_fixture(tmp_path)
    run = root.children["analysis"].children[
        "stimulus_epoch_behavior_summary_runs"
    ].children["summary_linear_only_v1"]
    run.attrs["source_refs_sha256"] = "f" * 64
    _patch_source(monkeypatch, zarr_path, root)
    manifest = tmp_path / "cohort.json"
    _write_manifest(manifest, zarr_path=zarr_path)
    with pytest.raises(exporter.ProviderEpochBehaviorCohortError, match="source_refs digest"):
        exporter.build_provider_epoch_behavior_cohort_plan(
            manifest,
            output_root=tmp_path / "exports",
            analysis_run_id="talk-linear-only-v1",
        )


def test_epoch_selection_source_timeline_recording_mismatch_is_rejected(
    tmp_path: Path, monkeypatch
):
    zarr_path, root, _recording_id, _summary_run = _source_fixture(tmp_path)
    run = root.children["analysis"].children[
        "stimulus_epoch_behavior_summary_runs"
    ].children["summary_linear_only_v1"]
    run.attrs["source_refs"]["epoch_selection"]["record"]["source_timeline"][
        "recording_id"
    ] = "different-recording"
    run.attrs["source_refs_sha256"] = canonical_json_sha256(run.attrs["source_refs"])
    _patch_source(monkeypatch, zarr_path, root)
    manifest = tmp_path / "cohort.json"
    _write_manifest(manifest, zarr_path=zarr_path)
    with pytest.raises(
        exporter.ProviderEpochBehaviorCohortError,
        match="epoch-selection recording identity",
    ):
        exporter.build_provider_epoch_behavior_cohort_plan(
            manifest,
            output_root=tmp_path / "exports",
            analysis_run_id="talk-linear-only-v1",
        )


def test_source_schema_mismatch_is_rejected(tmp_path: Path, monkeypatch):
    zarr_path, root, _recording_id, _summary_run = _source_fixture(tmp_path)
    run = root.children["analysis"].children[
        "stimulus_epoch_behavior_summary_runs"
    ].children["summary_linear_only_v1"]
    original = run.tables["per_epoch_fish"]
    malformed = np.zeros(1, dtype=original.dtype.descr[:-1])
    for name in malformed.dtype.names or ():
        malformed[name] = original[name]
    run.tables["per_epoch_fish"] = malformed
    _patch_source(monkeypatch, zarr_path, root)
    manifest = tmp_path / "cohort.json"
    _write_manifest(manifest, zarr_path=zarr_path)
    with pytest.raises(exporter.ProviderEpochBehaviorCohortError, match="dtype"):
        exporter.build_provider_epoch_behavior_cohort_plan(
            manifest,
            output_root=tmp_path / "exports",
            analysis_run_id="talk-linear-only-v1",
        )


def test_semantic_v2_export_carries_exact_role_identity_per_row(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zarr_path, root, recording_id, summary_run = _source_fixture(
        tmp_path,
        semantic=True,
    )
    _patch_source(monkeypatch, zarr_path, root)
    manifest = tmp_path / "semantic-cohort.json"
    _write_manifest(
        manifest,
        zarr_path=zarr_path,
        recording_id=recording_id,
        summary_run=summary_run,
        schema_version=exporter.SEMANTIC_INPUT_SCHEMA_VERSION,
    )

    plan = exporter.build_provider_epoch_behavior_cohort_plan(
        manifest,
        output_root=tmp_path / "exports",
        analysis_run_id="semantic-talk-v2",
    )

    assert plan["schema_version"] == exporter.SEMANTIC_EXPORT_SCHEMA_VERSION
    assert plan["epoch_binding_mode"] == exporter.SEMANTIC_EPOCH_BINDING_MODE
    fish_rows = plan["rows_by_table"][exporter.TABLE_FISH]
    assert [row["epoch_id"] for row in fish_rows] == [4, 8, 12]
    assert [row["epoch_index"] for row in fish_rows] == [0, 1, 2]
    assert [row["analysis_role"] for row in fish_rows] == list(
        exporter.SEMANTIC_ROLES
    )
    assert all(
        row["protocol_semantic_hash"] == f"sha256:{'9' * 64}"
        for row in fish_rows
    )
    semantic_contract = exporter.table_contracts_for_disposition(
        "linear_only",
        schema_version=exporter.SEMANTIC_EXPORT_SCHEMA_VERSION,
    )[exporter.TABLE_FISH]
    assert "protocol_semantic_step_ref" in {
        field.name for field in semantic_contract.fields
    }

    result = exporter.export_provider_epoch_behavior_cohort(
        manifest,
        output_root=tmp_path / "exports",
        analysis_run_id="semantic-talk-v2",
        apply=True,
        generation_id="semantic-generation-1",
    )
    publication = json.loads(
        Path(result["publication"]["manifest_path"]).read_text(encoding="utf-8")
    )
    assert publication["export_schema_version"] == 2
    assert publication["epoch_binding_mode"] == exporter.SEMANTIC_EPOCH_BINDING_MODE
    assert publication["protocol_to_acquisition_alignment"] == (
        "sealed_epoch_selection_proxy_not_physical_presentation"
    )

    import pyarrow.parquet as pq

    part = (
        tmp_path
        / "exports"
        / "v2"
        / ".generations"
        / "analysis_run_id=semantic-talk-v2"
        / "generation=semantic-generation-1"
        / "tables"
        / exporter.TABLE_FISH
        / "part-00000.parquet"
    )
    table = pq.ParquetFile(part).read()
    assert table["analysis_role"].to_pylist() == list(exporter.SEMANTIC_ROLES)
    assert table.schema.metadata[b"palette.export_schema_version"] == b"2"

    from fisheye.utils.plot_provider_epoch_behavior_cohort import (
        plot_provider_epoch_behavior_cohort_parquet,
    )

    bout_part = part.parents[1] / exporter.TABLE_BOUTS / "part-00000.parquet"
    plotted = plot_provider_epoch_behavior_cohort_parquet(
        bouts_parquet=bout_part,
        fish_parquet=part,
        manifest_path=Path(result["publication"]["manifest_path"]),
        output_dir=tmp_path / "semantic-plots",
        prefix="semantic-export",
    )
    assert plotted["receipt"]["schema_version"] == 4
    assert plotted["receipt"]["expected_epoch_labels"] == list(
        exporter.SEMANTIC_ROLES
    )


def test_semantic_v2_manifest_rejects_legacy_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zarr_path, root, recording_id, summary_run = _source_fixture(tmp_path)
    _patch_source(monkeypatch, zarr_path, root)
    manifest = tmp_path / "semantic-cohort.json"
    _write_manifest(
        manifest,
        zarr_path=zarr_path,
        recording_id=recording_id,
        summary_run=summary_run,
        schema_version=exporter.SEMANTIC_INPUT_SCHEMA_VERSION,
    )

    with pytest.raises(exporter.ProviderEpochBehaviorCohortError, match="schema"):
        exporter.build_provider_epoch_behavior_cohort_plan(
            manifest,
            output_root=tmp_path / "exports",
            analysis_run_id="semantic-talk-v2",
        )


def test_semantic_v2_rejects_row_identity_tampering(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zarr_path, root, recording_id, summary_run = _source_fixture(
        tmp_path,
        semantic=True,
    )
    run = root.children["analysis"].children[
        "stimulus_epoch_behavior_summary_runs"
    ].children[summary_run]
    run.tables["per_epoch_fish"][1]["protocol_semantic_step_index"] = 2
    _patch_source(monkeypatch, zarr_path, root)
    manifest = tmp_path / "semantic-cohort.json"
    _write_manifest(
        manifest,
        zarr_path=zarr_path,
        recording_id=recording_id,
        summary_run=summary_run,
        schema_version=exporter.SEMANTIC_INPUT_SCHEMA_VERSION,
    )

    with pytest.raises(
        exporter.ProviderEpochBehaviorCohortError,
        match="row protocol-semantic identity",
    ):
        exporter.build_provider_epoch_behavior_cohort_plan(
            manifest,
            output_root=tmp_path / "exports",
            analysis_run_id="semantic-talk-v2",
        )


def test_semantic_v2_rejects_manifest_identity_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zarr_path, root, recording_id, summary_run = _source_fixture(
        tmp_path,
        semantic=True,
    )
    _patch_source(monkeypatch, zarr_path, root)
    manifest = tmp_path / "semantic-cohort.json"
    _write_manifest(
        manifest,
        zarr_path=zarr_path,
        recording_id=recording_id,
        summary_run=summary_run,
        schema_version=exporter.SEMANTIC_INPUT_SCHEMA_VERSION,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["entries"][0]["protocol_semantic_selection_manifest_sha256"] = (
        "0" * 64
    )
    unsigned = {
        key: value for key, value in payload.items() if key != "manifest_payload_sha256"
    }
    payload["manifest_payload_sha256"] = canonical_json_sha256(unsigned)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        exporter.ProviderEpochBehaviorCohortError,
        match="frozen cohort entry",
    ):
        exporter.build_provider_epoch_behavior_cohort_plan(
            manifest,
            output_root=tmp_path / "exports",
            analysis_run_id="semantic-talk-v2",
        )
