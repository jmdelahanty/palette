from __future__ import annotations

import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fisheye.analytics_exports.arrow_contracts import (
    ARROW_TABLE_CONTRACTS,
    arrow_contract_envelope,
    exact_arrow_schema,
)
from fisheye.analytics_exports.arrow_contract_core import payload_sha256
from apps.marimo.components.training_response import (
    filter_training_response_rows,
    strategy_transition_sankey_figure,
    training_response_scatter_figure,
)
from fisheye.analytics_exports.capabilities import resolve_capabilities
from fisheye.analytics_exports.contracts import (
    CHASER_DISTANCE_SUMMARY_TABLE,
    CHASER_EGOCENTRIC_SUMMARY_TABLE,
    CHASER_EPOCH_BEHAVIOR_TABLE,
    CHASER_SPEED_DISTANCE_TABLE,
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
    TABLE_CONTRACTS,
    contract_snapshot,
)
from fisheye.analytics_exports.derived_publication import (
    publish_derived_table_generation,
)
from fisheye.analytics_exports.publication import sha256_file
from fisheye.analytics_exports.registry_identity import (
    build_registry_identity_receipt,
    build_registry_identity_source,
)
from fisheye.training_response.cohort import (
    classify_training_response_features,
    discover_training_response_clusters,
)
from fisheye.training_response.contracts import (
    ARROW_CONTRACT_ENVELOPE_SCHEMA_ID as RESPONSE_ARROW_ENVELOPE_SCHEMA_ID,
    LEGACY_ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION,
    LEGACY_EXACT_SCHEMA_VERSION,
    LEGACY_V2_ARROW_TABLE_CONTRACTS,
    SCHEMA_ID as RESPONSE_SCHEMA_ID,
    TRAINING_RESPONSE_TABLES,
    TrainingResponseConfig,
    normalize_training_response_rows,
    training_response_arrow_contract_envelope,
)
from fisheye.training_response.features import derive_training_response_features
from fisheye.training_response.query import (
    discover_training_response_catalog,
    scan_training_response_qc_rows,
    scan_training_response_table,
    select_training_response_run_id,
    load_training_response_manifest,
)
from fisheye.training_response.validation import validate_training_response_run
from fisheye.training_response.workflow import (
    build_training_response_tables,
    run_training_response_analytics,
)


def _identity(recording_id: str) -> dict[str, str]:
    return {
        "recording_id": recording_id,
        "acquisition_batch_id": f"session_{int(recording_id.rsplit('_', 1)[-1]) // 2}",
        "subject_id": f"subject_{recording_id}",
    }


def _behavior_rows(recording_id: str, *, scale: float = 1.0, dropout: float = 0.0):
    common = {
        **_identity(recording_id),
        "zarr_path": f"/{recording_id}_analysis.zarr",
        "duration_s": 60.0,
        "tracking_dropout_fraction": dropout,
        "arena_radius_mm": 40.0,
        "mean_bout_path_length_mm": 2.0,
        "mean_abs_bout_net_heading_change_deg": 12.0,
    }
    return [
        {
            **common,
            "window_id": 0,
            "window_label": "pre_event",
            "mean_speed_mm_s": 2.0,
            "p95_speed_mm_s": 8.0,
            "total_path_mm": 120.0,
            "bout_count": 20,
            "bout_rate_per_min": 20.0,
            "wall_fraction": 0.60,
            "median_distance_from_arena_center_mm": 30.0,
        },
        {
            **common,
            "window_id": 1,
            "window_label": "training_event",
            "mean_speed_mm_s": 2.0 * scale,
            "p95_speed_mm_s": 8.0 * scale,
            "total_path_mm": 120.0 * scale,
            "bout_count": 20,
            "bout_rate_per_min": 20.0 * scale,
            "wall_fraction": 0.60 + 0.05 * math.log2(scale),
            "median_distance_from_arena_center_mm": 30.0 + math.log2(scale),
        },
    ]


def _distance_rows(recording_id: str, *, aggressive_training_p50: float = 30.0):
    rows = []
    values = {
        ("pre_event", "aggressive"): (0, 20.0, 8.0, 0.20),
        ("training_event", "aggressive"): (
            0,
            aggressive_training_p50,
            aggressive_training_p50 - 12.0,
            max(0.01, 0.35 - aggressive_training_p50 / 200.0),
        ),
        ("pre_event", "inert"): (1, 45.0, 25.0, 0.08),
        ("training_event", "inert"): (1, 50.0, 30.0, 0.10),
    }
    for window_id, window in enumerate(("pre_event", "training_event")):
        for role in ("aggressive", "inert"):
            chaser_index, p50, p05, within = values[(window, role)]
            rows.append(
                {
                    **_identity(recording_id),
                    "window_id": window_id,
                    "window_label": window,
                    "chaser_index": chaser_index,
                    "behavior_class": role,
                    "p05_distance_mm": p05,
                    "p50_distance_mm": p50,
                    "fraction_within_threshold": within,
                    "threshold_mm": 20.0,
                    "mean_distance_mm": p50 + 2.0,
                }
            )
    return rows


def _egocentric_rows(recording_id: str):
    return [
        {
            **_identity(recording_id),
            "window_id": window_id,
            "window_label": window,
            "chaser_index": chaser_index,
            "behavior_class": role,
            "mean_alignment_cos": value,
            "circular_mean_bearing_deg": 0.0,
            "fraction_front_45": 0.25 + value / 10.0,
            "fraction_behind_45": 0.25 - value / 10.0,
            "circular_resultant_length": 0.3,
        }
        for window_id, (window, value) in enumerate(
            (("pre_event", 0.1), ("training_event", 0.2))
        )
        for chaser_index, role in enumerate(("aggressive", "inert"))
    ]


def _speed_distance_rows(recording_id: str):
    return [
        {
            **_identity(recording_id),
            "window_id": 1,
            "window_label": "training_event",
            "chaser_index": chaser_index,
            "distance_bin_index": bin_index,
            "distance_bin_center_mm": center,
            "mean_speed_mm_s": mean_speed,
            "speed_sample_count": 2,
        }
        for chaser_index in (0, 1)
        for bin_index, (center, mean_speed) in enumerate(((5.0, 10.0), (25.0, 4.0)))
    ]


def test_feature_builder_separates_pre_change_role_and_proximity_metrics() -> None:
    row = derive_training_response_features(
        recording_id="recording_001",
        acquisition_batch_id="session_0",
        subject_id="subject_recording_001",
        source_export_run_id="source_001",
        behavior_rows=_behavior_rows("recording_001", scale=2.0),
        distance_rows=_distance_rows("recording_001"),
        egocentric_rows=_egocentric_rows("recording_001"),
        speed_distance_rows=_speed_distance_rows("recording_001"),
        protocol_name="RedScare",
    )

    assert row["feature_status"] == "complete"
    assert math.isclose(row["mean_speed_mm_s_log2_ratio"], 1.0, abs_tol=1e-6)
    assert math.isclose(row["wall_fraction_delta"], 0.05, abs_tol=1e-12)
    assert row["aggressive_p50_distance_mm_delta"] == 10.0
    assert row["training_role_p50_distance_contrast_mm"] == -20.0
    assert row["aggressive_near_minus_far_speed_mm_s"] == 6.0
    assert row["temporal_training_features_available"] is False
    assert row["temporal_training_feature_reason"] == (
        "training_time_bins_and_samples_not_exported"
    )


def test_feature_builder_rejects_low_tracking_coverage() -> None:
    row = derive_training_response_features(
        recording_id="recording_001",
        acquisition_batch_id="session_0",
        subject_id="subject_recording_001",
        source_export_run_id="source_001",
        behavior_rows=_behavior_rows("recording_001", dropout=0.30),
        distance_rows=_distance_rows("recording_001"),
    )

    assert row["feature_status"] == "invalid"
    assert "pre_tracking_coverage_below_threshold" in row["feature_reason"]
    assert "training_tracking_coverage_below_threshold" in row["feature_reason"]
    classified = classify_training_response_features([row])
    assert classified[0]["classification_status"] == "invalid"
    assert classified[0]["aggressive_proximity_score"] is None
    assert classified[0]["aggressive_proximity_metric_count"] == 0


def test_classification_uses_clear_cohort_proximity_vocabulary() -> None:
    features = []
    for index, p50 in enumerate((10.0, 18.0, 26.0, 34.0, 42.0, 50.0)):
        recording_id = f"recording_{index}"
        features.append(
            derive_training_response_features(
                recording_id=recording_id,
                acquisition_batch_id=_identity(recording_id)["acquisition_batch_id"],
                subject_id=_identity(recording_id)["subject_id"],
                source_export_run_id="source_001",
                behavior_rows=_behavior_rows(recording_id, scale=0.5 + index * 0.3),
                distance_rows=_distance_rows(recording_id, aggressive_training_p50=p50),
                speed_distance_rows=_speed_distance_rows(recording_id),
            )
        )
    rows = classify_training_response_features(
        features, config=TrainingResponseConfig(relative_score_threshold=0.6)
    )
    by_id = {row["recording_id"]: row for row in rows}

    assert by_id["recording_0"]["aggressive_proximity_state"] == "closer_than_cohort"
    assert by_id["recording_5"]["aggressive_proximity_state"] == "farther_than_cohort"
    assert all("exposure" not in row["aggressive_proximity_state"] for row in rows)
    assert all(row["causal_avoidance_inference_permitted"] is False for row in rows)
    assert all("profile_separation_score" in row for row in rows)


def test_build_tables_preserves_multi_session_multi_subject_identity() -> None:
    recording_ids = [f"recording_{index}" for index in range(6)]
    tables = build_training_response_tables(
        source_export_run_id="source_001",
        behavior_rows=[row for item in recording_ids for row in _behavior_rows(item)],
        distance_rows=[row for item in recording_ids for row in _distance_rows(item)],
        egocentric_rows=[
            row for item in recording_ids for row in _egocentric_rows(item)
        ],
        speed_distance_rows=[
            row for item in recording_ids for row in _speed_distance_rows(item)
        ],
        recording_protocols={item: "GoodCopBadCop" for item in recording_ids},
        config=TrainingResponseConfig(cluster_stability_resamples=2),
    )

    assert set(tables) == {
        "training_response_features",
        "training_response_classification",
        "training_response_clusters",
    }
    assert {len(rows) for rows in tables.values()} == {6}
    assert {row["protocol_name"] for row in tables["training_response_features"]} == {
        "GoodCopBadCop"
    }
    for rows in tables.values():
        assert {
            (row["recording_id"], row["acquisition_batch_id"], row["subject_id"])
            for row in rows
        } == {
            (
                recording_id,
                _identity(recording_id)["acquisition_batch_id"],
                _identity(recording_id)["subject_id"],
            )
            for recording_id in recording_ids
        }


def test_build_tables_rejects_conflicting_subject_binding_for_recording() -> None:
    behavior = _behavior_rows("recording_0")
    distance = _distance_rows("recording_0")
    distance[0] = {**distance[0], "subject_id": "different_subject"}

    with pytest.raises(ValueError, match="conflicting batch/subject bindings"):
        build_training_response_tables(
            source_export_run_id="source_001",
            behavior_rows=behavior,
            distance_rows=distance,
        )


def test_cluster_discovery_reports_missing_stability_instead_of_validating_it() -> None:
    rows = []
    for index in range(8):
        group = -4.0 if index < 4 else 4.0
        rows.append(
            {
                "recording_id": f"recording_{index}",
                "acquisition_batch_id": f"session_{index // 2}",
                "subject_id": f"subject_recording_{index}",
                "training_window_id": 1,
                "classification_status": "complete",
                "locomotor_response_score": group,
                "boundary_response_score": group,
                "aggressive_proximity_score": group,
                "role_distance_selectivity_score": -group,
                "close_contact_vigor_score": group,
            }
        )
    clusters = discover_training_response_clusters(
        rows,
        config=TrainingResponseConfig(
            cluster_max_components=3,
            cluster_min_rows_per_component=3,
            cluster_stability_threshold=0.6,
            cluster_stability_resamples=0,
        ),
    )

    assert {row["selected_component_count"] for row in clusters} == {2}
    assert {row["cluster_status"] for row in clusters} == {"stability_unavailable"}
    assert all(row["cluster_stability_threshold"] == 0.6 for row in clusters)


def _source_rows(recording_ids: list[str]):
    return {
        CHASER_EPOCH_BEHAVIOR_TABLE: [
            row for item in recording_ids for row in _behavior_rows(item)
        ],
        CHASER_DISTANCE_SUMMARY_TABLE: [
            row for item in recording_ids for row in _distance_rows(item)
        ],
        CHASER_EGOCENTRIC_SUMMARY_TABLE: [
            row for item in recording_ids for row in _egocentric_rows(item)
        ],
        CHASER_SPEED_DISTANCE_TABLE: [
            row for item in recording_ids for row in _speed_distance_rows(item)
        ],
    }


def _write_source_export(root: Path, run_id: str) -> dict[str, Path]:
    rows_by_table = _source_rows([f"recording_{index}" for index in range(6)])
    generation_path = (
        Path("v1") / ".generations" / f"export_run_id={run_id}" / "generation=test"
    )
    parts: dict[str, Path] = {}
    schemas: dict[str, tuple[str, ...]] = {}
    for table_name, raw_rows in rows_by_table.items():
        rows = []
        for raw in raw_rows:
            row: dict[str, object] = {}
            for field in ARROW_TABLE_CONTRACTS[table_name].fields:
                if field.nullable:
                    row[field.name] = None
                elif field.arrow_type in {"int32", "int64"}:
                    row[field.name] = 1
                elif field.arrow_type == "float64":
                    row[field.name] = 1.0
                elif field.arrow_type == "bool":
                    row[field.name] = True
                else:
                    row[field.name] = "fixture"
            row.update(raw)
            recording_id = str(raw["recording_id"])
            row["zarr_path"] = f"/{recording_id}_analysis.zarr"
            row["acquisition_batch_id"] = _identity(recording_id)[
                "acquisition_batch_id"
            ]
            row["subject_id"] = f"subject-{recording_id}"
            row["export_schema_version"] = EXPORT_SCHEMA_VERSION
            row["table_name"] = table_name
            rows.append(row)
        schema = exact_arrow_schema(
            table_name,
            metadata={
                b"palette.export_schema_id": EXPORT_SCHEMA_ID.encode(),
                b"palette.export_schema_version": str(EXPORT_SCHEMA_VERSION).encode(),
                b"palette.table_contract": json.dumps(
                    TABLE_CONTRACTS[table_name].to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
            },
        )
        table = pa.Table.from_pylist(rows, schema=schema)
        part = root / generation_path / "tables" / table_name / "part-00000.parquet"
        part.parent.mkdir(parents=True)
        pq.write_table(table, part)
        parts[table_name] = part
        schemas[table_name] = tuple(table.schema.names)
    capabilities = [
        item.capability_id for item in resolve_capabilities(schemas) if item.available
    ]
    manifest = root / "v1" / "manifests" / f"export_run_id={run_id}.json"
    manifest.parent.mkdir(parents=True)
    relative_parts = {
        name: (generation_path / "tables" / name / path.name).as_posix()
        for name, path in parts.items()
    }
    recording_ids = [f"recording_{index}" for index in range(6)]
    source_zarrs = [f"/{recording_id}_analysis.zarr" for recording_id in recording_ids]
    registry_identity = build_registry_identity_receipt(
        registry_path=Path("/registry/test.sqlite"),
        sources=[
            build_registry_identity_source(
                zarr_path=path,
                rows=[
                    {
                        "dataset_id": index + 1,
                        "recording_id": recording_id,
                        "acquisition_batch_id": _identity(recording_id)[
                            "acquisition_batch_id"
                        ],
                        "acquisition_batch_snapshot_id": (
                            f"00000000-0000-4000-8000-{index // 2 + 1:012d}"
                        ),
                        "acquisition_batch_schema_id": (
                            "palette.registry.acquisition_batch.v1"
                        ),
                        "acquisition_batch_creation_registry_schema_version": 1,
                        "acquisition_batch_identity_status": "explicit",
                        "acquisition_batch_assignment_snapshot_id": (
                            f"10000000-0000-4000-8000-{index + 1:012d}"
                        ),
                        "acquisition_batch_assignment_batch_id": (
                            "20000000-0000-4000-8000-000000000001"
                        ),
                        "acquisition_batch_assignment_revision": 1,
                        "acquisition_batch_supersedes_assignment_snapshot_id": None,
                        "acquisition_batch_assignment_schema_id": (
                            "palette.registry.acquisition_batch_assignment.v1"
                        ),
                        "acquisition_batch_assignment_registry_schema_version": 1,
                        "acquisition_batch_assignment_method": "manual_test",
                        "acquisition_batch_assigned_by": "test",
                        "acquisition_batch_assigned_at_utc": (
                            "2026-08-10T00:00:00+00:00"
                        ),
                        "fish_id": f"subject-{recording_id}",
                        "subject_count": 1,
                        "subject_ids_json": None,
                    }
                ],
            )
            for index, (recording_id, path) in enumerate(
                zip(recording_ids, source_zarrs, strict=True)
            )
        ],
    )
    manifest.write_text(
        json.dumps(
            {
                "export_run_id": run_id,
                "schema_id": EXPORT_SCHEMA_ID,
                "schema_version": EXPORT_SCHEMA_VERSION,
                "source_zarrs": source_zarrs,
                "registry_identity": registry_identity,
                "tables_requested": list(rows_by_table),
                "table_contracts": contract_snapshot(list(rows_by_table)),
                "arrow_schema_contracts": arrow_contract_envelope(list(rows_by_table)),
                "row_counts_by_table": {
                    name: len(rows) for name, rows in rows_by_table.items()
                },
                "part_files_by_table": {name: [relative_parts[name]] for name in parts},
                "capabilities": capabilities,
                "publication": {
                    "schema_id": "palette.analytics_export.publication",
                    "schema_version": 1,
                    "state": "complete",
                    "generation_id": "test",
                    "generation_path": generation_path.as_posix(),
                    "parts_by_table": {
                        name: [
                            {
                                "path": relative_parts[name],
                                "sha256": sha256_file(path),
                                "size_bytes": path.stat().st_size,
                                "row_count": len(rows_by_table[name]),
                            }
                        ]
                        for name, path in parts.items()
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return parts


def test_workflow_publishes_separate_validated_lazy_tables(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "training"
    source_parts = _write_source_export(source_root, "source_001")
    source_hashes = {name: path.read_bytes() for name, path in source_parts.items()}

    result = run_training_response_analytics(
        source_export_root=source_root,
        source_export_run_id="source_001",
        output_root=output_root,
        analysis_run_id="training_001",
        config=TrainingResponseConfig(cluster_stability_resamples=2),
    )

    assert result["source_export_mutated"] is False
    assert result["schema_version"] == 3
    assert result["source_registry_identity_receipt"]["sources"]
    assert result["temporal_adaptation_status"].startswith("unavailable")
    assert result["output_validation"]["status"] == "valid"
    assert (
        validate_training_response_run(output_root, "training_001")["table_count"] == 3
    )
    assert all(
        path.read_bytes() == source_hashes[name] for name, path in source_parts.items()
    )
    lazy = scan_training_response_table(
        output_root,
        "training_001",
        "training_response_classification",
        columns=("recording_id", "aggressive_proximity_state"),
    )
    assert lazy.collect().shape == (6, 2)
    qc = scan_training_response_qc_rows(output_root, "training_001").collect()
    assert qc.height == 6
    assert qc["acquisition_batch_id"].n_unique() == 3
    assert qc["subject_id"].n_unique() == 6
    catalog = discover_training_response_catalog(output_root)
    assert (
        select_training_response_run_id(
            catalog, "latest", source_export_run_id="source_001"
        )
        == "training_001"
    )
    with pytest.raises(ValueError, match="no ready training-response"):
        select_training_response_run_id(
            catalog,
            "latest",
            source_export_run_id="source_001",
            source_export_manifest_sha256="not-the-published-hash",
        )


def test_training_catalog_rejects_digest_valid_but_ineligible_v3_run(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "training"
    _write_source_export(source_root, "source_001")
    run_training_response_analytics(
        source_export_root=source_root,
        source_export_run_id="source_001",
        output_root=output_root,
        analysis_run_id="training_ineligible",
        config=TrainingResponseConfig(cluster_stability_resamples=2),
    )
    manifest_path = (
        output_root / "v2" / "manifests" / "analysis_run_id=training_ineligible.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["publication"]["selector_eligible"] = False
    payload["manifest_payload_sha256"] = payload_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "manifest_payload_sha256"
        }
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    catalog = discover_training_response_catalog(output_root)

    assert catalog.entries == ()
    assert len(catalog.diagnostics) == 1
    assert "not selector eligible" in catalog.diagnostics[0].message


def test_strict_v3_reader_rejects_v2_and_explicit_v2_adapter_accepts_it(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "legacy_training"
    recording_ids = [f"recording_{index}" for index in range(6)]
    current = build_training_response_tables(
        source_export_run_id="source_001",
        behavior_rows=[row for item in recording_ids for row in _behavior_rows(item)],
        distance_rows=[row for item in recording_ids for row in _distance_rows(item)],
        egocentric_rows=[
            row for item in recording_ids for row in _egocentric_rows(item)
        ],
        speed_distance_rows=[
            row for item in recording_ids for row in _speed_distance_rows(item)
        ],
        config=TrainingResponseConfig(cluster_stability_resamples=0),
    )
    normalized = {
        table_name: normalize_training_response_rows(
            table_name,
            rows,
            analysis_run_id="legacy_v2",
        )
        for table_name, rows in current.items()
    }
    legacy_rows: dict[str, list[dict[str, object]]] = {}
    for table_name, rows in normalized.items():
        names = tuple(
            field.name for field in LEGACY_V2_ARROW_TABLE_CONTRACTS[table_name].fields
        )
        legacy_rows[table_name] = [
            {
                **{name: row[name] for name in names},
                "schema_version": LEGACY_EXACT_SCHEMA_VERSION,
                "method_version": "1",
            }
            for row in rows
        ]
    config = TrainingResponseConfig(cluster_stability_resamples=0)
    publish_derived_table_generation(
        output_root=output_root,
        analysis_run_id="legacy_v2",
        rows_by_table=legacy_rows,
        table_names=TRAINING_RESPONSE_TABLES,
        contracts=LEGACY_V2_ARROW_TABLE_CONTRACTS,
        arrow_contract_envelope=training_response_arrow_contract_envelope(
            schema_version=LEGACY_EXACT_SCHEMA_VERSION
        ),
        arrow_envelope_schema_id=RESPONSE_ARROW_ENVELOPE_SCHEMA_ID,
        arrow_envelope_schema_version=(LEGACY_ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION),
        manifest_fields={
            "schema_id": RESPONSE_SCHEMA_ID,
            "schema_version": LEGACY_EXACT_SCHEMA_VERSION,
            "created_at_utc": "2026-08-10T00:00:00+00:00",
            "source_export_root": "/immutable/source",
            "source_export_run_id": "source_001",
            "source_export_manifest_sha256": "a" * 64,
            "source_collection_manifest_sha256": None,
            "source_validation": {"status": "valid"},
            "feature_config": config.to_dict(),
            "source_export_mutated": False,
            "interpretation_guardrail": "legacy v2 compatibility fixture",
            "temporal_adaptation_status": "unavailable",
        },
        footer_metadata={
            b"palette.schema_id": RESPONSE_SCHEMA_ID.encode(),
            b"palette.schema_version": b"2",
            b"palette.training_response_config": json.dumps(
                config.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode(),
        },
    )

    with pytest.raises(ValueError, match="unsupported training-response schema"):
        load_training_response_manifest(output_root, "legacy_v2")
    manifest = load_training_response_manifest(
        output_root,
        "legacy_v2",
        allow_legacy_v2=True,
    )
    assert manifest["schema_version"] == LEGACY_EXACT_SCHEMA_VERSION
    rows = scan_training_response_table(
        output_root,
        "legacy_v2",
        "training_response_features",
        columns=("recording_id",),
        allow_legacy_v2=True,
    ).collect()
    assert rows.height == 6


def test_training_workflow_rejects_source_manifest_generation_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "training"
    _write_source_export(source_root, "source_001")
    import fisheye.training_response.workflow as workflow

    real_load = workflow._load_object_snapshot

    def replace_after_snapshot(path: Path):
        snapshot = real_load(path)
        replacement = json.loads(path.read_text(encoding="utf-8"))
        replacement["replacement_generation"] = "newer"
        path.write_text(json.dumps(replacement), encoding="utf-8")
        return snapshot

    monkeypatch.setattr(workflow, "_load_object_snapshot", replace_after_snapshot)
    with pytest.raises(ValueError, match="changed during training-response planning"):
        run_training_response_analytics(
            source_export_root=source_root,
            source_export_run_id="source_001",
            output_root=output_root,
            analysis_run_id="training_snapshot",
            config=TrainingResponseConfig(cluster_stability_resamples=2),
        )
    assert not (
        output_root / "v2" / "manifests" / "analysis_run_id=training_snapshot.json"
    ).exists()


def test_training_response_validation_rejects_rehashed_identity_binding_tamper(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "training"
    _write_source_export(source_root, "source_001")
    run_training_response_analytics(
        source_export_root=source_root,
        source_export_run_id="source_001",
        output_root=output_root,
        analysis_run_id="training_identity_tamper",
        config=TrainingResponseConfig(cluster_stability_resamples=0),
    )
    manifest_path = (
        output_root
        / "v2"
        / "manifests"
        / "analysis_run_id=training_identity_tamper.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_registry_identity_receipt"]["sources"][0]["subject_id"] = (
        "tampered_subject"
    )
    manifest["manifest_payload_sha256"] = payload_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "manifest_payload_sha256"
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="receipt differs from the digest-bound"):
        validate_training_response_run(output_root, "training_identity_tamper")


def test_training_workflow_rejects_symlinked_source_manifest_namespace(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    outside = tmp_path / "outside"
    outside.mkdir()
    (source_root / "v1").mkdir(parents=True)
    (source_root / "v1" / "manifests").symlink_to(
        outside,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="must not be a symlink"):
        run_training_response_analytics(
            source_export_root=source_root,
            source_export_run_id="source_001",
            output_root=tmp_path / "output",
            analysis_run_id="training_001",
        )

    assert not any(outside.iterdir())


def test_training_response_component_filters_and_labels_noncausal_axes() -> None:
    rows = [
        {
            "recording_id": "recording_001",
            "protocol_name": "RedScare",
            "classification_status": "complete",
            "primary_training_profile": "active_distance_maintenance",
            "aggressive_proximity_score": 1.5,
            "locomotor_response_score": 1.0,
        },
        {
            "recording_id": "recording_002",
            "protocol_name": "GoodCopBadCop",
            "classification_status": "invalid",
        },
    ]
    filtered = filter_training_response_rows(
        rows, protocols=("RedScare",), statuses=("complete",)
    )
    figure = training_response_scatter_figure(filtered)

    assert [row["recording_id"] for row in filtered] == ["recording_001"]
    assert figure is not None
    assert figure.layout.xaxis.title.text == "Aggressive proximity score (farther →)"
    assert "avoid" not in figure.layout.title.text.lower()


def test_strategy_sankey_matches_only_complete_recording_level_pairs() -> None:
    baseline = [
        {
            "recording_id": "a",
            "classification_status": "complete",
            "primary_strategy": "broad_even_explorer",
        },
        {
            "recording_id": "b",
            "classification_status": "complete",
            "primary_strategy": "active_wall_following",
        },
        {
            "recording_id": "c",
            "classification_status": "invalid",
            "primary_strategy": "unavailable",
        },
    ]
    training = [
        {
            "recording_id": "a",
            "protocol_name": "RedScare",
            "classification_status": "complete",
            "primary_training_profile": "active_distance_maintenance",
        },
        {
            "recording_id": "b",
            "protocol_name": "GoodCopBadCop",
            "classification_status": "complete",
            "primary_training_profile": "active_distance_maintenance",
        },
        {
            "recording_id": "c",
            "protocol_name": "RedScare",
            "classification_status": "complete",
            "primary_training_profile": "low_activity_close_proximity",
        },
    ]
    figure = strategy_transition_sankey_figure(baseline, training)

    assert figure is not None
    sankey = figure.data[0]
    assert sum(sankey.link.value) == 2
    assert "Baseline · broad_even_explorer" in sankey.node.label
    assert "Training · active_distance_maintenance" in sankey.node.label
    assert "2 matched focal-fish sessions" in figure.layout.title.text
    assert strategy_transition_sankey_figure([], training) is None
    with pytest.raises(ValueError, match="duplicate complete training row"):
        strategy_transition_sankey_figure(baseline, [training[0], training[0]])
