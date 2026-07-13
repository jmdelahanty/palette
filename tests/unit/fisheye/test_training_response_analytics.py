from __future__ import annotations

import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from apps.marimo.components.training_response import (
    filter_training_response_rows,
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
from fisheye.training_response.cohort import (
    classify_training_response_features,
    discover_training_response_clusters,
)
from fisheye.training_response.contracts import TrainingResponseConfig
from fisheye.training_response.features import derive_training_response_features
from fisheye.training_response.query import (
    discover_training_response_catalog,
    scan_training_response_qc_rows,
    scan_training_response_table,
    select_training_response_run_id,
)
from fisheye.training_response.validation import validate_training_response_run
from fisheye.training_response.workflow import (
    build_training_response_tables,
    run_training_response_analytics,
)


def _behavior_rows(recording_id: str, *, scale: float = 1.0, dropout: float = 0.0):
    common = {
        "recording_id": recording_id,
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
                    "recording_id": recording_id,
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
            "recording_id": recording_id,
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
            "recording_id": recording_id,
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
                source_export_run_id="source_001",
                behavior_rows=_behavior_rows(recording_id, scale=0.5 + index * 0.3),
                distance_rows=_distance_rows(
                    recording_id, aggressive_training_p50=p50
                ),
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


def test_build_tables_returns_one_row_per_recording_per_table() -> None:
    recording_ids = [f"recording_{index}" for index in range(6)]
    tables = build_training_response_tables(
        source_export_run_id="source_001",
        behavior_rows=[row for item in recording_ids for row in _behavior_rows(item)],
        distance_rows=[row for item in recording_ids for row in _distance_rows(item)],
        egocentric_rows=[row for item in recording_ids for row in _egocentric_rows(item)],
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
    assert {
        row["protocol_name"] for row in tables["training_response_features"]
    } == {"GoodCopBadCop"}


def test_cluster_discovery_reports_missing_stability_instead_of_validating_it() -> None:
    rows = []
    for index in range(8):
        group = -4.0 if index < 4 else 4.0
        rows.append(
            {
                "recording_id": f"recording_{index}",
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
    parts: dict[str, Path] = {}
    schemas: dict[str, tuple[str, ...]] = {}
    for table_name, raw_rows in rows_by_table.items():
        rows = []
        for raw in raw_rows:
            row = {column: None for column in TABLE_CONTRACTS[table_name].required_columns}
            row.update(raw)
            row["export_schema_version"] = EXPORT_SCHEMA_VERSION
            row["table_name"] = table_name
            rows.append(row)
        table = pa.Table.from_pylist(rows).replace_schema_metadata(
            {
                b"palette.export_schema_id": EXPORT_SCHEMA_ID.encode(),
                b"palette.export_schema_version": str(EXPORT_SCHEMA_VERSION).encode(),
                b"palette.table_contract": json.dumps(
                    TABLE_CONTRACTS[table_name].to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
            }
        )
        part = root / "v1" / table_name / f"export_run_id={run_id}" / "part-00000.parquet"
        part.parent.mkdir(parents=True)
        pq.write_table(table, part)
        parts[table_name] = part
        schemas[table_name] = tuple(table.schema.names)
    capabilities = [
        item.capability_id for item in resolve_capabilities(schemas) if item.available
    ]
    manifest = root / "v1" / "manifests" / f"export_run_id={run_id}.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "export_run_id": run_id,
                "schema_id": EXPORT_SCHEMA_ID,
                "schema_version": EXPORT_SCHEMA_VERSION,
                "tables_requested": list(rows_by_table),
                "table_contracts": contract_snapshot(list(rows_by_table)),
                "row_counts_by_table": {
                    name: len(rows) for name, rows in rows_by_table.items()
                },
                "part_files_by_table": {
                    name: [str(path)] for name, path in parts.items()
                },
                "capabilities": capabilities,
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
    assert result["temporal_adaptation_status"].startswith("unavailable")
    assert result["output_validation"]["status"] == "valid"
    assert validate_training_response_run(output_root, "training_001")["table_count"] == 3
    assert all(path.read_bytes() == source_hashes[name] for name, path in source_parts.items())
    lazy = scan_training_response_table(
        output_root,
        "training_001",
        "training_response_classification",
        columns=("recording_id", "aggressive_proximity_state"),
    )
    assert lazy.collect().shape == (6, 2)
    catalog = discover_training_response_catalog(output_root)
    assert select_training_response_run_id(
        catalog, "latest", source_export_run_id="source_001"
    ) == "training_001"
    with pytest.raises(ValueError, match="no ready training-response"):
        select_training_response_run_id(
            catalog,
            "latest",
            source_export_run_id="source_001",
            source_export_manifest_sha256="not-the-published-hash",
        )
    qc_rows = scan_training_response_qc_rows(output_root, "training_001").collect()
    assert qc_rows.shape[0] == 6
    assert "primary_training_profile" in qc_rows.columns
    assert "selected_component_count" in qc_rows.columns


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
