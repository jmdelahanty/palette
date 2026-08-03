from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fisheye.analytics_exports.arrow_contracts import (
    ARROW_TABLE_CONTRACTS,
    arrow_contract_envelope,
    exact_arrow_schema,
)
from fisheye.analytics_exports.contracts import (
    BASELINE_BEHAVIOR_SUMMARY_TABLE,
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
    TABLE_CONTRACTS,
    contract_snapshot,
)
from fisheye.analytics_exports.publication import sha256_file
from fisheye.analytics_exports.validation import ExportValidationError
from fisheye.baseline_strategy.cohort import (
    classify_strategy_features,
    discover_strategy_clusters,
)
from fisheye.baseline_strategy.contracts import StrategyFeatureConfig
from fisheye.baseline_strategy.features import (
    derive_baseline_strategy_features,
    derive_exploration_episodes,
    expected_wall_fraction,
)
from fisheye.baseline_strategy.query import scan_strategy_table
from fisheye.baseline_strategy.workflow import build_strategy_tables, run_strategy_analytics
from fisheye.baseline_strategy.validation import validate_strategy_analytics_run


def _summary(recording_id: str = "recording_001", **overrides):
    row = {
        "export_run_id": "export_001",
        "recording_id": recording_id,
        "track_id": 0,
        "baseline_window_id": 0,
        "baseline_window_label": "pre",
        "zarr_path": f"/{recording_id}_analysis.zarr",
        "duration_s": 20.0,
        "tracking_dropout_fraction": 0.0,
        "mean_speed_mm_s": 2.0,
        "median_speed_mm_s": 1.5,
        "p95_speed_mm_s": 4.0,
        "total_path_mm": 120.0,
        "bout_count": 20,
        "bout_rate_per_min": 60.0,
        "arena_radius_mm": 10.0,
        "wall_band_mm": 2.0,
        "wall_fraction": 0.7,
        "mean_center_distance_norm": 0.8,
        "spatial_entropy_normalized": 0.7,
        "quadrant_entropy_normalized": 0.9,
        "spatial_max_cell_fraction": 0.15,
        "quadrant_max_fraction": 0.3,
    }
    row.update(overrides)
    return row


def _samples() -> list[dict[str, object]]:
    rows = []
    for index in range(60):
        time_s = index * 0.1
        active = 5 <= index < 20 or 30 <= index < 50
        angle = index * 0.16
        radius = 9.2 if active else 8.5
        rows.append(
            {
                "recording_id": "recording_001",
                "track_id": 0,
                "baseline_window_id": 0,
                "source_frame": index,
                "relative_time_s": time_s,
                "x_arena_mm": radius * math.cos(angle),
                "y_arena_mm": radius * math.sin(angle),
                "speed_mm_s": 2.0 if active else 0.1,
                "frame_path_distance_mm": 0.2 if active else 0.01,
                "wall": radius >= 8.0,
                "position_valid": True,
                "sample_valid": True,
            }
        )
    return rows


def test_expected_wall_fraction_uses_circular_accessible_area() -> None:
    assert np.isclose(expected_wall_fraction(10.0, 2.0), 0.36)
    assert expected_wall_fraction(10.0, 10.0) == 1.0
    assert expected_wall_fraction(None, 2.0) is None


def test_feature_builder_combines_summary_time_samples_and_episodes() -> None:
    time_bins = [
        {
            "recording_id": "recording_001",
            "track_id": 0,
            "baseline_window_id": 0,
            "time_bin_index": index,
            "relative_start_s": index * 5.0,
            "relative_end_s": (index + 1) * 5.0,
            "wall_fraction": wall,
            "mean_speed_mm_s": 2.0,
            "distance_travelled_mm": 20.0,
            "mean_center_distance_mm": wall * 10.0,
            "bout_count": 4,
        }
        for index, wall in enumerate((0.9, 0.8, 0.5, 0.3))
    ]
    feature, episodes = derive_baseline_strategy_features(
        _summary(),
        time_bins,
        _samples(),
        config=StrategyFeatureConfig(min_sample_count=10),
    )

    assert feature["feature_status"] == "complete"
    assert feature["sample_features_available"] is True
    assert feature["time_bin_features_available"] is True
    assert np.isclose(feature["expected_uniform_wall_fraction"], 0.36)
    assert feature["wall_enrichment_ratio"] > 1.0
    assert feature["boundary_distance_method"] == "circle_radius_minus_center_distance_v1"
    assert feature["wall_fraction_denominator"] == "valid_position_frames"
    assert feature["active_wall_fraction_denominator"] == "active_valid_portable_samples"
    assert feature["wall_fraction_delta_late_minus_early"] < 0
    assert feature["wall_fraction_slope_per_baseline"] < 0
    assert 0 <= feature["occupancy_entropy_accessible_normalized"] <= 1
    assert feature["exploration_episode_count"] == 2
    assert len(episodes) == 2
    assert all(episode["path_length_method"] == "portable_sample_xy_chord_sum" for episode in episodes)


def test_episode_builder_does_not_invent_rows_without_samples() -> None:
    assert derive_exploration_episodes(_summary(), []) == []


def test_exported_boundary_distance_is_authoritative_for_wall_samples() -> None:
    samples = _samples()
    for sample in samples:
        sample["distance_to_arena_boundary_mm"] = 0.5
        sample["wall"] = False
        sample["boundary_distance_method"] = "test_mask_distance_transform_v1"
    feature, episodes = derive_baseline_strategy_features(
        _summary(boundary_distance_method="test_mask_distance_transform_v1"),
        sample_rows=samples,
        config=StrategyFeatureConfig(min_sample_count=10),
    )

    assert feature["boundary_distance_sample_source"] == (
        "exported_distance_to_arena_boundary_mm"
    )
    assert feature["active_wall_fraction"] == 1.0
    assert all(episode["wall_sample_fraction"] == 1.0 for episode in episodes)


def _feature(recording_id: str, value: float, *, wall: bool = False) -> dict[str, object]:
    return {
        "recording_id": recording_id,
        "track_id": 0,
        "baseline_window_id": 0,
        "feature_status": "complete",
        "path_per_min_mm": math.exp(max(-4.0, value + 3.0)),
        "bout_rate_per_min": math.exp(max(-4.0, value + 2.0)),
        "active_sample_fraction": 1.0 / (1.0 + math.exp(-value)),
        "p95_speed_mm_s": math.exp(max(-4.0, value + 1.0)),
        "wall_enrichment_log2": value if wall else -value,
        "mean_center_distance_norm": 0.9 if wall else 0.3,
        "active_wall_fraction": 0.9 if wall else 0.1,
        "wall_following_episode_fraction": 0.9 if wall else 0.1,
        "occupancy_coverage_fraction": 1.0 / (1.0 + math.exp(-value)),
        "occupancy_entropy_accessible_normalized": 1.0 / (1.0 + math.exp(-value)),
        "occupancy_js_divergence_uniform": 1.0 / (1.0 + math.exp(value)),
        "occupancy_max_cell_fraction": 1.0 / (1.0 + math.exp(value)),
        "dominant_dwell_cell_fraction": 1.0 / (1.0 + math.exp(value)),
        "dominant_to_second_dwell_ratio": math.exp(max(-3.0, -value + 1.0)),
        "dominant_dwell_return_fraction": 1.0 / (1.0 + math.exp(value)),
        "dominant_dwell_visit_count": max(1, int(5 - value)),
        "wall_fraction_delta_late_minus_early": -value,
        "wall_fraction_slope_per_baseline": -value,
        "center_distance_norm_delta_late_minus_early": -value,
        "center_distance_norm_slope_per_baseline": -value,
    }


def test_classification_keeps_activity_and_boundary_as_separate_axes() -> None:
    features = [
        _feature("inactive", -3.0),
        _feature("middle_a", -0.2),
        _feature("middle_b", 0.2),
        _feature("active_wall", 3.0, wall=True),
    ]
    classified = classify_strategy_features(
        features,
        config=StrategyFeatureConfig(relative_score_threshold=0.6),
    )
    by_id = {row["recording_id"]: row for row in classified}

    assert by_id["inactive"]["activity_state"] == "inactive"
    assert by_id["active_wall"]["activity_state"] == "active"
    assert by_id["active_wall"]["boundary_strategy"] == "wall_following"
    assert by_id["active_wall"]["anxiety_inference_permitted"] is False
    assert by_id["active_wall"]["confidence_semantics"] == "descriptive_distance_not_probability"


def test_cluster_discovery_reports_model_selection_and_assignment_uncertainty() -> None:
    rows = []
    for index in range(8):
        group = -4.0 if index < 4 else 4.0
        jitter = (index % 4) * 0.03
        rows.append(
            {
                "recording_id": f"recording_{index}",
                "track_id": 0,
                "baseline_window_id": 0,
                "classification_status": "complete",
                "activity_score": group + jitter,
                "boundary_score": group + jitter,
                "spatial_distribution_score": group + jitter,
                "home_base_score": -group + jitter,
                "temporal_expansion_score": group + jitter,
            }
        )
    clusters = discover_strategy_clusters(
        rows,
        config=StrategyFeatureConfig(
            cluster_max_components=3,
            cluster_stability_resamples=4,
            random_seed=2,
        ),
    )

    assert {row["selected_component_count"] for row in clusters} == {2}
    assert {row["cluster_id"] for row in clusters} == {0, 1}
    assert all(row["cluster_probability"] > 0.9 for row in clusters)
    assert all(row["cluster_semantics"].startswith("unsupervised") for row in clusters)


def test_build_strategy_tables_returns_all_four_derived_tables() -> None:
    summaries = [_summary(f"recording_{index}", total_path_mm=50.0 + 20 * index) for index in range(6)]
    tables = build_strategy_tables(summary_rows=summaries)

    assert set(tables) == {
        "baseline_strategy_features",
        "baseline_exploration_episodes",
        "baseline_strategy_classification",
        "baseline_strategy_clusters",
    }
    assert len(tables["baseline_strategy_features"]) == 6
    assert len(tables["baseline_strategy_classification"]) == 6
    assert len(tables["baseline_strategy_clusters"]) == 6
    assert tables["baseline_exploration_episodes"] == []


def _write_summary_only_export(root: Path, run_id: str) -> Path:
    table_name = BASELINE_BEHAVIOR_SUMMARY_TABLE
    generation_path = (
        Path("v1")
        / ".generations"
        / f"export_run_id={run_id}"
        / "generation=test"
    )
    rows = []
    for index in range(6):
        row = {}
        for field in ARROW_TABLE_CONTRACTS[table_name].fields:
            if field.nullable:
                row[field.name] = None
            elif field.arrow_type in {"int32", "int64"}:
                row[field.name] = 1
            elif field.arrow_type == "float64":
                row[field.name] = 1.0
            else:
                row[field.name] = "value"
        row.update(
            _summary(
                f"recording_{index}",
                total_path_mm=20.0 + 50.0 * index,
                bout_rate_per_min=5.0 + 10.0 * index,
                wall_fraction=0.2 + 0.1 * index,
                mean_center_distance_norm=0.3 + 0.1 * index,
                spatial_entropy_normalized=0.3 + 0.1 * index,
                quadrant_entropy_normalized=0.4 + 0.08 * index,
            )
        )
        # Production analytics rows receive export identity from their immutable
        # manifest/directory, not from a duplicated Parquet column.
        row.pop("export_run_id", None)
        row["export_schema_version"] = EXPORT_SCHEMA_VERSION
        row["table_name"] = table_name
        row["source_lineage_hash"] = hashlib.sha256(
            row["recording_id"].encode("utf-8")
        ).hexdigest()
        row["source_refs_json"] = "{}"
        rows.append(row)
    part = root / generation_path / "tables" / table_name / "part-00000.parquet"
    part.parent.mkdir(parents=True)
    table = pa.Table.from_pylist(
        rows,
        schema=exact_arrow_schema(
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
        ),
    )
    pq.write_table(table, part)
    manifest = root / "v1" / "manifests" / f"export_run_id={run_id}.json"
    manifest.parent.mkdir(parents=True)
    relative_part = (generation_path / "tables" / table_name / part.name).as_posix()
    manifest.write_text(
        json.dumps(
            {
                "export_run_id": run_id,
                "schema_id": EXPORT_SCHEMA_ID,
                "schema_version": EXPORT_SCHEMA_VERSION,
                "tables_requested": [table_name],
                "table_contracts": contract_snapshot([table_name]),
                "arrow_schema_contracts": arrow_contract_envelope([table_name]),
                "row_counts_by_table": {table_name: len(rows)},
                "part_files_by_table": {table_name: [relative_part]},
                "capabilities": ["core.baseline.behavior_summary"],
                "publication": {
                    "schema_id": "palette.analytics_export.publication",
                    "schema_version": 1,
                    "state": "complete",
                    "generation_id": "test",
                    "generation_path": generation_path.as_posix(),
                    "parts_by_table": {
                        table_name: [
                            {
                                "path": relative_part,
                                "sha256": sha256_file(part),
                                "size_bytes": part.stat().st_size,
                                "row_count": len(rows),
                            }
                        ]
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return part


def test_workflow_reads_validated_export_and_publishes_separate_manifest(tmp_path: Path) -> None:
    source_root = tmp_path / "source_export"
    output_root = tmp_path / "derived_analytics"
    source_part = _write_summary_only_export(source_root, "source_001")
    source_bytes = source_part.read_bytes()

    result = run_strategy_analytics(
        source_export_root=source_root,
        source_export_run_id="source_001",
        output_root=output_root,
        analysis_run_id="strategy_001",
        config=StrategyFeatureConfig(cluster_stability_resamples=2),
    )

    assert source_part.read_bytes() == source_bytes
    assert result["source_export_mutated"] is False
    assert result["source_export_run_id"] == "source_001"
    assert result["row_provenance"] == {
        "source_export_run_id": "source_001",
        "status": "complete",
    }
    assert len(result["source_export_manifest_sha256"]) == 64
    assert result["row_counts_by_table"]["baseline_strategy_features"] == 6
    assert result["row_counts_by_table"]["baseline_strategy_classification"] == 6
    assert Path(result["manifest_path"]).is_file()
    assert result["output_validation"]["status"] == "valid"
    assert validate_strategy_analytics_run(output_root, "strategy_001")["table_count"] == 4
    feature_part = Path(result["part_files_by_table"]["baseline_strategy_features"][0])
    assert feature_part.is_file()
    feature_table = pq.ParquetFile(feature_part).read()
    assert feature_table.num_rows == 6
    assert set(feature_table.column("source_export_run_id").to_pylist()) == {"source_001"}
    for table_name, parts in result["part_files_by_table"].items():
        for part_path in parts:
            table = pq.ParquetFile(part_path).read(columns=["source_export_run_id"])
            assert set(table.column(0).to_pylist()) == {"source_001"}, table_name
    lazy = scan_strategy_table(
        output_root,
        "strategy_001",
        "baseline_strategy_features",
        columns=("recording_id", "feature_status"),
    )
    assert lazy.collect().shape == (6, 2)


def test_workflow_binds_manifest_digest_to_loaded_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "source_export"
    output_root = tmp_path / "derived_analytics"
    _write_summary_only_export(source_root, "source_001")
    manifest_path = (
        source_root
        / "v1"
        / "manifests"
        / "export_run_id=source_001.json"
    )
    expected_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    import fisheye.baseline_strategy.workflow as workflow

    real_load = workflow._load_manifest_snapshot

    def replace_after_snapshot(root: Path, run_id: str):
        snapshot = real_load(root, run_id)
        replacement = json.loads(manifest_path.read_text(encoding="utf-8"))
        replacement["replacement_generation"] = "newer"
        manifest_path.write_text(json.dumps(replacement), encoding="utf-8")
        return snapshot

    monkeypatch.setattr(workflow, "_load_manifest_snapshot", replace_after_snapshot)
    result = run_strategy_analytics(
        source_export_root=source_root,
        source_export_run_id="source_001",
        output_root=output_root,
        analysis_run_id="strategy_snapshot",
        config=StrategyFeatureConfig(cluster_stability_resamples=2),
    )

    assert result["source_export_manifest_sha256"] == expected_digest


def test_strategy_workflow_rejects_symlinked_source_manifest_namespace(
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
        run_strategy_analytics(
            source_export_root=source_root,
            source_export_run_id="source_001",
            output_root=tmp_path / "output",
            analysis_run_id="strategy_001",
        )

    assert not any(outside.iterdir())


def test_workflow_rejects_unexpected_source_identity_column(tmp_path: Path) -> None:
    source_root = tmp_path / "source_export"
    output_root = tmp_path / "derived_analytics"
    part = _write_summary_only_export(source_root, "source_001")
    table = pq.ParquetFile(part).read().append_column(
        "export_run_id", pa.array(["wrong_export"] * 6)
    )
    pq.write_table(table, part)

    with pytest.raises(
        ExportValidationError,
        match="digest mismatch|physical Arrow fields",
    ):
        run_strategy_analytics(
            source_export_root=source_root,
            source_export_run_id="source_001",
            output_root=output_root,
            analysis_run_id="strategy_001",
        )
