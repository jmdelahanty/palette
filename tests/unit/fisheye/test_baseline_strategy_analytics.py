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
    BASELINE_BEHAVIOR_TIME_BINS_TABLE,
    BASELINE_KINEMATIC_SAMPLES_TABLE,
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
    TABLE_CONTRACTS,
    contract_snapshot,
)
from fisheye.analytics_exports.publication import sha256_file
from fisheye.analytics_exports.registry_identity import (
    build_registry_identity_receipt,
    build_registry_identity_source,
)
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


def _registry_identity(recording_id: str) -> dict[str, str]:
    return {
        "session_id": f"session-{recording_id}",
        "subject_id": f"subject-{recording_id}",
    }


def _summary(recording_id: str = "recording_001", **overrides):
    row = {
        "export_run_id": "export_001",
        "recording_id": recording_id,
        **_registry_identity(recording_id),
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
                **_registry_identity("recording_001"),
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
            **_registry_identity("recording_001"),
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
        **_registry_identity(recording_id),
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
                **_registry_identity(f"recording_{index}"),
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


def _write_summary_only_export(
    root: Path,
    run_id: str,
    *,
    include_time_bins: bool = False,
    include_samples: bool = False,
) -> Path:
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
    parts_by_table = {table_name: [part]}
    row_counts = {table_name: len(rows)}
    capabilities = ["core.baseline.behavior_summary"]

    if include_time_bins:
        time_table = BASELINE_BEHAVIOR_TIME_BINS_TABLE
        time_rows = []
        for recording_index in range(6):
            for bin_index, wall_fraction in enumerate((0.8, 0.2)):
                row = {}
                for field in ARROW_TABLE_CONTRACTS[time_table].fields:
                    if field.nullable:
                        row[field.name] = None
                    elif field.arrow_type in {"int32", "int64"}:
                        row[field.name] = 1
                    elif field.arrow_type == "float64":
                        row[field.name] = 1.0
                    else:
                        row[field.name] = "value"
                recording_id = f"recording_{recording_index}"
                row.update(
                    {
                        "export_schema_version": EXPORT_SCHEMA_VERSION,
                        "table_name": time_table,
                        "recording_id": recording_id,
                        **_registry_identity(recording_id),
                        "zarr_path": f"/{recording_id}_analysis.zarr",
                        "source_lineage_hash": hashlib.sha256(
                            recording_id.encode("utf-8")
                        ).hexdigest(),
                        "source_refs_json": "{}",
                        "track_id": 0,
                        "baseline_window_id": 0,
                        "baseline_window_label": "pre",
                        "time_bin_index": bin_index,
                        "relative_start_s": float(bin_index * 5),
                        "relative_end_s": float((bin_index + 1) * 5),
                        "time_bin_duration_s": 5.0,
                        "source_start_frame": bin_index * 50,
                        "source_end_frame": (bin_index + 1) * 50 - 1,
                        "wall_fraction": wall_fraction,
                        "mean_speed_mm_s": 2.0,
                        "distance_travelled_mm": 10.0,
                        "mean_center_distance_mm": wall_fraction * 10.0,
                    }
                )
                time_rows.append(row)
        time_part = (
            root
            / generation_path
            / "tables"
            / time_table
            / "part-00000.parquet"
        )
        time_part.parent.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(
                time_rows,
                schema=exact_arrow_schema(
                    time_table,
                    metadata={
                        b"palette.export_schema_id": EXPORT_SCHEMA_ID.encode(),
                        b"palette.export_schema_version": str(
                            EXPORT_SCHEMA_VERSION
                        ).encode(),
                        b"palette.table_contract": json.dumps(
                            TABLE_CONTRACTS[time_table].to_dict(),
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode(),
                    },
                ),
            ),
            time_part,
        )
        parts_by_table[time_table] = [time_part]
        row_counts[time_table] = len(time_rows)
        capabilities.append("core.baseline.behavior_time_bins")

    if include_samples:
        sample_table = BASELINE_KINEMATIC_SAMPLES_TABLE
        sample_rows = []
        for recording_index in range(6):
            recording_id = f"recording_{recording_index}"
            for source_sample_index, payload in enumerate(_samples()):
                row = {}
                for field in ARROW_TABLE_CONTRACTS[sample_table].fields:
                    if field.nullable:
                        row[field.name] = None
                    elif field.arrow_type in {"int32", "int64"}:
                        row[field.name] = 1
                    elif field.arrow_type == "float64":
                        row[field.name] = 1.0
                    elif field.arrow_type == "bool":
                        row[field.name] = True
                    else:
                        row[field.name] = "value"
                row.update(payload)
                row.update(
                    {
                        "export_schema_version": EXPORT_SCHEMA_VERSION,
                        "table_name": sample_table,
                        "recording_id": recording_id,
                        **_registry_identity(recording_id),
                        "zarr_path": f"/{recording_id}_analysis.zarr",
                        "source_lineage_hash": hashlib.sha256(
                            recording_id.encode("utf-8")
                        ).hexdigest(),
                        "source_refs_json": "{}",
                        "track_id": 0,
                        "baseline_window_id": 0,
                        "baseline_window_label": "pre",
                        "source_sample_index": source_sample_index,
                        "source_time_s": float(payload["relative_time_s"]),
                        "sampling_policy": "all_source_samples_v1",
                        "sampling_stride_frames": 1,
                        "requested_sample_rate_hz": None,
                        "source_sample_rate_hz": 10.0,
                        "nominal_sample_rate_hz": 10.0,
                        "effective_sample_rate_hz": 10.0,
                    }
                )
                sample_rows.append(row)
        sample_part = (
            root
            / generation_path
            / "tables"
            / sample_table
            / "part-00000.parquet"
        )
        sample_part.parent.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(
                sample_rows,
                schema=exact_arrow_schema(
                    sample_table,
                    metadata={
                        b"palette.export_schema_id": EXPORT_SCHEMA_ID.encode(),
                        b"palette.export_schema_version": str(
                            EXPORT_SCHEMA_VERSION
                        ).encode(),
                        b"palette.table_contract": json.dumps(
                            TABLE_CONTRACTS[sample_table].to_dict(),
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode(),
                    },
                ),
            ),
            sample_part,
        )
        parts_by_table[sample_table] = [sample_part]
        row_counts[sample_table] = len(sample_rows)
        capabilities.append("core.baseline.kinematic_samples")

    manifest = root / "v1" / "manifests" / f"export_run_id={run_id}.json"
    manifest.parent.mkdir(parents=True)
    relative_parts_by_table = {
        name: [
            (generation_path / "tables" / name / path.name).as_posix()
            for path in paths
        ]
        for name, paths in parts_by_table.items()
    }
    publication_parts = {
        name: [
            {
                "path": relative_path,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "row_count": row_counts[name],
            }
            for path, relative_path in zip(paths, relative_parts_by_table[name])
        ]
        for name, paths in parts_by_table.items()
    }
    table_names = list(parts_by_table)
    source_zarrs = [f"/recording_{index}_analysis.zarr" for index in range(6)]
    registry_identity = build_registry_identity_receipt(
        registry_path=Path("/registry/test.sqlite"),
        sources=[
            build_registry_identity_source(
                zarr_path=path,
                rows=[
                    {
                        "dataset_id": index + 1,
                        "recording_id": f"recording_{index}",
                        "experimental_session_id": f"session-recording_{index}",
                        "experimental_session_snapshot_id": (
                            f"00000000-0000-4000-8000-{index + 1:012d}"
                        ),
                        "experimental_session_schema_id": (
                            "palette.registry.experimental_session.v1"
                        ),
                        "experimental_session_creation_registry_schema_version": 1,
                        "experimental_session_identity_status": "explicit",
                        "experimental_session_assignment_snapshot_id": (
                            f"10000000-0000-4000-8000-{index + 1:012d}"
                        ),
                        "experimental_session_assignment_batch_id": (
                            "20000000-0000-4000-8000-000000000001"
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
                        "fish_id": f"subject-recording_{index}",
                        "subject_count": 1,
                        "subject_ids_json": None,
                    }
                ],
            )
            for index, path in enumerate(source_zarrs)
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
                "tables_requested": table_names,
                "table_contracts": contract_snapshot(table_names),
                "arrow_schema_contracts": arrow_contract_envelope(table_names),
                "row_counts_by_table": row_counts,
                "part_files_by_table": relative_parts_by_table,
                "capabilities": capabilities,
                "publication": {
                    "schema_id": "palette.analytics_export.publication",
                    "schema_version": 1,
                    "state": "complete",
                    "generation_id": "test",
                    "generation_path": generation_path.as_posix(),
                    "parts_by_table": publication_parts,
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
    feature_part = output_root / result["part_files_by_table"][
        "baseline_strategy_features"
    ][0]
    assert feature_part.is_file()
    feature_table = pq.ParquetFile(feature_part).read()
    assert feature_table.num_rows == 6
    assert set(feature_table.column("source_export_run_id").to_pylist()) == {"source_001"}
    for table_name, parts in result["part_files_by_table"].items():
        for part_path in parts:
            table = pq.ParquetFile(output_root / part_path).read(
                columns=["source_export_run_id"]
            )
            assert set(table.column(0).to_pylist()) in (
                set(),
                {"source_001"},
            ), table_name
    lazy = scan_strategy_table(
        output_root,
        "strategy_001",
        "baseline_strategy_features",
        columns=("recording_id", "feature_status"),
    )
    assert lazy.collect().shape == (6, 2)


def test_workflow_consumes_manifest_selected_exact_time_bins(tmp_path: Path) -> None:
    source_root = tmp_path / "source_export"
    output_root = tmp_path / "derived_analytics"
    _write_summary_only_export(
        source_root,
        "source_with_time_bins",
        include_time_bins=True,
    )

    result = run_strategy_analytics(
        source_export_root=source_root,
        source_export_run_id="source_with_time_bins",
        output_root=output_root,
        analysis_run_id="strategy_with_time_bins",
        config=StrategyFeatureConfig(cluster_stability_resamples=2),
    )

    feature_part = output_root / result["part_files_by_table"][
        "baseline_strategy_features"
    ][0]
    rows = pq.ParquetFile(feature_part).read().to_pylist()
    assert len(rows) == 6
    assert all(row["time_bin_features_available"] is True for row in rows)
    assert all(row["wall_fraction_delta_late_minus_early"] < 0 for row in rows)
    assert all(row["wall_fraction_slope_per_baseline"] < 0 for row in rows)


def test_workflow_rejects_rehashed_unexpected_time_bin_column(tmp_path: Path) -> None:
    source_root = tmp_path / "source_export"
    output_root = tmp_path / "derived_analytics"
    run_id = "tampered_time_bins"
    _write_summary_only_export(source_root, run_id, include_time_bins=True)
    manifest_path = (
        source_root / "v1" / "manifests" / f"export_run_id={run_id}.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_part = payload["part_files_by_table"][BASELINE_BEHAVIOR_TIME_BINS_TABLE][0]
    part = source_root / relative_part
    table = pq.ParquetFile(part).read().append_column(
        "future_source_metric",
        pa.array([999.0] * 12, type=pa.float64()),
    )
    pq.write_table(table, part)
    entry = payload["publication"]["parts_by_table"][
        BASELINE_BEHAVIOR_TIME_BINS_TABLE
    ][0]
    entry["sha256"] = sha256_file(part)
    entry["size_bytes"] = part.stat().st_size
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExportValidationError, match="physical Arrow fields"):
        run_strategy_analytics(
            source_export_root=source_root,
            source_export_run_id=run_id,
            output_root=output_root,
            analysis_run_id="strategy_rejects_tampering",
        )


def test_workflow_consumes_manifest_selected_exact_samples(tmp_path: Path) -> None:
    source_root = tmp_path / "source_export"
    output_root = tmp_path / "derived_analytics"
    _write_summary_only_export(
        source_root,
        "source_with_samples",
        include_samples=True,
    )

    result = run_strategy_analytics(
        source_export_root=source_root,
        source_export_run_id="source_with_samples",
        output_root=output_root,
        analysis_run_id="strategy_with_samples",
        config=StrategyFeatureConfig(
            min_sample_count=10,
            cluster_stability_resamples=2,
        ),
    )

    feature_part = output_root / result["part_files_by_table"][
        "baseline_strategy_features"
    ][0]
    feature_rows = pq.ParquetFile(feature_part).read().to_pylist()
    assert len(feature_rows) == 6
    assert all(row["sample_features_available"] is True for row in feature_rows)
    assert result["row_counts_by_table"]["baseline_exploration_episodes"] == 12


def test_workflow_rejects_rehashed_unexpected_sample_column(tmp_path: Path) -> None:
    source_root = tmp_path / "source_export"
    output_root = tmp_path / "derived_analytics"
    run_id = "tampered_samples"
    _write_summary_only_export(source_root, run_id, include_samples=True)
    manifest_path = (
        source_root / "v1" / "manifests" / f"export_run_id={run_id}.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_part = payload["part_files_by_table"][BASELINE_KINEMATIC_SAMPLES_TABLE][0]
    part = source_root / relative_part
    table = pq.ParquetFile(part).read()
    table = table.append_column(
        "future_source_metric",
        pa.array([999.0] * table.num_rows, type=pa.float64()),
    )
    pq.write_table(table, part)
    entry = payload["publication"]["parts_by_table"][
        BASELINE_KINEMATIC_SAMPLES_TABLE
    ][0]
    entry["sha256"] = sha256_file(part)
    entry["size_bytes"] = part.stat().st_size
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExportValidationError, match="physical Arrow fields"):
        run_strategy_analytics(
            source_export_root=source_root,
            source_export_run_id=run_id,
            output_root=output_root,
            analysis_run_id="strategy_rejects_sample_tampering",
        )


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
