from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fisheye.analytics_exports.contracts import (
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
    TABLE_CONTRACTS,
    contract_snapshot,
)
from fisheye.group_statistics.goodcopbadcop import (
    DESCRIPTIVE_TABLE,
    GoodCopBadCopStatisticsConfig,
    SUMMARY_TABLE,
    compute_goodcopbadcop_descriptive_summaries,
    compute_goodcopbadcop_statistics,
    metric_specs_for_families,
    write_goodcopbadcop_statistics,
)
from fisheye.group_statistics.paired import (
    benjamini_hochberg,
    compute_one_sample_signed_rank,
    compute_paired_contrast,
)
from fisheye.utils.compute_group_statistics import main as compute_group_statistics_main


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path / "part-00000.parquet")


def _make_goodcopbadcop_export(root: Path, export_run_id: str = "source_export") -> Path:
    export_root = root / "palette_analytics"
    rows: list[dict] = []
    values = {
        "r1": {"pre": 1.0, "training": 2.0, "post": 1.5},
        "r2": {"pre": 2.0, "training": 4.0, "post": 2.5},
        "r3": {"pre": 3.0, "training": 6.0, "post": 4.0},
    }
    label_by_condition = {
        "pre": "pre_event",
        "training": "training_event",
        "post": "post_event",
    }
    for recording_id, condition_values in values.items():
        for chaser_index in (0, 1):
            for condition, value in condition_values.items():
                offset = float(chaser_index)
                rows.append(
                    {
                        "recording_id": recording_id,
                        "window_label": label_by_condition[condition],
                        "chaser_index": chaser_index,
                        "chaser_column_index": chaser_index,
                        "behavior_class": "unknown",
                        "mean_distance_mm": value + offset,
                        "p50_distance_mm": value + offset,
                        "fraction_within_threshold": (value + offset) / 10.0,
                        "collection_id": "collection_test",
                        "collection_manifest_sha256": "abc123",
                    }
                )
    _write_rows(
        export_root / "v1" / "chaser_epoch_distance_summary" / f"export_run_id={export_run_id}",
        rows,
    )
    role_by_recording = {
        "r1": ("aggressive", "inert"),
        "r2": ("inert", "aggressive"),
        "r3": ("aggressive", "inert"),
    }
    object_phase_rows = [
        {
            "recording_id": recording_id,
            "object_column_index": object_column_index,
            "object_role": role,
        }
        for recording_id, roles in role_by_recording.items()
        for object_column_index, role in enumerate(roles)
    ]
    _write_rows(
        export_root
        / "v1"
        / "chaser_quadrant_occupancy_chaser_phase"
        / f"export_run_id={export_run_id}",
        object_phase_rows,
    )
    manifest = {
        "export_run_id": export_run_id,
        "schema_id": EXPORT_SCHEMA_ID,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "table_contracts": contract_snapshot(
            [
                "chaser_epoch_distance_summary",
                "chaser_quadrant_occupancy_chaser_phase",
            ]
        ),
        "row_counts_by_table": {
            "chaser_epoch_distance_summary": len(rows),
            "chaser_quadrant_occupancy_chaser_phase": len(object_phase_rows),
        },
        "collection_manifest": {
            "collection_id": "collection_test",
            "manifest_sha256": "abc123",
        },
    }
    manifest_dir = export_root / "v1" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"export_run_id={export_run_id}.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return export_root


def _make_goodcopbadcop_cra_export(root: Path, export_run_id: str = "source_export") -> Path:
    export_root = root / "palette_analytics"
    rows = [
        {
            "recording_id": recording_id,
            "fish_id": "0",
            "endpoint_status": "computed",
            "chaser_count": 2,
            "pairwise_role_contrast_policy": "not_computed_at_recording_level",
        }
        for recording_id in ("r1", "r2", "r3")
    ]
    deltas = {
        "r1": {"aggressive": (1.0, -0.1), "inert": (0.5, 0.1)},
        "r2": {"aggressive": (2.0, -0.2), "inert": (1.0, -0.1)},
        "r3": {"aggressive": (3.0, -0.3), "inert": (1.5, 0.0)},
    }
    phase_rows: list[dict] = []
    for recording_id, role_deltas in deltas.items():
        for object_index, (role, (distance_delta, occupancy_delta)) in enumerate(
            role_deltas.items()
        ):
            for phase_axis_index, (phase_label, distance, occupancy) in enumerate(
                (
                    ("pre_static", 10.0, 0.5),
                    ("post_static", 10.0 + distance_delta, 0.5 + occupancy_delta),
                )
            ):
                phase_rows.append(
                    {
                        "recording_id": recording_id,
                        "fish_id": "0",
                        "phase_axis_index": phase_axis_index,
                        "phase_label": phase_label,
                        "object_column_index": object_index,
                        "object_index": object_index,
                        "object_role": role,
                        "behavior_class": role,
                        "median_distance_mm": distance,
                        "occupancy_fraction": occupancy,
                    }
                )
    _write_rows(
        export_root
        / "v1"
        / "chaser_quadrant_occupancy_summary"
        / f"export_run_id={export_run_id}",
        rows,
    )
    _write_rows(
        export_root
        / "v1"
        / "chaser_quadrant_occupancy_chaser_phase"
        / f"export_run_id={export_run_id}",
        phase_rows,
    )
    manifest = {
        "export_run_id": export_run_id,
        "schema_id": EXPORT_SCHEMA_ID,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "table_contracts": contract_snapshot(
            [
                "chaser_quadrant_occupancy_summary",
                "chaser_quadrant_occupancy_chaser_phase",
            ]
        ),
        "row_counts_by_table": {
            "chaser_quadrant_occupancy_summary": len(rows),
            "chaser_quadrant_occupancy_chaser_phase": len(phase_rows),
        },
        "collection_manifest": {
            "collection_id": "collection_test",
            "manifest_sha256": "abc123",
        },
    }
    manifest_dir = export_root / "v1" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"export_run_id={export_run_id}.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return export_root


def _make_goodcopbadcop_epoch_behavior_export(root: Path, export_run_id: str = "source_export") -> Path:
    export_root = root / "palette_analytics"
    rows: list[dict] = []
    values = {
        "r1": {"pre": 10.0, "training": 20.0, "post": 15.0},
        "r2": {"pre": 12.0, "training": 24.0, "post": 18.0},
        "r3": {"pre": 14.0, "training": 28.0, "post": 21.0},
    }
    label_by_condition = {
        "pre": "pre_event",
        "training": "training_event",
        "post": "post_event",
    }
    for recording_id, condition_values in values.items():
        for condition, speed in condition_values.items():
            rows.append(
                {
                    "recording_id": recording_id,
                    "window_label": label_by_condition[condition],
                    "mean_speed_mm_s": speed,
                    "bout_count": int(speed / 10.0),
                    "bout_rate_per_min": speed / 2.0,
                    "mean_bout_duration_s": speed / 1000.0,
                    "median_bout_duration_s": speed / 1100.0,
                    "mean_bout_path_length_mm": speed / 25.0,
                    "median_bout_path_length_mm": speed / 30.0,
                    "mean_bout_net_heading_change_deg": speed / 8.0,
                    "median_bout_net_heading_change_deg": speed / 9.0,
                    "mean_abs_bout_net_heading_change_deg": speed / 5.0,
                    "median_abs_bout_net_heading_change_deg": speed / 6.0,
                    "mean_bout_heading_path_deg": speed / 4.0,
                    "median_bout_heading_path_deg": speed / 4.5,
                    "inter_bout_interval_count": int(speed / 20.0),
                    "mean_inter_bout_interval_s": speed / 100.0,
                    "median_inter_bout_interval_s": speed / 120.0,
                    "mean_distance_from_arena_center_mm": speed / 3.0,
                    "median_distance_from_arena_center_mm": speed / 3.5,
                    "wall_fraction": speed / 100.0,
                    "wall_time_s": speed / 2.0,
                    "collection_id": "collection_test",
                    "collection_manifest_sha256": "abc123",
                }
            )
    _write_rows(
        export_root / "v1" / "chaser_epoch_behavior_summary" / f"export_run_id={export_run_id}",
        rows,
    )
    manifest = {
        "export_run_id": export_run_id,
        "schema_id": EXPORT_SCHEMA_ID,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "table_contracts": contract_snapshot(["chaser_epoch_behavior_summary"]),
        "row_counts_by_table": {"chaser_epoch_behavior_summary": len(rows)},
        "collection_manifest": {
            "collection_id": "collection_test",
            "manifest_sha256": "abc123",
        },
    }
    manifest_dir = export_root / "v1" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"export_run_id={export_run_id}.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return export_root


def test_paired_contrast_uses_recording_level_sign_flip() -> None:
    stats = compute_paired_contrast(
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        unit_count=3,
        minimum_recordings=3,
        bootstrap_iterations=0,
        permutation_iterations=100,
        confidence_level=0.95,
        rng=__import__("numpy").random.default_rng(0),
    )

    assert stats.status == "computed"
    assert stats.mean_difference == pytest.approx(2.0)
    assert stats.median_difference == pytest.approx(2.0)
    assert stats.std_difference == pytest.approx(1.0)
    assert stats.effect_size == pytest.approx(2.0)
    assert stats.p_value == pytest.approx(0.25)
    assert stats.test_method == "paired_sign_flip_exact"
    assert stats.permutation_iterations == 8


def test_one_sample_signed_rank_uses_exact_wilcoxon_and_rank_biserial() -> None:
    stats = compute_one_sample_signed_rank(
        [1.0, 2.0, 3.0],
        unit_count=3,
        minimum_recordings=3,
        bootstrap_iterations=0,
        confidence_level=0.95,
        rng=__import__("numpy").random.default_rng(0),
    )

    assert stats.status == "computed"
    assert stats.mean_difference == pytest.approx(2.0)
    assert stats.median_difference == pytest.approx(2.0)
    assert stats.effect_size == pytest.approx(1.0)
    assert stats.p_value == pytest.approx(0.25)
    assert stats.test_method == "wilcoxon_signed_rank_exact"


def test_benjamini_hochberg_adjusts_with_monotonicity() -> None:
    assert benjamini_hochberg([0.01, 0.04, 0.03, None]) == pytest.approx([0.03, 0.04, 0.04, None])


def test_goodcopbadcop_statistics_computes_and_writes_summary(tmp_path: Path) -> None:
    export_root = _make_goodcopbadcop_export(tmp_path)
    config = GoodCopBadCopStatisticsConfig(
        export_root=export_root,
        source_export_run_id="source_export",
        stats_run_id="stats_test",
        metrics=metric_specs_for_families(("chaser_distance",)),
        bootstrap_iterations=0,
        minimum_recordings=3,
        random_seed=0,
    )

    rows, manifest = compute_goodcopbadcop_statistics(config)
    descriptive_rows = compute_goodcopbadcop_descriptive_summaries(config)

    assert manifest["status_counts"] == {"computed": 18}
    assert manifest["input_tables"] == [
        "chaser_epoch_distance_summary",
        "chaser_quadrant_occupancy_chaser_phase",
    ]
    assert manifest["parameters"]["role_mapping_table"] == (
        "chaser_quadrant_occupancy_chaser_phase"
    )
    assert manifest["row_counts_by_table"][SUMMARY_TABLE] == 18
    assert len(descriptive_rows) == 18
    descriptive_target = next(
        row
        for row in descriptive_rows
        if row["metric_name"] == "p50_distance_mm"
        and row["condition_name"] == "training"
        and json.loads(row["group_key_json"])["behavior_class"] == "aggressive"
    )
    assert descriptive_target["mean"] == pytest.approx(13.0 / 3.0)
    assert descriptive_target["std_dev"] == pytest.approx(2.081665999)
    assert descriptive_target["sem"] == pytest.approx(2.081665999 / (3.0 ** 0.5))
    target = next(
        row
        for row in rows
        if row["metric_name"] == "p50_distance_mm"
        and row["contrast_name"] == "training-pre"
        and json.loads(row["group_key_json"])["behavior_class"] == "aggressive"
    )
    assert target["source_export_run_id"] == "source_export"
    assert target["collection_id"] == "collection_test"
    assert target["paired_unit_count"] == 3
    assert target["mean_a"] == pytest.approx(7.0 / 3.0)
    assert target["mean_b"] == pytest.approx(13.0 / 3.0)
    assert target["mean_difference"] == pytest.approx(2.0)
    assert target["p_value"] == pytest.approx(0.25)
    assert target["q_value"] is not None
    assert target["test_method"] == "paired_sign_flip_exact"

    written = write_goodcopbadcop_statistics(
        rows,
        manifest,
        export_root=export_root,
        stats_run_id="stats_test",
        descriptive_rows=descriptive_rows,
    )
    assert Path(written["manifest_path"]).is_file()
    part = export_root / "v1" / SUMMARY_TABLE / "export_run_id=stats_test" / "part-00000.parquet"
    assert part.is_file()
    table = pq.read_table(part).to_pylist()
    assert len(table) == 18
    assert all(row["export_schema_version"] == EXPORT_SCHEMA_VERSION for row in table)
    assert all(row["table_name"] == SUMMARY_TABLE for row in table)
    summary_metadata = pq.ParquetFile(part).schema_arrow.metadata or {}
    assert summary_metadata[b"palette.export_schema_id"].decode() == EXPORT_SCHEMA_ID
    assert json.loads(summary_metadata[b"palette.table_contract"]) == TABLE_CONTRACTS[
        SUMMARY_TABLE
    ].to_dict()
    descriptive_part = export_root / "v1" / DESCRIPTIVE_TABLE / "export_run_id=stats_test" / "part-00000.parquet"
    assert descriptive_part.is_file()
    descriptive_table = pq.read_table(descriptive_part).to_pylist()
    assert len(descriptive_table) == 18
    assert written["schema_version"] == EXPORT_SCHEMA_VERSION
    assert set(written["capabilities"]) == {
        "group.statistics",
        "group.descriptive_statistics",
    }


def test_goodcopbadcop_statistics_computes_cra_primary_endpoint_wilcoxon(tmp_path: Path) -> None:
    export_root = _make_goodcopbadcop_cra_export(tmp_path)
    config = GoodCopBadCopStatisticsConfig(
        export_root=export_root,
        source_export_run_id="source_export",
        stats_run_id="stats_cra",
        metrics=metric_specs_for_families(("cra_primary_endpoint",)),
        bootstrap_iterations=0,
        minimum_recordings=3,
        random_seed=0,
    )

    rows, manifest = compute_goodcopbadcop_statistics(config)

    assert manifest["status_counts"] == {"computed": 6}
    assert manifest["input_tables"] == [
        "chaser_quadrant_occupancy_chaser_phase",
        "chaser_quadrant_occupancy_summary",
    ]
    assert manifest["row_counts_by_table"][SUMMARY_TABLE] == 6
    target = next(row for row in rows if row["metric_name"] == "delta_agg")
    assert target["metric_family"] == "cra_primary_endpoint"
    assert target["source_table"] == "chaser_quadrant_occupancy_summary"
    assert target["contrast_name"] == "vs-zero"
    assert target["condition_a"] == "zero"
    assert target["condition_b"] == "observed"
    assert target["primary"] is False
    assert target["paired_unit_count"] == 3
    assert target["mean_a"] == 0.0
    assert target["mean_b"] == pytest.approx(2.0)
    assert target["mean_difference"] == pytest.approx(2.0)
    assert target["median_difference"] == pytest.approx(2.0)
    assert target["effect_size"] == pytest.approx(1.0)
    specificity = next(row for row in rows if row["metric_name"] == "specificity_distance")
    assert specificity["primary"] is True
    occupancy_specificity = next(row for row in rows if row["metric_name"] == "specificity_occupancy")
    assert occupancy_specificity["primary"] is False
    assert target["p_value"] == pytest.approx(0.25)
    assert target["test_method"] == "wilcoxon_signed_rank_exact"

    inert = next(row for row in rows if row["metric_name"] == "delta_inert")
    assert inert["primary"] is False
    assert inert["p_value"] == pytest.approx(0.25)
    assert inert["effect_size"] == pytest.approx(1.0)


def test_goodcopbadcop_statistics_computes_epoch_behavior_metrics(tmp_path: Path) -> None:
    export_root = _make_goodcopbadcop_epoch_behavior_export(tmp_path)
    config = GoodCopBadCopStatisticsConfig(
        export_root=export_root,
        source_export_run_id="source_export",
        stats_run_id="stats_epoch_behavior",
        metrics=metric_specs_for_families(("epoch_behavior",)),
        bootstrap_iterations=0,
        minimum_recordings=3,
        random_seed=0,
    )

    rows, manifest = compute_goodcopbadcop_statistics(config)
    descriptive_rows = compute_goodcopbadcop_descriptive_summaries(config)

    assert manifest["input_tables"] == ["chaser_epoch_behavior_summary"]
    assert manifest["row_counts_by_table"][SUMMARY_TABLE] == 36
    assert manifest["status_counts"] == {"computed": 36}
    assert len(descriptive_rows) == 36
    pre_ibi = next(
        row
        for row in descriptive_rows
        if row["metric_name"] == "mean_inter_bout_interval_s"
        and row["condition_name"] == "pre"
    )
    assert pre_ibi["mean"] == pytest.approx(0.12)
    assert pre_ibi["std_dev"] == pytest.approx(0.02)
    target = next(
        row
        for row in rows
        if row["metric_name"] == "mean_inter_bout_interval_s"
        and row["contrast_name"] == "post-pre"
    )
    assert target["metric_family"] == "epoch_behavior"
    assert target["source_table"] == "chaser_epoch_behavior_summary"
    assert target["exploratory"] is True
    assert target["paired_unit_count"] == 3
    assert target["mean_a"] == pytest.approx(0.12)
    assert target["mean_b"] == pytest.approx(0.18)
    assert target["mean_difference"] == pytest.approx(0.06)


def test_compute_group_statistics_cli_dry_run_and_apply(tmp_path: Path, capsys) -> None:
    export_root = _make_goodcopbadcop_export(tmp_path)
    args = [
        "--profile",
        "chaser",
        "--source-export-run-id",
        "source_export",
        "--export-root",
        str(export_root),
        "--stats-run-id",
        "stats_cli",
        "--metrics",
        "chaser_distance",
        "--bootstrap-iterations",
        "0",
        "--minimum-recordings",
        "3",
    ]

    assert compute_group_statistics_main(args) == 0
    dry_output = capsys.readouterr().out
    assert "dry_run\ttrue" in dry_output
    assert f"rows\t{DESCRIPTIVE_TABLE}\t18" in dry_output
    assert not (export_root / "v1" / SUMMARY_TABLE / "export_run_id=stats_cli").exists()

    assert compute_group_statistics_main([*args, "--apply"]) == 0
    apply_output = capsys.readouterr().out
    assert "manifest\t" in apply_output
    assert (export_root / "v1" / SUMMARY_TABLE / "export_run_id=stats_cli" / "part-00000.parquet").is_file()
    assert (export_root / "v1" / DESCRIPTIVE_TABLE / "export_run_id=stats_cli" / "part-00000.parquet").is_file()
