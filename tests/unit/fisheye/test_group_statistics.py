from __future__ import annotations

import json
import concurrent.futures
import hashlib
import os
from pathlib import Path
import threading

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fisheye.analytics_exports.arrow_contracts import (
    ARROW_TABLE_CONTRACTS,
    exact_arrow_schema,
)
from fisheye.analytics_exports.contracts import (
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
    TABLE_CONTRACTS,
    contract_snapshot,
)
from fisheye.analytics_exports.publication import (
    PUBLICATION_SCHEMA_ID,
    PUBLICATION_SCHEMA_VERSION,
    export_manifest_path,
    generation_relative_path,
    manifest_selected_part_files,
    publication_generation_root,
    sha256_file,
)
from fisheye.analytics_exports.validation import (
    ExportValidationError,
    validate_export_run,
)
from fisheye.group_analytics_viewer.query import (
    ViewerContext,
    query_group_statistics,
    resolve_statistics_run_id,
)
from fisheye.group_statistics.goodcopbadcop import (
    DEFAULT_METRICS,
    DESCRIPTIVE_TABLE,
    GoodCopBadCopStatisticsConfig,
    MetricSpec,
    SUMMARY_TABLE,
    compute_goodcopbadcop_outputs,
    compute_goodcopbadcop_descriptive_summaries,
    compute_goodcopbadcop_statistics,
    metric_specs_for_families,
    write_goodcopbadcop_statistics,
    _descriptive_result_id,
    _result_id,
)
from fisheye.group_statistics.legacy_arrow import (
    legacy_group_statistics_arrow_envelope,
    legacy_group_statistics_contract_snapshot,
    validate_legacy_group_statistics_payload,
)
from fisheye.group_statistics.paired import (
    benjamini_hochberg,
    compute_one_sample_signed_rank,
    compute_paired_contrast,
)
from fisheye.group_statistics.acquisition_batch_cluster import (
    fit_acquisition_batch_random_intercept,
)
from fisheye.utils.compute_group_statistics import main as compute_group_statistics_main


_DEFAULT_ACQUISITION_BATCH = object()


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path / "part-00000.parquet")


def _registry_identity(
    recording_id: str,
    *,
    acquisition_batch_id: str | None | object = _DEFAULT_ACQUISITION_BATCH,
) -> dict[str, str | None]:
    resolved_batch = (
        f"batch-{recording_id}"
        if acquisition_batch_id is _DEFAULT_ACQUISITION_BATCH
        else acquisition_batch_id
    )
    return {
        "acquisition_batch_id": resolved_batch,
        "subject_id": f"subject-{recording_id}",
    }


def _make_goodcopbadcop_export(
    root: Path,
    export_run_id: str = "source_export",
    *,
    values_by_recording: dict[str, dict[str, float]] | None = None,
    acquisition_batch_by_recording: dict[str, str | None] | None = None,
    roles_by_recording: dict[str, tuple[str, str]] | None = None,
) -> Path:
    export_root = root / "palette_analytics"
    rows: list[dict] = []
    values = values_by_recording or {
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
                        **_registry_identity(
                            recording_id,
                            acquisition_batch_id=(
                                acquisition_batch_by_recording.get(recording_id)
                                if acquisition_batch_by_recording is not None
                                else _DEFAULT_ACQUISITION_BATCH
                            ),
                        ),
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
        export_root
        / "v1"
        / "chaser_epoch_distance_summary"
        / f"export_run_id={export_run_id}",
        rows,
    )
    role_by_recording = roles_by_recording or {
        "r1": ("aggressive", "inert"),
        "r2": ("inert", "aggressive"),
        "r3": ("aggressive", "inert"),
    }
    object_phase_rows = [
        {
            "recording_id": recording_id,
            **_registry_identity(
                recording_id,
                acquisition_batch_id=(
                    acquisition_batch_by_recording.get(recording_id)
                    if acquisition_batch_by_recording is not None
                    else _DEFAULT_ACQUISITION_BATCH
                ),
            ),
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


def _publish_legacy_group_statistics_copy(
    export_root: Path,
    exact_payload: dict[str, object],
    *,
    legacy_run_id: str,
) -> dict[str, object]:
    """Create a test-only copy of the historical inferred-v2 publication."""

    generation_id = "legacy-inferred-v2"
    generation_root = publication_generation_root(
        export_root,
        legacy_run_id,
        generation_id,
    )
    generation_relative = generation_relative_path(
        legacy_run_id,
        generation_id,
    )
    output_tables = [str(value) for value in exact_payload["output_tables"]]
    contracts = legacy_group_statistics_contract_snapshot(output_tables)
    part_files: dict[str, list[str]] = {}
    row_counts: dict[str, int] = {}
    inventory: dict[str, list[dict[str, object]]] = {}
    removed_by_table = {
        SUMMARY_TABLE: {"metric_unit", "effect_size_kind", "ci_estimand"},
        DESCRIPTIVE_TABLE: {"metric_unit"},
    }

    for table_name in output_tables:
        source_parts = manifest_selected_part_files(
            export_root,
            str(exact_payload["export_run_id"]),
            table_name,
        )
        rows: list[dict[str, object]] = []
        for source_part in source_parts:
            for source_row in pq.read_table(source_part).to_pylist():
                row = {
                    key: value
                    for key, value in source_row.items()
                    if key not in removed_by_table[table_name]
                }
                row["stats_run_id"] = legacy_run_id
                rows.append(row)
        table = pa.Table.from_pylist(rows)
        metadata = {
            b"palette.export_schema_id": EXPORT_SCHEMA_ID.encode("utf-8"),
            b"palette.export_schema_version": str(EXPORT_SCHEMA_VERSION).encode(
                "ascii"
            ),
            b"palette.table_contract": json.dumps(
                contracts[table_name],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
            b"palette.arrow_schema_mode": b"inferred_v2_compatibility",
        }
        table = table.replace_schema_metadata(metadata)
        part_path = generation_root / "tables" / table_name / "part-00000.parquet"
        part_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, part_path)
        relative_path = (
            generation_relative / "tables" / table_name / part_path.name
        ).as_posix()
        part_files[table_name] = [relative_path]
        row_counts[table_name] = len(rows)
        inventory[table_name] = [
            {
                "path": relative_path,
                "sha256": sha256_file(part_path),
                "size_bytes": part_path.stat().st_size,
                "row_count": len(rows),
            }
        ]

    legacy_payload = json.loads(json.dumps(exact_payload))
    legacy_payload.update(
        {
            "export_run_id": legacy_run_id,
            "manifest_path": str(export_manifest_path(export_root, legacy_run_id)),
            "table_contracts": contracts,
            "arrow_schema_contracts": legacy_group_statistics_arrow_envelope(
                output_tables
            ),
            "row_counts_by_table": row_counts,
            "part_files_by_table": part_files,
            "publication": {
                "schema_id": PUBLICATION_SCHEMA_ID,
                "schema_version": PUBLICATION_SCHEMA_VERSION,
                "state": "complete",
                "generation_id": generation_id,
                "generation_path": generation_relative.as_posix(),
                "parts_by_table": inventory,
            },
        }
    )
    manifest_path = export_manifest_path(export_root, legacy_run_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(legacy_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return legacy_payload


def _make_goodcopbadcop_cra_export(
    root: Path,
    export_run_id: str = "source_export",
    *,
    deltas_by_recording: dict[str, dict[str, tuple[float, float]]] | None = None,
    acquisition_batch_by_recording: dict[str, str] | None = None,
) -> Path:
    export_root = root / "palette_analytics"
    rows = [
        {
            "recording_id": recording_id,
            **_registry_identity(
                recording_id,
                acquisition_batch_id=(acquisition_batch_by_recording or {}).get(
                    recording_id
                ),
            ),
            "fish_id": "0",
            "endpoint_status": "computed",
            "chaser_count": 2,
            "pairwise_role_contrast_policy": "not_computed_at_recording_level",
        }
        for recording_id in (
            (deltas_by_recording or {}).keys()
            if deltas_by_recording is not None
            else ("r1", "r2", "r3")
        )
    ]
    deltas = deltas_by_recording or {
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
                        **_registry_identity(
                            recording_id,
                            acquisition_batch_id=(
                                acquisition_batch_by_recording or {}
                            ).get(recording_id),
                        ),
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


def _make_goodcopbadcop_epoch_behavior_export(
    root: Path, export_run_id: str = "source_export"
) -> Path:
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
                    **_registry_identity(recording_id),
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
        export_root
        / "v1"
        / "chaser_epoch_behavior_summary"
        / f"export_run_id={export_run_id}",
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


def _repeated_acquisition_batch_design() -> tuple[
    dict[str, str],
    dict[str, dict[str, float]],
    dict[str, tuple[str, str]],
    dict[str, dict[str, tuple[float, float]]],
]:
    batches: dict[str, str] = {}
    paired_values: dict[str, dict[str, float]] = {}
    roles: dict[str, tuple[str, str]] = {}
    cra_deltas: dict[str, dict[str, tuple[float, float]]] = {}
    for batch_index in range(1, 6):
        for replicate_index in range(2):
            recording_id = f"s{batch_index}-r{replicate_index + 1}"
            batches[recording_id] = f"session-{batch_index}"
            baseline = 10.0 + float(batch_index) + 0.2 * replicate_index
            training_delta = 0.6 * batch_index + 0.1 * replicate_index
            paired_values[recording_id] = {
                "pre": baseline,
                "training": baseline + training_delta,
                "post": baseline + 0.45 * training_delta,
            }
            roles[recording_id] = ("aggressive", "inert")
            cra_deltas[recording_id] = {
                "aggressive": (
                    0.5 + 0.3 * batch_index + 0.05 * replicate_index,
                    -0.02 * batch_index - 0.005 * replicate_index,
                ),
                "inert": (
                    0.1 + 0.1 * batch_index + 0.02 * replicate_index,
                    0.01 * batch_index + 0.002 * replicate_index,
                ),
            }
    return batches, paired_values, roles, cra_deltas


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
    assert benjamini_hochberg([0.01, 0.04, 0.03, None]) == pytest.approx(
        [0.03, 0.04, 0.04, None]
    )


def test_acquisition_batch_random_intercept_reports_clustered_inference_and_icc() -> (
    None
):
    result = fit_acquisition_batch_random_intercept(
        [1.0, 1.2, 2.0, 2.2, 3.0, 3.2, 4.0, 4.2],
        ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
        confidence_level=0.95,
        minimum_acquisition_batches=4,
    )

    assert result.status == "computed"
    assert result.cluster_count == 4
    assert result.unit_count == 8
    assert result.mean == pytest.approx(2.6)
    assert result.standard_error is not None
    assert result.p_value is not None
    assert result.intraclass_correlation is not None
    assert 0.0 <= result.intraclass_correlation <= 1.0


def test_acquisition_batch_random_intercept_requires_repeated_session_observations() -> (
    None
):
    result = fit_acquisition_batch_random_intercept(
        [1.0, 2.0, 3.0],
        ["s1", "s2", "s3"],
        confidence_level=0.95,
        minimum_acquisition_batches=3,
    )

    assert result.status == "unavailable"
    assert result.reason == "no_repeated_acquisition_batch_observations"
    assert result.cluster_count == result.unit_count == 3


def test_default_exploratory_fdr_families_have_multiple_registered_metrics() -> None:
    family_counts: dict[str, int] = {}
    for spec in DEFAULT_METRICS:
        if spec.exploratory:
            family_counts[spec.metric_family] = (
                family_counts.get(spec.metric_family, 0) + 1
            )

    assert family_counts
    assert all(count > 1 for count in family_counts.values())


def test_group_statistics_rejects_zero_minimum_recordings(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="minimum_recordings must be an integer >= 1"):
        GoodCopBadCopStatisticsConfig(
            export_root=tmp_path,
            source_export_run_id="source_export",
            stats_run_id="stats_invalid",
            minimum_recordings=0,
        )


def test_group_statistics_requires_explicit_metric_tier() -> None:
    with pytest.raises(TypeError, match="primary"):
        MetricSpec(
            metric_family="new_family",
            source_table="new_table",
            metric_name="new_metric",
            group_keys=(),
        )

    with pytest.raises(ValueError, match="exactly one"):
        GoodCopBadCopStatisticsConfig(
            export_root=Path("/tmp/unused"),
            source_export_run_id="source_export",
            stats_run_id="stats_invalid_tier",
            metrics=(
                MetricSpec(
                    metric_family="new_family",
                    source_table="new_table",
                    metric_name="new_metric",
                    group_keys=(),
                    primary=True,
                    exploratory=True,
                ),
            ),
        )


def test_group_statistics_rejects_nonconservative_minimum_acquisition_batches(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError, match="minimum_acquisition_batches must be an integer >= 3"
    ):
        GoodCopBadCopStatisticsConfig(
            export_root=tmp_path,
            source_export_run_id="source_export",
            stats_run_id="stats_invalid_sessions",
            minimum_acquisition_batches=2,
        )


def test_subject_level_statistics_accept_explicit_missing_acquisition_batches(
    tmp_path: Path,
) -> None:
    export_root = _make_goodcopbadcop_export(
        tmp_path,
        acquisition_batch_by_recording={"r1": None, "r2": None, "r3": None},
    )
    rows, manifest = compute_goodcopbadcop_statistics(
        GoodCopBadCopStatisticsConfig(
            export_root=export_root,
            source_export_run_id="source_export",
            stats_run_id="stats_subject_level",
            metrics=metric_specs_for_families(("chaser_distance",)),
            bootstrap_iterations=0,
            minimum_recordings=3,
            allow_legacy_export_layout=True,
        )
    )

    assert rows
    assert all(row["status"] == "computed" for row in rows)
    assert all(row["cluster_mode"] == "none" for row in rows)
    assert all(row["cluster_status"] == "disabled" for row in rows)
    assert manifest["parameters"]["cluster"] == "none"


def test_requested_batch_adjustment_rejects_partial_batch_identity(
    tmp_path: Path,
) -> None:
    export_root = _make_goodcopbadcop_export(
        tmp_path,
        acquisition_batch_by_recording={
            "r1": "batch-a",
            "r2": None,
            "r3": "batch-b",
        },
    )
    rows, _manifest = compute_goodcopbadcop_statistics(
        GoodCopBadCopStatisticsConfig(
            export_root=export_root,
            source_export_run_id="source_export",
            stats_run_id="stats_missing_batch",
            metrics=metric_specs_for_families(("chaser_distance",)),
            bootstrap_iterations=0,
            minimum_recordings=3,
            minimum_acquisition_batches=3,
            cluster="acquisition_batch",
            allow_legacy_export_layout=True,
        )
    )

    assert rows
    assert all(row["status"] == "computed" for row in rows)
    assert all(row["cluster_status"] == "unavailable" for row in rows)
    assert all(
        row["cluster_reason"] == "missing_acquisition_batch_identity" for row in rows
    )
    assert all(row["clustered_p_value"] is None for row in rows)


def test_goodcopbadcop_statistics_computes_and_writes_summary(tmp_path: Path) -> None:
    export_root = _make_goodcopbadcop_export(tmp_path)
    config = GoodCopBadCopStatisticsConfig(
        export_root=export_root,
        source_export_run_id="source_export",
        stats_run_id="stats_test",
        metrics=metric_specs_for_families(("chaser_distance",)),
        bootstrap_iterations=0,
        minimum_recordings=3,
        cluster="acquisition_batch",
        random_seed=0,
        allow_legacy_export_layout=True,
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
    assert manifest["parameters"]["allow_legacy_export_layout"] is True
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
    assert descriptive_target["sem"] == pytest.approx(2.081665999 / (3.0**0.5))
    target = next(
        row
        for row in rows
        if row["metric_name"] == "p50_distance_mm"
        and row["contrast_name"] == "training-pre"
        and json.loads(row["group_key_json"])["behavior_class"] == "aggressive"
    )
    assert target["source_export_run_id"] == "source_export"
    assert target["collection_id"] == "collection_test"
    assert target["metric_unit"] == "mm"
    assert target["effect_size_kind"] == ("paired_mean_difference_over_sample_sd")
    assert target["ci_estimand"] == "paired_mean_difference"
    assert target["paired_unit_count"] == 3
    assert target["mean_a"] == pytest.approx(7.0 / 3.0)
    assert target["mean_b"] == pytest.approx(13.0 / 3.0)
    assert target["mean_difference"] == pytest.approx(2.0)
    assert target["p_value"] == pytest.approx(0.25)
    assert target["q_value"] is not None
    assert target["multiple_comparison_family"] == "primary|chaser_distance"
    assert target["cluster_mode"] == "acquisition_batch"
    assert target["cluster_method"] == "acquisition_batch_random_intercept_reml_v1"
    assert target["cluster_count"] == 3
    assert target["cluster_status"] == "unavailable"
    assert target["cluster_reason"] == "acquisition_batch_count<10"
    assert target["clustered_p_value"] is None
    assert target["clustered_q_value"] is None
    assert target["intraclass_correlation"] is None
    assert target["test_method"] == "paired_sign_flip_exact"
    assert json.loads(target["parameters_json"])["allow_legacy_export_layout"] is True

    written = write_goodcopbadcop_statistics(
        rows,
        manifest,
        export_root=export_root,
        stats_run_id="stats_test",
        descriptive_rows=descriptive_rows,
    )
    assert Path(written["manifest_path"]).is_file()
    part = manifest_selected_part_files(export_root, "stats_test", SUMMARY_TABLE)[0]
    assert part.is_file()
    table = pq.read_table(part).to_pylist()
    assert len(table) == 18
    assert all(row["export_schema_version"] == EXPORT_SCHEMA_VERSION for row in table)
    assert all(row["table_name"] == SUMMARY_TABLE for row in table)
    summary_metadata = pq.ParquetFile(part).schema_arrow.metadata or {}
    assert summary_metadata[b"palette.export_schema_id"].decode() == EXPORT_SCHEMA_ID
    assert summary_metadata[b"palette.arrow_schema_mode"] == b"exact"
    assert (
        pq.ParquetFile(part).schema_arrow.remove_metadata()
        == exact_arrow_schema(
            SUMMARY_TABLE,
            metadata={},
        ).remove_metadata()
    )
    assert (
        json.loads(summary_metadata[b"palette.table_contract"])
        == TABLE_CONTRACTS[SUMMARY_TABLE].to_dict()
    )
    descriptive_part = manifest_selected_part_files(
        export_root,
        "stats_test",
        DESCRIPTIVE_TABLE,
    )[0]
    assert descriptive_part.is_file()
    descriptive_table = pq.read_table(descriptive_part).to_pylist()
    assert len(descriptive_table) == 18
    descriptive_schema = pq.ParquetFile(descriptive_part).schema_arrow
    assert (
        descriptive_schema.remove_metadata()
        == exact_arrow_schema(
            DESCRIPTIVE_TABLE,
            metadata={},
        ).remove_metadata()
    )
    assert descriptive_schema.metadata[b"palette.arrow_schema_mode"] == b"exact"
    assert written["arrow_schema_contracts"]["inferred_v2_compatibility_tables"] == []
    assert written["schema_version"] == EXPORT_SCHEMA_VERSION
    assert set(written["capabilities"]) == {
        "group.statistics",
        "group.descriptive_statistics",
    }


def test_paired_acquisition_batch_cluster_inference_survives_arrow_publication_and_readback(
    tmp_path: Path,
) -> None:
    batches, paired_values, roles, _cra_deltas = _repeated_acquisition_batch_design()
    export_root = _make_goodcopbadcop_export(
        tmp_path,
        values_by_recording=paired_values,
        acquisition_batch_by_recording=batches,
        roles_by_recording=roles,
    )
    config = GoodCopBadCopStatisticsConfig(
        export_root=export_root,
        source_export_run_id="source_export",
        stats_run_id="stats_repeated_paired",
        metrics=metric_specs_for_families(("chaser_distance",)),
        bootstrap_iterations=0,
        minimum_recordings=10,
        minimum_acquisition_batches=5,
        cluster="acquisition_batch",
        random_seed=0,
        allow_legacy_export_layout=True,
    )

    rows, descriptive_rows, manifest = compute_goodcopbadcop_outputs(config)
    assert rows
    assert all(row["status"] == "computed" for row in rows)
    assert all(
        row["cluster_status"] in {"computed", "boundary_zero_variance"} for row in rows
    )
    assert all(row["cluster_count"] == 5 for row in rows)
    assert all(row["clustered_p_value"] is not None for row in rows)
    assert all(row["clustered_q_value"] is not None for row in rows)
    assert manifest["parameters"]["minimum_acquisition_batches"] == 5
    assert manifest["fdr_families"] == [
        {
            "family_id": "primary|chaser_distance",
            "analysis_tier": "primary",
            "metric_family": "chaser_distance",
            "result_count": len(rows),
            "naive_test_count": len(rows),
            "naive_test_status": "multiple_tests",
            "clustered_test_count": len(rows),
            "clustered_test_status": "multiple_tests",
        }
    ]

    write_goodcopbadcop_statistics(
        rows,
        manifest,
        export_root=export_root,
        stats_run_id="stats_repeated_paired",
        descriptive_rows=descriptive_rows,
    )
    assert (
        validate_export_run(
            export_root,
            "stats_repeated_paired",
        )["status"]
        == "valid"
    )
    persisted = {
        row["stat_result_id"]: row
        for row in pq.read_table(
            manifest_selected_part_files(
                export_root,
                "stats_repeated_paired",
                SUMMARY_TABLE,
            )[0]
        ).to_pylist()
    }
    assert set(persisted) == {row["stat_result_id"] for row in rows}
    for row in rows:
        actual = persisted[row["stat_result_id"]]
        assert actual["clustered_p_value"] == pytest.approx(row["clustered_p_value"])
        assert actual["clustered_q_value"] == pytest.approx(row["clustered_q_value"])
        assert actual["intraclass_correlation"] == pytest.approx(
            row["intraclass_correlation"]
        )


def test_one_sample_acquisition_batch_cluster_inference_survives_arrow_publication_and_readback(
    tmp_path: Path,
) -> None:
    batches, _paired_values, _roles, cra_deltas = _repeated_acquisition_batch_design()
    export_root = _make_goodcopbadcop_cra_export(
        tmp_path,
        deltas_by_recording=cra_deltas,
        acquisition_batch_by_recording=batches,
    )
    metrics = tuple(
        spec
        for spec in metric_specs_for_families(("cra_primary_endpoint",))
        if spec.metric_name in {"delta_agg", "delta_inert"}
    )
    config = GoodCopBadCopStatisticsConfig(
        export_root=export_root,
        source_export_run_id="source_export",
        stats_run_id="stats_repeated_one_sample",
        metrics=metrics,
        bootstrap_iterations=0,
        minimum_recordings=10,
        minimum_acquisition_batches=5,
        cluster="acquisition_batch",
        random_seed=0,
        allow_legacy_export_layout=True,
    )

    rows, descriptive_rows, manifest = compute_goodcopbadcop_outputs(config)
    assert len(rows) == 2
    assert all(row["contrast_name"] == "vs-zero" for row in rows)
    assert all(row["status"] == "computed" for row in rows)
    assert all(
        row["cluster_status"] in {"computed", "boundary_zero_variance"} for row in rows
    )
    assert all(row["cluster_count"] == 5 for row in rows)
    assert all(row["clustered_q_value"] is not None for row in rows)
    assert manifest["fdr_families"] == [
        {
            "family_id": "exploratory|cra_primary_endpoint",
            "analysis_tier": "exploratory",
            "metric_family": "cra_primary_endpoint",
            "result_count": 2,
            "naive_test_count": 2,
            "naive_test_status": "multiple_tests",
            "clustered_test_count": 2,
            "clustered_test_status": "multiple_tests",
        }
    ]

    write_goodcopbadcop_statistics(
        rows,
        manifest,
        export_root=export_root,
        stats_run_id="stats_repeated_one_sample",
        descriptive_rows=descriptive_rows,
    )
    assert (
        validate_export_run(
            export_root,
            "stats_repeated_one_sample",
        )["status"]
        == "valid"
    )
    persisted_rows = pq.read_table(
        manifest_selected_part_files(
            export_root,
            "stats_repeated_one_sample",
            SUMMARY_TABLE,
        )[0]
    ).to_pylist()
    assert [row["stat_result_id"] for row in persisted_rows] == [
        row["stat_result_id"] for row in rows
    ]
    assert [row["clustered_q_value"] for row in persisted_rows] == pytest.approx(
        [row["clustered_q_value"] for row in rows]
    )


def test_singleton_exploratory_family_persists_explicit_test_count_status(
    tmp_path: Path,
) -> None:
    export_root = _make_goodcopbadcop_cra_export(tmp_path)
    metric = next(
        spec
        for spec in metric_specs_for_families(("cra_primary_endpoint",))
        if spec.metric_name == "delta_agg"
    )
    rows, manifest = compute_goodcopbadcop_statistics(
        GoodCopBadCopStatisticsConfig(
            export_root=export_root,
            source_export_run_id="source_export",
            stats_run_id="stats_singleton_family",
            metrics=(metric,),
            bootstrap_iterations=0,
            minimum_recordings=3,
            allow_legacy_export_layout=True,
        )
    )

    assert len(rows) == 1
    assert manifest["fdr_families"] == [
        {
            "family_id": "exploratory|cra_primary_endpoint",
            "analysis_tier": "exploratory",
            "metric_family": "cra_primary_endpoint",
            "result_count": 1,
            "naive_test_count": 1,
            "naive_test_status": "singleton_test",
            "clustered_test_count": 0,
            "clustered_test_status": "no_eligible_tests",
        }
    ]
    written = write_goodcopbadcop_statistics(
        rows,
        manifest,
        export_root=export_root,
        stats_run_id="stats_singleton_family",
    )
    persisted_manifest = json.loads(
        Path(written["manifest_path"]).read_text(encoding="utf-8")
    )
    assert persisted_manifest["fdr_families"] == manifest["fdr_families"]


def test_group_statistics_legacy_reader_is_explicit_and_exact_is_preferred(
    tmp_path: Path,
) -> None:
    export_root = _make_goodcopbadcop_export(tmp_path)
    config = GoodCopBadCopStatisticsConfig(
        export_root=export_root,
        source_export_run_id="source_export",
        stats_run_id="stats_exact",
        metrics=metric_specs_for_families(("chaser_distance",)),
        bootstrap_iterations=0,
        minimum_recordings=3,
        random_seed=0,
        allow_legacy_export_layout=True,
    )
    rows, descriptive_rows, manifest = compute_goodcopbadcop_outputs(config)
    exact_payload = write_goodcopbadcop_statistics(
        rows,
        manifest,
        export_root=export_root,
        stats_run_id="stats_exact",
        descriptive_rows=descriptive_rows,
    )
    exact_context = ViewerContext(
        export_root=export_root,
        export_run_id="source_export",
        stats_run_id="stats_exact",
    )
    projected = query_group_statistics(
        exact_context,
        metric_name="p50_distance_mm",
    )
    assert projected["rows"]
    assert {
        "metric_unit",
        "effect_size_kind",
        "ci_estimand",
        "missing_policy",
        "parameters_json",
    } <= set(projected["rows"][0])
    assert projected["rows"][0]["metric_unit"] == "mm"
    assert projected["rows"][0]["effect_size_kind"] == (
        "paired_mean_difference_over_sample_sd"
    )
    assert projected["rows"][0]["ci_estimand"] == "paired_mean_difference"
    legacy_payload = _publish_legacy_group_statistics_copy(
        export_root,
        exact_payload,
        legacy_run_id="stats_legacy",
    )
    legacy_payload["created_at_utc"] = "9999-12-31T23:59:59+00:00"
    export_manifest_path(export_root, "stats_legacy").write_text(
        json.dumps(legacy_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validate_legacy_group_statistics_payload(
        export_root,
        "stats_legacy",
        legacy_payload,
    )
    strict_context = ViewerContext(
        export_root=export_root,
        export_run_id="source_export",
        stats_run_id="stats_legacy",
    )
    with pytest.raises(ExportValidationError):
        resolve_statistics_run_id(strict_context)
    opted_in = ViewerContext(
        export_root=export_root,
        export_run_id="source_export",
        stats_run_id="stats_legacy",
        allow_legacy_statistics=True,
    )
    assert resolve_statistics_run_id(opted_in) == "stats_legacy"
    auto = ViewerContext(
        export_root=export_root,
        export_run_id="source_export",
        allow_legacy_statistics=True,
    )
    assert resolve_statistics_run_id(auto) == "stats_exact"

    legacy_payload["arrow_schema_contracts"] = {"tampered": True}
    export_manifest_path(export_root, "stats_legacy").write_text(
        json.dumps(legacy_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="neither the current exact contract"):
        resolve_statistics_run_id(opted_in)


def test_group_statistics_exact_writer_rejects_row_shape_and_identity_tampering(
    tmp_path: Path,
) -> None:
    export_root = _make_goodcopbadcop_export(tmp_path)
    config = GoodCopBadCopStatisticsConfig(
        export_root=export_root,
        source_export_run_id="source_export",
        stats_run_id="stats_shape",
        metrics=metric_specs_for_families(("chaser_distance",)),
        bootstrap_iterations=0,
        minimum_recordings=3,
        random_seed=0,
        allow_legacy_export_layout=True,
    )
    rows, descriptive_rows, manifest = compute_goodcopbadcop_outputs(config)

    unexpected = [dict(row) for row in rows]
    unexpected[0]["surprise"] = 1
    with pytest.raises(ValueError, match="unexpected fields"):
        write_goodcopbadcop_statistics(
            unexpected,
            manifest,
            export_root=export_root,
            stats_run_id="stats_shape",
            descriptive_rows=descriptive_rows,
        )

    missing = [dict(row) for row in rows]
    del missing[0]["created_at_utc"]
    with pytest.raises(ValueError, match="null/missing non-nullable"):
        write_goodcopbadcop_statistics(
            missing,
            manifest,
            export_root=export_root,
            stats_run_id="stats_shape",
            descriptive_rows=descriptive_rows,
        )

    bad_identity = [dict(row) for row in rows]
    bad_identity[0]["source_export_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="invalid source_export_manifest_sha256"):
        write_goodcopbadcop_statistics(
            bad_identity,
            manifest,
            export_root=export_root,
            stats_run_id="stats_shape",
            descriptive_rows=descriptive_rows,
        )

    with pytest.raises(ValueError, match="duplicate stat_result_id"):
        write_goodcopbadcop_statistics(
            [*rows, dict(rows[0])],
            manifest,
            export_root=export_root,
            stats_run_id="stats_shape",
            descriptive_rows=descriptive_rows,
        )

    resigned = [dict(row) for row in rows]
    resigned[0]["metric_family"] = "tampered_family"
    resigned[0]["source_export_manifest_path"] = "/wrong/source.json"
    resigned[0]["stat_result_id"] = _result_id(resigned[0])
    with pytest.raises(ValueError, match="invalid metric_family"):
        write_goodcopbadcop_statistics(
            resigned,
            manifest,
            export_root=export_root,
            stats_run_id="stats_shape",
            descriptive_rows=descriptive_rows,
        )

    computed_with_skipped_method = [dict(row) for row in rows]
    computed_with_skipped_method[0]["test_method"] = "skipped"
    computed_with_skipped_method[0]["stat_result_id"] = _result_id(
        computed_with_skipped_method[0]
    )
    with pytest.raises(ValueError, match="inconsistent skipped method"):
        write_goodcopbadcop_statistics(
            computed_with_skipped_method,
            manifest,
            export_root=export_root,
            stats_run_id="stats_shape",
            descriptive_rows=descriptive_rows,
        )

    for invalid_q_value in (None, 0.123456789):
        invalid_fdr = [dict(row) for row in rows]
        invalid_fdr[0]["q_value"] = invalid_q_value
        with pytest.raises(ValueError, match="invalid FDR q_value"):
            write_goodcopbadcop_statistics(
                invalid_fdr,
                manifest,
                export_root=export_root,
                stats_run_id="stats_shape",
                descriptive_rows=descriptive_rows,
            )

    reversed_ci = [dict(row) for row in rows]
    reversed_ci[0]["ci_low"] = 2.0
    reversed_ci[0]["ci_high"] = 1.0
    with pytest.raises(ValueError, match="invalid CI bounds"):
        write_goodcopbadcop_statistics(
            reversed_ci,
            manifest,
            export_root=export_root,
            stats_run_id="stats_shape",
            descriptive_rows=descriptive_rows,
        )

    bad_descriptive_unit = [dict(row) for row in descriptive_rows]
    bad_descriptive_unit[0]["unit"] = "fish"
    with pytest.raises(ValueError, match="invalid unit"):
        write_goodcopbadcop_statistics(
            rows,
            manifest,
            export_root=export_root,
            stats_run_id="stats_shape",
            descriptive_rows=bad_descriptive_unit,
        )

    absent_descriptive_values = [dict(row) for row in descriptive_rows]
    for field_name in ("sum", "mean", "median", "min", "max"):
        absent_descriptive_values[0][field_name] = None
    absent_descriptive_values[0]["descriptive_result_id"] = _descriptive_result_id(
        absent_descriptive_values[0]
    )
    with pytest.raises(ValueError, match="lacks descriptive location values"):
        write_goodcopbadcop_statistics(
            rows,
            manifest,
            export_root=export_root,
            stats_run_id="stats_shape",
            descriptive_rows=absent_descriptive_values,
        )

    absent_dispersion = [dict(row) for row in descriptive_rows]
    assert absent_dispersion[0]["unit_count"] >= 2
    absent_dispersion[0]["std_dev"] = None
    absent_dispersion[0]["sem"] = None
    absent_dispersion[0]["descriptive_result_id"] = _descriptive_result_id(
        absent_dispersion[0]
    )
    with pytest.raises(ValueError, match="lacks descriptive dispersion values"):
        write_goodcopbadcop_statistics(
            rows,
            manifest,
            export_root=export_root,
            stats_run_id="stats_shape",
            descriptive_rows=absent_dispersion,
        )

    wrong_fdr_manifest = json.loads(json.dumps(manifest))
    wrong_fdr_manifest["parameters"]["fdr_method"] = "bonferroni"
    with pytest.raises(ValueError, match="FDR method is invalid"):
        write_goodcopbadcop_statistics(
            rows,
            wrong_fdr_manifest,
            export_root=export_root,
            stats_run_id="stats_shape",
            descriptive_rows=descriptive_rows,
        )


def test_group_statistics_all_null_inference_columns_keep_exact_physical_types(
    tmp_path: Path,
) -> None:
    export_root = _make_goodcopbadcop_export(tmp_path)
    config = GoodCopBadCopStatisticsConfig(
        export_root=export_root,
        source_export_run_id="source_export",
        stats_run_id="stats_skipped",
        metrics=metric_specs_for_families(("chaser_distance",)),
        bootstrap_iterations=0,
        minimum_recordings=100,
        cluster="acquisition_batch",
        random_seed=0,
        allow_legacy_export_layout=True,
    )
    rows, descriptive_rows, manifest = compute_goodcopbadcop_outputs(config)
    assert rows
    assert all(row["status"] == "skipped" for row in rows)
    assert all(row["q_value"] is None for row in rows)
    assert all(row["effect_size"] is None for row in rows)
    assert all(row["cluster_status"] == "unavailable" for row in rows)
    assert all(
        str(row["cluster_reason"]).startswith("naive_inference_ineligible:")
        for row in rows
    )
    assert all(row["clustered_p_value"] is None for row in rows)
    assert all(row["clustered_q_value"] is None for row in rows)

    tampered_skipped = [dict(row) for row in rows]
    tampered_skipped[0]["effect_size"] = 1.0
    tampered_skipped[0]["stat_result_id"] = _result_id(tampered_skipped[0])
    with pytest.raises(ValueError, match="inferential values for a low-count skip"):
        write_goodcopbadcop_statistics(
            tampered_skipped,
            manifest,
            export_root=export_root,
            stats_run_id="stats_skipped",
            descriptive_rows=descriptive_rows,
        )

    payload = write_goodcopbadcop_statistics(
        rows,
        manifest,
        export_root=export_root,
        stats_run_id="stats_skipped",
        descriptive_rows=descriptive_rows,
    )
    summary_part = manifest_selected_part_files(
        export_root,
        "stats_skipped",
        SUMMARY_TABLE,
    )[0]
    schema = pq.ParquetFile(summary_part).schema_arrow
    assert schema.field("q_value").type == pa.float64()
    assert schema.field("effect_size").type == pa.float64()
    assert schema.field("skip_reason").type == pa.string()
    assert (
        schema.remove_metadata()
        == exact_arrow_schema(
            SUMMARY_TABLE,
            metadata={},
        ).remove_metadata()
    )
    assert payload["arrow_schema_contracts"]["inferred_v2_compatibility_tables"] == []


def test_group_statistics_zero_rows_publish_schema_bearing_empty_part(
    tmp_path: Path,
) -> None:
    export_root = _make_goodcopbadcop_export(tmp_path)
    config = GoodCopBadCopStatisticsConfig(
        export_root=export_root,
        source_export_run_id="source_export",
        stats_run_id="stats_empty",
        metrics=(),
        bootstrap_iterations=0,
        permutation_iterations=0,
        minimum_recordings=3,
        random_seed=0,
        allow_legacy_export_layout=True,
    )
    rows, descriptive_rows, manifest = compute_goodcopbadcop_outputs(config)
    assert rows == []
    assert descriptive_rows == []

    payload = write_goodcopbadcop_statistics(
        rows,
        manifest,
        export_root=export_root,
        stats_run_id="stats_empty",
    )
    parts = manifest_selected_part_files(
        export_root,
        "stats_empty",
        SUMMARY_TABLE,
    )
    assert len(parts) == 1
    parquet_file = pq.ParquetFile(parts[0])
    assert parquet_file.metadata.num_rows == 0
    assert (
        parquet_file.schema_arrow.remove_metadata()
        == exact_arrow_schema(
            SUMMARY_TABLE,
            metadata={},
        ).remove_metadata()
    )
    assert payload["row_counts_by_table"][SUMMARY_TABLE] == 0


@pytest.mark.parametrize("table_name", (SUMMARY_TABLE, DESCRIPTIVE_TABLE))
def test_group_statistics_exact_contract_has_unique_ordered_fields(
    table_name: str,
) -> None:
    fields = ARROW_TABLE_CONTRACTS[table_name].fields
    names = tuple(field.name for field in fields)
    assert len(names) == len(set(names))


def test_goodcopbadcop_statistics_computes_cra_primary_endpoint_wilcoxon(
    tmp_path: Path,
) -> None:
    export_root = _make_goodcopbadcop_cra_export(tmp_path)
    config = GoodCopBadCopStatisticsConfig(
        export_root=export_root,
        source_export_run_id="source_export",
        stats_run_id="stats_cra",
        metrics=metric_specs_for_families(("cra_primary_endpoint",)),
        bootstrap_iterations=0,
        minimum_recordings=3,
        random_seed=0,
        allow_legacy_export_layout=True,
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
    specificity = next(
        row for row in rows if row["metric_name"] == "specificity_distance"
    )
    assert specificity["primary"] is True
    occupancy_specificity = next(
        row for row in rows if row["metric_name"] == "specificity_occupancy"
    )
    assert occupancy_specificity["primary"] is False
    assert target["p_value"] == pytest.approx(0.25)
    assert target["test_method"] == "wilcoxon_signed_rank_exact"
    assert target["metric_unit"] == "mm"
    assert target["effect_size_kind"] == "rank_biserial_correlation"
    assert target["ci_estimand"] == "one_sample_median"

    inert = next(row for row in rows if row["metric_name"] == "delta_inert")
    assert inert["primary"] is False
    assert inert["p_value"] == pytest.approx(0.25)
    assert inert["effect_size"] == pytest.approx(1.0)


def test_goodcopbadcop_statistics_computes_epoch_behavior_metrics(
    tmp_path: Path,
) -> None:
    export_root = _make_goodcopbadcop_epoch_behavior_export(tmp_path)
    config = GoodCopBadCopStatisticsConfig(
        export_root=export_root,
        source_export_run_id="source_export",
        stats_run_id="stats_epoch_behavior",
        metrics=metric_specs_for_families(("epoch_behavior",)),
        bootstrap_iterations=0,
        minimum_recordings=3,
        random_seed=0,
        allow_legacy_export_layout=True,
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
    families = {row["multiple_comparison_family"] for row in rows}
    assert families == {"exploratory|epoch_behavior"}
    assert (
        sum(
            row["multiple_comparison_family"] == "exploratory|epoch_behavior"
            for row in rows
        )
        == 36
    )


def test_group_statistics_rejects_missing_registry_subject_identity(
    tmp_path: Path,
) -> None:
    export_root = _make_goodcopbadcop_epoch_behavior_export(tmp_path)
    part = (
        export_root
        / "v1"
        / "chaser_epoch_behavior_summary"
        / "export_run_id=source_export"
        / "part-00000.parquet"
    )
    table = pq.read_table(part).drop(["subject_id"])
    pq.write_table(table, part)

    with pytest.raises(ValueError, match="subject_id"):
        compute_goodcopbadcop_statistics(
            GoodCopBadCopStatisticsConfig(
                export_root=export_root,
                source_export_run_id="source_export",
                stats_run_id="stats_missing_subject",
                metrics=metric_specs_for_families(("epoch_behavior",)),
                bootstrap_iterations=0,
                minimum_recordings=3,
                allow_legacy_export_layout=True,
            )
        )


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
        "--allow-legacy-export-layout",
    ]

    assert compute_group_statistics_main(args) == 0
    dry_output = capsys.readouterr().out
    assert "dry_run\ttrue" in dry_output
    assert f"rows\t{DESCRIPTIVE_TABLE}\t18" in dry_output
    assert not (export_root / "v1" / SUMMARY_TABLE / "export_run_id=stats_cli").exists()

    assert compute_group_statistics_main([*args, "--apply"]) == 0
    apply_output = capsys.readouterr().out
    assert "manifest\t" in apply_output
    assert manifest_selected_part_files(export_root, "stats_cli", SUMMARY_TABLE)
    assert manifest_selected_part_files(export_root, "stats_cli", DESCRIPTIVE_TABLE)


def test_statistics_concurrent_first_publication_uses_compare_and_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_root = _make_goodcopbadcop_export(tmp_path)
    config = GoodCopBadCopStatisticsConfig(
        export_root=export_root,
        source_export_run_id="source_export",
        stats_run_id="stats_race",
        metrics=metric_specs_for_families(("chaser_distance",)),
        bootstrap_iterations=0,
        minimum_recordings=3,
        random_seed=0,
        allow_legacy_export_layout=True,
    )
    rows, manifest = compute_goodcopbadcop_statistics(config)
    import fisheye.group_statistics.goodcopbadcop as statistics

    real_validate = statistics.validate_staged_publication
    ready_to_commit = threading.Barrier(2)

    def synchronized_validate(staging_root: Path, payload: object) -> None:
        real_validate(staging_root, payload)
        ready_to_commit.wait(timeout=10)

    monkeypatch.setattr(
        statistics, "validate_staged_publication", synchronized_validate
    )

    def publish() -> dict[str, object]:
        return write_goodcopbadcop_statistics(
            rows,
            manifest,
            export_root=export_root,
            stats_run_id="stats_race",
        )

    outcomes: list[object] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(publish) for _index in range(2)]
        for future in futures:
            try:
                outcomes.append(future.result())
            except Exception as exc:
                outcomes.append(exc)

    assert sum(isinstance(item, dict) for item in outcomes) == 1
    assert (
        sum(
            isinstance(item, RuntimeError) and "changed during publication" in str(item)
            for item in outcomes
        )
        == 1
    )
    assert validate_export_run(export_root, "stats_race")["status"] == "valid"


def test_statistics_manifest_commit_failure_preserves_previous_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_root = _make_goodcopbadcop_export(tmp_path)
    config = GoodCopBadCopStatisticsConfig(
        export_root=export_root,
        source_export_run_id="source_export",
        stats_run_id="stats_commit_failure",
        metrics=metric_specs_for_families(("chaser_distance",)),
        bootstrap_iterations=0,
        minimum_recordings=3,
        random_seed=0,
        allow_legacy_export_layout=True,
    )
    rows, manifest = compute_goodcopbadcop_statistics(config)
    first = write_goodcopbadcop_statistics(
        rows,
        manifest,
        export_root=export_root,
        stats_run_id="stats_commit_failure",
    )
    manifest_path = Path(first["manifest_path"])
    first_generation = first["publication"]["generation_id"]
    import fisheye.group_statistics.goodcopbadcop as statistics

    real_replace = os.replace

    def fail_manifest(source: object, destination: object) -> None:
        if Path(destination) == manifest_path:
            raise OSError("injected statistics manifest commit failure")
        real_replace(source, destination)

    monkeypatch.setattr(statistics.os, "replace", fail_manifest)
    with pytest.raises(OSError, match="statistics manifest commit failure"):
        write_goodcopbadcop_statistics(
            rows,
            manifest,
            export_root=export_root,
            stats_run_id="stats_commit_failure",
            overwrite=True,
        )

    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["publication"]["generation_id"] == first_generation
    assert validate_export_run(export_root, "stats_commit_failure")["status"] == "valid"


def test_statistics_inventory_failure_cleans_hidden_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_root = _make_goodcopbadcop_export(tmp_path)
    config = GoodCopBadCopStatisticsConfig(
        export_root=export_root,
        source_export_run_id="source_export",
        stats_run_id="stats_inventory_failure",
        metrics=metric_specs_for_families(("chaser_distance",)),
        bootstrap_iterations=0,
        minimum_recordings=3,
        random_seed=0,
        allow_legacy_export_layout=True,
    )
    rows, manifest = compute_goodcopbadcop_statistics(config)
    import fisheye.group_statistics.goodcopbadcop as statistics

    monkeypatch.setattr(
        statistics,
        "sha256_file",
        lambda _path: (_ for _ in ()).throw(RuntimeError("injected inventory failure")),
    )
    with pytest.raises(RuntimeError, match="inventory failure"):
        write_goodcopbadcop_statistics(
            rows,
            manifest,
            export_root=export_root,
            stats_run_id="stats_inventory_failure",
        )

    assert not list((export_root / "v1" / ".staging").glob("*"))


def test_statistics_table_reads_bind_one_loaded_source_manifest_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_root = _make_goodcopbadcop_export(tmp_path)
    source_path = export_root / "v1" / "manifests" / "export_run_id=source_export.json"
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["snapshot_marker"] = "original"
    source_path.write_text(json.dumps(payload), encoding="utf-8")
    import fisheye.group_statistics.goodcopbadcop as statistics

    real_read = statistics._read_export_table
    seen_markers: list[object] = []

    def read_from_snapshot(
        root: Path,
        source_manifest: dict[str, object],
        table: str,
        *,
        allow_legacy_layout: bool = False,
    ):
        seen_markers.append(source_manifest.get("snapshot_marker"))
        if len(seen_markers) == 1:
            replaced = json.loads(source_path.read_text(encoding="utf-8"))
            replaced["snapshot_marker"] = "replacement"
            source_path.write_text(json.dumps(replaced), encoding="utf-8")
        return real_read(
            root,
            source_manifest,
            table,
            allow_legacy_layout=allow_legacy_layout,
        )

    monkeypatch.setattr(statistics, "_read_export_table", read_from_snapshot)
    compute_goodcopbadcop_statistics(
        GoodCopBadCopStatisticsConfig(
            export_root=export_root,
            source_export_run_id="source_export",
            stats_run_id="stats_snapshot",
            metrics=metric_specs_for_families(("chaser_distance",)),
            bootstrap_iterations=0,
            minimum_recordings=3,
            random_seed=0,
            allow_legacy_export_layout=True,
        )
    )

    assert len(seen_markers) >= 2
    assert set(seen_markers) == {"original"}


def test_statistics_cli_binds_summary_and_descriptive_to_one_source_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_root = _make_goodcopbadcop_export(tmp_path)
    source_path = export_root / "v1" / "manifests" / "export_run_id=source_export.json"
    original_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    import fisheye.group_statistics.goodcopbadcop as statistics

    real_load = statistics._load_json_snapshot
    load_count = 0

    def replace_after_snapshot(path: Path):
        nonlocal load_count
        load_count += 1
        snapshot = real_load(path)
        replacement = dict(snapshot[0])
        replacement["replacement_generation"] = load_count
        path.write_text(json.dumps(replacement), encoding="utf-8")
        return snapshot

    monkeypatch.setattr(statistics, "_load_json_snapshot", replace_after_snapshot)
    assert (
        compute_group_statistics_main(
            [
                "--profile",
                "chaser",
                "--source-export-run-id",
                "source_export",
                "--export-root",
                str(export_root),
                "--stats-run-id",
                "stats_one_snapshot",
                "--metrics",
                "chaser_distance",
                "--bootstrap-iterations",
                "0",
                "--minimum-recordings",
                "3",
                "--allow-legacy-export-layout",
                "--apply",
            ]
        )
        == 0
    )

    assert load_count == 1
    published_manifest = json.loads(
        (
            export_root / "v1" / "manifests" / "export_run_id=stats_one_snapshot.json"
        ).read_text(encoding="utf-8")
    )
    assert published_manifest["source_export_manifest_sha256"] == original_digest
    for table_name in (SUMMARY_TABLE, DESCRIPTIVE_TABLE):
        rows = pq.read_table(
            manifest_selected_part_files(
                export_root,
                "stats_one_snapshot",
                table_name,
            )[0]
        ).to_pylist()
        assert rows
        assert {row["source_export_manifest_sha256"] for row in rows} == {
            original_digest
        }


@pytest.mark.parametrize(
    "source_run_id",
    ["../escape", "bad\nrun", "bad:run", "café"],
)
def test_statistics_rejects_nonportable_source_run_before_manifest_read(
    tmp_path: Path,
    source_run_id: str,
) -> None:
    with pytest.raises(ValueError, match="Invalid source export run ID"):
        compute_goodcopbadcop_statistics(
            GoodCopBadCopStatisticsConfig(
                export_root=tmp_path / "exports",
                source_export_run_id=source_run_id,
                stats_run_id="stats_safe",
                metrics=metric_specs_for_families(("chaser_distance",)),
                allow_legacy_export_layout=True,
            )
        )


def test_statistics_legacy_mode_rejects_unsafe_payload_run_identity(
    tmp_path: Path,
) -> None:
    export_root = _make_goodcopbadcop_export(tmp_path)
    source_path = export_root / "v1" / "manifests" / "export_run_id=source_export.json"
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["export_run_id"] = "../escape"
    source_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="run identity"):
        compute_goodcopbadcop_statistics(
            GoodCopBadCopStatisticsConfig(
                export_root=export_root,
                source_export_run_id="source_export",
                stats_run_id="stats_safe",
                metrics=metric_specs_for_families(("chaser_distance",)),
                allow_legacy_export_layout=True,
            )
        )
