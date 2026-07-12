from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.analysis.chaser_distance_runs import write_chaser_distance_run
from fisheye.analysis.chaser_egocentric_bearing import (
    build_chaser_egocentric_bearing_result,
    write_chaser_egocentric_bearing_component,
)
from fisheye.analysis.cra_primary_endpoint import build_cra_primary_endpoint_result, write_cra_primary_endpoint_component
from fisheye.analysis.cra_near_field import build_cra_near_field_result, write_cra_near_field_component
from fisheye.analysis.goodcopbadcop_epoch_behavior_summary import (
    build_goodcopbadcop_epoch_behavior_summary_result,
    write_goodcopbadcop_epoch_behavior_summary_component,
)
from fisheye.group_analytics_viewer.query import (
    _enrich_chaser_behavior_rows,
    _summary,
    build_context,
    build_health_report,
    query_chaser_histogram,
    query_chaser_summary,
    query_cra_near_field_object_phase,
    query_cra_near_field_curves,
    query_cra_near_field_summary,
    query_cra_object_phase,
    query_cra_quadrant_occupancy_density,
    query_cra_specificity,
    query_cra_summary,
    query_egocentric_histogram,
    query_egocentric_summary,
    query_epoch_bout_histogram,
    query_epoch_center_distance_histogram,
    query_epoch_inter_bout_interval_histogram,
    query_epoch_speed_summary,
    query_export_summary,
    query_group_statistics,
    query_options,
    query_position_occupancy_grid_options,
    query_position_occupancy_histogram,
    query_recordings,
    query_speed_distance_bins,
    query_spatial_occupancy,
    rebin_position_occupancy_rows,
)
from fisheye.group_statistics.goodcopbadcop import (
    GoodCopBadCopStatisticsConfig,
    compute_goodcopbadcop_descriptive_summaries,
    compute_goodcopbadcop_statistics,
    metric_specs_for_families,
    write_goodcopbadcop_statistics,
)
from fisheye.utils.export_cross_recording_analytics import export_sources
from fisheye.utils import serve_group_analytics_viewer
from tests.unit.fisheye.test_goodcopbadcop_interactive import (
    _make_archive_with_detection_occupancy,
    _make_chaser_result,
)
from tests.unit.fisheye.test_chaser_egocentric_bearing import _add_track_kinematics_run
from tests.unit.fisheye.test_export_cross_recording_analytics import (
    _add_goodcopbadcop_cra_protocol_metadata,
    _add_goodcopbadcop_swim_bout_run,
)
from tests.unit.fisheye.test_cra_near_field import _add_circle_geometry


def test_group_analytics_summary_reports_sample_std_and_sem() -> None:
    summary = _summary([1.0, 3.0, 5.0])

    assert summary["n"] == 3
    assert summary["mean"] == pytest.approx(3.0)
    assert summary["std_dev"] == pytest.approx(2.0)
    assert summary["sem"] == pytest.approx(2.0 / (3.0 ** 0.5))


def test_unknown_chaser_roles_are_resolved_by_recording_and_object_column() -> None:
    distance_rows = [
        {
            "recording_id": "recording-a",
            "chaser_column_index": 0,
            "behavior_class": "unknown",
        },
        {
            "recording_id": "recording-a",
            "chaser_column_index": 1,
            "behavior_class": "unknown",
        },
    ]
    object_phase_rows = [
        {
            "recording_id": "recording-a",
            "object_column_index": 0,
            "object_role": "aggressive",
            "raw_color_hex": "#ff0000",
        },
        {
            "recording_id": "recording-a",
            "object_column_index": 1,
            "object_role": "inert",
            "raw_color_hex": "#1600ff",
        },
    ]

    enriched = _enrich_chaser_behavior_rows(distance_rows, object_phase_rows)

    assert [row["behavior_class"] for row in enriched] == ["aggressive", "inert"]
    assert [row["raw_color_hex"] for row in enriched] == ["#ff0000", "#1600ff"]


def _make_goodcopbadcop_export(tmp_path: Path):
    source = _make_archive_with_detection_occupancy(tmp_path)
    _add_goodcopbadcop_cra_protocol_metadata(source)
    write_chaser_distance_run(source, _make_chaser_result(source), overwrite=True)
    cra_result = build_cra_primary_endpoint_result(source, chaser_distance_run="chaser_distance_1")
    write_cra_primary_endpoint_component(source, cra_result, overwrite=True)
    _add_circle_geometry(source)
    near_field_result = build_cra_near_field_result(
        source,
        chaser_distance_run="chaser_distance_1",
        cra_primary_endpoint_component="object_relative_pre_post_v1",
        r_zone_mm=2.0,
        r_in_mm=2.0,
        r_out_mm=3.0,
        percentile_values=(5.0, 10.0),
        radial_bin_edges_mm=(0.0, 2.0, 4.0, 8.0),
        cdf_thresholds_mm=(2.0, 4.0),
        perimeter_band_mm=2.0,
    )
    write_cra_near_field_component(source, near_field_result, overwrite=True)
    _add_track_kinematics_run(source)
    _add_goodcopbadcop_swim_bout_run(source)
    epoch_behavior_result = build_goodcopbadcop_epoch_behavior_summary_result(
        source,
        chaser_distance_run="chaser_distance_1",
    )
    write_goodcopbadcop_epoch_behavior_summary_component(source, epoch_behavior_result, overwrite=True)
    egocentric_result = build_chaser_egocentric_bearing_result(
        source,
        chaser_distance_run="chaser_distance_1",
        track_kinematics_run="tk_1",
        distance_bin_width_mm=2.0,
        bearing_bin_width_deg=90.0,
    )
    write_chaser_egocentric_bearing_component(source, egocentric_result, overwrite=True)
    output = tmp_path / "exports" / "palette_analytics"
    export_sources(
        [source],
        output_root=output,
        export_run_id="viewer_export",
        tables=(
            "position_occupancy_histogram_2d",
            "chaser_epoch_spatial_occupancy_zones",
            "chaser_epoch_distance_summary",
            "chaser_epoch_behavior_summary",
            "chaser_epoch_bout_histogram",
            "chaser_epoch_inter_bout_interval_histogram",
            "chaser_epoch_center_distance_histogram",
            "chaser_speed_distance_bins",
            "chaser_epoch_distance_histogram",
            "chaser_cra_primary_endpoint_summary",
            "chaser_cra_primary_endpoint_object_phase",
            "chaser_cra_quadrant_occupancy",
            "chaser_cra_near_field_summary",
            "chaser_cra_near_field_object_phase",
            "chaser_cra_near_field_radial_density",
            "chaser_cra_near_field_distance_cdf",
            "chaser_egocentric_epoch_summary",
            "chaser_egocentric_distance_bearing_histogram",
        ),
        jobs=1,
    )
    return build_context(export_root=output, export_run_id="viewer_export")


def _write_goodcopbadcop_statistics(context, *, families=("chaser_distance",)) -> None:
    config = GoodCopBadCopStatisticsConfig(
        export_root=context.export_root,
        source_export_run_id=context.export_run_id,
        stats_run_id="stats_viewer",
        metrics=metric_specs_for_families(families),
        bootstrap_iterations=0,
        minimum_recordings=1,
        random_seed=0,
    )
    rows, manifest = compute_goodcopbadcop_statistics(config)
    descriptive_rows = compute_goodcopbadcop_descriptive_summaries(config)
    write_goodcopbadcop_statistics(
        rows,
        manifest,
        export_root=context.export_root,
        stats_run_id="stats_viewer",
        descriptive_rows=descriptive_rows,
    )


def test_group_analytics_viewer_queries_goodcopbadcop_export(tmp_path: Path) -> None:
    context = _make_goodcopbadcop_export(tmp_path)

    health = build_health_report(context)
    assert health.ok is True

    summary = query_export_summary(context)
    assert summary["source_recording_count"] == 1
    assert summary["row_counts_by_table"]["position_occupancy_histogram_2d"] == 12
    assert summary["row_counts_by_table"]["chaser_epoch_spatial_occupancy_zones"] == 12
    assert summary["row_counts_by_table"]["chaser_epoch_distance_summary"] == 6
    assert summary["row_counts_by_table"]["chaser_epoch_behavior_summary"] == 3
    assert summary["row_counts_by_table"]["chaser_epoch_bout_histogram"] == 183
    assert summary["row_counts_by_table"]["chaser_epoch_inter_bout_interval_histogram"] == 3
    assert summary["row_counts_by_table"]["chaser_epoch_center_distance_histogram"] == 9
    assert summary["row_counts_by_table"]["chaser_speed_distance_bins"] == 18
    assert summary["row_counts_by_table"]["chaser_epoch_distance_histogram"] == 18
    assert summary["row_counts_by_table"]["chaser_cra_primary_endpoint_summary"] == 1
    assert summary["row_counts_by_table"]["chaser_cra_primary_endpoint_object_phase"] == 4
    assert summary["row_counts_by_table"]["chaser_cra_quadrant_occupancy"] == 8
    assert summary["row_counts_by_table"]["chaser_cra_near_field_summary"] == 1
    assert summary["row_counts_by_table"]["chaser_cra_near_field_object_phase"] == 4
    assert summary["row_counts_by_table"]["chaser_cra_near_field_radial_density"] == 12
    assert summary["row_counts_by_table"]["chaser_cra_near_field_distance_cdf"] == 8
    assert summary["row_counts_by_table"]["chaser_egocentric_epoch_summary"] == 6
    assert summary["row_counts_by_table"]["chaser_egocentric_distance_bearing_histogram"] == 72
    assert summary["statistics"]["available"] is False

    options = query_options(context)
    assert [item["window_label"] for item in options["windows"]] == [
        "pre_event",
        "training_event",
        "post_event",
    ]
    assert options["chasers"] == [0, 1]
    assert options["cra_phases"] == [
        {"phase_axis_index": 0, "phase_label": "pre_static"},
        {"phase_axis_index": 1, "phase_label": "post_static"},
    ]
    assert options["cra_object_roles"] == ["aggressive", "inert"]
    assert options["cra_object_phase_metrics"][0]["metric"] == "median_distance_mm"
    assert options["epoch_speed_metrics"][0]["metric"] == "mean_speed_mm_s"
    assert "mean_inter_bout_interval_s" in {item["metric"] for item in options["epoch_speed_metrics"]}
    assert "mean_bout_net_heading_change_deg" in {item["metric"] for item in options["epoch_speed_metrics"]}
    assert options["epoch_bout_histogram_metrics"][0]["metric"] == "bout_path_length_mm"
    assert options["epoch_inter_bout_interval_histogram_metrics"][0]["metric"] == "inter_bout_interval_s"
    assert options["cra_near_field_object_phase_metrics"][0]["metric"] == "approach_p05_mm"
    assert options["egocentric_metrics"][0]["metric"] == "mean_alignment_cos"

    position_grids = query_position_occupancy_grid_options(context)
    assert len(position_grids) == 1
    assert position_grids[0]["recording_count"] == 1
    assert "2 × 2 native bins" in position_grids[0]["label"]
    position = query_position_occupancy_histogram(
        context,
        grid_id=position_grids[0]["grid_id"],
    )
    assert position["available"] is True
    assert position["recording_count"] == 1
    assert len(position["rows"]) == 12
    pre_position = [
        row for row in position["rows"] if row["window_label"] == "pre_event"
    ]
    assert len(pre_position) == 4
    assert sum(row["pooled_count"] for row in pre_position) == 4
    assert sum(row["pooled_probability"] for row in pre_position) == pytest.approx(1.0)
    rebinned_position = rebin_position_occupancy_rows(
        position["rows"],
        x_bin_factor=2,
        y_bin_factor=2,
    )
    assert len(rebinned_position) == 3
    assert all(row["pooled_probability"] == pytest.approx(1.0) for row in rebinned_position)

    spatial = query_spatial_occupancy(context, metric="time_s", value_mode="total")
    pre_top_left = next(
        row
        for row in spatial["rows"]
        if row["window_label"] == "pre_event" and row["zone_id"] == "top_left"
    )
    assert pre_top_left["value"] == pytest.approx(0.1)
    assert pre_top_left["recording_count"] == 1
    assert "std_dev" in pre_top_left
    assert "sem" in pre_top_left
    assert pre_top_left["std_dev"] is None
    assert pre_top_left["sem"] is None

    chaser = query_chaser_summary(context, metric="p50_distance_mm", stat="mean")
    post_chaser_1 = next(
        row
        for row in chaser["rows"]
        if row["window_label"] == "post_event" and row["chaser_index"] == 1
    )
    assert post_chaser_1["value"] == pytest.approx(6.0)
    assert post_chaser_1["behavior_class"] == "inert"
    assert post_chaser_1["raw_color_hex"] == "#0000ff"
    assert post_chaser_1["std_dev"] is None
    assert post_chaser_1["sem"] is None

    histogram = query_chaser_histogram(context, window_label="pre_event", chaser_index=0)
    first_bin = next(row for row in histogram["rows"] if row["distance_bin_index"] == 0)
    assert first_bin["pooled_count"] == 1
    assert first_bin["pooled_total_count"] == 3
    assert first_bin["pooled_density"] == pytest.approx(1.0 / 6.0, rel=1e-5)

    epoch_speed = query_epoch_speed_summary(context, metric="mean_speed_mm_s", stat="mean")
    pre_speed = next(row for row in epoch_speed["rows"] if row["window_label"] == "pre_event")
    assert epoch_speed["available"] is True
    assert epoch_speed["source_table"] == "chaser_epoch_behavior_summary"
    assert epoch_speed["source_label"] == "persisted_epoch_behavior"
    assert pre_speed["value"] == pytest.approx(20.0)
    assert pre_speed["recording_count"] == 1

    epoch_ibi = query_epoch_speed_summary(context, metric="mean_inter_bout_interval_s", stat="mean")
    pre_ibi = next(row for row in epoch_ibi["rows"] if row["window_label"] == "pre_event")
    assert epoch_ibi["source_table"] == "chaser_epoch_behavior_summary"
    assert pre_ibi["value"] == pytest.approx(0.06)

    epoch_heading = query_epoch_speed_summary(context, metric="mean_bout_net_heading_change_deg", stat="mean")
    pre_heading = next(row for row in epoch_heading["rows"] if row["window_label"] == "pre_event")
    assert epoch_heading["metric_label"] == "Mean net bout heading change (deg)"
    assert pre_heading["value"] == pytest.approx(0.0)

    bout_hist = query_epoch_bout_histogram(context, metric="bout_duration_s", window_label="pre_event")
    assert bout_hist["available"] is True
    assert bout_hist["source_table"] == "chaser_epoch_bout_histogram"
    assert bout_hist["metric_label"] == "Bout duration (s)"
    assert sum(row["pooled_count"] for row in bout_hist["rows"]) == 2
    assert sum(row["pooled_fraction"] for row in bout_hist["rows"]) == pytest.approx(1.0)
    first_bout_bin = bout_hist["rows"][0]
    assert first_bout_bin["bin_left"] is not None
    assert first_bout_bin["bin_right"] is not None
    assert first_bout_bin["recording_count"] == 1

    ibi_hist = query_epoch_inter_bout_interval_histogram(context, window_label="pre_event")
    assert ibi_hist["available"] is True
    assert ibi_hist["source_table"] == "chaser_epoch_inter_bout_interval_histogram"
    assert ibi_hist["metric_label"] == "Inter-bout interval (s)"
    assert sum(row["pooled_count"] for row in ibi_hist["rows"]) == 1
    assert sum(row["pooled_fraction"] for row in ibi_hist["rows"]) == pytest.approx(1.0)

    speed_distance = query_speed_distance_bins(context, window_label="pre_event", chaser_index=0)
    speed_bin = next(row for row in speed_distance["rows"] if row["distance_bin_index"] == 0)
    assert speed_distance["available"] is True
    assert speed_bin["pooled_speed_sample_count"] == 2
    assert speed_bin["pooled_mean_speed_mm_s"] == pytest.approx(5.0)
    assert speed_bin["recording_count"] == 1

    cra_object_phase = query_cra_object_phase(context, metric="occupancy_fraction", stat="mean")
    post_aggressive = next(
        row
        for row in cra_object_phase["rows"]
        if row["phase_label"] == "post_static" and row["object_role"] == "aggressive"
    )
    assert post_aggressive["object_quadrant_label"] == "bottom_right"
    assert post_aggressive["recording_count"] == 1
    assert post_aggressive["value"] == pytest.approx(0.0)
    assert post_aggressive["std_dev"] is None
    assert post_aggressive["sem"] is None

    cra_summary = query_cra_summary(context)
    assert cra_summary["row_count"] == 1
    assert cra_summary["statuses"] == ["computed"]
    delta_occ = next(row for row in cra_summary["metrics"] if row["metric"] == "delta_occ_agg")
    assert delta_occ["mean"] == pytest.approx(-1.0)
    assert cra_summary["rows"][0]["post_aggressive_quadrant"] == "bottom_right"
    assert cra_summary["rows"][0]["source_component_fingerprint"]

    cra_specificity = query_cra_specificity(context, bootstrap_iterations=0)
    assert cra_specificity["available"] is True
    assert cra_specificity["recording_count"] == 1
    assert len(cra_specificity["distance_slope_rows"]) == 2
    assert len(cra_specificity["distance_specificity_rows"]) == 1
    assert len(cra_specificity["occupancy_index_slope_rows"]) == 2
    assert len(cra_specificity["occupancy_index_specificity_rows"]) == 1
    assert cra_specificity["occupancy_index_specificity_rows"][0]["occupancy_index_specificity"] == pytest.approx(-8.0 / 3.0)

    quadrant_density = query_cra_quadrant_occupancy_density(
        context,
        bandwidth=0.05,
        bootstrap_iterations=0,
    )
    assert quadrant_density["available"] is True
    assert quadrant_density["row_count"] == 8
    assert quadrant_density["chance"] == pytest.approx(0.25)
    assert quadrant_density["kde"]["bandwidth"] == pytest.approx(0.05)
    assert quadrant_density["statistics"]["n"] == 1
    assert quadrant_density["statistics"]["test_method"].startswith("wilcoxon_signed_rank")
    assert quadrant_density["statistics"]["median_difference"] == pytest.approx(-1.0)
    assert len(quadrant_density["density_rows"]) == 404
    pre_fish_phase = next(
        row
        for row in quadrant_density["fish_phase_rows"]
        if row["phase_label"] == "pre_static"
    )
    assert pre_fish_phase["top_left_occ"] == pytest.approx(1.0)
    assert pre_fish_phase["chaser_quadrant"] == "top_left"
    assert pre_fish_phase["chaser_quadrant_occ"] == pytest.approx(1.0)
    assert pre_fish_phase["nonchaser_occ_pooled"] == [0.0, 0.0, 0.0]
    post_fish_phase = next(
        row
        for row in quadrant_density["fish_phase_rows"]
        if row["phase_label"] == "post_static"
    )
    assert post_fish_phase["bottom_left_occ"] == pytest.approx(1.0)
    assert post_fish_phase["chaser_quadrant"] == "bottom_right"
    assert post_fish_phase["chaser_quadrant_occ"] == pytest.approx(0.0)
    paired = quadrant_density["paired_rows"][0]
    assert paired["pre_chaser_quadrant"] == "top_left"
    assert paired["post_chaser_quadrant"] == "bottom_right"
    assert paired["delta_chaser_quadrant_occ"] == pytest.approx(-1.0)

    near_field_object_phase = query_cra_near_field_object_phase(
        context,
        metric="near_zone_occupancy_fraction",
        stat="mean",
    )
    near_field_post_aggressive = next(
        row
        for row in near_field_object_phase["rows"]
        if row["phase_label"] == "post_static" and row["object_role"] == "aggressive"
    )
    assert near_field_post_aggressive["recording_count"] == 1
    assert near_field_post_aggressive["value"] == pytest.approx(0.0)
    assert near_field_post_aggressive["std_dev"] is None
    assert near_field_post_aggressive["sem"] is None

    near_field_summary = query_cra_near_field_summary(context)
    assert near_field_summary["row_count"] == 1
    assert near_field_summary["statuses"] == ["computed"]
    nearzone_specificity = next(
        row for row in near_field_summary["metrics"] if row["metric"] == "nearzone_occ_specificity"
    )
    assert nearzone_specificity["mean"] == pytest.approx(-1.0)
    assert near_field_summary["rows"][0]["geometry_status"] == "circle"

    near_field_curves = query_cra_near_field_curves(context)
    assert near_field_curves["available"] is True
    assert near_field_curves["radial_row_count"] == 12
    assert near_field_curves["cdf_row_count"] == 8
    assert len(near_field_curves["radial_rows"]) == 12
    assert len(near_field_curves["cdf_rows"]) == 8
    first_cdf = next(
        row
        for row in near_field_curves["cdf_rows"]
        if row["phase_label"] == "pre_static" and row["object_role"] == "aggressive"
    )
    assert first_cdf["recording_count"] == 1
    assert first_cdf["distance_threshold_mm"] == pytest.approx(2.0)

    egocentric = query_egocentric_summary(context, metric="mean_alignment_cos", stat="mean")
    pre_egocentric_chaser_0 = next(
        row
        for row in egocentric["rows"]
        if row["window_label"] == "pre_event" and row["chaser_index"] == 0
    )
    assert pre_egocentric_chaser_0["recording_count"] == 1
    assert pre_egocentric_chaser_0["value"] is not None
    assert pre_egocentric_chaser_0["std_dev"] is None
    assert pre_egocentric_chaser_0["sem"] is None

    egocentric_histogram = query_egocentric_histogram(context, window_label="pre_event", chaser_index=0)
    first_egocentric_bin = next(
        row
        for row in egocentric_histogram["rows"]
        if row["distance_bin_index"] == 0 and row["bearing_bin_index"] == 0
    )
    assert first_egocentric_bin["pooled_total_count"] == 3
    assert first_egocentric_bin["distance_bin_center_mm"] == pytest.approx(1.0)
    assert first_egocentric_bin["bearing_bin_center_deg"] == pytest.approx(-135.0)

    recordings = query_recordings(context)
    assert recordings["row_count"] == 1
    assert recordings["rows"][0]["pre_event_coverage_pct"] == pytest.approx(100.0)
    assert recordings["rows"][0]["post_event_chaser_1_p50_mm"] == pytest.approx(6.0)
    assert recordings["rows"][0]["pre_event_mean_speed_mm_s"] == pytest.approx(20.0)
    assert recordings["rows"][0]["post_event_mean_speed_mm_s"] == pytest.approx(70.0)
    assert recordings["rows"][0]["cra_endpoint_status"] == "computed"
    assert recordings["rows"][0]["cra_delta_occ_agg"] == pytest.approx(-1.0)
    assert recordings["rows"][0]["cra_post_aggressive_quadrant"] == "bottom_right"
    assert recordings["rows"][0]["cra_near_field_status"] == "computed"
    assert recordings["rows"][0]["cra_near_field_nearzone_occ_specificity"] == pytest.approx(-1.0)
    assert recordings["rows"][0]["cra_near_field_geometry_status"] == "circle"
    assert recordings["rows"][0]["pre_event_chaser_0_alignment"] is not None


def test_group_analytics_viewer_queries_matching_statistics_export(tmp_path: Path) -> None:
    context = _make_goodcopbadcop_export(tmp_path)
    _write_goodcopbadcop_statistics(context)

    summary = query_export_summary(context)
    assert summary["statistics"]["available"] is True
    assert summary["statistics"]["stats_run_id"] == "stats_viewer"
    assert summary["statistics"]["row_count"] == 18
    assert summary["statistics"]["descriptive_row_count"] == 18

    chaser = query_chaser_summary(context, metric="p50_distance_mm", stat="mean")
    assert chaser["summary_source"] == "persisted_descriptive_summary"
    post_chaser_1 = next(
        row
        for row in chaser["rows"]
        if row["window_label"] == "post_event" and row["chaser_index"] == 1
    )
    assert post_chaser_1["summary_source"] == "persisted_descriptive_summary"
    assert post_chaser_1["value"] == pytest.approx(6.0)

    statistics = query_group_statistics(context, metric_name="p50_distance_mm")
    assert statistics["available"] is True
    assert statistics["stats_run_id"] == "stats_viewer"
    assert statistics["row_count"] == 6
    first = statistics["rows"][0]
    assert first["metric_family"] == "chaser_distance"
    assert first["group"] in {"aggressive", "inert"}
    assert first["paired_unit_count"] == 1
    assert first["mean_difference"] is not None
    assert "p_value" in first


def test_group_analytics_viewer_prefers_epoch_behavior_descriptive_summary(tmp_path: Path) -> None:
    context = _make_goodcopbadcop_export(tmp_path)
    _write_goodcopbadcop_statistics(context, families=("epoch_behavior",))

    summary = query_export_summary(context)
    assert summary["statistics"]["descriptive_row_count"] > 0

    epoch_ibi = query_epoch_speed_summary(context, metric="mean_inter_bout_interval_s", stat="mean")
    assert epoch_ibi["source_table"] == "chaser_epoch_behavior_summary"
    assert epoch_ibi["summary_source"] == "persisted_descriptive_summary"
    pre_ibi = next(row for row in epoch_ibi["rows"] if row["window_label"] == "pre_event")
    assert pre_ibi["summary_source"] == "persisted_descriptive_summary"
    assert pre_ibi["value"] == pytest.approx(0.06)

    center_hist = query_epoch_center_distance_histogram(context, window_label="pre_event")
    assert center_hist["available"] is True
    assert center_hist["source_table"] == "chaser_epoch_center_distance_histogram"
    assert sum(row["pooled_count"] for row in center_hist["rows"]) == 3

    bout_hist = query_epoch_bout_histogram(context, metric="bout_path_length_mm", window_label="pre_event")
    assert bout_hist["available"] is True
    assert bout_hist["source_table"] == "chaser_epoch_bout_histogram"
    assert sum(row["pooled_count"] for row in bout_hist["rows"]) == 2

    ibi_hist = query_epoch_inter_bout_interval_histogram(context, window_label="pre_event")
    assert ibi_hist["available"] is True
    assert ibi_hist["source_table"] == "chaser_epoch_inter_bout_interval_histogram"
    assert sum(row["pooled_count"] for row in ibi_hist["rows"]) == 1


def test_group_analytics_viewer_queries_cra_primary_endpoint_statistics(tmp_path: Path) -> None:
    context = _make_goodcopbadcop_export(tmp_path)
    _write_goodcopbadcop_statistics(context, families=("cra_primary_endpoint",))

    summary = query_export_summary(context)
    assert summary["statistics"]["available"] is True
    assert summary["statistics"]["stats_run_id"] == "stats_viewer"

    statistics = query_group_statistics(context, metric_family="cra_primary_endpoint")
    assert statistics["available"] is True
    assert statistics["stats_run_id"] == "stats_viewer"
    assert statistics["source_export_run_id"] == context.export_run_id
    assert statistics["row_count"] == 6
    by_metric = {row["metric_name"]: row for row in statistics["rows"]}
    assert by_metric["delta_occ_agg"]["test_method"].startswith("wilcoxon_signed_rank")
    assert by_metric["specificity_distance"]["contrast_name"] == "vs-zero"
    assert by_metric["specificity_distance"]["primary"] is True
    assert by_metric["specificity_occupancy"]["primary"] is False
    assert by_metric["delta_agg"]["primary"] is False
    assert by_metric["delta_inert"]["primary"] is False
    assert by_metric["delta_occ_agg"]["paired_unit_count"] == 1


def test_group_analytics_viewer_queries_cra_near_field_statistics(tmp_path: Path) -> None:
    context = _make_goodcopbadcop_export(tmp_path)
    _write_goodcopbadcop_statistics(context, families=("cra_near_field",))

    summary = query_export_summary(context)
    assert summary["statistics"]["available"] is True
    assert summary["statistics"]["stats_run_id"] == "stats_viewer"

    statistics = query_group_statistics(context, metric_family="cra_near_field")
    assert statistics["available"] is True
    assert statistics["stats_run_id"] == "stats_viewer"
    assert statistics["source_export_run_id"] == context.export_run_id
    assert statistics["row_count"] == 9
    by_metric = {row["metric_name"]: row for row in statistics["rows"]}
    assert by_metric["nearzone_occ_specificity"]["test_method"].startswith("wilcoxon_signed_rank")
    assert by_metric["approach_p05_specificity"]["contrast_name"] == "vs-zero"
    assert by_metric["nearzone_occ_delta_inert"]["primary"] is False
    assert by_metric["nearzone_occ_specificity"]["paired_unit_count"] == 1


def test_group_analytics_viewer_rejects_unknown_metric(tmp_path: Path) -> None:
    context = _make_goodcopbadcop_export(tmp_path)

    with pytest.raises(ValueError, match="Unsupported spatial metric"):
        query_spatial_occupancy(context, metric="not_a_metric")
    with pytest.raises(ValueError, match="Unsupported chaser metric"):
        query_chaser_summary(context, metric="not_a_metric")
    with pytest.raises(ValueError, match="Unsupported epoch behavior metric"):
        query_epoch_speed_summary(context, metric="not_a_metric")
    with pytest.raises(ValueError, match="Unsupported epoch metric histogram metric"):
        query_epoch_bout_histogram(context, metric="not_a_metric")
    with pytest.raises(ValueError, match="Unsupported epoch metric histogram metric"):
        query_epoch_inter_bout_interval_histogram(context, metric="not_a_metric")
    with pytest.raises(ValueError, match="Unsupported CRA object-phase metric"):
        query_cra_object_phase(context, metric="not_a_metric")
    with pytest.raises(ValueError, match="Unsupported CRA summary metric"):
        query_cra_summary(context, metric="not_a_metric")
    with pytest.raises(ValueError, match="Unsupported CRA near-field object-phase metric"):
        query_cra_near_field_object_phase(context, metric="not_a_metric")
    with pytest.raises(ValueError, match="Unsupported CRA near-field summary metric"):
        query_cra_near_field_summary(context, metric="not_a_metric")
    with pytest.raises(ValueError, match="Unsupported egocentric metric"):
        query_egocentric_summary(context, metric="not_a_metric")


def test_serve_group_analytics_viewer_host_warning(capsys) -> None:
    assert serve_group_analytics_viewer._host_is_loopback("127.0.0.1") is True
    assert serve_group_analytics_viewer._host_is_loopback("0.0.0.0") is False

    serve_group_analytics_viewer._print_network_exposure_warning("127.0.0.1", 8770)
    captured = capsys.readouterr()
    assert captured.err == ""

    serve_group_analytics_viewer._print_network_exposure_warning("0.0.0.0", 8770)
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "SSH port forwarding" in captured.err
