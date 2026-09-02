from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from fisheye.group_statistics.validated_behavior_appearance import APPEARANCE_COLUMNS
from fisheye.group_statistics.validated_behavior_distribution_specs import (
    DEFAULT_DISTRIBUTION_METRICS,
    DistributionMetricSpec,
    SCOPE_ORDER,
    validate_distribution_metric_specs,
)
from fisheye.group_statistics.validated_behavior_distribution_report import (
    ValidatedBehaviorDistributionReportError,
    read_validated_behavior_distribution_report,
    render_validated_behavior_distribution_report,
)
from fisheye.group_statistics.validated_behavior_distribution_views import (
    CENTRAL_99_RANGE,
    FULL_EVIDENCE_RANGE,
    TRACE_METHOD_ID,
    TRACE_SCHEMA_ID,
    TRACE_SCHEMA_VERSION,
    ValidatedBehaviorDistributionViewSource,
    _even_display_indices,
    build_distribution_view_payload,
    resolve_distribution_display_range,
    validate_distribution_view_payload,
    validate_motion_trace_payload,
)
from fisheye.group_statistics.validated_behavior_distributions import (
    ValidatedBehaviorDistributionConfig,
    ValidatedBehaviorDistributionError,
    ValidatedBehaviorDistributionResult,
    _AxisAudit,
    _SparseAccumulator,
    _cohort_bin_rows,
    _bin_indices,
    _finalize_recording_bins,
    _materialize_bound_intervals,
    _reduce_metric_values,
    _resolve_axis,
    _transition_scope_masks,
    derive_bout_heading_values,
    read_validated_behavior_distributions,
    wrap_heading_delta_degrees,
    write_validated_behavior_distributions,
)
from fisheye.visualization.validated_behavior_distributions import (
    render_distribution_figure,
)
from apps.marimo.components.validated_behavior_distributions import (
    validated_behavior_distribution_figure,
    validated_behavior_motion_trace_figure,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _duration_spec() -> DistributionMetricSpec:
    return DistributionMetricSpec(
        metric_id="fixture.duration_s",
        metric_family="fixture",
        source_surface="bout_observations",
        value_column="duration_s",
        unit="s",
        bin_width=0.1,
        lower_bound=0.0,
        upper_bound=None,
        coverage_policy="zero_anchored_cover_valid_max",
        weighting_ids=("event",),
        group_columns=(),
        validity_policy_id="finite_nonnegative_canonical_bout_value_v1",
        scope_binding_id="sealed_bout_source_row_epoch_membership_v1",
        interpretation="Fixture duration",
    )


def _appearance(source_digest: str) -> dict[str, object]:
    row = {
        "recording_id": "r1",
        "chaser_identity_code": 1,
        "chaser_index": 0,
        "chaser_identity": "stimulus-exact:chaser_index:0",
        "behavior_role_code": 1,
        "behavior_role": "aggressive",
        "stimulus_run_path": "analysis/stimulus_runs/exact",
        "source_protocol_sha256": "1" * 64,
        "experimental_color_r": 0.0,
        "experimental_color_g": 0.0,
        "experimental_color_b": 1.0,
        "experimental_color_a": 1.0,
        "experimental_color_hex": "#0000ff",
        "experimental_color_css": "rgba(0, 0, 255, 1)",
        "contrast_outline_hex": "#ffffff",
        "plotly_role_symbol": "star",
        "matplotlib_role_marker": "*",
        "appearance_schema_id": "palette.visualization.chaser_appearance_projection",
        "appearance_schema_version": 1,
        "appearance_policy_id": "protocol_rgba_independent_behavior_role_glyph_v1",
        "appearance_projection_sha256": "2" * 64,
        "occurrence_binding_sha256": "3" * 64,
        "color_semantics": "experimental_protocol_rgba",
        "role_semantics": "independent_marker_shape_and_text",
        "color_role_independence": True,
    }
    query = {
        "export_run_id": "fixture-export",
        "export_manifest_record_sha256": source_digest,
        "export_plan_sha256": "4" * 64,
        "table_name": "chaser_occurrences",
        "table_contract_sha256": "5" * 64,
        "grain": "fixture",
        "selected_columns": list(APPEARANCE_COLUMNS),
        "predicate_description": "all exact rows",
        "analysis_unit_policy_sha256": "6" * 64,
        "capability_policy": "fixture",
        "semantic_metadata": {},
    }
    body = {
        "schema_id": "palette.analytics.validated_behavior.chaser_appearance_dimension",
        "schema_version": 1,
        "method_id": "phase_c_chaser_occurrence_projection_v1",
        "status": "complete",
        "source_table": "chaser_occurrences",
        "join_fields": ["recording_id", "chaser_identity_code"],
        "color_semantics": "experimental_protocol_rgba",
        "role_semantics": "independent_marker_shape_and_text",
        "color_role_independence": True,
        "source_query_identity": query,
        "rows": [row],
    }
    return {**body, "record_sha256": canonical_json_sha256(body)}


def test_default_distribution_registry_is_unique_and_declares_weighting():
    ids = [spec.metric_id for spec in DEFAULT_DISTRIBUTION_METRICS]
    assert len(ids) == len(set(ids))
    assert {spec.source_surface for spec in DEFAULT_DISTRIBUTION_METRICS} == {
        "bout_observations",
        "inter_bout_interval_observations",
        "provider_motion_samples",
        "chaser_relative_samples",
    }
    assert next(
        spec
        for spec in DEFAULT_DISTRIBUTION_METRICS
        if spec.metric_id == "motion.filtered_speed_mm_s"
    ).weighting_ids == ("frame", "time")
    with pytest.raises(ValueError, match="must be unique"):
        validate_distribution_metric_specs((_duration_spec(), _duration_spec()))


def test_heading_reducer_matches_inclusive_wrapped_producer_rule():
    net, path = derive_bout_heading_values(
        acquisition_frames=np.asarray([0, 1, 2, 3], dtype=np.int64),
        smoothed_heading_deg=np.asarray([170.0, -170.0, -160.0, 50.0]),
        angular_sample_valid=np.asarray([True, True, True, True]),
        bout_start_frames=np.asarray([0, 2]),
        bout_end_frames=np.asarray([2, 3]),
    )
    assert net.tolist() == pytest.approx([30.0, -150.0])
    assert path.tolist() == pytest.approx([30.0, 150.0])
    assert float(wrap_heading_delta_degrees(180.0)) == -180.0


def test_producer_intervals_keep_exact_float_and_cross_epoch_whole_session_evidence():
    bouts = [
        {
            "bout_row_id": 0,
            "bout_id": 10,
            "start_acquisition_frame_id": 0,
            "end_acquisition_frame_id": 2,
        },
        {
            "bout_row_id": 1,
            "bout_id": 11,
            "start_acquisition_frame_id": 5,
            "end_acquisition_frame_id": 6,
        },
        {
            "bout_row_id": 2,
            "bout_id": 12,
            "start_acquisition_frame_id": 10,
            "end_acquisition_frame_id": 12,
        },
    ]
    epochs = [
        {
            "epoch_window_id": 1,
            "analysis_role": "chaser_pre",
            "start_frame": 0,
            "end_frame_exclusive": 10,
            "source_interval_sha256": "a" * 64,
        },
        {
            "epoch_window_id": 2,
            "analysis_role": "chaser_training",
            "start_frame": 10,
            "end_frame_exclusive": 20,
            "source_interval_sha256": "b" * 64,
        },
    ]
    dtype = np.dtype(
        [
            ("interval_id", "i8"),
            ("valid", "?"),
            ("prev_bout_id", "i8"),
            ("next_bout_id", "i8"),
            ("prev_end_frame", "i8"),
            ("next_start_frame", "i8"),
            ("interval_frames", "i8"),
            ("prev_end_time_s", "f8"),
            ("next_start_time_s", "f8"),
            ("interval_s", "f8"),
        ]
    )
    raw_intervals = np.asarray(
        [
            (0, True, 10, 11, 2, 5, 3, 0.02, 0.05, 0.30000000000000004),
            (1, True, 11, 12, 6, 10, 4, 0.06, 0.10, 0.4000000000000001),
        ],
        dtype=dtype,
    )
    rows, fps = _materialize_bound_intervals(
        canonical_rows=bouts,
        tables=SimpleNamespace(
            run_attrs={"fps": 10.0}, inter_bout_intervals=raw_intervals
        ),
        epochs=epochs,
    )
    assert fps == 10.0
    assert [row["interval_s"] for row in rows] == [
        0.30000000000000004,
        0.4000000000000001,
    ]
    assert rows[0]["analysis_role"] == "chaser_pre"
    assert rows[1]["analysis_role"] is None
    assert rows[1]["epoch_membership_state"] == "cross_epoch_or_outside"


def test_open_axis_uses_strict_upper_edge_when_maximum_is_on_an_edge():
    recipe = _resolve_axis(
        _duration_spec(),
        _AxisAudit(candidate_count=2, valid_count=2, minimum=0.0, maximum=0.2),
    )
    assert recipe["resolved_upper_bound"] == pytest.approx(0.3)
    assert recipe["bin_count"] == 3


def test_log_axis_preserves_orders_of_magnitude_without_linear_bin_explosion():
    tortuosity = next(
        spec
        for spec in DEFAULT_DISTRIBUTION_METRICS
        if spec.metric_id == "bout.tortuosity"
    )
    recipe = _resolve_axis(
        tortuosity,
        _AxisAudit(
            candidate_count=4,
            valid_count=4,
            minimum=0.0012,
            maximum=6695.1,
        ),
    )
    assert recipe["axis_scale"] == "log10"
    assert recipe["bin_count"] < 100
    indices = _bin_indices(np.asarray([0.0012, 1.0, 6695.1], dtype=np.float64), recipe)
    assert np.all(np.diff(indices) > 0)


def test_time_weighted_epoch_scope_rejects_boundary_crossing_transitions():
    epochs = [
        {
            "analysis_role": "chaser_pre",
            "start_frame": 0,
            "end_frame_exclusive": 10,
        },
        {
            "analysis_role": "chaser_training",
            "start_frame": 10,
            "end_frame_exclusive": 20,
        },
        {
            "analysis_role": "chaser_post",
            "start_frame": 20,
            "end_frame_exclusive": 30,
        },
    ]
    masks = _transition_scope_masks(
        np.asarray([9, 10, 11, 20], dtype=np.int64),
        np.asarray([1, 1, 1, 9], dtype=np.int64),
        epochs,
    )
    assert masks["whole_session"].tolist() == [True, True, True, True]
    assert masks["chaser_pre"].tolist() == [True, False, False, False]
    assert masks["chaser_training"].tolist() == [False, False, True, False]
    assert masks["chaser_post"].tolist() == [False, False, False, False]


def _fixture_reduction():
    source_digest = "a" * 64
    spec = _duration_spec()
    config = ValidatedBehaviorDistributionConfig(
        distribution_run_id="fixture-distributions-v1", metric_specs=(spec,)
    )
    accumulator = _SparseAccumulator()
    values_by_recording = {
        "r1": np.asarray([0.05, 0.15]),
        "r2": np.asarray([0.05, 0.05, 0.05, 0.15]),
    }
    for recording_id, values in values_by_recording.items():
        scopes = {
            scope: (
                np.ones(values.shape, dtype=bool)
                if scope == "whole_session"
                else np.zeros(values.shape, dtype=bool)
            )
            for scope in SCOPE_ORDER
        }
        _reduce_metric_values(
            accumulator,
            config=config,
            spec=spec,
            source_export_run_id="fixture-export",
            source_export_manifest_sha256=source_digest,
            recording_id=recording_id,
            values=values,
            scope_masks=scopes,
            base_valid=np.ones(values.shape, dtype=bool),
            group_arrays={},
            identity_arrays={"source": np.asarray(["exact"] * values.size)},
            time_weights_s=None,
            valid_duration_by_scope={scope: 1.0 for scope in SCOPE_ORDER},
        )
    recipes, support, sparse = _finalize_recording_bins(
        config=config,
        source_export_run_id="fixture-export",
        source_export_manifest_sha256=source_digest,
        accumulator=accumulator,
    )
    cohort = _cohort_bin_rows(
        config=config,
        source_export_run_id="fixture-export",
        source_export_manifest_sha256=source_digest,
        parent_recording_count=3,
        recipes=recipes,
        support_rows=support,
        sparse_rows=sparse,
    )
    return config, recipes, support, sparse, cohort


def test_cohort_bins_separate_equal_recording_and_pooled_observation_weights():
    _config, _recipes, _support, _sparse, cohort = _fixture_reduction()
    first = next(
        row
        for row in cohort
        if row["scope_id"] == "whole_session" and row["bin_index"] == 0
    )
    assert first["mean_recording_fraction"] == pytest.approx(0.625)
    assert first["pooled_fraction"] == pytest.approx(4.0 / 6.0)
    assert first["finite_recording_count"] == 2
    assert first["noncontributor_recording_count"] == 1
    empty = next(
        row
        for row in cohort
        if row["scope_id"] == "chaser_pre" and row["bin_index"] == 0
    )
    assert empty["finite_recording_count"] == 0
    assert empty["excluded_zero_denominator_recording_count"] == 2
    assert empty["mean_recording_fraction"] is None


def test_frame_and_time_weighting_are_distinct_declared_reductions():
    spec = DistributionMetricSpec(
        metric_id="fixture.motion",
        metric_family="fixture",
        source_surface="provider_motion_samples",
        value_column="value",
        unit="mm/s",
        bin_width=1.0,
        lower_bound=0.0,
        upper_bound=None,
        coverage_policy="zero_anchored_cover_valid_max",
        weighting_ids=("frame", "time"),
        group_columns=(),
        validity_policy_id="fixture_valid",
        scope_binding_id="fixture_scope",
        interpretation="Fixture motion",
    )
    config = ValidatedBehaviorDistributionConfig(
        distribution_run_id="fixture-weighting-v1", metric_specs=(spec,)
    )
    accumulator = _SparseAccumulator()
    values = np.asarray([0.2, 0.2, 1.2])
    scopes = {
        scope: (
            np.ones(values.shape, dtype=bool)
            if scope == "whole_session"
            else np.zeros(values.shape, dtype=bool)
        )
        for scope in SCOPE_ORDER
    }
    _reduce_metric_values(
        accumulator,
        config=config,
        spec=spec,
        source_export_run_id="fixture-export",
        source_export_manifest_sha256="a" * 64,
        recording_id="r1",
        values=values,
        scope_masks=scopes,
        base_valid=np.ones(values.shape, dtype=bool),
        group_arrays={},
        identity_arrays={"source": np.asarray(["exact"] * 3)},
        time_weights_s=np.asarray([0.1, 0.1, 0.8]),
        valid_duration_by_scope={scope: 1.0 for scope in SCOPE_ORDER},
    )
    recipes, support, sparse = _finalize_recording_bins(
        config=config,
        source_export_run_id="fixture-export",
        source_export_manifest_sha256="a" * 64,
        accumulator=accumulator,
    )
    cohort = _cohort_bin_rows(
        config=config,
        source_export_run_id="fixture-export",
        source_export_manifest_sha256="a" * 64,
        parent_recording_count=1,
        recipes=recipes,
        support_rows=support,
        sparse_rows=sparse,
    )
    frame_first = next(
        row
        for row in cohort
        if row["scope_id"] == "whole_session"
        and row["weighting_id"] == "frame"
        and row["bin_index"] == 0
    )
    time_first = next(
        row
        for row in cohort
        if row["scope_id"] == "whole_session"
        and row["weighting_id"] == "time"
        and row["bin_index"] == 0
    )
    assert frame_first["mean_recording_fraction"] == pytest.approx(2.0 / 3.0)
    assert time_first["mean_recording_fraction"] == pytest.approx(0.2)


def _write_fixture_distribution(tmp_path: Path) -> Path:
    config, recipes, support, sparse, cohort = _fixture_reduction()
    source_digest = "a" * 64
    result = ValidatedBehaviorDistributionResult(
        config=config,
        source_export={
            "path": "/tmp/fixture-export",
            "export_run_id": "fixture-export",
            "export_manifest_record_sha256": source_digest,
        },
        cohort_summary={"parent_recording_count": 3},
        source_queries=(),
        epoch_child_receipts=(),
        histogram_recipes=recipes,
        chaser_appearance_dimension=_appearance(source_digest),
        bout_observations=(),
        inter_bout_interval_observations=(),
        recording_support=support,
        recording_nonzero_bins=sparse,
        cohort_bins=cohort,
    )
    target = tmp_path / "distribution"
    write_validated_behavior_distributions(result, target)
    return target


def test_distribution_publication_round_trip_and_tamper_detection(tmp_path: Path):
    target = _write_fixture_distribution(tmp_path)
    reopened = read_validated_behavior_distributions(target)
    assert reopened["distribution_run_id"] == "fixture-distributions-v1"

    path = target / "cohort_distribution_bins.parquet"
    with path.open("ab") as handle:
        handle.write(b"stale")
    with pytest.raises(ValidatedBehaviorDistributionError, match="bytes differ"):
        read_validated_behavior_distributions(target)


def test_shared_payload_feeds_static_and_plotly_renderers(tmp_path: Path):
    target = _write_fixture_distribution(tmp_path)
    source = ValidatedBehaviorDistributionViewSource.open(target)
    payload = build_distribution_view_payload(source, "fixture.duration_s", "event")
    validate_distribution_view_payload(payload)
    assert payload["scope_order"] == list(SCOPE_ORDER)
    assert (
        payload["source_distribution"]["distribution_manifest_sha256"]
        == source.cache_identity
    )

    static = render_distribution_figure(payload)
    try:
        static.canvas.draw()
        assert len(static.axes) == 4
    finally:
        plt.close(static)
    interactive = validated_behavior_distribution_figure(payload)
    rendered = interactive.to_plotly_json()
    assert len(rendered["data"]) >= 4
    assert "Mean recording fraction" in rendered["layout"]["yaxis"]["title"]["text"]


def test_central_display_range_retains_whole_bins_without_mutating_payload(
    tmp_path: Path,
):
    target = _write_fixture_distribution(tmp_path)
    source = ValidatedBehaviorDistributionViewSource.open(target)
    original = build_distribution_view_payload(source, "fixture.duration_s", "event")
    original_json = json.dumps(dict(original), sort_keys=True)
    payload = json.loads(json.dumps(dict(original)))
    whole_rows = [
        row for row in payload["cohort_rows"] if row["scope_id"] == "whole_session"
    ]
    assert len(whole_rows) >= 2
    fractions = [0.995] + [0.0] * (len(whole_rows) - 2) + [0.005]
    for row, fraction in zip(whole_rows, fractions, strict=True):
        row["mean_recording_fraction"] = fraction
        row["median_recording_fraction"] = fraction
        row["pooled_fraction"] = fraction
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    payload["payload_sha256"] = canonical_json_sha256(body)
    validate_distribution_view_payload(payload)

    central = resolve_distribution_display_range(
        payload, display_range_id=CENTRAL_99_RANGE
    )
    full = resolve_distribution_display_range(
        payload, display_range_id=FULL_EVIDENCE_RANGE
    )
    assert central["effective_display_range_id"] == CENTRAL_99_RANGE
    assert central["display_upper_bound"] == pytest.approx(whole_rows[0]["bin_right"])
    assert central["minimum_series_fraction_retained"] == pytest.approx(0.995)
    assert full["display_upper_bound"] == pytest.approx(
        payload["histogram_recipe"]["resolved_upper_bound"]
    )
    assert json.dumps(dict(original), sort_keys=True) == original_json

    static = render_distribution_figure(payload, display_range_id=CENTRAL_99_RANGE)
    try:
        static.canvas.draw()
        assert static.axes[0].get_xlim()[1] == pytest.approx(
            central["display_upper_bound"]
        )
    finally:
        plt.close(static)
    interactive = validated_behavior_distribution_figure(
        payload, display_range_id=CENTRAL_99_RANGE
    ).to_plotly_json()
    assert (
        interactive["layout"]["meta"]["display_range"]["display_range_sha256"]
        == central["display_range_sha256"]
    )


def test_static_distribution_report_is_atomic_and_digest_validated(tmp_path: Path):
    target = _write_fixture_distribution(tmp_path)
    source = ValidatedBehaviorDistributionViewSource.open(target)
    report_dir = tmp_path / "report"
    manifest = render_validated_behavior_distribution_report(
        source,
        report_run_id="fixture-distribution-report-v1",
        output_dir=report_dir,
    )
    reopened = read_validated_behavior_distribution_report(report_dir)
    assert reopened["record_sha256"] == manifest["record_sha256"]
    assert reopened["schema_version"] == 2
    assert reopened["renderer"]["display_range_id"] == CENTRAL_99_RANGE
    assert len(reopened["artifacts"]) == 1
    assert reopened["artifacts"][0]["display_range"]["display_only"] is True
    assert (report_dir / "index.html").is_file()


def test_static_distribution_report_reader_retains_v1_compatibility(tmp_path: Path):
    target = _write_fixture_distribution(tmp_path)
    source = ValidatedBehaviorDistributionViewSource.open(target)
    report_dir = tmp_path / "legacy-report"
    render_validated_behavior_distribution_report(
        source,
        report_run_id="fixture-legacy-distribution-report-v1",
        output_dir=report_dir,
    )
    path = report_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    manifest["method_id"] = "shared_payload_matplotlib_distribution_report_v1"
    for field in ("display_range_id", "display_range_label", "display_range_policy"):
        manifest["renderer"].pop(field)
    for artifact in manifest["artifacts"]:
        artifact.pop("requested_display_range_id")
        artifact.pop("display_range")
    body = {key: value for key, value in manifest.items() if key != "record_sha256"}
    manifest["record_sha256"] = canonical_json_sha256(body)
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    reopened = read_validated_behavior_distribution_report(report_dir)
    assert reopened["schema_version"] == 1


def test_static_distribution_report_rejects_semantically_false_range(tmp_path: Path):
    target = _write_fixture_distribution(tmp_path)
    source = ValidatedBehaviorDistributionViewSource.open(target)
    report_dir = tmp_path / "stale-range-report"
    render_validated_behavior_distribution_report(
        source,
        report_run_id="fixture-stale-range-report-v2",
        output_dir=report_dir,
    )
    path = report_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    display = manifest["artifacts"][0]["display_range"]
    display["minimum_series_fraction_retained"] = 0.5
    display["maximum_series_fraction_omitted"] = 0.5
    range_body = {
        key: value for key, value in display.items() if key != "display_range_sha256"
    }
    display["display_range_sha256"] = canonical_json_sha256(range_body)
    body = {key: value for key, value in manifest.items() if key != "record_sha256"}
    manifest["record_sha256"] = canonical_json_sha256(body)
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(
        ValidatedBehaviorDistributionReportError, match="display range is invalid"
    ):
        read_validated_behavior_distribution_report(report_dir)


def test_motion_trace_payload_preserves_decimation_endpoints_and_frame_time_identity():
    assert _even_display_indices(10, 4).tolist() == [0, 3, 6, 9]
    metric = next(
        spec.to_dict()
        for spec in DEFAULT_DISTRIBUTION_METRICS
        if spec.metric_id == "motion.filtered_speed_mm_s"
    )
    points = [
        {
            "source_row_index": 0,
            "acquisition_frame_id": 100,
            "time_s": 1.0,
            "coordinate": 1.0,
            "value": 2.5,
            "valid": True,
        },
        {
            "source_row_index": 9,
            "acquisition_frame_id": 109,
            "time_s": 1.09,
            "coordinate": 1.09,
            "value": None,
            "valid": False,
        },
    ]
    body = {
        "schema_id": TRACE_SCHEMA_ID,
        "schema_version": TRACE_SCHEMA_VERSION,
        "method_id": TRACE_METHOD_ID,
        "metric": metric,
        "recording_id": "r1",
        "provider_role": "keypoint",
        "position_provider_id": "keypoint_anatomical_triad_mean.v1",
        "position_provider_digest": "1" * 64,
        "coordinate_id": "time",
        "coordinate_column": "time_s",
        "coordinate_choice_semantics": "display_only_same_exact_rows",
        "source_row_count": 10,
        "source_valid_count": 9,
        "display_point_count": 2,
        "max_display_points": 4,
        "decimation_id": "deterministic_even_index_endpoint_preserving_display_only_v1",
        "points": points,
        "source_query_identity": {"table_name": "provider_motion_samples"},
        "source_distribution_manifest_sha256": "2" * 64,
        "source_export_manifest_sha256": "3" * 64,
    }
    payload = {**body, "payload_sha256": canonical_json_sha256(body)}
    validate_motion_trace_payload(payload)
    figure = validated_behavior_motion_trace_figure(payload)
    assert figure.to_plotly_json()["layout"]["xaxis"]["title"]["text"] == "Time (s)"
    stale = {**payload, "coordinate_id": "frame"}
    with pytest.raises(ValueError, match="digest is stale"):
        validate_motion_trace_payload(stale)
