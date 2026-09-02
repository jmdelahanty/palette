from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

from apps.marimo.components.validated_behavior_group_statistics import (
    validated_behavior_statistics_figure,
)

from fisheye.analytics_exports.validated_behavior_phase_b_contracts import (
    PHASE_B_PROFILE_ID,
    PHASE_B_TABLE_SPECS,
)
from fisheye.analytics_exports.validated_behavior_phase_c_contracts import (
    PHASE_C_PROFILE_ID,
    PHASE_C_TABLE_SPECS,
)
from fisheye.group_statistics.validated_behavior import (
    ValidatedBehaviorGroupStatisticsConfig,
    ValidatedBehaviorGroupStatisticsError,
    compute_validated_behavior_group_statistics,
    read_validated_behavior_group_statistics_sandbox,
    write_validated_behavior_group_statistics_sandbox,
)
from fisheye.group_statistics.validated_behavior_specs import (
    CHASER_EPOCH_CONDITIONS,
    DEFAULT_VALIDATED_BEHAVIOR_HISTOGRAMS,
    DEFAULT_VALIDATED_BEHAVIOR_METRICS,
    ValidatedBehaviorMetricSpec,
    metric_specs_for_families,
)
from fisheye.group_statistics.validated_behavior_report import (
    ValidatedBehaviorStatisticsReportError,
    read_validated_behavior_statistics_report,
    render_validated_behavior_statistics_report,
)
from fisheye.group_statistics.validated_behavior_views import (
    ValidatedBehaviorStatisticsViewError,
    ValidatedBehaviorStatisticsViewSource,
    available_statistics_views,
    build_statistics_view_payload,
    validate_statistics_view_payload,
)
from fisheye.utils.compute_validated_behavior_group_statistics import main as cli_main
from fisheye.visualization.validated_behavior_group_statistics import (
    render_statistics_view,
)


def _metric() -> ValidatedBehaviorMetricSpec:
    return ValidatedBehaviorMetricSpec(
        metric_id="core_behavior.mean_speed_mm_s",
        metric_family="core_behavior",
        source_table="epoch_behavior_summary",
        value_column="mean_speed_mm_s",
        unit="mm/s",
        condition_column="analysis_role",
        expected_conditions=CHASER_EPOCH_CONDITIONS,
        group_columns=(),
        contrast_set_id="chaser_epoch_v1",
        multiplicity_family="fixture.epoch_contrasts",
        retain_recording_values=True,
        interpretation="Fixture mean speed",
    )


@dataclass(frozen=True)
class _FakeTable:
    dataset: "_FakeDataset"
    name: str
    frame: pl.DataFrame

    @property
    def spec(self):
        return self.dataset.table_specs[self.name]

    def scan(self, *, columns=None, predicate=None):
        lazy = self.frame.lazy()
        if predicate is not None:
            lazy = lazy.filter(predicate)
        if columns is not None:
            lazy = lazy.select(*columns)
        return lazy

    def query_identity(self, *, columns=None, predicate_description=None):
        return {
            "export_run_id": self.dataset.export_run_id,
            "export_manifest_record_sha256": self.dataset.cache_identity,
            "export_plan_sha256": self.dataset.manifest["export_plan"]["plan_sha256"],
            "table_name": self.name,
            "table_contract_sha256": self.spec.contract.payload_sha256,
            "grain": self.spec.grain,
            "selected_columns": list(columns or ()),
            "predicate_description": predicate_description,
            "analysis_unit_policy_sha256": self.dataset.manifest[
                "analysis_unit_policy"
            ]["sha256"],
            "capability_policy": self.spec.capability_policy,
            "semantic_metadata": dict(self.spec.semantic_metadata),
        }


class _FakeDataset:
    def __init__(
        self,
        frames: dict[str, pl.DataFrame],
        *,
        table_specs=PHASE_B_TABLE_SPECS,
        profile_id: str = PHASE_B_PROFILE_ID,
    ):
        self.root = Path("/tmp/fixture_validated_behavior")
        self.export_run_id = "fixture-phase-b"
        self.cache_identity = "a" * 64
        self.validation_mode = "receipt"
        self.table_specs = table_specs
        self.manifest = {
            "record_sha256": self.cache_identity,
            "export_plan": {"plan_sha256": "b" * 64},
            "export_profile": {"profile_id": profile_id},
            "analysis_unit_policy": {
                "sha256": "c" * 64,
                "record": {
                    "analysis_unit_kind": "recording",
                    "member_id_field": "recording_id",
                    "policy_id": "fixture_recording_units",
                },
            },
        }
        self._frames = frames

    @property
    def table_names(self):
        return tuple(self._frames)

    def table(self, name: str):
        return _FakeTable(self, name, self._frames[name])


def _frames(*, duplicate: bool = False, nonfinite: bool = False):
    recordings = ("r1", "r2", "r3")
    cohort = pl.DataFrame(
        {
            "recording_id": recordings,
            "analysis_unit_kind": ["recording"] * 3,
            "analysis_unit_id": recordings,
            "membership_state": ["included"] * 3,
            "acquisition_batch_id": [None, None, None],
            "acquisition_batch_identity_status": ["missing_historical_not_inferred"]
            * 3,
        }
    )
    bundles = pl.DataFrame(
        {"recording_id": recordings, "bundle_state": ["complete"] * 3}
    )
    capabilities = pl.DataFrame(
        {
            "recording_id": recordings,
            "capability_id": ["epoch_behavior"] * 3,
            "state": ["available"] * 3,
        }
    )
    rows = []
    for index, recording in enumerate(recordings, start=1):
        for multiplier, condition in zip((1.0, 2.0, 3.0), CHASER_EPOCH_CONDITIONS):
            value = float(index) * multiplier
            if nonfinite and recording == "r3" and condition == "chaser_training":
                value = np.nan
            rows.append(
                {
                    "recording_id": recording,
                    "analysis_role": condition,
                    "mean_speed_mm_s": value,
                    "total_path_mm": value * 100.0,
                    "tracking_dropout_fraction": min(value / 100.0, 1.0),
                }
            )
    if duplicate:
        rows.append(dict(rows[0]))
    return {
        "cohort_recordings": cohort,
        "recording_bundles": bundles,
        "recording_capabilities": capabilities,
        "epoch_behavior_summary": pl.DataFrame(rows),
    }


def _compute(*, duplicate: bool = False, nonfinite: bool = False):
    dataset = _FakeDataset(_frames(duplicate=duplicate, nonfinite=nonfinite))
    config = ValidatedBehaviorGroupStatisticsConfig(
        statistics_run_id="fixture-stats-v1",
        metric_specs=(_metric(),),
        bootstrap_iterations=100,
        permutation_iterations=100,
        confidence_level=0.95,
        minimum_recordings=2,
        random_seed=17,
    )
    return compute_validated_behavior_group_statistics(dataset, config)


def _phase_c_appearance_rows() -> pl.DataFrame:
    rows = []
    for recording_id in ("r1", "r2", "r3"):
        for code, role, color, symbol, marker in (
            (1, "aggressive", "#0000ff", "star", "*"),
            (2, "inert", "#ff0000", "circle", "o"),
        ):
            red, green, blue = (
                int(color[index : index + 2], 16) / 255.0 for index in (1, 3, 5)
            )
            rows.append(
                {
                    "recording_id": recording_id,
                    "chaser_identity_code": code,
                    "chaser_index": code - 1,
                    "chaser_identity": f"stimulus-exact:chaser_index:{code - 1}",
                    "behavior_role_code": code,
                    "behavior_role": role,
                    "stimulus_run_path": "analysis/stimulus_runs/stimulus-exact",
                    "source_protocol_sha256": "1" * 64,
                    "experimental_color_r": red,
                    "experimental_color_g": green,
                    "experimental_color_b": blue,
                    "experimental_color_a": 1.0,
                    "experimental_color_hex": color,
                    "experimental_color_css": (
                        f"rgba({red * 255:.0f}, {green * 255:.0f}, "
                        f"{blue * 255:.0f}, 1)"
                    ),
                    "contrast_outline_hex": "#ffffff",
                    "plotly_role_symbol": symbol,
                    "matplotlib_role_marker": marker,
                    "appearance_schema_id": (
                        "palette.visualization.chaser_appearance_projection"
                    ),
                    "appearance_schema_version": 1,
                    "appearance_policy_id": (
                        "protocol_rgba_independent_behavior_role_glyph_v1"
                    ),
                    "appearance_projection_sha256": (
                        f"{code + int(recording_id[1])}" * 64
                    )[:64],
                    "occurrence_binding_sha256": "8" * 64,
                    "color_semantics": "experimental_protocol_rgba",
                    "role_semantics": "independent_marker_shape_and_text",
                    "color_role_independence": True,
                }
            )
    return pl.DataFrame(rows)


def test_default_registry_is_composable_and_unique():
    ids = [spec.metric_id for spec in DEFAULT_VALIDATED_BEHAVIOR_METRICS]
    assert len(ids) == len(set(ids))
    core = metric_specs_for_families(("core_behavior",))
    assert len(core) == 8
    assert {spec.source_table for spec in core} == {"epoch_behavior_summary"}
    assert all(len(spec.spec_sha256) == 64 for spec in core)
    distance = metric_specs_for_families(("distance_traveled",))
    assert [spec.metric_id for spec in distance] == [
        "distance_traveled.session_total_path_mm",
        "distance_traveled.epoch_total_path_mm",
        "distance_traveled.epoch_mean_speed_mm_s",
        "distance_traveled.epoch_tracking_dropout_fraction",
    ]
    session = distance[0]
    assert session.source_table == "provider_motion_samples"
    assert session.recording_reducer == "terminal_at_max_order_v1"
    assert session.reducer_order_column == "track_sample_row_id"
    assert "source_manifest_sha256" in session.source_identity_columns
    histogram_ids = [spec.metric_id for spec in DEFAULT_VALIDATED_BEHAVIOR_HISTOGRAMS]
    assert histogram_ids == [
        "body_bearing_polar.recording_fraction",
        "body_bearing_distance.recording_joint_fraction",
    ]
    assert all(
        len(spec.spec_sha256) == 64 for spec in DEFAULT_VALIDATED_BEHAVIOR_HISTOGRAMS
    )


def test_phase_c_statistics_persist_exact_chaser_appearance_dimension(
    tmp_path: Path,
):
    frames = _frames()
    frames["chaser_occurrences"] = _phase_c_appearance_rows()
    dataset = _FakeDataset(
        frames,
        table_specs=PHASE_C_TABLE_SPECS,
        profile_id=PHASE_C_PROFILE_ID,
    )
    config = ValidatedBehaviorGroupStatisticsConfig(
        statistics_run_id="fixture-phase-c-appearance",
        metric_specs=(_metric(),),
        bootstrap_iterations=0,
        permutation_iterations=0,
        confidence_level=0.95,
        minimum_recordings=2,
        random_seed=17,
    )

    result = compute_validated_behavior_group_statistics(dataset, config)

    dimension = result.source_export["chaser_appearance_dimension"]
    assert dimension["record_sha256"]
    assert len(dimension["rows"]) == 6
    assert dimension["rows"][0]["experimental_color_hex"] == "#0000ff"
    assert dimension["rows"][0]["behavior_role"] == "aggressive"
    assert dimension["rows"][0]["plotly_role_symbol"] == "star"

    output = tmp_path / "phase-c-statistics"
    write_validated_behavior_group_statistics_sandbox(result, output)
    source = ValidatedBehaviorStatisticsViewSource.open(output)
    payload = build_statistics_view_payload(source, "core_behavior")
    validate_statistics_view_payload(payload)
    assert payload["chaser_appearance_dimension"]["record_sha256"] == dimension[
        "record_sha256"
    ]
    assert payload["behavior_role_styles"]["aggressive"] == {
        "aggregate_color_hex": "#0000ff",
        "aggregate_color_css": "rgba(0, 0, 255, 1)",
        "aggregate_color_policy": "unique_protocol_rgba_across_occurrences",
        "experimental_color_hex_values": ["#0000ff"],
        "experimental_color_css_values": ["rgba(0, 0, 255, 1)"],
        "plotly_role_symbol": "star",
        "matplotlib_role_marker": "*",
        "color_role_independence": True,
    }


def test_cli_lists_families_without_opening_a_dataset(capsys):
    assert cli_main(("--list-families",)) == 0
    output = capsys.readouterr().out.splitlines()
    assert "core_behavior" in output
    assert "trial_response" in output
    assert "body_bearing_polar" in output
    assert "body_bearing_distance" in output
    assert "distance_traveled" in output


def _provider_motion_rows(*, gap: bool = False, mixed_identity: bool = False):
    rows = []
    for recording_index, recording_id in enumerate(("r1", "r2", "r3"), start=1):
        orders = (10, 12, 13) if gap and recording_id == "r2" else (10, 11, 12)
        for local_index, order in enumerate(orders):
            rows.append(
                {
                    "recording_id": recording_id,
                    "membership_member_sha256": "1" * 64,
                    "bundle_set_member_sha256": "2" * 64,
                    "bundle_record_sha256": "3" * 64,
                    "source_binding_key": "provider_motion",
                    "source_run_path": (
                        "analysis/provider_track_motion_runs/mixed"
                        if mixed_identity
                        and recording_id == "r2"
                        and local_index == 2
                        else "analysis/provider_track_motion_runs/exact"
                    ),
                    "source_manifest_sha256": "4" * 64,
                    "source_verification_digest": "5" * 64,
                    "provider_role": "keypoint",
                    "position_provider_id": "canonical-keypoints",
                    "position_provider_digest": "6" * 64,
                    "track_id": 7,
                    "track_sample_row_id": order,
                    "cumulative_path_distance_mm": float(
                        recording_index * 100 + local_index * 10
                    ),
                }
            )
    return pl.DataFrame(rows)


def _compute_distance_session(*, gap: bool = False, mixed_identity: bool = False):
    frames = _frames()
    frames["provider_motion_samples"] = _provider_motion_rows(
        gap=gap, mixed_identity=mixed_identity
    )
    spec = next(
        item
        for item in DEFAULT_VALIDATED_BEHAVIOR_METRICS
        if item.metric_id == "distance_traveled.session_total_path_mm"
    )
    return compute_validated_behavior_group_statistics(
        _FakeDataset(frames),
        ValidatedBehaviorGroupStatisticsConfig(
            statistics_run_id="fixture-distance-session-v1",
            metric_specs=(spec,),
            bootstrap_iterations=0,
            permutation_iterations=0,
            confidence_level=0.95,
            minimum_recordings=2,
            random_seed=17,
        ),
    )


def test_terminal_distance_reducer_selects_exact_final_row_per_recording():
    result = _compute_distance_session()
    assert [row["value"] for row in result.recording_values] == [120.0, 220.0, 320.0]
    assert {row["condition"] for row in result.recording_values} == {"__all__"}
    assert result.descriptive_statistics[0]["mean"] == pytest.approx(220.0)
    assert result.recording_values[0]["recording_reducer"] == (
        "terminal_at_max_order_v1"
    )
    query = result.source_queries[0]
    assert query["selected_columns"][-2:] == [
        "track_sample_row_id",
        "cumulative_path_distance_mm",
    ]


@pytest.mark.parametrize(
    ("gap", "mixed_identity"),
    ((True, False), (False, True)),
)
def test_terminal_distance_reducer_fails_closed_on_axis_or_identity_divergence(
    gap: bool,
    mixed_identity: bool,
):
    with pytest.raises(
        ValidatedBehaviorGroupStatisticsError,
        match="constant source identity and a unique gapless order axis",
    ):
        _compute_distance_session(gap=gap, mixed_identity=mixed_identity)


def test_distance_family_round_trips_through_shared_static_and_interactive_views(
    tmp_path: Path,
):
    frames = _frames()
    frames["provider_motion_samples"] = _provider_motion_rows()
    specs = metric_specs_for_families(("distance_traveled",))
    result = compute_validated_behavior_group_statistics(
        _FakeDataset(frames),
        ValidatedBehaviorGroupStatisticsConfig(
            statistics_run_id="fixture-distance-view-v1",
            metric_specs=specs,
            bootstrap_iterations=0,
            permutation_iterations=0,
            confidence_level=0.95,
            minimum_recordings=2,
            random_seed=17,
        ),
    )
    statistics_dir = tmp_path / "distance-stats"
    write_validated_behavior_group_statistics_sandbox(result, statistics_dir)
    source = ValidatedBehaviorStatisticsViewSource.open(statistics_dir)
    assert [item.view_id for item in available_statistics_views(source)] == [
        "distance_traveled"
    ]
    payload = build_statistics_view_payload(source, "distance_traveled")
    validate_statistics_view_payload(payload)
    assert len(payload["metric_catalog"]) == 4
    assert len(payload["recording_rows"]) == 30

    static = render_statistics_view(payload)
    static.canvas.draw()
    assert len([axis for axis in static.axes if axis.get_visible()]) == 4

    interactive = validated_behavior_statistics_figure(
        payload,
        metric_id="distance_traveled.session_total_path_mm",
    )
    assert any(
        "All" in (tuple(trace.x) if trace.x is not None else ())
        for trace in interactive.data
    )


def test_recording_level_descriptive_and_paired_results_are_deterministic():
    first = _compute()
    second = _compute()
    assert first.recording_values == second.recording_values
    assert first.descriptive_statistics == second.descriptive_statistics
    assert first.paired_contrasts == second.paired_contrasts
    assert len(first.recording_values) == 9
    assert len(first.descriptive_statistics) == 3
    assert len(first.paired_contrasts) == 3

    pre = next(
        row for row in first.descriptive_statistics if row["condition"] == "chaser_pre"
    )
    assert pre["parent_recording_count"] == 3
    assert pre["finite_recording_count"] == 3
    assert pre["mean"] == pytest.approx(2.0)
    training_pre = next(
        row for row in first.paired_contrasts if row["contrast_id"] == "training-pre"
    )
    assert training_pre["paired_unit_count"] == 3
    assert training_pre["mean_difference"] == pytest.approx(2.0)
    assert training_pre["difference_direction"] == "condition_b_minus_condition_a"
    assert training_pre["analysis_status"] == "exploratory"
    assert training_pre["q_value"] is not None


def test_nonfinite_recording_is_retained_as_explicit_exclusion():
    result = _compute(nonfinite=True)
    training = next(
        row
        for row in result.descriptive_statistics
        if row["condition"] == "chaser_training"
    )
    assert training["contributor_recording_count"] == 3
    assert training["finite_recording_count"] == 2
    assert training["excluded_nonfinite_recording_count"] == 1
    training_pre = next(
        row for row in result.paired_contrasts if row["contrast_id"] == "training-pre"
    )
    assert training_pre["eligible_recording_count"] == 3
    assert training_pre["paired_unit_count"] == 2
    assert training_pre["excluded_unit_count"] == 1


def test_duplicate_recording_condition_rows_fail_closed():
    with pytest.raises(
        ValidatedBehaviorGroupStatisticsError,
        match="not one exact row",
    ):
        _compute(duplicate=True)


def test_analysis_unit_alias_fails_closed():
    frames = _frames()
    frames["cohort_recordings"] = frames["cohort_recordings"].with_columns(
        pl.when(pl.col("recording_id") == "r3")
        .then(pl.lit("reused-subject"))
        .otherwise(pl.col("analysis_unit_id"))
        .alias("analysis_unit_id")
    )
    dataset = _FakeDataset(frames)
    config = ValidatedBehaviorGroupStatisticsConfig(
        statistics_run_id="fixture-stats-v1",
        metric_specs=(_metric(),),
        bootstrap_iterations=10,
        permutation_iterations=10,
        minimum_recordings=2,
    )
    with pytest.raises(
        ValidatedBehaviorGroupStatisticsError,
        match="aliases recording_id",
    ):
        compute_validated_behavior_group_statistics(dataset, config)


def _histogram_frames(
    *,
    overweight_first_recording: bool = False,
    ambiguous_identity: bool = False,
    invalid_bearing: bool = False,
) -> dict[str, pl.DataFrame]:
    frames = _frames()
    rows = []
    for recording_index, recording in enumerate(("r1", "r2", "r3")):
        for condition in CHASER_EPOCH_CONDITIONS:
            for role, identity in (("aggressive", "blue"), ("inert", "yellow")):
                if overweight_first_recording:
                    bearings = [5.0] * 100 if recording == "r1" else [-175.0]
                    distances = [2.0] * len(bearings)
                else:
                    bearings = [-175.0, -5.0, 5.0, 175.0]
                    distances = [2.0, 7.0, 12.0, 17.0]
                for value_index, (bearing, distance) in enumerate(
                    zip(bearings, distances, strict=True)
                ):
                    selected_identity = identity
                    if (
                        ambiguous_identity
                        and recording == "r1"
                        and condition == "chaser_pre"
                        and role == "aggressive"
                        and value_index == 0
                    ):
                        selected_identity = "unexpected-second-identity"
                    selected_bearing = bearing
                    if (
                        invalid_bearing
                        and recording == "r1"
                        and condition == "chaser_pre"
                        and role == "aggressive"
                        and value_index == 0
                    ):
                        selected_bearing = 181.0
                    rows.append(
                        {
                            "recording_id": recording,
                            "epoch_role": condition,
                            "behavior_role": role,
                            "chaser_identity": selected_identity,
                            "selection_member": True,
                            "chaser_occurrence_member": True,
                            "chaser_behavior_role_valid": True,
                            "body_bearing_valid": True,
                            "relative_physical_valid": True,
                            "body_bearing_deg": selected_bearing,
                            "relative_distance_mm": distance,
                        }
                    )
                rows.append(
                    {
                        "recording_id": recording,
                        "epoch_role": None,
                        "behavior_role": role,
                        "chaser_identity": identity,
                        "selection_member": True,
                        "chaser_occurrence_member": True,
                        "chaser_behavior_role_valid": True,
                        "body_bearing_valid": True,
                        "relative_physical_valid": True,
                        "body_bearing_deg": float(recording_index),
                        "relative_distance_mm": 1.0,
                    }
                )
    frames["body_relative_samples"] = pl.DataFrame(rows)
    return frames


def _compute_histograms(**frame_options):
    dataset = _FakeDataset(_histogram_frames(**frame_options))
    config = ValidatedBehaviorGroupStatisticsConfig(
        statistics_run_id="fixture-bearing-histograms-v2",
        metric_specs=(),
        histogram_specs=DEFAULT_VALIDATED_BEHAVIOR_HISTOGRAMS,
        bootstrap_iterations=0,
        permutation_iterations=0,
        minimum_recordings=2,
    )
    return compute_validated_behavior_group_statistics(dataset, config)


def test_recording_histograms_persist_counts_denominators_and_equal_weight():
    result = _compute_histograms(overweight_first_recording=True)
    polar = [
        row
        for row in result.recording_histogram_bins
        if row["metric_family"] == "body_bearing_polar"
    ]
    joint = [
        row
        for row in result.recording_histogram_bins
        if row["metric_family"] == "body_bearing_distance"
    ]
    assert len(polar) == 3 * 3 * 2 * 36
    assert len(joint) == 3 * 3 * 2 * 12

    front = next(
        row
        for row in result.histogram_descriptive_statistics
        if row["metric_family"] == "body_bearing_polar"
        and row["condition"] == "chaser_pre"
        and '"behavior_role":"aggressive"' in row["group_key_json"]
        and row["axis_0_bin_start"] == 0.0
    )
    assert front["mean_fraction"] == pytest.approx(1.0 / 3.0)
    assert front["source_bin_count_sum"] == 100
    assert front["source_denominator_count_sum"] == 102
    assert front["finite_recording_count"] == 3

    polar_recipe = next(
        row
        for row in result.histogram_recipes
        if row["metric_family"] == "body_bearing_polar"
    )
    assert polar_recipe["source_audit"]["null_condition_row_count"] == 18
    joint_recipe = next(
        row
        for row in result.histogram_recipes
        if row["metric_family"] == "body_bearing_distance"
    )
    distance_axis = next(
        axis for axis in joint_recipe["resolved_axes"] if axis["axis_id"] == "distance"
    )
    assert distance_axis["resolved_upper_bound"] == 5.0


def test_recording_histograms_fail_closed_on_identity_or_axis_contract():
    with pytest.raises(
        ValidatedBehaviorGroupStatisticsError,
        match="one exact histogram identity",
    ):
        _compute_histograms(ambiguous_identity=True)
    with pytest.raises(
        ValidatedBehaviorGroupStatisticsError,
        match="outside its contract",
    ):
        _compute_histograms(invalid_bearing=True)


def test_histogram_publication_and_shared_renderers_are_receipt_bound(
    tmp_path: Path,
):
    result = _compute_histograms()
    statistics_dir = tmp_path / "histogram-stats"
    manifest = write_validated_behavior_group_statistics_sandbox(result, statistics_dir)
    assert manifest["schema_version"] == 2
    assert len(manifest["histogram_recipes"]) == 2
    assert pq.read_table(
        statistics_dir / "recording_histogram_bins.parquet"
    ).num_rows == 3 * 3 * 2 * (36 + 48)

    source = ValidatedBehaviorStatisticsViewSource.open(statistics_dir)
    assert [item.view_id for item in available_statistics_views(source)] == [
        "body_bearing_polar",
        "body_bearing_distance",
    ]
    for view_id in ("body_bearing_polar", "body_bearing_distance"):
        payload = build_statistics_view_payload(source, view_id)
        validate_statistics_view_payload(payload)
        assert len(payload["histogram_recipes"]) == 1
        static = render_statistics_view(payload)
        static.canvas.draw()
        polar_axes = [axis for axis in static.axes if axis.name == "polar"]
        assert len(polar_axes) == 6
        assert all(axis.get_thetamin() == -180.0 for axis in polar_axes)
        assert all(axis.get_thetamax() == 180.0 for axis in polar_axes)
        interactive = validated_behavior_statistics_figure(payload)
        assert (
            interactive.layout.meta["statistics_manifest_sha256"]
            == source.cache_identity
        )
        assert interactive.layout.meta["histogram_recipe_sha256"] == [
            payload["histogram_recipes"][0]["histogram_recipe_sha256"]
        ]


def test_sandbox_writer_is_atomic_bound_and_non_overwriting(tmp_path: Path):
    result = _compute()
    output = tmp_path / "stats"
    manifest = write_validated_behavior_group_statistics_sandbox(result, output)
    persisted = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["record_sha256"] == manifest["record_sha256"]
    assert persisted["selector_eligible"] is False
    assert persisted["production_authority"] is False
    assert persisted["acquisition_batch_adjustment"].startswith("not_performed")
    assert pq.read_table(output / "recording_metric_values.parquet").num_rows == 9
    assert pq.read_table(output / "descriptive_statistics.parquet").num_rows == 3
    assert pq.read_table(output / "paired_contrasts.parquet").num_rows == 3
    reopened = read_validated_behavior_group_statistics_sandbox(output)
    assert reopened["record_sha256"] == manifest["record_sha256"]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_validated_behavior_group_statistics_sandbox(result, output)


def test_shared_view_payload_binds_static_and_interactive_renderers(tmp_path: Path):
    statistics_dir = tmp_path / "stats"
    write_validated_behavior_group_statistics_sandbox(_compute(), statistics_dir)
    source = ValidatedBehaviorStatisticsViewSource.open(statistics_dir)
    assert [item.view_id for item in available_statistics_views(source)] == [
        "core_behavior"
    ]
    payload = build_statistics_view_payload(source, "core_behavior")
    validate_statistics_view_payload(payload)
    assert (
        payload["source_statistics"]["statistics_manifest_sha256"]
        == source.cache_identity
    )
    assert len(payload["recording_rows"]) == 9
    assert len(payload["descriptive_rows"]) == 3
    assert len(payload["contrast_rows"]) == 3

    static_figure = render_statistics_view(payload)
    static_path = tmp_path / "core.png"
    static_figure.savefig(static_path, dpi=72, bbox_inches="tight")
    assert static_path.stat().st_size > 0

    interactive_figure = validated_behavior_statistics_figure(payload)
    assert (
        interactive_figure.layout.meta["view_payload_sha256"]
        == payload["payload_sha256"]
    )
    assert (
        interactive_figure.layout.meta["statistics_manifest_sha256"]
        == source.cache_identity
    )
    assert len(interactive_figure.data) > 0

    stale = dict(payload)
    stale["label"] = "mutated"
    with pytest.raises(ValidatedBehaviorStatisticsViewError, match="digest is stale"):
        validate_statistics_view_payload(stale)


def test_static_report_is_atomic_digest_bound_and_non_overwriting(tmp_path: Path):
    statistics_dir = tmp_path / "stats"
    write_validated_behavior_group_statistics_sandbox(_compute(), statistics_dir)
    source = ValidatedBehaviorStatisticsViewSource.open(statistics_dir)
    report_dir = tmp_path / "report"
    manifest = render_validated_behavior_statistics_report(
        source,
        report_run_id="fixture-report-v1",
        output_dir=report_dir,
        view_ids=("core_behavior",),
        dpi=72,
    )
    reopened = read_validated_behavior_statistics_report(report_dir)
    assert reopened["record_sha256"] == manifest["record_sha256"]
    assert (
        reopened["source_statistics"]["statistics_manifest_sha256"]
        == source.cache_identity
    )
    assert (
        reopened["view_payloads"][0]["view_payload_sha256"]
        == manifest["artifacts"][0]["view_payload_sha256"]
    )
    assert (report_dir / "index.html").is_file()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        render_validated_behavior_statistics_report(
            source,
            report_run_id="fixture-report-v1",
            output_dir=report_dir,
            view_ids=("core_behavior",),
            dpi=72,
        )

    artifact = report_dir / str(manifest["artifacts"][0]["path"])
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(
        ValidatedBehaviorStatisticsReportError,
        match="size is stale",
    ):
        read_validated_behavior_statistics_report(report_dir)
