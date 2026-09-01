"""Unit tests for aggressive-vs-inert validated-behavior role contrasts."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from fisheye.group_statistics.paired import benjamini_hochberg
from fisheye.group_statistics.validated_behavior_role_contrasts import (
    ACQUISITION_BATCH_ADJUSTMENT,
    ANALYSIS_STATUS,
    BOUT_ASSOCIATION_PARQUET_NAME,
    BOUT_ASSOCIATION_SPEC_VERSION,
    CENSORING_NONE,
    CENSORING_PRIMARY,
    DISTANCE_BIN_METRICS,
    DISTANCE_BIN_PARQUET_NAME,
    DISTANCE_BIN_SPEC_VERSION,
    EPOCH_ROLES,
    IBI_CELLS_PARQUET_NAME,
    IBI_SHAPE_SPEC_VERSION,
    IBI_STAT_COLUMNS,
    PRIMARY_EPOCH_ROLE,
    QUANTILE_SHAPE_PARQUET_NAME,
    QUANTILE_SHAPE_SPEC_VERSION,
    RoleContrastInputError,
    RoleContrastParameters,
    SECONDARY_EPOCH_ROLES,
    SHAPE_QUANTILES,
    SOURCE_TABLE_METRICS,
    SPEC_VERSION,
    TOWARD_FRACTION_METRIC,
    build_ibi_cells,
    compute_bout_association_contrast_rows,
    compute_distance_bin_contrast_rows,
    compute_ibi_shape_contrast_rows,
    compute_quantile_shape_contrast_rows,
    compute_role_contrast_rows,
    read_role_contrasts_manifest,
    validate_recording_fps,
    write_role_contrasts,
)

RADIAL_METRICS = SOURCE_TABLE_METRICS["radial_near_field_summary"]
QUADRANT_METRICS = SOURCE_TABLE_METRICS["same_quadrant_occupancy"]

PARAMS = RoleContrastParameters(
    bootstrap_iterations=200,
    permutation_iterations=500,
    seed=7,
)


def _rows(
    recording_values: dict[str, dict[str, float]],
    *,
    metrics: tuple[str, ...],
    provider_roles: tuple[str, ...] = ("detection", "keypoint"),
    epoch_roles: tuple[str, ...] = EPOCH_ROLES,
) -> pl.DataFrame:
    """One row per recording x provider x epoch x role; values shared across
    provider/epoch cells so per-cell expectations are easy to reason about."""

    rows = []
    for recording_id, role_values in recording_values.items():
        for provider_role in provider_roles:
            for epoch_role in epoch_roles:
                for behavior_role, value in role_values.items():
                    row = {
                        "recording_id": recording_id,
                        "provider_role": provider_role,
                        "epoch_role": epoch_role,
                        "behavior_role": behavior_role,
                    }
                    for metric in metrics:
                        row[metric] = value
                    rows.append(row)
    return pl.DataFrame(rows)


def _frames(
    recording_values: dict[str, dict[str, float]],
    **kwargs,
) -> dict[str, pl.DataFrame]:
    return {
        "radial_near_field_summary": _rows(
            recording_values, metrics=RADIAL_METRICS, **kwargs
        ),
        "same_quadrant_occupancy": _rows(
            recording_values, metrics=QUADRANT_METRICS, **kwargs
        ),
    }


def test_pairing_computes_aggressive_minus_inert_per_recording() -> None:
    frames = _frames(
        {
            "rec_a": {"aggressive": 0.5, "inert": 0.1},
            "rec_b": {"aggressive": 0.3, "inert": 0.2},
            "rec_c": {"aggressive": 0.2, "inert": 0.4},
        }
    )
    results = compute_role_contrast_rows(frames, parameters=PARAMS)

    expected_rows = len(RADIAL_METRICS + QUADRANT_METRICS) * 2 * len(EPOCH_ROLES)
    assert results.height == expected_rows

    diffs = [0.4, 0.1, -0.2]
    for row in results.iter_rows(named=True):
        assert row["eligible_recording_count"] == 3
        assert row["paired_unit_count"] == 3
        assert row["excluded_nonfinite_count"] == 0
        assert row["mean_difference"] == pytest.approx(np.mean(diffs))
        assert row["median_difference"] == pytest.approx(np.median(diffs))
        assert row["mean_aggressive"] == pytest.approx(np.mean([0.5, 0.3, 0.2]))
        assert row["mean_inert"] == pytest.approx(np.mean([0.1, 0.2, 0.4]))
        assert row["contrast"] == "aggressive_minus_inert"
        assert row["spec_version"] == SPEC_VERSION
        assert row["analysis_status"] == ANALYSIS_STATUS
        assert row["acquisition_batch_adjustment"] == ACQUISITION_BATCH_ADJUSTMENT


def test_pairing_ignores_recordings_missing_one_role() -> None:
    frames = _frames(
        {
            "rec_a": {"aggressive": 0.5, "inert": 0.1},
            "rec_b": {"aggressive": 0.3, "inert": 0.2},
            "rec_only_aggressive": {"aggressive": 0.9},
        }
    )
    results = compute_role_contrast_rows(frames, parameters=PARAMS)
    for row in results.iter_rows(named=True):
        assert row["eligible_recording_count"] == 3
        assert row["paired_unit_count"] == 2
        assert row["excluded_nonfinite_count"] == 1
        assert row["mean_difference"] == pytest.approx(0.25)


def test_sign_flip_exact_p_value_on_constructed_case() -> None:
    # Four recordings with a constant difference of +0.1: the exact sign-flip
    # null has 2**4 = 16 equally likely mean magnitudes and only the two
    # all-same-sign assignments reach |mean| >= 0.1, so p = 2/16 = 0.125.
    frames = _frames(
        {
            "rec_a": {"aggressive": 0.2, "inert": 0.1},
            "rec_b": {"aggressive": 0.3, "inert": 0.2},
            "rec_c": {"aggressive": 0.4, "inert": 0.3},
            "rec_d": {"aggressive": 0.5, "inert": 0.4},
        }
    )
    results = compute_role_contrast_rows(frames, parameters=PARAMS)
    for row in results.iter_rows(named=True):
        assert row["test_method"] == "paired_sign_flip_exact"
        assert row["p_value"] == pytest.approx(2.0 / 16.0)
        assert row["permutation_iterations"] == 16


def test_random_sign_flip_used_above_exact_threshold_and_deterministic() -> None:
    rng = np.random.default_rng(11)
    values = {
        f"rec_{index:02d}": {
            "aggressive": float(0.3 + 0.05 * rng.standard_normal()),
            "inert": float(0.1 + 0.05 * rng.standard_normal()),
        }
        for index in range(25)
    }
    frames = _frames(values, epoch_roles=(PRIMARY_EPOCH_ROLE,))
    first = compute_role_contrast_rows(frames, parameters=PARAMS)
    second = compute_role_contrast_rows(frames, parameters=PARAMS)
    for row in first.iter_rows(named=True):
        assert row["test_method"] == "paired_sign_flip_random"
        assert row["permutation_iterations"] == 500
    assert first.equals(second)


def test_bh_families_are_per_source_table_and_epoch() -> None:
    rng = np.random.default_rng(3)
    values = {
        f"rec_{index}": {
            "aggressive": float(rng.normal(0.4, 0.05)),
            "inert": float(rng.normal(0.1, 0.05)),
        }
        for index in range(6)
    }
    frames = _frames(values)
    results = compute_role_contrast_rows(frames, parameters=PARAMS)

    for (family,), subset in results.group_by("multiplicity_family"):
        source_table, epoch_role = family.split(":")
        assert epoch_role in EPOCH_ROLES
        expected_size = len(SOURCE_TABLE_METRICS[source_table]) * 2
        assert subset.height == expected_size
        assert subset["family_size"].unique().to_list() == [expected_size]
        assert expected_size > 1  # size-1 FDR families are forbidden
        expected_q = benjamini_hochberg(subset["p_value"].to_list())
        for actual, expected in zip(subset["q_value"].to_list(), expected_q):
            assert actual == pytest.approx(expected)


def test_nonfinite_values_are_excluded_and_counted() -> None:
    frames = _frames(
        {
            "rec_a": {"aggressive": 0.5, "inert": 0.1},
            "rec_b": {"aggressive": 0.3, "inert": 0.2},
            "rec_nan": {"aggressive": float("nan"), "inert": 0.3},
            "rec_inf": {"aggressive": 0.4, "inert": float("inf")},
        }
    )
    results = compute_role_contrast_rows(frames, parameters=PARAMS)
    for row in results.iter_rows(named=True):
        assert row["eligible_recording_count"] == 4
        assert row["paired_unit_count"] == 2
        assert row["excluded_nonfinite_count"] == 2
        assert row["mean_difference"] == pytest.approx(0.25)
        assert math.isfinite(row["ci_low"]) and math.isfinite(row["ci_high"])


def test_pre_and_post_rows_carry_park_position_asymmetry_caveat() -> None:
    frames = _frames(
        {
            "rec_a": {"aggressive": 0.5, "inert": 0.1},
            "rec_b": {"aggressive": 0.3, "inert": 0.2},
        }
    )
    results = compute_role_contrast_rows(frames, parameters=PARAMS)
    primary = results.filter(pl.col("epoch_role") == PRIMARY_EPOCH_ROLE)
    secondary = results.filter(pl.col("epoch_role").is_in(SECONDARY_EPOCH_ROLES))
    assert primary.height > 0 and secondary.height > 0
    assert primary["park_position_asymmetry"].unique().to_list() == [False]
    assert primary["contrast_tier"].unique().to_list() == ["primary"]
    assert secondary["park_position_asymmetry"].unique().to_list() == [True]
    assert secondary["contrast_tier"].unique().to_list() == ["secondary"]
    assert set(secondary["epoch_role"].unique().to_list()) == set(
        SECONDARY_EPOCH_ROLES
    )


def test_duplicate_grain_rows_are_rejected() -> None:
    frames = _frames({"rec_a": {"aggressive": 0.5, "inert": 0.1}})
    duplicated = pl.concat(
        [
            frames["radial_near_field_summary"],
            frames["radial_near_field_summary"].head(1),
        ]
    )
    frames["radial_near_field_summary"] = duplicated
    with pytest.raises(RoleContrastInputError, match="duplicated"):
        compute_role_contrast_rows(frames, parameters=PARAMS)


def test_unexpected_behavior_role_is_rejected() -> None:
    frames = _frames({"rec_a": {"aggressive": 0.5, "inert": 0.1}})
    frames["same_quadrant_occupancy"] = frames["same_quadrant_occupancy"].with_columns(
        pl.lit("random_non_chasing").alias("behavior_role")
    )
    with pytest.raises(RoleContrastInputError, match="behavior_role"):
        compute_role_contrast_rows(frames, parameters=PARAMS)


def test_write_and_read_back_manifest(tmp_path) -> None:
    frames = _frames(
        {
            "rec_a": {"aggressive": 0.5, "inert": 0.1},
            "rec_b": {"aggressive": 0.3, "inert": 0.2},
        }
    )
    results = compute_role_contrast_rows(frames, parameters=PARAMS)
    out = tmp_path / "contrasts"
    write_role_contrasts(
        out,
        results,
        source_export_run_id="test-run",
        source_export_manifest_sha256="ab" * 32,
        source_export_root="/somewhere/publication",
        parameters=PARAMS,
    )
    manifest = read_role_contrasts_manifest(out)
    assert manifest["spec_version"] == SPEC_VERSION
    assert manifest["analysis_status"] == ANALYSIS_STATUS
    assert manifest["acquisition_batch_adjustment"] == ACQUISITION_BATCH_ADJUSTMENT
    assert manifest["row_count"] == results.height
    assert manifest["source_export"]["export_run_id"] == "test-run"
    assert manifest["source_export"]["export_manifest_record_sha256"] == "ab" * 32
    assert set(manifest["files"]) == {"role_contrasts.parquet"}
    assert all(size > 1 for size in manifest["multiplicity_family_sizes"].values())

    round_trip = pl.read_parquet(out / "role_contrasts.parquet")
    assert round_trip.equals(results)

    with pytest.raises(FileExistsError):
        write_role_contrasts(
            out,
            results,
            source_export_run_id="test-run",
            source_export_manifest_sha256="ab" * 32,
            source_export_root="/somewhere/publication",
            parameters=PARAMS,
        )
    write_role_contrasts(
        out,
        results,
        source_export_run_id="test-run",
        source_export_manifest_sha256="ab" * 32,
        source_export_root="/somewhere/publication",
        parameters=PARAMS,
        overwrite=True,
    )


# --- GROUP A1: distance-binned bout response ---------------------------------


def _distance_bin_frame() -> pl.DataFrame:
    bins = ((0, 0.0, 8.0), (1, 8.0, 16.0))
    values: dict[str, dict[str, dict[int, float]]] = {
        "r1": {"aggressive": {0: 1.0, 1: 5.0}, "inert": {0: 0.5, 1: 1.0}},
        "r2": {"aggressive": {0: 2.0, 1: 6.0}, "inert": {0: 1.0, 1: 1.5}},
        "r3": {"aggressive": {0: 3.0, 1: 7.0}, "inert": {0: float("nan"), 1: 2.0}},
        "r4": {"aggressive": {0: 4.0, 1: 8.0}, "inert": {1: 2.5}},
    }
    rows = []
    for recording_id, role_map in values.items():
        for epoch_role in EPOCH_ROLES:
            for behavior_role, bin_map in role_map.items():
                for index, start, end in bins:
                    if index not in bin_map:
                        continue
                    row = {
                        "recording_id": recording_id,
                        "semantic_role": epoch_role,
                        "behavior_role": behavior_role,
                        "distance_bin_index": index,
                        "distance_bin_start_mm": start,
                        "distance_bin_end_mm": end,
                    }
                    for metric in DISTANCE_BIN_METRICS:
                        row[metric] = bin_map[index]
                    rows.append(row)
    return pl.DataFrame(rows)


def test_distance_bin_pairing_with_missing_bins_and_nans() -> None:
    results = compute_distance_bin_contrast_rows(
        _distance_bin_frame(), parameters=PARAMS, min_paired=2
    )
    assert results.height == 3 * 2 * len(DISTANCE_BIN_METRICS)
    for row in results.iter_rows(named=True):
        assert row["spec_version"] == DISTANCE_BIN_SPEC_VERSION
        assert row["analysis_status"] == ANALYSIS_STATUS
        assert row["acquisition_batch_adjustment"] == ACQUISITION_BATCH_ADJUSTMENT
        assert row["status"] == "computed"
        if row["distance_bin_index"] == 0:
            # r3 has a NaN inert value, r4 has no inert row at all.
            assert row["eligible_recording_count"] == 4
            assert row["paired_unit_count"] == 2
            assert row["excluded_nonfinite_count"] == 2
            assert row["mean_difference"] == pytest.approx(0.75)
        else:
            assert row["eligible_recording_count"] == 4
            assert row["paired_unit_count"] == 4
            assert row["mean_difference"] == pytest.approx(4.75)
        expected_secondary = row["epoch_role"] != PRIMARY_EPOCH_ROLE
        assert row["park_position_asymmetry"] is expected_secondary
        assert row["contrast_tier"] == (
            "secondary" if expected_secondary else "primary"
        )


def test_distance_bin_min_paired_skip_path() -> None:
    results = compute_distance_bin_contrast_rows(
        _distance_bin_frame(), parameters=PARAMS, min_paired=3
    )
    skipped = results.filter(pl.col("distance_bin_index") == 0)
    computed = results.filter(pl.col("distance_bin_index") == 1)
    for row in skipped.iter_rows(named=True):
        assert row["status"] == "skipped"
        assert row["skip_reason"] == "paired_unit_count<3"
        assert row["p_value"] is None
        assert row["q_value"] is None
        assert row["test_method"] == "skipped"
        # Only the 5 bin-1 metrics are tested within each epoch family.
        assert row["family_size"] == len(DISTANCE_BIN_METRICS)
    for row in computed.iter_rows(named=True):
        assert row["status"] == "computed"
        assert row["p_value"] is not None
        assert row["q_value"] is not None
        assert row["family_size"] == len(DISTANCE_BIN_METRICS)


# --- GROUP A2: bout/chaser associations --------------------------------------


def _association_frame() -> pl.DataFrame:
    rows: list[dict] = []

    def add(
        recording_id: str,
        behavior_role: str,
        onset: float,
        delta: float,
        toward: bool,
        *,
        semantic_role: str = PRIMARY_EPOCH_ROLE,
        base_valid: bool = True,
        directed_valid: bool = True,
    ) -> None:
        rows.append(
            {
                "recording_id": recording_id,
                "semantic_role": semantic_role,
                "behavior_role": behavior_role,
                "base_valid": base_valid,
                "directed_valid": directed_valid,
                "distance_at_onset_mm": onset,
                "delta_distance_mm": delta,
                "turn_toward_chaser": toward,
            }
        )

    # Near-onset toward fractions: r1 aggressive 4/6 vs inert 1/5,
    # r2 aggressive 3/6 vs inert 2/5.
    toward_counts = {"r1": (4, 1), "r2": (3, 2)}
    for recording_id, (aggressive_toward, inert_toward) in toward_counts.items():
        for i in range(6):
            add(recording_id, "aggressive", 5.0, -1.0, i < aggressive_toward)
        for i in range(5):
            add(recording_id, "inert", 5.0, 0.5, i < inert_toward)
    # r3 has only 3 near aggressive bouts: below the default min_bouts of 5.
    for i in range(3):
        add("r3", "aggressive", 5.0, -1.0, True)
    # Onset bin 10-20 mm delta medians: aggressive 3.0/4.0 vs inert 0.5/1.0.
    bin_medians = {"r1": (3.0, 0.5), "r2": (4.0, 1.0)}
    for recording_id, (aggressive_median, inert_median) in bin_medians.items():
        for offset in (-1.0, -0.5, 0.0, 0.5, 1.0):
            add(recording_id, "aggressive", 15.0, aggressive_median + offset, False)
            add(recording_id, "inert", 15.0, inert_median + offset, False)
    # Rows the filters must drop.
    add("r1", "aggressive", 5.0, -9.0, True, base_valid=False)
    add("r1", "aggressive", 5.0, -9.0, True, directed_valid=False)
    add("r1", "aggressive", float("nan"), -9.0, True)
    add("r1", "aggressive", 5.0, -9.0, True, semantic_role="chaser_pre")
    return pl.DataFrame(rows)


def test_association_toward_fraction_and_min_bouts() -> None:
    results = compute_bout_association_contrast_rows(
        _association_frame(), parameters=PARAMS, min_bouts=5, min_paired=2
    )
    toward = results.filter(pl.col("metric") == TOWARD_FRACTION_METRIC)
    assert toward.height == 1
    row = toward.row(0, named=True)
    # r3's 3-bout aggressive cell is excluded by min_bouts, so it is not
    # even eligible; the invalid/pre rows must not raise the fractions.
    assert row["eligible_recording_count"] == 2
    assert row["paired_unit_count"] == 2
    diffs = [4 / 6 - 1 / 5, 3 / 6 - 2 / 5]
    assert row["mean_difference"] == pytest.approx(sum(diffs) / 2)
    assert row["onset_bin_start_mm"] is None
    assert row["onset_bin_end_mm"] is None
    assert row["epoch_role"] == PRIMARY_EPOCH_ROLE
    assert row["spec_version"] == BOUT_ASSOCIATION_SPEC_VERSION

    relaxed = compute_bout_association_contrast_rows(
        _association_frame(), parameters=PARAMS, min_bouts=3, min_paired=2
    )
    relaxed_row = relaxed.filter(pl.col("metric") == TOWARD_FRACTION_METRIC).row(
        0, named=True
    )
    # With min_bouts=3 the r3 aggressive cell exists but has no inert partner.
    assert relaxed_row["eligible_recording_count"] == 3
    assert relaxed_row["paired_unit_count"] == 2


def test_association_onset_bin_medians_and_skips() -> None:
    results = compute_bout_association_contrast_rows(
        _association_frame(), parameters=PARAMS, min_bouts=5, min_paired=2
    )
    bin_10_20 = results.filter(
        (pl.col("onset_bin_start_mm") == 10.0) & (pl.col("onset_bin_end_mm") == 20.0)
    )
    assert bin_10_20.height == 1
    row = bin_10_20.row(0, named=True)
    assert row["status"] == "computed"
    assert row["mean_difference"] == pytest.approx(((3.0 - 0.5) + (4.0 - 1.0)) / 2)
    # The 0-10 bin is fed by the near-onset bouts; farther bins are empty.
    empty_bins = results.filter(pl.col("onset_bin_start_mm") >= 20.0)
    assert empty_bins.height == 3
    for empty_row in empty_bins.iter_rows(named=True):
        assert empty_row["status"] == "skipped"
        assert empty_row["p_value"] is None
    tested = results.filter(pl.col("status") == "computed")
    assert set(results["family_size"].unique().to_list()) == {tested.height}


# --- GROUP B: quantile distribution shape ------------------------------------


def _quantile_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    bout_rows: list[dict] = []

    def add_bout(
        recording_id: str, frame_id: int, value: float, tortuosity: float
    ) -> None:
        bout_rows.append(
            {
                "recording_id": recording_id,
                "bout_row_id": len(bout_rows),
                "start_acquisition_frame_id": frame_id,
                "duration_s": value,
                "path_length_mm": value,
                "peak_speed_mm_s": value,
                "tortuosity": tortuosity,
            }
        )

    shifts = {"r1": 1.0, "r2": 2.0}
    for recording_id, shift in shifts.items():
        for i in range(30):
            value = float(i + 1)
            add_bout(recording_id, 10 + i, value, value)  # chaser_pre
            add_bout(recording_id, 110 + i, value + 0.5, value + 0.5)  # training
            post_value = value + shift
            if recording_id == "r2" and i < 28:
                post_tortuosity = float("nan")  # only 2 finite: invalid cell
            elif recording_id == "r1" and i < 3:
                post_tortuosity = float("nan")  # 27 finite: valid, 3 excluded
            else:
                post_tortuosity = post_value
            add_bout(recording_id, 210 + i, post_value, post_tortuosity)
    epochs = pl.DataFrame(
        [
            {
                "recording_id": recording_id,
                "analysis_role": role,
                "start_frame": start,
                "end_frame_exclusive": end,
            }
            for recording_id in shifts
            for role, start, end in (
                ("chaser_pre", 0, 100),
                ("chaser_training", 100, 200),
                ("chaser_post", 200, 300),
            )
        ]
    )
    return pl.DataFrame(bout_rows), epochs


def test_quantile_shape_shift_and_quantile_values() -> None:
    bouts, epochs = _quantile_frames()
    results = compute_quantile_shape_contrast_rows(
        bouts, epochs, parameters=PARAMS, min_bouts_per_cell=5, min_paired=2
    )
    assert results.height == 4 * 3 * len(SHAPE_QUANTILES)
    duration_pre_post = results.filter(
        (pl.col("metric") == "duration_s")
        & (pl.col("contrast") == "chaser_post_minus_chaser_pre")
    )
    assert duration_pre_post.height == len(SHAPE_QUANTILES)
    for row in duration_pre_post.iter_rows(named=True):
        # Post is pre shifted by +1 (r1) and +2 (r2): every quantile moves by
        # the shift, so the paired mean difference is 1.5 at every quantile.
        assert row["spec_version"] == QUANTILE_SHAPE_SPEC_VERSION
        assert row["paired_unit_count"] == 2
        assert row["mean_difference"] == pytest.approx(1.5)
        assert row["nonfinite_bout_values_excluded"] == 0
    q50 = duration_pre_post.filter(pl.col("quantile") == 0.50).row(0, named=True)
    assert q50["mean_epoch_a"] == pytest.approx(float(np.quantile(range(1, 31), 0.5)))


def test_quantile_nonfinite_tortuosity_accounting() -> None:
    bouts, epochs = _quantile_frames()
    results = compute_quantile_shape_contrast_rows(
        bouts, epochs, parameters=PARAMS, min_bouts_per_cell=5, min_paired=2
    )
    tortuosity_pre_post = results.filter(
        (pl.col("metric") == "tortuosity")
        & (pl.col("contrast") == "chaser_post_minus_chaser_pre")
    )
    for row in tortuosity_pre_post.iter_rows(named=True):
        # r2's post cell has only 2 finite tortuosity values (< 5): the cell
        # is excluded, leaving one pair, which is below min_paired.
        assert row["eligible_recording_count"] == 2
        assert row["paired_unit_count"] == 1
        assert row["excluded_nonfinite_count"] == 1
        assert row["status"] == "skipped"
        assert row["p_value"] is None
        # r1's post cell dropped 3 NaN bout values.
        assert row["nonfinite_bout_values_excluded"] == 3
    tortuosity_family = results.filter(pl.col("metric") == "tortuosity")
    tested = tortuosity_family.filter(pl.col("status") == "computed")
    assert tested.height == 5  # only chaser_training_minus_chaser_pre survives
    assert set(tortuosity_family["family_size"].unique().to_list()) == {5}
    assert set(tortuosity_family["multiplicity_family"].unique().to_list()) == {
        "canonical_swim_bouts:tortuosity"
    }


def test_quantile_bh_families_are_per_metric() -> None:
    bouts, epochs = _quantile_frames()
    results = compute_quantile_shape_contrast_rows(
        bouts, epochs, parameters=PARAMS, min_bouts_per_cell=5, min_paired=2
    )
    for (family,), subset in results.group_by("multiplicity_family"):
        metric = family.split(":")[1]
        assert subset["metric"].unique().to_list() == [metric]
        tested = subset.filter(pl.col("p_value").is_not_null())
        expected_q = benjamini_hochberg(tested["p_value"].to_list())
        for actual, expected in zip(tested["q_value"].to_list(), expected_q):
            assert actual == pytest.approx(expected)


def test_quantile_epoch_assignment_boundary() -> None:
    bout_rows = []
    values = {"chaser_pre": (94, 1.0), "chaser_training": (100, 11.0)}
    for role, (first_frame, first_value) in values.items():
        for i in range(6):
            bout_rows.append(
                {
                    "recording_id": "r1",
                    "bout_row_id": len(bout_rows),
                    "start_acquisition_frame_id": first_frame + i,
                    "duration_s": first_value + i,
                    "path_length_mm": first_value + i,
                    "peak_speed_mm_s": first_value + i,
                    "tortuosity": first_value + i,
                }
            )
    # chaser_post bouts so post cells exist too.
    for i in range(6):
        bout_rows.append(
            {
                "recording_id": "r1",
                "bout_row_id": len(bout_rows),
                "start_acquisition_frame_id": 200 + i,
                "duration_s": 21.0 + i,
                "path_length_mm": 21.0 + i,
                "peak_speed_mm_s": 21.0 + i,
                "tortuosity": 21.0 + i,
            }
        )
    epochs = pl.DataFrame(
        [
            {
                "recording_id": "r1",
                "analysis_role": role,
                "start_frame": start,
                "end_frame_exclusive": end,
            }
            for role, start, end in (
                ("chaser_pre", 0, 100),
                ("chaser_training", 100, 200),
                ("chaser_post", 200, 300),
            )
        ]
    )
    results = compute_quantile_shape_contrast_rows(
        pl.DataFrame(bout_rows),
        epochs,
        parameters=PARAMS,
        min_bouts_per_cell=5,
        min_paired=1,
    )
    row = results.filter(
        (pl.col("metric") == "duration_s")
        & (pl.col("contrast") == "chaser_training_minus_chaser_pre")
        & (pl.col("quantile") == 0.50)
    ).row(0, named=True)
    # Frame 99 belongs to chaser_pre; frame 100 to chaser_training.
    assert row["mean_epoch_a"] == pytest.approx(3.5)
    assert row["mean_epoch_b"] == pytest.approx(13.5)

    overlapping = pl.concat([epochs, epochs.head(1)])
    with pytest.raises(RoleContrastInputError, match="more than one"):
        compute_quantile_shape_contrast_rows(
            pl.DataFrame(bout_rows),
            overlapping,
            parameters=PARAMS,
            min_bouts_per_cell=5,
            min_paired=1,
        )


def test_manifest_with_extra_tables(tmp_path) -> None:
    frames = _frames(
        {
            "rec_a": {"aggressive": 0.5, "inert": 0.1},
            "rec_b": {"aggressive": 0.3, "inert": 0.2},
        }
    )
    results = compute_role_contrast_rows(frames, parameters=PARAMS)
    distance_bins = compute_distance_bin_contrast_rows(
        _distance_bin_frame(), parameters=PARAMS, min_paired=2
    )
    associations = compute_bout_association_contrast_rows(
        _association_frame(), parameters=PARAMS, min_bouts=5, min_paired=2
    )
    bouts, epochs = _quantile_frames()
    quantile_shape = compute_quantile_shape_contrast_rows(
        bouts, epochs, parameters=PARAMS, min_bouts_per_cell=5, min_paired=2
    )
    out = tmp_path / "bundle"
    write_role_contrasts(
        out,
        results,
        source_export_run_id="test-run",
        source_export_manifest_sha256="ab" * 32,
        source_export_root="/somewhere/publication",
        parameters=PARAMS,
        extra_tables={
            DISTANCE_BIN_PARQUET_NAME: (distance_bins, DISTANCE_BIN_SPEC_VERSION),
            BOUT_ASSOCIATION_PARQUET_NAME: (
                associations,
                BOUT_ASSOCIATION_SPEC_VERSION,
            ),
            QUANTILE_SHAPE_PARQUET_NAME: (
                quantile_shape,
                QUANTILE_SHAPE_SPEC_VERSION,
            ),
        },
        thresholds={
            "min_paired": 2,
            "min_association_bouts": 5,
            "min_bouts_per_cell": 5,
        },
    )
    manifest = read_role_contrasts_manifest(out)
    expected_files = {
        "role_contrasts.parquet",
        DISTANCE_BIN_PARQUET_NAME,
        BOUT_ASSOCIATION_PARQUET_NAME,
        QUANTILE_SHAPE_PARQUET_NAME,
    }
    assert set(manifest["files"]) == expected_files
    assert set(manifest["tables"]) == expected_files
    assert (
        manifest["tables"][DISTANCE_BIN_PARQUET_NAME]["spec_version"]
        == DISTANCE_BIN_SPEC_VERSION
    )
    total = (
        results.height
        + distance_bins.height
        + associations.height
        + quantile_shape.height
    )
    assert manifest["row_count"] == total
    assert manifest["thresholds"] == {
        "min_paired": 2,
        "min_association_bouts": 5,
        "min_bouts_per_cell": 5,
    }
    for name, frame in (
        (DISTANCE_BIN_PARQUET_NAME, distance_bins),
        (BOUT_ASSOCIATION_PARQUET_NAME, associations),
        (QUANTILE_SHAPE_PARQUET_NAME, quantile_shape),
    ):
        assert pl.read_parquet(out / name).equals(frame)


# --- GROUP C: inter-bout-interval shape with dropout censoring ----------------


def _ibi_windows(recording_id: str = "r1") -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "recording_id": recording_id,
                "analysis_role": role,
                "start_frame": start,
                "end_frame_exclusive": end,
            }
            for role, start, end in (
                ("chaser_pre", 0, 10_000),
                ("chaser_training", 10_000, 20_000),
                ("chaser_post", 20_000, 30_000),
            )
        ]
    )


def _ibi_bouts(recording_id: str = "r1") -> pl.DataFrame:
    rows: list[dict] = []

    def add(start: int, end: int) -> None:
        rows.append(
            {
                "recording_id": recording_id,
                "track_id": 0,
                "start_acquisition_frame_id": start,
                "end_acquisition_frame_id": end,
            }
        )

    # 26 pre bouts (start i*20, end i*20+5): 25 IBIs of 15 frames = 0.15 s,
    # then one long IBI from frame 505 to the straddle bout at 9990.
    for i in range(26):
        add(i * 20, i * 20 + 5)
    add(9_990, 9_995)  # its successor starts in chaser_training: straddle pair
    # 26 training bouts: 25 IBIs of 25 frames = 0.25 s; the pair from the last
    # training bout to the first post bout straddles the boundary too.
    for i in range(26):
        add(10_000 + i * 30, 10_000 + i * 30 + 5)
    # 26 post bouts: 25 IBIs of 35 frames = 0.35 s.
    for i in range(26):
        add(20_000 + i * 40, 20_000 + i * 40 + 5)
    return pl.DataFrame(rows)


def _ibi_valid_frames(invalid: tuple[int, ...] = (107, 125)) -> np.ndarray:
    return np.setdiff1d(
        np.arange(0, 30_000, dtype=np.int64), np.asarray(invalid, dtype=np.int64)
    )


def test_ibi_censoring_and_adjacent_valid_frames() -> None:
    # Frame 107 sits inside the open interval (105, 120): that IBI must be
    # censored.  Frame 125 is a bout-start boundary frame, NOT inside any open
    # interval (the next interval is (125, 140) exclusive): nothing censored.
    cells = build_ibi_cells(
        _ibi_bouts(),
        _ibi_windows(),
        fps_by_recording={"r1": 100.0},
        valid_frames_by_recording={"r1": _ibi_valid_frames()},
        min_ibis_per_cell=5,
    )
    assert cells.height == 3 * 2
    pre_primary = cells.filter(
        (pl.col("epoch_role") == "chaser_pre")
        & (pl.col("censoring") == CENSORING_PRIMARY)
    ).row(0, named=True)
    pre_none = cells.filter(
        (pl.col("epoch_role") == "chaser_pre") & (pl.col("censoring") == CENSORING_NONE)
    ).row(0, named=True)
    assert pre_primary["n_ibis"] == 26  # 25 short + the long 505->9990 IBI
    assert pre_primary["n_censored"] == 1
    assert pre_primary["n_used"] == 25
    assert pre_primary["censored_fraction"] == pytest.approx(1 / 26)
    assert pre_none["n_ibis"] == 26
    assert pre_none["n_censored"] == 0
    assert pre_none["n_used"] == 26
    for row in (pre_primary, pre_none):
        assert row["q50_s"] == pytest.approx(0.15)
        assert row["spec_version"] == IBI_SHAPE_SPEC_VERSION
        assert row["analysis_status"] == ANALYSIS_STATUS
        assert row["acquisition_batch_adjustment"] == ACQUISITION_BATCH_ADJUSTMENT
    # The 94.85 s outlier survives both variants (all interior frames valid).
    assert pre_primary["frac_gt_5s"] == pytest.approx(1 / 25)
    assert pre_none["frac_gt_5s"] == pytest.approx(1 / 26)
    training = cells.filter(
        (pl.col("epoch_role") == "chaser_training")
        & (pl.col("censoring") == CENSORING_PRIMARY)
    ).row(0, named=True)
    assert training["n_censored"] == 0
    assert training["q50_s"] == pytest.approx(0.25)


def test_ibi_epoch_boundary_straddle_pairs_are_dropped() -> None:
    cells = build_ibi_cells(
        _ibi_bouts(),
        _ibi_windows(),
        fps_by_recording={"r1": 100.0},
        valid_frames_by_recording={"r1": _ibi_valid_frames(())},
        min_ibis_per_cell=5,
    )
    by_epoch = {
        row["epoch_role"]: row["n_ibis"]
        for row in cells.filter(pl.col("censoring") == CENSORING_NONE).iter_rows(
            named=True
        )
    }
    # The 9995->10000 and 10755->20000 pairs straddle boundaries: neither is
    # counted in any epoch cell.
    assert by_epoch == {"chaser_pre": 26, "chaser_training": 25, "chaser_post": 25}


def test_ibi_contrast_values_and_censoring_families() -> None:
    cells = build_ibi_cells(
        _ibi_bouts(),
        _ibi_windows(),
        fps_by_recording={"r1": 100.0},
        valid_frames_by_recording={"r1": _ibi_valid_frames()},
        min_ibis_per_cell=5,
    )
    results = compute_ibi_shape_contrast_rows(cells, parameters=PARAMS, min_paired=1)
    assert results.height == 2 * len(IBI_STAT_COLUMNS) * 3
    q50_pre_post = results.filter(
        (pl.col("metric") == "q50_s")
        & (pl.col("contrast") == "chaser_post_minus_chaser_pre")
    )
    assert q50_pre_post.height == 2
    for row in q50_pre_post.iter_rows(named=True):
        assert row["paired_unit_count"] == 1
        assert row["mean_difference"] == pytest.approx(0.35 - 0.15)
    for (family,), subset in results.group_by("multiplicity_family"):
        assert family in {
            f"ibi_shape:{CENSORING_PRIMARY}",
            f"ibi_shape:{CENSORING_NONE}",
        }
        assert subset.height == len(IBI_STAT_COLUMNS) * 3
        tested = subset.filter(pl.col("p_value").is_not_null())
        assert subset["family_size"].unique().to_list() == [tested.height]
        assert tested.height > 1
        expected_q = benjamini_hochberg(tested["p_value"].to_list())
        for actual, expected in zip(tested["q_value"].to_list(), expected_q):
            assert actual == pytest.approx(expected)


def test_ibi_fps_assertion() -> None:
    validate_recording_fps({"r1": 100.0, "r2": 100.09, "r3": 99.91})
    with pytest.raises(RoleContrastInputError, match="fps"):
        validate_recording_fps({"r1": 100.2})
    with pytest.raises(RoleContrastInputError, match="fps"):
        validate_recording_fps({"r1": 99.8})
    with pytest.raises(RoleContrastInputError, match="fps"):
        validate_recording_fps({"r1": float("nan")})
    with pytest.raises(RoleContrastInputError, match="fps"):
        build_ibi_cells(
            _ibi_bouts(),
            _ibi_windows(),
            fps_by_recording={"r1": 98.0},
            valid_frames_by_recording={"r1": _ibi_valid_frames(())},
            min_ibis_per_cell=5,
        )


def test_ibi_min_count_skip_and_cell_manifest(tmp_path) -> None:
    cells = build_ibi_cells(
        _ibi_bouts(),
        _ibi_windows(),
        fps_by_recording={"r1": 100.0},
        valid_frames_by_recording={"r1": _ibi_valid_frames()},
        min_ibis_per_cell=26,
    )
    pre_primary = cells.filter(
        (pl.col("epoch_role") == "chaser_pre")
        & (pl.col("censoring") == CENSORING_PRIMARY)
    ).row(0, named=True)
    pre_none = cells.filter(
        (pl.col("epoch_role") == "chaser_pre") & (pl.col("censoring") == CENSORING_NONE)
    ).row(0, named=True)
    # 25 usable IBIs < 26 under censoring: the cell is invalid with no stats;
    # the uncensored variant keeps all 26 and stays valid.
    assert pre_primary["cell_valid"] is False
    assert pre_primary["q50_s"] is None
    assert pre_none["cell_valid"] is True

    frames = _frames(
        {
            "rec_a": {"aggressive": 0.5, "inert": 0.1},
            "rec_b": {"aggressive": 0.3, "inert": 0.2},
        }
    )
    role_results = compute_role_contrast_rows(frames, parameters=PARAMS)
    out = tmp_path / "with-cells"
    write_role_contrasts(
        out,
        role_results,
        source_export_run_id="test-run",
        source_export_manifest_sha256="ab" * 32,
        source_export_root="/somewhere/publication",
        parameters=PARAMS,
        extra_tables={IBI_CELLS_PARQUET_NAME: (cells, IBI_SHAPE_SPEC_VERSION)},
    )
    manifest = read_role_contrasts_manifest(out)
    record = manifest["tables"][IBI_CELLS_PARQUET_NAME]
    assert record["spec_version"] == IBI_SHAPE_SPEC_VERSION
    assert record["row_count"] == cells.height
    # The cell table carries no multiplicity families.
    assert record["multiplicity_family_sizes"] == {}
    assert pl.read_parquet(out / IBI_CELLS_PARQUET_NAME).equals(cells)
