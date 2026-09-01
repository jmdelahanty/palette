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
    EPOCH_ROLES,
    PRIMARY_EPOCH_ROLE,
    RoleContrastInputError,
    RoleContrastParameters,
    SECONDARY_EPOCH_ROLES,
    SOURCE_TABLE_METRICS,
    SPEC_VERSION,
    compute_role_contrast_rows,
    read_role_contrasts_manifest,
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
