"""Aggressive-vs-inert chaser role contrasts over a validated-behavior export.

Motivation: the epoch axis (pre/training/post) is mechanically confounded for
near-field and quadrant metrics because both chasers are PARKED at corner
presets outside training (near-zone occupancy is ~0 for everyone) and move
during training.  The scientifically meaningful contrast is aggressive vs
inert WITHIN the chaser_training epoch: both chasers move during training and
only the aggressive one carries a pursuit contingency.

This module computes, per recording x provider_role, the paired
aggressive-minus-inert difference for a fixed metric roster, with
recording-equal-weight statistics (bootstrap CI + paired sign-flip test) and
Benjamini-Hochberg q-values within per-source-table multiplicity families.

The same contrast within chaser_pre and chaser_post is emitted as a clearly
labeled secondary tier carrying ``park_position_asymmetry = True``: the two
chasers park at different corners, so pre/post role contrasts confound role
with park position.

Everything here is exploratory.  Acquisition-batch identity is historically
missing for this cohort, so no batch/cluster adjustment is performed and every
row says so explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl

from fisheye.group_statistics.paired import (
    benjamini_hochberg,
    bootstrap_mean_ci,
    paired_sign_flip_p_value,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

SPEC_VERSION = "validated_behavior_role_contrasts_v1"
ANALYSIS_STATUS = "exploratory"
ACQUISITION_BATCH_ADJUSTMENT = "not_performed"
CONTRAST_NAME = "aggressive_minus_inert"

AGGRESSIVE_ROLE = "aggressive"
INERT_ROLE = "inert"
BEHAVIOR_ROLES = (AGGRESSIVE_ROLE, INERT_ROLE)

PRIMARY_EPOCH_ROLE = "chaser_training"
SECONDARY_EPOCH_ROLES = ("chaser_pre", "chaser_post")
EPOCH_ROLES = (PRIMARY_EPOCH_ROLE,) + SECONDARY_EPOCH_ROLES

PAIR_KEY = ("recording_id", "provider_role", "epoch_role", "behavior_role")

#: Metric roster per source table.  Every family below has size > 1 so no
#: size-1 FDR family can exist (a known historical failure mode in this repo).
SOURCE_TABLE_METRICS: Mapping[str, tuple[str, ...]] = {
    "radial_near_field_summary": (
        "near_zone_fraction_valid",
        "near_zone_enrichment_geometric",
        "near_zone_entry_rate_per_min_valid_time",
        "distance_p50_mm",
        "fish_wall_distance_p50_mm",
    ),
    "same_quadrant_occupancy": (
        "same_quadrant_fraction_valid",
        "same_quadrant_fraction_candidate",
    ),
}

PARQUET_NAME = "role_contrasts.parquet"
MANIFEST_NAME = "manifest.json"

# --- Distance-binned bout response role contrasts (GROUP A1) ---------------

DISTANCE_BIN_SPEC_VERSION = "validated_behavior_distance_bin_contrasts_v1"
DISTANCE_BIN_SOURCE_TABLE = "bout_response_distance_bins"
DISTANCE_BIN_PARQUET_NAME = "distance_bin_contrasts.parquet"
DISTANCE_BIN_METRICS = (
    "bout_rate_per_min",
    "median_duration_s",
    "median_path_length_mm",
    "median_net_displacement_mm",
    "median_peak_speed_mm_s",
)
DISTANCE_BIN_PAIR_KEY = (
    "recording_id",
    "semantic_role",
    "behavior_role",
    "distance_bin_index",
)
DEFAULT_MIN_PAIRED = 20

# --- Bout/chaser association role contrasts (GROUP A2) ---------------------

BOUT_ASSOCIATION_SPEC_VERSION = "validated_behavior_bout_association_contrasts_v1"
BOUT_ASSOCIATION_SOURCE_TABLE = "bout_chaser_associations"
BOUT_ASSOCIATION_PARQUET_NAME = "bout_association_contrasts.parquet"
NEAR_ONSET_MAX_MM = 15.0
ONSET_BIN_EDGES_MM: tuple[tuple[float, float], ...] = (
    (0.0, 10.0),
    (10.0, 20.0),
    (20.0, 30.0),
    (30.0, 40.0),
    (40.0, math.inf),
)
DEFAULT_MIN_ASSOCIATION_BOUTS = 5

# --- Pre-vs-post distribution-shape quantile contrasts (GROUP B) ------------

QUANTILE_SHAPE_SPEC_VERSION = "validated_behavior_quantile_shape_contrasts_v1"
SWIM_BOUT_SOURCE_TABLE = "canonical_swim_bouts"
SEMANTIC_EPOCHS_TABLE = "semantic_epochs"
QUANTILE_SHAPE_PARQUET_NAME = "quantile_shape_contrasts.parquet"
QUANTILE_METRICS = (
    "duration_s",
    "path_length_mm",
    "peak_speed_mm_s",
    "tortuosity",
)
SHAPE_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
EPOCH_PAIRS: tuple[tuple[str, str], ...] = (
    ("chaser_pre", "chaser_post"),
    ("chaser_pre", "chaser_training"),
    ("chaser_training", "chaser_post"),
)
DEFAULT_MIN_BOUTS_PER_CELL = 20

# --- Inter-bout-interval shape with dropout censoring (GROUP C) -------------

IBI_SHAPE_SPEC_VERSION = "validated_behavior_ibi_shape_contrasts_v1"
IBI_SHAPE_PARQUET_NAME = "ibi_shape_contrasts.parquet"
IBI_CELLS_PARQUET_NAME = "ibi_cell_statistics.parquet"
MOTION_SOURCE_TABLE = "provider_motion_samples"
MOTION_PROVIDER_ROLE = "keypoint"
IBI_FAMILY_PREFIX = "ibi_shape"
CENSORING_PRIMARY = "valid_span_required"
CENSORING_NONE = "none"
CENSORING_VARIANTS = (CENSORING_PRIMARY, CENSORING_NONE)
IBI_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
IBI_STAT_COLUMNS = (
    "q10_s",
    "q25_s",
    "q50_s",
    "q75_s",
    "q90_s",
    "mean_s",
    "cv",
    "frac_gt_2s",
    "frac_gt_5s",
)
DEFAULT_MIN_IBIS_PER_CELL = 20
EXPECTED_ACQUISITION_FPS = 100.0
FPS_TOLERANCE_FRACTION = 0.001

_EXPLORATORY_STAMP = {
    "analysis_status": ANALYSIS_STATUS,
    "acquisition_batch_adjustment": ACQUISITION_BATCH_ADJUSTMENT,
}


class RoleContrastInputError(ValueError):
    """The source frame does not match the expected role-contrast grain."""


@dataclass(frozen=True)
class RoleContrastParameters:
    bootstrap_iterations: int = 10_000
    permutation_iterations: int = 10_000
    confidence_level: float = 0.95
    seed: int = 20_260_901

    def __post_init__(self) -> None:
        if type(self.bootstrap_iterations) is not int or self.bootstrap_iterations <= 0:
            raise ValueError("bootstrap_iterations must be a positive integer")
        if (
            type(self.permutation_iterations) is not int
            or self.permutation_iterations <= 0
        ):
            raise ValueError("permutation_iterations must be a positive integer")
        if not (0.0 < float(self.confidence_level) < 1.0):
            raise ValueError("confidence_level must be in (0, 1)")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bootstrap_iterations": self.bootstrap_iterations,
            "permutation_iterations": self.permutation_iterations,
            "confidence_level": self.confidence_level,
            "seed": self.seed,
        }


def _contrast_rng(seed: int, key: Sequence[str]) -> np.random.Generator:
    """Return a deterministic per-contrast generator independent of row order."""

    digest = hashlib.sha256("\x1f".join(key).encode("utf-8")).digest()
    entropy = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(np.random.SeedSequence([int(seed), entropy]))


def validate_role_contrast_frame(source_table: str, frame: pl.DataFrame) -> None:
    """Fail closed when the frame cannot support one-row-per-role pairing."""

    metrics = SOURCE_TABLE_METRICS[source_table]
    required = set(PAIR_KEY) | set(metrics)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RoleContrastInputError(f"{source_table}: missing columns: {missing}")
    roles = set(frame["behavior_role"].unique().to_list())
    if not roles <= set(BEHAVIOR_ROLES):
        raise RoleContrastInputError(
            f"{source_table}: unexpected behavior_role values: "
            f"{sorted(roles - set(BEHAVIOR_ROLES))}"
        )
    epochs = set(frame["epoch_role"].unique().to_list())
    if not epochs <= set(EPOCH_ROLES):
        raise RoleContrastInputError(
            f"{source_table}: unexpected epoch_role values: "
            f"{sorted(epochs - set(EPOCH_ROLES))}"
        )
    duplicated = (
        frame.group_by(list(PAIR_KEY)).len().filter(pl.col("len") > 1)
    )
    if duplicated.height:
        raise RoleContrastInputError(
            f"{source_table}: {duplicated.height} duplicated "
            f"(recording, provider_role, epoch_role, behavior_role) cells; "
            "expected exactly one row per cell"
        )


def _paired_values(
    frame: pl.DataFrame,
    *,
    provider_role: str,
    epoch_role: str,
    metric: str,
) -> tuple[pl.DataFrame, int]:
    """Return one row per eligible recording with aggressive/inert values."""

    subset = frame.filter(
        (pl.col("provider_role") == provider_role)
        & (pl.col("epoch_role") == epoch_role)
    )
    aggressive = subset.filter(pl.col("behavior_role") == AGGRESSIVE_ROLE).select(
        "recording_id", pl.col(metric).alias("aggressive_value")
    )
    inert = subset.filter(pl.col("behavior_role") == INERT_ROLE).select(
        "recording_id", pl.col(metric).alias("inert_value")
    )
    paired = aggressive.join(inert, on="recording_id", how="full", coalesce=True)
    eligible = int(paired.height)
    return paired.sort("recording_id"), eligible


def compute_role_contrast_rows(
    frames: Mapping[str, pl.DataFrame],
    *,
    parameters: RoleContrastParameters | None = None,
) -> pl.DataFrame:
    """Compute every role contrast row, with BH q-values within families.

    ``frames`` maps source-table name -> long frame at the
    recording x provider_role x epoch_role x behavior_role grain.  Multiplicity
    families are one per source table x epoch_role, so the primary
    chaser_training family is never diluted by the caveated pre/post
    secondaries (and vice versa).  Family sizes are asserted > 1.
    """

    params = parameters or RoleContrastParameters()
    unknown = sorted(set(frames) - set(SOURCE_TABLE_METRICS))
    if unknown:
        raise RoleContrastInputError(f"Unknown source tables: {unknown}")

    rows: list[dict[str, Any]] = []
    for source_table in SOURCE_TABLE_METRICS:
        if source_table not in frames:
            raise RoleContrastInputError(f"Missing source table frame: {source_table}")
        frame = frames[source_table]
        validate_role_contrast_frame(source_table, frame)
        provider_roles = sorted(frame["provider_role"].unique().to_list())
        epoch_roles = [
            role
            for role in EPOCH_ROLES
            if role in set(frame["epoch_role"].unique().to_list())
        ]
        for epoch_role in epoch_roles:
            for metric in SOURCE_TABLE_METRICS[source_table]:
                for provider_role in provider_roles:
                    paired, eligible = _paired_values(
                        frame,
                        provider_role=provider_role,
                        epoch_role=epoch_role,
                        metric=metric,
                    )
                    aggressive = paired["aggressive_value"].to_numpy().astype(np.float64)
                    inert = paired["inert_value"].to_numpy().astype(np.float64)
                    finite = np.isfinite(aggressive) & np.isfinite(inert)
                    diffs = aggressive[finite] - inert[finite]
                    paired_n = int(diffs.size)
                    excluded = eligible - paired_n

                    rng = _contrast_rng(
                        params.seed,
                        (SPEC_VERSION, source_table, metric, provider_role, epoch_role),
                    )
                    if paired_n:
                        ci_low, ci_high = bootstrap_mean_ci(
                            diffs,
                            iterations=params.bootstrap_iterations,
                            confidence_level=params.confidence_level,
                            rng=rng,
                        )
                        p_value, test_method, effective_permutations = (
                            paired_sign_flip_p_value(
                                diffs,
                                iterations=params.permutation_iterations,
                                rng=rng,
                            )
                        )
                        mean_diff = float(np.mean(diffs))
                        median_diff = float(np.median(diffs))
                        mean_aggressive = float(np.mean(aggressive[finite]))
                        mean_inert = float(np.mean(inert[finite]))
                    else:
                        ci_low = ci_high = p_value = None
                        mean_diff = median_diff = None
                        mean_aggressive = mean_inert = None
                        test_method = "paired_sign_flip_unavailable"
                        effective_permutations = 0

                    rows.append(
                        {
                            "spec_version": SPEC_VERSION,
                            "analysis_status": ANALYSIS_STATUS,
                            "acquisition_batch_adjustment": ACQUISITION_BATCH_ADJUSTMENT,
                            "source_table": source_table,
                            "metric": metric,
                            "provider_role": provider_role,
                            "epoch_role": epoch_role,
                            "contrast": CONTRAST_NAME,
                            "contrast_tier": (
                                "primary"
                                if epoch_role == PRIMARY_EPOCH_ROLE
                                else "secondary"
                            ),
                            "park_position_asymmetry": epoch_role != PRIMARY_EPOCH_ROLE,
                            "eligible_recording_count": eligible,
                            "paired_unit_count": paired_n,
                            "excluded_nonfinite_count": excluded,
                            "mean_aggressive": mean_aggressive,
                            "mean_inert": mean_inert,
                            "mean_difference": mean_diff,
                            "median_difference": median_diff,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "confidence_level": params.confidence_level,
                            "bootstrap_iterations": params.bootstrap_iterations,
                            "p_value": p_value,
                            "test_method": test_method,
                            "permutation_iterations": int(effective_permutations),
                            "multiplicity_family": f"{source_table}:{epoch_role}",
                            "family_size": 0,
                            "q_value": None,
                            "seed": params.seed,
                        }
                    )

    families: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        families.setdefault(row["multiplicity_family"], []).append(index)
    for family, indexes in families.items():
        if len(indexes) <= 1:
            raise RoleContrastInputError(
                f"Multiplicity family {family!r} has size {len(indexes)}; "
                "size-1 FDR families are forbidden"
            )
        q_values = benjamini_hochberg([rows[i]["p_value"] for i in indexes])
        for i, q_value in zip(indexes, q_values):
            rows[i]["family_size"] = len(indexes)
            rows[i]["q_value"] = q_value

    schema = {
        "spec_version": pl.Utf8,
        "analysis_status": pl.Utf8,
        "acquisition_batch_adjustment": pl.Utf8,
        "source_table": pl.Utf8,
        "metric": pl.Utf8,
        "provider_role": pl.Utf8,
        "epoch_role": pl.Utf8,
        "contrast": pl.Utf8,
        "contrast_tier": pl.Utf8,
        "park_position_asymmetry": pl.Boolean,
        "eligible_recording_count": pl.Int64,
        "paired_unit_count": pl.Int64,
        "excluded_nonfinite_count": pl.Int64,
        "mean_aggressive": pl.Float64,
        "mean_inert": pl.Float64,
        "mean_difference": pl.Float64,
        "median_difference": pl.Float64,
        "ci_low": pl.Float64,
        "ci_high": pl.Float64,
        "confidence_level": pl.Float64,
        "bootstrap_iterations": pl.Int64,
        "p_value": pl.Float64,
        "test_method": pl.Utf8,
        "permutation_iterations": pl.Int64,
        "multiplicity_family": pl.Utf8,
        "family_size": pl.Int64,
        "q_value": pl.Float64,
        "seed": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema)


def load_role_contrast_frames(dataset: Any) -> dict[str, pl.DataFrame]:
    """Collect only the pairing-key and metric columns from the export."""

    frames: dict[str, pl.DataFrame] = {}
    for source_table, metrics in SOURCE_TABLE_METRICS.items():
        columns = list(PAIR_KEY) + list(metrics)
        frames[source_table] = (
            dataset.table(source_table).scan(columns=columns).collect()
        )
    return frames


def _paired_stats(
    minuend: np.ndarray,
    subtrahend: np.ndarray,
    *,
    params: RoleContrastParameters,
    rng: np.random.Generator,
    min_paired: int,
) -> dict[str, Any]:
    """Shared paired-difference statistics (minuend - subtrahend)."""

    minuend = np.asarray(minuend, dtype=np.float64).reshape(-1)
    subtrahend = np.asarray(subtrahend, dtype=np.float64).reshape(-1)
    finite = np.isfinite(minuend) & np.isfinite(subtrahend)
    diffs = minuend[finite] - subtrahend[finite]
    paired_n = int(diffs.size)
    base: dict[str, Any] = {
        "paired_unit_count": paired_n,
        "mean_minuend": float(np.mean(minuend[finite])) if paired_n else None,
        "mean_subtrahend": float(np.mean(subtrahend[finite])) if paired_n else None,
    }
    if paired_n < int(min_paired):
        return {
            **base,
            "mean_difference": None,
            "median_difference": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": None,
            "test_method": "skipped",
            "permutation_iterations": 0,
            "status": "skipped",
            "skip_reason": f"paired_unit_count<{int(min_paired)}",
        }
    ci_low, ci_high = bootstrap_mean_ci(
        diffs,
        iterations=params.bootstrap_iterations,
        confidence_level=params.confidence_level,
        rng=rng,
    )
    p_value, test_method, effective_permutations = paired_sign_flip_p_value(
        diffs, iterations=params.permutation_iterations, rng=rng
    )
    return {
        **base,
        "mean_difference": float(np.mean(diffs)),
        "median_difference": float(np.median(diffs)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "test_method": test_method,
        "permutation_iterations": int(effective_permutations),
        "status": "computed",
        "skip_reason": None,
    }


def _apply_bh_within_families(rows: list[dict[str, Any]]) -> None:
    """Assign q_value/family_size in place; family_size counts tested rows."""

    families: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        families.setdefault(row["multiplicity_family"], []).append(index)
    for family, indexes in families.items():
        tested = [i for i in indexes if rows[i]["p_value"] is not None]
        if len(tested) <= 1:
            raise RoleContrastInputError(
                f"Multiplicity family {family!r} has {len(tested)} tested rows; "
                "size-1 FDR families are forbidden"
            )
        q_values = benjamini_hochberg([rows[i]["p_value"] for i in tested])
        for i, q_value in zip(tested, q_values):
            rows[i]["q_value"] = q_value
        for i in indexes:
            rows[i]["family_size"] = len(tested)


_COMMON_STAT_SCHEMA = {
    "eligible_recording_count": pl.Int64,
    "paired_unit_count": pl.Int64,
    "excluded_nonfinite_count": pl.Int64,
    "mean_aggressive": pl.Float64,
    "mean_inert": pl.Float64,
    "mean_difference": pl.Float64,
    "median_difference": pl.Float64,
    "ci_low": pl.Float64,
    "ci_high": pl.Float64,
    "confidence_level": pl.Float64,
    "bootstrap_iterations": pl.Int64,
    "p_value": pl.Float64,
    "test_method": pl.Utf8,
    "permutation_iterations": pl.Int64,
    "status": pl.Utf8,
    "skip_reason": pl.Utf8,
    "multiplicity_family": pl.Utf8,
    "family_size": pl.Int64,
    "q_value": pl.Float64,
    "seed": pl.Int64,
}


def _role_stat_fields(
    stats: Mapping[str, Any], *, eligible: int, params: RoleContrastParameters
) -> dict[str, Any]:
    return {
        "eligible_recording_count": int(eligible),
        "paired_unit_count": stats["paired_unit_count"],
        "excluded_nonfinite_count": int(eligible) - stats["paired_unit_count"],
        "mean_aggressive": stats["mean_minuend"],
        "mean_inert": stats["mean_subtrahend"],
        "mean_difference": stats["mean_difference"],
        "median_difference": stats["median_difference"],
        "ci_low": stats["ci_low"],
        "ci_high": stats["ci_high"],
        "confidence_level": params.confidence_level,
        "bootstrap_iterations": params.bootstrap_iterations,
        "p_value": stats["p_value"],
        "test_method": stats["test_method"],
        "permutation_iterations": stats["permutation_iterations"],
        "status": stats["status"],
        "skip_reason": stats["skip_reason"],
        "family_size": 0,
        "q_value": None,
        "seed": params.seed,
    }


def _role_pair_arrays(
    subset: pl.DataFrame, value_column: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """Full-join aggressive/inert cells per recording; return arrays + eligible."""

    aggressive = subset.filter(pl.col("behavior_role") == AGGRESSIVE_ROLE).select(
        "recording_id", pl.col(value_column).alias("aggressive_value")
    )
    inert = subset.filter(pl.col("behavior_role") == INERT_ROLE).select(
        "recording_id", pl.col(value_column).alias("inert_value")
    )
    paired = aggressive.join(inert, on="recording_id", how="full", coalesce=True).sort(
        "recording_id"
    )
    return (
        paired["aggressive_value"].cast(pl.Float64).fill_null(float("nan")).to_numpy(),
        paired["inert_value"].cast(pl.Float64).fill_null(float("nan")).to_numpy(),
        int(paired.height),
    )


def compute_distance_bin_contrast_rows(
    frame: pl.DataFrame,
    *,
    parameters: RoleContrastParameters | None = None,
    min_paired: int = DEFAULT_MIN_PAIRED,
) -> pl.DataFrame:
    """Aggressive-minus-inert bout-response contrasts per distance bin.

    Grain in: recording x semantic_role x behavior_role x distance bin, with
    NaN metric values where a bin holds no bouts.  Inert near bins are thin
    (the inert dot does not pursue), so contrasts below ``min_paired`` pairs
    are emitted as status="skipped" rather than as fragile estimates.
    """

    params = parameters or RoleContrastParameters()
    required = set(DISTANCE_BIN_PAIR_KEY) | {
        "distance_bin_start_mm",
        "distance_bin_end_mm",
    } | set(DISTANCE_BIN_METRICS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RoleContrastInputError(
            f"{DISTANCE_BIN_SOURCE_TABLE}: missing columns: {missing}"
        )
    roles = set(frame["behavior_role"].unique().to_list())
    if not roles <= set(BEHAVIOR_ROLES):
        raise RoleContrastInputError(
            f"{DISTANCE_BIN_SOURCE_TABLE}: unexpected behavior_role values: "
            f"{sorted(roles - set(BEHAVIOR_ROLES))}"
        )
    epochs = set(frame["semantic_role"].unique().to_list())
    if not epochs <= set(EPOCH_ROLES):
        raise RoleContrastInputError(
            f"{DISTANCE_BIN_SOURCE_TABLE}: unexpected semantic_role values: "
            f"{sorted(epochs - set(EPOCH_ROLES))}"
        )
    duplicated = (
        frame.group_by(list(DISTANCE_BIN_PAIR_KEY)).len().filter(pl.col("len") > 1)
    )
    if duplicated.height:
        raise RoleContrastInputError(
            f"{DISTANCE_BIN_SOURCE_TABLE}: {duplicated.height} duplicated "
            "(recording, semantic_role, behavior_role, bin) cells"
        )

    bins = (
        frame.select(
            "distance_bin_index", "distance_bin_start_mm", "distance_bin_end_mm"
        )
        .unique()
        .sort("distance_bin_index")
    )
    rows: list[dict[str, Any]] = []
    for epoch_role in (role for role in EPOCH_ROLES if role in epochs):
        for bin_index, bin_start, bin_end in bins.iter_rows():
            subset = frame.filter(
                (pl.col("semantic_role") == epoch_role)
                & (pl.col("distance_bin_index") == bin_index)
            )
            for metric in DISTANCE_BIN_METRICS:
                aggressive, inert, eligible = _role_pair_arrays(subset, metric)
                rng = _contrast_rng(
                    params.seed,
                    (
                        DISTANCE_BIN_SPEC_VERSION,
                        DISTANCE_BIN_SOURCE_TABLE,
                        metric,
                        epoch_role,
                        f"bin_{bin_index}",
                    ),
                )
                stats = _paired_stats(
                    aggressive, inert, params=params, rng=rng, min_paired=min_paired
                )
                rows.append(
                    {
                        "spec_version": DISTANCE_BIN_SPEC_VERSION,
                        **_EXPLORATORY_STAMP,
                        "source_table": DISTANCE_BIN_SOURCE_TABLE,
                        "metric": metric,
                        "epoch_role": epoch_role,
                        "distance_bin_index": int(bin_index),
                        "distance_bin_start_mm": float(bin_start),
                        "distance_bin_end_mm": float(bin_end),
                        "contrast": CONTRAST_NAME,
                        "contrast_tier": (
                            "primary"
                            if epoch_role == PRIMARY_EPOCH_ROLE
                            else "secondary"
                        ),
                        "park_position_asymmetry": epoch_role != PRIMARY_EPOCH_ROLE,
                        "min_paired_threshold": int(min_paired),
                        "multiplicity_family": (
                            f"{DISTANCE_BIN_SOURCE_TABLE}:{epoch_role}"
                        ),
                        **_role_stat_fields(stats, eligible=eligible, params=params),
                    }
                )
    _apply_bh_within_families(rows)
    schema = {
        "spec_version": pl.Utf8,
        "analysis_status": pl.Utf8,
        "acquisition_batch_adjustment": pl.Utf8,
        "source_table": pl.Utf8,
        "metric": pl.Utf8,
        "epoch_role": pl.Utf8,
        "distance_bin_index": pl.Int64,
        "distance_bin_start_mm": pl.Float64,
        "distance_bin_end_mm": pl.Float64,
        "contrast": pl.Utf8,
        "contrast_tier": pl.Utf8,
        "park_position_asymmetry": pl.Boolean,
        "min_paired_threshold": pl.Int64,
        **_COMMON_STAT_SCHEMA,
    }
    return pl.DataFrame(rows, schema=schema)


def _association_metric_name(bin_start: float, bin_end: float) -> str:
    if math.isinf(bin_end):
        return f"median_delta_distance_mm_onset_ge_{int(bin_start)}mm"
    return (
        f"median_delta_distance_mm_onset_{int(bin_start):02d}_{int(bin_end):02d}mm"
    )


TOWARD_FRACTION_METRIC = (
    f"turn_toward_chaser_fraction_onset_lt_{int(NEAR_ONSET_MAX_MM)}mm"
)


def compute_bout_association_contrast_rows(
    frame: pl.DataFrame,
    *,
    parameters: RoleContrastParameters | None = None,
    min_bouts: int = DEFAULT_MIN_ASSOCIATION_BOUTS,
    min_paired: int = DEFAULT_MIN_PAIRED,
) -> pl.DataFrame:
    """Bout-level chaser-association role contrasts within chaser_training.

    Filters to base_valid & directed_valid bouts with a finite onset
    distance, then contrasts (aggressive - inert) per recording:
    the near-onset turn-toward fraction (onset < 15 mm) and the median
    delta_distance_mm in 10 mm onset bins.  A recording x role cell needs at
    least ``min_bouts`` qualifying bouts to contribute.
    """

    params = parameters or RoleContrastParameters()
    required = {
        "recording_id",
        "semantic_role",
        "behavior_role",
        "base_valid",
        "directed_valid",
        "distance_at_onset_mm",
        "delta_distance_mm",
        "turn_toward_chaser",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RoleContrastInputError(
            f"{BOUT_ASSOCIATION_SOURCE_TABLE}: missing columns: {missing}"
        )
    filtered = frame.filter(
        pl.col("base_valid")
        & pl.col("directed_valid")
        & pl.col("distance_at_onset_mm").is_finite()
        & (pl.col("semantic_role") == PRIMARY_EPOCH_ROLE)
        & pl.col("behavior_role").is_in(list(BEHAVIOR_ROLES))
    )

    metric_cells: list[tuple[str, float | None, float | None, pl.DataFrame]] = []
    near = filtered.filter(pl.col("distance_at_onset_mm") < NEAR_ONSET_MAX_MM)
    toward_cells = (
        near.group_by("recording_id", "behavior_role")
        .agg(
            pl.len().alias("cell_bout_count"),
            pl.col("turn_toward_chaser").cast(pl.Float64).mean().alias("value"),
        )
        .filter(pl.col("cell_bout_count") >= int(min_bouts))
    )
    metric_cells.append((TOWARD_FRACTION_METRIC, None, None, toward_cells))
    for bin_start, bin_end in ONSET_BIN_EDGES_MM:
        sub = filtered.filter(
            (pl.col("distance_at_onset_mm") >= bin_start)
            & (pl.col("distance_at_onset_mm") < bin_end)
            & pl.col("delta_distance_mm").is_finite()
        )
        cells = (
            sub.group_by("recording_id", "behavior_role")
            .agg(
                pl.len().alias("cell_bout_count"),
                pl.col("delta_distance_mm").median().alias("value"),
            )
            .filter(pl.col("cell_bout_count") >= int(min_bouts))
        )
        metric_cells.append(
            (_association_metric_name(bin_start, bin_end), bin_start, bin_end, cells)
        )

    rows: list[dict[str, Any]] = []
    for metric, bin_start, bin_end, cells in metric_cells:
        aggressive, inert, eligible = _role_pair_arrays(cells, "value")
        rng = _contrast_rng(
            params.seed,
            (
                BOUT_ASSOCIATION_SPEC_VERSION,
                BOUT_ASSOCIATION_SOURCE_TABLE,
                metric,
                PRIMARY_EPOCH_ROLE,
            ),
        )
        stats = _paired_stats(
            aggressive, inert, params=params, rng=rng, min_paired=min_paired
        )
        rows.append(
            {
                "spec_version": BOUT_ASSOCIATION_SPEC_VERSION,
                **_EXPLORATORY_STAMP,
                "source_table": BOUT_ASSOCIATION_SOURCE_TABLE,
                "metric": metric,
                "epoch_role": PRIMARY_EPOCH_ROLE,
                "onset_bin_start_mm": bin_start,
                "onset_bin_end_mm": bin_end,
                "contrast": CONTRAST_NAME,
                "contrast_tier": "primary",
                "park_position_asymmetry": False,
                "min_bouts_per_cell": int(min_bouts),
                "min_paired_threshold": int(min_paired),
                "multiplicity_family": (
                    f"{BOUT_ASSOCIATION_SOURCE_TABLE}:{PRIMARY_EPOCH_ROLE}"
                ),
                **_role_stat_fields(stats, eligible=eligible, params=params),
            }
        )
    _apply_bh_within_families(rows)
    schema = {
        "spec_version": pl.Utf8,
        "analysis_status": pl.Utf8,
        "acquisition_batch_adjustment": pl.Utf8,
        "source_table": pl.Utf8,
        "metric": pl.Utf8,
        "epoch_role": pl.Utf8,
        "onset_bin_start_mm": pl.Float64,
        "onset_bin_end_mm": pl.Float64,
        "contrast": pl.Utf8,
        "contrast_tier": pl.Utf8,
        "park_position_asymmetry": pl.Boolean,
        "min_bouts_per_cell": pl.Int64,
        "min_paired_threshold": pl.Int64,
        **_COMMON_STAT_SCHEMA,
    }
    return pl.DataFrame(rows, schema=schema)


def compute_quantile_shape_contrast_rows(
    bouts: pl.DataFrame,
    epochs: pl.DataFrame,
    *,
    parameters: RoleContrastParameters | None = None,
    min_bouts_per_cell: int = DEFAULT_MIN_BOUTS_PER_CELL,
    min_paired: int = DEFAULT_MIN_PAIRED,
) -> pl.DataFrame:
    """Distribution-SHAPE contrasts across epochs (no role axis).

    Assigns each canonical swim bout to a chaser epoch by
    ``start_acquisition_frame_id in [start_frame, end_frame_exclusive)``,
    computes per recording x epoch quantiles (q10..q90) of each bout metric
    over finite values only (with excluded-count accounting), then contrasts
    epoch pairs paired per recording on each quantile.  A recording x epoch
    cell needs at least ``min_bouts_per_cell`` finite bout values.
    """

    params = parameters or RoleContrastParameters()
    bout_required = {
        "recording_id",
        "bout_row_id",
        "start_acquisition_frame_id",
    } | set(QUANTILE_METRICS)
    missing = sorted(bout_required - set(bouts.columns))
    if missing:
        raise RoleContrastInputError(
            f"{SWIM_BOUT_SOURCE_TABLE}: missing columns: {missing}"
        )
    epoch_required = {
        "recording_id",
        "analysis_role",
        "start_frame",
        "end_frame_exclusive",
    }
    missing = sorted(epoch_required - set(epochs.columns))
    if missing:
        raise RoleContrastInputError(
            f"{SEMANTIC_EPOCHS_TABLE}: missing columns: {missing}"
        )

    windows = epochs.filter(pl.col("analysis_role").is_in(list(EPOCH_ROLES)))
    joined = (
        bouts.join(windows, on="recording_id", how="inner")
        .filter(
            (pl.col("start_acquisition_frame_id") >= pl.col("start_frame"))
            & (pl.col("start_acquisition_frame_id") < pl.col("end_frame_exclusive"))
        )
        .rename({"analysis_role": "epoch_role"})
    )
    ambiguous = (
        joined.group_by("recording_id", "bout_row_id").len().filter(pl.col("len") > 1)
    )
    if ambiguous.height:
        raise RoleContrastInputError(
            f"{ambiguous.height} bouts fall inside more than one chaser epoch "
            "window; overlapping windows are not supported"
        )

    rows: list[dict[str, Any]] = []
    quantile_columns = [f"q{int(round(q * 100))}" for q in SHAPE_QUANTILES]
    for metric in QUANTILE_METRICS:
        finite_expr = pl.col(metric).is_finite().fill_null(False)
        aggregations = [
            pl.len().alias("cell_bout_count"),
            finite_expr.sum().alias("finite_bout_count"),
        ] + [
            pl.col(metric)
            .filter(finite_expr)
            .quantile(q, interpolation="linear")
            .alias(column)
            for q, column in zip(SHAPE_QUANTILES, quantile_columns)
        ]
        cells = (
            joined.group_by("recording_id", "epoch_role")
            .agg(aggregations)
            .with_columns(
                (pl.col("cell_bout_count") - pl.col("finite_bout_count")).alias(
                    "nonfinite_bout_count"
                )
            )
            .filter(pl.col("finite_bout_count") >= int(min_bouts_per_cell))
        )
        for epoch_a, epoch_b in EPOCH_PAIRS:
            cells_a = cells.filter(pl.col("epoch_role") == epoch_a)
            cells_b = cells.filter(pl.col("epoch_role") == epoch_b)
            for q, column in zip(SHAPE_QUANTILES, quantile_columns):
                side_a = cells_a.select(
                    "recording_id",
                    pl.col(column).alias("value_a"),
                    pl.col("nonfinite_bout_count").alias("nonfinite_a"),
                )
                side_b = cells_b.select(
                    "recording_id",
                    pl.col(column).alias("value_b"),
                    pl.col("nonfinite_bout_count").alias("nonfinite_b"),
                )
                paired = side_a.join(
                    side_b, on="recording_id", how="full", coalesce=True
                ).sort("recording_id")
                eligible = int(paired.height)
                value_a = (
                    paired["value_a"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
                )
                value_b = (
                    paired["value_b"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
                )
                both = paired.drop_nulls(["value_a", "value_b"])
                nonfinite_excluded = int(
                    (both["nonfinite_a"] + both["nonfinite_b"]).sum() or 0
                )
                rng = _contrast_rng(
                    params.seed,
                    (
                        QUANTILE_SHAPE_SPEC_VERSION,
                        SWIM_BOUT_SOURCE_TABLE,
                        metric,
                        column,
                        epoch_a,
                        epoch_b,
                    ),
                )
                stats = _paired_stats(
                    value_b, value_a, params=params, rng=rng, min_paired=min_paired
                )
                rows.append(
                    {
                        "spec_version": QUANTILE_SHAPE_SPEC_VERSION,
                        **_EXPLORATORY_STAMP,
                        "source_table": SWIM_BOUT_SOURCE_TABLE,
                        "metric": metric,
                        "quantile": float(q),
                        "epoch_a": epoch_a,
                        "epoch_b": epoch_b,
                        "contrast": f"{epoch_b}_minus_{epoch_a}",
                        "min_bouts_per_cell": int(min_bouts_per_cell),
                        "min_paired_threshold": int(min_paired),
                        "eligible_recording_count": eligible,
                        "paired_unit_count": stats["paired_unit_count"],
                        "excluded_nonfinite_count": (
                            eligible - stats["paired_unit_count"]
                        ),
                        "nonfinite_bout_values_excluded": nonfinite_excluded,
                        "mean_epoch_a": stats["mean_subtrahend"],
                        "mean_epoch_b": stats["mean_minuend"],
                        "mean_difference": stats["mean_difference"],
                        "median_difference": stats["median_difference"],
                        "ci_low": stats["ci_low"],
                        "ci_high": stats["ci_high"],
                        "confidence_level": params.confidence_level,
                        "bootstrap_iterations": params.bootstrap_iterations,
                        "p_value": stats["p_value"],
                        "test_method": stats["test_method"],
                        "permutation_iterations": stats["permutation_iterations"],
                        "status": stats["status"],
                        "skip_reason": stats["skip_reason"],
                        "multiplicity_family": f"{SWIM_BOUT_SOURCE_TABLE}:{metric}",
                        "family_size": 0,
                        "q_value": None,
                        "seed": params.seed,
                    }
                )
    _apply_bh_within_families(rows)
    schema = {
        "spec_version": pl.Utf8,
        "analysis_status": pl.Utf8,
        "acquisition_batch_adjustment": pl.Utf8,
        "source_table": pl.Utf8,
        "metric": pl.Utf8,
        "quantile": pl.Float64,
        "epoch_a": pl.Utf8,
        "epoch_b": pl.Utf8,
        "contrast": pl.Utf8,
        "min_bouts_per_cell": pl.Int64,
        "min_paired_threshold": pl.Int64,
        "eligible_recording_count": pl.Int64,
        "paired_unit_count": pl.Int64,
        "excluded_nonfinite_count": pl.Int64,
        "nonfinite_bout_values_excluded": pl.Int64,
        "mean_epoch_a": pl.Float64,
        "mean_epoch_b": pl.Float64,
        "mean_difference": pl.Float64,
        "median_difference": pl.Float64,
        "ci_low": pl.Float64,
        "ci_high": pl.Float64,
        "confidence_level": pl.Float64,
        "bootstrap_iterations": pl.Int64,
        "p_value": pl.Float64,
        "test_method": pl.Utf8,
        "permutation_iterations": pl.Int64,
        "status": pl.Utf8,
        "skip_reason": pl.Utf8,
        "multiplicity_family": pl.Utf8,
        "family_size": pl.Int64,
        "q_value": pl.Float64,
        "seed": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema)


def load_distance_bin_frame(dataset: Any) -> pl.DataFrame:
    columns = list(DISTANCE_BIN_PAIR_KEY) + [
        "distance_bin_start_mm",
        "distance_bin_end_mm",
    ] + list(DISTANCE_BIN_METRICS)
    return dataset.table(DISTANCE_BIN_SOURCE_TABLE).scan(columns=columns).collect()


def load_bout_association_frame(dataset: Any) -> pl.DataFrame:
    columns = [
        "recording_id",
        "semantic_role",
        "behavior_role",
        "base_valid",
        "directed_valid",
        "distance_at_onset_mm",
        "delta_distance_mm",
        "turn_toward_chaser",
    ]
    return dataset.table(BOUT_ASSOCIATION_SOURCE_TABLE).scan(columns=columns).collect()


def load_quantile_shape_frames(dataset: Any) -> tuple[pl.DataFrame, pl.DataFrame]:
    bout_columns = [
        "recording_id",
        "bout_row_id",
        "start_acquisition_frame_id",
    ] + list(QUANTILE_METRICS)
    bouts = dataset.table(SWIM_BOUT_SOURCE_TABLE).scan(columns=bout_columns).collect()
    epochs = (
        dataset.table(SEMANTIC_EPOCHS_TABLE)
        .scan(
            columns=[
                "recording_id",
                "analysis_role",
                "start_frame",
                "end_frame_exclusive",
            ]
        )
        .collect()
    )
    return bouts, epochs


def validate_recording_fps(
    fps_by_recording: Mapping[str, float],
    *,
    expected_fps: float = EXPECTED_ACQUISITION_FPS,
    tolerance_fraction: float = FPS_TOLERANCE_FRACTION,
) -> None:
    """Fail closed when any recording's measured fps departs from expected."""

    tolerance = float(expected_fps) * float(tolerance_fraction)
    bad = {
        recording_id: float(fps)
        for recording_id, fps in fps_by_recording.items()
        if not math.isfinite(float(fps)) or abs(float(fps) - expected_fps) > tolerance
    }
    if bad:
        raise RoleContrastInputError(
            f"Measured acquisition fps departs from {expected_fps} by more than "
            f"{tolerance_fraction:.3%} for: {bad}"
        )


def _ibi_cell_stats(values: np.ndarray) -> dict[str, float | None]:
    quantiles = np.quantile(values, IBI_QUANTILES)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return {
        "q10_s": float(quantiles[0]),
        "q25_s": float(quantiles[1]),
        "q50_s": float(quantiles[2]),
        "q75_s": float(quantiles[3]),
        "q90_s": float(quantiles[4]),
        "mean_s": mean,
        "cv": std / mean if mean > 0 else None,
        "frac_gt_2s": float(np.mean(values > 2.0)),
        "frac_gt_5s": float(np.mean(values > 5.0)),
    }


def build_ibi_cells(
    bouts: pl.DataFrame,
    epochs: pl.DataFrame,
    *,
    fps_by_recording: Mapping[str, float],
    valid_frames_by_recording: Mapping[str, Any],
    min_ibis_per_cell: int = DEFAULT_MIN_IBIS_PER_CELL,
) -> pl.DataFrame:
    """Per recording x epoch x censoring-variant inter-bout-interval cells.

    IBI = (next bout start - current bout end) / fps between consecutive
    bouts of the same track.  An IBI belongs to an epoch only when BOTH its
    endpoints lie inside that epoch's window (pairs straddling an epoch
    boundary are dropped).  Under ``censoring="valid_span_required"`` an IBI
    is censored when any frame in its open interval (bout_end, next_start)
    is missing from or invalid in the tracked motion samples - the tracker
    preferentially loses still fish, so uncensored long-pause tails can be
    inflated by missed bouts.  ``censoring="none"`` keeps every IBI so the
    dropout sensitivity is itself visible.
    """

    required = {
        "recording_id",
        "track_id",
        "start_acquisition_frame_id",
        "end_acquisition_frame_id",
    }
    missing = sorted(required - set(bouts.columns))
    if missing:
        raise RoleContrastInputError(
            f"{SWIM_BOUT_SOURCE_TABLE}: missing columns: {missing}"
        )
    windows = epochs.filter(pl.col("analysis_role").is_in(list(EPOCH_ROLES)))
    duplicated = (
        windows.group_by("recording_id", "analysis_role").len().filter(pl.col("len") > 1)
    )
    if duplicated.height:
        raise RoleContrastInputError(
            f"{SEMANTIC_EPOCHS_TABLE}: {duplicated.height} duplicated "
            "(recording, analysis_role) windows; one window per epoch required"
        )

    validate_recording_fps(fps_by_recording)
    rows: list[dict[str, Any]] = []
    for recording_id in sorted(bouts["recording_id"].unique().to_list()):
        if recording_id not in fps_by_recording:
            raise RoleContrastInputError(f"Missing fps for recording {recording_id}")
        fps = float(fps_by_recording[recording_id])
        valid_frames = np.sort(
            np.asarray(
                valid_frames_by_recording.get(recording_id, ()), dtype=np.int64
            ).reshape(-1)
        )
        rec_bouts = bouts.filter(pl.col("recording_id") == recording_id).sort(
            "track_id", "start_acquisition_frame_id"
        )
        starts = rec_bouts["start_acquisition_frame_id"].to_numpy().astype(np.int64)
        ends = rec_bouts["end_acquisition_frame_id"].to_numpy().astype(np.int64)
        tracks = rec_bouts["track_id"].to_numpy()
        if starts.size >= 2:
            same_track = tracks[1:] == tracks[:-1]
            gap_start = ends[:-1][same_track]
            gap_end = starts[1:][same_track]
            ordered = gap_end >= gap_start
            gap_start = gap_start[ordered]
            gap_end = gap_end[ordered]
        else:
            gap_start = np.empty(0, dtype=np.int64)
            gap_end = np.empty(0, dtype=np.int64)
        ibi_seconds = (gap_end - gap_start).astype(np.float64) / fps
        open_length = np.maximum(gap_end - gap_start - 1, 0)
        valid_inside = np.searchsorted(
            valid_frames, gap_end, side="left"
        ) - np.searchsorted(valid_frames, gap_start, side="right")
        censored = valid_inside < open_length

        rec_windows = windows.filter(pl.col("recording_id") == recording_id)
        for epoch_role, window_start, window_end in rec_windows.select(
            "analysis_role", "start_frame", "end_frame_exclusive"
        ).iter_rows():
            in_window = (
                (gap_start >= window_start)
                & (gap_start < window_end)
                & (gap_end >= window_start)
                & (gap_end < window_end)
            )
            values = ibi_seconds[in_window]
            cell_censored = censored[in_window]
            for censoring in CENSORING_VARIANTS:
                if censoring == CENSORING_PRIMARY:
                    used = values[~cell_censored]
                    n_censored = int(cell_censored.sum())
                else:
                    used = values
                    n_censored = 0
                n_ibis = int(values.size)
                cell_valid = int(used.size) >= int(min_ibis_per_cell)
                stats: dict[str, float | None] = {
                    column: None for column in IBI_STAT_COLUMNS
                }
                if cell_valid:
                    stats = _ibi_cell_stats(used)
                rows.append(
                    {
                        "spec_version": IBI_SHAPE_SPEC_VERSION,
                        **_EXPLORATORY_STAMP,
                        "recording_id": recording_id,
                        "epoch_role": epoch_role,
                        "censoring": censoring,
                        "fps": fps,
                        "n_ibis": n_ibis,
                        "n_censored": n_censored,
                        "censored_fraction": (
                            n_censored / n_ibis if n_ibis else None
                        ),
                        "n_used": int(used.size),
                        "cell_valid": cell_valid,
                        "min_ibis_per_cell": int(min_ibis_per_cell),
                        **stats,
                    }
                )
    schema = {
        "spec_version": pl.Utf8,
        "analysis_status": pl.Utf8,
        "acquisition_batch_adjustment": pl.Utf8,
        "recording_id": pl.Utf8,
        "epoch_role": pl.Utf8,
        "censoring": pl.Utf8,
        "fps": pl.Float64,
        "n_ibis": pl.Int64,
        "n_censored": pl.Int64,
        "censored_fraction": pl.Float64,
        "n_used": pl.Int64,
        "cell_valid": pl.Boolean,
        "min_ibis_per_cell": pl.Int64,
        **{column: pl.Float64 for column in IBI_STAT_COLUMNS},
    }
    return pl.DataFrame(rows, schema=schema)


def compute_ibi_shape_contrast_rows(
    cells: pl.DataFrame,
    *,
    parameters: RoleContrastParameters | None = None,
    min_paired: int = DEFAULT_MIN_PAIRED,
) -> pl.DataFrame:
    """Paired epoch contrasts over the IBI cell statistics, per censoring."""

    params = parameters or RoleContrastParameters()
    rows: list[dict[str, Any]] = []
    for censoring in CENSORING_VARIANTS:
        valid_cells = cells.filter(
            (pl.col("censoring") == censoring) & pl.col("cell_valid")
        )
        for stat in IBI_STAT_COLUMNS:
            for epoch_a, epoch_b in EPOCH_PAIRS:
                side_a = valid_cells.filter(pl.col("epoch_role") == epoch_a).select(
                    "recording_id", pl.col(stat).alias("value_a")
                )
                side_b = valid_cells.filter(pl.col("epoch_role") == epoch_b).select(
                    "recording_id", pl.col(stat).alias("value_b")
                )
                paired = side_a.join(
                    side_b, on="recording_id", how="full", coalesce=True
                ).sort("recording_id")
                eligible = int(paired.height)
                value_a = (
                    paired["value_a"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
                )
                value_b = (
                    paired["value_b"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
                )
                rng = _contrast_rng(
                    params.seed,
                    (
                        IBI_SHAPE_SPEC_VERSION,
                        IBI_FAMILY_PREFIX,
                        censoring,
                        stat,
                        epoch_a,
                        epoch_b,
                    ),
                )
                stats = _paired_stats(
                    value_b, value_a, params=params, rng=rng, min_paired=min_paired
                )
                rows.append(
                    {
                        "spec_version": IBI_SHAPE_SPEC_VERSION,
                        **_EXPLORATORY_STAMP,
                        "source_table": SWIM_BOUT_SOURCE_TABLE,
                        "metric": stat,
                        "censoring": censoring,
                        "epoch_a": epoch_a,
                        "epoch_b": epoch_b,
                        "contrast": f"{epoch_b}_minus_{epoch_a}",
                        "min_paired_threshold": int(min_paired),
                        "eligible_recording_count": eligible,
                        "paired_unit_count": stats["paired_unit_count"],
                        "excluded_nonfinite_count": (
                            eligible - stats["paired_unit_count"]
                        ),
                        "mean_epoch_a": stats["mean_subtrahend"],
                        "mean_epoch_b": stats["mean_minuend"],
                        "mean_difference": stats["mean_difference"],
                        "median_difference": stats["median_difference"],
                        "ci_low": stats["ci_low"],
                        "ci_high": stats["ci_high"],
                        "confidence_level": params.confidence_level,
                        "bootstrap_iterations": params.bootstrap_iterations,
                        "p_value": stats["p_value"],
                        "test_method": stats["test_method"],
                        "permutation_iterations": stats["permutation_iterations"],
                        "status": stats["status"],
                        "skip_reason": stats["skip_reason"],
                        "multiplicity_family": f"{IBI_FAMILY_PREFIX}:{censoring}",
                        "family_size": 0,
                        "q_value": None,
                        "seed": params.seed,
                    }
                )
    _apply_bh_within_families(rows)
    schema = {
        "spec_version": pl.Utf8,
        "analysis_status": pl.Utf8,
        "acquisition_batch_adjustment": pl.Utf8,
        "source_table": pl.Utf8,
        "metric": pl.Utf8,
        "censoring": pl.Utf8,
        "epoch_a": pl.Utf8,
        "epoch_b": pl.Utf8,
        "contrast": pl.Utf8,
        "min_paired_threshold": pl.Int64,
        "eligible_recording_count": pl.Int64,
        "paired_unit_count": pl.Int64,
        "excluded_nonfinite_count": pl.Int64,
        "mean_epoch_a": pl.Float64,
        "mean_epoch_b": pl.Float64,
        "mean_difference": pl.Float64,
        "median_difference": pl.Float64,
        "ci_low": pl.Float64,
        "ci_high": pl.Float64,
        "confidence_level": pl.Float64,
        "bootstrap_iterations": pl.Int64,
        "p_value": pl.Float64,
        "test_method": pl.Utf8,
        "permutation_iterations": pl.Int64,
        "status": pl.Utf8,
        "skip_reason": pl.Utf8,
        "multiplicity_family": pl.Utf8,
        "family_size": pl.Int64,
        "q_value": pl.Float64,
        "seed": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema)


def load_ibi_inputs(
    dataset: Any,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, float], dict[str, np.ndarray]]:
    """Stream per-recording motion validity without materializing the table.

    Returns (bouts, epochs, fps_by_recording, valid_frames_by_recording).
    The motion table is scanned one recording at a time with column pruning
    (recording_id, acquisition_frame_id, time_s, linear_sample_valid); only
    the sorted VALID acquisition frame ids are retained per recording, so a
    frame absent from the motion table counts as invalid for censoring.
    """

    bouts = (
        dataset.table(SWIM_BOUT_SOURCE_TABLE)
        .scan(
            columns=[
                "recording_id",
                "track_id",
                "start_acquisition_frame_id",
                "end_acquisition_frame_id",
            ]
        )
        .collect()
    )
    epochs = (
        dataset.table(SEMANTIC_EPOCHS_TABLE)
        .scan(
            columns=[
                "recording_id",
                "analysis_role",
                "start_frame",
                "end_frame_exclusive",
            ]
        )
        .collect()
    )
    motion = dataset.table(MOTION_SOURCE_TABLE)
    fps_by_recording: dict[str, float] = {}
    valid_frames_by_recording: dict[str, np.ndarray] = {}
    for recording_id in sorted(bouts["recording_id"].unique().to_list()):
        samples = (
            motion.scan(
                columns=[
                    "recording_id",
                    "provider_role",
                    "acquisition_frame_id",
                    "time_s",
                    "linear_sample_valid",
                ],
                predicate=(
                    (pl.col("recording_id") == recording_id)
                    & (pl.col("provider_role") == MOTION_PROVIDER_ROLE)
                ),
            )
            .select("acquisition_frame_id", "time_s", "linear_sample_valid")
            .collect()
        )
        if samples.height < 2:
            raise RoleContrastInputError(
                f"{MOTION_SOURCE_TABLE}: no {MOTION_PROVIDER_ROLE} samples for "
                f"recording {recording_id}"
            )
        frame_span = float(
            samples["acquisition_frame_id"].max() - samples["acquisition_frame_id"].min()
        )
        time_span = float(samples["time_s"].max() - samples["time_s"].min())
        if time_span <= 0:
            raise RoleContrastInputError(
                f"{MOTION_SOURCE_TABLE}: degenerate time axis for {recording_id}"
            )
        fps_by_recording[recording_id] = frame_span / time_span
        valid_frames_by_recording[recording_id] = np.sort(
            samples.filter(pl.col("linear_sample_valid"))["acquisition_frame_id"]
            .to_numpy()
            .astype(np.int64)
        )
    validate_recording_fps(fps_by_recording)
    return bouts, epochs, fps_by_recording, valid_frames_by_recording


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _family_sizes(results: pl.DataFrame) -> dict[str, int]:
    if "multiplicity_family" not in results.columns:
        return {}
    return {
        family: int(size)
        for family, size in results.group_by("multiplicity_family")
        .len()
        .sort("multiplicity_family")
        .iter_rows()
    }


def write_role_contrasts(
    output_dir: str | Path,
    results: pl.DataFrame,
    *,
    source_export_run_id: str,
    source_export_manifest_sha256: str,
    source_export_root: str,
    parameters: RoleContrastParameters,
    overwrite: bool = False,
    extra_tables: Mapping[str, tuple[pl.DataFrame, str]] | None = None,
    thresholds: Mapping[str, int] | None = None,
) -> Path:
    """Write the contrast parquet files + one ``manifest.json`` directory.

    ``extra_tables`` maps additional parquet file names to
    ``(frame, spec_version)`` pairs, written next to ``role_contrasts.parquet``
    and recorded in the same manifest.
    """

    out = Path(output_dir).expanduser()
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory {out} is not empty; pass overwrite to replace it"
        )
    out.mkdir(parents=True, exist_ok=True)

    tables: dict[str, tuple[pl.DataFrame, str]] = {
        PARQUET_NAME: (results, SPEC_VERSION)
    }
    for name, (frame, spec_version) in (extra_tables or {}).items():
        if name in tables:
            raise ValueError(f"Duplicate contrast table file name: {name}")
        tables[name] = (frame, spec_version)

    files: dict[str, str] = {}
    table_records: dict[str, Any] = {}
    total_rows = 0
    for name, (frame, spec_version) in tables.items():
        path = out / name
        frame.write_parquet(path)
        files[name] = _sha256_file(path)
        total_rows += int(frame.height)
        table_records[name] = {
            "spec_version": spec_version,
            "row_count": int(frame.height),
            "multiplicity_family_sizes": _family_sizes(frame),
        }

    manifest: dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "analysis_status": ANALYSIS_STATUS,
        "acquisition_batch_adjustment": ACQUISITION_BATCH_ADJUSTMENT,
        "source_export": {
            "export_run_id": source_export_run_id,
            "export_manifest_record_sha256": source_export_manifest_sha256,
            "export_root": str(source_export_root),
        },
        "parameters": parameters.to_dict(),
        "source_table_metrics": {
            table: list(metrics) for table, metrics in SOURCE_TABLE_METRICS.items()
        },
        "row_count": total_rows,
        "multiplicity_family_sizes": _family_sizes(results),
        "tables": table_records,
        "files": files,
    }
    if thresholds is not None:
        manifest["thresholds"] = {
            key: int(value) for key, value in sorted(thresholds.items())
        }
    manifest["record_sha256"] = canonical_json_sha256(manifest)
    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def read_role_contrasts_manifest(output_dir: str | Path) -> dict[str, Any]:
    """Read and integrity-check a role-contrasts manifest and its parquet."""

    out = Path(output_dir).expanduser()
    manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected a JSON object in {out / MANIFEST_NAME}")
    persisted = manifest.get("record_sha256")
    body = {key: value for key, value in manifest.items() if key != "record_sha256"}
    if canonical_json_sha256(body) != persisted:
        raise ValueError(f"Role-contrasts manifest self-digest mismatch in {out}")
    for name, expected in manifest["files"].items():
        actual = _sha256_file(out / name)
        if actual != expected:
            raise ValueError(f"Role-contrasts file digest mismatch for {name} in {out}")
    return manifest


def format_contrast_table(results: pl.DataFrame, *, epoch_role: str) -> str:
    """Render a compact fixed-width report table for one epoch's contrasts."""

    subset = results.filter(pl.col("epoch_role") == epoch_role).sort(
        "source_table", "metric", "provider_role"
    )
    lines = [
        f"{'metric':<42}{'provider':<11}{'mean_diff':>12}"
        f"{'ci_low':>12}{'ci_high':>12}{'p':>10}{'q':>10}{'n':>5}"
    ]
    for row in subset.iter_rows(named=True):
        def _fmt(value: float | None, width: int = 12) -> str:
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                return f"{'NA':>{width}}"
            return f"{value:>{width}.5f}"

        lines.append(
            f"{row['metric']:<42}{row['provider_role']:<11}"
            f"{_fmt(row['mean_difference'])}{_fmt(row['ci_low'])}"
            f"{_fmt(row['ci_high'])}{_fmt(row['p_value'], 10)}"
            f"{_fmt(row['q_value'], 10)}{row['paired_unit_count']:>5d}"
        )
    return "\n".join(lines)


def format_stats_table(
    results: pl.DataFrame, label_columns: Sequence[str]
) -> str:
    """Render a compact fixed-width table with the shared statistics columns."""

    def _fmt(value: Any, width: int) -> str:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return f"{'NA':>{width}}"
        return f"{value:>{width}.5f}"

    label_rows = [
        ["" if row[column] is None else str(row[column]) for column in label_columns]
        for row in results.iter_rows(named=True)
    ]
    widths = [
        max(len(column), *(len(row[i]) for row in label_rows)) + 2
        for i, column in enumerate(label_columns)
    ]
    header = "".join(
        f"{column:<{width}}" for column, width in zip(label_columns, widths)
    )
    lines = [
        header
        + f"{'mean_diff':>12}{'ci_low':>12}{'ci_high':>12}"
        + f"{'p':>10}{'q':>10}{'n':>5}  status"
    ]
    for labels, row in zip(label_rows, results.iter_rows(named=True)):
        lines.append(
            "".join(f"{label:<{width}}" for label, width in zip(labels, widths))
            + f"{_fmt(row['mean_difference'], 12)}{_fmt(row['ci_low'], 12)}"
            + f"{_fmt(row['ci_high'], 12)}{_fmt(row['p_value'], 10)}"
            + f"{_fmt(row['q_value'], 10)}{row['paired_unit_count']:>5d}"
            + f"  {row['status']}"
        )
    return "\n".join(lines)


__all__ = [
    "ACQUISITION_BATCH_ADJUSTMENT",
    "ANALYSIS_STATUS",
    "BOUT_ASSOCIATION_PARQUET_NAME",
    "BOUT_ASSOCIATION_SPEC_VERSION",
    "CENSORING_NONE",
    "CENSORING_PRIMARY",
    "CENSORING_VARIANTS",
    "CONTRAST_NAME",
    "DEFAULT_MIN_ASSOCIATION_BOUTS",
    "DEFAULT_MIN_BOUTS_PER_CELL",
    "DEFAULT_MIN_IBIS_PER_CELL",
    "DEFAULT_MIN_PAIRED",
    "DISTANCE_BIN_METRICS",
    "DISTANCE_BIN_PARQUET_NAME",
    "DISTANCE_BIN_SPEC_VERSION",
    "EPOCH_PAIRS",
    "EPOCH_ROLES",
    "EXPECTED_ACQUISITION_FPS",
    "IBI_CELLS_PARQUET_NAME",
    "IBI_SHAPE_PARQUET_NAME",
    "IBI_SHAPE_SPEC_VERSION",
    "IBI_STAT_COLUMNS",
    "MANIFEST_NAME",
    "MOTION_PROVIDER_ROLE",
    "MOTION_SOURCE_TABLE",
    "NEAR_ONSET_MAX_MM",
    "ONSET_BIN_EDGES_MM",
    "PARQUET_NAME",
    "PRIMARY_EPOCH_ROLE",
    "QUANTILE_METRICS",
    "QUANTILE_SHAPE_PARQUET_NAME",
    "QUANTILE_SHAPE_SPEC_VERSION",
    "RoleContrastInputError",
    "RoleContrastParameters",
    "SECONDARY_EPOCH_ROLES",
    "SHAPE_QUANTILES",
    "SOURCE_TABLE_METRICS",
    "SPEC_VERSION",
    "TOWARD_FRACTION_METRIC",
    "build_ibi_cells",
    "compute_bout_association_contrast_rows",
    "compute_distance_bin_contrast_rows",
    "compute_ibi_shape_contrast_rows",
    "compute_quantile_shape_contrast_rows",
    "compute_role_contrast_rows",
    "format_contrast_table",
    "format_stats_table",
    "load_bout_association_frame",
    "load_distance_bin_frame",
    "load_ibi_inputs",
    "load_quantile_shape_frames",
    "load_role_contrast_frames",
    "read_role_contrasts_manifest",
    "validate_recording_fps",
    "validate_role_contrast_frame",
    "write_role_contrasts",
]
