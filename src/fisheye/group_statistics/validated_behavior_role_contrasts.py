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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_role_contrasts(
    output_dir: str | Path,
    results: pl.DataFrame,
    *,
    source_export_run_id: str,
    source_export_manifest_sha256: str,
    source_export_root: str,
    parameters: RoleContrastParameters,
    overwrite: bool = False,
) -> Path:
    """Write ``role_contrasts.parquet`` + ``manifest.json`` into a directory."""

    out = Path(output_dir).expanduser()
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory {out} is not empty; pass overwrite to replace it"
        )
    out.mkdir(parents=True, exist_ok=True)

    parquet_path = out / PARQUET_NAME
    results.write_parquet(parquet_path)

    family_sizes = {
        family: int(size)
        for family, size in results.group_by("multiplicity_family")
        .len()
        .sort("multiplicity_family")
        .iter_rows()
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
        "row_count": int(results.height),
        "multiplicity_family_sizes": family_sizes,
        "files": {PARQUET_NAME: _sha256_file(parquet_path)},
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


__all__ = [
    "ACQUISITION_BATCH_ADJUSTMENT",
    "ANALYSIS_STATUS",
    "CONTRAST_NAME",
    "EPOCH_ROLES",
    "MANIFEST_NAME",
    "PARQUET_NAME",
    "PRIMARY_EPOCH_ROLE",
    "RoleContrastInputError",
    "RoleContrastParameters",
    "SECONDARY_EPOCH_ROLES",
    "SOURCE_TABLE_METRICS",
    "SPEC_VERSION",
    "compute_role_contrast_rows",
    "format_contrast_table",
    "load_role_contrast_frames",
    "read_role_contrasts_manifest",
    "validate_role_contrast_frame",
    "write_role_contrasts",
]
