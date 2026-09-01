"""Strategy-state analysis over one validated-behavior cohort export.

Four deterministic stages over the phase-B validated-behavior export:

1. A corner-fair feature matrix per recording x epoch (``chaser_pre`` /
   ``chaser_post``) built from export summary tables plus two cross-branch
   derived products consumed strictly as validated input Parquet files
   (twin-excess corner nulls, IBI cell statistics).
2. Gaussian-mixture strategy clustering in PCA space with BIC model
   selection, pre-to-post transition structure, arena-stratified
   permutation tests, responder tiers, and a bootstrap ARI stability check.
3. A leave-recording-out stimulus-blind window decoder (pluggable window
   definitions) over canonical swim bouts.
4. A direction decomposition of each fish's pre-to-post displacement onto
   the punctuated cluster axis, plus disposition checks.

Everything is exploratory-grade: every persisted row is stamped
``analysis_status="exploratory"`` and
``acquisition_batch_adjustment="not_performed"``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

SPEC_VERSION = "validated_behavior_strategy_states_v1"
ANALYSIS_STATUS = "exploratory"
ACQUISITION_BATCH_ADJUSTMENT = "not_performed"

EPOCH_ROLES = ("chaser_pre", "chaser_post")

FEATURE_COLUMNS = (
    "nz_excess",
    "dist_excess",
    "fish_wall_distance_p50_mm",
    "occupancy_entropy",
    "bout_rate_per_min",
    "mean_abs_bout_net_heading_change_deg",
    "ibi_gt2s",
)

#: Features that must be present for every recording x epoch (fail closed).
REQUIRED_FEATURE_SOURCES = ("twin", "wall", "entropy", "bout")
#: The only imputable feature (median impute + flag column).
IMPUTABLE_FEATURE = "ibi_gt2s"

REQUIRED_TWIN_COLUMNS = (
    "recording_id",
    "provider_role",
    "epoch_role",
    "behavior_role",
    "near_zone_fraction_valid_excess",
    "distance_p50_mm_excess",
)
REQUIRED_IBI_COLUMNS = (
    "recording_id",
    "epoch_role",
    "censoring",
    "cell_valid",
    "frac_gt_2s",
)

DECODER_FEATURE_COLUMNS = (
    "n_bouts",
    "bout_duration_s_median",
    "bout_duration_s_q90",
    "bout_path_length_mm_median",
    "bout_path_length_mm_q10",
    "bout_peak_speed_mm_s_median",
    "bout_peak_speed_mm_s_q10",
    "bout_net_displacement_mm_median",
    "bout_tortuosity_finite_median",
    "ibi_s_median",
    "ibi_s_q90",
    "ibi_frac_gt_2s",
)

_ARENA_PATTERN = re.compile(r"_arena_(\d+)")


@dataclass(frozen=True)
class StrategyStatesConfig:
    """All seeds and thresholds for the four strategy-state stages."""

    random_seed: int = 20260901
    k_min: int = 1
    k_max: int = 6
    pca_components: int = 3
    gmm_n_init: int = 20
    gmm_covariance_type: str = "full"
    permutation_iterations: int = 10_000
    bootstrap_ari_refits: int = 100
    escape_dominant_threshold: float = 0.75
    freeze_leaning_threshold: float = 0.25
    window_duration_s: float = 60.0
    expected_fps: float = 100.0
    fps_relative_tolerance: float = 0.001
    ibi_long_threshold_s: float = 2.0
    logistic_c: float = 0.5
    logistic_max_iter: int = 5000

    def to_dict(self) -> dict[str, Any]:
        return {
            "random_seed": self.random_seed,
            "k_min": self.k_min,
            "k_max": self.k_max,
            "pca_components": self.pca_components,
            "gmm_n_init": self.gmm_n_init,
            "gmm_covariance_type": self.gmm_covariance_type,
            "permutation_iterations": self.permutation_iterations,
            "bootstrap_ari_refits": self.bootstrap_ari_refits,
            "escape_dominant_threshold": self.escape_dominant_threshold,
            "freeze_leaning_threshold": self.freeze_leaning_threshold,
            "window_duration_s": self.window_duration_s,
            "expected_fps": self.expected_fps,
            "fps_relative_tolerance": self.fps_relative_tolerance,
            "ibi_long_threshold_s": self.ibi_long_threshold_s,
            "logistic_c": self.logistic_c,
            "logistic_max_iter": self.logistic_max_iter,
        }


class StrategyStatesInputError(ValueError):
    """Raised when a required input is missing or malformed (fail closed)."""


# ---------------------------------------------------------------------------
# Shared small helpers
# ---------------------------------------------------------------------------


def parse_arena(recording_id: str) -> str:
    match = _ARENA_PATTERN.search(recording_id)
    if match is None:
        raise StrategyStatesInputError(
            f"recording_id {recording_id!r} carries no '_arena_<n>' token; "
            "arena-stratified permutation tests cannot proceed"
        )
    return match.group(1)


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def occupancy_entropy(fractions: Sequence[float]) -> float:
    """Shannon entropy (nats) over strictly positive occupancy fractions."""

    values = np.asarray(fractions, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        raise StrategyStatesInputError(
            "occupancy entropy requires at least one positive occupancy fraction"
        )
    total = float(values.sum())
    p = values / total
    return float(-np.sum(p * np.log(p)))


def _stamp(frame: Any) -> Any:
    import polars as pl

    return frame.with_columns(
        pl.lit(SPEC_VERSION).alias("spec_version"),
        pl.lit(ANALYSIS_STATUS).alias("analysis_status"),
        pl.lit(ACQUISITION_BATCH_ADJUSTMENT).alias("acquisition_batch_adjustment"),
    ).select(
        ["spec_version", "analysis_status", "acquisition_batch_adjustment"]
        + [
            name
            for name in frame.columns
            if name
            not in ("spec_version", "analysis_status", "acquisition_batch_adjustment")
        ]
    )


# ---------------------------------------------------------------------------
# Stage 1 — cross-branch inputs and the feature matrix
# ---------------------------------------------------------------------------


def load_twin_excess_features(path: Path) -> Any:
    """Load the twin-excess parquet; keypoint/aggressive rows, pre/post only."""

    import polars as pl

    frame = pl.read_parquet(path)
    missing = sorted(set(REQUIRED_TWIN_COLUMNS) - set(frame.columns))
    if missing:
        raise StrategyStatesInputError(
            f"twin-excess parquet {path} is missing required columns {missing}; "
            f"required: {list(REQUIRED_TWIN_COLUMNS)}"
        )
    selected = frame.filter(
        (pl.col("provider_role") == "keypoint")
        & (pl.col("behavior_role") == "aggressive")
        & pl.col("epoch_role").is_in(list(EPOCH_ROLES))
    ).select(
        "recording_id",
        "epoch_role",
        pl.col("near_zone_fraction_valid_excess").alias("nz_excess"),
        pl.col("distance_p50_mm_excess").alias("dist_excess"),
    )
    if selected.is_empty():
        raise StrategyStatesInputError(
            f"twin-excess parquet {path} has no keypoint/aggressive pre/post rows"
        )
    duplicated = selected.group_by(["recording_id", "epoch_role"]).len().filter(
        pl.col("len") > 1
    )
    if not duplicated.is_empty():
        raise StrategyStatesInputError(
            "twin-excess parquet is not unique per recording x epoch after the "
            f"keypoint/aggressive filter: {duplicated.head(5).to_dicts()}"
        )
    return selected


def load_ibi_cell_features(path: Path) -> Any:
    """Load the IBI-cells parquet; valid_span_required and cell_valid only."""

    import polars as pl

    frame = pl.read_parquet(path)
    missing = sorted(set(REQUIRED_IBI_COLUMNS) - set(frame.columns))
    if missing:
        raise StrategyStatesInputError(
            f"IBI-cells parquet {path} is missing required columns {missing}; "
            f"required: {list(REQUIRED_IBI_COLUMNS)}"
        )
    selected = frame.filter(
        (pl.col("censoring") == "valid_span_required")
        & pl.col("cell_valid")
        & pl.col("epoch_role").is_in(list(EPOCH_ROLES))
    ).select(
        "recording_id",
        "epoch_role",
        pl.col("frac_gt_2s").alias("ibi_gt2s"),
    )
    duplicated = selected.group_by(["recording_id", "epoch_role"]).len().filter(
        pl.col("len") > 1
    )
    if not duplicated.is_empty():
        raise StrategyStatesInputError(
            "IBI-cells parquet is not unique per recording x epoch after the "
            f"valid_span_required/cell_valid filter: {duplicated.head(5).to_dicts()}"
        )
    return selected


def assemble_feature_matrix(
    *,
    twin: Any,
    wall: Any,
    entropy: Any,
    bout: Any,
    ibi: Any,
) -> Any:
    """Join the five per-(recording, epoch) feature sources.

    ``twin``/``wall``/``entropy``/``bout`` are REQUIRED for every recording x
    epoch cell: a missing row fails closed. ``ibi`` alone may be missing and is
    median-imputed (pooled median over observed cells) with an
    ``ibi_gt2s_imputed`` flag column.

    Expected input columns (each unique per recording_id x epoch_role):
      twin:    recording_id, epoch_role, nz_excess, dist_excess
      wall:    recording_id, epoch_role, fish_wall_distance_p50_mm
      entropy: recording_id, epoch_role, occupancy_entropy
      bout:    recording_id, epoch_role, bout_rate_per_min,
               mean_abs_bout_net_heading_change_deg
      ibi:     recording_id, epoch_role, ibi_gt2s
    """

    import polars as pl

    sources = {"twin": twin, "wall": wall, "entropy": entropy, "bout": bout}
    recordings: set[str] = set()
    for frame in sources.values():
        recordings.update(frame["recording_id"].to_list())
    recordings_sorted = sorted(recordings)
    grid = pl.DataFrame(
        {
            "recording_id": [r for r in recordings_sorted for _ in EPOCH_ROLES],
            "epoch_role": [e for _ in recordings_sorted for e in EPOCH_ROLES],
        }
    )

    joined = grid
    for name, frame in sources.items():
        joined = joined.join(frame, on=["recording_id", "epoch_role"], how="left")

    required_by_source = {
        "twin": ("nz_excess", "dist_excess"),
        "wall": ("fish_wall_distance_p50_mm",),
        "entropy": ("occupancy_entropy",),
        "bout": ("bout_rate_per_min", "mean_abs_bout_net_heading_change_deg"),
    }
    for source_name, columns in required_by_source.items():
        for column in columns:
            bad = joined.filter(
                pl.col(column).is_null() | ~pl.col(column).is_finite()
            )
            if not bad.is_empty():
                cells = bad.select("recording_id", "epoch_role").head(10).to_dicts()
                raise StrategyStatesInputError(
                    f"required feature {column!r} (source {source_name!r}) is "
                    f"missing or non-finite for {bad.height} recording x epoch "
                    f"cells; first cells: {cells}. Required sources "
                    f"{list(REQUIRED_FEATURE_SOURCES)} fail closed — only "
                    f"{IMPUTABLE_FEATURE!r} may be imputed."
                )

    joined = joined.join(ibi, on=["recording_id", "epoch_role"], how="left")
    observed = joined.filter(
        pl.col("ibi_gt2s").is_not_null() & pl.col("ibi_gt2s").is_finite()
    )["ibi_gt2s"]
    if observed.len() == 0:
        raise StrategyStatesInputError(
            "no observed IBI cells at all; cannot median-impute ibi_gt2s"
        )
    ibi_median = float(observed.median())
    joined = joined.with_columns(
        (pl.col("ibi_gt2s").is_null() | ~pl.col("ibi_gt2s").is_finite())
        .alias("ibi_gt2s_imputed"),
        pl.when(pl.col("ibi_gt2s").is_null() | ~pl.col("ibi_gt2s").is_finite())
        .then(pl.lit(ibi_median))
        .otherwise(pl.col("ibi_gt2s"))
        .alias("ibi_gt2s"),
    )
    return joined.select(
        "recording_id",
        "epoch_role",
        *FEATURE_COLUMNS,
        "ibi_gt2s_imputed",
    ).sort(["recording_id", "epoch_role"])


# ---------------------------------------------------------------------------
# Stage 2 — clustering, transitions, permutation tests
# ---------------------------------------------------------------------------


@dataclass
class ClusterResult:
    selected_k: int
    bic_table: Any  # polars DataFrame: k, bic
    assignments: Any  # polars DataFrame: recording_id, epoch_role, label, post_*
    cluster_feature_means: Any  # polars DataFrame per cluster x feature
    z_matrix: np.ndarray
    z_index: Any  # polars DataFrame recording_id, epoch_role aligned with rows
    feature_means: np.ndarray
    feature_stds: np.ndarray
    labels: np.ndarray
    posteriors: np.ndarray


def _zscore_pooled(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0)
    stds = np.where(stds > 0.0, stds, 1.0)
    return (matrix - means) / stds, means, stds


def fit_strategy_clusters(features: Any, config: StrategyStatesConfig) -> ClusterResult:
    """Z-score pooled, PCA(3), full-covariance GMM with BIC-selected k."""

    import polars as pl
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture

    index = features.select("recording_id", "epoch_role")
    matrix = features.select(list(FEATURE_COLUMNS)).to_numpy().astype(np.float64)
    if not np.all(np.isfinite(matrix)):
        raise StrategyStatesInputError("feature matrix contains non-finite values")
    z, means, stds = _zscore_pooled(matrix)
    pca = PCA(n_components=config.pca_components, random_state=config.random_seed)
    reduced = pca.fit_transform(z)

    bic_rows = []
    best_model = None
    best_k = None
    best_bic = math.inf
    for k in range(config.k_min, config.k_max + 1):
        model = GaussianMixture(
            n_components=k,
            covariance_type=config.gmm_covariance_type,
            n_init=config.gmm_n_init,
            random_state=config.random_seed,
        )
        model.fit(reduced)
        bic = float(model.bic(reduced))
        bic_rows.append({"k": k, "bic": bic})
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_model = model
    assert best_model is not None and best_k is not None

    labels = best_model.predict(reduced).astype(np.int64)
    posteriors = best_model.predict_proba(reduced)

    # Relabel clusters by descending size for deterministic reporting.
    counts = np.bincount(labels, minlength=best_k)
    order = np.argsort(-counts, kind="stable")
    remap = np.empty(best_k, dtype=np.int64)
    remap[order] = np.arange(best_k)
    labels = remap[labels]
    posteriors = posteriors[:, order]

    assignments = index.with_columns(
        pl.Series("cluster_label", labels),
    )
    for cluster in range(best_k):
        assignments = assignments.with_columns(
            pl.Series(f"posterior_cluster_{cluster}", posteriors[:, cluster])
        )

    mean_rows = []
    for cluster in range(best_k):
        mask = labels == cluster
        z_mean = z[mask].mean(axis=0)
        raw_mean = matrix[mask].mean(axis=0)
        for feature_index, feature_name in enumerate(FEATURE_COLUMNS):
            mean_rows.append(
                {
                    "cluster_label": cluster,
                    "cluster_size": int(mask.sum()),
                    "feature": feature_name,
                    "mean_z": float(z_mean[feature_index]),
                    "mean_raw": float(raw_mean[feature_index]),
                }
            )

    return ClusterResult(
        selected_k=int(best_k),
        bic_table=pl.DataFrame(bic_rows),
        assignments=assignments,
        cluster_feature_means=pl.DataFrame(mean_rows),
        z_matrix=z,
        z_index=index,
        feature_means=means,
        feature_stds=stds,
        labels=labels,
        posteriors=posteriors,
    )


def g_statistic(table: np.ndarray) -> float:
    """Log-likelihood-ratio (G-test) statistic of independence for a table."""

    observed = np.asarray(table, dtype=np.float64)
    if observed.ndim != 2:
        raise ValueError("contingency table must be 2-dimensional")
    total = observed.sum()
    if total <= 0:
        return 0.0
    row = observed.sum(axis=1, keepdims=True)
    col = observed.sum(axis=0, keepdims=True)
    expected = row @ col / total
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(
            observed > 0.0, observed * np.log(observed / expected), 0.0
        )
    return float(2.0 * np.nansum(terms))


def stratified_permutation_pvalue(
    values: np.ndarray,
    strata: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    *,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Permute ``values`` within each stratum; add-one-corrected p-value.

    Returns ``(observed_statistic, p_value)``.
    """

    values = np.asarray(values)
    strata = np.asarray(strata)
    if values.shape[0] != strata.shape[0]:
        raise ValueError("values and strata must be aligned")
    observed = float(statistic_fn(values))
    stratum_indices = [
        np.flatnonzero(strata == stratum) for stratum in np.unique(strata)
    ]
    exceed = 0
    for _ in range(int(iterations)):
        permuted = values.copy()
        for idx in stratum_indices:
            permuted[idx] = values[idx][rng.permutation(idx.size)]
        if float(statistic_fn(permuted)) >= observed - 1e-12:
            exceed += 1
    p_value = (exceed + 1.0) / (float(iterations) + 1.0)
    return observed, p_value


def classify_responders(
    escape_freeze: Any, config: StrategyStatesConfig
) -> Any:
    """Per-recording responder tier from trial escape/freeze summaries.

    Input columns: recording_id, escape_speed_class (bool), freeze_candidate
    (bool). Tier: escape_dominant if esc_frac >= threshold, else
    freeze_leaning if freeze_frac >= threshold, else mixed.
    """

    import polars as pl

    summary = (
        escape_freeze.group_by("recording_id")
        .agg(
            pl.col("escape_speed_class").cast(pl.Float64).mean().alias("esc_frac"),
            pl.col("freeze_candidate").cast(pl.Float64).mean().alias("freeze_frac"),
            pl.len().alias("trial_count"),
        )
        .sort("recording_id")
    )
    return summary.with_columns(
        pl.when(pl.col("esc_frac") >= config.escape_dominant_threshold)
        .then(pl.lit("escape_dominant"))
        .when(pl.col("freeze_frac") >= config.freeze_leaning_threshold)
        .then(pl.lit("freeze_leaning"))
        .otherwise(pl.lit("mixed"))
        .alias("responder_tier")
    )


def _contingency(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    row_values, row_codes = np.unique(rows, return_inverse=True)
    col_values, col_codes = np.unique(cols, return_inverse=True)
    table = np.zeros((row_values.size, col_values.size), dtype=np.float64)
    np.add.at(table, (row_codes, col_codes), 1.0)
    return table


@dataclass
class TransitionResult:
    per_recording: Any  # polars DataFrame
    hard_matrix: dict[str, Any]
    posterior_flow_matrix: dict[str, Any]
    transition_g: float
    transition_p_value: float
    responder_g: float
    responder_p_value: float
    bootstrap_ari_mean: float
    bootstrap_ari_median: float
    bootstrap_ari_values: list[float]


def compute_transitions(
    cluster_result: ClusterResult,
    responders: Any,
    features: Any,
    config: StrategyStatesConfig,
) -> TransitionResult:
    """Pre-to-post transition structure plus stratified permutation tests."""

    import polars as pl

    assignments = cluster_result.assignments
    k = cluster_result.selected_k
    posterior_cols = [f"posterior_cluster_{c}" for c in range(k)]

    pre = assignments.filter(pl.col("epoch_role") == "chaser_pre").select(
        "recording_id",
        pl.col("cluster_label").alias("pre_cluster"),
        *[pl.col(c).alias(f"pre_{c}") for c in posterior_cols],
    )
    post = assignments.filter(pl.col("epoch_role") == "chaser_post").select(
        "recording_id",
        pl.col("cluster_label").alias("post_cluster"),
        *[pl.col(c).alias(f"post_{c}") for c in posterior_cols],
    )
    paired = pre.join(post, on="recording_id", how="inner").sort("recording_id")
    if paired.height != pre.height or paired.height != post.height:
        raise StrategyStatesInputError(
            "pre/post cluster assignments do not pair one-to-one per recording"
        )
    paired = paired.with_columns(
        pl.col("recording_id")
        .map_elements(parse_arena, return_dtype=pl.String)
        .alias("arena")
    )
    paired = paired.join(responders, on="recording_id", how="left")
    missing_resp = paired.filter(pl.col("responder_tier").is_null())
    if not missing_resp.is_empty():
        raise StrategyStatesInputError(
            "responder tier missing for recordings: "
            f"{missing_resp['recording_id'].to_list()[:5]}"
        )

    pre_labels = paired["pre_cluster"].to_numpy()
    post_labels = paired["post_cluster"].to_numpy()
    arenas = paired["arena"].to_numpy()
    responder_tiers = paired["responder_tier"].to_numpy()

    hard = np.zeros((k, k), dtype=np.float64)
    np.add.at(hard, (pre_labels, post_labels), 1.0)
    pre_post = paired.select(
        [pl.col(f"pre_posterior_cluster_{c}") for c in range(k)]
    ).to_numpy()
    post_post = paired.select(
        [pl.col(f"post_posterior_cluster_{c}") for c in range(k)]
    ).to_numpy()
    flows = pre_post.T @ post_post

    rng = np.random.default_rng(config.random_seed)
    transition_g, transition_p = stratified_permutation_pvalue(
        post_labels,
        arenas,
        lambda perm: g_statistic(_contingency(pre_labels, perm)),
        iterations=config.permutation_iterations,
        rng=rng,
    )

    # Responder -> post-cluster beyond pre-cluster AND arena: permute
    # responder tiers within pre-cluster x arena strata.
    responder_strata = np.array(
        [f"{p}|{a}" for p, a in zip(pre_labels, arenas)], dtype=object
    )
    responder_g, responder_p = stratified_permutation_pvalue(
        responder_tiers.astype(object),
        responder_strata,
        lambda perm: g_statistic(_contingency(perm, post_labels)),
        iterations=config.permutation_iterations,
        rng=rng,
    )

    ari_values = bootstrap_cluster_stability(
        features, cluster_result, config, refits=config.bootstrap_ari_refits
    )

    per_recording = paired.select(
        "recording_id",
        "arena",
        "pre_cluster",
        "post_cluster",
        *[f"pre_posterior_cluster_{c}" for c in range(k)],
        *[f"post_posterior_cluster_{c}" for c in range(k)],
        "esc_frac",
        "freeze_frac",
        "trial_count",
        "responder_tier",
    )

    return TransitionResult(
        per_recording=per_recording,
        hard_matrix={"k": k, "matrix": hard.tolist()},
        posterior_flow_matrix={"k": k, "matrix": flows.tolist()},
        transition_g=float(transition_g),
        transition_p_value=float(transition_p),
        responder_g=float(responder_g),
        responder_p_value=float(responder_p),
        bootstrap_ari_mean=float(np.mean(ari_values)),
        bootstrap_ari_median=float(np.median(ari_values)),
        bootstrap_ari_values=[float(v) for v in ari_values],
    )


def bootstrap_cluster_stability(
    features: Any,
    cluster_result: ClusterResult,
    config: StrategyStatesConfig,
    *,
    refits: int,
) -> list[float]:
    """Refit the full pipeline on recording-level bootstrap resamples.

    Each refit resamples recordings with replacement (both epochs travel
    together), refits z-score/PCA/GMM at the selected k on the resample, then
    predicts all original rows; ARI is computed against the original labels.
    """

    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture

    matrix = features.select(list(FEATURE_COLUMNS)).to_numpy().astype(np.float64)
    recording_ids = features["recording_id"].to_numpy()
    unique_recordings = np.unique(recording_ids)
    rows_by_recording = {
        rec: np.flatnonzero(recording_ids == rec) for rec in unique_recordings
    }
    rng = np.random.default_rng(config.random_seed + 1)
    k = cluster_result.selected_k
    base_labels = cluster_result.labels

    values: list[float] = []
    for refit_index in range(int(refits)):
        sampled = rng.choice(unique_recordings, size=unique_recordings.size, replace=True)
        rows = np.concatenate([rows_by_recording[rec] for rec in sampled])
        sample = matrix[rows]
        z_sample, means, stds = _zscore_pooled(sample)
        pca = PCA(
            n_components=config.pca_components,
            random_state=config.random_seed + 2 + refit_index,
        )
        pca.fit(z_sample)
        model = GaussianMixture(
            n_components=k,
            covariance_type=config.gmm_covariance_type,
            n_init=config.gmm_n_init,
            random_state=config.random_seed + 2 + refit_index,
        )
        model.fit(pca.transform(z_sample))
        z_all = (matrix - means) / stds
        predicted = model.predict(pca.transform(z_all))
        values.append(float(adjusted_rand_score(base_labels, predicted)))
    return values


# ---------------------------------------------------------------------------
# Stage 3 — stimulus-blind window decoder
# ---------------------------------------------------------------------------


def default_epoch_windows(
    epoch_rows: Sequence[Mapping[str, Any]],
    fps: float,
    window_duration_s: float,
) -> list[dict[str, Any]]:
    """Non-overlapping fixed windows fully inside each epoch.

    ``epoch_rows`` carry ``analysis_role``/``start_frame``/
    ``end_frame_exclusive``. Pluggable: a future anticipation probe can pass
    a different window builder to :func:`build_window_features`.
    """

    window_frames = int(round(window_duration_s * fps))
    windows: list[dict[str, Any]] = []
    for row in epoch_rows:
        start = int(row["start_frame"])
        end = int(row["end_frame_exclusive"])
        count = (end - start) // window_frames
        for index in range(count):
            windows.append(
                {
                    "epoch_role": str(row["analysis_role"]),
                    "window_index": index,
                    "start_frame": start + index * window_frames,
                    "end_frame_exclusive": start + (index + 1) * window_frames,
                }
            )
    return windows


def _quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def build_window_features(
    bouts: Any,
    epochs: Any,
    fps_by_recording: Mapping[str, float],
    config: StrategyStatesConfig,
    *,
    window_builder: Callable[..., list[dict[str, Any]]] | None = None,
) -> Any:
    """Per-window stimulus-blind bout features for the decoder.

    ``bouts``: canonical swim bout rows (recording_id, start/end acquisition
    frame ids, duration_s, path_length_mm, net_displacement_mm,
    peak_speed_mm_s, tortuosity). ``epochs``: semantic epoch rows
    (recording_id, analysis_role, start_frame, end_frame_exclusive) already
    limited to the epochs of interest. IBIs are computed between consecutive
    bouts within an epoch and assigned to the window containing the
    interval's start frame.
    """

    import polars as pl

    builder = window_builder or default_epoch_windows
    rows: list[dict[str, Any]] = []
    for recording_id, fps in sorted(fps_by_recording.items()):
        epoch_rows = epochs.filter(pl.col("recording_id") == recording_id).to_dicts()
        if not epoch_rows:
            raise StrategyStatesInputError(
                f"no semantic epochs for recording {recording_id}"
            )
        windows = builder(epoch_rows, fps, config.window_duration_s)
        rec_bouts = (
            bouts.filter(pl.col("recording_id") == recording_id)
            .sort("start_acquisition_frame_id")
        )
        starts = rec_bouts["start_acquisition_frame_id"].to_numpy()
        ends = rec_bouts["end_acquisition_frame_id"].to_numpy()
        duration = rec_bouts["duration_s"].to_numpy()
        path = rec_bouts["path_length_mm"].to_numpy()
        net = rec_bouts["net_displacement_mm"].to_numpy()
        peak = rec_bouts["peak_speed_mm_s"].to_numpy()
        tortuosity = rec_bouts["tortuosity"].to_numpy()

        for epoch_row in epoch_rows:
            epoch_role = str(epoch_row["analysis_role"])
            epoch_start = int(epoch_row["start_frame"])
            epoch_end = int(epoch_row["end_frame_exclusive"])
            in_epoch = (starts >= epoch_start) & (starts < epoch_end)
            e_starts = starts[in_epoch]
            e_ends = ends[in_epoch]
            # IBIs between consecutive bouts inside this epoch.
            ibi_start_frames = e_ends[:-1]
            ibi_s = (e_starts[1:] - e_ends[:-1]) / fps
            keep = ibi_s >= 0.0
            ibi_start_frames = ibi_start_frames[keep]
            ibi_s = ibi_s[keep]

            for window in windows:
                if window["epoch_role"] != epoch_role:
                    continue
                w_start = window["start_frame"]
                w_end = window["end_frame_exclusive"]
                in_window = (e_starts >= w_start) & (e_starts < w_end)
                ibi_in_window = (ibi_start_frames >= w_start) & (
                    ibi_start_frames < w_end
                )
                w_ibi = ibi_s[ibi_in_window]
                w_tort = tortuosity[in_epoch][in_window]
                w_tort = w_tort[np.isfinite(w_tort)]
                rows.append(
                    {
                        "recording_id": recording_id,
                        "epoch_role": epoch_role,
                        "window_index": int(window["window_index"]),
                        "start_frame": int(w_start),
                        "end_frame_exclusive": int(w_end),
                        "n_bouts": int(in_window.sum()),
                        "bout_duration_s_median": _quantile(
                            duration[in_epoch][in_window], 0.5
                        ),
                        "bout_duration_s_q90": _quantile(
                            duration[in_epoch][in_window], 0.9
                        ),
                        "bout_path_length_mm_median": _quantile(
                            path[in_epoch][in_window], 0.5
                        ),
                        "bout_path_length_mm_q10": _quantile(
                            path[in_epoch][in_window], 0.1
                        ),
                        "bout_peak_speed_mm_s_median": _quantile(
                            peak[in_epoch][in_window], 0.5
                        ),
                        "bout_peak_speed_mm_s_q10": _quantile(
                            peak[in_epoch][in_window], 0.1
                        ),
                        "bout_net_displacement_mm_median": _quantile(
                            net[in_epoch][in_window], 0.5
                        ),
                        "bout_tortuosity_finite_median": _quantile(w_tort, 0.5),
                        "ibi_s_median": _quantile(w_ibi, 0.5),
                        "ibi_s_q90": _quantile(w_ibi, 0.9),
                        "ibi_frac_gt_2s": (
                            float(
                                np.mean(w_ibi > config.ibi_long_threshold_s)
                            )
                            if w_ibi.size
                            else float("nan")
                        ),
                    }
                )
    return pl.DataFrame(rows)


@dataclass
class DecoderResult:
    run_name: str
    window_scores: Any  # polars DataFrame
    per_fish: Any  # polars DataFrame
    overall_auc: float
    per_fish_median_auc: float


def run_loro_decoder(
    windows: Any,
    config: StrategyStatesConfig,
    *,
    run_name: str,
    post_window_range: tuple[int, int] | None = None,
) -> DecoderResult:
    """Leave-recording-out logistic decoder pre vs post over window features."""

    import polars as pl
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    frame = windows
    if post_window_range is not None:
        low, high = post_window_range
        frame = frame.filter(
            (pl.col("epoch_role") == "chaser_pre")
            | (
                (pl.col("epoch_role") == "chaser_post")
                & (pl.col("window_index") >= low)
                & (pl.col("window_index") <= high)
            )
        )
    frame = frame.with_columns(
        (pl.col("epoch_role") == "chaser_post").cast(pl.Int64).alias("label")
    ).sort(["recording_id", "epoch_role", "window_index"])

    feature_matrix = frame.select(list(DECODER_FEATURE_COLUMNS)).to_numpy().astype(
        np.float64
    )
    labels = frame["label"].to_numpy()
    recording_ids = frame["recording_id"].to_numpy()

    scores = np.full(labels.shape[0], np.nan, dtype=np.float64)
    for recording_id in np.unique(recording_ids):
        test_mask = recording_ids == recording_id
        train_mask = ~test_mask
        x_train = feature_matrix[train_mask].copy()
        y_train = labels[train_mask]
        if np.unique(y_train).size < 2:
            raise StrategyStatesInputError(
                "training fold has a single class; decoder cannot run"
            )
        medians = np.nanmedian(x_train, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        x_train = np.where(np.isfinite(x_train), x_train, medians)
        means = x_train.mean(axis=0)
        stds = x_train.std(axis=0, ddof=0)
        stds = np.where(stds > 0.0, stds, 1.0)
        x_train = (x_train - means) / stds
        model = LogisticRegression(
            C=config.logistic_c,
            class_weight="balanced",
            max_iter=config.logistic_max_iter,
            random_state=config.random_seed,
        )
        model.fit(x_train, y_train)
        x_test = feature_matrix[test_mask].copy()
        x_test = np.where(np.isfinite(x_test), x_test, medians)
        x_test = (x_test - means) / stds
        scores[test_mask] = model.predict_proba(x_test)[:, 1]

    overall_auc = float(roc_auc_score(labels, scores))
    window_scores = frame.select(
        "recording_id", "epoch_role", "window_index", "label"
    ).with_columns(
        pl.Series("score", scores), pl.lit(run_name).alias("decoder_run")
    )

    per_fish_rows = []
    for recording_id in np.unique(recording_ids):
        mask = recording_ids == recording_id
        y = labels[mask]
        if np.unique(y).size < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y, scores[mask]))
        per_fish_rows.append(
            {
                "recording_id": str(recording_id),
                "decoder_run": run_name,
                "n_windows": int(mask.sum()),
                "n_post_windows": int(y.sum()),
                "auc": auc,
            }
        )
    per_fish = pl.DataFrame(per_fish_rows)
    finite_auc = per_fish.filter(pl.col("auc").is_finite())["auc"]
    per_fish_median = float(finite_auc.median()) if finite_auc.len() else float("nan")
    return DecoderResult(
        run_name=run_name,
        window_scores=window_scores,
        per_fish=per_fish,
        overall_auc=overall_auc,
        per_fish_median_auc=per_fish_median,
    )


# ---------------------------------------------------------------------------
# Stage 4 — direction decomposition
# ---------------------------------------------------------------------------


def punctuated_axis(
    cluster_result: ClusterResult,
) -> tuple[np.ndarray, int, int, bool]:
    """Cluster-difference axis in z-space with the punctuated sign convention.

    Uses the two largest clusters (labels 0 and 1 after size relabeling).
    Sign: +axis points toward lower occupancy entropy, lower bout rate, and a
    heavier long-IBI tail. Returns (unit_axis, cluster_a, cluster_b, flipped)
    where the axis is mean(cluster_b) - mean(cluster_a) before any flip.
    """

    labels = cluster_result.labels
    z = cluster_result.z_matrix
    counts = np.bincount(labels, minlength=cluster_result.selected_k)
    if cluster_result.selected_k < 2:
        raise StrategyStatesInputError(
            "punctuated axis requires at least two clusters"
        )
    order = np.argsort(-counts, kind="stable")
    cluster_a, cluster_b = int(order[0]), int(order[1])
    mean_a = z[labels == cluster_a].mean(axis=0)
    mean_b = z[labels == cluster_b].mean(axis=0)
    axis = mean_b - mean_a
    entropy_i = FEATURE_COLUMNS.index("occupancy_entropy")
    bout_i = FEATURE_COLUMNS.index("bout_rate_per_min")
    ibi_i = FEATURE_COLUMNS.index("ibi_gt2s")
    sign_score = -axis[entropy_i] - axis[bout_i] + axis[ibi_i]
    flipped = sign_score < 0.0
    if flipped:
        axis = -axis
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise StrategyStatesInputError("punctuated axis has zero length")
    return axis / norm, cluster_a, cluster_b, bool(flipped)


def decompose_displacements(
    cluster_result: ClusterResult, unit_axis: np.ndarray
) -> Any:
    """Per-fish z-space displacement split into parallel/orthogonal parts."""

    import polars as pl

    index = cluster_result.z_index.with_row_index("row")
    pre = index.filter(pl.col("epoch_role") == "chaser_pre")
    post = index.filter(pl.col("epoch_role") == "chaser_post")
    paired = pre.join(post, on="recording_id", how="inner", suffix="_post")
    rows = []
    for record in paired.to_dicts():
        d = (
            cluster_result.z_matrix[record["row_post"]]
            - cluster_result.z_matrix[record["row"]]
        )
        c_par = float(np.dot(d, unit_axis))
        residual = d - c_par * unit_axis
        c_orth = float(np.linalg.norm(residual))
        rows.append(
            {
                "recording_id": record["recording_id"],
                "displacement_norm": float(np.linalg.norm(d)),
                "component_parallel": c_par,
                "component_orthogonal": c_orth,
            }
        )
    return pl.DataFrame(rows).sort("recording_id")


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    from scipy.stats import spearmanr

    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    result = spearmanr(xa[mask], ya[mask])
    return float(result.statistic), float(result.pvalue)


def run_disposition_decoder(
    features: Any,
    responders: Any,
    config: StrategyStatesConfig,
) -> tuple[float, Any]:
    """LORO logistic regression: PRE features -> escape-dominant responder."""

    import polars as pl
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    pre = features.filter(pl.col("epoch_role") == "chaser_pre").join(
        responders.select("recording_id", "responder_tier", "esc_frac"),
        on="recording_id",
        how="inner",
    ).sort("recording_id")
    matrix = pre.select(list(FEATURE_COLUMNS)).to_numpy().astype(np.float64)
    labels = (
        pre["responder_tier"].to_numpy() == "escape_dominant"
    ).astype(np.int64)
    recording_ids = pre["recording_id"].to_numpy()
    if np.unique(labels).size < 2:
        raise StrategyStatesInputError(
            "disposition decoder needs both responder classes"
        )
    scores = np.full(labels.shape[0], np.nan, dtype=np.float64)
    for index in range(labels.shape[0]):
        mask = np.ones(labels.shape[0], dtype=bool)
        mask[index] = False
        if np.unique(labels[mask]).size < 2:
            continue
        x_train = matrix[mask]
        means = x_train.mean(axis=0)
        stds = x_train.std(axis=0, ddof=0)
        stds = np.where(stds > 0.0, stds, 1.0)
        model = LogisticRegression(
            C=config.logistic_c,
            class_weight="balanced",
            max_iter=config.logistic_max_iter,
            random_state=config.random_seed,
        )
        model.fit((x_train - means) / stds, labels[mask])
        scores[index] = model.predict_proba(
            ((matrix[index] - means) / stds).reshape(1, -1)
        )[0, 1]
    valid = np.isfinite(scores)
    auc = float(roc_auc_score(labels[valid], scores[valid]))
    score_frame = pl.DataFrame(
        {
            "recording_id": [str(r) for r in recording_ids],
            "escape_dominant": labels.astype(bool),
            "disposition_score": scores,
        }
    )
    return auc, score_frame


# ---------------------------------------------------------------------------
# Export-table gathering (used by the CLI; tests use synthetic frames)
# ---------------------------------------------------------------------------


def gather_export_inputs(dataset: Any, config: StrategyStatesConfig) -> dict[str, Any]:
    """Pull all needed slices from an opened validated-behavior export."""

    import polars as pl

    wall = (
        dataset.table("radial_near_field_summary")
        .scan(
            columns=[
                "recording_id",
                "provider_role",
                "epoch_role",
                "behavior_role",
                "fish_wall_distance_p50_mm",
            ],
            predicate=(
                (pl.col("provider_role") == "keypoint")
                & (pl.col("behavior_role") == "aggressive")
                & pl.col("epoch_role").is_in(list(EPOCH_ROLES))
            ),
        )
        .collect()
        .select("recording_id", "epoch_role", "fish_wall_distance_p50_mm")
    )

    occupancy = (
        dataset.table("spatial_occupancy_bins")
        .scan(
            columns=[
                "recording_id",
                "provider_role",
                "epoch_role",
                "occupancy_fraction_candidate_epoch",
            ],
            predicate=(
                (pl.col("provider_role") == "keypoint")
                & pl.col("epoch_role").is_in(list(EPOCH_ROLES))
                & (pl.col("occupancy_fraction_candidate_epoch") > 0.0)
            ),
        )
        .collect()
    )
    entropy = (
        occupancy.group_by(["recording_id", "epoch_role"])
        .agg(pl.col("occupancy_fraction_candidate_epoch").alias("fractions"))
        .with_columns(
            pl.col("fractions")
            .map_elements(occupancy_entropy, return_dtype=pl.Float64)
            .alias("occupancy_entropy")
        )
        .select("recording_id", "epoch_role", "occupancy_entropy")
    )

    bout = (
        dataset.table("epoch_behavior_summary")
        .scan(
            columns=[
                "recording_id",
                "window_label",
                "bout_rate_per_min",
                "mean_abs_bout_net_heading_change_deg",
            ],
            predicate=pl.col("window_label").is_in(list(EPOCH_ROLES)),
        )
        .collect()
        .rename({"window_label": "epoch_role"})
    )

    epochs = (
        dataset.table("semantic_epochs")
        .scan(
            columns=[
                "recording_id",
                "analysis_role",
                "start_frame",
                "end_frame_exclusive",
            ],
            predicate=pl.col("analysis_role").is_in(list(EPOCH_ROLES)),
        )
        .collect()
    )

    fps_frame = (
        dataset.table("provider_motion_samples")
        .scan(
            columns=[
                "recording_id",
                "provider_role",
                "acquisition_frame_id",
                "time_s",
            ],
            predicate=pl.col("provider_role") == "keypoint",
        )
        .group_by("recording_id")
        .agg(
            pl.col("acquisition_frame_id").min().alias("frame_min"),
            pl.col("acquisition_frame_id").max().alias("frame_max"),
            pl.col("time_s").min().alias("time_min"),
            pl.col("time_s").max().alias("time_max"),
        )
        .collect()
    )
    fps_by_recording: dict[str, float] = {}
    for row in fps_frame.to_dicts():
        span_s = float(row["time_max"]) - float(row["time_min"])
        span_frames = int(row["frame_max"]) - int(row["frame_min"])
        if span_s <= 0.0 or span_frames <= 0:
            raise StrategyStatesInputError(
                f"cannot derive fps for recording {row['recording_id']}"
            )
        fps = span_frames / span_s
        if abs(fps - config.expected_fps) > (
            config.expected_fps * config.fps_relative_tolerance
        ):
            raise StrategyStatesInputError(
                f"recording {row['recording_id']} fps {fps:.4f} deviates more "
                f"than {config.fps_relative_tolerance:.2%} from "
                f"{config.expected_fps}"
            )
        fps_by_recording[str(row["recording_id"])] = float(fps)

    bouts = (
        dataset.table("canonical_swim_bouts")
        .scan(
            columns=[
                "recording_id",
                "start_acquisition_frame_id",
                "end_acquisition_frame_id",
                "duration_s",
                "path_length_mm",
                "net_displacement_mm",
                "peak_speed_mm_s",
                "tortuosity",
            ]
        )
        .collect()
    )

    escape_freeze = (
        dataset.table("trial_escape_freeze_summaries")
        .scan(
            columns=[
                "recording_id",
                "behavior_role",
                "escape_speed_class",
                "freeze_candidate",
            ],
            predicate=pl.col("behavior_role") == "aggressive",
        )
        .collect()
        .select("recording_id", "escape_speed_class", "freeze_candidate")
    )

    return {
        "wall": wall,
        "entropy": entropy,
        "bout": bout,
        "epochs": epochs,
        "fps_by_recording": fps_by_recording,
        "bouts": bouts,
        "escape_freeze": escape_freeze,
    }


# ---------------------------------------------------------------------------
# Orchestration + persistence
# ---------------------------------------------------------------------------


@dataclass
class StrategyStatesOutputs:
    features: Any
    cluster_result: ClusterResult
    transitions: TransitionResult
    decoder_results: list[DecoderResult]
    direction: Any
    direction_stats: dict[str, Any]
    disposition: dict[str, Any]


def compute_strategy_states(
    *,
    twin: Any,
    ibi: Any,
    export_inputs: Mapping[str, Any],
    config: StrategyStatesConfig,
    post_window_range: tuple[int, int] | None = None,
) -> StrategyStatesOutputs:
    """Run all four stages on already-loaded inputs."""

    import polars as pl

    features = assemble_feature_matrix(
        twin=twin,
        wall=export_inputs["wall"],
        entropy=export_inputs["entropy"],
        bout=export_inputs["bout"],
        ibi=ibi,
    )
    cluster_result = fit_strategy_clusters(features, config)
    responders = classify_responders(export_inputs["escape_freeze"], config)
    transitions = compute_transitions(cluster_result, responders, features, config)

    windows = build_window_features(
        export_inputs["bouts"],
        export_inputs["epochs"],
        export_inputs["fps_by_recording"],
        config,
    )
    decoder_results = [
        run_loro_decoder(windows, config, run_name="all_windows")
    ]
    if post_window_range is not None:
        low, high = post_window_range
        decoder_results.append(
            run_loro_decoder(
                windows,
                config,
                run_name=f"post_windows_{low}_{high}",
                post_window_range=post_window_range,
            )
        )

    unit_axis, cluster_a, cluster_b, flipped = punctuated_axis(cluster_result)
    direction = decompose_displacements(cluster_result, unit_axis)
    per_fish_all = decoder_results[0].per_fish.select(
        "recording_id", pl.col("auc").alias("per_fish_auc")
    )
    direction = (
        direction.join(
            responders.select("recording_id", "esc_frac", "freeze_frac", "responder_tier"),
            on="recording_id",
            how="left",
        )
        .join(per_fish_all, on="recording_id", how="left")
    )

    direction_stats: dict[str, Any] = {
        "axis_cluster_low": cluster_a,
        "axis_cluster_high": cluster_b,
        "axis_sign_flipped": flipped,
        "unit_axis_z": {
            name: float(value)
            for name, value in zip(FEATURE_COLUMNS, unit_axis)
        },
    }
    for metric in ("esc_frac", "freeze_frac", "per_fish_auc"):
        for component in ("component_parallel", "component_orthogonal"):
            rho, p_value = spearman_rho(
                direction[metric].to_numpy(), direction[component].to_numpy()
            )
            direction_stats[f"spearman_{metric}_vs_{component}"] = {
                "rho": rho,
                "p_value": p_value,
            }

    pre_features = features.filter(pl.col("epoch_role") == "chaser_pre").join(
        responders.select("recording_id", "esc_frac"), on="recording_id", how="inner"
    )
    disposition: dict[str, Any] = {"pre_feature_vs_esc_frac": {}}
    for feature_name in FEATURE_COLUMNS:
        rho, p_value = spearman_rho(
            pre_features[feature_name].to_numpy(),
            pre_features["esc_frac"].to_numpy(),
        )
        disposition["pre_feature_vs_esc_frac"][feature_name] = {
            "rho": rho,
            "p_value": p_value,
        }
    disposition_auc, disposition_scores = run_disposition_decoder(
        features, responders, config
    )
    disposition["loro_pre_features_to_escape_dominant_auc"] = disposition_auc
    direction = direction.join(
        disposition_scores.select("recording_id", "disposition_score"),
        on="recording_id",
        how="left",
    )

    return StrategyStatesOutputs(
        features=features,
        cluster_result=cluster_result,
        transitions=transitions,
        decoder_results=decoder_results,
        direction=direction,
        direction_stats=direction_stats,
        disposition=disposition,
    )


def write_strategy_states_outputs(
    outputs: StrategyStatesOutputs,
    *,
    output_dir: Path,
    config: StrategyStatesConfig,
    source_export: Mapping[str, Any],
    input_parquets: Mapping[str, Mapping[str, str]],
    overwrite: bool = False,
) -> Path:
    """Persist all parquet outputs plus the sha256 manifest; returns manifest path."""

    import polars as pl

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster = outputs.cluster_result
    transitions = outputs.transitions

    assignments = cluster.assignments.join(
        outputs.features.select("recording_id", "epoch_role", "ibi_gt2s_imputed"),
        on=["recording_id", "epoch_role"],
        how="left",
    )
    frames: dict[str, Any] = {
        "strategy_features.parquet": outputs.features,
        "strategy_cluster_assignments.parquet": assignments,
        "strategy_bic.parquet": cluster.bic_table,
        "strategy_cluster_means.parquet": cluster.cluster_feature_means,
        "strategy_transitions.parquet": transitions.per_recording,
        "decoder_scores.parquet": pl.concat(
            [result.window_scores for result in outputs.decoder_results]
        ),
        "decoder_per_fish.parquet": pl.concat(
            [result.per_fish for result in outputs.decoder_results]
        ),
        "direction_decomposition.parquet": outputs.direction,
    }

    file_hashes: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    for name, frame in frames.items():
        path = output_dir / name
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite to replace it")
        stamped = _stamp(frame)
        stamped.write_parquet(path)
        file_hashes[name] = sha256_file(path)
        row_counts[name] = stamped.height

    cluster_sizes = (
        cluster.cluster_feature_means.select("cluster_label", "cluster_size")
        .unique()
        .sort("cluster_label")
        .to_dicts()
    )

    manifest: dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "analysis_status": ANALYSIS_STATUS,
        "acquisition_batch_adjustment": ACQUISITION_BATCH_ADJUSTMENT,
        "source_export": dict(source_export),
        "input_parquets": {k: dict(v) for k, v in input_parquets.items()},
        "parameters": config.to_dict(),
        "files": file_hashes,
        "row_counts": row_counts,
        "results": {
            "bic_table": cluster.bic_table.to_dicts(),
            "selected_k": cluster.selected_k,
            "cluster_sizes": cluster_sizes,
            "transition_hard_matrix": transitions.hard_matrix,
            "transition_posterior_flow_matrix": transitions.posterior_flow_matrix,
            "transition_g": transitions.transition_g,
            "transition_permutation_p_value": transitions.transition_p_value,
            "responder_g": transitions.responder_g,
            "responder_permutation_p_value": transitions.responder_p_value,
            "bootstrap_ari_mean": transitions.bootstrap_ari_mean,
            "bootstrap_ari_median": transitions.bootstrap_ari_median,
            "decoders": [
                {
                    "run": result.run_name,
                    "overall_window_auc": result.overall_auc,
                    "per_fish_median_auc": result.per_fish_median_auc,
                }
                for result in outputs.decoder_results
            ],
            "direction": outputs.direction_stats,
            "disposition": outputs.disposition,
        },
    }
    manifest["canonical_json_sha256"] = canonical_json_sha256(manifest)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass overwrite to replace it")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest_path


__all__ = [
    "ACQUISITION_BATCH_ADJUSTMENT",
    "ANALYSIS_STATUS",
    "DECODER_FEATURE_COLUMNS",
    "EPOCH_ROLES",
    "FEATURE_COLUMNS",
    "ClusterResult",
    "DecoderResult",
    "SPEC_VERSION",
    "StrategyStatesConfig",
    "StrategyStatesInputError",
    "StrategyStatesOutputs",
    "TransitionResult",
    "assemble_feature_matrix",
    "bootstrap_cluster_stability",
    "build_window_features",
    "canonical_json_sha256",
    "classify_responders",
    "compute_strategy_states",
    "compute_transitions",
    "decompose_displacements",
    "default_epoch_windows",
    "fit_strategy_clusters",
    "g_statistic",
    "gather_export_inputs",
    "load_ibi_cell_features",
    "load_twin_excess_features",
    "occupancy_entropy",
    "parse_arena",
    "punctuated_axis",
    "run_disposition_decoder",
    "run_loro_decoder",
    "sha256_file",
    "spearman_rho",
    "stratified_permutation_pvalue",
    "write_strategy_states_outputs",
]
