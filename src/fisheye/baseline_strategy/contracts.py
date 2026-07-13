"""Contracts and configuration for baseline strategy analytics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SCHEMA_ID = "palette.baseline_strategy_analytics"
SCHEMA_VERSION = 1
METHOD = "fish_rodent_open_field_strategy_features"
METHOD_VERSION = "1"

BASELINE_STRATEGY_FEATURES_TABLE = "baseline_strategy_features"
BASELINE_EXPLORATION_EPISODES_TABLE = "baseline_exploration_episodes"
BASELINE_STRATEGY_CLASSIFICATION_TABLE = "baseline_strategy_classification"
BASELINE_STRATEGY_CLUSTERS_TABLE = "baseline_strategy_clusters"

IDENTITY_COLUMNS = (
    "recording_id",
    "track_id",
    "baseline_window_id",
)


@dataclass(frozen=True)
class StrategyFeatureConfig:
    """Declared policy for geometry, activity, and QC feature derivation.

    Defaults are operational starting points, not biological truths.  They are
    serialized into every output row so a cohort result can be reproduced and
    compared only with compatible policies.
    """

    active_speed_mm_s: float = 1.0
    max_episode_gap_s: float = 0.5
    min_episode_duration_s: float = 0.2
    spatial_grid_size: int = 12
    dwell_grid_size: int = 8
    early_late_fraction: float = 0.25
    min_valid_position_fraction: float = 0.75
    min_sample_count: int = 20
    wall_following_tangential_threshold: float = 0.7
    relative_score_threshold: float = 0.75
    cluster_probability_threshold: float = 0.65
    cluster_max_components: int = 6
    cluster_stability_resamples: int = 25
    random_seed: int = 0

    def validate(self) -> None:
        if self.active_speed_mm_s < 0:
            raise ValueError("active_speed_mm_s must be non-negative")
        if self.max_episode_gap_s <= 0:
            raise ValueError("max_episode_gap_s must be positive")
        if self.min_episode_duration_s < 0:
            raise ValueError("min_episode_duration_s must be non-negative")
        if self.spatial_grid_size < 2 or self.dwell_grid_size < 2:
            raise ValueError("spatial and dwell grids must each be at least 2")
        if not 0 < self.early_late_fraction <= 0.5:
            raise ValueError("early_late_fraction must be in (0, 0.5]")
        if not 0 <= self.min_valid_position_fraction <= 1:
            raise ValueError("min_valid_position_fraction must be in [0, 1]")
        if self.min_sample_count < 1:
            raise ValueError("min_sample_count must be positive")
        if not 0 <= self.wall_following_tangential_threshold <= 1:
            raise ValueError("wall_following_tangential_threshold must be in [0, 1]")
        if self.relative_score_threshold <= 0:
            raise ValueError("relative_score_threshold must be positive")
        if not 0 < self.cluster_probability_threshold <= 1:
            raise ValueError("cluster_probability_threshold must be in (0, 1]")
        if self.cluster_max_components < 1:
            raise ValueError("cluster_max_components must be positive")
        if self.cluster_stability_resamples < 0:
            raise ValueError("cluster_stability_resamples must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def contract_fields(table_name: str) -> tuple[str, ...]:
    common = (
        "schema_id",
        "schema_version",
        "table_name",
        "method",
        "method_version",
        *IDENTITY_COLUMNS,
    )
    specific = {
        BASELINE_STRATEGY_FEATURES_TABLE: (
            "feature_status",
            "feature_reason",
            "sample_features_available",
            "time_bin_features_available",
        ),
        BASELINE_EXPLORATION_EPISODES_TABLE: (
            "episode_id",
            "episode_start_s",
            "episode_end_s",
            "episode_duration_s",
        ),
        BASELINE_STRATEGY_CLASSIFICATION_TABLE: (
            "classification_status",
            "activity_state",
            "boundary_strategy",
            "spatial_organization",
            "temporal_pattern",
            "primary_strategy",
        ),
        BASELINE_STRATEGY_CLUSTERS_TABLE: (
            "cluster_status",
            "cluster_id",
            "cluster_probability",
        ),
    }
    if table_name not in specific:
        raise KeyError(f"unknown baseline strategy table: {table_name}")
    return (*common, *specific[table_name])


__all__ = [
    "BASELINE_EXPLORATION_EPISODES_TABLE",
    "BASELINE_STRATEGY_CLASSIFICATION_TABLE",
    "BASELINE_STRATEGY_CLUSTERS_TABLE",
    "BASELINE_STRATEGY_FEATURES_TABLE",
    "IDENTITY_COLUMNS",
    "METHOD",
    "METHOD_VERSION",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "StrategyFeatureConfig",
    "contract_fields",
]
