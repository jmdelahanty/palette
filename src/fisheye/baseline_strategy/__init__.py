"""Read-only baseline behavioral-strategy analytics.

This package combines fish-specific bout and wall-affinity measures with
rodent open-field concepts such as progression episodes, dominant dwell zones,
and excursions.  It consumes immutable analytics-export rows and never writes
back to a recording Zarr or source export.
"""

from .cohort import classify_strategy_features, discover_strategy_clusters
from .contracts import (
    BASELINE_EXPLORATION_EPISODES_TABLE,
    BASELINE_STRATEGY_CLASSIFICATION_TABLE,
    BASELINE_STRATEGY_CLUSTERS_TABLE,
    BASELINE_STRATEGY_FEATURES_TABLE,
    StrategyFeatureConfig,
)
from .features import derive_baseline_strategy_features, derive_exploration_episodes
from .query import scan_strategy_table, strategy_table_parts
from .validation import StrategyAnalyticsValidationError, validate_strategy_analytics_run

__all__ = [
    "BASELINE_EXPLORATION_EPISODES_TABLE",
    "BASELINE_STRATEGY_CLASSIFICATION_TABLE",
    "BASELINE_STRATEGY_CLUSTERS_TABLE",
    "BASELINE_STRATEGY_FEATURES_TABLE",
    "StrategyFeatureConfig",
    "StrategyAnalyticsValidationError",
    "classify_strategy_features",
    "derive_baseline_strategy_features",
    "derive_exploration_episodes",
    "discover_strategy_clusters",
    "scan_strategy_table",
    "strategy_table_parts",
    "validate_strategy_analytics_run",
]
