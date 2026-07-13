"""Contracts for whole-training chaser response classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SCHEMA_ID = "palette.training_response_analytics"
SCHEMA_VERSION = 1
METHOD = "whole_training_chaser_response"
METHOD_VERSION = "1"

TRAINING_RESPONSE_FEATURES_TABLE = "training_response_features"
TRAINING_RESPONSE_CLASSIFICATION_TABLE = "training_response_classification"
TRAINING_RESPONSE_CLUSTERS_TABLE = "training_response_clusters"

IDENTITY_COLUMNS = ("recording_id", "training_window_id")


@dataclass(frozen=True)
class TrainingResponseConfig:
    pre_window_label: str = "pre_event"
    training_window_label: str = "training_event"
    aggressive_role: str = "aggressive"
    inert_role: str = "inert"
    min_valid_position_fraction: float = 0.75
    min_training_duration_s: float = 30.0
    relative_score_threshold: float = 0.75
    cluster_probability_threshold: float = 0.65
    cluster_stability_threshold: float = 0.60
    cluster_min_rows_per_component: int = 10
    cluster_max_components: int = 6
    cluster_stability_resamples: int = 25
    random_seed: int = 0

    def validate(self) -> None:
        if not self.pre_window_label or not self.training_window_label:
            raise ValueError("pre and training window labels are required")
        if self.pre_window_label == self.training_window_label:
            raise ValueError("pre and training window labels must differ")
        if not 0 <= self.min_valid_position_fraction <= 1:
            raise ValueError("min_valid_position_fraction must be in [0, 1]")
        if self.min_training_duration_s <= 0:
            raise ValueError("min_training_duration_s must be positive")
        if self.relative_score_threshold <= 0:
            raise ValueError("relative_score_threshold must be positive")
        if not 0 < self.cluster_probability_threshold <= 1:
            raise ValueError("cluster_probability_threshold must be in (0, 1]")
        if not 0 <= self.cluster_stability_threshold <= 1:
            raise ValueError("cluster_stability_threshold must be in [0, 1]")
        if self.cluster_min_rows_per_component < 2:
            raise ValueError("cluster_min_rows_per_component must be at least 2")
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
        "source_export_run_id",
    )
    specific = {
        TRAINING_RESPONSE_FEATURES_TABLE: (
            "feature_status",
            "feature_reason",
            "pre_window_label",
            "training_window_label",
            "training_duration_s",
        ),
        TRAINING_RESPONSE_CLASSIFICATION_TABLE: (
            "classification_status",
            "locomotor_response",
            "boundary_response",
            "aggressive_proximity_state",
            "role_distance_selectivity",
            "close_contact_vigor",
            "primary_training_profile",
        ),
        TRAINING_RESPONSE_CLUSTERS_TABLE: (
            "cluster_status",
            "cluster_id",
            "cluster_probability",
        ),
    }
    if table_name not in specific:
        raise KeyError(f"unknown training-response table: {table_name}")
    return (*common, *specific[table_name])


__all__ = [
    "IDENTITY_COLUMNS",
    "METHOD",
    "METHOD_VERSION",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "TRAINING_RESPONSE_CLASSIFICATION_TABLE",
    "TRAINING_RESPONSE_CLUSTERS_TABLE",
    "TRAINING_RESPONSE_FEATURES_TABLE",
    "TrainingResponseConfig",
    "contract_fields",
]
