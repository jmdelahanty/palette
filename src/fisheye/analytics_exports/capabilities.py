"""Capability resolution from strict V2 analytics table contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .contracts import (
    BASELINE_BEHAVIOR_SUMMARY_TABLE,
    BASELINE_BEHAVIOR_TIME_BINS_TABLE,
    BASELINE_KINEMATIC_SAMPLES_TABLE,
    CHASER_BOUT_EVENTS_TABLE,
    CHASER_BOUT_HISTOGRAM_TABLE,
    CHASER_CENTER_DISTANCE_HISTOGRAM_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_DISTANCE_CDF_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_CHASER_PHASE_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_RADIAL_DENSITY_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_SUMMARY_TABLE,
    CHASER_QUADRANT_OCCUPANCY_CHASER_PHASE_TABLE,
    CHASER_QUADRANT_OCCUPANCY_DENSITY_TABLE,
    CHASER_QUADRANT_OCCUPANCY_SUMMARY_TABLE,
    CHASER_DISTANCE_HISTOGRAM_TABLE,
    CHASER_DISTANCE_SUMMARY_TABLE,
    CHASER_EGOCENTRIC_HISTOGRAM_TABLE,
    CHASER_EGOCENTRIC_SUMMARY_TABLE,
    CHASER_EPOCH_BEHAVIOR_TABLE,
    CHASER_IBI_HISTOGRAM_TABLE,
    CHASER_SPATIAL_TABLE,
    CHASER_SPEED_DISTANCE_TABLE,
    DESCRIPTIVE_TABLE,
    EYE_TRACE_SAMPLES_TABLE,
    KINEMATICS_SAMPLES_TABLE,
    POSITION_OCCUPANCY_HISTOGRAM_TABLE,
    STATISTICS_TABLE,
    TABLE_CONTRACTS,
    validate_table_columns,
)


@dataclass(frozen=True)
class CapabilityDefinition:
    capability_id: str
    required_tables: tuple[str, ...]


@dataclass(frozen=True)
class CapabilityStatus:
    capability_id: str
    available: bool
    required_tables: tuple[str, ...]
    missing_tables: tuple[str, ...]
    missing_columns_by_table: Mapping[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, object]:
        return {
            "capability_id": self.capability_id,
            "available": self.available,
            "required_tables": list(self.required_tables),
            "missing_tables": list(self.missing_tables),
            "missing_columns_by_table": {
                table: list(columns)
                for table, columns in self.missing_columns_by_table.items()
            },
        }


CAPABILITY_DEFINITIONS = (
    CapabilityDefinition(
        "core.baseline.behavior_summary",
        (BASELINE_BEHAVIOR_SUMMARY_TABLE,),
    ),
    CapabilityDefinition(
        "core.baseline.behavior_time_bins",
        (BASELINE_BEHAVIOR_TIME_BINS_TABLE,),
    ),
    CapabilityDefinition(
        "core.baseline.kinematic_samples",
        (BASELINE_KINEMATIC_SAMPLES_TABLE,),
    ),
    CapabilityDefinition("core.eye.trace_samples", (EYE_TRACE_SAMPLES_TABLE,)),
    CapabilityDefinition("core.kinematics.samples", (KINEMATICS_SAMPLES_TABLE,)),
    CapabilityDefinition(
        "position.epoch.occupancy_histogram_2d",
        (POSITION_OCCUPANCY_HISTOGRAM_TABLE,),
    ),
    CapabilityDefinition("chaser.epoch.behavior_summary", (CHASER_EPOCH_BEHAVIOR_TABLE,)),
    CapabilityDefinition("chaser.epoch.bout_events", (CHASER_BOUT_EVENTS_TABLE,)),
    CapabilityDefinition("chaser.epoch.bout_histogram", (CHASER_BOUT_HISTOGRAM_TABLE,)),
    CapabilityDefinition("chaser.epoch.inter_bout_interval_histogram", (CHASER_IBI_HISTOGRAM_TABLE,)),
    CapabilityDefinition("chaser.epoch.spatial_occupancy", (CHASER_SPATIAL_TABLE,)),
    CapabilityDefinition(
        "chaser.epoch.center_distance_histogram",
        (CHASER_CENTER_DISTANCE_HISTOGRAM_TABLE,),
    ),
    CapabilityDefinition("chaser.distance.summary", (CHASER_DISTANCE_SUMMARY_TABLE,)),
    CapabilityDefinition("chaser.distance.histogram", (CHASER_DISTANCE_HISTOGRAM_TABLE,)),
    CapabilityDefinition("chaser.distance.speed_relationship", (CHASER_SPEED_DISTANCE_TABLE,)),
    CapabilityDefinition(
        "chaser.quadrant_occupancy",
        (
            CHASER_QUADRANT_OCCUPANCY_SUMMARY_TABLE,
            CHASER_QUADRANT_OCCUPANCY_CHASER_PHASE_TABLE,
            CHASER_QUADRANT_OCCUPANCY_DENSITY_TABLE,
        ),
    ),
    CapabilityDefinition(
        "chaser.near_field_occupancy",
        (
            CHASER_NEAR_FIELD_OCCUPANCY_SUMMARY_TABLE,
            CHASER_NEAR_FIELD_OCCUPANCY_CHASER_PHASE_TABLE,
            CHASER_NEAR_FIELD_OCCUPANCY_RADIAL_DENSITY_TABLE,
            CHASER_NEAR_FIELD_OCCUPANCY_DISTANCE_CDF_TABLE,
        ),
    ),
    CapabilityDefinition(
        "chaser.egocentric",
        (CHASER_EGOCENTRIC_SUMMARY_TABLE, CHASER_EGOCENTRIC_HISTOGRAM_TABLE),
    ),
    CapabilityDefinition("group.statistics", (STATISTICS_TABLE,)),
    CapabilityDefinition("group.descriptive_statistics", (DESCRIPTIVE_TABLE,)),
)


def resolve_capabilities(
    columns_by_table: Mapping[str, Sequence[str]],
) -> tuple[CapabilityStatus, ...]:
    """Resolve every known capability without accepting legacy table aliases."""

    statuses: list[CapabilityStatus] = []
    for definition in CAPABILITY_DEFINITIONS:
        missing_tables = tuple(
            table for table in definition.required_tables if table not in columns_by_table
        )
        missing_columns: dict[str, tuple[str, ...]] = {}
        for table in definition.required_tables:
            if table not in columns_by_table or table not in TABLE_CONTRACTS:
                continue
            missing = validate_table_columns(table, columns_by_table[table])
            if missing:
                missing_columns[table] = missing
        statuses.append(
            CapabilityStatus(
                capability_id=definition.capability_id,
                available=not missing_tables and not missing_columns,
                required_tables=definition.required_tables,
                missing_tables=missing_tables,
                missing_columns_by_table=missing_columns,
            )
        )
    return tuple(statuses)


__all__ = [
    "CAPABILITY_DEFINITIONS",
    "CapabilityDefinition",
    "CapabilityStatus",
    "resolve_capabilities",
]
