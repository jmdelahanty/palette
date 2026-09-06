"""One core-behavior authority plus additive chaser export relations.

The profile remains an installed table suite on ``validated_behavior/v1``.
It does not add a publisher, manifest, selector, or reader.  Complete core
facts are supplied by the five-grain core profile; the chaser component is an
audited subtraction of Phase C with its competing motion/body projections and
duplicate bout table removed before collision-checked composition.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Mapping

from .arrow_contract_core import field
from .validated_behavior_contracts import (
    ValidatedBehaviorTableSpec,
    compose_disjoint_table_specs,
)
from .validated_behavior_core_behavior_contracts import CORE_BEHAVIOR_TABLE_SPECS
from .validated_behavior_phase_c_contracts import PHASE_C_TABLE_SPECS

CORE_CHASER_EXPORT_PROFILE_ID = "validated_core_behavior_chaser_v1"
CORE_BOUND_CHASER_RELATIVE_PAIR_CAPABILITY = "core_bound_chaser_relative_pair"

# These names are removed deliberately, not overwritten during composition.
# The first three are owned by the generic/core suite; the next three are
# competing core-like Phase-C projections; recording_source_bindings is
# represented by the composite bundle/receipt rather than a second authority
# table.
CHASER_EXTENSION_OMITTED_TABLES = frozenset(
    {
        "cohort_recordings",
        "recording_bundles",
        "recording_capabilities",
        "recording_source_bindings",
        "provider_motion_samples",
        "bout_detector_signal_samples",
        "body_frame_samples",
        "canonical_swim_bouts",
    }
)


def _extension_specs() -> Mapping[str, ValidatedBehaviorTableSpec]:
    retained = {
        name: spec
        for name, spec in PHASE_C_TABLE_SPECS.items()
        if name not in CHASER_EXTENSION_OMITTED_TABLES
    }

    chaser_relative = retained["chaser_relative_samples"]
    retained["chaser_relative_samples"] = replace(
        chaser_relative,
        required_capability=CORE_BOUND_CHASER_RELATIVE_PAIR_CAPABILITY,
        foreign_keys=chaser_relative.foreign_keys
        + (
            (
                (
                    "export_run_id",
                    "recording_id",
                    "track_sample_id",
                ),
                "kinematics_samples",
                (
                    "export_run_id",
                    "recording_id",
                    "track_sample_index",
                ),
            ),
        ),
        semantic_metadata=chaser_relative.semantic_metadata
        + (
            (
                "fish_motion_authority",
                "foreign_key_to_selected_core_kinematics_samples",
            ),
            (
                "fish_position_columns",
                "receipt_bound_denormalized_core_projection_not_authority",
            ),
        ),
    )

    body_relative = retained["body_relative_samples"]
    retained["body_relative_samples"] = replace(
        body_relative,
        contract=replace(
            body_relative.contract,
            fields=body_relative.contract.fields
            + (
                field(
                    "core_subject_shape_row_index",
                    "int64",
                    nullable=True,
                ),
            ),
        ),
        foreign_keys=tuple(
            foreign_key
            for foreign_key in body_relative.foreign_keys
            if foreign_key[1] != "body_frame_samples"
        )
        + (
            (
                (
                    "export_run_id",
                    "recording_id",
                    "core_subject_shape_row_index",
                ),
                "subject_body_frame_samples",
                (
                    "export_run_id",
                    "recording_id",
                    "subject_shape_row_index",
                ),
            ),
        ),
        semantic_metadata=body_relative.semantic_metadata
        + (
            (
                "body_frame_authority",
                "foreign_key_to_selected_core_subject_body_frame_samples",
            ),
            (
                "missing_body_frame_join",
                "nullable_foreign_key_with_explicit_invalid_source_row_evidence",
            ),
        ),
    )

    for table_name in (
        "position_providers",
        "chaser_occurrences",
        "stimulus_native_state_support",
    ):
        retained[table_name] = replace(
            retained[table_name],
            required_capability=CORE_BOUND_CHASER_RELATIVE_PAIR_CAPABILITY,
        )
    return MappingProxyType(retained)


CORE_CHASER_EXTENSION_TABLE_SPECS = _extension_specs()
CORE_CHASER_TABLE_SPECS = compose_disjoint_table_specs(
    ("complete_core_behavior", CORE_BEHAVIOR_TABLE_SPECS),
    ("chaser_paradigm_extension", CORE_CHASER_EXTENSION_TABLE_SPECS),
)
CORE_CHASER_TABLE_NAMES = tuple(sorted(CORE_CHASER_TABLE_SPECS))


__all__ = [
    "CHASER_EXTENSION_OMITTED_TABLES",
    "CORE_CHASER_EXPORT_PROFILE_ID",
    "CORE_CHASER_EXTENSION_TABLE_SPECS",
    "CORE_CHASER_TABLE_NAMES",
    "CORE_CHASER_TABLE_SPECS",
    "CORE_BOUND_CHASER_RELATIVE_PAIR_CAPABILITY",
]
