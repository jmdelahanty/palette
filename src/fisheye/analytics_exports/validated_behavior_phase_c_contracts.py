"""Protocol-authored chaser appearance contracts for behavior exports.

Phase C preserves every Phase-B table and grain while replacing only the
``chaser_occurrences`` dimension with schema version 2.  The successor adds a
receipt-bound projection of the exact protocol RGBA values and keeps behavior
role as an independent semantic and glyph channel.  Phase-A and Phase-B
contracts remain unchanged so their immutable publications stay readable.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Mapping

from .arrow_contract_core import ArrowTableContract, field
from .validated_behavior_contracts import (
    TABLE_SCHEMA_NAMESPACE,
    ValidatedBehaviorTableSpec,
)
from .validated_behavior_phase_a_contracts import (
    CHASER_OCCURRENCES as PHASE_A_CHASER_OCCURRENCES,
)
from .validated_behavior_phase_b_contracts import PHASE_B_TABLE_SPECS

PHASE_C_PROFILE_ID = "validated_recording_behavior_phase_c_v1"


CHASER_OCCURRENCES_V2 = ArrowTableContract(
    table_name="chaser_occurrences",
    fields=PHASE_A_CHASER_OCCURRENCES.fields
    + (
        field("behavior_role_code", "int32"),
        field("experimental_color_r", "float64"),
        field("experimental_color_g", "float64"),
        field("experimental_color_b", "float64"),
        field("experimental_color_a", "float64"),
        field("experimental_color_hex", "string"),
        field("experimental_color_css", "string"),
        field("contrast_outline_hex", "string"),
        field("plotly_role_symbol", "string"),
        field("matplotlib_role_marker", "string"),
        field("appearance_schema_id", "string"),
        field("appearance_schema_version", "int32"),
        field("appearance_policy_id", "string"),
        field("appearance_projection_sha256", "string"),
        field("occurrence_binding_sha256", "string"),
        field("color_semantics", "string"),
        field("role_semantics", "string"),
        field("color_role_independence", "bool"),
    ),
    schema_version=2,
    schema_namespace=TABLE_SCHEMA_NAMESPACE,
    primary_key=PHASE_A_CHASER_OCCURRENCES.primary_key,
)


_BASE_CHASER_SPEC = PHASE_B_TABLE_SPECS["chaser_occurrences"]
CHASER_OCCURRENCES_V2_SPEC = replace(
    _BASE_CHASER_SPEC,
    contract=CHASER_OCCURRENCES_V2,
    semantic_metadata=(
        (
            "appearance_source",
            "exact_protocol_rgba_bound_through_relative_frame_receipt",
        ),
        ("color_semantics", "experimental_protocol_rgba"),
        ("role_semantics", "independent_marker_shape_and_text"),
        ("color_role_independence", "true"),
        ("appearance_fallback", "prohibited"),
        (
            "appearance_join",
            "recording_id_and_chaser_identity_code",
        ),
    ),
)


PHASE_C_TABLE_SPECS: Mapping[str, ValidatedBehaviorTableSpec] = MappingProxyType(
    {
        **PHASE_B_TABLE_SPECS,
        "chaser_occurrences": CHASER_OCCURRENCES_V2_SPEC,
    }
)
PHASE_C_TABLE_NAMES = tuple(PHASE_C_TABLE_SPECS)


__all__ = [
    "CHASER_OCCURRENCES_V2",
    "CHASER_OCCURRENCES_V2_SPEC",
    "PHASE_C_PROFILE_ID",
    "PHASE_C_TABLE_NAMES",
    "PHASE_C_TABLE_SPECS",
]
