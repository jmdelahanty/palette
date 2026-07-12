"""Canonical contracts for immutable cross-recording analytics exports."""

from .capabilities import CapabilityStatus, resolve_capabilities
from .contracts import (
    ALL_TABLES,
    CHASER_TABLES,
    DESCRIPTIVE_TABLE,
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
    STATISTICS_TABLE,
    TABLE_CONTRACTS,
    TableContract,
    canonicalize_export_row,
    contract_snapshot,
)

__all__ = [
    "ALL_TABLES",
    "CHASER_TABLES",
    "DESCRIPTIVE_TABLE",
    "EXPORT_SCHEMA_ID",
    "EXPORT_SCHEMA_VERSION",
    "STATISTICS_TABLE",
    "TABLE_CONTRACTS",
    "CapabilityStatus",
    "TableContract",
    "canonicalize_export_row",
    "contract_snapshot",
    "resolve_capabilities",
]
