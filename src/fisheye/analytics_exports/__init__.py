"""Canonical contracts for immutable cross-recording analytics exports."""

from .arrow_contracts import (
    ARROW_CONTRACT_ENVELOPE_SCHEMA_ID,
    ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION,
    ARROW_TABLE_CONTRACTS,
    EXACT_ARROW_SCHEMA_TABLES,
    arrow_contract_envelope,
    validate_arrow_contract_envelope,
    validate_arrow_schema,
)
from .capabilities import CapabilityStatus, resolve_capabilities
from .contracts import (
    ALL_TABLES,
    BASELINE_TABLES,
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
from .validation import ExportValidationError, validate_export_run, validate_export_runs

__all__ = [
    "ARROW_CONTRACT_ENVELOPE_SCHEMA_ID",
    "ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION",
    "ARROW_TABLE_CONTRACTS",
    "ALL_TABLES",
    "BASELINE_TABLES",
    "CHASER_TABLES",
    "DESCRIPTIVE_TABLE",
    "EXPORT_SCHEMA_ID",
    "EXPORT_SCHEMA_VERSION",
    "STATISTICS_TABLE",
    "TABLE_CONTRACTS",
    "CapabilityStatus",
    "ExportValidationError",
    "EXACT_ARROW_SCHEMA_TABLES",
    "TableContract",
    "canonicalize_export_row",
    "arrow_contract_envelope",
    "contract_snapshot",
    "resolve_capabilities",
    "validate_export_run",
    "validate_export_runs",
    "validate_arrow_contract_envelope",
    "validate_arrow_schema",
]
