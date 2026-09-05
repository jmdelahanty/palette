"""Exact, protocol-neutral contracts for validated-behavior cohort exports.

The publication engine owns only table mechanics and closed provenance.  A
behavior family may extend this suite with separately versioned table adapters;
it must not teach this module protocol formulas or silently reinterpret an old
table grain.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from .arrow_contract_core import (
    ArrowTableContract,
    contract_envelope,
    field,
)


ARROW_ENVELOPE_SCHEMA_ID = (
    "palette.analytics.validated_behavior.arrow_contracts"
)
ARROW_ENVELOPE_SCHEMA_VERSION = 1
TABLE_SCHEMA_NAMESPACE = "palette.analytics.validated_behavior.table"
CORE_METADATA_PROFILE_ID = "validated_behavior_core_metadata_v1"

CAPABILITY_POLICIES = frozenset(
    {
        "all_parent_metadata",
        "required_all_admitted",
        "optional_explicit_coverage",
        "capability_stratified_subset",
    }
)


@dataclass(frozen=True)
class ValidatedBehaviorTableSpec:
    """Declarative table semantics sealed by an export plan and manifest."""

    contract: ArrowTableContract
    grain: str
    capability_policy: str
    required_capability: str | None = None
    foreign_keys: tuple[tuple[tuple[str, ...], str, tuple[str, ...]], ...] = ()
    zero_rows_allowed: bool = False
    primary_key_validation: str = "unordered_unique_v1"
    semantic_metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.capability_policy not in CAPABILITY_POLICIES:
            raise ValueError(
                f"{self.contract.table_name}: unsupported capability policy"
            )
        if (
            self.capability_policy == "all_parent_metadata"
        ) != (self.required_capability is None):
            raise ValueError(
                f"{self.contract.table_name}: metadata and capability bindings disagree"
            )
        if self.required_capability is not None and not self.zero_rows_allowed:
            raise ValueError(
                f"{self.contract.table_name}: capability-scoped tables must permit "
                "explicit non-contributor or complete-no-event empty parts"
            )
        if self.primary_key_validation not in {
            "unordered_unique_v1",
            "strictly_increasing_v1",
        }:
            raise ValueError(
                f"{self.contract.table_name}: unsupported primary-key validation"
            )
        semantic_keys = tuple(key for key, _value in self.semantic_metadata)
        if (
            len(set(semantic_keys)) != len(semantic_keys)
            or any(
                type(key) is not str
                or not key
                or key != key.strip()
                or type(value) is not str
                or not value
                or value != value.strip()
                for key, value in self.semantic_metadata
            )
        ):
            raise ValueError(
                f"{self.contract.table_name}: semantic metadata is invalid"
            )
        names = {item.name for item in self.contract.fields}
        for local_fields, _target_table, _target_fields in self.foreign_keys:
            if not set(local_fields).issubset(names):
                raise ValueError(
                    f"{self.contract.table_name}: foreign-key fields are absent"
                )

    @property
    def table_name(self) -> str:
        return self.contract.table_name

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "table_name": self.table_name,
            "table_contract": self.contract.to_dict(),
            "grain": self.grain,
            "primary_key": list(self.contract.primary_key),
            "foreign_keys": [
                {
                    "local_fields": list(local),
                    "target_table": target,
                    "target_fields": list(target_fields),
                }
                for local, target, target_fields in self.foreign_keys
            ],
            "capability_policy": self.capability_policy,
            "required_capability": self.required_capability,
            "zero_rows_allowed": self.zero_rows_allowed,
        }
        # Preserve the established compact-profile records byte-for-byte.
        # Dense profiles opt in explicitly so key uniqueness is provable with
        # constant memory rather than a recording-sized Python set.
        if self.primary_key_validation != "unordered_unique_v1":
            result["primary_key_validation"] = self.primary_key_validation
        if self.semantic_metadata:
            result["semantic_metadata"] = {
                key: value for key, value in self.semantic_metadata
            }
        return result


def _contract(
    table_name: str,
    fields: Sequence[object],
    *,
    primary_key: Sequence[str],
) -> ArrowTableContract:
    return ArrowTableContract(
        table_name=table_name,
        fields=tuple(fields),
        primary_key=tuple(primary_key),
        schema_namespace=TABLE_SCHEMA_NAMESPACE,
    )


COHORT_RECORDINGS = _contract(
    "cohort_recordings",
    (
        field("export_run_id", "string"),
        field("member_ordinal", "int32"),
        field("membership_member_sha256", "string"),
        field("source_ordinal", "int32"),
        field("source_member_sha256", "string"),
        field("dataset_id", "string"),
        field("recording_id", "string"),
        field("analysis_zarr", "string"),
        field("protocol_names", "list<string>"),
        field("protocol_hashes", "list<string>"),
        field("source_subject_ids", "list<string>"),
        field("source_subject_identity_status", "string"),
        field("acquisition_batch_id", "string", nullable=True),
        field("acquisition_batch_identity_status", "string"),
        field("analysis_unit_kind", "string"),
        field("analysis_unit_id", "string"),
        field("membership_state", "string"),
        field("reason_code", "string", nullable=True),
    ),
    primary_key=("export_run_id", "recording_id"),
)

RECORDING_BUNDLES = _contract(
    "recording_bundles",
    (
        field("export_run_id", "string"),
        field("member_ordinal", "int32"),
        field("membership_member_sha256", "string"),
        field("bundle_set_member_sha256", "string"),
        field("recording_id", "string"),
        field("analysis_zarr", "string"),
        field("bundle_state", "string"),
        field("reason_code", "string", nullable=True),
        field("bundle_adapter_id", "string", nullable=True),
        field("bundle_path", "string", nullable=True),
        field("bundle_file_sha256", "string", nullable=True),
        field("bundle_record_sha256", "string", nullable=True),
        field("bundle_schema_id", "string", nullable=True),
        field("bundle_schema_version", "int32", nullable=True),
        field("bundle_method_id", "string", nullable=True),
        field("bundle_status", "string", nullable=True),
        field("bundle_binding_inventory_sha256", "string", nullable=True),
        field("capabilities_sha256", "string"),
    ),
    primary_key=("export_run_id", "recording_id"),
)

RECORDING_CAPABILITIES = _contract(
    "recording_capabilities",
    (
        field("export_run_id", "string"),
        field("member_ordinal", "int32"),
        field("membership_member_sha256", "string"),
        field("bundle_set_member_sha256", "string"),
        field("recording_id", "string"),
        field("capability_id", "string"),
        field("state", "string"),
        field("reason_code", "string", nullable=True),
        field("detail", "string", nullable=True),
        field("binding_json", "string", nullable=True),
        field("binding_sha256", "string", nullable=True),
        field("capabilities_sha256", "string"),
    ),
    primary_key=("export_run_id", "recording_id", "capability_id"),
)


CORE_TABLE_SPECS: Mapping[str, ValidatedBehaviorTableSpec] = MappingProxyType(
    {
        "cohort_recordings": ValidatedBehaviorTableSpec(
            contract=COHORT_RECORDINGS,
            grain="one row per closed parent-cohort recording",
            capability_policy="all_parent_metadata",
        ),
        "recording_bundles": ValidatedBehaviorTableSpec(
            contract=RECORDING_BUNDLES,
            grain="one row per parent recording and exact bundle state",
            capability_policy="all_parent_metadata",
            foreign_keys=(
                (
                    ("export_run_id", "recording_id"),
                    "cohort_recordings",
                    ("export_run_id", "recording_id"),
                ),
            ),
        ),
        "recording_capabilities": ValidatedBehaviorTableSpec(
            contract=RECORDING_CAPABILITIES,
            grain="one row per parent recording and profile-declared capability",
            capability_policy="all_parent_metadata",
            foreign_keys=(
                (
                    ("export_run_id", "recording_id"),
                    "cohort_recordings",
                    ("export_run_id", "recording_id"),
                ),
                (
                    ("export_run_id", "recording_id"),
                    "recording_bundles",
                    ("export_run_id", "recording_id"),
                ),
            ),
        ),
    }
)
CORE_TABLE_NAMES = tuple(CORE_TABLE_SPECS)
CORE_CONTRACTS: Mapping[str, ArrowTableContract] = MappingProxyType(
    {name: spec.contract for name, spec in CORE_TABLE_SPECS.items()}
)


def compose_disjoint_table_specs(
    *components: tuple[str, Mapping[str, ValidatedBehaviorTableSpec]],
) -> Mapping[str, ValidatedBehaviorTableSpec]:
    """Compose named table suites while rejecting collisions before insertion.

    A dictionary-unpacking merge cannot detect an overwritten table after the
    fact.  Composite behavior profiles use this helper so two components can
    never silently claim the same table name or scientific authority surface.
    """

    if not components:
        raise ValueError("Table-spec composition requires at least one component")
    combined: dict[str, ValidatedBehaviorTableSpec] = {}
    owners: dict[str, str] = {}
    for component_name, specs in components:
        if (
            type(component_name) is not str
            or not component_name
            or component_name != component_name.strip()
        ):
            raise ValueError("Table-spec component names must be normalized text")
        if not specs:
            raise ValueError(f"Table-spec component {component_name!r} is empty")
        collisions = sorted(set(combined).intersection(specs))
        if collisions:
            conflict_owners = {
                table_name: owners[table_name] for table_name in collisions
            }
            raise ValueError(
                f"Table-spec component {component_name!r} collides before composition: "
                f"tables={collisions!r}, existing_owners={conflict_owners!r}"
            )
        for table_name, spec in specs.items():
            combined[table_name] = spec
            owners[table_name] = component_name
    validate_table_specs(combined)
    return MappingProxyType(combined)


def validate_table_specs(
    specs: Mapping[str, ValidatedBehaviorTableSpec],
) -> tuple[str, ...]:
    """Return deterministic names after proving a closed, self-consistent suite."""

    if not specs:
        raise ValueError("Validated-behavior export requires at least one table")
    names = tuple(sorted(specs))
    if any(name != specs[name].table_name for name in names):
        raise ValueError("Validated-behavior table map keys and contracts differ")
    for name in names:
        contract = specs[name].contract
        contract_fields = {item.name for item in contract.fields}
        if not {"export_run_id", "recording_id"}.issubset(contract_fields):
            raise ValueError(
                f"{name}: recording-sharded tables require export_run_id and recording_id"
            )
        if contract.primary_key[:2] != ("export_run_id", "recording_id"):
            raise ValueError(
                f"{name}: primary key must begin with export_run_id, recording_id"
            )
        for local, target, target_fields in specs[name].foreign_keys:
            if target not in specs:
                raise ValueError(f"{name}: foreign-key target {target!r} is absent")
            target_names = {item.name for item in specs[target].contract.fields}
            if len(local) != len(target_fields) or not set(target_fields).issubset(
                target_names
            ):
                raise ValueError(f"{name}: foreign-key declaration is invalid")
            recording_prefix = ("export_run_id", "recording_id")
            if (
                tuple(local[:2]) != recording_prefix
                or tuple(target_fields[:2]) != recording_prefix
            ):
                raise ValueError(
                    f"{name}: foreign keys must be recording-scoped"
                )
    return names


def table_contract_envelope(
    specs: Mapping[str, ValidatedBehaviorTableSpec],
) -> dict[str, object]:
    names = validate_table_specs(specs)
    contracts = {name: specs[name].contract for name in names}
    return contract_envelope(
        names,
        known_table_names=names,
        contracts=contracts,
        schema_id=ARROW_ENVELOPE_SCHEMA_ID,
        schema_version=ARROW_ENVELOPE_SCHEMA_VERSION,
    )


__all__ = [
    "ARROW_ENVELOPE_SCHEMA_ID",
    "ARROW_ENVELOPE_SCHEMA_VERSION",
    "CAPABILITY_POLICIES",
    "CORE_CONTRACTS",
    "CORE_METADATA_PROFILE_ID",
    "CORE_TABLE_NAMES",
    "CORE_TABLE_SPECS",
    "ValidatedBehaviorTableSpec",
    "compose_disjoint_table_specs",
    "table_contract_envelope",
    "validate_table_specs",
]
