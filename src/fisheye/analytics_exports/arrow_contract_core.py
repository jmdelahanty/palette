"""Reusable exact Arrow declaration, envelope, and footer machinery.

Scientific table vocabularies remain owned by their family contract modules.
This module owns the representation mechanics so canonical, baseline-strategy,
and training-response publications cannot drift into different hashing or
physical-schema rules.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence


ARROW_TABLE_SCHEMA_VERSION = 1
_ENVELOPE_FIELDS = {
    "schema_id",
    "schema_version",
    "exact_tables",
    "inferred_v2_compatibility_tables",
    "payload_sha256",
}
_TABLE_FIELDS = {
    "schema_id",
    "schema_version",
    "table_name",
    "fields",
    "payload_sha256",
}
_KEYED_TABLE_FIELDS = _TABLE_FIELDS | {"primary_key"}
_FIELD_FIELDS = {"name", "arrow_type", "nullable"}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def payload_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class ArrowFieldContract:
    """One exact Arrow field declaration."""

    name: str
    arrow_type: str
    nullable: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "arrow_type": self.arrow_type,
            "nullable": self.nullable,
        }


@dataclass(frozen=True)
class ArrowTableContract:
    """Closed ordered Arrow schema for one maintained export table."""

    table_name: str
    fields: tuple[ArrowFieldContract, ...]
    schema_version: int = ARROW_TABLE_SCHEMA_VERSION
    schema_namespace: str = "palette.analytics_export.arrow_table"
    primary_key: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(field.name for field in self.fields)
        if len(set(names)) != len(names):
            raise ValueError(f"{self.table_name}: Arrow field names must be unique")
        if len(set(self.primary_key)) != len(self.primary_key):
            raise ValueError(f"{self.table_name}: primary-key fields must be unique")
        missing = tuple(name for name in self.primary_key if name not in names)
        if missing:
            raise ValueError(
                f"{self.table_name}: primary-key fields are absent from schema: {missing}"
            )

    @property
    def schema_id(self) -> str:
        return f"{self.schema_namespace}.{self.table_name}"

    def payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "table_name": self.table_name,
            "fields": [field.to_dict() for field in self.fields],
        }
        if self.primary_key:
            payload["primary_key"] = list(self.primary_key)
        return payload

    @property
    def payload_sha256(self) -> str:
        return payload_sha256(self.payload())

    def to_dict(self) -> dict[str, object]:
        return {**self.payload(), "payload_sha256": self.payload_sha256}


def field(
    name: str,
    arrow_type: str,
    *,
    nullable: bool = False,
) -> ArrowFieldContract:
    return ArrowFieldContract(name=name, arrow_type=arrow_type, nullable=nullable)


def _arrow_type(type_id: str) -> Any:
    import pyarrow as pa

    types = {
        "bool": pa.bool_(),
        "float32": pa.float32(),
        "float64": pa.float64(),
        "int8": pa.int8(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "list<string>": pa.list_(pa.string()),
        "string": pa.string(),
        "uint16": pa.uint16(),
        "uint64": pa.uint64(),
    }
    try:
        return types[type_id]
    except KeyError as exc:  # pragma: no cover - installed declarations are tested.
        raise ValueError(f"Unsupported Arrow contract type: {type_id}") from exc


def exact_schema(
    contract: ArrowTableContract,
    *,
    metadata: Mapping[bytes, bytes],
) -> Any:
    """Return one exact PyArrow schema with digest-bound footer metadata."""

    import pyarrow as pa

    fields = [
        pa.field(item.name, _arrow_type(item.arrow_type), nullable=item.nullable)
        for item in contract.fields
    ]
    contract_metadata = {
        **metadata,
        b"palette.arrow_schema_mode": b"exact",
        b"palette.arrow_schema_id": contract.schema_id.encode("utf-8"),
        b"palette.arrow_schema_version": str(contract.schema_version).encode("ascii"),
        b"palette.arrow_schema_sha256": contract.payload_sha256.encode("ascii"),
    }
    return pa.schema(fields, metadata=contract_metadata)


def validate_exact_schema(contract: ArrowTableContract, schema: Any) -> None:
    """Compare complete fields and contract footer metadata."""

    metadata = schema.metadata or {}
    expected = exact_schema(contract, metadata={})
    if schema.remove_metadata() != expected.remove_metadata():
        raise ValueError(
            f"{contract.table_name}: physical Arrow fields differ from the exact contract"
        )
    expected_metadata = expected.metadata or {}
    for key in (
        b"palette.arrow_schema_mode",
        b"palette.arrow_schema_id",
        b"palette.arrow_schema_version",
        b"palette.arrow_schema_sha256",
    ):
        if metadata.get(key) != expected_metadata[key]:
            raise ValueError(
                f"{contract.table_name}: Arrow footer contract metadata is invalid"
            )


def contract_envelope(
    table_names: Sequence[str],
    *,
    known_table_names: Sequence[str],
    contracts: Mapping[str, ArrowTableContract],
    schema_id: str,
    schema_version: int,
) -> dict[str, object]:
    """Build a closed exact/inferred partition for one contract suite."""

    names = tuple(table_names)
    if len(set(names)) != len(names):
        raise ValueError("Arrow contract table names must be unique")
    unknown = sorted(set(names) - set(known_table_names))
    if unknown:
        raise ValueError(f"Unknown analytics tables in Arrow contract: {unknown}")
    exact = {
        name: contracts[name].to_dict()
        for name in names
        if name in contracts
    }
    inferred = sorted(set(names) - set(exact))
    payload: dict[str, object] = {
        "schema_id": schema_id,
        "schema_version": schema_version,
        "exact_tables": exact,
        "inferred_v2_compatibility_tables": inferred,
    }
    return {**payload, "payload_sha256": payload_sha256(payload)}


def validate_contract_envelope(
    value: object,
    table_names: Sequence[str],
    *,
    known_table_names: Sequence[str],
    contracts: Mapping[str, ArrowTableContract],
    schema_id: str,
    schema_version: int,
) -> dict[str, object]:
    """Validate a suite envelope against installed declarations, not its hash."""

    if not isinstance(value, Mapping) or set(value) != _ENVELOPE_FIELDS:
        raise ValueError("Arrow contract envelope has an unexpected field set")
    if value.get("schema_id") != schema_id:
        raise ValueError("Arrow contract envelope schema ID is invalid")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != schema_version
    ):
        raise ValueError("Arrow contract envelope schema version is invalid")
    payload = {key: value[key] for key in _ENVELOPE_FIELDS - {"payload_sha256"}}
    if value.get("payload_sha256") != payload_sha256(payload):
        raise ValueError("Arrow contract envelope payload digest is invalid")
    expected = contract_envelope(
        table_names,
        known_table_names=known_table_names,
        contracts=contracts,
        schema_id=schema_id,
        schema_version=schema_version,
    )
    if dict(value) != expected:
        raise ValueError("Arrow contract envelope differs from installed contracts")
    exact = value.get("exact_tables")
    assert isinstance(exact, Mapping)
    for table_name, raw_contract in exact.items():
        if not isinstance(raw_contract, Mapping) or set(raw_contract) not in (
            _TABLE_FIELDS,
            _KEYED_TABLE_FIELDS,
        ):
            raise ValueError(f"{table_name}: Arrow table contract field set is invalid")
        raw_fields = raw_contract.get("fields")
        if not isinstance(raw_fields, list) or any(
            not isinstance(item, Mapping) or set(item) != _FIELD_FIELDS
            for item in raw_fields
        ):
            raise ValueError(f"{table_name}: Arrow field declarations are invalid")
    return expected


__all__ = [
    "ARROW_TABLE_SCHEMA_VERSION",
    "ArrowFieldContract",
    "ArrowTableContract",
    "canonical_bytes",
    "contract_envelope",
    "exact_schema",
    "field",
    "payload_sha256",
    "validate_contract_envelope",
    "validate_exact_schema",
]
