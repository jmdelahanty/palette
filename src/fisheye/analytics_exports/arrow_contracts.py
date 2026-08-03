"""Exact physical Arrow contracts for immutable analytics exports.

The logical V2 table contracts intentionally began as minimum-column
contracts.  This module versions the independent physical Arrow layer.  A
table is either governed by one installed exact schema or is named explicitly
as an inferred-V2 compatibility table in the closed manifest envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from .contracts import (
    POSITION_OCCUPANCY_HISTOGRAM_TABLE,
    RECORDING_SUMMARY_TABLE,
    TABLE_CONTRACTS,
)


ARROW_CONTRACT_ENVELOPE_SCHEMA_ID = "palette.analytics_export.arrow_contracts"
ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION = 1
ARROW_TABLE_SCHEMA_VERSION = 1
EXACT_ARROW_SCHEMA_TABLES = (
    POSITION_OCCUPANCY_HISTOGRAM_TABLE,
    RECORDING_SUMMARY_TABLE,
)

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
_FIELD_FIELDS = {"name", "arrow_type", "nullable"}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


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

    @property
    def schema_id(self) -> str:
        return f"palette.analytics_export.arrow_table.{self.table_name}"

    def payload(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "table_name": self.table_name,
            "fields": [field.to_dict() for field in self.fields],
        }

    @property
    def payload_sha256(self) -> str:
        return _sha256(self.payload())

    def to_dict(self) -> dict[str, object]:
        return {**self.payload(), "payload_sha256": self.payload_sha256}


def _field(name: str, arrow_type: str, *, nullable: bool = False) -> ArrowFieldContract:
    return ArrowFieldContract(name=name, arrow_type=arrow_type, nullable=nullable)


# Detection-occupancy is the first exact table because its writer has one
# closed row shape, its physical units are already frozen, and it exercises
# strings, booleans, integers, floats, nullable lineage, and a list field.
_POSITION_OCCUPANCY_FIELDS = (
    _field("export_schema_version", "int32"),
    _field("table_name", "string"),
    _field("recording_id", "string"),
    _field("zarr_path", "string"),
    _field("source_lineage_hash", "string"),
    _field("position_occupancy_run", "string"),
    _field("position_occupancy_path", "string"),
    _field("position_occupancy_schema_id", "string", nullable=True),
    _field("position_occupancy_schema_version", "int64", nullable=True),
    _field("source_detection_path", "string", nullable=True),
    _field("source_detection_kind", "string", nullable=True),
    _field("source_segment_kind", "string", nullable=True),
    _field("source_segment_path", "string", nullable=True),
    _field("source_refs_json", "string"),
    _field("window_index", "int64"),
    _field("window_id", "int64", nullable=True),
    _field("window_label", "string", nullable=True),
    _field("start_frame", "int64", nullable=True),
    _field("end_frame", "int64", nullable=True),
    _field("start_time_s", "float64", nullable=True),
    _field("end_time_s", "float64", nullable=True),
    _field("duration_s", "float64", nullable=True),
    _field("coordinate_frame", "string"),
    _field("source_coordinate_frame", "string"),
    _field("coordinate_origin", "string"),
    _field("x_axis_direction", "string"),
    _field("y_axis_direction", "string"),
    _field("image_width_px", "float64"),
    _field("image_height_px", "float64"),
    _field("normalized_grid_id", "string"),
    _field("normalized_grid_uniform", "bool"),
    _field("sparse_zero_bins_omitted", "bool"),
    _field("x_bin_count", "int64"),
    _field("y_bin_count", "int64"),
    _field("x_bin_index", "int64"),
    _field("x_bin_left_px", "float64"),
    _field("x_bin_right_px", "float64"),
    _field("x_bin_center_px", "float64"),
    _field("x_bin_width_px", "float64"),
    _field("x_bin_left_fraction", "float64"),
    _field("x_bin_right_fraction", "float64"),
    _field("x_bin_center_fraction", "float64"),
    _field("x_bin_width_fraction", "float64"),
    _field("y_bin_index", "int64"),
    _field("y_bin_left_px", "float64"),
    _field("y_bin_right_px", "float64"),
    _field("y_bin_center_px", "float64"),
    _field("y_bin_width_px", "float64"),
    _field("y_bin_left_fraction", "float64"),
    _field("y_bin_right_fraction", "float64"),
    _field("y_bin_center_fraction", "float64"),
    _field("y_bin_width_fraction", "float64"),
    _field("hist_count", "int64"),
    _field("window_detection_count", "int64", nullable=True),
    _field("covered_frame_count", "int64", nullable=True),
    _field("total_span_frames", "int64", nullable=True),
    _field("coverage_pct", "float64", nullable=True),
    _field("axis_order", "list<string>"),
    _field("source_bin_size_px", "float64", nullable=True),
    _field("collection_id", "string", nullable=True),
    _field("collection_manifest_sha256", "string", nullable=True),
    _field("collection_manifest_path", "string", nullable=True),
)


# Recording-summary has one closed producer row shape even when its source
# capabilities are absent.  The five shared identity fields and the always
# computed stimulus-step count are required.  Every other field depends on an
# optional stimulus, response, swim-bout, or collection source.  The derived
# protocol hash is a deprecated alias retained in physical v1 so this exact
# contract does not silently change the producer's current row vocabulary.
_RECORDING_SUMMARY_FIELDS = (
    _field("export_schema_version", "int32"),
    _field("table_name", "string"),
    _field("recording_id", "string"),
    _field("zarr_path", "string"),
    _field("source_lineage_hash", "string"),
    _field("stimulus_run", "string", nullable=True),
    _field("stimulus_response_run", "string", nullable=True),
    _field("swim_bout_run", "string", nullable=True),
    _field("stimulus_step_count", "int64"),
    _field("protocol_signature_schema", "string", nullable=True),
    _field("protocol_signature_hash", "string", nullable=True),
    _field("derived_protocol_hash", "string", nullable=True),
    _field("protocol_mode_sequence", "string", nullable=True),
    _field("protocol_duration_sequence_s", "string", nullable=True),
    _field("protocol_step_count", "int64", nullable=True),
    _field("source_track_kinematics_run", "string", nullable=True),
    _field("source_track_kinematics_type", "string", nullable=True),
    _field("source_bout_run", "string", nullable=True),
    _field("n_fish", "int64", nullable=True),
    _field("n_steps", "int64", nullable=True),
    _field("global_fish_count", "int64", nullable=True),
    _field("total_distance_mm_sum", "float64", nullable=True),
    _field("mean_speed_mm_s_mean", "float64", nullable=True),
    _field("fraction_moving_mean", "float64", nullable=True),
    _field("total_active_s_sum", "float64", nullable=True),
    _field("swim_bout_default_level", "string", nullable=True),
    _field("swim_bout_default_n_bouts", "int64", nullable=True),
    _field("swim_bout_default_mean_duration_s", "float64", nullable=True),
    _field("swim_bout_default_total_path_length_mm", "float64", nullable=True),
    _field("collection_id", "string", nullable=True),
    _field("collection_manifest_sha256", "string", nullable=True),
    _field("collection_manifest_path", "string", nullable=True),
)


ARROW_TABLE_CONTRACTS: dict[str, ArrowTableContract] = {
    POSITION_OCCUPANCY_HISTOGRAM_TABLE: ArrowTableContract(
        table_name=POSITION_OCCUPANCY_HISTOGRAM_TABLE,
        fields=_POSITION_OCCUPANCY_FIELDS,
    ),
    RECORDING_SUMMARY_TABLE: ArrowTableContract(
        table_name=RECORDING_SUMMARY_TABLE,
        fields=_RECORDING_SUMMARY_FIELDS,
    ),
}


def arrow_contract_envelope(table_names: Sequence[str]) -> dict[str, object]:
    """Build the closed exact/inferred partition for one export manifest."""

    names = tuple(table_names)
    if len(set(names)) != len(names):
        raise ValueError("Arrow contract table names must be unique")
    unknown = sorted(set(names) - set(TABLE_CONTRACTS))
    if unknown:
        raise ValueError(f"Unknown analytics tables in Arrow contract: {unknown}")
    exact = {
        name: ARROW_TABLE_CONTRACTS[name].to_dict()
        for name in names
        if name in ARROW_TABLE_CONTRACTS
    }
    inferred = sorted(set(names) - set(exact))
    payload: dict[str, object] = {
        "schema_id": ARROW_CONTRACT_ENVELOPE_SCHEMA_ID,
        "schema_version": ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION,
        "exact_tables": exact,
        "inferred_v2_compatibility_tables": inferred,
    }
    return {**payload, "payload_sha256": _sha256(payload)}


def validate_arrow_contract_envelope(
    value: object,
    table_names: Sequence[str],
) -> dict[str, object]:
    """Validate an envelope against installed contracts, not merely its hash."""

    if not isinstance(value, Mapping) or set(value) != _ENVELOPE_FIELDS:
        raise ValueError("Arrow contract envelope has an unexpected field set")
    if value.get("schema_id") != ARROW_CONTRACT_ENVELOPE_SCHEMA_ID:
        raise ValueError("Arrow contract envelope schema ID is invalid")
    if type(value.get("schema_version")) is not int or value.get(
        "schema_version"
    ) != ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION:
        raise ValueError("Arrow contract envelope schema version is invalid")
    payload = {key: value[key] for key in _ENVELOPE_FIELDS - {"payload_sha256"}}
    if value.get("payload_sha256") != _sha256(payload):
        raise ValueError("Arrow contract envelope payload digest is invalid")
    expected = arrow_contract_envelope(table_names)
    if dict(value) != expected:
        raise ValueError("Arrow contract envelope differs from installed contracts")
    exact = value.get("exact_tables")
    assert isinstance(exact, Mapping)
    for table_name, raw_contract in exact.items():
        if not isinstance(raw_contract, Mapping) or set(raw_contract) != _TABLE_FIELDS:
            raise ValueError(f"{table_name}: Arrow table contract field set is invalid")
        raw_fields = raw_contract.get("fields")
        if not isinstance(raw_fields, list) or any(
            not isinstance(item, Mapping) or set(item) != _FIELD_FIELDS
            for item in raw_fields
        ):
            raise ValueError(f"{table_name}: Arrow field declarations are invalid")
    return expected


def _arrow_type(type_id: str) -> Any:
    import pyarrow as pa

    types = {
        "bool": pa.bool_(),
        "float64": pa.float64(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "list<string>": pa.list_(pa.string()),
        "string": pa.string(),
    }
    try:
        return types[type_id]
    except KeyError as exc:  # pragma: no cover - installed declarations are tested.
        raise ValueError(f"Unsupported Arrow contract type: {type_id}") from exc


def exact_arrow_schema(table_name: str, *, metadata: Mapping[bytes, bytes]) -> Any:
    """Return the installed exact PyArrow schema with contract metadata."""

    import pyarrow as pa

    contract = ARROW_TABLE_CONTRACTS[table_name]
    fields = [
        pa.field(field.name, _arrow_type(field.arrow_type), nullable=field.nullable)
        for field in contract.fields
    ]
    contract_metadata = {
        **metadata,
        b"palette.arrow_schema_mode": b"exact",
        b"palette.arrow_schema_id": contract.schema_id.encode("utf-8"),
        b"palette.arrow_schema_version": str(contract.schema_version).encode("ascii"),
        b"palette.arrow_schema_sha256": contract.payload_sha256.encode("ascii"),
    }
    return pa.schema(fields, metadata=contract_metadata)


def validate_arrow_schema(table_name: str, schema: Any) -> None:
    """Validate one physical schema against its exact or compatibility mode."""

    metadata = schema.metadata or {}
    contract = ARROW_TABLE_CONTRACTS.get(table_name)
    if contract is None:
        if metadata.get(b"palette.arrow_schema_mode") != b"inferred_v2_compatibility":
            raise ValueError(f"{table_name}: Arrow schema compatibility mode is missing")
        return
    expected = exact_arrow_schema(table_name, metadata={})
    if schema.remove_metadata() != expected.remove_metadata():
        raise ValueError(f"{table_name}: physical Arrow fields differ from the exact contract")
    expected_metadata = expected.metadata or {}
    for key in (
        b"palette.arrow_schema_mode",
        b"palette.arrow_schema_id",
        b"palette.arrow_schema_version",
        b"palette.arrow_schema_sha256",
    ):
        if metadata.get(key) != expected_metadata[key]:
            raise ValueError(f"{table_name}: Arrow footer contract metadata is invalid")


__all__ = [
    "ARROW_CONTRACT_ENVELOPE_SCHEMA_ID",
    "ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION",
    "ARROW_TABLE_CONTRACTS",
    "EXACT_ARROW_SCHEMA_TABLES",
    "ArrowFieldContract",
    "ArrowTableContract",
    "arrow_contract_envelope",
    "exact_arrow_schema",
    "validate_arrow_contract_envelope",
    "validate_arrow_schema",
]
