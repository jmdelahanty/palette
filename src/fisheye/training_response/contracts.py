"""Contracts for whole-training chaser response classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Mapping, Sequence

from fisheye.analytics_exports.arrow_contract_core import (
    ArrowTableContract,
    canonical_bytes,
    contract_envelope,
    exact_schema,
    field,
    validate_contract_envelope,
    validate_exact_schema,
)


SCHEMA_ID = "palette.training_response_analytics"
LEGACY_SCHEMA_VERSION = 1
LEGACY_EXACT_SCHEMA_VERSION = 2
SCHEMA_VERSION = 3
METHOD = "whole_training_chaser_response"
METHOD_VERSION = "1"

ARROW_CONTRACT_ENVELOPE_SCHEMA_ID = (
    "palette.training_response_analytics.arrow_contracts"
)
LEGACY_ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION = 1
ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION = 2
ARROW_TABLE_SCHEMA_NAMESPACE = "palette.training_response_analytics.arrow_table"

TRAINING_RESPONSE_FEATURES_TABLE = "training_response_features"
TRAINING_RESPONSE_CLASSIFICATION_TABLE = "training_response_classification"
TRAINING_RESPONSE_CLUSTERS_TABLE = "training_response_clusters"

TRAINING_RESPONSE_TABLES = (
    TRAINING_RESPONSE_FEATURES_TABLE,
    TRAINING_RESPONSE_CLASSIFICATION_TABLE,
    TRAINING_RESPONSE_CLUSTERS_TABLE,
)

IDENTITY_COLUMNS = (
    "recording_id",
    "session_id",
    "subject_id",
    "training_window_id",
)
PRIMARY_KEY_COLUMNS = (
    "analysis_run_id",
    "recording_id",
    "session_id",
    "subject_id",
)
LEGACY_V2_PRIMARY_KEY_COLUMNS = ("analysis_run_id", "recording_id")

AGGRESSIVE_ROLE = "aggressive"
INERT_ROLE = "inert"

TRAINING_RESPONSE_CLUSTER_AXES = (
    "locomotor_response_score",
    "boundary_response_score",
    "aggressive_proximity_score",
    "role_distance_selectivity_score",
    "close_contact_vigor_score",
)


@dataclass(frozen=True)
class TrainingResponseConfig:
    pre_window_label: str = "pre_event"
    training_window_label: str = "training_event"
    aggressive_role: str = AGGRESSIVE_ROLE
    inert_role: str = INERT_ROLE
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
        if self.aggressive_role != AGGRESSIVE_ROLE or self.inert_role != INERT_ROLE:
            raise ValueError(
                "training-response v3 freezes role-derived columns to "
                "aggressive_role='aggressive' and inert_role='inert'"
            )
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


_COMMON_ARROW_FIELDS = (
    field("schema_id", "string"),
    field("schema_version", "int32"),
    field("table_name", "string"),
    field("method", "string"),
    field("method_version", "string"),
    field("recording_id", "string"),
    field("session_id", "string"),
    field("subject_id", "string"),
    # A missing training window is a valid invalid-feature outcome. It is
    # lineage, not part of the v3 primary key.
    field("training_window_id", "int64", nullable=True),
    field("source_export_run_id", "string"),
    field("protocol_name", "string", nullable=True),
)

_LEGACY_V2_COMMON_ARROW_FIELDS = tuple(
    item for item in _COMMON_ARROW_FIELDS if item.name not in {"session_id", "subject_id"}
)

_FEATURE_FLOAT_FIELDS = (
    "pre_duration_s",
    "training_duration_s",
    "pre_tracking_dropout_fraction",
    "training_tracking_dropout_fraction",
    "pre_valid_position_fraction",
    "training_valid_position_fraction",
    "pre_path_per_min_mm",
    "training_path_per_min_mm",
    "path_per_min_log2_ratio",
    "pre_mean_speed_mm_s",
    "training_mean_speed_mm_s",
    "mean_speed_mm_s_log2_ratio",
    "pre_p95_speed_mm_s",
    "training_p95_speed_mm_s",
    "p95_speed_mm_s_log2_ratio",
    "pre_bout_rate_per_min",
    "training_bout_rate_per_min",
    "bout_rate_per_min_log2_ratio",
    "pre_mean_bout_path_length_mm",
    "training_mean_bout_path_length_mm",
    "mean_bout_path_length_mm_log2_ratio",
    "pre_mean_abs_bout_net_heading_change_deg",
    "training_mean_abs_bout_net_heading_change_deg",
    "mean_abs_bout_net_heading_change_deg_log2_ratio",
    "arena_radius_mm",
    "pre_wall_fraction",
    "training_wall_fraction",
    "wall_fraction_delta",
    "pre_center_distance_norm",
    "training_center_distance_norm",
    "center_distance_norm_delta",
    "aggressive_pre_p05_distance_mm",
    "aggressive_training_p05_distance_mm",
    "aggressive_p05_distance_mm_delta",
    "aggressive_pre_p50_distance_mm",
    "aggressive_training_p50_distance_mm",
    "aggressive_p50_distance_mm_delta",
    "aggressive_pre_fraction_within_threshold",
    "aggressive_training_fraction_within_threshold",
    "aggressive_fraction_within_threshold_delta",
    "aggressive_threshold_mm",
    "aggressive_pre_mean_alignment_cos",
    "aggressive_training_mean_alignment_cos",
    "aggressive_mean_alignment_cos_delta",
    "aggressive_pre_fraction_front_45",
    "aggressive_training_fraction_front_45",
    "aggressive_fraction_front_45_delta",
    "aggressive_pre_fraction_behind_45",
    "aggressive_training_fraction_behind_45",
    "aggressive_fraction_behind_45_delta",
    "aggressive_pre_circular_resultant_length",
    "aggressive_training_circular_resultant_length",
    "aggressive_circular_resultant_length_delta",
    "aggressive_training_near_speed_mm_s",
    "aggressive_training_far_speed_mm_s",
    "aggressive_near_minus_far_speed_mm_s",
    "inert_pre_p05_distance_mm",
    "inert_training_p05_distance_mm",
    "inert_p05_distance_mm_delta",
    "inert_pre_p50_distance_mm",
    "inert_training_p50_distance_mm",
    "inert_p50_distance_mm_delta",
    "inert_pre_fraction_within_threshold",
    "inert_training_fraction_within_threshold",
    "inert_fraction_within_threshold_delta",
    "inert_threshold_mm",
    "inert_pre_mean_alignment_cos",
    "inert_training_mean_alignment_cos",
    "inert_mean_alignment_cos_delta",
    "inert_pre_fraction_front_45",
    "inert_training_fraction_front_45",
    "inert_fraction_front_45_delta",
    "inert_pre_fraction_behind_45",
    "inert_training_fraction_behind_45",
    "inert_fraction_behind_45_delta",
    "inert_pre_circular_resultant_length",
    "inert_training_circular_resultant_length",
    "inert_circular_resultant_length_delta",
    "inert_training_near_speed_mm_s",
    "inert_training_far_speed_mm_s",
    "inert_near_minus_far_speed_mm_s",
    "training_role_p50_distance_contrast_mm",
    "training_role_p05_distance_contrast_mm",
    "training_role_within_threshold_contrast",
)

_FEATURE_ARROW_FIELDS = (
    _COMMON_ARROW_FIELDS
    + (
        field("pre_window_label", "string"),
        field("training_window_label", "string"),
        field("zarr_path", "string", nullable=True),
        field("temporal_training_features_available", "bool"),
        field("temporal_training_feature_reason", "string"),
    )
    + tuple(field(name, "float64", nullable=True) for name in _FEATURE_FLOAT_FIELDS)
    + (
        field("feature_status", "string"),
        field("feature_reason", "string", nullable=True),
        field("interpretation_guardrail", "string"),
        field("analysis_run_id", "string"),
    )
)

_CLASSIFICATION_ARROW_FIELDS = _COMMON_ARROW_FIELDS + (
    field("zarr_path", "string", nullable=True),
    field("locomotor_response_score", "float64", nullable=True),
    field("locomotor_response_metric_count", "int64"),
    field("boundary_response_score", "float64", nullable=True),
    field("boundary_response_metric_count", "int64"),
    field("aggressive_proximity_score", "float64", nullable=True),
    field("aggressive_proximity_metric_count", "int64"),
    field("role_distance_selectivity_score", "float64", nullable=True),
    field("role_distance_selectivity_metric_count", "int64"),
    field("close_contact_vigor_score", "float64", nullable=True),
    field("close_contact_vigor_metric_count", "int64"),
    field("classification_status", "string"),
    field("classification_reason", "string", nullable=True),
    field("reference_scope", "string"),
    field("relative_score_threshold", "float64"),
    field("locomotor_response", "string"),
    field("boundary_response", "string"),
    field("aggressive_proximity_state", "string"),
    field("role_distance_selectivity", "string"),
    field("close_contact_vigor", "string"),
    field("primary_training_profile", "string"),
    field("profile_separation_semantics", "string"),
    field("causal_avoidance_inference_permitted", "bool"),
    field("temporal_adaptation_inference_permitted", "bool"),
    field("profile_separation_score", "float64", nullable=True),
    field("analysis_run_id", "string"),
)

_CLUSTER_ARROW_FIELDS = _COMMON_ARROW_FIELDS + (
    field("zarr_path", "string", nullable=True),
    field("cluster_status", "string"),
    field("cluster_reason", "string", nullable=True),
    field("cluster_id", "int64", nullable=True),
    field("cluster_probability", "float64", nullable=True),
    field("cluster_probability_threshold", "float64"),
    field("selected_component_count", "int64", nullable=True),
    field("selected_bic", "float64", nullable=True),
    # V1 inferred a variable struct from a dict whose keys depended on the
    # cohort. V2 stores canonical strict JSON instead.
    field("bic_by_component_count_json", "string", nullable=True),
    field("cluster_stability_median_ari", "float64", nullable=True),
    field("cluster_stability_threshold", "float64"),
    field("cluster_stability_resample_count", "int64"),
    field("cluster_min_rows_per_component", "int64"),
    field("cluster_axes", "list<string>"),
    field("cluster_semantics", "string"),
    field("analysis_run_id", "string"),
)

_LEGACY_V2_FEATURE_ARROW_FIELDS = (
    _LEGACY_V2_COMMON_ARROW_FIELDS
    + _FEATURE_ARROW_FIELDS[len(_COMMON_ARROW_FIELDS) :]
)
_LEGACY_V2_CLASSIFICATION_ARROW_FIELDS = (
    _LEGACY_V2_COMMON_ARROW_FIELDS
    + _CLASSIFICATION_ARROW_FIELDS[len(_COMMON_ARROW_FIELDS) :]
)
_LEGACY_V2_CLUSTER_ARROW_FIELDS = (
    _LEGACY_V2_COMMON_ARROW_FIELDS
    + _CLUSTER_ARROW_FIELDS[len(_COMMON_ARROW_FIELDS) :]
)

ARROW_TABLE_CONTRACTS: dict[str, ArrowTableContract] = {
    TRAINING_RESPONSE_FEATURES_TABLE: ArrowTableContract(
        table_name=TRAINING_RESPONSE_FEATURES_TABLE,
        fields=_FEATURE_ARROW_FIELDS,
        schema_version=SCHEMA_VERSION,
        schema_namespace=ARROW_TABLE_SCHEMA_NAMESPACE,
        primary_key=PRIMARY_KEY_COLUMNS,
    ),
    TRAINING_RESPONSE_CLASSIFICATION_TABLE: ArrowTableContract(
        table_name=TRAINING_RESPONSE_CLASSIFICATION_TABLE,
        fields=_CLASSIFICATION_ARROW_FIELDS,
        schema_version=SCHEMA_VERSION,
        schema_namespace=ARROW_TABLE_SCHEMA_NAMESPACE,
        primary_key=PRIMARY_KEY_COLUMNS,
    ),
    TRAINING_RESPONSE_CLUSTERS_TABLE: ArrowTableContract(
        table_name=TRAINING_RESPONSE_CLUSTERS_TABLE,
        fields=_CLUSTER_ARROW_FIELDS,
        schema_version=SCHEMA_VERSION,
        schema_namespace=ARROW_TABLE_SCHEMA_NAMESPACE,
        primary_key=PRIMARY_KEY_COLUMNS,
    ),
}

LEGACY_V2_ARROW_TABLE_CONTRACTS: dict[str, ArrowTableContract] = {
    TRAINING_RESPONSE_FEATURES_TABLE: ArrowTableContract(
        table_name=TRAINING_RESPONSE_FEATURES_TABLE,
        fields=_LEGACY_V2_FEATURE_ARROW_FIELDS,
        schema_version=LEGACY_EXACT_SCHEMA_VERSION,
        schema_namespace=ARROW_TABLE_SCHEMA_NAMESPACE,
        primary_key=LEGACY_V2_PRIMARY_KEY_COLUMNS,
    ),
    TRAINING_RESPONSE_CLASSIFICATION_TABLE: ArrowTableContract(
        table_name=TRAINING_RESPONSE_CLASSIFICATION_TABLE,
        fields=_LEGACY_V2_CLASSIFICATION_ARROW_FIELDS,
        schema_version=LEGACY_EXACT_SCHEMA_VERSION,
        schema_namespace=ARROW_TABLE_SCHEMA_NAMESPACE,
        primary_key=LEGACY_V2_PRIMARY_KEY_COLUMNS,
    ),
    TRAINING_RESPONSE_CLUSTERS_TABLE: ArrowTableContract(
        table_name=TRAINING_RESPONSE_CLUSTERS_TABLE,
        fields=_LEGACY_V2_CLUSTER_ARROW_FIELDS,
        schema_version=LEGACY_EXACT_SCHEMA_VERSION,
        schema_namespace=ARROW_TABLE_SCHEMA_NAMESPACE,
        primary_key=LEGACY_V2_PRIMARY_KEY_COLUMNS,
    ),
}

INTERPRETATION_GUARDRAIL = (
    "descriptive response profile; fear, anxiety, and escape success are not inferred"
)
CLASSIFICATION_REFERENCE_SCOPE = "combined_source_export_cohort_relative"
PROFILE_SEPARATION_SEMANTICS = "descriptive_distance_not_probability"
CLUSTER_SEMANTICS = "unsupervised_ids_require_posthoc_interpretation"


def training_response_arrow_contract_envelope(
    table_names: Sequence[str] = TRAINING_RESPONSE_TABLES,
    *,
    schema_version: int = SCHEMA_VERSION,
) -> dict[str, object]:
    """Return a closed exact Arrow envelope for one supported schema."""

    contracts, envelope_version = _contracts_for_schema_version(schema_version)

    return contract_envelope(
        table_names,
        known_table_names=TRAINING_RESPONSE_TABLES,
        contracts=contracts,
        schema_id=ARROW_CONTRACT_ENVELOPE_SCHEMA_ID,
        schema_version=envelope_version,
    )


def validate_training_response_arrow_contract_envelope(
    value: object,
    table_names: Sequence[str] = TRAINING_RESPONSE_TABLES,
    *,
    schema_version: int = SCHEMA_VERSION,
) -> dict[str, object]:
    """Validate an envelope against the installed declarations."""

    contracts, envelope_version = _contracts_for_schema_version(schema_version)

    return validate_contract_envelope(
        value,
        table_names,
        known_table_names=TRAINING_RESPONSE_TABLES,
        contracts=contracts,
        schema_id=ARROW_CONTRACT_ENVELOPE_SCHEMA_ID,
        schema_version=envelope_version,
    )


def _contracts_for_schema_version(
    schema_version: int,
) -> tuple[dict[str, ArrowTableContract], int]:
    if schema_version == SCHEMA_VERSION:
        return ARROW_TABLE_CONTRACTS, ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION
    if schema_version == LEGACY_EXACT_SCHEMA_VERSION:
        return (
            LEGACY_V2_ARROW_TABLE_CONTRACTS,
            LEGACY_ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION,
        )
    raise ValueError(
        "training-response exact schemas support only v3 and explicit legacy v2"
    )


def exact_training_response_arrow_schema(
    table_name: str,
    *,
    metadata: Mapping[bytes, bytes] | None = None,
    schema_version: int = SCHEMA_VERSION,
) -> Any:
    """Return the digest-bound exact Arrow schema for one supported version."""

    contracts, _ = _contracts_for_schema_version(schema_version)
    try:
        contract = contracts[table_name]
    except KeyError as exc:
        raise KeyError(f"unknown training-response table: {table_name}") from exc
    return exact_schema(contract, metadata=metadata or {})


def validate_training_response_arrow_schema(
    table_name: str,
    schema: Any,
    *,
    schema_version: int = SCHEMA_VERSION,
) -> None:
    """Reject any physical field or exact footer-contract drift."""

    contracts, _ = _contracts_for_schema_version(schema_version)
    try:
        contract = contracts[table_name]
    except KeyError as exc:
        raise KeyError(f"unknown training-response table: {table_name}") from exc
    validate_exact_schema(contract, schema)


def _canonical_bic_json(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        def reject_duplicate_pairs(
            pairs: list[tuple[str, object]],
        ) -> dict[str, object]:
            decoded: dict[str, object] = {}
            for key, item in pairs:
                if key in decoded:
                    raise ValueError(
                        "bic_by_component_count_json has duplicate component keys"
                    )
                decoded[key] = item
            return decoded

        try:
            decoded = json.loads(value, object_pairs_hook=reject_duplicate_pairs)
        except json.JSONDecodeError as exc:
            raise ValueError("bic_by_component_count_json is not valid JSON") from exc
        canonical = _canonical_bic_json(decoded)
        if canonical != value:
            raise ValueError("bic_by_component_count_json is not canonical JSON")
        return canonical
    if not isinstance(value, Mapping):
        raise ValueError("BIC-by-component-count must be a JSON object or mapping")
    normalized: dict[str, float] = {}
    for raw_count, raw_bic in value.items():
        if isinstance(raw_count, bool):
            raise ValueError("BIC component counts must be positive integers")
        try:
            count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise ValueError("BIC component counts must be positive integers") from exc
        if count < 1 or str(raw_count) != str(count):
            raise ValueError("BIC component counts must be canonical positive integers")
        key = str(count)
        if key in normalized:
            raise ValueError(f"duplicate normalized BIC component count: {count}")
        if isinstance(raw_bic, bool):
            raise ValueError("BIC values must be finite numbers")
        try:
            bic = float(raw_bic)
        except (TypeError, ValueError) as exc:
            raise ValueError("BIC values must be finite numbers") from exc
        if not math.isfinite(bic):
            raise ValueError("BIC values must be finite numbers")
        normalized[key] = bic
    return canonical_bytes(normalized).decode("ascii")


def _normalization_defaults(
    table_name: str,
    config: TrainingResponseConfig,
) -> dict[str, object]:
    if table_name == TRAINING_RESPONSE_FEATURES_TABLE:
        return {"interpretation_guardrail": INTERPRETATION_GUARDRAIL}
    if table_name == TRAINING_RESPONSE_CLASSIFICATION_TABLE:
        return {
            "reference_scope": CLASSIFICATION_REFERENCE_SCOPE,
            "relative_score_threshold": config.relative_score_threshold,
        }
    if table_name == TRAINING_RESPONSE_CLUSTERS_TABLE:
        return {
            "cluster_probability_threshold": config.cluster_probability_threshold,
            "cluster_stability_threshold": config.cluster_stability_threshold,
            "cluster_stability_resample_count": 0,
            "cluster_min_rows_per_component": config.cluster_min_rows_per_component,
            "cluster_axes": list(TRAINING_RESPONSE_CLUSTER_AXES),
            "cluster_semantics": CLUSTER_SEMANTICS,
        }
    raise KeyError(f"unknown training-response table: {table_name}")


def _normalize_value(name: str, arrow_type: str, value: object) -> object:
    if arrow_type == "string":
        if type(value) is not str:
            raise ValueError(f"{name} must be a string")
        return value
    if arrow_type == "bool":
        if type(value) is not bool:
            raise ValueError(f"{name} must be a bool")
        return value
    if arrow_type in {"int32", "int64"}:
        if type(value) is not int:
            raise ValueError(f"{name} must be an integer")
        if arrow_type == "int32" and not -(2**31) <= value < 2**31:
            raise ValueError(f"{name} is outside the int32 range")
        if arrow_type == "int64" and not -(2**63) <= value < 2**63:
            raise ValueError(f"{name} is outside the int64 range")
        return value
    if arrow_type == "float64":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{name} must be a finite number")
        return normalized
    if arrow_type == "list<string>":
        if not isinstance(value, (list, tuple)) or any(
            type(item) is not str for item in value
        ):
            raise ValueError(f"{name} must be a list of strings")
        return list(value)
    raise ValueError(f"unsupported training-response Arrow type: {arrow_type}")


def normalize_training_response_row(
    table_name: str,
    row: Mapping[str, Any],
    *,
    analysis_run_id: str | None = None,
    config: TrainingResponseConfig | None = None,
) -> dict[str, Any]:
    """Normalize one producer row to the closed v3 vocabulary."""

    try:
        contract = ARROW_TABLE_CONTRACTS[table_name]
    except KeyError as exc:
        raise KeyError(f"unknown training-response table: {table_name}") from exc
    config = config or TrainingResponseConfig()
    config.validate()
    normalized = dict(row)
    if table_name == TRAINING_RESPONSE_CLUSTERS_TABLE:
        legacy_present = "bic_by_component_count" in normalized
        exact_present = "bic_by_component_count_json" in normalized
        if legacy_present and exact_present:
            raise ValueError(
                "cluster row must not contain both legacy and v2 BIC representations"
            )
        if legacy_present:
            normalized["bic_by_component_count_json"] = _canonical_bic_json(
                normalized.pop("bic_by_component_count")
            )
        elif exact_present:
            normalized["bic_by_component_count_json"] = _canonical_bic_json(
                normalized["bic_by_component_count_json"]
            )
    if analysis_run_id is not None:
        existing = normalized.get("analysis_run_id")
        if existing is not None and existing != analysis_run_id:
            raise ValueError("analysis_run_id does not match the normalized run")
        normalized["analysis_run_id"] = analysis_run_id
    for name, value in _normalization_defaults(table_name, config).items():
        normalized.setdefault(name, value)
    expected_names = tuple(item.name for item in contract.fields)
    unexpected = sorted(set(normalized) - set(expected_names))
    if unexpected:
        raise ValueError(
            f"{table_name}: unexpected fields outside the frozen v3 role vocabulary: "
            f"{unexpected}"
        )
    output: dict[str, Any] = {}
    for item in contract.fields:
        value = normalized.get(item.name)
        if value is None:
            if not item.nullable:
                raise ValueError(f"{table_name}: required field is null: {item.name}")
            output[item.name] = None
        else:
            output[item.name] = _normalize_value(
                item.name,
                item.arrow_type,
                value,
            )
    expected_constants = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "table_name": table_name,
        "method": METHOD,
        "method_version": METHOD_VERSION,
    }
    for name, expected in expected_constants.items():
        if output[name] != expected:
            raise ValueError(
                f"{table_name}: {name} must equal the installed v3 value {expected!r}"
            )
    return output


def validate_training_response_primary_keys(
    table_name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    schema_version: int = SCHEMA_VERSION,
) -> None:
    """Reject null, empty, or duplicate table primary keys."""

    contracts, _ = _contracts_for_schema_version(schema_version)
    if table_name not in contracts:
        raise KeyError(f"unknown training-response table: {table_name}")
    primary_key = contracts[table_name].primary_key
    observed: set[tuple[str, ...]] = set()
    for row_index, row in enumerate(rows):
        values: list[str] = []
        for column in primary_key:
            value = row.get(column)
            if type(value) is not str or not value.strip():
                raise ValueError(
                    f"{table_name}: row {row_index} has null or empty primary-key "
                    f"field {column}"
                )
            values.append(value)
        key = tuple(values)
        if key in observed:
            raise ValueError(f"{table_name}: duplicate primary key {key!r}")
        observed.add(key)


def normalize_training_response_rows(
    table_name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    analysis_run_id: str,
    config: TrainingResponseConfig | None = None,
) -> list[dict[str, Any]]:
    """Normalize a complete batch and prove its v3 primary-key contract."""

    normalized = [
        normalize_training_response_row(
            table_name,
            row,
            analysis_run_id=analysis_run_id,
            config=config,
        )
        for row in rows
    ]
    validate_training_response_primary_keys(table_name, normalized)
    return normalized


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
    "AGGRESSIVE_ROLE",
    "ARROW_CONTRACT_ENVELOPE_SCHEMA_ID",
    "ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION",
    "ARROW_TABLE_CONTRACTS",
    "CLASSIFICATION_REFERENCE_SCOPE",
    "CLUSTER_SEMANTICS",
    "IDENTITY_COLUMNS",
    "INTERPRETATION_GUARDRAIL",
    "INERT_ROLE",
    "LEGACY_SCHEMA_VERSION",
    "LEGACY_EXACT_SCHEMA_VERSION",
    "LEGACY_ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION",
    "LEGACY_V2_ARROW_TABLE_CONTRACTS",
    "LEGACY_V2_PRIMARY_KEY_COLUMNS",
    "METHOD",
    "METHOD_VERSION",
    "PRIMARY_KEY_COLUMNS",
    "PROFILE_SEPARATION_SEMANTICS",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "TRAINING_RESPONSE_CLUSTER_AXES",
    "TRAINING_RESPONSE_CLASSIFICATION_TABLE",
    "TRAINING_RESPONSE_CLUSTERS_TABLE",
    "TRAINING_RESPONSE_FEATURES_TABLE",
    "TRAINING_RESPONSE_TABLES",
    "TrainingResponseConfig",
    "contract_fields",
    "exact_training_response_arrow_schema",
    "normalize_training_response_row",
    "normalize_training_response_rows",
    "training_response_arrow_contract_envelope",
    "validate_training_response_arrow_contract_envelope",
    "validate_training_response_arrow_schema",
    "validate_training_response_primary_keys",
]
