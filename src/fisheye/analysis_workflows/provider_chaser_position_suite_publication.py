"""Immutable publication for provider-aware chaser position summaries.

The scientific computation returns ordinary row dictionaries so canaries and
plots remain convenient.  Production consumers need a stricter boundary: row
evidence belongs in typed Zarr arrays, metadata stays bounded, exact source
authorities remain readable, and publication cannot activate a selector.

String columns use table-local integer registries.  Every floating-point
column has an explicit validity array; invalid values are stored as NaN and
cannot be mistaken for measured zeros.  Ordinary readers validate declarations
and shapes without re-hashing output payloads.  A separately named deep audit
recomputes the declared content hashes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.analysis.provider_chaser_position_suite import (
    METHOD_ID,
    SCHEMA_ID as SUITE_SCHEMA_ID,
    SCHEMA_VERSION as SUITE_SCHEMA_VERSION,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication import (
    load_protocol_semantic_chaser_selection_source_handle,
)
from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)


RUNS_PARENT_PATH = "analysis/provider_chaser_position_suite_runs"
RUNS_PREFIX = f"{RUNS_PARENT_PATH}/"
STORAGE_SCHEMA_ID = "palette.analysis.provider_chaser_position_suite"
STORAGE_SCHEMA_VERSION = 1
STORAGE_LAYOUT = "flat_typed_tables_v1"
PUBLICATION_SCHEMA_ID = f"{STORAGE_SCHEMA_ID}.publication"
PUBLICATION_SCHEMA_VERSION = 1
MANIFEST_ATTR = "provider_chaser_position_suite_manifest"
MANIFEST_DIGEST_ATTR = "provider_chaser_position_suite_manifest_sha256"
PUBLICATION_POLICY = "immutable_exact_sources_selector_ineligible_v1"
DEEP_AUDIT_POLICY = "explicit_recompute_declared_output_hashes_v1"
CANARY_SCHEMA_ID = "palette.provider_chaser_position_suite_canary"
CANARY_SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 262_144

_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SELECTOR_ALIASES = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_pending",
        "current",
        "current_run",
        "default",
        "selected",
        "selected_run",
        "authoritative",
        "authoritative_run",
    }
)
_FORBIDDEN_SELECTOR_ATTRS = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
        "selected",
        "selected_run",
        "current",
        "current_run",
        "default",
        "default_run",
    }
)
_ROW_TABLE_FIELDS = frozenset(
    {
        "epoch_roles",
        "per_epoch_chaser_metrics",
        "distance_cdf",
        "radial_occupancy",
        "quadrant_joint_occupancy",
        "role_contrasts",
        "role_radial_contrasts",
    }
)


class ProviderChaserPositionSuitePublicationError(ValueError):
    """Raised when a position-suite publication violates its contract."""


def _fail(message: str) -> None:
    raise ProviderChaserPositionSuitePublicationError(message)


@dataclass(frozen=True, slots=True)
class _Column:
    name: str
    logical_type: str


def _columns(*items: tuple[str, str]) -> tuple[_Column, ...]:
    return tuple(_Column(name, kind) for name, kind in items)


TABLE_COLUMNS: Mapping[str, tuple[_Column, ...]] = MappingProxyType(
    {
        "epoch_roles": _columns(
            ("analysis_role", "string"),
            ("window_id", "int64"),
            ("source_label", "string"),
            ("start_frame", "int64"),
            ("end_frame_exclusive", "int64"),
            ("source_interval_sha256", "string"),
        ),
        "per_epoch_chaser_metrics": _columns(
            ("analysis_role", "string"),
            ("epoch_window_id", "int64"),
            ("epoch_source_label", "string"),
            ("epoch_start_frame", "int64"),
            ("epoch_end_frame_exclusive", "int64"),
            ("source_interval_sha256", "string"),
            ("source_interval_frame_count", "int64"),
            ("epoch_provider_frame_count", "int64"),
            ("epoch_provider_frame_coverage_fraction", "float64"),
            ("chaser_column", "int64"),
            ("chaser_identity_code", "int64"),
            ("chaser_identity", "string"),
            ("behavior_role_code", "int64"),
            ("behavior_role", "string"),
            ("candidate_frame_count", "int64"),
            ("valid_distance_frame_count", "int64"),
            ("valid_distance_fraction", "float64"),
            ("distance_mean_mm", "float64"),
            ("distance_p05_mm", "float64"),
            ("distance_p25_mm", "float64"),
            ("distance_p50_mm", "float64"),
            ("distance_p75_mm", "float64"),
            ("distance_p95_mm", "float64"),
            ("same_quadrant_valid_frame_count", "int64"),
            ("same_quadrant_fraction_valid", "float64"),
            ("same_quadrant_fraction_candidate", "float64"),
            ("near_zone_frame_count", "int64"),
            ("near_zone_fraction_valid", "float64"),
            ("near_zone_fraction_candidate", "float64"),
            ("near_zone_dwell_s", "float64"),
            ("near_zone_expected_fraction_geometric", "float64"),
            ("near_zone_enrichment_geometric", "float64"),
            ("near_zone_entry_count", "int64"),
            ("near_zone_entry_rate_per_min_valid_time", "float64"),
            ("near_zone_valid_tracked_duration_s", "float64"),
            ("near_zone_complete_visit_median_dwell_s", "float64"),
            ("near_zone_complete_visit_total_dwell_s", "float64"),
            ("near_zone_invalid_gap_count", "int64"),
            ("near_zone_censor_event_count", "int64"),
            ("near_zone_boundary_censor_event_count", "int64"),
            ("near_zone_invalid_gap_censor_event_count", "int64"),
            ("wall_excluded_valid_frame_count", "int64"),
            ("fish_arena_radius_mean_mm", "float64"),
            ("fish_arena_radius_p50_mm", "float64"),
            ("fish_wall_distance_mean_mm", "float64"),
            ("fish_wall_distance_p50_mm", "float64"),
        ),
        "distance_cdf": _columns(
            ("analysis_role", "string"),
            ("epoch_window_id", "int64"),
            ("behavior_role", "string"),
            ("chaser_identity", "string"),
            ("threshold_mm", "float64"),
            ("fraction_at_or_below", "float64"),
        ),
        "radial_occupancy": _columns(
            ("analysis_role", "string"),
            ("epoch_window_id", "int64"),
            ("behavior_role", "string"),
            ("chaser_identity", "string"),
            ("bin_start_mm", "float64"),
            ("bin_end_mm", "float64"),
            ("observed_count", "int64"),
            ("observed_fraction", "float64"),
            ("expected_available_area_mm2_frames", "float64"),
            ("expected_fraction_geometric", "float64"),
            ("selection_index_geometric", "float64"),
            ("wall_excluded_observed_count", "int64"),
            ("wall_excluded_observed_fraction", "float64"),
            ("wall_excluded_expected_available_area_mm2_frames", "float64"),
            ("wall_excluded_expected_fraction_geometric", "float64"),
            ("wall_excluded_selection_index_geometric", "float64"),
        ),
        "quadrant_joint_occupancy": _columns(
            ("analysis_role", "string"),
            ("epoch_window_id", "int64"),
            ("behavior_role", "string"),
            ("chaser_identity", "string"),
            ("fish_quadrant", "string"),
            ("chaser_quadrant", "string"),
            ("valid_joint_frame_count", "int64"),
            ("valid_joint_fraction", "float64"),
        ),
        "role_contrasts": _columns(
            ("analysis_role", "string"),
            ("epoch_window_id", "int64"),
            ("metric", "string"),
            ("treatment_role", "string"),
            ("baseline_role", "string"),
            ("treatment_value", "float64"),
            ("baseline_value", "float64"),
            ("treatment_minus_baseline", "float64"),
        ),
        "role_radial_contrasts": _columns(
            ("analysis_role", "string"),
            ("epoch_window_id", "int64"),
            ("bin_start_mm", "float64"),
            ("bin_end_mm", "float64"),
            ("treatment_role", "string"),
            ("baseline_role", "string"),
            ("observed_fraction_treatment_minus_baseline", "float64"),
            ("selection_index_treatment_minus_baseline", "float64"),
        ),
    }
)


def _copy_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _copy_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _strict_json_object(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field_name} must be one JSON object.")
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        _fail(f"{field_name} is not strict JSON: {exc}")
    if not isinstance(decoded, dict):  # pragma: no cover
        _fail(f"{field_name} must decode to an object.")
    return decoded


def _run_name(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or value in {".", ".."}
        or value in _SELECTOR_ALIASES
        or _RUN_NAME_RE.fullmatch(value) is None
    ):
        _fail("run_name must be one concrete bare run name, not a selector or path.")
    return value


def _canonical_archive(value: str | Path) -> Path:
    archive = Path(value).expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr archive does not exist: {archive}")
    return archive


def _canonical_run_path(run_name: str) -> str:
    return f"{RUNS_PREFIX}{_run_name(run_name)}"


def _digest(value: object, *, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{field_name} must be one lowercase SHA-256 digest.")
    return value


def _validate_source_bindings(value: object) -> dict[str, Any]:
    bindings = _strict_json_object(value, field_name="report.source_bindings")
    required = {
        "provider_chaser_distance",
        "relative_frame",
        "epoch_candidate",
        "epoch_selection",
        "arena_geometry_and_scale",
        "provider_physical_frame",
        "recording_physical_frame",
        "physical_frame_equivalence",
        "source_camera_to_arena_mm_transform",
    }
    missing = sorted(required - set(bindings))
    if missing:
        _fail(f"Position-suite source bindings are incomplete: {missing!r}.")
    provider = bindings.get("provider_chaser_distance")
    if not isinstance(provider, Mapping):
        _fail("Exact provider chaser-distance source binding is malformed.")
    for field_name in ("run_name", "run_path"):
        if type(provider.get(field_name)) is not str or not provider.get(field_name):
            _fail(f"Provider source binding {field_name} is invalid.")
    if provider["run_path"] != (
        f"analysis/provider_chaser_distance_runs/{provider['run_name']}"
    ):
        _fail("Provider source run name and exact run path disagree.")
    _digest(provider.get("manifest_sha256"), field_name="provider.manifest_sha256")
    _digest(
        provider.get("source_receipt_sha256"),
        field_name="provider.source_receipt_sha256",
    )
    if provider.get("verification_mode") not in {
        "bounded_publication",
        "deep_audit",
    }:
        _fail("Provider source binding lacks a validated publication mode.")
    if not isinstance(provider.get("source_position_provider"), Mapping):
        _fail("Provider source position authority is missing.")
    for field_name in (
        "relative_frame",
        "epoch_candidate",
        "epoch_selection",
        "arena_geometry_and_scale",
        "provider_physical_frame",
        "recording_physical_frame",
        "physical_frame_equivalence",
        "source_camera_to_arena_mm_transform",
    ):
        if not isinstance(bindings.get(field_name), Mapping):
            _fail(f"Position-suite source binding {field_name!r} is malformed.")
    semantic = bindings.get("protocol_semantic_selection")
    if semantic is not None:
        if not isinstance(semantic, Mapping):
            _fail("Protocol-semantic selection source binding is malformed.")
        run_name = semantic.get("run_name")
        run_path = semantic.get("run_path")
        semantic_run_name = _run_name(run_name)
        if (
            run_path
            != (
                "analysis/protocol_semantic_chaser_selection_runs/"
                f"{semantic_run_name}"
            )
        ):
            _fail("Protocol-semantic selection exact run identity is invalid.")
        _digest(
            semantic.get("manifest_sha256"),
            field_name="protocol_semantic_selection.manifest_sha256",
        )
        _digest(
            semantic.get("selection_identity_sha256"),
            field_name="protocol_semantic_selection.selection_identity_sha256",
        )
        semantic_hash = semantic.get("protocol_semantic_hash")
        if (
            type(semantic_hash) is not str
            or not semantic_hash.startswith("sha256:")
            or len(semantic_hash) != 71
        ):
            _fail("Protocol-semantic selection hash is malformed.")
        _digest(
            semantic_hash.removeprefix("sha256:"),
            field_name="protocol_semantic_selection.protocol_semantic_hash",
        )
        _digest(
            semantic.get("palette_computed_trial_index_sha256"),
            field_name=(
                "protocol_semantic_selection."
                "palette_computed_trial_index_sha256"
            ),
        )
        if (
            semantic.get("trial_index_integrity_status")
            != "palette_computed_not_producer_asserted"
        ):
            _fail("Protocol-semantic trial-index integrity status is invalid.")
        if semantic.get("roles") != [
            "chaser_pre",
            "chaser_training",
            "chaser_post",
        ]:
            _fail("Protocol-semantic selection role hierarchy is invalid.")
        if semantic.get("step_end_interval_semantics") not in {
            "producer_contract_pending",
            "producer_declared_step_end_inclusive",
            "producer_declared_step_end_exclusive",
        }:
            _fail("Protocol-semantic step-end policy is invalid.")
        standalone_status = semantic.get("standalone_solid_black_status")
        if standalone_status not in {
            "not_applicable_protocol_has_no_standalone_solid_black",
            "present_not_selected",
            "selected",
        }:
            _fail("Protocol-semantic standalone-baseline status is invalid.")
        if (
            semantic.get("selector_eligible") is not False
            or semantic.get("production_authority") is not False
        ):
            _fail("Protocol-semantic selection must remain selector-ineligible.")
        source_epoch_selection = semantic.get("source_epoch_selection")
        if not isinstance(source_epoch_selection, Mapping):
            _fail("Protocol-semantic source epoch identity is missing.")
        for field_name in (
            "source_epoch_run_manifest_sha256",
            "source_epoch_run_manifest_payload_sha256",
            "source_epoch_logical_content_sha256",
            "source_epoch_lineage_hash",
            "source_epoch_lineage_payload_sha256",
            "source_timeline_digest",
            "selection_sha256",
        ):
            _digest(
                source_epoch_selection.get(field_name),
                field_name=f"protocol_semantic_selection.{field_name}",
            )
        source_epoch_path = source_epoch_selection.get("source_epoch_run_path")
        epoch_prefix = "analysis/stimulus_epoch_runs/"
        if (
            type(source_epoch_path) is not str
            or not source_epoch_path.startswith(epoch_prefix)
        ):
            _fail("Protocol-semantic source epoch run path is invalid.")
        _run_name(source_epoch_path[len(epoch_prefix) :])
        epoch_selection = bindings["epoch_selection"]
        if source_epoch_selection.get("selection_sha256") != epoch_selection.get(
            "selection_sha256"
        ):
            _fail(
                "Protocol-semantic and position-suite epoch selections differ."
            )
        epoch_candidate = bindings["epoch_candidate"]
        candidate_epoch_path = epoch_candidate.get("epoch_run_path")
        if (
            candidate_epoch_path is not None
            and source_epoch_selection.get("source_epoch_run_path")
            != candidate_epoch_path
        ):
            _fail(
                "Protocol-semantic and provider candidate epoch paths differ."
            )
        semantic_epochs = semantic.get("position_suite_epochs")
        if not isinstance(semantic_epochs, list) or len(semantic_epochs) != 3:
            _fail("Protocol-semantic position-suite epoch binding is malformed.")
        semantic_epochs_sha256 = _digest(
            semantic.get("position_suite_epochs_sha256"),
            field_name="protocol_semantic_selection.position_suite_epochs_sha256",
        )
        if canonical_json_sha256(semantic_epochs) != semantic_epochs_sha256:
            _fail("Protocol-semantic position-suite epoch digest is stale.")
        semantic_roles = semantic.get("semantic_role_bindings")
        if (
            not isinstance(semantic_roles, list)
            or any(not isinstance(row, Mapping) for row in semantic_roles)
            or [row.get("analysis_role") for row in semantic_roles]
            != ["chaser_pre", "chaser_training", "chaser_post"]
        ):
            _fail("Protocol-semantic per-role bindings are malformed.")
        semantic_roles_sha256 = _digest(
            semantic.get("semantic_role_bindings_sha256"),
            field_name="protocol_semantic_selection.semantic_role_bindings_sha256",
        )
        if canonical_json_sha256(semantic_roles) != semantic_roles_sha256:
            _fail("Protocol-semantic per-role binding digest is stale.")
        if semantic.get("position_suite_scope") != {
            "analysis_epoch_scope": "chaser_internal_windows",
            "behavior_role_contrast_scope": (
                "within_epoch_treatment_minus_baseline"
            ),
            "standalone_protocol_baseline_included": False,
            "standalone_protocol_baseline_status": standalone_status,
        }:
            _fail("Protocol-semantic position-suite scientific scope is invalid.")
    mode = bindings.get("epoch_binding_mode")
    expected_mode = (
        "protocol_semantic_selection_v2"
        if semantic is not None
        else "caller_bound_legacy_v1"
    )
    if mode not in (None, expected_mode):
        _fail("Position-suite epoch binding mode contradicts its source authority.")
    bindings["epoch_binding_mode"] = expected_mode
    return bindings


def _validate_protocol_semantic_source(
    archive: Path,
    bindings: Mapping[str, Any],
    *,
    expected_recording_id: str,
) -> None:
    semantic = bindings.get("protocol_semantic_selection")
    if semantic is None:
        return
    if not isinstance(semantic, Mapping):  # pragma: no cover - prevalidated
        _fail("Protocol-semantic selection source binding is malformed.")
    try:
        handle = load_protocol_semantic_chaser_selection_source_handle(
            archive,
            run_name=str(semantic["run_name"]),
            expected_recording_id=expected_recording_id,
            use_consolidated=True,
            deep_audit=True,
        )
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ProviderChaserPositionSuitePublicationError(
            f"Unable to reload exact protocol-semantic selection source: {exc}"
        ) from exc
    if handle.source_binding() != dict(semantic):
        _fail(
            "Position-suite protocol-semantic binding differs from its exact "
            "immutable source."
        )


def _readonly_array(value: Any) -> np.ndarray:
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


def _encode_table(
    table: str,
    rows_value: object,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(rows_value, Sequence) or isinstance(rows_value, (str, bytes)):
        _fail(f"suite.{table} must be one nonempty row sequence.")
    rows = list(rows_value)
    if not rows:
        _fail(f"suite.{table} must be nonempty.")
    columns = TABLE_COLUMNS[table]
    expected_fields = {column.name for column in columns}
    normalized_rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            _fail(f"suite.{table}[{index}] must be one row object.")
        if set(row) != expected_fields:
            missing = sorted(expected_fields - set(row))
            extra = sorted(set(row) - expected_fields)
            _fail(
                f"suite.{table}[{index}] columns differ; missing={missing!r}, "
                f"extra={extra!r}."
            )
        normalized_rows.append(row)

    arrays: dict[str, np.ndarray] = {
        f"{table}__row_index": np.arange(len(rows), dtype=np.int64)
    }
    contracts: list[dict[str, Any]] = [
        {
            "name": "row_index",
            "logical_type": "int64",
            "nullable": False,
            "value_array": f"{table}__row_index",
            "encoding": "dense_zero_based_row_index_v1",
        }
    ]
    registries: dict[str, Any] = {}
    for column in columns:
        path = f"{table}__{column.name}"
        values = [row[column.name] for row in normalized_rows]
        if column.logical_type == "string":
            if any(type(value) is not str or not value for value in values):
                _fail(f"suite.{table}.{column.name} must contain nonempty strings.")
            labels = sorted(set(values))
            if len(labels) > np.iinfo(np.int32).max:  # pragma: no cover
                _fail(f"suite.{table}.{column.name} registry is too large.")
            lookup = {label: index for index, label in enumerate(labels)}
            arrays[path] = np.asarray(
                [lookup[value] for value in values], dtype=np.int32
            )
            registry_id = f"{table}.{column.name}"
            registries[registry_id] = {
                "encoding": "sorted_zero_based_int32_codes_v1",
                "code_dtype": np.dtype(np.int32).str,
                "values": labels,
            }
            contracts.append(
                {
                    "name": column.name,
                    "logical_type": "string",
                    "nullable": False,
                    "value_array": path,
                    "encoding": "registry_code_v1",
                    "registry_id": registry_id,
                }
            )
        elif column.logical_type == "int64":
            if any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                for value in values
            ):
                _fail(f"suite.{table}.{column.name} must contain exact integers.")
            arrays[path] = np.asarray(values, dtype=np.int64)
            contracts.append(
                {
                    "name": column.name,
                    "logical_type": "int64",
                    "nullable": False,
                    "value_array": path,
                    "encoding": "dense_int64_v1",
                }
            )
        elif column.logical_type == "float64":
            validity = np.asarray([value is not None for value in values], dtype=bool)
            numeric = np.full(len(values), np.nan, dtype=np.float64)
            for index, value in enumerate(values):
                if value is None:
                    continue
                if isinstance(value, (bool, np.bool_)):
                    _fail(f"suite.{table}.{column.name} contains a boolean.")
                try:
                    number = float(value)
                except (TypeError, ValueError) as exc:
                    raise ProviderChaserPositionSuitePublicationError(
                        f"suite.{table}.{column.name} contains a non-number."
                    ) from exc
                if not np.isfinite(number):
                    _fail(f"suite.{table}.{column.name} contains non-finite data.")
                numeric[index] = number
            valid_path = f"{path}_valid"
            arrays[path] = numeric
            arrays[valid_path] = validity
            contracts.append(
                {
                    "name": column.name,
                    "logical_type": "float64",
                    "nullable": True,
                    "value_array": path,
                    "validity_array": valid_path,
                    "encoding": "float64_nan_with_explicit_validity_v1",
                }
            )
        else:  # pragma: no cover
            _fail(f"Unsupported logical type {column.logical_type!r}.")
    return arrays, contracts, registries


def _array_declarations(
    arrays: Mapping[str, np.ndarray],
    table_contracts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    ownership: dict[str, tuple[str, str, str]] = {}
    for table, contract in table_contracts.items():
        for column in contract["columns"]:
            ownership[column["value_array"]] = (table, column["name"], "value")
            if "validity_array" in column:
                ownership[column["validity_array"]] = (
                    table,
                    column["name"],
                    "validity",
                )
    declarations = []
    for path in sorted(arrays):
        values = np.asarray(arrays[path])
        table, column, role = ownership[path]
        declarations.append(
            {
                "path": path,
                "table": table,
                "column": column,
                "array_role": role,
                "dtype": values.dtype.str,
                "shape": list(values.shape),
                "content_sha256": array_values_sha256(values),
            }
        )
    return declarations


@dataclass(frozen=True, slots=True)
class PreparedProviderChaserPositionSuite:
    """Typed row arrays plus bounded computation and source metadata."""

    recording_id: str
    analysis_zarr: Path
    arrays: Mapping[str, np.ndarray] = field(repr=False, compare=False)
    table_contracts: Mapping[str, Any] = field(repr=False)
    value_registries: Mapping[str, Any] = field(repr=False)
    suite_metadata: Mapping[str, Any] = field(repr=False)
    source_bindings: Mapping[str, Any] = field(repr=False)
    temporal_alignment: Mapping[str, Any] = field(repr=False)
    temporal_caveat: str
    report_sha256: str
    suite_sha256: str
    array_declarations: tuple[Mapping[str, Any], ...] = field(repr=False)

    @property
    def total_row_count(self) -> int:
        return sum(int(item["row_count"]) for item in self.table_contracts.values())


def prepare_provider_chaser_position_suite(
    report_value: Mapping[str, Any],
    *,
    expected_analysis_zarr: str | Path | None = None,
) -> PreparedProviderChaserPositionSuite:
    """Convert one fully validated computation report to typed table arrays."""

    report = _strict_json_object(report_value, field_name="position_suite_report")
    if (
        report.get("schema_id") != CANARY_SCHEMA_ID
        or report.get("schema_version") != CANARY_SCHEMA_VERSION
    ):
        _fail("Position-suite report has the wrong validated computation schema.")
    if (
        report.get("status") != "computed_read_only"
        or report.get("selector_eligible") is not False
        or report.get("selection") != "none"
        or report.get("production_authority") is not False
        or report.get("registry_update") is not False
    ):
        _fail("Position-suite report is not an explicit read-only computation.")
    archive = _canonical_archive(report.get("analysis_zarr"))
    if expected_analysis_zarr is not None and archive != _canonical_archive(
        expected_analysis_zarr
    ):
        _fail("Position-suite report targets a different analysis Zarr.")
    recording_id = report.get("recording_id")
    if type(recording_id) is not str or not recording_id:
        _fail("Position-suite report recording_id is invalid.")
    suite = _strict_json_object(report.get("suite"), field_name="report.suite")
    if (
        suite.get("schema_id") != SUITE_SCHEMA_ID
        or suite.get("schema_version") != SUITE_SCHEMA_VERSION
    ):
        _fail("Position-suite result has the wrong scientific schema.")
    if (
        suite.get("method_id") != METHOD_ID
        or suite.get("scientific_scope") != "position_only"
    ):
        _fail("Position-suite result has the wrong method or scientific scope.")
    if set(TABLE_COLUMNS) != _ROW_TABLE_FIELDS:
        _fail("Internal table contract does not cover every row table.")

    arrays: dict[str, np.ndarray] = {}
    contracts: dict[str, Any] = {}
    registries: dict[str, Any] = {}
    for table in TABLE_COLUMNS:
        table_arrays, columns, table_registries = _encode_table(table, suite.get(table))
        arrays.update(table_arrays)
        registries.update(table_registries)
        contracts[table] = {
            "row_count": len(suite[table]),
            "row_axis": "dense_zero_based_table_row_v1",
            "columns": columns,
        }
    if len(arrays) != len(set(arrays)) or len(registries) != len(set(registries)):
        _fail("Prepared table arrays or registries are not uniquely named.")
    declarations = _array_declarations(arrays, contracts)
    suite_metadata = {
        key: value for key, value in suite.items() if key not in _ROW_TABLE_FIELDS
    }
    source_bindings = _validate_source_bindings(report.get("source_bindings"))
    semantic_selection = source_bindings.get("protocol_semantic_selection")
    if isinstance(semantic_selection, Mapping) and suite.get(
        "epoch_roles"
    ) != semantic_selection.get("position_suite_epochs"):
        _fail(
            "Computed position-suite epochs differ from the exact protocol-"
            "semantic selection."
        )
    temporal_alignment = _strict_json_object(
        report.get("temporal_alignment"), field_name="report.temporal_alignment"
    )
    if (
        temporal_alignment.get("temporal_alignment_class")
        != "controller_input_provenance_proxy"
        or temporal_alignment.get("physical_presentation_verified") is not False
    ):
        _fail(
            "Position-suite temporal alignment must retain the explicit controller-"
            "input proxy limitation."
        )
    temporal_caveat = report.get("temporal_caveat")
    if type(temporal_caveat) is not str or not temporal_caveat:
        _fail("Position-suite temporal caveat is missing.")
    readonly = {path: _readonly_array(values) for path, values in arrays.items()}
    return PreparedProviderChaserPositionSuite(
        recording_id=recording_id,
        analysis_zarr=archive,
        arrays=MappingProxyType(readonly),
        table_contracts=MappingProxyType(_copy_json(contracts)),
        value_registries=MappingProxyType(_copy_json(registries)),
        suite_metadata=MappingProxyType(_copy_json(suite_metadata)),
        source_bindings=MappingProxyType(_copy_json(source_bindings)),
        temporal_alignment=MappingProxyType(_copy_json(temporal_alignment)),
        temporal_caveat=temporal_caveat,
        report_sha256=canonical_json_sha256(report),
        suite_sha256=canonical_json_sha256(suite),
        array_declarations=tuple(MappingProxyType(item) for item in declarations),
    )


def _publication_manifest(
    prepared: PreparedProviderChaserPositionSuite,
    *,
    run_name: str,
    run_path: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_id": STORAGE_SCHEMA_ID,
        "schema_version": STORAGE_SCHEMA_VERSION,
        "storage_schema": {
            "schema_id": STORAGE_SCHEMA_ID,
            "schema_version": STORAGE_SCHEMA_VERSION,
            "layout": STORAGE_LAYOUT,
        },
        "scientific_schema": {
            "schema_id": SUITE_SCHEMA_ID,
            "schema_version": SUITE_SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "scientific_scope": "position_only",
        },
        "run_name": run_name,
        "run_path": run_path,
        "analysis_zarr": str(prepared.analysis_zarr),
        "recording_id": prepared.recording_id,
        "status": "complete_selector_ineligible",
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
        "production_selector_activation": False,
        "registry_update": False,
        "publication_policy": PUBLICATION_POLICY,
        "row_evidence_storage": "typed_zarr_arrays_not_attributes_v1",
        "dimensions": {
            "frame_count": int(prepared.suite_metadata["frame_count"]),
            "chaser_count": int(prepared.suite_metadata["chaser_count"]),
            "total_table_row_count": prepared.total_row_count,
            "table_row_counts": {
                table: int(contract["row_count"])
                for table, contract in prepared.table_contracts.items()
            },
        },
        "table_contracts": _copy_json(prepared.table_contracts),
        "value_registries": _copy_json(prepared.value_registries),
        "array_declarations": _copy_json(prepared.array_declarations),
        "suite_metadata": _copy_json(prepared.suite_metadata),
        "source_bindings": _copy_json(prepared.source_bindings),
        "source_bindings_sha256": canonical_json_sha256(dict(prepared.source_bindings)),
        "temporal_alignment": _copy_json(prepared.temporal_alignment),
        "temporal_caveat": prepared.temporal_caveat,
        "computation_report_sha256": prepared.report_sha256,
        "computed_suite_sha256": prepared.suite_sha256,
    }
    manifest = {**payload, "payload_digest": canonical_json_sha256(payload)}
    encoded = json.dumps(manifest, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > MAX_MANIFEST_BYTES:
        _fail(
            f"Compact position-suite manifest is {len(encoded)} bytes; maximum is "
            f"{MAX_MANIFEST_BYTES}."
        )
    return manifest


@dataclass(frozen=True, slots=True)
class ProviderChaserPositionSuitePublicationPlan:
    analysis_zarr: Path
    run_name: str
    run_path: str
    prepared: PreparedProviderChaserPositionSuite = field(repr=False)
    manifest: Mapping[str, Any] = field(repr=False)
    run_provenance: Mapping[str, Any] = field(repr=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_id": PUBLICATION_SCHEMA_ID,
            "schema_version": PUBLICATION_SCHEMA_VERSION,
            "status": "dry_run_plan",
            "analysis_zarr": str(self.analysis_zarr),
            "recording_id": self.prepared.recording_id,
            "run_name": self.run_name,
            "run_path": self.run_path,
            "manifest_sha256": canonical_json_sha256(dict(self.manifest)),
            "manifest_bytes": len(
                json.dumps(dict(self.manifest), separators=(",", ":")).encode()
            ),
            "array_count": len(self.prepared.arrays),
            "table_row_counts": dict(self.manifest["dimensions"]["table_row_counts"]),
            "total_table_row_count": self.prepared.total_row_count,
            "selector_eligible": False,
            "selection": "none",
            "production_authority": False,
            "registry_update": False,
            "target_exists": (self.analysis_zarr / self.run_path).exists(),
            "source_validation": "exact_bound_sources_via_position_suite_computation",
            "upstream_dense_hash_recomputation": False,
        }


def build_provider_chaser_position_suite_publication_plan(
    analysis_zarr: str | Path,
    *,
    report: Mapping[str, Any],
    run_name: str,
    expected_recording_id: str | None = None,
) -> ProviderChaserPositionSuitePublicationPlan:
    archive = _canonical_archive(analysis_zarr)
    name = _run_name(run_name)
    run_path = _canonical_run_path(name)
    if (archive / run_path).exists():
        raise FileExistsError(f"Refusing to replace existing run: {archive / run_path}")
    prepared = prepare_provider_chaser_position_suite(
        report, expected_analysis_zarr=archive
    )
    if (
        expected_recording_id is not None
        and prepared.recording_id != expected_recording_id
    ):
        _fail("Position-suite recording_id differs from the requested recording.")
    _validate_protocol_semantic_source(
        archive,
        prepared.source_bindings,
        expected_recording_id=prepared.recording_id,
    )
    manifest = _publication_manifest(prepared, run_name=name, run_path=run_path)
    source_provider = prepared.source_bindings.get("provider_chaser_distance")
    if not isinstance(source_provider, Mapping):
        _fail("Exact provider chaser-distance source binding is missing.")
    input_run_ids = {
        "provider_chaser_distance": str(source_provider.get("run_path")),
        "source_manifest_sha256": str(source_provider.get("manifest_sha256")),
    }
    semantic_selection = prepared.source_bindings.get(
        "protocol_semantic_selection"
    )
    if isinstance(semantic_selection, Mapping):
        input_run_ids.update(
            {
                "protocol_semantic_selection": str(
                    semantic_selection.get("run_path")
                ),
                "protocol_semantic_selection_manifest_sha256": str(
                    semantic_selection.get("manifest_sha256")
                ),
            }
        )
    run_provenance = build_writer_run_provenance(
        command="fisheye.analysis_workflows.provider_chaser_position_suite_publication",
        params={
            "run_name": name,
            "run_path": run_path,
            "publication_policy": PUBLICATION_POLICY,
            "scientific_method_id": METHOD_ID,
            "computed_suite_sha256": prepared.suite_sha256,
        },
        input_run_ids=input_run_ids,
    )
    return ProviderChaserPositionSuitePublicationPlan(
        analysis_zarr=archive,
        run_name=name,
        run_path=run_path,
        prepared=prepared,
        manifest=MappingProxyType(_copy_json(manifest)),
        run_provenance=MappingProxyType(_copy_json(run_provenance)),
    )


def _manifest_from_run(run: Any) -> tuple[dict[str, Any], str]:
    manifest = _strict_json_object(
        run.attrs.get(MANIFEST_ATTR), field_name=MANIFEST_ATTR
    )
    stored = run.attrs.get(MANIFEST_DIGEST_ATTR)
    observed = canonical_json_sha256(manifest)
    if stored != observed:
        _fail("Persistent position-suite manifest digest is stale.")
    if len(json.dumps(manifest, separators=(",", ":")).encode()) > MAX_MANIFEST_BYTES:
        _fail("Persistent position-suite manifest exceeds its bounded size contract.")
    return manifest, observed


def _load_declared_arrays(
    run: Any,
    *,
    declarations: Sequence[Mapping[str, Any]],
    verify_content_hashes: bool,
) -> dict[str, np.ndarray]:
    expected_paths = [item.get("path") for item in declarations]
    if any(type(path) is not str or not path for path in expected_paths):
        _fail("Persistent array declaration path is invalid.")
    if len(set(expected_paths)) != len(expected_paths):
        _fail("Persistent array declarations contain duplicate paths.")
    actual_paths = sorted(str(path) for path in run.array_keys())
    if actual_paths != sorted(expected_paths):
        _fail("Persistent array paths differ from the immutable manifest.")
    arrays: dict[str, np.ndarray] = {}
    for declaration in declarations:
        path = str(declaration["path"])
        try:
            node = run[path]
            if not isinstance(node, zarr.Array):
                raise TypeError("not an array")
            values = np.asarray(node[...])
        except (KeyError, OSError, TypeError, ValueError) as exc:
            _fail(f"Unable to read persistent array {path!r}: {exc}")
        if values.dtype.str != declaration.get("dtype"):
            _fail(f"Persistent array {path!r} dtype differs from its declaration.")
        if list(values.shape) != declaration.get("shape"):
            _fail(f"Persistent array {path!r} shape differs from its declaration.")
        if verify_content_hashes and array_values_sha256(values) != declaration.get(
            "content_sha256"
        ):
            _fail(f"Persistent array {path!r} content digest differs.")
        values.setflags(write=False)
        arrays[path] = values
    return arrays


def _validate_table_arrays(
    manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> None:
    contracts = manifest.get("table_contracts")
    registries = manifest.get("value_registries")
    dimensions = manifest.get("dimensions")
    if not isinstance(contracts, Mapping) or set(contracts) != set(TABLE_COLUMNS):
        _fail("Persistent table contracts are incomplete or unknown.")
    if not isinstance(registries, Mapping) or not isinstance(dimensions, Mapping):
        _fail("Persistent value registries or dimensions are missing.")
    counts = dimensions.get("table_row_counts")
    if not isinstance(counts, Mapping) or set(counts) != set(TABLE_COLUMNS):
        _fail("Persistent table row counts are incomplete.")
    total = 0
    for table, expected_columns in TABLE_COLUMNS.items():
        contract = contracts[table]
        if not isinstance(contract, Mapping):
            _fail(f"Persistent {table} contract is malformed.")
        row_count = contract.get("row_count")
        if (
            type(row_count) is not int
            or row_count <= 0
            or counts.get(table) != row_count
        ):
            _fail(f"Persistent {table} row count is invalid.")
        total += row_count
        columns = contract.get("columns")
        if not isinstance(columns, list):
            _fail(f"Persistent {table} columns are missing.")
        expected_names = ["row_index", *(column.name for column in expected_columns)]
        if [
            item.get("name") for item in columns if isinstance(item, Mapping)
        ] != expected_names:
            _fail(f"Persistent {table} logical columns differ from the storage schema.")
        row_index = arrays.get(f"{table}__row_index")
        if row_index is None or not np.array_equal(
            row_index, np.arange(row_count, dtype=np.int64)
        ):
            _fail(f"Persistent {table} row index is not exact and dense.")
        for column in columns[1:]:
            value_path = column.get("value_array")
            values = arrays.get(value_path)
            if values is None or values.shape != (row_count,):
                _fail(f"Persistent {table}.{column.get('name')} values are misaligned.")
            logical = column.get("logical_type")
            if logical == "float64":
                valid = arrays.get(column.get("validity_array"))
                if (
                    valid is None
                    or valid.shape != (row_count,)
                    or valid.dtype != np.bool_
                ):
                    _fail(
                        f"Persistent {table}.{column.get('name')} validity is invalid."
                    )
                if np.any(~np.isfinite(values[valid])) or np.any(
                    ~np.isnan(values[~valid])
                ):
                    _fail(
                        f"Persistent {table}.{column.get('name')} null encoding is invalid."
                    )
            elif logical == "string":
                registry = registries.get(column.get("registry_id"))
                if not isinstance(registry, Mapping) or not isinstance(
                    registry.get("values"), list
                ):
                    _fail(
                        f"Persistent {table}.{column.get('name')} registry is missing."
                    )
                if np.any(values < 0) or np.any(values >= len(registry["values"])):
                    _fail(
                        f"Persistent {table}.{column.get('name')} code is out of range."
                    )
            elif logical != "int64":
                _fail(f"Persistent {table}.{column.get('name')} type is unsupported.")
    if dimensions.get("total_table_row_count") != total:
        _fail("Persistent total table row count is stale.")


def _validate_persistent_run(
    path: Path,
    *,
    expected_manifest: Mapping[str, Any] | None = None,
    expected_run_path: str | None = None,
    verify_content_hashes: bool = False,
    run: Any | None = None,
) -> dict[str, Any]:
    if run is None:
        run = open_zarr_root(path, mode="r", use_consolidated=False)
    manifest, manifest_sha256 = _manifest_from_run(run)
    if expected_manifest is not None and manifest != dict(expected_manifest):
        _fail("Persistent run manifest differs from the prepared publication.")
    if expected_run_path is not None and manifest.get("run_path") != expected_run_path:
        _fail("Persistent run path differs from its exact publication path.")
    if (
        manifest.get("schema_id") != STORAGE_SCHEMA_ID
        or manifest.get("schema_version") != STORAGE_SCHEMA_VERSION
    ):
        _fail("Persistent run has the wrong storage schema identity.")
    if manifest.get("storage_schema") != {
        "schema_id": STORAGE_SCHEMA_ID,
        "schema_version": STORAGE_SCHEMA_VERSION,
        "layout": STORAGE_LAYOUT,
    }:
        _fail("Persistent run storage schema binding is invalid.")
    if (
        manifest.get("selector_eligible") is not False
        or manifest.get("selection") != "none"
        or manifest.get("production_authority") is not False
        or manifest.get("production_selector_activation") is not False
        or manifest.get("registry_update") is not False
    ):
        _fail("Persistent run is not explicitly selector-ineligible.")
    if canonical_json_sha256(manifest.get("source_bindings")) != manifest.get(
        "source_bindings_sha256"
    ):
        _fail("Persistent source-binding digest is stale.")
    if run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        _fail("Persistent provider chaser position-suite run is not complete.")
    if run.attrs.get("stage_selector_eligible") is not False:
        _fail("Persistent provider chaser position-suite run is selector eligible.")
    if set(run.attrs).intersection(_FORBIDDEN_SELECTOR_ATTRS):
        _fail("Persistent run contains forbidden selector pointer attributes.")
    if not isinstance(run.attrs.get("run_provenance"), Mapping):
        _fail("Persistent run provenance is missing.")
    declarations = manifest.get("array_declarations")
    if not isinstance(declarations, list):
        _fail("Persistent run array declarations are missing.")
    arrays = _load_declared_arrays(
        run,
        declarations=declarations,
        verify_content_hashes=verify_content_hashes,
    )
    _validate_table_arrays(manifest, arrays)
    return {
        "valid": True,
        "manifest_sha256": manifest_sha256,
        "run_path": manifest["run_path"],
        "array_count": len(arrays),
        "total_table_row_count": manifest["dimensions"]["total_table_row_count"],
        "table_row_counts": dict(manifest["dimensions"]["table_row_counts"]),
        "arrays": arrays,
    }


def _compact_validation(validation: Mapping[str, Any]) -> dict[str, Any]:
    arrays = validation.get("arrays")
    if not isinstance(arrays, Mapping):
        _fail("Persistent validation did not return exact arrays.")
    array_paths = sorted(str(path) for path in arrays)
    return {
        "valid": validation.get("valid") is True,
        "manifest_sha256": validation.get("manifest_sha256"),
        "run_path": validation.get("run_path"),
        "array_count": validation.get("array_count"),
        "total_table_row_count": validation.get("total_table_row_count"),
        "table_row_counts": validation.get("table_row_counts"),
        "array_path_count": len(array_paths),
        "array_paths_sha256": canonical_json_sha256(array_paths),
        "readable_array_declarations": f"{MANIFEST_ATTR}.array_declarations",
        "row_evidence_storage": "typed_zarr_arrays_not_publication_metadata_v1",
    }


def _write_local_run(
    plan: ProviderChaserPositionSuitePublicationPlan, local_run_path: Path
) -> None:
    if local_run_path.exists():
        raise FileExistsError(
            f"Local publication path already exists: {local_run_path}"
        )
    local_run_path.parent.mkdir(parents=True, exist_ok=True)
    run = zarr.open_group(
        str(local_run_path), mode="w-", zarr_format=3, use_consolidated=False
    )
    mark_run_started(
        run, run_name=plan.run_name, stage="provider_chaser_position_suite"
    )
    run.attrs.update(
        {
            "schema_id": STORAGE_SCHEMA_ID,
            "schema_version": STORAGE_SCHEMA_VERSION,
            "stage_selector_eligible": False,
            "selection": "none",
            "production_authority": False,
            "production_selector_activation": False,
            "registry_update": False,
            "run_provenance": json_attr_safe(dict(plan.run_provenance)),
            MANIFEST_ATTR: json_attr_safe(dict(plan.manifest)),
            MANIFEST_DIGEST_ATTR: canonical_json_sha256(dict(plan.manifest)),
        }
    )
    for path, values in plan.prepared.arrays.items():
        array = np.asarray(values)
        run.create_array(
            path,
            data=array,
            chunks=(max(1, min(int(array.shape[0]), 16384)),),
        )
    mark_run_complete(
        run, run_name=plan.run_name, run_provenance=dict(plan.run_provenance)
    )
    _validate_persistent_run(
        local_run_path,
        expected_manifest=plan.manifest,
        expected_run_path=plan.run_path,
    )


def publish_provider_chaser_position_suite_run(
    plan: ProviderChaserPositionSuitePublicationPlan,
    *,
    scratch_root: str | Path | None = None,
    copy_backend: str = "python",
) -> dict[str, Any]:
    """Atomically publish one immutable selector-ineligible suite."""

    scratch_parent = (
        Path(scratch_root).expanduser().resolve() if scratch_root is not None else None
    )
    if scratch_parent is not None:
        scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{plan.run_name}.provider_chaser_position_suite.",
        dir=str(scratch_parent) if scratch_parent is not None else None,
    ) as temporary:
        local_run_path = Path(temporary) / "run.zarr"
        _write_local_run(plan, local_run_path)
        parent_snapshot: dict[str, Any] | None = None

        def validate(path: Path) -> Mapping[str, Any]:
            return _compact_validation(
                _validate_persistent_run(
                    path,
                    expected_manifest=plan.manifest,
                    expected_run_path=plan.run_path,
                )
            )

        def prepare_parents(root: Any) -> tuple[Any]:
            nonlocal parent_snapshot
            analysis = root.require_group("analysis")
            parent = require_runs_parent(
                analysis,
                "provider_chaser_position_suite_runs",
                completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
            )
            if set(parent.attrs).intersection(_FORBIDDEN_SELECTOR_ATTRS):
                _fail("Position-suite parent already contains selector pointers.")
            if parent_snapshot is None:
                parent_snapshot = dict(parent.attrs)
            return (parent,)

        def complete(_root: Any, parent: Any, run: Any) -> None:
            run.attrs["stage_selector_eligible"] = False
            run.attrs["selection"] = "none"
            mark_run_complete(
                run,
                parent_group=parent,
                run_name=plan.run_name,
                run_provenance=dict(plan.run_provenance),
            )

        def verify(root: Any) -> None:
            parent = root[RUNS_PARENT_PATH]
            if parent_snapshot is None or dict(parent.attrs) != parent_snapshot:
                _fail("Selector-ineligible publication changed parent metadata.")
            _validate_protocol_semantic_source(
                plan.analysis_zarr,
                plan.prepared.source_bindings,
                expected_recording_id=plan.prepared.recording_id,
            )
            _validate_persistent_run(
                plan.analysis_zarr / plan.run_path,
                expected_manifest=plan.manifest,
                expected_run_path=plan.run_path,
            )

        result = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=plan.analysis_zarr,
                local_run_path=local_run_path,
                target_run_path=plan.analysis_zarr / plan.run_path,
                run_name=plan.run_name,
                lock_suffix="provider_chaser_position_suite_publication",
                publish_schema_id=PUBLICATION_SCHEMA_ID,
                policy=PUBLICATION_POLICY,
                rollback_policy="retain_failed_selector_ineligible_tombstone_v1",
                content_checksum=True,
            ),
            copy_backend=copy_backend,
            validate_run=validate,
            prepare_parents=prepare_parents,
            complete_run=complete,
            verify_pointers=verify,
            payload_metadata={
                "recording_id": plan.prepared.recording_id,
                "run_path": plan.run_path,
                "computed_suite_sha256": plan.prepared.suite_sha256,
                "selector_activation": "none",
            },
        )
    consolidation = consolidate_metadata_capture_expected_warnings(plan.analysis_zarr)
    metadata_equivalence = validate_direct_consolidated_subtree(
        plan.analysis_zarr, subtree_path=plan.run_path
    ).to_json()
    final = load_provider_chaser_position_suite_source_handle(
        plan.analysis_zarr,
        run_name=plan.run_name,
        expected_recording_id=plan.prepared.recording_id,
        use_consolidated=True,
    )
    return {
        "status": "published_selector_ineligible",
        "run_path": plan.run_path,
        "manifest_sha256": final.manifest_sha256,
        "array_count": len(final.arrays),
        "table_row_counts": dict(final.manifest["dimensions"]["table_row_counts"]),
        "total_table_row_count": final.total_table_row_count,
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
        "registry_update": False,
        "consolidation": consolidation,
        "metadata_equivalence": metadata_equivalence,
        "upstream_dense_hash_recomputation": False,
        "atomic_publication": result,
    }


@dataclass(frozen=True, slots=True, init=False)
class ProviderChaserPositionSuiteSourceHandle:
    """Read-only bounded view of one exact persistent position suite."""

    analysis_zarr: Path
    run_path: str
    run_name: str
    recording_id: str
    manifest: Mapping[str, Any] = field(repr=False)
    arrays: Mapping[str, np.ndarray] = field(repr=False, compare=False)
    metadata_equivalence: Mapping[str, Any] = field(repr=False)
    verification_mode: str
    _use_consolidated: bool = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self, *, _seal: object | None = None, **values: Any) -> None:
        if _seal is not _SOURCE_HANDLE_SEAL:
            raise TypeError("Position-suite handles require their strict loader.")
        for name, value in values.items():
            if name in {"manifest", "metadata_equivalence"}:
                value = MappingProxyType(_copy_json(value))
            elif name == "arrays":
                value = MappingProxyType(
                    {key: _readonly_array(array) for key, array in value.items()}
                )
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _SOURCE_HANDLE_SEAL)

    @property
    def manifest_sha256(self) -> str:
        return canonical_json_sha256(dict(self.manifest))

    @property
    def total_table_row_count(self) -> int:
        return int(self.manifest["dimensions"]["total_table_row_count"])

    def array(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise KeyError(f"Unknown position-suite array {name!r}.") from exc

    def table_rows(self, table: str) -> tuple[Mapping[str, Any], ...]:
        """Decode one bounded summary table according to its exact registries."""

        if table not in TABLE_COLUMNS:
            raise KeyError(f"Unknown position-suite table {table!r}.")
        contract = self.manifest["table_contracts"][table]
        row_count = int(contract["row_count"])
        registries = self.manifest["value_registries"]
        rows = [dict() for _ in range(row_count)]
        for column in contract["columns"][1:]:
            values = self.arrays[column["value_array"]]
            logical = column["logical_type"]
            if logical == "string":
                labels = registries[column["registry_id"]]["values"]
                decoded = [labels[int(code)] for code in values]
            elif logical == "float64":
                valid = self.arrays[column["validity_array"]]
                decoded = [
                    float(value) if bool(ok) else None
                    for value, ok in zip(values, valid, strict=True)
                ]
            else:
                decoded = [int(value) for value in values]
            for row, value in zip(rows, decoded, strict=True):
                row[column["name"]] = value
        return tuple(MappingProxyType(row) for row in rows)

    def assert_current(self) -> None:
        refreshed = load_provider_chaser_position_suite_source_handle(
            self.analysis_zarr,
            run_name=self.run_name,
            expected_recording_id=self.recording_id,
            use_consolidated=self._use_consolidated,
        )
        if refreshed.manifest_sha256 != self.manifest_sha256:
            _fail("Published position-suite run changed after sealing.")


_SOURCE_HANDLE_SEAL = object()


def load_provider_chaser_position_suite_source_handle(
    analysis_zarr: str | Path,
    *,
    run_name: str,
    expected_recording_id: str | None = None,
    use_consolidated: bool = True,
    deep_audit: bool = False,
) -> ProviderChaserPositionSuiteSourceHandle:
    if type(use_consolidated) is not bool:
        _fail("use_consolidated must be one exact boolean.")
    archive = _canonical_archive(analysis_zarr)
    name = _run_name(run_name)
    run_path = _canonical_run_path(name)
    try:
        metadata_equivalence = validate_direct_consolidated_subtree(
            archive, subtree_path=run_path
        ).to_json()
        root = open_zarr_root(archive, mode="r", use_consolidated=use_consolidated)
        run = root[run_path]
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        _fail(f"Unable to open exact provider chaser position-suite run: {exc}")
    if not isinstance(run, zarr.Group):
        _fail("Exact provider chaser position-suite path is not a Zarr group.")
    manifest, _ = _manifest_from_run(run)
    if (
        expected_recording_id is not None
        and manifest.get("recording_id") != expected_recording_id
    ):
        _fail("Persistent run recording_id differs from the requested recording.")
    validation = _validate_persistent_run(
        archive / run_path,
        expected_manifest=manifest,
        expected_run_path=run_path,
        verify_content_hashes=deep_audit,
        run=run,
    )
    _validate_protocol_semantic_source(
        archive,
        manifest["source_bindings"],
        expected_recording_id=str(manifest["recording_id"]),
    )
    return ProviderChaserPositionSuiteSourceHandle(
        analysis_zarr=archive,
        run_path=run_path,
        run_name=name,
        recording_id=str(manifest["recording_id"]),
        manifest=manifest,
        arrays=validation["arrays"],
        metadata_equivalence=metadata_equivalence,
        verification_mode=("deep_audit" if deep_audit else "bounded_publication"),
        _use_consolidated=use_consolidated,
        _seal=_SOURCE_HANDLE_SEAL,
    )


def deep_audit_provider_chaser_position_suite_run(
    analysis_zarr: str | Path,
    *,
    run_name: str,
    expected_recording_id: str | None = None,
) -> ProviderChaserPositionSuiteSourceHandle:
    return load_provider_chaser_position_suite_source_handle(
        analysis_zarr,
        run_name=run_name,
        expected_recording_id=expected_recording_id,
        use_consolidated=True,
        deep_audit=True,
    )


__all__ = [
    "DEEP_AUDIT_POLICY",
    "MANIFEST_ATTR",
    "MANIFEST_DIGEST_ATTR",
    "PUBLICATION_POLICY",
    "PUBLICATION_SCHEMA_ID",
    "PUBLICATION_SCHEMA_VERSION",
    "RUNS_PARENT_PATH",
    "RUNS_PREFIX",
    "STORAGE_LAYOUT",
    "STORAGE_SCHEMA_ID",
    "STORAGE_SCHEMA_VERSION",
    "TABLE_COLUMNS",
    "PreparedProviderChaserPositionSuite",
    "ProviderChaserPositionSuitePublicationError",
    "ProviderChaserPositionSuitePublicationPlan",
    "ProviderChaserPositionSuiteSourceHandle",
    "build_provider_chaser_position_suite_publication_plan",
    "deep_audit_provider_chaser_position_suite_run",
    "load_provider_chaser_position_suite_source_handle",
    "prepare_provider_chaser_position_suite",
    "publish_provider_chaser_position_suite_run",
]
