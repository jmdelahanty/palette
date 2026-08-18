"""Publish selector-ineligible provider occupancy-v2 analysis runs.

The numerical kernel lives in :mod:`fisheye.analysis.provider_occupancy_v2`.
This module is deliberately a publication boundary: it accepts one already
computed result and seven exact, digest-bound source records, plans the local
run, validates it, and only then performs an atomic non-promoting publication
under ``analysis/provider_occupancy_runs``.

The run stores no object arrays.  Occurrence identities are encoded as one
UTF-8 byte vector plus int64 offsets, which keeps the logical identity stable
across Zarr readers and languages while allowing arbitrary immutable IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from types import MappingProxyType
from typing import Any, Mapping
import uuid

import numpy as np
import zarr

from fisheye.analysis.provider_occupancy_v2 import (
    OccupancyGrid,
    OccupancyTimingPolicy,
    ProviderOccupancySummary,
    ProviderOccupancyV2Result,
    build_provider_occupancy_config_digest,
)
from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr.analysis_array_contracts import (
    AnalysisArrayDeclaration,
    AnalysisAuthorityRole,
)
from fisheye.shared.zarr.analysis_storage_planning import (
    AnalysisArrayStorageFacts,
    AnalysisStoragePlanReceipt,
    analysis_storage_plan_receipt_from_manifest,
    plan_analysis_storage,
)
from fisheye.shared.zarr.array_contracts import (
    ArrayContract,
    FLOAT64,
    INT64,
    UINT8,
)
from fisheye.shared.zarr.array_factory import (
    create_array_from_plan,
    validate_array_metadata_declaration_from_plan,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1, StorageProfile
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_ATTR,
    COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_NAME_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)


PROVIDER_OCCUPANCY_PARENT_PATH = "analysis/provider_occupancy_runs"
PROVIDER_OCCUPANCY_SCHEMA_ID = "palette.provider_occupancy_v2_run"
PROVIDER_OCCUPANCY_SCHEMA_VERSION = 1
PROVIDER_OCCUPANCY_MANIFEST_SCHEMA_ID = (
    "palette.provider_occupancy_v2_run_manifest"
)
PROVIDER_OCCUPANCY_MANIFEST_SCHEMA_VERSION = 1
PROVIDER_OCCUPANCY_MANIFEST_ATTR = "provider_occupancy_v2_manifest"
PROVIDER_OCCUPANCY_MANIFEST_DIGEST_ATTR = (
    "provider_occupancy_v2_manifest_sha256"
)
PROVIDER_OCCUPANCY_STORAGE_PLAN_ATTR = "provider_occupancy_v2_storage_plan"
PROVIDER_OCCUPANCY_PUBLICATION_ATTEMPT_ATTR = (
    "provider_occupancy_v2_publication_attempt_uuid"
)
PROVIDER_OCCUPANCY_PUBLISH_POLICY = "provider_occupancy_v2_atomic_nonpromoting_v1"
PROVIDER_OCCUPANCY_RETRY_POLICY = "new_immutable_run_name_required"
PROVIDER_OCCUPANCY_ARRAY_SCHEMA_ID = "palette.provider_occupancy_v2_array"
PROVIDER_OCCUPANCY_ARRAY_SCHEMA_VERSION = 1
PROVIDER_OCCUPANCY_MATERIALIZATION_SCHEMA_ID = (
    "palette.provider_occupancy_v2_materialization"
)

SOURCE_BINDING_NAMES = (
    "trajectory",
    "compiled_selection",
    "provider",
    "timing",
    "geometry",
    "transform",
    "fixed_grid_policy",
)
_SELECTOR_ALIASES = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_provider",
        "latest_any",
        "latest_materialized",
        "latest_composite",
        "latest_pending",
        "authoritative_run",
        "authoritative_run_provenance",
        "active_run",
        "active",
        "current_run",
        "current",
        "default_run",
        "default",
        "selected_run",
        "selected",
        "publication_generation",
        "publication_policy",
    }
)
_SELECTOR_NAME_PREFIXES = (
    "latest_",
    "authoritative_",
    "active_",
    "current_",
    "default_",
    "selected_",
    "publication_",
)
_SELECTOR_ATTRS = _SELECTOR_ALIASES
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class ProviderOccupancyV2MaterializationError(ValueError):
    """Raised when an occupancy result cannot be published safely."""


def _safe_run_name(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or value in {".", ".."}
        or _selector_like_name(value)
        or _RUN_NAME_RE.fullmatch(value) is None
    ):
        raise ProviderOccupancyV2MaterializationError(
            "run_name must be one exact non-selector child name."
        )
    return value


def _selector_like_name(value: object) -> bool:
    if type(value) is not str:
        return False
    lowered = value.lower()
    return lowered in _SELECTOR_ALIASES or lowered.startswith(_SELECTOR_NAME_PREFIXES)


def _require_provenance_parent(root: Any) -> Any:
    parent = require_runs_parent(
        root.require_group("analysis"),
        "provider_occupancy_runs",
        completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    )
    if parent.attrs.get(COMPLETION_EPOCH_ATTR) != COMPLETION_EPOCH_REQUIRE_PROVENANCE:
        raise ProviderOccupancyV2MaterializationError(
            "occupancy parent must require completion provenance"
        )
    return parent


def _canonical_record(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ProviderOccupancyV2MaterializationError(
            f"{label} record must be one nonempty mapping."
        )
    try:
        encoded = json.dumps(
            json_attr_safe(dict(value)),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        result = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ProviderOccupancyV2MaterializationError(
            f"{label} record is not strict canonical JSON."
        ) from exc
    if not isinstance(result, dict):  # pragma: no cover - defensive
        raise ProviderOccupancyV2MaterializationError(
            f"{label} record did not canonicalize to an object."
        )
    return result


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ProviderOccupancyV2MaterializationError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _bind_record(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"record", "sha256"}:
        raise ProviderOccupancyV2MaterializationError(
            f"{label} binding must contain exactly record and sha256."
        )
    record = _canonical_record(value["record"], label=f"{label}.record")
    digest = _require_sha256(value["sha256"], label=f"{label}.sha256")
    expected = canonical_json_sha256(record)
    if digest != expected:
        raise ProviderOccupancyV2MaterializationError(
            f"{label} digest does not match its canonical record."
        )
    return {"record": record, "sha256": digest}


@dataclass(frozen=True)
class ProviderOccupancyV2SourceBindings:
    """The exact seven source records required by one occupancy run."""

    values: Mapping[str, Mapping[str, Any]]

    def __post_init__(self) -> None:
        if not isinstance(self.values, Mapping) or set(self.values) != set(
            SOURCE_BINDING_NAMES
        ):
            raise ProviderOccupancyV2MaterializationError(
                "source bindings must contain exactly the seven required authorities."
            )
        normalized = {
            name: _bind_record(self.values[name], label=name)
            for name in SOURCE_BINDING_NAMES
        }
        object.__setattr__(
            self,
            "values",
            MappingProxyType(normalized),
        )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Mapping[str, Any]],
    ) -> "ProviderOccupancyV2SourceBindings":
        return cls(values=values)

    def as_record(self) -> dict[str, Any]:
        return {
            name: {
                "record": dict(self.values[name]["record"]),
                "sha256": self.values[name]["sha256"],
            }
            for name in SOURCE_BINDING_NAMES
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.as_record())


def _normalize_bindings(
    value: ProviderOccupancyV2SourceBindings | Mapping[str, Mapping[str, Any]],
) -> ProviderOccupancyV2SourceBindings:
    if type(value) is ProviderOccupancyV2SourceBindings:
        return value
    if not isinstance(value, Mapping):
        raise TypeError("source_bindings must be a strict source-binding mapping.")
    return ProviderOccupancyV2SourceBindings.from_mapping(value)


def _array_digest(value: np.ndarray) -> str:
    return sha256_array(np.asarray(value))


def _summary_identity(summary: ProviderOccupancySummary) -> dict[str, Any]:
    return {
        "occurrence_id": summary.occurrence_id,
        "counts_sha256": _array_digest(summary.counts),
        "occupancy_fraction_sha256": _array_digest(summary.occupancy_fraction),
        "occupancy_time_by_bin_s_sha256": _array_digest(
            summary.occupancy_time_by_bin_s
        ),
        "expected_selected_frames": summary.expected_selected_frames,
        "provider_present_count": summary.provider_present_count,
        "provider_valid_count": summary.provider_valid_count,
        "transform_invalid_count": summary.transform_invalid_count,
        "nonfinite_count": summary.nonfinite_count,
        "out_of_grid_count": summary.out_of_grid_count,
        "valid_in_grid_sample_count": summary.valid_in_grid_sample_count,
        "occupancy_time_s": summary.occupancy_time_s,
    }


def _result_identity(result: ProviderOccupancyV2Result) -> str:
    result.validate_conservation()
    payload = {
        "schema_id": result.schema_id,
        "schema_version": result.schema_version,
        "config_digest": result.config_digest,
        "edge_policy_id": result.edge_policy_id,
        "timing_policy_id": result.timing_policy_id,
        "fps_hz": result.fps_hz,
        "x_edges_sha256": _array_digest(result.x_edges),
        "y_edges_sha256": _array_digest(result.y_edges),
        "per_occurrence": [
            _summary_identity(summary) for summary in result.per_occurrence
        ],
        "pooled": _summary_identity(result.pooled),
    }
    return canonical_json_sha256(payload)


def _validate_result(result: object) -> ProviderOccupancyV2Result:
    if type(result) is not ProviderOccupancyV2Result:
        raise TypeError("result must be one exact ProviderOccupancyV2Result.")
    try:
        result.validate_conservation()
    except (TypeError, ValueError) as exc:
        raise ProviderOccupancyV2MaterializationError(
            "Provider occupancy result conservation is invalid."
        ) from exc
    try:
        grid = OccupancyGrid(result.x_edges, result.y_edges)
        timing = OccupancyTimingPolicy(result.fps_hz, result.timing_policy_id)
        expected_config = build_provider_occupancy_config_digest(grid, timing)
    except (TypeError, ValueError) as exc:
        raise ProviderOccupancyV2MaterializationError(
            "Provider occupancy result configuration is invalid."
        ) from exc
    if result.config_digest != expected_config:
        raise ProviderOccupancyV2MaterializationError(
            "Provider occupancy result configuration digest is stale."
        )
    return result


def _validate_binding_identity(
    result: ProviderOccupancyV2Result,
    bindings: ProviderOccupancyV2SourceBindings,
) -> None:
    """Reject a source/config binding that does not describe this result."""

    timing = bindings.values["timing"]["record"]
    if "fps_hz" in timing and float(timing["fps_hz"]) != result.fps_hz:
        raise ProviderOccupancyV2MaterializationError(
            "Timing source binding does not match result fps_hz."
        )
    if (
        "timing_policy_id" in timing
        and timing["timing_policy_id"] != result.timing_policy_id
    ):
        raise ProviderOccupancyV2MaterializationError(
            "Timing source binding does not match result timing policy."
        )
    fixed = bindings.values["fixed_grid_policy"]["record"]
    for key in ("config_digest", "occupancy_config_digest"):
        if key in fixed and fixed[key] != result.config_digest:
            raise ProviderOccupancyV2MaterializationError(
                "Fixed-grid source binding does not match result config digest."
            )
    if "x_edges" in fixed and list(fixed["x_edges"]) != result.x_edges.tolist():
        raise ProviderOccupancyV2MaterializationError(
            "Fixed-grid source binding x edges differ from the result."
        )
    if "y_edges" in fixed and list(fixed["y_edges"]) != result.y_edges.tolist():
        raise ProviderOccupancyV2MaterializationError(
            "Fixed-grid source binding y edges differ from the result."
        )
    trajectory = bindings.values["trajectory"]["record"]
    if "occupancy_config_digest" in trajectory and trajectory[
        "occupancy_config_digest"
    ] != result.config_digest:
        raise ProviderOccupancyV2MaterializationError(
            "Trajectory source binding has a stale occupancy config identity."
        )


def _selector_snapshot(parent: Any | None) -> dict[str, Any]:
    if parent is None:
        return {}
    return {
        name: json_attr_safe(parent.attrs[name])
        for name in _SELECTOR_ATTRS
        if name in parent.attrs
    }


def _get_node(group: Any, path: str) -> Any:
    node = group
    for component in path.strip("/").split("/"):
        node = node[component]
    return node


def _occurrence_encoding(
    result: ProviderOccupancyV2Result,
) -> tuple[np.ndarray, np.ndarray]:
    values = [
        "" if summary.occurrence_id is None else summary.occurrence_id
        for summary in result.per_occurrence
    ]
    offsets = [0]
    encoded = bytearray()
    for value in values:
        raw = value.encode("utf-8")
        encoded.extend(raw)
        offsets.append(len(encoded))
    return (
        np.asarray(offsets, dtype=np.int64),
        np.frombuffer(bytes(encoded), dtype=np.uint8).copy(),
    )


def _coverage(summary: ProviderOccupancySummary) -> dict[str, np.ndarray]:
    return {
        "expected_selected_frames": np.asarray(
            [summary.expected_selected_frames], dtype=np.int64
        ),
        "provider_present_count": np.asarray(
            [summary.provider_present_count], dtype=np.int64
        ),
        "provider_valid_count": np.asarray(
            [summary.provider_valid_count], dtype=np.int64
        ),
        "nonfinite_count": np.asarray([summary.nonfinite_count], dtype=np.int64),
        "transform_invalid_count": np.asarray(
            [summary.transform_invalid_count], dtype=np.int64
        ),
        "out_of_grid_count": np.asarray(
            [summary.out_of_grid_count], dtype=np.int64
        ),
        "valid_in_grid_sample_count": np.asarray(
            [summary.valid_in_grid_sample_count], dtype=np.int64
        ),
    }


def _result_arrays(result: ProviderOccupancyV2Result) -> dict[str, np.ndarray]:
    offsets, utf8 = _occurrence_encoding(result)
    if result.per_occurrence:
        per_counts = np.stack(
            [summary.counts for summary in result.per_occurrence], axis=0
        )
        per_fraction = np.stack(
            [summary.occupancy_fraction for summary in result.per_occurrence], axis=0
        )
        per_time = np.stack(
            [summary.occupancy_time_by_bin_s for summary in result.per_occurrence],
            axis=0,
        )
        per_coverage = {
            key: np.asarray(
                [getattr(summary, key) for summary in result.per_occurrence],
                dtype=np.int64,
            )
            for key in (
                "expected_selected_frames",
                "provider_present_count",
                "provider_valid_count",
                "nonfinite_count",
                "transform_invalid_count",
                "out_of_grid_count",
                "valid_in_grid_sample_count",
            )
        }
    else:
        height, width = result.pooled.counts.shape
        per_counts = np.empty((0, height, width), dtype=np.int64)
        per_fraction = np.empty((0, height, width), dtype=np.float64)
        per_time = np.empty((0, height, width), dtype=np.float64)
        per_coverage = {
            key: np.empty((0,), dtype=np.int64)
            for key in (
                "expected_selected_frames",
                "provider_present_count",
                "provider_valid_count",
                "nonfinite_count",
                "transform_invalid_count",
                "out_of_grid_count",
                "valid_in_grid_sample_count",
            )
        }
    arrays: dict[str, np.ndarray] = {
        "grid/x_edges": np.asarray(result.x_edges, dtype=np.float64),
        "grid/y_edges": np.asarray(result.y_edges, dtype=np.float64),
        "occurrence_id_offsets": offsets,
        "occurrence_id_utf8": utf8,
        "per_occurrence/counts": np.asarray(per_counts, dtype=np.int64),
        "per_occurrence/occupancy_fraction": np.asarray(
            per_fraction, dtype=np.float64
        ),
        "per_occurrence/occupancy_time_s": np.asarray(per_time, dtype=np.float64),
        "pooled/counts": np.asarray(result.pooled.counts, dtype=np.int64),
        "pooled/occupancy_fraction": np.asarray(
            result.pooled.occupancy_fraction, dtype=np.float64
        ),
        "pooled/occupancy_time_s": np.asarray(
            result.pooled.occupancy_time_by_bin_s, dtype=np.float64
        ),
    }
    for key, value in per_coverage.items():
        arrays[f"per_occurrence/{key}"] = value
    for key, value in _coverage(result.pooled).items():
        arrays[f"pooled/{key}"] = value
    return {path: np.ascontiguousarray(value) for path, value in arrays.items()}


def _array_contracts() -> dict[str, ArrayContract]:
    schema = PROVIDER_OCCUPANCY_ARRAY_SCHEMA_ID
    version = PROVIDER_OCCUPANCY_ARRAY_SCHEMA_VERSION
    contracts: dict[str, ArrayContract] = {
        "grid/x_edges": ArrayContract(
            schema,
            version,
            FLOAT64,
            ("x_edge",),
            ("x_edge",),
            "fixed arena-mm occupancy grid x edges",
            units="mm",
            coordinate_space="arena_mm",
        ),
        "grid/y_edges": ArrayContract(
            schema,
            version,
            FLOAT64,
            ("y_edge",),
            ("y_edge",),
            "fixed arena-mm occupancy grid y edges",
            units="mm",
            coordinate_space="arena_mm",
        ),
        "occurrence_id_offsets": ArrayContract(
            schema,
            version,
            INT64,
            ("occurrence_plus_one",),
            ("occurrence_plus_one",),
            "UTF-8 occurrence identity byte offsets",
            units="byte_offset",
        ),
        "occurrence_id_utf8": ArrayContract(
            schema,
            version,
            UINT8,
            ("occurrence_utf8_byte",),
            ("occurrence_utf8_byte",),
            "UTF-8 occurrence identity bytes",
            units="byte",
        ),
        "per_occurrence/counts": ArrayContract(
            schema,
            version,
            INT64,
            ("occurrence", "y_bin", "x_bin"),
            ("occurrence", "y_bin", "x_bin"),
            "raw per-occurrence valid in-grid sample counts",
            units="sample_count",
            coordinate_space="arena_mm",
        ),
        "per_occurrence/occupancy_fraction": ArrayContract(
            schema,
            version,
            FLOAT64,
            ("occurrence", "y_bin", "x_bin"),
            ("occurrence", "y_bin", "x_bin"),
            "per-occurrence occupancy fractions",
            units="fraction",
            coordinate_space="arena_mm",
        ),
        "per_occurrence/occupancy_time_s": ArrayContract(
            schema,
            version,
            FLOAT64,
            ("occurrence", "y_bin", "x_bin"),
            ("occurrence", "y_bin", "x_bin"),
            "per-occurrence occupancy time by bin",
            units="s",
            coordinate_space="arena_mm",
        ),
        "pooled/counts": ArrayContract(
            schema,
            version,
            INT64,
            ("y_bin", "x_bin"),
            ("y_bin", "x_bin"),
            "pooled valid in-grid sample counts",
            units="sample_count",
            coordinate_space="arena_mm",
        ),
        "pooled/occupancy_fraction": ArrayContract(
            schema,
            version,
            FLOAT64,
            ("y_bin", "x_bin"),
            ("y_bin", "x_bin"),
            "pooled occupancy fractions",
            units="fraction",
            coordinate_space="arena_mm",
        ),
        "pooled/occupancy_time_s": ArrayContract(
            schema,
            version,
            FLOAT64,
            ("y_bin", "x_bin"),
            ("y_bin", "x_bin"),
            "pooled occupancy time by bin",
            units="s",
            coordinate_space="arena_mm",
        ),
    }
    coverage_names = (
        "expected_selected_frames",
        "provider_present_count",
        "provider_valid_count",
        "nonfinite_count",
        "transform_invalid_count",
        "out_of_grid_count",
        "valid_in_grid_sample_count",
    )
    for prefix in ("per_occurrence", "pooled"):
        for name in coverage_names:
            contracts[f"{prefix}/{name}"] = ArrayContract(
                schema,
                version,
                INT64,
                ("occurrence",) if prefix == "per_occurrence" else ("pooled_summary",),
                ("occurrence",)
                if prefix == "per_occurrence"
                else ("pooled_summary",),
                f"{prefix} occupancy coverage {name}",
                units="sample_count",
            )
    return contracts


def _declarations(
    arrays: Mapping[str, np.ndarray],
) -> tuple[AnalysisArrayDeclaration, ...]:
    contracts = _array_contracts()
    declarations: list[AnalysisArrayDeclaration] = []
    for path in sorted(arrays):
        role = (
            AnalysisAuthorityRole.LINEAGE_INDEX
            if path in {"occurrence_id_offsets", "occurrence_id_utf8"}
            else AnalysisAuthorityRole.QUALITY_DIAGNOSTIC
            if "/" in path and path.rsplit("/", 1)[1]
            in {
                "expected_selected_frames",
                "provider_present_count",
                "provider_valid_count",
                "nonfinite_count",
                "transform_invalid_count",
                "out_of_grid_count",
                "valid_in_grid_sample_count",
            }
            else AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY
        )
        declarations.append(
            AnalysisArrayDeclaration(
                path=path,
                contract=contracts[path],
                required=True,
                access_pattern=AccessPattern.EAGER,
                write_mode=WriteMode.IMMUTABLE,
                authority_role=role,
                fill_semantics=(
                    "zero_storage_fill_nan_is_logical_empty_fraction"
                    if contracts[path].dtype is FLOAT64
                    else "zero"
                ),
                null_semantics=(
                    "NaN_means_undefined_empty_selection"
                    if contracts[path].dtype is FLOAT64
                    else "not_applicable"
                ),
                physical_policy_owner=PROVIDER_OCCUPANCY_MATERIALIZATION_SCHEMA_ID,
                byte_planner_adopted=True,
            )
        )
    return tuple(declarations)


def _storage_receipt(
    arrays: Mapping[str, np.ndarray],
    profile: StorageProfile,
) -> AnalysisStoragePlanReceipt:
    facts = {
        path: AnalysisArrayStorageFacts(
            path=path,
            shape=tuple(int(value) for value in array.shape),
            dtype=array.dtype,
            access_unit_semantics="one complete occupancy array record",
        )
        for path, array in arrays.items()
    }
    return plan_analysis_storage(
        _declarations(arrays),
        facts,
        profile=profile,
    )


def _array_records(
    arrays: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    return [
        {
            "path": path,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": _array_digest(array),
        }
        for path, array in sorted(arrays.items())
    ]


def _array_content_digest(records: list[dict[str, Any]]) -> str:
    return canonical_json_sha256(records)


@dataclass(frozen=True)
class ProviderOccupancyV2RunPlan:
    source_zarr: Path
    scratch_root: Path
    local_zarr: Path
    run_name: str
    result: ProviderOccupancyV2Result
    source_bindings: ProviderOccupancyV2SourceBindings
    arrays: Mapping[str, np.ndarray]
    storage_profile: StorageProfile
    storage_receipt: AnalysisStoragePlanReceipt
    parent_selector_attrs: Mapping[str, Any]
    publication_attempt_uuid: str
    result_identity_sha256: str
    run_provenance: Mapping[str, Any]

    @property
    def run_path(self) -> str:
        return f"{PROVIDER_OCCUPANCY_PARENT_PATH}/{self.run_name}"

    @property
    def local_run_path(self) -> Path:
        return self.local_zarr.joinpath(*self.run_path.split("/"))

    @property
    def target_run_path(self) -> Path:
        return self.source_zarr.joinpath(*self.run_path.split("/"))

    @property
    def source_bindings_sha256(self) -> str:
        return self.source_bindings.sha256

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_id": PROVIDER_OCCUPANCY_MATERIALIZATION_SCHEMA_ID,
            "schema_version": 1,
            "source_zarr": str(self.source_zarr),
            "scratch_root": str(self.scratch_root),
            "local_zarr": str(self.local_zarr),
            "run_name": self.run_name,
            "run_path": self.run_path,
            "result_identity_sha256": self.result_identity_sha256,
            "source_bindings_sha256": self.source_bindings_sha256,
            "storage_plan_digest": self.storage_receipt.as_manifest()[
                "payload_digest"
            ],
            "publication_policy": PROVIDER_OCCUPANCY_PUBLISH_POLICY,
            "retry_policy": PROVIDER_OCCUPANCY_RETRY_POLICY,
            "stage_selector_eligible": False,
            "parent_selector_attrs": dict(self.parent_selector_attrs),
        }


def _parent_for_plan(root: Any) -> Any | None:
    try:
        return _get_node(root, PROVIDER_OCCUPANCY_PARENT_PATH)
    except (KeyError, ValueError):
        return None


def plan_provider_occupancy_v2_run(
    source_zarr: str | Path,
    result: ProviderOccupancyV2Result,
    source_bindings: ProviderOccupancyV2SourceBindings
    | Mapping[str, Mapping[str, Any]],
    *,
    scratch_root: str | Path,
    run_name: str,
    storage_profile: StorageProfile = PUBLISHED_HTTP_V1,
    publication_attempt_uuid: str | None = None,
    _allow_existing_target: bool = False,
) -> ProviderOccupancyV2RunPlan:
    """Build a read-only publication plan for one exact result."""

    result = _validate_result(result)
    bindings = _normalize_bindings(source_bindings)
    _validate_binding_identity(result, bindings)
    source = Path(source_zarr).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    name = _safe_run_name(run_name)
    if not source.is_dir():
        raise FileNotFoundError(f"Analysis Zarr does not exist: {source}")
    if scratch == source or scratch.is_relative_to(source):
        raise ProviderOccupancyV2MaterializationError(
            "scratch_root must be outside the authoritative analysis Zarr."
        )
    if not isinstance(storage_profile, StorageProfile):
        raise TypeError("storage_profile must be one StorageProfile.")
    attempt = str(uuid.UUID(publication_attempt_uuid)) if publication_attempt_uuid else str(uuid.uuid4())
    local = scratch / f"{name}.zarr"
    target = source.joinpath(*f"{PROVIDER_OCCUPANCY_PARENT_PATH}/{name}".split("/"))
    if local.exists():
        raise FileExistsError(f"Local occupancy attempt already exists: {local}")
    if target.exists() and not _allow_existing_target:
        raise FileExistsError(f"Refusing existing occupancy run: {target}")
    root = open_zarr_root(source, mode="r", use_consolidated=False)
    arrays = _result_arrays(result)
    receipt = _storage_receipt(arrays, storage_profile)
    provenance = build_writer_run_provenance(
        command="provider_occupancy_v2_materializer",
        params={
            "run_name": name,
            "result_identity_sha256": _result_identity(result),
            "source_bindings_sha256": bindings.sha256,
            "storage_profile_id": storage_profile.profile_id,
        },
        input_run_ids={
            "trajectory": bindings.values["trajectory"]["sha256"],
            "compiled_selection": bindings.values["compiled_selection"]["sha256"],
            "provider": bindings.values["provider"]["sha256"],
        },
        cwd=Path(__file__).resolve().parents[4],
        include_system_context=False,
    )
    return ProviderOccupancyV2RunPlan(
        source_zarr=source,
        scratch_root=scratch,
        local_zarr=local,
        run_name=name,
        result=result,
        source_bindings=bindings,
        arrays=MappingProxyType(arrays),
        storage_profile=storage_profile,
        storage_receipt=receipt,
        parent_selector_attrs=MappingProxyType(
            _selector_snapshot(_parent_for_plan(root))
        ),
        publication_attempt_uuid=attempt,
        result_identity_sha256=_result_identity(result),
        run_provenance=MappingProxyType(json.loads(json.dumps(provenance))),
    )


def _manifest(
    plan: ProviderOccupancyV2RunPlan,
    *,
    status: str,
) -> dict[str, Any]:
    if status != RUN_STATUS_COMPLETE:
        raise ProviderOccupancyV2MaterializationError(
            "Only complete occupancy manifests may be published."
        )
    records = _array_records(plan.arrays)
    payload = {
        "namespace": PROVIDER_OCCUPANCY_PARENT_PATH,
        "row_axis": "provider_occupancy_v2",
        "run_name": plan.run_name,
        "run_path": plan.run_path,
        "status": status,
        "stage_selector_eligible": False,
        "result": {
            "schema_id": plan.result.schema_id,
            "schema_version": plan.result.schema_version,
            "config_digest": plan.result.config_digest,
            "edge_policy_id": plan.result.edge_policy_id,
            "timing_policy_id": plan.result.timing_policy_id,
            "fps_hz": plan.result.fps_hz,
            "result_identity_sha256": plan.result_identity_sha256,
        },
        "source_bindings": plan.source_bindings.as_record(),
        "source_bindings_sha256": plan.source_bindings_sha256,
        "logical_array_declarations": [
            declaration.as_manifest()
            for declaration in _declarations(plan.arrays)
        ],
        "arrays": records,
        "array_content_sha256": _array_content_digest(records),
        "physical_storage_plan": plan.storage_receipt.as_manifest(),
        "conservation": {
            "per_occurrence_count_sums": [
                int(summary.counts.sum(dtype=np.int64))
                for summary in plan.result.per_occurrence
            ],
            "per_occurrence_valid_in_grid_sample_counts": [
                summary.valid_in_grid_sample_count
                for summary in plan.result.per_occurrence
            ],
            "pooled_count_sum": int(plan.result.pooled.counts.sum(dtype=np.int64)),
            "pooled_valid_in_grid_sample_count": (
                plan.result.pooled.valid_in_grid_sample_count
            ),
            "empty_fraction_policy": "nan_when_no_valid_in_grid_samples_v1",
        },
        "publication": {
            "policy_id": PROVIDER_OCCUPANCY_PUBLISH_POLICY,
            "retry_policy": PROVIDER_OCCUPANCY_RETRY_POLICY,
            "publication_attempt_uuid": plan.publication_attempt_uuid,
            "selector_activation": "forbidden",
            "parent_selector_mutation": "forbidden",
        },
    }
    return {
        "schema_id": PROVIDER_OCCUPANCY_MANIFEST_SCHEMA_ID,
        "schema_version": PROVIDER_OCCUPANCY_MANIFEST_SCHEMA_VERSION,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }


def provider_occupancy_v2_manifest_digest(manifest: Mapping[str, Any]) -> str:
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        raise ProviderOccupancyV2MaterializationError(
            "Occupancy manifest envelope is not exact."
        )
    if (
        manifest["schema_id"] != PROVIDER_OCCUPANCY_MANIFEST_SCHEMA_ID
        or manifest["schema_version"] != PROVIDER_OCCUPANCY_MANIFEST_SCHEMA_VERSION
        or not isinstance(manifest["payload"], Mapping)
        or manifest["payload_digest"]
        != canonical_json_sha256(manifest["payload"])
    ):
        raise ProviderOccupancyV2MaterializationError(
            "Occupancy manifest identity or digest is invalid."
        )
    return str(manifest["payload_digest"])


def _fill_value(path: str) -> Any:
    return np.float64(0.0) if _array_contracts()[path].dtype is FLOAT64 else np.int64(0)


def _write_arrays(run: Any, plan: ProviderOccupancyV2RunPlan) -> None:
    entries = {entry.declaration.path: entry for entry in plan.storage_receipt.entries}
    for path, values in sorted(plan.arrays.items()):
        parent_path, _, leaf = path.rpartition("/")
        parent = run.require_group(parent_path) if parent_path else run
        entry = entries[path]
        array = create_array_from_plan(
            parent,
            name=leaf or path,
            contract=entry.declaration.contract,
            plan=entry.plan,
            fill_value=_fill_value(path),
            attributes={
                "units": entry.declaration.contract.units,
                "coordinate_space": entry.declaration.contract.coordinate_space,
                "authority_role": entry.declaration.authority_role.value,
            },
        )
        if values.size:
            array[...] = values


def _read_direct_declaration(path: Path) -> dict[str, Any]:
    metadata = path / "zarr.json"
    if not metadata.is_file():
        raise FileNotFoundError(f"Missing direct Zarr metadata: {metadata}")
    value = json.loads(metadata.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProviderOccupancyV2MaterializationError(
            f"Direct Zarr metadata is not an object: {metadata}"
        )
    return value


def _node_paths(run_dir: Path) -> tuple[set[str], set[str]]:
    groups: set[str] = set()
    arrays: set[str] = set()
    for metadata in sorted(run_dir.rglob("zarr.json")):
        declaration = _read_direct_declaration(metadata.parent)
        relative = metadata.parent.relative_to(run_dir).as_posix()
        path = "" if relative == "." else relative
        if declaration.get("node_type") == "group":
            groups.add(path)
        elif declaration.get("node_type") == "array":
            arrays.add(path)
        else:
            raise ProviderOccupancyV2MaterializationError(
                f"Unknown Zarr node type at {metadata}."
            )
    return groups, arrays


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.array_equal(left, right, equal_nan=True)
    )


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    plan: ProviderOccupancyV2RunPlan,
) -> tuple[Mapping[str, Any], AnalysisStoragePlanReceipt]:
    provider_occupancy_v2_manifest_digest(manifest)
    payload = manifest["payload"]
    expected_fields = {
        "namespace",
        "row_axis",
        "run_name",
        "run_path",
        "status",
        "stage_selector_eligible",
        "result",
        "source_bindings",
        "source_bindings_sha256",
        "logical_array_declarations",
        "arrays",
        "array_content_sha256",
        "physical_storage_plan",
        "conservation",
        "publication",
    }
    if set(payload) != expected_fields:
        raise ProviderOccupancyV2MaterializationError(
            "Occupancy manifest payload field set is not exact."
        )
    if (
        payload["namespace"] != PROVIDER_OCCUPANCY_PARENT_PATH
        or payload["row_axis"] != "provider_occupancy_v2"
        or payload["run_name"] != plan.run_name
        or payload["run_path"] != plan.run_path
        or payload["status"] != RUN_STATUS_COMPLETE
        or payload["stage_selector_eligible"] is not False
    ):
        raise ProviderOccupancyV2MaterializationError(
            "Occupancy manifest namespace or lifecycle is invalid."
        )
    result_payload = payload["result"]
    if not isinstance(result_payload, Mapping) or result_payload != {
        "schema_id": plan.result.schema_id,
        "schema_version": plan.result.schema_version,
        "config_digest": plan.result.config_digest,
        "edge_policy_id": plan.result.edge_policy_id,
        "timing_policy_id": plan.result.timing_policy_id,
        "fps_hz": plan.result.fps_hz,
        "result_identity_sha256": plan.result_identity_sha256,
    }:
        raise ProviderOccupancyV2MaterializationError(
            "Occupancy manifest result identity is stale."
        )
    bindings = ProviderOccupancyV2SourceBindings.from_mapping(
        payload["source_bindings"]
    )
    if (
        payload["source_bindings_sha256"] != bindings.sha256
        or bindings.sha256 != plan.source_bindings_sha256
    ):
        raise ProviderOccupancyV2MaterializationError(
            "Occupancy manifest source bindings are stale."
        )
    _validate_binding_identity(plan.result, bindings)
    records = payload["arrays"]
    expected_records = _array_records(plan.arrays)
    if records != expected_records:
        raise ProviderOccupancyV2MaterializationError(
            "Occupancy manifest array content declarations are stale."
        )
    if payload["array_content_sha256"] != _array_content_digest(expected_records):
        raise ProviderOccupancyV2MaterializationError(
            "Occupancy manifest array-content digest is stale."
        )
    declarations = payload["logical_array_declarations"]
    expected_declarations = [
        declaration.as_manifest() for declaration in _declarations(plan.arrays)
    ]
    if declarations != expected_declarations:
        raise ProviderOccupancyV2MaterializationError(
            "Occupancy logical array declarations are stale."
        )
    receipt = analysis_storage_plan_receipt_from_manifest(
        payload["physical_storage_plan"]
    )
    if [entry.declaration.path for entry in receipt.entries] != sorted(plan.arrays):
        raise ProviderOccupancyV2MaterializationError(
            "Occupancy manifest arrays differ from the storage plan."
        )
    if receipt.as_manifest() != plan.storage_receipt.as_manifest():
        raise ProviderOccupancyV2MaterializationError(
            "Occupancy manifest storage plan differs from the plan."
        )
    if payload["conservation"] != _manifest(
        plan, status=RUN_STATUS_COMPLETE
    )["payload"]["conservation"]:
        raise ProviderOccupancyV2MaterializationError(
            "Occupancy manifest conservation evidence is stale."
        )
    publication = payload["publication"]
    if not isinstance(publication, Mapping) or set(publication) != {
        "policy_id",
        "retry_policy",
        "publication_attempt_uuid",
        "selector_activation",
        "parent_selector_mutation",
    }:
        raise ProviderOccupancyV2MaterializationError(
            "Occupancy publication policy is not exact."
        )
    if (
        publication["policy_id"] != PROVIDER_OCCUPANCY_PUBLISH_POLICY
        or publication["retry_policy"] != PROVIDER_OCCUPANCY_RETRY_POLICY
        or publication["selector_activation"] != "forbidden"
        or publication["parent_selector_mutation"] != "forbidden"
    ):
        raise ProviderOccupancyV2MaterializationError(
            "Occupancy publication policy is invalid."
        )
    try:
        uuid.UUID(str(publication["publication_attempt_uuid"]))
    except (TypeError, ValueError) as exc:
        raise ProviderOccupancyV2MaterializationError(
            "Occupancy publication attempt UUID is invalid."
        ) from exc
    return payload, receipt


def _validate_run_group(
    run: Any,
    run_dir: Path,
    *,
    plan: ProviderOccupancyV2RunPlan,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    if (
        run.attrs.get("schema_id") != PROVIDER_OCCUPANCY_SCHEMA_ID
        or run.attrs.get("schema_version") != PROVIDER_OCCUPANCY_SCHEMA_VERSION
    ):
        errors.append("run schema identity is stale")
    try:
        manifest = run.attrs.get(PROVIDER_OCCUPANCY_MANIFEST_ATTR)
        payload, receipt = _validate_manifest(manifest, plan=plan)
        manifest_digest = provider_occupancy_v2_manifest_digest(manifest)
    except Exception as exc:
        raise ProviderOccupancyV2MaterializationError(
            f"Invalid provider occupancy manifest: {exc}"
        ) from exc
    if expected_manifest_sha256 is not None and manifest_digest != expected_manifest_sha256:
        errors.append("manifest digest differs from plan")
    if run.attrs.get(PROVIDER_OCCUPANCY_MANIFEST_DIGEST_ATTR) != manifest_digest:
        errors.append("manifest attribute digest is stale")
    if run.attrs.get(PROVIDER_OCCUPANCY_STORAGE_PLAN_ATTR) != receipt.as_manifest():
        errors.append("storage plan attribute is stale")
    if run.attrs.get(PROVIDER_OCCUPANCY_PUBLICATION_ATTEMPT_ATTR) != plan.publication_attempt_uuid:
        errors.append("publication attempt identity is stale")
    if not isinstance(run.attrs.get("run_provenance"), Mapping):
        errors.append("run provenance is missing")
    if run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        errors.append("run is not complete")
    if run.attrs.get(RUN_NAME_ATTR) != plan.run_name:
        errors.append("run name is stale")
    if run.attrs.get("stage_selector_eligible") is not False:
        errors.append("run is selector eligible")
    if any(_selector_like_name(name) for name in run.attrs):
        errors.append("run contains selector attributes")
    groups, array_paths = _node_paths(run_dir)
    expected_array_paths = set(plan.arrays)
    expected_group_paths = {"", "grid", "per_occurrence", "pooled"}
    if array_paths != expected_array_paths:
        errors.append("run array paths differ from the exact manifest")
    if groups != expected_group_paths:
        errors.append("run group paths differ from the exact manifest")
    entries = {entry.declaration.path: entry for entry in receipt.entries}
    for path, expected in plan.arrays.items():
        try:
            node = _get_node(run, path)
            observed = np.asarray(node[:])
            if not _arrays_equal(expected, observed):
                errors.append(f"array {path} decoded content differs")
            if _array_digest(observed) != _array_digest(expected):
                errors.append(f"array {path} content digest differs")
            entry = entries[path]
            declaration = _read_direct_declaration(run_dir.joinpath(*path.split("/")))
            metadata_errors = validate_array_metadata_declaration_from_plan(
                declaration,
                contract=entry.declaration.contract,
                plan=entry.plan,
                fill_value=_fill_value(path),
            )
            if metadata_errors:
                errors.append(f"array {path} metadata differs: {metadata_errors!r}")
        except Exception as exc:
            errors.append(f"array {path} is unreadable: {exc}")
    try:
        _validate_result(plan.result)
        if _result_identity(plan.result) != plan.result_identity_sha256:
            errors.append("planned result identity changed")
        _validate_binding_identity(plan.result, plan.source_bindings)
    except Exception as exc:
        errors.append(f"source/config identity changed: {exc}")
    if errors:
        raise ProviderOccupancyV2MaterializationError(
            "Invalid provider occupancy run: " + "; ".join(errors)
        )
    return {
        "valid": True,
        "run_path": plan.run_path,
        "manifest_sha256": manifest_digest,
        "array_content_sha256": payload["array_content_sha256"],
        "occurrence_count": len(plan.result.per_occurrence),
        "grid_shape": list(plan.result.pooled.counts.shape),
    }


def validate_provider_occupancy_v2_run(
    analysis_zarr: str | Path,
    run_path: str,
    *,
    result: ProviderOccupancyV2Result,
    source_bindings: ProviderOccupancyV2SourceBindings
    | Mapping[str, Mapping[str, Any]],
    use_consolidated: bool = True,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate one published run against its exact in-memory sources."""

    archive = Path(analysis_zarr).expanduser().resolve()
    expected_prefix = f"{PROVIDER_OCCUPANCY_PARENT_PATH}/"
    if (
        not run_path.startswith(expected_prefix)
        or "/" in run_path[len(expected_prefix) :]
    ):
        raise ProviderOccupancyV2MaterializationError(
            "run_path must name one exact provider occupancy run."
        )
    existing_root = open_zarr_root(
        archive,
        mode="r",
        use_consolidated=use_consolidated,
    )
    existing_run = existing_root[run_path]
    existing_attempt = existing_run.attrs.get(
        PROVIDER_OCCUPANCY_PUBLICATION_ATTEMPT_ATTR
    )
    if type(existing_attempt) is not str:
        raise ProviderOccupancyV2MaterializationError(
            "Published occupancy run lacks its publication attempt identity."
        )
    plan = plan_provider_occupancy_v2_run(
        analysis_zarr,
        result,
        source_bindings,
        scratch_root=Path(analysis_zarr).parent / ".occupancy-validation-scratch",
        run_name=run_path.rsplit("/", 1)[-1],
        _allow_existing_target=True,
        publication_attempt_uuid=existing_attempt,
    )
    if run_path != plan.run_path:
        raise ProviderOccupancyV2MaterializationError(
            "run_path is not under the provider occupancy parent."
        )
    root = existing_root
    return _validate_run_group(
        root[run_path],
        archive.joinpath(*run_path.split("/")),
        plan=plan,
        expected_manifest_sha256=expected_manifest_sha256,
    )


def _materialize_local(plan: ProviderOccupancyV2RunPlan) -> dict[str, Any]:
    if plan.local_zarr.exists():
        raise FileExistsError(f"Local occupancy Zarr already exists: {plan.local_zarr}")
    plan.scratch_root.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(
        str(plan.local_zarr), mode="w-", zarr_format=3, use_consolidated=False
    )
    parent = require_runs_parent(
        root.require_group("analysis"),
        "provider_occupancy_runs",
        completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    )
    run = parent.create_group(plan.run_name)
    mark_run_started(run, run_name=plan.run_name, stage=PROVIDER_OCCUPANCY_SCHEMA_ID)
    run.attrs.update(
        {
            "schema_id": PROVIDER_OCCUPANCY_SCHEMA_ID,
            "schema_version": PROVIDER_OCCUPANCY_SCHEMA_VERSION,
            "stage_selector_eligible": False,
            "run_provenance": json_attr_safe(dict(plan.run_provenance)),
            PROVIDER_OCCUPANCY_PUBLICATION_ATTEMPT_ATTR: plan.publication_attempt_uuid,
        }
    )
    _write_arrays(run, plan)
    complete_manifest = _manifest(plan, status=RUN_STATUS_COMPLETE)
    run.attrs[PROVIDER_OCCUPANCY_MANIFEST_ATTR] = json_attr_safe(complete_manifest)
    run.attrs[PROVIDER_OCCUPANCY_MANIFEST_DIGEST_ATTR] = provider_occupancy_v2_manifest_digest(
        complete_manifest
    )
    run.attrs[PROVIDER_OCCUPANCY_STORAGE_PLAN_ATTR] = json_attr_safe(
        plan.storage_receipt.as_manifest()
    )
    run.attrs["stage_selector_eligible"] = False
    # Validate the complete payload and its manifest while the run is still
    # private and incomplete.  Completion is the final lifecycle write.
    _validate_manifest(complete_manifest, plan=plan)
    mark_run_complete(
        run,
        parent_group=parent,
        run_name=plan.run_name,
        run_provenance=dict(plan.run_provenance),
    )
    if (
        run.attrs.get(PROVIDER_OCCUPANCY_MANIFEST_ATTR) != complete_manifest
        or run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
    ):
        raise ProviderOccupancyV2MaterializationError(
            "Complete occupancy run lost its exact manifest or completion marker."
        )
    consolidate_metadata_capture_expected_warnings(plan.local_zarr)
    local_direct_consolidated = validate_direct_consolidated_subtree(
        plan.local_zarr,
        subtree_path=plan.run_path,
    )
    validation = _validate_run_group(
        open_zarr_root(plan.local_zarr, mode="r", use_consolidated=True)[plan.run_path],
        plan.local_run_path,
        plan=plan,
    )
    return {
        "local_zarr": str(plan.local_zarr),
        "validation": validation,
        "direct_consolidated": local_direct_consolidated.to_json(),
    }


def publish_provider_occupancy_v2_run(
    plan: ProviderOccupancyV2RunPlan,
    *,
    copy_backend: str = "python",
    keep_scratch: bool = True,
) -> dict[str, Any]:
    """Publish one complete occupancy run without changing selectors."""

    try:
        current_result_identity = _result_identity(plan.result)
    except Exception as exc:
        raise ProviderOccupancyV2MaterializationError(
            "Occupancy result was mutated or became invalid after planning."
        ) from exc
    if current_result_identity != plan.result_identity_sha256:
        raise ProviderOccupancyV2MaterializationError(
            "Occupancy result was mutated after planning."
        )
    _validate_result(plan.result)
    _validate_binding_identity(plan.result, plan.source_bindings)
    if plan.target_run_path.exists():
        raise FileExistsError(f"Refusing existing occupancy target: {plan.target_run_path}")
    local = _materialize_local(plan)
    acceptance: dict[str, Any] = {}

    def validate(path: Path) -> Mapping[str, Any]:
        return _validate_run_group(
            open_zarr_root(path, mode="r", use_consolidated=False),
            path,
            plan=plan,
            expected_manifest_sha256=provider_occupancy_v2_manifest_digest(
                _manifest(plan, status=RUN_STATUS_COMPLETE)
            ),
        )

    def prepare(root: Any) -> tuple[Any]:
        return (
            _require_provenance_parent(root),
        )

    def complete(_root: Any, parent: Any, run: Any) -> None:
        _validate_result(plan.result)
        _validate_binding_identity(plan.result, plan.source_bindings)
        if _result_identity(plan.result) != plan.result_identity_sha256:
            raise ProviderOccupancyV2MaterializationError(
                "Occupancy result changed before completion."
            )
        mark_run_complete(
            run,
            parent_group=parent,
            run_name=plan.run_name,
            run_provenance=run.attrs.get("run_provenance"),
        )
        run.attrs["stage_selector_eligible"] = False

    def verify(root: Any) -> None:
        parent = _get_node(root, PROVIDER_OCCUPANCY_PARENT_PATH)
        if _selector_snapshot(parent) != dict(plan.parent_selector_attrs):
            raise ProviderOccupancyV2MaterializationError(
                "Provider occupancy publication changed parent selectors."
            )
        _validate_run_group(
            parent[plan.run_name],
            plan.target_run_path,
            plan=plan,
            expected_manifest_sha256=provider_occupancy_v2_manifest_digest(
                _manifest(plan, status=RUN_STATUS_COMPLETE)
            ),
        )

    def finalize(_root: Any, _parent: Any, _run: Any) -> None:
        _validate_result(plan.result)
        _validate_binding_identity(plan.result, plan.source_bindings)
        if _result_identity(plan.result) != plan.result_identity_sha256:
            raise ProviderOccupancyV2MaterializationError(
                "Occupancy result changed before final publication."
            )
        consolidate_metadata_capture_expected_warnings(plan.source_zarr)
        metadata = validate_direct_consolidated_subtree(
            plan.source_zarr,
            subtree_path=plan.run_path,
        )
        published = validate_provider_occupancy_v2_run(
            plan.source_zarr,
            plan.run_path,
            result=plan.result,
            source_bindings=plan.source_bindings,
            use_consolidated=True,
            expected_manifest_sha256=provider_occupancy_v2_manifest_digest(
                _manifest(plan, status=RUN_STATUS_COMPLETE)
            ),
        )
        acceptance.update(
            direct_consolidated=metadata.to_json(),
            consolidated_validation=published,
        )

    publication = atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=plan.source_zarr,
            local_run_path=plan.local_run_path,
            target_run_path=plan.target_run_path,
            run_name=plan.run_name,
            lock_suffix="provider-occupancy-v2",
            publish_schema_id=PROVIDER_OCCUPANCY_SCHEMA_ID,
            policy=PROVIDER_OCCUPANCY_PUBLISH_POLICY,
            rollback_policy="retain_failed_tombstone_leave_parent_selectors_untouched",
            content_checksum=True,
            publication_attempt_uuid=plan.publication_attempt_uuid,
        ),
        copy_backend=copy_backend,
        validate_run=validate,
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=verify,
        activate_run=finalize,
        repair_failed_publication_visibility=lambda _path: consolidate_metadata_capture_expected_warnings(
            plan.source_zarr
        ),
        accept_persisted_activation_on_callback_error=False,
        payload_metadata={
            "result_identity_sha256": plan.result_identity_sha256,
            "source_bindings_sha256": plan.source_bindings_sha256,
            "array_content_sha256": _array_content_digest(_array_records(plan.arrays)),
            "selector_ineligible": True,
        },
    )
    output = {
        "status": "complete",
        "plan": plan.as_dict(),
        "local": local,
        "publication": publication,
        "acceptance": acceptance,
        "run_path": plan.run_path,
    }
    if not keep_scratch and plan.local_zarr.exists():
        shutil.rmtree(plan.local_zarr)
    return json_attr_safe(output)


def materialize_provider_occupancy_v2(
    source_zarr: str | Path,
    result: ProviderOccupancyV2Result,
    source_bindings: ProviderOccupancyV2SourceBindings
    | Mapping[str, Mapping[str, Any]],
    *,
    scratch_root: str | Path,
    run_name: str,
    storage_profile: StorageProfile = PUBLISHED_HTTP_V1,
    publication_attempt_uuid: str | None = None,
    copy_backend: str = "python",
    apply: bool = False,
    keep_scratch: bool = True,
) -> dict[str, Any]:
    """Plan or publish one selector-ineligible provider occupancy run."""

    plan = plan_provider_occupancy_v2_run(
        source_zarr,
        result,
        source_bindings,
        scratch_root=scratch_root,
        run_name=run_name,
        storage_profile=storage_profile,
        publication_attempt_uuid=publication_attempt_uuid,
    )
    output: dict[str, Any] = {"status": "planned", "plan": plan.as_dict()}
    if not apply:
        return output
    output.update(
        publish_provider_occupancy_v2_run(
            plan,
            copy_backend=copy_backend,
            keep_scratch=keep_scratch,
        )
    )
    return output


build_provider_occupancy_v2_plan = plan_provider_occupancy_v2_run
publish_selector_ineligible_provider_occupancy_v2 = (
    publish_provider_occupancy_v2_run
)


__all__ = [
    "PROVIDER_OCCUPANCY_ARRAY_SCHEMA_ID",
    "PROVIDER_OCCUPANCY_ARRAY_SCHEMA_VERSION",
    "PROVIDER_OCCUPANCY_MANIFEST_ATTR",
    "PROVIDER_OCCUPANCY_MANIFEST_DIGEST_ATTR",
    "PROVIDER_OCCUPANCY_MANIFEST_SCHEMA_ID",
    "PROVIDER_OCCUPANCY_MANIFEST_SCHEMA_VERSION",
    "PROVIDER_OCCUPANCY_PARENT_PATH",
    "PROVIDER_OCCUPANCY_PUBLISH_POLICY",
    "PROVIDER_OCCUPANCY_RETRY_POLICY",
    "PROVIDER_OCCUPANCY_SCHEMA_ID",
    "PROVIDER_OCCUPANCY_SCHEMA_VERSION",
    "ProviderOccupancyV2MaterializationError",
    "ProviderOccupancyV2RunPlan",
    "ProviderOccupancyV2SourceBindings",
    "build_provider_occupancy_v2_plan",
    "materialize_provider_occupancy_v2",
    "plan_provider_occupancy_v2_run",
    "provider_occupancy_v2_manifest_digest",
    "publish_provider_occupancy_v2_run",
    "publish_selector_ineligible_provider_occupancy_v2",
    "validate_provider_occupancy_v2_run",
]
