"""Publish strict contrasts over published provider-occupancy-v2 runs.

The source adapter in this module intentionally understands the maintained
occupancy-v2 publication contract.  It does not accept a parallel summary
manifest or a top-level ``occupancy_fraction`` array.  A contrast is currently
defined only for the explicitly named ``pooled`` source scope; per-occurrence
scope is rejected until its identity/index policy is frozen.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Any
import uuid

import numpy as np
import zarr

from fisheye.analysis.provider_occupancy_contrast import CONTRAST_FIELD
from fisheye.analysis_workflows.materializers.provider_occupancy_v2 import (
    PROVIDER_OCCUPANCY_MANIFEST_ATTR,
    PROVIDER_OCCUPANCY_MANIFEST_DIGEST_ATTR,
    PROVIDER_OCCUPANCY_MANIFEST_SCHEMA_ID,
    PROVIDER_OCCUPANCY_MANIFEST_SCHEMA_VERSION,
    PROVIDER_OCCUPANCY_PARENT_PATH,
    PROVIDER_OCCUPANCY_SCHEMA_ID,
    PROVIDER_OCCUPANCY_SCHEMA_VERSION,
    provider_occupancy_v2_manifest_digest,
)
from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_ATTR,
    COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)


MATERIALIZATION_SCHEMA_ID = "palette.provider_occupancy_contrast_materialization"
MATERIALIZATION_SCHEMA_VERSION = 1
PUBLISH_SCHEMA_ID = "palette.provider_occupancy_contrast_publish"
CONTRAST_RUN_SCHEMA_ID = "palette.provider_occupancy_contrast_run_manifest"
CONTRAST_RUN_SCHEMA_VERSION = 1
PARENT_PATH = "analysis/provider_occupancy_contrast_runs"
SOURCE_PARENT_PATH = PROVIDER_OCCUPANCY_PARENT_PATH
SOURCE_MANIFEST_ATTR = PROVIDER_OCCUPANCY_MANIFEST_ATTR
SOURCE_MANIFEST_SHA256_ATTR = PROVIDER_OCCUPANCY_MANIFEST_DIGEST_ATTR
SOURCE_SCOPE_POOLED = "pooled"
MANIFEST_ATTR = "provider_occupancy_contrast_manifest"
MANIFEST_SHA256_ATTR = "provider_occupancy_contrast_manifest_sha256"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
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
_SOURCE_BINDING_NAMES = (
    "trajectory",
    "compiled_selection",
    "provider",
    "timing",
    "geometry",
    "transform",
    "fixed_grid_policy",
)
_SOURCE_MANIFEST_PAYLOAD_FIELDS = {
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
_COMPATIBILITY_FIELDS = (
    "schema_family",
    "schema_version",
    "provider_id",
    "estimator",
    "position_track_policy",
    "coordinate_frame",
    "transform",
    "geometry",
    "sample_unit",
    "denominator",
    "normalization",
    "recording_id",
    "subject_id",
    "timing_authority",
    "grid_policy",
    "edge_policy",
)


class ProviderOccupancyContrastMaterializationError(ValueError):
    """Raised when a source or contrast cannot be proven compatible."""


def _fail(field: str, message: str) -> ProviderOccupancyContrastMaterializationError:
    return ProviderOccupancyContrastMaterializationError(f"{field}: {message}")


def _digest(value: Any, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise _fail(field, "must be one lowercase SHA-256 digest")
    return value


def _text(value: Any, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise _fail(field, "must be one nonempty canonical string")
    return value


def _json_copy(value: Any, *, field: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise _fail(field, "must be strict JSON") from exc


def _safe_run_name(value: str) -> str:
    if (
        type(value) is not str
        or _RUN_NAME_RE.fullmatch(value) is None
        or _selector_like_name(value)
    ):
        raise _fail(
            "run_name",
            "must be one safe immutable name, not a selector alias or prefix",
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
        "provider_occupancy_contrast_runs",
        completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    )
    if parent.attrs.get(COMPLETION_EPOCH_ATTR) != COMPLETION_EPOCH_REQUIRE_PROVENANCE:
        raise ProviderOccupancyContrastMaterializationError(
            "contrast parent must require completion provenance"
        )
    return parent


def _canonical_source_path(value: str) -> str:
    path = _text(value, field="source_run_path").strip("/")
    expected = SOURCE_PARENT_PATH.split("/")
    parts = path.split("/")
    if len(parts) != len(expected) + 1 or parts[: len(expected)] != expected:
        raise _fail(
            "source_run_path",
            f"must be one exact {SOURCE_PARENT_PATH}/<run> path",
        )
    _safe_run_name(parts[-1])
    return "/".join(parts)


def _selector_snapshot(parent: Any | None) -> dict[str, Any]:
    if parent is None:
        return {}
    return {
        name: json_attr_safe(parent.attrs[name])
        for name in _SELECTOR_ATTRS
        if name in parent.attrs
    }


def _binding(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"record", "sha256"}:
        raise _fail(f"source_bindings.{name}", "must contain exactly record and sha256")
    record = _json_copy(value["record"], field=f"source_bindings.{name}.record")
    digest = _digest(value["sha256"], field=f"source_bindings.{name}.sha256")
    if canonical_json_sha256(record) != digest:
        raise _fail(f"source_bindings.{name}", "record digest is stale")
    return {"record": record, "sha256": digest}


def _binding_identity(bindings: Mapping[str, Mapping[str, Any]], name: str) -> dict[str, Any]:
    value = bindings[name]
    payload = value["record"]
    return {
        "id": f"provider_occupancy_v2_binding:{name}",
        "identity": f"provider_occupancy_v2_binding:{name}",
        "sha256": value["sha256"],
        "payload": payload,
    }


def _subrecord_identity(
    bindings: Mapping[str, Mapping[str, Any]],
    *,
    binding_name: str,
    subrecord_name: str,
    required_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """Convert one explicit stable policy subrecord into a pure identity.

    The outer trajectory/transform records identify a particular selection or
    run.  They are therefore not suitable as the cross-arm position policy or
    coordinate-frame identity.  Stable subrecords must be supplied by the
    producer binding and are digest-bound by the occupancy-v2 source manifest.
    """

    record = bindings[binding_name]["record"]
    value = record.get(subrecord_name)
    if not isinstance(value, Mapping) or not value:
        raise _fail(
            f"source_bindings.{binding_name}.{subrecord_name}",
            "one explicit stable policy subrecord is required",
        )
    payload = _json_copy(
        value,
        field=f"source_bindings.{binding_name}.{subrecord_name}",
    )
    if not isinstance(payload, dict):  # pragma: no cover - defensive
        raise _fail(
            f"source_bindings.{binding_name}.{subrecord_name}",
            "must remain one mapping",
        )
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise _fail(
            f"source_bindings.{binding_name}.{subrecord_name}",
            f"is missing required keys: {missing!r}",
        )
    identifier = next(
        (
            payload.get(key)
            for key in (
                "policy_id",
                "sample_unit_id",
                "frame_id",
                "coordinate_frame_id",
                "id",
            )
            if key in payload
        ),
        None,
    )
    identifier = _text(
        identifier,
        field=f"source_bindings.{binding_name}.{subrecord_name}.id",
    )
    if "source_rows_sha256" in payload:
        _digest(
            payload["source_rows_sha256"],
            field=f"source_bindings.{binding_name}.{subrecord_name}.source_rows_sha256",
        )
    if subrecord_name == "position_track_policy":
        trajectory = bindings["trajectory"]["record"]
        trajectory_manifest = trajectory.get("trajectory_run_manifest")
        trajectory_source_rows_sha256 = trajectory.get("source_rows_sha256")
        if trajectory_source_rows_sha256 is None and isinstance(
            trajectory_manifest, Mapping
        ):
            trajectory_source_rows_sha256 = trajectory_manifest.get(
                "source_rows_sha256"
            )
        provider_id = _required_scalar(bindings, key="provider_id", preferred="provider")
        recording_id = _required_scalar(bindings, key="recording_id")
        if payload.get("provider_id") != provider_id or payload.get("recording_id") != recording_id:
            raise _fail(
                f"source_bindings.trajectory.{subrecord_name}",
                "provider_id/recording_id do not bind the exact trajectory source",
            )
        if trajectory_source_rows_sha256 != payload.get("source_rows_sha256"):
            raise _fail(
                f"source_bindings.trajectory.{subrecord_name}",
                "source_rows_sha256 does not bind the trajectory record",
            )
    digest = canonical_json_sha256(payload)
    return {
        "id": identifier,
        "identity": identifier,
        "sha256": digest,
        "payload": payload,
    }


def _coordinate_frame_identity(
    bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve one explicit coordinate-frame subrecord without guessing."""

    candidates = (
        ("transform", "coordinate_frame"),
        ("geometry", "coordinate_frame"),
        ("trajectory", "coordinate_frame"),
    )
    present = [
        (binding_name, subrecord_name)
        for binding_name, subrecord_name in candidates
        if subrecord_name in bindings[binding_name]["record"]
    ]
    if not present:
        raise _fail(
            "source_bindings.coordinate_frame",
            "an explicit coordinate-frame subrecord is required",
        )
    identities = [
        _subrecord_identity(
            bindings,
            binding_name=binding_name,
            subrecord_name=subrecord_name,
        )
        for binding_name, subrecord_name in present
    ]
    if any(identity != identities[0] for identity in identities[1:]):
        raise _fail(
            "source_bindings.coordinate_frame",
            "explicit coordinate-frame subrecords disagree",
        )
    return identities[0]


def _required_scalar(
    bindings: Mapping[str, Mapping[str, Any]],
    *,
    key: str,
    preferred: str | None = None,
) -> str:
    names = (preferred,) if preferred is not None else _SOURCE_BINDING_NAMES
    values = [
        bindings[name]["record"][key]
        for name in names
        if key in bindings[name]["record"]
    ]
    if not values or any(type(value) is not str or not value for value in values):
        raise _fail(f"source_bindings.{key}", "one exact string identity is required")
    if any(value != values[0] for value in values[1:]):
        raise _fail(f"source_bindings.{key}", "binding records disagree")
    return values[0]


def _decode_occurrences(run: Any, *, run_path: str, arrays: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    offsets = np.asarray(run["occurrence_id_offsets"][:], dtype=np.int64)
    utf8 = np.asarray(run["occurrence_id_utf8"][:], dtype=np.uint8)
    if offsets.ndim != 1 or offsets.size == 0 or offsets[0] != 0:
        raise _fail("source_occurrences", "occurrence offsets are invalid")
    if offsets[-1] != utf8.size or np.any(np.diff(offsets) < 0):
        raise _fail("source_occurrences", "occurrence offsets do not bound UTF-8 bytes")
    source_digest = arrays["occurrence_id_utf8"]["sha256"]
    values: list[dict[str, Any]] = []
    for index in range(offsets.size - 1):
        try:
            occurrence_id = bytes(utf8[offsets[index] : offsets[index + 1]]).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _fail("source_occurrences", "occurrence UTF-8 is invalid") from exc
        if not occurrence_id:
            raise _fail("source_occurrences", "occurrence identity is empty")
        payload = {
            "run_path": run_path,
            "array_sha256": source_digest,
            "index": index,
            "occurrence_id": occurrence_id,
        }
        values.append(
            {
                "id": occurrence_id,
                "identity": occurrence_id,
                "sha256": canonical_json_sha256(payload),
                "payload": payload,
            }
        )
    if not values:
        raise _fail("source_occurrences", "pooled scope requires at least one occurrence")
    return values


def _array_records_from_manifest(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records = payload.get("arrays")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise _fail("source_manifest.arrays", "must be one nonempty declaration sequence")
    result: dict[str, dict[str, Any]] = {}
    for value in records:
        if not isinstance(value, Mapping) or set(value) != {"path", "dtype", "shape", "sha256"}:
            raise _fail("source_manifest.arrays", "array declarations are not exact")
        path = _text(value["path"], field="source_manifest.arrays.path").strip("/")
        if path in result:
            raise _fail("source_manifest.arrays", "duplicate array path")
        dtype = _text(value["dtype"], field=f"source_manifest.arrays.{path}.dtype")
        shape = value["shape"]
        if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
            raise _fail(f"source_manifest.arrays.{path}.shape", "must be a nonnegative integer list")
        result[path] = {
            "path": path,
            "dtype": dtype,
            "shape": list(shape),
            "sha256": _digest(value["sha256"], field=f"source_manifest.arrays.{path}.sha256"),
        }
    if payload.get("array_content_sha256") != canonical_json_sha256([result[path] for path in sorted(result)]):
        raise _fail("source_manifest.array_content_sha256", "is stale")
    logical = payload.get("logical_array_declarations")
    if not isinstance(logical, Sequence) or isinstance(logical, (str, bytes)):
        raise _fail("source_manifest.logical_array_declarations", "must be a declaration sequence")
    logical_paths = [item.get("path") for item in logical if isinstance(item, Mapping)]
    if sorted(logical_paths) != sorted(result):
        raise _fail("source_manifest.logical_array_declarations", "does not match arrays")
    return result


def _validate_source_manifest(
    archive: Path,
    *,
    arm: str,
    run_path: str,
    expected_manifest_digest: str,
    source_scope: str,
) -> dict[str, Any]:
    if source_scope != SOURCE_SCOPE_POOLED:
        raise _fail("source_scope", "only explicit pooled scope is implemented")
    canonical_path = _canonical_source_path(run_path)
    _digest(expected_manifest_digest, field=f"{arm}_manifest_digest")
    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    run = root[canonical_path]
    if run.attrs.get("schema_id") != PROVIDER_OCCUPANCY_SCHEMA_ID or run.attrs.get("schema_version") != PROVIDER_OCCUPANCY_SCHEMA_VERSION:
        raise _fail(f"{arm}.source_run", "is not an occupancy-v2 run")
    if run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE or run.attrs.get("stage_selector_eligible") is not False:
        raise _fail(f"{arm}.source_run", "must be complete and selector-ineligible")
    manifest = run.attrs.get(SOURCE_MANIFEST_ATTR)
    if not isinstance(manifest, Mapping):
        raise _fail(f"{arm}.source_manifest", "is missing")
    manifest = _json_copy(manifest, field=f"{arm}.source_manifest")
    if set(manifest) != {"schema_id", "schema_version", "payload", "payload_digest"}:
        raise _fail(f"{arm}.source_manifest", "does not use the occupancy-v2 envelope")
    try:
        manifest_digest = provider_occupancy_v2_manifest_digest(manifest)
    except Exception as exc:
        raise _fail(f"{arm}.source_manifest", "envelope or payload digest is invalid") from exc
    if manifest_digest != expected_manifest_digest or run.attrs.get(SOURCE_MANIFEST_SHA256_ATTR) != manifest_digest:
        raise _fail(f"{arm}.source_manifest", "manifest digest is stale or mismatched")
    payload = manifest["payload"]
    if set(payload) != _SOURCE_MANIFEST_PAYLOAD_FIELDS:
        raise _fail(f"{arm}.source_manifest.payload", "does not match occupancy-v2 fields")
    if payload["namespace"] != SOURCE_PARENT_PATH or payload["row_axis"] != "provider_occupancy_v2" or payload["run_path"] != canonical_path or payload["status"] != RUN_STATUS_COMPLETE or payload["stage_selector_eligible"] is not False:
        raise _fail(f"{arm}.source_manifest.payload", "namespace or lifecycle is stale")
    bindings_raw = payload["source_bindings"]
    if not isinstance(bindings_raw, Mapping) or set(bindings_raw) != set(_SOURCE_BINDING_NAMES):
        raise _fail(f"{arm}.source_bindings", "must contain the exact seven authorities")
    bindings = {name: _binding(bindings_raw[name], name=name) for name in _SOURCE_BINDING_NAMES}
    if payload["source_bindings_sha256"] != canonical_json_sha256(bindings):
        raise _fail(f"{arm}.source_bindings_sha256", "is stale")
    arrays = _array_records_from_manifest(payload)
    direct_consolidated = validate_direct_consolidated_subtree(archive, subtree_path=canonical_path).to_json()
    for path, declaration in arrays.items():
        try:
            values = np.asarray(run[path][:])
        except Exception as exc:
            raise _fail(f"{arm}.source_array.{path}", "is unreadable") from exc
        if values.dtype.str != declaration["dtype"] or list(values.shape) != declaration["shape"] or sha256_array(values) != declaration["sha256"]:
            raise _fail(f"{arm}.source_array.{path}", "does not match its declared bytes")
    required_arrays = {
        "grid/x_edges",
        "grid/y_edges",
        "occurrence_id_offsets",
        "occurrence_id_utf8",
        "pooled/counts",
        "pooled/occupancy_fraction",
        "pooled/valid_in_grid_sample_count",
    }
    if not required_arrays.issubset(arrays):
        raise _fail(f"{arm}.source_arrays", "pooled scope arrays are incomplete")
    result_payload = payload["result"]
    if not isinstance(result_payload, Mapping):
        raise _fail(f"{arm}.source_manifest.result", "is missing")
    x_edges = np.asarray(run["grid/x_edges"][:], dtype=np.float64)
    y_edges = np.asarray(run["grid/y_edges"][:], dtype=np.float64)
    fraction = np.asarray(run["pooled/occupancy_fraction"][:], dtype=np.float64)
    count = np.asarray(run["pooled/valid_in_grid_sample_count"][:], dtype=np.int64)
    if count.shape != (1,) or int(count[0]) <= 0:
        raise _fail(f"{arm}.pooled.valid_in_grid_sample_count", "must be one positive scalar")
    if fraction.ndim != 2 or fraction.shape != (y_edges.size - 1, x_edges.size - 1) or not np.isfinite(fraction).all() or np.any(fraction < 0) or np.any(fraction > 1):
        raise _fail(f"{arm}.pooled.occupancy_fraction", "is not a finite valid pooled fraction")
    if not np.isfinite(x_edges).all() or not np.isfinite(y_edges).all() or not np.all(np.diff(x_edges) > 0) or not np.all(np.diff(y_edges) > 0):
        raise _fail(f"{arm}.grid", "edges are invalid")
    fixed_grid_record = bindings["fixed_grid_policy"]["record"]
    grid_id = fixed_grid_record.get("grid_id", fixed_grid_record.get("immutable_id"))
    grid_id = _text(grid_id, field=f"{arm}.grid.id")
    grid_payload = {"id": grid_id, "x_edges": x_edges.tolist(), "y_edges": y_edges.tolist()}
    grid = {**grid_payload, "sha256": canonical_json_sha256(grid_payload)}
    # The occupancy-v2 attribute digest is the payload digest.  The pure
    # identity contract below deliberately binds the complete envelope, so its
    # digest must be the digest of that complete envelope rather than the
    # payload digest stored in the source attribute.
    source_manifest_envelope_digest = canonical_json_sha256(manifest)
    source_manifest_identity = {
        "run_id": canonical_path,
        "identity": canonical_path,
        "sha256": source_manifest_envelope_digest,
        "payload": manifest,
    }
    source_selections = [_binding_identity(bindings, "compiled_selection")]
    source_occurrences = _decode_occurrences(run, run_path=canonical_path, arrays=arrays)
    provider_id = _required_scalar(bindings, key="provider_id", preferred="provider")
    recording_id = _required_scalar(bindings, key="recording_id")
    subject_id = _required_scalar(bindings, key="subject_id")
    position_track_policy = _subrecord_identity(
        bindings,
        binding_name="trajectory",
        subrecord_name="position_track_policy",
        required_keys=(
            "policy_id",
            "row_axis",
            "source_rows_sha256",
            "provider_id",
            "recording_id",
        ),
    )
    sample_unit = _subrecord_identity(
        bindings,
        binding_name="trajectory",
        subrecord_name="sample_unit",
    )
    coordinate_frame = _coordinate_frame_identity(bindings)
    compatibility = {
        "schema_family": result_payload["schema_id"],
        "schema_version": result_payload["schema_version"],
        "provider_id": provider_id,
        "estimator": _binding_identity(bindings, "provider"),
        "position_track_policy": position_track_policy,
        "coordinate_frame": coordinate_frame,
        "transform": _binding_identity(bindings, "transform"),
        "geometry": _binding_identity(bindings, "geometry"),
        "sample_unit": sample_unit,
        "denominator": {
            "policy_id": "provider_occupancy_v2_valid_in_grid_sample_denominator_v1",
            "schema_id": result_payload["schema_id"],
            "schema_version": result_payload["schema_version"],
            "timing_policy_id": result_payload["timing_policy_id"],
            "fraction_array": "pooled/occupancy_fraction",
            "count_array": "pooled/counts",
            "valid_sample_count_array": "pooled/valid_in_grid_sample_count",
        },
        "normalization": {
            "policy_id": "source_occupancy_fraction_unchanged_v1",
            "fraction_array": "pooled/occupancy_fraction",
        },
        "recording_id": recording_id,
        "subject_id": subject_id,
        "timing_authority": _binding_identity(bindings, "timing"),
        "grid_policy": _binding_identity(bindings, "fixed_grid_policy"),
        "edge_policy": result_payload["edge_policy_id"],
    }
    summary = {
        "arm_role": arm,
        "occupancy_fraction": fraction,
        "valid_sample_count": int(count[0]),
        "grid": grid,
        "source_manifest": source_manifest_identity,
        "bindings_sha256": payload["source_bindings_sha256"],
        "source_selections": source_selections,
        "source_occurrences": source_occurrences,
        **compatibility,
    }
    return {
        "run_path": canonical_path,
        "manifest": manifest,
        "manifest_sha256": manifest_digest,
        "manifest_envelope_sha256": source_manifest_envelope_digest,
        "payload": payload,
        "bindings": bindings,
        "bindings_sha256": payload["source_bindings_sha256"],
        "arrays": arrays,
        "summary": summary,
        "scope": source_scope,
        "direct_consolidated": direct_consolidated,
    }


def build_pooled_occupancy_contrast_summary(
    analysis_zarr: str | Path,
    *,
    run_path: str,
    manifest_sha256: str,
    arm_role: str,
    source_scope: str = SOURCE_SCOPE_POOLED,
) -> dict[str, Any]:
    """Derive a pure-contrast summary from one exact published occupancy-v2 run."""

    if arm_role not in {"baseline", "treatment"}:
        raise _fail("arm_role", "must be baseline or treatment")
    archive = Path(analysis_zarr).expanduser().resolve()
    return _validate_source_manifest(
        archive,
        arm=arm_role,
        run_path=run_path,
        expected_manifest_digest=manifest_sha256,
        source_scope=source_scope,
    )["summary"]


@dataclass(frozen=True)
class ProviderOccupancyContrastMaterializationPlan:
    source_zarr: Path
    run_name: str
    run_path: str
    scratch_root: Path
    local_zarr: Path
    local_run_path: Path
    baseline_run_path: str
    treatment_run_path: str
    baseline_manifest_digest: str
    treatment_manifest_digest: str
    source_scope: str
    contrast_result: Mapping[str, Any]
    source_evidence: Mapping[str, Mapping[str, Any]]
    arrays: Mapping[str, np.ndarray]
    manifest: Mapping[str, Any]
    parent_selector_attrs: Mapping[str, Any]
    run_provenance: Mapping[str, Any]


def _validate_contrast_result(result: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(result, Mapping) or not result:
        raise _fail("contrast_result", "must be one nonempty mapping")
    if result.get("operation") != "difference" or result.get("formula") != "treatment.occupancy_fraction - baseline.occupancy_fraction":
        raise _fail("contrast_result", "only the fixed treatment-minus-baseline difference is supported")
    if result.get("baseline_role") != "baseline" or result.get("treatment_role") != "treatment" or result.get("schema_version") != 1:
        raise _fail("contrast_result", "roles or schema version are invalid")
    policy = result.get("policy")
    if not isinstance(policy, Mapping) or canonical_json_sha256(policy) != result.get("policy_digest"):
        raise _fail("contrast_result.policy_digest", "policy digest is stale")
    if not isinstance(policy.get("config"), Mapping) or canonical_json_sha256(policy["config"]) != result.get("config_digest"):
        raise _fail("contrast_result.config_digest", "config digest is stale")
    if policy.get("operation") != "difference" or policy.get("formula") != result.get("formula"):
        raise _fail("contrast_result.policy", "does not bind the fixed difference")
    difference = np.asarray(result.get(CONTRAST_FIELD), dtype=np.float64)
    x_edges = np.asarray(result.get("x_edges"), dtype=np.float64)
    y_edges = np.asarray(result.get("y_edges"), dtype=np.float64)
    if difference.ndim != 2 or difference.size == 0 or not np.isfinite(difference).all():
        raise _fail(CONTRAST_FIELD, "must be a nonempty finite float64 matrix")
    if x_edges.ndim != 1 or y_edges.ndim != 1 or x_edges.size < 2 or y_edges.size < 2 or not np.isfinite(x_edges).all() or not np.isfinite(y_edges).all() or not np.all(np.diff(x_edges) > 0) or not np.all(np.diff(y_edges) > 0):
        raise _fail("x_edges/y_edges", "grid edges are invalid")
    if difference.shape != (y_edges.size - 1, x_edges.size - 1):
        raise _fail("contrast_result", "difference shape does not match the grid")
    counts = result.get("valid_sample_counts")
    if not isinstance(counts, Mapping) or set(counts) != {"baseline", "treatment"} or any(type(counts[name]) is not int or counts[name] <= 0 for name in counts):
        raise _fail("valid_sample_counts", "must contain positive baseline and treatment counts")
    arms = result.get("source_arms")
    if not isinstance(arms, Mapping) or set(arms) != {"baseline", "treatment"}:
        raise _fail("source_arms", "must contain exactly both source arms")
    arm_copy: dict[str, Any] = {}
    for arm in ("baseline", "treatment"):
        if not isinstance(arms[arm], Mapping) or arms[arm].get("role") != arm:
            raise _fail(f"source_arms.{arm}", "role is invalid")
        arm_copy[arm] = _json_copy(arms[arm], field=f"source_arms.{arm}")
    return {
        "schema_id": _text(result.get("schema_id"), field="contrast_result.schema_id"),
        "schema_version": 1,
        "operation": "difference",
        "formula": result["formula"],
        "baseline_role": "baseline",
        "treatment_role": "treatment",
        CONTRAST_FIELD: np.array(difference, dtype=np.float64, copy=True),
        "x_edges": np.array(x_edges, dtype=np.float64, copy=True),
        "y_edges": np.array(y_edges, dtype=np.float64, copy=True),
        "valid_sample_counts": {name: int(counts[name]) for name in counts},
        "source_arms": arm_copy,
        "policy": _json_copy(policy, field="contrast_result.policy"),
        "policy_digest": str(result["policy_digest"]),
        "config_digest": str(result["config_digest"]),
    }


def _validate_source_pair(plan: ProviderOccupancyContrastMaterializationPlan | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(plan, ProviderOccupancyContrastMaterializationPlan):
        archive = plan.source_zarr
        source_scope = plan.source_scope
        paths = {"baseline": plan.baseline_run_path, "treatment": plan.treatment_run_path}
        digests = {"baseline": plan.baseline_manifest_digest, "treatment": plan.treatment_manifest_digest}
        expected_evidence = plan.source_evidence
        result = _validate_contrast_result(plan.contrast_result)
    else:
        archive = Path(plan["archive"])
        source_scope = str(plan["source_scope"])
        paths = plan["paths"]
        digests = plan["digests"]
        expected_evidence = {}
        result = _validate_contrast_result(plan["result"])
    evidence: dict[str, dict[str, Any]] = {}
    for arm in ("baseline", "treatment"):
        current = _validate_source_manifest(
            archive,
            arm=arm,
            run_path=paths[arm],
            expected_manifest_digest=digests[arm],
            source_scope=source_scope,
        )
        if expected_evidence and current["manifest"] != expected_evidence[arm]["manifest"]:
            raise _fail(f"{arm}.source_manifest", "changed after planning")
        if current["summary"]["source_manifest"] != result["source_arms"][arm]["source_manifest"]:
            raise _fail(f"{arm}.source_manifest", "does not match the contrast arm")
        if current["summary"]["source_selections"] != result["source_arms"][arm]["source_selections"] or current["summary"]["source_occurrences"] != result["source_arms"][arm]["source_occurrences"]:
            raise _fail(f"{arm}.source_arm", "selection or occurrence records changed")
        evidence[arm] = current
    baseline = evidence["baseline"]["summary"]
    treatment = evidence["treatment"]["summary"]
    for field in _COMPATIBILITY_FIELDS:
        if canonical_json_sha256(baseline[field]) != canonical_json_sha256(treatment[field]):
            raise _fail(f"compatibility.{field}", "baseline and treatment differ")
    if baseline["provider_id"] != treatment["provider_id"]:
        raise _fail("provider_id", "cross-provider contrasts are forbidden")
    if not np.array_equal(np.asarray(baseline["grid"]["x_edges"], dtype=np.float64), result["x_edges"]) or not np.array_equal(np.asarray(baseline["grid"]["y_edges"], dtype=np.float64), result["y_edges"]):
        raise _fail("grid", "source grid does not equal the contrast grid")
    expected = np.subtract(
        evidence["treatment"]["summary"]["occupancy_fraction"],
        evidence["baseline"]["summary"]["occupancy_fraction"],
        dtype=np.float64,
    )
    if not np.array_equal(expected, result[CONTRAST_FIELD]):
        raise _fail(CONTRAST_FIELD, "does not equal the exact source pooled arrays")
    for arm in ("baseline", "treatment"):
        if evidence[arm]["summary"]["valid_sample_count"] != result["valid_sample_counts"][arm]:
            raise _fail(f"{arm}.valid_sample_count", "does not equal the contrast result")
    return {"valid": True, "errors": [], "evidence": evidence}


def build_provider_occupancy_contrast_materialization_plan(
    analysis_zarr: str | Path,
    *,
    baseline_run_path: str,
    treatment_run_path: str,
    baseline_manifest_digest: str,
    treatment_manifest_digest: str,
    contrast_result: Mapping[str, Any],
    run_name: str,
    scratch_root: str | Path,
    source_scope: str,
    software_record: Mapping[str, Any] | None = None,
) -> ProviderOccupancyContrastMaterializationPlan:
    archive = Path(analysis_zarr).expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr does not exist: {archive}")
    if source_scope != SOURCE_SCOPE_POOLED:
        raise _fail("source_scope", "only explicit pooled scope is implemented")
    baseline_path = _canonical_source_path(baseline_run_path)
    treatment_path = _canonical_source_path(treatment_run_path)
    if baseline_path == treatment_path:
        raise _fail("source_run_path", "baseline and treatment paths must differ")
    name = _safe_run_name(run_name)
    run_path = f"{PARENT_PATH}/{name}"
    target = archive.joinpath(*run_path.split("/"))
    if target.exists():
        raise FileExistsError(f"Refusing existing immutable contrast run: {target}")
    normalized = _validate_contrast_result(contrast_result)
    evidence: dict[str, dict[str, Any]] = {}
    for arm, path, digest in (("baseline", baseline_path, baseline_manifest_digest), ("treatment", treatment_path, treatment_manifest_digest)):
        evidence[arm] = _validate_source_manifest(archive, arm=arm, run_path=path, expected_manifest_digest=digest, source_scope=source_scope)
    _validate_source_pair({"archive": archive, "source_scope": source_scope, "paths": {"baseline": baseline_path, "treatment": treatment_path}, "digests": {"baseline": baseline_manifest_digest, "treatment": treatment_manifest_digest}, "result": normalized})
    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    try:
        parent = root[PARENT_PATH]
    except KeyError:
        parent = None
    scratch = Path(scratch_root).expanduser().resolve()
    if scratch == archive or archive in scratch.parents:
        raise _fail("scratch_root", "must be outside the authoritative archive")
    local_zarr = scratch / f"provider_occupancy_contrast_{uuid.uuid4().hex}.zarr"
    arrays = {
        CONTRAST_FIELD: np.array(normalized[CONTRAST_FIELD], dtype=np.float64, copy=True),
        "x_edges": np.array(normalized["x_edges"], dtype=np.float64, copy=True),
        "y_edges": np.array(normalized["y_edges"], dtype=np.float64, copy=True),
    }
    software = _json_copy(software_record if software_record is not None else {"materializer": MATERIALIZATION_SCHEMA_ID}, field="software_record")
    run_provenance = build_writer_run_provenance(
        command="provider_occupancy_contrast_materializer",
        params={
            "run_name": name,
            "source_scope": source_scope,
            "baseline_manifest_digest": baseline_manifest_digest,
            "treatment_manifest_digest": treatment_manifest_digest,
        },
        input_run_ids={
            "baseline_occupancy": baseline_manifest_digest,
            "treatment_occupancy": treatment_manifest_digest,
        },
        cwd=Path(__file__).resolve().parents[4],
        include_system_context=False,
    )
    payload = {
        "schema_id": CONTRAST_RUN_SCHEMA_ID,
        "schema_version": CONTRAST_RUN_SCHEMA_VERSION,
        "run_name": name,
        "run_path": run_path,
        "status": RUN_STATUS_COMPLETE,
        "stage_selector_eligible": False,
        "source_scope": source_scope,
        "operation": "difference",
        "formula": normalized["formula"],
        "roles": {"baseline": "baseline", "treatment": "treatment"},
        "policy": normalized["policy"],
        "policy_digest": normalized["policy_digest"],
        "config_digest": normalized["config_digest"],
        "valid_sample_counts": normalized["valid_sample_counts"],
        "grid": {"x_edges": arrays["x_edges"].tolist(), "y_edges": arrays["y_edges"].tolist()},
        "source_runs": {
            arm: {
                "role": arm,
                "run_path": evidence[arm]["run_path"],
                "manifest_sha256": evidence[arm]["manifest_sha256"],
                "manifest_envelope_sha256": evidence[arm]["manifest_envelope_sha256"],
                "source_bindings_sha256": evidence[arm]["bindings_sha256"],
            }
            for arm in ("baseline", "treatment")
        },
        "source_arm_records": normalized["source_arms"],
        "source_manifest_bindings": {
            arm: {
                "run_path": evidence[arm]["run_path"],
                "manifest_sha256": evidence[arm]["manifest_sha256"],
                "manifest_envelope_sha256": evidence[arm]["manifest_envelope_sha256"],
                "source_manifest_attr": SOURCE_MANIFEST_ATTR,
                "source_manifest_digest_attr": SOURCE_MANIFEST_SHA256_ATTR,
                "source_bindings": evidence[arm]["payload"]["source_bindings"],
                "source_bindings_sha256": evidence[arm]["bindings_sha256"],
            }
            for arm in ("baseline", "treatment")
        },
        "arrays": [{"path": path, "shape": list(values.shape), "dtype": values.dtype.str, "sha256": sha256_array(values)} for path, values in sorted(arrays.items())],
        "provenance": {"software": software, "source_occupancy_schema_id": PROVIDER_OCCUPANCY_MANIFEST_SCHEMA_ID, "source_occupancy_schema_version": PROVIDER_OCCUPANCY_MANIFEST_SCHEMA_VERSION, "materialization_schema": MATERIALIZATION_SCHEMA_ID},
        "retry_policy": "immutable_named_run_no_replace_retry_requires_new_run_name_v1",
        "selector_policy": "selector_ineligible_parent_selectors_unchanged_v1",
        "parent_selector_attrs_before": _selector_snapshot(parent),
    }
    manifest = {"schema_id": CONTRAST_RUN_SCHEMA_ID, "schema_version": CONTRAST_RUN_SCHEMA_VERSION, "payload_sha256": canonical_json_sha256(payload), "payload": payload}
    return ProviderOccupancyContrastMaterializationPlan(
        source_zarr=archive,
        run_name=name,
        run_path=run_path,
        scratch_root=scratch,
        local_zarr=local_zarr,
        local_run_path=local_zarr.joinpath(*run_path.split("/")),
        baseline_run_path=baseline_path,
        treatment_run_path=treatment_path,
        baseline_manifest_digest=_digest(baseline_manifest_digest, field="baseline_manifest_digest"),
        treatment_manifest_digest=_digest(treatment_manifest_digest, field="treatment_manifest_digest"),
        source_scope=source_scope,
        contrast_result=normalized,
        source_evidence=evidence,
        arrays=arrays,
        manifest=manifest,
        parent_selector_attrs=_selector_snapshot(parent),
        run_provenance=run_provenance,
    )


def _validate_run(path: Path, *, expected_manifest: Mapping[str, Any]) -> dict[str, Any]:
    run = open_zarr_root(path, mode="r", use_consolidated=False)
    manifest = run.attrs.get(MANIFEST_ATTR)
    if manifest != expected_manifest or run.attrs.get(MANIFEST_SHA256_ATTR) != canonical_json_sha256(manifest):
        raise ProviderOccupancyContrastMaterializationError("contrast manifest is missing or stale")
    if run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE or run.attrs.get("stage_selector_eligible") is not False:
        raise ProviderOccupancyContrastMaterializationError("contrast run is incomplete or selector eligible")
    if any(_selector_like_name(name) for name in run.attrs):
        raise ProviderOccupancyContrastMaterializationError(
            "contrast run contains selector attributes"
        )
    if not isinstance(run.attrs.get("run_provenance"), Mapping):
        raise ProviderOccupancyContrastMaterializationError(
            "contrast run provenance is missing"
        )
    for declaration in manifest["payload"]["arrays"]:
        values = np.asarray(run[declaration["path"]][:])
        observed = {"path": declaration["path"], "shape": list(values.shape), "dtype": values.dtype.str, "sha256": sha256_array(values)}
        if observed != declaration:
            raise ProviderOccupancyContrastMaterializationError(f"contrast array drifted: {declaration['path']}")
    return {"valid": True, "errors": [], "manifest_sha256": canonical_json_sha256(manifest), "payload_sha256": manifest["payload_sha256"], "array_count": len(manifest["payload"]["arrays"])}


def _write_local(plan: ProviderOccupancyContrastMaterializationPlan) -> None:
    _validate_source_pair(plan)
    plan.scratch_root.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(plan.local_zarr), mode="w-", zarr_format=3, use_consolidated=False)
    run_parent = _require_provenance_parent(root)
    run = run_parent.create_group(plan.run_name)
    mark_run_started(run, run_name=plan.run_name, stage="provider_occupancy_contrast")
    for path, values in plan.arrays.items():
        parent_path, _, leaf = path.rpartition("/")
        array_parent = run.require_group(parent_path) if parent_path else run
        array_parent.create_array(leaf or path, data=values, chunks=tuple(max(1, min(int(size), 16384)) for size in values.shape))
    run.attrs[MANIFEST_ATTR] = json_attr_safe(plan.manifest)
    run.attrs[MANIFEST_SHA256_ATTR] = canonical_json_sha256(plan.manifest)
    run.attrs.update({"schema_id": CONTRAST_RUN_SCHEMA_ID, "schema_version": CONTRAST_RUN_SCHEMA_VERSION, "stage_selector_eligible": False, "source_scope": plan.source_scope, "policy": json_attr_safe(plan.contrast_result["policy"]), "policy_digest": plan.contrast_result["policy_digest"], "config_digest": plan.contrast_result["config_digest"], "source_runs": json_attr_safe(plan.manifest["payload"]["source_runs"]), "source_bindings": json_attr_safe(plan.manifest["payload"]["source_manifest_bindings"]), "valid_sample_counts": json_attr_safe(plan.contrast_result["valid_sample_counts"]), "retry_policy": plan.manifest["payload"]["retry_policy"], "run_provenance": json_attr_safe(dict(plan.run_provenance)), "direct_consolidated_metadata_equality": {"status": "pending_final_publication_consolidation"}})
    if run.attrs.get(MANIFEST_ATTR) != plan.manifest:
        raise ProviderOccupancyContrastMaterializationError(
            "contrast manifest was not persisted exactly before completion"
        )
    mark_run_complete(run, parent_group=run_parent, run_name=plan.run_name, run_provenance=plan.run_provenance)
    _validate_run(plan.local_run_path, expected_manifest=plan.manifest)


def publish_provider_occupancy_contrast_run(plan: ProviderOccupancyContrastMaterializationPlan, *, copy_backend: str = "python", keep_scratch: bool = False) -> dict[str, Any]:
    """Atomically publish one complete, selector-ineligible pooled contrast."""

    _write_local(plan)
    acceptance: dict[str, Any] = {}

    def validate(path: Path) -> Mapping[str, Any]:
        return _validate_run(path, expected_manifest=plan.manifest)

    def prepare(root: Any) -> tuple[Any]:
        return (_require_provenance_parent(root),)

    def complete(_root: Any, parent: Any, run: Any) -> None:
        run.attrs["stage_selector_eligible"] = False
        if not isinstance(run.attrs.get("run_provenance"), Mapping):
            run.attrs["run_provenance"] = json_attr_safe(dict(plan.run_provenance))
        mark_run_complete(
            run,
            parent_group=parent,
            run_name=plan.run_name,
            run_provenance=run.attrs["run_provenance"],
        )

    def verify(root: Any) -> None:
        parent = root[PARENT_PATH]
        if _selector_snapshot(parent) != dict(plan.parent_selector_attrs):
            raise RuntimeError("selector-ineligible contrast changed parent selectors")
        _validate_run(
            plan.source_zarr.joinpath(*plan.run_path.split("/")),
            expected_manifest=plan.manifest,
        )

    def finalize(_root: Any, _parent: Any, run: Any) -> None:
        _validate_source_pair(plan)
        consolidate_metadata_capture_expected_warnings(plan.source_zarr)
        receipt = validate_direct_consolidated_subtree(plan.source_zarr, subtree_path=plan.run_path)
        run.attrs["direct_consolidated_metadata_equality"] = receipt.to_json()
        consolidate_metadata_capture_expected_warnings(plan.source_zarr)
        final_receipt = validate_direct_consolidated_subtree(plan.source_zarr, subtree_path=plan.run_path)
        consolidated = open_zarr_root(plan.source_zarr, mode="r", use_consolidated=True)[plan.run_path]
        if consolidated.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE or consolidated.attrs.get("stage_selector_eligible") is not False:
            raise RuntimeError("consolidated contrast is not complete and selector-ineligible")
        acceptance["direct_consolidated"] = final_receipt.to_json()

    publication = atomic_publish_run_group(AtomicRunPublishSpec(source_zarr=plan.source_zarr, local_run_path=plan.local_run_path, target_run_path=plan.source_zarr.joinpath(*plan.run_path.split("/")), run_name=plan.run_name, lock_suffix="provider-occupancy-contrast-publish", publish_schema_id=PUBLISH_SCHEMA_ID, policy="named_selector_ineligible_provider_occupancy_difference_v2_source_bound", rollback_policy="retain_failed_tombstone_leave_parent_selectors_untouched", content_checksum=True), copy_backend=copy_backend, validate_run=validate, prepare_parents=prepare, complete_run=complete, verify_pointers=verify, activate_run=finalize, repair_failed_publication_visibility=lambda _target: consolidate_metadata_capture_expected_warnings(plan.source_zarr), accept_persisted_activation_on_callback_error=False, payload_metadata={"selector_eligible": False, "selection": "none", "source_scope": plan.source_scope, "operation": "difference"})
    result = {"status": "complete", "run_path": plan.run_path, "manifest_sha256": canonical_json_sha256(plan.manifest), "payload_sha256": plan.manifest["payload_sha256"], "selector_eligible": False, "selection": "none", "source_scope": plan.source_scope, "publication": publication, "validation": acceptance}
    if not keep_scratch and plan.local_zarr.exists():
        shutil.rmtree(plan.local_zarr)
    return json_attr_safe(result)


plan_provider_occupancy_contrast_run = build_provider_occupancy_contrast_materialization_plan
materialize_provider_occupancy_contrast_run = publish_provider_occupancy_contrast_run


__all__ = [
    "CONTRAST_RUN_SCHEMA_ID",
    "MANIFEST_ATTR",
    "MANIFEST_SHA256_ATTR",
    "MATERIALIZATION_SCHEMA_ID",
    "PARENT_PATH",
    "SOURCE_MANIFEST_ATTR",
    "SOURCE_MANIFEST_SHA256_ATTR",
    "SOURCE_SCOPE_POOLED",
    "ProviderOccupancyContrastMaterializationError",
    "ProviderOccupancyContrastMaterializationPlan",
    "build_pooled_occupancy_contrast_summary",
    "build_provider_occupancy_contrast_materialization_plan",
    "materialize_provider_occupancy_contrast_run",
    "plan_provider_occupancy_contrast_run",
    "publish_provider_occupancy_contrast_run",
]
