"""Materialize one compiled composable stimulus selection as a sealed candidate.

The compiler in :mod:`fisheye.analysis_workflows.composable_stimulus_selection`
is intentionally in-memory.  This module is its storage boundary: it copies one
exact ``CompiledSelection`` from node-local Zarr into
``analysis/stimulus_selection_runs/<run>`` through the shared atomic publisher.
The child is permanently selector-ineligible and no parent selector is ever
advanced.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.analysis_workflows.composable_stimulus_selection import (
    CompiledSelection,
    ResolvedInterval,
    ResolvedOccurrence,
    RoleMetadata,
    SourceMembership,
    TimelineAuthority,
    TrimSpec,
    canonical_json,
    canonical_sha256,
)
from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr.chunk_profiles import create_geometry_preload_array
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock_path,
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


PARENT_PATH = "analysis/stimulus_selection_runs"
MATERIALIZATION_SCHEMA_ID = "palette.composable_stimulus_selection_materialization"
MATERIALIZATION_SCHEMA_VERSION = 1
PUBLISH_SCHEMA_ID = "palette.composable_stimulus_selection_publish"
RUN_SCHEMA_ID = "palette.composable_stimulus_selection_run"
RETRY_POLICY = "new_immutable_run_name_required"
RUN_PATH_POLICY = "exact_non_selector_child_v1"
SELECTOR_INELIGIBLE_POLICY = "permanent_selector_ineligible_no_parent_pointer_update"
REQUESTED_JSON_ATTR = "requested_selection_json"
RESOLVED_JSON_ATTR = "resolved_selection_json"
TIMELINE_AUTHORITY_JSON_ATTR = "timeline_authority_json"
ARRAY_MANIFEST_JSON_ATTR = "logical_array_manifest_json"
ARRAY_MANIFEST_DIGEST_ATTR = "logical_array_manifest_sha256"
COMPILED_SELECTION_DIGEST_ATTR = "compiled_selection_sha256"

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


@dataclass(frozen=True)
class ComposableStimulusSelectionMaterializationPlan:
    """Immutable plan binding one compiled selection to one output name."""

    source_zarr: Path
    scratch_root: Path
    local_zarr: Path
    run_name: str
    request_digest: str
    resolved_digest: str
    compiled_selection_digest: str
    requested_json: str
    resolved_json: str
    timeline_authority_json: str
    parent_selector_snapshot: Mapping[str, Any]

    @property
    def run_path(self) -> str:
        return f"{PARENT_PATH}/{self.run_name}"

    @property
    def local_run_path(self) -> Path:
        return self.local_zarr.joinpath(*self.run_path.split("/"))

    @property
    def target_run_path(self) -> Path:
        return self.source_zarr.joinpath(*self.run_path.split("/"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": MATERIALIZATION_SCHEMA_ID,
            "schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "source_zarr": str(self.source_zarr),
            "scratch_root": str(self.scratch_root),
            "local_zarr": str(self.local_zarr),
            "local_run_path": str(self.local_run_path),
            "target_run_path": str(self.target_run_path),
            "run_path": self.run_path,
            "run_name": self.run_name,
            "request_digest": self.request_digest,
            "resolved_digest": self.resolved_digest,
            "compiled_selection_sha256": self.compiled_selection_digest,
            "requested_json_sha256": hashlib.sha256(
                self.requested_json.encode("utf-8")
            ).hexdigest(),
            "resolved_json_sha256": hashlib.sha256(
                self.resolved_json.encode("utf-8")
            ).hexdigest(),
            "timeline_authority_json_sha256": hashlib.sha256(
                self.timeline_authority_json.encode("utf-8")
            ).hexdigest(),
            "parent_selector_snapshot": json_attr_safe(
                dict(self.parent_selector_snapshot)
            ),
            "retry_policy": RETRY_POLICY,
            "selector_policy": SELECTOR_INELIGIBLE_POLICY,
        }


def _safe_run_name(value: str) -> str:
    if type(value) is not str:
        raise TypeError("run_name must be an exact string")
    name = value.strip()
    if (
        not name
        or name in {".", ".."}
        or name.lower() in _SELECTOR_ALIASES
        or _selector_like_name(name)
        or "/" in name
        or "\\" in name
        or any(character.isspace() for character in name)
    ):
        raise ValueError(
            "run_name must be one exact non-selector child name; selector aliases "
            "and path-like names are forbidden"
        )
    return name


def _group_at(root: Any, path: str) -> Any | None:
    node = root
    for component in path.strip("/").split("/"):
        if not component:
            continue
        try:
            node = node[component]
        except (KeyError, FileNotFoundError, TypeError):
            return None
    return node


def _selector_attr_name(name: object) -> bool:
    if type(name) is not str:
        return False
    lowered = name.lower()
    return lowered in _SELECTOR_ALIASES or lowered.startswith(_SELECTOR_NAME_PREFIXES)


def _selector_alias_attr_name(name: object) -> bool:
    return _selector_attr_name(name)


def _selector_like_name(value: object) -> bool:
    if type(value) is not str:
        return False
    lowered = value.lower()
    return lowered in _SELECTOR_ALIASES or lowered.startswith(_SELECTOR_NAME_PREFIXES)


def _require_provenance_parent(root: Any) -> Any:
    parent = require_runs_parent(
        root.require_group("analysis"),
        "stimulus_selection_runs",
        completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    )
    if parent.attrs.get(COMPLETION_EPOCH_ATTR) != COMPLETION_EPOCH_REQUIRE_PROVENANCE:
        raise RuntimeError(
            "stimulus-selection parent must require completion provenance"
        )
    return parent


def _selector_snapshot(parent: Any | None) -> dict[str, Any]:
    if parent is None:
        return {"exists": False, "attrs": {}}
    attrs = {
        str(key): json_attr_safe(value)
        for key, value in dict(parent.attrs).items()
        if _selector_attr_name(key)
    }
    return {"exists": True, "attrs": attrs}


def _require_same_selector_snapshot(root: Any, expected: Mapping[str, Any]) -> None:
    parent = _group_at(root, PARENT_PATH)
    observed = _selector_snapshot(parent)
    expected_attrs = dict(expected.get("attrs", {}))
    observed_attrs = dict(observed.get("attrs", {}))
    if expected_attrs != observed_attrs:
        raise RuntimeError(
            "stimulus-selection parent selector attributes changed: "
            f"before={expected_attrs!r}, after={observed_attrs!r}"
        )


def _compiled_snapshot(compiled: CompiledSelection) -> tuple[str, str, str, str, str]:
    requested_json = canonical_json(compiled.requested)
    resolved_json = canonical_json(compiled.resolved_payload())
    authority_json = canonical_json(compiled.authority.to_dict())
    return (
        requested_json,
        resolved_json,
        authority_json,
        str(compiled.request_digest),
        str(compiled.resolved_digest),
    )


def _assert_compiled_integrity(compiled: CompiledSelection) -> None:
    requested_json, resolved_json, _authority_json, request_digest, resolved_digest = (
        _compiled_snapshot(compiled)
    )
    if request_digest != canonical_sha256(json.loads(requested_json)):
        raise ValueError("CompiledSelection request_digest is stale")
    if resolved_digest != canonical_sha256(json.loads(resolved_json)):
        raise ValueError("CompiledSelection resolved_digest is stale")


def _compiled_digest(compiled: CompiledSelection) -> str:
    return canonical_sha256(compiled.to_dict())


def _assert_plan_matches_compiled(
    plan: ComposableStimulusSelectionMaterializationPlan,
    compiled: CompiledSelection,
) -> None:
    if type(compiled) is not CompiledSelection:
        raise TypeError("materialization input must be one exact CompiledSelection")
    _assert_compiled_integrity(compiled)
    requested_json, resolved_json, authority_json, request_digest, resolved_digest = (
        _compiled_snapshot(compiled)
    )
    observed = (
        request_digest,
        resolved_digest,
        _compiled_digest(compiled),
        requested_json,
        resolved_json,
        authority_json,
    )
    expected = (
        plan.request_digest,
        plan.resolved_digest,
        plan.compiled_selection_digest,
        plan.requested_json,
        plan.resolved_json,
        plan.timeline_authority_json,
    )
    if observed != expected:
        raise ValueError(
            "CompiledSelection changed after planning or has stale request/resolved "
            "digests; refusing materialization"
        )


def build_composable_stimulus_selection_materialization_plan(
    source_zarr: str | Path,
    *,
    compiled_selection: CompiledSelection,
    scratch_root: str | Path,
    run_name: str,
) -> ComposableStimulusSelectionMaterializationPlan:
    """Plan a candidate without creating scratch files or mutating the archive."""

    if type(compiled_selection) is not CompiledSelection:
        raise TypeError("compiled_selection must be one exact CompiledSelection")
    _assert_compiled_integrity(compiled_selection)
    source = Path(source_zarr).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source analysis Zarr not found: {source}")
    if source == scratch or source in scratch.parents or scratch in source.parents:
        raise ValueError("scratch_root and source_zarr must be disjoint")
    name = _safe_run_name(run_name)
    target = source.joinpath(*PARENT_PATH.split("/"), name)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing existing stimulus-selection run: {target}")
    local_zarr = scratch / f"composable-stimulus-selection-{name}.zarr"
    if local_zarr.exists() or local_zarr.is_symlink():
        raise FileExistsError(f"Refusing existing local materialization: {local_zarr}")

    root = open_zarr_root(source, mode="r", use_consolidated=False)
    parent = _group_at(root, PARENT_PATH)
    if parent is not None and name in parent:
        raise FileExistsError(f"Refusing existing stimulus-selection child: {name!r}")
    requested_json, resolved_json, authority_json, request_digest, resolved_digest = (
        _compiled_snapshot(compiled_selection)
    )
    return ComposableStimulusSelectionMaterializationPlan(
        source_zarr=source,
        scratch_root=scratch,
        local_zarr=local_zarr,
        run_name=name,
        request_digest=request_digest,
        resolved_digest=resolved_digest,
        compiled_selection_digest=_compiled_digest(compiled_selection),
        requested_json=requested_json,
        resolved_json=resolved_json,
        timeline_authority_json=authority_json,
        parent_selector_snapshot=_selector_snapshot(parent),
    )


def _utf8_matrix(values: Sequence[str], *, field: str) -> tuple[np.ndarray, np.ndarray]:
    encoded: list[bytes] = []
    for value in values:
        if type(value) is not str:
            raise ValueError(f"{field} contains a non-string value")
        try:
            encoded.append(value.encode("utf-8", errors="strict"))
        except UnicodeError as exc:
            raise ValueError(f"{field} contains invalid UTF-8") from exc
    width = max(1, max((len(value) for value in encoded), default=0))
    matrix = np.zeros((len(encoded), width), dtype=np.uint8)
    lengths = np.zeros((len(encoded),), dtype=np.int64)
    for index, value in enumerate(encoded):
        matrix[index, : len(value)] = np.frombuffer(value, dtype=np.uint8)
        lengths[index] = len(value)
    return matrix, lengths


def _json_text(value: Mapping[str, Any] | None) -> str:
    return "null" if value is None else canonical_json(value)


def _role_text(role: RoleMetadata | None) -> str:
    return "null" if role is None else canonical_json(role.to_dict())


def _trim_text(trim: TrimSpec | None) -> str:
    return "null" if trim is None else canonical_json(trim.to_dict())


def _selection_arrays(compiled: CompiledSelection) -> dict[str, np.ndarray]:
    resolved = compiled.resolved_intervals
    memberships = [
        membership
        for interval in resolved
        for membership in interval.source_memberships
    ]
    membership_offsets = [0]
    for interval in resolved:
        membership_offsets.append(
            membership_offsets[-1] + len(interval.source_memberships)
        )
    occurrence_intervals = [
        interval
        for occurrence in compiled.occurrences
        for interval in occurrence.intervals
    ]
    occurrence_offsets = [0]
    for occurrence in compiled.occurrences:
        occurrence_offsets.append(occurrence_offsets[-1] + len(occurrence.intervals))

    arrays: dict[str, np.ndarray] = {
        "resolved_interval_bounds": np.asarray(
            [[item.start_frame, item.end_frame] for item in resolved], dtype=np.int64
        ).reshape((-1, 2)),
        "pooled_interval_bounds": np.asarray(
            compiled.pooled_intervals, dtype=np.int64
        ).reshape((-1, 2)),
        "resolved_membership_offsets": np.asarray(
            membership_offsets, dtype=np.int64
        ),
        "membership_original_interval_bounds": np.asarray(
            [
                [item.original_start_frame, item.original_end_frame]
                for item in memberships
            ],
            dtype=np.int64,
        ).reshape((-1, 2)),
        "membership_selected_interval_bounds": np.asarray(
            [
                [item.selected_start_frame, item.selected_end_frame]
                for item in memberships
            ],
            dtype=np.int64,
        ).reshape((-1, 2)),
        "occurrence_interval_offsets": np.asarray(occurrence_offsets, dtype=np.int64),
        "occurrence_interval_bounds": np.asarray(
            occurrence_intervals, dtype=np.int64
        ).reshape((-1, 2)),
        "occurrence_frame_counts": np.asarray(
            [item.frame_count for item in compiled.occurrences], dtype=np.int64
        ),
    }
    membership_string_values = {
        "membership_reference_kind": [item.reference_kind for item in memberships],
        "membership_reference_id": [item.reference_id for item in memberships],
        "membership_occurrence_id": [item.occurrence_id for item in memberships],
        "membership_label": [item.label for item in memberships],
        "membership_role_json": [_role_text(item.role) for item in memberships],
        "membership_trim_json": [_trim_text(item.trim) for item in memberships],
    }
    occurrence_string_values = {
        "occurrence_reference_kind": [
            item.reference_kind for item in compiled.occurrences
        ],
        "occurrence_reference_id": [item.reference_id for item in compiled.occurrences],
        "occurrence_id": [item.occurrence_id for item in compiled.occurrences],
        "occurrence_label": [item.label for item in compiled.occurrences],
        "occurrence_role_json": [_role_text(item.role) for item in compiled.occurrences],
    }
    for prefix, values in (*membership_string_values.items(), *occurrence_string_values.items()):
        matrix, lengths = _utf8_matrix(values, field=prefix)
        arrays[f"{prefix}_utf8"] = matrix
        arrays[f"{prefix}_utf8_length"] = lengths
    return arrays


_ARRAY_SPECS: dict[str, tuple[tuple[str, ...], str, str, str]] = {
    "resolved_interval_bounds": (
        ("resolved_interval", "bound"),
        "acquisition_frame",
        "resolved interval half-open bounds",
        "timeline.acquisition_frame_domain",
    ),
    "pooled_interval_bounds": (
        ("pooled_interval", "bound"),
        "acquisition_frame",
        "pooled interval half-open bounds",
        "timeline.acquisition_frame_domain",
    ),
    "resolved_membership_offsets": (
        ("resolved_interval", "membership_offset"),
        "membership_index",
        "CSR offsets for interval source memberships",
        "selection.resolved_source_memberships",
    ),
    "membership_original_interval_bounds": (
        ("membership", "bound"),
        "acquisition_frame",
        "source original half-open bounds",
        "stimulus_authority.source_interval",
    ),
    "membership_selected_interval_bounds": (
        ("membership", "bound"),
        "acquisition_frame",
        "source selected half-open bounds after trim",
        "selection.selected_source_membership",
    ),
    "occurrence_interval_offsets": (
        ("occurrence", "interval_offset"),
        "interval_index",
        "CSR offsets for occurrence intervals",
        "selection.resolved_occurrences",
    ),
    "occurrence_interval_bounds": (
        ("occurrence_interval", "bound"),
        "acquisition_frame",
        "occurrence half-open bounds",
        "selection.resolved_occurrences",
    ),
    "occurrence_frame_counts": (
        ("occurrence",),
        "acquisition_frame",
        "resolved occurrence frame count",
        "selection.resolved_occurrences",
    ),
}
for _prefix in (
    "membership_reference_kind",
    "membership_reference_id",
    "membership_occurrence_id",
    "membership_label",
    "membership_role_json",
    "membership_trim_json",
):
    _ARRAY_SPECS[f"{_prefix}_utf8"] = (
        ("membership", "utf8_byte"),
        "utf8_byte",
        f"UTF-8 bytes for {_prefix}",
        "selection.resolved_source_memberships",
    )
    _ARRAY_SPECS[f"{_prefix}_utf8_length"] = (
        ("membership",),
        "utf8_byte",
        f"UTF-8 byte length for {_prefix}",
        "selection.resolved_source_memberships",
    )
for _prefix in (
    "occurrence_reference_kind",
    "occurrence_reference_id",
    "occurrence_id",
    "occurrence_label",
    "occurrence_role_json",
):
    _ARRAY_SPECS[f"{_prefix}_utf8"] = (
        ("occurrence", "utf8_byte"),
        "utf8_byte",
        f"UTF-8 bytes for {_prefix}",
        "selection.resolved_occurrences",
    )
    _ARRAY_SPECS[f"{_prefix}_utf8_length"] = (
        ("occurrence",),
        "utf8_byte",
        f"UTF-8 byte length for {_prefix}",
        "selection.resolved_occurrences",
    )
_EXPECTED_ARRAY_NAMES = tuple(sorted(_ARRAY_SPECS))


def _array_content_digest(array: Any) -> str:
    values = np.asarray(array[:])
    if values.dtype.kind in {"O", "U", "S"}:
        raise ValueError("logical arrays must not use object or string dtypes")
    digest = hashlib.sha256()
    digest.update(np.dtype(values.dtype).str.encode("ascii"))
    digest.update(canonical_json(list(values.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(values).tobytes(order="C"))
    return digest.hexdigest()


def _build_array_manifest(group: Any) -> dict[str, Any]:
    declarations: list[dict[str, Any]] = []
    for name in _EXPECTED_ARRAY_NAMES:
        array = group[name]
        axes, units, description, authority_role = _ARRAY_SPECS[name]
        declarations.append(
            {
                "path": name,
                "dtype": np.dtype(array.dtype).str,
                "shape": [int(value) for value in array.shape],
                "axes": list(axes),
                "units": units,
                "description": description,
                "authority_role": authority_role,
                "content_sha256": _array_content_digest(array),
            }
        )
    return {
        "schema_id": "palette.composable_stimulus_selection.logical_array_manifest",
        "schema_version": 1,
        "manifest_written_last": True,
        "array_count": len(declarations),
        "digest_algorithm": "sha256(dtype_shape_c_order_bytes)_v1",
        "arrays": declarations,
    }


def _strict_json_attr(attrs: Mapping[str, Any], name: str) -> tuple[str, Any]:
    value = attrs.get(name)
    if type(value) is not str:
        raise ValueError(f"{name} must be one strict canonical JSON string")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if canonical_json(decoded) != value:
        raise ValueError(f"{name} is not canonical strict JSON")
    return value, decoded


def _decode_utf8_vector(group: Any, prefix: str, expected_count: int) -> list[str]:
    matrix = np.asarray(group[f"{prefix}_utf8"][:])
    lengths = np.asarray(group[f"{prefix}_utf8_length"][:])
    if matrix.dtype != np.dtype("u1") or matrix.ndim != 2:
        raise ValueError(f"{prefix} UTF-8 matrix must be uint8 and two-dimensional")
    if lengths.dtype != np.dtype("<i8") or lengths.shape != (expected_count,):
        raise ValueError(f"{prefix} UTF-8 lengths are not int64 and row aligned")
    if matrix.shape[0] != expected_count or matrix.shape[1] < 1:
        raise ValueError(f"{prefix} UTF-8 matrix has the wrong shape")
    values: list[str] = []
    for row, length in zip(matrix, lengths):
        if int(length) < 0 or int(length) > matrix.shape[1]:
            raise ValueError(f"{prefix} UTF-8 length is outside its matrix")
        if np.any(row[int(length) :] != 0):
            raise ValueError(f"{prefix} UTF-8 padding is nonzero")
        try:
            values.append(bytes(row[: int(length)].tolist()).decode("utf-8"))
        except UnicodeError as exc:
            raise ValueError(f"{prefix} contains invalid UTF-8") from exc
    return values


def _decode_optional_json(value: str, *, field: str) -> Mapping[str, Any] | None:
    if value == "null":
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} is not valid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{field} must be null or one JSON object")
    if canonical_json(decoded) != value:
        raise ValueError(f"{field} is not canonical JSON")
    return dict(decoded)


def _role_from_text(value: str, *, field: str) -> RoleMetadata | None:
    payload = _decode_optional_json(value, field=field)
    if payload is None:
        return None
    role = RoleMetadata(
        role=payload.get("role"),
        label=payload.get("label"),
        metadata=payload.get("metadata", {}),
    )
    if canonical_json(role.to_dict()) != value:
        raise ValueError(f"{field} does not round-trip through RoleMetadata")
    return role


def _trim_from_text(value: str, *, field: str) -> TrimSpec | None:
    payload = _decode_optional_json(value, field=field)
    if payload is None:
        return None
    trim = TrimSpec(
        leading_seconds=payload.get("leading_seconds"),
        trailing_seconds=payload.get("trailing_seconds"),
        fps=payload.get("fps"),
        rounding_policy=payload.get("rounding_policy"),
    )
    if canonical_json(trim.to_dict()) != value:
        raise ValueError(f"{field} does not round-trip through TrimSpec")
    return trim


def _require_bounds(array: Any, *, name: str, upper: int | None = None) -> np.ndarray:
    values = np.asarray(array[:])
    if values.dtype != np.dtype("<i8") or values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"{name} must be an int64 (N,2) bounds array")
    if np.any(values[:, 0] < 0) or np.any(values[:, 1] <= values[:, 0]):
        raise ValueError(f"{name} contains invalid half-open bounds")
    if upper is not None and np.any(values[:, 1] > upper):
        raise ValueError(f"{name} exceeds the acquisition frame domain")
    return values


def _require_offsets(array: Any, *, name: str, last: int) -> np.ndarray:
    values = np.asarray(array[:])
    if values.dtype != np.dtype("<i8") or values.ndim != 1:
        raise ValueError(f"{name} must be an int64 one-dimensional offsets array")
    if not len(values) or int(values[0]) != 0 or int(values[-1]) != last:
        raise ValueError(f"{name} has invalid endpoints")
    if np.any(values[1:] < values[:-1]) or np.any(values < 0) or np.any(values > last):
        raise ValueError(f"{name} is not monotonic within its row domain")
    return values


def reconstruct_compiled_selection(run_path: str | Path) -> CompiledSelection:
    """Reconstruct the compiler object from one materialized run."""

    group = open_zarr_root(Path(run_path), mode="r", use_consolidated=False)
    attrs = dict(group.attrs)
    _, requested = _strict_json_attr(attrs, REQUESTED_JSON_ATTR)
    _, authority_payload = _strict_json_attr(attrs, TIMELINE_AUTHORITY_JSON_ATTR)
    authority = TimelineAuthority(**authority_payload)
    resolved_bounds = _require_bounds(
        group["resolved_interval_bounds"],
        name="resolved_interval_bounds",
        upper=authority.acquisition_frame_count,
    )
    pooled_bounds = _require_bounds(
        group["pooled_interval_bounds"],
        name="pooled_interval_bounds",
        upper=authority.acquisition_frame_count,
    )
    membership_count = int(np.asarray(group["membership_original_interval_bounds"][:]).shape[0])
    original_bounds = _require_bounds(
        group["membership_original_interval_bounds"],
        name="membership_original_interval_bounds",
        upper=authority.acquisition_frame_count,
    )
    selected_bounds = _require_bounds(
        group["membership_selected_interval_bounds"],
        name="membership_selected_interval_bounds",
        upper=authority.acquisition_frame_count,
    )
    if selected_bounds.shape != original_bounds.shape:
        raise ValueError("membership interval arrays are not row aligned")
    membership_offsets = _require_offsets(
        group["resolved_membership_offsets"],
        name="resolved_membership_offsets",
        last=membership_count,
    )
    if membership_offsets.shape != (len(resolved_bounds) + 1,):
        raise ValueError("resolved membership offsets do not match interval rows")
    membership_fields = {
        field: _decode_utf8_vector(group, field, membership_count)
        for field in (
            "membership_reference_kind",
            "membership_reference_id",
            "membership_occurrence_id",
            "membership_label",
            "membership_role_json",
            "membership_trim_json",
        )
    }
    resolved_intervals: list[ResolvedInterval] = []
    for index, (start, end) in enumerate(resolved_bounds):
        memberships: list[SourceMembership] = []
        for row in range(int(membership_offsets[index]), int(membership_offsets[index + 1])):
            memberships.append(
                SourceMembership(
                    reference_kind=membership_fields["membership_reference_kind"][row],
                    reference_id=membership_fields["membership_reference_id"][row],
                    occurrence_id=membership_fields["membership_occurrence_id"][row],
                    label=membership_fields["membership_label"][row],
                    original_start_frame=int(original_bounds[row, 0]),
                    original_end_frame=int(original_bounds[row, 1]),
                    selected_start_frame=int(selected_bounds[row, 0]),
                    selected_end_frame=int(selected_bounds[row, 1]),
                    role=_role_from_text(
                        membership_fields["membership_role_json"][row],
                        field="membership_role_json",
                    ),
                    trim=_trim_from_text(
                        membership_fields["membership_trim_json"][row],
                        field="membership_trim_json",
                    ),
                )
            )
        resolved_intervals.append(
            ResolvedInterval(
                start_frame=int(start),
                end_frame=int(end),
                source_memberships=tuple(memberships),
            )
        )

    occurrence_intervals = _require_bounds(
        group["occurrence_interval_bounds"],
        name="occurrence_interval_bounds",
        upper=authority.acquisition_frame_count,
    )
    occurrence_offsets = _require_offsets(
        group["occurrence_interval_offsets"],
        name="occurrence_interval_offsets",
        last=len(occurrence_intervals),
    )
    occurrence_count = len(occurrence_offsets) - 1
    occurrence_fields = {
        field: _decode_utf8_vector(group, field, occurrence_count)
        for field in (
            "occurrence_reference_kind",
            "occurrence_reference_id",
            "occurrence_id",
            "occurrence_label",
            "occurrence_role_json",
        )
    }
    frame_counts = np.asarray(group["occurrence_frame_counts"][:])
    if frame_counts.dtype != np.dtype("<i8") or frame_counts.shape != (occurrence_count,):
        raise ValueError("occurrence_frame_counts is not an int64 occurrence vector")
    occurrences: list[ResolvedOccurrence] = []
    for index in range(occurrence_count):
        start = int(occurrence_offsets[index])
        end = int(occurrence_offsets[index + 1])
        intervals = tuple(
            (int(row[0]), int(row[1])) for row in occurrence_intervals[start:end]
        )
        occurrence = ResolvedOccurrence(
            occurrence_id=occurrence_fields["occurrence_id"][index],
            reference_kind=occurrence_fields["occurrence_reference_kind"][index],
            reference_id=occurrence_fields["occurrence_reference_id"][index],
            label=occurrence_fields["occurrence_label"][index],
            role=_role_from_text(
                occurrence_fields["occurrence_role_json"][index],
                field="occurrence_role_json",
            ),
            intervals=intervals,
        )
        if int(frame_counts[index]) != occurrence.frame_count:
            raise ValueError("occurrence_frame_counts disagrees with occurrence bounds")
        occurrences.append(occurrence)
    request_digest = attrs.get("request_digest")
    resolved_digest = attrs.get("resolved_digest")
    if type(request_digest) is not str or type(resolved_digest) is not str:
        raise ValueError("materialized run lacks exact request/resolved digests")
    return CompiledSelection(
        selection_id=attrs.get("selection_id"),
        aggregation_policy=attrs.get("aggregation_policy"),
        authority=authority,
        requested=requested,
        request_digest=request_digest,
        resolved_intervals=tuple(resolved_intervals),
        pooled_intervals=tuple((int(row[0]), int(row[1])) for row in pooled_bounds),
        occurrences=tuple(occurrences),
        resolved_digest=resolved_digest,
    )


def _validate_array_manifest(group: Any) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        manifest_json, manifest = _strict_json_attr(group.attrs, ARRAY_MANIFEST_JSON_ATTR)
        expected_digest = group.attrs.get(ARRAY_MANIFEST_DIGEST_ATTR)
        if type(expected_digest) is not str or canonical_sha256(manifest) != expected_digest:
            errors.append("logical array manifest digest mismatch")
        if manifest.get("manifest_written_last") is not True:
            errors.append("logical array manifest is not marked manifest-last")
        if tuple(sorted(group.array_keys())) != _EXPECTED_ARRAY_NAMES:
            errors.append("run contains unexpected or missing logical arrays")
        observed = _build_array_manifest(group)
        if canonical_json(observed) != manifest_json:
            errors.append("logical array manifest does not match array declarations")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"invalid logical array manifest: {exc}")
        manifest = {}
    return manifest, errors


def validate_composable_stimulus_selection_run(
    run_path: str | Path,
    *,
    expected_compiled_selection: CompiledSelection | None = None,
    expected_request_digest: str | None = None,
    expected_resolved_digest: str | None = None,
) -> dict[str, Any]:
    """Validate one local or published selection run without repairing it."""

    errors: list[str] = []
    try:
        group = open_zarr_root(Path(run_path), mode="r", use_consolidated=False)
        attrs = dict(group.attrs)
    except Exception as exc:
        return {"valid": False, "errors": [f"cannot open run: {exc}"]}
    if attrs.get("schema_id") != RUN_SCHEMA_ID:
        errors.append("invalid selection-run schema")
    if attrs.get("stage_selector_eligible") is not False:
        errors.append("run is not permanently selector-ineligible")
    if attrs.get("selector_policy") != SELECTOR_INELIGIBLE_POLICY:
        errors.append("invalid selector policy")
    if attrs.get("retry_policy") != RETRY_POLICY:
        errors.append("invalid retry policy")
    if attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        errors.append("run is not complete")
    if any(_selector_alias_attr_name(key) for key in attrs):
        errors.append("run contains a selector alias attribute")
    for name in (REQUESTED_JSON_ATTR, RESOLVED_JSON_ATTR, TIMELINE_AUTHORITY_JSON_ATTR):
        try:
            _strict_json_attr(attrs, name)
        except ValueError as exc:
            errors.append(str(exc))
    try:
        compiled = reconstruct_compiled_selection(run_path)
        resolved_json = canonical_json(compiled.resolved_payload())
        requested_json = canonical_json(compiled.requested)
        if attrs.get(REQUESTED_JSON_ATTR) != requested_json:
            errors.append("requested selection JSON does not reconstruct exactly")
        if attrs.get(RESOLVED_JSON_ATTR) != resolved_json:
            errors.append("resolved selection JSON does not reconstruct exactly")
        if attrs.get("timeline_authority_sha256") != canonical_sha256(
            compiled.authority.to_dict()
        ):
            errors.append("timeline authority digest mismatch")
        if attrs.get("request_digest") != compiled.request_digest:
            errors.append("request digest does not match reconstructed selection")
        if attrs.get("resolved_digest") != compiled.resolved_digest:
            errors.append("resolved digest does not match reconstructed selection")
        if attrs.get(COMPILED_SELECTION_DIGEST_ATTR) != _compiled_digest(compiled):
            errors.append("compiled selection digest mismatch")
        if expected_compiled_selection is not None:
            try:
                _assert_plan_matches_compiled(
                    ComposableStimulusSelectionMaterializationPlan(
                        source_zarr=Path("."),
                        scratch_root=Path("."),
                        local_zarr=Path("."),
                        run_name=str(attrs.get(RUN_NAME_ATTR)),
                        request_digest=compiled.request_digest,
                        resolved_digest=compiled.resolved_digest,
                        compiled_selection_digest=_compiled_digest(compiled),
                        requested_json=requested_json,
                        resolved_json=resolved_json,
                        timeline_authority_json=canonical_json(
                            compiled.authority.to_dict()
                        ),
                        parent_selector_snapshot={},
                    ),
                    expected_compiled_selection,
                )
            except (TypeError, ValueError) as exc:
                errors.append(f"compiled selection differs from run: {exc}")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"normalized arrays cannot reconstruct selection: {exc}")
        compiled = None
    if expected_request_digest is not None and attrs.get("request_digest") != expected_request_digest:
        errors.append("stale request digest")
    if expected_resolved_digest is not None and attrs.get("resolved_digest") != expected_resolved_digest:
        errors.append("stale resolved digest")
    _manifest, manifest_errors = _validate_array_manifest(group)
    errors.extend(manifest_errors)
    return {
        "valid": not errors,
        "errors": errors,
        "array_count": len(_EXPECTED_ARRAY_NAMES),
        "request_digest": attrs.get("request_digest"),
        "resolved_digest": attrs.get("resolved_digest"),
        "logical_array_manifest_sha256": attrs.get(ARRAY_MANIFEST_DIGEST_ATTR),
    }


def _write_local_run(
    plan: ComposableStimulusSelectionMaterializationPlan,
    compiled: CompiledSelection,
) -> dict[str, Any]:
    _assert_plan_matches_compiled(plan, compiled)
    if plan.local_zarr.exists():
        raise FileExistsError(f"Refusing existing local Zarr: {plan.local_zarr}")
    plan.scratch_root.mkdir(parents=True, exist_ok=True)
    local_root = open_zarr_root(plan.local_zarr, mode="w-")
    parent = _require_provenance_parent(local_root)
    run = parent.create_group(plan.run_name)
    run.attrs.update(
        {
            "schema_id": RUN_SCHEMA_ID,
            "schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "selection_id": compiled.selection_id,
            "aggregation_policy": compiled.aggregation_policy,
            "selection_schema_id": "palette.composable_stimulus_selection_request.v1",
            "request_digest": compiled.request_digest,
            "resolved_digest": compiled.resolved_digest,
            COMPILED_SELECTION_DIGEST_ATTR: _compiled_digest(compiled),
            REQUESTED_JSON_ATTR: plan.requested_json,
            RESOLVED_JSON_ATTR: plan.resolved_json,
            TIMELINE_AUTHORITY_JSON_ATTR: plan.timeline_authority_json,
            "timeline_authority_sha256": canonical_sha256(
                compiled.authority.to_dict()
            ),
            "interval_policy_id": "half_open_acquisition_frame_v1",
            "selector_policy": SELECTOR_INELIGIBLE_POLICY,
            "stage_selector_eligible": False,
            "retry_policy": RETRY_POLICY,
            "run_path_policy": RUN_PATH_POLICY,
            "manifest_status": "pending_manifest_last_validation",
        }
    )
    mark_run_started(
        run,
        run_name=plan.run_name,
        stage="composable_stimulus_selection_materialization",
    )
    arrays = _selection_arrays(compiled)
    for name in sorted(arrays):
        create_geometry_preload_array(run, name, data=arrays[name], overwrite=False)
    # The run is intentionally incomplete until the exact manifest exists.
    # Reading every array here catches malformed physical writes while the
    # manifest itself is still absent; the full contract is validated below.
    for name in _EXPECTED_ARRAY_NAMES:
        _array_content_digest(run[name])
    manifest = _build_array_manifest(run)
    run.attrs[ARRAY_MANIFEST_JSON_ATTR] = canonical_json(manifest)
    run.attrs[ARRAY_MANIFEST_DIGEST_ATTR] = canonical_sha256(manifest)
    run.attrs["manifest_status"] = "complete_manifest_last"
    provenance = build_writer_run_provenance(
        command="composable_stimulus_selection_materializer",
        params={
            "run_name": plan.run_name,
            "selection_id": compiled.selection_id,
            "request_digest": compiled.request_digest,
            "resolved_digest": compiled.resolved_digest,
            "retry_policy": RETRY_POLICY,
        },
        input_run_ids={"compiled_selection": compiled.selection_id},
        cwd=Path(__file__).resolve().parents[4],
    )
    run.attrs["run_provenance"] = json_attr_safe(provenance)
    mark_run_complete(
        run,
        parent_group=parent,
        run_name=plan.run_name,
        run_provenance=provenance,
    )
    run.attrs["stage_selector_eligible"] = False
    local_validation = validate_composable_stimulus_selection_run(
        plan.local_run_path,
        expected_compiled_selection=compiled,
        expected_request_digest=plan.request_digest,
        expected_resolved_digest=plan.resolved_digest,
    )
    if not local_validation["valid"]:
        raise RuntimeError(f"Local selection run validation failed: {local_validation}")
    consolidate_metadata_capture_expected_warnings(plan.local_zarr)
    local_metadata = validate_direct_consolidated_subtree(
        plan.local_zarr,
        subtree_path=plan.run_path,
    )
    consolidated_validation = validate_composable_stimulus_selection_run(
        plan.local_run_path,
        expected_compiled_selection=compiled,
    )
    if not consolidated_validation["valid"]:
        raise RuntimeError(
            f"Consolidated local selection run validation failed: {consolidated_validation}"
        )
    return {
        "local_validation": local_validation,
        "local_direct_consolidated": local_metadata.to_json(),
        "local_consolidated_validation": consolidated_validation,
        "run_provenance": provenance,
    }


def _publish_local_run(
    plan: ComposableStimulusSelectionMaterializationPlan,
    *,
    copy_backend: str,
    local_result: Mapping[str, Any],
) -> dict[str, Any]:
    def prepare(root: zarr.Group) -> tuple[zarr.Group]:
        return (_require_provenance_parent(root),)

    def complete(_root: zarr.Group, parent: zarr.Group, run: zarr.Group) -> None:
        provenance = run.attrs.get("run_provenance")
        if not isinstance(provenance, Mapping):
            raise RuntimeError("published selection run lacks run provenance")
        mark_run_complete(
            run,
            parent_group=parent,
            run_name=plan.run_name,
            run_provenance=provenance,
        )
        run.attrs["stage_selector_eligible"] = False

    def verify(root: zarr.Group) -> None:
        _require_same_selector_snapshot(root, plan.parent_selector_snapshot)
        parent = _group_at(root, PARENT_PATH)
        if parent is None or plan.run_name not in parent:
            raise RuntimeError("published stimulus-selection run is missing")
        run = parent[plan.run_name]
        if (
            run.attrs.get("stage_selector_eligible") is not False
            or run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        ):
            raise RuntimeError("published selection run is not complete/ineligible")
        validation = validate_composable_stimulus_selection_run(
            plan.target_run_path,
            expected_request_digest=plan.request_digest,
            expected_resolved_digest=plan.resolved_digest,
        )
        if not validation["valid"]:
            raise RuntimeError(f"published selection run is invalid: {validation}")

    def seal_and_verify(root: zarr.Group, _parent: zarr.Group, run: zarr.Group) -> None:
        _require_same_selector_snapshot(root, plan.parent_selector_snapshot)
        consolidate_metadata_capture_expected_warnings(plan.source_zarr)
        metadata = validate_direct_consolidated_subtree(
            plan.source_zarr,
            subtree_path=plan.run_path,
        )
        run.attrs["direct_consolidated_subtree_receipt"] = metadata.to_json()
        # The receipt itself is part of the immutable run metadata, so refresh
        # the archive consolidation and compare the final declarations again.
        consolidate_metadata_capture_expected_warnings(plan.source_zarr)
        metadata = validate_direct_consolidated_subtree(
            plan.source_zarr,
            subtree_path=plan.run_path,
        )
        validation = validate_composable_stimulus_selection_run(
            plan.target_run_path,
            expected_request_digest=plan.request_digest,
            expected_resolved_digest=plan.resolved_digest,
        )
        if not validation["valid"]:
            raise RuntimeError(f"published selection run failed final validation: {validation}")
        if metadata.subtree_path != plan.run_path:
            raise RuntimeError("published selection subtree metadata path changed")

    return atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=plan.source_zarr,
            local_run_path=plan.local_run_path,
            target_run_path=plan.target_run_path,
            run_name=plan.run_name,
            lock_suffix="composable-stimulus-selection-materialization",
            publish_schema_id=PUBLISH_SCHEMA_ID,
            policy="node_local_plan_validate_manifest_last_atomic_nonpromoting_publish",
            rollback_policy="retain_failed_public_tombstone_leave_parent_selectors_untouched",
        ),
        copy_backend=copy_backend,
        validate_run=lambda path: validate_composable_stimulus_selection_run(
            path,
            expected_request_digest=plan.request_digest,
            expected_resolved_digest=plan.resolved_digest,
        ),
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=verify,
        activate_run=seal_and_verify,
        payload_metadata={
            "materialization": json_attr_safe(dict(plan.to_dict())),
            "local_result": json_attr_safe(dict(local_result)),
            "selector_ineligible": True,
            "selector_aliases_forbidden": sorted(_SELECTOR_ALIASES),
        },
    )


def materialize_composable_stimulus_selection_plan(
    plan: ComposableStimulusSelectionMaterializationPlan,
    compiled_selection: CompiledSelection,
    *,
    copy_backend: str = "python",
    keep_scratch: bool = False,
) -> dict[str, Any]:
    """Materialize one previously planned selection and publish it atomically."""

    _assert_plan_matches_compiled(plan, compiled_selection)
    local_result = _write_local_run(plan, compiled_selection)
    try:
        publication = _publish_local_run(
            plan,
            copy_backend=copy_backend,
            local_result=local_result,
        )
        return {
            "status": "complete",
            "mutates_archive": True,
            "plan": plan.to_dict(),
            **local_result,
            "publication": publication,
        }
    finally:
        if not keep_scratch and plan.local_zarr.exists():
            shutil.rmtree(plan.local_zarr)
        if not keep_scratch:
            local_lock = archive_metadata_publication_lock_path(plan.local_zarr)
            if local_lock.exists():
                local_lock.unlink()
            try:
                plan.scratch_root.rmdir()
            except OSError:
                pass


def materialize_composable_stimulus_selection(
    source_zarr: str | Path,
    *,
    compiled_selection: CompiledSelection,
    scratch_root: str | Path,
    run_name: str,
    copy_backend: str = "python",
    apply: bool = True,
    keep_scratch: bool = False,
) -> dict[str, Any]:
    """Plan and optionally publish one selector-ineligible selection run."""

    plan = build_composable_stimulus_selection_materialization_plan(
        source_zarr,
        compiled_selection=compiled_selection,
        scratch_root=scratch_root,
        run_name=run_name,
    )
    if not apply:
        return {
            "status": "planned",
            "mutates_archive": False,
            "plan": plan.to_dict(),
        }
    return materialize_composable_stimulus_selection_plan(
        plan,
        compiled_selection,
        copy_backend=copy_backend,
        keep_scratch=keep_scratch,
    )


__all__ = [
    "ARRAY_MANIFEST_DIGEST_ATTR",
    "ARRAY_MANIFEST_JSON_ATTR",
    "ComposableStimulusSelectionMaterializationPlan",
    "MATERIALIZATION_SCHEMA_ID",
    "PARENT_PATH",
    "PUBLISH_SCHEMA_ID",
    "RETRY_POLICY",
    "RUN_SCHEMA_ID",
    "build_composable_stimulus_selection_materialization_plan",
    "materialize_composable_stimulus_selection",
    "materialize_composable_stimulus_selection_plan",
    "reconstruct_compiled_selection",
    "validate_composable_stimulus_selection_run",
]
