"""Immutable selector-ineligible publication for semantic chaser selections."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.analysis.provider_chaser_position_suite import PositionSuiteEpoch
from fisheye.analysis_workflows.composable_epoch_selection_adapter import (
    EpochRoleBinding,
    _interval_by_binding,
    _require_selection,
    _selection_identity,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection import (
    ADAPTER_SCHEMA_ID,
    CHASER_WINDOW_ROLES,
    STANDALONE_SOLID_BLACK_ROLE,
    STEP_END_EXCLUSIVE,
    STEP_END_INCLUSIVE,
    STEP_END_PENDING,
    ProtocolSemanticChaserSelections,
    compile_protocol_semantic_chaser_selections,
    load_protocol_semantic_selection_evidence,
    load_protocol_semantic_timeline_evidence,
)
from fisheye.analysis_workflows.resolved_epoch_selection import (
    ResolvedEpochSelection,
    resolve_exact_stimulus_epoch_selection,
)
from fisheye.analysis_workflows.exact_immutable_child_validation_receipt import (
    VERIFICATION_MODE as EXACT_CHILD_VERIFICATION_MODE,
    read_exact_immutable_child_validation_receipt,
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


RUNS_PARENT_PATH = "analysis/protocol_semantic_chaser_selection_runs"
RUNS_PREFIX = f"{RUNS_PARENT_PATH}/"
STORAGE_SCHEMA_ID = "palette.protocol_semantic_chaser_selection_run"
STORAGE_SCHEMA_VERSION = 1
STORAGE_LAYOUT = "fixed_role_interval_rows_v1"
PUBLICATION_SCHEMA_ID = f"{STORAGE_SCHEMA_ID}.publication"
PUBLICATION_SCHEMA_VERSION = 1
MANIFEST_ATTR = "protocol_semantic_chaser_selection_manifest"
MANIFEST_DIGEST_ATTR = "protocol_semantic_chaser_selection_manifest_sha256"
PUBLICATION_POLICY = "immutable_semantic_hierarchy_selector_ineligible_v1"
MAX_MANIFEST_BYTES = 131_072

_ARRAY_DTYPES = MappingProxyType(
    {
        "role_code": np.dtype(np.int32),
        "source_window_id": np.dtype(np.int64),
        "selected_start_frame": np.dtype(np.int64),
        "selected_end_frame_exclusive": np.dtype(np.int64),
        "protocol_semantic_step_index": np.dtype(np.int64),
        "terminal_frame_excluded": np.dtype(bool),
    }
)
_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SELECTOR_ALIASES = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_pending",
        "current",
        "default",
        "selected",
        "authoritative_run",
    }
)
_SOURCE_HANDLE_SEAL = object()


class ProtocolSemanticChaserSelectionPublicationError(ValueError):
    """Raised when a semantic selection publication is incomplete or stale."""


def _fail(message: str) -> None:
    raise ProtocolSemanticChaserSelectionPublicationError(message)


def _copy_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _copy_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _strict_json_object(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be one JSON object.")
    try:
        normalized = _copy_json(value)
        decoded = json.loads(
            json.dumps(
                normalized,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolSemanticChaserSelectionPublicationError(
            f"{name} must contain strict JSON."
        ) from exc
    if not isinstance(decoded, dict):  # pragma: no cover - defensive
        _fail(f"{name} must decode to one object.")
    return decoded


def _run_name(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or value.lower() in _SELECTOR_ALIASES
        or _RUN_NAME_RE.fullmatch(value) is None
    ):
        _fail("run_name must be one exact non-selector child name.")
    return value


def _archive(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Analysis Zarr archive does not exist: {path}")
    return path


def _digest(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{name} must be one lowercase SHA-256 digest.")
    return value


def _binding_from_record(value: object) -> EpochRoleBinding:
    record = _strict_json_object(value, name="role binding")
    if set(record) != {"window_id", "source_interval_digest"}:
        _fail("Semantic role binding has an inexact field set.")
    return EpochRoleBinding(
        window_id=record["window_id"],
        source_interval_digest=record["source_interval_digest"],
    )


def _role_record(
    *,
    role: str,
    compiled: Any,
    source_selection: ResolvedEpochSelection,
) -> dict[str, Any]:
    if compiled.selection_id != role or len(compiled.resolved_intervals) != 1:
        _fail(f"Semantic selection role {role!r} is not one exact interval.")
    resolved = compiled.resolved_intervals[0]
    if len(resolved.source_memberships) != 1:
        _fail(f"Semantic selection role {role!r} has inexact source membership.")
    membership = resolved.source_memberships[0]
    if membership.role is None or membership.role.role != role:
        _fail(f"Semantic selection role {role!r} lost its exact role metadata.")
    metadata = _strict_json_object(
        membership.role.metadata,
        name=f"semantic selection metadata for {role}",
    )
    binding = _binding_from_record(metadata.get("role_binding"))
    source = _interval_by_binding(source_selection, binding)
    if (
        source.source_interval_digest != metadata.get("source_interval_digest")
        or source.occurrence_identity.get("occurrence_id")
        != membership.occurrence_id
        or membership.selected_start_frame != resolved.start_frame
        or membership.selected_end_frame != resolved.end_frame
    ):
        _fail(f"Semantic selection role {role!r} differs from its exact source.")
    step_index = metadata.get("protocol_semantic_step_index")
    terminal_excluded = metadata.get(
        "terminal_frame_excluded_pending_step_end_contract"
    )
    if type(step_index) is not int or type(terminal_excluded) is not bool:
        _fail(f"Semantic selection role {role!r} lacks containment evidence.")
    return {
        "role": role,
        "role_binding": binding.to_dict(),
        "source_window_id": int(source.window_id),
        "source_label": str(source.label),
        "source_interval_sha256": str(source.source_interval_digest),
        "source_occurrence_id": str(membership.occurrence_id),
        "source_start_frame": int(source.start_frame),
        "source_end_frame_exclusive": int(source.end_frame),
        "selected_start_frame": int(resolved.start_frame),
        "selected_end_frame_exclusive": int(resolved.end_frame),
        "protocol_semantic_step_index": step_index,
        "terminal_frame_excluded_pending_step_end_contract": terminal_excluded,
        "request_sha256": str(compiled.request_digest),
        "resolved_sha256": str(compiled.resolved_digest),
    }


def _role_records(
    selections: ProtocolSemanticChaserSelections,
    source_selection: ResolvedEpochSelection,
) -> tuple[dict[str, Any], ...]:
    roles = list(CHASER_WINDOW_ROLES)
    if selections.standalone_solid_black is not None:
        roles.insert(0, STANDALONE_SOLID_BLACK_ROLE)
    return tuple(
        _role_record(
            role=role,
            compiled=selections.named[role],
            source_selection=source_selection,
        )
        for role in roles
    )


def _arrays(records: tuple[Mapping[str, Any], ...]) -> dict[str, np.ndarray]:
    return {
        "role_code": np.arange(len(records), dtype=np.int32),
        "source_window_id": np.asarray(
            [row["source_window_id"] for row in records], dtype=np.int64
        ),
        "selected_start_frame": np.asarray(
            [row["selected_start_frame"] for row in records], dtype=np.int64
        ),
        "selected_end_frame_exclusive": np.asarray(
            [row["selected_end_frame_exclusive"] for row in records],
            dtype=np.int64,
        ),
        "protocol_semantic_step_index": np.asarray(
            [row["protocol_semantic_step_index"] for row in records],
            dtype=np.int64,
        ),
        "terminal_frame_excluded": np.asarray(
            [
                row[
                    "terminal_frame_excluded_pending_step_end_contract"
                ]
                for row in records
            ],
            dtype=bool,
        ),
    }


def _array_declarations(arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    return [
        {
            "path": name,
            "dtype": np.asarray(arrays[name]).dtype.str,
            "shape": list(np.asarray(arrays[name]).shape),
            "content_sha256": array_values_sha256(np.asarray(arrays[name])),
        }
        for name in sorted(arrays)
    ]


@dataclass(frozen=True, slots=True)
class ProtocolSemanticChaserSelectionPublicationPlan:
    analysis_zarr: Path
    run_name: str
    run_path: str
    recording_id: str
    source_selection_run_name: str
    manifest: Mapping[str, Any] = field(repr=False)
    arrays: Mapping[str, np.ndarray] = field(repr=False, compare=False)
    run_provenance: Mapping[str, Any] = field(repr=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_id": PUBLICATION_SCHEMA_ID,
            "schema_version": PUBLICATION_SCHEMA_VERSION,
            "status": "dry_run_plan",
            "analysis_zarr": str(self.analysis_zarr),
            "recording_id": self.recording_id,
            "run_name": self.run_name,
            "run_path": self.run_path,
            "manifest_sha256": canonical_json_sha256(dict(self.manifest)),
            "role_count": len(self.manifest["role_records"]),
            "roles": list(self.manifest["role_order"]),
            "selector_eligible": False,
            "selection": "none",
            "production_authority": False,
            "registry_update": False,
            "target_exists": (self.analysis_zarr / self.run_path).exists(),
        }


def build_protocol_semantic_chaser_selection_publication_plan(
    analysis_zarr: str | Path,
    *,
    selections: ProtocolSemanticChaserSelections,
    source_selection: ResolvedEpochSelection,
    run_name: str,
) -> ProtocolSemanticChaserSelectionPublicationPlan:
    archive = _archive(analysis_zarr)
    name = _run_name(run_name)
    run_path = f"{RUNS_PREFIX}{name}"
    if (archive / run_path).exists():
        raise FileExistsError(f"Refusing to replace existing run: {archive / run_path}")
    if type(selections) is not ProtocolSemanticChaserSelections:
        _fail("selections must be one exact compiled semantic hierarchy.")
    semantics = selections.protocol_evidence.step_end_interval_semantics
    frame_bound_binding = (
        selections.protocol_evidence.frame_bound_acquisition_binding
    )
    if semantics not in {STEP_END_PENDING, STEP_END_EXCLUSIVE}:
        _fail(
            "Producer-declared step-end policies are not publishable until their "
            "acquisition authority is materialized as exact half-open v2 evidence."
        )
    if semantics == STEP_END_EXCLUSIVE and (
        selections.protocol_evidence.snapshot.snapshot_schema_version != 2
        or frame_bound_binding is None
    ):
        _fail(
            "Exact half-open semantic publication requires sealed frame-bound v2 evidence."
        )
    try:
        resolved = _require_selection(source_selection)
    except ValueError as exc:
        raise ProtocolSemanticChaserSelectionPublicationError(str(exc)) from exc
    current_evidence = load_protocol_semantic_selection_evidence(
        archive,
        resolved,
        step_end_interval_semantics=(
            selections.protocol_evidence.step_end_interval_semantics
        ),
        frame_bound_source_binding=frame_bound_binding,
    )
    if current_evidence.to_dict() != selections.protocol_evidence.to_dict():
        _fail("Compiled semantic evidence differs from the current exact source.")
    current_timeline = load_protocol_semantic_timeline_evidence(
        archive,
        resolved,
    )
    if current_timeline.to_dict() != selections.timeline_evidence.to_dict():
        _fail("Compiled timeline evidence differs from the current exact source.")
    identity = selections.identity_record()
    source_identity = _selection_identity(resolved)
    if identity.get("resolved_epoch_selection") != source_identity:
        _fail("Semantic hierarchy differs from its exact epoch selection.")
    role_records = _role_records(selections, resolved)
    arrays = _arrays(role_records)
    recording_id = str(selections.timeline_authority.recording_id)
    capabilities = [row.to_dict() for row in selections.capability_assessments()]
    payload: dict[str, Any] = {
        "schema_id": STORAGE_SCHEMA_ID,
        "schema_version": STORAGE_SCHEMA_VERSION,
        "storage_layout": STORAGE_LAYOUT,
        "adapter_schema_id": ADAPTER_SCHEMA_ID,
        "run_name": name,
        "run_path": run_path,
        "recording_id": recording_id,
        "status": "complete_selector_ineligible",
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
        "production_selector_activation": False,
        "registry_update": False,
        "publication_policy": PUBLICATION_POLICY,
        "selection_identity": identity,
        "selection_identity_sha256": selections.identity_sha256,
        "protocol_evidence": current_evidence.to_dict(),
        "protocol_evidence_sha256": canonical_json_sha256(
            current_evidence.to_dict()
        ),
        "timeline_evidence": current_timeline.to_dict(),
        "timeline_evidence_sha256": canonical_json_sha256(
            current_timeline.to_dict()
        ),
        "source_epoch_selection_run_name": resolved.run_name,
        "source_epoch_selection": source_identity,
        "role_order": [row["role"] for row in role_records],
        "role_records": list(role_records),
        "capability_assessments": capabilities,
        "array_declarations": _array_declarations(arrays),
    }
    manifest = {**payload, "payload_digest": canonical_json_sha256(payload)}
    encoded = json.dumps(manifest, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > MAX_MANIFEST_BYTES:
        _fail("Semantic selection manifest exceeds its bounded metadata limit.")
    provenance = build_writer_run_provenance(
        command=(
            "fisheye.analysis_workflows."
            "protocol_semantic_chaser_selection_publication"
        ),
        params={
            "run_name": name,
            "run_path": run_path,
            "publication_policy": PUBLICATION_POLICY,
            "selection_identity_sha256": selections.identity_sha256,
        },
        input_run_ids={
            "source_epoch_selection": resolved.run_path,
            "source_stimulus": current_evidence.source_stimulus_path,
        },
        cwd=Path(__file__).resolve().parents[3],
    )
    readonly = {}
    for array_name, values in arrays.items():
        copied = np.array(values, copy=True)
        copied.setflags(write=False)
        readonly[array_name] = copied
    return ProtocolSemanticChaserSelectionPublicationPlan(
        analysis_zarr=archive,
        run_name=name,
        run_path=run_path,
        recording_id=recording_id,
        source_selection_run_name=resolved.run_name,
        manifest=MappingProxyType(_copy_json(manifest)),
        arrays=MappingProxyType(readonly),
        run_provenance=MappingProxyType(_copy_json(provenance)),
    )


def _manifest_from_run(run: Any) -> tuple[dict[str, Any], str]:
    manifest = _strict_json_object(run.attrs.get(MANIFEST_ATTR), name=MANIFEST_ATTR)
    digest = canonical_json_sha256(manifest)
    if run.attrs.get(MANIFEST_DIGEST_ATTR) != digest:
        _fail("Persistent semantic selection manifest digest is stale.")
    payload = {key: value for key, value in manifest.items() if key != "payload_digest"}
    if canonical_json_sha256(payload) != manifest.get("payload_digest"):
        _fail("Persistent semantic selection manifest payload digest is stale.")
    return manifest, digest


def _validate_role_rows(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    records = manifest.get("role_records")
    order = manifest.get("role_order")
    if (
        not isinstance(records, list)
        or not records
        or any(not isinstance(record, Mapping) for record in records)
        or not isinstance(order, list)
        or order != [record.get("role") for record in records]
    ):
        _fail("Persistent semantic selection role registry is malformed.")
    identity = manifest.get("selection_identity")
    if not isinstance(identity, Mapping):
        _fail("Persistent semantic selection identity is malformed.")
    expected_order = list(CHASER_WINDOW_ROLES)
    if identity.get("standalone_solid_black_status") == "selected":
        expected_order.insert(0, STANDALONE_SOLID_BLACK_ROLE)
    if order != expected_order or len(set(order)) != len(order):
        _fail("Persistent semantic selection has an inexact role set or order.")
    for field_name in (
        "source_window_id",
        "source_interval_sha256",
        "source_occurrence_id",
    ):
        values = [record.get(field_name) for record in records]
        if len(set(values)) != len(values):
            _fail(
                f"Persistent semantic selection has duplicate {field_name!r} values."
            )
    expected = {
        "role_code": np.arange(len(records), dtype=np.int32),
        "source_window_id": np.asarray(
            [record["source_window_id"] for record in records], dtype=np.int64
        ),
        "selected_start_frame": np.asarray(
            [record["selected_start_frame"] for record in records], dtype=np.int64
        ),
        "selected_end_frame_exclusive": np.asarray(
            [record["selected_end_frame_exclusive"] for record in records],
            dtype=np.int64,
        ),
        "protocol_semantic_step_index": np.asarray(
            [record["protocol_semantic_step_index"] for record in records],
            dtype=np.int64,
        ),
        "terminal_frame_excluded": np.asarray(
            [
                record[
                    "terminal_frame_excluded_pending_step_end_contract"
                ]
                for record in records
            ],
            dtype=bool,
        ),
    }
    for name, values in expected.items():
        if not np.array_equal(arrays[name], values):
            _fail(f"Persistent semantic selection array {name!r} is stale.")


def _validate_current_sources(
    archive: Path,
    manifest: Mapping[str, Any],
) -> ResolvedEpochSelection:
    run_name = manifest.get("source_epoch_selection_run_name")
    if type(run_name) is not str:
        _fail("Semantic selection source epoch run name is malformed.")
    try:
        selection = resolve_exact_stimulus_epoch_selection(archive, run_name=run_name)
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ProtocolSemanticChaserSelectionPublicationError(
            "Semantic selection source fingerprint differs from its published "
            f"authority: {exc}"
        ) from exc
    if _selection_identity(selection) != manifest.get("source_epoch_selection"):
        _fail("Semantic selection source epoch authority changed.")
    evidence_record = _strict_json_object(
        manifest.get("protocol_evidence"),
        name="protocol_evidence",
    )
    current_evidence = load_protocol_semantic_selection_evidence(
        archive,
        selection,
        step_end_interval_semantics=evidence_record.get(
            "step_end_interval_semantics"
        ),
        frame_bound_source_binding=evidence_record.get(
            "frame_bound_acquisition_binding"
        ),
    )
    if current_evidence.to_dict() != evidence_record:
        _fail("Semantic selection protocol evidence changed.")
    if canonical_json_sha256(evidence_record) != manifest.get(
        "protocol_evidence_sha256"
    ):
        _fail("Semantic selection protocol-evidence digest is stale.")
    timeline_record = _strict_json_object(
        manifest.get("timeline_evidence"),
        name="timeline_evidence",
    )
    current_timeline_evidence = load_protocol_semantic_timeline_evidence(
        archive,
        selection,
    )
    current_timeline = current_timeline_evidence.to_dict()
    if current_timeline != timeline_record:
        _fail("Semantic selection timeline evidence changed.")
    if canonical_json_sha256(timeline_record) != manifest.get(
        "timeline_evidence_sha256"
    ):
        _fail("Semantic selection timeline-evidence digest is stale.")
    bounds_by_index = current_evidence.bounds_by_step_index
    selection_identity = _strict_json_object(
        manifest.get("selection_identity"),
        name="selection_identity",
    )
    chaser_step = _strict_json_object(
        selection_identity.get("chaser_step"),
        name="selection_identity.chaser_step",
    )
    chaser_identity = _strict_json_object(
        chaser_step.get("identity"),
        name="selection_identity.chaser_step.identity",
    )
    chaser_step_index = chaser_identity.get("step_index")
    standalone_step = selection_identity.get("standalone_solid_black_step")
    standalone_step_index: int | None = None
    if standalone_step is not None:
        standalone_record = _strict_json_object(
            standalone_step,
            name="selection_identity.standalone_solid_black_step",
        )
        standalone_identity = _strict_json_object(
            standalone_record.get("identity"),
            name="selection_identity.standalone_solid_black_step.identity",
        )
        standalone_step_index = standalone_identity.get("step_index")
    if type(chaser_step_index) is not int or (
        standalone_step_index is not None and type(standalone_step_index) is not int
    ):
        _fail("Semantic selection identity has malformed producer step indices.")
    records = manifest["role_records"]
    role_bindings = {
        str(record["role"]): _binding_from_record(record.get("role_binding"))
        for record in records
    }
    try:
        recompiled = compile_protocol_semantic_chaser_selections(
            selection,
            timeline_evidence=current_timeline_evidence,
            protocol_evidence=current_evidence,
            role_bindings=role_bindings,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolSemanticChaserSelectionPublicationError(
            f"Semantic selection hierarchy cannot be recompiled: {exc}"
        ) from exc
    if recompiled.identity_record() != selection_identity:
        _fail("Semantic selection identity differs from current recompilation.")
    if list(_role_records(recompiled, selection)) != records:
        _fail("Semantic selection role records differ from current recompilation.")
    for record in records:
        binding = _binding_from_record(record.get("role_binding"))
        source = _interval_by_binding(selection, binding)
        step_index = record.get("protocol_semantic_step_index")
        if type(step_index) is not int or step_index not in bounds_by_index:
            _fail("Semantic selection role points at an unknown protocol step.")
        expected_step_index = (
            standalone_step_index
            if record.get("role") == STANDALONE_SOLID_BLACK_ROLE
            else chaser_step_index
        )
        if step_index != expected_step_index:
            _fail("Semantic selection role points at the wrong producer step.")
        raw = bounds_by_index[step_index]
        if current_evidence.step_end_interval_semantics == STEP_END_INCLUSIVE:
            usable_end = raw.end_camera_frame + 1
        else:
            usable_end = raw.end_camera_frame
        if int(source.start_frame) < raw.start_camera_frame:
            _fail("Semantic selection source interval crosses its protocol step.")
        expected_end = int(source.end_frame)
        terminal_excluded = False
        if expected_end > usable_end:
            if (
                current_evidence.step_end_interval_semantics == STEP_END_PENDING
                and expected_end == usable_end + 1
            ):
                expected_end = usable_end
                terminal_excluded = True
            else:
                _fail("Semantic selection source interval crosses its protocol step.")
        expected = {
            "source_window_id": int(source.window_id),
            "source_label": str(source.label),
            "source_interval_sha256": str(source.source_interval_digest),
            "source_occurrence_id": str(
                source.occurrence_identity.get("occurrence_id")
            ),
            "source_start_frame": int(source.start_frame),
            "source_end_frame_exclusive": int(source.end_frame),
            "selected_start_frame": int(source.start_frame),
            "selected_end_frame_exclusive": expected_end,
            "terminal_frame_excluded_pending_step_end_contract": terminal_excluded,
        }
        for field_name, value in expected.items():
            if record.get(field_name) != value:
                _fail(
                    f"Semantic selection role field {field_name!r} differs from "
                    "its current exact source."
                )
    return selection


def _validate_persistent_run(
    path: Path,
    *,
    expected_manifest: Mapping[str, Any] | None = None,
    expected_run_path: str | None = None,
    verify_content_hashes: bool = False,
    run: Any | None = None,
    source_archive: Path | None = None,
) -> dict[str, Any]:
    if run is None:
        run = open_zarr_root(path, mode="r", use_consolidated=False)
    manifest, manifest_sha256 = _manifest_from_run(run)
    if expected_manifest is not None and manifest != dict(expected_manifest):
        _fail("Persistent semantic selection manifest differs from its plan.")
    if expected_run_path is not None and manifest.get("run_path") != expected_run_path:
        _fail("Persistent semantic selection run path is stale.")
    if (
        manifest.get("schema_id") != STORAGE_SCHEMA_ID
        or manifest.get("schema_version") != STORAGE_SCHEMA_VERSION
        or manifest.get("storage_layout") != STORAGE_LAYOUT
        or manifest.get("adapter_schema_id") != ADAPTER_SCHEMA_ID
    ):
        _fail("Persistent semantic selection schema identity is invalid.")
    if (
        manifest.get("selector_eligible") is not False
        or manifest.get("selection") != "none"
        or manifest.get("production_authority") is not False
        or manifest.get("production_selector_activation") is not False
        or manifest.get("registry_update") is not False
        or run.attrs.get("stage_selector_eligible") is not False
        or run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
    ):
        _fail("Persistent semantic selection is not complete and selector-ineligible.")
    identity = _strict_json_object(
        manifest.get("selection_identity"),
        name="selection_identity",
    )
    if canonical_json_sha256(identity) != manifest.get("selection_identity_sha256"):
        _fail("Persistent semantic selection identity digest is stale.")
    declarations = manifest.get("array_declarations")
    if not isinstance(declarations, list):
        _fail("Persistent semantic selection array declarations are absent.")
    if set(run.array_keys()) != set(_ARRAY_DTYPES):
        _fail("Persistent semantic selection has unexpected or missing arrays.")
    arrays: dict[str, np.ndarray] = {}
    role_count = len(manifest.get("role_records", []))
    by_name = {row.get("path"): row for row in declarations}
    if set(by_name) != set(_ARRAY_DTYPES):
        _fail("Persistent semantic selection array inventory is stale.")
    for name, expected_dtype in _ARRAY_DTYPES.items():
        values = np.asarray(run[name][:])
        declaration = by_name[name]
        if (
            values.dtype != expected_dtype
            or values.shape != (role_count,)
            or declaration.get("dtype") != values.dtype.str
            or declaration.get("shape") != [role_count]
        ):
            _fail(f"Persistent semantic selection array {name!r} is malformed.")
        if verify_content_hashes and array_values_sha256(values) != declaration.get(
            "content_sha256"
        ):
            _fail(f"Persistent semantic selection array {name!r} digest is stale.")
        copied = np.array(values, copy=True)
        copied.setflags(write=False)
        arrays[name] = copied
    _validate_role_rows(manifest, arrays)
    if source_archive is not None:
        _validate_current_sources(source_archive, manifest)
    return {
        "valid": True,
        "manifest_sha256": manifest_sha256,
        "run_path": manifest["run_path"],
        "role_count": role_count,
        "arrays": arrays,
    }


def _write_local_run(
    plan: ProtocolSemanticChaserSelectionPublicationPlan,
    local_path: Path,
) -> None:
    if local_path.exists():
        raise FileExistsError(f"Local semantic selection path exists: {local_path}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    run = zarr.open_group(
        str(local_path),
        mode="w-",
        zarr_format=3,
        use_consolidated=False,
    )
    mark_run_started(run, run_name=plan.run_name, stage="protocol_semantic_selection")
    run.attrs.update(
        {
            "schema_id": STORAGE_SCHEMA_ID,
            "schema_version": STORAGE_SCHEMA_VERSION,
            "stage_selector_eligible": False,
            "selection": "none",
            "production_authority": False,
            "registry_update": False,
            "run_provenance": json_attr_safe(dict(plan.run_provenance)),
            MANIFEST_ATTR: json_attr_safe(dict(plan.manifest)),
            MANIFEST_DIGEST_ATTR: canonical_json_sha256(dict(plan.manifest)),
        }
    )
    for name, values in plan.arrays.items():
        run.create_array(name, data=np.asarray(values), chunks=(len(values),))
    mark_run_complete(
        run,
        run_name=plan.run_name,
        run_provenance=dict(plan.run_provenance),
    )
    _validate_persistent_run(
        local_path,
        expected_manifest=plan.manifest,
        expected_run_path=plan.run_path,
        verify_content_hashes=True,
    )


def publish_protocol_semantic_chaser_selection_run(
    plan: ProtocolSemanticChaserSelectionPublicationPlan,
    *,
    scratch_root: str | Path | None = None,
    copy_backend: str = "python",
) -> dict[str, Any]:
    """Atomically publish one immutable semantic hierarchy without selectors."""

    scratch = (
        Path(scratch_root).expanduser().resolve() if scratch_root is not None else None
    )
    if scratch is not None:
        scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{plan.run_name}.protocol-semantic-selection.",
        dir=str(scratch) if scratch is not None else None,
    ) as temporary:
        local_path = Path(temporary) / "run.zarr"
        _write_local_run(plan, local_path)
        parent_snapshot: dict[str, Any] | None = None

        def validate(path: Path) -> Mapping[str, Any]:
            result = _validate_persistent_run(
                path,
                expected_manifest=plan.manifest,
                expected_run_path=plan.run_path,
                verify_content_hashes=True,
            )
            return {
                key: value for key, value in result.items() if key != "arrays"
            }

        def prepare(root: Any) -> tuple[Any]:
            nonlocal parent_snapshot
            parent = require_runs_parent(
                root.require_group("analysis"),
                "protocol_semantic_chaser_selection_runs",
                completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
            )
            if set(parent.attrs).intersection(_SELECTOR_ALIASES):
                _fail("Semantic selection parent contains forbidden selectors.")
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
                _fail("Semantic selection publication changed parent metadata.")
            _validate_persistent_run(
                plan.analysis_zarr / plan.run_path,
                expected_manifest=plan.manifest,
                expected_run_path=plan.run_path,
                source_archive=plan.analysis_zarr,
            )

        publication = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=plan.analysis_zarr,
                local_run_path=local_path,
                target_run_path=plan.analysis_zarr / plan.run_path,
                run_name=plan.run_name,
                lock_suffix="protocol-semantic-chaser-selection-publication",
                publish_schema_id=PUBLICATION_SCHEMA_ID,
                policy=PUBLICATION_POLICY,
                rollback_policy="retain_failed_selector_ineligible_tombstone_v1",
                content_checksum=True,
            ),
            copy_backend=copy_backend,
            validate_run=validate,
            prepare_parents=prepare,
            complete_run=complete,
            verify_pointers=verify,
            payload_metadata={
                "recording_id": plan.recording_id,
                "run_path": plan.run_path,
                "selection_identity_sha256": plan.manifest[
                    "selection_identity_sha256"
                ],
                "selector_activation": "none",
            },
        )
    consolidation = consolidate_metadata_capture_expected_warnings(plan.analysis_zarr)
    metadata = validate_direct_consolidated_subtree(
        plan.analysis_zarr,
        subtree_path=plan.run_path,
    ).to_json()
    handle = load_protocol_semantic_chaser_selection_source_handle(
        plan.analysis_zarr,
        run_name=plan.run_name,
        expected_recording_id=plan.recording_id,
    )
    return {
        "status": "published_selector_ineligible",
        "run_path": handle.run_path,
        "manifest_sha256": handle.manifest_sha256,
        "selection_identity_sha256": handle.selection_identity_sha256,
        "role_count": len(handle.role_records),
        "roles": list(handle.role_records),
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
        "registry_update": False,
        "consolidation": consolidation,
        "metadata_equivalence": metadata,
        "atomic_publication": publication,
    }


@dataclass(frozen=True, slots=True, init=False)
class ProtocolSemanticChaserSelectionSourceHandle:
    """Read-only loader-minted semantic hierarchy for downstream candidates."""

    analysis_zarr: Path
    run_name: str
    run_path: str
    recording_id: str
    manifest: Mapping[str, Any] = field(repr=False)
    arrays: Mapping[str, np.ndarray] = field(repr=False, compare=False)
    metadata_equivalence: Mapping[str, Any] = field(repr=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self, *, _seal: object | None = None, **values: Any) -> None:
        if _seal is not _SOURCE_HANDLE_SEAL:
            raise TypeError("Semantic selection handles require their strict loader.")
        for name, value in values.items():
            if name in {"manifest", "metadata_equivalence"}:
                value = MappingProxyType(_copy_json(value))
            elif name == "arrays":
                value = MappingProxyType(
                    {key: np.array(array, copy=True) for key, array in value.items()}
                )
                for array in value.values():
                    array.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _SOURCE_HANDLE_SEAL)

    @property
    def manifest_sha256(self) -> str:
        return canonical_json_sha256(dict(self.manifest))

    @property
    def selection_identity_sha256(self) -> str:
        return str(self.manifest["selection_identity_sha256"])

    @property
    def protocol_semantic_hash(self) -> str:
        return str(self.manifest["selection_identity"]["protocol_semantic_hash"])

    @property
    def standalone_solid_black_status(self) -> str:
        return str(
            self.manifest["selection_identity"]["standalone_solid_black_status"]
        )

    @property
    def role_records(self) -> Mapping[str, Mapping[str, Any]]:
        return MappingProxyType(
            {
                str(record["role"]): MappingProxyType(_copy_json(record))
                for record in self.manifest["role_records"]
            }
        )

    def position_suite_epochs(self) -> tuple[PositionSuiteEpoch, ...]:
        records = self.role_records
        return tuple(
            PositionSuiteEpoch(
                analysis_role=role,
                window_id=int(records[role]["source_window_id"]),
                source_label=str(records[role]["source_label"]),
                start_frame=int(records[role]["selected_start_frame"]),
                end_frame=int(records[role]["selected_end_frame_exclusive"]),
                source_interval_sha256=str(
                    records[role]["source_interval_sha256"]
                ),
            )
            for role in CHASER_WINDOW_ROLES
        )

    def position_suite_epoch_records(self) -> list[dict[str, Any]]:
        return [
            {
                "analysis_role": epoch.analysis_role,
                "window_id": epoch.window_id,
                "source_label": epoch.source_label,
                "start_frame": epoch.start_frame,
                "end_frame_exclusive": epoch.end_frame,
                "source_interval_sha256": epoch.source_interval_sha256,
            }
            for epoch in self.position_suite_epochs()
        ]

    def source_binding(self) -> dict[str, Any]:
        position_epochs = self.position_suite_epoch_records()
        records = self.role_records
        semantic_roles = [
            {
                "analysis_role": role,
                "source_window_id": records[role]["source_window_id"],
                "source_interval_sha256": records[role][
                    "source_interval_sha256"
                ],
                "selected_start_frame": records[role]["selected_start_frame"],
                "selected_end_frame_exclusive": records[role][
                    "selected_end_frame_exclusive"
                ],
                "protocol_semantic_hash": self.protocol_semantic_hash,
                "protocol_semantic_step_index": records[role][
                    "protocol_semantic_step_index"
                ],
                "protocol_semantic_step_ref": (
                    "protocol_semantic_snapshot@recipe.steps"
                    f"[{records[role]['protocol_semantic_step_index']}]"
                ),
                "terminal_frame_excluded_pending_step_end_contract": records[
                    role
                ]["terminal_frame_excluded_pending_step_end_contract"],
            }
            for role in CHASER_WINDOW_ROLES
        ]
        return {
            "run_name": self.run_name,
            "run_path": self.run_path,
            "manifest_sha256": self.manifest_sha256,
            "selection_identity_sha256": self.selection_identity_sha256,
            "protocol_semantic_hash": self.protocol_semantic_hash,
            "palette_computed_trial_index_sha256": self.manifest[
                "selection_identity"
            ]["palette_computed_trial_index_sha256"],
            "trial_index_integrity_status": self.manifest[
                "selection_identity"
            ]["trial_index_integrity_status"],
            "standalone_solid_black_status": self.standalone_solid_black_status,
            "step_end_interval_semantics": self.manifest["protocol_evidence"][
                "step_end_interval_semantics"
            ],
            "source_epoch_selection": _copy_json(
                self.manifest["source_epoch_selection"]
            ),
            "roles": list(CHASER_WINDOW_ROLES),
            "position_suite_epochs": position_epochs,
            "position_suite_epochs_sha256": canonical_json_sha256(
                position_epochs
            ),
            "semantic_role_bindings": semantic_roles,
            "semantic_role_bindings_sha256": canonical_json_sha256(
                semantic_roles
            ),
            "position_suite_scope": {
                "analysis_epoch_scope": "chaser_internal_windows",
                "behavior_role_contrast_scope": (
                    "within_epoch_treatment_minus_baseline"
                ),
                "standalone_protocol_baseline_included": False,
                "standalone_protocol_baseline_status": (
                    self.standalone_solid_black_status
                ),
            },
            "selector_eligible": False,
            "production_authority": False,
        }

    def assert_current(self) -> None:
        receipt_path = self.metadata_equivalence.get("receipt_path")
        refreshed = load_protocol_semantic_chaser_selection_source_handle(
            self.analysis_zarr,
            run_name=self.run_name,
            expected_recording_id=self.recording_id,
            direct_validation_receipt=(
                str(receipt_path) if receipt_path is not None else None
            ),
        )
        if refreshed.manifest_sha256 != self.manifest_sha256:
            _fail("Semantic selection publication changed after handle creation.")


def load_protocol_semantic_chaser_selection_source_handle(
    analysis_zarr: str | Path,
    *,
    run_name: str,
    expected_recording_id: str | None = None,
    use_consolidated: bool = True,
    deep_audit: bool = False,
    direct_validation_receipt: str | Path | None = None,
) -> ProtocolSemanticChaserSelectionSourceHandle:
    if type(use_consolidated) is not bool:
        _fail("use_consolidated must be one exact boolean.")
    archive = _archive(analysis_zarr)
    name = _run_name(run_name)
    run_path = f"{RUNS_PREFIX}{name}"
    try:
        if direct_validation_receipt is None:
            metadata = validate_direct_consolidated_subtree(
                archive,
                subtree_path=run_path,
            ).to_json()
            root = open_zarr_root(
                archive,
                mode="r",
                use_consolidated=use_consolidated,
            )
            run = root[run_path]
            source_archive = archive
        else:
            receipt_path = Path(direct_validation_receipt).expanduser().resolve()
            receipt = read_exact_immutable_child_validation_receipt(
                receipt_path,
                expected_analysis_zarr=archive,
                expected_run_path=run_path,
                expected_recording_id=expected_recording_id,
                expected_manifest_attr=MANIFEST_ATTR,
                expected_manifest_digest_attr=MANIFEST_DIGEST_ATTR,
            )
            metadata = {
                "verification_mode": EXACT_CHILD_VERIFICATION_MODE,
                "receipt_path": str(receipt_path),
                "receipt_sha256": receipt["record_sha256"],
                "direct_metadata_inventory_sha256": receipt[
                    "direct_metadata_inventory"
                ]["inventory_sha256"],
                "archive_root_consolidated_metadata_reparse": False,
            }
            run = open_zarr_root(
                archive / run_path,
                mode="r",
                use_consolidated=False,
            )
            source_archive = None
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ProtocolSemanticChaserSelectionPublicationError(
            f"Unable to open exact semantic selection run: {exc}"
        ) from exc
    manifest, _ = _manifest_from_run(run)
    if (
        expected_recording_id is not None
        and manifest.get("recording_id") != expected_recording_id
    ):
        _fail("Semantic selection recording differs from expectation.")
    validation = _validate_persistent_run(
        archive / run_path,
        expected_manifest=manifest,
        expected_run_path=run_path,
        verify_content_hashes=deep_audit,
        run=run,
        source_archive=source_archive,
    )
    return ProtocolSemanticChaserSelectionSourceHandle(
        analysis_zarr=archive,
        run_name=name,
        run_path=run_path,
        recording_id=str(manifest["recording_id"]),
        manifest=manifest,
        arrays=validation["arrays"],
        metadata_equivalence=metadata,
        _seal=_SOURCE_HANDLE_SEAL,
    )


__all__ = [
    "MANIFEST_ATTR",
    "MANIFEST_DIGEST_ATTR",
    "PUBLICATION_POLICY",
    "RUNS_PARENT_PATH",
    "STORAGE_SCHEMA_ID",
    "STORAGE_SCHEMA_VERSION",
    "ProtocolSemanticChaserSelectionPublicationError",
    "ProtocolSemanticChaserSelectionPublicationPlan",
    "ProtocolSemanticChaserSelectionSourceHandle",
    "build_protocol_semantic_chaser_selection_publication_plan",
    "load_protocol_semantic_chaser_selection_source_handle",
    "publish_protocol_semantic_chaser_selection_run",
]
