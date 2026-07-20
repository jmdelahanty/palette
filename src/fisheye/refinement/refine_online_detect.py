#!/usr/bin/env python3
"""
Online Detection Refinement Pipeline

Refines canonical online target positions from stimulus runs by applying
smoothing, outlier removal, and gap interpolation to reduce tracking artifacts.

Workflow:
1. Load the exact ``target_position_xy`` child and its ``stimulus_state_key``
2. Rebind its v2 descriptor, row identity, and arena extent to persisted nodes
3. Smooth positions using Savitzky-Golay filter
4. Detect and remove outliers (large jumps, teleportation artifacts)
5. Interpolate small gaps
6. Publish both outputs atomically in the same native arena-relative space

Normal execution is future-canonical only. Historical group metadata, texture
labels, and scalar resolution ratios require an explicit offline migration.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import zarr
from rich.console import Console
from scipy.signal import savgol_filter

from ..analysis.chaser_metrics_loader import (
    CANONICAL_ONLINE_COORDINATE_HANDOFF_SCHEMA_ID,
    CANONICAL_ONLINE_COORDINATE_HANDOFF_SCHEMA_VERSION,
    CanonicalOnlineCoordinateHandoff,
    ChaserMetricsCoordinateContractError,
    load_canonical_online_coordinate_surface,
)
from ..shared.archive_identity import ArchiveIdentity, archive_identity
from ..shared.canonical_coordinate_publication import (
    BoundCanonicalCoordinateDescriptor,
    build_bound_canonical_coordinate_descriptor,
    load_bound_canonical_coordinate_descriptor,
    stamp_bound_canonical_coordinate_descriptors,
)
from ..shared.coordinate_descriptor import (
    CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
    COORDINATE_DESCRIPTOR_ATTR,
    CanonicalCoordinateDescriptor,
    parse_canonical_coordinate_descriptor,
)
from ..shared.coordinate_identity import (
    ROW_IDENTITY_CONTRACT_ATTR,
    STIMULUS_STATE_DOMAIN,
    STIMULUS_STATE_KEY_ARRAY_REF,
    STIMULUS_STATE_KEY_MODE,
    SOURCE_ROW_TEMPORAL_AUTHORITY_ATTR,
    SOURCE_ROW_TEMPORAL_AUTHORITY_DIGEST_ATTR,
    BoundRowIdentityContract,
    BoundSourceRowTemporalAuthority,
    build_row_identity_contract,
    identity_array_content_sha256,
    load_bound_row_identity_contract,
    load_bound_source_row_temporal_authority,
    stamp_and_bind_row_identity_contract,
    stamp_source_row_temporal_authority,
)
from ..shared.coordinate_reference import (
    BoundReferenceExtent,
)
from ..shared.coordinate_record import (
    BoundCoordinateRecord,
    bind_persisted_coordinate_record,
    stamp_and_bind_persisted_coordinate_record,
)
from ..shared.directed_transform_chain import BoundDirectedTransformChain
from ..shared.pixel_frame_authority import BoundPixelFrameAuthority
from ..shared.stimulus_coordinate_contract import (
    SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
    load_bound_stimulus_coordinate_evidence,
)
from ..shared.stimulus_frame_transform import (
    BoundStimulusFrameTransformEvidence,
)
from ..shared.system_metadata import get_environment_info, get_git_info
from ..shared.run_provenance import build_run_provenance_from_stage_record
from ..shared.selector_activation import (
    SelectorActivationError,
    activate_selector_eligible_run,
    write_activation_attr,
)
from ..shared.zarr_run_completion import (
    RUN_COMPLETED_AT_ATTR,
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_LATEST_COMPLETE_ATTR,
    RUN_LATEST_PENDING_ATTR,
    RUN_NAME_ATTR,
    RUN_STAGE_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_RUNNING,
    is_run_complete_in_parent,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
    require_runs_parent,
)

REFINED_ONLINE_GROUP = "refined_online_runs"
CANONICAL_ARENA_PROFILE = "arena_relative_canvas_px.top_left_y_down.v1"
ARENA_GEOMETRY_RECORD_ATTR = "arena_geometry_record"
ARENA_GEOMETRY_RECORD_DIGEST_ATTR = "arena_geometry_record_sha256"
SOURCE_MAPPING_ATTR = "source_coordinate_mapping"
SOURCE_MAPPING_DIGEST_ATTR = f"{SOURCE_MAPPING_ATTR}_sha256"
SOURCE_MAPPING_SCHEMA_ID = "palette.refined_online_source_coordinate_mapping"
SOURCE_MAPPING_SCHEMA_VERSION = 1
SURFACE_MANIFEST_ATTR = "coordinate_surface_manifest"
SURFACE_MANIFEST_DIGEST_ATTR = f"{SURFACE_MANIFEST_ATTR}_sha256"
SURFACE_MANIFEST_SCHEMA_ID = "palette.refined_online_coordinate_surface_manifest"
SURFACE_MANIFEST_SCHEMA_VERSION = 1
PROCESSING_RECORD_ATTR = "coordinate_processing_record"
PROCESSING_RECORD_DIGEST_ATTR = f"{PROCESSING_RECORD_ATTR}_sha256"
PROCESSING_RECORD_SCHEMA_ID = "palette.refined_online_coordinate_processing"
PROCESSING_RECORD_SCHEMA_VERSION = 1
COORDINATE_CONTRACT_EPOCH = "canonical_v2_future_only"
REFINED_ONLINE_PUBLICATION_OWNER_ATTR = "refined_online_publication_owner_uuid"
REFINED_ONLINE_PARENT_PUBLICATION_LEASE_ATTR = (
    "refined_online_publication_lease"
)
REFINED_ONLINE_PUBLICATION_GENERATION_ATTR = "publication_generation"
REFINED_ONLINE_PUBLICATION_POLICY_ATTR = "publication_policy"
REFINED_ONLINE_PUBLICATION_TOMBSTONE_ATTR = "refined_online_publication_tombstone"
REFINED_ONLINE_PUBLICATION_POLICY = (
    "owner_generation_guarded_selectors_then_eligibility_v1"
)
_BOUND_REFINED_ONLINE_EVIDENCE_SEAL = object()


class CanonicalOnlineRefinementError(ValueError):
    """Raised when normal refinement lacks exact canonical input evidence."""


@dataclass(frozen=True)
class CanonicalOnlinePositions:
    """Exact stimulus-state rows selected for canonical online refinement."""

    camera_frame_ids: np.ndarray
    source_acquisition_frame_index: np.ndarray
    positions: np.ndarray
    valid_mask: np.ndarray
    stimulus_state_key: np.ndarray
    stimulus_state_key_components: tuple[str, ...]
    source_row_indices: np.ndarray
    source_descriptor: CanonicalCoordinateDescriptor
    bound_source_descriptor: BoundCanonicalCoordinateDescriptor
    source_temporal_authority: BoundSourceRowTemporalAuthority
    source_handoff: Mapping[str, object]
    bound_source_handoff: CanonicalOnlineCoordinateHandoff
    stimulus_run: str
    chaser_index: int


@dataclass(frozen=True)
class _ResolvedCanonicalSource:
    point_node: Any
    key_node: Any
    rowset_node: Any
    descriptor: CanonicalCoordinateDescriptor
    bound_descriptor: BoundCanonicalCoordinateDescriptor
    identity: BoundRowIdentityContract
    arena_reference: BoundReferenceExtent
    reference_frame_authority: BoundPixelFrameAuthority
    transform_chain: BoundDirectedTransformChain
    frame_transform: BoundStimulusFrameTransformEvidence
    source_temporal_authority: BoundSourceRowTemporalAuthority
    source_acquisition_node: Any
    source_acquisition_frame_index: np.ndarray
    camera_frame_ids: np.ndarray
    lineage_records: tuple[BoundCoordinateRecord, ...]


@dataclass(frozen=True)
class BoundRefinedOnlineCoordinateEvidence:
    """Sealed live binding for one future canonical refined-online rowset."""

    archive_identity: ArchiveIdentity
    publication_status: str
    selector_eligible: bool
    row_identity: BoundRowIdentityContract
    source_temporal_authority: BoundSourceRowTemporalAuthority
    source_acquisition_frame_index: np.ndarray
    descriptor_bindings: tuple[
        tuple[str, BoundCanonicalCoordinateDescriptor], ...
    ]
    source_mapping: BoundCoordinateRecord
    processing_record: BoundCoordinateRecord
    surface_manifest: BoundCoordinateRecord
    source_stimulus_temporal_authority: BoundSourceRowTemporalAuthority
    _root: Any
    _refined_group: Any
    _seal: object

    def descriptor_for(
        self,
        stage: str,
    ) -> BoundCanonicalCoordinateDescriptor:
        for name, binding in self.descriptor_bindings:
            if name == stage:
                return binding
        raise CanonicalOnlineRefinementError(
            f"Refined coordinate stage {stage!r} is not published."
        )

    def assert_verified(self) -> None:
        if self._seal is not _BOUND_REFINED_ONLINE_EVIDENCE_SEAL:
            _fail("Refined-online evidence is not sealed.")
        current = _load_bound_refined_online_coordinate_evidence(
            self._root,
            self._refined_group,
            require_complete=self.publication_status == "complete",
            expected_selector_eligible=self.selector_eligible,
        )
        if (
            current.archive_identity != self.archive_identity
            or current.publication_status != self.publication_status
            or current.selector_eligible is not self.selector_eligible
            or current.row_identity.record_ref != self.row_identity.record_ref
            or current.row_identity.record_sha256
            != self.row_identity.record_sha256
            or current.source_temporal_authority.record_ref
            != self.source_temporal_authority.record_ref
            or current.source_temporal_authority.record_sha256
            != self.source_temporal_authority.record_sha256
            or tuple(
                (
                    name,
                    binding.descriptor.digest(),
                    f"/{binding.coordinate_node.path}",
                )
                for name, binding in current.descriptor_bindings
            )
            != tuple(
                (
                    name,
                    binding.descriptor.digest(),
                    f"/{binding.coordinate_node.path}",
                )
                for name, binding in self.descriptor_bindings
            )
            or current.source_mapping.record_ref != self.source_mapping.record_ref
            or current.source_mapping.record_sha256
            != self.source_mapping.record_sha256
            or current.processing_record.record_ref
            != self.processing_record.record_ref
            or current.processing_record.record_sha256
            != self.processing_record.record_sha256
            or current.surface_manifest.record_ref
            != self.surface_manifest.record_ref
            or current.surface_manifest.record_sha256
            != self.surface_manifest.record_sha256
            or current.source_stimulus_temporal_authority.record_ref
            != self.source_stimulus_temporal_authority.record_ref
            or current.source_stimulus_temporal_authority.record_sha256
            != self.source_stimulus_temporal_authority.record_sha256
            or not np.array_equal(
                current.source_acquisition_frame_index,
                self.source_acquisition_frame_index,
            )
        ):
            _fail("Refined-online coordinate evidence changed after binding.")


def _fail(message: str) -> None:
    raise CanonicalOnlineRefinementError(message)


def _print_after_commit(console: Console, message: str) -> None:
    """Best-effort reporting that cannot invalidate a committed publication."""

    try:
        console.print(message)
    except BaseException:
        return


def _canonical_mapping_digest(value: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalOnlineRefinementError(
            f"Canonical metadata is not deterministic JSON: {exc}."
        ) from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _archive_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(f"{label} must be one exact canonical archive path.")
    path = value[1:] if value.startswith("/") else value
    if not path or path.endswith("/") or "//" in path or "@" in path:
        _fail(f"{label} is not a canonical archive node path: {value!r}.")
    if any(part in {"", ".", ".."} for part in path.split("/")):
        _fail(f"{label} is not a canonical archive node path: {value!r}.")
    return path


def _node_at(root: zarr.Group, value: object, *, label: str) -> Any:
    path = _archive_path(value, label=label)
    try:
        return root[path]
    except (KeyError, TypeError) as exc:
        raise CanonicalOnlineRefinementError(
            f"{label} does not resolve to persisted node /{path}."
        ) from exc


def _bind_coordinate_record_ref(
    root: zarr.Group,
    *,
    record_ref: object,
    record_sha256: object,
    label: str,
) -> BoundCoordinateRecord:
    """Resolve one descriptor lineage ref to its exact persisted record."""

    if not isinstance(record_ref, str) or record_ref.count("@") != 1:
        _fail(f"{label} must be one exact /node/path@attribute reference.")
    node_ref, attr_name = record_ref.rsplit("@", 1)
    if not attr_name:
        _fail(f"{label} lacks its persisted record attribute name.")
    node = _node_at(root, node_ref, label=f"{label} node")
    try:
        bound = bind_persisted_coordinate_record(node, attr_name=attr_name)
    except Exception as exc:
        raise CanonicalOnlineRefinementError(
            f"{label} cannot be rebound to an exact persisted record: {exc}."
        ) from exc
    if bound.record_ref != record_ref or bound.record_sha256 != record_sha256:
        _fail(f"{label} path or digest is stale.")
    return bound


def _expected_source_paths(
    stimulus_run: str,
) -> tuple[str, str]:
    base = f"analysis/stimulus_runs/{stimulus_run}"
    return (
        f"{base}/tracking_data/chaser_states/target_position_xy",
        f"{base}/calibration/arena_geometry",
    )


def _resolve_persisted_canonical_source(
    root: zarr.Group,
    *,
    stimulus_run: str,
    point_path: object,
    rowset_path: object,
    key_path: object,
) -> _ResolvedCanonicalSource:
    """Rebind every source claim to exact persisted nodes."""

    expected_point, _expected_arena = _expected_source_paths(stimulus_run)
    canonical_point_path = _archive_path(point_path, label="source coordinate array")
    if canonical_point_path != expected_point:
        _fail(
            "Normal online refinement accepts only the exact canonical "
            f"target_position_xy child at /{expected_point}; found "
            f"/{canonical_point_path}. Run the explicit archive migration first."
        )
    canonical_rowset_path = _archive_path(rowset_path, label="source rowset")
    expected_rowset = expected_point.rsplit("/", 1)[0]
    if canonical_rowset_path != expected_rowset:
        _fail("Canonical target_position_xy is not owned by its exact chaser_states rowset.")
    canonical_key_path = _archive_path(key_path, label="source stimulus_state_key")
    if canonical_key_path != f"{expected_rowset}/{STIMULUS_STATE_KEY_ARRAY_REF}":
        _fail("Canonical target rows must use their exact sibling stimulus_state_key.")

    point_node = _node_at(root, canonical_point_path, label="source coordinate array")
    rowset_node = _node_at(root, canonical_rowset_path, label="source rowset")
    key_node = _node_at(root, canonical_key_path, label="source stimulus_state_key")
    if not isinstance(point_node, zarr.Array) or not isinstance(key_node, zarr.Array):
        _fail("Canonical target_position_xy and stimulus_state_key must be arrays.")
    if not isinstance(rowset_node, zarr.Group):
        _fail("Canonical chaser_states rowset must be a group.")
    if (
        point_node.attrs.get("semantic_role") != "target_position"
        or point_node.attrs.get("source_component_fields")
        != ["target_pos_x", "target_pos_y"]
    ):
        _fail("Canonical target_position_xy child role or component binding is invalid.")

    run_path = f"analysis/stimulus_runs/{stimulus_run}"
    run_group = _node_at(root, run_path, label="selected stimulus run")
    if not isinstance(run_group, zarr.Group):
        _fail("Selected stimulus run must be one exact persisted group.")
    try:
        evidence = load_bound_stimulus_coordinate_evidence(
            run_group,
            rowset_node,
            root_node=root,
        )
        identity = evidence.row_identity
    except Exception as exc:
        raise CanonicalOnlineRefinementError(
            f"Canonical source stimulus evidence is invalid: {exc}."
        ) from exc
    contract = identity.contract
    if (
        contract.domain != STIMULUS_STATE_DOMAIN
        or contract.mode != STIMULUS_STATE_KEY_MODE
        or contract.key_array.ref != STIMULUS_STATE_KEY_ARRAY_REF
    ):
        _fail("Source rows do not use the canonical stimulus_state_key profile.")

    try:
        raw_descriptor = parse_canonical_coordinate_descriptor(
            point_node.attrs.get(COORDINATE_DESCRIPTOR_ATTR)
        )
    except Exception as exc:
        raise CanonicalOnlineRefinementError(
            "Normal online refinement requires an array-specific schema-v2 "
            "descriptor; run the explicit legacy migration first."
        ) from exc
    if (
        raw_descriptor.profile_id != CANONICAL_ARENA_PROFILE
        or raw_descriptor.geometry_type != "point_xy"
        or raw_descriptor.components != ("x", "y")
        or raw_descriptor.component_units != ("px", "px")
    ):
        _fail(
            "Normal online refinement supports only canonical arena-relative "
            "target_position_xy points. Migrate unsupported coordinate spaces "
            "offline before refinement."
        )
    if (
        raw_descriptor.source_camera_overlay.status
        != CANONICAL_OVERLAY_REQUIRES_TRANSFORM
    ):
        _fail(
            "Canonical arena-relative stimulus positions must bind an exact "
            "arena-to-source-camera transform chain."
        )
    lineage_records = (
        evidence.surface_manifest,
        evidence.camera_mapping,
        evidence.frame_transform.manifest,
        evidence.import_lineage,
        evidence.output_manifest,
    )
    try:
        bound = load_bound_canonical_coordinate_descriptor(
            point_node,
            row_identity=identity,
            reference_frame_authority=(
                evidence.frame_transform.arena_relative_frame
            ),
            transform_chain=evidence.frame_transform.transform_chain,
            lineage_records=lineage_records,
        )
    except Exception as exc:
        raise CanonicalOnlineRefinementError(
            f"Canonical target-position authority cannot be verified exactly: {exc}."
        ) from exc
    source_acquisition_node = rowset_node.get(
        SOURCE_ACQUISITION_FRAME_INDEX_ARRAY
    )
    if not isinstance(source_acquisition_node, zarr.Array):
        _fail("Canonical stimulus source lacks source_acquisition_frame_index.")
    source_acquisition = np.asarray(source_acquisition_node[:])
    if (
        source_acquisition.dtype != np.dtype(np.int64)
        or source_acquisition.shape != (identity.leading_dimension,)
        or not np.array_equal(
            source_acquisition,
            evidence.source_acquisition_frame_index,
        )
    ):
        _fail("Stimulus acquisition-frame mapping is stale or misaligned.")
    return _ResolvedCanonicalSource(
        point_node=point_node,
        key_node=key_node,
        rowset_node=rowset_node,
        descriptor=bound.descriptor,
        bound_descriptor=bound,
        identity=identity,
        arena_reference=evidence.arena_reference,
        reference_frame_authority=evidence.frame_transform.arena_relative_frame,
        transform_chain=evidence.frame_transform.transform_chain,
        frame_transform=evidence.frame_transform,
        source_temporal_authority=evidence.source_temporal_authority,
        source_acquisition_node=source_acquisition_node,
        source_acquisition_frame_index=source_acquisition,
        camera_frame_ids=np.asarray(evidence.camera_frame_ids),
        lineage_records=lineage_records,
    )


def _select_stimulus_rows_for_refinement(
    *,
    source_keys: np.ndarray,
    components: tuple[str, ...],
    source_acquisition_frame_index: np.ndarray,
    camera_frame_ids: np.ndarray,
    chaser_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select by stimulus identity and order only by acquisition time."""

    keys = np.asarray(source_keys)
    acquisition = np.asarray(source_acquisition_frame_index)
    external_camera = np.asarray(camera_frame_ids)
    row_count = int(keys.shape[0]) if keys.ndim else -1
    if (
        row_count < 0
        or acquisition.shape != (row_count,)
        or acquisition.dtype != np.dtype(np.int64)
        or external_camera.shape != (row_count,)
        or external_camera.dtype != np.dtype(np.int64)
        or np.any(acquisition < 0)
        or np.any(external_camera < 0)
    ):
        _fail("Stimulus identity and temporal/provenance arrays are misaligned.")
    if "chaser_index" in components:
        component_index = components.index("chaser_index")
        if keys.ndim == 1:
            if components != ("chaser_index",):
                _fail("Composite stimulus identity must be a two-dimensional key.")
            identity_chaser = keys
        else:
            if keys.ndim != 2 or keys.shape[1] != len(components):
                _fail("Composite stimulus identity shape disagrees with its components.")
            identity_chaser = keys[:, component_index]
        selected = np.asarray(identity_chaser == chaser_index, dtype=bool)
    else:
        if chaser_index != 0:
            _fail(
                "Stimulus identity has no chaser_index component, so only "
                "chaser_index=0 is addressable."
            )
        selected = np.ones(row_count, dtype=bool)
    source_rows = np.flatnonzero(selected).astype(np.int64, copy=False)
    if source_rows.size == 0:
        _fail(f"Stimulus identity contains no rows for chaser_index={chaser_index}.")
    order = np.argsort(acquisition[source_rows], kind="stable")
    source_rows = source_rows[order]
    selected_acquisition = acquisition[source_rows]
    selected_camera = external_camera[source_rows]
    if (
        np.unique(selected_acquisition).shape[0] != selected_acquisition.shape[0]
        or (
            selected_acquisition.shape[0] > 1
            and np.any(np.diff(selected_acquisition) <= 0)
        )
    ):
        _fail(
            "Selected stimulus rows do not map one-to-one into increasing sealed "
            "acquisition-frame time."
        )
    return source_rows, selected_acquisition, selected_camera


def load_online_positions(
    zarr_path: str,
    stimulus_run: Optional[str] = None,
    chaser_index: int = 0,
    console: Optional[Console] = None,
) -> CanonicalOnlinePositions:
    """Load canonical stimulus rows directly through sealed acquisition time.

    External camera-frame identifiers are preserved as provenance but never
    select or order rows. Historical attrs, v1 descriptors, texture labels,
    root dimensions, and scalar resolution ratios are outside this normal path.
    """
    if console is None:
        console = Console()
    if isinstance(chaser_index, bool) or not isinstance(chaser_index, int) or chaser_index < 0:
        _fail("chaser_index must be a nonnegative integer.")

    root = zarr.open(str(zarr_path), mode="r")
    analysis = root.get("analysis")
    runs = analysis.get("stimulus_runs") if isinstance(analysis, zarr.Group) else None
    if not isinstance(runs, zarr.Group):
        _fail("Archive lacks canonical analysis/stimulus_runs.")
    stimulus_run_name = stimulus_run
    if stimulus_run_name is None:
        stimulus_run_name = runs.attrs.get("latest_complete")
    if not isinstance(stimulus_run_name, str) or not stimulus_run_name:
        _fail("No exact complete stimulus run was selected.")
    stim_group = runs.get(stimulus_run_name)
    if not isinstance(stim_group, zarr.Group):
        _fail(f"Selected stimulus run {stimulus_run_name!r} does not exist.")
    tracking = stim_group.get("tracking_data")
    rowset = tracking.get("chaser_states") if isinstance(tracking, zarr.Group) else None
    if not isinstance(rowset, zarr.Group):
        _fail("Selected stimulus run lacks canonical chaser_states.")

    point_path, _ = _expected_source_paths(stimulus_run_name)
    resolved = _resolve_persisted_canonical_source(
        root,
        stimulus_run=stimulus_run_name,
        point_path=point_path,
        rowset_path=point_path.rsplit("/", 1)[0],
        key_path=(
            f"{point_path.rsplit('/', 1)[0]}/{STIMULUS_STATE_KEY_ARRAY_REF}"
        ),
    )
    try:
        point_values, handoff, bound_handoff = (
            load_canonical_online_coordinate_surface(
                root,
                stim_group,
                rowset,
            )
        )
        bound_handoff.assert_verified()
    except ChaserMetricsCoordinateContractError as exc:
        raise CanonicalOnlineRefinementError(
            "Online refinement is canonical-only; migrate the selected run "
            f"before refinement: {exc}"
        ) from exc
    if (
        handoff.get("schema_id")
        != CANONICAL_ONLINE_COORDINATE_HANDOFF_SCHEMA_ID
        or handoff.get("schema_version")
        != CANONICAL_ONLINE_COORDINATE_HANDOFF_SCHEMA_VERSION
        or handoff.get("semantic_role") != "target_position"
        or archive_identity(root) != bound_handoff.archive_identity
        or bound_handoff.coordinate_descriptor.descriptor != resolved.descriptor
        or bound_handoff.source_temporal_authority.record_ref
        != resolved.source_temporal_authority.record_ref
        or bound_handoff.source_temporal_authority.record_sha256
        != resolved.source_temporal_authority.record_sha256
        or not np.array_equal(
            bound_handoff.source_acquisition_frame_index,
            resolved.source_acquisition_frame_index,
        )
    ):
        _fail("Canonical direct source handoff differs from persisted evidence.")

    mapping = handoff.get("stimulus_state_key_mapping")
    if not isinstance(mapping, dict):
        _fail("Canonical handoff lacks stimulus_state_key row mapping.")
    source_keys = np.asarray(resolved.key_node[:])
    components = resolved.identity.contract.key_array.components
    if (
        tuple(mapping.get("components", ())) != components
        or not np.array_equal(np.asarray(mapping.get("values")), source_keys)
        or point_values.shape != (resolved.identity.leading_dimension, 2)
    ):
        _fail("Canonical source values disagree with their sealed row identity.")

    source_rows, acquisition_frames, camera_frames = (
        _select_stimulus_rows_for_refinement(
            source_keys=source_keys,
            components=components,
            source_acquisition_frame_index=(
                resolved.source_acquisition_frame_index
            ),
            camera_frame_ids=resolved.camera_frame_ids,
            chaser_index=chaser_index,
        )
    )
    selected_positions = np.asarray(point_values[source_rows], dtype=np.float64)
    output_keys = source_keys[source_rows].copy()
    valid_mask = np.isfinite(selected_positions).all(axis=1)
    mapping["selected_source_row_indices"] = source_rows.tolist()
    mapping["selected_values"] = output_keys.tolist()
    mapping["selected_camera_frame_ids"] = camera_frames.tolist()
    mapping["selected_source_acquisition_frame_index"] = (
        acquisition_frames.tolist()
    )
    console.print(
        "[cyan]Canonical input:[/cyan] "
        f"{selected_positions.shape[0]} stimulus-state rows in "
        f"{resolved.descriptor.space_id}"
    )
    return CanonicalOnlinePositions(
        camera_frame_ids=camera_frames.copy(),
        source_acquisition_frame_index=acquisition_frames.copy(),
        positions=selected_positions.copy(),
        valid_mask=valid_mask,
        stimulus_state_key=output_keys,
        stimulus_state_key_components=components,
        source_row_indices=source_rows.copy(),
        source_descriptor=resolved.descriptor,
        bound_source_descriptor=resolved.bound_descriptor,
        source_temporal_authority=resolved.source_temporal_authority,
        source_handoff=copy.deepcopy(dict(handoff)),
        bound_source_handoff=bound_handoff,
        stimulus_run=stimulus_run_name,
        chaser_index=chaser_index,
    )


def smooth_positions(
    positions: np.ndarray,
    frames: np.ndarray,
    valid_mask: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth positions using Savitzky-Golay filter.

    Args:
        positions: Position array (N, 2)
        frames: Exact increasing source-acquisition frame indices for each row
        valid_mask: Boolean mask of valid positions
        window_length: Filter window length (must be odd)
        polyorder: Polynomial order for fitting

    Returns:
        Tuple of (smoothed_positions, smoothed_mask)
    """
    positions = np.asarray(positions)
    frames = np.asarray(frames)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if (
        positions.ndim != 2
        or positions.shape[1] != 2
        or frames.shape != (positions.shape[0],)
        or frames.dtype.kind not in "iu"
        or valid_mask.shape != (positions.shape[0],)
        or np.any(frames < 0)
        or (frames.shape[0] > 1 and np.any(np.diff(frames) <= 0))
    ):
        _fail(
            "Smoothing requires (N, 2) positions aligned to unique, increasing "
            "nonnegative source-acquisition frame indices and one valid mask."
        )

    smoothed = np.full_like(positions, np.nan)
    smoothed_mask = np.zeros(len(positions), dtype=bool)

    # Need at least window_length consecutive valid points to smooth
    if valid_mask.sum() < window_length:
        return positions.copy(), valid_mask.copy()

    # Find consecutive valid segments
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return positions.copy(), valid_mask.copy()

    # Group rows only when both their array indices and source-acquisition frame
    # identities are consecutive.  Row adjacency is not temporal adjacency
    # after a sparse stimulus-to-camera join.
    segments = []
    start_idx = valid_indices[0]
    for i in range(1, len(valid_indices)):
        if (
            valid_indices[i] != valid_indices[i - 1] + 1
            or frames[valid_indices[i]] != frames[valid_indices[i - 1]] + 1
        ):
            # Gap found, save segment
            segments.append((start_idx, valid_indices[i - 1]))
            start_idx = valid_indices[i]
    segments.append((start_idx, valid_indices[-1]))

    # Smooth each segment
    for start, end in segments:
        segment_length = end - start + 1

        if segment_length < window_length:
            # Too short to smooth, keep original
            smoothed[start : end + 1] = positions[start : end + 1]
            smoothed_mask[start : end + 1] = True
            continue

        # Apply Savitzky-Golay filter
        for axis in [0, 1]:
            smoothed[start : end + 1, axis] = savgol_filter(
                positions[start : end + 1, axis],
                window_length=window_length,
                polyorder=polyorder,
                mode="interp",
            )

        smoothed_mask[start : end + 1] = True

    return smoothed, smoothed_mask


def detect_outliers(
    positions: np.ndarray,
    frames: np.ndarray,
    valid_mask: np.ndarray,
    displacement_threshold: float = 100.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Detect outliers based on displacement threshold.

    Args:
        positions: Position array (N, 2)
        frames: Source-acquisition frame indices
        valid_mask: Boolean mask of valid positions
        displacement_threshold: Maximum reasonable displacement in native coordinate units

    Returns:
        Tuple of (outlier_mask, outlier_stats)
    """
    outlier_mask = np.zeros(len(positions), dtype=bool)

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < 2:
        return outlier_mask, {"outliers_detected": 0, "threshold": displacement_threshold}

    # Calculate frame-to-frame displacement
    for i in range(len(valid_indices) - 1):
        idx1 = valid_indices[i]
        idx2 = valid_indices[i + 1]

        # Only check consecutive frames
        if frames[idx2] - frames[idx1] != 1:
            continue

        displacement = np.linalg.norm(positions[idx2] - positions[idx1])

        if displacement > displacement_threshold:
            # Mark the second point as outlier (assumes first is correct)
            outlier_mask[idx2] = True

    outlier_stats = {
        "outliers_detected": int(outlier_mask.sum()),
        "threshold": float(displacement_threshold),
        "outlier_rate": float(outlier_mask.sum() / valid_mask.sum() * 100) if valid_mask.sum() > 0 else 0.0,
    }

    return outlier_mask, outlier_stats


def interpolate_gaps(
    positions: np.ndarray,
    frames: np.ndarray,
    valid_mask: np.ndarray,
    max_gap: int = 20,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Interpolate small gaps in position data.

    Args:
        positions: Position array (N, 2)
        frames: Source-acquisition frame indices
        valid_mask: Boolean mask of valid positions
        max_gap: Maximum gap size to interpolate (frames)

    Returns:
        Tuple of (interpolated_positions, interpolation_mask, interp_stats)
    """
    interpolated = positions.copy()
    interpolation_mask = np.zeros(len(positions), dtype=bool)

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < 2:
        return interpolated, interpolation_mask, {"gaps_filled": 0, "interpolated_frames": 0}

    # Find gaps
    gaps_filled = 0
    interpolated_frames = 0

    for i in range(len(valid_indices) - 1):
        idx1 = valid_indices[i]
        idx2 = valid_indices[i + 1]

        frame1 = frames[idx1]
        frame2 = frames[idx2]
        gap_size = frame2 - frame1 - 1

        if gap_size <= 0 or gap_size > max_gap:
            continue

        # Find indices to interpolate
        gap_indices = []
        for j in range(idx1 + 1, idx2):
            if frames[j] > frame1 and frames[j] < frame2:
                gap_indices.append(j)

        if len(gap_indices) == 0:
            continue

        # Linear interpolation
        t = (frames[gap_indices] - frame1) / (frame2 - frame1)
        for axis in [0, 1]:
            interpolated[gap_indices, axis] = positions[idx1, axis] + t * (
                positions[idx2, axis] - positions[idx1, axis]
            )

        interpolation_mask[gap_indices] = True
        gaps_filled += 1
        interpolated_frames += len(gap_indices)

    interp_stats = {
        "gaps_filled": gaps_filled,
        "interpolated_frames": interpolated_frames,
        "max_gap": max_gap,
    }

    return interpolated, interpolation_mask, interp_stats


def _array_chunks(values: np.ndarray) -> tuple[int, ...]:
    rows = max(1, min(10_000, int(values.shape[0])))
    return (rows,) if values.ndim == 1 else (rows, *values.shape[1:])


def _source_mapping_record(
    refined_group: zarr.Group,
    *,
    source: _ResolvedCanonicalSource,
    input_data: CanonicalOnlinePositions,
    output_identity: Any,
) -> dict[str, Any]:
    source_rows = np.asarray(input_data.source_row_indices, dtype=np.int64)
    source_values = np.asarray(source.point_node[:])[source_rows]
    output_key = np.asarray(input_data.stimulus_state_key)
    camera_frames = np.asarray(input_data.camera_frame_ids, dtype=np.int64)
    acquisition_frames = np.asarray(
        input_data.source_acquisition_frame_index,
        dtype=np.int64,
    )
    return {
        "schema_id": SOURCE_MAPPING_SCHEMA_ID,
        "schema_version": SOURCE_MAPPING_SCHEMA_VERSION,
        "mapping_operation": "exact_source_row_selection_then_value_refinement",
        "source_stimulus_run": input_data.stimulus_run,
        "source_chaser_index": int(input_data.chaser_index),
        "source_coordinate_array_ref": f"/{source.point_node.path}",
        "source_coordinate_descriptor_ref": (
            f"/{source.point_node.path}@{COORDINATE_DESCRIPTOR_ATTR}"
        ),
        "source_coordinate_descriptor_sha256": source.descriptor.digest(),
        "source_selected_position_content_sha256": (
            identity_array_content_sha256(source_values)
        ),
        "source_rowset_ref": f"/{source.rowset_node.path}",
        "source_row_identity_contract_ref": source.identity.record_ref,
        "source_row_identity_contract_sha256": source.identity.record_sha256,
        "source_key_array_ref": f"/{source.key_node.path}",
        "source_key_content_sha256": (
            source.identity.contract.key_array.content_sha256
        ),
        "source_reference_authority_ref": (
            source.reference_frame_authority.record_ref
        ),
        "source_reference_authority_sha256": (
            source.reference_frame_authority.record_sha256
        ),
        "source_transform_chain": [
            {
                "record_ref": item.record_ref,
                "record_sha256": item.record_sha256,
            }
            for item in source.transform_chain.transform_records
        ],
        "source_row_temporal_authority_ref": (
            source.source_temporal_authority.record_ref
        ),
        "source_row_temporal_authority_sha256": (
            source.source_temporal_authority.record_sha256
        ),
        "source_coordinate_lineage": [
            {
                "record_ref": record.record_ref,
                "record_sha256": record.record_sha256,
            }
            for record in source.lineage_records
        ],
        "source_row_count": int(source.identity.leading_dimension),
        "source_row_indices_array_ref": f"/{refined_group.path}/source_row_indices",
        "source_row_indices_content_sha256": (
            identity_array_content_sha256(source_rows)
        ),
        "output_row_identity_contract_ref": output_identity.record_ref,
        "output_row_identity_contract_sha256": output_identity.record_sha256,
        "output_key_array_ref": (
            f"/{refined_group.path}/{STIMULUS_STATE_KEY_ARRAY_REF}"
        ),
        "output_key_content_sha256": identity_array_content_sha256(output_key),
        "output_key_components": list(input_data.stimulus_state_key_components),
        "camera_frame_ids_array_ref": f"/{refined_group.path}/camera_frame_ids",
        "camera_frame_ids_content_sha256": identity_array_content_sha256(
            camera_frames
        ),
        "source_acquisition_frame_index_array_ref": (
            f"/{refined_group.path}/{SOURCE_ACQUISITION_FRAME_INDEX_ARRAY}"
        ),
        "source_acquisition_frame_index_content_sha256": (
            identity_array_content_sha256(acquisition_frames)
        ),
        "output_source_row_temporal_authority_ref": (
            f"/{refined_group.path}@{SOURCE_ROW_TEMPORAL_AUTHORITY_ATTR}"
        ),
        "output_source_row_temporal_authority_sha256": refined_group.attrs.get(
            SOURCE_ROW_TEMPORAL_AUTHORITY_DIGEST_ATTR
        ),
        "output_row_count": int(acquisition_frames.shape[0]),
        "row_identity_preserved_during_interpolation": True,
    }


def _surface_manifest(refined_group: zarr.Group) -> dict[str, Any]:
    filtered = refined_group["filtered/positions_px"]
    interpolated = refined_group["interpolated/positions_px"]
    return {
        "schema_id": SURFACE_MANIFEST_SCHEMA_ID,
        "schema_version": SURFACE_MANIFEST_SCHEMA_VERSION,
        "surfaces": [
            {
                "array_ref": f"/{refined_group.path}/filtered/positions_px",
                "semantic_role": "target_position",
                "coordinate_stage": "filtered",
                "content_sha256": identity_array_content_sha256(
                    np.asarray(filtered[:])
                ),
            },
            {
                "array_ref": f"/{refined_group.path}/interpolated/positions_px",
                "semantic_role": "target_position",
                "coordinate_stage": "interpolated",
                "content_sha256": identity_array_content_sha256(
                    np.asarray(interpolated[:])
                ),
            },
        ],
    }


def _walk_run_nodes(group: zarr.Group) -> list[Any]:
    nodes: list[Any] = [group]
    for name in group.keys():
        node = group[name]
        nodes.append(node)
        if isinstance(node, zarr.Group):
            nodes.extend(_walk_run_nodes(node)[1:])
    return nodes


def _mapping_attr(
    attrs: Mapping[str, Any],
    *,
    name: str,
    digest_name: str,
) -> Mapping[str, Any]:
    value = attrs.get(name)
    digest = attrs.get(digest_name)
    if not isinstance(value, Mapping):
        _fail(f"Refined run lacks exact {name} metadata.")
    if not isinstance(digest, str) or digest != _canonical_mapping_digest(value):
        _fail(f"Refined run {name} digest is absent or stale.")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        _fail(f"{label} fields are not closed (missing={missing}, unknown={unknown}).")


_SOURCE_MAPPING_FIELDS = {
    "schema_id",
    "schema_version",
    "mapping_operation",
    "source_stimulus_run",
    "source_chaser_index",
    "source_coordinate_array_ref",
    "source_coordinate_descriptor_ref",
    "source_coordinate_descriptor_sha256",
    "source_selected_position_content_sha256",
    "source_rowset_ref",
    "source_row_identity_contract_ref",
    "source_row_identity_contract_sha256",
    "source_key_array_ref",
    "source_key_content_sha256",
    "source_reference_authority_ref",
    "source_reference_authority_sha256",
    "source_transform_chain",
    "source_row_temporal_authority_ref",
    "source_row_temporal_authority_sha256",
    "source_coordinate_lineage",
    "source_row_count",
    "source_row_indices_array_ref",
    "source_row_indices_content_sha256",
    "output_row_identity_contract_ref",
    "output_row_identity_contract_sha256",
    "output_key_array_ref",
    "output_key_content_sha256",
    "output_key_components",
    "camera_frame_ids_array_ref",
    "camera_frame_ids_content_sha256",
    "source_acquisition_frame_index_array_ref",
    "source_acquisition_frame_index_content_sha256",
    "output_source_row_temporal_authority_ref",
    "output_source_row_temporal_authority_sha256",
    "output_row_count",
    "row_identity_preserved_during_interpolation",
}

_PROCESSING_RECORD_FIELDS = {
    "schema_id",
    "schema_version",
    "operation",
    "temporal_adjacency_rule",
    "parameters",
    "provenance",
    "source_mapping_ref",
    "source_mapping_sha256",
    "surface_manifest_ref",
    "surface_manifest_sha256",
    "filtered_positions_ref",
    "filtered_positions_sha256",
    "interpolated_positions_ref",
    "interpolated_positions_sha256",
}


def _processing_record(
    refined_group: zarr.Group,
    *,
    parameters: Mapping[str, Any],
    provenance: Mapping[str, Any],
    source_mapping: BoundCoordinateRecord,
    surface_manifest: BoundCoordinateRecord,
) -> dict[str, Any]:
    """Build exact algorithm and output evidence for coordinate publication."""

    return {
        "schema_id": PROCESSING_RECORD_SCHEMA_ID,
        "schema_version": PROCESSING_RECORD_SCHEMA_VERSION,
        "operation": "smooth_then_outlier_filter_then_gap_interpolation",
        "temporal_adjacency_rule": (
            "source_acquisition_frame_index_difference_equals_one"
        ),
        "parameters": copy.deepcopy(dict(parameters)),
        "provenance": copy.deepcopy(dict(provenance)),
        "source_mapping_ref": source_mapping.record_ref,
        "source_mapping_sha256": source_mapping.record_sha256,
        "surface_manifest_ref": surface_manifest.record_ref,
        "surface_manifest_sha256": surface_manifest.record_sha256,
        "filtered_positions_ref": f"/{refined_group.path}/filtered/positions_px",
        "filtered_positions_sha256": identity_array_content_sha256(
            np.asarray(refined_group["filtered/positions_px"][:])
        ),
        "interpolated_positions_ref": (
            f"/{refined_group.path}/interpolated/positions_px"
        ),
        "interpolated_positions_sha256": identity_array_content_sha256(
            np.asarray(refined_group["interpolated/positions_px"][:])
        ),
    }


def _load_bound_refined_online_coordinate_evidence(
    root: zarr.Group,
    refined_group: zarr.Group,
    *,
    require_complete: bool,
    expected_selector_eligible: bool,
) -> BoundRefinedOnlineCoordinateEvidence:
    """Load staged or complete evidence for the producer's transaction."""

    expected_status = "complete" if require_complete else "staging"
    if refined_group.attrs.get("publication_status") != expected_status:
        _fail(f"Refined run publication_status must be {expected_status!r}.")
    try:
        refined_runs = root[REFINED_ONLINE_GROUP]
    except Exception as exc:
        raise CanonicalOnlineRefinementError(
            "Refined run is detached from its canonical runs parent."
        ) from exc
    run_name = str(refined_group.path).rsplit("/", 1)[-1]
    if (
        refined_group.attrs.get(RUN_COMPLETION_CONTRACT_ATTR)
        != RUN_COMPLETION_CONTRACT
        or refined_group.attrs.get(RUN_NAME_ATTR) != run_name
        or refined_group.attrs.get(RUN_STAGE_ATTR) != "refine_online_detect"
        or refined_group.attrs.get("stage_selector_eligible")
        is not expected_selector_eligible
    ):
        _fail("Refined run lacks its exact canonical lifecycle identity.")
    generic_status = refined_group.attrs.get(RUN_COMPLETION_STATUS_ATTR)
    if require_complete:
        if generic_status != RUN_STATUS_COMPLETE or not is_run_complete_in_parent(
            refined_runs,
            refined_group,
            legacy_default=False,
        ):
            _fail("Refined run is not complete under the generic run lifecycle.")
    elif generic_status != RUN_STATUS_RUNNING:
        _fail("Staged refined run is not running under the generic run lifecycle.")
    if refined_group.attrs.get("coordinate_contract_epoch") != COORDINATE_CONTRACT_EPOCH:
        _fail("Refined run does not declare the future canonical coordinate epoch.")
    identity_owners = [
        node.path
        for node in _walk_run_nodes(refined_group)
        if ROW_IDENTITY_CONTRACT_ATTR in getattr(node, "attrs", {})
    ]
    if identity_owners != [refined_group.path]:
        _fail("Refined run must contain exactly one row-identity contract at its root.")

    key_node = refined_group.get(STIMULUS_STATE_KEY_ARRAY_REF)
    if not isinstance(key_node, zarr.Array):
        _fail("Refined run lacks its primary stimulus_state_key array.")
    try:
        output_identity = load_bound_row_identity_contract(refined_group, key_node)
    except Exception as exc:
        raise CanonicalOnlineRefinementError(
            f"Refined stimulus-state identity is invalid: {exc}."
        ) from exc
    output_contract = output_identity.contract
    if (
        output_contract.domain != STIMULUS_STATE_DOMAIN
        or output_contract.mode != STIMULUS_STATE_KEY_MODE
        or output_contract.key_array.ref != STIMULUS_STATE_KEY_ARRAY_REF
    ):
        _fail("Refined rows must use the canonical stimulus_state_key profile.")
    row_count = output_identity.leading_dimension

    try:
        mapping_bound = bind_persisted_coordinate_record(
            refined_group,
            attr_name=SOURCE_MAPPING_ATTR,
            digest_attr_name=SOURCE_MAPPING_DIGEST_ATTR,
        )
    except Exception as exc:
        raise CanonicalOnlineRefinementError(
            f"Refined source mapping cannot be rebound exactly: {exc}."
        ) from exc
    mapping = mapping_bound.record
    _require_exact_fields(mapping, _SOURCE_MAPPING_FIELDS, label=SOURCE_MAPPING_ATTR)
    if (
        mapping["schema_id"] != SOURCE_MAPPING_SCHEMA_ID
        or mapping["schema_version"] != SOURCE_MAPPING_SCHEMA_VERSION
        or mapping["mapping_operation"]
        != "exact_source_row_selection_then_value_refinement"
        or mapping["row_identity_preserved_during_interpolation"] is not True
    ):
        _fail("Refined source mapping schema or operation is unsupported.")
    stimulus_run = mapping["source_stimulus_run"]
    if not isinstance(stimulus_run, str) or not stimulus_run:
        _fail("Refined source mapping lacks its exact stimulus run.")
    if (
        refined_group.attrs.get("source_stimulus_run") != stimulus_run
        or isinstance(mapping["source_chaser_index"], bool)
        or not isinstance(mapping["source_chaser_index"], int)
        or mapping["source_chaser_index"] < 0
        or refined_group.attrs.get("chaser_index")
        != mapping["source_chaser_index"]
    ):
        _fail("Refined run source selection conflicts with its mapping record.")
    source = _resolve_persisted_canonical_source(
        root,
        stimulus_run=stimulus_run,
        point_path=mapping["source_coordinate_array_ref"],
        rowset_path=mapping["source_rowset_ref"],
        key_path=mapping["source_key_array_ref"],
    )
    expected_descriptor_ref = f"/{source.point_node.path}@{COORDINATE_DESCRIPTOR_ATTR}"
    if (
        mapping["source_coordinate_descriptor_ref"] != expected_descriptor_ref
        or mapping["source_coordinate_descriptor_sha256"] != source.descriptor.digest()
        or mapping["source_row_identity_contract_ref"] != source.identity.record_ref
        or mapping["source_row_identity_contract_sha256"]
        != source.identity.record_sha256
        or mapping["source_key_content_sha256"]
        != source.identity.contract.key_array.content_sha256
        or mapping["source_reference_authority_ref"]
        != source.reference_frame_authority.record_ref
        or mapping["source_reference_authority_sha256"]
        != source.reference_frame_authority.record_sha256
        or mapping["source_transform_chain"]
        != [
            {
                "record_ref": item.record_ref,
                "record_sha256": item.record_sha256,
            }
            for item in source.transform_chain.transform_records
        ]
        or mapping["source_row_temporal_authority_ref"]
        != source.source_temporal_authority.record_ref
        or mapping["source_row_temporal_authority_sha256"]
        != source.source_temporal_authority.record_sha256
        or mapping["source_row_count"] != source.identity.leading_dimension
        or mapping["source_coordinate_lineage"]
        != [
            {
                "record_ref": record.record_ref,
                "record_sha256": record.record_sha256,
            }
            for record in source.lineage_records
        ]
    ):
        _fail("Refined source mapping no longer matches exact source authority.")

    source_rows_node = _node_at(
        root,
        mapping["source_row_indices_array_ref"],
        label="refined source-row mapping",
    )
    frames_node = _node_at(
        root,
        mapping["camera_frame_ids_array_ref"],
        label="refined camera-frame mapping",
    )
    acquisition_node = _node_at(
        root,
        mapping["source_acquisition_frame_index_array_ref"],
        label="refined acquisition-frame mapping",
    )
    if not all(
        isinstance(node, zarr.Array)
        for node in (source_rows_node, frames_node, acquisition_node)
    ):
        _fail("Refined source-row and temporal mappings must be arrays.")
    expected_source_rows_path = f"{refined_group.path}/source_row_indices"
    expected_frames_path = f"{refined_group.path}/camera_frame_ids"
    expected_acquisition_path = (
        f"{refined_group.path}/{SOURCE_ACQUISITION_FRAME_INDEX_ARRAY}"
    )
    if (
        source_rows_node.path != expected_source_rows_path
        or frames_node.path != expected_frames_path
        or acquisition_node.path != expected_acquisition_path
    ):
        _fail("Refined mapped fields are not exact children of the output rowset.")
    source_rows = np.asarray(source_rows_node[:])
    camera_frames = np.asarray(frames_node[:])
    acquisition_frames = np.asarray(acquisition_node[:])
    if (
        source_rows.shape != (row_count,)
        or source_rows.dtype != np.dtype(np.int64)
        or camera_frames.shape != (row_count,)
        or camera_frames.dtype != np.dtype(np.int64)
        or acquisition_frames.shape != (row_count,)
        or acquisition_frames.dtype != np.dtype(np.int64)
        or np.any(source_rows < 0)
        or np.any(source_rows >= source.identity.leading_dimension)
        or np.unique(source_rows).shape[0] != row_count
        or np.any(camera_frames < 0)
        or np.any(acquisition_frames < 0)
        or np.unique(acquisition_frames).shape[0] != row_count
        or (row_count > 1 and np.any(np.diff(acquisition_frames) <= 0))
        or not np.array_equal(
            acquisition_frames,
            source.source_acquisition_frame_index[source_rows],
        )
        or not np.array_equal(camera_frames, source.camera_frame_ids[source_rows])
    ):
        _fail(
            "Refined mapped fields have invalid dtype, shape, range, or "
            "acquisition-frame uniqueness."
        )
    if (
        mapping["source_row_indices_content_sha256"]
        != identity_array_content_sha256(source_rows)
        or mapping["camera_frame_ids_content_sha256"]
        != identity_array_content_sha256(camera_frames)
        or mapping["source_acquisition_frame_index_content_sha256"]
        != identity_array_content_sha256(acquisition_frames)
    ):
        _fail("Refined mapped-field content digest is stale.")
    try:
        output_temporal_authority = load_bound_source_row_temporal_authority(
            refined_group,
            acquisition_node,
            source_row_identity=output_identity,
            acquisition_frame=source.frame_transform.acquisition_frame,
        )
    except Exception as exc:
        raise CanonicalOnlineRefinementError(
            f"Refined temporal authority cannot be rebound: {exc}."
        ) from exc
    if (
        mapping["output_source_row_temporal_authority_ref"]
        != output_temporal_authority.record_ref
        or mapping["output_source_row_temporal_authority_sha256"]
        != output_temporal_authority.record_sha256
    ):
        _fail("Refined output temporal-authority pointer is stale.")

    output_keys = np.asarray(key_node[:])
    source_keys = np.asarray(source.key_node[:])
    if not np.array_equal(output_keys, source_keys[source_rows]):
        _fail("Refined stimulus_state_key does not equal the exact selected source keys.")
    if (
        mapping["output_row_identity_contract_ref"] != output_identity.record_ref
        or mapping["output_row_identity_contract_sha256"]
        != output_identity.record_sha256
        or mapping["output_key_array_ref"] != f"/{key_node.path}"
        or mapping["output_key_content_sha256"]
        != identity_array_content_sha256(output_keys)
        or tuple(mapping["output_key_components"])
        != output_contract.key_array.components
        or output_contract.key_array.components
        != source.identity.contract.key_array.components
        or mapping["output_row_count"] != row_count
    ):
        _fail("Refined output identity mapping metadata is stale or inconsistent.")
    if "chaser_index" in output_contract.key_array.components:
        component_index = output_contract.key_array.components.index("chaser_index")
        chaser_values = (
            output_keys
            if output_keys.ndim == 1
            else output_keys[:, component_index]
        )
        if not np.all(chaser_values == mapping["source_chaser_index"]):
            _fail("Refined stimulus_state_key does not match the selected chaser.")
    selected_source_positions = np.asarray(source.point_node[:])[source_rows]
    if (
        mapping["source_selected_position_content_sha256"]
        != identity_array_content_sha256(selected_source_positions)
    ):
        _fail("Exact selected source coordinate values changed after refinement.")

    try:
        manifest_bound = bind_persisted_coordinate_record(
            refined_group,
            attr_name=SURFACE_MANIFEST_ATTR,
            digest_attr_name=SURFACE_MANIFEST_DIGEST_ATTR,
        )
        processing_bound = bind_persisted_coordinate_record(
            refined_group,
            attr_name=PROCESSING_RECORD_ATTR,
            digest_attr_name=PROCESSING_RECORD_DIGEST_ATTR,
        )
    except Exception as exc:
        raise CanonicalOnlineRefinementError(
            f"Refined surface or processing record cannot be rebound exactly: {exc}."
        ) from exc
    manifest = manifest_bound.record
    _require_exact_fields(
        manifest,
        {"schema_id", "schema_version", "surfaces"},
        label=SURFACE_MANIFEST_ATTR,
    )
    if manifest != _surface_manifest(refined_group):
        _fail("Refined coordinate-surface manifest is incomplete or stale.")
    processing = processing_bound.record
    _require_exact_fields(
        processing,
        _PROCESSING_RECORD_FIELDS,
        label=PROCESSING_RECORD_ATTR,
    )
    if (
        processing["schema_id"] != PROCESSING_RECORD_SCHEMA_ID
        or processing["schema_version"] != PROCESSING_RECORD_SCHEMA_VERSION
        or processing["operation"]
        != "smooth_then_outlier_filter_then_gap_interpolation"
        or processing["temporal_adjacency_rule"]
        != "source_acquisition_frame_index_difference_equals_one"
        or processing["parameters"] != refined_group.attrs.get("parameters")
        or processing["provenance"] != refined_group.attrs.get("provenance")
        or processing["source_mapping_ref"] != mapping_bound.record_ref
        or processing["source_mapping_sha256"] != mapping_bound.record_sha256
        or processing["surface_manifest_ref"] != manifest_bound.record_ref
        or processing["surface_manifest_sha256"] != manifest_bound.record_sha256
        or processing != _processing_record(
            refined_group,
            parameters=processing["parameters"],
            provenance=processing["provenance"],
            source_mapping=mapping_bound,
            surface_manifest=manifest_bound,
        )
    ):
        _fail("Refined coordinate processing record is stale or inconsistent.")
    expected_surface_refs = {
        item["array_ref"] for item in manifest["surfaces"]
    }
    descriptor_nodes: dict[str, zarr.Array] = {}
    for node in _walk_run_nodes(refined_group):
        attrs = getattr(node, "attrs", {})
        if COORDINATE_DESCRIPTOR_ATTR not in attrs:
            continue
        if not isinstance(node, zarr.Array):
            _fail("Coordinate descriptors may not be stored on refined-run groups.")
        descriptor_nodes[f"/{node.path}"] = node
    if set(descriptor_nodes) != expected_surface_refs:
        _fail(
            "Descriptor-bearing refined coordinate surfaces do not equal the "
            "closed surface manifest."
        )

    mapping_ref = f"/{refined_group.path}@{SOURCE_MAPPING_ATTR}"
    mapping_digest = refined_group.attrs[SOURCE_MAPPING_DIGEST_ATTR]
    descriptor_bindings: list[
        tuple[str, BoundCanonicalCoordinateDescriptor]
    ] = []
    for surface in manifest["surfaces"]:
        node = descriptor_nodes[surface["array_ref"]]
        if tuple(int(item) for item in node.shape) != (row_count, 2):
            _fail("Refined coordinate surface row count or component shape is invalid.")
        raw = node.attrs.get(COORDINATE_DESCRIPTOR_ATTR)
        if not isinstance(raw, Mapping) or raw.get("schema_version") != 2:
            _fail("Every refined coordinate surface must carry schema version 2.")
        if (
            node.attrs.get("semantic_role") != surface["semantic_role"]
            or node.attrs.get("coordinate_stage") != surface["coordinate_stage"]
            or node.attrs.get("source_coordinate_mapping_ref") != mapping_ref
            or node.attrs.get("source_coordinate_mapping_sha256") != mapping_digest
            or surface["content_sha256"]
            != identity_array_content_sha256(np.asarray(node[:]))
        ):
            _fail("Refined coordinate surface role or source-mapping link is stale.")
        try:
            bound = load_bound_canonical_coordinate_descriptor(
                node,
                row_identity=output_identity,
                reference_frame_authority=source.reference_frame_authority,
                transform_chain=source.transform_chain,
                lineage_records=(
                    *source.lineage_records,
                    mapping_bound,
                    processing_bound,
                    manifest_bound,
                ),
            )
        except Exception as exc:
            raise CanonicalOnlineRefinementError(
                f"Refined coordinate descriptor cannot be rebound exactly: {exc}."
            ) from exc
        descriptor = bound.descriptor
        if (
            descriptor.profile_id != CANONICAL_ARENA_PROFILE
            or descriptor.geometry_type != "point_xy"
            or descriptor.components != ("x", "y")
            or descriptor.component_units != ("px", "px")
            or descriptor.source_camera_overlay.status
            != CANONICAL_OVERLAY_REQUIRES_TRANSFORM
            or not descriptor.source_camera_overlay.transform_refs
        ):
            _fail("Refined positions do not preserve canonical native arena coordinates.")
        descriptor_bindings.append((surface["coordinate_stage"], bound))
    descriptors = [item.descriptor for _, item in descriptor_bindings]
    if descriptors[0] != descriptors[1]:
        _fail("Filtered and interpolated coordinates do not share one exact contract.")

    expected_row_arrays = {
        f"{refined_group.path}/{STIMULUS_STATE_KEY_ARRAY_REF}",
        f"{refined_group.path}/camera_frame_ids",
        f"{refined_group.path}/{SOURCE_ACQUISITION_FRAME_INDEX_ARRAY}",
        f"{refined_group.path}/source_row_indices",
        f"{refined_group.path}/original_valid_mask",
        f"{refined_group.path}/smoothed_mask",
        f"{refined_group.path}/outlier_mask",
        f"{refined_group.path}/filtered/positions_px",
        f"{refined_group.path}/filtered/valid_mask",
        f"{refined_group.path}/interpolated/positions_px",
        f"{refined_group.path}/interpolated/valid_mask",
        f"{refined_group.path}/interpolated/interpolation_mask",
    }
    actual_array_paths = {
        node.path
        for node in _walk_run_nodes(refined_group)
        if isinstance(node, zarr.Array)
    }
    if actual_array_paths != expected_row_arrays:
        _fail("Refined run contains missing or undeclared row arrays.")
    for path in expected_row_arrays:
        node = root[path]
        if not node.shape or int(node.shape[0]) != row_count:
            _fail(f"Refined row array /{path} is not aligned to stimulus_state_key.")
    for path in (
        f"{refined_group.path}/original_valid_mask",
        f"{refined_group.path}/smoothed_mask",
        f"{refined_group.path}/outlier_mask",
        f"{refined_group.path}/filtered/valid_mask",
        f"{refined_group.path}/interpolated/valid_mask",
        f"{refined_group.path}/interpolated/interpolation_mask",
    ):
        if np.dtype(root[path].dtype) != np.dtype(bool):
            _fail(f"Refined mask /{path} must be boolean.")
    original_valid = np.asarray(
        root[f"{refined_group.path}/original_valid_mask"][:], dtype=bool
    )
    smoothed = np.asarray(
        root[f"{refined_group.path}/smoothed_mask"][:], dtype=bool
    )
    outliers = np.asarray(
        root[f"{refined_group.path}/outlier_mask"][:], dtype=bool
    )
    filtered_values = np.asarray(
        root[f"{refined_group.path}/filtered/positions_px"][:]
    )
    filtered_valid = np.asarray(
        root[f"{refined_group.path}/filtered/valid_mask"][:], dtype=bool
    )
    interpolated_values = np.asarray(
        root[f"{refined_group.path}/interpolated/positions_px"][:]
    )
    final_valid = np.asarray(
        root[f"{refined_group.path}/interpolated/valid_mask"][:], dtype=bool
    )
    interpolation = np.asarray(
        root[f"{refined_group.path}/interpolated/interpolation_mask"][:],
        dtype=bool,
    )
    exact_source_valid = np.isfinite(selected_source_positions).all(axis=1)
    if (
        not np.array_equal(original_valid, exact_source_valid)
        or np.any(smoothed & ~original_valid)
        or np.any(outliers & ~smoothed)
        or not np.array_equal(filtered_valid, np.isfinite(filtered_values).all(axis=1))
        or not np.array_equal(filtered_valid, smoothed & ~outliers)
        or not np.array_equal(final_valid, np.isfinite(interpolated_values).all(axis=1))
        or np.any(interpolation & ~final_valid)
        or np.any(interpolation & filtered_valid)
    ):
        _fail("Refined masks do not exactly describe their row-aligned coordinate values.")
    acquisition_values = acquisition_frames.copy()
    acquisition_values.setflags(write=False)
    return BoundRefinedOnlineCoordinateEvidence(
        archive_identity=archive_identity(refined_group),
        publication_status=expected_status,
        selector_eligible=expected_selector_eligible,
        row_identity=output_identity,
        source_temporal_authority=output_temporal_authority,
        source_acquisition_frame_index=acquisition_values,
        descriptor_bindings=tuple(descriptor_bindings),
        source_mapping=mapping_bound,
        processing_record=processing_bound,
        surface_manifest=manifest_bound,
        source_stimulus_temporal_authority=source.source_temporal_authority,
        _root=root,
        _refined_group=refined_group,
        _seal=_BOUND_REFINED_ONLINE_EVIDENCE_SEAL,
    )


def load_bound_refined_online_coordinate_evidence(
    root: zarr.Group,
    refined_group: zarr.Group,
) -> BoundRefinedOnlineCoordinateEvidence:
    """Load one complete, selector-eligible refined-online publication.

    Staged evidence is deliberately private to the producer transaction. Public
    consumers cannot opt out of generic completion or publication validation.
    """

    return _load_bound_refined_online_coordinate_evidence(
        root,
        refined_group,
        require_complete=True,
        expected_selector_eligible=True,
    )


def _validate_refined_online_run(
    root: zarr.Group,
    refined_group: zarr.Group,
    *,
    require_complete: bool,
    expected_selector_eligible: bool,
) -> tuple[CanonicalCoordinateDescriptor, CanonicalCoordinateDescriptor]:
    """Validate staged or complete evidence inside the producer transaction."""

    evidence = _load_bound_refined_online_coordinate_evidence(
        root,
        refined_group,
        require_complete=require_complete,
        expected_selector_eligible=expected_selector_eligible,
    )
    return (
        evidence.descriptor_for("filtered").descriptor,
        evidence.descriptor_for("interpolated").descriptor,
    )


def validate_refined_online_run(
    root: zarr.Group,
    refined_group: zarr.Group,
) -> tuple[CanonicalCoordinateDescriptor, CanonicalCoordinateDescriptor]:
    """Validate one complete refined-online publication for public use."""

    return _validate_refined_online_run(
        root,
        refined_group,
        require_complete=True,
        expected_selector_eligible=True,
    )


def _write_refined_online_activation_attr(attrs: Any, name: str, value: Any) -> None:
    """One injectable activation write for hostile-concurrency tests."""

    write_activation_attr(attrs, name, value)


def _activate_refined_online_run(
    root: zarr.Group,
    refined_runs: zarr.Group,
    refined_group: zarr.Group,
    *,
    run_name: str,
    expected_publication_owner_uuid: str,
) -> None:
    """Lease one parent generation and expose eligibility as the commit point."""

    expected_path = f"{REFINED_ONLINE_GROUP}/{run_name}"

    def proof() -> tuple[Any, ...]:
        candidate = root[expected_path]
        evidence = _load_bound_refined_online_coordinate_evidence(
            root,
            candidate,
            require_complete=True,
            expected_selector_eligible=False,
        )
        return (
            candidate.attrs.get(REFINED_ONLINE_PUBLICATION_OWNER_ATTR),
            candidate.attrs.get(RUN_COMPLETION_STATUS_ATTR),
            candidate.attrs.get(RUN_COMPLETED_AT_ATTR),
            _canonical_mapping_digest(candidate.attrs.get("run_provenance", {})),
            evidence.archive_identity,
            evidence.row_identity.record_ref,
            evidence.row_identity.record_sha256,
            evidence.source_temporal_authority.record_sha256,
            evidence.source_mapping.record_sha256,
            evidence.processing_record.record_sha256,
            evidence.surface_manifest.record_sha256,
            tuple(
                (name, binding.descriptor.digest())
                for name, binding in evidence.descriptor_bindings
            ),
        )

    try:
        activate_selector_eligible_run(
            root,
            refined_runs,
            refined_group,
            parent_path=REFINED_ONLINE_GROUP,
            run_path=expected_path,
            run_name=run_name,
            owner_attr=REFINED_ONLINE_PUBLICATION_OWNER_ATTR,
            expected_owner_uuid=expected_publication_owner_uuid,
            policy_attr=REFINED_ONLINE_PUBLICATION_POLICY_ATTR,
            generation_attr=REFINED_ONLINE_PUBLICATION_GENERATION_ATTR,
            lease_attr=REFINED_ONLINE_PARENT_PUBLICATION_LEASE_ATTR,
            policy=REFINED_ONLINE_PUBLICATION_POLICY,
            lease_schema_id="palette.refined_online_publication_lease",
            proof_loader=proof,
            selector_attrs=(RUN_LATEST_COMPLETE_ATTR, "latest"),
            attr_writer=_write_refined_online_activation_attr,
        )
    except SelectorActivationError as exc:
        raise CanonicalOnlineRefinementError(
            f"Refined-online activation lost exact ownership: {exc}."
        ) from exc


def refine_online_positions(
    zarr_path: str,
    stimulus_run: Optional[str] = None,
    chaser_index: int = 0,
    window_length: int = 11,
    polyorder: int = 3,
    displacement_threshold: float = 100.0,
    max_gap: int = 20,
    console: Optional[Console] = None,
    created_at_utc: Optional[str] = None,
) -> str:
    """Refine online target positions with smoothing, outlier removal, and interpolation.

    Args:
        zarr_path: Path to zarr archive
        stimulus_run: Stimulus run name (defaults to latest)
        chaser_index: Chaser index to load
        window_length: Savitzky-Golay filter window length (must be odd)
        polyorder: Polynomial order for Savitzky-Golay filter
        displacement_threshold: Maximum reasonable displacement in native coordinate units
        max_gap: Maximum gap size to interpolate (frames)
        console: Rich console for output
        created_at_utc: Optional creation timestamp

    Returns:
        Name of created refined run
    """
    if console is None:
        console = Console()

    console.rule("[bold]Online Detection Refinement[/bold]")
    start_time = time.perf_counter()

    # Step 1: Load exact canonical rows before creating an output run.
    console.print("[bold]Step 1: Loading Online Data[/bold]")
    input_data = load_online_positions(
        zarr_path, stimulus_run, chaser_index, console
    )
    camera_frame_ids = input_data.camera_frame_ids
    frames = input_data.source_acquisition_frame_index
    positions = input_data.positions
    valid_mask = input_data.valid_mask
    coordinate_descriptor = input_data.source_descriptor
    original_valid_frames = int(valid_mask.sum())
    original_coverage = float(original_valid_frames / len(frames) * 100)

    console.print(f"  Stimulus run: [cyan]{input_data.stimulus_run}[/cyan]")
    console.print(f"  Identified stimulus-state rows: {len(frames)}")
    console.print(
        f"  Valid positions: {original_valid_frames} ({original_coverage:.1f}%)"
    )
    console.print(f"  Coordinate space: {coordinate_descriptor.space_id}")

    # Step 2: Smooth positions
    console.print("\n[bold]Step 2: Smoothing Positions[/bold]")
    console.print(f"  Window length: {window_length}")
    console.print(f"  Polynomial order: {polyorder}")

    smoothed_positions, smoothed_mask = smooth_positions(
        positions, frames, valid_mask, window_length, polyorder
    )

    console.print(f"  Smoothed frames: {smoothed_mask.sum()}")

    # Step 3: Detect outliers
    console.print("\n[bold]Step 3: Detecting Outliers[/bold]")
    native_units = (
        coordinate_descriptor.component_units[0]
        if len(set(coordinate_descriptor.component_units)) == 1
        else "native coordinate units"
    )
    console.print(f"  Displacement threshold: {displacement_threshold} {native_units}")

    outlier_mask, outlier_stats = detect_outliers(
        smoothed_positions, frames, smoothed_mask, displacement_threshold
    )

    console.print(f"  Outliers detected: {outlier_stats['outliers_detected']} ({outlier_stats['outlier_rate']:.2f}%)")

    # Remove outliers
    clean_mask = smoothed_mask & ~outlier_mask
    clean_positions = smoothed_positions.copy()
    clean_positions[~clean_mask] = np.nan

    # Step 4: Interpolate gaps
    console.print("\n[bold]Step 4: Interpolating Gaps[/bold]")
    console.print(f"  Max gap: {max_gap} frames")

    interpolated_positions, interpolation_mask, interp_stats = interpolate_gaps(
        clean_positions, frames, clean_mask, max_gap
    )

    console.print(f"  Gaps filled: {interp_stats['gaps_filled']}")
    console.print(f"  Interpolated frames: {interp_stats['interpolated_frames']}")

    # Final statistics
    final_valid = np.isfinite(interpolated_positions[:, 0]) & np.isfinite(interpolated_positions[:, 1])
    final_coverage = float(final_valid.sum() / len(frames) * 100)

    console.print("\n[bold]Coverage Comparison:[/bold]")
    console.print(
        f"  Original: {original_valid_frames} rows ({original_coverage:.1f}%)"
    )
    console.print(f"  After smoothing: {smoothed_mask.sum()} frames ({smoothed_mask.sum()/len(frames)*100:.1f}%)")
    console.print(f"  After outlier removal: {clean_mask.sum()} frames ({clean_mask.sum()/len(frames)*100:.1f}%)")
    console.print(f"  After interpolation: {final_valid.sum()} frames ({final_coverage:.1f}%)")

    # Step 5: Save
    console.print("\n[bold]Step 5: Saving Refined Run[/bold]")

    root = zarr.open(zarr_path, mode="a")

    # The live handoff retains exact store identity.  Recheck it before any
    # output parent or staging run is created so a same-path clone cannot
    # authorize writes in another archive.
    try:
        input_data.bound_source_handoff.assert_verified()
    except Exception as exc:
        raise CanonicalOnlineRefinementError(
            f"Canonical source handoff changed before publication: {exc}."
        ) from exc
    if archive_identity(root) != input_data.bound_source_handoff.archive_identity:
        _fail(
            "Canonical source handoff belongs to a different archive/store than "
            "the requested refined output."
        )

    refined_runs = require_runs_parent(root, REFINED_ONLINE_GROUP)

    occupied_pending = refined_runs.attrs.get(RUN_LATEST_PENDING_ATTR)
    if occupied_pending is not None and str(occupied_pending).strip():
        _fail(
            "Refined-online publication refused an occupied latest_pending; "
            "the existing writer must finish or fail before another starts."
        )

    # Create timestamped run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"refined_online_{timestamp}"
    publication_owner_uuid = str(uuid.uuid4())
    pointer_names = (
        "latest",
        RUN_LATEST_COMPLETE_ATTR,
        RUN_LATEST_PENDING_ATTR,
    )
    pointer_snapshot = {
        name: (
            name in refined_runs.attrs,
            copy.deepcopy(refined_runs.attrs.get(name)),
        )
        for name in pointer_names
    }
    try:
        refined_group = refined_runs.create_group(
            run_name,
            attributes={
                REFINED_ONLINE_PUBLICATION_OWNER_ATTR: publication_owner_uuid,
                "stage_selector_eligible": False,
                "publication_status": "staging",
            },
        )
        mark_run_started(
            refined_group,
            run_name=run_name,
            stage="refine_online_detect",
        )
        refined_runs.attrs[RUN_LATEST_PENDING_ATTR] = run_name
        if refined_runs.attrs.get(RUN_LATEST_PENDING_ATTR) != run_name:
            _fail("Refined-online latest_pending acquisition did not persist.")
        refined_group.attrs.update(
            {
                "publication_status": "staging",
                "coordinate_contract_epoch": COORDINATE_CONTRACT_EPOCH,
                "source_stimulus_run": input_data.stimulus_run,
                "chaser_index": int(chaser_index),
                "stage_selector_eligible": False,
            }
        )

        # Rebind after opening the mutable store so cached handoff values cannot
        # authorize a write after their source has changed.
        try:
            input_data.bound_source_handoff.assert_verified()
        except Exception as exc:
            raise CanonicalOnlineRefinementError(
                f"Canonical source handoff changed during publication: {exc}."
            ) from exc
        source_handoff = input_data.source_handoff
        key_mapping = source_handoff.get("stimulus_state_key_mapping")
        if not isinstance(key_mapping, Mapping):
            _fail("Canonical source key mapping vanished before publication.")
        source = _resolve_persisted_canonical_source(
            root,
            stimulus_run=input_data.stimulus_run,
            point_path=source_handoff.get("source_path"),
            rowset_path=source_handoff.get("rowset_path"),
            key_path=key_mapping.get("source_path"),
        )
        source_rows = np.asarray(input_data.source_row_indices, dtype=np.int64)
        if (
            source.descriptor != input_data.source_descriptor
            or tuple(
                (record.record_ref, record.record_sha256)
                for record in source.lineage_records
            )
            != tuple(
                (record.record_ref, record.record_sha256)
                for record in (
                    input_data.bound_source_handoff.surface_manifest,
                    input_data.bound_source_handoff.camera_mapping,
                    input_data.bound_source_handoff.frame_transform.manifest,
                    input_data.bound_source_handoff.import_lineage,
                    input_data.bound_source_handoff.output_manifest,
                )
            )
            or not np.array_equal(
                np.asarray(source.key_node[:])[source_rows],
                input_data.stimulus_state_key,
            )
            or not np.array_equal(
                np.asarray(source.point_node[:], dtype=np.float64)[source_rows],
                positions,
                equal_nan=True,
            )
            or not np.array_equal(
                source.source_acquisition_frame_index[source_rows],
                frames,
            )
            or not np.array_equal(
                source.camera_frame_ids[source_rows],
                camera_frame_ids,
            )
        ):
            _fail("Canonical source evidence changed before refined publication.")

        key_values = np.asarray(input_data.stimulus_state_key)
        key_node = refined_group.create_array(
            STIMULUS_STATE_KEY_ARRAY_REF,
            data=key_values,
            chunks=_array_chunks(key_values),
        )
        identity_contract = build_row_identity_contract(
            domain=STIMULUS_STATE_DOMAIN,
            values=key_values,
            components=input_data.stimulus_state_key_components,
        )
        output_identity = stamp_and_bind_row_identity_contract(
            refined_group,
            key_node,
            contract=identity_contract,
        )

        frames_node = refined_group.create_array(
            "camera_frame_ids",
            data=camera_frame_ids.astype(np.int64, copy=False),
            chunks=_array_chunks(camera_frame_ids),
        )
        source_acquisition_node = refined_group.create_array(
            SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
            data=frames.astype(np.int64, copy=False),
            chunks=_array_chunks(frames),
        )
        source_acquisition_node.attrs.update(
            {
                "semantic_role": "source_acquisition_frame_index",
                "target_domain": "acquisition_frame",
                "primary_row_identity": False,
                "source_rowset_ref": f"/{source.rowset_node.path}",
                "source_array_ref": f"/{source.source_acquisition_node.path}",
            }
        )
        try:
            stamp_source_row_temporal_authority(
                refined_group,
                source_acquisition_node,
                source_row_identity=output_identity,
                acquisition_frame=source.frame_transform.acquisition_frame,
            )
        except Exception as exc:
            raise CanonicalOnlineRefinementError(
                f"Unable to seal refined acquisition-frame authority: {exc}."
            ) from exc
        frames_node.attrs.update(
            {
                "semantic_role": "mapped_camera_frame_id",
                "primary_row_identity": False,
            }
        )
        source_rows_node = refined_group.create_array(
            "source_row_indices",
            data=source_rows,
            chunks=_array_chunks(source_rows),
        )
        source_rows_node.attrs.update(
            {
                "semantic_role": "source_row_mapping",
                "source_rowset_ref": f"/{source.rowset_node.path}",
                "primary_row_identity": False,
            }
        )
        for name, values in (
            ("original_valid_mask", valid_mask),
            ("smoothed_mask", smoothed_mask),
            ("outlier_mask", outlier_mask),
        ):
            refined_group.create_array(
                name,
                data=np.asarray(values, dtype=bool),
                chunks=_array_chunks(np.asarray(values)),
            )

        filtered_grp = refined_group.create_group("filtered")
        filtered_positions_array = filtered_grp.create_array(
            "positions_px",
            data=clean_positions,
            chunks=_array_chunks(clean_positions),
        )
        filtered_grp.create_array(
            "valid_mask",
            data=np.asarray(clean_mask, dtype=bool),
            chunks=_array_chunks(clean_mask),
        )
        filtered_grp.attrs.update(
            {
                "total_rows": len(frames),
                "valid_rows": int(clean_mask.sum()),
                "coverage_percent": float(clean_mask.sum() / len(frames) * 100),
                "smoothing_applied": True,
                "outliers_removed": outlier_stats["outliers_detected"],
            }
        )

        interp_grp = refined_group.create_group("interpolated")
        interpolated_positions_array = interp_grp.create_array(
            "positions_px",
            data=interpolated_positions,
            chunks=_array_chunks(interpolated_positions),
        )
        interp_grp.create_array(
            "valid_mask",
            data=np.asarray(final_valid, dtype=bool),
            chunks=_array_chunks(final_valid),
        )
        interp_grp.create_array(
            "interpolation_mask",
            data=np.asarray(interpolation_mask, dtype=bool),
            chunks=_array_chunks(interpolation_mask),
        )
        interp_grp.attrs.update(
            {
                "total_rows": len(frames),
                "valid_rows": int(final_valid.sum()),
                "coverage_percent": final_coverage,
                "gaps_filled": interp_stats["gaps_filled"],
                "interpolated_rows": interp_stats["interpolated_frames"],
            }
        )

        source_mapping = _source_mapping_record(
            refined_group,
            source=source,
            input_data=input_data,
            output_identity=output_identity,
        )
        source_mapping_bound = stamp_and_bind_persisted_coordinate_record(
            refined_group,
            source_mapping,
            attr_name=SOURCE_MAPPING_ATTR,
            digest_attr_name=SOURCE_MAPPING_DIGEST_ATTR,
        )
        mapping_ref = source_mapping_bound.record_ref
        for stage, node in (
            ("filtered", filtered_positions_array),
            ("interpolated", interpolated_positions_array),
        ):
            node.attrs.update(
                {
                    "semantic_role": "target_position",
                    "coordinate_stage": stage,
                    "source_coordinate_mapping_ref": mapping_ref,
                    "source_coordinate_mapping_sha256": (
                        source_mapping_bound.record_sha256
                    ),
                }
            )

        manifest_bound = stamp_and_bind_persisted_coordinate_record(
            refined_group,
            _surface_manifest(refined_group),
            attr_name=SURFACE_MANIFEST_ATTR,
            digest_attr_name=SURFACE_MANIFEST_DIGEST_ATTR,
        )

        duration = time.perf_counter() - start_time
        created_timestamp = created_at_utc or datetime.now(timezone.utc).isoformat()
        parameters = {
            "window_length": window_length,
            "polyorder": polyorder,
            "displacement_threshold": displacement_threshold,
            "displacement_threshold_units": native_units,
            "max_gap": max_gap,
        }
        coverage_stats = {
            "original": {
                "valid_rows": original_valid_frames,
                "coverage_percent": original_coverage,
            },
            "smoothed": {
                "valid_rows": int(smoothed_mask.sum()),
                "coverage_percent": float(smoothed_mask.sum() / len(frames) * 100),
            },
            "clean": {
                "valid_rows": int(clean_mask.sum()),
                "coverage_percent": float(clean_mask.sum() / len(frames) * 100),
                "outliers_removed": outlier_stats["outliers_detected"],
            },
            "final": {
                "valid_rows": int(final_valid.sum()),
                "coverage_percent": final_coverage,
                "interpolated_rows": interp_stats["interpolated_frames"],
            },
        }
        git_info = get_git_info()
        env_info = get_environment_info()
        platform_info = env_info.get("platform", {})
        provenance_record = {
            "stage": "refine_online_detect",
            "command": " ".join(sys.argv),
            "created_at_utc": created_timestamp,
            "version": git_info.get("short_hash") or git_info.get("commit_hash"),
            "git": {
                "commit": git_info.get("commit_hash"),
                "short": git_info.get("short_hash"),
                "branch": git_info.get("branch"),
                "is_dirty": git_info.get("is_dirty"),
                "remote": git_info.get("remote_url"),
            },
            "environment": {
                "hostname": platform_info.get("hostname", "unknown"),
                "python_version": platform_info.get("python_version", "unknown"),
                "system": platform_info.get("system", "unknown"),
                "release": platform_info.get("release", "unknown"),
            },
            "parameters": parameters,
            "inputs": {
                "stimulus_run": input_data.stimulus_run,
                "chaser_index": chaser_index,
                "source_coordinate_array_ref": f"/{source.point_node.path}",
                "source_coordinate_descriptor_sha256": source.descriptor.digest(),
                "source_row_identity_contract_ref": source.identity.record_ref,
                "source_row_identity_contract_sha256": source.identity.record_sha256,
            },
        }
        refined_group.attrs.update(
            {
                "refinement_timestamp": created_timestamp,
                "processing_time_seconds": float(duration),
                "operations": ["smooth", "outlier_removal", "interpolate"],
                "parameters": parameters,
                "coverage_stats": coverage_stats,
                "outlier_stats": outlier_stats,
                "interpolation_stats": interp_stats,
                "provenance": provenance_record,
            }
        )

        processing_bound = stamp_and_bind_persisted_coordinate_record(
            refined_group,
            _processing_record(
                refined_group,
                parameters=parameters,
                provenance=provenance_record,
                source_mapping=source_mapping_bound,
                surface_manifest=manifest_bound,
            ),
            attr_name=PROCESSING_RECORD_ATTR,
            digest_attr_name=PROCESSING_RECORD_DIGEST_ATTR,
        )
        lineage_records = (
            *source.lineage_records,
            source_mapping_bound,
            processing_bound,
            manifest_bound,
        )
        bindings = tuple(
            build_bound_canonical_coordinate_descriptor(
                node,
                profile_id=CANONICAL_ARENA_PROFILE,
                geometry_type="point_xy",
                components=("x", "y"),
                component_units=("px", "px"),
                pixel_convention=coordinate_descriptor.pixel_convention,
                row_identity=output_identity,
                reference_frame_authority=source.reference_frame_authority,
                source_camera_overlay_status=(
                    CANONICAL_OVERLAY_REQUIRES_TRANSFORM
                ),
                transform_chain=source.transform_chain,
                lineage_records=lineage_records,
            )
            for node in (filtered_positions_array, interpolated_positions_array)
        )
        stamp_bound_canonical_coordinate_descriptors(bindings)

        # Validate every persisted surface while the run is still
        # selector-ineligible. Generic completion is also written while the run
        # remains ineligible, so it cannot become an implicit latest candidate.
        _validate_refined_online_run(
            root,
            refined_group,
            require_complete=False,
            expected_selector_eligible=False,
        )
        refined_group.attrs["publication_status"] = "complete"
        run_provenance = build_run_provenance_from_stage_record(
            provenance_record,
            fallback_command="palette refine-online-detect",
        )
        if refined_runs.attrs.get(RUN_LATEST_PENDING_ATTR) != run_name:
            _fail("Refined-online staging ownership changed before completion.")
        if RUN_LATEST_PENDING_ATTR in refined_runs.attrs:
            del refined_runs.attrs[RUN_LATEST_PENDING_ATTR]
        mark_run_complete(
            refined_group,
            parent_group=refined_runs,
            run_name=run_name,
            run_provenance=run_provenance,
        )
        fresh_root = zarr.open(zarr_path, mode="r")
        fresh_group = fresh_root[REFINED_ONLINE_GROUP][run_name]
        fresh_evidence = _load_bound_refined_online_coordinate_evidence(
            fresh_root,
            fresh_group,
            require_complete=True,
            expected_selector_eligible=False,
        )
        fresh_evidence.assert_verified()
        _activate_refined_online_run(
            root,
            refined_runs,
            refined_group,
            run_name=run_name,
            expected_publication_owner_uuid=publication_owner_uuid,
        )
    except BaseException as publication_exc:
        cleanup_errors: list[str] = []
        failed_mark_error: BaseException | None = None
        try:
            if run_name in refined_runs:
                failed_group = refined_runs[run_name]
                if (
                    failed_group.attrs.get(REFINED_ONLINE_PUBLICATION_OWNER_ATTR)
                    == publication_owner_uuid
                ):
                    try:
                        failed_group.attrs["stage_selector_eligible"] = False
                        failed_group.attrs["publication_status"] = "failed"
                    except BaseException as cleanup_exc:  # pragma: no cover - hostile store
                        cleanup_errors.append(f"disarm failed run: {cleanup_exc}")
                    try:
                        if RUN_COMPLETED_AT_ATTR in failed_group.attrs:
                            del failed_group.attrs[RUN_COMPLETED_AT_ATTR]
                        mark_run_failed(
                            failed_group,
                            parent_group=refined_runs,
                            run_name=run_name,
                            error=str(publication_exc),
                        )
                    except BaseException as cleanup_exc:  # pragma: no cover - hostile store
                        failed_mark_error = cleanup_exc
                    try:
                        failed_group.attrs[
                            REFINED_ONLINE_PUBLICATION_TOMBSTONE_ATTR
                        ] = {
                            "schema_id": "palette.refined_online_publication_tombstone",
                            "schema_version": 1,
                            "publication_owner_uuid": publication_owner_uuid,
                            "run_name": run_name,
                            "public_path_retained": True,
                            "selector_eligible": False,
                            "retry_policy": "new_immutable_run_name_required",
                        }
                    except BaseException as cleanup_exc:  # pragma: no cover - hostile store
                        cleanup_errors.append(f"failed tombstone: {cleanup_exc}")
                    if failed_mark_error is not None:
                        cleanup_errors.append(f"failed marker: {failed_mark_error}")
                    try:
                        tombstone = failed_group.attrs.get(
                            REFINED_ONLINE_PUBLICATION_TOMBSTONE_ATTR
                        )
                        if (
                            failed_group.attrs.get(
                                REFINED_ONLINE_PUBLICATION_OWNER_ATTR
                            )
                            != publication_owner_uuid
                            or failed_group.attrs.get("stage_selector_eligible")
                            is not False
                            or failed_group.attrs.get(RUN_COMPLETION_STATUS_ATTR)
                            != "failed"
                            or not isinstance(tombstone, Mapping)
                            or tombstone.get("publication_owner_uuid")
                            != publication_owner_uuid
                        ):
                            raise RuntimeError(
                                "failed public tombstone did not persist exactly"
                            )
                    except BaseException as cleanup_exc:  # pragma: no cover - hostile store
                        cleanup_errors.append(f"verify failed tombstone: {cleanup_exc}")
        except BaseException as cleanup_exc:  # pragma: no cover - hostile store
            detail = f"staging run: {cleanup_exc}"
            if failed_mark_error is not None:
                detail += f"; failed marker: {failed_mark_error}"
            cleanup_errors.append(detail)
        try:
            for name, (was_present, previous_value) in pointer_snapshot.items():
                if name != RUN_LATEST_PENDING_ATTR:
                    continue
                if refined_runs.attrs.get(name) != run_name:
                    continue
                if was_present:
                    refined_runs.attrs[name] = previous_value
                elif name in refined_runs.attrs:
                    del refined_runs.attrs[name]
        except BaseException as cleanup_exc:  # pragma: no cover - hostile store
            cleanup_errors.append(f"selector attrs: {cleanup_exc}")
        if cleanup_errors:
            raise CanonicalOnlineRefinementError(
                f"Refined publication failed and cleanup was incomplete: {cleanup_errors!r}."
            ) from publication_exc
        raise

    _print_after_commit(
        console,
        f"[green]✓[/green] Refined run saved: {refined_group.path}",
    )
    _print_after_commit(
        console,
        f"[green]✓[/green] Processing completed in {duration:.2f} seconds",
    )

    return run_name


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Refine online target positions from stimulus runs"
    )
    parser.add_argument("zarr_path", help="Path to Palette zarr archive")
    parser.add_argument("--stimulus-run", help="Stimulus run name (defaults to latest)")
    parser.add_argument(
        "--chaser-index",
        type=int,
        default=0,
        help="Chaser index to process (default: 0)",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=11,
        help="Savitzky-Golay filter window length (must be odd, default: 11)",
    )
    parser.add_argument(
        "--polyorder",
        type=int,
        default=3,
        help="Polynomial order for Savitzky-Golay filter (default: 3)",
    )
    parser.add_argument(
        "--displacement-threshold",
        type=float,
        default=100.0,
        help="Maximum reasonable displacement in native coordinate units (default: 100)",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=20,
        help="Maximum gap size to interpolate in frames (default: 20)",
    )

    args = parser.parse_args(argv)

    console = Console()

    try:
        refine_online_positions(
            zarr_path=args.zarr_path,
            stimulus_run=args.stimulus_run,
            chaser_index=args.chaser_index,
            window_length=args.window_length,
            polyorder=args.polyorder,
            displacement_threshold=args.displacement_threshold,
            max_gap=args.max_gap,
            console=console,
        )
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
