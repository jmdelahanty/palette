"""Utilities to present online and offline chaser metrics through a common API.

The online-position path is intentionally canonical-only.  Historical
``chaser_states`` layouts belong in audit or migration tooling: a normal reader
must resolve one array-specific version-2 coordinate descriptor and the exact
persisted stimulus-state identity that owns its rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import zarr

from fisheye.shared.coordinate_descriptor import (
    COORDINATE_DESCRIPTOR_ATTR,
    COORDINATE_DESCRIPTOR_DIGEST_SUFFIX,
    CanonicalCoordinateDescriptor,
)
from fisheye.shared.canonical_coordinate_publication import (
    BoundCanonicalCoordinateDescriptor,
    load_bound_canonical_coordinate_descriptor,
)
from fisheye.shared.coordinate_identity import (
    ROW_IDENTITY_CONTRACT_ATTR,
    ROW_IDENTITY_CONTRACT_DIGEST_ATTR,
    STIMULUS_STATE_DOMAIN,
    STIMULUS_STATE_KEY_ARRAY_REF,
    STIMULUS_STATE_KEY_MODE,
    BoundRowIdentityContract,
    BoundSourceRowTemporalAuthority,
)
from fisheye.shared.coordinate_record import BoundCoordinateRecord
from fisheye.shared.coordinate_reference import BoundReferenceExtent
from fisheye.shared.archive_identity import ArchiveIdentity, archive_identity
from fisheye.shared.stimulus_coordinate_contract import (
    ARENA_GEOMETRY_RECORD_ATTR,
    ARENA_GEOMETRY_RECORD_DIGEST_ATTR,
    CAMERA_FRAME_IDS_ARRAY,
    SOURCE_ROW_INDICES_ARRAY,
    SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
    BoundStimulusCoordinateEvidence,
    StimulusCoordinateContractError,
    load_bound_stimulus_coordinate_evidence,
)
from fisheye.shared.stimulus_frame_transform import (
    BoundStimulusFrameTransformEvidence,
)
from fisheye.shared.proof_verification import proof_verification_operation
from fisheye.shared.zarr.columnar import load_structured_dataset
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)


PathLike = Union[str, Path]

CANONICAL_ONLINE_COORDINATE_HANDOFF_SCHEMA_ID = (
    "palette.canonical_online_coordinate_handoff"
)
CANONICAL_ONLINE_COORDINATE_HANDOFF_SCHEMA_VERSION = 1
_TARGET_POSITION_ARRAY = "target_position_xy"
_TARGET_POSITION_ROLE = "target_position"
_TARGET_POSITION_COMPONENT_FIELDS = ("target_pos_x", "target_pos_y")


class ChaserMetricsCoordinateContractError(ValueError):
    """Raised when canonical online coordinates cannot be interpreted exactly."""


@dataclass(frozen=True)
class CanonicalOnlineCoordinateHandoff:
    """Live exact-evidence handoff; never serialize this object into attrs."""

    coordinate_descriptor: BoundCanonicalCoordinateDescriptor
    row_identity: BoundRowIdentityContract
    arena_reference: BoundReferenceExtent
    frame_transform: BoundStimulusFrameTransformEvidence = dataclass_field(
        repr=False,
        compare=False,
    )
    source_temporal_authority: BoundSourceRowTemporalAuthority = dataclass_field(
        repr=False,
        compare=False,
    )
    surface_manifest: BoundCoordinateRecord
    import_lineage: BoundCoordinateRecord
    output_manifest: BoundCoordinateRecord
    camera_mapping: BoundCoordinateRecord
    archive_identity: ArchiveIdentity
    stimulus_state_key: np.ndarray = dataclass_field(repr=False, compare=False)
    camera_frame_ids: np.ndarray = dataclass_field(repr=False, compare=False)
    source_acquisition_frame_index: np.ndarray = dataclass_field(
        repr=False,
        compare=False,
    )
    source_row_indices: np.ndarray = dataclass_field(repr=False, compare=False)
    _run_group: Any = dataclass_field(repr=False, compare=False)
    _root_group: Any = dataclass_field(repr=False, compare=False)
    _chaser_group: Any = dataclass_field(repr=False, compare=False)
    _coordinate_node: Any = dataclass_field(repr=False, compare=False)

    @property
    def source_rowset(self) -> Any:
        """Exact persisted stimulus rowset for downstream sealed lineage."""

        return self._chaser_group

    def assert_verified(self) -> None:
        """Freshly resolve every persisted record and reject stale substitution."""

        try:
            store_path = getattr(self._run_group, "store_path", None)
            store = getattr(store_path, "store", None)
            if store is None:
                raise ChaserMetricsCoordinateContractError(
                    "Bound stimulus run no longer exposes its exact Zarr store."
                )
            if self.archive_identity.kind.startswith("store_object:"):
                # Memory/non-local stores are authorized by exact live object
                # identity. Re-resolve descendants through that same object;
                # opening in read mode may manufacture a read-only store clone.
                fresh_root = self._root_group
                fresh_run = self._run_group
            else:
                fresh_root = zarr.open_group(
                    store=store,
                    mode="r",
                )
                fresh_run = fresh_root[self._run_group.path]
            fresh_chaser = fresh_run["tracking_data"]["chaser_states"]
            coordinate_name = str(self._coordinate_node.path).rsplit("/", 1)[-1]
            fresh_coordinate_node = fresh_chaser[coordinate_name]
            current = load_bound_stimulus_coordinate_evidence(
                fresh_run,
                fresh_chaser,
                root_node=fresh_root,
            )
            current_descriptor = load_bound_canonical_coordinate_descriptor(
                fresh_coordinate_node,
                row_identity=current.row_identity,
                reference_frame_authority=(
                    current.frame_transform.arena_relative_frame
                ),
                transform_chain=current.frame_transform.transform_chain,
                lineage_records=(
                    current.surface_manifest,
                    current.camera_mapping,
                    current.frame_transform.manifest,
                    current.import_lineage,
                    current.output_manifest,
                ),
            )
            current_node_archive = archive_identity(fresh_coordinate_node)
        except Exception as exc:
            raise ChaserMetricsCoordinateContractError(
                f"Canonical coordinate handoff is no longer fresh: {exc}"
            ) from exc
        record_pairs = (
            (self.surface_manifest, current.surface_manifest),
            (self.camera_mapping, current.camera_mapping),
            (self.frame_transform.manifest, current.frame_transform.manifest),
            (self.import_lineage, current.import_lineage),
            (self.output_manifest, current.output_manifest),
        )
        mismatches: list[str] = []
        if current.archive_identity != self.archive_identity:
            mismatches.append("evidence_archive")
        if self.archive_identity != current_node_archive:
            mismatches.append("coordinate_archive")
        if (
            current.row_identity.record_ref != self.row_identity.record_ref
            or current.row_identity.record_sha256 != self.row_identity.record_sha256
        ):
            mismatches.append("row_identity")
        if (
            current.arena_reference.record_ref != self.arena_reference.record_ref
            or current.arena_reference.record_sha256
            != self.arena_reference.record_sha256
        ):
            mismatches.append("arena_reference")
        if (
            current_descriptor.descriptor.digest()
            != self.coordinate_descriptor.descriptor.digest()
        ):
            mismatches.append("coordinate_descriptor")
        if any(
            left.record_ref != right.record_ref
            or left.record_sha256 != right.record_sha256
            for left, right in record_pairs
        ):
            mismatches.append("lineage_records")
        if not np.array_equal(current.stimulus_state_key, self.stimulus_state_key):
            mismatches.append("stimulus_state_key")
        if not np.array_equal(current.camera_frame_ids, self.camera_frame_ids):
            mismatches.append("camera_frame_ids")
        if not np.array_equal(
            current.source_acquisition_frame_index,
            self.source_acquisition_frame_index,
        ):
            mismatches.append("source_acquisition_frame_index")
        if (
            current.source_temporal_authority.record_ref
            != self.source_temporal_authority.record_ref
            or current.source_temporal_authority.record_sha256
            != self.source_temporal_authority.record_sha256
        ):
            mismatches.append("source_temporal_authority")
        if not np.array_equal(current.source_row_indices, self.source_row_indices):
            mismatches.append("source_row_indices")
        if mismatches:
            raise ChaserMetricsCoordinateContractError(
                "Canonical coordinate handoff changed after it was bound: "
                f"{mismatches!r}."
            )


@dataclass
class ChaserMetricsBundle:
    """Normalized chaser metrics aligned by camera frame.

    Attributes
    ----------
    camera_frame_ids:
        Sorted camera-frame identifiers spanning the stimulus run.
    stimulus_frame_nums:
        Stimulus frame numbers aligned to ``camera_frame_ids``.
    timestamp_ns:
        Timestamps (nanoseconds) for each camera frame when available; -1 when unknown.
    trial_state:
        Trial state enum per frame (-1 when unavailable).
    metadata_mask:
        Boolean mask indicating whether the underlying metadata row is original (True)
        or interpolated/synthesized (False). ``None`` if no mask was stored.
    online:
        Dictionary of online chaser fields (arrays aligned to ``camera_frame_ids``).
    offline:
        Dictionary of offline metrics (arrays aligned to ``camera_frame_ids``). Vector
        quantities use shape ``(n_frames, 2)`` and scalars ``(n_frames,)``. Includes a
        ``"has_offline"`` boolean mask.
    provenance:
        Provenance details such as the stimulus run, metrics run, keypoint source, and
        chaser index.
    online_coordinate_metadata:
        Canonical handoff for the exact ``target_position_xy`` child array.  It
        contains that array's path and unmodified attrs, its validated version-2
        descriptor, the owning rowset identity contract, and an exact mapping
        from ``stimulus_state_key`` rows into this bundle's camera-frame rows.
        Parent-group coordinate attrs are never coordinate authority.
    """

    camera_frame_ids: np.ndarray
    stimulus_frame_nums: np.ndarray
    timestamp_ns: np.ndarray
    trial_state: np.ndarray
    metadata_mask: Optional[np.ndarray]
    online: Dict[str, np.ndarray]
    offline: Dict[str, np.ndarray]
    provenance: Dict[str, object]
    online_coordinate_metadata: Dict[str, object] = dataclass_field(default_factory=dict)
    online_coordinate_handoff: Optional[CanonicalOnlineCoordinateHandoff] = None


@proof_verification_operation
def load_chaser_metrics(
    zarr_path: PathLike,
    *,
    stimulus_run: Optional[str] = None,
    metrics_run: Optional[str] = None,
    chaser_index: int = 0,
) -> ChaserMetricsBundle:
    """Load online and offline chaser metrics in a unified layout."""

    root = zarr.open(str(zarr_path), mode="r")
    analysis_group = root.get("analysis")
    if analysis_group is None:
        raise ValueError("Zarr store does not contain analysis group.")

    stimulus_parent = analysis_group.get("stimulus_runs")
    if stimulus_parent is None:
        raise ValueError("Zarr store does not contain analysis/stimulus_runs.")
    stim_run = stimulus_run or _resolve_latest(stimulus_parent)
    if stim_run is None:
        raise ValueError(
            "Stimulus run not provided and no 'latest_complete' attribute present."
        )
    if stim_run not in stimulus_parent:
        raise ValueError(f"Stimulus run '{stim_run}' not found under analysis/stimulus_runs.")

    stim_group = stimulus_parent[stim_run]
    tracking_group = stim_group.get("tracking_data")
    chaser_group = (
        tracking_group.get("chaser_states")
        if isinstance(tracking_group, zarr.Group)
        else None
    )
    if not isinstance(chaser_group, zarr.Group):
        raise ChaserMetricsCoordinateContractError(
            "Stimulus run lacks canonical tracking_data/chaser_states."
        )
    camera_meta, stimulus_field, timestamp_field = _build_camera_metadata_map(stim_group)
    metadata_mask = _load_metadata_mask(stim_group, len(camera_meta))

    frames_sorted = np.array(sorted(camera_meta.keys()), dtype=np.int64)
    frame_to_index = {frame: idx for idx, frame in enumerate(frames_sorted)}

    stimulus_frames = np.array(
        [int(camera_meta[frame][stimulus_field]) for frame in frames_sorted], dtype=np.int64
    )
    if timestamp_field is not None:
        timestamps = np.array(
            [int(camera_meta[frame][timestamp_field]) for frame in frames_sorted], dtype=np.int64
        )
    else:
        timestamps = np.full(frames_sorted.shape, -1, dtype=np.int64)

    trial_state = np.full(frames_sorted.shape, -1, dtype=np.int16)
    (
        online_fields,
        online_coordinate_metadata,
        online_coordinate_handoff,
    ) = _extract_online_fields(
        root=root,
        stim_group=stim_group,
        frames_sorted=frames_sorted,
        frame_to_index=frame_to_index,
        trial_state=trial_state,
        chaser_index=chaser_index,
    )

    metrics_parent = analysis_group.get("chaser_fish_metrics")
    metrics_run_resolved: Optional[str]
    if metrics_parent is None:
        metrics_run_resolved = metrics_run
    elif metrics_run is not None:
        metrics_run_resolved = metrics_run
    else:
        metrics_run_resolved = _resolve_latest_complete_selector_eligible(
            metrics_parent
        )

    offline_fields: Dict[str, np.ndarray]
    provenance = {
        "stimulus_run": stim_run,
        "metrics_run": metrics_run_resolved,
        "chaser_index": int(chaser_index),
    }

    if metrics_parent is not None and metrics_run_resolved and metrics_run_resolved in metrics_parent:
        metrics_group = metrics_parent[metrics_run_resolved]
        _require_complete_selector_eligible_run(
            metrics_group,
            run_name=metrics_run_resolved,
            family="analysis/chaser_fish_metrics",
        )
        offline_fields = _extract_offline_fields(
            metrics_group=metrics_group,
            frames_sorted=frames_sorted,
            frame_to_index=frame_to_index,
        )
        provenance.update(
            {
                "source_stimulus_run": metrics_group.attrs.get("source_stimulus_run"),
                "source_keypoints_run": metrics_group.attrs.get("source_keypoints_run"),
            }
        )
    else:
        offline_fields = _empty_offline_fields(frames_sorted.shape[0])

    return ChaserMetricsBundle(
        camera_frame_ids=frames_sorted,
        stimulus_frame_nums=stimulus_frames,
        timestamp_ns=timestamps,
        trial_state=trial_state,
        metadata_mask=metadata_mask,
        online=online_fields,
        offline=offline_fields,
        provenance=provenance,
        online_coordinate_metadata=online_coordinate_metadata,
        online_coordinate_handoff=online_coordinate_handoff,
    )


def _resolve_latest(group: zarr.Group) -> Optional[str]:
    latest = group.attrs.get("latest_complete")
    if isinstance(latest, bytes):
        latest = latest.decode("utf-8", "ignore")
    if isinstance(latest, str) and latest in group:
        return latest
    return None


def _resolve_latest_complete_selector_eligible(
    group: zarr.Group,
) -> Optional[str]:
    latest = _resolve_latest(group)
    if latest is None:
        return None
    try:
        _require_complete_selector_eligible_run(
            group[latest],
            run_name=latest,
            family=f"/{group.path}",
        )
    except ValueError:
        return None
    return latest


def _require_complete_selector_eligible_run(
    run_group: zarr.Group,
    *,
    run_name: str,
    family: str,
) -> None:
    if (
        run_group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or run_group.attrs.get("stage_selector_eligible") is not True
    ):
        raise ValueError(
            f"{family}/{run_name} is not explicitly complete and "
            "stage-selector eligible."
        )


def _load_metadata_mask(stim_group: zarr.Group, length: int) -> Optional[np.ndarray]:
    for mask_name in ("camera_aligned_interpolation_mask", "interpolation_mask"):
        if mask_name not in stim_group:
            continue
        mask = np.asarray(stim_group[mask_name], dtype=bool)
        if mask.shape[0] == length:
            return mask
    return None


def _build_camera_metadata_map(
    stim_group: zarr.Group,
) -> Tuple[Dict[int, np.void], str, Optional[str]]:
    meta_group = stim_group.get("video_metadata")
    if not isinstance(meta_group, zarr.Group):
        raise ValueError("Stimulus run missing video_metadata group.")
    # This compatibility bundle is keyed by external camera-frame ID and cannot
    # represent duplicates. Future acquisition-time consumers must use
    # load_canonical_online_coordinate_surface, which preserves duplicate external
    # IDs as provenance and never uses them as row or temporal identity.
    aligned_metadata, _ = load_structured_dataset(meta_group, "frame_metadata")
    if aligned_metadata is None:
        raise ValueError("Stimulus run missing video_metadata/frame_metadata dataset.")

    aligned_stim_field = _resolve_struct_field(
        aligned_metadata,
        "stimulus_frame_num",
        "frame_number",
        "stim_frame_num",
    )
    aligned_camera_field = _resolve_struct_field(
        aligned_metadata,
        "triggering_camera_frame_id",
        "camera_frame_id",
    )
    timestamp_field = _maybe_resolve_struct_field(
        aligned_metadata,
        "timestamp_ns",
        "timestamp_ns_session",
        "relative_timestamp_ns",
    )

    camera_to_record: Dict[int, np.void] = {}
    seen_stimulus: set[int] = set()
    stimulus_values = _strict_integral_struct_field(
        aligned_metadata,
        aligned_stim_field,
    )
    camera_values = _strict_integral_struct_field(
        aligned_metadata,
        aligned_camera_field,
    )
    for record in aligned_metadata:
        row_index = len(camera_to_record)
        camera_id = int(camera_values[row_index])
        stimulation_id = int(stimulus_values[row_index])
        if camera_id in camera_to_record:
            raise ChaserMetricsCoordinateContractError(
                f"Video metadata contains duplicate camera frame {camera_id}."
            )
        if stimulation_id in seen_stimulus:
            raise ChaserMetricsCoordinateContractError(
                f"Video metadata contains duplicate stimulus frame {stimulation_id}."
            )
        camera_to_record[camera_id] = record
        seen_stimulus.add(stimulation_id)

    return camera_to_record, aligned_stim_field, timestamp_field


def _strict_integral_struct_field(array: np.ndarray, field: str) -> np.ndarray:
    values = np.asarray(array[field])
    if values.dtype.kind not in "iu":
        raise ChaserMetricsCoordinateContractError(
            f"Video metadata field {field!r} must be integer; fractional mappings "
            "are unsupported."
        )
    if values.dtype.kind == "u" and values.size and int(values.max()) > np.iinfo(np.int64).max:
        raise ChaserMetricsCoordinateContractError(
            f"Video metadata field {field!r} exceeds signed int64."
        )
    result = np.asarray(values, dtype=np.int64)
    if result.size and int(result.min()) < 0:
        raise ChaserMetricsCoordinateContractError(
            f"Video metadata field {field!r} contains negative identifiers."
        )
    return result


def _resolve_struct_field(array: np.ndarray, *candidates: str) -> str:
    names = array.dtype.names or ()
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise ValueError(f"Structured array missing expected field (tried {', '.join(candidates)})")


def _maybe_resolve_struct_field(array: np.ndarray, *candidates: str) -> Optional[str]:
    names = array.dtype.names or ()
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def _extract_online_fields(
    *,
    root: zarr.Group,
    stim_group: zarr.Group,
    frames_sorted: np.ndarray,
    frame_to_index: Dict[int, int],
    trial_state: np.ndarray,
    chaser_index: int,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, object],
    CanonicalOnlineCoordinateHandoff,
]:
    tracking_group = stim_group.get("tracking_data")
    if tracking_group is None:
        raise ChaserMetricsCoordinateContractError(
            "Stimulus run lacks the canonical tracking_data group."
        )
    chaser_group = tracking_group.get("chaser_states")
    if not isinstance(chaser_group, zarr.Group):
        raise ChaserMetricsCoordinateContractError(
            "Stimulus run tracking_data group lacks canonical columnar "
            "chaser_states."
        )

    chaser_states, _ = load_structured_dataset(
        tracking_group,
        "chaser_states",
    )
    (
        point_values,
        online_coordinate_metadata,
        online_coordinate_handoff,
    ) = _load_canonical_target_surface(
        root,
        stim_group,
        chaser_group,
    )
    dtype_names = chaser_states.dtype.names or ()
    camera_frames = np.asarray(online_coordinate_handoff.camera_frame_ids)
    source_mapping_rows = np.asarray(online_coordinate_handoff.source_row_indices)
    if (
        camera_frames.shape != (chaser_states.shape[0],)
        or source_mapping_rows.shape != (chaser_states.shape[0],)
    ):
        raise ChaserMetricsCoordinateContractError(
            "Bound camera mapping length disagrees with chaser_states."
        )

    if isinstance(chaser_index, bool) or not isinstance(chaser_index, int) or chaser_index < 0:
        raise ChaserMetricsCoordinateContractError(
            "chaser_index must be a nonnegative integer."
        )
    if "chaser_index" in dtype_names:
        mask = chaser_states["chaser_index"] == chaser_index
    else:
        if chaser_index != 0:
            raise ChaserMetricsCoordinateContractError(
                "chaser_states has no chaser_index field, so only chaser_index=0 "
                "is addressable."
            )
        mask = np.ones(chaser_states.shape[0], dtype=bool)
    if not np.any(mask):
        raise ChaserMetricsCoordinateContractError(
            f"chaser_states contains no rows for chaser_index={chaser_index}."
        )

    filtered_states = chaser_states[mask]
    filtered_camera_frames = camera_frames[mask]
    source_row_indices = np.asarray(source_mapping_rows[mask], dtype=np.int64)
    if source_row_indices.shape[0] != point_values[mask].shape[0]:
        raise ChaserMetricsCoordinateContractError(
            "Selected target-position rows disagree with selected chaser-state rows."
        )

    missing_camera_rows = filtered_camera_frames < 0
    if np.any(missing_camera_rows):
        missing = source_row_indices[missing_camera_rows].tolist()
        raise ChaserMetricsCoordinateContractError(
            "Canonical stimulus-state rows lack an exact camera-frame mapping: "
            f"source rows {missing!r}."
        )
    unique_camera_frames, camera_frame_counts = np.unique(
        filtered_camera_frames,
        return_counts=True,
    )
    duplicate_camera_frames = unique_camera_frames[camera_frame_counts > 1]
    if duplicate_camera_frames.size:
        raise ChaserMetricsCoordinateContractError(
            "Selected stimulus-state rows map ambiguously to duplicate camera "
            f"frames {duplicate_camera_frames.tolist()!r}."
        )
    outside_bundle = sorted(
        int(frame)
        for frame in filtered_camera_frames
        if int(frame) not in frame_to_index
    )
    if outside_bundle:
        raise ChaserMetricsCoordinateContractError(
            "Canonical stimulus-state rows reference camera frames absent from "
            f"video metadata: {outside_bundle!r}."
        )

    trial_state_field = "trial_state" if "trial_state" in dtype_names else None

    numeric_fields = [
        "distance_to_target_px",
        "distance_to_target_mm",
        "visual_angle_deg",
        "tau_ms",
    ]

    present_fields = [field for field in numeric_fields if field in dtype_names]
    online_arrays = {
        field: np.full(frames_sorted.shape, np.nan, dtype=np.float64)
        for field in present_fields
    }
    target_positions = np.full((frames_sorted.shape[0], 2), np.nan, dtype=np.float64)
    bundle_row_source_indices = np.full(frames_sorted.shape, -1, dtype=np.int64)

    filtered_points = np.asarray(point_values[mask], dtype=np.float64)
    for record, point, camera_frame, source_row_index in zip(
        filtered_states,
        filtered_points,
        filtered_camera_frames,
        source_row_indices,
        strict=True,
    ):
        idx = frame_to_index[camera_frame]
        target_positions[idx] = point
        bundle_row_source_indices[idx] = source_row_index
        if trial_state_field is not None:
            try:
                trial_state[idx] = int(record[trial_state_field])
            except Exception:
                trial_state[idx] = int(record[trial_state_field].item())
        for field in present_fields:
            try:
                online_arrays[field][idx] = float(record[field])
            except Exception:
                online_arrays[field][idx] = float(np.asarray(record[field]))

    online_arrays[_TARGET_POSITION_ARRAY] = target_positions
    online_arrays[_TARGET_POSITION_COMPONENT_FIELDS[0]] = target_positions[:, 0]
    online_arrays[_TARGET_POSITION_COMPONENT_FIELDS[1]] = target_positions[:, 1]

    identity_mapping = online_coordinate_metadata["stimulus_state_key_mapping"]
    assert isinstance(identity_mapping, dict)
    stimulus_state_key = np.asarray(identity_mapping["values"])
    identity_mapping["selected_source_row_indices"] = source_row_indices.tolist()
    identity_mapping["selected_values"] = stimulus_state_key[mask].tolist()
    identity_mapping["selected_camera_frame_ids"] = filtered_camera_frames.tolist()
    identity_mapping["bundle_row_source_indices"] = bundle_row_source_indices.tolist()

    return online_arrays, online_coordinate_metadata, online_coordinate_handoff


def _load_canonical_target_surface(
    root: zarr.Group,
    stim_group: zarr.Group,
    chaser_group: zarr.Group,
    *,
    stimulus_evidence: BoundStimulusCoordinateEvidence | None = None,
) -> Tuple[
    np.ndarray,
    Dict[str, object],
    CanonicalOnlineCoordinateHandoff,
]:
    """Load one exact target-position surface and its bound row identity.

    Selection is by the controlled semantic role and canonical array name.  A
    descriptor on ``chaser_group`` is deliberately ignored: only attrs on the
    selected point array can establish its coordinate semantics.
    """

    try:
        evidence = load_bound_stimulus_coordinate_evidence(
            stim_group,
            chaser_group,
            root_node=root,
        )
    except StimulusCoordinateContractError as exc:
        raise ChaserMetricsCoordinateContractError(
            f"Canonical stimulus coordinate evidence is invalid: {exc}"
        ) from exc
    if stimulus_evidence is not None:
        provided_records = (
            stimulus_evidence.surface_manifest,
            stimulus_evidence.camera_mapping,
            stimulus_evidence.frame_transform.manifest,
            stimulus_evidence.import_lineage,
            stimulus_evidence.output_manifest,
        )
        current_records = (
            evidence.surface_manifest,
            evidence.camera_mapping,
            evidence.frame_transform.manifest,
            evidence.import_lineage,
            evidence.output_manifest,
        )
        if (
            stimulus_evidence.archive_identity != evidence.archive_identity
            or any(
                provided.record_ref != current.record_ref
                or provided.record_sha256 != current.record_sha256
                for provided, current in zip(
                    provided_records,
                    current_records,
                    strict=True,
                )
            )
        ):
            raise ChaserMetricsCoordinateContractError(
                "Caller-provided stimulus evidence belongs to a different archive/store."
            )
    bound_identity = evidence.row_identity
    contract = bound_identity.contract
    if (
        contract.domain != STIMULUS_STATE_DOMAIN
        or contract.mode != STIMULUS_STATE_KEY_MODE
        or contract.key_array.ref != STIMULUS_STATE_KEY_ARRAY_REF
    ):
        raise ChaserMetricsCoordinateContractError(
            "chaser_states requires the canonical stimulus_state/stimulus_state_key "
            "identity profile."
        )

    row_count = contract.leading_dimension
    key_node = chaser_group[STIMULUS_STATE_KEY_ARRAY_REF]
    key_values = np.asarray(evidence.stimulus_state_key)

    manifest_surfaces = evidence.surface_manifest.record.get("surfaces")
    role_candidates = [
        (str(item.get("array_name")), chaser_group.get(str(item.get("array_name"))))
        for item in manifest_surfaces
        if isinstance(item, Mapping)
        and item.get("semantic_role") == _TARGET_POSITION_ROLE
    ] if isinstance(manifest_surfaces, list) else []
    if len(role_candidates) != 1:
        found = [name for name, _ in role_candidates]
        raise ChaserMetricsCoordinateContractError(
            "Canonical chaser_states must contain exactly one point surface with "
            f"semantic_role={_TARGET_POSITION_ROLE!r}; found {found!r}."
        )
    surface_name, point_node = role_candidates[0]
    if surface_name != _TARGET_POSITION_ARRAY:
        raise ChaserMetricsCoordinateContractError(
            f"semantic_role={_TARGET_POSITION_ROLE!r} must use exact array name "
            f"{_TARGET_POSITION_ARRAY!r}, found {surface_name!r}."
        )
    if not isinstance(point_node, zarr.Array):
        raise ChaserMetricsCoordinateContractError(
            f"{_TARGET_POSITION_ARRAY} must be an array, not a group."
        )
    if np.dtype(point_node.dtype).kind not in "fiu":
        raise ChaserMetricsCoordinateContractError(
            f"{_TARGET_POSITION_ARRAY} must have a numeric dtype."
        )

    point_attrs = dict(point_node.attrs)
    if point_attrs.get("source_component_fields") != list(
        _TARGET_POSITION_COMPONENT_FIELDS
    ):
        raise ChaserMetricsCoordinateContractError(
            f"{_TARGET_POSITION_ARRAY} must bind exact component fields "
            f"{list(_TARGET_POSITION_COMPONENT_FIELDS)!r}."
        )
    try:
        bound_descriptor = load_bound_canonical_coordinate_descriptor(
            point_node,
            row_identity=bound_identity,
            reference_frame_authority=(
                evidence.frame_transform.arena_relative_frame
            ),
            transform_chain=evidence.frame_transform.transform_chain,
            lineage_records=(
                evidence.surface_manifest,
                evidence.camera_mapping,
                evidence.frame_transform.manifest,
                evidence.import_lineage,
                evidence.output_manifest,
            ),
        )
    except Exception as exc:
        raise ChaserMetricsCoordinateContractError(
            f"Canonical target-position descriptor is invalid: {exc}"
        ) from exc
    descriptor = bound_descriptor.descriptor
    _validate_target_descriptor(descriptor)

    point_values = np.asarray(point_node[:])
    if point_values.shape != (row_count, 2):
        raise ChaserMetricsCoordinateContractError(
            f"{_TARGET_POSITION_ARRAY} must have exact shape ({row_count}, 2)."
        )

    record_bindings = {
        "surface_manifest": _serializable_record_binding(evidence.surface_manifest),
        "camera_mapping": _serializable_record_binding(evidence.camera_mapping),
        "frame_transform": _serializable_record_binding(
            evidence.frame_transform.manifest
        ),
        "import_lineage": _serializable_record_binding(evidence.import_lineage),
        "output_manifest": _serializable_record_binding(evidence.output_manifest),
    }
    metadata: Dict[str, object] = {
        "schema_id": CANONICAL_ONLINE_COORDINATE_HANDOFF_SCHEMA_ID,
        "schema_version": CANONICAL_ONLINE_COORDINATE_HANDOFF_SCHEMA_VERSION,
        "semantic_role": _TARGET_POSITION_ROLE,
        "source_path": point_node.path,
        "coordinate_descriptor": descriptor.to_dict(),
        "coordinate_descriptor_sha256": descriptor.digest(),
        "coordinate_descriptor_binding": {
            "array_path": point_node.path,
            "attr_name": COORDINATE_DESCRIPTOR_ATTR,
            "digest_attr_name": (
                f"{COORDINATE_DESCRIPTOR_ATTR}{COORDINATE_DESCRIPTOR_DIGEST_SUFFIX}"
            ),
        },
        "rowset_path": chaser_group.path,
        "row_identity_contract": contract.to_dict(),
        "row_identity_contract_sha256": bound_identity.record_sha256,
        "row_identity_record_ref": bound_identity.record_ref,
        "row_identity_record_sha256": bound_identity.record_sha256,
        "row_identity_binding": {
            "rowset_path": bound_identity.rowset_path,
            "key_array_path": bound_identity.key_array_path,
            "contract_attr_name": ROW_IDENTITY_CONTRACT_ATTR,
            "contract_digest_attr_name": ROW_IDENTITY_CONTRACT_DIGEST_ATTR,
        },
        "arena_reference": {
            "record_ref": evidence.arena_reference.record_ref,
            "record_sha256": evidence.arena_reference.record_sha256,
            "selector": evidence.arena_reference.selector,
            "width": evidence.arena_reference.width,
            "height": evidence.arena_reference.height,
            "units": evidence.arena_reference.units,
            "record_attr_name": ARENA_GEOMETRY_RECORD_ATTR,
            "record_digest_attr_name": ARENA_GEOMETRY_RECORD_DIGEST_ATTR,
            "width_field": "arena_region_width_px",
            "height_field": "arena_region_height_px",
        },
        "record_bindings": record_bindings,
        "archive_identity": {
            "kind": evidence.archive_identity.kind,
            "key": list(evidence.archive_identity.key),
        },
        "stimulus_state_key_mapping": {
            "source_path": key_node.path,
            "components": list(contract.key_array.components),
            "content_sha256": contract.key_array.content_sha256,
            "values": key_values.tolist(),
            "camera_frame_ids_path": chaser_group[CAMERA_FRAME_IDS_ARRAY].path,
            "camera_frame_ids_sha256": evidence.camera_mapping.record[
                "camera_frame_ids_sha256"
            ],
            "source_row_indices_path": chaser_group[SOURCE_ROW_INDICES_ARRAY].path,
            "source_row_indices_sha256": evidence.camera_mapping.record[
                "source_row_indices_sha256"
            ],
            "source_acquisition_frame_index_path": chaser_group[
                SOURCE_ACQUISITION_FRAME_INDEX_ARRAY
            ].path,
            "source_acquisition_frame_index_sha256": (
                evidence.camera_mapping.record[
                    "source_acquisition_frame_index_sha256"
                ]
            ),
            "source_row_temporal_authority_ref": (
                evidence.source_temporal_authority.record_ref
            ),
            "source_row_temporal_authority_sha256": (
                evidence.source_temporal_authority.record_sha256
            ),
        },
    }
    handoff_arrays = (
        key_values.copy(),
        np.asarray(evidence.camera_frame_ids).copy(),
        np.asarray(evidence.source_row_indices).copy(),
        np.asarray(evidence.source_acquisition_frame_index).copy(),
    )
    for values in handoff_arrays:
        values.setflags(write=False)
    handoff = CanonicalOnlineCoordinateHandoff(
        coordinate_descriptor=bound_descriptor,
        row_identity=bound_identity,
        arena_reference=evidence.arena_reference,
        frame_transform=evidence.frame_transform,
        source_temporal_authority=evidence.source_temporal_authority,
        surface_manifest=evidence.surface_manifest,
        import_lineage=evidence.import_lineage,
        output_manifest=evidence.output_manifest,
        camera_mapping=evidence.camera_mapping,
        archive_identity=evidence.archive_identity,
        stimulus_state_key=handoff_arrays[0],
        camera_frame_ids=handoff_arrays[1],
        source_row_indices=handoff_arrays[2],
        source_acquisition_frame_index=handoff_arrays[3],
        _root_group=root,
        _run_group=stim_group,
        _chaser_group=chaser_group,
        _coordinate_node=point_node,
    )
    return point_values, metadata, handoff


@proof_verification_operation
def load_canonical_online_coordinate_surface(
    root: zarr.Group,
    stimulus_run_group: zarr.Group,
    chaser_rowset: zarr.Group,
    *,
    stimulus_evidence: BoundStimulusCoordinateEvidence | None = None,
) -> Tuple[
    np.ndarray,
    Dict[str, object],
    CanonicalOnlineCoordinateHandoff,
]:
    """Public canonical-only reader for one exact stimulus target surface.

    Unlike :func:`load_chaser_metrics`, this boundary does not align or order
    rows through external camera-frame identifiers.  Consumers that operate in
    acquisition time can select ``stimulus_state_key`` rows and order them via
    the sealed ``source_acquisition_frame_index`` evidence in the returned
    handoff.
    """

    return _load_canonical_target_surface(
        root,
        stimulus_run_group,
        chaser_rowset,
        stimulus_evidence=stimulus_evidence,
    )


def _serializable_record_binding(record: BoundCoordinateRecord) -> Dict[str, str]:
    return {
        "record_ref": record.record_ref,
        "record_sha256": record.record_sha256,
        "attr_name": record.attr_name,
        "digest_attr_name": record.digest_attr_name,
    }


def _validate_target_descriptor(
    descriptor: CanonicalCoordinateDescriptor,
) -> None:
    if descriptor.profile_id != "arena_relative_canvas_px.top_left_y_down.v1":
        raise ChaserMetricsCoordinateContractError(
            "target_position_xy requires the canonical arena-relative canvas profile."
        )
    if descriptor.geometry_type != "point_xy":
        raise ChaserMetricsCoordinateContractError(
            "target_position_xy requires canonical geometry_type='point_xy'."
        )
    if descriptor.components != ("x", "y"):
        raise ChaserMetricsCoordinateContractError(
            "target_position_xy requires ordered x,y components."
        )
    if descriptor.component_units != ("px", "px"):
        raise ChaserMetricsCoordinateContractError(
            "target_position_xy requires an explicit pixel-space profile with px,px units."
        )
    if (
        descriptor.source_camera_overlay.status != "requires_transform"
        or not descriptor.source_camera_overlay.transform_refs
    ):
        raise ChaserMetricsCoordinateContractError(
            "target_position_xy must carry an exact transform chain to source-camera overlay coordinates."
        )


def _extract_offline_fields(
    *,
    metrics_group: zarr.Group,
    frames_sorted: np.ndarray,
    frame_to_index: Dict[int, int],
) -> Dict[str, np.ndarray]:
    frame_indices = np.asarray(metrics_group["frame_indices"], dtype=np.int64)
    valid_mask = np.asarray(metrics_group["valid_mask"], dtype=bool)

    first_valid_index: Dict[int, int] = {}
    for idx, (frame, is_valid) in enumerate(zip(frame_indices, valid_mask)):
        if not is_valid or frame < 0:
            continue
        if frame not in first_valid_index:
            first_valid_index[frame] = idx

    offline_scalars = {
        "distance_px": np.asarray(metrics_group["distance_px"], dtype=np.float64),
        "distance_mm": np.asarray(metrics_group["distance_mm"], dtype=np.float64),
        "angle_unsigned_deg": np.asarray(metrics_group["angle_unsigned_deg"], dtype=np.float64),
        "angle_signed_deg": np.asarray(metrics_group["angle_signed_deg"], dtype=np.float64),
    }
    if "heading_deg" in metrics_group:
        offline_scalars["heading_deg"] = np.asarray(metrics_group["heading_deg"], dtype=np.float64)
    offline_vectors = {
        "fish_centroid_px": np.asarray(metrics_group["fish_centroid_px"], dtype=np.float64),
        "chaser_position_px": np.asarray(metrics_group["chaser_position_px"], dtype=np.float64),
    }

    n_frames = frames_sorted.shape[0]
    output_scalars = {
        name: np.full(n_frames, np.nan, dtype=np.float64) for name in offline_scalars
    }
    output_vectors = {
        name: np.full((n_frames, 2), np.nan, dtype=np.float64) for name in offline_vectors
    }
    has_offline = np.zeros(n_frames, dtype=bool)

    for frame, roi_idx in first_valid_index.items():
        if frame not in frame_to_index:
            continue
        idx = frame_to_index[frame]
        has_offline[idx] = True
        for name, data in offline_scalars.items():
            output_scalars[name][idx] = float(data[roi_idx])
        for name, data in offline_vectors.items():
            output_vectors[name][idx] = data[roi_idx]

    offline_fields: Dict[str, np.ndarray] = {
        **output_scalars,
        **output_vectors,
        "has_offline": has_offline,
    }
    return offline_fields


def _empty_offline_fields(length: int) -> Dict[str, np.ndarray]:
    offline_fields = {
        "distance_px": np.full(length, np.nan, dtype=np.float64),
        "distance_mm": np.full(length, np.nan, dtype=np.float64),
        "angle_unsigned_deg": np.full(length, np.nan, dtype=np.float64),
        "angle_signed_deg": np.full(length, np.nan, dtype=np.float64),
        "fish_centroid_px": np.full((length, 2), np.nan, dtype=np.float64),
        "chaser_position_px": np.full((length, 2), np.nan, dtype=np.float64),
        "has_offline": np.zeros(length, dtype=bool),
    }
    return offline_fields
