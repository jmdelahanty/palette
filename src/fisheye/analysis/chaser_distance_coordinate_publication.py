"""Canonical coordinate evidence and publication for chaser-distance runs.

The normal path in this module is deliberately future-only.  It consumes one
complete canonical detection rowset and one complete canonical stimulus run,
then binds every spatial output to their exact row, frame, calibration, arena,
and collection authorities.  Historical dimension inference, resolution
ratios, unnamed homographies, and median scale fallbacks are not accepted.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from fisheye.shared.archive_identity import ArchiveIdentity, archive_identity
from fisheye.shared.array_measurement_descriptor import (
    ARRAY_MEASUREMENT_DESCRIPTOR_ATTR as MEASUREMENT_DESCRIPTOR_ATTR,
    ARRAY_MEASUREMENT_DESCRIPTOR_SCHEMA_ID,
    build_array_measurement_descriptor,
    load_bound_array_measurement_descriptor,
    stamp_and_bind_array_measurement_descriptor,
)
from fisheye.shared.canonical_coordinate_publication import (
    BoundCanonicalCoordinateDescriptor,
    build_bound_canonical_coordinate_descriptor,
    load_bound_canonical_coordinate_descriptor,
    stamp_bound_canonical_coordinate_descriptors,
)
from fisheye.shared.coordinate_descriptor import (
    CANONICAL_OVERLAY_DIRECT,
    CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
    COORDINATE_DESCRIPTOR_ATTR,
    COORDINATE_DESCRIPTOR_DIGEST_SUFFIX,
    CanonicalCollectionAxis,
    DigestBoundCoordinateRecordRef,
)
from fisheye.shared.coordinate_frame_record import array_payload_sha256
from fisheye.shared.coordinate_identity import (
    STIMULUS_STATE_DOMAIN,
    BoundRowIdentityContract,
    build_row_identity_contract,
    load_bound_row_identity_contract,
    stamp_and_bind_row_identity_contract,
)
from fisheye.shared.coordinate_record import (
    BoundCoordinateRecord,
    bind_persisted_coordinate_record,
    stamp_and_bind_persisted_coordinate_record,
)
from fisheye.shared.coordinate_reference import canonical_node_path
from fisheye.shared.directed_transform_v2 import (
    apply_bound_directed_transform_v2,
    require_bound_directed_transform_v2,
)
from fisheye.shared.detection_tables import (
    resolve_detection_instance_table,
    resolve_detection_source_pixel_authority,
)
from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
)
from fisheye.shared.proof_verification import proof_verification_operation
from fisheye.shared.stimulus_coordinate_contract import (
    BoundStimulusCoordinateEvidence,
    load_bound_stimulus_coordinate_evidence,
)
from fisheye.shared.zarr.canonical_detection_manifest import (
    canonical_detection_dimensions_from_manifest,
    require_active_coordinate_canonical_detection,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_LATEST_COMPLETE_ATTR,
    RUN_STATUS_COMPLETE,
)


COORDINATE_CONTRACT = "palette.chaser_distance.coordinate_contract.v1"
COORDINATE_CONTRACT_ATTR = "coordinate_contract"
COORDINATE_CONTRACT_EPOCH_ATTR = "coordinate_contract_epoch"
COORDINATE_CONTRACT_EPOCH = 1

INPUT_AUTHORITY_ATTR = "chaser_distance_input_authority"
INPUT_AUTHORITY_SCHEMA_ID = "palette.chaser_distance_input_authority"
DERIVATION_ATTR = "chaser_distance_coordinate_derivation"
DERIVATION_SCHEMA_ID = "palette.chaser_distance_coordinate_derivation"
MEASUREMENT_AUTHORITY_ATTR = "chaser_distance_measurement_authority"
MEASUREMENT_AUTHORITY_SCHEMA_ID = "palette.chaser_distance_measurement_authority"
EPOCH_WINDOW_AUTHORITY_ATTR = "epoch_window_identity_authority"
EPOCH_WINDOW_AUTHORITY_SCHEMA_ID = "palette.epoch_window_identity_authority"
CHASER_COLLECTION_AUTHORITY_ATTR = "chaser_collection_authority"
CHASER_COLLECTION_AUTHORITY_SCHEMA_ID = "palette.chaser_collection_authority"
SURFACE_MANIFEST_ATTR = "chaser_distance_surface_manifest"
SURFACE_MANIFEST_SCHEMA_ID = "palette.chaser_distance_surface_manifest"
PUBLICATION_SEAL_ATTR = "chaser_distance_publication_seal"
PUBLICATION_SEAL_SCHEMA_ID = "palette.chaser_distance_publication_seal"
CHASER_DISTANCE_PARENT_PUBLICATION_LEASE_ATTR = (
    "chaser_distance_publication_lease"
)
CHASER_DISTANCE_PUBLICATION_GENERATION_ATTR = "publication_generation"
CHASER_DISTANCE_PUBLICATION_POLICY_ATTR = "publication_policy"
CHASER_DISTANCE_PUBLICATION_POLICY = (
    "owner_generation_guarded_selectors_then_eligibility_v1"
)
SCHEMA_VERSION = 1
MEASUREMENT_DESCRIPTOR_SCHEMA_ID = ARRAY_MEASUREMENT_DESCRIPTOR_SCHEMA_ID

CAMERA_FRAME_KEY_ARRAY = "stimulus_state_key"
SOURCE_DETECTION_ROW_ARRAY = "source_detection_row_index"
CHASER_POSITION_ARRAY = "chaser_position_xy"

_BOUND_SOURCE_CONTEXT_SEAL = object()
_BOUND_RUN_SEAL = object()


class ChaserDistanceCoordinateError(ValueError):
    """Raised when canonical chaser-distance coordinate evidence is unsafe."""


def _fail(message: str) -> None:
    raise ChaserDistanceCoordinateError(message)


def _write_activation_attr(attrs: Any, key: str, value: Any) -> None:
    """One activation write, kept injectable for hostile store-fault tests."""

    attrs[key] = value


def _restore_attr(attrs: Any, key: str, *, existed: bool, value: Any) -> None:
    if existed:
        attrs[key] = copy.deepcopy(value)
        return
    if key in attrs:
        del attrs[key]


_ACTIVATION_PARENT_ATTRS = (
    RUN_LATEST_COMPLETE_ATTR,
    "latest",
    "latest_pending",
    "authoritative_run",
    "authoritative_run_provenance",
    CHASER_DISTANCE_PUBLICATION_POLICY_ATTR,
    CHASER_DISTANCE_PUBLICATION_GENERATION_ATTR,
    CHASER_DISTANCE_PARENT_PUBLICATION_LEASE_ATTR,
)


def _activation_snapshot(attrs: Any) -> dict[str, tuple[bool, Any]]:
    return {
        name: (name in attrs, copy.deepcopy(attrs.get(name)))
        for name in _ACTIVATION_PARENT_ATTRS
    }


def _activation_generation(
    snapshot: Mapping[str, tuple[bool, Any]],
) -> int:
    present, value = snapshot[CHASER_DISTANCE_PUBLICATION_GENERATION_ATTR]
    if not present:
        return 0
    if type(value) is not int or value < 0:
        _fail("Chaser-distance publication generation must be nonnegative integer.")
    return value


def _fresh_activation_parent(root_node: Any, expected_archive: ArchiveIdentity) -> Any:
    try:
        parent = root_node["analysis/chaser_distance_runs"]
    except Exception as exc:
        _fail(f"Chaser-distance activation parent disappeared: {exc}.")
    if (
        canonical_node_path(parent) != "analysis/chaser_distance_runs"
        or archive_identity(parent) != expected_archive
    ):
        _fail("Chaser-distance activation parent changed archives or paths.")
    return parent


def _require_activation_parent_state(
    parent: Any,
    snapshot: Mapping[str, tuple[bool, Any]],
    *,
    overrides: Mapping[str, tuple[bool, Any]],
) -> None:
    attrs = parent.attrs
    for name in _ACTIVATION_PARENT_ATTRS:
        present, value = overrides.get(name, snapshot[name])
        if (name in attrs) is not present or (
            present and attrs.get(name) != value
        ):
            _fail(
                "Chaser-distance activation observed concurrent parent mutation "
                f"for {name!r}."
            )


def _activation_lease_record(
    *,
    owner_uuid: str,
    run_path: str,
    base_generation: int,
) -> dict[str, Any]:
    return {
        "schema_id": "palette.chaser_distance_publication_lease",
        "schema_version": 1,
        "policy": CHASER_DISTANCE_PUBLICATION_POLICY,
        "owner_uuid": owner_uuid,
        "publication_owner": owner_uuid,
        "run_path": run_path,
        "base_generation": base_generation,
        "next_generation": base_generation + 1,
    }


def _owned_activation_epoch(
    parent: Any,
    lease: Mapping[str, Any],
    *,
    base_generation: int,
    next_generation: int,
) -> bool:
    attrs = parent.attrs
    generation = attrs.get(CHASER_DISTANCE_PUBLICATION_GENERATION_ATTR, 0)
    return (
        attrs.get(CHASER_DISTANCE_PARENT_PUBLICATION_LEASE_ATTR) == dict(lease)
        and type(generation) is int
        and generation in {base_generation, next_generation}
    )


def _rollback_owned_activation(
    root_node: Any,
    *,
    expected_archive: ArchiveIdentity,
    expected_path: str,
    expected_seal: Mapping[str, Any],
    lease: Mapping[str, Any],
    base_generation: int,
    next_generation: int,
    mutations: Sequence[tuple[str, tuple[bool, Any], Any]],
) -> None:
    """Restore only values still owned by this lease and generation epoch."""

    failures: list[str] = []
    parent = _fresh_activation_parent(root_node, expected_archive)
    if not _owned_activation_epoch(
        parent,
        lease,
        base_generation=base_generation,
        next_generation=next_generation,
    ):
        return
    try:
        current_run = root_node[expected_path]
        if (
            canonical_node_path(current_run) == expected_path
            and archive_identity(current_run) == expected_archive
            and current_run.attrs.get(PUBLICATION_SEAL_ATTR) == dict(expected_seal)
            and current_run.attrs.get("stage_selector_eligible") is not False
        ):
            current_run.attrs["stage_selector_eligible"] = False
    except BaseException as exc:  # pragma: no cover - hostile store
        failures.append(f"stage_selector_eligible: {exc}")
    for name, previous, written in reversed(tuple(mutations)):
        try:
            parent = _fresh_activation_parent(root_node, expected_archive)
            if not _owned_activation_epoch(
                parent,
                lease,
                base_generation=base_generation,
                next_generation=next_generation,
            ):
                break
            if parent.attrs.get(name) != written:
                continue
            existed, value = previous
            _restore_attr(parent.attrs, name, existed=existed, value=value)
        except BaseException as exc:  # pragma: no cover - hostile store
            failures.append(f"{name}: {exc}")
    if failures:
        raise ChaserDistanceCoordinateError(
            "Chaser-distance owned activation rollback was incomplete: "
            f"{failures!r}."
        )


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        _fail(f"Coordinate publication record is not canonical JSON: {exc}.")


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_detection_array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def _array(node: Any, *, label: str) -> np.ndarray:
    try:
        values = np.array(node[:], copy=True, order="C")
        dtype = np.dtype(node.dtype)
        shape = tuple(int(item) for item in node.shape)
    except Exception as exc:
        _fail(f"Unable to read exact {label} array: {exc}.")
    if values.dtype != dtype or values.shape != shape or values.dtype.hasobject:
        _fail(f"{label} values disagree with their declared dtype or shape.")
    return values


def _payload(node: Any) -> dict[str, Any]:
    values = _array(node, label=f"/{canonical_node_path(node)}")
    return {
        "array_ref": f"/{canonical_node_path(node)}",
        "dtype": values.dtype.str,
        "shape": [int(item) for item in values.shape],
        "content_sha256": array_payload_sha256(node),
    }


def _record_pointer(value: BoundCoordinateRecord) -> dict[str, str]:
    return {
        "record_ref": value.record_ref,
        "record_sha256": value.record_sha256,
    }


def _frame_pointer(value: Any) -> dict[str, str]:
    return {
        "record_ref": value.record_ref,
        "record_sha256": value.record_sha256,
    }


def _controlled_detection_path(value: str) -> str:
    path = str(value).strip().strip("/")
    parts = path.split("/")
    if (
        len(parts) != 2
        or parts[0] != "detect_runs"
        or not parts[1]
        or parts[1] in {".", ".."}
        or "/".join(parts) != path
    ):
        _fail(
            "Canonical chaser distance requires one exact detect_runs/<run> "
            "rowset path; refined or inferred paths are unsupported until their "
            "producer publishes the same strict observation contract."
        )
    return path


def _controlled_stimulus_run(value: str) -> str:
    name = str(value).strip()
    if not name or name in {".", ".."} or "/" in name:
        _fail("Canonical stimulus_run must be one exact run-name component.")
    return name


def _positive_fps(root: Any, acquisition: Any) -> tuple[float, str]:
    metadata = acquisition.record.source_video_metadata
    candidates = (
        (metadata.get("fps"), "acquisition.source_video_metadata.fps"),
        (getattr(root, "attrs", {}).get("fps"), "root.attrs[fps]"),
        (getattr(root, "attrs", {}).get("video_fps"), "root.attrs[video_fps]"),
    )
    for raw, authority in candidates:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        value = float(raw)
        if np.isfinite(value) and value > 0:
            return value, authority
    _fail(
        "Canonical chaser distance requires an explicit positive acquisition "
        "frame rate; no default frame rate is allowed."
    )


def _identity_component(
    evidence: BoundStimulusCoordinateEvidence,
    component: str,
) -> np.ndarray | None:
    components = tuple(evidence.row_identity.contract.key_array.components)
    if component not in components:
        return None
    values = np.asarray(evidence.stimulus_state_key)
    index = components.index(component)
    if values.ndim == 1:
        if len(components) != 1 or index != 0:
            _fail("Stimulus identity component layout is inconsistent.")
        return np.asarray(values)
    if values.ndim == 2 and values.shape[1] == len(components):
        return np.asarray(values[:, index])
    _fail("Stimulus identity key has an unsupported physical layout.")


def _dense_aligned_scalar(
    source_values: np.ndarray | None,
    source_frames: np.ndarray,
    *,
    total_frames: int,
) -> np.ndarray:
    out = np.full(total_frames, -1, dtype=np.int64)
    if source_values is None:
        return out
    values = np.asarray(source_values)
    if values.shape != source_frames.shape or values.dtype.kind not in "iu":
        _fail("Stimulus scalar identity/time field is not exact integer row data.")
    for frame, value in zip(source_frames, values, strict=True):
        frame_i = int(frame)
        value_i = int(value)
        if frame_i < 0 or frame_i >= total_frames:
            _fail("Stimulus acquisition-frame mapping is outside the exact frame domain.")
        if out[frame_i] not in {-1, value_i}:
            _fail("Stimulus rows disagree on a scalar value for one acquisition frame.")
        out[frame_i] = value_i
    return out


@dataclass(frozen=True, init=False)
class BoundChaserDistanceSourceContext:
    """Freshly verified exact sources for one canonical distance computation."""

    root_node: Any = field(repr=False, compare=False)
    archive_identity: ArchiveIdentity
    recording_id: str
    detection_path: str
    stimulus_run: str
    stimulus_path: str
    detection_manifest_digest: str
    detection_row_identity: Mapping[str, str]
    detection_temporal_authority: Mapping[str, str]
    detection_source_camera_frame: Any = field(repr=False, compare=False)
    stimulus: BoundStimulusCoordinateEvidence = field(repr=False, compare=False)
    detection_centers: np.ndarray = field(repr=False, compare=False)
    detection_frames: np.ndarray = field(repr=False, compare=False)
    detection_scores: np.ndarray = field(repr=False, compare=False)
    chaser_source_xy: np.ndarray = field(repr=False, compare=False)
    chaser_source_frames: np.ndarray = field(repr=False, compare=False)
    chaser_source_indices: np.ndarray = field(repr=False, compare=False)
    dense_stimulus_frame_num: np.ndarray = field(repr=False, compare=False)
    dense_timestamp_ns: np.ndarray = field(repr=False, compare=False)
    total_frames: int
    fps: float
    fps_authority: str
    pixels_per_mm_projector: float
    detection_centers_payload: Mapping[str, Any]
    detection_frames_payload: Mapping[str, Any]
    detection_scores_payload: Mapping[str, Any]
    chaser_source_payload: Mapping[str, Any]
    signature_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        root_node: Any,
        archive: ArchiveIdentity,
        recording_id: str,
        detection_path: str,
        stimulus_run: str,
        detection_manifest_digest: str,
        detection_row_identity: Mapping[str, str],
        detection_temporal_authority: Mapping[str, str],
        detection_source_camera_frame: Any,
        stimulus: BoundStimulusCoordinateEvidence,
        detection_centers: np.ndarray,
        detection_frames: np.ndarray,
        detection_scores: np.ndarray,
        chaser_source_xy: np.ndarray,
        chaser_source_frames: np.ndarray,
        chaser_source_indices: np.ndarray,
        dense_stimulus_frame_num: np.ndarray,
        dense_timestamp_ns: np.ndarray,
        total_frames: int,
        fps: float,
        fps_authority: str,
        pixels_per_mm_projector: float,
        detection_centers_payload: Mapping[str, Any],
        detection_frames_payload: Mapping[str, Any],
        detection_scores_payload: Mapping[str, Any],
        chaser_source_payload: Mapping[str, Any],
        signature_sha256: str,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BOUND_SOURCE_CONTEXT_SEAL:
            _fail("Bound chaser-distance source context cannot be constructed directly.")
        for name, value in {
            "detection_centers": detection_centers,
            "detection_frames": detection_frames,
            "detection_scores": detection_scores,
            "chaser_source_xy": chaser_source_xy,
            "chaser_source_frames": chaser_source_frames,
            "chaser_source_indices": chaser_source_indices,
            "dense_stimulus_frame_num": dense_stimulus_frame_num,
            "dense_timestamp_ns": dense_timestamp_ns,
        }.items():
            array = np.array(value, copy=True, order="C")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "root_node", root_node)
        object.__setattr__(self, "archive_identity", archive)
        object.__setattr__(self, "recording_id", recording_id)
        object.__setattr__(self, "detection_path", detection_path)
        object.__setattr__(self, "stimulus_run", stimulus_run)
        object.__setattr__(self, "stimulus_path", f"analysis/stimulus_runs/{stimulus_run}")
        object.__setattr__(self, "detection_manifest_digest", detection_manifest_digest)
        object.__setattr__(self, "detection_row_identity", dict(detection_row_identity))
        object.__setattr__(
            self,
            "detection_temporal_authority",
            dict(detection_temporal_authority),
        )
        object.__setattr__(
            self,
            "detection_source_camera_frame",
            detection_source_camera_frame,
        )
        object.__setattr__(self, "stimulus", stimulus)
        object.__setattr__(self, "total_frames", total_frames)
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "fps_authority", fps_authority)
        object.__setattr__(self, "pixels_per_mm_projector", pixels_per_mm_projector)
        object.__setattr__(self, "detection_centers_payload", dict(detection_centers_payload))
        object.__setattr__(self, "detection_frames_payload", dict(detection_frames_payload))
        object.__setattr__(self, "detection_scores_payload", dict(detection_scores_payload))
        object.__setattr__(self, "chaser_source_payload", dict(chaser_source_payload))
        object.__setattr__(self, "signature_sha256", signature_sha256)
        object.__setattr__(self, "_seal", _verification_seal)

    def assert_verified(self) -> None:
        try:
            current = load_chaser_distance_source_context(
                self.root_node,
                detection_path=self.detection_path,
                stimulus_run=self.stimulus_run,
            )
        except ChaserDistanceCoordinateError as exc:
            _fail(f"Chaser-distance source evidence changed after binding: {exc}")
        if (
            current.archive_identity != self.archive_identity
            or current.signature_sha256 != self.signature_sha256
        ):
            _fail("Chaser-distance source evidence changed after binding.")


def require_bound_chaser_distance_source_context(
    value: Any,
) -> BoundChaserDistanceSourceContext:
    value = _require_source_context_seal(value)
    value.assert_verified()
    return value


def _require_source_context_seal(
    value: Any,
) -> BoundChaserDistanceSourceContext:
    if (
        type(value) is not BoundChaserDistanceSourceContext
        or value._seal is not _BOUND_SOURCE_CONTEXT_SEAL
    ):
        _fail("A sealed chaser-distance source context is required.")
    return value


@proof_verification_operation
def load_chaser_distance_source_context(
    root_node: Any,
    *,
    detection_path: str,
    stimulus_run: str,
) -> BoundChaserDistanceSourceContext:
    """Load exact canonical detection, stimulus, transform, and scale evidence."""

    detection_path = _controlled_detection_path(detection_path)
    stimulus_run = _controlled_stimulus_run(stimulus_run)
    try:
        detection_manifest = require_active_coordinate_canonical_detection(
            root_node,
            group_path=detection_path,
        )
        detection_run = root_node[detection_path]
        detection_table = resolve_detection_instance_table(detection_run)
        dimensions = canonical_detection_dimensions_from_manifest(detection_manifest)
        _, acquisition = load_persisted_acquisition_camera_authority(root_node)
        stimulus_group = root_node[f"analysis/stimulus_runs/{stimulus_run}"]
        chaser_group = stimulus_group["tracking_data/chaser_states"]
        stimulus = load_bound_stimulus_coordinate_evidence(
            stimulus_group,
            chaser_group,
            root_node=root_node,
        )
    except ChaserDistanceCoordinateError:
        raise
    except Exception as exc:
        _fail(f"Canonical chaser-distance source preflight failed: {exc}.")

    stimulus_frame = stimulus.frame_transform.source_camera_frame
    payload = detection_manifest["payload"]
    source_evidence = payload["source_evidence"]
    source_pixel_authority = source_evidence.get("source_pixel_authority")
    if source_pixel_authority is None:
        source_pixel_authority = resolve_detection_source_pixel_authority(
            dict(detection_run.attrs)
        )
    if source_pixel_authority != _frame_pointer(stimulus_frame):
        _fail(
            "Detection and stimulus calibration do not bind the same exact "
            "source-camera frame."
        )
    acquisition_pointer = _frame_pointer(acquisition)
    recording_id = str(source_evidence["recording_identity"])
    if recording_id != acquisition.record.recording_id:
        _fail(
            "Detection canonical source evidence and the analysis archive do not "
            "bind the same recording identity."
        )
    if source_evidence["source_frame_authority"] != acquisition_pointer:
        _fail(
            "Detection canonical source evidence and the analysis archive do not "
            "bind the same exact acquisition-frame authority."
        )
    stimulus_acquisition = _frame_pointer(
        stimulus.source_temporal_authority.acquisition_frame
    )
    if acquisition_pointer != stimulus_acquisition:
        _fail("Detection and stimulus rows bind different acquisition-frame domains.")

    centers_node = detection_table["centers_img_xy"]
    frames_node = detection_table["source_acquisition_frame_index"]
    if "scores" not in detection_table or "confidence_scores" in detection_table:
        _fail(
            "Canonical chaser distance requires exactly detect_runs/<run>/scores; "
            "missing or competing score surfaces are unsupported."
        )
    scores_node = detection_table["scores"]
    centers = _array(centers_node, label="detection centers_img_xy")
    frames = _array(frames_node, label="detection source acquisition frames")
    scores = _array(scores_node, label="detection scores")
    if (
        centers.ndim != 2
        or centers.shape[1] != 2
        or centers.dtype.kind != "f"
        or frames.dtype != np.dtype("<i8")
        or frames.shape != (centers.shape[0],)
        or scores.dtype.kind != "f"
        or scores.shape != (centers.shape[0],)
        or not np.isfinite(centers).all()
        or not np.isfinite(scores).all()
    ):
        _fail("Canonical detection centers, frames, or scores have invalid exact layout.")
    manifest_arrays = payload["logical_content"]["document"]["arrays"]
    for path, values in (
        ("instances/centers_img_xy", centers),
        ("instances/source_acquisition_frame_index", frames),
        ("instances/scores", scores),
    ):
        if _canonical_detection_array_sha256(values) != manifest_arrays[path]["sha256"]:
            _fail(
                f"Canonical detection array {path!r} differs from its active manifest."
            )

    surface_matches = [
        item
        for item in stimulus.surface_manifest.record.get("surfaces", [])
        if item.get("semantic_role") == "chaser_position"
    ]
    if len(surface_matches) != 1 or surface_matches[0].get("array_name") != (
        CHASER_POSITION_ARRAY
    ):
        _fail("Stimulus manifest must identify exactly one chaser_position_xy surface.")
    chaser_node = chaser_group[CHASER_POSITION_ARRAY]
    chaser_xy = _array(chaser_node, label="stimulus chaser_position_xy")
    chaser_frames = np.array(
        stimulus.source_acquisition_frame_index,
        copy=True,
        order="C",
    )
    if (
        chaser_xy.ndim != 2
        or chaser_xy.shape[1] != 2
        or chaser_xy.dtype.kind != "f"
        or chaser_frames.dtype != np.dtype("<i8")
        or chaser_frames.shape != (chaser_xy.shape[0],)
    ):
        _fail("Canonical stimulus chaser positions or acquisition frames are invalid.")
    source_indices = _identity_component(stimulus, "chaser_index")
    if source_indices is None:
        if np.unique(chaser_frames).size != chaser_frames.size:
            _fail(
                "Stimulus rows with multiple chasers per acquisition frame "
                "must persist chaser_index in their exact row identity."
            )
        source_indices = np.zeros(chaser_xy.shape[0], dtype=np.int64)
    source_indices = np.asarray(source_indices)
    if (
        source_indices.dtype.kind not in "iu"
        or source_indices.shape != (chaser_xy.shape[0],)
        or (source_indices.size and int(np.min(source_indices)) < 0)
        or (source_indices.size and int(np.max(source_indices)) > np.iinfo(np.int16).max)
    ):
        _fail("Stimulus chaser_index identity is invalid or outside int16 range.")

    total_frames = int(acquisition.record.source_total_frames)
    if (
        dimensions.n_frames != total_frames
        or dimensions.n_instances != centers.shape[0]
    ):
        _fail(
            "Canonical detection dimensions differ from the acquisition or exact rowset."
        )
    if total_frames <= 0:
        _fail("Acquisition frame authority has no positive total frame count.")
    for values, label in (
        (frames, "detection"),
        (chaser_frames, "stimulus"),
    ):
        if values.size and (
            int(np.min(values)) < 0 or int(np.max(values)) >= total_frames
        ):
            _fail(f"{label} acquisition-frame mapping is outside the exact domain.")

    ppm = stimulus.frame_transform.selected_calibration.pixels_per_mm_projector
    if (
        isinstance(ppm, bool)
        or not isinstance(ppm, (int, float))
        or not np.isfinite(float(ppm))
        or float(ppm) <= 0
    ):
        _fail(
            "Selected stimulus calibration lacks an exact positive "
            "pixels_per_mm_projector; median and root fallbacks are forbidden."
        )
    fps, fps_authority = _positive_fps(root_node, acquisition)

    stimulus_frame_num = _identity_component(stimulus, "stimulus_frame_num")
    dense_stimulus = _dense_aligned_scalar(
        stimulus_frame_num,
        chaser_frames,
        total_frames=total_frames,
    )
    timestamp_values = None
    if "timestamp_ns" in chaser_group:
        timestamp_values = _array(chaser_group["timestamp_ns"], label="stimulus timestamp_ns")
    dense_timestamp = _dense_aligned_scalar(
        timestamp_values,
        chaser_frames,
        total_frames=total_frames,
    )

    signature = {
        "recording_id": recording_id,
        "detection_path": detection_path,
        "detection_manifest_digest": detection_manifest["payload_digest"],
        "detection_row_identity": {
            "record_ref": f"/{detection_path}@run_manifest.logical_content",
            "record_sha256": payload["logical_content"]["digest"],
        },
        "detection_temporal_authority": acquisition_pointer,
        "detection_centers": _payload(centers_node),
        "detection_frames": _payload(frames_node),
        "detection_scores": _payload(scores_node),
        "stimulus_run": stimulus_run,
        "stimulus_row_identity": {
            "record_ref": stimulus.row_identity.record_ref,
            "record_sha256": stimulus.row_identity.record_sha256,
        },
        "stimulus_temporal_authority": _record_pointer(
            stimulus.source_temporal_authority
        ),
        "stimulus_surface_manifest": _record_pointer(stimulus.surface_manifest),
        "stimulus_output_manifest": _record_pointer(stimulus.output_manifest),
        "stimulus_transform_manifest": _record_pointer(
            stimulus.frame_transform.manifest
        ),
        "selected_calibration_manifest": {
            "record_ref": (
                stimulus.frame_transform.selected_calibration.manifest_record_ref
            ),
            "record_sha256": stimulus.frame_transform.selected_calibration.manifest_sha256,
        },
        "chaser_position": _payload(chaser_node),
        "pixels_per_mm_projector": float(ppm),
        "fps": fps,
        "fps_authority": fps_authority,
        "total_frames": total_frames,
    }
    common_archive = archive_identity(root_node)
    return BoundChaserDistanceSourceContext(
        root_node=root_node,
        archive=common_archive,
        recording_id=recording_id,
        detection_path=detection_path,
        stimulus_run=stimulus_run,
        detection_manifest_digest=str(detection_manifest["payload_digest"]),
        detection_row_identity=signature["detection_row_identity"],
        detection_temporal_authority=acquisition_pointer,
        detection_source_camera_frame=stimulus_frame,
        stimulus=stimulus,
        detection_centers=centers,
        detection_frames=frames,
        detection_scores=scores,
        chaser_source_xy=chaser_xy,
        chaser_source_frames=chaser_frames,
        chaser_source_indices=np.asarray(source_indices, dtype=np.int64),
        dense_stimulus_frame_num=dense_stimulus,
        dense_timestamp_ns=dense_timestamp,
        total_frames=total_frames,
        fps=fps,
        fps_authority=fps_authority,
        pixels_per_mm_projector=float(ppm),
        detection_centers_payload=signature["detection_centers"],
        detection_frames_payload=signature["detection_frames"],
        detection_scores_payload=signature["detection_scores"],
        chaser_source_payload=signature["chaser_position"],
        signature_sha256=_mapping_sha256(signature),
        _verification_seal=_BOUND_SOURCE_CONTEXT_SEAL,
    )


def _apply_homography(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or transform.shape != (3, 3):
        _fail("Directed homography application requires (N,2) points and a 3x3 matrix.")
    if not np.isfinite(transform).all() or int(np.linalg.matrix_rank(transform)) != 3:
        _fail("Directed homography matrix is non-finite or singular.")
    homogeneous = np.column_stack((points, np.ones(points.shape[0], dtype=np.float64)))
    projected = (transform @ homogeneous.T).T
    denominator = projected[:, 2]
    if np.any(~np.isfinite(denominator)) or np.any(np.abs(denominator) <= 1e-12):
        _fail("Directed homography produced an invalid homogeneous denominator.")
    return projected[:, :2] / denominator[:, None]


def source_camera_to_arena_xy(
    points_xy: np.ndarray,
    *,
    context: BoundChaserDistanceSourceContext,
) -> np.ndarray:
    """Apply the exact inverse of the persisted arena-to-camera overlay chain."""

    context = _require_source_context_seal(context)
    transform = context.stimulus.frame_transform
    camera_to_canvas = transform.canvas_to_source_camera.inverse_of
    if camera_to_canvas is None:
        _fail("Stimulus transform evidence lacks the explicit camera-to-canvas forward link.")
    camera_to_canvas = require_bound_directed_transform_v2(camera_to_canvas)
    canvas = apply_bound_directed_transform_v2(points_xy, camera_to_canvas)
    arena_to_canvas = require_bound_directed_transform_v2(transform.arena_to_canvas)
    arena = _apply_homography(canvas, np.linalg.inv(arena_to_canvas.matrix))
    return np.asarray(arena, dtype=np.float64)


@dataclass(frozen=True)
class ChaserDistanceCoordinateArrays:
    camera_frame_index: np.ndarray
    stimulus_frame_num: np.ndarray
    timestamp_ns: np.ndarray
    source_detection_row_index: np.ndarray
    fish_centroid_img_xy: np.ndarray
    fish_centroid_arena_xy: np.ndarray
    chaser_indices: np.ndarray
    chaser_arena_xy: np.ndarray
    fish_valid: np.ndarray
    chaser_valid: np.ndarray
    distance_px: np.ndarray
    distance_mm: np.ndarray
    nearest_chaser_index: np.ndarray
    nearest_distance_mm: np.ndarray


@proof_verification_operation
def derive_chaser_distance_coordinate_arrays(
    context: BoundChaserDistanceSourceContext,
) -> ChaserDistanceCoordinateArrays:
    """Derive exact dense coordinate arrays from one sealed source context."""

    context = _require_source_context_seal(context)
    total = context.total_frames
    fish = np.full((total, 2), np.nan, dtype=np.float32)
    fish_valid = np.zeros(total, dtype=bool)
    selected_row = np.full(total, -1, dtype=np.int64)
    best_score = np.full(total, -np.inf, dtype=np.float64)
    for row, (frame, center, score) in enumerate(
        zip(
            context.detection_frames,
            context.detection_centers,
            context.detection_scores,
            strict=True,
        )
    ):
        frame_i = int(frame)
        # Preserve the historical >= tie policy explicitly: the final source
        # row wins when equal scores occur within one acquisition frame.
        if not fish_valid[frame_i] or float(score) >= best_score[frame_i]:
            fish[frame_i] = np.asarray(center, dtype=np.float32)
            fish_valid[frame_i] = True
            best_score[frame_i] = float(score)
            selected_row[frame_i] = int(row)

    indices = np.asarray(
        sorted(int(value) for value in np.unique(context.chaser_source_indices)),
        dtype=np.int16,
    )
    if indices.size == 0:
        _fail("Canonical stimulus source has no chaser identities.")
    columns = {int(value): index for index, value in enumerate(indices.tolist())}
    chaser = np.full((total, indices.size, 2), np.nan, dtype=np.float32)
    chaser_valid = np.zeros((total, indices.size), dtype=bool)
    for frame, source_index, point in zip(
        context.chaser_source_frames,
        context.chaser_source_indices,
        context.chaser_source_xy,
        strict=True,
    ):
        frame_i = int(frame)
        column = columns[int(source_index)]
        if chaser_valid[frame_i, column]:
            _fail(
                "Canonical stimulus source has duplicate rows for one "
                "acquisition-frame/chaser identity."
            )
        if np.isfinite(point).all():
            chaser[frame_i, column] = np.asarray(point, dtype=np.float32)
            chaser_valid[frame_i, column] = True

    fish_arena = np.full_like(fish, np.nan, dtype=np.float32)
    if np.any(fish_valid):
        fish_arena[fish_valid] = source_camera_to_arena_xy(
            fish[fish_valid].astype(np.float64),
            context=context,
        ).astype(np.float32)

    distance_px = np.full((total, indices.size), np.nan, dtype=np.float32)
    for column in range(indices.size):
        valid = fish_valid & chaser_valid[:, column] & np.isfinite(fish_arena).all(axis=1)
        delta = chaser[:, column, :] - fish_arena
        distance_px[valid, column] = np.linalg.norm(delta[valid], axis=1).astype(np.float32)
    distance_mm = (distance_px / np.float32(context.pixels_per_mm_projector)).astype(np.float32)
    nearest_index = np.full(total, -1, dtype=np.int16)
    nearest_mm = np.full(total, np.nan, dtype=np.float32)
    any_finite = np.isfinite(distance_mm).any(axis=1)
    if np.any(any_finite):
        filled = np.where(np.isfinite(distance_mm), distance_mm, np.inf)
        nearest_columns = np.argmin(filled[any_finite], axis=1)
        nearest_index[any_finite] = indices[nearest_columns]
        nearest_mm[any_finite] = filled[any_finite, nearest_columns].astype(np.float32)
    return ChaserDistanceCoordinateArrays(
        camera_frame_index=np.arange(total, dtype=np.int64),
        stimulus_frame_num=np.array(context.dense_stimulus_frame_num, copy=True),
        timestamp_ns=np.array(context.dense_timestamp_ns, copy=True),
        source_detection_row_index=selected_row,
        fish_centroid_img_xy=fish,
        fish_centroid_arena_xy=fish_arena,
        chaser_indices=indices,
        chaser_arena_xy=chaser,
        fish_valid=fish_valid,
        chaser_valid=chaser_valid,
        distance_px=distance_px,
        distance_mm=distance_mm,
        nearest_chaser_index=nearest_index,
        nearest_distance_mm=nearest_mm,
    )


def _equal_array(actual: Any, expected: np.ndarray, *, label: str) -> None:
    values = np.asarray(actual)
    if (
        values.dtype != expected.dtype
        or values.shape != expected.shape
        or not np.array_equal(values, expected, equal_nan=True)
    ):
        _fail(f"Persisted {label} differs from the exact canonical derivation.")


def _input_authority_record(
    context: BoundChaserDistanceSourceContext,
) -> dict[str, Any]:
    stimulus = context.stimulus
    transform = stimulus.frame_transform
    camera_to_canvas = transform.canvas_to_source_camera.inverse_of
    assert camera_to_canvas is not None
    return {
        "schema_id": INPUT_AUTHORITY_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "source_detection_path": context.detection_path,
        "source_detection_manifest_digest": context.detection_manifest_digest,
        "source_detection_row_identity": dict(context.detection_row_identity),
        "source_detection_temporal_authority": dict(
            context.detection_temporal_authority
        ),
        "source_detection_centers": dict(context.detection_centers_payload),
        "source_detection_frames": dict(context.detection_frames_payload),
        "source_detection_scores": dict(context.detection_scores_payload),
        "source_stimulus_path": context.stimulus_path,
        "source_stimulus_row_identity": {
            "record_ref": stimulus.row_identity.record_ref,
            "record_sha256": stimulus.row_identity.record_sha256,
        },
        "source_stimulus_temporal_authority": _record_pointer(
            stimulus.source_temporal_authority
        ),
        "source_chaser_positions": dict(context.chaser_source_payload),
        "source_stimulus_surface_manifest": _record_pointer(
            stimulus.surface_manifest
        ),
        "source_stimulus_output_manifest": _record_pointer(
            stimulus.output_manifest
        ),
        "source_stimulus_transform_manifest": _record_pointer(transform.manifest),
        "selected_calibration_manifest": {
            "record_ref": transform.selected_calibration.manifest_record_ref,
            "record_sha256": transform.selected_calibration.manifest_sha256,
        },
        "source_camera_frame": _frame_pointer(transform.source_camera_frame),
        "arena_relative_frame": _frame_pointer(transform.arena_relative_frame),
        "numeric_transform_direction": (
            "source_camera_image_px_to_selected_canvas_px_then_"
            "inverse_arena_to_selected_canvas_to_arena_relative_canvas_px"
        ),
        "camera_to_selected_canvas": {
            "record_ref": camera_to_canvas.record_ref,
            "record_sha256": camera_to_canvas.transform_sha256,
        },
        "arena_to_selected_canvas_forward_authority": {
            "record_ref": transform.arena_to_canvas.record_ref,
            "record_sha256": transform.arena_to_canvas.transform_sha256,
        },
        "arena_to_source_camera_overlay_chain": [
            {
                "record_ref": item.record_ref,
                "record_sha256": item.record_sha256,
            }
            for item in transform.transform_chain.transform_records
        ],
        "pixels_per_mm_projector": context.pixels_per_mm_projector,
        "pixels_per_mm_authority": (
            "selected_calibration_manifest.camera_calibration."
            "pixels_per_mm_projector"
        ),
        "total_frames": context.total_frames,
        "fps": context.fps,
        "fps_authority": context.fps_authority,
        "source_context_sha256": context.signature_sha256,
    }


def _derivation_record(run_group: Any, input_record: BoundCoordinateRecord) -> dict[str, Any]:
    positions = run_group["positions"]
    distances = run_group["distances"]
    return {
        "schema_id": DERIVATION_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "input_authority": _record_pointer(input_record),
        "operations": {
            "fish_row_selection": "max_score_then_last_source_row_v1",
            "fish_source_camera_to_arena": (
                "apply_explicit_camera_to_canvas_then_inverse_explicit_"
                "arena_to_canvas_v1"
            ),
            "distance_px": "euclidean_norm_chaser_minus_fish_in_shared_arena_frame_v1",
            "distance_mm": "distance_px_divided_by_selected_projector_pixels_per_mm_v1",
            "nearest_chaser": "finite_argmin_then_chaser_identity_v1",
        },
        "outputs": {
            "source_detection_row_index": _payload(
                positions[SOURCE_DETECTION_ROW_ARRAY]
            ),
            "fish_centroid_img_xy": _payload(positions["fish_centroid_img_xy"]),
            "fish_centroid_arena_xy": _payload(positions["fish_centroid_arena_xy"]),
            "chaser_arena_xy": _payload(positions["chaser_arena_xy"]),
            "fish_valid": _payload(positions["fish_valid"]),
            "chaser_valid": _payload(positions["chaser_valid"]),
            "distance_px": _payload(distances["distance_px"]),
            "distance_mm": _payload(distances["distance_mm"]),
            "nearest_chaser_index": _payload(distances["nearest_chaser_index"]),
            "nearest_distance_mm": _payload(distances["nearest_distance_mm"]),
        },
    }


def _measurement_record(
    context: BoundChaserDistanceSourceContext,
    input_record: BoundCoordinateRecord,
    derivation: BoundCoordinateRecord,
) -> dict[str, Any]:
    arena = context.stimulus.frame_transform.arena_relative_frame
    return {
        "schema_id": MEASUREMENT_AUTHORITY_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "quantity": "euclidean_distance",
        "basis_coordinate_space_id": "arena_relative_canvas_px",
        "basis_coordinate_frame": _frame_pointer(arena),
        "basis_origin": "arena_top_left",
        "basis_positive_x": "right",
        "basis_positive_y": "down",
        "basis_pixel_convention": "continuous",
        "reference_width_px": int(arena.endpoint.width),
        "reference_height_px": int(arena.endpoint.height),
        "pixel_units": "px",
        "physical_units": "mm",
        "conversion_direction": "arena_relative_canvas_px_distance_to_physical_mm",
        "conversion_formula": "distance_mm=distance_px/pixels_per_mm_projector",
        "pixels_per_mm_projector": context.pixels_per_mm_projector,
        "pixels_per_mm_authority": (
            "selected_calibration_manifest.camera_calibration."
            "pixels_per_mm_projector"
        ),
        "input_authority": _record_pointer(input_record),
        "derivation": _record_pointer(derivation),
        "source_camera_overlay_status": "not_suitable_scalar_measurement",
    }


def _collection_record(run_group: Any) -> dict[str, Any]:
    chasers = run_group["chasers"]
    indices = _array(chasers["chaser_index"], label="chaser_index")
    labels = _array(
        chasers["stimulus_instance_id_bytes"],
        label="stimulus_instance_id_bytes",
    )
    track_keys = _array(
        chasers["source_track_key_bytes"],
        label="source_track_key_bytes",
    )
    if (
        indices.ndim != 1
        or indices.dtype != np.dtype("<i2")
        or indices.size == 0
        or len(set(int(item) for item in indices)) != indices.size
        or not np.array_equal(indices, np.sort(indices))
        or labels.ndim != 2
        or labels.dtype != np.dtype("uint8")
        or labels.shape[0] != indices.size
        or track_keys.ndim != 2
        or track_keys.dtype != np.dtype("uint8")
        or track_keys.shape[0] != indices.size
    ):
        _fail("Chaser collection arrays have invalid dtype, shape, order, or identity.")

    def decoded(values: np.ndarray) -> tuple[str, ...]:
        result = tuple(
            bytes(row).split(b"\0", 1)[0].decode("utf-8", "strict")
            for row in values
        )
        if any(not value or value != value.strip() for value in result):
            _fail("Chaser collection labels must be non-empty canonical UTF-8 text.")
        return result

    decoded_labels = decoded(labels)
    decoded_tracks = decoded(track_keys)
    expected_labels = tuple(f"chaser:{int(index)}" for index in indices)
    expected_tracks = tuple(f"chaser_index:{int(index)}" for index in indices)
    if (
        decoded_labels != expected_labels
        or decoded_tracks != expected_tracks
        or len(set(decoded_labels)) != len(decoded_labels)
        or len(set(decoded_tracks)) != len(decoded_tracks)
    ):
        _fail(
            "Chaser collection labels must be the exact controlled identities "
            "derived from ordered chaser_index values."
        )
    return {
        "schema_id": CHASER_COLLECTION_AUTHORITY_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "axis": 1,
        "role": "chaser",
        "cardinality": int(indices.shape[0]),
        "ordered_label_array": _payload(chasers["stimulus_instance_id_bytes"]),
        "chaser_index": _payload(chasers["chaser_index"]),
        "source_track_key": _payload(chasers["source_track_key_bytes"]),
        "label_encoding": "null_terminated_utf8_rows_v1",
        "identity_rule": "ordered_unique_stimulus_instance_id_per_chaser_index_v1",
    }


def _collection_axis(record: BoundCoordinateRecord, cardinality: int) -> CanonicalCollectionAxis:
    return CanonicalCollectionAxis(
        axis=1,
        role="chaser",
        cardinality=cardinality,
        label_authority=DigestBoundCoordinateRecordRef(
            record_ref=record.record_ref,
            record_sha256=record.record_sha256,
        ),
    )


def _coordinate_bindings(
    context: BoundChaserDistanceSourceContext,
    run_group: Any,
    row_identity: BoundRowIdentityContract,
    input_record: BoundCoordinateRecord,
    derivation: BoundCoordinateRecord,
    collection_record: BoundCoordinateRecord,
    *,
    load: bool,
) -> dict[str, BoundCanonicalCoordinateDescriptor]:
    positions = run_group["positions"]
    collection = _collection_axis(
        collection_record,
        int(run_group["chasers/chaser_index"].shape[0]),
    )
    source_camera = context.detection_source_camera_frame
    frame_transform = context.stimulus.frame_transform
    specs: dict[str, dict[str, Any]] = {
        "positions/fish_centroid_img_xy": {
            "node": positions["fish_centroid_img_xy"],
            "profile_id": "source_camera_image_px.top_left_y_down.v1",
            "geometry_type": "point_xy",
            "components": ("x", "y"),
            "component_units": ("px", "px"),
            "pixel_convention": "continuous",
            "reference_frame_authority": source_camera,
            "source_camera_overlay_status": CANONICAL_OVERLAY_DIRECT,
            "lineage_records": (input_record, derivation),
        },
        "positions/fish_centroid_arena_xy": {
            "node": positions["fish_centroid_arena_xy"],
            "profile_id": "arena_relative_canvas_px.top_left_y_down.v1",
            "geometry_type": "point_xy",
            "components": ("x", "y"),
            "component_units": ("px", "px"),
            "pixel_convention": "continuous",
            "reference_frame_authority": frame_transform.arena_relative_frame,
            "source_camera_overlay_status": CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
            "transform_chain": frame_transform.transform_chain,
            "lineage_records": (input_record, derivation),
        },
        "positions/chaser_arena_xy": {
            "node": positions["chaser_arena_xy"],
            "profile_id": "arena_relative_canvas_px.top_left_y_down.v1",
            "geometry_type": "point_xy",
            "components": ("x", "y"),
            "component_units": ("px", "px"),
            "pixel_convention": "continuous",
            "reference_frame_authority": frame_transform.arena_relative_frame,
            "source_camera_overlay_status": CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
            "transform_chain": frame_transform.transform_chain,
            "lineage_records": (collection_record, input_record, derivation),
            "collection_axis": collection,
        },
    }
    result: dict[str, BoundCanonicalCoordinateDescriptor] = {}
    for path, spec in specs.items():
        node = spec.pop("node")
        if load:
            load_spec = {
                key: spec[key]
                for key in (
                    "reference_extent",
                    "reference_frame_authority",
                    "transform_chain",
                    "lineage_records",
                    "frame_record",
                )
                if key in spec
            }
            result[path] = load_bound_canonical_coordinate_descriptor(
                node,
                row_identity=row_identity,
                **load_spec,
            )
        else:
            result[path] = build_bound_canonical_coordinate_descriptor(
                node,
                row_identity=row_identity,
                **spec,
            )
    return result


def _coordinate_descriptor_pointer(
    value: BoundCanonicalCoordinateDescriptor,
) -> dict[str, str]:
    node = value.coordinate_node
    digest = node.attrs.get(
        f"{COORDINATE_DESCRIPTOR_ATTR}{COORDINATE_DESCRIPTOR_DIGEST_SUFFIX}"
    )
    if digest != value.descriptor.digest():
        _fail("Coordinate descriptor digest changed before measurement publication.")
    return {
        "record_ref": f"/{canonical_node_path(node)}@{COORDINATE_DESCRIPTOR_ATTR}",
        "record_sha256": str(digest),
    }


def _decode_text_rows(node: Any, *, label: str) -> tuple[str, ...]:
    values = _array(node, label=label)
    if values.ndim == 2 and values.dtype.kind in "ui":
        rows = (bytes(np.asarray(row, dtype=np.uint8)) for row in values)
    elif values.ndim == 1 and values.dtype.kind in "SV":
        rows = (bytes(value) for value in values)
    else:
        _fail(f"{label} uses an unsupported persisted text layout.")
    try:
        return tuple(row.split(b"\0", 1)[0].decode("utf-8", "strict") for row in rows)
    except UnicodeDecodeError as exc:
        _fail(f"{label} is not canonical UTF-8: {exc}.")


def _epoch_window_authority_record(
    context: BoundChaserDistanceSourceContext,
    run_group: Any,
) -> dict[str, Any]:
    summary = run_group["epoch_summary"]
    output_nodes = {
        name: summary[name]
        for name in (
            "window_id",
            "label_bytes",
            "start_frame",
            "end_frame",
        )
    }
    output_id = _array(output_nodes["window_id"], label="epoch window_id")
    output_start = _array(output_nodes["start_frame"], label="epoch start_frame")
    output_end = _array(output_nodes["end_frame"], label="epoch end_frame")
    output_labels = _decode_text_rows(
        output_nodes["label_bytes"],
        label="epoch label_bytes",
    )
    count = int(output_id.shape[0]) if output_id.ndim == 1 else -1
    if (
        output_id.dtype != np.dtype("<i4")
        or output_start.dtype != np.dtype("<i8")
        or output_end.dtype != np.dtype("<i8")
        or output_start.shape != (count,)
        or output_end.shape != (count,)
        or len(output_labels) != count
        or len(set(int(value) for value in output_id)) != count
    ):
        _fail("Persisted epoch-window identity arrays are invalid or non-unique.")

    expected_assignment = np.full(context.total_frames, -1, dtype=np.int32)
    for window_id, start, end in zip(
        output_id,
        output_start,
        output_end,
        strict=True,
    ):
        first = max(0, int(start))
        last = min(context.total_frames - 1, int(end))
        if last >= first:
            expected_assignment[first : last + 1] = int(window_id)
    assignment_node = run_group["frames/stimulus_epoch_window_id"]
    _equal_array(
        _array(assignment_node, label="stimulus_epoch_window_id"),
        expected_assignment,
        label="stimulus_epoch_window_id",
    )

    source_path = run_group.attrs.get("source_stimulus_epoch_path")
    source_payloads: dict[str, Any] = {}
    if source_path is None:
        if count != 0:
            _fail("Epoch windows have no exact source stimulus-epoch run.")
    else:
        path = str(source_path).strip().strip("/")
        parts = path.split("/")
        if (
            len(parts) != 3
            or parts[:2] != ["analysis", "stimulus_epoch_runs"]
            or not parts[2]
            or parts[2] in {".", ".."}
        ):
            _fail("source_stimulus_epoch_path is not one controlled run path.")
        try:
            source_windows = context.root_node[f"{path}/windows"]
            source_nodes = {
                name: source_windows[name]
                for name in (
                    "window_id",
                    "label_bytes",
                    "start_frame",
                    "end_frame",
                )
            }
        except Exception as exc:
            _fail(f"Exact source epoch-window arrays are missing: {exc}.")
        source_id = np.asarray(
            _array(source_nodes["window_id"], label="source epoch window_id"),
            dtype=np.int32,
        ).reshape(-1)
        source_start = np.asarray(
            _array(source_nodes["start_frame"], label="source epoch start_frame"),
            dtype=np.int64,
        ).reshape(-1)
        source_end = np.asarray(
            _array(source_nodes["end_frame"], label="source epoch end_frame"),
            dtype=np.int64,
        ).reshape(-1)
        source_labels = _decode_text_rows(
            source_nodes["label_bytes"],
            label="source epoch label_bytes",
        )
        if (
            not np.array_equal(output_id, source_id)
            or not np.array_equal(output_start, source_start)
            or not np.array_equal(output_end, source_end)
            or output_labels != source_labels
        ):
            _fail("Published epoch-window identity differs from its exact source run.")
        source_payloads = {
            name: _payload(node) for name, node in sorted(source_nodes.items())
        }
    return {
        "schema_id": EPOCH_WINDOW_AUTHORITY_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "identity_domain": "stimulus_epoch_window",
        "source_run_path": source_path,
        "source_arrays": source_payloads,
        "published_arrays": {
            name: _payload(node) for name, node in sorted(output_nodes.items())
        },
        "camera_frame_assignment": _payload(assignment_node),
        "mapping_operation": "inclusive_start_end_frame_last_window_wins_v1",
        "cardinality": count,
    }


def _measurement_descriptor_record(
    node: Any,
    *,
    quantity: str,
    units: str,
    operation: str,
    axes: Sequence[str],
    coordinate_inputs: Sequence[BoundCanonicalCoordinateDescriptor],
    measurement_inputs: Sequence[BoundCoordinateRecord],
    row_identity: BoundRowIdentityContract,
    collection: BoundCoordinateRecord,
    measurement_authority: BoundCoordinateRecord,
    derivation: BoundCoordinateRecord,
    epoch_windows: BoundCoordinateRecord | None = None,
) -> dict[str, Any]:
    axis_order = tuple(str(value) for value in axes)
    try:
        return build_array_measurement_descriptor(
            node,
            quantity=quantity,
            units=units,
            operation=operation,
            axes=axis_order,
            coordinate_inputs=coordinate_inputs,
            measurement_inputs=measurement_inputs,
            row_identity=row_identity,
            collection=collection,
            measurement_authority=measurement_authority,
            derivation=derivation,
            row_axis_name=("camera_frame" if "camera_frame" in axis_order else None),
            collection_axis_name=("chaser" if "chaser" in axis_order else None),
            collection_axis_role=("chaser" if "chaser" in axis_order else None),
            epoch_windows=epoch_windows,
        )
    except Exception as exc:
        _fail(f"Invalid shared measurement descriptor: {exc}.")


def _measurement_bindings(
    run_group: Any,
    coordinate_bindings: Mapping[str, BoundCanonicalCoordinateDescriptor],
    row_identity: BoundRowIdentityContract,
    collection: BoundCoordinateRecord,
    measurement_authority: BoundCoordinateRecord,
    derivation: BoundCoordinateRecord,
    epoch_windows: BoundCoordinateRecord,
    *,
    load: bool,
) -> dict[str, BoundCoordinateRecord]:
    fish = coordinate_bindings["positions/fish_centroid_arena_xy"]
    chaser = coordinate_bindings["positions/chaser_arena_xy"]
    coordinate_inputs = (fish, chaser)
    distances = run_group["distances"]
    summary = run_group["epoch_summary"]
    distributions = run_group["epoch_distributions"]
    result: dict[str, BoundCoordinateRecord] = {}

    def bind(
        path: str,
        node: Any,
        *,
        quantity: str,
        units: str,
        operation: str,
        axes: Sequence[str],
        input_paths: Sequence[str] = (),
        use_epoch_windows: bool = False,
    ) -> None:
        expected = _measurement_descriptor_record(
            node,
            quantity=quantity,
            units=units,
            operation=operation,
            axes=axes,
            coordinate_inputs=coordinate_inputs,
            measurement_inputs=tuple(result[name] for name in input_paths),
            row_identity=row_identity,
            collection=collection,
            measurement_authority=measurement_authority,
            derivation=derivation,
            epoch_windows=(epoch_windows if use_epoch_windows else None),
        )
        if load:
            try:
                result[path] = load_bound_array_measurement_descriptor(
                    node,
                    expected_record=expected,
                    attr_name=MEASUREMENT_DESCRIPTOR_ATTR,
                )
            except Exception as exc:
                _fail(
                    f"Measurement descriptor for {path!r} is stale or incomplete: {exc}."
                )
        else:
            try:
                result[path] = stamp_and_bind_array_measurement_descriptor(
                    node,
                    expected,
                    attr_name=MEASUREMENT_DESCRIPTOR_ATTR,
                )
            except Exception as exc:
                _fail(f"Unable to stamp measurement descriptor for {path!r}: {exc}.")

    bind(
        "distances/distance_px",
        distances["distance_px"],
        quantity="euclidean_distance",
        units="px",
        operation="euclidean_norm_chaser_minus_fish_in_shared_arena_frame_v1",
        axes=("camera_frame", "chaser"),
    )
    bind(
        "distances/distance_mm",
        distances["distance_mm"],
        quantity="euclidean_distance",
        units="mm",
        operation="divide_by_selected_projector_pixels_per_mm_v1",
        axes=("camera_frame", "chaser"),
        input_paths=("distances/distance_px",),
    )
    bind(
        "distances/nearest_distance_mm",
        distances["nearest_distance_mm"],
        quantity="nearest_euclidean_distance",
        units="mm",
        operation="finite_argmin_distance_then_value_v1",
        axes=("camera_frame",),
        input_paths=("distances/distance_mm",),
    )
    for name, operation in (
        ("mean_distance_mm", "finite_mean_within_epoch_window_v1"),
        ("min_distance_mm", "finite_minimum_within_epoch_window_v1"),
        ("p05_distance_mm", "finite_percentile_05_within_epoch_window_v1"),
        ("p50_distance_mm", "finite_percentile_50_within_epoch_window_v1"),
        ("p95_distance_mm", "finite_percentile_95_within_epoch_window_v1"),
    ):
        bind(
            f"epoch_summary/{name}",
            summary[name],
            quantity="epoch_aggregated_euclidean_distance",
            units="mm",
            operation=operation,
            axes=("stimulus_epoch_window", "chaser"),
            input_paths=("distances/distance_mm",),
            use_epoch_windows=True,
        )
    bind(
        "epoch_distributions/bin_edges_mm",
        distributions["bin_edges_mm"],
        quantity="distance_bin_edge",
        units="mm",
        operation="zero_to_ceil_global_finite_max_by_fixed_width_v1",
        axes=("distance_bin_edge",),
        input_paths=("distances/distance_mm",),
    )
    bind(
        "epoch_distributions/bin_centers_mm",
        distributions["bin_centers_mm"],
        quantity="distance_bin_center",
        units="mm",
        operation="adjacent_distance_bin_edge_midpoint_v1",
        axes=("distance_bin",),
        input_paths=("epoch_distributions/bin_edges_mm",),
    )
    bind(
        "epoch_distributions/hist_density",
        distributions["hist_density"],
        quantity="distance_distribution_density",
        units="per_mm",
        operation="histogram_count_divided_by_valid_count_and_bin_width_v1",
        axes=("stimulus_epoch_window", "chaser", "distance_bin"),
        input_paths=(
            "distances/distance_mm",
            "epoch_distributions/bin_edges_mm",
        ),
        use_epoch_windows=True,
    )
    return result


def _surface_manifest_record(
    run_group: Any,
    row_identity: BoundRowIdentityContract,
    bindings: Mapping[str, BoundCanonicalCoordinateDescriptor],
    measurement_bindings: Mapping[str, BoundCoordinateRecord],
    input_record: BoundCoordinateRecord,
    derivation: BoundCoordinateRecord,
    measurement: BoundCoordinateRecord,
    collection: BoundCoordinateRecord,
) -> dict[str, Any]:
    coordinate_surfaces = {}
    for path, binding in sorted(bindings.items()):
        node = binding.coordinate_node
        attrs = node.attrs
        coordinate_surfaces[path] = {
            "payload": _payload(node),
            "coordinate_descriptor_ref": (
                f"/{canonical_node_path(node)}@{COORDINATE_DESCRIPTOR_ATTR}"
            ),
            "coordinate_descriptor_sha256": attrs[
                f"{COORDINATE_DESCRIPTOR_ATTR}{COORDINATE_DESCRIPTOR_DIGEST_SUFFIX}"
            ],
        }
    measurement_surfaces: dict[str, Any] = {}
    for path, binding in sorted(measurement_bindings.items()):
        node = run_group[path]
        measurement_surfaces[path] = {
            "payload": _payload(node),
            "measurement_descriptor_ref": binding.record_ref,
            "measurement_descriptor_sha256": binding.record_sha256,
        }
    return {
        "schema_id": SURFACE_MANIFEST_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "coordinate_contract": COORDINATE_CONTRACT,
        "row_identity": {
            "record_ref": row_identity.record_ref,
            "record_sha256": row_identity.record_sha256,
        },
        "chaser_collection": _record_pointer(collection),
        "input_authority": _record_pointer(input_record),
        "derivation": _record_pointer(derivation),
        "measurement_authority": _record_pointer(measurement),
        "coordinate_surfaces": coordinate_surfaces,
        "measurement_surfaces": measurement_surfaces,
    }


def _seal_record(
    run_group: Any,
    row_identity: BoundRowIdentityContract,
    input_record: BoundCoordinateRecord,
    derivation: BoundCoordinateRecord,
    measurement: BoundCoordinateRecord,
    collection: BoundCoordinateRecord,
    epoch_windows: BoundCoordinateRecord,
    manifest: BoundCoordinateRecord,
) -> dict[str, Any]:
    protected_arrays = {
        CAMERA_FRAME_KEY_ARRAY: _payload(run_group[CAMERA_FRAME_KEY_ARRAY]),
        "frames/camera_frame_id": _payload(run_group["frames/camera_frame_id"]),
        "frames/stimulus_frame_num": _payload(run_group["frames/stimulus_frame_num"]),
        "frames/timestamp_ns": _payload(run_group["frames/timestamp_ns"]),
        "frames/stimulus_epoch_window_id": _payload(
            run_group["frames/stimulus_epoch_window_id"]
        ),
        "positions/source_detection_row_index": _payload(
            run_group[f"positions/{SOURCE_DETECTION_ROW_ARRAY}"]
        ),
        "positions/fish_centroid_img_xy": _payload(
            run_group["positions/fish_centroid_img_xy"]
        ),
        "positions/fish_centroid_arena_xy": _payload(
            run_group["positions/fish_centroid_arena_xy"]
        ),
        "positions/chaser_arena_xy": _payload(run_group["positions/chaser_arena_xy"]),
        "positions/fish_valid": _payload(run_group["positions/fish_valid"]),
        "positions/chaser_valid": _payload(run_group["positions/chaser_valid"]),
        "distances/distance_px": _payload(run_group["distances/distance_px"]),
        "distances/distance_mm": _payload(run_group["distances/distance_mm"]),
        "distances/nearest_chaser_index": _payload(
            run_group["distances/nearest_chaser_index"]
        ),
        "distances/nearest_distance_mm": _payload(
            run_group["distances/nearest_distance_mm"]
        ),
        "chasers/chaser_index": _payload(run_group["chasers/chaser_index"]),
        "chasers/stimulus_instance_id_bytes": _payload(
            run_group["chasers/stimulus_instance_id_bytes"]
        ),
        "chasers/source_track_key_bytes": _payload(
            run_group["chasers/source_track_key_bytes"]
        ),
    }
    return {
        "schema_id": PUBLICATION_SEAL_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "run_path": f"/{canonical_node_path(run_group)}",
        "coordinate_contract": COORDINATE_CONTRACT,
        "row_identity": {
            "record_ref": row_identity.record_ref,
            "record_sha256": row_identity.record_sha256,
        },
        "records": {
            "input_authority": _record_pointer(input_record),
            "derivation": _record_pointer(derivation),
            "measurement_authority": _record_pointer(measurement),
            "chaser_collection": _record_pointer(collection),
            "epoch_window_identity": _record_pointer(epoch_windows),
            "surface_manifest": _record_pointer(manifest),
        },
        "protected_arrays": protected_arrays,
    }


@dataclass(frozen=True, init=False)
class BoundChaserDistanceRun:
    run_path: str
    source_context: BoundChaserDistanceSourceContext = field(repr=False, compare=False)
    row_identity: BoundRowIdentityContract = field(repr=False, compare=False)
    coordinate_surfaces: Mapping[str, BoundCanonicalCoordinateDescriptor] = field(
        repr=False,
        compare=False,
    )
    measurement_surfaces: Mapping[str, BoundCoordinateRecord] = field(
        repr=False,
        compare=False,
    )
    input_authority: BoundCoordinateRecord = field(repr=False, compare=False)
    derivation: BoundCoordinateRecord = field(repr=False, compare=False)
    measurement_authority: BoundCoordinateRecord = field(repr=False, compare=False)
    chaser_collection: BoundCoordinateRecord = field(repr=False, compare=False)
    epoch_window_identity: BoundCoordinateRecord = field(repr=False, compare=False)
    surface_manifest: BoundCoordinateRecord = field(repr=False, compare=False)
    publication_seal: BoundCoordinateRecord = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self, **kwargs: Any) -> None:
        if kwargs.pop("_verification_seal", None) is not _BOUND_RUN_SEAL:
            _fail("Bound chaser-distance runs cannot be constructed directly.")
        for name, value in kwargs.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _BOUND_RUN_SEAL)


def _expected_arrays(run_group: Any, derived: ChaserDistanceCoordinateArrays) -> None:
    expected = {
        CAMERA_FRAME_KEY_ARRAY: derived.camera_frame_index,
        "frames/camera_frame_id": derived.camera_frame_index,
        "frames/stimulus_frame_num": derived.stimulus_frame_num,
        "frames/timestamp_ns": derived.timestamp_ns,
        f"positions/{SOURCE_DETECTION_ROW_ARRAY}": derived.source_detection_row_index,
        "positions/fish_centroid_img_xy": derived.fish_centroid_img_xy,
        "positions/fish_centroid_arena_xy": derived.fish_centroid_arena_xy,
        "positions/chaser_arena_xy": derived.chaser_arena_xy,
        "positions/fish_valid": derived.fish_valid,
        "positions/chaser_valid": derived.chaser_valid,
        "distances/distance_px": derived.distance_px,
        "distances/distance_mm": derived.distance_mm,
        "distances/nearest_chaser_index": derived.nearest_chaser_index,
        "distances/nearest_distance_mm": derived.nearest_distance_mm,
        "chasers/chaser_index": derived.chaser_indices,
    }
    for path, values in expected.items():
        try:
            node = run_group[path]
        except Exception as exc:
            _fail(f"Canonical output array {path!r} is missing: {exc}.")
        _equal_array(_array(node, label=path), values, label=path)


def _validate_derived_measurement_arrays(
    run_group: Any,
    derived: ChaserDistanceCoordinateArrays,
) -> None:
    """Recompute summaries/distributions before their records can be trusted."""

    summary = run_group["epoch_summary"]
    distributions = run_group["epoch_distributions"]
    starts = _array(summary["start_frame"], label="epoch_summary/start_frame")
    ends = _array(summary["end_frame"], label="epoch_summary/end_frame")
    window_ids = _array(summary["window_id"], label="epoch_summary/window_id")
    if (
        starts.dtype != np.dtype("<i8")
        or ends.dtype != np.dtype("<i8")
        or window_ids.dtype != np.dtype("<i4")
        or starts.ndim != 1
        or ends.shape != starts.shape
        or window_ids.shape != starts.shape
    ):
        _fail("Epoch summary window arrays have invalid exact layouts.")
    raw_threshold = summary.attrs.get("threshold_mm")
    if (
        isinstance(raw_threshold, bool)
        or not isinstance(raw_threshold, (int, float))
        or not np.isfinite(float(raw_threshold))
        or float(raw_threshold) <= 0
    ):
        _fail("Epoch summary requires one exact positive threshold_mm.")
    threshold = float(raw_threshold)
    n_windows = int(starts.shape[0])
    n_chasers = int(derived.distance_mm.shape[1])
    counts = np.zeros((n_windows, n_chasers), dtype=np.int64)
    mean = np.full((n_windows, n_chasers), np.nan, dtype=np.float32)
    minimum = np.full((n_windows, n_chasers), np.nan, dtype=np.float32)
    p05 = np.full((n_windows, n_chasers), np.nan, dtype=np.float32)
    p50 = np.full((n_windows, n_chasers), np.nan, dtype=np.float32)
    p95 = np.full((n_windows, n_chasers), np.nan, dtype=np.float32)
    fraction = np.full((n_windows, n_chasers), np.nan, dtype=np.float32)
    for window, (raw_start, raw_end) in enumerate(
        zip(starts, ends, strict=True)
    ):
        start = max(0, int(raw_start))
        end = min(derived.distance_mm.shape[0] - 1, int(raw_end))
        if end < start:
            continue
        for chaser in range(n_chasers):
            mask = (
                derived.fish_valid[start : end + 1]
                & derived.chaser_valid[start : end + 1, chaser]
                & np.isfinite(derived.distance_mm[start : end + 1, chaser])
            )
            values = derived.distance_mm[start : end + 1, chaser][mask]
            if values.size == 0:
                continue
            counts[window, chaser] = int(values.size)
            mean[window, chaser] = float(np.mean(values))
            minimum[window, chaser] = float(np.min(values))
            p05[window, chaser] = float(np.percentile(values, 5))
            p50[window, chaser] = float(np.percentile(values, 50))
            p95[window, chaser] = float(np.percentile(values, 95))
            fraction[window, chaser] = float(np.mean(values <= threshold))
    expected_summary = {
        "valid_frame_count": counts,
        "mean_distance_mm": mean,
        "min_distance_mm": minimum,
        "p05_distance_mm": p05,
        "p50_distance_mm": p50,
        "p95_distance_mm": p95,
        "fraction_within_threshold": fraction,
    }
    for name, expected in expected_summary.items():
        _equal_array(
            _array(summary[name], label=f"epoch_summary/{name}"),
            expected,
            label=f"epoch_summary/{name}",
        )

    raw_bin_width = distributions.attrs.get("bin_width_mm")
    if (
        isinstance(raw_bin_width, bool)
        or not isinstance(raw_bin_width, (int, float))
        or not np.isfinite(float(raw_bin_width))
        or float(raw_bin_width) <= 0
    ):
        _fail("Epoch distributions require one exact positive bin_width_mm.")
    bin_width = float(raw_bin_width)
    finite = derived.distance_mm[np.isfinite(derived.distance_mm)]
    max_distance = float(np.max(finite)) if finite.size else bin_width
    max_edge = max(
        bin_width,
        float(np.ceil(max_distance / bin_width) * bin_width),
    )
    bin_edges = np.arange(
        0.0,
        max_edge + bin_width * 0.5,
        bin_width,
        dtype=np.float32,
    )
    if bin_edges.shape[0] < 2:
        bin_edges = np.asarray([0.0, bin_width], dtype=np.float32)
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2.0).astype(np.float32)
    hist_counts = np.zeros(
        (n_windows, n_chasers, int(bin_centers.shape[0])),
        dtype=np.uint32,
    )
    density = np.zeros(hist_counts.shape, dtype=np.float32)
    for window, (raw_start, raw_end) in enumerate(
        zip(starts, ends, strict=True)
    ):
        start = max(0, int(raw_start))
        end = min(derived.distance_mm.shape[0] - 1, int(raw_end))
        if end < start:
            continue
        for chaser in range(n_chasers):
            mask = (
                derived.fish_valid[start : end + 1]
                & derived.chaser_valid[start : end + 1, chaser]
                & np.isfinite(derived.distance_mm[start : end + 1, chaser])
            )
            values = derived.distance_mm[start : end + 1, chaser][mask]
            if values.size == 0:
                continue
            histogram, _ = np.histogram(values, bins=bin_edges)
            hist_counts[window, chaser] = histogram.astype(np.uint32, copy=False)
            density[window, chaser] = histogram.astype(np.float32) / (
                float(values.size) * bin_width
            )
    expected_distributions = {
        "window_id": window_ids,
        "chaser_index": derived.chaser_indices,
        "bin_edges_mm": bin_edges,
        "bin_centers_mm": bin_centers,
        "hist_counts": hist_counts,
        "hist_density": density,
        "valid_sample_count": counts,
    }
    for name, expected in expected_distributions.items():
        _equal_array(
            _array(distributions[name], label=f"epoch_distributions/{name}"),
            expected,
            label=f"epoch_distributions/{name}",
        )


@proof_verification_operation
def publish_chaser_distance_coordinate_contract(
    root_node: Any,
    run_group: Any,
    *,
    source_context: BoundChaserDistanceSourceContext,
) -> BoundChaserDistanceRun:
    """Publish descriptors and an immutable seal before run activation."""

    context = require_bound_chaser_distance_source_context(source_context)
    if (
        archive_identity(root_node) != context.archive_identity
        or archive_identity(run_group) != context.archive_identity
    ):
        _fail("Chaser-distance source and output run span different archives/stores.")
    if run_group.attrs.get("stage_selector_eligible") is not False:
        _fail("Coordinate publication requires a staged selector-ineligible run.")
    for attr in (
        INPUT_AUTHORITY_ATTR,
        DERIVATION_ATTR,
        MEASUREMENT_AUTHORITY_ATTR,
        SURFACE_MANIFEST_ATTR,
        PUBLICATION_SEAL_ATTR,
    ):
        if attr in run_group.attrs or f"{attr}_sha256" in run_group.attrs:
            _fail(f"Future publication refuses occupied coordinate record {attr!r}.")
    if (
        EPOCH_WINDOW_AUTHORITY_ATTR in run_group["epoch_summary"].attrs
        or f"{EPOCH_WINDOW_AUTHORITY_ATTR}_sha256"
        in run_group["epoch_summary"].attrs
    ):
        _fail("Future publication refuses an occupied epoch-window authority.")
    for path in (
        "positions/fish_centroid_img_xy",
        "positions/fish_centroid_arena_xy",
        "positions/chaser_arena_xy",
    ):
        node = run_group[path]
        if (
            COORDINATE_DESCRIPTOR_ATTR in node.attrs
            or f"{COORDINATE_DESCRIPTOR_ATTR}{COORDINATE_DESCRIPTOR_DIGEST_SUFFIX}"
            in node.attrs
        ):
            _fail(f"Future publication refuses occupied coordinate attrs on {path!r}.")
    for path in (
        "distances/distance_px",
        "distances/distance_mm",
        "distances/nearest_distance_mm",
        "epoch_summary/mean_distance_mm",
        "epoch_summary/min_distance_mm",
        "epoch_summary/p05_distance_mm",
        "epoch_summary/p50_distance_mm",
        "epoch_summary/p95_distance_mm",
        "epoch_distributions/bin_edges_mm",
        "epoch_distributions/bin_centers_mm",
        "epoch_distributions/hist_density",
    ):
        node = run_group[path]
        if (
            MEASUREMENT_DESCRIPTOR_ATTR in node.attrs
            or f"{MEASUREMENT_DESCRIPTOR_ATTR}_sha256" in node.attrs
        ):
            _fail(f"Future publication refuses occupied measurement attrs on {path!r}.")

    derived = derive_chaser_distance_coordinate_arrays(context)
    _expected_arrays(run_group, derived)
    _validate_derived_measurement_arrays(run_group, derived)
    key_node = run_group[CAMERA_FRAME_KEY_ARRAY]
    identity = stamp_and_bind_row_identity_contract(
        run_group,
        key_node,
        contract=build_row_identity_contract(
            domain=STIMULUS_STATE_DOMAIN,
            values=derived.camera_frame_index,
            components=("camera_frame_index",),
        ),
    )
    input_record = stamp_and_bind_persisted_coordinate_record(
        run_group,
        _input_authority_record(context),
        attr_name=INPUT_AUTHORITY_ATTR,
    )
    derivation = stamp_and_bind_persisted_coordinate_record(
        run_group,
        _derivation_record(run_group, input_record),
        attr_name=DERIVATION_ATTR,
    )
    measurement = stamp_and_bind_persisted_coordinate_record(
        run_group,
        _measurement_record(context, input_record, derivation),
        attr_name=MEASUREMENT_AUTHORITY_ATTR,
    )
    collection = stamp_and_bind_persisted_coordinate_record(
        run_group["chasers"],
        _collection_record(run_group),
        attr_name=CHASER_COLLECTION_AUTHORITY_ATTR,
    )
    bindings = _coordinate_bindings(
        context,
        run_group,
        identity,
        input_record,
        derivation,
        collection,
        load=False,
    )
    stamp_bound_canonical_coordinate_descriptors(bindings.values())
    bindings = _coordinate_bindings(
        context,
        run_group,
        identity,
        input_record,
        derivation,
        collection,
        load=True,
    )
    epoch_windows = stamp_and_bind_persisted_coordinate_record(
        run_group["epoch_summary"],
        _epoch_window_authority_record(context, run_group),
        attr_name=EPOCH_WINDOW_AUTHORITY_ATTR,
    )
    measurement_bindings = _measurement_bindings(
        run_group,
        bindings,
        identity,
        collection,
        measurement,
        derivation,
        epoch_windows,
        load=False,
    )
    measurement_bindings = _measurement_bindings(
        run_group,
        bindings,
        identity,
        collection,
        measurement,
        derivation,
        epoch_windows,
        load=True,
    )
    manifest = stamp_and_bind_persisted_coordinate_record(
        run_group,
        _surface_manifest_record(
            run_group,
            identity,
            bindings,
            measurement_bindings,
            input_record,
            derivation,
            measurement,
            collection,
        ),
        attr_name=SURFACE_MANIFEST_ATTR,
    )
    seal = stamp_and_bind_persisted_coordinate_record(
        run_group,
        _seal_record(
            run_group,
            identity,
            input_record,
            derivation,
            measurement,
            collection,
            epoch_windows,
            manifest,
        ),
        attr_name=PUBLICATION_SEAL_ATTR,
    )
    run_group.attrs.update(
        {
            COORDINATE_CONTRACT_ATTR: COORDINATE_CONTRACT,
            COORDINATE_CONTRACT_EPOCH_ATTR: COORDINATE_CONTRACT_EPOCH,
        }
    )
    return BoundChaserDistanceRun(
        run_path=canonical_node_path(run_group),
        source_context=context,
        row_identity=identity,
        coordinate_surfaces=bindings,
        measurement_surfaces=measurement_bindings,
        input_authority=input_record,
        derivation=derivation,
        measurement_authority=measurement,
        chaser_collection=collection,
        epoch_window_identity=epoch_windows,
        surface_manifest=manifest,
        publication_seal=seal,
        _verification_seal=_BOUND_RUN_SEAL,
    )


@proof_verification_operation
def _load_bound_chaser_distance_run(
    root_node: Any,
    run_path: str,
    *,
    expected_selector_eligible: bool,
) -> BoundChaserDistanceRun:
    """Freshly verify one complete canonical run at a controlled lifecycle state."""

    path = str(run_path).strip().strip("/")
    if len(path.split("/")) == 1:
        path = f"analysis/chaser_distance_runs/{path}"
    parts = path.split("/")
    if len(parts) != 3 or parts[:2] != ["analysis", "chaser_distance_runs"]:
        _fail("Chaser-distance reader requires analysis/chaser_distance_runs/<run>.")
    try:
        run_group = root_node[path]
    except Exception as exc:
        _fail(f"Canonical chaser-distance run is missing: {exc}.")
    attrs = run_group.attrs
    if (
        attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or attrs.get("stage_selector_eligible") is not expected_selector_eligible
        or attrs.get(COORDINATE_CONTRACT_ATTR) != COORDINATE_CONTRACT
        or attrs.get(COORDINATE_CONTRACT_EPOCH_ATTR) != COORDINATE_CONTRACT_EPOCH
        or attrs.get("coordinate_publication_status") != "sealed_canonical_v2"
    ):
        _fail(
            "Canonical reader requires a complete coordinate publication with "
            f"stage_selector_eligible={expected_selector_eligible!r}."
        )
    input_record = bind_persisted_coordinate_record(run_group, attr_name=INPUT_AUTHORITY_ATTR)
    detection_path = input_record.record.get("source_detection_path")
    stimulus_path = input_record.record.get("source_stimulus_path")
    if not isinstance(stimulus_path, str) or not stimulus_path.startswith("analysis/stimulus_runs/"):
        _fail("Input authority has no exact controlled stimulus path.")
    context = load_chaser_distance_source_context(
        root_node,
        detection_path=str(detection_path),
        stimulus_run=stimulus_path.rsplit("/", 1)[-1],
    )
    if input_record.record != _input_authority_record(context):
        _fail("Chaser-distance input authority is stale or conflicts with live sources.")
    derived = derive_chaser_distance_coordinate_arrays(context)
    _expected_arrays(run_group, derived)
    _validate_derived_measurement_arrays(run_group, derived)
    identity = load_bound_row_identity_contract(
        run_group,
        run_group[CAMERA_FRAME_KEY_ARRAY],
    )
    derivation = bind_persisted_coordinate_record(run_group, attr_name=DERIVATION_ATTR)
    if derivation.record != _derivation_record(run_group, input_record):
        _fail("Chaser-distance coordinate derivation is stale or incomplete.")
    measurement = bind_persisted_coordinate_record(
        run_group,
        attr_name=MEASUREMENT_AUTHORITY_ATTR,
    )
    if measurement.record != _measurement_record(context, input_record, derivation):
        _fail("Chaser-distance measurement authority is stale or incomplete.")
    collection = bind_persisted_coordinate_record(
        run_group["chasers"],
        attr_name=CHASER_COLLECTION_AUTHORITY_ATTR,
    )
    if collection.record != _collection_record(run_group):
        _fail("Chaser collection authority is stale or incomplete.")
    bindings = _coordinate_bindings(
        context,
        run_group,
        identity,
        input_record,
        derivation,
        collection,
        load=True,
    )
    epoch_windows = bind_persisted_coordinate_record(
        run_group["epoch_summary"],
        attr_name=EPOCH_WINDOW_AUTHORITY_ATTR,
    )
    if epoch_windows.record != _epoch_window_authority_record(context, run_group):
        _fail("Epoch-window identity authority is stale or incomplete.")
    measurement_bindings = _measurement_bindings(
        run_group,
        bindings,
        identity,
        collection,
        measurement,
        derivation,
        epoch_windows,
        load=True,
    )
    manifest = bind_persisted_coordinate_record(run_group, attr_name=SURFACE_MANIFEST_ATTR)
    if manifest.record != _surface_manifest_record(
        run_group,
        identity,
        bindings,
        measurement_bindings,
        input_record,
        derivation,
        measurement,
        collection,
    ):
        _fail("Chaser-distance surface manifest is stale or incomplete.")
    seal = bind_persisted_coordinate_record(run_group, attr_name=PUBLICATION_SEAL_ATTR)
    if seal.record != _seal_record(
        run_group,
        identity,
        input_record,
        derivation,
        measurement,
        collection,
        epoch_windows,
        manifest,
    ):
        _fail("Chaser-distance publication seal is stale or incomplete.")
    return BoundChaserDistanceRun(
        run_path=path,
        source_context=context,
        row_identity=identity,
        coordinate_surfaces=bindings,
        measurement_surfaces=measurement_bindings,
        input_authority=input_record,
        derivation=derivation,
        measurement_authority=measurement,
        chaser_collection=collection,
        epoch_window_identity=epoch_windows,
        surface_manifest=manifest,
        publication_seal=seal,
        _verification_seal=_BOUND_RUN_SEAL,
    )


def load_bound_chaser_distance_run(
    root_node: Any,
    run_path: str,
) -> BoundChaserDistanceRun:
    """Freshly verify one complete, selector-eligible canonical run."""

    return _load_bound_chaser_distance_run(
        root_node,
        run_path,
        expected_selector_eligible=True,
    )


def activate_chaser_distance_run(
    root_node: Any,
    parent_group: Any,
    run_group: Any,
    *,
    run_name: str,
) -> None:
    """Lease selectors to one generation, publish them, then expose the child."""

    name = str(run_name).strip()
    expected_path = f"analysis/chaser_distance_runs/{name}"
    expected_archive = archive_identity(root_node)
    if (
        not name
        or "/" in name
        or canonical_node_path(parent_group) != "analysis/chaser_distance_runs"
        or canonical_node_path(run_group) != expected_path
        or expected_archive != archive_identity(parent_group)
        or expected_archive != archive_identity(run_group)
        or run_group.attrs.get("stage_selector_eligible") is not False
    ):
        _fail("Chaser-distance activation requires one exact staged run path.")
    snapshot = _activation_snapshot(parent_group.attrs)
    pending_present, pending_value = snapshot["latest_pending"]
    if pending_present and pending_value not in (None, ""):
        _fail("Chaser-distance activation refuses an occupied latest_pending selector.")
    base_generation = _activation_generation(snapshot)
    next_generation = base_generation + 1
    policy_present, policy = snapshot[CHASER_DISTANCE_PUBLICATION_POLICY_ATTR]
    if policy_present and policy != CHASER_DISTANCE_PUBLICATION_POLICY:
        _fail("Chaser-distance parent uses an unsupported publication policy.")
    if base_generation > 0 and not policy_present:
        _fail("Chaser-distance parent generation lacks its publication policy.")
    prior_lease_present, prior_lease = snapshot[
        CHASER_DISTANCE_PARENT_PUBLICATION_LEASE_ATTR
    ]
    if prior_lease_present:
        prior_owner = (
            prior_lease.get("owner_uuid")
            if isinstance(prior_lease, Mapping)
            else None
        )
        prior_path = (
            prior_lease.get("run_path")
            if isinstance(prior_lease, Mapping)
            else None
        )
        prior_name = (
            prior_path.removeprefix("analysis/chaser_distance_runs/")
            if isinstance(prior_path, str)
            else ""
        )
        if (
            base_generation == 0
            or not isinstance(prior_lease, Mapping)
            or prior_lease.get("schema_id")
            != "palette.chaser_distance_publication_lease"
            or prior_lease.get("schema_version") != 1
            or prior_lease.get("policy") != CHASER_DISTANCE_PUBLICATION_POLICY
            or not isinstance(prior_owner, str)
            or not prior_owner
            or prior_lease.get("publication_owner") != prior_owner
            or not isinstance(prior_path, str)
            or not prior_path.startswith("analysis/chaser_distance_runs/")
            or not prior_name
            or prior_name in {".", ".."}
            or "/" in prior_name
            or prior_lease.get("base_generation") != base_generation - 1
            or prior_lease.get("next_generation") != base_generation
        ):
            _fail("Chaser-distance parent has an invalid prior publication lease.")
    elif base_generation > 0:
        _fail("Chaser-distance parent has an invalid prior publication lease.")
    owner_uuid = str(uuid.uuid4())
    lease = _activation_lease_record(
        owner_uuid=owner_uuid,
        run_path=expected_path,
        base_generation=base_generation,
    )
    overrides: dict[str, tuple[bool, Any]] = {}
    mutations: list[tuple[str, tuple[bool, Any], Any]] = []

    def write_parent(name_: str, value: Any) -> None:
        parent = _fresh_activation_parent(root_node, expected_archive)
        _require_activation_parent_state(parent, snapshot, overrides=overrides)
        previous = overrides.get(name_, snapshot[name_])
        written = copy.deepcopy(value)
        mutations.append((name_, previous, written))
        _write_activation_attr(parent.attrs, name_, written)
        overrides[name_] = (True, written)
        parent = _fresh_activation_parent(root_node, expected_archive)
        _require_activation_parent_state(parent, snapshot, overrides=overrides)

    bound = _load_bound_chaser_distance_run(
        root_node,
        expected_path,
        expected_selector_eligible=False,
    )
    expected_seal = copy.deepcopy(bound.publication_seal.record)
    try:
        parent = _fresh_activation_parent(root_node, expected_archive)
        _require_activation_parent_state(parent, snapshot, overrides=overrides)
        write_parent(CHASER_DISTANCE_PARENT_PUBLICATION_LEASE_ATTR, lease)

        fresh = _load_bound_chaser_distance_run(
            root_node,
            expected_path,
            expected_selector_eligible=False,
        )
        if fresh.publication_seal.record_sha256 != bound.publication_seal.record_sha256:
            _fail("Chaser-distance candidate changed after lease acquisition.")

        # Selectors are published before eligibility. The lease and generation
        # are re-read around every mutation by write_parent().
        write_parent(RUN_LATEST_COMPLETE_ATTR, name)
        write_parent("latest", name)
        write_parent(
            CHASER_DISTANCE_PUBLICATION_POLICY_ATTR,
            CHASER_DISTANCE_PUBLICATION_POLICY,
        )
        write_parent(CHASER_DISTANCE_PUBLICATION_GENERATION_ATTR, next_generation)

        parent = _fresh_activation_parent(root_node, expected_archive)
        if (
            not _owned_activation_epoch(
                parent,
                lease,
                base_generation=base_generation,
                next_generation=next_generation,
            )
            or parent.attrs.get(RUN_LATEST_COMPLETE_ATTR) != name
            or parent.attrs.get("latest") != name
        ):
            _fail("Chaser-distance selector epoch did not persist before activation.")
        final = _load_bound_chaser_distance_run(
            root_node,
            expected_path,
            expected_selector_eligible=False,
        )
        if final.publication_seal.record_sha256 != bound.publication_seal.record_sha256:
            _fail("Chaser-distance candidate changed during selector publication.")
        final_run = root_node[expected_path]
        if (
            canonical_node_path(final_run) != expected_path
            or archive_identity(final_run) != expected_archive
            or final_run.attrs.get(PUBLICATION_SEAL_ATTR) != expected_seal
            or final_run.attrs.get("stage_selector_eligible") is not False
        ):
            _fail("Chaser-distance candidate lost exact ineligible ownership.")
        # Literal commit point. No fallible validation or metadata write follows.
        _write_activation_attr(final_run.attrs, "stage_selector_eligible", True)
    except BaseException as exc:
        try:
            parent = _fresh_activation_parent(root_node, expected_archive)
            current_run = root_node[expected_path]
            committed = (
                _owned_activation_epoch(
                    parent,
                    lease,
                    base_generation=base_generation,
                    next_generation=next_generation,
                )
                and parent.attrs.get(CHASER_DISTANCE_PUBLICATION_GENERATION_ATTR)
                == next_generation
                and parent.attrs.get(RUN_LATEST_COMPLETE_ATTR) == name
                and parent.attrs.get("latest") == name
                and current_run.attrs.get(PUBLICATION_SEAL_ATTR) == expected_seal
                and current_run.attrs.get("stage_selector_eligible") is True
            )
        except BaseException:
            committed = False
        if committed:
            return
        try:
            _rollback_owned_activation(
                root_node,
                expected_archive=expected_archive,
                expected_path=expected_path,
                expected_seal=expected_seal,
                lease=lease,
                base_generation=base_generation,
                next_generation=next_generation,
                mutations=mutations,
            )
        except BaseException as rollback_exc:  # pragma: no cover - hostile store
            raise ChaserDistanceCoordinateError(
                "Chaser-distance activation failed and owned rollback was incomplete: "
                f"{rollback_exc}."
            ) from exc
        raise


__all__ = [
    "CAMERA_FRAME_KEY_ARRAY",
    "CHASER_COLLECTION_AUTHORITY_ATTR",
    "CHASER_DISTANCE_PARENT_PUBLICATION_LEASE_ATTR",
    "CHASER_DISTANCE_PUBLICATION_GENERATION_ATTR",
    "CHASER_DISTANCE_PUBLICATION_POLICY",
    "CHASER_DISTANCE_PUBLICATION_POLICY_ATTR",
    "COORDINATE_CONTRACT",
    "COORDINATE_CONTRACT_ATTR",
    "COORDINATE_CONTRACT_EPOCH",
    "COORDINATE_CONTRACT_EPOCH_ATTR",
    "ChaserDistanceCoordinateArrays",
    "ChaserDistanceCoordinateError",
    "BoundChaserDistanceRun",
    "BoundChaserDistanceSourceContext",
    "DERIVATION_ATTR",
    "EPOCH_WINDOW_AUTHORITY_ATTR",
    "INPUT_AUTHORITY_ATTR",
    "MEASUREMENT_AUTHORITY_ATTR",
    "MEASUREMENT_DESCRIPTOR_ATTR",
    "PUBLICATION_SEAL_ATTR",
    "SOURCE_DETECTION_ROW_ARRAY",
    "SURFACE_MANIFEST_ATTR",
    "activate_chaser_distance_run",
    "derive_chaser_distance_coordinate_arrays",
    "load_bound_chaser_distance_run",
    "load_chaser_distance_source_context",
    "publish_chaser_distance_coordinate_contract",
    "require_bound_chaser_distance_source_context",
    "source_camera_to_arena_xy",
]
