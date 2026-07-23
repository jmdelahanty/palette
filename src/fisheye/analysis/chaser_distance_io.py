"""Verified logical reader for canonical chaser-distance publications.

Normal scientific readers cross the coordinate-publication boundary here.  A
run name is only discovery input: the exact selected child must be complete,
selector-eligible, and freshly satisfy the canonical chaser-distance seal.
Returned arrays are detached, read-only copies, and scalar coordinate semantics
come from typed persisted authorities rather than historical run attrs.

The current canonical publication does not yet seal the protocol-derived
behavior-role/color surfaces.  This reader therefore refuses to present those
surfaces as scientific authority.  Identity-bearing ``chaser_index`` and
controlled collection labels remain available because they are protected by
the publication seal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, NoReturn

import numpy as np
import zarr

from fisheye.analysis.chaser_distance_coordinate_publication import (
    BoundChaserDistanceRun,
    ChaserDistanceCoordinateError,
    load_bound_chaser_distance_run,
)
from fisheye.shared.archive_identity import ArchiveIdentity
from fisheye.shared.coordinate_descriptor import CanonicalCoordinateDescriptor
from fisheye.shared.coordinate_frame_record import array_values_sha256


CHASER_DISTANCE_READ_AUTHORITY_SCHEMA_ID = (
    "palette.chaser_distance_read_authority"
)
CHASER_DISTANCE_READ_AUTHORITY_SCHEMA_VERSION = 1
VERIFIED_AUTHORITY_STATUS = "verified_canonical_chaser_distance_v1"
UNAVAILABLE_BEHAVIOR_AUTHORITY_STATUS = "unavailable_unsealed_protocol_semantics"

_RUNS_PATH = "analysis/chaser_distance_runs"
_SNAPSHOT_SEAL = object()


class ChaserDistanceReadError(ChaserDistanceCoordinateError):
    """Raised when a normal chaser-distance read cannot remain fail-closed."""


def _fail(message: str) -> None:
    raise ChaserDistanceReadError(message)


def _controlled_child_name(value: Any, *, selector: str) -> str:
    if not isinstance(value, str):
        _fail(f"Chaser-distance {selector} must be one exact child name.")
    name = value.strip()
    if not name or name in {".", ".."} or "/" in name:
        _fail(f"Chaser-distance {selector} must be one exact child name.")
    return name


def _runs_parent(root: Any) -> Any:
    try:
        return root[_RUNS_PATH]
    except Exception as exc:
        _fail(f"Archive has no {_RUNS_PATH} group: {exc}.")


def resolve_chaser_distance_run_path(
    root: Any,
    *,
    run_name: str = "latest",
) -> tuple[str, str]:
    """Resolve one exact canonical child without scanning or ``latest`` fallback.

    ``latest`` means the exact manually approved ``authoritative_run`` pointer
    when present, otherwise the exact lifecycle ``latest_complete`` pointer.
    A malformed, missing, or stale pointer fails closed.  The historical
    ``latest`` attr and lexicographic child order are never consulted.
    """

    parent = _runs_parent(root)
    requested = str(run_name).strip()
    if not requested or requested == "latest":
        authoritative = parent.attrs.get("authoritative_run")
        if authoritative is not None:
            resolved = _controlled_child_name(
                authoritative,
                selector="authoritative_run",
            )
        else:
            resolved = _controlled_child_name(
                parent.attrs.get("latest_complete"),
                selector="latest_complete",
            )
    else:
        resolved = _controlled_child_name(requested, selector="explicit run")
    try:
        child = parent[resolved]
    except Exception as exc:
        _fail(f"Selected chaser-distance child {resolved!r} is missing: {exc}.")
    if not isinstance(child, zarr.Group):
        _fail(f"Selected chaser-distance child {resolved!r} is not a group.")
    return resolved, f"{_RUNS_PATH}/{resolved}"


def _payload_mapping(
    record: Mapping[str, Any],
    field_name: str,
) -> Mapping[str, Any]:
    value = record.get(field_name)
    if not isinstance(value, Mapping):
        _fail(f"Canonical chaser-distance record lacks {field_name!r} payloads.")
    return value


def _copy_exact_payload(
    run_group: Any,
    *,
    run_path: str,
    relative_path: str,
    payload: Any,
) -> np.ndarray:
    if not isinstance(payload, Mapping):
        _fail(f"Sealed payload for {relative_path!r} is malformed.")
    expected_ref = f"/{run_path}/{relative_path}"
    if payload.get("array_ref") != expected_ref:
        _fail(
            f"Sealed payload for {relative_path!r} points at a different array."
        )
    raw_shape = payload.get("shape")
    raw_dtype = payload.get("dtype")
    raw_digest = payload.get("content_sha256")
    if (
        not isinstance(raw_shape, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in raw_shape
        )
        or not isinstance(raw_dtype, str)
        or not isinstance(raw_digest, str)
    ):
        _fail(f"Sealed payload for {relative_path!r} has invalid metadata.")
    try:
        expected_dtype = np.dtype(raw_dtype)
        node = run_group[relative_path]
        values = np.array(node[:], copy=True, order="C")
        node_shape = tuple(int(value) for value in node.shape)
        node_dtype = np.dtype(node.dtype)
    except Exception as exc:
        _fail(f"Unable to copy sealed array {relative_path!r}: {exc}.")
    expected_shape = tuple(raw_shape)
    if (
        expected_dtype.hasobject
        or values.dtype.hasobject
        or values.shape != expected_shape
        or values.dtype != expected_dtype
        or node_shape != expected_shape
        or node_dtype != expected_dtype
        or array_values_sha256(values) != raw_digest
    ):
        _fail(
            f"Verified chaser-distance surface {expected_ref} changed payload, "
            "dtype, or shape while being copied."
        )
    values.setflags(write=False)
    return values


def _decode_sealed_text_rows(values: np.ndarray, *, label: str) -> tuple[str, ...]:
    array = np.asarray(values)
    if array.ndim != 2 or array.dtype.kind not in "ui":
        _fail(f"{label} does not use canonical null-terminated byte rows.")
    try:
        decoded = tuple(
            bytes(np.asarray(row, dtype=np.uint8))
            .split(b"\0", 1)[0]
            .decode("utf-8", "strict")
            for row in array
        )
    except UnicodeDecodeError as exc:
        _fail(f"{label} is not canonical UTF-8: {exc}.")
    return decoded


def _same_binding(left: Any, right: Any) -> bool:
    return (
        left.record_ref == right.record_ref
        and left.record_sha256 == right.record_sha256
    )


def _load_verified_binding(root: Any, run_path: str) -> BoundChaserDistanceRun:
    try:
        return load_bound_chaser_distance_run(root, run_path)
    except ChaserDistanceReadError:
        raise
    except ChaserDistanceCoordinateError as exc:
        _fail(str(exc))


def _require_same_fresh_binding(
    before: BoundChaserDistanceRun,
    after: BoundChaserDistanceRun,
) -> None:
    if (
        before.run_path != after.run_path
        or before.source_context.archive_identity
        != after.source_context.archive_identity
        or before.source_context.signature_sha256
        != after.source_context.signature_sha256
        or not _same_binding(before.row_identity, after.row_identity)
        or not _same_binding(before.input_authority, after.input_authority)
        or not _same_binding(before.derivation, after.derivation)
        or not _same_binding(before.measurement_authority, after.measurement_authority)
        or not _same_binding(before.chaser_collection, after.chaser_collection)
        or not _same_binding(before.epoch_window_identity, after.epoch_window_identity)
        or not _same_binding(before.surface_manifest, after.surface_manifest)
        or not _same_binding(before.publication_seal, after.publication_seal)
    ):
        _fail("Canonical chaser-distance publication changed while being copied.")


def _validated_arena_descriptor(
    bound: BoundChaserDistanceRun,
) -> CanonicalCoordinateDescriptor:
    fish = bound.coordinate_surfaces["positions/fish_centroid_arena_xy"].descriptor
    chaser = bound.coordinate_surfaces["positions/chaser_arena_xy"].descriptor
    expected = {
        "profile_id": "arena_relative_canvas_px.top_left_y_down.v1",
        "space_id": "arena_relative_canvas_px",
        "geometry_type": "point_xy",
        "components": ("x", "y"),
        "component_units": ("px", "px"),
        "origin": "arena_top_left",
        "positive_x": "right",
        "positive_y": "down",
        "pixel_convention": "continuous",
        "reference_units": "px",
    }
    for name, descriptor in (("fish", fish), ("chaser", chaser)):
        actual = {
            "profile_id": descriptor.profile_id,
            "space_id": descriptor.space_id,
            "geometry_type": descriptor.geometry_type,
            "components": descriptor.components,
            "component_units": descriptor.component_units,
            "origin": descriptor.origin,
            "positive_x": descriptor.positive_directions.x,
            "positive_y": descriptor.positive_directions.y,
            "pixel_convention": descriptor.pixel_convention,
            "reference_units": descriptor.reference_extent.units,
        }
        if actual != expected:
            _fail(f"Canonical {name} arena coordinate descriptor is unsupported.")
    if (
        fish.reference_extent.width != chaser.reference_extent.width
        or fish.reference_extent.height != chaser.reference_extent.height
        or fish.reference_extent.authority != chaser.reference_extent.authority
    ):
        _fail("Fish and chaser arena coordinates use different reference extents.")
    width = fish.reference_extent.width
    height = fish.reference_extent.height
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, (int, float))
        or not isinstance(height, (int, float))
        or not float(width).is_integer()
        or not float(height).is_integer()
        or int(width) <= 0
        or int(height) <= 0
    ):
        _fail("Arena coordinate descriptor has no exact positive pixel extent.")
    return fish


@dataclass(frozen=True, init=False)
class ChaserDistanceReadSnapshot:
    """Detached logical tables from one freshly verified canonical run."""

    run_name: str
    run_path: str
    recording_id: str
    authority_status: str
    archive_identity: ArchiveIdentity = field(repr=False, compare=False)
    source_detection_path: str
    source_stimulus_run: str
    source_stimulus_path: str
    source_stimulus_epoch_run: str | None
    source_stimulus_epoch_path: str | None
    fps: float
    total_frames: int
    pixels_per_mm_projector: float
    coordinate_space_id: str
    coordinate_origin: str
    positive_x: str
    positive_y: str
    reference_width_px: int
    reference_height_px: int
    pixel_convention: str
    arena_coordinate_descriptor: CanonicalCoordinateDescriptor
    source_camera_coordinate_descriptor: CanonicalCoordinateDescriptor
    coordinate_descriptor_sha256: Mapping[str, str]
    measurement_descriptor_sha256: Mapping[str, str]
    behavior_authority_status: str
    publication_seal_ref: str
    publication_seal_sha256: str
    surface_manifest_ref: str
    surface_manifest_sha256: str
    row_identity_ref: str
    row_identity_sha256: str
    stimulus_state_key: np.ndarray = field(repr=False, compare=False)
    camera_frame_id: np.ndarray = field(repr=False, compare=False)
    stimulus_frame_num: np.ndarray = field(repr=False, compare=False)
    timestamp_ns: np.ndarray = field(repr=False, compare=False)
    stimulus_epoch_window_id: np.ndarray = field(repr=False, compare=False)
    source_detection_row_index: np.ndarray = field(repr=False, compare=False)
    fish_centroid_img_xy: np.ndarray = field(repr=False, compare=False)
    fish_centroid_arena_xy: np.ndarray = field(repr=False, compare=False)
    chaser_arena_xy: np.ndarray = field(repr=False, compare=False)
    fish_valid: np.ndarray = field(repr=False, compare=False)
    chaser_valid: np.ndarray = field(repr=False, compare=False)
    distance_px: np.ndarray = field(repr=False, compare=False)
    distance_mm: np.ndarray = field(repr=False, compare=False)
    nearest_chaser_index: np.ndarray = field(repr=False, compare=False)
    nearest_distance_mm: np.ndarray = field(repr=False, compare=False)
    chaser_index: np.ndarray = field(repr=False, compare=False)
    stimulus_instance_ids: tuple[str, ...]
    source_track_keys: tuple[str, ...]
    epoch_window_id: np.ndarray = field(repr=False, compare=False)
    epoch_label_bytes: np.ndarray = field(repr=False, compare=False)
    epoch_labels: tuple[str, ...]
    epoch_start_frame: np.ndarray = field(repr=False, compare=False)
    epoch_end_frame: np.ndarray = field(repr=False, compare=False)
    _authority_record_payload: Mapping[str, Any] = field(
        repr=False,
        compare=False,
    )
    _seal: object = field(repr=False, compare=False)

    def __init__(self, **kwargs: Any) -> None:
        if kwargs.pop("_verification_seal", None) is not _SNAPSHOT_SEAL:
            _fail("Verified chaser-distance snapshots cannot be constructed directly.")
        for name, value in kwargs.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _SNAPSHOT_SEAL)

    @property
    def chaser_indices(self) -> np.ndarray:
        """Controlled alias for the sealed collection identity axis."""

        return self.chaser_index

    def authority_record(self) -> dict[str, Any]:
        """Return the immutable authority inherited by a derived writer."""

        if self.authority_status != VERIFIED_AUTHORITY_STATUS:
            _fail("Only a verified canonical read can mint downstream authority.")
        return _deep_plain_copy(self._authority_record_payload)

    def require_behavior_authority(self) -> NoReturn:
        """Fail until protocol-derived roles/colors receive their own sealed record."""

        _fail(
            "Canonical chaser-distance publication does not yet provide sealed "
            "behavior-role/color authority; this consumer must fail closed. "
            "Remediation: publish and validate a canonical behavior-role authority "
            "before enabling this analysis; raw protocol_json recovery is forbidden."
        )

    def require_arena_geometry_authority(self) -> NoReturn:
        """Fail until the arena centre/radius calibration is independently sealed."""

        _fail(
            "Canonical chaser-distance publication does not yet provide sealed arena "
            "centre/radius geometry authority; this consumer must fail closed. "
            "Remediation: publish exact arena geometry with calibration lineage before "
            "enabling this analysis; root attrs and inferred canvas geometry are "
            "forbidden recovery paths."
        )

    def require_stimulus_protocol_authority(self, semantic_label: str) -> NoReturn:
        """Fail until the requested protocol-derived semantics are sealed."""

        label = str(semantic_label).strip()
        if not label:
            _fail("Stimulus protocol authority requires one explicit semantic label.")
        _fail(
            f"Canonical chaser-distance publication does not yet seal protocol-derived "
            f"semantics for {label!r}; this consumer must fail closed. Remediation: "
            "publish the exact protocol fields and source-run lineage in a canonical "
            "authority before enabling this analysis; raw protocol_json, stimulus attrs, "
            "and inferred defaults are forbidden recovery paths."
        )

    def require_derived_surface_authority(self, relative_path: str) -> NoReturn:
        """Reject an unsealed derived component or visualization before navigation.

        The core publication seal currently protects only the logical arrays copied
        by :func:`load_chaser_distance_run`.  A child group merely living beneath a
        verified run does not make that derived payload authoritative.
        """

        if not isinstance(relative_path, str):
            _fail("Derived chaser-distance surface path must be one exact relative path.")
        normalized = "/".join(
            part for part in relative_path.strip("/").split("/") if part
        )
        if (
            not normalized
            or normalized in {".", ".."}
            or any(part in {".", ".."} for part in normalized.split("/"))
        ):
            _fail("Derived chaser-distance surface path must be one exact relative path.")
        _fail(
            f"Derived chaser-distance surface {normalized!r} under {self.run_path!r} "
            "is unavailable because it has no independently verified canonical "
            "publication seal. Remediation: republish that component or artifact "
            "with payload-bound semantic authority before normal use; raw child "
            "navigation and latest/sorted fallback are forbidden."
        )

    @property
    def attrs(self) -> None:
        """Reject accidental fallback to historical raw run attributes."""

        _fail(
            "This consumer still expects raw chaser-distance attrs; adopt the "
            "typed ChaserDistanceReadSnapshot fields before normal use."
        )

    def __getitem__(self, _key: Any) -> None:
        _fail(
            "This consumer still indexes raw chaser-distance Zarr groups; adopt "
            "the detached ChaserDistanceReadSnapshot arrays before normal use."
        )

    def get(self, _key: Any, _default: Any = None) -> None:
        _fail(
            "This consumer still navigates raw chaser-distance Zarr groups; adopt "
            "the detached ChaserDistanceReadSnapshot fields before normal use."
        )


def _deep_plain_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_plain_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_plain_copy(item) for item in value]
    if isinstance(value, list):
        return [_deep_plain_copy(item) for item in value]
    return value


def load_chaser_distance_run(
    root: Any,
    *,
    run_name: str = "latest",
) -> ChaserDistanceReadSnapshot:
    """Load one exact canonical chaser-distance run as detached logical tables."""

    resolved_name, run_path = resolve_chaser_distance_run_path(
        root,
        run_name=run_name,
    )
    bound = _load_verified_binding(root, run_path)
    run_group = root[run_path]
    seal = bound.publication_seal.record
    protected = _payload_mapping(seal, "protected_arrays")

    def protected_array(relative_path: str) -> np.ndarray:
        if relative_path not in protected:
            _fail(f"Publication seal does not protect {relative_path!r}.")
        return _copy_exact_payload(
            run_group,
            run_path=run_path,
            relative_path=relative_path,
            payload=protected[relative_path],
        )

    arrays = {
        name: protected_array(path)
        for name, path in {
            "stimulus_state_key": "stimulus_state_key",
            "camera_frame_id": "frames/camera_frame_id",
            "stimulus_frame_num": "frames/stimulus_frame_num",
            "timestamp_ns": "frames/timestamp_ns",
            "stimulus_epoch_window_id": "frames/stimulus_epoch_window_id",
            "source_detection_row_index": "positions/source_detection_row_index",
            "fish_centroid_img_xy": "positions/fish_centroid_img_xy",
            "fish_centroid_arena_xy": "positions/fish_centroid_arena_xy",
            "chaser_arena_xy": "positions/chaser_arena_xy",
            "fish_valid": "positions/fish_valid",
            "chaser_valid": "positions/chaser_valid",
            "distance_px": "distances/distance_px",
            "distance_mm": "distances/distance_mm",
            "nearest_chaser_index": "distances/nearest_chaser_index",
            "nearest_distance_mm": "distances/nearest_distance_mm",
            "chaser_index": "chasers/chaser_index",
            "stimulus_instance_id_bytes": "chasers/stimulus_instance_id_bytes",
            "source_track_key_bytes": "chasers/source_track_key_bytes",
        }.items()
    }

    epoch_record = bound.epoch_window_identity.record
    published_epoch = _payload_mapping(epoch_record, "published_arrays")
    epoch_paths = {
        "epoch_window_id": "epoch_summary/window_id",
        "epoch_label_bytes": "epoch_summary/label_bytes",
        "epoch_start_frame": "epoch_summary/start_frame",
        "epoch_end_frame": "epoch_summary/end_frame",
    }
    for name, relative_path in epoch_paths.items():
        leaf = relative_path.rsplit("/", 1)[-1]
        if leaf not in published_epoch:
            _fail(f"Epoch-window authority does not protect {relative_path!r}.")
        arrays[name] = _copy_exact_payload(
            run_group,
            run_path=run_path,
            relative_path=relative_path,
            payload=published_epoch[leaf],
        )

    arena = _validated_arena_descriptor(bound)
    camera = bound.coordinate_surfaces[
        "positions/fish_centroid_img_xy"
    ].descriptor
    input_authority = bound.input_authority.record
    measurement_authority = bound.measurement_authority.record
    fps = input_authority.get("fps")
    total_frames = input_authority.get("total_frames")
    pixels_per_mm = measurement_authority.get("pixels_per_mm_projector")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not np.isfinite(float(fps))
        or float(fps) <= 0
        or isinstance(total_frames, bool)
        or not isinstance(total_frames, int)
        or total_frames < 0
        or isinstance(pixels_per_mm, bool)
        or not isinstance(pixels_per_mm, (int, float))
        or not np.isfinite(float(pixels_per_mm))
        or float(pixels_per_mm) <= 0
        or float(pixels_per_mm) != float(input_authority.get("pixels_per_mm_projector"))
    ):
        _fail("Typed chaser-distance timing/scale authority is invalid.")
    if arrays["fish_centroid_arena_xy"].shape[0] != total_frames:
        _fail("Typed total_frames disagrees with the sealed camera-frame axis.")

    source_epoch_path = epoch_record.get("source_run_path")
    if source_epoch_path is not None and not isinstance(source_epoch_path, str):
        _fail("Epoch-window authority has an invalid source run path.")
    source_epoch_run = (
        source_epoch_path.rsplit("/", 1)[-1]
        if isinstance(source_epoch_path, str)
        else None
    )
    coordinate_digests = MappingProxyType(
        {
            path: value.descriptor.digest()
            for path, value in bound.coordinate_surfaces.items()
        }
    )
    measurement_digests = MappingProxyType(
        {
            path: value.record_sha256
            for path, value in bound.measurement_surfaces.items()
        }
    )
    authority_payload = MappingProxyType(
        {
            "schema_id": CHASER_DISTANCE_READ_AUTHORITY_SCHEMA_ID,
            "schema_version": CHASER_DISTANCE_READ_AUTHORITY_SCHEMA_VERSION,
            "run_ref": f"/{run_path}",
            "publication_seal": {
                "record_ref": bound.publication_seal.record_ref,
                "record_sha256": bound.publication_seal.record_sha256,
            },
            "surface_manifest": {
                "record_ref": bound.surface_manifest.record_ref,
                "record_sha256": bound.surface_manifest.record_sha256,
            },
            "row_identity": {
                "record_ref": bound.row_identity.record_ref,
                "record_sha256": bound.row_identity.record_sha256,
            },
            "input_authority": {
                "record_ref": bound.input_authority.record_ref,
                "record_sha256": bound.input_authority.record_sha256,
            },
            "measurement_authority": {
                "record_ref": bound.measurement_authority.record_ref,
                "record_sha256": bound.measurement_authority.record_sha256,
            },
            "chaser_collection": {
                "record_ref": bound.chaser_collection.record_ref,
                "record_sha256": bound.chaser_collection.record_sha256,
            },
            "epoch_window_identity": {
                "record_ref": bound.epoch_window_identity.record_ref,
                "record_sha256": bound.epoch_window_identity.record_sha256,
            },
            "coordinate_descriptor_sha256": dict(coordinate_digests),
            "measurement_descriptor_sha256": dict(measurement_digests),
        }
    )

    # Re-load every persisted authority after all values were copied.  This
    # rejects concurrent replacement/mutation instead of returning a mixed view.
    fresh = _load_verified_binding(root, run_path)
    _require_same_fresh_binding(bound, fresh)
    if str(run_name).strip() in {"", "latest"}:
        final_name, final_path = resolve_chaser_distance_run_path(
            root,
            run_name="latest",
        )
        if final_name != resolved_name or final_path != run_path:
            _fail("Chaser-distance selector changed while the run was being copied.")

    return ChaserDistanceReadSnapshot(
        run_name=resolved_name,
        run_path=run_path,
        recording_id=(
            bound.source_context.detection.frame_evidence.acquisition_frame.record.recording_id
        ),
        authority_status=VERIFIED_AUTHORITY_STATUS,
        archive_identity=bound.source_context.archive_identity,
        source_detection_path=str(input_authority["source_detection_path"]),
        source_stimulus_run=bound.source_context.stimulus_run,
        source_stimulus_path=bound.source_context.stimulus_path,
        source_stimulus_epoch_run=source_epoch_run,
        source_stimulus_epoch_path=source_epoch_path,
        fps=float(fps),
        total_frames=int(total_frames),
        pixels_per_mm_projector=float(pixels_per_mm),
        coordinate_space_id=arena.space_id,
        coordinate_origin=arena.origin,
        positive_x=arena.positive_directions.x,
        positive_y=arena.positive_directions.y,
        reference_width_px=int(arena.reference_extent.width),
        reference_height_px=int(arena.reference_extent.height),
        pixel_convention=arena.pixel_convention,
        arena_coordinate_descriptor=arena,
        source_camera_coordinate_descriptor=camera,
        coordinate_descriptor_sha256=coordinate_digests,
        measurement_descriptor_sha256=measurement_digests,
        behavior_authority_status=UNAVAILABLE_BEHAVIOR_AUTHORITY_STATUS,
        publication_seal_ref=bound.publication_seal.record_ref,
        publication_seal_sha256=bound.publication_seal.record_sha256,
        surface_manifest_ref=bound.surface_manifest.record_ref,
        surface_manifest_sha256=bound.surface_manifest.record_sha256,
        row_identity_ref=bound.row_identity.record_ref,
        row_identity_sha256=bound.row_identity.record_sha256,
        stimulus_instance_ids=_decode_sealed_text_rows(
            arrays.pop("stimulus_instance_id_bytes"),
            label="chasers/stimulus_instance_id_bytes",
        ),
        source_track_keys=_decode_sealed_text_rows(
            arrays.pop("source_track_key_bytes"),
            label="chasers/source_track_key_bytes",
        ),
        epoch_labels=_decode_sealed_text_rows(
            arrays["epoch_label_bytes"],
            label="epoch_summary/label_bytes",
        ),
        _authority_record_payload=authority_payload,
        _verification_seal=_SNAPSHOT_SEAL,
        **arrays,
    )


def reject_unsealed_chaser_derived_publication(
    root: Any,
    *,
    run_name: str,
    run_path: str,
    relative_path: str,
) -> NoReturn:
    """Preflight one exact base run, then reject an unsealed derived write.

    Derived writers call this before creating a group or updating a selector.  It
    deliberately centralizes the temporary publication quarantine so no writer can
    turn an unsealed payload into a newly selectable ``latest_complete`` component.
    """

    snapshot = load_chaser_distance_run(root, run_name=run_name)
    if snapshot.run_path != str(run_path).strip("/"):
        _fail(
            "Chaser derived-publication input changed between computation and write; "
            "the exact verified base run no longer matches the result lineage."
        )
    try:
        snapshot.require_derived_surface_authority(relative_path)
    except ChaserDistanceReadError as exc:
        _fail(
            f"Publication of derived chaser-distance surface {relative_path!r} is "
            "disabled until that component has an independently verified, "
            "payload-bound publication seal. No group or selector was mutated. "
            f"Base preflight: {exc}"
        )
    _fail("Unreachable derived-publication authority state.")


__all__ = [
    "CHASER_DISTANCE_READ_AUTHORITY_SCHEMA_ID",
    "CHASER_DISTANCE_READ_AUTHORITY_SCHEMA_VERSION",
    "ChaserDistanceReadError",
    "ChaserDistanceReadSnapshot",
    "UNAVAILABLE_BEHAVIOR_AUTHORITY_STATUS",
    "VERIFIED_AUTHORITY_STATUS",
    "load_chaser_distance_run",
    "reject_unsealed_chaser_derived_publication",
    "resolve_chaser_distance_run_path",
]
