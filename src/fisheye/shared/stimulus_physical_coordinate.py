"""Exact selected-stimulus authority for source-camera millimetres.

The selected calibration snapshot is the only normal source of camera scale.
This module bridges that persisted snapshot to the sealed physical-frame types
used by coordinate descriptors.  It deliberately has no root-calibration,
``pixel_to_mm``-attribute, resolution-ratio, or projector-scale fallback.

One authority is owned by one exact stimulus run and selected camera.  Every
load reopens the selected-calibration snapshot and acquisition camera frame,
then rebinds the source-camera pixel frame, selected-camera evidence, physical
frame, and their digest-bound manifest in the same archive.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

from fisheye.shared.archive_identity import (
    ArchiveIdentity,
    ArchiveIdentityError,
    archive_identity,
    require_same_archive,
)
from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_PUBLISHED,
    ACQUISITION_AUTHORITY_STATUS_ATTR,
    AcquisitionPublicationStatusError,
    load_acquisition_authority_publication_status,
)
from fisheye.shared.coordinate_frame_record import (
    FRAME_RECORD_DIGEST_SUFFIX,
    PHYSICAL_FRAME_CALIBRATION_ATTR,
    REFERENCE_EXTENT_FINITE,
    SELECTED_CAMERA_FRAME_EVIDENCE_ATTR,
    BoundPhysicalFrameCalibration,
    BoundSelectedCameraFrameEvidence,
    build_physical_frame_calibration_record,
    load_bound_physical_frame_calibration,
    load_bound_selected_camera_frame_evidence,
    parse_selected_camera_frame_evidence_record,
    stamp_physical_frame_calibration_record,
    verify_bound_coordinate_frame,
)
from fisheye.shared.coordinate_record import (
    BoundCoordinateRecord,
    bind_persisted_coordinate_record,
    coordinate_record_sha256,
    stamp_and_bind_persisted_coordinate_record,
)
from fisheye.shared.coordinate_reference import canonical_node_path
from fisheye.shared.directed_transform import TransformReferenceExtent
from fisheye.shared.pixel_frame_authority import (
    BoundAcquisitionCameraFrame,
    BoundPixelFrameAuthority,
    PixelFrameAuthorityError,
    load_persisted_acquisition_camera_authority,
    load_source_camera_pixel_frame_authority,
    stamp_source_camera_pixel_frame_authority,
)
from fisheye.shared.selected_calibration import (
    CANONICAL_HOMOGRAPHY_FROM_SPACE_ID,
    CANONICAL_HOMOGRAPHY_TO_SPACE_ID,
    SelectedCalibrationSnapshot,
    load_selected_calibration_manifest_attrs,
    load_selected_calibration_snapshot,
    require_bound_selected_calibration_snapshot,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_RUNNING,
)


STIMULUS_PHYSICAL_COORDINATE_MANIFEST_ATTR = (
    "stimulus_physical_coordinate_manifest"
)
STIMULUS_PHYSICAL_COORDINATE_MANIFEST_DIGEST_ATTR = (
    f"{STIMULUS_PHYSICAL_COORDINATE_MANIFEST_ATTR}_sha256"
)
STIMULUS_PHYSICAL_COORDINATE_MANIFEST_SCHEMA_ID = (
    "palette.stimulus_physical_coordinate_authority"
)
STIMULUS_PHYSICAL_COORDINATE_MANIFEST_SCHEMA_VERSION = 1

STIMULUS_PHYSICAL_COORDINATE_STATUS_ATTR = (
    "physical_coordinate_publication_status"
)
STIMULUS_PHYSICAL_COORDINATE_MANIFEST_REF_ATTR = (
    "physical_coordinate_manifest_ref"
)
STIMULUS_PHYSICAL_COORDINATE_MANIFEST_SHA256_ATTR = (
    "physical_coordinate_manifest_sha256"
)
STIMULUS_PHYSICAL_COORDINATE_REASON_CODE_ATTR = (
    "physical_coordinate_reason_code"
)
STIMULUS_PHYSICAL_COORDINATE_REASON_BOUND = "NONE"
STIMULUS_PHYSICAL_COORDINATE_REASON_NO_SCALE = (
    "SELECTED_CAMERA_PIXELS_PER_MM_MISSING"
)
STIMULUS_PHYSICAL_COORDINATE_REASON_NO_ACQUISITION = (
    "ACQUISITION_CAMERA_AUTHORITY_ABSENT"
)
STIMULUS_PHYSICAL_COORDINATE_BOUND_STATUS = (
    "bound_typed_source_camera_mm_v1"
)
STIMULUS_PHYSICAL_COORDINATE_OMITTED_NO_SCALE_STATUS = (
    "omitted_no_selected_camera_pixels_per_mm_v1"
)
STIMULUS_PHYSICAL_COORDINATE_OMITTED_NO_ACQUISITION_STATUS = (
    "omitted_no_acquisition_camera_authority_v1"
)
STIMULUS_PHYSICAL_COORDINATE_OMITTED_STATUSES = frozenset(
    {
        STIMULUS_PHYSICAL_COORDINATE_OMITTED_NO_SCALE_STATUS,
        STIMULUS_PHYSICAL_COORDINATE_OMITTED_NO_ACQUISITION_STATUS,
    }
)
STIMULUS_PHYSICAL_COORDINATE_INVALIDATED_STATUS = (
    "invalidated_parent_stimulus_run_failed_v1"
)
STIMULUS_PHYSICAL_COORDINATE_REASON_PARENT_RUN_FAILED = (
    "STIMULUS_RUN_PUBLICATION_FAILED"
)

_PIXEL_CONVENTION = "continuous"
_CAMERA_FRAME_CONTAINER = "coordinate_frames"
_SELECTED_CAMERA_EVIDENCE_NODE = "selected_camera_evidence"
_PHYSICAL_FRAME_NODE = "source_camera_physical_mm"
_BOUND_STIMULUS_PHYSICAL_COORDINATE_AUTHORITY_SEAL = object()


class StimulusPhysicalCoordinateError(ValueError):
    """Raised when selected physical-coordinate authority is inconsistent."""


class StimulusPhysicalCoordinateUnavailableError(
    StimulusPhysicalCoordinateError
):
    """Raised when an older run has no published physical authority at all."""


@dataclass(frozen=True, init=False)
class BoundStimulusPhysicalCoordinateAuthority:
    """Freshly rebound physical authority for one selected stimulus camera."""

    stimulus_run: str
    camera_id: str
    archive_identity: ArchiveIdentity
    selected_calibration: SelectedCalibrationSnapshot = field(
        repr=False,
        compare=False,
    )
    acquisition_frame: BoundAcquisitionCameraFrame = field(
        repr=False,
        compare=False,
    )
    source_camera_frame: BoundPixelFrameAuthority = field(
        repr=False,
        compare=False,
    )
    selected_camera_evidence: BoundSelectedCameraFrameEvidence = field(
        repr=False,
        compare=False,
    )
    physical_frame: BoundPhysicalFrameCalibration = field(
        repr=False,
        compare=False,
    )
    manifest: BoundCoordinateRecord = field(repr=False, compare=False)
    _root_node: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        stimulus_run: str,
        camera_id: str,
        archive_identity: ArchiveIdentity,
        selected_calibration: SelectedCalibrationSnapshot,
        acquisition_frame: BoundAcquisitionCameraFrame,
        source_camera_frame: BoundPixelFrameAuthority,
        selected_camera_evidence: BoundSelectedCameraFrameEvidence,
        physical_frame: BoundPhysicalFrameCalibration,
        manifest: BoundCoordinateRecord,
        root_node: Any,
        _verification_seal: object | None = None,
    ) -> None:
        if (
            _verification_seal
            is not _BOUND_STIMULUS_PHYSICAL_COORDINATE_AUTHORITY_SEAL
        ):
            raise StimulusPhysicalCoordinateError(
                "Bound stimulus physical-coordinate authority cannot be "
                "constructed directly."
            )
        object.__setattr__(self, "stimulus_run", stimulus_run)
        object.__setattr__(self, "camera_id", camera_id)
        object.__setattr__(self, "archive_identity", archive_identity)
        object.__setattr__(self, "selected_calibration", selected_calibration)
        object.__setattr__(self, "acquisition_frame", acquisition_frame)
        object.__setattr__(self, "source_camera_frame", source_camera_frame)
        object.__setattr__(
            self,
            "selected_camera_evidence",
            selected_camera_evidence,
        )
        object.__setattr__(self, "physical_frame", physical_frame)
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "_root_node", root_node)
        object.__setattr__(self, "_seal", _verification_seal)

    @property
    def mm_per_pixel(self) -> float:
        return float(self.physical_frame.record.mm_per_pixel)

    def assert_verified(self) -> None:
        if (
            getattr(self, "_seal", None)
            is not _BOUND_STIMULUS_PHYSICAL_COORDINATE_AUTHORITY_SEAL
        ):
            raise StimulusPhysicalCoordinateError(
                "Stimulus physical-coordinate authority is not sealed evidence."
            )
        current = load_stimulus_physical_coordinate_authority(
            self._root_node,
            stimulus_run=self.stimulus_run,
        )
        if current is None or (
            current.camera_id != self.camera_id
            or current.archive_identity != self.archive_identity
            or current.manifest.record_ref != self.manifest.record_ref
            or current.manifest.record_sha256 != self.manifest.record_sha256
            or current.physical_frame.record_ref
            != self.physical_frame.record_ref
            or current.physical_frame.record_sha256
            != self.physical_frame.record_sha256
        ):
            raise StimulusPhysicalCoordinateError(
                "Stimulus physical-coordinate authority changed after binding."
            )


def _child(node: Any, path: str) -> Any:
    current = node
    walked: list[str] = []
    for name in path.split("/"):
        if not name:
            continue
        walked.append(name)
        try:
            current = current[name]
        except (KeyError, TypeError, AttributeError) as exc:
            raise StimulusPhysicalCoordinateError(
                f"Required persisted node /{'/'.join(walked)} is missing."
            ) from exc
    return current


def _has_child(node: Any, path: str) -> bool:
    try:
        _child(node, path)
    except StimulusPhysicalCoordinateError:
        return False
    return True


def _acquisition_publication_is_truly_absent(root_node: Any) -> bool:
    """Distinguish a historical absence from partial/statusless publication."""

    root_attrs = getattr(root_node, "attrs", None)
    if not isinstance(root_attrs, Mapping):
        raise StimulusPhysicalCoordinateError(
            "Archive root does not expose acquisition publication attrs."
        )
    root_has_status = ACQUISITION_AUTHORITY_STATUS_ATTR in root_attrs
    raw_has_status = False
    if _has_child(root_node, "raw_video"):
        raw_video = _child(root_node, "raw_video")
        raw_attrs = getattr(raw_video, "attrs", None)
        if not isinstance(raw_attrs, Mapping):
            raise StimulusPhysicalCoordinateError(
                "raw_video does not expose acquisition publication attrs."
            )
        raw_has_status = ACQUISITION_AUTHORITY_STATUS_ATTR in raw_attrs
    has_container = _has_child(
        root_node,
        "analysis/acquisition_camera_frames",
    )
    return not (root_has_status or raw_has_status or has_container)


def _load_published_acquisition_camera_authority(
    root_node: Any,
    *,
    expected_camera_id: str,
) -> tuple[Any, BoundAcquisitionCameraFrame]:
    """Load exact acquisition evidence only through its published commit marker."""

    try:
        status = load_acquisition_authority_publication_status(root_node)
    except AcquisitionPublicationStatusError as exc:
        raise StimulusPhysicalCoordinateError(
            "Acquisition camera authority lacks exact mirrored typed publication "
            f"status: {exc}."
        ) from exc
    if status.status != ACQUISITION_AUTHORITY_PUBLISHED:
        raise StimulusPhysicalCoordinateError(
            "Source-camera physical authority requires a published acquisition "
            f"authority; found status={status.status!r}, "
            f"reason={status.reason_code!r}."
        )
    try:
        ownership, acquisition = load_persisted_acquisition_camera_authority(
            root_node,
            expected_camera_id=expected_camera_id,
        )
    except PixelFrameAuthorityError as exc:
        raise StimulusPhysicalCoordinateError(
            f"Published acquisition camera authority is invalid: {exc}."
        ) from exc
    expected_path = (
        "analysis/acquisition_camera_frames/"
        f"{acquisition.record.camera_id}"
    )
    if (
        status.authority_mode != ownership.record.mode
        or status.authority_path != expected_path
    ):
        raise StimulusPhysicalCoordinateError(
            "Published acquisition status mode/path disagrees with the exact "
            "persisted acquisition ownership and camera frame."
        )
    try:
        ownership.assert_verified()
        acquisition.assert_verified()
    except PixelFrameAuthorityError as exc:
        raise StimulusPhysicalCoordinateError(
            f"Published acquisition camera authority failed fresh verification: {exc}."
        ) from exc
    return ownership, acquisition


def _require_path(node: Any, expected: str, *, label: str) -> None:
    try:
        actual = canonical_node_path(node)
    except Exception as exc:
        raise StimulusPhysicalCoordinateError(
            f"{label} has no canonical persisted path: {exc}."
        ) from exc
    if actual != expected:
        raise StimulusPhysicalCoordinateError(
            f"{label} path /{actual} differs from exact expected /{expected}."
        )


def _attrs(node: Any, *, label: str) -> Any:
    attrs = getattr(node, "attrs", None)
    if not isinstance(attrs, Mapping) or not all(
        callable(getattr(attrs, name, None))
        for name in ("update", "__setitem__", "__delitem__")
    ):
        raise StimulusPhysicalCoordinateError(
            f"{label} must expose one mutable attrs transaction boundary."
        )
    return attrs


def _restore_attrs(attrs: Any, snapshot: Mapping[str, Any]) -> None:
    for name in tuple(attrs.keys()):
        del attrs[name]
    attrs.update(copy.deepcopy(dict(snapshot)))
    if dict(attrs) != dict(snapshot):
        raise RuntimeError("Physical-coordinate attrs rollback was not exact.")


def _ensure_group(parent: Any, name: str) -> tuple[Any, bool]:
    try:
        return parent[name], False
    except (KeyError, TypeError, AttributeError):
        pass
    create = getattr(parent, "create_group", None)
    if not callable(create):
        create = getattr(parent, "require_group", None)
    if not callable(create):
        raise StimulusPhysicalCoordinateError(
            f"Cannot create canonical group {name!r} beneath /{canonical_node_path(parent)}."
        )
    try:
        return create(name), True
    except Exception as exc:
        raise StimulusPhysicalCoordinateError(
            f"Unable to create canonical group {name!r}: {exc}."
        ) from exc


def _delete_child(parent: Any, name: str) -> None:
    try:
        del parent[name]
        return
    except Exception:
        children = getattr(parent, "children", None)
        if isinstance(children, dict) and name in children:
            del children[name]
            return
        raise


def _pointer(record_ref: str, record_sha256: str) -> dict[str, str]:
    return {
        "record_ref": str(record_ref),
        "record_sha256": str(record_sha256),
    }


def _fresh_selected_snapshot(
    root_node: Any,
    *,
    stimulus_run: str,
    expected_snapshot: SelectedCalibrationSnapshot | None = None,
) -> SelectedCalibrationSnapshot:
    run_path = f"analysis/stimulus_runs/{stimulus_run}"
    run_group = _child(root_node, run_path)
    _require_path(run_group, run_path, label="Stimulus run")
    calibration = _child(run_group, "calibration")
    attrs = getattr(calibration, "attrs", None)
    if not isinstance(attrs, Mapping):
        raise StimulusPhysicalCoordinateError(
            "Selected calibration group does not expose persisted attrs."
        )
    try:
        manifest = load_selected_calibration_manifest_attrs(attrs)
        camera = manifest.camera_calibration
        display = manifest.display_snapshot
        source_extent = TransformReferenceExtent(
            width=camera.native_width_px,
            height=camera.native_height_px,
            units="px",
            authority=(
                f"{manifest.camera_calibration_ref}"
                "@native_width_px,native_height_px"
            ),
        )
        target_extent = TransformReferenceExtent(
            width=display.width_px,
            height=display.height_px,
            units="px",
            authority=(
                f"analysis/stimulus_runs/{stimulus_run}/display_snapshot"
                "@selected_output_geometry"
            ),
        )
        selected = load_selected_calibration_snapshot(
            root_node,
            stimulus_run=stimulus_run,
            expected_camera_id=manifest.camera_id,
            expected_from_space_id=CANONICAL_HOMOGRAPHY_FROM_SPACE_ID,
            expected_to_space_id=CANONICAL_HOMOGRAPHY_TO_SPACE_ID,
            expected_source_reference_extent=source_extent,
            expected_target_reference_extent=target_extent,
        )
        selected = require_bound_selected_calibration_snapshot(selected)
    except Exception as exc:
        raise StimulusPhysicalCoordinateError(
            f"Selected calibration snapshot cannot be freshly rebound: {exc}."
        ) from exc
    if expected_snapshot is not None:
        expected = require_bound_selected_calibration_snapshot(expected_snapshot)
        if (
            expected.stimulus_run != selected.stimulus_run
            or expected.camera_id != selected.camera_id
            or expected.manifest != selected.manifest
            or expected.manifest_sha256 != selected.manifest_sha256
            or expected.archive_identity != selected.archive_identity
        ):
            raise StimulusPhysicalCoordinateError(
                "Fresh selected calibration differs from the supplied persisted snapshot."
            )
    return selected


def _selected_camera_record(
    selected: SelectedCalibrationSnapshot,
) -> Any:
    source = selected.manifest.source_camera
    if source.pixels_per_mm_camera is None:
        raise StimulusPhysicalCoordinateError(
            "Selected camera has no pixels_per_mm_camera evidence."
        )
    payload = source.to_dict()
    return parse_selected_camera_frame_evidence_record(
        {
            "schema_id": "palette.selected_camera_frame_evidence",
            "schema_version": 1,
            "source_camera": payload,
            "source_camera_sha256": coordinate_record_sha256(payload),
            "camera_id": source.active_camera_id,
            "native_width_px": source.native_width_px,
            "native_height_px": source.native_height_px,
            "pixels_per_mm_camera_selector": (
                "/selected_camera_record/pixels_per_mm_camera"
            ),
            "pixels_per_mm_camera": source.pixels_per_mm_camera,
        }
    )


def _manifest_record(
    *,
    selected: SelectedCalibrationSnapshot,
    acquisition: BoundAcquisitionCameraFrame,
    source_camera: BoundPixelFrameAuthority,
    selected_evidence: BoundSelectedCameraFrameEvidence,
    physical: BoundPhysicalFrameCalibration,
) -> dict[str, Any]:
    return {
        "schema_id": STIMULUS_PHYSICAL_COORDINATE_MANIFEST_SCHEMA_ID,
        "schema_version": STIMULUS_PHYSICAL_COORDINATE_MANIFEST_SCHEMA_VERSION,
        "stimulus_run": selected.stimulus_run,
        "camera_id": selected.camera_id,
        "selected_calibration": _pointer(
            selected.manifest_record_ref,
            selected.manifest_sha256,
        ),
        "selected_camera_source_evidence": _pointer(
            selected.camera_record_ref,
            coordinate_record_sha256(selected.manifest.source_camera.to_dict()),
        ),
        "acquisition_camera_frame": _pointer(
            acquisition.record_ref,
            acquisition.record_sha256,
        ),
        "source_camera_pixel_frame": _pointer(
            source_camera.record_ref,
            source_camera.record_sha256,
        ),
        "selected_camera_frame_evidence": _pointer(
            selected_evidence.record_ref,
            selected_evidence.record_sha256,
        ),
        "physical_frame": _pointer(
            physical.record_ref,
            physical.record_sha256,
        ),
        "scale": {
            "quantity": "mm_per_pixel",
            "value": float(physical.record.mm_per_pixel),
            "derivation": (
                "exact_binary64_reciprocal_of_selected_pixels_per_mm_camera_v1"
            ),
        },
    }


def _set_omission_status(run_group: Any, status: str) -> None:
    if status not in STIMULUS_PHYSICAL_COORDINATE_OMITTED_STATUSES:
        raise StimulusPhysicalCoordinateError(
            "Unsupported physical-coordinate omission status."
        )
    _assert_no_run_physical_records(run_group)
    reason = {
        STIMULUS_PHYSICAL_COORDINATE_OMITTED_NO_SCALE_STATUS: (
            STIMULUS_PHYSICAL_COORDINATE_REASON_NO_SCALE
        ),
        STIMULUS_PHYSICAL_COORDINATE_OMITTED_NO_ACQUISITION_STATUS: (
            STIMULUS_PHYSICAL_COORDINATE_REASON_NO_ACQUISITION
        ),
    }[status]
    attrs = _attrs(run_group, label="Stimulus run")
    snapshot = copy.deepcopy(dict(attrs))
    try:
        attrs[STIMULUS_PHYSICAL_COORDINATE_STATUS_ATTR] = status
        attrs[STIMULUS_PHYSICAL_COORDINATE_REASON_CODE_ATTR] = reason
        for name in (
            STIMULUS_PHYSICAL_COORDINATE_MANIFEST_REF_ATTR,
            STIMULUS_PHYSICAL_COORDINATE_MANIFEST_SHA256_ATTR,
        ):
            if name in attrs:
                del attrs[name]
    except Exception as exc:
        _restore_attrs(attrs, snapshot)
        raise StimulusPhysicalCoordinateError(
            f"Unable to persist physical-coordinate omission status: {exc}."
        ) from exc


def invalidate_stimulus_physical_coordinate_publication(run_group: Any) -> None:
    """Fail closed after a parent stimulus-run publication failure.

    Run-local frame records may remain useful forensic evidence on a failed run,
    but no loader may mistake them for a published physical authority.  The
    invalidation marker is deliberately incompatible with every loadable status,
    and manifest pointers are removed transactionally.
    """

    attrs = _attrs(run_group, label="Stimulus run")
    if STIMULUS_PHYSICAL_COORDINATE_STATUS_ATTR not in attrs:
        return
    snapshot = copy.deepcopy(dict(attrs))
    try:
        attrs[STIMULUS_PHYSICAL_COORDINATE_STATUS_ATTR] = (
            STIMULUS_PHYSICAL_COORDINATE_INVALIDATED_STATUS
        )
        attrs[STIMULUS_PHYSICAL_COORDINATE_REASON_CODE_ATTR] = (
            STIMULUS_PHYSICAL_COORDINATE_REASON_PARENT_RUN_FAILED
        )
        for name in (
            STIMULUS_PHYSICAL_COORDINATE_MANIFEST_REF_ATTR,
            STIMULUS_PHYSICAL_COORDINATE_MANIFEST_SHA256_ATTR,
        ):
            if name in attrs:
                del attrs[name]
    except Exception as exc:
        _restore_attrs(attrs, snapshot)
        raise StimulusPhysicalCoordinateError(
            f"Unable to invalidate failed stimulus physical authority: {exc}."
        ) from exc


def _require_parent_run_status(run_group: Any, *, expected: str) -> None:
    attrs = getattr(run_group, "attrs", None)
    if not isinstance(attrs, Mapping):
        raise StimulusPhysicalCoordinateError(
            "Stimulus run does not expose persisted completion attrs."
        )
    observed = attrs.get(RUN_COMPLETION_STATUS_ATTR)
    if observed != expected:
        raise StimulusPhysicalCoordinateError(
            "Stimulus physical-coordinate authority requires parent run status "
            f"{expected!r}; found {observed!r}."
        )


def _assert_no_run_physical_records(run_group: Any) -> None:
    """Prove that an omission state does not hide stale physical authority."""

    calibration = _child(run_group, "calibration")
    try:
        manifest = load_selected_calibration_manifest_attrs(calibration.attrs)
    except Exception as exc:
        raise StimulusPhysicalCoordinateError(
            f"Cannot validate physical omission against selected camera: {exc}."
        ) from exc
    camera = _child(calibration, manifest.camera_id)
    if not _has_child(camera, _CAMERA_FRAME_CONTAINER):
        return
    frame_container = _child(camera, _CAMERA_FRAME_CONTAINER)
    attrs = getattr(frame_container, "attrs", None)
    if not isinstance(attrs, Mapping):
        raise StimulusPhysicalCoordinateError(
            "Physical omission found an invalid coordinate-frame container."
        )
    forbidden_attrs = {
        STIMULUS_PHYSICAL_COORDINATE_MANIFEST_ATTR,
        STIMULUS_PHYSICAL_COORDINATE_MANIFEST_DIGEST_ATTR,
    } & set(attrs)
    forbidden_children = [
        name
        for name in (
            _SELECTED_CAMERA_EVIDENCE_NODE,
            _PHYSICAL_FRAME_NODE,
        )
        if _has_child(frame_container, name)
    ]
    if forbidden_attrs or forbidden_children:
        raise StimulusPhysicalCoordinateError(
            "Physical omission conflicts with stale/partial run-local physical "
            f"authority (attrs={sorted(forbidden_attrs)!r}, "
            f"nodes={forbidden_children!r})."
        )


def publish_stimulus_physical_coordinate_authority(
    root_node: Any,
    run_group: Any,
    *,
    stimulus_run: str,
    selected_calibration: SelectedCalibrationSnapshot,
) -> BoundStimulusPhysicalCoordinateAuthority | None:
    """Publish and freshly reload one run/camera physical authority.

    A run with no selected camera scale, or an archive with no acquisition
    authority container at all, receives an explicit omission status.  A
    present but malformed acquisition or coordinate authority fails closed.
    """

    expected_run_path = f"analysis/stimulus_runs/{stimulus_run}"
    _require_path(run_group, expected_run_path, label="Stimulus run")
    _require_parent_run_status(run_group, expected=RUN_STATUS_RUNNING)
    if archive_identity(root_node) != archive_identity(run_group):
        raise StimulusPhysicalCoordinateError(
            "Stimulus run and selected physical authority must share one archive."
        )
    selected = _fresh_selected_snapshot(
        root_node,
        stimulus_run=stimulus_run,
        expected_snapshot=selected_calibration,
    )
    if selected.pixels_per_mm_camera is None:
        _set_omission_status(
            run_group,
            STIMULUS_PHYSICAL_COORDINATE_OMITTED_NO_SCALE_STATUS,
        )
        return None
    if _acquisition_publication_is_truly_absent(root_node):
        _set_omission_status(
            run_group,
            STIMULUS_PHYSICAL_COORDINATE_OMITTED_NO_ACQUISITION_STATUS,
        )
        return None

    _, acquisition = _load_published_acquisition_camera_authority(
        root_node,
        expected_camera_id=selected.camera_id,
    )

    created: list[tuple[Any, str]] = []
    attrs_snapshots: list[tuple[Any, dict[str, Any]]] = []
    try:
        analysis = _child(root_node, "analysis")
        coordinate_frames, was_created = _ensure_group(
            analysis,
            "coordinate_frames",
        )
        if was_created:
            created.append((analysis, "coordinate_frames"))
        source_container, was_created = _ensure_group(
            coordinate_frames,
            "source_camera",
        )
        if was_created:
            created.append((coordinate_frames, "source_camera"))
        camera_container, was_created = _ensure_group(
            source_container,
            selected.camera_id,
        )
        if was_created:
            created.append((source_container, selected.camera_id))
        source_node, was_created = _ensure_group(
            camera_container,
            _PIXEL_CONVENTION,
        )
        if was_created:
            created.append((camera_container, _PIXEL_CONVENTION))
        source_attrs = _attrs(
            source_node,
            label=f"/{canonical_node_path(source_node)}",
        )
        empty_existing_placeholder = not was_created and not dict(source_attrs)
        if empty_existing_placeholder:
            # Earlier interrupted publishers could leave an empty group after
            # rolling back its attrs.  An empty node carries no authority and
            # is safe to initialize; any partial/non-empty attrs still fail
            # closed through the normal loader below.
            attrs_snapshots.append((source_attrs, {}))
        if was_created or empty_existing_placeholder:
            source_camera = stamp_source_camera_pixel_frame_authority(
                source_node,
                frame_id=f"{selected.camera_id}_source_camera",
                pixel_convention=_PIXEL_CONVENTION,
                acquisition_frame=acquisition,
            )
        else:
            source_camera = load_source_camera_pixel_frame_authority(
                source_node,
                acquisition_frame=acquisition,
            )

        camera_group = _child(
            run_group,
            f"calibration/{selected.camera_id}",
        )
        frame_container, was_created = _ensure_group(
            camera_group,
            _CAMERA_FRAME_CONTAINER,
        )
        if was_created:
            created.append((camera_group, _CAMERA_FRAME_CONTAINER))
        selected_node, was_created = _ensure_group(
            frame_container,
            _SELECTED_CAMERA_EVIDENCE_NODE,
        )
        if was_created:
            created.append((frame_container, _SELECTED_CAMERA_EVIDENCE_NODE))
        physical_node, was_created = _ensure_group(
            frame_container,
            _PHYSICAL_FRAME_NODE,
        )
        if was_created:
            created.append((frame_container, _PHYSICAL_FRAME_NODE))

        for node in (run_group, frame_container, selected_node, physical_node):
            attrs = _attrs(node, label=f"/{canonical_node_path(node)}")
            attrs_snapshots.append((attrs, copy.deepcopy(dict(attrs))))

        selected_record = _selected_camera_record(selected)
        selected_generic = stamp_and_bind_persisted_coordinate_record(
            selected_node,
            selected_record.to_dict(),
            attr_name=SELECTED_CAMERA_FRAME_EVIDENCE_ATTR,
            digest_attr_name=(
                f"{SELECTED_CAMERA_FRAME_EVIDENCE_ATTR}"
                f"{FRAME_RECORD_DIGEST_SUFFIX}"
            ),
        )
        selected_evidence = load_bound_selected_camera_frame_evidence(
            selected_node,
            expected_record_ref=selected_generic.record_ref,
            expected_record_sha256=selected_generic.record_sha256,
            expected_camera_id=selected.camera_id,
        )
        physical_record = build_physical_frame_calibration_record(
            frame_id=(
                f"{stimulus_run}_{selected.camera_id}_source_camera_physical_mm"
            ),
            source_camera_pixels=source_camera,
            selected_camera_evidence=selected_evidence,
            physical_extent_mode=REFERENCE_EXTENT_FINITE,
        )
        physical = stamp_physical_frame_calibration_record(
            physical_node,
            physical_record,
            expected_record_ref=(
                f"/{canonical_node_path(physical_node)}"
                f"@{PHYSICAL_FRAME_CALIBRATION_ATTR}"
            ),
            source_camera_pixels=source_camera,
            selected_camera_evidence=selected_evidence,
        )
        manifest_record = _manifest_record(
            selected=selected,
            acquisition=acquisition,
            source_camera=source_camera,
            selected_evidence=selected_evidence,
            physical=physical,
        )
        manifest = stamp_and_bind_persisted_coordinate_record(
            frame_container,
            manifest_record,
            attr_name=STIMULUS_PHYSICAL_COORDINATE_MANIFEST_ATTR,
            digest_attr_name=(
                STIMULUS_PHYSICAL_COORDINATE_MANIFEST_DIGEST_ATTR
            ),
        )
        run_attrs = _attrs(run_group, label="Stimulus run")
        run_attrs.update(
            {
                STIMULUS_PHYSICAL_COORDINATE_STATUS_ATTR: (
                    STIMULUS_PHYSICAL_COORDINATE_BOUND_STATUS
                ),
                STIMULUS_PHYSICAL_COORDINATE_REASON_CODE_ATTR: (
                    STIMULUS_PHYSICAL_COORDINATE_REASON_BOUND
                ),
                STIMULUS_PHYSICAL_COORDINATE_MANIFEST_REF_ATTR: (
                    manifest.record_ref
                ),
                STIMULUS_PHYSICAL_COORDINATE_MANIFEST_SHA256_ATTR: (
                    manifest.record_sha256
                ),
            }
        )
        require_same_archive(
            root_node,
            run_group,
            source_node,
            frame_container,
            selected_node,
            physical_node,
        )
        return _load_stimulus_physical_coordinate_authority(
            root_node,
            stimulus_run=stimulus_run,
            require_complete=False,
            require_selector_eligible=False,
        )
    except BaseException as exc:
        rollback_errors: list[str] = []
        for attrs, snapshot in reversed(attrs_snapshots):
            try:
                _restore_attrs(attrs, snapshot)
            except BaseException as rollback_exc:  # pragma: no cover - hostile store
                rollback_errors.append(str(rollback_exc))
        for parent, name in reversed(created):
            try:
                _delete_child(parent, name)
            except BaseException as rollback_exc:  # pragma: no cover - hostile store
                rollback_errors.append(str(rollback_exc))
        if rollback_errors:
            raise StimulusPhysicalCoordinateError(
                "Physical-coordinate publication failed and rollback was "
                f"incomplete: {rollback_errors!r}."
            ) from exc
        if isinstance(exc, StimulusPhysicalCoordinateError):
            raise
        if not isinstance(exc, Exception):
            raise
        raise StimulusPhysicalCoordinateError(
            f"Physical-coordinate publication failed: {exc}."
        ) from exc


def _load_stimulus_physical_coordinate_authority(
    root_node: Any,
    *,
    stimulus_run: str,
    require_complete: bool,
    require_selector_eligible: bool,
) -> BoundStimulusPhysicalCoordinateAuthority | None:
    """Freshly rebind one exact selected-run source-camera physical frame."""

    run_path = f"analysis/stimulus_runs/{stimulus_run}"
    run_group = _child(root_node, run_path)
    _require_path(run_group, run_path, label="Stimulus run")
    _require_parent_run_status(
        run_group,
        expected=RUN_STATUS_COMPLETE if require_complete else RUN_STATUS_RUNNING,
    )
    if (
        require_selector_eligible
        and run_group.attrs.get("stage_selector_eligible") is not True
    ):
        raise StimulusPhysicalCoordinateError(
            "Stimulus physical-coordinate authority may only be consumed from a "
            "complete, explicitly selector-eligible run."
        )
    selected = _fresh_selected_snapshot(
        root_node,
        stimulus_run=stimulus_run,
    )
    run_attrs = getattr(run_group, "attrs", None)
    if not isinstance(run_attrs, Mapping):
        raise StimulusPhysicalCoordinateError(
            "Stimulus run does not expose persisted attrs."
        )
    status = run_attrs.get(STIMULUS_PHYSICAL_COORDINATE_STATUS_ATTR)
    if status in STIMULUS_PHYSICAL_COORDINATE_OMITTED_STATUSES:
        expected_reason = {
            STIMULUS_PHYSICAL_COORDINATE_OMITTED_NO_SCALE_STATUS: (
                STIMULUS_PHYSICAL_COORDINATE_REASON_NO_SCALE
            ),
            STIMULUS_PHYSICAL_COORDINATE_OMITTED_NO_ACQUISITION_STATUS: (
                STIMULUS_PHYSICAL_COORDINATE_REASON_NO_ACQUISITION
            ),
        }[status]
        if run_attrs.get(STIMULUS_PHYSICAL_COORDINATE_REASON_CODE_ATTR) != (
            expected_reason
        ):
            raise StimulusPhysicalCoordinateError(
                "Physical omission has a missing or mismatched stable reason code."
            )
        if (
            STIMULUS_PHYSICAL_COORDINATE_MANIFEST_REF_ATTR in run_attrs
            or STIMULUS_PHYSICAL_COORDINATE_MANIFEST_SHA256_ATTR in run_attrs
        ):
            raise StimulusPhysicalCoordinateError(
                "Omitted physical authority carries contradictory manifest pointers."
            )
        if (
            status == STIMULUS_PHYSICAL_COORDINATE_OMITTED_NO_SCALE_STATUS
            and selected.pixels_per_mm_camera is not None
        ):
            raise StimulusPhysicalCoordinateError(
                "No-scale omission contradicts the selected calibration snapshot."
            )
        if (
            status
            == STIMULUS_PHYSICAL_COORDINATE_OMITTED_NO_ACQUISITION_STATUS
            and not _acquisition_publication_is_truly_absent(root_node)
        ):
            raise StimulusPhysicalCoordinateError(
                "No-acquisition omission is stale because acquisition publication "
                "status or an authority container is now present."
            )
        _assert_no_run_physical_records(run_group)
        return None
    if status is None:
        raise StimulusPhysicalCoordinateUnavailableError(
            "Stimulus run has no physical-coordinate publication status."
        )
    if status != STIMULUS_PHYSICAL_COORDINATE_BOUND_STATUS:
        raise StimulusPhysicalCoordinateError(
            f"Unsupported physical-coordinate publication status {status!r}."
        )
    if run_attrs.get(STIMULUS_PHYSICAL_COORDINATE_REASON_CODE_ATTR) != (
        STIMULUS_PHYSICAL_COORDINATE_REASON_BOUND
    ):
        raise StimulusPhysicalCoordinateError(
            "Bound physical authority has a missing or mismatched reason code."
        )
    if selected.pixels_per_mm_camera is None:
        raise StimulusPhysicalCoordinateError(
            "Bound physical authority contradicts missing selected camera scale."
        )

    try:
        _, acquisition = _load_published_acquisition_camera_authority(
            root_node,
            expected_camera_id=selected.camera_id,
        )
        source_node = _child(
            root_node,
            "analysis/coordinate_frames/source_camera/"
            f"{selected.camera_id}/{_PIXEL_CONVENTION}",
        )
        source_camera = load_source_camera_pixel_frame_authority(
            source_node,
            acquisition_frame=acquisition,
        )
        frame_container = _child(
            run_group,
            f"calibration/{selected.camera_id}/{_CAMERA_FRAME_CONTAINER}",
        )
        selected_node = _child(
            frame_container,
            _SELECTED_CAMERA_EVIDENCE_NODE,
        )
        physical_node = _child(frame_container, _PHYSICAL_FRAME_NODE)
        manifest = bind_persisted_coordinate_record(
            frame_container,
            attr_name=STIMULUS_PHYSICAL_COORDINATE_MANIFEST_ATTR,
            digest_attr_name=(
                STIMULUS_PHYSICAL_COORDINATE_MANIFEST_DIGEST_ATTR
            ),
        )
        manifest_payload = manifest.record
        selected_pointer = manifest_payload.get("selected_camera_frame_evidence")
        physical_pointer = manifest_payload.get("physical_frame")
        if type(selected_pointer) is not dict or type(physical_pointer) is not dict:
            raise StimulusPhysicalCoordinateError(
                "Physical-coordinate manifest lacks typed frame pointers."
            )
        selected_evidence = load_bound_selected_camera_frame_evidence(
            selected_node,
            expected_record_ref=selected_pointer.get("record_ref"),
            expected_record_sha256=selected_pointer.get("record_sha256"),
            expected_camera_id=selected.camera_id,
        )
        physical = load_bound_physical_frame_calibration(
            physical_node,
            expected_record_ref=physical_pointer.get("record_ref"),
            expected_record_sha256=physical_pointer.get("record_sha256"),
            expected_camera_id=selected.camera_id,
            source_camera_pixels=source_camera,
            selected_camera_evidence=selected_evidence,
        )
        verify_bound_coordinate_frame(
            physical,
            expected_kind="physical_frame_calibration",
        )
        expected_manifest = _manifest_record(
            selected=selected,
            acquisition=acquisition,
            source_camera=source_camera,
            selected_evidence=selected_evidence,
            physical=physical,
        )
        if manifest.record != expected_manifest:
            raise StimulusPhysicalCoordinateError(
                "Physical-coordinate manifest differs from exact live authorities."
            )
        if (
            run_attrs.get(STIMULUS_PHYSICAL_COORDINATE_MANIFEST_REF_ATTR)
            != manifest.record_ref
            or run_attrs.get(
                STIMULUS_PHYSICAL_COORDINATE_MANIFEST_SHA256_ATTR
            )
            != manifest.record_sha256
        ):
            raise StimulusPhysicalCoordinateError(
                "Stimulus-run physical manifest pointers are stale."
            )
        common_archive = require_same_archive(
            root_node,
            run_group,
            source_node,
            frame_container,
            selected_node,
            physical_node,
        )
    except StimulusPhysicalCoordinateError:
        raise
    except (ArchiveIdentityError, PixelFrameAuthorityError) as exc:
        raise StimulusPhysicalCoordinateError(
            f"Physical-coordinate authority cannot be rebound: {exc}."
        ) from exc
    except Exception as exc:
        raise StimulusPhysicalCoordinateError(
            f"Persisted physical-coordinate authority is invalid: {exc}."
        ) from exc
    if (
        selected.archive_identity != common_archive
        or acquisition.archive_identity != common_archive
        or source_camera.archive_identity != common_archive
        or selected_evidence.archive_identity != common_archive
        or physical.archive_identity != common_archive
        or manifest.archive_identity != common_archive
    ):
        raise StimulusPhysicalCoordinateError(
            "Physical-coordinate authorities cross archive/store boundaries."
        )
    return BoundStimulusPhysicalCoordinateAuthority(
        stimulus_run=stimulus_run,
        camera_id=selected.camera_id,
        archive_identity=common_archive,
        selected_calibration=selected,
        acquisition_frame=acquisition,
        source_camera_frame=source_camera,
        selected_camera_evidence=selected_evidence,
        physical_frame=physical,
        manifest=manifest,
        root_node=root_node,
        _verification_seal=(
            _BOUND_STIMULUS_PHYSICAL_COORDINATE_AUTHORITY_SEAL
        ),
    )


def _load_stimulus_physical_coordinate_authority_before_selection(
    root_node: Any,
    *,
    stimulus_run: str,
    require_complete: bool,
) -> BoundStimulusPhysicalCoordinateAuthority | None:
    """Internal validator for a running or complete ineligible stimulus run."""

    run_group = _child(root_node, f"analysis/stimulus_runs/{stimulus_run}")
    _require_path(
        run_group,
        f"analysis/stimulus_runs/{stimulus_run}",
        label="Stimulus run",
    )
    if run_group.attrs.get("stage_selector_eligible") is not False:
        raise StimulusPhysicalCoordinateError(
            "Pre-selection physical validation requires literal "
            "stage_selector_eligible=false."
        )
    return _load_stimulus_physical_coordinate_authority(
        root_node,
        stimulus_run=stimulus_run,
        require_complete=require_complete,
        require_selector_eligible=False,
    )


def load_stimulus_physical_coordinate_authority(
    root_node: Any,
    *,
    stimulus_run: str,
) -> BoundStimulusPhysicalCoordinateAuthority | None:
    """Load physical authority only from an explicitly complete stimulus run."""

    return _load_stimulus_physical_coordinate_authority(
        root_node,
        stimulus_run=stimulus_run,
        require_complete=True,
        require_selector_eligible=True,
    )


def require_bound_stimulus_physical_coordinate_authority(
    value: Any,
) -> BoundStimulusPhysicalCoordinateAuthority:
    if type(value) is not BoundStimulusPhysicalCoordinateAuthority:
        raise StimulusPhysicalCoordinateError(
            "A sealed persisted stimulus physical-coordinate authority is required."
        )
    if (
        getattr(value, "_seal", None)
        is not _BOUND_STIMULUS_PHYSICAL_COORDINATE_AUTHORITY_SEAL
    ):
        raise StimulusPhysicalCoordinateError(
            "A freshly loader-minted stimulus physical-coordinate authority is "
            "required."
        )
    value.assert_verified()
    return value


__all__ = [
    "BoundStimulusPhysicalCoordinateAuthority",
    "STIMULUS_PHYSICAL_COORDINATE_BOUND_STATUS",
    "STIMULUS_PHYSICAL_COORDINATE_MANIFEST_ATTR",
    "STIMULUS_PHYSICAL_COORDINATE_MANIFEST_DIGEST_ATTR",
    "STIMULUS_PHYSICAL_COORDINATE_MANIFEST_REF_ATTR",
    "STIMULUS_PHYSICAL_COORDINATE_MANIFEST_SCHEMA_ID",
    "STIMULUS_PHYSICAL_COORDINATE_MANIFEST_SCHEMA_VERSION",
    "STIMULUS_PHYSICAL_COORDINATE_MANIFEST_SHA256_ATTR",
    "STIMULUS_PHYSICAL_COORDINATE_INVALIDATED_STATUS",
    "STIMULUS_PHYSICAL_COORDINATE_OMITTED_NO_ACQUISITION_STATUS",
    "STIMULUS_PHYSICAL_COORDINATE_OMITTED_NO_SCALE_STATUS",
    "STIMULUS_PHYSICAL_COORDINATE_OMITTED_STATUSES",
    "STIMULUS_PHYSICAL_COORDINATE_REASON_BOUND",
    "STIMULUS_PHYSICAL_COORDINATE_REASON_CODE_ATTR",
    "STIMULUS_PHYSICAL_COORDINATE_REASON_NO_ACQUISITION",
    "STIMULUS_PHYSICAL_COORDINATE_REASON_NO_SCALE",
    "STIMULUS_PHYSICAL_COORDINATE_REASON_PARENT_RUN_FAILED",
    "STIMULUS_PHYSICAL_COORDINATE_STATUS_ATTR",
    "StimulusPhysicalCoordinateError",
    "StimulusPhysicalCoordinateUnavailableError",
    "invalidate_stimulus_physical_coordinate_publication",
    "load_stimulus_physical_coordinate_authority",
    "publish_stimulus_physical_coordinate_authority",
    "require_bound_stimulus_physical_coordinate_authority",
]
