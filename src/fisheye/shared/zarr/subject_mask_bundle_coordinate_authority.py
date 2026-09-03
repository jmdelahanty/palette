"""Strict reader for recording-level subject-mask coordinate-v4/v5 cores."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.crop_manifest import (
    CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    CROP_RUN_MANIFEST_ATTRIBUTE,
    crop_pixel_authority_from_manifest,
    validate_crop_run_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.subject_mask_bundle_publication import (
    SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR,
    SUBJECT_MASK_BUNDLE_FAMILY,
    SUBJECT_MASK_BUNDLE_MANIFEST_ATTRIBUTE,
    SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR,
    validate_subject_mask_bundle_admission,
)
from fisheye.shared.zarr.subject_mask_core_publication import (
    SUBJECT_MASK_CORE_COMPOSABLE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    SUBJECT_MASK_CORE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE,
)
from fisheye.shared.zarr_io import open_zarr_root


class SubjectMaskBundleCoordinateAuthorityError(ValueError):
    """Raised when a bundle cannot prove its recording coordinate authority."""


_BOUND_AUTHORITY_SEAL = object()


@dataclass(frozen=True, init=False)
class BoundRecordingSubjectMaskCoordinateAuthority:
    archive_path: Path
    bundle_id: str
    recording_identity: str
    crop_run_path: str
    raw_run_path: str
    refined_run_path: str
    camera_identity: str
    source_total_frames: int
    source_width: int
    source_height: int
    n_rois: int
    roi_height: int
    roi_width: int
    bundle_manifest: Mapping[str, Any] = field(repr=False)
    crop_manifest: Mapping[str, Any] = field(repr=False)
    raw_manifest: Mapping[str, Any] = field(repr=False)
    refined_manifest: Mapping[str, Any] = field(repr=False)
    raw_recording_assembly: Mapping[str, Any] = field(repr=False)
    refined_recording_assembly: Mapping[str, Any] = field(repr=False)
    assignment_keypoint_collection: Mapping[str, Any] = field(repr=False)
    coordinate_binding: Mapping[str, Any] = field(repr=False)
    admission_receipt: Mapping[str, Any] = field(repr=False)
    authority_digest: str
    active: bool
    _root: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        _verification_seal: object | None = None,
        **values: Any,
    ) -> None:
        if _verification_seal is not _BOUND_AUTHORITY_SEAL:
            raise SubjectMaskBundleCoordinateAuthorityError(
                "Recording subject-mask coordinate authorities cannot be "
                "constructed directly."
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _verification_seal)

    @property
    def raw_run(self) -> Any:
        return self._root[self.raw_run_path]

    @property
    def refined_run(self) -> Any:
        return self._root[self.refined_run_path]

    @property
    def crop_run(self) -> Any:
        return self._root[self.crop_run_path]

    @property
    def instance_key_node(self) -> Any:
        return self.refined_run["instance_key"]

    @property
    def source_crop_row_ids_node(self) -> Any:
        return self.refined_run["source_crop_row_ids"]

    @property
    def source_acquisition_frame_index_node(self) -> Any:
        return self.refined_run["source_acquisition_frame_index"]

    @property
    def frame_row_offsets_node(self) -> Any:
        return self.refined_run["frame_row_offsets"]

    @property
    def source_crop_xywh_node(self) -> Any:
        return self.refined_run["source_crop_xywh"]

    def require_translation_only_offsets(self) -> np.ndarray:
        """Return exact ROI origins after rejecting resized crop placement."""

        placement = np.asarray(self.source_crop_xywh_node[:])
        if placement.dtype != np.dtype("float32") or placement.shape != (
            self.n_rois,
            4,
        ):
            raise SubjectMaskBundleCoordinateAuthorityError(
                "Bundle source_crop_xywh is not exact float32[N,4]."
            )
        if not np.isfinite(placement).all():
            raise SubjectMaskBundleCoordinateAuthorityError(
                "Bundle source_crop_xywh contains non-finite placement."
            )
        expected_size = np.asarray(
            [self.roi_width, self.roi_height],
            dtype=np.float32,
        )
        if not np.array_equal(
            placement[:, 2:4],
            np.broadcast_to(expected_size, (self.n_rois, 2)),
        ):
            raise SubjectMaskBundleCoordinateAuthorityError(
                "Subject-shape translation requires source crop width/height "
                "to equal the dense ROI raster extent exactly."
            )
        return np.asarray(placement[:, :2], dtype=np.float64)


def _require_bundle_id(value: object) -> str:
    result = str(value or "").strip()
    if not result or "/" in result:
        raise SubjectMaskBundleCoordinateAuthorityError(
            "bundle_id must be one nonempty run name."
        )
    return result


def _receipt_bound_recording_assembly(
    manifest: Mapping[str, Any],
    *,
    run_path: str,
) -> dict[str, Any]:
    """Return the compact assembly pointer already sealed by the core manifest.

    Normal bundle admission has already validated the complete core manifest,
    its outer bundle member receipt, and the bundle cross-binding. Reopening
    and canonicalizing the large worker-evidence sidecar here would merely
    replay publication-time work. The explicit bundle-candidate validator
    remains the deep sidecar replay surface.
    """

    try:
        payload = manifest["payload"]
        source = payload["source"]
        binding = source["validation_receipt"]
        assembly = payload["coordinate_dependencies"]["document"][
            "recording_assembly"
        ]
    except (KeyError, TypeError) as exc:
        raise SubjectMaskBundleCoordinateAuthorityError(
            f"Coordinate core {run_path!r} lacks receipt-bound assembly evidence."
        ) from exc
    if (
        not isinstance(source, Mapping)
        or not isinstance(binding, Mapping)
        or not isinstance(assembly, Mapping)
        or assembly.get("source_run_path") != source.get("run_path")
        or assembly.get("source_validation_receipt_payload_digest")
        != binding.get("payload_digest")
    ):
        raise SubjectMaskBundleCoordinateAuthorityError(
            f"Coordinate core {run_path!r} assembly receipt binding changed."
        )
    return dict(assembly)


def _selected_bundle(
    root: Any,
    *,
    bundle_id: str | None,
    allow_inactive: bool,
) -> tuple[str, bool]:
    authority = root.attrs.get(SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR)
    if bundle_id is None:
        if not isinstance(authority, Mapping):
            raise SubjectMaskBundleCoordinateAuthorityError(
                "No activated subject-mask bundle authority is present."
            )
        selected = _require_bundle_id(authority.get("bundle_id"))
        if authority.get("bundle_path") != f"{SUBJECT_MASK_BUNDLE_FAMILY}/{selected}":
            raise SubjectMaskBundleCoordinateAuthorityError(
                "Activated subject-mask bundle path is invalid."
            )
        return selected, True
    selected = _require_bundle_id(bundle_id)
    if isinstance(authority, Mapping) and authority.get("bundle_id") == selected:
        return selected, True
    if not allow_inactive:
        raise SubjectMaskBundleCoordinateAuthorityError(
            "An explicit unselected bundle requires allow_inactive=True."
        )
    return selected, False


def load_recording_subject_mask_coordinate_authority(
    analysis_zarr: Path,
    *,
    bundle_id: str | None = None,
    allow_inactive: bool = False,
) -> BoundRecordingSubjectMaskCoordinateAuthority:
    """Load one activated or explicitly authorized inactive coordinate bundle."""

    if type(allow_inactive) is not bool:
        raise TypeError("allow_inactive must be an exact bool.")
    archive = analysis_zarr.expanduser().resolve()
    root = open_zarr_root(archive, mode="r")
    selected, active = _selected_bundle(
        root,
        bundle_id=bundle_id,
        allow_inactive=allow_inactive,
    )
    admission_receipt = validate_subject_mask_bundle_admission(
        analysis_zarr=archive,
        bundle_id=selected,
    )
    bundle = root[f"{SUBJECT_MASK_BUNDLE_FAMILY}/{selected}"]
    manifest = bundle.attrs.get(SUBJECT_MASK_BUNDLE_MANIFEST_ATTRIBUTE)
    if not isinstance(manifest, Mapping):
        raise SubjectMaskBundleCoordinateAuthorityError(
            "Subject-mask bundle manifest is absent."
        )
    if active:
        authority = root.attrs[SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR]
        if authority.get("bundle_manifest_digest") != manifest.get("payload_digest"):
            raise SubjectMaskBundleCoordinateAuthorityError(
                "Activated bundle authority binds another manifest."
            )
        if bundle.attrs.get(SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR) is not True:
            raise SubjectMaskBundleCoordinateAuthorityError(
                "Activated bundle is not selector eligible."
            )
    elif bundle.attrs.get(SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR) is not False:
        raise SubjectMaskBundleCoordinateAuthorityError(
            "Inactive explicit bundle unexpectedly became selector eligible."
        )

    payload = manifest["payload"]
    recording_identity = str(payload.get("recording_identity") or "")
    if not recording_identity or recording_identity != str(
        root.attrs.get("recording_id") or ""
    ):
        raise SubjectMaskBundleCoordinateAuthorityError(
            "Bundle recording identity differs from the archive."
        )
    cross = payload["cross_binding"]
    coordinate = cross.get("coordinate_contract")
    if not isinstance(coordinate, Mapping):
        raise SubjectMaskBundleCoordinateAuthorityError(
            "Bundle lacks coordinate-v4 cross-binding."
        )
    members = payload["members"]
    raw_path = str(members["raw"]["run_path"])
    refined_path = str(members["refined"]["run_path"])
    raw_manifest = root[raw_path].attrs.get(SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE)
    refined_manifest = root[refined_path].attrs.get(
        SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE
    )
    if (
        not isinstance(raw_manifest, Mapping)
        or not isinstance(refined_manifest, Mapping)
        or raw_manifest.get("schema_version")
        not in {
            SUBJECT_MASK_CORE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
            SUBJECT_MASK_CORE_COMPOSABLE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
        }
        or refined_manifest.get("schema_version") != raw_manifest.get("schema_version")
    ):
        raise SubjectMaskBundleCoordinateAuthorityError(
            "Bundle members are not matching coordinate-core-v4/v5 publications."
        )
    crop_path = str(coordinate["crop"]["run_path"])
    crop_run = root[crop_path]
    crop_manifest = crop_run.attrs.get(CROP_RUN_MANIFEST_ATTRIBUTE)
    if not isinstance(crop_manifest, Mapping):
        raise SubjectMaskBundleCoordinateAuthorityError(
            "Bound crop-v2 manifest is absent."
        )
    crop_errors = validate_crop_run_manifest(crop_manifest)
    if (
        crop_errors
        or crop_manifest.get("schema_version")
        != CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
        or crop_manifest.get("payload_digest")
        != coordinate["crop"]["manifest_payload_digest"]
    ):
        raise SubjectMaskBundleCoordinateAuthorityError(
            "Bound crop-v2 coordinate authority is invalid or stale."
        )
    pixel_authority = crop_pixel_authority_from_manifest(
        crop_manifest["payload"]["source_pixel_authority"]
    )
    dimensions = refined_manifest["payload"]["logical_schema"]["dimensions"]
    expected_dimensions = {
        "n_frames": pixel_authority.n_frames,
        "n_rois": int(crop_run["instance_key"].shape[0]),
    }
    if any(
        type(dimensions.get(name)) is not int or dimensions[name] != expected
        for name, expected in expected_dimensions.items()
    ):
        raise SubjectMaskBundleCoordinateAuthorityError(
            "Refined core dimensions differ from the bound crop authority."
        )
    for name in ("roi_height", "roi_width"):
        if type(dimensions.get(name)) is not int or dimensions[name] <= 0:
            raise SubjectMaskBundleCoordinateAuthorityError(
                f"Refined core {name} is not a positive exact integer."
            )
    if (
        pixel_authority.recording_identity != recording_identity
        or pixel_authority.recording_identity
        != crop_manifest["payload"]["source_refined_snapshot"]["recording_identity"]
    ):
        raise SubjectMaskBundleCoordinateAuthorityError(
            "Crop pixel, refined-detection, bundle, and archive recording "
            "identities disagree."
        )

    raw_assembly = _receipt_bound_recording_assembly(
        raw_manifest,
        run_path=raw_path,
    )
    refined_assembly = _receipt_bound_recording_assembly(
        refined_manifest,
        run_path=refined_path,
    )
    assignment_keypoints = refined_manifest["payload"]["coordinate_dependencies"][
        "document"
    ]["assignment_keypoints"]
    if (
        not isinstance(assignment_keypoints, Mapping)
        or assignment_keypoints.get("n_rois") != dimensions["n_rois"]
    ):
        raise SubjectMaskBundleCoordinateAuthorityError(
            "Refined assignment-keypoint collection differs from the row authority."
        )
    authority_document = {
        "recording_identity": recording_identity,
        "bundle_manifest_payload_digest": manifest["payload_digest"],
        "crop_manifest_payload_digest": crop_manifest["payload_digest"],
        "raw_manifest_payload_digest": raw_manifest["payload_digest"],
        "refined_manifest_payload_digest": refined_manifest["payload_digest"],
        "raw_producer_evidence_digest": raw_assembly["producer_evidence_digest"],
        "refined_producer_evidence_digest": refined_assembly[
            "producer_evidence_digest"
        ],
        "assignment_keypoint_collection_digest": canonical_json_sha256(
            assignment_keypoints
        ),
        "coordinate_cross_binding_digest": canonical_json_sha256(coordinate),
    }
    return BoundRecordingSubjectMaskCoordinateAuthority(
        archive_path=archive,
        bundle_id=selected,
        recording_identity=recording_identity,
        crop_run_path=crop_path,
        raw_run_path=raw_path,
        refined_run_path=refined_path,
        camera_identity=pixel_authority.camera_identity,
        source_total_frames=pixel_authority.n_frames,
        source_width=pixel_authority.source_width,
        source_height=pixel_authority.source_height,
        n_rois=dimensions["n_rois"],
        roi_height=dimensions["roi_height"],
        roi_width=dimensions["roi_width"],
        bundle_manifest=dict(manifest),
        crop_manifest=dict(crop_manifest),
        raw_manifest=dict(raw_manifest),
        refined_manifest=dict(refined_manifest),
        raw_recording_assembly=raw_assembly,
        refined_recording_assembly=refined_assembly,
        assignment_keypoint_collection=dict(assignment_keypoints),
        coordinate_binding=dict(coordinate),
        admission_receipt=dict(admission_receipt),
        authority_digest=canonical_json_sha256(authority_document),
        active=active,
        _root=root,
        _verification_seal=_BOUND_AUTHORITY_SEAL,
    )


def require_bound_recording_subject_mask_coordinate_authority(
    value: Any,
) -> BoundRecordingSubjectMaskCoordinateAuthority:
    """Require an authority created by this module's receipt-backed loader."""

    if (
        type(value) is not BoundRecordingSubjectMaskCoordinateAuthority
        or value._seal is not _BOUND_AUTHORITY_SEAL
    ):
        raise SubjectMaskBundleCoordinateAuthorityError(
            "A sealed recording subject-mask coordinate authority is required."
        )
    return value


__all__ = [
    "BoundRecordingSubjectMaskCoordinateAuthority",
    "SubjectMaskBundleCoordinateAuthorityError",
    "load_recording_subject_mask_coordinate_authority",
    "require_bound_recording_subject_mask_coordinate_authority",
]
