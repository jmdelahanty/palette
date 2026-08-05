"""Strict reader for recording-level subject-mask coordinate-v3 bundles."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from fisheye.shared.subject_mask_worker_receipt import (
    validate_recording_subject_mask_assembly_identity,
)
from fisheye.shared.zarr.crop_manifest import (
    CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    CROP_RUN_MANIFEST_ATTRIBUTE,
    validate_crop_run_manifest,
)
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr.subject_mask_bundle_publication import (
    SUBJECT_MASK_BUNDLE_AUTHORITY_ATTR,
    SUBJECT_MASK_BUNDLE_FAMILY,
    SUBJECT_MASK_BUNDLE_MANIFEST_ATTRIBUTE,
    SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR,
    validate_subject_mask_bundle_candidate,
)
from fisheye.shared.zarr.subject_mask_core_publication import (
    SUBJECT_MASK_CORE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE,
)
from fisheye.shared.zarr.subject_mask_schema import (
    RAW_SUBJECT_MASK_UINT8_SCHEMA_V1,
    REFINED_SUBJECT_MASK_CORE_SCHEMA_V1,
    SubjectMaskComponentRegistry,
    SubjectMaskDimensions,
)
from fisheye.shared.zarr.subject_mask_validation_receipt import (
    validate_subject_mask_source_validation_receipt,
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
    bundle_manifest: Mapping[str, Any] = field(repr=False)
    crop_manifest: Mapping[str, Any] = field(repr=False)
    raw_manifest: Mapping[str, Any] = field(repr=False)
    refined_manifest: Mapping[str, Any] = field(repr=False)
    raw_producer_evidence: Mapping[str, Any] = field(repr=False)
    refined_producer_evidence: Mapping[str, Any] = field(repr=False)
    coordinate_binding: Mapping[str, Any] = field(repr=False)
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


def _require_bundle_id(value: object) -> str:
    result = str(value or "").strip()
    if not result or "/" in result:
        raise SubjectMaskBundleCoordinateAuthorityError(
            "bundle_id must be one nonempty run name."
        )
    return result


def _strict_sidecar(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle, parse_constant=reject)
    if not isinstance(document, dict):
        raise ValueError(f"Expected one JSON object at {path}.")
    canonical_json_bytes(document)
    return document


def _dimensions_and_components(
    manifest: Mapping[str, Any],
) -> tuple[SubjectMaskDimensions, SubjectMaskComponentRegistry]:
    payload = manifest["payload"]
    logical = payload["logical_schema"]
    dimensions = logical["dimensions"]
    components = logical["components"]
    return (
        SubjectMaskDimensions(
            n_frames=dimensions["n_frames"],
            n_rois=dimensions["n_rois"],
            n_channels=dimensions["n_channels"],
            roi_height=dimensions["roi_height"],
            roi_width=dimensions["roi_width"],
        ),
        SubjectMaskComponentRegistry(tuple(components["labels"])),
    )


def _load_core_producer_evidence(
    archive: Path,
    root: Any,
    *,
    run_path: str,
    manifest: Mapping[str, Any],
    kind: str,
) -> dict[str, Any]:
    payload = manifest["payload"]
    source = payload["source"]
    binding = source["validation_receipt"]
    if not isinstance(binding, Mapping):
        raise SubjectMaskBundleCoordinateAuthorityError(
            f"Coordinate core {run_path!r} lacks source-validation evidence."
        )
    receipt_path = archive / str(binding.get("relative_path") or "")
    receipt = _strict_sidecar(receipt_path)
    receipt_bytes = canonical_json_bytes(receipt)
    if (
        receipt.get("payload_digest") != binding.get("payload_digest")
        or hashlib.sha256(receipt_bytes).hexdigest()
        != binding.get("document_sha256")
    ):
        raise SubjectMaskBundleCoordinateAuthorityError(
            f"Coordinate core {run_path!r} source receipt changed."
        )
    dimensions, components = _dimensions_and_components(manifest)
    schema = (
        RAW_SUBJECT_MASK_UINT8_SCHEMA_V1
        if kind == "raw_probability_uint8"
        else REFINED_SUBJECT_MASK_CORE_SCHEMA_V1
    )
    run = root[run_path]
    arrays = {
        binding.path: run[binding.path]
        for binding in schema.bindings
        if binding.required or binding.path in run
    }
    validated = validate_subject_mask_source_validation_receipt(
        receipt,
        kind=kind,
        source_run_path=source["run_path"],
        source_manifest=source["manifest"],
        schema=schema,
        arrays=arrays,
        dimensions=dimensions,
        components=components,
        threshold=(
            float(payload["logical_schema"]["threshold"])
            if kind == "raw_probability_uint8"
            else None
        ),
    )
    evidence = validated["payload"].get("producer_evidence")
    if not isinstance(evidence, dict):
        raise SubjectMaskBundleCoordinateAuthorityError(
            f"Coordinate core {run_path!r} lacks retained producer evidence."
        )
    stage = (
        "raw_subject_mask"
        if kind == "raw_probability_uint8"
        else "refined_subject_mask"
    )
    dependency = payload["coordinate_dependencies"]["document"][
        "recording_assembly"
    ]
    evidence = validate_recording_subject_mask_assembly_identity(
        evidence,
        kind=kind,
        stage_kind=stage,
        source_run_path=source["run_path"],
        n_rois=dimensions.n_rois,
    )
    if canonical_json_sha256(evidence) != dependency["producer_evidence_digest"]:
        raise SubjectMaskBundleCoordinateAuthorityError(
            f"Coordinate core {run_path!r} producer evidence binding changed."
        )
    return evidence


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
    validate_subject_mask_bundle_candidate(
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
    cross = payload["cross_binding"]
    coordinate = cross.get("coordinate_contract")
    if not isinstance(coordinate, Mapping):
        raise SubjectMaskBundleCoordinateAuthorityError(
            "Bundle lacks coordinate-v3 cross-binding."
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
        != SUBJECT_MASK_CORE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
        or refined_manifest.get("schema_version")
        != SUBJECT_MASK_CORE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    ):
        raise SubjectMaskBundleCoordinateAuthorityError(
            "Bundle members are not coordinate-core-v3 publications."
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

    raw_evidence = _load_core_producer_evidence(
        archive,
        root,
        run_path=raw_path,
        manifest=raw_manifest,
        kind="raw_probability_uint8",
    )
    refined_evidence = _load_core_producer_evidence(
        archive,
        root,
        run_path=refined_path,
        manifest=refined_manifest,
        kind="refined_dense_core",
    )
    recording_identity = str(payload.get("recording_identity") or "")
    if not recording_identity or recording_identity != str(
        root.attrs.get("recording_id") or ""
    ):
        raise SubjectMaskBundleCoordinateAuthorityError(
            "Bundle recording identity differs from the archive."
        )
    authority_document = {
        "recording_identity": recording_identity,
        "bundle_manifest_payload_digest": manifest["payload_digest"],
        "crop_manifest_payload_digest": crop_manifest["payload_digest"],
        "raw_manifest_payload_digest": raw_manifest["payload_digest"],
        "refined_manifest_payload_digest": refined_manifest["payload_digest"],
        "raw_producer_evidence_digest": canonical_json_sha256(raw_evidence),
        "refined_producer_evidence_digest": canonical_json_sha256(
            refined_evidence
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
        bundle_manifest=dict(manifest),
        crop_manifest=dict(crop_manifest),
        raw_manifest=dict(raw_manifest),
        refined_manifest=dict(refined_manifest),
        raw_producer_evidence=raw_evidence,
        refined_producer_evidence=refined_evidence,
        coordinate_binding=dict(coordinate),
        authority_digest=canonical_json_sha256(authority_document),
        active=active,
        _root=root,
        _verification_seal=_BOUND_AUTHORITY_SEAL,
    )


__all__ = [
    "BoundRecordingSubjectMaskCoordinateAuthority",
    "SubjectMaskBundleCoordinateAuthorityError",
    "load_recording_subject_mask_coordinate_authority",
]
