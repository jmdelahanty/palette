"""Strict source binding for the detection-box position estimator.

This adapter intentionally supports only one source family: an explicitly
named, current, complete, selector-eligible ``detect_runs/<run>`` canonical
coordinate publication accepted by
``load_persisted_detection_observation_geometry``.  It does not resolve a
latest run, read refined detections, read artifact runs, or adapt a legacy
layout.

Canonical detection schema v1 defines every persisted row as one accepted,
finite, positive-area detection box within the normalized camera extent.  The
adapter rechecks those exact invariants and binds the run manifest as its
validity authority; callers cannot supply or override row validity.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.coordinate_descriptor import (
    CanonicalCoordinateDescriptor,
    CanonicalFrameRecord,
    DigestBoundCoordinateRecordRef,
    PIXEL_FRAME_AUTHORITY_RECORD_KIND,
    build_canonical_coordinate_descriptor,
)
from fisheye.shared.coordinate_identity import (
    load_bound_row_identity_contract,
)
from fisheye.shared.coordinate_surface_contract import (
    CANONICAL_OVERLAY_DIRECT,
    SOURCE_CAMERA_BBOX_PIXEL_CONVENTION,
    SOURCE_CAMERA_BBOX_XYXY,
    SOURCE_CAMERA_PROFILE_ID,
)
from fisheye.shared.observation_coordinate_publication import (
    detection_observation_geometry_values,
    load_persisted_detection_observation_geometry,
    load_persisted_ineligible_detection_observation_geometry,
    require_bound_detection_observation_geometry,
)
from fisheye.shared.subject_position_expression import (
    BoundingBoxSourceBinding,
    PointExpressionBindings,
)
from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
    load_source_camera_pixel_frame_authority,
)
from fisheye.shared.zarr.canonical_detection_manifest import (
    CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    canonical_detection_dimensions_from_manifest,
    validate_canonical_detection_run_manifest,
    require_active_coordinate_canonical_detection,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.detection_schema import (
    CANONICAL_DETECTION_SCHEMA_V1,
    derive_canonical_detection_geometry,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_io import open_zarr_root


DETECTION_POSITION_SOURCE_SCHEMA_ID = (
    "palette.subject_position.detection_source_binding"
)
DETECTION_POSITION_SOURCE_SCHEMA_VERSION = 1
DETECTION_POSITION_SOURCE_KIND_VALUES = frozenset(
    {"legacy_conversion", "native_detection"}
)
DETECTION_POSITION_VALIDITY_POLICY_ID = (
    "canonical_detection_schema_v1_all_rows_valid.v1"
)
DETECTION_POSITION_ACTIVE_AUTHORITY_POLICY_ID = (
    "active_selector_coordinate_canonical_detection.v1"
)
DETECTION_POSITION_CANARY_AUTHORITY_POLICY_ID = (
    "sealed_selector_ineligible_coordinate_canonical_detection_candidate.v1"
)
DETECTION_POSITION_CANARY_PRODUCTION_SELECTOR_ACTIVATION = "deferred"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BOUND_DETECTION_POSITION_SOURCE_SEAL = object()


@dataclass(frozen=True)
class _ManifestBoundBBoxDescriptor:
    """Canonical bbox descriptor reconstructed from sealed v3 authorities."""

    descriptor: CanonicalCoordinateDescriptor
    reference_frame_authority: Any


@dataclass(frozen=True)
class _DigestAuthority:
    record_ref: str
    record_sha256: str


@dataclass(frozen=True)
class _ManifestDetectionFrameEvidence:
    source_camera_frame: Any
    bbox_source_camera_frame: Any


@dataclass(frozen=True)
class _ManifestDetectionGeometry:
    row_identity: Any
    bbox_image: _ManifestBoundBBoxDescriptor
    frame_evidence: _ManifestDetectionFrameEvidence
    bbox_projection: _DigestAuthority
    temporal_authority: _DigestAuthority


class DetectionPositionSourceError(ValueError):
    """Raised when a detection source cannot be bound without guessing."""


def _fail(message: str) -> None:
    raise DetectionPositionSourceError(message)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _readonly_array(value: Any, *, dtype: np.dtype, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        _fail(f"{name} is not a readable array: {exc}")
    if array.dtype != dtype:
        _fail(f"{name} must have exact dtype {dtype.str}; found {array.dtype!s}.")
    result = np.array(array, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _required_sha256(value: Any, *, name: str) -> str:
    digest = str(value or "").strip().lower().removeprefix("sha256:")
    if _SHA256_RE.fullmatch(digest) is None:
        _fail(f"{name} must be one lowercase SHA-256 digest.")
    return digest


def _required_text(value: Any, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        _fail(f"{name} must be one non-empty string.")
    return value.strip()


def _require_exact_run_path(run_path: Any) -> str:
    path = _required_text(run_path, name="run_path").strip("/")
    parts = path.split("/")
    if len(parts) != 2 or parts[0] != "detect_runs" or not parts[1]:
        _fail(
            "Detection position source requires one exact detect_runs/<run> path; "
            "latest resolution and other source families are unsupported."
        )
    return path


def _require_canary_run_path(run_path: Any) -> str:
    path = _require_exact_run_path(run_path)
    run_id = path.rsplit("/", 1)[1]
    if run_id in {"latest", "latest_complete", "latest_pending", "authoritative_run"}:
        _fail("Canary detection authority never resolves selector aliases.")
    return path


def _authority_record(value: Any, *, name: str) -> dict[str, str]:
    record_ref = _required_text(
        getattr(value, "record_ref", None), name=f"{name}.record_ref"
    )
    record_sha256 = _required_sha256(
        getattr(value, "record_sha256", None), name=f"{name}.record_sha256"
    )
    return {"record_ref": record_ref, "record_sha256": record_sha256}


def _require_manifest_binding(
    manifest: Mapping[str, Any],
    *,
    run_path: str,
    expected_selector_eligible: bool = True,
) -> tuple[str, dict[str, Any]]:
    if not isinstance(manifest, Mapping):
        _fail("The active detection authority did not return a manifest mapping.")
    if manifest.get("schema_version") != (
        CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    ):
        _fail("Detection position source requires canonical coordinate manifest v3.")
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping):
        _fail("Canonical detection manifest payload is missing.")
    if payload.get("run_id") != run_path.rsplit("/", 1)[1]:
        _fail("Canonical detection manifest does not bind the explicitly selected run.")
    source_kind = payload.get("source_evidence_kind")
    if source_kind not in DETECTION_POSITION_SOURCE_KIND_VALUES:
        _fail("Canonical detection source_evidence_kind is unsupported.")
    publication = payload.get("publication")
    if not isinstance(publication, Mapping):
        _fail("Canonical detection publication evidence is missing.")
    expected_publication = {
        "completion_status": "complete",
        "stage_selector_eligible": expected_selector_eligible,
        "metadata_state": "direct_and_consolidated_validated",
    }
    for name, expected in expected_publication.items():
        if publication.get(name) != expected:
            _fail(
                f"Canonical detection publication evidence {name!r} is stale or invalid."
            )
    for name in ("metadata_declarations_digest",):
        _required_sha256(publication.get(name), name=f"publication.{name}")
    manifest_digest = _required_sha256(
        manifest.get("payload_digest"),
        name="canonical detection manifest.payload_digest",
    )
    logical_content = payload.get("logical_content")
    if not isinstance(logical_content, Mapping):
        _fail("Canonical detection manifest lacks logical-content authority.")
    logical_content_digest = _required_sha256(
        logical_content.get("digest"), name="logical_content.digest"
    )
    return str(source_kind), {
        "manifest_digest": manifest_digest,
        "logical_content_digest": logical_content_digest,
        "publication": copy.deepcopy(dict(publication)),
    }


def _require_selector_ineligible_candidate(
    root: Any,
    *,
    run_path: str,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Require one explicitly named, sealed production-candidate run.

    This is deliberately separate from ``require_active_coordinate_canonical_detection``.
    It is a canary-only read authority: it never consults or resolves a selector and
    cannot turn an ineligible candidate into active production authority.
    """

    path = _require_canary_run_path(run_path)
    run_id = path.rsplit("/", 1)[1]
    try:
        family = root["detect_runs"]
        run = family[run_id]
    except (KeyError, TypeError, ValueError) as exc:
        _fail(f"Sealed canary detection run is missing: {path}")
        raise AssertionError("unreachable") from exc

    attrs = run.attrs
    if attrs.get("status") != "complete":
        _fail("Sealed canary detection run is not complete.")
    if attrs.get("palette_run_completion_status") != "complete":
        _fail("Sealed canary detection run lacks complete run-completion evidence.")
    if attrs.get("stage_selector_eligible") is not False:
        _fail("Sealed canary detection run must remain selector ineligible.")
    if attrs.get("immutable_snapshot") is not True:
        _fail("Sealed canary detection run is not an immutable snapshot.")
    if attrs.get("production_candidate") is not True:
        _fail("Sealed canary detection run lacks the production-candidate marker.")
    if attrs.get("production_selector_activation") != (
        DETECTION_POSITION_CANARY_PRODUCTION_SELECTOR_ACTIVATION
    ):
        _fail(
            "Sealed canary detection run has an invalid production activation marker."
        )

    # A selector-ineligible candidate must not be smuggled into any selector, and
    # no selector is ever used to find this run.  Missing selector attributes are
    # acceptable; a reference to this candidate is not.
    selector_refs = {
        name: family.attrs.get(name)
        for name in ("latest", "latest_complete", "latest_pending", "authoritative_run")
    }
    if any(value == run_id for value in selector_refs.values()):
        _fail("Sealed canary detection run is referenced by a selector.")

    manifest = attrs.get("run_manifest")
    if not isinstance(manifest, Mapping):
        _fail("Sealed canary detection run lacks its manifest.")
    try:
        errors = validate_canonical_detection_run_manifest(manifest)
    except Exception as exc:
        _fail(f"Canonical detection manifest validation failed: {exc}")
    if errors:
        _fail("Canonical detection manifest is invalid: " + "; ".join(errors))
    if manifest.get("schema_version") != (
        CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    ):
        _fail("Sealed canary detection manifest is not canonical v3.")
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping) or payload.get("run_id") != run_id:
        _fail("Sealed canary detection manifest run id differs from the selected run.")
    publication = payload.get("publication")
    if not isinstance(publication, Mapping):  # validator normally catches this
        _fail("Sealed canary detection publication evidence is missing.")
    if publication.get("completion_status") != "complete":
        _fail("Sealed canary detection publication is not complete.")
    if publication.get("stage_selector_eligible") is not False:
        _fail("Sealed canary detection publication must remain selector ineligible.")

    manifest_digest = _required_sha256(
        manifest.get("payload_digest"),
        name="canonical detection manifest.payload_digest",
    )
    authority_evidence: dict[str, Any] = {
        "policy_id": DETECTION_POSITION_CANARY_AUTHORITY_POLICY_ID,
        "policy_version": 1,
        "run_path": path,
        "run_completion": {
            "status": attrs.get("status"),
            "palette_run_completion_status": attrs.get("palette_run_completion_status"),
        },
        "selector_state": {
            "stage_selector_eligible": attrs.get("stage_selector_eligible"),
            "resolved_by": "explicit_run_path_only",
            "selector_references": selector_refs,
        },
        "production_candidate": {
            "production_candidate": attrs.get("production_candidate"),
            "production_selector_activation": attrs.get(
                "production_selector_activation"
            ),
        },
        "manifest": {
            "schema_version": manifest.get("schema_version"),
            "run_id": payload.get("run_id"),
            "payload_digest": manifest_digest,
            "publication_stage_selector_eligible": publication.get(
                "stage_selector_eligible"
            ),
        },
    }
    return manifest, authority_evidence


def _require_bbox_descriptor(geometry: Any) -> None:
    descriptor = geometry.bbox_image.descriptor
    expected = SOURCE_CAMERA_BBOX_XYXY.descriptor_kwargs()
    actual = {
        "profile_id": descriptor.profile_id,
        "geometry_type": descriptor.geometry_type,
        "components": descriptor.components,
        "component_units": descriptor.component_units,
        "pixel_convention": descriptor.pixel_convention,
        "source_camera_overlay_status": descriptor.source_camera_overlay.status,
    }
    if actual != expected:
        _fail(
            "Detection bbox descriptor is not the canonical direct source-camera "
            "half-open XYXY contract."
        )
    authority = geometry.bbox_image.reference_frame_authority
    bbox_frame = geometry.frame_evidence.bbox_source_camera_frame
    if authority is None or authority.record_sha256 != bbox_frame.record_sha256:
        _fail(
            "Detection bbox descriptor is not bound to its exact bbox frame authority."
        )
    if (
        descriptor.profile_id != SOURCE_CAMERA_PROFILE_ID
        or descriptor.pixel_convention != (SOURCE_CAMERA_BBOX_PIXEL_CONVENTION)
        or descriptor.source_camera_overlay.status != CANONICAL_OVERLAY_DIRECT
    ):
        _fail(
            "Detection bbox descriptor has an incompatible source-camera coordinate authority."
        )


def _canonical_detection_validity(
    bbox_norm: np.ndarray,
    *,
    run_path: str,
    manifest_evidence: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recheck canonical schema-v1 box validity and bind its persisted authority."""

    if bbox_norm.ndim != 2 or bbox_norm.shape[1:] != (4,):
        _fail("bbox_norm_coords must have exact shape [N, 4].")
    # Reconstruct normalized endpoints in the persisted floating-point dtype.
    # Canonical detections store center/extent values as float32.  Promoting
    # those rounded operands to float64 *before* reconstruction can turn an
    # endpoint that is exactly 1.0 in the storage arithmetic into a value a
    # few 1e-8 above 1.0.  That is not an out-of-frame box; it is a mixed-
    # precision validation artifact.  The schema invariant applies to the
    # persisted representation, so its arithmetic domain is authoritative.
    boxes = np.asarray(bbox_norm)
    half = np.asarray(0.5, dtype=boxes.dtype)
    x_min = boxes[:, 0] - boxes[:, 2] * half
    y_min = boxes[:, 1] - boxes[:, 3] * half
    x_max = boxes[:, 0] + boxes[:, 2] * half
    y_max = boxes[:, 1] + boxes[:, 3] * half
    schema_valid = bool(
        np.isfinite(boxes).all()
        and np.all(boxes[:, 2] > 0.0)
        and np.all(boxes[:, 3] > 0.0)
        and np.all(x_min >= 0.0)
        and np.all(y_min >= 0.0)
        and np.all(x_max <= 1.0)
        and np.all(y_max <= 1.0)
    )
    if not schema_valid:
        _fail(
            "Canonical detection rows violate schema-v1 finite, positive-area, "
            "in-extent box validity."
        )
    validity = np.ones(boxes.shape[0], dtype=bool)
    validity.setflags(write=False)
    record = {
        "schema_id": "palette.subject_position.detection_validity_authority",
        "schema_version": 1,
        "policy_id": DETECTION_POSITION_VALIDITY_POLICY_ID,
        "record_ref": f"/{run_path}@run_manifest",
        "record_sha256": manifest_evidence["manifest_digest"],
        "logical_content_sha256": manifest_evidence["logical_content_digest"],
        "validation": ("canonical_detection_schema_v1_finite_positive_area_in_extent"),
        "values_dtype": "bool",
        "values_shape": [boxes.shape[0]],
        "values_sha256": array_values_sha256(validity),
    }
    return validity, record


def _manifest_logical_arrays(
    root_node: Any,
    *,
    run_path: str,
    manifest: Mapping[str, Any],
) -> dict[str, np.ndarray] | None:
    """Load and revalidate one canonical-v3 ``instances/*`` table.

    Older in-memory adapter fixtures deliberately expose only the historical
    flat observation loader.  Returning ``None`` when the v3 table is absent
    preserves those explicit compatibility tests; a persisted canonical-v3
    run always has the table and therefore always takes this strict path.
    """

    try:
        run = root_node[run_path]
        instances = run["instances"]
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    del instances  # existence is the layout discriminator; paths stay manifest-owned

    dimensions = canonical_detection_dimensions_from_manifest(manifest)
    arrays: dict[str, np.ndarray] = {}
    try:
        for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths:
            values = np.asarray(run[path][...])
            snapshot = np.array(values, copy=True, order="C")
            snapshot.setflags(write=False)
            arrays[path] = snapshot
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        _fail(f"Canonical-v3 detection arrays are unreadable: {exc}.")
    try:
        CANONICAL_DETECTION_SCHEMA_V1.require(arrays, dimensions=dimensions)
    except (TypeError, ValueError) as exc:
        _fail(f"Canonical-v3 detection arrays violate their logical schema: {exc}.")

    logical_content = manifest["payload"]["logical_content"]
    document = logical_content.get("document")
    declarations = document.get("arrays") if isinstance(document, Mapping) else None
    if not isinstance(declarations, Mapping):
        _fail("Canonical-v3 detection manifest lacks exact array declarations.")
    for path, values in arrays.items():
        declaration = declarations.get(path)
        if not isinstance(declaration, Mapping):
            _fail(f"Canonical-v3 detection manifest does not declare {path!r}.")
        observed = {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "sha256": sha256_array(values),
        }
        expected = {
            "shape": declaration.get("shape"),
            "dtype": declaration.get("dtype"),
            "sha256": declaration.get("sha256"),
        }
        if observed != expected:
            _fail(f"Canonical-v3 detection array {path!r} differs from its manifest.")
    return arrays


def _manifest_bound_detection_geometry(
    root_node: Any,
    *,
    run_path: str,
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> tuple[_ManifestDetectionGeometry, dict[str, np.ndarray]]:
    """Bind canonical-v3 geometry without invoking the legacy flat reader."""

    dimensions = canonical_detection_dimensions_from_manifest(manifest)
    instance_key = arrays["instances/instance_key"]
    frame_indices = arrays["instances/frame_indices"]
    acquisition_frames = arrays["instances/source_acquisition_frame_index"]
    bbox_norm = arrays["instances/bbox_norm_coords"]
    bbox_img = arrays["instances/bbox_img_xyxy"]
    centers_img = arrays["instances/centers_img_xy"]
    if not np.array_equal(frame_indices.astype(np.int64), acquisition_frames):
        _fail(
            "Canonical-v3 detection frame_indices and acquisition frames are not "
            "the exact full-recording identity mapping."
        )
    expected_bbox, expected_centers = derive_canonical_detection_geometry(
        bbox_norm,
        source_width=int(dimensions.source_width),
        source_height=int(dimensions.source_height),
    )
    if not np.array_equal(bbox_img, expected_bbox):
        _fail("Canonical-v3 bbox_img_xyxy differs from its normalized projection.")
    if not np.array_equal(centers_img, expected_centers):
        _fail("Canonical-v3 centers_img_xy differs from its bbox midpoint.")

    try:
        _, acquisition = load_persisted_acquisition_camera_authority(root_node)
        camera_id = str(acquisition.record.camera_id)
        source_camera = load_source_camera_pixel_frame_authority(
            root_node[
                f"analysis/coordinate_frames/source_camera/{camera_id}/continuous"
            ],
            acquisition_frame=acquisition,
        )
        bbox_camera = load_source_camera_pixel_frame_authority(
            root_node[
                "analysis/coordinate_frames/source_camera/"
                f"{camera_id}/{SOURCE_CAMERA_BBOX_PIXEL_CONVENTION}"
            ],
            acquisition_frame=acquisition,
        )
    except Exception as exc:
        _fail(f"Canonical-v3 source-camera frame authority is invalid: {exc}.")

    run = root_node[run_path]
    pointer = run.attrs.get("source_pixel_authority")
    expected_pointer = {
        "record_ref": source_camera.record_ref,
        "record_sha256": source_camera.record_sha256,
    }
    if pointer != expected_pointer:
        _fail("Canonical-v3 source_pixel_authority is missing, stale, or mismatched.")
    endpoint = source_camera.endpoint
    if (
        int(endpoint.width) != int(dimensions.source_width)
        or int(endpoint.height) != int(dimensions.source_height)
        or int(acquisition.record.source_total_frames) != int(dimensions.n_frames)
    ):
        _fail("Canonical-v3 dimensions disagree with acquisition camera authority.")
    if acquisition_frames.size and (
        np.any(acquisition_frames < 0)
        or np.any(acquisition_frames >= int(acquisition.record.source_total_frames))
    ):
        _fail("Canonical-v3 rows fall outside the acquisition frame domain.")

    manifest_digest = _required_sha256(
        manifest.get("payload_digest"),
        name="canonical detection manifest.payload_digest",
    )
    manifest_ref = f"/{run_path}@run_manifest"
    try:
        identity = load_bound_row_identity_contract(
            run["instances"],
            run["instances/instance_key"],
        )
    except Exception as exc:
        _fail(f"Canonical-v3 detection row identity is invalid: {exc}.")
    if identity.leading_dimension != int(instance_key.shape[0]):
        _fail("Canonical-v3 row identity length differs from its instance table.")
    manifest_authority = DigestBoundCoordinateRecordRef(
        record_ref=manifest_ref,
        record_sha256=manifest_digest,
    )
    bbox_descriptor = build_canonical_coordinate_descriptor(
        **SOURCE_CAMERA_BBOX_XYXY.descriptor_kwargs(),
        reference_width=int(endpoint.width),
        reference_height=int(endpoint.height),
        reference_authority=DigestBoundCoordinateRecordRef(
            record_ref=bbox_camera.record_ref,
            record_sha256=bbox_camera.record_sha256,
        ),
        reference_selector="record",
        row_identity_contract=identity.contract,
        row_identity_record_ref=identity.record_ref,
        lineage_refs=(manifest_authority,),
        frame_record=CanonicalFrameRecord(
            kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
            record_ref=bbox_camera.record_ref,
            record_sha256=bbox_camera.record_sha256,
        ),
    )
    authority = _DigestAuthority(
        record_ref=manifest_ref,
        record_sha256=manifest_digest,
    )
    geometry = _ManifestDetectionGeometry(
        row_identity=identity,
        bbox_image=_ManifestBoundBBoxDescriptor(
            descriptor=bbox_descriptor,
            reference_frame_authority=bbox_camera,
        ),
        frame_evidence=_ManifestDetectionFrameEvidence(
            source_camera_frame=source_camera,
            bbox_source_camera_frame=bbox_camera,
        ),
        bbox_projection=authority,
        temporal_authority=authority,
    )
    return geometry, {
        "instance_key": instance_key,
        "source_acquisition_frame_index": acquisition_frames,
        "bbox_norm_coords": bbox_norm,
        "bbox_img_xyxy": bbox_img,
        "centers_img_xy": centers_img,
    }


def _build_source(
    root_node: Any,
    run_path: str,
    *,
    direct_consolidated_evidence: Mapping[str, Any] | None = None,
    authority_policy: str = DETECTION_POSITION_ACTIVE_AUTHORITY_POLICY_ID,
) -> "BoundDetectionPositionSource":
    if authority_policy == DETECTION_POSITION_ACTIVE_AUTHORITY_POLICY_ID:
        selected_manifest = require_active_coordinate_canonical_detection(
            root_node, group_path=run_path
        )
        authority_evidence: dict[str, Any] = {
            "policy_id": DETECTION_POSITION_ACTIVE_AUTHORITY_POLICY_ID,
            "policy_version": 1,
            "run_path": run_path,
            "resolved_by": "active_selector_exact_run_path",
        }
        expected_selector_eligible = True
    elif authority_policy == DETECTION_POSITION_CANARY_AUTHORITY_POLICY_ID:
        selected_manifest, authority_evidence = _require_selector_ineligible_candidate(
            root_node,
            run_path=run_path,
        )
        expected_selector_eligible = False
    else:
        _fail(f"Unsupported detection position authority policy: {authority_policy!r}.")
    source_kind, manifest_evidence = _require_manifest_binding(
        selected_manifest,
        run_path=run_path,
        expected_selector_eligible=expected_selector_eligible,
    )
    manifest_arrays = _manifest_logical_arrays(
        root_node,
        run_path=run_path,
        manifest=selected_manifest,
    )
    if manifest_arrays is None:
        geometry_loader = (
            load_persisted_detection_observation_geometry
            if authority_policy == DETECTION_POSITION_ACTIVE_AUTHORITY_POLICY_ID
            else load_persisted_ineligible_detection_observation_geometry
        )
        geometry = geometry_loader(root_node, run_path)
        geometry = require_bound_detection_observation_geometry(geometry)
        values = detection_observation_geometry_values(geometry)
        expected_rowset_paths = {run_path}
    else:
        geometry, values = _manifest_bound_detection_geometry(
            root_node,
            run_path=run_path,
            manifest=selected_manifest,
            arrays=manifest_arrays,
        )
        expected_rowset_paths = {f"{run_path}/instances"}
    if geometry.row_identity.rowset_path not in expected_rowset_paths:
        _fail("Detection geometry row identity is bound to another run.")
    _require_bbox_descriptor(geometry)
    instance_key = _readonly_array(
        values["instance_key"], dtype=np.dtype("<u8"), name="instance_key"
    )
    acquisition_frames = _readonly_array(
        values["source_acquisition_frame_index"],
        dtype=np.dtype("<i8"),
        name="source_acquisition_frame_index",
    )
    bbox = np.asarray(values["bbox_img_xyxy"])
    if bbox.ndim != 2 or bbox.shape != (instance_key.shape[0], 4):
        _fail("bbox_img_xyxy is not row-aligned with the canonical instance identity.")
    bbox = np.array(bbox, copy=True, order="C")
    bbox.setflags(write=False)
    bbox_norm = np.asarray(values["bbox_norm_coords"])
    if bbox_norm.shape != bbox.shape:
        _fail("bbox_norm_coords is not row-aligned with bbox_img_xyxy.")
    source_row_index = np.arange(instance_key.shape[0], dtype=np.dtype("<i8"))
    source_row_index.setflags(write=False)

    key_digest = array_values_sha256(instance_key)
    key_contract = geometry.row_identity.contract.key_array
    if key_contract.content_sha256 != key_digest:
        _fail("Canonical detection identities are reordered or stale after binding.")
    if key_contract.leading_dimension != instance_key.shape[0]:
        _fail("Canonical detection identity cardinality disagrees with its arrays.")

    validity, validity_record = _canonical_detection_validity(
        bbox_norm,
        run_path=run_path,
        manifest_evidence=manifest_evidence,
    )
    descriptor_record = geometry.bbox_image.descriptor.to_dict()
    binding_record: dict[str, Any] = {
        "schema_id": DETECTION_POSITION_SOURCE_SCHEMA_ID,
        "schema_version": DETECTION_POSITION_SOURCE_SCHEMA_VERSION,
        "source_modality": "detection",
        "source_kind": source_kind,
        "run_path": run_path,
        "authority": {
            "policy_id": authority_policy,
            "evidence": copy.deepcopy(authority_evidence),
            "evidence_sha256": canonical_json_sha256(authority_evidence),
        },
        "row_axis": "observation_instance",
        "row_identity": {
            "record_ref": geometry.row_identity.record_ref,
            "record_sha256": geometry.row_identity.record_sha256,
            "rowset_path": geometry.row_identity.rowset_path,
            "key_array_path": geometry.row_identity.key_array_path,
            "key_content_sha256": key_digest,
        },
        "source_arrays": {
            "instance_key": {
                "path": geometry.row_identity.key_array_path,
                "dtype": "uint64",
                "shape": [instance_key.shape[0]],
                "sha256": key_digest,
            },
            "source_acquisition_frame_index": {
                "path": (
                    f"{geometry.row_identity.rowset_path}/"
                    "source_acquisition_frame_index"
                ),
                "dtype": "int64",
                "shape": [acquisition_frames.shape[0]],
                "sha256": array_values_sha256(acquisition_frames),
            },
            "bbox_img_xyxy": {
                "path": f"{geometry.row_identity.rowset_path}/bbox_img_xyxy",
                "dtype": str(bbox.dtype),
                "shape": list(bbox.shape),
                "sha256": array_values_sha256(bbox),
                "coordinate_descriptor": descriptor_record,
                "coordinate_descriptor_sha256": geometry.bbox_image.descriptor.digest(),
                "lineage_record": _authority_record(
                    geometry.bbox_projection, name="bbox_projection"
                ),
            },
        },
        "source_camera_frame": _authority_record(
            geometry.frame_evidence.source_camera_frame,
            name="source_camera_frame",
        ),
        "bbox_source_camera_frame": _authority_record(
            geometry.frame_evidence.bbox_source_camera_frame,
            name="bbox_source_camera_frame",
        ),
        "temporal_authority": _authority_record(
            geometry.temporal_authority, name="temporal_authority"
        ),
        "canonical_detection_manifest": {
            "record_ref": f"/{run_path}@run_manifest",
            "record_sha256": manifest_evidence["manifest_digest"],
        },
        "metadata_evidence": manifest_evidence["publication"],
        "direct_consolidated_evidence": (
            {"mode": "explicit_open_root_test_adapter"}
            if direct_consolidated_evidence is None
            else copy.deepcopy(dict(direct_consolidated_evidence))
        ),
        "observation_validity": validity_record,
    }
    binding_digest = canonical_json_sha256(binding_record)
    bindings = PointExpressionBindings(
        bboxes=MappingProxyType(
            {
                "bbox_img_xyxy": BoundingBoxSourceBinding(
                    xyxy=bbox,
                    valid=validity,
                )
            }
        )
    )
    return BoundDetectionPositionSource(
        source_modality="detection",
        source_kind=source_kind,
        run_path=run_path,
        authority_policy=authority_policy,
        row_identity=geometry.row_identity,
        instance_key=instance_key,
        source_acquisition_frame_index=acquisition_frames,
        source_row_index=source_row_index,
        source_camera_frame=geometry.frame_evidence.source_camera_frame,
        bbox_source_camera_frame=geometry.frame_evidence.bbox_source_camera_frame,
        bbox_descriptor=geometry.bbox_image,
        source_binding_record=_freeze_json(binding_record),
        source_binding_digest=binding_digest,
        point_expression_bindings=bindings,
        observation_validity=validity,
        direct_consolidated_evidence=_freeze_json(
            binding_record["direct_consolidated_evidence"]
        ),
        _analysis_zarr=None,
        _root_node=root_node,
        _verification_seal=_BOUND_DETECTION_POSITION_SOURCE_SEAL,
    )


@dataclass(frozen=True, init=False)
class BoundDetectionPositionSource:
    """Sealed canonical detection source for ``detection_bbox_centroid.v1``."""

    source_modality: str
    source_kind: str
    run_path: str
    authority_policy: str
    row_identity: Any = field(repr=False)
    instance_key: np.ndarray
    source_acquisition_frame_index: np.ndarray
    source_row_index: np.ndarray
    source_camera_frame: Any = field(repr=False)
    bbox_source_camera_frame: Any = field(repr=False)
    bbox_descriptor: Any = field(repr=False)
    source_binding_record: Mapping[str, Any] = field(repr=False)
    source_binding_digest: str
    point_expression_bindings: PointExpressionBindings = field(repr=False)
    observation_validity: np.ndarray = field(repr=False)
    direct_consolidated_evidence: Mapping[str, Any] = field(repr=False)
    _analysis_zarr: Path | None = field(repr=False, compare=False)
    _root_node: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self, *, _verification_seal: object | None = None, **values: Any
    ) -> None:
        if _verification_seal is not _BOUND_DETECTION_POSITION_SOURCE_SEAL:
            _fail("Detection position sources must be built by the strict adapter.")
        values.setdefault(
            "authority_policy",
            DETECTION_POSITION_ACTIVE_AUTHORITY_POLICY_ID,
        )
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _verification_seal)

    @property
    def bindings(self) -> PointExpressionBindings:
        """Compatibility spelling for consumers that call them source bindings."""

        return self.point_expression_bindings

    @property
    def expression_bindings(self) -> PointExpressionBindings:
        """Common adapter spelling consumed by subject-position preparation."""

        return self.point_expression_bindings

    def revalidate(self) -> "BoundDetectionPositionSource":
        """Reload every persisted authority before downstream consumption."""

        return require_bound_detection_position_source(self)

    def assert_current(self) -> None:
        """Revalidate the exact selected family, arrays, and metadata."""

        require_bound_detection_position_source(self)


def load_persisted_detection_position_source(
    analysis_zarr: str | Path | Any,
    run_path: str,
) -> BoundDetectionPositionSource:
    """Bind one explicitly named current canonical detection observation run.

    The exact canonical run manifest and schema-v1 row invariants own validity.
    Another detection or refinement run is never tried.
    """

    return _load_persisted_detection_position_source(
        analysis_zarr,
        run_path,
        authority_policy=DETECTION_POSITION_ACTIVE_AUTHORITY_POLICY_ID,
        path_validator=_require_exact_run_path,
    )


def load_persisted_selector_ineligible_detection_position_source(
    analysis_zarr: str | Path | Any,
    run_path: str,
) -> BoundDetectionPositionSource:
    """Bind one exact sealed v3 production candidate for a canary only.

    This authority is intentionally not a fallback for the active source.  The
    caller must name the run explicitly, and the candidate must remain complete,
    immutable, production-marked, and selector-ineligible.  It never resolves a
    family ``latest`` pointer.
    """

    return _load_persisted_detection_position_source(
        analysis_zarr,
        run_path,
        authority_policy=DETECTION_POSITION_CANARY_AUTHORITY_POLICY_ID,
        path_validator=_require_canary_run_path,
    )


bind_detection_position_source = load_persisted_detection_position_source
bind_selector_ineligible_detection_position_source = (
    load_persisted_selector_ineligible_detection_position_source
)


def require_bound_detection_position_source(
    value: BoundDetectionPositionSource,
) -> BoundDetectionPositionSource:
    """Reopen all exact authorities and reject stale bound source state."""

    if (
        type(value) is not BoundDetectionPositionSource
        or value._seal is not _BOUND_DETECTION_POSITION_SOURCE_SEAL
    ):
        _fail("A sealed BoundDetectionPositionSource is required.")
    if value.authority_policy == DETECTION_POSITION_ACTIVE_AUTHORITY_POLICY_ID:
        loader = load_persisted_detection_position_source
    elif value.authority_policy == DETECTION_POSITION_CANARY_AUTHORITY_POLICY_ID:
        loader = load_persisted_selector_ineligible_detection_position_source
    else:
        _fail(
            f"Unsupported detection position authority policy: {value.authority_policy!r}."
        )
    current = (
        loader(value._analysis_zarr, value.run_path)
        if value._analysis_zarr is not None
        else _build_source(
            value._root_node,
            value.run_path,
            authority_policy=value.authority_policy,
        )
    )
    if current.source_binding_digest != value.source_binding_digest:
        _fail("Detection position source changed after binding.")
    return value


def _load_persisted_detection_position_source(
    analysis_zarr: str | Path | Any,
    run_path: str,
    *,
    authority_policy: str,
    path_validator: Any,
) -> BoundDetectionPositionSource:
    path = path_validator(run_path)
    if isinstance(analysis_zarr, (str, Path)):
        archive = Path(analysis_zarr).expanduser().resolve()
        if not archive.is_dir():
            _fail(f"Analysis Zarr does not exist: {archive}")
        receipt = validate_direct_consolidated_subtree(
            archive,
            subtree_path=path,
        ).to_json()
        direct_root = open_zarr_root(
            archive,
            mode="r",
            use_consolidated=False,
        )
        consolidated_root = open_zarr_root(
            archive,
            mode="r",
            use_consolidated=True,
        )
        direct = _build_source(
            direct_root,
            path,
            direct_consolidated_evidence=receipt,
            authority_policy=authority_policy,
        )
        consolidated = _build_source(
            consolidated_root,
            path,
            direct_consolidated_evidence=receipt,
            authority_policy=authority_policy,
        )
        if direct.source_binding_digest != consolidated.source_binding_digest:
            _fail("Direct and consolidated canonical detection evidence disagrees.")
        object.__setattr__(direct, "_analysis_zarr", archive)
        return direct
    return _build_source(
        analysis_zarr,
        path,
        authority_policy=authority_policy,
    )


__all__ = [
    "BoundDetectionPositionSource",
    "DETECTION_POSITION_ACTIVE_AUTHORITY_POLICY_ID",
    "DETECTION_POSITION_CANARY_AUTHORITY_POLICY_ID",
    "DETECTION_POSITION_CANARY_PRODUCTION_SELECTOR_ACTIVATION",
    "DETECTION_POSITION_SOURCE_SCHEMA_ID",
    "DETECTION_POSITION_SOURCE_SCHEMA_VERSION",
    "DETECTION_POSITION_VALIDITY_POLICY_ID",
    "DetectionPositionSourceError",
    "bind_detection_position_source",
    "bind_selector_ineligible_detection_position_source",
    "load_persisted_detection_position_source",
    "load_persisted_selector_ineligible_detection_position_source",
    "require_bound_detection_position_source",
]
