"""Strict subject-mask centroid sources for subject-position expressions.

This module is deliberately a position-source adapter, not a mask reader.  The
canonical subject-mask coordinate publication modules verify the run, source
rowset, coordinate records, and persisted centroid surface.  This adapter adds
the family-selector/currentness check, binds the anatomy profile, and projects
the persisted ROI-local centroids through the already sealed continuous
ROI-to-source-camera transform chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.shared.anatomy_profile import (
    AnatomyProfile,
    anatomy_profile_sha256,
    load_anatomy_profile,
    source_binding_sha256,
    validate_source_binding,
)
from fisheye.shared.directed_transform_chain import (
    apply_bound_directed_transform_chain,
    require_bound_directed_transform_chain,
)
from fisheye.shared.subject_mask_coordinate_publication import (
    load_persisted_subject_mask_coordinate_surfaces,
)
from fisheye.shared.refined_subject_mask_coordinate_publication import (
    load_persisted_refined_subject_mask_coordinate_surfaces,
)
from fisheye.shared.subject_position_expression import (
    ComponentSourceBinding,
    MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
    PointExpressionBindings,
    SUBJECT_BODY_MASK_CENTROID_ESTIMATOR_ID,
)
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_io import open_zarr_root


RAW_SUBJECT_MASK_SOURCE_KIND = "raw"
REFINED_SUBJECT_MASK_SOURCE_KIND = "refined"
SUBJECT_MASK_SOURCE_MODALITY = "subject_mask"

_SOURCE_CONFIG: dict[str, str] = {
    RAW_SUBJECT_MASK_SOURCE_KIND: "subject_mask_runs",
    REFINED_SUBJECT_MASK_SOURCE_KIND: "refined_subject_masks_runs",
}
_BOUND_SOURCE_SEAL = object()
_REQUIRED_ROLES_BY_ESTIMATOR = {
    MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID: (
        "swim_bladder",
        "eye_left",
        "eye_right",
    ),
    SUBJECT_BODY_MASK_CENTROID_ESTIMATOR_ID: ("subject_body",),
}


class SubjectMaskPositionSourceError(ValueError):
    """Raised when a subject-mask position source is not an exact authority."""


def _fail(message: str) -> None:
    raise SubjectMaskPositionSourceError(message)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _array(node: Any, *, name: str) -> np.ndarray:
    try:
        value = np.asarray(node[...])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise SubjectMaskPositionSourceError(
            f"Unable to read persisted subject-mask {name}."
        ) from exc
    return value


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _record_digest(value: Any) -> str | None:
    result = getattr(value, "record_sha256", None)
    return None if result is None else str(result)


def _node(root: Any, path: str) -> Any:
    try:
        return root[path]
    except (KeyError, IndexError, TypeError) as exc:
        raise SubjectMaskPositionSourceError(f"Missing persisted node {path!r}.") from exc


def _attrs(node: Any, *, path: str) -> Mapping[str, Any]:
    value = getattr(node, "attrs", None)
    if value is None:
        _fail(f"Persisted node {path!r} has no attributes.")
    try:
        return dict(value)
    except (TypeError, ValueError) as exc:
        raise SubjectMaskPositionSourceError(
            f"Persisted attributes at {path!r} are not readable."
        ) from exc


def _run_name(run_path: str, family: str) -> str:
    prefix = f"{family}/"
    if not isinstance(run_path, str) or not run_path.startswith(prefix):
        _fail(
            f"run_path must be an explicitly named run under {family!r}; "
            "no cross-family fallback is permitted."
        )
    run_name = run_path[len(prefix) :]
    if not run_name or "/" in run_name:
        _fail(f"run_path {run_path!r} is not one canonical named run.")
    return run_name


def _selector_evidence(root: Any, family: str, run_name: str) -> dict[str, Any]:
    parent_path = family
    parent_attrs = _attrs(_node(root, parent_path), path=parent_path)
    required = ("latest", "latest_complete")
    missing = [key for key in required if key not in parent_attrs]
    if missing:
        _fail(f"Family selector {family!r} is missing {missing!r}.")
    for key in required:
        if parent_attrs[key] != run_name:
            _fail(
                f"Family selector {family!r}.{key} does not select the named "
                f"run {run_name!r}."
            )
    authoritative = parent_attrs.get("authoritative_run")
    if authoritative is not None and authoritative != run_name:
        _fail(
            f"Family selector {family!r}.authoritative_run is stale for {run_name!r}."
        )
    return {
        "family": family,
        "run_name": run_name,
        "latest": parent_attrs["latest"],
        "latest_complete": parent_attrs["latest_complete"],
        "authoritative_run": authoritative,
    }


def _profile(value: AnatomyProfile | Mapping[str, Any] | str | Path) -> AnatomyProfile:
    if isinstance(value, AnatomyProfile):
        return value
    if isinstance(value, (str, Path)):
        return load_anatomy_profile(value)
    return AnatomyProfile.from_mapping(value)


def _surface_loader(source_kind: str) -> Any:
    if source_kind == RAW_SUBJECT_MASK_SOURCE_KIND:
        return load_persisted_subject_mask_coordinate_surfaces
    if source_kind == REFINED_SUBJECT_MASK_SOURCE_KIND:
        return load_persisted_refined_subject_mask_coordinate_surfaces
    _fail(f"Unsupported subject-mask source_kind {source_kind!r}.")


def _validated_binding(
    profile: AnatomyProfile,
    binding_id: str,
) -> tuple[dict[str, Any], dict[str, str], tuple[str, ...]]:
    binding = validate_source_binding(profile, profile.binding(binding_id))
    source_schema = binding["source_schema"]
    if source_schema.get("modality") != SUBJECT_MASK_SOURCE_MODALITY:
        _fail("The anatomy binding does not declare modality='subject_mask'.")
    if source_schema.get("authority") != "declared_schema":
        _fail("Subject-mask position sources require a declared source schema.")
    labels = tuple(source_schema.get("labels", ()))
    if not labels or len(set(labels)) != len(labels):
        _fail("The declared subject-mask source schema has invalid labels.")
    role_bindings: dict[str, str] = {}
    for item in binding["role_bindings"]:
        role_bindings[str(item["role_id"])] = str(item["source_label"])
    if set(role_bindings.values()) != set(labels):
        _fail("The anatomy role binding must cover the exact declared source labels.")
    return binding, role_bindings, labels


def _surface_descriptor_node(surfaces: Any) -> Any:
    try:
        return surfaces.centroid_xy.coordinate_node
    except AttributeError as exc:
        raise SubjectMaskPositionSourceError(
            "The sealed subject-mask loader did not expose centroid_xy."
        ) from exc


def _load_surface_arrays(
    root: Any,
    run_path: str,
    surfaces: Any,
    *,
    labels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centroid_xy = _array(_surface_descriptor_node(surfaces), name="centroid_xy")
    centroid_valid = _array(
        _node(root, f"{run_path}/metrics/centroid_valid"),
        name="centroid_valid",
    )
    available_channels = _array(
        _node(root, f"{run_path}/available_channels"),
        name="available_channels",
    )
    instance_key = _array(_node(root, f"{run_path}/instance_key"), name="instance_key")
    source_frame = _array(
        _node(root, f"{run_path}/source_acquisition_frame_index"),
        name="source_acquisition_frame_index",
    )
    if centroid_xy.dtype != np.dtype("float32") or centroid_xy.ndim != 3:
        _fail("Persisted centroid_xy must be float32 with shape [N, C, 2].")
    if centroid_xy.shape[2] != 2:
        _fail("Persisted centroid_xy must have exactly two continuous coordinates.")
    if centroid_valid.dtype != np.dtype(bool) or centroid_valid.shape != centroid_xy.shape[:2]:
        _fail("Persisted centroid_valid must be bool with shape [N, C].")
    if available_channels.dtype != np.dtype(bool) or available_channels.shape != (centroid_xy.shape[1],):
        _fail("Persisted available_channels must be bool with shape [C].")
    if instance_key.dtype != np.dtype("uint64") or instance_key.shape != (centroid_xy.shape[0],):
        _fail("Persisted instance_key must be uint64 with exact row cardinality.")
    if source_frame.dtype != np.dtype("int64") or source_frame.shape != (centroid_xy.shape[0],):
        _fail(
            "Persisted source_acquisition_frame_index must be int64 with exact row "
            "cardinality."
        )
    if tuple(labels) != tuple(getattr(surfaces.context, "labels", ())):
        _fail("Sealed subject-mask coordinates and source-schema labels disagree.")
    if not np.all(np.isfinite(centroid_xy)):
        _fail("Persisted centroid_xy contains non-finite values.")
    if np.any(centroid_xy[~centroid_valid] != 0.0):
        _fail("Invalid persisted centroid rows must use the exact zero sentinel.")
    return centroid_xy, centroid_valid, available_channels, instance_key, source_frame


def _assert_surface_identity(surfaces: Any, *, rows: int, run_path: str) -> Any:
    context = getattr(surfaces, "context", None)
    if context is None or getattr(context, "run_path", None) != run_path:
        _fail("Sealed subject-mask coordinates are not bound to the requested run.")
    row_identity = getattr(context, "row_identity", None)
    if row_identity is None or int(row_identity.leading_dimension) != rows:
        _fail("Subject-mask row identity does not match the persisted centroid rows.")
    chain = getattr(context, "continuous_chain", None)
    if chain is None:
        _fail("Subject-mask coordinates do not expose a continuous transform chain.")
    try:
        chain = require_bound_directed_transform_chain(chain)
    except Exception as exc:
        raise SubjectMaskPositionSourceError(
            "Subject-mask continuous transform chain is not sealed."
        ) from exc
    if chain.descriptor_pixel_convention != "continuous":
        _fail("Subject-mask centroid descriptor is not continuous ROI-local XY.")
    if chain.source_camera_pixel_convention != "continuous":
        _fail("Subject-mask source-camera frame is not continuous XY.")
    return row_identity, chain


def _surface_digest(surfaces: Any) -> dict[str, Any]:
    return {
        "context": _record_digest(surfaces.context.context_record),
        "inventory": _record_digest(surfaces.inventory),
        "derivation": _record_digest(getattr(surfaces, "derivation", None)),
    }


def _evidence_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_thaw(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_bound_source(
    archive_path: Path,
    direct_root: Any,
    consolidated_root: Any,
    *,
    source_kind: str,
    run_path: str,
    profile: AnatomyProfile,
    binding_id: str,
    required_role_ids: Sequence[str] | None,
) -> "BoundSubjectMaskPositionSource":
    try:
        family = _SOURCE_CONFIG[source_kind]
    except KeyError:
        _fail(f"Unsupported subject-mask source_kind {source_kind!r}.")
    loader = _surface_loader(source_kind)
    run_name = _run_name(run_path, family)
    direct_selector = _selector_evidence(direct_root, family, run_name)
    consolidated_selector = _selector_evidence(consolidated_root, family, run_name)
    if direct_selector != consolidated_selector:
        _fail("Direct and consolidated family selectors disagree.")
    try:
        direct_surfaces = loader(direct_root, run_path)
        consolidated_surfaces = loader(consolidated_root, run_path)
    except Exception as exc:
        raise SubjectMaskPositionSourceError(
            f"The sealed {source_kind} subject-mask coordinate loader rejected {run_path!r}."
        ) from exc
    direct_surface_digest = _surface_digest(direct_surfaces)
    consolidated_surface_digest = _surface_digest(consolidated_surfaces)
    if direct_surface_digest != consolidated_surface_digest:
        _fail("Direct and consolidated subject-mask coordinate evidence disagrees.")
    binding, role_mapping, declared_labels = _validated_binding(profile, binding_id)
    persisted_labels = tuple(getattr(direct_surfaces.context, "labels", ()))
    if persisted_labels != declared_labels:
        _fail(
            "Persisted subject-mask label order differs from the exact declared "
            "source schema."
        )
    if len(set(persisted_labels)) != len(persisted_labels):
        _fail("Persisted subject-mask labels contain duplicates.")
    centroid_xy, centroid_valid, available, instance_key, source_frame = _load_surface_arrays(
        direct_root,
        run_path,
        direct_surfaces,
        labels=persisted_labels,
    )
    consolidated_arrays = _load_surface_arrays(
        consolidated_root,
        run_path,
        consolidated_surfaces,
        labels=persisted_labels,
    )
    for left, right, label in zip(
        (centroid_xy, centroid_valid, available, instance_key, source_frame),
        consolidated_arrays,
        ("centroid_xy", "centroid_valid", "available_channels", "instance_key", "source_frame"),
    ):
        if not np.array_equal(left, right):
            _fail(f"Direct and consolidated {label} evidence disagree.")
    row_identity, chain = _assert_surface_identity(
        direct_surfaces, rows=centroid_xy.shape[0], run_path=run_path
    )
    row_contract = getattr(row_identity, "contract", None)
    if row_contract is None or getattr(row_contract, "mode", None) != "instance_key":
        _fail("Subject-mask position sources require instance_key row identity.")
    source_row_index = np.arange(centroid_xy.shape[0], dtype=np.int64)
    source_label_indices = {label: index for index, label in enumerate(persisted_labels)}
    required_roles = tuple(role_mapping) if required_role_ids is None else tuple(required_role_ids)
    if not required_roles or len(set(required_roles)) != len(required_roles):
        _fail("required_role_ids must name one or more distinct anatomy roles.")
    unknown_roles = sorted(set(required_roles) - set(role_mapping))
    if unknown_roles:
        _fail(f"The anatomy binding has no source mapping for roles {unknown_roles!r}.")
    for role_id in required_roles:
        label = role_mapping[role_id]
        if not bool(available[source_label_indices[label]]):
            _fail(f"Required subject-mask channel {label!r} is unavailable.")
    try:
        projected = apply_bound_directed_transform_chain(
            centroid_xy,
            chain,
            row_identity=chain.row_identity if chain.row_identity is not None else None,
        )
    except Exception as exc:
        raise SubjectMaskPositionSourceError(
            "Unable to project ROI-local subject-mask centroids to source-camera XY."
        ) from exc
    projected = np.asarray(projected)
    if projected.shape != centroid_xy.shape or not np.all(np.isfinite(projected)):
        _fail("Projected subject-mask centroids are not finite with the source shape.")
    role_bindings = {
        role_id: ComponentSourceBinding(
            centroids=projected[:, source_label_indices[label], :],
            valid=centroid_valid[:, source_label_indices[label]],
        )
        for role_id, label in role_mapping.items()
    }
    evidence = {
        "metadata": {
            "direct_mode": "unconsolidated",
            "consolidated_mode": "consolidated",
            "direct_consolidated_subtree": run_path,
        },
        "family_selector_direct": direct_selector,
        "family_selector_consolidated": consolidated_selector,
        "surface_direct": direct_surface_digest,
        "surface_consolidated": consolidated_surface_digest,
    }
    source_payload = {
        "source_kind": source_kind,
        "run_path": run_path,
        "authority_evidence": _evidence_digest(evidence),
        "row_identity": getattr(row_identity, "record_sha256", None),
        "instance_key": _array_digest(instance_key),
        "source_acquisition_frame_index": _array_digest(source_frame),
        "source_row_index": _array_digest(source_row_index),
        "centroid_xy": _array_digest(projected),
        "centroid_valid": _array_digest(centroid_valid),
        "available_channels": _array_digest(available),
    }
    return BoundSubjectMaskPositionSource(
        source_modality=SUBJECT_MASK_SOURCE_MODALITY,
        source_kind=source_kind,
        run_path=run_path,
        row_identity=row_identity,
        instance_key=instance_key,
        source_acquisition_frame_index=source_frame,
        source_row_index=source_row_index,
        source_camera_frame=chain.source_camera_frame_authority,
        labels=persisted_labels,
        role_mapping=role_mapping,
        source_binding_record=binding,
        source_binding_digest=source_binding_sha256(binding),
        expression_bindings=PointExpressionBindings(components=role_bindings),
        centroid_xy_source_camera=projected,
        centroid_valid=centroid_valid,
        available_channels=available,
        direct_consolidated_evidence=evidence,
        source_payload_digest=_evidence_digest(source_payload),
        anatomy_profile_digest=anatomy_profile_sha256(profile),
        _analysis_zarr=archive_path,
        _anatomy_profile_payload=profile.payload,
        _binding_id=binding_id,
        _required_role_ids=required_roles,
        _seal=_BOUND_SOURCE_SEAL,
    )


@dataclass(frozen=True, init=False)
class BoundSubjectMaskPositionSource:
    """Sealed source-camera centroid bindings for one canonical mask run."""

    source_modality: str
    source_kind: str
    run_path: str
    row_identity: Any = field(repr=False, compare=False)
    instance_key: np.ndarray = field(repr=False, compare=False)
    source_acquisition_frame_index: np.ndarray = field(repr=False, compare=False)
    source_row_index: np.ndarray = field(repr=False, compare=False)
    source_camera_frame: Any = field(repr=False, compare=False)
    labels: tuple[str, ...]
    role_mapping: Mapping[str, str]
    source_binding_record: Mapping[str, Any] = field(repr=False, compare=False)
    source_binding_digest: str
    expression_bindings: PointExpressionBindings = field(repr=False, compare=False)
    centroid_xy_source_camera: np.ndarray = field(repr=False, compare=False)
    centroid_valid: np.ndarray = field(repr=False, compare=False)
    available_channels: np.ndarray = field(repr=False, compare=False)
    direct_consolidated_evidence: Mapping[str, Any] = field(repr=False, compare=False)
    source_payload_digest: str
    anatomy_profile_digest: str
    _analysis_zarr: Path = field(repr=False, compare=False)
    _anatomy_profile_payload: Mapping[str, Any] = field(repr=False, compare=False)
    _binding_id: str = field(repr=False, compare=False)
    _required_role_ids: tuple[str, ...] = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self, *, _seal: object | None = None, **values: Any) -> None:
        if _seal is not _BOUND_SOURCE_SEAL:
            _fail("Subject-mask position sources cannot be constructed directly.")
        for name, value in values.items():
            if name in {"role_mapping", "source_binding_record", "direct_consolidated_evidence"}:
                value = _freeze(value)
            elif name in {
                "instance_key",
                "source_acquisition_frame_index",
                "source_row_index",
                "centroid_xy_source_camera",
                "centroid_valid",
                "available_channels",
            }:
                value = np.asarray(value)
                value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _seal)

    def revalidate(self) -> "BoundSubjectMaskPositionSource":
        """Reload direct and consolidated evidence and reject stale bindings."""

        current = load_subject_mask_position_source(
            self._analysis_zarr,
            run_path=self.run_path,
            source_kind=self.source_kind,
            anatomy_profile=self._anatomy_profile_payload,
            binding_id=self._binding_id,
            required_role_ids=self._required_role_ids,
        )
        if current.source_payload_digest != self.source_payload_digest:
            _fail("Bound subject-mask position source changed after it was sealed.")
        if current.source_binding_digest != self.source_binding_digest:
            _fail("Bound subject-mask source binding changed after it was sealed.")
        return current

    @property
    def ordered_labels(self) -> tuple[str, ...]:
        """Persisted channel order, retained separately from role order."""

        return self.labels

    @property
    def anatomy_profile(self) -> AnatomyProfile:
        """Exact anatomy authority used to seal this source."""

        return AnatomyProfile.from_mapping(self._anatomy_profile_payload)

    @property
    def binding_id(self) -> str:
        """Exact anatomy source-binding identity used by this adapter."""

        return self._binding_id


def load_subject_mask_position_source(
    analysis_zarr: str | Path,
    *,
    run_path: str,
    source_kind: str,
    anatomy_profile: AnatomyProfile | Mapping[str, Any] | str | Path,
    binding_id: str,
    required_role_ids: Sequence[str] | None = None,
) -> BoundSubjectMaskPositionSource:
    """Load one explicitly named, current, complete subject-mask source.

    ``run_path`` is mandatory and is never resolved from a ``latest`` selector.
    The selected family must itself point at that exact run in both direct and
    consolidated metadata.  Raw and refined source kinds are intentionally
    separate and never substitute for one another.
    """

    if not run_path:
        _fail("An explicit run_path is required; latest fallback is disabled.")
    if not isinstance(analysis_zarr, (str, Path)):
        _fail("analysis_zarr must be the path to the canonical analysis Zarr.")
    archive_path = Path(analysis_zarr).expanduser().resolve()
    profile = _profile(anatomy_profile)
    direct_root = open_zarr_root(archive_path, mode="r", use_consolidated=False)
    consolidated_root = open_zarr_root(archive_path, mode="r", use_consolidated=True)
    if source_kind not in _SOURCE_CONFIG:
        _fail(f"Unsupported subject-mask source_kind {source_kind!r}.")
    validate_direct_consolidated_subtree(archive_path, subtree_path=run_path)
    return _read_bound_source(
        archive_path,
        direct_root,
        consolidated_root,
        source_kind=source_kind,
        run_path=run_path,
        profile=profile,
        binding_id=binding_id,
        required_role_ids=required_role_ids,
    )


def require_bound_subject_mask_position_source(
    value: BoundSubjectMaskPositionSource,
) -> BoundSubjectMaskPositionSource:
    """Require a sealed source and revalidate its immutable dependencies."""

    if type(value) is not BoundSubjectMaskPositionSource or value._seal is not _BOUND_SOURCE_SEAL:
        _fail("A sealed BoundSubjectMaskPositionSource is required.")
    return value.revalidate()


def load_subject_mask_position_source_for_estimator(
    analysis_zarr: str | Path,
    *,
    run_path: str,
    source_kind: str,
    anatomy_profile: AnatomyProfile | Mapping[str, Any] | str | Path,
    binding_id: str,
    estimator_id: str,
) -> BoundSubjectMaskPositionSource:
    """Bind only the anatomy roles required by one registered mask estimator."""

    try:
        required_roles = _REQUIRED_ROLES_BY_ESTIMATOR[estimator_id]
    except KeyError:
        _fail(f"Unsupported subject-mask position estimator {estimator_id!r}.")
    return load_subject_mask_position_source(
        analysis_zarr,
        run_path=run_path,
        source_kind=source_kind,
        anatomy_profile=anatomy_profile,
        binding_id=binding_id,
        required_role_ids=required_roles,
    )


__all__ = [
    "BoundSubjectMaskPositionSource",
    "RAW_SUBJECT_MASK_SOURCE_KIND",
    "REFINED_SUBJECT_MASK_SOURCE_KIND",
    "SUBJECT_MASK_SOURCE_MODALITY",
    "SubjectMaskPositionSourceError",
    "load_subject_mask_position_source",
    "load_subject_mask_position_source_for_estimator",
    "require_bound_subject_mask_position_source",
]
