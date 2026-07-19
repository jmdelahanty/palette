"""Strict selected-camera calibration snapshots for future Palette writers.

The schema intentionally has one authority and no compatibility fallbacks::

    analysis/stimulus_runs/<stimulus_run>/calibration
      attrs:
        schema_id = "palette.selected_calibration_snapshot"
        schema_version = 1
        stimulus_run = <stimulus_run>
        active_camera_id = <camera_id>
        active_camera_calibration_ref =
          "analysis/stimulus_runs/<stimulus_run>/calibration/<camera_id>"
        active_camera_transform_ref =
          "analysis/stimulus_runs/<stimulus_run>/calibration/<camera_id>/homography_matrix"
        active_camera_transform_sha256 = <directed-transform metadata digest>
        selected_calibration_manifest = <canonical manifest mapping>
        selected_calibration_manifest_sha256 = <manifest digest>

      <camera_id>
        attrs:
          schema_id = "palette.camera_calibration_snapshot"
          schema_version = 1
          camera_id = <camera_id>
          native_width_px = <positive integer>
          native_height_px = <positive integer>
          pixels_per_mm_camera = <optional positive float>
          pixel_to_mm = <optional reciprocal of pixels_per_mm_camera>
          pixel_to_mm_derivation = "reciprocal_of_pixels_per_mm_camera_v1"
          pixels_per_mm_projector = <optional positive float>
          z_eff_mm = <optional positive float>

        homography_matrix
          attrs populated by ``stamp_directed_homography``

    analysis/stimulus_runs/<stimulus_run>/display_snapshot
      attrs:
        selected_output_name = <output identity>
        selected_output_geometry = <canonical WxH+X+Y>
        selected_output_transform_token = "normal"
        source_display_dataset_path = "/display_snapshot/selected_output_block"
        source_display_dataset_sha256 = <SHA-256 of exact UTF-8 H5 payload>
        source_display_dataset_digest_canonicalization = "utf8_bytes_v1"

All refs are canonical archive-relative paths.  Consumers must name the
stimulus run, expected camera, direction, and exact extents.  This module never
resolves ``latest``, scans camera groups, reads root/global calibration, derives
missing source values, or merges fields from different sources.

For a Citrus camera-view-to-final-canvas homography, the target extent authority
is the selected run's mirrored
``display_snapshot@selected_output_geometry`` field (for example,
``1920x1080+3840+0``), not a dimension reconstructed from coordinate ranges or
root Zarr metadata.

The manifest binds these camera/display fields, transform ref/digest, exact
numeric and YAML matrix identities, their equality, the H5-declared
source/destination frames, axes, origin, image space, camera and canvas, plus
calibration-artifact lineage in one canonical digest.  Selected-calibration v1
admits only source-camera-image to final stimulus-canvas direction.  Native
dimensions and calibration scalars come only from a digest-bound, exact
``active_camera_id`` selection in ``/calibration_snapshot/arena_config_json``;
duplicate camera-group scalar attrs must agree.  ``pixel_to_mm`` is derived only
as the declared reciprocal of the bound camera pixels-per-mm value.

The writer helper ``stamp_selected_calibration_snapshot`` preflights every
mutation target and validates sealed, builder-produced camera, display, and
homography source evidence before stamping any attrs.  Attr writes use exact
pre-call snapshots and roll every target back if any write fails; an incomplete
rollback is reported explicitly and never returned as success.  It does not
open an H5 file.  An
importer must call the three ``build_selected_*_source_evidence_from_h5_values``
helpers with values read from the exact selected H5 nodes before any Zarr
mutation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
import hashlib
import json
import math
from numbers import Integral, Real
import re
from typing import Any, Mapping

import numpy as np
import yaml

from fisheye.shared.directed_transform import (
    BoundDirectedHomography,
    DirectedTransformError,
    TransformReferenceExtent,
    build_directed_homography,
    directed_homography_attrs,
    homography_matrix_sha256,
    load_bound_directed_homography,
)


SELECTED_CALIBRATION_SCHEMA_ID = "palette.selected_calibration_snapshot"
SELECTED_CALIBRATION_SCHEMA_VERSION = 1
CAMERA_CALIBRATION_SCHEMA_ID = "palette.camera_calibration_snapshot"
CAMERA_CALIBRATION_SCHEMA_VERSION = 1
SELECTED_CALIBRATION_MANIFEST_SCHEMA_ID = "palette.selected_calibration_manifest"
SELECTED_CALIBRATION_MANIFEST_SCHEMA_VERSION = 1
SOURCE_HOMOGRAPHY_SEMANTICS_SCHEMA_ID = (
    "palette.source_homography_semantics"
)
SOURCE_HOMOGRAPHY_SEMANTICS_SCHEMA_VERSION = 1
SELECTED_CAMERA_SOURCE_SCHEMA_ID = "palette.selected_camera_source_evidence"
SELECTED_CAMERA_SOURCE_SCHEMA_VERSION = 1
SELECTED_DISPLAY_SOURCE_SCHEMA_ID = "palette.selected_display_source_evidence"
SELECTED_DISPLAY_SOURCE_SCHEMA_VERSION = 1
SELECTED_HOMOGRAPHY_SOURCE_SCHEMA_ID = (
    "palette.selected_homography_source_evidence"
)
SELECTED_HOMOGRAPHY_SOURCE_SCHEMA_VERSION = 1

CANONICAL_HOMOGRAPHY_FROM_SPACE_ID = "source_camera_image_px"
CANONICAL_HOMOGRAPHY_TO_SPACE_ID = "stimulus_canvas_px"
CANONICAL_SOURCE_FRAME = "camera_view_px"
CANONICAL_DEST_FRAME = "final_display_canvas_px"
CANONICAL_AXES = "x_right_y_down"
CANONICAL_COORDINATE_ORIGIN = "top_left"
CANONICAL_IMAGE_SPACE = "raw"
CANONICAL_MATRIX_AGREEMENT = "canonical_float64_exact_v1"

STIMULUS_RUN_ATTR = "stimulus_run"
ACTIVE_CAMERA_ID_ATTR = "active_camera_id"
ACTIVE_CAMERA_CALIBRATION_REF_ATTR = "active_camera_calibration_ref"
ACTIVE_CAMERA_TRANSFORM_REF_ATTR = "active_camera_transform_ref"
ACTIVE_CAMERA_TRANSFORM_SHA256_ATTR = "active_camera_transform_sha256"
CAMERA_ID_ATTR = "camera_id"
HOMOGRAPHY_ARRAY_NAME = "homography_matrix"
DISPLAY_SNAPSHOT_GROUP_NAME = "display_snapshot"
SELECTED_OUTPUT_NAME_ATTR = "selected_output_name"
SELECTED_OUTPUT_GEOMETRY_ATTR = "selected_output_geometry"
SELECTED_OUTPUT_TRANSFORM_TOKEN_ATTR = "selected_output_transform_token"
SOURCE_DISPLAY_DATASET_PATH_ATTR = "source_display_dataset_path"
SOURCE_DISPLAY_DATASET_SHA256_ATTR = "source_display_dataset_sha256"
SOURCE_DISPLAY_DATASET_DIGEST_CANONICALIZATION_ATTR = (
    "source_display_dataset_digest_canonicalization"
)
SOURCE_DISPLAY_DATASET_DIGEST_CANONICALIZATION = "utf8_bytes_v1"
SOURCE_DISPLAY_GROUP_PATH = "/display_snapshot"
SOURCE_DISPLAY_DATASET_PATH = "/display_snapshot/selected_output_block"
SOURCE_DISPLAY_ATTRS_DIGEST_CANONICALIZATION = "canonical_json_sort_keys_v1"
SOURCE_DISPLAY_EVIDENCE_ATTR = "source_display_evidence"
SOURCE_DISPLAY_EVIDENCE_DIGEST_SUFFIX = "_sha256"
SOURCE_ARENA_CONFIG_DATASET_PATH = "/calibration_snapshot/arena_config_json"
SOURCE_ARENA_CONFIG_DIGEST_CANONICALIZATION = "utf8_bytes_v1"
SOURCE_CAMERA_RECORD_DIGEST_CANONICALIZATION = "canonical_json_sort_keys_v1"
SOURCE_CAMERA_ATTRS_DIGEST_CANONICALIZATION = "canonical_json_sort_keys_v1"
PIXEL_TO_MM_DERIVATION = "reciprocal_of_pixels_per_mm_camera_v1"
Z_EFF_MM_SOURCE_FIELD = "calculated_z_eff_mm"
Z_EFF_MM_DERIVATION = "positive_identity_else_unavailable_v1"
NUMERIC_HOMOGRAPHY_PAYLOAD_SOURCE = "runtime_arena_config.homography_matrix"
YAML_HOMOGRAPHY_PAYLOAD_SOURCE = "resolved_calibration_artifact_file"
YAML_HOMOGRAPHY_SERIALIZATION_FORMAT = "opencv_yml"
SOURCE_HOMOGRAPHY_ATTRS_DIGEST_CANONICALIZATION = "canonical_json_sort_keys_v1"
SOURCE_HOMOGRAPHY_YAML_DIGEST_CANONICALIZATION = "utf8_bytes_v1"
SOURCE_HOMOGRAPHY_EVIDENCE_ATTR = "source_homography_evidence"
SOURCE_HOMOGRAPHY_EVIDENCE_DIGEST_SUFFIX = "_sha256"
SUPPORTED_OUTPUT_TRANSFORM_TOKEN = "normal"
SELECTED_CALIBRATION_MANIFEST_ATTR = "selected_calibration_manifest"
SELECTED_CALIBRATION_MANIFEST_DIGEST_SUFFIX = "_sha256"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LOWER_HEX_RE = re.compile(r"^[0-9a-f]+$")
_OUTPUT_GEOMETRY_RE = re.compile(
    r"^(?P<width>[1-9][0-9]*)x(?P<height>[1-9][0-9]*)"
    r"(?P<x>[+-](?:0|[1-9][0-9]*))(?P<y>[+-](?:0|[1-9][0-9]*))$"
)
_XRANDR_OUTPUT_HEADER_RE = re.compile(r"^\S+\s+(?:connected|disconnected)(?:\s|$)")
_SELECTED_OUTPUT_HEADER_RE = re.compile(
    r"^(?P<name>\S+) connected "
    r"(?P<geometry>[1-9][0-9]*x[1-9][0-9]*[+-](?:0|[1-9][0-9]*)"
    r"[+-](?:0|[1-9][0-9]*)) "
    r"\((?P<transform_raw>[^()\r\n]+)\)(?:\s.*)?$"
)

# A nominal ``Verified*`` type is not sufficient to mark evidence as having
# passed the exact-source builders: dataclass subclasses otherwise expose a
# public constructor with all of the base fields.  This private, process-local
# token is an API-misuse boundary (not a hostile-code security mechanism).
_VERIFIED_SOURCE_EVIDENCE_SEAL = object()


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that fails closed on duplicate mapping keys."""


def _construct_unique_yaml_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_yaml_mapping,
)
_SCALAR_ATTRS = (
    "pixels_per_mm_camera",
    "pixel_to_mm",
    "pixels_per_mm_projector",
    "z_eff_mm",
)
_SOURCE_CAMERA_SCALAR_FIELDS = (
    "pixels_per_mm_camera",
    "pixels_per_mm_projector",
    "real_world_ref_mm",
)
_SOURCE_ARTIFACT_FIELDS = frozenset(
    {
        "source_h5_path",
        "source_homography_semantics",
        "source_homography_semantics_sha256",
        "homography_artifact_path",
        "homography_artifact_checksum_algorithm",
        "homography_artifact_checksum",
        "homography_artifact_size_bytes",
        "homography_artifact_mtime_unix_ns",
        "homography_provenance_schema",
    }
)
_SOURCE_HOMOGRAPHY_SEMANTICS_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "source_frame",
        "dest_frame",
        "axes",
        "coordinate_origin",
        "image_space",
        "camera_id",
        "canvas_name",
        "numeric_dataset_path",
        "yaml_dataset_path",
        "numeric_matrix_sha256",
        "yaml_matrix_sha256",
        "matrix_agreement",
        "numeric_payload_source",
        "yaml_payload_source",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "stimulus_run",
        "camera_id",
        "camera_calibration_ref",
        "transform_ref",
        "transform_sha256",
        "matrix_sha256",
        "camera_calibration",
        "display_snapshot",
        "source_artifact",
        "source_camera",
        "source_display",
        "source_homography",
    }
)
_CAMERA_MANIFEST_FIELDS = frozenset(
    {
        "native_width_px",
        "native_height_px",
        "pixels_per_mm_camera",
        "pixel_to_mm",
        "pixel_to_mm_derivation",
        "pixels_per_mm_projector",
        "z_eff_mm",
    }
)
_SELECTED_CAMERA_SOURCE_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "source_h5_path",
        "arena_config_dataset_path",
        "arena_config_dataset_sha256",
        "arena_config_dataset_digest_canonicalization",
        "active_camera_id",
        "selected_camera_record_index",
        "selected_camera_record",
        "selected_camera_record_sha256",
        "selected_camera_record_digest_canonicalization",
        "camera_group_path",
        "camera_group_scalar_attrs",
        "camera_group_scalar_attrs_sha256",
        "camera_group_scalar_attrs_digest_canonicalization",
        "z_eff_mm_source_field",
        "z_eff_mm_source_value",
        "z_eff_mm_derivation",
    }
)
_DISPLAY_MANIFEST_FIELDS = frozenset(
    {
        "ref",
        "selected_output_name",
        "selected_output_geometry",
        "selected_output_transform_token",
        "width_px",
        "height_px",
        "offset_x_px",
        "offset_y_px",
        "source_h5_dataset_path",
        "source_h5_dataset_sha256",
        "source_h5_dataset_digest_canonicalization",
    }
)
_DISPLAY_SOURCE_ATTR_FIELDS = frozenset(
    {
        "selected_output_name",
        "selected_output_connection_state",
        "selected_output_geometry",
        "selected_output_transform_token",
        "selected_output_transform_raw",
    }
)
_SELECTED_DISPLAY_SOURCE_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "source_h5_path",
        "display_group_path",
        "display_group_attrs",
        "display_group_attrs_sha256",
        "display_group_attrs_digest_canonicalization",
        "selected_output_dataset_path",
        "selected_output_dataset_sha256",
        "selected_output_dataset_digest_canonicalization",
        "selected_output_first_line",
        "selected_output_name",
        "selected_output_connection_state",
        "selected_output_geometry",
        "selected_output_transform_token",
        "selected_output_transform_raw",
    }
)
_HOMOGRAPHY_SHARED_ATTR_FIELDS = frozenset(
    {
        "source_frame",
        "dest_frame",
        "axes",
        "coordinate_origin",
        "image_space",
        "camera_id",
        "canvas_name",
        "homography_provenance_schema",
        "homography_artifact_path",
        "homography_artifact_exists",
        "homography_artifact_checksum_algorithm",
        "homography_artifact_checksum",
        "homography_artifact_size_bytes",
        "homography_artifact_mtime_unix_ns",
    }
)
_HOMOGRAPHY_NUMERIC_ATTR_FIELDS = _HOMOGRAPHY_SHARED_ATTR_FIELDS | frozenset(
    {"homography_payload_source"}
)
_HOMOGRAPHY_YAML_ATTR_FIELDS = _HOMOGRAPHY_SHARED_ATTR_FIELDS | frozenset(
    {"homography_payload_source", "serialization_format"}
)
_SELECTED_HOMOGRAPHY_SOURCE_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "source_h5_path",
        "camera_id",
        "numeric_dataset_path",
        "numeric_matrix",
        "numeric_matrix_sha256",
        "numeric_dataset_attrs",
        "numeric_dataset_attrs_sha256",
        "numeric_dataset_attrs_digest_canonicalization",
        "yaml_dataset_path",
        "yaml_dataset_sha256",
        "yaml_dataset_digest_canonicalization",
        "yaml_matrix_sha256",
        "yaml_dataset_attrs",
        "yaml_dataset_attrs_sha256",
        "yaml_dataset_attrs_digest_canonicalization",
        "matrix_agreement",
    }
)


class SelectedCalibrationError(DirectedTransformError):
    """Raised when a selected calibration snapshot is absent or incoherent."""


@dataclass(frozen=True)
class SelectedCalibrationPaths:
    """Canonical archive-relative paths for one run/camera selection."""

    stimulus_run_path: str
    display_snapshot_path: str
    calibration_path: str
    camera_calibration_path: str
    homography_array_path: str


def _initialize_verified_source_evidence(
    instance: Any,
    base_type: type[Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    seal: object | None,
    evidence_name: str,
) -> None:
    """Initialize a sealed evidence subclass through its private factory only."""

    if seal is not _VERIFIED_SOURCE_EVIDENCE_SEAL:
        raise SelectedCalibrationError(
            f"{evidence_name} cannot be constructed directly; use its exact-H5 "
            "builder."
        )
    parsed = base_type(*args, **dict(kwargs))
    for item in fields(base_type):
        object.__setattr__(instance, item.name, getattr(parsed, item.name))
    object.__setattr__(instance, "_source_evidence_seal", seal)


@dataclass(frozen=True)
class SelectedCameraSourceEvidence:
    """Exact active-camera values parsed from one named H5 authority.

    The builder computes this record from the exact raw arena-config dataset and
    the attrs of the exact active-camera H5 group.  Parsing a persisted mapping
    validates its digests and cross-field agreement, but does not reopen H5.
    """

    source_h5_path: str
    arena_config_dataset_path: str
    arena_config_dataset_sha256: str
    arena_config_dataset_digest_canonicalization: str
    active_camera_id: str
    selected_camera_record_index: int
    selected_camera_record: Mapping[str, Any]
    selected_camera_record_sha256: str
    selected_camera_record_digest_canonicalization: str
    camera_group_path: str
    camera_group_scalar_attrs: Mapping[str, float | None]
    camera_group_scalar_attrs_sha256: str
    camera_group_scalar_attrs_digest_canonicalization: str
    z_eff_mm_source_field: str
    z_eff_mm_source_value: float | None
    z_eff_mm_derivation: str
    native_width_px: int
    native_height_px: int
    pixels_per_mm_camera: float | None
    pixels_per_mm_projector: float | None
    real_world_ref_mm: float | None
    pixel_to_mm: float | None
    z_eff_mm: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": SELECTED_CAMERA_SOURCE_SCHEMA_ID,
            "schema_version": SELECTED_CAMERA_SOURCE_SCHEMA_VERSION,
            "source_h5_path": self.source_h5_path,
            "arena_config_dataset_path": self.arena_config_dataset_path,
            "arena_config_dataset_sha256": self.arena_config_dataset_sha256,
            "arena_config_dataset_digest_canonicalization": (
                self.arena_config_dataset_digest_canonicalization
            ),
            "active_camera_id": self.active_camera_id,
            "selected_camera_record_index": self.selected_camera_record_index,
            "selected_camera_record": json.loads(
                _canonical_json(self.selected_camera_record)
            ),
            "selected_camera_record_sha256": self.selected_camera_record_sha256,
            "selected_camera_record_digest_canonicalization": (
                self.selected_camera_record_digest_canonicalization
            ),
            "camera_group_path": self.camera_group_path,
            "camera_group_scalar_attrs": dict(self.camera_group_scalar_attrs),
            "camera_group_scalar_attrs_sha256": (
                self.camera_group_scalar_attrs_sha256
            ),
            "camera_group_scalar_attrs_digest_canonicalization": (
                self.camera_group_scalar_attrs_digest_canonicalization
            ),
            "z_eff_mm_source_field": self.z_eff_mm_source_field,
            "z_eff_mm_source_value": self.z_eff_mm_source_value,
            "z_eff_mm_derivation": self.z_eff_mm_derivation,
        }


@dataclass(frozen=True, init=False)
class VerifiedSelectedCameraSourceEvidence(SelectedCameraSourceEvidence):
    """Source evidence produced only after parsing exact H5 values in the builder."""

    _source_evidence_seal: object = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(self, *args: Any, _seal: object | None = None, **kwargs: Any) -> None:
        _initialize_verified_source_evidence(
            self,
            SelectedCameraSourceEvidence,
            args,
            kwargs,
            seal=_seal,
            evidence_name="Verified selected-camera source evidence",
        )


@dataclass(frozen=True)
class SelectedDisplaySourceEvidence:
    """Digest-bound selected-output evidence persisted in the run manifest."""

    source_h5_path: str
    display_group_path: str
    display_group_attrs: Mapping[str, str]
    display_group_attrs_sha256: str
    display_group_attrs_digest_canonicalization: str
    selected_output_dataset_path: str
    selected_output_dataset_sha256: str
    selected_output_dataset_digest_canonicalization: str
    selected_output_first_line: str
    selected_output_name: str
    selected_output_connection_state: str
    selected_output_geometry: str
    selected_output_transform_token: str
    selected_output_transform_raw: str
    width_px: int
    height_px: int
    offset_x_px: int
    offset_y_px: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": SELECTED_DISPLAY_SOURCE_SCHEMA_ID,
            "schema_version": SELECTED_DISPLAY_SOURCE_SCHEMA_VERSION,
            "source_h5_path": self.source_h5_path,
            "display_group_path": self.display_group_path,
            "display_group_attrs": dict(self.display_group_attrs),
            "display_group_attrs_sha256": self.display_group_attrs_sha256,
            "display_group_attrs_digest_canonicalization": (
                self.display_group_attrs_digest_canonicalization
            ),
            "selected_output_dataset_path": self.selected_output_dataset_path,
            "selected_output_dataset_sha256": self.selected_output_dataset_sha256,
            "selected_output_dataset_digest_canonicalization": (
                self.selected_output_dataset_digest_canonicalization
            ),
            "selected_output_first_line": self.selected_output_first_line,
            "selected_output_name": self.selected_output_name,
            "selected_output_connection_state": self.selected_output_connection_state,
            "selected_output_geometry": self.selected_output_geometry,
            "selected_output_transform_token": self.selected_output_transform_token,
            "selected_output_transform_raw": self.selected_output_transform_raw,
        }

    def digest(self) -> str:
        return _canonical_json_sha256(
            self.to_dict(),
            field_name="source_display",
        )


@dataclass(frozen=True, init=False)
class VerifiedSelectedDisplaySourceEvidence(SelectedDisplaySourceEvidence):
    """Display evidence derived from exact H5 bytes and group attrs."""

    _source_evidence_seal: object = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(self, *args: Any, _seal: object | None = None, **kwargs: Any) -> None:
        _initialize_verified_source_evidence(
            self,
            SelectedDisplaySourceEvidence,
            args,
            kwargs,
            seal=_seal,
            evidence_name="Verified selected-display source evidence",
        )


@dataclass(frozen=True)
class SelectedHomographySourceEvidence:
    """Digest-bound numeric/YAML homography evidence for one exact H5 camera."""

    source_h5_path: str
    camera_id: str
    numeric_dataset_path: str
    numeric_matrix: tuple[tuple[float, float, float], ...]
    numeric_matrix_sha256: str
    numeric_dataset_attrs: Mapping[str, Any]
    numeric_dataset_attrs_sha256: str
    numeric_dataset_attrs_digest_canonicalization: str
    yaml_dataset_path: str
    yaml_dataset_sha256: str
    yaml_dataset_digest_canonicalization: str
    yaml_matrix_sha256: str
    yaml_dataset_attrs: Mapping[str, Any]
    yaml_dataset_attrs_sha256: str
    yaml_dataset_attrs_digest_canonicalization: str
    matrix_agreement: str

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.numeric_matrix, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": SELECTED_HOMOGRAPHY_SOURCE_SCHEMA_ID,
            "schema_version": SELECTED_HOMOGRAPHY_SOURCE_SCHEMA_VERSION,
            "source_h5_path": self.source_h5_path,
            "camera_id": self.camera_id,
            "numeric_dataset_path": self.numeric_dataset_path,
            "numeric_matrix": [list(row) for row in self.numeric_matrix],
            "numeric_matrix_sha256": self.numeric_matrix_sha256,
            "numeric_dataset_attrs": dict(self.numeric_dataset_attrs),
            "numeric_dataset_attrs_sha256": self.numeric_dataset_attrs_sha256,
            "numeric_dataset_attrs_digest_canonicalization": (
                self.numeric_dataset_attrs_digest_canonicalization
            ),
            "yaml_dataset_path": self.yaml_dataset_path,
            "yaml_dataset_sha256": self.yaml_dataset_sha256,
            "yaml_dataset_digest_canonicalization": (
                self.yaml_dataset_digest_canonicalization
            ),
            "yaml_matrix_sha256": self.yaml_matrix_sha256,
            "yaml_dataset_attrs": dict(self.yaml_dataset_attrs),
            "yaml_dataset_attrs_sha256": self.yaml_dataset_attrs_sha256,
            "yaml_dataset_attrs_digest_canonicalization": (
                self.yaml_dataset_attrs_digest_canonicalization
            ),
            "matrix_agreement": self.matrix_agreement,
        }

    def digest(self) -> str:
        return _canonical_json_sha256(
            self.to_dict(),
            field_name="source_homography",
        )


@dataclass(frozen=True, init=False)
class VerifiedSelectedHomographySourceEvidence(SelectedHomographySourceEvidence):
    """Homography evidence derived from exact numeric/YAML H5 nodes."""

    _source_evidence_seal: object = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(self, *args: Any, _seal: object | None = None, **kwargs: Any) -> None:
        _initialize_verified_source_evidence(
            self,
            SelectedHomographySourceEvidence,
            args,
            kwargs,
            seal=_seal,
            evidence_name="Verified selected-homography source evidence",
        )


@dataclass(frozen=True)
class CameraCalibrationManifest:
    """Digest-bound camera dimensions and scalar calibration values."""

    native_width_px: int
    native_height_px: int
    pixels_per_mm_camera: float | None
    pixel_to_mm: float | None
    pixel_to_mm_derivation: str
    pixels_per_mm_projector: float | None
    z_eff_mm: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "native_width_px": self.native_width_px,
            "native_height_px": self.native_height_px,
            "pixels_per_mm_camera": self.pixels_per_mm_camera,
            "pixel_to_mm": self.pixel_to_mm,
            "pixel_to_mm_derivation": self.pixel_to_mm_derivation,
            "pixels_per_mm_projector": self.pixels_per_mm_projector,
            "z_eff_mm": self.z_eff_mm,
        }


@dataclass(frozen=True)
class DisplaySnapshotManifest:
    """Exact persisted display selection and its parsed viewport geometry."""

    ref: str
    selected_output_name: str
    selected_output_geometry: str
    selected_output_transform_token: str
    width_px: int
    height_px: int
    offset_x_px: int
    offset_y_px: int
    source_h5_dataset_path: str
    source_h5_dataset_sha256: str
    source_h5_dataset_digest_canonicalization: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "selected_output_name": self.selected_output_name,
            "selected_output_geometry": self.selected_output_geometry,
            "selected_output_transform_token": self.selected_output_transform_token,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "offset_x_px": self.offset_x_px,
            "offset_y_px": self.offset_y_px,
            "source_h5_dataset_path": self.source_h5_dataset_path,
            "source_h5_dataset_sha256": self.source_h5_dataset_sha256,
            "source_h5_dataset_digest_canonicalization": (
                self.source_h5_dataset_digest_canonicalization
            ),
        }


@dataclass(frozen=True)
class SourceHomographySemantics:
    """Canonical H5 declarations that justify one camera-to-canvas matrix."""

    source_frame: str
    dest_frame: str
    axes: str
    coordinate_origin: str
    image_space: str
    camera_id: str
    canvas_name: str
    numeric_dataset_path: str
    yaml_dataset_path: str
    numeric_matrix_sha256: str
    yaml_matrix_sha256: str
    matrix_agreement: str
    numeric_payload_source: str
    yaml_payload_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": SOURCE_HOMOGRAPHY_SEMANTICS_SCHEMA_ID,
            "schema_version": SOURCE_HOMOGRAPHY_SEMANTICS_SCHEMA_VERSION,
            "source_frame": self.source_frame,
            "dest_frame": self.dest_frame,
            "axes": self.axes,
            "coordinate_origin": self.coordinate_origin,
            "image_space": self.image_space,
            "camera_id": self.camera_id,
            "canvas_name": self.canvas_name,
            "numeric_dataset_path": self.numeric_dataset_path,
            "yaml_dataset_path": self.yaml_dataset_path,
            "numeric_matrix_sha256": self.numeric_matrix_sha256,
            "yaml_matrix_sha256": self.yaml_matrix_sha256,
            "matrix_agreement": self.matrix_agreement,
            "numeric_payload_source": self.numeric_payload_source,
            "yaml_payload_source": self.yaml_payload_source,
        }

    def digest(self) -> str:
        return source_homography_semantics_digest(self)


@dataclass(frozen=True)
class SourceArtifactLineage:
    """Exact Citrus/H5 source artifact evidence for one homography."""

    source_h5_path: str
    source_homography_semantics: SourceHomographySemantics
    source_homography_semantics_sha256: str
    homography_artifact_path: str
    homography_artifact_checksum_algorithm: str
    homography_artifact_checksum: str
    homography_artifact_size_bytes: int
    homography_artifact_mtime_unix_ns: int
    homography_provenance_schema: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_h5_path": self.source_h5_path,
            "source_homography_semantics": (
                self.source_homography_semantics.to_dict()
            ),
            "source_homography_semantics_sha256": (
                self.source_homography_semantics_sha256
            ),
            "homography_artifact_path": self.homography_artifact_path,
            "homography_artifact_checksum_algorithm": (
                self.homography_artifact_checksum_algorithm
            ),
            "homography_artifact_checksum": self.homography_artifact_checksum,
            "homography_artifact_size_bytes": self.homography_artifact_size_bytes,
            "homography_artifact_mtime_unix_ns": (
                self.homography_artifact_mtime_unix_ns
            ),
            "homography_provenance_schema": self.homography_provenance_schema,
        }


@dataclass(frozen=True)
class SelectedCalibrationManifest:
    """Canonical digest-bound identity for a selected calibration snapshot."""

    stimulus_run: str
    camera_id: str
    camera_calibration_ref: str
    transform_ref: str
    transform_sha256: str
    matrix_sha256: str
    camera_calibration: CameraCalibrationManifest
    display_snapshot: DisplaySnapshotManifest
    source_artifact: SourceArtifactLineage
    source_camera: SelectedCameraSourceEvidence
    source_display: SelectedDisplaySourceEvidence
    source_homography: SelectedHomographySourceEvidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": SELECTED_CALIBRATION_MANIFEST_SCHEMA_ID,
            "schema_version": SELECTED_CALIBRATION_MANIFEST_SCHEMA_VERSION,
            "stimulus_run": self.stimulus_run,
            "camera_id": self.camera_id,
            "camera_calibration_ref": self.camera_calibration_ref,
            "transform_ref": self.transform_ref,
            "transform_sha256": self.transform_sha256,
            "matrix_sha256": self.matrix_sha256,
            "camera_calibration": self.camera_calibration.to_dict(),
            "display_snapshot": self.display_snapshot.to_dict(),
            "source_artifact": self.source_artifact.to_dict(),
            "source_camera": self.source_camera.to_dict(),
            "source_display": self.source_display.to_dict(),
            "source_homography": self.source_homography.to_dict(),
        }

    def digest(self) -> str:
        return selected_calibration_manifest_digest(self)


@dataclass(frozen=True)
class SelectedCalibrationSnapshot:
    """One coherent selected-camera calibration with no inherited fields."""

    stimulus_run: str
    paths: SelectedCalibrationPaths
    camera_id: str
    homography: BoundDirectedHomography
    source_reference_extent: TransformReferenceExtent
    target_reference_extent: TransformReferenceExtent
    display_output_name: str
    display_output_geometry: str
    display_output_transform_token: str
    pixels_per_mm_camera: float | None
    pixel_to_mm: float | None
    pixels_per_mm_projector: float | None
    z_eff_mm: float | None
    manifest: SelectedCalibrationManifest
    manifest_sha256: str


def _required_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SelectedCalibrationError(
            f"{field_name} must be a non-empty string without surrounding whitespace."
        )
    return value


def _exact_mapping(
    value: Any,
    *,
    fields: frozenset[str],
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelectedCalibrationError(f"{field_name} must be a mapping.")
    actual = frozenset(value)
    if actual != fields:
        missing = sorted(fields - actual)
        unknown = sorted(actual - fields)
        raise SelectedCalibrationError(
            f"{field_name} fields are invalid; missing={missing}, unknown={unknown}."
        )
    return value


def _canonical_json(value: Any, *, field_name: str = "value") -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise SelectedCalibrationError(
            f"{field_name} must contain only finite JSON values."
        ) from exc


def _canonical_json_sha256(value: Any, *, field_name: str) -> str:
    return hashlib.sha256(
        _canonical_json(value, field_name=field_name).encode("utf-8")
    ).hexdigest()


def _exact_utf8_payload(value: Any, *, field_name: str) -> tuple[bytes, str]:
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    else:
        raise SelectedCalibrationError(f"{field_name} must be UTF-8 bytes or text.")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SelectedCalibrationError(f"{field_name} must be valid UTF-8.") from exc
    return raw, text


def _load_json_without_duplicate_keys(
    text: str,
    *,
    field_name: str,
    invalid_message: str,
) -> Any:
    """Parse JSON while rejecting duplicate names in every nested object."""

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for name, item in pairs:
            if name in value:
                raise SelectedCalibrationError(
                    f"{field_name} contains duplicate JSON key {name!r}."
                )
            value[name] = item
        return value

    try:
        return json.loads(text, object_pairs_hook=reject_duplicate_pairs)
    except json.JSONDecodeError as exc:
        raise SelectedCalibrationError(invalid_message) from exc


def _load_arena_config_json(text: str) -> Any:
    """Parse exact arena-config text without JSON's last-key-wins ambiguity."""

    return _load_json_without_duplicate_keys(
        text,
        field_name="arena_config_raw",
        invalid_message="arena_config_raw is invalid JSON.",
    )


def _canonical_archive_ref(value: Any, *, field_name: str) -> str:
    text = _required_text(value, field_name=field_name)
    if text != text.strip("/"):
        raise SelectedCalibrationError(
            f"{field_name} must be a canonical archive-relative path."
        )
    parts = text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise SelectedCalibrationError(
            f"{field_name} must be a canonical archive-relative path."
        )
    return text


def _absolute_source_path(value: Any, *, field_name: str) -> str:
    text = _required_text(value, field_name=field_name)
    if (
        not text.startswith("/")
        or text.endswith("/")
        or "//" in text
        or any(part in {".", ".."} for part in text.split("/"))
    ):
        raise SelectedCalibrationError(f"{field_name} must be an absolute path.")
    return text


def _path_segment(value: Any, *, field_name: str) -> str:
    text = _required_text(value, field_name=field_name)
    if text in {".", ".."} or "/" in text:
        raise SelectedCalibrationError(f"{field_name} must be one archive path segment.")
    return text


def selected_calibration_paths(
    *,
    stimulus_run: str,
    camera_id: str,
) -> SelectedCalibrationPaths:
    """Return the only paths accepted by the strict snapshot schema."""

    run = _path_segment(stimulus_run, field_name="stimulus_run")
    camera = _path_segment(camera_id, field_name="camera_id")
    run_path = f"analysis/stimulus_runs/{run}"
    calibration_path = f"{run_path}/calibration"
    camera_path = f"{calibration_path}/{camera}"
    return SelectedCalibrationPaths(
        stimulus_run_path=run_path,
        display_snapshot_path=f"{run_path}/{DISPLAY_SNAPSHOT_GROUP_NAME}",
        calibration_path=calibration_path,
        camera_calibration_path=camera_path,
        homography_array_path=f"{camera_path}/{HOMOGRAPHY_ARRAY_NAME}",
    )


def parse_selected_output_geometry(value: Any) -> tuple[int, int, int, int]:
    """Parse one canonical XRandR ``WxH+X+Y`` geometry string."""

    text = _required_text(value, field_name=SELECTED_OUTPUT_GEOMETRY_ATTR)
    match = _OUTPUT_GEOMETRY_RE.fullmatch(text)
    if match is None:
        raise SelectedCalibrationError(
            "selected_output_geometry must use canonical WxH+X+Y syntax."
        )
    width = int(match.group("width"))
    height = int(match.group("height"))
    offset_x = int(match.group("x"))
    offset_y = int(match.group("y"))
    canonical = f"{width}x{height}{offset_x:+d}{offset_y:+d}"
    if text != canonical:
        raise SelectedCalibrationError(
            "selected_output_geometry is not canonically encoded."
        )
    return width, height, offset_x, offset_y


def source_display_dataset_sha256(value: Any) -> str:
    """Digest the exact selected-output H5 scalar using the declared encoding."""

    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SelectedCalibrationError(
                "Source display dataset bytes must be UTF-8."
            ) from exc
    elif isinstance(value, str):
        text = value
    else:
        raise SelectedCalibrationError(
            "Source display dataset must be a UTF-8 string or bytes."
        )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_display_group_attrs(attrs: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(attrs, Mapping):
        raise SelectedCalibrationError("display_group_attrs must be a mapping.")
    normalized = {
        name: _required_text(
            attrs.get(name),
            field_name=f"display_group_attrs.{name}",
        )
        for name in _DISPLAY_SOURCE_ATTR_FIELDS
    }
    if normalized["selected_output_connection_state"] != "connected":
        raise SelectedCalibrationError("Selected display output must be connected.")
    if normalized["selected_output_transform_token"] != SUPPORTED_OUTPUT_TRANSFORM_TOKEN:
        raise SelectedCalibrationError(
            "Only selected_output_transform_token='normal' is supported."
        )
    parse_selected_output_geometry(normalized["selected_output_geometry"])
    transform_tokens = normalized["selected_output_transform_raw"].split()
    if not transform_tokens or transform_tokens[0] != normalized[
        "selected_output_transform_token"
    ]:
        raise SelectedCalibrationError(
            "Display transform token does not match selected_output_transform_raw."
        )
    return normalized


def _parse_selected_output_block(value: Any) -> tuple[bytes, str, dict[str, str]]:
    raw, text = _exact_utf8_payload(value, field_name="selected_output_block_raw")
    lines = [line for line in text.splitlines() if line.strip()]
    headers = [line for line in lines if _XRANDR_OUTPUT_HEADER_RE.match(line.lstrip())]
    if len(headers) != 1 or not lines or headers[0] != lines[0]:
        raise SelectedCalibrationError(
            "selected_output_block must contain exactly one unambiguous output header."
        )
    first_line = headers[0]
    match = _SELECTED_OUTPUT_HEADER_RE.fullmatch(first_line)
    if match is None:
        raise SelectedCalibrationError(
            "selected_output_block header is malformed or not connected."
        )
    transform_raw = match.group("transform_raw")
    transform_tokens = transform_raw.split()
    if not transform_tokens:
        raise SelectedCalibrationError(
            "selected_output_block transform declaration is empty."
        )
    parsed = {
        "selected_output_name": match.group("name"),
        "selected_output_connection_state": "connected",
        "selected_output_geometry": match.group("geometry"),
        "selected_output_transform_token": transform_tokens[0],
        "selected_output_transform_raw": transform_raw,
    }
    parse_selected_output_geometry(parsed["selected_output_geometry"])
    return raw, first_line, parsed


def parse_selected_display_source_evidence(
    value: Any,
) -> SelectedDisplaySourceEvidence:
    """Parse persisted display evidence without reopening its source H5 file."""

    if isinstance(value, SelectedDisplaySourceEvidence):
        value = value.to_dict()
    payload = _exact_mapping(
        value,
        fields=_SELECTED_DISPLAY_SOURCE_FIELDS,
        field_name="source_display",
    )
    if payload["schema_id"] != SELECTED_DISPLAY_SOURCE_SCHEMA_ID:
        raise SelectedCalibrationError("Unsupported selected-display source schema_id.")
    version = payload["schema_version"]
    if isinstance(version, bool) or version != SELECTED_DISPLAY_SOURCE_SCHEMA_VERSION:
        raise SelectedCalibrationError(
            "Unsupported selected-display source schema_version."
        )
    source_h5_path = _absolute_source_path(
        payload["source_h5_path"],
        field_name="source_display.source_h5_path",
    )
    display_group_path = _absolute_source_path(
        payload["display_group_path"],
        field_name="source_display.display_group_path",
    )
    if display_group_path != SOURCE_DISPLAY_GROUP_PATH:
        raise SelectedCalibrationError("Display evidence identifies the wrong H5 group.")
    group_attrs_raw = _exact_mapping(
        payload["display_group_attrs"],
        fields=_DISPLAY_SOURCE_ATTR_FIELDS,
        field_name="source_display.display_group_attrs",
    )
    group_attrs = _normalize_display_group_attrs(group_attrs_raw)
    group_digest = _required_text(
        payload["display_group_attrs_sha256"],
        field_name="source_display.display_group_attrs_sha256",
    )
    if _SHA256_RE.fullmatch(group_digest) is None:
        raise SelectedCalibrationError("Display group attrs digest is invalid.")
    if _canonical_json_sha256(
        group_attrs,
        field_name="source_display.display_group_attrs",
    ) != group_digest:
        raise SelectedCalibrationError(
            "Display group attrs digest does not match its content."
        )
    group_canonicalization = _required_text(
        payload["display_group_attrs_digest_canonicalization"],
        field_name="source_display.display_group_attrs_digest_canonicalization",
    )
    if group_canonicalization != SOURCE_DISPLAY_ATTRS_DIGEST_CANONICALIZATION:
        raise SelectedCalibrationError(
            "Unsupported display group attrs digest canonicalization."
        )
    dataset_path = _absolute_source_path(
        payload["selected_output_dataset_path"],
        field_name="source_display.selected_output_dataset_path",
    )
    if dataset_path != SOURCE_DISPLAY_DATASET_PATH:
        raise SelectedCalibrationError(
            "Display evidence identifies the wrong selected-output dataset."
        )
    dataset_digest = _required_text(
        payload["selected_output_dataset_sha256"],
        field_name="source_display.selected_output_dataset_sha256",
    )
    if _SHA256_RE.fullmatch(dataset_digest) is None:
        raise SelectedCalibrationError("Selected-output dataset digest is invalid.")
    dataset_canonicalization = _required_text(
        payload["selected_output_dataset_digest_canonicalization"],
        field_name=(
            "source_display.selected_output_dataset_digest_canonicalization"
        ),
    )
    if dataset_canonicalization != SOURCE_DISPLAY_DATASET_DIGEST_CANONICALIZATION:
        raise SelectedCalibrationError(
            "Unsupported selected-output dataset digest canonicalization."
        )
    first_line = _required_text(
        payload["selected_output_first_line"],
        field_name="source_display.selected_output_first_line",
    )
    _raw, parsed_first_line, block_fields = _parse_selected_output_block(first_line)
    if parsed_first_line != first_line:
        raise SelectedCalibrationError("Selected-output first-line evidence is invalid.")
    persisted_fields = {
        name: _required_text(
            payload[name],
            field_name=f"source_display.{name}",
        )
        for name in _DISPLAY_SOURCE_ATTR_FIELDS
    }
    if persisted_fields != block_fields or persisted_fields != group_attrs:
        raise SelectedCalibrationError(
            "Display block, group attrs, and normalized selected-output fields disagree."
        )
    width, height, offset_x, offset_y = parse_selected_output_geometry(
        persisted_fields["selected_output_geometry"]
    )
    return SelectedDisplaySourceEvidence(
        source_h5_path=source_h5_path,
        display_group_path=display_group_path,
        display_group_attrs=group_attrs,
        display_group_attrs_sha256=group_digest,
        display_group_attrs_digest_canonicalization=group_canonicalization,
        selected_output_dataset_path=dataset_path,
        selected_output_dataset_sha256=dataset_digest,
        selected_output_dataset_digest_canonicalization=dataset_canonicalization,
        selected_output_first_line=first_line,
        selected_output_name=persisted_fields["selected_output_name"],
        selected_output_connection_state=persisted_fields[
            "selected_output_connection_state"
        ],
        selected_output_geometry=persisted_fields["selected_output_geometry"],
        selected_output_transform_token=persisted_fields[
            "selected_output_transform_token"
        ],
        selected_output_transform_raw=persisted_fields[
            "selected_output_transform_raw"
        ],
        width_px=width,
        height_px=height,
        offset_x_px=offset_x,
        offset_y_px=offset_y,
    )


def build_selected_display_source_evidence_from_h5_values(
    *,
    source_h5_path: str,
    display_group_path: str,
    display_group_attrs: Mapping[str, Any],
    selected_output_dataset_path: str,
    selected_output_block_raw: bytes | str,
) -> VerifiedSelectedDisplaySourceEvidence:
    """Build display evidence from exact values read from named H5 nodes."""

    source_path = _absolute_source_path(source_h5_path, field_name="source_h5_path")
    normalized_group_path = _absolute_source_path(
        display_group_path,
        field_name="display_group_path",
    )
    if normalized_group_path != SOURCE_DISPLAY_GROUP_PATH:
        raise SelectedCalibrationError("display_group_path must be '/display_snapshot'.")
    normalized_dataset_path = _absolute_source_path(
        selected_output_dataset_path,
        field_name="selected_output_dataset_path",
    )
    if normalized_dataset_path != SOURCE_DISPLAY_DATASET_PATH:
        raise SelectedCalibrationError(
            "selected_output_dataset_path identifies the wrong H5 dataset."
        )
    group_attrs = _normalize_display_group_attrs(display_group_attrs)
    raw, first_line, block_fields = _parse_selected_output_block(
        selected_output_block_raw
    )
    if block_fields != group_attrs:
        raise SelectedCalibrationError(
            "Exact selected-output H5 bytes and display group attrs disagree."
        )
    parsed = parse_selected_display_source_evidence(
        {
            "schema_id": SELECTED_DISPLAY_SOURCE_SCHEMA_ID,
            "schema_version": SELECTED_DISPLAY_SOURCE_SCHEMA_VERSION,
            "source_h5_path": source_path,
            "display_group_path": normalized_group_path,
            "display_group_attrs": group_attrs,
            "display_group_attrs_sha256": _canonical_json_sha256(
                group_attrs,
                field_name="display_group_attrs",
            ),
            "display_group_attrs_digest_canonicalization": (
                SOURCE_DISPLAY_ATTRS_DIGEST_CANONICALIZATION
            ),
            "selected_output_dataset_path": normalized_dataset_path,
            "selected_output_dataset_sha256": hashlib.sha256(raw).hexdigest(),
            "selected_output_dataset_digest_canonicalization": (
                SOURCE_DISPLAY_DATASET_DIGEST_CANONICALIZATION
            ),
            "selected_output_first_line": first_line,
            **block_fields,
        }
    )
    return VerifiedSelectedDisplaySourceEvidence(
        **vars(parsed),
        _seal=_VERIFIED_SOURCE_EVIDENCE_SEAL,
    )


def selected_display_evidence_attrs(
    value: SelectedDisplaySourceEvidence,
) -> dict[str, Any]:
    if not isinstance(value, SelectedDisplaySourceEvidence):
        raise SelectedCalibrationError("Display evidence must be parsed.")
    evidence = parse_selected_display_source_evidence(value)
    return {
        SOURCE_DISPLAY_EVIDENCE_ATTR: evidence.to_dict(),
        f"{SOURCE_DISPLAY_EVIDENCE_ATTR}{SOURCE_DISPLAY_EVIDENCE_DIGEST_SUFFIX}": (
            evidence.digest()
        ),
    }


def load_selected_display_evidence_attrs(
    attrs: Mapping[str, Any],
) -> SelectedDisplaySourceEvidence:
    digest_name = (
        f"{SOURCE_DISPLAY_EVIDENCE_ATTR}{SOURCE_DISPLAY_EVIDENCE_DIGEST_SUFFIX}"
    )
    if SOURCE_DISPLAY_EVIDENCE_ATTR not in attrs or digest_name not in attrs:
        raise SelectedCalibrationError("Persisted source-display evidence is missing.")
    evidence = parse_selected_display_source_evidence(
        attrs[SOURCE_DISPLAY_EVIDENCE_ATTR]
    )
    digest = _required_text(attrs[digest_name], field_name=digest_name)
    if _SHA256_RE.fullmatch(digest) is None or digest != evidence.digest():
        raise SelectedCalibrationError(
            "Persisted source-display evidence digest does not match."
        )
    return evidence


def _parse_normalized_homography_attrs(
    value: Any,
    *,
    kind: str,
    camera_id: str,
) -> dict[str, Any]:
    fields = (
        _HOMOGRAPHY_NUMERIC_ATTR_FIELDS
        if kind == "numeric"
        else _HOMOGRAPHY_YAML_ATTR_FIELDS
    )
    payload = _exact_mapping(
        value,
        fields=fields,
        field_name=f"{kind}_dataset_attrs",
    )
    expected_text = {
        "source_frame": CANONICAL_SOURCE_FRAME,
        "dest_frame": CANONICAL_DEST_FRAME,
        "axes": CANONICAL_AXES,
        "coordinate_origin": CANONICAL_COORDINATE_ORIGIN,
        "image_space": CANONICAL_IMAGE_SPACE,
        "camera_id": camera_id,
        "homography_provenance_schema": "citrus.homography_provenance.v1",
        "homography_payload_source": (
            NUMERIC_HOMOGRAPHY_PAYLOAD_SOURCE
            if kind == "numeric"
            else YAML_HOMOGRAPHY_PAYLOAD_SOURCE
        ),
    }
    if kind == "yaml":
        expected_text["serialization_format"] = YAML_HOMOGRAPHY_SERIALIZATION_FORMAT
    normalized: dict[str, Any] = {}
    for name, expected in expected_text.items():
        actual = _required_text(payload[name], field_name=f"{kind}_attrs.{name}")
        if actual != expected:
            raise SelectedCalibrationError(
                f"{kind} homography attr {name!r} must be {expected!r}."
            )
        normalized[name] = actual
    normalized["canvas_name"] = _required_text(
        payload["canvas_name"],
        field_name=f"{kind}_attrs.canvas_name",
    )
    artifact_exists = payload["homography_artifact_exists"]
    if artifact_exists is not True:
        raise SelectedCalibrationError(
            f"{kind} homography artifact must be recorded as existing."
        )
    normalized["homography_artifact_exists"] = True
    normalized["homography_artifact_path"] = _absolute_source_path(
        payload["homography_artifact_path"],
        field_name=f"{kind}_attrs.homography_artifact_path",
    )
    algorithm = _required_text(
        payload["homography_artifact_checksum_algorithm"],
        field_name=f"{kind}_attrs.homography_artifact_checksum_algorithm",
    )
    if algorithm not in {"fnv1a64", "sha256"}:
        raise SelectedCalibrationError("Unsupported homography artifact checksum algorithm.")
    checksum = _required_text(
        payload["homography_artifact_checksum"],
        field_name=f"{kind}_attrs.homography_artifact_checksum",
    )
    expected_length = 16 if algorithm == "fnv1a64" else 64
    if len(checksum) != expected_length or _LOWER_HEX_RE.fullmatch(checksum) is None:
        raise SelectedCalibrationError(
            f"{kind} homography artifact checksum is invalid."
        )
    normalized["homography_artifact_checksum_algorithm"] = algorithm
    normalized["homography_artifact_checksum"] = checksum
    normalized["homography_artifact_size_bytes"] = _positive_int(
        payload["homography_artifact_size_bytes"],
        field_name=f"{kind}_attrs.homography_artifact_size_bytes",
    )
    normalized["homography_artifact_mtime_unix_ns"] = _nonnegative_int(
        payload["homography_artifact_mtime_unix_ns"],
        field_name=f"{kind}_attrs.homography_artifact_mtime_unix_ns",
    )
    return normalized


def _normalize_h5_homography_attrs(
    attrs: Mapping[str, Any],
    *,
    kind: str,
    camera_id: str,
) -> dict[str, Any]:
    if not isinstance(attrs, Mapping):
        raise SelectedCalibrationError(f"{kind}_dataset_attrs must be a mapping.")
    algorithm = _required_text(
        attrs.get("homography_artifact_checksum_algorithm"),
        field_name=f"{kind}_attrs.homography_artifact_checksum_algorithm",
    )
    checksum_key = f"homography_artifact_checksum_{algorithm}"
    artifact_exists_raw = attrs.get("homography_artifact_exists")
    artifact_exists = (
        artifact_exists_raw is True
        or (
            isinstance(artifact_exists_raw, str)
            and artifact_exists_raw.lower() == "true"
        )
    )
    normalized = {
        name: attrs.get(name)
        for name in (
            _HOMOGRAPHY_NUMERIC_ATTR_FIELDS
            if kind == "numeric"
            else _HOMOGRAPHY_YAML_ATTR_FIELDS
        )
    }
    normalized["homography_artifact_exists"] = artifact_exists
    normalized["homography_artifact_checksum"] = attrs.get(checksum_key)
    return _parse_normalized_homography_attrs(
        normalized,
        kind=kind,
        camera_id=camera_id,
    )


def _parse_opencv_homography_yaml(value: Any) -> tuple[bytes, np.ndarray]:
    raw, text = _exact_utf8_payload(value, field_name="yaml_dataset_raw")
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%YAML") or (stripped == "---" and not cleaned_lines):
            continue
        cleaned_lines.append(line.replace("!!opencv-matrix", ""))
    try:
        parsed = yaml.load(
            "\n".join(cleaned_lines),
            Loader=_UniqueKeySafeLoader,
        )
    except yaml.YAMLError as exc:
        raise SelectedCalibrationError("YAML homography payload is invalid.") from exc
    if not isinstance(parsed, Mapping):
        raise SelectedCalibrationError("YAML homography payload must be a mapping.")
    matrix_payload = parsed.get("homography_matrix")
    if not isinstance(matrix_payload, Mapping):
        raise SelectedCalibrationError("YAML homography_matrix record is missing.")
    if frozenset(matrix_payload) != {"rows", "cols", "dt", "data"}:
        raise SelectedCalibrationError(
            "YAML homography_matrix fields must be exactly rows, cols, dt, and data."
        )
    if matrix_payload.get("rows") != 3 or matrix_payload.get("cols") != 3:
        raise SelectedCalibrationError("YAML homography matrix must declare 3x3 shape.")
    if matrix_payload.get("dt") != "d":
        raise SelectedCalibrationError(
            "YAML homography matrix must declare OpenCV float64 dt='d'."
        )
    data = matrix_payload.get("data")
    if not isinstance(data, (list, tuple)) or len(data) != 9:
        raise SelectedCalibrationError("YAML homography matrix must contain nine values.")
    try:
        matrix = np.asarray(data, dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError) as exc:
        raise SelectedCalibrationError("YAML homography values are invalid.") from exc
    # The shared matrix validator is reached through its digest helper.
    homography_matrix_sha256(matrix)
    return raw, matrix


def _homography_matrix_tuple(value: Any) -> tuple[tuple[float, float, float], ...]:
    matrix = np.asarray(value, dtype=np.float64)
    homography_matrix_sha256(matrix)
    return tuple(tuple(float(item) for item in row) for row in matrix)


def parse_selected_homography_source_evidence(
    value: Any,
) -> SelectedHomographySourceEvidence:
    """Parse persisted numeric/YAML evidence without reopening source H5."""

    if isinstance(value, SelectedHomographySourceEvidence):
        value = value.to_dict()
    payload = _exact_mapping(
        value,
        fields=_SELECTED_HOMOGRAPHY_SOURCE_FIELDS,
        field_name="source_homography",
    )
    if payload["schema_id"] != SELECTED_HOMOGRAPHY_SOURCE_SCHEMA_ID:
        raise SelectedCalibrationError(
            "Unsupported selected-homography source schema_id."
        )
    version = payload["schema_version"]
    if isinstance(version, bool) or version != SELECTED_HOMOGRAPHY_SOURCE_SCHEMA_VERSION:
        raise SelectedCalibrationError(
            "Unsupported selected-homography source schema_version."
        )
    source_h5_path = _absolute_source_path(
        payload["source_h5_path"],
        field_name="source_homography.source_h5_path",
    )
    camera_id = _path_segment(
        payload["camera_id"],
        field_name="source_homography.camera_id",
    )
    expected_prefix = f"/calibration_snapshot/{camera_id}"
    numeric_path = _absolute_source_path(
        payload["numeric_dataset_path"],
        field_name="source_homography.numeric_dataset_path",
    )
    yaml_path = _absolute_source_path(
        payload["yaml_dataset_path"],
        field_name="source_homography.yaml_dataset_path",
    )
    if numeric_path != f"{expected_prefix}/homography_matrix":
        raise SelectedCalibrationError("Numeric homography path is not the selected camera.")
    if yaml_path != f"{expected_prefix}/homography_matrix_yml":
        raise SelectedCalibrationError("YAML homography path is not the selected camera.")
    matrix_tuple = _homography_matrix_tuple(payload["numeric_matrix"])
    matrix = np.asarray(matrix_tuple, dtype=np.float64)
    numeric_digest = _required_text(
        payload["numeric_matrix_sha256"],
        field_name="source_homography.numeric_matrix_sha256",
    )
    yaml_matrix_digest = _required_text(
        payload["yaml_matrix_sha256"],
        field_name="source_homography.yaml_matrix_sha256",
    )
    if (
        _SHA256_RE.fullmatch(numeric_digest) is None
        or _SHA256_RE.fullmatch(yaml_matrix_digest) is None
    ):
        raise SelectedCalibrationError("Homography matrix digest is invalid.")
    if homography_matrix_sha256(matrix) != numeric_digest:
        raise SelectedCalibrationError(
            "Numeric homography matrix digest does not match its content."
        )
    if numeric_digest != yaml_matrix_digest:
        raise SelectedCalibrationError(
            "Numeric and YAML homography matrices do not agree exactly."
        )
    agreement = _required_text(
        payload["matrix_agreement"],
        field_name="source_homography.matrix_agreement",
    )
    if agreement != CANONICAL_MATRIX_AGREEMENT:
        raise SelectedCalibrationError("Unsupported homography matrix agreement.")
    numeric_attrs = _parse_normalized_homography_attrs(
        payload["numeric_dataset_attrs"],
        kind="numeric",
        camera_id=camera_id,
    )
    yaml_attrs = _parse_normalized_homography_attrs(
        payload["yaml_dataset_attrs"],
        kind="yaml",
        camera_id=camera_id,
    )
    for kind, attrs_record, digest_name, canonicalization_name in (
        (
            "numeric",
            numeric_attrs,
            "numeric_dataset_attrs_sha256",
            "numeric_dataset_attrs_digest_canonicalization",
        ),
        (
            "yaml",
            yaml_attrs,
            "yaml_dataset_attrs_sha256",
            "yaml_dataset_attrs_digest_canonicalization",
        ),
    ):
        digest = _required_text(
            payload[digest_name],
            field_name=f"source_homography.{digest_name}",
        )
        if _SHA256_RE.fullmatch(digest) is None or digest != _canonical_json_sha256(
            attrs_record,
            field_name=f"source_homography.{kind}_dataset_attrs",
        ):
            raise SelectedCalibrationError(
                f"{kind} homography attrs digest does not match."
            )
        canonicalization = _required_text(
            payload[canonicalization_name],
            field_name=f"source_homography.{canonicalization_name}",
        )
        if canonicalization != SOURCE_HOMOGRAPHY_ATTRS_DIGEST_CANONICALIZATION:
            raise SelectedCalibrationError(
                f"Unsupported {kind} homography attrs digest canonicalization."
            )
    numeric_shared = {name: numeric_attrs[name] for name in _HOMOGRAPHY_SHARED_ATTR_FIELDS}
    yaml_shared = {name: yaml_attrs[name] for name in _HOMOGRAPHY_SHARED_ATTR_FIELDS}
    if numeric_shared != yaml_shared:
        raise SelectedCalibrationError(
            "Numeric and YAML homography controlled semantics or lineage disagree."
        )
    yaml_dataset_digest = _required_text(
        payload["yaml_dataset_sha256"],
        field_name="source_homography.yaml_dataset_sha256",
    )
    if _SHA256_RE.fullmatch(yaml_dataset_digest) is None:
        raise SelectedCalibrationError("YAML dataset digest is invalid.")
    yaml_canonicalization = _required_text(
        payload["yaml_dataset_digest_canonicalization"],
        field_name="source_homography.yaml_dataset_digest_canonicalization",
    )
    if yaml_canonicalization != SOURCE_HOMOGRAPHY_YAML_DIGEST_CANONICALIZATION:
        raise SelectedCalibrationError(
            "Unsupported YAML dataset digest canonicalization."
        )
    return SelectedHomographySourceEvidence(
        source_h5_path=source_h5_path,
        camera_id=camera_id,
        numeric_dataset_path=numeric_path,
        numeric_matrix=matrix_tuple,
        numeric_matrix_sha256=numeric_digest,
        numeric_dataset_attrs=numeric_attrs,
        numeric_dataset_attrs_sha256=payload["numeric_dataset_attrs_sha256"],
        numeric_dataset_attrs_digest_canonicalization=payload[
            "numeric_dataset_attrs_digest_canonicalization"
        ],
        yaml_dataset_path=yaml_path,
        yaml_dataset_sha256=yaml_dataset_digest,
        yaml_dataset_digest_canonicalization=yaml_canonicalization,
        yaml_matrix_sha256=yaml_matrix_digest,
        yaml_dataset_attrs=yaml_attrs,
        yaml_dataset_attrs_sha256=payload["yaml_dataset_attrs_sha256"],
        yaml_dataset_attrs_digest_canonicalization=payload[
            "yaml_dataset_attrs_digest_canonicalization"
        ],
        matrix_agreement=agreement,
    )


def build_selected_homography_source_evidence_from_h5_values(
    *,
    source_h5_path: str,
    expected_camera_id: str,
    numeric_dataset_path: str,
    numeric_matrix: Any,
    numeric_dataset_attrs: Mapping[str, Any],
    yaml_dataset_path: str,
    yaml_dataset_raw: bytes | str,
    yaml_dataset_attrs: Mapping[str, Any],
) -> VerifiedSelectedHomographySourceEvidence:
    """Build evidence by independently validating exact numeric and YAML H5 nodes."""

    source_path = _absolute_source_path(source_h5_path, field_name="source_h5_path")
    camera_id = _path_segment(expected_camera_id, field_name="expected_camera_id")
    expected_prefix = f"/calibration_snapshot/{camera_id}"
    numeric_path = _absolute_source_path(
        numeric_dataset_path,
        field_name="numeric_dataset_path",
    )
    yaml_path = _absolute_source_path(yaml_dataset_path, field_name="yaml_dataset_path")
    if numeric_path != f"{expected_prefix}/homography_matrix":
        raise SelectedCalibrationError("numeric_dataset_path is not the selected camera.")
    if yaml_path != f"{expected_prefix}/homography_matrix_yml":
        raise SelectedCalibrationError("yaml_dataset_path is not the selected camera.")
    matrix_tuple = _homography_matrix_tuple(numeric_matrix)
    matrix = np.asarray(matrix_tuple, dtype=np.float64)
    yaml_raw, yaml_matrix = _parse_opencv_homography_yaml(yaml_dataset_raw)
    numeric_digest = homography_matrix_sha256(matrix)
    yaml_matrix_digest = homography_matrix_sha256(yaml_matrix)
    if numeric_digest != yaml_matrix_digest:
        raise SelectedCalibrationError(
            "Exact numeric and YAML H5 homography payloads disagree."
        )
    numeric_attrs = _normalize_h5_homography_attrs(
        numeric_dataset_attrs,
        kind="numeric",
        camera_id=camera_id,
    )
    yaml_attrs = _normalize_h5_homography_attrs(
        yaml_dataset_attrs,
        kind="yaml",
        camera_id=camera_id,
    )
    numeric_shared = {name: numeric_attrs[name] for name in _HOMOGRAPHY_SHARED_ATTR_FIELDS}
    yaml_shared = {name: yaml_attrs[name] for name in _HOMOGRAPHY_SHARED_ATTR_FIELDS}
    if numeric_shared != yaml_shared:
        raise SelectedCalibrationError(
            "Numeric and YAML H5 homography semantics or artifact lineage disagree."
        )
    parsed = parse_selected_homography_source_evidence(
        {
            "schema_id": SELECTED_HOMOGRAPHY_SOURCE_SCHEMA_ID,
            "schema_version": SELECTED_HOMOGRAPHY_SOURCE_SCHEMA_VERSION,
            "source_h5_path": source_path,
            "camera_id": camera_id,
            "numeric_dataset_path": numeric_path,
            "numeric_matrix": [list(row) for row in matrix_tuple],
            "numeric_matrix_sha256": numeric_digest,
            "numeric_dataset_attrs": numeric_attrs,
            "numeric_dataset_attrs_sha256": _canonical_json_sha256(
                numeric_attrs,
                field_name="numeric_dataset_attrs",
            ),
            "numeric_dataset_attrs_digest_canonicalization": (
                SOURCE_HOMOGRAPHY_ATTRS_DIGEST_CANONICALIZATION
            ),
            "yaml_dataset_path": yaml_path,
            "yaml_dataset_sha256": hashlib.sha256(yaml_raw).hexdigest(),
            "yaml_dataset_digest_canonicalization": (
                SOURCE_HOMOGRAPHY_YAML_DIGEST_CANONICALIZATION
            ),
            "yaml_matrix_sha256": yaml_matrix_digest,
            "yaml_dataset_attrs": yaml_attrs,
            "yaml_dataset_attrs_sha256": _canonical_json_sha256(
                yaml_attrs,
                field_name="yaml_dataset_attrs",
            ),
            "yaml_dataset_attrs_digest_canonicalization": (
                SOURCE_HOMOGRAPHY_ATTRS_DIGEST_CANONICALIZATION
            ),
            "matrix_agreement": CANONICAL_MATRIX_AGREEMENT,
        }
    )
    return VerifiedSelectedHomographySourceEvidence(
        **vars(parsed),
        _seal=_VERIFIED_SOURCE_EVIDENCE_SEAL,
    )


def source_homography_semantics_from_evidence(
    value: SelectedHomographySourceEvidence,
) -> SourceHomographySemantics:
    evidence = parse_selected_homography_source_evidence(value)
    numeric_attrs = evidence.numeric_dataset_attrs
    yaml_attrs = evidence.yaml_dataset_attrs
    return parse_source_homography_semantics(
        {
            "schema_id": SOURCE_HOMOGRAPHY_SEMANTICS_SCHEMA_ID,
            "schema_version": SOURCE_HOMOGRAPHY_SEMANTICS_SCHEMA_VERSION,
            "source_frame": numeric_attrs["source_frame"],
            "dest_frame": numeric_attrs["dest_frame"],
            "axes": numeric_attrs["axes"],
            "coordinate_origin": numeric_attrs["coordinate_origin"],
            "image_space": numeric_attrs["image_space"],
            "camera_id": evidence.camera_id,
            "canvas_name": numeric_attrs["canvas_name"],
            "numeric_dataset_path": evidence.numeric_dataset_path,
            "yaml_dataset_path": evidence.yaml_dataset_path,
            "numeric_matrix_sha256": evidence.numeric_matrix_sha256,
            "yaml_matrix_sha256": evidence.yaml_matrix_sha256,
            "matrix_agreement": evidence.matrix_agreement,
            "numeric_payload_source": numeric_attrs["homography_payload_source"],
            "yaml_payload_source": yaml_attrs["homography_payload_source"],
        },
        camera_id=evidence.camera_id,
    )


def source_artifact_from_homography_evidence(
    value: SelectedHomographySourceEvidence,
) -> SourceArtifactLineage:
    evidence = parse_selected_homography_source_evidence(value)
    semantics = source_homography_semantics_from_evidence(evidence)
    attrs = evidence.numeric_dataset_attrs
    return parse_source_artifact_lineage(
        {
            "source_h5_path": evidence.source_h5_path,
            "source_homography_semantics": semantics.to_dict(),
            "source_homography_semantics_sha256": semantics.digest(),
            "homography_artifact_path": attrs["homography_artifact_path"],
            "homography_artifact_checksum_algorithm": attrs[
                "homography_artifact_checksum_algorithm"
            ],
            "homography_artifact_checksum": attrs["homography_artifact_checksum"],
            "homography_artifact_size_bytes": attrs[
                "homography_artifact_size_bytes"
            ],
            "homography_artifact_mtime_unix_ns": attrs[
                "homography_artifact_mtime_unix_ns"
            ],
            "homography_provenance_schema": attrs[
                "homography_provenance_schema"
            ],
        },
        camera_id=evidence.camera_id,
    )


def selected_homography_evidence_attrs(
    value: SelectedHomographySourceEvidence,
) -> dict[str, Any]:
    if not isinstance(value, SelectedHomographySourceEvidence):
        raise SelectedCalibrationError("Homography evidence must be parsed.")
    evidence = parse_selected_homography_source_evidence(value)
    return {
        SOURCE_HOMOGRAPHY_EVIDENCE_ATTR: evidence.to_dict(),
        (
            f"{SOURCE_HOMOGRAPHY_EVIDENCE_ATTR}"
            f"{SOURCE_HOMOGRAPHY_EVIDENCE_DIGEST_SUFFIX}"
        ): evidence.digest(),
    }


def load_selected_homography_evidence_attrs(
    attrs: Mapping[str, Any],
) -> SelectedHomographySourceEvidence:
    digest_name = (
        f"{SOURCE_HOMOGRAPHY_EVIDENCE_ATTR}"
        f"{SOURCE_HOMOGRAPHY_EVIDENCE_DIGEST_SUFFIX}"
    )
    if SOURCE_HOMOGRAPHY_EVIDENCE_ATTR not in attrs or digest_name not in attrs:
        raise SelectedCalibrationError("Persisted source-homography evidence is missing.")
    evidence = parse_selected_homography_source_evidence(
        attrs[SOURCE_HOMOGRAPHY_EVIDENCE_ATTR]
    )
    digest = _required_text(attrs[digest_name], field_name=digest_name)
    if _SHA256_RE.fullmatch(digest) is None or digest != evidence.digest():
        raise SelectedCalibrationError(
            "Persisted source-homography evidence digest does not match."
        )
    return evidence


def _display_manifest_from_attrs(
    attrs: Mapping[str, Any],
    *,
    display_ref: str,
) -> DisplaySnapshotManifest:
    output_name = _required_text(
        attrs.get(SELECTED_OUTPUT_NAME_ATTR),
        field_name=SELECTED_OUTPUT_NAME_ATTR,
    )
    geometry = _required_text(
        attrs.get(SELECTED_OUTPUT_GEOMETRY_ATTR),
        field_name=SELECTED_OUTPUT_GEOMETRY_ATTR,
    )
    transform_token = _required_text(
        attrs.get(SELECTED_OUTPUT_TRANSFORM_TOKEN_ATTR),
        field_name=SELECTED_OUTPUT_TRANSFORM_TOKEN_ATTR,
    )
    if transform_token != SUPPORTED_OUTPUT_TRANSFORM_TOKEN:
        raise SelectedCalibrationError(
            "Only selected_output_transform_token='normal' is supported."
        )
    width, height, offset_x, offset_y = parse_selected_output_geometry(geometry)
    source_dataset_path = _absolute_source_path(
        attrs.get(SOURCE_DISPLAY_DATASET_PATH_ATTR),
        field_name=SOURCE_DISPLAY_DATASET_PATH_ATTR,
    )
    if source_dataset_path != "/display_snapshot/selected_output_block":
        raise SelectedCalibrationError(
            "source_display_dataset_path must identify the selected H5 output block."
        )
    source_dataset_sha256 = _required_text(
        attrs.get(SOURCE_DISPLAY_DATASET_SHA256_ATTR),
        field_name=SOURCE_DISPLAY_DATASET_SHA256_ATTR,
    )
    if _SHA256_RE.fullmatch(source_dataset_sha256) is None:
        raise SelectedCalibrationError(
            "source_display_dataset_sha256 must be a lowercase SHA-256 digest."
        )
    source_digest_canonicalization = _required_text(
        attrs.get(SOURCE_DISPLAY_DATASET_DIGEST_CANONICALIZATION_ATTR),
        field_name=SOURCE_DISPLAY_DATASET_DIGEST_CANONICALIZATION_ATTR,
    )
    if (
        source_digest_canonicalization
        != SOURCE_DISPLAY_DATASET_DIGEST_CANONICALIZATION
    ):
        raise SelectedCalibrationError(
            "Unsupported source display dataset digest canonicalization."
        )
    return DisplaySnapshotManifest(
        ref=_canonical_archive_ref(display_ref, field_name="display_snapshot.ref"),
        selected_output_name=output_name,
        selected_output_geometry=geometry,
        selected_output_transform_token=transform_token,
        width_px=width,
        height_px=height,
        offset_x_px=offset_x,
        offset_y_px=offset_y,
        source_h5_dataset_path=source_dataset_path,
        source_h5_dataset_sha256=source_dataset_sha256,
        source_h5_dataset_digest_canonicalization=(
            source_digest_canonicalization
        ),
    )


def selected_calibration_pointer_attrs(
    *,
    stimulus_run: str,
    camera_id: str,
    transform_sha256: str,
) -> dict[str, Any]:
    """Build the calibration-parent attrs that a future importer must persist."""

    paths = selected_calibration_paths(
        stimulus_run=stimulus_run,
        camera_id=camera_id,
    )
    digest = _required_text(transform_sha256, field_name="transform_sha256")
    if _SHA256_RE.fullmatch(digest) is None:
        raise SelectedCalibrationError(
            "transform_sha256 must be a lowercase 64-character SHA-256 digest."
        )
    return {
        "schema_id": SELECTED_CALIBRATION_SCHEMA_ID,
        "schema_version": SELECTED_CALIBRATION_SCHEMA_VERSION,
        STIMULUS_RUN_ATTR: stimulus_run,
        ACTIVE_CAMERA_ID_ATTR: camera_id,
        ACTIVE_CAMERA_CALIBRATION_REF_ATTR: paths.camera_calibration_path,
        ACTIVE_CAMERA_TRANSFORM_REF_ATTR: paths.homography_array_path,
        ACTIVE_CAMERA_TRANSFORM_SHA256_ATTR: digest,
    }


def _positive_scalar(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SelectedCalibrationError(f"{field_name} must be a positive finite float.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise SelectedCalibrationError(
            f"{field_name} must be a positive finite float."
        ) from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise SelectedCalibrationError(f"{field_name} must be a positive finite float.")
    return numeric


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise SelectedCalibrationError(f"{field_name} must be a positive integer.")
    return int(value)


def _nonnegative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise SelectedCalibrationError(
            f"{field_name} must be a non-negative integer."
        )
    return int(value)


def _signed_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise SelectedCalibrationError(f"{field_name} must be an integer.")
    return int(value)


def _optional_positive_scalar(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _positive_scalar(value, field_name=field_name)


def _optional_nonnegative_source_scalar(
    value: Any,
    *,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SelectedCalibrationError(
            f"{field_name} must be a non-negative finite float or null."
        )
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise SelectedCalibrationError(
            f"{field_name} must be a non-negative finite float or null."
        )
    return numeric


def source_arena_config_dataset_sha256(value: Any) -> str:
    """Digest the exact UTF-8 bytes read from the arena-config H5 dataset."""

    raw, _text = _exact_utf8_payload(
        value,
        field_name="arena_config_raw",
    )
    return hashlib.sha256(raw).hexdigest()


def selected_camera_record_digest(value: Any) -> str:
    """Digest one exact JSON camera record using canonical JSON encoding."""

    if not isinstance(value, Mapping):
        raise SelectedCalibrationError("selected_camera_record must be a mapping.")
    return _canonical_json_sha256(value, field_name="selected_camera_record")


def camera_group_scalar_attrs_digest(value: Any) -> str:
    """Digest the exact normalized scalar attr subset for the selected camera."""

    payload = _exact_mapping(
        value,
        fields=frozenset(_SOURCE_CAMERA_SCALAR_FIELDS),
        field_name="camera_group_scalar_attrs",
    )
    normalized = {
        name: _optional_positive_scalar(
            payload[name],
            field_name=f"camera_group_scalar_attrs.{name}",
        )
        for name in _SOURCE_CAMERA_SCALAR_FIELDS
    }
    return _canonical_json_sha256(
        normalized,
        field_name="camera_group_scalar_attrs",
    )


def parse_selected_camera_source_evidence(
    value: Any,
) -> SelectedCameraSourceEvidence:
    """Parse digest-bound source claims without opening or searching an H5 file."""

    if isinstance(value, SelectedCameraSourceEvidence):
        value = value.to_dict()
    payload = _exact_mapping(
        value,
        fields=_SELECTED_CAMERA_SOURCE_FIELDS,
        field_name="source_camera",
    )
    if payload["schema_id"] != SELECTED_CAMERA_SOURCE_SCHEMA_ID:
        raise SelectedCalibrationError("Unsupported selected-camera source schema_id.")
    version = payload["schema_version"]
    if isinstance(version, bool) or version != SELECTED_CAMERA_SOURCE_SCHEMA_VERSION:
        raise SelectedCalibrationError(
            "Unsupported selected-camera source schema_version."
        )

    source_h5_path = _absolute_source_path(
        payload["source_h5_path"],
        field_name="source_camera.source_h5_path",
    )
    arena_config_dataset_path = _absolute_source_path(
        payload["arena_config_dataset_path"],
        field_name="source_camera.arena_config_dataset_path",
    )
    if arena_config_dataset_path != SOURCE_ARENA_CONFIG_DATASET_PATH:
        raise SelectedCalibrationError(
            "Selected-camera source must identify the exact arena_config_json dataset."
        )
    arena_config_digest = _required_text(
        payload["arena_config_dataset_sha256"],
        field_name="source_camera.arena_config_dataset_sha256",
    )
    if _SHA256_RE.fullmatch(arena_config_digest) is None:
        raise SelectedCalibrationError(
            "Arena-config dataset digest must be a lowercase SHA-256 value."
        )
    arena_config_canonicalization = _required_text(
        payload["arena_config_dataset_digest_canonicalization"],
        field_name=(
            "source_camera.arena_config_dataset_digest_canonicalization"
        ),
    )
    if arena_config_canonicalization != SOURCE_ARENA_CONFIG_DIGEST_CANONICALIZATION:
        raise SelectedCalibrationError(
            "Unsupported arena-config dataset digest canonicalization."
        )

    active_camera_id = _path_segment(
        payload["active_camera_id"],
        field_name="source_camera.active_camera_id",
    )
    record_index = _nonnegative_int(
        payload["selected_camera_record_index"],
        field_name="source_camera.selected_camera_record_index",
    )
    record_raw = payload["selected_camera_record"]
    if not isinstance(record_raw, Mapping):
        raise SelectedCalibrationError(
            "source_camera.selected_camera_record must be a mapping."
        )
    record = json.loads(
        _canonical_json(
            record_raw,
            field_name="source_camera.selected_camera_record",
        )
    )
    record_digest = _required_text(
        payload["selected_camera_record_sha256"],
        field_name="source_camera.selected_camera_record_sha256",
    )
    if _SHA256_RE.fullmatch(record_digest) is None:
        raise SelectedCalibrationError(
            "Selected-camera record digest must be a lowercase SHA-256 value."
        )
    if selected_camera_record_digest(record) != record_digest:
        raise SelectedCalibrationError(
            "Selected-camera record digest does not match its content."
        )
    record_canonicalization = _required_text(
        payload["selected_camera_record_digest_canonicalization"],
        field_name=(
            "source_camera.selected_camera_record_digest_canonicalization"
        ),
    )
    if record_canonicalization != SOURCE_CAMERA_RECORD_DIGEST_CANONICALIZATION:
        raise SelectedCalibrationError(
            "Unsupported selected-camera record digest canonicalization."
        )
    record_camera_id = _path_segment(
        record.get("camera_id"),
        field_name="source_camera.selected_camera_record.camera_id",
    )
    if record_camera_id != active_camera_id:
        raise SelectedCalibrationError(
            "Selected camera record does not match arena-config active_camera_id."
        )

    camera_group_path = _absolute_source_path(
        payload["camera_group_path"],
        field_name="source_camera.camera_group_path",
    )
    expected_camera_group_path = f"/calibration_snapshot/{active_camera_id}"
    if camera_group_path != expected_camera_group_path:
        raise SelectedCalibrationError(
            "Selected camera group path does not match active_camera_id."
        )
    scalar_attrs_raw = _exact_mapping(
        payload["camera_group_scalar_attrs"],
        fields=frozenset(_SOURCE_CAMERA_SCALAR_FIELDS),
        field_name="source_camera.camera_group_scalar_attrs",
    )
    scalar_attrs = {
        name: _optional_positive_scalar(
            scalar_attrs_raw[name],
            field_name=f"source_camera.camera_group_scalar_attrs.{name}",
        )
        for name in _SOURCE_CAMERA_SCALAR_FIELDS
    }
    scalar_attrs_digest = _required_text(
        payload["camera_group_scalar_attrs_sha256"],
        field_name="source_camera.camera_group_scalar_attrs_sha256",
    )
    if _SHA256_RE.fullmatch(scalar_attrs_digest) is None:
        raise SelectedCalibrationError(
            "Selected camera-group attrs digest must be a lowercase SHA-256 value."
        )
    if camera_group_scalar_attrs_digest(scalar_attrs) != scalar_attrs_digest:
        raise SelectedCalibrationError(
            "Selected camera-group scalar attrs digest does not match its content."
        )
    scalar_attrs_canonicalization = _required_text(
        payload["camera_group_scalar_attrs_digest_canonicalization"],
        field_name=(
            "source_camera.camera_group_scalar_attrs_digest_canonicalization"
        ),
    )
    if scalar_attrs_canonicalization != SOURCE_CAMERA_ATTRS_DIGEST_CANONICALIZATION:
        raise SelectedCalibrationError(
            "Unsupported selected camera-group attrs digest canonicalization."
        )

    record_scalars = {
        name: _optional_positive_scalar(
            record.get(name),
            field_name=f"source_camera.selected_camera_record.{name}",
        )
        for name in _SOURCE_CAMERA_SCALAR_FIELDS
    }
    for name in _SOURCE_CAMERA_SCALAR_FIELDS:
        if record_scalars[name] != scalar_attrs[name]:
            raise SelectedCalibrationError(
                f"Selected camera record and camera-group attr {name!r} disagree."
            )

    z_eff_source_field = _required_text(
        payload["z_eff_mm_source_field"],
        field_name="source_camera.z_eff_mm_source_field",
    )
    if z_eff_source_field != Z_EFF_MM_SOURCE_FIELD:
        raise SelectedCalibrationError("Unsupported z_eff_mm source field.")
    z_eff_source_value = _optional_nonnegative_source_scalar(
        payload["z_eff_mm_source_value"],
        field_name="source_camera.z_eff_mm_source_value",
    )
    z_eff_derivation = _required_text(
        payload["z_eff_mm_derivation"],
        field_name="source_camera.z_eff_mm_derivation",
    )
    if z_eff_derivation != Z_EFF_MM_DERIVATION:
        raise SelectedCalibrationError("Unsupported z_eff_mm derivation.")

    native_width_px = _positive_int(
        record.get("native_width_px"),
        field_name="source_camera.selected_camera_record.native_width_px",
    )
    native_height_px = _positive_int(
        record.get("native_height_px"),
        field_name="source_camera.selected_camera_record.native_height_px",
    )
    ppm_camera = record_scalars["pixels_per_mm_camera"]
    pixel_to_mm = None if ppm_camera is None else 1.0 / ppm_camera
    z_eff_mm = (
        z_eff_source_value
        if z_eff_source_value is not None and z_eff_source_value > 0
        else None
    )
    return SelectedCameraSourceEvidence(
        source_h5_path=source_h5_path,
        arena_config_dataset_path=arena_config_dataset_path,
        arena_config_dataset_sha256=arena_config_digest,
        arena_config_dataset_digest_canonicalization=arena_config_canonicalization,
        active_camera_id=active_camera_id,
        selected_camera_record_index=record_index,
        selected_camera_record=record,
        selected_camera_record_sha256=record_digest,
        selected_camera_record_digest_canonicalization=record_canonicalization,
        camera_group_path=camera_group_path,
        camera_group_scalar_attrs=scalar_attrs,
        camera_group_scalar_attrs_sha256=scalar_attrs_digest,
        camera_group_scalar_attrs_digest_canonicalization=(
            scalar_attrs_canonicalization
        ),
        z_eff_mm_source_field=z_eff_source_field,
        z_eff_mm_source_value=z_eff_source_value,
        z_eff_mm_derivation=z_eff_derivation,
        native_width_px=native_width_px,
        native_height_px=native_height_px,
        pixels_per_mm_camera=ppm_camera,
        pixels_per_mm_projector=record_scalars["pixels_per_mm_projector"],
        real_world_ref_mm=record_scalars["real_world_ref_mm"],
        pixel_to_mm=pixel_to_mm,
        z_eff_mm=z_eff_mm,
    )


def build_selected_camera_source_evidence_from_h5_values(
    *,
    source_h5_path: str,
    arena_config_raw: bytes | str,
    camera_group_path: str,
    camera_group_attrs: Mapping[str, Any],
    expected_camera_id: str,
) -> VerifiedSelectedCameraSourceEvidence:
    """Build evidence from values already read from exact, named H5 nodes.

    This helper never searches for a camera and never opens H5.  The importer is
    responsible for reading ``arena_config_raw`` from
    ``/calibration_snapshot/arena_config_json`` and ``camera_group_attrs`` from
    ``/calibration_snapshot/<expected_camera_id>`` before calling it.
    """

    source_path = _absolute_source_path(source_h5_path, field_name="source_h5_path")
    expected_camera = _path_segment(
        expected_camera_id,
        field_name="expected_camera_id",
    )
    raw, text = _exact_utf8_payload(
        arena_config_raw,
        field_name="arena_config_raw",
    )
    arena_config = _load_arena_config_json(text)
    if not isinstance(arena_config, Mapping):
        raise SelectedCalibrationError("arena_config_raw must contain a JSON object.")
    _canonical_json(arena_config, field_name="arena_config_raw")
    active_camera = _path_segment(
        arena_config.get("active_camera_id"),
        field_name="arena_config.active_camera_id",
    )
    if active_camera != expected_camera:
        raise SelectedCalibrationError(
            "arena_config active_camera_id does not match expected_camera_id."
        )
    records = arena_config.get("camera_calibrations")
    if not isinstance(records, list):
        raise SelectedCalibrationError(
            "arena_config.camera_calibrations must be a list."
        )
    matches = [
        (index, record)
        for index, record in enumerate(records)
        if isinstance(record, Mapping) and record.get("camera_id") == active_camera
    ]
    if len(matches) != 1:
        raise SelectedCalibrationError(
            "arena_config must contain exactly one active-camera calibration record."
        )
    record_index, record_raw = matches[0]
    record = json.loads(
        _canonical_json(record_raw, field_name="selected_camera_record")
    )
    normalized_group_path = _absolute_source_path(
        camera_group_path,
        field_name="camera_group_path",
    )
    expected_group_path = f"/calibration_snapshot/{active_camera}"
    if normalized_group_path != expected_group_path:
        raise SelectedCalibrationError(
            "camera_group_path does not identify the exact active camera group."
        )
    if not isinstance(camera_group_attrs, Mapping):
        raise SelectedCalibrationError("camera_group_attrs must be a mapping.")
    scalar_attrs = {
        name: _optional_positive_scalar(
            camera_group_attrs.get(name),
            field_name=f"camera_group_attrs.{name}",
        )
        for name in _SOURCE_CAMERA_SCALAR_FIELDS
    }
    z_eff_source_value = _optional_nonnegative_source_scalar(
        arena_config.get(Z_EFF_MM_SOURCE_FIELD),
        field_name=f"arena_config.{Z_EFF_MM_SOURCE_FIELD}",
    )
    parsed = parse_selected_camera_source_evidence(
        {
            "schema_id": SELECTED_CAMERA_SOURCE_SCHEMA_ID,
            "schema_version": SELECTED_CAMERA_SOURCE_SCHEMA_VERSION,
            "source_h5_path": source_path,
            "arena_config_dataset_path": SOURCE_ARENA_CONFIG_DATASET_PATH,
            "arena_config_dataset_sha256": hashlib.sha256(raw).hexdigest(),
            "arena_config_dataset_digest_canonicalization": (
                SOURCE_ARENA_CONFIG_DIGEST_CANONICALIZATION
            ),
            "active_camera_id": active_camera,
            "selected_camera_record_index": record_index,
            "selected_camera_record": record,
            "selected_camera_record_sha256": selected_camera_record_digest(record),
            "selected_camera_record_digest_canonicalization": (
                SOURCE_CAMERA_RECORD_DIGEST_CANONICALIZATION
            ),
            "camera_group_path": normalized_group_path,
            "camera_group_scalar_attrs": scalar_attrs,
            "camera_group_scalar_attrs_sha256": (
                camera_group_scalar_attrs_digest(scalar_attrs)
            ),
            "camera_group_scalar_attrs_digest_canonicalization": (
                SOURCE_CAMERA_ATTRS_DIGEST_CANONICALIZATION
            ),
            "z_eff_mm_source_field": Z_EFF_MM_SOURCE_FIELD,
            "z_eff_mm_source_value": z_eff_source_value,
            "z_eff_mm_derivation": Z_EFF_MM_DERIVATION,
        }
    )
    return VerifiedSelectedCameraSourceEvidence(
        **vars(parsed),
        _seal=_VERIFIED_SOURCE_EVIDENCE_SEAL,
    )


def camera_calibration_attrs(
    source_camera: SelectedCameraSourceEvidence,
) -> dict[str, Any]:
    """Build camera attrs only from one validated selected-camera source record."""

    if not isinstance(source_camera, SelectedCameraSourceEvidence):
        raise SelectedCalibrationError(
            "camera_calibration_attrs requires parsed selected-camera evidence."
        )
    evidence = parse_selected_camera_source_evidence(source_camera)
    payload: dict[str, Any] = {
        "schema_id": CAMERA_CALIBRATION_SCHEMA_ID,
        "schema_version": CAMERA_CALIBRATION_SCHEMA_VERSION,
        CAMERA_ID_ATTR: evidence.active_camera_id,
        "native_width_px": evidence.native_width_px,
        "native_height_px": evidence.native_height_px,
        "pixel_to_mm_derivation": PIXEL_TO_MM_DERIVATION,
    }
    candidates = {
        "pixels_per_mm_camera": evidence.pixels_per_mm_camera,
        "pixel_to_mm": evidence.pixel_to_mm,
        "pixels_per_mm_projector": evidence.pixels_per_mm_projector,
        "z_eff_mm": evidence.z_eff_mm,
    }
    for name, value in candidates.items():
        if value is not None:
            payload[name] = _positive_scalar(value, field_name=name)
    _validate_scalar_reciprocity(payload)
    return payload


def _require_attrs(node: Any, *, path: str) -> Mapping[str, Any]:
    attrs = getattr(node, "attrs", None)
    if not isinstance(attrs, Mapping):
        raise SelectedCalibrationError(f"{path!r} must expose an attrs mapping.")
    return attrs


def _require_child(parent: Any, name: str, *, parent_path: str) -> Any:
    try:
        return parent[name]
    except (KeyError, TypeError, AttributeError) as exc:
        path = f"{parent_path}/{name}" if parent_path else name
        raise SelectedCalibrationError(f"Required calibration path {path!r} is missing.") from exc


def _require_schema(
    attrs: Mapping[str, Any],
    *,
    path: str,
    schema_id: str,
    schema_version: int,
) -> None:
    if attrs.get("schema_id") != schema_id:
        raise SelectedCalibrationError(
            f"{path!r} has unsupported or missing schema_id; expected {schema_id!r}."
        )
    version = attrs.get("schema_version")
    if isinstance(version, bool) or version != schema_version:
        raise SelectedCalibrationError(
            f"{path!r} has unsupported or missing schema_version; "
            f"expected {schema_version}."
        )


def _require_exact_attr(
    attrs: Mapping[str, Any],
    name: str,
    expected: str,
    *,
    path: str,
) -> None:
    actual = attrs.get(name)
    if actual != expected:
        raise SelectedCalibrationError(
            f"{path!r} attr {name!r} mismatch: expected {expected!r}, found {actual!r}."
        )


def _read_scalars(attrs: Mapping[str, Any]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for name in _SCALAR_ATTRS:
        values[name] = (
            _positive_scalar(attrs[name], field_name=name) if name in attrs else None
        )
    _validate_scalar_reciprocity(values)
    return values


def _validate_scalar_reciprocity(attrs: Mapping[str, Any]) -> None:
    ppm = attrs.get("pixels_per_mm_camera")
    pixel_to_mm = attrs.get("pixel_to_mm")
    if ppm is None or pixel_to_mm is None:
        return
    ppm_value = _positive_scalar(ppm, field_name="pixels_per_mm_camera")
    pixel_to_mm_value = _positive_scalar(pixel_to_mm, field_name="pixel_to_mm")
    if not math.isclose(
        ppm_value * pixel_to_mm_value,
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise SelectedCalibrationError(
            "pixels_per_mm_camera and pixel_to_mm disagree within one camera snapshot."
        )


def _camera_manifest_from_attrs(
    attrs: Mapping[str, Any],
) -> CameraCalibrationManifest:
    scalars = _read_scalars(attrs)
    manifest = CameraCalibrationManifest(
        native_width_px=_positive_int(
            attrs.get("native_width_px"),
            field_name="native_width_px",
        ),
        native_height_px=_positive_int(
            attrs.get("native_height_px"),
            field_name="native_height_px",
        ),
        pixels_per_mm_camera=scalars["pixels_per_mm_camera"],
        pixel_to_mm=scalars["pixel_to_mm"],
        pixel_to_mm_derivation=_required_text(
            attrs.get("pixel_to_mm_derivation"),
            field_name="pixel_to_mm_derivation",
        ),
        pixels_per_mm_projector=scalars["pixels_per_mm_projector"],
        z_eff_mm=scalars["z_eff_mm"],
    )
    _validate_camera_manifest_derivations(manifest)
    return manifest


def _validate_camera_manifest_derivations(
    manifest: CameraCalibrationManifest,
) -> None:
    if manifest.pixel_to_mm_derivation != PIXEL_TO_MM_DERIVATION:
        raise SelectedCalibrationError(
            "camera_calibration.pixel_to_mm_derivation is unsupported."
        )
    expected_pixel_to_mm = (
        None
        if manifest.pixels_per_mm_camera is None
        else 1.0 / manifest.pixels_per_mm_camera
    )
    if manifest.pixel_to_mm != expected_pixel_to_mm:
        raise SelectedCalibrationError(
            "camera_calibration.pixel_to_mm is not the exact declared reciprocal."
        )


def _parse_camera_manifest(value: Any) -> CameraCalibrationManifest:
    payload = _exact_mapping(
        value,
        fields=_CAMERA_MANIFEST_FIELDS,
        field_name="camera_calibration",
    )
    manifest = CameraCalibrationManifest(
        native_width_px=_positive_int(
            payload["native_width_px"],
            field_name="camera_calibration.native_width_px",
        ),
        native_height_px=_positive_int(
            payload["native_height_px"],
            field_name="camera_calibration.native_height_px",
        ),
        pixels_per_mm_camera=_optional_positive_scalar(
            payload["pixels_per_mm_camera"],
            field_name="camera_calibration.pixels_per_mm_camera",
        ),
        pixel_to_mm=_optional_positive_scalar(
            payload["pixel_to_mm"],
            field_name="camera_calibration.pixel_to_mm",
        ),
        pixel_to_mm_derivation=_required_text(
            payload["pixel_to_mm_derivation"],
            field_name="camera_calibration.pixel_to_mm_derivation",
        ),
        pixels_per_mm_projector=_optional_positive_scalar(
            payload["pixels_per_mm_projector"],
            field_name="camera_calibration.pixels_per_mm_projector",
        ),
        z_eff_mm=_optional_positive_scalar(
            payload["z_eff_mm"],
            field_name="camera_calibration.z_eff_mm",
        ),
    )
    _validate_camera_manifest_derivations(manifest)
    _validate_scalar_reciprocity(manifest.to_dict())
    return manifest


def _parse_display_manifest(value: Any) -> DisplaySnapshotManifest:
    payload = _exact_mapping(
        value,
        fields=_DISPLAY_MANIFEST_FIELDS,
        field_name="display_snapshot",
    )
    ref = _canonical_archive_ref(
        payload["ref"],
        field_name="display_snapshot.ref",
    )
    output_name = _required_text(
        payload["selected_output_name"],
        field_name="display_snapshot.selected_output_name",
    )
    geometry = _required_text(
        payload["selected_output_geometry"],
        field_name="display_snapshot.selected_output_geometry",
    )
    token = _required_text(
        payload["selected_output_transform_token"],
        field_name="display_snapshot.selected_output_transform_token",
    )
    if token != SUPPORTED_OUTPUT_TRANSFORM_TOKEN:
        raise SelectedCalibrationError(
            "Only selected_output_transform_token='normal' is supported."
        )
    width, height, offset_x, offset_y = parse_selected_output_geometry(geometry)
    parsed = DisplaySnapshotManifest(
        ref=ref,
        selected_output_name=output_name,
        selected_output_geometry=geometry,
        selected_output_transform_token=token,
        width_px=_positive_int(
            payload["width_px"],
            field_name="display_snapshot.width_px",
        ),
        height_px=_positive_int(
            payload["height_px"],
            field_name="display_snapshot.height_px",
        ),
        offset_x_px=_signed_int(
            payload["offset_x_px"],
            field_name="display_snapshot.offset_x_px",
        ),
        offset_y_px=_signed_int(
            payload["offset_y_px"],
            field_name="display_snapshot.offset_y_px",
        ),
        source_h5_dataset_path=_absolute_source_path(
            payload["source_h5_dataset_path"],
            field_name="display_snapshot.source_h5_dataset_path",
        ),
        source_h5_dataset_sha256=_required_text(
            payload["source_h5_dataset_sha256"],
            field_name="display_snapshot.source_h5_dataset_sha256",
        ),
        source_h5_dataset_digest_canonicalization=_required_text(
            payload["source_h5_dataset_digest_canonicalization"],
            field_name=(
                "display_snapshot.source_h5_dataset_digest_canonicalization"
            ),
        ),
    )
    if (
        parsed.width_px,
        parsed.height_px,
        parsed.offset_x_px,
        parsed.offset_y_px,
    ) != (width, height, offset_x, offset_y):
        raise SelectedCalibrationError(
            "Parsed display dimensions/offsets do not match selected_output_geometry."
        )
    if parsed.source_h5_dataset_path != "/display_snapshot/selected_output_block":
        raise SelectedCalibrationError(
            "Display source path does not identify the selected H5 output block."
        )
    if _SHA256_RE.fullmatch(parsed.source_h5_dataset_sha256) is None:
        raise SelectedCalibrationError(
            "Display source digest must be a lowercase SHA-256 digest."
        )
    if (
        parsed.source_h5_dataset_digest_canonicalization
        != SOURCE_DISPLAY_DATASET_DIGEST_CANONICALIZATION
    ):
        raise SelectedCalibrationError(
            "Unsupported display source digest canonicalization."
        )
    return parsed


def parse_source_homography_semantics(
    value: Any,
    *,
    camera_id: str,
) -> SourceHomographySemantics:
    """Parse the canonical H5 semantic evidence for one selected homography."""

    if isinstance(value, SourceHomographySemantics):
        value = value.to_dict()
    payload = _exact_mapping(
        value,
        fields=_SOURCE_HOMOGRAPHY_SEMANTICS_FIELDS,
        field_name="source_homography_semantics",
    )
    if payload["schema_id"] != SOURCE_HOMOGRAPHY_SEMANTICS_SCHEMA_ID:
        raise SelectedCalibrationError(
            "Unsupported source-homography semantics schema_id."
        )
    version = payload["schema_version"]
    if (
        isinstance(version, bool)
        or version != SOURCE_HOMOGRAPHY_SEMANTICS_SCHEMA_VERSION
    ):
        raise SelectedCalibrationError(
            "Unsupported source-homography semantics schema_version."
        )

    camera = _path_segment(camera_id, field_name="camera_id")
    semantic_camera = _path_segment(
        payload["camera_id"],
        field_name="source_homography_semantics.camera_id",
    )
    if semantic_camera != camera:
        raise SelectedCalibrationError(
            "Source homography semantics camera_id does not match the selected camera."
        )
    controlled_text = {
        "source_frame": CANONICAL_SOURCE_FRAME,
        "dest_frame": CANONICAL_DEST_FRAME,
        "axes": CANONICAL_AXES,
        "coordinate_origin": CANONICAL_COORDINATE_ORIGIN,
        "image_space": CANONICAL_IMAGE_SPACE,
        "matrix_agreement": CANONICAL_MATRIX_AGREEMENT,
    }
    normalized: dict[str, str] = {}
    for name, expected in controlled_text.items():
        actual = _required_text(
            payload[name],
            field_name=f"source_homography_semantics.{name}",
        )
        if actual != expected:
            raise SelectedCalibrationError(
                f"Source homography semantic {name!r} must be {expected!r}."
            )
        normalized[name] = actual

    numeric_path = _absolute_source_path(
        payload["numeric_dataset_path"],
        field_name="source_homography_semantics.numeric_dataset_path",
    )
    yaml_path = _absolute_source_path(
        payload["yaml_dataset_path"],
        field_name="source_homography_semantics.yaml_dataset_path",
    )
    expected_prefix = f"/calibration_snapshot/{camera}"
    if numeric_path != f"{expected_prefix}/homography_matrix":
        raise SelectedCalibrationError(
            "Numeric homography dataset does not identify the selected camera."
        )
    if yaml_path != f"{expected_prefix}/homography_matrix_yml":
        raise SelectedCalibrationError(
            "YAML homography dataset does not identify the selected camera."
        )
    numeric_digest = _required_text(
        payload["numeric_matrix_sha256"],
        field_name="source_homography_semantics.numeric_matrix_sha256",
    )
    yaml_digest = _required_text(
        payload["yaml_matrix_sha256"],
        field_name="source_homography_semantics.yaml_matrix_sha256",
    )
    if (
        _SHA256_RE.fullmatch(numeric_digest) is None
        or _SHA256_RE.fullmatch(yaml_digest) is None
    ):
        raise SelectedCalibrationError(
            "Source homography matrix digests must be lowercase SHA-256 values."
        )
    if numeric_digest != yaml_digest:
        raise SelectedCalibrationError(
            "Numeric and YAML homography matrices do not agree exactly."
        )
    numeric_payload_source = _required_text(
        payload["numeric_payload_source"],
        field_name="source_homography_semantics.numeric_payload_source",
    )
    yaml_payload_source = _required_text(
        payload["yaml_payload_source"],
        field_name="source_homography_semantics.yaml_payload_source",
    )
    if numeric_payload_source != NUMERIC_HOMOGRAPHY_PAYLOAD_SOURCE:
        raise SelectedCalibrationError("Unsupported numeric homography payload source.")
    if yaml_payload_source != YAML_HOMOGRAPHY_PAYLOAD_SOURCE:
        raise SelectedCalibrationError("Unsupported YAML homography payload source.")

    return SourceHomographySemantics(
        source_frame=normalized["source_frame"],
        dest_frame=normalized["dest_frame"],
        axes=normalized["axes"],
        coordinate_origin=normalized["coordinate_origin"],
        image_space=normalized["image_space"],
        camera_id=semantic_camera,
        canvas_name=_required_text(
            payload["canvas_name"],
            field_name="source_homography_semantics.canvas_name",
        ),
        numeric_dataset_path=numeric_path,
        yaml_dataset_path=yaml_path,
        numeric_matrix_sha256=numeric_digest,
        yaml_matrix_sha256=yaml_digest,
        matrix_agreement=normalized["matrix_agreement"],
        numeric_payload_source=numeric_payload_source,
        yaml_payload_source=yaml_payload_source,
    )


def canonical_source_homography_semantics_json(value: Any) -> str:
    """Serialize canonical source semantics for independent digest binding."""

    camera_id = (
        value.camera_id
        if isinstance(value, SourceHomographySemantics)
        else value.get("camera_id") if isinstance(value, Mapping) else None
    )
    semantics = parse_source_homography_semantics(value, camera_id=camera_id)
    return json.dumps(
        semantics.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def source_homography_semantics_digest(value: Any) -> str:
    canonical = canonical_source_homography_semantics_json(value).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def parse_source_artifact_lineage(
    value: Any,
    *,
    camera_id: str,
) -> SourceArtifactLineage:
    """Parse the exact H5/calibration-artifact lineage required by strict imports."""

    if isinstance(value, SourceArtifactLineage):
        value = value.to_dict()
    payload = _exact_mapping(
        value,
        fields=_SOURCE_ARTIFACT_FIELDS,
        field_name="source_artifact",
    )
    camera = _path_segment(camera_id, field_name="camera_id")
    semantics = parse_source_homography_semantics(
        payload["source_homography_semantics"],
        camera_id=camera,
    )
    semantics_digest = _required_text(
        payload["source_homography_semantics_sha256"],
        field_name="source_artifact.source_homography_semantics_sha256",
    )
    if _SHA256_RE.fullmatch(semantics_digest) is None:
        raise SelectedCalibrationError(
            "Source homography semantics digest must be a lowercase SHA-256 value."
        )
    if semantics.digest() != semantics_digest:
        raise SelectedCalibrationError(
            "Source homography semantics digest does not match its content."
        )
    algorithm = _required_text(
        payload["homography_artifact_checksum_algorithm"],
        field_name="source_artifact.homography_artifact_checksum_algorithm",
    )
    if algorithm not in {"fnv1a64", "sha256"}:
        raise SelectedCalibrationError(
            "Unsupported homography artifact checksum algorithm."
        )
    checksum = _required_text(
        payload["homography_artifact_checksum"],
        field_name="source_artifact.homography_artifact_checksum",
    )
    expected_length = 16 if algorithm == "fnv1a64" else 64
    if len(checksum) != expected_length or _LOWER_HEX_RE.fullmatch(checksum) is None:
        raise SelectedCalibrationError(
            "Homography artifact checksum is invalid for its declared algorithm."
        )
    provenance_schema = _required_text(
        payload["homography_provenance_schema"],
        field_name="source_artifact.homography_provenance_schema",
    )
    if provenance_schema != "citrus.homography_provenance.v1":
        raise SelectedCalibrationError("Unsupported homography provenance schema.")
    return SourceArtifactLineage(
        source_h5_path=_absolute_source_path(
            payload["source_h5_path"],
            field_name="source_artifact.source_h5_path",
        ),
        source_homography_semantics=semantics,
        source_homography_semantics_sha256=semantics_digest,
        homography_artifact_path=_absolute_source_path(
            payload["homography_artifact_path"],
            field_name="source_artifact.homography_artifact_path",
        ),
        homography_artifact_checksum_algorithm=algorithm,
        homography_artifact_checksum=checksum,
        homography_artifact_size_bytes=_positive_int(
            payload["homography_artifact_size_bytes"],
            field_name="source_artifact.homography_artifact_size_bytes",
        ),
        homography_artifact_mtime_unix_ns=_nonnegative_int(
            payload["homography_artifact_mtime_unix_ns"],
            field_name="source_artifact.homography_artifact_mtime_unix_ns",
        ),
        homography_provenance_schema=provenance_schema,
    )


def parse_selected_calibration_manifest(value: Any) -> SelectedCalibrationManifest:
    """Strictly parse a canonical selected-calibration manifest."""

    if isinstance(value, SelectedCalibrationManifest):
        value = value.to_dict()
    if isinstance(value, (bytes, bytearray)):
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SelectedCalibrationError("Manifest bytes must be UTF-8.") from exc
    if isinstance(value, str):
        value = _load_json_without_duplicate_keys(
            value,
            field_name="selected_calibration_manifest",
            invalid_message="Manifest JSON is invalid.",
        )
    payload = _exact_mapping(
        value,
        fields=_MANIFEST_FIELDS,
        field_name="selected_calibration_manifest",
    )
    if payload["schema_id"] != SELECTED_CALIBRATION_MANIFEST_SCHEMA_ID:
        raise SelectedCalibrationError("Unsupported selected-calibration manifest schema_id.")
    version = payload["schema_version"]
    if (
        isinstance(version, bool)
        or version != SELECTED_CALIBRATION_MANIFEST_SCHEMA_VERSION
    ):
        raise SelectedCalibrationError(
            "Unsupported selected-calibration manifest schema_version."
        )
    stimulus_run = _path_segment(payload["stimulus_run"], field_name="stimulus_run")
    camera_id = _path_segment(payload["camera_id"], field_name="camera_id")
    paths = selected_calibration_paths(
        stimulus_run=stimulus_run,
        camera_id=camera_id,
    )
    camera_ref = _canonical_archive_ref(
        payload["camera_calibration_ref"],
        field_name="camera_calibration_ref",
    )
    transform_ref = _canonical_archive_ref(
        payload["transform_ref"],
        field_name="transform_ref",
    )
    if camera_ref != paths.camera_calibration_path:
        raise SelectedCalibrationError("Manifest camera_calibration_ref mismatch.")
    if transform_ref != paths.homography_array_path:
        raise SelectedCalibrationError("Manifest transform_ref mismatch.")
    transform_sha256 = _required_text(
        payload["transform_sha256"],
        field_name="transform_sha256",
    )
    if _SHA256_RE.fullmatch(transform_sha256) is None:
        raise SelectedCalibrationError("Manifest transform_sha256 is invalid.")
    matrix_sha256 = _required_text(
        payload["matrix_sha256"],
        field_name="matrix_sha256",
    )
    if _SHA256_RE.fullmatch(matrix_sha256) is None:
        raise SelectedCalibrationError("Manifest matrix_sha256 is invalid.")
    display_snapshot = _parse_display_manifest(payload["display_snapshot"])
    if display_snapshot.ref != paths.display_snapshot_path:
        raise SelectedCalibrationError("Manifest display snapshot ref mismatch.")
    source_artifact = parse_source_artifact_lineage(
        payload["source_artifact"],
        camera_id=camera_id,
    )
    source_camera = parse_selected_camera_source_evidence(payload["source_camera"])
    source_display = parse_selected_display_source_evidence(payload["source_display"])
    source_homography = parse_selected_homography_source_evidence(
        payload["source_homography"]
    )
    if source_camera.active_camera_id != camera_id:
        raise SelectedCalibrationError(
            "Selected-camera source active_camera_id does not match the manifest."
        )
    if len(
        {
            source_camera.source_h5_path,
            source_display.source_h5_path,
            source_homography.source_h5_path,
            source_artifact.source_h5_path,
        }
    ) != 1:
        raise SelectedCalibrationError(
            "Camera, display, and homography evidence identify different source H5 files."
        )
    if source_homography.camera_id != camera_id:
        raise SelectedCalibrationError(
            "Selected-homography camera does not match the manifest camera."
        )
    derived_artifact = source_artifact_from_homography_evidence(source_homography)
    if source_artifact != derived_artifact:
        raise SelectedCalibrationError(
            "Source artifact lineage does not match exact homography evidence."
        )
    if (
        source_artifact.source_homography_semantics.numeric_matrix_sha256
        != matrix_sha256
    ):
        raise SelectedCalibrationError(
            "Source homography matrix digest does not match the selected matrix."
        )
    if source_homography.numeric_matrix_sha256 != matrix_sha256:
        raise SelectedCalibrationError(
            "Selected-homography evidence does not match the manifest matrix."
        )
    expected_display = DisplaySnapshotManifest(
        ref=paths.display_snapshot_path,
        selected_output_name=source_display.selected_output_name,
        selected_output_geometry=source_display.selected_output_geometry,
        selected_output_transform_token=source_display.selected_output_transform_token,
        width_px=source_display.width_px,
        height_px=source_display.height_px,
        offset_x_px=source_display.offset_x_px,
        offset_y_px=source_display.offset_y_px,
        source_h5_dataset_path=source_display.selected_output_dataset_path,
        source_h5_dataset_sha256=source_display.selected_output_dataset_sha256,
        source_h5_dataset_digest_canonicalization=(
            source_display.selected_output_dataset_digest_canonicalization
        ),
    )
    if display_snapshot != expected_display:
        raise SelectedCalibrationError(
            "Manifest display values do not match exact selected-output evidence."
        )
    camera_calibration = _parse_camera_manifest(payload["camera_calibration"])
    source_camera_calibration = _camera_manifest_from_attrs(
        camera_calibration_attrs(source_camera)
    )
    if camera_calibration != source_camera_calibration:
        raise SelectedCalibrationError(
            "Manifest camera values do not match the exact selected-camera source."
        )
    return SelectedCalibrationManifest(
        stimulus_run=stimulus_run,
        camera_id=camera_id,
        camera_calibration_ref=camera_ref,
        transform_ref=transform_ref,
        transform_sha256=transform_sha256,
        matrix_sha256=matrix_sha256,
        camera_calibration=camera_calibration,
        display_snapshot=display_snapshot,
        source_artifact=source_artifact,
        source_camera=source_camera,
        source_display=source_display,
        source_homography=source_homography,
    )


def build_selected_calibration_manifest(
    *,
    stimulus_run: str,
    camera_id: str,
    transform_sha256: str,
    matrix_sha256: str,
    camera_calibration: CameraCalibrationManifest | Mapping[str, Any],
    display_snapshot: DisplaySnapshotManifest | Mapping[str, Any],
    source_artifact: SourceArtifactLineage | Mapping[str, Any],
    source_camera: SelectedCameraSourceEvidence,
    source_display: SelectedDisplaySourceEvidence,
    source_homography: SelectedHomographySourceEvidence,
) -> SelectedCalibrationManifest:
    """Build a canonical manifest for a writer after all evidence is persisted."""

    paths = selected_calibration_paths(
        stimulus_run=stimulus_run,
        camera_id=camera_id,
    )
    payload = {
        "schema_id": SELECTED_CALIBRATION_MANIFEST_SCHEMA_ID,
        "schema_version": SELECTED_CALIBRATION_MANIFEST_SCHEMA_VERSION,
        "stimulus_run": stimulus_run,
        "camera_id": camera_id,
        "camera_calibration_ref": paths.camera_calibration_path,
        "transform_ref": paths.homography_array_path,
        "transform_sha256": transform_sha256,
        "matrix_sha256": matrix_sha256,
        "camera_calibration": (
            camera_calibration.to_dict()
            if isinstance(camera_calibration, CameraCalibrationManifest)
            else camera_calibration
        ),
        "display_snapshot": (
            display_snapshot.to_dict()
            if isinstance(display_snapshot, DisplaySnapshotManifest)
            else display_snapshot
        ),
        "source_artifact": (
            source_artifact.to_dict()
            if isinstance(source_artifact, SourceArtifactLineage)
            else source_artifact
        ),
        "source_camera": (
            source_camera.to_dict()
            if isinstance(source_camera, SelectedCameraSourceEvidence)
            else source_camera
        ),
        "source_display": (
            source_display.to_dict()
            if isinstance(source_display, SelectedDisplaySourceEvidence)
            else source_display
        ),
        "source_homography": (
            source_homography.to_dict()
            if isinstance(source_homography, SelectedHomographySourceEvidence)
            else source_homography
        ),
    }
    return parse_selected_calibration_manifest(payload)


def canonical_selected_calibration_manifest_json(value: Any) -> str:
    manifest = parse_selected_calibration_manifest(value)
    return json.dumps(
        manifest.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def selected_calibration_manifest_digest(value: Any) -> str:
    canonical = canonical_selected_calibration_manifest_json(value).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def selected_calibration_manifest_attrs(value: Any) -> dict[str, Any]:
    manifest = parse_selected_calibration_manifest(value)
    return {
        SELECTED_CALIBRATION_MANIFEST_ATTR: manifest.to_dict(),
        (
            f"{SELECTED_CALIBRATION_MANIFEST_ATTR}"
            f"{SELECTED_CALIBRATION_MANIFEST_DIGEST_SUFFIX}"
        ): manifest.digest(),
    }


def load_selected_calibration_manifest_attrs(
    attrs: Mapping[str, Any],
) -> SelectedCalibrationManifest:
    digest_name = (
        f"{SELECTED_CALIBRATION_MANIFEST_ATTR}"
        f"{SELECTED_CALIBRATION_MANIFEST_DIGEST_SUFFIX}"
    )
    if SELECTED_CALIBRATION_MANIFEST_ATTR not in attrs:
        raise SelectedCalibrationError("Selected-calibration manifest attr is missing.")
    if digest_name not in attrs:
        raise SelectedCalibrationError(
            "Selected-calibration manifest digest attr is missing."
        )
    manifest = parse_selected_calibration_manifest(
        attrs[SELECTED_CALIBRATION_MANIFEST_ATTR]
    )
    stored_digest = _required_text(attrs[digest_name], field_name=digest_name)
    if _SHA256_RE.fullmatch(stored_digest) is None:
        raise SelectedCalibrationError("Selected-calibration manifest digest is invalid.")
    if manifest.digest() != stored_digest:
        raise SelectedCalibrationError(
            "Selected-calibration manifest digest does not match its content."
        )
    return manifest


def _require_exact_node_path(node: Any, *, expected_path: str) -> None:
    actual = getattr(node, "path", None)
    if actual != expected_path:
        raise SelectedCalibrationError(
            f"Node path mismatch: expected {expected_path!r}, found {actual!r}."
        )


def _require_mutable_attrs(node: Any, *, path: str) -> Any:
    attrs = getattr(node, "attrs", None)
    if attrs is None or not hasattr(attrs, "update") or not hasattr(attrs, "__delitem__"):
        raise SelectedCalibrationError(f"{path!r} must expose mutable attrs.")
    return attrs


def _require_sealed_source_evidence(
    value: Any,
    *,
    verified_type: type[Any],
    evidence_name: str,
) -> None:
    if (
        type(value) is not verified_type
        or getattr(value, "_source_evidence_seal", None)
        is not _VERIFIED_SOURCE_EVIDENCE_SEAL
    ):
        raise SelectedCalibrationError(
            f"Stamper requires builder-validated {evidence_name} source evidence."
        )


def _snapshot_attrs_for_transaction(
    targets: tuple[tuple[str, Any], ...],
) -> tuple[tuple[str, Any, dict[str, Any]], ...]:
    snapshots: list[tuple[str, Any, dict[str, Any]]] = []
    for path, attrs in targets:
        try:
            snapshot = copy.deepcopy(dict(attrs))
        except Exception as exc:
            raise SelectedCalibrationError(
                f"Unable to snapshot attrs for {path!r} before stamping."
            ) from exc
        snapshots.append((path, attrs, snapshot))
    return tuple(snapshots)


def _restore_attrs_after_failed_transaction(
    snapshots: tuple[tuple[str, Any, dict[str, Any]], ...],
) -> tuple[str, ...]:
    """Best-effort exact rollback; return targets that could not be restored."""

    failures: list[str] = []
    for path, attrs, snapshot in snapshots:
        try:
            for name in tuple(attrs.keys()):
                if name not in snapshot:
                    del attrs[name]
            attrs.update(copy.deepcopy(snapshot))
            if dict(attrs) != snapshot:
                raise RuntimeError("restored attrs differ from the pre-call snapshot")
        except Exception as exc:  # pragma: no cover - exercised by hostile fakes
            failures.append(f"{path}: {type(exc).__name__}: {exc}")
    return tuple(failures)


def stamp_selected_calibration_snapshot(
    calibration_group: Any,
    camera_group: Any,
    display_snapshot_group: Any,
    homography_array: Any,
    *,
    stimulus_run: str,
    camera_id: str,
    source_camera: VerifiedSelectedCameraSourceEvidence,
    source_display: VerifiedSelectedDisplaySourceEvidence,
    source_homography: VerifiedSelectedHomographySourceEvidence,
) -> SelectedCalibrationManifest:
    """Derive and stamp one snapshot from three builder-verified H5 sources."""

    paths = selected_calibration_paths(
        stimulus_run=stimulus_run,
        camera_id=camera_id,
    )
    for node, expected_path in (
        (calibration_group, paths.calibration_path),
        (camera_group, paths.camera_calibration_path),
        (display_snapshot_group, paths.display_snapshot_path),
        (homography_array, paths.homography_array_path),
    ):
        _require_exact_node_path(node, expected_path=expected_path)

    # Preflight every mutation target before validating any remaining evidence.
    # No attribute deletion or update may happen before all four targets pass.
    camera_attrs = _require_mutable_attrs(
        camera_group,
        path=paths.camera_calibration_path,
    )
    calibration_attrs = _require_mutable_attrs(
        calibration_group,
        path=paths.calibration_path,
    )
    display_attrs = _require_mutable_attrs(
        display_snapshot_group,
        path=paths.display_snapshot_path,
    )
    homography_attrs = _require_mutable_attrs(
        homography_array,
        path=paths.homography_array_path,
    )
    _require_sealed_source_evidence(
        source_camera,
        verified_type=VerifiedSelectedCameraSourceEvidence,
        evidence_name="selected-camera",
    )
    selected_camera = parse_selected_camera_source_evidence(source_camera)
    if selected_camera.active_camera_id != camera_id:
        raise SelectedCalibrationError(
            "Selected-camera evidence does not match the requested camera_id."
        )
    _require_sealed_source_evidence(
        source_display,
        verified_type=VerifiedSelectedDisplaySourceEvidence,
        evidence_name="selected-display",
    )
    selected_display = parse_selected_display_source_evidence(source_display)
    _require_sealed_source_evidence(
        source_homography,
        verified_type=VerifiedSelectedHomographySourceEvidence,
        evidence_name="selected-homography",
    )
    selected_homography = parse_selected_homography_source_evidence(
        source_homography
    )
    if selected_homography.camera_id != camera_id:
        raise SelectedCalibrationError(
            "Selected-homography evidence does not match the requested camera_id."
        )
    if len(
        {
            selected_camera.source_h5_path,
            selected_display.source_h5_path,
            selected_homography.source_h5_path,
        }
    ) != 1:
        raise SelectedCalibrationError(
            "Camera, display, and homography evidence identify different source H5 files."
        )
    display_payload: dict[str, Any] = {
        SELECTED_OUTPUT_NAME_ATTR: selected_display.selected_output_name,
        "selected_output_connection_state": (
            selected_display.selected_output_connection_state
        ),
        SELECTED_OUTPUT_GEOMETRY_ATTR: selected_display.selected_output_geometry,
        SELECTED_OUTPUT_TRANSFORM_TOKEN_ATTR: (
            selected_display.selected_output_transform_token
        ),
        "selected_output_transform_raw": (
            selected_display.selected_output_transform_raw
        ),
        SOURCE_DISPLAY_DATASET_PATH_ATTR: (
            selected_display.selected_output_dataset_path
        ),
        SOURCE_DISPLAY_DATASET_SHA256_ATTR: (
            selected_display.selected_output_dataset_sha256
        ),
        SOURCE_DISPLAY_DATASET_DIGEST_CANONICALIZATION_ATTR: (
            selected_display.selected_output_dataset_digest_canonicalization
        ),
    }
    display_payload.update(selected_display_evidence_attrs(selected_display))
    display_manifest = _display_manifest_from_attrs(
        display_payload,
        display_ref=paths.display_snapshot_path,
    )
    source_extent = TransformReferenceExtent(
        width=selected_camera.native_width_px,
        height=selected_camera.native_height_px,
        units="px",
        authority=(
            f"{paths.camera_calibration_path}@native_width_px,native_height_px"
        ),
    )
    target_extent = TransformReferenceExtent(
        width=selected_display.width_px,
        height=selected_display.height_px,
        units="px",
        authority=(
            f"{paths.display_snapshot_path}@{SELECTED_OUTPUT_GEOMETRY_ATTR}"
        ),
    )
    try:
        persisted_matrix = homography_array[:]
    except Exception as exc:
        raise SelectedCalibrationError(
            "Unable to read persisted homography matrix before stamping."
        ) from exc
    persisted_matrix_sha256 = homography_matrix_sha256(persisted_matrix)
    if persisted_matrix_sha256 != selected_homography.numeric_matrix_sha256:
        raise SelectedCalibrationError(
            "Persisted matrix does not match the canonical H5 numeric/YAML evidence."
        )
    transform = build_directed_homography(
        transform_id=f"camera_{camera_id}_to_stimulus_canvas_{stimulus_run}_v1",
        matrix=selected_homography.matrix,
        from_space_id=CANONICAL_HOMOGRAPHY_FROM_SPACE_ID,
        to_space_id=CANONICAL_HOMOGRAPHY_TO_SPACE_ID,
        source_reference_extent=source_extent,
        target_reference_extent=target_extent,
        calibration_ref=paths.camera_calibration_path,
        camera_id=camera_id,
    )
    lineage = source_artifact_from_homography_evidence(selected_homography)
    homography_payload = directed_homography_attrs(transform)
    homography_payload.update(
        selected_homography_evidence_attrs(selected_homography)
    )

    camera_payload = camera_calibration_attrs(selected_camera)
    camera_manifest = _camera_manifest_from_attrs(camera_payload)
    manifest = build_selected_calibration_manifest(
        stimulus_run=stimulus_run,
        camera_id=camera_id,
        transform_sha256=transform.digest(),
        matrix_sha256=persisted_matrix_sha256,
        camera_calibration=camera_manifest,
        display_snapshot=display_manifest,
        source_artifact=lineage,
        source_camera=selected_camera,
        source_display=selected_display,
        source_homography=selected_homography,
    )
    transaction_snapshots = _snapshot_attrs_for_transaction(
        (
            (paths.camera_calibration_path, camera_attrs),
            (paths.display_snapshot_path, display_attrs),
            (paths.homography_array_path, homography_attrs),
            (paths.calibration_path, calibration_attrs),
        )
    )
    try:
        for name in (
            *_SCALAR_ATTRS,
            "native_width_px",
            "native_height_px",
            "pixel_to_mm_derivation",
        ):
            if name in camera_attrs:
                del camera_attrs[name]
        camera_attrs.update(camera_payload)
        display_attrs.update(display_payload)
        homography_attrs.update(homography_payload)
        calibration_attrs.update(
            selected_calibration_pointer_attrs(
                stimulus_run=stimulus_run,
                camera_id=camera_id,
                transform_sha256=transform.digest(),
            )
        )
        calibration_attrs.update(selected_calibration_manifest_attrs(manifest))
    except Exception as exc:
        rollback_failures = _restore_attrs_after_failed_transaction(
            transaction_snapshots
        )
        if rollback_failures:
            details = "; ".join(rollback_failures)
            raise SelectedCalibrationError(
                "Selected-calibration attr stamping failed and exact rollback was "
                f"incomplete: {details}"
            ) from exc
        raise SelectedCalibrationError(
            "Selected-calibration attr stamping failed; all attr targets were "
            "restored to their exact pre-call state."
        ) from exc
    return manifest


def load_selected_calibration_snapshot(
    root: Any,
    *,
    stimulus_run: str,
    expected_camera_id: str,
    expected_from_space_id: str,
    expected_to_space_id: str,
    expected_source_reference_extent: TransformReferenceExtent | Mapping[str, Any],
    expected_target_reference_extent: TransformReferenceExtent | Mapping[str, Any],
) -> SelectedCalibrationSnapshot:
    """Load exactly one selected run/camera snapshot and reject all mismatches."""

    if (
        expected_from_space_id != CANONICAL_HOMOGRAPHY_FROM_SPACE_ID
        or expected_to_space_id != CANONICAL_HOMOGRAPHY_TO_SPACE_ID
    ):
        raise SelectedCalibrationError(
            "Canonical direction mismatch: selected-calibration v1 only supports "
            f"{CANONICAL_HOMOGRAPHY_FROM_SPACE_ID!r} to "
            f"{CANONICAL_HOMOGRAPHY_TO_SPACE_ID!r} direction."
        )
    paths = selected_calibration_paths(
        stimulus_run=stimulus_run,
        camera_id=expected_camera_id,
    )
    analysis = _require_child(root, "analysis", parent_path="")
    runs = _require_child(analysis, "stimulus_runs", parent_path="analysis")
    run = _require_child(
        runs,
        stimulus_run,
        parent_path="analysis/stimulus_runs",
    )
    display_snapshot_group = _require_child(
        run,
        DISPLAY_SNAPSHOT_GROUP_NAME,
        parent_path=paths.stimulus_run_path,
    )
    display_attrs = _require_attrs(
        display_snapshot_group,
        path=paths.display_snapshot_path,
    )
    display_manifest = _display_manifest_from_attrs(
        display_attrs,
        display_ref=paths.display_snapshot_path,
    )
    persisted_display_evidence = load_selected_display_evidence_attrs(display_attrs)
    calibration = _require_child(
        run,
        "calibration",
        parent_path=paths.stimulus_run_path,
    )
    calibration_attrs = _require_attrs(calibration, path=paths.calibration_path)
    _require_schema(
        calibration_attrs,
        path=paths.calibration_path,
        schema_id=SELECTED_CALIBRATION_SCHEMA_ID,
        schema_version=SELECTED_CALIBRATION_SCHEMA_VERSION,
    )
    for name, expected in (
        (STIMULUS_RUN_ATTR, stimulus_run),
        (ACTIVE_CAMERA_ID_ATTR, expected_camera_id),
        (ACTIVE_CAMERA_CALIBRATION_REF_ATTR, paths.camera_calibration_path),
        (ACTIVE_CAMERA_TRANSFORM_REF_ATTR, paths.homography_array_path),
    ):
        _require_exact_attr(
            calibration_attrs,
            name,
            expected,
            path=paths.calibration_path,
        )
    selected_digest = _required_text(
        calibration_attrs.get(ACTIVE_CAMERA_TRANSFORM_SHA256_ATTR),
        field_name=ACTIVE_CAMERA_TRANSFORM_SHA256_ATTR,
    )
    if _SHA256_RE.fullmatch(selected_digest) is None:
        raise SelectedCalibrationError(
            f"{ACTIVE_CAMERA_TRANSFORM_SHA256_ATTR} must be a lowercase SHA-256 digest."
        )
    manifest = load_selected_calibration_manifest_attrs(calibration_attrs)
    if manifest.stimulus_run != stimulus_run:
        raise SelectedCalibrationError("Manifest stimulus_run mismatch.")
    if manifest.camera_id != expected_camera_id:
        raise SelectedCalibrationError("Manifest camera_id mismatch.")
    if manifest.camera_calibration_ref != paths.camera_calibration_path:
        raise SelectedCalibrationError("Manifest camera_calibration_ref mismatch.")
    if manifest.transform_ref != paths.homography_array_path:
        raise SelectedCalibrationError("Manifest transform_ref mismatch.")
    if manifest.transform_sha256 != selected_digest:
        raise SelectedCalibrationError("Manifest transform digest pointer mismatch.")
    if manifest.display_snapshot != display_manifest:
        raise SelectedCalibrationError(
            "Persisted display snapshot does not match the selected manifest."
        )
    if manifest.source_display != persisted_display_evidence:
        raise SelectedCalibrationError(
            "Persisted display evidence does not match the selected manifest."
        )

    camera_group = _require_child(
        calibration,
        expected_camera_id,
        parent_path=paths.calibration_path,
    )
    camera_attrs = _require_attrs(
        camera_group,
        path=paths.camera_calibration_path,
    )
    _require_schema(
        camera_attrs,
        path=paths.camera_calibration_path,
        schema_id=CAMERA_CALIBRATION_SCHEMA_ID,
        schema_version=CAMERA_CALIBRATION_SCHEMA_VERSION,
    )
    _require_exact_attr(
        camera_attrs,
        CAMERA_ID_ATTR,
        expected_camera_id,
        path=paths.camera_calibration_path,
    )
    camera_manifest = _camera_manifest_from_attrs(camera_attrs)
    if manifest.camera_calibration != camera_manifest:
        raise SelectedCalibrationError(
            "Persisted camera calibration attrs do not match the selected manifest."
        )
    source_extent = TransformReferenceExtent(
        width=camera_manifest.native_width_px,
        height=camera_manifest.native_height_px,
        units="px",
        authority=(
            f"{paths.camera_calibration_path}@native_width_px,native_height_px"
        ),
    )
    target_extent = TransformReferenceExtent(
        width=display_manifest.width_px,
        height=display_manifest.height_px,
        units="px",
        authority=(
            f"{paths.display_snapshot_path}@{SELECTED_OUTPUT_GEOMETRY_ATTR}"
        ),
    )
    matrix_node = _require_child(
        camera_group,
        HOMOGRAPHY_ARRAY_NAME,
        parent_path=paths.camera_calibration_path,
    )
    matrix_attrs = _require_attrs(matrix_node, path=paths.homography_array_path)
    persisted_homography_evidence = load_selected_homography_evidence_attrs(
        matrix_attrs
    )
    if manifest.source_homography != persisted_homography_evidence:
        raise SelectedCalibrationError(
            "Persisted homography evidence does not match the selected manifest."
        )
    try:
        homography = load_bound_directed_homography(
            matrix_node,
            array_path=paths.homography_array_path,
            expected_from_space_id=CANONICAL_HOMOGRAPHY_FROM_SPACE_ID,
            expected_to_space_id=CANONICAL_HOMOGRAPHY_TO_SPACE_ID,
            expected_camera_id=expected_camera_id,
            expected_source_reference_extent=expected_source_reference_extent,
            expected_target_reference_extent=expected_target_reference_extent,
            expected_transform_sha256=selected_digest,
            expected_calibration_ref=paths.camera_calibration_path,
        )
    except DirectedTransformError as exc:
        raise SelectedCalibrationError(
            f"Selected homography {paths.homography_array_path!r} is invalid: {exc}"
        ) from exc
    if homography.array_path != paths.homography_array_path:
        raise SelectedCalibrationError("Selected homography array path mismatch.")
    if homography.matrix_sha256 != manifest.matrix_sha256:
        raise SelectedCalibrationError(
            "Selected homography matrix digest does not match the manifest."
        )
    if homography.transform.source_reference_extent != source_extent:
        raise SelectedCalibrationError(
            "Persisted camera extent does not match the directed transform."
        )
    if homography.transform.target_reference_extent != target_extent:
        raise SelectedCalibrationError(
            "Persisted display extent does not match the directed transform."
        )

    manifest_digest = manifest.digest()
    return SelectedCalibrationSnapshot(
        stimulus_run=stimulus_run,
        paths=paths,
        camera_id=expected_camera_id,
        homography=homography,
        source_reference_extent=source_extent,
        target_reference_extent=target_extent,
        display_output_name=display_manifest.selected_output_name,
        display_output_geometry=display_manifest.selected_output_geometry,
        display_output_transform_token=(
            display_manifest.selected_output_transform_token
        ),
        pixels_per_mm_camera=camera_manifest.pixels_per_mm_camera,
        pixel_to_mm=camera_manifest.pixel_to_mm,
        pixels_per_mm_projector=camera_manifest.pixels_per_mm_projector,
        z_eff_mm=camera_manifest.z_eff_mm,
        manifest=manifest,
        manifest_sha256=manifest_digest,
    )


__all__ = [
    "SELECTED_CALIBRATION_SCHEMA_ID",
    "SELECTED_CALIBRATION_SCHEMA_VERSION",
    "CAMERA_CALIBRATION_SCHEMA_ID",
    "CAMERA_CALIBRATION_SCHEMA_VERSION",
    "SELECTED_CALIBRATION_MANIFEST_SCHEMA_ID",
    "SELECTED_CALIBRATION_MANIFEST_SCHEMA_VERSION",
    "SOURCE_HOMOGRAPHY_SEMANTICS_SCHEMA_ID",
    "SOURCE_HOMOGRAPHY_SEMANTICS_SCHEMA_VERSION",
    "SELECTED_CAMERA_SOURCE_SCHEMA_ID",
    "SELECTED_CAMERA_SOURCE_SCHEMA_VERSION",
    "SELECTED_DISPLAY_SOURCE_SCHEMA_ID",
    "SELECTED_DISPLAY_SOURCE_SCHEMA_VERSION",
    "SELECTED_HOMOGRAPHY_SOURCE_SCHEMA_ID",
    "SELECTED_HOMOGRAPHY_SOURCE_SCHEMA_VERSION",
    "CANONICAL_HOMOGRAPHY_FROM_SPACE_ID",
    "CANONICAL_HOMOGRAPHY_TO_SPACE_ID",
    "CANONICAL_SOURCE_FRAME",
    "CANONICAL_DEST_FRAME",
    "CANONICAL_AXES",
    "CANONICAL_COORDINATE_ORIGIN",
    "CANONICAL_IMAGE_SPACE",
    "CANONICAL_MATRIX_AGREEMENT",
    "STIMULUS_RUN_ATTR",
    "ACTIVE_CAMERA_ID_ATTR",
    "ACTIVE_CAMERA_CALIBRATION_REF_ATTR",
    "ACTIVE_CAMERA_TRANSFORM_REF_ATTR",
    "ACTIVE_CAMERA_TRANSFORM_SHA256_ATTR",
    "CAMERA_ID_ATTR",
    "HOMOGRAPHY_ARRAY_NAME",
    "DISPLAY_SNAPSHOT_GROUP_NAME",
    "SELECTED_OUTPUT_NAME_ATTR",
    "SELECTED_OUTPUT_GEOMETRY_ATTR",
    "SELECTED_OUTPUT_TRANSFORM_TOKEN_ATTR",
    "SOURCE_DISPLAY_DATASET_PATH_ATTR",
    "SOURCE_DISPLAY_DATASET_SHA256_ATTR",
    "SOURCE_DISPLAY_DATASET_DIGEST_CANONICALIZATION_ATTR",
    "SOURCE_DISPLAY_DATASET_DIGEST_CANONICALIZATION",
    "SOURCE_DISPLAY_GROUP_PATH",
    "SOURCE_DISPLAY_DATASET_PATH",
    "SOURCE_DISPLAY_ATTRS_DIGEST_CANONICALIZATION",
    "SOURCE_DISPLAY_EVIDENCE_ATTR",
    "SOURCE_DISPLAY_EVIDENCE_DIGEST_SUFFIX",
    "SOURCE_ARENA_CONFIG_DATASET_PATH",
    "SOURCE_ARENA_CONFIG_DIGEST_CANONICALIZATION",
    "SOURCE_CAMERA_RECORD_DIGEST_CANONICALIZATION",
    "SOURCE_CAMERA_ATTRS_DIGEST_CANONICALIZATION",
    "NUMERIC_HOMOGRAPHY_PAYLOAD_SOURCE",
    "YAML_HOMOGRAPHY_PAYLOAD_SOURCE",
    "YAML_HOMOGRAPHY_SERIALIZATION_FORMAT",
    "SOURCE_HOMOGRAPHY_ATTRS_DIGEST_CANONICALIZATION",
    "SOURCE_HOMOGRAPHY_YAML_DIGEST_CANONICALIZATION",
    "SOURCE_HOMOGRAPHY_EVIDENCE_ATTR",
    "SOURCE_HOMOGRAPHY_EVIDENCE_DIGEST_SUFFIX",
    "PIXEL_TO_MM_DERIVATION",
    "Z_EFF_MM_SOURCE_FIELD",
    "Z_EFF_MM_DERIVATION",
    "SUPPORTED_OUTPUT_TRANSFORM_TOKEN",
    "SELECTED_CALIBRATION_MANIFEST_ATTR",
    "SELECTED_CALIBRATION_MANIFEST_DIGEST_SUFFIX",
    "SelectedCalibrationError",
    "SelectedCalibrationPaths",
    "SelectedCameraSourceEvidence",
    "SelectedDisplaySourceEvidence",
    "SelectedHomographySourceEvidence",
    "CameraCalibrationManifest",
    "DisplaySnapshotManifest",
    "SourceHomographySemantics",
    "SourceArtifactLineage",
    "SelectedCalibrationManifest",
    "SelectedCalibrationSnapshot",
    "selected_calibration_paths",
    "parse_selected_output_geometry",
    "source_display_dataset_sha256",
    "source_arena_config_dataset_sha256",
    "selected_camera_record_digest",
    "camera_group_scalar_attrs_digest",
    "parse_selected_camera_source_evidence",
    "build_selected_camera_source_evidence_from_h5_values",
    "parse_selected_display_source_evidence",
    "build_selected_display_source_evidence_from_h5_values",
    "selected_display_evidence_attrs",
    "load_selected_display_evidence_attrs",
    "parse_selected_homography_source_evidence",
    "build_selected_homography_source_evidence_from_h5_values",
    "source_homography_semantics_from_evidence",
    "source_artifact_from_homography_evidence",
    "selected_homography_evidence_attrs",
    "load_selected_homography_evidence_attrs",
    "selected_calibration_pointer_attrs",
    "camera_calibration_attrs",
    "parse_source_homography_semantics",
    "canonical_source_homography_semantics_json",
    "source_homography_semantics_digest",
    "parse_source_artifact_lineage",
    "parse_selected_calibration_manifest",
    "build_selected_calibration_manifest",
    "canonical_selected_calibration_manifest_json",
    "selected_calibration_manifest_digest",
    "selected_calibration_manifest_attrs",
    "load_selected_calibration_manifest_attrs",
    "stamp_selected_calibration_snapshot",
    "load_selected_calibration_snapshot",
]
