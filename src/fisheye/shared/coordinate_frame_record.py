"""Sealed coordinate-frame authorities for canonical coordinate publication.

Frame metadata is useful only when every scientific claim is tied to the exact
persisted node that proves it.  This module consequently separates pure, strict
record parsing from the sealed bindings accepted by future writers:

* physical millimetres are derived only from an exact source-camera pixel frame
  and exact selected-camera ``pixels_per_mm_camera`` evidence;
* fish-anatomical frames are derived only from an exact canonical source
  descriptor, row identity, typed contract and estimator records, and hashed
  materialized origin/axis/validity arrays; and
* every participating node must belong to one explicitly bound archive/store.

The APIs intentionally do not expose a generic "evidence" wrapper.  A mapping,
path, digest, or numerically plausible scale is never sufficient writer
authority.  Historical coordinate frames that cannot satisfy these typed
bindings belong in audit/migration code and fail closed here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.shared.archive_identity import (
    ArchiveIdentity,
    ArchiveIdentityError,
    require_same_archive,
)
from fisheye.shared.coordinate_descriptor import (
    CANONICAL_COORDINATE_PROFILES,
    COORDINATE_DESCRIPTOR_ATTR,
    CanonicalCoordinateDescriptor,
    CoordinateDescriptorError,
    load_canonical_coordinate_descriptor_attrs,
)
from fisheye.shared.coordinate_identity import (
    INSTANCE_KEY_MODE,
    OBSERVATION_INSTANCE_DOMAIN,
    TRACK_SAMPLE_KEY_MODE,
    TRACK_SAMPLE_DOMAIN,
    BoundRowIdentityContract,
    RowIdentityContractError,
    require_bound_row_identity_contract,
)
from fisheye.shared.coordinate_reference import (
    CoordinateReferenceError,
    canonical_node_path,
)
from fisheye.shared.coordinate_record import (
    BoundCoordinateRecord,
    CoordinateRecordError,
    verify_bound_coordinate_record,
)
from fisheye.shared.pixel_frame_authority import (
    SOURCE_CAMERA_IMAGE_SPACE_ID,
    BoundPixelFrameAuthority,
    PixelFrameAuthorityError,
    require_source_camera_pixel_frame_authority,
)
from fisheye.shared.selected_calibration import (
    SelectedCalibrationError,
    VerifiedSelectedCameraSourceEvidence,
    parse_selected_camera_source_evidence,
    require_verified_selected_camera_source_evidence,
)


PHYSICAL_FRAME_CALIBRATION_SCHEMA_ID = "palette.physical_frame_calibration"
FISH_ANATOMICAL_BODY_FRAME_SCHEMA_ID = "palette.fish_anatomical_body_frame"
SELECTED_CAMERA_FRAME_EVIDENCE_SCHEMA_ID = "palette.selected_camera_frame_evidence"
BODY_FRAME_CONTRACT_SCHEMA_ID = "palette.body_frame_contract"
BODY_FRAME_ESTIMATOR_SCHEMA_ID = "palette.body_frame_estimator"
BODY_ESTIMATOR_SOURCE_BUNDLE_SCHEMA_ID = "palette.body_estimator_source_bundle"
BODY_ESTIMATOR_SOURCE_MANIFEST_SCHEMA_ID = "palette.body_estimator_source_manifest"

COORDINATE_FRAME_RECORD_SCHEMA_VERSION = 1
COORDINATE_FRAME_RECORD_CANONICALIZATION = "canonical_json_sort_keys_v1"
ARRAY_PAYLOAD_CANONICALIZATION = "numpy_dtype_shape_c_order_bytes_v1"

PHYSICAL_FRAME_CALIBRATION_KIND = "physical_frame_calibration"
FISH_ANATOMICAL_BODY_FRAME_KIND = "fish_anatomical_body_frame"
COORDINATE_FRAME_RECORD_KINDS = frozenset(
    {PHYSICAL_FRAME_CALIBRATION_KIND, FISH_ANATOMICAL_BODY_FRAME_KIND}
)

PHYSICAL_FRAME_CALIBRATION_ATTR = "physical_frame_calibration"
FISH_ANATOMICAL_BODY_FRAME_ATTR = "fish_anatomical_body_frame"
SELECTED_CAMERA_FRAME_EVIDENCE_ATTR = "selected_camera_frame_evidence"
BODY_FRAME_CONTRACT_ATTR = "body_frame_contract"
BODY_FRAME_ESTIMATOR_ATTR = "body_frame_estimator"
BODY_ESTIMATOR_SOURCE_MANIFEST_ATTR = "body_estimator_source_manifest"
FRAME_RECORD_DIGEST_SUFFIX = "_sha256"
FRAME_RECORD_SELECTOR = "record"

REFERENCE_EXTENT_FINITE = "finite"
REFERENCE_EXTENT_UNBOUNDED = "unbounded"
REFERENCE_EXTENT_NOT_APPLICABLE = "not_applicable"
REFERENCE_EXTENT_MODES = frozenset(
    {
        REFERENCE_EXTENT_FINITE,
        REFERENCE_EXTENT_UNBOUNDED,
        REFERENCE_EXTENT_NOT_APPLICABLE,
    }
)

PIXELS_PER_MM = "pixels_per_mm"
MM_PER_PIXEL = "mm_per_pixel"
CAMERA_SCALE_QUANTITY = "pixels_per_mm_camera"
PIXEL_SCALE_RECIPROCAL_DERIVATION = (
    "exact_binary64_reciprocal_of_selected_pixels_per_mm_camera_v1"
)

SOURCE_CAMERA_PROFILE_ID = "source_camera_image_px.top_left_y_down.v1"
PHYSICAL_SOURCE_CAMERA_PROFILE_ID = "physical_mm.source_camera_y_down.v1"
PHYSICAL_FRAME_COMPATIBLE_PROFILE_IDS = (PHYSICAL_SOURCE_CAMERA_PROFILE_ID,)
SUPPORTED_BODY_SOURCE_PROFILE_IDS = frozenset(
    {SOURCE_CAMERA_PROFILE_ID, PHYSICAL_SOURCE_CAMERA_PROFILE_ID}
)
SOURCE_CAMERA_SPACE_ID = SOURCE_CAMERA_IMAGE_SPACE_ID
PHYSICAL_SPACE_ID = "physical_mm"

PHYSICAL_ORIGIN = "physical_frame_origin"
PHYSICAL_SOURCE_ORIGIN_RELATION = "coincident_with_source_camera_top_left"
PHYSICAL_POSITIVE_X = "right"
PHYSICAL_POSITIVE_Y = "down"

BODY_ORIGINS = frozenset(
    {
        "estimator_defined_anatomical_origin",
        "eye_pair_midpoint",
        "swim_bladder",
        "body_centroid",
        "spline_arclength_zero",
    }
)
BODY_ESTIMATOR_METHODS = frozenset(
    {
        "keypoint_head_axis",
        "mask_component_axis",
        "body_spline_with_anchor_polarity",
    }
)
BODY_ESTIMATOR_CONFIGURATION_SCHEMAS = {
    "keypoint_head_axis": (
        "palette.keypoint_head_axis_parameters",
        frozenset({"eye_left", "eye_right", "posterior_anchor"}),
    ),
    "mask_component_axis": (
        "palette.mask_component_axis_parameters",
        frozenset({"eye_left", "eye_right", "posterior_anchor"}),
    ),
    "body_spline_with_anchor_polarity": (
        "palette.body_spline_anchor_parameters",
        frozenset({"eye_left", "eye_right", "posterior_anchor"}),
    ),
}
BODY_ESTIMATOR_REQUIRED_ANCHORS = {
    "eye_left": "eye_left",
    "eye_right": "eye_right",
    "posterior_anchor": "swim_bladder",
}
BODY_ESTIMATOR_FORMULAS = {
    "keypoint_head_axis": "eye_midpoint_and_posterior_anchor_v1",
    "mask_component_axis": "eye_component_midpoint_and_posterior_component_v1",
    "body_spline_with_anchor_polarity": "oriented_endpoint_chord_with_eye_side_v1",
}
BODY_ANGLE_CONVENTION = "math_ccw_degrees_after_y_flip"
BODY_POSITIVE_X = "anterior"
BODY_POSITIVE_Y = "anatomical_left"
BODY_FORWARD_AXIS_DIRECTION = "posterior_to_anterior"
BODY_LEFT_AXIS_DIRECTION = "perpendicular_toward_anatomical_left"
BODY_AXIS_UNITS = "unitless"
BODY_INVALID_ROW_ENCODING = "all_geometry_nan_when_axis_valid_false_v1"
BODY_AXIS_NORMAL_TOLERANCE_FLOAT32 = 5e-5
BODY_AXIS_NORMAL_TOLERANCE_FLOAT64 = 1e-10
BODY_ROW_IDENTITY_PROFILES = frozenset(
    {
        (OBSERVATION_INSTANCE_DOMAIN, INSTANCE_KEY_MODE),
        (TRACK_SAMPLE_DOMAIN, TRACK_SAMPLE_KEY_MODE),
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.:+-]+$")
_ATTR_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]*$")
_FRAME_ID_RE = re.compile(r"^[a-z][a-z0-9_.:+-]*$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]*$")
_SCHEMA_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:+-]*$")

_SELECTED_CAMERA_SEAL = object()
_BODY_CONTRACT_SEAL = object()
_BODY_ESTIMATOR_SEAL = object()
_BODY_SOURCE_SEAL = object()
_BODY_ESTIMATOR_SOURCE_SEAL = object()
_BODY_GEOMETRY_SEAL = object()
_FRAME_SEAL = object()


@dataclass(frozen=True)
class CoordinateFrameRecordIssue:
    code: str
    path: str
    message: str


class CoordinateFrameRecordError(ValueError):
    """One or more fail-closed frame authority errors."""

    def __init__(self, issues: Sequence[CoordinateFrameRecordIssue]):
        normalized = tuple(issues) or (
            CoordinateFrameRecordIssue(
                "frame_record_invalid", "$", "Coordinate-frame record is invalid."
            ),
        )
        self.issues = normalized
        super().__init__(
            "; ".join(
                f"{issue.code} at {issue.path}: {issue.message}" for issue in normalized
            )
        )


def _fail(code: str, path: str, message: str) -> None:
    raise CoordinateFrameRecordError((CoordinateFrameRecordIssue(code, path, message),))


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _raw_canonical_equal(raw: Any, canonical: Any) -> bool:
    if type(raw) is not type(canonical):
        return False
    if isinstance(canonical, Mapping):
        return set(raw) == set(canonical) and all(
            _raw_canonical_equal(raw[name], canonical[name]) for name in canonical
        )
    if isinstance(canonical, (list, tuple)):
        return len(raw) == len(canonical) and all(
            _raw_canonical_equal(left, right)
            for left, right in zip(raw, canonical, strict=True)
        )
    if isinstance(canonical, np.ndarray):
        return bool(np.array_equal(raw, canonical, equal_nan=True))
    result = raw == canonical
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if isinstance(value, (bytes, bytearray)):
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            _fail("json_invalid", path, f"JSON bytes are not UTF-8: {exc}.")
    if isinstance(value, str):
        try:
            value = json.loads(value, object_pairs_hook=_reject_duplicate_pairs)
        except (json.JSONDecodeError, ValueError) as exc:
            _fail("json_invalid", path, f"JSON is invalid: {exc}.")
    if not isinstance(value, Mapping):
        _fail("mapping_required", path, "A mapping or JSON object is required.")
    return value


def _exact_fields(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], *, path: str
) -> None:
    issues = [
        CoordinateFrameRecordIssue(
            "missing_field", f"{path}.{name}", "Required field is missing."
        )
        for name in sorted(set(expected) - set(value))
    ]
    issues.extend(
        CoordinateFrameRecordIssue(
            "unknown_field", f"{path}.{name}", "Field is not part of this schema."
        )
        for name in sorted(set(value) - set(expected))
    )
    if issues:
        raise CoordinateFrameRecordError(issues)


def _text(value: Any, *, path: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail("text_invalid", path, "Canonical non-empty text is required.")
    if pattern is not None and pattern.fullmatch(value) is None:
        _fail("text_invalid", path, f"Value {value!r} is not canonical.")
    return value


def _digest(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail("digest_invalid", path, "A lowercase SHA-256 digest is required.")
    return value


def _exact_int(value: Any, *, path: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        _fail("integer_invalid", path, "An exact integer is required.")
    result = int(value)
    if result < minimum:
        _fail("integer_invalid", path, f"Value must be at least {minimum}.")
    return result


def _positive_float(value: Any, *, path: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        _fail("number_invalid", path, "A numeric value is required.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        _fail("number_invalid", path, "Value must be positive and finite.")
    return result


def _canonical_record_ref(value: Any, *, path: str) -> str:
    text = _text(value, path=path)
    if text.count("@") > 1:
        _fail("record_ref_invalid", path, "Record ref has multiple attr selectors.")
    node_path, separator, attr_name = text.partition("@")
    if (
        not node_path.startswith("/")
        or node_path == "/"
        or node_path.endswith("/")
        or "//" in node_path
    ):
        _fail(
            "record_ref_invalid",
            path,
            "An absolute canonical archive path is required.",
        )
    if any(
        segment in {"", ".", ".."} or _PATH_SEGMENT_RE.fullmatch(segment) is None
        for segment in node_path[1:].split("/")
    ):
        _fail("record_ref_invalid", path, "Record ref path is noncanonical.")
    if separator and _ATTR_NAME_RE.fullmatch(attr_name) is None:
        _fail("record_ref_invalid", path, "Record attr selector is noncanonical.")
    return text


@dataclass(frozen=True)
class DigestBoundFrameRecordRef:
    record_ref: str
    record_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "record_ref": self.record_ref,
            "record_sha256": self.record_sha256,
        }


def _parse_record_ref(value: Any, *, path: str) -> DigestBoundFrameRecordRef:
    payload = _mapping(value, path=path)
    _exact_fields(payload, {"record_ref", "record_sha256"}, path=path)
    return DigestBoundFrameRecordRef(
        _canonical_record_ref(payload["record_ref"], path=f"{path}.record_ref"),
        _digest(payload["record_sha256"], path=f"{path}.record_sha256"),
    )


def _node_path(node: Any, *, allow_root: bool = False) -> str:
    value = getattr(node, "path", None)
    if allow_root and value in {"", "/"}:
        return ""
    try:
        return canonical_node_path(node)
    except CoordinateReferenceError as exc:
        _fail("node_path_invalid", "node.path", str(exc))
    raise AssertionError("unreachable")


def _node_record_ref(node: Any, attr_name: str) -> str:
    name = _text(attr_name, path="attr_name", pattern=_ATTR_NAME_RE)
    return f"/{_node_path(node)}@{name}"


def _archive_for_nodes(*nodes: Any) -> ArchiveIdentity:
    try:
        return require_same_archive(*nodes)
    except ArchiveIdentityError as exc:
        _fail("archive_mismatch", "$", str(exc))
    raise AssertionError("unreachable")


def _require_archive_identity(
    expected: ArchiveIdentity, *nodes: Any
) -> ArchiveIdentity:
    current = _archive_for_nodes(*nodes)
    if current != expected:
        _fail("archive_mismatch", "$", "Evidence moved to a different archive/store.")
    return current


def _require_row_identity(
    value: Any,
    *,
    body_frame: bool = False,
) -> BoundRowIdentityContract:
    try:
        bound = require_bound_row_identity_contract(value)
    except RowIdentityContractError as exc:
        _fail("row_identity_unverified", "row_identity", str(exc))
    if body_frame and (
        bound.contract.domain,
        bound.contract.mode,
    ) not in BODY_ROW_IDENTITY_PROFILES:
        _fail(
            "body_row_identity_unsupported",
            "row_identity",
            "Fish-anatomical body frames accept only canonical "
            "observation_instance/instance_key or "
            "track_sample/track_sample_key identity; stimulus_state is not a "
            "biological observation or track-sample identity.",
        )
    return bound


def _attrs(node: Any, *, path: str) -> Any:
    attrs = getattr(node, "attrs", None)
    if (
        not isinstance(attrs, Mapping)
        or not callable(getattr(attrs, "update", None))
        or not callable(getattr(attrs, "__setitem__", None))
        or not callable(getattr(attrs, "__delitem__", None))
    ):
        _fail("attrs_unavailable", path, "Node must expose a mutable attrs mapping.")
    return attrs


def _restore_attrs(attrs: Any, snapshot: Mapping[str, Any]) -> None:
    for name in list(attrs):
        if name not in snapshot:
            del attrs[name]
    for name, value in snapshot.items():
        attrs[name] = copy.deepcopy(value)
    if not _raw_canonical_equal(dict(attrs), dict(snapshot)):
        raise RuntimeError("restored attrs differ from their pre-call snapshot")


def _preflight_attrs_transaction(attrs: Any, *, path: str) -> None:
    """Reject untrusted mutation behavior without probing the live mapping.

    A write/delete probe can itself become an unrecoverable partial mutation
    when a hostile mapping fails during cleanup.  Canonical publication accepts
    only the exact built-in mapping used by deterministic fakes and the exact
    Zarr ``Attributes`` implementation used by persisted archives.  Capability
    checks are read-only and occur before the first scientific attr write.
    """

    trusted = type(attrs) is dict
    if not trusted:
        try:
            from zarr.core.attributes import Attributes as ZarrAttributes
        except ImportError:  # pragma: no cover - Palette depends on zarr
            ZarrAttributes = None  # type: ignore[assignment,misc]
        trusted = ZarrAttributes is not None and type(attrs) is ZarrAttributes
    if not trusted:
        _fail(
            "stamp_preflight_failed",
            path,
            "Attrs mapping type is not an exact trusted dict or Zarr Attributes implementation; no write was attempted.",
        )
    for operation in ("update", "__setitem__", "__delitem__"):
        if not callable(getattr(attrs, operation, None)):
            _fail(
                "stamp_preflight_failed",
                path,
                f"Attrs mapping lacks required {operation} mutation support; no write was attempted.",
            )


def _transactional_stamp(
    node: Any,
    *,
    attr_name: str,
    payload: Mapping[str, Any],
    reload_and_verify: Any,
) -> Any:
    attrs = _attrs(node, path=f"/{_node_path(node)}")
    _preflight_attrs_transaction(attrs, path=f"/{_node_path(node)}")
    snapshot = copy.deepcopy(dict(attrs))
    expected = copy.deepcopy(snapshot)
    expected.update(
        {
            attr_name: copy.deepcopy(dict(payload)),
            f"{attr_name}{FRAME_RECORD_DIGEST_SUFFIX}": _mapping_sha256(payload),
        }
    )
    try:
        attrs.update(copy.deepcopy(expected))
        if not _raw_canonical_equal(dict(attrs), expected):
            _fail(
                "stamp_reload_mismatch",
                f"/{_node_path(node)}",
                "Reloaded attrs differ from the exact intended frame-record write.",
            )
        result = reload_and_verify()
        if not _raw_canonical_equal(dict(attrs), expected):
            _fail(
                "stamp_reload_mismatch",
                f"/{_node_path(node)}",
                "Attrs changed during frame-record verification.",
            )
        return result
    except Exception as exc:
        try:
            _restore_attrs(attrs, snapshot)
        except Exception as rollback_exc:  # pragma: no cover - hostile attrs mapping
            _fail(
                "stamp_rollback_failed",
                f"/{_node_path(node)}",
                f"Stamp failed ({exc}); rollback also failed ({rollback_exc}).",
            )
        if isinstance(exc, CoordinateFrameRecordError):
            raise
        _fail(
            "stamp_failed",
            f"/{_node_path(node)}",
            f"Stamp failed and attrs were restored: {exc}.",
        )
    raise AssertionError("unreachable")


def _load_record_attrs(
    node: Any,
    *,
    attr_name: str,
    parser: Any,
) -> tuple[str, Any, str]:
    attrs = _attrs(node, path=f"/{_node_path(node)}")
    digest_name = f"{attr_name}{FRAME_RECORD_DIGEST_SUFFIX}"
    if attr_name not in attrs or digest_name not in attrs:
        _fail(
            "record_attr_missing",
            f"/{_node_path(node)}",
            "Record and digest attrs are required.",
        )
    raw_record = attrs[attr_name]
    parsed = parser(raw_record)
    if not _raw_canonical_equal(raw_record, parsed.to_dict()):
        _fail(
            "record_noncanonical",
            f"/{_node_path(node)}@{attr_name}",
            "Persisted frame record is not exact canonical JSON form.",
        )
    digest = _mapping_sha256(parsed.to_dict())
    if _digest(attrs[digest_name], path=f"/{_node_path(node)}@{digest_name}") != digest:
        _fail(
            "record_digest_mismatch",
            f"/{_node_path(node)}@{attr_name}",
            "Stored digest does not match canonical record content.",
        )
    return _node_record_ref(node, attr_name), parsed, digest


@dataclass(frozen=True)
class _ArrayPayloadSnapshot:
    """One immutable array read plus its exact declared metadata and digest."""

    values: np.ndarray = field(repr=False, compare=False)
    dtype: str
    shape: tuple[int, ...]
    content_sha256: str


def _array_content_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    header = {
        "canonicalization": ARRAY_PAYLOAD_CANONICALIZATION,
        "dtype": np.lib.format.dtype_to_descr(values.dtype),
        "shape": [int(item) for item in values.shape],
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json(header).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _declared_array_metadata(node: Any) -> tuple[str, tuple[int, ...]]:
    path = f"/{_node_path(node)}"
    try:
        dtype = np.dtype(getattr(node, "dtype")).str
    except (AttributeError, TypeError) as exc:
        _fail("array_dtype_invalid", f"{path}.dtype", str(exc))
    raw_shape = getattr(node, "shape", None)
    if not isinstance(raw_shape, (tuple, list)):
        _fail("array_shape_invalid", f"{path}.shape", "Exact array shape is required.")
    shape = tuple(
        _exact_int(item, path=f"{path}.shape[{index}]")
        for index, item in enumerate(raw_shape)
    )
    return dtype, shape


def _read_array_snapshot(node: Any) -> _ArrayPayloadSnapshot:
    """Read once into owned memory and reject concurrent metadata changes."""

    path = f"/{_node_path(node)}"
    before_dtype, before_shape = _declared_array_metadata(node)
    try:
        values = np.array(node[:], copy=True, order="C")
    except Exception as exc:
        _fail("array_unreadable", path, str(exc))
    after_dtype, after_shape = _declared_array_metadata(node)
    if (
        before_dtype != after_dtype
        or before_shape != after_shape
        or values.dtype.str != before_dtype
        or values.shape != before_shape
    ):
        _fail(
            "array_metadata_value_mismatch",
            path,
            "Read array dtype/shape differs from stable node metadata.",
        )
    if values.dtype.hasobject:
        _fail(
            "array_dtype_invalid",
            f"{path}.dtype",
            "Object-reference arrays do not have deterministic payload bytes.",
        )
    values.setflags(write=False)
    return _ArrayPayloadSnapshot(
        values=values,
        dtype=before_dtype,
        shape=before_shape,
        content_sha256=_array_content_sha256(values),
    )


def _recheck_array_snapshot(node: Any, snapshot: _ArrayPayloadSnapshot) -> None:
    """Close the validation/hash TOCTOU window with one complete second read."""

    current = _read_array_snapshot(node)
    if (
        current.dtype != snapshot.dtype
        or current.shape != snapshot.shape
        or current.content_sha256 != snapshot.content_sha256
    ):
        _fail(
            "array_changed_during_binding",
            f"/{_node_path(node)}",
            "Array payload or metadata changed while its authority was being bound.",
        )


def array_payload_sha256(node: Any) -> str:
    """Hash a stable exact materialized array payload."""

    snapshot = _read_array_snapshot(node)
    _recheck_array_snapshot(node, snapshot)
    return snapshot.content_sha256


def array_values_sha256(values: Any) -> str:
    """Hash one owned/in-memory array with the canonical payload grammar."""

    array = np.asarray(values)
    if array.dtype.hasobject:
        _fail(
            "array_dtype_invalid",
            "values.dtype",
            "Object-reference arrays do not have deterministic payload bytes.",
        )
    return _array_content_sha256(array)


@dataclass(frozen=True)
class SelectedCameraFrameEvidenceRecord:
    source_camera: Mapping[str, Any]
    source_camera_sha256: str
    camera_id: str
    native_width_px: int
    native_height_px: int
    pixels_per_mm_camera_selector: str
    pixels_per_mm_camera: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": SELECTED_CAMERA_FRAME_EVIDENCE_SCHEMA_ID,
            "schema_version": COORDINATE_FRAME_RECORD_SCHEMA_VERSION,
            "source_camera": copy.deepcopy(dict(self.source_camera)),
            "source_camera_sha256": self.source_camera_sha256,
            "camera_id": self.camera_id,
            "native_width_px": self.native_width_px,
            "native_height_px": self.native_height_px,
            "pixels_per_mm_camera_selector": self.pixels_per_mm_camera_selector,
            "pixels_per_mm_camera": self.pixels_per_mm_camera,
        }

    def digest(self) -> str:
        return _mapping_sha256(self.to_dict())


def parse_selected_camera_frame_evidence_record(
    value: Any,
) -> SelectedCameraFrameEvidenceRecord:
    """Strictly parse persisted selected-camera physical evidence."""

    if isinstance(value, SelectedCameraFrameEvidenceRecord):
        value = value.to_dict()
    payload = _mapping(value, path="$")
    _exact_fields(
        payload,
        {
            "schema_id",
            "schema_version",
            "source_camera",
            "source_camera_sha256",
            "camera_id",
            "native_width_px",
            "native_height_px",
            "pixels_per_mm_camera_selector",
            "pixels_per_mm_camera",
        },
        path="$",
    )
    if payload["schema_id"] != SELECTED_CAMERA_FRAME_EVIDENCE_SCHEMA_ID:
        _fail(
            "schema_invalid",
            "$.schema_id",
            "Unsupported selected-camera evidence schema.",
        )
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        _fail(
            "schema_invalid",
            "$.schema_version",
            "Unsupported selected-camera evidence version.",
        )
    try:
        selected = parse_selected_camera_source_evidence(payload["source_camera"])
    except SelectedCalibrationError as exc:
        _fail("selected_camera_source_invalid", "$.source_camera", str(exc))
    source_payload = selected.to_dict()
    source_digest = _mapping_sha256(source_payload)
    if (
        _digest(payload["source_camera_sha256"], path="$.source_camera_sha256")
        != source_digest
    ):
        _fail(
            "selected_camera_source_digest_mismatch",
            "$.source_camera_sha256",
            "Digest does not match exact selected-camera source evidence.",
        )
    camera_id = _text(payload["camera_id"], path="$.camera_id", pattern=_IDENTIFIER_RE)
    width = _exact_int(payload["native_width_px"], path="$.native_width_px", minimum=1)
    height = _exact_int(
        payload["native_height_px"], path="$.native_height_px", minimum=1
    )
    selector = payload["pixels_per_mm_camera_selector"]
    if selector != "/selected_camera_record/pixels_per_mm_camera":
        _fail(
            "selected_camera_scale_selector_invalid",
            "$.pixels_per_mm_camera_selector",
            "Selected-camera physical v1 requires the exact camera ppm field, never projector ppm.",
        )
    ppm = _positive_float(
        payload["pixels_per_mm_camera"], path="$.pixels_per_mm_camera"
    )
    if camera_id != selected.active_camera_id:
        _fail(
            "selected_camera_identity_mismatch",
            "$.camera_id",
            "Camera ID does not match exact selected-camera source evidence.",
        )
    if width != selected.native_width_px or height != selected.native_height_px:
        _fail(
            "selected_camera_extent_mismatch",
            "$",
            "Native dimensions do not match exact selected-camera source evidence.",
        )
    if selected.pixels_per_mm_camera is None or ppm != selected.pixels_per_mm_camera:
        _fail(
            "selected_camera_scale_mismatch",
            "$.pixels_per_mm_camera",
            "Scale must exactly equal selected pixels_per_mm_camera evidence.",
        )
    return SelectedCameraFrameEvidenceRecord(
        source_camera=source_payload,
        source_camera_sha256=source_digest,
        camera_id=camera_id,
        native_width_px=width,
        native_height_px=height,
        pixels_per_mm_camera_selector=selector,
        pixels_per_mm_camera=ppm,
    )


@dataclass(frozen=True, init=False)
class BoundSelectedCameraFrameEvidence:
    record_ref: str
    record_sha256: str
    record: SelectedCameraFrameEvidenceRecord
    archive_identity: ArchiveIdentity = field(repr=False, compare=False)
    _node: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        record_ref: str,
        record_sha256: str,
        record: SelectedCameraFrameEvidenceRecord,
        archive_identity: ArchiveIdentity,
        node: Any,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _SELECTED_CAMERA_SEAL:
            _fail(
                "selected_camera_evidence_unsealed",
                "$",
                "Selected-camera evidence must come from its typed stamp/loader.",
            )
        object.__setattr__(self, "record_ref", record_ref)
        object.__setattr__(self, "record_sha256", record_sha256)
        object.__setattr__(self, "record", record)
        object.__setattr__(self, "archive_identity", archive_identity)
        object.__setattr__(self, "_node", node)
        object.__setattr__(self, "_seal", _verification_seal)

    @property
    def ref(self) -> DigestBoundFrameRecordRef:
        return DigestBoundFrameRecordRef(self.record_ref, self.record_sha256)


def _build_selected_camera_record(
    source: VerifiedSelectedCameraSourceEvidence,
) -> SelectedCameraFrameEvidenceRecord:
    try:
        parsed = require_verified_selected_camera_source_evidence(source)
    except SelectedCalibrationError as exc:
        _fail("selected_camera_source_unverified", "source_camera", str(exc))
    if parsed.pixels_per_mm_camera is None:
        _fail(
            "selected_camera_scale_missing",
            "source_camera",
            "Selected camera lacks pixels_per_mm_camera.",
        )
    payload = parsed.to_dict()
    return parse_selected_camera_frame_evidence_record(
        {
            "schema_id": SELECTED_CAMERA_FRAME_EVIDENCE_SCHEMA_ID,
            "schema_version": 1,
            "source_camera": payload,
            "source_camera_sha256": _mapping_sha256(payload),
            "camera_id": parsed.active_camera_id,
            "native_width_px": parsed.native_width_px,
            "native_height_px": parsed.native_height_px,
            "pixels_per_mm_camera_selector": "/selected_camera_record/pixels_per_mm_camera",
            "pixels_per_mm_camera": parsed.pixels_per_mm_camera,
        }
    )


def load_bound_selected_camera_frame_evidence(
    node: Any,
    *,
    expected_record_ref: str,
    expected_record_sha256: str,
    expected_camera_id: str,
) -> BoundSelectedCameraFrameEvidence:
    """Bind an exact persisted selected-camera evidence record."""

    identity = _archive_for_nodes(node)
    record_ref, record, digest = _load_record_attrs(
        node,
        attr_name=SELECTED_CAMERA_FRAME_EVIDENCE_ATTR,
        parser=parse_selected_camera_frame_evidence_record,
    )
    if record_ref != _canonical_record_ref(
        expected_record_ref, path="expected_record_ref"
    ):
        _fail(
            "record_path_mismatch",
            "expected_record_ref",
            "Selected-camera evidence path differs from expectation.",
        )
    if digest != _digest(expected_record_sha256, path="expected_record_sha256"):
        _fail(
            "record_digest_mismatch",
            "expected_record_sha256",
            "Selected-camera evidence digest differs from expectation.",
        )
    if record.camera_id != _text(
        expected_camera_id, path="expected_camera_id", pattern=_IDENTIFIER_RE
    ):
        _fail(
            "selected_camera_identity_mismatch",
            "expected_camera_id",
            "Selected-camera evidence names a different camera.",
        )
    return BoundSelectedCameraFrameEvidence(
        record_ref=record_ref,
        record_sha256=digest,
        record=record,
        archive_identity=identity,
        node=node,
        _verification_seal=_SELECTED_CAMERA_SEAL,
    )


def verify_bound_selected_camera_frame_evidence(
    value: BoundSelectedCameraFrameEvidence,
) -> BoundSelectedCameraFrameEvidence:
    if (
        type(value) is not BoundSelectedCameraFrameEvidence
        or value._seal is not _SELECTED_CAMERA_SEAL
    ):
        _fail(
            "selected_camera_evidence_unsealed",
            "$",
            "A sealed selected-camera evidence binding is required.",
        )
    current = load_bound_selected_camera_frame_evidence(
        value._node,
        expected_record_ref=value.record_ref,
        expected_record_sha256=value.record_sha256,
        expected_camera_id=value.record.camera_id,
    )
    if current.record != value.record:
        _fail(
            "selected_camera_evidence_stale",
            value.record_ref,
            "Selected-camera evidence changed after binding.",
        )
    _require_archive_identity(value.archive_identity, value._node)
    return value


def stamp_selected_camera_frame_evidence(
    node: Any,
    *,
    source_camera: VerifiedSelectedCameraSourceEvidence,
) -> BoundSelectedCameraFrameEvidence:
    """Persist and bind evidence built from exact named H5 values."""

    _archive_for_nodes(node)
    record = _build_selected_camera_record(source_camera)
    expected_ref = _node_record_ref(node, SELECTED_CAMERA_FRAME_EVIDENCE_ATTR)
    digest = record.digest()
    return _transactional_stamp(
        node,
        attr_name=SELECTED_CAMERA_FRAME_EVIDENCE_ATTR,
        payload=record.to_dict(),
        reload_and_verify=lambda: load_bound_selected_camera_frame_evidence(
            node,
            expected_record_ref=expected_ref,
            expected_record_sha256=digest,
            expected_camera_id=record.camera_id,
        ),
    )


@dataclass(frozen=True)
class FrameExtent:
    mode: str
    width: float | None
    height: float | None
    units: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "width": self.width,
            "height": self.height,
            "units": self.units,
        }


def _parse_frame_extent(value: Any, *, path: str) -> FrameExtent:
    payload = _mapping(value, path=path)
    _exact_fields(payload, {"mode", "width", "height", "units"}, path=path)
    mode = payload["mode"]
    if mode not in REFERENCE_EXTENT_MODES:
        _fail("extent_mode_invalid", f"{path}.mode", "Unsupported extent mode.")
    width = payload["width"]
    height = payload["height"]
    units = payload["units"]
    if mode == REFERENCE_EXTENT_FINITE:
        width = _positive_float(width, path=f"{path}.width")
        height = _positive_float(height, path=f"{path}.height")
        if units != "mm":
            _fail(
                "extent_units_invalid",
                f"{path}.units",
                "Finite physical extent uses mm.",
            )
    elif mode == REFERENCE_EXTENT_UNBOUNDED:
        if width is not None or height is not None or units != "mm":
            _fail(
                "extent_mode_mismatch",
                path,
                "Unbounded extent requires null dimensions and mm units.",
            )
    elif width is not None or height is not None or units != "not_applicable":
        _fail(
            "extent_mode_mismatch",
            path,
            "Not-applicable extent requires null dimensions and not_applicable units.",
        )
    return FrameExtent(mode=mode, width=width, height=height, units=units)


@dataclass(frozen=True)
class PhysicalFrameCalibrationRecord:
    frame_id: str
    source_camera_pixels: DigestBoundFrameRecordRef
    selected_camera_evidence: DigestBoundFrameRecordRef
    camera_id: str
    pixels_per_mm_camera: float
    mm_per_pixel: float
    physical_extent: FrameExtent
    compatible_profile_ids: tuple[str, ...]

    @property
    def kind(self) -> str:
        return PHYSICAL_FRAME_CALIBRATION_KIND

    @property
    def origin(self) -> str:
        return PHYSICAL_ORIGIN

    @property
    def positive_x(self) -> str:
        return PHYSICAL_POSITIVE_X

    @property
    def positive_y(self) -> str:
        return PHYSICAL_POSITIVE_Y

    @property
    def source_origin_relation(self) -> str:
        return PHYSICAL_SOURCE_ORIGIN_RELATION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": PHYSICAL_FRAME_CALIBRATION_SCHEMA_ID,
            "schema_version": 1,
            "kind": self.kind,
            "frame_id": self.frame_id,
            "coordinate_units": "mm",
            "origin": PHYSICAL_ORIGIN,
            "source_origin_relation": self.source_origin_relation,
            "positive_directions": {
                "x": self.positive_x,
                "y": self.positive_y,
            },
            "compatible_profile_ids": list(self.compatible_profile_ids),
            "source_space_id": SOURCE_CAMERA_SPACE_ID,
            "source_camera_pixels": self.source_camera_pixels.to_dict(),
            "selected_camera_evidence": self.selected_camera_evidence.to_dict(),
            "camera_id": self.camera_id,
            "scale": {
                "quantity": CAMERA_SCALE_QUANTITY,
                "pixels_per_mm_camera": self.pixels_per_mm_camera,
                "mm_per_pixel": self.mm_per_pixel,
                "derivation": PIXEL_SCALE_RECIPROCAL_DERIVATION,
            },
            "physical_extent": self.physical_extent.to_dict(),
        }

    def canonical_json(self) -> str:
        return physical_frame_calibration_record_json(self)

    def digest(self) -> str:
        return physical_frame_calibration_record_sha256(self)


def parse_physical_frame_calibration_record(
    value: Any,
) -> PhysicalFrameCalibrationRecord:
    """Parse the deliberately narrow source-camera-to-mm v1 record."""

    if isinstance(value, PhysicalFrameCalibrationRecord):
        value = value.to_dict()
    payload = _mapping(value, path="$")
    _exact_fields(
        payload,
        {
            "schema_id",
            "schema_version",
            "kind",
            "frame_id",
            "coordinate_units",
            "origin",
            "source_origin_relation",
            "positive_directions",
            "compatible_profile_ids",
            "source_space_id",
            "source_camera_pixels",
            "selected_camera_evidence",
            "camera_id",
            "scale",
            "physical_extent",
        },
        path="$",
    )
    if (
        payload["schema_id"] != PHYSICAL_FRAME_CALIBRATION_SCHEMA_ID
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
    ):
        _fail("schema_invalid", "$", "Unsupported physical-frame schema.")
    if payload["kind"] != PHYSICAL_FRAME_CALIBRATION_KIND:
        _fail("kind_invalid", "$.kind", "Record is not a physical frame.")
    if (
        payload["coordinate_units"] != "mm"
        or payload["source_space_id"] != SOURCE_CAMERA_SPACE_ID
    ):
        _fail(
            "source_space_unsupported",
            "$.source_space_id",
            "Physical v1 supports only source_camera_image_px to mm.",
        )
    if payload["origin"] != PHYSICAL_ORIGIN or payload["positive_directions"] != {
        "x": "right",
        "y": "down",
    }:
        _fail(
            "physical_axes_invalid",
            "$",
            "Physical v1 preserves source-camera top-left/+X-right/+Y-down semantics.",
        )
    if payload["source_origin_relation"] != PHYSICAL_SOURCE_ORIGIN_RELATION:
        _fail(
            "physical_source_origin_relation_invalid",
            "$.source_origin_relation",
            "Physical v1 origin must be explicitly coincident with the exact "
            "source-camera top-left origin.",
        )
    compatible_profile_ids = payload["compatible_profile_ids"]
    if (
        not isinstance(compatible_profile_ids, list)
        or tuple(compatible_profile_ids) != PHYSICAL_FRAME_COMPATIBLE_PROFILE_IDS
    ):
        _fail(
            "physical_profile_compatibility_invalid",
            "$.compatible_profile_ids",
            "Physical v1 is compatible only with the explicit source-camera-relative "
            "millimetre profile; arena physical coordinates require their own "
            "direction-labelled transform authority.",
        )
    scale = _mapping(payload["scale"], path="$.scale")
    _exact_fields(
        scale,
        {"quantity", "pixels_per_mm_camera", "mm_per_pixel", "derivation"},
        path="$.scale",
    )
    if scale["quantity"] != CAMERA_SCALE_QUANTITY:
        _fail(
            "scale_quantity_invalid",
            "$.scale.quantity",
            "Physical v1 requires selected camera pixels_per_mm_camera.",
        )
    if scale["derivation"] != PIXEL_SCALE_RECIPROCAL_DERIVATION:
        _fail(
            "scale_derivation_invalid",
            "$.scale.derivation",
            "Unsupported scale derivation.",
        )
    ppm = _positive_float(
        scale["pixels_per_mm_camera"], path="$.scale.pixels_per_mm_camera"
    )
    mm_per_pixel = _positive_float(scale["mm_per_pixel"], path="$.scale.mm_per_pixel")
    if mm_per_pixel != 1.0 / ppm:
        _fail(
            "scale_reciprocal_mismatch",
            "$.scale",
            "mm_per_pixel must be the exact binary64 reciprocal of selected camera ppm.",
        )
    extent = _parse_frame_extent(payload["physical_extent"], path="$.physical_extent")
    return PhysicalFrameCalibrationRecord(
        frame_id=_text(payload["frame_id"], path="$.frame_id", pattern=_FRAME_ID_RE),
        source_camera_pixels=_parse_record_ref(
            payload["source_camera_pixels"], path="$.source_camera_pixels"
        ),
        selected_camera_evidence=_parse_record_ref(
            payload["selected_camera_evidence"], path="$.selected_camera_evidence"
        ),
        camera_id=_text(
            payload["camera_id"], path="$.camera_id", pattern=_IDENTIFIER_RE
        ),
        pixels_per_mm_camera=ppm,
        mm_per_pixel=mm_per_pixel,
        physical_extent=extent,
        compatible_profile_ids=PHYSICAL_FRAME_COMPATIBLE_PROFILE_IDS,
    )


def physical_frame_calibration_record_json(value: Any) -> str:
    return _canonical_json(parse_physical_frame_calibration_record(value).to_dict())


def physical_frame_calibration_record_sha256(value: Any) -> str:
    return hashlib.sha256(
        physical_frame_calibration_record_json(value).encode("utf-8")
    ).hexdigest()


def _physical_record_from_source(
    *,
    frame_id: str,
    source_camera_pixels: BoundPixelFrameAuthority,
    selected_camera_evidence: BoundSelectedCameraFrameEvidence,
    physical_extent_mode: str,
) -> PhysicalFrameCalibrationRecord:
    try:
        source = require_source_camera_pixel_frame_authority(source_camera_pixels)
    except PixelFrameAuthorityError as exc:
        _fail("source_camera_frame_unverified", "source_camera_pixels", str(exc))
    selected_bound = verify_bound_selected_camera_frame_evidence(
        selected_camera_evidence
    )
    _archive_for_nodes(source._authority_node, selected_bound._node)
    selected = selected_bound.record
    camera_id = source.record.lineage["camera_id"]
    if camera_id != selected.camera_id:
        _fail(
            "camera_identity_mismatch",
            "selected_camera_evidence",
            "Acquisition camera and selected calibration name different cameras.",
        )
    if (
        source.endpoint.width != selected.native_width_px
        or source.endpoint.height != selected.native_height_px
    ):
        _fail(
            "camera_native_extent_mismatch",
            "selected_camera_evidence",
            "Acquisition dimensions differ from selected camera native dimensions.",
        )
    ppm = selected.pixels_per_mm_camera
    mm_per_pixel = 1.0 / ppm
    if physical_extent_mode == REFERENCE_EXTENT_FINITE:
        extent = FrameExtent(
            mode=REFERENCE_EXTENT_FINITE,
            width=float(source.endpoint.width) * mm_per_pixel,
            height=float(source.endpoint.height) * mm_per_pixel,
            units="mm",
        )
    elif physical_extent_mode == REFERENCE_EXTENT_UNBOUNDED:
        extent = FrameExtent(REFERENCE_EXTENT_UNBOUNDED, None, None, "mm")
    elif physical_extent_mode == REFERENCE_EXTENT_NOT_APPLICABLE:
        extent = FrameExtent(
            REFERENCE_EXTENT_NOT_APPLICABLE, None, None, "not_applicable"
        )
    else:
        _fail(
            "extent_mode_invalid",
            "physical_extent_mode",
            "Unsupported physical extent mode.",
        )
    return parse_physical_frame_calibration_record(
        {
            "schema_id": PHYSICAL_FRAME_CALIBRATION_SCHEMA_ID,
            "schema_version": 1,
            "kind": PHYSICAL_FRAME_CALIBRATION_KIND,
            "frame_id": frame_id,
            "coordinate_units": "mm",
            "origin": PHYSICAL_ORIGIN,
            "source_origin_relation": PHYSICAL_SOURCE_ORIGIN_RELATION,
            "positive_directions": {"x": "right", "y": "down"},
            "compatible_profile_ids": list(PHYSICAL_FRAME_COMPATIBLE_PROFILE_IDS),
            "source_space_id": SOURCE_CAMERA_SPACE_ID,
            "source_camera_pixels": DigestBoundFrameRecordRef(
                source.record_ref, source.record_sha256
            ).to_dict(),
            "selected_camera_evidence": selected_bound.ref.to_dict(),
            "camera_id": camera_id,
            "scale": {
                "quantity": CAMERA_SCALE_QUANTITY,
                "pixels_per_mm_camera": ppm,
                "mm_per_pixel": mm_per_pixel,
                "derivation": PIXEL_SCALE_RECIPROCAL_DERIVATION,
            },
            "physical_extent": extent.to_dict(),
        }
    )


def build_physical_frame_calibration_record(
    *,
    frame_id: str,
    source_camera_pixels: BoundPixelFrameAuthority,
    selected_camera_evidence: BoundSelectedCameraFrameEvidence,
    physical_extent_mode: str = REFERENCE_EXTENT_FINITE,
) -> PhysicalFrameCalibrationRecord:
    """Build physical v1 solely from exact native camera pixels and camera ppm."""

    return _physical_record_from_source(
        frame_id=frame_id,
        source_camera_pixels=source_camera_pixels,
        selected_camera_evidence=selected_camera_evidence,
        physical_extent_mode=physical_extent_mode,
    )


@dataclass(frozen=True, init=False)
class BoundPhysicalFrameCalibration:
    kind: str
    record_ref: str
    record_sha256: str
    selector: str
    reference_width: float | None
    reference_height: float | None
    reference_units: str
    extent_mode: str
    coordinate_units: str
    origin: str
    positive_x: str
    positive_y: str
    source_origin_relation: str
    compatible_profile_ids: tuple[str, ...]
    record: PhysicalFrameCalibrationRecord
    source_camera_pixels: BoundPixelFrameAuthority
    selected_camera_evidence: BoundSelectedCameraFrameEvidence
    archive_identity: ArchiveIdentity = field(repr=False, compare=False)
    _node: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        record_ref: str,
        record_sha256: str,
        record: PhysicalFrameCalibrationRecord,
        source_camera_pixels: BoundPixelFrameAuthority,
        selected_camera_evidence: BoundSelectedCameraFrameEvidence,
        archive_identity: ArchiveIdentity,
        node: Any,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _FRAME_SEAL:
            _fail(
                "frame_unsealed", "$", "Physical frame must come from its stamp/loader."
            )
        extent = record.physical_extent
        values = {
            "kind": PHYSICAL_FRAME_CALIBRATION_KIND,
            "record_ref": record_ref,
            "record_sha256": record_sha256,
            "selector": FRAME_RECORD_SELECTOR,
            "reference_width": extent.width
            if extent.mode == REFERENCE_EXTENT_FINITE
            else None,
            "reference_height": extent.height
            if extent.mode == REFERENCE_EXTENT_FINITE
            else None,
            "reference_units": extent.units,
            "extent_mode": extent.mode,
            "coordinate_units": "mm",
            "origin": record.origin,
            "positive_x": record.positive_x,
            "positive_y": record.positive_y,
            "source_origin_relation": record.source_origin_relation,
            "compatible_profile_ids": record.compatible_profile_ids,
            "record": record,
            "source_camera_pixels": source_camera_pixels,
            "selected_camera_evidence": selected_camera_evidence,
            "archive_identity": archive_identity,
            "_node": node,
            "_seal": _verification_seal,
        }
        for name, item in values.items():
            object.__setattr__(self, name, item)


def _validate_physical_authorities(
    record: PhysicalFrameCalibrationRecord,
    *,
    frame_node: Any,
    source_camera_pixels: BoundPixelFrameAuthority,
    selected_camera_evidence: BoundSelectedCameraFrameEvidence,
) -> tuple[BoundPixelFrameAuthority, BoundSelectedCameraFrameEvidence]:
    try:
        source = require_source_camera_pixel_frame_authority(source_camera_pixels)
    except PixelFrameAuthorityError as exc:
        _fail("source_camera_frame_unverified", "source_camera_pixels", str(exc))
    selected = verify_bound_selected_camera_frame_evidence(
        selected_camera_evidence
    )
    identity = _archive_for_nodes(
        frame_node,
        source._authority_node,
        selected._node,
    )
    if source.archive_identity != identity:
        _fail(
            "archive_mismatch",
            "$",
            "Physical frame authorities do not share one archive/store.",
        )
    frame_path = _node_path(frame_node)
    dependency_paths = {
        _node_path(source._authority_node),
        _node_path(selected._node),
    }
    if frame_path in dependency_paths:
        _fail(
            "dependency_cycle",
            "$",
            "Physical frame cannot overwrite one of its own authorities.",
        )
    if len(dependency_paths) != 2:
        _fail(
            "dependency_alias", "$", "Physical authorities must occupy distinct nodes."
        )
    if (
        record.source_camera_pixels
        != DigestBoundFrameRecordRef(source.record_ref, source.record_sha256)
        or record.selected_camera_evidence != selected.ref
    ):
        _fail(
            "physical_authority_mismatch",
            "$",
            "Physical record does not bind exact camera-pixel/calibration authorities.",
        )
    selected_record = selected.record
    camera_id = source.record.lineage["camera_id"]
    if (
        record.camera_id != camera_id
        or record.camera_id != selected_record.camera_id
    ):
        _fail(
            "camera_identity_mismatch",
            "$.camera_id",
            "Physical frame camera identities disagree.",
        )
    if (
        source.endpoint.width != selected_record.native_width_px
        or source.endpoint.height != selected_record.native_height_px
    ):
        _fail(
            "camera_native_extent_mismatch",
            "$",
            "Acquisition dimensions differ from selected camera native dimensions.",
        )
    if (
        record.pixels_per_mm_camera != selected_record.pixels_per_mm_camera
        or record.mm_per_pixel != 1.0 / selected_record.pixels_per_mm_camera
    ):
        _fail(
            "physical_scale_binding_mismatch",
            "$.scale",
            "Physical scale is not exact selected camera ppm.",
        )
    if record.physical_extent.mode == REFERENCE_EXTENT_FINITE:
        expected_width = float(source.endpoint.width) * record.mm_per_pixel
        expected_height = float(source.endpoint.height) * record.mm_per_pixel
        if (
            record.physical_extent.width != expected_width
            or record.physical_extent.height != expected_height
        ):
            _fail(
                "physical_extent_scale_mismatch",
                "$.physical_extent",
                "Finite physical extent does not equal exact native pixels times mm_per_pixel.",
            )
    return source, selected


def load_bound_physical_frame_calibration(
    node: Any,
    *,
    expected_record_ref: str,
    expected_record_sha256: str,
    expected_camera_id: str,
    source_camera_pixels: BoundPixelFrameAuthority,
    selected_camera_evidence: BoundSelectedCameraFrameEvidence,
) -> BoundPhysicalFrameCalibration:
    record_ref, record, digest = _load_record_attrs(
        node,
        attr_name=PHYSICAL_FRAME_CALIBRATION_ATTR,
        parser=parse_physical_frame_calibration_record,
    )
    if record_ref != _canonical_record_ref(
        expected_record_ref, path="expected_record_ref"
    ):
        _fail(
            "record_path_mismatch",
            "expected_record_ref",
            "Physical frame path differs from expectation.",
        )
    if digest != _digest(expected_record_sha256, path="expected_record_sha256"):
        _fail(
            "record_digest_mismatch",
            "expected_record_sha256",
            "Physical frame digest differs from expectation.",
        )
    if record.camera_id != _text(
        expected_camera_id, path="expected_camera_id", pattern=_IDENTIFIER_RE
    ):
        _fail(
            "camera_identity_mismatch",
            "expected_camera_id",
            "Physical frame uses another camera.",
        )
    source, selected = _validate_physical_authorities(
        record,
        frame_node=node,
        source_camera_pixels=source_camera_pixels,
        selected_camera_evidence=selected_camera_evidence,
    )
    identity = _archive_for_nodes(
        node, source._authority_node, selected._node
    )
    if identity != source.archive_identity:
        _fail(
            "archive_mismatch",
            "$",
            "Physical frame authorities moved between archives.",
        )
    return BoundPhysicalFrameCalibration(
        record_ref=record_ref,
        record_sha256=digest,
        record=record,
        source_camera_pixels=source,
        selected_camera_evidence=selected,
        archive_identity=identity,
        node=node,
        _verification_seal=_FRAME_SEAL,
    )


def stamp_physical_frame_calibration_record(
    node: Any,
    record: Any,
    *,
    expected_record_ref: str,
    source_camera_pixels: BoundPixelFrameAuthority,
    selected_camera_evidence: BoundSelectedCameraFrameEvidence,
) -> BoundPhysicalFrameCalibration:
    """Transactionally write, reload, and fully verify a physical frame."""

    parsed = parse_physical_frame_calibration_record(record)
    source, selected = _validate_physical_authorities(
        parsed,
        frame_node=node,
        source_camera_pixels=source_camera_pixels,
        selected_camera_evidence=selected_camera_evidence,
    )
    actual_ref = _node_record_ref(node, PHYSICAL_FRAME_CALIBRATION_ATTR)
    if actual_ref != _canonical_record_ref(
        expected_record_ref, path="expected_record_ref"
    ):
        _fail(
            "record_path_mismatch",
            "expected_record_ref",
            "Physical frame node differs from requested path.",
        )
    return _transactional_stamp(
        node,
        attr_name=PHYSICAL_FRAME_CALIBRATION_ATTR,
        payload=parsed.to_dict(),
        reload_and_verify=lambda: load_bound_physical_frame_calibration(
            node,
            expected_record_ref=actual_ref,
            expected_record_sha256=parsed.digest(),
            expected_camera_id=parsed.camera_id,
            source_camera_pixels=source,
            selected_camera_evidence=selected,
        ),
    )


@dataclass(frozen=True)
class BodyFrameAxes:
    positive_x: str = BODY_POSITIVE_X
    positive_y: str = BODY_POSITIVE_Y
    forward_axis_direction: str = BODY_FORWARD_AXIS_DIRECTION
    left_axis_direction: str = BODY_LEFT_AXIS_DIRECTION
    axis_units: str = BODY_AXIS_UNITS

    def to_dict(self) -> dict[str, str]:
        return {
            "positive_x": self.positive_x,
            "positive_y": self.positive_y,
            "forward_axis_direction": self.forward_axis_direction,
            "left_axis_direction": self.left_axis_direction,
            "axis_units": self.axis_units,
        }


def _parse_body_axes(value: Any, *, path: str) -> BodyFrameAxes:
    payload = _mapping(value, path=path)
    expected = BodyFrameAxes().to_dict()
    _exact_fields(payload, set(expected), path=path)
    if dict(payload) != expected:
        _fail(
            "body_axes_invalid",
            path,
            "Body axes are fixed to +X anterior and +Y anatomical left.",
        )
    return BodyFrameAxes()


@dataclass(frozen=True)
class BodyFrameContractRecord:
    axes: BodyFrameAxes = BodyFrameAxes()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": BODY_FRAME_CONTRACT_SCHEMA_ID,
            "schema_version": 1,
            "axes": self.axes.to_dict(),
            "angle_convention": BODY_ANGLE_CONVENTION,
            "origin_array_ref": "origin_xy",
            "forward_axis_array_ref": "forward_axis_xy",
            "left_axis_array_ref": "left_axis_xy",
            "axis_valid_array_ref": "axis_valid",
            "invalid_row_encoding": BODY_INVALID_ROW_ENCODING,
        }

    def digest(self) -> str:
        return _mapping_sha256(self.to_dict())


def parse_body_frame_contract_record(value: Any) -> BodyFrameContractRecord:
    if isinstance(value, BodyFrameContractRecord):
        value = value.to_dict()
    payload = _mapping(value, path="$")
    _exact_fields(
        payload,
        {
            "schema_id",
            "schema_version",
            "axes",
            "angle_convention",
            "origin_array_ref",
            "forward_axis_array_ref",
            "left_axis_array_ref",
            "axis_valid_array_ref",
            "invalid_row_encoding",
        },
        path="$",
    )
    if (
        payload["schema_id"] != BODY_FRAME_CONTRACT_SCHEMA_ID
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
    ):
        _fail("schema_invalid", "$", "Unsupported body-frame contract schema.")
    expected = BodyFrameContractRecord().to_dict()
    if dict(payload) != expected:
        # Parse axes separately to provide a stable scientific mismatch code.
        _parse_body_axes(payload["axes"], path="$.axes")
        _fail(
            "body_contract_invalid",
            "$",
            "Body-frame contract differs from canonical v1.",
        )
    return BodyFrameContractRecord()


def build_body_frame_contract_record() -> BodyFrameContractRecord:
    return BodyFrameContractRecord()


@dataclass(frozen=True, init=False)
class BoundBodyFrameContract:
    record_ref: str
    record_sha256: str
    record: BodyFrameContractRecord
    archive_identity: ArchiveIdentity = field(repr=False, compare=False)
    _node: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        record_ref: str,
        record_sha256: str,
        record: BodyFrameContractRecord,
        archive_identity: ArchiveIdentity,
        node: Any,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BODY_CONTRACT_SEAL:
            _fail(
                "body_contract_unsealed",
                "$",
                "Body contract must come from its typed stamp/loader.",
            )
        for name, item in locals().copy().items():
            if name not in {"self", "node", "_verification_seal"}:
                object.__setattr__(self, name, item)
        object.__setattr__(self, "_node", node)
        object.__setattr__(self, "_seal", _verification_seal)

    @property
    def ref(self) -> DigestBoundFrameRecordRef:
        return DigestBoundFrameRecordRef(self.record_ref, self.record_sha256)


def load_bound_body_frame_contract(
    node: Any,
    *,
    expected_record_ref: str,
    expected_record_sha256: str,
) -> BoundBodyFrameContract:
    identity = _archive_for_nodes(node)
    record_ref, record, digest = _load_record_attrs(
        node,
        attr_name=BODY_FRAME_CONTRACT_ATTR,
        parser=parse_body_frame_contract_record,
    )
    if record_ref != _canonical_record_ref(
        expected_record_ref, path="expected_record_ref"
    ):
        _fail(
            "record_path_mismatch",
            "expected_record_ref",
            "Body contract path differs from expectation.",
        )
    if digest != _digest(expected_record_sha256, path="expected_record_sha256"):
        _fail(
            "record_digest_mismatch",
            "expected_record_sha256",
            "Body contract digest differs from expectation.",
        )
    return BoundBodyFrameContract(
        record_ref=record_ref,
        record_sha256=digest,
        record=record,
        archive_identity=identity,
        node=node,
        _verification_seal=_BODY_CONTRACT_SEAL,
    )


def verify_bound_body_frame_contract(
    value: BoundBodyFrameContract,
) -> BoundBodyFrameContract:
    if (
        type(value) is not BoundBodyFrameContract
        or value._seal is not _BODY_CONTRACT_SEAL
    ):
        _fail(
            "body_contract_unsealed", "$", "A sealed body-frame contract is required."
        )
    current = load_bound_body_frame_contract(
        value._node,
        expected_record_ref=value.record_ref,
        expected_record_sha256=value.record_sha256,
    )
    if current.record != value.record:
        _fail(
            "body_contract_stale",
            value.record_ref,
            "Body contract changed after binding.",
        )
    if current.archive_identity != value.archive_identity:
        _fail(
            "archive_mismatch", "$", "Body contract moved to a different archive/store."
        )
    return value


def stamp_body_frame_contract(
    node: Any,
    *,
    record: BodyFrameContractRecord | Mapping[str, Any] | str,
) -> BoundBodyFrameContract:
    _archive_for_nodes(node)
    parsed = parse_body_frame_contract_record(record)
    ref = _node_record_ref(node, BODY_FRAME_CONTRACT_ATTR)
    return _transactional_stamp(
        node,
        attr_name=BODY_FRAME_CONTRACT_ATTR,
        payload=parsed.to_dict(),
        reload_and_verify=lambda: load_bound_body_frame_contract(
            node,
            expected_record_ref=ref,
            expected_record_sha256=parsed.digest(),
        ),
    )


def _json_value(value: Any, *, path: str) -> Any:
    """Return a detached canonical JSON value or fail on exotic/NaN content."""

    try:
        return json.loads(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as exc:
        _fail("json_value_invalid", path, str(exc))
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class BodyFrameEstimatorRecord:
    method: str
    implementation_version: str
    configuration_schema_id: str
    configuration: Mapping[str, Any]
    formula_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": BODY_FRAME_ESTIMATOR_SCHEMA_ID,
            "schema_version": 1,
            "method": self.method,
            "implementation_version": self.implementation_version,
            "configuration_schema_id": self.configuration_schema_id,
            "configuration": copy.deepcopy(dict(self.configuration)),
            "formula_id": self.formula_id,
        }

    def digest(self) -> str:
        return _mapping_sha256(self.to_dict())


def parse_body_frame_estimator_record(value: Any) -> BodyFrameEstimatorRecord:
    if isinstance(value, BodyFrameEstimatorRecord):
        value = value.to_dict()
    payload = _mapping(value, path="$")
    _exact_fields(
        payload,
        {
            "schema_id",
            "schema_version",
            "method",
            "implementation_version",
            "configuration_schema_id",
            "configuration",
            "formula_id",
        },
        path="$",
    )
    if (
        payload["schema_id"] != BODY_FRAME_ESTIMATOR_SCHEMA_ID
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
    ):
        _fail("schema_invalid", "$", "Unsupported body-frame estimator schema.")
    method = payload["method"]
    if method not in BODY_ESTIMATOR_METHODS:
        _fail(
            "estimator_method_invalid",
            "$.method",
            "Unsupported body-frame estimator method.",
        )
    version = _text(
        payload["implementation_version"],
        path="$.implementation_version",
        pattern=_VERSION_RE,
    )
    config_schema = _text(
        payload["configuration_schema_id"],
        path="$.configuration_schema_id",
        pattern=_SCHEMA_RE,
    )
    config = _mapping(payload["configuration"], path="$.configuration")
    normalized = _json_value(dict(config), path="$.configuration")
    expected_schema, expected_fields = BODY_ESTIMATOR_CONFIGURATION_SCHEMAS[method]
    if config_schema != expected_schema:
        _fail(
            "estimator_configuration_schema_mismatch",
            "$.configuration_schema_id",
            f"Estimator method {method!r} requires {expected_schema!r}.",
        )
    _exact_fields(normalized, expected_fields, path="$.configuration")
    for name in sorted(expected_fields):
        _text(normalized[name], path=f"$.configuration.{name}", pattern=_IDENTIFIER_RE)
    if len(set(normalized.values())) != len(normalized):
        _fail(
            "estimator_anchor_alias",
            "$.configuration",
            "Estimator source and polarity/anchor labels must be distinct.",
        )
    if normalized != BODY_ESTIMATOR_REQUIRED_ANCHORS:
        _fail(
            "estimator_anchor_contract_mismatch",
            "$.configuration",
            "Canonical body estimators require exact eye_left, eye_right, and swim_bladder anchors.",
        )
    formula_id = _text(
        payload["formula_id"],
        path="$.formula_id",
        pattern=_IDENTIFIER_RE,
    )
    if formula_id != BODY_ESTIMATOR_FORMULAS[method]:
        _fail(
            "estimator_formula_mismatch",
            "$.formula_id",
            "Estimator formula does not match the controlled method definition.",
        )
    return BodyFrameEstimatorRecord(
        method,
        version,
        config_schema,
        normalized,
        formula_id,
    )


def build_body_frame_estimator_record(
    *,
    method: str,
    implementation_version: str,
    configuration_schema_id: str,
    configuration: Mapping[str, Any],
) -> BodyFrameEstimatorRecord:
    return parse_body_frame_estimator_record(
        {
            "schema_id": BODY_FRAME_ESTIMATOR_SCHEMA_ID,
            "schema_version": 1,
            "method": method,
            "implementation_version": implementation_version,
            "configuration_schema_id": configuration_schema_id,
            "configuration": configuration,
            "formula_id": BODY_ESTIMATOR_FORMULAS.get(method),
        }
    )


@dataclass(frozen=True, init=False)
class BoundBodyFrameEstimator:
    record_ref: str
    record_sha256: str
    record: BodyFrameEstimatorRecord
    archive_identity: ArchiveIdentity = field(repr=False, compare=False)
    _node: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        record_ref: str,
        record_sha256: str,
        record: BodyFrameEstimatorRecord,
        archive_identity: ArchiveIdentity,
        node: Any,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BODY_ESTIMATOR_SEAL:
            _fail(
                "body_estimator_unsealed",
                "$",
                "Body estimator must come from its typed stamp/loader.",
            )
        for name, item in locals().copy().items():
            if name not in {"self", "node", "_verification_seal"}:
                object.__setattr__(self, name, item)
        object.__setattr__(self, "_node", node)
        object.__setattr__(self, "_seal", _verification_seal)

    @property
    def ref(self) -> DigestBoundFrameRecordRef:
        return DigestBoundFrameRecordRef(self.record_ref, self.record_sha256)


def load_bound_body_frame_estimator(
    node: Any, *, expected_record_ref: str, expected_record_sha256: str
) -> BoundBodyFrameEstimator:
    identity = _archive_for_nodes(node)
    record_ref, record, digest = _load_record_attrs(
        node,
        attr_name=BODY_FRAME_ESTIMATOR_ATTR,
        parser=parse_body_frame_estimator_record,
    )
    if record_ref != _canonical_record_ref(
        expected_record_ref, path="expected_record_ref"
    ):
        _fail(
            "record_path_mismatch",
            "expected_record_ref",
            "Body estimator path differs from expectation.",
        )
    if digest != _digest(expected_record_sha256, path="expected_record_sha256"):
        _fail(
            "record_digest_mismatch",
            "expected_record_sha256",
            "Body estimator digest differs from expectation.",
        )
    return BoundBodyFrameEstimator(
        record_ref=record_ref,
        record_sha256=digest,
        record=record,
        archive_identity=identity,
        node=node,
        _verification_seal=_BODY_ESTIMATOR_SEAL,
    )


def verify_bound_body_frame_estimator(
    value: BoundBodyFrameEstimator,
) -> BoundBodyFrameEstimator:
    if (
        type(value) is not BoundBodyFrameEstimator
        or value._seal is not _BODY_ESTIMATOR_SEAL
    ):
        _fail(
            "body_estimator_unsealed", "$", "A sealed body-frame estimator is required."
        )
    current = load_bound_body_frame_estimator(
        value._node,
        expected_record_ref=value.record_ref,
        expected_record_sha256=value.record_sha256,
    )
    if current.record != value.record:
        _fail(
            "body_estimator_stale",
            value.record_ref,
            "Body estimator changed after binding.",
        )
    if current.archive_identity != value.archive_identity:
        _fail(
            "archive_mismatch",
            "$",
            "Body estimator moved to a different archive/store.",
        )
    return value


def stamp_body_frame_estimator(
    node: Any, *, record: BodyFrameEstimatorRecord | Mapping[str, Any] | str
) -> BoundBodyFrameEstimator:
    _archive_for_nodes(node)
    parsed = parse_body_frame_estimator_record(record)
    ref = _node_record_ref(node, BODY_FRAME_ESTIMATOR_ATTR)
    return _transactional_stamp(
        node,
        attr_name=BODY_FRAME_ESTIMATOR_ATTR,
        payload=parsed.to_dict(),
        reload_and_verify=lambda: load_bound_body_frame_estimator(
            node, expected_record_ref=ref, expected_record_sha256=parsed.digest()
        ),
    )


@dataclass(frozen=True, init=False)
class BoundBodySourceCoordinateDescriptor:
    """Exact canonical coordinate surface from which body axes were estimated."""

    record_ref: str
    record_sha256: str
    descriptor: CanonicalCoordinateDescriptor
    source_payload: "BodyGeometryArrayRecord"
    row_identity: BoundRowIdentityContract
    source_camera_pixels: BoundPixelFrameAuthority | None
    physical_frame: BoundPhysicalFrameCalibration | None
    lineage_records: tuple[BoundCoordinateRecord, ...]
    archive_identity: ArchiveIdentity = field(repr=False, compare=False)
    _coordinate_node: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        record_ref: str,
        record_sha256: str,
        descriptor: CanonicalCoordinateDescriptor,
        source_payload: "BodyGeometryArrayRecord",
        row_identity: BoundRowIdentityContract,
        source_camera_pixels: BoundPixelFrameAuthority | None,
        physical_frame: BoundPhysicalFrameCalibration | None,
        lineage_records: tuple[BoundCoordinateRecord, ...],
        archive_identity: ArchiveIdentity,
        coordinate_node: Any,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BODY_SOURCE_SEAL:
            _fail(
                "body_source_descriptor_unsealed",
                "$",
                "Body source must come from its exact descriptor binder.",
            )
        for name, item in locals().copy().items():
            if name not in {"self", "coordinate_node", "_verification_seal"}:
                object.__setattr__(self, name, item)
        object.__setattr__(self, "_coordinate_node", coordinate_node)
        object.__setattr__(self, "_seal", _verification_seal)

    @property
    def ref(self) -> DigestBoundFrameRecordRef:
        return DigestBoundFrameRecordRef(self.record_ref, self.record_sha256)

    @property
    def coordinate_units(self) -> str:
        return CANONICAL_COORDINATE_PROFILES[self.descriptor.profile_id].coordinate_unit

    @property
    def positive_x(self) -> str:
        return self.descriptor.positive_directions.x

    @property
    def positive_y(self) -> str:
        return self.descriptor.positive_directions.y

    @property
    def dependency_nodes(self) -> tuple[Any, ...]:
        """Every exact persisted node retained by the source binding."""

        nodes: list[Any] = [
            self._coordinate_node,
            self.row_identity._rowset_node,
            self.row_identity._key_array_node,
        ]
        if self.source_camera_pixels is not None:
            nodes.append(self.source_camera_pixels._authority_node)
        if self.physical_frame is not None:
            nodes.extend(
                (
                    self.physical_frame._node,
                    self.physical_frame.source_camera_pixels._authority_node,
                    self.physical_frame.selected_camera_evidence._node,
                )
            )
        nodes.extend(item._node for item in self.lineage_records)
        unique: list[Any] = []
        seen: set[tuple[str, ArchiveIdentity]] = set()
        for node in nodes:
            key = (_node_path(node), _archive_for_nodes(node))
            if key not in seen:
                seen.add(key)
                unique.append(node)
        return tuple(unique)


def _coordinate_shape(node: Any) -> tuple[int, ...]:
    raw = getattr(node, "shape", None)
    if not isinstance(raw, (tuple, list)):
        _fail(
            "coordinate_shape_invalid",
            "coordinate_node.shape",
            "Coordinate node must expose a shape.",
        )
    result = tuple(
        _exact_int(item, path=f"coordinate_node.shape[{index}]")
        for index, item in enumerate(raw)
    )
    if not result:
        _fail(
            "coordinate_shape_invalid",
            "coordinate_node.shape",
            "Coordinate surface requires a leading row dimension.",
        )
    return result


def bind_body_source_coordinate_descriptor(
    coordinate_node: Any,
    *,
    row_identity: BoundRowIdentityContract,
    source_camera_pixels: BoundPixelFrameAuthority | None = None,
    physical_frame: BoundPhysicalFrameCalibration | None = None,
    lineage_records: Sequence[BoundCoordinateRecord] = (),
) -> BoundBodySourceCoordinateDescriptor:
    """Bind exact source payload, descriptor, identity, and complete lineage.

    The descriptor is first parsed only to select the deliberately narrow body
    source profile.  It is then loaded through the canonical publication
    boundary with sealed records for every lineage ref; arbitrary path/digest
    pairs never authorize a body-frame source.
    """

    row_identity = _require_row_identity(row_identity, body_frame=True)
    shape = _coordinate_shape(coordinate_node)
    source_payload_snapshot = _read_array_snapshot(coordinate_node)
    if source_payload_snapshot.values.dtype.kind not in "fiu":
        _fail(
            "body_source_dtype_invalid",
            "coordinate_node.dtype",
            "Body-frame source coordinates must use a real numeric dtype.",
        )
    attrs = getattr(coordinate_node, "attrs", None)
    if not isinstance(attrs, Mapping):
        _fail(
            "attrs_unavailable",
            "coordinate_node",
            "Coordinate node must expose descriptor attrs.",
        )
    try:
        descriptor = load_canonical_coordinate_descriptor_attrs(
            attrs,
            row_identity_contract=row_identity.contract,
            expected_row_identity_record_ref=row_identity.record_ref,
            owner_shape=shape,
        )
    except CoordinateDescriptorError as exc:
        _fail(
            "source_descriptor_invalid",
            f"/{_node_path(coordinate_node)}@{COORDINATE_DESCRIPTOR_ATTR}",
            str(exc),
        )
    if descriptor.profile_id not in SUPPORTED_BODY_SOURCE_PROFILE_IDS:
        _fail(
            "body_source_profile_unsupported",
            "$.profile_id",
            "Body-frame v1 supports exact source-camera pixels or its bound physical-mm frame; ROI/model/body inputs require typed directed lineage not yet accepted here.",
        )

    verified_additional_lineage: list[BoundCoordinateRecord] = []
    for index, item in enumerate(lineage_records):
        try:
            current = verify_bound_coordinate_record(item)
        except CoordinateRecordError as exc:
            _fail(
                "source_descriptor_lineage_unverified",
                f"lineage_records[{index}]",
                str(exc),
            )
        verified_additional_lineage.append(current)

    complete_lineage: list[BoundCoordinateRecord]
    if descriptor.profile_id == SOURCE_CAMERA_PROFILE_ID:
        if source_camera_pixels is None or physical_frame is not None:
            _fail(
                "body_source_authority_mismatch",
                "$",
                "Source-camera body input requires its exact acquisition-owned pixel-frame authority only.",
            )
        try:
            camera = require_source_camera_pixel_frame_authority(source_camera_pixels)
        except PixelFrameAuthorityError as exc:
            _fail("source_camera_frame_unverified", "source_camera_pixels", str(exc))
        complete_lineage = list(verified_additional_lineage)
        expected_frame: BoundPhysicalFrameCalibration | None = None
    else:
        if (
            physical_frame is None
            or source_camera_pixels is not None
        ):
            _fail(
                "body_source_authority_mismatch",
                "$",
                "Physical body input requires one exact physical frame authority only.",
            )
        frame = verify_bound_coordinate_frame(
            physical_frame, expected_kind=PHYSICAL_FRAME_CALIBRATION_KIND
        )
        if descriptor.profile_id not in frame.compatible_profile_ids:
            _fail(
                "physical_profile_incompatible",
                "$.profile_id",
                "Physical frame does not explicitly authorize this coordinate profile.",
            )
        camera = None
        expected_frame = frame
        complete_lineage = list(verified_additional_lineage)

    lineage_pairs = [
        (item.record_ref, item.record_sha256) for item in complete_lineage
    ]
    if len(lineage_pairs) != len(set(lineage_pairs)):
        _fail(
            "source_descriptor_lineage_duplicate",
            "lineage_records",
            "Each exact lineage record may be resolved only once.",
        )
    try:
        # Imported lazily: canonical publication resolves frame evidence by
        # importing this module lazily in the opposite direction.
        from fisheye.shared.canonical_coordinate_publication import (
            load_bound_canonical_coordinate_descriptor,
        )

        canonical = load_bound_canonical_coordinate_descriptor(
            coordinate_node,
            row_identity=row_identity,
            reference_frame_authority=camera,
            lineage_records=tuple(complete_lineage),
            frame_record=expected_frame,
        )
    except (CoordinateDescriptorError, CoordinateRecordError) as exc:
        _fail(
            "source_descriptor_lineage_unverified",
            f"/{_node_path(coordinate_node)}@{COORDINATE_DESCRIPTOR_ATTR}",
            str(exc),
        )
    descriptor = canonical.descriptor
    dependency_nodes = [
        coordinate_node,
        row_identity._rowset_node,
        row_identity._key_array_node,
        *(item._node for item in complete_lineage),
    ]
    if camera is not None:
        dependency_nodes.append(camera._authority_node)
        expected_archive = camera.archive_identity
    else:
        assert expected_frame is not None
        dependency_nodes.extend(
            (
                expected_frame._node,
                expected_frame.source_camera_pixels._authority_node,
                expected_frame.selected_camera_evidence._node,
            )
        )
        expected_archive = expected_frame.archive_identity
    identity = _archive_for_nodes(*dependency_nodes)
    if identity != row_identity.archive_identity or identity != expected_archive:
        _fail(
            "archive_mismatch",
            "$",
            "Body source descriptor authorities do not share one archive/store.",
        )
    owner_path = _node_path(coordinate_node)
    if not owner_path.startswith(f"{row_identity.rowset_path}/"):
        _fail(
            "source_descriptor_rowset_mismatch",
            "coordinate_node.path",
            "Source coordinates are not descendants of the exact identity rowset.",
        )
    _recheck_array_snapshot(coordinate_node, source_payload_snapshot)
    source_payload = _geometry_array_record_from_snapshot(
        coordinate_node,
        source_payload_snapshot,
    )
    return BoundBodySourceCoordinateDescriptor(
        record_ref=f"/{_node_path(coordinate_node)}@{COORDINATE_DESCRIPTOR_ATTR}",
        record_sha256=descriptor.digest(),
        descriptor=descriptor,
        source_payload=source_payload,
        row_identity=row_identity,
        source_camera_pixels=camera,
        physical_frame=expected_frame,
        lineage_records=tuple(complete_lineage),
        archive_identity=identity,
        coordinate_node=coordinate_node,
        _verification_seal=_BODY_SOURCE_SEAL,
    )


def verify_bound_body_source_coordinate_descriptor(
    value: BoundBodySourceCoordinateDescriptor,
) -> BoundBodySourceCoordinateDescriptor:
    if (
        type(value) is not BoundBodySourceCoordinateDescriptor
        or value._seal is not _BODY_SOURCE_SEAL
    ):
        _fail(
            "body_source_descriptor_unsealed",
            "$",
            "A sealed body source descriptor is required.",
        )
    current = bind_body_source_coordinate_descriptor(
        value._coordinate_node,
        row_identity=value.row_identity,
        source_camera_pixels=value.source_camera_pixels,
        physical_frame=value.physical_frame,
        lineage_records=value.lineage_records,
    )
    if (
        current.record_ref != value.record_ref
        or current.record_sha256 != value.record_sha256
        or current.descriptor != value.descriptor
        or current.source_payload != value.source_payload
        or tuple(
            (item.record_ref, item.record_sha256)
            for item in current.lineage_records
        )
        != tuple(
            (item.record_ref, item.record_sha256)
            for item in value.lineage_records
        )
    ):
        _fail(
            "body_source_descriptor_stale",
            value.record_ref,
            "Source coordinate payload, descriptor, or lineage changed after binding.",
        )
    if current.archive_identity != value.archive_identity:
        _fail(
            "archive_mismatch",
            "$",
            "Body source descriptor moved to a different archive/store.",
        )
    return value


@dataclass(frozen=True)
class BodyGeometryArrayRecord:
    array_ref: str
    dtype: str
    shape: tuple[int, ...]
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "array_ref": self.array_ref,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "content_sha256": self.content_sha256,
            "canonicalization": ARRAY_PAYLOAD_CANONICALIZATION,
        }


def _parse_geometry_array_record(value: Any, *, path: str) -> BodyGeometryArrayRecord:
    payload = _mapping(value, path=path)
    _exact_fields(
        payload,
        {"array_ref", "dtype", "shape", "content_sha256", "canonicalization"},
        path=path,
    )
    ref = _canonical_record_ref(payload["array_ref"], path=f"{path}.array_ref")
    if "@" in ref:
        _fail(
            "array_ref_invalid",
            f"{path}.array_ref",
            "Geometry array refs cannot select attrs.",
        )
    dtype_text = _text(payload["dtype"], path=f"{path}.dtype")
    try:
        canonical_dtype = np.dtype(dtype_text).str
    except TypeError as exc:
        _fail("array_dtype_invalid", f"{path}.dtype", str(exc))
    if dtype_text != canonical_dtype:
        _fail(
            "array_dtype_invalid",
            f"{path}.dtype",
            "Array dtype must use NumPy canonical .str form.",
        )
    raw_shape = payload["shape"]
    if not isinstance(raw_shape, list):
        _fail(
            "array_shape_invalid", f"{path}.shape", "Shape must be a JSON integer list."
        )
    shape = tuple(
        _exact_int(item, path=f"{path}.shape[{index}]")
        for index, item in enumerate(raw_shape)
    )
    if payload["canonicalization"] != ARRAY_PAYLOAD_CANONICALIZATION:
        _fail(
            "array_canonicalization_invalid",
            f"{path}.canonicalization",
            "Unsupported array content canonicalization.",
        )
    return BodyGeometryArrayRecord(
        ref,
        dtype_text,
        shape,
        _digest(payload["content_sha256"], path=f"{path}.content_sha256"),
    )


@dataclass(frozen=True)
class BodyEstimatorSourceBundleRecord:
    method: str
    formula_id: str
    source_descriptor: DigestBoundFrameRecordRef
    source_payload: BodyGeometryArrayRecord
    source_schema: DigestBoundFrameRecordRef
    labels: tuple[str, ...]
    support_arrays: Mapping[str, BodyGeometryArrayRecord]
    producer_manifest: DigestBoundFrameRecordRef

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": BODY_ESTIMATOR_SOURCE_BUNDLE_SCHEMA_ID,
            "schema_version": 1,
            "method": self.method,
            "formula_id": self.formula_id,
            "source_descriptor": self.source_descriptor.to_dict(),
            "source_payload": self.source_payload.to_dict(),
            "source_schema": self.source_schema.to_dict(),
            "labels": list(self.labels),
            "support_arrays": {
                name: self.support_arrays[name].to_dict()
                for name in sorted(self.support_arrays)
            },
            "producer_manifest": self.producer_manifest.to_dict(),
        }

    def digest(self) -> str:
        return _mapping_sha256(self.to_dict())


def _parse_body_estimator_source_bundle_record(
    value: Any,
) -> BodyEstimatorSourceBundleRecord:
    if isinstance(value, BodyEstimatorSourceBundleRecord):
        value = value.to_dict()
    payload = _mapping(value, path="$")
    _exact_fields(
        payload,
        {
            "schema_id",
            "schema_version",
            "method",
            "formula_id",
            "source_descriptor",
            "source_payload",
            "source_schema",
            "labels",
            "support_arrays",
            "producer_manifest",
        },
        path="$",
    )
    if (
        payload["schema_id"] != BODY_ESTIMATOR_SOURCE_BUNDLE_SCHEMA_ID
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
    ):
        _fail("schema_invalid", "$", "Unsupported estimator-source bundle schema.")
    method = payload["method"]
    if method not in BODY_ESTIMATOR_METHODS:
        _fail("estimator_method_invalid", "$.method", "Unsupported estimator method.")
    formula_id = _text(payload["formula_id"], path="$.formula_id", pattern=_IDENTIFIER_RE)
    if formula_id != BODY_ESTIMATOR_FORMULAS[method]:
        _fail(
            "estimator_formula_mismatch",
            "$.formula_id",
            "Estimator source bundle uses an unsupported formula for its method.",
        )
    raw_labels = payload["labels"]
    if not isinstance(raw_labels, list) or not raw_labels:
        _fail("source_labels_invalid", "$.labels", "A non-empty label list is required.")
    labels = tuple(
        _text(item, path=f"$.labels[{index}]", pattern=_IDENTIFIER_RE)
        for index, item in enumerate(raw_labels)
    )
    if len(labels) != len(set(labels)):
        _fail("source_labels_duplicate", "$.labels", "Source labels must be unique.")
    raw_arrays = _mapping(payload["support_arrays"], path="$.support_arrays")
    expected_roles = {
        "keypoint_head_axis": {"validity"},
        "mask_component_axis": {"validity"},
        "body_spline_with_anchor_polarity": {"polarity_anchors", "polarity_valid"},
    }[method]
    _exact_fields(raw_arrays, expected_roles, path="$.support_arrays")
    return BodyEstimatorSourceBundleRecord(
        method=method,
        formula_id=formula_id,
        source_descriptor=_parse_record_ref(
            payload["source_descriptor"], path="$.source_descriptor"
        ),
        source_payload=_parse_geometry_array_record(
            payload["source_payload"], path="$.source_payload"
        ),
        source_schema=_parse_record_ref(
            payload["source_schema"], path="$.source_schema"
        ),
        labels=labels,
        support_arrays={
            name: _parse_geometry_array_record(
                raw_arrays[name], path=f"$.support_arrays.{name}"
            )
            for name in sorted(expected_roles)
        },
        producer_manifest=_parse_record_ref(
            payload["producer_manifest"], path="$.producer_manifest"
        ),
    )


@dataclass(frozen=True, init=False)
class BoundBodyEstimatorSourceBundle:
    record: BodyEstimatorSourceBundleRecord
    source_descriptor: BoundBodySourceCoordinateDescriptor
    estimator: BoundBodyFrameEstimator
    source_schema: BoundCoordinateRecord
    producer_manifest: BoundCoordinateRecord
    archive_identity: ArchiveIdentity = field(repr=False, compare=False)
    _support_nodes: Mapping[str, Any] = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        record: BodyEstimatorSourceBundleRecord,
        source_descriptor: BoundBodySourceCoordinateDescriptor,
        estimator: BoundBodyFrameEstimator,
        source_schema: BoundCoordinateRecord,
        producer_manifest: BoundCoordinateRecord,
        archive_identity: ArchiveIdentity,
        support_nodes: Mapping[str, Any],
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BODY_ESTIMATOR_SOURCE_SEAL:
            _fail(
                "estimator_source_unsealed",
                "$",
                "Estimator source must come from a typed exact-source binder.",
            )
        object.__setattr__(self, "record", record)
        object.__setattr__(self, "source_descriptor", source_descriptor)
        object.__setattr__(self, "estimator", estimator)
        object.__setattr__(self, "source_schema", source_schema)
        object.__setattr__(self, "producer_manifest", producer_manifest)
        object.__setattr__(self, "archive_identity", archive_identity)
        object.__setattr__(self, "_support_nodes", dict(support_nodes))
        object.__setattr__(self, "_seal", _verification_seal)


def _schema_labels(
    record: BoundCoordinateRecord,
    *,
    method: str,
) -> tuple[str, ...]:
    expected = {
        "keypoint_head_axis": ("palette.keypoint_label_schema", "labels"),
        "mask_component_axis": ("palette.mask_component_geometry_schema", "components"),
        "body_spline_with_anchor_polarity": (
            "palette.body_spline_polarity_schema",
            "anchors",
        ),
    }
    schema_id, labels_field = expected[method]
    raw = record.record
    if set(raw) != {"schema_id", "schema_version", labels_field}:
        _fail(
            "estimator_source_schema_invalid",
            "source_schema",
            "Estimator source schema has missing or unknown fields.",
        )
    if raw["schema_id"] != schema_id or type(raw["schema_version"]) is not int or raw[
        "schema_version"
    ] != 1:
        _fail(
            "estimator_source_schema_invalid",
            "source_schema",
            f"Method {method!r} requires exact schema {schema_id!r} version 1.",
        )
    raw_labels = raw[labels_field]
    if not isinstance(raw_labels, list) or not raw_labels:
        _fail("source_labels_invalid", "source_schema", "Labels must be a non-empty list.")
    labels = tuple(
        _text(item, path=f"source_schema.{labels_field}[{index}]", pattern=_IDENTIFIER_RE)
        for index, item in enumerate(raw_labels)
    )
    if len(labels) != len(set(labels)):
        _fail("source_labels_duplicate", "source_schema", "Labels must be unique.")
    return labels


def build_body_estimator_source_manifest_record(
    *,
    method: str,
    source_descriptor: BoundBodySourceCoordinateDescriptor,
    estimator: BoundBodyFrameEstimator,
    source_schema: BoundCoordinateRecord,
    support_nodes: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact rowset-owned manifest that a typed body writer must stamp."""

    source = verify_bound_body_source_coordinate_descriptor(source_descriptor)
    estimator = verify_bound_body_frame_estimator(estimator)
    try:
        schema = verify_bound_coordinate_record(source_schema)
    except CoordinateRecordError as exc:
        _fail("estimator_source_schema_unverified", "source_schema", str(exc))
    if method not in BODY_ESTIMATOR_METHODS or estimator.record.method != method:
        _fail(
            "estimator_source_method_mismatch",
            "estimator",
            "Manifest method must equal the exact selected estimator method.",
        )
    labels = _schema_labels(schema, method=method)
    expected_roles = {
        "keypoint_head_axis": {"validity"},
        "mask_component_axis": {"validity"},
        "body_spline_with_anchor_polarity": {
            "polarity_anchors",
            "polarity_valid",
        },
    }[method]
    if set(support_nodes) != expected_roles:
        _fail(
            "estimator_source_support_roles_invalid",
            "support_arrays",
            "Support-array roles must exactly match the controlled estimator method.",
        )
    rowset_path = source.row_identity.rowset_path
    source_path = _node_path(source._coordinate_node)
    support_paths = {name: _node_path(node) for name, node in support_nodes.items()}
    if source_path.rpartition("/")[0] != rowset_path or any(
        path.rpartition("/")[0] != rowset_path for path in support_paths.values()
    ):
        _fail(
            "estimator_source_rowset_path_mismatch",
            "support_arrays",
            "Source coordinates and every support array must be exact direct children of the owning identity rowset.",
        )
    if len({source_path, *support_paths.values()}) != 1 + len(support_paths):
        _fail(
            "estimator_source_role_alias",
            "support_arrays",
            "Source and support-array roles must use distinct exact nodes.",
        )
    snapshots = {
        name: _read_array_snapshot(node) for name, node in support_nodes.items()
    }
    identity = _archive_for_nodes(
        source._coordinate_node,
        estimator._node,
        schema._node,
        source.row_identity._rowset_node,
        source.row_identity._key_array_node,
        *support_nodes.values(),
    )
    if any(
        item != identity
        for item in (
            source.archive_identity,
            estimator.archive_identity,
            schema.archive_identity,
        )
    ):
        _fail("archive_mismatch", "$", "Estimator source authorities span archives.")
    for name, node in support_nodes.items():
        _recheck_array_snapshot(node, snapshots[name])
    return {
        "schema_id": BODY_ESTIMATOR_SOURCE_MANIFEST_SCHEMA_ID,
        "schema_version": 1,
        "producer_contract": "palette.body_estimator_exact_source_v1",
        "method": method,
        "formula_id": estimator.record.formula_id,
        "row_identity": {
            "record_ref": source.row_identity.record_ref,
            "record_sha256": source.row_identity.record_sha256,
        },
        "source_descriptor": source.ref.to_dict(),
        "source_payload": source.source_payload.to_dict(),
        "source_schema": {
            "record_ref": schema.record_ref,
            "record_sha256": schema.record_sha256,
        },
        "labels": list(labels),
        "support_arrays": {
            name: _geometry_array_record_from_snapshot(node, snapshots[name]).to_dict()
            for name, node in sorted(support_nodes.items())
        },
    }


def _bind_body_estimator_source(
    *,
    method: str,
    source_descriptor: BoundBodySourceCoordinateDescriptor,
    estimator: BoundBodyFrameEstimator,
    source_schema: BoundCoordinateRecord,
    support_nodes: Mapping[str, Any],
    producer_manifest: BoundCoordinateRecord,
) -> BoundBodyEstimatorSourceBundle:
    source = verify_bound_body_source_coordinate_descriptor(source_descriptor)
    estimator = verify_bound_body_frame_estimator(estimator)
    try:
        schema = verify_bound_coordinate_record(source_schema)
    except CoordinateRecordError as exc:
        _fail("estimator_source_schema_unverified", "source_schema", str(exc))
    if estimator.record.method != method:
        _fail(
            "estimator_source_method_mismatch",
            "estimator",
            "Typed source bundle method differs from the selected estimator.",
        )
    labels = _schema_labels(schema, method=method)
    config_values = tuple(estimator.record.configuration.values())
    if any(item not in labels for item in config_values):
        _fail(
            "estimator_anchor_missing",
            "estimator.configuration",
            "Every configured estimator anchor/component must exist in the exact source schema.",
        )
    shape = source.source_payload.shape
    leading = source.row_identity.leading_dimension
    snapshots = {name: _read_array_snapshot(node) for name, node in support_nodes.items()}
    if method in {"keypoint_head_axis", "mask_component_axis"}:
        expected_shape = (leading, len(labels), 2)
        if method == "keypoint_head_axis":
            source_geometry_valid = (
                source.descriptor.geometry_type == "points_xy"
                and len(shape) == 3
                and shape == expected_shape
            )
        else:
            collection = source.descriptor.collection_axis
            source_geometry_valid = (
                source.descriptor.geometry_type == "point_xy"
                and len(shape) == 3
                and shape == expected_shape
                and collection is not None
                and collection.axis == 1
                and collection.role == "subject_component"
                and collection.cardinality == len(labels)
                and collection.label_authority.record_ref == schema.record_ref
                and collection.label_authority.record_sha256
                == schema.record_sha256
            )
        if not source_geometry_valid:
            _fail(
                "estimator_source_geometry_invalid",
                "source_descriptor",
                (
                    "Keypoint sources require points_xy with shape (N, labels, 2); "
                    "mask-component sources require collected point_xy with the "
                    "exact subject-component label authority and that same shape."
                ),
            )
        validity = snapshots.get("validity")
        if validity is None or validity.values.dtype != np.dtype("bool") or validity.shape != (
            leading,
            len(labels),
        ):
            _fail(
                "estimator_source_validity_invalid",
                "support_arrays.validity",
                "Exact boolean validity with shape (N, labels) is required.",
            )
    else:
        if source.descriptor.geometry_type != "polyline_xy" or len(shape) != 3 or shape[0] != leading or shape[2] != 2 or shape[1] < 2:
            _fail(
                "estimator_source_geometry_invalid",
                "source_descriptor",
                "Spline source requires polyline_xy with shape (N, P>=2, 2).",
            )
        anchors = snapshots.get("polarity_anchors")
        valid = snapshots.get("polarity_valid")
        if (
            anchors is None
            or anchors.values.dtype.kind not in "fiu"
            or anchors.shape != (leading, 3, 2)
            or valid is None
            or valid.values.dtype != np.dtype("bool")
            or valid.shape != (leading, 3)
        ):
            _fail(
                "estimator_source_polarity_invalid",
                "support_arrays",
                "Spline polarity requires numeric eye_left/eye_right/swim_bladder (N,3,2) anchors and exact boolean (N,3) validity.",
            )
    nodes = [
        source._coordinate_node,
        estimator._node,
        schema._node,
        source.row_identity._rowset_node,
        source.row_identity._key_array_node,
        *support_nodes.values(),
    ]
    identity = _archive_for_nodes(*nodes)
    if any(
        item != identity
        for item in (source.archive_identity, estimator.archive_identity, schema.archive_identity)
    ):
        _fail("archive_mismatch", "$", "Estimator source authorities span archives.")
    for name, node in support_nodes.items():
        _recheck_array_snapshot(node, snapshots[name])
    expected_manifest = build_body_estimator_source_manifest_record(
        method=method,
        source_descriptor=source,
        estimator=estimator,
        source_schema=schema,
        support_nodes=support_nodes,
    )
    try:
        manifest = verify_bound_coordinate_record(producer_manifest)
    except CoordinateRecordError as exc:
        _fail("estimator_source_manifest_unverified", "producer_manifest", str(exc))
    expected_manifest_ref = (
        f"/{source.row_identity.rowset_path}@{BODY_ESTIMATOR_SOURCE_MANIFEST_ATTR}"
    )
    if (
        manifest._node is not source.row_identity._rowset_node
        or manifest.record_ref != expected_manifest_ref
        or manifest.record != expected_manifest
        or manifest.archive_identity != identity
    ):
        _fail(
            "estimator_source_manifest_mismatch",
            "producer_manifest",
            "The exact owning-rowset producer manifest does not bind the selected source, schema, supports, formula, and row identity.",
        )
    record = _parse_body_estimator_source_bundle_record(
        {
            "schema_id": BODY_ESTIMATOR_SOURCE_BUNDLE_SCHEMA_ID,
            "schema_version": 1,
            "method": method,
            "formula_id": estimator.record.formula_id,
            "source_descriptor": source.ref.to_dict(),
            "source_payload": source.source_payload.to_dict(),
            "source_schema": {
                "record_ref": schema.record_ref,
                "record_sha256": schema.record_sha256,
            },
            "labels": list(labels),
            "support_arrays": {
                name: _geometry_array_record_from_snapshot(node, snapshots[name]).to_dict()
                for name, node in sorted(support_nodes.items())
            },
            "producer_manifest": {
                "record_ref": manifest.record_ref,
                "record_sha256": manifest.record_sha256,
            },
        }
    )
    return BoundBodyEstimatorSourceBundle(
        record=record,
        source_descriptor=source,
        estimator=estimator,
        source_schema=schema,
        producer_manifest=manifest,
        archive_identity=identity,
        support_nodes=support_nodes,
        _verification_seal=_BODY_ESTIMATOR_SOURCE_SEAL,
    )


def bind_keypoint_head_axis_source(
    *,
    source_descriptor: BoundBodySourceCoordinateDescriptor,
    estimator: BoundBodyFrameEstimator,
    keypoint_schema: BoundCoordinateRecord,
    validity_node: Any,
    producer_manifest: BoundCoordinateRecord,
) -> BoundBodyEstimatorSourceBundle:
    return _bind_body_estimator_source(
        method="keypoint_head_axis",
        source_descriptor=source_descriptor,
        estimator=estimator,
        source_schema=keypoint_schema,
        support_nodes={"validity": validity_node},
        producer_manifest=producer_manifest,
    )


def bind_mask_component_axis_source(
    *,
    source_descriptor: BoundBodySourceCoordinateDescriptor,
    estimator: BoundBodyFrameEstimator,
    component_schema: BoundCoordinateRecord,
    validity_node: Any,
    producer_manifest: BoundCoordinateRecord,
) -> BoundBodyEstimatorSourceBundle:
    return _bind_body_estimator_source(
        method="mask_component_axis",
        source_descriptor=source_descriptor,
        estimator=estimator,
        source_schema=component_schema,
        support_nodes={"validity": validity_node},
        producer_manifest=producer_manifest,
    )


def bind_body_spline_with_anchor_polarity_source(
    *,
    source_descriptor: BoundBodySourceCoordinateDescriptor,
    estimator: BoundBodyFrameEstimator,
    polarity_schema: BoundCoordinateRecord,
    polarity_anchors_node: Any,
    polarity_valid_node: Any,
    producer_manifest: BoundCoordinateRecord,
) -> BoundBodyEstimatorSourceBundle:
    return _bind_body_estimator_source(
        method="body_spline_with_anchor_polarity",
        source_descriptor=source_descriptor,
        estimator=estimator,
        source_schema=polarity_schema,
        support_nodes={
            "polarity_anchors": polarity_anchors_node,
            "polarity_valid": polarity_valid_node,
        },
        producer_manifest=producer_manifest,
    )


def verify_bound_body_estimator_source(
    value: Any,
) -> BoundBodyEstimatorSourceBundle:
    if (
        type(value) is not BoundBodyEstimatorSourceBundle
        or value._seal is not _BODY_ESTIMATOR_SOURCE_SEAL
    ):
        _fail("estimator_source_unsealed", "$", "A sealed typed estimator source is required.")
    current = _bind_body_estimator_source(
        method=value.record.method,
        source_descriptor=value.source_descriptor,
        estimator=value.estimator,
        source_schema=value.source_schema,
        support_nodes=value._support_nodes,
        producer_manifest=value.producer_manifest,
    )
    if current.record != value.record or current.archive_identity != value.archive_identity:
        _fail(
            "estimator_source_stale",
            "$",
            "Estimator source schema, values, validity, anchors, or archive changed.",
        )
    return value


@dataclass(frozen=True)
class BodyFrameGeometryRecord:
    origin_xy: BodyGeometryArrayRecord
    forward_axis_xy: BodyGeometryArrayRecord
    left_axis_xy: BodyGeometryArrayRecord
    axis_valid: BodyGeometryArrayRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_xy": self.origin_xy.to_dict(),
            "forward_axis_xy": self.forward_axis_xy.to_dict(),
            "left_axis_xy": self.left_axis_xy.to_dict(),
            "axis_valid": self.axis_valid.to_dict(),
        }


def _parse_body_geometry_record(
    value: Any, *, path: str = "$.geometry"
) -> BodyFrameGeometryRecord:
    payload = _mapping(value, path=path)
    _exact_fields(
        payload,
        {"origin_xy", "forward_axis_xy", "left_axis_xy", "axis_valid"},
        path=path,
    )
    return BodyFrameGeometryRecord(
        origin_xy=_parse_geometry_array_record(
            payload["origin_xy"], path=f"{path}.origin_xy"
        ),
        forward_axis_xy=_parse_geometry_array_record(
            payload["forward_axis_xy"], path=f"{path}.forward_axis_xy"
        ),
        left_axis_xy=_parse_geometry_array_record(
            payload["left_axis_xy"], path=f"{path}.left_axis_xy"
        ),
        axis_valid=_parse_geometry_array_record(
            payload["axis_valid"], path=f"{path}.axis_valid"
        ),
    )


def _geometry_array_record_from_snapshot(
    node: Any,
    snapshot: _ArrayPayloadSnapshot,
) -> BodyGeometryArrayRecord:
    return BodyGeometryArrayRecord(
        array_ref=f"/{_node_path(node)}",
        dtype=snapshot.dtype,
        shape=snapshot.shape,
        content_sha256=snapshot.content_sha256,
    )


def _validate_axis_geometry_values(
    *,
    origin: np.ndarray,
    forward: np.ndarray,
    left: np.ndarray,
    valid: np.ndarray,
    positive_y: str,
) -> None:
    if origin.dtype.kind != "f" or forward.dtype.kind != "f" or left.dtype.kind != "f":
        _fail(
            "geometry_dtype_invalid",
            "geometry",
            "Origin and axis arrays must use floating dtype.",
        )
    if origin.dtype != forward.dtype or origin.dtype != left.dtype:
        _fail(
            "geometry_dtype_mismatch",
            "geometry",
            "Origin and axis arrays must share one exact dtype.",
        )
    if origin.dtype not in (
        np.dtype("<f4"),
        np.dtype("<f8"),
        np.dtype("=f4"),
        np.dtype("=f8"),
    ):
        # np.dtype equality is byte-order agnostic for native values; this also
        # rejects float16/extended/structured arrays.
        _fail(
            "geometry_dtype_invalid",
            "geometry",
            "Geometry dtype must be float32 or float64.",
        )
    if valid.dtype != np.dtype("bool"):
        _fail(
            "axis_valid_dtype_invalid",
            "geometry.axis_valid",
            "axis_valid must use exact boolean dtype.",
        )
    valid_mask = valid.astype(bool, copy=False)
    invalid_mask = ~valid_mask
    for name, values in (
        ("origin_xy", origin),
        ("forward_axis_xy", forward),
        ("left_axis_xy", left),
    ):
        if np.any(~np.isfinite(values[valid_mask])):
            _fail(
                "geometry_nonfinite_valid",
                f"geometry.{name}",
                "Valid rows must contain finite geometry.",
            )
        if np.any(~np.isnan(values[invalid_mask])):
            _fail(
                "geometry_invalid_row_encoding",
                f"geometry.{name}",
                "Invalid rows must contain only NaN geometry.",
            )
    if not np.any(valid_mask):
        return
    forward_valid = forward[valid_mask].astype(np.float64)
    left_valid = left[valid_mask].astype(np.float64)
    tolerance = (
        BODY_AXIS_NORMAL_TOLERANCE_FLOAT32
        if origin.dtype.itemsize == 4
        else BODY_AXIS_NORMAL_TOLERANCE_FLOAT64
    )
    forward_norm = np.linalg.norm(forward_valid, axis=1)
    left_norm = np.linalg.norm(left_valid, axis=1)
    dot = np.einsum("ij,ij->i", forward_valid, left_valid)
    determinant = (
        forward_valid[:, 0] * left_valid[:, 1] - forward_valid[:, 1] * left_valid[:, 0]
    )
    if np.any(np.abs(forward_norm - 1.0) > tolerance) or np.any(
        np.abs(left_norm - 1.0) > tolerance
    ):
        _fail(
            "axis_norm_invalid",
            "geometry",
            "Valid forward/left vectors must have unit norm.",
        )
    if np.any(np.abs(dot) > tolerance):
        _fail(
            "axis_orthogonality_invalid",
            "geometry",
            "Valid forward/left vectors must be orthogonal.",
        )
    if positive_y == "down":
        expected_determinant = -1.0
    elif positive_y == "up":
        expected_determinant = 1.0
    else:
        _fail(
            "source_axis_polarity_unsupported",
            "source_descriptor.positive_y",
            "Cannot resolve anatomical-left polarity for this source axis.",
        )
    if np.any(np.abs(determinant - expected_determinant) > tolerance):
        _fail(
            "axis_polarity_invalid",
            "geometry.left_axis_xy",
            "Left-axis polarity disagrees with anatomical-left in the source coordinate basis.",
        )


def _derive_body_geometry_from_exact_source(
    *,
    estimator_source: BoundBodyEstimatorSourceBundle,
    source_values: np.ndarray,
    support_values: Mapping[str, np.ndarray],
    output_dtype: np.dtype[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recompute the controlled estimator formula from its digest-bound inputs."""

    method = estimator_source.record.method
    labels = estimator_source.record.labels
    config = estimator_source.estimator.record.configuration
    expected_config = BODY_ESTIMATOR_REQUIRED_ANCHORS
    if config != expected_config:
        _fail(
            "estimator_anchor_contract_mismatch",
            "estimator.configuration",
            "Body geometry can only be derived from the controlled three-anchor contract.",
        )
    indices = {
        role: labels.index(label) for role, label in expected_config.items()
    }
    leading = source_values.shape[0]
    expected_origin = np.full((leading, 2), np.nan, dtype=output_dtype)
    expected_forward = np.full((leading, 2), np.nan, dtype=output_dtype)
    expected_left = np.full((leading, 2), np.nan, dtype=output_dtype)
    expected_valid = np.zeros((leading,), dtype=np.bool_)

    if method in {"keypoint_head_axis", "mask_component_axis"}:
        anchors = source_values
        validity = support_values["validity"]
        source_row_finite = np.ones((leading,), dtype=np.bool_)
    else:
        anchors = support_values["polarity_anchors"]
        validity = support_values["polarity_valid"]
        source_row_finite = np.all(np.isfinite(source_values), axis=(1, 2))

    eye_left_col = indices["eye_left"]
    eye_right_col = indices["eye_right"]
    posterior_col = indices["posterior_anchor"]
    positive_y = estimator_source.source_descriptor.positive_y
    if positive_y not in {"down", "up"}:
        _fail(
            "source_axis_polarity_unsupported",
            "source_descriptor.positive_y",
            "Controlled body estimators require an explicit source Y direction.",
        )

    for row in range(leading):
        required_valid = bool(
            validity[row, eye_left_col]
            and validity[row, eye_right_col]
            and validity[row, posterior_col]
        )
        row_anchors = np.asarray(
            anchors[
                row,
                (eye_left_col, eye_right_col, posterior_col),
                :,
            ],
            dtype=np.float64,
        )
        if (
            not required_valid
            or not source_row_finite[row]
            or not np.all(np.isfinite(row_anchors))
        ):
            continue
        eye_left, eye_right, posterior_anchor = row_anchors
        origin = (eye_left + eye_right) / 2.0

        if method == "body_spline_with_anchor_polarity":
            first = np.asarray(source_values[row, 0], dtype=np.float64)
            last = np.asarray(source_values[row, -1], dtype=np.float64)
            first_distance = float(np.linalg.norm(first - origin))
            last_distance = float(np.linalg.norm(last - origin))
            if not np.isfinite(first_distance + last_distance) or first_distance == last_distance:
                continue
            anterior, posterior = (
                (first, last)
                if first_distance < last_distance
                else (last, first)
            )
            direction = anterior - posterior
        else:
            direction = origin - posterior_anchor

        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 0.0:
            continue
        forward = direction / norm
        left = (
            np.asarray([forward[1], -forward[0]], dtype=np.float64)
            if positive_y == "down"
            else np.asarray([-forward[1], forward[0]], dtype=np.float64)
        )
        # Labelled eye polarity and posterior-anchor polarity are both strict.
        if float(np.dot(eye_left - eye_right, left)) <= 0.0:
            continue
        if method == "body_spline_with_anchor_polarity" and float(
            np.dot(origin - posterior_anchor, forward)
        ) <= 0.0:
            continue
        expected_valid[row] = True
        expected_origin[row] = origin.astype(output_dtype)
        expected_forward[row] = forward.astype(output_dtype)
        expected_left[row] = left.astype(output_dtype)

    return expected_origin, expected_forward, expected_left, expected_valid


@dataclass(frozen=True, init=False)
class BoundBodyFrameGeometry:
    record: BodyFrameGeometryRecord
    row_identity: BoundRowIdentityContract
    estimator_source: BoundBodyEstimatorSourceBundle
    source_descriptor: BoundBodySourceCoordinateDescriptor
    archive_identity: ArchiveIdentity = field(repr=False, compare=False)
    _frame_node: Any = field(repr=False, compare=False)
    _origin_node: Any = field(repr=False, compare=False)
    _forward_node: Any = field(repr=False, compare=False)
    _left_node: Any = field(repr=False, compare=False)
    _valid_node: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        record: BodyFrameGeometryRecord,
        row_identity: BoundRowIdentityContract,
        estimator_source: BoundBodyEstimatorSourceBundle,
        archive_identity: ArchiveIdentity,
        frame_node: Any,
        origin_node: Any,
        forward_node: Any,
        left_node: Any,
        valid_node: Any,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BODY_GEOMETRY_SEAL:
            _fail(
                "body_geometry_unsealed",
                "$",
                "Body geometry must come from its exact array binder.",
            )
        source_descriptor = estimator_source.source_descriptor
        for name, item in locals().copy().items():
            if name not in {
                "self",
                "frame_node",
                "origin_node",
                "forward_node",
                "left_node",
                "valid_node",
                "_verification_seal",
            }:
                object.__setattr__(self, name, item)
        object.__setattr__(self, "_frame_node", frame_node)
        object.__setattr__(self, "_origin_node", origin_node)
        object.__setattr__(self, "_forward_node", forward_node)
        object.__setattr__(self, "_left_node", left_node)
        object.__setattr__(self, "_valid_node", valid_node)
        object.__setattr__(self, "_seal", _verification_seal)


def bind_body_frame_geometry(
    frame_node: Any,
    *,
    origin_xy_node: Any,
    forward_axis_xy_node: Any,
    left_axis_xy_node: Any,
    axis_valid_node: Any,
    row_identity: BoundRowIdentityContract,
    estimator_source: BoundBodyEstimatorSourceBundle,
) -> BoundBodyFrameGeometry:
    """Hash and validate exact body-frame geometry arrays."""

    row_identity = _require_row_identity(row_identity, body_frame=True)
    estimator_source = verify_bound_body_estimator_source(estimator_source)
    source = estimator_source.source_descriptor
    if (
        source.row_identity.record_ref != row_identity.record_ref
        or source.row_identity.record_sha256 != row_identity.record_sha256
    ):
        _fail(
            "row_identity_binding_mismatch",
            "row_identity",
            "Body geometry must use the source descriptor's exact row identity, not an unrelated same-length identity.",
        )
    nodes = (
        frame_node,
        origin_xy_node,
        forward_axis_xy_node,
        left_axis_xy_node,
        axis_valid_node,
        row_identity._rowset_node,
        row_identity._key_array_node,
        source._coordinate_node,
    )
    identity = _archive_for_nodes(*nodes)
    if identity != row_identity.archive_identity or identity != source.archive_identity:
        _fail(
            "archive_mismatch",
            "$",
            "Body geometry authorities do not share one archive/store.",
        )
    frame_path = _node_path(frame_node)
    expected_paths = {
        "origin_xy": f"{frame_path}/origin_xy",
        "forward_axis_xy": f"{frame_path}/forward_axis_xy",
        "left_axis_xy": f"{frame_path}/left_axis_xy",
        "axis_valid": f"{frame_path}/axis_valid",
    }
    actual_paths = {
        "origin_xy": _node_path(origin_xy_node),
        "forward_axis_xy": _node_path(forward_axis_xy_node),
        "left_axis_xy": _node_path(left_axis_xy_node),
        "axis_valid": _node_path(axis_valid_node),
    }
    if actual_paths != expected_paths or len(set(actual_paths.values())) != 4:
        _fail(
            "geometry_array_path_mismatch",
            "geometry",
            "Geometry arrays must be distinct exact children of the frame node.",
        )
    leading = row_identity.leading_dimension
    snapshots = {
        "origin_xy": _read_array_snapshot(origin_xy_node),
        "forward_axis_xy": _read_array_snapshot(forward_axis_xy_node),
        "left_axis_xy": _read_array_snapshot(left_axis_xy_node),
        "axis_valid": _read_array_snapshot(axis_valid_node),
    }
    support_snapshots = {
        name: _read_array_snapshot(node)
        for name, node in estimator_source._support_nodes.items()
    }
    source_snapshot = _read_array_snapshot(source._coordinate_node)
    if (
        source_snapshot.dtype != source.source_payload.dtype
        or source_snapshot.shape != source.source_payload.shape
        or source_snapshot.content_sha256 != source.source_payload.content_sha256
    ):
        _fail(
            "estimator_source_payload_mismatch",
            "source_descriptor",
            "Live estimator source values differ from the digest-bound source payload.",
        )
    origin = snapshots["origin_xy"].values
    forward = snapshots["forward_axis_xy"].values
    left = snapshots["left_axis_xy"].values
    valid = snapshots["axis_valid"].values
    for name, values in (
        ("origin_xy", origin),
        ("forward_axis_xy", forward),
        ("left_axis_xy", left),
    ):
        if values.shape != (leading, 2):
            _fail(
                "geometry_shape_mismatch",
                f"geometry.{name}",
                f"Expected shape {(leading, 2)!r}, found {values.shape!r}.",
            )
    if valid.shape != (leading,):
        _fail(
            "geometry_shape_mismatch",
            "geometry.axis_valid",
            f"Expected shape {(leading,)!r}, found {valid.shape!r}.",
        )
    _validate_axis_geometry_values(
        origin=origin,
        forward=forward,
        left=left,
        valid=valid,
        positive_y=source.positive_y,
    )
    expected_origin, expected_forward, expected_left, expected_valid = (
        _derive_body_geometry_from_exact_source(
            estimator_source=estimator_source,
            source_values=source_snapshot.values,
            support_values={
                name: snapshot.values for name, snapshot in support_snapshots.items()
            },
            output_dtype=origin.dtype,
        )
    )
    if not np.array_equal(valid, expected_valid):
        _fail(
            "estimator_source_validity_mismatch",
            "geometry.axis_valid",
            "axis_valid must exactly equal validity rederived from the selected formula, support validity, finiteness, and degeneracy rules.",
        )
    for name, actual, expected in (
        ("origin_xy", origin, expected_origin),
        ("forward_axis_xy", forward, expected_forward),
        ("left_axis_xy", left, expected_left),
    ):
        if not np.array_equal(actual, expected, equal_nan=True):
            _fail(
                "estimator_formula_output_mismatch",
                f"geometry.{name}",
                "Persisted geometry does not exactly equal the controlled formula rederived from its digest-bound inputs.",
            )
    for node, name in (
        (origin_xy_node, "origin_xy"),
        (forward_axis_xy_node, "forward_axis_xy"),
        (left_axis_xy_node, "left_axis_xy"),
        (axis_valid_node, "axis_valid"),
    ):
        _recheck_array_snapshot(node, snapshots[name])
    for name, node in estimator_source._support_nodes.items():
        _recheck_array_snapshot(node, support_snapshots[name])
    _recheck_array_snapshot(source._coordinate_node, source_snapshot)
    record = BodyFrameGeometryRecord(
        origin_xy=_geometry_array_record_from_snapshot(
            origin_xy_node, snapshots["origin_xy"]
        ),
        forward_axis_xy=_geometry_array_record_from_snapshot(
            forward_axis_xy_node, snapshots["forward_axis_xy"]
        ),
        left_axis_xy=_geometry_array_record_from_snapshot(
            left_axis_xy_node, snapshots["left_axis_xy"]
        ),
        axis_valid=_geometry_array_record_from_snapshot(
            axis_valid_node, snapshots["axis_valid"]
        ),
    )
    return BoundBodyFrameGeometry(
        record=record,
        row_identity=row_identity,
        estimator_source=estimator_source,
        archive_identity=identity,
        frame_node=frame_node,
        origin_node=origin_xy_node,
        forward_node=forward_axis_xy_node,
        left_node=left_axis_xy_node,
        valid_node=axis_valid_node,
        _verification_seal=_BODY_GEOMETRY_SEAL,
    )


def verify_bound_body_frame_geometry(
    value: BoundBodyFrameGeometry,
) -> BoundBodyFrameGeometry:
    if (
        type(value) is not BoundBodyFrameGeometry
        or value._seal is not _BODY_GEOMETRY_SEAL
    ):
        _fail(
            "body_geometry_unsealed",
            "$",
            "A sealed body-frame geometry binding is required.",
        )
    current = bind_body_frame_geometry(
        value._frame_node,
        origin_xy_node=value._origin_node,
        forward_axis_xy_node=value._forward_node,
        left_axis_xy_node=value._left_node,
        axis_valid_node=value._valid_node,
        row_identity=value.row_identity,
        estimator_source=value.estimator_source,
    )
    if current.record != value.record:
        _fail(
            "body_geometry_stale",
            "$",
            "Body geometry payload or metadata changed after binding.",
        )
    if current.archive_identity != value.archive_identity:
        _fail(
            "archive_mismatch", "$", "Body geometry moved to a different archive/store."
        )
    return value


@dataclass(frozen=True)
class BodyFrameRowIdentityRecord:
    record_ref: str
    record_sha256: str
    leading_dimension: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_ref": self.record_ref,
            "record_sha256": self.record_sha256,
            "leading_dimension": self.leading_dimension,
        }


def _parse_body_row_identity(value: Any, *, path: str) -> BodyFrameRowIdentityRecord:
    payload = _mapping(value, path=path)
    _exact_fields(
        payload, {"record_ref", "record_sha256", "leading_dimension"}, path=path
    )
    record_ref = _canonical_record_ref(payload["record_ref"], path=f"{path}.record_ref")
    if not record_ref.endswith("@row_identity_contract"):
        _fail(
            "row_identity_ref_invalid",
            f"{path}.record_ref",
            "Exact row_identity_contract attr is required.",
        )
    return BodyFrameRowIdentityRecord(
        record_ref,
        _digest(payload["record_sha256"], path=f"{path}.record_sha256"),
        _exact_int(payload["leading_dimension"], path=f"{path}.leading_dimension"),
    )


@dataclass(frozen=True)
class FishAnatomicalBodyFrameRecord:
    frame_id: str
    coordinate_units: str
    origin_definition: str
    source_descriptor: DigestBoundFrameRecordRef
    source_coordinate_payload: BodyGeometryArrayRecord
    estimator_source: BodyEstimatorSourceBundleRecord
    source_profile_id: str
    body_frame_contract: DigestBoundFrameRecordRef
    estimator: DigestBoundFrameRecordRef
    geometry: BodyFrameGeometryRecord
    row_identity: BodyFrameRowIdentityRecord

    @property
    def kind(self) -> str:
        return FISH_ANATOMICAL_BODY_FRAME_KIND

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": FISH_ANATOMICAL_BODY_FRAME_SCHEMA_ID,
            "schema_version": 1,
            "kind": self.kind,
            "frame_id": self.frame_id,
            "coordinate_units": self.coordinate_units,
            "origin": "body_frame_origin",
            "origin_definition": self.origin_definition,
            "axes": BodyFrameAxes().to_dict(),
            "angle_convention": BODY_ANGLE_CONVENTION,
            "source_descriptor": self.source_descriptor.to_dict(),
            "source_coordinate_payload": self.source_coordinate_payload.to_dict(),
            "estimator_source": self.estimator_source.to_dict(),
            "source_profile_id": self.source_profile_id,
            "body_frame_contract": self.body_frame_contract.to_dict(),
            "estimator": self.estimator.to_dict(),
            "geometry": self.geometry.to_dict(),
            "row_identity": self.row_identity.to_dict(),
            "extent": {
                "mode": REFERENCE_EXTENT_NOT_APPLICABLE,
                "width": None,
                "height": None,
                "units": "not_applicable",
            },
        }

    def canonical_json(self) -> str:
        return fish_anatomical_body_frame_record_json(self)

    def digest(self) -> str:
        return fish_anatomical_body_frame_record_sha256(self)


def parse_fish_anatomical_body_frame_record(
    value: Any,
) -> FishAnatomicalBodyFrameRecord:
    """Strictly parse a materialized, identity-bound anatomical frame."""

    if isinstance(value, FishAnatomicalBodyFrameRecord):
        value = value.to_dict()
    payload = _mapping(value, path="$")
    _exact_fields(
        payload,
        {
            "schema_id",
            "schema_version",
            "kind",
            "frame_id",
            "coordinate_units",
            "origin",
            "origin_definition",
            "axes",
            "angle_convention",
            "source_descriptor",
            "source_coordinate_payload",
            "estimator_source",
            "source_profile_id",
            "body_frame_contract",
            "estimator",
            "geometry",
            "row_identity",
            "extent",
        },
        path="$",
    )
    if (
        payload["schema_id"] != FISH_ANATOMICAL_BODY_FRAME_SCHEMA_ID
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
    ):
        _fail("schema_invalid", "$", "Unsupported fish-anatomical frame schema.")
    if payload["kind"] != FISH_ANATOMICAL_BODY_FRAME_KIND:
        _fail("kind_invalid", "$.kind", "Record is not a fish-anatomical frame.")
    if payload["origin"] != "body_frame_origin":
        _fail(
            "origin_invalid",
            "$.origin",
            "Canonical descriptor-facing origin is body_frame_origin.",
        )
    origin_definition = payload["origin_definition"]
    if origin_definition != "eye_pair_midpoint":
        _fail(
            "origin_definition_invalid",
            "$.origin_definition",
            "Controlled body-estimator formulas require the exact eye-pair midpoint origin.",
        )
    _parse_body_axes(payload["axes"], path="$.axes")
    if payload["angle_convention"] != BODY_ANGLE_CONVENTION:
        _fail(
            "angle_convention_invalid",
            "$.angle_convention",
            "Unsupported body angle convention.",
        )
    source_profile_id = payload["source_profile_id"]
    if source_profile_id not in SUPPORTED_BODY_SOURCE_PROFILE_IDS:
        _fail(
            "body_source_profile_unsupported",
            "$.source_profile_id",
            "Unsupported body source profile.",
        )
    expected_units = CANONICAL_COORDINATE_PROFILES[source_profile_id].coordinate_unit
    if payload["coordinate_units"] != expected_units:
        _fail(
            "coordinate_units_mismatch",
            "$.coordinate_units",
            "Body units must exactly match source descriptor units.",
        )
    extent = _parse_frame_extent(payload["extent"], path="$.extent")
    if extent.mode != REFERENCE_EXTENT_NOT_APPLICABLE:
        _fail(
            "body_extent_invalid",
            "$.extent",
            "Body-frame extent is explicitly not applicable.",
        )
    return FishAnatomicalBodyFrameRecord(
        frame_id=_text(payload["frame_id"], path="$.frame_id", pattern=_FRAME_ID_RE),
        coordinate_units=expected_units,
        origin_definition=origin_definition,
        source_descriptor=_parse_record_ref(
            payload["source_descriptor"], path="$.source_descriptor"
        ),
        source_coordinate_payload=_parse_geometry_array_record(
            payload["source_coordinate_payload"],
            path="$.source_coordinate_payload",
        ),
        estimator_source=_parse_body_estimator_source_bundle_record(
            payload["estimator_source"]
        ),
        source_profile_id=source_profile_id,
        body_frame_contract=_parse_record_ref(
            payload["body_frame_contract"], path="$.body_frame_contract"
        ),
        estimator=_parse_record_ref(payload["estimator"], path="$.estimator"),
        geometry=_parse_body_geometry_record(payload["geometry"]),
        row_identity=_parse_body_row_identity(
            payload["row_identity"], path="$.row_identity"
        ),
    )


def fish_anatomical_body_frame_record_json(value: Any) -> str:
    return _canonical_json(parse_fish_anatomical_body_frame_record(value).to_dict())


def fish_anatomical_body_frame_record_sha256(value: Any) -> str:
    return hashlib.sha256(
        fish_anatomical_body_frame_record_json(value).encode("utf-8")
    ).hexdigest()


def _require_body_dependency_graph(
    *,
    frame_node: Any,
    contract: BoundBodyFrameContract,
    estimator_source: BoundBodyEstimatorSourceBundle,
    geometry: BoundBodyFrameGeometry,
    row_identity: BoundRowIdentityContract,
) -> ArchiveIdentity:
    """Reject self-source, cycles, and cross-role node aliases.

    Geometry arrays intentionally live below the frame group, but every role
    is a distinct persisted node.  The source's complete sealed dependency set
    is also included, so a lineage/calibration node cannot double as a body
    contract, estimator, geometry array, identity node, or output frame.
    """

    source = estimator_source.source_descriptor
    estimator = estimator_source.estimator
    roles: dict[str, Any] = {
        "frame": frame_node,
        "source_coordinate": source._coordinate_node,
        "estimator_source_schema": estimator_source.source_schema._node,
        "body_contract": contract._node,
        "estimator": estimator._node,
        "geometry_origin": geometry._origin_node,
        "geometry_forward": geometry._forward_node,
        "geometry_left": geometry._left_node,
        "geometry_valid": geometry._valid_node,
        "rowset": row_identity._rowset_node,
        "row_key": row_identity._key_array_node,
    }
    for name, node in estimator_source._support_nodes.items():
        roles[f"estimator_support_{name}"] = node
    paths = {name: _node_path(node) for name, node in roles.items()}
    by_path: dict[str, list[str]] = {}
    for name, path in paths.items():
        by_path.setdefault(path, []).append(name)
    aliases = {path: names for path, names in by_path.items() if len(names) > 1}
    if aliases:
        if "frame" in {name for names in aliases.values() for name in names}:
            _fail(
                "dependency_cycle",
                "$",
                f"Body output frame aliases one of its dependencies: {aliases!r}.",
            )
        _fail(
            "dependency_alias",
            "$",
            f"Distinct body authority roles alias persisted nodes: {aliases!r}.",
        )

    source_core_paths = {
        paths["source_coordinate"],
        paths["rowset"],
        paths["row_key"],
    }
    collection = source.descriptor.collection_axis
    shared_mask_schema_path = (
        paths["estimator_source_schema"]
        if (
            estimator_source.record.method == "mask_component_axis"
            and collection is not None
            and collection.role == "subject_component"
            and collection.label_authority.record_ref
            == estimator_source.source_schema.record_ref
            and collection.label_authority.record_sha256
            == estimator_source.source_schema.record_sha256
        )
        else None
    )
    occupied_non_source = {
        path
        for role, path in paths.items()
        if role not in {"source_coordinate", "rowset", "row_key"}
    }
    extra_source_nodes: list[Any] = []
    for node in source.dependency_nodes:
        path = _node_path(node)
        # A collected mask-component point surface deliberately uses the exact
        # estimator component schema as its collection-label authority.  This
        # is one authority serving one semantic role, not a cross-role alias or
        # cycle; the schema node is already present in `roles`.
        if path == shared_mask_schema_path:
            continue
        if path in occupied_non_source:
            code = "dependency_cycle" if path == paths["frame"] else "dependency_alias"
            _fail(
                code,
                "$",
                "A source coordinate/lineage authority aliases a body output, "
                "geometry, contract, or estimator node.",
            )
        if path not in source_core_paths:
            extra_source_nodes.append(node)

    all_nodes = tuple(roles.values()) + tuple(extra_source_nodes)
    identity = _archive_for_nodes(*all_nodes)
    if any(
        item != identity
        for item in (
            source.archive_identity,
            contract.archive_identity,
            estimator.archive_identity,
            estimator_source.archive_identity,
            geometry.archive_identity,
            row_identity.archive_identity,
        )
    ):
        _fail(
            "archive_mismatch",
            "$",
            "Body-frame dependency graph does not retain one archive/store.",
        )
    return identity


def build_fish_anatomical_body_frame_record(
    *,
    frame_id: str,
    origin_definition: str,
    body_frame_contract: BoundBodyFrameContract,
    estimator_source: BoundBodyEstimatorSourceBundle,
    geometry: BoundBodyFrameGeometry,
    row_identity: BoundRowIdentityContract,
) -> FishAnatomicalBodyFrameRecord:
    """Build a body record solely from exact typed persisted authorities."""

    contract = verify_bound_body_frame_contract(body_frame_contract)
    estimator_source = verify_bound_body_estimator_source(estimator_source)
    source = estimator_source.source_descriptor
    estimator_bound = estimator_source.estimator
    geometry_bound = verify_bound_body_frame_geometry(geometry)
    row_identity = _require_row_identity(row_identity, body_frame=True)
    if (
        geometry_bound.estimator_source.record != estimator_source.record
        or geometry_bound.estimator_source.archive_identity
        != estimator_source.archive_identity
    ):
        _fail(
            "estimator_source_binding_mismatch",
            "geometry",
            "Geometry was not validated against the exact typed estimator source.",
        )
    if (
        geometry_bound.row_identity.record_ref != row_identity.record_ref
        or geometry_bound.row_identity.record_sha256 != row_identity.record_sha256
        or source.row_identity.record_ref != row_identity.record_ref
        or source.row_identity.record_sha256 != row_identity.record_sha256
    ):
        _fail(
            "row_identity_binding_mismatch",
            "row_identity",
            "Body authorities require the same exact row identity contract.",
        )
    _require_body_dependency_graph(
        frame_node=geometry_bound._frame_node,
        contract=contract,
        estimator_source=estimator_source,
        geometry=geometry_bound,
        row_identity=row_identity,
    )
    return parse_fish_anatomical_body_frame_record(
        {
            "schema_id": FISH_ANATOMICAL_BODY_FRAME_SCHEMA_ID,
            "schema_version": 1,
            "kind": FISH_ANATOMICAL_BODY_FRAME_KIND,
            "frame_id": frame_id,
            "coordinate_units": source.coordinate_units,
            "origin": "body_frame_origin",
            "origin_definition": origin_definition,
            "axes": contract.record.axes.to_dict(),
            "angle_convention": BODY_ANGLE_CONVENTION,
            "source_descriptor": source.ref.to_dict(),
            "source_coordinate_payload": source.source_payload.to_dict(),
            "estimator_source": estimator_source.record.to_dict(),
            "source_profile_id": source.descriptor.profile_id,
            "body_frame_contract": contract.ref.to_dict(),
            "estimator": estimator_bound.ref.to_dict(),
            "geometry": geometry_bound.record.to_dict(),
            "row_identity": {
                "record_ref": row_identity.record_ref,
                "record_sha256": row_identity.record_sha256,
                "leading_dimension": row_identity.leading_dimension,
            },
            "extent": {
                "mode": REFERENCE_EXTENT_NOT_APPLICABLE,
                "width": None,
                "height": None,
                "units": "not_applicable",
            },
        }
    )


@dataclass(frozen=True, init=False)
class BoundFishAnatomicalBodyFrame:
    kind: str
    record_ref: str
    record_sha256: str
    selector: str
    reference_width: None
    reference_height: None
    reference_units: str
    extent_mode: str
    coordinate_units: str
    origin: str
    origin_definition: str
    positive_x: str
    positive_y: str
    record: FishAnatomicalBodyFrameRecord
    estimator_source: BoundBodyEstimatorSourceBundle
    source_descriptor: BoundBodySourceCoordinateDescriptor
    body_frame_contract: BoundBodyFrameContract
    estimator: BoundBodyFrameEstimator
    geometry: BoundBodyFrameGeometry
    row_identity: BoundRowIdentityContract
    archive_identity: ArchiveIdentity = field(repr=False, compare=False)
    _node: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        record_ref: str,
        record_sha256: str,
        record: FishAnatomicalBodyFrameRecord,
        body_frame_contract: BoundBodyFrameContract,
        estimator_source: BoundBodyEstimatorSourceBundle,
        geometry: BoundBodyFrameGeometry,
        row_identity: BoundRowIdentityContract,
        archive_identity: ArchiveIdentity,
        node: Any,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _FRAME_SEAL:
            _fail("frame_unsealed", "$", "Body frame must come from its stamp/loader.")
        source_descriptor = estimator_source.source_descriptor
        estimator = estimator_source.estimator
        values = {
            "kind": FISH_ANATOMICAL_BODY_FRAME_KIND,
            "record_ref": record_ref,
            "record_sha256": record_sha256,
            "selector": FRAME_RECORD_SELECTOR,
            "reference_width": None,
            "reference_height": None,
            "reference_units": "not_applicable",
            "extent_mode": REFERENCE_EXTENT_NOT_APPLICABLE,
            "coordinate_units": record.coordinate_units,
            "origin": "body_frame_origin",
            "origin_definition": record.origin_definition,
            "positive_x": BODY_POSITIVE_X,
            "positive_y": BODY_POSITIVE_Y,
            "record": record,
            "estimator_source": estimator_source,
            "source_descriptor": source_descriptor,
            "body_frame_contract": body_frame_contract,
            "estimator": estimator,
            "geometry": geometry,
            "row_identity": row_identity,
            "archive_identity": archive_identity,
            "_node": node,
            "_seal": _verification_seal,
        }
        for name, item in values.items():
            object.__setattr__(self, name, item)


def _validate_body_authorities(
    record: FishAnatomicalBodyFrameRecord,
    *,
    frame_node: Any,
    body_frame_contract: BoundBodyFrameContract,
    estimator_source: BoundBodyEstimatorSourceBundle,
    geometry: BoundBodyFrameGeometry,
    row_identity: BoundRowIdentityContract,
) -> ArchiveIdentity:
    contract = verify_bound_body_frame_contract(body_frame_contract)
    estimator_source = verify_bound_body_estimator_source(estimator_source)
    source = estimator_source.source_descriptor
    estimator_bound = estimator_source.estimator
    geometry_bound = verify_bound_body_frame_geometry(geometry)
    row_identity = _require_row_identity(row_identity, body_frame=True)
    if _node_path(frame_node) != _node_path(geometry_bound._frame_node):
        _fail(
            "geometry_frame_node_mismatch",
            "geometry",
            "Geometry belongs to a different frame node.",
        )
    frame_path = _node_path(frame_node)
    identity = _require_body_dependency_graph(
        frame_node=frame_node,
        contract=contract,
        estimator_source=estimator_source,
        geometry=geometry_bound,
        row_identity=row_identity,
    )
    if (
        record.source_descriptor != source.ref
        or record.body_frame_contract != contract.ref
        or record.estimator != estimator_bound.ref
        or record.estimator_source != estimator_source.record
    ):
        _fail(
            "body_authority_mismatch",
            "$",
            "Body record does not bind exact source/contract/estimator records.",
        )
    if record.geometry != geometry_bound.record:
        _fail(
            "body_geometry_binding_mismatch",
            "$.geometry",
            "Body record does not bind exact materialized geometry payloads.",
        )
    if (
        record.row_identity.record_ref != row_identity.record_ref
        or record.row_identity.record_sha256 != row_identity.record_sha256
        or record.row_identity.leading_dimension != row_identity.leading_dimension
    ):
        _fail(
            "row_identity_binding_mismatch",
            "$.row_identity",
            "Body record does not bind exact row identity.",
        )
    if (
        record.source_profile_id != source.descriptor.profile_id
        or record.coordinate_units != source.coordinate_units
    ):
        _fail(
            "source_descriptor_binding_mismatch",
            "$.source_descriptor",
            "Body source semantics differ from exact descriptor.",
        )
    if record.source_coordinate_payload != source.source_payload:
        _fail(
            "source_coordinate_payload_mismatch",
            "$.source_coordinate_payload",
            "Body record does not bind the exact source-coordinate dtype, "
            "shape, and content digest.",
        )
    if (
        record.geometry.origin_xy.array_ref != f"/{frame_path}/origin_xy"
        or record.geometry.forward_axis_xy.array_ref != f"/{frame_path}/forward_axis_xy"
        or record.geometry.left_axis_xy.array_ref != f"/{frame_path}/left_axis_xy"
        or record.geometry.axis_valid.array_ref != f"/{frame_path}/axis_valid"
    ):
        _fail(
            "geometry_array_path_mismatch",
            "$.geometry",
            "Body record does not name exact frame child arrays.",
        )
    return identity


def load_bound_fish_anatomical_body_frame(
    node: Any,
    *,
    expected_record_ref: str,
    expected_record_sha256: str,
    expected_source_profile_id: str,
    expected_coordinate_units: str,
    expected_estimator_method: str,
    body_frame_contract: BoundBodyFrameContract,
    estimator_source: BoundBodyEstimatorSourceBundle,
    geometry: BoundBodyFrameGeometry,
    row_identity: BoundRowIdentityContract,
) -> BoundFishAnatomicalBodyFrame:
    record_ref, record, digest = _load_record_attrs(
        node,
        attr_name=FISH_ANATOMICAL_BODY_FRAME_ATTR,
        parser=parse_fish_anatomical_body_frame_record,
    )
    if record_ref != _canonical_record_ref(
        expected_record_ref, path="expected_record_ref"
    ):
        _fail(
            "record_path_mismatch",
            "expected_record_ref",
            "Body frame path differs from expectation.",
        )
    if digest != _digest(expected_record_sha256, path="expected_record_sha256"):
        _fail(
            "record_digest_mismatch",
            "expected_record_sha256",
            "Body frame digest differs from expectation.",
        )
    identity = _validate_body_authorities(
        record,
        frame_node=node,
        body_frame_contract=body_frame_contract,
        estimator_source=estimator_source,
        geometry=geometry,
        row_identity=row_identity,
    )
    if (
        record.source_profile_id != expected_source_profile_id
        or record.coordinate_units != expected_coordinate_units
    ):
        _fail(
            "source_descriptor_binding_mismatch",
            "expected_source_profile_id",
            "Body source semantics differ from consumer expectation.",
        )
    if estimator_source.estimator.record.method != expected_estimator_method:
        _fail(
            "estimator_method_mismatch",
            "expected_estimator_method",
            "Body estimator differs from consumer expectation.",
        )
    return BoundFishAnatomicalBodyFrame(
        record_ref=record_ref,
        record_sha256=digest,
        record=record,
        body_frame_contract=body_frame_contract,
        estimator_source=estimator_source,
        geometry=geometry,
        row_identity=row_identity,
        archive_identity=identity,
        node=node,
        _verification_seal=_FRAME_SEAL,
    )


def stamp_fish_anatomical_body_frame_record(
    node: Any,
    record: Any,
    *,
    expected_record_ref: str,
    body_frame_contract: BoundBodyFrameContract,
    estimator_source: BoundBodyEstimatorSourceBundle,
    geometry: BoundBodyFrameGeometry,
    row_identity: BoundRowIdentityContract,
) -> BoundFishAnatomicalBodyFrame:
    """Transactionally write, reload, and fully verify a body frame."""

    parsed = parse_fish_anatomical_body_frame_record(record)
    actual_ref = _node_record_ref(node, FISH_ANATOMICAL_BODY_FRAME_ATTR)
    if actual_ref != _canonical_record_ref(
        expected_record_ref, path="expected_record_ref"
    ):
        _fail(
            "record_path_mismatch",
            "expected_record_ref",
            "Body frame node differs from requested path.",
        )
    _validate_body_authorities(
        parsed,
        frame_node=node,
        body_frame_contract=body_frame_contract,
        estimator_source=estimator_source,
        geometry=geometry,
        row_identity=row_identity,
    )
    return _transactional_stamp(
        node,
        attr_name=FISH_ANATOMICAL_BODY_FRAME_ATTR,
        payload=parsed.to_dict(),
        reload_and_verify=lambda: load_bound_fish_anatomical_body_frame(
            node,
            expected_record_ref=actual_ref,
            expected_record_sha256=parsed.digest(),
            expected_source_profile_id=parsed.source_profile_id,
            expected_coordinate_units=parsed.coordinate_units,
            expected_estimator_method=estimator_source.estimator.record.method,
            body_frame_contract=body_frame_contract,
            estimator_source=estimator_source,
            geometry=geometry,
            row_identity=row_identity,
        ),
    )


BoundCoordinateFrame = BoundPhysicalFrameCalibration | BoundFishAnatomicalBodyFrame


def verify_bound_coordinate_frame(
    value: BoundCoordinateFrame,
    *,
    expected_kind: str,
) -> BoundCoordinateFrame:
    """Re-read the frame record and every retained typed authority."""

    if expected_kind not in COORDINATE_FRAME_RECORD_KINDS:
        _fail("kind_invalid", "expected_kind", "Unsupported frame kind expectation.")
    if type(value) is BoundPhysicalFrameCalibration:
        if (
            value._seal is not _FRAME_SEAL
            or expected_kind != PHYSICAL_FRAME_CALIBRATION_KIND
        ):
            _fail(
                "frame_unsealed",
                "$",
                "Physical frame binding is forged or kind-mismatched.",
            )
        current = load_bound_physical_frame_calibration(
            value._node,
            expected_record_ref=value.record_ref,
            expected_record_sha256=value.record_sha256,
            expected_camera_id=value.record.camera_id,
            source_camera_pixels=value.source_camera_pixels,
            selected_camera_evidence=value.selected_camera_evidence,
        )
        if (
            current.record != value.record
            or current.archive_identity != value.archive_identity
        ):
            _fail(
                "frame_record_stale",
                value.record_ref,
                "Physical frame or its archive identity changed after binding.",
            )
        return value
    if type(value) is BoundFishAnatomicalBodyFrame:
        if (
            value._seal is not _FRAME_SEAL
            or expected_kind != FISH_ANATOMICAL_BODY_FRAME_KIND
        ):
            _fail(
                "frame_unsealed",
                "$",
                "Body frame binding is forged or kind-mismatched.",
            )
        current = load_bound_fish_anatomical_body_frame(
            value._node,
            expected_record_ref=value.record_ref,
            expected_record_sha256=value.record_sha256,
            expected_source_profile_id=value.record.source_profile_id,
            expected_coordinate_units=value.record.coordinate_units,
            expected_estimator_method=value.estimator.record.method,
            body_frame_contract=value.body_frame_contract,
            estimator_source=value.estimator_source,
            geometry=value.geometry,
            row_identity=value.row_identity,
        )
        if (
            current.record != value.record
            or current.archive_identity != value.archive_identity
        ):
            _fail(
                "frame_record_stale",
                value.record_ref,
                "Body frame or its archive identity changed after binding.",
            )
        return value
    _fail("frame_unsealed", "$", "Canonical frames must come from typed bound loaders.")
    raise AssertionError("unreachable")


__all__ = [
    "ARRAY_PAYLOAD_CANONICALIZATION",
    "BODY_ANGLE_CONVENTION",
    "BODY_ESTIMATOR_SOURCE_BUNDLE_SCHEMA_ID",
    "BODY_ESTIMATOR_SOURCE_MANIFEST_ATTR",
    "BODY_ESTIMATOR_SOURCE_MANIFEST_SCHEMA_ID",
    "BODY_ESTIMATOR_METHODS",
    "BODY_FRAME_CONTRACT_ATTR",
    "BODY_FRAME_CONTRACT_SCHEMA_ID",
    "BODY_FRAME_ESTIMATOR_ATTR",
    "BODY_FRAME_ESTIMATOR_SCHEMA_ID",
    "BODY_INVALID_ROW_ENCODING",
    "BoundBodyFrameContract",
    "BoundBodyFrameEstimator",
    "BoundBodyFrameGeometry",
    "BoundBodyEstimatorSourceBundle",
    "BoundBodySourceCoordinateDescriptor",
    "BoundCoordinateFrame",
    "BoundFishAnatomicalBodyFrame",
    "BoundPhysicalFrameCalibration",
    "BoundSelectedCameraFrameEvidence",
    "COORDINATE_FRAME_RECORD_CANONICALIZATION",
    "COORDINATE_FRAME_RECORD_KINDS",
    "COORDINATE_FRAME_RECORD_SCHEMA_VERSION",
    "CAMERA_SCALE_QUANTITY",
    "CoordinateFrameRecordError",
    "CoordinateFrameRecordIssue",
    "DigestBoundFrameRecordRef",
    "FISH_ANATOMICAL_BODY_FRAME_ATTR",
    "FISH_ANATOMICAL_BODY_FRAME_KIND",
    "FISH_ANATOMICAL_BODY_FRAME_SCHEMA_ID",
    "FRAME_RECORD_SELECTOR",
    "FishAnatomicalBodyFrameRecord",
    "MM_PER_PIXEL",
    "PHYSICAL_SOURCE_CAMERA_PROFILE_ID",
    "PHYSICAL_FRAME_CALIBRATION_ATTR",
    "PHYSICAL_FRAME_CALIBRATION_KIND",
    "PHYSICAL_FRAME_CALIBRATION_SCHEMA_ID",
    "PIXELS_PER_MM",
    "PhysicalFrameCalibrationRecord",
    "REFERENCE_EXTENT_FINITE",
    "REFERENCE_EXTENT_NOT_APPLICABLE",
    "REFERENCE_EXTENT_UNBOUNDED",
    "SELECTED_CAMERA_FRAME_EVIDENCE_ATTR",
    "SELECTED_CAMERA_FRAME_EVIDENCE_SCHEMA_ID",
    "SOURCE_CAMERA_PROFILE_ID",
    "array_payload_sha256",
    "array_values_sha256",
    "bind_body_frame_geometry",
    "bind_body_spline_with_anchor_polarity_source",
    "bind_body_source_coordinate_descriptor",
    "bind_keypoint_head_axis_source",
    "bind_mask_component_axis_source",
    "build_body_frame_contract_record",
    "build_body_frame_estimator_record",
    "build_body_estimator_source_manifest_record",
    "build_fish_anatomical_body_frame_record",
    "build_physical_frame_calibration_record",
    "fish_anatomical_body_frame_record_json",
    "fish_anatomical_body_frame_record_sha256",
    "load_bound_body_frame_contract",
    "load_bound_body_frame_estimator",
    "load_bound_fish_anatomical_body_frame",
    "load_bound_physical_frame_calibration",
    "load_bound_selected_camera_frame_evidence",
    "parse_body_frame_contract_record",
    "parse_body_frame_estimator_record",
    "parse_fish_anatomical_body_frame_record",
    "parse_physical_frame_calibration_record",
    "parse_selected_camera_frame_evidence_record",
    "physical_frame_calibration_record_json",
    "physical_frame_calibration_record_sha256",
    "stamp_body_frame_contract",
    "stamp_body_frame_estimator",
    "stamp_fish_anatomical_body_frame_record",
    "stamp_physical_frame_calibration_record",
    "stamp_selected_camera_frame_evidence",
    "verify_bound_body_frame_contract",
    "verify_bound_body_frame_estimator",
    "verify_bound_body_frame_geometry",
    "verify_bound_body_estimator_source",
    "verify_bound_body_source_coordinate_descriptor",
    "verify_bound_coordinate_frame",
    "verify_bound_selected_camera_frame_evidence",
]
