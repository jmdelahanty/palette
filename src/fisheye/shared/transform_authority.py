"""Exact typed authorities for canonical version-2 directed transforms.

Transform endpoints are sealed :mod:`pixel_frame_authority` values, never a
caller-labelled extent. Acquisition owns the source-camera frame;
selected-calibration evidence only authorizes the camera-to-display map.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Mapping

import numpy as np

from fisheye.shared.archive_identity import (
    ArchiveIdentity,
    ArchiveIdentityError,
    archive_identity,
)
from fisheye.shared.coordinate_identity import (
    BoundRowIdentityContract,
    RowIdentityContractError,
)
from fisheye.shared.coordinate_reference import canonical_node_path
from fisheye.shared.pixel_frame_authority import (
    ARENA_RELATIVE_CANVAS_SPACE_ID,
    DETECTOR_NORMALIZED_SPACE_ID,
    DETECTOR_MODEL_INPUT_SPACE_ID,
    NORMALIZED_TO_PIXEL_CENTER_INDEX_V1,
    NORMALIZED_TO_PIXEL_EDGE_EXTENT_V1,
    PROJECTIVE_XY_DIRECT_V1,
    ROI_LOCAL_SPACE_ID,
    SCALE_XY_EDGE_ALIGNED_V1,
    SCALE_XY_PIXEL_CENTER_V1,
    SOURCE_CAMERA_IMAGE_SPACE_ID,
    SOURCE_CAMERA_NORMALIZED_SPACE_ID,
    STIMULUS_CANVAS_SPACE_ID,
    TRANSLATION_XY_DIRECT_V1,
    CROP_PLACEMENT_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PADDED_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PADDED_PIXEL_CENTER_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PADDED_PIXEL_EDGE_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PIXEL_CENTER_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR,
    BoundPixelFrameAuthority,
    PixelFrameAuthorityError,
    PixelFrameEndpoint,
    array_values_sha256,
    model_input_to_roi_matrix,
    normalized_to_pixel_matrix,
    require_arena_relative_canvas_pixel_frame_authority,
    require_bound_pixel_frame_authority,
    require_bound_crop_placement_ownership,
    require_model_input_pixel_frame_authority,
    require_normalized_pixel_frame_authority,
    require_roi_pixel_frame_authority,
    require_selected_canvas_pixel_frame_authority,
    require_source_camera_pixel_frame_authority,
    require_trusted_coordinate_attrs,
)


TRANSFORM_AUTHORITY_SCHEMA_ID = "palette.transform_authority"
TRANSFORM_AUTHORITY_SCHEMA_VERSION = 2
TRANSFORM_AUTHORITY_ATTR = "transform_authority"
TRANSFORM_AUTHORITY_DIGEST_ATTR = "transform_authority_sha256"
TRANSFORM_AUTHORITY_PIXEL_CENTER_ATTR = "transform_authority_pixel_center"
TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR = "transform_authority_pixel_edge_half_open"
TRANSFORM_AUTHORITY_ATTRS = frozenset(
    {
        TRANSFORM_AUTHORITY_ATTR,
        TRANSFORM_AUTHORITY_PIXEL_CENTER_ATTR,
        TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
    }
)


def _require_crop_ownership_attr_pair(
    record: "TransformAuthorityRecord",
    source_frame: BoundPixelFrameAuthority,
    *,
    attr_name: str,
) -> None:
    if record.kind != CROP_PLACEMENT_AUTHORITY_KIND:
        return
    source = require_roi_pixel_frame_authority(source_frame)
    ownership = require_bound_crop_placement_ownership(
        source._context.get("crop_placement_ownership")
    )
    expected = {
        TRANSFORM_AUTHORITY_ATTR: {
            CROP_PLACEMENT_OWNERSHIP_ATTR,
            CROP_PLACEMENT_PADDED_OWNERSHIP_ATTR,
        },
        TRANSFORM_AUTHORITY_PIXEL_CENTER_ATTR: {
            CROP_PLACEMENT_PIXEL_CENTER_OWNERSHIP_ATTR,
            CROP_PLACEMENT_PADDED_PIXEL_CENTER_OWNERSHIP_ATTR,
        },
        TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR: {
            CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR,
            CROP_PLACEMENT_PADDED_PIXEL_EDGE_OWNERSHIP_ATTR,
        },
    }[attr_name]
    if ownership.attr_name not in expected:
        raise TransformAuthorityError(
            "Crop-placement transform authority is cross-wired to the wrong "
            "closed ownership attr."
        )


TRANSFORM_AUTHORITY_CANONICALIZATION = "canonical_json_sort_keys_v1"

SELECTED_CALIBRATION_AUTHORITY_KIND = "selected_calibration"
CROP_PLACEMENT_AUTHORITY_KIND = "crop_placement"
MODEL_INPUT_PREPROCESSING_AUTHORITY_KIND = "model_input_preprocessing"
ARENA_CANVAS_PLACEMENT_AUTHORITY_KIND = "arena_canvas_placement"
NORMALIZED_TO_PIXEL_AUTHORITY_KIND = "normalized_to_pixel"
TRANSFORM_AUTHORITY_KINDS = frozenset(
    {
        SELECTED_CALIBRATION_AUTHORITY_KIND,
        CROP_PLACEMENT_AUTHORITY_KIND,
        MODEL_INPUT_PREPROCESSING_AUTHORITY_KIND,
        ARENA_CANVAS_PLACEMENT_AUTHORITY_KIND,
        NORMALIZED_TO_PIXEL_AUTHORITY_KIND,
    }
)

AuthorityEndpoint = PixelFrameEndpoint

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_AUTHORITY_ID_RE = re.compile(r"^[A-Za-z0-9_.:+-]+$")
_BOUND_AUTHORITY_SEAL = object()


class TransformAuthorityError(ValueError):
    """Raised when persisted transform authority is incomplete or stale."""


def _required_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TransformAuthorityError(
            f"{field_name} must be a non-empty string without surrounding whitespace."
        )
    return value


def _sha256(value: Any, *, field_name: str) -> str:
    text = _required_text(value, field_name=field_name)
    if _SHA256_RE.fullmatch(text) is None:
        raise TransformAuthorityError(
            f"{field_name} must be a lowercase 64-character SHA-256 digest."
        )
    return text


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise TransformAuthorityError(
            "Transform-authority records must contain finite canonical JSON values."
        ) from exc


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _exact_json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (
            type(left) is type(right)
            and set(left) == set(right)
            and all(_exact_json_equal(left[name], right[name]) for name in left)
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _exact_json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def _expected_attrs_after_update(
    snapshot: Mapping[str, Any],
    intended: Mapping[str, Any],
) -> dict[str, Any]:
    expected = copy.deepcopy(dict(snapshot))
    expected.update(copy.deepcopy(dict(intended)))
    return expected


def _require_exact_attrs_state(
    attrs: Any,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if not _exact_json_equal(dict(attrs), dict(expected)):
        raise TransformAuthorityError(
            f"{label} attrs differ from the exact snapshot plus intended payload."
        )


def _restore_exact_attrs(attrs: Any, snapshot: Mapping[str, Any]) -> None:
    for name in tuple(attrs.keys()):
        del attrs[name]
    attrs.update(copy.deepcopy(dict(snapshot)))
    if not _exact_json_equal(dict(attrs), dict(snapshot)):
        raise RuntimeError("restored attrs differ type-strictly from snapshot")


def _exact_fields(
    value: Any, *, expected: frozenset[str], field_name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TransformAuthorityError(f"{field_name} must be a mapping.")
    actual = frozenset(value)
    if actual != expected:
        raise TransformAuthorityError(
            f"{field_name} fields are invalid; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}."
        )
    return value


@dataclass(frozen=True)
class AuthorityPayload:
    record_ref: str
    record_sha256: str
    selector: str

    def to_dict(self) -> dict[str, str]:
        return {
            "record_ref": self.record_ref,
            "record_sha256": self.record_sha256,
            "selector": self.selector,
        }


@dataclass(frozen=True)
class AuthorityRowIdentity:
    record_ref: str
    record_sha256: str
    leading_dimension: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_ref": self.record_ref,
            "record_sha256": self.record_sha256,
            "leading_dimension": self.leading_dimension,
        }


@dataclass(frozen=True)
class TransformAuthorityRecord:
    authority_id: str
    kind: str
    source: AuthorityEndpoint
    target: AuthorityEndpoint
    payload: AuthorityPayload
    sampling_formula: str
    camera_id: str | None
    row_identity: AuthorityRowIdentity | None
    semantics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_id": TRANSFORM_AUTHORITY_SCHEMA_ID,
            "schema_version": TRANSFORM_AUTHORITY_SCHEMA_VERSION,
            "authority_id": self.authority_id,
            "kind": self.kind,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "payload": self.payload.to_dict(),
            "sampling_formula": self.sampling_formula,
            "semantics": copy.deepcopy(dict(self.semantics)),
        }
        if self.camera_id is not None:
            result["camera_id"] = self.camera_id
        if self.row_identity is not None:
            result["row_identity"] = self.row_identity.to_dict()
        return result

    def digest(self) -> str:
        return _mapping_sha256(self.to_dict())


def _payload(node: Any) -> AuthorityPayload:
    return AuthorityPayload(
        record_ref=f"/{canonical_node_path(node)}@array_values",
        record_sha256=array_values_sha256(node),
        selector="array_values",
    )


def _identity(value: BoundRowIdentityContract | None) -> AuthorityRowIdentity | None:
    if value is None:
        return None
    try:
        value.assert_verified()
    except RowIdentityContractError as exc:
        raise TransformAuthorityError(f"Row identity is stale: {exc}") from exc
    return AuthorityRowIdentity(
        record_ref=value.record_ref,
        record_sha256=value.record_sha256,
        leading_dimension=value.leading_dimension,
    )


def _parse_endpoint(value: Any, *, field_name: str) -> AuthorityEndpoint:
    payload = _exact_fields(
        value,
        expected=frozenset(
            {"space_id", "width", "height", "units", "pixel_convention", "authority"}
        ),
        field_name=field_name,
    )
    authority = _exact_fields(
        payload["authority"],
        expected=frozenset({"record_ref", "record_sha256", "selector"}),
        field_name=f"{field_name}.authority",
    )
    expected_spaces = {
        SOURCE_CAMERA_IMAGE_SPACE_ID,
        STIMULUS_CANVAS_SPACE_ID,
        ARENA_RELATIVE_CANVAS_SPACE_ID,
        ROI_LOCAL_SPACE_ID,
        DETECTOR_MODEL_INPUT_SPACE_ID,
        SOURCE_CAMERA_NORMALIZED_SPACE_ID,
        DETECTOR_NORMALIZED_SPACE_ID,
    }
    if payload["space_id"] not in expected_spaces:
        raise TransformAuthorityError(f"{field_name}.space_id is unsupported.")
    expected_units = (
        "normalized"
        if payload["space_id"]
        in {SOURCE_CAMERA_NORMALIZED_SPACE_ID, DETECTOR_NORMALIZED_SPACE_ID}
        else "px"
    )
    if payload["units"] != expected_units:
        raise TransformAuthorityError(
            f"{field_name}.units must be {expected_units!r} for its controlled space."
        )
    for name in ("width", "height"):
        if type(payload[name]) is not int or payload[name] <= 0:
            raise TransformAuthorityError(
                f"{field_name}.{name} must be an exact positive integer."
            )
    ref = _required_text(
        authority["record_ref"], field_name=f"{field_name}.authority.record_ref"
    )
    if not ref.startswith("/") or not ref.endswith("@pixel_frame_authority"):
        raise TransformAuthorityError(
            f"{field_name} must reference a persisted typed pixel-frame authority."
        )
    convention = _required_text(
        payload["pixel_convention"], field_name=f"{field_name}.pixel_convention"
    )
    from fisheye.shared.coordinate_descriptor import PIXEL_CONVENTIONS

    if convention not in PIXEL_CONVENTIONS - {"not_applicable"}:
        raise TransformAuthorityError(f"{field_name}.pixel_convention is unsupported.")
    return AuthorityEndpoint(
        space_id=payload["space_id"],
        width=payload["width"],
        height=payload["height"],
        units=expected_units,
        pixel_convention=convention,
        record_ref=ref,
        record_sha256=_sha256(
            authority["record_sha256"],
            field_name=f"{field_name}.authority.record_sha256",
        ),
        selector=_required_text(
            authority["selector"], field_name=f"{field_name}.authority.selector"
        ),
    )


def _parse_payload(value: Any) -> AuthorityPayload:
    payload = _exact_fields(
        value,
        expected=frozenset({"record_ref", "record_sha256", "selector"}),
        field_name="payload",
    )
    ref = _required_text(payload["record_ref"], field_name="payload.record_ref")
    if not ref.startswith("/") or not ref.endswith("@array_values"):
        raise TransformAuthorityError("payload.record_ref must identify array values.")
    if payload["selector"] != "array_values":
        raise TransformAuthorityError("payload.selector must be 'array_values'.")
    return AuthorityPayload(
        record_ref=ref,
        record_sha256=_sha256(
            payload["record_sha256"], field_name="payload.record_sha256"
        ),
        selector="array_values",
    )


def _parse_identity(value: Any) -> AuthorityRowIdentity:
    payload = _exact_fields(
        value,
        expected=frozenset({"record_ref", "record_sha256", "leading_dimension"}),
        field_name="row_identity",
    )
    if (
        type(payload["leading_dimension"]) is not int
        or payload["leading_dimension"] < 0
    ):
        raise TransformAuthorityError(
            "row_identity.leading_dimension must be an exact nonnegative integer."
        )
    return AuthorityRowIdentity(
        record_ref=_required_text(
            payload["record_ref"], field_name="row_identity.record_ref"
        ),
        record_sha256=_sha256(
            payload["record_sha256"], field_name="row_identity.record_sha256"
        ),
        leading_dimension=payload["leading_dimension"],
    )


def parse_transform_authority(value: Any) -> TransformAuthorityRecord:
    if isinstance(value, TransformAuthorityRecord):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise TransformAuthorityError("Transform authority must be a mapping.")
    required = {
        "schema_id",
        "schema_version",
        "authority_id",
        "kind",
        "source",
        "target",
        "payload",
        "sampling_formula",
        "semantics",
    }
    allowed = required | {"camera_id", "row_identity"}
    actual = set(value)
    if not required.issubset(actual) or not actual.issubset(allowed):
        raise TransformAuthorityError(
            f"Transform-authority fields are invalid; missing={sorted(required - actual)}, "
            f"unknown={sorted(actual - allowed)}."
        )
    if value["schema_id"] != TRANSFORM_AUTHORITY_SCHEMA_ID:
        raise TransformAuthorityError("Unsupported transform-authority schema_id.")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != TRANSFORM_AUTHORITY_SCHEMA_VERSION
    ):
        raise TransformAuthorityError("Unsupported transform-authority schema_version.")
    authority_id = _required_text(value["authority_id"], field_name="authority_id")
    if _AUTHORITY_ID_RE.fullmatch(authority_id) is None:
        raise TransformAuthorityError("authority_id is not canonical.")
    kind = _required_text(value["kind"], field_name="kind")
    if kind not in TRANSFORM_AUTHORITY_KINDS:
        raise TransformAuthorityError(f"Unsupported authority kind {kind!r}.")
    source = _parse_endpoint(value["source"], field_name="source")
    target = _parse_endpoint(value["target"], field_name="target")
    formula = _required_text(value["sampling_formula"], field_name="sampling_formula")
    camera_id = (
        _required_text(value["camera_id"], field_name="camera_id")
        if "camera_id" in value
        else None
    )
    row_identity = (
        _parse_identity(value["row_identity"]) if "row_identity" in value else None
    )
    if not isinstance(value["semantics"], Mapping):
        raise TransformAuthorityError("semantics must be a mapping.")
    semantics = json.loads(_canonical_json(value["semantics"]))

    if kind == SELECTED_CALIBRATION_AUTHORITY_KIND:
        if (
            source.space_id != SOURCE_CAMERA_IMAGE_SPACE_ID
            or target.space_id != STIMULUS_CANVAS_SPACE_ID
            or camera_id is None
            or row_identity is not None
            or formula != PROJECTIVE_XY_DIRECT_V1
        ):
            raise TransformAuthorityError(
                "Selected calibration direction/semantics are invalid."
            )
    elif kind == CROP_PLACEMENT_AUTHORITY_KIND:
        expected_formula = (
            SCALE_XY_PIXEL_CENTER_V1
            if source.pixel_convention == "pixel_center"
            else SCALE_XY_EDGE_ALIGNED_V1
        )
        if (
            source.space_id != ROI_LOCAL_SPACE_ID
            or target.space_id != SOURCE_CAMERA_IMAGE_SPACE_ID
            or camera_id is None
            or row_identity is None
            or source.pixel_convention != target.pixel_convention
            or formula != expected_formula
            or semantics != {"layout": "xywh"}
        ):
            raise TransformAuthorityError(
                "Crop-placement direction/semantics are invalid."
            )
    elif kind == MODEL_INPUT_PREPROCESSING_AUTHORITY_KIND:
        if (
            source.space_id != DETECTOR_MODEL_INPUT_SPACE_ID
            or target.space_id != ROI_LOCAL_SPACE_ID
            or camera_id is not None
            or row_identity is not None
            or source.pixel_convention != target.pixel_convention
            or formula != TRANSLATION_XY_DIRECT_V1
        ):
            raise TransformAuthorityError(
                "Model preprocessing direction/semantics are invalid."
            )
    elif kind == ARENA_CANVAS_PLACEMENT_AUTHORITY_KIND:
        expected_semantics = _exact_fields(
            semantics,
            expected=frozenset({"layout", "origin_x_px", "origin_y_px"}),
            field_name="semantics",
        )
        if (
            source.space_id != ARENA_RELATIVE_CANVAS_SPACE_ID
            or target.space_id != STIMULUS_CANVAS_SPACE_ID
            or camera_id is not None
            or row_identity is not None
            or source.pixel_convention != target.pixel_convention
            or formula != TRANSLATION_XY_DIRECT_V1
            or expected_semantics["layout"] != "arena_to_selected_canvas_translation_v1"
            or type(expected_semantics["origin_x_px"]) is not int
            or type(expected_semantics["origin_y_px"]) is not int
        ):
            raise TransformAuthorityError(
                "Arena placement direction/semantics are invalid."
            )
    else:
        if source.space_id == SOURCE_CAMERA_NORMALIZED_SPACE_ID:
            expected_target = SOURCE_CAMERA_IMAGE_SPACE_ID
        elif source.space_id == DETECTOR_NORMALIZED_SPACE_ID:
            expected_target = DETECTOR_MODEL_INPUT_SPACE_ID
        else:
            expected_target = None
        expected_semantics = _exact_fields(
            semantics,
            expected=frozenset({"reference_width_px", "reference_height_px"}),
            field_name="semantics",
        )
        if (
            expected_target is None
            or target.space_id != expected_target
            or source.units != "normalized"
            or target.units != "px"
            or camera_id is not None
            or row_identity is not None
            or source.pixel_convention != "continuous"
            or formula
            not in {
                NORMALIZED_TO_PIXEL_EDGE_EXTENT_V1,
                NORMALIZED_TO_PIXEL_CENTER_INDEX_V1,
            }
            or expected_semantics["reference_width_px"] != target.width
            or expected_semantics["reference_height_px"] != target.height
            or (formula == NORMALIZED_TO_PIXEL_CENTER_INDEX_V1)
            != (target.pixel_convention == "pixel_center")
        ):
            raise TransformAuthorityError(
                "Normalized-to-pixel direction/semantics are invalid."
            )
    return TransformAuthorityRecord(
        authority_id=authority_id,
        kind=kind,
        source=source,
        target=target,
        payload=_parse_payload(value["payload"]),
        sampling_formula=formula,
        camera_id=camera_id,
        row_identity=row_identity,
        semantics=semantics,
    )


def _same_frame_endpoint(
    record: AuthorityEndpoint, frame: BoundPixelFrameAuthority
) -> bool:
    return record == frame.endpoint


def _matrix(node: Any) -> np.ndarray:
    try:
        raw = np.asarray(node[:])
    except Exception as exc:
        raise TransformAuthorityError(
            "Unable to read exact transform payload."
        ) from exc
    if raw.dtype.str != "<f8" or raw.shape != (3, 3) or not np.isfinite(raw).all():
        raise TransformAuthorityError(
            "Constant/projective payload must be finite little-endian float64 3x3."
        )
    if int(np.linalg.matrix_rank(raw)) != 3:
        raise TransformAuthorityError("Transform payload matrix must be nonsingular.")
    return raw


def _homography_digest(matrix: np.ndarray) -> str:
    return hashlib.sha256(
        b"float64_little_endian_c_order_v1\x00"
        + matrix.astype("<f8", copy=False).tobytes(order="C")
    ).hexdigest()


def _selected_semantics(
    source_frame: BoundPixelFrameAuthority,
    target_frame: BoundPixelFrameAuthority,
    selected_calibration_snapshot: Any,
    matrix_node: Any,
) -> tuple[dict[str, Any], str]:
    from fisheye.shared.selected_calibration import (
        require_bound_selected_calibration_snapshot,
    )

    selected = require_bound_selected_calibration_snapshot(
        selected_calibration_snapshot
    )
    camera = selected.manifest.source_camera
    homography = selected.manifest.source_homography
    if (
        selected.archive_identity != source_frame.archive_identity
        or selected.archive_identity != target_frame.archive_identity
        or selected.archive_identity != archive_identity(matrix_node)
    ):
        raise TransformAuthorityError(
            "Selected snapshot, typed endpoints, and v2 payload must share one archive."
        )
    if camera.active_camera_id != homography.camera_id:
        raise TransformAuthorityError("Selected camera and homography camera differ.")
    if (
        source_frame.record.lineage.get("camera_id") != camera.active_camera_id
        or source_frame.endpoint.width != camera.native_width_px
        or source_frame.endpoint.height != camera.native_height_px
    ):
        raise TransformAuthorityError(
            "Selected-camera evidence conflicts with the acquisition-owned camera frame."
        )
    matrix_digest = _homography_digest(_matrix(matrix_node))
    if matrix_digest != homography.numeric_matrix_sha256:
        raise TransformAuthorityError(
            "Persisted matrix conflicts with builder-verified homography evidence."
        )
    return (
        {
            "selected_calibration_manifest": {
                "record_ref": selected.manifest_record_ref,
                "record_sha256": selected.manifest_sha256,
            },
            "source_camera": {
                "record_ref": selected.camera_record_ref,
                "record_sha256": _mapping_sha256(camera.to_dict()),
            },
            "source_homography": {
                "record_ref": selected.homography_record_ref,
                "record_sha256": homography.digest(),
            },
            "source_matrix_sha256": matrix_digest,
            "external_h5_freshness": "persisted_import_snapshot",
        },
        camera.active_camera_id,
    )


def _model_semantics(source_frame: BoundPixelFrameAuthority) -> dict[str, Any]:
    raw = source_frame.record.lineage.get("preprocessing")
    if not isinstance(raw, Mapping):
        raise TransformAuthorityError(
            "Model-input frame lacks preprocessing semantics."
        )
    return json.loads(_canonical_json(raw))


def arena_to_selected_canvas_matrix(
    source_frame: BoundPixelFrameAuthority,
    target_frame: BoundPixelFrameAuthority,
) -> np.ndarray:
    source = require_arena_relative_canvas_pixel_frame_authority(source_frame)
    target = require_selected_canvas_pixel_frame_authority(target_frame)
    selected_pointer = source.record.lineage.get("selected_canvas_frame")
    if selected_pointer != {
        "record_ref": target.record_ref,
        "record_sha256": target.record_sha256,
    }:
        raise TransformAuthorityError(
            "Arena frame does not target the exact selected canvas."
        )
    origin = source.record.lineage.get("origin_in_selected_canvas_px")
    if (
        not isinstance(origin, Mapping)
        or type(origin.get("x")) is not int
        or type(origin.get("y")) is not int
    ):
        raise TransformAuthorityError("Arena frame lacks exact integer placement.")
    return np.asarray(
        [
            [1.0, 0.0, float(origin["x"])],
            [0.0, 1.0, float(origin["y"])],
            [0.0, 0.0, 1.0],
        ],
        dtype="<f8",
    )


def _arena_semantics(source_frame: BoundPixelFrameAuthority) -> dict[str, Any]:
    origin = source_frame.record.lineage["origin_in_selected_canvas_px"]
    return {
        "layout": "arena_to_selected_canvas_translation_v1",
        "origin_x_px": int(origin["x"]),
        "origin_y_px": int(origin["y"]),
    }


def _normalized_semantics(target_frame: BoundPixelFrameAuthority) -> dict[str, int]:
    target = require_bound_pixel_frame_authority(target_frame)
    return {
        "reference_width_px": target.endpoint.width,
        "reference_height_px": target.endpoint.height,
    }


def _require_archive(
    node: Any,
    payload_node: Any,
    source_frame: BoundPixelFrameAuthority,
    target_frame: BoundPixelFrameAuthority,
    row_identity: BoundRowIdentityContract | None,
) -> ArchiveIdentity:
    try:
        common = archive_identity(node)
        identities = [
            archive_identity(payload_node),
            source_frame.archive_identity,
            target_frame.archive_identity,
        ]
        if row_identity is not None:
            identities.append(row_identity.archive_identity)
    except ArchiveIdentityError as exc:
        raise TransformAuthorityError(str(exc)) from exc
    if any(item != common for item in identities):
        raise TransformAuthorityError(
            "Transform authority evidence comes from different archives/stores."
        )
    return common


def _validate(
    record: TransformAuthorityRecord,
    *,
    payload_node: Any,
    source_frame: BoundPixelFrameAuthority,
    target_frame: BoundPixelFrameAuthority,
    row_identity: BoundRowIdentityContract | None,
    selected_calibration_snapshot: Any | None,
) -> None:
    try:
        source = require_bound_pixel_frame_authority(source_frame)
        target = require_bound_pixel_frame_authority(target_frame)
    except PixelFrameAuthorityError as exc:
        raise TransformAuthorityError(
            f"Typed endpoint authority is stale: {exc}"
        ) from exc
    if not _same_frame_endpoint(record.source, source) or not _same_frame_endpoint(
        record.target, target
    ):
        raise TransformAuthorityError(
            "Transform authority endpoints do not equal exact typed pixel frames."
        )
    if record.payload != _payload(payload_node):
        raise TransformAuthorityError("Transform authority payload is stale.")
    if record.row_identity != _identity(row_identity):
        raise TransformAuthorityError("Transform authority row identity is stale.")
    if record.kind == SELECTED_CALIBRATION_AUTHORITY_KIND:
        require_source_camera_pixel_frame_authority(source)
        require_selected_canvas_pixel_frame_authority(target)
        semantics, camera_id = _selected_semantics(
            source,
            target,
            selected_calibration_snapshot,
            payload_node,
        )
        if record.semantics != semantics or record.camera_id != camera_id:
            raise TransformAuthorityError("Selected-calibration evidence changed.")
    elif record.kind == CROP_PLACEMENT_AUTHORITY_KIND:
        roi = require_roi_pixel_frame_authority(source)
        camera = require_source_camera_pixel_frame_authority(target)
        ownership = require_bound_crop_placement_ownership(
            roi._context.get("crop_placement_ownership")
        )
        if (
            row_identity is None
            or roi.row_identity is None
            or _identity(row_identity) != _identity(roi.row_identity)
        ):
            raise TransformAuthorityError(
                "Crop authority does not use the ROI frame's exact observation identity."
            )
        placement_pointer = ownership.record.crop_placement
        if placement_pointer != _payload(payload_node).to_dict():
            raise TransformAuthorityError(
                "Crop transform payload is not the ROI frame's exact placement record."
            )
        camera_id = camera.record.lineage["camera_id"]
        if record.camera_id != camera_id:
            raise TransformAuthorityError("Crop target camera changed.")
        values = np.asarray(payload_node[:])
        if (
            values.dtype.kind not in {"i", "u", "f"}
            or values.ndim != 2
            or values.shape != (row_identity.leading_dimension, 4)
        ):
            raise TransformAuthorityError(
                "Crop placement must be real numeric with exact shape (N,4)."
            )
    elif record.kind == MODEL_INPUT_PREPROCESSING_AUTHORITY_KIND:
        model = require_model_input_pixel_frame_authority(source)
        roi = require_roi_pixel_frame_authority(target)
        if model.record.lineage.get("roi_frame") != {
            "record_ref": roi.record_ref,
            "record_sha256": roi.record_sha256,
        }:
            raise TransformAuthorityError(
                "Model-input frame targets a different ROI frame."
            )
        if (
            model.record.lineage.get("preprocessing_payload")
            != _payload(payload_node).to_dict()
        ):
            raise TransformAuthorityError(
                "Model transform payload is not its exact preprocessing record."
            )
        if record.semantics != _model_semantics(model):
            raise TransformAuthorityError("Model preprocessing semantics changed.")
        transform = model._context["transform"]
        if not np.array_equal(
            _matrix(payload_node), model_input_to_roi_matrix(transform)
        ):
            raise TransformAuthorityError("Model preprocessing matrix changed.")
    elif record.kind == ARENA_CANVAS_PLACEMENT_AUTHORITY_KIND:
        arena = require_arena_relative_canvas_pixel_frame_authority(source)
        canvas = require_selected_canvas_pixel_frame_authority(target)
        if record.semantics != _arena_semantics(arena) or not np.array_equal(
            _matrix(payload_node), arena_to_selected_canvas_matrix(arena, canvas)
        ):
            raise TransformAuthorityError(
                "Arena-to-selected-canvas placement changed or has the wrong direction."
            )
    else:
        normalized = require_normalized_pixel_frame_authority(source)
        pixel = require_bound_pixel_frame_authority(target)
        linked = normalized._context.get("pixel_frame")
        if linked is None or linked.record_ref != pixel.record_ref:
            raise TransformAuthorityError(
                "Normalized endpoint does not bind the exact target pixel frame."
            )
        if record.semantics != _normalized_semantics(pixel) or not np.array_equal(
            _matrix(payload_node), normalized_to_pixel_matrix(pixel)
        ):
            raise TransformAuthorityError(
                "Normalized-to-pixel formula or reference dimensions changed."
            )


@dataclass(frozen=True, init=False)
class BoundTransformAuthority:
    record: TransformAuthorityRecord
    authority_path: str
    record_ref: str
    record_sha256: str
    attr_name: str
    source_frame: BoundPixelFrameAuthority = field(repr=False, compare=False)
    target_frame: BoundPixelFrameAuthority = field(repr=False, compare=False)
    row_identity: BoundRowIdentityContract | None = field(repr=False, compare=False)
    _archive_identity: ArchiveIdentity = field(repr=False, compare=False)
    _authority_node: Any = field(repr=False, compare=False)
    _payload_node: Any = field(repr=False, compare=False)
    _selected_calibration_snapshot: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        record: TransformAuthorityRecord,
        authority_path: str,
        source_frame: BoundPixelFrameAuthority,
        target_frame: BoundPixelFrameAuthority,
        row_identity: BoundRowIdentityContract | None,
        archive: ArchiveIdentity,
        authority_node: Any,
        payload_node: Any,
        attr_name: str,
        selected_calibration_snapshot: Any = None,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BOUND_AUTHORITY_SEAL:
            raise TransformAuthorityError(
                "Bound transform authorities cannot be constructed directly."
            )
        object.__setattr__(self, "record", record)
        object.__setattr__(self, "authority_path", authority_path)
        object.__setattr__(self, "record_ref", f"/{authority_path}@{attr_name}")
        object.__setattr__(self, "record_sha256", record.digest())
        object.__setattr__(self, "attr_name", attr_name)
        object.__setattr__(self, "source_frame", source_frame)
        object.__setattr__(self, "target_frame", target_frame)
        object.__setattr__(self, "row_identity", row_identity)
        object.__setattr__(self, "_archive_identity", archive)
        object.__setattr__(self, "_authority_node", authority_node)
        object.__setattr__(self, "_payload_node", payload_node)
        object.__setattr__(
            self,
            "_selected_calibration_snapshot",
            selected_calibration_snapshot,
        )
        object.__setattr__(self, "_seal", _verification_seal)

    @property
    def archive_identity(self) -> ArchiveIdentity:
        return self._archive_identity

    def assert_verified(self) -> None:
        if self._seal is not _BOUND_AUTHORITY_SEAL:
            raise TransformAuthorityError("Transform authority is not sealed evidence.")
        current = load_bound_transform_authority(
            self._authority_node,
            payload_node=self._payload_node,
            source_frame=self.source_frame,
            target_frame=self.target_frame,
            row_identity=self.row_identity,
            selected_calibration_snapshot=self._selected_calibration_snapshot,
            attr_name=self.attr_name,
        )
        if (
            current.record != self.record
            or current.record_ref != self.record_ref
            or current.record_sha256 != self.record_sha256
            or current.archive_identity != self.archive_identity
        ):
            raise TransformAuthorityError("Persisted transform authority changed.")


def load_bound_transform_authority(
    authority_node: Any,
    *,
    payload_node: Any,
    source_frame: BoundPixelFrameAuthority,
    target_frame: BoundPixelFrameAuthority,
    row_identity: BoundRowIdentityContract | None = None,
    selected_calibration_snapshot: Any = None,
    attr_name: str = TRANSFORM_AUTHORITY_ATTR,
) -> BoundTransformAuthority:
    if attr_name not in TRANSFORM_AUTHORITY_ATTRS:
        raise TransformAuthorityError(
            f"Unsupported transform-authority attr {attr_name!r}."
        )
    archive = _require_archive(
        authority_node,
        payload_node,
        source_frame,
        target_frame,
        row_identity,
    )
    attrs = getattr(authority_node, "attrs", None)
    if not isinstance(attrs, Mapping):
        raise TransformAuthorityError("Authority node must expose persisted attrs.")
    digest_attr = f"{attr_name}_sha256"
    raw = attrs.get(attr_name)
    record = parse_transform_authority(raw)
    specialized_convention = {
        TRANSFORM_AUTHORITY_PIXEL_CENTER_ATTR: "pixel_center",
        TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR: "pixel_edge_half_open",
    }.get(attr_name)
    if specialized_convention is not None and (
        record.kind != CROP_PLACEMENT_AUTHORITY_KIND
        or source_frame.pixel_convention != specialized_convention
    ):
        convention_label = specialized_convention.replace("_", "-")
        raise TransformAuthorityError(
            f"The {convention_label} transform-authority attr is reserved "
            f"for a {convention_label} crop-placement transform."
        )
    _require_crop_ownership_attr_pair(
        record,
        source_frame,
        attr_name=attr_name,
    )
    if not isinstance(raw, Mapping) or not _exact_json_equal(raw, record.to_dict()):
        raise TransformAuthorityError(
            "Raw persisted authority mapping is not its parsed canonical form."
        )
    stored = _sha256(
        attrs.get(digest_attr),
        field_name=digest_attr,
    )
    if stored != record.digest():
        raise TransformAuthorityError("Persisted transform-authority digest is stale.")
    _validate(
        record,
        payload_node=payload_node,
        source_frame=source_frame,
        target_frame=target_frame,
        row_identity=row_identity,
        selected_calibration_snapshot=selected_calibration_snapshot,
    )
    return BoundTransformAuthority(
        record=record,
        authority_path=canonical_node_path(authority_node),
        source_frame=source_frame,
        target_frame=target_frame,
        row_identity=row_identity,
        archive=archive,
        authority_node=authority_node,
        payload_node=payload_node,
        attr_name=attr_name,
        selected_calibration_snapshot=selected_calibration_snapshot,
        _verification_seal=_BOUND_AUTHORITY_SEAL,
    )


def require_bound_transform_authority(value: Any) -> BoundTransformAuthority:
    if (
        type(value) is not BoundTransformAuthority
        or value._seal is not _BOUND_AUTHORITY_SEAL
    ):
        raise TransformAuthorityError("A sealed bound transform authority is required.")
    value.assert_verified()
    return value


def _stamp(
    node: Any,
    *,
    record: TransformAuthorityRecord,
    payload_node: Any,
    source_frame: BoundPixelFrameAuthority,
    target_frame: BoundPixelFrameAuthority,
    row_identity: BoundRowIdentityContract | None,
    selected_calibration_snapshot: Any = None,
    attr_name: str = TRANSFORM_AUTHORITY_ATTR,
) -> BoundTransformAuthority:
    if attr_name not in TRANSFORM_AUTHORITY_ATTRS:
        raise TransformAuthorityError(
            f"Unsupported transform-authority attr {attr_name!r}."
        )
    try:
        attrs = require_trusted_coordinate_attrs(node, label="Transform authority")
    except PixelFrameAuthorityError as exc:
        raise TransformAuthorityError(str(exc)) from exc
    _require_archive(node, payload_node, source_frame, target_frame, row_identity)
    parsed = parse_transform_authority(record)
    specialized_convention = {
        TRANSFORM_AUTHORITY_PIXEL_CENTER_ATTR: "pixel_center",
        TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR: "pixel_edge_half_open",
    }.get(attr_name)
    if specialized_convention is not None and (
        parsed.kind != CROP_PLACEMENT_AUTHORITY_KIND
        or source_frame.pixel_convention != specialized_convention
    ):
        convention_label = specialized_convention.replace("_", "-")
        raise TransformAuthorityError(
            f"The {convention_label} transform-authority attr is reserved "
            f"for a {convention_label} crop-placement transform."
        )
    _require_crop_ownership_attr_pair(
        parsed,
        source_frame,
        attr_name=attr_name,
    )
    _validate(
        parsed,
        payload_node=payload_node,
        source_frame=source_frame,
        target_frame=target_frame,
        row_identity=row_identity,
        selected_calibration_snapshot=selected_calibration_snapshot,
    )
    snapshot = copy.deepcopy(dict(attrs))
    digest_attr = f"{attr_name}_sha256"
    intended = {
        attr_name: parsed.to_dict(),
        digest_attr: parsed.digest(),
    }
    expected = _expected_attrs_after_update(snapshot, intended)
    try:
        attrs.update(copy.deepcopy(intended))
        _require_exact_attrs_state(
            attrs,
            expected,
            label="Transform-authority post-write",
        )
        bound = load_bound_transform_authority(
            node,
            payload_node=payload_node,
            source_frame=source_frame,
            target_frame=target_frame,
            row_identity=row_identity,
            selected_calibration_snapshot=selected_calibration_snapshot,
            attr_name=attr_name,
        )
        _require_exact_attrs_state(
            require_trusted_coordinate_attrs(
                node,
                label="Reloaded transform authority",
            ),
            expected,
            label="Transform-authority post-reload",
        )
        return bound
    except Exception as exc:
        try:
            _restore_exact_attrs(attrs, snapshot)
        except Exception as rollback_exc:  # pragma: no cover
            raise TransformAuthorityError(
                f"Authority stamp failed and rollback was incomplete: {rollback_exc}"
            ) from exc
        if isinstance(exc, TransformAuthorityError):
            raise
        raise TransformAuthorityError(f"Authority stamp failed: {exc}") from exc


def stamp_selected_calibration_transform_authority(
    authority_node: Any,
    *,
    authority_id: str,
    source_matrix_node: Any,
    source_frame: BoundPixelFrameAuthority,
    target_frame: BoundPixelFrameAuthority,
    selected_calibration_snapshot: Any,
) -> BoundTransformAuthority:
    source = require_source_camera_pixel_frame_authority(source_frame)
    target = require_selected_canvas_pixel_frame_authority(target_frame)
    semantics, camera_id = _selected_semantics(
        source,
        target,
        selected_calibration_snapshot,
        source_matrix_node,
    )
    record = TransformAuthorityRecord(
        authority_id=_required_text(authority_id, field_name="authority_id"),
        kind=SELECTED_CALIBRATION_AUTHORITY_KIND,
        source=source.endpoint,
        target=target.endpoint,
        payload=_payload(source_matrix_node),
        sampling_formula=PROJECTIVE_XY_DIRECT_V1,
        camera_id=camera_id,
        row_identity=None,
        semantics=semantics,
    )
    return _stamp(
        authority_node,
        record=record,
        payload_node=source_matrix_node,
        source_frame=source,
        target_frame=target,
        row_identity=None,
        selected_calibration_snapshot=selected_calibration_snapshot,
    )


def stamp_crop_placement_transform_authority(
    placement_node: Any,
    *,
    authority_id: str,
    source_frame: BoundPixelFrameAuthority,
    target_frame: BoundPixelFrameAuthority,
    attr_name: str = TRANSFORM_AUTHORITY_ATTR,
) -> BoundTransformAuthority:
    source = require_roi_pixel_frame_authority(source_frame)
    target = require_source_camera_pixel_frame_authority(target_frame)
    ownership = require_bound_crop_placement_ownership(
        source._context.get("crop_placement_ownership")
    )
    if ownership._placement_node is not placement_node:
        raise TransformAuthorityError(
            "Crop transform must use the exact crop-writer-owned placement node."
        )
    row_identity = source.row_identity
    if row_identity is None:
        raise TransformAuthorityError("ROI frame lacks exact row identity.")
    formula = (
        SCALE_XY_PIXEL_CENTER_V1
        if source.pixel_convention == "pixel_center"
        else SCALE_XY_EDGE_ALIGNED_V1
    )
    camera_id = target.record.lineage["camera_id"]
    record = TransformAuthorityRecord(
        authority_id=_required_text(authority_id, field_name="authority_id"),
        kind=CROP_PLACEMENT_AUTHORITY_KIND,
        source=source.endpoint,
        target=target.endpoint,
        payload=_payload(placement_node),
        sampling_formula=formula,
        camera_id=camera_id,
        row_identity=_identity(row_identity),
        semantics={"layout": "xywh"},
    )
    return _stamp(
        placement_node,
        record=record,
        payload_node=placement_node,
        source_frame=source,
        target_frame=target,
        row_identity=row_identity,
        attr_name=attr_name,
    )


def stamp_model_input_transform_authority(
    authority_node: Any,
    *,
    authority_id: str,
    matrix_node: Any,
    source_frame: BoundPixelFrameAuthority,
    target_frame: BoundPixelFrameAuthority,
) -> BoundTransformAuthority:
    source = require_model_input_pixel_frame_authority(source_frame)
    target = require_roi_pixel_frame_authority(target_frame)
    record = TransformAuthorityRecord(
        authority_id=_required_text(authority_id, field_name="authority_id"),
        kind=MODEL_INPUT_PREPROCESSING_AUTHORITY_KIND,
        source=source.endpoint,
        target=target.endpoint,
        payload=_payload(matrix_node),
        sampling_formula=TRANSLATION_XY_DIRECT_V1,
        camera_id=None,
        row_identity=None,
        semantics=_model_semantics(source),
    )
    return _stamp(
        authority_node,
        record=record,
        payload_node=matrix_node,
        source_frame=source,
        target_frame=target,
        row_identity=None,
    )


def stamp_arena_to_selected_canvas_transform_authority(
    authority_node: Any,
    *,
    authority_id: str,
    matrix_node: Any,
    source_frame: BoundPixelFrameAuthority,
    target_frame: BoundPixelFrameAuthority,
) -> BoundTransformAuthority:
    source = require_arena_relative_canvas_pixel_frame_authority(source_frame)
    target = require_selected_canvas_pixel_frame_authority(target_frame)
    expected = arena_to_selected_canvas_matrix(source, target)
    if not np.array_equal(_matrix(matrix_node), expected):
        raise TransformAuthorityError(
            "Arena placement payload must be exact arena-relative to selected-canvas translation."
        )
    record = TransformAuthorityRecord(
        authority_id=_required_text(authority_id, field_name="authority_id"),
        kind=ARENA_CANVAS_PLACEMENT_AUTHORITY_KIND,
        source=source.endpoint,
        target=target.endpoint,
        payload=_payload(matrix_node),
        sampling_formula=TRANSLATION_XY_DIRECT_V1,
        camera_id=None,
        row_identity=None,
        semantics=_arena_semantics(source),
    )
    return _stamp(
        authority_node,
        record=record,
        payload_node=matrix_node,
        source_frame=source,
        target_frame=target,
        row_identity=None,
    )


def stamp_normalized_to_pixel_transform_authority(
    authority_node: Any,
    *,
    authority_id: str,
    matrix_node: Any,
    source_frame: BoundPixelFrameAuthority,
    target_frame: BoundPixelFrameAuthority,
) -> BoundTransformAuthority:
    source = require_normalized_pixel_frame_authority(source_frame)
    target = require_bound_pixel_frame_authority(target_frame)
    linked = source._context.get("pixel_frame")
    if linked is None or linked.record_ref != target.record_ref:
        raise TransformAuthorityError(
            "Normalized authority must target its exact bound pixel frame."
        )
    expected = normalized_to_pixel_matrix(target)
    if not np.array_equal(_matrix(matrix_node), expected):
        raise TransformAuthorityError(
            "Normalized-to-pixel payload conflicts with the controlled W/H formula."
        )
    formula = source.record.lineage["normalization_formula"]
    record = TransformAuthorityRecord(
        authority_id=_required_text(authority_id, field_name="authority_id"),
        kind=NORMALIZED_TO_PIXEL_AUTHORITY_KIND,
        source=source.endpoint,
        target=target.endpoint,
        payload=_payload(matrix_node),
        sampling_formula=formula,
        camera_id=None,
        row_identity=None,
        semantics=_normalized_semantics(target),
    )
    return _stamp(
        authority_node,
        record=record,
        payload_node=matrix_node,
        source_frame=source,
        target_frame=target,
        row_identity=None,
    )


__all__ = [
    "ARENA_CANVAS_PLACEMENT_AUTHORITY_KIND",
    "CROP_PLACEMENT_AUTHORITY_KIND",
    "DETECTOR_MODEL_INPUT_SPACE_ID",
    "MODEL_INPUT_PREPROCESSING_AUTHORITY_KIND",
    "NORMALIZED_TO_PIXEL_AUTHORITY_KIND",
    "ROI_LOCAL_SPACE_ID",
    "SELECTED_CALIBRATION_AUTHORITY_KIND",
    "SOURCE_CAMERA_IMAGE_SPACE_ID",
    "STIMULUS_CANVAS_SPACE_ID",
    "TRANSFORM_AUTHORITY_ATTR",
    "TRANSFORM_AUTHORITY_ATTRS",
    "TRANSFORM_AUTHORITY_CANONICALIZATION",
    "TRANSFORM_AUTHORITY_DIGEST_ATTR",
    "TRANSFORM_AUTHORITY_KINDS",
    "TRANSFORM_AUTHORITY_PIXEL_CENTER_ATTR",
    "TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR",
    "TRANSFORM_AUTHORITY_SCHEMA_ID",
    "TRANSFORM_AUTHORITY_SCHEMA_VERSION",
    "AuthorityEndpoint",
    "AuthorityPayload",
    "AuthorityRowIdentity",
    "BoundTransformAuthority",
    "TransformAuthorityError",
    "TransformAuthorityRecord",
    "array_values_sha256",
    "arena_to_selected_canvas_matrix",
    "load_bound_transform_authority",
    "model_input_to_roi_matrix",
    "parse_transform_authority",
    "require_bound_transform_authority",
    "stamp_crop_placement_transform_authority",
    "stamp_arena_to_selected_canvas_transform_authority",
    "stamp_model_input_transform_authority",
    "stamp_normalized_to_pixel_transform_authority",
    "stamp_selected_calibration_transform_authority",
]
