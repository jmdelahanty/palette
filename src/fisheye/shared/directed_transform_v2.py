"""Strict, typed-endpoint version-2 directed transforms."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Mapping

import numpy as np

from fisheye.shared.archive_identity import ArchiveIdentity, ArchiveIdentityError, archive_identity
from fisheye.shared.coordinate_identity import BoundRowIdentityContract, RowIdentityContractError
from fisheye.shared.coordinate_reference import canonical_node_path
from fisheye.shared.directed_transform import (
    DIRECTED_TRANSFORM_ATTR,
    DIRECTED_TRANSFORM_DIGEST_SUFFIX,
    DirectedTransformError,
    directed_transform_digest,
    homography_matrix_sha256,
    parse_directed_homography,
)
from fisheye.shared.pixel_frame_authority import (
    NORMALIZED_TO_PIXEL_CENTER_INDEX_V1,
    NORMALIZED_TO_PIXEL_EDGE_EXTENT_V1,
    PROJECTIVE_XY_DIRECT_V1,
    SCALE_XY_EDGE_ALIGNED_V1,
    SCALE_XY_PIXEL_CENTER_V1,
    TRANSLATION_XY_DIRECT_V1,
    BoundPixelFrameAuthority,
    PixelFrameAuthorityError,
    PixelFrameEndpoint,
    require_bound_pixel_frame_authority,
    require_trusted_coordinate_attrs,
)
from fisheye.shared.transform_authority import (
    ARENA_CANVAS_PLACEMENT_AUTHORITY_KIND,
    CROP_PLACEMENT_AUTHORITY_KIND,
    MODEL_INPUT_PREPROCESSING_AUTHORITY_KIND,
    NORMALIZED_TO_PIXEL_AUTHORITY_KIND,
    SELECTED_CALIBRATION_AUTHORITY_KIND,
    TRANSFORM_AUTHORITY_ATTR,
    TRANSFORM_AUTHORITY_PIXEL_CENTER_ATTR,
    TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
    AuthorityPayload,
    AuthorityRowIdentity,
    BoundTransformAuthority,
    TransformAuthorityError,
    array_values_sha256,
    require_bound_transform_authority,
)


DIRECTED_TRANSFORM_V2_SCHEMA_ID = "palette.directed_transform"
DIRECTED_TRANSFORM_V2_SCHEMA_VERSION = 2
DIRECTED_TRANSFORM_V2_ATTR = "directed_transform_v2"
DIRECTED_TRANSFORM_V2_DIGEST_ATTR = "directed_transform_v2_sha256"
DIRECTED_TRANSFORM_V2_PIXEL_CENTER_ATTR = "directed_transform_v2_pixel_center"
DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR = (
    "directed_transform_v2_pixel_edge_half_open"
)
DIRECTED_TRANSFORM_V2_ATTRS = frozenset(
    {
        DIRECTED_TRANSFORM_V2_ATTR,
        DIRECTED_TRANSFORM_V2_PIXEL_CENTER_ATTR,
        DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR,
    }
)


def _require_transform_authority_attr_pair(
    authority: BoundTransformAuthority,
    *,
    attr_name: str,
) -> None:
    if authority.record.kind != CROP_PLACEMENT_AUTHORITY_KIND:
        return
    expected = {
        DIRECTED_TRANSFORM_V2_ATTR: TRANSFORM_AUTHORITY_ATTR,
        DIRECTED_TRANSFORM_V2_PIXEL_CENTER_ATTR: (
            TRANSFORM_AUTHORITY_PIXEL_CENTER_ATTR
        ),
        DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR: TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
    }[attr_name]
    if authority.attr_name != expected:
        raise DirectedTransformV2Error(
            "Directed crop-placement transform is cross-wired to the wrong "
            "closed transform-authority attr."
        )
DIRECTED_TRANSFORM_V2_DIRECTION = "source_to_target"
DIRECTED_TRANSFORM_V2_CANONICALIZATION = "canonical_json_sort_keys_v1"
MIGRATION_ELIGIBILITY_SCHEMA_ID = (
    "palette.directed_transform_v1_migration_eligibility"
)
MIGRATION_ELIGIBILITY_SCHEMA_VERSION = 1
MIGRATION_ELIGIBILITY_ATTR = "directed_transform_v1_migration_eligibility"
MIGRATION_ELIGIBILITY_DIGEST_ATTR = (
    "directed_transform_v1_migration_eligibility_sha256"
)
MIGRATION_ELIGIBILITY_BASIS = (
    "artifact_specific_explicit_continuous_endpoint_review_v1"
)

HOMOGRAPHY_KIND = "homography"
AFFINE_2D_CONSTANT_KIND = "affine_2d_constant"
AFFINE_2D_ROWWISE_KIND = "affine_2d_rowwise"
DIRECTED_TRANSFORM_V2_KINDS = frozenset(
    {HOMOGRAPHY_KIND, AFFINE_2D_CONSTANT_KIND, AFFINE_2D_ROWWISE_KIND}
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TRANSFORM_ID_RE = re.compile(r"^[A-Za-z0-9_.:+-]+$")
_BOUND_TRANSFORM_SEAL = object()
_HISTORICAL_TRANSFORM_ATTRS = frozenset(
    {"directed_transform", "directed_transform_sha256"}
)


class DirectedTransformV2Error(ValueError):
    """Raised when a v2 transform is unresolved, stale, or misapplied."""


def _required_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DirectedTransformV2Error(
            f"{field_name} must be a non-empty string without surrounding whitespace."
        )
    return value


def _sha256(value: Any, *, field_name: str) -> str:
    text = _required_text(value, field_name=field_name)
    if _SHA256_RE.fullmatch(text) is None:
        raise DirectedTransformV2Error(
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
        raise DirectedTransformV2Error(
            "Directed-transform records must contain finite canonical JSON values."
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
        raise DirectedTransformV2Error(
            f"{label} attrs differ from the exact snapshot plus intended payload."
        )


def _restore_exact_attrs(attrs: Any, snapshot: Mapping[str, Any]) -> None:
    for name in tuple(attrs.keys()):
        del attrs[name]
    attrs.update(copy.deepcopy(dict(snapshot)))
    if not _exact_json_equal(dict(attrs), dict(snapshot)):
        raise RuntimeError("restored attrs differ type-strictly from snapshot")


def _require_exact_builtin_json(value: Any, *, field_name: str) -> None:
    if isinstance(value, Mapping):
        if type(value) is not dict:
            raise DirectedTransformV2Error(
                f"{field_name} mappings must use the exact built-in dict type."
            )
        for name, item in value.items():
            if type(name) is not str:
                raise DirectedTransformV2Error(
                    f"{field_name} mapping keys must use the exact string type."
                )
            _require_exact_builtin_json(item, field_name=f"{field_name}.{name}")
        return
    if isinstance(value, list):
        if type(value) is not list:
            raise DirectedTransformV2Error(
                f"{field_name} sequences must use the exact built-in list type."
            )
        for index, item in enumerate(value):
            _require_exact_builtin_json(
                item,
                field_name=f"{field_name}[{index}]",
            )
        return
    if type(value) not in {str, int, float, bool, type(None)}:
        raise DirectedTransformV2Error(
            f"{field_name} contains a non-canonical scalar type."
        )


def _require_exact_legacy_migration_scalars(raw: Any) -> None:
    _require_exact_builtin_json(raw, field_name="legacy migration record")
    if type(raw) is not dict:  # defensive; exact JSON validation already rejects it
        raise DirectedTransformV2Error(
            "Legacy migration record must use the exact built-in dict type."
        )
    if type(raw.get("schema_version")) is not int:
        raise DirectedTransformV2Error(
            "Legacy migration schema_version must be an exact integer."
        )
    for name in (
        "schema_id",
        "transform_id",
        "kind",
        "from_space_id",
        "to_space_id",
        "direction",
        "calibration_ref",
        "matrix_sha256",
        "camera_id",
    ):
        if name in raw and type(raw[name]) is not str:
            raise DirectedTransformV2Error(
                f"Legacy migration {name} must use the exact string type."
            )
    for extent_name in ("source_reference_extent", "target_reference_extent"):
        extent = raw.get(extent_name)
        if type(extent) is not dict:
            raise DirectedTransformV2Error(
                f"Legacy migration {extent_name} must use an exact dict."
            )
        for name in ("width", "height"):
            if type(extent.get(name)) is not int or extent[name] <= 0:
                raise DirectedTransformV2Error(
                    f"Legacy migration {extent_name}.{name} must be an exact positive integer."
                )
        for name in ("units", "authority"):
            if type(extent.get(name)) is not str or not extent[name]:
                raise DirectedTransformV2Error(
                    f"Legacy migration {extent_name}.{name} must be an exact non-empty string."
                )


def _exact_fields(value: Any, *, expected: frozenset[str], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DirectedTransformV2Error(f"{field_name} must be a mapping.")
    actual = frozenset(value)
    if actual != expected:
        raise DirectedTransformV2Error(
            f"{field_name} fields are invalid; missing={sorted(expected-actual)}, "
            f"unknown={sorted(actual-expected)}."
        )
    return value


@dataclass(frozen=True)
class TransformAuthorityPointer:
    kind: str
    record_ref: str
    record_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "record_ref": self.record_ref,
            "record_sha256": self.record_sha256,
        }


@dataclass(frozen=True)
class ExactTransformReference:
    relationship: str
    record_ref: str
    record_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "relationship": self.relationship,
            "record_ref": self.record_ref,
            "record_sha256": self.record_sha256,
        }


@dataclass(frozen=True)
class DirectedTransformV2:
    transform_id: str
    kind: str
    from_space_id: str
    to_space_id: str
    source: PixelFrameEndpoint
    target: PixelFrameEndpoint
    transform_authority: TransformAuthorityPointer
    payload: AuthorityPayload
    sampling_formula: str
    camera_id: str | None
    row_identity: AuthorityRowIdentity | None
    inverse_of: ExactTransformReference | None

    @property
    def direction(self) -> str:
        return DIRECTED_TRANSFORM_V2_DIRECTION

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_id": DIRECTED_TRANSFORM_V2_SCHEMA_ID,
            "schema_version": DIRECTED_TRANSFORM_V2_SCHEMA_VERSION,
            "transform_id": self.transform_id,
            "kind": self.kind,
            "from_space_id": self.from_space_id,
            "to_space_id": self.to_space_id,
            "direction": self.direction,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "transform_authority": self.transform_authority.to_dict(),
            "payload": self.payload.to_dict(),
            "sampling_formula": self.sampling_formula,
        }
        if self.camera_id is not None:
            result["camera_id"] = self.camera_id
        if self.row_identity is not None:
            result["row_identity"] = self.row_identity.to_dict()
        if self.inverse_of is not None:
            result["inverse_of"] = self.inverse_of.to_dict()
        return result

    def digest(self) -> str:
        return _mapping_sha256(self.to_dict())


def _parse_endpoint(value: Any, *, field_name: str) -> PixelFrameEndpoint:
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
    units = _required_text(payload["units"], field_name=f"{field_name}.units")
    if units not in {"px", "normalized"}:
        raise DirectedTransformV2Error(
            f"{field_name}.units must be 'px' or 'normalized'."
        )
    for name in ("width", "height"):
        if type(payload[name]) is not int or payload[name] <= 0:
            raise DirectedTransformV2Error(
                f"{field_name}.{name} must be an exact positive integer."
            )
    ref = _required_text(
        authority["record_ref"], field_name=f"{field_name}.authority.record_ref"
    )
    if not ref.startswith("/") or not ref.endswith("@pixel_frame_authority"):
        raise DirectedTransformV2Error(
            f"{field_name} must reference a typed persisted pixel frame."
        )
    return PixelFrameEndpoint(
        space_id=_required_text(payload["space_id"], field_name=f"{field_name}.space_id"),
        width=payload["width"],
        height=payload["height"],
        units=units,
        pixel_convention=_required_text(
            payload["pixel_convention"], field_name=f"{field_name}.pixel_convention"
        ),
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
        raise DirectedTransformV2Error("payload.record_ref must identify array values.")
    if payload["selector"] != "array_values":
        raise DirectedTransformV2Error("payload.selector must be 'array_values'.")
    return AuthorityPayload(
        record_ref=ref,
        record_sha256=_sha256(payload["record_sha256"], field_name="payload.record_sha256"),
        selector="array_values",
    )


def _parse_authority(value: Any) -> TransformAuthorityPointer:
    payload = _exact_fields(
        value,
        expected=frozenset({"kind", "record_ref", "record_sha256"}),
        field_name="transform_authority",
    )
    return TransformAuthorityPointer(
        kind=_required_text(payload["kind"], field_name="transform_authority.kind"),
        record_ref=_required_text(
            payload["record_ref"], field_name="transform_authority.record_ref"
        ),
        record_sha256=_sha256(
            payload["record_sha256"], field_name="transform_authority.record_sha256"
        ),
    )


def _parse_identity(value: Any) -> AuthorityRowIdentity:
    payload = _exact_fields(
        value,
        expected=frozenset({"record_ref", "record_sha256", "leading_dimension"}),
        field_name="row_identity",
    )
    if type(payload["leading_dimension"]) is not int or payload["leading_dimension"] < 0:
        raise DirectedTransformV2Error(
            "row_identity.leading_dimension must be an exact nonnegative integer."
        )
    return AuthorityRowIdentity(
        record_ref=_required_text(payload["record_ref"], field_name="row_identity.record_ref"),
        record_sha256=_sha256(
            payload["record_sha256"], field_name="row_identity.record_sha256"
        ),
        leading_dimension=payload["leading_dimension"],
    )


def _parse_inverse(value: Any) -> ExactTransformReference:
    payload = _exact_fields(
        value,
        expected=frozenset({"relationship", "record_ref", "record_sha256"}),
        field_name="inverse_of",
    )
    if payload["relationship"] != "inverse_of":
        raise DirectedTransformV2Error("inverse_of.relationship is invalid.")
    ref = _required_text(payload["record_ref"], field_name="inverse_of.record_ref")
    if not ref.endswith(f"@{DIRECTED_TRANSFORM_V2_ATTR}"):
        raise DirectedTransformV2Error("inverse_of must reference an exact v2 record.")
    return ExactTransformReference(
        relationship="inverse_of",
        record_ref=ref,
        record_sha256=_sha256(
            payload["record_sha256"], field_name="inverse_of.record_sha256"
        ),
    )


def parse_directed_transform_v2(value: Any) -> DirectedTransformV2:
    if isinstance(value, DirectedTransformV2):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise DirectedTransformV2Error("Directed transform must be a mapping.")
    required = {
        "schema_id",
        "schema_version",
        "transform_id",
        "kind",
        "from_space_id",
        "to_space_id",
        "direction",
        "source",
        "target",
        "transform_authority",
        "payload",
        "sampling_formula",
    }
    allowed = required | {"camera_id", "row_identity", "inverse_of"}
    actual = set(value)
    if not required.issubset(actual) or not actual.issubset(allowed):
        raise DirectedTransformV2Error(
            f"Directed-transform fields are invalid; missing={sorted(required-actual)}, "
            f"unknown={sorted(actual-allowed)}."
        )
    if value["schema_id"] != DIRECTED_TRANSFORM_V2_SCHEMA_ID:
        raise DirectedTransformV2Error("Unsupported directed-transform schema_id.")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != DIRECTED_TRANSFORM_V2_SCHEMA_VERSION
    ):
        raise DirectedTransformV2Error(
            "Canonical transform binding requires exact integer schema version 2."
        )
    if value["direction"] != DIRECTED_TRANSFORM_V2_DIRECTION:
        raise DirectedTransformV2Error("Transform direction must be source_to_target.")
    transform_id = _required_text(value["transform_id"], field_name="transform_id")
    if _TRANSFORM_ID_RE.fullmatch(transform_id) is None:
        raise DirectedTransformV2Error("transform_id is not canonical.")
    kind = _required_text(value["kind"], field_name="kind")
    if kind not in DIRECTED_TRANSFORM_V2_KINDS:
        raise DirectedTransformV2Error(f"Unsupported transform kind {kind!r}.")
    source = _parse_endpoint(value["source"], field_name="source")
    target = _parse_endpoint(value["target"], field_name="target")
    from_space = _required_text(value["from_space_id"], field_name="from_space_id")
    to_space = _required_text(value["to_space_id"], field_name="to_space_id")
    if source.space_id != from_space or target.space_id != to_space:
        raise DirectedTransformV2Error("from/to spaces disagree with typed endpoints.")
    row_identity = _parse_identity(value["row_identity"]) if "row_identity" in value else None
    if (kind == AFFINE_2D_ROWWISE_KIND) != (row_identity is not None):
        raise DirectedTransformV2Error("Only rowwise transforms carry row identity.")
    inverse = _parse_inverse(value["inverse_of"]) if "inverse_of" in value else None
    if inverse is not None and kind != HOMOGRAPHY_KIND:
        raise DirectedTransformV2Error("Only homographies may be explicit inverses.")
    formula = _required_text(value["sampling_formula"], field_name="sampling_formula")
    allowed_formulas = {
        HOMOGRAPHY_KIND: {PROJECTIVE_XY_DIRECT_V1},
        AFFINE_2D_CONSTANT_KIND: {
            TRANSLATION_XY_DIRECT_V1,
            NORMALIZED_TO_PIXEL_EDGE_EXTENT_V1,
            NORMALIZED_TO_PIXEL_CENTER_INDEX_V1,
        },
        AFFINE_2D_ROWWISE_KIND: {
            SCALE_XY_EDGE_ALIGNED_V1,
            SCALE_XY_PIXEL_CENTER_V1,
        },
    }
    if formula not in allowed_formulas[kind]:
        raise DirectedTransformV2Error("Transform sampling formula is invalid for its kind.")
    normalized_formula = formula in {
        NORMALIZED_TO_PIXEL_EDGE_EXTENT_V1,
        NORMALIZED_TO_PIXEL_CENTER_INDEX_V1,
    }
    if not normalized_formula and source.pixel_convention != target.pixel_convention:
        raise DirectedTransformV2Error(
            "Current canonical transforms require identical source/target pixel conventions."
        )
    if (formula == SCALE_XY_PIXEL_CENTER_V1) != (source.pixel_convention == "pixel_center"):
        if kind == AFFINE_2D_ROWWISE_KIND:
            raise DirectedTransformV2Error(
                "Crop scale formula conflicts with endpoint pixel convention."
            )
    if normalized_formula:
        if (
            source.units != "normalized"
            or target.units != "px"
            or source.pixel_convention != "continuous"
            or (formula == NORMALIZED_TO_PIXEL_CENTER_INDEX_V1)
            != (target.pixel_convention == "pixel_center")
        ):
            raise DirectedTransformV2Error(
                "Normalized sampling formula conflicts with source/target units or pixel sampling convention."
            )
    if formula == TRANSLATION_XY_DIRECT_V1 and source.units != target.units:
        raise DirectedTransformV2Error(
            "Translation transforms require identical source/target units."
        )
    return DirectedTransformV2(
        transform_id=transform_id,
        kind=kind,
        from_space_id=from_space,
        to_space_id=to_space,
        source=source,
        target=target,
        transform_authority=_parse_authority(value["transform_authority"]),
        payload=_parse_payload(value["payload"]),
        sampling_formula=formula,
        camera_id=(
            _required_text(value["camera_id"], field_name="camera_id")
            if "camera_id" in value
            else None
        ),
        row_identity=row_identity,
        inverse_of=inverse,
    )


def _identity(value: BoundRowIdentityContract | None) -> AuthorityRowIdentity | None:
    if value is None:
        return None
    try:
        value.assert_verified()
    except RowIdentityContractError as exc:
        raise DirectedTransformV2Error(f"Row identity is stale: {exc}") from exc
    return AuthorityRowIdentity(
        record_ref=value.record_ref,
        record_sha256=value.record_sha256,
        leading_dimension=value.leading_dimension,
    )


def _payload(node: Any) -> AuthorityPayload:
    return AuthorityPayload(
        record_ref=f"/{canonical_node_path(node)}@array_values",
        record_sha256=array_values_sha256(node),
        selector="array_values",
    )


def _values(node: Any) -> np.ndarray:
    try:
        values = np.asarray(node[:])
    except Exception as exc:
        raise DirectedTransformV2Error("Unable to read exact transform payload.") from exc
    if values.dtype.hasobject:
        raise DirectedTransformV2Error("Transform payload cannot use object dtype.")
    return np.ascontiguousarray(values)


def _matrix(values: Any) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.str != "<f8" or raw.shape != (3, 3) or not np.isfinite(raw).all():
        raise DirectedTransformV2Error(
            "Matrix must be finite little-endian float64 with shape (3,3)."
        )
    if int(np.linalg.matrix_rank(raw)) != 3:
        raise DirectedTransformV2Error("Matrix must be nonsingular.")
    return raw


def _expected_kind(authority_kind: str) -> str:
    return {
        SELECTED_CALIBRATION_AUTHORITY_KIND: HOMOGRAPHY_KIND,
        MODEL_INPUT_PREPROCESSING_AUTHORITY_KIND: AFFINE_2D_CONSTANT_KIND,
        CROP_PLACEMENT_AUTHORITY_KIND: AFFINE_2D_ROWWISE_KIND,
        ARENA_CANVAS_PLACEMENT_AUTHORITY_KIND: AFFINE_2D_CONSTANT_KIND,
        NORMALIZED_TO_PIXEL_AUTHORITY_KIND: AFFINE_2D_CONSTANT_KIND,
    }[authority_kind]


def _require_archive(
    node: Any,
    authority: BoundTransformAuthority,
    source_frame: BoundPixelFrameAuthority,
    target_frame: BoundPixelFrameAuthority,
    row_identity: BoundRowIdentityContract | None,
    inverse_of: BoundDirectedTransformV2 | None,
) -> ArchiveIdentity:
    try:
        common = archive_identity(node)
        identities = [
            authority.archive_identity,
            source_frame.archive_identity,
            target_frame.archive_identity,
        ]
        if row_identity is not None:
            identities.append(row_identity.archive_identity)
        if inverse_of is not None:
            identities.append(inverse_of.archive_identity)
    except ArchiveIdentityError as exc:
        raise DirectedTransformV2Error(str(exc)) from exc
    if any(item != common for item in identities):
        raise DirectedTransformV2Error(
            "Transform evidence comes from different archives/stores."
        )
    return common


def _validate_payload(transform: DirectedTransformV2, values: np.ndarray) -> None:
    if transform.kind == AFFINE_2D_ROWWISE_KIND:
        if values.ndim != 2 or values.shape[1:] != (4,):
            raise DirectedTransformV2Error("Rowwise payload must have shape (N,4).")
        if not np.issubdtype(values.dtype, np.number):
            raise DirectedTransformV2Error("Rowwise payload must be numeric.")
        numeric = values.astype(np.float64)
        if not np.isfinite(numeric).all() or np.any(numeric[:, 2:] <= 0):
            raise DirectedTransformV2Error("Rowwise xywh values are invalid.")
        assert transform.row_identity is not None
        if values.shape[0] != transform.row_identity.leading_dimension:
            raise DirectedTransformV2Error("Rowwise payload count mismatches identity.")
        return
    matrix = _matrix(values)
    if transform.kind == AFFINE_2D_CONSTANT_KIND and not np.array_equal(
        matrix[2], np.asarray([0.0, 0.0, 1.0])
    ):
        raise DirectedTransformV2Error("Affine bottom row must be exactly [0,0,1].")


INVERSE_COMPOSITION_ATOL = 1e-9


def _validate_inverse_pair(forward_values: Any, inverse_values: Any) -> None:
    """Validate a persisted inverse without platform-dependent reinversion."""

    forward = _matrix(forward_values)
    inverse = _matrix(inverse_values)
    identity = np.eye(3, dtype=np.float64)
    for product in (forward @ inverse, inverse @ forward):
        if not np.allclose(
            product,
            identity,
            rtol=0.0,
            atol=INVERSE_COMPOSITION_ATOL,
            equal_nan=False,
        ):
            raise DirectedTransformV2Error(
                "Persisted forward and inverse matrices do not compose to identity."
            )


def _validate(
    transform: DirectedTransformV2,
    *,
    node: Any,
    authority: BoundTransformAuthority,
    source_frame: BoundPixelFrameAuthority,
    target_frame: BoundPixelFrameAuthority,
    row_identity: BoundRowIdentityContract | None,
    inverse_of: BoundDirectedTransformV2 | None,
) -> np.ndarray:
    _require_archive(node, authority, source_frame, target_frame, row_identity, inverse_of)
    try:
        bound_authority = require_bound_transform_authority(authority)
        source = require_bound_pixel_frame_authority(source_frame)
        target = require_bound_pixel_frame_authority(target_frame)
    except (TransformAuthorityError, PixelFrameAuthorityError) as exc:
        raise DirectedTransformV2Error(f"Transform evidence is stale: {exc}") from exc
    if transform.kind != _expected_kind(bound_authority.record.kind):
        raise DirectedTransformV2Error("Transform kind is not authorized.")
    if transform.source != source.endpoint or transform.target != target.endpoint:
        raise DirectedTransformV2Error("Transform does not use exact typed endpoint frames.")
    if inverse_of is None:
        authority_endpoints_match = (
            bound_authority.source_frame.record_ref == source.record_ref
            and bound_authority.target_frame.record_ref == target.record_ref
            and bound_authority.record.source == source.endpoint
            and bound_authority.record.target == target.endpoint
        )
    else:
        authority_endpoints_match = (
            bound_authority.source_frame.record_ref == target.record_ref
            and bound_authority.target_frame.record_ref == source.record_ref
            and bound_authority.record.source == target.endpoint
            and bound_authority.record.target == source.endpoint
        )
    if not authority_endpoints_match:
        raise DirectedTransformV2Error("Transform endpoints differ from its exact authority.")
    pointer = TransformAuthorityPointer(
        kind=bound_authority.record.kind,
        record_ref=bound_authority.record_ref,
        record_sha256=bound_authority.record_sha256,
    )
    if transform.transform_authority != pointer:
        raise DirectedTransformV2Error("Transform authority pointer is stale.")
    if transform.payload != _payload(node):
        raise DirectedTransformV2Error("Transform payload pointer is stale.")
    if transform.row_identity != _identity(row_identity):
        raise DirectedTransformV2Error("Transform row identity is stale.")
    if transform.camera_id != bound_authority.record.camera_id:
        raise DirectedTransformV2Error("Transform camera differs from its authority.")
    if transform.sampling_formula != bound_authority.record.sampling_formula:
        raise DirectedTransformV2Error("Transform sampling formula differs from its authority.")
    values = _values(node)
    _validate_payload(transform, values)

    if inverse_of is None:
        if transform.inverse_of is not None:
            raise DirectedTransformV2Error("Explicit inverse binding is missing.")
        if transform.payload != bound_authority.record.payload:
            raise DirectedTransformV2Error("Forward payload differs from its authority.")
    else:
        forward = require_bound_directed_transform_v2(inverse_of)
        if forward.transform.inverse_of is not None:
            raise DirectedTransformV2Error("An inverse-of-inverse is forbidden.")
        if bound_authority.record.kind != SELECTED_CALIBRATION_AUTHORITY_KIND:
            raise DirectedTransformV2Error("Only selected homographies support inverse records.")
        expected = ExactTransformReference(
            relationship="inverse_of",
            record_ref=forward.record_ref,
            record_sha256=forward.transform_sha256,
        )
        if transform.inverse_of != expected:
            raise DirectedTransformV2Error("inverse_of does not bind the exact forward record.")
        if forward.authority.record_ref != bound_authority.record_ref:
            raise DirectedTransformV2Error("Forward and inverse authorities differ.")
        if transform.source != forward.transform.target or transform.target != forward.transform.source:
            raise DirectedTransformV2Error("Inverse does not swap exact typed endpoints.")
        _validate_inverse_pair(forward.payload_values, values)
    return values


@dataclass(frozen=True, init=False)
class BoundDirectedTransformV2:
    transform: DirectedTransformV2
    array_path: str
    transform_sha256: str
    attr_name: str
    payload_values: np.ndarray = field(repr=False, compare=False)
    authority: BoundTransformAuthority = field(repr=False, compare=False)
    source_frame: BoundPixelFrameAuthority = field(repr=False, compare=False)
    target_frame: BoundPixelFrameAuthority = field(repr=False, compare=False)
    row_identity: BoundRowIdentityContract | None = field(repr=False, compare=False)
    inverse_of: BoundDirectedTransformV2 | None = field(repr=False, compare=False)
    _archive_identity: ArchiveIdentity = field(repr=False, compare=False)
    _node: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        transform: DirectedTransformV2,
        array_path: str,
        payload_values: np.ndarray,
        authority: BoundTransformAuthority,
        source_frame: BoundPixelFrameAuthority,
        target_frame: BoundPixelFrameAuthority,
        row_identity: BoundRowIdentityContract | None,
        inverse_of: BoundDirectedTransformV2 | None,
        archive: ArchiveIdentity,
        node: Any,
        attr_name: str,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BOUND_TRANSFORM_SEAL:
            raise DirectedTransformV2Error("Bound transforms cannot be constructed directly.")
        values = np.array(payload_values, copy=True, order="C")
        values.setflags(write=False)
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "array_path", array_path)
        object.__setattr__(self, "transform_sha256", transform.digest())
        object.__setattr__(self, "attr_name", attr_name)
        object.__setattr__(self, "payload_values", values)
        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "source_frame", source_frame)
        object.__setattr__(self, "target_frame", target_frame)
        object.__setattr__(self, "row_identity", row_identity)
        object.__setattr__(self, "inverse_of", inverse_of)
        object.__setattr__(self, "_archive_identity", archive)
        object.__setattr__(self, "_node", node)
        object.__setattr__(self, "_seal", _verification_seal)

    @property
    def record_ref(self) -> str:
        return f"/{self.array_path}@{self.attr_name}"

    @property
    def matrix(self) -> np.ndarray:
        if self.transform.kind == AFFINE_2D_ROWWISE_KIND:
            raise DirectedTransformV2Error("Rowwise transform has no constant matrix.")
        return self.payload_values

    @property
    def archive_identity(self) -> ArchiveIdentity:
        return self._archive_identity

    def assert_verified(self) -> None:
        if self._seal is not _BOUND_TRANSFORM_SEAL:
            raise DirectedTransformV2Error("Transform is not sealed evidence.")
        current = load_bound_directed_transform_v2(
            self._node,
            authority=self.authority,
            source_frame=self.source_frame,
            target_frame=self.target_frame,
            row_identity=self.row_identity,
            inverse_of=self.inverse_of,
            attr_name=self.attr_name,
        )
        if (
            current.transform != self.transform
            or current.transform_sha256 != self.transform_sha256
            or current.archive_identity != self.archive_identity
            or not np.array_equal(current.payload_values, self.payload_values)
        ):
            raise DirectedTransformV2Error("Persisted transform changed after binding.")


def load_bound_directed_transform_v2(
    node: Any,
    *,
    authority: BoundTransformAuthority,
    source_frame: BoundPixelFrameAuthority,
    target_frame: BoundPixelFrameAuthority,
    row_identity: BoundRowIdentityContract | None = None,
    inverse_of: BoundDirectedTransformV2 | None = None,
    attr_name: str = DIRECTED_TRANSFORM_V2_ATTR,
) -> BoundDirectedTransformV2:
    if attr_name not in DIRECTED_TRANSFORM_V2_ATTRS:
        raise DirectedTransformV2Error(
            f"Unsupported directed-transform-v2 attr {attr_name!r}."
        )
    archive = _require_archive(node, authority, source_frame, target_frame, row_identity, inverse_of)
    attrs = getattr(node, "attrs", None)
    if not isinstance(attrs, Mapping):
        raise DirectedTransformV2Error("Transform node must expose persisted attrs.")
    historical = sorted(_HISTORICAL_TRANSFORM_ATTRS & set(attrs))
    if historical:
        raise DirectedTransformV2Error(
            f"Canonical transform carries historical attrs {historical!r}."
        )
    digest_attr = f"{attr_name}_sha256"
    raw = attrs.get(attr_name)
    transform = parse_directed_transform_v2(raw)
    specialized = {
        DIRECTED_TRANSFORM_V2_PIXEL_CENTER_ATTR: (
            "pixel_center",
            SCALE_XY_PIXEL_CENTER_V1,
        ),
        DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR: (
            "pixel_edge_half_open",
            SCALE_XY_EDGE_ALIGNED_V1,
        ),
    }.get(attr_name)
    if specialized is not None and (
        transform.kind != AFFINE_2D_ROWWISE_KIND
        or transform.sampling_formula != specialized[1]
        or source_frame.pixel_convention != specialized[0]
    ):
        convention_label = specialized[0].replace("_", "-")
        raise DirectedTransformV2Error(
            f"The {convention_label} directed-transform attr is reserved for a "
            f"{convention_label} rowwise crop-placement transform."
        )
    _require_transform_authority_attr_pair(authority, attr_name=attr_name)
    if not isinstance(raw, Mapping) or not _exact_json_equal(raw, transform.to_dict()):
        raise DirectedTransformV2Error(
            "Raw persisted transform mapping is not its parsed canonical form."
        )
    stored = _sha256(
        attrs.get(digest_attr),
        field_name=digest_attr,
    )
    if stored != transform.digest():
        raise DirectedTransformV2Error("Persisted transform digest is stale.")
    values = _validate(
        transform,
        node=node,
        authority=authority,
        source_frame=source_frame,
        target_frame=target_frame,
        row_identity=row_identity,
        inverse_of=inverse_of,
    )
    return BoundDirectedTransformV2(
        transform=transform,
        array_path=canonical_node_path(node),
        payload_values=values,
        authority=authority,
        source_frame=source_frame,
        target_frame=target_frame,
        row_identity=row_identity,
        inverse_of=inverse_of,
        archive=archive,
        node=node,
        attr_name=attr_name,
        _verification_seal=_BOUND_TRANSFORM_SEAL,
    )


def _validate_migration_eligibility(
    node: Any,
    *,
    raw: Any,
    stored_digest: Any,
    legacy_digest: str,
    transform: DirectedTransformV2,
    source_frame: BoundPixelFrameAuthority,
    target_frame: BoundPixelFrameAuthority,
    selected: Any,
) -> None:
    _require_exact_builtin_json(raw, field_name="migration eligibility")
    payload = _exact_fields(
        raw,
        expected=frozenset(
            {
                "schema_id",
                "schema_version",
                "decision",
                "evidence_basis",
                "legacy_transform",
                "source_frame_authority",
                "target_frame_authority",
                "source_pixel_convention",
                "target_pixel_convention",
                "source_extent_authority",
                "target_extent_authority",
                "canonicalization",
            }
        ),
        field_name="migration eligibility",
    )
    if (
        payload["schema_id"] != MIGRATION_ELIGIBILITY_SCHEMA_ID
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != MIGRATION_ELIGIBILITY_SCHEMA_VERSION
        or payload["decision"] != "eligible"
        or payload["evidence_basis"] != MIGRATION_ELIGIBILITY_BASIS
        or payload["canonicalization"] != DIRECTED_TRANSFORM_V2_CANONICALIZATION
    ):
        raise DirectedTransformV2Error(
            "Migration eligibility is not an exact controlled eligible decision."
        )
    legacy_pointer = _exact_fields(
        payload["legacy_transform"],
        expected=frozenset({"record_ref", "record_sha256"}),
        field_name="migration eligibility legacy_transform",
    )
    source_pointer = _exact_fields(
        payload["source_frame_authority"],
        expected=frozenset({"record_ref", "record_sha256"}),
        field_name="migration eligibility source_frame_authority",
    )
    target_pointer = _exact_fields(
        payload["target_frame_authority"],
        expected=frozenset({"record_ref", "record_sha256"}),
        field_name="migration eligibility target_frame_authority",
    )
    expected_legacy_pointer = {
        "record_ref": f"/{canonical_node_path(node)}@{DIRECTED_TRANSFORM_ATTR}",
        "record_sha256": legacy_digest,
    }
    expected_source_pointer = {
        "record_ref": source_frame.record_ref,
        "record_sha256": source_frame.record_sha256,
    }
    expected_target_pointer = {
        "record_ref": target_frame.record_ref,
        "record_sha256": target_frame.record_sha256,
    }
    if (
        not _exact_json_equal(legacy_pointer, expected_legacy_pointer)
        or not _exact_json_equal(source_pointer, expected_source_pointer)
        or not _exact_json_equal(target_pointer, expected_target_pointer)
    ):
        raise DirectedTransformV2Error(
            "Migration eligibility does not bind the exact legacy record and typed endpoint authorities."
        )
    for name in (
        "source_pixel_convention",
        "target_pixel_convention",
        "source_extent_authority",
        "target_extent_authority",
    ):
        if type(payload[name]) is not str or not payload[name]:
            raise DirectedTransformV2Error(
                f"Migration eligibility {name} must be an exact non-empty string."
            )
    if (
        payload["source_pixel_convention"] != "continuous"
        or payload["target_pixel_convention"] != "continuous"
        or transform.source.pixel_convention != "continuous"
        or transform.target.pixel_convention != "continuous"
        or source_frame.pixel_convention != "continuous"
        or target_frame.pixel_convention != "continuous"
    ):
        raise DirectedTransformV2Error(
            "Migration eligibility admits only explicitly reviewed continuous endpoints; v1 pixel convention is never caller-inferred."
        )
    if (
        payload["source_extent_authority"]
        != selected.source_reference_extent.authority
        or payload["target_extent_authority"]
        != selected.target_reference_extent.authority
    ):
        raise DirectedTransformV2Error(
            "Migration eligibility extent authorities do not equal selected-calibration authority strings."
        )
    canonical = json.loads(_canonical_json(payload))
    if not _exact_json_equal(raw, canonical):
        raise DirectedTransformV2Error(
            "Migration eligibility record is not its exact canonical mapping."
        )
    if type(stored_digest) is not str or stored_digest != _mapping_sha256(canonical):
        raise DirectedTransformV2Error("Migration eligibility digest is stale.")


def validate_migration_only_v1_v2_coexistence(
    node: Any,
    *,
    authority: BoundTransformAuthority,
    source_frame: BoundPixelFrameAuthority,
    target_frame: BoundPixelFrameAuthority,
) -> DirectedTransformV2:
    """Validate an exact v1/v2 homography pair for metadata migration only.

    Normal future writers and readers continue to reject coexistence.  This
    read-only gate exists solely for a migration that has independently proved
    both records describe the same persisted payload, direction, typed
    endpoints, camera, calibration snapshot, and digest.
    """

    attrs = getattr(node, "attrs", None)
    if not isinstance(attrs, Mapping):
        raise DirectedTransformV2Error("Migration node must expose persisted attrs.")
    required_attrs = {
        DIRECTED_TRANSFORM_ATTR,
        f"{DIRECTED_TRANSFORM_ATTR}{DIRECTED_TRANSFORM_DIGEST_SUFFIX}",
        DIRECTED_TRANSFORM_V2_ATTR,
        DIRECTED_TRANSFORM_V2_DIGEST_ATTR,
        MIGRATION_ELIGIBILITY_ATTR,
        MIGRATION_ELIGIBILITY_DIGEST_ATTR,
    }
    if not required_attrs.issubset(attrs):
        raise DirectedTransformV2Error(
            "Migration coexistence requires complete v1 and v2 records/digests plus controlled eligibility evidence."
        )
    legacy_raw = attrs[DIRECTED_TRANSFORM_ATTR]
    _require_exact_legacy_migration_scalars(legacy_raw)
    try:
        legacy = parse_directed_homography(legacy_raw)
    except DirectedTransformError as exc:
        raise DirectedTransformV2Error(f"Legacy migration record is invalid: {exc}") from exc
    if not _exact_json_equal(legacy_raw, legacy.to_dict()):
        raise DirectedTransformV2Error(
            "Legacy migration mapping is not its exact canonical form."
        )
    legacy_digest = attrs[
        f"{DIRECTED_TRANSFORM_ATTR}{DIRECTED_TRANSFORM_DIGEST_SUFFIX}"
    ]
    if (
        type(legacy_digest) is not str
        or legacy_digest != directed_transform_digest(legacy)
    ):
        raise DirectedTransformV2Error("Legacy migration digest is stale.")
    raw = attrs[DIRECTED_TRANSFORM_V2_ATTR]
    _require_exact_builtin_json(raw, field_name="migration v2 record")
    transform = parse_directed_transform_v2(raw)
    if not isinstance(raw, Mapping) or not _exact_json_equal(raw, transform.to_dict()):
        raise DirectedTransformV2Error(
            "Migration v2 mapping is not its exact canonical form."
        )
    v2_digest = attrs[DIRECTED_TRANSFORM_V2_DIGEST_ATTR]
    if type(v2_digest) is not str or v2_digest != transform.digest():
        raise DirectedTransformV2Error("Migration v2 digest is stale.")
    if transform.kind != HOMOGRAPHY_KIND or transform.inverse_of is not None:
        raise DirectedTransformV2Error(
            "Migration coexistence currently admits forward homographies only."
        )
    if legacy.source_transform is not None:
        raise DirectedTransformV2Error(
            "Migration coexistence rejects implicit/legacy inverse records."
        )
    try:
        values = _validate(
            transform,
            node=node,
            authority=authority,
            source_frame=source_frame,
            target_frame=target_frame,
            row_identity=None,
            inverse_of=None,
        )
    except ValueError as exc:
        raise DirectedTransformV2Error(
            f"Migration authority or typed endpoint evidence is stale: {exc}"
        ) from exc
    if (
        legacy.from_space_id != transform.from_space_id
        or legacy.to_space_id != transform.to_space_id
        or legacy.camera_id != transform.camera_id
        or legacy.source_reference_extent.width != transform.source.width
        or legacy.source_reference_extent.height != transform.source.height
        or legacy.target_reference_extent.width != transform.target.width
        or legacy.target_reference_extent.height != transform.target.height
        or legacy.source_reference_extent.units != "px"
        or legacy.target_reference_extent.units != "px"
        or legacy.matrix_sha256 != homography_matrix_sha256(values)
    ):
        raise DirectedTransformV2Error(
            "Migration v1/v2 direction, endpoints, camera, extent, or payload disagree."
        )
    manifest_pointer = authority.record.semantics.get(
        "selected_calibration_manifest"
    )
    try:
        from fisheye.shared.selected_calibration import (
            require_bound_selected_calibration_snapshot,
        )

        selected = require_bound_selected_calibration_snapshot(
            authority._selected_calibration_snapshot
        )
    except Exception as exc:
        raise DirectedTransformV2Error(
            "Migration requires the exact sealed selected-calibration snapshot."
        ) from exc
    if (
        authority.record.kind != SELECTED_CALIBRATION_AUTHORITY_KIND
        or not isinstance(manifest_pointer, Mapping)
        or manifest_pointer
        != {
            "record_ref": selected.manifest_record_ref,
            "record_sha256": selected.manifest_sha256,
        }
        or legacy.calibration_ref != selected.manifest.camera_calibration_ref
        or legacy.source_reference_extent.authority
        != selected.source_reference_extent.authority
        or legacy.target_reference_extent.authority
        != selected.target_reference_extent.authority
    ):
        raise DirectedTransformV2Error(
            "Migration legacy calibration_ref or extent authority strings do not equal the exact selected calibration lineage."
        )
    _validate_migration_eligibility(
        node,
        raw=attrs[MIGRATION_ELIGIBILITY_ATTR],
        stored_digest=attrs[MIGRATION_ELIGIBILITY_DIGEST_ATTR],
        legacy_digest=legacy_digest,
        transform=transform,
        source_frame=source_frame,
        target_frame=target_frame,
        selected=selected,
    )
    _require_archive(
        node,
        authority,
        source_frame,
        target_frame,
        None,
        None,
    )
    return transform


def require_bound_directed_transform_v2(value: Any) -> BoundDirectedTransformV2:
    if type(value) is not BoundDirectedTransformV2 or value._seal is not _BOUND_TRANSFORM_SEAL:
        raise DirectedTransformV2Error("A sealed bound v2 transform is required.")
    value.assert_verified()
    return value


def _record(
    *,
    transform_id: str,
    node: Any,
    authority: BoundTransformAuthority,
    source_frame: BoundPixelFrameAuthority,
    target_frame: BoundPixelFrameAuthority,
    row_identity: BoundRowIdentityContract | None,
) -> DirectedTransformV2:
    bound = require_bound_transform_authority(authority)
    source = require_bound_pixel_frame_authority(source_frame)
    target = require_bound_pixel_frame_authority(target_frame)
    return DirectedTransformV2(
        transform_id=_required_text(transform_id, field_name="transform_id"),
        kind=_expected_kind(bound.record.kind),
        from_space_id=source.space_id,
        to_space_id=target.space_id,
        source=source.endpoint,
        target=target.endpoint,
        transform_authority=TransformAuthorityPointer(
            kind=bound.record.kind,
            record_ref=bound.record_ref,
            record_sha256=bound.record_sha256,
        ),
        payload=_payload(node),
        sampling_formula=bound.record.sampling_formula,
        camera_id=bound.record.camera_id,
        row_identity=_identity(row_identity),
        inverse_of=None,
    )


def _stamp(
    node: Any,
    *,
    transform: DirectedTransformV2,
    authority: BoundTransformAuthority,
    source_frame: BoundPixelFrameAuthority,
    target_frame: BoundPixelFrameAuthority,
    row_identity: BoundRowIdentityContract | None,
    inverse_of: BoundDirectedTransformV2 | None,
    attr_name: str = DIRECTED_TRANSFORM_V2_ATTR,
) -> BoundDirectedTransformV2:
    if attr_name not in DIRECTED_TRANSFORM_V2_ATTRS:
        raise DirectedTransformV2Error(
            f"Unsupported directed-transform-v2 attr {attr_name!r}."
        )
    try:
        attrs = require_trusted_coordinate_attrs(node, label="Directed transform")
    except PixelFrameAuthorityError as exc:
        raise DirectedTransformV2Error(str(exc)) from exc
    historical = sorted(_HISTORICAL_TRANSFORM_ATTRS & set(attrs))
    if historical:
        raise DirectedTransformV2Error(
            f"Canonical transform publication refuses historical attrs {historical!r}."
        )
    parsed = parse_directed_transform_v2(transform)
    specialized = {
        DIRECTED_TRANSFORM_V2_PIXEL_CENTER_ATTR: (
            "pixel_center",
            SCALE_XY_PIXEL_CENTER_V1,
        ),
        DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR: (
            "pixel_edge_half_open",
            SCALE_XY_EDGE_ALIGNED_V1,
        ),
    }.get(attr_name)
    if specialized is not None and (
        parsed.kind != AFFINE_2D_ROWWISE_KIND
        or parsed.sampling_formula != specialized[1]
        or source_frame.pixel_convention != specialized[0]
    ):
        convention_label = specialized[0].replace("_", "-")
        raise DirectedTransformV2Error(
            f"The {convention_label} directed-transform attr is reserved for a "
            f"{convention_label} rowwise crop-placement transform."
        )
    _require_transform_authority_attr_pair(authority, attr_name=attr_name)
    _validate(
        parsed,
        node=node,
        authority=authority,
        source_frame=source_frame,
        target_frame=target_frame,
        row_identity=row_identity,
        inverse_of=inverse_of,
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
            label="Directed-transform-v2 post-write",
        )
        bound = load_bound_directed_transform_v2(
            node,
            authority=authority,
            source_frame=source_frame,
            target_frame=target_frame,
            row_identity=row_identity,
            inverse_of=inverse_of,
            attr_name=attr_name,
        )
        _require_exact_attrs_state(
            require_trusted_coordinate_attrs(
                node,
                label="Reloaded directed transform",
            ),
            expected,
            label="Directed-transform-v2 post-reload",
        )
        return bound
    except Exception as exc:
        try:
            _restore_exact_attrs(attrs, snapshot)
        except Exception as rollback_exc:  # pragma: no cover
            raise DirectedTransformV2Error(
                f"Transform stamp failed and rollback was incomplete: {rollback_exc}"
            ) from exc
        if isinstance(exc, DirectedTransformV2Error):
            raise
        raise DirectedTransformV2Error(f"Transform stamp failed: {exc}") from exc


def stamp_directed_transform_v2(
    node: Any,
    *,
    transform_id: str,
    authority: BoundTransformAuthority,
    source_frame: BoundPixelFrameAuthority,
    target_frame: BoundPixelFrameAuthority,
    row_identity: BoundRowIdentityContract | None = None,
    attr_name: str = DIRECTED_TRANSFORM_V2_ATTR,
) -> BoundDirectedTransformV2:
    transform = _record(
        transform_id=transform_id,
        node=node,
        authority=authority,
        source_frame=source_frame,
        target_frame=target_frame,
        row_identity=row_identity,
    )
    return _stamp(
        node,
        transform=transform,
        authority=authority,
        source_frame=source_frame,
        target_frame=target_frame,
        row_identity=row_identity,
        inverse_of=None,
        attr_name=attr_name,
    )


def stamp_explicit_inverse_directed_transform_v2(
    inverse_node: Any,
    *,
    transform_id: str,
    forward: BoundDirectedTransformV2,
) -> BoundDirectedTransformV2:
    bound = require_bound_directed_transform_v2(forward)
    if bound.transform.kind != HOMOGRAPHY_KIND:
        raise DirectedTransformV2Error("Only homographies support explicit inverse records.")
    if bound.transform.inverse_of is not None:
        raise DirectedTransformV2Error("An explicit inverse cannot be inverted again.")
    inverse = DirectedTransformV2(
        transform_id=_required_text(transform_id, field_name="transform_id"),
        kind=HOMOGRAPHY_KIND,
        from_space_id=bound.transform.to_space_id,
        to_space_id=bound.transform.from_space_id,
        source=bound.transform.target,
        target=bound.transform.source,
        transform_authority=bound.transform.transform_authority,
        payload=_payload(inverse_node),
        sampling_formula=bound.transform.sampling_formula,
        camera_id=bound.transform.camera_id,
        row_identity=None,
        inverse_of=ExactTransformReference(
            relationship="inverse_of",
            record_ref=bound.record_ref,
            record_sha256=bound.transform_sha256,
        ),
    )
    return _stamp(
        inverse_node,
        transform=inverse,
        authority=bound.authority,
        source_frame=bound.target_frame,
        target_frame=bound.source_frame,
        row_identity=None,
        inverse_of=bound,
    )


def _numeric_points(points_xy: Any) -> np.ndarray:
    try:
        points = np.asarray(points_xy, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DirectedTransformV2Error("Points must contain numeric XY values.") from exc
    if points.ndim < 1 or points.shape[-1] != 2 or not np.isfinite(points).all():
        raise DirectedTransformV2Error("Points must be finite with shape (...,2).")
    return points


def _apply_matrix(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    shape = points.shape
    flat = points.reshape(-1, 2)
    homogeneous = np.column_stack((flat, np.ones(flat.shape[0], dtype=np.float64)))
    projected = (matrix @ homogeneous.T).T
    if np.any(np.abs(projected[:, 2]) <= 1e-12):
        raise DirectedTransformV2Error("Transform produced near-zero homogeneous w.")
    result = projected[:, :2] / projected[:, 2, None]
    if not np.isfinite(result).all():
        raise DirectedTransformV2Error("Transform produced non-finite coordinates.")
    return result.reshape(shape)


def _same_identity(left: BoundRowIdentityContract, right: BoundRowIdentityContract) -> bool:
    return (
        left.archive_identity == right.archive_identity
        and left.record_ref == right.record_ref
        and left.record_sha256 == right.record_sha256
        and left.rowset_path == right.rowset_path
        and left.key_array_path == right.key_array_path
        and left.leading_dimension == right.leading_dimension
    )


def apply_bound_directed_transform_v2(
    points_xy: Any,
    transform: BoundDirectedTransformV2,
    *,
    row_identity: BoundRowIdentityContract | None = None,
) -> np.ndarray:
    bound = require_bound_directed_transform_v2(transform)
    points = _numeric_points(points_xy)
    if bound.transform.kind != AFFINE_2D_ROWWISE_KIND:
        if row_identity is not None:
            raise DirectedTransformV2Error("Constant transform rejects row identity.")
        return _apply_matrix(points, _matrix(bound.payload_values))
    if row_identity is None or bound.row_identity is None:
        raise DirectedTransformV2Error("Rowwise transform requires exact row identity.")
    try:
        row_identity.assert_verified()
    except RowIdentityContractError as exc:
        raise DirectedTransformV2Error(f"Application identity is stale: {exc}") from exc
    if not _same_identity(row_identity, bound.row_identity):
        raise DirectedTransformV2Error("Application identity differs from transform identity.")
    placements = bound.payload_values.astype(np.float64)
    if points.ndim < 2 or points.shape[0] != placements.shape[0]:
        raise DirectedTransformV2Error("Rowwise points must match placement rows.")
    scales = placements[:, 2:] / np.asarray(
        [bound.transform.source.width, bound.transform.source.height], dtype=np.float64
    )
    offsets = placements[:, :2]
    broadcast = (placements.shape[0],) + (1,) * (points.ndim - 2) + (2,)
    if bound.transform.sampling_formula == SCALE_XY_PIXEL_CENTER_V1:
        return (
            (points + 0.5) * scales.reshape(broadcast)
            + offsets.reshape(broadcast)
            - 0.5
        )
    if bound.transform.sampling_formula != SCALE_XY_EDGE_ALIGNED_V1:
        raise DirectedTransformV2Error("Unsupported rowwise sampling formula.")
    return points * scales.reshape(broadcast) + offsets.reshape(broadcast)


__all__ = [
    "AFFINE_2D_CONSTANT_KIND",
    "AFFINE_2D_ROWWISE_KIND",
    "DIRECTED_TRANSFORM_V2_ATTR",
    "DIRECTED_TRANSFORM_V2_ATTRS",
    "DIRECTED_TRANSFORM_V2_CANONICALIZATION",
    "DIRECTED_TRANSFORM_V2_DIGEST_ATTR",
    "DIRECTED_TRANSFORM_V2_DIRECTION",
    "DIRECTED_TRANSFORM_V2_KINDS",
    "DIRECTED_TRANSFORM_V2_PIXEL_CENTER_ATTR",
    "DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR",
    "DIRECTED_TRANSFORM_V2_SCHEMA_ID",
    "DIRECTED_TRANSFORM_V2_SCHEMA_VERSION",
    "HOMOGRAPHY_KIND",
    "MIGRATION_ELIGIBILITY_ATTR",
    "MIGRATION_ELIGIBILITY_BASIS",
    "MIGRATION_ELIGIBILITY_DIGEST_ATTR",
    "MIGRATION_ELIGIBILITY_SCHEMA_ID",
    "MIGRATION_ELIGIBILITY_SCHEMA_VERSION",
    "BoundDirectedTransformV2",
    "DirectedTransformV2",
    "DirectedTransformV2Error",
    "ExactTransformReference",
    "TransformAuthorityPointer",
    "apply_bound_directed_transform_v2",
    "load_bound_directed_transform_v2",
    "validate_migration_only_v1_v2_coexistence",
    "parse_directed_transform_v2",
    "require_bound_directed_transform_v2",
    "stamp_directed_transform_v2",
    "stamp_explicit_inverse_directed_transform_v2",
]
