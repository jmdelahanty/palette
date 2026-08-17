"""Versioned, modality-neutral anatomy profiles and source bindings.

This module owns the semantic contract between an anatomy profile and a
measurement source.  It deliberately does not evaluate expressions.  Numeric
evaluators and materialized position publishers are later consumers of the
canonical records validated here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import copy
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

from fisheye.shared.pose_schema import (
    normalize_ordered_skeleton_edges,
    schema_payload_from_package,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)


ANATOMY_PROFILE_SCHEMA_ID = "palette.anatomy_profile"
ANATOMY_PROFILE_SCHEMA_VERSION = 1
SOURCE_BINDING_SCHEMA_ID = "palette.anatomy_source_role_binding"
SOURCE_BINDING_SCHEMA_VERSION = 1
SUPPORTED_MODALITIES = ("keypoint", "subject_mask")
SUPPORTED_RECIPE_KINDS = ("point", "axis")
SUPPORTED_POINT_OPERATIONS = (
    "role_point",
    "midpoint",
    "mean_points",
    "bbox_centroid",
)
SUPPORTED_AXIS_OPERATIONS = ("axis",)
KEYPOINT_SKELETON_SEMANTICS_SCHEMA_ID = "palette.keypoint.skeleton_semantics"
KEYPOINT_SKELETON_SEMANTICS_SCHEMA_VERSION = 1

_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AnatomyProfileError(ValueError):
    """Raised when an anatomy profile or source binding is invalid."""


@dataclass(frozen=True)
class AnatomyRole:
    """One stable, modality-neutral biological role."""

    role_id: str
    description: str


@dataclass(frozen=True)
class AnatomyRecipe:
    """One named expression record; expressions are not evaluated here."""

    recipe_id: str
    kind: str
    required_roles: tuple[str, ...]
    expression: Mapping[str, Any]
    description: str | None


@dataclass(frozen=True)
class AnatomyProfile:
    """Validated canonical anatomy profile."""

    profile_id: str
    profile_version: int
    roles: tuple[AnatomyRole, ...]
    recipes: tuple[AnatomyRecipe, ...]
    source_bindings: tuple[Mapping[str, Any], ...]
    profile_sha256: str
    payload: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AnatomyProfile":
        normalized = validate_anatomy_profile(value)
        return _profile_from_normalized(normalized)

    @classmethod
    def from_json(cls, path: str | Path) -> "AnatomyProfile":
        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AnatomyProfileError(f"Unable to read anatomy profile {source}: {exc}") from exc
        return cls.from_mapping(value)

    def recipe(self, recipe_id: str) -> AnatomyRecipe:
        for recipe in self.recipes:
            if recipe.recipe_id == recipe_id:
                return recipe
        raise AnatomyProfileError(
            f"Unknown recipe {recipe_id!r} for anatomy profile {self.profile_id!r}."
        )

    def binding(self, binding_id: str) -> dict[str, Any]:
        for binding in self.source_bindings:
            if binding["binding_id"] == binding_id:
                return _thaw_json(binding)
        raise AnatomyProfileError(
            f"Unknown source binding {binding_id!r} for anatomy profile {self.profile_id!r}."
        )


def _fail(path: str, message: str) -> None:
    raise AnatomyProfileError(f"{path}: {message}")


def _mapping(value: object, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "expected a JSON object")
    return dict(value)


def _sequence(value: object, *, path: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(path, "expected a JSON array")
    return list(value)


def _text(value: object, *, path: str, identifier: bool = False) -> str:
    if type(value) is not str or not value:
        _fail(path, "expected a non-empty string")
    if identifier and _ID_RE.fullmatch(value) is None:
        _fail(path, f"invalid stable identifier {value!r}")
    return value


def _positive_int(value: object, *, path: str) -> int:
    if type(value) is not int or value <= 0:
        _fail(path, "expected a positive integer")
    return value


def _sha256(value: object, *, path: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(path, "expected a lowercase SHA-256 digest")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
    path: str,
) -> None:
    allowed = required | optional
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        _fail(path, f"unknown fields: {', '.join(unknown)}")
    if missing:
        _fail(path, f"missing fields: {', '.join(missing)}")


def _sorted_unique_strings(
    value: object,
    *,
    path: str,
    identifier: bool = True,
    require_at_least: int = 0,
) -> tuple[str, ...]:
    values = _sequence(value, path=path)
    if len(values) < require_at_least:
        _fail(path, f"requires at least {require_at_least} entries")
    normalized = tuple(
        _text(item, path=f"{path}[{index}]", identifier=identifier)
        for index, item in enumerate(values)
    )
    if len(set(normalized)) != len(normalized):
        _fail(path, "entries must be unique")
    return tuple(sorted(normalized))


def _canonical_expression(value: Mapping[str, Any], *, path: str) -> dict[str, Any]:
    """Validate and canonicalize a point/axis expression record."""

    expression = _mapping(value, path=path)
    op = _text(expression.get("op"), path=f"{path}.op", identifier=True)
    if op == "role_point":
        _exact_keys(
            expression,
            required={"op", "role_id"},
            optional=set(),
            path=path,
        )
        return {
            "op": op,
            "role_id": _text(
                expression["role_id"],
                path=f"{path}.role_id",
                identifier=True,
            ),
        }

    if op in {"midpoint", "mean_points"}:
        required = {"op", "points"}
        optional = {"weighting"} if op == "mean_points" else set()
        _exact_keys(expression, required=required, optional=optional, path=path)
        points = _sequence(expression["points"], path=f"{path}.points")
        if len(points) < (2 if op == "midpoint" else 2):
            _fail(f"{path}.points", "requires at least two point expressions")
        canonical_points = [
            _canonical_expression(
                _mapping(point, path=f"{path}.points[{index}]"),
                path=f"{path}.points[{index}]",
            )
            for index, point in enumerate(points)
        ]
        if any(point.get("op") == "axis" for point in canonical_points):
            _fail(f"{path}.points", "point expressions cannot contain an axis")
        point_roles = _expression_roles(canonical_points)
        if len(point_roles) != len(set(point_roles)):
            _fail(f"{path}.points", "a recipe cannot use one anatomy role more than once")
        if op == "midpoint" and len(canonical_points) != 2:
            _fail(f"{path}.points", "midpoint requires exactly two point expressions")
        result: dict[str, Any] = {"op": op}
        if op == "mean_points":
            weighting = expression.get("weighting")
            if weighting != "equal_per_point":
                _fail(
                    f"{path}.weighting",
                    "version 1 supports only 'equal_per_point'",
                )
            result["weighting"] = weighting
        result["points"] = sorted(canonical_points, key=_canonical_json_sort_key)
        return result

    if op == "bbox_centroid":
        _exact_keys(
            expression,
            required={"op", "array_ref"},
            optional=set(),
            path=path,
        )
        return {
            "op": op,
            "array_ref": _text(expression["array_ref"], path=f"{path}.array_ref"),
        }

    if op == "axis":
        _exact_keys(
            expression,
            required={"op", "from", "to"},
            optional=set(),
            path=path,
        )
        from_expression = _canonical_expression(
            _mapping(expression["from"], path=f"{path}.from"),
            path=f"{path}.from",
        )
        to_expression = _canonical_expression(
            _mapping(expression["to"], path=f"{path}.to"),
            path=f"{path}.to",
        )
        if from_expression.get("op") == "axis" or to_expression.get("op") == "axis":
            _fail(path, "axis endpoints must be point expressions")
        return {"op": op, "from": from_expression, "to": to_expression}

    _fail(f"{path}.op", f"unsupported operation {op!r}")


def _expression_roles(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        if value.get("op") == "role_point":
            return (str(value["role_id"]),)
        if value.get("op") in {"midpoint", "mean_points"}:
            roles: list[str] = []
            for point in value["points"]:
                roles.extend(_expression_roles(point))
            return tuple(roles)
        if value.get("op") == "axis":
            return _expression_roles(value["from"]) + _expression_roles(value["to"])
    return ()


def _canonical_json_sort_key(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _source_labels(value: object, *, path: str) -> list[str]:
    labels = _sequence(value, path=path)
    labels_text = [
        _text(label, path=f"{path}[{index}]") for index, label in enumerate(labels)
    ]
    if not labels_text:
        _fail(path, "must contain at least one source label")
    if len(set(labels_text)) != len(labels_text):
        _fail(path, "source labels must be unique")
    return labels_text


def _canonical_keypoint_skeleton_semantics(
    value: Mapping[str, Any],
    *,
    path: str,
) -> dict[str, Any]:
    document = dict(value)
    _exact_keys(
        document,
        required={
            "schema_id",
            "schema_version",
            "skeleton_id",
            "kpt_shape",
            "keypoint_labels",
            "nodes",
            "edges",
            "heading_computation",
            "heading_computation_source",
        },
        optional=set(),
        path=path,
    )
    if document["schema_id"] != KEYPOINT_SKELETON_SEMANTICS_SCHEMA_ID:
        _fail(
            f"{path}.schema_id",
            f"expected {KEYPOINT_SKELETON_SEMANTICS_SCHEMA_ID!r}",
        )
    if document["schema_version"] != KEYPOINT_SKELETON_SEMANTICS_SCHEMA_VERSION:
        _fail(
            f"{path}.schema_version",
            f"expected {KEYPOINT_SKELETON_SEMANTICS_SCHEMA_VERSION}",
        )
    skeleton_id = _text(
        document["skeleton_id"], path=f"{path}.skeleton_id", identifier=True
    )
    shape = _sequence(document["kpt_shape"], path=f"{path}.kpt_shape")
    if (
        len(shape) != 2
        or type(shape[0]) is not int
        or shape[0] <= 0
        or type(shape[1]) is not int
        or shape[1] != 2
    ):
        _fail(f"{path}.kpt_shape", "expected exact positive [K, 2] shape")
    labels = _source_labels(
        document["keypoint_labels"], path=f"{path}.keypoint_labels"
    )
    if len(labels) != shape[0]:
        _fail(
            f"{path}.keypoint_labels",
            "cardinality does not match kpt_shape",
        )
    nodes = _sequence(document["nodes"], path=f"{path}.nodes")
    expected_nodes = [
        {"id": index, "name": label} for index, label in enumerate(labels)
    ]
    if nodes != expected_nodes:
        _fail(f"{path}.nodes", "must exactly enumerate the ordered labels")
    try:
        normalize_ordered_skeleton_edges(
            document["edges"],
            n_keypoints=shape[0],
            field=f"{path}.edges",
        )
    except ValueError as exc:
        _fail(f"{path}.edges", str(exc))
    heading = _mapping(
        document["heading_computation"],
        path=f"{path}.heading_computation",
    )
    heading_source = _text(
        document["heading_computation_source"],
        path=f"{path}.heading_computation_source",
    )
    return _canonicalize_json_value(
        {
            "schema_id": KEYPOINT_SKELETON_SEMANTICS_SCHEMA_ID,
            "schema_version": KEYPOINT_SKELETON_SEMANTICS_SCHEMA_VERSION,
            "skeleton_id": skeleton_id,
            "kpt_shape": shape,
            "keypoint_labels": labels,
            "nodes": nodes,
            "edges": document["edges"],
            "heading_computation": heading,
            "heading_computation_source": heading_source,
        }
    )


def _canonical_source_schema(
    value: Mapping[str, Any],
    *,
    path: str,
    pose_schema_base_dir: Path | None = None,
) -> dict[str, Any]:
    source_schema = dict(value)
    authority = _text(
        source_schema.get("authority"),
        path=f"{path}.authority",
        identifier=True,
    )
    if authority == "pose_schema_package":
        _exact_keys(
            source_schema,
            required={
                "authority",
                "modality",
                "package_name",
                "package_payload",
                "package_sha256",
            },
            optional=set(),
            path=path,
        )
        if source_schema["modality"] != "keypoint":
            _fail(
                f"{path}.modality",
                "pose-schema package authority requires keypoint modality",
            )
        package_name = _text(
            source_schema["package_name"],
            path=f"{path}.package_name",
            identifier=True,
        )
        package_payload = _canonicalize_json_value(
            _mapping(source_schema["package_payload"], path=f"{path}.package_payload")
        )
        expected_package_digest = canonical_json_sha256(package_payload)
        actual_package_digest = _sha256(
            source_schema["package_sha256"], path=f"{path}.package_sha256"
        )
        if actual_package_digest != expected_package_digest:
            _fail(
                f"{path}.package_sha256",
                "stale canonical pose-schema package payload digest",
            )
        try:
            _schema, authoritative_payload = schema_payload_from_package(
                package_name,
                base_dir=pose_schema_base_dir,
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            _fail(path, f"unable to resolve authoritative pose-schema package: {exc}")
        authoritative_payload = _canonicalize_json_value(authoritative_payload)
        if package_payload != authoritative_payload:
            _fail(
                f"{path}.package_payload",
                "does not match the exact current pose-schema package payload",
            )
        _source_labels(
            package_payload.get("keypoint_labels"),
            path=f"{path}.package_payload.keypoint_labels",
        )
        if package_payload.get("name") != package_name:
            _fail(
                f"{path}.package_payload.name",
                "does not match package_name",
            )
        return _canonicalize_json_value(
            {
                "authority": authority,
                "modality": "keypoint",
                "package_name": package_name,
                "package_payload": package_payload,
                "package_sha256": actual_package_digest,
            }
        )

    if authority == "keypoint_skeleton_semantics":
        _exact_keys(
            source_schema,
            required={
                "authority",
                "modality",
                "skeleton_document",
                "skeleton_sha256",
            },
            optional=set(),
            path=path,
        )
        if source_schema["modality"] != "keypoint":
            _fail(
                f"{path}.modality",
                "keypoint skeleton authority requires keypoint modality",
            )
        skeleton_document = _canonical_keypoint_skeleton_semantics(
            _mapping(
                source_schema["skeleton_document"],
                path=f"{path}.skeleton_document",
            ),
            path=f"{path}.skeleton_document",
        )
        expected_digest = canonical_json_sha256(skeleton_document)
        actual_digest = _sha256(
            source_schema["skeleton_sha256"],
            path=f"{path}.skeleton_sha256",
        )
        if actual_digest != expected_digest:
            _fail(
                f"{path}.skeleton_sha256",
                "stale canonical skeleton-semantics document digest",
            )
        return _canonicalize_json_value(
            {
                "authority": authority,
                "modality": "keypoint",
                "skeleton_document": skeleton_document,
                "skeleton_sha256": actual_digest,
            }
        )

    if authority != "declared_schema":
        _fail(f"{path}.authority", f"unsupported source-schema authority {authority!r}")
    _exact_keys(
        source_schema,
        required={
            "authority",
            "schema_id",
            "schema_version",
            "modality",
            "labels",
            "schema_sha256",
        },
        optional=set(),
        path=path,
    )
    _text(source_schema["schema_id"], path=f"{path}.schema_id", identifier=True)
    _positive_int(source_schema["schema_version"], path=f"{path}.schema_version")
    modality = _text(source_schema["modality"], path=f"{path}.modality", identifier=True)
    if modality != "subject_mask":
        _fail(
            f"{path}.modality",
            "declared-schema authority is supported only for subject_mask modality",
        )
    source_schema["labels"] = _source_labels(
        source_schema["labels"], path=f"{path}.labels"
    )
    expected_digest = canonical_json_sha256(_without_digest(source_schema, "schema_sha256"))
    actual_digest = _sha256(source_schema["schema_sha256"], path=f"{path}.schema_sha256")
    if actual_digest != expected_digest:
        _fail(
            f"{path}.schema_sha256",
            f"stale digest; expected {expected_digest}, got {actual_digest}",
        )
    source_schema["schema_id"] = str(source_schema["schema_id"])
    source_schema["schema_version"] = int(source_schema["schema_version"])
    source_schema["modality"] = modality
    return source_schema


def _canonical_source_binding(
    profile: AnatomyProfile | Mapping[str, Any],
    value: Mapping[str, Any],
    *,
    path: str,
    pose_schema_base_dir: Path | None = None,
) -> dict[str, Any]:
    # ``validate_anatomy_profile`` calls this helper while it is assembling
    # the profile digest.  Do not recursively validate the profile here; the
    # public entry point validates a mapping before it reaches this helper.
    profile_mapping = (
        _thaw_json(profile.payload)
        if isinstance(profile, AnatomyProfile)
        else dict(profile)
    )
    profile_id = str(profile_mapping["profile_id"])
    profile_version = int(profile_mapping["profile_version"])
    role_ids = {str(role["role_id"]) for role in profile_mapping["roles"]}
    recipes = {str(recipe["recipe_id"]): recipe for recipe in profile_mapping["recipes"]}

    binding = dict(value)
    _exact_keys(
        binding,
        required={
            "schema_id",
            "schema_version",
            "binding_id",
            "profile_id",
            "profile_version",
            "source_schema",
            "role_bindings",
            "advertised_recipe_ids",
            "binding_sha256",
        },
        optional={"source_local_compatibility"},
        path=path,
    )
    if binding["schema_id"] != SOURCE_BINDING_SCHEMA_ID:
        _fail(f"{path}.schema_id", f"expected {SOURCE_BINDING_SCHEMA_ID!r}")
    if (
        type(binding["schema_version"]) is not int
        or binding["schema_version"] != SOURCE_BINDING_SCHEMA_VERSION
    ):
        _fail(f"{path}.schema_version", f"expected {SOURCE_BINDING_SCHEMA_VERSION}")
    _text(binding["binding_id"], path=f"{path}.binding_id", identifier=True)
    if binding["profile_id"] != profile_id:
        _fail(f"{path}.profile_id", "does not match the anatomy profile")
    if type(binding["profile_version"]) is not int or binding["profile_version"] != profile_version:
        _fail(f"{path}.profile_version", "does not match the anatomy profile")

    source_schema = _canonical_source_schema(
        _mapping(binding["source_schema"], path=f"{path}.source_schema"),
        path=f"{path}.source_schema",
        pose_schema_base_dir=pose_schema_base_dir,
    )
    if source_schema["authority"] == "pose_schema_package":
        source_labels = set(source_schema["package_payload"]["keypoint_labels"])
    elif source_schema["authority"] == "keypoint_skeleton_semantics":
        source_labels = set(
            source_schema["skeleton_document"]["keypoint_labels"]
        )
    else:
        source_labels = set(source_schema["labels"])
    role_bindings = _sequence(binding["role_bindings"], path=f"{path}.role_bindings")
    normalized_bindings: list[dict[str, str]] = []
    bound_roles: set[str] = set()
    bound_labels: set[str] = set()
    for index, raw_mapping in enumerate(role_bindings):
        mapping = _mapping(raw_mapping, path=f"{path}.role_bindings[{index}]")
        _exact_keys(
            mapping,
            required={"role_id", "source_label"},
            optional=set(),
            path=f"{path}.role_bindings[{index}]",
        )
        role_id = _text(
            mapping["role_id"],
            path=f"{path}.role_bindings[{index}].role_id",
            identifier=True,
        )
        source_label = _text(
            mapping["source_label"],
            path=f"{path}.role_bindings[{index}].source_label",
        )
        if role_id not in role_ids:
            _fail(f"{path}.role_bindings[{index}].role_id", f"unknown anatomy role {role_id!r}")
        if source_label not in source_labels:
            _fail(
                f"{path}.role_bindings[{index}].source_label",
                f"source label {source_label!r} is not declared by the source schema",
            )
        if role_id in bound_roles:
            _fail(f"{path}.role_bindings", f"duplicate anatomy role mapping {role_id!r}")
        if source_label in bound_labels:
            _fail(f"{path}.role_bindings", f"duplicate source-label mapping {source_label!r}")
        bound_roles.add(role_id)
        bound_labels.add(source_label)
        normalized_bindings.append({"role_id": role_id, "source_label": source_label})
    normalized_bindings.sort(key=lambda item: item["role_id"])

    advertised = _sorted_unique_strings(
        binding["advertised_recipe_ids"],
        path=f"{path}.advertised_recipe_ids",
    )
    for recipe_id in advertised:
        recipe = recipes.get(recipe_id)
        if recipe is None:
            _fail(f"{path}.advertised_recipe_ids", f"unknown recipe {recipe_id!r}")
        required_roles = set(recipe["required_roles"])
        missing = sorted(required_roles - bound_roles)
        if missing:
            _fail(
                f"{path}.advertised_recipe_ids",
                f"recipe {recipe_id!r} is unsupported; missing roles: {', '.join(missing)}",
            )

    if "source_local_compatibility" in binding:
        compatibility = _mapping(
            binding["source_local_compatibility"],
            path=f"{path}.source_local_compatibility",
        )
        _exact_keys(
            compatibility,
            required={"kind", "metadata_path", "authority"},
            optional=set(),
            path=f"{path}.source_local_compatibility",
        )
        if compatibility["kind"] != "inline_pose_heading":
            _fail(
                f"{path}.source_local_compatibility.kind",
                "only 'inline_pose_heading' is supported",
            )
        if compatibility["authority"] != "source_schema":
            _fail(
                f"{path}.source_local_compatibility.authority",
                "must remain explicitly source_schema-authoritative",
            )
        _text(
            compatibility["metadata_path"],
            path=f"{path}.source_local_compatibility.metadata_path",
        )
        binding["source_local_compatibility"] = dict(compatibility)

    binding["schema_id"] = SOURCE_BINDING_SCHEMA_ID
    binding["schema_version"] = SOURCE_BINDING_SCHEMA_VERSION
    binding["profile_id"] = profile_id
    binding["profile_version"] = profile_version
    binding["source_schema"] = source_schema
    binding["role_bindings"] = normalized_bindings
    binding["advertised_recipe_ids"] = list(advertised)
    expected_digest = canonical_json_sha256(_without_digest(binding, "binding_sha256"))
    actual_digest = _sha256(binding["binding_sha256"], path=f"{path}.binding_sha256")
    if actual_digest != expected_digest:
        _fail(
            f"{path}.binding_sha256",
            f"stale digest; expected {expected_digest}, got {actual_digest}",
        )
    return _canonicalize_json_value(binding)


def _without_digest(value: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    return {
        key: _thaw_json(item) for key, item in value.items() if key != digest_field
    }


def _canonicalize_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _canonicalize_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonicalize_json_value(item) for item in value]
    return copy.deepcopy(value)


def _freeze_json(value: Any) -> Any:
    """Recursively freeze one validated JSON value for process-global use."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    """Return a detached mutable JSON copy of a frozen or ordinary value."""

    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_thaw_json(item) for item in value]
    return copy.deepcopy(value)


def _canonicalize_profile_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _thaw_json(value)
    payload["roles"] = sorted(payload["roles"], key=lambda item: item["role_id"])
    for role in payload["roles"]:
        role["role_id"] = str(role["role_id"])
    payload["recipes"] = sorted(payload["recipes"], key=lambda item: item["recipe_id"])
    for recipe in payload["recipes"]:
        recipe["required_roles"] = sorted(recipe["required_roles"])
        recipe["expression"] = _canonical_expression(
            _mapping(recipe["expression"], path=f"$.recipes[{recipe['recipe_id']}].expression"),
            path=f"$.recipes[{recipe['recipe_id']}].expression",
        )
    if "source_bindings" in payload:
        for binding in payload["source_bindings"]:
            if isinstance(binding.get("role_bindings"), list):
                binding["role_bindings"] = sorted(
                    binding["role_bindings"], key=lambda item: item["role_id"]
                )
            if isinstance(binding.get("advertised_recipe_ids"), list):
                binding["advertised_recipe_ids"] = sorted(
                    binding["advertised_recipe_ids"]
                )
        payload["source_bindings"] = sorted(
            payload["source_bindings"], key=lambda item: item["binding_id"]
        )
    return _canonicalize_json_value(payload)


def validate_anatomy_profile(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a canonical v1 anatomy profile mapping."""

    profile = _mapping(value, path="$")
    _exact_keys(
        profile,
        required={
            "schema_id",
            "schema_version",
            "profile_id",
            "profile_version",
            "digest_algorithm",
            "roles",
            "recipes",
            "profile_sha256",
        },
        optional={"description", "source_bindings"},
        path="$",
    )
    if profile["schema_id"] != ANATOMY_PROFILE_SCHEMA_ID:
        _fail("$.schema_id", f"expected {ANATOMY_PROFILE_SCHEMA_ID!r}")
    if (
        type(profile["schema_version"]) is not int
        or profile["schema_version"] != ANATOMY_PROFILE_SCHEMA_VERSION
    ):
        _fail("$.schema_version", f"expected {ANATOMY_PROFILE_SCHEMA_VERSION}")
    profile_id = _text(profile["profile_id"], path="$.profile_id", identifier=True)
    profile_version = profile["profile_version"]
    if type(profile_version) is not int or profile_version != 1:
        _fail("$.profile_version", "only version 1 is supported")
    if profile["digest_algorithm"] != CANONICAL_JSON_DIGEST_ALGORITHM:
        _fail("$.digest_algorithm", f"expected {CANONICAL_JSON_DIGEST_ALGORITHM!r}")
    if "description" in profile:
        _text(profile["description"], path="$.description")

    roles = _sequence(profile["roles"], path="$.roles")
    if not roles:
        _fail("$.roles", "must contain at least one role")
    normalized_roles: list[dict[str, str]] = []
    role_ids: set[str] = set()
    for index, raw_role in enumerate(roles):
        role = _mapping(raw_role, path=f"$.roles[{index}]")
        _exact_keys(
            role,
            required={"role_id", "description"},
            optional=set(),
            path=f"$.roles[{index}]",
        )
        role_id = _text(role["role_id"], path=f"$.roles[{index}].role_id", identifier=True)
        description = _text(role["description"], path=f"$.roles[{index}].description")
        if role_id in role_ids:
            _fail("$.roles", f"duplicate role {role_id!r}")
        role_ids.add(role_id)
        normalized_roles.append({"role_id": role_id, "description": description})

    recipes = _sequence(profile["recipes"], path="$.recipes")
    if not recipes:
        _fail("$.recipes", "must contain at least one named recipe")
    normalized_recipes: list[dict[str, Any]] = []
    recipe_ids: set[str] = set()
    for index, raw_recipe in enumerate(recipes):
        recipe = _mapping(raw_recipe, path=f"$.recipes[{index}]")
        _exact_keys(
            recipe,
            required={"recipe_id", "kind", "required_roles", "expression"},
            optional={"description"},
            path=f"$.recipes[{index}]",
        )
        recipe_id = _text(
            recipe["recipe_id"],
            path=f"$.recipes[{index}].recipe_id",
            identifier=True,
        )
        kind = _text(recipe["kind"], path=f"$.recipes[{index}].kind", identifier=True)
        if kind not in SUPPORTED_RECIPE_KINDS:
            _fail(f"$.recipes[{index}].kind", f"unsupported recipe kind {kind!r}")
        if recipe_id in recipe_ids:
            _fail("$.recipes", f"duplicate recipe {recipe_id!r}")
        required_roles = _sorted_unique_strings(
            recipe["required_roles"],
            path=f"$.recipes[{index}].required_roles",
        )
        unknown_roles = sorted(set(required_roles) - role_ids)
        if unknown_roles:
            _fail(
                f"$.recipes[{index}].required_roles",
                f"unknown roles: {', '.join(unknown_roles)}",
            )
        expression = _canonical_expression(
            _mapping(recipe["expression"], path=f"$.recipes[{index}].expression"),
            path=f"$.recipes[{index}].expression",
        )
        expression_kind = "axis" if expression.get("op") == "axis" else "point"
        if expression_kind != kind:
            _fail(
                f"$.recipes[{index}].expression",
                f"expression kind {expression_kind!r} does not match recipe kind {kind!r}",
            )
        expression_roles = set(_expression_roles(expression))
        if expression_roles != set(required_roles):
            _fail(
                f"$.recipes[{index}]",
                "required_roles must exactly match roles referenced by expression",
            )
        normalized_recipe: dict[str, Any] = {
            "recipe_id": recipe_id,
            "kind": kind,
            "required_roles": list(required_roles),
            "expression": expression,
        }
        if "description" in recipe:
            normalized_recipe["description"] = _text(
                recipe["description"], path=f"$.recipes[{index}].description"
            )
        recipe_ids.add(recipe_id)
        normalized_recipes.append(normalized_recipe)

    normalized = dict(profile)
    normalized["schema_id"] = ANATOMY_PROFILE_SCHEMA_ID
    normalized["schema_version"] = ANATOMY_PROFILE_SCHEMA_VERSION
    normalized["profile_id"] = profile_id
    normalized["profile_version"] = 1
    normalized["roles"] = normalized_roles
    normalized["recipes"] = normalized_recipes

    source_bindings = profile.get("source_bindings", [])
    raw_bindings = _sequence(source_bindings, path="$.source_bindings")
    normalized["source_bindings"] = []
    binding_ids: set[str] = set()
    for index, raw_binding in enumerate(raw_bindings):
        binding = _mapping(raw_binding, path=f"$.source_bindings[{index}]")
        binding_id = _text(
            binding.get("binding_id"),
            path=f"$.source_bindings[{index}].binding_id",
            identifier=True,
        )
        if binding_id in binding_ids:
            _fail("$.source_bindings", f"duplicate binding {binding_id!r}")
        binding_ids.add(binding_id)
        # Binding validation is performed below against the exact normalized
        # role/recipe set before the profile digest is checked.
        normalized["source_bindings"].append(binding)

    normalized = _canonicalize_profile_payload(normalized)
    normalized["source_bindings"] = [
        _canonical_source_binding(
            normalized,
            binding,
            path=f"$.source_bindings[{index}]",
        )
        for index, binding in enumerate(normalized["source_bindings"])
    ]
    normalized["source_bindings"] = sorted(
        normalized["source_bindings"], key=lambda item: item["binding_id"]
    )
    normalized["profile_sha256"] = _sha256(
        profile["profile_sha256"], path="$.profile_sha256"
    )
    final_expected_digest = canonical_json_sha256(
        _without_digest(normalized, "profile_sha256")
    )
    if normalized["profile_sha256"] != final_expected_digest:
        _fail(
            "$.profile_sha256",
            f"stale digest; expected {final_expected_digest}, got {normalized['profile_sha256']}",
        )
    return normalized


def validate_source_binding(
    profile: AnatomyProfile | Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    pose_schema_base_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate one explicit source-schema role binding.

    The function never inspects or rewrites an inline pose heading contract.
    An optional ``source_local_compatibility`` record is retained as explicitly
    source-schema-authoritative metadata and is never promoted to a shared
    anatomy profile reference.
    """

    if isinstance(profile, AnatomyProfile):
        profile_value: AnatomyProfile | Mapping[str, Any] = profile
    else:
        profile_value = AnatomyProfile.from_mapping(profile)
    return _canonical_source_binding(
        profile_value,
        _mapping(binding, path="$"),
        path="$",
        pose_schema_base_dir=pose_schema_base_dir,
    )


def canonical_anatomy_profile_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical profile payload without its self-digest.

    This helper is intended for builders and tests that must compute the exact
    ``profile_sha256`` before passing the complete document to strict
    validation.  It performs deterministic canonicalization but does not
    weaken validation of the completed document.
    """

    profile = _mapping(value, path="$")
    profile.pop("profile_sha256", None)
    return _canonicalize_profile_payload(profile)


def build_anatomy_profile_document(value: Mapping[str, Any]) -> dict[str, Any]:
    """Build and strictly validate one canonical self-digested profile."""

    profile = canonical_anatomy_profile_payload(value)
    profile["profile_sha256"] = canonical_json_sha256(profile)
    return validate_anatomy_profile(profile)


def anatomy_profile_sha256(value: AnatomyProfile | Mapping[str, Any]) -> str:
    """Return the digest of a canonical profile payload without its digest."""

    payload = (
        _thaw_json(value.payload)
        if isinstance(value, AnatomyProfile)
        else validate_anatomy_profile(value)
    )
    return canonical_json_sha256(_without_digest(payload, "profile_sha256"))


def source_schema_sha256(value: Mapping[str, Any]) -> str:
    """Return the digest of a source-schema identity record without its digest."""

    source_schema = _mapping(value, path="$")
    if source_schema.get("authority") == "pose_schema_package":
        payload = _mapping(source_schema.get("package_payload"), path="$.package_payload")
        return canonical_json_sha256(_canonicalize_json_value(payload))
    if source_schema.get("authority") == "keypoint_skeleton_semantics":
        document = _mapping(
            source_schema.get("skeleton_document"), path="$.skeleton_document"
        )
        return canonical_json_sha256(_canonicalize_json_value(document))
    source_schema.pop("schema_sha256", None)
    return canonical_json_sha256(_canonicalize_json_value(source_schema))


def source_binding_sha256(value: Mapping[str, Any]) -> str:
    """Return the digest of a canonical source binding without its digest."""

    binding = _mapping(value, path="$")
    binding.pop("binding_sha256", None)
    role_bindings = binding.get("role_bindings")
    if isinstance(role_bindings, Sequence) and not isinstance(
        role_bindings, (str, bytes, bytearray)
    ):
        binding["role_bindings"] = sorted(
            (_thaw_json(item) for item in role_bindings),
            key=lambda item: str(item.get("role_id", ""))
            if isinstance(item, Mapping)
            else "",
        )
    advertised_recipe_ids = binding.get("advertised_recipe_ids")
    if isinstance(advertised_recipe_ids, Sequence) and not isinstance(
        advertised_recipe_ids, (str, bytes, bytearray)
    ):
        binding["advertised_recipe_ids"] = sorted(str(item) for item in advertised_recipe_ids)
    return canonical_json_sha256(_canonicalize_json_value(binding))


def _profile_from_normalized(value: Mapping[str, Any]) -> AnatomyProfile:
    roles = tuple(
        AnatomyRole(role_id=item["role_id"], description=item["description"])
        for item in value["roles"]
    )
    recipes = tuple(
        AnatomyRecipe(
            recipe_id=item["recipe_id"],
            kind=item["kind"],
            required_roles=tuple(item["required_roles"]),
            expression=_freeze_json(item["expression"]),
            description=item.get("description"),
        )
        for item in value["recipes"]
    )
    return AnatomyProfile(
        profile_id=value["profile_id"],
        profile_version=value["profile_version"],
        roles=roles,
        recipes=recipes,
        source_bindings=tuple(
            _freeze_json(item) for item in value.get("source_bindings", [])
        ),
        profile_sha256=value["profile_sha256"],
        payload=_freeze_json(value),
    )


__all__ = [
    "ANATOMY_PROFILE_SCHEMA_ID",
    "ANATOMY_PROFILE_SCHEMA_VERSION",
    "AnatomyProfile",
    "AnatomyProfileError",
    "AnatomyRecipe",
    "AnatomyRole",
    "KEYPOINT_SKELETON_SEMANTICS_SCHEMA_ID",
    "KEYPOINT_SKELETON_SEMANTICS_SCHEMA_VERSION",
    "SOURCE_BINDING_SCHEMA_ID",
    "SOURCE_BINDING_SCHEMA_VERSION",
    "SUPPORTED_MODALITIES",
    "anatomy_profile_sha256",
    "build_anatomy_profile_document",
    "canonical_anatomy_profile_payload",
    "load_anatomy_profile",
    "source_binding_sha256",
    "source_schema_sha256",
    "validate_anatomy_profile",
    "validate_source_binding",
]


def load_anatomy_profile(path: str | Path) -> AnatomyProfile:
    """Load and validate a JSON anatomy profile."""

    return AnatomyProfile.from_json(path)
