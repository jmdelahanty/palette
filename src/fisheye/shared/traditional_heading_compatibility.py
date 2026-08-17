"""Fail-closed compatibility for the traditional-v3 inline heading recipe.

The historical traditional keypoint writer stores a scalar heading recipe in
``pose_schema.metadata.heading_computation``.  This module is the narrow
boundary that permits that declaration to participate in the newer anatomy
and body-frame contracts.  It deliberately accepts only explicit documents;
it never resolves a latest run, a default schema, or an implicit binding.

The returned receipt is the only supported heading authority from this
adapter.  A caller must first obtain a valid receipt before it can consume
the normalized inline recipe.
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

from fisheye.shared.anatomy_profile import (
    AnatomyProfile,
    AnatomyProfileError,
    anatomy_profile_sha256,
    load_anatomy_profile,
    source_binding_sha256,
    validate_source_binding,
)
from fisheye.shared.pose_schema import (
    Node,
    PoseSchema,
    schema_to_attr_payload,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

TRADITIONAL_V3_SCHEMA_NAME = "traditional_v3"
TRADITIONAL_V3_SKELETON_ID = "pose_skel_traditional_v3"
TRADITIONAL_V3_BINDING_ID = "zebrafish_larva_keypoint_traditional_v3_v1"
TRADITIONAL_V3_PROFILE_ID = "zebrafish_larva_anatomy.v1"
TRADITIONAL_V3_RECIPE_ID = "anterior_axis"
HEADING_SCALAR_CONVENTION = "atan2(-dy,x)_degrees"

_EXPECTED_ROLE_IDS = ("swim_bladder", "eye_left", "eye_right")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TraditionalHeadingCompatibilityError(ValueError):
    """Raised when an inline traditional-v3 heading is not authoritative."""


def _fail(message: str) -> None:
    raise TraditionalHeadingCompatibilityError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return copy.deepcopy(value)


def _canonical(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                _plain(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        _fail(f"value is not canonical JSON: {exc}")


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    name: str,
) -> None:
    actual = set(value)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        _fail(f"{name} has an invalid shape ({'; '.join(details)})")


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_pose_schema(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize either a package JSON document or a persisted payload."""

    document = _require_mapping(value, name="pose_schema")
    name = document.get("name")
    nodes_value = document.get("nodes")
    if type(name) is not str or not name:
        _fail("pose_schema.name is required")
    if not isinstance(nodes_value, Sequence) or isinstance(
        nodes_value, (str, bytes, bytearray)
    ):
        _fail("pose_schema.nodes must be an array")

    nodes: list[Node] = []
    for index, raw_node in enumerate(nodes_value):
        node = _require_mapping(raw_node, name=f"pose_schema.nodes[{index}]")
        if set(node) != {"id", "name"}:
            _fail(f"pose_schema.nodes[{index}] must contain only id and name")
        if type(node["id"]) is not int or type(node["name"]) is not str:
            _fail(f"pose_schema.nodes[{index}] has invalid id or name")
        nodes.append(Node(id=node["id"], name=node["name"]))

    raw_edges = document.get("edges", [])
    if not isinstance(raw_edges, Sequence) or isinstance(
        raw_edges, (str, bytes, bytearray)
    ):
        _fail("pose_schema.edges must be an array")
    edges: list[list[int]] = []
    for index, raw_edge in enumerate(raw_edges):
        if (
            not isinstance(raw_edge, Sequence)
            or isinstance(raw_edge, (str, bytes, bytearray))
            or len(raw_edge) != 2
            or any(type(item) is not int for item in raw_edge)
        ):
            _fail(f"pose_schema.edges[{index}] must be a pair of integers")
        edges.append([int(raw_edge[0]), int(raw_edge[1])])

    metadata = _require_mapping(document.get("metadata"), name="pose_schema.metadata")
    schema = PoseSchema(
        name=name,
        nodes=nodes,
        edges=edges,
        metadata=dict(metadata),
    )
    payload = schema_to_attr_payload(schema)

    if (
        "keypoint_labels" in document
        and _plain(document["keypoint_labels"]) != payload["keypoint_labels"]
    ):
        _fail("pose_schema.keypoint_labels disagrees with named nodes")
    if (
        "kpt_shape" in document
        and _plain(document["kpt_shape"]) != payload["kpt_shape"]
    ):
        _fail("pose_schema.kpt_shape disagrees with named nodes")
    if "skeleton_id" in document and document["skeleton_id"] != payload["skeleton_id"]:
        _fail("pose_schema.skeleton_id disagrees with metadata")
    return _canonical(payload)


def load_explicit_pose_schema(path: str | Path) -> dict[str, Any]:
    """Read exactly one pose-schema file and normalize it without mutation."""

    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TraditionalHeadingCompatibilityError(
            f"unable to read explicit pose schema {source}: {exc}"
        ) from exc
    return _canonical_pose_schema(_require_mapping(document, name="pose_schema"))


def _validated_profile(value: AnatomyProfile | Mapping[str, Any]) -> AnatomyProfile:
    try:
        if isinstance(value, AnatomyProfile):
            profile = AnatomyProfile.from_mapping(_plain(value.payload))
        else:
            profile = AnatomyProfile.from_mapping(value)
    except (AnatomyProfileError, TypeError, ValueError) as exc:
        raise TraditionalHeadingCompatibilityError(
            f"anatomy profile is invalid: {exc}"
        ) from exc
    if anatomy_profile_sha256(profile) != profile.profile_sha256:
        _fail("anatomy profile digest is stale")
    return profile


def _recipe_payload(profile: AnatomyProfile) -> dict[str, Any]:
    recipe = profile.recipe(TRADITIONAL_V3_RECIPE_ID)
    payload: dict[str, Any] = {
        "recipe_id": recipe.recipe_id,
        "kind": recipe.kind,
        "required_roles": list(recipe.required_roles),
        "expression": _plain(recipe.expression),
    }
    if recipe.description is not None:
        payload["description"] = recipe.description
    return _canonical(payload)


def _validate_anatomy_recipe(profile: AnatomyProfile) -> dict[str, Any]:
    recipe = _recipe_payload(profile)
    if recipe["kind"] != "axis":
        _fail("anatomy anterior_axis recipe must be an axis")
    if set(recipe["required_roles"]) != set(_EXPECTED_ROLE_IDS):
        _fail("anatomy anterior_axis has renamed or missing roles")
    expected_expression = {
        "op": "axis",
        "from": {"op": "role_point", "role_id": "swim_bladder"},
        "to": {
            "op": "midpoint",
            "points": [
                {"op": "role_point", "role_id": "eye_left"},
                {"op": "role_point", "role_id": "eye_right"},
            ],
        },
    }
    if recipe["expression"] != expected_expression:
        _fail(
            "anatomy anterior_axis is not the controlled swim-bladder-to-eye-midpoint axis"
        )
    return recipe


def _validate_binding(
    profile: AnatomyProfile,
    source_binding: Mapping[str, Any],
    *,
    pose_schema: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    binding_input = _plain(_require_mapping(source_binding, name="source_binding"))
    binding_id = binding_input.get("binding_id")
    if binding_id != TRADITIONAL_V3_BINDING_ID:
        _fail(f"unexpected source binding {binding_id!r}")
    try:
        validated = validate_source_binding(profile, binding_input)
    except (AnatomyProfileError, TypeError, ValueError) as exc:
        raise TraditionalHeadingCompatibilityError(
            f"source binding is invalid: {exc}"
        ) from exc

    profile_binding = profile.binding(TRADITIONAL_V3_BINDING_ID)
    if _canonical(validated) != _canonical(profile_binding):
        _fail(
            "source binding is not the exact binding published by the explicit anatomy profile"
        )
    if source_binding_sha256(validated) != validated.get("binding_sha256"):
        _fail("source binding digest is stale")
    if validated.get("profile_id") != TRADITIONAL_V3_PROFILE_ID:
        _fail("source binding has the wrong anatomy profile")

    source_schema = _require_mapping(
        validated.get("source_schema"), name="source_binding.source_schema"
    )
    if source_schema.get("authority") != "pose_schema_package":
        _fail("traditional-v3 compatibility requires a pose-schema package binding")
    if source_schema.get("package_name") != TRADITIONAL_V3_SCHEMA_NAME:
        _fail("source binding names the wrong pose schema package")
    package_payload = _canonical(
        _require_mapping(
            source_schema.get("package_payload"), name="source_schema.package_payload"
        )
    )
    if source_schema.get("package_sha256") != canonical_json_sha256(package_payload):
        _fail("source-schema package digest is stale")

    raw_role_bindings = validated.get("role_bindings")
    if not isinstance(raw_role_bindings, Sequence) or isinstance(
        raw_role_bindings, (str, bytes, bytearray)
    ):
        _fail("source binding role_bindings must be an array")
    role_to_label: dict[str, str] = {}
    for item in raw_role_bindings:
        mapping = _require_mapping(item, name="source_binding.role_bindings entry")
        role_id, source_label = mapping.get("role_id"), mapping.get("source_label")
        if type(role_id) is not str or type(source_label) is not str:
            _fail(
                "source binding role mappings require string role_id and source_label"
            )
        role_to_label[role_id] = source_label
    if set(role_to_label) != set(_EXPECTED_ROLE_IDS):
        _fail("source binding has renamed or missing controlled anatomy roles")
    if any(role_to_label[role] != role for role in _EXPECTED_ROLE_IDS):
        _fail("source binding renamed a traditional-v3 anatomy role")

    advertised = validated.get("advertised_recipe_ids")
    if (
        not isinstance(advertised, Sequence)
        or TRADITIONAL_V3_RECIPE_ID not in advertised
    ):
        _fail("source binding does not advertise the anterior_axis recipe")
    compatibility = validated.get("source_local_compatibility")
    if compatibility != {
        "kind": "inline_pose_heading",
        "metadata_path": "metadata.heading_computation",
        "authority": "source_schema",
    }:
        _fail(
            "source binding does not advertise the controlled inline heading authority"
        )
    return validated, role_to_label


def _canonical_inline_heading(
    pose_schema: Mapping[str, Any],
    *,
    role_to_label: Mapping[str, str],
) -> dict[str, Any]:
    metadata = _require_mapping(
        pose_schema.get("metadata"), name="pose_schema.metadata"
    )
    raw = _require_mapping(
        metadata.get("heading_computation"),
        name="pose_schema.metadata.heading_computation",
    )
    _require_exact_keys(
        raw,
        required={
            "version",
            "enabled",
            "origin",
            "direction_from",
            "direction_to",
            "dependent_keypoints",
        },
        name="inline heading computation",
    )
    if raw["version"] != 1 or raw["enabled"] is not True:
        _fail("inline heading computation must be enabled version 1")

    def midpoint(value: object, *, name: str) -> list[str]:
        mapping = _require_mapping(value, name=name)
        _require_exact_keys(mapping, required={"op", "labels"}, name=name)
        if mapping["op"] != "midpoint":
            _fail(f"{name} must be the controlled eye-pair midpoint")
        labels = mapping["labels"]
        if not isinstance(labels, Sequence) or isinstance(
            labels, (str, bytes, bytearray)
        ):
            _fail(f"{name}.labels must be an array")
        if len(labels) != 2 or any(type(label) is not str for label in labels):
            _fail(f"{name}.labels must contain exactly two labels")
        resolved_roles = [
            next((role for role, label in role_to_label.items() if label == item), None)
            for item in labels
        ]
        if set(resolved_roles) != {"eye_left", "eye_right"}:
            _fail(f"{name} must resolve through the controlled eye roles")
        return [role_to_label["eye_left"], role_to_label["eye_right"]]

    origin = midpoint(raw["origin"], name="inline heading origin")
    direction_to = midpoint(raw["direction_to"], name="inline heading direction_to")
    if origin != direction_to:
        _fail("inline heading origin and direction_to must use the same midpoint")

    direction_from = _require_mapping(
        raw["direction_from"], name="inline heading direction_from"
    )
    _require_exact_keys(
        direction_from, required={"op", "label"}, name="inline heading direction_from"
    )
    if (
        direction_from.get("op") != "keypoint"
        or direction_from.get("label") != role_to_label["swim_bladder"]
    ):
        _fail("inline heading direction_from must resolve to swim_bladder")

    dependent = raw["dependent_keypoints"]
    if not isinstance(dependent, Sequence) or isinstance(
        dependent, (str, bytes, bytearray)
    ):
        _fail("inline heading dependent_keypoints must be an array")
    if len(dependent) != len(_EXPECTED_ROLE_IDS) or any(
        type(item) is not str for item in dependent
    ):
        _fail(
            "inline heading dependent_keypoints must contain the three controlled roles"
        )
    resolved_dependent = {
        role
        for item in dependent
        for role, label in role_to_label.items()
        if label == item
    }
    if resolved_dependent != set(_EXPECTED_ROLE_IDS):
        _fail("inline heading dependent_keypoints have renamed or missing roles")

    return _canonical(
        {
            "version": 1,
            "enabled": True,
            "origin": {"op": "midpoint", "labels": origin},
            "direction_from": {
                "op": "keypoint",
                "label": role_to_label["swim_bladder"],
            },
            "direction_to": {"op": "midpoint", "labels": direction_to},
            "dependent_keypoints": [role_to_label[role] for role in _EXPECTED_ROLE_IDS],
        }
    )


def _receipt_digest(payload: Mapping[str, Any]) -> str:
    return canonical_json_sha256(payload)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class TraditionalHeadingCompatibilityReceipt:
    """Immutable proof that a traditional-v3 inline heading is usable."""

    schema_name: str
    skeleton_id: str
    schema_sha256: str
    profile_id: str
    profile_version: int
    profile_sha256: str
    binding_id: str
    binding_sha256: str
    recipe_id: str
    recipe_sha256: str
    heading_computation: Mapping[str, Any]
    heading_sha256: str
    scalar_convention: str
    receipt_sha256: str

    def validated_heading_computation(self) -> Mapping[str, Any]:
        """Return the inline recipe only after rechecking this receipt digest."""

        body = self.as_dict()
        actual = body.pop("receipt_sha256")
        if actual != _receipt_digest(body):
            raise TraditionalHeadingCompatibilityError(
                "compatibility receipt digest is stale"
            )
        if self.scalar_convention != HEADING_SCALAR_CONVENTION:
            raise TraditionalHeadingCompatibilityError(
                "compatibility receipt convention is invalid"
            )
        return self.heading_computation

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_id": "palette.traditional_heading_compatibility_receipt",
            "schema_version": 1,
            "schema_name": self.schema_name,
            "skeleton_id": self.skeleton_id,
            "schema_sha256": self.schema_sha256,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "profile_sha256": self.profile_sha256,
            "binding_id": self.binding_id,
            "binding_sha256": self.binding_sha256,
            "recipe_id": self.recipe_id,
            "recipe_sha256": self.recipe_sha256,
            "heading_computation": _plain(self.heading_computation),
            "heading_sha256": self.heading_sha256,
            "scalar_convention": self.scalar_convention,
            "receipt_sha256": self.receipt_sha256,
        }


def validate_traditional_v3_heading_compatibility(
    *,
    pose_schema: Mapping[str, Any],
    anatomy_profile: AnatomyProfile | Mapping[str, Any],
    source_binding: Mapping[str, Any],
) -> TraditionalHeadingCompatibilityReceipt:
    """Validate explicit documents and return an immutable compatibility receipt."""

    pose_payload = _canonical_pose_schema(pose_schema)
    if pose_payload.get("name") != TRADITIONAL_V3_SCHEMA_NAME:
        _fail("explicit pose schema must be traditional_v3")
    if pose_payload.get("skeleton_id") != TRADITIONAL_V3_SKELETON_ID:
        _fail("explicit pose schema has the wrong skeleton identity")

    profile = _validated_profile(anatomy_profile)
    if profile.profile_id != TRADITIONAL_V3_PROFILE_ID or profile.profile_version != 1:
        _fail("explicit anatomy profile is not zebrafish_larva_anatomy.v1")
    recipe = _validate_anatomy_recipe(profile)
    validated_binding, role_to_label = _validate_binding(
        profile,
        source_binding,
        pose_schema=pose_payload,
    )
    heading = _canonical_inline_heading(pose_payload, role_to_label=role_to_label)
    schema_digest = canonical_json_sha256(pose_payload)
    package_payload = _canonical(validated_binding["source_schema"]["package_payload"])
    if package_payload != pose_payload:
        _fail(
            "explicit pose schema does not match the bound traditional-v3 package payload"
        )
    if schema_digest != validated_binding["source_schema"]["package_sha256"]:
        _fail("explicit pose-schema digest does not match the controlled binding")
    recipe_digest = canonical_json_sha256(recipe)
    heading_digest = canonical_json_sha256(heading)
    body = {
        "schema_id": "palette.traditional_heading_compatibility_receipt",
        "schema_version": 1,
        "schema_name": pose_payload["name"],
        "skeleton_id": pose_payload["skeleton_id"],
        "schema_sha256": schema_digest,
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_sha256": profile.profile_sha256,
        "binding_id": validated_binding["binding_id"],
        "binding_sha256": validated_binding["binding_sha256"],
        "recipe_id": recipe["recipe_id"],
        "recipe_sha256": recipe_digest,
        "heading_computation": heading,
        "heading_sha256": heading_digest,
        "scalar_convention": HEADING_SCALAR_CONVENTION,
    }
    receipt_digest = _receipt_digest(body)
    return TraditionalHeadingCompatibilityReceipt(
        schema_name=body["schema_name"],
        skeleton_id=body["skeleton_id"],
        schema_sha256=body["schema_sha256"],
        profile_id=body["profile_id"],
        profile_version=body["profile_version"],
        profile_sha256=body["profile_sha256"],
        binding_id=body["binding_id"],
        binding_sha256=body["binding_sha256"],
        recipe_id=body["recipe_id"],
        recipe_sha256=body["recipe_sha256"],
        heading_computation=_freeze(heading),
        heading_sha256=body["heading_sha256"],
        scalar_convention=body["scalar_convention"],
        receipt_sha256=receipt_digest,
    )


def load_traditional_v3_heading_compatibility(
    *,
    pose_schema_path: str | Path,
    anatomy_profile_path: str | Path,
    source_binding_id: str,
) -> TraditionalHeadingCompatibilityReceipt:
    """Load exactly the named files and binding, then validate them."""

    if source_binding_id != TRADITIONAL_V3_BINDING_ID:
        _fail("source_binding_id must name the controlled traditional-v3 binding")
    pose_schema = load_explicit_pose_schema(pose_schema_path)
    try:
        profile = load_anatomy_profile(anatomy_profile_path)
        binding = profile.binding(source_binding_id)
    except (AnatomyProfileError, OSError, ValueError) as exc:
        raise TraditionalHeadingCompatibilityError(
            f"unable to load explicit anatomy profile/binding: {exc}"
        ) from exc
    return validate_traditional_v3_heading_compatibility(
        pose_schema=pose_schema,
        anatomy_profile=profile,
        source_binding=binding,
    )


__all__ = [
    "HEADING_SCALAR_CONVENTION",
    "TRADITIONAL_V3_BINDING_ID",
    "TRADITIONAL_V3_PROFILE_ID",
    "TRADITIONAL_V3_RECIPE_ID",
    "TRADITIONAL_V3_SCHEMA_NAME",
    "TRADITIONAL_V3_SKELETON_ID",
    "TraditionalHeadingCompatibilityError",
    "TraditionalHeadingCompatibilityReceipt",
    "load_explicit_pose_schema",
    "load_traditional_v3_heading_compatibility",
    "validate_traditional_v3_heading_compatibility",
]
