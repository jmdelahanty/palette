"""Cross-contract binding for anatomy-backed subject-position estimators.

The anatomy profile owns modality-neutral biological roles and recipes.  The
point-expression evaluator owns modality-specific leaves.  This module is the
small, explicit bridge between those authorities; it never discovers a source
schema from matching labels and never selects an estimator policy.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Final, Mapping

from fisheye.shared.anatomy_profile import AnatomyProfile, AnatomyProfileError
from fisheye.shared.subject_position_expression import (
    canonicalize_estimator_profile,
    canonicalize_point_expression,
    point_expression_digest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_bytes


ANATOMY_EXPRESSION_BINDING_SCHEMA_ID: Final = (
    "palette.subject_position_anatomy_expression_binding"
)
ANATOMY_EXPRESSION_BINDING_SCHEMA_VERSION: Final = 1

_LEAF_OPERATION_BY_MODALITY: Final = {
    "keypoint": "keypoint",
    "subject_mask": "component_centroid",
}


@dataclass(frozen=True)
class ResolvedAnatomyPointExpression:
    """One exact anatomy recipe lowered through one exact source binding."""

    _record_json: bytes

    @property
    def record(self) -> dict[str, Any]:
        return json.loads(self._record_json)

    @property
    def expression(self) -> dict[str, Any]:
        return deepcopy(self.record["expression"])

    @property
    def digest(self) -> str:
        return hashlib.sha256(self._record_json).hexdigest()


def _as_profile(value: AnatomyProfile | Mapping[str, Any]) -> AnatomyProfile:
    return value if isinstance(value, AnatomyProfile) else AnatomyProfile.from_mapping(value)


def _lower_point_expression(
    value: Mapping[str, Any],
    *,
    leaf_operation: str,
    source_label_by_role: Mapping[str, str],
) -> dict[str, Any]:
    op = value["op"]
    if op == "role_point":
        role_id = value["role_id"]
        try:
            source_label = source_label_by_role[role_id]
        except KeyError as exc:
            raise AnatomyProfileError(
                f"Recipe role {role_id!r} has no source binding."
            ) from exc
        return {"op": leaf_operation, "role": source_label}
    if op == "midpoint":
        points = [
            _lower_point_expression(
                point,
                leaf_operation=leaf_operation,
                source_label_by_role=source_label_by_role,
            )
            for point in value["points"]
        ]
        return {"op": "midpoint", "point_a": points[0], "point_b": points[1]}
    if op == "mean_points":
        return {
            "op": "mean_points",
            "points": [
                _lower_point_expression(
                    point,
                    leaf_operation=leaf_operation,
                    source_label_by_role=source_label_by_role,
                )
                for point in value["points"]
            ],
            "weighting": "equal_per_point",
        }
    raise AnatomyProfileError(
        f"Anatomy point recipe uses unsupported operation {op!r}."
    )


def _source_schema_identity(source_schema: Mapping[str, Any]) -> tuple[str, str]:
    authority = source_schema.get("authority")
    if authority == "pose_schema_package":
        package_payload = source_schema.get("package_payload")
        if not isinstance(package_payload, Mapping):
            raise AnatomyProfileError("Pose-schema package payload is unavailable.")
        schema_id = package_payload.get("skeleton_id")
        digest = source_schema.get("package_sha256")
    elif authority == "declared_schema":
        schema_id = source_schema.get("schema_id")
        digest = source_schema.get("schema_sha256")
    else:
        raise AnatomyProfileError(
            f"Unsupported source-schema authority {authority!r}."
        )
    if type(schema_id) is not str or not schema_id:
        raise AnatomyProfileError("Source schema has no exact semantic identity.")
    if type(digest) is not str or len(digest) != 64:
        raise AnatomyProfileError("Source schema has no exact canonical digest.")
    return schema_id, digest


def resolve_anatomy_point_expression(
    profile: AnatomyProfile | Mapping[str, Any],
    *,
    binding_id: str,
    recipe_id: str,
) -> ResolvedAnatomyPointExpression:
    """Lower one advertised point recipe through an exact source-role binding."""

    resolved_profile = _as_profile(profile)
    binding = resolved_profile.binding(binding_id)
    recipe = resolved_profile.recipe(recipe_id)
    if recipe.kind != "point":
        raise AnatomyProfileError(
            f"Recipe {recipe_id!r} is {recipe.kind!r}, not a point recipe."
        )
    if recipe_id not in binding["advertised_recipe_ids"]:
        raise AnatomyProfileError(
            f"Binding {binding_id!r} does not advertise recipe {recipe_id!r}."
        )
    modality = binding["source_schema"]["modality"]
    try:
        leaf_operation = _LEAF_OPERATION_BY_MODALITY[modality]
    except KeyError as exc:
        raise AnatomyProfileError(
            f"Source modality {modality!r} cannot provide point leaves."
        ) from exc
    source_label_by_role = {
        item["role_id"]: item["source_label"] for item in binding["role_bindings"]
    }
    expression = canonicalize_point_expression(
        _lower_point_expression(
            recipe.expression,
            leaf_operation=leaf_operation,
            source_label_by_role=source_label_by_role,
        )
    )
    source_schema = binding["source_schema"]
    source_schema_id, source_schema_sha256 = _source_schema_identity(source_schema)
    record = {
        "schema_id": ANATOMY_EXPRESSION_BINDING_SCHEMA_ID,
        "schema_version": ANATOMY_EXPRESSION_BINDING_SCHEMA_VERSION,
        "anatomy_profile_id": resolved_profile.profile_id,
        "anatomy_profile_version": resolved_profile.profile_version,
        "anatomy_profile_sha256": resolved_profile.profile_sha256,
        "source_binding_id": binding["binding_id"],
        "source_binding_sha256": binding["binding_sha256"],
        "source_schema_authority": source_schema["authority"],
        "source_schema_id": source_schema_id,
        "source_schema_sha256": source_schema_sha256,
        "source_modality": modality,
        "recipe_id": recipe.recipe_id,
        "expression": expression,
        "expression_sha256": point_expression_digest(expression),
    }
    return ResolvedAnatomyPointExpression(_record_json=canonical_json_bytes(record))


def require_estimator_anatomy_expression(
    estimator_profile: Mapping[str, Any],
    anatomy_profile: AnatomyProfile | Mapping[str, Any],
    *,
    binding_id: str,
    recipe_id: str,
) -> ResolvedAnatomyPointExpression:
    """Require an estimator's expression to equal one exact lowered recipe."""

    estimator = canonicalize_estimator_profile(estimator_profile)
    resolved = resolve_anatomy_point_expression(
        anatomy_profile,
        binding_id=binding_id,
        recipe_id=recipe_id,
    )
    if estimator["anatomy_profile_id"] != resolved.record["anatomy_profile_id"]:
        raise AnatomyProfileError(
            "Estimator anatomy profile does not match the bound anatomy authority."
        )
    if estimator["source_modality"] != resolved.record["source_modality"]:
        raise AnatomyProfileError(
            "Estimator modality does not match the bound source schema."
        )
    if estimator["expression"] != resolved.record["expression"]:
        raise AnatomyProfileError(
            "Estimator expression does not match the exact lowered anatomy recipe."
        )
    return resolved


__all__ = [
    "ANATOMY_EXPRESSION_BINDING_SCHEMA_ID",
    "ANATOMY_EXPRESSION_BINDING_SCHEMA_VERSION",
    "ResolvedAnatomyPointExpression",
    "require_estimator_anatomy_expression",
    "resolve_anatomy_point_expression",
]
