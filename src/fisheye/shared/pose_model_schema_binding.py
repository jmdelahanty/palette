"""Exact model-artifact to ordered pose-schema bindings.

Canonical keypoint publication must not infer keypoint identity from a package
default or from model cardinality.  This module binds one content-addressed
model artifact to the ordered pose schema recorded by its hash-verified
training manifest, with the registry skeleton record used as independent
consistency evidence.
"""

from __future__ import annotations

import copy
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping

from fisheye.shared.pose_schema import Node, PoseSchema, canonicalize_keypoint_label


POSE_MODEL_SCHEMA_BINDING_SCHEMA_ID = "palette.pose_model_schema_binding"
POSE_MODEL_SCHEMA_BINDING_SCHEMA_VERSION = 1
POSE_MODEL_SCHEMA_BINDING_CANONICALIZATION = "canonical_json_sort_keys_v1"
REGISTERED_TRAINING_MANIFEST_AUTHORITY = "registered_training_manifest_v1"
EXPLICIT_DIGEST_BOUND_AUTHORITY = "explicit_digest_bound_assertion_v1"
REGISTERED_CONSISTENCY_POLICY = (
    "manifest_primary_all_populated_registry_fields_must_agree_v1"
)
EXPLICIT_CONSISTENCY_POLICY = "explicit_digest_bound_exact_schema_assertion_v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PoseModelSchemaBindingError(ValueError):
    """Raised when model-to-pose-schema evidence is missing or inconsistent."""


def _fail(message: str) -> None:
    raise PoseModelSchemaBindingError(message)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        _fail(f"Pose model-schema binding is not exact JSON: {exc}.")
    raise AssertionError("unreachable")


def _digest(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path, *, role: str) -> str:
    digest = sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        _fail(f"Unable to hash {role} {path}: {exc}.")
    return digest.hexdigest()


def _required_text(value: Any, *, field: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        _fail(f"{field} must be one nonempty canonical string.")
    return value


def _required_sha256(value: Any, *, field: str) -> str:
    text = _required_text(value, field=field).lower()
    if _SHA256_RE.fullmatch(text) is None:
        _fail(f"{field} must be one lowercase SHA-256 digest.")
    return text


def _optional_sha256(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_sha256(value, field=field)


def _shape(value: Any, *, field: str) -> list[int]:
    if (
        type(value) is not list
        or len(value) != 2
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        _fail(f"{field} must be an exact positive [keypoints, dimensions] list.")
    return list(value)


def _labels(value: Any, *, cardinality: int, field: str) -> list[str]:
    if type(value) is not list or len(value) != cardinality:
        _fail(f"{field} must contain exactly {cardinality} ordered labels.")
    labels: list[str] = []
    for item in value:
        labels.append(_required_text(item, field=field))
    if len(set(labels)) != len(labels):
        _fail(f"{field} must contain unique ordered labels.")
    return labels


def _edges(value: Any, *, cardinality: int, field: str) -> list[list[int]] | None:
    if value is None:
        return None
    if type(value) is not list:
        _fail(f"{field} must be an exact edge list or null.")
    result: list[list[int]] = []
    for edge in value:
        if (
            type(edge) is not list
            or len(edge) != 2
            or any(type(item) is not int for item in edge)
            or any(item < 0 or item >= cardinality for item in edge)
            or edge[0] == edge[1]
        ):
            _fail(f"{field} contains an invalid keypoint edge.")
        result.append(list(edge))
    return result


def _json_column(value: Any, *, field: str) -> Any:
    if value is None:
        return None
    if type(value) is not str:
        _fail(f"Registry {field} must be JSON text or null.")
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        _fail(f"Registry {field} is invalid JSON: {exc}.")


def _pose_schema_from_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    if manifest.get("task") != "pose":
        _fail("Hash-verified training manifest is not a pose manifest.")
    raw = manifest.get("pose_schema")
    if type(raw) is not dict:
        _fail("Hash-verified pose training manifest lacks pose_schema.")
    model_shape = _shape(raw.get("kpt_shape"), field="manifest pose_schema.kpt_shape")
    cardinality = model_shape[0]
    labels = _labels(
        raw.get("keypoint_labels"),
        cardinality=cardinality,
        field="manifest pose_schema.keypoint_labels",
    )
    edges = _edges(
        raw.get("skeleton"),
        cardinality=cardinality,
        field="manifest pose_schema.skeleton",
    )
    skeleton_id = _required_text(
        raw.get("skeleton_id"),
        field="manifest pose_schema.skeleton_id",
    )
    return {
        "skeleton_id": skeleton_id,
        "model_kpt_shape": model_shape,
        "keypoint_labels": labels,
        "edges": edges,
        "manifest_path": str(manifest_path),
    }


def _registry_schema_from_row(row: Mapping[str, Any], *, cardinality: int) -> dict[str, Any] | None:
    skeleton_id = row.get("registry_skeleton_id")
    spec_sha256 = row.get("registry_skeleton_spec_sha256")
    spec_json = row.get("registry_skeleton_spec_json")
    shape_column = _json_column(row.get("registry_kpt_shape_json"), field="kpt_shape_json")
    labels_column = _json_column(
        row.get("registry_keypoint_labels_json"),
        field="keypoint_labels_json",
    )
    edges_column = _json_column(row.get("registry_edges_json"), field="edges_json")
    if all(
        value is None
        for value in (
            skeleton_id,
            spec_sha256,
            spec_json,
            shape_column,
            labels_column,
            edges_column,
        )
    ):
        return None
    skeleton_text = _required_text(skeleton_id, field="registry skeleton_id")
    spec_digest = _required_sha256(spec_sha256, field="registry skeleton spec_sha256")
    if type(spec_json) is not str:
        _fail("Registry skeleton spec_json must be exact JSON text.")
    try:
        spec_payload = json.loads(spec_json)
    except json.JSONDecodeError as exc:
        _fail(f"Registry skeleton spec_json is invalid JSON: {exc}.")
    if type(spec_payload) is not dict or set(spec_payload) != {
        "kpt_shape",
        "keypoint_labels",
        "skeleton_edges",
    }:
        _fail("Registry skeleton spec_json does not use the exact controlled payload.")
    if _digest(spec_payload) != spec_digest:
        _fail(
            "Registry skeleton spec_sha256 disagrees with the exact parsed "
            "spec_json payload."
        )

    shape_raw = spec_payload.get("kpt_shape")
    labels_raw = spec_payload.get("keypoint_labels")
    edges_raw = spec_payload.get("skeleton_edges")
    if shape_column != shape_raw:
        _fail("Registry skeleton kpt_shape_json disagrees with its exact spec_json payload.")
    if labels_column != labels_raw:
        _fail(
            "Registry skeleton keypoint_labels_json disagrees with its exact "
            "spec_json payload."
        )
    if edges_column != edges_raw:
        _fail("Registry skeleton edges_json disagrees with its exact spec_json payload.")

    shape = _shape(shape_raw, field="registry skeleton kpt_shape") if shape_raw is not None else None
    labels = (
        _labels(labels_raw, cardinality=cardinality, field="registry skeleton labels")
        if labels_raw is not None
        else None
    )
    edges = (
        _edges(edges_raw, cardinality=cardinality, field="registry skeleton edges")
        if edges_raw is not None
        else None
    )
    return {
        "skeleton_id": skeleton_text,
        "spec_sha256": spec_digest,
        "kpt_shape": shape,
        "keypoint_labels": labels,
        "edges": edges,
    }


def _require_populated_registry_agreement(
    manifest_schema: Mapping[str, Any],
    registry_schema: Mapping[str, Any] | None,
) -> list[list[int]]:
    manifest_shape = manifest_schema["model_kpt_shape"]
    manifest_labels = manifest_schema["keypoint_labels"]
    manifest_edges = manifest_schema["edges"]
    if registry_schema is None:
        return list(manifest_edges or [])
    registry_shape = registry_schema.get("kpt_shape")
    registry_labels = registry_schema.get("keypoint_labels")
    registry_edges = registry_schema.get("edges")
    if registry_shape is not None and registry_shape != manifest_shape:
        _fail("Registry skeleton kpt_shape disagrees with the hash-verified training manifest.")
    if registry_labels is not None and registry_labels != manifest_labels:
        _fail(
            "Registry skeleton ordered keypoint labels disagree with the "
            "hash-verified training manifest."
        )
    if manifest_edges is not None and registry_edges is not None and registry_edges != manifest_edges:
        _fail("Registry skeleton edges disagree with the hash-verified training manifest.")
    return list(manifest_edges if manifest_edges is not None else (registry_edges or []))


def _controlled_heading_metadata(
    labels: list[str],
) -> tuple[dict[str, Any] | None, str | None]:
    role_to_label: dict[str, str] = {}
    for label in labels:
        canonical = canonicalize_keypoint_label(label)
        if canonical is not None and canonical not in role_to_label:
            role_to_label[canonical] = label
    heading_labels = ("swim_bladder", "eye_left", "eye_right")
    if not all(label in role_to_label for label in heading_labels):
        return None, None
    bladder = role_to_label["swim_bladder"]
    eye_left = role_to_label["eye_left"]
    eye_right = role_to_label["eye_right"]
    return (
        {
            "version": 1,
            "enabled": True,
            "origin": {
                "op": "midpoint",
                "labels": [eye_left, eye_right],
            },
            "direction_from": {"op": "keypoint", "label": bladder},
            "direction_to": {
                "op": "midpoint",
                "labels": [eye_left, eye_right],
            },
            "dependent_keypoints": [bladder, eye_left, eye_right],
        },
        "authoritative_ordered_labels_controlled_policy_v1",
    )


def _binding_record(
    *,
    binding_kind: str,
    model_sha256: str,
    run_id: str | None,
    set_id: str | None,
    manifest_path: str | None,
    manifest_sha256: str | None,
    manifest_skeleton_id: str,
    registry_schema: Mapping[str, Any] | None,
    model_kpt_shape: list[int],
    labels: list[str],
    edges: list[list[int]],
    assertion_id: str | None = None,
) -> dict[str, Any]:
    source = (
        f"training_manifest_sha256:{manifest_sha256}"
        if binding_kind == REGISTERED_TRAINING_MANIFEST_AUTHORITY
        else f"explicit_assertion:{assertion_id}"
    )
    heading_computation, heading_source = _controlled_heading_metadata(labels)
    schema_payload = {
        "name": manifest_skeleton_id,
        "skeleton_id": manifest_skeleton_id,
        "kpt_shape": [int(model_kpt_shape[0]), 2],
        "keypoint_labels": list(labels),
        "nodes": [
            {"id": int(index), "name": label}
            for index, label in enumerate(labels)
        ],
        "edges": copy.deepcopy(edges),
        "metadata": {
            "model_kpt_shape": list(model_kpt_shape),
            "training_manifest_skeleton_id": manifest_skeleton_id,
            "registry_skeleton_id": (
                registry_schema.get("skeleton_id") if registry_schema is not None else None
            ),
            "registry_skeleton_spec_sha256": (
                registry_schema.get("spec_sha256") if registry_schema is not None else None
            ),
            "heading_computation": heading_computation,
            "heading_computation_source": heading_source,
        },
        "source": source,
    }
    record = {
        "schema_id": POSE_MODEL_SCHEMA_BINDING_SCHEMA_ID,
        "schema_version": POSE_MODEL_SCHEMA_BINDING_SCHEMA_VERSION,
        "canonicalization": POSE_MODEL_SCHEMA_BINDING_CANONICALIZATION,
        "binding_kind": binding_kind,
        "model": {
            "role": "keypoint_model",
            "sha256": model_sha256,
            "registry_run_id": run_id,
            "registry_set_id": set_id,
        },
        "authority": {
            "training_manifest_path": manifest_path,
            "training_manifest_sha256": manifest_sha256,
            "assertion_id": assertion_id,
            "registry_skeleton_id": (
                registry_schema.get("skeleton_id") if registry_schema is not None else None
            ),
            "registry_skeleton_spec_sha256": (
                registry_schema.get("spec_sha256") if registry_schema is not None else None
            ),
            "consistency_policy": (
                REGISTERED_CONSISTENCY_POLICY
                if binding_kind == REGISTERED_TRAINING_MANIFEST_AUTHORITY
                else EXPLICIT_CONSISTENCY_POLICY
            ),
        },
        "pose_schema": schema_payload,
    }
    return {**record, "binding_sha256": _digest(record)}


def _resolve_training_manifest_path(
    *,
    registered_path: Path,
    model_path: Path,
    expected_sha256: str,
) -> tuple[Path, str]:
    """Resolve a missing workstation path to its immutable model input copy.

    Training runs retain their original registry path for historical identity,
    but deployed models package the exact input manifest below
    ``<run>/inputs``.  A present registered path always wins and must match its
    digest.  Fallback is permitted only when that path is absent and the one
    deterministic packaged copy has the same registered SHA-256.
    """

    if registered_path.is_file():
        observed = _sha256_file(
            registered_path,
            role="pose training manifest",
        )
        if observed != expected_sha256:
            _fail("Pose training manifest content disagrees with its registered digest.")
        return registered_path, observed

    packaged = model_path.expanduser().resolve().parent.parent / "inputs" / registered_path.name
    if not packaged.is_file():
        _fail(
            "Registered pose training manifest is unavailable and its exact "
            f"packaged model input copy is missing: registered={registered_path}, "
            f"packaged={packaged}."
        )
    observed = _sha256_file(
        packaged,
        role="packaged pose training manifest",
    )
    if observed != expected_sha256:
        _fail(
            "Packaged pose training manifest content disagrees with the "
            "registered digest."
        )
    return packaged, observed


def resolve_registered_pose_model_schema_binding(
    registry: Any,
    *,
    run_id: str,
    expected_set_id: str,
    expected_model_path: str,
    expected_model_sha256: str | None,
) -> dict[str, Any]:
    """Resolve one selected registry model to exact ordered manifest semantics."""

    row = registry.conn.execute(
        """
        SELECT
            tr.run_id AS run_id,
            tr.set_id AS set_id,
            tr.task_type AS run_task_type,
            ts.task_type AS set_task_type,
            tr.model_path AS training_run_model_path,
            tr.model_sha256 AS training_run_model_sha256,
            tm.model_path AS training_model_path,
            tm.model_sha256 AS training_model_sha256,
            COALESCE(tm.model_path, tr.model_path) AS model_path,
            COALESCE(tm.model_sha256, tr.model_sha256) AS model_sha256,
            tr.manifest_path AS manifest_path,
            tr.manifest_sha256 AS manifest_sha256,
            tr.skeleton_id AS training_run_skeleton_id,
            ts.skeleton_id AS training_set_skeleton_id,
            COALESCE(tr.skeleton_id, ts.skeleton_id) AS registry_skeleton_id,
            pss.spec_sha256 AS registry_skeleton_spec_sha256,
            pss.spec_json AS registry_skeleton_spec_json,
            pss.kpt_shape_json AS registry_kpt_shape_json,
            pss.keypoint_labels_json AS registry_keypoint_labels_json,
            pss.edges_json AS registry_edges_json
        FROM training_runs tr
        LEFT JOIN training_models tm ON tm.run_id = tr.run_id
        LEFT JOIN training_sets ts ON ts.set_id = tr.set_id
        LEFT JOIN pose_skeleton_specs pss
          ON pss.skeleton_id = COALESCE(tr.skeleton_id, ts.skeleton_id)
        WHERE tr.run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        _fail(f"Selected pose training run {run_id!r} is absent from the registry.")
    row_map = dict(row)
    if row_map.get("set_id") != expected_set_id:
        _fail("Selected pose model set changed during exact schema resolution.")
    if row_map.get("run_task_type") not in (None, "pose") or row_map.get("set_task_type") not in (None, "pose"):
        _fail("Selected training row is not registered as a pose model.")
    run_model_path = row_map.get("training_run_model_path")
    materialized_model_path = row_map.get("training_model_path")
    if (
        run_model_path is not None
        and materialized_model_path is not None
        and run_model_path != materialized_model_path
    ):
        _fail("training_runs and training_models disagree on pose model path.")
    run_model_sha256 = row_map.get("training_run_model_sha256")
    materialized_model_sha256 = row_map.get("training_model_sha256")
    if (
        run_model_sha256 is not None
        and materialized_model_sha256 is not None
        and str(run_model_sha256).lower() != str(materialized_model_sha256).lower()
    ):
        _fail("training_runs and training_models disagree on pose model digest.")
    run_skeleton_id = row_map.get("training_run_skeleton_id")
    set_skeleton_id = row_map.get("training_set_skeleton_id")
    if (
        run_skeleton_id is not None
        and set_skeleton_id is not None
        and run_skeleton_id != set_skeleton_id
    ):
        _fail("training_runs and training_sets disagree on pose skeleton identity.")
    if row_map.get("model_path") != expected_model_path:
        _fail("Selected pose model path changed during exact schema resolution.")
    model_sha256 = _required_sha256(row_map.get("model_sha256"), field="registry model_sha256")
    if expected_model_sha256 is not None and model_sha256 != expected_model_sha256.lower():
        _fail("Selected pose model digest changed during exact schema resolution.")
    actual_model_sha256 = _sha256_file(
        Path(expected_model_path).expanduser(),
        role="selected pose model",
    )
    if actual_model_sha256 != model_sha256:
        _fail("Selected pose model content disagrees with its registered digest.")
    registered_manifest_path = Path(
        _required_text(row_map.get("manifest_path"), field="registry manifest_path")
    ).expanduser()
    expected_manifest_sha256 = _required_sha256(
        row_map.get("manifest_sha256"),
        field="registry manifest_sha256",
    )
    manifest_path, actual_manifest_sha256 = _resolve_training_manifest_path(
        registered_path=registered_manifest_path,
        model_path=Path(expected_model_path),
        expected_sha256=expected_manifest_sha256,
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"Unable to read exact pose training manifest {manifest_path}: {exc}.")
    if type(manifest) is not dict:
        _fail("Pose training manifest must be one exact JSON mapping.")
    manifest_set_id = manifest.get("set_id")
    if manifest_set_id is not None and manifest_set_id != expected_set_id:
        _fail("Pose training manifest set_id disagrees with the selected registry row.")
    manifest_schema = _pose_schema_from_manifest(manifest, manifest_path=manifest_path)
    registry_schema = _registry_schema_from_row(
        row_map,
        cardinality=manifest_schema["model_kpt_shape"][0],
    )
    edges = _require_populated_registry_agreement(manifest_schema, registry_schema)
    binding = _binding_record(
        binding_kind=REGISTERED_TRAINING_MANIFEST_AUTHORITY,
        model_sha256=model_sha256,
        run_id=run_id,
        set_id=expected_set_id,
        manifest_path=str(manifest_path.resolve()),
        manifest_sha256=actual_manifest_sha256,
        manifest_skeleton_id=manifest_schema["skeleton_id"],
        registry_schema=registry_schema,
        model_kpt_shape=manifest_schema["model_kpt_shape"],
        labels=manifest_schema["keypoint_labels"],
        edges=edges,
    )
    return validate_pose_model_schema_binding(binding, expected_model_sha256=model_sha256)


def build_explicit_pose_model_schema_binding(
    *,
    model_sha256: str,
    assertion_id: str,
    skeleton_id: str,
    model_kpt_shape: list[int],
    keypoint_labels: list[str],
    edges: list[list[int]] | None,
) -> dict[str, Any]:
    """Build an explicit digest-bound assertion for non-registry model paths."""

    digest = _required_sha256(model_sha256, field="model_sha256")
    assertion = _required_text(assertion_id, field="assertion_id")
    skeleton = _required_text(skeleton_id, field="skeleton_id")
    shape = _shape(model_kpt_shape, field="model_kpt_shape")
    labels = _labels(
        keypoint_labels,
        cardinality=shape[0],
        field="keypoint_labels",
    )
    normalized_edges = _edges(edges, cardinality=shape[0], field="edges") or []
    binding = _binding_record(
        binding_kind=EXPLICIT_DIGEST_BOUND_AUTHORITY,
        model_sha256=digest,
        run_id=None,
        set_id=None,
        manifest_path=None,
        manifest_sha256=None,
        manifest_skeleton_id=skeleton,
        registry_schema=None,
        model_kpt_shape=shape,
        labels=labels,
        edges=normalized_edges,
        assertion_id=assertion,
    )
    return validate_pose_model_schema_binding(binding, expected_model_sha256=digest)


def validate_pose_model_schema_binding(
    value: Any,
    *,
    expected_model_sha256: str,
) -> dict[str, Any]:
    """Return an exact copied binding only after full digest/semantic validation."""

    if type(value) is not dict:
        _fail("Canonical keypoint inference requires one exact model-schema binding mapping.")
    binding = copy.deepcopy(value)
    expected_fields = {
        "schema_id",
        "schema_version",
        "canonicalization",
        "binding_kind",
        "model",
        "authority",
        "pose_schema",
        "binding_sha256",
    }
    if set(binding) != expected_fields:
        _fail("Model-schema binding fields do not match the controlled schema.")
    if (
        binding.get("schema_id") != POSE_MODEL_SCHEMA_BINDING_SCHEMA_ID
        or binding.get("schema_version") != POSE_MODEL_SCHEMA_BINDING_SCHEMA_VERSION
        or binding.get("canonicalization") != POSE_MODEL_SCHEMA_BINDING_CANONICALIZATION
        or binding.get("binding_kind")
        not in {REGISTERED_TRAINING_MANIFEST_AUTHORITY, EXPLICIT_DIGEST_BOUND_AUTHORITY}
    ):
        _fail("Model-schema binding schema or authority kind is unsupported.")
    supplied_digest = _required_sha256(
        binding.get("binding_sha256"),
        field="binding_sha256",
    )
    record = {key: value for key, value in binding.items() if key != "binding_sha256"}
    if _digest(record) != supplied_digest:
        _fail("Model-schema binding digest does not match its exact record.")
    model = binding.get("model")
    if type(model) is not dict or set(model) != {
        "role",
        "sha256",
        "registry_run_id",
        "registry_set_id",
    }:
        _fail("Model-schema binding model identity is incomplete.")
    if model.get("role") != "keypoint_model":
        _fail("Model-schema binding does not identify a keypoint model.")
    bound_model_sha256 = _required_sha256(model.get("sha256"), field="binding model sha256")
    if bound_model_sha256 != _required_sha256(
        expected_model_sha256,
        field="expected model sha256",
    ):
        _fail("Model-schema binding belongs to different model content.")
    authority = binding.get("authority")
    if type(authority) is not dict or set(authority) != {
        "training_manifest_path",
        "training_manifest_sha256",
        "assertion_id",
        "registry_skeleton_id",
        "registry_skeleton_spec_sha256",
        "consistency_policy",
    }:
        _fail("Model-schema binding authority evidence is incomplete.")
    expected_consistency_policy = (
        REGISTERED_CONSISTENCY_POLICY
        if binding["binding_kind"] == REGISTERED_TRAINING_MANIFEST_AUTHORITY
        else EXPLICIT_CONSISTENCY_POLICY
    )
    if authority.get("consistency_policy") != expected_consistency_policy:
        _fail("Model-schema binding consistency policy is unsupported.")
    if binding["binding_kind"] == REGISTERED_TRAINING_MANIFEST_AUTHORITY:
        _required_text(model.get("registry_run_id"), field="registry_run_id")
        _required_text(model.get("registry_set_id"), field="registry_set_id")
        _required_text(
            authority.get("training_manifest_path"),
            field="training_manifest_path",
        )
        _required_sha256(
            authority.get("training_manifest_sha256"),
            field="training_manifest_sha256",
        )
        _required_text(
            authority.get("registry_skeleton_id"),
            field="authority registry_skeleton_id",
        )
        _required_sha256(
            authority.get("registry_skeleton_spec_sha256"),
            field="authority registry_skeleton_spec_sha256",
        )
        if authority.get("assertion_id") is not None:
            _fail("Registered manifest bindings cannot carry an explicit assertion_id.")
    else:
        if model.get("registry_run_id") is not None or model.get("registry_set_id") is not None:
            _fail("Explicit model-schema assertions cannot claim registry identity.")
        if authority.get("training_manifest_path") is not None or authority.get("training_manifest_sha256") is not None:
            _fail("Explicit model-schema assertions cannot claim training-manifest evidence.")
        if authority.get("registry_skeleton_id") is not None or authority.get("registry_skeleton_spec_sha256") is not None:
            _fail("Explicit model-schema assertions cannot claim registry skeleton evidence.")
        _required_text(authority.get("assertion_id"), field="assertion_id")
    schema = binding.get("pose_schema")
    if type(schema) is not dict or set(schema) != {
        "name",
        "skeleton_id",
        "kpt_shape",
        "keypoint_labels",
        "nodes",
        "edges",
        "metadata",
        "source",
    }:
        _fail("Bound pose schema does not use the complete controlled payload.")
    name = _required_text(schema.get("name"), field="pose_schema.name")
    skeleton_id = _required_text(schema.get("skeleton_id"), field="pose_schema.skeleton_id")
    if name != skeleton_id:
        _fail("Bound pose schema name must equal its explicit skeleton identity.")
    runtime_shape = _shape(schema.get("kpt_shape"), field="pose_schema.kpt_shape")
    if runtime_shape[1] != 2:
        _fail("Published keypoint pose schema must use runtime coordinate dimension 2.")
    labels = _labels(
        schema.get("keypoint_labels"),
        cardinality=runtime_shape[0],
        field="pose_schema.keypoint_labels",
    )
    nodes = schema.get("nodes")
    if type(nodes) is not list or nodes != [
        {"id": index, "name": label}
        for index, label in enumerate(labels)
    ]:
        _fail("Bound pose schema nodes must exactly enumerate the ordered labels.")
    _edges(schema.get("edges"), cardinality=runtime_shape[0], field="pose_schema.edges")
    metadata = schema.get("metadata")
    if type(metadata) is not dict or set(metadata) != {
        "model_kpt_shape",
        "training_manifest_skeleton_id",
        "registry_skeleton_id",
        "registry_skeleton_spec_sha256",
        "heading_computation",
        "heading_computation_source",
    }:
        _fail("Bound pose schema metadata is incomplete.")
    model_shape = _shape(metadata.get("model_kpt_shape"), field="metadata.model_kpt_shape")
    if model_shape[0] != runtime_shape[0]:
        _fail("Model and runtime keypoint cardinality differ in the schema binding.")
    if metadata.get("training_manifest_skeleton_id") != skeleton_id:
        _fail("Bound pose schema skeleton identity differs from its authority metadata.")
    if metadata.get("registry_skeleton_id") != authority.get("registry_skeleton_id"):
        _fail(
            "Bound pose schema registry skeleton identity differs from its "
            "authority evidence."
        )
    if metadata.get("registry_skeleton_spec_sha256") != authority.get(
        "registry_skeleton_spec_sha256"
    ):
        _fail(
            "Bound pose schema registry skeleton digest differs from its "
            "authority evidence."
        )
    heading = metadata.get("heading_computation")
    heading_source = metadata.get("heading_computation_source")
    expected_heading, expected_heading_source = _controlled_heading_metadata(labels)
    if heading != expected_heading or heading_source != expected_heading_source:
        _fail(
            "Bound pose schema heading policy differs from the controlled policy "
            "for its authoritative ordered labels."
        )
    expected_source = (
        f"training_manifest_sha256:{authority['training_manifest_sha256']}"
        if binding["binding_kind"] == REGISTERED_TRAINING_MANIFEST_AUTHORITY
        else f"explicit_assertion:{authority['assertion_id']}"
    )
    if schema.get("source") != expected_source:
        _fail("Bound pose schema source differs from its exact authority evidence.")
    return binding


def pose_schema_from_model_binding(
    value: Any,
    *,
    expected_model_sha256: str,
) -> tuple[PoseSchema, dict[str, Any], dict[str, Any]]:
    """Return runtime PoseSchema, persisted attrs, and the validated binding."""

    binding = validate_pose_model_schema_binding(
        value,
        expected_model_sha256=expected_model_sha256,
    )
    payload = copy.deepcopy(binding["pose_schema"])
    schema = PoseSchema(
        name=payload["name"],
        nodes=[Node(id=node["id"], name=node["name"]) for node in payload["nodes"]],
        edges=copy.deepcopy(payload["edges"]),
        metadata=copy.deepcopy(payload["metadata"]),
    )
    return schema, payload, binding


def load_pose_model_schema_binding(path: str | Path) -> dict[str, Any]:
    """Load an explicit binding file; semantic validation occurs with model content."""

    binding_path = Path(path).expanduser()
    try:
        value = json.loads(binding_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"Unable to read pose model-schema binding {binding_path}: {exc}.")
    if type(value) is not dict:
        _fail("Pose model-schema binding file must contain one JSON mapping.")
    return value


__all__ = [
    "EXPLICIT_DIGEST_BOUND_AUTHORITY",
    "EXPLICIT_CONSISTENCY_POLICY",
    "POSE_MODEL_SCHEMA_BINDING_SCHEMA_ID",
    "POSE_MODEL_SCHEMA_BINDING_SCHEMA_VERSION",
    "PoseModelSchemaBindingError",
    "REGISTERED_TRAINING_MANIFEST_AUTHORITY",
    "REGISTERED_CONSISTENCY_POLICY",
    "build_explicit_pose_model_schema_binding",
    "load_pose_model_schema_binding",
    "pose_schema_from_model_binding",
    "resolve_registered_pose_model_schema_binding",
    "validate_pose_model_schema_binding",
]
