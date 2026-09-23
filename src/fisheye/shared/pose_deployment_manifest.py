"""Versioned, relocatable pose deployment manifests and an unpatched reader.

v1 remains the historical model-package manifest emitted by promotion. v2 is
a standalone transfer bundle: all artifact paths are relative to its manifest
directory; weights are sealed lineage, not a runtime file requirement.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

from fisheye.shared.artifact_fingerprint import require_artifact_content_identity
from fisheye.shared.pose_model_input_contract import POSE_MODEL_INPUT_CONTRACT_SCHEMA_ID
from fisheye.shared.pose_model_skeleton import validate_pose_model_skeleton
from fisheye.shared.pose_onnx_interface import inspect_pose_onnx_interface
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
)

CANONICAL_ONNX_MANIFEST_SCHEMA_ID = "palette.canonical_onnx_model_manifest"
POSE_DEPLOYMENT_MANIFEST_SCHEMA_VERSION = 2
MAX_JSON_BYTES = 8 * 1024 * 1024
ARTIFACT_ROLES = (
    "onnx",
    "source_export_manifest",
    "source_canonical_manifest",
    "training_manifest",
    "pose_model_input_contract",
    "pose_model_skeleton",
)


def require_sha256(value: Any, *, field: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be one lowercase SHA-256")
    return value


def _object(value: Any, *, field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{field} must be an object")
    return value


def read_json_document(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"JSON artifact must be a regular nonsymlink file: {path}")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"JSON artifact exceeds the 8 MiB deployment budget: {path}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    if type(value) is not dict:
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def artifact_reference(path: Path, *, relative_path: str) -> dict[str, str]:
    identity = require_artifact_content_identity(path, role="pose deployment artifact")
    return {"relative_path": relative_path, "sha256": identity["sha256"]}


def resolve_artifact(root: Path, reference: Any, *, role: str) -> Path:
    if type(reference) is not dict or not {"relative_path", "sha256"} <= set(reference):
        raise ValueError(f"Missing deployment artifact reference for {role}")
    relative = reference["relative_path"]
    if (
        type(relative) is not str
        or not relative
        or "\\" in relative
        or "\x00" in relative
    ):
        raise ValueError(f"Unsafe relative path for {role}")
    parts = PurePosixPath(relative)
    if (
        parts.is_absolute()
        or ".." in parts.parts
        or str(parts) != relative
        or relative == "."
    ):
        raise ValueError(f"Unsafe relative path for {role}")
    path = root / relative
    if any(
        parent.is_symlink()
        for parent in (path, *path.parents)
        if parent == root or root in parent.parents
    ):
        raise ValueError(f"Symlinked deployment artifact for {role}")
    require_artifact_content_identity(
        path,
        role=role,
        expected_sha256=require_sha256(reference["sha256"], field=f"{role}.sha256"),
    )
    return path


def canonical_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_id": CANONICAL_ONNX_MANIFEST_SCHEMA_ID,
        "schema_version": POSE_DEPLOYMENT_MANIFEST_SCHEMA_VERSION,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }


def validate_pose_deployment_document(document: Any, *, root: Path) -> dict[str, Any]:
    """Verify bytes, training/schema joins, and the declared ONNX interface.

    No registry, original weights, source Zarr, or GPU is required. Acceptance
    and activation are deliberately not claims made by this metadata reader.
    """
    if (
        type(document) is not dict
        or set(document) != {"schema_id", "schema_version", "payload_digest", "payload"}
        or document.get("schema_id") != CANONICAL_ONNX_MANIFEST_SCHEMA_ID
        or type(document.get("schema_version")) is not int
        or document["schema_version"] != POSE_DEPLOYMENT_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported canonical pose deployment manifest schema/version"
        )
    payload = document["payload"]
    fields = {
        "status",
        "task",
        "run_id",
        "set_id",
        "path_base",
        "selector_activation",
        "weights",
        "onnx_interface",
        "producer",
        *ARTIFACT_ROLES,
    }
    if type(payload) is not dict or set(payload) != fields:
        raise ValueError("Incomplete or unknown pose deployment manifest fields")
    if document["payload_digest"] != canonical_json_sha256(payload):
        raise ValueError("Canonical pose deployment manifest payload digest is stale")
    if (
        payload["status"] != "complete"
        or payload["task"] != "pose"
        or payload["selector_activation"] is not False
    ):
        raise ValueError(
            "Deployment manifest must be a complete, non-activating pose package"
        )
    if payload["path_base"] != "manifest_directory":
        raise ValueError("Unsupported deployment artifact path base")
    for field in ("run_id", "set_id"):
        if (
            type(payload[field]) is not str
            or not payload[field]
            or payload[field].strip() != payload[field]
        ):
            raise ValueError(f"Missing {field}")
    if type(payload["weights"]) is not dict or set(payload["weights"]) != {"sha256"}:
        raise ValueError("Weights must be a digest-only lineage reference")
    weights_sha = require_sha256(payload["weights"]["sha256"], field="weights.sha256")
    producer = _object(payload["producer"], field="producer")
    if set(producer) != {
        "module",
        "git",
        "python_version",
        "onnx_version",
        "sqlite_version",
        "source_sha256",
    }:
        raise ValueError("Missing or unknown deployment producer fields")
    if producer["module"] != "fisheye.utils.export_pose_deployment_bundle":
        raise ValueError("Unsupported deployment producer")
    source_digests = _object(producer["source_sha256"], field="producer.source_sha256")
    if not source_digests:
        raise ValueError("Missing deployment producer source identities")
    for name, digest in source_digests.items():
        require_sha256(digest, field=f"producer.source_sha256.{name}")
    paths = {}
    for role in ARTIFACT_ROLES:
        fields = {"relative_path", "sha256"} | (
            {"payload_digest"} if role == "pose_model_input_contract" else set()
        )
        if type(payload[role]) is not dict or set(payload[role]) != fields:
            raise ValueError(f"Unexpected artifact reference fields for {role}")
        paths[role] = resolve_artifact(root, payload[role], role=role)
    if len(set(paths.values())) != len(paths):
        raise ValueError("Deployment artifacts must have distinct paths")
    skeleton = validate_pose_model_skeleton(
        read_json_document(paths["pose_model_skeleton"]),
        expected_onnx_sha256=payload["onnx"]["sha256"],
    )
    source = skeleton["source"]
    if (
        source["weights_sha256"] != weights_sha
        or source["run_id"] != payload["run_id"]
        or source["set_id"] != payload["set_id"]
        or source["training_manifest_sha256"] != payload["training_manifest"]["sha256"]
    ):
        raise ValueError("Deployment manifest and skeleton source identities disagree")
    training = read_json_document(paths["training_manifest"])
    schema = training.get("pose_schema", {})
    expected_schema = {
        "skeleton_id": skeleton["skeleton_id"],
        "kpt_shape": skeleton["kpt_shape"],
        "keypoint_labels": [node["name"] for node in skeleton["nodes"]],
    }
    if training.get("task") != "pose" or training.get("set_id") != payload["set_id"]:
        raise ValueError("Training manifest does not belong to this model set")
    if type(schema) is not dict or any(
        canonical_json_bytes(schema.get(key)) != canonical_json_bytes(value)
        for key, value in expected_schema.items()
    ):
        raise ValueError("Training manifest disagrees with the skeleton projection")
    if schema.get("skeleton") is not None and canonical_json_bytes(
        schema["skeleton"]
    ) != canonical_json_bytes(skeleton["edges"]):
        raise ValueError("Training edges disagree with the skeleton projection")
    contract = read_json_document(paths["pose_model_input_contract"])
    if (
        contract.get("schema_id") != POSE_MODEL_INPUT_CONTRACT_SCHEMA_ID
        or type(contract.get("schema_version")) is not int
        or contract["schema_version"] != 3
        or contract.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
        or contract.get("payload_digest")
        != canonical_json_sha256(contract.get("payload"))
        or contract["payload_digest"]
        != payload["pose_model_input_contract"]["payload_digest"]
    ):
        raise ValueError("Unsupported or stale pose input contract")
    input_payload = _object(contract["payload"], field="input contract payload")
    model = _object(input_payload.get("model"), field="input contract model")
    model_weights = _object(model.get("weights"), field="input contract weights")
    evidence = _object(input_payload.get("evidence"), field="input contract evidence")
    training_evidence = _object(
        evidence.get("training_manifest"), field="input training manifest"
    )
    training_input = _object(
        input_payload.get("training_input"), field="training input"
    )
    if (
        input_payload.get("status") != "complete"
        or model.get("run_id") != payload["run_id"]
        or model.get("set_id") != payload["set_id"]
        or model_weights.get("sha256") != weights_sha
        or training_evidence.get("sha256") != source["training_manifest_sha256"]
    ):
        raise ValueError(
            "Pose input contract disagrees with the skeleton model identity"
        )
    export = read_json_document(paths["source_export_manifest"])
    original = read_json_document(paths["source_canonical_manifest"])
    original_payload = _object(
        original.get("payload"), field="source canonical payload"
    )
    for role in (
        "onnx",
        "weights",
        "pose_model_input_contract",
        "source_export_manifest",
    ):
        _object(original_payload.get(role), field=f"source canonical {role}")
    for role in ("onnx", "weights"):
        _object(export.get(role), field=f"source export {role}")
    if (
        original.get("schema_id") != CANONICAL_ONNX_MANIFEST_SCHEMA_ID
        or type(original.get("schema_version")) is not int
        or original["schema_version"] != 1
        or original.get("payload_digest") != canonical_json_sha256(original_payload)
        or original_payload.get("status") != "complete"
        or original_payload.get("task") != "pose"
        or original_payload.get("run_id") != payload["run_id"]
        or original_payload.get("set_id") != payload["set_id"]
        or original_payload.get("onnx", {}).get("sha256") != payload["onnx"]["sha256"]
        or original_payload.get("weights", {}).get("sha256") != weights_sha
        or original_payload.get("pose_model_input_contract", {}).get("sha256")
        != payload["pose_model_input_contract"]["sha256"]
        or original_payload["pose_model_input_contract"].get("payload_digest")
        != contract["payload_digest"]
        or original_payload.get("source_export_manifest", {}).get("sha256")
        != payload["source_export_manifest"]["sha256"]
        or export.get("run_id") != payload["run_id"]
        or export.get("onnx", {}).get("sha256") != payload["onnx"]["sha256"]
        or export.get("weights", {}).get("sha256") != weights_sha
    ):
        raise ValueError(
            "Source export/canonical provenance disagrees with this deployment"
        )
    interface = inspect_pose_onnx_interface(
        paths["onnx"],
        kpt_shape=skeleton["kpt_shape"],
        network_shape_hw=training_input.get("network_shape_hw"),
    )
    if canonical_json_bytes(interface) != canonical_json_bytes(
        payload["onnx_interface"]
    ):
        raise ValueError("Declared deployment interface differs from the ONNX file")
    return {"manifest": document, "skeleton": skeleton, "input_contract": contract}


def load_pose_deployment_bundle(
    manifest_path: Path, *, expected_manifest_sha256: str
) -> dict[str, Any]:
    """Open a bundle using a digest received through the trusted handoff."""
    require_artifact_content_identity(
        manifest_path,
        role="canonical pose deployment manifest",
        expected_sha256=require_sha256(
            expected_manifest_sha256, field="expected_manifest_sha256"
        ),
    )
    return validate_pose_deployment_document(
        read_json_document(manifest_path), root=manifest_path.parent
    )
