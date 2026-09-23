"""Portable skeleton projection of Palette's existing model-schema authority.

This is a deployment view, not a new skeleton resolver. Its directly usable
fields must exactly match the enclosed, validated training-manifest binding.
The sidecar's file SHA-256 is recorded by the canonical deployment manifest;
``binding_sha256`` retains its existing, separate serialization grammar.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from fisheye.shared.pose_model_schema_binding import (
    REGISTERED_TRAINING_MANIFEST_AUTHORITY,
    validate_pose_model_schema_binding,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_bytes

POSE_MODEL_SKELETON_SCHEMA_ID = "palette.pose_model_skeleton"
POSE_MODEL_SKELETON_SCHEMA_VERSION = 1
POSE_MODEL_SKELETON_FILENAME = "pose_model_skeleton.json"


def _onnx_digest(value: Any) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(
            "ONNX SHA-256 must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def build_pose_model_skeleton(
    model_schema_binding: dict[str, Any], *, onnx_sha256: str
) -> dict[str, Any]:
    """Project the exact registered training order, retaining full evidence.

    ``kpt_shape`` is the model shape, e.g. [3, 3], not the coordinate-only
    [3, 2] runtime representation stored inside the historical binding.
    """
    if type(model_schema_binding) is not dict:
        raise ValueError("A registered model-schema binding is required")
    model = model_schema_binding.get("model")
    if type(model) is not dict:
        raise ValueError("A registered model identity is required")
    binding = validate_pose_model_schema_binding(
        model_schema_binding, expected_model_sha256=model.get("sha256")
    )
    if binding["binding_kind"] != REGISTERED_TRAINING_MANIFEST_AUTHORITY:
        raise ValueError(
            "Deployment skeletons require registered training-manifest evidence"
        )
    schema = binding["pose_schema"]
    return {
        "schema_id": POSE_MODEL_SKELETON_SCHEMA_ID,
        "schema_version": POSE_MODEL_SKELETON_SCHEMA_VERSION,
        "skeleton_id": schema["skeleton_id"],
        "nodes": [
            {"id": index, "name": name}
            for index, name in enumerate(schema["keypoint_labels"])
        ],
        "edges": copy.deepcopy(schema["edges"]),
        "kpt_shape": list(schema["metadata"]["model_kpt_shape"]),
        "source": {
            "run_id": binding["model"]["registry_run_id"],
            "set_id": binding["model"]["registry_set_id"],
            "weights_sha256": binding["model"]["sha256"],
            "onnx_sha256": _onnx_digest(onnx_sha256),
            "training_manifest_sha256": binding["authority"][
                "training_manifest_sha256"
            ],
        },
        "model_schema_binding": binding,
    }


def validate_pose_model_skeleton(
    value: Any, *, expected_onnx_sha256: str
) -> dict[str, Any]:
    """Reject unsupported versions, stale bindings, or competing projections."""
    if (
        type(value) is not dict
        or value.get("schema_id") != POSE_MODEL_SKELETON_SCHEMA_ID
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != POSE_MODEL_SKELETON_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported pose-model skeleton schema/version")
    source = value.get("source")
    if type(source) is not dict or source.get("onnx_sha256") != _onnx_digest(
        expected_onnx_sha256
    ):
        raise ValueError("Pose-model skeleton belongs to a different ONNX artifact")
    expected = build_pose_model_skeleton(
        value.get("model_schema_binding"), onnx_sha256=expected_onnx_sha256
    )
    # Strict serialized equality also rejects bool-as-int and unknown fields.
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise ValueError(
            "Skeleton projection disagrees with the exact model-schema binding"
        )
    return expected
