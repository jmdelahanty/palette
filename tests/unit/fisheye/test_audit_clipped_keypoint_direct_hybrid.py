from __future__ import annotations

import pytest

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils.audit_clipped_keypoint_direct_hybrid import (
    _validate_shard_model_claim,
)
from tests.unit.fisheye.test_keypoint_publication import _pose_binding


def test_shard_model_claim_compares_identity_not_embedded_binding_shape() -> None:
    binding = _pose_binding()
    model_sha = binding["model"]["sha256"]
    attrs = {
        "model_resolution_selected_set_id": "set_v1",
        "model_resolution_selected_run_id": "run_v1",
        "model_resolution_selected_model_path": "/models/model.pt",
        "pose_schema": binding["pose_schema"],
        "run_provenance": {
            "input_artifacts": [
                {
                    "role": "keypoint_model",
                    "path": "/models/model.pt",
                    "sha256": model_sha,
                }
            ]
        },
    }
    expected = {
        "set_id": "set_v1",
        "run_id": "run_v1",
        "path": "/models/model.pt",
        "sha256": model_sha,
        "pose_model_schema_binding_digest": canonical_json_sha256(binding),
        "pose_model_schema_binding": binding,
    }

    _validate_shard_model_claim(attrs, binding=binding, expected_model=expected)

    attrs["model_resolution_selected_run_id"] = "different"
    with pytest.raises(ValueError, match="model binding differs"):
        _validate_shard_model_claim(attrs, binding=binding, expected_model=expected)
