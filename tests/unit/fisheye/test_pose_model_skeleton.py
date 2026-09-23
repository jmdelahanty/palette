"""Preserve the authoritative model order when projecting a deployment sidecar."""

from copy import deepcopy

import pytest

from fisheye.shared.pose_model_schema_binding import (
    build_explicit_pose_model_schema_binding,
)
from fisheye.shared.pose_model_skeleton import (
    build_pose_model_skeleton,
    validate_pose_model_skeleton,
)


def _binding(count=3):
    labels = ["swim_bladder", "eye_left", "eye_right"]
    labels += [f"tail_{index}" for index in range(count - 3)]
    return build_explicit_pose_model_schema_binding(
        model_sha256="a" * 64,
        assertion_id="test-only-explicit-assertion",
        skeleton_id="pose_schema:example_v1",
        model_kpt_shape=[count, 3],
        keypoint_labels=labels,
        edges=[[0, 1], [0, 2], [1, 2]],
    )


def test_sidecar_requires_registered_training_evidence_not_a_manual_assertion():
    with pytest.raises(ValueError, match="registered"):
        build_pose_model_skeleton(_binding(), onnx_sha256="b" * 64)


def _registered_binding(count=3):
    """Exact serialized binding, without fabricating any production evidence."""
    from hashlib import sha256
    import json

    binding = _binding(count)
    binding["binding_kind"] = "registered_training_manifest_v1"
    binding["model"].update(registry_run_id="run_v1", registry_set_id="set_v1")
    binding["authority"].update(
        training_manifest_path="/source/training.manifest.json",
        training_manifest_sha256="c" * 64,
        assertion_id=None,
        registry_skeleton_id="registry_skeleton",
        registry_skeleton_spec_sha256="d" * 64,
        consistency_policy="manifest_primary_all_populated_registry_fields_must_agree_v1",
    )
    binding["pose_schema"]["metadata"].update(
        registry_skeleton_id="registry_skeleton", registry_skeleton_spec_sha256="d" * 64
    )
    binding["pose_schema"]["source"] = "training_manifest_sha256:" + "c" * 64
    record = {key: value for key, value in binding.items() if key != "binding_sha256"}
    binding["binding_sha256"] = sha256(
        json.dumps(
            record, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()
    return binding


@pytest.mark.parametrize("count", [3, 19])
def test_projection_preserves_training_order_edges_and_model_shape(count):
    binding = _registered_binding(count)
    before = deepcopy(binding)
    sidecar = build_pose_model_skeleton(binding, onnx_sha256="b" * 64)
    assert binding == before
    assert sidecar["schema_id"] == "palette.pose_model_skeleton"
    assert sidecar["schema_version"] == 1
    assert sidecar["nodes"] == binding["pose_schema"]["nodes"]
    assert sidecar["edges"] == [[0, 1], [0, 2], [1, 2]]
    assert sidecar["kpt_shape"] == [count, 3]
    assert sidecar["model_schema_binding"]["pose_schema"]["kpt_shape"] == [count, 2]
    assert sidecar["source"]["training_manifest_sha256"] == "c" * 64
    assert (
        validate_pose_model_skeleton(sidecar, expected_onnx_sha256="b" * 64) == sidecar
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(schema_version=2),
        lambda value: value.update(schema_version=True),
        lambda value: value.update(kpt_shape=[3, 2]),
        lambda value: value["nodes"].reverse(),
        lambda value: value["nodes"][0].update(id=True),
        lambda value: value["edges"].append([0, 19]),
        lambda value: value.update(skeleton_id="another_schema"),
        lambda value: value["source"].update(training_manifest_sha256="e" * 64),
        lambda value: value["source"].update(run_id="other_run"),
        lambda value: value["model_schema_binding"]["pose_schema"]["nodes"].reverse(),
    ],
)
def test_rejects_tampering_and_projection_disagreement(mutate):
    sidecar = build_pose_model_skeleton(_registered_binding(), onnx_sha256="b" * 64)
    mutate(sidecar)
    with pytest.raises(ValueError):
        validate_pose_model_skeleton(sidecar, expected_onnx_sha256="b" * 64)


def test_rejects_wrong_model_export():
    sidecar = build_pose_model_skeleton(_registered_binding(), onnx_sha256="b" * 64)
    with pytest.raises(ValueError, match="ONNX"):
        validate_pose_model_skeleton(sidecar, expected_onnx_sha256="e" * 64)
