from __future__ import annotations

import onnx
import pytest

from fisheye.shared.pose_onnx_interface import inspect_pose_onnx_interface
from tests.unit.fisheye.pose_deployment_helpers import source_bundle


@pytest.mark.parametrize(
    "problem",
    [
        "wrong_k",
        "nms",
        "task",
        "class_index",
        "duplicate_metadata",
        "dynamic_input",
        "wrong_output",
        "external_tensor",
        "wrong_network",
    ],
)
def test_rejects_unsupported_or_contradictory_onnx_interfaces(tmp_path, problem):
    _, source = source_bundle(tmp_path)
    path = source["onnx"]
    model = onnx.load(path)
    properties = {p.key: p.value for p in model.metadata_props}
    network = [192, 192]
    if problem in {"wrong_k", "nms", "task", "class_index"}:
        key, value = {
            "wrong_k": ("kpt_shape", "[19, 3]"),
            "nms": ("args", "{'nms': True}"),
            "task": ("task", "detect"),
            "class_index": ("names", "{1: 'fish'}"),
        }[problem]
        properties[key] = value
        onnx.helper.set_model_props(model, properties)
    elif problem == "duplicate_metadata":
        item = model.metadata_props.add()
        item.key, item.value = "task", "pose"
    elif problem == "dynamic_input":
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    elif problem == "wrong_output":
        model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 15
    elif problem == "external_tensor":
        tensor = model.graph.node[0].attribute[0].t
        tensor.data_location = onnx.TensorProto.EXTERNAL
        item = tensor.external_data.add()
        item.key, item.value = "location", "must_not_load.bin"
    else:
        network = [256, 256]
    path.write_bytes(model.SerializeToString())
    with pytest.raises(ValueError):
        inspect_pose_onnx_interface(path, kpt_shape=[3, 3], network_shape_hw=network)
