"""Inspect the declared static Ultralytics raw-pose deployment interface.

This validates tensor/metadata compatibility, not numerical ONNX/TRT parity.
It neither imports PyTorch nor executes the model, and never loads external
tensor files. Other decoder profiles need an explicit supported contract.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

PROFILE_ID = "ultralytics_raw_pose_static_v1"
MAX_ONNX_BYTES = 512 * 1024 * 1024


def inspect_pose_onnx_interface(
    path: Path, *, kpt_shape: list[int], network_shape_hw: list[int]
) -> dict[str, Any]:
    import onnx

    if not path.is_file() or path.is_symlink():
        raise ValueError("ONNX must be a regular nonsymlink file")
    if (
        type(kpt_shape) is not list
        or len(kpt_shape) != 2
        or any(type(item) is not int or item <= 0 for item in kpt_shape)
        or kpt_shape[1] not in (2, 3)
        or type(network_shape_hw) is not list
        or len(network_shape_hw) != 2
        or any(type(item) is not int or item <= 0 for item in network_shape_hw)
    ):
        raise ValueError(
            "Pose and network shapes must contain supported positive integers"
        )
    if path.stat().st_size > MAX_ONNX_BYTES:
        raise ValueError(
            "ONNX exceeds the static pose deployment profile's 512 MiB budget"
        )
    model = onnx.load(str(path), load_external_data=False)
    # The serialized file must be self-contained; do not follow external paths.
    from onnx.external_data_helper import _get_all_tensors, uses_external_data

    if any(uses_external_data(tensor) for tensor in _get_all_tensors(model)):
        raise ValueError(
            "External ONNX tensor data is unsupported in this deployment profile"
        )
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as exc:
        raise ValueError(f"Invalid ONNX model: {exc}") from exc
    metadata = {item.key: item.value for item in model.metadata_props}
    if len(metadata) != len(model.metadata_props):
        raise ValueError("ONNX contains duplicate metadata properties")
    if metadata.get("task") != "pose":
        raise ValueError("ONNX metadata does not declare task=pose")
    try:
        if len(metadata["kpt_shape"]) > 64:
            raise ValueError("ONNX kpt_shape exceeds the profile budget")
        declared_shape = json.loads(metadata["kpt_shape"])
        names_text, arguments_text = metadata["names"], metadata["args"]
        if max(len(names_text), len(arguments_text)) > 65536:
            raise ValueError("ONNX metadata exceeds the profile budget")
        names, arguments = ast.literal_eval(names_text), ast.literal_eval(
            arguments_text
        )
    except (KeyError, ValueError, SyntaxError, RecursionError) as exc:
        raise ValueError("ONNX pose metadata is missing or malformed") from exc
    if (
        type(declared_shape) is not list
        or any(type(item) is not int for item in declared_shape)
        or declared_shape != kpt_shape
        or len(kpt_shape) != 2
        or kpt_shape[0] <= 0
        or kpt_shape[1] not in (2, 3)
    ):
        raise ValueError("ONNX kpt_shape differs from the bound model skeleton")
    if (
        type(names) is not dict
        or not names
        or any(type(key) is not int for key in names)
    ):
        raise ValueError("ONNX class names must be indexed from zero")
    if set(names) != set(range(len(names))) or any(
        type(name) is not str or not name for name in names.values()
    ):
        raise ValueError("ONNX class names are incomplete")
    if type(arguments) is not dict or arguments.get("nms") is not False:
        raise ValueError("Only explicit nms=False raw pose exports are supported")

    def tensor_contract(value):
        tensor = value.type.tensor_type
        shape = []
        for dimension in tensor.shape.dim:
            if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
                raise ValueError(
                    "Dynamic/unknown ONNX dimensions need a different deployment profile"
                )
            shape.append(int(dimension.dim_value))
        return {
            "name": value.name,
            "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
            "shape": shape,
        }

    inputs = [tensor_contract(value) for value in model.graph.input]
    outputs = [tensor_contract(value) for value in model.graph.output]
    if inputs != [
        {"name": "images", "dtype": "FLOAT", "shape": [1, 3, *network_shape_hw]}
    ]:
        raise ValueError(
            "ONNX input binding differs from the validated preprocessing contract"
        )
    channels = 4 + len(names) + kpt_shape[0] * kpt_shape[1]
    if (
        len(outputs) != 1
        or outputs[0]["name"] != "output0"
        or outputs[0]["dtype"] != "FLOAT"
        or len(outputs[0]["shape"]) != 3
        or outputs[0]["shape"][:2] != [1, channels]
    ):
        raise ValueError(
            "ONNX output binding differs from the declared raw pose decoder"
        )
    return {
        "profile_id": PROFILE_ID,
        "inputs": inputs,
        "outputs": outputs,
        "class_names": [names[index] for index in range(len(names))],
        "kpt_shape": list(kpt_shape),
        "embedded_nms": False,
    }
