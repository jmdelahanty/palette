"""Validate a completed experimental pose run and stage its model-source bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else str(dim.dim_param)
        for dim in value.type.tensor_type.shape.dim
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-epochs", type=int, required=True)
    args = parser.parse_args()

    run = args.run_dir.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text())
    assert int(config["training_params"]["epochs"]) == args.expected_epochs
    assert len(config["datasets"]) == 1
    dataset = next(iter(config["datasets"].values()))
    zarr_path = Path(dataset["zarr_path"]).resolve(strict=True)
    root = json.loads((zarr_path / "zarr.json").read_text())["attributes"]
    training_export = root["training_export"]
    logical_hash = training_export["logical_dataset_hash"]
    labels = list(training_export["keypoint_labels"])
    kpt_shape = list(training_export["kpt_shape"])
    assert labels == ["swim_bladder", "eye_left", "eye_right"]
    assert kpt_shape == [3, 3] == list(config["kpt_shape"])
    assert root["stage_selector_eligible"] is False
    assert logical_hash["algorithm"] == "sha256"

    with (run / "results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows and int(rows[-1]["epoch"]) == args.expected_epochs
    metrics = {
        key: float(rows[-1][column])
        for key, column in {
            "pose_map50": "metrics/mAP50(P)",
            "pose_map50_95": "metrics/mAP50-95(P)",
            "box_map50": "metrics/mAP50(B)",
            "box_map50_95": "metrics/mAP50-95(B)",
        }.items()
    }

    weights = run / "weights" / "best.pt"
    onnx_path = run / "exports" / "onnx" / f"{run.name}.onnx"
    export_manifest_path = onnx_path.with_name(f"{run.name}.onnx.manifest.json")
    export_manifest = json.loads(export_manifest_path.read_text())
    weights_hash = sha256(weights)
    onnx_hash = sha256(onnx_path)
    assert export_manifest["run_id"] == run.name
    assert export_manifest["weights"]["sha256"] == weights_hash
    assert export_manifest["onnx"]["sha256"] == onnx_hash
    assert export_manifest["export"]["input_shape"] == [1, 3, 192, 192]
    assert export_manifest["export"]["opset"] == 17
    assert export_manifest["export"]["dynamic"] is False

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    assert len(graph.graph.input) == len(graph.graph.output) == 1
    input_name = graph.graph.input[0].name
    output_name = graph.graph.output[0].name
    input_shape = shape(graph.graph.input[0])
    output_shape = shape(graph.graph.output[0])
    assert input_shape == [1, 3, 192, 192]
    assert output_shape == [1, 5 + 3 * len(labels), 756]
    assert graph.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert graph.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert [(entry.domain, entry.version) for entry in graph.opset_import] == [("", 17)]
    assert all(node.op_type != "NonMaxSuppression" for node in graph.graph.node)
    assert all(node.domain in ("", "ai.onnx") for node in graph.graph.node)

    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    sample = rng.random((1, 3, 192, 192), dtype=np.float32)
    torch_model = YOLO(str(weights)).model.eval().cpu()
    with torch.inference_mode():
        torch_output = torch_model(torch.from_numpy(sample))
    if isinstance(torch_output, (tuple, list)):
        torch_output = torch_output[0]
    torch_array = torch_output.detach().numpy()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path), options, providers=["CPUExecutionProvider"]
    )
    onnx_array = session.run([output_name], {input_name: sample})[0]
    np.testing.assert_allclose(onnx_array, torch_array, rtol=1e-3, atol=1e-3)
    max_abs_diff = float(np.max(np.abs(onnx_array - torch_array)))

    receipt_path = run / "pose_training_runtime_receipt.json"
    receipt = json.loads(receipt_path.read_text())["payload"]
    assert receipt["status"] == "verified"
    assert receipt["model_input_shape_hw"] == [192, 192]

    bundle = run / "model_sources" / "pose" / run.name
    bundle.mkdir(parents=True, exist_ok=False)
    shutil.copy2(onnx_path, bundle / onnx_path.name)
    manifest = {
        "schema_id": "palette.pose_model_source_bundle",
        "schema_version": 1,
        "status": "experimental_selector_ineligible",
        "training_run_id": run.name,
        "training_epochs_completed": args.expected_epochs,
        "dataset": {
            "zarr_path": str(zarr_path),
            "set_id": training_export["set_id"],
            "logical_dataset_sha256": logical_hash["digest"],
            "source_manifest_sha256": training_export["manifest_sha256"],
            "split_policy": training_export["split"],
        },
        "model": {
            "architecture": "yolo11n-pose",
            "weights_path": str(weights),
            "weights_sha256": weights_hash,
            "onnx_filename": onnx_path.name,
            "onnx_sha256": onnx_hash,
            "keypoint_names": labels,
            "keypoint_shape": kpt_shape,
            "class_names": list(config["names"]),
        },
        "crop": {
            "size_px": 192,
            "recipe_id": "pose_head_fixed_192_truncated_centroid_v1",
            "longer_source_box_side_percentiles_px": {
                "pooled": {"p5": 122.920, "p50": 136.594, "p95": 158.625},
                "RedScare": {"p5": 126.019, "p50": 139.238, "p95": 156.863},
                "DefaultScreen": {"p5": 121.524, "p50": 129.132, "p95": 137.430},
                "GoodCopBadCop": {"p5": 122.053, "p50": 127.781, "p95": 133.950},
                "Sleepyfish_clipped": {"p5": 127.201, "p50": 146.065, "p95": 195.726},
            },
        },
        "runtime_contract": {
            "input_name": input_name,
            "input_shape": input_shape,
            "input_dtype": "float32",
            "output_name": output_name,
            "output_shape": output_shape,
            "output_dtype": "float32",
            "opset": 17,
            "dynamic_shapes": False,
            "nms_in_graph": False,
            "preprocessing": config["preprocessing"],
        },
        "validation": {
            "onnx_checker": "passed",
            "onnxruntime_vs_pytorch_max_abs_diff": max_abs_diff,
            "held_out_crop_metrics": metrics,
        },
        "provenance": {
            "training_config_path": str(config_path),
            "training_config_sha256": sha256(config_path),
            "training_runtime_receipt_path": str(receipt_path),
            "training_runtime_receipt_sha256": sha256(receipt_path),
            "palette_export_manifest_path": str(export_manifest_path),
            "palette_export_manifest_sha256": sha256(export_manifest_path),
        },
    }
    manifest_path = bundle / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "bundle": str(bundle),
                "manifest": str(manifest_path),
                "metrics": metrics,
                "onnx_sha256": onnx_hash,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
