"""Small real ONNX/registry/input-contract fixtures for deployment tests."""

from contextlib import closing
import json
from pathlib import Path
from unittest.mock import patch

import onnx
from onnx import TensorProto, helper

from fisheye.registry.db import Registry
from fisheye.shared import pose_model_input_contract as input_contract
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils.promote_registered_pose_model import _canonical_onnx_manifest
from tests.unit.fisheye.test_promote_registered_pose_model import (
    RUN_ID,
    SET_ID,
    _sha,
    _source_package,
)


def source_bundle(tmp_path: Path, *, count: int = 3) -> tuple[Path, dict]:
    package = _source_package(tmp_path)
    root = package["root"]
    manifest = root / "inputs" / f"{SET_ID}.manifest.json"
    document = json.loads(manifest.read_text())
    labels = ["swim_bladder", "eye_left", "eye_right"]
    labels += [f"tail_{index}" for index in range(count - 3)]
    edges = [[0, 1], [0, 2], [1, 2]]
    document["pose_schema"] = {
        "skeleton_id": "pose_schema:example_v1",
        "kpt_shape": [count, 3],
        "keypoint_labels": labels,
        "skeleton": edges,
    }
    write_json_atomic(manifest, document)
    receipt_path = root / "pose_training_runtime_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["payload"]["training_manifest"]["sha256"] = _sha(manifest)
    receipt["payload_sha256"] = canonical_json_sha256(receipt["payload"])
    write_json_atomic(receipt_path, receipt)

    output_shape = [1, 5 + count * 3, 756]
    tensor = helper.make_tensor(
        "constant", TensorProto.FLOAT, output_shape, [0.0] * (output_shape[1] * 756)
    )
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Constant", [], ["output0"], value=tensor)],
            "fixture_pose",
            [
                helper.make_tensor_value_info(
                    "images", TensorProto.FLOAT, [1, 3, 192, 192]
                )
            ],
            [helper.make_tensor_value_info("output0", TensorProto.FLOAT, output_shape)],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    helper.set_model_props(
        model,
        {
            "task": "pose",
            "kpt_shape": json.dumps([count, 3]),
            "names": "{0: 'fish'}",
            "args": "{'nms': False}",
        },
    )
    onnx.save(model, package["onnx"])
    export = json.loads(package["onnx_manifest"].read_text())
    export.update(
        task="pose",
        source_manifest={
            "set_id": SET_ID,
            "manifest_path": str(manifest),
            "manifest_sha256": _sha(manifest),
        },
    )
    export["onnx"]["sha256"] = _sha(package["onnx"])
    write_json_atomic(package["onnx_manifest"], export)

    contract_path = root / "pose_model_input_contract.json"
    with patch.object(
        input_contract.importlib.metadata, "version", return_value="8.3.214"
    ):
        contract = input_contract.build_runtime_receipt_pose_model_input_contract(
            set_id=SET_ID,
            run_id=RUN_ID,
            model_package_root=root,
            weights_relative_path=Path("weights/best.pt"),
            training_manifest_relative_path=manifest.relative_to(root),
            training_report_relative_path=Path("training_report.yaml"),
            training_args_relative_path=Path("args.yaml"),
            training_runtime_receipt_relative_path=Path(
                "pose_training_runtime_receipt.json"
            ),
            model_stride=32,
            runtime_ultralytics_versions=("8.3.214",),
        )
    write_json_atomic(contract_path, contract)
    plan = {
        "run_id": RUN_ID,
        "set_id": SET_ID,
        "relative": {
            key: str(package[key].relative_to(root))
            for key in ("onnx", "onnx_manifest", "weights")
        },
        "digests": {
            key: _sha(package[key]) for key in ("onnx", "onnx_manifest", "weights")
        },
    }
    plan["relative"]["model"] = plan["relative"].pop("weights")
    plan["digests"]["model"] = plan["digests"].pop("weights")
    canonical = root / "exports/onnx" / f"{RUN_ID}.canonical.manifest.json"
    write_json_atomic(
        canonical,
        _canonical_onnx_manifest(
            plan=plan, package_root=root, contract_path=contract_path
        ),
    )
    registry_path = tmp_path / "registry.sqlite"
    with closing(Registry(registry_path)) as registry:
        skeleton_id = registry.upsert_pose_skeleton_spec(
            kpt_shape=[count, 3],
            keypoint_labels=labels,
            edges=edges,
            name="fixture_schema",
        )
        registry.upsert_training_set(
            set_id=SET_ID,
            name=SET_ID,
            task_type="pose",
            query_filter=None,
            dataset_ids=[],
            skeleton_id=skeleton_id,
        )
        registry.record_training_run(
            run_id=RUN_ID,
            set_id=SET_ID,
            task_type="pose",
            config_path=None,
            manifest_path=manifest,
            manifest_sha256=_sha(manifest),
            skeleton_id=skeleton_id,
            model_path=package["weights"],
            model_sha256=_sha(package["weights"]),
            metrics_path=package["metrics"],
            metrics_sha256=_sha(package["metrics"]),
            status="success",
        )
        registry.record_onnx_model(
            run_id=RUN_ID,
            set_id=SET_ID,
            skeleton_id=skeleton_id,
            detection_model_run_id=RUN_ID,
            path=package["onnx"],
            sha256=_sha(package["onnx"]),
            manifest_path=canonical,
            manifest_sha256=_sha(canonical),
        )
    package.update(
        canonical=canonical, training_manifest=manifest, input_contract=contract_path
    )
    return registry_path, package
