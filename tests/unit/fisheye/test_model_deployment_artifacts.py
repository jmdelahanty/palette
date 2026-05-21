from __future__ import annotations

import json
from pathlib import Path

from fisheye.registry.db import Registry
from fisheye.utils import register_model_deployment_artifact as register_mod


def _record_detect_run(registry: Registry, tmp_path: Path) -> None:
    model_path = tmp_path / "best.pt"
    model_path.write_text("weights", encoding="utf-8")
    registry.record_training_run(
        run_id="detect_run_a",
        set_id="detect_set_a",
        task_type="detect",
        config_path=None,
        manifest_path=None,
        model_path=model_path,
        metrics_path=None,
        model_sha256="model_sha",
        status="success",
        final_metrics={"mAP50": 0.9},
    )


def test_model_deployment_artifact_schema_and_writer(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        _record_detect_run(registry, tmp_path)
        onnx_path = tmp_path / "detect.onnx"
        onnx_path.write_text("onnx", encoding="utf-8")
        registry.record_onnx_model(
            run_id="detect_run_a",
            set_id="detect_set_a",
            detection_model_run_id="detect_run_a",
            path=onnx_path,
            sha256="onnx_sha",
            manifest_path=None,
            manifest_sha256=None,
            opset=11,
            nms_conf=0.3,
            nms_iou=0.65,
            nms_topk=1,
        )

        artifact_id = registry.record_model_deployment_artifact(
            run_id="detect_run_a",
            source_onnx_run_id="detect_run_a",
            source_onnx_path=onnx_path,
            source_onnx_sha256="onnx_sha",
            deployment_runtime="orange",
            target_hardware_class="A16",
            target_gpu_name="NVIDIA A16",
            target_compute_capability="8.6",
            precision="fp16",
            engine_path=tmp_path / "detect_a16.engine",
            engine_sha256="engine_sha",
            manifest_path=tmp_path / "detect_a16.tensorrt.manifest.json",
            manifest_sha256="manifest_sha",
            status="preferred",
            validation_summary={"steady_detect_p95_ms": 3.869},
            trtexec_path=Path("/usr/local/TensorRT/bin/trtexec"),
            trt_version="10.0.1",
            cuda_version="12.4",
            builder_optimization_level=5,
            avg_timing=32,
            profiling_verbosity="detailed",
            cuda_graph=False,
            nms_conf=0.3,
            nms_iou=0.65,
            nms_topk=1,
            metadata={"source": "unit-test"},
        )

        rows = registry.query_model_deployment_artifacts(
            deployment_runtime="orange",
            target_hardware_class="A16",
            status="preferred",
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["artifact_id"] == artifact_id
        assert row["run_id"] == "detect_run_a"
        assert row["source_onnx_run_id"] == "detect_run_a"
        assert row["artifact_kind"] == "tensorrt_engine"
        assert row["deployment_runtime"] == "orange"
        assert row["target_hardware_class"] == "A16"
        assert row["target_gpu_name"] == "NVIDIA A16"
        assert row["target_compute_capability"] == "8.6"
        assert row["precision"] == "fp16"
        assert row["engine_sha256"] == "engine_sha"
        assert row["status"] == "preferred"
        assert row["builder_optimization_level"] == 5
        assert row["avg_timing"] == 32
        assert row["profiling_verbosity"] == "detailed"
        assert row["cuda_graph"] == 0
        assert row["nms_conf"] == 0.3
        assert row["nms_iou"] == 0.65
        assert row["nms_topk"] == 1
        assert json.loads(row["validation_summary_json"]) == {
            "steady_detect_p95_ms": 3.869
        }
    finally:
        registry.close()


def test_register_model_deployment_artifact_cli_dry_run_extracts_manifest(tmp_path: Path) -> None:
    engine_path = tmp_path / "detect_a16.engine"
    onnx_path = tmp_path / "detect.onnx"
    engine_path.write_bytes(b"engine")
    onnx_path.write_bytes(b"onnx")
    manifest_path = tmp_path / "detect_a16.tensorrt.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "detect_run_a",
                "onnx": {"path": str(onnx_path), "sha256": "onnx_sha"},
                "engine": {"path": str(engine_path), "sha256": "engine_sha"},
                "export": {
                    "precision": "fp16",
                    "nms": {"conf": 0.3, "iou": 0.65, "topk": 1},
                },
                "trt": {
                    "precision": "fp16",
                    "trtexec_path": "/opt/tensorrt/bin/trtexec",
                    "cuda_graph": False,
                    "command": [
                        "trtexec",
                        "--builderOptimizationLevel=5",
                        "--avgTiming=32",
                        "--profilingVerbosity=detailed",
                    ],
                },
                "build_env": {
                    "cuda_version": "12.4",
                    "tensorrt_version": "10.0.1",
                    "trtexec_runtime": {
                        "selected_device_name": "NVIDIA A16",
                        "compute_capability": "8.6",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    args = register_mod.main(
        [
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--manifest-path",
            str(manifest_path),
            "--deployment-runtime",
            "orange",
            "--target-hardware-class",
            "A16",
            "--status",
            "candidate",
        ]
    )
    assert args == 0

    payload = register_mod.build_registration_payload(
        register_mod.argparse.Namespace(
            registry=tmp_path / "registry.sqlite",
            run_id=None,
            artifact_id=None,
            source_onnx_run_id=None,
            source_onnx_path=None,
            source_onnx_sha256=None,
            artifact_kind="tensorrt_engine",
            deployment_runtime="orange",
            target_hardware_class="A16",
            target_gpu_name=None,
            target_compute_capability=None,
            precision=None,
            engine_path=None,
            engine_sha256=None,
            manifest_path=manifest_path,
            manifest_sha256=None,
            status="candidate",
            validation_summary_json=None,
            trtexec_path=None,
            trt_version=None,
            cuda_version=None,
            builder_optimization_level=None,
            avg_timing=None,
            profiling_verbosity=None,
            cuda_graph=None,
            nms_conf=None,
            nms_iou=None,
            nms_topk=None,
            apply=False,
        )
    )
    assert payload["run_id"] == "detect_run_a"
    assert payload["target_gpu_name"] == "NVIDIA A16"
    assert payload["target_compute_capability"] == "8.6"
    assert payload["builder_optimization_level"] == 5
    assert payload["avg_timing"] == 32
    assert payload["profiling_verbosity"] == "detailed"
    assert payload["nms_conf"] == 0.3
    assert payload["nms_iou"] == 0.65
    assert payload["nms_topk"] == 1
