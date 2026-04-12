"""Smoke checks for detection preflight invocation audit metadata."""

import json
import hashlib
import sqlite3
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.diagnostics import prepare_detect_training
from fisheye.registry.db import Registry


def _write_base_detect_config(path: Path) -> None:
    payload = {
        "train": "./dummy_train.txt",
        "val": "./dummy_val.txt",
        "nc": 1,
        "names": ["fish"],
        "datasets": {},
        "training_params": {
            "model": "yolo11n.pt",
            "epochs": 1,
            "batch": 1,
            "imgsz": 640,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _create_minimal_detect_zarr(path: Path, *, session_uuid: str = "session_smoke_001") -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["session_uuid"] = session_uuid

    raw = root.create_group("raw_video")
    raw.create_array(
        "images_ds",
        data=np.zeros((2, 16, 16), dtype=np.uint8),
        chunks=(1, 16, 16),
    )

    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_smoke_001"
    refined = refined_parent.create_group("refined_detect_smoke_001")
    instances = refined.create_group("instances")
    instances.create_array("refined_row_ids", data=np.array([0], dtype=np.int64), chunks=(1,))
    instances.create_array("frame_indices", data=np.array([0], dtype=np.int32), chunks=(1,))
    instances.create_array("frame_offsets", data=np.array([0, 1], dtype=np.int64), chunks=(2,))
    instances.create_array(
        "bbox_img_xyxy",
        data=np.array([[2.0, 2.0, 8.0, 8.0]], dtype=np.float64),
        chunks=(1, 4),
    )
    instances.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float64),
        chunks=(1, 4),
    )
    instances.create_array("source_kind_codes", data=np.array([1], dtype=np.int8), chunks=(1,))
    instances.create_array("manual_edit_flags", data=np.array([True], dtype=bool), chunks=(1,))
    instances.create_array("source_detect_row_index", data=np.array([0], dtype=np.int32), chunks=(1,))
    instances.create_array("frame_counts", data=np.array([1], dtype=np.int32), chunks=(1,))
    refined.attrs["source_detect_run"] = "detect_smoke_001"
    refined.attrs["detect_review_status"] = {"state": "approved", "resolved_group": "refined"}

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_smoke_001"
    crop_group = crop_parent.create_group("crop_smoke_001")
    crop_group.attrs["detection_source_type"] = "refined"
    crop_group.attrs["detection_source_path"] = "refined_detect_runs/refined_detect_smoke_001/instances"
    crop_group.attrs["detect_review_status"] = {"state": "approved"}
    crop_group.attrs["crop_review_status"] = {"state": "approved"}
    crop_group.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
        chunks=(1, 4),
    )
    crop_group.create_array(
        "detection_source",
        data=np.array([0], dtype=np.int8),
        chunks=(1,),
    )
    crop_group.create_array(
        "frame_indices",
        data=np.array([0], dtype=np.int64),
        chunks=(1,),
    )


def test_prepare_detect_training_persists_invocation_metadata(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample.zarr"
    _create_minimal_detect_zarr(zarr_path)

    base_config_path = tmp_path / "detect_config.yaml"
    _write_base_detect_config(base_config_path)

    registry_path = tmp_path / "palette_registry.sqlite"
    monkeypatch.setenv("PALETTE_REGISTRY_PATH", str(registry_path))
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        "fisheye.utils.system.get_git_info",
        lambda: {
            "commit_hash": "abc123",
            "short_hash": "abc123",
            "branch": "main",
            "is_dirty": False,
        },
    )
    monkeypatch.setattr(
        "fisheye.utils.system.get_environment_summary",
        lambda: {
            "environment_type": "conda",
            "environment_name": "pytest-env",
            "python_version": "3.11",
            "total_packages": 3,
            "key_packages": {"numpy": "0.0-test"},
        },
    )
    monkeypatch.setattr(
        "fisheye.utils.system.get_platform_info",
        lambda **_kwargs: {
            "hostname": "pytest-host",
            "username": "pytest-user",
            "system": "Linux",
            "release": "test",
        },
    )

    argv = [
        str(zarr_path),
        "--base-config",
        str(base_config_path),
        "--set-name",
        "smoke_set",
        "--set-version",
        "1",
    ]
    prepare_detect_training.main(argv)

    config_path = tmp_path / "runs" / "configs" / "detect" / "smoke_set_v001.yaml"
    manifest_path = config_path.with_suffix(".manifest.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    invocation = manifest.get("invocation")
    assert isinstance(invocation, dict)
    assert invocation.get("tool") == "fisheye.diagnostics.prepare_detect_training"
    assert invocation.get("argv") == argv
    assert invocation.get("args", {}).get("set_name") == "smoke_set"
    assert invocation.get("git", {}).get("commit_hash") == "abc123"
    assert invocation.get("environment", {}).get("environment_name") == "pytest-env"
    assert manifest["datasets"][0]["manual_edited_bboxes"] == 1

    with sqlite3.connect(registry_path) as conn:
        row = conn.execute(
            "SELECT invocation_json, query_filter FROM training_sets WHERE set_id = ?",
            ("detect_smoke_set_v001",),
        ).fetchone()
    assert row is not None
    invocation_json, query_filter_json = row
    assert invocation_json is not None
    registry_invocation = json.loads(invocation_json)
    assert registry_invocation.get("argv") == argv
    query_filter = json.loads(query_filter_json)
    assert query_filter.get("set_name") == "smoke_set"


def test_prepare_detect_training_preserves_set_id_with_explicit_out_config(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample.zarr"
    _create_minimal_detect_zarr(zarr_path)

    base_config_path = tmp_path / "detect_config.yaml"
    _write_base_detect_config(base_config_path)

    registry_path = tmp_path / "palette_registry.sqlite"
    monkeypatch.setenv("PALETTE_REGISTRY_PATH", str(registry_path))
    monkeypatch.chdir(tmp_path)

    out_config = tmp_path / "custom" / "detect_custom.yaml"
    argv = [
        str(zarr_path),
        "--base-config",
        str(base_config_path),
        "--set-name",
        "smoke_set",
        "--set-version",
        "1",
        "--out-config",
        str(out_config),
    ]
    prepare_detect_training.main(argv)

    out_manifest = out_config.with_suffix(".manifest.json")
    assert out_manifest.exists()
    manifest = json.loads(out_manifest.read_text(encoding="utf-8"))
    assert manifest.get("set_id") == "detect_smoke_set_v001"
    assert manifest.get("set_name") == "smoke_set"
    assert manifest.get("set_version") == 1

    with sqlite3.connect(registry_path) as conn:
        row = conn.execute(
            "SELECT set_id FROM training_sets WHERE set_id = ?",
            ("detect_smoke_set_v001",),
        ).fetchone()
    assert row is not None


def test_prepare_detect_training_manifest_prefers_canonical_registry_dataset_id(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "sample.zarr"
    _create_minimal_detect_zarr(zarr_path, session_uuid="session_smoke_001")

    base_config_path = tmp_path / "detect_config.yaml"
    _write_base_detect_config(base_config_path)

    registry_path = tmp_path / "palette_registry.sqlite"
    db = Registry(registry_path)
    # Seed a conflicting base dataset_id on a different path so registry scan
    # resolves this dataset to the collision-safe canonical suffix form.
    db.upsert_dataset(
        "session_smoke_001",
        session_uuid="session_smoke_001",
        zarr_path=tmp_path / "other_sample.zarr",
        recording_id="session_smoke_001",
    )
    db.upsert_dataset(
        "session_smoke_001:legacy",
        session_uuid="session_smoke_001",
        zarr_path=zarr_path,
        recording_id="session_smoke_001",
    )
    db.close()

    monkeypatch.setenv("PALETTE_REGISTRY_PATH", str(registry_path))
    monkeypatch.chdir(tmp_path)

    out_config = tmp_path / "custom" / "detect_custom.yaml"
    prepare_detect_training.main(
        [
            str(zarr_path),
            "--base-config",
            str(base_config_path),
            "--out-config",
            str(out_config),
            "--registry",
            str(registry_path),
        ]
    )

    manifest = json.loads(out_config.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    with sqlite3.connect(registry_path) as conn:
        row = conn.execute(
            "SELECT dataset_id FROM datasets WHERE zarr_path = ? LIMIT 1;",
            (str(zarr_path),),
        ).fetchone()
    assert row is not None
    canonical_dataset_id = str(row[0])
    assert manifest["datasets"][0]["dataset_id"] == canonical_dataset_id
    assert canonical_dataset_id
    assert manifest["datasets"][0]["session_uuid"] == "session_smoke_001"


def test_registry_record_training_run_persists_invocation_json(tmp_path: Path) -> None:
    registry_path = tmp_path / "palette_registry.sqlite"
    registry = Registry(registry_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("abc", encoding="utf-8")
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    invocation = {
        "tool": "fisheye.training.train_detection",
        "argv": ["config.yaml", "--log-registry"],
    }
    final_metrics = {
        "precision": 0.91,
        "recall": 0.88,
        "mAP50": 0.95,
        "mAP50_95": 0.72,
    }
    registry.record_training_run(
        run_id="run_smoke_001",
        set_id="detect_smoke_set_v001",
        config_path=config_path,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        config_sha256=config_hash,
        status="success",
        final_metrics=final_metrics,
        invocation=invocation,
    )
    registry.close()

    with sqlite3.connect(registry_path) as conn:
        row = conn.execute(
            "SELECT invocation_json, config_sha256, status, final_metrics_json FROM training_runs WHERE run_id = ?",
            ("run_smoke_001",),
        ).fetchone()
    assert row is not None
    assert row[0] is not None
    assert json.loads(row[0]) == invocation
    assert row[1] == config_hash
    assert row[2] == "success"
    assert json.loads(row[3]) == final_metrics


def test_registry_record_training_run_status_transitions(tmp_path: Path) -> None:
    registry_path = tmp_path / "palette_registry.sqlite"
    registry = Registry(registry_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("abc", encoding="utf-8")

    run_id = "run_status_001"
    registry.record_training_run(
        run_id=run_id,
        set_id="detect_smoke_set_v001",
        config_path=config_path,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status="in_progress",
        final_metrics={"stage": "training_started"},
    )
    registry.record_training_run(
        run_id=run_id,
        set_id="detect_smoke_set_v001",
        config_path=config_path,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status="success",
        final_metrics={"precision": 0.9},
    )
    registry.close()

    with sqlite3.connect(registry_path) as conn:
        row = conn.execute(
            "SELECT status, final_metrics_json FROM training_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    assert row is not None
    assert row[0] == "success"
    assert json.loads(row[1]) == {"precision": 0.9}


def test_registry_record_training_run_dual_writes_training_models(tmp_path: Path) -> None:
    registry_path = tmp_path / "palette_registry.sqlite"
    registry = Registry(registry_path)
    config_path = tmp_path / "config.yaml"
    metrics_path = tmp_path / "results.csv"
    model_path = tmp_path / "best.pt"
    config_path.write_text("abc", encoding="utf-8")
    metrics_path.write_text("epoch,loss\n1,0.1\n", encoding="utf-8")
    model_path.write_text("weights", encoding="utf-8")

    run_id = "run_detection_model_001"
    set_id = "detect_smoke_set_v001"
    registry.record_training_run(
        run_id=run_id,
        set_id=set_id,
        config_path=config_path,
        manifest_path=None,
        model_path=model_path,
        metrics_path=metrics_path,
        model_sha256="sha_model",
        metrics_sha256="sha_metrics",
        status="success",
        final_metrics={"mAP50": 0.9},
    )
    registry.close()

    with sqlite3.connect(registry_path) as conn:
        row = conn.execute(
            """
            SELECT set_id, model_path, model_sha256, metrics_path, metrics_sha256, status, final_metrics_json
            FROM training_models
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
    assert row is not None
    assert row[0] == set_id
    assert row[1] == str(model_path)
    assert row[2] == "sha_model"
    assert row[3] == str(metrics_path)
    assert row[4] == "sha_metrics"
    assert row[5] == "success"
    assert json.loads(row[6]) == {"mAP50": 0.9}


def test_registry_record_model_export_dual_writes_format_tables(tmp_path: Path) -> None:
    registry_path = tmp_path / "palette_registry.sqlite"
    registry = Registry(registry_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("abc", encoding="utf-8")

    run_id = "run_export_model_001"
    set_id = "detect_smoke_set_v001"
    registry.record_training_run(
        run_id=run_id,
        set_id=set_id,
        config_path=config_path,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status="in_progress",
        final_metrics={"stage": "start"},
    )

    onnx_path = tmp_path / "best.onnx"
    onnx_manifest = tmp_path / "best.onnx.manifest.json"
    trt_path = tmp_path / "best_fp16.engine"
    trt_manifest = tmp_path / "best_fp16.tensorrt.manifest.json"
    onnx_path.write_text("onnx", encoding="utf-8")
    onnx_manifest.write_text(
        json.dumps(
            {
                "export": {
                    "opset": 11,
                    "input_shape": [1, 3, 640, 640],
                    "imgsz": [640, 640],
                    "nms": {"conf": 0.31, "iou": 0.67, "topk": 2},
                }
            }
        ),
        encoding="utf-8",
    )
    trt_path.write_text("engine", encoding="utf-8")
    trt_manifest.write_text(
        json.dumps(
            {
                "export": {
                    "input_shape": [1, 3, 640, 640],
                    "imgsz": [640, 640],
                    "nms": {"conf": 0.31, "iou": 0.67, "topk": 2},
                }
            }
        ),
        encoding="utf-8",
    )

    registry.record_model_export(
        run_id=run_id,
        export_type="onnx",
        path=onnx_path,
        manifest_path=onnx_manifest,
        metadata={
            "sha256": "onnx_sha",
            "manifest_sha256": "onnx_manifest_sha",
            "build_env": {
                "torch_version": "2.6.0+cu124",
                "cuda_version": "12.4",
                "system_hostname": "hostA",
            },
            "requires_plugins": True,
            "plugin_ops": ["TRT::EfficientNMS_TRT"],
            "plugin_versions": {"TRT::EfficientNMS_TRT": "1"},
        },
    )
    registry.record_model_export(
        run_id=run_id,
        export_type="tensorrt",
        path=trt_path,
        manifest_path=trt_manifest,
        metadata={
            "sha256": "trt_sha",
            "manifest_sha256": "trt_manifest_sha",
            "precision": "fp16",
            "build_env": {
                "tensorrt_version": "10.0.1",
                "cuda_version": "12.4",
                "system_hostname": "hostA",
                "gpu_name": "NVIDIA RTX A6000",
                "torch_device": {"compute_capability": "8.6"},
            },
            "trt_device_info": {
                "selected_device_name": "NVIDIA RTX A6000",
                "selected_device_uuid": "GPU-abc",
                "compute_capability": "8.6",
            },
        },
    )
    registry.close()

    with sqlite3.connect(registry_path) as conn:
        onnx_row = conn.execute(
            """
            SELECT set_id, path, sha256, manifest_path, manifest_sha256,
                   opset, nms_conf, nms_iou, nms_topk,
                   input_shape, img_h, img_w, max_batch, dynamic_shapes, file_size_bytes,
                   exporter_torch_version, exporter_cuda_version, exporter_hostname,
                   requires_plugins, plugin_ops_json, plugin_versions_json
            FROM onnx_models
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        trt_row = conn.execute(
            """
            SELECT set_id, precision, path, sha256, manifest_path, manifest_sha256,
                   nms_conf, nms_iou, nms_topk,
                   input_shape, img_h, img_w, max_batch, dynamic_shapes, file_size_bytes,
                   trt_version, cuda_version, compute_capability, gpu_name, gpu_uuid, system_hostname,
                   requires_plugins, plugin_ops_json, plugin_versions_json
            FROM tensorrt_models
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
    assert onnx_row is not None
    assert onnx_row[0] == set_id
    assert onnx_row[1] == str(onnx_path)
    assert onnx_row[2] == "onnx_sha"
    assert onnx_row[3] == str(onnx_manifest)
    assert onnx_row[4] == "onnx_manifest_sha"
    assert onnx_row[5] == 11
    assert float(onnx_row[6]) == pytest.approx(0.31)
    assert float(onnx_row[7]) == pytest.approx(0.67)
    assert onnx_row[8] == 2
    assert onnx_row[9] == "[1, 3, 640, 640]"
    assert onnx_row[10] == 640
    assert onnx_row[11] == 640
    assert onnx_row[12] == 1
    assert onnx_row[13] == 0
    assert onnx_row[14] == onnx_path.stat().st_size
    assert onnx_row[15] == "2.6.0+cu124"
    assert onnx_row[16] == "12.4"
    assert onnx_row[17] == "hostA"
    assert onnx_row[18] == 1
    assert json.loads(onnx_row[19]) == ["TRT::EfficientNMS_TRT"]
    assert json.loads(onnx_row[20]) == {"TRT::EfficientNMS_TRT": "1"}

    assert trt_row is not None
    assert trt_row[0] == set_id
    assert trt_row[1] == "fp16"
    assert trt_row[2] == str(trt_path)
    assert trt_row[3] == "trt_sha"
    assert trt_row[4] == str(trt_manifest)
    assert trt_row[5] == "trt_manifest_sha"
    assert float(trt_row[6]) == pytest.approx(0.31)
    assert float(trt_row[7]) == pytest.approx(0.67)
    assert trt_row[8] == 2
    assert trt_row[9] == "[1, 3, 640, 640]"
    assert trt_row[10] == 640
    assert trt_row[11] == 640
    assert trt_row[12] == 1
    assert trt_row[13] == 0
    assert trt_row[14] == trt_path.stat().st_size
    assert trt_row[15] == "10.0.1"
    assert trt_row[16] == "12.4"
    assert trt_row[17] == "8.6"
    assert trt_row[18] == "NVIDIA RTX A6000"
    assert trt_row[19] == "GPU-abc"
    assert trt_row[20] == "hostA"
    assert trt_row[21] == 1
    assert json.loads(trt_row[22]) == ["TRT::EfficientNMS_TRT"]
    assert json.loads(trt_row[23]) == {"TRT::EfficientNMS_TRT": "1"}
