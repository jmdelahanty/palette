from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from unittest.mock import patch

from fisheye.shared import pose_model_input_contract as contract_module
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils.promote_registered_pose_model import promote


RUN_ID = "pose_run_v1"
SET_ID = "pose_set_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_package(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "source" / RUN_ID
    (root / "weights").mkdir(parents=True)
    (root / "inputs").mkdir()
    (root / "exports" / "onnx").mkdir(parents=True)
    weights = root / "weights" / "best.pt"
    weights.write_bytes(b"model")
    metrics = root / "results.csv"
    metrics.write_text("epoch,metric\n1,1\n", encoding="utf-8")
    onnx = root / "exports" / "onnx" / f"{RUN_ID}.onnx"
    onnx.write_bytes(b"onnx")
    manifest = root / "inputs" / f"{SET_ID}.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "task": "pose",
                "set_id": SET_ID,
                "input_format": "gray",
                "roi_pixel_contract_name": "materialized_pose_gray_uint8_exact_v1",
            }
        ),
        encoding="utf-8",
    )
    report = root / "training_report.yaml"
    report.write_text(
        """training_params:
  imgsz: 192
  rect: false
training_history:
  ultralytics_version: 8.3.214
  source_zarr_metadata:
    training.zarr:
      crop_info:
        roi_size: [192, 192]
""",
        encoding="utf-8",
    )
    (root / "args.yaml").write_text(
        "task: pose\nimgsz: 192\nrect: false\nmulti_scale: false\n",
        encoding="utf-8",
    )
    transform = {
        "name": "identity",
        "native_shape_hw": [192, 192],
        "model_shape_hw": [192, 192],
        "pad_top": 0,
        "pad_bottom": 0,
        "pad_left": 0,
        "pad_right": 0,
        "coordinate_mapping": "native_xy = model_xy - [pad_left, pad_top]",
    }
    split = {
        "schema_id": "palette.pose_training_preprocessing_runtime.v2",
        "augmentation_enabled": False,
        "channel_transform": "luma_repeat_three",
        "normalization": "uint8_div_255",
        "padding_value_uint8": 0,
        "model_input_shape_hw": [192, 192],
        "sources": {"training.zarr": transform},
    }
    runtime_payload = {
        "status": "verified",
        "training_manifest": {"path": "/source.json", "sha256": _sha(manifest)},
        "model_input_shape_hw": [192, 192],
        "preprocessing_contract": {
            "spatial_transform": "auto",
            "padding_value_uint8": 0,
            "channel_transform": "luma_repeat_three",
            "normalization": "uint8_div_255",
            "interpolation": "none",
        },
        "effective_arguments": {
            "status": "exact_match",
            "effective": {
                "imgsz": 192,
                "rect": False,
                "multi_scale": False,
                "augment": False,
            },
        },
        "datasets": {"train": split, "val": split},
        "first_batch": {
            "status": "verified",
            "raw_dtype": "uint8",
            "normalized_dtype": "float32",
            "raw_shape_nchw": [4, 3, 192, 192],
            "normalized_shape_nchw": [4, 3, 192, 192],
        },
    }
    (root / "pose_training_runtime_receipt.json").write_text(
        json.dumps(
            {
                "schema_id": "palette.pose_training_runtime_receipt.v2",
                "payload": runtime_payload,
                "payload_sha256": canonical_json_sha256(runtime_payload),
            }
        ),
        encoding="utf-8",
    )
    onnx_manifest = root / "exports" / "onnx" / f"{RUN_ID}.onnx.manifest.json"
    onnx_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": RUN_ID,
                "weights": {"path": str(weights), "sha256": _sha(weights)},
                "onnx": {"path": str(onnx), "sha256": _sha(onnx)},
            }
        ),
        encoding="utf-8",
    )
    return {
        "root": root,
        "weights": weights,
        "metrics": metrics,
        "onnx": onnx,
        "onnx_manifest": onnx_manifest,
    }


def _registry(tmp_path: Path, package: dict[str, Path]) -> tuple[Path, Path]:
    registry = tmp_path / "registry.sqlite"
    connection = sqlite3.connect(registry)
    connection.executescript(
        """
        CREATE TABLE training_runs (
          run_id TEXT PRIMARY KEY, set_id TEXT, task_type TEXT, status TEXT,
          model_path TEXT, model_sha256 TEXT, metrics_path TEXT, metrics_sha256 TEXT
        );
        CREATE TABLE training_models (
          run_id TEXT PRIMARY KEY, set_id TEXT, task_type TEXT, status TEXT,
          model_path TEXT, model_sha256 TEXT, metrics_path TEXT, metrics_sha256 TEXT,
          metadata_json TEXT
        );
        CREATE TABLE onnx_models (
          run_id TEXT PRIMARY KEY, set_id TEXT, path TEXT, sha256 TEXT,
          manifest_path TEXT, manifest_sha256 TEXT, metadata_json TEXT
        );
        """
    )
    common = (
        RUN_ID,
        SET_ID,
        "pose",
        "success",
        str(package["weights"]),
        _sha(package["weights"]),
        str(package["metrics"]),
        _sha(package["metrics"]),
    )
    connection.execute(
        "INSERT INTO training_runs VALUES (?,?,?,?,?,?,?,?)", common
    )
    connection.execute(
        "INSERT INTO training_models VALUES (?,?,?,?,?,?,?,?,?)", common + ("{}",)
    )
    connection.execute(
        "INSERT INTO onnx_models VALUES (?,?,?,?,?,?,?)",
        (
            RUN_ID,
            SET_ID,
            str(package["onnx"]),
            _sha(package["onnx"]),
            str(package["onnx_manifest"]),
            _sha(package["onnx_manifest"]),
            "{}",
        ),
    )
    connection.commit()
    connection.close()
    backup = tmp_path / "registry.backup.sqlite"
    shutil.copy2(registry, backup)
    receipt = tmp_path / "registry.backup.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_id": "palette.registry_backup_receipt",
                "schema_version": 1,
                "status": "complete",
                "source_registry": str(registry.resolve()),
                "source_sha256": _sha(registry),
                "backup_path": str(backup.resolve()),
                "backup_sha256": _sha(backup),
            }
        ),
        encoding="utf-8",
    )
    return registry, receipt


def _args(
    *, registry: Path, backup_receipt: Path, destination_root: Path, apply: bool
) -> argparse.Namespace:
    return argparse.Namespace(
        registry=registry,
        model_run_id=RUN_ID,
        destination_model_root=destination_root,
        training_manifest_relative_path=Path("inputs") / f"{SET_ID}.manifest.json",
        training_report_relative_path=Path("training_report.yaml"),
        training_args_relative_path=Path("args.yaml"),
        training_runtime_receipt_relative_path=Path(
            "pose_training_runtime_receipt.json"
        ),
        model_stride=32,
        runtime_ultralytics_version=[],
        registry_backup_receipt=backup_receipt,
        dry_run=not apply,
        apply=apply,
    )


def test_dry_run_validates_without_publishing(tmp_path: Path) -> None:
    package = _source_package(tmp_path)
    registry, receipt = _registry(tmp_path, package)
    destination_root = tmp_path / "models"
    with patch.object(
        contract_module.importlib.metadata, "version", return_value="8.3.214"
    ):
        result = promote(
            _args(
                registry=registry,
                backup_receipt=receipt,
                destination_root=destination_root,
                apply=False,
            )
        )
    assert result["status"] == "planned"
    assert not destination_root.exists()


def test_apply_publishes_exact_copy_and_rebinds_registry(tmp_path: Path) -> None:
    package = _source_package(tmp_path)
    source_model_sha = _sha(package["weights"])
    registry, receipt = _registry(tmp_path, package)
    destination_root = tmp_path / "models"
    with patch.object(
        contract_module.importlib.metadata, "version", return_value="8.3.214"
    ):
        result = promote(
            _args(
                registry=registry,
                backup_receipt=receipt,
                destination_root=destination_root,
                apply=True,
            )
        )

    destination = destination_root / "pose" / SET_ID / RUN_ID
    assert result["status"] == "complete"
    assert _sha(destination / "weights" / "best.pt") == source_model_sha
    assert package["weights"].is_file()
    assert (destination / "pose_model_input_contract.json").is_file()
    assert (destination / "model_package_publication.json").is_file()
    canonical = destination / "exports" / "onnx" / f"{RUN_ID}.canonical.manifest.json"
    assert canonical.is_file()

    connection = sqlite3.connect(registry)
    connection.row_factory = sqlite3.Row
    training = connection.execute(
        "SELECT * FROM training_runs WHERE run_id=?", (RUN_ID,)
    ).fetchone()
    onnx = connection.execute(
        "SELECT * FROM onnx_models WHERE run_id=?", (RUN_ID,)
    ).fetchone()
    metadata = json.loads(
        connection.execute(
            "SELECT metadata_json FROM training_models WHERE run_id=?", (RUN_ID,)
        ).fetchone()[0]
    )
    connection.close()
    assert training["model_path"] == str(destination / "weights" / "best.pt")
    assert onnx["path"] == str(destination / "exports" / "onnx" / f"{RUN_ID}.onnx")
    assert onnx["manifest_path"] == str(canonical)
    assert metadata["canonical_package_promotion"]["selector_activation"] is False
