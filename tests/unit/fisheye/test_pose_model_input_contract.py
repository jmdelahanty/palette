from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.shared import pose_model_input_contract as mod
from fisheye.shared.pose_model_input_contract import (
    PoseModelInputContractError,
    build_historical_pose_model_input_contract,
    load_pose_model_input_contract,
    validate_pose_runtime_compatibility,
)


SET_ID = "pose_training_v1"
RUN_ID = "pose_run_v1"


def _package(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "model"
    (root / "weights").mkdir(parents=True)
    (root / "inputs").mkdir()
    weights = root / "weights" / "best.pt"
    weights.write_bytes(b"exact model bytes")
    (root / "inputs" / "training.manifest.json").write_text(
        json.dumps(
            {
                "task": "pose",
                "set_id": SET_ID,
                "input_format": "gray",
                "imgsz": [256, 256],
                "roi_pixel_contract_name": "mono_luma_v1",
                "pose_schema": {
                    "skeleton_id": "pose_skeleton_v1",
                    "kpt_shape": [2, 3],
                    "keypoint_labels": ["head", "tail"],
                    "skeleton": [[0, 1]],
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "training_report.yaml").write_text(
        """training_params:
  imgsz: 256
  rect: false
kpt_shape: [2, 3]
training_history:
  ultralytics_version: 8.3.214
  source_zarr_metadata:
    training.zarr:
      crop_info:
        roi_size: [512, 512]
""",
        encoding="utf-8",
    )
    (root / "args.yaml").write_text(
        "task: pose\nimgsz: 256\nrect: false\nmulti_scale: false\n",
        encoding="utf-8",
    )
    return root, weights


def _document(root: Path) -> dict[str, object]:
    return build_historical_pose_model_input_contract(
        set_id=SET_ID,
        run_id=RUN_ID,
        model_package_root=root,
        weights_relative_path=Path("weights/best.pt"),
        training_manifest_relative_path=Path("inputs/training.manifest.json"),
        training_report_relative_path=Path("training_report.yaml"),
        training_args_relative_path=Path("args.yaml"),
        model_stride=32,
    )


def test_historical_contract_derives_scale_matched_runtime_plan(tmp_path: Path) -> None:
    root, weights = _package(tmp_path)
    document = _document(root)
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(document), encoding="utf-8")

    binding = load_pose_model_input_contract(
        contract,
        model_path=weights,
        expected_set_id=SET_ID,
        expected_run_id=RUN_ID,
        expected_model_sha256=document["payload"]["model"]["weights"]["sha256"],
    )
    plan = binding.plan_for_native_shape((348, 348))

    assert binding.training_source_shape_hw == (512, 512)
    assert binding.network_shape_hw == (256, 256)
    assert plan.network_imgsz == 256
    assert plan.input_mode == "numpy-list"
    assert plan.model_stride == 32
    assert binding.runtime_ultralytics_versions == ("8.3.214",)
    assert (
        binding.preprocessing_probe["output_sha256"]
        == "d141f8e12a791d6b4b0c99ae3dfc24c6d6c11b63f9739df755d1d7bbe4b1d35a"
    )
    assert plan.classification == "scale_matched_diagnostic_not_training_context"
    assert plan.transform.to_attrs() == {
        "name": "pad_to_size",
        "native_shape_hw": [348, 348],
        "model_shape_hw": [512, 512],
        "pad_top": 82,
        "pad_bottom": 82,
        "pad_left": 82,
        "pad_right": 82,
        "coordinate_mapping": "native_xy = model_xy - [pad_left, pad_top]",
    }


def test_contract_rejects_tampered_evidence_and_digest(tmp_path: Path) -> None:
    root, weights = _package(tmp_path)
    document = _document(root)
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(document), encoding="utf-8")
    expected_sha = document["payload"]["model"]["weights"]["sha256"]

    (root / "args.yaml").write_text("task: pose\nimgsz: 512\n", encoding="utf-8")
    with pytest.raises(PoseModelInputContractError, match="digest changed"):
        load_pose_model_input_contract(
            contract,
            model_path=weights,
            expected_set_id=SET_ID,
            expected_run_id=RUN_ID,
            expected_model_sha256=expected_sha,
        )

    root, weights = _package(tmp_path / "second")
    document = _document(root)
    document["payload"]["training_input"]["network_shape_hw"] = [512, 512]
    contract = tmp_path / "tampered.json"
    contract.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(PoseModelInputContractError, match="payload digest is stale"):
        load_pose_model_input_contract(
            contract,
            model_path=weights,
            expected_set_id=SET_ID,
            expected_run_id=RUN_ID,
            expected_model_sha256=document["payload"]["model"]["weights"]["sha256"],
        )


def test_scale_matched_plan_rejects_native_roi_larger_than_training_source(
    tmp_path: Path,
) -> None:
    root, weights = _package(tmp_path)
    document = _document(root)
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(document), encoding="utf-8")
    binding = load_pose_model_input_contract(
        contract,
        model_path=weights,
        expected_set_id=SET_ID,
        expected_run_id=RUN_ID,
        expected_model_sha256=document["payload"]["model"]["weights"]["sha256"],
    )

    with pytest.raises(PoseModelInputContractError, match="cannot shrink"):
        binding.plan_for_native_shape((640, 640))


def test_runtime_requires_approved_version_and_exact_preprocessing_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, weights = _package(tmp_path)
    document = build_historical_pose_model_input_contract(
        set_id=SET_ID,
        run_id=RUN_ID,
        model_package_root=root,
        weights_relative_path=Path("weights/best.pt"),
        training_manifest_relative_path=Path("inputs/training.manifest.json"),
        training_report_relative_path=Path("training_report.yaml"),
        training_args_relative_path=Path("args.yaml"),
        model_stride=32,
        runtime_ultralytics_versions=("8.3.169",),
    )
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(document), encoding="utf-8")
    binding = load_pose_model_input_contract(
        contract,
        model_path=weights,
        expected_set_id=SET_ID,
        expected_run_id=RUN_ID,
        expected_model_sha256=document["payload"]["model"]["weights"]["sha256"],
    )
    assert binding.runtime_ultralytics_versions == ("8.3.169", "8.3.214")
    assert validate_pose_runtime_compatibility(binding)[
        "runtime_ultralytics_version"
    ] == "8.3.214"

    monkeypatch.setattr(mod.importlib.metadata, "version", lambda _name: "8.3.168")
    with pytest.raises(PoseModelInputContractError, match="not an approved runtime"):
        validate_pose_runtime_compatibility(binding)
