from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fisheye.shared import pose_model_input_contract as mod
from fisheye.shared.pose_model_input_contract import (
    PoseModelInputContractError,
    build_empirical_pose_runtime_profile,
    build_historical_pose_model_input_contract,
    build_pose_model_input_contract_v2,
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
    # Reconstruct the immutable historical artifact under its recorded builder
    # version. Runtime equivalence for the maintained 8.3.169 environment is
    # tested separately below against the artifact's exact probe.
    with patch.object(mod.importlib.metadata, "version", return_value="8.3.214"):
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


def _v2_document(root: Path) -> dict[str, object]:
    historical = _document(root)
    common = {
        "native_shape_hw": (348, 348),
        "model_stride": 32,
        "runtime_ultralytics_versions": ("8.3.169", "8.3.214"),
        "evidence_id": "batman_reviewed_input_comparison_v1",
        "evidence_artifact_path": (
            "/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/"
            "training/batman_training_canary_20260806_v1/"
            "2026-07-21T19-38-32Z_arena_2_Batman_reviewed_"
            "inference_candidates_v2.zarr"
        ),
        "evidence_receipt_sha256": (
            "c900247af3e2cc92be8a1c7367d30ffda8b0448b0dcad5ba1db79c8d909c54e4"
        ),
        "evidence_total_rows": 181,
    }
    rejected = build_empirical_pose_runtime_profile(
        profile_id="batman_348_numpy_512_to_256_v1",
        status="rejected",
        classification="empirically_rejected_scale_matched_resize",
        submitted_shape_hw=(512, 512),
        network_shape_hw=(256, 256),
        input_mode="numpy-list",
        evidence_successful_rows=0,
        **common,
    )
    accepted = build_empirical_pose_runtime_profile(
        profile_id="batman_348_tensor_pad_352_v1",
        status="accepted",
        classification="empirically_accepted_native_scale_stride_pad",
        submitted_shape_hw=(352, 352),
        network_shape_hw=(352, 352),
        input_mode="tensor",
        evidence_successful_rows=136,
        **common,
    )
    return build_pose_model_input_contract_v2(
        historical_contract=historical,
        runtime_profiles=(rejected, accepted),
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
    with patch.object(mod.importlib.metadata, "version", return_value="8.3.214"):
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
    ] == mod.importlib.metadata.version("ultralytics")

    monkeypatch.setattr(mod.importlib.metadata, "version", lambda _name: "8.3.168")
    with pytest.raises(PoseModelInputContractError, match="not an approved runtime"):
        validate_pose_runtime_compatibility(binding)


def test_v2_contract_selects_exact_empirical_tensor_profile(tmp_path: Path) -> None:
    root, weights = _package(tmp_path)
    document = _v2_document(root)
    contract = tmp_path / "contract_v2.json"
    contract.write_text(json.dumps(document), encoding="utf-8")
    binding = load_pose_model_input_contract(
        contract,
        model_path=weights,
        expected_set_id=SET_ID,
        expected_run_id=RUN_ID,
        expected_model_sha256=document["payload"]["model"]["weights"]["sha256"],
    )

    plan = binding.plan_for_native_shape((348, 348))

    assert binding.schema_version == 2
    assert len(binding.runtime_profiles) == 2
    assert plan.profile_id == "batman_348_tensor_pad_352_v1"
    assert plan.classification == "empirically_accepted_native_scale_stride_pad"
    assert plan.input_mode == "tensor"
    assert plan.network_shape_hw == (352, 352)
    assert plan.transform.to_attrs() == {
        "name": "pad_to_size",
        "native_shape_hw": [348, 348],
        "model_shape_hw": [352, 352],
        "pad_top": 2,
        "pad_bottom": 2,
        "pad_left": 2,
        "pad_right": 2,
        "coordinate_mapping": "native_xy = model_xy - [pad_left, pad_top]",
    }
    assert (
        validate_pose_runtime_compatibility(binding, plan)["preprocessing_probe"]
        == plan.preprocessing_probe
    )


def test_v2_contract_fails_closed_without_exact_accepted_native_shape(
    tmp_path: Path,
) -> None:
    root, weights = _package(tmp_path)
    document = _v2_document(root)
    contract = tmp_path / "contract_v2.json"
    contract.write_text(json.dumps(document), encoding="utf-8")
    binding = load_pose_model_input_contract(
        contract,
        model_path=weights,
        expected_set_id=SET_ID,
        expected_run_id=RUN_ID,
        expected_model_sha256=document["payload"]["model"]["weights"]["sha256"],
    )

    with pytest.raises(PoseModelInputContractError, match="no exact accepted profile"):
        binding.plan_for_native_shape((512, 512))


def test_v2_contract_rejects_recomputed_digest_profile_tampering(
    tmp_path: Path,
) -> None:
    root, weights = _package(tmp_path)
    document = _v2_document(root)
    document["payload"]["runtime_profiles"][1]["submitted_shape_hw"] = [384, 384]
    document["payload_digest"] = mod.canonical_json_sha256(document["payload"])
    contract = tmp_path / "tampered_v2.json"
    contract.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(PoseModelInputContractError, match="probe geometry"):
        load_pose_model_input_contract(
            contract,
            model_path=weights,
            expected_set_id=SET_ID,
            expected_run_id=RUN_ID,
            expected_model_sha256=document["payload"]["model"]["weights"]["sha256"],
        )


def test_checked_in_batman_v2_contract_is_reproducible() -> None:
    repository = Path(__file__).resolve().parents[3]
    directory = (
        repository / "docs" / "diagnostics" / "batman_keypoint_v2_candidate_20260805"
    )
    historical = json.loads(
        (directory / "pose_model_input_contract.json").read_text(encoding="utf-8")
    )
    current = json.loads(
        (directory / "pose_model_input_contract_v2.json").read_text(encoding="utf-8")
    )

    rebuilt = build_pose_model_input_contract_v2(
        historical_contract=historical,
        runtime_profiles=tuple(current["payload"]["runtime_profiles"]),
    )

    assert rebuilt == current


def test_checked_in_goodbatbadbat_384_successor_is_reproducible() -> None:
    repository = Path(__file__).resolve().parents[3]
    directory = (
        repository / "docs" / "diagnostics" / "batman_keypoint_v2_candidate_20260805"
    )
    historical = json.loads(
        (directory / "pose_model_input_contract.json").read_text(encoding="utf-8")
    )
    current = json.loads(
        (directory / "pose_model_input_contract_v2.json").read_text(encoding="utf-8")
    )
    successor = json.loads(
        (
            directory
            / "pose_model_input_contract_v2_goodbatbadbat_384_v1.json"
        ).read_text(encoding="utf-8")
    )
    profile = build_empirical_pose_runtime_profile(
        profile_id="goodbatbadbat_384_tensor_identity_v1",
        status="accepted",
        classification="empirically_accepted_native_identity_stride_aligned",
        native_shape_hw=(384, 384),
        submitted_shape_hw=(384, 384),
        network_shape_hw=(384, 384),
        model_stride=32,
        input_mode="tensor",
        runtime_ultralytics_versions=("8.3.169",),
        evidence_id="goodbatbadbat_hybrid_crop_pose_384_visual_review_v1",
        evidence_artifact_path=(
            "/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/"
            "training/goodbatbadbat_hybrid_pose_384_canary_20260815_v1"
        ),
        evidence_receipt_sha256=(
            "fbfd4020e8cd4bb75818e63698b4cbd4cca4a8a51d3c2eea373cac5c95a9f27e"
        ),
        evidence_total_rows=256,
        evidence_successful_rows=253,
    )
    rebuilt = build_pose_model_input_contract_v2(
        historical_contract=historical,
        runtime_profiles=tuple(current["payload"]["runtime_profiles"])
        + (profile,),
    )

    assert rebuilt == successor
    assert successor["payload_digest"] == (
        "cd2b6050ef24cbcaf70cb5c73a4812225077739c5639ba202d701d6e4ca568ef"
    )
    assert successor["payload"]["runtime_profiles"][-1]["native_to_submitted"][
        "policy"
    ] == "identity"
