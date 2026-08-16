from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import zarr

from fisheye.shared.model_input_transform import resolve_model_input_transform
from fisheye.shared.pose_model_input_contract import PoseModelInputRuntimePlan
from fisheye.shared.zarr.keypoint_manifest import keypoint_preprocessing_from_manifest
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)
from fisheye.utils import run_whole_recording_keypoint_terminal as mod
from tests.unit.fisheye.test_keypoint_publication import _pose_binding


def test_terminal_runner_stages_cache_and_never_writes_analysis_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    analysis = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "recording_id": "recording-a",
            "width": 4512,
            "height": 4512,
            "total_frames": 12,
        }
    )
    root.create_group("crop_runs").create_group("crop_v2")
    cache_payload = tmp_path / "cache.bin"
    cache_payload.write_bytes(bytes(range(16)))
    cache_manifest = tmp_path / "cache.json"
    cache_document = {
        "array": {
            "bin_path": cache_payload.name,
            "shape": [1, 4, 4],
            "dtype": "uint8",
            "sha256": hashlib.sha256(cache_payload.read_bytes()).hexdigest(),
        },
        "source": {"archive_path": str(analysis), "crop_run_name": "crop_v2"},
        "builder": {"pixel_contract": {"domain": "roi_pixels"}},
    }
    cache_manifest.write_text(json.dumps(cache_document), encoding="utf-8")
    binding = _pose_binding()
    expected_model_path = tmp_path / "model" / "weights" / "best.pt"
    expected_model_path.parent.mkdir(parents=True)
    expected_model_path.write_bytes(b"model")
    expected_model_sha256 = "d" * 64
    model_input_contract = tmp_path / "pose_model_input_contract.json"
    model_input_contract.write_text("{}", encoding="utf-8")
    expected_transform = resolve_model_input_transform(
        (4, 4), mode="pad_to_size", model_hw=(32, 32)
    )
    runtime_plan = PoseModelInputRuntimePlan(
        transform=expected_transform,
        network_shape_hw=(16, 16),
        model_stride=16,
        input_mode="numpy-list",
        profile_id="scale_matched_center_pad_ultralytics_v1",
        classification="scale_matched_diagnostic_not_training_context",
        contract_path=model_input_contract,
        contract_sha256="c" * 64,
        contract_payload_digest="e" * 64,
    )
    contract = SimpleNamespace(
        ultralytics_version="8.3.214",
        plan_for_native_shape=lambda _shape: runtime_plan,
        to_json=lambda: {
            "path": str(model_input_contract),
            "sha256": "c" * 64,
            "payload_digest": "e" * 64,
        },
    )
    monkeypatch.setattr(
        mod, "load_pose_model_input_contract", lambda *_args, **_kwargs: contract
    )
    monkeypatch.setattr(
        mod,
        "validate_pose_runtime_compatibility",
        lambda _binding, _runtime_plan: {
            "runtime_ultralytics_version": "8.3.214",
            "approved_runtime_ultralytics_versions": ["8.3.169", "8.3.214"],
            "preprocessing_probe": {"output_sha256": "a" * 64},
        },
    )

    def fake_inference(**kwargs):
        assert kwargs["output"] != analysis
        assert kwargs["output_parent"] == "keypoint_shard_runs"
        assert kwargs["coordinate_contract_mode"] == "legacy_noncanonical"
        assert kwargs["roi_cache_expected_archive_path"] == analysis
        assert kwargs["imgsz"] == 16
        assert kwargs["model_input_size"] == 32
        assert kwargs["expected_model_stride"] == 16
        assert kwargs["model_input_transform_mode"] == "pad_to_size"
        local = zarr.open_group(str(kwargs["output"]), mode="a", use_consolidated=False)
        assert local.attrs["recording_id"] == "recording-a"
        assert local.attrs["width"] == 4512
        assert local.attrs["height"] == 4512
        assert local.attrs["total_frames"] == 12
        run = local.require_group("keypoint_shard_runs").create_group(
            kwargs["run_name"]
        )
        payload_sha256 = cache_document["array"]["sha256"]
        run.attrs.update(
            {
                "status": "complete",
                "stage_selector_eligible": False,
                RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE,
                "input_mode_effective": "numpy-list",
                "model_input_transform": expected_transform.to_attrs(),
                "model_input_stride": 16,
                "ultralytics_version": "8.3.214",
                "parameters": {
                    "imgsz": 16,
                    "model_input_size": 32,
                    "model_predict_rect": False,
                },
                "confidence_threshold": 0.25,
                "iou_threshold": 0.5,
                "max_detections": 1,
                "source_roi_cache_staging": {
                    "staged": True,
                    "copy": {
                        "verification": "single_pass_copy_stream_sha256_v1",
                        "source_sha256": payload_sha256,
                        "staged_sha256": payload_sha256,
                    },
                },
            }
        )
        arrays = {
            "instance_key": np.asarray([7], dtype=np.uint64),
            "source_crop_row_ids": np.asarray([0], dtype=np.int64),
            "source_acquisition_frame_index": np.asarray([3], dtype=np.int64),
            "frame_indices": np.asarray([3], dtype=np.int64),
            "keypoints_roi": np.zeros((1, 3, 2), dtype=np.float64),
            "keypoints_img": np.zeros((1, 3, 2), dtype=np.float64),
            "keypoint_confidences": np.ones((1, 3), dtype=np.float64),
            "confidence": np.ones(1, dtype=np.float64),
            "pose_bbox_xyxy_roi": np.zeros((1, 4), dtype=np.float64),
            "pose_bbox_xyxy_img": np.zeros((1, 4), dtype=np.float64),
            "detection_success": np.ones(1, dtype=bool),
            "pose_failure_codes": np.zeros(1, dtype=np.uint8),
        }
        for name, values in arrays.items():
            run.create_array(name, data=values, chunks=values.shape)
        return SimpleNamespace(
            ok=True,
            keypoint_run=kwargs["run_name"],
            resolution_payload={
                "selected": {
                    "model_path": str(expected_model_path),
                    "model_sha256": expected_model_sha256,
                },
                "artifacts": {"model_pose_schema_binding": binding},
            },
            to_dict=lambda: {},
        )

    monkeypatch.setattr(mod, "run_keypoints_with_registry_model", fake_inference)
    output = tmp_path / "terminal.zarr"
    receipt = mod.run_whole_recording_keypoint_terminal(
        recording_id="recording-a",
        recording_dir=tmp_path / "recording",
        analysis_zarr=analysis,
        crop_run="crop_v2",
        cache_manifest=cache_manifest,
        registry=tmp_path / "registry.sqlite",
        model_set_id="pose-set",
        model_run_id="pose-run",
        expected_model_path=expected_model_path,
        expected_model_sha256=expected_model_sha256,
        model_input_contract=model_input_contract,
        pose_schema="traditional_v2",
        terminal_run_id="terminal-a",
        terminal_output=output,
        scratch_root=tmp_path / "scratch",
        batch_size=8,
        device="0",
        input_mode="numpy-list",
        model_input_size=32,
        network_input_size=16,
        model_input_transform_mode="pad_to_size",
        model_input_stride=16,
    )

    payload = receipt["payload"]
    assert payload["analysis_zarr"] == str(analysis.resolve())
    assert payload["production_state_changes"] == []
    assert payload["cache"]["payload_size_bytes"] == 16
    assert payload["cache"]["dtype"] == "|u1"
    assert (
        payload["cache"]["staging_verification"]["copy"]["verification"]
        == "single_pass_copy_stream_sha256_v1"
    )
    preprocessing = keypoint_preprocessing_from_manifest(payload["preprocessing"])
    assert preprocessing.document["model_input_transform"] == (
        expected_transform.to_attrs()
    )
    assert preprocessing.document["model_input_stride"] == 16
    assert preprocessing.document["model_input_runtime"]["network_shape_hw"] == [
        16,
        16,
    ]
    assert payload["pose_failure_codes"]["histogram"]["none"] == 1
    assert (
        payload["row_terminal_semantics"]
        == "every_crop_row_present_with_exact_pose_failure_code_v2"
    )
    assert not (analysis / "keypoints_runs").exists()
    terminal = zarr.open_group(str(output), mode="r", use_consolidated=True)
    assert terminal["keypoint_terminal_runs/terminal-a/instance_key"][:].tolist() == [7]


def test_stage_crop_shell_derives_root_dimensions_from_unstaged_raw_video(
    tmp_path: Path,
) -> None:
    analysis = tmp_path / "recording_analysis.zarr"
    source = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    raw_video = source.create_group("raw_video")
    raw_video.create_array(
        "images_full",
        data=np.zeros((2, 3, 4), dtype=np.uint8),
        chunks=(1, 3, 4),
    )
    crop = source.create_group("crop_runs").create_group("crop_v2")
    crop.attrs["sentinel"] = "copied"

    staged_path = mod._stage_crop_shell(  # noqa: SLF001
        analysis,
        crop_run="crop_v2",
        destination=tmp_path / "scratch",
    )

    staged = zarr.open_group(str(staged_path), mode="r", use_consolidated=False)
    assert staged.attrs["total_frames"] == 2
    assert staged.attrs["video_height"] == 3
    assert staged.attrs["video_width"] == 4
    assert staged["crop_runs/crop_v2"].attrs["sentinel"] == "copied"
    assert "raw_video" not in staged
