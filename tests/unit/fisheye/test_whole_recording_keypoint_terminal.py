from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import zarr

from fisheye.shared.model_input_transform import resolve_model_input_transform
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
    expected_transform = resolve_model_input_transform(
        (4, 4), mode="pad_to_size", model_hw=(32, 32)
    )

    def fake_inference(**kwargs):
        assert kwargs["output"] != analysis
        assert kwargs["output_parent"] == "keypoint_shard_runs"
        assert kwargs["coordinate_contract_mode"] == "legacy_noncanonical"
        assert kwargs["roi_cache_expected_archive_path"] == analysis
        assert kwargs["imgsz"] == 32
        assert kwargs["expected_model_stride"] == 32
        assert kwargs["model_input_transform_mode"] == "pad_to_size"
        local = zarr.open_group(str(kwargs["output"]), mode="a", use_consolidated=False)
        run = local.require_group("keypoint_shard_runs").create_group(
            kwargs["run_name"]
        )
        payload_sha256 = cache_document["array"]["sha256"]
        run.attrs.update(
            {
                "status": "complete",
                "stage_selector_eligible": False,
                RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE,
                "input_mode_effective": "tensor",
                "model_input_transform": expected_transform.to_attrs(),
                "model_input_stride": 32,
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
        }
        for name, values in arrays.items():
            run.create_array(name, data=values, chunks=values.shape)
        return SimpleNamespace(
            ok=True,
            keypoint_run=kwargs["run_name"],
            resolution_payload={
                "selected": {"model_path": "/models/pose.pt", "model_sha256": "d" * 64},
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
        pose_schema="traditional_v2",
        terminal_run_id="terminal-a",
        terminal_output=output,
        scratch_root=tmp_path / "scratch",
        batch_size=8,
        device="0",
        input_mode="tensor",
        model_input_size=32,
        model_input_transform_mode="pad_to_size",
        model_input_stride=32,
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
    assert preprocessing.document["model_input_stride"] == 32
    assert not (analysis / "keypoints_runs").exists()
    terminal = zarr.open_group(str(output), mode="r", use_consolidated=True)
    assert terminal["keypoint_terminal_runs/terminal-a/instance_key"][:].tolist() == [7]
