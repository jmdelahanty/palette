"""Path helper tests for detection training output defaults."""

from pathlib import Path
import sys
import json

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.training.train_detection import (
    _apply_zarr_loader_training_param_overrides,
    _build_default_run_name,
    _extract_runtime_imgsz,
    _infer_set_slug,
    _imgsz_to_config_value,
    _should_enable_rect_for_non_square_inputs,
    get_zarr_metadata,
    _snapshot_training_inputs,
    _strip_manifest_suffixes,
)


def test_strip_manifest_suffixes_handles_repeated_suffix() -> None:
    assert _strip_manifest_suffixes("detect_cedar_v001.manifest") == "detect_cedar_v001"
    assert _strip_manifest_suffixes("detect_cedar_v001.manifest.manifest") == "detect_cedar_v001"


def test_infer_set_slug_prefers_set_id_over_config_stem() -> None:
    cfg = Path("/tmp/detect_cedar_v001.manifest.yaml")
    assert _infer_set_slug("detect_cedar_v002", cfg) == "detect_cedar_v002"
    assert _infer_set_slug(None, cfg) == "detect_cedar_v001"


def test_snapshot_training_inputs_copies_config_manifest_and_invocation(tmp_path: Path) -> None:
    config = tmp_path / "detect.yaml"
    manifest = tmp_path / "detect.manifest.json"
    run_dir = tmp_path / "run"
    config.write_text("task: detect\n", encoding="utf-8")
    manifest.write_text('{"set_id":"detect_cedar_v001"}\n', encoding="utf-8")

    written = _snapshot_training_inputs(
        run_dir=run_dir,
        config_path=config,
        manifest_path=manifest,
        invocation_payload={"tool": "fisheye.training.train_detection", "argv": ["detect.yaml"]},
    )

    assert (run_dir / "inputs" / "detect.yaml").exists()
    assert (run_dir / "inputs" / "detect.manifest.json").exists()
    invocation_path = run_dir / "inputs" / "train_invocation.json"
    assert invocation_path.exists()
    payload = json.loads(invocation_path.read_text(encoding="utf-8"))
    assert payload["tool"] == "fisheye.training.train_detection"
    assert len(written) == 3


def test_build_default_run_name_uses_manifest_hints() -> None:
    run_name = _build_default_run_name(
        manifest_summary={
            "manifest_rig_name": "omnifin0",
            "manifest_dish_design": "cedar dish",
            "manifest_canvas_name": "DefaultScreen",
            "manifest_set_id": "detect_cedar_shadow_v005",
            "manifest_task": "detect",
            "manifest_sha256": "12345678abcdef00112233445566778899aabbccddeeff001122334455667788",
        },
        task_fallback="detect",
        timestamp="20260206-200000",
        pid=1234,
    )
    assert run_name == "omnifin0_cedar_dish_defaultscreen_v005_detect_20260206-200000_12345678"


def test_build_default_run_name_falls_back_to_set_slug_when_identity_missing() -> None:
    run_name = _build_default_run_name(
        manifest_summary={
            "manifest_set_slug": "detect_cedar_shadow_v001",
            "manifest_task": "detect",
            "manifest_sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        },
        task_fallback="detect",
        timestamp="20260206-200000",
        pid=1234,
    )
    assert run_name == "unknown_rig_cedar_shadow_v001_detect_20260206-200000_abcdef12"


def test_get_zarr_metadata_reads_merged_layout_counts(tmp_path: Path) -> None:
    zarr_path = tmp_path / "merged_detect.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    raw = root.require_group("raw_video")
    raw.require_array("images_ds", shape=(5, 8, 8), dtype=np.uint8, overwrite=True)[:] = 0
    raw.attrs["fps"] = 60.0
    crop_parent = root.require_group("crop_runs")
    crop_parent.attrs["latest"] = "merged_export_test"
    crop = crop_parent.require_group("merged_export_test")
    crop.require_array("bbox_norm_coords", shape=(5, 4), dtype=np.float32, overwrite=True)[:] = 0
    crop.require_array("frame_indices", shape=(5,), dtype=np.int64, overwrite=True)[:] = np.arange(5, dtype=np.int64)
    crop.attrs["detection_source_type"] = "manual"
    crop.attrs["includes_interpolated"] = False
    crop.attrs["detection_source_path"] = "merged://source_index"

    metadata = get_zarr_metadata([str(zarr_path)])
    entry = metadata[zarr_path.name]

    assert entry["video_frames"] == 5
    assert entry["frame_height"] == 8
    assert entry["frame_width"] == 8
    assert entry["crop_info"]["total_rois"] == 5
    assert entry["crop_info"]["source_type"] == "manual"


def test_should_enable_rect_for_non_square_inputs() -> None:
    metadata = {
        "square.zarr": {"frame_height": 640, "frame_width": 640},
        "rect.zarr": {"frame_height": 640, "frame_width": 1024},
    }
    assert _should_enable_rect_for_non_square_inputs(metadata) is True


def test_should_not_enable_rect_for_all_square_inputs() -> None:
    metadata = {
        "a.zarr": {"frame_height": 640, "frame_width": 640},
        "b.zarr": {"frame_height": 512, "frame_width": 512},
    }
    assert _should_enable_rect_for_non_square_inputs(metadata) is False


def test_apply_zarr_loader_training_param_overrides_sets_neutral_builtin_aug() -> None:
    params, custom = _apply_zarr_loader_training_param_overrides(
        {
            "model": "yolo11n.pt",
            "epochs": 10,
            "batch": 16,
            "imgsz": 640,
            "lr0": 0.001,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "patience": 5,
            "device": "0",
            "hsv_v": 0.15,
            "degrees": 5.0,
            "fliplr": 0.5,
        }
    )
    assert params["mosaic"] == 0.0
    assert params["mixup"] == 0.0
    assert params["cutmix"] == 0.0
    assert params["copy_paste"] == 0.0
    assert params["close_mosaic"] == 0
    assert params["augment"] is False
    assert "auto_augment" in params and params["auto_augment"] is None
    assert custom["hsv_v"] == 0.15
    assert custom["degrees"] == 5.0
    assert custom["fliplr"] == 0.5


def test_imgsz_to_config_value_returns_scalar_for_square() -> None:
    assert _imgsz_to_config_value(640, 640) == 640


def test_imgsz_to_config_value_returns_list_for_rectangular() -> None:
    assert _imgsz_to_config_value(640, 1024) == [640, 1024]


def test_extract_runtime_imgsz_reads_trainer_args() -> None:
    class _Args:
        imgsz = [512, 768]

    class _Trainer:
        args = _Args()

    class _Model:
        trainer = _Trainer()

    assert _extract_runtime_imgsz(_Model(), 640) == (512, 768)


def test_extract_runtime_imgsz_falls_back_when_trainer_missing() -> None:
    class _Model:
        trainer = None

    assert _extract_runtime_imgsz(_Model(), [320, 640]) == (320, 640)
