"""Path helper tests for detection training output defaults."""

from pathlib import Path
import sys
import json

import numpy as np
import torch
import zarr
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from fisheye.training import train_detection as td

from fisheye.training.train_detection import (
    _cleanup_trainer_dataloaders,
    _apply_zarr_loader_training_param_overrides,
    _build_default_run_name,
    _export_detection_artifacts,
    _extract_runtime_imgsz,
    _infer_set_slug,
    _imgsz_to_config_value,
    _should_enable_rect_for_non_square_inputs,
    ChunkAwareBatchSampler,
    InputPipelineProfiler,
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
            "chunk_cache_size": 64,
            "persistent_workers": True,
            "chunk_locality_sampling": True,
            "num_workers": 8,
            "prefetch_factor": 3,
            "deterministic_val": True,
            "val_num_workers": 0,
        }
    )
    assert params["mosaic"] == 0.0
    assert params["mixup"] == 0.0
    assert params["cutmix"] == 0.0
    assert params["copy_paste"] == 0.0
    assert params["close_mosaic"] == 0
    assert params["augment"] is False
    assert "auto_augment" in params and params["auto_augment"] is None
    assert "chunk_cache_size" not in params
    assert "persistent_workers" not in params
    assert "chunk_locality_sampling" not in params
    assert "num_workers" not in params
    assert "prefetch_factor" not in params
    assert "deterministic_val" not in params
    assert "val_num_workers" not in params
    assert custom["hsv_v"] == 0.15
    assert custom["degrees"] == 5.0
    assert custom["fliplr"] == 0.5
    assert custom["chunk_cache_size"] == 64
    assert custom["persistent_workers"] is True
    assert custom["chunk_locality_sampling"] is True
    assert custom["num_workers"] == 8
    assert custom["prefetch_factor"] == 3
    assert custom["deterministic_val"] is True
    assert custom["val_num_workers"] == 0


def test_chunk_aware_batch_sampler_groups_by_chunk_when_not_shuffling() -> None:
    class _Dataset:
        indices = [("a", 0), ("a", 1), ("a", 2), ("a", 3)]
        frame_index_cache = {"a": np.array([0, 1, 64, 65], dtype=np.int64)}
        detect_frame_chunk_len = {"a": 64}

    sampler = ChunkAwareBatchSampler(_Dataset(), batch_size=2, seed=42, shuffle=False)
    batches = list(iter(sampler))
    assert batches == [[0, 1], [2, 3]]
    assert len(sampler) == 2


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


def test_input_pipeline_profiler_collects_stage_timings() -> None:
    profiler = InputPipelineProfiler(enabled=True)
    profiler.record_dataset_sample(
        {
            "samples": 1,
            "zarr_read_s": 0.001,
            "augment_preprocess_s": 0.002,
            "getitem_total_s": 0.004,
        }
    )
    profiler.record_collate(0.003, batch_size=8)
    batch = {"img": torch.zeros((8, 3, 16, 16), dtype=torch.float32)}
    profiler.record_batch_wait(0.005, batch)
    profiler.record_preprocess_to_device(0.006, batch)

    summary = profiler.summary()
    assert summary["enabled"] is True
    assert summary["stages"]["dataset_zarr_read"]["calls"] == 1
    assert summary["stages"]["dataset_augment_preprocess"]["calls"] == 1
    assert summary["stages"]["dataset_getitem_total"]["calls"] == 1
    assert summary["stages"]["collate"]["calls"] == 1
    assert summary["stages"]["dataloader_wait"]["calls"] == 1
    assert summary["stages"]["preprocess_to_device"]["calls"] == 1
    assert summary["stages"]["dataloader_wait"]["samples"] == 8
    assert summary["stages"]["preprocess_to_device"]["samples"] == 8


def test_cleanup_trainer_dataloaders_shuts_down_worker_iterators() -> None:
    class _Iterator:
        def __init__(self):
            self.closed = False

        def _shutdown_workers(self):
            self.closed = True

    class _Loader:
        def __init__(self, iterator):
            self._iterator = iterator

    class _Validator:
        def __init__(self, loader):
            self.dataloader = loader
            self.test_loader = None

    class _Trainer:
        def __init__(self, loader):
            self.train_loader = loader
            self.test_loader = None
            self.validator = _Validator(loader)

    class _Model:
        def __init__(self, trainer):
            self.trainer = trainer

    iterator = _Iterator()
    loader = _Loader(iterator)
    model = _Model(_Trainer(loader))

    cleaned = _cleanup_trainer_dataloaders(model)
    assert cleaned == 1
    assert iterator.closed is True
    assert loader._iterator is None


def test_input_pipeline_profiler_disabled_summary() -> None:
    profiler = InputPipelineProfiler(enabled=False)
    assert profiler.summary() == {"enabled": False}


def test_export_detection_artifacts_reuses_existing_onnx_for_trt_only(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    weights_path = run_dir / "weights" / "best.pt"
    onnx_path = run_dir / "exports" / "onnx" / "run123.onnx"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_bytes(b"weights")
    onnx_path.write_bytes(b"onnx")

    calls: list[str] = []

    def _fake_run_subprocess(cmd, console, label, log_path=None):
        calls.append(str(label))
        return True

    monkeypatch.setattr(td, "_run_subprocess", _fake_run_subprocess)

    class _Args:
        export_onnx = False
        export_trt = True
        onnx_path = None
        onnx_opset = 11
        onnx_simplify = False
        nms_conf = 0.8
        nms_iou = 0.65
        nms_topk = 1
        trt_precision = "fp16"
        trtexec = None
        trt_cuda_graph = False
        trt_profiling = False
        trt_verbose = False

    out = _export_detection_artifacts(
        run_dir=run_dir,
        run_id="run123",
        weights_path=weights_path,
        training_params={"imgsz": 640, "device": "0"},
        export_imgsz=None,
        args=_Args(),
        manifest_summary={},
        console=Console(record=True),
    )

    assert out["onnx_source"] == "existing"
    assert Path(out["onnx_path"]) == onnx_path
    assert all(label != "ONNX export" for label in calls)
    assert any(label == "TensorRT export" for label in calls)


def test_export_detection_artifacts_passes_onnx_metadata_args(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    weights_path = run_dir / "weights" / "best.pt"
    onnx_path = run_dir / "exports" / "onnx" / "run123.onnx"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_bytes(b"weights")
    onnx_path.write_bytes(b"onnx")

    args = type("Args", (), {})()
    args.export_onnx = True
    args.export_trt = False
    args.onnx_path = str(onnx_path)
    args.onnx_opset = 11
    args.onnx_simplify = False
    args.nms_conf = 0.8
    args.nms_iou = 0.65
    args.nms_topk = 1
    args.trt_precision = "fp16"
    args.trtexec = None
    args.trt_cuda_graph = False
    args.trt_profiling = False
    args.trt_verbose = False

    out = _export_detection_artifacts(
        run_dir=run_dir,
        run_id="run123",
        weights_path=weights_path,
        training_params={"imgsz": 640, "device": "0"},
        export_imgsz=None,
        args=args,
        manifest_summary={
            "manifest_set_id": "detect_cedar_shadow_v007",
            "manifest_sha256": "ABCDEF1234",
        },
        console=Console(record=True),
    )
    onnx_cmd = out["onnx_command"]
    assert "--meta-run-id" in onnx_cmd
    assert "--meta-set-id" in onnx_cmd
    assert "--meta-manifest-sha256" in onnx_cmd
    assert onnx_cmd[onnx_cmd.index("--meta-run-id") + 1] == "run123"
    assert onnx_cmd[onnx_cmd.index("--meta-set-id") + 1] == "detect_cedar_shadow_v007"
    assert onnx_cmd[onnx_cmd.index("--meta-manifest-sha256") + 1] == "abcdef1234"


def test_collect_onnx_output_contract_reads_names_shapes_and_dtypes(tmp_path: Path) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except Exception:
        return

    out0 = helper.make_tensor_value_info("num_dets", TensorProto.INT32, [1, 1])
    out1 = helper.make_tensor_value_info("bboxes", TensorProto.FLOAT, [1, 1, 4])
    graph = helper.make_graph(nodes=[], name="g", inputs=[], outputs=[out0, out1], initializer=[])
    model = helper.make_model(graph)
    onnx_path = tmp_path / "toy.onnx"
    onnx.save(model, str(onnx_path))

    contract = td._collect_onnx_output_contract(onnx_path)
    assert contract[0]["name"] == "num_dets"
    assert contract[0]["shape"] == [1, 1]
    assert contract[0]["dtype"] == "INT32"
    assert contract[1]["name"] == "bboxes"
    assert contract[1]["shape"] == [1, 1, 4]
    assert contract[1]["dtype"] == "FLOAT"


def test_parse_trtexec_device_info_text_extracts_structured_fields() -> None:
    raw = "\n".join(
        [
            "[02/07/2026-17:10:05] [I] TensorRT version: 10.0.1",
            "[02/07/2026-17:10:05] [I] Selected Device: NVIDIA RTX A6000",
            "[02/07/2026-17:10:05] [I] Selected Device ID: 0",
            "[02/07/2026-17:10:05] [I] Selected Device UUID: GPU-abc123",
            "[02/07/2026-17:10:05] [I] Compute Capability: 8.6",
            "[02/07/2026-17:10:05] [I] SMs: 84",
            "[02/07/2026-17:10:05] [I] Device Global Memory: 48536 MiB",
            "[02/07/2026-17:10:05] [I] Memory Bus Width: 384 bits (ECC disabled)",
        ]
    )
    info = td._parse_trtexec_device_info_text(raw)
    assert info["trtexec_reported_version"] == "10.0.1"
    assert info["selected_device_name"] == "NVIDIA RTX A6000"
    assert info["selected_device_id"] == 0
    assert info["selected_device_uuid"] == "GPU-abc123"
    assert info["compute_capability"] == "8.6"
    assert info["sm_count"] == 84
    assert info["device_global_memory_mib"] == 48536
    assert info["memory_bus_width_bits"] == 384
