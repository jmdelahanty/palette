"""Path helper tests for detection training output defaults."""

from pathlib import Path
import sys
import json

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.training.train_detection import (
    _build_default_run_name,
    _infer_set_slug,
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
            "manifest_dish_design": "cedar dish",
            "manifest_canvas_name": "DefaultScreen",
            "manifest_task": "detect",
        },
        task_fallback="detect",
        timestamp="20260206-200000",
        pid=1234,
    )
    assert run_name == "cedar_dish_defaultscreen_detect_20260206-200000_1234"


def test_build_default_run_name_falls_back_to_set_slug_when_identity_missing() -> None:
    run_name = _build_default_run_name(
        manifest_summary={
            "manifest_set_slug": "detect_cedar_shadow_v001",
            "manifest_task": "detect",
        },
        task_fallback="detect",
        timestamp="20260206-200000",
        pid=1234,
    )
    assert run_name == "detect_cedar_shadow_v001_detect_20260206-200000_1234"


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
    assert entry["crop_info"]["total_rois"] == 5
    assert entry["crop_info"]["source_type"] == "manual"
