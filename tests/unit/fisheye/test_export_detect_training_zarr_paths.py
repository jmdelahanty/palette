"""Path resolution tests for merged detection-training Zarr export."""

from pathlib import Path
import sys

import numpy as np
import zarr
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.diagnostics.prepare_detect_training import DatasetManifest
from fisheye.utils.export_detect_training_zarr import (
    _copy_indexed_frames,
    _ensure_suffix,
    _normalize_manifest_stem,
    _resolve_detection_source_path,
    _resolve_default_dataset_root,
    _write_merged_config,
)


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.vindex = self

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, item, value) -> None:
        self.data[item] = value


def _dataset_manifest(detection_source_path: str | None) -> DatasetManifest:
    return DatasetManifest(
        name="sample",
        zarr_path="/tmp/sample.zarr",
        dataset_id="sample",
        session_uuid="sample",
        crop_run="crop_001",
        bbox_array_path="crop_runs/crop_001/bbox_norm_coords",
        detection_source_type="manual",
        detection_source_path=detection_source_path,
        includes_interpolated=False,
        input_format="gray",
        images_ds_shape=[8, 8],
        total_bboxes=3,
        invalid_bboxes=0,
    )


def test_resolve_detection_source_path_ignores_group_path_and_uses_crop_array(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    crop = root.create_group("crop_runs").create_group("crop_001")
    crop.create_array("bbox_norm_coords", data=np.zeros((3, 4), dtype=np.float32))
    crop.create_array("detection_source", data=np.array([0, 0, 1], dtype=np.int8))
    root.create_group("refined_detect_runs").create_group("refined_detect_001").create_group("manual")

    manifest = _dataset_manifest("refined_detect_runs/refined_detect_001/manual")
    resolved = _resolve_detection_source_path(root, manifest)

    assert resolved == "crop_runs/crop_001/detection_source"


def test_resolve_detection_source_path_handles_manifest_array_path_suffix(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    crop = root.create_group("crop_runs").create_group("crop_001")
    crop.create_array("bbox_norm_coords", data=np.zeros((3, 4), dtype=np.float32))
    manual = root.create_group("refined_detect_runs").create_group("refined_detect_001").create_group("manual")
    manual.create_array("detection_source", data=np.array([0, 0, 0], dtype=np.int8))

    manifest = _dataset_manifest("refined_detect_runs/refined_detect_001/manual")
    resolved = _resolve_detection_source_path(root, manifest)

    assert resolved == "refined_detect_runs/refined_detect_001/manual/detection_source"


def test_copy_indexed_frames_reports_progress_callback() -> None:
    src = _FakeArray(np.arange(20, dtype=np.int32).reshape(10, 2))
    dst = _FakeArray(np.zeros((4, 2), dtype=np.int32))
    frame_indices = np.array([2, 5, 7, 9], dtype=np.int64)
    copied_batches: list[int] = []

    _copy_indexed_frames(
        src,
        frame_indices,
        dst,
        dest_start=0,
        batch_size=2,
        on_copied=lambda n: copied_batches.append(n),
    )

    np.testing.assert_array_equal(dst.data, src.data[frame_indices])
    assert copied_batches == [2, 2]


def test_normalize_manifest_stem_strips_manifest_suffix_repeatedly() -> None:
    assert _normalize_manifest_stem("detect_cedar_v001.manifest") == "detect_cedar_v001"
    assert _normalize_manifest_stem("detect_cedar_v001.manifest.manifest") == "detect_cedar_v001"


def test_ensure_suffix_is_idempotent() -> None:
    assert _ensure_suffix("detect_cedar_v001", "_merged") == "detect_cedar_v001_merged"
    assert _ensure_suffix("detect_cedar_v001_merged", "_merged") == "detect_cedar_v001_merged"


def test_write_merged_config_creates_dummy_paths_next_to_output(tmp_path: Path) -> None:
    source_config = tmp_path / "source.yaml"
    source_config.write_text(
        yaml.safe_dump(
            {
                "train": "./dummy_train.txt",
                "val": "./dummy_val.txt",
                "datasets": {},
                "training_params": {"model": "yolo11n.pt"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    out_config = tmp_path / "export" / "merged.yaml"
    created = _write_merged_config(
        source_config_path=source_config,
        out_config=out_config,
        merged_zarr=tmp_path / "merged.zarr",
        dataset_name="merged_set",
        source_type="manual",
        input_format="gray",
        train_ratio=0.8,
        val_ratio=0.2,
        random_seed=42,
    )

    assert (out_config.parent / "dummy_train.txt").exists()
    assert (out_config.parent / "dummy_val.txt").exists()
    assert {(out_config.parent / "dummy_train.txt").resolve(), (out_config.parent / "dummy_val.txt").resolve()} == {
        path.resolve() for path in created
    }


def test_resolve_default_dataset_root_prefers_env_override(monkeypatch, tmp_path: Path) -> None:
    expected = tmp_path / "training_datasets"
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(expected))
    assert _resolve_default_dataset_root() == expected.resolve()
