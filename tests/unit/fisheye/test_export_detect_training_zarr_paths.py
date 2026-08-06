"""Path resolution tests for merged detection-training Zarr export."""

from pathlib import Path
import sys

import numpy as np
import pytest
import zarr
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.diagnostics.prepare_detect_training import DatasetManifest, TrainingManifest
from fisheye.shared.zarr.detect_frame_decisions import (
    clear_detect_frame_decision,
    set_detect_frame_negative,
)
from fisheye.training.zarr_yolo_dataset_loader import (
    ZarrDatasetConfig,
    create_zarr_dataset,
)
from fisheye.utils.export_detect_training_zarr import (
    MergeResult,
    MergeSourceSpec,
    _build_merged_manifest_payload,
    _copy_indexed_frames,
    _discover_merge_sources,
    _ensure_suffix,
    _export_merged,
    _normalize_manifest_stem,
    _resolve_detection_source_path,
    _resolve_default_dataset_root,
    _write_merged_config,
    validate_merged_training_zarr,
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


def _write_detect_source_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    raw = root.create_group("raw_video")
    raw.attrs["pixel_contract_name"] = "legacy_gray_uint8_v1"
    raw.attrs["decode_backend"] = "legacy_import_gray"
    raw.create_array(
        "images_ds",
        data=np.arange(6 * 8 * 8, dtype=np.uint8).reshape(6, 8, 8),
        chunks=(3, 8, 8),
    )

    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_001"
    refined = refined_parent.create_group("refined_detect_001")
    instances = refined.create_group("instances")
    instances.create_array(
        "bbox_norm_coords",
        data=np.zeros((4, 4), dtype=np.float32),
        chunks=(4, 4),
    )
    instances.create_array(
        "frame_indices",
        data=np.array([1, 2, 4, 5], dtype=np.int64),
        chunks=(4,),
    )
    instances.create_array(
        "source_kind_codes",
        data=np.array([1, 3, 1, 1], dtype=np.int8),
        chunks=(4,),
    )
    instances.create_array(
        "manual_edit_flags",
        data=np.array([False, True, False, False], dtype=bool),
        chunks=(4,),
    )
    instances.create_array(
        "refined_row_ids",
        data=np.array([100, 101, 102, 103], dtype=np.int64),
        chunks=(4,),
    )
    instances.create_array(
        "source_detect_row_index",
        data=np.array([200, -1, 202, 203], dtype=np.int32),
        chunks=(4,),
    )
    instances.create_array(
        "instance_key",
        data=np.array([1000, 1001, 1002, 1003], dtype=np.uint64),
        chunks=(4,),
    )


def _write_multi_instance_negative_source_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update({"width": 8, "height": 8, "n_frames": 4})
    raw = root.create_group("raw_video")
    raw.attrs.update(
        {"pixel_contract_name": "legacy_gray_uint8_v1", "decode_backend": "test"}
    )
    raw.create_array(
        "images_ds",
        data=np.stack(
            [np.full((8, 8), value, dtype=np.uint8) for value in range(4)],
            axis=0,
        ),
        chunks=(2, 8, 8),
    )
    raw.create_array(
        "original_frame_indices",
        data=np.asarray([100, 101, 102, 103], dtype=np.int64),
        chunks=(4,),
    )
    parent = root.create_group("refined_detect_runs")
    parent.attrs["latest"] = "reviewed"
    refined = parent.create_group("reviewed")
    instances = refined.create_group("instances")
    instances.create_array(
        "bbox_norm_coords",
        data=np.asarray(
            [
                [0.25, 0.25, 0.2, 0.2],
                [0.40, 0.40, 0.2, 0.2],
                [0.70, 0.70, 0.1, 0.1],
            ],
            dtype=np.float32,
        ),
        chunks=(3, 4),
    )
    instances.create_array(
        "frame_indices",
        data=np.asarray([0, 2, 2], dtype=np.int32),
        chunks=(3,),
    )
    instances.create_array(
        "source_kind_codes",
        data=np.asarray([3, 3, 3], dtype=np.int8),
        chunks=(3,),
    )
    instances.create_array(
        "manual_edit_flags",
        data=np.asarray([True, True, True], dtype=bool),
        chunks=(3,),
    )
    instances.create_array(
        "refined_row_ids",
        data=np.asarray([10, 11, 12], dtype=np.int64),
        chunks=(3,),
    )
    instances.create_array(
        "source_detect_row_index",
        data=np.asarray([-1, -1, -1], dtype=np.int32),
        chunks=(3,),
    )
    instances.create_array(
        "class_ids",
        data=np.asarray([0, 1, 2], dtype=np.int32),
        chunks=(3,),
    )
    instances.create_array(
        "instance_key",
        data=np.asarray([2000, 2001, 2002], dtype=np.uint64),
        chunks=(3,),
    )
    for frame_index in (1, 3):
        set_detect_frame_negative(
            root,
            source_refined_detect_run="reviewed",
            n_frames=4,
            frame_index=frame_index,
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


def test_export_merged_detect_training_zarr_preserves_source_row_lineage(tmp_path: Path) -> None:
    source_zarr = tmp_path / "source_detect.zarr"
    out_zarr = tmp_path / "merged_detect.zarr"
    manifest_path = tmp_path / "detect.manifest.json"
    _write_detect_source_zarr(source_zarr)
    manifest_path.write_text("{}", encoding="utf-8")

    manifest = TrainingManifest(
        created_at_utc="2026-02-06T00:00:00+00:00",
        task="detect",
        source_type="refined",
        input_format="gray",
        imgsz=[640, 640],
        datasets=[
            DatasetManifest(
                name="source_dataset",
                zarr_path=str(source_zarr),
                dataset_id="source_dataset",
                session_uuid="source_dataset",
                crop_run=None,
                bbox_array_path="refined_detect_runs/refined_detect_001/instances/bbox_norm_coords",
                detection_source_type="refined",
                detection_source_path="refined_detect_runs/refined_detect_001/instances",
                includes_interpolated=False,
                input_format="gray",
                images_ds_shape=[8, 8],
                total_bboxes=4,
                invalid_bboxes=0,
            )
        ],
        provenance_policy="warn",
        set_name="cedar_shadow",
        set_version=1,
        set_id="detect_cedar_shadow_v001",
    )

    result = _export_merged(
        manifest=manifest,
        manifest_path=manifest_path,
        out_zarr=out_zarr,
        merged_dataset_id=None,
        overwrite=True,
        train_ratio=0.5,
        val_ratio=0.5,
        test_ratio=0.0,
        seed=42,
        include_rgb=False,
        copy_batch_size=2,
        invocation={},
    )

    assert result.total_samples == 4
    assert result.total_interpolated == 0
    validate_merged_training_zarr(out_zarr, expected_input_format="gray", expected_total_samples=4)
    root = zarr.open_group(str(out_zarr), mode="r")
    raw = root["raw_video"]
    assert raw.attrs["pixel_contract_name"] == "legacy_gray_uint8_v1"
    assert raw.attrs["decode_backend"] == "legacy_import_gray"
    assert raw["images_ds"].attrs["pixel_contract_name"] == "legacy_gray_uint8_v1"
    assert raw["images_ds"].attrs["decode_backend"] == "legacy_import_gray"
    assert root.attrs["training_export"]["pixel_contract_name"] == "legacy_gray_uint8_v1"
    assert root.attrs["training_export"]["pixel_contract_names_by_format"] == {
        "gray": ["legacy_gray_uint8_v1"]
    }
    latest = root["refined_detect_runs"].attrs["latest"]
    instances = root[f"refined_detect_runs/{latest}/instances"]
    assert "crop_runs" not in root
    assert np.asarray(instances["frame_indices"][:], dtype=np.int64).tolist() == [0, 1, 2, 3]
    assert np.asarray(instances["source_kind_codes"][:], dtype=np.int8).tolist() == [1, 3, 1, 1]
    assert np.asarray(instances["manual_edit_flags"][:], dtype=bool).tolist() == [False, True, False, False]
    assert np.asarray(root["source_index/source_roi_idx"][:], dtype=np.int64).tolist() == [0, 1, 2, 3]
    assert np.asarray(root["source_index/source_refined_row_ids"][:], dtype=np.int64).tolist() == [
        100,
        101,
        102,
        103,
    ]
    assert np.asarray(root["source_index/source_detect_row_index"][:], dtype=np.int64).tolist() == [
        200,
        -1,
        202,
        203,
    ]
    assert np.asarray(instances["instance_key"][:], dtype=np.uint64).tolist() == [
        1000,
        1001,
        1002,
        1003,
    ]


def test_export_bridge_preserves_negative_and_multi_instance_frames(tmp_path: Path) -> None:
    source_zarr = tmp_path / "reviewed_source.zarr"
    out_zarr = tmp_path / "frame_supervised.zarr"
    manifest_path = tmp_path / "detect.manifest.json"
    _write_multi_instance_negative_source_zarr(source_zarr)
    manifest_path.write_text("{}", encoding="utf-8")
    manifest = TrainingManifest(
        created_at_utc="2026-08-06T00:00:00+00:00",
        task="detect",
        source_type="refined",
        input_format="gray",
        imgsz=[8, 8],
        datasets=[
            DatasetManifest(
                name="reviewed",
                zarr_path=str(source_zarr),
                dataset_id="reviewed",
                session_uuid="reviewed",
                crop_run=None,
                bbox_array_path=(
                    "refined_detect_runs/reviewed/instances/bbox_norm_coords"
                ),
                detection_source_type="refined",
                detection_source_path="refined_detect_runs/reviewed/instances",
                includes_interpolated=False,
                input_format="gray",
                images_ds_shape=[8, 8],
                total_bboxes=3,
                invalid_bboxes=0,
            )
        ],
        provenance_policy="strict",
        set_name="reviewed",
        set_version=1,
        set_id="reviewed_v001",
    )

    result = _export_merged(
        manifest=manifest,
        manifest_path=manifest_path,
        out_zarr=out_zarr,
        merged_dataset_id=None,
        overwrite=True,
        train_ratio=0.5,
        val_ratio=0.5,
        test_ratio=0.0,
        seed=42,
        include_rgb=False,
        copy_batch_size=2,
        invocation={},
    )

    assert result.total_samples == 4
    assert result.total_instances == 3
    summary = validate_merged_training_zarr(
        out_zarr,
        expected_input_format="gray",
        expected_total_samples=4,
    )
    assert summary["sample_axis"] == "frame"
    assert summary["total_instances"] == 3
    out = zarr.open_group(str(out_zarr), mode="r")
    run = out["refined_detect_runs"].attrs["latest"]
    instances = out[f"refined_detect_runs/{run}/instances"]
    assert np.asarray(instances["frame_counts"][:]).tolist() == [1, 0, 2, 0]
    assert np.asarray(instances["frame_offsets"][:]).tolist() == [0, 1, 1, 3, 3]
    assert np.asarray(instances["frame_indices"][:]).tolist() == [0, 2, 2]
    assert np.asarray(instances["instance_key"][:], dtype=np.uint64).tolist() == [
        2000,
        2001,
        2002,
    ]
    assert np.asarray(out["detection_training_supervision/label_state_codes"][:]).tolist() == [
        1,
        2,
        1,
        2,
    ]
    assert np.asarray(out["source_index/source_frame_idx"][:]).tolist() == [0, 1, 2, 3]
    assert np.asarray(out["source_index/source_roi_idx"][:]).tolist() == [0, 1, 2]

    config = ZarrDatasetConfig(
        datasets={
            "reviewed": {
                "zarr_path": str(out_zarr),
                "source_type": "refined",
                "input_format": "gray",
                "split": {"train": 0.5, "val": 0.5},
            }
        },
        task="detect",
        sampling_strategy="proportional",
    )
    train = create_zarr_dataset(config, mode="train")
    val = create_zarr_dataset(config, mode="val")
    datasets = (train, val)
    observed = {}
    for dataset in datasets:
        for _path, frame_index in dataset.indices:
            label = dataset._get_bbox_data(str(out_zarr), int(frame_index))
            observed[int(frame_index)] = (
                int(label["bboxes"].shape[0]),
                label["cls"].astype(int).tolist(),
            )
    assert observed == {0: (1, [0]), 1: (0, []), 2: (2, [1, 2]), 3: (0, [])}


def test_export_bridge_rejects_unresolved_review_frame(tmp_path: Path) -> None:
    source_zarr = tmp_path / "incomplete_review.zarr"
    _write_multi_instance_negative_source_zarr(source_zarr)
    root = zarr.open_group(str(source_zarr), mode="a", use_consolidated=False)
    clear_detect_frame_decision(
        root,
        source_refined_detect_run="reviewed",
        n_frames=4,
        frame_index=3,
    )
    manifest = TrainingManifest(
        created_at_utc="2026-08-06T00:00:00+00:00",
        task="detect",
        source_type="refined",
        input_format="gray",
        imgsz=[8, 8],
        datasets=[
            DatasetManifest(
                name="reviewed",
                zarr_path=str(source_zarr),
                dataset_id="reviewed",
                session_uuid="reviewed",
                crop_run=None,
                bbox_array_path=(
                    "refined_detect_runs/reviewed/instances/bbox_norm_coords"
                ),
                detection_source_type="refined",
                detection_source_path="refined_detect_runs/reviewed/instances",
                includes_interpolated=False,
                input_format="gray",
                images_ds_shape=[8, 8],
                total_bboxes=3,
                invalid_bboxes=0,
            )
        ],
        provenance_policy="strict",
        set_name="reviewed",
        set_version=1,
        set_id="reviewed_v001",
    )

    with pytest.raises(ValueError, match="unresolved frames"):
        _discover_merge_sources(manifest, need_gray=True, need_rgb=False)


def test_export_bridge_rejects_missing_stable_instance_keys(tmp_path: Path) -> None:
    source_zarr = tmp_path / "missing_keys.zarr"
    out_zarr = tmp_path / "merged.zarr"
    manifest_path = tmp_path / "detect.manifest.json"
    _write_multi_instance_negative_source_zarr(source_zarr)
    root = zarr.open_group(str(source_zarr), mode="a", use_consolidated=False)
    del root["refined_detect_runs/reviewed/instances/instance_key"]
    manifest_path.write_text("{}", encoding="utf-8")
    manifest = TrainingManifest(
        created_at_utc="2026-08-06T00:00:00+00:00",
        task="detect",
        source_type="refined",
        input_format="gray",
        imgsz=[8, 8],
        datasets=[
            DatasetManifest(
                name="reviewed",
                zarr_path=str(source_zarr),
                dataset_id="reviewed",
                session_uuid="reviewed",
                crop_run=None,
                bbox_array_path=(
                    "refined_detect_runs/reviewed/instances/bbox_norm_coords"
                ),
                detection_source_type="refined",
                detection_source_path="refined_detect_runs/reviewed/instances",
                includes_interpolated=False,
                input_format="gray",
                images_ds_shape=[8, 8],
                total_bboxes=3,
                invalid_bboxes=0,
            )
        ],
        provenance_policy="strict",
    )

    with pytest.raises(ValueError, match="requires source instance_key"):
        _export_merged(
            manifest=manifest,
            manifest_path=manifest_path,
            out_zarr=out_zarr,
            merged_dataset_id=None,
            overwrite=True,
            train_ratio=0.5,
            val_ratio=0.5,
            test_ratio=0.0,
            seed=42,
            include_rgb=False,
            copy_batch_size=2,
            invocation={},
        )


def test_export_merged_detect_training_zarr_rejects_interpolated_source_rows(tmp_path: Path) -> None:
    source_zarr = tmp_path / "source_detect.zarr"
    out_zarr = tmp_path / "merged_detect.zarr"
    manifest_path = tmp_path / "detect.manifest.json"
    _write_detect_source_zarr(source_zarr)
    root = zarr.open_group(str(source_zarr), mode="a")
    root["refined_detect_runs/refined_detect_001/instances/source_kind_codes"][:] = np.array(
        [1, 2, 1, 1],
        dtype=np.int8,
    )
    manifest_path.write_text("{}", encoding="utf-8")

    manifest = TrainingManifest(
        created_at_utc="2026-02-06T00:00:00+00:00",
        task="detect",
        source_type="refined",
        input_format="gray",
        imgsz=[640, 640],
        datasets=[
            DatasetManifest(
                name="source_dataset",
                zarr_path=str(source_zarr),
                dataset_id="source_dataset",
                session_uuid="source_dataset",
                crop_run=None,
                bbox_array_path="refined_detect_runs/refined_detect_001/instances/bbox_norm_coords",
                detection_source_type="refined",
                detection_source_path="refined_detect_runs/refined_detect_001/instances",
                includes_interpolated=True,
                input_format="gray",
                images_ds_shape=[8, 8],
                total_bboxes=4,
                invalid_bboxes=0,
            )
        ],
        provenance_policy="warn",
        set_name="cedar_shadow",
        set_version=1,
        set_id="detect_cedar_shadow_v001",
    )

    with np.testing.assert_raises_regex(ValueError, "refuses 1 legacy interpolated rows"):
        _export_merged(
            manifest=manifest,
            manifest_path=manifest_path,
            out_zarr=out_zarr,
            merged_dataset_id=None,
            overwrite=True,
            train_ratio=0.5,
            val_ratio=0.5,
            test_ratio=0.0,
            seed=42,
            include_rgb=False,
            copy_batch_size=2,
            invocation={},
        )


def test_discover_merge_sources_rejects_required_pixel_contract_mismatch(tmp_path: Path) -> None:
    source_zarr = tmp_path / "source_detect.zarr"
    _write_detect_source_zarr(source_zarr)
    manifest = TrainingManifest(
        created_at_utc="2026-02-06T00:00:00+00:00",
        task="detect",
        source_type="refined",
        input_format="gray",
        imgsz=[640, 640],
        datasets=[
            DatasetManifest(
                name="source_dataset",
                zarr_path=str(source_zarr),
                dataset_id="source_dataset",
                session_uuid="source_dataset",
                crop_run=None,
                bbox_array_path="refined_detect_runs/refined_detect_001/instances/bbox_norm_coords",
                detection_source_type="refined",
                detection_source_path="refined_detect_runs/refined_detect_001/instances",
                includes_interpolated=False,
                input_format="gray",
                images_ds_shape=[8, 8],
                total_bboxes=4,
                invalid_bboxes=0,
            )
        ],
        provenance_policy="warn",
    )

    with np.testing.assert_raises_regex(ValueError, "pixel contract mismatch"):
        _discover_merge_sources(
            manifest,
            need_gray=True,
            need_rgb=False,
            required_pixel_contract_name="orange_mono_pynvvc_luma_uint8_v1",
        )


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


def test_build_merged_manifest_payload_carries_identity_fields(tmp_path: Path) -> None:
    source_zarr = tmp_path / "source.zarr"
    manifest = TrainingManifest(
        created_at_utc="2026-02-06T00:00:00+00:00",
        task="detect",
        source_type="manual",
        input_format="gray",
        imgsz=[640, 640],
        datasets=[
            DatasetManifest(
                name="source_dataset",
                zarr_path=str(source_zarr),
                dataset_id="source_dataset",
                session_uuid="source_dataset",
                crop_run="crop_001",
                bbox_array_path="crop_runs/crop_001/bbox_norm_coords",
                detection_source_type="manual",
                detection_source_path="crop_runs/crop_001/detection_source",
                includes_interpolated=False,
                input_format="gray",
                images_ds_shape=[8, 8],
                total_bboxes=3,
                invalid_bboxes=0,
            )
        ],
        provenance_policy="warn",
        set_name="cedar_shadow",
        set_version=1,
        set_id="detect_cedar_shadow_v001",
    )

    merge_result = MergeResult(
        run_name="merged_export_20260206T000000Z",
        total_samples=3,
        total_real=3,
        total_interpolated=0,
        source_kind_counts={"1": 3},
        train_indices=np.array([0, 1], dtype=np.int64),
        val_indices=np.array([2], dtype=np.int64),
        test_indices=np.array([], dtype=np.int64),
        source_specs=[
            MergeSourceSpec(
                ordinal=1,
                dataset_name="source_dataset",
                dataset_id="source_dataset",
                source_zarr=source_zarr,
                bbox_path="crop_runs/crop_001/bbox_norm_coords",
                frame_indices_path="crop_runs/crop_001/frame_indices",
                detection_source_path="crop_runs/crop_001/detection_source",
                detection_source_type="manual",
                sample_count=3,
                has_gray=True,
                has_rgb=False,
                gray_shape=(8, 8),
                rgb_shape=None,
                gray_dtype=np.dtype(np.uint8),
                rgb_dtype=None,
                gray_chunks=(3, 8, 8),
                rgb_chunks=None,
                fps=60.0,
            )
        ],
        downsample_formats=["gray"],
        training_input_format="gray",
    )

    payload = _build_merged_manifest_payload(
        manifest=manifest,
        manifest_payload={
            "set_id": "detect_cedar_shadow_v001",
            "set_name": "cedar_shadow",
            "datasets": [
                {
                    "name": "source_dataset",
                    "zarr_path": str(source_zarr),
                    "dataset_id": "source_dataset",
                    "provenance": {
                        "arena": {"dish_design": "cedar"},
                        "rig_info": {"canvas_name": "shadow", "rig_id": "omnifin0"},
                    },
                }
            ],
        },
        merged_zarr=tmp_path / "merged.zarr",
        merged_dataset_id="detect_cedar_shadow_v001_merged",
        merged_dataset_name="cedar_shadow_merged",
        run_name="merged_export_20260206T000000Z",
        out_manifest=tmp_path / "merged.manifest.json",
        out_config=tmp_path / "merged.yaml",
        merge_result=merge_result,
        include_rgb=False,
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,
        seed=123,
    )

    assert payload["dish_design"] == "cedar"
    assert payload["canvas_name"] == "shadow"
    assert payload["rig_name"] == "omnifin0"
    assert payload["datasets"][0]["dish_design"] == "cedar"
    assert payload["datasets"][0]["canvas_name"] == "shadow"
    assert payload["datasets"][0]["rig_id"] == "omnifin0"
    assert payload["datasets"][0]["crop_run"] is None
    assert payload["datasets"][0]["detection_source_type"] == "refined"
    assert (
        payload["datasets"][0]["bbox_array_path"]
        == "refined_detect_runs/merged_export_20260206T000000Z/instances/bbox_norm_coords"
    )
    assert payload["datasets"][0]["detection_source_present"] is False
    assert payload["datasets"][0]["includes_interpolated"] is False
    assert payload["datasets"][0]["source_kind_counts"] == {"1": 3}
    assert (
        payload["merged_export"]["canonical_label_path"]
        == "refined_detect_runs/merged_export_20260206T000000Z/instances"
    )
    assert payload["merged_export"]["counts"]["source_count"] == 1
    assert payload["merged_export"]["source_datasets"][0]["dish_design"] == "cedar"
    assert payload["merged_export"]["source_datasets"][0]["canvas_name"] == "shadow"
    assert payload["merged_export"]["source_datasets"][0]["rig_id"] == "omnifin0"
