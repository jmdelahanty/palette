from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.flat_roi_cache import (
    FLAT_ROI_CACHE_LAYOUT,
    FLAT_ROI_CACHE_SCHEMA,
)
from fisheye.shared.roi_pixel_contract import (
    ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
    orange_mono_pynvvc_luma_pixel_contract,
)
from fisheye.utils import regenerate_training_crops_pynvvc as mod
from fisheye.shared.zarr.training_crop_materialization import (
    SAMPLED_TRAINING_IMAGES_FULL_MATERIALIZATION_PROVIDER,
    TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE,
    TrainingCropMaterializationError,
    bind_training_crop_materialization,
    build_training_crop_materialization_binding,
)
from fisheye.shared.zarr.training_crop_materialization_publication import (
    create_sampled_images_full_training_crop_artifact,
    create_training_crop_artifact,
    enrich_sampled_training_dataset,
    publish_training_crop_materialization,
)
from fisheye.shared.zarr.detect_frame_decisions import (
    FRAME_REVIEW_CONTRACT_ATTR,
    FRAME_REVIEW_CONTRACT_ID,
    set_detect_frame_negative,
)
from fisheye.shared.zarr.crop_schema import derive_crop_placement_geometry
from fisheye.shared.zarr.detection_schema import (
    derive_canonical_detection_geometry,
)
from fisheye.shared.zarr.training_dataset_composition import (
    TRAINING_DATASET_COMPOSITION_ATTRIBUTE,
    TrainingDatasetCompositionError,
    bind_training_dataset_composition,
)
from fisheye.training.detection_frame_supervision import (
    build_detection_frame_supervision_plan,
)
from fisheye.utils.regenerate_training_crops_pynvvc import (
    regenerate_training_crops_pynvvc,
)


class _FakePynvvcReader:
    def __init__(self, frames: list[np.ndarray]) -> None:
        import torch

        self.source_height = int(frames[0].shape[0])
        self.source_width = int(frames[0].shape[1])
        self._frames = [
            torch.from_numpy(
                np.vstack(
                    [
                        frame,
                        np.zeros(
                            (max(1, frame.shape[0] // 2), frame.shape[1]),
                            dtype=np.uint8,
                        ),
                    ]
                )
            )
            for frame in frames
        ]
        self._offset = 0

    def decode_next(self, count: int):
        result = self._frames[self._offset : self._offset + int(count)]
        self._offset += len(result)
        return result

    def iter_frames(self):
        while self._offset < len(self._frames):
            frame = self._frames[self._offset]
            self._offset += 1
            yield frame

    def close(self) -> None:
        pass


class _FakeStreamMetadata:
    def __init__(self, *, height: int, width: int, num_frames: int) -> None:
        self.height = height
        self.width = width
        self.num_frames = num_frames


class _FakeIndexedPynvvcDecoder:
    def __init__(self, frames: list[np.ndarray]) -> None:
        import torch

        self._height = int(frames[0].shape[0])
        self._width = int(frames[0].shape[1])
        self.requested: list[int] = []
        self._frames = [
            torch.from_numpy(
                np.vstack(
                    [
                        frame,
                        np.zeros(
                            (max(1, frame.shape[0] // 2), frame.shape[1]),
                            dtype=np.uint8,
                        ),
                    ]
                )
            )
            for frame in frames
        ]

    def get_stream_metadata(self) -> _FakeStreamMetadata:
        return _FakeStreamMetadata(
            height=self._height, width=self._width, num_frames=len(self._frames)
        )

    def get_batch_frames_by_index(self, indices: list[int]):
        self.requested.extend(int(index) for index in indices)
        return [self._frames[int(index)] for index in indices]


class _FakeRootForFrameIndex:
    def __init__(self, attrs: dict[str, object]) -> None:
        self.attrs = attrs

    def get(self, _name: str):
        return None


def _make_training_archive(tmp_path: Path) -> tuple[Path, list[np.ndarray]]:
    zarr_path = tmp_path / "recording_training.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["zarr_purpose"] = "training"
    root.attrs["source_video_path"] = str(tmp_path / "source.mp4")
    root.attrs["width"] = 5
    root.attrs["height"] = 4
    root.attrs["source_video_total_frames"] = 5

    raw = root.create_group("raw_video")
    raw.create_array(
        "original_frame_indices",
        data=np.array([0, 2, 4], dtype=np.int32),
        overwrite=True,
    )

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    crop_parent.attrs["latest_materialized"] = "crop_001"
    crop_parent.attrs["latest_any"] = "crop_001"
    crop = crop_parent.create_group("crop_001")
    crop.attrs["crop_storage_mode"] = "materialized"
    crop.attrs["video_source_type"] = "zarr"
    crop.attrs["roi_size"] = [2, 2]
    crop.attrs["source_video_path"] = str(tmp_path / "source.mp4")

    frames = [
        np.arange(4 * 5, dtype=np.uint8).reshape(4, 5) + np.uint8(frame_idx * 20)
        for frame_idx in range(5)
    ]
    frame_indices = np.array([0, 1, 2], dtype=np.int64)
    roi_coordinates = np.array([[0, 0], [1, 1], [3, 2]], dtype=np.int32)
    stale_roi_images = np.zeros((3, 2, 2), dtype=np.uint8)

    crop.create_array("frame_indices", data=frame_indices, overwrite=True)
    crop.create_array("roi_coordinates_full", data=roi_coordinates, overwrite=True)
    crop.create_array(
        "bbox_norm_coords", data=np.zeros((3, 4), dtype=np.float32), overwrite=True
    )
    crop.create_array("roi_images", data=stale_roi_images, overwrite=True)
    return zarr_path, frames


def _make_external_cache_materialization(
    tmp_path: Path,
) -> tuple[Path, Path, Path, np.ndarray]:
    source_path = tmp_path / "source_analysis.zarr"
    source = zarr.open_group(str(source_path), mode="w")
    source_crop = source.require_group("crop_runs").create_group("crop_v2")
    source_crop.attrs.update(
        {
            "crop_storage_mode": "geometry_only",
            "coordinate_contract": "canonical_v2",
            "roi_size": [2, 3],
            "width": 8,
            "height": 6,
            "crop_signature": "crop-signature-v2",
            "crop_revision": "crop-revision-v2",
            "run_manifest": {
                "schema_id": "palette.stage.crop_geometry.run_manifest",
                "schema_version": 1,
                "payload_digest": "a" * 64,
            },
        }
    )
    source_crop.create_array(
        "frame_indices",
        data=np.asarray([1, 1, 4], dtype=np.int64),
        overwrite=True,
    )
    source_crop.create_array(
        "source_acquisition_frame_index",
        data=np.asarray([1, 1, 4], dtype=np.int64),
        overwrite=True,
    )
    source_crop.create_array(
        "roi_coordinates_full",
        data=np.asarray([[0, 0], [2, 1], [4, 3]], dtype=np.int32),
        overwrite=True,
    )
    source_crop.create_array(
        "instance_key",
        data=np.asarray([101, 102, 103], dtype=np.uint64),
        overwrite=True,
    )
    source_crop.create_array(
        "source_refined_row_ids",
        data=np.asarray([10, 11, 12], dtype=np.int64),
        overwrite=True,
    )
    source_crop.create_array(
        "frame_row_offsets",
        data=np.asarray([0, 0, 2, 2, 2, 3], dtype=np.int64),
        overwrite=True,
    )
    source_crop.create_array(
        "bbox_norm_coords",
        data=np.asarray(
            [
                [0.25, 0.25, 0.25, 1 / 3],
                [0.5, 0.5, 0.25, 1 / 3],
                [0.75, 0.75, 0.25, 1 / 3],
            ],
            dtype=np.float32,
        ),
        overwrite=True,
    )
    source_crop.create_array(
        "bbox_img_xyxy",
        data=np.asarray(
            [[1, 0.5, 3, 2.5], [3, 2, 5, 4], [5, 3.5, 7, 5.5]], dtype=np.float32
        ),
        overwrite=True,
    )
    source_crop.create_array(
        "centers_img_xy",
        data=np.asarray([[2, 1.5], [4, 3], [6, 4.5]], dtype=np.float32),
        overwrite=True,
    )
    source_crop.create_array(
        "roi_sizes_full",
        data=np.asarray([[3, 2], [3, 2], [3, 2]], dtype=np.int32),
        overwrite=True,
    )
    source_crop.create_array(
        "source_crop_xywh",
        data=np.asarray([[0, 0, 3, 2], [2, 1, 3, 2], [4, 3, 3, 2]], dtype=np.float32),
        overwrite=True,
    )
    source_crop.create_array(
        "bbox_roi_xyxy",
        data=np.asarray(
            [[1, 0.5, 3, 2.5], [1, 1, 3, 3], [1, 0.5, 3, 2.5]], dtype=np.float32
        ),
        overwrite=True,
    )
    source_crop.create_array(
        "source_row_signature",
        data=np.arange(3 * 32, dtype=np.uint8).reshape(3, 32),
        overwrite=True,
    )

    target_path = tmp_path / "target_training.zarr"
    target = zarr.open_group(str(target_path), mode="w")
    target.attrs["zarr_purpose"] = "training"
    target.require_group("crop_runs")
    raw = target.create_group("raw_video")
    raw.create_array(
        "images_full",
        data=np.zeros((2, 6, 8), dtype=np.uint8),
        overwrite=True,
    )
    raw.create_array(
        "images_ds",
        data=np.zeros((2, 3, 4), dtype=np.uint8),
        overwrite=True,
    )
    raw.create_array(
        "original_frame_indices",
        data=np.asarray([1, 4], dtype=np.int64),
        overwrite=True,
    )
    boxes = np.asarray(
        [
            [0.25, 0.25, 0.25, 1 / 3],
            [0.5, 0.5, 0.25, 1 / 3],
            [0.75, 0.75, 0.25, 1 / 3],
        ],
        dtype=np.float32,
    )
    for parent_name, run_name in (
        ("detect_runs", "detect_review_seed"),
        ("refined_detect_runs", "refined_detect_review_seed"),
    ):
        parent = target.create_group(parent_name)
        parent.attrs["latest_complete"] = run_name
        run = parent.create_group(run_name)
        run.attrs["status"] = "complete"
        if parent_name == "refined_detect_runs":
            run.attrs["detect_review_status"] = {
                "state": "approved",
                "intended_use": "training",
                "resolved_group": "refined",
            }
        instances = run.create_group("instances")
        instances.create_array(
            "frame_indices",
            data=np.asarray([0, 0, 1], dtype=np.int32),
            overwrite=True,
        )
        instances.create_array(
            "bbox_norm_coords",
            data=boxes,
            overwrite=True,
        )
        instances.create_array(
            "instance_key",
            data=np.asarray([101, 102, 103], dtype=np.uint64),
            overwrite=True,
        )

    pixels = np.arange(3 * 2 * 3, dtype=np.uint8).reshape(3, 2, 3)
    payload_path = tmp_path / "crop_v2.bin"
    payload_path.write_bytes(pixels.tobytes(order="C"))
    payload_sha256 = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    contract = orange_mono_pynvvc_luma_pixel_contract()
    manifest_path = tmp_path / "crop_v2.json"
    manifest = {
        "schema": FLAT_ROI_CACHE_SCHEMA,
        "layout": FLAT_ROI_CACHE_LAYOUT,
        "cache_complete": True,
        "source": {
            "archive_path": str(source_path.resolve()),
            "crop_run_name": "crop_v2",
        },
        "array": {
            "bin_path": payload_path.name,
            "dtype": "uint8",
            "shape": [3, 2, 3],
            "order": "C",
            "total_bytes": int(pixels.nbytes),
            "sha256": payload_sha256,
        },
        "builder": {
            "pixel_contract": contract,
            "pixel_contract_name": contract["name"],
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source_path, target_path, manifest_path, pixels


def test_regenerate_training_crops_pynvvc_writes_new_luma_crop_run(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path, frames = _make_training_archive(tmp_path)
    monkeypatch.setattr(
        mod, "_open_pynvvc_luma_reader", lambda _video_path: _FakePynvvcReader(frames)
    )

    report = regenerate_training_crops_pynvvc(
        zarr_path=zarr_path,
        source_crop_run="crop_001",
        target_crop_run="crop_001_pynvvc_luma",
        decode_chunk_frames=2,
    )

    assert report["status"] == "ok"
    assert report["source_frame_index_mapping"]["mode"] == "original_frame_indices"
    root = zarr.open_group(str(zarr_path), mode="r")
    crop_parent = root["crop_runs"]
    assert crop_parent.attrs["latest"] == "crop_001"
    target = crop_parent["crop_001_pynvvc_luma"]
    assert target.attrs["status"] == "completed"
    assert target.attrs["source_crop_run"] == "crop_001"
    assert (
        target.attrs["training_materialization_schema"]
        == mod.TRAINING_CROP_MATERIALIZATION_SCHEMA
    )
    assert (
        target.attrs["training_materialization_provider"]
        == mod.SOURCE_VIDEO_MATERIALIZATION_PROVIDER
    )
    assert set(target.attrs["training_materialization_provider_contract"]) == set(
        mod.TRAINING_CROP_MATERIALIZATION_PROVIDERS
    )
    assert (
        target.attrs["roi_pixel_contract"]["name"]
        == ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME
    )
    assert (
        target.attrs["roi_pixel_contract_name"] == ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME
    )
    assert np.array_equal(
        target["source_frame_indices"][:], np.array([0, 2, 4], dtype=np.int64)
    )
    expected = np.stack(
        [
            frames[0][0:2, 0:2],
            frames[2][1:3, 1:3],
            frames[4][2:4, 3:5],
        ],
        axis=0,
    )
    assert np.array_equal(target["roi_images"][:], expected)
    assert np.array_equal(
        target["frame_indices"][:], np.array([0, 1, 2], dtype=np.int64)
    )
    assert "bbox_norm_coords" in target
    direct_target = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)[
        "crop_runs/crop_001_pynvvc_luma"
    ]
    consolidated_target = zarr.open_group(
        str(zarr_path), mode="r", use_consolidated=True
    )["crop_runs/crop_001_pynvvc_luma"]
    assert dict(consolidated_target.attrs) == dict(direct_target.attrs)


def test_external_source_video_provider_remains_first_class(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_path, frames = _make_training_archive(tmp_path)
    target_path = tmp_path / "external_target_training.zarr"
    target_root = zarr.open_group(str(target_path), mode="w")
    target_root.attrs["zarr_purpose"] = "training"
    target_root.require_group("crop_runs")
    monkeypatch.setattr(
        mod,
        "_open_pynvvc_luma_reader",
        lambda _video_path: _FakePynvvcReader(frames),
    )

    report = regenerate_training_crops_pynvvc(
        zarr_path=target_path,
        source_zarr_path=source_path,
        source_crop_run="crop_001",
        target_crop_run="crop_external_video",
        video_path=tmp_path / "source.mp4",
    )

    assert report["status"] == "ok"
    assert (
        report["materialization_provider"] == mod.SOURCE_VIDEO_MATERIALIZATION_PROVIDER
    )
    run = zarr.open_group(str(target_path), mode="r", use_consolidated=False)[
        "crop_runs/crop_external_video"
    ]
    assert run.attrs["training_crop_materialization_binding_status"] == (
        "legacy_source_missing_crop_v2_identity"
    )
    assert "training_crop_materialization_binding" not in run.attrs


def test_regenerate_training_crops_pynvvc_frame_domain_mapping_matches_legacy_full_array(
    tmp_path: Path,
) -> None:
    zarr_path, _frames = _make_training_archive(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="r")
    crop_frame_indices = np.asarray(
        root["crop_runs/crop_001/frame_indices"][:], dtype=np.int64
    )

    mapped, metadata = mod._map_source_frame_indices(
        root=root,
        crop_frame_indices=crop_frame_indices,
        mode="original_frame_indices",
    )

    original_frame_indices = np.asarray(
        root["raw_video/original_frame_indices"][:], dtype=np.int64
    )
    legacy_mapped = original_frame_indices[crop_frame_indices]
    np.testing.assert_array_equal(mapped, legacy_mapped)
    assert metadata == {
        "mode": "original_frame_indices",
        "original_frame_indices_available": True,
        "original_frame_indices_length": 3,
    }


def test_load_clipped_source_frame_mapping_reads_required_parquet_columns(
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    archive_path = tmp_path / "recording_clipped_training.zarr"
    archive_path.mkdir()
    video_a = tmp_path / "clips" / "clip_000000" / "Cam2010093.mp4"
    video_b = tmp_path / "clips" / "clip_000001" / "Cam2010093.mp4"
    index_path = archive_path / "source_frame_index.parquet"
    table = pa.table(
        {
            "sample_index": pa.array([0, 1, 2], type=pa.int64()),
            "video_path": pa.array([str(video_a), str(video_a), str(video_b)]),
            "clip_local_frame_index": pa.array([11, 12, 3], type=pa.int64()),
        }
    )
    pq.write_table(table, index_path)

    mapping = mod._load_clipped_source_frame_mapping(
        root=_FakeRootForFrameIndex(
            {"source_frame_index_path": "source_frame_index.parquet"}
        ),
        archive_path=archive_path,
        crop_frame_indices=np.array([2, 0], dtype=np.int64),
        mode="source_frame_index_parquet",
    )

    assert mapping is not None
    assert mapping["mode"] == "source_frame_index_parquet"
    assert mapping["source_frame_index_path"] == str(index_path)
    assert np.array_equal(
        mapping["source_frame_indices"], np.array([2, 0], dtype=np.int64)
    )
    assert np.array_equal(
        mapping["source_clip_local_frame_indices"], np.array([3, 11], dtype=np.int64)
    )
    assert np.array_equal(
        mapping["source_clip_indices"], np.array([-1, -1], dtype=np.int64)
    )
    assert mapping["video_frame_to_rows"][video_b][3] == [0]
    assert mapping["video_frame_to_rows"][video_a][11] == [1]


def test_regenerate_training_crops_pynvvc_dry_run_does_not_write(
    tmp_path: Path,
) -> None:
    zarr_path, _frames = _make_training_archive(tmp_path)

    report = regenerate_training_crops_pynvvc(
        zarr_path=zarr_path,
        source_crop_run="crop_001",
        target_crop_run="crop_preview",
        dry_run=True,
    )

    assert report["status"] == "dry_run"
    root = zarr.open_group(str(zarr_path), mode="r")
    assert "crop_preview" not in root["crop_runs"]


def test_regenerate_training_crops_pynvvc_auto_uses_indexed_for_sparse_frames(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path, _frames = _make_training_archive(tmp_path)
    frames = [
        np.arange(4 * 5, dtype=np.uint8).reshape(4, 5) + np.uint8(frame_idx % 200)
        for frame_idx in range(201)
    ]
    root = zarr.open_group(str(zarr_path), mode="a")
    root.attrs["source_video_total_frames"] = 201
    root["raw_video/original_frame_indices"][:] = np.array(
        [0, 100, 200], dtype=np.int32
    )
    indexed_decoder = _FakeIndexedPynvvcDecoder(frames)
    monkeypatch.setattr(
        mod,
        "_open_pynvvc_luma_indexed_decoder",
        lambda _video_path: indexed_decoder,
    )
    monkeypatch.setattr(
        mod,
        "_open_pynvvc_luma_reader",
        lambda _video_path: (_ for _ in ()).throw(
            AssertionError("sequential reader should not be used")
        ),
    )

    report = regenerate_training_crops_pynvvc(
        zarr_path=zarr_path,
        source_crop_run="crop_001",
        target_crop_run="crop_001_pynvvc_luma_indexed",
        decode_mode="auto",
        decode_chunk_frames=2,
    )

    assert report["status"] == "ok"
    assert report["decode_mode_effective"] == "indexed"
    assert indexed_decoder.requested == [0, 100, 200]
    target = zarr.open_group(str(zarr_path), mode="r")[
        "crop_runs/crop_001_pynvvc_luma_indexed"
    ]
    assert target.attrs["decode_mode_effective"] == "indexed"
    expected = np.stack(
        [
            frames[0][0:2, 0:2],
            frames[100][1:3, 1:3],
            frames[200][2:4, 3:5],
        ],
        axis=0,
    )
    assert np.array_equal(target["roi_images"][:], expected)


def test_materialize_training_crops_from_external_verified_flat_cache(
    tmp_path: Path,
) -> None:
    source_path, target_path, manifest_path, pixels = (
        _make_external_cache_materialization(tmp_path)
    )

    report = regenerate_training_crops_pynvvc(
        zarr_path=target_path,
        source_zarr_path=source_path,
        source_crop_run="crop_v2",
        target_crop_run="crop_v2_training_materialized",
        roi_cache_manifest=manifest_path,
        cache_copy_batch_rows=2,
    )

    assert report["status"] == "ok"
    assert (
        report["materialization_provider"]
        == mod.VERIFIED_CACHE_MATERIALIZATION_PROVIDER
    )
    assert report["source_zarr_path"] == str(source_path.resolve())
    target = zarr.open_group(str(target_path), mode="r", use_consolidated=False)[
        "crop_runs/crop_v2_training_materialized"
    ]
    np.testing.assert_array_equal(target["roi_images"][:], pixels)
    np.testing.assert_array_equal(
        target["instance_key"][:], np.asarray([101, 102, 103], dtype=np.uint64)
    )
    np.testing.assert_array_equal(
        target["source_frame_indices"][:],
        np.asarray([1, 1, 4], dtype=np.int64),
    )
    assert target.attrs["crop_storage_mode"] == "materialized"
    assert (
        target.attrs["training_materialization_schema"]
        == mod.TRAINING_CROP_MATERIALIZATION_SCHEMA
    )
    assert (
        target.attrs["training_materialization_provider"]
        == mod.VERIFIED_CACHE_MATERIALIZATION_PROVIDER
    )
    assert set(target.attrs["training_materialization_provider_contract"]) == set(
        mod.TRAINING_CROP_MATERIALIZATION_PROVIDERS
    )
    assert target.attrs["source_roi_cache_verified"] is True
    assert (
        target.attrs["source_roi_cache_independence"]
        == "roi_images_copied_into_training_zarr_no_runtime_cache_dependency"
    )
    assert target.attrs["coordinate_contract"] == (
        "training_materialized_from_crop_v2_v1"
    )
    assert target.attrs["training_crop_materialization_binding_status"] == "strict_v1"
    assert (
        target.attrs[TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE]["payload"][
            "dimensions"
        ]["row_count"]
        == 3
    )
    bound = bind_training_crop_materialization(
        target_path,
        run_id="crop_v2_training_materialized",
    )
    assert bound.row_count == 3
    assert bound.roi_shape == (2, 3)

    # The training Zarr remains usable after the ephemeral cache is removed.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    (manifest_path.parent / manifest["array"]["bin_path"]).unlink()
    np.testing.assert_array_equal(target["roi_images"][:], pixels)


def test_sampled_images_full_binding_requires_no_external_crop_row_ids(
    tmp_path: Path,
) -> None:
    target_path = tmp_path / "sampled_training.zarr"
    root = zarr.open_group(str(target_path), mode="a", use_consolidated=False)
    root.attrs["zarr_purpose"] = "training"
    run = root.require_group("crop_runs").create_group("sampled_images_full")
    frames = np.asarray([0, 2, 2], dtype=np.int64)
    acquisition_frames = np.asarray([10, 20, 20], dtype=np.int64)
    boxes = np.asarray(
        [
            [0.25, 0.25, 0.25, 1 / 3],
            [0.5, 0.5, 0.25, 1 / 3],
            [0.75, 0.75, 0.25, 1 / 3],
        ],
        dtype=np.float32,
    )
    bbox_img, centers = derive_canonical_detection_geometry(
        boxes,
        source_width=8,
        source_height=6,
    )
    sizes = np.repeat(np.asarray([[3, 2]], dtype=np.int32), 3, axis=0)
    coordinates, source_crop, bbox_roi = derive_crop_placement_geometry(
        centers,
        bbox_img,
        sizes,
    )
    arrays = {
        "instance_key": np.asarray([101, 102, 103], dtype=np.uint64),
        "source_refined_row_ids": np.asarray([7, 8, 9], dtype=np.int64),
        "frame_indices": frames,
        "source_acquisition_frame_index": acquisition_frames,
        "frame_row_offsets": np.asarray([0, 1, 1, 3], dtype=np.int64),
        "bbox_norm_coords": boxes,
        "bbox_img_xyxy": bbox_img,
        "centers_img_xy": centers,
        "roi_coordinates_full": coordinates,
        "roi_sizes_full": sizes,
        "source_crop_xywh": source_crop,
        "bbox_roi_xyxy": bbox_roi,
        "source_row_signature": np.arange(3 * 32, dtype=np.uint8).reshape(3, 32),
        "source_frame_indices": acquisition_frames,
    }
    for name, values in arrays.items():
        run.create_array(name, data=values, overwrite=True)
    pixels = np.arange(3 * 2 * 3, dtype=np.uint8).reshape(3, 2, 3)
    run.create_array("roi_images", data=pixels, overwrite=True)
    run.attrs.update(
        {
            "status": "completed",
            "stage_selector_eligible": False,
            "training_materialization_schema": mod.TRAINING_CROP_MATERIALIZATION_SCHEMA,
            "training_materialization_provider": (
                SAMPLED_TRAINING_IMAGES_FULL_MATERIALIZATION_PROVIDER
            ),
            "source_crop_archive_path": str(target_path),
            "source_crop_run": "reviewed",
            "source_crop_path": "refined_detect_runs/reviewed/instances",
            "source_crop_manifest_binding": {
                "authority_kind": "sampled_detection_review",
                "source_refined_detect_run": "reviewed",
                "source_frame_decision_digest": "a" * 64,
            },
            "height": 6,
            "width": 8,
            "source_images_path": "raw_video/images_full",
            "source_images_dtype": "uint8",
            "source_images_shape": [3, 6, 8],
            "source_refined_detect_run": "reviewed",
            "source_frame_decision_path": ("detect_frame_decision_runs/reviewed"),
            "source_frame_decision_digest": "a" * 64,
            "padding_mode": "zero_outside_source_frame",
            "pixel_verification": "all_rows_byte_equal_to_source_window_v1",
        }
    )
    binding = build_training_crop_materialization_binding(run)
    run.attrs[TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE] = binding
    zarr.consolidate_metadata(str(target_path))

    bound = bind_training_crop_materialization(
        target_path,
        run_id="sampled_images_full",
    )
    assert bound.row_count == 3
    assert "source_crop_row_ids" not in bound.run_group
    assert bound.binding["payload"]["provider"] == (
        SAMPLED_TRAINING_IMAGES_FULL_MATERIALIZATION_PROVIDER
    )

    mutable = zarr.open_group(str(target_path), mode="a", use_consolidated=False)[
        "crop_runs/sampled_images_full"
    ]
    mutable["source_frame_indices"][0] = np.int64(999)
    with pytest.raises(
        TrainingCropMaterializationError,
        match="source_frame_indices must equal source acquisition-frame identity",
    ):
        build_training_crop_materialization_binding(mutable)


def _make_reviewed_sampled_images_full_base(tmp_path: Path) -> Path:
    path = tmp_path / "reviewed_sampled_base.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "zarr_purpose": "training",
            "training_artifact_status": "review_complete",
            "stage_selector_eligible": False,
        }
    )
    raw = root.create_group("raw_video")
    images = np.stack(
        [
            np.arange(6 * 8, dtype=np.uint8).reshape(6, 8),
            np.full((6, 8), 100, dtype=np.uint8),
            np.arange(6 * 8, dtype=np.uint8).reshape(6, 8) + np.uint8(50),
        ]
    )
    raw.create_array("images_full", data=images)
    raw.create_array("images_ds", data=images)
    raw.create_array(
        "original_frame_indices",
        data=np.asarray([10, 20, 30], dtype=np.int32),
    )
    boxes = np.asarray(
        [
            [0.125, 1 / 6, 0.125, 1 / 6],
            [0.5, 0.5, 0.25, 1 / 3],
            [0.75, 0.75, 0.25, 1 / 3],
        ],
        dtype=np.float32,
    )
    frames = np.asarray([0, 2, 2], dtype=np.int32)
    keys = np.asarray([101, 102, 103], dtype=np.uint64)
    detected = root.require_group("detect_runs").create_group("detected")
    detected.attrs["status"] = "completed"
    detected_instances = detected.create_group("instances")
    detected_instances.create_array("frame_indices", data=frames)
    detected_instances.create_array("bbox_norm_coords", data=boxes)
    detected_instances.create_array("instance_key", data=keys)
    refined = root.require_group("refined_detect_runs").create_group("reviewed")
    refined.attrs.update(
        {
            "status": "completed",
            "stage_selector_eligible": False,
            "source_detect_run": "detected",
            FRAME_REVIEW_CONTRACT_ATTR: FRAME_REVIEW_CONTRACT_ID,
        }
    )
    instances = refined.create_group("instances")
    instances.create_array("frame_indices", data=frames)
    instances.create_array("bbox_norm_coords", data=np.asarray(boxes, dtype=np.float64))
    instances.create_array("instance_key", data=keys)
    instances.create_array(
        "refined_row_ids",
        data=np.asarray([7, 8, 9], dtype=np.int64),
    )
    set_detect_frame_negative(
        root,
        source_refined_detect_run="reviewed",
        n_frames=3,
        frame_index=1,
    )
    supervision = build_detection_frame_supervision_plan(
        root,
        bbox_path="refined_detect_runs/reviewed/instances/bbox_norm_coords",
        frame_indices_path="refined_detect_runs/reviewed/instances/frame_indices",
        n_frames=3,
    )
    refined.attrs["detect_review_status"] = {
        "state": "approved",
        "method": "manual",
        "intended_use": "training",
        "reviewer": "reviewer",
        "authority_scope": "selector_ineligible_training_candidate",
        "selector_ineligible_candidate_receipt": {
            "schema_id": "palette.selector_ineligible_detection_review_receipt",
            "schema_version": 1,
            "status": "complete",
            "authority_scope": "selector_ineligible_training_candidate",
            "source_refined_detect_run": "reviewed",
            "frame_decision_path": "detect_frame_decision_runs/reviewed",
            "frame_decision_digest": supervision.source_decision_digest,
            "frame_decision_would_materialize": False,
            "frame_count": 3,
            "instance_count": 3,
            "positive_frame_count": 2,
            "negative_frame_count": 1,
            "parent_selectors_updated": False,
            "stage_selector_eligible": False,
            "metadata_mode": "direct_mutable",
        },
        "authoritative_approval": {
            "status": "deferred_selector_ineligible",
            "reason_code": "CANDIDATE_ONLY",
            "run": "reviewed",
        },
    }
    zarr.consolidate_metadata(str(path))
    return path


def test_sampled_images_full_artifact_publishes_reviewed_rows_atomically(
    tmp_path: Path,
) -> None:
    base = _make_reviewed_sampled_images_full_base(tmp_path)
    destination = tmp_path / "reviewed_sampled_crops.zarr"
    scratch = tmp_path / "node-local" / "job"
    scratch.mkdir(parents=True)

    result = create_sampled_images_full_training_crop_artifact(
        destination=destination,
        base_training_zarr=base,
        run_id="reviewed_4px",
        refined_run_id="reviewed",
        scratch_root=scratch,
        roi_size_wh=(4, 4),
    )

    assert result["status"] == "complete"
    assert result["row_count"] == 3
    assert result["materialization"]["positive_frame_count"] == 2
    assert result["materialization"]["negative_frame_count"] == 1
    assert result["materialization"]["pixel_rows_verified"] == 3
    assert result["detect_run"] == "detected"
    assert len(result["composition_digest"]) == 64
    bound = bind_training_dataset_composition(
        destination,
        crop_run_id="reviewed_4px",
        detect_run_id="detected",
        refined_run_id="reviewed",
    )
    assert bound.binding["payload_digest"] == result["composition_digest"]
    root = zarr.open_group(str(destination), mode="r", use_consolidated=True)
    run = root["crop_runs/reviewed_4px"]
    assert run.attrs["training_materialization_provider"] == (
        SAMPLED_TRAINING_IMAGES_FULL_MATERIALIZATION_PROVIDER
    )
    assert run.attrs["summary_statistics"]["padded_rows"] == 1
    np.testing.assert_array_equal(run["frame_indices"][:], [0, 2, 2])
    np.testing.assert_array_equal(run["frame_row_offsets"][:], [0, 1, 1, 3])
    np.testing.assert_array_equal(run["source_frame_indices"][:], [10, 30, 30])
    np.testing.assert_array_equal(run["instance_key"][:], [101, 102, 103])
    assert run["bbox_norm_coords"].dtype == np.dtype(np.float32)
    expected_padded = np.zeros((4, 4), dtype=np.uint8)
    expected_padded[1:, 1:] = np.asarray(root["raw_video/images_full"][0, :3, :3])
    np.testing.assert_array_equal(run["roi_images"][0], expected_padded)
    assert root.attrs["stage_selector_eligible"] is False
    assert root.attrs["registry_activation"] == "deferred"
    detection_binding = root.attrs[TRAINING_DATASET_COMPOSITION_ATTRIBUTE]["payload"][
        "review_surfaces"
    ]["detection"]
    assert detection_binding["review_authority_scope"] == (
        "selector_ineligible_training_candidate"
    )
    assert len(detection_binding["review_status_digest"]) == 64
    assert "crop_runs" not in zarr.open_group(
        str(base), mode="r", use_consolidated=True
    )
    assert not list(destination.parent.glob(f".{destination.name}.publish_tmp.*"))


def test_sampled_images_full_artifact_fails_closed_on_unreviewed_frame(
    tmp_path: Path,
) -> None:
    base = _make_reviewed_sampled_images_full_base(tmp_path)
    direct = zarr.open_group(str(base), mode="a", use_consolidated=False)
    decisions = direct["detect_frame_decision_runs/reviewed"]
    decisions["decision_codes"][1] = np.uint8(0)
    decisions["reason_codes"][1] = np.uint16(0)
    destination = tmp_path / "must_not_publish.zarr"
    scratch = tmp_path / "node-local" / "invalid"
    scratch.mkdir(parents=True)

    with pytest.raises(
        ValueError,
        match="incomplete|complete bound positive/negative",
    ):
        create_sampled_images_full_training_crop_artifact(
            destination=destination,
            base_training_zarr=base,
            run_id="reviewed_4px",
            refined_run_id="reviewed",
            scratch_root=scratch,
            roi_size_wh=(4, 4),
        )

    assert not destination.exists()


def test_sampled_images_full_artifact_fails_closed_on_stale_approval_receipt(
    tmp_path: Path,
) -> None:
    base = _make_reviewed_sampled_images_full_base(tmp_path)
    direct = zarr.open_group(str(base), mode="a", use_consolidated=False)
    refined = direct["refined_detect_runs/reviewed"]
    review = dict(refined.attrs["detect_review_status"])
    receipt = dict(review["selector_ineligible_candidate_receipt"])
    receipt["frame_decision_digest"] = "0" * 64
    review["selector_ineligible_candidate_receipt"] = receipt
    refined.attrs["detect_review_status"] = review
    destination = tmp_path / "stale_approval_must_not_publish.zarr"
    scratch = tmp_path / "node-local" / "stale-approval"
    scratch.mkdir(parents=True)

    with pytest.raises(
        TrainingDatasetCompositionError,
        match="approval receipt is stale or malformed",
    ):
        create_sampled_images_full_training_crop_artifact(
            destination=destination,
            base_training_zarr=base,
            run_id="reviewed_4px",
            refined_run_id="reviewed",
            scratch_root=scratch,
            roi_size_wh=(4, 4),
        )

    assert not destination.exists()


def test_publish_training_crop_materialization_atomically_from_cache(
    tmp_path: Path,
) -> None:
    source_path, target_path, manifest_path, pixels = (
        _make_external_cache_materialization(tmp_path)
    )
    scratch = tmp_path / "node-local" / "job"
    scratch.mkdir(parents=True)

    result = publish_training_crop_materialization(
        destination=target_path,
        source_zarr=source_path,
        source_crop_run="crop_v2",
        run_id="crop_v2_atomic",
        scratch_root=scratch,
        roi_cache_manifest=manifest_path,
    )

    assert result["status"] == "complete"
    assert result["stage_selector_eligible"] is False
    assert result["row_count"] == 3
    root = zarr.open_group(str(target_path), mode="r", use_consolidated=True)
    run = root["crop_runs/crop_v2_atomic"]
    np.testing.assert_array_equal(run["roi_images"][:], pixels)
    assert run.attrs["stage_selector_eligible"] is False
    assert root["crop_runs"].attrs.get("latest") != "crop_v2_atomic"


def test_enrich_sampled_training_dataset_preserves_detection_review(
    tmp_path: Path,
) -> None:
    source_path, target_path, manifest_path, pixels = (
        _make_external_cache_materialization(tmp_path)
    )
    scratch = tmp_path / "node-local" / "composition"
    scratch.mkdir(parents=True)

    result = enrich_sampled_training_dataset(
        destination=target_path,
        source_zarr=source_path,
        source_crop_run="crop_v2",
        run_id="crop_v2_composed",
        scratch_root=scratch,
        roi_cache_manifest=manifest_path,
    )

    assert result["status"] == "complete"
    assert result["detect_run"] == "detect_review_seed"
    assert result["refined_detect_run"] == "refined_detect_review_seed"
    bound = bind_training_dataset_composition(
        target_path,
        crop_run_id="crop_v2_composed",
    )
    assert bound.crop.row_count == 3
    root = zarr.open_group(str(target_path), mode="r", use_consolidated=True)
    np.testing.assert_array_equal(
        root["crop_runs/crop_v2_composed/roi_images"][:], pixels
    )
    assert (
        root.attrs[TRAINING_DATASET_COMPOSITION_ATTRIBUTE]["payload"][
            "review_surfaces"
        ]["detection"]["row_identity"]
        == "instance_key"
    )


def test_instance_key_selection_preserves_multiple_rows_in_one_frame(
    tmp_path: Path,
) -> None:
    source_path, target_path, manifest_path, pixels = (
        _make_external_cache_materialization(tmp_path)
    )

    report = regenerate_training_crops_pynvvc(
        zarr_path=target_path,
        source_zarr_path=source_path,
        source_crop_run="crop_v2",
        target_crop_run="crop_v2_two_subjects",
        roi_cache_manifest=manifest_path,
        source_instance_keys=[102, 101],
    )

    assert report["total_rois"] == 2
    assert report["source_row_selection"]["mode"] == "instance_key_subset"
    run = zarr.open_group(str(target_path), mode="r", use_consolidated=False)[
        "crop_runs/crop_v2_two_subjects"
    ]
    np.testing.assert_array_equal(run["instance_key"][:], [101, 102])
    np.testing.assert_array_equal(run["frame_indices"][:], [1, 1])
    np.testing.assert_array_equal(run["source_crop_row_ids"][:], [0, 1])
    np.testing.assert_array_equal(run["frame_row_offsets"][:], [0, 0, 2, 2, 2, 2])
    np.testing.assert_array_equal(run["roi_images"][:], pixels[:2])
    bound = bind_training_crop_materialization(
        target_path,
        run_id="crop_v2_two_subjects",
    )
    assert bound.row_count == 2


def test_create_training_crop_artifact_publishes_whole_zarr_atomically(
    tmp_path: Path,
) -> None:
    source_path, base_training, manifest_path, pixels = (
        _make_external_cache_materialization(tmp_path)
    )
    destination = tmp_path / "batman_training.zarr"
    scratch = tmp_path / "local" / "job"
    scratch.mkdir(parents=True)

    result = create_training_crop_artifact(
        destination=destination,
        base_training_zarr=base_training,
        source_zarr=source_path,
        source_crop_run="crop_v2",
        run_id="crop_v2_training",
        scratch_root=scratch,
        roi_cache_manifest=manifest_path,
        source_instance_keys=[101, 102],
    )

    assert result["status"] == "complete"
    assert result["row_count"] == 2
    assert result["registry_activation"] == "deferred"
    root = zarr.open_group(str(destination), mode="r", use_consolidated=True)
    assert root.attrs["training_artifact_status"] == "complete"
    assert root.attrs["stage_selector_eligible"] is False
    assert "raw_video/images_full" in root
    assert "detect_runs/detect_review_seed" in root
    assert "refined_detect_runs/refined_detect_review_seed" in root
    np.testing.assert_array_equal(
        root["crop_runs/crop_v2_training/roi_images"][:], pixels[:2]
    )
    assert not list(destination.parent.glob(f".{destination.name}.publish_tmp.*"))


def test_composed_enrichment_rejects_crop_without_refined_detection_identity(
    tmp_path: Path,
) -> None:
    source_path, target_path, manifest_path, _pixels = (
        _make_external_cache_materialization(tmp_path)
    )
    target = zarr.open_group(str(target_path), mode="a", use_consolidated=False)
    target["refined_detect_runs/refined_detect_review_seed/instances/instance_key"][
        2
    ] = np.uint64(999)
    scratch = tmp_path / "node-local" / "bad-join"
    scratch.mkdir(parents=True)

    with pytest.raises(
        TrainingDatasetCompositionError,
        match="instance_key=103 is absent",
    ):
        enrich_sampled_training_dataset(
            destination=target_path,
            source_zarr=source_path,
            source_crop_run="crop_v2",
            run_id="crop_bad_join",
            scratch_root=scratch,
            roi_cache_manifest=manifest_path,
        )
    assert not (target_path / "crop_runs" / "crop_bad_join").exists()


def test_composed_enrichment_requires_detection_review_approval(
    tmp_path: Path,
) -> None:
    source_path, target_path, manifest_path, _pixels = (
        _make_external_cache_materialization(tmp_path)
    )
    target = zarr.open_group(str(target_path), mode="a", use_consolidated=False)
    del target["refined_detect_runs/refined_detect_review_seed"].attrs[
        "detect_review_status"
    ]
    scratch = tmp_path / "node-local" / "unapproved"
    scratch.mkdir(parents=True)

    with pytest.raises(
        TrainingDatasetCompositionError,
        match="approved before crop enrichment",
    ):
        enrich_sampled_training_dataset(
            destination=target_path,
            source_zarr=source_path,
            source_crop_run="crop_v2",
            run_id="crop_unapproved",
            scratch_root=scratch,
            roi_cache_manifest=manifest_path,
        )
    assert not (target_path / "crop_runs" / "crop_unapproved").exists()


def test_materialize_training_crops_rejects_tampered_flat_cache(
    tmp_path: Path,
) -> None:
    source_path, target_path, manifest_path, _pixels = (
        _make_external_cache_materialization(tmp_path)
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload_path = manifest_path.parent / manifest["array"]["bin_path"]
    payload = bytearray(payload_path.read_bytes())
    payload[0] ^= 0xFF
    payload_path.write_bytes(payload)

    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        regenerate_training_crops_pynvvc(
            zarr_path=target_path,
            source_zarr_path=source_path,
            source_crop_run="crop_v2",
            target_crop_run="crop_v2_training_materialized",
            roi_cache_manifest=manifest_path,
        )


def test_materialize_training_crops_rejects_named_but_changed_pixel_contract(
    tmp_path: Path,
) -> None:
    source_path, target_path, manifest_path, _pixels = (
        _make_external_cache_materialization(tmp_path)
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["builder"]["pixel_contract"]["image_representation"] = "changed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="does not exactly match"):
        regenerate_training_crops_pynvvc(
            zarr_path=target_path,
            source_zarr_path=source_path,
            source_crop_run="crop_v2",
            target_crop_run="crop_v2_training_materialized",
            roi_cache_manifest=manifest_path,
        )
