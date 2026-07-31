from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.segmentation import infer_unet_subject_masks as mod
from fisheye.shared.run_provenance import RUN_PROVENANCE_ATTR


def test_parser_defaults_to_probability_shards_and_accepts_regular_override() -> None:
    parser = mod._build_arg_parser()

    default_args = parser.parse_args(["recording.zarr", "model.pt"])
    regular_args = parser.parse_args(
        ["recording.zarr", "model.pt", "--no-mask-probs-sharding"]
    )

    assert default_args.mask_probs_shard_rois == mod.DEFAULT_MASK_PROBS_SHARD_ROIS
    assert regular_args.mask_probs_shard_rois is None


def _complete_partition_contract_fixture() -> tuple[object, object, object]:
    crop = _FakeGroup()
    crop.create_array(
        "frame_indices",
        data=np.asarray([0, 0, 2, 3], dtype=np.int64),
        overwrite=True,
    )
    crop.create_array(
        "frame_row_offsets",
        data=np.asarray([0, 2, 2, 3, 4], dtype=np.int64),
        overwrite=True,
    )
    binding = {
        "schema_id": "palette.acquisition_video_frame_window",
        "schema_version": 1,
        "recording_identity": "recording-1",
        "camera_identity": "camera-1",
        "clip_id": "clip-0",
        "actual_start_frame": 0,
        "end_frame_exclusive": 2,
        "frame_count": 2,
        "clip_index_document_sha256": "1" * 64,
        "clip_video_sha256": "2" * 64,
    }
    manifest = {
        "schema_id": "palette.crop_pixel_work_package",
        "schema_version": 1,
        "status": "complete",
        "package_id": "3" * 64,
        "selection": {
            "identity_mode": "instance_key",
            "ordering": "ascending_source_crop_row",
            "row_count": 2,
            "source_crop_total_rows": 4,
        },
        "materialization_binding": binding,
        "builder": {
            "semantics": "global_crop_rows_from_authenticated_acquisition_video_window_v1"
        },
    }
    source = SimpleNamespace(
        pixel_materialization_id="3" * 64,
        pixel_materialization_manifest="/scratch/work-package.json",
        source_crop_row_ids=np.asarray([0, 1], dtype=np.int64),
        frame_indices=np.asarray([0, 0], dtype=np.int64),
        _roi_images=SimpleNamespace(manifest=manifest),
    )
    args = SimpleNamespace(
        roi_work_package_role=mod.ROI_WORK_PACKAGE_ROLE_COMPLETE_PARTITION,
        source_collection_id="collection-1",
        source_collection_path="/groups/benchmark/plan.json",
        source_clip_id="clip-0",
        source_clip_index=0,
        source_work_unit_id="collection-1:clip-0",
        source_shard_id="clip-0",
    )
    return crop, source, args


def test_complete_partition_role_requires_and_records_exact_frame_offset_coverage() -> (
    None
):
    crop, source, args = _complete_partition_contract_fixture()

    attrs = mod._roi_work_package_publication_attrs(
        crop_group=crop,
        crop_source=source,
        selected_crop_rows=source.source_crop_row_ids,
        total_rois=2,
        args=args,
    )

    assert attrs["roi_work_package_role"] == "complete_collection_partition"
    assert attrs["canonical_finalization_policy"] == (
        "collection_shard_finalization_allowed"
    )
    contract = attrs["collection_partition_contract"]
    assert contract["payload"]["crop_rows"] == {
        "start": 0,
        "stop": 2,
        "count": 2,
        "source_crop_total_rows": 4,
    }
    assert contract["payload_digest"] == mod._canonical_document_sha256(
        contract["payload"]
    )


def test_work_package_role_defaults_to_delta_and_cannot_enter_collection_finalizer() -> (
    None
):
    crop, source, args = _complete_partition_contract_fixture()
    args.roi_work_package_role = None

    attrs = mod._roi_work_package_publication_attrs(
        crop_group=crop,
        crop_source=source,
        selected_crop_rows=source.source_crop_row_ids,
        total_rois=2,
        args=args,
    )

    assert attrs["incremental_materialization_role"] == "delta_replacement_rows"
    assert attrs["canonical_finalization_policy"] == ("incremental_compaction_required")
    assert "collection_partition_contract" not in attrs


def test_complete_partition_role_rejects_missing_collection_identity() -> None:
    crop, source, args = _complete_partition_contract_fixture()
    args.source_work_unit_id = None

    with pytest.raises(ValueError, match="missing.*source_work_unit_id"):
        mod._roi_work_package_publication_attrs(
            crop_group=crop,
            crop_source=source,
            selected_crop_rows=source.source_crop_row_ids,
            total_rois=2,
            args=args,
        )


def test_complete_partition_role_rejects_partial_frame_window_rows() -> None:
    crop, source, args = _complete_partition_contract_fixture()
    source.source_crop_row_ids = np.asarray([0], dtype=np.int64)
    source.frame_indices = np.asarray([0], dtype=np.int64)
    source._roi_images.manifest["selection"]["row_count"] = 1

    with pytest.raises(ValueError, match="do not exactly cover"):
        mod._roi_work_package_publication_attrs(
            crop_group=crop,
            crop_source=source,
            selected_crop_rows=source.source_crop_row_ids,
            total_rois=1,
            args=args,
        )


class _FakeArray:
    def __init__(
        self,
        data: np.ndarray,
        *,
        chunks: tuple[int, ...] | None = None,
        shards: tuple[int, ...] | None = None,
        fill_value=0,
    ) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.ndim = self._data.ndim
        self.dtype = self._data.dtype
        self.chunks = chunks or ((self.shape[0],) if self.shape else (1,))
        self.shards = shards
        self.fill_value = fill_value
        self.attrs: dict[str, object] = {}
        self.compressors = ()
        self.chunk_codecs = None
        self.filters = None

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, item, value) -> None:
        self._data[item] = value


class _FakeGroup:
    def __init__(self) -> None:
        self.attrs: dict[str, object] = {}
        self._children: dict[str, object] = {}

    def create_group(
        self,
        name: str,
        *,
        attributes: dict[str, object] | None = None,
    ) -> "_FakeGroup":
        group = _FakeGroup()
        if attributes is not None:
            group.attrs.update(attributes)
        self._children[name] = group
        return group

    def require_group(self, name: str) -> "_FakeGroup":
        existing = self._children.get(name)
        if isinstance(existing, _FakeGroup):
            return existing
        group = _FakeGroup()
        self._children[name] = group
        return group

    def create_array(
        self,
        name: str,
        *,
        data: np.ndarray | None = None,
        shape: tuple[int, ...] | None = None,
        dtype=None,
        chunks: tuple[int, ...] | None = None,
        shards: tuple[int, ...] | None = None,
        fill_value=None,
        overwrite: bool = False,
        **_kwargs,
    ):
        if not overwrite and name in self._children:
            raise ValueError(f"{name} already exists")
        if data is None:
            if shape is None:
                raise ValueError("shape required when data is omitted")
            arr = np.full(
                shape,
                0 if fill_value is None else fill_value,
                dtype=dtype or np.float32,
            )
        else:
            arr = np.asarray(data, dtype=dtype)
        fake = _FakeArray(arr, chunks=chunks, shards=shards, fill_value=fill_value)
        self._children[name] = fake
        return fake

    def get(self, name: str, default=None):
        return self._children.get(name, default)

    def __getitem__(self, name: str):
        return self._children[name]

    def __contains__(self, name: str) -> bool:
        return name in self._children

    def __delitem__(self, name: str) -> None:
        del self._children[name]

    def group_keys(self):
        return [
            name
            for name, value in self._children.items()
            if isinstance(value, _FakeGroup)
        ]

    def keys(self):
        return self._children.keys()


def _build_fake_root() -> _FakeGroup:
    root = _FakeGroup()
    root.attrs["width"] = 6
    root.attrs["height"] = 6
    crop_parent = root.create_group("crop_runs")
    crop = crop_parent.create_group("crop_geometry")
    crop.attrs["crop_storage_mode"] = "geometry_only"
    crop.attrs["crop_signature"] = "sig-001"
    crop.attrs["crop_revision"] = 4
    crop.attrs["detect_review_status_ref"] = (
        "refined_detect_runs/refined_001/review_status"
    )
    crop.attrs["source_detect_run"] = "detect_001"
    crop.attrs["detection_source_path"] = "detect_runs/detect_001"
    crop.attrs["video_source_path"] = "/tmp/source-video.mp4"
    crop.create_array(
        "frame_indices", data=np.array([0, 1], dtype=np.int32), overwrite=True
    )
    crop.create_array(
        "source_acquisition_frame_index",
        data=np.array([0, 1], dtype=np.int64),
        overwrite=True,
    )
    crop.create_array(
        "instance_key",
        data=np.array([101, 102], dtype=np.uint64),
        overwrite=True,
    )
    crop.create_array(
        "roi_coordinates_full",
        data=np.array([[0, 0], [1, 1]], dtype=np.int32),
        overwrite=True,
    )
    crop.create_array(
        "detection_indices", data=np.array([0, 1], dtype=np.int32), overwrite=True
    )
    crop.create_array(
        "detection_source", data=np.array([0, 0], dtype=np.int8), overwrite=True
    )
    keypoint_parent = root.create_group("refined_keypoints_runs")
    keypoint_parent.attrs["latest"] = "refined_kp_001"
    keypoint = keypoint_parent.create_group("refined_kp_001")
    keypoint.attrs["keypoint_labels"] = ["eye_left", "eye_right", "swim_bladder"]
    keypoint.create_array(
        "keypoints_roi",
        data=np.array(
            [
                [[1.0, 1.0], [3.0, 1.0], [2.0, 2.0]],
                [[1.0, 2.0], [3.0, 2.0], [2.0, 3.0]],
            ],
            dtype=np.float32,
        ),
        overwrite=True,
    )
    keypoint.create_array(
        "refined_success", data=np.array([True, True], dtype=bool), overwrite=True
    )
    keypoint.create_array(
        "usable_keypoints", data=np.array([True, True], dtype=bool), overwrite=True
    )
    return root


def _write_fake_raw_outputs(
    run_group: _FakeGroup,
    *,
    row_count: int,
    channel_count: int,
    height: int,
    width: int,
    mask_probs_dtype: str,
    write_masks_roi: bool,
    validation_accumulators=None,
) -> None:
    probability_dtype = np.uint8 if mask_probs_dtype == "uint8" else np.float16
    probabilities = np.zeros(
        (row_count, channel_count, height, width), dtype=probability_dtype
    )
    binary = np.zeros_like(probabilities, dtype=np.uint8)
    metrics_values = {
        "prob_max": np.zeros((row_count, channel_count), dtype=np.float32),
        "mask_present": np.zeros((row_count, channel_count), dtype=bool),
        "area_px": np.zeros((row_count, channel_count), dtype=np.float32),
        "centroid_xy": np.zeros((row_count, channel_count, 2), dtype=np.float32),
        "centroid_valid": np.zeros((row_count, channel_count), dtype=bool),
        "bbox_xyxy": np.zeros((row_count, channel_count, 4), dtype=np.float32),
        "bbox_valid": np.zeros((row_count, channel_count), dtype=bool),
    }
    run_group.create_array("mask_probs_roi", data=probabilities, overwrite=True)
    if write_masks_roi:
        run_group.create_array("masks_roi", data=binary, overwrite=True)
    run_group.create_array(
        "available_channels",
        data=np.ones((channel_count,), dtype=bool),
        overwrite=True,
    )
    metrics = run_group.require_group("metrics")
    for name, values in metrics_values.items():
        metrics.create_array(name, data=values, overwrite=True)
    if validation_accumulators is not None:
        mod._append_raw_worker_validation_batch(
            validation_accumulators,
            mod._SubjectMaskOutputBatch(
                start=0,
                stop=row_count,
                probs_out=probabilities,
                binary=binary if write_masks_roi else None,
                metrics=metrics_values,
            ),
        )


def test_normalize_checkpoint_state_dict_strips_torch_compile_prefix() -> None:
    state = {
        "_orig_mod.inc.block.0.weight": object(),
        "_orig_mod.out.bias": object(),
    }

    normalized = mod._normalize_checkpoint_state_dict(state)

    assert set(normalized) == {"inc.block.0.weight", "out.bias"}
    assert normalized["inc.block.0.weight"] is state["_orig_mod.inc.block.0.weight"]


def test_postprocess_logits_on_device_returns_storage_outputs_and_compact_metrics() -> (
    None
):
    logits = torch.tensor(
        [
            [
                [[0.0, 2.0], [-2.0, 4.0]],
                [[-4.0, -2.0], [-1.0, -0.25]],
            ],
            [
                [[-10.0, -10.0], [-10.0, -10.0]],
                [[0.5, 0.75], [1.0, 1.25]],
            ],
        ],
        dtype=torch.float32,
    )

    probs, binary, metrics = mod._postprocess_logits_on_device(
        logits,
        mask_probs_dtype="uint8",
        return_binary=True,
    )

    expected_probs = np.round(torch.sigmoid(logits).numpy() * 255.0).astype(np.uint8)
    expected_binary = (expected_probs >= 128).astype(np.uint8)
    np.testing.assert_array_equal(probs, expected_probs)
    np.testing.assert_array_equal(binary, expected_binary)
    np.testing.assert_allclose(
        metrics["prob_max"], expected_probs.max(axis=(2, 3)).astype(np.float32) / 255.0
    )
    np.testing.assert_array_equal(
        metrics["mask_present"], expected_binary.sum(axis=(2, 3)) > 0
    )
    np.testing.assert_allclose(
        metrics["area_px"], expected_binary.sum(axis=(2, 3)).astype(np.float32)
    )
    for channel_idx in range(expected_binary.shape[1]):
        expected_spatial = mod._compute_channel_spatial_metrics(
            expected_binary[:, channel_idx]
        )
        np.testing.assert_allclose(
            metrics["centroid_xy"][:, channel_idx, :], expected_spatial["centroid_xy"]
        )
        np.testing.assert_array_equal(
            metrics["centroid_valid"][:, channel_idx],
            expected_spatial["centroid_valid"],
        )
        np.testing.assert_allclose(
            metrics["bbox_xyxy"][:, channel_idx, :], expected_spatial["bbox_xyxy"]
        )
        np.testing.assert_array_equal(
            metrics["bbox_valid"][:, channel_idx], expected_spatial["bbox_valid"]
        )

    _probs_without_binary, skipped_binary, skipped_metrics = (
        mod._postprocess_logits_on_device(
            logits,
            mask_probs_dtype="uint8",
            return_binary=False,
        )
    )
    assert skipped_binary is None
    np.testing.assert_array_equal(
        skipped_metrics["mask_present"], metrics["mask_present"]
    )
    np.testing.assert_allclose(skipped_metrics["centroid_xy"], metrics["centroid_xy"])


def test_subject_mask_bbox_uses_positive_extent_half_open_pixel_edges() -> None:
    binary = np.zeros((1, 1, 4, 5), dtype=np.uint8)
    binary[0, 0, 1, 2] = 1

    tensor_metrics = mod._compute_spatial_metrics_from_binary_tensor(
        torch.from_numpy(binary)
    )
    channel_metrics = mod._compute_channel_spatial_metrics(binary[:, 0])

    expected = np.asarray([[2.0, 1.0, 3.0, 2.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        tensor_metrics["bbox_xyxy"].cpu().numpy()[:, 0],
        expected,
    )
    np.testing.assert_array_equal(channel_metrics["bbox_xyxy"], expected)
    assert np.all(expected[:, 2:] - expected[:, :2] == 1.0)


def test_write_subject_mask_outputs_can_skip_binary_masks_roi() -> None:
    class _FakeCropSource:
        total_rois = 2
        roi_shape = (2, 2)
        roi_array = _FakeArray(np.zeros((2, 2, 2), dtype=np.uint8), chunks=(1, 2, 2))

        def read_slice(self, start: int, stop: int) -> np.ndarray:
            return np.zeros((stop - start, 2, 2), dtype=np.uint8)

    class _FakeModel(torch.nn.Module):
        def forward(self, imgs: torch.Tensor) -> torch.Tensor:
            batch, _channels, height, width = imgs.shape
            return torch.zeros(
                (batch, 3, height, width), device=imgs.device, dtype=torch.float32
            )

    run_group = _FakeGroup()

    duration = mod._write_subject_mask_outputs(
        run_group,
        _FakeModel(),
        _FakeCropSource(),
        batch_size=1,
        device=torch.device("cpu"),
        mask_labels=("subject_body", "eyes_union", "swim_bladder"),
        mask_probs_chunk_rois=1,
        mask_probs_shard_rois=None,
        mask_probs_dtype="uint8",
        write_masks_roi=False,
        async_output=False,
        output_queue_size=2,
        model_input_transform=mod.resolve_model_input_transform((2, 2)),
        show_progress=False,
        console=mod.Console(),
        timing_profiler=None,
    )

    assert duration >= 0.0
    assert "mask_probs_roi" in run_group
    assert "masks_roi" not in run_group
    np.testing.assert_array_equal(
        np.asarray(run_group["available_channels"][:], dtype=bool),
        np.asarray([True, True, True], dtype=bool),
    )
    assert run_group["mask_probs_roi"].shape == (2, 3, 2, 2)
    assert run_group["mask_probs_roi"].dtype == np.dtype(np.uint8)
    np.testing.assert_array_equal(
        np.asarray(run_group["metrics"]["mask_present"][:], dtype=bool),
        np.ones((2, 3), dtype=bool),
    )
    np.testing.assert_allclose(
        np.asarray(run_group["metrics"]["area_px"][:], dtype=np.float32),
        np.full((2, 3), 4.0, dtype=np.float32),
    )


def test_write_subject_mask_outputs_pads_model_input_and_writes_native_shape() -> None:
    class _FakeCropSource:
        total_rois = 1
        roi_shape = (2, 2)
        roi_array = _FakeArray(np.zeros((1, 2, 2), dtype=np.uint8), chunks=(1, 2, 2))

        def read_slice(self, start: int, stop: int) -> np.ndarray:
            return np.ones((stop - start, 2, 2), dtype=np.uint8)

    seen: dict[str, tuple[int, ...]] = {}

    class _FakeModel(torch.nn.Module):
        def forward(self, imgs: torch.Tensor) -> torch.Tensor:
            seen["input_shape"] = tuple(imgs.shape)
            batch, _channels, height, width = imgs.shape
            return torch.zeros(
                (batch, 3, height, width), device=imgs.device, dtype=torch.float32
            )

    run_group = _FakeGroup()
    transform = mod.resolve_model_input_transform(
        (2, 2), mode="pad_to_size", model_hw=(4, 4)
    )

    mod._write_subject_mask_outputs(
        run_group,
        _FakeModel(),
        _FakeCropSource(),
        batch_size=1,
        device=torch.device("cpu"),
        mask_labels=("subject_body", "eyes_union", "swim_bladder"),
        mask_probs_chunk_rois=1,
        mask_probs_shard_rois=None,
        mask_probs_dtype="uint8",
        write_masks_roi=True,
        async_output=False,
        output_queue_size=2,
        model_input_transform=transform,
        show_progress=False,
        console=mod.Console(),
        timing_profiler=None,
    )

    assert seen["input_shape"] == (1, 1, 4, 4)
    assert run_group["mask_probs_roi"].shape == (1, 3, 2, 2)
    assert run_group["masks_roi"].shape == (1, 3, 2, 2)


def test_write_subject_mask_outputs_async_matches_serial_outputs() -> None:
    class _FakeCropSource:
        total_rois = 3
        roi_shape = (2, 2)
        roi_array = _FakeArray(np.zeros((3, 2, 2), dtype=np.uint8), chunks=(1, 2, 2))

        def read_slice(self, start: int, stop: int) -> np.ndarray:
            return np.zeros((stop - start, 2, 2), dtype=np.uint8)

    class _FakeModel(torch.nn.Module):
        def forward(self, imgs: torch.Tensor) -> torch.Tensor:
            batch, _channels, height, width = imgs.shape
            return torch.full(
                (batch, 3, height, width), 2.0, device=imgs.device, dtype=torch.float32
            )

    serial_group = _FakeGroup()
    async_group = _FakeGroup()
    kwargs = {
        "model": _FakeModel(),
        "roi_source": _FakeCropSource(),
        "batch_size": 1,
        "device": torch.device("cpu"),
        "mask_labels": ("subject_body", "eyes_union", "swim_bladder"),
        "mask_probs_chunk_rois": 1,
        "mask_probs_shard_rois": None,
        "mask_probs_dtype": "uint8",
        "write_masks_roi": False,
        "model_input_transform": mod.resolve_model_input_transform((2, 2)),
        "show_progress": False,
        "console": mod.Console(),
        "timing_profiler": None,
    }

    mod._write_subject_mask_outputs(
        serial_group,
        async_output=False,
        output_queue_size=2,
        **kwargs,
    )
    mod._write_subject_mask_outputs(
        async_group,
        async_output=True,
        output_queue_size=1,
        **kwargs,
    )

    np.testing.assert_array_equal(
        np.asarray(async_group["mask_probs_roi"][:]),
        np.asarray(serial_group["mask_probs_roi"][:]),
    )
    for name in (
        "prob_max",
        "mask_present",
        "area_px",
        "centroid_xy",
        "centroid_valid",
        "bbox_xyxy",
        "bbox_valid",
    ):
        np.testing.assert_array_equal(
            np.asarray(async_group["metrics"][name][:]),
            np.asarray(serial_group["metrics"][name][:]),
        )
    assert "masks_roi" not in async_group


def test_postpack_probability_shards_exactly_replaces_working_array() -> None:
    run_group = _FakeGroup()
    values = np.arange(5 * 3 * 2 * 2, dtype=np.uint8).reshape(5, 3, 2, 2)
    run_group.create_array(
        mod.MASK_PROBS_WORKING_ARRAY,
        data=values,
        chunks=(1, 1, 2, 2),
        overwrite=True,
    )

    summary = mod._postpack_probability_shards(
        run_group,
        source_name=mod.MASK_PROBS_WORKING_ARRAY,
        shard_rows=4,
        profiler=mod.InferenceTimingProfiler(enabled=False),
    )

    assert mod.MASK_PROBS_WORKING_ARRAY not in run_group
    assert mod.MASK_PROBS_CANONICAL_ARRAY in run_group
    packed = run_group[mod.MASK_PROBS_CANONICAL_ARRAY]
    np.testing.assert_array_equal(np.asarray(packed[:]), values)
    assert packed.chunks == (1, 1, 2, 2)
    assert packed.shards == (4, 1, 2, 2)
    assert packed.attrs["storage_layout"] == "indexed_sharding_v1"
    assert summary["exact_match"] is True
    assert summary["source_sha256"] == summary["destination_sha256"]
    assert summary["inner_chunks_per_shard"] == 4


def test_postpack_probability_shards_uses_real_indexed_sharding() -> None:
    run_group = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    values = np.arange(9 * 2 * 3 * 3, dtype=np.uint8).reshape(9, 2, 3, 3)
    run_group.create_array(
        mod.MASK_PROBS_WORKING_ARRAY,
        data=values,
        chunks=(2, 1, 3, 3),
        overwrite=True,
    )

    summary = mod._postpack_probability_shards(
        run_group,
        source_name=mod.MASK_PROBS_WORKING_ARRAY,
        shard_rows=8,
        profiler=mod.InferenceTimingProfiler(enabled=False),
    )

    packed = run_group[mod.MASK_PROBS_CANONICAL_ARRAY]
    assert packed.chunks == (2, 1, 3, 3)
    assert packed.shards == (8, 1, 3, 3)
    np.testing.assert_array_equal(np.asarray(packed[:]), values)
    assert summary["exact_match"] is True


def test_postpack_probability_shards_rejects_unaligned_outer_rows() -> None:
    run_group = _FakeGroup()
    run_group.create_array(
        mod.MASK_PROBS_WORKING_ARRAY,
        data=np.zeros((5, 3, 2, 2), dtype=np.uint8),
        chunks=(2, 1, 2, 2),
        overwrite=True,
    )

    with pytest.raises(ValueError, match="integer multiple"):
        mod._postpack_probability_shards(
            run_group,
            source_name=mod.MASK_PROBS_WORKING_ARRAY,
            shard_rows=5,
            profiler=mod.InferenceTimingProfiler(enabled=False),
        )

    assert mod.MASK_PROBS_WORKING_ARRAY in run_group
    assert mod.MASK_PROBS_CANONICAL_ARRAY not in run_group


def test_postpack_probability_shards_preserves_working_array_on_digest_mismatch(
    monkeypatch,
) -> None:
    run_group = _FakeGroup()
    run_group.create_array(
        mod.MASK_PROBS_WORKING_ARRAY,
        data=np.arange(5 * 3 * 2 * 2, dtype=np.uint8).reshape(5, 3, 2, 2),
        chunks=(1, 1, 2, 2),
        overwrite=True,
    )
    digests = iter(("source-digest", "destination-digest"))
    monkeypatch.setattr(
        mod, "_probability_array_digest", lambda *_args, **_kwargs: next(digests)
    )

    with pytest.raises(RuntimeError, match="digest mismatch"):
        mod._postpack_probability_shards(
            run_group,
            source_name=mod.MASK_PROBS_WORKING_ARRAY,
            shard_rows=4,
            profiler=mod.InferenceTimingProfiler(enabled=False),
        )

    assert mod.MASK_PROBS_WORKING_ARRAY in run_group
    assert mod.MASK_PROBS_CANONICAL_ARRAY in run_group


def test_double_buffered_probability_writer_handles_crossing_batches_and_partial_shard() -> (
    None
):
    run_group = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    values = np.arange(17 * 2 * 3 * 3, dtype=np.uint8).reshape(17, 2, 3, 3)
    destination = run_group.create_array(
        mod.MASK_PROBS_CANONICAL_ARRAY,
        shape=values.shape,
        dtype=values.dtype,
        chunks=(2, 1, 3, 3),
        shards=(8, 1, 3, 3),
        overwrite=True,
    )
    writer = mod._DoubleBufferedProbabilityShardWriter(
        destination,
        shard_rows=8,
        profiler=mod.InferenceTimingProfiler(enabled=True),
    )

    writer[0:3] = values[0:3]
    writer[3:11] = values[3:11]
    writer[11:17] = values[11:17]
    summary = writer.finish(validation_row_step=2)

    np.testing.assert_array_equal(np.asarray(destination[:]), values)
    assert summary["schema_id"] == mod.MASK_PROBS_DIRECT_SHARDING_SCHEMA
    assert summary["write_mode"] == "double_buffered_direct"
    assert summary["buffer_count"] == 2
    assert summary["full_row_shards_written"] == 2
    assert summary["partial_row_shards_written"] == 1
    assert summary["source_working_array_created"] is False
    assert (
        summary["source_sha256_by_channel"] == summary["destination_sha256_by_channel"]
    )
    assert summary["source_sha256"] == summary["destination_sha256"]
    assert summary["exact_match"] is True


def test_double_buffered_probability_writer_rejects_nonsequential_batches() -> None:
    destination = _FakeArray(
        np.zeros((8, 2, 2, 2), dtype=np.uint8),
        chunks=(2, 1, 2, 2),
        shards=(4, 1, 2, 2),
    )
    writer = mod._DoubleBufferedProbabilityShardWriter(
        destination,
        shard_rows=4,
        profiler=mod.InferenceTimingProfiler(enabled=False),
    )

    with pytest.raises(ValueError, match="must be sequential"):
        writer[1:3] = np.zeros((2, 2, 2, 2), dtype=np.uint8)

    writer[0:8] = np.zeros((8, 2, 2, 2), dtype=np.uint8)
    writer.finish(validation_row_step=2)


def test_double_buffered_probability_writer_fails_closed_on_destination_digest_mismatch(
    monkeypatch,
) -> None:
    destination = _FakeArray(
        np.zeros((5, 2, 2, 2), dtype=np.uint8),
        chunks=(1, 1, 2, 2),
        shards=(4, 1, 2, 2),
    )
    writer = mod._DoubleBufferedProbabilityShardWriter(
        destination,
        shard_rows=4,
        profiler=mod.InferenceTimingProfiler(enabled=False),
    )
    values = np.arange(5 * 2 * 2 * 2, dtype=np.uint8).reshape(5, 2, 2, 2)
    writer[0:5] = values
    monkeypatch.setattr(
        mod,
        "_probability_array_digests_by_channel",
        lambda *_args, **_kwargs: ["0" * 64, "1" * 64],
    )

    with pytest.raises(RuntimeError, match="digest mismatch"):
        writer.finish(validation_row_step=1)

    np.testing.assert_array_equal(np.asarray(destination[:]), values)


def test_write_subject_mask_outputs_writes_double_buffered_probability_shards() -> None:
    class _FakeCropSource:
        total_rois = 5
        roi_shape = (2, 2)
        roi_array = _FakeArray(np.zeros((5, 2, 2), dtype=np.uint8), chunks=(1, 2, 2))

        def read_slice(self, start: int, stop: int) -> np.ndarray:
            return np.zeros((stop - start, 2, 2), dtype=np.uint8)

    class _FakeModel(torch.nn.Module):
        def forward(self, imgs: torch.Tensor) -> torch.Tensor:
            batch, _channels, height, width = imgs.shape
            return torch.full(
                (batch, 3, height, width), 2.0, device=imgs.device, dtype=torch.float32
            )

    run_group = _FakeGroup()
    mod._write_subject_mask_outputs(
        run_group,
        _FakeModel(),
        _FakeCropSource(),
        batch_size=2,
        device=torch.device("cpu"),
        mask_labels=("subject_body", "eyes_union", "swim_bladder"),
        mask_probs_chunk_rois=1,
        mask_probs_shard_rois=4,
        mask_probs_dtype="uint8",
        write_masks_roi=False,
        async_output=True,
        output_queue_size=1,
        model_input_transform=mod.resolve_model_input_transform((2, 2)),
        show_progress=False,
        console=mod.Console(),
        timing_profiler=None,
    )

    assert mod.MASK_PROBS_WORKING_ARRAY not in run_group
    assert run_group[mod.MASK_PROBS_CANONICAL_ARRAY].shards == (4, 1, 2, 2)
    summary = run_group.attrs["mask_probs_shard_write"]
    assert summary["status"] == "complete"
    assert summary["write_mode"] == "double_buffered_direct"
    assert summary["buffer_count"] == 2
    assert summary["source_working_array_created"] is False


def test_subject_mask_shard_inference_supports_geometry_only_crop_with_temporary_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    checkpoint_path = tmp_path / "subject_unet.pt"
    checkpoint_path.write_text("", encoding="utf-8")

    seen: dict[str, object] = {}
    fake_root = _build_fake_root()
    cache_dir = tmp_path / "roi-cache" / "fake-cache.zarr"
    cache_dir.mkdir(parents=True, exist_ok=True)

    class _FakeCropSource:
        def __init__(self, crop_group) -> None:
            self.crop_group = crop_group
            self.crop_run_name = "crop_geometry"
            self.total_rois = 2
            self.roi_shape = (4, 4)
            self.roi_array = None
            self.storage_mode = "geometry_only"
            self.roi_read_mode = "temporary_cache"
            self.roi_cache_policy = "always"
            self.roi_cache_used = True
            self.roi_cache_key = "cache-key-001"
            self.roi_cache_path = str(cache_dir)
            self.roi_cache_canonical_path = (
                "/groups/cache/fake-cache.flat_roi_cache.json"
            )
            self.roi_live_acceleration_requested = "cpu"
            self.roi_live_acceleration_effective = "cpu"
            self.roi_live_acceleration_fallback_reason = None
            self.roi_live_gpu_chunk_frames = 11
            self.frame_source_kind = "raw_video/images_full"
            self.frame_source_path = None

        def read_slice(self, start: int, stop: int) -> np.ndarray:
            return np.zeros((stop - start, 4, 4), dtype=np.uint8)

        def close(self) -> None:
            return None

    def _fake_load_checkpoint(_path: Path, _device) -> tuple[object, dict[str, object]]:
        return object(), {
            "label_schema_id": "subject_v1_union",
            "mask_labels": ["subject_body", "eyes_union", "swim_bladder"],
            "best_val_dice": 0.88,
        }

    def _fake_write_subject_mask_outputs(
        run_group,
        model,
        roi_source,
        *,
        batch_size,
        device,
        mask_labels,
        mask_probs_chunk_rois,
        mask_probs_shard_rois,
        mask_probs_dtype,
        write_masks_roi,
        async_output,
        output_queue_size,
        model_input_transform,
        show_progress,
        console,
        timing_profiler,
        input_pixels_sha256,
        validation_accumulators,
    ) -> float:
        del model, batch_size, device, console
        seen["mask_labels"] = tuple(mask_labels)
        seen["mask_probs_chunk_rois"] = mask_probs_chunk_rois
        seen["mask_probs_shard_rois"] = mask_probs_shard_rois
        seen["mask_probs_dtype"] = mask_probs_dtype
        seen["write_masks_roi"] = write_masks_roi
        seen["async_output"] = async_output
        seen["output_queue_size"] = output_queue_size
        seen["model_input_transform"] = model_input_transform.to_attrs()
        seen["show_progress"] = show_progress
        batch = roi_source.read_slice(0, roi_source.total_rois)
        input_pixels_sha256.update(np.ascontiguousarray(batch).view(np.uint8))
        seen["batch_shape"] = batch.shape
        seen["roi_read_mode"] = roi_source.roi_read_mode
        seen["roi_cache_used"] = roi_source.roi_cache_used
        seen["roi_cache_path"] = roi_source.roi_cache_path
        seen["roi_cache_canonical_path"] = roi_source.roi_cache_canonical_path
        _write_fake_raw_outputs(
            run_group,
            row_count=roi_source.total_rois,
            channel_count=3,
            height=roi_source.roi_shape[0],
            width=roi_source.roi_shape[1],
            mask_probs_dtype=mask_probs_dtype,
            write_masks_roi=write_masks_roi,
            validation_accumulators=validation_accumulators,
        )
        if timing_profiler is not None:
            timing_profiler.record("roi_read", 0.01, items=roi_source.total_rois)
        return 0.25

    monkeypatch.setattr(mod, "_load_checkpoint", _fake_load_checkpoint)
    monkeypatch.setattr(
        mod, "_write_subject_mask_outputs", _fake_write_subject_mask_outputs
    )
    monkeypatch.setattr(mod.zarr, "open", lambda *_args, **_kwargs: fake_root)
    monkeypatch.setattr(
        mod.CropImageSource,
        "open",
        lambda root, **_kwargs: _FakeCropSource(root["crop_runs"]["crop_geometry"]),
    )
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda **_kwargs: {
            "platform": {
                "hostname": "test-host",
                "system": "Linux",
                "release": "6.0",
                "python_version": "3.11",
                "machine": "x86_64",
            },
            "environment": {"name": "test-env"},
        },
    )
    monkeypatch.setattr(
        mod,
        "get_git_info",
        lambda: {
            "commit_hash": "abc123",
            "short_hash": "abc123",
            "branch": "main",
            "is_dirty": False,
            "remote_url": "git@example.com:palette.git",
        },
    )

    mod.main(
        [
            str(zarr_path),
            "--checkpoint",
            str(checkpoint_path),
            "--crop-run",
            "crop_geometry",
            "--output-parent",
            "subject_mask_shard_runs",
            "--roi-cache-policy",
            "always",
            "--roi-cache-dir",
            str(tmp_path / "roi-cache"),
            "--roi-live-acceleration",
            "cpu",
            "--roi-live-gpu-chunk-frames",
            "11",
            "--mask-probs-chunk-rois",
            "2",
            "--mask-probs-shard-rois",
            "4",
            "--assignment-keypoint-group",
            "refined_keypoints_runs",
            "--assignment-keypoint-run",
            "refined_kp_001",
            "--profile-timings",
        ]
    )

    subject_parent = fake_root["subject_mask_shard_runs"]
    run_group = subject_parent[subject_parent.group_keys()[0]]

    assert seen["batch_shape"] == (2, 4, 4)
    assert seen["mask_labels"] == ("subject_body", "eyes_union", "swim_bladder")
    assert seen["roi_read_mode"] == "temporary_cache"
    assert seen["roi_cache_used"] is True
    assert Path(str(seen["roi_cache_path"])).exists()
    assert (
        seen["roi_cache_canonical_path"]
        == "/groups/cache/fake-cache.flat_roi_cache.json"
    )
    assert seen["mask_probs_chunk_rois"] == 2
    assert seen["mask_probs_shard_rois"] == 4
    assert seen["mask_probs_dtype"] == "uint8"
    assert seen["write_masks_roi"] is False
    assert seen["async_output"] is True
    assert seen["output_queue_size"] == 2
    assert seen["show_progress"] is False

    assert run_group.attrs["source_crop_run"] == "crop_geometry"
    assert run_group.attrs["source_crop_storage_mode"] == "geometry_only"
    assert run_group.attrs["source_crop_signature"] == "sig-001"
    assert run_group.attrs["source_crop_revision"] == 4
    assert (
        run_group.attrs["source_detect_review_status_ref"]
        == "refined_detect_runs/refined_001/review_status"
    )
    assert run_group.attrs["source_roi_read_mode"] == "temporary_cache"
    assert run_group.attrs["roi_cache_policy"] == "always"
    assert run_group.attrs["source_roi_cache_used"] is True
    assert run_group.attrs["source_roi_cache_path"]
    assert (
        run_group.attrs["source_roi_cache_canonical_path"]
        == "/groups/cache/fake-cache.flat_roi_cache.json"
    )
    assert run_group.attrs["source_roi_live_acceleration_requested"] == "cpu"
    assert run_group.attrs["source_roi_live_acceleration_effective"] == "cpu"
    assert run_group.attrs["source_roi_live_gpu_chunk_frames"] == 11
    assert run_group.attrs["label_schema_id"] == "subject_v1_union"
    assert list(run_group.attrs["mask_labels"]) == [
        "subject_body",
        "eyes_union",
        "swim_bladder",
    ]
    assert run_group.attrs["assignment_keypoint_group"] == "refined_keypoints_runs"
    assert run_group.attrs["assignment_keypoints_run"] == "refined_kp_001"
    assert (
        run_group.attrs["assignment_keypoint_contract"]
        == "subject_eyes_union_assignment_keypoints_v1"
    )
    assert run_group.attrs["assignment_keypoint_role"] == "eyes_union_lr_assignment"
    assert run_group.attrs["assignment_keypoint_selection"] == "cli_explicit"
    assert run_group.attrs["assignment_keypoint_success_dataset"] == "usable_keypoints"
    assert run_group.attrs["assignment_keypoint_eye_indices"] == {
        "eye_left": 0,
        "eye_right": 1,
    }
    assert run_group.attrs["method"] == "unet_subject_mask_segmenter"
    assert run_group.attrs["run_semantics"] == "unet_subject_mask_inference"
    assert run_group.attrs["mask_probs_shard_rois"] == 4
    assert run_group.attrs["mask_probs_storage_layout"] == "indexed_sharding_v1"
    assert run_group.attrs["mask_probs_storage_policy"] == "default_indexed_sharding_v1"
    assert run_group.attrs["mask_probs_default_shard_rois"] == 2048
    assert run_group.attrs["profile_timings_enabled"] is True
    provenance_inputs = run_group.attrs["provenance"]["inputs"]
    assert provenance_inputs["source_crop_signature"] == "sig-001"
    assert provenance_inputs["source_crop_revision"] == 4
    assert (
        provenance_inputs["source_detect_review_status_ref"]
        == "refined_detect_runs/refined_001/review_status"
    )
    assert (
        provenance_inputs["roi_cache_canonical_path"]
        == "/groups/cache/fake-cache.flat_roi_cache.json"
    )
    assert provenance_inputs["assignment_keypoint_group"] == "refined_keypoints_runs"
    assert provenance_inputs["assignment_keypoints_run"] == "refined_kp_001"
    assert run_group.attrs["provenance"]["parameters"]["mask_probs_shard_rois"] == 4
    assert (
        run_group.attrs["provenance"]["parameters"]["mask_probs_storage_policy"]
        == "default_indexed_sharding_v1"
    )
    assert run_group["mask_probs_roi"].shape == (2, 3, 4, 4)
    assert "masks_roi" not in run_group
    np.testing.assert_array_equal(
        np.asarray(run_group["detection_source"][:]),
        np.asarray([0, 0], dtype=np.int8),
    )


def test_canonical_writer_omits_legacy_detection_source_through_publication(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Exercise main's canonical ordering with a crop that carries the legacy array."""

    zarr_path = tmp_path / "recording_analysis.zarr"
    checkpoint_path = tmp_path / "subject_unet.pt"
    checkpoint_path.write_text("", encoding="utf-8")
    fake_root = _build_fake_root()
    crop_group = fake_root["crop_runs"]["crop_geometry"]
    crop_group.path = "crop_runs/crop_geometry"
    crop_group.attrs["crop_storage_mode"] = "materialized"
    assert "detection_source" in crop_group

    selected = {
        "source_crop_row_ids": np.asarray([0, 1], dtype="<i8"),
        "instance_key": np.asarray([101, 102], dtype="<u8"),
        "source_acquisition_frame_index": np.asarray([0, 1], dtype="<i8"),
        "source_crop_xywh": np.asarray(
            [[1, 2, 4, 4], [3, 4, 4, 4]],
            dtype="<i4",
        ),
    }

    class _CanonicalEvidence:
        crop_geometry = type(
            "CropGeometry",
            (),
            {
                "row_identity": type(
                    "RowIdentity",
                    (),
                    {"leading_dimension": 2},
                )()
            },
        )()
        roi_frame = type(
            "RoiFrame",
            (),
            {
                "endpoint": type(
                    "Endpoint",
                    (),
                    {"height": 4, "width": 4},
                )()
            },
        )()

    class _CanonicalCropSource:
        def __init__(self) -> None:
            self.crop_group = crop_group
            self.crop_run_name = "crop_geometry"
            self.total_rois = 2
            self.roi_shape = (4, 4)
            self.roi_array = _FakeArray(
                np.zeros((2, 4, 4), dtype=np.uint8),
                chunks=(1, 4, 4),
            )
            self._roi_images = self.roi_array
            self.source_crop_row_ids = None
            self.roi_coordinates_full = selected["source_crop_xywh"][:, :2]
            self.frame_indices = selected["source_acquisition_frame_index"]
            self.storage_mode = "materialized"
            self.frame_source_kind = "roi_images"
            self.frame_source_path = None
            self.roi_read_mode = "materialized_crop_run"
            self.roi_cache_policy = "auto"
            self.roi_cache_used = False
            self.roi_cache_backend = None
            self.roi_cache_key = None
            self.roi_cache_path = None
            self.roi_cache_canonical_path = None
            self.roi_live_acceleration_requested = "auto"
            self.roi_live_acceleration_effective = "cpu"
            self.roi_live_acceleration_fallback_reason = None
            self.roi_live_gpu_chunk_frames = 256
            self.pixel_materialization_id = None
            self.pixel_materialization_manifest = None
            self.roi_pixel_contract = None
            self.roi_image_representation = "grayscale_uint8"

        def read_slice(self, start: int, stop: int) -> np.ndarray:
            return np.zeros((stop - start, 4, 4), dtype=np.uint8)

        def close(self) -> None:
            return None

    def _run_at(root, run_path: str):
        parent_name, run_name = run_path.split("/", 1)
        return root[parent_name][run_name]

    def _fake_load_checkpoint(_path: Path, _device):
        return object(), {
            "label_schema_id": "subject_v1_union",
            "mask_labels": ["subject_body", "eyes_union", "swim_bladder"],
            "best_val_dice": 0.88,
        }

    def _fake_write_outputs(run_group, _model, roi_source, **_kwargs) -> float:
        shape = (roi_source.total_rois, 3, *roi_source.roi_shape)
        output_values = {
            "mask_probs_roi": np.zeros(shape, dtype=np.uint8),
            "metrics/prob_max": np.zeros((2, 3), dtype=np.float32),
            "metrics/mask_present": np.zeros((2, 3), dtype=bool),
            "metrics/area_px": np.zeros((2, 3), dtype=np.float32),
            "metrics/centroid_xy": np.zeros((2, 3, 2), dtype=np.float32),
            "metrics/centroid_valid": np.zeros((2, 3), dtype=bool),
            "metrics/bbox_xyxy": np.zeros((2, 3, 4), dtype=np.float32),
            "metrics/bbox_valid": np.zeros((2, 3), dtype=bool),
        }
        run_group.create_array(
            "mask_probs_roi",
            data=output_values["mask_probs_roi"],
            overwrite=True,
        )
        run_group.create_array(
            "available_channels",
            data=np.ones((3,), dtype=bool),
            overwrite=True,
        )
        metrics = run_group.require_group("metrics")
        metrics.create_array(
            "prob_max",
            data=output_values["metrics/prob_max"],
            overwrite=True,
        )
        metrics.create_array(
            "mask_present",
            data=output_values["metrics/mask_present"],
            overwrite=True,
        )
        metrics.create_array(
            "area_px",
            data=output_values["metrics/area_px"],
            overwrite=True,
        )
        metrics.create_array(
            "centroid_xy",
            data=output_values["metrics/centroid_xy"],
            overwrite=True,
        )
        metrics.create_array(
            "centroid_valid",
            data=output_values["metrics/centroid_valid"],
            overwrite=True,
        )
        metrics.create_array(
            "bbox_xyxy",
            data=output_values["metrics/bbox_xyxy"],
            overwrite=True,
        )
        metrics.create_array(
            "bbox_valid",
            data=output_values["metrics/bbox_valid"],
            overwrite=True,
        )
        for path, accumulator in _kwargs["validation_accumulators"].items():
            accumulator.append(0, output_values[path])
        return 0.01

    observed: dict[str, bool] = {}

    def _fake_prepare(root, run_path: str, **_kwargs):
        assert "detection_source" not in _run_at(root, run_path)
        observed["prepared_without_detection_source"] = True
        return object()

    def _fake_publish(root, run_path: str, **_kwargs):
        assert "detection_source" not in _run_at(root, run_path)
        observed["published_without_detection_source"] = True
        return object()

    def _fake_activate(
        _root,
        parent,
        _surfaces,
        *,
        run_name: str,
        **_kwargs,
    ) -> None:
        run = parent[run_name]
        assert "detection_source" not in run
        parent.attrs["latest_complete"] = run_name
        parent.attrs["latest"] = run_name
        if "latest_pending" in parent.attrs:
            del parent.attrs["latest_pending"]
        run.attrs["stage_selector_eligible"] = True

    monkeypatch.setattr(mod, "_load_checkpoint", _fake_load_checkpoint)
    monkeypatch.setattr(mod, "_write_subject_mask_outputs", _fake_write_outputs)
    monkeypatch.setattr(mod.zarr, "open", lambda *_args, **_kwargs: fake_root)
    monkeypatch.setattr(
        mod.CropImageSource,
        "open",
        lambda *_args, **_kwargs: _CanonicalCropSource(),
    )
    monkeypatch.setattr(
        mod,
        "load_persisted_subject_mask_crop_source",
        lambda *_args, **_kwargs: _CanonicalEvidence(),
    )
    monkeypatch.setattr(
        mod,
        "require_direct_subject_mask_crop_pixel_source",
        lambda source, _pixels: source,
    )
    monkeypatch.setattr(
        mod,
        "selected_subject_mask_crop_values",
        lambda *_args, **_kwargs: {
            name: value.copy() for name, value in selected.items()
        },
    )
    monkeypatch.setattr(mod, "prepare_subject_mask_coordinate_context", _fake_prepare)
    monkeypatch.setattr(
        mod,
        "capture_subject_mask_coordinate_publication_checkpoint",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(mod, "publish_subject_mask_coordinate_surfaces", _fake_publish)
    monkeypatch.setattr(
        mod,
        "_activate_validated_subject_mask_coordinate_surfaces",
        _fake_activate,
    )
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda **_kwargs: {"platform": {}, "environment": {}},
    )
    monkeypatch.setattr(
        mod,
        "get_git_info",
        lambda: {"commit_hash": "abc123", "short_hash": "abc123"},
    )

    mod.main(
        [
            str(zarr_path),
            "--checkpoint",
            str(checkpoint_path),
            "--crop-run",
            "crop_geometry",
            "--run-name",
            "canonical_no_detection_source",
            "--device",
            "cpu",
            "--no-mask-probs-sharding",
            "--no-progress",
            "--defer-registry-status",
        ]
    )

    run = fake_root["subject_mask_runs"]["canonical_no_detection_source"]
    assert observed == {
        "prepared_without_detection_source": True,
        "published_without_detection_source": True,
    }
    assert "detection_source" not in run
    assert run.attrs["stage_selector_eligible"] is True


def test_infer_unet_subject_masks_writes_lr_checkpoint_schema(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    checkpoint_path = tmp_path / "subject_unet_lr.pt"
    checkpoint_path.write_text("", encoding="utf-8")

    seen: dict[str, object] = {}
    fake_root = _build_fake_root()

    class _FakeCropSource:
        def __init__(self, crop_group) -> None:
            self.crop_group = crop_group
            self.crop_run_name = "crop_geometry"
            self.total_rois = 2
            self.roi_shape = (4, 4)
            self.roi_array = None
            self.storage_mode = "geometry_only"
            self.roi_read_mode = "live"
            self.roi_cache_policy = "never"
            self.roi_cache_used = False
            self.roi_cache_key = None
            self.roi_cache_path = None
            self.roi_live_acceleration_requested = "cpu"
            self.roi_live_acceleration_effective = "cpu"
            self.roi_live_acceleration_fallback_reason = None
            self.roi_live_gpu_chunk_frames = 32
            self.frame_source_kind = "raw_video/images_full"
            self.frame_source_path = None

        def read_slice(self, start: int, stop: int) -> np.ndarray:
            return np.zeros((stop - start, 4, 4), dtype=np.uint8)

        def close(self) -> None:
            return None

    def _fake_load_checkpoint(_path: Path, _device) -> tuple[object, dict[str, object]]:
        return object(), {
            "label_schema_id": "subject_v1_lr",
            "mask_labels": ["subject_body", "eye_left", "eye_right", "swim_bladder"],
            "best_val_dice": 0.91,
        }

    def _fake_write_subject_mask_outputs(
        run_group,
        model,
        roi_source,
        *,
        batch_size,
        device,
        mask_labels,
        mask_probs_chunk_rois,
        mask_probs_shard_rois,
        mask_probs_dtype,
        write_masks_roi,
        async_output,
        output_queue_size,
        model_input_transform,
        show_progress,
        console,
        timing_profiler,
        input_pixels_sha256,
        validation_accumulators,
    ) -> float:
        del (
            model,
            batch_size,
            device,
            mask_probs_chunk_rois,
            mask_probs_shard_rois,
            async_output,
            output_queue_size,
            model_input_transform,
            show_progress,
            console,
            timing_profiler,
        )
        seen["mask_labels"] = tuple(mask_labels)
        input_pixels_sha256.update(
            np.ascontiguousarray(roi_source.read_slice(0, roi_source.total_rois)).view(
                np.uint8
            )
        )
        channel_count = len(mask_labels)
        _write_fake_raw_outputs(
            run_group,
            row_count=roi_source.total_rois,
            channel_count=channel_count,
            height=roi_source.roi_shape[0],
            width=roi_source.roi_shape[1],
            mask_probs_dtype=mask_probs_dtype,
            write_masks_roi=write_masks_roi,
            validation_accumulators=validation_accumulators,
        )
        return 0.25

    monkeypatch.setattr(mod, "_load_checkpoint", _fake_load_checkpoint)
    monkeypatch.setattr(
        mod, "_write_subject_mask_outputs", _fake_write_subject_mask_outputs
    )
    monkeypatch.setattr(mod.zarr, "open", lambda *_args, **_kwargs: fake_root)
    monkeypatch.setattr(
        mod.CropImageSource,
        "open",
        lambda root, **_kwargs: _FakeCropSource(root["crop_runs"]["crop_geometry"]),
    )
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda **_kwargs: {
            "platform": {
                "hostname": "test-host",
                "system": "Linux",
                "release": "6.0",
                "python_version": "3.11",
                "machine": "x86_64",
            },
            "environment": {"name": "test-env"},
        },
    )
    monkeypatch.setattr(
        mod,
        "get_git_info",
        lambda: {
            "commit_hash": "abc123",
            "short_hash": "abc123",
            "branch": "main",
            "is_dirty": False,
            "remote_url": "git@example.com:palette.git",
        },
    )

    mod.main(
        [
            str(zarr_path),
            "--checkpoint",
            str(checkpoint_path),
            "--crop-run",
            "crop_geometry",
            "--output-parent",
            "subject_mask_shard_runs",
            "--write-masks-roi",
        ]
    )

    subject_parent = fake_root["subject_mask_shard_runs"]
    run_group = subject_parent[subject_parent.group_keys()[0]]

    assert seen["mask_labels"] == (
        "subject_body",
        "eye_left",
        "eye_right",
        "swim_bladder",
    )
    assert run_group.attrs["label_schema_id"] == "subject_v1_lr"
    assert list(run_group.attrs["mask_labels"]) == [
        "subject_body",
        "eye_left",
        "eye_right",
        "swim_bladder",
    ]
    assert run_group["mask_probs_roi"].shape == (2, 4, 4, 4)
    assert run_group["masks_roi"].shape == (2, 4, 4, 4)
    for label in ("subject_body", "eye_left", "eye_right", "swim_bladder"):
        provenance = run_group["components"][label]["provenance"].attrs
        assert provenance["source_label_schema_id"] == "subject_v1_lr"
        assert provenance["source_channels"] == [label]


def test_infer_unet_subject_masks_can_resolve_checkpoint_from_registry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    checkpoint_path = tmp_path / "registry_selected.pt"
    checkpoint_bytes = b"registry checkpoint"
    checkpoint_path.write_bytes(checkpoint_bytes)
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()

    fake_root = _build_fake_root()
    seen: dict[str, object] = {}

    class _FakeCropSource:
        def __init__(self, crop_group) -> None:
            self.crop_group = crop_group
            self.crop_run_name = "crop_geometry"
            self.total_rois = 2
            self.roi_shape = (4, 4)
            self.roi_array = None
            self.storage_mode = "geometry_only"
            self.roi_read_mode = "live"
            self.roi_cache_policy = "never"
            self.roi_cache_used = False
            self.roi_cache_key = None
            self.roi_cache_path = None
            self.roi_live_acceleration_requested = "cpu"
            self.roi_live_acceleration_effective = "cpu"
            self.roi_live_acceleration_fallback_reason = None
            self.roi_live_gpu_chunk_frames = 32
            self.frame_source_kind = "raw_video/images_full"
            self.frame_source_path = None

        def read_slice(self, start: int, stop: int) -> np.ndarray:
            return np.zeros((stop - start, 4, 4), dtype=np.uint8)

        def close(self) -> None:
            return None

    def _fake_resolve_registry_checkpoint(args):
        seen["coverage_class"] = args.model_coverage_class
        return checkpoint_path, {
            "mode": "registry",
            "task": "subject_masks",
            "registry_path": str(tmp_path / "registry.sqlite"),
            "resolved_at_utc": "2026-04-25T00:00:00+00:00",
            "selected": {
                "run_id": "subject_masks_union_all_components_v001",
                "set_id": "subject_mask_set",
                "model_path": str(checkpoint_path),
                "coverage_class": "dense_all_components",
                "component_coverage_key": "body+eyes+swim_bladder",
                "best_metric_name": "best_val_dice",
                "best_metric_value": 0.947,
                "model_sha256": checkpoint_sha,
            },
            "candidates": [],
            "parameters": {"coverage_class": args.model_coverage_class},
        }

    def _fake_load_checkpoint(path: Path, _device) -> tuple[object, dict[str, object]]:
        seen["checkpoint_path"] = path
        return object(), {
            "label_schema_id": "subject_v1_union",
            "mask_labels": ["subject_body", "eyes_union", "swim_bladder"],
            "best_val_dice": 0.947,
        }

    def _fake_write_subject_mask_outputs(
        run_group,
        model,
        roi_source,
        *,
        batch_size,
        device,
        mask_labels,
        mask_probs_chunk_rois,
        mask_probs_shard_rois,
        mask_probs_dtype,
        write_masks_roi,
        async_output,
        output_queue_size,
        model_input_transform,
        show_progress,
        console,
        timing_profiler,
        input_pixels_sha256,
        validation_accumulators,
    ) -> float:
        del (
            model,
            batch_size,
            device,
            mask_probs_chunk_rois,
            mask_probs_shard_rois,
            output_queue_size,
            model_input_transform,
            console,
            timing_profiler,
        )
        seen["async_output"] = async_output
        seen["show_progress"] = show_progress
        input_pixels_sha256.update(
            np.ascontiguousarray(roi_source.read_slice(0, roi_source.total_rois)).view(
                np.uint8
            )
        )
        channel_count = len(mask_labels)
        _write_fake_raw_outputs(
            run_group,
            row_count=roi_source.total_rois,
            channel_count=channel_count,
            height=roi_source.roi_shape[0],
            width=roi_source.roi_shape[1],
            mask_probs_dtype=mask_probs_dtype,
            write_masks_roi=write_masks_roi,
            validation_accumulators=validation_accumulators,
        )
        return 0.25

    monkeypatch.setattr(
        mod, "_resolve_registry_checkpoint", _fake_resolve_registry_checkpoint
    )
    monkeypatch.setattr(mod, "_load_checkpoint", _fake_load_checkpoint)
    monkeypatch.setattr(
        mod, "_write_subject_mask_outputs", _fake_write_subject_mask_outputs
    )
    monkeypatch.setattr(mod.zarr, "open", lambda *_args, **_kwargs: fake_root)
    monkeypatch.setattr(
        mod.CropImageSource,
        "open",
        lambda root, **_kwargs: _FakeCropSource(root["crop_runs"]["crop_geometry"]),
    )
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda **_kwargs: {
            "platform": {
                "hostname": "test-host",
                "system": "Linux",
                "release": "6.0",
                "python_version": "3.11",
                "machine": "x86_64",
            },
            "environment": {"name": "test-env"},
        },
    )
    monkeypatch.setattr(
        mod,
        "get_git_info",
        lambda: {
            "commit_hash": "abc123",
            "short_hash": "abc123",
            "branch": "main",
            "is_dirty": False,
            "remote_url": "git@example.com:palette.git",
        },
    )

    mod.main(
        [
            str(zarr_path),
            "--resolve-model-from-registry",
            "--model-coverage-class",
            "dense_all_components",
            "--crop-run",
            "crop_geometry",
            "--output-parent",
            "subject_mask_shard_runs",
            "--async-output",
            "--no-progress",
        ]
    )

    subject_parent = fake_root["subject_mask_shard_runs"]
    run_group = subject_parent[subject_parent.group_keys()[0]]

    assert seen["coverage_class"] == "dense_all_components"
    assert seen["checkpoint_path"] == checkpoint_path
    assert seen["async_output"] is True
    assert seen["show_progress"] is False
    assert run_group.attrs["source_checkpoint"] == str(checkpoint_path)
    assert run_group.attrs["model_resolution_mode"] == "registry"
    assert run_group.attrs["model_resolution_task"] == "subject_masks"
    assert (
        run_group.attrs["model_resolution_selected_run_id"]
        == "subject_masks_union_all_components_v001"
    )
    assert (
        run_group.attrs["model_resolution_selected_coverage_class"]
        == "dense_all_components"
    )
    assert (
        run_group.attrs["model_resolution_selected_component_coverage_key"]
        == "body+eyes+swim_bladder"
    )
    assert run_group.attrs["model_resolution_selected_metric_value"] == 0.947
    provenance = run_group.attrs["provenance"]
    assert provenance["inputs"]["model_resolution"]["selected"]["run_id"] == (
        "subject_masks_union_all_components_v001"
    )
    run_provenance = run_group.attrs[RUN_PROVENANCE_ATTR]
    artifacts = run_provenance["input_artifacts"]
    assert artifacts == [
        {
            "role": "subject_mask_unet_checkpoint",
            "path": str(checkpoint_path),
            "fingerprint_scheme": "content_v1",
            "sha256": checkpoint_sha,
            "size_bytes": len(checkpoint_bytes),
            "mtime_ns": checkpoint_path.stat().st_mtime_ns,
            "source": "direct_scientific_commit_rehash",
        }
    ]
