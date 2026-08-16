from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from fisheye.shared.crop_image_source import CropImageSource
from fisheye.shared.crop_pixel_work_package import (
    CropPixelWorkPackageError,
    LEGACY_SOURCE_BINDING_PROFILE,
    SIGNED_SOURCE_BINDING_PROFILE,
    STRICT_SOURCE_BINDING_PROFILE,
    build_crop_pixel_work_package_from_source,
    build_crop_pixel_work_package_from_video_window,
    cleanup_unreferenced_crop_pixel_work_package_generations,
    open_crop_pixel_work_package,
)
from fisheye.shared.roi_pixel_contract import (
    SOURCE_PIXELS_RAW_CAMERA_VIDEO,
    crop_run_pixel_contract,
    orange_mono_pynvvc_luma_pixel_contract,
)
from fisheye.shared.row_lineage import copy_selected_crop_row_lineage_arrays
from fisheye.shared.row_source_signature import build_row_source_signatures
from fisheye.segmentation.infer_unet_subject_masks import (
    _validate_package_subject_mask_selection,
    _write_package_subject_mask_crop_placement,
)
from fisheye.utils import build_crop_pixel_work_package as build_package_cli
from tests.unit.fisheye.test_crop_consumer import _strict_crop


def test_build_package_cli_accepts_authenticated_flat_cache_provider(
    tmp_path: Path,
) -> None:
    args = build_package_cli._parser().parse_args(
        [
            str(tmp_path / "analysis.zarr"),
            "--crop-run",
            "crop_hybrid_pixels",
            "--manifest",
            str(tmp_path / "package.json"),
            "--crop-row",
            "0",
            "--roi-cache-manifest",
            str(tmp_path / "authenticated.flat_roi_cache.json"),
            "--roi-cache-expected-archive-path",
            str(tmp_path / "canonical_analysis.zarr"),
        ]
    )

    assert args.crop_run == "crop_hybrid_pixels"
    assert args.roi_cache_manifest == (
        tmp_path / "authenticated.flat_roi_cache.json"
    )
    assert args.roi_cache_expected_archive_path == (
        tmp_path / "canonical_analysis.zarr"
    )


def _crop_root() -> tuple[Any, Any]:
    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    crop = root.require_group("crop_runs").create_group("crop_001")
    pixels = np.arange(5 * 3 * 4, dtype=np.uint8).reshape(5, 3, 4)
    keys = np.asarray([101, 102, 103, 104, 105], dtype=np.uint64)
    frames = np.asarray([0, 1, 1, 3, 4], dtype=np.int64)
    coordinates = np.asarray(
        [[0, 0], [4, 5], [8, 9], [12, 13], [16, 17]], dtype=np.int32
    )
    signatures = build_row_source_signatures(
        stage="crop",
        instance_keys=keys,
        content_components={
            "frame_indices": frames,
            "roi_coordinates_full": coordinates,
        },
        compatibility_context={"roi_size": [3, 4], "pixel_source": "test"},
    )
    crop.create_array("roi_images", data=pixels, chunks=(2, 3, 4))
    crop.create_array("instance_key", data=keys, chunks=(2,))
    crop.create_array("frame_indices", data=frames, chunks=(2,))
    crop.create_array(
        "frame_counts",
        data=np.bincount(frames, minlength=6).astype(np.int32),
        chunks=(6,),
    )
    crop.create_array("detection_indices", data=np.arange(5, dtype=np.int64))
    crop.create_array("roi_coordinates_full", data=coordinates, chunks=(2, 2))
    crop.create_array(
        "source_acquisition_frame_index",
        data=frames,
        chunks=(2,),
    )
    crop.create_array(
        "source_crop_xywh",
        data=np.column_stack(
            (
                coordinates.astype(np.float32),
                np.full((5,), 4, dtype=np.float32),
                np.full((5,), 3, dtype=np.float32),
            )
        ),
        chunks=(2, 4),
    )
    crop.create_array(
        "source_row_signature", data=signatures.signatures, chunks=(2, 32)
    )
    crop.create_array(
        "detection_source", data=np.asarray([0, 1, 0, 1, 0], dtype=np.int8)
    )
    crop.attrs.update(signatures.spec.to_attrs())
    crop.attrs.update(
        {
            "crop_storage_mode": "materialized",
            "roi_size": [3, 4],
            "crop_revision": 2,
            "crop_signature": {"schema": 2, "source": "test"},
            "source_pixel_fingerprint": "source-pixels-sha256",
            "roi_pixel_contract": crop_run_pixel_contract(
                crop_storage_mode="materialized",
                video_source_type="raw_video",
                acceleration="cpu",
            ),
        }
    )
    return root, crop


def _build(
    root: Any,
    tmp_path: Path,
    *,
    rows: list[int] | None = None,
    overwrite: bool = False,
) -> tuple[dict[str, Any], Path]:
    source = CropImageSource.open(root, crop_run="crop_001")
    manifest = tmp_path / "delta-rois.json"
    try:
        result = build_crop_pixel_work_package_from_source(
            source,
            target_crop_rows=[1, 3] if rows is None else rows,
            manifest_path=manifest,
            archive_path=tmp_path / "recording.analysis.zarr",
            batch_rows=1,
            overwrite=overwrite,
        )
    finally:
        source.close()
    return result, manifest


def test_subset_package_roundtrip_and_crop_source_binding(tmp_path: Path) -> None:
    root, crop = _crop_root()
    manifest, manifest_path = _build(root, tmp_path)

    assert manifest["selection"]["row_count"] == 2
    assert manifest["array"]["total_bytes"] == 2 * 3 * 4
    assert manifest["source"]["source_binding_profile"] == (
        LEGACY_SOURCE_BINDING_PROFILE
    )
    package = open_crop_pixel_work_package(
        manifest_path,
        expected_archive_path=tmp_path / "recording.analysis.zarr",
        expected_crop_run="crop_001",
        root=root,
    )
    try:
        np.testing.assert_array_equal(package.crop_row_indices, [1, 3])
        np.testing.assert_array_equal(package.instance_keys, [102, 104])
        np.testing.assert_array_equal(
            package.pixels[:], np.asarray(crop["roi_images"][[1, 3]])
        )
    finally:
        package.close()

    source = CropImageSource.open_work_package(
        root,
        manifest_path=manifest_path,
        zarr_path=tmp_path / "recording.analysis.zarr",
        crop_run="crop_001",
    )
    try:
        assert source.total_rois == 2
        assert source.roi_read_mode == "crop_pixel_work_package"
        assert source.pixel_materialization_id == manifest["package_id"]
        np.testing.assert_array_equal(source.source_crop_row_ids, [1, 3])
        np.testing.assert_array_equal(
            source.read_slice(0, 2), np.asarray(crop["roi_images"][[1, 3]])
        )
    finally:
        source.close()


def test_acquisition_crop_video_pixels_are_a_current_package_source(
    tmp_path: Path,
) -> None:
    root, crop = _crop_root()
    crop.attrs["source_pixels"] = "acquisition_crop_video"
    crop.attrs["roi_pixel_provider"] = "acquisition_crop_video"
    crop.attrs["roi_pixel_contract"] = orange_mono_pynvvc_luma_pixel_contract()

    manifest, manifest_path = _build(root, tmp_path)

    assert manifest["pixel_contract"]["source_pixels"] == ("acquisition_crop_video")
    assert manifest["source"]["source_binding_profile"] == (
        SIGNED_SOURCE_BINDING_PROFILE
    )
    package = open_crop_pixel_work_package(manifest_path, root=root)
    try:
        assert package.pixel_contract["source_pixels"] == ("acquisition_crop_video")
    finally:
        package.close()


def test_strict_full_frame_crop_package_binds_manifest_and_decoder(
    tmp_path: Path,
) -> None:
    crop = _strict_crop(tmp_path)
    archive = Path(str(crop.store.root)).parent.parent
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    pixels = np.arange(4 * 8 * 8, dtype=np.uint8).reshape(4, 8, 8)

    class _StrictSource:
        crop_group = crop
        crop_run_name = "strict_crop"
        total_rois = 4
        roi_shape = (8, 8)
        roi_pixel_contract = orange_mono_pynvvc_luma_pixel_contract(
            source_pixels=SOURCE_PIXELS_RAW_CAMERA_VIDEO,
        )
        frame_indices = np.asarray(crop["frame_indices"][:], dtype=np.int64)
        roi_coordinates_full = np.asarray(
            crop["roi_coordinates_full"][:],
            dtype=np.int32,
        )
        storage_mode = "geometry_only"
        roi_read_mode = "flat_bin_roi_cache"
        frame_source_kind = "source_video_path"

        def __init__(self) -> None:
            self.root = root

        def read_indices(self, rows):
            return pixels[np.asarray(rows, dtype=np.int64)]

        def _build_frame_source_identity(self) -> dict[str, str]:
            return {"profile": "strict-test-full-frame-video"}

    manifest_path = tmp_path / "strict-work-package.json"
    manifest = build_crop_pixel_work_package_from_source(
        _StrictSource(),
        target_crop_rows=[0, 1, 2],
        manifest_path=manifest_path,
        archive_path=archive,
    )

    assert manifest["source"]["source_binding_profile"] == (
        STRICT_SOURCE_BINDING_PROFILE
    )
    assert manifest["source"]["crop_run_reference"]["profile"] == (
        "immutable_run_manifest_v1"
    )
    assert manifest["pixel_contract"]["source_pixels"] == "raw_camera_video"
    package = open_crop_pixel_work_package(manifest_path, root=root)
    try:
        np.testing.assert_array_equal(package.frame_indices, [0, 0, 2])
    finally:
        package.close()


def test_strict_full_frame_crop_package_rejects_acquisition_video_contract(
    tmp_path: Path,
) -> None:
    crop = _strict_crop(tmp_path)

    class _WrongSource:
        crop_group = crop
        crop_run_name = "strict_crop"
        total_rois = 4
        roi_shape = (8, 8)
        roi_pixel_contract = orange_mono_pynvvc_luma_pixel_contract()
        frame_indices = np.asarray(crop["frame_indices"][:], dtype=np.int64)
        roi_coordinates_full = np.asarray(
            crop["roi_coordinates_full"][:],
            dtype=np.int32,
        )
        storage_mode = "geometry_only"
        root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)

    with pytest.raises(CropPixelWorkPackageError, match="authoritative"):
        build_crop_pixel_work_package_from_source(
            _WrongSource(),
            target_crop_rows=[0],
            manifest_path=tmp_path / "wrong.json",
            archive_path=tmp_path / "strict_crop.zarr",
        )


def test_strict_video_window_package_preserves_global_crop_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crop = _strict_crop(tmp_path)
    archive = Path(str(crop.store.root)).parent.parent
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    expected_pixels = np.arange(3 * 8 * 8, dtype=np.uint8).reshape(3, 8, 8)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"authenticated-stream-copy-window")

    class _StrictSource:
        crop_group = crop
        crop_run_name = "strict_crop"
        total_rois = 4
        roi_shape = (8, 8)
        frame_shape = (80, 100)
        frame_indices = np.asarray(crop["frame_indices"][:], dtype=np.int64)
        roi_coordinates_full = np.asarray(
            crop["roi_coordinates_full"][:], dtype=np.int32
        )
        storage_mode = "geometry_only"

        def __init__(self) -> None:
            self.root = root

    def _fake_materialize(**kwargs: Any) -> dict[str, Any]:
        np.testing.assert_array_equal(kwargs["frame_indices"], [0, 0, 2])
        output = Path(kwargs["output_path"])
        output.write_bytes(expected_pixels.tobytes(order="C"))
        return {
            "schema_id": "palette.pynvvc_luma_roi_payload",
            "schema_version": 1,
            "path": str(output),
            "shape": [3, 8, 8],
            "dtype": "uint8",
            "order": "C",
            "total_bytes": int(expected_pixels.nbytes),
            "sha256": hashlib.sha256(expected_pixels.tobytes()).hexdigest(),
            "decode_backend": "pynvvc_luma",
            "duration_seconds": 0.01,
            "timing": {"rows": 3},
        }

    monkeypatch.setattr(
        "fisheye.shared.crop_pixel_work_package.write_pynvvc_luma_roi_payload",
        _fake_materialize,
    )
    clip_index_digest = "1" * 64
    binding = {
        "schema_id": "palette.acquisition_video_frame_window",
        "schema_version": 1,
        "recording_identity": "recording-001",
        "camera_identity": "camera-001",
        "clip_id": "clip_000000",
        "actual_start_frame": 0,
        "end_frame_exclusive": 3,
        "frame_count": 3,
        "clip_index_document_sha256": clip_index_digest,
        "clip_video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
    }
    manifest_path = tmp_path / "window-package.json"
    manifest = build_crop_pixel_work_package_from_video_window(
        _StrictSource(),
        target_crop_rows=[0, 1, 2],
        video_path=video,
        source_video_frame_offset=0,
        source_video_frame_count=3,
        frame_window_binding=binding,
        manifest_path=manifest_path,
        archive_path=archive,
        batch_rows=2,
    )

    assert manifest["materialization_binding"] == binding
    assert manifest["builder"]["decode_backend"] == "pynvvc_luma"
    package = open_crop_pixel_work_package(manifest_path, root=root)
    try:
        np.testing.assert_array_equal(package.crop_row_indices, [0, 1, 2])
        np.testing.assert_array_equal(package.frame_indices, [0, 0, 2])
        np.testing.assert_array_equal(package.pixels[:], expected_pixels)
    finally:
        package.close()


def test_video_window_package_rejects_changed_clip_bytes(tmp_path: Path) -> None:
    crop = _strict_crop(tmp_path)
    archive = Path(str(crop.store.root)).parent.parent
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"changed")

    class _StrictSource:
        crop_group = crop
        crop_run_name = "strict_crop"
        total_rois = 4
        roi_shape = (8, 8)
        frame_shape = (80, 100)
        frame_indices = np.asarray(crop["frame_indices"][:], dtype=np.int64)
        roi_coordinates_full = np.asarray(
            crop["roi_coordinates_full"][:], dtype=np.int32
        )
        storage_mode = "geometry_only"

        def __init__(self) -> None:
            self.root = root

    with pytest.raises(CropPixelWorkPackageError, match="digest differs"):
        build_crop_pixel_work_package_from_video_window(
            _StrictSource(),
            target_crop_rows=[0],
            video_path=video,
            source_video_frame_offset=0,
            source_video_frame_count=1,
            frame_window_binding={
                "schema_id": "palette.acquisition_video_frame_window",
                "schema_version": 1,
                "recording_identity": "recording-001",
                "camera_identity": "camera-001",
                "clip_id": "clip_000000",
                "actual_start_frame": 0,
                "end_frame_exclusive": 1,
                "frame_count": 1,
                "clip_index_document_sha256": "1" * 64,
                "clip_video_sha256": "0" * 64,
            },
            manifest_path=tmp_path / "bad-window.json",
            archive_path=archive,
        )


def test_package_id_is_stable_across_retry_generations(tmp_path: Path) -> None:
    root, _crop = _crop_root()
    first, manifest_path = _build(root, tmp_path)
    second, _ = _build(root, tmp_path, overwrite=True)

    assert second["package_id"] == first["package_id"]
    assert second["array"]["bin_path"] != first["array"]["bin_path"]
    assert second["rows"]["path"] != first["rows"]["path"]
    package = open_crop_pixel_work_package(manifest_path, root=root)
    package.close()


def test_failed_manifest_switch_preserves_previous_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _crop = _crop_root()
    first, manifest_path = _build(root, tmp_path)
    real_replace = os.replace

    def _fail_manifest_switch(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == manifest_path:
            raise OSError("simulated manifest publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", _fail_manifest_switch)
    with pytest.raises(OSError, match="publication failure"):
        _build(root, tmp_path, overwrite=True)

    package = open_crop_pixel_work_package(manifest_path, root=root)
    try:
        assert package.package_id == first["package_id"]
        np.testing.assert_array_equal(package.crop_row_indices, [1, 3])
    finally:
        package.close()

    cleanup_plan = cleanup_unreferenced_crop_pixel_work_package_generations(
        manifest_path
    )
    assert cleanup_plan["unreferenced_file_count"] == 2
    assert cleanup_plan["unreferenced_bytes"] > 0
    cleanup_result = cleanup_unreferenced_crop_pixel_work_package_generations(
        manifest_path, apply=True
    )
    assert cleanup_result["unreferenced_file_count"] == 2
    for path in cleanup_result["unreferenced_files"]:
        assert not Path(path).exists()
    current = open_crop_pixel_work_package(manifest_path, root=root)
    current.close()


def test_payload_and_live_crop_changes_fail_closed(tmp_path: Path) -> None:
    root, crop = _crop_root()
    manifest, manifest_path = _build(root, tmp_path)
    payload_path = tmp_path / manifest["array"]["bin_path"]
    with payload_path.open("r+b") as handle:
        handle.seek(0)
        handle.write(b"\xff")
    with pytest.raises(CropPixelWorkPackageError, match="payload digest"):
        open_crop_pixel_work_package(manifest_path, root=root)

    _build(root, tmp_path, overwrite=True)
    crop["instance_key"][1] = np.uint64(999)
    with pytest.raises(CropPixelWorkPackageError, match="keys no longer match"):
        open_crop_pixel_work_package(manifest_path, root=root)


def test_selected_lineage_is_identical_for_keypoint_and_mask_fanout(
    tmp_path: Path,
) -> None:
    root, crop = _crop_root()
    _manifest, manifest_path = _build(root, tmp_path)
    source = CropImageSource.open_work_package(
        root,
        manifest_path=manifest_path,
        zarr_path=tmp_path / "recording.analysis.zarr",
    )
    try:
        keypoint = root.require_group("keypoint_shard_runs").create_group("kp_delta")
        masks = root.require_group("subject_mask_shard_runs").create_group("sm_delta")
        for target in (keypoint, masks):
            result = copy_selected_crop_row_lineage_arrays(
                target, crop, source.source_crop_row_ids
            )
            assert "instance_key" in result.copied
            assert "source_crop_row_ids" in result.copied

        selected_mask_values = _write_package_subject_mask_crop_placement(
            masks,
            crop,
            source.source_crop_row_ids,
        )
        _validate_package_subject_mask_selection(
            masks,
            crop,
            source.source_crop_row_ids,
            expected=selected_mask_values,
        )

        for name in (
            "instance_key",
            "frame_indices",
            "frame_counts",
            "detection_indices",
            "source_crop_row_ids",
        ):
            np.testing.assert_array_equal(keypoint[name][:], masks[name][:])
        np.testing.assert_array_equal(keypoint["instance_key"][:], [102, 104])
        np.testing.assert_array_equal(keypoint["frame_indices"][:], [1, 3])
        np.testing.assert_array_equal(keypoint["frame_counts"][:], [0, 1, 0, 1, 0, 0])
        np.testing.assert_array_equal(keypoint["source_crop_row_ids"][:], [1, 3])
        assert masks["source_crop_xywh"].dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(
            masks["source_crop_xywh"][:],
            np.asarray(crop["source_crop_xywh"][[1, 3]]),
        )
    finally:
        source.close()


def test_package_subject_mask_selection_rejects_missing_or_wrong_placement(
    tmp_path: Path,
) -> None:
    root, crop = _crop_root()
    _manifest, manifest_path = _build(root, tmp_path)
    source = CropImageSource.open_work_package(
        root,
        manifest_path=manifest_path,
        zarr_path=tmp_path / "recording.analysis.zarr",
    )
    try:
        missing = root.require_group("subject_mask_shard_runs").create_group(
            "missing_placement"
        )
        copy_selected_crop_row_lineage_arrays(
            missing,
            crop,
            source.source_crop_row_ids,
        )
        with pytest.raises(RuntimeError, match="source_crop_xywh"):
            _validate_package_subject_mask_selection(
                missing,
                crop,
                source.source_crop_row_ids,
            )

        wrong = root["subject_mask_shard_runs"].create_group("wrong_placement")
        copy_selected_crop_row_lineage_arrays(
            wrong,
            crop,
            source.source_crop_row_ids,
        )
        wrong.create_array(
            "source_crop_xywh",
            data=np.asarray(crop["source_crop_xywh"][[1, 3]], dtype=np.float64),
        )
        with pytest.raises(RuntimeError, match="dtype"):
            _validate_package_subject_mask_selection(
                wrong,
                crop,
                source.source_crop_row_ids,
            )
    finally:
        source.close()


@pytest.mark.parametrize("rows", [[], [1, 1], [3, 1], [-1], [5]])
def test_invalid_work_package_selections_fail(rows: list[int], tmp_path: Path) -> None:
    root, _crop = _crop_root()
    source = CropImageSource.open(root, crop_run="crop_001")
    try:
        with pytest.raises(CropPixelWorkPackageError):
            build_crop_pixel_work_package_from_source(
                source,
                target_crop_rows=rows,
                manifest_path=tmp_path / "invalid.json",
                archive_path=tmp_path / "recording.analysis.zarr",
            )
    finally:
        source.close()


def test_inference_cli_packages_are_mutually_exclusive_and_shard_only(
    tmp_path: Path,
) -> None:
    from fisheye.detection.detect_keypoints_yolo import (
        _build_arg_parser as keypoint_parser,
        detect_keypoints_yolo,
    )
    from fisheye.segmentation.infer_unet_subject_masks import (
        _build_arg_parser as mask_parser,
        main as mask_main,
    )

    package_path = tmp_path / "package.json"
    keypoint_args = keypoint_parser().parse_args(
        [
            "recording.zarr",
            "--model",
            "model.pt",
            "--output-parent",
            "keypoint_shard_runs",
            "--coordinate-contract-mode",
            "legacy_noncanonical",
            "--roi-work-package-manifest",
            str(package_path),
        ]
    )
    assert keypoint_args.roi_work_package_manifest == package_path
    assert keypoint_args.coordinate_contract_mode == "legacy_noncanonical"
    with pytest.raises(SystemExit):
        keypoint_parser().parse_args(
            [
                "recording.zarr",
                "--model",
                "model.pt",
                "--roi-cache-manifest",
                "cache.json",
                "--roi-work-package-manifest",
                str(package_path),
            ]
        )
    with pytest.raises(ValueError, match="keypoint_shard_runs"):
        detect_keypoints_yolo(
            "missing.zarr",
            "missing.pt",
            roi_work_package_manifest=package_path,
            coordinate_contract_mode="legacy_noncanonical",
        )

    mask_args = mask_parser().parse_args(
        [
            "recording.zarr",
            "model.pt",
            "--output-parent",
            "subject_mask_shard_runs",
            "--roi-work-package-manifest",
            str(package_path),
        ]
    )
    assert mask_args.roi_work_package_manifest == package_path
    with pytest.raises(SystemExit):
        mask_parser().parse_args(
            [
                "recording.zarr",
                "model.pt",
                "--roi-cache-manifest",
                "cache.json",
                "--roi-work-package-manifest",
                str(package_path),
            ]
        )
    with pytest.raises(ValueError, match="subject_mask_shard_runs"):
        mask_main(
            [
                "missing.zarr",
                "missing.pt",
                "--roi-work-package-manifest",
                str(package_path),
            ]
        )


def test_collection_finalizers_reject_incremental_delta_shards() -> None:
    from fisheye.refinement.finalize_subject_masks import (
        _load_subject_mask_shard_sources,
    )
    from fisheye.utils.finalize_keypoint_shards import _resolve_shard

    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    for parent_name, run_name in (
        ("keypoint_shard_runs", "kp_delta"),
        ("subject_mask_shard_runs", "sm_delta"),
    ):
        run = root.require_group(parent_name).create_group(run_name)
        run.attrs["palette_run_completion_status"] = "complete"
        run.attrs["incremental_materialization_role"] = "delta_replacement_rows"

    with pytest.raises(ValueError, match="base-plus-delta compactor"):
        _resolve_shard(root, "kp_delta")
    with pytest.raises(ValueError, match="base-plus-delta compactor"):
        _load_subject_mask_shard_sources(root, ["sm_delta"])
