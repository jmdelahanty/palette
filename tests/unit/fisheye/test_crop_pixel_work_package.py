from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from fisheye.shared.crop_image_source import CropImageSource
from fisheye.shared.crop_pixel_work_package import (
    CropPixelWorkPackageError,
    build_crop_pixel_work_package_from_source,
    cleanup_unreferenced_crop_pixel_work_package_generations,
    open_crop_pixel_work_package,
)
from fisheye.shared.roi_pixel_contract import crop_run_pixel_contract
from fisheye.shared.row_lineage import copy_selected_crop_row_lineage_arrays
from fisheye.shared.row_source_signature import build_row_source_signatures


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
