from __future__ import annotations

import json
from pathlib import Path
import tarfile

import numpy as np
import zarr

from fisheye.shared.refined_subject_mask_encoded_chunks import (
    ENCODED_MASK_PAYLOAD_NAME,
    ENCODED_PACKAGE_SCHEMA_ID,
    prepare_global_mask_chunk_grid,
)
from fisheye.utils import finalize_subject_mask_clip_package as mod


def test_finalize_subject_mask_clip_package_writes_tar_and_cleans_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_zarr = tmp_path / "source.zarr"
    root = zarr.open_group(str(source_zarr), mode="w")
    root.require_group("crop_runs").create_group("crop_collection")
    root.require_group("subject_mask_shard_runs").create_group("subject_clip")
    root.require_group("refined_keypoints_runs").create_group("refined_keypoints")

    def fake_finalize_subject_mask_run(root, **kwargs):  # noqa: ANN001, ANN003
        refined = root.require_group("refined_subject_masks_runs").create_group(kwargs["refined_run"])
        refined.attrs["method"] = "test_finalizer"
        refined.attrs["mask_labels"] = ["subject_body"]
        refined.create_array("masks_roi", data=np.zeros((1, 1, 2, 2), dtype=np.uint8), overwrite=True)
        return {"status": "ok", "refined_run": kwargs["refined_run"]}

    monkeypatch.setattr(mod, "finalize_subject_mask_run", fake_finalize_subject_mask_run)

    staging_root = tmp_path / "scratch"
    package_path = tmp_path / "nrs" / "refined_clip.tar.gz"
    result = mod.finalize_subject_mask_clip_package(
        source_zarr=source_zarr,
        subject_shard_run="subject_clip",
        target_crop_run="crop_collection",
        refined_run="refined_clip",
        package_path=package_path,
        staging_root=staging_root,
        components=("subject_body",),
    )

    assert result["status"] == "ok"
    assert package_path.is_file()
    assert not staging_root.exists()
    with tarfile.open(package_path, "r:gz") as tar:
        names = set(tar.getnames())
        assert "package.json" in names
        assert "refined_subject_masks_runs/refined_clip/zarr.json" in names
        package_member = tar.extractfile("package.json")
        assert package_member is not None
        package = json.loads(package_member.read().decode("utf-8"))
    assert package["schema_id"] == mod.PACKAGE_SCHEMA_ID
    assert package["run_group_path"] == "refined_subject_masks_runs/refined_clip"


def test_finalize_subject_mask_clip_package_can_emit_encoded_v2(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_zarr = tmp_path / "source.zarr"
    root = zarr.open_group(str(source_zarr), mode="w")
    crop = root.require_group("crop_runs").create_group("crop_collection")
    crop.create_array(
        "source_crop_row_ids",
        data=np.asarray([10, 11], dtype=np.int64),
        chunks=(2,),
        overwrite=True,
    )
    root.require_group("subject_mask_shard_runs").create_group("subject_clip")
    grid_manifest = tmp_path / "global_grid.json"
    prepare_global_mask_chunk_grid(
        zarr_path=source_zarr,
        crop_run="crop_collection",
        output_manifest=grid_manifest,
        mask_labels=("subject_body",),
        mask_height=2,
        mask_width=2,
        dense_mask_row_chunk=2,
    )

    def fake_finalize_subject_mask_run(root, **kwargs):  # noqa: ANN001, ANN003
        refined = root.require_group("refined_subject_masks_runs").create_group(kwargs["refined_run"])
        refined.attrs["method"] = "test_finalizer"
        refined.attrs["mask_labels"] = ["subject_body"]
        refined.create_array(
            "masks_roi",
            data=np.ones((2, 1, 2, 2), dtype=np.uint8),
            chunks=(2, 1, 2, 2),
            overwrite=True,
        )
        refined.create_array(
            "source_crop_row_ids",
            data=np.asarray([10, 11], dtype=np.int64),
            chunks=(2,),
            overwrite=True,
        )
        return {"status": "ok", "refined_run": kwargs["refined_run"]}

    monkeypatch.setattr(mod, "finalize_subject_mask_run", fake_finalize_subject_mask_run)
    package_path = tmp_path / "refined_clip_v2.tar.gz"
    result = mod.finalize_subject_mask_clip_package(
        source_zarr=source_zarr,
        subject_shard_run="subject_clip",
        target_crop_run="crop_collection",
        refined_run="refined_clip",
        package_path=package_path,
        staging_root=tmp_path / "scratch_v2",
        components=("subject_body",),
        global_mask_grid_manifest=grid_manifest,
        encoded_mask_copy_workers=2,
    )

    assert result["schema_id"] == ENCODED_PACKAGE_SCHEMA_ID
    with tarfile.open(package_path, "r:gz") as tar:
        names = set(tar.getnames())
        package_member = tar.extractfile("package.json")
        assert package_member is not None
        package = json.loads(package_member.read().decode("utf-8"))
    assert package["schema_id"] == ENCODED_PACKAGE_SCHEMA_ID
    assert package["encoded_global_masks_roi"]["complete_row_chunk_count"] == 1
    assert f"{ENCODED_MASK_PAYLOAD_NAME}/zarr.json" in names
    assert any(name.startswith(f"{ENCODED_MASK_PAYLOAD_NAME}/c/0/") for name in names)
