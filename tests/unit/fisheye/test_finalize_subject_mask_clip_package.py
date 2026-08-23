from __future__ import annotations

import json
from pathlib import Path
import tarfile

import numpy as np
import pytest
import zarr

from fisheye.shared.refined_subject_mask_encoded_chunks import (
    ENCODED_MASK_PAYLOAD_NAME,
    ENCODED_PACKAGE_SCHEMA_ID,
    prepare_global_mask_chunk_grid,
)
from fisheye.utils import finalize_subject_mask_clip_package as mod


def test_default_staging_root_is_unique_per_lsf_array_element(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("LSB_JOBID", "123")
    monkeypatch.setenv("LSB_JOBINDEX", "7")

    assert mod._default_staging_root() == (  # noqa: SLF001
        tmp_path / "palette_refined_subject_mask_clip_package_123_7"
    )


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
        refined = root.require_group("refined_subject_masks_runs").create_group(
            kwargs["refined_run"]
        )
        refined.attrs["method"] = "test_finalizer"
        refined.attrs["mask_labels"] = ["subject_body"]
        refined.create_array(
            "masks_roi", data=np.zeros((1, 1, 2, 2), dtype=np.uint8), overwrite=True
        )
        return {"status": "ok", "refined_run": kwargs["refined_run"]}

    monkeypatch.setattr(
        mod, "finalize_subject_mask_run", fake_finalize_subject_mask_run
    )

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
        refined = root.require_group("refined_subject_masks_runs").create_group(
            kwargs["refined_run"]
        )
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

    monkeypatch.setattr(
        mod, "finalize_subject_mask_run", fake_finalize_subject_mask_run
    )
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


def test_finalize_clip_package_embeds_receipt_composed_publication_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_zarr = tmp_path / "source_evidence.zarr"
    root = zarr.open_group(str(source_zarr), mode="w", zarr_format=3)
    root.require_group("crop_runs").create_group("crop_collection")
    root.require_group("subject_mask_shard_runs").create_group("subject_clip")

    def fake_finalize(root, **kwargs):  # noqa: ANN001, ANN003
        assert kwargs["mask_rle_validation_mode"] == "full"
        refined = root.require_group("refined_subject_masks_runs").create_group(
            kwargs["refined_run"]
        )
        refined.attrs["mask_labels"] = ["subject_body"]
        refined.attrs["stage_selector_eligible"] = False
        refined.create_array(
            "masks_roi",
            data=np.zeros((1, 1, 2, 2), dtype=np.uint8),
            overwrite=True,
        )
        return {"status": "ok"}

    def fake_evidence(**kwargs):  # noqa: ANN003
        destination = kwargs["destination"]
        (destination / "raw_final_layout_unit").mkdir(parents=True)
        (destination / "refined_final_layout_unit").mkdir()
        (destination / "quality_partition").mkdir()
        (destination / "sampled_contour_receipt.json").write_text(
            "{}\n", encoding="utf-8"
        )
        return {
            "schema_id": "palette.subject_mask.clip_publication_evidence",
            "schema_version": 1,
            "producer_commit": "c" * 40,
            "work_unit_id": "unit_0",
            "work_unit_index": 0,
            "source_clip_id": "clip_0",
            "source_clip_index": 0,
            "global_frame_interval": {"start_frame": 0, "stop_frame": 10},
            "global_row_interval": {"start_row": 0, "stop_row": 1},
        }

    monkeypatch.setattr(mod, "finalize_subject_mask_run", fake_finalize)
    monkeypatch.setattr(mod, "_refined_worker_proof", lambda *_args: {"ok": True})
    monkeypatch.setattr(mod, "_build_publication_evidence", fake_evidence)
    package_path = tmp_path / "refined_evidence.tar.gz"

    result = mod.finalize_subject_mask_clip_package(
        source_zarr=source_zarr,
        subject_shard_run="subject_clip",
        target_crop_run="crop_collection",
        refined_run="refined_clip",
        package_path=package_path,
        staging_root=tmp_path / "scratch_evidence",
        require_production_proof=True,
        publication_evidence_producer_commit="c" * 40,
        work_unit_id="unit_0",
        work_unit_index=0,
        source_clip_id="clip_0",
        source_clip_index=0,
        global_frame_start=0,
        global_frame_stop=10,
    )

    assert result["publication_evidence"]["work_unit_id"] == "unit_0"
    with tarfile.open(package_path, "r:gz") as tar:
        names = set(tar.getnames())
        manifest_file = tar.extractfile("package.json")
        assert manifest_file is not None
        manifest = json.loads(manifest_file.read().decode("utf-8"))
    assert "publication_evidence/raw_final_layout_unit" in names
    assert "publication_evidence/refined_final_layout_unit" in names
    assert "publication_evidence/quality_partition" in names
    assert "publication_evidence/sampled_contour_receipt.json" in names
    assert manifest["publication_evidence"]["producer_commit"] == "c" * 40
    assert manifest["requested_mask_validation_mode"] == "auto"
    assert manifest["effective_mask_validation_mode"] == "full"
    assert result["requested_mask_validation_mode"] == "auto"
    assert result["effective_mask_validation_mode"] == "full"


def test_publication_evidence_rejects_inexact_compact_validation_before_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = False

    def unexpected_stage(**_kwargs):  # noqa: ANN003
        nonlocal staged
        staged = True
        raise AssertionError("staging must not start for an invalid publication plan")

    monkeypatch.setattr(mod, "_stage_zarr_with_local_refined_parent", unexpected_stage)

    with pytest.raises(ValueError, match="requires --mask-rle-validation-mode full"):
        mod.finalize_subject_mask_clip_package(
            source_zarr=tmp_path / "source.zarr",
            subject_shard_run="subject_clip",
            target_crop_run="crop_collection",
            refined_run="refined_clip",
            package_path=tmp_path / "refined_clip.tar.gz",
            mask_storage="dense_and_bitpacked",
            mask_rle_validation_mode="invariants",
            require_production_proof=True,
            publication_evidence_producer_commit="c" * 40,
            work_unit_id="unit_0",
            work_unit_index=0,
            source_clip_id="clip_0",
            source_clip_index=0,
            global_frame_start=0,
            global_frame_stop=10,
        )

    assert staged is False


def test_publication_evidence_requires_sampled_contours_before_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = False

    def unexpected_stage(**_kwargs):  # noqa: ANN003
        nonlocal staged
        staged = True
        raise AssertionError("staging must not start for an invalid publication plan")

    monkeypatch.setattr(mod, "_stage_zarr_with_local_refined_parent", unexpected_stage)

    with pytest.raises(ValueError, match="requires sampled component contours"):
        mod.finalize_subject_mask_clip_package(
            source_zarr=tmp_path / "source.zarr",
            subject_shard_run="subject_clip",
            target_crop_run="crop_collection",
            refined_run="refined_clip",
            package_path=tmp_path / "refined_clip.tar.gz",
            mask_storage="dense_and_bitpacked",
            mask_rle_validation_mode="full",
            write_sampled_component_contours=False,
            require_production_proof=True,
            publication_evidence_producer_commit="c" * 40,
            work_unit_id="unit_0",
            work_unit_index=0,
            source_clip_id="clip_0",
            source_clip_index=0,
            global_frame_start=0,
            global_frame_stop=10,
        )

    assert staged is False


def test_publication_evidence_binds_real_worker_receipts_and_values(
    tmp_path: Path,
) -> None:
    from tests.unit.fisheye.test_subject_mask_recording_bundle_publication import (
        _draft,
        _install_worker_sampled_contours,
    )

    draft = _draft(
        tmp_path,
        raw_parent="subject_mask_shard_runs",
        raw_slices={"raw_clip_0": slice(0, 2), "raw_clip_1": slice(2, 4)},
        refined_slices={
            "refined_clip_0": slice(0, 2),
            "refined_clip_1": slice(2, 4),
        },
    )
    _install_worker_sampled_contours(
        draft, refined_runs=("refined_clip_0", "refined_clip_1")
    )
    root = zarr.open_group(str(draft), mode="r", use_consolidated=False)

    evidence = mod._build_publication_evidence(  # noqa: SLF001
        root=root,
        staged_zarr=draft,
        raw_run=root["subject_mask_shard_runs/raw_clip_0"],
        refined_run=root["refined_subject_masks_runs/refined_clip_0"],
        crop_run=root["crop_runs/crop_001"],
        destination=tmp_path / "evidence",
        producer_commit="c" * 40,
        work_unit_id="unit_0",
        work_unit_index=0,
        source_clip_id="clip_0",
        source_clip_index=0,
        global_frame_start=0,
        global_frame_stop=1,
        quality_compute_workers=1,
    )

    assert evidence["global_row_interval"] == {"start_row": 0, "stop_row": 2}
    assert (tmp_path / "evidence/raw_final_layout_unit/receipt.json").is_file()
    assert (tmp_path / "evidence/refined_final_layout_unit/receipt.json").is_file()
    assert (tmp_path / "evidence/sampled_contour_receipt.json").is_file()
    assert (tmp_path / "evidence/quality_partition/receipt.json").is_file()
