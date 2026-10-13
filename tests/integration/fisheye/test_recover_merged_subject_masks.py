"""Publication/retry checks through the real recovery adapter and publisher."""

import numpy as np
import pytest
import zarr
from zarr.core.dtype import VariableLengthUTF8

from fisheye.training.recover_merged_subject_masks import recover_subject_masks
from fisheye.training.recover_merged_training_recording import (
    SOURCE_ONLY_SCHEMA_ID,
    _sha256_array,
)


@pytest.fixture
def sources(tmp_path):
    archive, merged = tmp_path / "recording.zarr", tmp_path / "merged.zarr"
    root = zarr.open_group(str(archive), mode="w", use_consolidated=False)
    images = np.zeros((2, 512, 512), np.uint8)
    boxes = np.array([[0.5, 0.5, 0.2, 0.2]] * 2, np.float32)
    frames = np.array([4, 8], np.int64)
    head = np.array([[[256, 220], [248, 200], [264, 200]]] * 2, np.float32)
    arrays = {
        "pose/roi_images": images,
        "pose/keypoints_roi": head,
        "pose/source_merged_row": np.array([20, 21]),
        "pose/source_frame_idx": frames,
        "pose/detect_local_row": np.array([0, 1], np.int32),
        "pose/crop_bbox_norm_coords": boxes,
        "detect/images_ds": np.zeros((2, 640, 640), np.uint8),
        "detect/bbox_norm_coords": boxes,
        "detect/source_merged_row": np.array([40, 41]),
        "detect/source_frame_idx": frames,
        "detect/detect_only_local_row": np.array([], np.int32),
    }
    for path, data in arrays.items():
        family, name = path.split("/")
        root.require_group(f"recovered_sources/{family}").create_array(name, data=data)
    root.attrs.update(
        {
            "schema_id": SOURCE_ONLY_SCHEMA_ID,
            "zarr_purpose": "training",
            "training_artifact_status": "complete",
            "stage_selector_eligible": False,
            "recording_id": "rec",
            "recovery_mode": "source_only",
            "pose_source_row_count": 2,
            "detect_source_row_count": 2,
            "detect_only_row_count": 0,
            "source_pose": {"run_id": "pose", "dataset_id": "rec:pose"},
            "source_detect": {"run_id": "detect", "dataset_id": "rec"},
            "review_snapshot": {
                "species": "Danio rerio",
                "pose_review_state": "approved",
                "pose_review_intended_use": "training",
                "pose_review_method": "manual",
            },
            "array_sha256": {
                f"recovered_sources/{k}": _sha256_array(v) for k, v in arrays.items()
            },
        }
    )
    root.require_group("crop_runs").attrs.update(
        {"latest": "prior", "latest_complete": "prior", "authoritative_run": "prior"}
    )
    zarr.consolidate_metadata(str(archive))
    source = zarr.open_group(str(merged), mode="w", use_consolidated=False)
    crop = source.require_group("crop_runs/merged_subject_masks")
    for name, data in {
        "roi_images": images,
        "crop_bbox_norm_coords": boxes,
        "detection_source": np.zeros(2, np.int8),
    }.items():
        crop.create_array(name, data=data)
    yy, xx = np.mgrid[:512, :512]
    body = ((xx - 256) / 18) ** 2 + ((yy - 260) / 80) ** 2 <= 1
    swim = ((xx - 256) / 10) ** 2 + ((yy - 220) / 14) ** 2 <= 1
    masks = np.repeat(np.stack([body, swim, swim])[None], 2, axis=0).astype(np.uint8)
    group = source.require_group("subject_mask_runs/merged_subject_masks")
    group.create_array("masks_roi", data=masks)
    group.create_array("target_valid_channels", data=np.ones((2, 3), bool))
    group.attrs.update(
        {
            "mask_labels": ["subject_body", "eyes_union", "swim_bladder"],
            "label_schema_id": "subject_v1_union",
        }
    )
    index = source.require_group("source_index")
    ids = index.create_array(
        "source_dataset_id", shape=(1,), dtype=VariableLengthUTF8()
    )
    ids[:] = ["rec:zmask"]
    index.create_array("source_dataset_idx", data=np.zeros(2, np.int32))
    index.create_array("source_frame_idx", data=frames)
    zarr.consolidate_metadata(str(merged))
    return archive, merged


def test_recovery_publication_preserves_selectors_and_refuses_edited_resume(sources):
    archive, merged = sources
    before = zarr.open_group(str(archive), mode="r", use_consolidated=True).attrs[
        "array_sha256"
    ]
    result = recover_subject_masks(
        archive=archive, merged=merged, version="v1", apply=True
    )
    root = zarr.open_group(str(archive), mode="r+", use_consolidated=False)
    assert root.attrs["array_sha256"] == before
    assert root["crop_runs"].attrs["latest"] == "prior"
    assert root["crop_runs"].attrs["authoritative_run"] == "prior"
    assert result["row_count"] == 2
    assert result["pose_schema"] == "head_tail11_fins_v2"
    assert root[result["paths"]["pose_edit"]]["keypoints_roi"].shape == (2, 19, 2)
    assert "tail19" in result["tasks"][0]["task_id"]
    assert not root[result["paths"]["pose_edit"]]["training_eligible"][:].any()
    resumed = recover_subject_masks(
        archive=archive, merged=merged, version="v1", apply=True, resume=True
    )
    assert resumed["publications"] == []
    root[result["paths"]["pose_edit"]]["keypoints_roi"][0, 0] = [1, 2]
    with pytest.raises(ValueError, match="edited, or conflicting"):
        recover_subject_masks(
            archive=archive, merged=merged, version="v1", apply=True, resume=True
        )


def test_conflicting_pixels_leave_no_new_runs(sources):
    archive, merged = sources
    root = zarr.open_group(str(merged), mode="a", use_consolidated=False)
    root["crop_runs/merged_subject_masks/roi_images"][0, 0, 0] = 1
    with pytest.raises(ValueError, match="pixels"):
        recover_subject_masks(archive=archive, merged=merged, version="v1", apply=True)
    assert not (archive / "subject_mask_runs").exists()


def test_new_version_can_use_corrected_dense_masks_without_overwriting_old_seed(
    sources,
):
    archive, merged = sources
    first = recover_subject_masks(
        archive=archive,
        merged=merged,
        version="v1",
        apply=True,
        pose_schema="head_tail11_fins_v1",
    )
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    mask_path = first["paths"]["mask_edit"]
    root[mask_path]["masks_roi"][0, 0, 1, 1] = 1
    second = recover_subject_masks(
        archive=archive,
        merged=merged,
        version="v2",
        apply=True,
        refined_mask_run=mask_path.split("/")[1],
    )
    refreshed = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    assert refreshed[first["paths"]["mask"]]["masks_roi"][0, 0, 1, 1] == 0
    assert refreshed[second["paths"]["mask"]]["masks_roi"][0, 0, 1, 1] == 1
    assert not refreshed[second["paths"]["seed"]]["tail_valid"][0]
    assert second["source_bindings"]["refined_mask_snapshot"]["run_path"] == mask_path
    assert refreshed[first["paths"]["seed"]]["keypoints_roi"].shape == (2, 18, 2)
    assert refreshed[second["paths"]["seed"]]["keypoints_roi"].shape == (2, 19, 2)
    assert not refreshed[second["paths"]["seed"]]["snout_valid"][0]
    assert refreshed[second["paths"]["seed"]]["snout_valid"][1]


def test_failed_publication_can_resume_without_changing_previous_children(
    sources, monkeypatch
):
    from fisheye.training import recover_merged_subject_masks as producer

    archive, merged = sources
    original = producer.atomic_publish_run_group
    calls = []

    def fail_second(spec, **kwargs):
        calls.append(spec.target_run_path)
        if len(calls) == 2:
            raise OSError("injected copy failure")
        return original(spec, **kwargs)

    monkeypatch.setattr(producer, "atomic_publish_run_group", fail_second)
    with pytest.raises(OSError, match="copy failure"):
        recover_subject_masks(archive=archive, merged=merged, version="v1", apply=True)
    assert calls[0].exists() and not calls[1].exists()
    monkeypatch.setattr(producer, "atomic_publish_run_group", original)
    result = recover_subject_masks(
        archive=archive, merged=merged, version="v1", apply=True, resume=True
    )
    assert len(result["publications"]) == 4
    assert len(result["tasks"]) >= 1
