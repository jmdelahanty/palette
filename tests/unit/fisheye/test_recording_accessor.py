from __future__ import annotations

import numpy as np
import zarr

from fisheye.shared.mask_store import write_component_rle_mask_store_from_dense
from fisheye.shared.recording import open_recording
from fisheye.shared.run_resolution import RunResolution
from fisheye.shared.zarr_run_completion import mark_run_complete, set_authoritative_run


def _open_tmp_store(path):
    return zarr.open_group(str(path), mode="w")


def _complete_run(root, parent_path: str, run_name: str):
    parent = root
    for part in parent_path.split("/"):
        parent = parent.require_group(part)
    run = parent.require_group(run_name)
    mark_run_complete(run, parent_group=parent, run_name=run_name)
    return run


def test_recording_detections_resolve_authoritative_over_later_smoke(tmp_path) -> None:
    zarr_path = tmp_path / "recording_accessor_detect.zarr"
    root = _open_tmp_store(zarr_path)
    approved = _complete_run(root, "detect_runs", "detect_approved")
    approved.create_array("frame_indices", data=np.asarray([1, 3], dtype=np.int32), overwrite=True)
    approved.create_array(
        "bbox_norm_coords",
        data=np.asarray([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=np.float32),
        overwrite=True,
    )
    set_authoritative_run(root["detect_runs"], "detect_approved", approved_by="jeremy")

    smoke = _complete_run(root, "detect_runs", "detect_smoke")
    smoke.create_array("frame_indices", data=np.asarray([99], dtype=np.int32), overwrite=True)

    rec = open_recording(zarr_path)
    read = rec.detections()

    assert read.run_name == "detect_approved"
    assert read.parent_path == "detect_runs"
    assert read.resolution.mode is RunResolution.AUTHORITATIVE
    assert read.resolution.resolution_source == "authoritative"
    assert read.resolution.run_name == "detect_approved"
    assert read.arrays["frame_indices"].tolist() == [1, 3]


def test_recording_subject_masks_decode_compact_store_without_materializing_dense(tmp_path) -> None:
    zarr_path = tmp_path / "recording_accessor_masks.zarr"
    root = _open_tmp_store(zarr_path)
    approved = _complete_run(root, "refined_subject_masks_runs", "masks_approved")
    approved.attrs["mask_labels"] = ["body", "eyes"]
    dense_masks = np.zeros((2, 2, 5, 6), dtype=np.uint8)
    dense_masks[0, 0, 1:4, 2:5] = 1
    dense_masks[1, 1, 0:2, 0:3] = 1
    dense = approved.create_array("masks_roi", data=dense_masks, overwrite=True)
    write_component_rle_mask_store_from_dense(
        approved,
        dense,
        component_names=("body", "eyes"),
        encode_row_chunk_size=1,
    )
    del approved["masks_roi"]
    approved.attrs["masks_roi_materialized"] = False
    set_authoritative_run(root["refined_subject_masks_runs"], "masks_approved", approved_by="jeremy")

    smoke = _complete_run(root, "refined_subject_masks_runs", "masks_smoke")
    smoke.attrs["mask_labels"] = ["body", "eyes"]
    smoke.create_array("masks_roi", data=np.zeros((1, 2, 5, 6), dtype=np.uint8), overwrite=True)

    rec = open_recording(zarr_path)
    read = rec.subject_masks(rows=0, channels="body")

    assert read.run_name == "masks_approved"
    assert read.mask_encoding == "component_rle_v1"
    assert read.mask_labels == ("body",)
    assert read.masks is not None
    assert read.masks.shape == (1, 1, 5, 6)
    assert int(np.count_nonzero(read.masks)) == 9

    reopened = zarr.open_group(str(zarr_path), mode="r")
    assert "masks_roi" not in reopened["refined_subject_masks_runs"]["masks_approved"]


def test_fisheye_api_exports_accessor_and_verbs() -> None:
    from fisheye.api import DetectRequest, Recording, RunResolution, detect, open_recording as api_open

    assert api_open is open_recording
    assert Recording.__name__ == "Recording"
    assert RunResolution.AUTHORITATIVE.value == "authoritative"
    assert DetectRequest.__name__ == "DetectRequest"
    assert callable(detect)
