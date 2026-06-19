from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes
import zarr

from fisheye.refinement import subject_body_mask_qc as mod
from fisheye.shared.detect_reason_codec import read_reason_labels


def _body_mask(height: int = 32, width: int = 32) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[5:27, 14:18] = 1
    return mask


def _build_refined_root() -> zarr.Group:
    root = zarr.group()
    parent = root.create_group("refined_subject_masks_runs")
    parent.attrs["latest"] = "refined_001"
    run = parent.create_group("refined_001")
    run.attrs["mask_labels"] = ["subject_body"]
    run.attrs["label_schema_id"] = "subject_v1_body"
    run.create_array("available_channels", data=np.asarray([True], dtype=bool), overwrite=True)
    masks = np.zeros((3, 1, 32, 32), dtype=np.uint8)
    masks[0, 0] = _body_mask()
    masks[1, 0] = _body_mask()
    masks[1, 0, 15:17, 4:28] = 1
    masks[1, 0, 21:23, 4:28] = 1
    masks[2, 0] = 0
    masks[2, 0, 5:9, 5:9] = 1
    masks[2, 0, 20:24, 20:24] = 1
    run.create_array("masks_roi", data=masks, chunks=(1, 1, 32, 32), overwrite=True)
    return root


def test_compute_subject_body_mask_qc_flags_missing_fragmented_and_branched_masks() -> None:
    masks = np.zeros((4, 32, 32), dtype=np.uint8)
    masks[0] = _body_mask()
    masks[1, 4:8, 4:8] = 1
    masks[1, 20:24, 20:24] = 1
    masks[2] = _body_mask()
    masks[2, 15:17, 4:28] = 1
    masks[2, 21:23, 4:28] = 1

    batch = mod.compute_subject_body_mask_qc(masks)

    assert batch.reason_labels[0] == "ok"
    assert bool(batch.severe_qc_failure[0]) is False
    assert bool(batch.requires_review[0]) is False

    assert "fragmented_subject_body_mask" in str(batch.reason_labels[1])
    assert bool(batch.severe_qc_failure[1]) is True
    assert int(batch.component_count[1]) == 2

    assert "branched_body_mask" in str(batch.reason_labels[2])
    assert "excess_body_skeleton_endpoints" in str(batch.reason_labels[2])
    assert bool(batch.severe_qc_failure[2]) is True
    assert int(batch.skeleton_endpoint_count[2]) > 2
    assert int(batch.skeleton_branchpoint_count[2]) > 0

    assert "missing_subject_body_mask" in str(batch.reason_labels[3])
    assert bool(batch.severe_qc_failure[3]) is True


def test_hole_metrics_matches_scipy_filled_area_fraction() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:28, 4:28] = True
    mask[10:14, 10:16] = False
    mask[19:23, 18:24] = False

    filled = binary_fill_holes(mask)
    expected_area = float(np.count_nonzero(filled & ~mask))
    expected_fraction = float(expected_area / max(1.0, float(np.count_nonzero(filled))))

    area, fraction = mod._hole_metrics(mask, float(np.count_nonzero(mask)))

    assert area == expected_area
    np.testing.assert_allclose(fraction, expected_fraction)


def test_write_subject_body_mask_qc_group_persists_component_qc() -> None:
    root = _build_refined_root()

    summary = mod.write_subject_body_mask_qc_group(root, refined_run="refined_001", chunk_size=2)

    assert summary["status"] == "updated"
    assert summary["row_count"] == 3
    assert summary["severe_qc_failure_count"] == 2
    run = root["refined_subject_masks_runs"]["refined_001"]
    qc = run["components"]["subject_body"]["qc"]
    assert qc.attrs["schema_id"] == mod.SUBJECT_BODY_MASK_QC_SCHEMA_ID
    assert qc.attrs["method"] == mod.SUBJECT_BODY_MASK_QC_METHOD
    assert run.attrs["subject_body_mask_qc_status"] == "computed"
    assert qc["severe_qc_failure"][:].tolist() == [False, True, True]
    assert qc["requires_review"][:].tolist() == [False, True, True]
    assert qc["component_count"][:].tolist() == [1, 1, 2]
    reasons = read_reason_labels(qc)
    assert reasons is not None
    assert reasons[0] == "ok"
    assert "branched_body_mask" in str(reasons[1])
    assert "fragmented_subject_body_mask" in str(reasons[2])
