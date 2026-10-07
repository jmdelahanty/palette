"""Apply QC recomputes only QC chunks that changed, with the same result as a full refresh."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import zarr

from fisheye.labeling import web_subject_mask_apply_qc as qc_mod
from fisheye.labeling.web_subject_mask_apply_qc import refresh_subject_mask_apply_qc_locked
from fisheye.tune import refined_subject_mask_review as review_mod

RUN = "refined_subject_masks_001"
ROWS, SIZE = 80, 24  # three 32-row QC chunks
COMPONENTS = ("subject_body", "eye_left", "eye_right", "swim_bladder")
_VOLATILE = ("_at_utc", "timing", "duration", "refreshed_at")


def _masks(rows=ROWS):
    yy, xx = np.mgrid[:SIZE, :SIZE]
    masks = np.zeros((rows, 4, SIZE, SIZE), dtype=np.uint8)
    for row in range(rows):
        dx = row % 5 - 2
        masks[row, 0] = (((xx - 12 - dx) / (5 + row % 3)) ** 2 + ((yy - 12) / 9) ** 2) <= 1
        masks[row, 1] = ((xx - 9 - dx) ** 2 + (yy - 6) ** 2) <= 2 + row % 2
        masks[row, 2] = ((xx - 15 - dx) ** 2 + (yy - 6) ** 2) <= 2 + (row + 1) % 2
        masks[row, 3] = (((xx - 12 - dx) / 2) ** 2 + ((yy - 13) / 3) ** 2) <= 1
    return masks


def _build(path):
    """A subject-mask review archive with ROWS rows (shape of the shared 2-row test builder)."""

    root = zarr.open_group(str(path), mode="w")
    frames = np.arange(10, 10 + ROWS, dtype=np.int32)
    crop = root.create_group("crop_runs")
    crop.attrs["latest"] = "crop_001"
    crop = crop.create_group("crop_001")
    crop.attrs.update(crop_storage_mode="geometry_only", crop_signature={"signature_version": 2, "crop_revision": 4},
                      crop_revision=4, detect_review_status_ref="refined_detect_runs/refined_detect_001/review_status")
    crop.create_array("roi_images", data=np.full((ROWS, SIZE, SIZE), 60, dtype=np.uint8))
    crop.create_array("frame_indices", data=frames)
    crop.create_array("roi_coordinates_full", data=np.zeros((ROWS, 2), dtype=np.int32))
    parent = root.create_group("subject_mask_runs")
    parent.attrs["latest"] = "subject_masks_001"
    subject = parent.create_group("subject_masks_001")
    subject.attrs.update(
        source_crop_run="crop_001", source_crop_storage_mode="geometry_only",
        source_crop_signature="{'signature_version': 2, 'crop_revision': 4}", source_crop_revision=4,
        source_detect_review_status_ref="refined_detect_runs/refined_detect_001/review_status",
        method="subject_mask_threshold_lr_v1", mask_labels=list(COMPONENTS), label_schema_id="subject_v1_lr",
        source_keypoint_group="refined_keypoints_runs", source_keypoints_run="refined_kp_001",
    )
    subject.create_array("detection_source", data=np.zeros((ROWS,), dtype=np.int8))
    subject.create_array("frame_indices", data=frames)
    subject.create_array("detection_indices", data=np.arange(ROWS, dtype=np.int32))
    subject.create_array("frame_counts", data=np.ones((ROWS,), dtype=np.int32))
    subject.create_array("source_crop_row_ids", data=np.arange(ROWS, dtype=np.int64))
    subject.create_array("available_channels", data=np.ones((4,), dtype=np.bool_))
    masks = _masks()
    subject.create_array("masks_roi", data=masks)
    subject.create_group("metrics").create_array("mask_present", data=masks.reshape(ROWS, 4, -1).any(axis=2))
    source, refined = review_mod.prepare_refined_subject_run(
        root, subject_run="subject_masks_001", refined_run=RUN, components=COMPONENTS,
    )
    return root, source, refined


def _qc(path, revision):
    with review_mod._refined_subject_write_lock(path, refined_run=RUN):
        return refresh_subject_mask_apply_qc_locked(
            root=review_mod.open_zarr_root(path, mode="a"), refined_run=RUN, expected_edit_revision=revision,
        )


def _edit(path, rows, revision):
    """Change the body and left-eye pixels of ``rows`` and bump the revision, as an Apply does."""

    run = zarr.open_group(str(path), mode="a", use_consolidated=False)[f"refined_subject_masks_runs/{RUN}"]
    for row in rows:
        edited = np.asarray(run["masks_roi"][row], dtype=np.uint8)
        edited[0, 2:5, 2:8] = 1      # body grows a blob
        edited[1] = 0
        edited[1, 4:7, 8:11] = 1     # left eye moves
        run["masks_roi"][row] = edited
    run.attrs["edit_revision"] = revision


def _stable(value):
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items() if not any(m in k for m in _VOLATILE)}
    if isinstance(value, list):
        return [_stable(v) for v in value]
    return value


def _snapshot(path):
    run = zarr.open_group(str(path), mode="r", use_consolidated=False)[f"refined_subject_masks_runs/{RUN}"]
    arrays, attrs = {}, {}

    def visit(group, prefix=""):
        attrs[prefix] = _stable({k: v for k, v in group.attrs.items() if k != "component_metric_qc_chunk_timings"})
        for name, child in group.members():
            key = f"{prefix}/{name}"
            if isinstance(child, zarr.Group):
                visit(child, key)
            else:
                arrays[key] = (np.asarray(child[...]), tuple(child.chunks))
                attrs[key] = _stable(dict(child.attrs))

    visit(run)
    return arrays, attrs


@pytest.fixture
def verified_pair(tmp_path, monkeypatch):
    monkeypatch.setenv(qc_mod.STAGING_DIR_ENV, str(tmp_path))
    scoped = tmp_path / "scoped.zarr"
    root, _source, refined = _build(scoped)
    refined.group.attrs["edit_revision"] = 1
    first = _qc(scoped, 1)
    assert first["qc_scope"] == "full" and first["qc_chunk_count"] == 3
    full = tmp_path / "full.zarr"
    shutil.copytree(scoped, full)
    return scoped, full


def _force_full(path):
    run = zarr.open_group(str(path), mode="a", use_consolidated=False)[f"refined_subject_masks_runs/{RUN}"]
    policy = dict(run.attrs["browser_apply_qc_policy"])
    policy.pop(qc_mod.CHUNK_DIGESTS_KEY)
    run.attrs["browser_apply_qc_policy"] = policy


@pytest.mark.parametrize("rows", [[3], [3, 40], [0, 31, 32, 79], []])
def test_chunk_scoped_qc_equals_a_full_refresh(verified_pair, rows):
    scoped, full = verified_pair
    for path in (scoped, full):
        _edit(path, rows, 2)
    _force_full(full)

    scoped_result, full_result = _qc(scoped, 2), _qc(full, 2)
    touched = {row // qc_mod.QC_ROW_CHUNK for row in rows}
    assert scoped_result["qc_scope"] == "chunks" and scoped_result["qc_chunks_refreshed"] == len(touched)
    assert full_result["qc_scope"] == "full"
    scoped_arrays, scoped_attrs = _snapshot(scoped)
    full_arrays, full_attrs = _snapshot(full)
    assert scoped_arrays.keys() == full_arrays.keys()
    for key, (data, chunks) in scoped_arrays.items():
        np.testing.assert_array_equal(data, full_arrays[key][0], err_msg=key)
        assert chunks == full_arrays[key][1], key
    assert scoped_attrs == full_attrs


def test_a_changed_derived_value_in_an_untouched_chunk_is_recomputed(verified_pair):
    scoped, full = verified_pair
    run = zarr.open_group(str(scoped), mode="a", use_consolidated=False)[f"refined_subject_masks_runs/{RUN}"]
    run["components/subject_body/metrics/solidity"][70] = 0.123  # chunk 2, no pixel edit there
    for path in (scoped, full):
        _edit(path, [3], 2)
    _force_full(full)
    result = _qc(scoped, 2)
    assert result["qc_scope"] == "chunks" and result["qc_chunks_refreshed"] == 2
    _qc(full, 2)
    np.testing.assert_array_equal(
        _snapshot(scoped)[0]["/components/subject_body/metrics/solidity"][0],
        _snapshot(full)[0]["/components/subject_body/metrics/solidity"][0],
    )


def test_a_scoped_refresh_that_would_change_an_untouched_row_falls_back_to_full(verified_pair, monkeypatch):
    scoped, _full = verified_pair
    _edit(scoped, [3], 2)
    real = qc_mod._derived_row_snapshot
    calls = []

    def drifting(run, row_count):
        snapshot = real(run, row_count)
        calls.append(1)
        if len(calls) == 2:  # the after-refresh snapshot: pretend row 70 (untouched chunk) changed
            snapshot["components/subject_body/area_px"] = snapshot["components/subject_body/area_px"].copy()
            snapshot["components/subject_body/area_px"][70] += 1
        return snapshot

    monkeypatch.setattr(qc_mod, "_derived_row_snapshot", drifting)
    result = _qc(scoped, 2)
    assert result["qc_scope"] == "full" and "area_px" in result["qc_scope_fallback_reason"]


def test_a_rerun_at_the_same_revision_and_a_first_qc_refresh_everything(verified_pair):
    scoped, _full = verified_pair
    assert _qc(scoped, 1)["qc_scope"] == "full"
