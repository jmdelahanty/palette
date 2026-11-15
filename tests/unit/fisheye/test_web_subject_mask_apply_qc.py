from __future__ import annotations

import numpy as np
import pytest

from fisheye.labeling.web_subject_mask_apply_qc import refresh_subject_mask_apply_qc_locked
from fisheye.refinement import finalize_subject_masks as finalizer
from fisheye.shared.detect_reason_codec import read_reason_labels, update_reason_rows
from fisheye.shared.refined_subject_eye_geometry import write_refined_subject_eye_geometry
from fisheye.tune import refined_subject_mask_review as review_mod
from tests.unit.fisheye.test_refined_subject_mask_review import _build_subject_review_root


def test_full_browser_qc_refresh_preserves_masks_manual_tags_and_compact_stale(tmp_path) -> None:
    zarr_path = tmp_path / "qc.zarr"
    root = _build_subject_review_root(zarr_path=zarr_path)
    source, refined = review_mod.prepare_refined_subject_run(
        root, subject_run="subject_masks_001", refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )
    run = refined.group
    run.attrs["edit_revision"] = 1
    update_reason_rows(
        run["components/subject_body"], np.asarray([0]),
        np.asarray(["operator_note|needs_review_metric_old"], dtype=object),
    )
    masks = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    edited = masks.copy()
    edited[0, 0] = 1
    review_mod._apply_refined_subject_roi_rows(
        source=source, refined=refined, roi_indices=[0], edited_masks_batch=edited[0:1],
        component_names=("subject_body",),
    )
    before = np.asarray(run["masks_roi"][:], dtype=np.uint8).copy()
    with review_mod._refined_subject_write_lock(zarr_path, refined_run=refined.run_name):
        result = refresh_subject_mask_apply_qc_locked(
            root=review_mod.open_zarr_root(zarr_path, mode="a"),
            refined_run=refined.run_name, expected_edit_revision=1,
        )
    fresh = review_mod.open_zarr_root(zarr_path, mode="r", use_consolidated=False)
    refreshed = fresh["refined_subject_masks_runs/refined_subject_masks_001"]
    assert result["qc_status"] == "complete"
    np.testing.assert_array_equal(refreshed["masks_roi"][:], before)
    assert bool(refreshed.attrs["metrics_stale"]) is False
    assert bool(refreshed.attrs["contours_stale"]) is False
    assert bool(refreshed.attrs["derived_mask_caches_stale"]) is True
    assert "operator_note" in str(read_reason_labels(refreshed["components/subject_body"])[0])
    assert "needs_review_metric_old" not in str(read_reason_labels(refreshed["components/subject_body"])[0])


def test_browser_qc_refuses_unsupported_metric_contract_before_write(tmp_path) -> None:
    zarr_path = tmp_path / "unsupported.zarr"
    root = _build_subject_review_root(zarr_path=zarr_path)
    _source, refined = review_mod.prepare_refined_subject_run(
        root, subject_run="subject_masks_001", refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )
    refined.group.attrs["edit_revision"] = 1
    refined.group.attrs["component_metric_level"] = "cheap"
    before = np.asarray(refined.group["metrics/area_px"][:]).copy()
    with review_mod._refined_subject_write_lock(zarr_path, refined_run=refined.run_name):
        with pytest.raises(RuntimeError, match="declares 'cheap'"):
            refresh_subject_mask_apply_qc_locked(
                root=review_mod.open_zarr_root(zarr_path, mode="a"),
                refined_run=refined.run_name, expected_edit_revision=1,
            )
    np.testing.assert_array_equal(refined.group["metrics/area_px"][:], before)


def test_browser_qc_refuses_declared_component_qc_policy_before_write(tmp_path) -> None:
    zarr_path = tmp_path / "component-policy.zarr"
    root = _build_subject_review_root(zarr_path=zarr_path)
    _source, refined = review_mod.prepare_refined_subject_run(
        root, subject_run="subject_masks_001", refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )
    refined.group.attrs["edit_revision"] = 1
    refined.group["components/subject_body/metrics"].attrs["qc_policy"] = {"id": "other-policy"}
    before = np.asarray(refined.group["metrics/area_px"][:]).copy()
    with review_mod._refined_subject_write_lock(zarr_path, refined_run=refined.run_name):
        with pytest.raises(RuntimeError, match="metrics qc_policy"):
            refresh_subject_mask_apply_qc_locked(
                root=review_mod.open_zarr_root(zarr_path, mode="a"),
                refined_run=refined.run_name, expected_edit_revision=1,
            )
    np.testing.assert_array_equal(refined.group["metrics/area_px"][:], before)


def test_browser_qc_halfway_failure_stays_stale_then_retries_at_same_revision(tmp_path, monkeypatch) -> None:
    zarr_path = tmp_path / "retry.zarr"
    root = _build_subject_review_root(zarr_path=zarr_path)
    source, refined = review_mod.prepare_refined_subject_run(
        root, subject_run="subject_masks_001", refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )
    refined.group.attrs["edit_revision"] = 1
    edited = np.asarray(refined.group["masks_roi"][0:1], dtype=np.uint8)
    edited[0, 0] = 1
    review_mod._apply_refined_subject_roi_rows(
        source=source, refined=refined, roi_indices=[0], edited_masks_batch=edited,
        component_names=("subject_body",),
    )
    pixels = np.asarray(refined.group["masks_roi"][:], dtype=np.uint8).copy()
    original_write = finalizer._write_mask_local_metrics_chunk
    calls = 0

    def fail_second_metric_write(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected halfway QC failure")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(finalizer, "_write_mask_local_metrics_chunk", fail_second_metric_write)
    with review_mod._refined_subject_write_lock(zarr_path, refined_run=refined.run_name):
        with pytest.raises(OSError, match="halfway QC"):
            refresh_subject_mask_apply_qc_locked(
                root=review_mod.open_zarr_root(zarr_path, mode="a"),
                refined_run=refined.run_name, expected_edit_revision=1,
            )
    monkeypatch.setattr(finalizer, "_write_mask_local_metrics_chunk", original_write)
    fresh = review_mod.open_zarr_root(zarr_path, mode="r", use_consolidated=False)
    run = fresh["refined_subject_masks_runs/refined_subject_masks_001"]
    assert bool(run.attrs["metrics_stale"]) is True
    assert bool(run.attrs["contours_stale"]) is True
    assert run.attrs.get("browser_apply_qc_policy") is None
    np.testing.assert_array_equal(run["masks_roi"][:], pixels)
    with review_mod._refined_subject_write_lock(zarr_path, refined_run=refined.run_name):
        result = refresh_subject_mask_apply_qc_locked(
            root=review_mod.open_zarr_root(zarr_path, mode="a"),
            refined_run=refined.run_name, expected_edit_revision=1,
        )
    assert result["qc_status"] == "complete"
    assert run.attrs["edit_revision"] == 1
    np.testing.assert_array_equal(run["masks_roi"][:], pixels)
    calls = 0
    monkeypatch.setattr(finalizer, "_write_mask_local_metrics_chunk", fail_second_metric_write)
    with review_mod._refined_subject_write_lock(zarr_path, refined_run=refined.run_name):
        with pytest.raises(OSError, match="halfway QC"):
            refresh_subject_mask_apply_qc_locked(
                root=review_mod.open_zarr_root(zarr_path, mode="a"),
                refined_run=refined.run_name, expected_edit_revision=1,
            )
    monkeypatch.setattr(finalizer, "_write_mask_local_metrics_chunk", original_write)
    again = review_mod.open_zarr_root(zarr_path, mode="r", use_consolidated=False)
    retry_run = again["refined_subject_masks_runs/refined_subject_masks_001"]
    assert bool(retry_run.attrs["metrics_stale"]) is True
    assert bool(retry_run.attrs["contours_stale"]) is True
    assert retry_run.attrs.get("browser_apply_qc_policy") is None


def test_browser_qc_refreshes_eye_ellipses_contours_and_pair_relation(tmp_path) -> None:
    zarr_path = tmp_path / "eyes.zarr"
    root = _build_subject_review_root(zarr_path=zarr_path)
    source, refined = review_mod.prepare_refined_subject_run(
        root, subject_run="subject_masks_001", refined_run="refined_subject_masks_001",
        components=("subject_body", "eye_left", "eye_right", "swim_bladder"),
    )
    run = refined.group
    write_refined_subject_eye_geometry(run)
    run.attrs["edit_revision"] = 1
    edited = np.asarray(run["masks_roi"][0:1], dtype=np.uint8)
    edited[0, 1] = 0
    edited[0, 2] = 0
    edited[0, 1, 1:5, 1:5] = 1
    edited[0, 2, 3:7, 3:7] = 1
    review_mod._apply_refined_subject_roi_rows(
        source=source, refined=refined, roi_indices=[0], edited_masks_batch=edited,
        component_names=("eye_left", "eye_right"),
    )
    pixels = np.asarray(run["masks_roi"][:], dtype=np.uint8).copy()
    with review_mod._refined_subject_write_lock(zarr_path, refined_run=refined.run_name):
        refresh_subject_mask_apply_qc_locked(
            root=review_mod.open_zarr_root(zarr_path, mode="a"),
            refined_run=refined.run_name, expected_edit_revision=1,
        )
    fresh = review_mod.open_zarr_root(zarr_path, mode="r", use_consolidated=False)
    refreshed = fresh["refined_subject_masks_runs/refined_subject_masks_001"]
    np.testing.assert_array_equal(refreshed["masks_roi"][:], pixels)
    assert bool(refreshed.attrs["contours_stale"]) is False
    assert bool(refreshed["relations/eye_pair/metrics/separation_valid"][0]) is True
    assert float(refreshed["relations/eye_pair/metrics/separation_px"][0]) > 0
    for name in ("eye_left", "eye_right"):
        component = refreshed["components"][name]
        assert bool(component["geometry/ellipse_success"][0]) is True
        assert int(component["contours/len"][0]) > 0
        assert component["contours"].attrs["min_points"] == 1


def test_browser_qc_refuses_incompatible_eye_contour_before_write(tmp_path) -> None:
    zarr_path = tmp_path / "eye-method.zarr"
    root = _build_subject_review_root(zarr_path=zarr_path)
    _source, refined = review_mod.prepare_refined_subject_run(
        root, subject_run="subject_masks_001", refined_run="refined_subject_masks_001",
        components=("subject_body", "eye_left", "eye_right", "swim_bladder"),
    )
    write_refined_subject_eye_geometry(refined.group)
    refined.group.attrs["edit_revision"] = 1
    refined.group["components/eye_left/contours"].attrs["min_points"] = 2
    before = np.asarray(refined.group["metrics/area_px"][:]).copy()
    with review_mod._refined_subject_write_lock(zarr_path, refined_run=refined.run_name):
        with pytest.raises(RuntimeError, match="eye_left contour min_points"):
            refresh_subject_mask_apply_qc_locked(
                root=review_mod.open_zarr_root(zarr_path, mode="a"),
                refined_run=refined.run_name, expected_edit_revision=1,
            )
    np.testing.assert_array_equal(refined.group["metrics/area_px"][:], before)
