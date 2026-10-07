"""Apply QC on a node-local copy of the run gives the same archive as in-place QC."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import zarr

from fisheye.labeling import web_subject_mask_apply_qc as qc_mod
from fisheye.labeling.web_subject_mask_apply_qc import refresh_subject_mask_apply_qc_locked
from fisheye.shared.zarr import local_group_staging as staging
from fisheye.tune import refined_subject_mask_review as review_mod
from tests.unit.fisheye.test_refined_subject_mask_review import _build_subject_review_root

RUN = "refined_subject_masks_001"
_VOLATILE = ("_at_utc", "timing", "duration")


def _edited_archive(path):
    root = _build_subject_review_root(zarr_path=path)
    source, refined = review_mod.prepare_refined_subject_run(
        root, subject_run="subject_masks_001", refined_run=RUN, components=("subject_body", "swim_bladder"),
    )
    refined.group.attrs["edit_revision"] = 1
    edited = np.asarray(refined.group["masks_roi"][0:1], dtype=np.uint8)
    edited[0, 0, 2:6, 2:6] = 1
    review_mod._apply_refined_subject_roi_rows(
        source=source, refined=refined, roi_indices=[0], edited_masks_batch=edited, component_names=("subject_body",),
    )
    return path


def _qc(path):
    with review_mod._refined_subject_write_lock(path, refined_run=RUN):
        return refresh_subject_mask_apply_qc_locked(
            root=review_mod.open_zarr_root(path, mode="a"), refined_run=RUN, expected_edit_revision=1,
        )


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
        attrs[prefix] = _stable(dict(group.attrs))
        for name, child in group.members():
            key = f"{prefix}/{name}"
            if isinstance(child, zarr.Group):
                visit(child, key)
            else:
                arrays[key] = np.asarray(child[...])
                attrs[key] = _stable(dict(child.attrs))

    visit(run)
    return arrays, attrs


def test_staged_qc_matches_in_place_qc(tmp_path, monkeypatch):
    staged = _edited_archive(tmp_path / "staged.zarr")
    in_place = tmp_path / "in_place.zarr"
    shutil.copytree(staged, in_place)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv(qc_mod.STAGING_DIR_ENV, str(scratch))

    staged_result = _qc(staged)
    monkeypatch.setenv(qc_mod.LOCAL_STAGING_ENV, "0")
    in_place_result = _qc(in_place)

    assert staged_result["qc_execution"] == "local_staging" and staged_result["qc_files_written"] > 0
    assert in_place_result["qc_execution"] == "in_place"
    assert {k: v for k, v in staged_result.items() if not k.startswith("qc_files") and k != "qc_execution"} == {
        k: v for k, v in in_place_result.items() if k != "qc_execution"
    }
    staged_arrays, staged_attrs = _snapshot(staged)
    in_place_arrays, in_place_attrs = _snapshot(in_place)
    assert staged_arrays.keys() == in_place_arrays.keys()
    for key in staged_arrays:
        np.testing.assert_array_equal(staged_arrays[key], in_place_arrays[key], err_msg=key)
    assert staged_attrs == in_place_attrs
    assert list(scratch.iterdir()) == []  # the staged copy is removed


def test_a_staged_copy_whose_pixels_changed_is_not_synced(tmp_path, monkeypatch):
    path = _edited_archive(tmp_path / "a.zarr")
    pixels = np.asarray(zarr.open_group(str(path), mode="r")[f"refined_subject_masks_runs/{RUN}/masks_roi"][:])
    original = qc_mod._refresh_and_verify

    def tamper(root, refined_run, expected_edit_revision):
        result = original(root, refined_run, expected_edit_revision)
        root[f"refined_subject_masks_runs/{RUN}/masks_roi"][0, 0, 0, 0] ^= 1
        return result

    monkeypatch.setattr(qc_mod, "_refresh_and_verify", tamper)
    with pytest.raises(staging.StagedSyncRefused, match="masks_roi"):
        _qc(path)
    run = zarr.open_group(str(path), mode="r", use_consolidated=False)[f"refined_subject_masks_runs/{RUN}"]
    np.testing.assert_array_equal(np.asarray(run["masks_roi"][:]), pixels)
    assert run.attrs["metrics_stale"] is True and run.attrs.get("browser_apply_qc_policy") is None


def test_sync_writes_changes_removes_files_and_writes_group_metadata_last(tmp_path, monkeypatch):
    archive = tmp_path / "x.zarr"
    group = archive / "g"
    (group / "a").mkdir(parents=True)
    for path, text in ((archive / "zarr.json", "root"), (group / "zarr.json", "attrs-v1"),
                       (group / "a" / "keep", "same"), (group / "a" / "old", "gone"), (group / "a" / "edit", "v1")):
        path.write_text(text)
    staged = staging.stage_group(archive, "g", tmp_path / "scratch")
    (staged.local_group / "a" / "old").unlink()
    (staged.local_group / "a" / "edit").write_text("v2")
    (staged.local_group / "a" / "new").write_text("added")
    (staged.local_group / "zarr.json").write_text("attrs-v2")
    order = []
    real_write = staging._write_file
    monkeypatch.setattr(staging, "_write_file", lambda source, target: (order.append(target.name), real_write(source, target)))
    report = staging.sync_group_back(staged)
    assert order[-1] == "zarr.json" and sorted(order[:-1]) == ["edit", "new"]
    assert report == {"files_written": 3, "files_removed": 1, "files_total": 4}
    assert (group / "a" / "edit").read_text() == "v2" and not (group / "a" / "old").exists()
    assert (group / "zarr.json").read_text() == "attrs-v2" and (archive / "zarr.json").read_text() == "root"
