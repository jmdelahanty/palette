"""Mask-driven successors preserve saved manual labels and immutable history."""

import numpy as np
import pytest
import zarr

from fisheye.training.mask_tail_apply_refresh import regenerate_training_tail_version
from fisheye.training.recovered_mask_review_payload import (
    array_hashes,
    build_review_payload,
)
from fisheye.tune import keypoint_review_backend as editor
from fisheye.shared.detect_reason_codec import read_reason_labels, write_reason_columns


@pytest.fixture(params=[False, True], ids=["recovered", "native"])
def reviewed_archive(tmp_path, request):
    path = tmp_path / "training.zarr"
    root = zarr.open_group(str(path), mode="w", use_consolidated=False)
    root.attrs.update(
        zarr_purpose="training",
        recording_id="rec",
        stage_selector_eligible=False,
        schema_id="palette.training.merged_pose_detect_recovery_source.v1",
    )
    yy, xx = np.mgrid[:128, :128]
    body = ((xx - 64) / 13) ** 2 + ((yy - 60) / 48) ** 2 <= 1
    swim = ((xx - 64) / 7) ** 2 + ((yy - 43) / 9) ** 2 <= 1
    eyes = (((xx - 58) ** 2 + (yy - 26) ** 2) <= 9) | (
        ((xx - 70) ** 2 + (yy - 26) ** 2) <= 9
    )
    masks = np.repeat(np.stack([body, eyes, swim])[None], 2, axis=0).astype(np.uint8)
    arrays = {
        "roi_images": np.repeat((body * 100).astype(np.uint8)[None], 2, axis=0),
        "masks_roi": masks,
        "head_keypoints_roi": np.array(
            [[[64, 43], [58, 26], [70, 26]]] * 2, np.float32
        ),
        "frame_indices": np.array([15, 31]),
        "source_merged_row": np.array([200, 300]),
        "source_pose_local_row": np.array([3, 5]),
        "source_bbox_norm_coords": np.zeros((2, 4)),
        "detection_source": np.zeros(2, np.int8),
        "target_valid_channels": np.ones((2, 3), bool),
    }
    binding = {
        "mask_run": "original",
        "recording_id": "rec",
        "mask_run_attrs": {"label_schema_id": "subject_v1_union"},
    }
    if request.param:
        arrays["source_training_crop_row_ids"] = np.array([8, 2])
        arrays["source_keypoints_roi"] = np.concatenate(
            [arrays["head_keypoints_roi"], np.array([[[64, 13]]] * 2, np.float32)],
            axis=1,
        )
        from fisheye.training.recover_merged_training_recording import _sha256_array

        binding.update(
            source_kind="native_reviewed_training_masks_v1",
            keypoint_run="old_pose",
            source_array_sha256={
                "keypoints": _sha256_array(arrays["source_keypoints_roi"])
            },
            source_keypoint_labels=[
                "swim_bladder",
                "eye_left",
                "eye_right",
                "snout_tip",
            ],
        )
    result = build_review_payload(
        root,
        arrays,
        ("subject_body", "eyes_union", "swim_bladder"),
        binding,
        version="v1",
        native=request.param,
    )
    session = editor.resolve_review_session(
        str(path),
        refined_run=result["paths"]["pose_edit"].split("/")[1],
        include_all=True,
    )
    for row in range(2):
        points = np.asarray(session.kp_roi_arr[row]).copy()
        points[14:18] = [[74, 43], [80, 46], [54, 43], [48, 46]]
        points[4, 0] += 1  # A deliberately manual tail point must survive.
        editor.save_roi_correction(session, position=row, points=points)
    mask = root[result["paths"]["mask_edit"]]
    mask.attrs["edit_revision"] = 1
    mask["masks_roi"][1, 0, 1, 1] = 1  # Regression: fragmented body.
    return path, root, result


def refresh(source, **overrides):
    path, root, result = source
    args = dict(
        archive=path,
        refined_mask_run=result["paths"]["mask_edit"].split("/")[1],
        refined_keypoint_run=result["paths"]["pose_edit"].split("/")[1],
        apply_id="apply-one",
        expected_mask_revision=1,
    )
    return regenerate_training_tail_version(**{**args, **overrides})


def test_new_version_keeps_manual_points_failures_and_source_history(reviewed_archive):
    path, root, old_result = reviewed_archive
    old = root[old_result["paths"]["pose_edit"]]
    reasons = read_reason_labels(old)
    reasons[0] = str(reasons[0]) + "|operator_checked_snout"
    write_reason_columns(old, reasons, chunk_size=2, overwrite=True)
    before = {name: array_hashes(root[p]) for name, p in old_result["paths"].items()}
    manual = np.asarray(old["keypoint_manual_edit"][:])
    points = np.asarray(old["keypoints_roi"][:])
    selectors = {
        family: dict(root[family].attrs)
        for family in (
            "crop_runs",
            "keypoints_runs",
            "refined_keypoints_runs",
            "subject_mask_runs",
            "refined_subject_masks_runs",
        )
    }
    result = refresh(reviewed_archive)
    current = zarr.open_group(str(path), mode="r", use_consolidated=False)
    new = current[result["paths"]["pose_edit"]]
    seed = current[result["paths"]["seed"]]
    np.testing.assert_array_equal(new["keypoints_roi"][:][manual], points[manual])
    np.testing.assert_array_equal(new["keypoint_manual_edit"][:], manual)
    np.testing.assert_array_equal(
        new["keypoint_origin"][:][manual], np.full(manual.sum(), 3)
    )
    np.testing.assert_array_equal(
        new["keypoints_roi"][:][~manual], seed["keypoints_roi"][:][~manual]
    )
    assert new["training_eligible"][:].tolist() == [True, False]
    assert new["tail_valid"][:].tolist() == [True, False]
    assert result["failures"][0]["roi_idx"] == 1
    assert result["source_mask_edit_revision"] == 1
    assert result["manual_point_count"] == int(manual.sum())
    reasons = read_reason_labels(new)
    assert "operator_checked_snout" in reasons[0]
    assert "needs_manual_fins" not in reasons[0]
    assert "needs_manual_fins" not in reasons[1]
    np.testing.assert_array_equal(new["keypoint_confidences"][:][manual], 1.0)
    for name, p in old_result["paths"].items():
        assert array_hashes(current[p]) == before[name]
    for family, attrs in selectors.items():
        assert dict(current[family].attrs) == attrs
    reopened = editor.resolve_review_session(
        str(path),
        refined_run=result["paths"]["pose_edit"].split("/")[1],
        include_all=True,
    )
    assert reopened.recovered_roi_only


def test_same_source_retry_reuses_version_and_preserves_successor_edits(
    reviewed_archive,
):
    first = refresh(reviewed_archive)
    path, _, _ = reviewed_archive
    session = editor.resolve_review_session(
        str(path),
        refined_run=first["paths"]["pose_edit"].split("/")[1],
        include_all=True,
    )
    points = np.asarray(session.kp_roi_arr[0]).copy()
    points[14, 0] += 1
    editor.save_roi_correction(session, position=0, points=points)
    second = refresh(reviewed_archive)
    assert first["paths"] == second["paths"]
    np.testing.assert_array_equal(session.kp_roi_arr[0], points)


def test_retry_does_not_rebind_apply_to_changed_original_labels(reviewed_archive):
    refresh(reviewed_archive)
    path, _, original = reviewed_archive
    session = editor.resolve_review_session(
        str(path),
        refined_run=original["paths"]["pose_edit"].split("/")[1],
        include_all=True,
    )
    points = np.asarray(session.kp_roi_arr[0]).copy()
    points[14, 0] += 1
    editor.save_roi_correction(session, position=0, points=points)
    with pytest.raises(ValueError, match="Source changed since this Apply"):
        refresh(reviewed_archive)


@pytest.mark.parametrize(
    "fault",
    ["revision", "row_identity", "manual_origin", "unknown_origin", "recipe", "schema"],
)
def test_refuses_wrong_or_tampered_sources_before_publication(reviewed_archive, fault):
    _, root, result = reviewed_archive
    mask = root[result["paths"]["mask_edit"]]
    pose = root[result["paths"]["pose_edit"]]
    if fault == "revision":
        mask.attrs["edit_revision"] = 2
    elif fault == "row_identity":
        mask["frame_indices"][:] = [31, 15]
    elif fault == "manual_origin":
        pose["keypoint_origin"][0, 14] = 2
    elif fault == "unknown_origin":
        pose["keypoint_origin"][0, 5] = 99
    elif fault == "recipe":
        pose.attrs["derivation_recipe"] = {"recipe": "unrecognized"}
    else:
        pose.attrs["keypoint_labels"] = list(reversed(pose.attrs["keypoint_labels"]))
    with pytest.raises((ValueError, RuntimeError)):
        refresh(reviewed_archive)
    assert len(list(root["refined_keypoints_runs"].group_keys())) == 1


def test_partial_publication_retries_without_replacing_first_child(
    reviewed_archive, monkeypatch
):
    from fisheye.training import recover_merged_subject_masks as producer

    original = producer.atomic_publish_run_group
    calls = []

    def interrupted(spec, **kwargs):
        calls.append(spec.target_run_path)
        if len(calls) == 2:
            raise OSError("interrupted publication")
        return original(spec, **kwargs)

    monkeypatch.setattr(producer, "atomic_publish_run_group", interrupted)
    with pytest.raises(OSError, match="interrupted publication"):
        refresh(reviewed_archive)
    first_metadata = (calls[0] / "zarr.json").read_bytes()
    monkeypatch.setattr(producer, "atomic_publish_run_group", original)
    _, root, old = reviewed_archive
    pose = root[old["paths"]["pose_edit"]]
    saved = np.asarray(pose["keypoints_roi"][0]).copy()
    changed = saved.copy()
    changed[14, 0] += 1
    pose["keypoints_roi"][0] = changed
    with pytest.raises(ValueError, match="Source changed since this Apply"):
        refresh(reviewed_archive)
    pose["keypoints_roi"][0] = saved
    result = refresh(reviewed_archive)
    assert result["status"] == "generated"
    assert (calls[0] / "zarr.json").read_bytes() == first_metadata


def test_source_change_during_publication_refuses_stale_snapshot(
    reviewed_archive, monkeypatch
):
    from fisheye.training import mask_tail_apply_refresh as refresh_mod

    original = refresh_mod.publish_review_payload
    _, root, result = reviewed_archive

    def changed(*args, **kwargs):
        root[result["paths"]["mask_edit"]]["masks_roi"][0, 0, 1, 2] = 1
        return original(*args, **kwargs)

    monkeypatch.setattr(refresh_mod, "publish_review_payload", changed)
    with pytest.raises(ValueError, match="source changed"):
        refresh(reviewed_archive)
    assert len(list(root["refined_keypoints_runs"].group_keys())) == 1


def test_manual_clear_remains_cleared_and_ineligible(reviewed_archive):
    path, root, old = reviewed_archive
    session = editor.resolve_review_session(
        str(path), refined_run=old["paths"]["pose_edit"].split("/")[1], include_all=True
    )
    editor.mark_no_keypoints(session, position=0)
    result = refresh(reviewed_archive)
    current = zarr.open_group(str(path), mode="r", use_consolidated=False)
    pose = current[result["paths"]["pose_edit"]]
    assert np.isnan(pose["keypoints_roi"][0]).all()
    assert not bool(pose["training_eligible"][0])


def browser_context(source, tmp_path):
    from types import SimpleNamespace
    from fisheye.labeling.assignment_store import LabelingStore

    path, root, result = source
    store = LabelingStore(tmp_path / "review.sqlite")
    store.assign_recording(
        recording_id="rec", assignee_user="reviewer", assigned_by="reviewer"
    )
    pose_name = result["paths"]["pose_edit"].split("/")[1]
    mask_name = result["paths"]["mask_edit"].split("/")[1]
    store.upsert_task(
        recording_id="rec",
        task_id="original-pose",
        workflow_kind="keypoints",
        run_name=pose_name,
        scope={"zarr_path": str(path), "refined_run": pose_name},
    )
    store.upsert_task(
        recording_id="rec",
        task_id="original-mask",
        workflow_kind="subject_mask_component",
        run_name=mask_name,
        component_name="subject_body",
        scope={"zarr_path": str(path), "refined_run": mask_name},
    )
    runtime = SimpleNamespace(
        root=root,
        zarr_path=str(path),
        task_id="original-mask",
        recording_id="rec",
        user="reviewer",
        component_name="subject_body",
        refined=SimpleNamespace(
            group=root[result["paths"]["mask_edit"]], run_name=mask_name
        ),
    )
    return store, runtime


def test_browser_offers_successor_without_resetting_source_or_opened_successor(
    reviewed_archive, tmp_path
):
    from fisheye.labeling.web_mask_tail_refresh import (
        refresh_training_tail_after_mask_apply,
    )

    store, runtime = browser_context(reviewed_archive, tmp_path)
    original = store.get_task("original-pose")
    lease = store.create_session(task_id="original-pose", user="reviewer")
    result = refresh_training_tail_after_mask_apply(
        store=store, runtime=runtime, apply_id="mask-apply", expected_mask_revision=1
    )
    assert result["tail_refresh_status"] == "complete"
    assert store.get_task("original-pose") == original
    assert store.get_session(lease.session_id)["closed_at_utc"] is None
    task = next(
        task
        for task in result["tail_refresh_tasks"]
        if task["workflow_kind"] == "keypoints"
    )
    store.update_task_state(task_id=task["task_id"], state="complete")
    retry = refresh_training_tail_after_mask_apply(
        store=store, runtime=runtime, apply_id="mask-apply", expected_mask_revision=1
    )
    assert retry["tail_refresh_version"] == result["tail_refresh_version"]
    assert store.get_task(task["task_id"])["state"] == "complete"
    store.close()


def test_pending_paired_pose_checkpoint_blocks_successor_publication(
    reviewed_archive, tmp_path
):
    from fisheye.labeling.web_mask_tail_refresh import (
        refresh_training_tail_after_mask_apply,
    )

    store, runtime = browser_context(reviewed_archive, tmp_path)
    lease = store.create_session(task_id="original-pose", user="reviewer")
    store.upsert_session_checkpoint(
        session_id=lease.session_id,
        task_id="original-pose",
        recording_id="rec",
        user="reviewer",
        workflow_kind="keypoints",
        target_run_path="refined_keypoints_runs/source",
        target_edit_revision=0,
        source_rowset_path="crop_runs/source",
        roi_idx=0,
        component_name="keypoints",
        payload={"test_checkpoint": True},
        metadata={},
    )
    with pytest.raises(RuntimeError, match="Apply the paired keypoint"):
        refresh_training_tail_after_mask_apply(
            store=store,
            runtime=runtime,
            apply_id="mask-apply",
            expected_mask_revision=1,
        )
    assert len(list(runtime.root["refined_keypoints_runs"].group_keys())) == 1
    assert store.count_unapplied_session_checkpoints(task_id="original-pose") == 1
    store.close()
