"""Mask-driven successors preserve saved manual labels and immutable history."""

import numpy as np
import pytest
import zarr

from fisheye.training.mask_tail_apply_refresh import (
    regenerate_training_tail_version,
    upgrade_training_tail_geometry_version,
)
from fisheye.training.recovered_mask_review_payload import (
    array_hashes,
    build_review_payload,
)
from fisheye.tune import keypoint_review_backend as editor
from fisheye.shared.detect_reason_codec import read_reason_labels, write_reason_columns
from fisheye.training.mask_tail_border_acceptance import (
    ATTR as TAIL_ACCEPTANCE_ATTR,
    apply_acceptance_actions,
    body_digest,
    expected_row_identity,
)


@pytest.fixture(autouse=True, params=["v2", "v1"])
def successor_format(request, monkeypatch):
    """Every successor behavior holds for the current (v2) and historical (v1) format."""
    from fisheye.training import mask_tail_apply_refresh as refresh_mod

    monkeypatch.setattr(refresh_mod, "DEFAULT_SUCCESSOR_FORMAT", request.param)
    return request.param


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


def test_explicit_geometry_upgrade_keeps_other_seed_rows_and_manual_clears(reviewed_archive):
    path, root, old_result = reviewed_archive
    old = root[old_result["paths"]["pose_edit"]]
    seed = root[old_result["paths"]["seed"]]
    old_seed = {name: np.asarray(array[:]).copy() for name, array in seed.arrays()}
    old_points = np.asarray(old["keypoints_roi"][:]).copy()
    old_points[1, 5] = np.nan
    old["keypoints_roi"][:] = old_points
    old_manual = np.asarray(old["keypoint_manual_edit"][:]).copy()
    old_manual[1, 5] = True
    old["keypoint_manual_edit"][:] = old_manual
    old_origins = np.asarray(old["keypoint_origin"][:]).copy()
    old_origins[1, 5] = 3
    old["keypoint_origin"][:] = old_origins
    reasons = read_reason_labels(old)
    reasons[1] = str(reasons[1]) + "|tail_derivation_failed:snout_extension_too_long"
    write_reason_columns(old, reasons, chunk_size=2, overwrite=True)
    old_pose = {name: np.asarray(array[:]).copy() for name, array in old.arrays()}
    old_pose_reasons = read_reason_labels(old).copy()
    result = upgrade_training_tail_geometry_version(
        archive=path,
        refined_mask_run=old_result["paths"]["mask_edit"].split("/")[1],
        refined_keypoint_run=old_result["paths"]["pose_edit"].split("/")[1],
        upgrade_id="explicit-upgrade-one",
        expected_mask_revision=1,
        target_rows=[1],
    )
    published = zarr.open_group(str(path), mode="r", use_consolidated=False)
    new_seed = published[result["paths"]["seed"]]
    new_pose = published[result["paths"]["pose_edit"]]
    for name, values in old_seed.items():
        if name in new_seed and new_seed[name].shape == values.shape:
            np.testing.assert_array_equal(new_seed[name][0], values[0])
    for name, values in old_pose.items():
        if name != "reason_bytes" and name in new_pose and new_pose[name].shape == values.shape:
            np.testing.assert_array_equal(new_pose[name][0], values[0])
    assert read_reason_labels(new_pose)[0] == old_pose_reasons[0]
    # A stale long-snout annotation is rechecked against current masks. Here
    # the current fragmented mask fails for another reason, so it stays legacy.
    assert new_seed["tail_derivation_method_code"][:].tolist() == [0, 0]
    assert np.isnan(new_pose["keypoints_roi"][1, 5]).all()
    assert bool(new_pose["keypoint_manual_edit"][1, 5])
    assert result["upgrade_target_rows"] == [1]
    assert result["tasks"][0]["scope"]["target_roi_indices"] == [1]
    assert all(
        task["scope"].get("target_roi_indices") == [1]
        for task in result["tasks"]
    )
    assert result["source_bindings"]["mask_apply_refresh"]["geometry_upgrade"]["target_rows"] == [1]


def test_geometry_upgrade_refuses_nonfailure_and_records_changed_non_target_mask(reviewed_archive):
    path, root, old_result = reviewed_archive
    args = dict(
        archive=path,
        refined_mask_run=old_result["paths"]["mask_edit"].split("/")[1],
        refined_keypoint_run=old_result["paths"]["pose_edit"].split("/")[1],
        upgrade_id="explicit-upgrade-invalid",
        expected_mask_revision=1,
    )
    with pytest.raises(ValueError, match="recorded long-snout failure"):
        upgrade_training_tail_geometry_version(**args, target_rows=[1])
    old = root[old_result["paths"]["pose_edit"]]
    reasons = read_reason_labels(old)
    reasons[1] = str(reasons[1]) + "|tail_derivation_failed:snout_extension_too_long"
    write_reason_columns(old, reasons, chunk_size=2, overwrite=True)
    mask = root[old_result["paths"]["mask_edit"]]
    mask["masks_roi"][0, 0, 1, 1] = 1
    result = upgrade_training_tail_geometry_version(**args, target_rows=[1])
    changed = result["source_bindings"]["mask_apply_refresh"]["geometry_upgrade"]["non_target_mask_changed_rows"]
    assert list(changed) == ["0"]
    assert changed["0"]["source_mask_sha256"] != changed["0"]["current_mask_sha256"]


def test_geometry_upgrade_refuses_tampered_immutable_seed(reviewed_archive):
    path, root, old_result = reviewed_archive
    old = root[old_result["paths"]["pose_edit"]]
    reasons = read_reason_labels(old)
    reasons[1] = str(reasons[1]) + "|tail_derivation_failed:snout_extension_too_long"
    write_reason_columns(old, reasons, chunk_size=2, overwrite=True)
    seed = root[old_result["paths"]["seed"]]
    seed["tail_arc_length_px"][0] = 12345.0
    with pytest.raises(ValueError, match="source seed identity is invalid"):
        upgrade_training_tail_geometry_version(
            archive=path,
            refined_mask_run=old_result["paths"]["mask_edit"].split("/")[1],
            refined_keypoint_run=old_result["paths"]["pose_edit"].split("/")[1],
            upgrade_id="tampered-seed",
            expected_mask_revision=1,
            target_rows=[1],
        )


@pytest.mark.parametrize("invalid_row", [1.9, True, "1"])
def test_geometry_upgrade_refuses_noninteger_target_rows(reviewed_archive, invalid_row):
    path, _, old_result = reviewed_archive
    with pytest.raises(ValueError, match="integer row indices"):
        upgrade_training_tail_geometry_version(
            archive=path,
            refined_mask_run=old_result["paths"]["mask_edit"].split("/")[1],
            refined_keypoint_run=old_result["paths"]["pose_edit"].split("/")[1],
            upgrade_id="bad-target-row",
            expected_mask_revision=1,
            target_rows=[invalid_row],
        )


def test_later_apply_preserves_mixed_row_methods_from_real_curled_mask(tmp_path):
    from pathlib import Path

    frozen_path = Path(__file__).resolve().parents[2] / "fixtures" / "redscare_head_anchored_tail_rows_v1.npz"
    with np.load(frozen_path) as frozen:
        sample = [0, 8]  # ROI 16 needs v4; ROI 18 is a successful legacy row.
        masks = frozen["masks"][sample]
        heads = frozen["head"][sample]
        labels = tuple(frozen["labels"])
    path = tmp_path / "mixed.zarr"
    root = zarr.open_group(str(path), mode="w", use_consolidated=False)
    root.attrs.update(
        zarr_purpose="training", recording_id="mixed", stage_selector_eligible=False,
        schema_id="palette.training.merged_pose_detect_recovery_source.v1",
    )
    result = build_review_payload(
        root,
        {
            "roi_images": np.zeros((2, 384, 384), np.uint8),
            "masks_roi": masks,
            "head_keypoints_roi": heads,
            "frame_indices": np.array([11184, 11186]),
            "source_bbox_norm_coords": np.zeros((2, 4)),
            "detection_source": np.zeros(2, np.int8),
            "target_valid_channels": np.ones((2, len(labels)), bool),
        },
        labels,
        {"mask_run": "original", "recording_id": "mixed", "mask_run_attrs": {"label_schema_id": "subject_v1_lr"}},
        version="original",
    )
    source_mask = result["paths"]["mask_edit"].split("/")[1]
    source_pose = result["paths"]["pose_edit"].split("/")[1]
    first = upgrade_training_tail_geometry_version(
        archive=path, refined_mask_run=source_mask, refined_keypoint_run=source_pose,
        upgrade_id="mixed-upgrade", expected_mask_revision=0, target_rows=[0],
    )
    published = zarr.open_group(str(path), mode="r", use_consolidated=False)
    first_seed = published[first["paths"]["seed"]]
    assert first_seed["tail_derivation_method_code"][:].tolist() == [1, 0]
    assert first_seed["tail_valid"][:].tolist() == [True, True]
    editable = zarr.open_group(str(path), mode="a", use_consolidated=False)[first["paths"]["pose_edit"]]
    editable["tail_derivation_method_code"][0] = 0
    with pytest.raises(ValueError, match="method identity or seed is stale"):
        regenerate_training_tail_version(
            archive=path,
            refined_mask_run=first["paths"]["mask_edit"].split("/")[1],
            refined_keypoint_run=first["paths"]["pose_edit"].split("/")[1],
            apply_id="tampered-followup", expected_mask_revision=0,
        )
    editable["tail_derivation_method_code"][0] = 1
    second = regenerate_training_tail_version(
        archive=path,
        refined_mask_run=first["paths"]["mask_edit"].split("/")[1],
        refined_keypoint_run=first["paths"]["pose_edit"].split("/")[1],
        apply_id="mixed-followup-apply", expected_mask_revision=0,
    )
    later = zarr.open_group(str(path), mode="r", use_consolidated=False)[second["paths"]["seed"]]
    assert later["tail_derivation_method_code"][:].tolist() == [1, 0]
    np.testing.assert_array_equal(later["keypoints_roi"][1], first_seed["keypoints_roi"][1])


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


def test_visible_endpoint_acceptance_survives_successor_and_revocation_restores_strict(reviewed_archive):
    path, root, old_result = reviewed_archive
    mask = root[old_result["paths"]["mask_edit"]]
    body = np.asarray(mask["masks_roi"][0, 0]).copy()
    before_digest = body_digest(body)
    body[108:128, 64] = 1
    mask["masks_roi"][0, 0] = body
    apply_acceptance_actions(
        mask, mask_labels=tuple(mask.attrs["mask_labels"]), revision=1,
        actions=[{
            "roi_idx": 0,
            "action": {"action": "accept", "reason": "A very small visible tip is clipped"},
            "before_body_sha256": before_digest,
            "row_identity": expected_row_identity(mask, 0),
            "user": "reviewer@example.org",
            "timestamp": "2026-09-23T12:00:00+00:00",
        }],
    )
    first = refresh(reviewed_archive)
    published = zarr.open_group(str(path), mode="a", use_consolidated=False)
    seed = published[first["paths"]["seed"]]
    assert seed["tail_valid"][:].tolist() == [True, False]
    assert seed["tail_tip_truncated"][:].tolist() == [True, False]
    assert seed["keypoint_origin"][0, 13] == 2
    assert "tail_tip_is_visible_crop_endpoint" in read_reason_labels(seed)[0]
    next_mask = published[first["paths"]["mask_edit"]]
    assert next_mask.attrs[TAIL_ACCEPTANCE_ATTR]["0"]["source_crop_run"] == first["source_crop_run"]
    next_mask.attrs["edit_revision"] = 2
    second = regenerate_training_tail_version(
        archive=path,
        refined_mask_run=first["paths"]["mask_edit"].split("/")[1],
        refined_keypoint_run=first["paths"]["pose_edit"].split("/")[1],
        apply_id="apply-two", expected_mask_revision=2,
    )
    second_seed = published[second["paths"]["seed"]]
    assert second_seed["tail_tip_truncated"][0]
    second_mask = published[second["paths"]["mask_edit"]]
    second_mask.attrs[TAIL_ACCEPTANCE_ATTR] = {}
    second_mask.attrs["edit_revision"] = 3
    third = regenerate_training_tail_version(
        archive=path,
        refined_mask_run=second["paths"]["mask_edit"].split("/")[1],
        refined_keypoint_run=second["paths"]["pose_edit"].split("/")[1],
        apply_id="apply-three", expected_mask_revision=3,
    )
    third_seed = published[third["paths"]["seed"]]
    assert "tail_tip_truncated" not in third_seed
    assert not third_seed["tail_valid"][0]
    assert "body_touches_crop_border" in read_reason_labels(third_seed)[0]


def test_body_change_invalidates_only_the_edited_roi_acceptance(reviewed_archive):
    _, root, initial = reviewed_archive
    mask = root[initial["paths"]["mask_edit"]]
    labels = tuple(mask.attrs["mask_labels"])
    original = np.asarray(mask["masks_roi"][0, 0]).copy()
    clipped = original.copy()
    clipped[108:128, 64] = 1
    mask["masks_roi"][0, 0] = clipped
    actor = {"action": "accept", "reason": "Only the visible tail tip is clipped"}
    apply_acceptance_actions(mask, mask_labels=labels, revision=2, actions=[{
        "roi_idx": 0, "action": actor, "before_body_sha256": body_digest(original),
        "row_identity": expected_row_identity(mask, 0), "user": "reviewer",
        "timestamp": "2026-09-23T12:00:00+00:00",
    }])
    accepted = mask.attrs[TAIL_ACCEPTANCE_ATTR]["0"]
    other_before = np.asarray(mask["masks_roi"][1, 0]).copy()
    other_after = other_before.copy(); other_after[2, 2] = 1
    mask["masks_roi"][1, 0] = other_after
    apply_acceptance_actions(mask, mask_labels=labels, revision=3, actions=[{
        "roi_idx": 1, "action": None, "before_body_sha256": body_digest(other_before),
        "row_identity": expected_row_identity(mask, 1), "user": "reviewer",
        "timestamp": "2026-09-23T12:01:00+00:00",
    }])
    assert mask.attrs[TAIL_ACCEPTANCE_ATTR]["0"] == accepted
    changed = clipped.copy(); changed[65, 64] = 0
    mask["masks_roi"][0, 0] = changed
    apply_acceptance_actions(mask, mask_labels=labels, revision=4, actions=[{
        "roi_idx": 0, "action": None, "before_body_sha256": body_digest(clipped),
        "row_identity": expected_row_identity(mask, 0), "user": "reviewer",
        "timestamp": "2026-09-23T12:02:00+00:00",
    }])
    assert mask.attrs[TAIL_ACCEPTANCE_ATTR] == {}


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
    # Enforcement correction (2026-09-26): the replaced pose task is superseded
    # and its session closed, so no edit can land in the older version.
    superseded = store.get_task("original-pose")
    assert superseded["state"] == "superseded"
    assert {k: v for k, v in superseded.items() if k not in {"state", "updated_at_utc"}} == {
        k: v for k, v in original.items() if k not in {"state", "updated_at_utc"}
    }
    assert store.get_session(lease.session_id)["closed_at_utc"] is not None
    # The applied mask task stays open so its review can be completed.
    assert store.get_task("original-mask")["state"] == "pending"
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
    with pytest.raises(RuntimeError, match="Apply the saved keypoint edits in task original-pose"):
        refresh_training_tail_after_mask_apply(
            store=store,
            runtime=runtime,
            apply_id="mask-apply",
            expected_mask_revision=1,
        )
    assert len(list(runtime.root["refined_keypoints_runs"].group_keys())) == 1
    assert store.count_unapplied_session_checkpoints(task_id="original-pose") == 1
    assert store.get_task("original-pose")["state"] == "pending"
    store.close()


@pytest.mark.parametrize("fault", ["immutable_pixels", "missing_child"])
def test_completed_browser_effect_refuses_damaged_publication(
    reviewed_archive, tmp_path, fault
):
    from fisheye.labeling.web_mask_tail_refresh import (
        refresh_training_tail_after_mask_apply,
    )
    from fisheye.training.recovered_mask_review_payload import run_paths

    store, runtime = browser_context(reviewed_archive, tmp_path)
    try:
        result = refresh_training_tail_after_mask_apply(
            store=store, runtime=runtime, apply_id="sealed", expected_mask_revision=1
        )
        # The recorded paths are authoritative for the exact published version.
        event = store.get_event_for_target(
            task_id="original-mask",
            event_type="mask_apply_tail_successor",
            target={"apply_id": "sealed"},
        )
        native = (
            event["after"]["source_bindings"].get("source_kind")
            == "native_reviewed_training_masks_v1"
        )
        paths = run_paths(result["tail_refresh_version"], native=native)
        root = zarr.open_group(str(runtime.zarr_path), mode="a", use_consolidated=False)
        if fault == "immutable_pixels":
            seed = root[paths["seed"]]
            points = np.asarray(seed["keypoints_roi"][0]).copy()
            points[0, 0] += 1
            seed["keypoints_roi"][0] = points
        else:
            del root[paths["mask"]]
        with pytest.raises(ValueError, match="immutable|missing publication"):
            refresh_training_tail_after_mask_apply(
                store=store,
                runtime=runtime,
                apply_id="sealed",
                expected_mask_revision=1,
            )
    finally:
        store.close()


def _stage_applied_pose_checkpoint(store, *, roi_idx, applied_at, target_run):
    """An applied keypoint checkpoint as the store records one (test setup only)."""

    lease = store.create_session(task_id="original-pose", user="reviewer")
    checkpoint = store.upsert_session_checkpoint(
        session_id=lease.session_id,
        task_id="original-pose",
        recording_id="rec",
        user="reviewer",
        workflow_kind="keypoints",
        target_run_path=f"refined_keypoints_runs/{target_run}",
        target_edit_revision=0,
        source_rowset_path="crop_runs/source",
        roi_idx=roi_idx,
        component_name="keypoints",
        payload={"operation": "replace_points"},
        metadata={},
    )
    store.conn.execute(
        "UPDATE labeling_session_checkpoints SET state = 'applied', applied_at_utc = ? "
        "WHERE checkpoint_id = ?;",
        (applied_at, checkpoint["checkpoint_id"]),
    )
    store.conn.commit()
    store.close_session(session_id=lease.session_id, user="reviewer")
    return checkpoint["checkpoint_id"]


def test_superseded_task_refuses_new_edits_and_is_hidden(reviewed_archive, tmp_path):
    from fisheye.labeling.web_mask_tail_refresh import (
        refresh_training_tail_after_mask_apply,
    )

    store, runtime = browser_context(reviewed_archive, tmp_path)
    result = refresh_training_tail_after_mask_apply(
        store=store, runtime=runtime, apply_id="mask-apply", expected_mask_revision=1
    )
    visible = {t["task_id"] for t in store.list_tasks_for_user("reviewer")}
    offered = {t["task_id"] for t in result["tail_refresh_tasks"]}
    assert "original-pose" not in visible
    assert offered <= visible and "original-mask" in visible
    with pytest.raises(PermissionError):
        store.create_session(task_id="original-pose", user="reviewer")
    with pytest.raises(RuntimeError, match="replaced by a newer one"):
        store.upsert_session_checkpoint(
            session_id="any",
            task_id="original-pose",
            recording_id="rec",
            user="reviewer",
            workflow_kind="keypoints",
            target_run_path="refined_keypoints_runs/source",
            target_edit_revision=0,
            source_rowset_path=None,
            roi_idx=0,
            component_name="keypoints",
            payload={},
        )
    with pytest.raises(ValueError, match="replaced by a newer one"):
        store.update_task_state(task_id="original-pose", state="pending", user="reviewer")
    store.close()


def test_failed_successor_publication_restores_superseded_tasks(
    reviewed_archive, tmp_path, monkeypatch
):
    from fisheye.training import mask_tail_apply_refresh as refresh_mod
    from fisheye.labeling.web_mask_tail_refresh import (
        refresh_training_tail_after_mask_apply,
    )

    store, runtime = browser_context(reviewed_archive, tmp_path)

    def fail(**_kwargs):
        assert store.get_task("original-pose")["state"] == "superseded"
        raise OSError("publication interrupted")

    monkeypatch.setattr(refresh_mod, "regenerate_training_tail_version", fail)
    with pytest.raises(OSError):
        refresh_training_tail_after_mask_apply(
            store=store, runtime=runtime, apply_id="mask-apply", expected_mask_revision=1
        )
    assert store.get_task("original-pose")["state"] == "pending"
    events = [
        row["event_type"]
        for row in store.conn.execute(
            "SELECT event_type FROM labeling_task_events WHERE task_id = 'original-pose'"
        )
    ]
    assert "task_superseded" in events and "task_supersede_reverted" in events
    store.close()


def test_stranded_older_version_edit_blocks_mask_apply(
    reviewed_archive, tmp_path, monkeypatch
):
    from fisheye.labeling import web_mask_tail_refresh as refresh_web
    from fisheye.labeling.tail_successor_lineage import StrandedRow

    store, runtime = browser_context(reviewed_archive, tmp_path)
    monkeypatch.setattr(
        refresh_web,
        "stranded_keypoint_rows",
        lambda **_kwargs: [
            StrandedRow(
                roi_idx=0, source_run="older", source_task_id="t",
                checkpoint_ids=["c"], applied_at_utc="2999", disposition="carry",
            )
        ],
    )
    with pytest.raises(RuntimeError, match="labeled in an older version"):
        refresh_web.refresh_training_tail_after_mask_apply(
            store=store, runtime=runtime, apply_id="mask-apply", expected_mask_revision=1
        )
    assert store.get_task("original-pose")["state"] == "pending"
    assert len(list(runtime.root["refined_keypoints_runs"].group_keys())) == 1
    store.close()


@pytest.mark.parametrize("touched", [False, True])
def test_completing_mask_review_retires_untouched_mask_successor(
    reviewed_archive, tmp_path, touched
):
    from fisheye.labeling.web_mask_tail_refresh import (
        refresh_training_tail_after_mask_apply,
    )

    store, runtime = browser_context(reviewed_archive, tmp_path)
    store.conn.execute(
        "INSERT INTO labeling_checkpoint_apply_receipts (apply_id, task_id, component_name, "
        "state, checkpoint_count, checkpoints_json, claimed_at_utc) VALUES "
        "('mask-apply', 'original-mask', 'subject_body', 'applied', 0, '[]', '2026-01-01');"
    )
    store.conn.commit()
    result = refresh_training_tail_after_mask_apply(
        store=store, runtime=runtime, apply_id="mask-apply", expected_mask_revision=1
    )
    by_kind = {t["workflow_kind"]: t["task_id"] for t in result["tail_refresh_tasks"]}
    if touched:
        lease = store.create_session(task_id=by_kind["subject_mask_component"], user="reviewer")
        store.upsert_session_checkpoint(
            session_id=lease.session_id,
            task_id=by_kind["subject_mask_component"],
            recording_id="rec",
            user="reviewer",
            workflow_kind="subject_mask_component",
            target_run_path="refined_subject_masks_runs/x",
            target_edit_revision=0,
            source_rowset_path=None,
            roi_idx=0,
            component_name="subject_body",
            payload={},
        )
    store.update_task_state(task_id="original-mask", state="complete", user="reviewer")
    mask_state = store.get_task(by_kind["subject_mask_component"])["state"]
    assert mask_state == ("pending" if touched else "superseded")
    # Keypoint review of the new version is still real work.
    assert store.get_task(by_kind["keypoints"])["state"] == "pending"
    store.close()


def test_carry_forward_moves_stranded_edit_into_newest_version(reviewed_archive, tmp_path):
    from fisheye.labeling import carry_forward_tail_keypoints as carry
    from fisheye.labeling.web_mask_tail_refresh import (
        refresh_training_tail_after_mask_apply,
    )

    path, root, result = reviewed_archive
    store, runtime = browser_context(reviewed_archive, tmp_path)
    source_run = result["paths"]["pose_edit"].split("/")[1]
    checkpoint_id = _stage_applied_pose_checkpoint(
        store, roi_idx=0, applied_at="2000-01-01T00:00:00+00:00", target_run=source_run
    )
    offer = refresh_training_tail_after_mask_apply(
        store=store, runtime=runtime, apply_id="mask-apply", expected_mask_revision=1
    )
    newest_task = next(t for t in offer["tail_refresh_tasks"] if t["workflow_kind"] == "keypoints")
    newest_run = store.get_task(newest_task["task_id"])["run_name"]
    before = np.asarray(root[f"refined_keypoints_runs/{newest_run}/keypoints_roi"][0]).copy()

    # A labeler kept editing the older version after the snapshot (the bug).
    session = editor.resolve_review_session(str(path), refined_run=source_run, include_all=True)
    points = np.asarray(session.kp_roi_arr[0]).copy()
    points[15] += 2.0
    editor.save_roi_correction(session, position=0, points=points)
    store.conn.execute(
        "UPDATE labeling_session_checkpoints SET applied_at_utc = '2999-01-01T00:00:00+00:00' "
        "WHERE checkpoint_id = ?;",
        (checkpoint_id,),
    )
    store.conn.commit()

    plans = [p for p in carry.plan_recording(store, "rec") if "rows" in p]
    assert len(plans) == 1 and plans[0]["target_run"] == newest_run
    rows = plans[0]["rows"]
    assert [(r["roi_idx"], r["disposition"]) for r in rows] == [(0, "carry")]
    assert 15 in rows[0]["manual_keypoints"]

    outcome = carry.carry_rows(store, plans[0], user="reviewer")
    assert outcome["carried_rows"] == 1
    assert carry.verify_carried(plans[0])["rows_not_matching"] == []
    after = np.asarray(root[f"refined_keypoints_runs/{newest_run}/keypoints_roi"][0])
    np.testing.assert_allclose(after[15], points[15])
    untouched = [i for i in range(len(after)) if i not in rows[0]["manual_keypoints"]]
    np.testing.assert_allclose(after[untouched], before[untouched], equal_nan=True)
    carried = store.list_recording_applied_checkpoints(recording_id="rec", workflow_kind="keypoints")
    provenance = [c["metadata"].get("carried_from") for c in carried if c["task_id"] == newest_task["task_id"]]
    assert provenance and provenance[0]["source_checkpoint_ids"] == [checkpoint_id]
    # Carried once; a second plan finds nothing left to carry.
    again = [p for p in carry.plan_recording(store, "rec") if "rows" in p][0]["rows"]
    assert {r["disposition"] for r in again} <= {"already_present", "target_edit_newer"}
    store.close()
