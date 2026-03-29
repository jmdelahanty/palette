from __future__ import annotations

import numpy as np
import zarr

from fisheye.diagnostics import review_refined_eye_mask_failures as mod


def _build_archive(tmp_path):
    root = zarr.open_group(str(tmp_path / "archive.zarr"), mode="w")
    eye_parent = root.create_group("eye_masks_runs")
    eye = eye_parent.create_group("eye_a")
    eye.attrs["source_crop_run"] = "crop_a"
    eye.attrs["source_keypoint_run"] = "kp_a"
    eye.attrs["source_keypoint_group"] = "keypoints_runs"
    refined_parent = root.create_group("refined_eye_masks_runs")
    refined = refined_parent.create_group("refined_a")
    refined.attrs["source_eye_masks_run"] = "eye_a"
    metrics = refined.create_group("metrics")
    reason = metrics.create_array(
        "reason",
        shape=(5,),
        chunks=(5,),
        dtype=zarr.core.dtype.VariableLengthUTF8(),
        fill_value="",
    )
    reason[:] = np.array(
        [
            "union_source|split_by_keypoint|ellipse_fail_pair",
            "union_source|filtered_pair",
            "union_source|ellipse_fail_pair|small_area_pair",
            "refined",
            "union_source|ellipse_fail_left",
        ],
        dtype=object,
    )
    keypoints = root.create_group("keypoints_runs")
    keypoints.create_group("kp_a")
    crop_runs = root.create_group("crop_runs")
    crop_runs.create_group("crop_a")
    return tmp_path / "archive.zarr"


def test_reason_tag_indices_matches_compound_tags() -> None:
    values = [
        "a|b|ellipse_fail_pair",
        "ellipse_fail_left",
        "",
        "ellipse_fail_pair|small_area_pair",
    ]
    assert mod._reason_tag_indices(values, "ellipse_fail_pair") == [0, 3]


def test_select_indices_subsamples_stably() -> None:
    selected = mod._select_indices(list(range(10)), limit=3, seed=7)
    assert selected == sorted(selected)
    assert len(selected) == 3


def test_review_refined_eye_mask_failures_resolves_lineage_and_calls_viewer(
    tmp_path, monkeypatch
) -> None:
    archive = _build_archive(tmp_path)
    captured = {}

    def _fake_create_viewer(
        zarr_path,
        eye_run,
        crop_run,
        keypoint_run,
        refined_runs=None,
        frame_flag_file=None,
        roi_indices=None,
    ):
        captured["zarr_path"] = zarr_path
        captured["eye_run"] = eye_run
        captured["crop_run"] = crop_run
        captured["keypoint_run"] = keypoint_run
        captured["refined_runs"] = refined_runs
        captured["frame_flag_file"] = frame_flag_file
        captured["roi_indices"] = list(roi_indices or [])

    monkeypatch.setattr(mod, "create_viewer", _fake_create_viewer)

    mod.review_refined_eye_mask_failures(
        archive,
        refined_run="refined_a",
        reason_tag="ellipse_fail_pair",
        limit=None,
    )

    assert captured["eye_run"] == "eye_a"
    assert captured["crop_run"] == "crop_a"
    assert captured["keypoint_run"] == "kp_a"
    assert captured["refined_runs"] == ["refined_a"]
    assert captured["roi_indices"] == [0, 2]


def test_review_refined_eye_mask_failures_requires_matching_tag(tmp_path) -> None:
    archive = _build_archive(tmp_path)

    try:
        mod.review_refined_eye_mask_failures(
            archive,
            refined_run="refined_a",
            reason_tag="does_not_exist",
        )
    except ValueError as exc:
        assert "matched reason tag" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
