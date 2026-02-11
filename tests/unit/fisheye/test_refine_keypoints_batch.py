from pathlib import Path

import zarr

from fisheye.utils.refine_keypoints_batch import _build_plans


def _make_archive(
    root: Path,
    recording: str,
    zarr_name: str,
    *,
    zarr_purpose: str | None,
    with_keypoints: bool = True,
    with_refined: bool = False,
) -> Path:
    zarr_path = root / recording / "zarr" / zarr_name
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(zarr_path), mode="w")
    if zarr_purpose is not None:
        group.attrs["zarr_purpose"] = zarr_purpose
    if with_keypoints:
        keypoints = group.create_group("keypoints_runs")
        keypoints.create_group("keypoints_001")
        keypoints.attrs["latest"] = "keypoints_001"
    if with_refined:
        refined = group.create_group("refined_keypoints_runs")
        refined.create_group("refined_keypoints_001")
        refined.attrs["latest"] = "refined_keypoints_001"
    return zarr_path


def test_build_plans_analysis_filter_skips_training(tmp_path: Path) -> None:
    analysis = _make_archive(
        tmp_path,
        "rec_a",
        "rec_a_analysis.zarr",
        zarr_purpose="analysis",
    )
    training = _make_archive(
        tmp_path,
        "rec_b",
        "rec_b_training.zarr",
        zarr_purpose="training",
    )

    plans = _build_plans(
        [tmp_path],
        recursive=True,
        keypoint_run=None,
        skip_existing=False,
        zarr_use_filter="analysis",
    )
    by_name = {plan.zarr_path.name: plan for plan in plans}

    assert by_name[analysis.name].status == "ok"
    assert by_name[analysis.name].keypoint_run == "keypoints_001"
    assert by_name[training.name].status == "skipped"
    assert "wanted=analysis" in (by_name[training.name].reason or "")


def test_build_plans_any_filter_includes_all_uses(tmp_path: Path) -> None:
    analysis = _make_archive(
        tmp_path,
        "rec_a",
        "rec_a_analysis.zarr",
        zarr_purpose="analysis",
    )
    training = _make_archive(
        tmp_path,
        "rec_b",
        "rec_b_training.zarr",
        zarr_purpose="training",
    )

    plans = _build_plans(
        [tmp_path],
        recursive=True,
        keypoint_run=None,
        skip_existing=False,
        zarr_use_filter="any",
    )
    by_name = {plan.zarr_path.name: plan for plan in plans}
    assert by_name[analysis.name].status == "ok"
    assert by_name[training.name].status == "ok"


def test_build_plans_filter_uses_name_suffix_when_attr_missing(tmp_path: Path) -> None:
    inferred_analysis = _make_archive(
        tmp_path,
        "rec_a",
        "rec_a_analysis.zarr",
        zarr_purpose=None,
    )
    unknown = _make_archive(
        tmp_path,
        "rec_b",
        "rec_b_custom.zarr",
        zarr_purpose=None,
    )

    plans = _build_plans(
        [tmp_path],
        recursive=True,
        keypoint_run=None,
        skip_existing=False,
        zarr_use_filter="analysis",
    )
    by_name = {plan.zarr_path.name: plan for plan in plans}
    assert by_name[inferred_analysis.name].status == "ok"
    assert by_name[unknown.name].status == "skipped"
    assert "found=unknown" in (by_name[unknown.name].reason or "")


def test_build_plans_skips_when_refined_present_by_default(tmp_path: Path) -> None:
    zarr_path = _make_archive(
        tmp_path,
        "rec_a",
        "rec_a_analysis.zarr",
        zarr_purpose="analysis",
        with_refined=True,
    )

    plans = _build_plans(
        [tmp_path],
        recursive=True,
        keypoint_run=None,
        skip_existing=True,
        zarr_use_filter="analysis",
    )
    by_name = {plan.zarr_path.name: plan for plan in plans}

    assert by_name[zarr_path.name].status == "skipped"
    assert by_name[zarr_path.name].reason == "refined_keypoints_runs present"


def test_build_plans_marks_missing_when_requested_run_not_found(tmp_path: Path) -> None:
    zarr_path = _make_archive(
        tmp_path,
        "rec_a",
        "rec_a_analysis.zarr",
        zarr_purpose="analysis",
    )

    plans = _build_plans(
        [tmp_path],
        recursive=True,
        keypoint_run="missing_run",
        skip_existing=False,
        zarr_use_filter="analysis",
    )
    by_name = {plan.zarr_path.name: plan for plan in plans}

    assert by_name[zarr_path.name].status == "missing"
    assert by_name[zarr_path.name].reason == "keypoint_run not found"
