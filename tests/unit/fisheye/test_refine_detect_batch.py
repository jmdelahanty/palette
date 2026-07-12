from pathlib import Path

import pytest
import zarr

from fisheye.utils import refine_detect_batch as mod
from fisheye.utils.refine_detect_batch import _build_plans


def _make_archive(
    root: Path,
    recording: str,
    zarr_name: str,
    *,
    zarr_purpose: str | None,
    with_detect: bool = True,
) -> Path:
    zarr_path = root / recording / "zarr" / zarr_name
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(zarr_path), mode="w")
    if zarr_purpose is not None:
        group.attrs["zarr_purpose"] = zarr_purpose
    if with_detect:
        detect = group.create_group("detect_runs")
        detect.create_group("detect_001")
        detect.attrs["latest"] = "detect_001"
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
        detect_run=None,
        skip_existing=False,
        zarr_use_filter="analysis",
    )
    by_name = {plan.zarr_path.name: plan for plan in plans}

    assert by_name[analysis.name].status == "ok"
    assert by_name[analysis.name].detect_run == "detect_001"
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
        detect_run=None,
        skip_existing=False,
        zarr_use_filter="any",
    )
    by_name = {plan.zarr_path.name: plan for plan in plans}
    assert by_name[analysis.name].status == "ok"
    assert by_name[training.name].status == "ok"


def test_build_plans_accepts_direct_zarr_directory(tmp_path: Path) -> None:
    analysis = _make_archive(
        tmp_path,
        "rec_a",
        "rec_a_analysis.zarr",
        zarr_purpose="analysis",
    )

    plans = _build_plans(
        [analysis],
        recursive=False,
        detect_run="detect_001",
        skip_existing=False,
        zarr_use_filter="analysis",
    )

    assert len(plans) == 1
    assert plans[0].zarr_path == analysis
    assert plans[0].status == "ok"
    assert plans[0].detect_run == "detect_001"


def test_build_plans_reads_latest_detect_run_when_consolidated_metadata_is_stale(
    tmp_path: Path,
) -> None:
    analysis = _make_archive(
        tmp_path,
        "rec_a",
        "rec_a_analysis.zarr",
        zarr_purpose="analysis",
    )
    zarr.consolidate_metadata(str(analysis))
    live_root = zarr.open_group(str(analysis), mode="a", use_consolidated=False)
    detect = live_root["detect_runs"]
    detect.create_group("detect_002")
    detect.attrs["latest"] = "detect_002"

    plans = _build_plans(
        [analysis],
        recursive=False,
        detect_run=None,
        skip_existing=False,
        zarr_use_filter="analysis",
    )

    assert len(plans) == 1
    assert plans[0].status == "ok"
    assert plans[0].detect_run == "detect_002"


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
        detect_run=None,
        skip_existing=False,
        zarr_use_filter="analysis",
    )
    by_name = {plan.zarr_path.name: plan for plan in plans}
    assert by_name[inferred_analysis.name].status == "ok"
    assert by_name[unknown.name].status == "skipped"
    assert "found=unknown" in (by_name[unknown.name].reason or "")


@pytest.mark.parametrize("argv", [["--max-gap", "5"], ["--method", "linear"]])
def test_main_rejects_deprecated_interpolation_args(argv: list[str]) -> None:
    with pytest.raises(SystemExit, match="Interpolation overrides are deprecated and unsupported"):
        mod.main(argv)
