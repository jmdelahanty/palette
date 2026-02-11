from __future__ import annotations

from pathlib import Path

import zarr

from fisheye.utils.clean_incomplete_crop_runs import _build_plan, main


def _make_crop_archive(tmp_path: Path, name: str) -> Path:
    zarr_path = tmp_path / name / "zarr" / f"{name}_analysis.zarr"
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(zarr_path), mode="w")
    parent = root.create_group("crop_runs")

    completed = parent.create_group("crop_1")
    completed.attrs["status"] = "completed"
    completed.attrs["created_at_utc"] = "2026-02-10T00:00:00+00:00"

    failed = parent.create_group("crop_2")
    failed.attrs["status"] = "failed"
    failed.attrs["created_at_utc"] = "2026-02-10T00:01:00+00:00"

    running = parent.create_group("crop_3")
    running.attrs["status"] = "running"
    running.attrs["created_at_utc"] = "2026-02-10T00:02:00+00:00"

    parent.attrs["latest"] = "crop_3"
    return zarr_path


def test_build_plan_targets_failed_and_running(tmp_path: Path) -> None:
    zarr_path = _make_crop_archive(tmp_path, "rec1")
    plan = _build_plan(
        zarr_path,
        remove_non_completed=False,
        remove_statuses={"failed", "running"},
    )
    assert plan.delete_runs == ["crop_2", "crop_3"]
    assert plan.latest_before == "crop_3"
    assert plan.latest_after == "crop_1"


def test_main_apply_deletes_incomplete_and_repairs_latest(tmp_path: Path) -> None:
    zarr_path = _make_crop_archive(tmp_path, "rec2")

    rc = main([str(zarr_path), "--apply"])
    assert rc == 0

    root = zarr.open_group(str(zarr_path), mode="r")
    crop_parent = root["crop_runs"]
    assert sorted(list(crop_parent.group_keys())) == ["crop_1"]
    assert crop_parent.attrs.get("latest") == "crop_1"

