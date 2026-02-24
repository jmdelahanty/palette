from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3

import numpy as np
import zarr

from fisheye.utils import review_eye_masks_batch as mod


def _mk_refined_run(
    parent: zarr.Group,
    run_name: str,
    *,
    review_state: str | None,
) -> None:
    run = parent.create_group(run_name)
    run.create_array("masks_roi", data=np.zeros((1, 2, 4, 4), dtype=np.uint8))
    if review_state is not None:
        run.attrs["eye_mask_review_status"] = {"state": review_state}


def test_build_plans_filters_missing_review_status(tmp_path: Path) -> None:
    z_a = tmp_path / "a_training.zarr"
    root_a = zarr.open_group(str(z_a), mode="w")
    p_a = root_a.create_group("refined_eye_masks_runs")
    _mk_refined_run(p_a, "refined_a", review_state=None)
    p_a.attrs["latest"] = "refined_a"

    z_b = tmp_path / "b_training.zarr"
    root_b = zarr.open_group(str(z_b), mode="w")
    p_b = root_b.create_group("refined_eye_masks_runs")
    _mk_refined_run(p_b, "refined_b", review_state="approved")
    p_b.attrs["latest"] = "refined_b"

    plans = mod._build_plans(
        [tmp_path],
        recursive=True,
        refined_run=None,
        status_filter="missing",
        zarr_use="training",
    )

    ok = [plan for plan in plans if plan.status == "ok"]
    assert len(ok) == 1
    assert ok[0].zarr_path.name == "a_training.zarr"
    assert ok[0].review_state is None


def test_build_plans_filters_specific_review_state(tmp_path: Path) -> None:
    z_p = tmp_path / "p_analysis.zarr"
    root_p = zarr.open_group(str(z_p), mode="w")
    p = root_p.create_group("refined_eye_masks_runs")
    _mk_refined_run(p, "refined_p", review_state="pending")
    p.attrs["latest"] = "refined_p"

    plans = mod._build_plans(
        [z_p],
        recursive=False,
        refined_run=None,
        status_filter="pending",
        zarr_use="analysis",
    )
    ok = [plan for plan in plans if plan.status == "ok"]
    assert len(ok) == 1
    assert ok[0].review_state == "pending"


def test_build_plans_marks_missing_when_no_refined_runs(tmp_path: Path) -> None:
    z = tmp_path / "m_training.zarr"
    zarr.open_group(str(z), mode="w")

    plans = mod._build_plans(
        [z],
        recursive=False,
        refined_run=None,
        status_filter="missing",
        zarr_use="training",
    )
    assert len(plans) == 1
    assert plans[0].status == "missing"
    assert "no refined_eye_masks_runs" in str(plans[0].reason)


def test_build_plans_from_registry_filters_missing_review_status(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    recordings_root = tmp_path / "recordings"
    a_path = recordings_root / "a_training.zarr"
    b_path = recordings_root / "b_training.zarr"
    c_path = recordings_root / "c_analysis.zarr"

    with sqlite3.connect(str(registry_path)) as conn:
        conn.execute(
            """
            CREATE TABLE recording_eye_mask_performance_latest (
                zarr_path TEXT,
                run_name TEXT,
                review_state TEXT,
                zarr_use TEXT,
                stage_group TEXT
            );
            """
        )
        conn.executemany(
            """
            INSERT INTO recording_eye_mask_performance_latest
                (zarr_path, run_name, review_state, zarr_use, stage_group)
            VALUES (?, ?, ?, ?, ?);
            """,
            [
                (str(a_path), "refined_a", None, "training", "refined_eye_masks_runs"),
                (str(b_path), "refined_b", "approved", "training", "refined_eye_masks_runs"),
                (str(c_path), "refined_c", None, "analysis", "refined_eye_masks_runs"),
                (str(a_path), "eye_masks_a", None, "training", "eye_masks_runs"),
            ],
        )

    plans = mod._build_plans_from_registry(
        registry_path,
        roots=[recordings_root],
        refined_run=None,
        status_filter="missing",
        zarr_use="training",
    )
    ok = [plan for plan in plans if plan.status == "ok"]
    assert len(ok) == 1
    assert ok[0].zarr_path == a_path
    assert ok[0].refined_run == "refined_a"
    assert ok[0].review_state is None


def test_build_plans_from_registry_honors_explicit_refined_run(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    target_path = tmp_path / "recordings" / "target_training.zarr"

    with sqlite3.connect(str(registry_path)) as conn:
        conn.execute("CREATE TABLE datasets (dataset_id TEXT, zarr_path TEXT, status TEXT);")
        conn.execute(
            """
            CREATE TABLE eye_mask_performance (
                dataset_id TEXT,
                stage_group TEXT,
                run_name TEXT,
                review_state TEXT,
                zarr_use TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO datasets (dataset_id, zarr_path, status) VALUES (?, ?, ?);",
            ("ds_target", str(target_path), None),
        )
        conn.executemany(
            """
            INSERT INTO eye_mask_performance
                (dataset_id, stage_group, run_name, review_state, zarr_use)
            VALUES (?, ?, ?, ?, ?);
            """,
            [
                ("ds_target", "refined_eye_masks_runs", "refined_target", None, "training"),
                ("ds_target", "refined_eye_masks_runs", "refined_other", "approved", "training"),
                ("ds_target", "eye_masks_runs", "eye_masks_target", None, "training"),
            ],
        )

    plans = mod._build_plans_from_registry(
        registry_path,
        roots=[tmp_path / "recordings"],
        refined_run="refined_target",
        status_filter="missing",
        zarr_use="training",
    )
    ok = [plan for plan in plans if plan.status == "ok"]
    assert len(ok) == 1
    assert ok[0].zarr_path == target_path
    assert ok[0].refined_run == "refined_target"


def _viewer_args(*, registry: Path | None) -> argparse.Namespace:
    return argparse.Namespace(
        padding=16,
        scale_percent=220,
        edit_zoom=4,
        frame_flag_file="eye_mask_frame_flags.json",
        review_state="approved",
        review_method="manual",
        review_intended_use="training",
        registry=registry,
        crop_run=None,
        keypoint_run=None,
        keypoint_group=None,
        reviewer=None,
        review_notes=None,
    )


def test_viewer_cmd_includes_registry_when_set(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    plan = mod.ReviewPlan(
        zarr_path=tmp_path / "recordings" / "a_training.zarr",
        refined_run="refined_a",
        review_state=None,
        status="ok",
    )
    cmd = mod._viewer_cmd(_viewer_args(registry=registry_path), plan)
    assert "--registry" in cmd
    idx = cmd.index("--registry")
    assert cmd[idx + 1] == str(registry_path)


def test_viewer_cmd_omits_registry_when_unset(tmp_path: Path) -> None:
    plan = mod.ReviewPlan(
        zarr_path=tmp_path / "recordings" / "a_training.zarr",
        refined_run="refined_a",
        review_state=None,
        status="ok",
    )
    cmd = mod._viewer_cmd(_viewer_args(registry=None), plan)
    assert "--registry" not in cmd
