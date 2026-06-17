from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from fisheye.utils import review_dish_masks as mod


def _write_registry(path: Path, root: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE datasets (
                dataset_id TEXT PRIMARY KEY,
                recording_id TEXT,
                zarr_path TEXT,
                zarr_use TEXT,
                status TEXT
            );

            CREATE TABLE recordings (
                recording_id TEXT PRIMARY KEY,
                dish_design TEXT,
                camera_id TEXT
            );

            CREATE TABLE recording_step_status (
                dataset_id TEXT,
                step_name TEXT,
                status TEXT
            );
            """
        )
        conn.executemany(
            """
            INSERT INTO recordings(recording_id, dish_design, camera_id)
            VALUES (?, ?, ?);
            """,
            [
                ("rec_missing", "palm1", "2010093"),
                ("rec_present", "palm1", "2010094"),
                ("rec_training", "palm1", "2010095"),
                ("rec_missing_dataset", "palm1", "2010096"),
                ("rec_outside", "palm1", "2010097"),
            ],
        )
        conn.executemany(
            """
            INSERT INTO datasets(dataset_id, recording_id, zarr_path, zarr_use, status)
            VALUES (?, ?, ?, ?, ?);
            """,
            [
                ("ds_missing", "rec_missing", str(root / "missing_analysis.zarr"), "analysis", None),
                (
                    "ds_missing_duplicate",
                    "rec_missing",
                    str(root / "missing_analysis.zarr"),
                    "analysis",
                    None,
                ),
                ("ds_present", "rec_present", str(root / "present_analysis.zarr"), "analysis", None),
                ("ds_training", "rec_training", str(root / "training.zarr"), "training", None),
                (
                    "ds_missing_dataset",
                    "rec_missing_dataset",
                    str(root / "missing_dataset_analysis.zarr"),
                    "analysis",
                    "missing",
                ),
                ("ds_outside", "rec_outside", str(root.parent / "outside_analysis.zarr"), "analysis", None),
            ],
        )
        conn.execute(
            "INSERT INTO recording_step_status(dataset_id, step_name, status) VALUES (?, 'dish_mask', 'ok');",
            ("ds_present",),
        )
        conn.commit()
    finally:
        conn.close()


def test_build_plans_from_registry_lists_missing_analysis_dish_masks(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    root = tmp_path / "recordings"
    _write_registry(registry_path, root)

    plans = mod._build_plans_from_registry(
        registry_path,
        roots=[root],
        chamber_filter=None,
        camera_filter=None,
        only_missing=True,
        only_present=False,
    )

    assert [plan.zarr_path for plan in plans] == [root / "missing_analysis.zarr"]
    assert plans[0].chamber == "palm1"
    assert plans[0].camera_id == "2010093"
    assert plans[0].has_mask is False


def test_build_plans_from_registry_filters_present_camera_and_chamber(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    root = tmp_path / "recordings"
    _write_registry(registry_path, root)

    plans = mod._build_plans_from_registry(
        registry_path,
        roots=[root],
        chamber_filter="palm1",
        camera_filter="2010094",
        only_missing=False,
        only_present=True,
    )

    assert [plan.zarr_path for plan in plans] == [root / "present_analysis.zarr"]
    assert plans[0].has_mask is True


def test_main_registry_mode_requires_registry(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        mod.main([str(tmp_path), "--source", "registry"])
    assert excinfo.value.code == 2


def test_main_registry_mode_launches_tuner_with_registry(monkeypatch, tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    root = tmp_path / "recordings"
    plans = [
        mod.ReviewPlan(
            zarr_path=root / "missing_analysis.zarr",
            chamber="palm1",
            camera_id="2010093",
            has_mask=False,
            status="ok",
        )
    ]
    monkeypatch.setattr(mod, "_build_plans_from_registry", lambda *_args, **_kwargs: plans)
    monkeypatch.setattr(mod, "_build_plans", lambda *_args, **_kwargs: [])

    seen: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool = False) -> None:  # noqa: ARG001
        seen.append(cmd)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    rc = mod.main(
        [
            str(root),
            "--source",
            "registry",
            "--registry",
            str(registry_path),
            "--limit",
            "1",
        ]
    )

    assert rc == 0
    assert len(seen) == 1
    cmd = seen[0]
    assert cmd[1:3] == ["-m", "fisheye.tune.mask_tuner"]
    assert str(root / "missing_analysis.zarr") in cmd
    assert "--registry" in cmd
    assert str(registry_path) in cmd
