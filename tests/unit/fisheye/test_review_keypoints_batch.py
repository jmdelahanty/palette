from __future__ import annotations

import sqlite3
from pathlib import Path

from fisheye.utils import review_keypoints_batch as mod


def _write_registry_for_latest(path: Path, recordings_root: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE datasets (
                dataset_id TEXT PRIMARY KEY,
                zarr_path TEXT,
                zarr_use TEXT,
                status TEXT
            );

            CREATE TABLE keypoint_quality_current (
                dataset_id TEXT,
                refined_run TEXT,
                review_state TEXT,
                review_timestamp_utc TEXT,
                refined_created_utc TEXT,
                quality_updated_utc TEXT
            );
            """
        )
        conn.executemany(
            "INSERT INTO datasets(dataset_id, zarr_path, zarr_use, status) VALUES (?, ?, ?, ?)",
            [
                ("ds_a", str(recordings_root / "a_analysis.zarr"), "analysis", None),
                ("ds_b", str(recordings_root / "b_training.zarr"), "training", None),
                ("ds_c", "/tmp/outside/c_analysis.zarr", "analysis", None),
                ("ds_d", str(recordings_root / "d_analysis.zarr"), "analysis", "missing"),
            ],
        )
        conn.executemany(
            """
            INSERT INTO keypoint_quality_current(
                dataset_id, refined_run, review_state, review_timestamp_utc, refined_created_utc, quality_updated_utc
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                ("ds_a", "refined_a", None, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
                ("ds_b", "refined_b", "approved", "2026-02-02T00:00:00Z", "2026-02-02T00:00:00Z", "2026-02-02T00:00:00Z"),
                ("ds_c", "refined_c", "pending", "2026-02-03T00:00:00Z", "2026-02-03T00:00:00Z", "2026-02-03T00:00:00Z"),
                ("ds_d", "refined_d", "needs_review", "2026-02-04T00:00:00Z", "2026-02-04T00:00:00Z", "2026-02-04T00:00:00Z"),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def test_build_plans_from_registry_filters_scope_and_zarr_use(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    recordings_root = tmp_path / "recordings"
    _write_registry_for_latest(registry_path, recordings_root)

    plans = mod._build_plans_from_registry(
        registry_path,
        roots=[recordings_root],
        refined_run=None,
        zarr_use="analysis",
    )

    ok = [plan for plan in plans if plan.status == "ok"]
    assert len(ok) == 1
    assert ok[0].zarr_path == recordings_root / "a_analysis.zarr"
    assert ok[0].refined_run == "refined_a"
    assert ok[0].review_state is None


def test_build_plans_from_registry_honors_explicit_refined_run(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    recordings_root = tmp_path / "recordings"
    target_path = recordings_root / "target_analysis.zarr"

    conn = sqlite3.connect(str(registry_path))
    try:
        conn.executescript(
            """
            CREATE TABLE datasets (
                dataset_id TEXT PRIMARY KEY,
                zarr_path TEXT,
                zarr_use TEXT,
                status TEXT
            );

            CREATE TABLE keypoint_quality (
                dataset_id TEXT,
                refined_run TEXT,
                review_state TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO datasets(dataset_id, zarr_path, zarr_use, status) VALUES (?, ?, ?, ?)",
            ("ds_target", str(target_path), "analysis", None),
        )
        conn.executemany(
            "INSERT INTO keypoint_quality(dataset_id, refined_run, review_state) VALUES (?, ?, ?)",
            [
                ("ds_target", "refined_target", "needs_review"),
                ("ds_target", "refined_other", "approved"),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    plans = mod._build_plans_from_registry(
        registry_path,
        roots=[recordings_root],
        refined_run="refined_target",
        zarr_use="analysis",
    )
    ok = [plan for plan in plans if plan.status == "ok"]
    assert len(ok) == 1
    assert ok[0].zarr_path == target_path
    assert ok[0].refined_run == "refined_target"
    assert ok[0].review_state == "needs_review"


def test_main_registry_only_requires_registry(tmp_path: Path) -> None:
    rc = mod.main([str(tmp_path), "--manual", "--registry-only"])
    assert rc == 2


def test_main_registry_mode_launches_keypoint_review(monkeypatch, tmp_path: Path) -> None:
    plans = [
        mod.ReviewPlan(
            zarr_path=tmp_path / "recordings" / "a_analysis.zarr",
            refined_run="refined_a",
            status="ok",
            review_state=None,
        )
    ]
    monkeypatch.setattr(mod, "_build_plans_from_registry", lambda *_args, **_kwargs: plans)

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("filesystem fallback should not run")

    monkeypatch.setattr(mod, "_build_plans", _fail_if_called)

    seen_cmds: list[list[str]] = []

    def _fake_run(cmd: list[str], check: bool = False) -> None:  # noqa: ARG001
        seen_cmds.append(cmd)

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    rc = mod.main(
        [
            str(tmp_path / "recordings"),
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--manual",
            "--no-prompt",
        ]
    )
    assert rc == 0
    assert len(seen_cmds) == 1
    cmd = seen_cmds[0]
    assert cmd[1:3] == ["-m", "fisheye.tune.keypoint_review"]
    assert "--manual" in cmd
    assert "--refined-run" in cmd
    assert "refined_a" in cmd
