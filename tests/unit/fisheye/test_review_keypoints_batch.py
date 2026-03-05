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
                quality_updated_utc TEXT,
                usable_keypoints INTEGER,
                total_keypoints INTEGER
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
                dataset_id, refined_run, review_state, review_timestamp_utc, refined_created_utc, quality_updated_utc,
                usable_keypoints, total_keypoints
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("ds_a", "refined_a", None, "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", 90, 100),
                ("ds_b", "refined_b", "approved", "2026-02-02T00:00:00Z", "2026-02-02T00:00:00Z", "2026-02-02T00:00:00Z", 80, 100),
                ("ds_c", "refined_c", "pending", "2026-02-03T00:00:00Z", "2026-02-03T00:00:00Z", "2026-02-03T00:00:00Z", 100, 100),
                ("ds_d", "refined_d", "needs_review", "2026-02-04T00:00:00Z", "2026-02-04T00:00:00Z", "2026-02-04T00:00:00Z", 75, 100),
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
                review_state TEXT,
                usable_keypoints INTEGER,
                total_keypoints INTEGER
            );
            """
        )
        conn.execute(
            "INSERT INTO datasets(dataset_id, zarr_path, zarr_use, status) VALUES (?, ?, ?, ?)",
            ("ds_target", str(target_path), "analysis", None),
        )
        conn.executemany(
            "INSERT INTO keypoint_quality(dataset_id, refined_run, review_state, usable_keypoints, total_keypoints) VALUES (?, ?, ?, ?, ?)",
            [
                ("ds_target", "refined_target", "needs_review", 40, 100),
                ("ds_target", "refined_other", "approved", 100, 100),
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


def test_build_plans_from_registry_filters_out_no_failures_when_requested(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    recordings_root = tmp_path / "recordings"
    _write_registry_for_latest(registry_path, recordings_root)

    plans = mod._build_plans_from_registry(
        registry_path,
        roots=[recordings_root],
        refined_run=None,
        zarr_use="analysis",
        require_failures=True,
    )

    ok = [plan for plan in plans if plan.status == "ok"]
    assert len(ok) == 1
    assert ok[0].zarr_path == recordings_root / "a_analysis.zarr"
    assert ok[0].refined_run == "refined_a"


def test_build_plans_from_registry_filters_not_approved(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    recordings_root = tmp_path / "recordings"
    _write_registry_for_latest(registry_path, recordings_root)

    plans = mod._build_plans_from_registry(
        registry_path,
        roots=[recordings_root],
        refined_run=None,
        zarr_use="any",
        review_state_filter="not_approved",
    )

    ok = [plan for plan in plans if plan.status == "ok"]
    assert len(ok) == 1
    assert ok[0].zarr_path == recordings_root / "a_analysis.zarr"
    assert ok[0].review_state is None


def test_build_plans_from_registry_filters_missing_review_state(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    recordings_root = tmp_path / "recordings"
    _write_registry_for_latest(registry_path, recordings_root)

    plans = mod._build_plans_from_registry(
        registry_path,
        roots=[recordings_root],
        refined_run=None,
        zarr_use="any",
        review_state_filter="missing",
    )

    ok = [plan for plan in plans if plan.status == "ok"]
    assert len(ok) == 1
    assert ok[0].zarr_path == recordings_root / "a_analysis.zarr"
    assert ok[0].review_state is None


def test_build_plans_from_registry_excludes_runs_with_no_review_failures(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    recordings_root = tmp_path / "recordings"
    _write_registry_for_latest(registry_path, recordings_root)

    monkeypatch.setattr(mod, "_run_has_review_failures", lambda _path, _run: False)

    plans = mod._build_plans_from_registry(
        registry_path,
        roots=[recordings_root],
        refined_run=None,
        zarr_use="analysis",
        require_failures=True,
    )

    assert plans == []


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

    seen_runs: list[tuple[list[str], dict[str, object]]] = []

    def _fake_run(cmd: list[str], check: bool = False, **kwargs: object) -> None:  # noqa: ARG001
        seen_runs.append((cmd, kwargs))

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
    assert len(seen_runs) == 1
    cmd, kwargs = seen_runs[0]
    assert cmd[1:3] == ["-m", "fisheye.tune.keypoint_review"]
    assert "--manual" in cmd
    assert "--refined-run" in cmd
    assert "refined_a" in cmd
    assert "--review-intended-use" not in cmd
    env = kwargs.get("env")
    assert isinstance(env, dict)
    assert env.get("PALETTE_REGISTRY_PATH") == str((tmp_path / "registry.sqlite").resolve())


def test_main_registry_mode_passes_explicit_review_intended_use(monkeypatch, tmp_path: Path) -> None:
    plans = [
        mod.ReviewPlan(
            zarr_path=tmp_path / "recordings" / "a_analysis.zarr",
            refined_run="refined_a",
            status="ok",
            review_state=None,
        )
    ]
    monkeypatch.setattr(mod, "_build_plans_from_registry", lambda *_args, **_kwargs: plans)
    monkeypatch.setattr(mod, "_build_plans", lambda *_args, **_kwargs: [])

    seen_runs: list[tuple[list[str], dict[str, object]]] = []

    def _fake_run(cmd: list[str], check: bool = False, **kwargs: object) -> None:  # noqa: ARG001
        seen_runs.append((cmd, kwargs))

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    rc = mod.main(
        [
            str(tmp_path / "recordings"),
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--manual",
            "--review-intended-use",
            "full_recording",
            "--no-prompt",
        ]
    )
    assert rc == 0
    assert len(seen_runs) == 1
    cmd, kwargs = seen_runs[0]
    assert "--review-intended-use" in cmd
    idx = cmd.index("--review-intended-use")
    assert cmd[idx + 1] == "full_recording"
    env = kwargs.get("env")
    assert isinstance(env, dict)
    assert env.get("PALETTE_REGISTRY_PATH") == str((tmp_path / "registry.sqlite").resolve())


def test_main_registry_mode_passes_review_state_filter(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_build_plans_from_registry(*_args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(mod, "_build_plans_from_registry", _fake_build_plans_from_registry)
    monkeypatch.setattr(mod, "_build_plans", lambda *_args, **_kwargs: [])

    rc = mod.main(
        [
            str(tmp_path / "recordings"),
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--manual",
            "--review-state-filter",
            "not_approved",
            "--list",
        ]
    )
    assert rc == 0
    assert captured["review_state_filter"] == "not_approved"
