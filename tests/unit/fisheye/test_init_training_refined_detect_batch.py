from __future__ import annotations

import sqlite3
from pathlib import Path

from fisheye.utils import init_training_refined_detect_batch as mod


class _FakeRegistry:
    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """
            CREATE TABLE datasets (
                dataset_id TEXT,
                recording_id TEXT,
                zarr_path TEXT,
                zarr_use TEXT,
                status TEXT
            );
            """
        )


class _FakeGroup(dict[str, object]):
    def __init__(self, *, attrs: dict[str, object] | None = None) -> None:
        super().__init__()
        self.attrs = attrs or {}


def _seed_dataset(registry: _FakeRegistry, *, zarr_path: Path) -> None:
    registry.conn.execute(
        "INSERT INTO datasets VALUES (?, ?, ?, ?, ?);",
        ("ds_train", "rec_a", str(zarr_path), "training", "ok"),
    )


def test_build_plans_uses_latest_complete_detect_and_deterministic_refined_run(
    monkeypatch,
    tmp_path: Path,
) -> None:
    registry = _FakeRegistry()
    zarr_path = tmp_path / "rec_training.zarr"
    zarr_path.mkdir()
    _seed_dataset(registry, zarr_path=zarr_path)
    detect_parent = _FakeGroup(attrs={"latest_complete": "detect_seed"})
    detect_parent["detect_seed"] = _FakeGroup()
    root = _FakeGroup()
    root["detect_runs"] = detect_parent

    monkeypatch.setattr(mod, "_open_zarr_group", lambda *_args, **_kwargs: root)

    plans = mod.build_plans(
        registry,  # type: ignore[arg-type]
        run_id="red_scare_detection_review_001",
        refined_run_name=None,
        detect_run=None,
        path_contains=("rec_training",),
        recording_ids=(),
        scope_paths=(tmp_path,),
    )

    assert len(plans) == 1
    assert plans[0].status == "ok"
    assert plans[0].detect_run == "detect_seed"
    assert plans[0].refined_run == "refined_detect_training_review_red_scare_detection_review_001"


def test_build_plans_skips_existing_refined_run(monkeypatch, tmp_path: Path) -> None:
    registry = _FakeRegistry()
    zarr_path = tmp_path / "rec_training.zarr"
    zarr_path.mkdir()
    _seed_dataset(registry, zarr_path=zarr_path)
    detect_parent = _FakeGroup(attrs={"latest": "detect_seed"})
    detect_parent["detect_seed"] = _FakeGroup()
    refined_parent = _FakeGroup()
    refined_parent["refined_detect_training_review_run_001"] = _FakeGroup()
    root = _FakeGroup()
    root["detect_runs"] = detect_parent
    root["refined_detect_runs"] = refined_parent

    monkeypatch.setattr(mod, "_open_zarr_group", lambda *_args, **_kwargs: root)

    plans = mod.build_plans(
        registry,  # type: ignore[arg-type]
        run_id="run_001",
        refined_run_name=None,
        detect_run=None,
        path_contains=("rec_training",),
        recording_ids=(),
        scope_paths=(tmp_path,),
    )

    assert plans[0].status == "skipped"
    assert plans[0].reason == "refined_run_exists"


def test_build_plans_reports_missing_explicit_detect_run(monkeypatch, tmp_path: Path) -> None:
    registry = _FakeRegistry()
    zarr_path = tmp_path / "rec_training.zarr"
    zarr_path.mkdir()
    _seed_dataset(registry, zarr_path=zarr_path)
    root = _FakeGroup()
    root["detect_runs"] = _FakeGroup(attrs={"latest": "detect_seed"})

    monkeypatch.setattr(mod, "_open_zarr_group", lambda *_args, **_kwargs: root)

    plans = mod.build_plans(
        registry,  # type: ignore[arg-type]
        run_id="run_001",
        refined_run_name=None,
        detect_run="missing_seed",
        path_contains=("rec_training",),
        recording_ids=(),
        scope_paths=(tmp_path,),
    )

    assert plans[0].status == "missing"
    assert plans[0].reason == "missing_detect_run"
    assert plans[0].detect_run == "missing_seed"
