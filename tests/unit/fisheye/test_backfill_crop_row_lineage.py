from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fisheye.utils import backfill_crop_row_lineage as mod


def _write_group(path: Path, attrs: dict[str, object] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.joinpath("zarr.json").write_text(
        json.dumps({"attributes": attrs or {}, "zarr_format": 3, "node_type": "group"}),
        encoding="utf-8",
    )


def _write_array_meta(path: Path, shape: tuple[int, ...]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.joinpath("zarr.json").write_text(
        json.dumps({"shape": list(shape), "zarr_format": 3, "node_type": "array"}),
        encoding="utf-8",
    )


def _fixture_zarr(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "recording_training.zarr"
    _write_group(zarr_path)
    _write_group(zarr_path / "crop_runs", {"latest": "crop_001"})
    _write_group(
        zarr_path / "crop_runs" / "crop_001",
        {"detection_source_path": "refined_detect_runs/refined_001/instances"},
    )
    _write_array_meta(zarr_path / "crop_runs" / "crop_001" / "detection_indices", (4,))
    _write_group(zarr_path / "refined_detect_runs")
    _write_group(zarr_path / "refined_detect_runs" / "refined_001")
    _write_group(zarr_path / "refined_detect_runs" / "refined_001" / "instances")
    _write_array_meta(
        zarr_path / "refined_detect_runs" / "refined_001" / "instances" / "refined_row_ids",
        (3,),
    )
    _write_array_meta(
        zarr_path / "refined_detect_runs" / "refined_001" / "instances" / "source_detect_row_index",
        (3,),
    )
    return zarr_path


def test_build_crop_identity_payload_marks_unmappable_rows_minus_one() -> None:
    payload = mod.build_crop_identity_payload(
        detection_indices=np.array([0, 2, 3, -1], dtype=np.int64),
        refined_row_ids=np.array([100, 101, 102], dtype=np.int64),
        source_detect_row_index=np.array([5, -1, 7], dtype=np.int32),
    )

    assert payload.source_refined_row_ids.tolist() == [100, 102, -1, -1]
    assert payload.source_detect_row_index.tolist() == [5, 7, -1, -1]
    assert payload.mappable_rows == 2
    assert payload.unmappable_rows == 2


def test_plan_crop_backfill_plans_missing_identity_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = _fixture_zarr(tmp_path)

    def fake_open_array(path: Path) -> np.ndarray:
        if path.name == "detection_indices":
            return np.array([0, 2, 3, -1], dtype=np.int64)
        if path.name == "refined_row_ids":
            return np.array([100, 101, 102], dtype=np.int64)
        if path.name == "source_detect_row_index":
            return np.array([5, -1, 7], dtype=np.int32)
        raise AssertionError(path)

    monkeypatch.setattr(mod, "_open_array", fake_open_array)

    plan = mod.plan_crop_backfill(zarr_path, "crop_001", overwrite=False)

    assert plan.status == "planned"
    assert plan.crop_rows == 4
    assert plan.source_rows == 3
    assert plan.mappable_rows == 2
    assert plan.unmappable_rows == 2
    assert [(item.name, item.action) for item in plan.array_plans] == [
        ("source_refined_row_ids", "write"),
        ("source_detect_row_index", "write"),
    ]


def test_plan_crop_backfill_skips_existing_mismatch_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = _fixture_zarr(tmp_path)
    _write_array_meta(zarr_path / "crop_runs" / "crop_001" / "source_refined_row_ids", (4,))
    _write_array_meta(zarr_path / "crop_runs" / "crop_001" / "source_detect_row_index", (4,))

    monkeypatch.setattr(
        mod,
        "_open_array",
        lambda path: np.array([0, 1, 2, 3], dtype=np.int64)
        if path.name == "detection_indices"
        else np.array([100, 101, 102], dtype=np.int64),
    )
    monkeypatch.setattr(mod, "_existing_array_equal", lambda path, desired: (True, False))

    plan = mod.plan_crop_backfill(zarr_path, "crop_001", overwrite=False)

    assert plan.status == "skipped"
    assert {item.action for item in plan.array_plans} == {"skip_mismatch"}


def test_plan_crop_backfill_overwrites_existing_mismatch_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = _fixture_zarr(tmp_path)
    _write_array_meta(zarr_path / "crop_runs" / "crop_001" / "source_refined_row_ids", (4,))
    _write_array_meta(zarr_path / "crop_runs" / "crop_001" / "source_detect_row_index", (4,))

    monkeypatch.setattr(
        mod,
        "_open_array",
        lambda path: np.array([0, 1, 2, 3], dtype=np.int64)
        if path.name == "detection_indices"
        else np.array([100, 101, 102], dtype=np.int64),
    )
    monkeypatch.setattr(mod, "_existing_array_equal", lambda path, desired: (True, False))

    plan = mod.plan_crop_backfill(zarr_path, "crop_001", overwrite=True)

    assert plan.status == "planned"
    assert {item.action for item in plan.array_plans} == {"overwrite"}


def test_select_crop_runs_uses_direct_latest_metadata(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    _write_group(zarr_path)
    _write_group(zarr_path / "crop_runs", {"latest_materialized": "crop_b"})
    _write_group(zarr_path / "crop_runs" / "crop_a")
    _write_group(zarr_path / "crop_runs" / "crop_b")

    assert mod._select_crop_runs(zarr_path, requested=(), limit="latest") == ["crop_b"]
