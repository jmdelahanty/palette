from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr

from fisheye.utils import backfill_detect_review_status as mod


def _make_refined_detect_zarr(path: Path) -> Path:
    root = zarr.open_group(store=path, mode="w")
    parent = root.create_group("refined_detect_runs")
    parent.attrs["latest"] = "refined_1"
    run = parent.create_group("refined_1")
    group = run.create_group("interpolated")
    group.create_array("frame_indices", data=np.asarray([0, 1], dtype=np.int32))
    group.create_array(
        "bbox_norm_coords",
        data=np.asarray([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.1, 0.1]], dtype=np.float64),
    )
    group.create_array("frame_counts", data=np.asarray([1, 1], dtype=np.int32))
    return path


def test_backfill_approved_status_sets_authority_without_legacy_pointer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zarr_path = _make_refined_detect_zarr(tmp_path / "rec.zarr")
    calls: list[dict[str, Any]] = []

    def fake_approve(**kwargs: Any) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {"status": "ok", "reason_code": "OK", "run": kwargs["refined_run_name"]}

    monkeypatch.setattr(mod, "_approve_refined_detect_authority", fake_approve)

    rc = mod.main([str(zarr_path), "--apply", "--reviewer", "operator1"])

    assert rc == 0
    assert len(calls) == 1
    assert calls[0]["zarr_path"] == zarr_path
    assert calls[0]["refined_run_name"] == "refined_1"
    root = zarr.open_group(store=zarr_path, mode="r")
    parent = root["refined_detect_runs"]
    status = dict(parent["refined_1"].attrs["detect_review_status"])
    assert status["state"] == "approved"
    assert status["authoritative_approval"] == {
        "status": "ok",
        "reason_code": "OK",
        "run": "refined_1",
    }
    assert "detect_review_status_latest" not in parent.attrs


def test_backfill_pending_status_does_not_set_authority_or_legacy_pointer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zarr_path = _make_refined_detect_zarr(tmp_path / "rec.zarr")

    def fail_if_called(**_kwargs: Any) -> dict[str, object]:
        raise AssertionError("pending review status should not approve authority")

    monkeypatch.setattr(mod, "_approve_refined_detect_authority", fail_if_called)

    rc = mod.main([str(zarr_path), "--apply", "--state", "pending"])

    assert rc == 0
    root = zarr.open_group(store=zarr_path, mode="r")
    parent = root["refined_detect_runs"]
    status = dict(parent["refined_1"].attrs["detect_review_status"])
    assert status["state"] == "pending"
    assert "authoritative_approval" not in status
    assert "detect_review_status_latest" not in parent.attrs
