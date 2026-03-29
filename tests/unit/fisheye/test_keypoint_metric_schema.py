from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest
import zarr
import zarr.api.synchronous as zarr_sync_api
import zarr.core.sync as zarr_sync
from zarr.storage import MemoryStore

from fisheye.pose.metric_schema import (
    compute_derived_metric_results,
    ensure_derived_metric_storage,
    metric_schema_from_package,
    resolve_metric_schema_for_group,
)


@pytest.fixture(autouse=True)
def _patch_zarr_sync(monkeypatch):
    def _sync_via_asyncio_run(coro, loop=None, timeout=None):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result = {}
        error = {}

        def _runner():
            try:
                result["value"] = asyncio.run(coro)
            except Exception as exc:  # pragma: no cover - defensive
                error["exc"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()

        if "exc" in error:
            raise error["exc"]
        return result.get("value")

    monkeypatch.setattr(zarr_sync, "sync", _sync_via_asyncio_run)
    monkeypatch.setattr(zarr_sync_api, "sync", _sync_via_asyncio_run)


def test_compute_derived_metric_results_for_traditional_v2_row() -> None:
    schema = metric_schema_from_package("traditional_v2")
    labels = ["swim_bladder", "eye_left", "eye_right", "snout_tip", "tail_tip"]
    points = np.array(
        [
            [1.0, 1.0],  # swim_bladder
            [2.0, 0.0],  # eye_left
            [2.0, 2.0],  # eye_right
            [3.0, 1.0],  # snout_tip
            [0.0, 1.0],  # tail_tip
        ],
        dtype=np.float64,
    )

    result = compute_derived_metric_results(
        points,
        keypoint_labels=labels,
        schema=schema,
        roi_diagonal=5.0,
    )

    np.testing.assert_allclose(result.values, np.array([3.0, 1.0, 2.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(result.values_norm, np.array([0.6, 0.2, 0.4, 0.4], dtype=np.float32))
    np.testing.assert_array_equal(result.valid, np.array([True, True, True, True], dtype=bool))


def test_resolve_metric_schema_for_group_uses_pose_schema(tmp_path) -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    run = root.create_group("refined_keypoints_runs").create_group("refined_v2")
    run.attrs["pose_schema"] = {
        "name": "traditional_v2",
        "skeleton_id": "pose_skel_traditional_v2",
    }

    schema = resolve_metric_schema_for_group(run, required=True)
    assert schema is not None
    assert schema.schema_name == "traditional_v2_derived_metrics"
    assert schema.metric_labels == ["total_length", "tail_length", "head_length", "eye_span"]


def test_ensure_derived_metric_storage_writes_arrays_and_attrs(tmp_path) -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    run = root.create_group("refined_keypoints_runs").create_group("refined_v2")
    schema = metric_schema_from_package("traditional_v2")

    storage = ensure_derived_metric_storage(
        run,
        schema=schema,
        row_count=3,
        chunk_len=2,
        roi_diagonal=10.0,
        overwrite=False,
    )

    assert storage.values.shape == (3, 4)
    assert storage.values_norm.shape == (3, 4)
    assert storage.valid.shape == (3, 4)
    assert run.attrs["derived_metric_schema_id"] == "traditional_v2_derived_metrics"
    assert run.attrs["derived_metric_labels"] == ["total_length", "tail_length", "head_length", "eye_span"]
    assert run.attrs["derived_metric_count"] == 4
