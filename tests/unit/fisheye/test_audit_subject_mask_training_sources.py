from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import pytest
import zarr
import zarr.api.synchronous as zarr_sync_api
import zarr.core.sync as zarr_sync
from zarr.storage import MemoryStore

from fisheye.utils.audit_subject_mask_training_sources import (
    audit_selected_subject_mask_training_sources,
)


LABELS = ["subject_body", "eye_left", "eye_right", "swim_bladder"]


@pytest.fixture(autouse=True)
def _patch_zarr_sync(monkeypatch):
    def _sync_via_asyncio_run(coro, loop=None, timeout=None):
        del loop, timeout
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


def _root() -> zarr.Group:
    return zarr.open_group(store=MemoryStore(), mode="w")


def _write_crop(root: zarr.Group) -> None:
    parent = root.create_group("crop_runs")
    parent.attrs["latest"] = "crop_001"
    crop = parent.create_group("crop_001")
    crop.create_array("roi_images", data=np.zeros((3, 8, 8), dtype=np.uint8), chunks=(3, 8, 8))


def _write_subject_run(
    parent: zarr.Group,
    *,
    run_name: str,
    available: tuple[bool, bool, bool, bool],
    review_state: str,
) -> zarr.Group:
    parent.attrs["latest"] = run_name
    run = parent.create_group(run_name)
    run.attrs["source_crop_run"] = "crop_001"
    run.attrs["label_schema_id"] = "subject_v1_lr"
    run.attrs["mask_labels"] = list(LABELS)
    run.attrs["component_review_statuses"] = {
        label: {"state": review_state, "intended_use": "training"}
        for label, is_available in zip(LABELS, available)
        if is_available
    }
    masks = np.zeros((3, 4, 8, 8), dtype=np.uint8)
    present = np.zeros((3, 4), dtype=np.bool_)
    for index, is_available in enumerate(available):
        if not is_available:
            continue
        masks[:, index, 1 + index : 2 + index, 1 + index : 2 + index] = 1
        present[:, index] = True
    run.create_array("masks_roi", data=masks, chunks=(3, 4, 8, 8))
    run.create_array("available_channels", data=np.asarray(available, dtype=np.bool_), chunks=(4,))
    run.require_group("metrics").create_array("mask_present", data=present, chunks=(3, 4))
    return run


def _write_source_root(
    *,
    stage_group: str = "refined_subject_masks_runs",
    available: tuple[bool, bool, bool, bool] = (True, True, True, True),
    review_state: str = "approved",
) -> zarr.Group:
    root = _root()
    _write_crop(root)
    raw_parent = root.create_group("subject_mask_runs")
    _write_subject_run(
        raw_parent,
        run_name="subject_masks_001",
        available=available,
        review_state=review_state,
    )
    if stage_group == "refined_subject_masks_runs":
        refined_parent = root.create_group("refined_subject_masks_runs")
        _write_subject_run(
            refined_parent,
            run_name="refined_subject_masks_001",
            available=available,
            review_state=review_state,
        )
    return root


def _manifest_source(
    path: Path,
    *,
    stage_group: str = "refined_subject_masks_runs",
    available: tuple[bool, bool, bool, bool] = (True, True, True, True),
    review_state: str = "approved",
) -> Mapping[str, object]:
    run_name = "refined_subject_masks_001" if stage_group == "refined_subject_masks_runs" else "subject_masks_001"
    return {
        "name": "source_a",
        "dataset_id": "dataset_a",
        "zarr_path": str(path),
        "source_stage_group": stage_group,
        "source_subject_mask_run": run_name,
        "source_crop_run": "crop_001",
        "label_schema_id": "subject_v1_lr",
        "total_samples": 3,
        "available_components": sorted(label for label, is_available in zip(LABELS, available) if is_available),
        "component_quality": [
            {
                "component_name": label,
                "available": int(is_available),
                "review_state": review_state if is_available else None,
                "review_intended_use": "training" if is_available else None,
                "rows_with_component_mask_rate": 1.0 if is_available else 0.0,
                "lifecycle_state": review_state if is_available else "na",
            }
            for label, is_available in zip(LABELS, available)
        ],
    }


def _opener(roots: Dict[str, zarr.Group]):
    def _open(path: Path) -> zarr.Group:
        return roots[str(path)]

    return _open


def test_audit_subject_mask_training_sources_accepts_refined_manifest_source() -> None:
    path = Path("/tmp/source_a.zarr")
    root = _write_source_root()

    summary = audit_selected_subject_mask_training_sources(
        [_manifest_source(path)],
        open_zarr=_opener({str(path): root}),
    )

    assert summary["valid"] is True
    assert summary["errors"] == []
    assert summary["sources"][0]["available_components"] == sorted(LABELS)
    assert summary["sources"][0]["component_rate_parity_checked"] == sorted(LABELS)


def test_audit_subject_mask_training_sources_reports_availability_mismatch() -> None:
    path = Path("/tmp/source_a.zarr")
    root = _write_source_root(available=(True, True, True, False))

    summary = audit_selected_subject_mask_training_sources(
        [_manifest_source(path, available=(True, True, True, True))],
        open_zarr=_opener({str(path): root}),
    )

    assert summary["valid"] is False
    assert any("available_components mismatch" in message for message in summary["errors"])
    assert any("swim_bladder" in message and "availability mismatch" in message for message in summary["errors"])


def test_audit_subject_mask_training_sources_reports_review_state_mismatch() -> None:
    path = Path("/tmp/source_a.zarr")
    root = _write_source_root(review_state="pending")

    summary = audit_selected_subject_mask_training_sources(
        [_manifest_source(path, review_state="approved")],
        open_zarr=_opener({str(path): root}),
    )

    assert summary["valid"] is False
    assert any("review_state mismatch" in message for message in summary["errors"])
    assert any("not approved for training export" in message for message in summary["errors"])


def test_audit_subject_mask_training_sources_allows_pending_refined_when_requested() -> None:
    path = Path("/tmp/source_a.zarr")
    root = _write_source_root(review_state="pending")

    strict = audit_selected_subject_mask_training_sources(
        [_manifest_source(path, review_state="pending")],
        open_zarr=_opener({str(path): root}),
    )
    allowed = audit_selected_subject_mask_training_sources(
        [_manifest_source(path, review_state="pending")],
        open_zarr=_opener({str(path): root}),
        allow_unapproved_refined=True,
    )

    assert strict["valid"] is False
    assert any("not approved for training export" in message for message in strict["errors"])
    assert allowed["valid"] is True
    assert allowed["errors"] == []


def test_audit_subject_mask_training_sources_accepts_raw_subject_source() -> None:
    path = Path("/tmp/source_raw.zarr")
    root = _write_source_root(stage_group="subject_mask_runs", review_state="pending")

    summary = audit_selected_subject_mask_training_sources(
        [_manifest_source(path, stage_group="subject_mask_runs", review_state="pending")],
        open_zarr=_opener({str(path): root}),
    )

    assert summary["valid"] is True
    assert summary["sources"][0]["source_stage_group"] == "subject_mask_runs"
