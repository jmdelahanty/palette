from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from fisheye.labeling.task_generation import _detect_review_status_for_zarr


def _write_group(path: Path, attrs: dict[str, object] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": attrs or {},
    }
    (path / "zarr.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _review_status(state: str) -> dict[str, str]:
    return {"state": state, "intended_use": "training"}


@pytest.mark.parametrize(
    "module_name",
    (
        "fisheye.utils.run_megabouts_batch_pipeline",
        "fisheye.utils.run_movement_bout_batch_pipeline",
        "fisheye.utils.run_subject_mask_batch_pipeline",
    ),
)
@pytest.mark.parametrize("include_legacy_attr", (False, True))
def test_batch_latest_group_name_prefers_latest_with_or_without_legacy_attr(
    tmp_path: Path,
    module_name: str,
    include_legacy_attr: bool,
) -> None:
    module = importlib.import_module(module_name)
    parent = tmp_path / "refined_detect_runs"
    attrs = {"latest": "latest_reviewed"}
    if include_legacy_attr:
        attrs["detect_review_status_latest"] = "legacy_reviewed"
    _write_group(parent, attrs)
    _write_group(parent / "legacy_reviewed")
    _write_group(parent / "latest_reviewed")

    assert module._latest_group_name(parent) == "latest_reviewed"


@pytest.mark.parametrize("include_legacy_attr", (False, True))
def test_task_generation_detect_review_status_prefers_latest_with_or_without_legacy_attr(
    tmp_path: Path,
    include_legacy_attr: bool,
) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    parent = zarr_path / "refined_detect_runs"
    attrs = {"latest": "latest_reviewed"}
    if include_legacy_attr:
        attrs["detect_review_status_latest"] = "legacy_reviewed"
    _write_group(parent, attrs)
    _write_group(parent / "legacy_reviewed", {"detect_review_status": _review_status("rejected")})
    _write_group(parent / "latest_reviewed", {"detect_review_status": _review_status("approved")})

    status = _detect_review_status_for_zarr(zarr_path)

    assert status["approved"] is True
    assert status["review_run"] == "latest_reviewed"
    assert status["review_state"] == "approved"
