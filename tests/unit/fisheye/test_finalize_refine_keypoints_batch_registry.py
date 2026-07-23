from pathlib import Path
import uuid

import pytest
import zarr

from fisheye.refinement import refine_keypoints as refine_mod
from fisheye.shared.zarr_run_completion import mark_run_complete
from fisheye.utils.finalize_refine_keypoints_batch_registry import (
    _select_refined_run,
    finalize_refine_keypoints_batch_registry,
)


class _FakeGroup(dict):
    def __init__(self, *args, attrs=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = dict(attrs or {})

    def group_keys(self):
        return self.keys()


def test_selector_ineligible_diagnostic_cannot_be_registry_finalized() -> None:
    prior = _FakeGroup(
        attrs={
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
            "source_keypoints_run": "keypoints_001",
            "palette_run_completed_at_utc": "2026-01-01T00:00:00+00:00",
        }
    )
    diagnostic = _FakeGroup(
        attrs={
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
            "source_keypoints_run": "keypoints_001",
            "palette_run_completed_at_utc": "2026-02-01T00:00:00+00:00",
        }
    )
    parent = _FakeGroup(
        {"prior": prior, "diagnostic": diagnostic},
        attrs={"latest": "diagnostic"},
    )
    root = _FakeGroup({"refined_keypoints_runs": parent})

    run_name, selected = _select_refined_run(
        root,
        requested_keypoint_run="keypoints_001",
    )

    assert run_name == "prior"
    assert selected is prior


def test_persist_then_raise_diagnostic_creation_is_never_legacy_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.group()
    parent = root.create_group(refine_mod.REFINED_KEYPOINT_GROUP)
    original_create_group = zarr.Group.create_group

    def _persist_then_raise(self, name, *args, **kwargs):
        child = original_create_group(self, name, *args, **kwargs)
        if (
            getattr(self, "path", None) == refine_mod.REFINED_KEYPOINT_GROUP
            and name == "interrupted_diagnostic"
        ):
            raise RuntimeError("injected create acknowledgement failure")
        return child

    monkeypatch.setattr(zarr.Group, "create_group", _persist_then_raise)

    with pytest.raises(
        RuntimeError,
        match="injected create acknowledgement failure",
    ):
        refine_mod._create_refined_keypoint_diagnostic_candidate(
            parent,
            run_name="interrupted_diagnostic",
            started_at_utc="2026-07-20T12:00:00+00:00",
        )

    interrupted = parent["interrupted_diagnostic"]
    uuid.UUID(
        interrupted.attrs[refine_mod.REFINED_KEYPOINT_DIAGNOSTIC_OWNER_ATTR]
    )
    assert interrupted.attrs["palette_run_completion_status"] == "running"
    assert interrupted.attrs["stage_selector_eligible"] is False
    assert interrupted.attrs["coordinate_contract"] == (
        "palette.refined_keypoints.legacy_unverified_nonselector.v1"
    )
    with pytest.raises(
        RuntimeError,
        match="No complete selector-eligible refined keypoint run found",
    ):
        _select_refined_run(root, requested_keypoint_run=None)


def test_blank_unmarked_child_cannot_use_finalizer_legacy_fallback() -> None:
    blank = _FakeGroup(attrs={"source_keypoints_run": "keypoints_001"})
    parent = _FakeGroup({"interrupted": blank}, attrs={"latest": "interrupted"})
    root = _FakeGroup({"refined_keypoints_runs": parent})

    with pytest.raises(
        RuntimeError,
        match="No complete selector-eligible refined keypoint run found",
    ):
        _select_refined_run(root, requested_keypoint_run="keypoints_001")


def test_finalize_refine_keypoints_batch_registry_dry_run_selects_matching_run(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "rec" / "zarr" / "rec_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    parent = root.create_group("refined_keypoints_runs")
    parent.attrs["latest"] = "refined_keypoints_001"
    run = parent.create_group("refined_keypoints_001")
    run.attrs["source_keypoints_run"] = "keypoints_001"
    run.attrs["method"] = "refine_keypoints"
    run.attrs["stage_selector_eligible"] = True
    run.attrs["summary_statistics"] = {
        "total_rois": 10,
        "refined_success": 9,
        "usable_keypoints": 8,
        "pass_rate_percent": 90.0,
    }
    mark_run_complete(run, parent_group=parent, run_name="refined_keypoints_001")

    run_root = tmp_path / "batch"
    run_root.mkdir()
    (run_root / "zarr_paths.txt").write_text(str(zarr_path) + "\n", encoding="utf-8")

    report = finalize_refine_keypoints_batch_registry(
        run_root,
        registry_path=tmp_path / "missing.sqlite",
        keypoint_run="keypoints_001",
        apply=False,
    )

    assert report["status"] == "ok"
    assert report["finalized_count"] == 1
    assert report["upserted_status_rows"] == 0
    assert report["finalized"][0]["run_name"] == "refined_keypoints_001"
    assert report["finalized"][0]["coverage_pct"] == 90.0
