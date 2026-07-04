from __future__ import annotations

import json
from pathlib import Path
import re
from hashlib import sha256
import warnings

import numpy as np
import pytest
import zarr

from fisheye.registry import stage_complete as stage_complete_mod
from fisheye.registry.db import DatasetMetadata, Registry
from fisheye.registry.stage_complete import emit_stage_completion
from fisheye.shared import zarr_run_completion as completion_mod
from fisheye.utils import backfill_completion_epoch as backfill_mod
from fisheye.utils import triage_completion_epoch_blockers as triage_mod
from fisheye.shared.zarr_run_completion import (
    AUTHORITATIVE_RUN_ATTR,
    AUTHORITATIVE_RUN_PROVENANCE_ATTR,
    COMPLETION_EPOCH_ATTR,
    COMPLETION_EPOCH_STRICT,
    RUN_COMPLETED_AT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_RUNNING,
    clear_authoritative_run,
    describe_run_parent,
    effective_legacy_default,
    iter_run_parent_summaries,
    is_run_complete,
    is_run_complete_in_parent,
    mark_run_complete,
    mark_run_failed,
    mark_run_pending,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
    resolve_authoritative_run_name,
    resolve_latest_complete_run_name,
    set_authoritative_run,
)


class FakeGroup(dict[str, object]):
    def __init__(self, *, attrs: dict[str, object] | None = None) -> None:
        super().__init__()
        self.attrs = attrs if attrs is not None else {}

    def group_keys(self):
        return [key for key, value in self.items() if isinstance(value, FakeGroup)]

    def require_group(self, name: str) -> "FakeGroup":
        group: FakeGroup = self
        for part in [piece for piece in str(name).split("/") if piece]:
            child = group.get(part)
            if child is None:
                child = FakeGroup()
                group[part] = child
            if not isinstance(child, FakeGroup):
                raise TypeError(f"{part!r} exists and is not a group")
            group = child
        return group

    @property
    def path(self) -> str:
        return "/fake"


class FailingSetAttrs(dict[str, object]):
    def __init__(self, *, fail_keys: set[str]) -> None:
        super().__init__()
        self.fail_keys = fail_keys

    def __setitem__(self, key: str, value: object) -> None:
        if key in self.fail_keys:
            raise RuntimeError(f"refusing attr write for {key}")
        super().__setitem__(key, value)


class FakeArray:
    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        self.shape = shape
        self.dtype = np.dtype(dtype)


def _add_valid_detect_arrays(run: FakeGroup) -> None:
    run["frame_indices"] = FakeArray((2,), "int32")
    run["bbox_norm_coords"] = FakeArray((2, 4), "float32")
    run["scores"] = FakeArray((2,), "float32")
    run["class_ids"] = FakeArray((2,), "int32")
    run["frame_counts"] = FakeArray((3,), "int32")
    run["n_detections"] = FakeArray((3,), "int32")


def _add_valid_detect_quality_arrays(run: FakeGroup) -> None:
    run["quality_flags"] = FakeArray((3,), "int8")
    run["detection_quality_labels"] = FakeArray((2,), "int8")


def _add_valid_crop_arrays(run: FakeGroup) -> None:
    run["roi_coordinates_full"] = FakeArray((2, 2), "int32")
    run["bbox_norm_coords"] = FakeArray((2, 4), "float32")
    run["frame_indices"] = FakeArray((2,), "int32")
    run["frame_counts"] = FakeArray((3,), "int32")
    run["detection_indices"] = FakeArray((2,), "int32")


def _add_valid_refined_detect_subgroups(run: FakeGroup) -> None:
    source = FakeGroup()
    instances = FakeGroup()
    run["source_detections"] = source
    run["instances"] = instances

    source["source_detect_row_index"] = FakeArray((2,), "int32")
    source["frame_indices"] = FakeArray((2,), "int32")
    source["bbox_img_xyxy"] = FakeArray((2, 4), "float64")
    source["bbox_norm_coords"] = FakeArray((2, 4), "float64")
    source["decision_codes"] = FakeArray((2,), "int8")
    source["resolved_refined_row_id"] = FakeArray((2,), "int64")
    source["reason_bytes"] = FakeArray((2, 16), "uint8")
    source["reason"] = FakeArray((2,), "U16")

    instances["refined_row_ids"] = FakeArray((2,), "int64")
    instances["frame_indices"] = FakeArray((2,), "int32")
    instances["frame_offsets"] = FakeArray((3,), "int64")
    instances["bbox_img_xyxy"] = FakeArray((2, 4), "float64")
    instances["bbox_norm_coords"] = FakeArray((2, 4), "float64")
    instances["source_kind_codes"] = FakeArray((2,), "int8")
    instances["manual_edit_flags"] = FakeArray((2,), "bool")
    instances["source_detect_row_index"] = FakeArray((2,), "int32")
    instances["frame_counts"] = FakeArray((2,), "int32")
    instances["reason_bytes"] = FakeArray((2, 16), "uint8")
    instances["reason"] = FakeArray((2,), "U16")


def _add_valid_refined_keypoints_arrays(run: FakeGroup) -> None:
    run["frame_indices"] = FakeArray((2,), "int32")
    run["frame_counts"] = FakeArray((3,), "int32")
    run["detection_indices"] = FakeArray((2,), "int32")
    run["detection_source"] = FakeArray((2,), "int8")
    run["retune_id"] = FakeArray((2,), "int32")
    run["keypoints_roi"] = FakeArray((2, 3, 2), "float64")
    run["keypoints_img"] = FakeArray((2, 3, 2), "float64")
    run["keypoints_norm"] = FakeArray((2, 3, 2), "float64")
    run["heading"] = FakeArray((2,), "float64")
    run["confidence"] = FakeArray((2,), "float64")
    run["triangle_area"] = FakeArray((2,), "float64")
    run["min_angle"] = FakeArray((2,), "float64")
    run["triangle_angles"] = FakeArray((2, 3), "float64")
    run["quality_labels"] = FakeArray((2,), "int8")
    for name in (
        "refined_success",
        "source_success",
        "flip_corrected",
        "heading_finite",
        "heading_usable",
        "confidence_valid",
        "geometry_valid",
        "usable_keypoints",
    ):
        run[name] = FakeArray((2,), "bool")


def _add_valid_arena_assignment_arrays(run: FakeGroup) -> None:
    run["arena_ids"] = FakeArray((2,), "int32")
    run["n_detections_per_arena"] = FakeArray((3, 2), "int32")


def _add_valid_tracking_arrays(run: FakeGroup) -> None:
    run["track_ids"] = FakeArray((2,), "int32")
    run["arena_ids"] = FakeArray((2,), "int32")
    run["frame_indices"] = FakeArray((2,), "int32")
    run["source_row_indices"] = FakeArray((2,), "int32")
    run["track_ids_present"] = FakeArray((1,), "int32")
    run["track_arena_ids"] = FakeArray((1,), "int32")


def _add_valid_keypoints_arrays(run: FakeGroup) -> None:
    run["frame_indices"] = FakeArray((2,), "int32")
    run["frame_counts"] = FakeArray((3,), "int32")
    run["detection_indices"] = FakeArray((2,), "int32")
    run["keypoints_roi"] = FakeArray((2, 5, 2), "float64")
    run["keypoints_img"] = FakeArray((2, 5, 2), "float64")
    run["keypoints_norm"] = FakeArray((2, 5, 2), "float64")
    run["heading"] = FakeArray((2,), "float64")
    run["confidence"] = FakeArray((2,), "float64")
    run["keypoint_confidences"] = FakeArray((2, 5), "float64")
    run["effective_threshold"] = FakeArray((2,), "float64")
    run["effective_se2_radius"] = FakeArray((2,), "float64")
    run["detection_success"] = FakeArray((2,), "bool")
    run["detection_source"] = FakeArray((2,), "int8")
    run["heading_finite"] = FakeArray((2,), "bool")
    run["heading_usable"] = FakeArray((2,), "bool")
    run["n_keypoints"] = FakeArray((3,), "int32")


def _add_valid_track_kinematics_surface(run: FakeGroup) -> None:
    run["track_ids"] = FakeArray((2,), "int32")
    run["track_arena_ids"] = FakeArray((2,), "int32")
    for attr_name in (
        "track_manifest",
        "provenance",
        "created_at_utc",
        "git_commit",
        "git_branch",
        "method",
        "fps",
        "source_tracking_run",
        "summary",
        "num_tracks",
    ):
        run.attrs[attr_name] = "ok"


_PROMOTED_STAGE_COMPLETION_CASES = (
    pytest.param(
        "detect",
        "detect_runs",
        "detect_001",
        _add_valid_detect_arrays,
        "detect",
        "detect: missing required array 'frame_indices'",
        id="detect",
    ),
    pytest.param(
        "crop",
        "crop_runs",
        "crop_001",
        _add_valid_crop_arrays,
        "crop",
        "crop: missing required array 'roi_coordinates_full'",
        id="crop",
    ),
    pytest.param(
        "refined_keypoints",
        "refined_keypoints_runs",
        "refined_keypoints_001",
        _add_valid_refined_keypoints_arrays,
        "refined_keypoints",
        "refined_keypoints: missing required array 'frame_indices'",
        id="refined_keypoints",
    ),
    pytest.param(
        "arena_assignment",
        "arena_assignment_runs",
        "arena_assignment_001",
        _add_valid_arena_assignment_arrays,
        "arena_assignment",
        "arena_assignment: missing required array 'arena_ids'",
        id="arena_assignment",
    ),
    pytest.param(
        "tracking",
        "tracking_runs",
        "tracking_001",
        _add_valid_tracking_arrays,
        "tracking",
        "tracking: missing required array 'track_ids'",
        id="tracking",
    ),
)


def test_require_runs_parent_stamps_new_empty_parent_strict() -> None:
    root = FakeGroup()

    parent = require_runs_parent(root, "detect_runs")

    assert root["detect_runs"] is parent
    assert parent.attrs[COMPLETION_EPOCH_ATTR] == COMPLETION_EPOCH_STRICT
    assert effective_legacy_default(parent) is False


def test_require_runs_parent_leaves_existing_parent_legacy_until_backfill() -> None:
    root = FakeGroup()
    parent = FakeGroup()
    parent["legacy_run"] = FakeGroup()
    root["detect_runs"] = parent

    resolved = require_runs_parent(root, "detect_runs")

    assert resolved is parent
    assert COMPLETION_EPOCH_ATTR not in parent.attrs
    assert effective_legacy_default(parent) is True


def test_parent_epoch_controls_unmarked_child_completion() -> None:
    parent = FakeGroup(attrs={COMPLETION_EPOCH_ATTR: COMPLETION_EPOCH_STRICT})
    run = FakeGroup()
    parent["run_001"] = run

    assert is_run_complete(run) is True
    assert is_run_complete_in_parent(parent, run) is False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert is_run_complete_in_parent(parent, run, legacy_default=True) is True
    assert resolve_latest_complete_run_name(parent) is None
    assert resolve_latest_complete_run_name(parent, legacy_default=True) == "run_001"


def test_legacy_parent_warns_once_when_accepting_unmarked_child() -> None:
    completion_mod._LEGACY_COMPLETION_WARNING_KEYS.clear()
    parent = FakeGroup()
    parent["run_001"] = FakeGroup()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert resolve_latest_complete_run_name(parent) == "run_001"
        assert resolve_latest_complete_run_name(parent) == "run_001"

    matching = [
        warning
        for warning in caught
        if "treating unmarked child runs as legacy-complete" in str(warning.message)
    ]
    assert len(matching) == 1


def test_strict_parent_stray_unmarked_group_cannot_win_latest_resolution() -> None:
    parent = FakeGroup(
        attrs={
            COMPLETION_EPOCH_ATTR: COMPLETION_EPOCH_STRICT,
            "latest": "zzz_debug",
            "latest_complete": "run_001",
        }
    )
    complete = FakeGroup()
    stray = FakeGroup()
    parent["run_001"] = complete
    parent["zzz_debug"] = stray
    mark_run_complete(complete, parent_group=parent, run_name="run_001")
    parent.attrs["latest"] = "zzz_debug"

    assert resolve_latest_complete_run_name(parent) == "run_001"


def test_latest_resolver_skips_incomplete_contract_run() -> None:
    parent = FakeGroup(attrs={"latest": "new"})
    legacy = FakeGroup()
    new = FakeGroup()
    parent["old"] = legacy
    parent["new"] = new
    mark_run_started(new, run_name="new", stage="detect")

    assert new.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_RUNNING
    assert is_run_complete(new) is False
    assert resolve_latest_complete_run_name(parent) == "old"


def test_mark_complete_publishes_latest_and_latest_complete() -> None:
    parent = FakeGroup()
    run = FakeGroup()
    parent["run_001"] = run
    mark_run_started(run, run_name="run_001", stage="detect")

    mark_run_complete(run, parent_group=parent, run_name="run_001")

    assert is_run_complete(run) is True
    assert parent.attrs["latest"] == "run_001"
    assert parent.attrs["latest_complete"] == "run_001"


def test_latest_resolver_handles_slash_qualified_nested_run_names() -> None:
    parent = FakeGroup()
    run = parent.require_group("offline/run_001")
    mark_run_started(run, run_name="offline/run_001", stage="track_kinematics")

    mark_run_complete(run, parent_group=parent, run_name="offline/run_001")

    assert resolve_latest_complete_run_name(parent) == "offline/run_001"
    summary = describe_run_parent(parent, parent_path="analysis/track_kinematics_runs")
    assert summary["resolved_latest_complete"] == "offline/run_001"
    assert summary["latest_exists"] is True


def test_authoritative_resolver_falls_back_to_latest_when_unset() -> None:
    parent = FakeGroup()
    run = parent.require_group("run_001")
    mark_run_complete(run, parent_group=parent, run_name="run_001")

    assert resolve_authoritative_run_name(parent) == "run_001"


def test_set_authoritative_run_writes_pointer_and_provenance() -> None:
    parent = FakeGroup()
    run = parent.require_group("run_001")
    mark_run_complete(run, parent_group=parent, run_name="run_001")

    provenance = set_authoritative_run(
        parent,
        "run_001",
        approved_by="jeremy",
        approved_at="2026-07-02T12:00:00+00:00",
        git_sha="abc123",
        note="reviewed",
    )

    assert parent.attrs[AUTHORITATIVE_RUN_ATTR] == "run_001"
    assert parent.attrs[AUTHORITATIVE_RUN_PROVENANCE_ATTR] == provenance
    assert provenance == {
        "approved_by": "jeremy",
        "approved_at": "2026-07-02T12:00:00+00:00",
        "git_sha": "abc123",
        "note": "reviewed",
    }
    assert resolve_authoritative_run_name(parent) == "run_001"
    summary = describe_run_parent(parent, parent_path="detect_runs")
    assert summary["authoritative_run"] == "run_001"
    assert summary["resolved_authoritative_run"] == "run_001"
    assert summary["authoritative_run_provenance"] == provenance


def test_clear_authoritative_run_removes_pointer_and_provenance() -> None:
    parent = FakeGroup()
    run = parent.require_group("run_001")
    mark_run_complete(run, parent_group=parent, run_name="run_001")
    set_authoritative_run(parent, "run_001", approved_by="jeremy")

    clear_authoritative_run(parent)

    assert AUTHORITATIVE_RUN_ATTR not in parent.attrs
    assert AUTHORITATIVE_RUN_PROVENANCE_ATTR not in parent.attrs
    assert resolve_authoritative_run_name(parent) == "run_001"


def test_authoritative_resolver_does_not_mutate_parent_attrs() -> None:
    parent = FakeGroup()
    run = parent.require_group("run_001")
    mark_run_complete(run, parent_group=parent, run_name="run_001")
    before = dict(parent.attrs)

    assert resolve_authoritative_run_name(parent) == "run_001"

    assert dict(parent.attrs) == before


def test_newer_complete_run_does_not_change_authoritative_pointer() -> None:
    parent = FakeGroup()
    reviewed = parent.require_group("reviewed")
    smoke = parent.require_group("zzz_smoke")
    mark_run_complete(reviewed, parent_group=parent, run_name="reviewed")
    set_authoritative_run(parent, "reviewed", approved_by="jeremy")

    mark_run_complete(smoke, parent_group=parent, run_name="zzz_smoke")

    assert resolve_latest_complete_run_name(parent) == "zzz_smoke"
    assert resolve_authoritative_run_name(parent) == "reviewed"
    assert parent.attrs[AUTHORITATIVE_RUN_ATTR] == "reviewed"


def test_set_authoritative_run_rejects_missing_or_incomplete_run() -> None:
    parent = FakeGroup()
    pending = parent.require_group("pending")
    mark_run_started(pending, run_name="pending", stage="detect")

    with pytest.raises(ValueError, match="does not exist"):
        set_authoritative_run(parent, "missing", approved_by="jeremy")
    with pytest.raises(ValueError, match="not complete"):
        set_authoritative_run(parent, "pending", approved_by="jeremy")

    assert AUTHORITATIVE_RUN_ATTR not in parent.attrs
    assert AUTHORITATIVE_RUN_PROVENANCE_ATTR not in parent.attrs


def test_set_authoritative_run_rolls_back_provenance_if_pointer_write_fails() -> None:
    parent = FakeGroup(attrs=FailingSetAttrs(fail_keys=set()))
    run = parent.require_group("run_001")
    mark_run_complete(run, parent_group=parent, run_name="run_001")
    parent.attrs.fail_keys.add(AUTHORITATIVE_RUN_ATTR)

    with pytest.raises(RuntimeError, match="refusing attr write"):
        set_authoritative_run(parent, "run_001", approved_by="jeremy")

    assert AUTHORITATIVE_RUN_ATTR not in parent.attrs
    assert AUTHORITATIVE_RUN_PROVENANCE_ATTR not in parent.attrs


def test_note_pending_latest_restores_previous_complete_pointer() -> None:
    parent = FakeGroup(attrs={"latest": "run_001", "latest_complete": "run_001"})
    complete = FakeGroup()
    pending = FakeGroup()
    parent["run_001"] = complete
    parent["run_002"] = pending
    mark_run_complete(complete, parent_group=parent, run_name="run_001")

    parent.attrs["latest"] = "run_002"
    mark_run_started(pending, run_name="run_002", stage="detect")
    note_pending_latest(parent, "run_002")

    assert parent.attrs["latest"] == "run_001"
    assert parent.attrs["latest_pending"] == "run_002"
    assert resolve_latest_complete_run_name(parent) == "run_001"


def test_mark_run_pending_does_not_publish_latest() -> None:
    parent = FakeGroup(attrs={"latest": "run_001", "latest_complete": "run_001"})
    complete = FakeGroup()
    pending = FakeGroup()
    parent["run_001"] = complete
    parent["run_002"] = pending
    mark_run_complete(complete, parent_group=parent, run_name="run_001")
    mark_run_started(pending, run_name="run_002", stage="detect")

    mark_run_pending(parent, "run_002")

    assert parent.attrs["latest"] == "run_001"
    assert parent.attrs["latest_complete"] == "run_001"
    assert parent.attrs["latest_pending"] == "run_002"
    assert resolve_latest_complete_run_name(parent) == "run_001"


def test_describe_run_parent_reports_unsafe_incomplete_latest() -> None:
    parent = FakeGroup(attrs={"latest": "run_002", "latest_complete": "run_001"})
    complete = FakeGroup()
    pending = FakeGroup()
    parent["run_001"] = complete
    parent["run_002"] = pending
    mark_run_complete(complete, parent_group=parent, run_name="run_001")
    parent.attrs["latest"] = "run_002"
    mark_run_started(pending, run_name="run_002", stage="detect")

    summary = describe_run_parent(parent, parent_path="detect_runs")

    assert summary["unsafe"] is True
    assert summary["unsafe_reasons"] == ["latest_incomplete:running"]
    assert summary["resolved_latest_complete"] == "run_001"
    assert summary["incomplete_runs"] == ["run_002"]


def test_describe_run_parent_strict_legacy_reports_latest_without_contract() -> None:
    parent = FakeGroup(attrs={"latest": "legacy"})
    parent["legacy"] = FakeGroup()

    summary = describe_run_parent(parent, parent_path="detect_runs", legacy_default=False)

    assert summary["unsafe"] is True
    assert summary["unsafe_reasons"] == ["latest_incomplete:legacy_missing_contract"]
    assert summary["incomplete_runs"] == ["legacy"]


def test_iter_run_parent_summaries_finds_nested_quality_reports() -> None:
    root = FakeGroup()
    detect_parent = FakeGroup()
    detect_run = FakeGroup()
    quality_parent = FakeGroup(attrs={"latest": "quality_001"})
    quality_run = FakeGroup()
    root["detect_runs"] = detect_parent
    detect_parent["detect_001"] = detect_run
    detect_run["quality_reports"] = quality_parent
    quality_parent["quality_001"] = quality_run
    mark_run_started(quality_run, run_name="quality_001", stage="detect_quality")

    summaries = list(iter_run_parent_summaries(root))

    paths = {summary["parent_path"] for summary in summaries}
    assert "detect_runs/detect_001/quality_reports" in paths
    quality_summary = next(
        summary
        for summary in summaries
        if summary["parent_path"] == "detect_runs/detect_001/quality_reports"
    )
    assert quality_summary["unsafe"] is True
    assert quality_summary["unsafe_reasons"] == ["latest_incomplete:running"]


def test_emit_stage_completion_refuses_incomplete_opted_in_run(tmp_path: Path) -> None:
    root = FakeGroup()
    detect_parent = FakeGroup()
    run = FakeGroup()
    root["detect_runs"] = detect_parent
    detect_parent["detect_001"] = run
    mark_run_started(run, run_name="detect_001", stage="detect")

    class FakeRegistry:
        def close(self) -> None:
            pass

    wrote = emit_stage_completion(
        root,  # type: ignore[arg-type]
        tmp_path / "archive.zarr",
        step_name="detect",
        status="ok",
        source="unit_test",
        run_name="detect_001",
        registry=FakeRegistry(),  # type: ignore[arg-type]
        auto_registry_from_env=False,
        upsert_dataset_row=False,
        metadata=type("Metadata", (), {"dataset_id": "d", "recording_id": "r"})(),
        upsert_step_status_fn=lambda *args, **kwargs: None,
        invalidate_steps_fn=lambda *args, **kwargs: None,
    )

    assert wrote is False


def test_emit_stage_completion_refuses_unmarked_run_under_strict_parent(tmp_path: Path) -> None:
    root = FakeGroup()
    detect_parent = FakeGroup(attrs={COMPLETION_EPOCH_ATTR: COMPLETION_EPOCH_STRICT})
    run = FakeGroup()
    root["detect_runs"] = detect_parent
    detect_parent["detect_001"] = run
    _add_valid_detect_arrays(run)

    class FakeRegistry:
        def close(self) -> None:
            pass

    called = False

    def _upsert(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True

    wrote = emit_stage_completion(
        root,  # type: ignore[arg-type]
        tmp_path / "archive.zarr",
        step_name="detect",
        status="ok",
        source="unit_test",
        run_name="detect_001",
        registry=FakeRegistry(),  # type: ignore[arg-type]
        auto_registry_from_env=False,
        upsert_dataset_row=False,
        metadata=type("Metadata", (), {"dataset_id": "d", "recording_id": "r"})(),
        upsert_step_status_fn=_upsert,
        invalidate_steps_fn=lambda *args, **kwargs: None,
    )

    assert wrote is False
    assert called is False


def test_emit_stage_completion_refuses_unresolved_ok_run(tmp_path: Path) -> None:
    root = FakeGroup()
    root["detect_runs"] = FakeGroup()

    class FakeRegistry:
        def close(self) -> None:
            pass

    called = False

    def _upsert(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True

    wrote = emit_stage_completion(
        root,  # type: ignore[arg-type]
        tmp_path / "archive.zarr",
        step_name="detect",
        status="ok",
        source="unit_test",
        run_name="missing_detect",
        registry=FakeRegistry(),  # type: ignore[arg-type]
        auto_registry_from_env=False,
        upsert_dataset_row=False,
        metadata=type("Metadata", (), {"dataset_id": "d", "recording_id": "r"})(),
        upsert_step_status_fn=_upsert,
        invalidate_steps_fn=lambda *args, **kwargs: None,
    )

    assert wrote is False
    assert called is False


def test_legacy_default_true_production_uses_are_retired() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    pattern = re.compile(r"legacy_default\s*=\s*True")
    matches: list[str] = []
    for path in (repo_root / "src" / "fisheye").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                matches.append(f"{path.relative_to(repo_root)}:{line_no}")

    assert not matches, "\n".join(matches)


def test_raw_runs_parent_creation_uses_completion_helper() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    pattern = re.compile(r"(?:require_group|create_group)\(\s*[\"'][^\"']*_runs[\"']")
    allowed = {
        # finalized_runs is an experiment_index collection table, not a stage run parent.
        "src/fisheye/utils/create_clipped_analysis_zarr.py",
        "src/fisheye/utils/finalize_clipped_detect_refine_workflow.py",
    }
    matches: list[str] = []
    for path in (repo_root / "src" / "fisheye").rglob("*.py"):
        relative = path.relative_to(repo_root).as_posix()
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not pattern.search(line):
                continue
            if relative in allowed and "finalized_runs" in line:
                continue
            matches.append(f"{relative}:{line_no}:{line.strip()}")

    assert not matches, "\n".join(matches)


def test_backfill_marks_valid_legacy_child_before_stamping_parent() -> None:
    parent = FakeGroup()
    child = FakeGroup()
    _add_valid_detect_arrays(child)
    parent["detect_001"] = child

    dry_report = backfill_mod._summarize_parent(
        "detect_runs",
        parent,
        apply=False,
        timestamp_utc="2026-06-10T00:00:00+00:00",
    )

    assert dry_report["status"] == "would_stamp"
    assert dry_report["would_mark_child_count"] == 1
    assert dry_report["marked_child_count"] == 0
    assert COMPLETION_EPOCH_ATTR not in parent.attrs
    assert RUN_COMPLETION_STATUS_ATTR not in child.attrs

    apply_report = backfill_mod._summarize_parent(
        "detect_runs",
        parent,
        apply=True,
        timestamp_utc="2026-06-10T00:00:00+00:00",
    )

    assert apply_report["status"] == "stamped"
    assert apply_report["would_mark_child_count"] == 1
    assert apply_report["marked_child_count"] == 1
    assert child.attrs[RUN_COMPLETION_STATUS_ATTR] == "complete"
    assert child.attrs["palette_run_stage"] == "detect"
    assert parent.attrs[COMPLETION_EPOCH_ATTR] == COMPLETION_EPOCH_STRICT
    completed_at = child.attrs[RUN_COMPLETED_AT_ATTR]

    second_apply_report = backfill_mod._summarize_parent(
        "detect_runs",
        parent,
        apply=True,
        timestamp_utc="2026-06-11T00:00:00+00:00",
    )

    assert second_apply_report["status"] == "already_strict"
    assert second_apply_report["would_mark_child_count"] == 0
    assert second_apply_report["marked_child_count"] == 0
    assert child.attrs[RUN_COMPLETED_AT_ATTR] == completed_at


def test_backfill_ignores_nonlatest_invalid_legacy_child_when_stamping_parent() -> None:
    parent = FakeGroup(attrs={"latest": "detect_good", "latest_complete": "detect_good"})
    good = FakeGroup()
    bad = FakeGroup()
    _add_valid_detect_arrays(good)
    parent["detect_good"] = good
    parent["detect_old_bad"] = bad

    dry_report = backfill_mod._summarize_parent(
        "detect_runs",
        parent,
        apply=False,
        timestamp_utc="2026-06-10T00:00:00+00:00",
    )

    assert dry_report["status"] == "would_stamp"
    assert dry_report["unverified_child_count"] == 0
    assert dry_report["ignored_legacy_child_count"] == 1
    assert dry_report["would_mark_child_count"] == 1
    by_name = {child["run_name"]: child for child in dry_report["children"]}
    assert by_name["detect_old_bad"]["verification"] == "invalid"
    assert by_name["detect_old_bad"]["ignored_for_parent_epoch"] is True
    assert by_name["detect_old_bad"]["ignore_reason"] == "non_latest_legacy_contract_mismatch"

    apply_report = backfill_mod._summarize_parent(
        "detect_runs",
        parent,
        apply=True,
        timestamp_utc="2026-06-10T00:00:00+00:00",
    )

    assert apply_report["status"] == "stamped"
    assert apply_report["ignored_legacy_child_count"] == 1
    assert parent.attrs[COMPLETION_EPOCH_ATTR] == COMPLETION_EPOCH_STRICT
    assert good.attrs[RUN_COMPLETION_STATUS_ATTR] == "complete"
    assert RUN_COMPLETION_STATUS_ATTR not in bad.attrs
    assert is_run_complete_in_parent(parent, good) is True
    assert is_run_complete_in_parent(parent, bad) is False


def test_backfill_blocks_parent_when_latest_legacy_child_is_invalid() -> None:
    parent = FakeGroup(attrs={"latest": "detect_bad"})
    good = FakeGroup()
    bad = FakeGroup()
    _add_valid_detect_arrays(good)
    parent["detect_good"] = good
    parent["detect_bad"] = bad

    report = backfill_mod._summarize_parent(
        "detect_runs",
        parent,
        apply=True,
        timestamp_utc="2026-06-10T00:00:00+00:00",
    )

    assert report["status"] == "blocked"
    assert report["unverified_child_count"] == 1
    assert report["ignored_legacy_child_count"] == 0
    assert report["would_mark_child_count"] == 0
    assert COMPLETION_EPOCH_ATTR not in parent.attrs
    assert RUN_COMPLETION_STATUS_ATTR not in good.attrs
    assert RUN_COMPLETION_STATUS_ATTR not in bad.attrs


def test_backfill_write_failure_leaves_parent_epoch_unstamped() -> None:
    parent = FakeGroup(attrs=FailingSetAttrs(fail_keys={COMPLETION_EPOCH_ATTR}))
    child = FakeGroup()
    _add_valid_detect_arrays(child)
    parent["detect_001"] = child

    report = backfill_mod._summarize_parent(
        "detect_runs",
        parent,
        apply=True,
        timestamp_utc="2026-06-10T00:00:00+00:00",
    )

    assert report["status"] == "write_failed"
    assert report["write_error_phase"] == "stamp_parent_epoch"
    assert report["write_error_type"] == "RuntimeError"
    assert report["would_mark_child_count"] == 1
    assert report["marked_child_count"] == 1
    assert child.attrs[RUN_COMPLETION_STATUS_ATTR] == "complete"
    assert COMPLETION_EPOCH_ATTR not in parent.attrs


def test_backfill_blocks_parent_without_stage_spec() -> None:
    parent = FakeGroup()
    parent["legacy_001"] = FakeGroup()

    report = backfill_mod._summarize_parent(
        "analysis/custom_runs",
        parent,
        apply=True,
        timestamp_utc="2026-06-10T00:00:00+00:00",
    )

    assert report["status"] == "blocked"
    assert report["stage_spec_available"] is False
    assert report["unverified_child_count"] == 1
    assert COMPLETION_EPOCH_ATTR not in parent.attrs


def test_backfill_stamps_parent_with_already_contracted_incomplete_children() -> None:
    parent = FakeGroup()
    running = FakeGroup()
    failed = FakeGroup()
    parent["running_001"] = running
    parent["failed_001"] = failed
    mark_run_started(running, run_name="running_001", stage="detect")
    mark_run_started(failed, run_name="failed_001", stage="detect")
    mark_run_failed(failed, error="boom")

    report = backfill_mod._summarize_parent(
        "detect_runs",
        parent,
        apply=True,
        timestamp_utc="2026-06-10T00:00:00+00:00",
    )

    assert report["status"] == "stamped"
    assert parent.attrs[COMPLETION_EPOCH_ATTR] == COMPLETION_EPOCH_STRICT
    assert report["marked_child_count"] == 0
    by_name = {child["run_name"]: child for child in report["children"]}
    assert by_name["running_001"]["verification"] == "has_contract"
    assert by_name["running_001"]["completion_status"] == "running"
    assert by_name["failed_001"]["verification"] == "has_contract"
    assert by_name["failed_001"]["completion_status"] == "failed"
    assert is_run_complete_in_parent(parent, running) is False
    assert is_run_complete_in_parent(parent, failed) is False


def test_backfill_keeps_heterogeneous_analysis_families_unmapped_until_validated() -> None:
    # These legacy families have multiple historical layouts in real stores.
    # They should stay no-spec/deferred until they get layout-specific validators.
    for parent_path in (
        "analysis/swim_bout_runs",
        "analysis/stimulus_response_runs",
    ):
        parent = FakeGroup()
        parent["legacy_001"] = FakeGroup()

        report = backfill_mod._summarize_parent(
            parent_path,
            parent,
            apply=False,
            timestamp_utc="2026-06-10T00:00:00+00:00",
        )

        assert report["status"] == "blocked"
        assert report["stage"] is None
        assert report["stage_spec_available"] is False
        assert report["children"][0]["reason"] == "no_stage_array_spec"


def test_backfill_parent_filters_skip_nonmatching_parent_without_mutation() -> None:
    parent = FakeGroup()
    child = FakeGroup()
    parent["detect_001"] = child

    filtered = backfill_mod._summarize_parent(
        "detect_runs",
        parent,
        apply=True,
        timestamp_utc="2026-06-10T00:00:00+00:00",
        selected_stages=frozenset({"crop"}),
    )

    assert filtered["status"] == "filtered"
    assert filtered["reason"] == "filter_mismatch"
    assert filtered["stage"] == "detect"
    assert filtered["filter"] == {"stages": ["crop"], "parent_paths": []}
    assert COMPLETION_EPOCH_ATTR not in parent.attrs
    assert RUN_COMPLETION_STATUS_ATTR not in child.attrs

    selected = backfill_mod._summarize_parent(
        "detect_runs",
        parent,
        apply=False,
        timestamp_utc="2026-06-10T00:00:00+00:00",
        selected_stages=frozenset({"detect"}),
    )

    assert selected["status"] == "blocked"
    assert selected["unverified_child_count"] == 1


def test_backfill_apply_requires_filter_unless_broad_apply_is_explicit() -> None:
    assert (
        backfill_mod._apply_filter_error(
            apply=True,
            stages=[],
            parent_paths=[],
            allow_broad_apply=False,
        )
        is not None
    )
    assert (
        backfill_mod._apply_filter_error(
            apply=True,
            stages=["subject_masks"],
            parent_paths=[],
            allow_broad_apply=False,
        )
        is None
    )
    assert (
        backfill_mod._apply_filter_error(
            apply=True,
            stages=[],
            parent_paths=[],
            allow_broad_apply=True,
        )
        is None
    )
    assert (
        backfill_mod._apply_filter_error(
            apply=False,
            stages=[],
            parent_paths=[],
            allow_broad_apply=False,
        )
        is None
    )


def test_backfill_post_apply_expectations_require_apply(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _fake_discover(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("zarr discovery should not run for invalid apply-only expectations")

    def _fake_backfill(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("backfill should not run for invalid apply-only expectations")

    monkeypatch.setattr(backfill_mod, "_discover_zarrs", _fake_discover)
    monkeypatch.setattr(backfill_mod, "backfill_completion_epoch", _fake_backfill)

    try:
        backfill_mod.main(
            [
                "--recordings-root",
                str(tmp_path / "recordings"),
                "--stage",
                "detect",
                "--expect-applied-stamped-parent-count",
                "1",
            ]
        )
    except SystemExit as exc:
        assert "--expect-applied-stamped-parent-count can only be used with --apply" in str(exc)
    else:
        raise AssertionError("expected SystemExit from apply-only expectation without --apply")


def test_backfill_expected_count_errors_report_drift() -> None:
    report = {
        "store_count": 2,
        "non_ok_store_count": 0,
        "blocked_parent_count": 0,
        "filtered_parent_count": 5,
        "would_stamp_parent_count": 7,
        "write_failed_parent_count": 0,
        "would_mark_child_count": 11,
        "ignored_legacy_child_count": 3,
    }

    assert backfill_mod._expected_count_errors(
        report,
        {
            "store_count": 2,
            "non_ok_store_count": 0,
            "blocked_parent_count": 0,
            "filtered_parent_count": 5,
            "would_stamp_parent_count": 7,
            "write_failed_parent_count": 0,
            "would_mark_child_count": 11,
            "ignored_legacy_child_count": 3,
        },
    ) == []
    assert backfill_mod._expected_count_errors(
        report,
        {
            "blocked_parent_count": 1,
            "non_ok_store_count": 1,
            "would_stamp_parent_count": 8,
            "write_failed_parent_count": 1,
            "would_mark_child_count": 12,
            "ignored_legacy_child_count": 4,
        },
    ) == [
        "blocked_parent_count: expected 1, observed 0",
        "non_ok_store_count: expected 1, observed 0",
        "would_stamp_parent_count: expected 8, observed 7",
        "write_failed_parent_count: expected 1, observed 0",
        "would_mark_child_count: expected 12, observed 11",
        "ignored_legacy_child_count: expected 4, observed 3",
    ]


def _fake_backfill_report(
    *,
    apply: bool,
    blocked_parent_count: int = 0,
    filtered_parent_count: int = 0,
    would_stamp_parent_count: int = 0,
    stamped_parent_count: int = 0,
    non_ok_store_count: int = 0,
    write_failed_parent_count: int = 0,
    would_mark_child_count: int = 0,
    marked_child_count: int = 0,
) -> dict[str, object]:
    blocked_parent = {
        "parent_path": "detect_runs",
        "status": "blocked",
        "stage": "detect",
        "stage_spec_available": True,
        "child_count": 1,
        "verified_child_count": 0,
        "unverified_child_count": 1,
        "latest": "detect_bad",
        "latest_complete": None,
        "children": [
            {
                "run_name": "detect_bad",
                "verification": "invalid",
                "errors": ["detect: missing required array 'frame_indices'"],
            }
        ],
    }
    write_failed_parent = {
        "parent_path": "detect_runs",
        "status": "write_failed",
        "stage": "detect",
        "stage_spec_available": True,
        "child_count": 1,
        "verified_child_count": 1,
        "unverified_child_count": 0,
        "marked_child_count": 1,
        "latest": "detect_001",
        "latest_complete": None,
        "write_error_phase": "stamp_parent_epoch",
        "write_error_type": "RuntimeError",
        "children": [
            {
                "run_name": "detect_001",
                "verification": "validated_legacy_complete",
                "marked_complete": True,
            }
        ],
    }
    parents = []
    if blocked_parent_count:
        parents.extend(blocked_parent.copy() for _ in range(blocked_parent_count))
    if write_failed_parent_count:
        parents.extend(write_failed_parent.copy() for _ in range(write_failed_parent_count))
    summary = backfill_mod._build_summary(
        [
            {
                "zarr_path": "/tmp/fake.zarr",
                "parents": parents,
            }
        ]
    )
    return {
        "schema_id": "palette.backfill_completion_epoch_report.v1",
        "timestamp_utc": "2026-06-10T00:00:00+00:00",
        "apply": apply,
        "filters": {"stages": ["detect"], "parent_paths": []},
        "store_count": 1,
        "ok_store_count": 1,
        "non_ok_store_count": non_ok_store_count,
        "blocked_parent_count": blocked_parent_count,
        "filtered_parent_count": filtered_parent_count,
        "would_stamp_parent_count": would_stamp_parent_count,
        "stamped_parent_count": stamped_parent_count,
        "write_failed_parent_count": write_failed_parent_count,
        "would_mark_child_count": would_mark_child_count,
        "marked_child_count": marked_child_count,
        "summary": summary,
        "stores": [
            {
                "zarr_path": "/tmp/fake.zarr",
                "status": "ok",
                "parent_count": len(parents),
                "non_ok_store_count": 0,
                "blocked_parent_count": blocked_parent_count,
                "filtered_parent_count": filtered_parent_count,
                "would_stamp_parent_count": would_stamp_parent_count,
                "stamped_parent_count": stamped_parent_count,
                "write_failed_parent_count": write_failed_parent_count,
                "would_mark_child_count": would_mark_child_count,
                "marked_child_count": marked_child_count,
                "parents": parents,
            }
        ],
    }


def test_backfill_main_apply_aborts_on_blocked_preflight_without_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[bool] = []

    def _fake_backfill(*_args, apply: bool, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(bool(apply))
        if apply:
            raise AssertionError("apply call should not run after blocked preflight")
        return _fake_backfill_report(apply=False, blocked_parent_count=1)

    monkeypatch.setattr(backfill_mod, "backfill_completion_epoch", _fake_backfill)

    output_json = tmp_path / "blocked_preflight.json"
    blocked_jsonl = tmp_path / "blocked_preflight.jsonl"
    try:
        backfill_mod.main(
            [
                str(tmp_path / "fake.zarr"),
                "--stage",
                "detect",
                "--apply",
                "--output-json",
                str(output_json),
                "--blocked-jsonl",
                str(blocked_jsonl),
                "--no-stdout",
            ]
        )
    except SystemExit as exc:
        assert "Refusing --apply because selected scope contains blocked parents" in str(exc)
    else:
        raise AssertionError("expected SystemExit from blocked apply preflight")

    assert calls == [False]
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["apply_aborted"] is True
    assert payload["apply_abort_reason"] == "blocked_parents_present"
    rows = [json.loads(line) for line in blocked_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["parent_path"] == "detect_runs"


def test_backfill_main_apply_aborts_on_expectation_drift_without_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[bool] = []

    def _fake_backfill(*_args, apply: bool, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(bool(apply))
        if apply:
            raise AssertionError("apply call should not run after drift preflight")
        return _fake_backfill_report(
            apply=False,
            blocked_parent_count=0,
            filtered_parent_count=2,
            would_stamp_parent_count=3,
        )

    monkeypatch.setattr(backfill_mod, "backfill_completion_epoch", _fake_backfill)

    output_json = tmp_path / "drift_preflight.json"
    try:
        backfill_mod.main(
            [
                str(tmp_path / "fake.zarr"),
                "--stage",
                "detect",
                "--apply",
                "--expect-would-stamp-parent-count",
                "4",
                "--output-json",
                str(output_json),
                "--summary-only",
                "--no-stdout",
            ]
        )
    except SystemExit as exc:
        assert "preflight counts differ" in str(exc)
    else:
        raise AssertionError("expected SystemExit from drifted apply preflight")

    assert calls == [False]
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["apply_aborted"] is True
    assert payload["apply_abort_reason"] == "expectation_mismatch"
    assert payload["expectation_errors"] == [
        "would_stamp_parent_count: expected 4, observed 3"
    ]


def test_backfill_main_apply_aborts_on_non_ok_store_preflight_without_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[bool] = []

    def _fake_backfill(*_args, apply: bool, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(bool(apply))
        if apply:
            raise AssertionError("apply call should not run after non-ok store preflight")
        return _fake_backfill_report(
            apply=False,
            blocked_parent_count=0,
            filtered_parent_count=0,
            would_stamp_parent_count=0,
            non_ok_store_count=1,
        )

    monkeypatch.setattr(backfill_mod, "backfill_completion_epoch", _fake_backfill)

    output_json = tmp_path / "non_ok_store_preflight.json"
    try:
        backfill_mod.main(
            [
                str(tmp_path / "missing.zarr"),
                "--stage",
                "detect",
                "--apply",
                "--allow-blocked-apply",
                "--output-json",
                str(output_json),
                "--summary-only",
                "--no-stdout",
            ]
        )
    except SystemExit as exc:
        assert "one or more target zarr stores are missing" in str(exc)
    else:
        raise AssertionError("expected SystemExit from non-ok store apply preflight")

    assert calls == [False]
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["apply_aborted"] is True
    assert payload["apply_abort_reason"] == "non_ok_stores_present"
    assert payload["preflight_counts"]["non_ok_store_count"] == 1


def test_backfill_main_allow_blocked_apply_is_explicit_partial_apply_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[bool] = []

    def _fake_backfill(*_args, apply: bool, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(bool(apply))
        return _fake_backfill_report(
            apply=apply,
            blocked_parent_count=1,
            filtered_parent_count=0,
            would_stamp_parent_count=2 if not apply else 0,
        )

    monkeypatch.setattr(backfill_mod, "backfill_completion_epoch", _fake_backfill)

    output_json = tmp_path / "allowed_blocked_apply.json"
    rc = backfill_mod.main(
        [
            str(tmp_path / "fake.zarr"),
            "--stage",
            "detect",
            "--apply",
            "--allow-blocked-apply",
            "--expect-blocked-parent-count",
            "1",
            "--expect-would-stamp-parent-count",
            "2",
            "--output-json",
            str(output_json),
            "--no-stdout",
        ]
    )

    assert rc == 0
    assert calls == [False, True]
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["apply"] is True
    assert "apply_aborted" not in payload
    assert payload["expected_counts"] == {
        "blocked_parent_count": 1,
        "would_stamp_parent_count": 2,
    }
    assert payload["preflight_counts"]["blocked_parent_count"] == 1
    assert payload["preflight_counts"]["would_stamp_parent_count"] == 2
    assert payload["preflight_counts"]["stamped_parent_count"] == 0


def test_backfill_main_apply_exits_nonzero_on_write_failure_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[bool] = []

    def _fake_backfill(*_args, apply: bool, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(bool(apply))
        return _fake_backfill_report(
            apply=apply,
            blocked_parent_count=0,
            filtered_parent_count=0,
            would_stamp_parent_count=1 if not apply else 0,
            write_failed_parent_count=1 if apply else 0,
        )

    monkeypatch.setattr(backfill_mod, "backfill_completion_epoch", _fake_backfill)

    output_json = tmp_path / "write_failed_apply.json"
    write_failed_jsonl = tmp_path / "write_failed_apply.jsonl"
    try:
        backfill_mod.main(
            [
                str(tmp_path / "fake.zarr"),
                "--stage",
                "detect",
                "--apply",
                "--expect-would-stamp-parent-count",
                "1",
                "--output-json",
                str(output_json),
                "--write-failed-jsonl",
                str(write_failed_jsonl),
                "--summary-only",
                "--no-stdout",
            ]
        )
    except SystemExit as exc:
        assert "failed during attr writes" in str(exc)
    else:
        raise AssertionError("expected SystemExit from write-failed apply")

    assert calls == [False, True]
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["apply_failed"] is True
    assert payload["apply_failure_reason"] == "write_failed_parents_present"
    assert payload["write_failed_parent_count"] == 1
    assert payload["preflight_counts"]["would_stamp_parent_count"] == 1
    rows = [
        json.loads(line)
        for line in write_failed_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["schema_id"] == "palette.backfill_completion_epoch_write_failed_parent.v1"
    assert rows[0]["parent_path"] == "detect_runs"
    assert rows[0]["stage"] == "detect"
    assert rows[0]["write_error_phase"] == "stamp_parent_epoch"


def test_backfill_main_apply_exits_nonzero_on_post_apply_count_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[bool] = []

    def _fake_backfill(*_args, apply: bool, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(bool(apply))
        if not apply:
            return _fake_backfill_report(
                apply=False,
                blocked_parent_count=0,
                filtered_parent_count=0,
                would_stamp_parent_count=2,
            )
        return _fake_backfill_report(
            apply=True,
            blocked_parent_count=0,
            filtered_parent_count=0,
            stamped_parent_count=1,
            marked_child_count=4,
        )

    monkeypatch.setattr(backfill_mod, "backfill_completion_epoch", _fake_backfill)

    output_json = tmp_path / "post_apply_drift.json"
    try:
        backfill_mod.main(
            [
                str(tmp_path / "fake.zarr"),
                "--stage",
                "detect",
                "--apply",
                "--expect-would-stamp-parent-count",
                "2",
                "--expect-applied-stamped-parent-count",
                "2",
                "--expect-applied-marked-child-count",
                "5",
                "--output-json",
                str(output_json),
                "--summary-only",
                "--no-stdout",
            ]
        )
    except SystemExit as exc:
        assert "Post-apply counts differed" in str(exc)
    else:
        raise AssertionError("expected SystemExit from post-apply count drift")

    assert calls == [False, True]
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["apply_failed"] is True
    assert payload["apply_failure_reason"] == "post_apply_expectation_mismatch"
    assert payload["post_apply_expected_counts"] == {
        "stamped_parent_count": 2,
        "marked_child_count": 5,
    }
    assert payload["post_apply_expectation_errors"] == [
        "stamped_parent_count: expected 2, observed 1",
        "marked_child_count: expected 5, observed 4",
    ]
    assert payload["preflight_counts"]["would_stamp_parent_count"] == 2


def test_backfill_summary_counts_blocked_and_stampable_parents() -> None:
    blocked = {
        "parent_path": "analysis/custom_runs",
        "status": "blocked",
        "stage": None,
        "child_count": 1,
        "unverified_child_count": 1,
        "children": [
            {
                "run_name": "legacy_001",
                "verification": "unverified",
                "reason": "no_stage_array_spec",
            }
        ],
    }
    invalid = {
        "parent_path": "detect_runs",
        "status": "blocked",
        "stage": "detect",
        "child_count": 1,
        "unverified_child_count": 1,
        "children": [
            {
                "run_name": "bad_detect",
                "verification": "invalid",
                "errors": ["detect: missing required array 'frame_indices'"],
            }
        ],
    }
    deprecated = {
        "parent_path": "eye_masks_runs",
        "status": "blocked",
        "stage": "eye_masks",
        "child_count": 1,
        "unverified_child_count": 1,
        "children": [
            {
                "run_name": "eye_masks_001",
                "verification": "invalid",
                "errors": ["eye_masks: missing required array 'masks_roi'"],
            }
        ],
    }
    stampable = {
        "parent_path": "crop_runs",
        "status": "would_stamp",
        "stage": "crop",
        "child_count": 1,
        "unverified_child_count": 0,
        "latest": "crop_good",
        "latest_complete": "crop_good",
        "children": [
            {
                "run_name": "crop_old_bad",
                "verification": "invalid",
                "ignored_for_parent_epoch": True,
                "ignore_reason": "non_latest_legacy_contract_mismatch",
                "errors": ["crop: missing required array 'bbox_norm_coords'"],
            }
        ],
    }
    write_failed = {
        "parent_path": "detect_runs",
        "status": "write_failed",
        "stage": "detect",
        "child_count": 1,
        "marked_child_count": 1,
        "write_error_phase": "stamp_parent_epoch",
        "write_error_type": "RuntimeError",
        "children": [],
    }

    summary = backfill_mod._build_summary(
        [
            {
                "zarr_path": "/tmp/example.zarr",
                "parents": [blocked, invalid, deprecated, stampable, write_failed],
            }
        ]
    )

    assert summary["parent_status_counts"] == {
        "blocked": 3,
        "would_stamp": 1,
        "write_failed": 1,
    }
    assert summary["blocked_parent_counts_by_stage_or_path"] == {
        "analysis/custom_runs": 1,
        "detect": 1,
        "eye_masks": 1,
    }
    assert summary["would_stamp_parent_counts_by_stage_or_path"] == {"crop": 1}
    assert summary["write_failed_parent_counts_by_stage_or_path"] == {"detect": 1}
    assert summary["write_failed_parent_ranked_by_stage_or_path"] == [
        {"key": "detect", "count": 1},
    ]
    assert summary["write_failed_parent_examples"] == [
        {
            "zarr_path": "/tmp/example.zarr",
            "parent_path": "detect_runs",
            "stage": "detect",
            "child_count": 1,
            "marked_child_count": 1,
            "write_error_phase": "stamp_parent_epoch",
            "write_error_run_name": None,
            "write_error_type": "RuntimeError",
        }
    ]
    assert summary["blocked_child_reason_counts"] == {"invalid": 2, "no_stage_array_spec": 1}
    assert summary["blocked_child_first_error_counts_top50"] == {
        "detect: missing required array 'frame_indices'": 1,
        "eye_masks: missing required array 'masks_roi'": 1,
    }
    assert summary["ignored_legacy_child_reason_counts"] == {"invalid": 1}
    assert summary["ignored_legacy_child_first_error_counts_top50"] == {
        "crop: missing required array 'bbox_norm_coords'": 1,
    }
    assert summary["ignored_legacy_parent_examples"] == [
        {
            "zarr_path": "/tmp/example.zarr",
            "parent_path": "crop_runs",
            "stage": "crop",
            "latest": "crop_good",
            "latest_complete": "crop_good",
            "ignored_legacy_child_count": 1,
        }
    ]
    assert summary["blocked_parent_ranked_by_stage_or_path"] == [
        {"key": "analysis/custom_runs", "count": 1},
        {"key": "detect", "count": 1},
        {"key": "eye_masks", "count": 1},
    ]
    assert summary["would_stamp_parent_ranked_by_stage_or_path"] == [
        {"key": "crop", "count": 1},
    ]
    assert summary["blocked_child_first_error_ranked_top50"] == [
        {"key": "detect: missing required array 'frame_indices'", "count": 1},
        {"key": "eye_masks: missing required array 'masks_roi'", "count": 1},
    ]
    assert summary["backfill_scope_plan"] == [
        {
            "key": "detect",
            "recommendation": "write_failed_retry_after_fix",
            "blocked_parent_count": 1,
            "would_stamp_parent_count": 0,
            "stamped_parent_count": 0,
            "write_failed_parent_count": 1,
            "filter": {"stage": "detect"},
        },
        {
            "key": "crop",
            "recommendation": "ready_to_apply_if_approved",
            "blocked_parent_count": 0,
            "would_stamp_parent_count": 1,
            "stamped_parent_count": 0,
            "write_failed_parent_count": 0,
            "filter": {"stage": "crop"},
        },
        {
            "key": "analysis/custom_runs",
            "recommendation": "blocked_triage_required",
            "blocked_parent_count": 1,
            "would_stamp_parent_count": 0,
            "stamped_parent_count": 0,
            "write_failed_parent_count": 0,
            "filter": {"parent_path": "analysis/custom_runs"},
        },
        {
            "key": "eye_masks",
            "recommendation": "deprecated_scope_not_backfilled",
            "blocked_parent_count": 1,
            "would_stamp_parent_count": 0,
            "stamped_parent_count": 0,
            "write_failed_parent_count": 0,
            "filter": {"stage": "eye_masks"},
        },
    ]


def test_backfill_verifies_nested_track_kinematics_children() -> None:
    parent = FakeGroup()
    run = parent.require_group("offline/tk_001")
    _add_valid_track_kinematics_surface(run)

    report = backfill_mod._summarize_parent(
        "analysis/track_kinematics_runs",
        parent,
        apply=False,
        timestamp_utc="2026-06-10T00:00:00+00:00",
    )

    assert report["status"] == "would_stamp"
    assert report["stage"] == "track_kinematics"
    assert report["child_count"] == 1
    assert report["children"][0]["run_name"] == "offline/tk_001"
    assert report["children"][0]["verification"] == "validated_legacy_complete"


def test_backfill_apply_marks_nested_track_kinematics_children_complete() -> None:
    parent = FakeGroup()
    run = parent.require_group("offline/tk_001")
    _add_valid_track_kinematics_surface(run)

    report = backfill_mod._summarize_parent(
        "analysis/track_kinematics_runs",
        parent,
        apply=True,
        timestamp_utc="2026-06-10T00:00:00+00:00",
    )

    assert report["status"] == "stamped"
    assert parent.attrs[COMPLETION_EPOCH_ATTR] == COMPLETION_EPOCH_STRICT
    assert run.attrs[RUN_COMPLETION_STATUS_ATTR] == "complete"
    assert run.attrs["palette_run_name"] == "offline/tk_001"


def test_backfill_blocked_parent_jsonl_rows_are_compact_for_triage(tmp_path: Path) -> None:
    report = {
        "timestamp_utc": "2026-06-10T00:00:00+00:00",
        "apply": False,
        "stores": [
            {
                "zarr_path": "/tmp/example.zarr",
                "parents": [
                    {
                        "parent_path": "detect_runs",
                        "status": "blocked",
                        "stage": "detect",
                        "stage_spec_available": True,
                        "child_count": 4,
                        "verified_child_count": 1,
                        "unverified_child_count": 2,
                        "ignored_legacy_child_count": 1,
                        "latest": "bad_detect",
                        "latest_complete": "good_detect",
                        "children": [
                            {
                                "run_name": "bad_detect",
                                "verification": "invalid",
                                "errors": ["detect: missing required array 'frame_indices'"],
                                "warnings": ["optional array missing"],
                            },
                            {
                                "run_name": "custom",
                                "verification": "unverified",
                                "reason": "no_stage_array_spec",
                            },
                            {
                                "run_name": "old_ignored_detect",
                                "verification": "invalid",
                                "ignored_for_parent_epoch": True,
                                "ignore_reason": "non_latest_legacy_contract_mismatch",
                                "errors": ["detect: missing required array 'scores'"],
                            },
                            {
                                "run_name": "good_detect",
                                "verification": "has_contract",
                            },
                        ],
                    },
                    {
                        "parent_path": "crop_runs",
                        "status": "would_stamp",
                        "stage": "crop",
                        "children": [],
                    },
                ],
            }
        ],
    }

    rows = backfill_mod._blocked_parent_rows(report, max_children=1)

    assert len(rows) == 1
    assert rows[0]["zarr_path"] == "/tmp/example.zarr"
    assert rows[0]["parent_path"] == "detect_runs"
    assert rows[0]["blocked_child_reason_counts"] == {
        "invalid": 1,
        "no_stage_array_spec": 1,
    }
    assert rows[0]["blocked_child_first_error_counts_top10"] == {
        "detect: missing required array 'frame_indices'": 1,
    }
    assert rows[0]["blocked_child_examples"] == [
        {
            "run_name": "bad_detect",
            "verification": "invalid",
            "first_error": "detect: missing required array 'frame_indices'",
            "error_count": 1,
            "warning_count": 1,
        }
    ]

    output_jsonl = tmp_path / "blocked.jsonl"
    backfill_mod._write_jsonl_rows(rows, output_jsonl)

    persisted = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert persisted == rows


def test_backfill_write_failed_parent_jsonl_rows_are_compact_for_triage(tmp_path: Path) -> None:
    report = {
        "timestamp_utc": "2026-06-10T00:00:00+00:00",
        "apply": True,
        "stores": [
            {
                "zarr_path": "/tmp/example.zarr",
                "parents": [
                    {
                        "parent_path": "detect_runs",
                        "status": "write_failed",
                        "stage": "detect",
                        "stage_spec_available": True,
                        "child_count": 2,
                        "marked_child_count": 1,
                        "completion_epoch_before": None,
                        "completion_epoch_after": None,
                        "latest": "detect_001",
                        "latest_complete": None,
                        "write_error_phase": "stamp_parent_epoch",
                        "write_error_type": "RuntimeError",
                        "write_error": "refusing attr write",
                        "children": [],
                    },
                    {
                        "parent_path": "crop_runs",
                        "status": "stamped",
                        "stage": "crop",
                        "children": [],
                    },
                ],
            }
        ],
    }

    rows = backfill_mod._write_failed_parent_rows(report)

    assert rows == [
        {
            "schema_id": "palette.backfill_completion_epoch_write_failed_parent.v1",
            "timestamp_utc": "2026-06-10T00:00:00+00:00",
            "apply": True,
            "zarr_path": "/tmp/example.zarr",
            "parent_path": "detect_runs",
            "status": "write_failed",
            "stage": "detect",
            "stage_spec_available": True,
            "child_count": 2,
            "marked_child_count": 1,
            "completion_epoch_before": None,
            "completion_epoch_after": None,
            "latest": "detect_001",
            "latest_complete": None,
            "write_error_phase": "stamp_parent_epoch",
            "write_error_run_name": None,
            "write_error_type": "RuntimeError",
            "write_error": "refusing attr write",
        }
    ]

    output_jsonl = tmp_path / "write_failed.jsonl"
    backfill_mod._write_jsonl_rows(rows, output_jsonl)

    persisted = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert persisted == rows


def test_backfill_summary_payload_preserves_apply_abort_metadata() -> None:
    report = {
        "timestamp_utc": "2026-06-10T00:00:00+00:00",
        "apply": False,
        "filters": {"stages": ["crop"], "parent_paths": []},
        "store_count": 1,
        "ok_store_count": 1,
        "non_ok_store_count": 0,
        "blocked_parent_count": 1,
        "filtered_parent_count": 2,
        "would_stamp_parent_count": 3,
        "stamped_parent_count": 0,
        "marked_child_count": 0,
        "summary": {
            "parent_status_counts": {"blocked": 1, "filtered": 2, "would_stamp": 3},
        },
        "apply_aborted": True,
        "apply_abort_reason": "blocked_parents_present",
        "apply_abort_message": "Refusing --apply because selected scope contains blocked parents.",
        "expected_counts": {"blocked_parent_count": 0},
        "expectation_errors": ["blocked_parent_count: expected 0, observed 1"],
    }

    payload = backfill_mod._summary_payload(report)

    assert payload["schema_id"] == "palette.backfill_completion_epoch_report.v1.summary"
    assert payload["apply_aborted"] is True
    assert payload["apply_abort_reason"] == "blocked_parents_present"
    assert "Refusing --apply" in payload["apply_abort_message"]
    assert payload["expected_counts"] == {"blocked_parent_count": 0}
    assert payload["expectation_errors"] == ["blocked_parent_count: expected 0, observed 1"]


def test_completion_epoch_blocker_triage_groups_recommendations(tmp_path: Path) -> None:
    rows = [
        {
            "zarr_path": "/tmp/a.zarr",
            "parent_path": "analysis/custom_runs",
            "stage": None,
            "stage_spec_available": False,
            "latest": "custom_001",
            "blocked_child_reason_counts": {"no_stage_array_spec": 2},
            "blocked_child_first_error_counts_top10": {},
            "blocked_child_examples": [
                {"run_name": "custom_001", "verification": "unverified"},
            ],
        },
        {
            "zarr_path": "/tmp/b.zarr",
            "parent_path": "refined_detect_runs",
            "stage": "refined_detect",
            "stage_spec_available": True,
            "latest": "refine_001",
            "blocked_child_reason_counts": {"invalid": 1},
            "blocked_child_first_error_counts_top10": {
                "refined_detect: missing required subgroup 'source_detections'": 1,
            },
            "blocked_child_examples": [
                {
                    "run_name": "refine_001",
                    "verification": "invalid",
                    "first_error": "refined_detect: missing required subgroup 'source_detections'",
                },
            ],
        },
        {
            "zarr_path": "/tmp/c.zarr",
            "parent_path": "analysis/swim_bout_runs",
            "stage": None,
            "stage_spec_available": False,
            "latest": "bouts_001",
            "blocked_child_reason_counts": {"no_stage_array_spec": 1},
            "blocked_child_first_error_counts_top10": {},
            "blocked_child_examples": [
                {"run_name": "bouts_001", "verification": "unverified"},
            ],
        },
        {
            "zarr_path": "/tmp/d.zarr",
            "parent_path": "eye_masks_runs",
            "stage": "eye_masks",
            "stage_spec_available": True,
            "latest": "eye_masks_001",
            "blocked_child_reason_counts": {"invalid": 1},
            "blocked_child_first_error_counts_top10": {
                "eye_masks: missing required array 'masks_roi'": 1,
            },
            "blocked_child_examples": [
                {
                    "run_name": "eye_masks_001",
                    "verification": "invalid",
                    "first_error": "eye_masks: missing required array 'masks_roi'",
                },
            ],
        },
    ]

    report = triage_mod.build_triage_report(rows, examples_per_group=1)

    assert report["schema_id"] == "palette.completion_epoch_blocker_triage.v1"
    assert report["blocked_parent_count"] == 4
    by_key = {group["key"]: group for group in report["groups"]}
    assert by_key["analysis/custom_runs"]["recommendation"] == "add_stage_array_spec_or_defer_scope"
    assert by_key["eye_masks"]["recommendation"] == "defer_deprecated_scope"
    assert (
        by_key["analysis/swim_bout_runs"]["recommendation"]
        == "defer_scope_until_layout_specific_validator"
    )
    assert (
        by_key["refined_detect"]["recommendation"]
        == "review_stage_spec_compatibility_or_backfill_missing_surface"
    )
    assert by_key["refined_detect"]["blocked_child_first_error_counts_top20"] == [
        {
            "key": "refined_detect: missing required subgroup 'source_detections'",
            "count": 1,
        }
    ]

    blocked_jsonl = tmp_path / "blocked.jsonl"
    blocked_jsonl.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    persisted_rows = triage_mod._read_jsonl(blocked_jsonl)
    assert persisted_rows == rows


def test_emit_stage_completion_requires_root_for_ok_run_with_prebuilt_metadata(tmp_path: Path) -> None:
    class FakeRegistry:
        def close(self) -> None:
            pass

    called = False

    def _upsert(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True

    wrote = emit_stage_completion(
        None,
        tmp_path / "archive.zarr",
        step_name="detect",
        status="ok",
        source="unit_test",
        run_name="detect_001",
        registry=FakeRegistry(),  # type: ignore[arg-type]
        auto_registry_from_env=False,
        upsert_dataset_row=False,
        metadata=type("Metadata", (), {"dataset_id": "d", "recording_id": "r"})(),
        upsert_step_status_fn=_upsert,
        invalidate_steps_fn=lambda *args, **kwargs: None,
    )

    assert wrote is False
    assert called is False


@pytest.mark.parametrize(
    ("step_name", "parent_name", "run_name", "_populate_valid", "_expected_stage", "expected_error"),
    _PROMOTED_STAGE_COMPLETION_CASES,
)
def test_emit_stage_completion_refuses_invalid_promoted_stage_arrays_by_default(
    tmp_path: Path,
    step_name: str,
    parent_name: str,
    run_name: str,
    _populate_valid,
    _expected_stage: str,
    expected_error: str,
) -> None:
    root = FakeGroup()
    parent = FakeGroup()
    run = FakeGroup()
    root[parent_name] = parent
    parent[run_name] = run
    mark_run_started(run, run_name=run_name, stage=step_name)
    mark_run_complete(run, parent_group=parent, run_name=run_name)

    class FakeRegistry:
        def close(self) -> None:
            pass

    class FakeConsole:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def print(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self.messages.append(" ".join(str(arg) for arg in args))

    called = False

    def _upsert(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True

    console = FakeConsole()
    wrote = emit_stage_completion(
        root,  # type: ignore[arg-type]
        tmp_path / "archive.zarr",
        step_name=step_name,
        status="ok",
        source="unit_test",
        run_name=run_name,
        registry=FakeRegistry(),  # type: ignore[arg-type]
        auto_registry_from_env=False,
        upsert_dataset_row=False,
        metadata=type("Metadata", (), {"dataset_id": "d", "recording_id": "r"})(),
        console=console,  # type: ignore[arg-type]
        upsert_step_status_fn=_upsert,
        invalidate_steps_fn=lambda *args, **kwargs: None,
    )

    assert wrote is False
    assert called is False
    assert expected_error in "\n".join(console.messages)


def test_emit_stage_completion_refuses_keypoints_missing_required_array(tmp_path: Path) -> None:
    root = FakeGroup()
    keypoints_parent = FakeGroup()
    run = FakeGroup()
    root["keypoints_runs"] = keypoints_parent
    keypoints_parent["keypoints_001"] = run
    _add_valid_keypoints_arrays(run)
    del run["n_keypoints"]
    mark_run_started(run, run_name="keypoints_001", stage="keypoints")
    mark_run_complete(run, parent_group=keypoints_parent, run_name="keypoints_001")

    class FakeRegistry:
        def close(self) -> None:
            pass

    called = False

    def _upsert(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True

    wrote = emit_stage_completion(
        root,  # type: ignore[arg-type]
        tmp_path / "archive.zarr",
        step_name="keypoints",
        status="ok",
        source="unit_test",
        run_name="keypoints_001",
        registry=FakeRegistry(),  # type: ignore[arg-type]
        auto_registry_from_env=False,
        upsert_dataset_row=False,
        metadata=type("Metadata", (), {"dataset_id": "d", "recording_id": "r"})(),
        upsert_step_status_fn=_upsert,
        invalidate_steps_fn=lambda *args, **kwargs: None,
    )

    assert wrote is False
    assert called is False


def test_emit_stage_completion_accepts_complete_opted_in_run(tmp_path: Path) -> None:
    root = FakeGroup()
    detect_parent = FakeGroup()
    run = FakeGroup()
    root["detect_runs"] = detect_parent
    detect_parent["detect_001"] = run
    mark_run_started(run, run_name="detect_001", stage="detect")
    _add_valid_detect_arrays(run)
    mark_run_complete(run, parent_group=detect_parent, run_name="detect_001")

    captured: dict[str, object] = {}

    class FakeRegistry:
        def close(self) -> None:
            pass

    wrote = emit_stage_completion(
        root,  # type: ignore[arg-type]
        tmp_path / "archive.zarr",
        step_name="detect",
        status="ok",
        source="unit_test",
        run_name="detect_001",
        details_json={"existing": "kept"},
        registry=FakeRegistry(),  # type: ignore[arg-type]
        auto_registry_from_env=False,
        upsert_dataset_row=False,
        metadata=type("Metadata", (), {"dataset_id": "d", "recording_id": "r"})(),
        upsert_step_status_fn=lambda *args, **kwargs: captured.update(kwargs),
        invalidate_steps_fn=lambda *args, **kwargs: None,
    )

    assert wrote is True
    assert captured["step_name"] == "detect"
    assert captured["status"] == "ok"
    assert captured["details_json"]["existing"] == "kept"  # type: ignore[index]
    assert captured["details_json"]["stage_array_validation_status"] == "ok"  # type: ignore[index]
    assert captured["details_json"]["stage_array_validation_stage"] == "detect"  # type: ignore[index]
    assert captured["details_json"]["stage_array_validation_enforced"] is True  # type: ignore[index]
    assert captured["details_json"]["stage_array_validation_warnings"] == [  # type: ignore[index]
        "detect: missing optional array 'centers_px'"
    ]


@pytest.mark.parametrize(
    ("step_name", "parent_name", "run_name", "populate_valid", "expected_stage", "_expected_error"),
    _PROMOTED_STAGE_COMPLETION_CASES,
)
def test_emit_stage_completion_accepts_valid_promoted_stage_run(
    tmp_path: Path,
    step_name: str,
    parent_name: str,
    run_name: str,
    populate_valid,
    expected_stage: str,
    _expected_error: str,
) -> None:
    root = FakeGroup()
    parent = FakeGroup()
    run = FakeGroup()
    root[parent_name] = parent
    parent[run_name] = run
    mark_run_started(run, run_name=run_name, stage=step_name)
    populate_valid(run)
    mark_run_complete(run, parent_group=parent, run_name=run_name)

    captured: dict[str, object] = {}

    class FakeRegistry:
        def close(self) -> None:
            pass

    wrote = emit_stage_completion(
        root,  # type: ignore[arg-type]
        tmp_path / "archive.zarr",
        step_name=step_name,
        status="ok",
        source="unit_test",
        run_name=run_name,
        registry=FakeRegistry(),  # type: ignore[arg-type]
        auto_registry_from_env=False,
        upsert_dataset_row=False,
        metadata=type("Metadata", (), {"dataset_id": "d", "recording_id": "r"})(),
        upsert_step_status_fn=lambda *args, **kwargs: captured.update(kwargs),
        invalidate_steps_fn=lambda *args, **kwargs: None,
    )

    assert wrote is True
    details = captured["details_json"]
    assert details["stage_array_validation_status"] == "ok"  # type: ignore[index]
    assert details["stage_array_validation_stage"] == expected_stage  # type: ignore[index]
    assert details["stage_array_validation_enforced"] is True  # type: ignore[index]


def test_emit_stage_completion_uses_effective_recording_dataset_id(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "session_a" / "zarr" / "session_a_training.zarr"
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["session_uuid"] = "session_a"
    root.attrs["recording_id"] = "session_a"
    root.attrs["zarr_use"] = "training"
    detect_parent = root.create_group("detect_runs")
    detect_run = detect_parent.create_group("detect_001")
    detect_run.create_array("frame_indices", data=np.asarray([0, 1], dtype=np.int32), overwrite=True)
    detect_run.create_array("bbox_norm_coords", data=np.zeros((2, 4), dtype=np.float32), overwrite=True)
    detect_run.create_array("scores", data=np.ones((2,), dtype=np.float32), overwrite=True)
    detect_run.create_array("class_ids", data=np.zeros((2,), dtype=np.int32), overwrite=True)
    detect_run.create_array("frame_counts", data=np.asarray([1, 1, 0], dtype=np.int32), overwrite=True)
    detect_run.create_array("n_detections", data=np.asarray([1, 1, 0], dtype=np.int32), overwrite=True)
    mark_run_complete(detect_run, parent_group=detect_parent, run_name="detect_001")

    wrote = emit_stage_completion(
        root,
        zarr_path,
        step_name="detect",
        status="ok",
        source="unit_test",
        run_name="detect_001",
        registry=registry,
        auto_registry_from_env=False,
        invalidate_on_ok=False,
    )

    assert wrote is True
    expected_dataset_id = f"session_a:z{sha256(str(zarr_path.resolve()).encode('utf-8')).hexdigest()[:12]}"
    rows = registry.conn.execute(
        """
        SELECT dataset_id, step_name, status, run_name
        FROM recording_step_status
        ORDER BY dataset_id, step_name;
        """
    ).fetchall()
    assert [dict(row) for row in rows] == [
        {
            "dataset_id": expected_dataset_id,
            "step_name": "detect",
            "status": "ok",
            "run_name": "detect_001",
        }
    ]
    assert registry.conn.execute(
        "SELECT COUNT(*) FROM recording_step_status WHERE dataset_id = 'session_a';"
    ).fetchone()[0] == 0
    registry.close()


def test_emit_stage_completion_non_ok_status_bypasses_run_validation(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeRegistry:
        def close(self) -> None:
            pass

    wrote = emit_stage_completion(
        None,
        tmp_path / "archive.zarr",
        step_name="detect",
        status="missing",
        source="unit_test",
        run_name="detect_001",
        registry=FakeRegistry(),  # type: ignore[arg-type]
        auto_registry_from_env=False,
        upsert_dataset_row=False,
        metadata=type("Metadata", (), {"dataset_id": "d", "recording_id": "r"})(),
        upsert_step_status_fn=lambda *args, **kwargs: captured.update(kwargs),
        invalidate_steps_fn=lambda *args, **kwargs: None,
    )

    assert wrote is True
    assert captured["status"] == "missing"
    assert captured["details_json"] is None


def test_emit_stage_completion_records_missing_stage_spec_warning(tmp_path: Path) -> None:
    root = FakeGroup()
    parent = FakeGroup()
    run = FakeGroup()
    root["custom_stage_runs"] = parent
    parent["custom_001"] = run
    mark_run_started(run, run_name="custom_001", stage="custom_stage")
    mark_run_complete(run, parent_group=parent, run_name="custom_001")

    captured: dict[str, object] = {}

    class FakeRegistry:
        def close(self) -> None:
            pass

    wrote = emit_stage_completion(
        root,  # type: ignore[arg-type]
        tmp_path / "archive.zarr",
        step_name="custom_stage",
        status="ok",
        source="unit_test",
        run_name="custom_001",
        registry=FakeRegistry(),  # type: ignore[arg-type]
        auto_registry_from_env=False,
        upsert_dataset_row=False,
        metadata=type("Metadata", (), {"dataset_id": "d", "recording_id": "r"})(),
        upsert_step_status_fn=lambda *args, **kwargs: captured.update(kwargs),
        invalidate_steps_fn=lambda *args, **kwargs: None,
    )

    assert wrote is True
    details = captured["details_json"]
    assert details["stage_array_validation_status"] == "no_spec"  # type: ignore[index]
    assert details["stage_array_validation_enforced"] is False  # type: ignore[index]
    assert details["stage_array_validation_warnings"] == [  # type: ignore[index]
        "no StageSpec for step 'custom_stage'; skipped array validation"
    ]


def test_emit_stage_completion_resolves_nested_detect_quality_run(tmp_path: Path) -> None:
    root = FakeGroup()
    detect_parent = FakeGroup()
    detect_run = FakeGroup()
    quality_parent = FakeGroup()
    quality_run = FakeGroup()
    root["detect_runs"] = detect_parent
    detect_parent["detect_001"] = detect_run
    detect_run["quality_reports"] = quality_parent
    quality_parent["quality_001"] = quality_run
    _add_valid_detect_quality_arrays(quality_run)
    mark_run_started(quality_run, run_name="quality_001", stage="detect_quality")
    mark_run_complete(quality_run, parent_group=quality_parent, run_name="quality_001")

    captured: dict[str, object] = {}

    class FakeRegistry:
        def close(self) -> None:
            pass

    wrote = emit_stage_completion(
        root,  # type: ignore[arg-type]
        tmp_path / "archive.zarr",
        step_name="detect_quality",
        status="ok",
        source="unit_test",
        run_name="quality_001",
        registry=FakeRegistry(),  # type: ignore[arg-type]
        auto_registry_from_env=False,
        upsert_dataset_row=False,
        metadata=type("Metadata", (), {"dataset_id": "d", "recording_id": "r"})(),
        upsert_step_status_fn=lambda *args, **kwargs: captured.update(kwargs),
        invalidate_steps_fn=lambda *args, **kwargs: None,
    )

    assert wrote is True
    details = captured["details_json"]
    assert details["stage_array_validation_status"] == "ok"  # type: ignore[index]
    assert details["stage_array_validation_stage"] == "detect_quality"  # type: ignore[index]
    assert details["stage_array_validation_enforced"] is True  # type: ignore[index]


def test_emit_stage_completion_refuses_invalid_detect_quality_by_default(tmp_path: Path) -> None:
    root = FakeGroup()
    detect_parent = FakeGroup()
    detect_run = FakeGroup()
    quality_parent = FakeGroup()
    quality_run = FakeGroup()
    root["detect_runs"] = detect_parent
    detect_parent["detect_001"] = detect_run
    detect_run["quality_reports"] = quality_parent
    quality_parent["quality_001"] = quality_run
    mark_run_started(quality_run, run_name="quality_001", stage="detect_quality")
    mark_run_complete(quality_run, parent_group=quality_parent, run_name="quality_001")

    class FakeRegistry:
        def close(self) -> None:
            pass

    called = False

    def _upsert(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True

    wrote = emit_stage_completion(
        root,  # type: ignore[arg-type]
        tmp_path / "archive.zarr",
        step_name="detect_quality",
        status="ok",
        source="unit_test",
        run_name="quality_001",
        registry=FakeRegistry(),  # type: ignore[arg-type]
        auto_registry_from_env=False,
        upsert_dataset_row=False,
        metadata=type("Metadata", (), {"dataset_id": "d", "recording_id": "r"})(),
        upsert_step_status_fn=_upsert,
        invalidate_steps_fn=lambda *args, **kwargs: None,
    )

    assert wrote is False
    assert called is False


def test_emit_stage_completion_uses_explicit_nested_completion_group_path(tmp_path: Path) -> None:
    root = FakeGroup()
    clips = FakeGroup()
    clip = FakeGroup()
    cameras = FakeGroup()
    camera = FakeGroup()
    detect_parent = FakeGroup()
    detect_run = FakeGroup()
    quality_parent = FakeGroup()
    quality_run = FakeGroup()
    root["clips"] = clips
    clips["clip_000000"] = clip
    clip["cameras"] = cameras
    cameras["2010093"] = camera
    camera["detect_runs"] = detect_parent
    detect_parent["detect_clip"] = detect_run
    detect_run["quality_reports"] = quality_parent
    quality_parent["quality_clip"] = quality_run
    _add_valid_detect_quality_arrays(quality_run)
    mark_run_started(quality_run, run_name="quality_clip", stage="detect_quality")
    mark_run_complete(quality_run, parent_group=quality_parent, run_name="quality_clip")

    captured: dict[str, object] = {}

    class FakeRegistry:
        def close(self) -> None:
            pass

    wrote = emit_stage_completion(
        root,  # type: ignore[arg-type]
        tmp_path / "archive.zarr",
        step_name="detect_quality",
        status="ok",
        source="unit_test",
        run_name="quality_clip",
        registry=FakeRegistry(),  # type: ignore[arg-type]
        auto_registry_from_env=False,
        upsert_dataset_row=False,
        metadata=type("Metadata", (), {"dataset_id": "d", "recording_id": "r"})(),
        completion_group_path=(
            "clips/clip_000000/cameras/2010093/detect_runs/"
            "detect_clip/quality_reports/quality_clip"
        ),
        upsert_step_status_fn=lambda *args, **kwargs: captured.update(kwargs),
        invalidate_steps_fn=lambda *args, **kwargs: None,
    )

    assert wrote is True
    details = captured["details_json"]
    assert details["stage_array_validation_status"] == "ok"  # type: ignore[index]
    assert details["stage_array_validation_stage"] == "detect_quality"  # type: ignore[index]
    assert details["stage_array_validation_enforced"] is True  # type: ignore[index]


def test_emit_stage_completion_resolves_detect_quality_by_direct_path_when_parent_metadata_stale(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = FakeGroup()
    detect_parent = FakeGroup()
    detect_run = FakeGroup()
    quality_parent = FakeGroup(
        attrs={
            "latest": "quality_fresh",
            "latest_complete": "quality_fresh",
        }
    )
    direct_quality_run = FakeGroup()
    root["detect_runs"] = detect_parent
    detect_parent["detect_001"] = detect_run
    detect_run["quality_reports"] = quality_parent
    _add_valid_detect_quality_arrays(direct_quality_run)
    mark_run_started(direct_quality_run, run_name="quality_fresh", stage="detect_quality")
    mark_run_complete(direct_quality_run, run_name="quality_fresh")

    opened_paths: list[Path] = []

    def _open_direct(path: Path, *, mode: str):  # type: ignore[no-untyped-def]
        opened_paths.append(Path(path))
        assert mode == "r"
        return direct_quality_run

    monkeypatch.setattr(stage_complete_mod, "open_zarr_group_direct", _open_direct)

    captured: dict[str, object] = {}

    class FakeRegistry:
        def close(self) -> None:
            pass

    zarr_path = tmp_path / "archive.zarr"
    wrote = emit_stage_completion(
        root,  # type: ignore[arg-type]
        zarr_path,
        step_name="detect_quality",
        status="ok",
        source="unit_test",
        run_name="quality_fresh",
        registry=FakeRegistry(),  # type: ignore[arg-type]
        auto_registry_from_env=False,
        upsert_dataset_row=False,
        metadata=type("Metadata", (), {"dataset_id": "d", "recording_id": "r"})(),
        upsert_step_status_fn=lambda *args, **kwargs: captured.update(kwargs),
        invalidate_steps_fn=lambda *args, **kwargs: None,
    )

    assert wrote is True
    assert opened_paths == [
        zarr_path.resolve()
        / "detect_runs"
        / "detect_001"
        / "quality_reports"
        / "quality_fresh"
    ]
    details = captured["details_json"]
    assert details["stage_array_validation_status"] == "ok"  # type: ignore[index]
    assert details["stage_array_validation_stage"] == "detect_quality"  # type: ignore[index]
    assert details["stage_array_validation_enforced"] is True  # type: ignore[index]


def test_emit_stage_completion_resolves_tracks_alias_to_tracking_spec(tmp_path: Path) -> None:
    root = FakeGroup()
    tracking_parent = FakeGroup()
    run = FakeGroup()
    root["tracking_runs"] = tracking_parent
    tracking_parent["tracks_001"] = run
    _add_valid_tracking_arrays(run)
    mark_run_started(run, run_name="tracks_001", stage="tracking")
    mark_run_complete(run, parent_group=tracking_parent, run_name="tracks_001")

    captured: dict[str, object] = {}

    class FakeRegistry:
        def close(self) -> None:
            pass

    wrote = emit_stage_completion(
        root,  # type: ignore[arg-type]
        tmp_path / "archive.zarr",
        step_name="tracks",
        status="ok",
        source="unit_test",
        run_name="tracks_001",
        registry=FakeRegistry(),  # type: ignore[arg-type]
        auto_registry_from_env=False,
        upsert_dataset_row=False,
        metadata=type("Metadata", (), {"dataset_id": "d", "recording_id": "r"})(),
        upsert_step_status_fn=lambda *args, **kwargs: captured.update(kwargs),
        invalidate_steps_fn=lambda *args, **kwargs: None,
    )

    assert wrote is True
    details = captured["details_json"]
    assert details["stage_array_validation_status"] == "ok"  # type: ignore[index]
    assert details["stage_array_validation_stage"] == "tracking"  # type: ignore[index]
    assert details["stage_array_validation_enforced"] is True  # type: ignore[index]


def test_emit_stage_completion_resolves_refine_alias_to_refined_detect_spec(tmp_path: Path) -> None:
    root = FakeGroup()
    refined_parent = FakeGroup()
    run = FakeGroup()
    root["refined_detect_runs"] = refined_parent
    refined_parent["refined_001"] = run
    _add_valid_refined_detect_subgroups(run)
    mark_run_started(run, run_name="refined_001", stage="refined_detect")
    mark_run_complete(run, parent_group=refined_parent, run_name="refined_001")

    captured: dict[str, object] = {}

    class FakeRegistry:
        def close(self) -> None:
            pass

    wrote = emit_stage_completion(
        root,  # type: ignore[arg-type]
        tmp_path / "archive.zarr",
        step_name="refine",
        status="ok",
        source="unit_test",
        run_name="refined_001",
        registry=FakeRegistry(),  # type: ignore[arg-type]
        auto_registry_from_env=False,
        upsert_dataset_row=False,
        metadata=type("Metadata", (), {"dataset_id": "d", "recording_id": "r"})(),
        upsert_step_status_fn=lambda *args, **kwargs: captured.update(kwargs),
        invalidate_steps_fn=lambda *args, **kwargs: None,
    )

    assert wrote is True
    details = captured["details_json"]
    assert details["stage_array_validation_status"] == "ok"  # type: ignore[index]
    assert details["stage_array_validation_stage"] == "refined_detect"  # type: ignore[index]


def test_emit_stage_completion_records_refined_detect_missing_subgroup_in_shadow_mode(
    tmp_path: Path,
) -> None:
    root = FakeGroup()
    refined_parent = FakeGroup()
    run = FakeGroup()
    root["refined_detect_runs"] = refined_parent
    refined_parent["refined_001"] = run
    run["instances"] = FakeGroup()
    mark_run_started(run, run_name="refined_001", stage="refined_detect")
    mark_run_complete(run, parent_group=refined_parent, run_name="refined_001")

    captured: dict[str, object] = {}

    class FakeRegistry:
        def close(self) -> None:
            pass

    wrote = emit_stage_completion(
        root,  # type: ignore[arg-type]
        tmp_path / "archive.zarr",
        step_name="refined_detect",
        status="ok",
        source="unit_test",
        run_name="refined_001",
        registry=FakeRegistry(),  # type: ignore[arg-type]
        auto_registry_from_env=False,
        upsert_dataset_row=False,
        metadata=type("Metadata", (), {"dataset_id": "d", "recording_id": "r"})(),
        upsert_step_status_fn=lambda *args, **kwargs: captured.update(kwargs),
        invalidate_steps_fn=lambda *args, **kwargs: None,
    )

    assert wrote is True
    details = captured["details_json"]
    assert details["stage_array_validation_status"] == "invalid"  # type: ignore[index]
    assert details["stage_array_validation_stage"] == "refined_detect"  # type: ignore[index]
    assert "refined_detect: missing required subgroup 'source_detections'" in details[  # type: ignore[operator]
        "stage_array_validation_errors"
    ]


def test_emit_stage_completion_refuses_refined_detect_missing_subgroup_when_enforced(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = FakeGroup()
    refined_parent = FakeGroup()
    run = FakeGroup()
    root["refined_detect_runs"] = refined_parent
    refined_parent["refined_001"] = run
    run["instances"] = FakeGroup()
    mark_run_started(run, run_name="refined_001", stage="refined_detect")
    mark_run_complete(run, parent_group=refined_parent, run_name="refined_001")

    monkeypatch.setattr(
        stage_complete_mod,
        "_ENFORCE_STAGE_ARRAY_VALIDATION_FOR",
        frozenset({"refined_detect"}),
    )

    class FakeRegistry:
        def close(self) -> None:
            pass

    called = False

    def _upsert(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True

    wrote = emit_stage_completion(
        root,  # type: ignore[arg-type]
        tmp_path / "archive.zarr",
        step_name="refined_detect",
        status="ok",
        source="unit_test",
        run_name="refined_001",
        registry=FakeRegistry(),  # type: ignore[arg-type]
        auto_registry_from_env=False,
        upsert_dataset_row=False,
        metadata=type("Metadata", (), {"dataset_id": "d", "recording_id": "r"})(),
        upsert_step_status_fn=_upsert,
        invalidate_steps_fn=lambda *args, **kwargs: None,
    )

    assert wrote is False
    assert called is False


def test_emit_stage_completion_real_zarr_writes_shadow_validation_details(tmp_path: Path) -> None:
    zarr_path = tmp_path / "archive.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    detect_parent = root.create_group("detect_runs")
    detect_run = detect_parent.create_group("detect_001")
    mark_run_started(detect_run, run_name="detect_001", stage="detect")
    detect_run.create_array("frame_indices", data=np.asarray([0, 1], dtype=np.int32))
    detect_run.create_array(
        "bbox_norm_coords",
        data=np.asarray([[0.5, 0.5, 0.1, 0.1], [0.4, 0.4, 0.2, 0.2]], dtype=np.float32),
    )
    detect_run.create_array("scores", data=np.asarray([0.9, 0.8], dtype=np.float32))
    detect_run.create_array("class_ids", data=np.asarray([0, 0], dtype=np.int32))
    detect_run.create_array("frame_counts", data=np.asarray([1, 1], dtype=np.int32))
    mark_run_complete(detect_run, parent_group=detect_parent, run_name="detect_001")

    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.upsert_dataset(
            "dataset_real",
            session_uuid="dataset_real",
            zarr_path=zarr_path,
            recording_id="recording_real",
        )
        metadata = DatasetMetadata(
            dataset_id="dataset_real",
            session_uuid=None,
            recording_id="recording_real",
            zarr_use=None,
            zarr_purpose=None,
            source_layout=None,
            source_frame_index_path=None,
            source_recording_frame_index_path=None,
            source_frame_index_schema=None,
        )

        wrote = emit_stage_completion(
            root,
            zarr_path,
            step_name="detect",
            status="ok",
            source="unit_test_real_zarr",
            run_name="detect_001",
            registry=registry,
            auto_registry_from_env=False,
            upsert_dataset_row=False,
            metadata=metadata,
            invalidate_steps_fn=lambda *args, **kwargs: None,
        )

        assert wrote is True
        row = registry.conn.execute(
            """
            SELECT status, details_json
            FROM recording_step_status
            WHERE dataset_id = ? AND step_name = ?;
            """,
            ("dataset_real", "detect"),
        ).fetchone()
        assert row is not None
        assert row["status"] == "ok"
        assert '"stage_array_validation_status":"ok"' in row["details_json"]
        assert '"stage_array_validation_enforced":true' in row["details_json"]
    finally:
        registry.close()
