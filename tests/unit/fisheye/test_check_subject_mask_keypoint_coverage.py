from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fisheye.diagnostics import check_subject_mask_keypoint_coverage as mod


class _FakeGroup(dict):
    def __init__(self, *args: Any, attrs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, default)

    def group_keys(self) -> list[str]:
        keys: list[str] = []
        for key, value in self.items():
            if isinstance(value, _FakeGroup):
                keys.append(str(key))
        return keys


def _keypoint_root() -> tuple[_FakeGroup, _FakeGroup]:
    crop = _FakeGroup({"frame_indices": np.array([100, 101, 102], dtype=np.int64)})
    crop_parent = _FakeGroup({"crop_001": crop}, attrs={"latest": "crop_001"})

    keypoints_roi = np.array(
        [
            [[9.0, 9.0], [4.0, 2.0], [5.0, 5.0], [2.0, 2.0]],
            [[9.0, 9.0], [4.0, 2.0], [6.0, 5.0], [2.0, 2.0]],
            [[9.0, 9.0], [3.0, 3.0], [7.0, 5.0], [3.0, 3.0]],
        ],
        dtype=np.float32,
    )
    kp = _FakeGroup(
        {
            "keypoints_roi": keypoints_roi,
            "detection_success": np.array([True, True, True], dtype=bool),
        },
        attrs={
            "keypoint_labels": ["tail_tip", "eye_right", "swim_bladder", "eye_left"],
        },
    )
    kp_parent = _FakeGroup({"kp_001": kp}, attrs={"latest": "kp_001"})

    root = _FakeGroup(
        {
            "crop_runs": crop_parent,
            "refined_keypoints_runs": kp_parent,
        }
    )
    return root, kp


def _add_refined_subject_run(
    root: _FakeGroup,
    *,
    masks_roi: np.ndarray,
    mask_labels: list[str],
    available_channels: np.ndarray | None,
    attrs: dict[str, Any] | None = None,
    metrics_mask_present: np.ndarray | None = None,
) -> _FakeGroup:
    run_attrs = {
        "source_crop_run": "crop_001",
        "source_keypoint_group": "refined_keypoints_runs",
        "source_keypoints_run": "kp_001",
        "label_schema_id": "subject_v1_lr" if "eye_left" in mask_labels else "subject_v1_union",
        "mask_labels": mask_labels,
        "component_review_statuses": {
            "eye_left": {"state": "approved"},
            "eye_right": {"state": "approved"},
            "eyes_union": {"state": "approved"},
        },
    }
    if attrs:
        run_attrs.update(attrs)

    run = _FakeGroup({"masks_roi": masks_roi}, attrs=run_attrs)
    if available_channels is not None:
        run["available_channels"] = available_channels
    if metrics_mask_present is not None:
        run["metrics"] = _FakeGroup({"mask_present": metrics_mask_present})

    parent = _FakeGroup({"refined_subject_masks_001": run}, attrs={"latest": "refined_subject_masks_001"})
    root["refined_subject_masks_runs"] = parent
    return run


def _add_subject_run(
    root: _FakeGroup,
    *,
    run_name: str,
    masks_roi: np.ndarray,
    mask_labels: list[str],
    available_channels: np.ndarray,
    latest: str | None = None,
) -> _FakeGroup:
    run = _FakeGroup(
        {
            "masks_roi": masks_roi,
            "available_channels": available_channels,
        },
        attrs={
            "source_crop_run": "crop_001",
            "source_keypoint_group": "refined_keypoints_runs",
            "source_keypoints_run": "kp_001",
            "label_schema_id": "subject_v1_lr" if "eye_left" in mask_labels else "subject_v1_union",
            "mask_labels": mask_labels,
        },
    )
    parent = root.get("subject_mask_runs")
    if parent is None:
        parent = _FakeGroup(attrs={})
        root["subject_mask_runs"] = parent
    parent[run_name] = run
    parent.attrs["latest"] = latest or run_name
    return run


def test_analyze_root_reports_lr_failures_using_label_resolved_keypoint_indices(tmp_path: Path) -> None:
    root, _kp = _keypoint_root()
    masks = np.zeros((3, 4, 8, 8), dtype=np.uint8)
    masks[0, 1, 1:3, 1:3] = 1
    masks[0, 2, 1:3, 4:6] = 1
    masks[1, 1, 1:3, 1:3] = 1
    _add_refined_subject_run(
        root,
        masks_roi=masks,
        mask_labels=["subject_body", "eye_left", "eye_right", "swim_bladder"],
        available_channels=np.asarray([False, True, True, False], dtype=bool),
    )

    report = mod._analyze_root(
        root=root,
        zarr_path=tmp_path / "demo_analysis.zarr",
        stage="auto",
        subject_run=None,
        eye_mode="auto",
        keypoint_group=None,
        keypoint_run=None,
        allow_latest_keypoint_fallback=False,
        sample_limit=5,
    )

    assert report.status == "fail"
    assert report.subject_stage == "refined_subject_masks_runs"
    assert report.eye_component_mode == "lr"
    assert report.eye_component_indices == {"eye_left": 1, "eye_right": 2}
    assert report.keypoint_eye_indices == {"eye_left": 3, "eye_right": 1}
    assert report.keypoint_valid_rows == 2
    assert report.rows_with_eye_component_masks == 1
    assert report.rows_missing_eye_component_masks == 1
    assert report.failure_targets == [{"roi_idx": 1, "frame_idx": 101}]
    assert report.sample_missing == [{"roi_idx": 1, "frame_idx": 101}]


def test_analyze_root_auto_selects_eye_capable_subject_run_over_component_scoped_refined_latest(
    tmp_path: Path,
) -> None:
    root, _kp = _keypoint_root()
    refined_body_masks = np.ones((3, 1, 8, 8), dtype=np.uint8)
    _add_refined_subject_run(
        root,
        masks_roi=refined_body_masks,
        mask_labels=["subject_body"],
        available_channels=np.asarray([True], dtype=bool),
    )

    lr_masks = np.zeros((3, 4, 8, 8), dtype=np.uint8)
    lr_masks[:, 1, 1:3, 1:3] = 1
    lr_masks[:, 2, 1:3, 4:6] = 1
    _add_subject_run(
        root,
        run_name="subject_masks_legacy_eye_bridge_001",
        masks_roi=lr_masks,
        mask_labels=["subject_body", "eye_left", "eye_right", "swim_bladder"],
        available_channels=np.asarray([False, True, True, False], dtype=bool),
    )

    union_masks = np.zeros((3, 3, 8, 8), dtype=np.uint8)
    union_masks[:, 1, 2:4, 2:4] = 1
    _add_subject_run(
        root,
        run_name="subject_masks_union_latest_001",
        masks_roi=union_masks,
        mask_labels=["subject_body", "eyes_union", "swim_bladder"],
        available_channels=np.asarray([True, True, True], dtype=bool),
        latest="subject_masks_union_latest_001",
    )

    report = mod._analyze_root(
        root=root,
        zarr_path=tmp_path / "demo_training.zarr",
        stage="auto",
        subject_run=None,
        eye_mode="auto",
        keypoint_group=None,
        keypoint_run=None,
        allow_latest_keypoint_fallback=False,
        sample_limit=5,
    )

    assert report.status == "pass"
    assert report.subject_stage == "subject_mask_runs"
    assert report.subject_run == "subject_masks_legacy_eye_bridge_001"
    assert report.eye_component_mode == "lr"
    assert report.rows_missing_eye_component_masks == 0


def test_analyze_root_uses_union_component_when_lr_components_are_not_available(tmp_path: Path) -> None:
    root, _kp = _keypoint_root()
    masks = np.zeros((3, 3, 8, 8), dtype=np.uint8)
    masks[0, 1, 1:4, 1:4] = 1
    masks[0, 1, 1:4, 4:7] = 1
    masks[1, 1, 1:4, 1:4] = 1
    masks[1, 1, 1:4, 4:7] = 1
    _add_refined_subject_run(
        root,
        masks_roi=masks,
        mask_labels=["subject_body", "eyes_union", "swim_bladder"],
        available_channels=np.asarray([False, True, False], dtype=bool),
    )

    report = mod._analyze_root(
        root=root,
        zarr_path=tmp_path / "demo_training.zarr",
        stage="auto",
        subject_run=None,
        eye_mode="auto",
        keypoint_group=None,
        keypoint_run=None,
        allow_latest_keypoint_fallback=False,
        sample_limit=5,
    )

    assert report.status == "pass"
    assert report.eye_component_mode == "union"
    assert report.eye_component_indices == {"eyes_union": 1}
    assert report.keypoint_valid_rows == 2
    assert report.rows_with_eye_component_masks == 2
    assert report.rows_missing_eye_component_masks == 0
    assert report.eyes_union_assignment_status == "ready"
    assert report.eyes_union_assignment_summary["assigned_rows"] == 2
    assert report.eyes_union_assignment_summary["keypoint_valid_assigned_rows"] == 2
    assert report.eyes_union_assignment_summary["keypoint_valid_failed_rows"] == 0


def test_analyze_root_reports_union_assignment_not_ready_for_unsplittable_union(
    tmp_path: Path,
) -> None:
    root, _kp = _keypoint_root()
    masks = np.zeros((3, 3, 8, 8), dtype=np.uint8)
    masks[0, 1, 2:4, 2:4] = 1
    masks[1, 1, 2:4, 2:4] = 1
    _add_refined_subject_run(
        root,
        masks_roi=masks,
        mask_labels=["subject_body", "eyes_union", "swim_bladder"],
        available_channels=np.asarray([False, True, False], dtype=bool),
    )

    report = mod._analyze_root(
        root=root,
        zarr_path=tmp_path / "demo_training.zarr",
        stage="auto",
        subject_run=None,
        eye_mode="auto",
        keypoint_group=None,
        keypoint_run=None,
        allow_latest_keypoint_fallback=False,
        sample_limit=5,
    )

    assert report.status == "pass"
    assert report.eye_component_mode == "union"
    assert report.eyes_union_assignment_status == "not_ready"
    assert report.eyes_union_assignment_summary["assigned_rows"] == 0
    assert report.eyes_union_assignment_summary["keypoint_valid_failed_rows"] == 2


def test_analyze_root_uses_assignment_keypoint_lineage_for_union_audit(tmp_path: Path) -> None:
    root, _kp = _keypoint_root()
    masks = np.zeros((3, 3, 8, 8), dtype=np.uint8)
    masks[0, 1, 1:4, 1:4] = 1
    masks[0, 1, 1:4, 4:7] = 1
    masks[1, 1, 1:4, 1:4] = 1
    masks[1, 1, 1:4, 4:7] = 1
    run = _add_subject_run(
        root,
        run_name="subject_masks_union_canary_001",
        masks_roi=masks,
        mask_labels=["subject_body", "eyes_union", "swim_bladder"],
        available_channels=np.asarray([False, True, False], dtype=bool),
    )
    run.attrs["source_keypoint_group"] = "refined_keypoints_runs"
    run.attrs["source_keypoints_run"] = "missing_source_kp"
    run.attrs["source_keypoint_run"] = "missing_source_kp"
    run.attrs["assignment_keypoint_group"] = "refined_keypoints_runs"
    run.attrs["assignment_keypoints_run"] = "kp_001"

    report = mod._analyze_root(
        root=root,
        zarr_path=tmp_path / "demo_training.zarr",
        stage="subject_mask_runs",
        subject_run="subject_masks_union_canary_001",
        eye_mode="union",
        keypoint_group=None,
        keypoint_run=None,
        allow_latest_keypoint_fallback=False,
        sample_limit=5,
    )

    assert report.status == "pass"
    assert report.keypoint_run == "kp_001"
    assert report.eyes_union_assignment_status == "ready"
    assert "using_assignment_keypoint_lineage:refined_keypoints_runs/kp_001" in report.notes


def test_analyze_root_requires_available_channels_for_modern_surface(tmp_path: Path) -> None:
    root, _kp = _keypoint_root()
    masks = np.zeros((3, 4, 8, 8), dtype=np.uint8)
    _add_refined_subject_run(
        root,
        masks_roi=masks,
        mask_labels=["subject_body", "eye_left", "eye_right", "swim_bladder"],
        available_channels=None,
    )

    report = mod._analyze_root(
        root=root,
        zarr_path=tmp_path / "demo_analysis.zarr",
        stage="auto",
        subject_run=None,
        eye_mode="auto",
        keypoint_group=None,
        keypoint_run=None,
        allow_latest_keypoint_fallback=False,
        sample_limit=5,
    )

    assert report.status == "missing"
    assert "missing available_channels" in str(report.reason)


def test_analyze_root_requires_keypoint_lineage_unless_latest_fallback_is_enabled(tmp_path: Path) -> None:
    root, _kp = _keypoint_root()
    masks = np.ones((3, 4, 8, 8), dtype=np.uint8)
    _add_refined_subject_run(
        root,
        masks_roi=masks,
        mask_labels=["subject_body", "eye_left", "eye_right", "swim_bladder"],
        available_channels=np.asarray([False, True, True, False], dtype=bool),
        attrs={
            "source_keypoint_group": None,
            "source_keypoints_run": None,
            "source_keypoint_run": None,
        },
    )

    missing_report = mod._analyze_root(
        root=root,
        zarr_path=tmp_path / "demo_analysis.zarr",
        stage="auto",
        subject_run=None,
        eye_mode="auto",
        keypoint_group=None,
        keypoint_run=None,
        allow_latest_keypoint_fallback=False,
        sample_limit=5,
    )
    assert missing_report.status == "missing"
    assert "missing keypoint lineage" in str(missing_report.reason)

    fallback_report = mod._analyze_root(
        root=root,
        zarr_path=tmp_path / "demo_analysis.zarr",
        stage="auto",
        subject_run=None,
        eye_mode="auto",
        keypoint_group=None,
        keypoint_run=None,
        allow_latest_keypoint_fallback=True,
        sample_limit=5,
    )
    assert fallback_report.status == "pass"
    assert "fallback_latest:refined_keypoints_runs/kp_001" in fallback_report.notes


def test_analyze_root_uses_eye_source_subject_lineage_for_multisource_refined_run(
    tmp_path: Path,
) -> None:
    root, _kp = _keypoint_root()
    source_masks = np.zeros((3, 4, 8, 8), dtype=np.uint8)
    source_masks[:, 1, 1:3, 1:3] = 1
    source_masks[:, 2, 1:3, 4:6] = 1
    _add_subject_run(
        root,
        run_name="subject_masks_legacy_eye_bridge_001",
        masks_roi=source_masks,
        mask_labels=["subject_body", "eye_left", "eye_right", "swim_bladder"],
        available_channels=np.asarray([False, True, True, False], dtype=bool),
    )

    refined_masks = np.ones((3, 4, 8, 8), dtype=np.uint8)
    _add_refined_subject_run(
        root,
        masks_roi=refined_masks,
        mask_labels=["subject_body", "eye_left", "eye_right", "swim_bladder"],
        available_channels=np.asarray([True, True, True, True], dtype=bool),
        attrs={
            "source_keypoint_group": "refined_keypoints_runs",
            "source_keypoints_run": None,
            "source_keypoint_run": None,
            "provenance": {
                "inputs": {
                    "source_eye_subject_mask_run": "subject_masks_legacy_eye_bridge_001",
                },
            },
        },
    )

    report = mod._analyze_root(
        root=root,
        zarr_path=tmp_path / "demo_training.zarr",
        stage="auto",
        subject_run=None,
        eye_mode="auto",
        keypoint_group=None,
        keypoint_run=None,
        allow_latest_keypoint_fallback=False,
        sample_limit=5,
    )

    assert report.status == "pass"
    assert report.subject_stage == "refined_subject_masks_runs"
    assert report.keypoint_group == "refined_keypoints_runs"
    assert report.keypoint_run == "kp_001"
    assert "using_source_eye_subject_lineage:subject_mask_runs/subject_masks_legacy_eye_bridge_001" in report.notes


def test_write_frame_flag_file_emits_eye_review_compatible_json(tmp_path: Path) -> None:
    first = mod.CoverageReport(
        zarr_path=tmp_path / "first_training.zarr",
        status="fail",
        failure_targets=[
            {"roi_idx": 2, "frame_idx": 10},
            {"roi_idx": 3},
        ],
    )
    second = mod.CoverageReport(
        zarr_path=tmp_path / "second_training.zarr",
        status="pass",
        failure_targets=[{"roi_idx": 99, "frame_idx": 99}],
    )

    out_path = tmp_path / "nested" / "subject_mask_eye_flags.json"
    zarr_count, target_count = mod._write_frame_flag_file(out_path, [first, second])

    assert zarr_count == 1
    assert target_count == 2
    assert out_path.read_text(encoding="utf-8").endswith("\n")
    assert out_path.read_text(encoding="utf-8")
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload == {
        str(tmp_path / "first_training.zarr"): [
            {"frame_idx": 10, "roi_idx": 2},
            {"roi_idx": 3},
        ]
    }


def test_write_repair_plan_file_emits_lineage_and_candidate_commands(tmp_path: Path) -> None:
    zarr_path = tmp_path / "first_training.zarr"
    frame_flag_path = tmp_path / "flags.json"
    report = mod.CoverageReport(
        zarr_path=zarr_path,
        status="fail",
        reason="keypoint-valid rows missing required subject-mask eye component(s).",
        subject_stage="subject_mask_runs",
        subject_run="subject_masks_eye_bridge",
        label_schema_id="subject_v1_lr",
        eye_component_mode="lr",
        eye_component_indices={"eye_left": 1, "eye_right": 2},
        source_refined_eye_masks_run="refined_eye_masks_source",
        source_eye_masks_run="eye_masks_source",
        latest_refined_eye_masks_run="refined_eye_masks_latest",
        keypoint_group="refined_keypoints_runs",
        keypoint_run="refined_keypoints_checked",
        success_dataset="refined_success",
        keypoint_eye_indices={"eye_left": 1, "eye_right": 2},
        failure_targets=[
            {"roi_idx": 2, "frame_idx": 10},
            {"roi_idx": 3},
        ],
    )

    out_path = tmp_path / "nested" / "repair-plan.jsonl"
    row_count = mod._write_repair_plan_file(
        out_path,
        [report],
        frame_flag_file=frame_flag_path,
    )

    assert row_count == 2
    rows = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["target"] for row in rows] == [
        {"frame_idx": 10, "roi_idx": 2},
        {"roi_idx": 3},
    ]
    first = rows[0]
    assert first["zarr"] == str(zarr_path)
    assert first["subject_stage"] == "subject_mask_runs"
    assert first["subject_run"] == "subject_masks_eye_bridge"
    assert first["source_refined_eye_masks_run"] == "refined_eye_masks_source"
    assert first["latest_refined_eye_masks_run"] == "refined_eye_masks_latest"
    assert first["keypoint_group"] == "refined_keypoints_runs"
    assert first["keypoint_run"] == "refined_keypoints_checked"
    assert first["frame_flag_file"] == str(frame_flag_path)
    assert first["classification_required"] is True
    assert "fish_present_no_keypoints" in first["classification_options"]

    eye_repair = first["repair_options"]["eye_mask_review"]
    assert eye_repair["argv"][:4] == [
        "scripts/py",
        "-m",
        "fisheye.tune.eye_mask_review",
        str(zarr_path),
    ]
    assert "--refined-run" in eye_repair["argv"]
    assert "refined_eye_masks_source" in eye_repair["argv"]
    assert "--frame-flag-file" in eye_repair["argv"]
    assert str(frame_flag_path) in eye_repair["argv"]

    keypoint_repair = first["repair_options"]["keypoint_review"]
    assert keypoint_repair["argv"][:4] == [
        "scripts/py",
        "-m",
        "fisheye.tune.keypoint_review",
        str(zarr_path),
    ]
    assert "--refined-run" in keypoint_repair["argv"]
    assert "refined_keypoints_checked" in keypoint_repair["argv"]
    assert "--frames" in keypoint_repair["argv"]
    assert str(frame_flag_path) in keypoint_repair["argv"]
    assert isinstance(keypoint_repair["shell"], str)
