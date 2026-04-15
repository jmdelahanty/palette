from __future__ import annotations

from pathlib import Path

from fisheye.shared import subject_mask_registry_status as mod


class _FakeGroup(dict[str, object]):
    def __init__(self, *, attrs: dict[str, object] | None = None) -> None:
        super().__init__()
        self.attrs = attrs or {}

    def get(self, key: str, default: object = None) -> object:
        return super().get(key, default)


def test_emit_subject_mask_stage_completion_packages_runtime_details(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_emit_stage_completion(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured["args"] = args
        captured["kwargs"] = kwargs
        return True

    monkeypatch.setattr(mod, "emit_stage_completion", _fake_emit_stage_completion)

    root = _FakeGroup()
    run_group = _FakeGroup(
        attrs={
            "summary_statistics": {
                "rows_total": 12,
                "rows_with_nonempty_masks": 11,
                "duration_seconds": 7.5,
            },
            "subject_mask_review_status": {
                "state": "approved",
                "method": "manual",
                "intended_use": "training",
            },
            "method": "subject_mask_threshold_lr_v1",
            "source_crop_run": "crop_001",
            "source_keypoints_run": "refined_keypoints_001",
            "source_keypoint_group": "refined_keypoints_runs",
            "label_schema_id": "subject_v1_union",
            "run_semantics": "subject_masks_union_v1",
        }
    )

    ok = mod.emit_subject_mask_stage_completion(
        root,
        Path("/tmp/example_training.zarr"),
        run_group=run_group,
        run_name="subject_masks_001",
        source="runtime_subject_masks_test",
        console=None,
        invalidate_on_ok=True,
    )

    assert ok is True
    kwargs = captured["kwargs"]
    assert kwargs["step_name"] == "subject_masks"
    assert kwargs["status"] == "ok"
    assert kwargs["run_name"] == "subject_masks_001"
    assert kwargs["method"] == "subject_mask_threshold_lr_v1"
    assert kwargs["review_status_json"] == {
        "state": "approved",
        "method": "manual",
        "intended_use": "training",
    }
    assert kwargs["details_json"] == {
        "reason": "present",
        "latest_selector": "runtime_subject_mask_write",
        "source_crop_run": "crop_001",
        "source_keypoints_run": "refined_keypoints_001",
        "source_keypoint_group": "refined_keypoints_runs",
        "label_schema_id": "subject_v1_union",
        "run_semantics": "subject_masks_union_v1",
        "rows_total": 12,
        "rows_with_nonempty_masks": 11,
        "duration_seconds": 7.5,
    }


def test_emit_refined_subject_mask_stage_completion_includes_stale_payload_details(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_emit_stage_completion(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured["args"] = args
        captured["kwargs"] = kwargs
        return True

    monkeypatch.setattr(mod, "emit_stage_completion", _fake_emit_stage_completion)

    root = _FakeGroup()
    run_group = _FakeGroup(
        attrs={
            "summary_statistics": {"duration_seconds": 5.0},
            "refined_subject_mask_review_status": {
                "state": "needs_review",
                "method": "manual",
                "intended_use": "training",
            },
            "source_subject_mask_stale": {
                "state": "stale",
                "reason": "source_subject_mask_rows_changed",
            },
            "method": "refine_subject_masks",
            "source_subject_mask_run": "subject_masks_001",
            "label_schema_id": "subject_v1_union",
            "mask_labels": ["subject_body", "swim_bladder"],
        }
    )

    ok = mod.emit_refined_subject_mask_stage_completion(
        root,
        Path("/tmp/example_training.zarr"),
        run_group=run_group,
        run_name="refined_subject_masks_001",
        source="runtime_refined_subject_masks_test",
        console=None,
        invalidate_on_ok=True,
    )

    assert ok is True
    kwargs = captured["kwargs"]
    assert kwargs["step_name"] == "refined_subject_masks"
    assert kwargs["status"] == "ok"
    assert kwargs["run_name"] == "refined_subject_masks_001"
    assert kwargs["method"] == "refine_subject_masks"
    assert kwargs["review_status_json"] == {
        "state": "needs_review",
        "method": "manual",
        "intended_use": "training",
    }
    assert kwargs["details_json"] == {
        "reason": "present",
        "latest_selector": "runtime_refined_subject_mask_write",
        "source_subject_mask_run": "subject_masks_001",
        "label_schema_id": "subject_v1_union",
        "component_names": ["subject_body", "swim_bladder"],
        "stale_state": "stale",
        "stale_reason": "source_subject_mask_rows_changed",
        "duration_seconds": 5.0,
    }
