from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.shared import derived_analysis_registry_status as mod


class _Run:
    def __init__(self, attrs: dict[str, object]) -> None:
        self.attrs = attrs


def test_eye_angle_completion_uses_canonical_stage_and_run(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_emit(root, zarr_path, **kwargs):  # type: ignore[no-untyped-def]
        captured.update({"root": root, "zarr_path": zarr_path, **kwargs})
        return True

    monkeypatch.setattr(mod, "emit_stage_completion", fake_emit)
    root = object()
    run = _Run(
        {
            "method": "ellipse_and_centroid_eye_angles",
            "source_subject_shape_run": "shape_1",
            "source_keypoints_run": "kp_1",
            "num_detections": 12,
            "num_frames": 20,
        }
    )

    wrote = mod.emit_eye_angle_stage_completion(
        root,  # type: ignore[arg-type]
        Path("/archive.zarr"),
        run_group=run,  # type: ignore[arg-type]
        run_name="eye_1",
        source="unit_test",
    )

    assert wrote is True
    assert captured["step_name"] == "eye_angles"
    assert captured["run_name"] == "eye_1"
    assert captured["status"] == "ok"
    assert captured["source"] == "unit_test"
    assert captured["details_json"] == {
        "reason": "present",
        "latest_selector": "runtime_eye_angle_publication",
        "source_subject_shape_run": "shape_1",
        "source_keypoints_run": "kp_1",
        "num_detections": 12,
        "num_frames": 20,
    }


def test_track_completion_preserves_scope_in_registry_run_name(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_emit(root, zarr_path, **kwargs):  # type: ignore[no-untyped-def]
        captured.update({"root": root, "zarr_path": zarr_path, **kwargs})
        return True

    monkeypatch.setattr(mod, "emit_stage_completion", fake_emit)
    run = _Run(
        {
            "method": "track_kinematics_offline",
            "source_tracking_run": "tracks_1",
            "num_tracks": 2,
            "fps": 100.0,
        }
    )

    wrote = mod.emit_track_kinematics_stage_completion(
        object(),  # type: ignore[arg-type]
        Path("/archive.zarr"),
        run_group=run,  # type: ignore[arg-type]
        run_name="track_1",
        run_type="offline",
        source="unit_test",
    )

    assert wrote is True
    assert captured["step_name"] == "track_kinematics"
    assert captured["run_name"] == "offline/track_1"
    assert captured["method"] == "track_kinematics_offline"
    assert captured["details_json"] == {
        "reason": "present",
        "latest_selector": "runtime_track_kinematics_publication",
        "run_type": "offline",
        "source_tracking_run": "tracks_1",
        "num_tracks": 2,
        "fps": 100.0,
    }


@pytest.mark.parametrize(
    ("run_name", "run_type"),
    (("scope/run", "offline"), ("run", "invalid")),
)
def test_track_completion_rejects_ambiguous_identity(
    run_name: str,
    run_type: str,
) -> None:
    with pytest.raises(ValueError):
        mod.emit_track_kinematics_stage_completion(
            object(),  # type: ignore[arg-type]
            Path("/archive.zarr"),
            run_group=_Run({}),  # type: ignore[arg-type]
            run_name=run_name,
            run_type=run_type,
            source="unit_test",
        )


def test_generic_completion_projects_exact_maintained_stage(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_emit(root, zarr_path, **kwargs):  # type: ignore[no-untyped-def]
        captured.update({"root": root, "zarr_path": zarr_path, **kwargs})
        return True

    monkeypatch.setattr(mod, "emit_stage_completion", fake_emit)
    run = _Run(
        {
            "schema_id": "palette.swim_bout_runs",
            "schema_version": 8,
            "layout": "compact_tabular_v2",
            "method": "multi_level_bout_detection",
            "source_refs": {"track_kinematics_run": "track_1"},
            "array_schema_manifest_sha256": "a" * 64,
        }
    )

    wrote = mod.emit_derived_analysis_stage_completion(
        object(),  # type: ignore[arg-type]
        Path("/archive.zarr"),
        stage_id="swim_bouts",
        run_group=run,  # type: ignore[arg-type]
        run_name="bouts_1",
        source="unit_test",
    )

    assert wrote is True
    assert captured["step_name"] == "swim_bouts"
    assert captured["run_name"] == "bouts_1"
    assert captured["method"] == "multi_level_bout_detection"
    assert captured["details_json"] == {
        "reason": "present",
        "latest_selector": "serialized_derived_analysis_finalizer",
        "artifact_path": "analysis/swim_bout_runs/bouts_1",
        "schema_id": "palette.swim_bout_runs",
        "schema_version": 8,
        "layout": "compact_tabular_v2",
        "source_refs": {"track_kinematics_run": "track_1"},
        "array_schema_manifest_digest": "a" * 64,
    }


def test_generic_completion_rejects_non_derived_stage() -> None:
    with pytest.raises(ValueError, match="not a derived-analysis stage"):
        mod.emit_derived_analysis_stage_completion(
            object(),  # type: ignore[arg-type]
            Path("/archive.zarr"),
            stage_id="refined_keypoints",
            run_group=_Run({}),  # type: ignore[arg-type]
            run_name="kp_1",
            source="unit_test",
        )
