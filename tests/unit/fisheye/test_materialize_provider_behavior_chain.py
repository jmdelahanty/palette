import json
from pathlib import Path

import pytest

from fisheye.utils import materialize_provider_behavior_chain as mod


def _task_payload(archive: Path, *, schema_version: int) -> dict[str, object]:
    return {
        "schema_id": mod.TASK_SCHEMA_ID,
        "schema_version": schema_version,
        "recording_id": "recording",
        "analysis_zarr": str(archive),
        "source_runs": {
            "keypoint": "keypoints_v1",
            "body_frame": "body_frame_v1",
            "stimulus": "stimulus_v1",
        },
        "output_runs": {
            "position": "position_v1",
            "tracking": "tracking_v1",
            "motion": "motion_v1",
            "swim_bouts": "bouts_v1",
            "stimulus_epochs_v1": "epochs_v1",
            "stimulus_epochs_v2": "epochs_v2",
            "epoch_summary": "summary_v1",
        },
        "fps": 250.0,
        "metric_disposition": mod.LINEAR_ONLY_DISPOSITION,
    }


def _write_task(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "task.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_task_v1_preserves_legacy_summary_contract(tmp_path: Path) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    payload = _task_payload(archive, schema_version=1)

    loaded = mod.load_task(_write_task(tmp_path, payload))

    assert "protocol_semantic_selection_run" not in loaded


def test_task_v1_rejects_semantic_successor_field(tmp_path: Path) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    payload = _task_payload(archive, schema_version=1)
    payload["protocol_semantic_selection_run"] = "semantic_v2"

    with pytest.raises(mod.ProviderBehaviorChainError, match="task v2"):
        mod.load_task(_write_task(tmp_path, payload))


def test_task_v2_requires_and_accepts_exact_semantic_run(tmp_path: Path) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    payload = _task_payload(archive, schema_version=2)
    path = _write_task(tmp_path, payload)
    with pytest.raises(mod.ProviderBehaviorChainError, match="requires"):
        mod.load_task(path)

    payload["protocol_semantic_selection_run"] = "semantic_v2"
    loaded = mod.load_task(_write_task(tmp_path, payload))

    assert loaded["protocol_semantic_selection_run"] == "semantic_v2"


def test_epoch_summary_uses_physical_filtered_speed(monkeypatch, tmp_path: Path) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    captured: dict[str, object] = {}

    def fake_materialize(source_zarr: Path, **kwargs: object) -> dict[str, object]:
        captured["source_zarr"] = source_zarr
        captured.update(kwargs)
        return {"status": "published"}

    monkeypatch.setattr(
        mod,
        "materialize_provider_epoch_behavior_summary",
        fake_materialize,
    )

    result = mod._summary(  # noqa: SLF001 - focused orchestration contract test
        {
            "analysis_zarr": str(archive),
            "protocol_semantic_selection_run": "semantic_v2",
            "output_runs": {
                "epoch_summary": "summary_v1",
                "stimulus_epochs_v2": "epochs_v2",
                "motion": "motion_v1",
                "swim_bouts": "bouts_v1",
            },
        },
        tmp_path / "scratch",
    )

    assert captured["source_zarr"] == archive
    assert captured["speed_level"] == mod.SUMMARY_SPEED_LEVEL == "filtered"
    assert captured["protocol_semantic_selection_run_name"] == "semantic_v2"
    assert result["status"] == "published"
