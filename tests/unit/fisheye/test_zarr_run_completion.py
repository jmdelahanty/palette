from __future__ import annotations

from pathlib import Path

from fisheye.registry.stage_complete import emit_stage_completion
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_RUNNING,
    is_run_complete,
    mark_run_complete,
    mark_run_started,
    note_pending_latest,
    resolve_latest_complete_run_name,
)


class FakeGroup(dict[str, object]):
    def __init__(self, *, attrs: dict[str, object] | None = None) -> None:
        super().__init__()
        self.attrs = attrs or {}

    def group_keys(self):
        return [key for key, value in self.items() if isinstance(value, FakeGroup)]

    @property
    def path(self) -> str:
        return "/fake"


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


def test_emit_stage_completion_accepts_complete_opted_in_run(tmp_path: Path) -> None:
    root = FakeGroup()
    detect_parent = FakeGroup()
    run = FakeGroup()
    root["detect_runs"] = detect_parent
    detect_parent["detect_001"] = run
    mark_run_started(run, run_name="detect_001", stage="detect")
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
