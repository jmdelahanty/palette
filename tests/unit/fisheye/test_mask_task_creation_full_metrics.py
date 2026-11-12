"""New subject-mask tasks get full component metrics on their run automatically.

Mask Apply refuses runs that declare cheap metrics, so every task-creation
command upgrades the target run (or warns when it cannot yet).
"""

from __future__ import annotations

import inspect
import json

import pytest
import zarr

from fisheye.labeling import upgrade_mask_run_metrics as upgrade_mod
from fisheye.labeling import web
from fisheye.labeling.upgrade_mask_run_metrics import ensure_full_metrics_for_tasks
from fisheye.refinement import finalize_subject_masks as finalizer
from tests.unit.fisheye.test_mask_tail_apply_refresh import browser_context, reviewed_archive  # noqa: F401


@pytest.fixture
def cheap_run(reviewed_archive, tmp_path, monkeypatch):
    path, root, initial = reviewed_archive
    run_name = initial["paths"]["mask_edit"].split("/")[1]
    finalizer.refresh_refined_subject_mask_metrics_run(root, refined_run=run_name, metric_level="cheap")
    store, _runtime = browser_context(reviewed_archive, tmp_path)
    monkeypatch.setattr(upgrade_mod, "DEFAULT_BACKUP_DIR", tmp_path / "metric-backups")
    try:
        yield store, path, run_name, store.get_task("original-mask")["scope"]
    finally:
        store.close()


def _level(path, run_name):
    run = zarr.open_group(str(path), mode="r", use_consolidated=False)[f"refined_subject_masks_runs/{run_name}"]
    return run.attrs.get("component_metric_level")


def _add_task(store, run_name, scope, task_id, capsys):
    code = web.main([
        "--store", str(store.path), "add-task", "--task-id", task_id, "--recording-id", "rec",
        "--workflow-kind", "subject_mask_component", "--run-name", run_name, "--component-name", "subject_body",
        "--stage-group", "refined_subject_masks_runs", "--scope-json", json.dumps(scope), "--actor", "operator",
    ])
    return code, json.loads(capsys.readouterr().out)


def test_add_task_upgrades_a_cheap_run_and_a_second_task_is_a_no_op(cheap_run, tmp_path, capsys):
    store, path, run_name, scope = cheap_run
    assert _level(path, run_name) == "cheap"

    code, report = _add_task(store, run_name, scope, "new-mask-task", capsys)
    assert code == 0 and report["task"]["task_id"] == "new-mask-task"
    [upgrade] = report["subject_mask_metric_upgrades"]
    assert upgrade["status"] == "upgraded" and upgrade["task_ids"] == ["new-mask-task"]
    assert upgrade["backup_path"].startswith(str(tmp_path / "metric-backups"))
    assert _level(path, run_name) == "full"
    assert not any(w["code"] == "subject_mask_task_run_metrics_not_full" for w in report["warnings"])

    code, report = _add_task(store, run_name, scope, "another-mask-task", capsys)
    assert code == 0 and [u["status"] for u in report["subject_mask_metric_upgrades"]] == ["already_full"]


def test_open_session_on_the_run_leaves_it_cheap_and_warns(cheap_run, capsys):
    store, path, run_name, scope = cheap_run
    store.create_session(task_id="original-mask", user="reviewer")

    code, report = _add_task(store, run_name, scope, "new-mask-task", capsys)
    assert code == 0 and store.get_task("new-mask-task") is not None
    [upgrade] = report["subject_mask_metric_upgrades"]
    assert upgrade["status"] == "not_upgraded" and any("open editor sessions" in r for r in upgrade["refusals"])
    assert "subject_mask_task_run_metrics_not_full" in report["warning_codes"]
    assert _level(path, run_name) == "cheap"


def test_only_open_subject_mask_tasks_are_considered(cheap_run):
    store, path, run_name, scope = cheap_run
    tasks = [
        {"task_id": "k", "workflow_kind": "keypoints", "state": "pending", "scope": scope},
        {"task_id": "done", "workflow_kind": "subject_mask_component", "state": "complete", "scope": scope},
        None,
    ]
    assert ensure_full_metrics_for_tasks(store.path, tasks) == []
    assert _level(path, run_name) == "cheap"


@pytest.mark.parametrize("command", ["add-task", "import-tasks", "import-batch-plan", "generate-subject-mask-tasks"])
def test_every_task_creating_command_upgrades_mask_runs(command):
    source = inspect.getsource(web.main)
    branch = source.split(f'if args.command == "{command}":', 1)[1].split("if args.command ==", 1)[0]
    assert "_attach_mask_task_metric_upgrades(" in branch
