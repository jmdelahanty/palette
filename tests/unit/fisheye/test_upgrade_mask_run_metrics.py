"""Cheap-metric mask runs: upgrade to full metrics, then finish a pending Apply.

Reproduces the labeling_work.sqlite case: a run finalized with cheap metrics
lets Apply write pixels but refuses the Apply QC, leaving its effects owed.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from fisheye.labeling import web
from fisheye.labeling.admin_apply_on_behalf import apply_on_behalf, apply_plan
from fisheye.labeling.upgrade_mask_run_metrics import (
    MetricUpgradeRefused,
    _masks_sha256,
    upgrade_plan,
    upgrade_to_full,
)
from fisheye.refinement import finalize_subject_masks as finalizer
from tests.unit.fisheye.test_labeling_web_routes import _running_server
from tests.unit.fisheye.test_mask_tail_apply_refresh import browser_context, reviewed_archive  # noqa: F401
from tests.unit.fisheye.test_web_mask_tail_apply_integration import request


@pytest.fixture
def cheap_run_with_saved_row(reviewed_archive, tmp_path):
    path, root, initial = reviewed_archive
    run_name = initial["paths"]["mask_edit"].split("/")[1]
    finalizer.refresh_refined_subject_mask_metrics_run(root, refined_run=run_name, metric_level="cheap")
    assert root[initial["paths"]["mask_edit"]].attrs["component_metric_level"] == "cheap"
    store, _runtime = browser_context(reviewed_archive, tmp_path)
    store.upsert_labeling_user(user_id="reviewer", status="active")
    mask_run = root[initial["paths"]["mask_edit"]]
    edited = np.asarray(mask_run["masks_roi"][1, 0]).copy()
    edited[4:9, 100:107] = 1
    session = store.create_session(task_id="original-mask", user="reviewer")
    route = f"/api/sessions/{session.session_id}/subject-mask"
    with _running_server(store, user="reviewer") as base:
        status, nav = request(base, route + "/nav", {"position": 1})
        assert status == 200, nav
        status, saved = request(base, route + "/save", {
            "mask": web._raw_array_payload(edited), "target_token": nav["state"]["target_token"],
        })
        assert status == 200, saved
    store.close_session(session_id=session.session_id, user="reviewer")
    try:
        yield store, path, run_name, edited
    finally:
        store.close()


def _receipt(store, apply_id):
    row = store.conn.execute(
        "SELECT state, secondary_effects_state FROM labeling_checkpoint_apply_receipts WHERE apply_id = ?;",
        (apply_id,),
    ).fetchone()
    return tuple(row) if row else None


def test_cheap_run_apply_is_refused_then_upgrade_and_retry_complete_it(cheap_run_with_saved_row, tmp_path):
    store, path, run_name, edited = cheap_run_with_saved_row

    first = apply_on_behalf(store.path, "original-mask", actor="operator", skip_backup=True)["outcome"]
    assert not first["ok"] and "requires full component metrics" in first["details"]
    apply_id = first["apply_id"]
    # Pixels were written; the QC/effects are owed, which blocks a new Apply.
    run = zarr.open_group(str(path), mode="r", use_consolidated=False)[f"refined_subject_masks_runs/{run_name}"]
    np.testing.assert_array_equal(np.asarray(run["masks_roi"][1, 0]), edited)
    assert _receipt(store, apply_id) == ("applied", "pending")
    assert any("--retry-apply-id" in r for r in apply_plan(store, "original-mask")["refusals"])

    plan = upgrade_plan(store.path, path, run_name)
    assert plan["ok"] and plan["run_metric_level"] == "cheap"
    tasks = {t["task_id"]: t for t in plan["tasks"]}
    assert tasks["original-mask"]["pending_effect_apply_ids"] == [apply_id]
    pixels_before = _masks_sha256(run)

    report = upgrade_to_full(store.path, path, run_name, backup_dir=tmp_path / "metric-backups")

    run = zarr.open_group(str(path), mode="r", use_consolidated=False)[f"refined_subject_masks_runs/{run_name}"]
    assert run.attrs["component_metric_level"] == "full"
    assert all(run["components"][name].attrs["component_metric_level"] == "full" for name in plan["mask_labels"])
    assert run.attrs["metrics_stale"] is False and run.attrs["contours_stale"] is False
    assert _masks_sha256(run) == pixels_before == report["masks_roi_sha256"]
    assert run.attrs["edit_revision"] == report["edit_revision"]
    assert run.attrs["component_metric_level_upgrade"]["from"] == "cheap"
    backup = zarr.open_group(report["backup_path"], mode="r", use_consolidated=False)
    assert backup.attrs["component_metric_level"] == "cheap"
    np.testing.assert_array_equal(np.asarray(backup["masks_roi"]), np.asarray(run["masks_roi"]))

    retry = apply_on_behalf(
        store.path, "original-mask", actor="operator", skip_backup=True, retry_apply_id=apply_id,
    )["outcome"]
    assert retry["ok"], retry
    assert retry["apply_id"] == apply_id and retry["retry"] is True
    assert _receipt(store, apply_id) == ("applied", "complete")
    run = zarr.open_group(str(path), mode="r", use_consolidated=False)[f"refined_subject_masks_runs/{run_name}"]
    assert run.attrs["browser_apply_qc_policy"]["metric_level"] == "full"
    np.testing.assert_array_equal(np.asarray(run["masks_roi"][1, 0]), edited)


def test_upgrade_refuses_open_sessions_and_full_runs(cheap_run_with_saved_row, tmp_path):
    store, path, run_name, _edited = cheap_run_with_saved_row
    store.create_session(task_id="original-mask", user="reviewer")
    with pytest.raises(MetricUpgradeRefused, match="open editor sessions"):
        upgrade_to_full(store.path, path, run_name, backup_dir=tmp_path / "b")
    run = zarr.open_group(str(path), mode="r", use_consolidated=False)[f"refined_subject_masks_runs/{run_name}"]
    assert run.attrs["component_metric_level"] == "cheap"
    assert not (tmp_path / "b").exists()


def test_retry_must_name_the_tasks_unfinished_apply(cheap_run_with_saved_row):
    store, *_ = cheap_run_with_saved_row
    plan = apply_plan(store, "original-mask", retry_apply_id="not-a-real-apply")
    assert not plan["ok"] and any("retry needs exactly" in r for r in plan["refusals"])
