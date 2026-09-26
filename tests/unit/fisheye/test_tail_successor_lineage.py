"""Stranded-edit detection over a mask-Apply version lineage."""

from __future__ import annotations

import numpy as np

from fisheye.labeling.tail_successor_lineage import (
    FamilyLineage,
    stranded_keypoint_rows,
)

K = 4


def _run(points, manual):
    return {"keypoints_roi": np.asarray(points, float), "keypoint_manual_edit": np.asarray(manual, bool)}


def _root(**runs):
    return {f"refined_keypoints_runs/{name}": run for name, run in runs.items()}


def _checkpoint(run, roi, applied, cid="c"):
    return {
        "state": "applied",
        "target_run_path": f"refined_keypoints_runs/{run}",
        "roi_idx": roi,
        "applied_at_utc": applied,
        "checkpoint_id": cid,
        "task_id": f"task-{run}",
    }


# A -> B (created t=2) -> C (created t=4); D is a sibling of B from A (t=3).
LINEAGE = FamilyLineage(
    "refined_keypoints_runs",
    parent={"A": None, "B": "A", "C": "B", "D": "A"},
    started_at={"A": "0", "B": "2", "C": "4", "D": "3"},
)


def _base():
    return np.zeros((2, K, 2)) + 10.0, np.zeros((2, K), bool)


def test_ancestor_edit_before_snapshot_is_present_and_after_is_stranded():
    points, manual = _base()
    edited = points.copy(); edited[0, 1] = [12.0, 13.0]
    edited_manual = manual.copy(); edited_manual[0, 1] = True
    root = _root(A=_run(edited, edited_manual), B=_run(points, manual), C=_run(points, manual), D=_run(points, manual))
    assert stranded_keypoint_rows(
        root=root, checkpoints=[_checkpoint("A", 0, "1")], lineage=LINEAGE, target_run="C"
    ) == []
    rows = stranded_keypoint_rows(
        root=root, checkpoints=[_checkpoint("A", 0, "5")], lineage=LINEAGE, target_run="C"
    )
    assert [(r.roi_idx, r.source_run, r.disposition, r.manual_keypoints) for r in rows] == [(0, "A", "carry", [1])]
    assert rows[0].points[1] == [12.0, 13.0] and rows[0].points[0] == [10.0, 10.0]


def test_sibling_branch_edit_is_stranded_unless_target_lineage_edit_is_newer():
    points, manual = _base()
    edited = points.copy(); edited[1, 2] = [1.0, 2.0]
    edited_manual = manual.copy(); edited_manual[1, 2] = True
    root = _root(A=_run(points, manual), B=_run(points, manual), C=_run(points, manual), D=_run(edited, edited_manual))
    carry = stranded_keypoint_rows(
        root=root, checkpoints=[_checkpoint("D", 1, "3.5")], lineage=LINEAGE, target_run="C"
    )
    assert [r.disposition for r in carry] == ["carry"]
    newer = stranded_keypoint_rows(
        root=root,
        checkpoints=[_checkpoint("D", 1, "3.5"), _checkpoint("C", 1, "6", cid="n")],
        lineage=LINEAGE,
        target_run="C",
    )
    assert [r.disposition for r in newer] == ["target_edit_newer"]


def test_identical_value_is_already_present_and_manual_clear_is_not_carryable():
    points, manual = _base()
    manual_same = manual.copy(); manual_same[0, 0] = True
    root = _root(A=_run(points, manual), B=_run(points, manual), C=_run(points, manual_same), D=_run(points, manual_same))
    rows = stranded_keypoint_rows(
        root=root, checkpoints=[_checkpoint("D", 0, "3.5")], lineage=LINEAGE, target_run="C"
    )
    assert [r.disposition for r in rows] == ["already_present"]
    cleared = points.copy(); cleared[0, 3] = np.nan
    cleared_manual = manual.copy(); cleared_manual[0, 3] = True
    root["refined_keypoints_runs/D"] = _run(cleared, cleared_manual)
    rows = stranded_keypoint_rows(
        root=root, checkpoints=[_checkpoint("D", 0, "3.5")], lineage=LINEAGE, target_run="C"
    )
    assert [r.disposition for r in rows] == ["not_carryable"]


def test_lineage_helpers():
    assert LINEAGE.component("C") == frozenset("ABCD")
    assert LINEAGE.newest_leaf(LINEAGE.component("A")) == "C"
    assert LINEAGE.snapshot_cutoffs("C") == {"C": None, "B": "4", "A": "2"}


def test_pending_edit_in_target_blocks_carry():
    points, manual = _base()
    edited = points.copy(); edited[1, 2] = [1.0, 2.0]
    edited_manual = manual.copy(); edited_manual[1, 2] = True
    root = _root(A=_run(points, manual), B=_run(points, manual), C=_run(points, manual), D=_run(edited, edited_manual))
    pending = [{**_checkpoint("C", 1, None), "state": "active"}]
    rows = stranded_keypoint_rows(
        root=root, checkpoints=[_checkpoint("D", 1, "3.5")], lineage=LINEAGE,
        target_run="C", pending_checkpoints=pending,
    )
    assert [r.disposition for r in rows] == ["target_edit_pending"]


def test_old_draft_applied_late_does_not_beat_newer_work_in_newest_version():
    """The 2026-09-26 arena 2 shape: 438a -> 94a2.

    A draft edited in the older version at 17:06 but applied at 19:30 must
    lose to the same row edited later (17:30) in the newest version and
    applied earlier (19:00). Ranking by apply time would carry the draft.
    """
    from fisheye.labeling.tail_successor_lineage import checkpoint_edit_times

    lineage = FamilyLineage(
        "refined_keypoints_runs",
        parent={"old": None, "new": "old"},
        started_at={"old": "16:00", "new": "16:57"},
    )
    points, manual = _base()
    draft = points.copy(); draft[0, 1] = [12.0, 12.0]
    draft_manual = manual.copy(); draft_manual[0, 1] = True
    redo = points.copy(); redo[0, 1] = [13.0, 13.0]
    redo_manual = manual.copy(); redo_manual[0, 1] = True
    root = _root(old=_run(draft, draft_manual), new=_run(redo, redo_manual))
    checkpoints = [
        {**_checkpoint("old", 0, "19:30", cid="draft"), "task_id": "task-old"},
        {**_checkpoint("new", 0, "19:00", cid="redo"), "task_id": "task-new"},
    ]
    saves = [
        {"task_id": "task-old", "created_at_utc": "17:06", "target": {"roi_idx": 0}},
        {"task_id": "task-new", "created_at_utc": "17:30", "target": {"roi_idx": 0}},
    ]
    times = checkpoint_edit_times(checkpoints, saves)
    assert times == {"draft": "17:06", "redo": "17:30"}
    rows = stranded_keypoint_rows(
        root=root, checkpoints=checkpoints, lineage=lineage, target_run="new", edit_times=times
    )
    assert [r.disposition for r in rows] == ["target_edit_newer"]
    # Without edit times (apply-time ranking) the draft would wrongly win.
    naive = stranded_keypoint_rows(root=root, checkpoints=checkpoints, lineage=lineage, target_run="new")
    assert [r.disposition for r in naive] == ["carry"]


def test_edit_time_prefers_recorded_metadata_and_carried_source():
    from fisheye.labeling.tail_successor_lineage import checkpoint_edit_times

    checkpoints = [
        {**_checkpoint("x", 0, "20:00", cid="a"), "metadata": {"edited_at_utc": "18:00"}},
        {**_checkpoint("x", 1, "20:00", cid="b"),
         "metadata": {"edited_at_utc": "20:00", "carried_from": {"source_edited_at_utc": "10:00"}}},
        {**_checkpoint("x", 2, "20:00", cid="c")},
    ]
    assert checkpoint_edit_times(checkpoints, []) == {"a": "18:00", "b": "10:00", "c": "20:00"}
