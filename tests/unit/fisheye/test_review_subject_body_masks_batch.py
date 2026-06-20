from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from fisheye.utils import review_subject_body_masks_batch as mod


class FakeArray:
    def __init__(self, data: object) -> None:
        self._data = np.asarray(data)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self._data.shape)

    def __getitem__(self, item: object) -> np.ndarray:
        return self._data[item]


class FakeGroup(dict[str, object]):
    def __init__(self) -> None:
        super().__init__()
        self.attrs: dict[str, object] = {}

    def create_group(self, name: str) -> "FakeGroup":
        group = FakeGroup()
        self[name] = group
        return group

    def create_array(self, name: str, data: object) -> FakeArray:
        arr = FakeArray(data)
        self[name] = arr
        return arr


class FakeMaskStore:
    def __init__(self, n_rows: int) -> None:
        self.n_rows = int(n_rows)


def _mk_subject_run(
    parent: FakeGroup,
    run_name: str,
    *,
    include_subject_body: bool = True,
    run_semantics: str | None = "traditional_subject_body_inference",
) -> None:
    run = parent.create_group(run_name)
    labels = ["subject_body", "swim_bladder"] if include_subject_body else ["swim_bladder"]
    run.attrs["mask_labels"] = labels
    if run_semantics is not None:
        run.attrs["run_semantics"] = run_semantics
    run.create_array("masks_roi", data=np.zeros((1, len(labels), 4, 4), dtype=np.uint8))
    run.create_array("available_channels", data=np.ones((len(labels),), dtype=bool))


def _mk_refined_run(
    parent: FakeGroup,
    run_name: str,
    *,
    source_subject_mask_run: str,
    review_state: str | None,
    include_subject_body: bool = True,
    stale_rows: list[int] | None = None,
) -> None:
    run = parent.create_group(run_name)
    labels = ["subject_body"] if include_subject_body else ["swim_bladder"]
    run.attrs["mask_labels"] = labels
    run.attrs["source_subject_mask_run"] = source_subject_mask_run
    run.create_array("masks_roi", data=np.zeros((1, len(labels), 4, 4), dtype=np.uint8))
    run.create_array("available_channels", data=np.ones((len(labels),), dtype=bool))
    if review_state is not None:
        run.attrs["component_review_statuses"] = {"subject_body": {"state": review_state}}
    if include_subject_body:
        components = run.create_group("components")
        subject_body = components.create_group("subject_body")
        if stale_rows is not None:
            subject_body.attrs["source_update_pending_rows"] = list(stale_rows)


def _install_fake_roots(
    monkeypatch: pytest.MonkeyPatch,
    roots: dict[Path, FakeGroup],
) -> None:
    monkeypatch.setattr(mod.zarr, "Group", FakeGroup)
    monkeypatch.setattr(mod, "_iter_zarr", lambda paths, recursive: list(roots))
    monkeypatch.setattr(mod.zarr, "open_group", lambda path, mode="r", use_consolidated=False: roots[Path(path)])


def test_build_plans_uses_subject_run_when_refined_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    zarr_path = tmp_path / "a_training.zarr"
    root = FakeGroup()
    parent = root.create_group("subject_mask_runs")
    _mk_subject_run(parent, "subject_a")
    parent.attrs["latest"] = "subject_a"
    _install_fake_roots(monkeypatch, {zarr_path: root})

    plans = mod._build_plans(
        [zarr_path],
        recursive=False,
        subject_run=None,
        subject_run_semantics=None,
        refined_run=None,
        status_filter="missing",
        zarr_use="training",
    )

    ok = [plan for plan in plans if plan.status == "ok"]
    assert len(ok) == 1
    assert ok[0].subject_run == "subject_a"
    assert ok[0].refined_run is None
    assert ok[0].stage_label == "subject_mask_runs -> new refined run"


def test_build_plans_prefers_existing_refined_body_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "b_training.zarr"
    root = FakeGroup()
    subject_parent = root.create_group("subject_mask_runs")
    _mk_subject_run(subject_parent, "subject_b")
    subject_parent.attrs["latest"] = "subject_b"
    refined_parent = root.create_group("refined_subject_masks_runs")
    _mk_refined_run(
        refined_parent,
        "refined_b",
        source_subject_mask_run="subject_b",
        review_state="approved",
    )
    refined_parent.attrs["latest"] = "refined_b"
    _install_fake_roots(monkeypatch, {zarr_path: root})

    plans = mod._build_plans(
        [zarr_path],
        recursive=False,
        subject_run=None,
        subject_run_semantics=None,
        refined_run=None,
        status_filter="approved",
        zarr_use="training",
    )

    ok = [plan for plan in plans if plan.status == "ok"]
    assert len(ok) == 1
    assert ok[0].subject_run == "subject_b"
    assert ok[0].refined_run == "refined_b"
    assert ok[0].review_state == "approved"
    assert ok[0].stage_label == "refined_subject_masks_runs"


def test_build_plans_filters_to_stale_refined_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_path = tmp_path / "stale_training.zarr"
    clean_path = tmp_path / "clean_training.zarr"

    stale_root = FakeGroup()
    stale_subject_parent = stale_root.create_group("subject_mask_runs")
    _mk_subject_run(stale_subject_parent, "subject_stale")
    stale_subject_parent.attrs["latest"] = "subject_stale"
    stale_refined_parent = stale_root.create_group("refined_subject_masks_runs")
    _mk_refined_run(
        stale_refined_parent,
        "refined_stale",
        source_subject_mask_run="subject_stale",
        review_state="approved",
        stale_rows=[0, 0],
    )
    stale_refined_parent.attrs["latest"] = "refined_stale"

    clean_root = FakeGroup()
    clean_subject_parent = clean_root.create_group("subject_mask_runs")
    _mk_subject_run(clean_subject_parent, "subject_clean")
    clean_subject_parent.attrs["latest"] = "subject_clean"
    clean_refined_parent = clean_root.create_group("refined_subject_masks_runs")
    _mk_refined_run(
        clean_refined_parent,
        "refined_clean",
        source_subject_mask_run="subject_clean",
        review_state="needs_review",
        stale_rows=[],
    )
    clean_refined_parent.attrs["latest"] = "refined_clean"

    _install_fake_roots(monkeypatch, {stale_path: stale_root, clean_path: clean_root})

    plans = mod._build_plans(
        [stale_path, clean_path],
        recursive=False,
        subject_run=None,
        subject_run_semantics=None,
        refined_run=None,
        status_filter="stale",
        zarr_use="training",
    )

    ok = [plan for plan in plans if plan.status == "ok"]
    assert len(ok) == 1
    assert ok[0].zarr_path == stale_path
    assert ok[0].refined_run == "refined_stale"
    assert ok[0].roi_indices == [0]


def test_pending_stale_roi_indices_uses_mask_store_count_without_dense_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = FakeGroup()
    components = run.create_group("components")
    subject_body = components.create_group("subject_body")
    subject_body.attrs["source_update_pending_rows"] = [0, 1, 1, 2, 99]
    monkeypatch.setattr(mod, "open_mask_store", lambda run_group, prefer="dense": FakeMaskStore(2))

    assert "masks_roi" not in run
    assert mod._pending_stale_roi_indices(run, "subject_body") == [0, 1]


def test_build_plans_skips_swim_only_refined_and_falls_back_to_subject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "c_training.zarr"
    root = FakeGroup()
    subject_parent = root.create_group("subject_mask_runs")
    _mk_subject_run(subject_parent, "subject_c")
    subject_parent.attrs["latest"] = "subject_c"
    refined_parent = root.create_group("refined_subject_masks_runs")
    _mk_refined_run(
        refined_parent,
        "refined_swim_only",
        source_subject_mask_run="subject_c",
        review_state=None,
        include_subject_body=False,
    )
    refined_parent.attrs["latest"] = "refined_swim_only"
    _install_fake_roots(monkeypatch, {zarr_path: root})

    plans = mod._build_plans(
        [zarr_path],
        recursive=False,
        subject_run=None,
        subject_run_semantics=None,
        refined_run=None,
        status_filter="missing",
        zarr_use="training",
    )

    ok = [plan for plan in plans if plan.status == "ok"]
    assert len(ok) == 1
    assert ok[0].refined_run is None
    assert ok[0].subject_run == "subject_c"
    assert ok[0].stage_label == "subject_mask_runs -> new refined run"


def test_build_plans_from_registry_prefers_refined_rows(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    zarr_path = tmp_path / "recordings" / "d_training.zarr"

    with sqlite3.connect(str(registry_path)) as conn:
        conn.execute(
            """
            CREATE TABLE subject_mask_component_quality_overview (
                zarr_path TEXT,
                stage_group TEXT,
                run_name TEXT,
                component_name TEXT,
                review_state TEXT,
                zarr_use TEXT,
                available INTEGER,
                source_subject_mask_run TEXT
            );
            """
        )
        conn.executemany(
            """
            INSERT INTO subject_mask_component_quality_overview
                (zarr_path, stage_group, run_name, component_name, review_state, zarr_use, available, source_subject_mask_run)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                (str(zarr_path), "subject_mask_runs", "subject_d", "subject_body", None, "training", 1, None),
                (
                    str(zarr_path),
                    "refined_subject_masks_runs",
                    "refined_d",
                    "subject_body",
                    "pending",
                    "training",
                    1,
                    "subject_d",
                ),
            ],
        )

    plans = mod._build_plans_from_registry(
        registry_path,
        roots=[tmp_path / "recordings"],
        subject_run=None,
        refined_run=None,
        status_filter="pending",
        zarr_use="training",
    )

    ok = [plan for plan in plans if plan.status == "ok"]
    assert len(ok) == 1
    assert ok[0].zarr_path == zarr_path
    assert ok[0].subject_run == "subject_d"
    assert ok[0].refined_run == "refined_d"
    assert ok[0].review_state == "pending"


def test_build_plans_subject_run_semantics_uses_latest_matching_source_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "sam_training.zarr"
    root = FakeGroup()
    parent = root.create_group("subject_mask_runs")
    _mk_subject_run(parent, "subject_old_sam", run_semantics="sam_body_mask_inference")
    _mk_subject_run(parent, "subject_new_traditional", run_semantics="traditional_subject_body_inference")
    parent.attrs["latest"] = "subject_new_traditional"
    _install_fake_roots(monkeypatch, {zarr_path: root})

    plans = mod._build_plans(
        [zarr_path],
        recursive=False,
        subject_run=None,
        subject_run_semantics="sam_body_mask_inference",
        refined_run=None,
        status_filter="missing",
        zarr_use="training",
    )

    ok = [plan for plan in plans if plan.status == "ok"]
    assert len(ok) == 1
    assert ok[0].subject_run == "subject_old_sam"
    assert ok[0].refined_run is None


def test_build_plans_subject_run_semantics_rejects_explicit_non_matching_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "explicit_training.zarr"
    root = FakeGroup()
    parent = root.create_group("subject_mask_runs")
    _mk_subject_run(parent, "subject_explicit", run_semantics="traditional_subject_body_inference")
    parent.attrs["latest"] = "subject_explicit"
    _install_fake_roots(monkeypatch, {zarr_path: root})

    plans = mod._build_plans(
        [zarr_path],
        recursive=False,
        subject_run="subject_explicit",
        subject_run_semantics="sam_body_mask_inference",
        refined_run=None,
        status_filter="missing",
        zarr_use="training",
    )

    assert len(plans) == 1
    assert plans[0].status == "missing"
    assert plans[0].reason == "subject_mask_runs/subject_explicit run_semantics != sam_body_mask_inference"


def test_build_plans_from_registry_uses_subject_row_when_refined_absent(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    zarr_path = tmp_path / "recordings" / "e_training.zarr"

    with sqlite3.connect(str(registry_path)) as conn:
        conn.execute(
            """
            CREATE TABLE subject_mask_component_quality_overview (
                zarr_path TEXT,
                stage_group TEXT,
                run_name TEXT,
                component_name TEXT,
                review_state TEXT,
                zarr_use TEXT,
                available INTEGER,
                source_subject_mask_run TEXT
            );
            """
        )
        conn.execute(
            """
            INSERT INTO subject_mask_component_quality_overview
                (zarr_path, stage_group, run_name, component_name, review_state, zarr_use, available, source_subject_mask_run)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (str(zarr_path), "subject_mask_runs", "subject_e", "subject_body", None, "training", 1, None),
        )

    plans = mod._build_plans_from_registry(
        registry_path,
        roots=[tmp_path / "recordings"],
        subject_run=None,
        refined_run=None,
        status_filter="missing",
        zarr_use="training",
    )

    ok = [plan for plan in plans if plan.status == "ok"]
    assert len(ok) == 1
    assert ok[0].subject_run == "subject_e"
    assert ok[0].refined_run is None
    assert ok[0].stage_label == "subject_mask_runs -> new refined run"


def test_filter_stale_registry_plans_marks_missing_source_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recordings" / "stale_training.zarr"
    (zarr_path / "subject_mask_runs" / "current_subject").mkdir(parents=True)
    plan = mod.ReviewPlan(
        zarr_path=zarr_path,
        subject_run="missing_registry_subject",
        refined_run=None,
        review_state=None,
        status="ok",
        stage_label="subject_mask_runs -> new refined run",
    )

    [filtered] = mod._filter_stale_registry_plans([plan])

    assert filtered.status == "filtered"
    assert filtered.reason == "registry source subject_mask_runs/missing_registry_subject missing on disk"


def test_build_plans_from_registry_honors_explicit_refined_run(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    zarr_path = tmp_path / "recordings" / "f_training.zarr"

    with sqlite3.connect(str(registry_path)) as conn:
        conn.execute("CREATE TABLE datasets (dataset_id TEXT, zarr_path TEXT, status TEXT);")
        conn.execute(
            """
            CREATE TABLE subject_mask_component_quality (
                dataset_id TEXT,
                stage_group TEXT,
                run_name TEXT,
                component_name TEXT,
                review_state TEXT,
                zarr_use TEXT,
                available INTEGER,
                source_subject_mask_run TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO datasets (dataset_id, zarr_path, status) VALUES (?, ?, ?);",
            ("ds_f", str(zarr_path), None),
        )
        conn.executemany(
            """
            INSERT INTO subject_mask_component_quality
                (dataset_id, stage_group, run_name, component_name, review_state, zarr_use, available, source_subject_mask_run)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                (
                    "ds_f",
                    "refined_subject_masks_runs",
                    "refined_target",
                    "subject_body",
                    None,
                    "training",
                    1,
                    "subject_f",
                ),
                (
                    "ds_f",
                    "refined_subject_masks_runs",
                    "refined_other",
                    "subject_body",
                    "approved",
                    "training",
                    1,
                    "subject_f",
                ),
            ],
        )

    plans = mod._build_plans_from_registry(
        registry_path,
        roots=[tmp_path / "recordings"],
        subject_run=None,
        refined_run="refined_target",
        status_filter="missing",
        zarr_use="training",
    )

    ok = [plan for plan in plans if plan.status == "ok"]
    assert len(ok) == 1
    assert ok[0].zarr_path == zarr_path
    assert ok[0].refined_run == "refined_target"
    assert ok[0].subject_run == "subject_f"


def _viewer_args() -> argparse.Namespace:
    return argparse.Namespace(
        padding=18,
        scale_percent=220,
        edit_zoom=8,
        review_state="approved",
        review_method="manual",
        review_intended_use="training",
        crop_run=None,
        reviewer=None,
        review_notes=None,
    )


def test_viewer_cmd_passes_subject_and_refined_runs(tmp_path: Path) -> None:
    plan = mod.ReviewPlan(
        zarr_path=tmp_path / "recordings" / "g_training.zarr",
        subject_run="subject_g",
        refined_run="refined_g",
        review_state=None,
        status="ok",
    )
    cmd = mod._viewer_cmd(_viewer_args(), plan)
    assert "fisheye.tune.refined_subject_mask_review" in cmd
    assert "--subject-run" in cmd
    assert cmd[cmd.index("--subject-run") + 1] == "subject_g"
    assert "--refined-run" in cmd
    assert cmd[cmd.index("--refined-run") + 1] == "refined_g"
    assert "--components" in cmd
    assert cmd[cmd.index("--components") + 1] == "subject_body"
    assert "--component" in cmd
    assert cmd[cmd.index("--component") + 1] == "subject_body"


def test_viewer_cmd_omits_refined_run_for_first_pass(tmp_path: Path) -> None:
    plan = mod.ReviewPlan(
        zarr_path=tmp_path / "recordings" / "h_training.zarr",
        subject_run="subject_h",
        refined_run=None,
        review_state=None,
        status="ok",
    )
    cmd = mod._viewer_cmd(_viewer_args(), plan)
    assert "--subject-run" in cmd
    assert "--refined-run" not in cmd


def test_viewer_cmd_passes_first_roi_index_for_stale_subset(tmp_path: Path) -> None:
    plan = mod.ReviewPlan(
        zarr_path=tmp_path / "recordings" / "i_training.zarr",
        subject_run="subject_i",
        refined_run="refined_i",
        review_state="needs_review",
        status="ok",
        roi_indices=[3, 7, 12],
    )
    cmd = mod._viewer_cmd(_viewer_args(), plan)
    assert "--roi-index" in cmd
    assert cmd[cmd.index("--roi-index") + 1] == "3"


def test_viewer_cmd_translates_scale_percent_to_display_scale(tmp_path: Path) -> None:
    plan = mod.ReviewPlan(
        zarr_path=tmp_path / "recordings" / "j_training.zarr",
        subject_run="subject_j",
        refined_run="refined_j",
        review_state="pending",
        status="ok",
    )
    cmd = mod._viewer_cmd(_viewer_args(), plan)
    assert "--display-scale" in cmd
    assert cmd[cmd.index("--display-scale") + 1] == "2.2"
