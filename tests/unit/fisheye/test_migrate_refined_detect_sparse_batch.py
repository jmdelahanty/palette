from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fisheye.utils import migrate_refined_detect_sparse as single
from fisheye.utils import migrate_refined_detect_sparse_batch as mod


class _FakeGroup:
    def __init__(self, children: dict[str, "_FakeGroup"] | None = None) -> None:
        self._children: dict[str, _FakeGroup] = children or {}
        self.attrs: dict[str, Any] = {}

    def create_group(self, name: str) -> "_FakeGroup":
        child = _FakeGroup()
        self._children[name] = child
        return child

    def get(self, name: str):
        return self._children.get(name)

    def group_keys(self):
        return list(self._children.keys())

    def keys(self):
        return self._children.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._children

    def __getitem__(self, key: str) -> "_FakeGroup":
        return self._children[key]


def _build_root(
    *,
    zarr_use: str = "analysis",
    refined_latest: str | None = "refined_detect_001",
    sparse: bool = False,
    legacy: bool = False,
    create_source_run: bool = True,
) -> _FakeGroup:
    root = _FakeGroup()
    root.attrs["zarr_use"] = zarr_use
    refined_parent = root.create_group("refined_detect_runs")
    if refined_latest is not None:
        refined_parent.attrs["latest"] = refined_latest
        if create_source_run or refined_latest != "refined_detect_001":
            refined = refined_parent.create_group(refined_latest)
            if sparse:
                refined.create_group("instances")
                refined.create_group("source_detections")
            if legacy:
                refined.create_group("manual_a")
        if refined_latest != "refined_detect_001":
            refined_parent.create_group("refined_detect_001")
    return root


def test_plan_one_skips_already_migrated_runs_without_force(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "already_sparse.zarr"
    root = _build_root(sparse=True)

    monkeypatch.setattr(mod, "_open_root", lambda path, mode="r": root)  # noqa: ARG005
    monkeypatch.setattr(mod, "has_sparse_curated_refined_detect_instances_arrays", lambda run: "instances" in run)
    monkeypatch.setattr(mod, "has_curated_refined_source_detections_projection", lambda run: "source_detections" in run)

    plan = mod._plan_one(
        zarr_path,
        zarr_use="analysis",
        refined_run=None,
        detect_run=None,
        quality_run=None,
        allow_missing_quality=False,
        ignore_legacy_groups=False,
        promote_latest_requested=True,
        force_promote_nonlatest=False,
        force=False,
    )

    assert plan.status == "skipped"
    assert plan.reason == "already migrated to sparse refined surfaces"
    assert plan.current_sparse_surface is True


def test_plan_one_reports_conflict_when_legacy_groups_require_override(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "legacy.zarr"
    root = _build_root(legacy=True)

    monkeypatch.setattr(mod, "_open_root", lambda path, mode="r": root)  # noqa: ARG005
    monkeypatch.setattr(mod, "has_sparse_curated_refined_detect_instances_arrays", lambda run: "instances" in run)
    monkeypatch.setattr(mod, "has_curated_refined_source_detections_projection", lambda run: "source_detections" in run)

    def _raise_conflict(*_args, **_kwargs):
        raise single.SparseMigrationConflictError(
            "legacy groups present",
            legacy_sparse_groups=["manual_a"],
            output_refined_run_name="refined_detect_001_sparse",
        )

    monkeypatch.setattr(mod, "build_sparse_migration_plan", _raise_conflict)

    plan = mod._plan_one(
        zarr_path,
        zarr_use="analysis",
        refined_run=None,
        detect_run=None,
        quality_run=None,
        allow_missing_quality=False,
        ignore_legacy_groups=False,
        promote_latest_requested=True,
        force_promote_nonlatest=False,
        force=False,
    )

    assert plan.status == "conflict"
    assert plan.source_refined_run == "refined_detect_001"
    assert plan.output_refined_run == "refined_detect_001_sparse"
    assert plan.legacy_sparse_groups == ["manual_a"]


def test_plan_one_reports_conflict_for_nonlatest_default_promotion(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "nonlatest.zarr"
    root = _build_root(refined_latest="refined_detect_002")

    monkeypatch.setattr(mod, "_open_root", lambda path, mode="r": root)  # noqa: ARG005
    monkeypatch.setattr(mod, "has_sparse_curated_refined_detect_instances_arrays", lambda run: "instances" in run)
    monkeypatch.setattr(mod, "has_curated_refined_source_detections_projection", lambda run: "source_detections" in run)

    def _fake_build(*_args, **_kwargs):
        return single.SparseMigrationPlan(
            source_refined_run_name="refined_detect_001",
            output_refined_run_name="refined_detect_001_sparse",
            parent_latest_refined_run_name="refined_detect_002",
            source_is_parent_latest=False,
            source_detect_run="detect_001",
            source_quality_run="quality_001",
            total_frames=10,
            coverage_frame_source="full",
            coverage_frames_full=10,
            sampled_import=False,
            sampled_import_meta={},
            legacy_sparse_groups=[],
            legacy_group_policy="none_present",
            current_summary={},
            planned_summary={},
            coverage_comparison={},
            write_payload={},
        )

    monkeypatch.setattr(mod, "build_sparse_migration_plan", _fake_build)

    plan = mod._plan_one(
        zarr_path,
        zarr_use="analysis",
        refined_run="refined_detect_001",
        detect_run=None,
        quality_run=None,
        allow_missing_quality=False,
        ignore_legacy_groups=False,
        promote_latest_requested=True,
        force_promote_nonlatest=False,
        force=False,
    )

    assert plan.status == "conflict"
    assert plan.source_refined_run == "refined_detect_001"
    assert plan.parent_latest_refined_run == "refined_detect_002"
    assert plan.promotion_requested is True
    assert plan.promotion_allowed is False
    assert "--no-promote-latest" in str(plan.reason)


def test_main_dry_run_reports_ok_skipped_and_conflict_rows(monkeypatch, capsys, tmp_path: Path) -> None:
    ok_path = tmp_path / "ok.zarr"
    skip_path = tmp_path / "skip.zarr"
    conflict_path = tmp_path / "conflict.zarr"
    mapping = {
        ok_path: _build_root(sparse=False),
        skip_path: _build_root(sparse=True),
        conflict_path: _build_root(legacy=True),
    }
    monkeypatch.setattr(mod, "_iter_zarr", lambda roots, recursive=False: [ok_path, skip_path, conflict_path])  # noqa: ARG005
    monkeypatch.setattr(mod, "_open_root", lambda path, mode="r": mapping[path])  # noqa: ARG005
    monkeypatch.setattr(mod, "has_sparse_curated_refined_detect_instances_arrays", lambda run: "instances" in run)
    monkeypatch.setattr(mod, "has_curated_refined_source_detections_projection", lambda run: "source_detections" in run)

    def _fake_build(root, **_kwargs):  # noqa: ANN001
        if "manual_a" in root["refined_detect_runs"]["refined_detect_001"]:
            raise single.SparseMigrationConflictError(
                "legacy groups present",
                legacy_sparse_groups=["manual_a"],
                output_refined_run_name="refined_detect_001_sparse",
            )
        return single.SparseMigrationPlan(
            source_refined_run_name="refined_detect_001",
            output_refined_run_name="refined_detect_001_sparse",
            parent_latest_refined_run_name="refined_detect_001",
            source_is_parent_latest=True,
            source_detect_run="detect_001",
            source_quality_run="quality_001",
            total_frames=10,
            coverage_frame_source="full",
            coverage_frames_full=10,
            sampled_import=False,
            sampled_import_meta={},
            legacy_sparse_groups=[],
            legacy_group_policy="none_present",
            current_summary={},
            planned_summary={
                "source_refined_run": "refined_detect_001",
                "output_refined_run": "refined_detect_001_sparse",
                "total_instances": 9,
                "total_source_detections": 10,
                "multi_instance_frames": 0,
            },
            coverage_comparison={},
            write_payload={},
        )

    monkeypatch.setattr(mod, "build_sparse_migration_plan", _fake_build)

    rc = mod.main([str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Planned refined-detect sparse migration (dry-run):" in out
    assert "planned_instances: 9" in out
    assert "already migrated to sparse refined surfaces" in out
    assert "conflict: 1" in out
    assert "ok: 1" in out
    assert "skipped: 1" in out


def test_main_apply_writes_json_report(monkeypatch, capsys, tmp_path: Path) -> None:
    zarr_path = tmp_path / "apply.zarr"
    report_path = tmp_path / "reports" / "batch.json"
    root = _build_root(sparse=False)

    monkeypatch.setattr(mod, "_iter_zarr", lambda roots, recursive=False: [zarr_path])  # noqa: ARG005
    monkeypatch.setattr(mod, "_open_root", lambda path, mode="r": root)  # noqa: ARG005
    monkeypatch.setattr(mod, "has_sparse_curated_refined_detect_instances_arrays", lambda run: "instances" in run)
    monkeypatch.setattr(mod, "has_curated_refined_source_detections_projection", lambda run: "source_detections" in run)

    def _fake_build(root, **_kwargs):  # noqa: ANN001
        return single.SparseMigrationPlan(
            source_refined_run_name="refined_detect_001",
            output_refined_run_name="refined_detect_001_sparse",
            parent_latest_refined_run_name="refined_detect_001",
            source_is_parent_latest=True,
            source_detect_run="detect_001",
            source_quality_run="quality_001",
            total_frames=10,
            coverage_frame_source="full",
            coverage_frames_full=10,
            sampled_import=False,
            sampled_import_meta={},
            legacy_sparse_groups=[],
            legacy_group_policy="none_present",
            current_summary={},
            planned_summary={
                "source_refined_run": "refined_detect_001",
                "output_refined_run": "refined_detect_001_sparse",
                "total_instances": 9,
                "total_source_detections": 10,
                "multi_instance_frames": 0,
            },
            coverage_comparison={},
            write_payload={},
        )

    monkeypatch.setattr(mod, "build_sparse_migration_plan", _fake_build)
    monkeypatch.setattr(
        mod,
        "apply_sparse_migration",
        lambda root, zarr_path, plan, command=None, promote_latest=True, force_promote_nonlatest=False: {  # noqa: ARG005
            "applied": True,
            "promoted_to_latest": bool(promote_latest),
            "force_promote_nonlatest": bool(force_promote_nonlatest),
            "review_status": {"state": "pending"},
        },
    )

    rc = mod.main([str(tmp_path), "--apply", "--json-report", str(report_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Applying refined-detect sparse migration:" in out
    assert "applied: True" in out

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "apply"
    assert payload["summary"]["applied"] == 1
    assert payload["summary"]["failed_apply"] == 0
    assert len(payload["rows"]) == 1
    row = payload["rows"][0]
    assert row["zarr_path"] == str(zarr_path)
    assert row["applied"] is True
    assert row["source_refined_run"] == "refined_detect_001"
    assert row["output_refined_run"] == "refined_detect_001_sparse"
    assert row["apply_result"]["review_status"]["state"] == "pending"
