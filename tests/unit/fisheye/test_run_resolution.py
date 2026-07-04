from __future__ import annotations

import sqlite3
from pathlib import Path

from fisheye.shared.run_resolution import RunResolution, resolve_run
from fisheye.shared.stage_run_groups import stage_run_parent_paths
from fisheye.shared.zarr_run_completion import mark_run_complete, set_authoritative_run


class FakeGroup(dict[str, object]):
    def __init__(self, *, attrs: dict[str, object] | None = None) -> None:
        super().__init__()
        self.attrs = attrs if attrs is not None else {}

    def group_keys(self):
        return [key for key, value in self.items() if isinstance(value, FakeGroup)]

    def require_group(self, name: str) -> "FakeGroup":
        group: FakeGroup = self
        for part in [piece for piece in str(name).split("/") if piece]:
            child = group.get(part)
            if child is None:
                child = FakeGroup()
                group[part] = child
            if not isinstance(child, FakeGroup):
                raise TypeError(f"{part!r} exists and is not a group")
            group = child
        return group


def _complete(parent: FakeGroup, name: str) -> FakeGroup:
    run = parent.require_group(name)
    mark_run_complete(run, parent_group=parent, run_name=name)
    return run


def test_authoritative_resolution_prefers_pointer_over_later_complete() -> None:
    parent = FakeGroup()
    _complete(parent, "reviewed")
    set_authoritative_run(parent, "reviewed", approved_by="jeremy")
    _complete(parent, "zzz_smoke")

    resolved = resolve_run(parent, RunResolution.AUTHORITATIVE, parent_path="detect_runs")

    assert resolved.run_name == "reviewed"
    assert resolved.resolution_source == "authoritative"
    assert resolved.fallback_used is False
    assert resolved.parent_path == "detect_runs"
    assert resolved.run_group is parent["reviewed"]


def test_authoritative_resolution_falls_back_to_latest_complete_when_unset() -> None:
    parent = FakeGroup()
    _complete(parent, "run_001")

    resolved = resolve_run(parent, RunResolution.AUTHORITATIVE, parent_path="detect_runs")

    assert resolved.run_name == "run_001"
    assert resolved.resolution_source == "latest_complete"
    assert resolved.fallback_used is True


def test_latest_complete_resolution_ignores_authoritative_pointer() -> None:
    parent = FakeGroup()
    _complete(parent, "reviewed")
    set_authoritative_run(parent, "reviewed", approved_by="jeremy")
    _complete(parent, "zzz_smoke")

    resolved = resolve_run(parent, RunResolution.LATEST_COMPLETE, parent_path="detect_runs")

    assert resolved.run_name == "zzz_smoke"
    assert resolved.resolution_source == "latest_complete"
    assert resolved.run_group is parent["zzz_smoke"]


def test_source_match_resolves_named_lineage_without_latest_fallback() -> None:
    parent = FakeGroup()
    _complete(parent, "detect_used")
    _complete(parent, "detect_later")
    source = FakeGroup(attrs={"source_detect_run": "detect_used"})

    resolved = resolve_run(
        parent,
        RunResolution.SOURCE_MATCH,
        parent_path="detect_runs",
        source_group=source,
        source_attr="source_detect_run",
    )

    assert resolved.run_name == "detect_used"
    assert resolved.resolution_source == "source_match"
    assert resolved.source_attr == "source_detect_run"


def test_source_match_can_resolve_from_root_and_parent_path() -> None:
    root = FakeGroup()
    parent = root.require_group("detect_runs")
    _complete(parent, "detect_used")
    source = FakeGroup(attrs={"source_detect_run": "detect_used"})

    resolved = resolve_run(
        None,
        RunResolution.SOURCE_MATCH,
        root=root,
        parent_path="detect_runs",
        source_group=source,
        source_attr="source_detect_run",
    )

    assert resolved.run_name == "detect_used"
    assert resolved.run_group is parent["detect_used"]
    assert resolved.parent_path == "detect_runs"


def test_inventory_latest_reads_registry_view(tmp_path: Path) -> None:
    registry = tmp_path / "palette_registry.sqlite"
    conn = sqlite3.connect(registry)
    conn.execute(
        """
        CREATE TABLE recording_step_status_latest (
            dataset_id TEXT,
            recording_id TEXT,
            step_name TEXT,
            run_name TEXT,
            updated_utc TEXT
        );
        """
    )
    conn.execute(
        """
        INSERT INTO recording_step_status_latest
            (dataset_id, recording_id, step_name, run_name, updated_utc)
        VALUES
            ('dataset_001', 'rec_001', 'detect', 'detect_inventory', '2026-07-04T00:00:00+00:00');
        """
    )
    conn.commit()
    conn.close()

    resolved = resolve_run(
        None,
        RunResolution.INVENTORY_LATEST,
        registry=registry,
        dataset_id="dataset_001",
        stage="detect",
    )

    assert resolved.run_name == "detect_inventory"
    assert resolved.resolution_source == "inventory_latest"
    assert resolved.dataset_id == "dataset_001"
    assert resolved.stage == "detect"


def test_stage_run_group_map_includes_live_background_bridge() -> None:
    assert stage_run_parent_paths("background") == ("background_runs", "background")
    assert stage_run_parent_paths("refined_detect") == ("refined_detect_runs", "refined_runs")
