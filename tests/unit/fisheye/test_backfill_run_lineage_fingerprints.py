from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fisheye.utils import backfill_run_lineage_fingerprints as backfill


class FakeGroup:
    def __init__(
        self,
        attrs: dict[str, Any] | None = None,
        children: dict[str, "FakeGroup"] | None = None,
    ) -> None:
        self.attrs = attrs or {}
        self.children = children or {}

    def __contains__(self, key: str) -> bool:
        return key in self.children

    def __getitem__(self, key: str) -> "FakeGroup":
        return self.children[key]

    def group_keys(self) -> list[str]:
        return list(self.children)


def test_backfill_dry_run_does_not_write(monkeypatch, tmp_path: Path) -> None:
    run = FakeGroup(
        {
            "schema_id": "analysis.swim_bout_runs",
            "schema_version": 6,
            "detection_method": "peak_event",
            "method_version": "detect_bouts_multi_level.v7",
            "source_track_kinematics_run": "tk_1",
            "parameters": {"min_peak_prominence_mm_s": 4.0},
            "provenance": {
                "parameters": {"zarr_path": "/tmp/archive.zarr", "overwrite": True},
                "inputs": {"source_track_path": "analysis/track_kinematics_runs/offline/tk_1"},
                "git": {"commit": "abc123", "branch": "feature"},
            },
        }
    )
    root = FakeGroup(
        children={
            "analysis": FakeGroup(
                children={
                    "swim_bout_runs": FakeGroup(children={"bouts_1": run}),
                }
            )
        }
    )
    monkeypatch.setattr(backfill, "open_zarr_root", lambda path, mode="r": root)

    results = backfill.backfill_zarr_run_lineage_fingerprints(tmp_path / "a.zarr")

    assert [result.action for result in results] == ["would_write"]
    assert "source_fingerprint" not in run.attrs


def test_backfill_apply_writes_best_effort_payload(monkeypatch, tmp_path: Path) -> None:
    run = FakeGroup(
        {
            "schema_id": "analysis.swim_bout_runs",
            "schema_version": 6,
            "detection_method": "peak_event",
            "method_version": "detect_bouts_multi_level.v7",
            "source_track_kinematics_run": "tk_1",
            "parameters": {"min_peak_prominence_mm_s": 4.0},
            "provenance": {
                "parameters": {"zarr_path": "/tmp/archive.zarr", "overwrite": True},
                "inputs": {"source_track_path": "analysis/track_kinematics_runs/offline/tk_1"},
                "git": {"commit": "abc123", "branch": "feature"},
            },
        }
    )
    root = FakeGroup(
        children={
            "analysis": FakeGroup(
                children={
                    "swim_bout_runs": FakeGroup(children={"bouts_1": run}),
                }
            )
        }
    )
    monkeypatch.setattr(backfill, "open_zarr_root", lambda path, mode="r": root)

    results = backfill.backfill_zarr_run_lineage_fingerprints(
        tmp_path / "a.zarr",
        apply=True,
    )

    assert [result.action for result in results] == ["wrote"]
    assert run.attrs["fingerprint_status"] == "best_effort"
    assert len(run.attrs["source_fingerprint"]) == 64
    payload = json.loads(run.attrs["lineage_payload_json"])
    assert payload["parameters"]["min_peak_prominence_mm_s"] == 4.0
    assert "zarr_path" not in json.dumps(payload, sort_keys=True)
    assert "feature" not in json.dumps(payload, sort_keys=True)
    assert payload["code"]["git_commit"] == "abc123"


def test_backfill_skips_existing_without_overwrite(monkeypatch, tmp_path: Path) -> None:
    run = FakeGroup({"source_fingerprint": "existing"})
    root = FakeGroup(
        children={
            "analysis": FakeGroup(
                children={
                    "swim_bout_runs": FakeGroup(children={"bouts_1": run}),
                }
            )
        }
    )
    monkeypatch.setattr(backfill, "open_zarr_root", lambda path, mode="r": root)

    results = backfill.backfill_zarr_run_lineage_fingerprints(
        tmp_path / "a.zarr",
        apply=True,
    )

    assert [result.action for result in results] == ["skip_existing"]
    assert run.attrs == {"source_fingerprint": "existing"}
