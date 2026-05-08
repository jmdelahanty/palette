from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fisheye.utils import build_virtual_collection_manifest as builder
from fisheye.utils.virtual_collection_manifest import verify_manifest_sha256


class FakeGroup:
    def __init__(self, attrs: dict[str, Any] | None = None) -> None:
        self.attrs = attrs or {}


class FakeRoot(FakeGroup):
    def __init__(self) -> None:
        super().__init__(
            {
                "protocol_signature_hash": "protocol_sig",
                "protocol_semantic_hash": "protocol_sem",
            }
        )
        self.latest: dict[str, str] = {}
        self.runs: dict[tuple[str, str], FakeGroup] = {}


def _install_fake_zarr(monkeypatch, root: FakeRoot) -> None:
    def fake_open_zarr_root(path: Path, *, mode: str = "r") -> FakeRoot:
        assert mode == "r"
        return root

    def fake_resolve_zarr_run(
        fake_root: FakeRoot,
        parent_path,
        run_name: str | None,
        **kwargs,
    ) -> tuple[FakeGroup, str]:
        parent = "/".join(parent_path) if not isinstance(parent_path, str) else parent_path
        resolved = run_name or fake_root.latest.get(parent)
        if resolved is None:
            raise ValueError(f"{parent} has no latest")
        group = fake_root.runs.get((parent, resolved))
        if group is None:
            raise ValueError(f"{resolved} not found under {parent}")
        return group, resolved

    monkeypatch.setattr(builder, "open_zarr_root", fake_open_zarr_root)
    monkeypatch.setattr(builder, "resolve_zarr_run", fake_resolve_zarr_run)


def _fake_root() -> FakeRoot:
    root = FakeRoot()
    run_data = {
        ("analysis/track_kinematics_runs/offline", "tk_latest"): {
            "schema_id": "palette.analysis.track_kinematics",
            "schema_version": 1,
            "method": "hysteresis_latch",
            "method_version": "v1",
            "source_fingerprint": "track_fp",
            "lineage_hash": "track_lineage",
        },
        ("analysis/swim_bout_runs", "bouts_latest"): {
            "schema_id": "palette.analysis.swim_bouts",
            "schema_version": 1,
            "method": "peak_event",
            "method_version": "v1",
            "source_fingerprint": "bout_fp",
            "fingerprint_status": "best_effort",
        },
        ("analysis/bout_kinematics_runs", "bk_latest"): {
            "schema_id": "palette.analysis.bout_kinematics",
            "schema_version": 1,
            "method": "bout_kinematics",
            "method_version": "v1",
            "source_fingerprint": "bk_fp",
        },
    }
    for (parent, run_id), attrs in run_data.items():
        root.runs[(parent, run_id)] = FakeGroup(attrs)
        root.latest[parent] = run_id
    return root


def test_build_manifest_from_explicit_zarr_paths(monkeypatch, tmp_path: Path) -> None:
    root = _fake_root()
    _install_fake_zarr(monkeypatch, root)
    zarr_path = tmp_path / "2026-01-28T19-22-28Z_arena_1_DefaultScreen_analysis.zarr"
    zarr_path.mkdir()
    training_path = tmp_path / "2026-01-28T19-22-28Z_arena_1_DefaultScreen_training.zarr"
    training_path.mkdir()

    manifest = builder.build_manifest_from_zarrs(
        [zarr_path],
        collection_id="movement_test_v001",
        collection_name="Movement Test",
        created_utc="2026-05-07T12:00:00Z",
    )

    assert verify_manifest_sha256(manifest)
    assert manifest["query"]["registry_snapshot_status"] == "not_registry_derived"
    assert manifest["export_profiles"][0]["profile_id"] == "movement_bouts"
    record = manifest["records"][0]
    assert record["recording_id"] == "2026-01-28T19-22-28Z_arena_1_DefaultScreen"
    assert record["dataset_id"].startswith("analysis_")
    assert record["locator_at_selection"]["storage_tier"] == "hot_nvme"
    assert record["training_locator_at_selection"]["uri"] == str(training_path.resolve())
    assert record["status"]["included"] is True
    assert record["source_runs"]["track_kinematics_run"]["run_id"] == "tk_latest"
    assert record["source_runs"]["track_kinematics_run"]["selection"] == "resolved_latest"
    assert record["source_runs"]["track_kinematics_run"]["fingerprint_status"] == "complete"
    assert record["source_runs"]["swim_bout_run"]["fingerprint_status"] == "best_effort"
    assert record["source_runs"]["tail_kinematics_run"]["present"] is False
    assert record["source_runs"]["tail_kinematics_run"]["reason"] == "not_generated"


def test_build_manifest_excludes_records_missing_required_run(monkeypatch, tmp_path: Path) -> None:
    root = _fake_root()
    del root.runs[("analysis/bout_kinematics_runs", "bk_latest")]
    _install_fake_zarr(monkeypatch, root)
    zarr_path = tmp_path / "recording_analysis.zarr"
    zarr_path.mkdir()

    manifest = builder.build_manifest_from_zarrs(
        [zarr_path],
        collection_id="movement_test_v001",
        collection_name="Movement Test",
        created_utc="2026-05-07T12:00:00Z",
    )

    record = manifest["records"][0]
    assert record["status"]["included"] is False
    assert record["source_runs"]["bout_kinematics_run"]["present"] is False
    assert record["source_runs"]["bout_kinematics_run"]["required"] is True
    assert record["status"]["exclusions"]


def test_build_manifest_cli_writes_stamped_manifest(monkeypatch, tmp_path: Path, capsys) -> None:
    root = _fake_root()
    _install_fake_zarr(monkeypatch, root)
    zarr_path = tmp_path / "recording_analysis.zarr"
    zarr_path.mkdir()
    output = tmp_path / "manifests" / "movement.manifest.json"

    rc = builder.main(
        [
            "--collection-id",
            "movement_test_v001",
            "--collection-name",
            "Movement Test",
            "--output",
            str(output),
            str(zarr_path),
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["manifest_sha256"] == captured.out.strip()
    assert verify_manifest_sha256(payload)
