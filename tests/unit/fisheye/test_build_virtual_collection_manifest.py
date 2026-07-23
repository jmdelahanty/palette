from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
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

    def fake_load_chaser_distance_run(
        fake_root: FakeRoot,
        *,
        run_name: str = "latest",
    ) -> SimpleNamespace:
        parent = "analysis/chaser_distance_runs"
        resolved = (
            fake_root.latest.get(parent)
            if str(run_name).strip() in {"", "latest"}
            else str(run_name).strip()
        )
        if not resolved or (parent, resolved) not in fake_root.runs:
            raise builder.ChaserDistanceReadError(
                f"{resolved!r} not found under {parent}"
            )
        return SimpleNamespace(
            authority_status=builder.VERIFIED_AUTHORITY_STATUS,
            run_name=resolved,
            run_path=f"{parent}/{resolved}",
            publication_seal_ref=f"/{parent}/{resolved}@publication_seal",
            publication_seal_sha256=f"seal_{resolved}",
            surface_manifest_ref=f"/{parent}/{resolved}@surface_manifest",
            surface_manifest_sha256=f"manifest_{resolved}",
            row_identity_ref=f"/{parent}/{resolved}@row_identity",
            row_identity_sha256=f"rows_{resolved}",
        )

    monkeypatch.setattr(builder, "open_zarr_root", fake_open_zarr_root)
    monkeypatch.setattr(builder, "resolve_zarr_run", fake_resolve_zarr_run)
    monkeypatch.setattr(
        builder,
        "load_chaser_distance_run",
        fake_load_chaser_distance_run,
    )


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


def _fake_chaser_root() -> FakeRoot:
    root = FakeRoot()
    run_data = {
        ("analysis/detection_occupancy_runs", "occupancy_latest"): {
            "schema_id": "palette.detection_occupancy.v1",
            "schema_version": 1,
            "source_fingerprint": "occupancy_fp",
        },
        ("analysis/chaser_distance_runs", "chaser_latest"): {
            "schema_id": "palette.chaser_distance.v1",
            "schema_version": 1,
            "source_fingerprint": "chaser_fp",
        },
        ("analysis/track_kinematics_runs/offline", "track_latest"): {
            "schema_id": "palette.analysis.track_kinematics",
            "schema_version": 1,
            "source_fingerprint": "track_fp",
        },
    }
    for (parent, run_id), attrs in run_data.items():
        root.runs[(parent, run_id)] = FakeGroup(attrs)
        root.latest[parent] = run_id
    return root


def _registry_fixture(path: Path, rows: list[tuple[Any, ...]]) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE dataset_context_current (
                dataset_id TEXT,
                recording_id TEXT,
                zarr_path TEXT,
                protocol_name TEXT,
                zarr_use TEXT,
                dataset_status TEXT
            );
            CREATE TABLE recording_stimulus_mode_counts (
                dataset_id TEXT,
                recording_id TEXT,
                stimulus_run_id TEXT,
                protocol_hash TEXT,
                protocol_name TEXT,
                is_latest INTEGER,
                stimulus_mode TEXT,
                step_count INTEGER,
                total_duration_s REAL
            );
            """
        )
        for row in rows:
            conn.execute(
                """
                INSERT INTO dataset_context_current VALUES (?, ?, ?, ?, ?, ?)
                """,
                row[:6],
            )
            conn.execute(
                """
                INSERT INTO recording_stimulus_mode_counts
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (row[0], row[1], *row[6:]),
            )


def test_chaser_manifest_selection_uses_only_detached_verified_authority(
    monkeypatch,
) -> None:
    snapshot = SimpleNamespace(
        authority_status=builder.VERIFIED_AUTHORITY_STATUS,
        run_name="sealed_run",
        run_path="analysis/chaser_distance_runs/sealed_run",
        publication_seal_ref=(
            "/analysis/chaser_distance_runs/sealed_run@chaser_distance_publication_seal"
        ),
        publication_seal_sha256="a" * 64,
        surface_manifest_ref=(
            "/analysis/chaser_distance_runs/sealed_run@chaser_distance_surface_manifest"
        ),
        surface_manifest_sha256="b" * 64,
        row_identity_ref=(
            "/analysis/chaser_distance_runs/sealed_run@row_identity_contract"
        ),
        row_identity_sha256="c" * 64,
    )

    def reject_generic_selector(*_args, **_kwargs):
        raise AssertionError("generic selector/raw child access is forbidden")

    def strict_load(root, *, run_name: str):
        assert root is raw_navigation_trap
        assert run_name == "latest"
        return snapshot

    raw_navigation_trap = object()
    monkeypatch.setattr(builder, "resolve_zarr_run", reject_generic_selector)
    monkeypatch.setattr(builder, "load_chaser_distance_run", strict_load)

    entry, warning = builder._resolve_run_entry(  # noqa: SLF001
        raw_navigation_trap,
        parent_path=("analysis", "chaser_distance_runs"),
        run_name=None,
        run_path_prefix="analysis/chaser_distance_runs",
        required=True,
        run_label="Chaser-distance run",
    )

    assert warning is None
    assert entry["path"] == snapshot.run_path
    assert entry["source_fingerprint"] == snapshot.publication_seal_sha256
    assert entry["lineage_hash"] == snapshot.surface_manifest_sha256
    assert entry["authority_status"] == builder.VERIFIED_AUTHORITY_STATUS


def test_chaser_manifest_selection_fails_closed_on_incompatible_publication(
    monkeypatch,
) -> None:
    def reject_generic_selector(*_args, **_kwargs):
        raise AssertionError("generic selector fallback is forbidden")

    def reject_publication(_root, *, run_name: str):
        assert run_name == "explicit_run"
        raise builder.ChaserDistanceReadError("publication seal is stale")

    monkeypatch.setattr(builder, "resolve_zarr_run", reject_generic_selector)
    monkeypatch.setattr(builder, "load_chaser_distance_run", reject_publication)

    entry, warning = builder._resolve_run_entry(  # noqa: SLF001
        object(),
        parent_path="analysis/chaser_distance_runs",
        run_name="explicit_run",
        run_path_prefix="analysis/chaser_distance_runs",
        required=True,
        run_label="Chaser-distance run",
    )

    assert entry["present"] is False
    assert entry["reason"] == "required_run_incompatible"
    assert entry["fingerprint_status"] == "not_applicable"
    assert "canonical preflight failed closed" in str(warning)


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


def test_build_manifest_goodcopbadcop_profile_resolves_required_runs(monkeypatch, tmp_path: Path) -> None:
    root = FakeRoot()
    run_data = {
        ("analysis/detection_occupancy_runs", "occupancy_latest"): {
            "schema_id": "palette.detection_occupancy.v1",
            "schema_version": 1,
            "method": "detection_centroid_epoch_occupancy",
            "method_version": "1",
            "source_fingerprint": "occupancy_fp",
        },
        ("analysis/chaser_distance_runs", "chaser_latest"): {
            "schema_id": "palette.chaser_distance.v1",
            "schema_version": 1,
            "method": "offline_detection_to_chaser_distance",
            "method_version": "1",
            "source_fingerprint": "chaser_fp",
        },
    }
    for (parent, run_id), attrs in run_data.items():
        root.runs[(parent, run_id)] = FakeGroup(attrs)
        root.latest[parent] = run_id
    _install_fake_zarr(monkeypatch, root)
    zarr_path = tmp_path / "recording_goodcopbadcop_analysis.zarr"
    zarr_path.mkdir()

    manifest = builder.build_manifest_from_zarrs(
        [zarr_path],
        collection_id="goodcopbadcop_test_v001",
        collection_name="GoodCopBadCop Test",
        profile_id=builder.PROFILE_GOODCOPBADCOP_CHASER,
        created_utc="2026-06-21T12:00:00Z",
    )

    assert verify_manifest_sha256(manifest)
    assert manifest["export_profiles"][0]["profile_id"] == "goodcopbadcop_chaser"
    record = manifest["records"][0]
    assert record["status"]["included"] is True
    assert record["source_runs"]["detection_occupancy_run"]["required"] is True
    assert record["source_runs"]["detection_occupancy_run"]["run_id"] == "occupancy_latest"
    assert record["source_runs"]["chaser_distance_run"]["required"] is True
    assert record["source_runs"]["chaser_distance_run"]["run_id"] == "chaser_latest"
    assert "swim_bout_run" not in record["source_runs"]


def test_select_registry_stimulus_datasets_combines_protocols_by_mode(tmp_path: Path) -> None:
    registry = tmp_path / "palette_registry.sqlite"
    _registry_fixture(
        registry,
        [
            (
                "dataset_red",
                "recording_red",
                str(tmp_path / "red_analysis.zarr"),
                "RedScare",
                "analysis",
                "active",
                "stim_red",
                "hash_red",
                "RedScare",
                1,
                "CHASER",
                2,
                60.0,
            ),
            (
                "dataset_goodcop",
                "recording_goodcop",
                str(tmp_path / "goodcop_analysis.zarr"),
                "GoodCopBadCop",
                "analysis",
                "active",
                "stim_goodcop",
                "hash_goodcop",
                "GoodCopBadCop",
                1,
                "chaser",
                3,
                90.0,
            ),
            (
                "dataset_grating",
                "recording_grating",
                str(tmp_path / "grating_analysis.zarr"),
                "Optomotor",
                "analysis",
                "active",
                "stim_grating",
                "hash_grating",
                "Optomotor",
                1,
                "MOVING_GRATING",
                2,
                60.0,
            ),
        ],
    )

    selected = builder.select_registry_stimulus_datasets(
        registry,
        stimulus_mode="chaser",
    )

    assert [row.dataset_id for row in selected] == ["dataset_goodcop", "dataset_red"]
    assert {row.protocol_name for row in selected} == {"GoodCopBadCop", "RedScare"}
    assert {row.stimulus_mode for row in selected} == {"CHASER"}


def test_build_manifest_from_registry_uses_protocol_neutral_chaser_profile(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = _fake_chaser_root()
    _install_fake_zarr(monkeypatch, root)
    red_path = tmp_path / "red_analysis.zarr"
    goodcop_path = tmp_path / "goodcop_analysis.zarr"
    red_path.mkdir()
    goodcop_path.mkdir()
    registry = tmp_path / "palette_registry.sqlite"
    _registry_fixture(
        registry,
        [
            (
                "dataset_red",
                "recording_red",
                str(red_path),
                "RedScare",
                "analysis",
                "active",
                "stim_red",
                "hash_red",
                "RedScare",
                1,
                "CHASER",
                2,
                60.0,
            ),
            (
                "dataset_goodcop",
                "recording_goodcop",
                str(goodcop_path),
                "GoodCopBadCop",
                "analysis",
                "active",
                "stim_goodcop",
                "hash_goodcop",
                "GoodCopBadCop",
                1,
                "CHASER",
                3,
                90.0,
            ),
        ],
    )

    manifest = builder.build_manifest_from_registry(
        registry,
        stimulus_mode="CHASER",
        collection_id="all_chaser_v001",
        collection_name="All Chaser Recordings",
        created_utc="2026-07-12T12:00:00Z",
    )

    assert verify_manifest_sha256(manifest)
    assert manifest["export_profiles"][0]["profile_id"] == "chaser"
    assert manifest["query"]["filters"]["normalized_stimulus_mode"] == "CHASER"
    assert manifest["query"]["filters"]["protocol_name"] is None
    assert {row["dataset_id"] for row in manifest["records"]} == {
        "dataset_red",
        "dataset_goodcop",
    }
    assert {row["protocol"]["protocol_name"] for row in manifest["records"]} == {
        "RedScare",
        "GoodCopBadCop",
    }
    assert all(
        row["source_runs"]["track_kinematics_run"]["required"]
        for row in manifest["records"]
    )


def test_registry_cli_defaults_to_chaser_profile_and_shared_storage(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = _fake_chaser_root()
    _install_fake_zarr(monkeypatch, root)
    zarr_path = tmp_path / "red_analysis.zarr"
    zarr_path.mkdir()
    registry = tmp_path / "palette_registry.sqlite"
    _registry_fixture(
        registry,
        [
            (
                "dataset_red",
                "recording_red",
                str(zarr_path),
                "RedScare",
                "analysis",
                "active",
                "stim_red",
                "hash_red",
                "RedScare",
                1,
                "CHASER",
                2,
                60.0,
            ),
        ],
    )
    output = tmp_path / "all_chaser.manifest.json"

    rc = builder.main(
        [
            "--registry",
            str(registry),
            "--stimulus-mode",
            "CHASER",
            "--collection-id",
            "all_chaser_v001",
            "--collection-name",
            "All Chaser",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["export_profiles"][0]["profile_id"] == "chaser"
    assert manifest["records"][0]["locator_at_selection"]["storage_tier"] == "shared_groups"


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
