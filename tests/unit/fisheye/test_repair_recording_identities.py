from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from fisheye.registry.repair_recording_identities import (
    apply_plans,
    build_plans,
    validate_plans,
)


def _write_metadata(recording_dir: Path, *, alias: str, canonical: str) -> None:
    (recording_dir / "zarr" / f"{canonical}_analysis.zarr").mkdir(parents=True)
    (recording_dir / "zarr" / f"{canonical}_training.zarr").mkdir(parents=True)
    (recording_dir / "recording_manifest.json").write_text(
        json.dumps(
            {
                "recording_id": alias,
                "session_uuid": alias,
                "recording_name": canonical,
                "historical": {"recording_id": alias},
            }
        ),
        encoding="utf-8",
    )
    for use, recording_id in (("analysis", canonical), ("training", alias)):
        path = recording_dir / "zarr" / f"{canonical}_{use}.zarr" / "zarr.json"
        path.write_text(
            json.dumps(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {
                        "recording_id": recording_id,
                        "session_uuid": alias,
                        "historical": {"recording_id": alias},
                    },
                }
            ),
            encoding="utf-8",
        )


def _registry(path: Path, *, alias: str, canonical: str, recording_dir: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(
        """
        CREATE TABLE recordings (
            recording_id TEXT PRIMARY KEY,
            session_uuid TEXT,
            recording_name TEXT,
            recording_path TEXT
        );
        CREATE TABLE datasets (
            dataset_id TEXT PRIMARY KEY,
            recording_id TEXT,
            zarr_path TEXT
        );
        CREATE TABLE recording_step_status (
            dataset_id TEXT,
            recording_id TEXT,
            step_name TEXT,
            PRIMARY KEY (dataset_id, step_name)
        );
        CREATE TABLE recording_artifacts (
            artifact_id TEXT PRIMARY KEY,
            recording_id TEXT NOT NULL REFERENCES recordings(recording_id) ON DELETE CASCADE
        );
        """
    )
    conn.executemany(
        "INSERT INTO recordings(recording_id,session_uuid,recording_name,recording_path) VALUES(?,?,?,?);",
        (
            (canonical, alias, canonical, str(recording_dir)),
            (alias, alias, canonical, str(recording_dir)),
        ),
    )
    analysis = recording_dir / "zarr" / f"{canonical}_analysis.zarr"
    training = recording_dir / "zarr" / f"{canonical}_training.zarr"
    conn.executemany(
        "INSERT INTO datasets(dataset_id,recording_id,zarr_path) VALUES(?,?,?);",
        (("analysis", canonical, str(analysis)), ("training", alias, str(training))),
    )
    conn.execute(
        "INSERT INTO recording_step_status(dataset_id,recording_id,step_name) VALUES('training',?,'keypoints');",
        (alias,),
    )
    conn.execute(
        "INSERT INTO recording_artifacts(artifact_id,recording_id) VALUES('artifact',?);",
        (alias,),
    )
    conn.commit()
    conn.row_factory = sqlite3.Row
    return conn


def test_recording_identity_repair_consolidates_alias_and_preserves_session_uuid(tmp_path: Path) -> None:
    canonical = "2026-06-23T16-01-09Z_arena_2_RedScare"
    alias = "2026-06-23T16-01-09Z_arena_2"
    recording_dir = tmp_path / "recordings" / canonical
    _write_metadata(recording_dir, alias=alias, canonical=canonical)
    conn = _registry(tmp_path / "registry.sqlite", alias=alias, canonical=canonical, recording_dir=recording_dir)
    backup = tmp_path / "registry.backup.sqlite"
    try:
        plans = build_plans(conn, recording_dirs=[recording_dir])
        assert len(plans) == 1
        assert plans[0].aliases == (alias,)
        assert plans[0].registry_counts == {
            "datasets": 1,
            "recording_artifacts": 1,
            "recording_step_status": 1,
            "recordings": 1,
        }

        result = apply_plans(conn, plans=plans, backup_path=backup)
        validation = validate_plans(conn, plans=plans)

        assert result["status"] == "complete"
        assert validation == {"status": "ok", "issues": []}
        assert backup.is_file()
        assert conn.execute(
            "SELECT COUNT(*) FROM recordings WHERE recording_id = ?", (alias,)
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT recording_id FROM datasets WHERE dataset_id='training'"
        ).fetchone()[0] == canonical
        assert conn.execute(
            "SELECT recording_id FROM recording_artifacts WHERE artifact_id='artifact'"
        ).fetchone()[0] == canonical
    finally:
        conn.close()

    manifest = json.loads((recording_dir / "recording_manifest.json").read_text())
    training = json.loads(
        (
            recording_dir
            / "zarr"
            / f"{canonical}_training.zarr"
            / "zarr.json"
        ).read_text()
    )
    assert manifest["recording_id"] == canonical
    assert manifest["session_uuid"] == alias
    assert manifest["historical"]["recording_id"] == alias
    assert training["attributes"]["recording_id"] == canonical
    assert training["attributes"]["session_uuid"] == alias
    assert training["attributes"]["historical"]["recording_id"] == alias


def test_recording_identity_repair_dry_plan_does_not_modify_files(tmp_path: Path) -> None:
    canonical = "recording_RedScare"
    alias = "recording"
    recording_dir = tmp_path / "recordings" / canonical
    _write_metadata(recording_dir, alias=alias, canonical=canonical)
    manifest_path = recording_dir / "recording_manifest.json"
    before = manifest_path.read_bytes()
    conn = _registry(tmp_path / "registry.sqlite", alias=alias, canonical=canonical, recording_dir=recording_dir)
    try:
        plans = build_plans(conn, recording_dirs=[recording_dir])
        assert len(plans[0].metadata_edits) == 2
        assert manifest_path.read_bytes() == before
        assert conn.execute(
            "SELECT recording_id FROM datasets WHERE dataset_id='training'"
        ).fetchone()[0] == alias
    finally:
        conn.close()
