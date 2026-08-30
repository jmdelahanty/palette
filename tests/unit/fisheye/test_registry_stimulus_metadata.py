from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest
import zarr

from fisheye.analysis.import_stimulus_to_zarr import (
    _bind_protocol_semantic_steps,
    _materialize_protocol_execution_index,
    _materialize_protocol_semantic_snapshot,
)
from fisheye.registry.db import Registry
from fisheye.registry.extractors.stimulus_metadata import extract_stimulus_metadata
from fisheye.shared.protocol_semantic_contract import (
    ProtocolSemanticContractError,
    validate_protocol_semantic_snapshot,
)
from fisheye.shared.protocol_execution_contract import (
    CHASER_PHASE_NAMES,
    validate_protocol_execution_index,
)


def _write_archive(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "mixed_analysis.zarr"
    root = zarr.open_group(zarr_path, mode="w")
    root.attrs.update(
        {
            "dataset_id": "dataset_mixed",
            "recording_id": "mixed",
            "session_uuid": "mixed",
            "zarr_purpose": "analysis",
            "zarr_use": "analysis",
            "recording_name": "mixed",
            "recording_type": "behavior",
        }
    )
    parent = root.require_group("analysis/stimulus_runs")
    run = parent.create_group("stimulus_001")
    run.attrs["protocol_json"] = json.dumps(
        {
            "protocol_name": "Mixed canary",
            "steps": [
                {
                    "name": "grating one",
                    "duration_s": 5.0,
                    "parameters": {"stimulus_mode": "MOVING_GRATING"},
                },
                {
                    "name": "grating two",
                    "duration_s": 7.0,
                    "parameters": {"stimulus_mode": "MOVING_GRATING"},
                },
                {
                    "name": "loom",
                    "duration_s": 3.0,
                    "parameters": {"stimulus_mode": "LOOMING_DOT"},
                },
            ],
        }
    )
    steps = run.create_group("steps")
    for index, (mode, duration) in enumerate(
        (("MOVING_GRATING", 5.0), ("MOVING_GRATING", 7.0), ("LOOMING_DOT", 3.0))
    ):
        step = steps.create_group(f"step_{index}")
        step.attrs.update(
            {
                "step_index": index,
                "step_name": f"step {index}",
                "stimulus_mode": mode,
                "start_camera_frame": index * 100,
                "end_camera_frame": (index + 1) * 100,
                "duration_s": duration,
            }
        )
    parent.attrs["latest"] = "stimulus_001"
    return zarr_path


def _write_semantic_archive(
    tmp_path: Path,
    *,
    snapshot_v2: bool = False,
) -> tuple[Path, str]:
    zarr_path = tmp_path / "semantic_analysis.zarr"
    root = zarr.open_group(zarr_path, mode="w")
    root.attrs.update(
        {
            "dataset_id": "dataset_semantic",
            "recording_id": "semantic",
            "session_uuid": "semantic",
            "zarr_purpose": "analysis",
            "zarr_use": "analysis",
            "recording_name": "semantic",
            "recording_type": "behavior",
        }
    )
    parent = root.require_group("analysis/stimulus_runs")
    run = parent.create_group("stimulus_001")
    run.attrs["protocol_json"] = json.dumps(
        {
            "protocol_name": "GoodBatBadBat semantic canary",
            "steps": [
                {
                    "name": "standalone baseline",
                    "duration_s": 300.0,
                    "stimulus_mode_str": "SOLID_BLACK",
                },
                {
                    "name": "chaser",
                    "duration_s": 1500.0,
                    "stimulus_mode_str": "CHASER",
                },
            ],
        }
    )
    semantic = {
        "identity": {
            "iti_stimulus_mode_id": 99,
            "steps": [
                {
                    "duration": {"scale": "1e-3", "unit": "s", "value": 300000},
                    "parameters": {"color_type_id": 0},
                    "post_stimulus_iti": {
                        "scale": "1e-3",
                        "unit": "s",
                        "value": 0,
                    },
                    "stimulus_mode_id": 4,
                },
                {
                    "duration": {"scale": "1e-3", "unit": "s", "value": 1500000},
                    "parameters": {},
                    "post_stimulus_iti": {
                        "scale": "1e-3",
                        "unit": "s",
                        "value": 0,
                    },
                    "stimulus_mode_id": 12,
                },
            ],
        },
        "normalization_policy": "citrus.protocol.semantic.v1",
        "schema_id": "citrus.protocol.semantic",
        "schema_version": 1,
    }
    semantic_json = json.dumps(semantic, sort_keys=True, separators=(",", ":"))
    semantic_hash = "sha256:" + sha256(semantic_json.encode("utf-8")).hexdigest()
    trial_payload = {
            "normalization_policy": (
                "citrus.protocol.trial_index.v2"
                if snapshot_v2
                else "citrus.protocol.trial_index.v1"
            ),
            "protocol_semantic_hash": semantic_hash,
            "schema_id": "citrus.protocol.trial_index",
            "schema_version": 2 if snapshot_v2 else 1,
            "steps": [
                {
                    "duration_s": 300.0,
                    "features": {
                        "color_name": "black",
                        "resolved_color": {
                            "color_space": "srgb",
                            "rgba8": [0, 0, 0, 255],
                        },
                    },
                    "index_status": "detailed",
                    "post_stimulus_iti_s": 0.0,
                    "step_index": 0,
                    "stimulus_family": "solid_color",
                    "stimulus_mode": "SOLID_BLACK",
                    "stimulus_mode_id": 4,
                },
                {
                    "duration_s": 1500.0,
                    "features": {},
                    "index_status": "detailed",
                    "post_stimulus_iti_s": 0.0,
                    "step_index": 1,
                    "stimulus_family": "chaser",
                    "stimulus_mode": "CHASER",
                    "stimulus_mode_id": 12,
                },
            ],
        }
    trial_json = json.dumps(
        trial_payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    trial_hash = "sha256:" + sha256(trial_json.encode("utf-8")).hexdigest()
    snapshot = validate_protocol_semantic_snapshot(
        semantic_hash=semantic_hash,
        semantic_json=semantic_json,
        trial_index_json=trial_json,
        trial_index_hash=trial_hash if snapshot_v2 else None,
        snapshot_schema_version=2 if snapshot_v2 else 1,
        snapshot_policy_id=(
            "citrus.protocol.snapshot.v2"
            if snapshot_v2
            else "citrus.protocol.snapshot.legacy_v1"
        ),
    )
    execution = None
    if snapshot_v2:
        execution_steps = []
        for identity, bounds in zip(snapshot.steps, ((10, 20), (20, 30))):
            start, end = bounds
            interval = {
                "start_stimulus_frame_inclusive": start,
                "end_stimulus_frame_exclusive": end,
                "first_camera_frame_id": 1000 + start,
                "last_camera_frame_id": 1000 + end - 1,
            }
            record = {
                "step_index": identity.step_index,
                "stimulus_mode_id": identity.stimulus_mode_id,
                "completion_status": "completed",
                "end_reason": "completed",
                "interval": interval,
            }
            if identity.stimulus_mode == "CHASER":
                record["chaser_phases"] = {
                    "chaser_pre": {
                        **interval,
                        "end_stimulus_frame_exclusive": 22,
                        "last_camera_frame_id": 1022,
                    },
                    "chaser_training": {
                        **interval,
                        "start_stimulus_frame_inclusive": 22,
                        "end_stimulus_frame_exclusive": 28,
                        "first_camera_frame_id": 1022,
                        "last_camera_frame_id": 1028,
                    },
                    "chaser_post": {
                        **interval,
                        "start_stimulus_frame_inclusive": 28,
                        "first_camera_frame_id": 1028,
                    },
                }
            execution_steps.append(record)
        execution_payload = {
            "authoritative_interval_axis": "stimulus_frame_num",
            "camera_frame_role": "correspondence_only",
            "chaser_repositioning_ownership": (
                "before_chaser_post_start_belongs_to_training;"
                "at_or_after_belongs_to_post"
            ),
            "policy_id": (
                "citrus.protocol.execution_index."
                "half_open_stimulus_frames.v1"
            ),
            "protocol_trial_index_hash": snapshot.trial_index_sha256,
            "schema_id": "citrus.protocol.execution_index",
            "schema_version": 1,
            "status": "complete",
            "steps": execution_steps,
        }
        execution_json = json.dumps(
            execution_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        execution = validate_protocol_execution_index(
            execution_json=execution_json,
            execution_hash=(
                "sha256:" + sha256(execution_json.encode("utf-8")).hexdigest()
            ),
            snapshot=snapshot,
        )
    steps = run.create_group("steps")
    for identity, bounds in zip(snapshot.steps, ((862, 30863), (30864, 180864))):
        step = steps.create_group(f"step_{identity.step_index}")
        step.attrs.update(
            {
                "step_index": identity.step_index,
                "step_name": f"step {identity.step_index}",
                "stimulus_mode_id": identity.stimulus_mode_id,
                "stimulus_mode": identity.stimulus_mode,
                "start_camera_frame": bounds[0],
                "end_camera_frame": bounds[1],
                "duration_s": identity.duration_s,
            }
        )
        if execution is not None:
            realized = execution.steps[identity.step_index]
            step.attrs.update(
                {
                    "authoritative_interval_axis": "stimulus_frame_num",
                    "start_stimulus_frame_inclusive": (
                        realized.interval.start_stimulus_frame_inclusive
                    ),
                    "end_stimulus_frame_exclusive": (
                        realized.interval.end_stimulus_frame_exclusive
                    ),
                    "first_camera_frame_id_correspondence": (
                        realized.interval.first_camera_frame_id
                    ),
                    "last_camera_frame_id_correspondence": (
                        realized.interval.last_camera_frame_id
                    ),
                    "camera_frame_role": "correspondence_only",
                    "execution_completion_status": realized.completion_status,
                    "execution_end_reason": realized.end_reason,
                }
            )
            step.attrs.pop("start_camera_frame")
            step.attrs.pop("end_camera_frame")
            if realized.chaser_phases is not None:
                phases = step.create_group("execution_phases")
                for phase_name in CHASER_PHASE_NAMES:
                    phase = phases.create_group(phase_name)
                    phase.attrs.update(
                        {
                            **realized.chaser_phases[phase_name].to_record(),
                            "authoritative_interval_axis": "stimulus_frame_num",
                            "camera_frame_role": "correspondence_only",
                            "acquisition_containment_status": (
                                "unavailable_without_sealed_"
                                "stimulus_to_acquisition_mapping"
                            ),
                        }
                    )
    _bind_protocol_semantic_steps(run, snapshot)
    if execution is not None:
        steps = run["steps"]
        steps.attrs.update(
            {
                "source": "citrus_protocol_execution_index",
                "authoritative_interval_axis": "stimulus_frame_num",
                "interval_policy_id": (
                    "citrus.protocol.execution_index."
                    "half_open_stimulus_frames.v1"
                ),
                "camera_frame_role": "correspondence_only",
                "protocol_execution_hash": execution.execution_hash,
                "protocol_execution_status": execution.status,
                "acquisition_containment_status": (
                    "unavailable_without_sealed_"
                    "stimulus_to_acquisition_mapping"
                ),
            }
        )
    _materialize_protocol_semantic_snapshot(run, snapshot)
    if execution is not None:
        correspondence = np.asarray(
            [(frame, 1000 + frame // 2) for frame in range(10, 30)],
            dtype=np.dtype(
                [
                    ("stimulus_frame_num", np.uint64),
                    ("triggering_camera_frame_id", np.uint64),
                ]
            ),
        )
        _materialize_protocol_execution_index(
            run,
            execution,
            frame_metadata=correspondence,
        )
    parent.attrs["latest"] = "stimulus_001"
    return zarr_path, semantic_hash


def test_extract_stimulus_metadata_normalizes_steps_and_mode_counts(tmp_path: Path) -> None:
    zarr_path = _write_archive(tmp_path)
    root = zarr.open_group(zarr_path, mode="r", use_consolidated=False)

    result = extract_stimulus_metadata(
        root,
        zarr_path=zarr_path,
        recording_id="mixed",
    )

    assert len(result.protocols) == 1
    assert result.protocols[0]["protocol_name"] == "Mixed canary"
    assert len(result.protocol_steps) == 3
    assert len(result.recording_steps) == 3
    assert [
        (row["stimulus_mode"], row["step_count"], row["total_duration_s"])
        for row in result.recording_modes
    ] == [("LOOMING_DOT", 1, 3.0), ("MOVING_GRATING", 2, 12.0)]


def test_registry_scan_exposes_protocol_steps_and_recording_mode_counts(
    tmp_path: Path,
) -> None:
    zarr_path = _write_archive(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        dataset_id = registry.scan_zarr(zarr_path)
        protocol = registry.conn.execute(
            "SELECT protocol_name, step_count FROM stimulus_protocols"
        ).fetchone()
        steps = registry.conn.execute(
            """
            SELECT step_index, stimulus_mode
            FROM recording_stimulus_steps
            WHERE dataset_id = ?
            ORDER BY step_index
            """,
            (dataset_id,),
        ).fetchall()
        modes = registry.conn.execute(
            """
            SELECT stimulus_mode, step_count, total_duration_s
            FROM recording_stimulus_mode_counts
            WHERE dataset_id = ? AND is_latest = 1
            ORDER BY stimulus_mode
            """,
            (dataset_id,),
        ).fetchall()
    finally:
        registry.close()

    assert protocol is not None
    assert tuple(protocol) == ("Mixed canary", 3)
    assert [tuple(row) for row in steps] == [
        (0, "MOVING_GRATING"),
        (1, "MOVING_GRATING"),
        (2, "LOOMING_DOT"),
    ]
    assert [tuple(row) for row in modes] == [
        ("LOOMING_DOT", 1, 3.0),
        ("MOVING_GRATING", 2, 12.0),
    ]


def test_registry_extracts_exact_semantic_recipe_and_step_identity(
    tmp_path: Path,
) -> None:
    zarr_path, semantic_hash = _write_semantic_archive(tmp_path)
    root = zarr.open_group(zarr_path, mode="r", use_consolidated=False)

    result = extract_stimulus_metadata(
        root,
        zarr_path=zarr_path,
        recording_id="semantic",
    )

    run = result.recording_runs[0]
    assert run["protocol_semantic_status"] == "verified"
    assert run["protocol_semantic_hash"] == semantic_hash
    assert run["protocol_recipe_label"] == "SOLID_BLACK -> CHASER"
    assert json.loads(run["protocol_recipe_mode_sequence_json"]) == [
        "SOLID_BLACK",
        "CHASER",
    ]
    assert run["protocol_trial_index_integrity_status"] == (
        "palette_computed_not_producer_asserted"
    )
    assert [
        (
            row["protocol_semantic_step_index"],
            row["stimulus_family"],
            row["display_context"],
        )
        for row in result.recording_steps
    ] == [
        (0, "solid_color", "solid_black"),
        (1, "chaser", "chaser"),
    ]


def test_registry_extracts_and_persists_sealed_v2_execution_proxy(
    tmp_path: Path,
) -> None:
    zarr_path, _semantic_hash = _write_semantic_archive(
        tmp_path,
        snapshot_v2=True,
    )
    root = zarr.open_group(zarr_path, mode="r", use_consolidated=False)
    result = extract_stimulus_metadata(
        root,
        zarr_path=zarr_path,
        recording_id="semantic-v2",
    )

    run = result.recording_runs[0]
    assert run["protocol_snapshot_schema_version"] == 2
    assert run["palette_computed_trial_index_sha256"] is None
    assert run["producer_protocol_trial_index_hash"].startswith("sha256:")
    assert run["protocol_trial_index_integrity_status"] == (
        "producer_asserted_exact_bytes"
    )
    assert run["protocol_execution_status"] == "complete"
    assert run["protocol_execution_hash"].startswith("sha256:")
    assert run["protocol_interval_axis"] == "stimulus_frame_num"
    assert run["protocol_acquisition_containment_status"] == (
        "unavailable_without_sealed_stimulus_to_acquisition_mapping"
    )
    assert run["protocol_frame_correspondence_proxy_status"] == "complete"
    assert len(run["protocol_frame_correspondence_proxy_manifest_sha256"]) == 64
    assert run["protocol_frame_correspondence_proxy_missing_count"] == 0
    assert result.recording_steps[1]["start_stimulus_frame_inclusive"] == 20
    assert result.recording_steps[1]["end_stimulus_frame_exclusive"] == 30
    assert set(
        json.loads(result.recording_steps[1]["protocol_execution_phases_json"])
    ) == set(CHASER_PHASE_NAMES)

    registry = Registry(tmp_path / "registry-v2.sqlite")
    try:
        dataset_id = registry.scan_zarr(zarr_path)
        stored = registry.conn.execute(
            """
            SELECT protocol_snapshot_schema_version,
                   protocol_execution_status,
                   protocol_interval_axis,
                   protocol_acquisition_containment_status,
                   protocol_frame_correspondence_proxy_status,
                   protocol_frame_correspondence_proxy_manifest_sha256,
                   protocol_frame_correspondence_proxy_missing_count
            FROM recording_stimulus_runs
            WHERE dataset_id = ? AND is_latest = 1
            """,
            (dataset_id,),
        ).fetchone()
    finally:
        registry.close()
    assert tuple(stored) == (
        2,
        "complete",
        "stimulus_frame_num",
        "unavailable_without_sealed_stimulus_to_acquisition_mapping",
        "complete",
        run["protocol_frame_correspondence_proxy_manifest_sha256"],
        0,
    )


def test_registry_rejects_tampered_v2_execution_proxy(tmp_path: Path) -> None:
    zarr_path, _semantic_hash = _write_semantic_archive(
        tmp_path,
        snapshot_v2=True,
    )
    writable = zarr.open_group(zarr_path, mode="a", use_consolidated=False)
    phase = writable[
        "analysis/stimulus_runs/stimulus_001/protocol_execution/"
        "frame_correspondence_proxy/chaser_phase_id"
    ]
    phase[0] = 2
    root = zarr.open_group(zarr_path, mode="r", use_consolidated=False)

    with pytest.raises(ProtocolSemanticContractError, match="not sealed"):
        extract_stimulus_metadata(
            root,
            zarr_path=zarr_path,
            recording_id="semantic-v2",
        )


def test_registry_scan_makes_semantic_cohorts_queryable(tmp_path: Path) -> None:
    zarr_path, semantic_hash = _write_semantic_archive(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        dataset_id = registry.scan_zarr(zarr_path)
        run = registry.conn.execute(
            """
            SELECT protocol_semantic_status, protocol_semantic_hash,
                   protocol_recipe_label, protocol_recipe_mode_sequence_json
            FROM recording_stimulus_runs
            WHERE dataset_id = ? AND is_latest = 1
            """,
            (dataset_id,),
        ).fetchone()
        steps = registry.conn.execute(
            """
            SELECT protocol_semantic_step_index, stimulus_family,
                   display_context, protocol_semantic_stimulus_mode_id
            FROM recording_stimulus_steps
            WHERE dataset_id = ?
            ORDER BY protocol_semantic_step_index
            """,
            (dataset_id,),
        ).fetchall()
        view = registry.conn.execute(
            """
            SELECT DISTINCT protocol_semantic_hash, protocol_recipe_label
            FROM recording_stimulus_mode_counts
            WHERE dataset_id = ? AND is_latest = 1
            """,
            (dataset_id,),
        ).fetchone()
    finally:
        registry.close()

    assert tuple(run) == (
        "verified",
        semantic_hash,
        "SOLID_BLACK -> CHASER",
        '["SOLID_BLACK","CHASER"]',
    )
    assert [tuple(row) for row in steps] == [
        (0, "solid_color", "solid_black", 4),
        (1, "chaser", "chaser", 12),
    ]
    assert tuple(view) == (semantic_hash, "SOLID_BLACK -> CHASER")


def test_registry_extraction_rejects_corrupt_semantic_trial_bytes(
    tmp_path: Path,
) -> None:
    zarr_path, _semantic_hash = _write_semantic_archive(tmp_path)
    root = zarr.open_group(zarr_path, mode="a", use_consolidated=False)
    trial = root[
        "analysis/stimulus_runs/stimulus_001/"
        "protocol_semantic_snapshot/protocol_trial_index_json_utf8"
    ]
    trial[0] = ord("[")

    with pytest.raises(
        ProtocolSemanticContractError,
        match="protocol_trial_index_json (is not valid JSON|must contain one JSON object)",
    ):
        extract_stimulus_metadata(
            root,
            zarr_path=zarr_path,
            recording_id="semantic",
        )


def test_migration_72_keeps_preexisting_stimulus_rows_semantically_unknown(
    tmp_path: Path,
) -> None:
    class RegistryThroughMigration72(Registry):
        def _schema_migrations(self):
            return [
                migration
                for migration in super()._schema_migrations()
                if migration[0] <= 72
            ]

    registry_path = tmp_path / "registry_v71.sqlite"
    with sqlite3.connect(registry_path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_utc TEXT NOT NULL
            );
            INSERT INTO schema_version VALUES (
                71, 'registry_instance_identity', '2026-08-24T00:00:00Z'
            );
            CREATE TABLE registry_identity (
                singleton_id INTEGER PRIMARY KEY,
                registry_uuid TEXT NOT NULL UNIQUE,
                identity_provenance TEXT NOT NULL,
                minted_at_utc TEXT NOT NULL
            );
            INSERT INTO registry_identity VALUES (
                1, '12345678-1234-4234-8234-123456789abc',
                'schema_managed', '2026-08-24T00:00:00Z'
            );
            CREATE TABLE datasets (dataset_id TEXT PRIMARY KEY);
            INSERT INTO datasets VALUES ('legacy_dataset');
            CREATE TABLE stimulus_protocols (
                protocol_hash TEXT PRIMARY KEY,
                protocol_name TEXT,
                step_count INTEGER NOT NULL,
                protocol_json TEXT NOT NULL,
                definition_source TEXT NOT NULL,
                extracted_utc TEXT NOT NULL
            );
            INSERT INTO stimulus_protocols VALUES (
                'old_hash', 'legacy', 1, '{}', 'materialized_steps',
                '2026-08-24T00:00:00Z'
            );
            CREATE TABLE recording_stimulus_runs (
                dataset_id TEXT NOT NULL,
                recording_id TEXT,
                stimulus_run_id TEXT NOT NULL,
                protocol_hash TEXT NOT NULL,
                protocol_name TEXT,
                is_latest INTEGER NOT NULL,
                step_count INTEGER NOT NULL,
                source_path TEXT NOT NULL,
                source_metadata_sha256 TEXT NOT NULL,
                source_zarr_path TEXT NOT NULL,
                extracted_utc TEXT NOT NULL,
                PRIMARY KEY (dataset_id, stimulus_run_id)
            );
            INSERT INTO recording_stimulus_runs VALUES (
                'legacy_dataset', 'legacy_recording', 'stimulus_001',
                'old_hash', 'legacy', 1, 1,
                'analysis/stimulus_runs/stimulus_001', 'source_hash',
                '/tmp/legacy.zarr', '2026-08-24T00:00:00Z'
            );
            CREATE TABLE recording_stimulus_steps (
                dataset_id TEXT NOT NULL,
                stimulus_run_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                step_name TEXT,
                stimulus_mode TEXT NOT NULL,
                start_camera_frame INTEGER,
                end_camera_frame INTEGER,
                duration_s REAL,
                step_attrs_json TEXT NOT NULL,
                PRIMARY KEY (dataset_id, stimulus_run_id, step_index)
            );
            INSERT INTO recording_stimulus_steps VALUES (
                'legacy_dataset', 'stimulus_001', 0, 'legacy step', 'CHASER',
                0, 100, 1.0, '{}'
            );
            CREATE TABLE recording_stimulus_modes (
                dataset_id TEXT NOT NULL,
                stimulus_run_id TEXT NOT NULL,
                stimulus_mode TEXT NOT NULL,
                step_count INTEGER NOT NULL,
                total_duration_s REAL,
                PRIMARY KEY (dataset_id, stimulus_run_id, stimulus_mode)
            );
            INSERT INTO recording_stimulus_modes VALUES (
                'legacy_dataset', 'stimulus_001', 'CHASER', 1, 1.0
            );
            """
        )

    # This fixture intentionally models only the tables migration 72 touches.
    # Pin the migration horizon so later, unrelated migrations cannot make the
    # historical boundary fixture pretend to be a complete latest-version DB.
    registry = RegistryThroughMigration72(registry_path)
    try:
        run = registry.conn.execute(
            """
            SELECT protocol_semantic_status, protocol_semantic_hash,
                   protocol_recipe_label
            FROM recording_stimulus_runs
            WHERE dataset_id = 'legacy_dataset'
            """
        ).fetchone()
        step = registry.conn.execute(
            """
            SELECT protocol_semantic_status, protocol_semantic_hash,
                   protocol_semantic_step_index, display_context
            FROM recording_stimulus_steps
            WHERE dataset_id = 'legacy_dataset'
            """
        ).fetchone()
        version = registry._current_schema_version()
    finally:
        registry.close()

    assert version == 72
    assert tuple(run) == (None, None, None)
    assert tuple(step) == (None, None, None, None)
