"""B7: only the identity authority writes receipt-bound registry identity rows.

Enforcement corrections from docs/design/2026-10-07-intake-single-writer
(B7 and decision 4).  Every test uses a temporary registry and a real
receipt-bound current-source import produced by the importer fixture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import zarr

from fisheye.registry.db import DatasetMetadata, Registry
from fisheye.registry.recording_identity_authority import (
    RecordingIdentityAuthorityError,
)
from fisheye.registry.stage_complete import emit_stage_completion
from tests.unit.fisheye.test_recording_import_authority_integration import (
    _publish_current_import,
)


def _finalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Registry, Path, str]:
    plan, receipt = _publish_current_import(tmp_path, monkeypatch)
    registry = Registry(tmp_path / "registry.sqlite")
    finalized = registry.finalize_current_source_import(
        zarr_path=plan.zarr_path,
        receipt=receipt,
        decided_by="pytest",
    )
    return registry, plan.zarr_path, finalized.identity.dataset_id


def _recording_row(registry: Registry, recording_id: str = "recording-a") -> dict[str, Any]:
    row = registry.conn.execute(
        "SELECT * FROM recordings WHERE recording_id = ?;", (recording_id,)
    ).fetchone()
    assert row is not None
    return dict(row)


def _dataset_row(registry: Registry, dataset_id: str) -> dict[str, Any]:
    row = registry.conn.execute(
        "SELECT * FROM datasets WHERE dataset_id = ?;", (dataset_id,)
    ).fetchone()
    assert row is not None
    return dict(row)


def _without(row: dict[str, Any], *fields: str) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in fields}


def _derived_root(zarr_path: Path, **attrs: Any) -> zarr.Group:
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3)
    root.attrs.update(attrs)
    return root


_CONTRADICTING_CONTEXT = {
    "recording_id": "recording-a",
    "session_uuid": "session-a",
    "zarr_purpose": "training",
    "zarr_use": "training",
    "recording_type": "stale-training-copy",
    "recording_subtype": "embedded",
    "behavior_mode": "embedded",
    "artifact_schema_id": "training_v9",
    "recording_intent": "stale-intent",
    "data_origin": "synthetic",
    "context_source": "stale-copy",
}


def test_in_tree_derived_zarr_cannot_change_bound_recording_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, source_path, source_dataset_id = _finalized(tmp_path, monkeypatch)
    try:
        before_recording = _recording_row(registry)
        before_source = _dataset_row(registry, source_dataset_id)
        derived_path = source_path.parent / "training.zarr"
        root = _derived_root(derived_path, **_CONTRADICTING_CONTEXT)

        derived_dataset_id = registry.register_from_root(root, derived_path)

        assert derived_dataset_id != source_dataset_id
        derived = _dataset_row(registry, derived_dataset_id)
        assert derived["recording_id"] == "recording-a"
        assert derived["zarr_path"] == str(derived_path.resolve())
        assert _recording_row(registry) == before_recording
        assert _dataset_row(registry, source_dataset_id) == before_source
    finally:
        registry.close()


def test_out_of_tree_training_zarr_registers_its_own_dataset_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, _source_path, source_dataset_id = _finalized(tmp_path, monkeypatch)
    try:
        before_recording = _recording_row(registry)
        before_source = _dataset_row(registry, source_dataset_id)
        derived_path = tmp_path / "training" / "datasets" / "set-a" / "train.zarr"
        root = _derived_root(derived_path, **_CONTRADICTING_CONTEXT)

        derived_dataset_id = registry.register_from_root(root, derived_path)

        assert derived_dataset_id != source_dataset_id
        derived = _dataset_row(registry, derived_dataset_id)
        assert derived["recording_id"] == "recording-a"
        assert derived["artifact_kind"] == "derived_training_merge"
        assert derived["zarr_use"] == "training"
        assert _recording_row(registry) == before_recording
        assert _dataset_row(registry, source_dataset_id) == before_source
        # Replay is idempotent and still leaves the authority rows alone.
        assert registry.register_from_root(root, derived_path) == derived_dataset_id
        assert _recording_row(registry) == before_recording
    finally:
        registry.close()


def test_derived_zarr_sharing_base_dataset_id_gets_its_own_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A derived root whose base dataset id names the bound row must not take it over."""

    registry, _source_path, source_dataset_id = _finalized(tmp_path, monkeypatch)
    try:
        before_source = _dataset_row(registry, source_dataset_id)
        derived_path = tmp_path / "derived" / "copy.zarr"
        root = _derived_root(
            derived_path,
            recording_id="derived-recording",
            session_uuid=source_dataset_id,
            zarr_purpose="analysis",
        )
        # The derived root claims the bound dataset id as its session.  It
        # cannot rewrite the bound dataset's locator; it gets its own row.
        derived_dataset_id = registry.register_from_root(root, derived_path)
        assert derived_dataset_id != source_dataset_id
        assert _dataset_row(registry, source_dataset_id) == before_source
        assert _dataset_row(registry, derived_dataset_id)["zarr_path"] == str(
            derived_path.resolve()
        )
    finally:
        registry.close()


def test_derived_zarr_with_contradicting_session_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, source_path, _source_dataset_id = _finalized(tmp_path, monkeypatch)
    try:
        before_recording = _recording_row(registry)
        dataset_count = registry.conn.execute(
            "SELECT COUNT(*) FROM datasets;"
        ).fetchone()[0]
        derived_path = source_path.parent / "training.zarr"
        root = _derived_root(
            derived_path,
            **{**_CONTRADICTING_CONTEXT, "session_uuid": "session-other"},
        )
        with pytest.raises(
            RecordingIdentityAuthorityError,
            match="bound to a verified source",
        ):
            registry.register_from_root(root, derived_path)
        assert _recording_row(registry) == before_recording
        assert registry.conn.execute(
            "SELECT COUNT(*) FROM datasets;"
        ).fetchone()[0] == dataset_count
    finally:
        registry.close()


def test_upsert_recording_refuses_bound_context_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, _source_path, _dataset_id = _finalized(tmp_path, monkeypatch)
    try:
        before = _recording_row(registry)
        with pytest.raises(
            RecordingIdentityAuthorityError,
            match="recording_type",
        ):
            registry.upsert_recording(
                recording_id="recording-a",
                recording_type="other",
            )
        assert _recording_row(registry) == before
        # An exact replay of authority-owned values changes nothing.
        registry.upsert_recording(
            recording_id="recording-a",
            session_uuid="session-a",
            recording_type="behavior",
        )
        assert _recording_row(registry) == before
    finally:
        registry.close()


def test_upsert_dataset_refuses_bound_context_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, source_path, dataset_id = _finalized(tmp_path, monkeypatch)
    try:
        before = _dataset_row(registry, dataset_id)
        with pytest.raises(RecordingIdentityAuthorityError, match="zarr_use"):
            registry.upsert_dataset(
                dataset_id,
                session_uuid="session-a",
                zarr_path=source_path.resolve(),
                recording_id="recording-a",
                zarr_use="training",
            )
        assert _dataset_row(registry, dataset_id) == before
        registry.upsert_dataset(
            dataset_id,
            session_uuid="session-a",
            zarr_path=source_path.resolve(),
            recording_id="recording-a",
            zarr_use="analysis",
        )
        assert _without(_dataset_row(registry, dataset_id), "last_seen_utc") == (
            _without(before, "last_seen_utc")
        )
    finally:
        registry.close()


def test_stage_completion_on_current_source_updates_status_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, source_path, dataset_id = _finalized(tmp_path, monkeypatch)
    try:
        before_dataset = _dataset_row(registry, dataset_id)
        before_recording = _recording_row(registry)
        stale = DatasetMetadata(
            dataset_id="session-a",
            session_uuid="session-a",
            recording_id="recording-a",
            zarr_use="training",
            zarr_purpose="training",
            source_layout=None,
            source_frame_index_path=None,
            source_recording_frame_index_path=None,
            source_frame_index_schema=None,
        )
        wrote = emit_stage_completion(
            None,
            source_path,
            step_name="detect",
            status="error",
            source="pytest",
            registry=registry,
            metadata=stale,
            invalidate_on_ok=False,
        )
        assert wrote is True
        assert _without(_dataset_row(registry, dataset_id), "last_seen_utc") == (
            _without(before_dataset, "last_seen_utc")
        )
        assert _recording_row(registry) == before_recording
        status = registry.conn.execute(
            """
            SELECT dataset_id, recording_id, status
            FROM recording_step_status WHERE step_name = 'detect';
            """
        ).fetchall()
        assert [tuple(row) for row in status] == [
            (dataset_id, "recording-a", "error")
        ]
    finally:
        registry.close()


def test_stage_completion_from_current_source_root_uses_bound_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, source_path, dataset_id = _finalized(tmp_path, monkeypatch)
    try:
        before_dataset = _dataset_row(registry, dataset_id)
        dataset_count = registry.conn.execute(
            "SELECT COUNT(*) FROM datasets;"
        ).fetchone()[0]
        root = zarr.open_group(
            str(source_path), mode="r", zarr_format=3, use_consolidated=False
        )
        assert emit_stage_completion(
            root,
            source_path,
            step_name="detect",
            status="error",
            source="pytest",
            registry=registry,
            invalidate_on_ok=False,
        )
        assert registry.conn.execute(
            "SELECT COUNT(*) FROM datasets;"
        ).fetchone()[0] == dataset_count
        assert _without(_dataset_row(registry, dataset_id), "last_seen_utc") == (
            _without(before_dataset, "last_seen_utc")
        )
        row = registry.conn.execute(
            "SELECT dataset_id FROM recording_step_status WHERE step_name = 'detect';"
        ).fetchone()
        assert row["dataset_id"] == dataset_id
    finally:
        registry.close()


def test_current_source_projection_mirrors_authority_null(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, source_path, _dataset_id = _finalized(tmp_path, monkeypatch)
    try:
        # Simulate stale values left by an earlier projection or writer.
        registry.conn.execute(
            """
            UPDATE recordings
            SET recording_intent = 'stale-intent',
                data_origin = 'stale-origin',
                context_source = 'stale-source',
                rig_id = 'stale-rig'
            WHERE recording_id = 'recording-a';
            """
        )
        registry.conn.commit()

        registry.refresh_bound_current_source_import(zarr_path=source_path)

        row = _recording_row(registry)
        assert row["recording_intent"] is None
        assert row["data_origin"] is None
        assert row["context_source"] is None
        assert row["rig_id"] is None
        # Values the authority does declare are still projected.
        assert row["recording_type"] == "behavior"
        assert row["camera_id"] == "2010093"
    finally:
        registry.close()


def test_historical_projection_keeps_coalesce(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recordings" / "legacy" / "zarr" / "analysis.zarr"
    root = _derived_root(
        zarr_path,
        recording_id="legacy-recording",
        session_uuid="legacy-session",
        zarr_purpose="analysis",
        recording_type="behavior",
        recording_intent="historical-intent",
        data_origin="real",
    )
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.register_from_root(root, zarr_path)
        del root.attrs["recording_intent"]
        del root.attrs["data_origin"]
        registry.register_from_root(root, zarr_path)
        row = _recording_row(registry, "legacy-recording")
        assert row["recording_intent"] == "historical-intent"
        assert row["data_origin"] == "real"
        # Historical rows remain writable through the generic upsert.
        registry.upsert_recording(
            recording_id="legacy-recording", recording_type="calibration"
        )
        assert _recording_row(registry, "legacy-recording")["recording_type"] == (
            "calibration"
        )
    finally:
        registry.close()


def test_recording_entity_backfill_skips_sibling_of_verified_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fisheye.registry.maintenance import _backfill_recording_entities

    registry, source_path, _source_dataset_id = _finalized(tmp_path, monkeypatch)
    try:
        sibling_path = source_path.parent / "training.zarr"
        root = _derived_root(
            sibling_path,
            recording_id="recording-a",
            session_uuid="session-a",
            zarr_purpose="training",
        )
        sibling_dataset_id = registry.register_from_root(root, sibling_path)
        before = _recording_row(registry)
        recording_count = registry.conn.execute(
            "SELECT COUNT(*) FROM recordings;"
        ).fetchone()[0]

        _backfill_recording_entities(registry, dry_run=False)
        registry.conn.commit()

        assert _recording_row(registry) == before
        # No legacy ``recording_id = session_uuid`` row is minted and the
        # sibling stays linked to the authority-bound recording.
        assert registry.conn.execute(
            "SELECT COUNT(*) FROM recordings;"
        ).fetchone()[0] == recording_count
        assert _dataset_row(registry, sibling_dataset_id)["recording_id"] == (
            "recording-a"
        )
    finally:
        registry.close()
