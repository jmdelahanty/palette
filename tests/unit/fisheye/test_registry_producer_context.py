"""Registry migration 074: rows record where their scientific context came from.

Producer-declared rows (``citrus.parent_recording_context``) carry an optional
free-label subtype independent of behavior_mode; operator/legacy rows keep the
historical vocabulary and subtype==behavior_mode rules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.registry.maintenance import _check_registry_integrity
from fisheye.registry.migrations import MIGRATION_METHODS
from fisheye.registry.recording_context_audit import recording_context_row_issues
from tests.unit.fisheye.registry_test_fixtures import registry_from_empty_template

PRODUCER = "citrus.parent_recording_context"
PRODUCER_FIELDS = {
    "context_source": PRODUCER,
    "recording_context_schema_version": 2,
    "recording_intent": "stimulus_experiment",
    "data_origin": "acquired",
}
VOCAB = {
    "allowed_recording_types": {"behavior"},
    "allowed_subtypes_by_type": {"behavior": {"free", "embedded"}},
}


def _row(**fields) -> dict:
    row = {
        "recording_id": "rec",
        "recording_type": "behavior",
        "recording_subtype": None,
        "behavior_mode": "embedded",
        "artifact_schema_id": "orange_transfer_parent_v1",
        "context_source": None,
        "recording_context_schema_version": None,
        "recording_intent": None,
        "data_origin": None,
    }
    row.update(fields)
    return row


def _codes(row: dict) -> set[str]:
    return {code for code, _ in recording_context_row_issues(row, **VOCAB)}


def test_migration_074_is_registered_and_adds_nullable_columns(tmp_path: Path) -> None:
    assert MIGRATION_METHODS[-1] == (
        74, "recording_producer_context", "_migration_074_recording_producer_context"
    )
    registry = registry_from_empty_template(tmp_path / "registry.sqlite")
    columns = {
        row["name"]: row
        for row in registry.conn.execute("PRAGMA table_info(recordings);").fetchall()
    }
    for name in PRODUCER_FIELDS:
        assert name in columns and not columns[name]["notnull"]
    registry.close()


def test_upsert_round_trips_and_never_clears_producer_context(tmp_path: Path) -> None:
    registry = registry_from_empty_template(tmp_path / "registry.sqlite")
    registry.upsert_recording(
        recording_id="rec", recording_type="behavior", behavior_mode="embedded",
        **PRODUCER_FIELDS,
    )
    registry.upsert_recording(recording_id="rec", recording_name="renamed")
    row = registry.conn.execute(
        "SELECT recording_subtype, context_source, recording_context_schema_version, "
        "recording_intent, data_origin FROM recordings WHERE recording_id = 'rec';"
    ).fetchone()
    assert row["recording_subtype"] is None  # not specified stays NULL
    assert {k: row[k] for k in PRODUCER_FIELDS} == PRODUCER_FIELDS
    registry.close()


def test_producer_row_may_omit_subtype_or_use_a_free_label() -> None:
    assert _codes(_row(**PRODUCER_FIELDS)) == set()
    assert _codes(_row(recording_subtype="dish_stimulus", **PRODUCER_FIELDS)) == set()


def test_legacy_row_keeps_the_historical_rules() -> None:
    assert _codes(_row()) == {"recording_missing_subtype"}
    assert _codes(_row(recording_subtype="dish_stimulus")) == {
        "recording_invalid_subtype",
        "recording_behavior_mode_mismatch",
    }
    assert _codes(_row(recording_subtype="free")) == {"recording_behavior_mode_mismatch"}
    assert _codes(_row(recording_subtype="embedded")) == set()


@pytest.mark.parametrize(
    "field, value",
    [
        ("recording_context_schema_version", 3),
        ("recording_context_schema_version", None),
        ("recording_intent", "unknown"),
        ("data_origin", None),
    ],
)
def test_producer_row_checks_its_declared_context(field: str, value) -> None:
    row = _row(**{**PRODUCER_FIELDS, field: value})
    assert _codes(row) == {f"recording_invalid_{field}"}


def test_producer_row_still_needs_known_type_and_behavior_mode() -> None:
    codes = _codes(_row(recording_type="stimulus", behavior_mode="swimming", **PRODUCER_FIELDS))
    assert codes == {"recording_invalid_type", "recording_invalid_behavior_mode"}


def test_integrity_check_scopes_subtype_rules_by_context_source(tmp_path: Path) -> None:
    registry = registry_from_empty_template(tmp_path / "registry.sqlite")
    registry.upsert_recording(
        recording_id="producer", recording_type="behavior", behavior_mode="embedded",
        recording_subtype="dish_stimulus", artifact_schema_id="orange_transfer_parent_v1",
        **PRODUCER_FIELDS,
    )
    registry.upsert_recording(
        recording_id="legacy", recording_type="behavior", behavior_mode="embedded",
        recording_subtype="dish_stimulus", artifact_schema_id="orange_transfer_parent_v1",
    )
    flagged = {
        issue.run_id
        for issue in _check_registry_integrity(registry)
        if issue.code.startswith("recording_") and "subtype" in issue.code
    }
    assert flagged == {"legacy"}
    registry.close()
