from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.analysis_workflows.materializers import (
    recording_behavior_distributions as subject,
)


def _write(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _record(**overrides):
    result = {
        "scope_id": "minute_one",
        "scope_label": "Minute one",
        "start_timestamp_ns_session": 0,
        "end_timestamp_ns_session_exclusive": 60_000_000_000,
    }
    result.update(overrides)
    return result


def test_named_time_bracket_json_preserves_exact_types(tmp_path: Path) -> None:
    brackets = subject._load_time_brackets(
        _write(tmp_path / "brackets.json", [_record()])
    )

    assert len(brackets) == 1
    assert brackets[0].scope_id == "minute_one"
    assert brackets[0].start_timestamp_ns_session == 0
    assert type(brackets[0].start_timestamp_ns_session) is int


@pytest.mark.parametrize(
    "override",
    [
        {"scope_id": 7},
        {"scope_label": None},
        {"start_timestamp_ns_session": "0"},
        {"end_timestamp_ns_session_exclusive": 1.5},
        {"start_timestamp_ns_session": False},
    ],
)
def test_named_time_bracket_json_rejects_coercible_or_boolean_fields(
    tmp_path: Path,
    override,
) -> None:
    path = _write(tmp_path / "brackets.json", [_record(**override)])

    with pytest.raises(
        subject.RecordingBehaviorDistributionMaterializerError,
        match="inexact JSON field types",
    ):
        subject._load_time_brackets(path)


def test_named_time_bracket_json_rejects_extra_fields(tmp_path: Path) -> None:
    path = _write(tmp_path / "brackets.json", [_record(comment="exploratory")])

    with pytest.raises(
        subject.RecordingBehaviorDistributionMaterializerError,
        match="inexact field set",
    ):
        subject._load_time_brackets(path)
