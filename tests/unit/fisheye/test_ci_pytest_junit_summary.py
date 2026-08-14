from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci_pytest_junit_summary import summarize_junit_reports
from scripts.ci_pytest_timings import (
    build_duration_baseline,
    measured_duration_seconds,
    runtime_source_sha256,
)
from scripts.run_ci_pytest_shards_local import build_shard_command


def _write_junit(path: Path, body: str) -> Path:
    path.write_text(
        f'<?xml version="1.0" encoding="utf-8"?><testsuites>{body}</testsuites>',
        encoding="utf-8",
    )
    return path


def test_summary_aggregates_parameterized_cases_by_file(tmp_path: Path) -> None:
    report = _write_junit(
        tmp_path / "shard.xml",
        """
        <testsuite name="pytest">
          <testcase classname="tests.unit.test_alpha" name="test_a[one]"
                    file="tests/unit/test_alpha.py" time="1.25" />
          <testcase classname="tests.unit.test_alpha" name="test_a[two]"
                    file="tests/unit/test_alpha.py" time="2.5" />
          <testcase classname="tests.unit.test_beta" name="test_b"
                    file="tests/unit/test_beta.py" time="0.125" />
        </testsuite>
        """,
    )

    summary = summarize_junit_reports([report], shard_index=3)

    assert summary["schema_id"] == "palette.ci_pytest_file_durations"
    assert summary["schema_version"] == 1
    assert summary["shard_index"] == 3
    assert summary["testcase_count"] == 3
    assert summary["duration_seconds"] == 3.875
    assert summary["files"] == {
        "tests/unit/test_alpha.py": {
            "testcase_count": 2,
            "duration_seconds": 3.75,
        },
        "tests/unit/test_beta.py": {
            "testcase_count": 1,
            "duration_seconds": 0.125,
        },
    }


@pytest.mark.parametrize(
    "classname",
    ["tests.unit.test_alpha", "tests.unit.test_alpha.TestCases"],
)
def test_summary_uses_classname_when_legacy_file_attribute_is_absent(
    tmp_path: Path,
    classname: str,
) -> None:
    report = _write_junit(
        tmp_path / "shard.xml",
        f'<testsuite><testcase classname="{classname}" '
        'name="test_a" time="1" /></testsuite>',
    )

    summary = summarize_junit_reports([report], shard_index=0)

    assert summary["files"] == {
        "tests/unit/test_alpha.py": {
            "testcase_count": 1,
            "duration_seconds": 1.0,
        }
    }


def test_summary_attributes_imported_cases_to_collector_module(tmp_path: Path) -> None:
    report = _write_junit(
        tmp_path / "shard.xml",
        '<testsuite><testcase '
        'classname="tests.unit.test_collector" name="test_a" '
        'file="tests/unit/shared_cases.py" time="2" /></testsuite>',
    )

    summary = summarize_junit_reports([report], shard_index=0)

    assert summary["files"] == {
        "tests/unit/test_collector.py": {
            "testcase_count": 1,
            "duration_seconds": 2.0,
        }
    }


@pytest.mark.parametrize("duration", ["-1", "nan", "infinity", "bad"])
def test_summary_rejects_invalid_durations(
    tmp_path: Path,
    duration: str,
) -> None:
    report = _write_junit(
        tmp_path / "shard.xml",
        '<testsuite><testcase classname="tests.unit.test_alpha" '
        f'name="test_a" time="{duration}" /></testsuite>',
    )

    with pytest.raises(ValueError, match="invalid duration"):
        summarize_junit_reports([report], shard_index=0)


def test_summary_rejects_unsafe_paths(tmp_path: Path) -> None:
    report = _write_junit(
        tmp_path / "shard.xml",
        '<testsuite><testcase name="test_a" '
        'file="../test_alpha.py" time="1" /></testsuite>',
    )

    with pytest.raises(ValueError, match="unsafe file path"):
        summarize_junit_reports([report], shard_index=0)


def test_local_shard_command_matches_hosted_ci_and_isolates_pytest_state(
    tmp_path: Path,
) -> None:
    command = build_shard_command(
        shard_index=3,
        shard_count=12,
        shard_root=tmp_path / "shard-03",
    )

    assert command[1].endswith("scripts/ci_pytest_shard.py")
    assert command[2:6] == ["--shard-index", "3", "--shard-count", "12"]
    assert "not gpu" in command
    assert "--durations=25" in command
    assert any(argument.startswith("--junitxml=") for argument in command)
    assert any(argument.startswith("--basetemp=") for argument in command)
    assert any(argument.startswith("cache_dir=") for argument in command)


def test_duration_baseline_is_hash_bound_and_falls_back_after_source_change(
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "tests/unit/test_alpha.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_alpha():\n    assert True\n", encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        """{
          "schema_id": "palette.ci_pytest_file_durations",
          "schema_version": 1,
          "files": {
            "tests/unit/test_alpha.py": {
              "duration_seconds": 12.5,
              "testcase_count": 1
            }
          }
        }""",
        encoding="utf-8",
    )
    baseline = build_duration_baseline(
        [summary_path],
        repository_root=tmp_path,
        source_ref="local-test",
        recorded_at_utc="2026-08-14T00:00:00Z",
    )

    assert measured_duration_seconds(
        test_file,
        repository_root=tmp_path,
        baseline=baseline,
    ) == 12.5

    test_file.write_text("def test_alpha():\n    assert False\n", encoding="utf-8")
    assert (
        measured_duration_seconds(
            test_file,
            repository_root=tmp_path,
            baseline=baseline,
        )
        is None
    )


def test_refine_collector_runtime_identity_includes_shared_case_source(
    tmp_path: Path,
) -> None:
    collector = (
        tmp_path
        / "tests/unit/fisheye/test_refine_online_coordinate_loading.py"
    )
    helper = (
        tmp_path
        / "tests/unit/fisheye/refine_online_coordinate_contract_cases.py"
    )
    collector.parent.mkdir(parents=True)
    collector.write_text("from .refine_online_coordinate_contract_cases import *\n")
    helper.write_text("def test_case():\n    assert True\n")
    before = runtime_source_sha256(collector, repository_root=tmp_path)

    helper.write_text("def test_case():\n    assert False\n")

    assert runtime_source_sha256(collector, repository_root=tmp_path) != before
