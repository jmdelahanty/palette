from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CI_WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"


def _ci_jobs() -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(CI_WORKFLOW_PATH.read_text(encoding="utf-8"))
    jobs = payload.get("jobs")
    assert isinstance(jobs, dict), "CI workflow must declare a jobs mapping."
    return jobs


def _run_commands(job: dict[str, Any]) -> str:
    steps = job.get("steps")
    assert isinstance(steps, list), "Every quality gate job must declare steps."
    return "\n".join(
        str(step.get("run") or "") for step in steps if isinstance(step, dict)
    )


def _normalized_needs(job: dict[str, Any]) -> set[str]:
    needs = job.get("needs", ())
    if isinstance(needs, str):
        return {needs}
    assert isinstance(needs, (list, tuple)), "Job needs must be a string or sequence."
    return {str(value) for value in needs}


def test_ci_quality_gates_are_independent_jobs() -> None:
    jobs = _ci_jobs()
    gate_commands = {
        "import-boundaries": "lint-imports --config pyproject.toml",
        "file-size-ratchet": "python scripts/check_file_size_ratchet.py",
        "quality": "python -m pytest --collect-only -q tests/",
    }

    for job_name, command in gate_commands.items():
        assert job_name in jobs, f"Missing independent CI gate job: {job_name}."
        assert command in _run_commands(jobs[job_name])

    assert "python scripts/check_fps_authority_access.py" in _run_commands(
        jobs["import-boundaries"]
    )
    assert (
        "git diff --exit-code -- scripts/fps_authority_access_ratchet_baseline.json"
        in _run_commands(jobs["import-boundaries"])
    )

    for job_name in gate_commands:
        gate_dependencies = _normalized_needs(jobs[job_name]) & gate_commands.keys()
        assert not gate_dependencies, (
            f"CI gate {job_name!r} depends on other quality gates: "
            f"{sorted(gate_dependencies)}. A failed gate must not suppress another."
        )


def test_each_quality_gate_command_has_exactly_one_job_owner() -> None:
    jobs = _ci_jobs()
    commands = {
        "lint-imports --config pyproject.toml": "import-boundaries",
        "python scripts/check_fps_authority_access.py": "import-boundaries",
        "git diff --exit-code -- scripts/fps_authority_access_ratchet_baseline.json": (
            "import-boundaries"
        ),
        "python scripts/check_file_size_ratchet.py": "file-size-ratchet",
        "python -m pytest --collect-only -q tests/": "quality",
    }

    for command, expected_owner in commands.items():
        owners = [
            job_name
            for job_name, job in jobs.items()
            if command in _run_commands(job)
        ]
        assert owners == [expected_owner]


def test_test_shards_publish_complete_file_timing_evidence() -> None:
    test_job = _ci_jobs()["tests"]
    assert test_job["strategy"]["matrix"]["shard"] == list(range(16))
    steps = test_job["steps"]
    by_name = {step.get("name"): step for step in steps}

    run_command = str(by_name["Run non-GPU test shard"]["run"])
    assert "--shard-count 16" in run_command
    assert "--junitxml=" in run_command
    assert "junit_family=legacy" in run_command
    assert "junit_duration_report=total" in run_command
    assert "ci_pytest_junit_summary.py" in str(
        by_name["Summarize per-file pytest durations"]["run"]
    )
    upload = by_name["Upload pytest duration evidence"]
    assert upload["uses"] == "actions/upload-artifact@v4"
    assert upload["if"] == "always()"


def test_subject_shape_fixture_restores_for_all_shards_but_has_one_writer() -> None:
    steps = _ci_jobs()["tests"]["steps"]
    by_name = {step.get("name"): step for step in steps}

    assert "--shard-count 16" in str(
        by_name["Select canonical subject-shape fixture owner"]["run"]
    )
    assert "--shard-count 16" in str(
        by_name["Select canonical subject-mask finalizer fixture owner"]["run"]
    )
    restore = by_name["Restore canonical subject-shape source fixture"]
    assert restore["uses"] == "actions/cache/restore@v4"
    assert "if" not in restore
    save = by_name["Save canonical subject-shape source fixture"]
    assert save["uses"] == "actions/cache/save@v4"
    assert "subject-shape-fixture-owner.outputs.selected == 'true'" in save["if"]
