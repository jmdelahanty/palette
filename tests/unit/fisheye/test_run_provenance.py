from __future__ import annotations

from pathlib import Path

from fisheye.shared.run_provenance import build_run_provenance
from fisheye.shared.run_provenance import scheduler_context
from fisheye.shared.run_provenance import sha256_payload
from fisheye.shared.run_provenance import validate_run_provenance


def _valid_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "git_sha": "a" * 40,
        "config_hash": "b" * 64,
        "params": {},
        "input_run_ids": {},
        "command": "",
        "fisheye_version": None,
    }
    payload.update(overrides)
    return payload


def test_run_provenance_validator_requires_only_git_sha_and_config_hash_values() -> None:
    result = validate_run_provenance(_valid_payload(command="", fisheye_version=None))

    assert result.valid is True
    assert result.errors == ()


def test_run_provenance_validator_rejects_missing_value_required_fields() -> None:
    result = validate_run_provenance(_valid_payload(git_sha="", config_hash=None))

    assert result.valid is False
    assert result.missing_value_required == ("git_sha", "config_hash")
    assert "missing value-required fields" in result.errors[0]


def test_run_provenance_validator_rejects_missing_structural_keys() -> None:
    payload = {
        "git_sha": "a" * 40,
        "config_hash": "b" * 64,
    }

    result = validate_run_provenance(payload)

    assert result.valid is False
    assert result.missing_structural == (
        "params",
        "input_run_ids",
        "command",
        "fisheye_version",
    )


def test_build_run_provenance_hashes_normalized_params_without_system_context() -> None:
    left = build_run_provenance(
        command="unit",
        params={"path": Path("/tmp/a"), "a": 1},
        input_run_ids={"detect": "detect_001"},
        include_system_context=False,
    )
    right = build_run_provenance(
        command="unit",
        params={"a": 1, "path": Path("/tmp/a")},
        input_run_ids={"detect": "detect_001"},
        include_system_context=False,
    )

    assert left["config_hash"] == right["config_hash"]
    assert left["config_hash"] == sha256_payload({"a": 1, "path": "/tmp/a"})
    assert left["input_run_ids"] == {"detect": "detect_001"}
    assert "system" not in left


def test_scheduler_context_captures_lsf_job_identity(monkeypatch) -> None:
    monkeypatch.setenv("LSB_JOBID", "12345")
    monkeypatch.setenv("LSB_JOBINDEX", "7")

    lsf = scheduler_context()["lsf"]
    assert lsf["lsb_jobid"] == "12345"
    assert lsf["lsb_jobindex"] == "7"
