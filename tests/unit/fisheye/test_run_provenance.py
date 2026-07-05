from __future__ import annotations

from pathlib import Path

from fisheye.shared.run_provenance import append_input_artifacts
from fisheye.shared.run_provenance import build_run_provenance
from fisheye.shared.run_provenance import build_run_provenance_from_stage_record
from fisheye.shared.run_provenance import build_writer_run_provenance
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


def test_run_provenance_validator_does_not_require_input_artifacts() -> None:
    result = validate_run_provenance(_valid_payload())

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
    assert left["input_artifacts"] == []
    assert "system" not in left


def test_build_writer_run_provenance_defaults_to_minimal_valid_payload() -> None:
    payload = build_writer_run_provenance(
        command="unit-writer",
        params={"b": 2, "a": 1},
        input_run_ids={"detect": "detect_001"},
    )

    assert payload["command"] == "unit-writer"
    assert payload["config_hash"] == sha256_payload({"a": 1, "b": 2})
    assert payload["input_run_ids"] == {"detect": "detect_001"}
    assert payload["input_artifacts"] == []
    assert "system" not in payload
    assert validate_run_provenance(payload).valid is True


def test_build_run_provenance_accepts_input_artifacts() -> None:
    payload = build_run_provenance(
        command="unit",
        params={},
        input_artifacts=[
            {
                "role": "detect_model",
                "path": Path("/tmp/model.pt"),
                "fingerprint_scheme": "content_v1",
                "sha256": "a" * 64,
            }
        ],
        include_system_context=False,
    )

    assert payload["input_artifacts"] == [
        {
            "role": "detect_model",
            "path": "/tmp/model.pt",
            "fingerprint_scheme": "content_v1",
            "sha256": "a" * 64,
        }
    ]
    assert validate_run_provenance(payload).valid is True


def test_append_input_artifacts_merges_into_caller_supplied_provenance() -> None:
    provenance = _valid_payload(
        params={"threshold": 0.5},
        input_artifacts=[
            {
                "role": "detect_model",
                "path": "/tmp/old.pt",
                "fingerprint_scheme": "content_v1",
                "sha256": "a" * 64,
            }
        ],
    )

    merged = append_input_artifacts(
        provenance,
        [
            {
                "role": "keypoint_model",
                "path": Path("/tmp/pose.pt"),
                "fingerprint_scheme": "content_v1",
                "sha256": "b" * 64,
            }
        ],
    )

    assert merged is not provenance
    assert merged["params"] == {"threshold": 0.5}
    assert merged["input_artifacts"] == [
        {
            "role": "detect_model",
            "path": "/tmp/old.pt",
            "fingerprint_scheme": "content_v1",
            "sha256": "a" * 64,
        },
        {
            "role": "keypoint_model",
            "path": "/tmp/pose.pt",
            "fingerprint_scheme": "content_v1",
            "sha256": "b" * 64,
        },
    ]


def test_append_input_artifacts_replaces_duplicate_role_path() -> None:
    provenance = _valid_payload(
        input_artifacts=[
            {
                "role": "detect_model",
                "path": "/tmp/model.pt",
                "fingerprint_scheme": "content_v1",
                "sha256": "a" * 64,
            }
        ],
    )

    merged = append_input_artifacts(
        provenance,
        [
            {
                "role": "detect_model",
                "path": "/tmp/model.pt",
                "fingerprint_scheme": "content_v1",
                "sha256": "c" * 64,
                "source": "computed",
            }
        ],
    )

    assert merged["input_artifacts"] == [
        {
            "role": "detect_model",
            "path": "/tmp/model.pt",
            "fingerprint_scheme": "content_v1",
            "sha256": "c" * 64,
            "source": "computed",
        }
    ]


def test_build_run_provenance_from_stage_record_reuses_parameters_and_inputs() -> None:
    payload = build_run_provenance_from_stage_record(
        {
            "stage": "detect_quality",
            "command": "quality --run detect_001",
            "parameters": {"jump_threshold": 100},
            "inputs": {"source_detect_run": "detect_001"},
        }
    )

    assert payload["command"] == "quality --run detect_001"
    assert payload["params"] == {"jump_threshold": 100}
    assert payload["input_run_ids"] == {"source_detect_run": "detect_001"}
    assert validate_run_provenance(payload).valid is True


def test_scheduler_context_captures_lsf_job_identity(monkeypatch) -> None:
    monkeypatch.setenv("LSB_JOBID", "12345")
    monkeypatch.setenv("LSB_JOBINDEX", "7")

    lsf = scheduler_context()["lsf"]
    assert lsf["lsb_jobid"] == "12345"
    assert lsf["lsb_jobindex"] == "7"
