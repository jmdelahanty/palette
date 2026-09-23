"""Full Draft 2020-12 validation of the pinned shared transfer-v2 grammar."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
import pytest

from fisheye.shared.recording_transfer_snapshot import (
    MARKER_NAME,
    SNAPSHOT_PATH,
    build_snapshot,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
SCHEMA = json.loads((FIXTURES / "recording_transfer_v2.schema.json").read_bytes())
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())


def test_shared_schema_is_valid_draft_2020_12() -> None:
    assert SCHEMA["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    Draft202012Validator.check_schema(SCHEMA)


def test_all_declared_schema_formats_have_active_checkers() -> None:
    def declared_formats(value):
        if isinstance(value, dict):
            if "format" in value:
                yield value["format"]
            for nested in value.values():
                yield from declared_formats(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from declared_formats(nested)

    assert set(declared_formats(SCHEMA)) <= set(VALIDATOR.format_checker.checkers)


@pytest.mark.parametrize(
    "timestamp", ["not-a-time", "2026-09-06", "2026-99-99T25:61:00Z"]
)
def test_schema_refuses_malformed_delivery_time(timestamp: str) -> None:
    marker = json.loads(
        (FIXTURES / "recording_transfer_v2/rolling" / MARKER_NAME).read_bytes()
    )
    marker["delivery"]["created_utc"] = timestamp
    assert not VALIDATOR.is_valid(marker)


@pytest.mark.parametrize("bundle", ["whole", "rolling", "failed_optional_proof"])
@pytest.mark.parametrize("relative", [MARKER_NAME, SNAPSHOT_PATH])
def test_exact_shared_envelopes_validate(bundle: str, relative: str) -> None:
    payload = json.loads(
        (FIXTURES / "recording_transfer_v2" / bundle / relative).read_bytes()
    )
    VALIDATOR.validate(payload)


@pytest.mark.parametrize("bundle", ["whole", "rolling", "failed_optional_proof"])
def test_palette_reconstructed_transport_uses_same_full_schema(bundle: str) -> None:
    root = FIXTURES / "recording_transfer_v2" / bundle
    VALIDATOR.validate(build_snapshot(root, MARKER_NAME, destination=True))


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 1),
        ("schema_id", "citrus.transfer_completion_marker.v3"),
        ("required_consumer_profile", "legacy_whole_recording"),
        ("snapshot_id", "sha256:not-a-digest"),
        ("recording_layout", "guess_from_number_of_files"),
        ("parent_recording_count", 0),
        ("status", "incomplete"),
        ("extra", "not permitted"),
    ],
)
def test_schema_refuses_invalid_marker(field: str, value) -> None:
    marker = json.loads(
        (FIXTURES / "recording_transfer_v2/rolling" / MARKER_NAME).read_bytes()
    )
    marker[field] = value
    assert not VALIDATOR.is_valid(marker)


@pytest.mark.parametrize("kind", ["inventory", "parent", "clip", "output", "frame_map"])
def test_schema_nested_objects_are_closed(kind: str) -> None:
    snapshot = json.loads(
        (FIXTURES / "recording_transfer_v2/rolling" / SNAPSHOT_PATH).read_bytes()
    )
    targets = {
        "inventory": snapshot["inventory"][0],
        "parent": snapshot["parents"][0],
        "clip": snapshot["parents"][0]["clips"][0],
        "output": snapshot["parents"][0]["clips"][0]["outputs"][0],
        "frame_map": snapshot["parents"][0]["clips"][0]["outputs"][0]["frame_map"],
    }
    targets[kind]["unexpected"] = "refuse"
    assert not VALIDATOR.is_valid(snapshot)


def test_full_schema_validation_is_not_semantic_proof_acceptance() -> None:
    # A valid envelope deliberately does not assert codec/clock/proof admission.
    snapshot = json.loads(
        (
            FIXTURES / "recording_transfer_v2/failed_optional_proof" / SNAPSHOT_PATH
        ).read_bytes()
    )
    VALIDATOR.validate(snapshot)
    assert (
        snapshot["finalization"]["semantic_receipt_acceptance"]
        == "not_evaluated_by_transfer"
    )
    malformed = deepcopy(snapshot)
    malformed["inventory"][0]["size_bytes"] = -1
    assert not VALIDATOR.is_valid(malformed)
