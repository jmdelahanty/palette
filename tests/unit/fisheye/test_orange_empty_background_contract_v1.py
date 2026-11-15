"""Orange empty-dish background v1 (agent-contracts PR 120, merged 7601cef2): pins and the contract's own fixtures."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files as resource_files
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

# Copied byte-for-byte from agent-contracts 7601cef2 orange-palette/empty_background_v1/fixtures/.
FIXTURES = Path(__file__).resolve().parents[2] / "contracts" / "orange_empty_background_v1"

PINS = {
    "orange_empty_background_capture_v1.schema.json": "b6b147c5c3f04b5db4e79e1e1659453559c263ea5934088e717e88ca2f64bf03",
    "orange_image_levels_v1.schema.json": "8e0ac8488d3a7a4c86da0545b63639bb005df8f742c65922fae92068f1e94ef2",
    "orange_recording_empty_background_v1.schema.json": "5ea84ae2052304a3d26c9fae6cbb6f56b15f4798059683e792f1003d69c86933",
}


def _schema_bytes(name: str) -> bytes:
    return resource_files("fisheye.shared").joinpath("contracts").joinpath(name).read_bytes()


def _schemas() -> dict[str, dict]:
    return {name: json.loads(_schema_bytes(name)) for name in PINS}


def _validator(document_name: str) -> Draft202012Validator:
    schemas = _schemas()
    # The descriptor references the levels schema by $id, so the set resolves together.
    registry = Registry().with_resources(
        (schema["$id"], Resource.from_contents(schema)) for schema in schemas.values()
    )
    schema_name = (
        "orange_empty_background_capture_v1.schema.json"
        if document_name.startswith("capture_")
        else "orange_recording_empty_background_v1.schema.json"
    )
    return Draft202012Validator(schemas[schema_name], registry=registry)


@pytest.mark.parametrize("name", sorted(PINS))
def test_packaged_schema_matches_pin(name: str) -> None:
    assert hashlib.sha256(_schema_bytes(name)).hexdigest() == PINS[name]


@pytest.mark.parametrize("name", sorted(PINS))
def test_packaged_schema_is_valid_2020_12(name: str) -> None:
    Draft202012Validator.check_schema(_schemas()[name])


@pytest.mark.parametrize(
    "path", sorted((FIXTURES / "valid").glob("*.json")), ids=lambda p: p.name
)
def test_valid_fixture_is_accepted(path: Path) -> None:
    errors = list(_validator(path.name).iter_errors(json.loads(path.read_text())))
    assert errors == []


@pytest.mark.parametrize(
    "path", sorted((FIXTURES / "invalid").glob("*.json")), ids=lambda p: p.name
)
def test_invalid_fixture_is_refused(path: Path) -> None:
    assert not _validator(path.name).is_valid(json.loads(path.read_text()))


def test_fixture_counts_match_contract() -> None:
    assert len(list((FIXTURES / "valid").glob("*.json"))) == 3
    assert len(list((FIXTURES / "invalid").glob("*.json"))) == 15
