"""Orange empty-dish background v2 (agent-contracts PR 129, merged 603da1fe): pins and the contract's own fixtures."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files as resource_files
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

# Copied byte-for-byte from agent-contracts 603da1fe orange-palette/empty_background_v2/fixtures/.
FIXTURES = Path(__file__).resolve().parents[2] / "contracts" / "orange_empty_background_v2"

PINS = {
    "orange_empty_background_capture_v2.schema.json": "c80c9888e7575d908d8fb82b0485dd3fc0cded684fc56109ea27de1ae843de29",
    "orange_image_levels_v1.schema.json": "8e0ac8488d3a7a4c86da0545b63639bb005df8f742c65922fae92068f1e94ef2",
    "orange_recording_empty_background_v2.schema.json": "88edb64b480048c07f0e08342e9503b00b5f7ee9c95078dbf1423f127a1d0f99",
    "orange_requester_v1.schema.json": "58819f17c23feb0bb526c7254d088af24434899ced57b3093c3c8a64bdc4a687",
}


def _schema_bytes(name: str) -> bytes:
    return resource_files("fisheye.shared").joinpath("contracts").joinpath(name).read_bytes()


def _schemas() -> dict[str, dict]:
    return {name: json.loads(_schema_bytes(name)) for name in PINS}


def _validator(document_name: str) -> Draft202012Validator:
    schemas = _schemas()
    # Capture and block reference the requester and levels schemas by $id.
    registry = Registry().with_resources(
        (schema["$id"], Resource.from_contents(schema)) for schema in schemas.values()
    )
    schema_name = (
        "orange_empty_background_capture_v2.schema.json"
        if document_name.startswith("capture_")
        else "orange_recording_empty_background_v2.schema.json"
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


def test_v1_recording_block_is_not_a_v2_block() -> None:
    # A v1 block has no schema_version; v2 requires it, so the grammars never overlap.
    v1_block = json.loads(
        (FIXTURES.parent / "orange_empty_background_v1" / "valid" / "snapshot_not_selected.json").read_text()
    )
    assert "schema_version" not in v1_block
    assert not _validator("snapshot_").is_valid(v1_block)


def test_fixture_counts_match_contract() -> None:
    assert len(list((FIXTURES / "valid").glob("*.json"))) == 6
    assert len(list((FIXTURES / "invalid").glob("*.json"))) == 21
