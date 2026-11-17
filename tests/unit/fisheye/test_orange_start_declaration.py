"""Orange recording start declaration v1/v2 and subject confirmation v2 (agent-contracts #138)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from fisheye.shared import orange_start_declaration as declaration
from fisheye.shared.orange_start_declaration import StartDeclarationError
from fisheye.shared.unified_h5.common import canonical_json

FIXTURES = Path(__file__).resolve().parents[2] / "contracts" / "orange_recording_start_declaration_v2"
ORCHARD = FIXTURES / "orchard_v2"

# The contract's CROSS_FIELD table: every other invalid fixture is refused by a schema.
CROSS_FIELD = {
    "declaration_v2_orchard_sign_off_by_differs.json": "R1",
    "declaration_v2_orchard_plan_copy_path_wrong_hash.json": "R2",
    "declaration_v2_orchard_sign_off_copy_path_wrong_hash.json": "R2",
    "confirmation_spec_copy_path_mismatch.json": "C3",
    "confirmation_config_differs_from_declaration.json": "C1",
    "confirmation_cameras_differ_from_resolution.json": "C2",
    "declaration_dish_id_257_bytes.json": "R3",
    "declaration_v2_orchard_plan_id_257_bytes.json": "R3",
    "declaration_v2_orchard_entry_id_257_bytes.json": "R3",
    "declaration_schema_version_float.json": "T1",
    "declaration_v2_subject_references_schema_version_float.json": "T1",
    "declaration_subject_count_float.json": "T1",
    "declaration_v2_orchard_plan_revision_float.json": "T1",
}


def _check(path: Path) -> None:
    document = declaration.parse_document(path.read_bytes(), label=path.name)
    if document.get("schema_id") == declaration.CONFIRMATION_SCHEMA_ID:
        declaration.validate_confirmation_v2(document)
    else:
        declaration.validate_declaration(document)


def _orchard_declaration() -> dict:
    return json.loads(next(ORCHARD.glob("*.orange_declaration.json")).read_bytes())


def _orchard_files() -> dict[str, bytes]:
    return {p.relative_to(ORCHARD).as_posix(): p.read_bytes()
            for p in (ORCHARD / "run_plan").rglob("*") if p.is_file()}


def test_pinned_schemas_match_the_packaged_bytes():
    for name in declaration.SCHEMA_FILES:
        assert declaration._pinned(name)["additionalProperties"] is False


@pytest.mark.parametrize(
    "path", sorted((FIXTURES / "valid").glob("*.json")) + sorted(ORCHARD.glob("*.json")),
    ids=lambda p: p.name,
)
def test_valid_fixtures_are_accepted(path):
    _check(path)


@pytest.mark.parametrize("path", sorted((FIXTURES / "invalid").glob("*.json")), ids=lambda p: p.name)
def test_invalid_fixtures_are_refused_for_their_rule(path):
    rule = CROSS_FIELD.get(path.name, "schema")
    with pytest.raises(StartDeclarationError, match=f"^orange_start_declaration:{rule}:"):
        _check(path)


def test_fixture_counts_match_the_contract():
    assert len(list((FIXTURES / "valid").glob("*.json"))) == 7
    assert len(list((FIXTURES / "invalid").glob("*.json"))) == 39


@pytest.mark.parametrize("name", sorted(n for n, rule in CROSS_FIELD.items() if rule == "T1"))
def test_float_tokens_pass_the_schemas_alone(name):
    """Why T1 exists: json.loads turns 2.0 into a float that JSON Schema calls an integer."""

    from jsonschema import Draft202012Validator

    document = json.loads((FIXTURES / "invalid" / name).read_bytes())
    schema = declaration._pinned(f"declaration_v{int(document['schema_version'])}")
    assert list(Draft202012Validator(schema).iter_errors(document)) == []


def test_r5_an_orchard_plan_must_name_every_recording_camera():
    confirmation = json.loads((FIXTURES / "valid" / "confirmation_run_spec_orchard_v2.json").read_bytes())
    confirmation["cameras"]["2010097"] = {"dish_id": None, "subject_count": None}
    with pytest.raises(StartDeclarationError, match="^orange_start_declaration:R5:"):
        declaration.validate_confirmation_v2(confirmation)


def test_stated_blank_is_read_from_the_declaration_not_the_cameras_view():
    confirmation = json.loads((FIXTURES / "valid" / "confirmation_local_control_operator_v2.json").read_bytes())
    assert confirmation["cameras"]["2010096"] == {"dish_id": None, "subject_count": None}
    assert declaration.stated_blank(confirmation, "2010096")
    assert not declaration.stated_blank(confirmation, "2010093")
    panel = json.loads((FIXTURES / "valid" / "confirmation_gui_operator.json").read_bytes())
    assert not any(declaration.stated_blank(panel, camera) for camera in panel["cameras"])


def test_the_orchard_copies_verify_and_are_recorded():
    files = _orchard_files()
    record = declaration.check_run_plan(_orchard_declaration()["source"], files.get)
    assert record == {"copies_verified": sorted(files), "yaml_json_equivalence": "not_performed"}


def _sign_off_with(**changes) -> tuple[dict, dict[str, bytes]]:
    """The orchard fixture with its sign-off entry edited and re-sealed under its new hash."""

    source = copy.deepcopy(_orchard_declaration()["source"])
    files = _orchard_files()
    entry = json.loads(files.pop(source["sign_off_copy_relative_path"]))
    entry.update(changes)
    data = canonical_json(entry)
    sha = hashlib.sha256(data).hexdigest()
    source["sign_off"]["entry_sha256"] = sha
    source["sign_off_copy_relative_path"] = f"run_plan/sign_off/{sha}.json"
    files[source["sign_off_copy_relative_path"]] = data
    return source, files


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"by": "someone"}, "entry by differs"),
        ({"entry_id": "signoff-2"}, "entry entry_id differs"),
        ({"at_utc": "2026-11-18T14:56:00Z"}, "entry at_utc differs"),
        ({"plan_sha256": "0" * 64}, "entry plan_sha256 differs"),
        ({"file_sha256": "0" * 64}, "entry file_sha256 differs"),
        ({"covers_sessions": ["S2"]}, "entry covers_sessions"),
        ({"covers_sessions": "S1"}, "entry covers_sessions"),
    ],
)
def test_a_sign_off_entry_that_does_not_bind_the_plan_refuses(changes, match):
    source, files = _sign_off_with(**changes)
    with pytest.raises(StartDeclarationError, match=f"^orange_start_declaration:sign_off: {match}"):
        declaration.check_run_plan(source, files.get)


@pytest.mark.parametrize("key", ["plan_copy_relative_path", "plan_json_copy_relative_path",
                                 "sign_off_copy_relative_path"])
@pytest.mark.parametrize("damage", ["missing", "tampered"])
def test_a_missing_or_tampered_plan_copy_refuses(key, damage):
    source = _orchard_declaration()["source"]
    files = _orchard_files()
    if damage == "missing":
        del files[source[key]]
    else:
        files[source[key]] += b" "
    with pytest.raises(StartDeclarationError, match="^orange_start_declaration:run_plan_copy:"):
        declaration.check_run_plan(source, files.get)


@pytest.mark.parametrize("key", ["plan_json_copy_relative_path", "sign_off_copy_relative_path"])
def test_a_plan_copy_that_is_not_canonical_json_refuses(key):
    source = copy.deepcopy(_orchard_declaration()["source"])
    files = _orchard_files()
    data = json.dumps(json.loads(files.pop(source[key])), indent=2).encode()
    sha = hashlib.sha256(data).hexdigest()
    if key == "plan_json_copy_relative_path":
        source["plan_sha256"] = sha
        source[key] = f"run_plan/{sha}.json"
    else:
        source["sign_off"]["entry_sha256"] = sha
        source[key] = f"run_plan/sign_off/{sha}.json"
    files[source[key]] = data
    with pytest.raises(StartDeclarationError, match="not canonical JSON"):
        declaration.check_run_plan(source, files.get)


def _spec(sent, **extra) -> bytes:
    return json.dumps({"experiment_id": "x", "fixed": {"declaration": sent, **extra}}, indent=2).encode()


def test_the_spec_declaration_must_equal_the_confirmations():
    sent = _orchard_declaration()
    # Orange re-serialized the confirmation's copy (sorted keys); equal as JSON passes,
    # and the spec's own float parameters are not the declaration's tokens.
    declaration.check_spec_declaration(_spec(sent, gain=2.5), json.loads(canonical_json(sent)))
    other = copy.deepcopy(sent)
    other["declared_by"] = "someone"
    with pytest.raises(StartDeclarationError, match="^orange_start_declaration:spec_declaration:"):
        declaration.check_spec_declaration(_spec(other), sent)
    with pytest.raises(StartDeclarationError, match="^orange_start_declaration:spec_declaration:"):
        declaration.check_spec_declaration(json.dumps({"fixed": {}}).encode(), sent)
    floated = copy.deepcopy(sent)
    floated["source"]["plan_revision"] = 5.0
    with pytest.raises(StartDeclarationError, match="^orange_start_declaration:T1:"):
        declaration.check_spec_declaration(_spec(floated), sent)


def test_check_declared_files_records_the_source_verbatim():
    confirmation = json.loads((FIXTURES / "valid" / "confirmation_run_spec_orchard_v2.json").read_bytes())
    requester = confirmation["requester"]
    files = {**_orchard_files(), requester["spec_copy_relative_path"]: None}
    spec = _spec(confirmation["declaration"])
    # The fixture names a real spec sha; stand in a spec with that declaration under its own hash.
    sha = hashlib.sha256(spec).hexdigest()
    requester["spec_sha256"] = sha
    requester["spec_copy_relative_path"] = f"run_spec/{sha}.json"
    files[requester["spec_copy_relative_path"]] = spec
    record = declaration.check_declared_files(confirmation, files.get)
    assert record["orange_declaration_schema_version"] == 2
    assert record["orange_declaration_source"] == confirmation["declaration"]["source"]
    assert record["orange_run_plan_check"]["yaml_json_equivalence"] == "not_performed"
    del files[requester["spec_copy_relative_path"]]
    with pytest.raises(StartDeclarationError, match="^orange_start_declaration:spec_copy:"):
        declaration.check_declared_files(confirmation, files.get)


def test_a_panel_confirmation_names_no_files():
    panel = json.loads((FIXTURES / "valid" / "confirmation_gui_operator.json").read_bytes())
    assert declaration.check_declared_files(panel, {}.get) == {}
