"""Orange's recording start declaration in ``subject_confirmation.json`` v2.

agent-contracts #138 (merged ``d568b42f``), ``orange-palette/recording_start_declaration_v2/``.
Orange writes ``subject_confirmation.json`` at the recording root before the
start snapshot is sealed. A v2 file names who confirmed the subjects
(``confirmed_by``), the declaration Orange accepted (``null`` for a panel
confirmation) and its requester. A declaration is v1 or v2; v2 adds a required
``source`` (``operator``, ``orange_run_spec`` or ``orchard_run_plan``) and
stated blanks (``{"blank": true}`` camera entries).

Palette checks the pinned schemas plus the contract's rules the schemas cannot
express: T1 (integer tokens only), R1-R3 and R5 on the declaration, C1-C3 on
the confirmation. Files the declaration names must be sealed beside it: the
``run_spec/`` copy, whose ``fixed.declaration`` must equal the declaration, and
for an orchard plan its three ``run_plan/`` copies, each hashing to its name.
The sign-off entry must bind the same plan and session. ``source`` is recorded
verbatim as a claim; "signed" is a named approval, recorded and never a gate.
The plan YAML is hash-checked only; its YAML-to-JSON equivalence is not
recomputed (the contract makes it optional) and the record says so.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from importlib.resources import files
import json
from typing import Any, Callable, Mapping

from fisheye.shared.unified_h5.common import canonical_json

CONFIRMATION_SCHEMA_ID = "orange.recording_subject_confirmation"
DECLARATION_SCHEMA_ID = "orange.recording_start_declaration"
ORCHARD = "orchard_run_plan"
# agent-contracts PR 138, merged d568b42f, orange-palette/recording_start_declaration_v2/schemas/.
SCHEMA_FILES = {
    "declaration_v1": (
        "orange_recording_start_declaration_v1.schema.json",
        "ce22a927bb1359723198319a50c1e689e993d2a8507e862c49e516c92a868e87",
    ),
    "declaration_v2": (
        "orange_recording_start_declaration_v2.schema.json",
        "ee1250699018edbf4243cec030d9192aa31f2fd77153f2c788264116d44551f2",
    ),
    "confirmation_v2": (
        "orange_recording_subject_confirmation_v2.schema.json",
        "f65f6991b159ae4dc31f44a5924745c07e5f313d08e8669e5726511aa100d6dd",
    ),
}
_TEXT_LIMIT_BYTES = 256

# source_path -> the sealed file's bytes (checked against the sealed
# inventory), or None when the file is not sealed.
ReadSealed = Callable[[str], "bytes | None"]


class StartDeclarationError(ValueError):
    """The confirmation or its declaration breaks the pinned contract."""


def _refuse(rule: str, detail: str) -> StartDeclarationError:
    return StartDeclarationError(f"orange_start_declaration:{rule}: {detail}")


def _pinned(name: str) -> dict[str, Any]:
    file_name, sha256 = SCHEMA_FILES[name]
    data = files("fisheye.shared").joinpath("contracts").joinpath(file_name).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise StartDeclarationError("packaged_contract_drift:" + file_name)
    return json.loads(data)


@lru_cache(maxsize=1)
def _validator():
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    # The confirmation refers to both declaration schemas by $id; only the
    # pinned copies resolve, nothing is fetched.
    declarations = [_pinned("declaration_v1"), _pinned("declaration_v2")]
    registry = Registry().with_resources(
        (schema["$id"], Resource.from_contents(schema)) for schema in declarations
    )
    return Draft202012Validator(_pinned("confirmation_v2"), registry=registry)


def _no_float(text: str) -> Any:
    raise _refuse("T1", f"float token {text}")


def _no_constant(text: str) -> Any:
    raise _refuse("T1", f"non-JSON constant {text}")


def parse_document(data: bytes, *, label: str) -> Any:
    """T1: no field of these documents is a float, and Orange refuses 5.0 where
    it reads an integer while JSON Schema accepts it, so any float token refuses."""

    try:
        return json.loads(data, parse_float=_no_float, parse_constant=_no_constant)
    except StartDeclarationError:
        raise
    except ValueError as exc:
        raise _refuse("json", f"{label}: {exc}") from exc


def _has_float(value: Any) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, Mapping):
        return any(_has_float(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_float(item) for item in value)
    return False


def _utf8_len(value: str) -> int:
    return len(value.encode("utf-8"))


def resolve_cameras(config: Mapping[str, Any], cameras: list[str]) -> dict[str, dict[str, Any]]:
    """Orange ``SubjectReferencesConfig::Resolve``: the camera value, else the
    default, else null; a stated blank replaces the default with nulls."""

    default = config.get("default", {})
    resolved = {}
    for camera in cameras:
        entry = config.get("cameras", {}).get(camera)
        if entry and entry.get("blank"):
            resolved[camera] = {"dish_id": None, "subject_count": None}
            continue
        merged = {**default, **(entry or {})}
        resolved[camera] = {"dish_id": merged.get("dish_id"), "subject_count": merged.get("subject_count")}
    return resolved


def declaration_errors(declaration: Mapping[str, Any]) -> list[str]:
    """R1-R3 (port of ``declaration_errors`` in the contract's
    ``test_recording_start_declaration.py``, the reference)."""

    errors = []
    source = declaration.get("source") or {}
    orchard = source.get("kind") == ORCHARD
    if orchard:
        if source["sign_off"]["by"] != declaration["declared_by"]:
            errors.append("R1 sign_off.by differs from declared_by")
        expected = {
            "plan_copy_relative_path": f"run_plan/{source['file_sha256']}.yaml",
            "plan_json_copy_relative_path": f"run_plan/{source['plan_sha256']}.json",
            "sign_off_copy_relative_path": f"run_plan/sign_off/{source['sign_off']['entry_sha256']}.json",
        }
        errors += [f"R2 {key} is not {value}" for key, value in expected.items() if source[key] != value]
    references = declaration["subject_references"]
    entries = [references.get("default", {}), *references.get("cameras", {}).values()]
    if any(_utf8_len(entry.get("dish_id", "")) > _TEXT_LIMIT_BYTES for entry in entries):
        errors.append("R3 dish_id exceeds 256 UTF-8 bytes")
    if orchard:
        texts = {
            "plan_id": source["plan_id"],
            "plan_session_id": source["plan_session_id"],
            "sign_off.entry_id": source["sign_off"]["entry_id"],
        }
        errors += [
            f"R3 {key} is not 1..256 UTF-8 bytes"
            for key, value in texts.items()
            if not 1 <= _utf8_len(value) <= _TEXT_LIMIT_BYTES
        ]
    return errors


def confirmation_errors(confirmation: Mapping[str, Any]) -> list[str]:
    """C1-C3 (the contract's ``confirmation_errors``) plus R5, which Orange
    enforces at start and Palette rechecks from the recording's cameras."""

    errors = []
    declaration = confirmation["declaration"]
    if declaration is not None:
        errors += declaration_errors(declaration)
        if confirmation["config"] != declaration["subject_references"]:
            errors.append("C1 config differs from declaration.subject_references")
        if (declaration.get("source") or {}).get("kind") == ORCHARD:
            named = set(declaration["subject_references"].get("cameras", {}))
            missing = sorted(set(confirmation["cameras"]) - named)
            if missing:
                errors.append(f"R5 orchard_run_plan does not name cameras {missing}")
    if confirmation["cameras"] != resolve_cameras(confirmation["config"], list(confirmation["cameras"])):
        errors.append("C2 cameras are not config resolved per recording camera")
    requester = confirmation["requester"]
    if (requester["transport"] == "run_spec"
            and requester["spec_copy_relative_path"] != f"run_spec/{requester['spec_sha256']}.json"):
        errors.append("C3 spec_copy_relative_path is not run_spec/<spec_sha256>.json")
    return errors


def validate_confirmation_v2(confirmation: Any) -> dict[str, Any]:
    """The pinned v2 schema, then the cross-field rules; refuse on the first error."""

    from jsonschema.exceptions import best_match

    error = best_match(_validator().iter_errors(confirmation))
    if error is not None:
        where = "/".join(str(part) for part in error.absolute_path) or "(root)"
        raise _refuse("schema", f"{where}: {error.message}")
    errors = confirmation_errors(confirmation)
    if errors:
        rule, _, detail = errors[0].partition(" ")
        raise _refuse(rule, detail)
    return dict(confirmation)


def validate_declaration(declaration: Any) -> dict[str, Any]:
    """A bare declaration (v1 or v2): its pinned schema, then R1-R3."""

    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import best_match

    version = declaration.get("schema_version") if isinstance(declaration, Mapping) else None
    name = {1: "declaration_v1", 2: "declaration_v2"}.get(version) if type(version) is int else None
    if name is None:
        raise _refuse("schema", f"unsupported declaration schema_version {version!r}")
    error = best_match(Draft202012Validator(_pinned(name)).iter_errors(declaration))
    if error is not None:
        where = "/".join(str(part) for part in error.absolute_path) or "(root)"
        raise _refuse("schema", f"{where}: {error.message}")
    errors = declaration_errors(declaration)
    if errors:
        rule, _, detail = errors[0].partition(" ")
        raise _refuse(rule, detail)
    return dict(declaration)


def _sealed_bytes(read_sealed: ReadSealed, path: str, sha256: str, *, rule: str) -> bytes:
    data = read_sealed(path)
    if data is None:
        raise _refuse(rule, f"{path} is not sealed with the recording")
    if hashlib.sha256(data).hexdigest() != sha256:
        raise _refuse(rule, f"{path} does not hash to {sha256}")
    return data


def _canonical_document(data: bytes, *, label: str, rule: str) -> dict[str, Any]:
    document = parse_document(data, label=label)
    if not isinstance(document, dict) or canonical_json(document) != data:
        raise _refuse(rule, f"{label} is not canonical JSON")
    return document


def check_spec_declaration(spec_bytes: bytes, declaration: Mapping[str, Any]) -> None:
    """The sealed run spec carries the declaration exactly as Orange received it;
    the confirmation's copy is re-serialized, so they must be equal as JSON."""

    # The spec's own parameters may be floats; T1 covers the declaration only.
    try:
        spec = json.loads(spec_bytes, parse_constant=_no_constant)
    except StartDeclarationError:
        raise
    except ValueError as exc:
        raise _refuse("json", f"run_spec: {exc}") from exc
    fixed = spec.get("fixed") if isinstance(spec, Mapping) else None
    sent = fixed.get("declaration") if isinstance(fixed, Mapping) else None
    if _has_float(sent):
        raise _refuse("T1", "float token in run_spec fixed.declaration")
    if sent is None or canonical_json(sent) != canonical_json(declaration):
        raise _refuse("spec_declaration", "run_spec fixed.declaration differs from the confirmation's declaration")


def check_run_plan(source: Mapping[str, Any], read_sealed: ReadSealed) -> dict[str, Any]:
    """An orchard plan's three sealed copies, and the sign-off entry's binding."""

    _sealed_bytes(read_sealed, source["plan_copy_relative_path"], source["file_sha256"], rule="run_plan_copy")
    plan_json = _sealed_bytes(
        read_sealed, source["plan_json_copy_relative_path"], source["plan_sha256"], rule="run_plan_copy"
    )
    _canonical_document(plan_json, label=source["plan_json_copy_relative_path"], rule="run_plan_copy")
    sign_off = source["sign_off"]
    entry = _canonical_document(
        _sealed_bytes(read_sealed, source["sign_off_copy_relative_path"], sign_off["entry_sha256"],
                      rule="run_plan_copy"),
        label=source["sign_off_copy_relative_path"],
        rule="run_plan_copy",
    )
    for key in ("entry_id", "by", "at_utc"):
        if entry.get(key) != sign_off[key]:
            raise _refuse("sign_off", f"entry {key} differs from source.sign_off")
    for key in ("plan_sha256", "file_sha256"):
        if entry.get(key) != source[key]:
            raise _refuse("sign_off", f"entry {key} differs from source")
    sessions = entry.get("covers_sessions")
    if not isinstance(sessions, list) or source["plan_session_id"] not in sessions:
        raise _refuse("sign_off", "entry covers_sessions does not include source.plan_session_id")
    return {
        "copies_verified": sorted(
            source[key] for key in
            ("plan_copy_relative_path", "plan_json_copy_relative_path", "sign_off_copy_relative_path")
        ),
        "yaml_json_equivalence": "not_performed",
    }


def check_declared_files(confirmation: Mapping[str, Any], read_sealed: ReadSealed) -> dict[str, Any]:
    """Check the files a v2 confirmation's declaration names; return what to record."""

    declaration = confirmation["declaration"]
    if declaration is None:
        return {}
    record: dict[str, Any] = {"orange_declaration_schema_version": declaration["schema_version"]}
    if confirmation["confirmed_by"] == "run_spec":
        requester = confirmation["requester"]
        spec = _sealed_bytes(
            read_sealed, requester["spec_copy_relative_path"], requester["spec_sha256"], rule="spec_copy"
        )
        check_spec_declaration(spec, declaration)
    source = declaration.get("source")
    if source is not None:
        record["orange_declaration_source"] = dict(source)
        if source["kind"] == ORCHARD:
            record["orange_run_plan_check"] = check_run_plan(source, read_sealed)
    return record


def stated_blank(confirmation: Mapping[str, Any], camera: str) -> bool:
    """Whether the declaration states this camera blank (``{"blank": true}``)."""

    declaration = confirmation.get("declaration")
    if declaration is None:
        return False
    entry = declaration["subject_references"].get("cameras", {}).get(camera)
    return bool(entry and entry.get("blank"))


__all__ = [
    "SCHEMA_FILES",
    "StartDeclarationError",
    "check_declared_files",
    "confirmation_errors",
    "declaration_errors",
    "parse_document",
    "resolve_cameras",
    "stated_blank",
    "validate_confirmation_v2",
    "validate_declaration",
]
