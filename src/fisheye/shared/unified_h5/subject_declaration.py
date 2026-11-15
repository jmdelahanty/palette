"""Citrus subject declaration v2 in ``/metadata/session`` (agent-contracts #126).

Citrus records each arena's start declaration (``citrus.subject_declaration``
v2, ``citrus-orange/subject_declaration_v2_2026-11-15.md`` section 5) as up to
three scalar UTF-8 string datasets, each stored as canonical JSON with
``checksum_algorithm`` "sha256" and ``checksum_sha256`` "sha256:<hex>" over
the stored bytes:

- ``subject_declaration_json``: the declaration Citrus accepted;
- ``run_plan_reference_json``: its ``run_plan``, written only for source
  ``orchard_run_plan``, with ``run_plan_reference_sha256`` equal to its checksum;
- ``subject_declaration_sender_json``: the ``declare_subjects`` peer
  (``SO_PEERCRED``), written only beside the run-plan reference.

The internal manifest binds the bytes (every dataset is a dependency); the
checksum attributes are checked here because the manifest does not cover
attributes. The declaration is provenance, not identity: the sealed snapshot
v4 stays the subject authority, and it must agree with the declaration (the
contract's pairing rule). A file without a declaration is unaffected.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from importlib.resources import files
import json
import re
from typing import Any

import h5py

from .common import canonical_json, digest, parse_json, require
from .hdf5_types import dataset_bytes, type_descriptor

SESSION = "/metadata/session"
DECLARATION = SESSION + "/subject_declaration_json"
RUN_PLAN_REFERENCE = SESSION + "/run_plan_reference_json"
SENDER = SESSION + "/subject_declaration_sender_json"
SNAPSHOT = "/metadata/zebrobot/snapshot_json"
# agent-contracts #126 (merged df49b8d), citrus-orange/; unchanged on main 1d1945a.
SCHEMA = (
    "citrus_subject_declaration_v2.schema.json",
    "6db74024af547d8a84077ee8a6af8063e89b35ab77cfa4f3acbe894124378ce2",
)
ORCHARD = "orchard_run_plan"
SENDER_KEYS = frozenset({"uid", "gid", "pid", "user", "request_id"})
CHECKSUM = re.compile(r"sha256:[0-9a-f]{64}\Z")


@lru_cache(maxsize=None)
def _schema() -> dict[str, Any]:
    name, sha256 = SCHEMA
    data = files("fisheye.shared").joinpath("contracts").joinpath(name).read_bytes()
    require(hashlib.sha256(data).hexdigest() == sha256, "packaged_contract_drift:" + name)
    return json.loads(data)


def invariant_errors(declaration: dict[str, Any]) -> list[str]:
    """Rules D1-D3 (port of ``invariant_errors`` in agent-contracts
    ``citrus-orange/test_subject_declaration_v2.py``, the reference)."""

    errors = []
    members = declaration["member_fish_ids"]
    if declaration["subject_type"] == "dish_group":
        if members != sorted(members):
            errors.append("D1_members_not_sorted")
        if members and len(members) != declaration["subject_count"]:
            errors.append("D2_members_not_subject_count")
        if declaration["source"] == ORCHARD and not members:
            errors.append("D3_run_plan_group_without_members")
    return errors


def _text(value: Any) -> Any:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _sealed_json(h5, path: str, *, also: str | None = None) -> dict[str, Any]:
    """One canonical JSON dataset and its checksum attributes."""

    dataset = h5[path]
    require(isinstance(dataset, h5py.Dataset), f"subject_declaration_not_dataset:{path}")
    spec = type_descriptor(dataset.id.get_type())
    require(
        dataset.shape == ()
        and spec.get("class") == "string"
        and spec.get("variable_length")
        and spec.get("character_set") == "utf8",
        f"subject_declaration_not_scalar_utf8:{path}",
    )
    data = dataset_bytes(dataset)
    attrs = dataset.attrs
    require(_text(attrs.get("checksum_algorithm")) == "sha256", f"subject_declaration_checksum_algorithm:{path}")
    names = ("checksum_sha256",) + ((also,) if also else ())
    for name in names:
        value = _text(attrs.get(name))
        require(
            type(value) is str and CHECKSUM.match(value) is not None,
            f"subject_declaration_checksum_form:{path}#{name}",
        )
        require(value == digest(data), f"subject_declaration_checksum_mismatch:{path}#{name}")
    return parse_json(data, label=path, canonical=True)


def _check_sender(sender: dict[str, Any]) -> None:
    require(set(sender) == SENDER_KEYS, "subject_declaration_sender_keys")
    for name in ("uid", "gid", "pid"):
        require(type(sender[name]) is int and sender[name] >= 0, f"subject_declaration_sender_value:{name}")
    for name in ("user", "request_id"):
        require(type(sender[name]) is str and sender[name] != "", f"subject_declaration_sender_value:{name}")


def _check_pairing(h5, declaration: dict[str, Any]) -> None:
    """The sealed snapshot v4 must describe the subject the declaration started."""

    require(SNAPSHOT in h5, "subject_declaration_without_snapshot")
    snapshot = parse_json(dataset_bytes(h5[SNAPSHOT]), label=SNAPSHOT)
    require(snapshot.get("schema_version") == 4, "subject_declaration_requires_snapshot_v4")
    require(snapshot.get("dish_id") == declaration["dish_id"], "subject_declaration_pairing:dish_id")
    if declaration["dish_id"] is None:
        return
    subject = snapshot.get("subject")
    require(isinstance(subject, dict), "subject_declaration_pairing:subject")
    for name in ("subject_type", "subject_count"):
        require(subject.get(name) == declaration[name], f"subject_declaration_pairing:{name}")
    if declaration["subject_type"] == "individual":
        require(subject.get("subject_id") == declaration["fish_id"], "subject_declaration_pairing:fish_id")
        if declaration["unit_id"] is not None:
            fish = snapshot.get("fish_reference")
            require(
                isinstance(fish, dict) and fish.get("mzb_fish_unit_id") == declaration["unit_id"],
                "subject_declaration_pairing:unit_id",
            )
    members = snapshot.get("members") or []
    require(isinstance(members, list), "subject_declaration_pairing:members")
    served = {member.get("mzb_fish_id") for member in members if isinstance(member, dict)}
    require(
        len(served) == len(members) and served == set(declaration["member_fish_ids"]),
        "subject_declaration_pairing:member_fish_ids",
    )


def validate_subject_declaration(h5) -> dict[str, Any] | None:
    """Check the session's subject declaration v2; return it, or None when absent."""

    if DECLARATION not in h5:
        require(
            RUN_PLAN_REFERENCE not in h5 and SENDER not in h5,
            "subject_declaration_companion_without_declaration",
        )
        return None
    declaration = _sealed_json(h5, DECLARATION)
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(_schema(), format_checker=Draft202012Validator.FORMAT_CHECKER)
    errors = sorted(error.message for error in validator.iter_errors(declaration))
    require(not errors, f"subject_declaration_schema:{errors[0] if errors else ''}")
    invariants = invariant_errors(declaration)
    require(not invariants, f"subject_declaration_invariant:{invariants[0] if invariants else ''}")
    orchard = declaration["source"] == ORCHARD
    require((RUN_PLAN_REFERENCE in h5) == orchard, "subject_declaration_run_plan_reference_presence")
    require((SENDER in h5) == orchard, "subject_declaration_sender_presence")
    if orchard:
        reference = _sealed_json(h5, RUN_PLAN_REFERENCE, also="run_plan_reference_sha256")
        require(
            canonical_json(reference) == canonical_json(declaration["run_plan"]),
            "subject_declaration_run_plan_reference_mismatch",
        )
        _check_sender(_sealed_json(h5, SENDER))
    _check_pairing(h5, declaration)
    return declaration


__all__ = [
    "DECLARATION",
    "RUN_PLAN_REFERENCE",
    "SCHEMA",
    "SENDER",
    "invariant_errors",
    "validate_subject_declaration",
]
