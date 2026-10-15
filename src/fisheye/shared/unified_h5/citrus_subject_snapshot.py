"""Citrus subject identity v3 and v4: the sealed snapshot is the authority.

Citrus's finalization digest covers H5 datasets but not attributes, so the
subject identity (agent-contracts PR 52, ``subject_identity_v3_2026-10-05.md``)
is sealed in ``/metadata/zebrobot/snapshot_json``. Palette validates that
document against the pinned v3 schema, treats it as authoritative and refuses
any ``/metadata/subject`` attribute that disagrees with it or that pairs
wrongly with a null (null in the snapshot means the attribute is absent).

Snapshot ``schema_version`` 2 is the pre-contract document earlier Citrus
writers emit; contract v2 reused that number and is never emitted (Jeremy,
2026-10-05, PR 52). A v2 document is recorded as pre-contract with no identity
rules, unless it carries a contract-only key, which is refused as the
never-emitted contract v2. Any version other than 2, 3 or 4 is refused.

v4 (agent-contracts ``subject_identity_v4_2026-10-14.md``): for an individual
``subject_id`` is the MetaZebrobot fish UUID and equals
``fish_reference.mzb_fish_id``; v3's ``fish_reference`` status/reason become a
registration record (mode, status, reason), and ``usage_link_request`` seals
the usage link Citrus sends at Start. ``mint``, ``registration_request``,
``members`` and ``usage_link_request`` are snapshot-only, and the
``fish_reference_status``/``_reason`` attrs are refused. The cross-field rules
R1-R6 are checked here because a JSON Schema cannot express them; the
``recording_ref`` check against the file's own identity is
``check_usage_link_recording_ref``.
Only fields Citrus pairs 1:1 with attrs are compared, including the sealed dish
biology (genotype, species, sex, cross, date of fertilization) and, when a cross
is sealed, its line strain; display copies such as ``fish_count``,
``days_post_fertilization`` or the rendered ``parents`` string are not. MetaZebrobot shapes and meaning are owned by MetaZebrobot; see
``fisheye.shared.zebrobot_subject_reference.MZB_PIN``.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from importlib.resources import files
import json
from typing import Any, Mapping

from fisheye.shared.zebrobot_subject_reference import MZB_PIN

from .common import UnifiedH5ContractError, require
from .metadata import string_attributes

SNAPSHOT_PATH = "/metadata/zebrobot/snapshot_json"
PRE_CONTRACT_VERSION = 2
CONTRACT_VERSION = 3
CONTRACT_V4_VERSION = 4
# version -> (packaged schema, sha256).
SCHEMAS = {
    # agent-contracts PR 52 at e16f6251, amended 2026-10-14 by PR 110 head 3b3f214 (merged f918d312)
    # (Citrus PR 43 533ce76; contract md sha256 6c23e38d...): three not_collected
    # fish reasons, which require the optional fish_reference.mzb_fish_unit_id, and
    # fish_list_unavailable's failed request in errors[]; schema_version 3.
    3: ("citrus_zebrobot_snapshot_v3.schema.json",
        "ff362fe03245819660f449b60b00a78cf7004d0db97073d41005179c588f068c"),
    # agent-contracts PR 103 (0b468c7), amended by #109 and #112 (merge d329774);
    # attr types stated by #114 (5d9b172, contract md sha256 5912d082...).
    4: ("citrus_zebrobot_snapshot_v4.schema.json",
        "b2f3f71067d116498df795c0de507ac72d42cc37ead9255e7a43acc83e2b5a31"),
}
# The v3 pin under its pre-v4 names (callers and witness tests use them).
SCHEMA_FILE, SCHEMA_SHA256 = SCHEMAS[CONTRACT_VERSION]
# Keys only the never-emitted contract v2 carries: such a v2 document is refused.
CONTRACT_ONLY_KEYS = frozenset(
    {"api_schema_version", "subject_lookup_status", "fish_reference", "errors"}
)
# The contract declares these as int64 attrs; every other attr is a string. v4
# adds the listed fish count (agent-contracts #114) and refuses a string copy.
INTEGER_ATTRS = frozenset({"dish_revision", "mzb_fish_revision", "mzb_dish_listed_fish_count_at_start"})


def _dish(snapshot):
    return snapshot.get("dish") or {}


def _fish(snapshot):
    return snapshot.get("fish_reference") or {}


# attribute name -> its value in the snapshot (None when the attr must be absent).
_COMMON_PAIRED_FIELDS = {
    "subject_id": lambda s: s["subject"]["subject_id"],
    "subject_type": lambda s: s["subject"]["subject_type"],
    "subject_count": lambda s: s["subject"]["subject_count"],
    "dish_id": lambda s: s["dish_id"],
    "subject_lookup_status": lambda s: s["subject_lookup_status"],
    "subject_lookup_reason": lambda s: s["subject_lookup_reason"],
    "dish_uuid": lambda s: _dish(s).get("dish_uuid"),
    "dish_revision": lambda s: _dish(s).get("dish_revision"),
    "dish_updated_at": lambda s: _dish(s).get("dish_updated_at"),
    "mzb_fish_id": lambda s: _fish(s).get("mzb_fish_id"),
    "mzb_fish_revision": lambda s: _fish(s).get("mzb_fish_revision"),
    "mzb_fish_updated_at": lambda s: _fish(s).get("mzb_fish_updated_at"),
    # Optional in v3 since the 2026-10-14 amendment (files before it carry no
    # key and no attr); required, nullable, in v4.
    "mzb_fish_unit_id": lambda s: _fish(s).get("mzb_fish_unit_id"),
    # Sealed dish biology; Citrus writes each attr verbatim from the same value.
    "cross_id": lambda s: _dish(s).get("cross_id"),
    "genotype": lambda s: _dish(s).get("genotype"),
    "species": lambda s: _dish(s).get("species"),
    "sex": lambda s: _dish(s).get("sex"),
    "date_of_fertilization": lambda s: _dish(s).get("dof"),
}
PAIRED_FIELDS = {
    **_COMMON_PAIRED_FIELDS,
    "fish_reference_status": lambda s: s["fish_reference"]["status"],
    "fish_reference_reason": lambda s: s["fish_reference"]["reason"],
}
PAIRED_FIELDS_V4 = {
    **_COMMON_PAIRED_FIELDS,
    "mzb_fish_subject_label": lambda s: _fish(s).get("mzb_fish_subject_label"),
    "mzb_dish_listed_fish_count_at_start": lambda s: _fish(s).get("mzb_dish_listed_fish_count_at_start"),
    "mzb_registration_mode": lambda s: _fish(s).get("mzb_registration_mode"),
    "mzb_registration_status": lambda s: _fish(s).get("mzb_registration_status"),
    "mzb_registration_reason": lambda s: _fish(s).get("mzb_registration_reason"),
}
# v3 attrs that a v4 file must not carry.
REMOVED_IN_V4_ATTRS = ("fish_reference_status", "fish_reference_reason")
# Paired only when the snapshot seals a cross: Citrus writes ``line_strain``
# for every collected dish, falling back to the genotype when no cross is served.
CROSS_PAIRED_FIELDS = {
    "line_strain": lambda s: s["cross"]["line_strain"],
}


@lru_cache(maxsize=None)
def _schema(version: int = CONTRACT_VERSION) -> dict[str, Any]:
    name, sha256 = SCHEMAS[version]
    data = files("fisheye.shared").joinpath("contracts").joinpath(name).read_bytes()
    require(hashlib.sha256(data).hexdigest() == sha256, "packaged_contract_drift:" + name)
    return json.loads(data)


def v4_invariant_errors(snapshot: Mapping[str, Any]) -> list[str]:
    """Cross-field rules R1-R6 of subject identity v4.

    Port of ``invariant_errors`` in agent-contracts
    ``citrus-palette/test_citrus_zebrobot_snapshot_v4.py`` (the reference).
    """

    errors = []
    subject, fish, members = snapshot["subject"], snapshot["fish_reference"], snapshot["members"]
    fish_path = f"/dishes/{snapshot['dish_id']}/fish"
    if fish is not None:
        if subject["subject_id"] != fish["mzb_fish_id"]:
            errors.append("R1_subject_id_differs_from_mzb_fish_id")
        request = fish["registration_request"]
        if request is not None:
            if request["body"]["fish_id"] != fish["mzb_fish_id"]:
                errors.append("R2_request_fish_id_differs_from_mzb_fish_id")
            if request["path"] != fish_path:
                errors.append("R2_request_path_not_declared_dish")
        if (fish["mzb_registration_mode"] != "existing"
                and fish["mzb_registration_status"] != "registered"
                and not any(e["method"] == "POST" and e["endpoint"] == fish_path
                            for e in snapshot["errors"])):
            errors.append("R3_unregistered_created_fish_without_post_error")
    if members is not None:
        ids = [member["mzb_fish_id"] for member in members]
        if len(ids) != len(set(ids)):
            errors.append("R4_duplicate_member")
        if len(ids) > subject["subject_count"]:
            errors.append("R4_members_exceed_subject_count")
    if snapshot["dish"] is not None and snapshot["dish"]["dish_id"] != snapshot["dish_id"]:
        errors.append("R5_dish_id_differs")
    link = snapshot["usage_link_request"]
    if link is not None:
        body = link["body"]
        if body["dish_id"] != snapshot["dish_id"] or body["subject_count"] != subject["subject_count"]:
            errors.append("R6_usage_link_dish_or_count_differs")
        expected = ((fish["mzb_fish_id"], fish["mzb_fish_revision"], fish["mzb_fish_unit_id"])
                    if fish is not None else (None, None, None))
        if (body["fish_id"], body["fish_revision"], body["unit_id"]) != expected:
            errors.append("R6_usage_link_fish_differs")
    return errors


def check_usage_link_recording_ref(
    snapshot: Mapping[str, Any], *, observation_context_id: str | None, session_uuid: str | None
) -> None:
    """A v4 usage link names this file: its bound context, else its Citrus session.

    ``observation_context_id`` comes from the file's digest-covered recording
    association claims (None for an unbound session); ``session_uuid`` from
    ``/metadata/session``.
    """

    link = snapshot.get("usage_link_request") if snapshot.get("schema_version") == 4 else None
    if link is None:
        return
    body = link["body"]
    if observation_context_id is not None:
        expected = ("observation_context", observation_context_id)
    else:
        expected = ("citrus_session", session_uuid)
    require(
        expected[1] is not None and (body["ref_kind"], body["recording_ref"]) == expected,
        f"citrus_snapshot_usage_link_recording_ref:{body['ref_kind']}",
    )


def subject_attributes(descriptors: Mapping[str, Any]) -> dict[str, Any]:
    """Decode ``/metadata/subject``: scalar strings, plus the two int64 revisions."""

    integers = {}
    strings = {}
    for name, descriptor in descriptors.items():
        spec = descriptor.get("type", {})
        if name in INTEGER_ATTRS and spec.get("class") == "integer":
            require(
                descriptor.get("shape") == [] and spec.get("signed") and spec.get("size_bytes") == 8,
                f"attribute_not_scalar_int64:{name}",
            )
            integers[name] = int.from_bytes(
                bytes.fromhex(descriptor["payload_hex"]), "little", signed=True
            )
        else:
            strings[name] = descriptor
    return {**string_attributes(strings), **integers}


def _same(attr: Any, value: Any) -> bool:
    if isinstance(value, bool) or isinstance(attr, bool):
        return False
    if isinstance(value, int):
        return str(attr).strip() == str(value)
    return attr == value


def _check_fish_list_unavailable_error(snapshot: Mapping[str, Any]) -> None:
    """The failed request names the snapshot's own dish (the schema cannot say so).

    The schema already requires an errors[] entry whose endpoint is
    ``/dishes/<id>/fish`` (dish served) or ``/dishes/<id>/(citrus-snapshot|fish)``.
    """

    if snapshot["fish_reference"]["reason"] != "fish_list_unavailable":
        return
    dish_id = snapshot["dish_id"]
    suffixes = ("fish",) if snapshot["subject_lookup_status"] == "collected" else ("citrus-snapshot", "fish")
    wanted = {f"/dishes/{dish_id}/{suffix}" for suffix in suffixes} if isinstance(dish_id, str) else set()
    endpoints = {
        error.get("endpoint") for error in snapshot["errors"] if isinstance(error, Mapping)
    }
    require(
        bool(wanted & endpoints),
        "citrus_snapshot_fish_list_unavailable_error_not_for_dish:" + ",".join(sorted(wanted)),
    )


def admit_subject_snapshot(
    snapshot: Mapping[str, Any] | None, attributes: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the sealed snapshot against the subject attrs; return provenance."""

    if snapshot is None:
        return {"citrus_snapshot_status": "absent"}
    require(isinstance(snapshot, Mapping), "citrus_snapshot_not_object")
    version = snapshot.get("schema_version")
    if type(version) is int and version == PRE_CONTRACT_VERSION:
        require(
            not CONTRACT_ONLY_KEYS & set(snapshot), "citrus_snapshot_contract_v2_never_emitted"
        )
        return {"citrus_snapshot_status": "pre_contract", "citrus_snapshot_schema_version": 2}
    require(
        type(version) is int and version in SCHEMAS,
        f"citrus_snapshot_schema_version_unsupported:{version!r}",
    )
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(
        _schema(version), format_checker=Draft202012Validator.FORMAT_CHECKER
    ) if version == CONTRACT_V4_VERSION else Draft202012Validator(_schema(version))
    errors = sorted(e.message for e in validator.iter_errors(snapshot))
    require(not errors, f"citrus_snapshot_schema:{errors[0] if errors else ''}")
    if version == CONTRACT_V4_VERSION:
        invariants = v4_invariant_errors(snapshot)
        require(not invariants, f"citrus_snapshot_invariant:{invariants[0] if invariants else ''}")
        for name in REMOVED_IN_V4_ATTRS:
            require(name not in attributes, f"citrus_subject_attr_removed_in_v4:{name}")
        for name in INTEGER_ATTRS & set(attributes):
            require(type(attributes[name]) is int, f"citrus_subject_attr_not_int64:{name}")
        paired = dict(PAIRED_FIELDS_V4)
    else:
        _check_fish_list_unavailable_error(snapshot)
        paired = dict(PAIRED_FIELDS)
    if snapshot["cross"] is not None:
        paired.update(CROSS_PAIRED_FIELDS)
    for name, field in paired.items():
        value = field(snapshot)
        if value is None:
            require(name not in attributes, f"citrus_subject_attr_without_snapshot_value:{name}")
        else:
            require(name in attributes, f"citrus_subject_snapshot_value_without_attr:{name}")
            require(_same(attributes[name], value), f"citrus_subject_attr_mismatch:{name}")
    service = snapshot["zebrobot_service"]
    served = service["consumer_schema_sha256"]
    return {
        "citrus_snapshot_status": "admitted",
        "citrus_snapshot_schema_version": version,
        "citrus_snapshot_schema_sha256": SCHEMAS[version][1],
        "zebrobot_service_commit": service["service_commit"],
        "zebrobot_service_commit_dirty": service["service_commit_dirty"],
        "zebrobot_consumer_schema_sha256": served,
        "zebrobot_consumer_schema_matches_pin": (
            None if served is None else served == MZB_PIN["consumer_openapi_sha256"]
        ),
    }


__all__ = [
    "CONTRACT_V4_VERSION",
    "CONTRACT_VERSION",
    "SCHEMAS",
    "SCHEMA_SHA256",
    "SNAPSHOT_PATH",
    "UnifiedH5ContractError",
    "admit_subject_snapshot",
    "check_usage_link_recording_ref",
    "subject_attributes",
    "v4_invariant_errors",
]
