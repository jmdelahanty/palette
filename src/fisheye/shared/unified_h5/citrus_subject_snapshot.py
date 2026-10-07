"""Citrus subject identity v3: the sealed snapshot is the authority.

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
never-emitted contract v2. Any version other than 2 or 3 is refused.
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
SCHEMA_FILE = "citrus_zebrobot_snapshot_v3.schema.json"
# agent-contracts PR 52 at e16f6251 (contract md sha256 b6c80313...).
SCHEMA_SHA256 = "6d84dcfd97fca45cdf40363124b208f3547e6333c77bb8ed78fb5c0d2def6f26"
# Keys only the never-emitted contract v2 carries: such a v2 document is refused.
CONTRACT_ONLY_KEYS = frozenset(
    {"api_schema_version", "subject_lookup_status", "fish_reference", "errors"}
)
# The contract declares these two as int64 attrs; every other attr is a string.
INTEGER_ATTRS = frozenset({"dish_revision", "mzb_fish_revision"})


def _dish(snapshot):
    return snapshot.get("dish") or {}


# attribute name -> its value in the snapshot (None when the attr must be absent).
PAIRED_FIELDS = {
    "subject_id": lambda s: s["subject"]["subject_id"],
    "subject_type": lambda s: s["subject"]["subject_type"],
    "subject_count": lambda s: s["subject"]["subject_count"],
    "dish_id": lambda s: s["dish_id"],
    "subject_lookup_status": lambda s: s["subject_lookup_status"],
    "subject_lookup_reason": lambda s: s["subject_lookup_reason"],
    "dish_uuid": lambda s: _dish(s).get("dish_uuid"),
    "dish_revision": lambda s: _dish(s).get("dish_revision"),
    "dish_updated_at": lambda s: _dish(s).get("dish_updated_at"),
    "fish_reference_status": lambda s: s["fish_reference"]["status"],
    "fish_reference_reason": lambda s: s["fish_reference"]["reason"],
    "mzb_fish_id": lambda s: s["fish_reference"]["mzb_fish_id"],
    "mzb_fish_revision": lambda s: s["fish_reference"]["mzb_fish_revision"],
    "mzb_fish_updated_at": lambda s: s["fish_reference"]["mzb_fish_updated_at"],
    # Sealed dish biology; Citrus writes each attr verbatim from the same value.
    "cross_id": lambda s: _dish(s).get("cross_id"),
    "genotype": lambda s: _dish(s).get("genotype"),
    "species": lambda s: _dish(s).get("species"),
    "sex": lambda s: _dish(s).get("sex"),
    "date_of_fertilization": lambda s: _dish(s).get("dof"),
}
# Paired only when the snapshot seals a cross: Citrus writes ``line_strain``
# for every collected dish, falling back to the genotype when no cross is served.
CROSS_PAIRED_FIELDS = {
    "line_strain": lambda s: s["cross"]["line_strain"],
}


@lru_cache(maxsize=1)
def _schema() -> dict[str, Any]:
    data = files("fisheye.shared").joinpath("contracts").joinpath(SCHEMA_FILE).read_bytes()
    require(hashlib.sha256(data).hexdigest() == SCHEMA_SHA256, "packaged_contract_drift:" + SCHEMA_FILE)
    return json.loads(data)


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
        type(version) is int and version == CONTRACT_VERSION,
        f"citrus_snapshot_schema_version_unsupported:{version!r}",
    )
    from jsonschema import Draft202012Validator

    errors = sorted(e.message for e in Draft202012Validator(_schema()).iter_errors(snapshot))
    require(not errors, f"citrus_snapshot_schema:{errors[0] if errors else ''}")
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
        "citrus_snapshot_schema_version": CONTRACT_VERSION,
        "citrus_snapshot_schema_sha256": SCHEMA_SHA256,
        "zebrobot_service_commit": service["service_commit"],
        "zebrobot_service_commit_dirty": service["service_commit_dirty"],
        "zebrobot_consumer_schema_sha256": served,
        "zebrobot_consumer_schema_matches_pin": (
            None if served is None else served == MZB_PIN["consumer_openapi_sha256"]
        ),
    }


__all__ = [
    "CONTRACT_VERSION",
    "SNAPSHOT_PATH",
    "UnifiedH5ContractError",
    "admit_subject_snapshot",
    "subject_attributes",
]
