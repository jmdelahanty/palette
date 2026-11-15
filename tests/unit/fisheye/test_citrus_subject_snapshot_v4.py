"""Citrus subject identity v4 (agent-contracts PR 103, final d329774): the contract's own fixtures."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from fisheye.shared.subject_fields import from_h5_attributes
from fisheye.shared.unified_h5 import UnifiedH5ContractError
from fisheye.shared.unified_h5.citrus_subject_snapshot import (
    admit_subject_snapshot,
    check_usage_link_recording_ref,
)

# Copied byte-for-byte from agent-contracts d329774 citrus-palette/fixtures/subject_v4/.
FIXTURES = Path(__file__).resolve().parents[2] / "contracts" / "citrus_subject_v4"
ACCEPTED = sorted((FIXTURES / "accepted").glob("*.json"))
REFUSED_KIND = json.loads((FIXTURES / "refused" / "expected.json").read_text())
REFUSED = sorted(p for p in (FIXTURES / "refused").glob("*.json") if p.name != "expected.json")


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def project_attrs(doc: dict) -> dict:
    """``/metadata/subject`` as a v4 writer projects it (contract section "attrs")."""

    dish = doc.get("dish") or {}
    fish = doc.get("fish_reference") or {}
    subject = doc.get("subject") or {}
    values = {
        "subject_id": subject.get("subject_id"),
        "subject_type": subject.get("subject_type"),
        "subject_count": subject.get("subject_count"),
        "dish_id": doc.get("dish_id"),
        "subject_lookup_status": doc.get("subject_lookup_status"),
        "subject_lookup_reason": doc.get("subject_lookup_reason"),
        "dish_uuid": dish.get("dish_uuid"),
        "dish_revision": dish.get("dish_revision"),
        "dish_updated_at": dish.get("dish_updated_at"),
        "cross_id": dish.get("cross_id"),
        "genotype": dish.get("genotype"),
        "species": dish.get("species"),
        "sex": dish.get("sex"),
        "date_of_fertilization": dish.get("dof"),
        "line_strain": (doc.get("cross") or {}).get("line_strain"),
        **{name: fish.get(name) for name in (
            "mzb_fish_id", "mzb_fish_revision", "mzb_fish_updated_at", "mzb_fish_subject_label",
            "mzb_fish_unit_id", "mzb_dish_listed_fish_count_at_start",
            "mzb_registration_mode", "mzb_registration_status", "mzb_registration_reason")},
    }
    attrs = {k: v for k, v in values.items() if v is not None}
    if "subject_count" in attrs:
        attrs["subject_count"] = str(attrs["subject_count"])
    return attrs


def test_fixture_sets_match_the_contract():
    assert len(ACCEPTED) == 10
    assert len(REFUSED) == 26
    assert sorted(p.stem for p in REFUSED) == sorted(REFUSED_KIND)


@pytest.mark.parametrize("path", ACCEPTED, ids=lambda p: p.stem)
def test_accepted_fixtures_are_admitted(path):
    doc = _load(path)
    admitted = admit_subject_snapshot(doc, project_attrs(doc))
    assert admitted["citrus_snapshot_status"] == "admitted"
    assert admitted["citrus_snapshot_schema_version"] == 4
    assert admitted["citrus_snapshot_schema_sha256"].startswith("b2f3f710")


@pytest.mark.parametrize("path", REFUSED, ids=lambda p: p.stem)
def test_refused_fixtures_are_refused_for_the_contract_reason(path):
    doc = _load(path)
    kind = "citrus_snapshot_schema" if REFUSED_KIND[path.stem] == "schema" else "citrus_snapshot_invariant"
    if path.stem == "v3_document":
        kind = "citrus_snapshot_schema_version_unsupported|citrus_snapshot_schema"
    with pytest.raises(UnifiedH5ContractError, match=kind):
        admit_subject_snapshot(doc, project_attrs(doc))


def _registered() -> dict:
    return _load(FIXTURES / "accepted" / "individual_existing_registered.json")


@pytest.mark.parametrize("name", ["fish_reference_status", "fish_reference_reason"])
def test_v3_fish_reference_attrs_are_refused(name):
    doc = _registered()
    with pytest.raises(UnifiedH5ContractError, match=f"attr_removed_in_v4:{name}"):
        admit_subject_snapshot(doc, {**project_attrs(doc), name: "collected"})


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda a: a.update(mzb_registration_status="queued"), "attr_mismatch:mzb_registration_status"),
        (lambda a: a.pop("mzb_registration_mode"), "snapshot_value_without_attr:mzb_registration_mode"),
        (lambda a: a.update(mzb_registration_reason="x"), "attr_without_snapshot_value:mzb_registration_reason"),
        (lambda a: a.update(mzb_fish_id="0" * 8 + "-0000-4000-8000-" + "0" * 12), "attr_mismatch:mzb_fish_id"),
    ],
)
def test_v4_attrs_must_pair_with_the_snapshot(mutate, reason):
    doc = _registered()
    attrs = project_attrs(doc)
    mutate(attrs)
    with pytest.raises(UnifiedH5ContractError, match=reason):
        admit_subject_snapshot(doc, attrs)


def test_subject_id_must_equal_mzb_fish_id():
    doc = copy.deepcopy(_registered())
    doc["subject"]["subject_id"] = "0" * 8 + "-0000-4000-8000-" + "0" * 12
    with pytest.raises(UnifiedH5ContractError, match="R1_subject_id_differs_from_mzb_fish_id"):
        admit_subject_snapshot(doc, project_attrs(doc))


@pytest.mark.parametrize("path", ACCEPTED, ids=lambda p: p.stem)
def test_translator_carries_the_registration_record(path):
    doc = _load(path)
    fields = from_h5_attributes(project_attrs(doc))
    fish = doc["fish_reference"]
    if fish is None:
        assert "mzb_registration" not in fields
    else:
        assert fields["mzb_registration"] == {
            "mode": fish["mzb_registration_mode"],
            "status": fish["mzb_registration_status"],
            "reason": fish["mzb_registration_reason"] or "",
        }
        assert fields["mzb_fish_id"] == doc["subject"]["subject_id"]
        assert fields.get("mzb_fish_subject_label") == fish["mzb_fish_subject_label"]
    assert "fish_reference" not in fields
    assert "unresolved" not in fields


def test_translator_records_an_unknown_registration_as_unresolved():
    fields = from_h5_attributes({"mzb_registration_mode": "existing", "mzb_registration_status": "maybe"})
    assert "mzb_registration" not in fields
    assert "mzb_registration" in fields["unresolved"]


def _minted() -> dict:
    return _load(FIXTURES / "accepted" / "individual_created_at_start_refused.json")


def test_listed_fish_count_is_an_int64_attr():
    doc = _minted()
    attrs = project_attrs(doc)
    assert attrs["mzb_dish_listed_fish_count_at_start"] == 2
    assert admit_subject_snapshot(doc, attrs)["citrus_snapshot_status"] == "admitted"
    with pytest.raises(UnifiedH5ContractError, match="attr_not_int64:mzb_dish_listed_fish_count_at_start"):
        admit_subject_snapshot(doc, {**attrs, "mzb_dish_listed_fish_count_at_start": "2"})
    with pytest.raises(UnifiedH5ContractError, match="attr_mismatch:mzb_dish_listed_fish_count_at_start"):
        admit_subject_snapshot(doc, {**attrs, "mzb_dish_listed_fish_count_at_start": 3})


def test_unit_id_pairs_with_the_snapshot():
    doc = _registered()
    attrs = project_attrs(doc)
    assert attrs["mzb_fish_unit_id"] == "well-A1"
    with pytest.raises(UnifiedH5ContractError, match="attr_mismatch:mzb_fish_unit_id"):
        admit_subject_snapshot(doc, {**attrs, "mzb_fish_unit_id": "well-B2"})


def test_usage_link_names_the_bound_context_or_the_unbound_session():
    bound = _registered()
    ref = bound["usage_link_request"]["body"]["recording_ref"]
    check_usage_link_recording_ref(bound, observation_context_id=ref, session_uuid="s")
    with pytest.raises(UnifiedH5ContractError, match="usage_link_recording_ref:observation_context"):
        check_usage_link_recording_ref(bound, observation_context_id="obsctx_" + "0" * 64, session_uuid="s")
    # A bound file's link must name its context, not its session.
    with pytest.raises(UnifiedH5ContractError, match="usage_link_recording_ref"):
        check_usage_link_recording_ref(bound, observation_context_id=None, session_uuid=ref)
    unbound = _load(FIXTURES / "accepted" / "individual_existing_registered_unbound.json")
    session = unbound["usage_link_request"]["body"]["recording_ref"]
    check_usage_link_recording_ref(unbound, observation_context_id=None, session_uuid=session)
    with pytest.raises(UnifiedH5ContractError, match="usage_link_recording_ref:citrus_session"):
        check_usage_link_recording_ref(unbound, observation_context_id="obsctx_" + "3" * 64,
                                       session_uuid=session)
    with pytest.raises(UnifiedH5ContractError, match="usage_link_recording_ref:citrus_session"):
        check_usage_link_recording_ref(unbound, observation_context_id=None, session_uuid=None)
    no_dish = _load(FIXTURES / "accepted" / "no_dish.json")
    check_usage_link_recording_ref(no_dish, observation_context_id=None, session_uuid=None)
