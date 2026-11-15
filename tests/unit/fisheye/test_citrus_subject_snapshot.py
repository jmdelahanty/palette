"""Citrus subject identity v3: the sealed snapshot is authoritative over attrs."""

from __future__ import annotations

import copy

import h5py
import numpy as np
import pytest

from fisheye.shared.unified_h5 import UnifiedH5ContractError
from fisheye.shared.unified_h5.citrus_subject_snapshot import (
    admit_subject_snapshot,
    subject_attributes,
)
from fisheye.shared.unified_h5.metadata import native_attributes
from fisheye.shared.zebrobot_subject_reference import MZB_PIN

UUID = "28c29cc6-a1ef-4382-9dc5-414fbb445d92"
SUBJECT = "4bd7a2f6-92b7-42b8-ba28-6e49cb3cc0c9"
PIN = "f5280e430d4b5f10c3643cb89a6187eacc45fdac7af55b2e81f5e315b7b754dc"
V3 = {
    "schema_version": 3, "api_schema_version": 2, "queried_at_utc": "2026-10-05T19:30:00Z",
    "dish_id": "19220_1", "subject_lookup_status": "collected", "subject_lookup_reason": None,
    "dish": {
        "dish_id": "19220_1", "dish_uuid": UUID, "dish_revision": 1,
        "dish_updated_at": "2026-10-02 16:28:20", "cross_id": "19220", "genotype": "g",
        "dof": "20260928", "dpf": 7, "fish_count": 32, "species": "Danio rerio", "sex": "unknown",
    },
    "cross": None,
    "fish_reference": {
        "status": "not_collected", "reason": "operator_did_not_select",
        "mzb_fish_id": None, "mzb_fish_revision": None, "mzb_fish_updated_at": None,
    },
    "errors": [],
    "zebrobot_service": {
        "service_commit": "509a3eb88d6ff44fe07e7ea20d212be5eafe46b7",
        "service_commit_dirty": False, "consumer_schema_sha256": PIN,
    },
    "subject": {"subject_id": SUBJECT, "subject_type": "individual", "subject_count": 1},
}
ATTRS = {
    "subject_id": SUBJECT, "subject_type": "individual", "subject_count": "1",
    "dish_id": "19220_1", "subject_lookup_status": "collected", "dish_uuid": UUID,
    "dish_revision": 1, "dish_updated_at": "2026-10-02 16:28:20",
    "fish_reference_status": "not_collected", "fish_reference_reason": "operator_did_not_select",
    # Sealed biology, paired with the snapshot dish (B9).
    "genotype": "g", "species": "Danio rerio", "sex": "unknown", "cross_id": "19220",
    "date_of_fertilization": "20260928",
    # Display copies are not paired; line_strain is unsealed while cross is null.
    "fish_count": "32", "days_post_fertilization": "invalid_format",
    "parents": "p [F]", "line_strain": "unsealed",
}
PRE_CONTRACT = {
    "data_origin": "synthetic", "dish": {"cross_id": "c", "fish_count": 1, "species": "Danio rerio"},
    "dish_id": "synthetic-arena_1", "schema_version": 2,
}


def test_v3_is_admitted_with_the_serving_build():
    admitted = admit_subject_snapshot(V3, ATTRS)
    assert admitted["citrus_snapshot_status"] == "admitted"
    assert admitted["citrus_snapshot_schema_version"] == 3
    assert admitted["zebrobot_consumer_schema_sha256"] == PIN
    # 509a3eb8's slice is recorded, but no longer matches: the fish list changed meaning.
    assert admitted["zebrobot_consumer_schema_matches_pin"] is False
    current = copy.deepcopy(V3)
    current["zebrobot_service"]["consumer_schema_sha256"] = MZB_PIN["consumer_openapi_sha256"]
    assert admit_subject_snapshot(current, ATTRS)["zebrobot_consumer_schema_matches_pin"] is True


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda a: a.update(subject_id="other"), "attr_mismatch:subject_id"),
        (lambda a: a.update(subject_count="2"), "attr_mismatch:subject_count"),
        (lambda a: a.update(dish_revision=2), "attr_mismatch:dish_revision"),
        (lambda a: a.pop("dish_uuid"), "snapshot_value_without_attr:dish_uuid"),
        (lambda a: a.update(mzb_fish_id="f1"), "attr_without_snapshot_value:mzb_fish_id"),
        (lambda a: a.update(genotype="other"), "attr_mismatch:genotype"),
        (lambda a: a.update(species="Danionella cerebrum"), "attr_mismatch:species"),
        (lambda a: a.update(sex="F"), "attr_mismatch:sex"),
        (lambda a: a.update(cross_id=""), "attr_mismatch:cross_id"),
        (lambda a: a.update(date_of_fertilization="20260929"), "attr_mismatch:date_of_fertilization"),
        (lambda a: a.pop("genotype"), "snapshot_value_without_attr:genotype"),
        (lambda a: a.pop("species"), "snapshot_value_without_attr:species"),
        (lambda a: a.pop("sex"), "snapshot_value_without_attr:sex"),
    ],
)
def test_attrs_must_pair_with_the_snapshot(mutate, reason):
    attrs = dict(ATTRS)
    mutate(attrs)
    with pytest.raises(UnifiedH5ContractError, match=reason):
        admit_subject_snapshot(V3, attrs)


CROSS = {"cross_id": "19220", "line_strain": "AB", "parents": [{"identifier": "p", "sex": "F"}]}


def test_sealed_cross_pairs_line_strain():
    snapshot = copy.deepcopy(V3)
    snapshot["cross"] = CROSS
    attrs = {**ATTRS, "line_strain": "AB"}
    assert admit_subject_snapshot(snapshot, attrs)["citrus_snapshot_status"] == "admitted"
    with pytest.raises(UnifiedH5ContractError, match="attr_mismatch:line_strain"):
        admit_subject_snapshot(snapshot, {**attrs, "line_strain": "other"})
    without = dict(attrs)
    without.pop("line_strain")
    with pytest.raises(UnifiedH5ContractError, match="snapshot_value_without_attr:line_strain"):
        admit_subject_snapshot(snapshot, without)


def test_no_dish_means_no_biology_attrs():
    snapshot = copy.deepcopy(V3)
    snapshot.update(
        subject_lookup_status="lookup_failed", subject_lookup_reason="dish_lookup_http_404",
        dish=None, errors=[{"endpoint": "/dishes/19220_1/citrus-snapshot", "kind": "http",
                             "http_status": 404, "detail_error": None, "message": "not found"}],
    )
    attrs = {k: v for k, v in ATTRS.items() if k not in {
        "dish_uuid", "dish_revision", "dish_updated_at", "genotype", "species", "sex",
        "cross_id", "date_of_fertilization", "line_strain"}}
    attrs["subject_lookup_reason"] = "dish_lookup_http_404"
    attrs["subject_lookup_status"] = "lookup_failed"
    assert admit_subject_snapshot(snapshot, attrs)["citrus_snapshot_status"] == "admitted"
    with pytest.raises(UnifiedH5ContractError, match="attr_without_snapshot_value:genotype"):
        admit_subject_snapshot(snapshot, {**attrs, "genotype": "g"})


def test_no_dish_means_no_subject_attrs():
    snapshot = copy.deepcopy(V3)
    snapshot.update(
        dish_id=None, subject_lookup_status="not_collected", subject_lookup_reason="no_dish_declared",
        dish=None, subject={"subject_id": None, "subject_type": None, "subject_count": None},
    )
    snapshot["fish_reference"]["reason"] = "no_dish_declared"
    attrs = {"subject_lookup_status": "not_collected", "subject_lookup_reason": "no_dish_declared",
             "fish_reference_status": "not_collected", "fish_reference_reason": "no_dish_declared"}
    assert admit_subject_snapshot(snapshot, attrs)["citrus_snapshot_status"] == "admitted"
    with pytest.raises(UnifiedH5ContractError, match="attr_without_snapshot_value:subject_id"):
        admit_subject_snapshot(snapshot, {**attrs, "subject_id": SUBJECT})


def test_drifted_build_is_recorded_not_refused():
    snapshot = copy.deepcopy(V3)
    snapshot["zebrobot_service"]["consumer_schema_sha256"] = "0" * 64
    assert admit_subject_snapshot(snapshot, ATTRS)["zebrobot_consumer_schema_matches_pin"] is False


def test_pre_contract_v2_is_recorded_without_identity_rules():
    assert admit_subject_snapshot(PRE_CONTRACT, {"subject_id": "x"}) == {
        "citrus_snapshot_status": "pre_contract", "citrus_snapshot_schema_version": 2,
    }


def test_never_emitted_contract_v2_is_refused_not_downgraded():
    contract_v2 = copy.deepcopy(V3)
    contract_v2["schema_version"] = 2
    del contract_v2["zebrobot_service"], contract_v2["subject"]
    with pytest.raises(UnifiedH5ContractError, match="contract_v2_never_emitted"):
        admit_subject_snapshot(contract_v2, ATTRS)


@pytest.mark.parametrize("version", [1, 5, "3", "4", None])
def test_other_versions_are_refused(version):
    with pytest.raises(UnifiedH5ContractError, match="schema_version_unsupported"):
        admit_subject_snapshot({**V3, "schema_version": version}, ATTRS)


def test_v2_and_v3_refuse_each_others_documents():
    with pytest.raises(UnifiedH5ContractError, match="citrus_snapshot_schema"):
        admit_subject_snapshot({k: v for k, v in V3.items() if k != "subject"}, ATTRS)


def test_int64_revisions_decode_and_other_integers_refuse(tmp_path):
    path = tmp_path / "attrs.h5"
    with h5py.File(path, "w") as h5:
        node = h5.create_group("subject")
        node.attrs["dish_revision"] = np.int64(3)
        node.attrs["dish_uuid"] = UUID
        node.attrs.create("subject_count", 1, dtype=np.int64)
    with h5py.File(path, "r") as h5:
        descriptors = native_attributes(h5["subject"])
    with pytest.raises(UnifiedH5ContractError, match="attribute_not_scalar_string:subject_count"):
        subject_attributes(descriptors)
    descriptors.pop("subject_count")
    assert subject_attributes(descriptors) == {"dish_revision": 3, "dish_uuid": UUID}


# --- 2026-10-14 amendment (agent-contracts PR 110, Citrus PR 43) -------------

UNIT = "19220_1_WP1:A1"


def _picked(unit=UNIT):
    snapshot = copy.deepcopy(V3)
    snapshot["fish_reference"] = {
        "status": "collected", "reason": None, "mzb_fish_id": "f1", "mzb_fish_revision": 3,
        "mzb_fish_updated_at": "2026-10-14 09:00:00", "mzb_fish_unit_id": unit,
    }
    attrs = dict(ATTRS, fish_reference_status="collected", mzb_fish_id="f1",
                 mzb_fish_revision=3, mzb_fish_updated_at="2026-10-14 09:00:00")
    attrs.pop("fish_reference_reason")
    if unit is not None:
        attrs["mzb_fish_unit_id"] = unit
    return snapshot, attrs


@pytest.mark.parametrize(
    "reason",
    ["operator_did_not_select", "operator_declined_pick", "no_registered_fish", "fish_list_unavailable"],
)
def test_every_not_collected_reason_is_admitted(reason):
    snapshot = copy.deepcopy(V3)
    snapshot["fish_reference"].update(reason=reason, mzb_fish_unit_id=None)
    if reason == "fish_list_unavailable":
        snapshot["errors"] = [{"endpoint": "/dishes/19220_1/fish", "kind": "http", "http_status": 503, "detail_error": "database_error", "message": "HTTP 503"}]
    attrs = dict(ATTRS, fish_reference_reason=reason)
    assert admit_subject_snapshot(snapshot, attrs)["citrus_snapshot_status"] == "admitted"


def test_a_file_from_before_the_amendment_has_no_unit_key_and_is_admitted():
    assert "mzb_fish_unit_id" not in V3["fish_reference"]
    assert admit_subject_snapshot(V3, ATTRS)["citrus_snapshot_status"] == "admitted"


def test_a_picked_fish_pairs_its_unit_with_the_attr():
    snapshot, attrs = _picked()
    assert admit_subject_snapshot(snapshot, attrs)["citrus_snapshot_status"] == "admitted"


def test_a_picked_fish_without_a_served_unit_has_no_attr():
    snapshot, attrs = _picked(unit=None)
    assert admit_subject_snapshot(snapshot, attrs)["citrus_snapshot_status"] == "admitted"
    attrs["mzb_fish_unit_id"] = UNIT
    with pytest.raises(UnifiedH5ContractError, match="attr_without_snapshot_value:mzb_fish_unit_id"):
        admit_subject_snapshot(snapshot, attrs)


def test_the_unit_attr_must_equal_the_sealed_unit():
    snapshot, attrs = _picked()
    attrs["mzb_fish_unit_id"] = "19220_1_WP1:A2"
    with pytest.raises(UnifiedH5ContractError, match="attr_mismatch:mzb_fish_unit_id"):
        admit_subject_snapshot(snapshot, attrs)
    attrs.pop("mzb_fish_unit_id")
    with pytest.raises(UnifiedH5ContractError, match="snapshot_value_without_attr:mzb_fish_unit_id"):
        admit_subject_snapshot(snapshot, attrs)


def test_an_unsealed_unit_attr_is_refused_on_a_file_without_the_key():
    with pytest.raises(UnifiedH5ContractError, match="attr_without_snapshot_value:mzb_fish_unit_id"):
        admit_subject_snapshot(V3, dict(ATTRS, mzb_fish_unit_id=UNIT))


@pytest.mark.parametrize(
    "fish_reference",
    [
        # A unit only for a collected fish.
        {"status": "not_collected", "reason": "operator_declined_pick", "mzb_fish_id": None,
         "mzb_fish_revision": None, "mzb_fish_updated_at": None, "mzb_fish_unit_id": UNIT},
        {"status": "lookup_failed", "reason": "fish_lookup_invalid_body", "mzb_fish_id": "f1",
         "mzb_fish_revision": None, "mzb_fish_updated_at": None, "mzb_fish_unit_id": UNIT},
        # Never a number or another shape.
        {"status": "collected", "reason": None, "mzb_fish_id": "f1", "mzb_fish_revision": 3,
         "mzb_fish_updated_at": "2026-10-14 09:00:00", "mzb_fish_unit_id": 7},
        # The new reasons are not_collected reasons only.
        {"status": "lookup_failed", "reason": "fish_list_unavailable", "mzb_fish_id": "f1",
         "mzb_fish_revision": None, "mzb_fish_updated_at": None},
        {"status": "not_collected", "reason": "fish_pick_required", "mzb_fish_id": None,
         "mzb_fish_revision": None, "mzb_fish_updated_at": None},
    ],
)
def test_the_schema_ties_the_unit_and_reasons_to_the_status(fish_reference):
    snapshot = copy.deepcopy(V3)
    snapshot["fish_reference"] = fish_reference
    with pytest.raises(UnifiedH5ContractError, match="citrus_snapshot_schema"):
        admit_subject_snapshot(snapshot, ATTRS)


@pytest.mark.parametrize(
    "lookup, endpoint, admitted",
    [
        ("collected", "/dishes/19220_1/fish", True),
        ("collected", "/dishes/19220_2/fish", False),
        ("lookup_failed", "/dishes/19220_1/citrus-snapshot", True),
        ("lookup_failed", "/dishes/19220_2/citrus-snapshot", False),
    ],
)
def test_fish_list_unavailable_names_the_snapshots_own_dish(lookup, endpoint, admitted):
    """The schema checks the endpoint's shape; Palette checks its {dish_id}."""

    snapshot = copy.deepcopy(V3)
    snapshot["fish_reference"].update(reason="fish_list_unavailable", mzb_fish_unit_id=None)
    snapshot["errors"] = [{"endpoint": endpoint, "kind": "http", "http_status": 503, "detail_error": None, "message": "HTTP 503"}]
    attrs = dict(ATTRS, fish_reference_reason="fish_list_unavailable")
    if lookup != "collected":
        snapshot.update(subject_lookup_status=lookup, subject_lookup_reason="dish_lookup_http_404", dish=None)
        attrs = {k: v for k, v in attrs.items() if k not in {
            "dish_uuid", "dish_revision", "dish_updated_at", "genotype", "species", "sex",
            "cross_id", "date_of_fertilization"}}
        attrs.update(subject_lookup_status=lookup, subject_lookup_reason="dish_lookup_http_404")
    if admitted:
        assert admit_subject_snapshot(snapshot, attrs)["citrus_snapshot_status"] == "admitted"
    else:
        with pytest.raises(UnifiedH5ContractError, match="fish_list_unavailable_error_not_for_dish"):
            admit_subject_snapshot(snapshot, attrs)
