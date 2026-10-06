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
    # Display copies are not paired.
    "fish_count": "32", "days_post_fertilization": "invalid_format", "cross_id": "",
}
PRE_CONTRACT = {
    "data_origin": "synthetic", "dish": {"cross_id": "c", "fish_count": 1, "species": "Danio rerio"},
    "dish_id": "synthetic-arena_1", "schema_version": 2,
}


def test_v3_is_admitted_with_the_serving_build():
    admitted = admit_subject_snapshot(V3, ATTRS)
    assert admitted["citrus_snapshot_status"] == "admitted"
    assert admitted["citrus_snapshot_schema_version"] == 3
    assert admitted["zebrobot_consumer_schema_matches_pin"] is True


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda a: a.update(subject_id="other"), "attr_mismatch:subject_id"),
        (lambda a: a.update(subject_count="2"), "attr_mismatch:subject_count"),
        (lambda a: a.update(dish_revision=2), "attr_mismatch:dish_revision"),
        (lambda a: a.pop("dish_uuid"), "snapshot_value_without_attr:dish_uuid"),
        (lambda a: a.update(mzb_fish_id="f1"), "attr_without_snapshot_value:mzb_fish_id"),
    ],
)
def test_attrs_must_pair_with_the_snapshot(mutate, reason):
    attrs = dict(ATTRS)
    mutate(attrs)
    with pytest.raises(UnifiedH5ContractError, match=reason):
        admit_subject_snapshot(V3, attrs)


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


@pytest.mark.parametrize("version", [1, 4, "3", None])
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
