"""Canonical subject fields: every source translates to the same names and types."""

from __future__ import annotations

import copy

import pytest
import zarr

from fisheye.shared import subject_fields as sf
from fisheye.shared import zebrobot_subject_reference as zsr
from fisheye.shared.subject_metadata import publish_subject_metadata, resolve_subject_metadata

UUID = "28c29cc6-a1ef-4382-9dc5-414fbb445d92"
SUBJECT = "4bd7a2f6-92b7-42b8-ba28-6e49cb3cc0c9"

# One fish, one dish, one recording day (fertilized 2026-08-04, recorded 2026-08-12).
LEGACY_H5 = {  # legacy /subject_metadata, as production Citrus wrote it (string attrs)
    "fish_id": SUBJECT, "subject_type": "individual", "subject_count": "1",
    "dish_id": "18867_10", "cross_id": "18867", "source_dish_id": "18867_10",
    "source_cross_id": "18867", "fish_count": "10", "source_dish_population_count": "10",
    "genotype": "AB [AB IC] SEPT25", "line_strain": "AB [AB IC] SEPT25",
    "species": "Danio rerio", "sex": "unknown", "parents": "8130_M13_A2 [F]; 6728_M19_E5 [M]",
    "date_of_fertilization": "20260804", "days_post_fertilization": "8",
    "queried_at_utc": "2026-08-12T21:59:55Z",
}
CITRUS_V3 = {  # unified /metadata/subject (Citrus subject identity v3)
    **{k: v for k, v in LEGACY_H5.items() if k != "fish_id"},
    "subject_id": SUBJECT, "dish_uuid": UUID, "dish_revision": 3,
    "dish_updated_at": "2026-08-10 09:00:00",
    "subject_lookup_status": "collected", "fish_reference_status": "not_collected",
    "fish_reference_reason": "operator_did_not_select",
}
SERVED = {  # MetaZebrobot citrus-snapshot for the same dish
    "schema_version": 2, "dish_id": "18867_10", "dish_uuid": UUID, "revision": 3,
    "updated_at": "2026-08-10 09:00:00", "cross_id": "18867", "genotype": "AB [AB IC] SEPT25",
    "dof": "20260804", "dpf": 63, "fish_count": 10, "species": "Danio rerio", "sex": "unknown",
    "line_strain": "AB [AB IC] SEPT25",
    "parents": [{"identifier": "8130_M13_A2", "sex": "F"}, {"identifier": "6728_M19_E5", "sex": "M"}],
}
ORANGE_REFERENCE = {
    "schema_id": "orange.recording_subject_reference", "schema_version": 1,
    "status": "collected", "reason": "",
    "zebrobot": {
        "base_url": "http://zebrobot", "endpoint": "/dishes/18867_10/citrus-snapshot",
        "fish_endpoint": "/dishes/18867_10/fish", "api_schema_version": 2,
        "queried_at_utc": "2026-08-12T21:59:55Z", "http_status": 200, "error": None,
    },
    "dish": {"dish_id": "18867_10", "dish_uuid": UUID, "revision": 3,
             "updated_at": "2026-08-10 09:00:00"},
    "dish_fish": [], "dish_fish_lookup": {"status": "complete", "http_status": 200, "error": None},
    "subject_count": 1,
}


def _orange_metadata():
    def fetch(url):
        return (200, copy.deepcopy(SERVED)) if url.endswith("/citrus-snapshot") else (200, {"items": []})

    return zsr.resolve_subject_reference(ORANGE_REFERENCE, camera="CAM-1", fetch=fetch).metadata


def _published(tmp_path, name, metadata):
    root = zarr.open_group(str(tmp_path / f"{name}.zarr"), mode="w")
    publish_subject_metadata(root, metadata, source_artifact={"kind": name})
    return resolve_subject_metadata(root, allow_legacy=False).subject


# Facts every source can state about this fish.
SHARED = ("species", "sex", "genotype", "line_strain", "cross_id", "parents", "dish_id",
          "date_of_fertilization", "dpf_at_acquisition", "subject_count", "subject_type",
          "source_dish_population_count")
# Facts only the MetaZebrobot-aware sources (Citrus v3, Orange) carry.
DISH_IDENTITY = ("dish_uuid", "dish_revision", "dish_updated_at", "subject_lookup")


def test_one_fish_reads_the_same_from_every_source(tmp_path):
    legacy = _published(tmp_path, "legacy", LEGACY_H5)
    citrus = _published(tmp_path, "citrus", CITRUS_V3)
    orange = _published(tmp_path, "orange", _orange_metadata())

    for name in SHARED:
        assert legacy[name] == citrus[name] == orange[name], name
    for name in DISH_IDENTITY:
        assert citrus[name] == orange[name], name
    assert legacy["dpf_at_acquisition"] == 8  # recording date, never the server's same-day dpf (63)
    assert legacy["parents"] == [{"identifier": "8130_M13_A2", "sex": "F"},
                                 {"identifier": "6728_M19_E5", "sex": "M"}]
    assert legacy["subject_ids"] == citrus["subject_ids"] == [SUBJECT]
    assert "subject_ids" not in orange  # Orange never names a Citrus subject
    assert orange["intake_observations"]["dish_changed_since_recording"] is False
    for subject in (legacy, citrus, orange):
        assert "unresolved" not in subject


@pytest.mark.parametrize(
    "value, expected",
    [
        ("8750_M08_B2 [unknown]", [{"identifier": "8750_M08_B2", "sex": "unknown"}]),
        ("6203_M10_E8 [M]; 4861_M22_C9 [unknown]",
         [{"identifier": "6203_M10_E8", "sex": "M"}, {"identifier": "4861_M22_C9", "sex": "unknown"}]),
        ('[{"identifier": "A", "sex": "F"}]', [{"identifier": "A", "sex": "F"}]),
        (["B"], [{"identifier": "B", "sex": None}]),
        ("", None),
        (None, None),
    ],
)
def test_parents_parse_to_one_shape(value, expected):
    assert sf.parse_parents(value) == expected


def test_unparseable_values_are_left_out_with_a_reason():
    subject = sf.from_h5_attributes(
        {"days_post_fertilization": "invalid_format", "date_of_fertilization": "2026-08-04",
         "subject_count": "one", "subject_type": "synthetic_fish"}
    )
    assert "dpf_at_acquisition" not in subject and "subject_count" not in subject
    assert set(subject["unresolved"]) == {
        "dpf_at_acquisition", "date_of_fertilization", "subject_count", "subject_type"}


def test_manual_group_is_a_dish_group_and_keeps_declared_age():
    subject = sf.canonical_subject_fields(
        {"species": "Danio rerio", "subject_count": 3, "dpf_at_acquisition": 7,
         "days_post_fertilization": 7, "subject_type": "group", "identity_scope": "count_only_no_subject_ids",
         "source": "recording_manifest_manual_assertion", "status": "user_asserted"}
    )
    assert sf.source_kind({"source": "recording_manifest_manual_assertion"}) == "manual_assertion"
    assert subject["subject_type"] == "dish_group"
    assert subject["dpf_at_acquisition"] == 7 and subject["subject_count"] == 3


def test_subject_type_follows_the_count_when_undeclared():
    assert sf.from_orange_reference({"subject_count": 4})["subject_type"] == "dish_group"
    assert "subject_type" not in sf.from_orange_reference({})


def test_unknown_lookup_status_is_unresolved():
    subject = sf.from_h5_attributes({"subject_lookup_status": "maybe"})
    assert "subject_lookup" not in subject
    assert subject["unresolved"]["subject_lookup"] == "unknown status: 'maybe'"


def test_v2_records_keep_the_fields_fixed_at_publish(tmp_path, monkeypatch):
    root = zarr.open_group(str(tmp_path / "v2.zarr"), mode="w")
    published = publish_subject_metadata(root, CITRUS_V3, source_artifact={"kind": "test"},
                                         translator="h5_attributes")
    assert published.record["schema_id"] == "palette.subject_metadata.v2"
    assert published.record["subject_translator"] == "h5_attributes"
    stored = dict(published.subject)
    # A later translator change must not alter (or invalidate) a stored record.
    monkeypatch.setitem(sf.TRANSLATORS, "h5_attributes", lambda *_args: {"changed": True})
    assert resolve_subject_metadata(root, allow_legacy=False).subject == stored


def test_v1_records_still_translate_on_read(tmp_path):
    root = zarr.open_group(str(tmp_path / "v1.zarr"), mode="w")
    published = publish_subject_metadata(root, LEGACY_H5, source_artifact={"kind": "test"})
    assert published.record["schema_id"] == "palette.subject_metadata.v1"
    assert published.subject["dpf_at_acquisition"] == 8


def test_unknown_translator_is_refused(tmp_path):
    root = zarr.open_group(str(tmp_path / "x.zarr"), mode="w")
    with pytest.raises(ValueError, match="Unknown subject translator"):
        publish_subject_metadata(root, LEGACY_H5, source_artifact={"kind": "test"}, translator="guess")
